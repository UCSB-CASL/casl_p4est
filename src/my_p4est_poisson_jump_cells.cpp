#ifdef P4_TO_P8
#include "my_p8est_poisson_jump_cells.h"
#else
#include "my_p4est_poisson_jump_cells.h"
#endif

my_p4est_poisson_jump_cells_t::my_p4est_poisson_jump_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t* nodes_)
  : cell_ngbd(ngbd_c), p4est(ngbd_c->get_p4est()), ghost(ngbd_c->get_ghost()), nodes(nodes_),
    xyz_min(ngbd_c->get_p4est()->connectivity->vertices + 3*ngbd_c->get_p4est()->connectivity->tree_to_vertex[0]),
    xyz_max(ngbd_c->get_p4est()->connectivity->vertices + 3*ngbd_c->get_p4est()->connectivity->tree_to_vertex[P4EST_CHILDREN*(ngbd_c->get_p4est()->trees->elem_count - 1) + P4EST_CHILDREN - 1]),
    tree_dimensions(ngbd_c->get_tree_dimensions()),
    periodicity(ngbd_c->get_hierarchy()->get_periodicity()), interface_manager(NULL)
{
  // set up the KSP solver
  PetscErrorCode ierr;
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  mu_minus = mu_plus = -1.0;
  add_diag_minus = add_diag_plus = 0.0;
  user_rhs_minus = user_rhs_plus = NULL;
  face_velocity_minus = face_velocity_plus = NULL;
  interp_jump_normal_velocity = NULL;
  jump_u = jump_normal_flux_u = NULL;
  user_initial_guess = NULL;
  solution = rhs = NULL;
  interp_jump_u           = NULL;
  interp_jump_normal_flux = NULL;

  A = NULL;
  A_null_space = NULL;
  bc = NULL;
  matrix_is_set = rhs_is_set = false;

  const splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;

  // Domain and grid parameters
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    dxyz_min[dim] = tree_dimensions[dim]/(double) (1 << data->max_lvl);
}

my_p4est_poisson_jump_cells_t::~my_p4est_poisson_jump_cells_t()
{
  PetscErrorCode ierr;
  if (A                       != NULL)  { ierr = MatDestroy(A);                       CHKERRXX(ierr); }
  if (A_null_space            != NULL)  { ierr = MatNullSpaceDestroy (A_null_space);  CHKERRXX(ierr); }
  if (ksp                     != NULL)  { ierr = KSPDestroy(ksp);                     CHKERRXX(ierr); }
  if (solution                != NULL)  { ierr = VecDestroy(solution);                CHKERRXX(ierr); }
  if (rhs                     != NULL)  { ierr = VecDestroy(rhs);                     CHKERRXX(ierr); }
  if (interp_jump_u           != NULL)  { delete interp_jump_u;                                       }
  if (interp_jump_normal_flux != NULL)  { delete interp_jump_normal_flux;                             }
}

void my_p4est_poisson_jump_cells_t::preallocate_matrix()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(!matrix_is_set);

  if (A != NULL){
    ierr = MatDestroy(A); CHKERRXX(ierr); }

  const PetscInt num_owned_global = p4est->global_num_quadrants;
  const PetscInt num_owned_local  = p4est->local_num_quadrants;

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> nlocal_nonzeros(num_owned_local), nghost_nonzeros(num_owned_local);

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; q++)
    {
      const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      get_numbers_of_cells_involved_in_equation_for_quad(quad_idx, tree_idx, nlocal_nonzeros[quad_idx], nghost_nonzeros[quad_idx]);
    }
  }

  ierr = MatSeqAIJSetPreallocation(A, 0, nlocal_nonzeros.data()); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, nlocal_nonzeros.data(), 0, nghost_nonzeros.data()); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_cells_t::set_interface(my_p4est_interface_manager_t* interface_manager_)
{
  P4EST_ASSERT(interface_manager_ != NULL);
  interface_manager = interface_manager_;

  if(!interface_manager->is_grad_phi_set())
    interface_manager->set_grad_phi();

  matrix_is_set = rhs_is_set = false;
  return;
}

void my_p4est_poisson_jump_cells_t::set_jumps(Vec jump_u_, Vec jump_normal_flux_u_)
{
  if(!interface_is_set())
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::set_jumps(): the interface manager must be set before the jumps");
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
#ifdef P4EST_DEBUG
  P4EST_ASSERT(jump_u_              == NULL || VecIsSetForNodes(jump_u_,              interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, 1));
  P4EST_ASSERT(jump_normal_flux_u_  == NULL || VecIsSetForNodes(jump_normal_flux_u_,  interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, 1));
#endif

  jump_u              = jump_u_;
  jump_normal_flux_u  = jump_normal_flux_u_;

  if(interp_jump_u != NULL && jump_u == NULL){
    delete interp_jump_u;
    interp_jump_u = NULL;
  }
  if(interp_jump_u == NULL && jump_u != NULL)
    interp_jump_u = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
  if(jump_u != NULL)
    interp_jump_u->set_input(jump_u, linear);

  if(interp_jump_normal_flux != NULL && jump_normal_flux_u == NULL){
    delete interp_jump_normal_flux;
    interp_jump_normal_flux = NULL;
  }
  if(interp_jump_normal_flux == NULL && jump_normal_flux_u != NULL)
    interp_jump_normal_flux = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);

  if(jump_normal_flux_u != NULL)
    interp_jump_normal_flux->set_input(jump_normal_flux_u, linear);

  rhs_is_set = false;
  return;
}

linear_combination_of_dof_t
my_p4est_poisson_jump_cells_t::stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& oriented_dir,
                                                                             set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side,
                                                                             linear_combination_of_dof_t *vstar_on_face_for_stable_projection) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  P4EST_ASSERT(!is_quad_Wall(p4est, tree_idx, quad, oriented_dir)); // we do not allow for wall faces in here
  double xyz_[P4EST_DIM];

  direct_neighbors.clear();
  cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, oriented_dir);
  P4EST_ASSERT(direct_neighbors.size() > 0);

  const bool quad_is_major            = (direct_neighbors.size() > 1 || (direct_neighbors.size() == 1 && direct_neighbors.begin()->level == quad->level));
  const bool major_quad_is_leading    = (oriented_dir%2 == (quad_is_major ? 0 : 1));
  const p4est_quadrant_t& major_quad  = (quad_is_major ? *quad : *direct_neighbors.begin());
  quad_xyz_fr_q((quad_is_major ? quad_idx : major_quad.p.piggy3.local_num), (quad_is_major ? tree_idx : major_quad.p.piggy3.which_tree), p4est, ghost, xyz_);
  const char sgn_major_quad = (interface_manager->phi_at_point(xyz_) <= 0.0 ? -1 : 1);
  all_cell_centers_on_same_side = true;

  set_of_neighboring_quadrants *minor_quads = NULL;
  if(quad_is_major)
    minor_quads = &direct_neighbors;
  else
  {
    minor_quads = new set_of_neighboring_quadrants;
    cell_ngbd->find_neighbor_cells_of_cell(*minor_quads, major_quad.p.piggy3.local_num, major_quad.p.piggy3.which_tree, oriented_dir + (oriented_dir%2 == 0 ? 1 : -1)); // find all other quads sharing the face
  }

  const double shared_surface = pow((double) P4EST_QUADRANT_LEN(major_quad.level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1); // (logical) shared surface
  double discretization_distance = 0.0;
#ifdef DEBUG
  double split_face_check = 0.0; bool quad_is_among_sharers = quad_is_major;
#endif
  for (set_of_neighboring_quadrants::const_iterator it = minor_quads->begin(); it != minor_quads->end(); ++it)
  {
    const double surface_ratio = pow((double) P4EST_QUADRANT_LEN(it->level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1)/shared_surface;
    discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(major_quad.level) + P4EST_QUADRANT_LEN(it->level))/(double) P4EST_ROOT_LEN;
#ifdef DEBUG
    split_face_check += surface_ratio; quad_is_among_sharers = quad_is_among_sharers || it->p.piggy3.local_num == quad_idx;
#endif
    quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_);
    const char sgn_minor_quad = (interface_manager->phi_at_point(xyz_) <= 0.0 ? -1 : 1);
    all_cell_centers_on_same_side = all_cell_centers_on_same_side && !signs_of_phi_are_different(sgn_major_quad, sgn_minor_quad);
  }
  P4EST_ASSERT(quad_is_among_sharers && fabs(split_face_check - 1.0) < EPS);
  discretization_distance *= tree_dimensions[oriented_dir/2];

  linear_combination_of_dof_t local_derivative_operator;
  if(vstar_on_face_for_stable_projection != NULL)
    vstar_on_face_for_stable_projection->clear();
  for (set_of_neighboring_quadrants::const_iterator it = minor_quads->begin(); it != minor_quads->end(); ++it)
  {
    const double surface_ratio = pow((double) P4EST_QUADRANT_LEN(it->level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1)/shared_surface;
    local_derivative_operator.add_term(it->p.piggy3.local_num, (major_quad_is_leading ? -1.0 : +1.0)*surface_ratio/discretization_distance);
    if(vstar_on_face_for_stable_projection != NULL)
      vstar_on_face_for_stable_projection->add_term(interface_manager->get_faces()->q2f(it->p.piggy3.local_num, oriented_dir + (quad_is_major ? (oriented_dir%2 == 0 ? 1 : -1) : 0)),
                                                    pow(2.0, (double)(major_quad.level - it->level)*(P4EST_DIM - 1)));
  }
  local_derivative_operator.add_term((quad_is_major ? quad_idx : major_quad.p.piggy3.local_num), (major_quad_is_leading ? +1.0 : -1.0)/discretization_distance);

  if(minor_quads != &direct_neighbors)
    delete minor_quads;

  return local_derivative_operator;
}

void my_p4est_poisson_jump_cells_t::solve_linear_system()
{
  PetscErrorCode ierr;

  if (solution == NULL) {
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr);
    if(user_initial_guess != NULL){
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
      ierr = VecCopyGhost(user_initial_guess, solution); CHKERRXX(ierr);
    }
  }

  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  // we need update ghost values of the solution for accurate calculation of the extended interface values
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

#ifdef CASL_THROWS
  KSPConvergedReason termination_reason;
  ierr = KSPGetConvergedReason(ksp, &termination_reason); CHKERRXX(ierr);
  if(termination_reason <= 0)
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::solve_linear_system() : the Krylov solver failed to converge for a linear system to solve, the KSPConvergedReason code is " + std::to_string(termination_reason)); // collective runtime_error throw
#endif
  return;
}

PetscErrorCode my_p4est_poisson_jump_cells_t::setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type, const double &tolerance_on_rel_residual) const
{
  PetscErrorCode ierr;
  ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER); CHKERRQ(ierr);
  // set ksp type
  ierr = KSPSetType(ksp, (A_null_space != NULL ? KSPGMRES : ksp_type)); CHKERRQ(ierr); // PetSc advises GMRES for problems with nullspaces
  ierr = KSPSetTolerances(ksp, tolerance_on_rel_residual, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRQ(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
   * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
   * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
   */
  if (!strcmp(pc_type, PCHYPRE)){
    // a large value for this factor was found to be more robust for the jump problems (especially with Daniil's FV solver)
    // should be in )0, 1(, HYPRE manual suggests 0.25 for 2D problems, 0.5 for 3D,
    // (arbitrarily) set to 0.7 because it seemed to work for all test cases
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.7"); CHKERRQ(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRQ(ierr);

    /* 3- Truncation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRQ(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (A_null_space != NULL){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRQ(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  return ierr;
}

void my_p4est_poisson_jump_cells_t::setup_linear_system()
{
  if(matrix_is_set && rhs_is_set)
    return;

  if(!matrix_is_set)
    preallocate_matrix();

  PetscErrorCode ierr;
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(interface_is_set()); // otherwise, well, it's going to be hard...

  if(!rhs_is_set)
  {
    if(rhs == NULL){
      ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr); }
    P4EST_ASSERT(VecIsSetForCells(rhs, p4est, ghost, 1, false));
  }
  int nullspace_contains_constant_vector = !matrix_is_set; // converted to integer because of required MPI collective determination thereafter + we don't care about that if the matrix is already set

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      build_discretization_for_quad(q + tree->quadrants_offset, tree_idx, &nullspace_contains_constant_vector);
  }

  if(!matrix_is_set)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

    // check for null space
    if(A_null_space != NULL){
      ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);
      A_null_space = NULL;
    }
    ierr = MPI_Allreduce(MPI_IN_PLACE, &nullspace_contains_constant_vector, 1, MPI_INT, MPI_LAND, p4est->mpicomm); CHKERRXX(ierr);
    if (nullspace_contains_constant_vector)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, NULL, &A_null_space); CHKERRXX(ierr);
      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
    }
  }
  /* [Raphael (05/17/2020) :
   * removing the null space from the rhs seems redundant with PetSc operations done under the
   * hood in KSPSolve (see the source code of KSPSolve_Private in /src/kscp/ksp/interface/itfunc.c
   * for more details). --> So long as MatSetNullSpace() was called appropriately on the matrix
   * operator, this operation will be executed in the pre-steps of KSPSolve on a COPY of the provided
   * RHS vector]
   * --> this is what we want : we need the _unmodified_ RHS thereafter, i.e. as we have built it, and
   * we let PetSc do its magic under the hood every time we call KSPSolve, otherwise iteratively
   * correcting and updating the RHS would becomes very complex and require extra info coming from those
   * possibly non-empty nullspace contributions...
   * */

  matrix_is_set = rhs_is_set = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_cells_t::project_face_velocities(const my_p4est_faces_t *faces, Vec *flux_minus, Vec *flux_plus) const
{
  P4EST_ASSERT(faces->get_p4est() == p4est); // the faces must be built from the same computational grid
  P4EST_ASSERT((face_velocity_minus == NULL && face_velocity_plus == NULL) ||
               (VecsAreSetForFaces(face_velocity_minus, faces, 1) && VecsAreSetForFaces(face_velocity_plus, faces, 1))); // the face-sampled velocities vstart and vnp1 vectors vectors must either be all defined or be all NULL.
  P4EST_ASSERT(flux_minus == NULL || VecsAreSetForFaces(flux_minus, faces, 1));
  P4EST_ASSERT(flux_plus == NULL || VecsAreSetForFaces(flux_plus, faces, 1));

#ifdef CASL_THROWS
  if(solution == NULL)
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::project_face_velocities(): requires the solution, have you called called solve() before?");
#endif

  const bool velocities_provided            = (face_velocity_minus != NULL && face_velocity_plus != NULL);
  double *flux_minus_p[P4EST_DIM]           = {DIM(NULL, NULL, NULL)};
  double *flux_plus_p[P4EST_DIM]            = {DIM(NULL, NULL, NULL)};
  double *face_velocity_minus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *face_velocity_plus_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  PetscErrorCode ierr;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(flux_minus != NULL){
      ierr = VecGetArray(flux_minus[dim], &flux_minus_p[dim]); CHKERRXX(ierr); }
    if(flux_plus != NULL){
      ierr = VecGetArray(flux_plus[dim], &flux_plus_p[dim]); CHKERRXX(ierr); }
    if(velocities_provided)
    {
      ierr = VecGetArray(face_velocity_minus[dim],  &face_velocity_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(face_velocity_plus[dim],   &face_velocity_plus_p[dim]);  CHKERRXX(ierr);
    }
  }

  // layer faces, first
  for (u_char dim = 0; dim < P4EST_DIM; ++dim){
    for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
    {
      const p4est_locidx_t f_idx = faces->get_layer_face(dim, k);
      local_projection_for_face(f_idx, dim, faces, flux_minus_p, flux_plus_p, face_velocity_minus_p, face_velocity_plus_p);
    }
    // start the ghost updates
    if(flux_minus != NULL){
      ierr = VecGhostUpdateBegin(flux_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    if(flux_plus != NULL && flux_plus[dim] != flux_minus[dim]){
      ierr = VecGhostUpdateBegin(flux_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    if(velocities_provided){
      ierr = VecGhostUpdateBegin(face_velocity_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(face_velocity_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  // inner faces, second
  for (u_char dim = 0; dim < P4EST_DIM; ++dim){
    for (size_t k = 0; k < faces->get_local_size(dim); ++k)
    {
      const p4est_locidx_t f_idx = faces->get_local_face(dim, k);
      local_projection_for_face(f_idx, dim, faces, flux_minus_p, flux_plus_p, face_velocity_minus_p, face_velocity_plus_p);
    }
    // finish the ghost updates
    if(flux_minus != NULL){
      ierr = VecGhostUpdateEnd(flux_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    if(flux_plus != NULL && flux_plus[dim] != flux_minus[dim]){
      ierr = VecGhostUpdateEnd(flux_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    if(velocities_provided){
      ierr = VecGhostUpdateEnd(face_velocity_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(face_velocity_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(flux_minus != NULL){
      ierr = VecRestoreArray(flux_minus[dim], &flux_minus_p[dim]); CHKERRXX(ierr); }
    if(flux_plus != NULL){
      ierr = VecRestoreArray(flux_plus[dim], &flux_plus_p[dim]); CHKERRXX(ierr); }
    if(velocities_provided)
    {
      ierr = VecRestoreArray(face_velocity_minus[dim],  &face_velocity_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(face_velocity_plus[dim],   &face_velocity_plus_p[dim]);  CHKERRXX(ierr);
    }
  }

  return;
}
