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
  extrapolation_minus = extrapolation_plus = NULL;
  interp_jump_u           = NULL;
  interp_jump_normal_flux = NULL;

  A = NULL;
  sqrt_reciprocal_diagonal = my_own_nullspace_vector = NULL;
  A_null_space = NULL;
  scale_system_by_diagonals = true;
  bc = NULL;
  matrix_is_set = rhs_is_set = false;
  extrapolations_are_set = false;
  use_extrapolations_in_sharp_flux_calculations = false;

  relative_tolerance    = 1.0e-12;
  absolute_tolerance    = PETSC_DEFAULT;
  divergence_tolerance  = PETSC_DEFAULT;
  max_ksp_iterations    = PETSC_DEFAULT;

  const splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;

  // Domain and grid parameters
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    dxyz_min[dim] = tree_dimensions[dim]/(double) (1 << data->max_lvl);
}

my_p4est_poisson_jump_cells_t::~my_p4est_poisson_jump_cells_t()
{
  PetscErrorCode ierr;
  if (A                         != NULL)  { ierr = MatDestroy(A);                         CHKERRXX(ierr); }
  if (sqrt_reciprocal_diagonal  != NULL)  { ierr = VecDestroy(sqrt_reciprocal_diagonal);  CHKERRXX(ierr); }
  if (my_own_nullspace_vector   != NULL)  { ierr = VecDestroy(my_own_nullspace_vector);   CHKERRXX(ierr); }
  if (A_null_space              != NULL)  { ierr = MatNullSpaceDestroy (A_null_space);    CHKERRXX(ierr); }
  if (ksp                       != NULL)  { ierr = KSPDestroy(ksp);                       CHKERRXX(ierr); }
  if (solution                  != NULL)  { ierr = VecDestroy(solution);                  CHKERRXX(ierr); }
  if (extrapolation_minus       != NULL)  { ierr = VecDestroy(extrapolation_minus);       CHKERRXX(ierr); }
  if (extrapolation_plus        != NULL)  { ierr = VecDestroy(extrapolation_plus);        CHKERRXX(ierr); }
  if (rhs                       != NULL)  { ierr = VecDestroy(rhs);                       CHKERRXX(ierr); }
  if (interp_jump_u             != NULL)  { delete interp_jump_u;                                         }
  if (interp_jump_normal_flux   != NULL)  { delete interp_jump_normal_flux;                               }
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
  extrapolations_are_set = false;
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
  extrapolations_are_set = false;
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

void my_p4est_poisson_jump_cells_t::pointwise_operation_with_sqrt_of_diag(size_t num_vectors, ...) const
{
  P4EST_ASSERT(sqrt_reciprocal_diagonal != NULL);
  PetscErrorCode ierr;
  va_list ap;
  va_start(ap, num_vectors);
  std::vector<Vec> vectors(num_vectors);
  std::vector<int> operation(num_vectors);
  std::vector<bool> is_ghosted(num_vectors);
  std::vector<double *> vectors_p(num_vectors);
  const double *sqrt_reciprocal_diagonal_p;
  ierr = VecGetArrayRead(sqrt_reciprocal_diagonal, &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);
  for (size_t k = 0; k < num_vectors; ++k){
    // get what we need
    vectors[k]    = va_arg(ap, Vec);
    operation[k]  = va_arg(ap, int);
#ifdef CASL_THROWS
    switch (operation[k]) { // --> check the validity of that input only at initialization and only in DEBUG
    case multiply_by_sqrt_D:
    case divide_by_sqrt_D:
      break;
    default:
      throw std::invalid_argument("my_p4est_poisson_jump_cells::pointwise_operation_with_sqrt_of_diag : unknown operation");
      break;
    }
#endif
    ierr = VecGetArray(vectors[k], &vectors_p[k]); CHKERRXX(ierr);
    // check if it's ghosted
    Vec tmp;
    ierr = VecGhostGetLocalForm(vectors[k], &tmp); CHKERRXX(ierr);
    is_ghosted[k] = (tmp != NULL);
    ierr = VecGhostRestoreLocalForm(vectors[k], &tmp); CHKERRXX(ierr);
  }
  // do the desired task(s) now
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k)
  {
    const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k);
    for (size_t nn = 0; nn < num_vectors; ++nn)
    {
      if(operation[nn] == multiply_by_sqrt_D)
        vectors_p[nn][quad_idx] /= sqrt_reciprocal_diagonal_p[quad_idx];
      else
        vectors_p[nn][quad_idx] *= sqrt_reciprocal_diagonal_p[quad_idx];
    }
  }
  for (size_t nn = 0; nn < num_vectors; ++nn)
    if(is_ghosted[nn]){
      ierr = VecGhostUpdateBegin(vectors[nn], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k)
  {
    const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k);
    for (size_t nn = 0; nn < num_vectors; ++nn)
    {
      if(operation[nn] == multiply_by_sqrt_D)
        vectors_p[nn][quad_idx] /= sqrt_reciprocal_diagonal_p[quad_idx];
      else
        vectors_p[nn][quad_idx] *= sqrt_reciprocal_diagonal_p[quad_idx];
    }
  }
  for (size_t nn = 0; nn < num_vectors; ++nn){
    if(is_ghosted[nn]){
      ierr = VecGhostUpdateEnd(vectors[nn], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    ierr = VecRestoreArray(vectors[nn], &vectors_p[nn]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(sqrt_reciprocal_diagonal, &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);

  va_end(ap);
  return;
}

void my_p4est_poisson_jump_cells_t::solve_linear_system()
{
  PetscErrorCode ierr;

  bool sensible_guess = (solution != NULL);
  if (solution == NULL) {
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr);
    if(user_initial_guess != NULL){
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
      ierr = VecCopyGhost(user_initial_guess, solution); CHKERRXX(ierr);
      sensible_guess = true;
    }
  }

  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  if(scale_system_by_diagonals)
  {
    if(sensible_guess) // scale the initial guess as well in that case...
      pointwise_operation_with_sqrt_of_diag(2, solution, multiply_by_sqrt_D, rhs, divide_by_sqrt_D);
    else
      pointwise_operation_with_sqrt_of_diag(1, rhs, divide_by_sqrt_D); // you need to scale the rhs
  }
  // solve the system
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  // finalize :
  if(scale_system_by_diagonals) // get the true solution and scale the rhs back to its original state (critical in case of iterative method playing with the rhs) scale the initial guess as well in that case...
    pointwise_operation_with_sqrt_of_diag(2, solution, divide_by_sqrt_D, rhs, multiply_by_sqrt_D); // ghost updates done therein...
  else
  {
    // we need update ghost values of the solution for accurate calculation of the extended interface values in xGFM
    ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  KSPConvergedReason termination_reason;
  ierr = KSPGetConvergedReason(ksp, &termination_reason); CHKERRXX(ierr);
  if(termination_reason <= 0)
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::solve_linear_system() : the Krylov solver failed to converge for a linear system to solve, the KSPConvergedReason code is " + std::to_string(termination_reason)); // collective runtime_error throw
#endif
  extrapolations_are_set = false;
  return;
}

PetscErrorCode my_p4est_poisson_jump_cells_t::setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type) const
{
  PetscErrorCode ierr;
  ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER); CHKERRQ(ierr);
  // set ksp type
  ierr = KSPSetType(ksp, (A_null_space != NULL ? KSPGMRES : ksp_type)); CHKERRQ(ierr); // PetSc advises GMRES for problems with nullspaces

  ierr = KSPSetTolerances(ksp, relative_tolerance, absolute_tolerance, divergence_tolerance, max_ksp_iterations); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRQ(ierr);

  if (!strcmp(pc_type, PCHYPRE)){
    // One can find more details about the following parameters in "Parallel Algebraic Multigrid Methods - High
    // Performance Preconditioners", Yang, Ulrike Meier, Numerical solution of partial differential equations
    // on parallel computers. Springer, Berlin, Heidelberg, 2006. 209-236.
    // --> https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/yang1.pdf

    // A large value for the following one was found to be more robust for the jump problems, in particular
    // when using Daniil's FV solver with large ratios of diffusion coefficients. (The analysis was done for
    // the cell FV solver with no constant coefficient, i.e. for pure Poisson problems)
    // The value of this parameter should be in )0, 1( and HYPRE manual suggests 0.25 for 2D problems, 0.5
    // for 3D. I believe those values are implicitly understood as "when considering the standard discretization
    // of the Laplace operators on uniform grids", though.
    // I found that such low values lead to successful resolution of the linear systems only when scaling the
    // systems by their diagonals. Otherwise a (very) large value (e.g. 0.9) was required to make it work fine.
    // (from my understanding of the above paper, this parameter is basically a threshold determining whether or not
    // offdiagonal terms are to be considered during coarsening steps in the Algebraic MultiGrid (AMG) methods : it
    // translates into an (algebraic) criterion excluding points that are "not sufficiently strongly connected" to
    // each other when coarsening. --> For jump problems with large coefficient ratios, you may actually need to exclude
    // points belonging to the other subdomain when coarsening your solution for best results... Hence the need for a
    // large (i.e. more selective) threshold value if not scaling the system by its diagonal
#ifdef P4_TO_P8
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.50"); CHKERRQ(ierr);
#else
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.25"); CHKERRQ(ierr);
#endif

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRQ(ierr);

    /* 3- Truncation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0"); CHKERRQ(ierr);

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
  int original_nullspace_contains_constant_vector = !matrix_is_set; // converted to integer because of required MPI collective determination thereafter + we don't care about that if the matrix is already set

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      build_discretization_for_quad(q + tree->quadrants_offset, tree_idx, &original_nullspace_contains_constant_vector);
  }

  if(!matrix_is_set)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

    // reset null space and check if we need to build one
    if(A_null_space != NULL){
      ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);
      A_null_space = NULL;
    }
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &original_nullspace_contains_constant_vector, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    if(scale_system_by_diagonals)
    {
      ierr = delete_and_nullify_vector(sqrt_reciprocal_diagonal); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(my_own_nullspace_vector); CHKERRXX(ierr);
      double L2_norm_my_own_nullspace_vector = 0.0;
      ierr = VecCreateNoGhostCells(p4est, &sqrt_reciprocal_diagonal); CHKERRXX(ierr);
      if(original_nullspace_contains_constant_vector) {
        ierr = VecCreateNoGhostCells(p4est, &my_own_nullspace_vector); CHKERRXX(ierr); }

      ierr = MatGetDiagonal(A, sqrt_reciprocal_diagonal); CHKERRXX(ierr);
      double *sqrt_reciprocal_diagonal_p;
      double *my_own_nullspace_p = NULL;
      ierr = VecGetArray(sqrt_reciprocal_diagonal, &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);
      if(my_own_nullspace_vector != NULL){
        ierr = VecGetArray(my_own_nullspace_vector, &my_own_nullspace_p); CHKERRXX(ierr); }
      for (p4est_locidx_t quad_idx = 0; quad_idx < p4est->local_num_quadrants; ++quad_idx) {
        if(fabs(sqrt_reciprocal_diagonal_p[quad_idx]) < EPS)
        {
          if(my_own_nullspace_p != NULL)
            my_own_nullspace_p[quad_idx] = 1.0;
          sqrt_reciprocal_diagonal_p[quad_idx] = 1.0; // not touching that one, too risky...
        }
        else
        {
          if(my_own_nullspace_p != NULL)
            my_own_nullspace_p[quad_idx] = sqrt(fabs(sqrt_reciprocal_diagonal_p[quad_idx]));
          sqrt_reciprocal_diagonal_p[quad_idx] = 1.0/sqrt(fabs(sqrt_reciprocal_diagonal_p[quad_idx]));
        }
        if(my_own_nullspace_p != NULL)
          L2_norm_my_own_nullspace_vector += SQR(my_own_nullspace_p[quad_idx]);
      }
      if(my_own_nullspace_vector != NULL){
        ierr = VecRestoreArray(my_own_nullspace_vector, &my_own_nullspace_p); CHKERRXX(ierr); }
      ierr = VecRestoreArray(sqrt_reciprocal_diagonal, &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);

      // scale the matrix in a symmetric fashion! (we do need to scale the rhs too, but that operation
      // is done right before calling KSPSolve and undone right after since we need to iteratively
      // update it in the xGFM strategy)
      ierr = MatDiagonalScale(A, sqrt_reciprocal_diagonal, sqrt_reciprocal_diagonal); CHKERRXX(ierr);
      if(original_nullspace_contains_constant_vector){
        mpiret = MPI_Allreduce(MPI_IN_PLACE, &L2_norm_my_own_nullspace_vector, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
        L2_norm_my_own_nullspace_vector = sqrt(L2_norm_my_own_nullspace_vector);
        ierr = VecScale(my_own_nullspace_vector, 1.0/L2_norm_my_own_nullspace_vector);
        ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &my_own_nullspace_vector, &A_null_space); CHKERRXX(ierr);
      }
    }
    else if(original_nullspace_contains_constant_vector){
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, NULL, &A_null_space); CHKERRXX(ierr);
    }

    if(A_null_space != NULL)
    {
      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr); // --> required to handle the rhs right under the hood (see note "about removing nullspaces from RHS" here under)
      /*
       * - In case of xGFM, if the constant vector is in the right nullspace, it's also in the left nullspace
       * since the discretization creates an SPD matrix
       * - In case of the FV approach, if the constant vector is in the right null space, it is also in the left,
       * by construction. Indeed, whichever nonsymmetric, correction-function-related contribution entering the
       * discretized equation for quadrant (A) neighbor with (B) also enters the discretized equation for quadrant
       * (B) neighbor with (A) but with an opposite sign (balance of fluxes).
       * */
    }
  }
  /* [Raphael (05/17/2020) : "about removing nullspaces from RHS"
   * removing the null space from the rhs seems redundant with PetSc operations done under the
   * hood in KSPSolve (see the source code of KSPSolve_Private in /src/ksp/ksp/interface/itfunc.c
   * for more details). --> So long as MatSetTransposeNullSpace() was called appropriately on the
   * matrix of interest, this operation will be executed in the pre-steps of KSPSolve on a COPY of
   * the provided RHS vector]
   * --> this is precisely what we want : we need the _unmodified_ RHS thereafter, i.e. as we have
   * built it, and we let PetSc do its magic under the hood every time we call KSPSolve, otherwise
   * iteratively correcting and updating the RHS would become very complex and require extra info
   * coming from those possibly non-empty nullspace contributions...
   * */

  matrix_is_set = rhs_is_set = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_cells_t::project_face_velocities(const my_p4est_faces_t *faces, Vec *flux_minus, Vec *flux_plus)
{
  P4EST_ASSERT(faces->get_p4est() == p4est); // the faces must be built from the same computational grid
  P4EST_ASSERT((face_velocity_minus == NULL && face_velocity_plus == NULL) ||
               (VecsAreSetForFaces(face_velocity_minus, faces, 1) && VecsAreSetForFaces(face_velocity_plus, faces, 1))); // the face-sampled velocities vstart and vnp1 vectors vectors must either be all defined or be all NULL.
  P4EST_ASSERT(flux_minus == NULL || VecsAreSetForFaces(flux_minus, faces, 1));
  P4EST_ASSERT(flux_plus == NULL || VecsAreSetForFaces(flux_plus, faces, 1));

#ifdef CASL_THROWS
  if((extrapolation_minus == NULL || extrapolation_plus == NULL) && solution == NULL)
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::project_face_velocities(): requires either the extrapolated or the sharp solution(s), have you called solve() before?");
#endif

  // check if we want/need to use extrapolations only when calculating flux components...
  if(use_extrapolations_in_sharp_flux_calculations && (extrapolation_minus == NULL || extrapolation_plus == NULL || !extrapolations_are_set))
    extrapolate_solution_from_either_side_to_the_other(20);

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

void my_p4est_poisson_jump_cells_t::extrapolate_solution_from_either_side_to_the_other(const u_int& n_pseudo_time_iterations, const u_char& degree)
{
  if(extrapolations_are_set)
    return;

  P4EST_ASSERT(n_pseudo_time_iterations > 0);
  P4EST_ASSERT(solution != NULL);

  PetscErrorCode ierr;
  const double *sharp_solution_p;
  double *extrapolation_minus_p, *extrapolation_plus_p;
  Vec tmp_plus, tmp_minus;
  double *tmp_plus_p, *tmp_minus_p;
  ierr = VecCreateGhostCells(p4est, ghost, &tmp_plus); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est, ghost, &tmp_minus); CHKERRXX(ierr);
  if(extrapolation_minus == NULL){
    ierr = VecCreateGhostCells(p4est, ghost, &extrapolation_minus); CHKERRXX(ierr); }
  if(extrapolation_plus == NULL){
    ierr = VecCreateGhostCells(p4est, ghost, &extrapolation_plus); CHKERRXX(ierr); }
  ierr = VecGetArrayRead(solution, &sharp_solution_p); CHKERRXX(ierr);
  ierr = VecGetArray(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
  ierr = VecGetArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
  ierr = VecGetArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
  // normal derivatives of the solution (required if degree >= 1)
  Vec normal_derivative_of_solution_minus = NULL;
  Vec normal_derivative_of_solution_plus  = NULL;
  double *normal_derivative_of_solution_minus_p = NULL;
  double *normal_derivative_of_solution_plus_p  = NULL;

  // get pointers, initialize normal derivatives of the fields, etc.
  if(degree >= 1)
  {
    ierr = VecCreateGhostCells(p4est, ghost, &normal_derivative_of_solution_minus); CHKERRXX(ierr);
    ierr = VecCreateGhostCells(p4est, ghost, &normal_derivative_of_solution_plus); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_solution_minus, &normal_derivative_of_solution_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_solution_plus, &normal_derivative_of_solution_plus_p); CHKERRXX(ierr);
    extrapolation_operator_minus.clear();
    extrapolation_operator_plus.clear();
  }
  // INITIALIZE extrapolation
  // local layer cells first
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k)
    initialize_extrapolation_local(cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k), cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k),
                                   sharp_solution_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p, degree);
  // start updates
  ierr = VecGhostUpdateBegin(extrapolation_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(extrapolation_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(degree >= 1)
  {
    ierr = VecGhostUpdateBegin(normal_derivative_of_solution_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(normal_derivative_of_solution_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  // local inner cells
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k)
    initialize_extrapolation_local(cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k), cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k),
                                   sharp_solution_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p, degree);
  // finish updates
  ierr = VecGhostUpdateEnd(extrapolation_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(extrapolation_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(degree >= 1)
  {
    ierr = VecGhostUpdateEnd(normal_derivative_of_solution_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(normal_derivative_of_solution_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* EXTRAPOLATE normal derivatives of solution */
  if(degree >= 1)
  {
    for (u_int iter = 0; iter < n_pseudo_time_iterations; ++iter) {
      // local layer cells first
      for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k)
        extrapolate_normal_derivatives_local(cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k),
                                             tmp_minus_p, tmp_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p);
      // start updates
      ierr = VecGhostUpdateBegin(tmp_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(tmp_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      // local inner cells
      for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k)
        extrapolate_normal_derivatives_local(cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k),
                                             tmp_minus_p, tmp_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p);
      // complete updates
      ierr = VecGhostUpdateEnd(tmp_minus, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(tmp_plus, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);

      // swap (n) and (n + 1) pseudo time iterates (more efficient to swap pointers than copying large chunks of data)
      ierr = VecRestoreArray(normal_derivative_of_solution_minus, &normal_derivative_of_solution_minus_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_derivative_of_solution_plus, &normal_derivative_of_solution_plus_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
      std::swap(tmp_minus, normal_derivative_of_solution_minus);
      std::swap(tmp_plus, normal_derivative_of_solution_plus);
      ierr = VecGetArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_solution_plus, &normal_derivative_of_solution_plus_p); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_solution_minus, &normal_derivative_of_solution_minus_p); CHKERRXX(ierr);
    }
  }

  /* EXTRAPOLATE the solution */
  for (u_int iter = 0; iter < n_pseudo_time_iterations; ++iter) {
    // local layer cells first
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k)
      extrapolate_solution_local(cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k), cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k), sharp_solution_p,
                                 tmp_minus_p, tmp_plus_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p);
    // start updates
    ierr = VecGhostUpdateBegin(tmp_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(tmp_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    // local inner cells
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k)
      extrapolate_solution_local(cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k), cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k), sharp_solution_p,
                                 tmp_minus_p, tmp_plus_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p);
    // start updates
    ierr = VecGhostUpdateEnd(tmp_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(tmp_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // swap (n) and (n + 1) pseudo time iterates (more efficient to swap pointers than copying large chunks of data)
    ierr = VecRestoreArray(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
    std::swap(tmp_minus, extrapolation_minus);
    std::swap(tmp_plus, extrapolation_plus);
    ierr = VecGetArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
    ierr = VecGetArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
  }

  // restore data pointers and delete locally created vectors
  if(degree >= 1)
  {
    ierr = VecRestoreArray(normal_derivative_of_solution_minus, &normal_derivative_of_solution_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_solution_plus, &normal_derivative_of_solution_plus_p); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(normal_derivative_of_solution_minus);  CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(normal_derivative_of_solution_plus); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tmp_minus, &tmp_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tmp_plus, &tmp_plus_p); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(tmp_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(tmp_plus); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &sharp_solution_p); CHKERRXX(ierr);

  extrapolations_are_set = true;

  return;
}

void my_p4est_poisson_jump_cells_t::extrapolate_normal_derivatives_local(const p4est_locidx_t& quad_idx,
                                                                         double* tmp_minus_p, double* tmp_plus_p,
                                                                         const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p) const
{
  tmp_minus_p[quad_idx] = normal_derivative_of_solution_minus_p[quad_idx];
  std::map<p4est_locidx_t, extrapolation_operator_t>::const_iterator it = extrapolation_operator_minus.find(quad_idx);
#ifdef P4EST_DEBUG
  bool found_one = false;
#endif
  if(it != extrapolation_operator_minus.end())
  {
    tmp_minus_p[quad_idx] -= it->second.n_dot_grad(normal_derivative_of_solution_minus_p)*it->second.dtau;
#ifdef P4EST_DEBUG
    found_one = true;
#endif
  }

  tmp_plus_p[quad_idx] = normal_derivative_of_solution_plus_p[quad_idx];
  it = extrapolation_operator_plus.find(quad_idx);
  if(it != extrapolation_operator_plus.end())
  {
    tmp_plus_p[quad_idx] -= it->second.n_dot_grad(normal_derivative_of_solution_plus_p)*it->second.dtau;
#ifdef P4EST_DEBUG
    found_one = true;
#endif
  }
  P4EST_ASSERT(found_one);
  return;
}

