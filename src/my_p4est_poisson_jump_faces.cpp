#include "my_p4est_poisson_jump_faces.h"

//#ifdef P4_TO_P8
//#include <src/my_p8est_vtk.h>
//#else
//#include <src/my_p4est_vtk.h>
//#endif

my_p4est_poisson_jump_faces_t::my_p4est_poisson_jump_faces_t(const my_p4est_faces_t *faces_, const p4est_nodes_t* nodes_)
  : faces(faces_), p4est(faces_->get_p4est()), ghost(faces_->get_ghost()), nodes(nodes_), ngbd_c(faces_->get_ngbd_c()),
    xyz_min(faces_->get_p4est()->connectivity->vertices + 3*faces_->get_p4est()->connectivity->tree_to_vertex[0]),
    xyz_max(faces_->get_p4est()->connectivity->vertices + 3*faces_->get_p4est()->connectivity->tree_to_vertex[P4EST_CHILDREN*(faces_->get_p4est()->trees->elem_count - 1) + P4EST_CHILDREN - 1]),
    tree_dimensions(faces_->get_tree_dimensions()), periodicity(faces_->get_periodicity())
{
  // set up the KSP solvers
  PetscErrorCode ierr;
  mu_minus = mu_plus = -1.0;
  add_diag_minus = add_diag_plus = 0.0;
  // initialize data to be set by user to NULL (reference 'unset' state)
  user_rhs_minus = user_rhs_plus = NULL;
  jump_u_dot_n = jump_tangential_stress = NULL;
  interp_jump_u_dot_n = interp_jump_tangential_stress = NULL;
  user_initial_guess = NULL;

  max_ksp_iterations    = PETSC_DEFAULT;
  relative_tolerance    = 1.0e-12;
  absolute_tolerance    = PETSC_DEFAULT;
  divergence_tolerance  = PETSC_DEFAULT;
  max_iter              = INT_MAX;        // default is "no limit" on number of internal iterations (get that f****** stress balance)

  const splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    solution[dim] = NULL;
    extrapolation_minus[dim]  = NULL; extrapolation_operator_minus[dim].clear();
    extrapolation_plus[dim]   = NULL; extrapolation_operator_plus[dim].clear();
    matrix[dim] = NULL;
    sqrt_reciprocal_diagonal[dim] = NULL;
    my_own_nullspace_vector[dim] = NULL;
    null_space[dim] = NULL;
    ierr = KSPCreate(p4est->mpicomm, &ksp[dim]); CHKERRXX(ierr);
    rhs[dim] = NULL;
    matrix_is_set[dim] = false;
    rhs_is_set[dim] = false;
    dxyz_min[dim] = tree_dimensions[dim]/(double) (1 << data->max_lvl);
    voronoi_cell[dim].resize(faces->num_local[dim]); // default behavior is *NOT* on the fly
    all_voronoi_cells_are_set[dim] = false;
    extrapolation_operators_are_stored_and_set[dim] = false;
  }
  bc = NULL;
  scale_systems_by_diagonals = false;
  extrapolations_are_set = false;

  voronoi_on_the_fly = false;
}

my_p4est_poisson_jump_faces_t::~my_p4est_poisson_jump_faces_t()
{
  PetscErrorCode ierr;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = delete_and_nullify_vector(solution[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_minus[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_plus[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(sqrt_reciprocal_diagonal[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(my_own_nullspace_vector[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(rhs[dim]); CHKERRXX(ierr);
    if(matrix[dim] != NULL)                   { ierr = MatDestroy(matrix[dim]);               CHKERRXX(ierr); }
    if(null_space[dim] != NULL)               { ierr = MatNullSpaceDestroy(null_space[dim]);  CHKERRXX(ierr); }
    if(ksp[dim] != NULL)                      { ierr = KSPDestroy(ksp[dim]);                  CHKERRXX(ierr); }
  }
  if(interp_jump_u_dot_n != NULL)           { delete interp_jump_u_dot_n;           }
  if(interp_jump_tangential_stress != NULL) { delete interp_jump_tangential_stress; }
}

void my_p4est_poisson_jump_faces_t::preallocate_matrix(const u_char& dim)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_faces_matrix_preallocation, A[dim], dim, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(!matrix_is_set[dim]);

  if (matrix[dim] != NULL){
    ierr = MatDestroy(matrix[dim]); CHKERRXX(ierr); }

  const PetscInt num_owned_local  = faces->num_local[dim];
  const PetscInt num_owned_global = faces->proc_offset[dim][p4est->mpisize];

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &matrix[dim]); CHKERRXX(ierr);
  ierr = MatSetType(matrix[dim], MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(matrix[dim], num_owned_local, num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(matrix[dim]); CHKERRXX(ierr);

  std::vector<PetscInt> nlocal_nonzeros(num_owned_local), nghost_nonzeros(num_owned_local);

  for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dim]; ++f_idx)
    get_numbers_of_faces_involved_in_equation_for_face(dim, f_idx, nlocal_nonzeros[f_idx], nghost_nonzeros[f_idx]);

  ierr = MatSeqAIJSetPreallocation(matrix[dim], 0, nlocal_nonzeros.data()); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(matrix[dim], 0, nlocal_nonzeros.data(), 0, nghost_nonzeros.data()); CHKERRXX(ierr);

  if(!all_voronoi_cells_are_set[dim] && !voronoi_on_the_fly)
    all_voronoi_cells_are_set[dim] = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_faces_matrix_preallocation, A[dim], dim, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_faces_t::set_interface(my_p4est_interface_manager_t* interface_manager_)
{
  P4EST_ASSERT(interface_manager_ != NULL);
  interface_manager = interface_manager_;

  if(!interface_manager->is_grad_phi_set())
    interface_manager->set_grad_phi();

  if(!interface_manager->is_curvature_set())
    interface_manager->set_curvature(); // you may need it (if jump_u_dot_n is nonzero)!

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    matrix_is_set[dim] = rhs_is_set[dim] = false;

  extrapolations_are_set = false;
  return;
}

void my_p4est_poisson_jump_faces_t::set_jumps(Vec jump_u_dot_n_, Vec jump_tangential_stress_)
{
  if(!interface_is_set())
    throw std::runtime_error("my_p4est_poisson_jump_faces_t::set_jumps(): the interface manager must be set before the jumps");
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
#ifdef P4EST_DEBUG
  P4EST_ASSERT(jump_u_dot_n_            == NULL || VecIsSetForNodes(jump_u_dot_n_,            interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, 1));
  P4EST_ASSERT(jump_tangential_stress_  == NULL || VecIsSetForNodes(jump_tangential_stress_,  interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, P4EST_DIM));
#endif

  jump_u_dot_n            = jump_u_dot_n_;
  jump_tangential_stress  = jump_tangential_stress_;

  if(interp_jump_u_dot_n != NULL && jump_u_dot_n == NULL){
    delete interp_jump_u_dot_n;
    interp_jump_u_dot_n = NULL;
  }
  if(interp_jump_u_dot_n == NULL && jump_u_dot_n != NULL)
    interp_jump_u_dot_n = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
  if(jump_u_dot_n != NULL)
    interp_jump_u_dot_n->set_input(jump_u_dot_n, linear);

  if(interp_jump_tangential_stress != NULL && jump_tangential_stress  == NULL){
    delete interp_jump_tangential_stress;
    interp_jump_tangential_stress = NULL;
  }
  if(interp_jump_tangential_stress == NULL && jump_tangential_stress != NULL)
    interp_jump_tangential_stress = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);

  if(jump_tangential_stress != NULL)
    interp_jump_tangential_stress->set_input(jump_tangential_stress, linear, P4EST_DIM);

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    rhs_is_set[dim] = false;

  extrapolations_are_set = false;
  return;
}

void my_p4est_poisson_jump_faces_t::pointwise_operation_with_sqrt_of_diag(const u_char& dim, size_t num_vectors, ...) const
{
  P4EST_ASSERT(sqrt_reciprocal_diagonal[dim] != NULL);
  PetscErrorCode ierr;
  va_list ap;
  va_start(ap, num_vectors);
  std::vector<Vec> vectors(num_vectors);
  std::vector<int> operation(num_vectors);
  std::vector<bool> is_ghosted(num_vectors);
  std::vector<double *> vectors_p(num_vectors);
  const double *sqrt_reciprocal_diagonal_p;
  ierr = VecGetArrayRead(sqrt_reciprocal_diagonal[dim], &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);
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
  for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
  {
    const p4est_locidx_t face_idx = faces->get_layer_face(dim, k);
    for (size_t nn = 0; nn < num_vectors; ++nn)
    {
      if(operation[nn] == multiply_by_sqrt_D)
        vectors_p[nn][face_idx] /= sqrt_reciprocal_diagonal_p[face_idx];
      else
        vectors_p[nn][face_idx] *= sqrt_reciprocal_diagonal_p[face_idx];
    }
  }
  for (size_t nn = 0; nn < num_vectors; ++nn)
    if(is_ghosted[nn]){
      ierr = VecGhostUpdateBegin(vectors[nn], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  for (size_t k = 0; k < faces->get_local_size(dim); ++k)
  {
    const p4est_locidx_t face_idx = faces->get_local_face(dim, k);
    for (size_t nn = 0; nn < num_vectors; ++nn)
    {
      if(operation[nn] == multiply_by_sqrt_D)
        vectors_p[nn][face_idx] /= sqrt_reciprocal_diagonal_p[face_idx];
      else
        vectors_p[nn][face_idx] *= sqrt_reciprocal_diagonal_p[face_idx];
    }
  }
  for (size_t nn = 0; nn < num_vectors; ++nn){
    if(is_ghosted[nn]){
      ierr = VecGhostUpdateEnd(vectors[nn], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    ierr = VecRestoreArray(vectors[nn], &vectors_p[nn]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(sqrt_reciprocal_diagonal[dim], &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);

  va_end(ap);
  return;
}

void my_p4est_poisson_jump_faces_t::solve_linear_systems()
{
  PetscErrorCode ierr;

  bool sensible_guess[P4EST_DIM] = {DIM(false, false, false)};
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    sensible_guess[dim] = (solution[dim] != NULL);
    if (solution[dim] == NULL) {
      ierr = VecCreateGhostFaces(p4est, faces, &solution[dim], dim); CHKERRXX(ierr);
      if(user_initial_guess != NULL && user_initial_guess[dim] != NULL){
        ierr = KSPSetInitialGuessNonzero(ksp[dim], PETSC_TRUE); CHKERRXX(ierr);
        ierr = VecCopyGhost(user_initial_guess[dim], solution[dim]); CHKERRXX(ierr);
        sensible_guess[dim] = true;
      }
    }
  }

  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_faces_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(scale_systems_by_diagonals)
    {
      if(sensible_guess[dim]) // scale the initial guess as well in that case...
        pointwise_operation_with_sqrt_of_diag(dim, 2, solution[dim], multiply_by_sqrt_D, rhs[dim], divide_by_sqrt_D);
      else
        pointwise_operation_with_sqrt_of_diag(dim, 1, rhs[dim], divide_by_sqrt_D); // you need to scale the rhs
    }
    // solve the system
    ierr = KSPSolve(ksp[dim], rhs[dim], solution[dim]); CHKERRXX(ierr);

    // check that everything went well
#ifdef CASL_THROWS
    KSPConvergedReason termination_reason;
    ierr = KSPGetConvergedReason(ksp[dim], &termination_reason); CHKERRXX(ierr);
    if(termination_reason <= 0)
      throw std::runtime_error("my_p4est_poisson_jump_faces_t::solve_linear_system() : the Krylov solver failed to converge for a linear system to solve, the KSPConvergedReason code is " + std::to_string(termination_reason)); // collective runtime_error throw
#endif

    // finalize :
    if(scale_systems_by_diagonals) // get the true solution and scale the rhs back to its original state (critical in case of iterative method playing with the rhs) scale the initial guess as well in that case...
      pointwise_operation_with_sqrt_of_diag(dim, 2, solution[dim], divide_by_sqrt_D, rhs[dim], multiply_by_sqrt_D); // ghost updates done therein...
    else
    {
      // we need update ghost values of the solution for accurate calculation of the extended interface values in xGFM
      ierr = VecGhostUpdateBegin(solution[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  if(!scale_systems_by_diagonals)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      // complete ghost update
      ierr = VecGhostUpdateEnd(solution[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_faces_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  extrapolations_are_set = false;
  return;
}

PetscErrorCode my_p4est_poisson_jump_faces_t::setup_linear_solver(const u_char& dim, const KSPType& ksp_type, const PCType& pc_type) const
{
  PetscErrorCode ierr;
  ierr = KSPSetOperators(ksp[dim], matrix[dim], matrix[dim], SAME_PRECONDITIONER); CHKERRQ(ierr);
  // set ksp type
  ierr = KSPSetType(ksp[dim], (null_space[dim] != NULL ? KSPGMRES : ksp_type)); CHKERRQ(ierr); // PetSc advises GMRES for problems with nullspaces

  ierr = KSPSetTolerances(ksp[dim], relative_tolerance, absolute_tolerance, divergence_tolerance, max_ksp_iterations); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp[dim]); CHKERRQ(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp[dim], &pc); CHKERRQ(ierr);
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
    if (null_space[dim] != NULL){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRQ(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  return ierr;
}

void my_p4est_poisson_jump_faces_t::setup_linear_system(const u_char& dim)
{
  if(matrix_is_set[dim] && rhs_is_set[dim])
    return;

  if(!matrix_is_set[dim])
    preallocate_matrix(dim);

  PetscErrorCode ierr;
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_faces_setup_linear_system, A[dim], dim, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(interface_is_set()); // otherwise, well, it's going to be hard...

  if(!rhs_is_set[dim])
  {
    if(rhs[dim] == NULL){
      ierr = VecCreateNoGhostFaces(p4est, faces, &rhs[dim], dim); CHKERRXX(ierr); }
  }
  int original_nullspace_contains_constant_vector = !matrix_is_set[dim]; // converted to integer because of required MPI collective determination thereafter + we don't care about that if the matrix is already set

  for (p4est_locidx_t face_idx = 0; face_idx < faces->num_local[dim]; ++face_idx)
    build_discretization_for_face(dim, face_idx, &original_nullspace_contains_constant_vector);

  if(!matrix_is_set[dim])
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(matrix[dim], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (matrix[dim], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

    // reset null space and check if we need to build one
    if(null_space[dim] != NULL){
      ierr = MatNullSpaceDestroy(null_space[dim]); CHKERRXX(ierr);
      null_space[dim] = NULL;
    }
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &original_nullspace_contains_constant_vector, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    if(scale_systems_by_diagonals)
    {
      ierr = delete_and_nullify_vector(sqrt_reciprocal_diagonal[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(my_own_nullspace_vector[dim]); CHKERRXX(ierr);
      double L2_norm_my_own_nullspace_vector = 0.0;
      ierr = VecCreateNoGhostFaces(p4est, faces, &sqrt_reciprocal_diagonal[dim], dim); CHKERRXX(ierr);
      if(original_nullspace_contains_constant_vector) {
        ierr = VecCreateNoGhostFaces(p4est, faces, &my_own_nullspace_vector[dim], dim); CHKERRXX(ierr); }

      ierr = MatGetDiagonal(matrix[dim], sqrt_reciprocal_diagonal[dim]); CHKERRXX(ierr);
      double *sqrt_reciprocal_diagonal_p;
      double *my_own_nullspace_p = NULL;
      ierr = VecGetArray(sqrt_reciprocal_diagonal[dim], &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);
      if(my_own_nullspace_vector[dim] != NULL){
        ierr = VecGetArray(my_own_nullspace_vector[dim], &my_own_nullspace_p); CHKERRXX(ierr); }
      for (p4est_locidx_t face_idx = 0; face_idx < faces->num_local[dim]; ++face_idx) {
        if(fabs(sqrt_reciprocal_diagonal_p[face_idx]) < EPS)
        {
          if(my_own_nullspace_p != NULL)
            my_own_nullspace_p[face_idx] = 1.0;
          sqrt_reciprocal_diagonal_p[face_idx] = 1.0; // not touching that one, too risky...
        }
        else
        {
          if(my_own_nullspace_p != NULL)
            my_own_nullspace_p[face_idx] = sqrt(fabs(sqrt_reciprocal_diagonal_p[face_idx]));
          sqrt_reciprocal_diagonal_p[face_idx] = 1.0/sqrt(fabs(sqrt_reciprocal_diagonal_p[face_idx]));
        }
        if(my_own_nullspace_p != NULL)
          L2_norm_my_own_nullspace_vector += SQR(my_own_nullspace_p[face_idx]);
      }
      if(my_own_nullspace_vector[dim] != NULL){
        ierr = VecRestoreArray(my_own_nullspace_vector[dim], &my_own_nullspace_p); CHKERRXX(ierr); }
      ierr = VecRestoreArray(sqrt_reciprocal_diagonal[dim], &sqrt_reciprocal_diagonal_p); CHKERRXX(ierr);

      // scale the matrix in a symmetric fashion! (we do need to scale the rhs too, but that operation
      // is done right before calling KSPSolve and undone right after since we need to iteratively
      // update it in the xGFM strategy)
      ierr = MatDiagonalScale(matrix[dim], sqrt_reciprocal_diagonal[dim], sqrt_reciprocal_diagonal[dim]); CHKERRXX(ierr);
      if(original_nullspace_contains_constant_vector){
        mpiret = MPI_Allreduce(MPI_IN_PLACE, &L2_norm_my_own_nullspace_vector, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
        L2_norm_my_own_nullspace_vector = sqrt(L2_norm_my_own_nullspace_vector);
        ierr = VecScale(my_own_nullspace_vector[dim], 1.0/L2_norm_my_own_nullspace_vector);
        ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &my_own_nullspace_vector[dim], &null_space[dim]); CHKERRXX(ierr);
      }
    }
    else if(original_nullspace_contains_constant_vector){
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, NULL, &null_space[dim]); CHKERRXX(ierr);
    }

    if(null_space[dim] != NULL)
    {
      ierr = MatSetNullSpace(matrix[dim], null_space[dim]); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(matrix[dim], null_space[dim]); CHKERRXX(ierr); // --> required to handle the rhs right under the hood (see note "about removing nullspaces from RHS" here under)
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

  matrix_is_set[dim] = rhs_is_set[dim] = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_faces_setup_linear_system, A[dim], dim, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_faces_t::extrapolate_solution_from_either_side_to_the_other(const u_int& n_pseudo_time_iterations, const u_char& degree, double* sharp_max_component)
{
  if(extrapolations_are_set)
    return;

  P4EST_ASSERT(n_pseudo_time_iterations > 0);
  P4EST_ASSERT(solution != NULL && ANDD(solution[0] != NULL, solution[1] != NULL, solution[2] != NULL));

  PetscErrorCode ierr;
  const double *sharp_solution_p[P4EST_DIM];
  double *extrapolation_minus_p[P4EST_DIM], *extrapolation_plus_p[P4EST_DIM];
  Vec tmp_plus[P4EST_DIM], tmp_minus[P4EST_DIM];
  double *tmp_plus_p[P4EST_DIM], *tmp_minus_p[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostFaces(p4est, faces, &tmp_plus[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &tmp_minus[dim], dim); CHKERRXX(ierr);
    if(extrapolation_minus[dim] == NULL){
      ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_minus[dim], dim); CHKERRXX(ierr); }
    if(extrapolation_plus[dim] == NULL){
      ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_plus[dim], dim); CHKERRXX(ierr); }

    ierr = VecGetArrayRead(solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_minus[dim], &extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_plus[dim], &extrapolation_plus_p[dim]); CHKERRXX(ierr);
  }
  // normal derivatives of the solution (required if degree >= 1)
  Vec normal_derivative_of_solution_minus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  Vec normal_derivative_of_solution_plus[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *normal_derivative_of_solution_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  double *normal_derivative_of_solution_plus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_of_solution_minus_read_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_of_solution_plus_read_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};

  // get pointers, initialize normal derivatives of the fields, etc.
  if(degree >= 1)
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_of_solution_minus[dim], dim); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_of_solution_plus[dim], dim); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_p[dim]); CHKERRXX(ierr);
      if(!extrapolation_operators_are_stored_and_set[dim])
      {
        extrapolation_operator_minus[dim].clear();
        extrapolation_operator_plus[dim].clear();
      }
    }
  }
  // INITIALIZE extrapolation
  // local layer faces first
  if(sharp_max_component != NULL)
    sharp_max_component[0] = sharp_max_component[1] = 0.0;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
      initialize_extrapolation_local(dim, faces->get_layer_face(dim, k),
                                     sharp_solution_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p, degree, sharp_max_component);
    // start updates
    ierr = VecGhostUpdateBegin(extrapolation_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(extrapolation_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(degree >= 1)
    {
      ierr = VecGhostUpdateBegin(normal_derivative_of_solution_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(normal_derivative_of_solution_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  // local inner faces
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    for (size_t k = 0; k < faces->get_local_size(dim); ++k)
      initialize_extrapolation_local(dim, faces->get_local_face(dim, k),
                                     sharp_solution_p, extrapolation_minus_p, extrapolation_plus_p, normal_derivative_of_solution_minus_p, normal_derivative_of_solution_plus_p, degree, sharp_max_component);

    // finish updates
    ierr = VecGhostUpdateEnd(extrapolation_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(extrapolation_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(degree >= 1)
    {
      ierr = VecGhostUpdateEnd(normal_derivative_of_solution_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(normal_derivative_of_solution_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    extrapolation_operators_are_stored_and_set[dim] = true; // if they were not known yet, now they are!
  }
  // restore pointers
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(extrapolation_minus[dim], &extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(extrapolation_plus[dim], &extrapolation_plus_p[dim]); CHKERRXX(ierr);
    if(degree >= 1)
    {
      ierr = VecRestoreArray(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_p[dim]); CHKERRXX(ierr);
    }
  }
  if(sharp_max_component != NULL){
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, sharp_max_component, 2, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret); }

  /* EXTRAPOLATE normal derivatives of solution */
  if(degree >= 1)
  {
    for (u_int iter = 0; iter < n_pseudo_time_iterations; ++iter) {
      // get pointers
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        ierr = VecGetArray(tmp_minus[dim], &tmp_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArray(tmp_plus[dim], &tmp_plus_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArrayRead(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_read_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArrayRead(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_read_p[dim]); CHKERRXX(ierr);
      }

      // local layer faces first
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
          extrapolate_normal_derivatives_local(dim, faces->get_layer_face(dim, k),
                                               tmp_minus_p, tmp_plus_p, normal_derivative_of_solution_minus_read_p, normal_derivative_of_solution_plus_read_p);
        // start updates
        ierr = VecGhostUpdateBegin(tmp_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(tmp_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }

      // local inner faces first
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        for (size_t k = 0; k < faces->get_local_size(dim); ++k)
          extrapolate_normal_derivatives_local(dim, faces->get_local_face(dim, k),
                                               tmp_minus_p, tmp_plus_p, normal_derivative_of_solution_minus_read_p, normal_derivative_of_solution_plus_read_p);
        // finish updates
        ierr = VecGhostUpdateEnd(tmp_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(tmp_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }

      // restore pointers pointers
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        ierr = VecRestoreArray(tmp_minus[dim], &tmp_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArray(tmp_plus[dim], &tmp_plus_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_read_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_read_p[dim]); CHKERRXX(ierr);
      }

      // swap (n) and (n + 1) pseudo time iterates (more efficient to swap pointers than copying large chunks of data)
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        std::swap(tmp_minus[dim], normal_derivative_of_solution_minus[dim]);
        std::swap(tmp_plus[dim], normal_derivative_of_solution_plus[dim]);
      }
    }
  }

  const double *extrapolation_minus_read_p[P4EST_DIM], *extrapolation_plus_read_p[P4EST_DIM];
  // get normal derivative pointers
  if(degree >= 1)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecGetArrayRead(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_read_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_read_p[dim]); CHKERRXX(ierr);
    }

  /* EXTRAPOLATE the solution */
  for (u_int iter = 0; iter < n_pseudo_time_iterations; ++iter) {
    // get pointers
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecGetArray(tmp_minus[dim], &tmp_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_plus[dim], &tmp_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(extrapolation_minus[dim], &extrapolation_minus_read_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(extrapolation_plus[dim], &extrapolation_plus_read_p[dim]); CHKERRXX(ierr);
    }

    // local layer faces first
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
        extrapolate_solution_local(dim, faces->get_layer_face(dim, k), sharp_solution_p,
                                   tmp_minus_p, tmp_plus_p, extrapolation_minus_read_p, extrapolation_plus_read_p, normal_derivative_of_solution_minus_read_p, normal_derivative_of_solution_plus_read_p);
      // start updates
      ierr = VecGhostUpdateBegin(tmp_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(tmp_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    // local inner faces
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      for (size_t k = 0; k < faces->get_local_size(dim); ++k)
        extrapolate_solution_local(dim, faces->get_local_face(dim, k), sharp_solution_p,
                                   tmp_minus_p, tmp_plus_p, extrapolation_minus_read_p, extrapolation_plus_read_p, normal_derivative_of_solution_minus_read_p, normal_derivative_of_solution_plus_read_p);
      // start updates
      ierr = VecGhostUpdateEnd(tmp_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(tmp_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    // restore pointers pointers
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecRestoreArray(tmp_minus[dim], &tmp_minus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_plus[dim], &tmp_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extrapolation_minus[dim], &extrapolation_minus_read_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extrapolation_plus[dim], &extrapolation_plus_read_p[dim]); CHKERRXX(ierr);
    }

    // swap (n) and (n + 1) pseudo time iterates (more efficient to swap pointers than copying large chunks of data)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      std::swap(tmp_minus[dim], extrapolation_minus[dim]);
      std::swap(tmp_plus[dim], extrapolation_plus[dim]);
    }
  }

//  {
//    Vec normal_derivative_of_solution_minus_on_cells, normal_derivative_of_solution_plus_on_cells;
//    double *normal_derivative_of_solution_minus_on_cells_p, *normal_derivative_of_solution_plus_on_cells_p;
//    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &normal_derivative_of_solution_minus_on_cells); CHKERRXX(ierr);
//    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &normal_derivative_of_solution_plus_on_cells); CHKERRXX(ierr);
//    ierr = VecGetArray(normal_derivative_of_solution_minus_on_cells, &normal_derivative_of_solution_minus_on_cells_p); CHKERRXX(ierr);
//    ierr = VecGetArray(normal_derivative_of_solution_plus_on_cells, &normal_derivative_of_solution_plus_on_cells_p); CHKERRXX(ierr);


//    for (p4est_locidx_t quad_idx = 0; quad_idx < p4est->local_num_quadrants; ++quad_idx) {
//      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//        normal_derivative_of_solution_minus_on_cells_p[P4EST_DIM*quad_idx + dim] = normal_derivative_of_solution_minus_read_p[dim][faces->q2f(quad_idx, 2*dim)];
//        normal_derivative_of_solution_plus_on_cells_p[P4EST_DIM*quad_idx + dim] = normal_derivative_of_solution_plus_read_p[dim][faces->q2f(quad_idx, 2*dim)];
//      }
//    }
//    ierr = VecGhostUpdateBegin(normal_derivative_of_solution_minus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(normal_derivative_of_solution_plus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(normal_derivative_of_solution_minus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(normal_derivative_of_solution_plus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    std::vector<Vec_for_vtk_export_t> to_export;
//    to_export.push_back(Vec_for_vtk_export_t(normal_derivative_of_solution_minus_on_cells, "normal_derivative_minus"));
//    to_export.push_back(Vec_for_vtk_export_t(normal_derivative_of_solution_plus_on_cells, "normal_derivative_plus"));

//    my_p4est_vtk_write_all_general_lists(p4est, nodes, ghost, P4EST_FALSE, P4EST_FALSE, "/home/regan/workspace/projects/poisson_jump_faces/circle_shape_2D/vtu/standard/normal_derivatives", NULL, NULL, NULL, &to_export);
//    to_export.clear();

//    ierr = VecRestoreArray(normal_derivative_of_solution_minus_on_cells, &normal_derivative_of_solution_minus_on_cells_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(normal_derivative_of_solution_plus_on_cells, &normal_derivative_of_solution_plus_on_cells_p); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(normal_derivative_of_solution_minus_on_cells); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(normal_derivative_of_solution_plus_on_cells); CHKERRXX(ierr);
//  }

  // restore data pointers and delete locally created vectors
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    if(degree >= 1)
    {
      ierr = VecRestoreArrayRead(normal_derivative_of_solution_minus[dim], &normal_derivative_of_solution_minus_read_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(normal_derivative_of_solution_plus[dim], &normal_derivative_of_solution_plus_read_p[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(normal_derivative_of_solution_minus[dim]);  CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(normal_derivative_of_solution_plus[dim]); CHKERRXX(ierr);
    }
    ierr = delete_and_nullify_vector(tmp_minus[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(tmp_plus[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
  }

  extrapolations_are_set = true;
  return;
}

void my_p4est_poisson_jump_faces_t::extrapolate_normal_derivatives_local(const u_char& dim, const p4est_locidx_t& face_idx,
                                                                         double* tmp_minus_p[P4EST_DIM], double* tmp_plus_p[P4EST_DIM],
                                                                         const double* normal_derivative_of_solution_minus_p[P4EST_DIM], const double* normal_derivative_of_solution_plus_p[P4EST_DIM]) const
{
  tmp_minus_p[dim][face_idx] = normal_derivative_of_solution_minus_p[dim][face_idx];
  P4EST_ASSERT(extrapolation_operators_are_stored_and_set[dim]);
  std::map<p4est_locidx_t, extrapolation_operator_t>::const_iterator it = extrapolation_operator_minus[dim].find(face_idx);
#ifdef P4EST_DEBUG
  bool found_one = false;
#endif
  if(it != extrapolation_operator_minus[dim].end())
  {
    tmp_minus_p[dim][face_idx] -= it->second.n_dot_grad(normal_derivative_of_solution_minus_p[dim])*it->second.dtau;
#ifdef P4EST_DEBUG
    found_one = true;
#endif
  }

  tmp_plus_p[dim][face_idx] = normal_derivative_of_solution_plus_p[dim][face_idx];
  it = extrapolation_operator_plus[dim].find(face_idx);
  if(it != extrapolation_operator_plus[dim].end())
  {
    tmp_plus_p[dim][face_idx] -= it->second.n_dot_grad(normal_derivative_of_solution_plus_p[dim])*it->second.dtau;
#ifdef P4EST_DEBUG
    found_one = true;
#endif
  }
  P4EST_ASSERT(found_one);
  return;
}

void my_p4est_poisson_jump_faces_t::print_partition_VTK(const char *file, const u_char &dir)
{
  if(voronoi_on_the_fly)
    throw std::invalid_argument("my_p4est_poisson_jump_faces_t::print_partition_VTK: can't operate with voronoi_on_the_fly.");
  else
  {
#ifdef P4_TO_P8
    bool periodic[] = {0, 0, 0};
    Voronoi3D::print_VTK_format(voronoi_cell[dir], file, xyz_min, xyz_max, periodic);
#else
    Voronoi2D::print_VTK_format(voronoi_cell[dir], file);
#endif
  }
}
