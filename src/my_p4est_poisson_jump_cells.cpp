#include "my_p4est_poisson_jump_cells.h"

my_p4est_poisson_jump_cells_t::my_p4est_poisson_jump_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t* nodes_)
  : cell_ngbd(ngbd_c), p4est(ngbd_c->get_p4est()), ghost(ngbd_c->get_ghost()), nodes(nodes_),
    xyz_min(ngbd_c->get_p4est()->connectivity->vertices + 3*ngbd_c->get_p4est()->connectivity->tree_to_vertex[0]),
    xyz_max(ngbd_c->get_p4est()->connectivity->vertices + 3*ngbd_c->get_p4est()->connectivity->tree_to_vertex[P4EST_CHILDREN*(ngbd_c->get_p4est()->trees->elem_count - 1) + P4EST_CHILDREN - 1]),
    tree_dimensions(ngbd_c->get_tree_dimensions()),
    periodicity(ngbd_c->get_hierarchy()->get_periodicity())
{
  // set up the KSP solver
  PetscErrorCode ierr;
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  mu_minus = mu_plus = -1.0;
  add_diag_minus = add_diag_plus = 0.0;
  user_rhs_minus = user_rhs_plus =  jump_u = jump_normal_flux_u = NULL;
  rhs = NULL;
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
  P4EST_ASSERT(jump_u_ != NULL && jump_normal_flux_u_ != NULL);
  if(!interface_is_set())
    throw std::runtime_error("my_p4est_poisson_jump_cells_t::set_jumps(): the interface manager must be set before the jumps");
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
#ifdef P4EST_DEBUG
  P4EST_ASSERT(VecIsSetForNodes(jump_u_,              interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, 1));
  P4EST_ASSERT(VecIsSetForNodes(jump_normal_flux_u_,  interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, 1));
#endif

  jump_u              = jump_u_;
  jump_normal_flux_u  = jump_normal_flux_u_;

  if(interp_jump_u == NULL)
    interp_jump_u = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
  interp_jump_u->set_input(jump_u, linear);

  if(interp_jump_normal_flux == NULL)
    interp_jump_normal_flux = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
  interp_jump_normal_flux->set_input(jump_normal_flux_u_, linear);

  rhs_is_set = false;
  return;
}

KSPConvergedReason my_p4est_poisson_jump_cells_t::solve_linear_system()
{
  PetscErrorCode ierr;

  if(solution == NULL) {
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); }

  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  // we need update ghost values of the solution for accurate calculation of the extended interface values
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  KSPConvergedReason convergence_reason;
  ierr = KSPGetConvergedReason(ksp, &convergence_reason); CHKERRXX(ierr);

  return convergence_reason; // positive values indicate convergence
}

PetscErrorCode my_p4est_poisson_jump_cells_t::setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type, const double &tolerance_on_rel_residual) const
{
  PetscErrorCode ierr;
  ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER); CHKERRQ(ierr);
  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRQ(ierr);
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
    /* 1- Strong threshold:
     * Between 0 to 1
     * "0 "gives better convergence rate (in 3D).
     * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
    */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRQ(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRQ(ierr);

    /* 3- Trancation factor
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

