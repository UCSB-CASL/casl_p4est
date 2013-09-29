#include "petsc_logging.h"

/* define proper logging events
 * Note that these are global variables to ensure all subsequent calls to the
 * same functions are logged properly
 */

// PoissonSolverNodeBase
PetscLogEvent log_PoissonSolverNodeBase_matrix_preallocation;
PetscLogEvent log_PoissonSolverNodeBase_matrix_setup;
PetscLogEvent log_PoissonSolverNodeBase_rhsvec_setup;
PetscLogEvent log_PoissonSolverNodeBase_solve;

// InterpolatingFunction
PetscLogEvent log_InterpolatingFunction_interpolate;

// SemiLagrangian
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_Vec;
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_CF2;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_Vec;

// my_p4est_level_set
PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
PetscLogEvent log_my_p4est_level_set_reinit_2nd_order;
PetscLogEvent log_my_p4est_level_set_extend_over_interface;
PetscLogEvent log_my_p4est_level_set_extend_from_interface;

// my_p4est_hierarchy_t
PetscLogEvent log_my_p4est_hierarchy_t;

// my_p4est_node_neighbors_t
PetscLogEvent log_my_p4est_node_neighbors_t;

// functions
PetscLogEvent log_my_p4est_vtk_write_all;
PetscLogEvent log_my_p4est_nodes_new;
PetscLogEvent log_my_p4est_new;
PetscLogEvent log_my_p4est_ghost_new;

void register_petsc_logs()
{
  PetscErrorCode ierr;

  // PoissonSolverNodeBase
  ierr = PetscLogEventRegister("PoissonSolverNodeBase::matrix_preallocation    ", 0, &log_PoissonSolverNodeBase_matrix_preallocation);   CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBase::matrix_setup            ", 0, &log_PoissonSolverNodeBase_matrix_setup);           CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBase::rhsvec_setup            ", 0, &log_PoissonSolverNodeBase_rhsvec_setup);           CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBase::solve                   ", 0, &log_PoissonSolverNodeBase_solve);                  CHKERRXX(ierr);

  // InterpolatingFunction
  ierr = PetscLogEventRegister("InterpolatingFunction::interpolate             ", 0, &log_InterpolatingFunction_interpolate);            CHKERRXX(ierr);

  // Semilagrangian
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_Vec       ", 0, &log_Semilagrangian_advect_from_n_to_np1_Vec);      CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_CF2       ", 0, &log_Semilagrangian_advect_from_n_to_np1_CF2);      CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_Vec  ", 0, &log_Semilagrangian_update_p4est_second_order_Vec); CHKERRXX(ierr);

  // my_p4est_level_set
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1st_order           ", 0, &log_my_p4est_level_set_reinit_1st_order);          CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_2nd_order           ", 0, &log_my_p4est_level_set_reinit_2nd_order);          CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_over_interface      ", 0, &log_my_p4est_level_set_extend_over_interface);     CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_from_interface      ", 0, &log_my_p4est_level_set_extend_from_interface);     CHKERRXX(ierr);

  // my_p4est_hierarchy_t
  ierr = PetscLogEventRegister("my_p4est_hierarchy_t                           ", 0, &log_my_p4est_hierarchy_t);                         CHKERRXX(ierr);

  // my_p4est_node_neighbors_t
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t                      ", 0, &log_my_p4est_node_neighbors_t);                    CHKERRXX(ierr);

  // functions
  ierr = PetscLogEventRegister("my_p4est_vtk_write_all                         ", 0, &log_my_p4est_vtk_write_all);                       CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_nodes_new                             ", 0, &log_my_p4est_nodes_new);                           CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_new                                   ", 0, &log_my_p4est_new);                                 CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_ghost_new                             ", 0, &log_my_p4est_ghost_new);                           CHKERRXX(ierr);
}
