#include "petsc_logging.h"
#include "petsc_compatibility.h"
#include <p4est.h>

/* define proper logging events
 * Note that these are global variables to ensure all subsequent calls to the
 * same functions are logged properly
 */

PetscLogEvent log_my_p4est_poisson_nodes_voronoi_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_nodes_voronoi_matrix_setup;
PetscLogEvent log_my_p4est_poisson_nodes_voronoi_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_nodes_voronoi_KSPSolve;
PetscLogEvent log_my_p4est_poisson_nodes_voronoi_solve;

// PoissonSolverNodeBaseJump
PetscLogEvent log_PoissonSolverNodeBasedJump_matrix_preallocation;
PetscLogEvent log_PoissonSolverNodeBasedJump_setup_linear_system;
PetscLogEvent log_PoissonSolverNodeBasedJump_rhsvec_setup;
PetscLogEvent log_PoissonSolverNodeBasedJump_KSPSolve;
PetscLogEvent log_PoissonSolverNodeBasedJump_solve;
PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_points;
PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_cell;
PetscLogEvent log_PoissonSolverNodeBasedJump_interpolate_to_tree;

PetscLogEvent log_PoissonSolverNodeBasedJumpExtended_matrix_preallocation;
PetscLogEvent log_PoissonSolverNodeBasedJumpExtended_setup_linear_system;
PetscLogEvent log_PoissonSolverNodeBasedJumpExtended_rhsvec_setup;
PetscLogEvent log_PoissonSolverNodeBasedJumpExtended_KSPSolve;
PetscLogEvent log_PoissonSolverNodeBasedJumpExtended_solve;

// my_p4est_poisson_nodes_t
PetscLogEvent log_my_p4est_poisson_nodes_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_nodes_matrix_setup;
PetscLogEvent log_my_p4est_poisson_nodes_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_nodes_jump_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_nodes_solve;
PetscLogEvent log_my_p4est_poisson_nodes_KSPSolve;

// my_p4est_poisson_nodes_mls_t
PetscLogEvent log_my_p4est_poisson_nodes_mls_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_nodes_mls_setup_linear_system;
PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs;
PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix;
PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_rhs;
PetscLogEvent log_my_p4est_poisson_nodes_mls_preassemble_linear_system;
PetscLogEvent log_my_p4est_poisson_nodes_mls_KSPSolve;
PetscLogEvent log_my_p4est_poisson_nodes_mls_solve;
PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes;
PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections;
PetscLogEvent log_my_p4est_poisson_nodes_mls_determine_node_types;
PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize;
PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submatrix;
PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_rhs_jump;
PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_submat_main_jump;
PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_matrix;
PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_main;
PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_jump;
PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_robin;
PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_diag;
PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal;
PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal;
PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling;

// my_p4est_poisson_nodes_mls_sc_t
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_matrix_setup;
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_solve;
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_KSPSolve;
PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_compute_volumes;

// my_p4est_poisson_cells_t
PetscLogEvent log_my_p4est_poisson_cells_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_cells_matrix_setup;
PetscLogEvent log_my_p4est_poisson_cells_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_cells_solve;
PetscLogEvent log_my_p4est_poisson_cells_KSPSolve;

// my_p4est_poisson_jump_cells_t
PetscLogEvent log_my_p4est_poisson_jump_cells_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_jump_cells_setup_linear_system;
PetscLogEvent log_my_p4est_poisson_jump_cells_KSPSolve;

// my_p4est_xgfm_cells_t
PetscLogEvent log_my_p4est_xgfm_cells_solve_for_sharp_solution;
PetscLogEvent log_my_p4est_xgfm_cells_extend_interface_values;
PetscLogEvent log_my_p4est_xgfm_cells_interpolate_cell_extension_to_interface_capturing_nodes;
PetscLogEvent log_my_p4est_xgfm_cells_update_rhs_and_residual;

// my_p4est_poisson_faces_t
PetscLogEvent log_my_p4est_poisson_faces_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_faces_setup_linear_system;
PetscLogEvent log_my_p4est_poisson_faces_solve;
PetscLogEvent log_my_p4est_poisson_faces_KSPSolve;

// my_p4est_navier_stokes_t
PetscLogEvent log_my_p4est_navier_stokes_viscosity;
PetscLogEvent log_my_p4est_navier_stokes_projection;
PetscLogEvent log_my_p4est_navier_stokes_update;

// my_p4est_interpolation_t
PetscLogEvent log_my_p4est_interpolation_interpolate;
PetscLogEvent log_my_p4est_interpolation_process_local;
PetscLogEvent log_my_p4est_interpolation_process_queries;
PetscLogEvent log_my_p4est_interpolation_process_replies;
PetscLogEvent log_my_p4est_interpolation_all_reduce;

// my_p4est_semi_lagrangian_t
PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2;
PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order;
PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order;
PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_CF2;
PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_1st_order;
PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_2nd_order;
PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_multiple_phi;
PetscLogEvent log_my_p4est_semi_lagrangian_grid_gen_iter[P4EST_MAXLEVEL];

// my_p4est_trajectory_of_point
PetscLogEvent log_trajectory_from_np1_to_n;
PetscLogEvent log_trajectory_from_np1_to_nm1;
PetscLogEvent log_trajectory_from_np1_to_nm1_faces;

// my_p4est_level_set
PetscLogEvent log_my_p4est_level_set_reinit_1_iter_1st_order;
PetscLogEvent log_my_p4est_level_set_reinit_1_iter_2nd_order;
PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
PetscLogEvent log_my_p4est_level_set_reinit_2nd_order;
PetscLogEvent log_my_p4est_level_set_reinit_1st_time_2nd_space;
PetscLogEvent log_my_p4est_level_set_reinit_2nd_time_1st_space;
PetscLogEvent log_my_p4est_level_set_geometric_extrapolation_over_interface;
PetscLogEvent log_my_p4est_level_set_extend_over_interface_TVD;
PetscLogEvent log_my_p4est_level_set_extend_from_interface;
PetscLogEvent log_my_p4est_level_set_extend_from_interface_TVD;
PetscLogEvent log_my_p4est_level_set_compute_derivatives;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_1_iter;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_Vec;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_CF2;

// my_p4est_level_set_faces_t
PetscLogEvent log_my_p4est_level_set_faces_geometric_extrapolation_over_interface;

// my_p4est_level_set_cells_t
PetscLogEvent log_my_p4est_level_set_cells_geometric_extrapolation_over_interface;

// my_p4est_hierarchy_t
PetscLogEvent log_my_p4est_hierarchy_t;
PetscLogEvent log_my_p4est_hierarchy_t_find_smallest_quad;

// quad_neighbor_nodes_of_node_t
PetscLogEvent log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quad_interp;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dz_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_m00;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_p00;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0m0;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0p0;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central_00m;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central_00p;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_gradient;
PetscLogEvent log_quad_neighbor_nodes_of_node_t_laplace;

// my_p4est_node_neighbors_t
PetscLogEvent log_my_p4est_node_neighbors_t;
PetscLogEvent log_my_p4est_node_neighbors_t_dxx_central;
PetscLogEvent log_my_p4est_node_neighbors_t_dyy_central;
PetscLogEvent log_my_p4est_node_neighbors_t_dzz_central;
PetscLogEvent log_my_p4est_node_neighbors_t_1st_derivatives_central;
PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central;
PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_block;

// my_p4est_faces_t
PetscLogEvent log_my_p4est_faces_t;
PetscLogEvent log_my_p4est_faces_notify_t;
PetscLogEvent log_my_p4est_faces_compute_voronoi_cell_t;

// functions
PetscLogEvent log_my_p4est_vtk_write_all;
PetscLogEvent log_my_p4est_nodes_new;
PetscLogEvent log_my_p4est_new;
PetscLogEvent log_my_p4est_ghost_new;
PetscLogEvent log_my_p4est_copy;
PetscLogEvent log_my_p4est_ghost_expand;
PetscLogEvent log_my_p4est_refine;
PetscLogEvent log_my_p4est_coarsen;
PetscLogEvent log_my_p4est_partition;
PetscLogEvent log_my_p4est_balance;
PetscLogEvent log_my_sc_notify;
PetscLogEvent log_my_sc_notify_allgather;

// poisson multialloy
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_initialize_solvers;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_c0;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_c1;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_t;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_c0;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_c1;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_t;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_compute_c0n;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n;
PetscLogEvent log_my_p4est_poisson_nodes_multialloy_adjust_c0;

// multialloy
PetscLogEvent log_my_p4est_multialloy_one_step;
PetscLogEvent log_my_p4est_multialloy_compute_dt;
PetscLogEvent log_my_p4est_multialloy_compute_geometric_properties;
PetscLogEvent log_my_p4est_multialloy_compute_velocity;
PetscLogEvent log_my_p4est_multialloy_compute_solid;
PetscLogEvent log_my_p4est_multialloy_update_grid;
PetscLogEvent log_my_p4est_multialloy_update_grid_history;
PetscLogEvent log_my_p4est_multialloy_update_grid_transfer_data;
PetscLogEvent log_my_p4est_multialloy_update_grid_regularize_front;
PetscLogEvent log_my_p4est_multialloy_save_vtk;

// biofilm
PetscLogEvent log_my_p4est_biofilm_one_step;
PetscLogEvent log_my_p4est_biofilm_compute_dt;
PetscLogEvent log_my_p4est_biofilm_compute_geometric_properties;
PetscLogEvent log_my_p4est_biofilm_compute_velocity;
PetscLogEvent log_my_p4est_biofilm_solve_concentration;
PetscLogEvent log_my_p4est_biofilm_solve_pressure;
PetscLogEvent log_my_p4est_biofilm_update_grid;
PetscLogEvent log_my_p4est_biofilm_save_vtk;


void register_petsc_logs()
{
  PetscErrorCode ierr;

  // PoissonSolverNodeBaseJump
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::matrix_preallocation        ", 0, &log_PoissonSolverNodeBasedJump_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::setup_linear_system         ", 0, &log_PoissonSolverNodeBasedJump_setup_linear_system); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::rhsvec_setup                ", 0, &log_PoissonSolverNodeBasedJump_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::solve                       ", 0, &log_PoissonSolverNodeBasedJump_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::compute_voronoi_points      ", 0, &log_PoissonSolverNodeBasedJump_compute_voronoi_points); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::compute_voronoi_cell        ", 0, &log_PoissonSolverNodeBasedJump_compute_voronoi_cell); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJump::interpolate_to_tree         ", 0, &log_PoissonSolverNodeBasedJump_interpolate_to_tree); CHKERRXX(ierr);

  // PoissonSolverNodeBaseJumpExtended
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJumpExt::matrix_preallocation     ", 0, &log_PoissonSolverNodeBasedJumpExtended_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJumpExt::setup_linear_system      ", 0, &log_PoissonSolverNodeBasedJumpExtended_setup_linear_system); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJumpExt::rhsvec_setup             ", 0, &log_PoissonSolverNodeBasedJumpExtended_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("PoissonSolverNodeBasedJumpExt::solve                    ", 0, &log_PoissonSolverNodeBasedJumpExtended_solve); CHKERRXX(ierr);

  // my_p4est_poisson_nodes_t
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::matrix_preallocation            ", 0, &log_my_p4est_poisson_nodes_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::matrix_setup                    ", 0, &log_my_p4est_poisson_nodes_matrix_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::rhsvec_setup                    ", 0, &log_my_p4est_poisson_nodes_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::jump_rhsvec_setup               ", 0, &log_my_p4est_poisson_nodes_jump_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::solve                           ", 0, &log_my_p4est_poisson_nodes_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::KSPSolve                        ", 0, &log_my_p4est_poisson_nodes_KSPSolve); CHKERRXX(ierr);

  // my_p4est_poisson_nodes_t
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::matrix_preallocation     ", 0, &log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::matrix_setup             ", 0, &log_my_p4est_poisson_nodes_mls_sc_matrix_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::rhsvec_setup             ", 0, &log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::solve                    ", 0, &log_my_p4est_poisson_nodes_mls_sc_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::KSPSolve                 ", 0, &log_my_p4est_poisson_nodes_mls_sc_KSPSolve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls_sc::compute_volumes          ", 0, &log_my_p4est_poisson_nodes_mls_sc_compute_volumes); CHKERRXX(ierr);

  // my_p4est_poisson_cells_t
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::matrix_preallocation            ", 0, &log_my_p4est_poisson_cells_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::matrix_setup                    ", 0, &log_my_p4est_poisson_cells_matrix_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::rhsvec_setup                    ", 0, &log_my_p4est_poisson_cells_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::solve                           ", 0, &log_my_p4est_poisson_cells_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::KSPSolve                        ", 0, &log_my_p4est_poisson_cells_KSPSolve); CHKERRXX(ierr);

  // my_p4est_poisson_jump_cells_t
  ierr = PetscLogEventRegister("my_p4est_poisson_jump_cells::matrix_preallocation       ", 0, &log_my_p4est_poisson_jump_cells_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_jump_cells::setup_linear_system        ", 0, &log_my_p4est_poisson_jump_cells_setup_linear_system); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_jump_cells::KSPSolve                   ", 0, &log_my_p4est_poisson_jump_cells_KSPSolve); CHKERRXX(ierr);

  // my_p4est_xgfm_cells_t
  ierr = PetscLogEventRegister("my_p4est_xgfm_cells::extend_interface_values            ", 0, &log_my_p4est_xgfm_cells_extend_interface_values); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_xgfm_cells::interpolate_cell_extension_to_interface_capturing_nodes",
                                                                                           0, &log_my_p4est_xgfm_cells_interpolate_cell_extension_to_interface_capturing_nodes); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_xgfm_cells::update_rhs_and_residual            ", 0, &log_my_p4est_xgfm_cells_update_rhs_and_residual); CHKERRXX(ierr);

  // my_p4est_poisson_faces_t
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::matrix_preallocation            ", 0, &log_my_p4est_poisson_faces_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::setup_linear_system             ", 0, &log_my_p4est_poisson_faces_setup_linear_system); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::solve                           ", 0, &log_my_p4est_poisson_faces_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::KSPSolve                        ", 0, &log_my_p4est_poisson_faces_KSPSolve); CHKERRXX(ierr);

  // my_p4est_navier_stokes_t
  ierr = PetscLogEventRegister("my_p4est_navier_stokes::viscosity                       ", 0, &log_my_p4est_navier_stokes_viscosity); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_navier_stokes::projection                      ", 0, &log_my_p4est_navier_stokes_projection); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_navier_stokes::update                          ", 0, &log_my_p4est_navier_stokes_update); CHKERRXX(ierr);

  // my_p4est_interpolation
  ierr = PetscLogEventRegister("my_p4est_interpolation::interpolate                     ", 0, &log_my_p4est_interpolation_interpolate); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_local                   ", 0, &log_my_p4est_interpolation_process_local); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_queries                 ", 0, &log_my_p4est_interpolation_process_queries); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_replies                 ", 0, &log_my_p4est_interpolation_process_replies); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::all_reduce                      ", 0, &log_my_p4est_interpolation_all_reduce); CHKERRXX(ierr);

  // my_p4est_semi_lagrangian_t
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::advect_from_n_to_np1_CF2    ", 0, &log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::advect_from_n_to_np1_1st    ", 0, &log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::advect_from_n_to_np1_2nd    ", 0, &log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::update_p4est_CF2            ", 0, &log_my_p4est_semi_lagrangian_update_p4est_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::update_p4est_1st_order      ", 0, &log_my_p4est_semi_lagrangian_update_p4est_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::update_p4est_2nd_order      ", 0, &log_my_p4est_semi_lagrangian_update_p4est_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_semi_lagrangian_t::update_p4est_multiple_phi   ", 0, &log_my_p4est_semi_lagrangian_update_p4est_multiple_phi); CHKERRXX(ierr);
  

  // my_p4est_trajectory_of_point
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_n                            ", 0, &log_trajectory_from_np1_to_n); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_nm1                          ", 0, &log_trajectory_from_np1_to_nm1); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_nm1_faces                    ", 0, &log_trajectory_from_np1_to_nm1_faces); CHKERRXX(ierr);

	for (short i = 0; i < P4EST_MAXLEVEL; i++) {
		char logname [128]; 
    sprintf(logname,"my_p4est_semi_lagrangian_t::grid_gen_iter_%02d            ", i);
    ierr = PetscLogEventRegister(logname, 0, &log_my_p4est_semi_lagrangian_grid_gen_iter[i]); CHKERRXX(ierr);
	}
  // my_p4est_level_set
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1st_order                    ", 0, &log_my_p4est_level_set_reinit_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_2nd_order                    ", 0, &log_my_p4est_level_set_reinit_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1st_time_2nd_space           ", 0, &log_my_p4est_level_set_reinit_1st_time_2nd_space); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_2nd_time_1st_space           ", 0, &log_my_p4est_level_set_reinit_2nd_time_1st_space); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1_iter_1st_order             ", 0, &log_my_p4est_level_set_reinit_1_iter_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1_iter_2nd_order             ", 0, &log_my_p4est_level_set_reinit_1_iter_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::log_my_p4est_level_set_geometric_extrapolation_over_interface",
                                                                                           0, &log_my_p4est_level_set_geometric_extrapolation_over_interface); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_over_interface_TVD           ", 0, &log_my_p4est_level_set_extend_over_interface_TVD); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_from_interface               ", 0, &log_my_p4est_level_set_extend_from_interface); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_from_interface_TVD           ", 0, &log_my_p4est_level_set_extend_from_interface_TVD); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::compute_derivatives                 ", 0, &log_my_p4est_level_set_compute_derivatives); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_1_iter   ", 0, &log_my_p4est_level_set_advect_in_normal_direction_1_iter); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_CF2      ", 0, &log_my_p4est_level_set_advect_in_normal_direction_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_Vec      ", 0, &log_my_p4est_level_set_advect_in_normal_direction_Vec); CHKERRXX(ierr);

  // my_p4est_level_set_faces_t
  ierr = PetscLogEventRegister("my_p4est_level_set_faces::geometric_extrapolation_over_interface", 0, &log_my_p4est_level_set_faces_geometric_extrapolation_over_interface); CHKERRXX(ierr);

  // my_p4est_level_set_cells_t
  ierr = PetscLogEventRegister("my_p4est_level_set_cells::geometric_extrapolation_over_interface", 0, &log_my_p4est_level_set_cells_geometric_extrapolation_over_interface);  CHKERRXX(ierr);

  // my_p4est_hierarchy_t
  ierr = PetscLogEventRegister("my_p4est_hierarchy_t::init                              ", 0, &log_my_p4est_hierarchy_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_hierarchy_t::find_smallest_quad                ", 0, &log_my_p4est_hierarchy_t_find_smallest_quad); CHKERRXX(ierr);

  // my_p4est_faces_t
  ierr = PetscLogEventRegister("my_p4est_faces_t::init                                  ", 0, &log_my_p4est_faces_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_faces_t::notify                                ", 0, &log_my_p4est_faces_notify_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_faces_t::compute_voronoi_cell                  ", 0, &log_my_p4est_faces_compute_voronoi_cell_t); CHKERRXX(ierr);


  // quad_neighbor_nodes_of_node_t
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::ngbd_with_quad_interp    ", 0, &log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::x_ngbd_with_quad_interp  ", 0, &log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::y_ngbd_with_quad_interp  ", 0, &log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::z_ngbd_with_quad_interp  ", 0, &log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quad_interp); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dx_central               ", 0, &log_quad_neighbor_nodes_of_node_t_dx_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dy_central               ", 0, &log_quad_neighbor_nodes_of_node_t_dy_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dz_central               ", 0, &log_quad_neighbor_nodes_of_node_t_dz_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dxx_central              ", 0, &log_quad_neighbor_nodes_of_node_t_dxx_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dxx_central_m00          ", 0, &log_quad_neighbor_nodes_of_node_t_dxx_central_m00); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dxx_central_p00          ", 0, &log_quad_neighbor_nodes_of_node_t_dxx_central_p00); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dyy_central              ", 0, &log_quad_neighbor_nodes_of_node_t_dyy_central); CHKERRXX(ierr);  
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dyy_central_0m0          ", 0, &log_quad_neighbor_nodes_of_node_t_dyy_central_0m0); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dyy_central_0p0          ", 0, &log_quad_neighbor_nodes_of_node_t_dyy_central_0p0); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dzz_central              ", 0, &log_quad_neighbor_nodes_of_node_t_dzz_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dzz_central_00m          ", 0, &log_quad_neighbor_nodes_of_node_t_dzz_central_00m); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::dzz_central_00p          ", 0, &log_quad_neighbor_nodes_of_node_t_dzz_central_00p); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::gradient                 ", 0, &log_quad_neighbor_nodes_of_node_t_gradient); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("quad_neighbor_nodes_of_node_t::laplace                  ", 0, &log_quad_neighbor_nodes_of_node_t_laplace); CHKERRXX(ierr);

  // my_p4est_node_neighbors_t
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::init                         ", 0, &log_my_p4est_node_neighbors_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dxx_central                  ", 0, &log_my_p4est_node_neighbors_t_dxx_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dyy_central                  ", 0, &log_my_p4est_node_neighbors_t_dyy_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dzz_central                  ", 0, &log_my_p4est_node_neighbors_t_dzz_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::1st_derivatives_cent         ", 0, &log_my_p4est_node_neighbors_t_1st_derivatives_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::2nd_derivatives_cent         ", 0, &log_my_p4est_node_neighbors_t_2nd_derivatives_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::2nd_derivatives_cent_block   ", 0, &log_my_p4est_node_neighbors_t_2nd_derivatives_central_block); CHKERRXX(ierr);

  // functions
  ierr = PetscLogEventRegister("my_p4est_vtk_write_all                                  ", 0, &log_my_p4est_vtk_write_all); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_nodes_new                                      ", 0, &log_my_p4est_nodes_new); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_new                                            ", 0, &log_my_p4est_new); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_ghost_new                                      ", 0, &log_my_p4est_ghost_new); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_copy                                           ", 0, &log_my_p4est_copy); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_ghost_expand                                   ", 0, &log_my_p4est_ghost_expand); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_refine                                         ", 0, &log_my_p4est_refine); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_coarsen                                        ", 0, &log_my_p4est_coarsen); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_partition                                      ", 0, &log_my_p4est_partition); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_balance                                        ", 0, &log_my_p4est_balance); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_sc_notify                                            ", 0, &log_my_sc_notify); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_sc_notify_allgather                                  ", 0, &log_my_sc_notify_allgather); CHKERRXX(ierr);

  // poisson nodes mls
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::matrix_preallocation              ", 0, &log_my_p4est_poisson_nodes_mls_matrix_preallocation             ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::setup_linear_system               ", 0, &log_my_p4est_poisson_nodes_mls_setup_linear_system              ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::discretize_matrix_and_rhs         ", 0, &log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs        ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::discretize_matrix                 ", 0, &log_my_p4est_poisson_nodes_mls_discretize_matrix                ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::discretize_rhs                    ", 0, &log_my_p4est_poisson_nodes_mls_discretize_rhs                   ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::preassemble_linear_system         ", 0, &log_my_p4est_poisson_nodes_mls_preassemble_linear_system        ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::KSPSolve                          ", 0, &log_my_p4est_poisson_nodes_mls_KSPSolve                         ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::solve                             ", 0, &log_my_p4est_poisson_nodes_mls_solve                            ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::compute_finite_volumes            ", 0, &log_my_p4est_poisson_nodes_mls_compute_finite_volumes           ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::compute_finite_volumes_connections", 0, &log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::determine_node_types              ", 0, &log_my_p4est_poisson_nodes_mls_determine_node_types             ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::discretize                        ", 0, &log_my_p4est_poisson_nodes_mls_discretize                       ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::assemble_submatrix                ", 0, &log_my_p4est_poisson_nodes_mls_assemble_submatrix               ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::correct_rhs_jump                  ", 0, &log_my_p4est_poisson_nodes_mls_correct_rhs_jump                 ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::correct_submat_main_jump          ", 0, &log_my_p4est_poisson_nodes_mls_correct_submat_main_jump         ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::assemble_matrix                   ", 0, &log_my_p4est_poisson_nodes_mls_assemble_matrix                  ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::assemble_submat_main              ", 0, &log_my_p4est_poisson_nodes_mls_assemble_submat_main             ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::assemble_submat_jump              ", 0, &log_my_p4est_poisson_nodes_mls_assemble_submat_jump             ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::add_submat_robin                  ", 0, &log_my_p4est_poisson_nodes_mls_add_submat_robin                 ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::add_submat_diag                   ", 0, &log_my_p4est_poisson_nodes_mls_add_submat_diag                  ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::scale_matrix_by_diagonal          ", 0, &log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal         ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::scale_rhs_by_diagonal             ", 0, &log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal            ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_mls::compute_diagonal_scaling          ", 0, &log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling         ); CHKERRXX(ierr);

  // multialloy
  ierr = PetscLogEventRegister("my_p4est_multialloy::one_step                     ", 0, &log_my_p4est_multialloy_one_step                    ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::compute_dt                   ", 0, &log_my_p4est_multialloy_compute_dt                  ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::compute_geometric_properties ", 0, &log_my_p4est_multialloy_compute_geometric_properties); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::compute_velocity             ", 0, &log_my_p4est_multialloy_compute_velocity            ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::compute_solid                ", 0, &log_my_p4est_multialloy_compute_solid               ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::update_grid                  ", 0, &log_my_p4est_multialloy_update_grid                 ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::update_grid::history         ", 0, &log_my_p4est_multialloy_update_grid_history         ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::update_grid::transfer_data   ", 0, &log_my_p4est_multialloy_update_grid_transfer_data   ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::update_grid::regularize_front", 0, &log_my_p4est_multialloy_update_grid_regularize_front); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_multialloy::save_vtk                     ", 0, &log_my_p4est_multialloy_save_vtk                    ); CHKERRXX(ierr);

  // poisson multialloy
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::initialize_solvers", 0, &log_my_p4est_poisson_nodes_multialloy_initialize_solvers); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve             ", 0, &log_my_p4est_poisson_nodes_multialloy_solve             ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_c0          ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_c0          ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_c1          ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_c1          ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_t           ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_t           ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_psi_c0      ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_psi_c0      ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_psi_c1      ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_psi_c1      ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::solve_psi_t       ", 0, &log_my_p4est_poisson_nodes_multialloy_solve_psi_t       ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::compute_c0n       ", 0, &log_my_p4est_poisson_nodes_multialloy_compute_c0n       ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::compute_psi_c0n   ", 0, &log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n   ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes_multialloy::adjust_c0         ", 0, &log_my_p4est_poisson_nodes_multialloy_adjust_c0         ); CHKERRXX(ierr);

  // biofilm
  ierr = PetscLogEventRegister("my_p4est_biofilm::one_step                    ", 0, &log_my_p4est_biofilm_one_step                    ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::compute_dt                  ", 0, &log_my_p4est_biofilm_compute_dt                  ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::compute_geometric_properties", 0, &log_my_p4est_biofilm_compute_geometric_properties); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::compute_velocity            ", 0, &log_my_p4est_biofilm_compute_velocity            ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::solve_concentration         ", 0, &log_my_p4est_biofilm_solve_concentration         ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::solve_pressure              ", 0, &log_my_p4est_biofilm_solve_pressure              ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::update_grid                 ", 0, &log_my_p4est_biofilm_update_grid                 ); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_biofilm::save_vtk                    ", 0, &log_my_p4est_biofilm_save_vtk                    ); CHKERRXX(ierr);
}
