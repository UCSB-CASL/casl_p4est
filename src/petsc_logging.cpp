#include "petsc_logging.h"
#include "petsc_compatibility.h"
#include <p4est.h>

/* define proper logging events
 * Note that these are global variables to ensure all subsequent calls to the
 * same functions are logged properly
 */

// PoissonSolverNodeBaseJump
PetscLogEvent log_PoissonSolverNodeBasedJump_matrix_preallocation;
PetscLogEvent log_PoissonSolverNodeBasedJump_setup_linear_system;
PetscLogEvent log_PoissonSolverNodeBasedJump_rhsvec_setup;
PetscLogEvent log_PoissonSolverNodeBasedJump_KSPSolve;
PetscLogEvent log_PoissonSolverNodeBasedJump_solve;
PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_points;
PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_cell;
PetscLogEvent log_PoissonSolverNodeBasedJump_interpolate_to_tree;

// my_p4est_poisson_nodes_t
PetscLogEvent log_my_p4est_poisson_nodes_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_nodes_matrix_setup;
PetscLogEvent log_my_p4est_poisson_nodes_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_nodes_solve;
PetscLogEvent log_my_p4est_poisson_nodes_KSPSolve;

// my_p4est_poisson_cells_t
PetscLogEvent log_my_p4est_poisson_cells_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_cells_matrix_setup;
PetscLogEvent log_my_p4est_poisson_cells_rhsvec_setup;
PetscLogEvent log_my_p4est_poisson_cells_solve;
PetscLogEvent log_my_p4est_poisson_cells_KSPSolve;

// my_p4est_poisson_faces_t
PetscLogEvent log_my_p4est_poisson_faces_compute_voronoi_cell;
PetscLogEvent log_my_p4est_poisson_faces_matrix_preallocation;
PetscLogEvent log_my_p4est_poisson_faces_setup_linear_system;
PetscLogEvent log_my_p4est_poisson_faces_solve;
PetscLogEvent log_my_p4est_poisson_faces_KSPSolve;

// my_p4est_interpolation_t
PetscLogEvent log_my_p4est_interpolation_interpolate;
PetscLogEvent log_my_p4est_interpolation_process_local;
PetscLogEvent log_my_p4est_interpolation_process_queries;
PetscLogEvent log_my_p4est_interpolation_process_replies;
PetscLogEvent log_my_p4est_interpolation_all_reduce;

// SemiLagrangian
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_Vec;
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_CF2;
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_CFL_Vec;
PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_CFL_CF2;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_Vec;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_CFL_Vec;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_CFL_CF2;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_CF2;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_CF2_grid;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_CF2_value;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_last_grid_CF2;
PetscLogEvent log_Semilagrangian_update_p4est_second_order_last_grid_Vec;
PetscLogEvent log_Semilagrangian_grid_gen_iter[P4EST_MAXLEVEL];

// my_p4est_trajectory_of_point
PetscLogEvent log_trajectory_from_np1_to_n;
PetscLogEvent log_trajectory_from_np1_to_nm1;
PetscLogEvent log_trajectory_from_np1_to_nm1_faces;

// my_p4est_level_set
PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
PetscLogEvent log_my_p4est_level_set_reinit_2nd_order;
PetscLogEvent log_my_p4est_level_set_reinit_1st_time_2nd_space;
PetscLogEvent log_my_p4est_level_set_reinit_2nd_time_1st_space;
PetscLogEvent log_my_p4est_level_set_reinit_1_iter_1st_order;
PetscLogEvent log_my_p4est_level_set_reinit_1_iter_2nd_order;
PetscLogEvent log_my_p4est_level_set_extend_over_interface;
PetscLogEvent log_my_p4est_level_set_extend_over_interface_TVD;
PetscLogEvent log_my_p4est_level_set_extend_from_interface;
PetscLogEvent log_my_p4est_level_set_extend_from_interface_TVD;
PetscLogEvent log_my_p4est_level_set_compute_derivatives;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_1_iter;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_Vec;
PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_CF2;

// my_p4est_level_set_faces_t
PetscLogEvent log_my_p4est_level_set_faces_extend_over_interface;

// my_p4est_level_set_cells_t
PetscLogEvent log_my_p4est_level_set_cells_extend_over_interface;

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

// my_p4est_node_neighbors_t
PetscLogEvent log_my_p4est_node_neighbors_t;
PetscLogEvent log_my_p4est_node_neighbors_t_dxx_central;
PetscLogEvent log_my_p4est_node_neighbors_t_dyy_central;
PetscLogEvent log_my_p4est_node_neighbors_t_dzz_central;
PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central;
PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_block;

// my_p4est_faces_t
PetscLogEvent log_my_p4est_faces_t;

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
PetscLogEvent log_my_sc_notify;
PetscLogEvent log_my_sc_notify_allgather;

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

  // my_p4est_poisson_nodes_t
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::matrix_preallocation            ", 0, &log_my_p4est_poisson_nodes_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::matrix_setup                    ", 0, &log_my_p4est_poisson_nodes_matrix_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::rhsvec_setup                    ", 0, &log_my_p4est_poisson_nodes_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_nodes::solve                           ", 0, &log_my_p4est_poisson_nodes_solve); CHKERRXX(ierr);

  // my_p4est_poisson_cells_t
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::matrix_preallocation            ", 0, &log_my_p4est_poisson_cells_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::matrix_setup                    ", 0, &log_my_p4est_poisson_cells_matrix_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::rhsvec_setup                    ", 0, &log_my_p4est_poisson_cells_rhsvec_setup); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::solve                           ", 0, &log_my_p4est_poisson_cells_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_cells::KSPSolve                        ", 0, &log_my_p4est_poisson_cells_KSPSolve); CHKERRXX(ierr);

  // my_p4est_poisson_faces_t
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::compute_voronoi_cell            ", 0, &log_my_p4est_poisson_faces_compute_voronoi_cell); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::matrix_preallocation            ", 0, &log_my_p4est_poisson_faces_matrix_preallocation); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::setup_linear_system             ", 0, &log_my_p4est_poisson_faces_setup_linear_system); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::solve                           ", 0, &log_my_p4est_poisson_faces_solve); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_poisson_faces::KSPSolve                        ", 0, &log_my_p4est_poisson_faces_KSPSolve); CHKERRXX(ierr);

  // my_p4est_interpolation
  ierr = PetscLogEventRegister("my_p4est_interpolation::interpolate                     ", 0, &log_my_p4est_interpolation_interpolate); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_local                   ", 0, &log_my_p4est_interpolation_process_local); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_queries                 ", 0, &log_my_p4est_interpolation_process_queries); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::process_replies                 ", 0, &log_my_p4est_interpolation_process_replies); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_interpolation::all_reduce                      ", 0, &log_my_p4est_interpolation_all_reduce); CHKERRXX(ierr);

  // Semilagrangian
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_Vec                ", 0, &log_Semilagrangian_advect_from_n_to_np1_Vec); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_CF2                ", 0, &log_Semilagrangian_advect_from_n_to_np1_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_CFL_Vec            ", 0, &log_Semilagrangian_advect_from_n_to_np1_CFL_Vec); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::advect_from_n_to_np1_CFL_CF2            ", 0, &log_Semilagrangian_advect_from_n_to_np1_CFL_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_Vec           ", 0, &log_Semilagrangian_update_p4est_second_order_Vec); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_CFL_Vec       ", 0, &log_Semilagrangian_update_p4est_second_order_CFL_Vec); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_CFL_CF2       ", 0, &log_Semilagrangian_update_p4est_second_order_CFL_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_CF2           ", 0, &log_Semilagrangian_update_p4est_second_order_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_CF2_value     ", 0, &log_Semilagrangian_update_p4est_second_order_CF2_value); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_CF2_grid      ", 0, &log_Semilagrangian_update_p4est_second_order_CF2_grid); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_last_grid_CF2 ", 0, &log_Semilagrangian_update_p4est_second_order_last_grid_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("Semilagrangian::update_p4est_second_order_last_grid_Vec ", 0, &log_Semilagrangian_update_p4est_second_order_last_grid_Vec); CHKERRXX(ierr);
  

  // my_p4est_trajectory_of_point
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_n                            ", 0, &log_trajectory_from_np1_to_n); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_nm1                          ", 0, &log_trajectory_from_np1_to_nm1); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("log_trajectory_from_np1_to_nm1_faces                    ", 0, &log_trajectory_from_np1_to_nm1_faces); CHKERRXX(ierr);

	for (short i = 0; i < P4EST_MAXLEVEL; i++) {
		char logname [128]; 
		sprintf(logname,"Semilagrangian::grid_gen_iter_%02d                        ", i);
		ierr = PetscLogEventRegister(logname, 0, &log_Semilagrangian_grid_gen_iter[i]); CHKERRXX(ierr);	
	}
  // my_p4est_level_set
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1st_order                    ", 0, &log_my_p4est_level_set_reinit_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_2nd_order                    ", 0, &log_my_p4est_level_set_reinit_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1st_time_2nd_space           ", 0, &log_my_p4est_level_set_reinit_1st_time_2nd_space); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_2nd_time_1st_space           ", 0, &log_my_p4est_level_set_reinit_2nd_time_1st_space); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1_iter_1st_order             ", 0, &log_my_p4est_level_set_reinit_1_iter_1st_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::reinit_1_iter_2nd_order             ", 0, &log_my_p4est_level_set_reinit_1_iter_2nd_order); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_over_interface               ", 0, &log_my_p4est_level_set_extend_over_interface); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_over_interface_TVD           ", 0, &log_my_p4est_level_set_extend_over_interface_TVD); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_from_interface               ", 0, &log_my_p4est_level_set_extend_from_interface); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::extend_from_interface_TVD           ", 0, &log_my_p4est_level_set_extend_from_interface_TVD); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_1_iter   ", 0, &log_my_p4est_level_set_advect_in_normal_direction_1_iter); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_CF2      ", 0, &log_my_p4est_level_set_advect_in_normal_direction_CF2); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_level_set::advect_in_normal_direction_Vec      ", 0, &log_my_p4est_level_set_advect_in_normal_direction_Vec); CHKERRXX(ierr);

  // my_p4est_level_set_faces_t
  ierr = PetscLogEventRegister("my_p4est_level_set_faces::extend_over_interface         ", 0, &log_my_p4est_level_set_faces_extend_over_interface); CHKERRXX(ierr);

  // my_p4est_level_set_cells_t
  ierr = PetscLogEventRegister("my_p4est_level_set_cells::extend_over_interface         ", 0, &log_my_p4est_level_set_cells_extend_over_interface); CHKERRXX(ierr);

  // my_p4est_hierarchy_t
  ierr = PetscLogEventRegister("my_p4est_hierarchy_t::init                              ", 0, &log_my_p4est_hierarchy_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_hierarchy_t::find_smallest_quad                ", 0, &log_my_p4est_hierarchy_t_find_smallest_quad); CHKERRXX(ierr);

  // my_p4est_faces_t
  ierr = PetscLogEventRegister("my_p4est_faces_t::init                                  ", 0, &log_my_p4est_faces_t); CHKERRXX(ierr);

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

  // my_p4est_node_neighbors_t
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::init                         ", 0, &log_my_p4est_node_neighbors_t); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dxx_central                  ", 0, &log_my_p4est_node_neighbors_t_dxx_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dyy_central                  ", 0, &log_my_p4est_node_neighbors_t_dyy_central); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_p4est_node_neighbors_t::dzz_central                  ", 0, &log_my_p4est_node_neighbors_t_dzz_central); CHKERRXX(ierr);
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
  ierr = PetscLogEventRegister("my_sc_notify                                            ", 0, &log_my_sc_notify); CHKERRXX(ierr);
  ierr = PetscLogEventRegister("my_sc_notify_allgather                                  ", 0, &log_my_sc_notify_allgather); CHKERRXX(ierr);
}
