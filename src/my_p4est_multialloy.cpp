#ifdef P4_TO_P8
#include "my_p8est_multialloy.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_utils.h>
#else
#include "my_p4est_multialloy.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_utils.h>
#endif

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_multialloy_one_step;
extern PetscLogEvent log_my_p4est_multialloy_compute_dt;
extern PetscLogEvent log_my_p4est_multialloy_compute_geometric_properties;
extern PetscLogEvent log_my_p4est_multialloy_compute_velocity;
extern PetscLogEvent log_my_p4est_multialloy_compute_solid;
extern PetscLogEvent log_my_p4est_multialloy_update_grid;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_history;
extern PetscLogEvent log_my_p4est_multialloy_save_vtk;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_transfer_data;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_regularize_front;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

my_p4est_node_neighbors_t *my_p4est_multialloy_t::v_ngbd;
double *my_p4est_multialloy_t::v_c_p, **my_p4est_multialloy_t::v_c_d_p, **my_p4est_multialloy_t::v_c_dd_p, **my_p4est_multialloy_t::v_normal_p;
double my_p4est_multialloy_t::v_factor;

my_p4est_multialloy_t::my_p4est_multialloy_t(int num_comps, int num_time_layers)
{
  num_comps_       = num_comps;
  num_time_layers_ = num_time_layers;

  ts_.resize(num_time_layers_);
  tl_.resize(num_time_layers_);
  cl_.resize(num_time_layers_, vec_and_ptr_array_t(num_comps_));

  psi_cl_.resize(num_comps_);

  front_velo_.resize(num_time_layers_);
  front_velo_norm_.resize(num_time_layers_);

  dt_.resize(num_time_layers_);

  history_cs_.resize(num_comps_);

  solute_diff_.assign(num_comps_, 1.e-5);
  part_coeff_ .assign(num_comps_, .86);

  latent_heat_     = 2350;    // J.cm-3
  density_l_       = 8.88e-3; // kg.cm-3
  density_s_       = 8.88e-3; // kg.cm-3
  heat_capacity_l_ = 0.46e3;  // J.kg-1.K-1
  heat_capacity_s_ = 0.46e3;  // J.kg-1.K-1
  thermal_cond_l_  = 6.07e-1; // W.cm-1.K-1
  thermal_cond_s_  = 6.07e-1; // W.cm-1.K-1
  melting_temp_    = 1728;    // K
  liquidus_slope_  = NULL;
  liquidus_value_  = NULL;
  vol_heat_gen_    = &zero_cf;

  num_seeds_ = 1;
  eps_c_.assign(1, &zero_cf);
  eps_v_.assign(1, &zero_cf);


  contr_bc_type_temp_ = NEUMANN;
  contr_bc_type_conc_.resize(num_comps_, NEUMANN);

  contr_bc_value_temp_ = NULL;
  contr_bc_value_conc_.assign(num_comps_, NULL);

  wall_bc_type_temp_ = NULL;
  wall_bc_type_conc_.assign(num_comps_, NULL);

  wall_bc_value_temp_ = NULL;
  wall_bc_value_conc_.assign(num_comps_, NULL);

  scaling_ = 1.;

  pin_every_n_iterations_ = 50;
  max_iterations_         = 50;

  bc_tolerance_           = 1.e-5;
  phi_thresh_             = 0.001;
  cfl_number_             = 0.1;

  time_   = 0;
  dt_max_ = DBL_MAX;
  dt_min_ = DBL_MIN;

  use_superconvergent_robin_ = false;
  use_points_on_interface_   = true;
  update_c0_robin_           = 0;
  save_history_              = true;
  enforce_planar_front_      = false;

  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v2;
  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v2;

  dendrite_cut_off_fraction_ = .75;
  dendrite_min_length_       = .05;

  front_smoothing_ = 0;
  curvature_smoothing_ = 0;
  curvature_smoothing_steps_ = 0;

  connectivity_ = NULL;
  p4est_ = NULL;
  ghost_ = NULL;
  nodes_ = NULL;
  hierarchy_ = NULL;
  ngbd_ = NULL;
}




my_p4est_multialloy_t::~my_p4est_multialloy_t()
{
  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  contr_phi_.destroy();
  front_phi_.destroy();
  front_curvature_.destroy();
  front_curvature_filtered_.destroy();

  contr_phi_dd_.destroy();
  front_phi_dd_.destroy();
  front_normal_.destroy();

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  for (i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].destroy();
    ts_[i].destroy();
    cl_[i].destroy();

    front_velo_     [i].destroy();
    front_velo_norm_[i].destroy();
  }

  cl0_grad_.destroy();

  psi_tl_.destroy();
  psi_ts_.destroy();
  psi_cl_.destroy();

  //--------------------------------------------------
  // Geometry on the auxiliary grid
  //--------------------------------------------------
  history_front_phi_.destroy();
  history_front_phi_nm1_.destroy();
  history_front_curvature_.destroy();
  history_front_velo_norm_.destroy();
  history_tf_.destroy();
  history_cs_.destroy();
  history_seed_.destroy();

  seed_map_.destroy();

  dendrite_number_.destroy();
  dendrite_tip_.destroy();
  bc_error_.destroy();

  /* destroy the p4est and its connectivity structure */
  delete ngbd_;
  delete hierarchy_;
  p4est_nodes_destroy(nodes_);
  p4est_ghost_destroy(ghost_);
  p4est_destroy      (p4est_);

  delete history_ngbd_;
  delete history_hierarchy_;
  p4est_nodes_destroy(history_nodes_);
  p4est_ghost_destroy(history_ghost_);
  p4est_destroy      (history_p4est_);

  my_p4est_brick_destroy(connectivity_, &brick_);

  delete sp_crit_;
}

void my_p4est_multialloy_t::set_front(Vec phi)
{
  VecCopyGhost(phi, front_phi_.vec);
  compute_geometric_properties_front();

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];

  foreach_node(n, history_nodes_)
  {
    node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(front_phi_.vec, interpolation_between_grids_);
  interp.interpolate(history_front_phi_.vec);

  VecCopyGhost(history_front_phi_.vec, history_front_phi_nm1_.vec);

  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(history_front_curvature_.vec);
}

void my_p4est_multialloy_t::set_container(Vec phi)
{
  VecCopyGhost(phi, contr_phi_.vec);
  compute_geometric_properties_contr();

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];

  foreach_node(n, history_nodes_)
  {
    node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(front_phi_.vec, interpolation_between_grids_);
  interp.interpolate(history_front_phi_.vec);

  VecCopyGhost(history_front_phi_.vec, history_front_phi_nm1_.vec);

  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(history_front_curvature_.vec);
}

void my_p4est_multialloy_t::initialize(MPI_Comm mpi_comm, double xyz_min[], double xyz_max[], int nxyz[], int periodicity[], CF_2 &level_set, int lmin, int lmax, double lip)
{

  /* create main p4est grid */
  connectivity_ = my_p4est_brick_new(nxyz, xyz_min, xyz_max, &brick_, periodicity);
  p4est_        = my_p4est_new(mpi_comm, connectivity_, 0, NULL, NULL);

  sp_crit_ = new splitting_criteria_cf_t(lmin, lmax, &level_set, lip);

  p4est_->user_pointer = (void*)(sp_crit_);
  my_p4est_refine(p4est_, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est_, P4EST_FALSE, NULL);

  ghost_ = my_p4est_ghost_new(p4est_, P4EST_CONNECT_FULL);
  nodes_ = my_p4est_nodes_new(p4est_, ghost_);

  hierarchy_ = new my_p4est_hierarchy_t(p4est_, ghost_, &brick_);
  ngbd_ = new my_p4est_node_neighbors_t(hierarchy_, nodes_);
  ngbd_->init_neighbors();

  /* create auxiliary p4est grid for keeping values in solid */
  history_p4est_ = p4est_copy(p4est_, P4EST_FALSE);
  history_ghost_ = my_p4est_ghost_new(history_p4est_, P4EST_CONNECT_FULL);
  history_nodes_ = my_p4est_nodes_new(history_p4est_, history_ghost_);

  history_hierarchy_ = new my_p4est_hierarchy_t(history_p4est_, history_ghost_, &brick_);
  history_ngbd_      = new my_p4est_node_neighbors_t(history_hierarchy_, history_nodes_);
  history_ngbd_->init_neighbors();

  /* determine the smallest cell size */
  ::dxyz_min(p4est_, dxyz_);
  dxyz_min_ = MIN(DIM(dxyz_[0], dxyz_[1], dxyz_[2]));
  dxyz_max_ = MAX(DIM(dxyz_[0], dxyz_[1], dxyz_[2]));
  diag_ = sqrt(SUMD(SQR(dxyz_[0]), SQR(dxyz_[1]), SQR(dxyz_[2])));

  dxyz_close_interface_ = 1.2*dxyz_max_;

  // front_phi_ and front_phi_dd_ are templates for all other vectors

  // allocate memory for physical fields
  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  front_phi_.create(p4est_, nodes_);
  front_phi_dd_.create(p4est_, nodes_);
  front_curvature_.create(front_phi_.vec);
  front_curvature_filtered_.create(front_phi_.vec);
  front_normal_.create(front_phi_dd_.vec);

  contr_phi_.create(p4est_, nodes_);
  contr_phi_dd_.create(p4est_, nodes_);

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  /* temperature */
  for (i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].create(front_phi_.vec);
    ts_[i].create(front_phi_.vec);
    cl_[i].create(front_phi_.vec);
    front_velo_     [i].create(front_phi_dd_.vec);
    front_velo_norm_[i].create(front_phi_.vec);
  }

  cl0_grad_.create(front_phi_dd_.vec);

  seed_map_.create(front_phi_.vec);

  dendrite_number_.create(front_phi_.vec);
  dendrite_tip_.create(front_phi_.vec);
  bc_error_.create(front_phi_.vec);

  psi_tl_.create(front_phi_.vec);
  psi_ts_.create(front_phi_.vec);
  psi_cl_.create(front_phi_.vec);

  VecSetGhost(psi_tl_.vec, 0.);
  VecSetGhost(psi_ts_.vec, 0.);
  for (i = 0; i < num_comps_; ++i)
  {
    VecSetGhost(psi_cl_.vec[i], 0.);
  }

  //--------------------------------------------------
  // Geometry on the auxiliary grid
  //--------------------------------------------------
  history_front_phi_.create(history_p4est_, history_nodes_);
  history_front_phi_nm1_.create(history_front_phi_.vec);
  history_front_curvature_.create(history_front_phi_.vec);
  history_front_velo_norm_.create(history_front_phi_.vec);

  history_tf_.create(history_front_phi_.vec);
  history_cs_.create(history_front_phi_.vec);
  history_seed_.create(history_front_phi_.vec);

}




void my_p4est_multialloy_t::compute_geometric_properties_front()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);

//  // solidification front
//  front_curvature_.destroy();
//  front_phi_dd_   .destroy();
//  front_normal_   .destroy();

//  front_curvature_.create(front_phi_.vec);
//  front_phi_dd_   .create(p4est_, nodes_);
//  front_normal_   .create(front_phi_dd_.vec);

  ngbd_->second_derivatives_central(front_phi_.vec, front_phi_dd_.vec);
  compute_normals_and_mean_curvature(*ngbd_, front_phi_.vec, front_normal_.vec, front_curvature_.vec);

  // flatten curvature values
  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  ls.extend_from_interface_to_whole_domain_TVD_in_place(front_phi_.vec, front_curvature_.vec, front_phi_.vec, 20);

//  front_curvature_.get_array();

//  double kappa_max = 1./dxyz_min_;

//  foreach_node(n, nodes_)
//  {
//    if      (front_curvature_.ptr[n] > kappa_max) front_curvature_.ptr[n] = kappa_max;
//    else if (front_curvature_.ptr[n] <-kappa_max) front_curvature_.ptr[n] =-kappa_max;
//  }

//  front_curvature_.restore_array();

  if (curvature_smoothing_ != 0.0 && curvature_smoothing_steps_ > 0) compute_filtered_curvature();

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_multialloy_t::compute_geometric_properties_contr()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
  ngbd_->second_derivatives_central(contr_phi_.vec, contr_phi_dd_.vec);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::compute_velocity()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);

  // TODO: implement a smarter extend from interface
  vec_and_ptr_dim_t c0_dd(front_normal_.vec);

  ngbd_->second_derivatives_central(cl_[0].vec[0], c0_dd.vec);

  // flattened interface concentration
  vec_and_ptr_t c_interface(front_phi_.vec);

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, cl_[0].vec[0], c_interface.vec);

  vec_and_ptr_dim_t front_velo_tmp(front_phi_dd_.vec);
  vec_and_ptr_t     front_velo_norm_tmp(front_phi_.vec);

  cl_[0]       .get_array();
  cl0_grad_    .get_array();
  front_normal_.get_array();
  c0_dd        .get_array();

  set_velo_interpolation(ngbd_, cl_[0].ptr[0], cl0_grad_.ptr, c0_dd.ptr, front_normal_.ptr, solute_diff_[0]/(1.-part_coeff_[0]));
  ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_norm_tmp.vec, front_velo_norm_[0].vec, 20, NULL, 0, 0, &velo);

  cl_[0]       .restore_array();
  cl0_grad_    .restore_array();
  front_normal_.restore_array();
  c0_dd        .restore_array();

  c0_dd.destroy();


  //cl_[0]             .get_array();
  //cl0_grad_          .get_array();
  //front_normal_      .get_array();
  //c_interface        .get_array();
  //front_velo_tmp     .get_array();
  //front_velo_norm_tmp.get_array();
  //
  //double xyz[P4EST_DIM];
  //
  //quad_neighbor_nodes_of_node_t qnnn;
  //for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  //{
  //  p4est_locidx_t n = ngbd_->get_layer_node(i);
////    qnnn = ngbd_->get_neighbors(n);
  //
  //  XCODE( front_velo_tmp.ptr[0][n] = -cl0_grad_.ptr[0][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  YCODE( front_velo_tmp.ptr[1][n] = -cl0_grad_.ptr[1][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  ZCODE( front_velo_tmp.ptr[2][n] = -cl0_grad_.ptr[2][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //
  //  front_velo_norm_tmp.ptr[n] = SUMD(front_velo_tmp.ptr[0][n]*front_normal_.ptr[0][n],
  //                                    front_velo_tmp.ptr[1][n]*front_normal_.ptr[1][n],
  //                                    front_velo_tmp.ptr[2][n]*front_normal_.ptr[2][n]);
  //}
  //
  //ierr = VecGhostUpdateBegin(front_velo_norm_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //foreach_dimension(dim)
  //{
  //  ierr = VecGhostUpdateBegin(front_velo_tmp.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //}
  //
  //for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  //{
  //  p4est_locidx_t n = ngbd_->get_local_node(i);
  //  //    qnnn = ngbd_->get_neighbors(n);
  //
  //  XCODE( front_velo_tmp.ptr[0][n] = -cl0_grad_.ptr[0][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  YCODE( front_velo_tmp.ptr[1][n] = -cl0_grad_.ptr[1][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  ZCODE( front_velo_tmp.ptr[2][n] = -cl0_grad_.ptr[2][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //
  //  front_velo_norm_tmp.ptr[n] = SUMD(front_velo_tmp.ptr[0][n]*front_normal_.ptr[0][n],
  //                                    front_velo_tmp.ptr[1][n]*front_normal_.ptr[1][n],
  //                                    front_velo_tmp.ptr[2][n]*front_normal_.ptr[2][n]);
  //}
  //
  //ierr = VecGhostUpdateEnd(front_velo_norm_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //foreach_dimension(dim)
  //{
  //  ierr = VecGhostUpdateEnd(front_velo_tmp.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //}
  //
  //cl_[0]             .restore_array();
  //cl0_grad_          .restore_array();
  //front_normal_      .restore_array();
  //c_interface        .restore_array();
  //front_velo_tmp     .restore_array();
  //front_velo_norm_tmp.restore_array();
  //
  //ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_norm_tmp.vec, front_velo_norm_[0].vec);

  foreach_dimension(dim)
  {
    VecPointwiseMultGhost(front_velo_[0].vec[dim], front_velo_norm_[0].vec, front_normal_.vec[dim]);
    //ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_tmp.vec[dim], front_velo_[0].vec[dim]);
  }

  c_interface.destroy();
  front_velo_norm_tmp.destroy();
  front_velo_tmp.destroy();

  if (enforce_planar_front_)
  {
    vec_and_ptr_t ones(front_phi_.vec);
    VecSetGhost(ones.vec, 1.);

    double velo_norm_avg = integrate_over_interface(p4est_, nodes_, front_phi_.vec, front_velo_norm_[0].vec)
                           /integrate_over_interface(p4est_, nodes_, front_phi_.vec, ones.vec);

    VecSetGhost(front_velo_norm_[0].vec, velo_norm_avg);

    foreach_dimension(dim)
    {
      VecCopyGhost(front_normal_.vec[dim], front_velo_[0].vec[dim]);
      VecScaleGhost(front_velo_[0].vec[dim], velo_norm_avg);
    }

    ones.destroy();
  }

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
}






void my_p4est_multialloy_t::compute_dt()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);

  double velo_norm_max = 0;
  double curvature_max = 0;

  front_phi_         .get_array();
  front_curvature_   .get_array();
  front_velo_norm_[0].get_array();

  foreach_local_node(n, nodes_)
  {
    if (fabs(front_phi_.ptr[n]) < dxyz_close_interface_)
    {
      velo_norm_max = MAX(velo_norm_max, fabs(front_velo_norm_[0].ptr[n]));
      curvature_max = MAX(curvature_max, fabs(front_curvature_.ptr[n]));
    }
  }

  double buffer[] = {velo_norm_max, curvature_max};
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, buffer, 2, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  velo_norm_max = buffer[0];
  curvature_max = buffer[1];

  front_velo_norm_max_ = velo_norm_max;

  for (i = 1; i < num_time_layers_; ++i)
  {
    dt_[i] = dt_[i-1];
  }
  dt_[0] = cfl_number_ * dxyz_min_/MAX(fabs(velo_norm_max),EPS);

  dt_[0] = MIN(dt_[0], dt_max_);
  dt_[0] = MAX(dt_[0], dt_min_);

  PetscPrintf(p4est_->mpicomm, "curvature max = %e, velo max = %e, dt = %e\n", curvature_max, velo_norm_max/scaling_, dt_[0]);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_multialloy_t::update_grid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid...\n");

  // advect interface and update p4est
  p4est_t       *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_, ngbd_);

  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2);
  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);

  if (num_time_layers_ == 2) sl.update_p4est(front_velo_[0].vec, dt_[0], front_phi_.vec, NULL, contr_phi_.vec);
  else   sl.update_p4est(front_velo_[1].vec, front_velo_[0].vec, dt_[1], dt_[0], front_phi_.vec);


  /* interpolate the quantities onto the new grid */
  // also shit n+1 -> n

  PetscPrintf(p4est_->mpicomm, "Transfering data between grids...\n");
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  foreach_node(n, nodes_np1)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  front_phi_dd_.destroy();
  front_phi_dd_.create(p4est_np1, nodes_np1);
  front_curvature_.destroy();
  front_curvature_.create(front_phi_.vec);
  front_curvature_filtered_.destroy();
  front_curvature_filtered_.create(front_phi_.vec);
  front_normal_.destroy();
  front_normal_.create(front_phi_dd_.vec);

  /* temperature */
  for (j = num_time_layers_-1; j > 0; --j)
  {
    tl_[j].destroy();
    tl_[j].create(front_phi_.vec);
    interp.set_input(tl_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(tl_[j].vec);

    ts_[j].destroy();
    ts_[j].create(front_phi_.vec);
    interp.set_input(ts_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(ts_[j].vec);

    cl_[j].destroy();
    cl_[j].create(front_phi_.vec);
    for (i = 0; i < num_comps_; ++i)
    {
      interp.set_input(cl_[j-1].vec[i], interpolation_between_grids_);
      interp.interpolate(cl_[j].vec[i]);
    }

    front_velo_norm_[j].destroy();
    front_velo_norm_[j].create(front_phi_.vec);
    interp.set_input(front_velo_norm_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(front_velo_norm_[j].vec);

    front_velo_[j].destroy();
    front_velo_[j].create(front_phi_dd_.vec);
    foreach_dimension(dim)
    {
      interp.set_input(front_velo_[j-1].vec[dim], interpolation_between_grids_);
      interp.interpolate(front_velo_[j].vec[dim]);
    }
  }

  tl_[0].destroy();
  tl_[0].create(front_phi_.vec);
  VecCopyGhost(tl_[1].vec, tl_[0].vec);
  ts_[0].destroy();
  ts_[0].create(front_phi_.vec);
  VecCopyGhost(ts_[1].vec, ts_[0].vec);
  cl_[0].destroy();
  cl_[0].create(front_phi_.vec);
  for (i = 0; i < num_comps_; ++i)
  {
    VecCopyGhost(cl_[1].vec[i], cl_[0].vec[i]);
  }
  front_velo_[0].destroy();
  front_velo_[0].create(front_phi_dd_.vec);
  front_velo_norm_[0].destroy();
  front_velo_norm_[0].create(front_phi_.vec);

  vec_and_ptr_t contr_phi_tmp(p4est_np1, nodes_np1);
  interp.set_input(contr_phi_.vec, linear);
  interp.interpolate(contr_phi_tmp.vec);
  contr_phi_.destroy();
  contr_phi_.set(contr_phi_tmp.vec);
  contr_phi_dd_.destroy();
  contr_phi_dd_.create(p4est_np1, nodes_np1);

  cl0_grad_.destroy();
  cl0_grad_.create(front_phi_dd_.vec);

  dendrite_number_.destroy();
  dendrite_number_.create(front_phi_.vec);
  dendrite_tip_.destroy();
  dendrite_tip_.create(front_phi_.vec);
  bc_error_.destroy();
  bc_error_.create(front_phi_.vec);

  if (num_seeds_ == 1)
  {
    seed_map_.destroy();
    seed_map_.create(front_phi_.vec);
    VecSetGhost(seed_map_.vec, 0.);
  } else {
    vec_and_ptr_t tmp(front_phi_.vec);
    interp.set_input(seed_map_.vec, linear);
    interp.interpolate(tmp.vec);
    seed_map_.destroy();
    seed_map_.set(tmp.vec);
  }

  vec_and_ptr_t psi_tl_tmp(front_phi_.vec);
  interp.set_input(psi_tl_.vec, linear);
  interp.interpolate(psi_tl_tmp.vec);
  psi_tl_.destroy();
  psi_tl_.set(psi_tl_tmp.vec);

  vec_and_ptr_t psi_ts_tmp(front_phi_.vec);
  interp.set_input(psi_ts_.vec, linear);
  interp.interpolate(psi_ts_tmp.vec);
  psi_ts_.destroy();
  psi_ts_.set(psi_ts_tmp.vec);

  vec_and_ptr_array_t psi_cl_tmp(num_comps_, front_phi_.vec);
  for (i = 0; i < num_comps_; ++i)
  {
    interp.set_input(psi_cl_.vec[i], linear);
    interp.interpolate(psi_cl_tmp.vec[i]);
  }
  psi_cl_.destroy();
  psi_cl_.set(psi_cl_tmp.vec.data());

  p4est_destroy      (p4est_); p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);

  regularize_front();

  /* reinitialize phi */
  my_p4est_level_set_t ls_new(ngbd_);
//  ls_new.reinitialize_1st_order_time_2nd_order_space(front_phi_.vec, 20);
  ls_new.reinitialize_2nd_order(front_phi_.vec, 30);

  if (num_seeds_ > 1)
  {
    VecScaleGhost(front_phi_.vec, -1.);
    ls_new.extend_Over_Interface_TVD_Full(front_phi_.vec, seed_map_.vec, 20, 0);
    seed_map_.get_array();
    foreach_node(n, nodes_) seed_map_.ptr[n] = round(seed_map_.ptr[n]);
    seed_map_.restore_array();
    VecScaleGhost(front_phi_.vec, -1.);
  }

  /* second derivatives, normals, curvature, angles */
  compute_geometric_properties_front();
  compute_geometric_properties_contr();

  /* refine history_p4est_ */
  update_grid_history();

  PetscPrintf(p4est_->mpicomm, "Done \n");
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::update_grid_history()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_history, 0, 0, 0, 0); CHKERRXX(ierr);
  PetscPrintf(p4est_->mpicomm, "Refining auxiliary p4est for storing data...\n");

  Vec tmp = history_front_phi_.vec;
  history_front_phi_.vec = history_front_phi_nm1_.vec;
  history_front_phi_nm1_.vec = tmp;

  p4est_t       *history_p4est_np1 = p4est_copy(history_p4est_, P4EST_FALSE);
  p4est_ghost_t *history_ghost_np1 = my_p4est_ghost_new(history_p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *history_nodes_np1 = my_p4est_nodes_new(history_p4est_np1, history_ghost_np1);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)history_ngbd_->p4est->user_pointer;

  vec_and_ptr_t front_phi_np1(history_p4est_np1, history_nodes_np1);

  bool is_grid_changing = true;
  int  counter          = 0;

  while (is_grid_changing)
  {
    // interpolate from a coarse grid to a fine one
    front_phi_np1.get_array();

    my_p4est_interpolation_nodes_t interp(ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, history_nodes_np1)
    {
      node_xyz_fr_n(n, history_p4est_np1, history_nodes_np1, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(front_phi_.vec, linear);
    interp.interpolate(front_phi_np1.ptr);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine(history_p4est_np1, history_nodes_np1, front_phi_np1.ptr);

    front_phi_np1.restore_array();

    if (is_grid_changing)
    {
      my_p4est_partition(history_p4est_np1, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(history_ghost_np1); history_ghost_np1 = my_p4est_ghost_new(history_p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(history_nodes_np1); history_nodes_np1 = my_p4est_nodes_new(history_p4est_np1, history_ghost_np1);

      front_phi_np1.destroy();
      front_phi_np1.create(history_p4est_np1, history_nodes_np1);
    }

    counter++;
  }

  history_p4est_np1->user_pointer = (void*) sp_old;

  history_front_phi_.destroy();
  history_front_phi_.set(front_phi_np1.vec);

  // transfer variables to the new grid
  my_p4est_interpolation_nodes_t history_interp(history_ngbd_);

  double xyz[P4EST_DIM];
  foreach_node(n, history_nodes_np1)
  {
    node_xyz_fr_n(n, history_p4est_np1, history_nodes_np1, xyz);
    history_interp.add_point(n, xyz);
  }

  vec_and_ptr_t tf_tmp(history_front_phi_.vec);
  history_interp.set_input(history_tf_.vec, linear);
  history_interp.interpolate(tf_tmp.vec);
  history_tf_.destroy();
  history_tf_.set(tf_tmp.vec);

  vec_and_ptr_array_t cs_tmp(num_comps_, history_front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    history_interp.set_input(history_cs_.vec[i], linear);
    history_interp.interpolate(cs_tmp.vec[i]);
  }
  history_cs_.destroy();
  history_cs_.set(cs_tmp.vec.data());

  vec_and_ptr_t curv_tmp(history_front_phi_.vec);
  history_interp.set_input(history_front_curvature_.vec, linear);
  history_interp.interpolate(curv_tmp.vec);
  history_front_curvature_.destroy();
  history_front_curvature_.set(curv_tmp.vec);

  vec_and_ptr_t velo_tmp(history_front_phi_.vec);
  history_interp.set_input(history_front_velo_norm_.vec, linear);
  history_interp.interpolate(velo_tmp.vec);
  history_front_velo_norm_.destroy();
  history_front_velo_norm_.set(velo_tmp.vec);

  vec_and_ptr_t phi_nm1_tmp(history_front_phi_.vec);
  history_interp.set_input(history_front_phi_nm1_.vec, linear);
  history_interp.interpolate(phi_nm1_tmp.vec);
  history_front_phi_nm1_.destroy();
  history_front_phi_nm1_.set(phi_nm1_tmp.vec);

  vec_and_ptr_t seed_tmp(history_front_phi_.vec);
  history_interp.set_input(history_seed_.vec, linear);
  history_interp.interpolate(seed_tmp.vec);
  history_seed_.destroy();
  history_seed_.set(seed_tmp.vec);

  p4est_destroy(history_p4est_);       history_p4est_ = history_p4est_np1;
  p4est_ghost_destroy(history_ghost_); history_ghost_ = history_ghost_np1;
  p4est_nodes_destroy(history_nodes_); history_nodes_ = history_nodes_np1;
  history_hierarchy_->update(history_p4est_, history_ghost_);
  history_ngbd_->update(history_hierarchy_, history_nodes_);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_history, 0, 0, 0, 0); CHKERRXX(ierr);
}

int my_p4est_multialloy_t::one_step()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);

  time_ += dt_[0];

  // update time in interface and boundary conditions
//  gibbs_thomson_->t = time_;
  vol_heat_gen_ ->t = time_;

  contr_bc_value_temp_->t = time_;
  wall_bc_value_temp_ ->t = time_;

  for (i = 0; i < num_comps_; ++i)
  {
    contr_bc_value_conc_[i]->t = time_;
    wall_bc_value_conc_ [i]->t = time_;
  }

  // compute right-hand sides
  vec_and_ptr_t       rhs_tl(tl_[0].vec);
  vec_and_ptr_t       rhs_ts(ts_[0].vec);
  vec_and_ptr_array_t rhs_cl(num_comps_, cl_[0].vec.data());

  rhs_tl.get_array();
  rhs_ts.get_array();
  rhs_cl.get_array();

  for (i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].get_array();
    ts_[i].get_array();
    cl_[i].get_array();
  }

  double xyz[P4EST_DIM];
  double heat_gen = 0;

  std::vector<double> time_coeffs;

  variable_step_BDF_implicit(num_time_layers_-1, dt_, time_coeffs);

  foreach_node(n, nodes_)
  {
    if (vol_heat_gen_ != NULL)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      heat_gen = vol_heat_gen_->value(xyz);
    }

    rhs_tl.ptr[n] = 0;
    rhs_ts.ptr[n] = 0;

    for (j = 0; j < num_comps_; ++j)
    {
      rhs_cl.ptr[j][n] = 0;
    }

    for (i = 1; i < num_time_layers_; ++i)
    {
      rhs_tl.ptr[n] -= time_coeffs[i]*tl_[i].ptr[n];
      rhs_ts.ptr[n] -= time_coeffs[i]*ts_[i].ptr[n];

      for (j = 0; j < num_comps_; ++j)
      {
        rhs_cl.ptr[j][n] -= time_coeffs[i]*cl_[i].ptr[j][n];
      }
    }

    double test = rhs_tl.ptr[n]*density_l_*heat_capacity_l_/dt_[0];

    rhs_tl.ptr[n] = rhs_tl.ptr[n]*density_l_*heat_capacity_l_/dt_[0] + heat_gen;
    rhs_ts.ptr[n] = rhs_ts.ptr[n]*density_s_*heat_capacity_s_/dt_[0] + heat_gen;

    for (int i = 0; i < num_comps_; ++i)
    {
      rhs_cl.ptr[i][n] = rhs_cl.ptr[i][n]/dt_[0];
    }
  }

  rhs_tl.restore_array();
  rhs_ts.restore_array();
  rhs_cl.restore_array();

  for (i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].restore_array();
    ts_[i].restore_array();
    cl_[i].restore_array();
  }

  vector<double> conc_diag(num_comps_, time_coeffs[0]/dt_[0]);

  // solve coupled system of equations
  my_p4est_poisson_nodes_multialloy_t solver_all_in_one(ngbd_, num_comps_);

  Vec curvature_to_use = (curvature_smoothing_ != 0.0 && curvature_smoothing_steps_ > 0) ? front_curvature_filtered_.vec : front_curvature_.vec;

  solver_all_in_one.set_front(front_phi_.vec, front_phi_dd_.vec, front_normal_.vec, curvature_to_use);

  solver_all_in_one.set_composition_parameters(conc_diag.data(), solute_diff_.data(), part_coeff_.data());
  solver_all_in_one.set_thermal_parameters(latent_heat_,
                                           density_l_*heat_capacity_l_*time_coeffs[0]/dt_[0], thermal_cond_l_,
                                           density_s_*heat_capacity_s_*time_coeffs[0]/dt_[0], thermal_cond_s_);
  solver_all_in_one.set_gibbs_thomson(zero_cf);
  solver_all_in_one.set_liquidus(melting_temp_, liquidus_value_, liquidus_slope_);
  solver_all_in_one.set_undercoolings(num_seeds_, seed_map_.vec, eps_v_.data(), eps_c_.data());

  solver_all_in_one.set_rhs(rhs_tl.vec, rhs_ts.vec, rhs_cl.vec.data());

  if (contr_phi_.vec != NULL)
  {
    solver_all_in_one.set_container(contr_phi_.vec, contr_phi_dd_.vec);
    solver_all_in_one.set_container_conditions_thermal(contr_bc_type_temp_, *contr_bc_value_temp_);
    solver_all_in_one.set_container_conditions_composition(contr_bc_type_conc_.data(), contr_bc_value_conc_.data());
  }

  vector<CF_DIM *> zeros_cf(num_comps_, &zero_cf);

  solver_all_in_one.set_front_conditions(zero_cf, zero_cf, zeros_cf.data());

  solver_all_in_one.set_wall_conditions_thermal(*wall_bc_type_temp_, *wall_bc_value_temp_);
  solver_all_in_one.set_wall_conditions_composition(wall_bc_type_conc_.data(), wall_bc_value_conc_.data());

  solver_all_in_one.set_pin_every_n_iterations(pin_every_n_iterations_);
  solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
  solver_all_in_one.set_use_points_on_interface(use_points_on_interface_);
  solver_all_in_one.set_update_c0_robin(update_c0_robin_);
  solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin_);

  my_p4est_interpolation_nodes_t interp_c0_n(ngbd_);
  interp_c0_n.set_input(cl_[1].vec[0], linear);
  solver_all_in_one.set_c0_guess(interp_c0_n);

//  bc_error_.destroy();
//  bc_error_.create(front_phi_.vec);

//  cl0_grad_.destroy();
//  cl0_grad_.create(p4est_, nodes_);

  double bc_error_max = 0;

//  solver_all_in_one.set_verbose_mode(1);

  int one_step_iterations = solver_all_in_one.solve(tl_[0].vec, ts_[0].vec, cl_[0].vec.data(), cl0_grad_.vec, bc_error_.vec, bc_error_max, 1,
      NULL, NULL, psi_tl_.vec, psi_ts_.vec, psi_cl_.vec.data());

  // compute velocity
  compute_velocity();
  compute_solid();

  rhs_tl.destroy();
  rhs_ts.destroy();
  rhs_cl.destroy();

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);
  return one_step_iterations;
}

void my_p4est_multialloy_t::save_VTK(int iter)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);

  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name[1000];
  sprintf(name, "%s/vtu/multialloy_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);

  // cell data
  std::vector<double *>    cell_data;
  std::vector<std::string> cell_data_names;

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est_, ghost_, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_->first_local_tree; tree_idx <= p4est_->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost_->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, q);
    l_p[p4est_->local_num_quadrants+q] = quad->level;
  }

  cell_data.push_back(l_p); cell_data_names.push_back("leaf_level");

  // point data
  std::vector<double *>    point_data;
  std::vector<std::string> point_data_names;

  front_phi_.get_array(); point_data.push_back(front_phi_.ptr); point_data_names.push_back("phi");

  if (contr_phi_.vec != NULL)
  {
    contr_phi_.get_array(); point_data.push_back(contr_phi_.ptr); point_data_names.push_back("contr");
  }

  tl_[0].get_array(); point_data.push_back(tl_[0].ptr); point_data_names.push_back("tl");
  ts_[0].get_array(); point_data.push_back(ts_[0].ptr); point_data_names.push_back("ts");
  cl_[0].get_array();
  for (int i = 0; i < num_comps_; ++i)
  {
    char numstr[21];
    sprintf(numstr, "%d", i);
    std::string name("cl");
    point_data.push_back(cl_[0].ptr[i]); point_data_names.push_back(name + numstr);
  }

  front_velo_norm_[0]      .get_array(); point_data.push_back(front_velo_norm_[0].ptr);       point_data_names.push_back("vn");
  front_curvature_         .get_array(); point_data.push_back(front_curvature_.ptr);          point_data_names.push_back("kappa");
  bc_error_                .get_array(); point_data.push_back(bc_error_.ptr);                 point_data_names.push_back("bc_error");
  dendrite_number_         .get_array(); point_data.push_back(dendrite_number_.ptr);          point_data_names.push_back("dendrite_number");
  dendrite_tip_            .get_array(); point_data.push_back(dendrite_tip_.ptr);             point_data_names.push_back("dendrite_tip");
  front_curvature_filtered_.get_array(); point_data.push_back(front_curvature_filtered_.ptr); point_data_names.push_back("kappa_filt");
  seed_map_                .get_array(); point_data.push_back(seed_map_.ptr);                 point_data_names.push_back("seed_num");

  VecScaleGhost(front_velo_norm_[0].vec, 1./scaling_);

  my_p4est_vtk_write_all_vector_form(p4est_, nodes_, ghost_,
                                     P4EST_TRUE, P4EST_TRUE,
                                     name,
                                     point_data, point_data_names,
                                     cell_data, cell_data_names);

  VecScaleGhost(front_velo_norm_[0].vec, scaling_);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  front_phi_.restore_array();

  if (contr_phi_.vec != NULL)
  {
    contr_phi_.restore_array();
  }

  tl_[0].restore_array();
  ts_[0].restore_array();
  cl_[0].restore_array();

  front_velo_norm_[0]      .restore_array();
  front_curvature_         .restore_array();
  bc_error_                .restore_array();
  dendrite_number_         .restore_array();
  dendrite_tip_            .restore_array();
  front_curvature_filtered_.restore_array();
  seed_map_                .restore_array();

  PetscPrintf(p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::save_VTK_solid(int iter)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);

  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(history_p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)history_p4est_->user_pointer;

  char name[1000];

  sprintf(name, "%s/vtu/multialloy_solid_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);

  // cell data
  std::vector<double *>    cell_data;
  std::vector<std::string> cell_data_names;

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(history_p4est_, history_ghost_, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = history_p4est_->first_local_tree; tree_idx <= history_p4est_->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(history_p4est_->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<history_ghost_->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&history_ghost_->ghosts, q);
    l_p[history_p4est_->local_num_quadrants+q] = quad->level;
  }

  cell_data.push_back(l_p); cell_data_names.push_back("leaf_level");

  // point data
  std::vector<double *>    point_data;
  std::vector<std::string> point_data_names;

  history_front_phi_      .get_array(); point_data.push_back(history_front_phi_.ptr);       point_data_names.push_back("phi");
  history_front_curvature_.get_array(); point_data.push_back(history_front_curvature_.ptr); point_data_names.push_back("kappa");
  history_front_velo_norm_.get_array(); point_data.push_back(history_front_velo_norm_.ptr); point_data_names.push_back("vn");
  history_seed_           .get_array(); point_data.push_back(history_seed_.ptr);            point_data_names.push_back("seed");
  history_tf_             .get_array(); point_data.push_back(history_tf_.ptr);              point_data_names.push_back("tf");
  history_cs_             .get_array();
  for (int i = 0; i < num_comps_; ++i)
  {
    char numstr[21];
    sprintf(numstr, "%d", i);
    std::string name("cl");
    point_data.push_back(history_cs_.ptr[i]); point_data_names.push_back(name + numstr);
  }

  VecScaleGhost(history_front_velo_norm_.vec, 1./scaling_);

  my_p4est_vtk_write_all_vector_form(history_p4est_, history_nodes_, history_ghost_,
                                     P4EST_TRUE, P4EST_TRUE,
                                     name,
                                     point_data, point_data_names,
                                     cell_data, cell_data_names);

  VecScaleGhost(history_front_velo_norm_.vec, scaling_);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  history_front_phi_      .restore_array();
  history_front_curvature_.restore_array();
  history_front_velo_norm_.restore_array();
  history_tf_             .restore_array();
  history_cs_             .restore_array();
  history_seed_           .restore_array();

  PetscPrintf(history_p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::count_dendrites(int iter)
{
  // this code assumes dendrites grow in the positive y-direction

  // find boundaries of the "mushy zone"
  double mushy_zone_min = brick_.xyz_max[1];
  double mushy_zone_max = brick_.xyz_min[1];

  front_phi_.get_array();
  foreach_local_node(n, nodes_)
  {
    double y_coord = node_y_fr_n(n, p4est_, nodes_);

    if (front_phi_.ptr[n] > 0 && y_coord > mushy_zone_max) mushy_zone_max = y_coord;
    if (front_phi_.ptr[n] < 0 && y_coord < mushy_zone_min) mushy_zone_min = y_coord;
  }
  front_phi_.restore_array();

  double buffer[] = { mushy_zone_min, mushy_zone_max};
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &buffer, 2, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  mushy_zone_min = buffer[0];
  mushy_zone_max = buffer[1];

//  dendrite_number_.destroy();
//  dendrite_tip_   .destroy();

//  dendrite_number_.create(front_phi_.vec);
//  dendrite_tip_   .create(front_phi_.vec);

  VecSetGhost(dendrite_number_.vec, -1.);
  VecSetGhost(dendrite_tip_.vec, -1.);

  // cut off denrites' base
  double cut_off_length = mushy_zone_max - dendrite_cut_off_fraction_*MAX(dendrite_min_length_, mushy_zone_max-mushy_zone_min);

  vec_and_ptr_t phi_cut(front_phi_.vec);
  VecCopyGhost(front_phi_.vec, phi_cut.vec);

  phi_cut.get_array();
  foreach_node(n, nodes_)
  {
    if (node_y_fr_n(n, p4est_, nodes_) < cut_off_length) phi_cut.ptr[n] = -1;
  }
  phi_cut.restore_array();

  // enumerate dendrite by using Arthur's parallel region-growing code
  num_dendrites_ = 0;
  compute_islands_numbers(*ngbd_, phi_cut.vec, num_dendrites_, dendrite_number_.vec);
  phi_cut.destroy();

  ierr = PetscPrintf(p4est_->mpicomm, "%d prominent dendrites found.\n", num_dendrites_); CHKERRXX(ierr);

  // find the tip of every dendrite
  std::vector<double> x_tip(num_dendrites_, 0);
  std::vector<double> y_tip(num_dendrites_, 0);

  dendrite_number_.get_array();
  front_phi_      .get_array();

  foreach_local_node(n, nodes_)
  {
    int i = (int) dendrite_number_.ptr[n];
    if (i >= 0)
    {
      const quad_neighbor_nodes_of_node_t qnnn = ngbd_->get_neighbors(n);
      int j = (int) qnnn.f_0p0_linear(dendrite_number_.ptr);
      if (j < 0)
      {
        double phi_n   = front_phi_.ptr[n];
        double phi_nei = qnnn.f_0p0_linear(front_phi_.ptr);

        double y_coord = node_y_fr_n(n, p4est_, nodes_) + fabs(phi_n)/(fabs(phi_n)+fabs(phi_nei)) * dxyz_[1];

        if (y_coord > y_tip[i])
        {
          y_tip[i] = y_coord;
          x_tip[i] = node_x_fr_n(n, p4est_, nodes_);
        }
      }
    }
  }
  dendrite_number_.restore_array();
  front_phi_      .restore_array();

  std::vector<double> y_tip_g(num_dendrites_, 0);
  std::vector<double> x_tip_g(num_dendrites_, 0);

  mpiret = MPI_Allreduce(y_tip.data(), y_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  for (unsigned int i = 0; i < num_dendrites_; ++i)
  {
    if (y_tip[i] != y_tip_g[i]) x_tip[i] = -10;
  }

  mpiret = MPI_Allreduce(x_tip.data(), x_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  // auxiliary field that shows found tip locations
  dendrite_tip_.get_array();
  foreach_node(n, nodes_)
  {
    dendrite_tip_.ptr[n] = -1;
    double x_coord = node_x_fr_n(n, p4est_, nodes_);
    for (unsigned short i = 0; i < num_dendrites_; ++i)
    {
      if (fabs(x_tip_g[i] - x_coord) < EPS) dendrite_tip_.ptr[n] = i;
    }
  }
  dendrite_tip_.restore_array();

  // sample quantities along the tip of every dendrite
  // specifically: phi, c0, c1, t, vn, c0s, c1s, tf, kappa, velo

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;
  int nb_sample_points = brick_.nxyztrees[1]*pow(2, data->max_lvl)+1;

  std::vector<double> line_phi;
  std::vector<double> line_tl;
  std::vector<double> line_ts;
  std::vector<double> line_tf;
  std::vector<double> line_vn;
  std::vector<double> line_kappa;
  std::vector<double> line_vf;

  std::vector< std::vector<double> > line_cl(num_comps_);
  std::vector< std::vector<double> > line_cs(num_comps_);

  for (unsigned short dendrite_idx = 0; dendrite_idx < num_dendrites_; ++dendrite_idx)
  {
    double xyz0[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_.xyz_min[1] };
    double xyz1[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_.xyz_max[1] };

    double xyz[P4EST_DIM];
    double dxyz[P4EST_DIM];

    foreach_dimension(dim) dxyz[dim] = (xyz1[dim]-xyz0[dim])/((double) nb_sample_points - 1.);

    line_phi  .clear(); line_phi  .resize(nb_sample_points, -DBL_MAX);
    line_tl   .clear(); line_tl   .resize(nb_sample_points, -DBL_MAX);
    line_ts   .clear(); line_ts   .resize(nb_sample_points, -DBL_MAX);
    line_tf   .clear(); line_tf   .resize(nb_sample_points, -DBL_MAX);
    line_vn   .clear(); line_vn   .resize(nb_sample_points, -DBL_MAX);
    line_vf   .clear(); line_vf   .resize(nb_sample_points, -DBL_MAX);
    line_kappa.clear(); line_kappa.resize(nb_sample_points, -DBL_MAX);

    for (int i = 0; i < num_comps_; ++i)
    {
      line_cl[i].clear(); line_cl[i].resize(nb_sample_points, -DBL_MAX);
      line_cs[i].clear(); line_cs[i].resize(nb_sample_points, -DBL_MAX);
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);
    my_p4est_interpolation_nodes_t h_interp(history_ngbd_);

    for (unsigned int i = 0; i < nb_sample_points; ++i)
    {
      foreach_dimension(dim) xyz[dim] = xyz0[dim] + ((double) i) * dxyz[dim];
      interp.add_point_local(i, xyz);
      h_interp.add_point_local(i, xyz);
    }

    interp.set_input(front_phi_.vec, linear); interp.interpolate_local(line_phi.data());
    interp.set_input(front_velo_norm_[0].vec, linear); interp.interpolate_local(line_vn.data());
    interp.set_input(tl_[0].vec, linear); interp.interpolate_local(line_tl.data());
    interp.set_input(ts_[0].vec, linear); interp.interpolate_local(line_ts.data());
    for (int i = 0; i < num_comps_; ++i)
    {
      interp.set_input(cl_[0].vec[i], linear); interp.interpolate_local(line_cl[i].data());
    }

    h_interp.set_input(history_tf_.vec, linear); h_interp.interpolate_local(line_tf.data());
    h_interp.set_input(history_front_curvature_.vec, linear); h_interp.interpolate_local(line_kappa.data());
    h_interp.set_input(history_front_velo_norm_.vec, linear); h_interp.interpolate_local(line_vf.data());
    for (i = 0; i < num_comps_; ++i)
    {
      h_interp.set_input(history_cs_.vec[i], linear); h_interp.interpolate_local(line_cs[i].data());
    }

    int mpiret;

    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_phi.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_tl .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_ts .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_tf .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_vn .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_vf .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_kappa.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    for (i = 0; i < num_comps_; ++i)
    {
      mpiret = MPI_Allreduce(MPI_IN_PLACE, line_cl[i].data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, line_cs[i].data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    }

    const char* out_dir = getenv("OUT_DIR");

    char dirname[1024];
    sprintf(dirname, "%s/dendrites/%05d", out_dir, iter);

    std::ostringstream command;
    command << "mkdir -p " << dirname;
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    char filename[1024];
    if (p4est_->mpirank == 0)
    {
      std::ios_base::openmode mode = (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app);
      sprintf(filename, "%s/%s.txt", dirname, "phi"); save_vector(filename, line_phi, mode);
      sprintf(filename, "%s/%s.txt", dirname, "tl");  save_vector(filename, line_tl,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "ts");  save_vector(filename, line_ts,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "tf");  save_vector(filename, line_tf,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "vn");  save_vector(filename, line_vn,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "vf");  save_vector(filename, line_vf,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "kappa"); save_vector(filename, line_kappa, mode);

      for (int i = 0; i < num_comps_; ++i)
      {
        sprintf(filename, "%s/%s%d.txt", dirname, "cl", i);  save_vector(filename, line_cl[i], mode);
        sprintf(filename, "%s/%s%d.txt", dirname, "cs", i);  save_vector(filename, line_cs[i], mode);
      }
    }
  }
}

void my_p4est_multialloy_t::sample_along_line(const double xyz0[], const double xyz1[], const unsigned int nb_points, Vec data, std::vector<double> out)
{
  out.clear();
  out.resize(nb_points, -DBL_MAX);

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  double dxyz[P4EST_DIM];

  foreach_dimension(dim) dxyz[dim] = (xyz1[dim]-xyz0[dim])/((double) nb_points - 1.);

  for (unsigned int i = 0; i < nb_points; ++i)
  {
    foreach_dimension(dim) xyz[dim] = xyz0[dim] + ((double) i) * dxyz[dim];
    interp.add_point(i, xyz);
  }

  interp.set_input(data, linear);
  interp.interpolate(out.data());

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, out.data(), nb_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_multialloy_t::regularize_front()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
  /* remove problem geometries */
  PetscPrintf(p4est_->mpicomm, "Removing problem geometries...\n");

  vec_and_ptr_t front_phi_cur;

  front_phi_cur.set(front_phi_.vec);

  p4est_t       *p4est_cur = p4est_;
  p4est_nodes_t *nodes_cur = nodes_;
  p4est_ghost_t *ghost_cur = ghost_;
  my_p4est_node_neighbors_t *ngbd_cur = ngbd_;
  my_p4est_hierarchy_t *hierarchy_cur = hierarchy_;

  if (front_smoothing_ != 0)
  {
    p4est_cur = p4est_copy(p4est_, P4EST_FALSE);
    ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
    nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

    front_phi_cur.create(front_phi_.vec);
    VecCopyGhost(front_phi_.vec, front_phi_cur.vec);

    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est_->user_pointer;
    bool is_grid_changing = true;
    while (is_grid_changing)
    {
      front_phi_cur.get_array();
      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl-front_smoothing_, sp_old->lip);
      is_grid_changing = sp.refine_and_coarsen(p4est_cur, nodes_cur, front_phi_cur.ptr);
      front_phi_cur.restore_array();

      if (is_grid_changing)
      {
        my_p4est_partition(p4est_cur, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_cur); ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_cur); nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

        front_phi_cur.destroy();
        front_phi_cur.create(p4est_cur, nodes_cur);

        my_p4est_interpolation_nodes_t interp(ngbd_);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_cur)
        {
          node_xyz_fr_n(n, p4est_cur, nodes_cur, xyz);
          interp.add_point(n, xyz);
        }

        interp.set_input(front_phi_.vec, linear); // we know that it is not really an interpolation, rather just a transfer, so therefore linear
        interp.interpolate(front_phi_cur.vec);
      }
    }

    hierarchy_cur = new my_p4est_hierarchy_t(p4est_cur, ghost_cur, &brick_);
    ngbd_cur = new my_p4est_node_neighbors_t(hierarchy_cur, nodes_cur);
    ngbd_cur->init_neighbors();
  }

  vec_and_ptr_t front_phi_tmp(front_phi_cur.vec);

  p4est_locidx_t nei_n[num_neighbors_cube];
  bool           nei_e[num_neighbors_cube];

  double band = 2.*diag_;

  front_phi_tmp.get_array();
  front_phi_cur.get_array();

  // first pass: smooth out extremely curved regions
  // TODO: make it iterative
  bool is_changed = false;
  foreach_local_node(n, nodes_cur)
  {
    if (fabs(front_phi_cur.ptr[n]) < band)
    {
      ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

      unsigned short num_neg = 0;
      unsigned short num_pos = 0;

      for (unsigned short nn = 0; nn < num_neighbors_cube; ++nn)
      {
        if (front_phi_cur.ptr[nei_n[nn]] <= 0) num_neg++;
        if (front_phi_cur.ptr[nei_n[nn]] >= 0) num_pos++;
      }

      if ( (front_phi_cur.ptr[n] <= 0 && num_neg < 3) ||
           (front_phi_cur.ptr[n] >= 0 && num_pos < 3) )
      {
//        front_phi_cur.ptr[n] = front_phi_cur.ptr[n] < 0 ? 10.*EPS : -10.*EPS;
//        front_phi_cur.ptr[n] = front_phi_cur.ptr[n] < 0 ? 10.*EPS : -10.*EPS;
        if (num_neg < 3) front_phi_cur.ptr[n] =  0.01*diag_;
        if (num_pos < 3) front_phi_cur.ptr[n] = -0.01*diag_;

        // check if node is a layer node (= a ghost node for another process)
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
        if (ni->pad8 != 0) is_changed = true;
//        throw;
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  if (is_changed)
  {
    ierr = VecGhostUpdateBegin(front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  VecCopyGhost(front_phi_cur.vec, front_phi_tmp.vec);

  // second pass: bridge narrow gaps
  // TODO: develop a more general approach that works in 3D as well
  double new_phi_val = .5*dxyz_min_*pow(2., front_smoothing_);
  is_changed = false;
  bool is_ghost_changed = false;
  foreach_local_node(n, nodes_cur)
  {
    if (front_phi_cur.ptr[n] < 0 && front_phi_cur.ptr[n] > -band)
    {
      ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

      bool merge = (front_phi_cur.ptr[nei_n[nn_m00]] > 0 &&
                   front_phi_cur.ptr[nei_n[nn_p00]] > 0 &&
          front_phi_cur.ptr[nei_n[nn_0m0]] > 0 &&
          front_phi_cur.ptr[nei_n[nn_0p0]] > 0)
          || ((front_phi_cur.ptr[nei_n[nn_m00]] > 0 && front_phi_cur.ptr[nei_n[nn_p00]] > 0) &&
          (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_0m0]] < 0 || front_phi_cur.ptr[nei_n[nn_pm0]] < 0) &&
          (front_phi_cur.ptr[nei_n[nn_mp0]] < 0 || front_phi_cur.ptr[nei_n[nn_0p0]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0))
          || ((front_phi_cur.ptr[nei_n[nn_0m0]] > 0 && front_phi_cur.ptr[nei_n[nn_0p0]] > 0) &&
          (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_m00]] < 0 || front_phi_cur.ptr[nei_n[nn_mp0]] < 0) &&
          (front_phi_cur.ptr[nei_n[nn_pm0]] < 0 || front_phi_cur.ptr[nei_n[nn_p00]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0));

      if (merge)
      {
        front_phi_tmp.ptr[n] = new_phi_val;

        // check if node is a layer node (= a ghost node for another process)
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
        if (ni->pad8 != 0) is_ghost_changed = true;

        is_changed = true;
      }

    }
  }

  front_phi_tmp.restore_array();
  front_phi_cur.restore_array();

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed,       1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_ghost_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  if (is_ghost_changed)
  {
    ierr = VecGhostUpdateBegin(front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // third pass: look for isolated pools of liquid and remove them
  if (is_changed) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
  {
    int num_islands = 0;
    vec_and_ptr_t island_number(front_phi_cur.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd_cur, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1)
    {
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_cur)
      {
        if (island_number.ptr[n] >= 0)
        {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i)
      {
        if (island_area[i] > island_area_max)
        {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // solidify all but the biggest pool
      foreach_node(n, nodes_cur)
      {
        if (front_phi_tmp.ptr[n] < 0 && island_number.ptr[n] != main_island)
        {
          front_phi_tmp.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      front_phi_tmp.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
  }

//  front_phi_cur.destroy();
//  front_phi_cur.set(front_phi_tmp.vec);
  VecCopyGhost(front_phi_tmp.vec, front_phi_cur.vec);
  front_phi_tmp.destroy();

  // iterpolate back onto fine grid
  if (front_smoothing_ != 0)
  {
    my_p4est_level_set_t ls(ngbd_cur);
    ls.reinitialize_1st_order_time_2nd_order_space(front_phi_cur.vec, 20);

    my_p4est_interpolation_nodes_t interp(ngbd_cur);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes_)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(front_phi_cur.vec, quadratic_non_oscillatory_continuous_v2); // we know that it is not really an interpolation, rather just a transfer, so therefore linear
    interp.interpolate(front_phi_.vec);

    front_phi_cur.destroy();
    delete ngbd_cur;
    delete hierarchy_cur;
    p4est_nodes_destroy(nodes_cur);
    p4est_ghost_destroy(ghost_cur);
    p4est_destroy(p4est_cur);
  } else {
    front_phi_.set(front_phi_cur.vec);
  }
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::compute_solid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_solid, 0, 0, 0, 0); CHKERRXX(ierr);
  // get new values for nodes that just solidified
  history_front_phi_    .get_array();
  history_front_phi_nm1_.get_array();

  // first, figure out which nodes just became solid
  my_p4est_interpolation_nodes_t interp(ngbd_);
  std::vector<p4est_locidx_t> freezing_nodes;
  double xyz[P4EST_DIM];
  foreach_node(n, history_nodes_)
  {
    if (history_front_phi_    .ptr[n] > 0 &&
        history_front_phi_nm1_.ptr[n] < 0)
    {
      node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
      interp.add_point(n, xyz);
      freezing_nodes.push_back(n);
    }
  }

  // get values concentrations and temperature at those nodes
  vec_and_ptr_array_t cl_new(num_comps_, history_front_phi_.vec);
  vec_and_ptr_array_t cl_old(num_comps_, history_front_phi_.vec);

  vec_and_ptr_t tl_new(history_front_phi_.vec);
  vec_and_ptr_t tl_old(history_front_phi_.vec);

  for (i = 0; i < num_comps_; ++i)
  {
    interp.set_input(cl_[1].vec[i], linear); interp.interpolate(cl_old.vec[i]);
    interp.set_input(cl_[0].vec[i], linear); interp.interpolate(cl_new.vec[i]);
  }
  interp.set_input(tl_[1].vec, linear); interp.interpolate(tl_old.vec);
  interp.set_input(tl_[0].vec, linear); interp.interpolate(tl_new.vec);

  interp.set_input(front_curvature_.vec,    linear); interp.interpolate(history_front_curvature_.vec);
  interp.set_input(front_velo_norm_[0].vec, linear); interp.interpolate(history_front_velo_norm_.vec);

  interp.set_input(seed_map_.vec, linear); interp.interpolate(history_seed_.vec);

  cl_old.get_array();
  cl_new.get_array();
  tl_old.get_array();
  tl_new.get_array();

  history_cs_.get_array();
  history_tf_.get_array();

  for (unsigned int i = 0; i < freezing_nodes.size(); ++i)
  {
    p4est_locidx_t n = freezing_nodes[i];

    // estimate the time when the interface crossed a node
    double tau = (fabs(history_front_phi_nm1_.ptr[n])+EPS)
                 /(fabs(history_front_phi_nm1_.ptr[n]) + fabs(history_front_phi_.ptr[n]) + EPS);

    for (int i = 0; i < num_comps_; ++i)
    {
      history_cs_.ptr[i][n] = part_coeff_[i]*(cl_old.ptr[i][n]*(1.-tau) + cl_new.ptr[i][n]*tau);
    }
    history_tf_.ptr[n] = tl_old.ptr[n]*(1.-tau) + tl_new.ptr[n]*tau;
  }

  cl_old.restore_array();
  cl_new.restore_array();
  tl_old.restore_array();
  tl_new.restore_array();

  history_cs_.restore_array();
  history_tf_.restore_array();

  cl_old.destroy();
  cl_new.destroy();
  tl_old.destroy();
  tl_new.destroy();

  my_p4est_level_set_t ls(history_ngbd_);
  VecScaleGhost(history_front_phi_.vec, -1.);
  for (int i = 0; i < num_comps_; ++i)
  {
    ls.extend_Over_Interface_TVD(history_front_phi_.vec, history_cs_.vec[i], 5, 1);
  }
  ls.extend_Over_Interface_TVD(history_front_phi_.vec, history_tf_.vec,  5, 1);
  ls.extend_Over_Interface_TVD(history_front_phi_.vec, history_front_curvature_.vec,  5, 1);
  ls.extend_Over_Interface_TVD(history_front_phi_.vec, history_front_velo_norm_.vec,  5, 1);
  ls.extend_Over_Interface_TVD(history_front_phi_.vec, history_seed_.vec,  5, 0);
  VecScaleGhost(history_front_phi_.vec, -1.);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_solid, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::compute_filtered_curvature()
{
  double smoothing = SQR(curvature_smoothing_*diag_);

  my_p4est_interpolation_nodes_t interp(ngbd_);
  interp.set_input(front_phi_.vec, linear);
  VecCopyGhost(front_phi_.vec, front_curvature_filtered_.vec);

  my_p4est_poisson_nodes_mls_t solver(ngbd_);

  solver.set_mu(smoothing/double(curvature_smoothing_steps_));
  solver.set_diag(1.);
  solver.set_rhs(front_curvature_filtered_.vec);
//  solver.set_wc(neumann_cf, zero_cf);
  solver.set_wc(dirichlet_cf, interp);

  for (int i = 0; i < curvature_smoothing_steps_; ++i)
  {
    solver.solve(front_curvature_filtered_.vec, true);
  }

  VecAXPBYGhost(front_curvature_filtered_.vec, -1., 1., front_phi_.vec);
  VecScaleGhost(front_curvature_filtered_.vec, 1./smoothing);

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(linear);

  ls.extend_from_interface_to_whole_domain_TVD_in_place(front_phi_.vec, front_curvature_filtered_.vec, front_phi_.vec);

}
