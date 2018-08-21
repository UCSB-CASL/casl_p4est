#ifdef P4_TO_P8
#include "my_p8est_multialloy_optimized.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_multialloy_optimized.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
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
extern PetscLogEvent log_my_p4est_multialloy_update_grid;
extern PetscLogEvent log_my_p4est_multialloy_save_vtk;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif




my_p4est_multialloy_t::my_p4est_multialloy_t(my_p4est_node_neighbors_t *ngbd)
  : brick_(ngbd->myb), connectivity_(ngbd->p4est->connectivity), p4est_(ngbd->p4est), ghost_(ngbd->ghost), nodes_(ngbd->nodes), hierarchy_(ngbd->hierarchy), ngbd_(ngbd),
    tl_n_(NULL), tl_np1_(NULL),
    ts_n_(NULL), ts_np1_(NULL),
    c0_n_(NULL), c0_np1_(NULL),
    c1_n_(NULL), c1_np1_(NULL),
    c0s_(NULL),
    c1s_(NULL),
    normal_velocity_n_(NULL),
    normal_velocity_np1_(NULL),
    phi_(NULL),
    kappa_(NULL),
    phi_smooth_(NULL),
    bc_error_(NULL)
{
  /* these default values are for a NiCu alloy, as presented in
   * A Sharp Computational Method for the Simulation of the Solidification of Binary Alloys
   * by Maxime Theillard, Frederic Gibou, Tresa Pollock
   */
//  double rho           = 8.88e-3;  /* kg.cm-3    */
//  double heat_capacity = 4.6e2;    /* J.kg-1.K-1 */
  latent_heat_          = 2350;      /* J.cm-3      */
  thermal_conductivity_ = 6.07e-1;   /* W.cm-1.K-1  */
  thermal_diffusivity_  = 1.486e-1;  /* cm2.s-1     */  /* = thermal_conductivity / (rho*heat_capacity) */

  cooling_velocity_     = 0.01;      /* cm.s-1      */

  Dl0_                  = 1e-5;      /* cm2.s-1     */
  kp0_                  = 0.86;
  c00_                  = 0.20831;   /* at frac.    */
  ml0_                  = -357;      /* K / at frac.*/

  Dl1_                  = 1e-5;      /* cm2.s-1     */
  kp1_                  = 0.86;
  c01_                  = 0.20;      /* at frac.    */
  ml1_                  = -357;      /* K / at frac.*/

  ::dxyz_min(p4est_, dxyz_);
#ifdef P4_TO_P8
  dxyz_min_ = MIN(dxyz_[0],dxyz_[1],dxyz_[2]);
  dxyz_max_ = MAX(dxyz_[0],dxyz_[1],dxyz_[2]);
  diag_ = sqrt(SQR(dxyz_[0])+SQR(dxyz_[1])+SQR(dxyz_[2]));
#else
  dxyz_min_ = MIN(dxyz_[0],dxyz_[1]);
  dxyz_max_ = MAX(dxyz_[0],dxyz_[1]);
  diag_ = sqrt(SQR(dxyz_[0])+SQR(dxyz_[1]));
#endif
//  dxyz_close_interface = 4*dxyz_min;
  dxyz_close_interface_ = 1.2*dxyz_max_;

  eps_c_   = &zero_;
  eps_v_   = &zero_;
  GT_      = &zero_;
  jump_t_  = &zero_;
  jump_tn_ = &zero_;
  c0_flux_ = &zero_;
  c1_flux_ = &zero_;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_interface_n_  [dir] = NULL;
    v_interface_np1_[dir] = NULL;
    normal_[dir] = NULL;
    phi_dd_[dir] = NULL;
  }

  cfl_number_ = 0.1;
  pin_every_n_steps_ = 3;
  bc_tolerance_ = 1.e-5;
  max_iterations_ = 50;
  phi_thresh_ = 0.001;

  time_ = 0;
  time_limit_ = DBL_MAX;

  use_continuous_stencil_    = true;
  use_one_sided_derivatives_ = false;
  use_superconvergent_robin_ = false;
  use_superconvergent_jump_  = false;
  use_points_on_interface_   = true;
  update_c0_robin_           = false;
  zero_negative_velocity_    = false;

  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v1;

  history_p4est_ = p4est_copy(p4est_, P4EST_FALSE);
  history_ghost_ = my_p4est_ghost_new(history_p4est_, P4EST_CONNECT_FULL);
  history_nodes_ = my_p4est_nodes_new(history_p4est_, history_ghost_);

  history_hierarchy_ = new my_p4est_hierarchy_t(history_p4est_, history_ghost_, brick_);
  history_ngbd_      = new my_p4est_node_neighbors_t(history_hierarchy_, history_nodes_);
  history_ngbd_->init_neighbors();

  history_kappa_ = NULL;
  history_velo_  = NULL;

  dendrite_number_ = NULL;
  dendrite_tip_    = NULL;
  dendrite_cut_off_fraction_ = .75;
  dendrite_min_length_ = .05;

  shift_connectivity_ = NULL;
  shift_p4est_        = NULL;

  shift_ghost_ = NULL;
  shift_nodes_ = NULL;

  shift_hierarchy_ = NULL;
  shift_ngbd_      = NULL;

  shift_phi_ = NULL;
  shift_kappa_ = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    shift_normal_[dir] = NULL;
    shift_phi_dd_[dir] = NULL;
  }
}




my_p4est_multialloy_t::~my_p4est_multialloy_t()
{
  if(tl_n_   !=NULL) { ierr = VecDestroy(tl_n_);   CHKERRXX(ierr); }
  if(tl_np1_ !=NULL) { ierr = VecDestroy(tl_np1_); CHKERRXX(ierr); }

  if(ts_n_   !=NULL) { ierr = VecDestroy(ts_n_);   CHKERRXX(ierr); }
  if(ts_np1_ !=NULL) { ierr = VecDestroy(ts_np1_); CHKERRXX(ierr); }

  if(tf_     !=NULL) { ierr = VecDestroy(tf_);     CHKERRXX(ierr); }

  if(c0_n_   !=NULL) { ierr = VecDestroy(c0_n_);   CHKERRXX(ierr); }
  if(c0_np1_ !=NULL) { ierr = VecDestroy(c0_np1_); CHKERRXX(ierr); }
  if(c1_n_   !=NULL) { ierr = VecDestroy(c1_n_);   CHKERRXX(ierr); }
  if(c1_np1_ !=NULL) { ierr = VecDestroy(c1_np1_); CHKERRXX(ierr); }

  if(c0s_ !=NULL) { ierr = VecDestroy(c0s_); CHKERRXX(ierr); }
  if(c1s_ !=NULL) { ierr = VecDestroy(c1s_); CHKERRXX(ierr); }

  if(history_kappa_ !=NULL) { ierr = VecDestroy(history_kappa_); CHKERRXX(ierr); }
  if(history_velo_  !=NULL) { ierr = VecDestroy(history_velo_);  CHKERRXX(ierr); }

  if(normal_velocity_n_   !=NULL) { ierr = VecDestroy(normal_velocity_n_);   CHKERRXX(ierr); }
  if(normal_velocity_np1_ !=NULL) { ierr = VecDestroy(normal_velocity_np1_); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(v_interface_n_  [dir] != NULL) { ierr = VecDestroy(v_interface_n_  [dir]); CHKERRXX(ierr); }
    if(v_interface_np1_[dir] != NULL) { ierr = VecDestroy(v_interface_np1_[dir]); CHKERRXX(ierr); }
    if(normal_         [dir] != NULL) { ierr = VecDestroy(normal_         [dir]); CHKERRXX(ierr); }
    if(phi_dd_         [dir] != NULL) { ierr = VecDestroy(phi_dd_         [dir]); CHKERRXX(ierr); }
  }

  if(phi_  !=NULL) { ierr = VecDestroy(phi_);   CHKERRXX(ierr); }
  if(kappa_!=NULL) { ierr = VecDestroy(kappa_); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(shift_normal_ [dir] != NULL) { ierr = VecDestroy(shift_normal_ [dir]); CHKERRXX(ierr); }
    if(shift_phi_dd_ [dir] != NULL) { ierr = VecDestroy(shift_phi_dd_ [dir]); CHKERRXX(ierr); }
  }

  if(shift_phi_  !=NULL) { ierr = VecDestroy(shift_phi_);   CHKERRXX(ierr); }
  if(shift_kappa_!=NULL) { ierr = VecDestroy(shift_kappa_); CHKERRXX(ierr); }

  if(history_phi_     !=NULL) { ierr = VecDestroy(history_phi_);     CHKERRXX(ierr); }
  if(history_phi_nm1_ !=NULL) { ierr = VecDestroy(history_phi_nm1_); CHKERRXX(ierr); }
//#ifdef P4_TO_P8
//  if(theta_xz_!=NULL) { ierr = VecDestroy(theta_xz_); CHKERRXX(ierr); }
//  if(theta_yz_!=NULL) { ierr = VecDestroy(theta_yz_); CHKERRXX(ierr); }
//#else
//  if(theta_!=NULL) { ierr = VecDestroy(theta_); CHKERRXX(ierr); }
//#endif

  if(bc_error_ !=NULL) { ierr = VecDestroy(bc_error_);   CHKERRXX(ierr); }

  if(phi_smooth_ != NULL) { ierr = VecDestroy(phi_smooth_); CHKERRXX(ierr); }

  if(dendrite_number_ != NULL) { ierr = VecDestroy(dendrite_number_); CHKERRXX(ierr); }
  if(dendrite_tip_    != NULL) { ierr = VecDestroy(dendrite_tip_);    CHKERRXX(ierr); }

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

  my_p4est_brick_destroy(connectivity_, brick_);

  delete shift_ngbd_;
  delete shift_hierarchy_;
  p4est_nodes_destroy(shift_nodes_);
  p4est_ghost_destroy(shift_ghost_);
  p4est_destroy      (shift_p4est_);

  my_p4est_brick_destroy(shift_connectivity_, &shift_brick_);
}

void my_p4est_multialloy_t::set_normal_velocity(Vec v)
{
  normal_velocity_np1_ = v;

  Vec src;
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(v, &v_interface_np1_[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(v_interface_np1_[dir], &src); CHKERRXX(ierr);
    ierr = VecSet(src, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(v_interface_np1_[dir], &src); CHKERRXX(ierr);

    ierr = VecDuplicate(v, &v_interface_n_[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(v_interface_n_[dir], &src); CHKERRXX(ierr);
    ierr = VecSet(src, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(v_interface_n_[dir], &src); CHKERRXX(ierr);
  }


  ierr = VecCreateGhostNodes(history_p4est_, history_nodes_, &history_velo_); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  for(size_t n = 0; n < history_nodes_->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(normal_velocity_np1_, interpolation_between_grids_);
  interp.interpolate(history_velo_);
}




void my_p4est_multialloy_t::compute_geometric_properties()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);

  /* second order derivatives */
  foreach_dimension(dim)
  {
    if (phi_dd_[dim] != NULL) { ierr = VecDestroy(phi_dd_[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_dd_[dim]); CHKERRXX(ierr);
  }
  ngbd_->second_derivatives_central(phi_, phi_dd_);

  /* normal and curvature */
  foreach_dimension(dim)
  {
    if (normal_[dim] != NULL) { ierr = VecDestroy(normal_[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &normal_[dim]); CHKERRXX(ierr);
  }

  if (kappa_ != NULL) { ierr = VecDestroy(kappa_); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_, nodes_, &kappa_); CHKERRXX(ierr);

  compute_normals_and_mean_curvature(*ngbd_, phi_, normal_, kappa_);

  vec_and_ptr_t kappa_tmp;

  ierr = VecDuplicate(kappa_, &kappa_tmp.vec); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  ls.extend_from_interface_to_whole_domain_TVD(phi_, kappa_, kappa_tmp.vec);

//  kappa_tmp.get_array();

//  double kappa_max = 1./dxyz_min_;

//  foreach_node(n, nodes_)
//  {
//    if      (kappa_tmp.ptr[n] > kappa_max) kappa_tmp.ptr[n] = kappa_max;
//    else if (kappa_tmp.ptr[n] <-kappa_max) kappa_tmp.ptr[n] =-kappa_max;
//  }

//  kappa_tmp.restore_array();

  ierr = VecDestroy(kappa_); CHKERRXX(ierr);

  kappa_ = kappa_tmp.vec;

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::compute_velocity()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);

  Vec c_interface; ierr = VecDuplicate(phi_, &c_interface); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0_np1_, c_interface);

  Vec v_gamma[P4EST_DIM];
  double *v_gamma_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(normal_[dir], &v_gamma[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
  }

  Vec vn;
  ierr = VecDuplicate(phi_, &vn); CHKERRXX(ierr);

  double *c0_np1_p, *c_interface_p;
  ierr = VecGetArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);

  double *normal_p[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGetArray(normal_[dim], &normal_p[dim]); CHKERRXX(ierr);
  }

  double *vn_p;
  ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);

  double xyz[P4EST_DIM];

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);

    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    double c0_flux_val = c0_flux_->value(xyz);

    v_gamma_p[0][n] = (c0_flux_val*normal_p[0][n]-qnnn.dx_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    v_gamma_p[1][n] = (c0_flux_val*normal_p[1][n]-qnnn.dy_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = (c0_flux_val*normal_p[2][n]-qnnn.dz_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n] + v_gamma_p[2][n]*normal_p[2][n]);
#else
    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n]);
#endif
  }

  ierr = VecGhostUpdateBegin(vn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_local_node(i);
    qnnn = ngbd_->get_neighbors(n);

    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    double c0_flux_val = c0_flux_->value(xyz);

    v_gamma_p[0][n] = (c0_flux_val*normal_p[0][n]-qnnn.dx_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    v_gamma_p[1][n] = (c0_flux_val*normal_p[1][n]-qnnn.dy_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = (c0_flux_val*normal_p[2][n]-qnnn.dz_central(c0_np1_p))*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n] + v_gamma_p[2][n]*normal_p[2][n]);
#else
    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n]);
#endif
  }

  ierr = VecGhostUpdateEnd(vn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(vn, &vn_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);

  ierr = VecDestroy(c_interface); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecRestoreArray(normal_[dim], &normal_p[dim]); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);

    if (v_interface_np1_[dir] != NULL) { ierr = VecDestroy(v_interface_np1_[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_[dir], &v_interface_np1_[dir]); CHKERRXX(ierr);

    ls.extend_from_interface_to_whole_domain_TVD(phi_, v_gamma[dir], v_interface_np1_[dir]);
//    ls.extend_from_interface_to_whole_domain_TVD(phi_, phi_smooth_, v_gamma[dir], v_interface_np1_[dir]);
    ierr = VecDestroy(v_gamma[dir]); CHKERRXX(ierr);
  }

  if (normal_velocity_np1_ != NULL) { ierr = VecDestroy(normal_velocity_np1_); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_, &normal_velocity_np1_); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi_, vn, normal_velocity_np1_);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, phi_smooth_, vn, normal_velocity_np1_);
  ierr = VecDestroy(vn); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
}






void my_p4est_multialloy_t::compute_dt()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute the size of smallest detail, i.e. maximum curvature */
  const double *kappa_p;
  ierr = VecGetArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);

  const double *phi_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  const double *vn_p;
//  ierr = VecGetArrayRead(normal_velocity_n_, &vn_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normal_velocity_np1_, &vn_p); CHKERRXX(ierr);

  double kappa_max = 0;
  vgamma_max_ = 0;
  double kv_max = 0;

  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
    if(fabs(phi_p[n]) < dxyz_close_interface_)
    {
      vgamma_max_ = MAX(vgamma_max_, fabs(vn_p[n]));
      kappa_max = MAX(kappa_max, fabs(kappa_p[n]));
      kv_max = MAX(kv_max, fabs(MIN(kappa_max, 1./dxyz_min_)*vn_p[n]));
    }

//  ierr = VecRestoreArrayRead(normal_velocity_n_, &vn_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normal_velocity_np1_, &vn_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);

  kappa_max = MIN(kappa_max, 1./dxyz_min_);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &kappa_max,   1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &vgamma_max_, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &kv_max,      1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = PetscPrintf(p4est_->mpicomm, "Maximum curvature = %e\n", kappa_max); CHKERRXX(ierr);

  /* compute the maximum velocity at the interface */
  const double *v_interface_np1_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  double u_max = 0;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
//    ierr = VecGetArrayRead(v_interface_np1_[dir], &v_interface_np1_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(v_interface_n_[dir], &v_interface_np1_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
      if(fabs(phi_p[n]) < dxyz_close_interface_)
        u_max = MAX(u_max, fabs(v_interface_np1_p[n]));

    ierr = VecRestoreArrayRead(v_interface_n_[dir], &v_interface_np1_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(v_interface_np1_[dir], &v_interface_np1_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  dt_nm1_ = dt_n_;
//  dt_n_ = cfl_number_ * MIN(dxyz_min_/vgamma_max_, dxyz_min_/cooling_velocity_, 1./MAX(kv_max, 1.e-6*cooling_velocity_/dxyz_min_)) ;
  dt_n_ = cfl_number_ * MIN(dxyz_min_/vgamma_max_, 1./MAX(kv_max, 1.e-6*cooling_velocity_/dxyz_min_)) ;

  double dt_bl = cfl_number_ * Dl0_/pow(vgamma_max_, 2.)/(1.-kp0_);

  PetscPrintf(p4est_->mpicomm, "VMAX = %e, VGAMMAMAX = %e, COOLING_VELO = %e, %e, %e\n", u_max, vgamma_max_, cooling_velocity_, dxyz_min_/vgamma_max_, 1./kv_max);
  PetscPrintf(p4est_->mpicomm, "dt = %e, dt2 = %e\n", dt_n_, dt_bl);
  PetscPrintf(p4est_->mpicomm, "Solutal layer = %e, nodes per solutal layer = %e\n", Dl0_/(1.-kp0_)/vgamma_max_, Dl0_/(1.-kp0_)/vgamma_max_/dxyz_min_);

  dt_n_ = MIN(dt_n_, dt_bl);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_multialloy_t::update_grid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid... ");

  if (phi_grid_refinement_ != 0 || shift_grids_)
  {
    p4est_t *p4est_np1 = p4est_copy(shift_p4est_, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    my_p4est_semi_lagrangian_t shift_sl(&p4est_np1, &nodes_np1, &ghost_np1, shift_ngbd_, shift_ngbd_);

    shift_sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2);
    shift_sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);

    shift_sl.set_ngbd_v(ngbd_);

    shift_sl.update_p4est(v_interface_np1_, dt_n_, shift_phi_);

    p4est_destroy(shift_p4est_);       shift_p4est_ = p4est_np1;
    p4est_ghost_destroy(shift_ghost_); shift_ghost_ = ghost_np1;
    p4est_nodes_destroy(shift_nodes_); shift_nodes_ = nodes_np1;
    shift_hierarchy_->update(shift_p4est_, shift_ghost_);
    shift_ngbd_->update(shift_hierarchy_, shift_nodes_);

    my_p4est_level_set_t shift_ls(shift_ngbd_);
    shift_ls.reinitialize_1st_order_time_2nd_order_space(shift_phi_);
  }

  // advect interface and update p4est
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_, ngbd_);

  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2);
  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);

  if (1) sl.update_p4est(v_interface_np1_, dt_n_, phi_);
  else   sl.update_p4est(v_interface_n_, v_interface_np1_, dt_nm1_, dt_n_, phi_);

  // expand ghost layer if needed
  if (use_continuous_stencil_ || use_one_sided_derivatives_)
  {
    // reset ghost and nodes
    p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_np1, ghost_np1);
    p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // reset phi
    Vec phi_tmp;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_tmp); CHKERRXX(ierr);

    double *phi_p;     ierr = VecGetArray(phi_,    &phi_p);     CHKERRXX(ierr);
    double *phi_tmp_p; ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
    {
      phi_tmp_p[n] = phi_p[n];
    }

    ierr = VecRestoreArray(phi_,    &phi_p);     CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(phi_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (phi_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(phi_); CHKERRXX(ierr);
    phi_ = phi_tmp;
  }

  /* interpolate the quantities onto the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  /* temperature */
  Vec tl_n_tmp;
  ierr = VecDuplicate(phi_, &tl_n_tmp); CHKERRXX(ierr);
  interp.set_input(tl_n_, interpolation_between_grids_);
  interp.interpolate(tl_n_tmp);
  ierr = VecDestroy(tl_n_); CHKERRXX(ierr);
  tl_n_ = tl_n_tmp;

  Vec ts_n_tmp;
  ierr = VecDuplicate(phi_, &ts_n_tmp); CHKERRXX(ierr);
  interp.set_input(ts_n_, interpolation_between_grids_);
  interp.interpolate(ts_n_tmp);
  ierr = VecDestroy(ts_n_); CHKERRXX(ierr);
  ts_n_ = ts_n_tmp;

  Vec tl_np1_tmp;
  ierr = VecDuplicate(phi_, &tl_np1_tmp); CHKERRXX(ierr);
  interp.set_input(tl_np1_, interpolation_between_grids_);
  interp.interpolate(tl_np1_tmp);
  ierr = VecDestroy(tl_np1_); CHKERRXX(ierr);
  tl_np1_ = tl_np1_tmp;

  Vec ts_np1_tmp;
  ierr = VecDuplicate(phi_, &ts_np1_tmp); CHKERRXX(ierr);
  interp.set_input(ts_np1_, interpolation_between_grids_);
  interp.interpolate(ts_np1_tmp);
  ierr = VecDestroy(ts_np1_); CHKERRXX(ierr);
  ts_np1_ = ts_np1_tmp;

  /* concentrartions */
  Vec c0_n_tmp;
  ierr = VecDuplicate(phi_, &c0_n_tmp); CHKERRXX(ierr);
  interp.set_input(c0_n_, interpolation_between_grids_);
  interp.interpolate(c0_n_tmp);
  ierr = VecDestroy(c0_n_); CHKERRXX(ierr);
  c0_n_ = c0_n_tmp;

  Vec c0_np1_tmp;
  ierr = VecDuplicate(phi_, &c0_np1_tmp); CHKERRXX(ierr);
  interp.set_input(c0_np1_, interpolation_between_grids_);
  interp.interpolate(c0_np1_tmp);
  ierr = VecDestroy(c0_np1_); CHKERRXX(ierr);
  c0_np1_ = c0_np1_tmp;

  Vec c1_n_tmp;
  ierr = VecDuplicate(phi_, &c1_n_tmp); CHKERRXX(ierr);
  interp.set_input(c1_n_, interpolation_between_grids_);
  interp.interpolate(c1_n_tmp);
  ierr = VecDestroy(c1_n_); CHKERRXX(ierr);
  c1_n_ = c1_n_tmp;

  Vec c1_np1_tmp;
  ierr = VecDuplicate(phi_, &c1_np1_tmp); CHKERRXX(ierr);
  interp.set_input(c1_np1_, interpolation_between_grids_);
  interp.interpolate(c1_np1_tmp);
  ierr = VecDestroy(c1_np1_); CHKERRXX(ierr);
  c1_np1_ = c1_np1_tmp;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(v_interface_n_[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_, &v_interface_n_[dir]); CHKERRXX(ierr);
    interp.set_input(v_interface_np1_[dir], interpolation_between_grids_);
    interp.interpolate(v_interface_n_[dir]);
    ierr = VecDestroy(v_interface_np1_[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_, &v_interface_np1_[dir]); CHKERRXX(ierr);
  }

  ierr = VecDestroy(normal_velocity_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &normal_velocity_n_); CHKERRXX(ierr);
  interp.set_input(normal_velocity_np1_, interpolation_between_grids_);
  interp.interpolate(normal_velocity_n_);

  ierr = VecDestroy(normal_velocity_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &normal_velocity_np1_); CHKERRXX(ierr);
  copy_ghosted_vec(normal_velocity_n_, normal_velocity_np1_);

  p4est_destroy(p4est_);       p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  if (phi_grid_refinement_ != 0 || shift_grids_)
  {
    my_p4est_interpolation_nodes_t interp_bwd(shift_ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes_)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      interp_bwd.add_point(n, xyz);
    }

    interp_bwd.set_input(shift_phi_, interpolation_between_grids_);
    interp_bwd.interpolate(phi_);
  }

  /* reinitialize phi */
  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_);

  /* second derivatives, normals, curvature, angles */
  compute_geometric_properties();

  /* refine history_p4est_ */
  if (1)
  {
    p4est_t       *history_p4est_np1 = p4est_copy(history_p4est_, P4EST_FALSE);
    p4est_ghost_t *history_ghost_np1 = my_p4est_ghost_new(history_p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *history_nodes_np1 = my_p4est_nodes_new(history_p4est_np1, history_ghost_np1);

    splitting_criteria_t* sp_old = (splitting_criteria_t*)history_ngbd_->p4est->user_pointer;

    Vec phi_np1;
    ierr = VecCreateGhostNodes(history_p4est_np1, history_nodes_np1, &phi_np1); CHKERRXX(ierr);

    bool is_grid_changing = true;

    int counter = 0;

    while (is_grid_changing)
    {
      // interpolate from a coarse grid to a fine one
      double* phi_np1_p;
      ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      my_p4est_interpolation_nodes_t interp(ngbd_);

      double xyz[P4EST_DIM];
      for(size_t n = 0; n < history_nodes_np1->indep_nodes.elem_count; ++n)
      {
        node_xyz_fr_n(n, history_p4est_np1, history_nodes_np1, xyz);
        interp.add_point(n, xyz);
      }

      interp.set_input(phi_, interpolation_between_grids_);
      interp.interpolate(phi_np1);

      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
      is_grid_changing = sp.refine(history_p4est_np1, history_nodes_np1, phi_np1_p);

      ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      if (is_grid_changing)
      {
        my_p4est_partition(history_p4est_np1, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(history_ghost_np1); history_ghost_np1 = my_p4est_ghost_new(history_p4est_np1, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(history_nodes_np1); history_nodes_np1 = my_p4est_nodes_new(history_p4est_np1, history_ghost_np1);

        ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(history_p4est_np1, history_nodes_np1, &phi_np1); CHKERRXX(ierr);
      }

      counter++;
    }

    history_p4est_np1->user_pointer = (void*) sp_old;

    ierr = VecDestroy(history_phi_); CHKERRXX(ierr);
    history_phi_ = phi_np1;

    // transfer variables to the new grid
    my_p4est_interpolation_nodes_t history_interp(history_ngbd_);

    double xyz[P4EST_DIM];
    for(size_t n = 0; n < history_nodes_np1->indep_nodes.elem_count; ++n)
    {
      node_xyz_fr_n(n, history_p4est_np1, history_nodes_np1, xyz);
      history_interp.add_point(n, xyz);
    }

    Vec tf_tmp;
    ierr = VecDuplicate(history_phi_, &tf_tmp); CHKERRXX(ierr);
    history_interp.set_input(tf_, interpolation_between_grids_);
    history_interp.interpolate(tf_tmp);
    ierr = VecDestroy(tf_); CHKERRXX(ierr);
    tf_ = tf_tmp;

    Vec c0s_tmp;
    ierr = VecDuplicate(history_phi_, &c0s_tmp); CHKERRXX(ierr);
    history_interp.set_input(c0s_, interpolation_between_grids_);
    history_interp.interpolate(c0s_tmp);
    ierr = VecDestroy(c0s_); CHKERRXX(ierr);
    c0s_ = c0s_tmp;

    Vec c1s_tmp;
    ierr = VecDuplicate(history_phi_, &c1s_tmp); CHKERRXX(ierr);
    history_interp.set_input(c1s_, interpolation_between_grids_);
    history_interp.interpolate(c1s_tmp);
    ierr = VecDestroy(c1s_); CHKERRXX(ierr);
    c1s_ = c1s_tmp;

    Vec history_kappa_tmp;
    ierr = VecDuplicate(history_phi_, &history_kappa_tmp); CHKERRXX(ierr);
    history_interp.set_input(history_kappa_, interpolation_between_grids_);
    history_interp.interpolate(history_kappa_tmp);
    ierr = VecDestroy(history_kappa_); CHKERRXX(ierr);
    history_kappa_ = history_kappa_tmp;

    Vec history_velo_tmp;
    ierr = VecDuplicate(history_phi_, &history_velo_tmp); CHKERRXX(ierr);
    history_interp.set_input(history_velo_, interpolation_between_grids_);
    history_interp.interpolate(history_velo_tmp);
    ierr = VecDestroy(history_velo_); CHKERRXX(ierr);
    history_velo_ = history_velo_tmp;

    Vec phi_nm1_tmp;
    ierr = VecDuplicate(history_phi_, &phi_nm1_tmp); CHKERRXX(ierr);
    history_interp.set_input(history_phi_nm1_, interpolation_between_grids_);
    history_interp.interpolate(phi_nm1_tmp);
    ierr = VecDestroy(history_phi_nm1_); CHKERRXX(ierr);
    history_phi_nm1_ = phi_nm1_tmp;

    p4est_destroy(history_p4est_);       history_p4est_ = history_p4est_np1;
    p4est_ghost_destroy(history_ghost_); history_ghost_ = history_ghost_np1;
    p4est_nodes_destroy(history_nodes_); history_nodes_ = history_nodes_np1;
    history_hierarchy_->update(history_p4est_, history_ghost_);
    history_ngbd_->update(history_hierarchy_, history_nodes_);

    // get new values for nodes that just solidified
    double *phi_nm1_p; ierr = VecGetArray(history_phi_nm1_, &phi_nm1_p); CHKERRXX(ierr);
    double *phi_p;     ierr = VecGetArray(history_phi_,     &phi_p);     CHKERRXX(ierr);

    double *c0s_p; ierr = VecGetArray(c0s_, &c0s_p); CHKERRXX(ierr);
    double *c1s_p; ierr = VecGetArray(c1s_, &c1s_p); CHKERRXX(ierr);
    double *tf_p;  ierr = VecGetArray(tf_, &tf_p);   CHKERRXX(ierr);

    my_p4est_interpolation_nodes_t interp(ngbd_);

    std::vector<p4est_locidx_t> freezing_nodes;

    // first, figure out which nodes just became solid
    foreach_node(n, history_nodes_)
    {
      if (phi_p[n] > 0 && phi_nm1_p[n] < 0)
      {
        node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
        interp.add_point(n, xyz);
        freezing_nodes.push_back(n);
      }
    }

    // get values concentrations and temperature at those nodes
    Vec c0_new; ierr = VecDuplicate(history_phi_, &c0_new); CHKERRXX(ierr);
    Vec c0_old; ierr = VecDuplicate(history_phi_, &c0_old); CHKERRXX(ierr);
    Vec c1_new; ierr = VecDuplicate(history_phi_, &c1_new); CHKERRXX(ierr);
    Vec c1_old; ierr = VecDuplicate(history_phi_, &c1_old); CHKERRXX(ierr);
    Vec tl_new; ierr = VecDuplicate(history_phi_, &tl_new); CHKERRXX(ierr);
    Vec tl_old; ierr = VecDuplicate(history_phi_, &tl_old); CHKERRXX(ierr);

    interp.set_input(c0_n_,   interpolation_between_grids_); interp.interpolate(c0_old);
    interp.set_input(c0_np1_, interpolation_between_grids_); interp.interpolate(c0_new);
    interp.set_input(c1_n_,   interpolation_between_grids_); interp.interpolate(c1_old);
    interp.set_input(c1_np1_, interpolation_between_grids_); interp.interpolate(c1_new);
    interp.set_input(tl_n_,   interpolation_between_grids_); interp.interpolate(tl_old);
    interp.set_input(tl_np1_, interpolation_between_grids_); interp.interpolate(tl_new);

    interp.set_input(kappa_,  interpolation_between_grids_); interp.interpolate(history_kappa_);

    interp.set_input(normal_velocity_np1_,  interpolation_between_grids_); interp.interpolate(history_velo_);

    double *c0_old_p; ierr = VecGetArray(c0_old, &c0_old_p); CHKERRXX(ierr);
    double *c0_new_p; ierr = VecGetArray(c0_new, &c0_new_p); CHKERRXX(ierr);
    double *c1_old_p; ierr = VecGetArray(c1_old, &c1_old_p); CHKERRXX(ierr);
    double *c1_new_p; ierr = VecGetArray(c1_new, &c1_new_p); CHKERRXX(ierr);
    double *tl_old_p; ierr = VecGetArray(tl_old, &tl_old_p); CHKERRXX(ierr);
    double *tl_new_p; ierr = VecGetArray(tl_new, &tl_new_p); CHKERRXX(ierr);

    for (unsigned int i = 0; i < freezing_nodes.size(); ++i)
    {
      p4est_locidx_t n = freezing_nodes[i];

      // estimate the time when the interface crossed a node
      double tau = fabs(phi_nm1_p[n])+EPS/(fabs(phi_nm1_p[n]) + fabs(phi_p[n]) + EPS);

      c0s_p[n] = kp0_*(c0_old_p[n]*(1.-tau) + c0_new_p[n]*tau);
      c1s_p[n] = kp1_*(c1_old_p[n]*(1.-tau) + c1_new_p[n]*tau);
      tf_p [n] = tl_old_p[n]*(1.-tau) + tl_new_p[n]*tau;
    }

    ierr = VecRestoreArray(history_phi_nm1_, &phi_nm1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(history_phi_,     &phi_p);     CHKERRXX(ierr);

    ierr = VecRestoreArray(c0s_, &c0s_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c1s_, &c1s_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tf_, &tf_p);   CHKERRXX(ierr);

    ierr = VecRestoreArray(c0_old, &c0_old_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c0_new, &c0_new_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c1_old, &c1_old_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c1_new, &c1_new_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tl_old, &tl_old_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tl_new, &tl_new_p); CHKERRXX(ierr);

    ierr = VecDestroy(c0_new); CHKERRXX(ierr);
    ierr = VecDestroy(c0_old); CHKERRXX(ierr);
    ierr = VecDestroy(c1_new); CHKERRXX(ierr);
    ierr = VecDestroy(c1_old); CHKERRXX(ierr);
    ierr = VecDestroy(tl_new); CHKERRXX(ierr);
    ierr = VecDestroy(tl_old); CHKERRXX(ierr);

    copy_ghosted_vec(history_phi_, history_phi_nm1_);

    my_p4est_level_set_t ls(history_ngbd_);

    invert_phi(history_nodes_, history_phi_);
    ls.extend_Over_Interface_TVD(history_phi_,c0s_, 5, 1);
    ls.extend_Over_Interface_TVD(history_phi_,c1s_, 5, 1);
    ls.extend_Over_Interface_TVD(history_phi_,tf_,  5, 1);
    ls.extend_Over_Interface_TVD(history_phi_,history_kappa_,  5, 1);
    ls.extend_Over_Interface_TVD(history_phi_,history_velo_,  5, 1);
    invert_phi(history_nodes_, history_phi_);
  }

  copy_ghosted_vec(c0_np1_, c0_n_);
  copy_ghosted_vec(c1_np1_, c1_n_);
  copy_ghosted_vec(tl_np1_, tl_n_);
  copy_ghosted_vec(ts_np1_, ts_n_);

  PetscPrintf(p4est_->mpicomm, "Done \n");

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);
}

int my_p4est_multialloy_t::one_step()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);

  time_ += dt_n_;

  GT_->t      = time_;
  jump_t_->t  = time_;
  jump_tn_->t = time_;
  c0_flux_->t = time_;
  c1_flux_->t = time_;

  rhs_tl_->t  = time_;
  rhs_ts_->t  = time_;
  rhs_c0_->t  = time_;
  rhs_c1_->t  = time_;

  Vec rhs_tl; ierr = VecDuplicate(tl_n_, &rhs_tl); CHKERRXX(ierr); copy_ghosted_vec(tl_n_, rhs_tl);
  Vec rhs_ts; ierr = VecDuplicate(ts_n_, &rhs_ts); CHKERRXX(ierr); copy_ghosted_vec(ts_n_, rhs_ts);
  Vec rhs_c0; ierr = VecDuplicate(c0_n_, &rhs_c0); CHKERRXX(ierr); copy_ghosted_vec(c0_n_, rhs_c0);
  Vec rhs_c1; ierr = VecDuplicate(c1_n_, &rhs_c1); CHKERRXX(ierr); copy_ghosted_vec(c1_n_, rhs_c1);

  double xyz[P4EST_DIM];

  double *rhs_tl_p; ierr = VecGetArray(rhs_tl, &rhs_tl_p); CHKERRXX(ierr);
  double *rhs_ts_p; ierr = VecGetArray(rhs_ts, &rhs_ts_p); CHKERRXX(ierr);
  double *rhs_c0_p; ierr = VecGetArray(rhs_c0, &rhs_c0_p); CHKERRXX(ierr);
  double *rhs_c1_p; ierr = VecGetArray(rhs_c1, &rhs_c1_p); CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    rhs_tl_p[n] += dt_n_*rhs_tl_->value(xyz);
    rhs_ts_p[n] += dt_n_*rhs_ts_->value(xyz);
    rhs_c0_p[n] += dt_n_*rhs_c0_->value(xyz);
    rhs_c1_p[n] += dt_n_*rhs_c1_->value(xyz);
  }

  ierr = VecRestoreArray(rhs_tl, &rhs_tl_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_ts, &rhs_ts_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_c0, &rhs_c0_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_c1, &rhs_c1_p); CHKERRXX(ierr);

  // solve coupled system of equations
  my_p4est_poisson_nodes_multialloy_t solver_all_in_one(ngbd_);
  solver_all_in_one.set_phi(phi_, phi_dd_, normal_, kappa_);
  solver_all_in_one.set_parameters(dt_n_, thermal_diffusivity_, thermal_conductivity_, latent_heat_, Tm_, Dl0_, kp0_, ml0_, Dl1_, kp1_, ml1_);
  solver_all_in_one.set_bc(bc_t_, bc_c0_, bc_c1_);
  solver_all_in_one.set_pin_every_n_steps(pin_every_n_steps_);
  solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
  solver_all_in_one.set_rhs(rhs_tl, rhs_ts, rhs_c0, rhs_c1);

  solver_all_in_one.set_use_continuous_stencil   (use_continuous_stencil_   );
  solver_all_in_one.set_use_one_sided_derivatives(use_one_sided_derivatives_);
  solver_all_in_one.set_use_points_on_interface  (use_points_on_interface_  );
  solver_all_in_one.set_update_c0_robin          (update_c0_robin_          );
  solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin_);
  solver_all_in_one.set_zero_negative_velocity   (zero_negative_velocity_);

  solver_all_in_one.set_GT(*GT_);
  solver_all_in_one.set_jump_t(*jump_t_);
  solver_all_in_one.set_jump_tn(*jump_tn_);
  solver_all_in_one.set_flux_c(*c0_flux_, *c1_flux_);
  solver_all_in_one.set_undercoolings(*eps_v_, *eps_c_);

  my_p4est_interpolation_nodes_t interp_c0_n(ngbd_);
  interp_c0_n.set_input(c0_n_, linear);

  solver_all_in_one.set_c0_guess(interp_c0_n);

  if (bc_error_ != NULL) { ierr = VecDestroy(bc_error_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &bc_error_); CHKERRXX(ierr);

//  copy_ghosted_vec(t_n_, t_np1_);
//  copy_ghosted_vec(c0_n_, c0_np1_);
//  copy_ghosted_vec(c1_n_, c1_np1_);

//  int one_step_iterations = solver_all_in_one.solve(t_np1_, c0_np1_, c1_np1_, bc_error_, bc_error_max_, dt_n_, cfl_number_);
  int one_step_iterations = solver_all_in_one.solve(tl_np1_, ts_np1_, c0_np1_, c1_np1_, bc_error_, bc_error_max_, dt_n_, 1.e6);

  // correct by a shifted grid
  if (0)
  {
    Vec shift_tl_n_; ierr = VecDuplicate(shift_phi_, &shift_tl_n_); CHKERRXX(ierr);
    Vec shift_ts_n_; ierr = VecDuplicate(shift_phi_, &shift_ts_n_); CHKERRXX(ierr);
    Vec shift_c0_n_; ierr = VecDuplicate(shift_phi_, &shift_c0_n_); CHKERRXX(ierr);
    Vec shift_c1_n_; ierr = VecDuplicate(shift_phi_, &shift_c1_n_); CHKERRXX(ierr);

    Vec shift_tl_np1_; ierr = VecDuplicate(shift_phi_, &shift_tl_np1_); CHKERRXX(ierr);
    Vec shift_ts_np1_; ierr = VecDuplicate(shift_phi_, &shift_ts_np1_); CHKERRXX(ierr);
    Vec shift_c0_np1_; ierr = VecDuplicate(shift_phi_, &shift_c0_np1_); CHKERRXX(ierr);
    Vec shift_c1_np1_; ierr = VecDuplicate(shift_phi_, &shift_c1_np1_); CHKERRXX(ierr);

    Vec shift_bc_error_; ierr = VecDuplicate(shift_phi_, &shift_bc_error_); CHKERRXX(ierr);

    my_p4est_interpolation_nodes_t interp_fwd(ngbd_);

    double xyz[P4EST_DIM];

    foreach_node(n, shift_nodes_)
    {
      node_xyz_fr_n(n, shift_p4est_, shift_nodes_, xyz);
      interp_fwd.add_point(n, xyz);
    }

    interp_fwd.set_input(tl_n_, interpolation_between_grids_); interp_fwd.interpolate(shift_tl_n_);
    interp_fwd.set_input(ts_n_, interpolation_between_grids_); interp_fwd.interpolate(shift_ts_n_);
    interp_fwd.set_input(c0_n_, interpolation_between_grids_); interp_fwd.interpolate(shift_c0_n_);
    interp_fwd.set_input(c1_n_, interpolation_between_grids_); interp_fwd.interpolate(shift_c1_n_);

    Vec rhs_tl; ierr = VecDuplicate(shift_tl_n_, &rhs_tl); CHKERRXX(ierr); copy_ghosted_vec(shift_tl_n_, rhs_tl);
    Vec rhs_ts; ierr = VecDuplicate(shift_ts_n_, &rhs_ts); CHKERRXX(ierr); copy_ghosted_vec(shift_ts_n_, rhs_ts);
    Vec rhs_c0; ierr = VecDuplicate(shift_c0_n_, &rhs_c0); CHKERRXX(ierr); copy_ghosted_vec(shift_c0_n_, rhs_c0);
    Vec rhs_c1; ierr = VecDuplicate(shift_c1_n_, &rhs_c1); CHKERRXX(ierr); copy_ghosted_vec(shift_c1_n_, rhs_c1);

    double *rhs_tl_p; ierr = VecGetArray(rhs_tl, &rhs_tl_p); CHKERRXX(ierr);
    double *rhs_ts_p; ierr = VecGetArray(rhs_ts, &rhs_ts_p); CHKERRXX(ierr);
    double *rhs_c0_p; ierr = VecGetArray(rhs_c0, &rhs_c0_p); CHKERRXX(ierr);
    double *rhs_c1_p; ierr = VecGetArray(rhs_c1, &rhs_c1_p); CHKERRXX(ierr);

    foreach_node(n, shift_nodes_)
    {
      node_xyz_fr_n(n, shift_p4est_, shift_nodes_, xyz);
      rhs_tl_p[n] += dt_n_*rhs_tl_->value(xyz);
      rhs_ts_p[n] += dt_n_*rhs_ts_->value(xyz);
      rhs_c0_p[n] += dt_n_*rhs_c0_->value(xyz);
      rhs_c1_p[n] += dt_n_*rhs_c1_->value(xyz);
    }

    ierr = VecRestoreArray(rhs_tl, &rhs_tl_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_ts, &rhs_ts_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_c0, &rhs_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_c1, &rhs_c1_p); CHKERRXX(ierr);

    // solve coupled system of equations
    my_p4est_poisson_nodes_multialloy_t solver_all_in_one(shift_ngbd_);
    solver_all_in_one.set_phi(shift_phi_, shift_phi_dd_, shift_normal_, shift_kappa_);
    solver_all_in_one.set_parameters(dt_n_, thermal_diffusivity_, thermal_conductivity_, latent_heat_, Tm_, Dl0_, kp0_, ml0_, Dl1_, kp1_, ml1_);
    solver_all_in_one.set_bc(bc_t_, bc_c0_, bc_c1_);
    solver_all_in_one.set_pin_every_n_steps(pin_every_n_steps_);
    solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
    solver_all_in_one.set_rhs(rhs_tl, rhs_ts, rhs_c0, rhs_c1);

    solver_all_in_one.set_use_continuous_stencil   (use_continuous_stencil_   );
    solver_all_in_one.set_use_one_sided_derivatives(use_one_sided_derivatives_);
    solver_all_in_one.set_use_points_on_interface  (use_points_on_interface_  );
    solver_all_in_one.set_update_c0_robin          (update_c0_robin_          );
    solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin_);
    solver_all_in_one.set_zero_negative_velocity   (zero_negative_velocity_);

    solver_all_in_one.set_GT(*GT_);
    solver_all_in_one.set_jump_t(*jump_t_);
    solver_all_in_one.set_jump_tn(*jump_tn_);
    solver_all_in_one.set_flux_c(*c0_flux_, *c1_flux_);
    solver_all_in_one.set_undercoolings(*eps_v_, *eps_c_);

    my_p4est_interpolation_nodes_t interp_bwd(shift_ngbd_);
    interp_bwd.set_input(shift_c0_n_, linear);

    solver_all_in_one.set_c0_guess(interp_bwd);

    int one_step_iterations = solver_all_in_one.solve(shift_tl_np1_, shift_ts_np1_, shift_c0_np1_, shift_c1_np1_, shift_bc_error_, bc_error_max_, dt_n_, 1.e6);

    ierr = VecDestroy(rhs_tl); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_ts); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c0); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c1); CHKERRXX(ierr);

    foreach_node(n, nodes_)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      interp_bwd.add_point(n, xyz);
    }

    // interpolate back
    Vec tl_np1_tmp; ierr = VecDuplicate(phi_, &tl_np1_tmp); CHKERRXX(ierr);
    Vec ts_np1_tmp; ierr = VecDuplicate(phi_, &ts_np1_tmp); CHKERRXX(ierr);
    Vec c0_np1_tmp; ierr = VecDuplicate(phi_, &c0_np1_tmp); CHKERRXX(ierr);
    Vec c1_np1_tmp; ierr = VecDuplicate(phi_, &c1_np1_tmp); CHKERRXX(ierr);

    interp_bwd.set_input(shift_tl_np1_, interpolation_between_grids_); interp_bwd.interpolate(tl_np1_tmp);
    interp_bwd.set_input(shift_ts_np1_, interpolation_between_grids_); interp_bwd.interpolate(ts_np1_tmp);
    interp_bwd.set_input(shift_c0_np1_, interpolation_between_grids_); interp_bwd.interpolate(c0_np1_tmp);
    interp_bwd.set_input(shift_c1_np1_, interpolation_between_grids_); interp_bwd.interpolate(c1_np1_tmp);

    ierr = VecDestroy(shift_tl_n_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_ts_n_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_c0_n_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_c1_n_); CHKERRXX(ierr);

    ierr = VecDestroy(shift_tl_np1_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_ts_np1_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_c0_np1_); CHKERRXX(ierr);
    ierr = VecDestroy(shift_c1_np1_); CHKERRXX(ierr);

    ierr = VecDestroy(shift_bc_error_); CHKERRXX(ierr);

    // take average
    double *tl_np1_tmp_p; ierr = VecGetArray(tl_np1_tmp, &tl_np1_tmp_p); CHKERRXX(ierr);
    double *ts_np1_tmp_p; ierr = VecGetArray(ts_np1_tmp, &ts_np1_tmp_p); CHKERRXX(ierr);
    double *c0_np1_tmp_p; ierr = VecGetArray(c0_np1_tmp, &c0_np1_tmp_p); CHKERRXX(ierr);
    double *c1_np1_tmp_p; ierr = VecGetArray(c1_np1_tmp, &c1_np1_tmp_p); CHKERRXX(ierr);

    double *tl_np1_p; ierr = VecGetArray(tl_np1_, &tl_np1_p); CHKERRXX(ierr);
    double *ts_np1_p; ierr = VecGetArray(ts_np1_, &ts_np1_p); CHKERRXX(ierr);
    double *c0_np1_p; ierr = VecGetArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
    double *c1_np1_p; ierr = VecGetArray(c1_np1_, &c1_np1_p); CHKERRXX(ierr);

    foreach_node(n, nodes_)
    {
//      tl_np1_p[n] = .5*(tl_np1_p[n] + tl_np1_tmp_p[n]);
//      ts_np1_p[n] = .5*(ts_np1_p[n] + ts_np1_tmp_p[n]);
//      c0_np1_p[n] = .5*(c0_np1_p[n] + c0_np1_tmp_p[n]);
//      c1_np1_p[n] = .5*(c1_np1_p[n] + c1_np1_tmp_p[n]);
      tl_np1_p[n] = tl_np1_tmp_p[n];
      ts_np1_p[n] = ts_np1_tmp_p[n];
      c0_np1_p[n] = c0_np1_tmp_p[n];
      c1_np1_p[n] = c1_np1_tmp_p[n];
    }

    ierr = VecRestoreArray(tl_np1_tmp, &tl_np1_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(ts_np1_tmp, &ts_np1_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c0_np1_tmp, &c0_np1_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c1_np1_tmp, &c1_np1_tmp_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(tl_np1_, &tl_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(ts_np1_, &ts_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c1_np1_, &c1_np1_p); CHKERRXX(ierr);

    ierr = VecDestroy(tl_np1_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(ts_np1_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(c0_np1_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(c1_np1_tmp); CHKERRXX(ierr);
  }

  // compute velocity
  compute_velocity();

  if (update_c0_robin_) solver_all_in_one.solve_c0_robin();

  ierr = VecDestroy(rhs_tl); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_ts); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_c0); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_c1); CHKERRXX(ierr);

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

#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/multialloy_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], brick_->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/multialloy_lvl_%d_%d_%d_%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], iter);
#endif

  const double *phi_p; ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  const double *tl_p; ierr = VecGetArrayRead(tl_np1_, &tl_p); CHKERRXX(ierr);
  const double *ts_p; ierr = VecGetArrayRead(ts_np1_, &ts_p); CHKERRXX(ierr);
  const double *c0_p; ierr = VecGetArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
  const double *c1_p; ierr = VecGetArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
  double *normal_velocity_np1_p; ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

  const double *dendrite_number_p; ierr = VecGetArrayRead(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  const double *dendrite_tip_p;    ierr = VecGetArrayRead(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

//  const double *c0s_p; ierr = VecGetArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
//  const double *c1s_p; ierr = VecGetArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

//  const double *tf_p; ierr = VecGetArrayRead(tf_, &tf_p); CHKERRXX(ierr);

  const double *kappa_p;
  ierr = VecGetArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);

  const double *bc_error_p;
  ierr = VecGetArrayRead(bc_error_, &bc_error_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] /= scaling_;

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


  my_p4est_vtk_write_all(  p4est_, nodes_, ghost_,
                           P4EST_TRUE, P4EST_TRUE,
                         #ifdef P4_TO_P8
                           10, 1, name,
                         #else
                           10, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
//                           VTK_POINT_DATA, "phi_smooth", phi_smooth_p,
                           VTK_POINT_DATA, "tl", tl_p,
                           VTK_POINT_DATA, "ts", ts_p,
//                           VTK_POINT_DATA, "tf", tf_p,
                           VTK_POINT_DATA, "c0", c0_p,
                           VTK_POINT_DATA, "c1", c1_p,
//                           VTK_POINT_DATA, "c0s", c0s_p,
//                           VTK_POINT_DATA, "c1s", c1s_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_POINT_DATA, "bc_error", bc_error_p,
                           VTK_POINT_DATA, "dendrite_number", dendrite_number_p,
                           VTK_POINT_DATA, "dendrite_tip", dendrite_tip_p,
                           VTK_CELL_DATA , "leaf_level", l_p);


  for(size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] *= scaling_;

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(bc_error_, &bc_error_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(phi_smooth_, &phi_smooth_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(tl_np1_, &tl_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(ts_np1_, &ts_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(tf_, &tf_p); CHKERRXX(ierr);

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

#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/multialloy_solid_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], brick_->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/multialloy_solid_lvl_%d_%d_%d_%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], iter);
#endif

  const double *phi_p; ierr = VecGetArrayRead(history_phi_, &phi_p); CHKERRXX(ierr);
  const double *c0s_p; ierr = VecGetArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
  const double *c1s_p; ierr = VecGetArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

  const double *tf_p; ierr = VecGetArrayRead(tf_, &tf_p); CHKERRXX(ierr);

  const double *kappa_p; ierr = VecGetArrayRead(history_kappa_, &kappa_p); CHKERRXX(ierr);
  const double *velo_p;  ierr = VecGetArrayRead(history_velo_,  &velo_p);  CHKERRXX(ierr);

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


  my_p4est_vtk_write_all(  history_p4est_, history_nodes_, history_ghost_,
                           P4EST_TRUE, P4EST_TRUE,
                         #ifdef P4_TO_P8
                           6, 1, name,
                         #else
                           6, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "tf", tf_p,
                           VTK_POINT_DATA, "c0s", c0s_p,
                           VTK_POINT_DATA, "c1s", c1s_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_POINT_DATA, "vn", velo_p,
                           VTK_CELL_DATA , "leaf_level", l_p);


  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(history_phi_, &phi_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(tf_, &tf_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(history_kappa_, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(history_velo_,  &velo_p);  CHKERRXX(ierr);

  PetscPrintf(history_p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::save_VTK_shift(int iter)
{
  if (phi_grid_refinement_ == 0 && !shift_grids_) return;

  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);

  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(shift_p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name[1000];

#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/multialloy_shift_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, shift_p4est_->mpisize, shift_brick_.nxyztrees[0], shift_brick_.nxyztrees[1], shift_brick_.nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/multialloy_shift_lvl_%d_%d_%d_%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, shift_p4est_->mpisize, shift_brick_.nxyztrees[0], shift_brick_.nxyztrees[1], iter);
#endif

  const double *phi_p; ierr = VecGetArrayRead(shift_phi_, &phi_p); CHKERRXX(ierr);
//  const double *tl_p; ierr = VecGetArrayRead(tl_np1_, &tl_p); CHKERRXX(ierr);
//  const double *ts_p; ierr = VecGetArrayRead(ts_np1_, &ts_p); CHKERRXX(ierr);
//  const double *c0_p; ierr = VecGetArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
//  const double *c1_p; ierr = VecGetArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
//  double *normal_velocity_np1_p; ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

//  const double *dendrite_number_p; ierr = VecGetArrayRead(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
//  const double *dendrite_tip_p;    ierr = VecGetArrayRead(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

//  const double *c0s_p; ierr = VecGetArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
//  const double *c1s_p; ierr = VecGetArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

//  const double *tf_p; ierr = VecGetArrayRead(tf_, &tf_p); CHKERRXX(ierr);

//  const double *kappa_p;
//  ierr = VecGetArrayRead(shift_kappa_, &kappa_p); CHKERRXX(ierr);

//  const double *bc_error_p;
//  ierr = VecGetArrayRead(bc_error_, &bc_error_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//    normal_velocity_np1_p[n] /= scaling_;

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(shift_p4est_, shift_ghost_, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = shift_p4est_->first_local_tree; tree_idx <= shift_p4est_->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(shift_p4est_->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<shift_ghost_->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&shift_ghost_->ghosts, q);
    l_p[shift_p4est_->local_num_quadrants+q] = quad->level;
  }


  my_p4est_vtk_write_all(  shift_p4est_, shift_nodes_, shift_ghost_,
                           P4EST_TRUE, P4EST_TRUE,
                         #ifdef P4_TO_P8
                           1, 1, name,
                         #else
                           1, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
//                           VTK_POINT_DATA, "phi_smooth", phi_smooth_p,
//                           VTK_POINT_DATA, "tl", tl_p,
//                           VTK_POINT_DATA, "ts", ts_p,
//                           VTK_POINT_DATA, "tf", tf_p,
//                           VTK_POINT_DATA, "c0", c0_p,
//                           VTK_POINT_DATA, "c1", c1_p,
//                           VTK_POINT_DATA, "c0s", c0s_p,
//                           VTK_POINT_DATA, "c1s", c1s_p,
//                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
//                           VTK_POINT_DATA, "kappa", kappa_p,
//                           VTK_POINT_DATA, "bc_error", bc_error_p,
//                           VTK_POINT_DATA, "dendrite_number", dendrite_number_p,
//                           VTK_POINT_DATA, "dendrite_tip", dendrite_tip_p,
                           VTK_CELL_DATA , "leaf_level", l_p);


//  for(size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
//    normal_velocity_np1_p[n] *= scaling_;

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(bc_error_, &bc_error_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(shift_phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(phi_smooth_, &phi_smooth_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(tl_np1_, &tl_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(ts_np1_, &ts_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(shift_kappa_, &kappa_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

//  ierr = VecRestoreArrayRead(tf_, &tf_p); CHKERRXX(ierr);

  PetscPrintf(shift_p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::compute_smoothed_phi()
{
  my_p4est_poisson_nodes_t solver(ngbd_);

  int num_smoothing_iterations = 50;
  double mu_smoothing = 3.e-8;

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setWallTypes(wall_bc_smoothing_);
  bc.setWallValues(zero_);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(zero_);

  solver.set_bc(bc);
  solver.set_mu(mu_smoothing);
  solver.set_diagonal(1.0);

  if (phi_smooth_ != NULL) { ierr = VecDestroy(phi_smooth_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &phi_smooth_); CHKERRXX(ierr);

  copy_ghosted_vec(phi_, phi_smooth_);

  Vec rhs;
  ierr = VecDuplicate(phi_, &rhs); CHKERRXX(ierr);

  for (int i = 0; i < num_smoothing_iterations; ++i)
  {
    copy_ghosted_vec(phi_smooth_, rhs);
    solver.set_rhs(rhs);
    solver.solve(phi_smooth_, true);
  }

  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_smooth_);
}

void my_p4est_multialloy_t::count_dendrites(int iter)
{
  // this code assumes dendrites grow in the positive y-direction

  // find boundaries of the "mushy zone"
  double mushy_zone_min = brick_->xyz_max[1];
  double mushy_zone_max = brick_->xyz_min[1];

  const double *phi_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  foreach_local_node(n, nodes_)
  {
    double y_coord = node_y_fr_n(n, p4est_, nodes_);

    if (phi_p[n] > 0 && y_coord > mushy_zone_max) mushy_zone_max = y_coord;
    if (phi_p[n] < 0 && y_coord < mushy_zone_min) mushy_zone_min = y_coord;
  }
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &mushy_zone_min, 1, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &mushy_zone_max, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  if(dendrite_number_ != NULL) { ierr = VecDestroy(dendrite_number_); CHKERRXX(ierr); }
  if(dendrite_tip_    != NULL) { ierr = VecDestroy(dendrite_tip_);    CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &dendrite_number_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &dendrite_tip_);    CHKERRXX(ierr);

  double *dendrite_number_p; ierr = VecGetArray(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  double *dendrite_tip_p;    ierr = VecGetArray(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    dendrite_number_p[n] = -1;
    dendrite_tip_p[n] = -1;
  }

  ierr = VecRestoreArray(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(dendrite_tip_,    &dendrite_tip_p);    CHKERRXX(ierr);

  // cut off denrites' base
  double cut_off_length = mushy_zone_max - dendrite_cut_off_fraction_*MAX(dendrite_min_length_, mushy_zone_max-mushy_zone_min);

  Vec phi_cut;
  ierr = VecDuplicate(phi_, &phi_cut); CHKERRXX(ierr);
  copy_ghosted_vec(phi_, phi_cut);

  double *phi_cut_p;
  ierr = VecGetArray(phi_cut, &phi_cut_p); CHKERRXX(ierr);
  foreach_node(n, nodes_)
  {
    if (node_y_fr_n(n, p4est_, nodes_) < cut_off_length) phi_cut_p[n] = -1;
  }
  ierr = VecRestoreArray(phi_cut, &phi_cut_p); CHKERRXX(ierr);

  // enumerate dendrite by using Arthur's parallel region-growing code
  num_dendrites_ = 0;
  compute_islands_numbers(*ngbd_, phi_cut, num_dendrites_, dendrite_number_);
  ierr = VecDestroy(phi_cut); CHKERRXX(ierr);

  ierr = PetscPrintf(p4est_->mpicomm, "%d prominent dendrites found.\n", num_dendrites_); CHKERRXX(ierr);

  // find the tip of every dendrite
  std::vector<double> x_tip(num_dendrites_, 0);
  std::vector<double> y_tip(num_dendrites_, 0);

  ierr = VecGetArray(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  foreach_local_node(n, nodes_)
  {
    int i = (int) dendrite_number_p[n];
    if (i >= 0)
    {
      const quad_neighbor_nodes_of_node_t qnnn = ngbd_->get_neighbors(n);
      int j = (int) qnnn.f_0p0_linear(dendrite_number_p);
      if (j < 0)
      {
        double phi_n   = phi_p[n];
        double phi_nei = qnnn.f_0p0_linear(phi_p);

        double y_coord = node_y_fr_n(n, p4est_, nodes_) + fabs(phi_n)/(fabs(phi_n)+fabs(phi_nei)) * dxyz_[1];

        if (y_coord > y_tip[i])
        {
          y_tip[i] = y_coord;
          x_tip[i] = node_x_fr_n(n, p4est_, nodes_);
        }
      }
    }
  }
  ierr = VecRestoreArray(dendrite_number_, &dendrite_number_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  std::vector<double> y_tip_g(num_dendrites_, 0);
  std::vector<double> x_tip_g(num_dendrites_, 0);

  mpiret = MPI_Allreduce(y_tip.data(), y_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  for (unsigned int i = 0; i < num_dendrites_; ++i)
  {
    if (y_tip[i] != y_tip_g[i]) x_tip[i] = -10;
  }

  mpiret = MPI_Allreduce(x_tip.data(), x_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  // auxiliary field that shows found tip locations
  ierr = VecGetArray(dendrite_tip_, &dendrite_tip_p); CHKERRXX(ierr);
  foreach_node(n, nodes_)
  {
    dendrite_tip_p[n] = -1;
    double x_coord = node_x_fr_n(n, p4est_, nodes_);
    for (unsigned short i = 0; i < num_dendrites_; ++i)
    {
      if (fabs(x_tip_g[i] - x_coord) < EPS) dendrite_tip_p[n] = i;
    }
  }
  ierr = VecRestoreArray(dendrite_tip_, &dendrite_tip_p); CHKERRXX(ierr);

  // sample quantities along the tip of every dendrite
  // specifically: phi, c0, c1, t, vn, c0s, c1s, tf, kappa, velo

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;
  int nb_sample_points = brick_->nxyztrees[1]*pow(2, data->max_lvl)+1;

  std::vector<double> line_phi;
  std::vector<double> line_c0;
  std::vector<double> line_c1;
  std::vector<double> line_t;
  std::vector<double> line_vn;
  std::vector<double> line_c0s;
  std::vector<double> line_c1s;
  std::vector<double> line_tf;
  std::vector<double> line_kappa;
  std::vector<double> line_velo;

  for (unsigned short dendrite_idx = 0; dendrite_idx < num_dendrites_; ++dendrite_idx)
  {
    double xyz0[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_->xyz_min[1] };
    double xyz1[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_->xyz_max[1] };

    double xyz[P4EST_DIM];
    double dxyz[P4EST_DIM];

    foreach_dimension(dim) dxyz[dim] = (xyz1[dim]-xyz0[dim])/((double) nb_sample_points - 1.);

    line_phi  .clear(); line_phi  .resize(nb_sample_points, -DBL_MAX);
    line_c0   .clear(); line_c0   .resize(nb_sample_points, -DBL_MAX);
    line_c1   .clear(); line_c1   .resize(nb_sample_points, -DBL_MAX);
    line_t    .clear(); line_t    .resize(nb_sample_points, -DBL_MAX);
    line_vn   .clear(); line_vn   .resize(nb_sample_points, -DBL_MAX);
    line_c0s  .clear(); line_c0s  .resize(nb_sample_points, -DBL_MAX);
    line_c1s  .clear(); line_c1s  .resize(nb_sample_points, -DBL_MAX);
    line_tf   .clear(); line_tf   .resize(nb_sample_points, -DBL_MAX);
    line_kappa.clear(); line_kappa.resize(nb_sample_points, -DBL_MAX);
    line_velo .clear(); line_velo .resize(nb_sample_points, -DBL_MAX);

    my_p4est_interpolation_nodes_t interp(ngbd_);
    my_p4est_interpolation_nodes_t h_interp(history_ngbd_);

    for (unsigned int i = 0; i < nb_sample_points; ++i)
    {
      foreach_dimension(dim) xyz[dim] = xyz0[dim] + ((double) i) * dxyz[dim];
      interp.add_point_local(i, xyz);
      h_interp.add_point_local(i, xyz);
    }

    interp.set_input(phi_,    linear); interp.interpolate_local(line_phi.data());
    interp.set_input(c0_np1_, linear); interp.interpolate_local(line_c0.data());
    interp.set_input(c1_np1_, linear); interp.interpolate_local(line_c1.data());
    interp.set_input(tl_np1_, linear); interp.interpolate_local(line_t.data());
    interp.set_input(normal_velocity_np1_, linear); interp.interpolate_local(line_vn.data());

    h_interp.set_input(c0s_, linear); h_interp.interpolate_local(line_c0s.data());
    h_interp.set_input(c1s_, linear); h_interp.interpolate_local(line_c1s.data());
    h_interp.set_input(tf_ , linear); h_interp.interpolate_local(line_tf .data());
    h_interp.set_input(history_kappa_, linear); h_interp.interpolate_local(line_kappa.data());
    h_interp.set_input(history_velo_ , linear); h_interp.interpolate_local(line_velo .data());

    int mpiret;

    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_phi.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_c0 .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_c1 .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_t  .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_vn .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_c0s  .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_c1s  .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_tf   .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_kappa.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_velo .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

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
      sprintf(filename, "%s/%s.txt", dirname, "phi"); save_vector(filename, line_phi, (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "c0");  save_vector(filename, line_c0 , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "c1");  save_vector(filename, line_c1 , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "t");   save_vector(filename, line_t  , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "vn");  save_vector(filename, line_vn , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));

      sprintf(filename, "%s/%s.txt", dirname, "c0s");   save_vector(filename, line_c0s  , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "c1s");   save_vector(filename, line_c1s  , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "tf");    save_vector(filename, line_tf   , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "kappa"); save_vector(filename, line_kappa, (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
      sprintf(filename, "%s/%s.txt", dirname, "velo");  save_vector(filename, line_velo , (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app));
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
