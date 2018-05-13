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
    c0_n_(NULL), c0_np1_(NULL), c0s_(NULL),
    c1_n_(NULL), c1_np1_(NULL), c1s_(NULL),
    normal_velocity_n_(NULL),
    normal_velocity_np1_(NULL),
    phi_(NULL),
    phi_smooth_(NULL),
    kappa_(NULL),
    bc_error_(NULL),
  #ifdef P4_TO_P8
    theta_xz_(NULL), theta_yz_(NULL)
  #else
    theta_(NULL)
  #endif
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

  if(c0s_   !=NULL) { ierr = VecDestroy(c0s_);   CHKERRXX(ierr); }
  if(c1s_   !=NULL) { ierr = VecDestroy(c1s_);   CHKERRXX(ierr); }

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
#ifdef P4_TO_P8
  if(theta_xz_!=NULL) { ierr = VecDestroy(theta_xz_); CHKERRXX(ierr); }
  if(theta_yz_!=NULL) { ierr = VecDestroy(theta_yz_); CHKERRXX(ierr); }
#else
  if(theta_!=NULL) { ierr = VecDestroy(theta_); CHKERRXX(ierr); }
#endif

  if(bc_error_ !=NULL) { ierr = VecDestroy(bc_error_);   CHKERRXX(ierr); }

  if (phi_smooth_ != NULL) { ierr = VecDestroy(phi_smooth_); CHKERRXX(ierr); }

  /* destroy the p4est and its connectivity structure */
  delete ngbd_;
  delete hierarchy_;
  p4est_nodes_destroy (nodes_);
  p4est_ghost_destroy(ghost_);
  p4est_destroy (p4est_);
  my_p4est_brick_destroy(connectivity_, brick_);
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

//  /* angle between normal and direction of growth */
//#ifdef P4_TO_P8
//  if (theta_xz_ != NULL) { ierr = VecDestroy(theta_xz_); CHKERRXX(ierr); }
//  if (theta_yz_ != NULL) { ierr = VecDestroy(theta_yz_); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_, &theta_xz_); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &theta_yz_); CHKERRXX(ierr);

//  Vec theta_xz_tmp; double *theta_xz_tmp_p;
//  Vec theta_yz_tmp; double *theta_yz_tmp_p;

//  ierr = VecDuplicate(phi_, &theta_xz_tmp); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &theta_yz_tmp); CHKERRXX(ierr);

//  ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
//  ierr = VecGetArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
//#else
//  if (theta_ != NULL) { ierr = VecDestroy(theta_); CHKERRXX(ierr); }
//  ierr = VecDuplicate(phi_, &theta_); CHKERRXX(ierr);

//  Vec theta_tmp; double *theta_tmp_p;
//  ierr = VecDuplicate(phi_, &theta_tmp); CHKERRXX(ierr);
//  ierr = VecGetArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
//#endif

//  double *normal_p[P4EST_DIM];
//  foreach_dimension(dim) { ierr = VecGetArray(normal_[dim], &normal_p[dim]); CHKERRXX(ierr); }

//  foreach_node(n, nodes_)
//  {
//#ifdef P4_TO_P8
//    theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
//    theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
//#else
//    theta_tmp_p[n] = atan2(-normal_p[1][n], -normal_p[0][n]);
//#endif
//  }

//  foreach_dimension(dim) { ierr = VecRestoreArray(normal_[dim], &normal_p[dim]); CHKERRXX(ierr); }

//#ifdef P4_TO_P8
////  ierr = VecRestoreArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
////  ierr = VecRestoreArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
////  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_xz_tmp, theta_xz_);
////  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_yz_tmp, theta_yz_);
////  ierr = VecDestroy(theta_xz_tmp); CHKERRXX(ierr);
////  ierr = VecDestroy(theta_yz_tmp); CHKERRXX(ierr);
//  ierr = VecDestroy(theta_xz_); CHKERRXX(ierr);
//  ierr = VecDestroy(theta_yz_); CHKERRXX(ierr);
//  theta_xz_ = theta_xz_tmp;
//  theta_yz_ = theta_yz_tmp;
//#else
////  ierr = VecRestoreArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
////  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_tmp, theta_);
////  ierr = VecDestroy(theta_tmp); CHKERRXX(ierr);
//  ierr = VecDestroy(theta_); CHKERRXX(ierr);
//  theta_ = theta_tmp;
//#endif

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

  PetscPrintf(p4est_->mpicomm, "VMAX = %e, VGAMMAMAX = %e, COOLING_VELO = %e, %e, %e\n", u_max, vgamma_max_, cooling_velocity_, dxyz_min_/vgamma_max_, 1./kv_max);
  PetscPrintf(p4est_->mpicomm, "dt = %e\n", dt_n_);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_multialloy_t::update_grid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid... ");

  // keep level-set function from previous time step
  Vec phi_nm1;
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_nm1); CHKERRXX(ierr);

  copy_ghosted_vec(phi_, phi_nm1); CHKERRXX(ierr);

//  Vec tm_old; ierr = VecDuplicate(phi_, &tm_old); CHKERRXX(ierr);
//  Vec tp_old; ierr = VecDuplicate(phi_, &tp_old); CHKERRXX(ierr);

//  double *phi_p;
//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);

//  double *t_p, *tm_p, *tp_p;
//  ierr = VecGetArray(t_np1_, &t_p); CHKERRXX(ierr);
//  ierr = VecGetArray(tm_old, &tm_p); CHKERRXX(ierr);
//  ierr = VecGetArray(tp_old, &tp_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//  {
//    tm_p[n] = t_p[n];
//    tp_p[n] = t_p[n];
//  }

//  ierr = VecRestoreArray(t_np1_, &t_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(tm_old, &tm_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(tp_old, &tp_p); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd_);

//  ls.extend_Over_Interface_TVD(phi_, tm_old);

//  invert_phi(nodes_, phi_);
//  ls.extend_Over_Interface_TVD(phi_, tp_old);
//  invert_phi(nodes_, phi_);

//  Vec c0_interface;  ierr = VecDuplicate(phi_, &c0_interface);  CHKERRXX(ierr);
//  Vec c0n_interface; ierr = VecDuplicate(phi_, &c0n_interface); CHKERRXX(ierr);

//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0_np1_, c0_interface);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0n_n_, c0n_interface);

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


  Vec phi_tmp;
  ierr = VecDuplicate(phi_, &phi_tmp); CHKERRXX(ierr);

  interp.set_input(phi_nm1, interpolation_between_grids_);
  interp.interpolate(phi_tmp);

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

  Vec tf_tmp;
  ierr = VecDuplicate(phi_, &tf_tmp); CHKERRXX(ierr);
  interp.set_input(tf_, interpolation_between_grids_);
  interp.interpolate(tf_tmp);
  ierr = VecDestroy(tf_); CHKERRXX(ierr);
  tf_ = tf_tmp;

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

  Vec c0s_tmp;
  ierr = VecDuplicate(phi_, &c0s_tmp); CHKERRXX(ierr);
  interp.set_input(c0s_, interpolation_between_grids_);
  interp.interpolate(c0s_tmp);
  ierr = VecDestroy(c0s_); CHKERRXX(ierr);
  c0s_ = c0s_tmp;

  Vec c1s_tmp;
  ierr = VecDuplicate(phi_, &c1s_tmp); CHKERRXX(ierr);
  interp.set_input(c1s_, interpolation_between_grids_);
  interp.interpolate(c1s_tmp);
  ierr = VecDestroy(c1s_); CHKERRXX(ierr);
  c1s_ = c1s_tmp;

  // get new values for nodes that just solidified
  double *phi_nm1_p; ierr = VecGetArray(phi_tmp, &phi_nm1_p); CHKERRXX(ierr);
  double *phi_p;     ierr = VecGetArray(phi_,    &phi_p);     CHKERRXX(ierr);

  double *c0s_p; ierr = VecGetArray(c0s_, &c0s_p); CHKERRXX(ierr);
  double *c1s_p; ierr = VecGetArray(c1s_, &c1s_p); CHKERRXX(ierr);

  double *c0_n_p; ierr = VecGetArray(c0_n_, &c0_n_p); CHKERRXX(ierr);
  double *c1_n_p; ierr = VecGetArray(c1_n_, &c1_n_p); CHKERRXX(ierr);

  double *c0_np1_p; ierr = VecGetArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
  double *c1_np1_p; ierr = VecGetArray(c1_np1_, &c1_np1_p); CHKERRXX(ierr);

  double *tf_p;     ierr = VecGetArray(tf_, &tf_p); CHKERRXX(ierr);
  double *tl_n_p;   ierr = VecGetArray(tl_n_, &tl_n_p); CHKERRXX(ierr);
  double *tl_np1_p; ierr = VecGetArray(tl_np1_, &tl_np1_p); CHKERRXX(ierr);

  foreach_node(n, nodes_np1)
  {
    if (phi_p[n] > 0 && phi_nm1_p[n] < 0)
    {
      // find the time when the interface crossed a node
      double tau = fabs(phi_nm1_p[n])+EPS/(fabs(phi_nm1_p[n]) + fabs(phi_p[n]) + EPS);

      c0s_p[n] = kp0_*(c0_n_p[n]*(1.-tau) + c0_np1_p[n]*tau);
      c1s_p[n] = kp1_*(c1_n_p[n]*(1.-tau) + c1_np1_p[n]*tau);
      tf_p [n] = tl_n_p[n]*(1.-tau) + tl_np1_p[n]*tau;
    }
  }

  ierr = VecRestoreArray(phi_tmp, &phi_nm1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_,    &phi_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(c0s_, &c0s_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c1s_, &c1s_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c0_n_, &c0_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c1_n_, &c1_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c0_np1_, &c0_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c1_np1_, &c1_np1_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(tf_, &tf_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_n_, &tl_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_np1_, &tl_np1_p); CHKERRXX(ierr);

  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(phi_nm1); CHKERRXX(ierr);

  copy_ghosted_vec(c0_np1_, c0_n_);
  copy_ghosted_vec(c1_np1_, c1_n_);
  copy_ghosted_vec(tl_np1_, tl_n_);
  copy_ghosted_vec(ts_np1_, ts_n_);

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

//  Vec kappa_n;
//  ierr = VecDuplicate(phi_, &kappa_n); CHKERRXX(ierr);
//  interp.set_input(kappa_, quadratic_non_oscillatory_continuous_v1);
//  interp.interpolate(kappa_n);
//  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
//  kappa_ = kappa_n;

  p4est_destroy(p4est_);       p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

//  /* help interface to not get stuck at grid nodes */
//  double kappa_thresh = 0.1/dxyz_min_;
//  double *phi_p, *normal_velocity_np1_p, *kappa_p;

//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa_, &kappa_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//  {
//    if (fabs(phi_p[n]) < phi_thresh_ && fabs(kappa_p[n]) > kappa_thresh)
//    {
//      phi_p[n] *= -1.;
//    }
//  }

//  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(kappa_, &kappa_p); CHKERRXX(ierr);

  /* reinitialize phi */
  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_);

  invert_phi(nodes_, phi_);
  ls_new.extend_Over_Interface_TVD(phi_, c0s_, 20, 1);
  ls_new.extend_Over_Interface_TVD(phi_, c1s_, 20, 1);
  ls_new.extend_Over_Interface_TVD(phi_, tf_,  20, 1);
  invert_phi(nodes_, phi_);

  /* second derivatives, normals, curvature, angles */
  compute_geometric_properties();

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
//  solver_all_in_one.set_rhs(t_n_, t_n_, c0_n_, c1_n_);
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


  ierr = PetscLogEventEnd(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);

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
  sprintf(name, "%s/vtu/bialloy_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], brick_->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/bialloy_lvl_%d_%d_%d_%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], iter);
#endif

  const double *phi_p; ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  const double *tl_p; ierr = VecGetArrayRead(tl_np1_, &tl_p); CHKERRXX(ierr);
  const double *ts_p; ierr = VecGetArrayRead(ts_np1_, &ts_p); CHKERRXX(ierr);
  const double *c0_p; ierr = VecGetArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
  const double *c1_p; ierr = VecGetArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
  double *normal_velocity_np1_p; ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

  const double *c0s_p; ierr = VecGetArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
  const double *c1s_p; ierr = VecGetArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

  const double *tf_p; ierr = VecGetArrayRead(tf_, &tf_p); CHKERRXX(ierr);

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
                           11, 1, name,
                         #else
                           11, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
//                           VTK_POINT_DATA, "phi_smooth", phi_smooth_p,
                           VTK_POINT_DATA, "tl", tl_p,
                           VTK_POINT_DATA, "ts", ts_p,
                           VTK_POINT_DATA, "tf", tf_p,
                           VTK_POINT_DATA, "c0", c0_p,
                           VTK_POINT_DATA, "c1", c1_p,
                           VTK_POINT_DATA, "c0s", c0s_p,
                           VTK_POINT_DATA, "c1s", c1s_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_POINT_DATA, "bc_error", bc_error_p,
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

  ierr = VecRestoreArrayRead(c0s_, &c0s_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c1s_, &c1s_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(tf_, &tf_p); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "VTK saved in %s\n", name);
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
