#include "my_p4est_multialloy_optimized.h"

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#endif




my_p4est_multialloy_t::my_p4est_multialloy_t(my_p4est_node_neighbors_t *ngbd)
  : brick_(ngbd->myb), connectivity_(ngbd->p4est->connectivity), p4est_(ngbd->p4est), ghost_(ngbd->ghost), nodes_(ngbd->nodes), hierarchy_(ngbd->hierarchy), ngbd_(ngbd),
    t_n_(NULL), t_np1_(NULL),
    c0_n_(NULL), c0_np1_(NULL),
    c1_n_(NULL), c1_np1_(NULL),
    normal_velocity_n_(NULL),
    normal_velocity_np1_(NULL),
    phi_(NULL),
    phi_smooth_(NULL),
    kappa_(NULL),
    eps_c_cf_(this), eps_v_cf_(this),
    c00_cf_(this),
    bc_error_(NULL),
  #ifdef P4_TO_P8
    theta_xz_(NULL), theta_yz_(NUL)
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
  epsilon_anisotropy_   = 0.05;
  epsilon_c_            = 2.7207e-5; /* cm.K     */
  epsilon_v_            = 2.27e-2;   /* s.K.cm-1 */

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

  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v1;
}




my_p4est_multialloy_t::~my_p4est_multialloy_t()
{
  if(t_n_   !=NULL) { ierr = VecDestroy(t_n_);   CHKERRXX(ierr); }
  if(t_np1_ !=NULL) { ierr = VecDestroy(t_np1_); CHKERRXX(ierr); }

  if(c0_n_   !=NULL) { ierr = VecDestroy(c0_n_);   CHKERRXX(ierr); }
  if(c0_np1_ !=NULL) { ierr = VecDestroy(c0_np1_); CHKERRXX(ierr); }
  if(c1_n_   !=NULL) { ierr = VecDestroy(c1_n_);   CHKERRXX(ierr); }
  if(c1_np1_ !=NULL) { ierr = VecDestroy(c1_np1_); CHKERRXX(ierr); }

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
  if(theta_!=NULL) { ierr = VecDestroy(theta_); CHKERRXX(ierr); }

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




//void my_p4est_multialloy_t::compute_normal_and_curvature()
//{
//  /* second order derivatives */
//  for (int dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    if (phi_dd_[dim] != NULL) { ierr = VecDestroy(phi_dd_[dim]); CHKERRXX(ierr); }
//    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_dd_[dim]); CHKERRXX(ierr);
//  }

//  ngbd_->second_derivatives_central(phi_, phi_dd_);

//  /* normal */
//  double *normal_p[P4EST_DIM];
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    if(normal_[dir]!=NULL) { ierr = VecDestroy(normal_[dir]); CHKERRXX(ierr); }
//    ierr = VecCreateGhostNodes(p4est_, nodes_, &normal_[dir]); CHKERRXX(ierr);
//    ierr = VecGetArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

//  const double *phi_p;
//  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);

//  quad_neighbor_nodes_of_node_t qnnn;
//  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_->get_layer_node(i);
//    qnnn = ngbd_->get_neighbors(n);
//    normal_p[0][n] = qnnn.dx_central(phi_p);
//    normal_p[1][n] = qnnn.dy_central(phi_p);
//#ifdef P4_TO_P8
//    normal_p[2][n] = qnnn.dz_central(phi_p);
//    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
//#else
//    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
//#endif

//    normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//    normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//#ifdef P4_TO_P8
//    normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//#endif
//  }

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGhostUpdateBegin(normal_[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_->get_local_node(i);
//    qnnn = ngbd_->get_neighbors(n);
//    normal_p[0][n] = qnnn.dx_central(phi_p);
//    normal_p[1][n] = qnnn.dy_central(phi_p);
//#ifdef P4_TO_P8
//    normal_p[2][n] = qnnn.dz_central(phi_p);
//    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
//#else
//    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
//#endif

//    normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//    normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//#ifdef P4_TO_P8
//    normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//#endif
//  }
//  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGhostUpdateEnd(normal_[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  /* curvature */
//  if(kappa_!=NULL) { ierr = VecDestroy(kappa_); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_, &kappa_); CHKERRXX(ierr);

//  Vec kappa_tmp;
//  ierr = VecDuplicate(kappa_, &kappa_tmp); CHKERRXX(ierr);
//  double *kappa_p;
//  ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
//  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_->get_layer_node(i);
//    qnnn = ngbd_->get_neighbors(n);
////#ifdef P4_TO_P8
////    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1./dxyz_max_), -1./dxyz_max_);
////#else
////    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1./dxyz_max_), -1./dxyz_max_);
////#endif
//#ifdef P4_TO_P8
//    kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]);
//#else
//    kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
//#endif
//  }
//  ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
//  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_->get_local_node(i);
//    qnnn = ngbd_->get_neighbors(n);
////#ifdef P4_TO_P8
////    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1./dxyz_max_), -1./dxyz_max_);
////#else
////    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1./dxyz_max_), -1./dxyz_max_);
////#endif
//    #ifdef P4_TO_P8
//        kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]);
//    #else
//        kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
//    #endif
//  }
//  ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
//  ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

////  my_p4est_level_set_t ls(ngbd);
////  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
////  ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

//  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
//  kappa_ = kappa_tmp;

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

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//  {
//#ifdef P4_TO_P8
//    theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
//    theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
//#else
//    theta_tmp_p[n] = atan2(-normal_p[1][n], -normal_p[0][n]);
//#endif
//  }

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

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

//} //*/

void my_p4est_multialloy_t::compute_normal_and_curvature()
{
  /* second order derivatives */
  for (int dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (phi_dd_[dim] != NULL) { ierr = VecDestroy(phi_dd_[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_dd_[dim]); CHKERRXX(ierr);
  }

  ngbd_->second_derivatives_central(phi_, phi_dd_);

  /* normal */
  double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(normal_[dir]!=NULL) { ierr = VecDestroy(normal_[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &normal_[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  const double *phi_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
#else
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(normal_[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_local_node(i);
    qnnn = ngbd_->get_neighbors(n);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
#else
#endif

  }
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(normal_[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* curvature */
  if(kappa_!=NULL) { ierr = VecDestroy(kappa_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &kappa_); CHKERRXX(ierr);

  Vec kappa_tmp;
  ierr = VecDuplicate(kappa_, &kappa_tmp); CHKERRXX(ierr);
  double *kappa_p;
  ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);
#ifdef P4_TO_P8
    kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]);
#else
    if (is_node_Wall(p4est_, ni))
    {
      kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
    } else {
      double phi_x = normal_p[0][n];
      double phi_y = normal_p[1][n];
      double phi_xx = qnnn.dxx_central(phi_p);
      double phi_yy = qnnn.dyy_central(phi_p);
      double phi_xy = .5*(qnnn.dx_central(normal_p[1])+qnnn.dy_central(normal_p[0]));
      kappa_p[n] = (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx)/pow(phi_x*phi_x+phi_y*phi_y,3./2.);
    }
#endif
  }

  ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);

  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_local_node(i);
    qnnn = ngbd_->get_neighbors(n);
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);
#ifdef P4_TO_P8
    kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]);
#else

    if (is_node_Wall(p4est_, ni))
    {
      kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
    } else {
      double phi_x = normal_p[0][n];
      double phi_y = normal_p[1][n];
      double phi_xx = qnnn.dxx_central(phi_p);
      double phi_yy = qnnn.dyy_central(phi_p);
      double phi_xy = .5*(qnnn.dx_central(normal_p[1])+qnnn.dy_central(normal_p[0]));
      kappa_p[n] = (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx)/pow(phi_x*phi_x+phi_y*phi_y,3./2.);
    }
#endif
  }
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);


  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

    normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
    normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd_);
  ls.extend_from_interface_to_whole_domain_TVD(phi_, kappa_tmp, kappa_);
  ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

//  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
//  kappa_ = kappa_tmp;

  /* angle between normal and direction of growth */
#ifdef P4_TO_P8
  if (theta_xz_ != NULL) { ierr = VecDestroy(theta_xz_); CHKERRXX(ierr); }
  if (theta_yz_ != NULL) { ierr = VecDestroy(theta_yz_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &theta_xz_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &theta_yz_); CHKERRXX(ierr);

  Vec theta_xz_tmp; double *theta_xz_tmp_p;
  Vec theta_yz_tmp; double *theta_yz_tmp_p;
  ierr = VecDuplicate(phi_, &theta_xz_tmp); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &theta_yz_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
#else
  if (theta_ != NULL) { ierr = VecDestroy(theta_); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_, &theta_); CHKERRXX(ierr);

  Vec theta_tmp; double *theta_tmp_p;
  ierr = VecDuplicate(phi_, &theta_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
#endif

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
    theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
#else
    theta_tmp_p[n] = atan2(-normal_p[1][n], -normal_p[0][n]);
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal_[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

#ifdef P4_TO_P8
//  ierr = VecRestoreArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_xz_tmp, theta_xz_);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_yz_tmp, theta_yz_);
//  ierr = VecDestroy(theta_xz_tmp); CHKERRXX(ierr);
//  ierr = VecDestroy(theta_yz_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(theta_xz_); CHKERRXX(ierr);
  ierr = VecDestroy(theta_yz_); CHKERRXX(ierr);
  theta_xz_ = theta_xz_tmp;
  theta_yz_ = theta_yz_tmp;
#else
//  ierr = VecRestoreArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, theta_tmp, theta_);
//  ierr = VecDestroy(theta_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(theta_); CHKERRXX(ierr);
  theta_ = theta_tmp;
#endif

}




void my_p4est_multialloy_t::compute_velocity()
{
  Vec c_interface; ierr = VecDuplicate(phi_, &c_interface); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd_);

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

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);

    v_gamma_p[0][n] = -qnnn.dx_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    v_gamma_p[1][n] = -qnnn.dy_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = -qnnn.dz_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#endif

    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n]);

//    if (vn_p[n] < 0)
//    {
//      vn_p[n] = 0;
//      v_gamma_p[0][n] = 0;
//      v_gamma_p[1][n] = 0;
//    }
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

    v_gamma_p[0][n] = -qnnn.dx_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
    v_gamma_p[1][n] = -qnnn.dy_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = -qnnn.dz_central(c0_np1_p)*Dl0_ / (1.-kp0_) / MAX(c_interface_p[n], 1e-7);
#endif

    vn_p[n] = (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n]);

//    if (vn_p[n] < 0)
//    {
//      vn_p[n] = 0;
//      v_gamma_p[0][n] = 0;
//      v_gamma_p[1][n] = 0;
//    }
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
    ierr = VecGetArray(normal_[dim], &normal_p[dim]); CHKERRXX(ierr);
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
}






void my_p4est_multialloy_t::compute_dt()
{
  /* compute the size of smallest detail, i.e. maximum curvature */
  const double *kappa_p;
  ierr = VecGetArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);

  const double *phi_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);

  const double *vn_p;
  ierr = VecGetArrayRead(normal_velocity_n_, &vn_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(normal_velocity_np1_, &vn_p); CHKERRXX(ierr);

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

  ierr = VecRestoreArrayRead(normal_velocity_n_, &vn_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(normal_velocity_np1_, &vn_p); CHKERRXX(ierr);
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
}




void my_p4est_multialloy_t::update_grid_eno()
{
  PetscPrintf(p4est_->mpicomm, "Updating grid... ");

  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1, ghost_np1);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  Vec phi_nm1;
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_nm1); CHKERRXX(ierr);

  copy_ghosted_vec(phi_, phi_nm1); CHKERRXX(ierr);

  Vec tm_old; ierr = VecDuplicate(phi_, &tm_old); CHKERRXX(ierr);
  Vec tp_old; ierr = VecDuplicate(phi_, &tp_old); CHKERRXX(ierr);

//  double *phi_p;
//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);

  double *t_p, *tm_p, *tp_p;
  ierr = VecGetArray(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecGetArray(tm_old, &tm_p); CHKERRXX(ierr);
  ierr = VecGetArray(tp_old, &tp_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    tm_p[n] = t_p[n];
    tp_p[n] = t_p[n];
  }

  ierr = VecRestoreArray(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tm_old, &tm_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tp_old, &tp_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd_);

  ls.extend_Over_Interface_TVD(phi_, tm_old);

  invert_phi();
  ls.extend_Over_Interface_TVD(phi_, tp_old);
  invert_phi();

//  Vec c0_interface;  ierr = VecDuplicate(phi_, &c0_interface);  CHKERRXX(ierr);
//  Vec c0n_interface; ierr = VecDuplicate(phi_, &c0n_interface); CHKERRXX(ierr);

//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0_np1_, c0_interface);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0n_n_, c0n_interface);

  my_p4est_interpolation_nodes_t interp(ngbd_);

  {

    double dt_tmp = ls.advect_in_normal_direction(normal_velocity_np1_, phi_, dt_n_);

    if (dt_n_ > dt_tmp) throw;

    /* save the old splitting criteria information */
    splitting_criteria_t* sp_old = (splitting_criteria_t*)ngbd_->p4est->user_pointer;

    Vec phi_np1;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    bool is_grid_changing = true;

    int counter = 0;
    while (is_grid_changing)
    {
      double xyz[P4EST_DIM];
      for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
      {
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
        interp.add_point(n, xyz);
      }

      interp.set_input(phi_, interpolation_between_grids_);
      interp.interpolate(phi_np1);

      double* phi_np1_p;
      ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
      is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1_p);

      ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      if (is_grid_changing)
      {
        my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est_np1, ghost_np1);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
        interp.clear();
      }

      counter++;
    }

    p4est_np1->user_pointer = (void*) sp_old;

    ierr = VecDestroy(phi_); CHKERRXX(ierr);
    phi_ = phi_np1;
  }

  /* interpolate the quantities on the new grid */

//  double xyz[P4EST_DIM];
//  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
//  {
//    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
//    interp.add_point(n, xyz);
//  }

  /* temperature */

  Vec phi_tmp;
  ierr = VecDuplicate(phi_, &phi_tmp); CHKERRXX(ierr);

  interp.set_input(phi_nm1, interpolation_between_grids_);
  interp.interpolate(phi_tmp);

  Vec tm_tmp; ierr = VecDuplicate(phi_, &tm_tmp); CHKERRXX(ierr);
  interp.set_input(tm_old, interpolation_between_grids_);
  interp.interpolate(tm_tmp);

  Vec tp_tmp; ierr = VecDuplicate(phi_, &tp_tmp); CHKERRXX(ierr);
  interp.set_input(tp_old, interpolation_between_grids_);
  interp.interpolate(tp_tmp);

  ierr = VecDestroy(t_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &t_n_); CHKERRXX(ierr);

  ierr = VecGetArray(t_n_, &t_p); CHKERRXX(ierr);
  ierr = VecGetArray(tm_tmp, &tm_p); CHKERRXX(ierr);
  ierr = VecGetArray(tp_tmp, &tp_p); CHKERRXX(ierr);

  double *phi_tmp_p;
  ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

  for(size_t n = 0; n < nodes_np1->indep_nodes.elem_count; ++n)
  {
    if (phi_tmp_p[n] < 0.)  t_p[n] = tm_p[n];
    else                    t_p[n] = tp_p[n];
  }

  ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_n_, &t_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tm_tmp, &tm_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tp_tmp, &tp_p); CHKERRXX(ierr);

  ierr = VecDestroy(tm_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(tp_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(tm_old); CHKERRXX(ierr);
  ierr = VecDestroy(tp_old); CHKERRXX(ierr);
  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(phi_nm1); CHKERRXX(ierr);

//  ierr = VecDestroy(t_n_); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &t_n_); CHKERRXX(ierr);
//  interp.set_input(t_np1_, interpolation_between_grids_);
//  interp.interpolate(t_n_);


  ierr = VecDestroy(t_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &t_np1_); CHKERRXX(ierr);
  copy_ghosted_vec(t_n_, t_np1_);


  /* concentrartions */

  ierr = VecDestroy(c0_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c0_n_); CHKERRXX(ierr);
  interp.set_input(c0_np1_, interpolation_between_grids_);
  interp.interpolate(c0_n_);

  ierr = VecDestroy(c0_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c0_np1_); CHKERRXX(ierr);
//  interp.set_input(c0_interface, interpolation_between_grids_);
//  interp.interpolate(c0_np1_);
  copy_ghosted_vec(c0_n_, c0_np1_);

//  ierr = VecDestroy(c0n_n_); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &c0n_n_); CHKERRXX(ierr);
//  interp.set_input(c0n_interface, interpolation_between_grids_);
//  interp.interpolate(c0n_n_);

//  ierr = VecDestroy(c0_interface); CHKERRXX(ierr);
//  ierr = VecDestroy(c0n_interface); CHKERRXX(ierr);

  ierr = VecDestroy(c1_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c1_n_); CHKERRXX(ierr);
  interp.set_input(c1_np1_, interpolation_between_grids_);
  interp.interpolate(c1_n_);

  ierr = VecDestroy(c1_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c1_np1_); CHKERRXX(ierr);
  copy_ghosted_vec(c1_n_, c1_np1_);


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

  Vec kappa_n;
  ierr = VecDuplicate(phi_, &kappa_n); CHKERRXX(ierr);
  interp.set_input(kappa_, quadratic_non_oscillatory_continuous_v1);
  interp.interpolate(kappa_n);
  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
  kappa_ = kappa_n;

  p4est_destroy(p4est_);       p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  /* help interface to not get stuck at grid nodes */
  double kappa_thresh = 0.1/dxyz_min_;
  double *phi_p, *normal_velocity_np1_p, *kappa_p;

  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa_, &kappa_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
  {
    if (fabs(phi_p[n]) < phi_thresh_ && fabs(kappa_p[n]) > kappa_thresh)
    {
      phi_p[n] *= -1.;
      std::cout << "Happened!\n";
    }
  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_, &kappa_p); CHKERRXX(ierr);

//  double *phi_p, *normal_velocity_np1_p, *kappa_p;

//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa_, &kappa_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//  {
//    if (fabs(phi_p[n]) < phi_thresh_ && fabs(kappa_p[n]) > kappa_thresh)
//    {
//      if (normal_velocity_np1_p[n] > 0. && phi_p[n] > 0 ||
//          normal_velocity_np1_p[n] < 0. && phi_p[n] < 0)
//      {
//        phi_p[n] *= -1.;
//        std::cout << "Happened!\n";
//      }
//    }
////    if (phi_p[n] > 0. && phi_p[n] < fraction*dxyz_min)
////      phi_p[n] *= -1;
//  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_, &kappa_p); CHKERRXX(ierr);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_);
//  ls_new.reinitialize_2nd_order(phi_);
//  ls_new.perturb_level_set_function(phi_, EPS);

  compute_normal_and_curvature();

  compute_smoothed_phi();

  PetscPrintf(p4est_->mpicomm, "Done \n");
}

void my_p4est_multialloy_t::update_grid()
{
  PetscPrintf(p4est_->mpicomm, "Updating grid... ");

  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1, ghost_np1);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  Vec phi_nm1;
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_nm1); CHKERRXX(ierr);

  copy_ghosted_vec(phi_, phi_nm1); CHKERRXX(ierr);

  Vec tm_old; ierr = VecDuplicate(phi_, &tm_old); CHKERRXX(ierr);
  Vec tp_old; ierr = VecDuplicate(phi_, &tp_old); CHKERRXX(ierr);

//  double *phi_p;
//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);

  double *t_p, *tm_p, *tp_p;
  ierr = VecGetArray(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecGetArray(tm_old, &tm_p); CHKERRXX(ierr);
  ierr = VecGetArray(tp_old, &tp_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    tm_p[n] = t_p[n];
    tp_p[n] = t_p[n];
  }

  ierr = VecRestoreArray(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tm_old, &tm_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tp_old, &tp_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd_);

  ls.extend_Over_Interface_TVD(phi_, tm_old);

  invert_phi();
  ls.extend_Over_Interface_TVD(phi_, tp_old);
  invert_phi();

//  Vec c0_interface;  ierr = VecDuplicate(phi_, &c0_interface);  CHKERRXX(ierr);
//  Vec c0n_interface; ierr = VecDuplicate(phi_, &c0n_interface); CHKERRXX(ierr);

//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0_np1_, c0_interface);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0n_n_, c0n_interface);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_, ngbd_);

  /* bousouf update this for second order in time */
//  invert_phi();
  if (0)
    sl.update_p4est(v_interface_np1_, dt_n_, phi_);
  else // Daniil: synchronized with the serial version (sort of)
    sl.update_p4est(v_interface_n_, v_interface_np1_, dt_nm1_, dt_n_, phi_);
//  invert_phi();

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  /* temperature */

  Vec phi_tmp;
  ierr = VecDuplicate(phi_, &phi_tmp); CHKERRXX(ierr);

  interp.set_input(phi_nm1, interpolation_between_grids_);
  interp.interpolate(phi_tmp);

  Vec tm_tmp; ierr = VecDuplicate(phi_, &tm_tmp); CHKERRXX(ierr);
  interp.set_input(tm_old, interpolation_between_grids_);
  interp.interpolate(tm_tmp);

  Vec tp_tmp; ierr = VecDuplicate(phi_, &tp_tmp); CHKERRXX(ierr);
  interp.set_input(tp_old, interpolation_between_grids_);
  interp.interpolate(tp_tmp);

  ierr = VecDestroy(t_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &t_n_); CHKERRXX(ierr);

  ierr = VecGetArray(t_n_, &t_p); CHKERRXX(ierr);
  ierr = VecGetArray(tm_tmp, &tm_p); CHKERRXX(ierr);
  ierr = VecGetArray(tp_tmp, &tp_p); CHKERRXX(ierr);

  double *phi_tmp_p;
  ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

  for(size_t n = 0; n < nodes_np1->indep_nodes.elem_count; ++n)
  {
    if (phi_tmp_p[n] < 0.)  t_p[n] = tm_p[n];
    else                    t_p[n] = tp_p[n];
  }

  ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_n_, &t_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tm_tmp, &tm_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tp_tmp, &tp_p); CHKERRXX(ierr);

  ierr = VecDestroy(tm_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(tp_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(tm_old); CHKERRXX(ierr);
  ierr = VecDestroy(tp_old); CHKERRXX(ierr);
  ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(phi_nm1); CHKERRXX(ierr);

//  ierr = VecDestroy(t_n_); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &t_n_); CHKERRXX(ierr);
//  interp.set_input(t_np1_, interpolation_between_grids_);
//  interp.interpolate(t_n_);


  ierr = VecDestroy(t_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &t_np1_); CHKERRXX(ierr);
  copy_ghosted_vec(t_n_, t_np1_);


  /* concentrartions */

  ierr = VecDestroy(c0_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c0_n_); CHKERRXX(ierr);
  interp.set_input(c0_np1_, interpolation_between_grids_);
  interp.interpolate(c0_n_);

  ierr = VecDestroy(c0_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c0_np1_); CHKERRXX(ierr);
//  interp.set_input(c0_interface, interpolation_between_grids_);
//  interp.interpolate(c0_np1_);
  copy_ghosted_vec(c0_n_, c0_np1_);

//  ierr = VecDestroy(c0n_n_); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_, &c0n_n_); CHKERRXX(ierr);
//  interp.set_input(c0n_interface, interpolation_between_grids_);
//  interp.interpolate(c0n_n_);

//  ierr = VecDestroy(c0_interface); CHKERRXX(ierr);
//  ierr = VecDestroy(c0n_interface); CHKERRXX(ierr);

  ierr = VecDestroy(c1_n_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c1_n_); CHKERRXX(ierr);
  interp.set_input(c1_np1_, interpolation_between_grids_);
  interp.interpolate(c1_n_);

  ierr = VecDestroy(c1_np1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_, &c1_np1_); CHKERRXX(ierr);
  copy_ghosted_vec(c1_n_, c1_np1_);


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

  Vec kappa_n;
  ierr = VecDuplicate(phi_, &kappa_n); CHKERRXX(ierr);
  interp.set_input(kappa_, quadratic_non_oscillatory_continuous_v1);
  interp.interpolate(kappa_n);
  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
  kappa_ = kappa_n;

  p4est_destroy(p4est_);       p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  /* help interface to not get stuck at grid nodes */
  double kappa_thresh = 0.1/dxyz_min_;
  double *phi_p, *normal_velocity_np1_p, *kappa_p;

  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa_, &kappa_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
  {
    if (fabs(phi_p[n]) < phi_thresh_ && fabs(kappa_p[n]) > kappa_thresh)
    {
      phi_p[n] *= -1.;
      std::cout << "Happened!\n";
    }
  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_, &kappa_p); CHKERRXX(ierr);

//  double *phi_p, *normal_velocity_np1_p, *kappa_p;

//  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa_, &kappa_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
//  {
//    if (fabs(phi_p[n]) < phi_thresh_ && fabs(kappa_p[n]) > kappa_thresh)
//    {
//      if (normal_velocity_np1_p[n] > 0. && phi_p[n] > 0 ||
//          normal_velocity_np1_p[n] < 0. && phi_p[n] < 0)
//      {
//        phi_p[n] *= -1.;
//        std::cout << "Happened!\n";
//      }
//    }
////    if (phi_p[n] > 0. && phi_p[n] < fraction*dxyz_min)
////      phi_p[n] *= -1;
//  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_, &kappa_p); CHKERRXX(ierr);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_);
//  ls_new.reinitialize_2nd_order(phi_);
//  ls_new.perturb_level_set_function(phi_, EPS);

  compute_normal_and_curvature();

  compute_smoothed_phi();

  PetscPrintf(p4est_->mpicomm, "Done \n");
}

//void my_p4est_multialloy_t::one_step()
//{
//  Vec normal_velocity_tmp;
//  ierr = VecDuplicate(phi, &normal_velocity_tmp); CHKERRXX(ierr);

//  Vec normal_velocity_cl;
//  ierr = VecDuplicate(phi, &normal_velocity_cl); CHKERRXX(ierr);

//  double *normal_velocity_tmp_p, *phi_p, *normal_velocity_np1_p;
//  double error = 1;
//  int iteration = 0;
//  matrices_are_constructed = false;

//  double error_cl = 1;
//  int iteration_cl = 0;

//  while(error>velocity_tol && iteration<100)
//  {
//    Vec src, out;
//    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostGetLocalForm(normal_velocity_tmp                , &out); CHKERRXX(ierr);
//    ierr = VecCopy(src, out); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_tmp                , &out); CHKERRXX(ierr);


//    solve_temperature();
//    error_cl = 1;
//    iteration_cl = 0;
//    while (error_cl > 1e-2 && iteration_cl < 100)
//    {
//      ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//      ierr = VecGhostGetLocalForm(normal_velocity_cl , &out); CHKERRXX(ierr);
//      ierr = VecCopy(src, out); CHKERRXX(ierr);
//      ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//      ierr = VecGhostRestoreLocalForm(normal_velocity_cl , &out); CHKERRXX(ierr);

//      solve_concentration_sec();
//      solve_concentration();
//      compute_normal_velocity();

//      double *normal_velocity_cl_p;
//      ierr = VecGetArray(normal_velocity_cl, &normal_velocity_cl_p); CHKERRXX(ierr);
//      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//      ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//      error_cl = 0;
//      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//      {
//        if(fabs(phi_p[n]) < dxyz_close_interface)
//          error_cl = MAX(error_cl, fabs(normal_velocity_cl_p[n]-normal_velocity_np1_p[n]));
//      }
//      ierr = VecRestoreArray(normal_velocity_cl, &normal_velocity_cl_p); CHKERRXX(ierr);
//      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//      ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_cl, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//      error_cl /= MAX(vgamma_max,1e-8);

//      ierr = PetscPrintf(p4est->mpicomm, "\t Convergence sub-iteration #%d, max_velo = %e, error_cl = %e\n", iteration_cl, vgamma_max, error_cl); CHKERRXX(ierr);

//      iteration_cl++;
//    }


//    ierr = VecGetArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//    error = 0;
//    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//    {
//      if(fabs(phi_p[n]) < dxyz_close_interface)
//        error = MAX(error,fabs(normal_velocity_tmp_p[n]-normal_velocity_np1_p[n]));
//    }
//    ierr = VecRestoreArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//    error /= MAX(vgamma_max,1e-8);

//    matrices_are_constructed = true;
//    ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, max_velo = %e, error = %e\n", iteration, vgamma_max, error); CHKERRXX(ierr);
//    iteration++;
//  }

//  compare_velocity_temperature_vs_concentration();

//  compute_velocity();
//  compute_dt();
//  update_grid();

//  ierr = VecDestroy(normal_velocity_tmp); CHKERRXX(ierr);
//  ierr = VecDestroy(normal_velocity_cl); CHKERRXX(ierr);
//}

//void my_p4est_multialloy_t::one_step()
//{
//  Vec normal_velocity_tmp;
//  ierr = VecDuplicate(phi, &normal_velocity_tmp); CHKERRXX(ierr);

////  Vec normal_velocity_np1_tmp;
////  ierr = VecDuplicate(phi, &normal_velocity_np1_tmp); CHKERRXX(ierr);

//  double *normal_velocity_tmp_p, *phi_p, *normal_velocity_np1_p, *normal_velocity_np1_tmp_p;
//  double error = 1;
//  int iteration = 0;
//  matrices_are_constructed = false;

//  while(error>velocity_tol && iteration<100)
//  {
//    Vec src, out;
//    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostGetLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);
//    ierr = VecCopy(src, out); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);

//    solve_temperature();

//    // compute velocity 2-robin 1-dirichlet
//    solve_concentration_sec();
//    solve_concentration();
//    compute_normal_velocity();

////    // save velocity
////    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
////    ierr = VecGhostGetLocalForm(normal_velocity_np1_tmp, &out); CHKERRXX(ierr);
////    ierr = VecCopy(src, out); CHKERRXX(ierr);
////    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
////    ierr = VecGhostRestoreLocalForm(normal_velocity_np1_tmp, &out); CHKERRXX(ierr);

////    // swap concentrations

////    // compute velocity 1-robin 2-dirichlet
////    solve_concentration_sec();
////    solve_concentration();
////    compute_normal_velocity();

////    // compute average velocity

////    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
////    ierr = VecGetArray(normal_velocity_np1_tmp, &normal_velocity_np1_tmp_p); CHKERRXX(ierr);

////    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
////    {
////      normal_velocity_np1_p[n] = 0.5*(normal_velocity_np1_p[n]+normal_velocity_np1_tmp_p[n]);
////    }

//    ierr = VecGetArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//    error = 0;
//    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//    {
//      if(fabs(phi_p[n]) < dxyz_close_interface)
//        error = MAX(error,fabs(normal_velocity_tmp_p[n]-normal_velocity_np1_p[n]));
//    }
//    ierr = VecRestoreArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//    error /= MAX(vgamma_max,1e-8);

//    matrices_are_constructed = true;
//    ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, max_velo = %e, error = %e\n", iteration, vgamma_max, error); CHKERRXX(ierr);
//    iteration++;
//  }

//  compare_velocity_temperature_vs_concentration();

//  compute_velocity();
//  compute_dt();
//  update_grid();

//  ierr = VecDestroy(normal_velocity_tmp); CHKERRXX(ierr);

//  first_step = false;
//}



void my_p4est_multialloy_t::one_step()
{
  // solve coupled system of equations
  my_p4est_poisson_nodes_multialloy_t solver_all_in_one(ngbd_);
  solver_all_in_one.set_phi(phi_, phi_dd_, normal_, kappa_, theta_, phi_smooth_);
  solver_all_in_one.set_parameters(dt_n_, thermal_diffusivity_, thermal_conductivity_, latent_heat_, Tm_, Dl0_, kp0_, ml0_, Dl1_, kp1_, ml1_);
  solver_all_in_one.set_bc(bc_t_, bc_c0_, bc_c1_);
  solver_all_in_one.set_GT(zero_);
  solver_all_in_one.set_undercoolings(eps_v_cf_, eps_c_cf_);
  solver_all_in_one.set_pin_every_n_steps(pin_every_n_steps_);
  solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
  solver_all_in_one.set_rhs(t_n_, t_n_, c0_n_, c1_n_);

  solver_all_in_one.set_jump_t(zero_);
  solver_all_in_one.set_flux_c(zero_, zero_);


//  Vec c0_gamma; ierr = VecDuplicate(phi_, &c0_gamma); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd_);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, c0_n_, c0_np1_);

  my_p4est_interpolation_nodes_t interp_c0_n(ngbd_);
  interp_c0_n.set_input(c0_n_, linear);

  solver_all_in_one.set_c0_guess(interp_c0_n);

  if (bc_error_ != NULL) { ierr = VecDestroy(bc_error_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_, &bc_error_); CHKERRXX(ierr);

//  solver_all_in_one.solve(t_np1_, c0_np1_, c1_np1_, bc_error_, bc_error_max_);

//  dt_nm1_ = dt_n_;
  solver_all_in_one.solve(t_np1_, c0_np1_, c1_np1_, bc_error_, bc_error_max_, dt_n_, cfl_number_);

//  solver_all_in_one.solve(t_np1_, c0_np1_, c1_np1_, bc_error_, bc_error_max_, c0n_n_);

//  ierr = VecDestroy(c0_gamma); CHKERRXX(ierr);
  // compute velocity
  compute_velocity();
//  compute_dt();
}

//void my_p4est_multialloy_t::smooth_velocity(Vec input)
//{
//  Vec output;
//  ierr = VecDuplicate(phi_, &output); CHKERRXX(ierr);

//  ierr = VecSet(output, 0.0); CHKERRXX(ierr);

//  double *input_ptr, *output_ptr;
//  ierr = VecGetArray(input, &input_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(output, &output_ptr); CHKERRXX(ierr);

//  const p4est_locidx_t *q2n = nodes_->local_nodes;

//  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

//  /* loop through ghosts */
//  for (p4est_locidx_t ghost_idx = 0; ghost_idx < ghost_->ghosts.elem_count; ++ghost_idx)
//  {
//    // get a ghost quadrant
//    p4est_quadrant_t* quad = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, ghost_idx);

//    // do only finest quadrant that are near interface
//    if (quad->level == data->max_lvl)
//    {
//      p4est_locidx_t offset = (p4est->local_num_quadrants + ghost_idx)*P4EST_CHILDREN;

//      // calculate average in the cell
//      double average = 0;
//      for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
//      {
//        p4est_locidx_t node_idx = q2n[offset + child_idx];
//        average += input_ptr[node_idx];
//      }

//      average /= (double) P4EST_CHILDREN;
//      average /= (double) P4EST_CHILDREN;

//      /* loop through nodes of a quadrant and put weights on those nodes, which are local */
//      for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
//      {
//        p4est_locidx_t node_idx = q2n[offset + child_idx];
//        if (node_idx < nodes_->num_owned_indeps)
//          output_ptr[node_idx] += average;
//      }
//    }
//  }


////  std::cout << "here?\n";
//  /* loop through local quadrants */

//  for(p4est_topidx_t tree_idx = p4est_->first_local_tree; tree_idx <= p4est_->last_local_tree; ++tree_idx)
//  {
//    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tree_idx);

//    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
//    {
//      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);

//      // do only finest quadrant that are near interface
//      if (quad->level == data->max_lvl)
//      {
//        p4est_locidx_t quad_idx_forest = quad_idx + tree->quadrants_offset;
//        p4est_locidx_t offset = quad_idx_forest*P4EST_CHILDREN;

//        // calculate average in the cell
//        double average = 0;
//        for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
//        {
//          p4est_locidx_t node_idx = q2n[offset + child_idx];
//          average += input_ptr[node_idx];
//        }

//        average /= (double) P4EST_CHILDREN;
//        average /= (double) P4EST_CHILDREN;

//        /* loop through nodes of a quadrant and put weights on those nodes, which are local */
//        for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
//        {
//          p4est_locidx_t node_idx = q2n[offset + child_idx];
//          if (node_idx < nodes_->num_owned_indeps)
//            output_ptr[node_idx] += average;
//        }
//      }
//    }
//  }

//  ierr = VecRestoreArray(input, &input_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(output, &output_ptr); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(output, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(output, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


//  my_p4est_level_set_t ls(ngbd_);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_, output, input);

//  ierr = VecDestroy(output); CHKERRXX(ierr);
//}



void my_p4est_multialloy_t::save_VTK(int iter)
{
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }
//  std::ostringstream command;
//  command << "mkdir -p " << out_dir << "/vtu";
//  int ret_sys = system(command.str().c_str());
//  if(ret_sys<0)
//    throw std::invalid_argument("my_p4est_multialloy_t::save_vtk could not create directory");

  char name[1000];
#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/bialloy_%d_%dx%dx%d.%05d", out_dir, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], brick_->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/bialloy_%d_%dx%d.%05d", out_dir, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], iter);
#endif

  const double *phi_smooth_p;
  const double *phi_p, *t_p, *c0_p, *c1_p;
  double *normal_velocity_np1_p;
  ierr = VecGetArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_smooth_, &phi_smooth_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

  const double *kappa_p;
  ierr = VecGetArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);

  const double *theta_p;
  ierr = VecGetArrayRead(theta_, &theta_p); CHKERRXX(ierr);

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
                           9, 1, name,
                         #else
                           9, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "phi_smooth", phi_smooth_p,
                           VTK_POINT_DATA, "t", t_p,
                           VTK_POINT_DATA, "c0", c0_p,
                           VTK_POINT_DATA, "c1", c1_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_POINT_DATA, "theta", theta_p,
                           VTK_POINT_DATA, "bc_error", bc_error_p,
                           VTK_CELL_DATA , "leaf_level", l_p);


  for(size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] *= scaling_;

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(bc_error_, &bc_error_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_smooth_, &phi_smooth_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(t_np1_, &t_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c0_np1_, &c0_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c1_np1_, &c1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(theta_, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1_, &normal_velocity_np1_p); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "VTK saved in %s\n", name);
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
