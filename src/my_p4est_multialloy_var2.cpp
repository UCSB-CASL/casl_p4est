#include "my_p4est_multialloy_var2.h"

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
  : brick(ngbd->myb), connectivity(ngbd->p4est->connectivity), p4est(ngbd->p4est), ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy), ngbd(ngbd),
    temperature_n(NULL), temperature_np1(NULL), t_interface(NULL),
    cs_n(NULL), cs_np1(NULL), cl_n(NULL), cl_np1(NULL), c_interface(NULL),
    cs_sec_n(NULL), cs_sec_np1(NULL), cl_sec_n(NULL), cl_sec_np1(NULL), c_sec_interface(NULL),
    normal_velocity_np1(NULL),
    phi(NULL),
    kappa(NULL), rhs(NULL),
    tl(NULL), ts(NULL),
  #ifdef P4_TO_P8
    theta_xz(NULL), theta_yz(NUL)
  #else
    theta(NULL)
  #endif
{
  solve_concentration_solid = false;

  /* these default values are for a NiCu alloy, as presented in
   * A Sharp Computational Method for the Simulation of the Solidification of Binary Alloys
   * by Maxime Theillard, Frederic Gibou, Tresa Pollock
   */
//  double rho           = 8.88e-3;  /* kg.cm-3    */
//  double heat_capacity = 4.6e2;    /* J.kg-1.K-1 */
  latent_heat          = 2350;      /* J.cm-3      */
  thermal_conductivity = 6.07e-1;   /* W.cm-1.K-1  */
  thermal_diffusivity  = 1.486e-1;  /* cm2.s-1     */  /* = thermal_conductivity / (rho*heat_capacity) */
  solute_diffusivity_l = 1e-5;      /* cm2.s-1     */
  solute_diffusivity_s = 1e-13;     /* cm2.s-1     */
  cooling_velocity     = 0.01;      /* cm.s-1      */
  kp                   = 0.86;
  c0                   = 0.40831;   /* at frac.    */
  ml                   = -357;      /* K / at frac.*/
  epsilon_anisotropy   = 0.05;
  epsilon_c            = 2.7207e-5; /* cm.K     */
  epsilon_v            = 2.27e-2;   /* s.K.cm-1 */

  solute_diffusivity_l_sec = 1e-5;      /* cm2.s-1     */
  solute_diffusivity_s_sec = 1e-13;     /* cm2.s-1     */
  kp_sec                   = 0.86;
  c0_sec                   = 0.0;   /* at frac.    */
  ml_sec                   = -357;      /* K / at frac.*/

  ::dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  dxyz_min = MIN(dxyz[0],dxyz[1],dxyz[2]);
  dxyz_max = MAX(dxyz[0],dxyz[1],dxyz[2]);
  diag = sqrt(SQR(dxyz[0])+SQR(dxyz[1])+SQR(dxyz[2]));
#else
  dxyz_min = MIN(dxyz[0],dxyz[1]);
  dxyz_max = MAX(dxyz[0],dxyz[1]);
  diag = sqrt(SQR(dxyz[0])+SQR(dxyz[1]));
#endif
//  dxyz_close_interface = 4*dxyz_min;
  dxyz_close_interface = 1.2*dxyz_max;


  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_interface_n  [dir] = NULL;
    v_interface_np1[dir] = NULL;
    normal[dir] = NULL;
  }

  dt_method = 0;
  velocity_tol = 1.e-5;
  first_step = true;
  cfl_number = 0.5;
  order_of_extension = 1;

  use_more_points_for_extension = false;
  use_quadratic_form = false;
  temperature_interpolation_simple = true;


  ierr = VecCreateGhostNodes(p4est, nodes, &temperature_multiplier); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &concentration_multiplier); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &concentration_multiplier_interface); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &concentration_sec_multiplier); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &bc_error); CHKERRXX(ierr);
}




my_p4est_multialloy_t::~my_p4est_multialloy_t()
{
  if(temperature_n  !=NULL) { ierr = VecDestroy(temperature_n);   CHKERRXX(ierr); }
  if(temperature_np1!=NULL) { ierr = VecDestroy(temperature_np1); CHKERRXX(ierr); }
  if(t_interface    !=NULL) { ierr = VecDestroy(t_interface);     CHKERRXX(ierr); }

  if(cl_n       !=NULL) { ierr = VecDestroy(cl_n);        CHKERRXX(ierr); }
  if(cs_n       !=NULL) { ierr = VecDestroy(cs_n);        CHKERRXX(ierr); }
  if(cl_np1     !=NULL) { ierr = VecDestroy(cl_np1);      CHKERRXX(ierr); }
  if(cs_np1     !=NULL) { ierr = VecDestroy(cs_np1);      CHKERRXX(ierr); }
  if(c_interface!=NULL) { ierr = VecDestroy(c_interface); CHKERRXX(ierr); }

  if(cl_sec_n       !=NULL) { ierr = VecDestroy(cl_sec_n);        CHKERRXX(ierr); }
  if(cs_sec_n       !=NULL) { ierr = VecDestroy(cs_sec_n);        CHKERRXX(ierr); }
  if(cl_sec_np1     !=NULL) { ierr = VecDestroy(cl_sec_np1);      CHKERRXX(ierr); }
  if(cs_sec_np1     !=NULL) { ierr = VecDestroy(cs_sec_np1);      CHKERRXX(ierr); }
  if(c_sec_interface!=NULL) { ierr = VecDestroy(c_sec_interface); CHKERRXX(ierr); }

  if(normal_velocity_np1!=NULL) { ierr = VecDestroy(normal_velocity_np1); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(v_interface_n  [dir] != NULL) { ierr = VecDestroy(v_interface_n  [dir]); CHKERRXX(ierr); }
    if(v_interface_np1[dir] != NULL) { ierr = VecDestroy(v_interface_np1[dir]); CHKERRXX(ierr); }
    if(normal         [dir] != NULL) { ierr = VecDestroy(normal         [dir]); CHKERRXX(ierr); }
  }

  if(kappa!=NULL) { ierr = VecDestroy(kappa); CHKERRXX(ierr); }
  if(phi  !=NULL) { ierr = VecDestroy(phi);   CHKERRXX(ierr); }

  if(rhs!=NULL) { ierr = VecDestroy(rhs); CHKERRXX(ierr); }


  ierr = VecDestroy(temperature_multiplier);   CHKERRXX(ierr);
  ierr = VecDestroy(concentration_multiplier);   CHKERRXX(ierr);
  ierr = VecDestroy(concentration_multiplier_interface);   CHKERRXX(ierr);
  ierr = VecDestroy(concentration_sec_multiplier);   CHKERRXX(ierr);
  ierr = VecDestroy(bc_error);   CHKERRXX(ierr);

  /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, brick);
}




void my_p4est_multialloy_t::set_parameters(double latent_heat,
                                         double thermal_conductivity,
                                         double thermal_diffusivity,
                                         double solute_diffusivity_l,
                                         double solute_diffusivity_s,
                                         double cooling_velocity,
                                         double kp,
                                         double c0,
                                         double ml,
                                         double Tm,
                                         double epsilon_anisotropy,
                                         double epsilon_c,
                                         double epsilon_v,
                                         double scaling, double solute_diffusivity_l_sec, double solute_diffusivity_s_sec, double kp_sec, double c0_sec, double ml_sec)
{
  this->latent_heat          = latent_heat;
  this->thermal_conductivity = thermal_conductivity;
  this->thermal_diffusivity  = thermal_diffusivity;
  this->solute_diffusivity_l = solute_diffusivity_l;
  this->solute_diffusivity_s = solute_diffusivity_s;
  this->cooling_velocity     = cooling_velocity;
  this->kp                   = kp;
  this->c0                   = c0;
  this->ml                   = ml;
  this->Tm                   = Tm;
  this->epsilon_anisotropy   = epsilon_anisotropy;
  this->epsilon_c            = epsilon_c;
  this->epsilon_v            = epsilon_v;
  this->scaling              = scaling;

  this->solute_diffusivity_l_sec = solute_diffusivity_l_sec;
  this->solute_diffusivity_s_sec = solute_diffusivity_s_sec;
  this->kp_sec                   = kp_sec;
  this->c0_sec                   = c0_sec;
  this->ml_sec                   = ml_sec;
}





void my_p4est_multialloy_t::set_phi(Vec phi)
{
  this->phi = phi;

#ifdef P4_TO_P8
  ierr = VecDuplicate(phi, &theta_xz); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &theta_yz); CHKERRXX(ierr);
#else
  ierr = VecDuplicate(phi, &theta); CHKERRXX(ierr);
#endif

  compute_normal_and_curvature();

  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
}




#ifdef P4_TO_P8
void my_p4est_multialloy_t::set_bc(WallBC3D& bc_wall_type_t,
                                WallBC3D& bc_wall_type_c,
                                CF_3& bc_wall_value_t,
                                CF_3& bc_wall_value_cs,
                                CF_3& bc_wall_value_cl)
#else
void my_p4est_multialloy_t::set_bc(WallBC2D& bc_wall_type_t,
                                WallBC2D& bc_wall_type_c,
                                CF_2& bc_wall_value_t,
                                CF_2& bc_wall_value_cs,
                                CF_2& bc_wall_value_cl, CF_2 &bc_wall_value_cs_sec, CF_2 &bc_wall_value_cl_sec)
#endif
{
  bc_t.setWallTypes(bc_wall_type_t);
  bc_t.setWallValues(bc_wall_value_t);

  bc_cs.setWallTypes(bc_wall_type_c);
  bc_cs.setWallValues(bc_wall_value_cs);
  bc_cs.setInterfaceType(DIRICHLET);

  bc_cl.setWallTypes(bc_wall_type_c);
  bc_cl.setWallValues(bc_wall_value_cl);
  bc_cl.setInterfaceType(DIRICHLET);

  bc_cs_sec.setWallTypes(bc_wall_type_c);
  bc_cs_sec.setWallValues(bc_wall_value_cs_sec);
  bc_cs_sec.setInterfaceType(DIRICHLET);

  bc_cl_sec.setWallTypes(bc_wall_type_c);
  bc_cl_sec.setWallValues(bc_wall_value_cl_sec);
  bc_cl_sec.setInterfaceType(ROBIN);
  bc_cl_sec.setInterfaceValue(zero);
}





void my_p4est_multialloy_t::set_temperature(Vec temperature)
{
  temperature_n = temperature;

  ierr = VecDuplicate(temperature_n, &temperature_np1); CHKERRXX(ierr);

  Vec src, out;
  ierr = VecGhostGetLocalForm(temperature_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(temperature_n, &t_interface); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::set_concentration(Vec cl, Vec cs, Vec cl_sec, Vec cs_sec)
{
  cl_n = cl;
  cs_n = cs;

  cl_sec_n = cl_sec;
  cs_sec_n = cs_sec;

  Vec src, out;

  ierr = VecDuplicate(cl_n, &cl_np1); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cl_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(cs_n, &cs_np1); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cs_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cs_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(cl_n, &c_interface); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(c_interface, &src); CHKERRXX(ierr);
  ierr = VecSet(src, c0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(c_interface, &src); CHKERRXX(ierr);



  ierr = VecDuplicate(cl_sec_n, &cl_sec_np1); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cl_sec_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cl_sec_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_sec_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_sec_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(cs_sec_n, &cs_sec_np1); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cs_sec_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cs_sec_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_sec_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_sec_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(cl_sec_n, &c_sec_interface); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(c_sec_interface, &src); CHKERRXX(ierr);
  ierr = VecSet(src, c0_sec); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(c_sec_interface, &src); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::set_normal_velocity(Vec v)
{
  normal_velocity_np1 = v;

  Vec src;
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(v, &v_interface_np1[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(v_interface_np1[dir], &src); CHKERRXX(ierr);
    ierr = VecSet(src, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(v_interface_np1[dir], &src); CHKERRXX(ierr);

    ierr = VecDuplicate(v, &v_interface_n[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(v_interface_n[dir], &src); CHKERRXX(ierr);
    ierr = VecSet(src, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(v_interface_n[dir], &src); CHKERRXX(ierr);
  }
}





void my_p4est_multialloy_t::set_dt( double dt )
{
  dt_nm1 = dt;
  dt_n   = dt;
}





void my_p4est_multialloy_t::compute_normal_and_curvature()
{
  /* normal */
  double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(normal[dir]!=NULL) { ierr = VecDestroy(normal[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &normal[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
  if(kappa!=NULL) { ierr = VecDestroy(kappa); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    qnnn = ngbd->get_neighbors(n);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
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
    ierr = VecGhostUpdateBegin(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    qnnn = ngbd->get_neighbors(n);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    normal_p[2][n] = qnnn.dz_central(phi_p);
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
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* curvature */

  Vec kappa_tmp;
  ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);
  double *kappa_p;
  ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
#else
    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dxyz_max), -1/dxyz_max);
#endif
  }
  ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
#else
    kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dxyz_max), -1/dxyz_max);
#endif
  }
  ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

  /* angle between normal and direction of growth */

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

#ifdef P4_TO_P8
  Vec theta_xz_tmp; double *theta_xz_tmp_p;
  Vec theta_yz_tmp; double *theta_yz_tmp_p;
  ierr = VecDuplicate(phi, &theta_xz_tmp); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &theta_yz_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
#else
  Vec theta_tmp; double *theta_tmp_p;
  ierr = VecDuplicate(phi, &theta_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
#endif


  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
    theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
#else
    theta_tmp_p[n] = atan2(normal_p[1][n], normal_p[0][n]);
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

#ifdef P4_TO_P8
  ierr = VecRestoreArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi, theta_xz_tmp, theta_xz);
  ls.extend_from_interface_to_whole_domain_TVD(phi, theta_yz_tmp, theta_yz);
  ierr = VecDestroy(theta_xz_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(theta_yz_tmp); CHKERRXX(ierr);
#else
  ierr = VecRestoreArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi, theta_tmp, theta);
  ierr = VecDestroy(theta_tmp); CHKERRXX(ierr);
#endif

}




void my_p4est_multialloy_t::compute_normal_velocity()
{
  Vec v_gamma;
  ierr = VecDuplicate(phi, &v_gamma); CHKERRXX(ierr);
  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
  double *cl_n_p, *c_interface_p;
  ierr = VecGetArray(cl_np1, &cl_n_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);
//  ierr = VecGetArray(cl_sec_np1, &cl_n_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  if(solve_concentration_solid)
  {
    double *cs_n_p;
    ierr = VecGetArray(cs_n, &cs_n_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n] + qnnn.dz_central(cl_n_p)*normal_p[2][n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*normal_p[0][n] + qnnn.dy_central(cs_n_p)*normal_p[1][n] + qnnn.dz_central(cs_n_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*normal_p[0][n] + qnnn.dy_central(cs_n_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = (dcs_dn*solute_diffusivity_s - dcl_dn*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
    }
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n] + qnnn.dz_central(cl_n_p)*normal_p[2][n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*normal_p[0][n] + qnnn.dy_central(cs_n_p)*normal_p[1][n] + qnnn.dz_central(cs_n_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*normal_p[0][n] + qnnn.dy_central(cs_n_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = (dcs_dn*solute_diffusivity_s - dcl_dn*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
    }
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(cs_n, &cs_n_p); CHKERRXX(ierr);
  }
  else
  {
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n] + qnnn.dz_central(cl_n_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = -dcl_dn*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
//      if (v_gamma_p[n] < 0) v_gamma_p[n] = 0;
    }
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n] + qnnn.dz_central(cl_n_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*normal_p[0][n] + qnnn.dy_central(cl_n_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = -dcl_dn*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
//      if (v_gamma_p[n] < 0) v_gamma_p[n] = 0;
    }
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(cl_np1, &cl_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(cl_sec_np1, &cl_n_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma, normal_velocity_np1);

  ierr = VecDestroy(v_gamma); CHKERRXX(ierr);

  /* compute maximum normal velocity for convergence of v_gamma scaling */
  vgamma_max = 0;
  const double *phi_p, *normal_velocity_np1_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(fabs(phi_p[n]) < dxyz_close_interface)
      vgamma_max = MAX(vgamma_max, fabs(normal_velocity_np1_p[n]));
  }
  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &vgamma_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}





void my_p4est_multialloy_t::compute_velocity()
{
  Vec v_gamma[P4EST_DIM];
  double *v_gamma_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(normal[dir], &v_gamma[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
  }

  double *cl_np1_p, *c_interface_p;
  ierr = VecGetArray(cl_np1, &cl_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);

//  ierr = VecGetArray(cl_sec_np1, &cl_np1_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

  double *normal_p[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGetArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
  }

  quad_neighbor_nodes_of_node_t qnnn;
  if(solve_concentration_solid)
  {
    double *cs_np1_p;
    ierr = VecGetArray(cs_np1, &cs_np1_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

      v_gamma_p[0][n] = (qnnn.dx_central(cs_np1_p)*solute_diffusivity_s - qnnn.dx_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[1][n] = (qnnn.dy_central(cs_np1_p)*solute_diffusivity_s - qnnn.dy_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      v_gamma_p[2][n] = (qnnn.dz_central(cs_np1_p)*solute_diffusivity_s - qnnn.dz_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGhostUpdateBegin(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

      v_gamma_p[0][n] = (qnnn.dx_central(cs_np1_p)*solute_diffusivity_s - qnnn.dx_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[1][n] = (qnnn.dy_central(cs_np1_p)*solute_diffusivity_s - qnnn.dy_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      v_gamma_p[2][n] = (qnnn.dz_central(cs_np1_p)*solute_diffusivity_s - qnnn.dz_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGhostUpdateEnd(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(cs_np1, &cs_np1_p); CHKERRXX(ierr);
  }
  else
  {
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

      v_gamma_p[0][n] = -qnnn.dx_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[1][n] = -qnnn.dy_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      v_gamma_p[2][n] = -qnnn.dz_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
      if (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n] < 0)
      {
        v_gamma_p[0][n] = 0;
        v_gamma_p[1][n] = 0;
      }
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGhostUpdateBegin(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

      v_gamma_p[0][n] = -qnnn.dx_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[1][n] = -qnnn.dy_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      v_gamma_p[2][n] = -qnnn.dz_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
      if (v_gamma_p[0][n]*normal_p[0][n] + v_gamma_p[1][n]*normal_p[1][n] < 0)
      {
        v_gamma_p[0][n] = 0;
        v_gamma_p[1][n] = 0;
      }
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGhostUpdateEnd(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  ierr = VecRestoreArray(cl_np1, &cl_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(cl_sec_np1, &cl_np1_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGetArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma[dir], v_interface_np1[dir]);
    ierr = VecDestroy(v_gamma[dir]); CHKERRXX(ierr);
  }
}





void my_p4est_multialloy_t::compute_velocity_from_temperature()
{
  /* compute the normal velocity from the temperature field and compare with the one computed from the concentration */
  Vec Tl, Ts;
  ierr = VecDuplicate(phi, &Tl); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &Ts); CHKERRXX(ierr);

  /* make copy of the temperature field */
  Vec Tnp1_loc, Tl_loc, Ts_loc;
  ierr = VecGhostGetLocalForm(Tl, &Tl_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(Ts, &Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_np1, &Tnp1_loc); CHKERRXX(ierr);
  ierr = VecCopy(Tnp1_loc, Tl_loc); CHKERRXX(ierr);
  ierr = VecCopy(Tnp1_loc, Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(Tl, &Tl_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(Ts, &Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_np1, &Tnp1_loc); CHKERRXX(ierr);

  /* extend temperatures over interface */
  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, Ts);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi, Tl);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  Vec v_gamma[P4EST_DIM];
  double *v_gamma_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(normal[dir], &v_gamma[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
  }

  const double *temperature_l_np1_p, *temperature_s_np1_p;
  ierr = VecGetArrayRead(Tl, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(Ts, &temperature_s_np1_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    v_gamma_p[0][n] = thermal_conductivity/latent_heat * (qnnn.dx_central(temperature_s_np1_p) - qnnn.dx_central(temperature_l_np1_p));
    v_gamma_p[1][n] = thermal_conductivity/latent_heat * (qnnn.dy_central(temperature_s_np1_p) - qnnn.dy_central(temperature_l_np1_p));
#ifdef P4_TO_P8
    v_gamma_p[2][n] = thermal_conductivity/latent_heat * (qnnn.dz_central(temperature_s_np1_p) - qnnn.dz_central(temperature_l_np1_p));
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    v_gamma_p[0][n] = thermal_conductivity/latent_heat * (qnnn.dx_central(temperature_s_np1_p) - qnnn.dx_central(temperature_l_np1_p));
    v_gamma_p[1][n] = thermal_conductivity/latent_heat * (qnnn.dy_central(temperature_s_np1_p) - qnnn.dy_central(temperature_l_np1_p));
#ifdef P4_TO_P8
    v_gamma_p[2][n] = thermal_conductivity/latent_heat * (qnnn.dz_central(temperature_s_np1_p) - qnnn.dz_central(temperature_l_np1_p));
#endif
  }
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(Tl, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Ts, &temperature_s_np1_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma[dir], v_interface_np1[dir]);
    ierr = VecDestroy(v_gamma[dir]); CHKERRXX(ierr);
  }

  ierr = VecDestroy(Tl); CHKERRXX(ierr);
  ierr = VecDestroy(Ts); CHKERRXX(ierr);
}









void my_p4est_multialloy_t::solve_temperature()
{
  double *phi_p, *rhs_p, *temperature_n_p, *normal_velocity_np1_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(temperature_n, &temperature_n_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  double scaling_jump = latent_heat/thermal_conductivity * dt_n * thermal_diffusivity;
  double p_000, p_m00, p_p00, p_0m0, p_0p0;
#ifdef P4_TO_P8
  double p_00m, p_00p;
#endif

//  my_p4est_interpolation_nodes_t normal_velo_cf(ngbd);
//  normal_velo_cf.set_input(normal_velocity_np1, linear);

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    rhs_p[n] = temperature_n_p[n];

    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    if (p_000*p_m00<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_m00_m0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_mm;
      else if(qnnn.d_m00_m0==0 && qnnn.d_m00_0p==0) neigh = qnnn.node_m00_mp;
      else if(qnnn.d_m00_p0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_pm;
      else                                          neigh = qnnn.node_m00_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
    }
    if (p_000*p_p00<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_p00_m0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_mm;
      else if(qnnn.d_p00_m0==0 && qnnn.d_p00_0p==0) neigh = qnnn.node_p00_mp;
      else if(qnnn.d_p00_p0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_pm;
      else                                          neigh = qnnn.node_p00_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
    }
    if (p_000*p_0m0<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_0m0_m0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_mm;
      else if(qnnn.d_0m0_m0==0 && qnnn.d_0m0_0p==0) neigh = qnnn.node_0m0_mp;
      else if(qnnn.d_0m0_p0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_pm;
      else                                          neigh = qnnn.node_0m0_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
    }
    if (p_000*p_0p0<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_0p0_m0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_mm;
      else if(qnnn.d_0p0_m0==0 && qnnn.d_0p0_0p==0) neigh = qnnn.node_0p0_mp;
      else if(qnnn.d_0p0_p0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_pm;
      else                                          neigh = qnnn.node_0p0_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
    }
    if (p_000*p_00m<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_00m_m0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_mm;
      else if(qnnn.d_00m_m0==0 && qnnn.d_00m_0p==0) neigh = qnnn.node_00m_mp;
      else if(qnnn.d_00m_p0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_pm;
      else                                          neigh = qnnn.node_00m_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
    }
    if (p_000*p_00p<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_00p_m0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_mm;
      else if(qnnn.d_00p_m0==0 && qnnn.d_00p_0p==0) neigh = qnnn.node_00p_mp;
      else if(qnnn.d_00p_p0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_pm;
      else                                          neigh = qnnn.node_00p_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
    }
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
    if (p_000*p_m00<=0){
      p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
    }
    if (p_000*p_p00<=0){
      p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
    }
    if (p_000*p_0m0<=0){
      p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
    }
    if (p_000*p_0p0<=0){
      p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
    }
#endif
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(temperature_n, &temperature_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  my_p4est_poisson_nodes_t solver_t(ngbd);
  solver_t.set_bc(bc_t);
  solver_t.set_mu(dt_n*thermal_diffusivity);
  solver_t.set_diagonal(1);
  solver_t.set_rhs(rhs);
//  solver_t.set_is_matrix_computed(matrices_are_constructed);

  solver_t.solve(temperature_np1);

  // extend from solid to liquid
  Vec ts_tmp; ierr = VecDuplicate(phi, &ts_tmp); CHKERRXX(ierr);
  Vec tl_tmp; ierr = VecDuplicate(phi, &tl_tmp); CHKERRXX(ierr);

  Vec src, out;
  ierr = VecGhostGetLocalForm(temperature_np1, &src); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(ts_tmp, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(ts_tmp, &out); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(tl_tmp, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(tl_tmp, &out); CHKERRXX(ierr);

  ierr = VecGhostRestoreLocalForm(temperature_np1, &src); CHKERRXX(ierr);

  double *ts_tmp_p, *tl_tmp_p;
  ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  Vec mask_s; double *mask_s_p; ierr = VecDuplicate(phi, &mask_s); CHKERRXX(ierr);
  Vec mask_l; double *mask_l_p; ierr = VecDuplicate(phi, &mask_l); CHKERRXX(ierr);

  ierr = VecGetArray(mask_s, &mask_s_p); CHKERRXX(ierr);
  ierr = VecGetArray(mask_l, &mask_l_p); CHKERRXX(ierr);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  // first, extend one point away from interface
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if (phi_p[n] < 0)
    {
      mask_s_p[n] = -1.;
      mask_l_p[n] = +1.;
    } else {
      mask_s_p[n] = +1.;
      mask_l_p[n] = -1.;
    }

    if (fabs(phi_p[n]) < 1.5*diag)
    {
      qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
      if (p_000 > 0. && (p_m00<=0 || p_p00<=0 || p_0m0<=0 || p_0p0<=0)){
        ts_tmp_p[n] += fabs(p_000) * normal_velocity_np1_p[n] * latent_heat/thermal_conductivity;
        mask_s_p[n] = -1;
      }
      if (p_000 < 0. && (p_m00>=0 || p_p00>=0 || p_0m0>=0 || p_0p0>=0)){
        tl_tmp_p[n] += fabs(p_000) * normal_velocity_np1_p[n] * latent_heat/thermal_conductivity;
        mask_l_p[n] = -1;
      }
#endif
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(mask_s, &mask_s_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask_l, &mask_l_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mask_s, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask_s, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mask_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(ts_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(ts_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(tl_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(tl_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);

  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask_s, ts_tmp, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, ts_tmp, 20, order_of_extension);

  // extend from liquid to solid
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n) {phi_p[n] *= -1;}
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask_l, tl_tmp, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, tl_tmp, 20, order_of_extension);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n) {phi_p[n] *= -1;}
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if (tl != NULL) { ierr = VecDestroy(tl); CHKERRXX(ierr); } tl = tl_tmp;
  if (ts != NULL) { ierr = VecDestroy(ts); CHKERRXX(ierr); } ts = ts_tmp;

  // take average
  Vec tavg;
  ierr = VecDuplicate(phi, &tavg); CHKERRXX(ierr);

  double *tavg_p;
  ierr = VecGetArray(tavg, &tavg_p); CHKERRXX(ierr);
  ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    tavg_p[n] = 0.5*(ts_tmp_p[n]+tl_tmp_p[n]);

  ierr = VecRestoreArray(tavg, &tavg_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  // extend average from interface
  ls.extend_from_interface_to_whole_domain_TVD(phi, tavg, t_interface);

  ierr = VecDestroy(tavg); CHKERRXX(ierr);

  ierr = VecDestroy(mask_s); CHKERRXX(ierr);
  ierr = VecDestroy(mask_l); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::solve_temperature_multiplier()
{
  double *phi_p, *rhs_p, *t_interface_p, *c_interface_p, *theta_p, *normal_velocity_np1_p, *kappa_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);

  double scaling_jump = dt_n*thermal_diffusivity;

  double p_000, p_m00, p_p00, p_0m0, p_0p0;
#ifdef P4_TO_P8
  double p_00m, p_00p;
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    rhs_p[n] = 0;

    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
    if (use_quadratic_form)
    {
      if (p_000*p_m00<=0){
        p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
        double jump = - 2.0/thermal_diffusivity *
            (t_interface_p[neigh] - Tm - ml_sec*c_interface_p[neigh]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*kappa_p[neigh]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*normal_velocity_np1_p[neigh]);
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[0]) * scaling_jump;
      }
      if (p_000*p_p00<=0){
        p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
        double jump = - 2.0/thermal_diffusivity *
            (t_interface_p[neigh] - Tm - ml_sec*c_interface_p[neigh]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*kappa_p[neigh]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*normal_velocity_np1_p[neigh]);
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[0]) * scaling_jump;
      }
      if (p_000*p_0m0<=0){
        p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
        double jump = - 2.0/thermal_diffusivity *
            (t_interface_p[neigh] - Tm - ml_sec*c_interface_p[neigh]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*kappa_p[neigh]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*normal_velocity_np1_p[neigh]);
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[1]) * scaling_jump;
      }
      if (p_000*p_0p0<=0){
        p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
        double jump = - 2.0/thermal_diffusivity *
            (t_interface_p[neigh] - Tm - ml_sec*c_interface_p[neigh]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*kappa_p[neigh]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[neigh]))*normal_velocity_np1_p[neigh]);
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[1]) * scaling_jump;
      }
    } else {
      double jump = 1.0/ml/thermal_diffusivity;
      if (p_000*p_m00<=0){
        p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[0]) * scaling_jump;
      }
      if (p_000*p_p00<=0){
        p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[0]) * scaling_jump;
      }
      if (p_000*p_0m0<=0){
        p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[1]) * scaling_jump;
      }
      if (p_000*p_0p0<=0){
        p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
        rhs_p[n] += fabs(phi_p[neigh]) * jump / SQR(dxyz[1]) * scaling_jump;
      }
    }
#endif
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

  BoundaryConditions2D bc_t_mult;

  bc_t_mult.setWallTypes(bc_t.getWallType());
  bc_t_mult.setWallValues(zero);

  my_p4est_poisson_nodes_t solver_t(ngbd);
  solver_t.set_bc(bc_t_mult);
  solver_t.set_mu(dt_n*thermal_diffusivity);
  solver_t.set_diagonal(1);
  solver_t.set_rhs(rhs);
//  solver_t.set_is_matrix_computed(matrices_are_constructed);

  solver_t.solve(temperature_multiplier);

  // extend from solid to liquid
  Vec ts_tmp; ierr = VecDuplicate(phi, &ts_tmp); CHKERRXX(ierr);
  Vec tl_tmp; ierr = VecDuplicate(phi, &tl_tmp); CHKERRXX(ierr);

  Vec src, out;
  ierr = VecGhostGetLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(ts_tmp, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(ts_tmp, &out); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(tl_tmp, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(tl_tmp, &out); CHKERRXX(ierr);

  ierr = VecGhostRestoreLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);

  double *ts_tmp_p, *tl_tmp_p;
  ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  Vec mask_s; double *mask_s_p; ierr = VecDuplicate(phi, &mask_s); CHKERRXX(ierr);
  Vec mask_l; double *mask_l_p; ierr = VecDuplicate(phi, &mask_l); CHKERRXX(ierr);

  ierr = VecGetArray(mask_s, &mask_s_p); CHKERRXX(ierr);
  ierr = VecGetArray(mask_l, &mask_l_p); CHKERRXX(ierr);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  // first, extend one point away from interface

  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if (phi_p[n] < 0)
    {
      mask_s_p[n] = -1.;
      mask_l_p[n] = +1.;
    } else {
      mask_s_p[n] = +1.;
      mask_l_p[n] = -1.;
    }

    if (fabs(phi_p[n]) < 2.*diag)
    {
      qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
      if (p_000 > 0. && (p_m00<=0 || p_p00<=0 || p_0m0<=0 || p_0p0<=0))
      {
        double jump;
        if (use_quadratic_form)
        {
          jump = - 2.0/thermal_diffusivity *
              (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
               + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
               + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]);
        } else {
          jump = 1.0/ml/thermal_diffusivity;
        }

        ts_tmp_p[n] += fabs(p_000) * jump;
        mask_s_p[n] = -1;
      }
      if (p_000 < 0. && (p_m00>=0 || p_p00>=0 || p_0m0>=0 || p_0p0>=0))
      {
        double jump;
        if (use_quadratic_form)
        {
          jump = - 2.0/thermal_diffusivity *
              (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
               + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
               + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]);
        } else {
          jump = 1.0/ml/thermal_diffusivity;
        }

        tl_tmp_p[n] += fabs(p_000) * jump;
        mask_l_p[n] = -1;
      }
#endif
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(mask_s, &mask_s_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask_l, &mask_l_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mask_s, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask_s, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mask_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(ts_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(ts_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(tl_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(tl_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask_s, ts_tmp, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, ts_tmp, 20, order_of_extension);

  // extend from liquid to solid
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n) {phi_p[n] *= -1;}
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask_l, tl_tmp, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, tl_tmp, 20, order_of_extension);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n) {phi_p[n] *= -1;}
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  // take average
  ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    ts_tmp_p[n] = 0.5*(ts_tmp_p[n]+tl_tmp_p[n]);

  ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

  // extend average from interface
  ls.extend_from_interface_to_whole_domain_TVD(phi, ts_tmp, temperature_multiplier);

  ierr = VecDestroy(ts_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(tl_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(mask_s); CHKERRXX(ierr);
  ierr = VecDestroy(mask_l); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi, temperature_multiplier);

//  // extend from solid to liquid
//  Vec ts_tmp;
//  ierr = VecDuplicate(phi, &ts_tmp); CHKERRXX(ierr);

//  Vec src, out;
//  ierr = VecGhostGetLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);
//  ierr = VecGhostGetLocalForm(ts_tmp, &out); CHKERRXX(ierr);
//  ierr = VecCopy(src, out); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(ts_tmp, &out); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi, ts_tmp, 20, order_of_extension);

//  // extend from liquid to solid
//  Vec tl_tmp;
//  ierr = VecDuplicate(phi, &tl_tmp); CHKERRXX(ierr);

//  ierr = VecGhostGetLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);
//  ierr = VecGhostGetLocalForm(tl_tmp, &out); CHKERRXX(ierr);
//  ierr = VecCopy(src, out); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(temperature_multiplier, &src); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(tl_tmp, &out); CHKERRXX(ierr);

//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//    phi_p[n] *= -1;
//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

//  ls.extend_Over_Interface_TVD(phi, tl_tmp, 20, order_of_extension);

//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//    phi_p[n] *= -1;
//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

//  // take average
//  double *ts_tmp_p, *tl_tmp_p;
//  ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
//  ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//    ts_tmp_p[n] = 0.5*(ts_tmp_p[n]+tl_tmp_p[n]);

//  ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);

//  // extend average from interface
//  ls.extend_from_interface_to_whole_domain_TVD(phi, ts_tmp, temperature_multiplier);

//  ierr = VecDestroy(ts_tmp); CHKERRXX(ierr);
//  ierr = VecDestroy(tl_tmp); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::solve_concentration()
{
  my_p4est_interpolation_nodes_t interface_value_c(ngbd);
  interface_value_c.set_input(c_interface, linear);
  bc_cl.setInterfaceValue(interface_value_c);

  /* compute the rhs for concentration */
  Vec src, out;
  ierr = VecGhostGetLocalForm(cl_n, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(rhs , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rhs , &out); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  my_p4est_poisson_nodes_t solver_c(ngbd);
  solver_c.set_phi(phi);
  solver_c.set_bc(bc_cl);
  solver_c.set_mu(dt_n*solute_diffusivity_l);
  solver_c.set_diagonal(1);
  solver_c.set_rhs(rhs);
//  solver_c.set_is_matrix_computed(matrices_are_constructed);

  solver_c.solve(cl_np1);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, cl_np1, 20, order_of_extension);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cs_np1, &src); CHKERRXX(ierr);
  ierr = VecSet(src, kp*c0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_np1, &src); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::solve_concentration_multiplier()
{
  /* initialize the boundary condition on the interface */
  Vec c_interface_tmp;
  ierr = VecDuplicate(phi, &c_interface_tmp); CHKERRXX(ierr);

  double *c_interface_tmp_p, *kappa_p, *t_interface_p, *normal_velocity_np1_p, *rhs_p, *theta_p;
  double *c_interface_p, *c_sec_interface_p, *concentration_sec_multiplier_p, *temperature_multiplier_p;
  ierr = VecGetArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(concentration_sec_multiplier, &concentration_sec_multiplier_p); CHKERRXX(ierr);
  ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
#else
    c_interface_tmp_p[n] = -1.0/(1.-kp)/c_interface_p[n] *
        (thermal_diffusivity*latent_heat/thermal_conductivity * temperature_multiplier_p[n]
         + (1.-kp_sec)*concentration_sec_multiplier_p[n]*c_sec_interface_p[n]
         + epsilon_v*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n]))/ml);
#endif
        rhs_p[n] = 0;
  }

  ierr = VecRestoreArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(concentration_sec_multiplier, &concentration_sec_multiplier_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, c_interface_tmp, concentration_multiplier_interface);

  my_p4est_interpolation_nodes_t interface_value_c(ngbd);
  interface_value_c.set_input(concentration_multiplier_interface, linear);

  BoundaryConditions2D bc_cl_mult;
  bc_cl_mult.setWallTypes(bc_cl.getWallType());
  bc_cl_mult.setWallValues(zero);
  bc_cl_mult.setInterfaceType(DIRICHLET);
  bc_cl_mult.setInterfaceValue(interface_value_c);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  my_p4est_poisson_nodes_t solver_c(ngbd);
  solver_c.set_phi(phi);
  solver_c.set_bc(bc_cl_mult);
  solver_c.set_mu(dt_n*solute_diffusivity_l);
  solver_c.set_diagonal(1);
  solver_c.set_rhs(rhs);
//  solver_c.set_is_matrix_computed(matrices_are_constructed);

  solver_c.solve(concentration_multiplier);

  ls.extend_Over_Interface_TVD(phi, concentration_multiplier, 20, order_of_extension);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecDestroy(c_interface_tmp); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::solve_concentration_sec()
{
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  Vec robin_coef;
  ierr = VecDuplicate(phi, &robin_coef); CHKERRXX(ierr);

  double *robin_coef_p;
  const double *normal_velocity_np1_p;
  ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
//    robin_coef_p[n] = -cooling_velocity*(1-kp)/solute_diffusivity_l;
    robin_coef_p[n] = -normal_velocity_np1_p[n]*(1-kp_sec)/solute_diffusivity_l_sec;
  }

  ierr = VecRestoreArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  Vec src, out;
  ierr = VecGhostGetLocalForm(cl_sec_n, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(rhs , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_sec_n, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rhs , &out); CHKERRXX(ierr);


  my_p4est_poisson_nodes_t solver_c(ngbd);
  solver_c.set_phi(phi);
  solver_c.set_bc(bc_cl_sec);
  solver_c.set_mu(dt_n*solute_diffusivity_l_sec);
  solver_c.set_diagonal(1);
  solver_c.set_rhs(rhs);
  solver_c.set_robin_coef(robin_coef);
  solver_c.solve(cl_sec_np1);

  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi, cl_sec_np1, 20, order_of_extension);

  Vec mask = solver_c.get_mask();

  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask, cl_sec_np1, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, cl_sec_np1, 20, order_of_extension);


//  double err_bc = 0;
//  const double *normal_p[P4EST_DIM];
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  double *cl_sec_np1_p;
//  VecGetArray(phi, &phi_p);
//  VecGetArray(robin_coef, &robin_coef_p);
//  VecGetArray(cl_sec_np1, &cl_sec_np1_p);
//  for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
//  {
//    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

//    /* cafeful ! minus sign because cl is defined for phi>0 */
//#ifdef P4_TO_P8
//    double dcl_dn = -(qnnn.dx_central(cl_sec_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_sec_np1_p)*normal_p[1][n] + qnnn.dz_central(cl_sec_np1_p)*normal_p[2][n]);
//#else
//    double dcl_dn = -(qnnn.dx_central(cl_sec_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_sec_np1_p)*normal_p[1][n]);
//#endif

//    if(phi_p[n]<0 && phi_p[n]>-dxyz_min)
//    {
//      double x = node_x_fr_n(n, p4est, nodes);
//      double y = node_y_fr_n(n, p4est, nodes);
//      err_bc = MAX(err_bc, fabs(dcl_dn + robin_coef_p[n]*cl_sec_np1_p[n] - bc_cl_sec.interfaceValue(x,y) ));
//    }
//  }
//  VecRestoreArray(cl_sec_np1, &cl_sec_np1_p);
//  VecRestoreArray(phi, &phi_p);
//  VecRestoreArray(robin_coef, &robin_coef_p);
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

  ierr = VecDestroy(robin_coef); CHKERRXX(ierr);

//  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_bc, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//  ierr = PetscPrintf(p4est->mpicomm, "Error on Robin boundary condition for cl_np1 : %e\n", err_bc); CHKERRXX(ierr);


  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi, cl_sec_np1, c_sec_interface);
}





void my_p4est_multialloy_t::solve_concentration_sec_multiplier()
{
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  Vec robin_coef;
  ierr = VecDuplicate(phi, &robin_coef); CHKERRXX(ierr);

  double *robin_coef_p;
  const double *normal_velocity_np1_p;
  ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  Vec interface_value;
  ierr = VecDuplicate(phi, &interface_value); CHKERRXX(ierr);

  double *interface_value_p;
  const double *t_interface_p, *c_interface_p, *kappa_p, *theta_p;
  ierr = VecGetArray(interface_value, &interface_value_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(theta, &theta_p); CHKERRXX(ierr);


  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
//    robin_coef_p[n] = -cooling_velocity*(1-kp)/solute_diffusivity_l;
    robin_coef_p[n] = -normal_velocity_np1_p[n]*(1.-kp_sec)/solute_diffusivity_l_sec;

    rhs_p[n] = 0.;

    if (use_quadratic_form)
    {
      interface_value_p[n] = 2.0*ml_sec/solute_diffusivity_l_sec*
          (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
           + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
           + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]);
    } else {
      interface_value_p[n] = -ml_sec/ml/solute_diffusivity_l_sec;
    }
  }

  ierr = VecRestoreArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(interface_value, &interface_value_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(theta, &theta_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interface_value_cl(ngbd);
  interface_value_cl.set_input(interface_value, linear);

  BoundaryConditions2D bc_cl_mult;
  bc_cl_mult.setWallTypes(bc_cl.getWallType());
  bc_cl_mult.setWallValues(zero);
  bc_cl_mult.setInterfaceType(ROBIN);
  bc_cl_mult.setInterfaceValue(interface_value_cl);

  my_p4est_poisson_nodes_t solver_c(ngbd);
  solver_c.set_phi(phi);
  solver_c.set_bc(bc_cl_mult);
  solver_c.set_mu(dt_n*solute_diffusivity_l_sec);
  solver_c.set_diagonal(1);
  solver_c.set_rhs(rhs);
  solver_c.set_robin_coef(robin_coef);

  Vec c_mult_tmp;
  ierr = VecDuplicate(phi, &c_mult_tmp); CHKERRXX(ierr);

  solver_c.solve(c_mult_tmp);

  my_p4est_level_set_t ls(ngbd);

  Vec mask = solver_c.get_mask();

  if (use_more_points_for_extension)
    ls.extend_Over_Interface_TVD(phi, mask, c_mult_tmp, 20, order_of_extension);
  else
    ls.extend_Over_Interface_TVD(phi, c_mult_tmp, 20, order_of_extension);

//  ls.extend_Over_Interface_TVD(phi, c_mult_tmp, 20, order_of_extension);

  ierr = VecDestroy(robin_coef); CHKERRXX(ierr);
  ierr = VecDestroy(interface_value); CHKERRXX(ierr);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi, c_mult_tmp, concentration_sec_multiplier);

  ierr = VecDestroy(c_mult_tmp); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::compute_dt()
{
  /* compute the size of smallest detail, i.e. maximum curvature */
  const double *kappa_p;
  ierr = VecGetArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  double kappa_max = 0;
  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(fabs(phi_p[n]) < dxyz_close_interface)
        kappa_max = MAX(kappa_max, fabs(kappa_p[n]));
    }
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  kappa_max = MIN(kappa_max, 1/dxyz_min);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &kappa_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = PetscPrintf(p4est->mpicomm, "Maximum curvature = %e\n", kappa_max); CHKERRXX(ierr);

  /* compute the maximum velocity at the interface */
  double u_max = 0;
  const double *v_interface_np1_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(v_interface_np1[dir], &v_interface_np1_p); CHKERRXX(ierr);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(fabs(phi_p[n]) < dxyz_close_interface)
        u_max = MAX(u_max, fabs(v_interface_np1_p[n]));
    }
    ierr = VecRestoreArrayRead(v_interface_np1[dir], &v_interface_np1_p); CHKERRXX(ierr);
  }
//  {
//    ierr = VecGetArrayRead(normal_velocity_np1, &v_interface_np1_p); CHKERRXX(ierr);
//    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//    {
//      if(fabs(phi_p[n]) < dxyz_close_interface)
//        u_max = MAX(u_max, fabs(v_interface_np1_p[n]));
//    }
//    ierr = VecRestoreArrayRead(normal_velocity_np1, &v_interface_np1_p); CHKERRXX(ierr);
//  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  dt_nm1 = dt_n;
  switch (dt_method) {
    case 0: dt_n = 1. * sqrt(dxyz_min)*dxyz_min * MIN(1/u_max, 1/cooling_velocity);
      break;
    case 1: dt_n = cfl_number * dxyz_min * MIN(1/u_max, 1/cooling_velocity);
      break;
  }
//  dt_n = 1 * dxyz_min / MAX(u_max,1e-7);
//  dt_n = 0.1 * dxyz_min * MIN(1/u_max, 1/cooling_velocity);
//  dt_n = 0.5 * dxyz_min / u_max;
  PetscPrintf(p4est->mpicomm, "VMAX = %e, VGAMMAMAX = %e, COOLING_VELO = %e\n", u_max, vgamma_max, cooling_velocity);

//  if(dt_n>0.5/MAX(1e-7, MAX(u_max,vgamma_max)*kappa_max))
//  if(0 && dt_n>0.5/(MAX(u_max,vgamma_max)*MAX(kappa_max,1/(100*dxyz_min))))
//  {
//    dt_n = MIN(dt_n, 0.5/MAX(1e-7, MAX(u_max,vgamma_max)*kappa_max));
//    dt_n = MIN(dt_n, 0.5/(MAX(u_max,vgamma_max)*MAX(kappa_max,1/(100*dxyz_min))));
//    ierr = PetscPrintf(p4est->mpicomm, "KAPPA LIMITING TIME STEP\n"); CHKERRXX(ierr);
//  }
  PetscPrintf(p4est->mpicomm, "dt = %e\n", dt_n);
}





void my_p4est_multialloy_t::update_grid()
{
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd, ngbd);

  /* bousouf update this for second order in time */
  if (first_step || 1)
    sl.update_p4est(v_interface_np1, dt_n, phi);
  else // Daniil: synchronized with the serial version (sort of)
    sl.update_p4est(v_interface_n, v_interface_np1, dt_nm1, dt_n, phi);

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  ierr = VecDestroy(temperature_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_n); CHKERRXX(ierr);

  if (temperature_interpolation_simple)
  {
    interp.set_input(temperature_np1, quadratic_non_oscillatory);
    interp.interpolate(temperature_n);

  } else {

    Vec tl_tmp; ierr = VecDuplicate(phi, &tl_tmp); CHKERRXX(ierr);
    interp.set_input(tl, quadratic_non_oscillatory);
    interp.interpolate(tl_tmp);

    Vec ts_tmp; ierr = VecDuplicate(phi, &ts_tmp); CHKERRXX(ierr);
    interp.set_input(ts, quadratic_non_oscillatory);
    interp.interpolate(ts_tmp);

    double *phi_p, *tl_tmp_p, *ts_tmp_p, *t_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);
    ierr = VecGetArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
    ierr = VecGetArray(temperature_n, &t_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      if (phi_p[n] < 0.)  t_p[n] = ts_tmp_p[n];
      else                t_p[n] = tl_tmp_p[n];
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(tl_tmp, &tl_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(ts_tmp, &ts_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(temperature_n, &t_p); CHKERRXX(ierr);

    ierr = VecDestroy(tl_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(ts_tmp); CHKERRXX(ierr);
  }

  ierr = VecDestroy(temperature_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_np1); CHKERRXX(ierr);

  ierr = VecDestroy(cl_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_n); CHKERRXX(ierr);
  interp.set_input(cl_np1, quadratic_non_oscillatory);
  interp.interpolate(cl_n);
  ierr = VecDestroy(cl_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_np1); CHKERRXX(ierr);

  ierr = VecDestroy(cs_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cs_n); CHKERRXX(ierr);
  interp.set_input(cs_np1, quadratic_non_oscillatory);
  interp.interpolate(cs_n);
  ierr = VecDestroy(cs_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cs_np1); CHKERRXX(ierr);


  ierr = VecDestroy(cl_sec_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_sec_n); CHKERRXX(ierr);
  interp.set_input(cl_sec_np1, quadratic_non_oscillatory);
  interp.interpolate(cl_sec_n);
  ierr = VecDestroy(cl_sec_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_sec_np1); CHKERRXX(ierr);

  ierr = VecDestroy(cs_sec_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cs_sec_n); CHKERRXX(ierr);
  interp.set_input(cs_sec_np1, quadratic_non_oscillatory);
  interp.interpolate(cs_sec_n);
  ierr = VecDestroy(cs_sec_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cs_sec_np1); CHKERRXX(ierr);


  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(v_interface_n[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &v_interface_n[dir]); CHKERRXX(ierr);
    interp.set_input(v_interface_np1[dir], quadratic_non_oscillatory);
    interp.interpolate(v_interface_n[dir]);
    ierr = VecDestroy(v_interface_np1[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &v_interface_np1[dir]); CHKERRXX(ierr);
  }

  Vec normal_velocity_n;
  ierr = VecDuplicate(phi, &normal_velocity_n); CHKERRXX(ierr);
  interp.set_input(normal_velocity_np1, quadratic_non_oscillatory);
  interp.interpolate(normal_velocity_n);
  ierr = VecDestroy(normal_velocity_np1); CHKERRXX(ierr);
  normal_velocity_np1 = normal_velocity_n;

//  Vec kappa_n;
//  ierr = VecDuplicate(phi, &kappa_n); CHKERRXX(ierr);
//  interp.set_input(kappa, quadratic_non_oscillatory);
//  interp.interpolate(kappa_n);
//  ierr = VecDestroy(kappa); CHKERRXX(ierr);
//  kappa = kappa_n;

  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

  ierr = VecDestroy(t_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &t_interface); CHKERRXX(ierr);

  ierr = VecDestroy(c_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c_interface); CHKERRXX(ierr);

  ierr = VecDestroy(c_sec_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c_sec_interface); CHKERRXX(ierr);

  ierr = VecDestroy(temperature_multiplier); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_multiplier); CHKERRXX(ierr);

  ierr = VecDestroy(concentration_multiplier); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &concentration_multiplier); CHKERRXX(ierr);

  ierr = VecDestroy(concentration_multiplier_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &concentration_multiplier_interface); CHKERRXX(ierr);

  ierr = VecDestroy(concentration_sec_multiplier); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &concentration_sec_multiplier); CHKERRXX(ierr);

  ierr = VecDestroy(bc_error); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &bc_error); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ierr = VecDestroy(theta_xz); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &theta_xz); CHKERRXX(ierr);
  ierr = VecDestroy(theta_yz); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &theta_yz); CHKERRXX(ierr);
#else
  ierr = VecDestroy(theta); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &theta); CHKERRXX(ierr);
#endif

  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  hierarchy->update(p4est, ghost);
  ngbd->update(hierarchy, nodes);

  /* help interface to not get stuck at grid nodes */
  double fraction = 0.0001;
  double *phi_p, *normal_velocity_np1_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
//    if (fabs(phi_p[n]) < fraction*dxyz_min)
//    {
//      if (normal_velocity_np1_p[n] > 0. && phi_p[n] > 0) phi_p[n] =-EPS;
//      if (normal_velocity_np1_p[n] < 0. && phi_p[n] < 0) phi_p[n] = EPS;
//    }
    if (phi_p[n] > 0. && phi_p[n] < fraction*dxyz_min)
      phi_p[n] *= -1;
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
//  ls.perturb_level_set_function(phi, EPS);

  compute_normal_and_curvature();
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
  Vec normal_velocity_tmp;
  ierr = VecDuplicate(phi, &normal_velocity_tmp); CHKERRXX(ierr);

//  Vec normal_velocity_np1_tmp;
//  ierr = VecDuplicate(phi, &normal_velocity_np1_tmp); CHKERRXX(ierr);

  double *normal_velocity_tmp_p, *phi_p;
  double error = 1;
  double error_bc = 1;
  int iteration = 0;
  matrices_are_constructed = false;

  Vec integrand;
  ierr = VecDuplicate(phi, &integrand); CHKERRXX(ierr);

  /* initialize the boundary condition on the interface */
  solve_temperature();
  solve_concentration_sec();

  Vec c_interface_tmp;
  ierr = VecDuplicate(phi, &c_interface_tmp); CHKERRXX(ierr);

  double *c_interface_tmp_p, *kappa_p, *t_interface_p, *normal_velocity_np1_p, *theta_p;
  ierr = VecGetArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  double *c_sec_interface_p;
  ierr = VecGetArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
#else
    c_interface_tmp_p[n] = (t_interface_p[n]-Tm-ml_sec*c_sec_interface_p[n])/ml
            + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]/ml
            + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]/ml;
#endif
  }
  ierr = VecRestoreArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, c_interface_tmp, c_interface);


  while(error>velocity_tol && iteration<5)
//  while(error_bc>velocity_tol && iteration<30)
  {
//    save_VTK(iteration);
//    Vec src, out;
//    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostGetLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);
//    ierr = VecCopy(src, out); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);

//    my_p4est_level_set_t ls(ngbd);
//    ls.extend_from_interface_to_whole_domain_TVD(phi, normal_velocity_tmp, normal_velocity_np1);

    /* solve for concentration and temperature */
    solve_concentration();
    compute_normal_velocity();
    solve_temperature();
    solve_concentration_sec();

    /* calculate bc error */
    double *integrand_p, *t_interface_p, *c_interface_p, *c_sec_interface_p, *kappa_p, *theta_p, *bc_error_p;
    ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);
    ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(bc_error, &bc_error_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);


    error_bc = 0;
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double error_tmp = (t_interface_p[n] - Tm - ml*c_interface_p[n] - ml_sec*c_sec_interface_p[n]
                          + epsilon_c*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n]))*kappa_p[n]
                          + epsilon_v*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n]))*normal_velocity_np1_p[n]);

      if(fabs(phi_p[n]) < dxyz_close_interface)
      {
        error_bc = MAX(error_bc, fabs(error_tmp));
      }

      bc_error_p[n] = error_tmp;
      integrand_p[n] = SQR(error_tmp);
    }

//    std::cout << "here?\n";

    ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(bc_error, &bc_error_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    double error_bc_l2 = sqrt(integrate_over_interface(p4est, nodes, phi, integrand));

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_bc, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, error on bc = %e, erron on bc (L2) = %e, vgammamax = %e\n", iteration, error_bc, error_bc_l2, vgamma_max); CHKERRXX(ierr);

    /* adjust interface concentration */

    solve_temperature_multiplier();
    solve_concentration_sec_multiplier();
    solve_concentration_multiplier();

    if      (iteration <  4) adjust_interface_concentration(false);
//    else if (iteration == 4) adjust_interface_concentration(true);

    iteration++;
  }

//  solve_concentration();
//  compute_normal_velocity();
//  solve_temperature();
//  solve_concentration_sec();

  compute_velocity();

//  for (short i = 0; i < 3; i++)
//    for(int dir=0; dir<P4EST_DIM; ++dir)
//    {
//      smooth_velocity(v_interface_np1[dir]);
//    }

  compute_dt();

  ierr = VecDestroy(normal_velocity_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(integrand); CHKERRXX(ierr);

  first_step = false;
}

void my_p4est_multialloy_t::smooth_velocity(Vec input)
{
  Vec output;
  ierr = VecDuplicate(phi, &output); CHKERRXX(ierr);

  ierr = VecSet(output, 0.0); CHKERRXX(ierr);

  double *input_ptr, *output_ptr;
  ierr = VecGetArray(input, &input_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(output, &output_ptr); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  /* loop through ghosts */
  for (p4est_locidx_t ghost_idx = 0; ghost_idx < ghost->ghosts.elem_count; ++ghost_idx)
  {
    // get a ghost quadrant
    p4est_quadrant_t* quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);

    // do only finest quadrant that are near interface
    if (quad->level == data->max_lvl)
    {
      p4est_locidx_t offset = (p4est->local_num_quadrants + ghost_idx)*P4EST_CHILDREN;

      // calculate average in the cell
      double average = 0;
      for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
      {
        p4est_locidx_t node_idx = q2n[offset + child_idx];
        average += input_ptr[node_idx];
      }

      average /= (double) P4EST_CHILDREN;
      average /= (double) P4EST_CHILDREN;

      /* loop through nodes of a quadrant and put weights on those nodes, which are local */
      for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
      {
        p4est_locidx_t node_idx = q2n[offset + child_idx];
        if (node_idx < nodes->num_owned_indeps)
          output_ptr[node_idx] += average;
      }
    }
  }


//  std::cout << "here?\n";
  /* loop through local quadrants */

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);

      // do only finest quadrant that are near interface
      if (quad->level == data->max_lvl)
      {
        p4est_locidx_t quad_idx_forest = quad_idx + tree->quadrants_offset;
        p4est_locidx_t offset = quad_idx_forest*P4EST_CHILDREN;

        // calculate average in the cell
        double average = 0;
        for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
        {
          p4est_locidx_t node_idx = q2n[offset + child_idx];
          average += input_ptr[node_idx];
        }

        average /= (double) P4EST_CHILDREN;
        average /= (double) P4EST_CHILDREN;

        /* loop through nodes of a quadrant and put weights on those nodes, which are local */
        for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
        {
          p4est_locidx_t node_idx = q2n[offset + child_idx];
          if (node_idx < nodes->num_owned_indeps)
            output_ptr[node_idx] += average;
        }
      }
    }
  }

  ierr = VecRestoreArray(input, &input_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(output, &output_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(output, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(output, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, output, input);

  ierr = VecDestroy(output); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::adjust_velocity()
{
  if (use_quadratic_form)
  {
    Vec integrand; double *integrand_p;
    ierr = VecDuplicate(phi, &integrand); CHKERRXX(ierr);

    double *t_interface_p, *c_interface_p, *kappa_p, *theta_p, *normal_velocity_np1_p;
    double *temperature_multiplier_p, *concentration_multiplier_p;

    ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

    ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double denom = 2.0*epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))
          * (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n])
          - (thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n]
             + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]);

      integrand_p[n] = denom*denom;
    }

    ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);

    double predicted_change = integrate_over_interface(p4est, nodes, phi, integrand);

    ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      integrand_p[n] = SQR(t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
                           + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
                           + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]);
    }

    ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);

    double functional = integrate_over_interface(p4est, nodes, phi, integrand);

    double factor = functional/fabs(predicted_change);

    ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double denom = 2.0*epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))
          * (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
             + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
             + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n])
          - (thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n]
             + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]);

      double change = 0;

      if (fabs(denom) > 1.e-12)
        change = -SQR(t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
                  + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
                  + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n])/denom;

//      normal_velocity_np1_p[n] += change;
      normal_velocity_np1_p[n] -= factor*denom;
    }

    ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

    ierr = VecDestroy(integrand); CHKERRXX(ierr);

  } else {

    double *t_interface_p, *c_interface_p, *kappa_p, *theta_p, *normal_velocity_np1_p;
    ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    double *temperature_multiplier_p, *concentration_multiplier_p;
    ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double denom = (thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n]
                      + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]
                      - epsilon_v*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n])));

      double change = 0;

//      if (fabs(denom) > 1.e-12)
        change = (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
                  + epsilon_c*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n]))*kappa_p[n]
                  + epsilon_v*(1.-15.*epsilon_anisotropy*cos(4.*theta_p[n]))*normal_velocity_np1_p[n])/denom;

      normal_velocity_np1_p[n] += change;
    }

    ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);
  }

//  // calculate functional
//  double *integrand_p, *t_interface_p, *c_interface_p;
//  ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);
//  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

//  const double *normal_p[P4EST_DIM];
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  double *kappa_p;
//  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    double theta = atan2(normal_p[1][n], normal_p[0][n]);
//    integrand_p[n] = (t_interface_p[n] - Tm  - ml*c_interface_p[n]
//                      +1.*epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta))*kappa_p[n]);
////      integrand_p[n] = SQR(t_interface_p[n] - Tm  - ml*c_interface_p[n]);
//  }

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

//  double functional_tmp = functional;
//  functional = dt_n*(integrate_over_interface(p4est, nodes, phi, integrand));

//  solve_temperature_multiplier();
//  solve_concentration_multiplier();

//  // calculate gradient
//  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

//  double *temperature_multiplier_p, *concentration_multiplier_p;
//  ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
//  ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

//  ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    integrand_p[n] = -SQR(dt_n*(thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n] + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]));
//  }

//  ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

//  double predicted_change = integrate_over_interface(p4est, nodes, phi, integrand);

//  factor = 0.9*functional/fabs(predicted_change);

//  // evolve velocity
//  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
//  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);

//  ierr = VecGetArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
//  ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

//  ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    double theta = atan2(normal_p[1][n], normal_p[0][n]);
////      normal_velocity_np1_p[n] += factor*dt_n*(thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n] + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]);

//    double denom = (thermal_diffusivity*latent_heat/thermal_conductivity*temperature_multiplier_p[n] + (1.-kp_sec)*c_interface_p[n]*concentration_multiplier_p[n]);
//    double change = 0;
//    if (fabs(denom) > 1.e-8)
//      change = (t_interface_p[n] - Tm  - ml*c_interface_p[n]
//                +1.*epsilon_c*(1.-15.*epsilon_anisotropy*cos(4.*theta))*kappa_p[n])/denom;
//    normal_velocity_np1_p[n] += change;
////      if (normal_velocity_np1_p[n] < 0.) normal_velocity_np1_p[n] = 0.;
//  }

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(integrand, &integrand_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(temperature_multiplier, &temperature_multiplier_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);

//  predicted_change = -functional;
////    predicted_change *= factor;


//  ierr = PetscPrintf(p4est->mpicomm, "Actual change = %e, predicted change = %e\n", ((functional-functional_tmp)/dt_n), (predicted_change/dt_n)); CHKERRXX(ierr);

////    compute_dt();

//  ierr = VecGetArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecGetArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);


//  double *bc_error_p;
//  ierr = VecGetArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//  error = 0;
//  error_bc = 0;
////    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    if(fabs(phi_p[n]) < dxyz_close_interface)
//    {

//      error = MAX(error,fabs(normal_velocity_tmp_p[n]-normal_velocity_np1_p[n]));
//      error_bc = MAX(error_bc, fabs(t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
//                                    + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
//                                    + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]));

//      bc_error_p[n] = (t_interface_p[n] - Tm - ml_sec*c_interface_p[n]
//                       + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]
//                       + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]);
//    } else {
//      bc_error_p[n] = 0.;
//    }
//  }

//  ierr = VecRestoreArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(c_sec_interface, &c_interface_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_bc, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
////    error /= MAX(vgamma_max,1e-8);

//  matrices_are_constructed = true;
////    ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, max_velo = %e, error = %e\n", iteration, vgamma_max, error); CHKERRXX(ierr);
//  ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, error on bc = %e, erron on bc (L2) = %e, max change in velocity = %e\n", iteration, error_bc, (functional/dt_n), error); CHKERRXX(ierr);
//  iteration++;
////    error = fabs(functional-functional_tmp)/dt_n;
}




void my_p4est_multialloy_t::adjust_interface_concentration(bool use_simple)
{
  /* calculate denominator */
  Vec v_gamma;
  ierr = VecDuplicate(phi, &v_gamma); CHKERRXX(ierr);
  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
  double *concentration_multiplier_p, *concentration_multiplier_interface_p, *normal_velocity_np1_p;
  ierr = VecGetArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);
  ierr = VecGetArray(concentration_multiplier_interface, &concentration_multiplier_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  if(solve_concentration_solid)
  {
  }
  else
  {
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(concentration_multiplier_p)*normal_p[0][n] + qnnn.dy_central(concentration_multiplier_p)*normal_p[1][n] + qnnn.dz_central(concentration_multiplier_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(concentration_multiplier_p)*normal_p[0][n] + qnnn.dy_central(concentration_multiplier_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = 1.0 - dcl_dn*solute_diffusivity_l - (1.-kp)*normal_velocity_np1_p[n]*concentration_multiplier_p[n];
    }
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(concentration_multiplier_p)*normal_p[0][n] + qnnn.dy_central(concentration_multiplier_p)*normal_p[1][n] + qnnn.dz_central(concentration_multiplier_p)*normal_p[2][n];
#else
      double dcl_dn = qnnn.dx_central(concentration_multiplier_p)*normal_p[0][n] + qnnn.dy_central(concentration_multiplier_p)*normal_p[1][n];
#endif
      v_gamma_p[n] = 1.0 - dcl_dn*solute_diffusivity_l - (1.-kp)*normal_velocity_np1_p[n]*concentration_multiplier_p[n];
    }
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(concentration_multiplier, &concentration_multiplier_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(concentration_multiplier_interface, &concentration_multiplier_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  Vec denominator;
  ierr = VecDuplicate(phi, &denominator); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma, denominator);

  ierr = VecDestroy(v_gamma); CHKERRXX(ierr);

  /* modify interface concentration */
  double *c_interface_p, *c_sec_interface_p, *t_interface_p, *kappa_p, *theta_p, *denominator_p;
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecGetArray(denominator, &denominator_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if (use_simple || fabs(denominator_p[n]) < 1.e-8)
    {
      c_interface_p[n] -= (c_interface_p[n] + ml_sec/ml * c_sec_interface_p[n] - (t_interface_p[n] - Tm)/ml
                           - epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]/ml
                           - epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]/ml)
                  / 1.0;
    } else {

    c_interface_p[n] -= (c_interface_p[n] + ml_sec/ml * c_sec_interface_p[n] - (t_interface_p[n] - Tm)/ml
                         - epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]/ml
                         - epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]/ml)
        / denominator_p[n];
    }
//        / 1.0;
//    c_interface_p[n] -= (c_interface_p[n] + ml_sec/ml * c_sec_interface_p[n] - (t_interface_p[n] - Tm)/ml
//                         - epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*kappa_p[n]/ml
//                         - epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta_p[n]))*normal_velocity_np1_p[n]/ml)
//        / 1.0;
  }
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_sec_interface, &c_sec_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(denominator, &denominator_p); CHKERRXX(ierr);

  ierr = VecDestroy(denominator); CHKERRXX(ierr);

//  /* compute maximum normal velocity for convergence of v_gamma scaling */
//  vgamma_max = 0;
//  const double *phi_p, *normal_velocity_np1_p;
//  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  {
//    if(fabs(phi_p[n]) < dxyz_close_interface)
//      vgamma_max = MAX(vgamma_max, fabs(normal_velocity_np1_p[n]));
//  }
//  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
//  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &vgamma_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}





void my_p4est_multialloy_t::compare_velocity_temperature_vs_concentration()
{
  /* compute the normal velocity from the temperature field and compare with the one computed from the concentration */
  Vec Tl, Ts;
  ierr = VecDuplicate(phi, &Tl); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &Ts); CHKERRXX(ierr);

  /* make copy of the temperature field */
  Vec Tnp1_loc, Tl_loc, Ts_loc;
  ierr = VecGhostGetLocalForm(Tl, &Tl_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(Ts, &Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_np1, &Tnp1_loc); CHKERRXX(ierr);
  ierr = VecCopy(Tnp1_loc, Tl_loc); CHKERRXX(ierr);
  ierr = VecCopy(Tnp1_loc, Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(Tl, &Tl_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(Ts, &Ts_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_np1, &Tnp1_loc); CHKERRXX(ierr);

  /* extend temperatures over interface */
  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, Ts);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi, Tl);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  /* compute normal velocity, vn = k/L [ dT/dn ]
   * k is the thermal conductivity
   * L is the latent heat
   */
  quad_neighbor_nodes_of_node_t qnnn;
  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
  double *Tl_p, *Ts_p;
  double *vn_p;
  ierr = VecGetArray(Tl, &Tl_p); CHKERRXX(ierr);
  ierr = VecGetArray(Ts, &Ts_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &vn_p); CHKERRXX(ierr);
  double err = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(fabs(phi_p[n])<dxyz_min)
    {
      ngbd->get_neighbors(n, qnnn);
      double dTl_dn = qnnn.dx_central(Tl_p)*normal_p[0][n] + qnnn.dy_central(Tl_p)*normal_p[1][n];
      double dTs_dn = qnnn.dx_central(Ts_p)*normal_p[0][n] + qnnn.dy_central(Ts_p)*normal_p[1][n];
      double vn = thermal_conductivity/latent_heat * (dTs_dn - dTl_dn);
//      err = MAX(err, fabs((vn-vn_p[n])/vn));
      err = MAX(err, fabs((vn-vn_p[n])));
    }
  }

  ierr = VecRestoreArray(normal_velocity_np1, &vn_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Tl, &Tl_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Ts, &Ts_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  ierr = PetscPrintf(p4est->mpicomm, "maximum difference between velocities = %e\n", err); CHKERRXX(ierr);

  ierr = VecDestroy(Tl); CHKERRXX(ierr);
  ierr = VecDestroy(Ts); CHKERRXX(ierr);
}



void my_p4est_multialloy_t::save_VTK(int iter)
{
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }
//  std::ostringstream command;
//  command << "mkdir -p " << out_dir << "/vtu";
//  int ret_sys = system(command.str().c_str());
//  if(ret_sys<0)
//    throw std::invalid_argument("my_p4est_multialloy_t::save_vtk could not create directory");
  
  char name[1000];
#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/bialloy_%d_%dx%dx%d.%05d", out_dir, p4est->mpisize, brick->nxyztrees[0], brick->nxyztrees[1], brick->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/bialloy_%d_%dx%d.%05d", out_dir, p4est->mpisize, brick->nxyztrees[0], brick->nxyztrees[1], iter);
#endif

  /* if the domain is periodic, create a temporary tree without periodicity for visualization */
  bool periodic = false;
  for(int dir=0; dir<P4EST_DIM; ++dir)
    periodic = periodic || (p4est->connectivity->tree_to_tree[P4EST_FACES*0 + 2*dir]!=0);

  my_p4est_brick_t brick_vis;
  p4est_connectivity_t *connectivity_vis = NULL;
  p4est_t *p4est_vis;
  p4est_ghost_t *ghost_vis;
  p4est_nodes_t *nodes_vis;
  Vec phi_vis;
  Vec temperature_vis;
  Vec cl_vis;
  Vec cl_sec_vis;
  Vec normal_velocity_vis;
  Vec kappa_vis;

  double *phi_vis_p;

  periodic = false; // no need to create a temporary tree anymore

  // TODO: get rid of creating temporary grid altogether

  if(periodic)
  {
    bool is_grid_changing = true;
    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
    my_p4est_interpolation_nodes_t interp(ngbd);

    double *v2c = p4est->connectivity->vertices;
    p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
    p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
    p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

    double xyz_min[P4EST_DIM];
    double xyz_max[P4EST_DIM];
    for (short i=0; i<P4EST_DIM; i++)
      xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
    for (short i=0; i<P4EST_DIM; i++)
      xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];

#ifdef P4_TO_P8
    int non_periodic[] = {0, 0, 0};
#else
    int non_periodic[] = {0, 0};
#endif
    connectivity_vis = my_p4est_brick_new(brick->nxyztrees, xyz_min, xyz_max, &brick_vis, non_periodic);

    p4est_vis = my_p4est_new(p4est->mpicomm, connectivity_vis, 0, NULL, NULL);
    ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
    nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
    ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);

    for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
      interp.add_point(n, xyz);
    }
    interp.set_input(phi, linear);
    interp.interpolate(phi_vis);

    while(is_grid_changing)
    {
      ierr = VecGetArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);
      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
      is_grid_changing = sp.refine_and_coarsen(p4est_vis, nodes_vis, phi_vis_p);
      ierr = VecRestoreArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);

      if(is_grid_changing)
      {
        my_p4est_partition(p4est_vis, P4EST_TRUE, NULL);
        p4est_ghost_destroy(ghost_vis); ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_vis); nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
        ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);

        interp.clear();
        for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
          interp.add_point(n, xyz);
        }
        interp.set_input(phi, linear);
        interp.interpolate(phi_vis);
      }
    }

    ierr = VecDuplicate(phi_vis, &temperature_vis); CHKERRXX(ierr);
    interp.set_input(temperature_n, linear);
    interp.interpolate(temperature_vis);

    ierr = VecDuplicate(phi_vis, &cl_vis); CHKERRXX(ierr);
    interp.set_input(cl_n, linear);
    interp.interpolate(cl_vis);

    ierr = VecDuplicate(phi_vis, &cl_sec_vis); CHKERRXX(ierr);
    interp.set_input(cl_sec_n, linear);
    interp.interpolate(cl_sec_vis);

    ierr = VecDuplicate(phi_vis, &normal_velocity_vis); CHKERRXX(ierr);
    interp.set_input(normal_velocity_np1, linear);
    interp.interpolate(normal_velocity_vis);

    ierr = VecDuplicate(phi_vis, &kappa_vis); CHKERRXX(ierr);
    interp.set_input(kappa, linear);
    interp.interpolate(kappa_vis);
  }
  else
  {
    p4est_vis = p4est;
    ghost_vis = ghost;
    nodes_vis = nodes;
    phi_vis = phi;
    temperature_vis = temperature_np1;
    cl_vis = cl_np1;
    cl_sec_vis = cl_sec_np1;
    normal_velocity_vis = normal_velocity_np1;
    kappa_vis = kappa;
  }

  const double *phi_p, *temperature_p, *cl_p, *cl_sec_p;
  double *normal_velocity_np1_p;
  ierr = VecGetArrayRead(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_vis, &temperature_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cl_vis, &cl_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cl_sec_vis, &cl_sec_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_vis, &normal_velocity_np1_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est_vis, ghost_vis, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_vis->first_local_tree; tree_idx <= p4est_vis->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_vis->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost_vis->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost_vis->ghosts, q);
    l_p[p4est_vis->local_num_quadrants+q] = quad->level;
  }

  for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] /= scaling;

  const double *kappa_p;
  ierr = VecGetArrayRead(kappa_vis, &kappa_p); CHKERRXX(ierr);

  double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  double *velo_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(v_interface_np1[dir], &velo_p[dir]); CHKERRXX(ierr);
  }

  double *bc_error_p;
  ierr = VecGetArray(bc_error, &bc_error_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(  p4est_vis, nodes_vis, ghost_vis,
                           P4EST_TRUE, P4EST_TRUE,
                         #ifdef P4_TO_P8
                           7, 1, name,
                         #else
                           7, 1, name,
                         #endif
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "temperature", temperature_p,
                           VTK_POINT_DATA, "concentration", cl_p,
                           VTK_POINT_DATA, "concentration_sec", cl_sec_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_POINT_DATA, "bc_error", bc_error_p,
//                           VTK_POINT_DATA, "normal_x", normal_p[0],
//                           VTK_POINT_DATA, "normal_y", normal_p[1],
//    #ifdef P4_TO_P8
//                           VTK_POINT_DATA, "normal_z", normal_p[2],
//    #endif
//      VTK_POINT_DATA, "velo_x", velo_p[0],
//      VTK_POINT_DATA, "velo_y", velo_p[1],
//    #ifdef P4_TO_P8
//      VTK_POINT_DATA, "velo_z", velo_p[2],
//    #endif
                           VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(bc_error, &bc_error_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_interface_np1[dir], &velo_p[dir]); CHKERRXX(ierr);
  }

  if(!periodic)
  {
    for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
      normal_velocity_np1_p[n] *= scaling;
  }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_vis, &temperature_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_vis, &cl_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_sec_vis, &cl_sec_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_vis, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_vis, &normal_velocity_np1_p); CHKERRXX(ierr);

  if(periodic)
  {
    ierr = VecDestroy(normal_velocity_vis); CHKERRXX(ierr);
    ierr = VecDestroy(cl_vis); CHKERRXX(ierr);
    ierr = VecDestroy(cl_sec_vis); CHKERRXX(ierr);
    ierr = VecDestroy(temperature_vis); CHKERRXX(ierr);
    ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
    ierr = VecDestroy(kappa_vis); CHKERRXX(ierr);
    p4est_nodes_destroy(nodes_vis);
    p4est_ghost_destroy(ghost_vis);
    p4est_destroy(p4est_vis);
    my_p4est_brick_destroy(connectivity_vis, &brick_vis);
  }

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", name);
}
