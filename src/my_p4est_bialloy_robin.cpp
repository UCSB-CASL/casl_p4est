#include "my_p4est_bialloy_robin.h"

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_poisson_nodes_voronoi.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_voronoi.h>
#endif




my_p4est_bialloy_t::my_p4est_bialloy_t(my_p4est_node_neighbors_t *ngbd)
  : brick(ngbd->myb), connectivity(ngbd->p4est->connectivity), p4est(ngbd->p4est), ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy), ngbd(ngbd),
    temperature_l_n(NULL), temperature_l_np1(NULL),
    temperature_s_n(NULL), temperature_s_np1(NULL),
    cl_n(NULL), cl_np1(NULL),
    normal_velocity_np1(NULL),
    phi(NULL),
    kappa(NULL), rhs(NULL)
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
  cooling_velocity     = 0.01;      /* cm.s-1      */
  kp                   = 0.86;
  c0                   = 0.40831;   /* at frac.    */
  ml                   = -357;      /* K / at frac.*/
  epsilon_anisotropy   = 0.05;
  epsilon_c            = 2.7207e-5; /* cm.K     */
  epsilon_v            = 2.27e-2;   /* s.K.cm-1 */

  ::dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  dxyz_min = MIN(dxyz[0],dxyz[1],dxyz[2]);
  dxyz_max = MAX(dxyz[0],dxyz[1],dxyz[2]);
#else
  dxyz_min = MIN(dxyz[0],dxyz[1]);
  dxyz_max = MAX(dxyz[0],dxyz[1]);
#endif
  dxyz_close_interface = 4*dxyz_min;


  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_interface_n  [dir] = NULL;
    v_interface_np1[dir] = NULL;
    normal[dir] = NULL;
  }
}




my_p4est_bialloy_t::~my_p4est_bialloy_t()
{
  if(temperature_l_n  !=NULL) { ierr = VecDestroy(temperature_l_n);   CHKERRXX(ierr); }
  if(temperature_l_np1!=NULL) { ierr = VecDestroy(temperature_l_np1); CHKERRXX(ierr); }
  if(temperature_s_n  !=NULL) { ierr = VecDestroy(temperature_s_n);   CHKERRXX(ierr); }
  if(temperature_s_np1!=NULL) { ierr = VecDestroy(temperature_s_np1); CHKERRXX(ierr); }

  if(cl_n       !=NULL) { ierr = VecDestroy(cl_n);        CHKERRXX(ierr); }
  if(cl_np1     !=NULL) { ierr = VecDestroy(cl_np1);      CHKERRXX(ierr); }
  if(cl_gamma   !=NULL) { ierr = VecDestroy(cl_gamma);    CHKERRXX(ierr); }

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

  /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, brick);
}




void my_p4est_bialloy_t::set_parameters( double latent_heat,
                                         double thermal_conductivity,
                                         double thermal_diffusivity,
                                         double solute_diffusivity_l,
                                         double cooling_velocity,
                                         double kp,
                                         double c0,
                                         double ml,
                                         double Tm,
                                         double epsilon_anisotropy,
                                         double epsilon_c,
                                         double epsilon_v,
                                         double scaling)
{
  this->latent_heat          = latent_heat;
  this->thermal_conductivity = thermal_conductivity;
  this->thermal_diffusivity  = thermal_diffusivity;
  this->solute_diffusivity_l = solute_diffusivity_l;
  this->cooling_velocity     = cooling_velocity;
  this->kp                   = kp;
  this->c0                   = c0;
  this->ml                   = ml;
  this->Tm                   = Tm;
  this->epsilon_anisotropy   = epsilon_anisotropy;
  this->epsilon_c            = epsilon_c;
  this->epsilon_v            = epsilon_v;
  this->scaling              = scaling;
}





void my_p4est_bialloy_t::set_phi(Vec phi)
{
  this->phi = phi;
  compute_normal_and_curvature();
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
}




#ifdef P4_TO_P8
void my_p4est_bialloy_t::set_bc(WallBC3D& bc_wall_type_t,
                                WallBC3D& bc_wall_type_c,
                                CF_3& bc_wall_value_t,
                                CF_3& bc_wall_value_cl)
#else
void my_p4est_bialloy_t::set_bc(WallBC2D& bc_wall_type_t,
                                WallBC2D& bc_wall_type_c,
                                CF_2& bc_wall_value_t,
                                CF_2& bc_wall_value_cl)
#endif
{
  bc_t.setWallTypes(bc_wall_type_t);
  bc_t.setWallValues(bc_wall_value_t);
  bc_t.setInterfaceType(DIRICHLET);

  bc_cl.setWallTypes(bc_wall_type_c);
  bc_cl.setWallValues(bc_wall_value_cl);
  bc_cl.setInterfaceType(ROBIN);
  bc_cl.setInterfaceValue(zero);
}





void my_p4est_bialloy_t::set_temperature(Vec temperature_l, Vec temperature_s)
{
  temperature_l_n = temperature_l;
  temperature_s_n = temperature_s;

  Vec src, out;

  ierr = VecDuplicate(temperature_s_n, &temperature_s_np1); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_s_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_s_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_s_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_s_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(temperature_l_n, &temperature_l_np1); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_l_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(temperature_l_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_l_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_l_np1, &out); CHKERRXX(ierr);
}





void my_p4est_bialloy_t::set_concentration(Vec cl)
{
  cl_n = cl;

  Vec src, out;

  ierr = VecDuplicate(cl_n, &cl_np1); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cl_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_np1, &out); CHKERRXX(ierr);

  ierr = VecDuplicate(cl_n, &cl_gamma); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cl_n    , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cl_gamma, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n    , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_gamma, &out); CHKERRXX(ierr);
}





void my_p4est_bialloy_t::set_normal_velocity(Vec v)
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





void my_p4est_bialloy_t::set_dt( double dt )
{
  dt_n   = dt;
}





void my_p4est_bialloy_t::compute_normal_and_curvature()
{
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

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);
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
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);
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

  Vec kappa_tmp;
  ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);
  double *kappa_p;
  ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);
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
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);
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
}




/*
 * Compute interface velocity as V_gamma = thermal_conductivity/L [ dT/dn ]
 */
void my_p4est_bialloy_t::compute_normal_velocity_from_temperature()
{
  Vec v_gamma;
  ierr = VecDuplicate(normal_velocity_np1, &v_gamma); CHKERRXX(ierr);

//  double vgamma_min = -1;
//  double lambda = 0.01;
  double lambda = 1.0;

  my_p4est_level_set_t ls(ngbd);
//  while (vgamma_min <= 0.0)
//  {
//    lambda /= 2.0;

  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  const double *temperature_l_np1_p, *temperature_s_np1_p;
  ierr = VecGetArrayRead(temperature_l_np1, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_s_np1, &temperature_s_np1_p); CHKERRXX(ierr);

  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  double *normal_velocity_p;
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
    double dtl_dn = qnnn.dx_central(temperature_l_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_l_np1_p)*normal_p[1][n] + qnnn.dz_central(temperature_l_np1_p)*normal_p[2][n];
    double dts_dn = qnnn.dx_central(temperature_s_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_s_np1_p)*normal_p[1][n] + qnnn.dz_central(temperature_s_np1_p)*normal_p[2][n];
#else
    double dtl_dn = qnnn.dx_central(temperature_l_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_l_np1_p)*normal_p[1][n];
    double dts_dn = qnnn.dx_central(temperature_s_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_s_np1_p)*normal_p[1][n];
#endif

    v_gamma_p[n] = thermal_conductivity/latent_heat * (dts_dn - dtl_dn);

    v_gamma_p[n] = (1.0-lambda)*normal_velocity_p[n] + lambda*v_gamma_p[n];
  }

  ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  double *phii_p;
  VecGetArray(phi, &phii_p);
  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
    double dtl_dn = qnnn.dx_central(temperature_l_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_l_np1_p)*normal_p[1][n] + qnnn.dz_central(temperature_l_np1_p)*normal_p[2][n];
    double dts_dn = qnnn.dx_central(temperature_s_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_s_np1_p)*normal_p[1][n] + qnnn.dz_central(temperature_s_np1_p)*normal_p[2][n];
#else
    double dtl_dn = qnnn.dx_central(temperature_l_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_l_np1_p)*normal_p[1][n];
    double dts_dn = qnnn.dx_central(temperature_s_np1_p)*normal_p[0][n] + qnnn.dy_central(temperature_s_np1_p)*normal_p[1][n];
#endif

    v_gamma_p[n] = thermal_conductivity/latent_heat * (dts_dn - dtl_dn);

    v_gamma_p[n] = (1.0-lambda)*normal_velocity_p[n] + lambda*v_gamma_p[n];

  }
  VecRestoreArray(phi, &phii_p);

  ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_l_np1, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_s_np1, &temperature_s_np1_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

//  Vec v_gamma_tmp;
//  ierr = VecDuplicate(normal_velocity_np1, &v_gamma_tmp); CHKERRXX(ierr);
//  ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma, v_gamma_tmp);
//  ierr = VecMin(v_gamma_tmp, NULL, &vgamma_min); CHKERRXX(ierr);
//  ierr = VecDestroy(v_gamma_tmp); CHKERRXX(ierr);
//  }

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
//  PetscPrintf(p4est->mpicomm, "THE VELOCITY FROM TEMPERATURE IS %g\n", vgamma_max);
}



void my_p4est_bialloy_t::compute_normal_velocity_from_concentration()
{
  Vec v_gamma;
  ierr = VecDuplicate(normal_velocity_np1, &v_gamma); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);

  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  const double *cl_np1_p;
  ierr = VecGetArrayRead(cl_np1, &cl_np1_p); CHKERRXX(ierr);

  const double *cl_gamma_p;
  ierr = VecGetArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);

  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    /* cafeful ! minus sign because cl is defined for phi>0 */
#ifdef P4_TO_P8
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n] + qnnn.dz_central(cl_np1_p)*normal_p[2][n]);
#else
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n]);
#endif

    v_gamma_p[n] = solute_diffusivity_l/((1-kp)*MAX(cl_gamma_p[n], 1e-7)) * dcl_dn;
  }

  ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    /* cafeful ! minus sign because cl is defined for phi>0 */
#ifdef P4_TO_P8
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n] + qnnn.dz_central(cl_np1_p)*normal_p[2][n]);
#else
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n]);
#endif

    v_gamma_p[n] = solute_diffusivity_l/((1-kp)*MAX(cl_gamma_p[n], 1e-7)) * dcl_dn;
  }

  ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_np1 , &cl_np1_p ); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

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
//  PetscPrintf(p4est->mpicomm, "THE VELOCITY FROM CONCENTRATION IS %g\n", vgamma_max);
}





void my_p4est_bialloy_t::compute_velocity_from_temperature()
{
  Vec v_gamma[P4EST_DIM];
  double *v_gamma_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(normal[dir], &v_gamma[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
  }

  const double *temperature_l_np1_p, *temperature_s_np1_p;
  ierr = VecGetArrayRead(temperature_l_np1, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_s_np1, &temperature_s_np1_p); CHKERRXX(ierr);

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

  ierr = VecRestoreArrayRead(temperature_l_np1, &temperature_l_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_s_np1, &temperature_s_np1_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma[dir], v_interface_np1[dir]);
    ierr = VecDestroy(v_gamma[dir]); CHKERRXX(ierr);
  }
}


void my_p4est_bialloy_t::compute_velocity_from_concentration()
{
  Vec v_gamma[P4EST_DIM];
  double *v_gamma_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(normal[dir], &v_gamma[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);

  const double *cl_gamma_p;
  ierr = VecGetArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);

  const double *cl_np1_p;
  ierr = VecGetArrayRead(cl_np1, &cl_np1_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    v_gamma_p[0][n] = -solute_diffusivity_l/(1-kp) / MAX(cl_gamma_p[n], 1e-7) * qnnn.dx_central(cl_np1_p);
    v_gamma_p[1][n] = -solute_diffusivity_l/(1-kp) / MAX(cl_gamma_p[n], 1e-7) * qnnn.dy_central(cl_np1_p);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = -solute_diffusivity_l/(1-kp) / MAX(c_gamma_p[n], 1e-7) * qnnn.dz_central(cl_np1_p);
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

    v_gamma_p[0][n] = -solute_diffusivity_l/(1-kp) / MAX(cl_gamma_p[n], 1e-7) * qnnn.dx_central(cl_np1_p);
    v_gamma_p[1][n] = -solute_diffusivity_l/(1-kp) / MAX(cl_gamma_p[n], 1e-7) * qnnn.dy_central(cl_np1_p);
#ifdef P4_TO_P8
    v_gamma_p[2][n] = -solute_diffusivity_l/(1-kp) / MAX(c_gamma_p[n], 1e-7) * qnnn.dz_central(cl_np1_p);
#endif
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(v_gamma[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v_gamma[dir], &v_gamma_p[dir]); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma[dir], v_interface_np1[dir]);
    ierr = VecDestroy(v_gamma[dir]); CHKERRXX(ierr);
  }
}




/*
 * Solve dT/dt = thermal_diffusivity * laplace(T)
 * with T_gamma = Tm + ml Cl + eps_c(theta) kappa + eps_v(theta) vn
 * inside liquid and solid phases
 */
void my_p4est_bialloy_t::solve_temperature()
{
  /* initialize the boundary condition for T on the interface */
  Vec temperature_interface_tmp;
  ierr = VecDuplicate(phi, &temperature_interface_tmp); CHKERRXX(ierr);

  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  double *temperature_interface_tmp_p;
  const double *kappa_p, *normal_velocity_np1_p, *cl_gamma_p;
  ierr = VecGetArray(temperature_interface_tmp, &temperature_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);

  const double *temp_p;
  ierr = VecGetArrayRead(temperature_s_np1, &temp_p); CHKERRXX(ierr);
  double lambda = 0.0;

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    double theta_xz = atan2(normal_p[2][n], normal_p[0][n]);
    double theta_yz = atan2(normal_p[2][n], normal_p[1][n]);
    temperature_interface_tmp_p[n] = Tm + ml*cl_np1_p[n]
        + epsilon_c*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*kappa_p[n]
        + epsilon_v*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*normal_velocity_np1_p[n];
#else
    double theta = atan2(normal_p[1][n], normal_p[0][n]);
    temperature_interface_tmp_p[n] = Tm + ml*(cl_gamma_p[n]);
//        + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta))*kappa_p[n]/ml
//        + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta))*normal_velocity_np1_p[n]/ml;
#endif
//    temperature_interface_tmp_p[n] = Tm;
//    temperature_interface_tmp_p[n] = 0;
//    temperature_interface_tmp_p[n] = Tm + ml*c0;
//    temperature_interface_tmp_p[n] = lambda*temp_p[n]
//        + (1.0-lambda)*temperature_interface_tmp_p[n];
  }
  ierr = VecRestoreArrayRead(temperature_s_np1, &temp_p); CHKERRXX(ierr);

//  double mm;
//  VecMax(temperature_interface_tmp, NULL, &mm);
//  std::cout << "Maximum is : " << mm << std::endl;

  ierr = VecRestoreArray(temperature_interface_tmp, &temperature_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_gamma, &cl_gamma_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  Vec temperature_interface;
  ierr = VecDuplicate(phi, &temperature_interface); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, temperature_interface_tmp, temperature_interface);
  ierr = VecDestroy(temperature_interface_tmp); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interface_value_temperature(ngbd);
  interface_value_temperature.set_input(temperature_interface, linear);
  bc_t.setInterfaceValue(interface_value_temperature);

  /* solve temperature in solid phase */
  Vec src, out;
  ierr = VecGhostGetLocalForm(temperature_s_n, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(rhs            , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_s_n, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rhs            , &out); CHKERRXX(ierr);

  my_p4est_poisson_nodes_t solver_t(ngbd);
  solver_t.set_phi(phi);
  solver_t.set_bc(bc_t);
  solver_t.set_mu(dt_n*thermal_diffusivity);
  solver_t.set_diagonal(1);
  solver_t.set_rhs(rhs);
  solver_t.solve(temperature_s_np1);

  ls.extend_Over_Interface_TVD(phi, temperature_s_np1);

  /* solve temperature in liquid phase */
  ierr = VecGhostGetLocalForm(temperature_l_n, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(rhs            , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_l_n, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rhs            , &out); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  solver_t.set_phi(phi);
  solver_t.set_bc(bc_t);
  solver_t.set_mu(dt_n*thermal_diffusivity);
  solver_t.set_diagonal(1);
  solver_t.set_rhs(rhs);
  solver_t.solve(temperature_l_np1);

  ls.extend_Over_Interface_TVD(phi, temperature_l_np1);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecDestroy(temperature_interface); CHKERRXX(ierr);
}




/*
 * Solve dC/dt = solute_diffusivity * laplace(C)
 * with D*grad(C).n + vn*(kp-1)*C = 0
 * inside liquid phase
 */
void my_p4est_bialloy_t::solve_concentration()
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

  my_p4est_level_set_t ls(ngbd);


//  Vec cl_interface_old;
//  Vec bc_value;
//  ierr = VecDuplicate(phi, &cl_interface_old); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi, &bc_value); CHKERRXX(ierr);

//  ls.extend_from_interface_to_whole_domain_TVD(phi, cl_n, cl_interface_old);

//  double *bc_value_p;
//  const double *cl_interface_old_p;
//  ierr = VecGetArray(bc_value, &bc_value_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(cl_interface_old, &cl_interface_old_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
//    robin_coef_p[n] = -cooling_velocity*(1-kp)/solute_diffusivity_l;
    robin_coef_p[n] = -normal_velocity_np1_p[n]*(1-kp)/solute_diffusivity_l;
//    robin_coef_p[n] = -1.0;

//    robin_coef_p[n] = 0.0;
//    bc_value_p[n] = normal_velocity_np1_p[n]*(1-kp)/solute_diffusivity_l * cl_interface_old_p[n];
  }

//  ierr = VecRestoreArray(bc_value, &bc_value_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(cl_interface_old, &cl_interface_old_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  my_p4est_interpolation_nodes_t bc_value_cl(ngbd);
//  bc_value_cl.set_input(bc_value, linear);

//  bc_cl.setInterfaceValue(bc_value_cl);

  Vec src, out;
  ierr = VecGhostGetLocalForm(cl_n, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(rhs , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rhs , &out); CHKERRXX(ierr);


//  my_p4est_poisson_nodes_t solver_c(ngbd);
  my_p4est_poisson_nodes_voronoi_t solver_c(ngbd);
  solver_c.set_phi(phi);
  solver_c.set_bc(bc_cl);
  solver_c.set_mu(dt_n*solute_diffusivity_l);
  solver_c.set_diagonal(1);
  solver_c.set_rhs(rhs);
  solver_c.set_robin_coef(robin_coef);
  solver_c.solve(cl_np1);

  ls.extend_Over_Interface_TVD(phi, cl_np1);

  double err_bc = 0;
  const double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }
  double *cl_np1_p;
  VecGetArray(phi, &phi_p);
  VecGetArray(robin_coef, &robin_coef_p);
  VecGetArray(cl_np1, &cl_np1_p);
  for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
  {
    const quad_neighbor_nodes_of_node_t &qnnn = ngbd->get_neighbors(n);

    /* cafeful ! minus sign because cl is defined for phi>0 */
#ifdef P4_TO_P8
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n] + qnnn.dz_central(cl_np1_p)*normal_p[2][n]);
#else
    double dcl_dn = -(qnnn.dx_central(cl_np1_p)*normal_p[0][n] + qnnn.dy_central(cl_np1_p)*normal_p[1][n]);
#endif

    if(phi_p[n]<0 && phi_p[n]>-dxyz_min)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
      err_bc = MAX(err_bc, fabs(dcl_dn + robin_coef_p[n]*cl_np1_p[n] - bc_cl.interfaceValue(x,y) ));
    }
  }
  VecRestoreArray(cl_np1, &cl_np1_p);
  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(robin_coef, &robin_coef_p);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  ierr = VecDestroy(robin_coef); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_bc, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = PetscPrintf(p4est->mpicomm, "Error on Robin boundary condition for cl_np1 : %e\n", err_bc); CHKERRXX(ierr);


  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi, cl_np1, cl_gamma);


//  ierr = VecDestroy(cl_interface_old); CHKERRXX(ierr);
//  ierr = VecDestroy(bc_value); CHKERRXX(ierr);

}




void my_p4est_bialloy_t::compute_dt()
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
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(v_interface_np1[dir], &v_interface_np1_p); CHKERRXX(ierr);
//    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//    {
//      if(fabs(phi_p[n]) < dxyz_close_interface)
//        u_max = MAX(u_max, fabs(v_interface_np1_p[n]));
//    }
//    ierr = VecRestoreArrayRead(v_interface_np1[dir], &v_interface_np1_p); CHKERRXX(ierr);
//  }
  {
    ierr = VecGetArrayRead(normal_velocity_np1, &v_interface_np1_p); CHKERRXX(ierr);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(fabs(phi_p[n]) < dxyz_close_interface)
        u_max = MAX(u_max, fabs(v_interface_np1_p[n]));
    }
    ierr = VecRestoreArrayRead(normal_velocity_np1, &v_interface_np1_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

//  dt_n = 1. * sqrt(dxyz_min)*dxyz_min * MIN(1./u_max, 1./cooling_velocity);
//  dt_n = 0.01 * sqrt(dxyz_min)*dxyz_min * MIN(1./u_max, 1./cooling_velocity);
  dt_n = 0.8 * dxyz_min * MIN(1/u_max, 1/cooling_velocity);
//  dt_n = .5 * .25 * dxyz_min * MIN(1/u_max, 1/cooling_velocity);
  PetscPrintf(p4est->mpicomm, "VMAX = %e, VGAMMAMAX = %e, COOLING_VELO = %e\n", u_max, vgamma_max, cooling_velocity);

//  if(dt_n>0.5/MAX(1e-7, MAX(u_max,vgamma_max)*kappa_max))
  if(0 && dt_n>0.5/(MAX(u_max,vgamma_max)*MAX(kappa_max,1/(100*dxyz_min))))
  {
//    dt_n = MIN(dt_n, 0.5/MAX(1e-7, MAX(u_max,vgamma_max)*kappa_max));
    dt_n = MIN(dt_n, 0.5/(MAX(u_max,vgamma_max)*MAX(kappa_max,1/(100*dxyz_min))));
    ierr = PetscPrintf(p4est->mpicomm, "KAPPA LIMITING TIME STEP\n"); CHKERRXX(ierr);
  }
//  dt_n = 1e-6;
  PetscPrintf(p4est->mpicomm, "dt = %e\n", dt_n);
}





void my_p4est_bialloy_t::update_grid()
{
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

  /* bousouf update this for second order in time */
  sl.update_p4est(v_interface_np1, dt_n, phi);

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  ierr = VecDestroy(temperature_l_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_l_n); CHKERRXX(ierr);
  interp.set_input(temperature_l_np1, quadratic_non_oscillatory);
  interp.interpolate(temperature_l_n);
  ierr = VecDestroy(temperature_l_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_l_np1); CHKERRXX(ierr);

  ierr = VecDestroy(temperature_s_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_s_n); CHKERRXX(ierr);
  interp.set_input(temperature_s_np1, quadratic_non_oscillatory);
  interp.interpolate(temperature_s_n);
  ierr = VecDestroy(temperature_s_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_s_np1); CHKERRXX(ierr);

  ierr = VecDestroy(cl_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_n); CHKERRXX(ierr);
  interp.set_input(cl_np1, quadratic_non_oscillatory);
  interp.interpolate(cl_n);
  ierr = VecDestroy(cl_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl_np1); CHKERRXX(ierr);

  Vec c_tmp;
  ierr = VecDuplicate(phi, &c_tmp); CHKERRXX(ierr);
  interp.set_input(cl_gamma, quadratic_non_oscillatory);
  interp.interpolate(c_tmp);
  ierr = VecDestroy(cl_gamma); CHKERRXX(ierr);
  cl_gamma = c_tmp;

  Vec src, out;
  ierr = VecGhostGetLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(cl_np1, &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_n  , &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cl_np1, &out); CHKERRXX(ierr);

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

  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  hierarchy->update(p4est, ghost);
  ngbd->update(hierarchy, nodes);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  ls.perturb_level_set_function(phi, EPS);

  compute_normal_and_curvature();
}





void my_p4est_bialloy_t::one_step()
{
  Vec normal_velocity_tmp;
  ierr = VecDuplicate(phi, &normal_velocity_tmp); CHKERRXX(ierr);

  Vec normal_velocity_cl;
  ierr = VecDuplicate(phi, &normal_velocity_cl); CHKERRXX(ierr);

  double *normal_velocity_tmp_p, *phi_p, *normal_velocity_np1_p;
  double error = 1;
  int iteration = 0;
  matrices_are_constructed = false;
//  compute_normal_velocity_from_temperature();

  while(error>1e-5 && iteration<1000)
  {
//    save_VTK(iteraion);
    Vec src, out;
    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);
    ierr = VecCopy(src, out); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(normal_velocity_tmp, &out); CHKERRXX(ierr);

//    compare_normal_velocity_temperature_vs_concentration();
    solve_temperature();
    compute_normal_velocity_from_temperature();
    solve_concentration();
//    compute_dt();


//    double error_cl = 1;
//    int iteration_cl = 0;
//    while(error_cl>1e-5 && iteration_cl<1)
//    {
//      ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//      ierr = VecGhostGetLocalForm(normal_velocity_cl , &out); CHKERRXX(ierr);
//      ierr = VecCopy(src, out); CHKERRXX(ierr);
//      ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
//      ierr = VecGhostRestoreLocalForm(normal_velocity_cl , &out); CHKERRXX(ierr);

//      solve_concentration();
//      compute_normal_velocity_from_concentration();

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
//      error_cl /= MAX(vgamma_max,1e-7);

//      ierr = PetscPrintf(p4est->mpicomm, "\tConvergence Cl iteration #%d, max_velo = %e, error_cl = %e\n", iteration_cl, vgamma_max, error_cl); CHKERRXX(ierr);

//      iteration_cl++;
//    }

//    compute_normal_velocity_from_temperature();

    ierr = VecGetArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
    error = 0;
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(fabs(phi_p[n]) < dxyz_close_interface)
        error = MAX(error, fabs(normal_velocity_tmp_p[n]-normal_velocity_np1_p[n]));
    }
    ierr = VecRestoreArray(normal_velocity_tmp, &normal_velocity_tmp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    error /= MAX(vgamma_max,1e-7);

    matrices_are_constructed = true;
    ierr = PetscPrintf(p4est->mpicomm, "Convergence iteration #%d, max_velo = %e, error = %e\n", iteration, vgamma_max, error); CHKERRXX(ierr);
    iteration++;
  }
//  throw std::invalid_argument("");
//  compare_velocity_temperature_vs_concentration();

//  compare_normal_velocity_temperature_vs_concentration();

  compute_velocity_from_temperature();
//  compute_velocity_from_concentration();
  compute_dt();
  update_grid();

  ierr = VecDestroy(normal_velocity_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(normal_velocity_cl ); CHKERRXX(ierr);
}




void my_p4est_bialloy_t::compare_normal_velocity_temperature_vs_concentration()
{
  Vec src, out;

  Vec vn_tmp;
  ierr = VecDuplicate(phi, &vn_tmp); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(vn_tmp             , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vn_tmp             , &out); CHKERRXX(ierr);

  Vec vn_t;
  ierr = VecDuplicate(phi, &vn_t); CHKERRXX(ierr);

  compute_normal_velocity_from_temperature();

  ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(vn_t               , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vn_t               , &out); CHKERRXX(ierr);

  Vec vn_c;
  ierr = VecDuplicate(phi, &vn_c); CHKERRXX(ierr);

  compute_normal_velocity_from_concentration();

  ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(vn_c               , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vn_c               , &out); CHKERRXX(ierr);


  const double *vn_c_p, *vn_t_p, *phi_p;
  ierr = VecGetArrayRead(vn_c, &vn_c_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vn_t, &vn_t_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi , &phi_p ); CHKERRXX(ierr);

  double err = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(fabs(phi_p[n])<dxyz_min)
    {
      err = MAX(err, fabs((vn_t_p[n]-vn_c_p[n])));//vn_t_p[n]));
    }
  }

  ierr = VecRestoreArrayRead(vn_c, &vn_c_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vn_t, &vn_t_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi , &phi_p ); CHKERRXX(ierr);

  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  ierr = PetscPrintf(p4est->mpicomm, "maximum difference between velocities = %e\n", err); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(normal_velocity_np1, &out); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(vn_tmp             , &src); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vn_tmp             , &src); CHKERRXX(ierr);

  ierr = VecDestroy(vn_tmp); CHKERRXX(ierr);
  ierr = VecDestroy(vn_t  ); CHKERRXX(ierr);
  ierr = VecDestroy(vn_c  ); CHKERRXX(ierr);
}



void my_p4est_bialloy_t::save_VTK(int iter)
{
  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to the correct path if you want to save visuals.\n");
    return;
  }

  char name[1000];
#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/bialloy_%d_%dx%dx%d.%05d", out_dir, p4est->mpisize, brick->nxyztrees[0], brick->nxyztrees[1], brick->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/bialloy_%d_%dx%d.%05d", out_dir, p4est->mpisize, brick->nxyztrees[0], brick->nxyztrees[1], iter);
#endif

  /* create a vector for the temperature instead of solid / liquid */
  Vec temperature;
  ierr = VecDuplicate(phi, &temperature); CHKERRXX(ierr);
  double *temperature_tmp_p;
  ierr = VecGetArray(temperature, &temperature_tmp_p); CHKERRXX(ierr);
  const double *phi_p, *temperature_s_p, *temperature_l_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_s_np1, &temperature_s_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_l_np1, &temperature_l_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    temperature_tmp_p[n] = (phi_p[n]<0 ? temperature_s_p[n] : temperature_l_p[n]);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_s_np1, &temperature_s_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_l_np1, &temperature_l_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(temperature, &temperature_tmp_p); CHKERRXX(ierr);

  /* if the domain is periodic, create a temporary tree without periodicity for visualization */
  bool periodic = false;
  for(int dir=0; dir<P4EST_DIM; ++dir)
    periodic = periodic || is_periodic(p4est, dir);

  my_p4est_brick_t brick_vis;
  p4est_connectivity_t *connectivity_vis = NULL;
  p4est_t *p4est_vis;
  p4est_ghost_t *ghost_vis;
  p4est_nodes_t *nodes_vis;
  Vec phi_vis;
  Vec temperature_vis;
  Vec cl_vis;
  Vec normal_velocity_vis;
  Vec kappa_vis;

  double *phi_vis_p;

  if(periodic)
  {
    bool is_grid_changing = true;
    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
    my_p4est_interpolation_nodes_t interp(ngbd);

    double xyz_min[P4EST_DIM];
    double xyz_max[P4EST_DIM];
    xyz_min_max(p4est, xyz_min, xyz_max);

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
    interp.set_input(temperature, linear);
    interp.interpolate(temperature_vis);

    ierr = VecDuplicate(phi_vis, &cl_vis); CHKERRXX(ierr);
    interp.set_input(cl_np1, linear);
    interp.interpolate(cl_vis);

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
    temperature_vis = temperature;
    cl_vis = cl_n;
    normal_velocity_vis = normal_velocity_np1;
    kappa_vis = kappa;
  }

  const double *temperature_p, *cl_p;
  double *normal_velocity_np1_p;
  ierr = VecGetArrayRead(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(temperature_vis, &temperature_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cl_vis, &cl_p); CHKERRXX(ierr);
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

  my_p4est_vtk_write_all(  p4est_vis, nodes_vis, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           5, 1, name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "temperature", temperature_p,
                           VTK_POINT_DATA, "concentration", cl_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_POINT_DATA, "kappa", kappa_p,
                           VTK_CELL_DATA , "leaf_level", l_p);

  if(!periodic)
  {
    for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
      normal_velocity_np1_p[n] *= scaling;
  }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
  ierr = VecDestroy(temperature); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(temperature_vis, &temperature_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cl_vis, &cl_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_vis, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_vis, &normal_velocity_np1_p); CHKERRXX(ierr);

  if(periodic)
  {
    ierr = VecDestroy(normal_velocity_vis); CHKERRXX(ierr);
    ierr = VecDestroy(cl_vis); CHKERRXX(ierr);
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

//void my_p4est_bialloy_t::solve_temperature_old()
//{
//  double *phi_p, *rhs_p, *temperature_n_p, *normal_velocity_np1_p;
//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
//  ierr = VecGetArray(temperature_n, &temperature_n_p); CHKERRXX(ierr);
//  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  double scaling_jump = latent_heat/thermal_conductivity * dt_n * thermal_diffusivity;
//  double p_000, p_m00, p_p00, p_0m0, p_0p0;
//#ifdef P4_TO_P8
//  double p_00m, p_00p;
//#endif

//  quad_neighbor_nodes_of_node_t qnnn;
//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  {
//    rhs_p[n] = temperature_n_p[n];

//    qnnn = ngbd->get_neighbors(n);
//#ifdef P4_TO_P8
//    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
//    if (p_000*p_m00<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_m00_m0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_mm;
//      else if(qnnn.d_m00_m0==0 && qnnn.d_m00_0p==0) neigh = qnnn.node_m00_mp;
//      else if(qnnn.d_m00_p0==0 && qnnn.d_m00_0m==0) neigh = qnnn.node_m00_pm;
//      else                                          neigh = qnnn.node_m00_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
//    }
//    if (p_000*p_p00<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_p00_m0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_mm;
//      else if(qnnn.d_p00_m0==0 && qnnn.d_p00_0p==0) neigh = qnnn.node_p00_mp;
//      else if(qnnn.d_p00_p0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_pm;
//      else                                          neigh = qnnn.node_p00_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
//    }
//    if (p_000*p_0m0<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_0m0_m0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_mm;
//      else if(qnnn.d_0m0_m0==0 && qnnn.d_0m0_0p==0) neigh = qnnn.node_0m0_mp;
//      else if(qnnn.d_0m0_p0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_pm;
//      else                                          neigh = qnnn.node_0m0_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
//    }
//    if (p_000*p_0p0<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_0p0_m0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_mm;
//      else if(qnnn.d_0p0_m0==0 && qnnn.d_0p0_0p==0) neigh = qnnn.node_0p0_mp;
//      else if(qnnn.d_0p0_p0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_pm;
//      else                                          neigh = qnnn.node_0p0_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
//    }
//    if (p_000*p_00m<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_00m_m0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_mm;
//      else if(qnnn.d_00m_m0==0 && qnnn.d_00m_0p==0) neigh = qnnn.node_00m_mp;
//      else if(qnnn.d_00m_p0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_pm;
//      else                                          neigh = qnnn.node_00m_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
//    }
//    if (p_000*p_00p<=0){
//      p4est_locidx_t neigh;
//      if     (qnnn.d_00p_m0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_mm;
//      else if(qnnn.d_00p_m0==0 && qnnn.d_00p_0p==0) neigh = qnnn.node_00p_mp;
//      else if(qnnn.d_00p_p0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_pm;
//      else                                          neigh = qnnn.node_00p_pp;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[2]) * scaling_jump;
//    }
//#else
//    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
//    if (p_000*p_m00<=0){
//      p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
//    }
//    if (p_000*p_p00<=0){
//      p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[0]) * scaling_jump;
//    }
//    if (p_000*p_0m0<=0){
//      p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
//    }
//    if (p_000*p_0p0<=0){
//      p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
//      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dxyz[1]) * scaling_jump;
//    }
//#endif
//  }

//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(temperature_n, &temperature_n_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  my_p4est_poisson_nodes_t solver_t(ngbd);
//  solver_t.set_bc(bc_t);
//  solver_t.set_mu(dt_n*thermal_diffusivity);
//  solver_t.set_diagonal(1);
//  solver_t.set_rhs(rhs);
////  solver_t.set_is_matrix_computed(matrices_are_constructed);

//  solver_t.solve(temperature_np1);

//  Vec src, out;
//  ierr = VecGhostGetLocalForm(temperature_np1, &src); CHKERRXX(ierr);
//  ierr = VecGhostGetLocalForm(t_interface    , &out); CHKERRXX(ierr);
//  ierr = VecCopy(src, out); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(temperature_np1, &src); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(t_interface    , &out); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi, t_interface);
//}

//void my_p4est_bialloy_t::solve_concentration_old()
//{
//  /* initialize the boundary condition on the interface */
//  Vec c_interface_tmp;
//  ierr = VecDuplicate(phi, &c_interface_tmp); CHKERRXX(ierr);

//  const double *normal_p[P4EST_DIM];
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }

//  double *c_interface_tmp_p, *kappa_p, *t_interface_p, *normal_velocity_np1_p;
//  ierr = VecGetArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
//  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//#ifdef P4_TO_P8
//    double theta_xz = atan2(normal_p[2][n], normal_p[0][n]);
//    double theta_yz = atan2(normal_p[2][n], normal_p[1][n]);
//    c_interface_tmp_p[n] = (t_interface_p[n]-Tm)/ml
//        + epsilon_c*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*kappa_p[n]/ml
//        + epsilon_v*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*normal_velocity_np1_p[n]/ml;
//#else
//    double theta = atan2(normal_p[1][n], normal_p[0][n]);
//    c_interface_tmp_p[n] = (t_interface_p[n]-Tm)/ml
//        + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta))*kappa_p[n]/ml
//        + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta))*normal_velocity_np1_p[n]/ml;
//#endif
//  }


//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
//  }
//  ierr = VecRestoreArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_from_interface_to_whole_domain_TVD(phi, c_interface_tmp, c_interface);

//  my_p4est_interpolation_nodes_t interface_value_c(ngbd);
//  interface_value_c.set_input(c_interface, linear);
//  bc_cl.setInterfaceValue(interface_value_c);

//  /* compute the rhs for concentration */
//  Vec src, out;
//  ierr = VecGhostGetLocalForm(cl_n, &src); CHKERRXX(ierr);
//  ierr = VecGhostGetLocalForm(rhs , &out); CHKERRXX(ierr);
//  ierr = VecCopy(src, out); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(cl_n, &src); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(rhs , &out); CHKERRXX(ierr);

//  double *phi_p;
//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//    phi_p[n] *= -1;
//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

//  my_p4est_poisson_nodes_t solver_c(ngbd);
//  solver_c.set_phi(phi);
//  solver_c.set_bc(bc_cl);
//  solver_c.set_mu(dt_n*solute_diffusivity_l);
//  solver_c.set_diagonal(1);
//  solver_c.set_rhs(rhs);
////  solver_c.set_is_matrix_computed(matrices_are_constructed);

//  solver_c.solve(cl_np1);

//  ls.extend_Over_Interface_TVD(phi, cl_np1);

//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//    phi_p[n] *= -1;
//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

//  ierr = VecGhostGetLocalForm(cs_np1, &src); CHKERRXX(ierr);
//  ierr = VecSet(src, kp*c0); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(cs_np1, &src); CHKERRXX(ierr);

//  ierr = VecDestroy(c_interface_tmp); CHKERRXX(ierr);
//}
