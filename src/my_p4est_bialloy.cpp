#include "my_p4est_bialloy.h"

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




my_p4est_bialloy_t::my_p4est_bialloy_t(my_p4est_node_neighbors_t *ngbd)
  : brick(ngbd->myb), connectivity(ngbd->p4est->connectivity), p4est(ngbd->p4est), ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy), ngbd(ngbd),
    temperature_n(NULL), temperature_np1(NULL), t_interface(NULL),
    cs_n(NULL), cs_np1(NULL), cl_n(NULL), cl_np1(NULL), c_interface(NULL),
    u_interface_n(NULL), u_interface_np1(NULL),
    v_interface_n(NULL), v_interface_np1(NULL),
    #ifdef P4_TO_P8
    w_interface_n(NULL), w_interface_np1(NULL),
    #endif
    normal_velocity_np1(NULL),
    phi(NULL), nx(NULL), ny(NULL),
    #ifdef P4_TO_P8
    nz(NULL),
    #endif
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
  solute_diffusivity_s = 1e-13;     /* cm2.s-1     */
  cooling_velocity     = 0.01;      /* cm.s-1      */
  kp                   = 0.86;
  c0                   = 0.40831;   /* at frac.    */
  ml                   = -357;      /* K / at frac.*/
  epsilon_anisotropy   = 0.05;
  epsilon_c            = 2.7207e-5; /* cm.K     */
  epsilon_v            = 2.27e-2;   /* s.K.cm-1 */

  splitting_criteria_t *data = (splitting_criteria_t*)this->p4est->user_pointer;

  p4est_topidx_t vm = this->p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = this->p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = this->p4est->connectivity->vertices[3*vm + 0];
  double ymin = this->p4est->connectivity->vertices[3*vm + 1];
  double xmax = this->p4est->connectivity->vertices[3*vp + 0];
  double ymax = this->p4est->connectivity->vertices[3*vp + 1];
  dx = (xmax-xmin) / pow(2., (double) data->max_lvl);
  dy = (ymax-ymin) / pow(2., (double) data->max_lvl);
#ifdef P4_TO_P8
  double zmin = this->p4est->connectivity->vertices[3*vm + 2];
  double zmax = this->p4est->connectivity->vertices[3*vp + 2];
  dz = (zmax-zmin) / pow(2., (double) data->max_lvl);
#endif
}




my_p4est_bialloy_t::~my_p4est_bialloy_t()
{
  if(temperature_n  !=NULL) { ierr = VecDestroy(temperature_n);   CHKERRXX(ierr); }
  if(temperature_np1!=NULL) { ierr = VecDestroy(temperature_np1); CHKERRXX(ierr); }
  if(t_interface    !=NULL) { ierr = VecDestroy(t_interface);     CHKERRXX(ierr); }

  if(cl_n       !=NULL) { ierr = VecDestroy(cl_n);        CHKERRXX(ierr); }
  if(cs_n       !=NULL) { ierr = VecDestroy(cs_n);        CHKERRXX(ierr); }
  if(cl_np1     !=NULL) { ierr = VecDestroy(cl_np1);      CHKERRXX(ierr); }
  if(cs_np1     !=NULL) { ierr = VecDestroy(cs_np1);      CHKERRXX(ierr); }
  if(c_interface!=NULL) { ierr = VecDestroy(c_interface); CHKERRXX(ierr); }

  if(normal_velocity_np1!=NULL) { ierr = VecDestroy(normal_velocity_np1); CHKERRXX(ierr); }
  if(u_interface_n      !=NULL) { ierr = VecDestroy(u_interface_n);       CHKERRXX(ierr); }
  if(v_interface_n      !=NULL) { ierr = VecDestroy(v_interface_n);       CHKERRXX(ierr); }
  if(u_interface_np1    !=NULL) { ierr = VecDestroy(u_interface_np1);     CHKERRXX(ierr); }
  if(v_interface_np1    !=NULL) { ierr = VecDestroy(v_interface_np1);     CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if(w_interface_n      !=NULL) { ierr = VecDestroy(w_interface_n);       CHKERRXX(ierr); }
  if(w_interface_np1    !=NULL) { ierr = VecDestroy(w_interface_np1);     CHKERRXX(ierr); }
#endif

  if(nx   !=NULL) { ierr = VecDestroy(nx);    CHKERRXX(ierr); }
  if(ny   !=NULL) { ierr = VecDestroy(ny);    CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if(nz   !=NULL) { ierr = VecDestroy(nz);    CHKERRXX(ierr); }
#endif
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
                                         double solute_diffusivity_s,
                                         double cooling_velocity,
                                         double kp,
                                         double c0,
                                         double ml,
                                         double Tm,
                                         double epsilon_anisotropy,
                                         double epsilon_c,
                                         double epsilon_v,
                                         double scaling )
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
                                CF_3& bc_wall_value_cs,
                                CF_3& bc_wall_value_cl)
#else
void my_p4est_bialloy_t::set_bc(WallBC2D& bc_wall_type_t,
                                WallBC2D& bc_wall_type_c,
                                CF_2& bc_wall_value_t,
                                CF_2& bc_wall_value_cs,
                                CF_2& bc_wall_value_cl)
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
}





void my_p4est_bialloy_t::set_temperature(Vec temperature)
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





void my_p4est_bialloy_t::set_concentration(Vec cl, Vec cs)
{
  cl_n = cl;
  cs_n = cs;

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
}





void my_p4est_bialloy_t::set_normal_velocity(Vec v)
{
  normal_velocity_np1 = v;

  Vec src;
  ierr = VecDuplicate(v, &u_interface_np1); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(u_interface_np1, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(u_interface_np1, &src); CHKERRXX(ierr);

  ierr = VecDuplicate(v, &v_interface_np1); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(v_interface_np1, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(v_interface_np1, &src); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ierr = VecDuplicate(v, &w_interface_np1); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(w_interface_np1, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(w_interface_np1, &src); CHKERRXX(ierr);
#endif

  ierr = VecDuplicate(v, &u_interface_n); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(u_interface_n, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(u_interface_n, &src); CHKERRXX(ierr);

  ierr = VecDuplicate(v, &v_interface_n); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(v_interface_n, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(v_interface_n, &src); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ierr = VecDuplicate(v, &w_interface_n); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(w_interface_n, &src); CHKERRXX(ierr);
  ierr = VecSet(src, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(w_interface_n, &src); CHKERRXX(ierr);
#endif
}





void my_p4est_bialloy_t::set_dt( double dt )
{
  dt_nm1 = dt;
  dt_n   = dt;
}





void my_p4est_bialloy_t::compute_normal_and_curvature()
{
  if(nx!=NULL)    { ierr = VecDestroy(nx)   ; CHKERRXX(ierr); }
  if(ny!=NULL)    { ierr = VecDestroy(ny)   ; CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if(nz!=NULL)    { ierr = VecDestroy(nz)   ; CHKERRXX(ierr); }
#endif
  if(kappa!=NULL) { ierr = VecDestroy(kappa); CHKERRXX(ierr); }

  ierr = VecCreateGhostNodes(p4est, nodes, &nx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est, nodes, &nz); CHKERRXX(ierr);
#endif
  ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);

  double *nx_p, *ny_p, *kappa_p, *phi_p;
  ierr = VecGetArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecGetArray(ny, &ny_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double *nz_p;
  ierr = VecGetArray(nz, &nz_p); CHKERRXX(ierr);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    qnnn = ngbd->get_neighbors(n);
    nx_p[n] = qnnn.dx_central(phi_p);
    ny_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    nz_p[n] = qnnn.dz_central(phi_p);
    double norm = sqrt(SQR(nx_p[n]) + SQR(ny_p[n]) + SQR(nz_p[n]));
#else
    double norm = sqrt(SQR(nx_p[n]) + SQR(ny_p[n]));
#endif

    nx_p[n] = norm<EPS ? 0 : nx_p[n]/norm;
    ny_p[n] = norm<EPS ? 0 : ny_p[n]/norm;
#ifdef P4_TO_P8
    nz_p[n] = norm<EPS ? 0 : nz_p[n]/norm;
#endif
  }

  ierr = VecGhostUpdateBegin(nx, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecGhostUpdateBegin(ny, INSERT_VALUES, SCATTER_FORWARD);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateBegin(nz, INSERT_VALUES, SCATTER_FORWARD);
#endif

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    qnnn = ngbd->get_neighbors(n);
    nx_p[n] = qnnn.dx_central(phi_p);
    ny_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    nz_p[n] = qnnn.dz_central(phi_p);
    double norm = sqrt(SQR(nx_p[n]) + SQR(ny_p[n]) + SQR(nz_p[n]));
#else
    double norm = sqrt(SQR(nx_p[n]) + SQR(ny_p[n]));
#endif

    nx_p[n] = norm<EPS ? 0 : nx_p[n]/norm;
    ny_p[n] = norm<EPS ? 0 : ny_p[n]/norm;
#ifdef P4_TO_P8
    nz_p[n] = norm<EPS ? 0 : nz_p[n]/norm;
#endif
  }
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateEnd(nx, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecGhostUpdateEnd(ny, INSERT_VALUES, SCATTER_FORWARD);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateEnd(nz, INSERT_VALUES, SCATTER_FORWARD);
#endif

  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
    kappa_p[n] = MAX(MIN(qnnn.dx_central(nx_p) + qnnn.dy_central(ny_p) + qnnn.dz_central(nz_p), 1/MAX(dx,dy,dz)), -1/MAX(dx,dy,dz));
#else
    kappa_p[n] = MAX(MIN(qnnn.dx_central(nx_p) + qnnn.dy_central(ny_p), 1/MAX(dx,dy)), -1/MAX(dx,dy));
#endif
  }
  ierr = VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD);
  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    qnnn = ngbd->get_neighbors(n);
#ifdef P4_TO_P8
    kappa_p[n] = MAX(MIN(qnnn.dx_central(nx_p) + qnnn.dy_central(ny_p) + qnnn.dz_central(nz_p), 1/MAX(dx,dy,dz)), -1/MAX(dx,dy,dz));
#else
    kappa_p[n] = MAX(MIN(qnnn.dx_central(nx_p) + qnnn.dy_central(ny_p), 1/MAX(dx,dy)), -1/MAX(dx,dy));
#endif
  }
  ierr = VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ny, &ny_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(nz, &nz_p); CHKERRXX(ierr);
#endif
}




void my_p4est_bialloy_t::compute_normal_velocity()
{
  Vec v_gamma;
  ierr = VecDuplicate(phi, &v_gamma); CHKERRXX(ierr);
  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  double *nx_p, *ny_p, *cl_n_p, *c_interface_p;
  ierr = VecGetArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecGetArray(ny, &ny_p); CHKERRXX(ierr);
  ierr = VecGetArray(cl_n, &cl_n_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double *nz_p;
  ierr = VecGetArray(nz, &nz_p); CHKERRXX(ierr);
#endif


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
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n] + qnnn.dz_central(cl_n_p)*nz_p[n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*nx_p[n] + qnnn.dy_central(cs_n_p)*ny_p[n] + qnnn.dz_central(cs_n_p)*nz_p[n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*nx_p[n] + qnnn.dy_central(cs_n_p)*ny_p[n];
#endif
      v_gamma_p[n] = (dcs_dn*solute_diffusivity_s - dcl_dn*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
    }
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n] + qnnn.dz_central(cl_n_p)*nz_p[n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*nx_p[n] + qnnn.dy_central(cs_n_p)*ny_p[n] + qnnn.dz_central(cs_n_p)*nz_p[n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n];
      double dcs_dn = qnnn.dx_central(cs_n_p)*nx_p[n] + qnnn.dy_central(cs_n_p)*ny_p[n];
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
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n] + qnnn.dz_central(cl_n_p)*nz_p[n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n];
#endif
      v_gamma_p[n] = -dcl_dn*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
    }
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n] + qnnn.dz_central(cl_n_p)*nz_p[n];
#else
      double dcl_dn = qnnn.dx_central(cl_n_p)*nx_p[n] + qnnn.dy_central(cl_n_p)*ny_p[n];
#endif
      v_gamma_p[n] = -dcl_dn*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
    }
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ny, &ny_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(nz, &nz_p); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(cl_n, &cl_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma, normal_velocity_np1);

  ierr = VecDestroy(v_gamma); CHKERRXX(ierr);
}





void my_p4est_bialloy_t::compute_velocity()
{
  Vec u_gamma;
  ierr = VecDuplicate(nx, &u_gamma); CHKERRXX(ierr);
  double *u_gamma_p;
  ierr = VecGetArray(u_gamma, &u_gamma_p); CHKERRXX(ierr);

  Vec v_gamma;
  ierr = VecDuplicate(ny, &v_gamma); CHKERRXX(ierr);
  double *v_gamma_p;
  ierr = VecGetArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Vec w_gamma;
  ierr = VecDuplicate(nz, &w_gamma); CHKERRXX(ierr);
  double *w_gamma_p;
  ierr = VecGetArray(w_gamma, &w_gamma_p); CHKERRXX(ierr);
#endif

  double *nx_p, *ny_p, *cl_np1_p, *c_interface_p;
  ierr = VecGetArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecGetArray(ny, &ny_p); CHKERRXX(ierr);
  ierr = VecGetArray(cl_np1, &cl_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(c_interface, &c_interface_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double *nz_p;
  ierr = VecGetArray(nz, &nz_p); CHKERRXX(ierr);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  if(solve_concentration_solid)
  {
    double *cs_np1_p;
    ierr = VecGetArray(cs_np1, &cs_np1_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

      u_gamma_p[n] = (qnnn.dx_central(cs_np1_p)*solute_diffusivity_s - qnnn.dx_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[n] = (qnnn.dy_central(cs_np1_p)*solute_diffusivity_s - qnnn.dy_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      w_gamma_p[n] = (qnnn.dz_central(cs_np1_p)*solute_diffusivity_s - qnnn.dz_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    ierr = VecGhostUpdateBegin(u_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateBegin(w_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

      u_gamma_p[n] = (qnnn.dx_central(cs_np1_p)*solute_diffusivity_s - qnnn.dx_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[n] = (qnnn.dy_central(cs_np1_p)*solute_diffusivity_s - qnnn.dy_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      w_gamma_p[n] = (qnnn.dz_central(cs_np1_p)*solute_diffusivity_s - qnnn.dz_central(cl_np1_p)*solute_diffusivity_l) / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    ierr = VecGhostUpdateEnd(u_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateEnd(w_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(cs_np1, &cs_np1_p); CHKERRXX(ierr);
  }
  else
  {
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      qnnn = ngbd->get_neighbors(n);

      u_gamma_p[n] = -qnnn.dx_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[n] = -qnnn.dy_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      w_gamma_p[n] = -qnnn.dz_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    ierr = VecGhostUpdateBegin(u_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateBegin(w_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      qnnn = ngbd->get_neighbors(n);

      u_gamma_p[n] = -qnnn.dx_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
      v_gamma_p[n] = -qnnn.dy_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#ifdef P4_TO_P8
      w_gamma_p[n] = -qnnn.dz_central(cl_np1_p)*solute_diffusivity_l / (1-kp) / MAX(c_interface_p[n], 1e-7);
#endif
    }
    ierr = VecGhostUpdateEnd(u_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(v_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateEnd(w_gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ny, &ny_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(nz, &nz_p); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(cl_np1, &cl_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(c_interface, &c_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(u_gamma, &u_gamma_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(v_gamma, &v_gamma_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(w_gamma, &w_gamma_p); CHKERRXX(ierr);
#endif

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, u_gamma, u_interface_np1);
  ls.extend_from_interface_to_whole_domain_TVD(phi, v_gamma, v_interface_np1);
#ifdef P4_TO_P8
  ls.extend_from_interface_to_whole_domain_TVD(phi, w_gamma, w_interface_np1);
#endif

  ierr = VecDestroy(u_gamma); CHKERRXX(ierr);
  ierr = VecDestroy(v_gamma); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(w_gamma); CHKERRXX(ierr);
#endif
}





void my_p4est_bialloy_t::solve_temperature()
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
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dx) * scaling_jump;
    }
    if (p_000*p_p00<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_p00_m0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_mm;
      else if(qnnn.d_p00_m0==0 && qnnn.d_p00_0p==0) neigh = qnnn.node_p00_mp;
      else if(qnnn.d_p00_p0==0 && qnnn.d_p00_0m==0) neigh = qnnn.node_p00_pm;
      else                                          neigh = qnnn.node_p00_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dx) * scaling_jump;
    }
    if (p_000*p_0m0<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_0m0_m0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_mm;
      else if(qnnn.d_0m0_m0==0 && qnnn.d_0m0_0p==0) neigh = qnnn.node_0m0_mp;
      else if(qnnn.d_0m0_p0==0 && qnnn.d_0m0_0m==0) neigh = qnnn.node_0m0_pm;
      else                                          neigh = qnnn.node_0m0_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dy) * scaling_jump;
    }
    if (p_000*p_0p0<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_0p0_m0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_mm;
      else if(qnnn.d_0p0_m0==0 && qnnn.d_0p0_0p==0) neigh = qnnn.node_0p0_mp;
      else if(qnnn.d_0p0_p0==0 && qnnn.d_0p0_0m==0) neigh = qnnn.node_0p0_pm;
      else                                          neigh = qnnn.node_0p0_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dy) * scaling_jump;
    }
    if (p_000*p_00m<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_00m_m0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_mm;
      else if(qnnn.d_00m_m0==0 && qnnn.d_00m_0p==0) neigh = qnnn.node_00m_mp;
      else if(qnnn.d_00m_p0==0 && qnnn.d_00m_0m==0) neigh = qnnn.node_00m_pm;
      else                                          neigh = qnnn.node_00m_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dz) * scaling_jump;
    }
    if (p_000*p_00p<=0){
      p4est_locidx_t neigh;
      if     (qnnn.d_00p_m0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_mm;
      else if(qnnn.d_00p_m0==0 && qnnn.d_00p_0p==0) neigh = qnnn.node_00p_mp;
      else if(qnnn.d_00p_p0==0 && qnnn.d_00p_0m==0) neigh = qnnn.node_00p_pm;
      else                                          neigh = qnnn.node_00p_pp;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dz) * scaling_jump;
    }
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
    if (p_000*p_m00<=0){
      p4est_locidx_t neigh = (qnnn.d_m00_m0==0) ? qnnn.node_m00_mm : qnnn.node_m00_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dx) * scaling_jump;
    }
    if (p_000*p_p00<=0){
      p4est_locidx_t neigh = (qnnn.d_p00_m0==0) ? qnnn.node_p00_mm : qnnn.node_p00_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dx) * scaling_jump;
    }
    if (p_000*p_0m0<=0){
      p4est_locidx_t neigh = (qnnn.d_0m0_m0==0) ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dy) * scaling_jump;
    }
    if (p_000*p_0p0<=0){
      p4est_locidx_t neigh = (qnnn.d_0p0_m0==0) ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
      rhs_p[n] += fabs(phi_p[neigh]) * normal_velocity_np1_p[neigh] / SQR(dy) * scaling_jump;
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

  Vec src, out;
  ierr = VecGhostGetLocalForm(temperature_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(t_interface    , &out); CHKERRXX(ierr);
  ierr = VecCopy(src, out); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(temperature_np1, &src); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(t_interface    , &out); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, t_interface);
}

void my_p4est_bialloy_t::solve_concentration()
{
  /* initialize the boundary condition on the interface */
  Vec c_interface_tmp;
  ierr = VecDuplicate(phi, &c_interface_tmp); CHKERRXX(ierr);

  double *c_interface_tmp_p, *nx_p, *ny_p, *kappa_p, *t_interface_p, *normal_velocity_np1_p;
  ierr = VecGetArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecGetArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecGetArray(ny, &ny_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double *nz_p;
  ierr = VecGetArray(nz, &nz_p); CHKERRXX(ierr);
#endif

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    double theta_xz = atan2(nz_p[n], nx_p[n]);
    double theta_yz = atan2(nz_p[n], ny_p[n]);
    c_interface_tmp_p[n] = (t_interface_p[n]-Tm)/ml
        + epsilon_c*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*kappa_p[n]/ml
        + epsilon_v*(1-15*epsilon_anisotropy*.5*(cos(4*theta_xz)+cos(4*theta_yz)))*normal_velocity_np1_p[n]/ml;
#else
    double theta = atan2(ny_p[n], nx_p[n]);
    c_interface_tmp_p[n] = (t_interface_p[n]-Tm)/ml
        + epsilon_c*(1-15*epsilon_anisotropy*cos(4*theta))*kappa_p[n]/ml
        + epsilon_v*(1-15*epsilon_anisotropy*cos(4*theta))*normal_velocity_np1_p[n]/ml;
#endif
  }

  ierr = VecRestoreArray(c_interface_tmp, &c_interface_tmp_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(nx, &nx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ny, &ny_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(nz, &nz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(t_interface, &t_interface_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, c_interface_tmp, c_interface);

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

  ls.extend_Over_Interface_TVD(phi, cl_np1);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] *= -1;
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(cs_np1, &src); CHKERRXX(ierr);
  ierr = VecSet(src, kp*c0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(cs_np1, &src); CHKERRXX(ierr);

  ierr = VecDestroy(c_interface_tmp); CHKERRXX(ierr);
}




void my_p4est_bialloy_t::compute_dt()
{
  double *u_interface_np1_p, *v_interface_np1_p;
  ierr = VecGetArray(u_interface_np1, &u_interface_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(v_interface_np1, &v_interface_np1_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double *w_interface_np1_p;
  ierr = VecGetArray(w_interface_np1, &w_interface_np1_p); CHKERRXX(ierr);
#endif

//  double *kappa_p;
//  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);

  double u_max = 0;
//  double kappa_min = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    u_max = MAX(u_max, MAX(fabs(u_interface_np1_p[n]), fabs(v_interface_np1_p[n]), fabs(w_interface_np1_p[n])));
#else
    u_max = MAX(u_max, fabs(u_interface_np1_p[n]), fabs(v_interface_np1_p[n]));
#endif
//    kappa_min = MIN(kappa_min, kappa_p[n]);
  }

//  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(u_interface_np1, &u_interface_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(v_interface_np1, &v_interface_np1_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(w_interface_np1, &w_interface_np1_p); CHKERRXX(ierr);
#endif

  MPI_Allreduce(MPI_IN_PLACE, &u_max    , 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
//  MPI_Allreduce(MPI_IN_PLACE, &kappa_min, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  dt_nm1 = dt_n;
  dt_n = .5 * MIN(dx,dy) * MIN(1/u_max, 1/cooling_velocity);

//  if(dt_n>0.5/MAX(1e-7, kappa_min))
//  {
//    dt_n = MIN(dt_n, 0.5/MAX(1e-7, kappa_min));
//    ierr = PetscPrintf(p4est->mpicomm, "KAPPA LIMITING TIME STEP\n"); CHKERRXX(ierr);
//  }
}





void my_p4est_bialloy_t::update_grid()
{
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

  /* bousouf update this for second order in time */
#ifdef P4_TO_P8
  Vec vel_n[]   = { u_interface_n,   v_interface_n,   w_interface_n   };
  Vec vel_np1[] = { u_interface_np1, v_interface_np1, w_interface_np1 };
#else
  Vec vel_n[]   = { u_interface_n,   v_interface_n   };
  Vec vel_np1[] = { u_interface_np1, v_interface_np1 };
#endif
  sl.update_p4est(vel_n, vel_np1, dt_nm1, dt_n, phi);

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd);

  /* NOTE: should we do ghost updates ? or go through interpolations ? */
//  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz [] =
    {
      node_x_fr_n(n, p4est_np1, nodes_np1),
      node_y_fr_n(n, p4est_np1, nodes_np1)
#ifdef P4_TO_P8
      , node_z_fr_n(n, p4est_np1, nodes_np1),
#endif
    };

    interp.add_point(n, xyz);
  }

  ierr = VecDestroy(temperature_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_n); CHKERRXX(ierr);
  interp.set_input(temperature_np1, quadratic_non_oscillatory);
  interp.interpolate(temperature_n);
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

  ierr = VecDestroy(u_interface_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &u_interface_n); CHKERRXX(ierr);
  interp.set_input(u_interface_np1, quadratic_non_oscillatory);
  interp.interpolate(u_interface_n);
  ierr = VecDestroy(u_interface_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &u_interface_np1); CHKERRXX(ierr);

  ierr = VecDestroy(v_interface_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &v_interface_n); CHKERRXX(ierr);
  interp.set_input(v_interface_np1, quadratic_non_oscillatory);
  interp.interpolate(v_interface_n);
  ierr = VecDestroy(v_interface_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &v_interface_np1); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ierr = VecDestroy(w_interface_n); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &w_interface_n); CHKERRXX(ierr);
  interp.set_input(w_interface_np1, quadratic_non_oscillatory);
  interp.interpolate(w_interface_n);
  ierr = VecDestroy(w_interface_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &w_interface_np1); CHKERRXX(ierr);
#endif

  Vec normal_velocity_n;
  ierr = VecDuplicate(phi, &normal_velocity_n); CHKERRXX(ierr);
  interp.set_input(normal_velocity_np1, quadratic_non_oscillatory);
  interp.interpolate(normal_velocity_n);
  ierr = VecDestroy(normal_velocity_np1); CHKERRXX(ierr);
  normal_velocity_np1 = normal_velocity_n;

  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

  ierr = VecDestroy(t_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &t_interface); CHKERRXX(ierr);

  ierr = VecDestroy(c_interface); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c_interface); CHKERRXX(ierr);

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
  Vec err;
  ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
  double *err_p, *phi_p, *normal_velocity_np1_p;
  double error = 1;
  int iteration = 0;
  matrices_are_constructed = false;

  while(error>1e-5 && iteration<10)
  {

    Vec src, out;
    ierr = VecGhostGetLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(err                , &out); CHKERRXX(ierr);
    ierr = VecCopy(src, out); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(normal_velocity_np1, &src); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(err                , &out); CHKERRXX(ierr);

    solve_temperature();

    solve_concentration();

    compute_normal_velocity();

    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);
    error = 0;
    long int nb_points = 0;
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      err_p[n] -= normal_velocity_np1_p[n];
#ifdef P4_TO_P8
      if(fabs(phi_p[n]) < 4*MIN(dx,dy,dz))
#else
      if(fabs(phi_p[n]) < 4*MIN(dx,dy))
#endif
      {
        error += fabs(err_p[n]);
        nb_points++;
      }
    }
    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

    MPI_Allreduce(MPI_IN_PLACE, &error    , 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE, &nb_points, 1, MPI_LONG  , MPI_SUM, p4est->mpicomm);

    error /= (double) nb_points;

    matrices_are_constructed = true;
    iteration++;
  }

  compute_velocity();
  compute_dt();
  update_grid();

  ierr = VecDestroy(err); CHKERRXX(ierr);
}





void my_p4est_bialloy_t::save_VTK(int iter)
{
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, "/home/guittet/code/Output/p4est_bialloy");
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/bialloy_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << iter;


  double *phi_p, *temperature_p, *cl_p, *normal_velocity_np1_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(temperature_n, &temperature_p); CHKERRXX(ierr);
  ierr = VecGetArray(cl_n, &cl_p); CHKERRXX(ierr);
  ierr = VecGetArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] /= scaling;

  my_p4est_vtk_write_all(  p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           4, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "temperature", temperature_p,
                           VTK_POINT_DATA, "concentration", cl_p,
                           VTK_POINT_DATA, "un", normal_velocity_np1_p,
                           VTK_CELL_DATA , "leaf_level", l_p);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    normal_velocity_np1_p[n] *= scaling;

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(temperature_n, &temperature_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(cl_n, &cl_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_velocity_np1, &normal_velocity_np1_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}
