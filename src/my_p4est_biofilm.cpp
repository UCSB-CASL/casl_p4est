#ifdef P4_TO_P8
#include "my_p8est_biofilm.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_biofilm.h"
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
extern PetscLogEvent log_my_p4est_biofilm_one_step;
extern PetscLogEvent log_my_p4est_biofilm_compute_dt;
extern PetscLogEvent log_my_p4est_biofilm_compute_geometric_properties;
extern PetscLogEvent log_my_p4est_biofilm_compute_velocity;
extern PetscLogEvent log_my_p4est_biofilm_solve_concentration;
extern PetscLogEvent log_my_p4est_biofilm_solve_pressure;
extern PetscLogEvent log_my_p4est_biofilm_update_grid;
extern PetscLogEvent log_my_p4est_biofilm_save_vtk;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif




my_p4est_biofilm_t::my_p4est_biofilm_t(my_p4est_node_neighbors_t *ngbd)
  : brick_(ngbd->myb), connectivity_(ngbd->p4est->connectivity), p4est_(ngbd->p4est), ghost_(ngbd->ghost), nodes_(ngbd->nodes), hierarchy_(ngbd->hierarchy), ngbd_(ngbd)
{
  dxyz_min(p4est_, dxyz_);
#ifdef P4_TO_P8
  dxyz_min_ = MIN(dxyz_[0], dxyz_[1], dxyz_[2]);
  dxyz_max_ = MAX(dxyz_[0], dxyz_[1], dxyz_[2]);
  diag_ = sqrt(SQR(dxyz_[0]) + SQR(dxyz_[1]) + SQR(dxyz_[2]));
#else
  dxyz_min_ = MIN(dxyz_[0], dxyz_[1]);
  dxyz_max_ = MAX(dxyz_[0], dxyz_[1]);
  diag_ = sqrt(SQR(dxyz_[0]) + SQR(dxyz_[1]));
#endif
  dxyz_close_interface_ = 1.2*dxyz_max_;

  /* problem parameters */
  Df_      = 1; /* diffusivity of nutrients in air           - m^2/s     */
  Db_      = 1; /* diffusivity of nutrients in biofilm       - m^2/s     */
  Da_      = 1; /* diffusivity of nutrients in agar          - m^2/s     */
  rho_     = 1; /* density of biofilm                        - kg/m^3    */
  gam_     =.5; /* biofilm yield per nutrient mass           -           */
  sigma_   = 0; /* surface tension of air/film interface     - N/m       */
  lambda_  = 1; /* mobility of biofilm                       - m^4/(N*s) */
  scaling_ = 1;
  steady_state_ = 0;

  /* level set */
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_free_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_agar_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &phi_biof_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &kappa_); CHKERRXX(ierr);

  foreach_dimension(dim)
  {
    ierr = VecCreateGhostNodes(p4est_, nodes_, &normal_[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(normal_[dim], &phi_biof_dd_[dim]); CHKERRXX(ierr);
  }

  kappa_max_ = 0;

  /* concentration */
  ierr = VecDuplicate(phi_free_, &Ca0_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &Cb0_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &Cf0_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &Ca1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &Cb1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &Cf1_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &C_); CHKERRXX(ierr);

  /* pressure */
  ierr = VecDuplicate(phi_free_, &P_); CHKERRXX(ierr);

  /* velocity */
  ierr = VecDuplicate(phi_free_, &vn_); CHKERRXX(ierr);
  vn_max_ = 0;

  foreach_dimension(dim)
  {
    ierr = VecDuplicate(normal_[dim], &v0_[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(normal_[dim], &v1_[dim]); CHKERRXX(ierr);
  }

  /* solving nonlinear equation */
  iteration_scheme_ = 1;
  tolerance_        = 1.e-5;
  max_iterations_   = 10;

  /* general solver parameters */
  use_taylor_correction_ = 1;
  use_sc_scheme_         = 1;
  integration_order_     = 2;
  extend_iterations_     = 40;

  /* time discretization */
  time_scheme_      = 1;
  advection_scheme_ = 1;

  time_       = 0;
  dt_max_     = 1;
  dt0_        = dt_max_;
  dt1_        = dt_max_;
  cfl_number_ = 0.1;

  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v2;

  set_ghosted_vec(P_, 0);
  set_ghosted_vec(vn_, 0);

  foreach_dimension(dim)
  {
    set_ghosted_vec(v0_[dim], 0);
    set_ghosted_vec(v1_[dim], 0);
  }

  use_godunov_scheme_ = false;
  first_iteration_ = true;
}





my_p4est_biofilm_t::~my_p4est_biofilm_t()
{
  if (Ca0_ !=NULL) { ierr = VecDestroy(Ca0_); CHKERRXX(ierr); }
  if (Cb0_ !=NULL) { ierr = VecDestroy(Cb0_); CHKERRXX(ierr); }
  if (Cf0_ !=NULL) { ierr = VecDestroy(Cf0_); CHKERRXX(ierr); }

  if (Ca1_ !=NULL) { ierr = VecDestroy(Ca1_); CHKERRXX(ierr); }
  if (Cb1_ !=NULL) { ierr = VecDestroy(Cb1_); CHKERRXX(ierr); }
  if (Cf1_ !=NULL) { ierr = VecDestroy(Cf1_); CHKERRXX(ierr); }

  if (C_ !=NULL) { ierr = VecDestroy(C_); CHKERRXX(ierr); }

  if (P_ !=NULL) { ierr = VecDestroy(P_); CHKERRXX(ierr); }

  if (vn_ !=NULL) { ierr = VecDestroy(vn_); CHKERRXX(ierr); }

  foreach_dimension(dim)
  {
    if(v0_         [dim] != NULL) { ierr = VecDestroy(v0_         [dim]); CHKERRXX(ierr); }
    if(v1_         [dim] != NULL) { ierr = VecDestroy(v1_         [dim]); CHKERRXX(ierr); }
    if(phi_biof_dd_[dim] != NULL) { ierr = VecDestroy(phi_biof_dd_[dim]); CHKERRXX(ierr); }
    if(normal_     [dim] != NULL) { ierr = VecDestroy(normal_     [dim]); CHKERRXX(ierr); }
  }

  if (kappa_ != NULL) { ierr = VecDestroy(kappa_); CHKERRXX(ierr); }

  if (phi_biof_ !=NULL) { ierr = VecDestroy(phi_biof_); CHKERRXX(ierr); }
  if (phi_agar_ !=NULL) { ierr = VecDestroy(phi_agar_); CHKERRXX(ierr); }
  if (phi_free_ !=NULL) { ierr = VecDestroy(phi_free_); CHKERRXX(ierr); }


  /* destroy the p4est and its connectivity structure */
  delete ngbd_;
  delete hierarchy_;
  p4est_nodes_destroy(nodes_);
  p4est_ghost_destroy(ghost_);
  p4est_destroy      (p4est_);
  my_p4est_brick_destroy(connectivity_, brick_);
}





void my_p4est_biofilm_t::compute_geometric_properties()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);

  /* assemble phi_biof_ */
  double *phi_biof_ptr; ierr = VecGetArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  double *phi_agar_ptr; ierr = VecGetArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  double *phi_free_ptr; ierr = VecGetArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    phi_biof_ptr[n] = MAX(phi_agar_ptr[n], phi_free_ptr[n]);
  }

  ierr = VecRestoreArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  /* second order derivatives */
  ngbd_->second_derivatives_central(phi_biof_, phi_biof_dd_);

  /* normal and curvature */
  compute_normals_and_mean_curvature(*ngbd_, phi_biof_, normal_, kappa_);

  // extend kappa from interface
  Vec kappa_tmp;

  ierr = VecDuplicate(phi_free_, &kappa_tmp); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  ls.extend_from_interface_to_whole_domain_TVD(phi_biof_, kappa_, kappa_tmp);

  ierr = VecDestroy(kappa_); CHKERRXX(ierr);

  kappa_ = kappa_tmp;

//  // truncate extremely high curvature values
//  double kappa_rez_max = 1./dxyz_min_;

//  double *kappa_ptr;

//  ierr = VecGetArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

//  foreach_node(n, nodes_)
//  {
//    if      (kappa_ptr[n] > kappa_rez_max) kappa_ptr[n] = kappa_rez_max;
//    else if (kappa_ptr[n] <-kappa_rez_max) kappa_ptr[n] =-kappa_rez_max;
//  }

//  ierr = VecRestoreArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

  if (curvature_smoothing_ > 0)
  {
    compute_filtered_curvature();
  }

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}





int my_p4est_biofilm_t::one_step()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_one_step, 0, 0, 0, 0); CHKERRXX(ierr);

  solve_concentration();
  solve_pressure();
  compute_velocity_from_pressure();

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_one_step, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::solve_concentration()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_solve_concentration, 0, 0, 0, 0); CHKERRXX(ierr);

  // time discretization coefficients
  double a0, a1, a2;

  if (steady_state_)
  {
    a0 = 0; a1 = 0; a2 = 0;
  }
  else if (time_scheme_ == 1 || first_iteration_)
  {
    a0 = 1./dt0_;
    a1 =-1./dt0_;
    a2 = 0;
  }
  else if (time_scheme_ == 2)
  {
    double r = dt0_/dt1_;
    a0 = (1.+2.*r)/(1.+r)/dt0_;
    a1 =-(1.+r)/dt0_;
    a2 = r*r/(1.+r)/dt0_;
  }
  else
  {
    throw;
  }

  // compute diffusivities, diagonal term, right-hand sides and guess
  double *phi_biof_ptr;
  double *phi_free_ptr;
  double *phi_agar_ptr;

  ierr = VecGetArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);

  double *C_ptr;
  double *Ca0_ptr, *Ca1_ptr;
  double *Cb0_ptr, *Cb1_ptr;
  double *Cf0_ptr, *Cf1_ptr;

  ierr = VecGetArray(C_,   &C_ptr);   CHKERRXX(ierr);
  ierr = VecGetArray(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cf0_, &Cf0_ptr); CHKERRXX(ierr);

  if (time_scheme_ == 2)
  {
    ierr = VecGetArray(Ca1_, &Ca1_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(Cb1_, &Cb1_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(Cf1_, &Cf1_ptr); CHKERRXX(ierr);
  }

  Vec mu_m; double *mu_m_ptr; ierr = VecDuplicate(phi_free_, &mu_m); CHKERRXX(ierr);
  Vec mu_p; double *mu_p_ptr; ierr = VecDuplicate(phi_free_, &mu_p); CHKERRXX(ierr);

  Vec diag_m; double *diag_m_ptr; ierr = VecDuplicate(phi_free_, &diag_m); CHKERRXX(ierr);
  Vec diag_p; double *diag_p_ptr; ierr = VecDuplicate(phi_free_, &diag_p); CHKERRXX(ierr);

  Vec rhs_tmp_m; double *rhs_tmp_m_ptr; ierr = VecDuplicate(phi_free_, &rhs_tmp_m); CHKERRXX(ierr);
  Vec rhs_tmp_p; double *rhs_tmp_p_ptr; ierr = VecDuplicate(phi_free_, &rhs_tmp_p); CHKERRXX(ierr);

  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(diag_m, &diag_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(diag_p, &diag_p_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(rhs_tmp_m, &rhs_tmp_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_tmp_p, &rhs_tmp_p_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    // diffusivities
    mu_m_ptr[n] = Db_;
    mu_p_ptr[n] = phi_free_ptr[n] > phi_agar_ptr[n] ? Df_ : Da_;

    // diagonal terms
    diag_m_ptr[n] = a0;
    diag_p_ptr[n] = a0;

    // guess
    if (phi_biof_ptr[n] < 0)
    {
      C_ptr[n] = Cb0_ptr[n];
    }
    else
    {
      if (phi_agar_ptr[n] > phi_free_ptr[n]) C_ptr[n] = Ca0_ptr[n];
      else C_ptr[n] = Cf0_ptr[n];
    }
  }

  // additions to rhs due to time discretization
  if (steady_state_)
  {
    foreach_node(n, nodes_)
    {
      rhs_tmp_m_ptr[n] = 0;
      rhs_tmp_p_ptr[n] = 0;
    }
  }
  else if (time_scheme_ == 1 || first_iteration_)
  {
    foreach_node(n, nodes_)
    {
      rhs_tmp_m_ptr[n] = -(a1*Cb0_ptr[n]);
      rhs_tmp_p_ptr[n] = phi_free_ptr[n] > phi_agar_ptr[n] ? -(a1*Cf0_ptr[n]) : -(a1*Ca0_ptr[n]);
    }
  }
  else if (time_scheme_ == 2)
  {
    foreach_node(n, nodes_)
    {
      rhs_tmp_m_ptr[n] = -(a1*Cb0_ptr[n] + a2*Cb1_ptr[n]);
      rhs_tmp_p_ptr[n] = phi_free_ptr[n] > phi_agar_ptr[n] ? -(a1*Cf0_ptr[n] + a2*Cf1_ptr[n]) : -(a1*Ca0_ptr[n] + a2*Ca1_ptr[n]);
    }
  }
  else
  {
    throw;
  }

  ierr = VecRestoreArray(mu_m, &mu_m_ptr);           CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p, &mu_p_ptr);           CHKERRXX(ierr);

  ierr = VecRestoreArray(diag_m, &diag_m_ptr);       CHKERRXX(ierr);
  ierr = VecRestoreArray(diag_p, &diag_p_ptr);       CHKERRXX(ierr);

  ierr = VecRestoreArray(rhs_tmp_m, &rhs_tmp_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_tmp_p, &rhs_tmp_p_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_biof_, &phi_biof_ptr);  CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_free_, &phi_free_ptr);  CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_agar_, &phi_agar_ptr);  CHKERRXX(ierr);

  ierr = VecRestoreArray(C_,   &C_ptr);              CHKERRXX(ierr);
  ierr = VecRestoreArray(Ca0_, &Ca0_ptr);            CHKERRXX(ierr);
  ierr = VecRestoreArray(Cb0_, &Cb0_ptr);            CHKERRXX(ierr);
  ierr = VecRestoreArray(Cf0_, &Cf0_ptr);            CHKERRXX(ierr);

  if (time_scheme_ == 2)
  {
    ierr = VecRestoreArray(Ca1_, &Ca1_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(Cb1_, &Cb1_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(Cf1_, &Cf1_ptr); CHKERRXX(ierr);
  }

  // initialize a Poisson solver
  my_p4est_poisson_nodes_mls_t poisson_solver(ngbd_);

  poisson_solver.set_jump_scheme(0);
  poisson_solver.set_use_sc_scheme(use_sc_scheme_);
  poisson_solver.set_integration_order(integration_order_);
  poisson_solver.set_mu(mu_m, mu_p);
  poisson_solver.set_wc(*bc_wall_type_, *bc_wall_value_);
  poisson_solver.set_rhs(rhs_tmp_m, rhs_tmp_p);

  if (Da_ == 0 && Df_ == 0)
  {
    poisson_solver.add_boundary (MLS_INTERSECTION, phi_biof_, NULL, NEUMANN, zero_cf, zero_cf);
  }
  else if (Da_ == 0)
  {
    poisson_solver.add_boundary (MLS_INTERSECTION, phi_agar_, NULL, NEUMANN, zero_cf, zero_cf);
    poisson_solver.add_interface(MLS_INTERSECTION, phi_free_, NULL, zero_cf, zero_cf);
  }
  else if (Df_ == 0)
  {
    poisson_solver.add_boundary (MLS_INTERSECTION, phi_free_, NULL, NEUMANN, zero_cf, zero_cf);
    poisson_solver.add_interface(MLS_INTERSECTION, phi_agar_, NULL, zero_cf, zero_cf);
  }
  else
  {
    poisson_solver.add_interface(MLS_INTERSECTION, phi_biof_, NULL, zero_cf, zero_cf);
  }

  poisson_solver.set_diag(diag_m, diag_p);
  poisson_solver.set_use_taylor_correction(use_taylor_correction_);
  poisson_solver.set_kink_treatment(1);
  poisson_solver.set_enfornce_diag_scaling(1);

  poisson_solver.set_nonlinear_term(1., *f_cf_, *fc_cf_,
                                    0., zero_f, zero_f);
  poisson_solver.set_solve_nonlinear_parameters(1, max_iterations_, tolerance_, 0);
  poisson_solver.solve_nonlinear(C_, true, true);

  // update fields with new values
  Vec tmp;
  tmp = Ca0_; Ca0_ = Ca1_; Ca1_ = tmp;
  tmp = Cb0_; Cb0_ = Cb1_; Cb1_ = tmp;
  tmp = Cf0_; Cf0_ = Cf1_; Cf1_ = tmp;

  ierr = VecGetArray(C_,   &C_ptr);   CHKERRXX(ierr);
  ierr = VecGetArray(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cf0_, &Cf0_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Ca1_, &Ca1_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cb1_, &Cb1_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cf1_, &Cf1_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    Cb0_ptr[n] = phi_biof_ptr[n] < 0 ? C_ptr[n] : Cb1_ptr[n];
    if (Da_ != 0) Ca0_ptr[n] = phi_agar_ptr[n] > 0 ? C_ptr[n] : Ca1_ptr[n];
    if (Df_ != 0) Cf0_ptr[n] = phi_free_ptr[n] > 0 ? C_ptr[n] : Cf1_ptr[n];
  }

  ierr = VecRestoreArray(C_,   &C_ptr);   CHKERRXX(ierr);
  ierr = VecRestoreArray(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cf0_, &Cf0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Ca1_, &Ca1_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cb1_, &Cb1_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cf1_, &Cf1_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  // extend fields
  VecScaleGhost(phi_agar_, -1.);
  VecScaleGhost(phi_free_, -1.);

  my_p4est_level_set_t ls(ngbd_);
  ls.extend_Over_Interface_TVD_Full(phi_biof_, Cb0_, extend_iterations_, 2);
  ls.extend_Over_Interface_TVD_Full(phi_agar_, Ca0_, extend_iterations_, 2);
  ls.extend_Over_Interface_TVD_Full(phi_free_, Cf0_, extend_iterations_, 2);

  VecScaleGhost(phi_agar_, -1.);
  VecScaleGhost(phi_free_, -1.);

  ierr = VecDestroy(mu_m);      CHKERRXX(ierr);
  ierr = VecDestroy(mu_p);      CHKERRXX(ierr);
  ierr = VecDestroy(diag_m);    CHKERRXX(ierr);
  ierr = VecDestroy(diag_p);    CHKERRXX(ierr);
  ierr = VecDestroy(rhs_tmp_m); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_tmp_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_solve_concentration, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::solve_pressure()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_solve_pressure, 0, 0, 0, 0); CHKERRXX(ierr);

  // compute rhs
  Vec rhs;

  ierr = VecDuplicate(phi_free_, &rhs); CHKERRXX(ierr);

  double *rhs_ptr;
  double *Cb0_ptr;

  ierr = VecGetArray(rhs,  &rhs_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_) { rhs_ptr[n] = gam_*(*f_cf_)(Cb0_ptr[n]); }

  ierr = VecRestoreArray(rhs,  &rhs_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);

//  double *rhs_ptr;
//  double *Cb0_ptr;
//  double *kappa_ptr;

//  ierr = VecGetArray(rhs,  &rhs_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

//  quad_neighbor_nodes_of_node_t qnnn;
//  foreach_local_node(n, nodes_)
//  {
//    qnnn = ngbd_->get_neighbors(n);
//    double kappa_xx = 0;
//    double kappa_yy = 0;
//    qnnn.laplace(kappa_ptr, kappa_xx, kappa_yy);
//    rhs_ptr[n] = gam_*(*f_cf_)(Cb0_ptr[n]) + lambda_*sigma_*(kappa_xx + kappa_yy);
//  }

//  ierr = VecRestoreArray(rhs,  &rhs_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // compute boundary conditions on free surface
  Vec bc_pressure_vec;

  ierr = VecDuplicate(phi_free_, &bc_pressure_vec); CHKERRXX(ierr);

  double *bc_pressure_vec_ptr;
  double *kappa_ptr;

  ierr = VecGetArray(bc_pressure_vec, &bc_pressure_vec_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_) { bc_pressure_vec_ptr[n] = sigma_*kappa_ptr[n]; }

  ierr = VecRestoreArray(bc_pressure_vec, &bc_pressure_vec_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

//  // assemble geometry
//  std::vector<Vec>      phi_array(2);
//  std::vector<action_t> action(2, INTERSECTION);
//  std::vector<int>      color(2);

//  phi_array[0] = phi_agar_; color[0] = 0;
//  phi_array[1] = phi_free_; color[1] = 1;

  // assemble bc
  my_p4est_interpolation_nodes_t bc_pressure(ngbd_);
  bc_pressure.set_input(bc_pressure_vec, linear);

//  std::vector<BoundaryConditionType> bc_interface_type(2);
//#ifdef P4_TO_P8
//  std::vector< CF_3 *> bc_interface_value(2);
//  std::vector< CF_3 *> bc_interface_coeff(2);
//#else
//  std::vector< CF_2 *> bc_interface_value(2);
//  std::vector< CF_2 *> bc_interface_coeff(2);
//#endif

//  bc_interface_type[0] = NEUMANN;   bc_interface_value[0] = &zero_cf;     bc_interface_coeff[0] = &zero_cf;
//  bc_interface_type[1] = DIRICHLET; bc_interface_value[1] = &bc_pressure; bc_interface_coeff[1] = &zero_cf;

  // initialize poisson solver
  my_p4est_poisson_nodes_mls_t solver(ngbd_);

  solver.set_use_sc_scheme(use_sc_scheme_);
  solver.set_integration_order(integration_order_);

  solver.add_boundary(MLS_INTERSECTION, phi_agar_, NULL, NEUMANN, zero_cf, zero_cf);
  solver.add_boundary(MLS_INTERSECTION, phi_free_, NULL, DIRICHLET, bc_pressure, zero_cf);

  solver.set_mu(lambda_);
  solver.set_rhs(rhs);

  solver.set_wc(*bc_wall_type_, zero_cf);

  solver.set_diag(0.0);

  solver.set_use_taylor_correction(1);
  solver.set_kink_treatment(1);
  solver.set_enfornce_diag_scaling(1);

  solver.solve(P_);

  my_p4est_level_set_t ls(ngbd_);
  ls.extend_Over_Interface_TVD_Full(phi_biof_, P_, extend_iterations_, 2);

//  double *P_ptr;

//  ierr = VecGetArray(P_,  &P_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

//  foreach_node(n, nodes_)
//  {
//    P_ptr[n] -= sigma_*kappa_ptr[n];
//  }

//  ierr = VecRestoreArray(P_,  &P_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(kappa_, &kappa_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(bc_pressure_vec); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_solve_pressure, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::compute_velocity_from_concentration()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vector velocity */
  Vec v0_tmp[P4EST_DIM]; double *v0_ptr[P4EST_DIM];

  foreach_dimension(dim)
  {
    ierr = VecDuplicate(normal_[dim], &v0_tmp[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(v0_tmp[dim], &v0_ptr[dim]);   CHKERRXX(ierr);
  }

  double *Cb0_ptr;

  ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);

  double factor = gam_*Db_/rho_;

  // compute values at layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);

    v0_ptr[0][n] = factor*qnnn.dx_central(Cb0_ptr);
    v0_ptr[1][n] = factor*qnnn.dy_central(Cb0_ptr);
#ifdef P4_TO_P8
    v0_ptr[2][n] = factor*qnnn.dz_central(Cb0_ptr);
#endif
  }

  // start communicating
  foreach_dimension(dim)
  {
    ierr = VecGhostUpdateBegin(v0_tmp[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // meanwhile compute values at strickly local nodes
  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_local_node(i);
    qnnn = ngbd_->get_neighbors(n);

    v0_ptr[0][n] = factor*qnnn.dx_central(Cb0_ptr);
    v0_ptr[1][n] = factor*qnnn.dy_central(Cb0_ptr);
#ifdef P4_TO_P8
    v0_ptr[2][n] = factor*qnnn.dz_central(Cb0_ptr);
#endif
  }

  // stop communicating
  foreach_dimension(dim)
  {
    ierr = VecGhostUpdateEnd(v0_tmp[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);

  foreach_dimension(dim)
  {
    ierr = VecRestoreArray(v0_tmp[dim], &v0_ptr[dim]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  foreach_dimension(dim)
  {
    ls.extend_from_interface_to_whole_domain_TVD(phi_free_, v0_tmp[dim], v0_[dim]);
    ierr = VecDestroy(v0_tmp[dim]); CHKERRXX(ierr);
  }


  /* compute normal velocity */
  Vec vn_tmp; double *vn_ptr;

  ierr = VecDuplicate(phi_free_, &vn_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(vn_tmp, &vn_ptr);     CHKERRXX(ierr);

  double *normal_ptr[P4EST_DIM];
  foreach_dimension(dim)
  {
    ierr = VecGetArray(normal_[dim], &normal_ptr[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(v0_[dim], &v0_ptr[dim]);         CHKERRXX(ierr);
  }

  foreach_node(n, nodes_)
  {
#ifdef P4_TO_P8
    vn_ptr[n] = v0_ptr[0][n]*normal_ptr[0][n] + v0_ptr[1][n]*normal_ptr[1][n] + v0_ptr[2][n]*normal_ptr[2][n];
#else
    vn_ptr[n] = v0_ptr[0][n]*normal_ptr[0][n] + v0_ptr[1][n]*normal_ptr[1][n];
#endif
  }

  foreach_dimension(dim)
  {
    ierr = VecRestoreArray(normal_[dim], &normal_ptr[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(v0_[dim], &v0_ptr[dim]);       CHKERRXX(ierr);
  }

  ls.extend_from_interface_to_whole_domain_TVD(phi_free_, vn_tmp, vn_);

  ierr = VecDestroy(vn_tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::compute_velocity_from_pressure()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vector velocity */
  Vec v0_tmp[P4EST_DIM]; double *v0_ptr[P4EST_DIM];

  foreach_dimension(dim)
  {
    ierr = VecDuplicate(normal_[dim], &v0_tmp[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(v0_tmp[dim], &v0_ptr[dim]);   CHKERRXX(ierr);
  }

  double *P_ptr;

  ierr = VecGetArray(P_, &P_ptr); CHKERRXX(ierr);

  // compute values at layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_layer_node(i);
    qnnn = ngbd_->get_neighbors(n);

    v0_ptr[0][n] = -lambda_*qnnn.dx_central(P_ptr);
    v0_ptr[1][n] = -lambda_*qnnn.dy_central(P_ptr);
#ifdef P4_TO_P8
    v0_ptr[2][n] = -lambda_*qnnn.dz_central(P_ptr);
#endif
  }

  // start communicating
  foreach_dimension(dim)
  {
    ierr = VecGhostUpdateBegin(v0_tmp[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // meanwhile compute values at strickly local nodes
  for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_->get_local_node(i);
    qnnn = ngbd_->get_neighbors(n);

    v0_ptr[0][n] = -lambda_*qnnn.dx_central(P_ptr);
    v0_ptr[1][n] = -lambda_*qnnn.dy_central(P_ptr);
#ifdef P4_TO_P8
    v0_ptr[2][n] = -lambda_*qnnn.dz_central(P_ptr);
#endif
  }

  // stop communicating
  foreach_dimension(dim)
  {
    ierr = VecGhostUpdateEnd(v0_tmp[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(P_, &P_ptr); CHKERRXX(ierr);

  foreach_dimension(dim)
  {
    ierr = VecRestoreArray(v0_tmp[dim], &v0_ptr[dim]); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  foreach_dimension(dim)
  {
    ls.extend_from_interface_to_whole_domain_TVD(phi_free_, v0_tmp[dim], v0_[dim]);
    ierr = VecDestroy(v0_tmp[dim]); CHKERRXX(ierr);
  }

  /* compute normal velocity */
  Vec vn_tmp; double *vn_ptr;

  ierr = VecDuplicate(phi_free_, &vn_tmp); CHKERRXX(ierr);
  ierr = VecGetArray(vn_tmp, &vn_ptr);     CHKERRXX(ierr);

  double *normal_ptr[P4EST_DIM];
  foreach_dimension(dim)
  {
    ierr = VecGetArray(normal_[dim], &normal_ptr[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(v0_[dim], &v0_ptr[dim]);         CHKERRXX(ierr);
  }

  foreach_node(n, nodes_)
  {
#ifdef P4_TO_P8
    vn_ptr[n] = v0_ptr[0][n]*normal_ptr[0][n] + v0_ptr[1][n]*normal_ptr[1][n] + v0_ptr[2][n]*normal_ptr[2][n];
#else
    vn_ptr[n] = v0_ptr[0][n]*normal_ptr[0][n] + v0_ptr[1][n]*normal_ptr[1][n];
#endif
  }

  foreach_dimension(dim)
  {
    ierr = VecRestoreArray(normal_[dim], &normal_ptr[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(v0_[dim], &v0_ptr[dim]);       CHKERRXX(ierr);
  }

//  my_p4est_interpolation_nodes_t phi_interp(ngbd_);
//  my_p4est_interpolation_nodes_t pre_interp(ngbd_);

//  phi_interp.set_input(phi_free_, linear);
//  pre_interp.set_input(P_, linear);

//  double phi[3][3];
//  double pre[3][3];
//  double x[P4EST_DIM];

//  double *phi_free_ptr;

//  ierr = VecGetArray(vn_tmp, &vn_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

//  foreach_local_node (n, nodes_)
//  {
//    if (fabs(phi_free_ptr[n]) < 15.*diag_)
//    {
//      node_xyz_fr_n(n, p4est_, nodes_, x);
//      for (short i = 0; i < 3; i++)
//        for (short j = 0; j < 3; j++)
//        {
//          pre[i][j] = pre_interp(x[0]+(i-1)*dxyz_[0],x[1]+(j-1)*dxyz_[1]);
//          phi[i][j] = phi_interp(x[0]+(i-1)*dxyz_[0],x[1]+(j-1)*dxyz_[1]);
//        }

//      const int i = 1, j = 1;

//      double phix  = (phi[i+1][j]-phi[i-1][j])/(2*dxyz_[0]);
//      double phiy  = (phi[i][j+1]-phi[i][j-1])/(2*dxyz_[1]);

//      double phin  = MAX(sqrt(phix*phix+phiy*phiy), EPS);

//      double nx = phix/phin;
//      double ny = phiy/phin;

//      double phi1  = (phi[i+1][j+1]-phi[i-1][j-1])/(2*diag_);
//      double phi2  = (phi[i-1][j+1]-phi[i+1][j-1])/(2*diag_);

//      phin         = MAX(sqrt(phi1*phi1+phi2*phi2), EPS);

//      double tx = phi1/phin;
//      double ty = phi2/phin;

//      double prex  = (pre[i+1][j]-pre[i-1][j])/(2*dxyz_[0]);
//      double prey  = (pre[i][j+1]-pre[i][j-1])/(2*dxyz_[1]);

//      double pre1  = (pre[i+1][j+1]-pre[i-1][j-1])/(2*diag_);
//      double pre2  = (pre[i-1][j+1]-pre[i+1][j-1])/(2*diag_);

//      vn_ptr[n] = .5*lambda_*(prex*nx + prey*ny + pre1*tx + pre2*ty);

//    } else {
//      vn_ptr[n] = 0;
//    }
//  }

//  ierr = VecRestoreArray(vn_tmp, &vn_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(vn_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(vn_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi_free_, vn_tmp, vn_);

  ierr = VecDestroy(vn_tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::compute_dt()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);

  const double *phi_free_ptr;
  const double *kappa_ptr;
  const double *vn_ptr;

  kappa_max_ = 0;
  vn_max_ = 0;

  double kvn_max = 0;

  ierr = VecGetArrayRead(phi_free_, &phi_free_ptr); CHKERRXX(ierr);
  ierr = VecGetArrayRead(kappa_, &kappa_ptr);       CHKERRXX(ierr);
  ierr = VecGetArrayRead(vn_, &vn_ptr);             CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    if(fabs(phi_free_ptr[n]) < dxyz_close_interface_)
    {
      vn_max_ = MAX(vn_max_, fabs(vn_ptr[n]));
      kappa_max_ = MAX(kappa_max_, fabs(kappa_ptr[n]));
      kvn_max = MAX(kvn_max, fabs(vn_ptr[n]*kappa_ptr[n]));
    }
  }

  ierr = VecRestoreArrayRead(phi_free_, &phi_free_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_, &kappa_ptr);       CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vn_, &vn_ptr);             CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &vn_max_,    1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &kappa_max_, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &kvn_max,    1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  dt1_ = dt0_;
  dt0_ = MIN(dt_max_, cfl_number_ * dxyz_min_/MAX(1.e-10, vn_max_), 1./MAX(1.e-10, kvn_max));

  PetscPrintf(p4est_->mpicomm, "vn_max = %e, kappa_max = %e, kvn_max = %e, dt = %e\n", vn_max_, kappa_max_, kvn_max, dt0_);

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::update_grid()
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid...\n");

  // advect interface and update p4est
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_, ngbd_);

  sl.set_phi_interpolation (quadratic_non_oscillatory_continuous_v2);
  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);

  std::vector<Vec>       phi_array(2, NULL);
  std::vector<mls_opn_t> action(2, MLS_INTERSECTION);

  phi_array[0] = phi_free_;
  phi_array[1] = phi_agar_;

  if (use_godunov_scheme_)
  {
    my_p4est_level_set_t ls_adv(ngbd_);
    dt0_ = ls_adv.advect_in_normal_direction(vn_, phi_free_, dt0_);
    sl.update_p4est(v0_, 0, phi_array, action, 0);
  }
  else if (advection_scheme_ == 1 || first_iteration_)
  {
    sl.update_p4est(v0_, dt0_, phi_array, action, 0);
  }
  else if (advection_scheme_ == 2)
  {
    sl.update_p4est(v1_, v0_, dt1_, dt0_, phi_array, action, 0);
  }

  phi_free_ = phi_array[0];
  phi_agar_ = phi_array[1];

  /* interpolate the quantities onto the new grid */
  PetscPrintf(p4est_->mpicomm, "Transfering data between grids...\n");

  ierr = VecDestroy(kappa_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &kappa_); CHKERRXX(ierr);

  ierr = VecDestroy(phi_biof_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &phi_biof_); CHKERRXX(ierr);

  foreach_dimension(dim)
  {
    ierr = VecDestroy(normal_[dim]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &normal_[dim]); CHKERRXX(ierr);

    ierr = VecDestroy(phi_biof_dd_[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(normal_[dim], &phi_biof_dd_[dim]); CHKERRXX(ierr);
  }

  ierr = VecDestroy(P_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &P_); CHKERRXX(ierr);

  ierr = VecDestroy(vn_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_free_, &vn_); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  /* concentration */
  Vec Ca0_tmp;
  ierr = VecDuplicate(phi_free_, &Ca0_tmp); CHKERRXX(ierr);
  interp.set_input(Ca0_, interpolation_between_grids_);
  interp.interpolate(Ca0_tmp);
  ierr = VecDestroy(Ca0_); CHKERRXX(ierr);
  Ca0_ = Ca0_tmp;

  Vec Cb0_tmp;
  ierr = VecDuplicate(phi_free_, &Cb0_tmp); CHKERRXX(ierr);
  interp.set_input(Cb0_, interpolation_between_grids_);
  interp.interpolate(Cb0_tmp);
  ierr = VecDestroy(Cb0_); CHKERRXX(ierr);
  Cb0_ = Cb0_tmp;

  Vec Cf0_tmp;
  ierr = VecDuplicate(phi_free_, &Cf0_tmp); CHKERRXX(ierr);
  interp.set_input(Cf0_, interpolation_between_grids_);
  interp.interpolate(Cf0_tmp);
  ierr = VecDestroy(Cf0_); CHKERRXX(ierr);
  Cf0_ = Cf0_tmp;

  Vec Ca1_tmp;
  ierr = VecDuplicate(phi_free_, &Ca1_tmp); CHKERRXX(ierr);
  interp.set_input(Ca1_, interpolation_between_grids_);
  interp.interpolate(Ca1_tmp);
  ierr = VecDestroy(Ca1_); CHKERRXX(ierr);
  Ca1_ = Ca1_tmp;

  Vec Cb1_tmp;
  ierr = VecDuplicate(phi_free_, &Cb1_tmp); CHKERRXX(ierr);
  interp.set_input(Cb1_, interpolation_between_grids_);
  interp.interpolate(Cb1_tmp);
  ierr = VecDestroy(Cb1_); CHKERRXX(ierr);
  Cb1_ = Cb1_tmp;

  Vec Cf1_tmp;
  ierr = VecDuplicate(phi_free_, &Cf1_tmp); CHKERRXX(ierr);
  interp.set_input(Cf1_, interpolation_between_grids_);
  interp.interpolate(Cf1_tmp);
  ierr = VecDestroy(Cf1_); CHKERRXX(ierr);
  Cf1_ = Cf1_tmp;

  Vec C_tmp;
  ierr = VecDuplicate(phi_free_, &C_tmp); CHKERRXX(ierr);
  interp.set_input(C_, interpolation_between_grids_);
  interp.interpolate(C_tmp);
  ierr = VecDestroy(C_); CHKERRXX(ierr);
  C_ = C_tmp;

  foreach_dimension(dim)
  {
    ierr = VecDestroy(v1_[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(normal_[dim], &v1_[dim]); CHKERRXX(ierr);
    interp.set_input(v0_[dim], interpolation_between_grids_);
    interp.interpolate(v1_[dim]);
    ierr = VecDestroy(v0_[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(normal_[dim], &v0_[dim]); CHKERRXX(ierr);
  }

  p4est_destroy(p4est_);       p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  /* "solidify" too narrow regions */
  if (1)
  {
    invert_phi(nodes_, phi_free_);
    PetscPrintf(p4est_->mpicomm, "Removing problem geometries...\n");

    Vec     phi_tmp;
    double *phi_tmp_ptr;
    double *phi_ptr;

    p4est_locidx_t nei_n[num_neighbors_cube];
    bool           nei_e[num_neighbors_cube];

    double band = diag_;

    ierr = VecDuplicate(phi_free_, &phi_tmp); CHKERRXX(ierr);

    ierr = VecGetArray(phi_tmp, &phi_tmp_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_free_,    &phi_ptr);     CHKERRXX(ierr);

    // first pass: smooth out extremely curved regions
    bool is_changed = false;
    foreach_local_node(n, nodes_)
    {
      if (fabs(phi_ptr[n]) < band)
      {
        ngbd_->get_all_neighbors(n, nei_n, nei_e);

        unsigned short num_neg = 0;
        unsigned short num_pos = 0;

        for (unsigned short nn = 0; nn < num_neighbors_cube; ++nn)
        {
          phi_ptr[nei_n[nn]] < 0 ? num_neg++ : num_pos++;
        }

        if ( (phi_ptr[n] <  0 && num_neg < 3) ||
             (phi_ptr[n] >= 0 && num_pos < 3) )
        {
          phi_ptr[n] = phi_ptr[n] <  0 ? EPS : -EPS;

          // check if node is a layer node (= a ghost node for another process)
          p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);
          if (ni->pad8 != 0) is_changed = true;
        }
      }
    }

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    if (is_changed)
    {
      ierr = VecGhostUpdateBegin(phi_free_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (phi_free_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    copy_ghosted_vec(phi_free_, phi_tmp);

    // second pass: bridge narrow gaps
    is_changed = false;
    foreach_local_node(n, nodes_)
    {
      if (phi_ptr[n] < 0 && phi_ptr[n] > -band)
      {
        ngbd_->get_all_neighbors(n, nei_n, nei_e);

        bool merge = (phi_ptr[nei_n[nn_m00]] > 0 && phi_ptr[nei_n[nn_p00]] > 0 && phi_ptr[nei_n[nn_0m0]] > 0 && phi_ptr[nei_n[nn_0p0]] > 0)
            || ((phi_ptr[nei_n[nn_m00]] > 0 && phi_ptr[nei_n[nn_p00]] > 0) &&
                (phi_ptr[nei_n[nn_mm0]] < 0 || phi_ptr[nei_n[nn_0m0]] < 0 || phi_ptr[nei_n[nn_pm0]] < 0) &&
                (phi_ptr[nei_n[nn_mp0]] < 0 || phi_ptr[nei_n[nn_0p0]] < 0 || phi_ptr[nei_n[nn_pp0]] < 0))
            || ((phi_ptr[nei_n[nn_0m0]] > 0 && phi_ptr[nei_n[nn_0p0]] > 0) &&
                (phi_ptr[nei_n[nn_mm0]] < 0 || phi_ptr[nei_n[nn_m00]] < 0 || phi_ptr[nei_n[nn_mp0]] < 0) &&
                (phi_ptr[nei_n[nn_pm0]] < 0 || phi_ptr[nei_n[nn_p00]] < 0 || phi_ptr[nei_n[nn_pp0]] < 0));

        if (merge)
        {
          phi_tmp_ptr[n] = .5*dxyz_min_;
          is_changed = true;
        }

      }
    }

    ierr = VecRestoreArray(phi_tmp, &phi_tmp_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_free_, &phi_ptr); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    if (is_changed)
    {
      ierr = VecGhostUpdateBegin(phi_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (phi_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    int area_thresh_ = 15;
    // third pass: look for isolated pools of liquid and remove them
    if (is_changed) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
    {
      int num_islands = 0;

      Vec     island_number;
      double *island_number_ptr;

      ierr = VecDuplicate(phi_free_, &island_number); CHKERRXX(ierr);

      invert_phi(nodes_, phi_tmp);
      compute_islands_numbers(*ngbd_, phi_tmp, num_islands, island_number);
      invert_phi(nodes_, phi_tmp);

      if (num_islands > 1)
      {
        std::vector<int> areas(num_islands, 0);

        ierr = VecGetArray(phi_tmp, &phi_tmp_ptr); CHKERRXX(ierr);
        ierr = VecGetArray(island_number, &island_number_ptr); CHKERRXX(ierr);

        foreach_local_node(n, nodes_)
        {
          if(island_number_ptr[n] >= 0)
          {
            areas[ (unsigned int) island_number_ptr[n]] ++;
          }
        }

        int mpiret = MPI_Allreduce(MPI_IN_PLACE, areas.data(), num_islands, MPI_INT, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

        foreach_node(n, nodes_)
        {
          if(island_number_ptr[n] >= 0)
          {
            if (areas[ (unsigned int) island_number_ptr[n]] < area_thresh_)
              phi_tmp_ptr[n] = .5*dxyz_min_;
          }
        }

        ierr = VecRestoreArray(phi_tmp, &phi_tmp_ptr); CHKERRXX(ierr);
        ierr = VecRestoreArray(island_number, &island_number_ptr); CHKERRXX(ierr);
      }

      ierr = VecDestroy(island_number); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi_free_); CHKERRXX(ierr);
    phi_free_ = phi_tmp;
    invert_phi(nodes_, phi_free_);
  }

  /* reinitialize phi_free_ */
  my_p4est_level_set_t ls_new(ngbd_);
  ls_new.reinitialize_1st_order_time_2nd_order_space(phi_free_);

  /* second derivatives, normals, curvature, angles */
  compute_geometric_properties();

  first_iteration_ = false;

  PetscPrintf(p4est_->mpicomm, "Done \n");

  ierr = PetscLogEventEnd(log_my_p4est_biofilm_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::save_VTK(int iter)
{
  ierr = PetscLogEventBegin(log_my_p4est_biofilm_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);

  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name[1000];

#ifdef P4_TO_P8
  sprintf(name, "%s/vtu/biofilm_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], brick_->nxyztrees[2], iter);
#else
  sprintf(name, "%s/vtu/biofilm_lvl_%d_%d_%d_%dx%d.%05d", out_dir, data->min_lvl, data->max_lvl, p4est_->mpisize, brick_->nxyztrees[0], brick_->nxyztrees[1], iter);
#endif

  const double *phi_agar_ptr; ierr = VecGetArrayRead(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  const double *phi_biof_ptr; ierr = VecGetArrayRead(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  const double *phi_free_ptr; ierr = VecGetArrayRead(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  const double *Ca0_ptr; ierr = VecGetArrayRead(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  const double *Cb0_ptr; ierr = VecGetArrayRead(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  const double *Cf0_ptr; ierr = VecGetArrayRead(Cf0_, &Cf0_ptr); CHKERRXX(ierr);

  const double *C_ptr; ierr = VecGetArrayRead(C_, &C_ptr); CHKERRXX(ierr);
  const double *P_ptr; ierr = VecGetArrayRead(P_, &P_ptr); CHKERRXX(ierr);

  const double *kappa_ptr;  ierr = VecGetArrayRead(kappa_, &kappa_ptr); CHKERRXX(ierr);

  double *vn_ptr; ierr = VecGetArray(vn_, &vn_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_) { vn_ptr[n] /= scaling_; }

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


  my_p4est_vtk_write_all(p4est_, nodes_, ghost_,
                         P4EST_TRUE, P4EST_TRUE,
                         10, 1, name,
                         VTK_POINT_DATA, "phi", phi_biof_ptr,
                         VTK_POINT_DATA, "phi_free", phi_free_ptr,
                         VTK_POINT_DATA, "phi_agar", phi_agar_ptr,
                         VTK_POINT_DATA, "C", C_ptr,
                         VTK_POINT_DATA, "Ca", Ca0_ptr,
                         VTK_POINT_DATA, "Cb", Cb0_ptr,
                         VTK_POINT_DATA, "Cf", Cf0_ptr,
                         VTK_POINT_DATA, "vn", vn_ptr,
                         VTK_POINT_DATA, "P", P_ptr,
                         VTK_POINT_DATA, "kappa", kappa_ptr,
                         VTK_CELL_DATA , "leaf_level", l_p);


  foreach_node(n, nodes_) { vn_ptr[n] *= scaling_; }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_free_, &phi_free_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Cf0_, &Cf0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(C_, &C_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(P_, &P_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa_, &kappa_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(vn_, &vn_ptr); CHKERRXX(ierr);


  PetscPrintf(p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_biofilm_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_biofilm_t::compute_concentration_global()
{
  PetscPrintf(p4est_->mpicomm, "Constructing C global\n");
  double *C_ptr; ierr = VecGetArray(C_, &C_ptr); CHKERRXX(ierr);
  double *Ca0_ptr; ierr = VecGetArray(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  double *Cb0_ptr; ierr = VecGetArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  double *Cf0_ptr; ierr = VecGetArray(Cf0_, &Cf0_ptr); CHKERRXX(ierr);

  double *phi_biof_ptr; ierr = VecGetArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  double *phi_agar_ptr; ierr = VecGetArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  double *phi_free_ptr; ierr = VecGetArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes_)
  {
    if (phi_biof_ptr[n] < 0)
    {
      C_ptr[n] = Cb0_ptr[n];
    }
    else
    {
      if (phi_agar_ptr[n] > phi_free_ptr[n]) C_ptr[n] = Ca0_ptr[n];
      else C_ptr[n] = Cf0_ptr[n];
    }
  }

  ierr = VecRestoreArray(C_, &C_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Ca0_, &Ca0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cb0_, &Cb0_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Cf0_, &Cf0_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_biof_, &phi_biof_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_agar_, &phi_agar_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_free_, &phi_free_ptr); CHKERRXX(ierr);
}

void my_p4est_biofilm_t::compute_filtered_curvature()
{
  double smoothing = SQR(curvature_smoothing_*diag_);

  my_p4est_interpolation_nodes_t interp(ngbd_);
  interp.set_input(phi_biof_, linear);
  VecCopyGhost(phi_biof_, kappa_);

  my_p4est_poisson_nodes_mls_t solver(ngbd_);

  solver.set_mu(smoothing/double(curvature_smoothing_steps_));
  solver.set_diag(1.);
  solver.set_rhs(kappa_);
//  solver.set_wc(neumann_cf, zero_cf);
  solver.set_wc(dirichlet_cf, interp);

  for (int i = 0; i < curvature_smoothing_steps_; ++i)
  {
    solver.solve(kappa_, true);
  }

  VecAXPBYGhost(kappa_, -1., 1., phi_biof_);
  VecScaleGhost(kappa_, 1./smoothing);

  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(linear);

  ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_biof_, kappa_, phi_biof_);
}
