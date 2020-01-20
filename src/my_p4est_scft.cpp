#include "my_p4est_scft.h"

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/my_p8est_macros.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_macros.h>
#endif

my_p4est_scft_t::my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd, int ns)
  : brick(ngbd->myb), connectivity(ngbd->p4est->connectivity), p4est(ngbd->p4est), ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy), ngbd(ngbd), ns(ns), solver_a(ngbd), solver_b(ngbd)
{
  scalling = 1;

  /* potentials */
  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &mu_p); CHKERRXX(ierr);

  mu_m_avg = 0;
  mu_p_avg = 0;

  /* densities */
  ierr = VecDuplicate(mu_m, &rho_a); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &rho_b); CHKERRXX(ierr);

  /* surface tensions */
  gamma_air = NULL;

  /* partition function and energy */
  Q = 0;
  energy = 0;
  energy_singular_part = 0;

  /* chain propagators */
  qf.resize(ns, NULL);
  qb.resize(ns, NULL);

  for (int i = 0; i < ns; i++)
  {
    ierr = VecDuplicate(mu_m, &qf[i]); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &qb[i]); CHKERRXX(ierr);
  }

  ierr = VecSetGhost(qf[0],    1.0); CHKERRXX(ierr);
  ierr = VecSetGhost(qb[ns-1], 1.0); CHKERRXX(ierr);

  /* chemical forces */
  ierr = VecDuplicate(mu_m, &force_p); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &force_m); CHKERRXX(ierr);

  force_p_avg = 0;
  force_m_avg = 0;
  force_p_max = 0;
  force_m_max = 0;

  /* default polymer */
  set_polymer(0.5, 20.);

  /* solver parameters */
  integration_order   = 1;
  cube_refinement     = 1;
  time_discretization = 1;

  /* auxiliary variables */
  num_surfaces = 0;
  lambda = 2;
  diag = 1;
  dxyz_min = 1;
  dxyz_max = 1;
  dxyz_close_interface = 1;

  volume = 0;

  ierr = VecDuplicate(mu_m, &exp_w_a); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &exp_w_b); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &integrating_vec); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &q_tmp); CHKERRXX(ierr);

  phi_smooth = NULL;
  mask       = NULL;

  /* DSA stuff */
  mu_t = NULL;

  nu_m = NULL;
  nu_p = NULL;

  nu_a = NULL;
  nu_b = NULL;

  psi_a = NULL;
  psi_b = NULL;

  force_nu_m = NULL;
  force_nu_p = NULL;

  /* determine the smallest cell size */
  ::dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  dxyz_min = MIN(dxyz[0], dxyz[1], dxyz[2]);
  dxyz_max = MAX(dxyz[0], dxyz[1], dxyz[2]);
  diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
#else
  dxyz_min = MIN(dxyz[0], dxyz[1]);
  dxyz_max = MAX(dxyz[0], dxyz[1]);
  diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
#endif
  dxyz_close_interface = 1.2*dxyz_max;
}

my_p4est_scft_t::~my_p4est_scft_t()
{
  if (mu_p  != NULL) { ierr = VecDestroy(mu_p);   CHKERRXX(ierr); }
  if (mu_m  != NULL) { ierr = VecDestroy(mu_m);   CHKERRXX(ierr); }
  if (rho_a != NULL) { ierr = VecDestroy(rho_a);  CHKERRXX(ierr); }
  if (rho_b != NULL) { ierr = VecDestroy(rho_b);  CHKERRXX(ierr); }

  if (force_p != NULL) { ierr = VecDestroy(force_p);  CHKERRXX(ierr); }
  if (force_m != NULL) { ierr = VecDestroy(force_m);  CHKERRXX(ierr); }

  if (rhs != NULL) { ierr = VecDestroy(rhs); CHKERRXX(ierr); }
  if (q_tmp != NULL) { ierr = VecDestroy(q_tmp); CHKERRXX(ierr); }
  if (integrating_vec != NULL) { ierr = VecDestroy(integrating_vec); CHKERRXX(ierr); }

  for (int i = 0; i < qf.size(); i++) { ierr = VecDestroy(qf.at(i));   CHKERRXX(ierr); }
  for (int i = 0; i < qb.size(); i++) { ierr = VecDestroy(qb.at(i));   CHKERRXX(ierr); }

  if (exp_w_a != NULL) { ierr = VecDestroy(exp_w_a); CHKERRXX(ierr); }
  if (exp_w_b != NULL) { ierr = VecDestroy(exp_w_b); CHKERRXX(ierr); }

  for (int surf_idx = 0; surf_idx < normal.size(); surf_idx++)
  {
    for (int dim = 0; dim < P4EST_DIM; dim++)
    {
      ierr = VecDestroy(normal[surf_idx][dim]); CHKERRXX(ierr);
    }
    delete[] normal[surf_idx];

    ierr = VecDestroy(kappa[surf_idx]); CHKERRXX(ierr);
  }

  // density optimization
  for (int i = 0; i < zf.size(); i++) { ierr = VecDestroy(zf[i]); CHKERRXX(ierr); }
  for (int i = 0; i < zb.size(); i++) { ierr = VecDestroy(zb[i]); CHKERRXX(ierr); }

  if (nu_m != NULL) { ierr = VecDestroy(nu_m); CHKERRXX(ierr); }
  if (nu_p != NULL) { ierr = VecDestroy(nu_p); CHKERRXX(ierr); }

  if (nu_a != NULL) { ierr = VecDestroy(nu_a); CHKERRXX(ierr); }
  if (nu_b != NULL) { ierr = VecDestroy(nu_b); CHKERRXX(ierr); }

  if (psi_a != NULL) { ierr = VecDestroy(psi_a); CHKERRXX(ierr); }
  if (psi_b != NULL) { ierr = VecDestroy(psi_b); CHKERRXX(ierr); }

  if (force_nu_m != NULL)    { ierr = VecDestroy(force_nu_m); CHKERRXX(ierr); }
  if (force_nu_p != NULL)    { ierr = VecDestroy(force_nu_p); CHKERRXX(ierr); }
}

void my_p4est_scft_t::set_polymer(double f, double XN)
{
  this->f   = f;
  this->XN  = XN;

  /* Discretization along the chain
   * ns       - total beads
   * ns-1     - total intervals
   * fns      - beads of type A
   * ns-fns+1 - beads of type B
   * fns-1    - intervals of A
   * ns-fns   - intervals of B
   */

  fns = round(double(ns) * f);

  ds_a = f      / double(fns - 1);
  ds_b = (1.-f) / double(ns - fns);

  ns_a = fns;
  ns_b = ns-fns+1;
}

void my_p4est_scft_t::add_boundary(Vec phi, mls_opn_t acn, CF_DIM &surf_energy_A, CF_DIM &surf_energy_B)
{
  this->phi.push_back(phi);
  this->action.push_back(acn);
  this->color.push_back(num_surfaces);
  this->gamma_a.push_back(&surf_energy_A);
  this->gamma_b.push_back(&surf_energy_B);
  num_surfaces++;
}

void my_p4est_scft_t::initialize_solvers()
{
//  for (int i = 0; i < bc_coeffs_a.size(); i++) { ierr = VecDestroy(bc_coeffs_a[i]); CHKERRXX(ierr); }
//  for (int i = 0; i < bc_coeffs_b.size(); i++) { ierr = VecDestroy(bc_coeffs_b[i]); CHKERRXX(ierr); }

//  bc_coeffs_a.resize(num_surfaces, NULL);
//  bc_coeffs_b.resize(num_surfaces, NULL);

//  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_a[i]); CHKERRXX(ierr); }
//  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_b[i]); CHKERRXX(ierr); }

//  for (short i = 0; i < bc_coeffs_a_cf.size(); ++i) { delete bc_coeffs_a_cf[i]; }
//  for (short i = 0; i < bc_coeffs_b_cf.size(); ++i) { delete bc_coeffs_b_cf[i]; }

//  bc_coeffs_a_cf.resize(num_surfaces, NULL);
//  bc_coeffs_b_cf.resize(num_surfaces, NULL);

//  for (short idx_surf = 0; idx_surf < num_surfaces; ++idx_surf)
//  {
//    bc_coeffs_a_cf[idx_surf] = new my_p4est_interpolation_nodes_t(ngbd);
//    bc_coeffs_b_cf[idx_surf] = new my_p4est_interpolation_nodes_t(ngbd);

//    ((my_p4est_interpolation_nodes_t *) bc_coeffs_a_cf[idx_surf])->set_input(bc_coeffs_a[idx_surf], linear);
//    ((my_p4est_interpolation_nodes_t *) bc_coeffs_b_cf[idx_surf])->set_input(bc_coeffs_b[idx_surf], linear);
//  }

  // chain propogator a
  for (int i = 0; i < num_surfaces; ++i)
  {
    solver_a.add_boundary(action[i], phi[i], NULL, ROBIN, zero_cf, zero_cf);
  }

  solver_a.set_mu(scalling*scalling);
  solver_a.set_diag(1./ds_a);
  solver_a.set_wc(bc_wall_type, zero_cf);
  solver_a.set_use_taylor_correction(true);
  solver_a.set_kink_treatment(true);
  solver_a.set_use_sc_scheme(false);
  solver_a.set_store_finite_volumes(true);
  solver_a.preassemble_linear_system();

  // chain propogator b
  for (int i = 0; i < num_surfaces; ++i)
  {
    solver_b.add_boundary(action[i], phi[i], NULL, ROBIN, zero_cf, zero_cf);
  }

  solver_b.set_mu(scalling*scalling);
  solver_b.set_diag(1./ds_b);
  solver_b.set_wc(bc_wall_type, zero_cf);
  solver_b.set_use_taylor_correction(true);
  solver_b.set_kink_treatment(true);
  solver_b.set_use_sc_scheme(false);
  solver_b.set_store_finite_volumes(true);
  solver_b.preassemble_linear_system();

  mask       = solver_a.get_mask();
  phi_smooth = solver_a.get_boundary_phi_eff();

  assemble_integrating_vec();

  Vec ones;
  ierr = VecDuplicate(mu_m, &ones); CHKERRXX(ierr);
  ierr = VecSetGhost(ones, 1.); CHKERRXX(ierr);

  assemble_integrating_vec();
  volume = integrate_over_domain_fast(ones);
  ierr = VecDestroy(ones); CHKERRXX(ierr);

  pw_bc_values.resize(num_surfaces);
  pw_bc_coeffs_a.resize(num_surfaces);
  pw_bc_coeffs_b.resize(num_surfaces);

  for (int i = 0; i < num_surfaces; ++i)
  {
    pw_bc_values[i].resize(solver_a.pw_bc_num_value_pts(i), 0.);
    pw_bc_coeffs_a[i].resize(solver_a.pw_bc_num_robin_pts(i), 0.);
    pw_bc_coeffs_b[i].resize(solver_a.pw_bc_num_robin_pts(i), 0.);

    solver_a.set_bc(i, ROBIN, pw_bc_values[i], pw_bc_values[i], pw_bc_coeffs_a[i]);
    solver_b.set_bc(i, ROBIN, pw_bc_values[i], pw_bc_values[i], pw_bc_coeffs_b[i]);
  }

  ierr = PetscPrintf(p4est->mpicomm, "new volume %e\n", volume); CHKERRXX(ierr);
}

void my_p4est_scft_t::initialize_bc_simple()
{
  if (num_surfaces > 0)
  {
    double xyz[P4EST_DIM];
    energy_singular_part = 0;

    for (int i = 0; i < num_surfaces; ++i) // loop through all surfaces
    {
      for (int j = 0; j < solver_a.pw_bc_num_robin_pts(i); ++j)
      {
        // compute robin coefficients
        solver_a.pw_bc_xyz_robin_pt(i, j, xyz);

        pw_bc_coeffs_a[i][j] = gamma_a[i]->value(xyz)*scalling;
        pw_bc_coeffs_b[i][j] = gamma_b[i]->value(xyz)*scalling;
      }
    }

    // let solvers know that bc's have been updated
    solver_a.set_new_submat_robin(true);
    solver_b.set_new_submat_robin(true);
  }
}

void my_p4est_scft_t::initialize_bc_smart(bool adaptive)
{
  if (num_surfaces > 0)
  {
    // create interpolation of mu_m
    CF_DIM *mu_cf = &zero_cf;

    my_p4est_interpolation_nodes_t interp(ngbd);
    if (adaptive)
    {
      ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      if (phi_smooth != NULL)
      {
        my_p4est_level_set_t ls(ngbd);
        ls.extend_Over_Interface_TVD_Full(phi_smooth, mu_m, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
      }

      interp.set_input(mu_m, linear);
      mu_cf = &interp;
    }

    double xyz[P4EST_DIM];
    energy_singular_part = 0;

    for (int i = 0; i < num_surfaces; ++i) // loop through all surfaces
    {
      boundary_conditions_t *bc = solver_a.get_bc(i);
      for (int j = 0; j < solver_a.pw_bc_num_robin_pts(i); ++j)
      {
        // compute robin coefficients
        solver_a.pw_bc_xyz_robin_pt(i, j, xyz);
        double mu_m_val = mu_cf->value(xyz);
        pw_bc_coeffs_a[i][j] = (gamma_a[i]->value(xyz)-gamma_b[i]->value(xyz))*(-mu_m_val/XN+0.5)*scalling;
        pw_bc_coeffs_b[i][j] = (gamma_a[i]->value(xyz)-gamma_b[i]->value(xyz))*(-mu_m_val/XN-0.5)*scalling;

        // calculate addition to energy from surface tensions
        solver_a.pw_bc_xyz_value_pt(i, j, xyz);
        double pw_integrand = 0.5*(gamma_a[i]->value(xyz)+gamma_b[i]->value(xyz))
                              +   (gamma_a[i]->value(xyz)-gamma_b[i]->value(xyz))*mu_cf->value(xyz)/XN;
        energy_singular_part += bc->areas[j]*pw_integrand;
      }
    }

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &energy_singular_part, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

//    energy_singular_part *= scalling;
//    energy_singular_part /= volume;
    energy_singular_part /= pow(scalling, P4EST_DIM - 1.);

    solver_a.set_new_submat_robin(true);
    solver_b.set_new_submat_robin(true);
  }
}

void my_p4est_scft_t::solve_for_propogators()
{
  // create exp_w vectors
  double *exp_w_a_ptr;
  double *exp_w_b_ptr;
  double *mu_p_ptr;
  double *mu_m_ptr;
  double *mask_ptr;

//  ierr = VecGetArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(mu_p,    &mu_p_ptr);    CHKERRXX(ierr);
//  ierr = VecGetArray(mu_m,    &mu_m_ptr);    CHKERRXX(ierr);
//  ierr = VecGetArray(mask,    &mask_ptr);    CHKERRXX(ierr);

//  for(p4est_locidx_t n=0; n < nodes->num_owned_indeps; ++n)
//  {
//    if (mask_ptr[n] < 0.)
//    {
//      exp_w_a_ptr[n] = exp(-0.5*(mu_p_ptr[n]-mu_m_ptr[n])*ds_a);
//      exp_w_b_ptr[n] = exp(-0.5*(mu_p_ptr[n]+mu_m_ptr[n])*ds_b);
//    } else {
//      exp_w_a_ptr[n] = 1.0;
//      exp_w_b_ptr[n] = 1.0;
//    }
//  }

//  ierr = VecRestoreArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_p,    &mu_p_ptr);    CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_m,    &mu_m_ptr);    CHKERRXX(ierr);
//  ierr = VecRestoreArray(mask,    &mask_ptr);    CHKERRXX(ierr);

//  // forward propagator
//  for (int is = 1; is < fns; is++)
//  {
//    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a); CHKERRXX(ierr);

//    diffusion_step(solver_a, ds_a, qf[is], q_tmp);

//    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_a); CHKERRXX(ierr);
//  }

//  for (int is = fns; is < ns; is++)
//  {
//    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b); CHKERRXX(ierr);

//    diffusion_step(solver_b, ds_b, qf[is], q_tmp);

//    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_b); CHKERRXX(ierr);
//  }

//  // backward propagator
//  for (int is = ns-2; is > fns-2; is--)
//  {
//    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b); CHKERRXX(ierr);

//    diffusion_step(solver_b, ds_b, qb[is], q_tmp);

//    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_b); CHKERRXX(ierr);
//  }

//  for (int is = fns-2; is > -1; is--)
//  {
//    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a); CHKERRXX(ierr);

//    diffusion_step(solver_a, ds_a, qb[is], q_tmp);

//    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_a); CHKERRXX(ierr);
//  }

  ierr = VecGetArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p,    &mu_p_ptr);    CHKERRXX(ierr);
  ierr = VecGetArray(mu_m,    &mu_m_ptr);    CHKERRXX(ierr);
  ierr = VecGetArray(mask,    &mask_ptr);    CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n < nodes->num_owned_indeps; ++n)
  {
    if (mask_ptr[n] < 0.)
    {
      exp_w_a_ptr[n] = 1./ds_a + (mu_p_ptr[n]-mu_m_ptr[n]);
      exp_w_b_ptr[n] = 1./ds_b + (mu_p_ptr[n]+mu_m_ptr[n]);
    } else {
      exp_w_a_ptr[n] = 0;
      exp_w_b_ptr[n] = 0;
    }
  }

  ierr = VecRestoreArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p,    &mu_p_ptr);    CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m,    &mu_m_ptr);    CHKERRXX(ierr);
  ierr = VecRestoreArray(mask,    &mask_ptr);    CHKERRXX(ierr);

  solver_a.set_diag(exp_w_a);
  solver_b.set_diag(exp_w_b);

  // forward propagator
  for (int is = 1;   is <= fns-1; is++) diffusion_step(solver_a, ds_a, qf[is], qf[is-1]);
  for (int is = fns; is <= ns -1; is++) diffusion_step(solver_b, ds_b, qf[is], qf[is-1]);

  // backward propagator
  for (int is = ns -2; is >= fns-1; is--) diffusion_step(solver_b, ds_b, qb[is], qb[is+1]);
  for (int is = fns-2; is >= 0;     is--) diffusion_step(solver_a, ds_a, qb[is], qb[is+1]);

}

void my_p4est_scft_t::diffusion_step(my_p4est_poisson_nodes_mls_t &solver, double ds, Vec &sol, Vec &sol_nm1)
{
  ierr = VecCopy(sol_nm1, rhs); CHKERRXX(ierr);
  ierr = VecScale(rhs, 1.0/ds); CHKERRXX(ierr);
  solver.set_rhs(rhs);

  // Solve linear system
  solver.solve(sol, true, false);
}

void my_p4est_scft_t::calculate_densities()
{
  // calculate densities
  double *time_integrand = new double [ns];

  double **qf_ptr = new double * [ns];
  double **qb_ptr = new double * [ns];

  double *rho_a_ptr;
  double *rho_b_ptr;

  double *mask_ptr;

  ierr = VecGetArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  for (int is = 0; is < ns; is++)
  {
    ierr = VecGetArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
    ierr = VecGetArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
  }

  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

  // calculate densities only for local nodes
  foreach_node(n, nodes)
  {
    if (mask_ptr[n] < 0)
    {
//      for (int is = 0; is < ns; is++)
//        time_integrand[is] = qf_ptr[is][n]*qb_ptr[is][n];

//      rho_a_ptr[n] = compute_rho_a(time_integrand);
//      rho_b_ptr[n] = compute_rho_b(time_integrand);
      rho_a_ptr[n] = 0;
      rho_b_ptr[n] = 0;

      for (int is = 1;   is <= fns-1; is++) rho_a_ptr[n] += qf_ptr[is][n]*qb_ptr[is-1][n];
      for (int is = fns; is <= ns-1;  is++) rho_b_ptr[n] += qf_ptr[is][n]*qb_ptr[is-1][n];

      rho_a_ptr[n] *= ds_a;
      rho_b_ptr[n] *= ds_b;

    } else {
      rho_a_ptr[n] = 0.0;
      rho_b_ptr[n] = 0.0;
    }
  }

  ierr = VecRestoreArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  for (int is = 0; is < ns; is++)
  {
    ierr = VecRestoreArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
    ierr = VecRestoreArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
  Q = integrate_over_domain_fast(qf[ns-1])/volume;
//  Q = .5*(integrate_over_domain_fast(qf[ns-1])+integrate_over_domain_fast(qb[0]))/volume;
//  Q = (integrate_over_domain_fast(rho_a) + integrate_over_domain_fast(rho_b))/volume;

  ierr = VecScale(rho_a, 1.0/Q); CHKERRXX(ierr);
  ierr = VecScale(rho_b, 1.0/Q); CHKERRXX(ierr);

  delete[] time_integrand;
  delete[] qf_ptr;
  delete[] qb_ptr;

  double mu_m_sqrd_int = integrate_over_domain_fast_squared(mu_m);

  energy = energy_singular_part + (mu_m_sqrd_int/XN - volume*log(Q))/pow(scalling, P4EST_DIM);

}

double my_p4est_scft_t::compute_rho_a(double *integrand)
{
  double result = .5*(integrand[0] + integrand[fns-1]);
  for (int i = 1; i < fns-1; i++) result += integrand[i];
  return result*ds_a;
}

double my_p4est_scft_t::compute_rho_b(double *integrand)
{
  double result = .5*(integrand[ns-1] + integrand[fns-1]);
  for (int i = fns; i < ns-1; i++) result += integrand[i];
  return result*ds_b;
}

void my_p4est_scft_t::update_potentials(bool update_mu_m, bool update_mu_p)
{
  double *rho_a_ptr;
  double *rho_b_ptr;

  ierr = VecGetArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  double *force_p_ptr;
  double *force_m_ptr;

  ierr = VecGetArray(force_p, &force_p_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(force_m, &force_m_ptr); CHKERRXX(ierr);

  double *mu_p_ptr;
  double *mu_m_ptr;

  ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  double *mask_ptr;

  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

  double rho_avg = (integrate_over_domain_fast(rho_a) + integrate_over_domain_fast(rho_b))/volume;

  foreach_node(n, nodes)
  {
    if (mask_ptr[n] < 0.)
    {
      force_p_ptr[n] = rho_a_ptr[n] + rho_b_ptr[n] - 1.0*rho_avg;
      force_m_ptr[n] = 2.0*mu_m_ptr[n]/XN - rho_a_ptr[n] + rho_b_ptr[n];
    }
    else
    {
      force_p_ptr[n] = -mu_p_ptr[n]/lambda;
      force_m_ptr[n] =  mu_m_ptr[n]/lambda;
    }
  }

  ierr = VecRestoreArray(force_p, &force_p_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(force_m, &force_m_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

  force_p_avg = sqrt(integrate_over_domain_fast_squared(force_p)/volume);
  force_m_avg = sqrt(integrate_over_domain_fast_squared(force_m)/volume);


  if (update_mu_m)
  {
    ierr = VecAXPBYGhost(mu_m, -lambda, 1., force_m); CHKERRXX(ierr);
  }

  if (update_mu_p)
  {
    ierr = VecAXPBYGhost(mu_p,  lambda, 1., force_p); CHKERRXX(ierr);
    mu_p_avg = integrate_over_domain_fast(mu_p)/volume;
    ierr = VecShiftGhost(mu_p, -mu_p_avg); CHKERRXX(ierr);
  }

//  mu_m_avg = integrate_over_domain_fast(mu_m)/volume;


//  ierr = VecPointwiseMult(mu_p, mu_p, mask); CHKERRXX(ierr);
//  ierr = VecPointwiseMult(mu_m, mu_m, mask); CHKERRXX(ierr);
}

void my_p4est_scft_t::save_VTK(int compt)
{
  ierr = VecGhostUpdateBegin(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);

//  ls.extend_Over_Interface_TVD(phi_smooth, rho_a);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_b);

//  ls.extend_Over_Interface_TVD(phi_smooth, mu_p);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_m);

  // name of output file
  char *out_dir;
  out_dir = getenv("OUT_DIR");

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/scft_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  //
  bool no_phi = 0;
  if (phi_smooth == NULL)
  {
    ierr = VecDuplicate(mu_m, &phi_smooth); CHKERRXX(ierr);
    VecSetGhost(phi_smooth, -1);
    no_phi = 1;
  }

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
  //

  // get access to data fields
  double *phi_p, *rho_a_p, *rho_b_p, *mu_p_p, *mu_m_p, *mask_p;

//  ierr = VecGetArray(phi->at(0), &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(qf[ns-1], &rho_a_p); CHKERRXX(ierr);
//  ierr = VecGetArray(qb[0], &rho_b_p); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(rho_a, &rho_a_p); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_p); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m, &mu_m_p); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p, &mu_p_p); CHKERRXX(ierr);
  ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);

  // write into file
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         6, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "rho_a", rho_a_p,
                         VTK_POINT_DATA, "rho_b", rho_b_p,
                         VTK_POINT_DATA, "mu_m", mu_m_p,
                         VTK_POINT_DATA, "mu_p", mu_p_p,
                         VTK_POINT_DATA, "mask", mask_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  // restore access to data fields
//  ierr = VecRestoreArray(phi->at(0), &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(qf[ns-1], &rho_a_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(qb[0], &rho_b_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_a, &rho_a_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m, &mu_m_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p, &mu_p_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);

  //
  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  if (no_phi)
  {
    ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);
    phi_smooth = NULL;
  }
  //

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



void my_p4est_scft_t::save_VTK_q(int compt)
{
  ierr = VecGhostUpdateBegin(qf[compt], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(qf[compt], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(qb[compt], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(qb[compt], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  char *out_dir;
  out_dir = getenv("OUT_DIR");

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/scft_q_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *qf_ptr, *qb_ptr, *qfqb_ptr, *mask_p;

  Vec qfqb;

  ierr = VecCreateGhostNodes(p4est, nodes, &qfqb); CHKERRXX(ierr);

  ierr = VecGetArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(qf[compt], &qf_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(qb[compt], &qb_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(qfqb, &qfqb_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);

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
  {
    qfqb_ptr[n] = qf_ptr[n]*qb_ptr[n];
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         5, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "qf", qf_ptr,
                         VTK_POINT_DATA, "qb", qb_ptr,
                         VTK_POINT_DATA, "qfqb", qfqb_ptr,
                         VTK_POINT_DATA, "mask", mask_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[compt], &qf_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[compt], &qb_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(qfqb, &qfqb_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);

  ierr = VecDestroy(qfqb); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}

void my_p4est_scft_t::assemble_integrating_vec()
{
  if (integrating_vec != NULL) { ierr = VecDestroy(integrating_vec); CHKERRXX(ierr); }

  ierr = VecCreateGhostNodes(p4est, nodes, &integrating_vec); CHKERRXX(ierr);

  ierr = VecSet(integrating_vec, 0.0); CHKERRXX(ierr);

  double *int_vec_ptr;

  ierr = VecGetArray(integrating_vec, &int_vec_ptr); CHKERRXX(ierr);
  const p4est_locidx_t *q2n = nodes->local_nodes;

  /* loop through local quadrants */

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);

      p4est_locidx_t quad_idx_forest = quad_idx + tree->quadrants_offset;

      /* get volume of a quadrant */
      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dV = (tree_xmax-tree_xmin)*dmin*(tree_ymax-tree_ymin)*dmin/4.0;
#ifdef P4_TO_P8
      dV *= (tree_zmax-tree_zmin)*dmin/2.0;
#endif

      /* loop through nodes of a quadrant and put weights on those nodes, which are local */
      p4est_locidx_t offset = quad_idx_forest*P4EST_CHILDREN;
      for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
      {
        p4est_locidx_t node_idx = q2n[offset + child_idx];
        if (node_idx < nodes->num_owned_indeps)
          int_vec_ptr[node_idx] += dV;
      }
    }
  }

  /* loop through ghosts */
  for (p4est_locidx_t ghost_idx = 0; ghost_idx < ghost->ghosts.elem_count; ++ghost_idx)
  {
    // get a ghost quadrant
    p4est_quadrant_t* quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);

    // get a tree to which the ghost quadrant belongs
    p4est_topidx_t tree_idx = quad->p.piggy3.which_tree;

    // get coordinates of the tree
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    // calculate volume per each node of the ghost quadrant
    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dV = (tree_xmax-tree_xmin)*dmin*(tree_ymax-tree_ymin)*dmin/4.0;
#ifdef P4_TO_P8
    dV *= (tree_zmax-tree_zmin)*dmin/2.0;
#endif

    // loop through nodes of a quadrant and put weights on those nodes, which are local
    p4est_locidx_t offset = (p4est->local_num_quadrants + ghost_idx)*P4EST_CHILDREN;
    for (int child_idx = 0; child_idx < P4EST_CHILDREN; child_idx++)
    {
      p4est_locidx_t node_idx = q2n[offset + child_idx];
      if (node_idx < nodes->num_owned_indeps)
        int_vec_ptr[node_idx] += dV;
    }
  }

  /* post processing step: multiply every weight by fraction */
  /* TODO: here we just use Vec node_vol from one of the solvers,
   * but in the future it should be replaced with a proper loop through nodes
   * and calculation of volume fraction for each node
   */

  if (phi_smooth != NULL)
  {
    // domain
    std::vector<double *> phi_ptr(num_surfaces, NULL);

    for (unsigned short i = 0; i < num_surfaces; i++)
    {
      ierr = VecGetArray(phi[i], &phi_ptr[i]); CHKERRXX(ierr);
    }

    double *phi_eff_ptr;
    ierr = VecGetArray(phi_smooth, &phi_eff_ptr); CHKERRXX(ierr);

    // data for refined cells
    unsigned short fv_size_x = 0;
    unsigned short fv_size_y = 0;
#ifdef P4_TO_P8
    unsigned short fv_size_z = 0;
#endif

    double fv_xmin, fv_xmax;
    double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
    double fv_zmin, fv_zmax;
#endif

    double xyz_C[P4EST_DIM];

    bool neighbors_exist[num_neighbors_cube];
    p4est_locidx_t neighbors[num_neighbors_cube];

    // interpolations
    my_p4est_interpolation_nodes_local_t phi_interp_local(ngbd);

    for(p4est_locidx_t n=0; n < nodes->num_owned_indeps; n++) // loop over nodes
    {
      // sample level-set function at cube nodes and check if crossed
      bool is_crossed = false;

      ngbd->get_all_neighbors(n, neighbors, neighbors_exist);

      for (unsigned short phi_idx = 0; phi_idx < num_surfaces; ++phi_idx)
      {
        bool is_one_positive = false;
        bool is_one_negative = false;

        for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
          if (neighbors_exist[i])
          {
            is_one_positive = is_one_positive || phi_ptr[phi_idx][neighbors[i]] > 0;
            is_one_negative = is_one_negative || phi_ptr[phi_idx][neighbors[i]] < 0;
          }

        is_crossed = is_crossed || (is_one_negative && is_one_positive);
      }

      if (!is_crossed && phi_eff_ptr[n] > 0.)
      {
        int_vec_ptr[n] = 0;
      }
      else if (is_crossed)
      {
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

        node_xyz_fr_n(n, p4est, nodes, xyz_C);
        double x_C  = xyz_C[0];
        double y_C  = xyz_C[1];
#ifdef P4_TO_P8
        double z_C  = xyz_C[2];
#endif

        phi_interp_local.initialize(n);

        // determine dimensions of cube
        fv_size_x = 0;
        fv_size_y = 0;
#ifdef P4_TO_P8
        fv_size_z = 0;
#endif
        if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dxyz[0];} else {fv_xmin = x_C;}
        if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dxyz[0];} else {fv_xmax = x_C;}

        if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dxyz[1];} else {fv_ymin = y_C;}
        if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dxyz[1];} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
        if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dxyz[2];} else {fv_zmin = z_C;}
        if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dxyz[2];} else {fv_zmax = z_C;}
#endif

        if (cube_refinement == 0)
        {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }
        // Reconstruct geometry
#ifdef P4_TO_P8
        double cube_xyz_min[] = { fv_xmin, fv_ymin, fv_zmin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax, fv_zmax };
        int  cube_mnk[] = { fv_size_x, fv_size_y, fv_size_z };
        cube3_mls_t cube(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order);
#else
        double cube_xyz_min[] = { fv_xmin, fv_ymin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax };
        int  cube_mnk[] = { fv_size_x, fv_size_y };
        cube2_mls_t cube(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order);
#endif

        // get points at which values of level-set functions are needed
        std::vector<double> x_grid; cube.get_x_coord(x_grid);
        std::vector<double> y_grid; cube.get_y_coord(y_grid);
#ifdef P4_TO_P8
        std::vector<double> z_grid; cube.get_z_coord(z_grid);
#endif
        unsigned int points_total = x_grid.size();

        std::vector<double> phi_cube(num_surfaces*points_total, -1);

        // compute values of level-set functions at needed points
        for (unsigned short phi_idx = 0; phi_idx < num_surfaces; ++phi_idx)
        {
          phi_interp_local.set_input(phi_ptr[phi_idx], linear);

          for (unsigned int i = 0; i < points_total; ++i)
          {
#ifdef P4_TO_P8
            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
#else
            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
#endif
          }
        }

        // reconstruct geometry
        reconstruct_cube(cube, phi_cube, action, color);

        // get quadrature points
        std::vector<double> cube_dom_w;
        std::vector<double> cube_dom_x;
        std::vector<double> cube_dom_y;
#ifdef P4_TO_P8
        std::vector<double> cube_dom_z;
#endif

#ifdef P4_TO_P8
        cube.quadrature_over_domain(cube_dom_w, cube_dom_x, cube_dom_y, cube_dom_z);
#else
        cube.quadrature_over_domain(cube_dom_w, cube_dom_x, cube_dom_y);
#endif

        // compute cut-cell volume
        double volume_cut_cell = 0.;

        for (unsigned int i = 0; i < cube_dom_w.size(); ++i)
        {
          volume_cut_cell += cube_dom_w[i];
        }

        int_vec_ptr[n] = volume_cut_cell;
      }

    }

    ierr = VecRestoreArray(phi_smooth, &phi_eff_ptr); CHKERRXX(ierr);

    for (unsigned short i = 0; i < num_surfaces; i++)
    {
      ierr = VecRestoreArray(phi[i], &phi_ptr[i]); CHKERRXX(ierr);
    }
  }

  ierr = VecRestoreArray(integrating_vec, &int_vec_ptr); CHKERRXX(ierr);
}

double my_p4est_scft_t::integrate_over_domain_fast(Vec f)
{
  double *integrating_vec_ptr;
  double *f_ptr;

  ierr = VecGetArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecGetArray(f,               &f_ptr);                CHKERRXX(ierr);

  double sum = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    sum += integrating_vec_ptr[n]*f_ptr[n];
  }

  ierr = VecRestoreArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecRestoreArray(f,               &f_ptr);                CHKERRXX(ierr);

  /* compute global sum */
  double sum_global = 0;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

double my_p4est_scft_t::integrate_over_domain_fast_squared(Vec f)
{
  double *integrating_vec_ptr;
  double *f_ptr;

  ierr = VecGetArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecGetArray(f,               &f_ptr);                CHKERRXX(ierr);

  double sum = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    sum += integrating_vec_ptr[n]*f_ptr[n]*f_ptr[n];
  }

  ierr = VecRestoreArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecRestoreArray(f,               &f_ptr);                CHKERRXX(ierr);

  /* compute global sum */
  double sum_global = 0;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

double my_p4est_scft_t::integrate_over_domain_fast_two(Vec f0, Vec f1)
{
  double *integrating_vec_ptr;
  double *f0_ptr, *f1_ptr;

  ierr = VecGetArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecGetArray(f0,              &f0_ptr);               CHKERRXX(ierr);
  ierr = VecGetArray(f1,              &f1_ptr);               CHKERRXX(ierr);

  double sum = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    sum += integrating_vec_ptr[n]*f0_ptr[n]*f1_ptr[n];
  }

  ierr = VecRestoreArray(integrating_vec, &integrating_vec_ptr);  CHKERRXX(ierr);
  ierr = VecRestoreArray(f0,              &f0_ptr);               CHKERRXX(ierr);
  ierr = VecRestoreArray(f1,              &f1_ptr);               CHKERRXX(ierr);

  /* compute global sum */
  double sum_global = 0;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

void my_p4est_scft_t::smooth_singularity_in_pressure_field()
{
  // turns out the best solution is to just start from zero
  VecSetGhost(mu_p, 0.0);
}

void my_p4est_scft_t::sync_and_extend()
{
  ierr = VecGhostUpdateBegin(qf[ns-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (qf[ns-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // extend over smoothed interface
  if (phi_smooth != NULL)
  {
    my_p4est_level_set_t ls(ngbd);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, qf[ns-1], 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, qb[0], 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, rho_a, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, rho_b, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, mu_m, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
    ls.extend_Over_Interface_TVD_Full(phi_smooth, mu_p, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, NULL, mask);
  }
}

void my_p4est_scft_t::compute_energy_shape_derivative(int phi_idx, Vec velo)
{
  my_p4est_level_set_t ls(ngbd);

  Vec energy_shape_deriv_tmp;
  ierr = VecDuplicate(mu_m, &energy_shape_deriv_tmp); CHKERRXX(ierr);

  // volumetric term
  double energy_shape_deriv_volumetric = ( 1.0 - (energy-log(Q)) )/volume;

  double rho_avg = (integrate_over_domain_fast(rho_a) + integrate_over_domain_fast(rho_b))/volume;

  // compute velocity
  double *energy_shape_deriv_ptr;
  double *qf_ptr, *rho_a_ptr, *bc_coeffs_a_ptr, *mu_m_ptr;
  double *qb_ptr, *rho_b_ptr, *bc_coeffs_b_ptr, *mu_p_ptr;
  double *kappa_ptr;

  ierr = VecGetArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecGetArray(qf[ns-1],              &qf_ptr                 ); CHKERRXX(ierr);
  ierr = VecGetArray(qb[0],                 &qb_ptr                 ); CHKERRXX(ierr);
  ierr = VecGetArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);
//  ierr = VecGetArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
//  ierr = VecGetArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
//  ierr = VecGetArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);
  // calculate laplace of total density
  Vec rho_total, rho_total_xx, rho_total_yy;
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_xx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_yy); CHKERRXX(ierr);

  double *rho_total_ptr;
  ierr = VecGetArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  foreach_node(n, nodes)
  {
    rho_total_ptr[n] = rho_a_ptr[n]+rho_b_ptr[n];
  }
  ierr = VecRestoreArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  ngbd->second_derivatives_central(rho_total, rho_total_xx, rho_total_yy);

  double *rho_total_xx_ptr;
  double *rho_total_yy_ptr;
  ierr = VecGetArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);

  double xyz[P4EST_DIM];
  foreach_node(n, nodes)
  {
    node_xyz_fr_n(n, p4est, nodes, xyz);

    double gamma_a_val = gamma_a[phi_idx]->value(xyz)*scalling;
    double gamma_b_val = gamma_b[phi_idx]->value(xyz)*scalling;

    energy_shape_deriv_ptr[n] = 0.0*energy_shape_deriv_volumetric
        + (mu_m_ptr[n]*mu_m_ptr[n]/XN - mu_p_ptr[n]*rho_avg)
        - 0.5*(qf_ptr[n]+qb_ptr[n])/Q;
//        + 1.0*kappa_ptr[n]*(rho_a_ptr[n]*gamma_a_val + rho_b_ptr[n]*gamma_b_val)/volume
//        - 1.0*2.0*(bc_coeffs_a_ptr[n]*rho_a_ptr[n]*gamma_a_val + bc_coeffs_b_ptr[n]*rho_b_ptr[n]*gamma_b_val)/volume;

    energy_shape_deriv_ptr[n] /= pow(scalling, P4EST_DIM);

//    energy_shape_deriv_ptr[n] = - (mu_m_sqrd_int/XN-mu_p_int)/volume/volume
//        + (mu_m_ptr[n]*mu_m_ptr[n]/XN - mu_p_ptr[n])/volume;

//    energy_shape_deriv_ptr[n] += 1.0/volume - 0.5*(qf_ptr[n]+qb_ptr[n])/Q/volume + 0.0*(rho_total_xx_ptr[n] + rho_total_yy_ptr[n])/volume;
//    energy_shape_deriv_ptr[n] = (0.5*(qf_ptr[n]+qb_ptr[n]) - 0.5*(rho_total_xx_ptr[n] + rho_total_yy_ptr[n])*Q);
//    energy_shape_deriv_ptr[n] -= 1.0*( kappa_ptr[n]*( bc_coeffs_a_ptr[n]*rho_a_ptr[n] +
//                                                      bc_coeffs_b_ptr[n]*rho_b_ptr[n] )
//                                   - 2.0*( bc_coeffs_a_ptr[n]*bc_coeffs_a_ptr[n]*rho_a_ptr[n] +
//                                           bc_coeffs_b_ptr[n]*bc_coeffs_b_ptr[n]*rho_b_ptr[n] ) )*Q;
//    energy_shape_deriv_ptr[n] -= addition_from_bc;
//   energy_shape_deriv_ptr[n] = kappa_ptr[n];
//    energy_shape_deriv_ptr[n] = 1.0/volume;
  }

  ierr = VecRestoreArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(rho_total); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_xx); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_yy); CHKERRXX(ierr);

  ierr = VecRestoreArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[ns-1],              &qf_ptr                 ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[0],                 &qb_ptr                 ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);

  // extrapolate from moving moving interface
//  ls.extend_from_interface_to_whole_domain_TVD(phi->at(phi_idx), energy_shape_deriv_tmp, velo);
  VecCopyGhost(energy_shape_deriv_tmp, velo);

  ierr = VecDestroy(energy_shape_deriv_tmp); CHKERRXX(ierr);
}

//void my_p4est_scft_t::compute_energy_shape_derivative_contact_term(int phi0_idx, int phi1_idx)
//{
//  if (energy_shape_deriv_contact_term != NULL) { ierr = VecDestroy(energy_shape_deriv_contact_term); CHKERRXX(ierr); }
//  ierr = VecDuplicate(mu_m, &energy_shape_deriv_contact_term); CHKERRXX(ierr);

//  double *energy_shape_deriv_contact_term_ptr;
//  double *rho_a_ptr, *normal_phi0_ptr[P4EST_DIM];
//  double *rho_b_ptr, *normal_phi1_ptr[P4EST_DIM];

//  ierr = VecGetArray(energy_shape_deriv_contact_term, &energy_shape_deriv_contact_term_ptr); CHKERRXX(ierr);

//  ierr = VecGetArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    ierr = VecGetArray(normal[phi0_idx][dim], &normal_phi0_ptr[dim]); CHKERRXX(ierr);
//    ierr = VecGetArray(normal[phi1_idx][dim], &normal_phi1_ptr[dim]); CHKERRXX(ierr);
//  }

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  {
//    double x = node_x_fr_n(n, p4est, nodes);
//    double y = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//    double z = node_z_fr_n(n, p4est, nodes);
//#endif

//    // compute contact angle between interfaces
//    double cos_theta = -(normal_phi0_ptr[0][n]*normal_phi1_ptr[0][n] + normal_phi0_ptr[1][n]*normal_phi1_ptr[1][n]);

//    double gamma_phi0 = rho_a_ptr[n]*(*gamma_a[phi0_idx])(x,y) + rho_b_ptr[n]*(*gamma_b[phi0_idx])(x,y);
//    double gamma_phi1 = rho_a_ptr[n]*(*gamma_a[phi1_idx])(x,y) + rho_b_ptr[n]*(*gamma_b[phi1_idx])(x,y);
//    double gamma_air_val  = (*gamma_air)(x,y);

////    energy_shape_deriv_contact_term_ptr[n] = -1.0*(gamma_phi0 - gamma_air_val + gamma_phi1*cos_theta)/sqrt(1.0-cos_theta*cos_theta)*Q*volume;
//    energy_shape_deriv_contact_term_ptr[n] = 1.0*scalling*(gamma_phi0 - gamma_air_val + gamma_phi1*cos_theta)/sqrt(1.0-cos_theta*cos_theta)/volume;
////    energy_shape_deriv_contact_term_ptr[n] = (1.0+cos_theta)/sqrt(1.0-cos_theta*cos_theta);
//  }

//  ierr = VecRestoreArray(energy_shape_deriv_contact_term, &energy_shape_deriv_contact_term_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    ierr = VecRestoreArray(normal[phi0_idx][dim], &normal_phi0_ptr[dim]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(normal[phi1_idx][dim], &normal_phi1_ptr[dim]); CHKERRXX(ierr);
//  }

//  // sync the derivative among procs
//  ierr = VecGhostUpdateBegin(energy_shape_deriv_contact_term, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (energy_shape_deriv_contact_term, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // extend over smoothed interface
//  Vec energy_shape_deriv_contact_term_tmp;
//  ierr = VecDuplicate(mu_m, &energy_shape_deriv_contact_term_tmp); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_from_interface_to_whole_domain_TVD(phi[phi0_idx], energy_shape_deriv_contact_term, energy_shape_deriv_contact_term_tmp);
//  ls.extend_from_interface_to_whole_domain_TVD(phi[phi1_idx], energy_shape_deriv_contact_term_tmp, energy_shape_deriv_contact_term);
////  ls.extend_Over_Interface_TVD(phi_smooth, contact_term_of_energy_shape_deriv);

//  ierr = VecDestroy(energy_shape_deriv_contact_term_tmp); CHKERRXX(ierr);
//}

//double my_p4est_scft_t::compute_change_in_energy(int phi_idx, Vec norm_velo, double dt)
//{
//  Vec integrand;
//  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

//  double *energy_shape_deriv_ptr; ierr = VecGetArray(energy_shape_deriv, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  double *integrand_ptr;          ierr = VecGetArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  double *norm_velo_ptr;          ierr = VecGetArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  {
//    integrand_ptr[n] = norm_velo_ptr[n]*energy_shape_deriv_ptr[n];
//  }

//  ierr = VecRestoreArray(energy_shape_deriv, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  my_p4est_integration_mls_t integrator(p4est, nodes);
//  integrator.set_phi(phi, action, color);

//  double val = integrator.integrate_over_interface(phi_idx, integrand)*dt;

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

//  return val;
//}

//double my_p4est_scft_t::compute_change_in_energy_contact_term(int phi0_idx, int phi1_idx, Vec norm_velo, double dt)
//{
//  Vec integrand;
//  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

//  double *energy_shape_deriv_ptr;
//  double *integrand_ptr;
//  double *norm_velo_ptr;

//  my_p4est_integration_mls_t integrator(p4est, nodes);
//  integrator.set_phi(phi, action, color);

//  ierr = VecGetArray(energy_shape_deriv_contact_term, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  ierr = VecGetArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    integrand_ptr[n] = norm_velo_ptr[n]*energy_shape_deriv_ptr[n];
//  }

//  ierr = VecRestoreArray(energy_shape_deriv_contact_term, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  double val = integrator.integrate_over_intersection(phi0_idx, phi1_idx, integrand)*dt;

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

//  return val;
//}

void my_p4est_scft_t::compute_normal_and_curvature()
{
  // delete old values
  for (int surf_idx = 0; surf_idx < normal.size(); surf_idx++)
  {
    for (int dim = 0; dim < P4EST_DIM; dim++)
    {
      ierr = VecDestroy(normal[surf_idx][dim]); CHKERRXX(ierr);
    }
    delete[] normal[surf_idx];

    ierr = VecDestroy(kappa[surf_idx]); CHKERRXX(ierr);
  }

  // allocate memory and compute
  normal.resize(num_surfaces, NULL);
  kappa.resize(num_surfaces, NULL);

  my_p4est_level_set_t ls(ngbd);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  for (int surf_idx = 0; surf_idx < normal.size(); surf_idx++)
  {
    // allocate
    normal[surf_idx] = new Vec[P4EST_DIM];

    if (surf_idx == 0) { for (int dim = 0; dim < P4EST_DIM; dim++) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[surf_idx][dim]); CHKERRXX(ierr); } }
    else               { for (int dim = 0; dim < P4EST_DIM; dim++) { ierr = VecDuplicate(normal[0][dim], &normal[surf_idx][dim]); CHKERRXX(ierr); } }

    ierr = VecDuplicate(phi[surf_idx], &kappa[surf_idx]); CHKERRXX(ierr);

    // compute
    compute_normals_and_mean_curvature(*ngbd, phi[surf_idx], normal[surf_idx], kappa[surf_idx]);

    // extend curvature in normal direction
    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi[surf_idx], kappa[surf_idx], phi[surf_idx]);
  }
}


//-----------------------------------------------------------------
// Density optimization
//-----------------------------------------------------------------

//void my_p4est_scft_t::dsa_initialize()
//{
//  // potentials
//  if (nu_m != NULL) { ierr = VecDestroy(nu_m); CHKERRXX(ierr); }
//  if (nu_p != NULL) { ierr = VecDestroy(nu_p); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &nu_m); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_smooth, &nu_p); CHKERRXX(ierr);

//  // chain propogators
//  for (int i = 0; i < zf.size(); i++) { ierr = VecDestroy(zf[i]); CHKERRXX(ierr); }
//  for (int i = 0; i < zb.size(); i++) { ierr = VecDestroy(zb[i]); CHKERRXX(ierr); }

//  zf.resize(ns, NULL);
//  zb.resize(ns, NULL);

//  for (int i = 0; i < ns; i++)
//  {
//    ierr = VecDuplicate(phi_smooth, &zf[i]); CHKERRXX(ierr);
//    ierr = VecDuplicate(phi_smooth, &zb[i]); CHKERRXX(ierr);
//  }

//  ierr = VecSet(zf[0000], 0.0); CHKERRXX(ierr);
//  ierr = VecSet(zb[ns-1], 0.0); CHKERRXX(ierr);

//  // potentials
//  if (nu_a != NULL) { ierr = VecDestroy(nu_a); CHKERRXX(ierr); }
//  if (nu_b != NULL) { ierr = VecDestroy(nu_b); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &nu_a); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_smooth, &nu_b); CHKERRXX(ierr);

//  if (force_nu_m != NULL)    { ierr = VecDestroy(force_nu_m); CHKERRXX(ierr); }
//  if (force_nu_p != NULL)    { ierr = VecDestroy(force_nu_p); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &force_nu_m); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_smooth, &force_nu_p); CHKERRXX(ierr);

//  if (mu_t != NULL)    { ierr = VecDestroy(mu_t); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &mu_t); CHKERRXX(ierr);

//  // create exp_w vectors
//  double *mask_ptr;
//  double *mu_p_ptr;
//  double *mu_m_ptr;
//  double *exp_w_a_ptr;
//  double *exp_w_b_ptr;

//  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

//  ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

//  ierr = VecGetArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);

//  foreach_node(n, nodes)
//  {
//    if (mask_ptr[n] < 0)
//    {
//      exp_w_a_ptr[n] = exp(-0.5*(mu_p_ptr[n]-mu_m_ptr[n])*ds_a);
//      exp_w_b_ptr[n] = exp(-0.5*(mu_p_ptr[n]+mu_m_ptr[n])*ds_b);
//    } else {
//      exp_w_a_ptr[n] = 1.0;
//      exp_w_b_ptr[n] = 1.0;
//    }
//  }

//  ierr = VecRestoreArray(exp_w_a, &exp_w_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(exp_w_b, &exp_w_b_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

//  if (psi_a != NULL) { ierr = VecDestroy(psi_a); CHKERRXX(ierr); }
//  if (psi_b != NULL) { ierr = VecDestroy(psi_b); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &psi_a); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_smooth, &psi_b); CHKERRXX(ierr);
//}

//void my_p4est_scft_t::dsa_initialize_fields()
//{
//  // potentials
//  if (nu_m != NULL) { ierr = VecDestroy(nu_m); CHKERRXX(ierr); }
//  if (nu_p != NULL) { ierr = VecDestroy(nu_p); CHKERRXX(ierr); }

//  ierr = VecDuplicate(phi_smooth, &nu_m); CHKERRXX(ierr);
//  ierr = VecDuplicate(phi_smooth, &nu_p); CHKERRXX(ierr);

//  ierr = VecSet(nu_m, 0.0); CHKERRXX(ierr);
//  ierr = VecSet(nu_p, 0.0); CHKERRXX(ierr);
//}

//void my_p4est_scft_t::dsa_solve_for_propogators()
//{
//  nu_0 = 2.0*integrate_over_domain_fast_two(mu_m, nu_m)/XN/volume;

//  // compute nu_a and nu_b
//  double *nu_p_ptr;
//  double *nu_m_ptr;
//  double *nu_a_ptr;
//  double *nu_b_ptr;
//  double *mask_ptr;

//  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

//  ierr = VecGetArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);

//  ierr = VecGetArray(nu_a, &nu_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(nu_b, &nu_b_ptr); CHKERRXX(ierr);

//  foreach_node(n, nodes)
//  {
//    if (mask_ptr[n] < 0)
//    {
//      nu_a_ptr[n] = (nu_p_ptr[n]-nu_m_ptr[n])+nu_0;
//      nu_b_ptr[n] = (nu_p_ptr[n]+nu_m_ptr[n])+nu_0;
//    } else {
//      nu_a_ptr[n] = 0.0;
//      nu_b_ptr[n] = 0.0;
//    }
//  }

//  ierr = VecRestoreArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(nu_a, &nu_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(nu_b, &nu_b_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

//  // forward propagator
//  for (int is = 1;   is < fns; is++) { dsa_diffusion_step(solver_a, ds_a, zf[is], zf[is-1], exp_w_a, qf[is], nu_a); }
//  for (int is = fns; is < ns;  is++) { dsa_diffusion_step(solver_b, ds_b, zf[is], zf[is-1], exp_w_b, qf[is], nu_b); }

//  // backward propagator
//  for (int is = ns-2;  is > fns-2; is--) { dsa_diffusion_step(solver_b, ds_a, zb[is], zb[is+1], exp_w_b, qb[is], nu_b); }
//  for (int is = fns-2; is > -1;    is--) { dsa_diffusion_step(solver_a, ds_b, zb[is], zb[is+1], exp_w_a, qb[is], nu_a); }
//}

//void my_p4est_scft_t::dsa_diffusion_step(my_p4est_poisson_nodes_mls_t *solver, double ds, Vec &sol, Vec &sol_nm1, Vec &exp_w, Vec &q, Vec &nu)
//{
//  // only fully implicit scheme at the moment (!)
//  ierr = VecCopy(sol_nm1, rhs); CHKERRXX(ierr);
//  ierr = VecPointwiseMult(rhs, rhs, exp_w); CHKERRXX(ierr);

//  ierr = VecPointwiseMult(q_tmp, q, nu); CHKERRXX(ierr);

//  ierr = VecAXPBY(rhs, -1.0, 1.0/ds, q_tmp);    CHKERRXX(ierr);

//  solver->set_rhs(rhs);

//  // Solve linear system
//  solver->solve(sol, true);

//  ierr = VecPointwiseMult(sol, sol, exp_w); CHKERRXX(ierr);
//}

//void my_p4est_scft_t::dsa_compute_densities()
//{
//  // calculate densities
//  std::vector<double> time_integrand(ns, 0);

//  std::vector<double *> qf_ptr(ns, NULL);
//  std::vector<double *> qb_ptr(ns, NULL);
//  std::vector<double *> zf_ptr(ns, NULL);
//  std::vector<double *> zb_ptr(ns, NULL);

//  double *mask_ptr;
//  double *rho_a_ptr;
//  double *rho_b_ptr;

//  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(psi_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(psi_b, &rho_b_ptr); CHKERRXX(ierr);

//  for (int is = 0; is < ns; is++)
//  {
//    ierr = VecGetArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
//    ierr = VecGetArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
//    ierr = VecGetArray(zf[is], &zf_ptr[is]); CHKERRXX(ierr);
//    ierr = VecGetArray(zb[is], &zb_ptr[is]); CHKERRXX(ierr);
//  }

//  // calculate densities only for local nodes
//  foreach_node(n, nodes)
//  {
//    if (mask_ptr[n] < 0)
//    {
//      for (int is = 0; is < ns; is++)
//        time_integrand[is] = qf_ptr[is][n]*zb_ptr[is][n] + zf_ptr[is][n]*qb_ptr[is][n];

//      rho_a_ptr[n] = compute_rho_a(time_integrand.data());
//      rho_b_ptr[n] = compute_rho_b(time_integrand.data());
//    } else {
//      rho_a_ptr[n] = 0.0;
//      rho_b_ptr[n] = 0.0;
//    }
//  }

//  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(psi_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(psi_b, &rho_b_ptr); CHKERRXX(ierr);

//  for (int is = 0; is < ns; is++)
//  {
//    ierr = VecRestoreArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(zf[is], &zf_ptr[is]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(zb[is], &zb_ptr[is]); CHKERRXX(ierr);
//  }

//  ierr = VecScale(psi_a, 1.0/Q); CHKERRXX(ierr);
//  ierr = VecScale(psi_b, 1.0/Q); CHKERRXX(ierr);

//  cost_function = dsa_compute_cost_function();
//}

//void my_p4est_scft_t::dsa_update_potentials()
//{
//  double *mask_ptr;

//  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

//  double *rho_a_ptr;
//  double *rho_b_ptr;

//  ierr = VecGetArray(psi_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(psi_b, &rho_b_ptr); CHKERRXX(ierr);

//  double *force_p_ptr;
//  double *force_m_ptr;

//  ierr = VecGetArray(force_nu_p, &force_p_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(force_nu_m, &force_m_ptr); CHKERRXX(ierr);

//  double *mu_m_ptr;
//  double *mu_t_ptr;

//  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);

//  double *nu_p_ptr;
//  double *nu_m_ptr;

//  ierr = VecGetArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);

//  double scale_factor = XN/Q/volume/volume;

//  foreach_node(n, nodes)
//  {
//    if (mask_ptr[n] < 0)
//    {
//      force_p_ptr[n] = rho_a_ptr[n] + rho_b_ptr[n];
//      force_m_ptr[n] = 2.0*(mu_m_ptr[n] - mu_t_ptr[n])/XN + 2.0*nu_m_ptr[n]/XN - rho_a_ptr[n] + rho_b_ptr[n];

//      nu_p_ptr[n] += 2.0*lambda*force_p_ptr[n];
//      nu_m_ptr[n] -= 2.0*lambda*force_m_ptr[n];
//    } else {
//      nu_p_ptr[n] = 0;
//      nu_m_ptr[n] = 0;
//    }
//  }

//  ierr = VecRestoreArray(force_nu_p, &force_p_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(force_nu_m, &force_m_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(psi_a, &rho_a_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(psi_b, &rho_b_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);

//  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

//  force_nu_p_avg = sqrt(integrate_over_domain_fast_squared(force_nu_p)/volume);
//  force_nu_m_avg = sqrt(integrate_over_domain_fast_squared(force_nu_m)/volume);

//  double nu_p_avg = integrate_over_domain_fast(nu_p)/volume;
//  double nu_m_avg = 2.0*integrate_over_domain_fast_two(mu_m, nu_m)/XN/volume;

//  ierr = VecShift(nu_p, -nu_p_avg); CHKERRXX(ierr);
//}

//void my_p4est_scft_t::dsa_sync_and_extend()
//{
//  ierr = VecGhostUpdateBegin(zf[ns-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (zf[ns-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(zb[0000], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (zb[0000], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(nu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (nu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(nu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (nu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // extend over smoothed interface
//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD_Full(phi_smooth, mask, zf[ns-1]);
//  ls.extend_Over_Interface_TVD_Full(phi_smooth, mask, zb[0000]);
//  ls.extend_Over_Interface_TVD_Full(phi_smooth, mask, nu_m);
//  ls.extend_Over_Interface_TVD_Full(phi_smooth, mask, nu_p);
//}

//void my_p4est_scft_t::dsa_compute_shape_gradient(int phi_idx, Vec velo)
//{
//  my_p4est_level_set_t ls(ngbd);
//  Vec density_shape_grad_tmp;
//  ierr = VecDuplicate(mu_m, &density_shape_grad_tmp); CHKERRXX(ierr);

//  double *density_shape_grad_ptr;
//  double *zf_ptr;
//  double *zb_ptr;
//  double *nu_m_ptr;
//  double *nu_p_ptr;
//  double *mu_m_ptr;
//  double *mu_t_ptr;
//  double *kappa_ptr;

//  ierr = VecGetArray(density_shape_grad_tmp, &density_shape_grad_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(zf[ns-1], &zf_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(zb[0000], &zb_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);

////  ierr = VecGetArray(kappa[phi_idx], &kappa_ptr); CHKERRXX(ierr);

////  double kappa_cr = 1.0/(5.0*dxyz_min);

//  foreach_local_node(n, nodes)
//  {
//    density_shape_grad_ptr[n] = SQR(mu_m_ptr[n] - mu_t_ptr[n])/XN
//        - 0.5*(zf_ptr[n] + zb_ptr[n])/Q
//        - (nu_p_ptr[n] + nu_0 - 2.0*mu_m_ptr[n]*nu_m_ptr[n]/XN);

////    double kappa_abs = fabs(kappa_ptr[n]);

////    if (kappa_abs > kappa_cr) density_shape_grad_ptr[n]*exp(-2.0*pow(kappa_abs/kappa_cr - 1.0, 2.0));
//  }

//  ierr = VecRestoreArray(density_shape_grad_tmp, &density_shape_grad_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(zf[ns-1], &zf_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(zb[0000], &zb_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);

////  ierr = VecRestoreArray(kappa[phi_idx], &kappa_ptr); CHKERRXX(ierr);

//  // sync the derivative among procs
//  ierr = VecGhostUpdateBegin(density_shape_grad_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (density_shape_grad_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // extrapolate from moving moving interface
//  ls.extend_Over_Interface_TVD_Full(phi_smooth, mask, density_shape_grad_tmp);
//  ls.extend_from_interface_to_whole_domain_TVD(phi->at(phi_idx), density_shape_grad_tmp, velo);

//  ierr = VecDestroy(density_shape_grad_tmp); CHKERRXX(ierr);
//}

//double my_p4est_scft_t::dsa_compute_cost_function()
//{
//  Vec integrand;
//  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

//  double *integrand_ptr; ierr = VecGetArray(integrand, &integrand_ptr); CHKERRXX(ierr);
//  double *mu_t_ptr;      ierr = VecGetArray(mu_t     , &mu_t_ptr     ); CHKERRXX(ierr);
//  double *mu_m_ptr;      ierr = VecGetArray(mu_m     , &mu_m_ptr     ); CHKERRXX(ierr);

//  foreach_node(n, nodes)
//  {
//    integrand_ptr[n] = SQR(mu_t_ptr[n] - mu_m_ptr[n]);
//  }

//  ierr = VecRestoreArray(integrand, &integrand_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_t     , &mu_t_ptr     ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(mu_m     , &mu_m_ptr     ); CHKERRXX(ierr);

//  double val = integrate_over_domain_fast(integrand)/XN;

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

//  return val;
//}

//double my_p4est_scft_t::dsa_compute_change_in_functional(int phi_idx, Vec norm_velo, Vec density_shape_grad, double dt)
//{
//  Vec integrand;
//  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

//  double *energy_shape_deriv_ptr; ierr = VecGetArray(density_shape_grad, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  double *integrand_ptr;          ierr = VecGetArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  double *norm_velo_ptr;          ierr = VecGetArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  foreach_node(n, nodes)
//  {
//    integrand_ptr[n] = norm_velo_ptr[n]*energy_shape_deriv_ptr[n];
//  }

//  ierr = VecRestoreArray(density_shape_grad, &energy_shape_deriv_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

//  my_p4est_integration_mls_t integrator(p4est, nodes);
//  integrator.set_phi(*phi, *action, color);

//  double val = integrator.integrate_over_interface(phi_idx, integrand)*dt;

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

//  return val;
//}
