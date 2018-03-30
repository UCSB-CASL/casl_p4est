#include "my_p4est_scft.h"

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

my_p4est_scft_t::my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd) :
  brick(ngbd->myb), connectivity(ngbd->p4est->connectivity), p4est(ngbd->p4est), ghost(ngbd->ghost), nodes(ngbd->nodes), hierarchy(ngbd->hierarchy), ngbd(ngbd),
  phi(NULL), action(NULL),
  mu_p(NULL), mu_m(NULL),
  rho_a(NULL), rho_b(NULL),
  gamma_a(NULL), gamma_b(NULL),
  rhs(NULL), rhs_old(NULL), add_to_rhs(NULL),
  solver_a(0, NULL), solver_b(0, NULL),
  force_m(NULL), force_p(NULL),
  phi_smooth(NULL),
  exp_w_a(0, NULL), exp_w_b(0, NULL),
  lambda(0.5),
  mask(NULL), integrating_vec(NULL), q_tmp(NULL),
  use_cn_scheme(false), scheme_coeff(1.0),
//  use_cn_scheme(true), scheme_coeff(2.0),
  singular_part_of_energy(0.0),
  energy_shape_deriv(NULL),
  energy_shape_deriv_alt(NULL),
  contact_term_of_energy_shape_deriv(NULL),
  num_of_refinements(0),
  degree_of_refinement(0)
{
  // Default parameters
  XN = 12.5;
  f = 0.5;
  ns = 100;
  ns_total = ns + 4*(degree_of_refinement - 1)*num_of_refinements;

  // Discretization along the chain
  ds = 1.0 / (double) (ns-1);
  fns = round((double) ns * f);
  fns_adaptive = fns + 2*num_of_refinements*(degree_of_refinement - 1);

  ds_adaptive.resize(ns_total-1, ds);
  ds_list.resize(num_of_refinements+1, ds);

  for (short i = 0; num_of_refinements; i++)
  {
    ds_list[i+1] = ds_list[i]/(double)(degree_of_refinement);
  }

  int iteration = 0;

  // A-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    double ds_local = ds/pow(degree_of_refinement, num_of_refinements-i);
    if (i == 0)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

  for (int i = 0; i < fns-3; i++) // bulk
  {
    ds_adaptive[iteration] = ds;
    iteration++;
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    double ds_local = ds/pow(degree_of_refinement, i+1);
    if (i == num_of_refinements-1)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }


  // B-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    double ds_local = ds/pow(degree_of_refinement, num_of_refinements-i);
    if (i == 0)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

  for (int i = 0; i < ns-fns-2; i++) // bulk
  {
    ds_adaptive[iteration] = ds;
    iteration++;
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    double ds_local = ds/pow(degree_of_refinement, i+1);
    if (i == num_of_refinements-1)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

  ::dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  dxyz_min = MIN(dxyz[0],dxyz[1],dxyz[2]);
  dxyz_max = MAX(dxyz[0],dxyz[1],dxyz[2]);
#else
  dxyz_min = MIN(dxyz[0],dxyz[1]);
  dxyz_max = MAX(dxyz[0],dxyz[1]);
#endif
}

my_p4est_scft_t::~my_p4est_scft_t()
{
  if (mu_p  != NULL) { ierr = VecDestroy(mu_p);   CHKERRXX(ierr); }
  if (mu_m  != NULL) { ierr = VecDestroy(mu_m);   CHKERRXX(ierr); }
  if (rho_a != NULL) { ierr = VecDestroy(rho_a);  CHKERRXX(ierr); }
  if (rho_b != NULL) { ierr = VecDestroy(rho_b);  CHKERRXX(ierr); }

  if (force_p != NULL) { ierr = VecDestroy(force_p);  CHKERRXX(ierr); }
  if (force_m != NULL) { ierr = VecDestroy(force_m);  CHKERRXX(ierr); }

  if (phi != NULL) for (int i = 0; i < phi->size(); i++) { ierr = VecDestroy(phi->at(i));   CHKERRXX(ierr); }

  for (int i = 0; i < bc_coeffs_a.size(); i++) { ierr = VecDestroy(bc_coeffs_a.at(i));   CHKERRXX(ierr); }
  for (int i = 0; i < bc_coeffs_b.size(); i++) { ierr = VecDestroy(bc_coeffs_b.at(i));   CHKERRXX(ierr); }

  if (rhs != NULL) { ierr = VecDestroy(rhs); CHKERRXX(ierr); }
  if (rhs_old != NULL) { ierr = VecDestroy(rhs_old); CHKERRXX(ierr); }
  if (add_to_rhs != NULL) { ierr = VecDestroy(add_to_rhs); CHKERRXX(ierr); }

  if (mask != NULL) { ierr = VecDestroy(mask); CHKERRXX(ierr); }
  if (integrating_vec != NULL) { ierr = VecDestroy(mask); CHKERRXX(ierr); }

  if (energy_shape_deriv != NULL) { ierr = VecDestroy(energy_shape_deriv); CHKERRXX(ierr); }
  if (energy_shape_deriv_alt != NULL) { ierr = VecDestroy(energy_shape_deriv_alt); CHKERRXX(ierr); }
  if (contact_term_of_energy_shape_deriv != NULL) { ierr = VecDestroy(contact_term_of_energy_shape_deriv); CHKERRXX(ierr); }

  for (int i = 0; i < qf.size(); i++) { ierr = VecDestroy(qf.at(i));   CHKERRXX(ierr); }
  for (int i = 0; i < qb.size(); i++) { ierr = VecDestroy(qb.at(i));   CHKERRXX(ierr); }

  for (int i = 0; i < exp_w_a.size(); i++) { ierr = VecDestroy(exp_w_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < exp_w_b.size(); i++) { ierr = VecDestroy(exp_w_b[i]); CHKERRXX(ierr); }

  for (short i = 0; i < solver_a.size(); ++i) { delete solver_a[i]; }
  for (short i = 0; i < solver_b.size(); ++i) { delete solver_b[i]; }

  for (int i = 0; i < f_a.size(); i++) { ierr = VecDestroy(f_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < f_b.size(); i++) { ierr = VecDestroy(f_b[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_a.size(); i++) { ierr = VecDestroy(g_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_b.size(); i++) { ierr = VecDestroy(g_b[i]); CHKERRXX(ierr); }

  for (int surf_idx = 0; surf_idx < normal.size(); surf_idx++)
  {
    for (int dim = 0; dim < P4EST_DIM; dim++)
    {
      ierr = VecDestroy(normal[surf_idx][dim]); CHKERRXX(ierr);
    }
    delete[] normal[surf_idx];

    ierr = VecDestroy(kappa[surf_idx]); CHKERRXX(ierr);
  }

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, brick);
}

void my_p4est_scft_t::set_parameters(double in_f, double in_XN, int in_ns, int in_num_of_refinements, int in_degree_of_refinement)
{
  f   = in_f;
  XN  = in_XN;
  ns  = in_ns;
  num_of_refinements = in_num_of_refinements;
  degree_of_refinement = in_degree_of_refinement;
  ns_total = ns + 4*(degree_of_refinement - 1)*num_of_refinements;

  // Discretization along the chain
  ds = 1.0 / (double) (ns-1);
  fns = round((double) ns * f);
  fns_adaptive = fns + 2*num_of_refinements*(degree_of_refinement - 1);

  ds_adaptive.resize(ns_total-1, ds);
  ds_list.resize(num_of_refinements+1, ds);

  for (short i = 0; i < num_of_refinements+1; i++)
  {
    ds_list[i] = ds/pow((double)degree_of_refinement, (double)i);
  }

  int iteration = 0;

  // A-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    double ds_local = ds/pow(degree_of_refinement, num_of_refinements-i);
    if (i == 0)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

  for (int i = 0; i < fns-3; i++) // bulk
  {
    ds_adaptive[iteration] = ds;
    iteration++;
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    double ds_local = ds/pow(degree_of_refinement, i+1);
    if (i == num_of_refinements-1)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }


  // B-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    double ds_local = ds/pow(degree_of_refinement, num_of_refinements-i);
    if (i == 0)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

  for (int i = 0; i < ns-fns-2; i++) // bulk
  {
    ds_adaptive[iteration] = ds;
    iteration++;
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    double ds_local = ds/pow(degree_of_refinement, i+1);
    if (i == num_of_refinements-1)
      for (short j = 0; j < degree_of_refinement; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
    else
      for (short j = 0; j < degree_of_refinement-1; j++)
      {
        ds_adaptive[iteration] = ds_local;
        iteration++;
      }
  }

//  for (int i = 0; i < ds_adaptive.size(); ++i)
//  {
//    ierr = PetscPrintf(p4est->mpicomm, "Steps: %d, %e;\n", i, ds_adaptive[i]); CHKERRXX(ierr);
//  }
}

void my_p4est_scft_t::initialize_bc_simple()
{
  // create Vec's

  for (int i = 0; i < bc_coeffs_a.size(); i++) { ierr = VecDestroy(bc_coeffs_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < bc_coeffs_b.size(); i++) { ierr = VecDestroy(bc_coeffs_b[i]); CHKERRXX(ierr); }

  bc_coeffs_a.resize(num_surfaces, NULL);
  bc_coeffs_b.resize(num_surfaces, NULL);

  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_b[i]); CHKERRXX(ierr); }

  // fill out Vec's

//  /* the most naive way */
//  Vec src, out;

//  for (int i = 0; i < num_surfaces; ++i)
//  {
//    ierr = VecGhostGetLocalForm(gamma_a->at(i),         &src); CHKERRXX(ierr);
//    ierr = VecGhostGetLocalForm(bc_coeffs_a.at(i),      &out); CHKERRXX(ierr);
//    ierr = VecCopy(src, out); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(gamma_a->at(i),     &src); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(bc_coeffs_a.at(i),  &out); CHKERRXX(ierr);

//    ierr = VecGhostGetLocalForm(gamma_b->at(i),         &src); CHKERRXX(ierr);
//    ierr = VecGhostGetLocalForm(bc_coeffs_b.at(i),      &out); CHKERRXX(ierr);
//    ierr = VecCopy(src, out); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(gamma_b->at(i),     &src); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(bc_coeffs_b.at(i),  &out); CHKERRXX(ierr);
//  }

  my_p4est_level_set_t ls(ngbd);

  double *mu_m_ptr, *bc_coeff_ptr;

  for (int i = 0; i < num_surfaces; ++i)
  {
    ierr = VecGetArray(bc_coeffs_a[i],  &bc_coeff_ptr); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
#endif

//      bc_coeff_ptr[n] = volume*((*gamma_a->at(i))(x,y)-(*gamma_b->at(i))(x,y))*(+0.5);
//      bc_coeff_ptr[n] = volume*(*gamma_a->at(i))(x,y);
      bc_coeff_ptr[n] = (*gamma_a->at(i))(x,y);
    }
    ierr = VecRestoreArray(bc_coeffs_a[i],  &bc_coeff_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(bc_coeffs_b[i],  &bc_coeff_ptr); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
#endif

//      bc_coeff_ptr[n] = volume*((*gamma_a->at(i))(x,y)-(*gamma_b->at(i))(x,y))*(-0.5);
//      bc_coeff_ptr[n] = volume*(*gamma_b->at(i))(x,y);
      bc_coeff_ptr[n] = (*gamma_b->at(i))(x,y);
    }
    ierr = VecRestoreArray(bc_coeffs_b[i],  &bc_coeff_ptr); CHKERRXX(ierr);

    ls.extend_Over_Interface_TVD(phi_smooth, bc_coeffs_a[i]);
    ls.extend_Over_Interface_TVD(phi_smooth, bc_coeffs_b[i]);
  }

  /* calculate addition to energy from surface tensions */
  singular_part_of_energy = 0;

  // loop through all surfaces
  Vec integrand;
  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

  my_p4est_integration_mls_t integration;
  integration.set_p4est(p4est, nodes);
  integration.set_phi(*phi, *action, color);

  double *integrand_ptr;

  for (int surf_idx = 0; surf_idx < num_surfaces; surf_idx++)
  {
    ierr = VecGetArray(integrand,   &integrand_ptr);    CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
  #ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
  #endif

      integrand_ptr[n] = 0.5*((*gamma_a->at(surf_idx))(x,y)+(*gamma_b->at(surf_idx))(x,y));
    }

    ierr = VecRestoreArray(integrand,   &integrand_ptr);    CHKERRXX(ierr);

    singular_part_of_energy += integration.integrate_over_interface(integrand, surf_idx);
  }

  ierr = VecDestroy(integrand); CHKERRXX(ierr);

}

void my_p4est_scft_t::initialize_bc_smart()
{
  // create Vec's if necessary
  for (int i = 0; i < bc_coeffs_a.size(); i++) { ierr = VecDestroy(bc_coeffs_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < bc_coeffs_b.size(); i++) { ierr = VecDestroy(bc_coeffs_b[i]); CHKERRXX(ierr); }

  bc_coeffs_a.resize(num_surfaces, NULL);
  bc_coeffs_b.resize(num_surfaces, NULL);

  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < num_surfaces; i++) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_b[i]); CHKERRXX(ierr); }

  // fill out Vec's

  /* using mu_m */

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.extend_Over_Interface_TVD(phi_smooth, mu_m);

  double *mu_m_ptr, *bc_coeff_ptr;

  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  for (int i = 0; i < num_surfaces; ++i)
  {
    ierr = VecGetArray(bc_coeffs_a[i],  &bc_coeff_ptr); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
#endif

      bc_coeff_ptr[n] = volume*((*gamma_a->at(i))(x,y)-(*gamma_b->at(i))(x,y))*(-mu_m_ptr[n]/XN+0.5);
    }
    ierr = VecRestoreArray(bc_coeffs_a[i],  &bc_coeff_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(bc_coeffs_b[i],  &bc_coeff_ptr); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
#endif

      bc_coeff_ptr[n] = volume*((*gamma_a->at(i))(x,y)-(*gamma_b->at(i))(x,y))*(-mu_m_ptr[n]/XN-0.5);
    }
    ierr = VecRestoreArray(bc_coeffs_b[i],  &bc_coeff_ptr); CHKERRXX(ierr);

//    ls.extend_Over_Interface_TVD(phi_smooth, bc_coeffs_a[i]);
//    ls.extend_Over_Interface_TVD(phi_smooth, bc_coeffs_b[i]);
  }

  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  /* calculate addition to energy from surface tensions */
  singular_part_of_energy = 0;

  // loop through all surfaces
  Vec integrand;
  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

  double *integrand_ptr;

  my_p4est_integration_mls_t integration;
  integration.set_p4est(p4est, nodes);
  integration.set_phi(*phi, *action, color);

  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  for (int surf_idx = 0; surf_idx < num_surfaces; surf_idx++)
  {
    ierr = VecGetArray(integrand,   &integrand_ptr);    CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
  #ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
  #endif

      integrand_ptr[n] = 0.5*((*gamma_a->at(surf_idx))(x,y)+(*gamma_b->at(surf_idx))(x,y)) + ((*gamma_a->at(surf_idx))(x,y)-(*gamma_b->at(surf_idx))(x,y))*mu_m_ptr[n]/XN;
    }

    ierr = VecRestoreArray(integrand,   &integrand_ptr);    CHKERRXX(ierr);

    singular_part_of_energy += integration.integrate_over_interface(integrand, surf_idx);
  }

  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(integrand); CHKERRXX(ierr);
}

void my_p4est_scft_t::set_force_and_bc(CF_2& f_a_cf, CF_2& f_b_cf, CF_2& g_a_cf, CF_2& g_b_cf)
{
  for (int i = 0; i < f_a.size(); i++) { ierr = VecDestroy(f_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < f_b.size(); i++) { ierr = VecDestroy(f_b[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_a.size(); i++) { ierr = VecDestroy(g_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_b.size(); i++) { ierr = VecDestroy(g_b[i]); CHKERRXX(ierr); }

  f_a.resize(ns_total, NULL);
  f_b.resize(ns_total, NULL);
  g_a.resize(ns_total, NULL);
  g_b.resize(ns_total, NULL);

  for (int i = 0; i < f_a.size(); i++) { ierr = VecCreateGhostNodes(p4est, nodes, &f_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < f_b.size(); i++) { ierr = VecCreateGhostNodes(p4est, nodes, &f_b[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_a.size(); i++) { ierr = VecCreateGhostNodes(p4est, nodes, &g_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < g_b.size(); i++) { ierr = VecCreateGhostNodes(p4est, nodes, &g_b[i]); CHKERRXX(ierr); }

  double s = 0;
  for (int i = 0; i < ns_total; i++)
  {
    f_a_cf.t = s; sample_cf_on_nodes(p4est, nodes, f_a_cf, f_a[i]);
    f_b_cf.t = s; sample_cf_on_nodes(p4est, nodes, f_b_cf, f_b[i]);
    g_a_cf.t = s; sample_cf_on_nodes(p4est, nodes, g_a_cf, g_a[i]);
    g_b_cf.t = s; sample_cf_on_nodes(p4est, nodes, g_b_cf, g_b[i]);
    s += ds_adaptive[i];
  }
}


void my_p4est_scft_t::set_exact_solutions(CF_2& qf_cf, CF_2& qb_cf)
{
  double s = 0;
  for (int i = 0; i < ns_total; i++)
  {
    qf_cf.t = s; sample_cf_on_nodes(p4est, nodes, qf_cf, qf[i]);
    qb_cf.t = s; sample_cf_on_nodes(p4est, nodes, qb_cf, qb[i]);
    s += ds_adaptive[i];
  }
}


void my_p4est_scft_t::set_ic(CF_2& u_exact)
{
  u_exact.t = 0;
  sample_cf_on_nodes(p4est, nodes, u_exact, qf[0]);
  sample_cf_on_nodes(p4est, nodes, u_exact, qb[ns_total-1]);
}

void my_p4est_scft_t::initialize_linear_system()
{
  bc_values.resize(num_surfaces, 0.0);
  bc_types.resize(num_surfaces, ROBIN);

  for (short i = 0; i < solver_a.size(); ++i) { delete solver_a[i]; }
  for (short i = 0; i < solver_b.size(); ++i) { delete solver_b[i]; }

  solver_a.resize(num_of_refinements+1, NULL);
  solver_b.resize(num_of_refinements+1, NULL);

  for (short i = 0; i < num_of_refinements+1; i++)
  {
    // chain propogator a
    solver_a[i] = new my_p4est_poisson_nodes_mls_t(ngbd);
    solver_a[i]->set_geometry(*phi, *action, color);
    solver_a[i]->set_mu(1.0);
    solver_a[i]->wall_value.set(0.0);
    solver_a[i]->set_bc_type(bc_types);
    solver_a[i]->set_diag_add(scheme_coeff/ds_list[i]);
    solver_a[i]->set_bc_coeffs(bc_coeffs_a);
    solver_a[i]->set_bc_values(bc_values);
    solver_a[i]->set_use_taylor_correction(true);
    solver_a[i]->set_keep_scalling(true);
    solver_a[i]->set_kinks_treatment(true);
    solver_a[i]->set_update_ghost_after_solving(false);

    solver_a[i]->reusing_matrix = false;
    solver_a[i]->compute_volumes();
    solver_a[i]->setup_negative_variable_coeff_laplace_matrix_sym();

    // chain propogator b
    solver_b[i] = new my_p4est_poisson_nodes_mls_t(ngbd);
    solver_b[i]->set_geometry(*phi, *action, color);
    solver_b[i]->set_mu(1.0);
    solver_b[i]->wall_value.set(0.0);
    solver_b[i]->set_bc_type(bc_types);
    solver_b[i]->set_diag_add(scheme_coeff/ds_list[i]);
    solver_b[i]->set_bc_coeffs(bc_coeffs_b);
    solver_b[i]->set_bc_values(bc_values);
    solver_b[i]->set_use_taylor_correction(true);
    solver_b[i]->set_keep_scalling(true);
    solver_b[i]->set_kinks_treatment(true);
    solver_b[i]->set_update_ghost_after_solving(false);

    solver_b[i]->reusing_matrix = false;
    solver_b[i]->compute_volumes();
    solver_b[i]->setup_negative_variable_coeff_laplace_matrix_sym();
  }

  /* allocate vectors */

  // chain propogators
  for (int i = 0; i < qf.size(); i++) { ierr = VecDestroy(qf[i]); CHKERRXX(ierr); }
  for (int i = 0; i < qb.size(); i++) { ierr = VecDestroy(qb[i]); CHKERRXX(ierr); }

  qf.resize(ns_total, NULL);
  qb.resize(ns_total, NULL);

  for (int i = 0; i < ns_total; i++)
  {
    ierr = VecDuplicate(phi_smooth, &qf[i]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_smooth, &qb[i]); CHKERRXX(ierr);
  }

  for (int i = 0; i < exp_w_a.size(); i++) { ierr = VecDestroy(exp_w_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < exp_w_b.size(); i++) { ierr = VecDestroy(exp_w_b[i]); CHKERRXX(ierr); }

  exp_w_a.resize(num_of_refinements+1, NULL);
  exp_w_b.resize(num_of_refinements+1, NULL);

  for (int i = 0; i < num_of_refinements+1; i++)
  {
    ierr = VecDuplicate(phi_smooth, &exp_w_a[i]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_smooth, &exp_w_b[i]); CHKERRXX(ierr);
  }

  ierr = VecSet(qf[0],1.0); CHKERRXX(ierr);
  ierr = VecSet(qb[ns_total-1], 1.0); CHKERRXX(ierr);

  if (q_tmp != NULL) { ierr = VecDestroy(q_tmp); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_smooth, &q_tmp); CHKERRXX(ierr);

  if (force_m != NULL)    { ierr = VecDestroy(force_m); CHKERRXX(ierr); }
  if (force_p != NULL)    { ierr = VecDestroy(force_p); CHKERRXX(ierr); }

  if (rhs != NULL)        { ierr = VecDestroy(rhs); CHKERRXX(ierr); }
  if (rhs_old != NULL)    { ierr = VecDestroy(rhs_old); CHKERRXX(ierr); }
  if (add_to_rhs != NULL) { ierr = VecDestroy(add_to_rhs); CHKERRXX(ierr); }

  if (mask != NULL)       { ierr = VecDestroy(mask); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_smooth, &force_m); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_smooth, &force_p); CHKERRXX(ierr);

  ierr = VecDuplicate(phi_smooth, &rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_smooth, &rhs_old); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_smooth, &add_to_rhs); CHKERRXX(ierr);

  ierr = VecDuplicate(phi_smooth, &mask); CHKERRXX(ierr);

  double *mask_ptr;
  ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if (solver_a[0]->is_calc(n)) mask_ptr[n] = 1.0;
    else mask_ptr[n] = 0.0;
  }

  ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void my_p4est_scft_t::solve_for_propogators()
{
  // create exp_w vectors
  double *mu_p_ptr;
  double *mu_m_ptr;
  double *exp_w_a_ptr;
  double *exp_w_b_ptr;

  ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

  for (int i = 0; i < num_of_refinements+1; ++i)
  {
    ierr = VecGetArray(exp_w_a[i], &exp_w_a_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(exp_w_b[i], &exp_w_b_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (solver_a[0]->is_calc(n))
      {
        exp_w_a_ptr[n] = exp(-0.5*(mu_p_ptr[n]-mu_m_ptr[n])*ds_list[i]);
        exp_w_b_ptr[n] = exp(-0.5*(mu_p_ptr[n]+mu_m_ptr[n])*ds_list[i]);
      } else {
        exp_w_a_ptr[n] = 1.0;
        exp_w_b_ptr[n] = 1.0;
      }

      //    if (exp_w_a_ptr[n] != exp_w_a_ptr[n]) std::cout << "nan!\n";
      //    if (exp_w_b_ptr[n] != exp_w_b_ptr[n]) std::cout << "nan!\n";
    }

    ierr = VecRestoreArray(exp_w_a[i], &exp_w_a_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(exp_w_b[i], &exp_w_b_ptr); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);

//  ierr = VecPointwiseMult(exp_w_a, exp_w_a, mask); CHKERRXX(ierr);
//  ierr = VecPointwiseMult(exp_w_b, exp_w_b, mask); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(exp_w_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(exp_w_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(exp_w_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(exp_w_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // forward propagator
//  for (int is = 1; is < fns; is++)
//  {
//    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a); CHKERRXX(ierr);

//    diffusion_step(solver_a, qf[is], q_tmp);

//    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_a); CHKERRXX(ierr);
//  }

//  for (int is = fns; is < ns; is++)
//  {
//    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b); CHKERRXX(ierr);

//    diffusion_step(solver_b, qf[is], q_tmp);

//    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_b); CHKERRXX(ierr);
//  }

//  // backward propagator
//  for (int is = ns-2; is > fns-2; is--)
//  {
//    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b); CHKERRXX(ierr);

//    diffusion_step(solver_b, qb[is], q_tmp);

//    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_b); CHKERRXX(ierr);
//  }

//  for (int is = fns-2; is > -1; is--)
//  {
//    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
//    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a); CHKERRXX(ierr);

//    diffusion_step(solver_a, qb[is], q_tmp);

//    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_a); CHKERRXX(ierr);
//  }

  /* FORWARD PROPAGATOR */
  int is = 0;

  // A-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    int num_of_solver = num_of_refinements-i;

    int steps = degree_of_refinement;

    if (i == 0) steps = degree_of_refinement;
    else        steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is++;
      ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[num_of_solver]); CHKERRXX(ierr);

      diffusion_step(solver_a[num_of_solver], qf[is], q_tmp, ds_list[num_of_solver], is, true);

      ierr = VecPointwiseMult(qf[is], qf[is], exp_w_a[num_of_solver]); CHKERRXX(ierr);
    }
  }

  for (int i = 0; i < fns-3; i++) // bulk
  {
    is++;

    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[0]); CHKERRXX(ierr);

    diffusion_step(solver_a[0], qf[is], q_tmp, ds, is, true);

    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_a[0]); CHKERRXX(ierr);
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    int num_of_solver = i+1;

    int steps = degree_of_refinement;

    if (i == num_of_refinements-1)  steps = degree_of_refinement;
    else                            steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is++;
      ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[num_of_solver]); CHKERRXX(ierr);

      diffusion_step(solver_a[num_of_solver], qf[is], q_tmp, ds_list[num_of_solver], is, true);

      ierr = VecPointwiseMult(qf[is], qf[is], exp_w_a[num_of_solver]); CHKERRXX(ierr);
    }
  }

  // B-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    int num_of_solver = num_of_refinements-i;

    int steps = degree_of_refinement;

    if (i == 0) steps = degree_of_refinement;
    else        steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is++;
      ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[num_of_solver]); CHKERRXX(ierr);
      diffusion_step(solver_b[num_of_solver], qf[is], q_tmp, ds_list[num_of_solver], is, true);
      ierr = VecPointwiseMult(qf[is], qf[is], exp_w_b[num_of_solver]); CHKERRXX(ierr);
    }
  }

  for (int i = 0; i < ns-fns-2; i++) // bulk
  {
    is++;

    ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[0]); CHKERRXX(ierr);

    diffusion_step(solver_b[0], qf[is], q_tmp, ds, is, true);

    ierr = VecPointwiseMult(qf[is], qf[is], exp_w_b[0]); CHKERRXX(ierr);
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    int num_of_solver = i+1;

    int steps = degree_of_refinement;

    if (i == num_of_refinements-1)  steps = degree_of_refinement;
    else                            steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is++;
      ierr = VecCopy(qf[is-1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[num_of_solver]); CHKERRXX(ierr);
      diffusion_step(solver_b[num_of_solver], qf[is], q_tmp, ds_list[num_of_solver], is, true);
      ierr = VecPointwiseMult(qf[is], qf[is], exp_w_b[num_of_solver]); CHKERRXX(ierr);
    }
  }

  /* BACKWARD PROPAGATOR */
  is = ns_total-1;

  // B-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    int num_of_solver = num_of_refinements-i;

    int steps = degree_of_refinement;

    if (i == 0) steps = degree_of_refinement;
    else        steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is--;
      ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[num_of_solver]); CHKERRXX(ierr);
      diffusion_step(solver_b[num_of_solver], qb[is], q_tmp, ds_list[num_of_solver], is, false);
      ierr = VecPointwiseMult(qb[is], qb[is], exp_w_b[num_of_solver]); CHKERRXX(ierr);
    }
  }

  for (int i = 0; i < ns-fns-2; i++) // bulk
  {
    is--;

    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[0]); CHKERRXX(ierr);

    diffusion_step(solver_b[0], qb[is], q_tmp, ds, is, false);

    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_b[0]); CHKERRXX(ierr);
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    int num_of_solver = i+1;

    int steps = degree_of_refinement;

    if (i == num_of_refinements-1)  steps = degree_of_refinement;
    else                            steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is--;
      ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_b[num_of_solver]); CHKERRXX(ierr);
      diffusion_step(solver_b[num_of_solver], qb[is], q_tmp, ds_list[num_of_solver], is, false);
      ierr = VecPointwiseMult(qb[is], qb[is], exp_w_b[num_of_solver]); CHKERRXX(ierr);
    }
  }

  // A-part
  for (short i = 0; i < num_of_refinements; i++) // starting refinement
  {
    int num_of_solver = num_of_refinements-i;

    int steps = degree_of_refinement;

    if (i == 0) steps = degree_of_refinement;
    else        steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is--;
      ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[num_of_solver]); CHKERRXX(ierr);
      diffusion_step(solver_a[num_of_solver], qb[is], q_tmp, ds_list[num_of_solver], is, false);
      ierr = VecPointwiseMult(qb[is], qb[is], exp_w_a[num_of_solver]); CHKERRXX(ierr);
    }
  }

  for (int i = 0; i < fns-3; i++) // bulk
  {
    is--;

    ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
    ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[0]); CHKERRXX(ierr);

    diffusion_step(solver_a[0], qb[is], q_tmp, ds, is, false);

    ierr = VecPointwiseMult(qb[is], qb[is], exp_w_a[0]); CHKERRXX(ierr);
  }

  for (short i = 0; i < num_of_refinements; i++) // ending refinement
  {
    int num_of_solver = i+1;

    int steps = degree_of_refinement;

    if (i == num_of_refinements-1)  steps = degree_of_refinement;
    else                            steps = degree_of_refinement-1;

    for (short j = 0; j < steps; j++)
    {
      is--;
      ierr = VecCopy(qb[is+1], q_tmp); CHKERRXX(ierr);
      ierr = VecPointwiseMult(q_tmp, q_tmp, exp_w_a[num_of_solver]); CHKERRXX(ierr);

      diffusion_step(solver_a[num_of_solver], qb[is], q_tmp, ds_list[num_of_solver], is, false);

      ierr = VecPointwiseMult(qb[is], qb[is], exp_w_a[num_of_solver]); CHKERRXX(ierr);
    }
  }

}

void my_p4est_scft_t::diffusion_step(my_p4est_poisson_nodes_mls_t *solver, Vec &sol, Vec &sol_nm1, double ds_local, int is, bool forward)
{
  std::vector<double> bc_vals_zero(num_surfaces, 0.0);
  std::vector<Vec> bc_vals_vecs;
//  if (use_cn_scheme)
//  {
//    ierr = VecSet(rhs, 0.0); CHKERRXX(ierr);

//    // Set up RHS
//    ierr = VecPointwiseMult(add_to_rhs, solver->node_vol, sol_nm1);           CHKERRXX(ierr);
//    ierr = VecPointwiseDivide(add_to_rhs, add_to_rhs, solver->scalling);  CHKERRXX(ierr);
//    ierr = VecScale(add_to_rhs, -4.0/ds_local);                         CHKERRXX(ierr);
//    ierr = MatMultAdd(solver->A, sol_nm1, add_to_rhs, add_to_rhs);  CHKERRXX(ierr);

//    // Calculate RHS as for Poisson eqn
//    solver->set_rhs(rhs);
//    solver->setup_negative_variable_coeff_laplace_rhsvec_sym();

//    ierr = VecPointwiseMult(add_to_rhs, add_to_rhs, mask); CHKERRXX(ierr);

//    ierr = VecAXPBY(solver->rhs, -1.0, 2.0, add_to_rhs); CHKERRXX(ierr);
//  } else {
    ierr = VecCopy(sol_nm1, rhs); CHKERRXX(ierr);
    ierr = VecScale(rhs, 1.0/ds_local); CHKERRXX(ierr);
    if (forward) {
      if (is < fns_adaptive)
        ierr = VecAXPBY(rhs, 1.0, 1.0, f_a[is]);
      else
        ierr = VecAXPBY(rhs, 1.0, 1.0, f_b[is]);
    }
    solver->set_rhs(rhs);
    if (forward) {
      if (is < fns_adaptive)
        bc_vals_vecs.push_back(g_a[is]);
      else
        bc_vals_vecs.push_back(g_b[is]);

      solver->set_bc_values(bc_vals_vecs);
    } else {
      solver->set_bc_values(bc_vals_zero);
    }
    solver->setup_negative_variable_coeff_laplace_rhsvec_sym();
//  }

  // Solve linear system
  ierr = VecPointwiseMult(sol, sol, mask); CHKERRXX(ierr);
  solver->solve_linear_system(sol, false);
}


void my_p4est_scft_t::set_geometry(std::vector<Vec> &in_phi, std::vector<action_t> &in_action)
{
  phi = &in_phi;
  action = &in_action;
  num_surfaces = in_phi.size();
  color.clear();
  for (int i=0; i<num_surfaces; ++i) color.push_back(i);

  // create smooth version of the domain (for extapolation purposes)

  if (phi_smooth != NULL) { ierr = VecDestroy(phi_smooth); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi->at(0), &phi_smooth); CHKERRXX(ierr);

  double *phi_smooth_ptr;
  ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

  std::vector<double *> phi_ptr(num_surfaces, NULL);

  for (int i = 0; i < num_surfaces; i++)
  {
    ierr = VecGetArray(phi->at(i), &phi_ptr[i]); CHKERRXX(ierr);
  }

  double epsilon = 10.*dxyz_min*dxyz_min;

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double phi_total = phi_ptr[0][n];
    double phi_current = phi_ptr[0][n];

    for (int i_phi = 1; i_phi < num_surfaces; i_phi++)
    {
      phi_current = phi_ptr[i_phi][n];

      if (action->at(i_phi) == INTERSECTION)
      {
        phi_total = 0.5*(phi_total+phi_current+sqrt(SQR(phi_total-phi_current)+epsilon));
      }
      else if (action->at(i_phi) == ADDITION)
      {
        phi_total = 0.5*(phi_total+phi_current-(sqrt(SQR(phi_total-phi_current)+epsilon)-epsilon/sqrt(SQR(phi_total-phi_current)+epsilon)));
      }
    }

    phi_smooth_ptr[n] = phi_total;
  }

  for (int i = 0; i < num_surfaces; i++)
  {
    ierr = VecRestoreArray(phi->at(i), &phi_ptr[i]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

  Vec ones;
  ierr = VecDuplicate(phi_smooth, &ones); CHKERRXX(ierr);
  ierr = VecSet(ones, 1.0); CHKERRXX(ierr);

  // TODO: move this stuff to set_geometry
  assemble_integrating_vec();
  volume = integrate_over_domain_fast(ones);
  ierr = VecDestroy(ones); CHKERRXX(ierr);

  ierr = PetscPrintf(p4est->mpicomm, "new volume %e\n", volume); CHKERRXX(ierr);

//  my_p4est_integration_mls_t integration;
//  integration.set_p4est(p4est, nodes);
//  integration.set_phi(*phi, *action, color);

//  volume = integration.measure_of_domain();

}

void my_p4est_scft_t::calculate_densities()
{
  // calculate densities
  double *time_integrand = new double [ns_total];

  double **qf_ptr = new double * [ns_total];
  double **qb_ptr = new double * [ns_total];

  double *rho_a_ptr;
  double *rho_b_ptr;

  ierr = VecGetArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  for (int is = 0; is < ns_total; is++)
  {
    ierr = VecGetArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
    ierr = VecGetArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
  }


  // calculate densities only for local nodes
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if (solver_a[0]->is_calc(n))
    {
      for (int is = 0; is < ns_total; is++)
        time_integrand[is] = qf_ptr[is][n]*qb_ptr[is][n];

      rho_a_ptr[n] = compute_rho_a(time_integrand);
      rho_b_ptr[n] = compute_rho_b(time_integrand);
    } else {
      rho_a_ptr[n] = 0.0;
      rho_b_ptr[n] = 0.0;
    }
  }

  ierr = VecRestoreArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  Vec integrand;
  ierr = VecCreateGhostNodes(p4est, nodes, &integrand); CHKERRXX(ierr);

  double *integrand_ptr;
  ierr = VecGetArray(integrand, &integrand_ptr); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    integrand_ptr[n] = qf_ptr[0][n]*qf_ptr[ns_total-1][n];
//    integrand_ptr[n] = qf_ptr[ns_total-1][n];
//    integrand_ptr[n] = 1.0;
  }

  ierr = VecRestoreArray(integrand, &integrand_ptr); CHKERRXX(ierr);

  Q = integrate_over_domain_fast(integrand);

  ierr = VecDestroy(integrand); CHKERRXX(ierr);

  for (int is = 0; is < ns_total; is++)
  {
    ierr = VecRestoreArray(qf[is], &qf_ptr[is]); CHKERRXX(ierr);
    ierr = VecRestoreArray(qb[is], &qb_ptr[is]); CHKERRXX(ierr);
  }


//  Q = (integrate_over_domain_fast(rho_a) + integrate_over_domain_fast(rho_b));

//  ierr = VecScale(rho_a, 1.0/Q); CHKERRXX(ierr);
//  ierr = VecScale(rho_b, 1.0/Q); CHKERRXX(ierr);

  delete[] time_integrand;
  delete[] qf_ptr;
  delete[] qb_ptr;

  double mu_m_sqrd_int = integrate_over_domain_fast_squared(mu_m);
  double mu_p_int = integrate_over_domain_fast(mu_p);

//  H = 1.0*(mu_m_sqrd_int/XN - mu_p_int)/volume - 1.0*log(Q);
//  H = -log(Q);
  H = Q;

}

double my_p4est_scft_t::compute_rho_a(double *integrand)
{
//  double result = 0.5*integrand[0];
//  for (int i = 1; i < fns; i++) result += integrand[i];
//  double result = 0.5*integrand[0]+0.5*integrand[fns-1];
//  for (int i = 1; i < fns-1; i++) result += integrand[i];
//  return result*ds;

  double result = 0;
  for (int i = 0; i < fns_adaptive-1; i++)
    result += 0.5*(integrand[i+1]+integrand[i])*ds_adaptive[i];
  return result;
}

double my_p4est_scft_t::compute_rho_b(double *integrand)
{
//  double result = 0.5*integrand[ns-1];
//  for (int i = fns; i < ns-1; i++) result += integrand[i];
//  double result = 0.5*integrand[ns-1]+0.5*integrand[fns-1];
//  for (int i = fns; i < ns-1; i++) result += integrand[i];
//  return result*ds;

  double result = 0;
  for (int i = fns_adaptive-1; i < ns_total-1; i++)
    result += 0.5*(integrand[i+1]+integrand[i])*ds_adaptive[i];
  return result;
}

void my_p4est_scft_t::save_VTK(int compt)
{
  ierr = VecGhostUpdateBegin(qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(ngbd);

//  ls.extend_Over_Interface_TVD(phi_smooth, rho_a);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_b);

//  ls.extend_Over_Interface_TVD(phi_smooth, mu_p);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_m);

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

  double *phi_p, *rho_a_p, *rho_b_p, *mu_p_p, *mu_m_p, *mask_p, *shape_derivative_p, *shape_derivative_alt_p;

//  ierr = VecGetArray(phi->at(0), &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(qf[ns-1], &rho_a_p); CHKERRXX(ierr);
//  ierr = VecGetArray(qb[0], &rho_b_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(rho_a, &rho_a_p); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_p); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m, &mu_m_p); CHKERRXX(ierr);
  ierr = VecGetArray(qf[ns_total-1], &mu_p_p); CHKERRXX(ierr);
  ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
  ierr = VecGetArray(energy_shape_deriv, &shape_derivative_p); CHKERRXX(ierr);
  ierr = VecGetArray(energy_shape_deriv_alt, &shape_derivative_alt_p); CHKERRXX(ierr);


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

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         8, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "rho_a", rho_a_p,
                         VTK_POINT_DATA, "rho_b", rho_b_p,
                         VTK_POINT_DATA, "mu_m", mu_m_p,
                         VTK_POINT_DATA, "mu_p", mu_p_p,
                         VTK_POINT_DATA, "mask", mask_p,
                         VTK_POINT_DATA, "shape derivative", shape_derivative_p,
                         VTK_POINT_DATA, "shape derivative alt", shape_derivative_alt_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

//  ierr = VecRestoreArray(phi->at(0), &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(qf[ns-1], &rho_a_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(qb[0], &rho_b_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_smooth, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_a, &rho_a_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m, &mu_m_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[ns_total-1], &mu_p_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(energy_shape_deriv, &shape_derivative_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(energy_shape_deriv_alt, &shape_derivative_alt_p); CHKERRXX(ierr);

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

  ierr = VecRestoreArray(integrating_vec, &int_vec_ptr); CHKERRXX(ierr);


  /* post processing step: multiply every weight by fraction */
  /* TODO: here we just use Vec node_vol from one of the solvers,
   * but in the future it should be replaced with a proper loop through nodes
   * and calculation of volume fraction for each node
   */
  my_p4est_poisson_nodes_mls_t aux_solver(ngbd);
  aux_solver.set_geometry(*phi, *action, color);
  aux_solver.compute_volumes();
  ierr = VecPointwiseMult(integrating_vec, integrating_vec, aux_solver.node_vol); CHKERRXX(ierr);
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

void my_p4est_scft_t::compute_energy_shape_derivative(int phi_idx)
{
  if (energy_shape_deriv != NULL) { ierr = VecDestroy(energy_shape_deriv); CHKERRXX(ierr); }
  ierr = VecDuplicate(mu_m, &energy_shape_deriv); CHKERRXX(ierr);

  Vec energy_shape_deriv_tmp;
  ierr = VecDuplicate(mu_m, &energy_shape_deriv_tmp); CHKERRXX(ierr);

  // compute velocity only on local nodes
  double *energy_shape_deriv_ptr;
  double *qf_start_ptr, *qf_end_ptr, *rho_a_ptr, *bc_coeffs_a_ptr, *mu_m_ptr;
  double *qb_start_ptr, *qb_end_ptr, *rho_b_ptr, *bc_coeffs_b_ptr, *mu_p_ptr;
  double *kappa_ptr;

//  ierr = VecGhostUpdateBegin(qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);



  // extend over smoothed interface
  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi_smooth, qf[ns_total-1]);
//  ls.extend_Over_Interface_TVD(phi_smooth, qb[0]);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_a);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_b);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_m);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_p);

//  my_p4est_integration_mls_t integration;
//  integration.set_p4est(p4est, nodes);
//  integration.set_phi(*phi, *action, color);

//  Vec integrand;
//  ierr = VecDuplicate(phi_smooth, &integrand); CHKERRXX(ierr);

//  double *integrand_ptr;

//  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(integrand, &integrand_ptr); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    integrand_ptr[n] = mu_m_ptr[n]*mu_m_ptr[n];
//  }

//  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(integrand, &integrand_ptr); CHKERRXX(ierr);

//  double mu_m_sqrd_int = integration.integrate_over_domain(integrand);
//  double mu_p_int = integration.integrate_over_domain(mu_p);

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

  // calculate laplace of total density
  Vec rho_total, rho_total_xx, rho_total_yy;
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_xx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_yy); CHKERRXX(ierr);

  double *rho_total_ptr;
  ierr = VecGetArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    rho_total_ptr[n] = rho_a_ptr[n]+rho_b_ptr[n];
  }
  ierr = VecRestoreArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi_smooth, rho_total);



  ngbd->second_derivatives_central(rho_total, rho_total_xx, rho_total_yy);


  double *phi_ptr;
  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    phi_ptr[n] += 1.0*dxyz_min;
  }
  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi_smooth, rho_total_xx);
  ls.extend_Over_Interface_TVD(phi_smooth, rho_total_yy);

  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    phi_ptr[n] -= 1.0*dxyz_min;
  }
  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);


  double *rho_total_xx_ptr;
  double *rho_total_yy_ptr;
  ierr = VecGetArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);

//  double mu_m_sqrd_int = integrate_over_domain_fast_squared(mu_m);
//  double mu_p_int = integrate_over_domain_fast(mu_p);

  // calculate F_a and F_b
  Vec F_a; ierr = VecCreateGhostNodes(p4est, nodes, &F_a); CHKERRXX(ierr);
  Vec F_b; ierr = VecCreateGhostNodes(p4est, nodes, &F_b); CHKERRXX(ierr);

  double *F_a_ptr; ierr = VecGetArray(F_a, &F_a_ptr); CHKERRXX(ierr);
  double *F_b_ptr; ierr = VecGetArray(F_b, &F_b_ptr); CHKERRXX(ierr);

  std::vector<double *> f_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(f_a[i], &f_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> f_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(f_b[i], &f_b_ptr[i]); CHKERRXX(ierr); }

  std::vector<double *> qb_ptr(ns_total, NULL);
  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = f_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = f_b_ptr[i][n]*qb_ptr[i][n];
    }

    F_a_ptr[n] = compute_rho_a(integrand_a.data());
    F_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(f_a[i], &f_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(f_b[i], &f_b_ptr[i]); CHKERRXX(ierr); }

  // calculate G_a and G_b
  Vec G_a; ierr = VecCreateGhostNodes(p4est, nodes, &G_a); CHKERRXX(ierr);
  Vec G_b; ierr = VecCreateGhostNodes(p4est, nodes, &G_b); CHKERRXX(ierr);

  double *G_a_ptr; ierr = VecGetArray(G_a, &G_a_ptr); CHKERRXX(ierr);
  double *G_b_ptr; ierr = VecGetArray(G_b, &G_b_ptr); CHKERRXX(ierr);

  std::vector<double *> g_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(g_a[i], &g_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> g_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(g_b[i], &g_b_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = g_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = g_b_ptr[i][n]*qb_ptr[i][n];
    }

    G_a_ptr[n] = compute_rho_a(integrand_a.data());
    G_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(g_a[i], &g_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(g_b[i], &g_b_ptr[i]); CHKERRXX(ierr); }

  // calculate normal derivatives of Robin coefficients
  Vec dn_bc_coeffs_a; ierr = VecCreateGhostNodes(p4est, nodes, &dn_bc_coeffs_a); CHKERRXX(ierr);
  Vec dn_bc_coeffs_b; ierr = VecCreateGhostNodes(p4est, nodes, &dn_bc_coeffs_b); CHKERRXX(ierr);

  Vec bc_coeffs_a_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_a_d[dim]); CHKERRXX(ierr); }
  Vec bc_coeffs_b_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_b_d[dim]); CHKERRXX(ierr); }

  ngbd->first_derivatives_central(bc_coeffs_a[phi_idx], bc_coeffs_a_d);
  ngbd->first_derivatives_central(bc_coeffs_b[phi_idx], bc_coeffs_b_d);

  double *normal_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  double *bc_coeffs_a_d_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_a_d[dim], &bc_coeffs_a_d_ptr[dim]); CHKERRXX(ierr); }
  double *bc_coeffs_b_d_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_b_d[dim], &bc_coeffs_b_d_ptr[dim]); CHKERRXX(ierr); }

  double *dn_bc_coeffs_a_ptr; ierr = VecGetArray(dn_bc_coeffs_a, &dn_bc_coeffs_a_ptr); CHKERRXX(ierr);
  double *dn_bc_coeffs_b_ptr; ierr = VecGetArray(dn_bc_coeffs_b, &dn_bc_coeffs_b_ptr); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    dn_bc_coeffs_a_ptr[n] = bc_coeffs_a_d_ptr[0][n] * normal_ptr[0][n] + bc_coeffs_a_d_ptr[1][n] * normal_ptr[1][n];
    dn_bc_coeffs_b_ptr[n] = bc_coeffs_b_d_ptr[0][n] * normal_ptr[0][n] + bc_coeffs_b_d_ptr[1][n] * normal_ptr[1][n];
  }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_a_d[dim], &bc_coeffs_a_d_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_b_d[dim], &bc_coeffs_b_d_ptr[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(bc_coeffs_a_d[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(bc_coeffs_b_d[dim]); CHKERRXX(ierr); }

  // compute dn_g_a and dn_b_b
  std::vector<Vec> gn_a(ns_total, NULL); for (int i = 0; i < ns_total; ++i) { ierr = VecCreateGhostNodes(p4est, nodes, &gn_a[i]); CHKERRXX(ierr); }
  std::vector<Vec> gn_b(ns_total, NULL); for (int i = 0; i < ns_total; ++i) { ierr = VecCreateGhostNodes(p4est, nodes, &gn_b[i]); CHKERRXX(ierr); }

  Vec g_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &g_d[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  double *g_d_ptr[P4EST_DIM];
  double *gn_ptr;

  for (int i = 0; i < ns_total; ++i)
  {
    ngbd->first_derivatives_central(g_a[i], g_d);

    ierr = VecGetArray(gn_a[i], &gn_ptr); CHKERRXX(ierr);
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      gn_ptr[n] = g_d_ptr[0][n] * normal_ptr[0][n] + g_d_ptr[1][n] * normal_ptr[1][n];
    }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    ierr = VecRestoreArray(gn_a[i], &gn_ptr); CHKERRXX(ierr);

    ngbd->first_derivatives_central(g_b[i], g_d);

    ierr = VecGetArray(gn_b[i], &gn_ptr); CHKERRXX(ierr);
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      gn_ptr[n] = g_d_ptr[0][n] * normal_ptr[0][n] + g_d_ptr[1][n] * normal_ptr[1][n];
    }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    ierr = VecRestoreArray(gn_b[i], &gn_ptr); CHKERRXX(ierr);
  }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(g_d[dim]); CHKERRXX(ierr); }

  // compute Gn_a and Gn_b
  Vec Gn_a; ierr = VecCreateGhostNodes(p4est, nodes, &Gn_a); CHKERRXX(ierr);
  Vec Gn_b; ierr = VecCreateGhostNodes(p4est, nodes, &Gn_b); CHKERRXX(ierr);

  double *Gn_a_ptr; ierr = VecGetArray(Gn_a, &Gn_a_ptr); CHKERRXX(ierr);
  double *Gn_b_ptr; ierr = VecGetArray(Gn_b, &Gn_b_ptr); CHKERRXX(ierr);

  std::vector<double *> gn_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(gn_a[i], &gn_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> gn_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(gn_b[i], &gn_b_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = gn_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = gn_b_ptr[i][n]*qb_ptr[i][n];
    }

    Gn_a_ptr[n] = compute_rho_a(integrand_a.data());
    Gn_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(gn_a[i], &gn_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(gn_b[i], &gn_b_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; ++i) { ierr = VecDestroy(gn_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; ++i) { ierr = VecDestroy(gn_b[i]); CHKERRXX(ierr); }

  ierr = VecGetArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecGetArray(qf[ns_total-1],        &qf_end_ptr             ); CHKERRXX(ierr);
  ierr = VecGetArray(qb[ns_total-1],        &qb_end_ptr             ); CHKERRXX(ierr);
  ierr = VecGetArray(qf[0],                 &qf_start_ptr           ); CHKERRXX(ierr);
  ierr = VecGetArray(qb[0],                 &qb_start_ptr           ); CHKERRXX(ierr);
  ierr = VecGetArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
  ierr = VecGetArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
  ierr = VecGetArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
#endif

    energy_shape_deriv_ptr[n] =
        0.5*(qf_end_ptr[n]*qb_end_ptr[n] + qf_start_ptr[n]*qb_start_ptr[n])
        - 0.5*(rho_total_xx_ptr[n] + rho_total_yy_ptr[n] - F_a_ptr[n] - F_b_ptr[n])
        - 1.0*(kappa_ptr[n]-2.0*bc_coeffs_a_ptr[n])*(bc_coeffs_a_ptr[n]*rho_a_ptr[n] - G_a_ptr[n]) - dn_bc_coeffs_a_ptr[n]*rho_a_ptr[n] + Gn_a_ptr[n]
        - 1.0*(kappa_ptr[n]-2.0*bc_coeffs_b_ptr[n])*(bc_coeffs_b_ptr[n]*rho_b_ptr[n] - G_b_ptr[n]) - dn_bc_coeffs_b_ptr[n]*rho_a_ptr[n] + Gn_b_ptr[n];

    energy_shape_deriv_ptr[n] *= 1.0;

//    energy_shape_deriv_ptr[n] =
//        0.5*(qf_end_ptr[n]*qb_end_ptr[n] + qf_start_ptr[n]*qb_start_ptr[n])
//        - 0.5*(rho_total_xx_ptr[n] + rho_total_yy_ptr[n]);

//    energy_shape_deriv_ptr[n] =
//        0.5*(qf_end_ptr[n]*qb_end_ptr[n] + qf_start_ptr[n]*qb_start_ptr[n]) + 0.5*(F_a_ptr[n] + F_b_ptr[n]);

//    energy_shape_deriv_ptr[n] = qf_end_ptr[n]*qf_start_ptr[n];
//    energy_shape_deriv_ptr[n] = 1.0 + F_a_ptr[n] + F_b_ptr[n];
//    energy_shape_deriv_ptr[n] = qf_end_ptr[n];
//    energy_shape_deriv_ptr[n] = 1.0;
  }

  ierr = VecRestoreArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);


  ierr = VecRestoreArray(F_a, &F_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(F_b, &F_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(G_a, &G_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(G_b, &G_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(dn_bc_coeffs_a, &dn_bc_coeffs_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(dn_bc_coeffs_b, &dn_bc_coeffs_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(Gn_a, &Gn_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Gn_b, &Gn_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[ns_total-1],        &qf_end_ptr             ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[ns_total-1],        &qb_end_ptr             ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[0],                 &qf_start_ptr           ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[0],                 &qb_start_ptr           ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);

  // sync the derivative among procs
  ierr = VecGhostUpdateBegin(energy_shape_deriv_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (energy_shape_deriv_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // extrapolate from moving moving interface

//  double *phi_ptr;
//  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    phi_ptr[n] += 1.1*dxyz_min;
//  }
//  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi_smooth, energy_shape_deriv_tmp);

//  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    phi_ptr[n] -= 1.1*dxyz_min;
//  }
//  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi->at(phi_idx), energy_shape_deriv_tmp, energy_shape_deriv);

  ierr = VecDestroy(energy_shape_deriv_tmp); CHKERRXX(ierr);


//  ls.extend_Over_Interface_TVD(phi_smooth, energy_shape_deriv_tmp);
//  ierr = VecDestroy(energy_shape_deriv); CHKERRXX(ierr);
//  energy_shape_deriv = energy_shape_deriv_tmp;


  ierr = VecDestroy(rho_total); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_xx); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_yy); CHKERRXX(ierr);

  ierr = VecDestroy(F_a); CHKERRXX(ierr);
  ierr = VecDestroy(F_b); CHKERRXX(ierr);

  ierr = VecDestroy(G_a); CHKERRXX(ierr);
  ierr = VecDestroy(G_b); CHKERRXX(ierr);

  ierr = VecDestroy(dn_bc_coeffs_a); CHKERRXX(ierr);
  ierr = VecDestroy(dn_bc_coeffs_b); CHKERRXX(ierr);

  ierr = VecDestroy(Gn_a); CHKERRXX(ierr);
  ierr = VecDestroy(Gn_b); CHKERRXX(ierr);

}

void my_p4est_scft_t::compute_energy_shape_derivative_alt(int phi_idx)
{
  if (energy_shape_deriv_alt != NULL) { ierr = VecDestroy(energy_shape_deriv_alt); CHKERRXX(ierr); }
  ierr = VecDuplicate(mu_m, &energy_shape_deriv_alt); CHKERRXX(ierr);

  Vec energy_shape_deriv_tmp;
  ierr = VecDuplicate(mu_m, &energy_shape_deriv_tmp); CHKERRXX(ierr);

  // compute velocity only on local nodes
  double *energy_shape_deriv_ptr;
  double *qf_start_ptr, *qf_end_ptr, *rho_a_ptr, *bc_coeffs_a_ptr, *mu_m_ptr;
  double *qb_start_ptr, *qb_end_ptr, *rho_b_ptr, *bc_coeffs_b_ptr, *mu_p_ptr;
  double *kappa_ptr;

//  ierr = VecGhostUpdateBegin(qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (qf[ns_total-1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (qb[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_a, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (rho_b, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


//  double *phi_ptr;
//  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    phi_ptr[n] += 1.0*dxyz_min;
//  }
//  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

  // extend over smoothed interface
  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi_smooth, qf[ns_total-1]);
//  ls.extend_Over_Interface_TVD(phi_smooth, qb[0]);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_a);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_b);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_m);
//  ls.extend_Over_Interface_TVD(phi_smooth, mu_p);

//  my_p4est_integration_mls_t integration;
//  integration.set_p4est(p4est, nodes);
//  integration.set_phi(*phi, *action, color);

//  Vec integrand;
//  ierr = VecDuplicate(phi_smooth, &integrand); CHKERRXX(ierr);

//  double *integrand_ptr;

//  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecGetArray(integrand, &integrand_ptr); CHKERRXX(ierr);

//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    integrand_ptr[n] = mu_m_ptr[n]*mu_m_ptr[n];
//  }

//  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
//  ierr = VecRestoreArray(integrand, &integrand_ptr); CHKERRXX(ierr);

//  double mu_m_sqrd_int = integration.integrate_over_domain(integrand);
//  double mu_p_int = integration.integrate_over_domain(mu_p);

//  ierr = VecDestroy(integrand); CHKERRXX(ierr);

  // calculate laplace of total density
  Vec rho_total, rho_total_xx, rho_total_yy;
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_xx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_total_yy); CHKERRXX(ierr);

  double *rho_total_ptr;
  ierr = VecGetArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    rho_total_ptr[n] = rho_a_ptr[n]+rho_b_ptr[n];
  }
  ierr = VecRestoreArray(rho_total, &rho_total_ptr); CHKERRXX(ierr);

  ls.extend_Over_Interface_TVD(phi_smooth, rho_total);

//  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
//  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//  {
//    phi_ptr[n] -= 1.0*dxyz_min;
//  }
//  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

  ngbd->second_derivatives_central(rho_total, rho_total_xx, rho_total_yy);

//  ls.extend_Over_Interface_TVD(phi_smooth, rho_total_xx);
//  ls.extend_Over_Interface_TVD(phi_smooth, rho_total_yy);

  double *rho_total_xx_ptr;
  double *rho_total_yy_ptr;
  ierr = VecGetArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);

//  double mu_m_sqrd_int = integrate_over_domain_fast_squared(mu_m);
//  double mu_p_int = integrate_over_domain_fast(mu_p);

  // calculate F_a and F_b
  Vec F_a; ierr = VecCreateGhostNodes(p4est, nodes, &F_a); CHKERRXX(ierr);
  Vec F_b; ierr = VecCreateGhostNodes(p4est, nodes, &F_b); CHKERRXX(ierr);

  double *F_a_ptr; ierr = VecGetArray(F_a, &F_a_ptr); CHKERRXX(ierr);
  double *F_b_ptr; ierr = VecGetArray(F_b, &F_b_ptr); CHKERRXX(ierr);

  std::vector<double *> f_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(f_a[i], &f_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> f_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(f_b[i], &f_b_ptr[i]); CHKERRXX(ierr); }

  std::vector<double *> qb_ptr(ns_total, NULL);
  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = f_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = f_b_ptr[i][n]*qb_ptr[i][n];
    }

    F_a_ptr[n] = compute_rho_a(integrand_a.data());
    F_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(f_a[i], &f_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(f_b[i], &f_b_ptr[i]); CHKERRXX(ierr); }

  // calculate G_a and G_b
  Vec G_a; ierr = VecCreateGhostNodes(p4est, nodes, &G_a); CHKERRXX(ierr);
  Vec G_b; ierr = VecCreateGhostNodes(p4est, nodes, &G_b); CHKERRXX(ierr);

  double *G_a_ptr; ierr = VecGetArray(G_a, &G_a_ptr); CHKERRXX(ierr);
  double *G_b_ptr; ierr = VecGetArray(G_b, &G_b_ptr); CHKERRXX(ierr);

  std::vector<double *> g_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(g_a[i], &g_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> g_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(g_b[i], &g_b_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = g_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = g_b_ptr[i][n]*qb_ptr[i][n];
    }

    G_a_ptr[n] = compute_rho_a(integrand_a.data());
    G_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(g_a[i], &g_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(g_b[i], &g_b_ptr[i]); CHKERRXX(ierr); }

  // calculate normal derivatives of Robin coefficients
  Vec dn_bc_coeffs_a; ierr = VecCreateGhostNodes(p4est, nodes, &dn_bc_coeffs_a); CHKERRXX(ierr);
  Vec dn_bc_coeffs_b; ierr = VecCreateGhostNodes(p4est, nodes, &dn_bc_coeffs_b); CHKERRXX(ierr);

  Vec bc_coeffs_a_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_a_d[dim]); CHKERRXX(ierr); }
  Vec bc_coeffs_b_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs_b_d[dim]); CHKERRXX(ierr); }

  ngbd->first_derivatives_central(bc_coeffs_a[phi_idx], bc_coeffs_a_d);
  ngbd->first_derivatives_central(bc_coeffs_b[phi_idx], bc_coeffs_b_d);

  double *normal_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  double *bc_coeffs_a_d_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_a_d[dim], &bc_coeffs_a_d_ptr[dim]); CHKERRXX(ierr); }
  double *bc_coeffs_b_d_ptr[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_b_d[dim], &bc_coeffs_b_d_ptr[dim]); CHKERRXX(ierr); }

  double *dn_bc_coeffs_a_ptr; ierr = VecGetArray(dn_bc_coeffs_a, &dn_bc_coeffs_a_ptr); CHKERRXX(ierr);
  double *dn_bc_coeffs_b_ptr; ierr = VecGetArray(dn_bc_coeffs_b, &dn_bc_coeffs_b_ptr); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    dn_bc_coeffs_a_ptr[n] = bc_coeffs_a_d_ptr[0][n] * normal_ptr[0][n] + bc_coeffs_a_d_ptr[1][n] * normal_ptr[1][n];
    dn_bc_coeffs_b_ptr[n] = bc_coeffs_b_d_ptr[0][n] * normal_ptr[0][n] + bc_coeffs_b_d_ptr[1][n] * normal_ptr[1][n];
  }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_a_d[dim], &bc_coeffs_a_d_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(bc_coeffs_b_d[dim], &bc_coeffs_b_d_ptr[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(bc_coeffs_a_d[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(bc_coeffs_b_d[dim]); CHKERRXX(ierr); }

  // compute dn_g_a and dn_b_b
  std::vector<Vec> gn_a(ns_total, NULL); for (int i = 0; i < ns_total; ++i) { ierr = VecCreateGhostNodes(p4est, nodes, &gn_a[i]); CHKERRXX(ierr); }
  std::vector<Vec> gn_b(ns_total, NULL); for (int i = 0; i < ns_total; ++i) { ierr = VecCreateGhostNodes(p4est, nodes, &gn_b[i]); CHKERRXX(ierr); }

  Vec g_d[P4EST_DIM]; for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &g_d[dim]); CHKERRXX(ierr); }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }

  double *g_d_ptr[P4EST_DIM];
  double *gn_ptr;

  for (int i = 0; i < ns_total; ++i)
  {
    ngbd->first_derivatives_central(g_a[i], g_d);

    ierr = VecGetArray(gn_a[i], &gn_ptr); CHKERRXX(ierr);
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      gn_ptr[n] = g_d_ptr[0][n] * normal_ptr[0][n] + g_d_ptr[1][n] * normal_ptr[1][n];
    }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    ierr = VecRestoreArray(gn_a[i], &gn_ptr); CHKERRXX(ierr);

    ngbd->first_derivatives_central(g_b[i], g_d);

    ierr = VecGetArray(gn_b[i], &gn_ptr); CHKERRXX(ierr);
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      gn_ptr[n] = g_d_ptr[0][n] * normal_ptr[0][n] + g_d_ptr[1][n] * normal_ptr[1][n];
    }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(g_d[dim], &g_d_ptr[dim]); CHKERRXX(ierr); }
    ierr = VecRestoreArray(gn_b[i], &gn_ptr); CHKERRXX(ierr);
  }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(normal[phi_idx][dim], &normal_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecDestroy(g_d[dim]); CHKERRXX(ierr); }

  // compute Gn_a and Gn_b
  Vec Gn_a; ierr = VecCreateGhostNodes(p4est, nodes, &Gn_a); CHKERRXX(ierr);
  Vec Gn_b; ierr = VecCreateGhostNodes(p4est, nodes, &Gn_b); CHKERRXX(ierr);

  double *Gn_a_ptr; ierr = VecGetArray(Gn_a, &Gn_a_ptr); CHKERRXX(ierr);
  double *Gn_b_ptr; ierr = VecGetArray(Gn_b, &Gn_b_ptr); CHKERRXX(ierr);

  std::vector<double *> gn_a_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(gn_a[i], &gn_a_ptr[i]); CHKERRXX(ierr); }
  std::vector<double *> gn_b_ptr(ns_total, NULL); for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(gn_b[i], &gn_b_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand_a(ns_total, NULL);
    std::vector<double> integrand_b(ns_total, NULL);

    for (int i = 0; i < ns_total; ++i)
    {
      integrand_a[i] = gn_a_ptr[i][n]*qb_ptr[i][n];
      integrand_b[i] = gn_b_ptr[i][n]*qb_ptr[i][n];
    }

    Gn_a_ptr[n] = compute_rho_a(integrand_a.data());
    Gn_b_ptr[n] = compute_rho_b(integrand_b.data());
  }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(gn_a[i], &gn_a_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(gn_b[i], &gn_b_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; ++i) { ierr = VecDestroy(gn_a[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; ++i) { ierr = VecDestroy(gn_b[i]); CHKERRXX(ierr); }

  // compute \nabla qf \cdot \nabla qb
  std::vector<Vec> dqfdqb(ns_total, NULL);
  for (int i = 0; i < ns_total; ++i) { ierr = VecCreateGhostNodes(p4est, nodes, &dqfdqb[i]); CHKERRXX(ierr); }

  Vec dqf[P4EST_DIM], dqb[P4EST_DIM];
  for (int dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &dqf[dim]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &dqb[dim]); CHKERRXX(ierr);
  }

  double *dqfdqb_ptr;
  double *dqf_ptr[P4EST_DIM];
  double *dqb_ptr[P4EST_DIM];

  for (int i = 0; i < ns_total; ++i)
  {
    ierr = VecGhostUpdateBegin(qf[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qf[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(qb[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qb[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ls.extend_Over_Interface_TVD(phi_smooth, qf[i]);
    ls.extend_Over_Interface_TVD(phi_smooth, qb[i]);

    ngbd->first_derivatives_central(qf[i], dqf);
    ngbd->first_derivatives_central(qb[i], dqb);

    ierr = VecGetArray(dqfdqb[i], &dqfdqb_ptr); CHKERRXX(ierr);

    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(dqf[dim], &dqf_ptr[dim]); CHKERRXX(ierr); }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(dqb[dim], &dqb_ptr[dim]); CHKERRXX(ierr); }

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      dqfdqb_ptr[n] = dqf_ptr[0][n]*dqb_ptr[0][n] + dqf_ptr[1][n]*dqb_ptr[1][n];
    }

    ierr = VecRestoreArray(dqfdqb[i], &dqfdqb_ptr); CHKERRXX(ierr);

    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(dqf[dim], &dqf_ptr[dim]); CHKERRXX(ierr); }
    for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(dqb[dim], &dqb_ptr[dim]); CHKERRXX(ierr); }
  }

  for (int dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecDestroy(dqf[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(dqb[dim]); CHKERRXX(ierr);
  }

  // compute qb qf_s + \nabla qb \nabla qf

  Vec pde_term;
  ierr = VecCreateGhostNodes(p4est, nodes, &pde_term); CHKERRXX(ierr);

  double *pde_term_ptr;
  ierr = VecGetArray(pde_term, &pde_term_ptr); CHKERRXX(ierr);

  std::vector<double *> qf_ptr(ns_total, NULL);
  std::vector<double *> dqdq_ptr(ns_total, NULL);

  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(qf[i], &qf_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecGetArray(dqfdqb[i], &dqdq_ptr[i]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    std::vector<double> integrand(ns_total, 0.0);

    for (int i = 0; i < ns_total; i++)
    {
      double qf_s;
      if      (i == 0)          qf_s = (qf_ptr[i+1][n]-qf_ptr[i][n])/ds_adaptive[i];
      else if (i == ns_total-1) qf_s = (qf_ptr[i][n]-qf_ptr[i-1][n])/ds_adaptive[i-1];
      else                      qf_s = (qf_ptr[i+1][n]-qf_ptr[i-1][n])/(ds_adaptive[i-1]+ds_adaptive[i]);
      integrand[i] = qb_ptr[i][n]*qf_s + dqdq_ptr[i][n];
    }

    pde_term_ptr[n] = compute_rho_a(integrand.data()) + compute_rho_b(integrand.data());
  }

  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qb[i], &qb_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(qf[i], &qf_ptr[i]); CHKERRXX(ierr); }
  for (int i = 0; i < ns_total; i++) { ierr = VecRestoreArray(dqfdqb[i], &dqdq_ptr[i]); CHKERRXX(ierr); }

  for (int i = 0; i < ns_total; ++i) { ierr = VecDestroy(dqfdqb[i]); CHKERRXX(ierr); }

  ierr = VecGetArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecGetArray(qf[ns_total-1],        &qf_end_ptr             ); CHKERRXX(ierr);
  ierr = VecGetArray(qb[ns_total-1],        &qb_end_ptr             ); CHKERRXX(ierr);
  ierr = VecGetArray(qf[0],                 &qf_start_ptr           ); CHKERRXX(ierr);
  ierr = VecGetArray(qb[0],                 &qb_start_ptr           ); CHKERRXX(ierr);
  ierr = VecGetArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
  ierr = VecGetArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
  ierr = VecGetArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecGetArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
#endif

    energy_shape_deriv_ptr[n] =
        qf_end_ptr[n]*qf_start_ptr[n] + F_a_ptr[n] + F_b_ptr[n]
        - pde_term_ptr[n]
        - (mu_p_ptr[n] - mu_m_ptr[n])*rho_a_ptr[n]
        - (mu_p_ptr[n] + mu_m_ptr[n])*rho_b_ptr[n]
//        - 1.0*kappa_ptr[n];
        - (kappa_ptr[n]-2.0*bc_coeffs_a_ptr[n])*(bc_coeffs_a_ptr[n]*rho_a_ptr[n] - G_a_ptr[n]) - dn_bc_coeffs_a_ptr[n]*rho_a_ptr[n] + Gn_a_ptr[n]
        - (kappa_ptr[n]-2.0*bc_coeffs_b_ptr[n])*(bc_coeffs_b_ptr[n]*rho_b_ptr[n] - G_b_ptr[n]) - dn_bc_coeffs_b_ptr[n]*rho_a_ptr[n] + Gn_b_ptr[n];

    energy_shape_deriv_ptr[n] *= 1;

//    energy_shape_deriv_ptr[n] =
//        0.5*(qf_end_ptr[n]*qb_end_ptr[n] + qf_start_ptr[n]*qb_start_ptr[n])
//        - 0.5*(rho_total_xx_ptr[n] + rho_total_yy_ptr[n]);

//    energy_shape_deriv_ptr[n] =
//        0.5*(qf_end_ptr[n]*qb_end_ptr[n] + qf_start_ptr[n]*qb_start_ptr[n]) + 0.5*(F_a_ptr[n] + F_b_ptr[n]);

//    energy_shape_deriv_ptr[n] = qf_end_ptr[n]*qf_start_ptr[n];
//    energy_shape_deriv_ptr[n] = 1.0 + F_a_ptr[n] + F_b_ptr[n];
//    energy_shape_deriv_ptr[n] = qf_end_ptr[n];
//    energy_shape_deriv_ptr[n] = 1.0;
  }

  ierr = VecRestoreArray(rho_total_xx, &rho_total_xx_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_total_yy, &rho_total_yy_ptr); CHKERRXX(ierr);


  ierr = VecRestoreArray(F_a, &F_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(F_b, &F_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(G_a, &G_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(G_b, &G_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(dn_bc_coeffs_a, &dn_bc_coeffs_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(dn_bc_coeffs_b, &dn_bc_coeffs_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(Gn_a, &Gn_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Gn_b, &Gn_b_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(pde_term, &pde_term_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(energy_shape_deriv_tmp,&energy_shape_deriv_ptr ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[ns_total-1],        &qf_end_ptr             ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[ns_total-1],        &qb_end_ptr             ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qf[0],                 &qf_start_ptr           ); CHKERRXX(ierr);
  ierr = VecRestoreArray(qb[0],                 &qb_start_ptr           ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_a,                 &rho_a_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b,                 &rho_b_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_coeffs_a[phi_idx],  &bc_coeffs_a_ptr        ); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_coeffs_b[phi_idx],  &bc_coeffs_b_ptr        ); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa[phi_idx],        &kappa_ptr              ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_m,                  &mu_m_ptr               ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_p,                  &mu_p_ptr               ); CHKERRXX(ierr);

  // sync the derivative among procs
  ierr = VecGhostUpdateBegin(energy_shape_deriv_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (energy_shape_deriv_tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // extrapolate from moving moving interface

  ls.extend_Over_Interface_TVD(phi_smooth, energy_shape_deriv_tmp);
  ls.extend_from_interface_to_whole_domain_TVD(phi->at(phi_idx), energy_shape_deriv_tmp, energy_shape_deriv_alt);

  ierr = VecDestroy(energy_shape_deriv_tmp); CHKERRXX(ierr);


////  double *phi_ptr;
////  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
////  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
////  {
////    phi_ptr[n] += 1.1*dxyz_min;
////  }
////  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

//  ls.extend_Over_Interface_TVD(phi_smooth, energy_shape_deriv_tmp);

////  ierr = VecGetArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);
////  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
////  {
////    phi_ptr[n] -= 1.1*dxyz_min;
////  }
////  ierr = VecRestoreArray(phi_smooth, &phi_ptr); CHKERRXX(ierr);

//  ierr = VecDestroy(energy_shape_deriv_alt); CHKERRXX(ierr);
//  energy_shape_deriv_alt = energy_shape_deriv_tmp;


  ierr = VecDestroy(rho_total); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_xx); CHKERRXX(ierr);
  ierr = VecDestroy(rho_total_yy); CHKERRXX(ierr);

  ierr = VecDestroy(F_a); CHKERRXX(ierr);
  ierr = VecDestroy(F_b); CHKERRXX(ierr);

  ierr = VecDestroy(G_a); CHKERRXX(ierr);
  ierr = VecDestroy(G_b); CHKERRXX(ierr);

  ierr = VecDestroy(dn_bc_coeffs_a); CHKERRXX(ierr);
  ierr = VecDestroy(dn_bc_coeffs_b); CHKERRXX(ierr);

  ierr = VecDestroy(Gn_a); CHKERRXX(ierr);
  ierr = VecDestroy(Gn_b); CHKERRXX(ierr);

  ierr = VecDestroy(pde_term); CHKERRXX(ierr);

}

void my_p4est_scft_t::compute_contact_term_of_energy_shape_derivative(int phi0_idx, int phi1_idx)
{
  if (contact_term_of_energy_shape_deriv != NULL) { ierr = VecDestroy(contact_term_of_energy_shape_deriv); CHKERRXX(ierr); }
  ierr = VecDuplicate(mu_m, &contact_term_of_energy_shape_deriv); CHKERRXX(ierr);

  double *contact_term_of_energy_shape_deriv_ptr;
  double *rho_a_ptr, *normal_phi0_ptr[P4EST_DIM];
  double *rho_b_ptr, *normal_phi1_ptr[P4EST_DIM];

  ierr = VecGetArray(contact_term_of_energy_shape_deriv, &contact_term_of_energy_shape_deriv_ptr); CHKERRXX(ierr);

  ierr = VecGetArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGetArray(normal[phi0_idx][dim], &normal_phi0_ptr[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(normal[phi1_idx][dim], &normal_phi1_ptr[dim]); CHKERRXX(ierr);
  }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
#endif

    // compute contact angle between interfaces
    double cos_theta = - (normal_phi0_ptr[0][n]*normal_phi1_ptr[0][n] + normal_phi0_ptr[1][n]*normal_phi1_ptr[1][n]);

    double gamma_phi0 = rho_a_ptr[n]*(*gamma_a->at(phi0_idx))(x,y) + rho_b_ptr[n]*(*gamma_b->at(phi0_idx))(x,y);
    double gamma_phi1 = rho_a_ptr[n]*(*gamma_a->at(phi1_idx))(x,y) + rho_b_ptr[n]*(*gamma_b->at(phi1_idx))(x,y);
    double gamma_air_val  = (*gamma_air)(x,y);

    contact_term_of_energy_shape_deriv_ptr[n] = (gamma_phi0 - gamma_air_val + gamma_phi1*cos_theta)/sqrt(1.0-cos_theta*cos_theta);
  }

  ierr = VecRestoreArray(contact_term_of_energy_shape_deriv, &contact_term_of_energy_shape_deriv_ptr); CHKERRXX(ierr);

  ierr = VecRestoreArray(rho_a, &rho_a_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rho_b, &rho_b_ptr); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecRestoreArray(normal[phi0_idx][dim], &normal_phi0_ptr[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal[phi1_idx][dim], &normal_phi1_ptr[dim]); CHKERRXX(ierr);
  }

  // sync the derivative among procs
  ierr = VecGhostUpdateBegin(contact_term_of_energy_shape_deriv, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (contact_term_of_energy_shape_deriv, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // extend over smoothed interface
//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi_smooth, contact_term_of_energy_shape_deriv);
}

double my_p4est_scft_t::compute_change_in_energy(int phi_idx, Vec norm_velo, double dt)
{
  Vec integrand;
  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

  double *energy_shape_deriv_ptr; ierr = VecGetArray(energy_shape_deriv, &energy_shape_deriv_ptr); CHKERRXX(ierr);
  double *integrand_ptr;          ierr = VecGetArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
  double *norm_velo_ptr;          ierr = VecGetArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);


  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    integrand_ptr[n] = norm_velo_ptr[n]*energy_shape_deriv_ptr[n];
  }

  ierr = VecRestoreArray(energy_shape_deriv, &energy_shape_deriv_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
  ierr = VecRestoreArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

  // sync velocity among procs
  ierr = VecGhostUpdateBegin(integrand, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (integrand, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // extend over smoothed interface
//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi_smooth, integrand);

  my_p4est_integration_mls_t integrator;
  integrator.set_p4est(p4est, nodes);
  integrator.set_phi(*phi, *action, color);

  double result = integrator.integrate_over_interface(integrand, phi_idx)*dt;

  return result;
}

double my_p4est_scft_t::compute_change_in_energy_alt(int phi_idx, Vec norm_velo, double dt)
{
  Vec integrand;
  ierr = VecDuplicate(mu_m, &integrand); CHKERRXX(ierr);

  double *energy_shape_deriv_ptr; ierr = VecGetArray(energy_shape_deriv_alt, &energy_shape_deriv_ptr); CHKERRXX(ierr);
  double *integrand_ptr;          ierr = VecGetArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
  double *norm_velo_ptr;          ierr = VecGetArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);


  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    integrand_ptr[n] = norm_velo_ptr[n]*energy_shape_deriv_ptr[n];
  }

  ierr = VecRestoreArray(energy_shape_deriv_alt, &energy_shape_deriv_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(integrand         , &integrand_ptr         ); CHKERRXX(ierr);
  ierr = VecRestoreArray(norm_velo         , &norm_velo_ptr         ); CHKERRXX(ierr);

  // sync velocity among procs
  ierr = VecGhostUpdateBegin(integrand, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (integrand, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // extend over smoothed interface
//  my_p4est_level_set_t ls(ngbd);
//  ls.extend_Over_Interface_TVD(phi_smooth, integrand);

  my_p4est_integration_mls_t integrator;
  integrator.set_p4est(p4est, nodes);
  integrator.set_phi(*phi, *action, color);

  double result = integrator.integrate_over_interface(integrand, phi_idx)*dt;

  return result;
}

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

  // allocate memory
  normal.resize(num_surfaces, NULL);
  kappa.resize(num_surfaces, NULL);

  for (int surf_idx = 0; surf_idx < normal.size(); surf_idx++)
  {
    normal[surf_idx] = new Vec[P4EST_DIM];
//    if (surf_idx == 0)
      for (int dim = 0; dim < P4EST_DIM; dim++) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[surf_idx][dim]); CHKERRXX(ierr); }
//    else
//      for (int dim = 0; dim < P4EST_DIM; dim++) { ierr = VecDuplicate(normal[0][dim], &normal[surf_idx][dim]); CHKERRXX(ierr); }

    ierr = VecCreateGhostNodes(p4est, nodes, &kappa[surf_idx]); CHKERRXX(ierr);
  }

  for (int surf_idx = 0; surf_idx < num_surfaces; surf_idx++)
  {
    double *normal_p[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGetArray(normal[surf_idx][dir], &normal_p[dir]); CHKERRXX(ierr);
    }

    const double *phi_p;
    ierr = VecGetArrayRead(phi->at(surf_idx), &phi_p); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      ngbd->get_neighbors(n, qnnn);
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
      ierr = VecGhostUpdateBegin(normal[surf_idx][dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_local_node(i);
      ngbd->get_neighbors(n, qnnn);
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
    ierr = VecRestoreArrayRead(phi->at(surf_idx), &phi_p); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGhostUpdateEnd(normal[surf_idx][dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    Vec kappa_tmp;
    ierr = VecDuplicate(kappa[surf_idx], &kappa_tmp); CHKERRXX(ierr);
    double *kappa_p;
    ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd->get_layer_node(i);
      ngbd->get_neighbors(n, qnnn);
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
      ngbd->get_neighbors(n, qnnn);
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
      ierr = VecRestoreArray(normal[surf_idx][dir], &normal_p[dir]); CHKERRXX(ierr);
    }

    my_p4est_level_set_t ls(ngbd);
    ls.extend_from_interface_to_whole_domain_TVD(phi->at(surf_idx), kappa_tmp, kappa[surf_idx]);
    ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

//    ierr = VecDestroy(kappa[surf_idx]); CHKERRXX(ierr);
//    kappa[surf_idx] = kappa_tmp;
  }

}

void my_p4est_scft_t::update_grid(Vec normal_velo, int surf_idx, double dt)
{
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

  // reconstruct vector velocity
  Vec velocity[P4EST_DIM];
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &velocity[dim]); CHKERRXX(ierr); }

  double *velocity_ptr[P4EST_DIM];
  double *normal_ptr[P4EST_DIM];
  double *normal_velo_ptr;

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(velocity[dim]        , &velocity_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecGetArray(normal[surf_idx][dim], &normal_ptr[dim]  ); CHKERRXX(ierr); }

  ierr = VecGetArray(normal_velo, &normal_velo_ptr); CHKERRXX(ierr);

  double v_max;

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    for (int dim = 0; dim < P4EST_DIM; ++dim)
      velocity_ptr[dim][n] = normal_velo_ptr[n]*normal_ptr[dim][n];

  }

  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(velocity[dim]        , &velocity_ptr[dim]); CHKERRXX(ierr); }
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecRestoreArray(normal[surf_idx][dim], &normal_ptr[dim]  ); CHKERRXX(ierr); }

  ierr = VecRestoreArray(normal_velo, &normal_velo_ptr); CHKERRXX(ierr);

  Vec velocity_interface[P4EST_DIM];
  for (int dim = 0; dim < P4EST_DIM; ++dim) { ierr = VecCreateGhostNodes(p4est, nodes, &velocity_interface[dim]); CHKERRXX(ierr); }

  my_p4est_level_set_t ls_old(ngbd);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ls_old.extend_from_interface_to_whole_domain_TVD(phi->at(surf_idx), velocity[dir], velocity_interface[dir]);
  }

  /* bousouf update this for second order in time */
//  double dt = sl.compute_dt(normal[0], normal[1]);
  sl.update_p4est(velocity_interface, dt, *phi, *action, surf_idx);

  for(int dim=0; dim<P4EST_DIM; ++dim)
  {
    ierr = VecDestroy(velocity[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(velocity_interface[dim]); CHKERRXX(ierr);
  }

  /* interpolate the quantities on the new grid */
  my_p4est_interpolation_nodes_t interp(ngbd);

  double xyz[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  Vec mu_m_tmp;
  ierr = VecDuplicate(phi->at(0), &mu_m_tmp); CHKERRXX(ierr);
  interp.set_input(mu_m, quadratic_non_oscillatory);
  interp.interpolate(mu_m_tmp);
  ierr = VecDestroy(mu_m); CHKERRXX(ierr);
  mu_m = mu_m_tmp;

  Vec mu_p_tmp;
  ierr = VecDuplicate(phi->at(0), &mu_p_tmp); CHKERRXX(ierr);
  interp.set_input(mu_p, quadratic_non_oscillatory);
  interp.interpolate(mu_p_tmp);
  ierr = VecDestroy(mu_p); CHKERRXX(ierr);
  mu_p = mu_p_tmp;

  Vec rho_a_tmp;
  ierr = VecDuplicate(phi->at(0), &rho_a_tmp); CHKERRXX(ierr);
  interp.set_input(rho_a, quadratic_non_oscillatory);
  interp.interpolate(rho_a_tmp);
  ierr = VecDestroy(rho_a); CHKERRXX(ierr);
  rho_a = rho_a_tmp;

  Vec rho_b_tmp;
  ierr = VecDuplicate(phi->at(0), &rho_b_tmp); CHKERRXX(ierr);
  interp.set_input(rho_b, quadratic_non_oscillatory);
  interp.interpolate(rho_b_tmp);
  ierr = VecDestroy(rho_b); CHKERRXX(ierr);
  rho_b = rho_b_tmp;

  interp.clear();

  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  hierarchy->update(p4est, ghost);
  ngbd->update(hierarchy, nodes);

  /* reinitialize and perturb phi */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi->at(surf_idx));
  set_geometry(*phi, *action);

//  initialize_bc_smart();
  initialize_bc_simple();
  initialize_linear_system();
  compute_normal_and_curvature();
}
