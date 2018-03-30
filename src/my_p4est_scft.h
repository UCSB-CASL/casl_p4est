#ifndef MY_P4EST_SCFT_T_H
#define MY_P4EST_SCFT_T_H

#include <src/types.h>
#include <src/casl_math.h>
#undef P4_TO_P8

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_level_set.h>
#endif

class my_p4est_scft_t
{
public:
  PetscErrorCode ierr;

  /* grid */
  my_p4est_brick_t          *brick;
  p4est_connectivity_t      *connectivity;
  p4est_t                   *p4est;
  p4est_ghost_t             *ghost;
  p4est_nodes_t             *nodes;
  my_p4est_hierarchy_t      *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  /* geometry */
  std::vector<Vec> *phi;
  std::vector<action_t> *action;

  /* potentials */
  Vec mu_m, mu_p;
  double mu_m_avg, mu_p_avg;

  /* densities */
  Vec rho_a, rho_b;

  /* surface tensions */
//  std::vector<Vec> *gamma_a, *gamma_b;
  std::vector<CF_2 *> *gamma_a, *gamma_b;
  CF_2 *gamma_air;

  /* Robin coefficients */
  std::vector<Vec> bc_coeffs_a, bc_coeffs_b;

  /* partition function and energy */
  double Q;
  double H, singular_part_of_energy;

  /* physical parameters */
  double f;
  double XN;
  int ns, fns;

  /* auxiliary variables */
  double ds;
  double ns_a, ns_b;
  int num_surfaces;
  double lambda;
  double dxyz[P4EST_DIM], dxyz_min, dxyz_max;

  double volume;

  Vec force_p, force_m;
  double force_p_avg, force_m_avg;

  Vec exp_w_a, exp_w_b;

  Vec rhs, rhs_old, add_to_rhs;

  std::vector<int> color;
  std::vector<double> bc_values;
  std::vector<BoundaryConditionType> bc_types;

  std::vector<Vec> qf;
  std::vector<Vec> qb;

  Vec phi_smooth;

  Vec mask;
  Vec integrating_vec;
  Vec q_tmp;

  double scheme_coeff; // 1 - fully implicit, 2 - crank-nicolson
  bool use_cn_scheme;

  std::vector<Vec *> normal;
  std::vector<Vec> kappa;

  Vec energy_shape_deriv;
  Vec energy_shape_deriv_contact_term;

  double dt_energy;

  /* Poisson solver */
  my_p4est_poisson_nodes_mls_t *solver_a, *solver_b;

//public:
  my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_scft_t();

  void set_geometry(std::vector<Vec>& in_phi, std::vector<action_t> &in_action);
  void set_parameters(double in_f, double in_XN, int in_ns);
  void set_surface_tensions(std::vector<CF_2 *>& in_gamma_a, std::vector<CF_2 *>& in_gamma_b, CF_2 &in_gamma_air)
  {
    gamma_a = &in_gamma_a; gamma_b = &in_gamma_b; gamma_air = &in_gamma_air;
  }

  void set_potentials(Vec in_mu_m,  Vec in_mu_p)  { mu_m  = in_mu_m;  mu_p  = in_mu_p;  }
  void set_densities (Vec in_rho_a, Vec in_rho_b) { rho_a = in_rho_a; rho_b = in_rho_b; }

  void initialize_bc_simple(); // a naive method that produces singularities in the pressure field
  void initialize_bc_smart();  // a method based on adjusting Robin coeff so that there is no sigularities in the pressure field

  void initialize_linear_system();

  void solve_for_propogators();
  void calculate_densities();
  void update_potentials();

  void smooth_singularity_in_pressure_field();

  double integrate_in_time(int start, int end, double *integrand);

  void diffusion_step(my_p4est_poisson_nodes_mls_t *solver, Vec &sol, Vec &sol_nm1);

  void save_VTK(int compt);

  double get_energy() { return H;}
  double get_pressure_force() { return force_p_avg; }
  double get_exchange_force() { return force_m_avg; }

  void assemble_integrating_vec();

  double integrate_over_domain_fast(Vec f);
  double integrate_over_domain_fast_squared(Vec f);
  double integrate_over_domain_fast_two(Vec f0, Vec f1);

  double compute_rho_a(double *integrand);
  double compute_rho_b(double *integrand);

  void compute_normal_and_curvature();

  void compute_energy_shape_derivative(int phi_idx);
  void compute_energy_shape_derivative_contact_term(int phi0_idx, int phi1_idx);

  double compute_change_in_energy(int phi_idx, Vec norm_velo, double dt);
  double compute_change_in_energy_contact_term(int phi0_idx, int phi1_idx, Vec norm_velo, double dt);

  void update_grid(Vec normal_velo, int surf_idx, double dt);


  void save_VTK_q(int compt);

  /* Density Optimization */

  Vec lam_m, lam_p, lam_a, lam_b;

  std::vector<Vec> lam_fwd, lam_bwd;

  Vec density_shape_grad;

  Vec force_lam_m, force_lam_p;

  Vec rho_lam_a, rho_lam_b;

  Vec mu_t;

  double force_lam_p_avg, force_lam_m_avg;

  double cost_func;

  double lam_0;

  double dt_density;

  void DO_initialize(CF_2 &mu_target_cf);
  void DO_initialize_fields();
  void DO_solve_for_propogators();
  void DO_diffusion_step(my_p4est_poisson_nodes_mls_t *solver, Vec &sol, Vec &sol_nm1, Vec &exp_w, Vec &q, Vec &lam);
  void DO_compute_densities();
  void DO_update_potentials();
  void DO_compute_shape_derivative(int phi_idx);
  double DO_compute_cost_functional();
  double DO_compute_change_in_functional(int phi_idx, Vec norm_velo, double dt);
  void DO_save_VTK(int compt);
  void DO_save_VTK_before_moving(int compt);

  double DO_get_cost_func() { return cost_func; }
  double DO_get_pressure_force() { return force_lam_p_avg; }
  double DO_get_exchange_force() { return force_lam_m_avg; }

};

#endif // MY_P4EST_SCFT_T_H
