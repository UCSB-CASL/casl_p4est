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

  std::vector<Vec> exp_w_a, exp_w_b;

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
  Vec energy_shape_deriv_alt;
  Vec contact_term_of_energy_shape_deriv;

  /* Poisson solver */
  std::vector<my_p4est_poisson_nodes_mls_t *> solver_a, solver_b;

  /* Adaptive discretization in time */
  int num_of_refinements;
  int degree_of_refinement;
  int ns_total;
  int fns_adaptive;
  std::vector<double> ds_list, ds_adaptive;

  std::vector<Vec> f_a, f_b, g_a, g_b;

//public:
  my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_scft_t();

  void set_geometry(std::vector<Vec>& in_phi, std::vector<action_t> &in_action);
  void set_parameters(double in_f, double in_XN, int in_ns, int in_num_of_refinements, int in_degree_of_refinement);
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

  void diffusion_step(my_p4est_poisson_nodes_mls_t *solver, Vec &sol, Vec &sol_nm1, double ds_local, int is, bool forward);

  void save_VTK(int compt);

  double get_energy() { return H;}
  double get_pressure_force() { return force_p_avg; }
  double get_exchange_force() { return force_m_avg; }

  void assemble_integrating_vec();

  double integrate_over_domain_fast(Vec f);
  double integrate_over_domain_fast_squared(Vec f);

  double compute_rho_a(double *integrand);
  double compute_rho_b(double *integrand);

  void compute_energy_shape_derivative(int phi_idx);
  void compute_energy_shape_derivative_alt(int phi_idx);
  void compute_normal_and_curvature();
  void compute_contact_term_of_energy_shape_derivative(int phi0_idx, int phi1_idx);
  double compute_change_in_energy(int phi_idx, Vec norm_velo, double dt);
  double compute_change_in_energy_alt(int phi_idx, Vec norm_velo, double dt);
  void update_grid(Vec normal_velo, int surf_idx, double dt);


  void set_force_and_bc(CF_2& f_a_cf, CF_2& f_b_cf, CF_2& g_a_cf, CF_2& g_b_cf);
  void set_ic(CF_2& u_exact);

  void set_exact_solutions(CF_2& qf_cf, CF_2& qb_cf);

  void save_VTK_q(int compt);

};

#endif // MY_P4EST_SCFT_T_H
