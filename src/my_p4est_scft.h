#ifndef MY_P4EST_SCFT_T_H
#define MY_P4EST_SCFT_T_H

#include <src/types.h>
#include <src/casl_math.h>

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_level_set.h>
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
  PetscErrorCode ierr;

  /* grid */
  my_p4est_brick_t          *brick;
  p4est_connectivity_t      *connectivity;
  p4est_t                   *p4est;
  p4est_ghost_t             *ghost;
  p4est_nodes_t             *nodes;
  my_p4est_hierarchy_t      *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  splitting_criteria_t *sp_crit;

  /* geometry */
  int num_surfaces;
  std::vector<Vec>       phi;
  std::vector<int>       color;
  std::vector<mls_opn_t> action;

  /* potentials */
  Vec    mu_m;
  Vec    mu_p;

  double mu_m_avg;
  double mu_p_avg;

  /* densities */
  Vec rho_a;
  Vec rho_b;
  double rho_avg;

  /* surface tensions */
  std::vector<CF_DIM *> gamma_a;
  std::vector<CF_DIM *> gamma_b;
  CF_DIM *gamma_air;
  std::vector<bool> grafting;

  /* grafting density */
  std::vector<CF_DIM *> grafting_density;

  /* Robin coefficients */
  std::vector< std::vector<double> > pw_bc_values;
  std::vector< std::vector<double> > pw_bc_coeffs_a;
  std::vector< std::vector<double> > pw_bc_coeffs_b;

  /* partition function and energy */
  double Q;
  double energy;
  double energy_singular_part;

  /* physical parameters */
  double f;
  double XN;
  int    ns, fns;
  double scaling;
  bool   grafted;
  double grafting_area;

  /* auxiliary variables */
  double ds_a, ds_b;
  double ns_a, ns_b;
  double lambda;
  double dxyz[P4EST_DIM], dxyz_min, dxyz_max, diag;
  double dxyz_close_interface;

  double volume;

  Vec    force_p;
  Vec    force_m;
  double force_p_avg, force_p_max;
  double force_m_avg, force_m_max;

  Vec exp_w_a;
  Vec exp_w_b;

  Vec rhs;

  std::vector<Vec> qf;
  std::vector<Vec> qb;

  Vec phi_smooth;

  Vec mask; // nodes where diffusion is solved (\phi < 0 and boundary nodes)
  Vec mask_wo_bc; // mask excluding boundary nodes
  Vec integrating_vec;
  Vec q_tmp;

  int time_discretization;
  int integration_order;
  int cube_refinement;
  int field_update; // 0 - explicit, 1 - approx SIS

  std::vector<Vec*> normal;
  std::vector<Vec>  kappa;

  /* Poisson solver */
  my_p4est_poisson_nodes_mls_t solver_a;
  my_p4est_poisson_nodes_mls_t solver_b;

public:
  my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd, int ns);
  ~my_p4est_scft_t();

  void set_lambda(double value) { lambda = value; }
  void set_polymer(double f, double XN, bool grafted=0);
  void add_boundary(Vec phi, mls_opn_t acn, CF_DIM &surf_energy_A, CF_DIM &surf_energy_B, bool grafting=0, CF_DIM *grafting_density=NULL);

  void initialize_solvers();
  void initialize_bc_simple(); // a naive method that produces singularities in the pressure field
  void initialize_bc_smart(bool adaptive = true, int bc_scheme=0);  // a method based on adjusting Robin coeff so that there is no sigularities in the pressure field

  void   diffusion_step(my_p4est_poisson_nodes_mls_t &solver, double ds, Vec &sol, Vec &sol_nm1);
  void   solve_for_propogators();
  void   calculate_densities();
  double compute_rho_a(double *integrand);
  double compute_rho_b(double *integrand);
  double integrate_in_time(int start, int end, double *integrand);
  void   update_potentials(bool update_mu_m=true, bool update_mu_p=true);
  void   smooth_singularity_in_pressure_field();

  void save_VTK(int compt, const char *absolute_path_to_folder=NULL);

  double get_energy() { return energy;}
  double get_pressure_force() { return force_p_avg; }
  double get_exchange_force() { return force_m_avg; }
  double get_rho_avg() { return rho_avg;}
  void   set_rho_avg(double val) { rho_avg = val;}


  void   assemble_integrating_vec();
  double integrate_over_domain_fast(Vec f);
  double integrate_over_domain_fast_squared(Vec f);
  double integrate_over_domain_fast_two(Vec f0, Vec f1);

  void compute_normal_and_curvature();

  void compute_energy_shape_derivative(int phi_idx, Vec velo, bool assume_convergence=true);
  void compute_energy_shape_derivative_contact_term(int phi0_idx, int phi1_idx);

  double compute_change_in_energy(int phi_idx, Vec norm_velo, double dt);
  double compute_change_in_energy_contact_term(int phi0_idx, int phi1_idx, Vec norm_velo, double dt);

  void sync_and_extend();

  inline p4est_t*       get_p4est() { return p4est; }
  inline p4est_nodes_t* get_nodes() { return nodes; }
  inline p4est_ghost_t* get_ghost() { return ghost; }
  inline my_p4est_node_neighbors_t* get_ngbd()  { return ngbd; }

  inline Vec get_mu_m() { return mu_m; }
  inline Vec get_mu_p() { return mu_p; }

  inline Vec get_rho_a() { return rho_a; }
  inline Vec get_rho_b() { return rho_b; }

  inline void set_scaling(double value) { scaling = value; }

  void save_VTK_q(int compt);

  inline int get_ns() { return ns; }

  inline double get_grafted_area() { return grafting_area; }
  inline Vec get_qf(int i) { return qf[i]; }
  inline Vec get_qb(int i) { return qb[i]; }


  /* Density Optimization */
  Vec mu_t;
  Vec nu_m;
  Vec nu_p;
  Vec nu_a;
  Vec nu_b;

  double nu_0;
  double cost_function;

  std::vector<Vec> zf;
  std::vector<Vec> zb;

  Vec force_nu_m;
  Vec force_nu_p;

  double force_nu_p_avg;
  double force_nu_m_avg;

  Vec psi_a;
  Vec psi_b;
  double psi_avg;

  void   dsa_initialize();
  void   dsa_initialize_fields();
  void   dsa_solve_for_propogators();
  void   dsa_diffusion_step(my_p4est_poisson_nodes_mls_t *solver, double ds, Vec &sol, Vec &sol_nm1, Vec &nu, Vec &q, Vec &qm1);
  void   dsa_compute_densities();
  void   dsa_update_potentials();
  void   dsa_compute_shape_gradient(int phi_idx, Vec velo);
  double dsa_compute_cost_function();
  double dsa_compute_change_in_functional(int phi_idx, Vec norm_velo, Vec density_grad_shape, double dt);
  void   dsa_save_VTK(int compt);
  void   dsa_save_VTK_before_moving(int compt);

  void   dsa_sync_and_extend();
  double dsa_get_nu_0()           { return nu_0; }
  double dsa_get_cost_function()  { return cost_function; }
  double dsa_get_pressure_force() { return force_nu_p_avg; }
  double dsa_get_exchange_force() { return force_nu_m_avg; }

  Vec    dsa_get_nu_m() { return nu_m; }
  Vec    dsa_get_nu_p() { return nu_p; }
  Vec    dsa_get_mu_t() { return mu_t; }

};

#endif // MY_P4EST_SCFT_T_H
