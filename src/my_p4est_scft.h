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
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

class my_p4est_scft_t
{
  /* grid */
  my_p4est_brick_t *brick;
  p4est_connectivity_t *connectivity;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  /* geometry */
  std::vector<Vec> *phi;
  std::vector<action_t> *action;

  /* potentials */
  Vec mu_m, mu_p;

  /* densities */
  Vec rho_a, rho_b;

  /* surface tensions */
  std::vector<Vec> gamma_a, gamma_b;

  /* Robin coefficients */
  std::vector<Vec> bc_coeffs_a, bc_coeffs_b;

  /* partition function and energy */
  double Q;
  double H;

  /* physical parameters */
  double N;
  double Xab;
  double b;
  int ns;

  /* auxiliary variables */
  double Rg;
  double XN;
  double ds;

  /* Poisson solver */
  my_p4est_poisson_nodes_mls_t *poisson_solver;

public:
  my_p4est_scft_t(my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_scft_t();

  void set_geometry(std::vector<Vec>& in_phi, std::vector<Vec>& in_action) {phi = &in_phi; action = &in_action;}
  void set_parameters(int in_N, double in_f, double in_Xab, double b, int in_ns);
  void set_surface_tensions(std::vector<Vec>& in_gamma_a, std::vector<Vec>& in_gamma_b) {gamma_a = &in_gamma_a; gamma_b = &in_gamma_b;}

  void set_potentials(Vec in_mu_m,  Vec in_mu_p)  {mu_m  = in_mu_m;  mu_p  = in_mu_p; }
  void set_densities (Vec in_rho_a, Vec in_rho_b) {rho_a = in_rho_a; rho_b = in_rho_b;}

  void initialize_bc_simple(); // a naive method that produces singularities in the pressure field
  void initialize_bc_smart();  // a method based on adjusting Robin coeff so that there is no sigularities in the pressure field

  void initialize_linear_system();

  void solve_for_propogators();
  void calculate_densities();
  void update_potentials();

  void initialize_bc_smart();
  void initialize_linear_system();
  void subtract_singularity_from_pressure_field();

  void print_VTK(char *output_file);

};

#endif // MY_P4EST_SCFT_T_H
