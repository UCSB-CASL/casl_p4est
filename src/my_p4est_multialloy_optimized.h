#ifndef MY_P4EST_MULTIALLOY_H
#define MY_P4EST_MULTIALLOY_H

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
#include <src/my_p4est_poisson_nodes_voronoi.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif


class my_p4est_multialloy_t
{
private:

  PetscErrorCode ierr;

#ifdef P4_TO_P8
  class ZERO : public CF_3
  {
  public:
    double operator()(double, double, double) const
    {
      return 0;
    }
  } zero;
#else
  class ZERO : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero;
#endif

#ifdef P4_TO_P8
  BoundaryConditions3D bc_t;
  BoundaryConditions3D bc_cs;
  BoundaryConditions3D bc_cl;
#else
  BoundaryConditions2D bc_t;
  BoundaryConditions2D bc_cs;
  BoundaryConditions2D bc_cl;
  BoundaryConditions2D bc_cs_sec;
  BoundaryConditions2D bc_cl_sec;
#endif

  /* grid */
  my_p4est_brick_t *brick;
  p4est_connectivity_t *connectivity;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  double dxyz[P4EST_DIM];
  double dxyz_max, dxyz_min;
  double dxyz_close_interface;
  double diag;

  /* temperature */
  Vec temperature_n, temperature_np1;
  Vec t_interface;

  /* concentration */
  Vec cs_n, cs_np1;
  Vec cl_n, cl_np1;
  Vec cs_sec_n, cs_sec_np1;
  Vec cl_sec_n, cl_sec_np1;
  Vec c_interface;
  Vec c_sec_interface;

  /* velocity */
  Vec v_interface_n  [P4EST_DIM];
  Vec v_interface_np1[P4EST_DIM];
  Vec normal_velocity_np1;
  /* max interface velocity in normal direction in band 4*MIN(dx,dy,dz) */
  double vgamma_max;

  /* level-set */
  Vec phi;
  Vec normal[P4EST_DIM];
  Vec kappa;

  /* physical parameters */
  double dt_nm1, dt_n;
  double cooling_velocity;     /* V */
  double latent_heat;          /* L */
  double thermal_conductivity; /* k */
  double thermal_diffusivity;  /* lambda, dT/dt = lambda Laplace(T) */
  double solute_diffusivity_l; /* Dl, dCl/dt = Dl Laplace(Cl) */
  double solute_diffusivity_s; /* Ds, dCs/dt = Ds Laplace(Cl) */
  double kp;                   /* partition coefficient */
  double c0;                   /* initial concentration */
  double ml;                   /* liquidus slope */
  double Tm;                   /* melting temperature */
  double epsilon_anisotropy;   /* anisotropy coefficient */
  double epsilon_c;            /* curvature undercooling coefficient */
  double epsilon_v;            /* kinetic undercooling coefficient */

  double solute_diffusivity_l_sec; /* Dl, dCl/dt = Dl Laplace(Cl) */
  double solute_diffusivity_s_sec; /* Ds, dCs/dt = Ds Laplace(Cl) */
  double kp_sec;                   /* partition coefficient */
  double c0_sec;                   /* initial concentration */
  double ml_sec;                   /* liquidus slope */

  bool solve_concentration_solid;

  double scaling;

  bool matrices_are_constructed;
  Vec rhs;

  int dt_method;

  double velocity_tol;

  bool first_step;

  double cfl_number;

  Vec temperature_multiplier;
  Vec concentration_multiplier;
  Vec concentration_multiplier_interface;
  Vec concentration_sec_multiplier;

  double functional;

  int order_of_extension;

  bool use_more_points_for_extension;
  bool use_quadratic_form;

  Vec bc_error;

#ifdef P4_TO_P8
  Vec theta_xz, theta_yz;
#else
  Vec theta;
#endif

  Vec ts, tl;
  bool temperature_interpolation_simple;

public:

  my_p4est_multialloy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_multialloy_t();

  void set_parameters( double latent_heat,
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
                       double scaling,
                       double solute_diffusivity_l_sec,
                       double solute_diffusivity_s_sec,
                       double kp_sec,
                       double c0_sec,
                       double ml_sec);

  void set_phi(Vec phi);

#ifdef P4_TO_P8
  void set_bc(WallBC3D& bc_wall_type_t,
              WallBC3D& bc_wall_type_c,
              CF_3& bc_wall_value_t,
              CF_3& bc_wall_value_cs,
              CF_3& bc_wall_value_cl);
#else
  void set_bc(WallBC2D& bc_wall_type_t,
              WallBC2D& bc_wall_type_c,
              CF_2& bc_wall_value_t,
              CF_2& bc_wall_value_cs,
              CF_2& bc_wall_value_cl,
              CF_2& bc_wall_value_cs_sec,
              CF_2& bc_wall_value_cl_sec);
#endif

  void set_temperature(Vec temperature);

  void set_concentration(Vec cl, Vec cs, Vec cl_sec, Vec cs_sec);

  void set_normal_velocity(Vec v);

  inline p4est_t* get_p4est() { return p4est; }

  inline p4est_nodes_t* get_nodes() { return nodes; }

  inline Vec get_phi() { return phi; }

  inline Vec get_normal_velocity() { return normal_velocity_np1; }

  inline double get_dt() { return dt_n; }

  inline double get_max_interface_velocity() { return vgamma_max; }

  void set_dt( double dt );

  void set_dt_method (int val) {dt_method = val;}

  void set_velocity_tol (double val) {velocity_tol = val;}

  void set_cfl (double val) {cfl_number = val;}

  void compute_normal_and_curvature();

  void compute_normal_velocity();

  void compute_velocity();
  void compute_velocity_from_temperature();

  void solve_temperature();

  void solve_concentration();

  void solve_concentration_sec();

  void compute_dt();

  void update_grid();

  void one_step();

  void compare_velocity_temperature_vs_concentration();

  void save_VTK(int iter);

  void solve_temperature_multiplier();
  void solve_concentration_multiplier();
  void solve_concentration_sec_multiplier();
  void adjust_interface_concentration(bool use_simple);

  void adjust_velocity();
  void smooth_velocity(Vec input);


  inline double eps_c(double theta)
  {
    return epsilon_c*(1.0-15.0*epsilon_anisotropy*cos(4.0*theta));
  }

  inline double eps_v(double theta)
  {
    return epsilon_v*(1.0-15.0*epsilon_anisotropy*cos(4.0*theta));
  }

  inline void copy_ghosted_vec(Vec input, Vec output)
  {
    Vec src, out;
    ierr = VecGhostGetLocalForm(input, &src); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(output, &out); CHKERRXX(ierr);
    ierr = VecCopy(src, out); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(input, &src); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  }
};


#endif /* MY_P4EST_MULTIALLOY_H */
