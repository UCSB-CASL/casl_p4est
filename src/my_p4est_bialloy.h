#ifndef MY_P4EST_BIALLOY_H
#define MY_P4EST_BIALLOY_H

#ifdef P4_TO_P8

#else
#include <src/casl_types.h>
#include <src/CASL_math.h>
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/my_p4est_interpolating_function_host.h>
#endif


class my_p4est_bialloy_t
{
private:

  PetscErrorCode ierr;

  BoundaryConditions2D bc_t;
  BoundaryConditions2D bc_cs;
  BoundaryConditions2D bc_cl;

  InterpolatingFunctionNodeBaseHost *interface_value_c;

  /* grid */
  my_p4est_brick_t *brick;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  double dx;
  double dy;

  /* temperature */
  Vec temperature_n, temperature_np1;
  Vec t_interface;

  /* concentration */
  Vec cs_n, cs_np1;
  Vec cl_n, cl_np1;
  Vec c_interface;

  /* velocity */
  Vec u_interface_n, u_interface_np1;
  Vec v_interface_n, v_interface_np1;
  Vec normal_velocity_np1;

  /* level-set */
  Vec phi;
  Vec nx, ny;
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
  double epsilon_anisotropy;   /* anisotropy coefficient */
  double epsilon_c;            /* curvature undercooling coefficient */
  double epsilon_v;            /* kinetic undercooling coefficient */

  bool solve_concentration_solid;

  bool matrices_are_constructed;
  PoissonSolverNodeBase solver_t;
  PoissonSolverNodeBase solver_c;
  Vec rhs;

public:

  my_p4est_bialloy_t(my_p4est_node_neighbors_t *ngbd);

  void set_parameters( double latent_heat,
                       double thermal_conductivity,
                       double thermal_diffusivity,
                       double solute_diffusivity_l,
                       double solute_diffusivity_s,
                       double cooling_velocity,
                       double kp,
                       double c0,
                       double ml,
                       double epsilon_anisotropy,
                       double epsilon_c,
                       double epsilon_v );

  void set_phi(Vec phi);

  void set_bc(WallBC2D& bc_wall_type_t,
              WallBC2D& bc_wall_type_c,
              CF_2& bc_wall_value_t,
              CF_2& bc_wall_value_cs,
              CF_2& bc_wall_value_cl);

  void set_temperature(Vec temperature);

  void set_concentration(Vec cl, Vec cs);

  void set_normal_velocity(Vec v);

  inline Vec get_phi() { return phi; }

  inline Vec get_normal_velocity() { return normal_velocity_np1; }

  inline double get_dt() { return dt_n; }

  void set_dt( double dt );

  void compute_normal_and_curvature();

  void compute_normal_velocity();

  void compute_velocity();

  void solve_temperature();

  void solve_concentration();

  void compute_dt();

  void update_grid();

  void one_step();

  void save_VTK(int iter);
};












#endif /* MY_P4EST_BIALLOY_H */
