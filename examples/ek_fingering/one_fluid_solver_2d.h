#ifndef ONE_FLUID_SOLVER_2D_H
#define ONE_FLUID_SOLVER_2D_H

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_node_neighbors.h>
#endif

class one_fluid_solver_t
{
  p4est_t* &p4est;
  p4est_ghost_t* &ghost;
  p4est_nodes_t* &nodes;
  my_p4est_brick_t *brick;

  p4est_t* p4est_nm1;
  p4est_ghost_t* ghost_nm1;
  p4est_nodes_t* nodes_nm1;
  Vec pressure_nm1, phi_nm1;

  double cfl, dtmax, dt_nm1;
  std::string method;

  // local variables
  splitting_criteria_t* sp;
  p4est_connectivity_t *conn;
#ifdef P4_TO_P8
  typedef CF_3 cf_t;
  typedef WallBC3D bc_wall_t;
#else
  typedef CF_2 cf_t;
  typedef WallBC2D bc_wall_t;
#endif
  cf_t *K_D, *K_EO, *gamma;
  cf_t *bc_wall_value;
  bc_wall_t *bc_wall_type;

  Vec kappa, nx[P4EST_DIM], n1[P4EST_DIM];
  Vec un, un_nm1;
  Vec vx_nm1[P4EST_DIM];

  void compute_normal_and_curvature_diagonal(my_p4est_node_neighbors_t& neighbors, Vec& phi);
  void compute_normal_velocity_diagonal(my_p4est_node_neighbors_t& neighbors, Vec& phi, Vec& pressure);

  double advect_interface_semi_lagrangian(Vec& phi, Vec& pressure);
  double advect_interface_semi_lagrangian_1st(Vec& phi, Vec& pressure);
  double advect_interface_godunov(Vec& phi, Vec& pressure);
  double advect_interface_normal(Vec& phi, Vec& pressure);
  double advect_interface_diagonal(Vec& phi, Vec& pressure);
  void solve_field(double t, Vec phi, Vec pressure);

public:
  one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick);
  ~one_fluid_solver_t(); 

  one_fluid_solver_t& set_properties(cf_t &K_D, cf_t &K_EO, cf_t &gamma);
  one_fluid_solver_t& set_bc_wall(bc_wall_t& bc_wall_type, cf_t& bc_wall_value);
  one_fluid_solver_t& set_integration(std::string method = "semi_lagrangian", double cfl = 5.0, double dtmax = DBL_MAX);
  double solve_one_step(double t, Vec &phi, Vec &pressure);
};

#endif // ONE_FLUID_SOLVER_2D_H
