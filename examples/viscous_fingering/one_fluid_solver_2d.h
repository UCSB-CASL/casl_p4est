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
  double alpha;
  CF_1 *Q, *I;
  cf_t *K_D, *K_EO, *gamma;
  cf_t *bc_wall_value;
  bc_wall_t *bc_wall_type;

  double advect_interface_semi_lagrangian(Vec& phi, Vec& pressure, Vec &potential, double cfl, double dtmax);
  double advect_interface_godunov(Vec& phi, Vec& pressure, Vec& potential, double cfl, double dtmax);
  void solve_fields(double t, Vec phi, Vec pressure, Vec potential);

public:
  one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick);
  ~one_fluid_solver_t();

  void set_properties(cf_t &K_D, cf_t &K_EO, cf_t &gamma);
  void set_injection_rates(CF_1& Q, CF_1& I, double alpha);
  void set_bc_wall(bc_wall_t& bc_wall_type, cf_t& bc_wall_value);
  double solve_one_step(double t, Vec &phi, Vec &pressure, Vec &potential, const std::string& method = "semi_lagrangian", double cfl = 5.0, double dtmax = DBL_MAX);
};

#endif // ONE_FLUID_SOLVER_2D_H
