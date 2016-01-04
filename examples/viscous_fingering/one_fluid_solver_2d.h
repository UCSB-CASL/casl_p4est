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
  my_p4est_brick_t* brick;

  // local variables
  splitting_criteria_t* sp;
  p4est_connectivity_t *conn;
#ifdef P4_TO_P8
  typedef CF_3 cf_t;
#else
  typedef CF_2 cf_t;
#endif
  cf_t *K_D, *gamma, *p_applied;


  double advect_interface_semi_lagrangian(Vec& phi, Vec& pressure, double cfl);
  double advect_interface_godunov(Vec& phi, Vec& pressure, double cfl);
  void solve_pressure(Vec& pressure);

public:
  one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick);
  ~one_fluid_solver_t();

  void set_properties(cf_t &K_D, cf_t &gamma, cf_t &p_applied);
  double solve_one_step(Vec &phi, Vec &pressure, double cfl = 5);
};

#endif // ONE_FLUID_SOLVER_2D_H
