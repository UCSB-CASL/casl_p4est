#ifndef ONE_FLUID_SOLVER_2D_H
#define ONE_FLUID_SOLVER_2D_H

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_node_neighbors.h>
#endif

class two_fluid_solver_t
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
  double viscosity_ratio, Ca;
  CF_1 *Q;
  cf_t *bc_wall_value;
  bc_wall_t *bc_wall_type;

  double advect_interface(Vec& phi, Vec& press_m, Vec &press_p, double cfl, double dtmax);
  void solve_fields_extended(double t, Vec phi, Vec press_m, Vec press_p);
  void solve_fields_voronoi(double t, Vec phi, Vec press_m, Vec press_p);

public:
  two_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick);
  ~two_fluid_solver_t();

  void set_properties(double viscosity_ratio, double Ca, CF_1& Q);
  void set_bc_wall(bc_wall_t& bc_wall_type, cf_t& bc_wall_value);
  double solve_one_step(double t, Vec &phi, Vec &press_m, Vec &press_p, double cfl = 5, double dtmax = DBL_MAX);
};

#endif // ONE_FLUID_SOLVER_2D_H
