#ifndef COUPLED_SOLVER_2D_H
#define COUPLED_SOLVER_2D_H

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_node_neighbors.h>
#endif

class coupled_solver_t
{
public:
  struct parameters {
    double alpha, beta, Ca, mue, eps, sigma;
  };

private:
  p4est_t* &p4est;
  p4est_ghost_t* &ghost;
  p4est_nodes_t* &nodes;
  my_p4est_brick_t *brick;

  // local variables
  splitting_criteria_t* sp;
  p4est_connectivity_t *conn;
#ifdef P4_TO_P8
  typedef CF_3 cf_t;
  typedef WallBC3D wall_bc_t;
  typedef BoundaryConditions3D bc_t;
#else
  typedef CF_2 cf_t;
  typedef WallBC2D wall_bc_t;
  typedef BoundaryConditions2D bc_t;
#endif
  parameters params;
  wall_bc_t *pressure_bc_type, *potential_bc_type;
  cf_t *pressure_bc_val, *potential_bc_val;
  const CF_1 *Q, *I;

  double advect_interface(Vec& phi,
                          Vec& pressure_m,  Vec& pressure_p,
                          Vec& potential_m, Vec& potential_p,
                          double cfl, double dtmax);

  void solve_fields(double t, Vec phi,
                    Vec pressure_m,  Vec pressure_p,
                    Vec potential_m, Vec potential_p);

public:
  coupled_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick);
  ~coupled_solver_t();

  void set_parameters(const parameters& p);
  void set_injection_rates(const CF_1& Q, const CF_1& I);
  void set_boundary_conditions(wall_bc_t& pressure_bc_type,  cf_t& pressure_bc_val,
                               wall_bc_t& potential_bc_type, cf_t& potential_bc_val);
  double solve_one_step(double t, Vec &phi,
                        Vec &pressure_m,  Vec &pressure_p,
                        Vec &potential_m, Vec &potential_p,
                        double cfl = 5, double dtmax = DBL_MAX);
};

#endif // COUPLED_SOLVER_2D_H
