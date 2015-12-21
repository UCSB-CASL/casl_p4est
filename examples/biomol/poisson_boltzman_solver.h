#ifndef POISSON_BOLTZMAN_SOLVER_H
#define POISSON_BOLTZMAN_SOLVER_H


#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>

class PoissonBoltzmanSolver{
  my_p4est_node_neighbors_t& neighbors;

  const p4est_t* p4est;
  const p4est_nodes_t *nodes;

  typedef enum {
    linearPB,
    nonlinearPB
  } solver_type;

  double edl, zeta;

  Vec phi;

public:
  PoissonBoltzmanSolver(my_p4est_node_neighbors_t& neighbors);
  void set_parameters(double edl, double zeta);
  void set_phi(Vec phi);
  void solve_linear(Vec& psi);
  void solve_nonlinear(Vec& psi, int itmax = 10, double tol = 1e-6);
};
#endif // POISSON_BOLTZMAN_SOLVER_H
