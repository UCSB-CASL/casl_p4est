#ifndef MY_P4EST_POISSON_BOLTZMANN_NODES_H
#define MY_P4EST_POISSON_BOLTZMANN_NODES_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#endif

class my_p4est_poisson_boltzmann_nodes_t{
  my_p4est_node_neighbors_t& neighbors;

  const p4est_t* p4est;
  const p4est_nodes_t *nodes;

  double edl, zeta;

  Vec phi;

public:
  my_p4est_poisson_boltzmann_nodes_t(my_p4est_node_neighbors_t& neighbors);
  void set_parameters(double edl, double zeta);
  void set_phi(Vec phi);
  void solve_linear(Vec& psi);
  void solve_nonlinear(Vec& psi, int itmax = 10, double tol = 1e-6);
};
#endif // MY_P4EST_POISSON_BOLTZMANN_NODES_H
