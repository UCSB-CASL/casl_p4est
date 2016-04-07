#ifndef MY_P4EST_EPITAXY_H
#define MY_P4EST_EPITAXY_H

#include <vector>

#include <src/types.h>

#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>

class my_p4est_epitaxy_t
{
private:
  PetscErrorCode ierr;

  class ZERO : public CF_2
  {
    double operator()(double, double) const
    {
      return 0;
    }
  } zero;

  my_p4est_brick_t *brick;
  p4est_t *p4est;
  p4est_connectivity_t *connectivity;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  Vec rho_g;

  std::vector<Vec> phi;
  std::vector<Vec> rho;
  std::vector<Vec> v[2];

  double dt_n;
  double D;
  double F;
  double sigma1;
  double Nuc;

  double alpha;
  double rho_avg;
  double rho_sqr_avg;

  double dxyz[P4EST_DIM];

public:
  my_p4est_epitaxy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_epitaxy_t();

  void compute_velocity();

  void solve_rho();

  void update_grid();

  void update_nucleation();

  void one_step();

  void save_vtk();
};


#endif /* MY_P4EST_EPITAXY_H */
