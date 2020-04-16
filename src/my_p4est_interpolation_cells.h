#ifndef MY_P4EST_INTERPOLATION_CELLS_H
#define MY_P4EST_INTERPOLATION_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_interpolation.h>
#else
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_interpolation.h>
#endif

class my_p4est_interpolation_cells_t : public my_p4est_interpolation_t
{
private:
  const p4est_nodes_t *nodes;
  const my_p4est_cell_neighbors_t *ngbd_c;

  Vec phi;

  const BoundaryConditionsDIM *bc;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_cells_t(const my_p4est_interpolation_cells_t& other);
  my_p4est_interpolation_cells_t& operator=(const my_p4est_interpolation_cells_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n)
    : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes), ngbd_c(ngbd_c), phi(NULL), bc(NULL) {};

  void set_input(Vec *F, Vec phi_, const BoundaryConditionsDIM *bc_, const size_t &n_vecs_);
  inline void set_input(Vec F, Vec phi_, const BoundaryConditionsDIM *bc_) { set_input(&F, phi_, bc_, 1); }

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
  void operator()(const double *xyz, double *results) const;

  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const unsigned int &comp) const;
};

#endif /* MY_P4EST_INTERPOLATION_CELLS_H */
