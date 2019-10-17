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

#ifdef P4_TO_P8
  const BoundaryConditions3D *bc;
#else
  const BoundaryConditions2D *bc;
#endif

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_cells_t(const my_p4est_interpolation_cells_t& other);
  my_p4est_interpolation_cells_t& operator=(const my_p4est_interpolation_cells_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n);

#ifdef P4_TO_P8
  void set_input(Vec F, Vec phi, const BoundaryConditions3D *bc);
#else
  void set_input(Vec F, Vec phi, const BoundaryConditions2D *bc);
#endif

  // interpolation methods
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif

  double interpolate(const p4est_quadrant_t &quad, const double *xyz) const;
};

#endif /* MY_P4EST_INTERPOLATION_CELLS_H */
