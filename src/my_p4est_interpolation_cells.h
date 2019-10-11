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

// Elyce: Class is used to interpolate data sampled at cell centers to the nodes

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

  double tree_dimension[P4EST_DIM], domain_dimension[P4EST_DIM];

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_cells_t(const my_p4est_interpolation_cells_t& other);
  my_p4est_interpolation_cells_t& operator=(const my_p4est_interpolation_cells_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n);

  using my_p4est_interpolation_t::set_input;
#ifdef P4_TO_P8
  void set_input(Vec F, Vec phi, const BoundaryConditions3D *bc) {set_input(&F, phi, bc, 1); }
  void set_input(Vec *F, Vec phi, const BoundaryConditions3D *bc, unsigned int n_vecs_);
#else
  void set_input(Vec F, Vec phi, const BoundaryConditions2D *bc) {set_input(&F, phi, bc, 1); }
  void set_input(Vec *F, Vec phi, const BoundaryConditions2D *bc, unsigned int n_vecs_);
#endif

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
#ifdef P4_TO_P8
  void operator()(double x, double y, double z, double* results) const;
#else
  void operator()(double x, double y, double* results) const;
#endif

  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results) const;
};

#endif /* MY_P4EST_INTERPOLATION_CELLS_H */
