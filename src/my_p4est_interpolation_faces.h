#ifndef MY_P4EST_INTERPOLATION_FACES_H
#define MY_P4EST_INTERPOLATION_FACES_H

#ifdef P4_TO_P8
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation.h>
#else
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation.h>
#endif

class my_p4est_interpolation_faces_t : public my_p4est_interpolation_t
{
private:
  const my_p4est_faces_t *faces;
  const my_p4est_cell_neighbors_t *ngbd_c;

  int dir;
  int order;

  Vec face_is_well_defined;

  BoundaryConditionsDIM* bc;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_faces_t(const my_p4est_interpolation_faces_t& other);
  my_p4est_interpolation_faces_t& operator=(const my_p4est_interpolation_faces_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces);

  using my_p4est_interpolation_t::set_input;
  void set_input(Vec F,   unsigned char dir, int order=2, Vec face_is_well_defined=NULL, BoundaryConditionsDIM *bc=NULL){ set_input(&F, dir, 1, order, face_is_well_defined, bc); }
  void set_input(Vec *F,  unsigned char dir, unsigned int n_vecs_, int order=2, Vec face_is_well_defined=NULL, BoundaryConditionsDIM *bc=NULL);

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
  void operator()(DIM(double x, double y, double z), double *results) const;

  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const unsigned int &comp) const;
};

#endif /* MY_P4EST_INTERPOLATION_NODES_H */
