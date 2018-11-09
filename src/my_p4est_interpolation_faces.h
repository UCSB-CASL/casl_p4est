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

#ifdef P4_TO_P8
  BoundaryConditions3D* bc;
#else
  BoundaryConditions2D* bc;
#endif

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_faces_t(const my_p4est_interpolation_faces_t& other);
  my_p4est_interpolation_faces_t& operator=(const my_p4est_interpolation_faces_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces);

#ifdef P4_TO_P8
  void set_input(Vec F, int dir, int order=2, Vec face_is_well_defined=NULL, BoundaryConditions3D *bc=NULL);
#else
  void set_input(Vec F, int dir, int order=2, Vec face_is_well_defined=NULL, BoundaryConditions2D *bc=NULL);
#endif

  // interpolation methods
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif
  inline double operator()(double xyz[]) const
  {
#ifdef P4_TO_P8
    return this->operator ()(xyz[0], xyz[1], xyz[2]);
#else
    return this->operator ()(xyz[0], xyz[1]);
#endif
  }

  double interpolate(const p4est_quadrant_t &quad, const double *xyz) const;
};

#endif /* MY_P4EST_INTERPOLATION_NODES_H */
