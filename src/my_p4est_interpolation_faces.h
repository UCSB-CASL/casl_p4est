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

  u_char which_face;
  u_char degree;

  Vec face_is_well_defined_dir;

  BoundaryConditionsDIM* bc_array;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_faces_t(const my_p4est_interpolation_faces_t& other);
  my_p4est_interpolation_faces_t& operator=(const my_p4est_interpolation_faces_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces)
    : my_p4est_interpolation_t(ngbd_n), faces(faces), ngbd_c(faces->ngbd_c), face_is_well_defined_dir(NULL), bc_array(NULL) { }

  void set_input(const Vec F, const u_char &dir_, const u_char &degree_= 2, Vec face_is_well_defined_dir_ = NULL, BoundaryConditionsDIM *bc_array_ = NULL)
  {
    set_input(&F, dir_, 1, degree_, face_is_well_defined_dir_, bc_array_);
  }
  void set_input(const Vec *F, const u_char &dir_, const size_t &n_vecs_, const u_char &degree_= 2, Vec face_is_well_defined_dir_ = NULL, BoundaryConditionsDIM *bc_array_ = NULL);

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
  void operator()(const double *xyz, double *results, const u_int&) const; // last argument is dummy in this case

  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const u_int &) const; // last argument is dummy in this case
};

#endif /* MY_P4EST_INTERPOLATION_NODES_H */
