#ifndef MY_P4EST_LEVELSET_FACES_H
#define MY_P4EST_LEVELSET_FACES_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_faces.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_faces.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/math.h>

class my_p4est_level_set_faces_t
{
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  my_p4est_node_neighbors_t *ngbd_n;
  my_p4est_faces_t *faces;

public:
  my_p4est_level_set_faces_t(my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces )
    : p4est(ngbd_n->p4est), nodes(ngbd_n->nodes), ngbd_n(ngbd_n), faces(faces)
  {}

  /* extrapolate using geometrical extrapolation */
#ifdef P4_TO_P8
  void extend_Over_Interface( Vec phi, Vec q, BoundaryConditions3D &bc, int dir, Vec face_is_well_defined, Vec dxyz_hodge=NULL, int order=2, int band_to_extend=INT_MAX ) const;
#else
  void extend_Over_Interface( Vec phi, Vec q, BoundaryConditions2D &bc, int dir, Vec face_is_well_defined, Vec dxyz_hodge=NULL, int order=2, int band_to_extend=INT_MAX ) const;
#endif
};

#endif /* MY_P4EST_LEVELSET_FACES_H */
