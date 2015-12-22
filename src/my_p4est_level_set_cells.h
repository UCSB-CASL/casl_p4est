#ifndef MY_P4EST_LEVELSET_CELLS_H
#define MY_P4EST_LEVELSET_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/math.h>

class my_p4est_level_set_cells_t
{
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_node_neighbors_t *ngbd_n;

public:
  my_p4est_level_set_cells_t(my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n)
    : p4est(ngbd_n->p4est), ghost(ngbd_n->ghost), nodes(ngbd_n->nodes), ngbd_c(ngbd_c), ngbd_n(ngbd_n)
  {}

  double integrate_over_interface(Vec phi, Vec f) const;

  /*!
   * \brief integrate f dot grad(phi) over the irregular interface phi
   * \param phi[in]       the level-set function
   * \param f[in]         the scalar to integrate
   * \param integral[out] the integral of f dot grad(phi) in each dimension
   */
  void integrate_over_interface(Vec phi, Vec f, double *integral) const;

  double integrate(Vec phi, Vec f) const;

  /* extrapolate using geometrical extrapolation */
#ifdef P4_TO_P8
  void extend_Over_Interface( Vec phi, Vec q, BoundaryConditions3D *bc, int order=2, int band_to_extend=INT_MAX ) const;
#else
  void extend_Over_Interface( Vec phi, Vec q, BoundaryConditions2D *bc, int order=2, int band_to_extend=INT_MAX ) const;
#endif
};

#endif /* MY_P4EST_LEVELSET_CELLS_H */
