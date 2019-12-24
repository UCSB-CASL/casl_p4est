#ifndef MY_P4EST_CELL_NEIGHBORS_H
#define MY_P4EST_CELL_NEIGHBORS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_utils.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_utils.h>
#endif

#include <vector>

class my_p4est_cell_neighbors_t {
private:
  friend class my_p4est_faces_t;
  friend class my_p4est_poisson_cells_t;
  friend class my_p4est_xgfm_cells_t;

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;

  void find_neighbor_cells_of_cell_recursive( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, DIM(char dir_x, char dir_y, char dir_z) ) const;

public:
  my_p4est_cell_neighbors_t( my_p4est_hierarchy_t *hierarchy_ )
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), myb(hierarchy_->myb)
  {
  }

  /**
   * @brief find the neighbor cell of a cell in the direction (dir_x, dir_y), any combination of directions is accepted
   * @return
   */
  void find_neighbor_cells_of_cell(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, DIM(char dir_x, char dir_y, char dir_z) ) const;

  /*!
   * \brief find the neighbor cells of a cell in a cartesian direction
   * \param ngbd the list of neighbor cells, to be filled
   * \param q the index of the quadrant in the local p4est, NOT in the tree tr
   * \param tr the tree to which the quadrant belongs
   * \param dir_f the cartesian direction to search, i.e. dir::f_m00, dir::f_p00 ...
   */
  inline void find_neighbor_cells_of_cell( std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t q, p4est_topidx_t tr, int dir_f) const
  {
    switch(dir_f)
    {
#ifdef P4_TO_P8
    case dir::f_m00: find_neighbor_cells_of_cell(ngbd, q, tr, -1, 0, 0); break;
    case dir::f_p00: find_neighbor_cells_of_cell(ngbd, q, tr,  1, 0, 0); break;
    case dir::f_0m0: find_neighbor_cells_of_cell(ngbd, q, tr,  0,-1, 0); break;
    case dir::f_0p0: find_neighbor_cells_of_cell(ngbd, q, tr,  0, 1, 0); break;
    case dir::f_00m: find_neighbor_cells_of_cell(ngbd, q, tr,  0, 0,-1); break;
    case dir::f_00p: find_neighbor_cells_of_cell(ngbd, q, tr,  0, 0, 1); break;
#else
    case dir::f_m00: find_neighbor_cells_of_cell(ngbd, q, tr, -1, 0); break;
    case dir::f_p00: find_neighbor_cells_of_cell(ngbd, q, tr,  1, 0); break;
    case dir::f_0m0: find_neighbor_cells_of_cell(ngbd, q, tr,  0,-1); break;
    case dir::f_0p0: find_neighbor_cells_of_cell(ngbd, q, tr,  0, 1); break;
#endif
    }
  }
};

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
