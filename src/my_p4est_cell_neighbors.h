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

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;

  /* local quadrants from 0 .. local_number_of_quadrants - 1
   * ghosts from local_number_of_quadrants .. local_number_of_quadrants + num_ghosts
   */
  std::vector<p4est_locidx_t> neighbor_cells;
  std::vector<p4est_locidx_t> offsets;
  p4est_locidx_t n_quads;

  void initialize_neighbors();

  /**
     * perform the recursive search to find the neighboring cells of a cell
     */
  void find_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_f);

  /**
     * find the neighboring cells of cell in a given direction, the possible directions are :
     * \param [in] q      the index of the quadrant whose neighbors are queried
     * \param [in] tr     the tree in which the quadrant is located
     * \param [in] dir    the direction to investigate ( 0 = left, 1 = right, 2 = bottom, 3 = top )
     */
  void find_neighbor_cells_of_cell( const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tr, int dir_f);

public:
  my_p4est_cell_neighbors_t( my_p4est_hierarchy_t *hierarchy_ )
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost),
      n_quads(p4est->local_num_quadrants + ghost->ghosts.elem_count)
  {
    neighbor_cells.reserve(P4EST_FACES * n_quads);
    offsets.resize(P4EST_FACES*n_quads + 1, 0);

    initialize_neighbors();
  }

  inline const p4est_locidx_t* begin(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &neighbor_cells[offsets[q*P4EST_FACES + dir_f]];
  }

  inline const p4est_locidx_t* end(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &neighbor_cells[offsets[q*P4EST_FACES + dir_f + 1]];
  }

};

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
