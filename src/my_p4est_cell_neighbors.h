#ifndef MY_P4EST_CELL_NEIGHBORS_H
#define MY_P4EST_CELL_NEIGHBORS_H

#include <p4est.h>
#include <p4est_ghost.h>

#include <src/my_p4est_hierarchy.h>

#include <vector>

class my_p4est_cell_neighbors_t {

private:
  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;

  /* local quadrants from 0 .. local_number_of_quadrants - 1
     * ghosts from local_number_of_quadrants .. local_number_of_quadrants + num_ghosts
     * possible directions :
     * 0 = left
     * 1 = right
     * 2 = bottom
     * 3 = top
     */
  std::vector< std::vector<p4est_locidx_t> > neighbor_cells [4];

  void initialize_neighbors();

  /**
     * perform the recursive search to find the neighboring cells of a cell
     */
  void find_neighbor_cells_of_cell_recursive( p4est_locidx_t q, p4est_topidx_t tr, int ind, int dir );

  /**
     * find the neighboring cells of cell in a given direction, the possible directions are :
     * \param [in] q      the index of the quadrant whose neighbors are queried
     * \param [in] tr     the tree in which the quadrant is located
     * \param [in] dir    the direction to investigate ( 0 = left, 1 = right, 2 = bottom, 3 = top )
     */
  void find_neighbor_cells_of_cell( p4est_locidx_t q, p4est_topidx_t tr, int dir );

public:
  my_p4est_cell_neighbors_t( my_p4est_hierarchy_t *hierarchy_ )
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost)
  {
    for(int i=0; i<4; ++i)
      neighbor_cells[i].resize(p4est->local_num_quadrants);
    initialize_neighbors();
  }

};

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
