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
public:
  /* local quadrants from 0 .. local_number_of_quadrants - 1
   * ghosts from local_number_of_quadrants .. local_number_of_quadrants + num_ghosts
   */
  struct quad_info_t{
    int8_t level;
    p4est_locidx_t locidx;
    p4est_gloidx_t gloidx;
  };

private:
  friend class PoissonSolverCellBase;
  friend class InterpolatingFunctionCellBase;

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;

  std::vector<quad_info_t> neighbor_cells;
  std::vector<p4est_locidx_t> offsets;
  p4est_locidx_t n_quads;


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

#ifdef P4_TO_P8
  void find_neighbor_cells_of_cell_recursive_test( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, char dir_x, char dir_y, char dir_z ) const;
#else
  void find_neighbor_cells_of_cell_recursive( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, char dir_x, char dir_y ) const;
#endif

public:
  my_p4est_cell_neighbors_t( my_p4est_hierarchy_t *hierarchy_ )
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), myb(hierarchy_->myb),
      n_quads(p4est->local_num_quadrants + ghost->ghosts.elem_count)
  {
  }

  /**
   * @brief initialize the buffers containing the information about the neighboring cell for
   * every local and ghost cell provided when instantiating the my_p4est_cell_neighbors_t structure.
   * This consumes a lot of memory, and it can improve the time performances of the code if repetitive
   * access to the neighbors information is required.
   */
  void init_neighbors();

  inline const quad_info_t* begin(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if(neighbor_cells.size()==0 || offsets.size()==0)
      throw std::invalid_argument("did you forget to call my_p4est_cell_neighbors_t::init_neighbors ?");
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &neighbor_cells[offsets[q*P4EST_FACES + dir_f]];
  }

  inline const quad_info_t* end(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if(neighbor_cells.size()==0 || offsets.size()==0)
      throw std::invalid_argument("did you forget to call my_p4est_cell_neighbors_t::init_neighbors ?");
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &neighbor_cells[offsets[q*P4EST_FACES + dir_f + 1]];
  }

  /**
   * @brief find the neighbor cell of a cell in the direction (dir_x, dir_y). Use this for finding corner/arete neighbors
   * @return
   */
#ifdef P4_TO_P8
  void find_neighbor_cells_of_cell_test(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir_x, char dir_y, char dir_z ) const;
#else
  void find_neighbor_cells_of_cell(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir_x, char dir_y ) const;
#endif

  void __attribute__((used)) print_debug(p4est_locidx_t q, FILE* stream = stdout);

};

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
