#ifndef MY_P4EST_HIERARCHY_H
#define MY_P4EST_HIERARCHY_H

#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_tools.h>

#include <vector>

#define CELL_LEAF   -1
#define NOT_A_P4EST_QUADRANT -1
#define REMOTE_OWNER -1

#ifndef P4EST_VTK_CELL_TYPE
#define P4EST_VTK_CELL_TYPE 8 /* VTK_PIXEL */
#endif


/*
 * position of the children
 *   --- ---
 *  | 2 | 3 |
 *   --- ---
 *  | 0 | 1 |
 *   --- ---
 */

struct HierarchyCell {
  /* index of the first child of the cell.
     * the children are located at child, child+1, ... child+P4EST_CHILDREN
     * if it is a leaf, set to CELL_LEAF */
  p4est_locidx_t child;

  /*
     * if quad is NOT_A_P4EST_QUADRANT, then the cell is not part of the p4est structure
     * if 0 < quad < num_quadrant, the corresponding quadrant is in the local quadrants of this processor / rank
     *        to get it from the tree it belongs to, you need to substract the tree offset
     * if num_quadrants <= quad, the corresponding quadrant is in the ghost layer
     *        to get it from the ghost layer, you need to substract the local_num_quadrants for this processor */
  p4est_locidx_t quad;

  /* the (integer) coordinates of the bottom left corner in the local tree */
  p4est_qcoord_t imin;
  p4est_qcoord_t jmin;

  /* the level of the cell */
  p4est_qcoord_t level;

  int owner_rank;

};


class my_p4est_hierarchy_t {
  friend class my_p4est_cell_neighbors_t;
  friend class my_p4est_node_neighbors_t;

  p4est_t       *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  std::vector< std::vector<HierarchyCell> > trees;

  void split(int tree_idx, int ind );
  int update_tree( int tree_idx, p4est_quadrant_t *quad );
  void construct_tree();

public:
  my_p4est_hierarchy_t( p4est_t *p4est_, p4est_ghost_t *ghost_, my_p4est_brick_t *myb_)
    : p4est(p4est_), ghost(ghost_), myb(myb_), trees(p4est->connectivity->num_trees)
  {
    for( size_t tr=0; tr<trees.size(); tr++)
    {
      struct HierarchyCell root = { CELL_LEAF, NOT_A_P4EST_QUADRANT, 0, 0, 0, REMOTE_OWNER};
      trees[tr].push_back(root);
    }
    construct_tree();
  }

  int find_smallest_quadrant_containing_point(double *xy, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;
  void write_vtk(const char* filename) const;
};

#endif /* !MY_P4EST_HIERARCHY_H */
