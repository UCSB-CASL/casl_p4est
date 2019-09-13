#ifndef MY_P4EST_HIERARCHY_H
#define MY_P4EST_HIERARCHY_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_tools.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_tools.h>
#endif

#include <vector>

// Following defines are used for child index in HierarchyCell
#define CELL_LEAF   -1          /* denotes a leaf cell */

// Following defines are used for owner_rank in HierarchyCell
#define REMOTE_OWNER -1         /* denotes that a cell does not have a owner in p4est representation, i.e. is a NOT_A_P4EST_QUADRANT */

// Following defines are used for the quad index in HierarchyCell
#define NOT_A_P4EST_QUADRANT -1 /* denotes a cell that does not exist in the p4est representation */
#define NOT_A_VALID_QUADRANT -2 /* denotes a quadrant that is not valid, i.e. does not exist in the hierarchy structure (e.g.
                                 * this is useful in dealing with quadrants that are outside the computational domain when
                                 * searching for neighboring quadrants of a quadrant or node */


#ifndef P4EST_VTK_CELL_TYPE
#ifdef P4_TO_P8
#define P4EST_VTK_CELL_TYPE 11 /* VTK_VOXEL */
#else
#define P4EST_VTK_CELL_TYPE 8  /* VTK_PIXEL */
#endif
#endif


/*
 * position of the children
 *   --- ---
 *  | 2 | 3 |
 *   --- ---
 *  | 0 | 1 |
 *   --- ---
 */

// forward declaration
#ifdef P4_TO_P8
#include "point3.h"
#else
#include "point2.h"
#endif

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
#ifdef P4_TO_P8
  p4est_qcoord_t kmin;
#endif

  /* the level of the cell */
  int8_t level;

  /* proc owner of the cell */
  int owner_rank;
};


class my_p4est_hierarchy_t {
  friend class my_p4est_cell_neighbors_t;
  friend class my_p4est_node_neighbors_t;

  p4est_t       *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  std::vector< std::vector<HierarchyCell> > trees;

  std::vector< p4est_locidx_t > local_inner_quadrant_index;
  std::vector< p4est_locidx_t > local_layer_quadrant_index;

  void split(int tree_idx, int ind );
  int update_tree( int tree_idx, const p4est_quadrant_t *quad );
  /*!
   * \brief construct_tree: constructs the local trees and fills the list of indices for local quadrants that are
   * either ghost for (an)other process(es) in local_layer_quadrant_index or owned locally only in local_inner_quadrant_index
   */
  void construct_tree();
#ifdef P4_TO_P8
  void find_quadrant_containing_point(const int* tr_xyz_orig, Point3& s, int& rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;
#else
  void find_quadrant_containing_point(const int* tr_xyz_orig, Point2& s, int& rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;
#endif

public:
  my_p4est_hierarchy_t( p4est_t *p4est_, p4est_ghost_t *ghost_, my_p4est_brick_t *myb_)
    : p4est(p4est_), ghost(ghost_), myb(myb_), trees(p4est->connectivity->num_trees)
  {
    for( size_t tr=0; tr<trees.size(); tr++)
    {
      HierarchyCell root =
      {
        CELL_LEAF, NOT_A_P4EST_QUADRANT, /* child, quad */
        0, 0,                            /* imin, jmin  */
  #ifdef P4_TO_P8
        0,                               /* kmin (3D only) */
  #endif
        0,                               /* level */
        REMOTE_OWNER                     /* owner's rank */
      };
      trees[tr].push_back(root);
    }
    construct_tree();
  }

  inline const HierarchyCell* get_cell(p4est_topidx_t tr, p4est_locidx_t q) const {return &trees[tr][q];}
  int find_smallest_quadrant_containing_point(double *xyz, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;
  p4est_locidx_t quad_idx_of_quad(const p4est_quadrant_t* quad, const p4est_topidx_t& tree_idx) const;
  void update(p4est_t *p4est_, p4est_ghost_t *ghost_);

  inline size_t get_layer_size() const { return local_layer_quadrant_index.size(); }
  inline size_t get_local_size() const { return local_inner_quadrant_index.size(); }
  inline p4est_locidx_t get_layer_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_layer_quadrant_index.at(i);
#endif
    return local_layer_quadrant_index[i];
  }
  inline p4est_locidx_t get_local_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_inner_quadrant_index.at(i);
#endif
    return local_inner_quadrant_index[i];
  }

  void write_vtk(const char* filename) const;

  size_t memory_estimate() const
  {
    size_t memory = 0;
    for (size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx)
      memory += (trees[tree_idx].size())*sizeof (HierarchyCell);
    memory += (local_inner_quadrant_index.size())*sizeof (p4est_locidx_t);
    memory += (local_layer_quadrant_index.size())*sizeof (p4est_locidx_t);

    return memory;
  }
};

#endif /* !MY_P4EST_HIERARCHY_H */
