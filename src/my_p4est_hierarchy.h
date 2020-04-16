#ifndef MY_P4EST_HIERARCHY_H
#define MY_P4EST_HIERARCHY_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_utils.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_utils.h>
#endif

#include <vector>

// Following defines are used for child index in HierarchyCell
#define CELL_LEAF   -1          /* denotes a leaf cell */

// Following defines are used for owner_rank in HierarchyCell
#define REMOTE_OWNER -1         /* denotes that a cell does not have a owner in p4est representation, i.e. is a NOT_A_P4EST_QUADRANT */

// Following defines are used for the quad index in HierarchyCell
#define NOT_A_P4EST_QUADRANT -1 /* denotes a cell that does not exist in the local p4est representation: not in the locally
                                 * owned quadrants and not in the ghosts. All space-filling quadrants outside of the local
                                 * domain partition are marked NOT_A_P4EST_QUADRANT */
#define NOT_A_VALID_QUADRANT -2 /* denotes a quadrant that is not valid, i.e. does not exist in the hierarchy structure (e.g.
                                 * this is useful in dealing with quadrants that are outside the (full) computational domain
                                 * when searching for neighboring quadrants of a quadrant or node.
                                 * (This NOT_A_VALID_QUADRANT is not used in the current class, but in find_neighbor_cell_of_node
                                 * from my_p4est_node_neighbors, for instance: if the node is a wall node and the domain is not
                                 * periodic, there might be no valid cell in the direction of interest and hence it is marked
                                 * NOT_A_VALID_QUADRANT */


#ifndef P4EST_VTK_CELL_TYPE
#ifdef P4_TO_P8
#define P4EST_VTK_CELL_TYPE 11 /* VTK_VOXEL */
#else
#define P4EST_VTK_CELL_TYPE 8  /* VTK_PIXEL */
#endif
#endif



// forward declaration
#ifdef P4_TO_P8
#include "point3.h"
#else
#include "point2.h"
#endif

/*!
 * \brief The HierarchyCell struct represents ANY cell in the hierarchy, leaf or not.
 * [Comments by Raphael Egan, October 2019]
 */
struct HierarchyCell {
  /*!
   * \brief child index of the 0th child of the cell in the std::vector<HierarchyCell>.
   * The kth child cell (0 <= k < P4EST_CHILDREN) is located at index child+k in the
   * same std::vector<HierarchyCell> of the my_p4est_hirerachy_t object.
   * k has the following values for the children cells in 2D
   *                 --- ---
   * ^ e_{y}        | 2 | 3 |
   * |               --- ---
   * |   e_{x}      | 0 | 1 |
   *  ---->          --- ---
   * In 3D, the above stands for the 4 back-facing children, i.e. the children oriented
   * toward smaller values of cartesian coordinate along e_{z}, while the 4 front-facing
   * children, i.e. the children oriented toward larger values of cartesian coordinate
   * along e_{z}, are given the following k values
   *                 --- ---
   * ^ e_{y}        | 6 | 7 |
   * |               --- ---
   * |   e_{x}      | 4 | 5 |
   *  ---->          --- ---
   * To summarize, a child cell is uniquely identified by P4EST_DIM boolean values
   * {is_toward_positive_x, is_toward_positive_y (, is_toward_positive_z)}
   * and the corresponding k value is then given by
   * k = 4*(is_toward_positive_z?1:0)+2*(is_toward_positive_y?1:0)+(is_toward_positive_x?1:0), in 3D,
   * k =                              2*(is_toward_positive_y?1:0)+(is_toward_positive_x?1:0), in 2D.
   *
   * If the current HierarchyCell is a leaf, child is set to CELL_LEAF
   */
  p4est_locidx_t child;

  /*!
   * \brief quad is the local quadrant index of the HierarchyCell IF IT IS A CELL_LEAF and if it is known by the
   * local domain partition (ghost quadrants included).
   * In such a case,
   * - if 0 <= quad < num_quadrant, the corresponding p4est_quadrant_t is owned by this processor / rank
   *   and it can be accessed with the 'quadrant' pointer obtained by
   *      p4est_tree_t *tree          = p4est_tree_array_index(p4est->trees, tree_index)
   *      p4est_quadrant_t *quadrant  = p4est_quadrant_array_index(tree->quadrants, quad-tree->quadrants_offset)
   * - if num_quadrants <= quad, the corresponding quadrant is in the ghost layer and it can be accessed locally
   *   with the 'quadrant' pointer obtained by
   *      p4est_quadrant_t *quadrant  = p4est_quadrant_array_index(&ghost->ghosts, quad-p4est->local_num_quadrants)
   *
   * If the HierarchyCell is not a leaf or if it is nothing but a space-filling HierarchyCell (i.e. representing
   * the rest of the computational domain that is not known by the local domain partition including ghost), quad is
   * set to NOT_A_P4EST_QUADRANT.
   */
  p4est_locidx_t quad;

  /*!
   * \brief imin, jmin and kmin are the (integer) coordinates of the bottom left corner of the HierarchyCell in the
   * local tree. Such integer coordinates range from 0 to P4EST_LAST_OFFSET(maximum level of refinement) and uniquely
   * define any grid node/cell in the forest.
   */
  p4est_qcoord_t imin;
  p4est_qcoord_t jmin;
#ifdef P4_TO_P8
  p4est_qcoord_t kmin;
#endif

  /*!
   * \brief level the level of refinement of the HierarchyCell
   */
  int8_t level;

  /*!
   * \brief owner_rank the rank of the process owning the HierarchyCell. If the HierarchyCell is nothing but a
   * space-filling HierarchyCell (i.e. representing the rest of the computational domain that is not known by the
   * local domain partition including ghost), owner_rank is set to REMOTE_OWNER.
   */
  int owner_rank;
};

struct local_and_tree_indices
{
  p4est_locidx_t local_idx; // cumulative over the local trees
  p4est_topidx_t tree_idx;
  local_and_tree_indices(const p4est_locidx_t& loc_idx = -1, const p4est_topidx_t& tr_idx = -1) : local_idx(loc_idx), tree_idx(tr_idx) {}
};

/*!
 * \brief The my_p4est_hierarchy_t class represents a local, single-process reconstruction of the
 * local domain partition in a "conventional" quad-/oc-tree fashion: every tree in the forest is
 * associated with a standard vector of HierarchyCell's constructed in order to match the locally
 * known grid-partition (including ghost quadrants), exactly. This standard vector encodes all cells
 * in the tree(s) (not only the leaves): the parent-to-children relationship is encoded through the
 * 'child' member value of every HierarchyCell (see description in HierarchyCell, here above)
 * The my_p4est_hierarchy objects are constructed by looping through every quadrant (local and ghost)
 * known by the current process: for each such quadrant q, the relevant tree in the hierarchy is refined
 * correspondingly, if needed, in order to capture q. The rest of the domain (i.e. not locally known)
 * is tagged as "NOT_A_P4EST_QUADRANT" and filled with cells that are as coarse as possible.
 *
 * Contrary to the p4est encoding of the local grid partition and of the ghost layer, the my_p4est_hierarchy
 * object stores a parent-to-children encoding map across all levels of refinments (from the root cell(s)
 * to the finest cell(s)). Therefore, this object allows for a straightforward identification of the
 * local/ghost quadrant that contains a point of interest, given its coordinates: first, the relevant
 * tree containing the given point is found, then the point's Cartesian coordinates are compared to the
 * center of the root cell in order to find which of its children cells contains the point of interest,
 * the procedure moves to that specific child cell and repeats recursively the same procedure with its own
 * center point until the considered cell is marked as a CELL_LEAF.
 * ILLUSTRATION:
 * say that the point of interest is 'x' and is located in the following two-dimensional 1/3 local tree
 *
 * _________________________________________________________________
 * |               |       |       |                               |
 * |               |       |       |                               |
 * |               |_______|_______|                               |
 * |               |       |       |                               |
 * |               |       |       |                               |
 * |_______________|_______|_______|                               |
 * |       |       |       |       |                               |
 * |       |       |       |       |                               |
 * |_______|_______|_______|_______|                               |
 * |       |       |       |       |                               |
 * |       |       |       |       |                               |
 * |_______|_______|_______|_______|_______________________________|
 * |       |       |       |       |       |       |               |
 * |       |       | x     |       |       |       |               |
 * |_______|_______|_______|_______|_______|_______|               |
 * |       |       |       |       |       |       |               |
 * |       |       |       |       |       |       |               |
 * |_______|_______|_______|_______|_______|_______|_______________|
 * |               |       |       |               |       |       |
 * |               |       |       |               |       |       |
 * |               |_______|_______|               |_______|_______|
 * |               |       |       |               |       |       |
 * |               |       |       |               |       |       |
 * |_______________|_______|_______|_______________|_______|_______|
 *
 * Starting from the root cell, the hierachy sees
 * _________________________________________________________________
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |          (child 2)            |          (child 3)            |
 * |          NOT LEAF             |            LEAF               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |_______________________________|_______________________________|
 * |                               |                               |
 * |                 x             |                               |
 * |                               |                               |
 * |                               |                               |
 * |          (child 0)            |         (child 1)             |
 * |          NOT LEAF             |          NOT LEAF             |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |                               |                               |
 * |_______________________________|_______________________________|
 *
 * and knows it needs to move on to its 0th child, from which it will repeat the procedure, since
 * child 0 is not a leaf cell. Restarting the procedure from child 0, the hierarchy sees
 * ________________________________
 * |               |               |
 * |    (child 2)  | x  (child 3)  |
 * |    NOT LEAF   |    NOT LEAF   |
 * |               |               |
 * |               |               |
 * |_______________|_______________|
 * |               |               |
 * |               |               |
 * |    child 0    |    (child 1)  |
 * |     LEAF      |    NOT LEAF   |
 * |               |               |
 * |_______________|_______________|
 * and knows it needs to move on to its 3rd child, from which it will repeat the procedure, since
 * child 3 is not a leaf cell. Restarting the procedure from child 3, the hierarchy sees
 *
 *                  _______________
 *                 |  (c2) | (c3)  |
 *                 | x LEAF|  LEAF |   ('ci' stands for 'child i')
 *                 |_______|_______|
 *                 | (c0)  | (c1)  |
 *                 | LEAF  | LEAF  |
 *                 |_______|_______|
 *
 * and the final leaf has been found to be the 2nd child cell (marked as CELL_LEAF) of the current cell.
 *
 * NOTE: this is a very convenient procedure, inspired from the former, serial casl library. However, it comes
 * with the extra cost of basically copying the entire grid with extra information. While p4est stores only the
 * leaf cells in increasing z-index order, the my_p4est_hierarchy needs to also store the parent-to-child
 * information, which may become significant for large levels of refinements... This was the standard
 * status of the library when the current team of developers/users joined the group (around 2015) and the
 * casl_p4est library was built and further developed upon this very fundamental building brick.
 *
 * However, it is conceptually and theoretically possible to perform the exact same tasks with built-in functions
 * from the p4est library exploiting binary searches based on the Morton index value of the smallest possible
 * quadrant containing the point of interest. One can actually find even more information using such functions,
 * like the process owning a remote point of interest (not known locally by the current process and/or its ghost
 * layer). This would probably have two serious advantages:
 * - no need to construct such a structure for every new grid;
 * - reduce memory requirements;
 * - very likely more efficient procedure overall (less random memory access).
 * [Comment by Raphael Egan, October 2019]
 */
class my_p4est_hierarchy_t {
  friend class my_p4est_cell_neighbors_t;
  friend class my_p4est_node_neighbors_t;

  p4est_t       *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  /*!
   * \brief trees: standard vector of standard vectors of HierarchyCell's
   * trees[tr_idx] is the standard vector of HierarchyCells built from the root cell corresponding to the tree
   * of index tr_idx.
   */
  std::vector< std::vector<HierarchyCell> > trees;

  /*!
   * \brief local_inner_quadrant_index: standard vector of local quadrant indices that are not seen as ghost by
   * any other process
   */
  std::vector< local_and_tree_indices > local_inner_quadrant;
  /*!
   * \brief local_layer_quadrant_index: standard vector of local quadrant indices that are seen as ghost by at
   * least one other process
   */
  std::vector< local_and_tree_indices > local_layer_quadrant;

  /*!
   * \brief split splits a HierarchyCell, adds the corresponding P4EST_CHILDREN HierarchyCell's to the same
   * relevant standard vector of HierarchyCell's
   * \param [in] tree_idx index of the tree in which the HierarchyCell needs to be split
   * \param [in] ind      index of the HierarchyCell to be split in trees[tree_idx]
   */
  void split(int tree_idx, int ind );
  /*!
   * \brief update_tree further refines the local hierarchy to match the existence of a given local quadrant
   * \param [in] tree_idx tree index in which the given quadrant exists
   * \param [in] quad     a pointer to the existing quadrant to be captured by the hierarchy
   * \return the index of the HierachyCell in trees[tree_idx] that matches the given quadrant, i.e.
   *          trees[tree_idx][returned value] is a HierarchyCell representation of quad
   */
  int update_tree( int tree_idx, const p4est_quadrant_t *quad );

  /*!
   * \brief construct_tree loops through every quadrant locally known (locally owned or ghost) and adds it to
   * as a match requirement for the my_p4est_hierarchy object. This also fills the list of indices for local
   * quadrants that are either ghost for (an)other process(es) in local_layer_quadrant_index or owned locally
   * only in local_inner_quadrant_index
   */
  void construct_tree();

  /*!
   * \brief find_quadrant_containing_point determines the quadrant locally known that contains a point
   * \param [in]    tr_xyz_orig     arrays of cartesian indices of the tree suspected to own the point of interest
   * \param [inout] s               the point of interest with coordinates scaled relatively to the tree, i.e. in [0, P4EST_ROOT_LEN]
   *                                (can also be slightly negative or slightly past P4EST_ROOT_LEN, the appropriate tree will be found then)
   * \param [out] rank              on return, rank of the processor owning the local quadrant, if it is found; unchanged if not found.
   * \param [out] best_match        on return, copy of the local quadrant if found, unchanged otherwise. (The p.piggy3 member is filled)
   * \param [out] remote_matches    on return, if a quadrant was not found locally, this vector is filled with theoretical candidate quadrants
   *                                of finest theoretical level of refinement, with their p.piggy1 members filled (i.e. owner rank and tree
   *                                index).
   */
  void find_quadrant_containing_point(const int* tr_xyz_orig, PointDIM& s, int& rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;
  bool periodic[P4EST_DIM];

public:
  /*!
   * \brief my_p4est_hierarchy_t constructor for my_p4est_hierarchy objects
   * \param [in] p4est_ the standard p4est_t structure representing the local domain partition
   * \param [in] ghost_ the standard p4est_ghost_t structure of of ghost quadrants known by the
   *                    local partition  (may be NULL if no ghost quadrant is known)
   * \param [in] myb_   macromesh description (cartesian array of elementary root cells)
   */
  my_p4est_hierarchy_t( p4est_t *p4est_, p4est_ghost_t *ghost_, my_p4est_brick_t *myb_)
    : p4est(p4est_), ghost(ghost_), myb(myb_), trees(p4est->connectivity->num_trees)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      periodic[dir] = is_periodic(p4est, dir);
    for( size_t tr=0; tr<trees.size(); tr++)
    {
      HierarchyCell root =
      {
        CELL_LEAF, NOT_A_P4EST_QUADRANT, /* child, quad */
        DIM(0, 0, 0),                    /* imin, jmin, kmin  */
        0,                               /* level */
        REMOTE_OWNER                     /* owner's rank */
      };
      trees[tr].push_back(root);
    }
    construct_tree();
  }

  inline const bool *get_periodicity() const { return periodic; }

  /*!
   * \brief get_cell gives read access to a constant HierarchyCell as constructed by the my_p4est_hierarchy object
   * \param [in] tr the tree index;
   * \param [in] q  the index of the HierarchyCell in that tree;
   * \return a pointer to the desired (constant) HierarchyCell
   */
  inline const HierarchyCell* get_cell(p4est_topidx_t tr, p4est_locidx_t q) const {return &trees[tr][q];}

  /*!
   * \brief find_smallest_quadrant_containing_point find the smallest (leaf) quadrant containing a point of interest
   * given by its cartesian coordinates. If the point lies to close
   * \param [in]    xyz             Cartesian coordinates of the point of interest (array of P4EST_DIM doubles)
   *                                xyz needs to be in the domain: the point is wrapped within the domain if domain
   *                                periodicity allows that.
   * \param [out] best_match        p4est_quadrant_t containing the point of interest on output. This argument must exist
   *                                beforehand, its member variables are wiped out at initiation stage of the function.
   *                                If found, the appropriate p4est_quadrant in p4est or ghost, is copied into best_match
   *                                and the p.piggy3 structure of best_match is also filled so that
   *                                best_match.p.piggy3.which_tree is the index of the tree owning best_match
   *                                best_match.p.piggy3.local_num  is either
   *                                  ~ the index of best match in tree->quadrants,
   *                                    where tree = p4est_tree_array_index(p4est->trees, tree_index),
   *                                    if the quadrant is owned locally (note that you don't need tree->quadrant_offest here);
   *                                  ~ or the index of best match in ghost->ghosts
   *                                    if the quadrant belongs to the ghost layer;
   *                                --> check if the returned value == mpirank to know if you are in the first or second case
   *                                If not found, best_match is left blank, as the result of P4EST_QUADRANT_INIT(&best_match)
   * \param [out] remote_matches    If a p4est_quadrant was not found locally, this vector is filled with candidate quadrants
   *                                of finest theoretical level of refinement, with their p.piggy1 members filled (i.e. owner
   *                                rank and tree index). This is relevant if the value returned by the function is -1.
   * [NOTE 1]: this function assumes that all building bricks in the macromesh description have the same size (i.e. no
   *           contraction/stretching, every building brick is identical).
   * [THROWS]: in DEBUG, this function throws an std::runtime_error exception if the procedure runs into an internal
   *           inconsistency (if it found a remote, space-filling HierarchyCell that is not marked NOT_A_P4EST_QUADRANT)
   * \return the returned values is either
   *  - the rank of the processor owning the quadrant in which the point of interest can be found, if the quadrant was found
   *    locally. (Therefore either mpirank is locally owned, or the rank of one of the neighbor process owning a ghost quadrant)
   *  - a value of -1 if no local quadrant could be found (the user is advised to check the content of remote_matches in such a
   *    case)
   */
  int find_smallest_quadrant_containing_point(const double *xyz, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches) const;

  /*!
   * \brief quad_idx_of_quad finds the index in the hierarchy tree of a quadrant
   * \param [in] quad     pointer to the quadrant whose local index is sought
   * \param [in] tree_idx index of the tree in which the quadrant lives
   * \return the index in the hierarchy tree of the HierarchyCell corresponding to the given quadrant
   */
  p4est_locidx_t quad_idx_of_quad(const p4est_quadrant_t* quad, const p4est_topidx_t& tree_idx) const;

  /*!
   * \brief update update the my_p4est_hiearchy to match the new p4est and ghost
   * \param [in] p4est_ pointer to the new p4est structure
   * \param [in] ghost_ pointer to the new ghost structure
   * [NOTE:] this procedure systematically wipes out the member variables of the object and reconstructs it from scratch, there
   *         is no shortcut return if unchanged pointer(s)
   */
  void update(p4est_t *p4est_, p4est_ghost_t *ghost_);

  inline size_t get_layer_size() const { return local_layer_quadrant.size(); }
  inline size_t get_inner_size() const { return local_inner_quadrant.size(); }
  inline p4est_locidx_t get_local_index_of_layer_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_layer_quadrant.at(i).local_idx;
#endif
    return local_layer_quadrant[i].local_idx;
  }
  inline p4est_topidx_t get_tree_index_of_layer_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_layer_quadrant.at(i).tree_idx;
#endif
    return local_layer_quadrant[i].tree_idx;
  }
  inline p4est_locidx_t get_local_index_of_inner_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_inner_quadrant.at(i).local_idx;
#endif
    return local_inner_quadrant[i].local_idx;
  }
  inline p4est_topidx_t get_tree_index_of_inner_quadrant(const size_t& i) const {
#ifdef CASL_THROWS
    return local_inner_quadrant.at(i).tree_idx;
#endif
    return local_inner_quadrant[i].tree_idx;
  }

  /*!
   * \brief write_vtk exports the local hierarchy in a vtk format
   * \param [in] filename name of the desired vtk file, every process exports such a file appending its rank to the desired name
   */
  void write_vtk(const char* filename) const;

  size_t memory_estimate() const
  {
    size_t memory = 0;
    for (size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx)
      memory += (trees[tree_idx].size())*sizeof (HierarchyCell);
    memory += (local_inner_quadrant.size())*sizeof (local_and_tree_indices);
    memory += (local_layer_quadrant.size())*sizeof (local_and_tree_indices);

    return memory;
  }
};

#endif /* !MY_P4EST_HIERARCHY_H */
