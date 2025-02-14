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

// forward declaration
#ifdef P4_TO_P8
#include "point3.h"
#else
#include "point2.h"
#endif

struct local_and_tree_indices
{
  p4est_locidx_t local_idx; // cumulative over the local trees
  p4est_topidx_t tree_idx;
  local_and_tree_indices(const p4est_locidx_t& loc_idx = -1, const p4est_topidx_t& tr_idx = -1) : local_idx(loc_idx), tree_idx(tr_idx) {}
};

class my_p4est_hierarchy_t {
  friend class my_p4est_cell_neighbors_t;
  friend class my_p4est_node_neighbors_t;

  p4est_t       *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  const static p4est_qcoord_t smallest_logical_quad_size = P4EST_QUADRANT_LEN(P4EST_QMAXLEVEL);

  /*!
   * \brief local_inner_quadrant_index: standard vector of local quadrant indices that are not seen as ghost by
   * any other process
   */
  std::vector<local_and_tree_indices> local_inner_quadrant;
  /*!
   * \brief local_layer_quadrant_index: standard vector of local quadrant indices that are seen as ghost by at
   * least one other process
   */
  std::vector<local_and_tree_indices> local_layer_quadrant;
  const bool periodic[P4EST_DIM];

  /*!
   * \brief find_quadrant_containing_point determines the quadrant locally known that contains a point
   * \param [in] tr_xyz_orig        arrays of cartesian indices of the tree suspected to own the point of interest
   * \param [inout] xyz_point       coordinates of the point of interest, scaled relatively to the tree, i.e. in [0, P4EST_ROOT_LEN]
   *                                (can also be slightly negative or slightly past P4EST_ROOT_LEN, the appropriate tree will be found then)
   * \param [inout] current_rank    the rank of the process owning the current best candidate quadrant found so far
   * \param [out] best_match        on return, copy of the local quadrant if found, unchanged otherwise. (The p.piggy3 member is filled)
   * \param [out] remote_matches    on return, if a quadrant was not found locally, this vector is filled with theoretical candidate quadrants
   *                                of finest theoretical level of refinement, with their p.piggy1 members filled (i.e. owner rank and tree
   *                                index).
   * \param [in] prioritize_local   flag prioritizing the choice of a local quadrant when several candidates of the same size are valid but
   *                                some of them are ghost quadrants.
   *                                NOTE: when set to true, this function's outcome may change for a given p4est as the number of
   *                                      processes used at runtime is changed!
   */
  void find_quadrant_containing_point(const int* tr_xyz_orig, double* xyz_point, int& current_rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches, const bool &prioritize_local) const;

  void find_quad_owning_pixel_quad(const p4est_quadrant_t& pixel_quad, const p4est_topidx_t& tree_idx, p4est_locidx_t& pos, bool& found_locally) const;

  void determine_local_and_layer_cells();

public:
  /*!
   * \brief my_p4est_hierarchy_t constructor for my_p4est_hierarchy objects
   * \param [in] p4est_ the standard p4est_t structure representing the local domain partition
   * \param [in] ghost_ the standard p4est_ghost_t structure of of ghost quadrants known by the
   *                    local partition  (may be NULL if no ghost quadrant is known)
   * \param [in] myb_   macromesh description (cartesian array of elementary root cells)
   */
  my_p4est_hierarchy_t(p4est_t *p4est_, p4est_ghost_t *ghost_, my_p4est_brick_t *myb_)
    : p4est(p4est_), ghost(ghost_), myb(myb_), periodic{DIM(is_periodic(p4est_, dir::x), is_periodic(p4est_, dir::y), is_periodic(p4est_, dir::z))}
  {
    determine_local_and_layer_cells();
  }

  inline const bool *get_periodicity() const { return periodic; }

  /*!
   * \brief find_smallest_quadrant_containing_point find the smallest (leaf) quadrant containing a point of interest
   * given by its cartesian coordinates. If the point lies to close
   * \param [in]    xyz             Cartesian coordinates of the point of interest (array of P4EST_DIM doubles)
   *                                xyz needs to be in the domain: the point is wrapped within the domain if domain
   *                                periodicity allows that.
   * \param [out] best_match        p4est_quadrant_t containing the point of interest on output. This argument must exist
   *                                beforehand, its member variables are wiped out at initiation stage of the function.
   *                                If found, the appropriate p4est_quadrant in p4est or ghost, is copied into best_match
   *                                and, except if the last flag argument is set to true, the p.piggy3 structure of best_match
   *                                is also filled so that
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
   * \param [in] prioritize_local   (optional) flag prioritizing the choice of a local quadrant when several candidates of the
   *                                same size are valid but some of them are ghost quadrants. This can happen when the given
   *                                point lies on a face shared between a local quadrant and a ghost quadrant of the same size
   *                                for instance. This feature was added in order to
   *                                    1)  enable on-the-fly calculations in interpolation of node-sampled fields up to the
   *                                        border of the computational domain (so long as the ghost cells are not smaller).
   *                                    2)  ensure that 'enough' neighborhood cells can be found when building local least-square
   *                                        interpolants for on-the-fly interpolation of cell- and face-sampled fields
   *                                NOTE: when set to true, this function's outcome may change for a given p4est as the number of
   *                                      processes used at runtime is changed!
   *                                default value is false
   * \param [in] set_cumulative_local_index_in_piggy3_of_best_match : (optional) flag setting the piggy3.local_num member of
   *                                best_match (if found) to be its local cumulative index cumulative (over the trees, then
   *                                the ghost quadrants, thereafter). This means that, if set to true, the p.piggy3 structure of
   *                                best_match (when found) is filled so that
   *                                best_match.p.piggy3.which_tree is the index of the tree owning best_match
   *                                best_match.p.piggy3.local_num is either the cumulative local index of the quadrant in the local
   *                                process (cumulative over local trees then ghost quadrants thereafter).
   *                                Default value is false (i.e. not doing it).
   * [NOTE 1]: this function assumes that all building bricks in the macromesh description have the same size (i.e. no
   *           contraction/stretching, every building brick is identical).
   * [THROWS]: in DEBUG, this function throws an std::runtime_error exception if the procedure runs into an internal
   *           inconsistency (if it found a remote, space-filling HierarchyCell that is not marked NOT_A_P4EST_QUADRANT)
   * \return the returned values is either
   *  - the rank of the process owning the quadrant in which the point of interest can be found, if the quadrant was found
   *    locally. (Therefore either mpirank if locally owned, or the rank of one of the neighbor process owning a ghost quadrant)
   *  - a value of -1 if no local quadrant could be found (the user is advised to check the content of remote_matches in such a
   *    case)
   */
  int find_smallest_quadrant_containing_point(const double *xyz, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches,
                                              const bool &prioritize_local = false, const bool &set_cumulative_local_index_in_piggy3_of_best_match = false) const;

  /*!
   * \brief get_all_quadrants_in establishes the list of local indices of the quadrants being found in another quadrant
   * \param [in]  quad                    pointer to a constant quadrant whose subresolving quadrants are sought
   * \param [in]  tree_idx                index of the tree owning the above quadrant
   * \param [out] list_of_local_quad_idx  vector of local indices of the quadrants of this object's p4est that can be
   *                                      found in the given quadrant. This vector is empty (size 0) on return if the given
   *                                      quadrant is finer than the leaf HiearchyCell found to contain it.
   */
  void get_all_quadrants_in(const p4est_quadrant_t* quad, const p4est_topidx_t& tree_idx, std::vector<p4est_locidx_t>& list_of_local_quad_idx) const;

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
   * \brief find_neighbor_cell_of_node finds the neighboring quadrant of a node in the given (i, j, k) direction. The direction
   * must be "diagonal" for the function to work (if the "nodes" are the nodes of corresponding p4est, i.e. standard usage)!
   * (e.g. (-1,1,1) ... no cartesian direction!).
   * \param [in] node_idx         local index of the node whose neighboring cell is looked for
   * \param [in] nodes            the p4est_node structure in which the latter node is stored
   * \param [in] i                the x search direction, -1 or 1
   * \param [in] j                the y search direction, -1 or 1
   * \param [in] k                the z search direction, -1 or 1, only in 3D
   * \param [out] quad_idx        the index of the found quadrant, in cumulative numbering over the trees. To fetch this quadrant
   *                              from its corresponding tree you need to substract the tree quadrant offset.
   *                              If no quadrant was found, this is set to NOT_A_P4EST_QUADRANT (not known from the local, possibly
   *                              ghosted domain partition) or NOT_A_VALID_QUADRANT (past the edge of a nonperiodic domain)
   * \param [out] owning_tree_idx the index of the tree in which the quadrant was found (valid and sensible if the quadrant was
   *                              actually found, of course)
   */
  void find_neighbor_cell_of_node(const p4est_locidx_t& node_idx, const p4est_nodes_t* nodes, DIM(const char& i, const char& j, const char& k),
                                  p4est_locidx_t& quad_idx, p4est_topidx_t& owning_tree_idx) const;

  size_t memory_estimate() const
  {
    size_t memory = 0;
    memory += (local_inner_quadrant.size())*sizeof (local_and_tree_indices);
    memory += (local_layer_quadrant.size())*sizeof (local_and_tree_indices);

    return memory;
  }
};

#endif /* !MY_P4EST_HIERARCHY_H */
