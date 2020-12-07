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

/*!
 * \brief The my_p4est_cell_neighbors_t class acts as an interface tool that enables the user to
 * find the cell neighbors of a cell. This class makes an extensive use of the hierarchy object
 * and its methods.
 * [Comments by Raphael Egan, February 2020]
 */

class my_p4est_node_neighbors_t;

struct comparator_of_neighbor_cell
{
  inline bool operator()(const p4est_quadrant_t& lhs, const p4est_quadrant_t& rhs) const
  {
    return lhs.p.piggy3.local_num < rhs.p.piggy3.local_num;
  }
};

typedef std::set<p4est_quadrant_t, comparator_of_neighbor_cell> set_of_neighboring_quadrants;

class my_p4est_cell_neighbors_t {
private:
  friend class my_p4est_poisson_cells_t;

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  const double tree_dimensions[P4EST_DIM];

  // recursive routine used to find the appropriate children of a cell sharing
  /*!
   * \brief find_neighbor_cells_of_cell_recursive: core recursive routine used to find the appropriate child(ren) of
   * the (hypothetical) quadrant Q' in the hierarchy that satisfy the queried neighborhood condition (see illustration
   * in comment here below for interpretation of Q')
   * \param [inout] ngbd:   the set of neighbor cells (not cleared but augmented with all candidates if not present in the list yet);
   * \param [in] tr:        the tree index of Q'
   * \param [in] ind:       the the index of the considered HierarchyCell in trees[tr] corresponding to the cell in the Hierarchy being analyzed
   * \param [in] dir_xyz:   the original search direction, as provided to the public routine here below (note that there is a mirror play of
   *                        search direction for fetching the appropriate child(ren), see the source code in the .cpp file for more details)
   * \param [in] smallest_quad_size: pointer to a p4est_qcoord_t value representing the logical size of the smallest quadrant found in a cell
   *                        neighborhood, the pointed value is updated every time a smaller quadrant is found. This operation is optional
   *                        and simply disregarded if the pointer is NULL.
   */
  void find_neighbor_cells_of_cell_recursive(set_of_neighboring_quadrants& ngbd, const p4est_topidx_t& tr, const int& ind, const char dir_xyz[P4EST_DIM], p4est_qcoord_t *smallest_quad_size) const;

public:
  my_p4est_cell_neighbors_t(my_p4est_hierarchy_t *hierarchy_)
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), myb(hierarchy_->myb),
      tree_dimensions{DIM((hierarchy_->myb->xyz_max[0] - hierarchy_->myb->xyz_min[0])/hierarchy_->myb->nxyztrees[0],
      (hierarchy_->myb->xyz_max[1] - hierarchy_->myb->xyz_min[1])/hierarchy_->myb->nxyztrees[1],
      (hierarchy_->myb->xyz_max[2] - hierarchy_->myb->xyz_min[2])/hierarchy_->myb->nxyztrees[2])}
  {

  }

  /*!
   * \brief find_neighbor_cells_of_cell finds (all) the neighbor cell(s) of a cell in the direction (dir_x, dir_y [, dir_z]), any
   * combination of directions is accepted.
   * \param [inout] ngbd:   the set of neighbor cells (not cleared on inuput but augmented with all candidates if not present in the list yet);
   * \param [in] quad_idx:  the (local) index of the quadrant whose neighbor(s) is (are) searched;
   * \param [in] tree_idx:  the tree index of that quadrant;
   * \param [in] dir_xyz:   array of P4EST_DIM Cartesian search directions (all components must be either -1, 0 or +1, no other value accepted)
   * \param [in] smallest_quad_size: (optional) pointer to a p4est_qcoord_t value representing the logical size of the smallest quadrant found
   *                        in a cell neighborhood, the pointed value is updated every time a smaller quadrant is found. This operation is optional
   *                        and simply disregarded if the pointer is NULL.
   *                        --> relevant for evaluating scaling distance in some least-square interpolation procedure.
   * NOTES AND REMARKS:
   * - ngbd is not cleared on input --> the set can only be larger on output
   * - the p.piggy3 members of the elements of ngbd are filled and valid on outputs (and so must it be on input as well if ngbd is not empty!!!)
   * GENERAL PROCEDURE:
   * Let Q be the quadrant whose neighbors are searched for. The method first creates a (hypothetical) quadrant Q' of the same size as Q
   * but located relatively to Q as if it was the requested neighbor. Based on the information from Q' the hierarchy is enquired:
   * - if Q' is entirely included in a local leaf (Q' is smaller than or of the same size as the actual local quadrant), the actual neighbor
   * cell that encloses Q' is added to ngbd (if not in there yet);
   * - if Q' is not a leaf, the routine enquires the hierarchy recursively to find the children of Q' that are in the required neighboring
   * relation with Q
   *
   * Examples in 2D:
   * Let the local grid be
   * ________________________________________________________________________________________________
   * |       |       |       |       |                                                               |
   * |       |       |       |       |                                                               |
   * |_______|_______|_______|_______|                                                               |
   * |       |       |       |       |                                                               |
   * |       |       |       |       |                                                               |
   * |_______|_______|_______|_______|                                                               |
   * |       |       |       |       |                                                               |
   * |       |       |       |       |                                                               |
   * |_______|_______|_______|_______|                                                               |
   * |       |   |   |       |   |   |                                                               |
   * |       |---|---|   D   |---|---|                              G                                |
   * |_______|___|_C_|_______|_E_|_F_|                                                               |
   * |       |       |               |                                                               |
   * |       |   B   |               |                                                               |
   * |_______|_______|       Q       |                                                               |
   * |       |       |               |                                                               |
   * |       |   A   |               |                                                               |
   * |_______|_______|_______________|                                                               |
   * |       |       |               |                                                               |
   * |       |   I   |               |                                                               |
   * |_______|_______|       H       |                                                               |
   * |       |       |               |                                                               |
   * |       |       |               |                                                               |
   * |_______|_______|_______________|_______________________________________________________________|
   * |       |       |       |       |                               |                               |
   * |       |       |       |       |                               |                               |
   * |_______|_______|_______|_______|                               |                               |
   * |       |       |       |       |                               |                               |
   * |       |       |       |       |                               |                               |
   * |_______|_______|_______|_______|                               |                               |
   * |               |       |       |                               |                               |
   * |               |       |       |                               |                               |
   * |               |_______|_______|                               |                               |
   * |               |       |       |                               |                               |
   * |               |       |       |                               |                               |
   * |_______________|_______|_______|_______________________________|_______________________________|
   *
   * The resulting behavior of find_neighbor_cells_of_cell() when called for cell Q (for every example, we assume that ngbd is empty on input)
   * ~ with dir_xyz = [-1,  0] --> ngbd = {A, B}
   * ~ with dir_xyz = [-1, +1] --> ngbd = {C}
   * ~ with dir_xyz = [ 0, +1] --> ngbd = {D, E, F}
   * ~ with dir_xyz = [+1, +1] --> ngbd = {G}
   * ~ with dir_xyz = [+1,  0] --> ngbd = {G}
   * ~ with dir_xyz = [+1, -1] --> ngbd = {G}
   * ~ with dir_xyz = [0,  -1] --> ngbd = {H}
   * ~ with dir_xyz = [-1, -1] --> ngbd = {I}
   *
   */
  void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const char dir_xyz[P4EST_DIM], p4est_qcoord_t *smallest_quad_size = NULL) const;

  /*!
   * \brief find_neighbor_cells_of_cell: wrapper to the general public function here above with individual
   * Cartesian search components as inputs
   */
  inline void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, DIM(char dir_x, char dir_y, char dir_z), p4est_qcoord_t *smallest_quad_size = NULL) const
  {
    char tmp[P4EST_DIM] = {DIM(dir_x, dir_y, dir_z)};
    find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, tmp, smallest_quad_size);
    return;
  }

  /*!
   * \brief find_neighbor_cells_of_cell: wrapper to the general public function here above in case of
   * a search for neighbors across a face
   * (not augmented with "p4est_qcoord_t* smallest_quad_size" feature input argument, since that would
   * create a series of ambiguous calls because '0' represents either a 0 integer value or the NULL pointer
   * in c++)
   */
  inline void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& q, const p4est_topidx_t& tr, const u_char& face_dir) const
  {
    char search[P4EST_DIM] = {DIM(0, 0, 0)}; search[face_dir/2] = (face_dir%2 == 1 ? 1 : -1);
    find_neighbor_cells_of_cell(ngbd, q, tr, search);
    return;
  }

  const p4est_t* get_p4est() const { return p4est; }
  const p4est_ghost_t* get_ghost() const { return ghost; }
  const my_p4est_brick_t* get_brick() const { return myb; }
  const double* get_tree_dimensions() const { return tree_dimensions; }
  const my_p4est_hierarchy_t* get_hierarchy() const { return hierarchy; }

  /*!
   * \brief gather_neighbor_cells_of_cell finds all neighbor cells of a cell in all cartesian directions (and any of their
   * combination) and adds them to a set_of_neighboring_quadrants. This routine looks for first degree neighbors by default
   * but it can be extended to second degree neighbors, if desired.
   * \param [in] quad_with_correct_local_num_in_piggy3 : quadrant whose neighbors are sought. This quadrant is also added to
   *                        the set.
   *                        [IMPORTANT REMARK 1:] the p.piggy3 value of the quadrant must be valid
   *                        [IMPORTANT REMARK 2:] the p.piggy3.local_num must be the CUMULATIVE local quadrant index over the
   *                                              local trees! (contrary to what is returned by find_smallest_quadrant() in
   *                                              my_p4est_hierarchy_t)!
   * \param [inout] ngbd : the set of neighbor cells (not cleared on input but augmented with all candidates if not present
   *                        in the list yet);
   * \param [in] add_second_degree_neighbors : (optional) boolean flag activating the search of second-degree neighbors if true
   *                        (default value is false)
   * \param [in] no_search : (optional) array of P4EST_DIM boolean flags de-activating search in Cartesian directions.
   *                        The neighbor in direction dir are not sought if no_search[dir] is true. All Cartesian directions
   *                        are probed by default.
   * \return the logical size of the smallest quadrant found in the nearby cell neighborhood (given quadrant and first-degree
   *                        neighbors ONLY!!!)
   *                        --> relevant for evaluating scaling distance in some least-square interpolation procedures.
   */
  p4est_qcoord_t gather_neighbor_cells_of_cell(const p4est_quadrant_t& quad_with_correct_local_num_in_piggy3, set_of_neighboring_quadrants& ngbd, const bool& add_second_degree_neighbors = false,
                                               const bool *no_search = NULL) const;

  /*!
   * \brief gather_neighbor_cells_of_node finds all neighbor cells of a node in all cartesian directions (and any of their
   * combination) and adds them to a set_of_neighboring_quadrants. This routine looks for first degree neighbors by default
   * but it can be extended to second degree neighbors, if desired.
   * \param [in] node_idx: local index of the node whose neighbor cells are sought
   * \param [in] nodes:    a pointer to a node structure in which the above node is stored
   * \param [inout] cell_neighbors: the set of neighbor cells (not cleared on input but augmented with all candidates if not
   *                           present in the list yet);
   * \param [in] add_second_degree_neighbors : (optional) boolean flag activating the search of second-degree neighbors if true
   *                           (default value is false)
   * \return the logical size of the smallest quadrant found in the first-degree cell neighborhood (first-degree only!!!)
   *                           --> relevant for evaluating scaling distance in some least-square interpolation procedures.
   */
  p4est_qcoord_t gather_neighbor_cells_of_node(const p4est_locidx_t& node_idx, const p4est_nodes_t* nodes, set_of_neighboring_quadrants& cell_neighbors, const bool& add_second_degree_neighbors = false) const;

};

/*!
 * \brief interpolate_cell_field_at_node interpolates a cell-sampled-field at a grid node using least-square interpolation
 * \param [in] node_idx   : local index of the node where the least-square interpolated value is desired;
 * \param [in] c_ngbd     : cell neighborhood information (built on the p4est grid)l
 * \param [in] n_ngbd     : node-neighborhood information (built on the nodes of the p4est grid)l
 * \param [in] cell_field : cell-sampled field to interpolatel
 * \param [in] bc         : (optional) boundary condition associated with the cell-sampled field to interpolatel
 * \param [in] phi        : (optional) node-sampled levelset function.
 * \return result of the least-square interpolation, at node of index nbode_idx, of valid neighboring cell-sampled values
 * of cell_field.
 * NOTES:
 * - if bc is disregarded (i.e. NULL) or if the interface-type is NOINTERFACE, all neighboring cell-sampled values are considered valid;
 * - if bc is NOT disregarded and if the interface-type is _not_ NOINTERFACE, then phi is REQUIRED and must be provided
 * (a local cell value is considered valid if the arithmetic average of the levelset sampled over the vertices of the cell is negative
 * OR if the corresponding cell is crossed by the levelset and the interface-type is NEUMANN.
 */
double interpolate_cell_field_at_node(const p4est_locidx_t& node_idx, const my_p4est_cell_neighbors_t* c_ngbd, const my_p4est_node_neighbors_t* n_ngbd, const Vec cell_field, const BoundaryConditionsDIM* bc = NULL, const Vec phi = NULL);

double get_lsqr_interpolation_at_node(const double xyz_node[P4EST_DIM], const my_p4est_cell_neighbors_t* ngbd_c, const set_of_neighboring_quadrants &ngbd_of_cells, const double &scaling,
                                      const double* cell_sampled_field_p, const BoundaryConditionsDIM* bc, const my_p4est_node_neighbors_t* ngbd_n, const double* node_sampled_phi_p,
                                      const u_char &degree = 2, const double &thresh_condition_number = 1.0e4, linear_combination_of_dof_t* interpolator = NULL);

inline double get_lsqr_interpolation_at_node(const double xyz_node[P4EST_DIM], const my_p4est_cell_neighbors_t* ngbd_c, const set_of_neighboring_quadrants &ngbd_of_cells, const double &scaling,
                                             const double* cell_sampled_field_p, const u_char &degree = 2, const double &thresh_condition_number = 1.0e4, linear_combination_of_dof_t* interpolator = NULL)
{
  return get_lsqr_interpolation_at_node(xyz_node, ngbd_c, ngbd_of_cells, scaling, cell_sampled_field_p, NULL, NULL, NULL, degree, thresh_condition_number, interpolator);
}

void get_lsqr_cell_gradient_operator_at_point(const double xyz_node[P4EST_DIM], const my_p4est_cell_neighbors_t* ngbd_c, const set_of_neighboring_quadrants &ngbd_of_cells, const double &scaling,
                                              linear_combination_of_dof_t grad_operator[], const bool& point_is_quad_center = false, const p4est_locidx_t& idx_of_quad_center = -1);

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
