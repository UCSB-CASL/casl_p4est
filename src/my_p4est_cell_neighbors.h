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
  friend class my_p4est_xgfm_cells_t;

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  double tree_dimensions[P4EST_DIM];

  // recursive routine used to find the appropriate children of a cell sharing
  /*!
   * \brief find_neighbor_cells_of_cell_recursive: core recursive routine used to find the appropriate child(ren) of
   * the (hypothetical) quadrant Q' in the hierarchy that satisfy the queried neighborhood condition (see illustration
   * in comment here below for interpretation of Q')
   * \param [inout] ngbd: the set of neighbor cells (not cleared but augmented with all candidates if not present in the list yet);
   * \param [in] tr:      the tree index of Q'
   * \param [in] ind:     the the index of the considered HierarchyCell in trees[tr] corresponding to the cell in the Hierarchy being analyzed
   * \param [in] dir_xyz: the original search direction, as provided to the public routine here below (note that there is a mirror play of
   *                      search direction for fetching the appropriate child(ren), see the source code in the .cpp file for more details)
   */
  void find_neighbor_cells_of_cell_recursive(set_of_neighboring_quadrants& ngbd, const p4est_topidx_t& tr, const int& ind, const char dir_xyz[P4EST_DIM]) const;

public:
  my_p4est_cell_neighbors_t(my_p4est_hierarchy_t *hierarchy_)
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), myb(hierarchy_->myb)
  {
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      tree_dimensions[dim] = (hierarchy_->myb->xyz_max[dim] - hierarchy_->myb->xyz_min[dim])/hierarchy_->myb->nxyztrees[dim];
  }

  /*!
   * \brief find_neighbor_cells_of_cell finds (all) the neighbor cell(s) of a cell in the direction (dir_x, dir_y [, dir_z]), any
   * combination of directions is accepted.
   * \param [inout] ngbd:  the set of neighbor cells (not cleared but augmented with all candidates if not present in the list yet);
   * \param [in] quad_idx: the (local) index of the quadrant whose neighbor(s) is (are) searched;
   * \param [in] tree_idx: the tree index of that quadrant;
   * \param [in] dir_xyz:  array of P4EST_DIM Cartesian search directions (all components must be either -1, 0 or +1, no other value accepted)
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
  void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const char dir_xyz[P4EST_DIM]) const;

  /*!
   * \brief find_neighbor_cells_of_cell: wrapper to the general public function here above with individual
   * Cartesian search components as inputs
   */
  inline void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, DIM(char dir_x, char dir_y, char dir_z)) const
  {
    char tmp[P4EST_DIM] = {DIM(dir_x, dir_y, dir_z)};
    find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, tmp);
    return;
  }

  /*!
   * \brief find_neighbor_cells_of_cell: wrapper to the general public function here above in case of
   * a search for neighbors across a face
   */
  inline void find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& q, const p4est_topidx_t& tr, const unsigned char& face_dir) const
  {
    char search[P4EST_DIM] = {DIM(0, 0, 0)}; search[face_dir/2] = (face_dir%2 == 1 ? 1 : -1);
    find_neighbor_cells_of_cell(ngbd, q, tr, DIM(search[0], search[1], search[2]));
    return;
  }

  const p4est_t* get_p4est() const { return p4est; }
  const my_p4est_brick_t* get_brick() const { return myb; }
  const double* get_tree_dimensions() const { return tree_dimensions; }

};

double interpolate_cell_field_at_node(const p4est_locidx_t& node_idx, const my_p4est_cell_neighbors_t* c_ngbd, const my_p4est_node_neighbors_t* n_ngbd, const Vec cell_field, const BoundaryConditionsDIM* bc = NULL, const Vec phi = NULL);

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
