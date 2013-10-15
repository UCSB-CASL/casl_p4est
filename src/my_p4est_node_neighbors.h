#ifndef MY_P4EST_NODE_NEIGHBORS_H
#define MY_P4EST_NODE_NEIGHBORS_H

#include <p4est.h>
#include <p4est_ghost.h>

#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <p4est_bits.h>
#include <src/my_p4est_quad_neighbor_nodes_of_node.h>
#include <src/my_p4est_hierarchy.h>

#include <vector>
#include <sstream>

class my_p4est_node_neighbors_t {
  friend class PoissonSolverNodeBase;
  friend class InterpolatingFunction;
  friend class my_p4est_level_set;

  /**
     * Initialize the QuadNeighborNodeOfNode information
     */
  void init_neighbors();

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *myb;
  std::vector< quad_neighbor_nodes_of_node_t > neighbors;

public:
  my_p4est_node_neighbors_t( my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), nodes(nodes_), myb(hierarchy_->myb),
      neighbors(nodes_->num_owned_indeps)
  {
    init_neighbors();
  }

  inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const {
#ifdef CASL_THROWS
    if (n<0 || n>=nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[ERROR]: Trying to access neighboring nodes of element " << n
          << " in the QNNN structure which is out of bound [0, " << nodes->num_owned_indeps
          << "). This probably means you are trying to acess neighboring nodes"
             " of a ghost nod. This is not supported." << std::endl;
      throw std::invalid_argument(oss.str());
    }
#endif
    return neighbors[n];
  }

  /**
     * This function is finds the neighboring cell of a node in the given (i,j) direction. The direction must be diagonal
     * for the function to work ! (e.g. (-1,1) ... no cartesian direction!).
     * \param [in] node          a pointer to the node whose neighboring cells are looked for
     * \param [in] i             the x search direction, -1 or 1
     * \param [in] j             the y search direction, -1 or 1
     * \param [out] quad         the index of the found quadrant, in mpirank numbering. To fetch this quadrant from its corresponding tree
     *                           you need to substract the tree quadrant offset. If no quadrant was found, this is set to -1 (e.g. edge of domain)
     * \param [out] nb_tree_idx  the index of the tree in which the quadrant was found
     *
     */
  void find_neighbor_cell_of_node( p4est_indep_t *node, char i, char j, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;

  /*!
   * \brief dxx_central compute dxx_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size
   */
  void dxx_central(const Vec f, Vec fxx) const;

  /*!
   * \brief dyy_central compute dyy_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size
   */
  void dyy_central(const Vec f, Vec fyy) const;

  /*!
   * \brief dxx_and_dyy_central computes both dxx_central and dyy_central at all
   * points. Theoretically this should have a better chance at hiding communications
   * than above calls combined.
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fdd PETSc _BLOCK_ vector to store dxx adn dyy results in.
   * A check is done to ensure it has the same size as f and block size = 2
   */
  void dxx_and_dyy_central(const Vec f, Vec fdd) const;

  /*!
   * \brief dxx_and_dyy_central computes both dxx_central and dyy_central at all
   * points. Similar to the function above except it use two regular vector in
   * place of a single blocked vector. Easier to use but more expensive in terms
   * of MPI. Also note that fxx and fyy cannot be obtained via VecDuplicate as
   * this would share the same VecScatter object and avoid simaltanous update.
   *
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size as f
   */
  void dxx_and_dyy_central(const Vec f, Vec fxx, Vec fyy) const;

};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
