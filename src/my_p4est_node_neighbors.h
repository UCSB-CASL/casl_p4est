#ifndef MY_P4EST_NODE_NEIGHBORS_H
#define MY_P4EST_NODE_NEIGHBORS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_quad_neighbor_nodes_of_node.h>
#include <src/my_p8est_hierarchy.h>
#include <p8est_bits.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_quad_neighbor_nodes_of_node.h>
#include <src/my_p4est_hierarchy.h>
#include <p4est_bits.h>
#endif

#include <vector>
#include <sstream>

class my_p4est_node_neighbors_t {
  friend class PoissonSolverNodeBase;
  friend class PoissonSolverCellBase;
  friend class PoissonSolverFaces;
  friend class PoissonSolverNodeBaseJump;
  friend class InterpolatingFunctionNodeBase;
  friend class InterpolatingFunctionNodeBaseBalanced;
  friend class InterpolatingFunctionNodeBaseHost;
  friend class InterpolatingFunctionNodeBaseNonblocking;
  friend class my_p4est_interpolation_t;
  friend class my_p4est_interpolation_nodes_t;
  friend class my_p4est_interpolation_cells_t;
  friend class my_p4est_interpolation_faces_t;
  friend class my_p4est_level_set;
  friend class SemiLagrangian;
  friend class my_p4est_bialloy_t;

  /**
     * Initialize the QuadNeighborNodeOfNode information
     */

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *myb;
  std::vector< quad_neighbor_nodes_of_node_t > neighbors;
  std::vector<bool> is_qnnn_valid;
  std::vector<p4est_locidx_t> layer_nodes;
  std::vector<p4est_locidx_t> local_nodes;  
  bool is_initialized;

  bool construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const;

public:
  my_p4est_node_neighbors_t( my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), nodes(nodes_), myb(hierarchy_->myb)
  {
    is_initialized = false;

    /* compute the layer and local nodes.
     * layer_nodes: This is a list of indices for nodes in the local range on this
     * processor (i.e. 0<= i < nodes->num_owned_indeps) that are taged as ghost
     * on at least another processor
     * local_nodes: This is a list of indices for nodes in the local range on this
     * processor that are not included in the layer_nodes
     *
     * With this subdivision, ANY computation on the local nodes should be decomposed
     * into four stages:
     * 1) do computation on the layer nodes
     * 2) call VecGhostUpdateBegin so that each processor begins sending messages
     * 3) do computation on the local nodes
     * 4) call VecGhostUpdateEnd to finish the update process
     *
     * This will effectively hide the communication steps 2,4 with the computation
     * step 3
     */

    layer_nodes.reserve(nodes->num_owned_shared);
    local_nodes.reserve(nodes->num_owned_indeps - nodes->num_owned_shared);

    for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i){
      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
      ni->pad8 == 0 ? local_nodes.push_back(i) : layer_nodes.push_back(i);
    }
  }

  inline size_t get_layer_size() const { return layer_nodes.size(); }
  inline size_t get_local_size() const { return local_nodes.size(); }
  inline p4est_locidx_t get_layer_node(size_t i) const {
#ifdef CASL_THROWS
    return layer_nodes.at(i);
#endif
    return layer_nodes[i];
  }
  inline p4est_locidx_t get_local_node(size_t i) const {
#ifdef CASL_THROWS
    return local_nodes.at(i);
#endif
    return local_nodes[i];
  }

  /**
   * @brief initialize the buffers containing the information about the neighboring nodes for
   * every local and ghost nodes provided when instantiating the my_p4est_node_neighbors_t structure.
   * This consumes a lot of memory, and it can improve the time performances of the code if repetitive
   * access to the neighbors information is required.
   */
  void init_neighbors();
  void clear_neighbors();
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_);
  
  inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const {
#ifdef CASL_THROWS
    if (!is_initialized)
      throw std::runtime_error("[ERROR]: operator[] can only be used if nodes are buffered. Either initialize the buffer by calling"
                               "'init_neighbors()' or consider calling 'get_neighbors()' to compute the neighbors on the fly.");

    if (is_qnnn_valid[n])
      return neighbors.at(n);
    else {
      std::ostringstream oss;
      oss << "[ERROR]: The neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank << " is invalid.";
      throw std::invalid_argument(oss.str().c_str());
    }
#else
    return neighbors[n];
#endif
  }

  inline quad_neighbor_nodes_of_node_t get_neighbors(p4est_locidx_t n) const {
    if (is_initialized) {
#ifdef CASL_THROWS
      if (is_qnnn_valid[n])
        return neighbors.at(n);
      else {
        std::ostringstream oss;
        oss << "[ERROR]: The neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank << " is invalid.";
        throw std::invalid_argument(oss.str().c_str());
      }
#else
      return neighbors[n];
#endif
    } else {
      quad_neighbor_nodes_of_node_t qnnn;
      get_neighbors(n, qnnn);
      return qnnn;
    }
  }

  inline void get_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const {
#ifdef CASL_THROWS
    bool err = construct_neighbors(n, qnnn);
    if (err){
      std::ostringstream oss;
      oss << "[ERROR]: Could not construct neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank;
      throw std::invalid_argument(oss.str().c_str());
    }
#else
    construct_neighbors(n, qnnn);
#endif
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
#ifdef P4_TO_P8
  void find_neighbor_cell_of_node( p4est_locidx_t n, char i, char j, char k, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;
#else
  void find_neighbor_cell_of_node( p4est_locidx_t n, char i, char j, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;
#endif

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

#ifdef P4_TO_P8
  /*!
   * \brief dzz_central compute dzz_central on all nodes and update the ghosts
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fzz PETSc vector to store the results in. A check is done to ensure they have the same size
   */
  void dzz_central(const Vec f, Vec fzz) const;
#endif

  /*!
   * \brief second_derivatives_central computes both dxx_central and dyy_central at all
   * points. Theoretically this should have a better chance at hiding communications
   * than above calls combined.
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fdd PETSc _BLOCK_ vector to store dxx adn dyy results in.
   * A check is done to ensure it has the same size as f and block size = P4EST_DIM
   */
  void second_derivatives_central(const Vec f, Vec fdd) const;

  /*!
   * \brief second_derivatives_central computes dxx, dyy, and dzz central at all
   * points. Similar to the function above except it use two regular vector in
   * place of a single blocked vector. Easier to use but more expensive in terms
   * of MPI. Also note that fxx, fyy, and fzz cannot be obtained via VecDuplicate as
   * this would share the same VecScatter object and avoid simaltanous update.
   *
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fzz PETSc vector to store the results in. A check is done to ensure they have the same size as f (only inn 3D)
   */
#ifdef P4_TO_P8
  void second_derivatives_central(const Vec f, Vec fxx, Vec fyy, Vec fzz) const;
#else
  void second_derivatives_central(const Vec f, Vec fxx, Vec fyy) const;
#endif

private:
#ifdef P4_TO_P8
  void second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy, Vec fzz) const;
#else
  void second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy) const;
#endif
};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
