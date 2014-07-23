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

#define NOT_A_VALID_QNNN -1

class ghost_qnnn_exception:public std::exception {
  std::string msg;
public:
  ghost_qnnn_exception(const std::string& msg)
    : msg(msg)
  {}
#ifdef P4_TO_P8
  ghost_qnnn_exception(p4est_locidx_t n, int i, int j, int k)
#else
  ghost_qnnn_exception(p4est_locidx_t n, int i, int j)
#endif
  {
    std::ostringstream oss;
    oss << "[Error]: Could not find a suitable neighboring quadrant in the (" << i << ", " << j <<
       #ifdef P4_TO_P8
           ", " << k <<
       #endif
           ") direction for the ghost node with locidx = " << n << ". If the neighborhood information on"
           " this node is required, make sure that the ghost layer is expanded far enough by calling the"
           " 'p4est_ghost_expand' method." << std::endl;

    msg = oss.str();
  }

  const char* what() const throw() { return msg.c_str(); }
  ~ghost_qnnn_exception() throw() {}
};

class local_qnnn_exception:public std::exception {
  std::string msg;
public:
  local_qnnn_exception(const std::string& msg)
    : msg(msg)
  {}
#ifdef P4_TO_P8
  local_qnnn_exception(p4est_locidx_t n, int i, int j, int k)
#else
  local_qnnn_exception(p4est_locidx_t n, int i, int j)
#endif
  {
    std::ostringstream oss;
    oss << "[Error]: Could not find a suitable neighboring quadrant in the (" << i << ", " << j <<
       #ifdef P4_TO_P8
           ", " << k <<
       #endif
           ") direction for the local node with locidx = " << n << ". This is most probably a bug in the"
           " implementation of 'my_p4est_hierarchy_t' class." << std::endl;
    msg = oss.str();
  }

  const char* what() const throw() { return msg.c_str(); }
  ~local_qnnn_exception() throw() {}
};

class my_p4est_node_neighbors_t {
  friend class PoissonSolverNodeBase;
  friend class PoissonSolverCellBase;
  friend class InterpolatingFunctionNodeBase;
  friend class my_p4est_level_set;
  friend class SemiLagrangian;

  /**
     * Initialize the QuadNeighborNodeOfNode information
     */

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *myb;
  std::vector< quad_neighbor_nodes_of_node_t > neighbors;
  std::vector<std::string> ghost_qnnn_exception_log;
  std::vector<p4est_locidx_t> ghost_qnnn_idx;
  std::vector<p4est_locidx_t> layer_nodes;
  std::vector<p4est_locidx_t> local_nodes;  
  bool is_initialized;

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
    if (i > layer_nodes.size())
      throw std::invalid_argument("[ERROR]: accessing beyod layer node size");
#endif
    return layer_nodes[i];
  }
  inline p4est_locidx_t get_local_node(size_t i) const {
#ifdef CASL_THROWS
    if (i > local_nodes.size())
      throw std::invalid_argument("[ERROR]: accessing beyod local node size");
#endif
    return local_nodes[i];
  }

  void init_neighbors();
  void clear_neighbors();
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_);
  
  inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const {
#ifdef CASL_THROWS
    if (!is_initialized)
      throw std::runtime_error("[ERROR]: operator[] can only be used if nodes are buffered. Either initialize the buffer by calling"
                               "'init_neighbors()' or consider calling 'get_neighbors()' to compute the neighbors on the fly.");

    if (n < nodes->num_owned_indeps)
      return neighbors.at(n);
    else {
      p4est_locidx_t g = n - nodes->num_owned_indeps;
      p4est_locidx_t qnnn_idx = ghost_qnnn_idx[g];
      if (qnnn_idx == NOT_A_VALID_QNNN)
        throw ghost_qnnn_exception(ghost_qnnn_exception_log[g]);
      else
        return neighbors.at(qnnn_idx);
    }
#else
    if (n < nodes->num_owned_indeps)
      return neighbors[n];
    else
      return neighbors[ghost_qnnn_idx[n - nodes->num_owned_indeps]];
#endif
  }

  void get_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const;

  inline quad_neighbor_nodes_of_node_t get_neighbors(p4est_locidx_t n) const {
    if (is_initialized) {
#ifdef CASL_THROWS
      if (n < nodes->num_owned_indeps)
        return neighbors.at(n);
      else {
        p4est_locidx_t g = n - nodes->num_owned_indeps;
        p4est_locidx_t qnnn_idx = ghost_qnnn_idx[g];
        if (qnnn_idx == NOT_A_VALID_QNNN)
          throw ghost_qnnn_exception(ghost_qnnn_exception_log[g]);
        else
          return neighbors.at(qnnn_idx);
      }
#else
      if (n < nodes->num_owned_indeps)
        return neighbors[n];
      else
        return neighbors[ghost_qnnn_idx[n - nodes->num_owned_indeps]];
#endif
    } else {
      quad_neighbor_nodes_of_node_t qnnn;
      get_neighbors(n, qnnn);
      return qnnn;
    }
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
