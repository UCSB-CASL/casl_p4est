#ifndef MY_P4EST_NODE_NEIGHBORS_H
#define MY_P4EST_NODE_NEIGHBORS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_quad_neighbor_nodes_of_node.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_cell_neighbors.h>
#include <p8est_bits.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_quad_neighbor_nodes_of_node.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_cell_neighbors.h>
#include <p4est_bits.h>
#endif

#include <vector>
#include <sstream>

/*!
 * \brief The my_p4est_node_neighbors_t class provides the user with node neighborhood information,
 * but also with routines calculating first and second derivatives of node-sampled fields as well as
 * a distinction between layer and inner local nodes to overlap communications with the bulk local
 * computation.
 * [Note:] several commented functions herebelow are duplicates of the original functions but allowing
 * for the user to not only construct/store the node neighborhood information, but also to construct/
 * store some associated basic operators for calculating (interpolated) neighbor scalar value, first or
 * second derivatives.. The motivating idea was to establish those operators once and for all in order to
 * shortcut the geometry-related calculations for every subsequent calls to the relevant routines and 
 * make calculations more efficient. However, this slightly complicated the interface and the efficiency 
 * gain was marginal for calculations involving single-scalar fields. The corresponding functions have 
 * been commented out in this class and in my_p4est_quad_neighbors_nodes_of_node to keep the bulk of 
 * that work.
 * [End of note]
 * [Second note:] this class assumes that all building bricks in the macromesh description have the same
 * size (i.e. no contraction/stretching, every building brick is identical).
 * [End of second note]
 */
class my_p4est_node_neighbors_t {
  friend class my_p4est_bialloy_t;
  friend class my_p4est_biofilm_t;
  friend class my_p4est_electroporation_t;
  friend class my_p4est_epitaxy_t;
  friend class my_p4est_integration_mls_t;
  friend class my_p4est_interpolation_cells_t;
  friend class my_p4est_interpolation_faces_t;
  friend class my_p4est_interpolation_nodes_t;
  friend class my_p4est_interpolation_nodes_local_t;
  friend class my_p4est_interpolation_t;
  friend class my_p4est_level_set_cells_t;
  friend class my_p4est_level_set_faces_t;
  friend class my_p4est_level_set_t;
  friend class my_p4est_multialloy_t;
  friend class my_p4est_navier_stokes_t;
  friend class my_p4est_poisson_cells_t;
  friend class my_p4est_poisson_jump_nodes_extended_t;
  friend class my_p4est_poisson_jump_nodes_voronoi_t;
  friend class my_p4est_poisson_jump_voronoi_block_t;
  friend class my_p4est_poisson_nodes_mls_sc_t;
  friend class my_p4est_poisson_nodes_mls_t;
  friend class my_p4est_poisson_nodes_multialloy_t;
  friend class my_p4est_poisson_nodes_t;
  friend class my_p4est_scft_t;
  friend class my_p4est_semi_lagrangian_t;
  friend class my_p4est_two_phase_flows_t;
  friend class my_p4est_xgfm_cells_t;

  /* Self-explanatory member variables */
  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *myb;
  /*!
   * \brief neighbors: standard vector listing the node neighborhood for all node that
   * are locally known such that the neighborhood can be fully determined. This vector
   * is filled and fully determined only if the flag is_initialized is true
   */
  std::vector< quad_neighbor_nodes_of_node_t > neighbors;
#ifdef CASL_THROWS
  /*!
   * \brief is_qnnn_valid: standard vector of validity markers for the nodes that are locally known
   */
  std::vector<bool> is_qnnn_valid;
#endif
  /*!
   * \brief layer_nodes: standard vector of indices for the nodes in the local range on
   * this processor (i.e. indices i such that 0 <= i < nodes->num_owned_indeps) but that
   * are tagged as ghost on at least one other processor
   */
  std::vector<p4est_locidx_t> layer_nodes;
  /*!
   * \brief local_nodes: standard vector of indices for the nodes in the local range on
   * this processor that are not included in the layer_nodes
   */
  std::vector<p4est_locidx_t> local_nodes;  
  /*!
   * \brief is_initialized: flag that is set to true when 'neighbors' is fully set and determined
   */
  bool is_initialized;
  /*!
   * \brief periodic periodicity flag, the domain is periodic along the cartesian directon dir
   * if periodic[dir] is true.
   */
  bool periodic[P4EST_DIM];

  /*!
   * \brief construct_neighbors constructs the full node neighborhood information for a local node
   * \param [in]    n     local index of the node whose neighborhood needs to be constructed
   * \param [out]   qnnn  node neighborhood of the local node of index n on output if valid.
   *                      (The object must exist beforehand)
   * \return a boolean indicating the validity of the constructed node neighborhood qnnn.
   *         IMPORTANT: returns true if invalid, false if valid!
   * [Some details of implementation in brief:]
   * 1) The routine searches for all the possible neighbor quadrants of node n. If any of them is
   *    not known locally, i.e., if the node n lies on the border of the local (ghosted) partition,
   *    the node neighborhood cannot be determined and the routine returns 'true'
   * 2) If all these neighboring cells exist and are well-defined, the node neighborhood is determined
   *    based on the knowledge of these neighboring cells (node indices of neighbors and distances along
   *    cartesian directions)
   * 3) If a neighbor cell cannot be found in a specific direction because node n is a wall node, the
   *    routine tries to fetch the second-degree neighbor in the wall-normal direction. If this is not
   *    successfull, the node neighborhood cannot be determined and the routine returns 'true'. If that
   *    second-degree node neighbor can be found, the index (indices) of the non-existing node neighbor(s)
   *    (i.e., the one(s) that would be across the wall) are mapped to this (these) second-degree neighbor,
   *    with a negative distance.
   *    (--> This last procedure may fail if the smallest quadrant close to a local wall node is a ghost
   *    quadrant and if the ghost layer has only one layer of cells)
   * [end of details of implementation in brief]
   * [Note for developers:]
   * There is a lot of internal code duplication in this function, there must be a way to address that
   * and make it cleaner. Otherwise beware of modifying ALL similar code snippets if any of them needs to
   * be modified.
   * [end of note for developers]
   */
  bool construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const;
  /* bool construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                           const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false) const;*/

  /*!
   * \brief update_all_but_hierarchy: inner private function to be called by the two different
   * public 'update' functions. This one updates the member variables p4est, ghost, and
   * nodes. It also clears and reconstructs the list of layer and local nodes.
   * If the node neighbors were previously initialized, this routine clears and reconstructs
   * them, as well as their validity flags (in DEBUG).
   * \param [in] p4est_ the new p4est structure;
   * \param [in] ghost_ the new ghost later;
   * \param [in] nodes_ the new node structure.
   */
  void update_all_but_hierarchy(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_);
  /* void update_all_but_hierarchy(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                                            const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false); */

  /*!
   * \brief set_layer_and_local_nodes self-explanatory
   */
  inline void set_layer_and_local_nodes()
  {
    /* Compute the layer and local nodes.
     * With this subdivision, ANY computation on the local nodes that needs to be
     * synchronized should be decomposed into four stages:
     * 1) do computation on the layer nodes
     * 2) call VecGhostUpdateBegin so that each processor begins sending (non-blocking) messages
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
      // ni->pad8 is the number of remote process(es) that list ni as one of their ghost nodes
      // Therefore ni is purely local if ni->pad8 is not 0, otherwise it is a layer node
    }
  }

public:
  /*!
   * \brief my_p4est_node_neighbors_t constructor, this does NOT build and store the node neighbors by default,
   * it simply sets the member variables that are required to determine the neighborhood of any node. If you want
   * the node neighbors to be built and stored, you need to call 'init()' after construction!
   * \param [in] hierarchy_ grid hirerachy
   * \param [in] nodes_     grid nodes
   */
  my_p4est_node_neighbors_t( my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), nodes(nodes_), myb(hierarchy_->myb)
  {
    is_initialized = false;
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
      periodic[dd] = is_periodic(p4est, dd);
    set_layer_and_local_nodes();
  }

  /*!
   * \brief get_hierarchy
   * \return returns a const pointer to the hierarchy structure
   */
  inline const my_p4est_hierarchy_t* get_hierarchy() const { return hierarchy; }

  /*!
   * \brief get_p4est
   * \return returns a const pointer to p4est structure
   */
  inline const p4est_t* get_p4est() const { return p4est; }

  /*!
   * \brief get_ghost
   * \return returns a const pointer to ghost structure
   */
  inline const p4est_ghost_t* get_ghost() const { return ghost; }

  /*!
   * \brief get_nodes
   * \return returns a const pointer to nodes structure
   */
  inline const p4est_nodes_t* get_nodes() const { return nodes; }

  /*!
   * \brief get_brick
   * \return returns a const pointer to brick structure
   */
  inline const my_p4est_brick_t* get_brick() const { return myb; }

  /*!
   * \brief get_layer_size
   * \return the number of nodes that are layer nodes
   */
  inline size_t get_layer_size() const { return layer_nodes.size(); }
  /*!
   * \brief get_local_size
   * \return the number of node that are locally owned and that are not ghost for any other process
   */
  inline size_t get_local_size() const { return local_nodes.size(); }
  /*!
   * \brief get_layer_node: layer nodes accessor
   * \param [in] i index number of the layer node of interet, must be in [0, get_layer_size()[
   * \return the local node index, in the nodes->indep_nodes array, for the ith layer node
   */
  inline p4est_locidx_t get_layer_node(size_t i) const {
#ifdef CASL_THROWS
    return layer_nodes.at(i);
#endif
    return layer_nodes[i];
  }
  /*!
   * \brief get_local_node: local node accessor
   * \param [in] i index number of the local node of interet, must be in [0, get_local_size()[
   * \return the local node index, in the nodes->indep_nodes array, for the ith local node
   */
  inline p4est_locidx_t get_local_node(size_t i) const {
#ifdef CASL_THROWS
    return local_nodes.at(i);
#endif
    return local_nodes[i];
  }

  inline bool neighbors_are_initialized() const { return is_initialized; }

  /*!
   * \brief init_neighbors: initialize the buffers containing the information about the neighboring nodes
   * for every local and ghost nodes provided when instantiating the my_p4est_node_neighbors_t class. This
   * consumes a lot of memory, but it greatly improves the time performances of the code if repetitive access
   * to the neighbor information is required.
   * [commented version:]
   * Equivalent version allowing the user to store local operators besides the node neighborhood information
   * to make execution more efficient if repetitive calls to derivatives are required.
   * [end of commented version]
   */
  void init_neighbors();
  /* void init_neighbors(const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                      const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false); */
  /*!
   * \brief clear_neighbors wipes out the content of the 'neighbors' buffer, their validity flags 'is_qnnn_valid' in DEBUG
   * and resets the flag is_initialized to false.
   */
  void clear_neighbors();
  /*!
   * \brief update updates the member variables hierarchy, p4est, ghost, and nodes. It also clears and reconstructs
   * the list of layer and local nodes.
   * If the node neighbors were previously initialized, this routine clears and reconstructs them, as well as their
   * validity flags in DEBUG.
   * \param [in] hierarchy_ the new grid hierarchy;
   * \param [in] nodes_     the new node structure;
   */
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_);
  /*!
   * \brief update updates the member variables hierarchy, p4est, ghost, and nodes. IMPORTANT: this one forecfully
   * reconstructs the grid hierarchy: if you have already updated the hierarchy independently, use the above method
   * instead for better performance!
   * This method also clears and reconstructs the list of layer and local nodes.
   * If the node neighbors were previously initialized, this routine clears and reconstructs them, as well as their
   * validity flags in DEBUG.
   * \param [in] p4est_ the new p4est structure;
   * \param [in] ghost_ the new ghost later;
   * \param [in] nodes_ the new node structure.
   */
  void update(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_);
  /*
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
              const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false);
  void update(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
              const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false);
  */
  
  /*!
   * \brief operator [] accesses the node neighborhood for local node of index n.
   *                    The my_p4est_node_neighbors MUST be initialized before using this function!
   * \param [in] n      local index of the node whose neighborhood is queried
   * \return a reference to the queried node neighborhood.
   * throws exception in DEBUG if not initialized or if the node neighborhood is invalid
   */
  inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const {
#ifdef CASL_THROWS
    if (!is_initialized)
      throw std::runtime_error("[ERROR]: operator[] can only be used if nodes are buffered. Either initialize the buffer by calling"
                               "'init_neighbors()' or consider calling 'get_neighbors()' to compute the neighbors on the fly.");

    if (is_qnnn_valid[n])
      return neighbors[n];
    else {
      std::ostringstream oss;
      oss << "[ERROR]: The neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank << " is invalid.";
      throw std::invalid_argument(oss.str().c_str());
    }
#else
    return neighbors[n];
#endif
  }

  /*!
   * \brief get_neighbors will provide the user with the node neighborhood information for local node n.
   *        If the my_p4est_node_neighbors was initialized, the neighborhood is accessed from the buffer,
   *        otherwise it is constructed.
   * \param [in] n        local index of the node whose neighborhood is queried
   * \return the queried node neighborhood
   * throws exception in DEBUG if the node neighborhood is invalid
   */
  /*inline quad_neighbor_nodes_of_node_t get_neighbors(p4est_locidx_t n, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                                                     const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false)*/
  inline quad_neighbor_nodes_of_node_t get_neighbors(p4est_locidx_t n) const
  {
    if (is_initialized) {
#ifdef CASL_THROWS
      if (is_qnnn_valid[n])
        return neighbors[n];
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
      bool err = construct_neighbors(n, qnnn);

      /*bool err = construct_neighbors(n, qnnn, set_and_store_linear_interpolators, set_and_store_second_derivatives_operators,
                                     set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
      if (err){
        std::ostringstream oss;
        oss << "[ERROR]: Could not construct neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank;
        throw std::invalid_argument(oss.str().c_str());
      }
      return qnnn;
    }
  }

  /*!
   * \brief get_neighbors will provide the user with the node neighborhood information for local node n.
   *        If the my_p4est_node_neighbors was initialized, the neighborhood is accessed from the buffer,
   *        and returned to the user throuht the out argument. Otherwise, it is constructed.
   * \param [in] n        local index of the node whose neighborhood is queried
   * \param [out] qnnn    reference of a node neighborhood, filled with neighborhood for node n on output
   * [note:]
   * This function copies data into the out buffer, which may be slow.
   * [end of note]
   * throws exception in DEBUG if not initialized or if the node neighborhood is invalid
   */
  /*inline void get_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                            const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false) const */
  inline void get_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const
  {
    if (is_initialized) {
#ifdef CASL_THROWS
      if (is_qnnn_valid[n])
        qnnn = neighbors[n];
      else {
        std::ostringstream oss;
        oss << "[ERROR]: The neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank << " is invalid.";
        throw std::invalid_argument(oss.str().c_str());
      }
#else
      qnnn = neighbors[n];
#endif
    }
    else {
#ifdef CASL_THROWS
      bool err = construct_neighbors(n, qnnn/*, set_and_store_linear_interpolators, set_and_store_second_derivatives_operators,
                                     set_and_store_gradient_operator, set_and_store_quadratic_interpolators*/);
      if (err){
        std::ostringstream oss;
        oss << "[ERROR]: Could not construct neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank;
        throw std::invalid_argument(oss.str().c_str());
      }
#else
      construct_neighbors(n, qnnn);
      /* construct_neighbors(n, qnnn, set_and_store_linear_interpolators, set_and_store_second_derivatives_operators,
                          set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/

#endif
    }
  }

  /*!
   * \brief get_neighbors will provide the user with the node neighborhood information for local node n.
   *        The my_p4est_node_neighbors MUST be initialized before using this function!
   * \param [in] n        local index of the node whose neighborhood is queried
   * \param [out] qnnn    reference of a pointer to a node neighborhood, points to the correct object on output
   * [note:]
   * No data copy besides the pointer, which is much faster than the data copy.
   * [end of note]
   * throws exception in DEBUG if not initialized or if the node neighborhood is invalid
   */
  void inline get_neighbors(p4est_locidx_t n, const quad_neighbor_nodes_of_node_t*& qnnn) const {
    if (is_initialized) {
#ifdef CASL_THROWS
      if (is_qnnn_valid[n])
        qnnn = &neighbors[n];
      else {
        std::ostringstream oss;
        oss << "[ERROR]: const quad_neighbor_nodes_of_node_t* get_neighbors(p4est_locidx_t n): the neighborhood information for the node with idx " << n << " on processor " << p4est->mpirank << " is invalid.";
        throw std::invalid_argument(oss.str().c_str());
        qnnn = NULL;
      }
#else
      qnnn = &neighbors[n];
#endif
    }
    else {
#ifdef CASL_THROWS
      std::ostringstream oss;
      oss << "const quad_neighbor_nodes_of_node_t* get_neighbors(p4est_locidx_t n): cannot be used with uninitialized neighbors; on processor " << p4est->mpirank;
      throw std::runtime_error(oss.str().c_str());
#endif
      qnnn = NULL;
    }
  }

  /*!
   * \brief find_neighbor_cell_of_node finds the neighboring quadrant of a node in the given (i,j, k) direction. The direction
   * must be "diagonal" for the function to work! (e.g. (-1,1,1) ... no cartesian direction!).
   * \param [in] n              local index of the node whose neighboring cell is looked for
   * \param [in] i              the x search direction, -1 or 1
   * \param [in] j              the y search direction, -1 or 1
   * \param [in] k              the z search direction, -1 or 1, only in 3D
   * \param [out] quad_idx      the index of the found quadrant, in cumulative numbering over the trees. To fetch this quadrant
   *                            from its corresponding tree you need to substract the tree quadrant offset.
   *                            If no quadrant was found, this is set to NOT_A_P4EST_QUADRANT (not known from the local ghost
   *                            domain partition) of NOT_A_VALID_QUADRANT (past the edge of a nonperiodic domain)
   * \param [out] nb_tree_idx   the index of the tree in which the quadrant was found (valid and sensible if the quadrant was
   *                            actually found, of course)
   */
   void find_neighbor_cell_of_node( p4est_locidx_t n, DIM(char i, char j, char k), p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx ) const;

   double gather_neighbor_cells_of_node(set_of_neighboring_quadrants& cell_neighbors, const my_p4est_cell_neighbors_t* cell_ngbd, const p4est_locidx_t& node_idx, const bool& add_second_degree_neighbors = false) const;

  /*!
   * \brief dd_central computes the second derivatives along the cartesian direction der on all nodes and updates the ghosts
   * \param [in]  f       (ghosted) PETSc vector(s) to compute the derivatives on. It must be of appropriate node-sampling size and cannot be block-structured.
   * \param [out] fdd     (ghosted) PETSc vector(s) to store the appropriate second derivatives in. A check is done to ensure is has the same size as f on input.
   * \param [in]  n_vecs  number of vectors in the arrays f and fdd
   * \param [in]  der     cartesian direction along which the second derivatives must be calculated (dir::x, dir::y or dir::z)
   */
  void dd_central(const Vec f[], Vec fdd[], const unsigned int& n_vecs, const unsigned char& der) const;
  inline void dd_central(const Vec f, Vec fdd, const unsigned char& der) const
  {
    dd_central(&f, &fdd, 1, der);
  }
  inline void dxx_central(const Vec f, Vec fxx) const
  {
    dd_central(f, fxx, dir::x);
  }
  inline void dyy_central(const Vec f, Vec fyy) const
  {
    dd_central(f, fyy, dir::y);
  }

#ifdef P4_TO_P8
  inline void dzz_central(const Vec f, Vec fzz) const
  {
    dd_central(f, fzz, dir::z);
  }
#endif

  /*!
   * \brief second_derivatives_central computes all second-derivatives of the fields in f at all points and stores
   * them in block-structured output Petsc Parallel vectors in fdd.
   * The field(s) f may be multi-component block-structured vector(s), in which case, all second derivatives of all
   * components of f are calculated within this function!
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the second derivatives on (can be of block size bs_f >= 1)
   * \param [out] fdd     array of n_vecs PETSc vector(s) to store second derivatives results in, must be of block size bs_f*P4EST_DIM.
   * \param [in]  n_vecs  number of vectors to handle in the f and fdd arrays
   * \param [in]  bs_f    block size of the vectors in f (default is 1)
   */
  void second_derivatives_central(const Vec f[], Vec fdd[], const unsigned int& n_vecs, const unsigned int &bs_f=1) const;
  inline void second_derivatives_central(const Vec f, Vec fdd, const unsigned int &bs_f=1) const
  {
    second_derivatives_central(&f, &fdd, 1, bs_f);
  }

  /*!
   * \brief second_derivatives_central computes dxx, dyy, and dzz central at all points. Similar to the function above
   * except it use two/three regular vector in place of a single blocked vector. Easier to use but more expensive in terms
   * of MPI communications. Also note that fxx, fyy, and fzz cannot be obtained via VecDuplicate as this would share the
   * same VecScatter object and prevents simultanous updates of ghost values.
   *
   * Note that vectors f, fxx, fyy and fzz can all be of blocksize bs >=1 but have to be all of the same size! If bs > 1, the
   * derivatives of all the components of f are calculated within this function.
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the derivatives on (can be of block size bs >= 1)
   * \param [out] fxx     array of n_vecs PETSc vector(s) to store the xx-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fyy     array of n_vecs PETSc vector(s) to store the yy-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fzz     array of n_vecs PETSc vector(s) to store the zz-derivative(s) results in. (only in 3D)
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [in]  n_vecs  number of vectors to handle in the above arrays
   * \param [in]  bs      block size of the vectors in f, fxx, fyy and fzz (default is 1)
   */
  void second_derivatives_central(const Vec f[], DIM(Vec fxx[], Vec fyy[], Vec fzz[]), const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void second_derivatives_central(const Vec f, DIM(Vec fxx, Vec fyy, Vec fzz), const unsigned int &bs=1) const { second_derivatives_central(&f, DIM(&fxx, &fyy, &fzz), 1, bs); }

  /*!
   * \brief second_derivatives_central_above_threshold computes dxx, dyy, and dzz
   * central at all points where f is greater than threshold. Similar to the function
   * but disregards points where f < threshold.
   *
   * \param [in]  f   PETSc vector to compute the derivaties on
   * \param [in]  thr double threshold value mentioned above
   * \param [out] fxx PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fyy PETSc vector to store the results in. A check is done to ensure they have the same size as f
   * \param [out] fzz PETSc vector to store the results in. A check is done to ensure they have the same size as f (only in 3D)
   */
  void second_derivatives_central_above_threshold(const Vec f, double thr, DIM(Vec fxx, Vec fyy, Vec fzz)) const;

  /*!
   * \brief second_derivatives_central computes the second derivative
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the derivatives on
   * \param [out] fxxyyzz array of array(s) of size P4EST_DIM x n_vecs of PETSc vectors to store all results in.
   *                      fxxyyzz[der][k] is the second derivative along cartesian direction der for the kth field, on output.
   * \param [in]  n_vecs  number of vectors to handle
   * \param [in]  bs      block size of the vectors in f, fxx, fyy and fzz (default is 1)
   */
  inline void second_derivatives_central(const Vec f[], Vec *fxxyyzz[P4EST_DIM], const unsigned int &n_vecs, const unsigned int &bs = 1) {
    second_derivatives_central(f, DIM(fxxyyzz[0], fxxyyzz[1], fxxyyzz[2]), n_vecs, bs);
  }
  inline void second_derivatives_central(const Vec f, Vec fxx[P4EST_DIM], const unsigned int &bs = 1)
  {
    second_derivatives_central(&f, DIM(&fxx[0], &fxx[1], &fxx[2]), 1, bs);
  }

  /*!
   * \brief first_derivatives_central computes all first-derivatives of the fields in f at all points and stores
   * them in block-structured output Petsc Parallel vectors in fd.
   * The field(s) f may be multi-component block-structured vector(s), in which case, all first derivatives of all
   * components of f are calculated within this function!
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the first derivatives on (can be of block size bs_f >= 1)
   * \param [out] fd      array of n_vecs PETSc vector(s) to store first derivatives results in, must be of block size bs_f*P4EST_DIM.
   * \param [in]  n_vecs  number of vectors to handle in the f and fd arrays
   * \param [in]  bs_f    block size of the vectors in f (default is 1)
   */
  void first_derivatives_central(const Vec f[], Vec fd[], const unsigned int& n_vecs, const unsigned int &bs_f=1) const;
  inline void first_derivatives_central(const Vec f, Vec fdd, const unsigned int &bs_f=1) const
  {
    first_derivatives_central(&f, &fdd, 1, bs_f);
  }

  /*!
   * \brief first_derivatives_central computes dx, dy, and dz central at all points. Similar to the function above
   * except it use two/three regular vector in place of a single blocked vector. Easier to use but more expensive in terms
   * of MPI communications. Also note that fx, fy, and fz cannot be obtained via VecDuplicate as this would share the
   * same VecScatter object and prevents simultanous updates of ghost values.
   *
   * Note that vectors f, fx, fy and fz can all be of blocksize bs >=1 but have to be all of the same size! If bs > 1, the
   * derivatives of all the components of f are calculated within this function.
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the derivatives on (can be of block size bs >= 1)
   * \param [out] fx      array of n_vecs PETSc vector(s) to store the x-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fy      array of n_vecs PETSc vector(s) to store the y-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fz      array of n_vecs PETSc vector(s) to store the z-derivative(s) results in. (only in 3D)
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [in]  n_vecs  number of vectors to handle in the above arrays
   * \param [in]  bs      block size of the vectors in f, fx, fy and fz (default is 1)
   */
  void first_derivatives_central(const Vec f[], DIM(Vec fx[], Vec fy[], Vec fz[]), const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void first_derivatives_central(const Vec f, DIM(Vec fx, Vec fy, Vec fz), const unsigned int &bs=1) const { first_derivatives_central(&f, DIM(&fx, &fy, &fz), 1, bs); }

  inline void first_derivatives_central(const Vec f[], Vec *fxyz[P4EST_DIM], const unsigned int &n_vecs, const unsigned int &bs=1) const
  {
    first_derivatives_central(f, DIM(fxyz[0], fxyz[1], fxyz[2]), n_vecs, bs);
  }
  inline void first_derivatives_central(const Vec f, Vec fxyz[P4EST_DIM], const unsigned int &bs=1) const
  {
    first_derivatives_central(f, DIM(fxyz[0], fxyz[1], fxyz[2]), bs);
  }

  // Daniil would have to commment on this one
  void get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists) const;

  /*!
   * \brief memory_estimate estimates the memory required to store this my-p4est_node_neighbors_t object in number of bytes
   * \return a lower bound estimation on the rough number of bytes required by this object
   */
  inline size_t memory_estimate() const
  {
    size_t memory = 0;
    memory += neighbors.size()*sizeof (quad_neighbor_nodes_of_node_t);
#ifdef CASL_THROWS
    memory += is_qnnn_valid.size()*sizeof (bool);
#endif
    memory += layer_nodes.size()*sizeof (p4est_locidx_t);
    memory += local_nodes.size()*sizeof (p4est_locidx_t);
    memory += sizeof (is_initialized);
    memory += P4EST_DIM*sizeof (bool); // periodic
    return memory;
  }

private:
  /*!
   * \brief second_derivatives_central_using_block is a private procedure to shortcut the public second_derivatives_central
   *        procedure iff DXX_USE_BLOCKS is defined. In such a case, the object, creates appropriate block vector(s) internally
   *        uses it for internal computation and communication purposes and then remaps the values back to the standard
   *        non-blocked structured vectors. The internally created block-structured vector is destroyed internally as well.
   *        This is not an optimal strategy as it involves quite a lot of data copy, in my opinion: either use non-blocked
   *        vectors and work on those only, or use block-structured vector in the rest of your project(s) as well (i.e. not
   *        only internally)
   *
   * Note that vectors f, fxx, fyy and fzz can all be of blocksize bs >=1 but have to be all of the same size! If bs > 1, the
   * derivatives of all the components of f are calculated within this function.
   * \param [in]  f       array of n_vecs PETSc vector(s) to compute the derivatives on (can be of block size bs >= 1)
   * \param [out] fxx     array of n_vecs PETSc vector(s) to store the xx-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fyy     array of n_vecs PETSc vector(s) to store the yy-derivative(s) results in.
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [out] fzz     array of n_vecs PETSc vector(s) to store the zz-derivative(s) results in. (only in 3D)
   *                      A check is done in DEBUG to ensure they have the same size as vectors in f
   * \param [in]  n_vecs  number of vectors to handle in the above arrays
   * \param [in]  bs      block size of the vectors in f, fxx, fyy and fzz (default is 1)
   */
  void second_derivatives_central_using_block(const Vec f[], DIM(Vec fxx[], Vec fyy[], Vec fzz[]), const unsigned int& n_vecs, const unsigned int &bs = 1) const;
  inline void second_derivatives_central_using_block(const Vec f, DIM(Vec fxx, Vec fyy, Vec fzz), const unsigned int &bs = 1) const
  {
    second_derivatives_central_using_block(&f, DIM(&fxx, &fyy, &fzz), 1, bs);
  }

};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
