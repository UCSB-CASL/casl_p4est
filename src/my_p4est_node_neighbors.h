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

class async_computation_t {
public:
  virtual void foreach_local_node(p4est_locidx_t n) const = 0;
  virtual void ghost_update_begin() const = 0;
  virtual void ghost_update_end() const = 0;
  ~async_computation_t () {}
};

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

  /**
     * Initialize the QuadNeighborNodeOfNode information
     */

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *myb;
  std::vector< quad_neighbor_nodes_of_node_t > neighbors;
#ifdef CASL_THROWS
  std::vector<bool> is_qnnn_valid;
#endif
  std::vector<p4est_locidx_t> layer_nodes;
  std::vector<p4est_locidx_t> local_nodes;  
  bool is_initialized;
  bool periodic[P4EST_DIM];

  bool construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn) const;
  /* bool construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t& qnnn, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                           const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false) const;*/

  /*!
   * \brief update_all_but_hierarchy: inner private function to be called by the two different
   * public 'update' functions. This one updates the member variables p4est, ghost, and
   * nodes. It also clears and reconstructs the list of layer and local nodes.
   * If the node neighbors were previously initialized, this routine clears and reconstructs
   * them, as well as their validity flags.
   * \param [in] p4est_ the new p4est structure;
   * \param [in] ghost_ the new ghost later;
   * \param [in] nodes_ the new node structure.
   */
  void update_all_but_hierarchy(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_);
  /* void update_all_but_hierarchy(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                                            const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false); */

  inline void set_layer_and_local_nodes()
  {
    /* compute the layer and local nodes.
     * layer_nodes: this is the list of indices for nodes in the local range on this
     * processor (i.e. indices i such that 0 <= i < nodes->num_owned_indeps) but that
     * are tagged as ghost on at least one other processor
     * local_nodes: this is the list of indices for nodes in the local range on this
     * processor that are not included in the layer_nodes
     *
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
      // Therefore ni is purely local if ni->pad8 is not 0, otherwise it is a node layer
    }
  }

public:
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
   * \return returns a const ptr to the hierarchy structure
   */
  inline const my_p4est_hierarchy_t* get_hierarchy() const { return hierarchy; }

  /*!
   * \brief get_p4est
   * \return returns a const ptr to p4est structure
   */
  inline const p4est_t* get_p4est() const { return p4est; }

  /*!
   * \brief get_ghost
   * \return returns a const ptr to ghost structure
   */
  inline const p4est_ghost_t* get_ghost() const { return ghost; }

  /*!
   * \brief get_nodes
   * \return returns a const ptr to nodes structure
   */
  inline const p4est_nodes_t* get_nodes() const { return nodes; }

  /*!
   * \brief get_brick
   * \return returns a const ptr to brick structure
   */
  inline const my_p4est_brick_t* get_brick() const { return myb; }

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
  /* void init_neighbors(const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
                      const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false); */
  void clear_neighbors();
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_);
  void update(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_);
  /*
  void update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
              const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false);
  void update(p4est_t* p4est_, p4est_ghost_t* ghost_, p4est_nodes_t* nodes_, const bool &set_and_store_linear_interpolators=false, const bool &set_and_store_second_derivatives_operators=false,
              const bool &set_and_store_gradient_operator=false, const bool &set_and_store_quadratic_interpolators=false);
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
   * \brief run_async_computation runs an asynchronous computation across processors by overlapping computation and communication
   * \param async the abstract computation that needs to be run for each local node
   */
  inline void run_async_computation(const async_computation_t& async) const {
    for (size_t i=0; i<layer_nodes.size(); i++)
      async.foreach_local_node(layer_nodes[i]);
    async.ghost_update_begin();
    for (size_t i=0; i<local_nodes.size(); i++)
      async.foreach_local_node(local_nodes[i]);
    async.ghost_update_end();
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
   * \brief dd_central computes the second derivatives along cartesian direction der on all nodes and update the ghosts
   * \param [in]  f       (ghosted) PETSc vector(s) to compute the derivatives on must be of appropriate node-sampling size and cannot be block-structured.
   * \param [out] fdd     (ghosted) PETSc vector(s) to store the appropriate second derivatives in. A check is done to ensure they have the same size.
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
   * \brief second_derivatives_central computes dxx_central, dyy_central (and dzz_central, in 3D) at all points.
   * Theoretically this should have a better chance at hiding communications than above calls combined. The field(s)
   * f may be a multi-component block-structured vector, in which case, all second derivatives of all components of f
   * are calculated within this function!
   * \param [in]  f       PETSc vector(s) to compute the second derivatives on (can be of block size bs_f >= 1)
   * \param [out] fdd     PETSc vector(s) to store second derivatives results in, must be of block size bs_f*P4EST_DIM.
   * \param [in]  n_vecs  number of vectors to handle
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
   * of MPI. Also note that fxx, fyy, and fzz cannot be obtained via VecDuplicate as this would share the same VecScatter
   * object and avoid simultanous update.
   *
   * Note that vectors f, fxx, fyy and fzz can all be of blocksize bs >=1 but have to be of the same size! If bs > 1, the
   * derivatives of all the components of f are calculated within this function.
   * \param [in]  f       PETSc vector(s) to compute the derivatives on, of block size bs
   * \param [out] fxx     PETSc vector(s) to store the xx-derivative(s) results in. A check is done to ensure they have the same size as f
   * \param [out] fyy     PETSc vector(s) to store the yy-derivative(s) results in. A check is done to ensure they have the same size as f
   * \param [out] fzz     PETSc vector(s) to store the zz-derivative(s) results in. A check is done to ensure they have the same size as f (only in 3D)
   * \param [in]  n_vecs  number of vectors to handle
   * \param [in]  bs      block size of the vectors in f, fxx, fyy and fzz (default is 1)
   */
#ifdef P4_TO_P8
  void second_derivatives_central(const Vec f[], Vec fxx[], Vec fyy[], Vec fzz[], const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void second_derivatives_central(const Vec f, Vec fxx, Vec fyy, Vec fzz, const unsigned int &bs=1) const { second_derivatives_central(&f, &fxx, &fyy, &fzz, 1, bs); }
#else
  void second_derivatives_central(const Vec f[], Vec fxx[], Vec fyy[], const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void second_derivatives_central(const Vec f, Vec fxx, Vec fyy, const unsigned int &bs=1) const { second_derivatives_central(&f, &fxx, &fyy, 1, bs); }
#endif

  /*!
   * \brief second_derivatives_central computes the second derivative
   * \param [in]  f       PETSc vector(s) to compute the derivaties on
   * \param [out] fxx     array of array(s) of size P4EST_DIM of PETSc vectors to store all results in.
   * \param [in]  n_vecs  number of vectors to handle
   * \param [in]  bs      block size of the vectors in f, fxx, fyy and fzz (default is 1)
   */
  inline void second_derivatives_central(const Vec f[], Vec *fxx[P4EST_DIM], const unsigned int &n_vecs, const unsigned int &bs = 1) {
#ifdef P4_TO_P8
    second_derivatives_central(f, fxx[0], fxx[1], fxx[2], n_vecs, bs);
#else
    second_derivatives_central(f, fxx[0], fxx[1], n_vecs, bs);
#endif
  }
  inline void second_derivatives_central(const Vec f, Vec fxx[P4EST_DIM], const unsigned int &bs = 1)
  {
#ifdef P4_TO_P8
    second_derivatives_central(&f, &fxx[0], &fxx[1], &fxx[2], 1, bs);
#else
    second_derivatives_central(&f, &fxx[0], &fxx[1], 1, bs);
#endif
  }

  /*!
   * \brief first_derivatives_central computes dx_central, dy_central (and dz_central, in 3D) at all points.
   * Theoretically this should have a better chance at hiding communications than above calls combined. The field(s)
   * f may be a multi-component block-structured vector, in which case, all second derivatives of all components of f
   * are calculated within this function!
   * \param [in]  f       PETSc vector(s) to compute the first derivatives on (can be of block size bs_f >= 1)
   * \param [out] fd      PETSc vector(s) to store first derivatives results in, must be of block size bs_f*P4EST_DIM.
   * \param [in]  n_vecs  number of vectors to handle
   * \param [in]  bs_f    block size of the vectors in f (default is 1)
   */
  void first_derivatives_central(const Vec f[], Vec fd[], const unsigned int& n_vecs, const unsigned int &bs_f=1) const;
  inline void first_derivatives_central(const Vec f, Vec fdd, const unsigned int &bs_f=1) const
  {
    first_derivatives_central(&f, &fdd, 1, bs_f);
  }

  /*!
   * \brief first_derivatives_central computes the first derivatives using central difference
   *
   * Note that vectors in f, fx, fy, fz can all be of blocksize bs >=1 but have to be of the same size! If bs > 1, the
   * derivatives of all the components of f are calculated within this function.
   * \param [in]  f       PETSc vector(s) to compute the derivatives on, of block size bs
   * \param [out] fx      PETSc vector(s) to store the x-derivative(s) results in. A check is done to ensure they have the same size as f
   * \param [out] fy      PETSc vector(s) to store the y-derivative(s) results in. A check is done to ensure they have the same size as f
   * \param [out] fz      PETSc vector(s) to store the z-derivative(s) results in. A check is done to ensure they have the same size as f (only in 3D)
   * \param [in]  n_vecs  number of vectors to handle
   * \param [in]  bs      block size of the vectors in f, fx, fy and fz (default is 1)
   */
#ifdef P4_TO_P8
  void first_derivatives_central(const Vec f[], Vec fx[], Vec fy[], Vec fz[], const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void first_derivatives_central(const Vec f, Vec fx, Vec fy, Vec fz, const unsigned int &bs=1) const { first_derivatives_central(&f, &fx, &fy, &fz, 1, bs); }
#else
  void first_derivatives_central(const Vec f[], Vec fx[], Vec fy[], const unsigned int& n_vecs, const unsigned int &bs=1) const;
  inline void first_derivatives_central(const Vec f, Vec fx, Vec fy, const unsigned int &bs=1) const { first_derivatives_central(&f, &fx, &fy, 1, bs); }
#endif

  inline void first_derivatives_central(const Vec f[], Vec *fxyz[P4EST_DIM], const unsigned int &n_vecs, const unsigned int &bs=1) const
  {
#ifdef P4_TO_P8
    first_derivatives_central(f, fxyz[0], fxyz[1], fxyz[2], n_vecs, bs);
#else
    first_derivatives_central(f, fxyz[0], fxyz[1], n_vecs, bs);
#endif
  }
  inline void first_derivatives_central(const Vec f, Vec fxyz[P4EST_DIM], const unsigned int &bs=1) const
  {
#ifdef P4_TO_P8
    first_derivatives_central(f, fxyz[0], fxyz[1], fxyz[2], bs);
#else
    first_derivatives_central(f, fxyz[0], fxyz[1], bs);
#endif
  }

  void get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists) const;

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
#ifdef P4_TO_P8
  void second_derivatives_central_using_block(const Vec f[], Vec fxx[], Vec fyy[], Vec fzz[], const unsigned int& n_vecs, const unsigned int &bs = 1) const;
  inline void second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy, Vec fzz, const unsigned int &bs = 1) const
  {
    second_derivatives_central_using_block(&f, &fxx, &fyy, &fzz, 1, bs);
  }
#else
  void second_derivatives_central_using_block(const Vec f[], Vec fxx[], Vec fyy[], const unsigned int& n_vecs, const unsigned int &bs = 1) const;
  inline void second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy, const unsigned int &bs = 1) const
  {
    second_derivatives_central_using_block(&f, &fxx, &fyy, 1, bs);
  }
#endif

};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
