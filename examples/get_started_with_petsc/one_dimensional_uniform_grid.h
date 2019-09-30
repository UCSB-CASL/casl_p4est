#ifndef ONE_DIMENSIONAL_UNIFORM_GRID_H
#define ONE_DIMENSIONAL_UNIFORM_GRID_H

#include <petsc.h>
#include <vector>
#include <my_mpi_world.h>

typedef struct ghost_node{
  ghost_node(PetscInt array_idx_=0, PetscInt local_idx_=0) : array_idx(array_idx_), local_idx(local_idx_) {}
  PetscInt array_idx; // index of the ghost node in the local (ghosted) array of any parallel Petsc vector (--> always greater than n_owned)
  PetscInt local_idx; // logical index of the ghost node, as seen from the current processor (hence, can be negative or larger than n_owned)
} ghost_node;


/*!
 * \brief The one_dimensional_uniform_grid class is an example class to support the logical link between Petsc parallel vectors and
 * grid node local indices. In this class, we consider that a uniform one-dimensional grid is distributed over several processes.
 * In this context, each process owns n_owned continuous grid nodes but they are also aware of ghost_layer_size ghost nodes on either
 * border of their subdomain IF such grid nodes actually exist. In the current class, we have assumed (and enforced in DEBUG mode)
 * that ghost_layer_size must be the same on all processes. The grid can be periodic or not.
 *
 * For instance, consider the following sequence of grid nodes for a non-periodic grid:
 *
 * xmin                                                                                            xmax
 * |                                                                                                  |
 * ....................................................................................................
 *
 * Let us now consider that this grid is evenly distributed over 5 processes with a ghost layer of size 2, then every process locally
 * knows of
 *
 * xmin                                                                                            xmax
 * |                                                                                                  |
 * ....................**                                                                                     for process of rank 0
 *                   **....................**                                                                 for process of rank 1
 *                                       **....................**                                             for process of rank 2
 *                                                           **....................**                         for process of rank 3
 *                                                                               **....................       for process of rank 4
 *
 * where '.' represents a node that the process owns whereas '*' represents a ghost grid node, i.e. a grid node that exists, that is
 * owned by another process and which may require for local calculations of interest (if it belongs to the discretization stencil for
 * some owned node, for instance).
 *
 * In the current context, this class was designed with the following convention: every grid node that is locally known is assigned
 * an index j_loc such that relates to its coordinate through x(node j_loc) = x_offset + j_loc*delta_x where x_offset is the coordinate
 * of the very first grid node owned by the process. In this framework, the local indices of ghost nodes to the left of the current
 * partition, if they exist, are negative (-1, -2, ---, -ghost_layer_size) while the local indices of the ghost nodes to the right of
 * the current partition, if they exist, are greater than or equal to n_owned (n_owned, n_owned+1, ---, n_owned+ghost_layer_size-1).
 * Therefore the local indices range from loc_idx_min (either -ghost_layer_size or 0) to loc_idx_max (n_owned-1 or n_owned+ghost_layer_size-1).
 *
 * In this framework, the global index of any node of local index j_loc is defined in a straighforward fashion as (offset_idx+j_loc)
 * where offset_idx is the total number nodes owned by processes of rank less than the current process' rank, i.e.
 * offset_idx = 0                                                                                             for process of rank 0
 * offset_idx = (offset_idx for rank (r-1) + number of nodes owned by process of rank (r-1))                  for process of rank r > 0
 *
 * The offset_on_rank vector stores the offset indices for every process and it is known by all process. The global number of grid nodes
 * is also known by all processes: it uniquely and unambiguously defines delta_x.
 *
 * The set of nodes that are known locally is divided into three mutually exclusive sets
 * 1) the inner_nodes : a vector of local indices of the locally owned nodes that are not to be shared or communicated with any
 *                      other process;
 * 2) the layer_nodes : a vector of local indices of the locally owned nodes that are seen as ghost nodes by other process(es) and, thus,
 *                      may need to be communicated with other process(es) at some point;
 * 3) the ghost nodes : a vector of ghost_node structures, i.e., a vector of couples of indices for every ghost nodes. These couples of
 *                      indices include:
 *                      i)  the local index of the considered ghost node (as explained above)
 *                      ii) the array index of the considered ghost node, i.e. the index of the corresponding ghost value stored in the
 *                          local array of a parallel Petsc vector. [read the comments in my_petsc_utils.h to understand why this is required]
 *                      The ghost nodes are stored by increasing local index.
 *
 * Developer: Raphael Egan (Sept. 2019)
 */
class one_dimensional_uniform_grid
{
  const my_mpi_world& mpi;

  PetscInt n_owned, n_global, ghost_layer_size;
  std::vector<PetscInt> offset_on_rank;
  const double xmin, xmax;
  const bool periodic;
  std::vector<PetscInt> layer_nodes;        // owned but seen as ghost for another proc (standard vector of local indices, not global)
  std::vector<PetscInt> inner_nodes;        // owned and invisible to other procs (standard vector  of local indices, not global)
  std::vector<ghost_node> ghost_nodes;      // owned by another proc but may be required for local calculation on locally owned nodes, with increasing local_idx


  PetscInt loc_idx_min, loc_idx_max;
  bool node_exists(PetscInt loc_idx) const { return (loc_idx >= loc_idx_min && loc_idx < loc_idx_max); }
  bool node_is_ghost(PetscInt loc_idx) const { return (((loc_idx_min <= loc_idx) && (loc_idx <0)) || (n_owned <= loc_idx && loc_idx < loc_idx_max)); }

  double second_derivative_of_field_at_node(PetscInt idx_loc, const double *node_sampled_field_p) const;

  void create_and_preallocate_compact_finite_difference_matrices(Mat &lhs_matrix, Mat &rhs_matrix, const unsigned int &OOA) const;
  PetscErrorCode create_compact_finite_differences_operators(Mat &lhs_matrix, Mat &rhs_matrix, const unsigned int &OOA) const;

public:
  /*!
   * \brief one_dimensional_uniform_grid simply creates an object but does not set it. A call to set_partition_and_ghosts is required to fully define the grid
   * \param [in] mpi_:      the mpi environment on which the grid is distributed;
   * \param [in] xmin_:     the left border of the one-dimensional computational domain;
   * \param [in] xmax_:     the right border of the one-dimensional computational domain;
   * \param [in] periodic_: defines the grid as periodic if true.
   */
  one_dimensional_uniform_grid(const my_mpi_world &mpi_,  const double &xmin_, const double &xmax_, const bool &periodic_) :
    mpi(mpi_), xmin(xmin_), xmax(xmax_), periodic(periodic_) { }

  /*!
   * \brief set_partition_and_ghosts determines the partition of the grid.
   * \param [in] n_local:           number of grid nodes owned by the current process (does not need to be constant over all processes);
   * \param [in] ghost_layer_size_: number of grid nodes required past either border of the local subdomain (if they exist).
   */
  void set_partition_and_ghosts(const PetscInt &n_local, const PetscInt &ghost_layer_size_);

  /*!
   * \brief get_x_of_node
   * \param [in] j_loc: local index of the node of interest
   * \return the coordinate of the local node of local index j_loc
   */
  double get_x_of_node(const PetscInt j_loc) const;
  /*!
   * \brief get_delta_x
   * \return delta x, i.e. the (uniform) grid spacing
   */
  double get_delta_x() const;
  /*!
   * \brief global_idx_of_local_node
   * \param [in] j_loc: local index of the node of interest
   * \return the global index of the node of local index j_loc
   */
  PetscInt global_idx_of_local_node(const PetscInt j_loc) const;

  inline PetscInt number_of_locally_owned_nodes() const { return n_owned; }
  inline PetscInt global_number_of_nodes() const { return n_global; }
  inline PetscInt number_of_ghost_nodes() const { return ghost_nodes.size(); }
  inline int mpisize() const { return mpi.size(); }
  inline int mpirank() const { return mpi.rank(); }
  inline MPI_Comm mpicomm() const { return mpi.comm(); }

  /*!
   * \brief local_idx_of_ghost_node
   * \param [in] k: index of the ghost node in the list of ghost nodes, i.e. value in [0, 1, ..., ghost_nodes.size()-1[
   * \return returns the local index of the kth ghost node
   */
  inline PetscInt local_idx_of_ghost_node(const PetscInt &k) const { return ghost_nodes[k].local_idx; }
  /*!
   * \brief array_idx_of_node
   * \param [in] j_loc: local index of the node of interest
   * \return the index associated with the entry in the local array of values in a parallel Petsc vector that corresponds to the local node
   * of local index j_loc
   */
  PetscInt array_idx_of_node(const PetscInt &j_loc) const;

  /*!
   * \brief calculate_second_derivative_of_field
   * \param [in]    node_sampled_field: the parallel Petsc vector of node-sampled values of the fields whose second derivative needs to be calculated
   * \param [inout] second_derivative: the parallel Petsc vector where the second derivatives will be stored. The vector needs to exist and to be of
   *                                   appropriate size before calling this function.
   */
  void calculate_second_derivative_of_field(Vec node_sampled_field, Vec second_derivative) const;

  /*!
   * \brief calculate_first_derivative_compact_fd
   * \param [in]    node_sampled_function                       : the parallel Petsc vector of node-sampled values of the fields whose first derivative
   *                                                              needs to be calculated
   * \param [inout] first_derivative_compact_finite_differences : the parallel Petsc vector where the first derivatives will be stored. The vector needs
   *                                                              to exist and to be of appropriate size before calling this function.
   * \param [in]    OOA                                         : desired order of accuracy (4 or 6)
   *
   * NOTE: the pure exact order of accuracy in the implementation is respected for periodic problems only. In case of non-periodic domain, the wall node
   * equations follow an off-centered discretization and the size of the discretization stencil is reduced if necessary to be entirely included in the domain
   */
  void calculate_first_derivative_compact_fd(Vec node_sampled_function, Vec first_derivative_compact_finite_differences, const unsigned int &OOA) const;

  /*!
   * \brief shuffle repartitions the grid nodes with random numbers of locally owned grid nodes without changing the global number of nodes
   */
  void shuffle();
  /*!
   * \brief remap scatters the data from a parallel Petsc vector corresponding to another grid with the same global number of grid nodes to the new, current
   * layout.
   * \param [in] vector_on_olg_grid:        parallel Petsc vector of node-values corresponding to another grid partition
   * \param [inout] vector_on_current_grid: parallel Petsc vector of node-values corresponding to the current grid partition, this one will be filled consistently
   *                                        with the values from vector_on_old_grid.
   */
  void remap(Vec vector_on_olg_grid, Vec vector_on_current_grid) const;

};

#endif // ONE_DIMENSIONAL_UNIFORM_GRID_H
