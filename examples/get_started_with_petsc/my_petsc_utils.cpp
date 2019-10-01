#include <my_petsc_utils.h>

/*
 * Petsc parallel vectors:
 * -----------------------
 * Petsc parallel vectors store (large) vectors in a distributed fashion, i.e on several processes. The full chunk
 * of data is divided into as many pieces of contiguous data as there are processes working together in the MPI
 * communicator. These pieces are distributed to the relevant processes, which store and can access them but not
 * the rest of the full chunk of data (not directly, at least).
 *
 * Every individual process can also request to be updated with values belonging to other process(es), i.e. ghost
 * values, if specified at the time of the vector construction.
 *
 * Every process has direct access to a local vector that encompasses the locally owned values AND the ghost values.
 * This local array of data always stores the locally owned values first; the ghost values are stored after the ones
 * owned locally.
 *
 * CONCLUSION: the indices of the ghost values in the local arrays associated with the parallel Petsc vector may not
 * correspond to local indices defined consistently with the geometry of the grid. For instance, in this illustrative
 * parallel one-dimensional framework, if the local grid partition of a process is defined as
 *
 *  values on node:             A   B   | L_{0} L_{1} ... L_{n_owned-2} L_{n_owned-1} | C         D
 *  local GRID index of node:   -2  -1  | 0     1     ... n_owned-2     n_owned-1     | n_owned   n_owned+1
 *
 * where A, B, C and D are values on ghost grid nodes while L_{i} is the value on local node i, the corresponding
 * storage in the local array of the Petsc parallel vector will be
 *  values on node:             L_{0} L_{1} ... L_{n_owned-2} L_{n_owned-1} | A         B         C         D
 *  index in local DATA ARRAY
 *  of Petsc vector:            0     1     ... n_owned-2     n_owned-1     | n_owned   n_owned+1 n_owned+2 n_owned+3
 *
 * NB: the order in which the ghost values A, B, C and D are stored is defined by the user at the time of construction
 * and according to a user-defined convention that the user is expected to consistenly follow. In this illustrative
 * example, we store the ghost values by increasing local grid index, but another convention could have been used,
 * of course...
 *
 */

/*!
 * \brief vec_create_on_one_dimensional_grid creates a parallel Petsc vector to sample node values on a parallel
 * one-dimensional grid with the appropriate ghost values to be "synchronizable".
 * \param [in]    grid  : the parallel one-dimensional grid on which node-sampled values will be stored in vv
 * \param [inout] vv    : a pointer to the vector to be created
 * NOTE: vv should not be pointing to an existing parallel Petsc vector beforehand or reference to that vector
 * will be lost and memory will leak!
 * \return a PetscErrorCode to be checked for successful operation
 */
PetscErrorCode vec_create_on_one_dimensional_grid(const one_dimensional_uniform_grid& grid, Vec *vv)
{
  PetscErrorCode ierr = 0;
  PetscInt num_local = grid.number_of_locally_owned_nodes();
  std::vector<PetscInt> global_indices_of_ghost_nodes(grid.number_of_ghost_nodes(), 0);
  PetscInt num_global = grid.global_number_of_nodes();

  for (PetscInt i = 0; i<grid.number_of_ghost_nodes(); ++i)
    global_indices_of_ghost_nodes[i] = grid.global_idx_of_local_node(grid.local_idx_of_ghost_node(i));

  ierr = VecCreateGhost(grid.mpicomm(), num_local, num_global,
                        grid.number_of_ghost_nodes(), global_indices_of_ghost_nodes.data(), vv); CHKERRQ(ierr);

  ierr = VecSetFromOptions(*vv); CHKERRQ(ierr);
  // the above line sets a few additional options for internal methods and data structure management if specified by
  // the user at execution time (no such options is set in 99.9% of cases, though)

  return ierr;
}

/*!
 * \brief sample_vector_on_grid: parses the grid nodes of a given grid, evaluates a given function and stores the
 * corresponding values in the given parallel Petsc vector, that is expected to be consistent with the given grid
 * \param [inout] v                 : parallel Petsc vector to be filled with the node-sampled values of the given
 *                                    function. The vector must exist and be of appropriate size
 * \param [in]    grid              : parallel one-dimensional grid defining the node to be considered
 * \param [in]    sampling_function : the function to be sampled
 * NOTE: ghost values are calculated as well, no synchronization required after completion (if sampling_function is
 * sufficiently well-defined for the ghost nodes as well)
 * \return a PetscErrorCode to be checked for successful operation
 */
PetscErrorCode sample_vector_on_grid(Vec v, const one_dimensional_uniform_grid &grid, cont_function &sampling_function)
{
  PetscErrorCode ierr;
  // PROPER WAY TO DO IT, ACCORDING TO THE DOCUMENTATION
//  PetscInt n_local_with_ghosts;
//  Vec v_ghost_local_form; // nothing created here, we just want an accessor to local data, this is nothing but pointer(s)
//  double *local_ghosted_array;
//  // get access to the local, ghosted vector
//  ierr = VecGhostGetLocalForm(v, &v_ghost_local_form); CHKERRQ(ierr);
//  ierr = VecGetSize(v_ghost_local_form, &n_local_with_ghosts); CHKERRQ(ierr);
//  ierr = n_local_with_ghosts != grid.number_of_locally_owned_nodes() + grid.number_of_ghost_nodes(); CHKERRQ(ierr);
//  ierr = VecGetArray(v_ghost_local_form, &local_ghosted_array); CHKERRQ(ierr);
//  // fill the local data array with the provided function
//  for (PetscInt idx = 0; idx < grid.number_of_locally_owned_nodes(); ++idx)
//    local_ghosted_array[idx] = sampling_function(grid.get_x_of_node(idx));
//  for (PetscInt ghost_idx = 0; ghost_idx < grid.number_of_ghost_nodes(); ++ghost_idx)
//    local_ghosted_array[grid.number_of_locally_owned_nodes()+ghost_idx] = sampling_function(grid.get_x_of_node(grid.local_idx_of_ghost_node(ghost_idx)));
//  ierr = VecRestoreArray(v_ghost_local_form, &local_ghosted_array); CHKERRQ(ierr);
//  ierr = VecGhostRestoreLocalForm(v, &v_ghost_local_form); CHKERRQ(ierr);

  // ACTUALLY WHAT WE DO:
  double *local_ghosted_array;
  ierr = VecGetArray(v, &local_ghosted_array); CHKERRXX(ierr);
  for (PetscInt idx = 0; idx < grid.number_of_locally_owned_nodes(); ++idx)
    local_ghosted_array[idx] = sampling_function(grid.get_x_of_node(idx));
  for (PetscInt ghost_idx = 0; ghost_idx < grid.number_of_ghost_nodes(); ++ghost_idx)
    local_ghosted_array[grid.number_of_locally_owned_nodes()+ghost_idx] = sampling_function(grid.get_x_of_node(grid.local_idx_of_ghost_node(ghost_idx)));
  ierr = VecRestoreArray(v, &local_ghosted_array); CHKERRXX(ierr);

  return  ierr;
}

/*!
 * \brief export_in_binary_format exports the data stored in a petsc parallel vector in a binary format on disk
 * \param [in] vv               : the Petsc parallel vector to be exported
 * \param [in] filename         : the absolute path to the file to be created for exportation
 * \return a PetscErrorCode to be checked for successful operation
 */
PetscErrorCode export_in_binary_format(Vec vv, const char* filename)
{
  PetscViewer petsc_exportation_tool;
  PetscErrorCode ierr;
  MPI_Comm mpicomm;
  ierr = PetscObjectGetComm((PetscObject) vv, &mpicomm); CHKERRQ(ierr);
  ierr = PetscViewerCreate(mpicomm, &petsc_exportation_tool); CHKERRQ(ierr);
  ierr = PetscViewerSetType(petsc_exportation_tool, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(petsc_exportation_tool, FILE_MODE_WRITE); CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(petsc_exportation_tool, filename);CHKERRQ(ierr);
  ierr = VecView(vv, petsc_exportation_tool); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(petsc_exportation_tool); CHKERRQ(ierr);
  return ierr;
}
