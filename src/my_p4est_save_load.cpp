#ifdef P4_TO_P8
#include "my_p8est_save_load.h"
#include <p8est_extended.h>
#else
#include "my_p4est_save_load.h"
#include <p4est_extended.h>
#endif

#include <algorithm>

/*!
 * \brief set_quadrant_data_for_exportation: sets the exportation-relevant data associated with a
 *                            a given quadrant in a p4est object. This function is of type
 *                            p4est_init_t and is called iteratively within p4est_reset_data
 * \param forest      [in]    pointer to the p4est object whose quadrant data must be set
 * \param which_tree  [in]    the tree index of the quadrant of interest
 * \param q           [inout] pointer to the quadrant of interest in the forest
 * IMPORTANT NOTE:  forest->user_pointer must be a pointer towards a (valid) structure of type
 *                  pointers_to_grid_data_info, including pointers to the nodes (mandatorily),
 *                  the faces (optional) and the corresponding node (mandatory) and face
 *                  (optional) global index offsets (per proc rank).
 *                  The allocated quadrant-data is supposed to be of size
 *                  a) P4EST_CHILDREN*sizeof(p4est_gloidx_t) if ptr->faces == NULL
 *                  b) (P4EST_CHILDREN + P4EST_FACES)*sizeof(p4est_gloidx_t) if ptr->faces != NULL
 *                  where ptr = (pointers_to_grid_data_info*) forest->user_pointer;
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
static void set_quadrant_data_for_exportation(p4est_t* forest, p4est_topidx_t which_tree, p4est_quadrant_t * q)
{
  pointers_to_grid_data_info* ptr = (pointers_to_grid_data_info*) forest->user_pointer;
  p4est_nodes_t* nodes                                    = ptr->nodes;
  const my_p4est_faces_t* faces                           = ptr->faces;
  const std::vector<p4est_gloidx_t>* node_offset_on_proc  = ptr->global_node_offsets;
  const std::vector<p4est_gloidx_t>* face_offset_on_proc[P4EST_DIM];
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    face_offset_on_proc[dir]                              = ptr->global_face_offsets[dir];

  p4est_tree_t* tree = p4est_tree_array_index(forest->trees, which_tree);
  P4EST_ASSERT(((char*)q - tree->quadrants.array)%sizeof(p4est_quadrant_t) == 0);
  p4est_locidx_t quad_idx = tree->quadrants_offset + ((char*)q - tree->quadrants.array)/sizeof(p4est_quadrant_t); // this should be the local index of the quadrant...
  P4EST_ASSERT((quad_idx >=0) && (quad_idx < forest->local_num_quadrants));

  p4est_gloidx_t *global_indices  = (p4est_gloidx_t*) q->p.user_data;

  for (u_char k = 0; k < P4EST_CHILDREN; ++k)
  {
    p4est_locidx_t local_node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+k];
    if(local_node_idx < nodes->num_owned_indeps) // local node
      global_indices[k] = local_node_idx + node_offset_on_proc->at(forest->mpirank);
    else // ghost node
    {
      P4EST_ASSERT((local_node_idx >= nodes->num_owned_indeps) && ((size_t) local_node_idx < nodes->indep_nodes.elem_count));
      p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes, local_node_idx);
      global_indices[k] = node_offset_on_proc->at(nodes->nonlocal_ranks[local_node_idx-nodes->num_owned_indeps]) + node->p.piggy3.local_num;
    }
  }

  if(faces != NULL)
  {
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      for (u_char ii = 0; ii < 2; ++ii) {
        p4est_locidx_t local_face_idx = faces->q2f(quad_idx, 2*dir+ii);
        if((local_face_idx >=0) && (local_face_idx < faces->num_local[dir]))
          global_indices[P4EST_CHILDREN+2*dir+ii] = local_face_idx + face_offset_on_proc[dir]->at(forest->mpirank);
        else if(local_face_idx >= faces->num_local[dir])
          global_indices[P4EST_CHILDREN+2*dir+ii] = faces->ghost_local_num[dir][local_face_idx-faces->num_local[dir]] + face_offset_on_proc[dir]->at(faces->nonlocal_ranks[dir][local_face_idx-faces->num_local[dir]]);
        else
        {
          P4EST_ASSERT(local_face_idx == NO_VELOCITY);
          global_indices[P4EST_CHILDREN+2*dir+ii] = local_face_idx;
        }
      }
    }
  }
}

PetscErrorCode VecDump(const char fname[], u_int n_vecs, const Vec *x, PetscBool skippheader, PetscBool usempiio)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscObjectGetComm((PetscObject)x[0],&comm);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  if (skippheader) { ierr = PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  if (usempiio) { ierr = PetscViewerBinarySetUseMPIIO(viewer, PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetName(viewer, fname);CHKERRQ(ierr);

  for (u_int kk = 0; kk < n_vecs; ++kk) {
    ierr = VecView(x[kk],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode VecScatterCreateChangeLayout(Vec from_disk, Vec in_memory, const int dtype,
                                            p4est_t *augmented_forest, p4est_ghost_t* ghost, const p4est_nodes_t* nodes, const my_p4est_faces_t* faces, const int dim,
                                            VecScatter& ctx)
{
  PetscErrorCode ierr = 0;
#ifdef CASL_THROWS
  PetscInt size_from_disk, size_in_memory;
  ierr = VecGetSize(from_disk, &size_from_disk); CHKERRXX(ierr);
  ierr = VecGetSize(in_memory, &size_in_memory); CHKERRXX(ierr);
  if (size_from_disk != size_in_memory)
    throw std::invalid_argument("[ERROR] VecScatterCreateChangeLayout: Change layout is only supported for vectors with the same global size");
#endif

  IS index_set_in_memory_local, index_set_from_disk;
  ISLocalToGlobalMapping l2g;
  PetscInt petsc_int_l2g_size;
  p4est_locidx_t l2g_size, dof_counter;
  std::vector<PetscInt> global_indices;
  const u_int blocksize = (dtype == NODE_BLOCK_VECTOR_DATA ? P4EST_DIM : 1); // default behavior is blocksize of 1

  ierr = VecGetLocalToGlobalMapping(in_memory, &l2g); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingGetSize(l2g, &petsc_int_l2g_size); CHKERRXX(ierr);
  ierr = ISCreateStride(augmented_forest->mpicomm, petsc_int_l2g_size, 0, 1, &index_set_in_memory_local); CHKERRXX(ierr);

  l2g_size = (p4est_locidx_t) petsc_int_l2g_size;
  // [Raphael]: I found the above conversion is needed whenever PetscInt and p4est_locidx_t
  // do not have the same size. It is usually the case on local machines or small clusters,
  // but it is not necessarily on STAMPEDE2 for instance and the code crashed because of this...
  global_indices.resize(l2g_size, -1); // initialize every corresponding global index to -1
  dof_counter = 0;

  switch (dtype) {
  case CELL_DATA:
  {
    P4EST_ASSERT(l2g_size == ((int) blocksize)*(augmented_forest->local_num_quadrants + ((p4est_locidx_t) ghost->ghosts.elem_count)));
    // global ordering is conserved in for the quadrants (stored and distributed following a z-order)
    for (p4est_topidx_t tree_idx = augmented_forest->first_local_tree; tree_idx <= augmented_forest->last_local_tree; ++tree_idx) {
      const p4est_tree_t* tree = p4est_tree_array_index(augmented_forest->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      {
        for (u_int comp = 0; comp < blocksize; ++comp)
          global_indices[blocksize*dof_counter + comp] = blocksize*(q + tree->quadrants_offset + augmented_forest->global_first_quadrant[augmented_forest->mpirank]) + comp;
        dof_counter++;
      }
    }
    for (int r = 0; r < augmented_forest->mpisize; ++r)
      for (p4est_locidx_t q = ghost->proc_offsets[r]; q < ghost->proc_offsets[r+1]; ++q) {
        const p4est_quadrant_t* quad = (const p4est_quadrant_t*) sc_array_index(&ghost->ghosts, q);
        for (u_int comp = 0; comp < blocksize; ++comp)
          global_indices[blocksize*dof_counter + comp] = (PetscInt) (blocksize*(quad->p.piggy3.local_num + augmented_forest->global_first_quadrant[r]) + comp);
        dof_counter++;
      }
    break;
  }
  case NODE_BLOCK_VECTOR_DATA:
  case NODE_DATA:
  {
    P4EST_ASSERT(nodes != NULL);
    P4EST_ASSERT(l2g_size == ((int) blocksize)*((p4est_locidx_t) nodes->indep_nodes.elem_count));
    P4EST_ASSERT(augmented_forest->data_size%(sizeof(p4est_gloidx_t)) == 0);
    u_int n_global_indices_per_quad = augmented_forest->data_size/(sizeof(p4est_gloidx_t));
    P4EST_ASSERT(n_global_indices_per_quad >= P4EST_CHILDREN);
    p4est_gloidx_t* ghost_data = (p4est_gloidx_t*) P4EST_ALLOC(p4est_gloidx_t, n_global_indices_per_quad*ghost->ghosts.elem_count);
    p4est_ghost_exchange_data (augmented_forest, ghost, ghost_data);

    // Calculate the global node offset per proc
    std::vector<p4est_gloidx_t> global_node_offset_on_proc(augmented_forest->mpisize, 0);
    for (int r = 1; r<augmented_forest->mpisize; ++r)
      global_node_offset_on_proc[r] = global_node_offset_on_proc[r - 1] + (p4est_gloidx_t)nodes->global_owned_indeps[r - 1];

    for (p4est_topidx_t tree_idx = augmented_forest->first_local_tree; tree_idx <= augmented_forest->last_local_tree; ++tree_idx) {
      p4est_tree_t* tree = p4est_tree_array_index(augmented_forest->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        const p4est_quadrant_t* quadrant = (const p4est_quadrant_t*) p4est_quadrant_array_index(&tree->quadrants, q);
        p4est_gloidx_t* quad_data = (p4est_gloidx_t*) quadrant->p.user_data;
        for (u_char i = 0; i < P4EST_CHILDREN; ++i) {
          p4est_locidx_t local_node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx + i];
          P4EST_ASSERT(local_node_idx >= 0 && local_node_idx < (p4est_locidx_t) nodes->indep_nodes.elem_count);
          if(global_indices.at(blocksize*local_node_idx) == -1)
          {
            for (u_int comp = 0; comp < blocksize; ++comp)
              global_indices[blocksize*local_node_idx + comp] = blocksize*quad_data[i] + comp;
            dof_counter++;
          }
        }
      }
    }
    for (size_t q = 0; q < ghost->ghosts.elem_count; ++q) {
      for (u_char i = 0; i < P4EST_CHILDREN; ++i) {
        p4est_locidx_t local_node_idx = nodes->local_nodes[P4EST_CHILDREN*(q+augmented_forest->local_num_quadrants) + i];
        P4EST_ASSERT(local_node_idx >= 0 && local_node_idx < (p4est_locidx_t) nodes->indep_nodes.elem_count);
        if(global_indices.at(blocksize*local_node_idx) == -1)
        {
          for (u_int comp = 0; comp < blocksize; ++comp)
            global_indices[blocksize*local_node_idx + comp] = blocksize*ghost_data[q*n_global_indices_per_quad + i] + comp;
          dof_counter++;
        }
      }
    }
    P4EST_FREE(ghost_data);
    break;
  }
  case FACE_DATA:
  {
    P4EST_ASSERT(faces != NULL);
    P4EST_ASSERT(ORD(dim==dir::x, dim==dir::y, dim==dir::z));
    P4EST_ASSERT(l2g_size == ((int) blocksize)*(faces->num_local[dim] + faces->num_ghost[dim]));
    P4EST_ASSERT(augmented_forest->data_size%(sizeof(p4est_gloidx_t)) == 0);
    u_int n_global_indices_per_quad = augmented_forest->data_size/(sizeof(p4est_gloidx_t));
    P4EST_ASSERT(n_global_indices_per_quad == P4EST_CHILDREN + P4EST_FACES);
    p4est_gloidx_t* ghost_data = (p4est_gloidx_t*) P4EST_ALLOC(p4est_gloidx_t, n_global_indices_per_quad*ghost->ghosts.elem_count);
    p4est_ghost_exchange_data (augmented_forest, ghost, ghost_data);

    // Calculate the global face offset per proc
    std::vector<p4est_gloidx_t> global_face_offset_on_proc(augmented_forest->mpisize, 0);
    for (int r = 1; r<augmented_forest->mpisize; ++r)
      global_face_offset_on_proc[r] = global_face_offset_on_proc[r - 1] + (p4est_gloidx_t)faces->global_owned_indeps[dim][r - 1];

    for (p4est_topidx_t tree_idx = augmented_forest->first_local_tree; tree_idx <= augmented_forest->last_local_tree; ++tree_idx) {
      p4est_tree_t* tree = p4est_tree_array_index(augmented_forest->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        const p4est_quadrant_t* quadrant = (const p4est_quadrant_t*) p4est_quadrant_array_index(&tree->quadrants, q);
        p4est_gloidx_t* quad_data = (p4est_gloidx_t*) quadrant->p.user_data;
        for (u_char i = 0; i < 2; ++i) {
          p4est_locidx_t local_face_idx = faces->q2f(quad_idx, 2*dim+i);
          P4EST_ASSERT((local_face_idx >= 0 && local_face_idx < (p4est_locidx_t)(faces->num_local[dim] + faces->num_ghost[dim])) || local_face_idx == NO_VELOCITY);
          if(local_face_idx == NO_VELOCITY)
            continue;
          if(global_indices[blocksize*local_face_idx] == -1)
          {
            for (u_int comp = 0; comp < blocksize; ++comp)
              global_indices[blocksize*local_face_idx + comp] = blocksize*quad_data[P4EST_CHILDREN + 2*dim + i] + comp;
            dof_counter++;
          }
        }
      }
    }
    for (size_t q = 0; q < ghost->ghosts.elem_count; ++q) {
      for (u_char i = 0; i < 2; ++i) {
        p4est_locidx_t local_face_idx = faces->q2f(q+augmented_forest->local_num_quadrants, 2*dim + i);
        P4EST_ASSERT((local_face_idx >= 0 && local_face_idx < (p4est_locidx_t)(faces->num_local[dim] + faces->num_ghost[dim])) || local_face_idx == NO_VELOCITY);
        if(local_face_idx == NO_VELOCITY)
          continue;
        if(global_indices[blocksize*local_face_idx] == -1)
        {
          for (u_int comp = 0; comp < blocksize; ++comp)
            global_indices[blocksize*local_face_idx + comp] = blocksize*ghost_data[q*n_global_indices_per_quad + P4EST_CHILDREN + 2*dim + i] + comp;
          dof_counter++;
        }
      }
    }
    P4EST_FREE(ghost_data);
    break;
  }
  default:
    throw std::runtime_error("VecScatterCreateChangeLayout: unknown data type, only cell-, node- or face-sampled values are handled...");
    break;
  }
  P4EST_ASSERT(((int) blocksize)*dof_counter == l2g_size);

  ierr = ISCreateGeneral(augmented_forest->mpicomm, petsc_int_l2g_size, global_indices.data(), PETSC_COPY_VALUES, &index_set_from_disk); CHKERRXX(ierr);

  Vec in_memory_local;
  ierr = VecGhostGetLocalForm(in_memory, &in_memory_local); CHKERRXX(ierr);
  if(ctx != NULL)
  {
    ierr = VecScatterDestroy(ctx); CHKERRXX(ierr);
  }
  ierr = VecScatterCreate(from_disk, index_set_from_disk, in_memory_local, index_set_in_memory_local, &ctx); CHKERRXX(ierr);

  ierr = ISDestroy(index_set_from_disk); CHKERRXX(ierr);
  ierr = ISDestroy(index_set_in_memory_local); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(in_memory, &in_memory_local); CHKERRXX(ierr);

  return ierr;
}

PetscErrorCode LoadVec(const char fname[], u_int n_vecs, Vec *x, const int dtype,
                       p4est_t* augmented_forest, p4est_ghost_t* ghost, const p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                       PetscBool skippheader, PetscBool usempiio)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscViewerCreate(augmented_forest->mpicomm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
  if (skippheader) { ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);CHKERRQ(ierr);
  if (usempiio) { ierr = PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);

  if(dtype == FACE_DATA && n_vecs%P4EST_DIM!=0)
    throw std::invalid_argument("LoadVec: the number of vectors to be loaded must be proportional to P4EST_DIM if loading FACE_DATA.");

  for (u_int kk = 0; kk < n_vecs; ++kk) {
    Vec vector_as_loaded;
    ierr = VecCreate(augmented_forest->mpicomm, &vector_as_loaded);CHKERRQ(ierr);
    ierr = VecLoad(vector_as_loaded, viewer);CHKERRQ(ierr);
    PetscInt expected_global_size, global_size;
    ierr = VecGetSize(vector_as_loaded, &expected_global_size); CHKERRQ(ierr);
    ierr = VecGetSize(x[kk], &global_size); CHKERRQ(ierr);
    if(global_size != expected_global_size){
      printf("Global size = %d, expected global size = %d \n",global_size,expected_global_size);
      throw std::invalid_argument("LoadVec: the passed vector and the vector on disk are not of the same size");}
    VecScatter ctx; ctx = NULL;
    ierr = VecScatterCreateChangeLayout(vector_as_loaded, x[kk], dtype, augmented_forest, ghost, nodes, faces, kk%P4EST_DIM, ctx); CHKERRQ(ierr);
    ierr = VecGhostChangeLayoutBegin(ctx, vector_as_loaded, x[kk]); CHKERRQ(ierr);
    ierr = VecGhostChangeLayoutEnd(ctx, vector_as_loaded, x[kk]); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x[kk]); CHKERRQ(ierr);
    ierr = VecDestroy(vector_as_loaded); CHKERRQ(ierr);
    ierr = VecScatterDestroy(ctx); CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  return ierr;
}

void my_p4est_save_forest(const char* absolute_path_to_file, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces)
{
  SC_CHECK_ABORTF((forest->data_size == 0), "my_p4est_save_forest: this function assumes that the p4est object has no cell-associated data, aborting...");
  void* user_pointer_to_restore = forest->user_pointer;

  // node offset in global ordering per proc (compute it once and pass it in the pointer to avoid repeated long calculations on large numbers of procs)
  std::vector<p4est_gloidx_t> global_node_offsets(forest->mpisize, 0);
  for (int r = 1; r<forest->mpisize; ++r)
    global_node_offsets[r] = global_node_offsets[r-1] + nodes->global_owned_indeps[r-1];

  std::vector<p4est_gloidx_t> global_face_offsets[P4EST_DIM];
  if(faces != NULL)
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      global_face_offsets[dir].resize(forest->mpisize, 0);
      for (int r = 1; r < forest->mpisize; ++r)
        global_face_offsets[dir][r] = global_face_offsets[dir][r-1] + faces->global_owned_indeps[dir][r-1];
    }

  pointers_to_grid_data_info ptr;
  ptr.nodes                 = nodes;
  ptr.faces                 = faces;
  ptr.global_node_offsets   = &global_node_offsets;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    ptr.global_face_offsets[dir] = &global_face_offsets[dir];

  p4est_reset_data(forest, ((sizeof(p4est_gloidx_t))*(P4EST_CHILDREN + ((faces == NULL)? 0 : P4EST_FACES))), set_quadrant_data_for_exportation, ((void*) &ptr));
  p4est_save_ext(absolute_path_to_file, forest, P4EST_TRUE, P4EST_TRUE); // save data, we'll need the cell data to scatter the data on the correct (new) degrees of freedom when the data is loaded back from disk
  p4est_reset_data(forest, 0, NULL, user_pointer_to_restore); // delete the quadrant-level data created for this specific purpose; and restore the user_pointer as it was beforehand
}

void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                                   const char* forest_filename, const vector<save_or_load_element_t>& elements)
{
  if(create_directory(absolute_path_to_folder, forest->mpirank, forest->mpicomm) != 0)
#ifdef CASL_THROWS
    throw std::invalid_argument("my_p4est_save_forest_and_data: invalid path to exportation folder");
#else
    return; // impossible to create the directory but we don't want to throw...
#endif

  char absolute_path_to_file[PATH_MAX];

  // save the p4est object with global order of dofs saved as cell-level data to recover correct data localization at reload stage
  sprintf(absolute_path_to_file, "%s/%s", absolute_path_to_folder, forest_filename);
  my_p4est_save_forest(absolute_path_to_file, forest, nodes, faces);

  // export the Petsc vectors
  for (size_t k = 0; k < elements.size(); k++) {
    /* get the name of the vector(s) for the export */
    sprintf(absolute_path_to_file, "%s/%s.petscbin", absolute_path_to_folder, elements[k].name.c_str());
    PetscErrorCode ierr = VecDump(absolute_path_to_file, elements[k].nvecs, elements[k].pointer_to_vecs); CHKERRXX(ierr);
  }
}

void my_p4est_save_forest_and_data_v(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
									 const char* forest_filename, u_int num_exports, va_list args)
{
  vector<save_or_load_element_t> elements;
  for (u_int i = 0; i < num_exports; ++i) {
	save_or_load_element_t to_add;
	to_add.name            = std::string(va_arg(args, const char*));
	to_add.nvecs           = va_arg(args, u_int);
	to_add.pointer_to_vecs = va_arg(args, Vec*);
	elements.push_back(to_add);
  }

  my_p4est_save_forest_and_data(absolute_path_to_folder, forest, nodes, faces, forest_filename, elements);
}

void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                                   const char* forest_filename, u_int num_exports, ...)
{
  va_list args;
  va_start(args, num_exports);
  my_p4est_save_forest_and_data_v(absolute_path_to_folder, forest, nodes, faces, forest_filename, num_exports, args);
  va_end(args);
}

void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes,
                                   const char* forest_filename, u_int num_exports, ...)
{
  va_list args;
  va_start(args, num_exports);
  my_p4est_save_forest_and_data_v(absolute_path_to_folder, forest, nodes, nullptr, forest_filename, num_exports, args);
  va_end(args);
}

void my_p4est_load_forest_and_data(const MPI_Comm& mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t& expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const p4est_bool_t& retrieve_brick, my_p4est_brick_t* &brick,
                                   const p4est_bool_t& create_faces_hierarchy_and_cell_neighbors, my_p4est_faces_t* &faces, my_p4est_hierarchy_t* &hierarchy, my_p4est_cell_neighbors_t* &ngbd_c,
                                   const char* forest_filename, const vector<save_or_load_element_t>& elements, const double& cfl)
{
  char absolute_path_to_file[PATH_MAX];

  // load the p4est object and the connectivity
  if(forest != NULL)
    p4est_destroy(forest);
  if(conn != NULL)
    p4est_connectivity_destroy(conn);
  sprintf(absolute_path_to_file, "%s/%s", absolute_path_to_folder, forest_filename);
  if(!file_exists(absolute_path_to_file))
    throw std::invalid_argument("my_p4est_load_forest_and_data: the p4est file can't be found on disk...");
  // load the forest WITH data to recover correct dof's ordering...
  forest = p4est_load_ext(absolute_path_to_file, mpi_comm, (P4EST_CHILDREN + (create_faces_hierarchy_and_cell_neighbors? P4EST_FACES : 0))*sizeof(p4est_gloidx_t), P4EST_TRUE, P4EST_TRUE, P4EST_TRUE, NULL, &conn);

  // create the ghosts
  if(ghost != NULL)
    p4est_ghost_destroy(ghost);
  ghost = my_p4est_ghost_new(forest, P4EST_CONNECT_FULL);
  if(expand_ghost)
  {
	if(cfl <= 0)
	  throw std::invalid_argument("my_p4est_load_forest_and_data: cfl constant must be strictly positive...");
    my_p4est_ghost_expand(forest, ghost);
    // if expanding ghost, the targeted usage is very likely to be for Navier-Stokes application
    // --> check if the aspect ratio of the cells is unconventional which may enforce a requirement
    // for further expansion of ghost layers
    double tree_dimensions[P4EST_DIM];
    p4est_topidx_t v_m  = conn->tree_to_vertex[0];                  // index of the front lower left corner of the first tree in the macro-mesh
    p4est_topidx_t v_p  = conn->tree_to_vertex[P4EST_CHILDREN - 1]; // index of the back upper right corner of the first tree in the macro-mesh
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      tree_dimensions[dir] = conn->vertices[3*v_p + dir] - conn->vertices[3*v_m + dir];
	int n_ghost_addtnl_expansions = cfl > 1? (int)ceil(cfl - 1) : (int)third_degree_ghost_are_required(tree_dimensions);
	for(int i = 0; i < n_ghost_addtnl_expansions; i++)
      my_p4est_ghost_expand(forest, ghost);
  }

  // create the nodes
  if(nodes != NULL)
    p4est_nodes_destroy(nodes);
  nodes = my_p4est_nodes_new(forest, ghost);

  // retrieve the brick, if required
  if(retrieve_brick)
  {
    if(brick != NULL)
    {
      if(brick->nxyz_to_treeid != NULL)
        P4EST_FREE(brick->nxyz_to_treeid);
      delete brick;
    }
    brick = my_p4est_recover_brick(conn);
  }

  if(create_faces_hierarchy_and_cell_neighbors)
  {
    if(hierarchy != NULL)
      delete hierarchy;
    hierarchy = new my_p4est_hierarchy_t(forest, ghost, brick);
    if(ngbd_c != NULL)
      delete ngbd_c;
    ngbd_c = new my_p4est_cell_neighbors_t(hierarchy);
    if(faces != NULL)
      delete faces;
    // the construction of faces requires forest->user_pointer to point to a valid splitting_criteria_t object,
    // with a correctly defined max_lvl.
    int8_t max_lvl = find_max_level(forest);
    splitting_criteria_t tmp_sp(0, max_lvl); // only max_lvl is relevant for the creation of faces
    forest->user_pointer = &tmp_sp; // forest->user_pointer was set to NULL at load stage (see above) --> no risk of leak here
    faces = new my_p4est_faces_t(forest, ghost, brick, ngbd_c);
  }

  // load the Petsc vectors and redistribute them accordingly
  PetscErrorCode ierr;
  for (size_t nn = 0; nn < elements.size(); nn++) {
    /* get the on-disk filename for the load */
    sprintf(absolute_path_to_file, "%s/%s.petscbin", absolute_path_to_folder, elements[nn].name.c_str());
    if(!file_exists(absolute_path_to_file))
      throw std::invalid_argument("my_p4est_load_forest_and_data: a petsc file can't be found on disk...");
    switch (elements[nn].DATA_SAMPLING) {
    case NODE_DATA:
    {
      for (u_int i = 0; i < elements[nn].nvecs; ++i) {
        ierr = VecCreateGhostNodes(forest, nodes, &elements[nn].pointer_to_vecs[i]); CHKERRXX(ierr);
      }
      break;
    }
    case CELL_DATA:
    {
      for (u_int i = 0; i < elements[nn].nvecs; ++i) {
        ierr = VecCreateGhostCells(forest, ghost, &elements[nn].pointer_to_vecs[i]); CHKERRXX(ierr);
      }
      break;
    }
    case FACE_DATA:
    {
      P4EST_ASSERT(elements[nn].nvecs%P4EST_DIM == 0);
      for (u_int i = 0; i < elements[nn].nvecs/P4EST_DIM; ++i) {
        for (u_char k = 0; k < P4EST_DIM; ++k) {
          ierr = VecCreateGhostFaces(forest, faces, &elements[nn].pointer_to_vecs[P4EST_DIM*i + k], k); CHKERRXX(ierr);
        }
      }
      break;
    }
    case NODE_BLOCK_VECTOR_DATA:
    {
      for (u_int i = 0; i < elements[nn].nvecs; ++i) {
        ierr = VecCreateGhostNodesBlock(forest, nodes, P4EST_DIM, &elements[nn].pointer_to_vecs[i]); CHKERRXX(ierr);
      }
      break;
    }
    default:
    {
      throw std::invalid_argument("my_p4est_load_forest_and_data: unknown data type. Use either NODE_DATA, CELL_DATA, FACE_DATA, NODE_BLOCK_VECTOR_DATA...");
      break;
    }
    }

    PetscErrorCode ierr = LoadVec(absolute_path_to_file, elements[nn].nvecs, elements[nn].pointer_to_vecs, elements[nn].DATA_SAMPLING, forest, ghost, nodes, faces); CHKERRXX(ierr);
  }
  p4est_reset_data(forest, 0, NULL, NULL);
}

void my_p4est_load_forest_and_data(const MPI_Comm& mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t& expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const p4est_bool_t& retrieve_brick, my_p4est_brick_t* &brick,
                                   const p4est_bool_t& create_faces_hierarchy_and_cell_neighbors, my_p4est_faces_t* &faces, my_p4est_hierarchy_t* &hierarchy, my_p4est_cell_neighbors_t* &ngbd_c,
                                   const char* forest_filename, u_int num_loads, ...)
{
  va_list args;
  va_start(args, num_loads);
  vector<save_or_load_element_t> elements;
  for (u_int i = 0; i < num_loads; ++i) {
    save_or_load_element_t to_add;
    to_add.name             = std::string(va_arg(args, const char*));
    to_add.DATA_SAMPLING    = va_arg(args, int);
    to_add.nvecs            = va_arg(args, u_int);
    to_add.pointer_to_vecs  = va_arg(args, Vec*);
    elements.push_back(to_add);
  }
  va_end(args);

  my_p4est_load_forest_and_data(mpi_comm, absolute_path_to_folder, forest, conn,
                                expand_ghost, ghost, nodes,
                                retrieve_brick, brick, create_faces_hierarchy_and_cell_neighbors, faces, hierarchy, ngbd_c,
                                forest_filename, elements);
}

void my_p4est_load_forest_and_data(const MPI_Comm& mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
								   const p4est_bool_t& expand_ghost, const double& cfl, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
								   const p4est_bool_t& retrieve_brick, my_p4est_brick_t* &brick,
								   const p4est_bool_t& create_faces_hierarchy_and_cell_neighbors, my_p4est_faces_t* &faces, my_p4est_hierarchy_t* &hierarchy, my_p4est_cell_neighbors_t* &ngbd_c,
								   const char* forest_filename, u_int num_loads, ...)
{
	va_list args;
	va_start(args, num_loads);
	vector<save_or_load_element_t> elements;
	for (u_int i = 0; i < num_loads; ++i) {
		save_or_load_element_t to_add;
		to_add.name             = std::string(va_arg(args, const char*));
		to_add.DATA_SAMPLING    = va_arg(args, int);
		to_add.nvecs            = va_arg(args, u_int);
		to_add.pointer_to_vecs  = va_arg(args, Vec*);
		elements.push_back(to_add);
	}
	va_end(args);

	my_p4est_load_forest_and_data(mpi_comm, absolute_path_to_folder, forest, conn,
								  expand_ghost, ghost, nodes,
								  retrieve_brick, brick, create_faces_hierarchy_and_cell_neighbors, faces, hierarchy, ngbd_c,
								  forest_filename, elements, cfl);
}

void my_p4est_load_forest_and_data(const MPI_Comm& mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t& expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const char* forest_filename, const std::vector<save_or_load_element_t>& elements, const double& cfl)
{
  // since we are not dealing with faces for this case, all the following need to be set to NULL
  my_p4est_brick_t* brick = nullptr;
  my_p4est_faces_t* faces = nullptr;
  my_p4est_hierarchy_t* hierarchy = nullptr;
  my_p4est_cell_neighbors_t* ngbd_c = nullptr;

  my_p4est_load_forest_and_data(mpi_comm, absolute_path_to_folder, forest, conn, expand_ghost, ghost, nodes,
                                P4EST_FALSE, brick, P4EST_FALSE, faces, hierarchy, ngbd_c,
                                forest_filename, elements, cfl);
}

void my_p4est_load_forest_and_data(const MPI_Comm mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const char* forest_filename, u_int num_loads, ...)
{
  va_list args;
  va_start(args, num_loads);
  vector<save_or_load_element_t> elements;
  for (u_int i = 0; i < num_loads; ++i) {
    save_or_load_element_t to_add;
    to_add.name             = std::string(va_arg(args, const char*));
    to_add.DATA_SAMPLING    = va_arg(args, int);
    to_add.nvecs            = va_arg(args, u_int);
    to_add.pointer_to_vecs  = va_arg(args, Vec*);
    elements.push_back(to_add);
  }
  va_end(args);

  my_p4est_load_forest_and_data(mpi_comm, absolute_path_to_folder, forest, conn, expand_ghost, ghost, nodes, forest_filename, elements);
}

void my_p4est_load_forest_and_data(const MPI_Comm& mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
								   const p4est_bool_t& expand_ghost, const double& cfl, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
								   const char* forest_filename, u_int num_loads, ...)
{
	va_list args;
	va_start(args, num_loads);
	vector<save_or_load_element_t> elements;
	for (u_int i = 0; i < num_loads; ++i) {
		save_or_load_element_t to_add;
		to_add.name             = std::string(va_arg(args, const char*));
		to_add.DATA_SAMPLING    = va_arg(args, int);
		to_add.nvecs            = va_arg(args, u_int);
		to_add.pointer_to_vecs  = va_arg(args, Vec*);
		elements.push_back(to_add);
	}
	va_end(args);

	my_p4est_load_forest_and_data(mpi_comm, absolute_path_to_folder, forest, conn, expand_ghost, ghost, nodes, forest_filename, elements, cfl);
}

my_p4est_brick_t* my_p4est_recover_brick(const p4est_connectivity_t* connectivity)
{
  my_p4est_brick_t* brick = new my_p4est_brick_t;
  brick->nxyz_to_treeid = P4EST_ALLOC(p4est_topidx_t, connectivity->num_trees);

  const p4est_topidx_t first_tree = 0;
  const p4est_topidx_t last_tree = connectivity->num_trees-1;
  const p4est_topidx_t* t2v = connectivity->tree_to_vertex;
  const double* v2c = connectivity->vertices;

  for (u_char i = 0; i < 3; ++i) {
    brick->xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree+P4EST_CHILDREN-1] + i];
    brick->xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree+0] + i];
  }
  // initialize the numbers of trees along each dimension
  brick->nxyztrees[0] = 1;
  brick->nxyztrees[1] = 1;
  brick->nxyztrees[2] = 1;
  if(connectivity->num_trees > 1)
  {
    double min_tree_dimensions[3];
    for (u_char i = 0; i < 3; ++i)
      min_tree_dimensions[i] = brick->xyz_max[i] - brick->xyz_min[i];
    // get the minimum possible tree dimension along each direction (for robust double comparisons later on)
    for (p4est_topidx_t tt = 0; tt < connectivity->num_trees; ++tt)
      for (u_char i = 0; i < 3; ++i)
        min_tree_dimensions[i] = MIN(min_tree_dimensions[i], v2c[3*t2v[P4EST_CHILDREN*tt+P4EST_CHILDREN-1]+i]-v2c[3*t2v[P4EST_CHILDREN*tt+0]+i]);
    std::vector<double> first_vertices_of_trees[3];
    for (u_char i = 0; i < 3; ++i)
      first_vertices_of_trees[i].resize(0);
    for (p4est_topidx_t tt = 0; tt < connectivity->num_trees; ++tt) {
      for (u_char i = 0; i < 3; ++i) {
        bool is_not_in_yet = true;
        for (size_t k = 0; k < first_vertices_of_trees[i].size(); ++k) {
          is_not_in_yet = is_not_in_yet && (fabs(v2c[3*t2v[P4EST_CHILDREN*tt+0]+i] - first_vertices_of_trees[i][k]) > 0.1*min_tree_dimensions[i]);
          if(!is_not_in_yet)
            break;
        }
        if(is_not_in_yet)
          first_vertices_of_trees[i].push_back(v2c[3*t2v[P4EST_CHILDREN*tt+0]+i]);
      }
    }
    for (u_char i = 0; i < 3; ++i)
    {
      std::sort(first_vertices_of_trees[i].begin(), first_vertices_of_trees[i].end());
      brick->nxyztrees[i] = first_vertices_of_trees[i].size();
    }
#ifndef P4_TO_P8
    P4EST_ASSERT(brick->nxyztrees[2] == 1);
#endif
    for (p4est_topidx_t tt = 0; tt < connectivity->num_trees; ++tt) {
      int cartesian_tree_idx[3];
      for (u_char i = 0; i < 3; ++i) {
        if(first_vertices_of_trees[i].size() == 1)
          cartesian_tree_idx[i] = 0; // safer in 3D otherwise we compare 0 to 0, risky...
        else
          for (cartesian_tree_idx[i] = 0; fabs(v2c[3*t2v[P4EST_CHILDREN*tt+0]+i] - first_vertices_of_trees[i][cartesian_tree_idx[i]]) > 0.1*min_tree_dimensions[i]; ++cartesian_tree_idx[i]) {}
      }
      brick->nxyz_to_treeid[brick->nxyztrees[0]*brick->nxyztrees[1]*cartesian_tree_idx[2] + brick->nxyztrees[0]*cartesian_tree_idx[1] + cartesian_tree_idx[0]] = tt;
    }
  }
  else // only one tree, not much to worry about...
    brick->nxyz_to_treeid[0] = 0;
  P4EST_ASSERT(connectivity->num_trees == brick->nxyztrees[0]*brick->nxyztrees[1]*brick->nxyztrees[2]);
  return brick;
}
