#ifndef MY_P4EST_SAVE_LOAD_H
#define MY_P4EST_SAVE_LOAD_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_faces.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_faces.h>
#endif

static const int NODE_DATA = 13846;
static const int CELL_DATA = 13847;
static const int FACE_DATA = 13848;
static const int NODE_BLOCK_VECTOR_DATA = 13849;

typedef enum
{
  SAVE=3541,
  LOAD
} save_or_load;

typedef struct
{
  std::string name;
  int DATA_SAMPLING;
  uint nvecs;
  Vec* pointer_to_vecs;
} save_or_load_element_t;

/*!
 * \brief The pointers_to_grid_data_info struct is what the user_pointer of the p4est object needs to point to
 *        during the creation of cell-associated data that are exported on disk WITH the p4est object
 * (those data are required afterwards for clean, accurate, partition-independent scatter of vector components)
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
struct pointers_to_grid_data_info
{
  p4est_nodes_t* nodes;
  const my_p4est_faces_t* faces;
  const std::vector<p4est_gloidx_t>* global_node_offsets;
  const std::vector<p4est_gloidx_t>* global_face_offsets[P4EST_DIM];
};

/*!
 * \brief VecDump exports n_vecs vectors pointed by x in a BINARY format
 * \param fname       [in]  name of the file to export the vectors x in (absolute path)
 * \param n_vecs      [in]  number of vectors to be exported
 * \param x           [in]  pointer to the vectors to be exported in the binary file
 * \param skippheader [in]  flag controlling whether the header is printed or not. If activated,
 *                          header information (i.e. information about the global and local size
 *                          of the vector) won't be exported. Default is PETSC_FALSE
 * \param usempiio    [in]  flag activating MPI-IO for reading/writing. Default is PETSC_TRUE
 * \return a regular Petsc Error Code.
 * [NOTE:] modified from the function taken from src/dm/examples/tutorials/ex15.c in the Petsc distribution
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
PetscErrorCode VecDump(const char fname[], u_int n_vecs, const Vec *x, PetscBool skippheader=PETSC_FALSE, PetscBool usempiio=PETSC_TRUE);

/*!
 * \brief VecScatterCreateChangeLayout: creates a Petsc scatter-environment object to redistribute vector components loaded
 *        from disk consistently with the new global ordering that might have been changed when reloading the p4est with a
 *        different partition (for instance, with a different number of procs)
 * \param from_disk         [in]    PETSc vector loaded from disk
 * \param in_memory         [in]    (New) PETSc vector created on the same p4est object as the from_disk vector, but on a supposedly
 *                                  different partition (new global ordering)
 * \param dtype             [in]    data sampling dype: either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA
 * \param augmented_forest  [in]    p4est object, as loaded from disk but possibly with a new number of processors than when exported.
 *                                  The quadrants of this p4est are supposed to point to cell-associated data as created in the
 *                                  my_p4est_save_forest function in order to recover the original global ordering of all vector
 *                                  components (hence the "augmented_").
 * \param ghost             [in]    ghost cells associated with the augmented_forest object
 * \param nodes             [in]    nodes associated with the augmented_forest object
 * \param faces             [in]    faces associated with the augmented_forest object
 * \param dim               [in]    the orientation of the face-normal (i.e. dir::x, dir::y or dir::z) (relevant for FACE_DATA only)
 * \param ctx               [inout] a reference for the created Petsc scatter-environment.
 * \return a regular Petsc Error code for success/failure test
 * [NOTE 0:]  the global ordering is unchanged only for CELL_DATA, since the quadrants are always stored in increasing Morton index.
 *            the global orderings of nodes and faces are partition-dependent, hence the need to export the original global ordering
 * [NOTE 1:]  from_disk and in_memory MUST have the same global size (by definition)
 * [NOTE 2:]  the ghost structure is required for all types of data (i.e. dtype == NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA OR FACE_DATA)
 * [NOTE 3:]  the nodes structure is required only for the NODE_DATA and NODE_BLOCK_VECTOR_DATA type (the pointer can be NULL otherwise)
 * [NOTE 4:]  the faces structure and dim are required only for the FACE_DATA type (the pointer can be NULL and dim can be any value, otherwise)
 * [NOTE 5:]  if not NULL on the input, ctx is destroyed before a new VecScatter is created to avoid memory leak --> ctx must be a valid reference or NULL
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
PetscErrorCode VecScatterCreateChangeLayout(Vec from_disk, Vec in_memory, const int dtype,
                                            p4est_t *augmented_forest, p4est_ghost_t* ghost, const p4est_nodes_t* nodes, const my_p4est_faces_t* faces, const int dim,
                                            VecScatter& ctx);

/*!
 * \brief LoadVec loads vectors from a binary Petsc-exported file on disk referred by path fname
 * \param fname             [in]    path to the binary Petsc-exported file on disk
 * \param n_vecs            [in]    number of vectors to be loaded
 * \param x                 [inout] pointer to a Vec object or array of Vec objects to be filled with the data loaded from disk
 *                                  (in new global ordering, i.e., _after_ appropriate scattering of data)
 * \param dtype             [in]    data sampling dype: either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA
 * \param augmented_forest  [in]    p4est object, as loaded from disk but possibly with a new number of processors than when exported.
 *                                  The quadrants of this p4est are supposed to point to cell-associated data as created in the
 *                                  my_p4est_save_forest function in order to recover the original global ordering of all vector
 *                                  components (hence the "augmented_").
 * \param ghost             [in]    ghost cells associated with the augmented_forest object
 * \param nodes             [in]    nodes associated with the augmented_forest object
 * \param faces             [in]    faces associated with the augmented_forest object
 * \param skippheader       [in]    flag controlling whether the header is read or not. If activated, header information (i.e.,
 *                                  information about the global and local size of the vector) won't be read
 * \param usempiio          [in]    flag activating MPI-IO for reading/writing. Default is PETSC_TRUE
 * \return a regular Petsc Error Code.
 * [NOTE:]  the vectors passed by the user must be of the same (global) size as the ones loaded from file... if not, this function
 *          throws an std::invalid_argument exception...
 * [NOTE:]  the function expects the number of vectors to load to be proportional to P4EST_DIM if dtype is FACE_DATA and will load the
 *          vectors in looping order dir::x, dir::y (, dir::z), dir::x, dir::y (, dir::z), dir::x, dir::y (, dir::z), ...
 * [NOTE:]  faces can be NULL if no FACE_DATA is loaded;
 * [NOTE:]  nodes can be NULL if no NODE_DATA or NODE_BLOCK_VECTOR_DATA is loaded;
 * [NOTE:]  modified from the function taken from src/dm/examples/tutorials/ex15.c in the Petsc distribution
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
PetscErrorCode LoadVec(const char fname[], u_int n_vecs, Vec *x, const int dtype,
                       p4est_t* augmented_forest, p4est_ghost_t* ghost, const p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                       PetscBool skippheader=PETSC_FALSE, PetscBool usempiio=PETSC_TRUE);

/*!
 * \brief my_p4est_save_forest saves a p4est object ALONG WITH CELL-ASSOCIATED DATA that hold the global indices for nodes and, possibly,
 *        faces. For every quadrant of local index quad, the corresponding quadrant's user-data p.user_data is made to point to an array
 *        of P4EST_CHILDREN p4est_gloidx_t values that are the global indices of the nodes of local indices
 *                                      nodes->local_nodes[p4EST_CHILDREN*quad+0...P4EST_CHILDREN-1]
 *        and, if faces are provided as well, P4EST_FACES more additional p4est_gloidx_t that are the global indices of the faces of local
 *        indices
 *                                      faces->q2f(quad_idx, 0...P4EST_FACES-1).
 *        These global indices are required in order to scatter the data correctly on a reloaded grid (possibly with a different number of
 *        procs)
 * \param absolute_path_to_file   [in]  path to the file to be created for exportation of the p4est with the data (absolute path)
 * \param forest                  [in]  pointer to the p4est object to be saved
 * \param nodes                   [in]  pointer to the p4est_nodes_t structure associated with the grid
 * \param faces                   [in]  pointer to a valid my_p4est_faces_t object or NULL if faces are irrelevant
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_save_forest(const char* absolute_path_to_file, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces);

/*!
 * \brief my_p4est_save_forest_and_data saves a p4est object (in augmented format, i.e. with cell-associated data as described in
 *        my_p4est_save_forest) AND relevant PETSc vectors
 *        WARNING:  this specific function should not be used directly, it is meant to be a basic structure for the two wrapper functions
 *                  here below, with the same name!
 * \param absolute_path_to_folder [in]  path to the folder where the relevant exportation files (for the p4est and the PETSc vectors)
 *                                      will be created and stored (absolute path)
 * \param forest                  [in]  pointer to the p4est object to be saved
 * \param nodes                   [in]  pointer to the p4est_nodes_t structure associated with the grid
 * \param faces                   [in]  pointer to a valid my_p4est_faces_t object or NULL if faces are irrelevant
 * \param forest_filename         [in]  name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param elements                [in]  vector of elements (PETSc vector(s)) to be exported into the the exportation folder
 *                                      Every element of the above is a struct of type save_or_load_element_t which contains:
 *                                      1) std::string name: the name of the file in which the (grouped) PETSc vector(s) will be exported (within the folder)
 *                                      [2) (--> irrelevant for exportation <--) int DATA_SAMPLING: type of data-sampling, either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA]
 *                                      3) u_int nvecs: the number of PETSc vector(s) within the (grouped) load (greater than or equal to 1)
 *                                      4) Vec* pointer_to_vecs: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be exported
 * [NOTE 1:]  faces MUST point toward a valid my_p4est_faces object if exporting face-sampled data.
 * [NOTE 2:]  if only node- and/or cell-sampled data are exported and reloaded afterwards, i.e. if no face-sampled data is relevant
 *            for exportation/reload, DO NOT pass faces as it would trigger the creation of unnecessary data...
 * [NOTE 3:]  this function throws an std::invalid_argument expection in CASL_THROWS mode if the folder path is invalid
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                                   const char* forest_filename, const vector<save_or_load_element_t>& elements);

/*!
 * Function to create a list of elements to be saved.  Basically, you can't forward variadic parameters from one function
 * to another.  You must have a function that takes in a va_list.  This is such function.
 * https://stackoverflow.com/questions/3530771/passing-variable-arguments-to-another-function-that-accepts-a-variable-argument
 * @see any of my_p4est_save_forest_and_data functions for a parameter description.
 */
void my_p4est_save_forest_and_data_v(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
									 const char* forest_filename, u_int num_exports, va_list args);

/*!
 * \brief my_p4est_save_forest_and_data saves a p4est object (in augmented format, i.e. with cell-associated data as described in
 *        my_p4est_save_forest) AND relevant (groups of) PETSc vectors
 * \param absolute_path_to_folder [in]  path to the folder where the relevant exportation files (for the p4est and the PETSc vectors)
 *                                      will be created and stored (absolute path)
 * \param forest                  [in]  pointer to the p4est object to be saved
 * \param nodes                   [in]  pointer to the p4est_nodes_t structure associated with the grid
 * \param faces                   [in]  pointer to a valid my_p4est_faces_t object
 * \param forest_filename         [in]  name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param num_exports             [in]  number of (grouped) exportations of PETSc vector(s) in the exportation folder
 * \param variable_list_of_args   [in]  variable list of arguments configuring the (grouped) PETSc exportations, structured in the following way:
 *                                      for each (grouped) exportation of PETSc vector(s), the function expetcs
 *                                      1) const char* filename: the name of the file in which the (grouped) PETSc vector(s) will be exported (within the exportation folder)
 *                                      2) u_int nvecs: the number of PETSc vector(s) within the (grouped) exportation (greater than or equal to 1)
 *                                      3) Vec* x: a pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be exported
 * [NOTE 1:]  faces MUST point toward a valid my_p4est_faces object!
 * [NOTE 2:]  this function throws an std::invalid_argument expection in CASL_THROWS mode if the folder path is invalid
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes, const my_p4est_faces_t* faces,
                                   const char* forest_filename, u_int num_exports, ...);

/*!
 * \brief my_p4est_save_forest_and_data saves a p4est object (in augmented format, i.e. with cell-associated data as described in
 *        my_p4est_save_forest) AND relevant PETSc vectors
 * \param absolute_path_to_folder [in]  path to the folder where the relevant exportation files (for the p4est and the PETSc vectors)
 *                                      will be created and stored (absolute path)
 * \param forest                  [in]  pointer to the p4est object to be saved
 * \param nodes                   [in]  pointer to the p4est_nodes_t structure associated with the grid
 * \param forest_filename         [in]  name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param num_exports             [in]  number of (grouped) exportations of PETSc vector(s) in the exportation folder
 * \param args                    [in]  variable list of arguments configuring the (grouped) exportations, structured in the following way:
 *                                      for each (grouped) exportation of PETSc vector(s), the function expetcs
 *                                      1) const char* filename: the name of the file in which the (grouped) PETSc vector(s) will be exported (within the folder)
 *                                      2) u_int nvecs: the number of PETSc vector(s) within the (grouped) exportation (greater than or equal to 1)
 *                                      3) Vec* x: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be exported
 * [NOTE 1:]  use this function if faces are irrelevant to your application (no face-sampled data)
 * [NOTE 2:]  this function throws an std::invalid_argument expection in CASL_THROWS mode if the folder path is invalid
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_save_forest_and_data(const char* absolute_path_to_folder, p4est_t* forest, p4est_nodes_t* nodes,
                                   const char* forest_filename, u_int num_exports, ...);

/*!
 * \brief my_p4est_load_forest_and_data loads an (augmented) p4est object from a file on-disk and PETSc vector(s) from files on disks. The vectors
 *        are automatically rescattered to the new grid partition (the p4est object can be reloaded with a different number of procs than when exported).
 *        WARNING:  this specific function should not be used directly, it is meant to be a basic structure for the two wrapper functions here below,
 *                  with the same name!
 * \param mpi_comm                [in]    MPI_Comm to which the newly loaded objects belong
 * \param absolute_path_to_folder [in]    path to the folder where the relevant exportation files (for the p4est and the PETSc vectors) have been stored (absolute path)
 * \param forest                  [inout] pointer to the loaded p4est object (the formerly pointed p4est is destroyed if not NULL on input)
 * \param conn                    [inout] pointer to the loaded connectivity (the formerly pointed connectivity is destroyed if not NULL on input)
 * \param expand_ghost            [in]    flag activating the expansion of the ghost layer if P4EST_TRUE
 * \param ghost                   [inout] pointer to the p4est_ghost_t object associated with the newly loaded and partitioned p4est object. The ghost are created using
 *                                        the P4EST_CONNECT_FULL protocol, and are expanded once if expand_ghost is P4EST_TRUE (the formerly pointed object is
 *                                        destroyed if not NULL on input)
 * \param nodes                   [inout] pointer to the p4est_nodes_t object associated with the newly loaded and partitioned p4est object and the (newly created)
 *                                        p4est_ghost_t objects (the formerly pointed object is destroyed if not NULL on input)
 * \param retrieve_brick          [in]    flag activating the brick reconstruction if P4EST_TRUE
 * \param brick                   [inout] pointer to the my_p4est_brick_t object associated with the newly loaded connectivity. The brick object is actually RECONSTRUCTED
 *                                        from the connectivity information (the formerly pointed brick object is destroyed if not NULL on input)
 * \param create_faces_hierarchy_and_cell_neighbors [in]  flag activating the construction of faces, hierarchy and cell neighborhood associated with the newly loaded p4est
 *                                                        object (mandatory for loading face-sampled data)
 * \param faces                   [inout] pointer to the my_p4est_faces_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed object
 *                                        is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param hierarchy               [inout] pointer to the my_p4est_hierarchy_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed
 *                                        object is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param ngbd_c                  [inout] pointer to the my_p4est_cell_neighbors_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed
 *                                        object is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param forest_filename         [in]    name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param elements                [in]    vector of elements (PETSc vector(s)) to load from the exportation folder
 *                                        Every element of the above is a struct of type save_or_load_element_t which contains:
 *                                        1) std::string name: the name of the file from which the (grouped) PETSc vector(s) will be loaded (within the folder)
 *                                        2) int DATA_SAMPLING: type of data-sampling, either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA
 *                                        3) u_int nvecs: the number of PETSc vector(s) within the (grouped) load (greater than or equal to 1)
 *                                        4) Vec* pointer_to_vecs: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be filled with the loaded vector(s)
 * [NOTE 1:]  faces MUST be as it was when the data were exported, as it determines the size of cell-associated data chunks: if the data where exported with faces == NULL,
 *            they MUST be reloaded with faces == NULL as well, otherwise the cell-level data will be mismatched
 * [NOTE 2:]  if face-sampled data are reloaded, this function expects the corresponding number of grouped vectors to be loaded from file to be divisible by P4EST_DIM
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_load_forest_and_data(const MPI_Comm mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const p4est_bool_t retrieve_brick, my_p4est_brick_t* &brick,
                                   const p4est_bool_t create_faces_hierarchy_and_cell_neighbors, my_p4est_faces_t* &faces, my_p4est_hierarchy_t* &hierarchy, my_p4est_cell_neighbors_t* &ngbd_c,
                                   const char* forest_filename, const vector<save_or_load_element_t>& elements);

/*!
 * \brief my_p4est_load_forest_and_data loads an (augmented) p4est object from a file on-disk and PETSc vector(s) from files on disks. The vectors
 *        are automatically rescattered to the new grid partition (the p4est object can be reloaded with a different number of procs than when exported).
 * \param mpi_comm                [in]    MPI_Comm to which the newly loaded objects belong
 * \param absolute_path_to_folder [in]    path to the folder where the relevant exportation files (for the p4est and the PETSc vectors) have been stored (absolute path)
 * \param forest                  [inout] pointer to the loaded p4est object (the formerly pointed p4est is destroyed if not NULL on input)
 * \param conn                    [inout] pointer to the loaded connectivity (the formerly pointed connectivity is destroyed if not NULL on input)
 * \param expand_ghost            [in]    flag activating the expansion of the ghost layer if P4EST_TRUE
 * \param ghost                   [inout] pointer to the p4est_ghost_t object associated with the newly loaded and partitioned p4est object. The ghost are created using
 *                                        the P4EST_CONNECT_FULL protocol, and are expanded once if expand_ghost is P4EST_TRUE (the formerly pointed object is
 *                                        destroyed if not NULL on input)
 * \param nodes                   [inout] pointer to the p4est_nodes_t object associated with the newly loaded and partitioned p4est object and the (newly created)
 *                                        p4est_ghost_t objects (the formerly pointed object is destroyed if not NULL on input)
 * \param retrieve_brick          [in]    flag activating the brick reconstruction if P4EST_TRUE
 * \param brick                   [inout] pointer to the my_p4est_brick_t object associated with the newly loaded connectivity. The brick object is actually RECONSTRUCTED
 *                                        from the connectivity information (the formerly pointed brick object is destroyed if not NULL on input)
 * \param create_faces_hierarchy_and_cell_neighbors [in]  flag activating the construction of faces, hierarchy and cell neighborhood associated with the newly loaded p4est
 *                                                        object (mandatory for loading face-sampled data)
 * \param faces                   [inout] pointer to the my_p4est_faces_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed object
 *                                        is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param hierarchy               [inout] pointer to the my_p4est_hierarchy_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed
 *                                        object is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param ngbd_c                  [inout] pointer to the my_p4est_cell_neighbors_t object associated with the newly loaded and partitioned p4est object. (The formerly pointed
 *                                        object is destroyed if not NULL on input) :::: relevant only if create_faces_hierarchy_and_cell_neighbors == P4EST_TRUE
 * \param forest_filename         [in]    name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param num_loads               [in]    number of loads of grouped PETSc vector(s) from the exportation folder
 * \param variable_list_of_args   [in, in, in, inout]
 *                                        variable list of arguments configuring the (grouped) loads, structured in the following way:
 *                                        for each load of (grouped) PETSc vector(s), the function expects
 *                                        1) const char* filename: the name of the file from which the (grouped) PETSc vector(s) will be loaded (within the folder)
 *                                        2) int dtype: type of data-sampling, either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA
 *                                        3) u_int nvecs: the number of PETSc vector(s) within the (grouped) load (greater than or equal to 1)
 *                                        4) Vec* x: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be filled with the loaded vector(s)
 * [NOTE 1:]  faces MUST be as it was when the data were exported, as it determines the size of cell-associated data chunks: if the data where exported with faces == NULL,
 *            they MUST be reloaded with faces == NULL as well, otherwise the cell-level data will be mismatched.
 *            In particular, face-sampled data can be reloaded from on-disk file in a reliable way only if the p4est object and the data where exported WITH global face indices
 *            exported at the cell-level
 * [NOTE 2:]  if face-sampled data are reloaded, this function expects the corresponding number of grouped vectors to be loaded from file to be divisible by p4EST_DIM
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_load_forest_and_data(const MPI_Comm mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const p4est_bool_t retrieve_brick, my_p4est_brick_t* &brick,
                                   const p4est_bool_t create_faces_hierarchy_and_cell_neighbors, my_p4est_faces_t* &faces, my_p4est_hierarchy_t* &hierarchy, my_p4est_cell_neighbors_t* &ngbd_c,
                                   const char* forest_filename, u_int num_loads, ...);

/*!
 * \brief my_p4est_load_forest_and_data loads an (augmented) p4est object from a file on-disk and PETSc vector(s) from files on disks. The vectors
 *        are automatically rescattered to the new grid partition (the p4est object can be reloaded with a different number of procs than when exported).
 * \param mpi_comm                [in]    MPI_Comm to which the newly loaded objects belong
 * \param absolute_path_to_folder [in]    path to the folder where the relevant exportation files (for the p4est and the PETSc vectors) have been stored (absolute path)
 * \param forest                  [inout] pointer to the loaded p4est object (the formerly pointed p4est is destroyed if not NULL on input)
 * \param conn                    [inout] pointer to the loaded connectivity (the formerly pointed connectivity is destroyed if not NULL on input)
 * \param expand_ghost            [in]    flag activating the expansion of the ghost layer if P4EST_TRUE
 * \param ghost                   [inout] pointer to the p4est_ghost_t object associated with the newly loaded and partitioned p4est object. The ghost are created using
 *                                        the P4EST_CONNECT_FULL protocol, and are expanded once if expand_ghost is P4EST_TRUE (the formerly pointed object is
 *                                        destroyed if not NULL on input)
 * \param nodes                   [inout] pointer to the p4est_nodes_t object associated with the newly loaded and partitioned p4est object and the (newly created)
 *                                        p4est_ghost_t objects (the formerly pointed object is destroyed if not NULL on input)
 * \param forest_filename         [in]    name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param elements                [in]    vector of elements (PETSc vector(s)) to load from the exportation folder
 *                                        Every element of the above is a struct of type save_or_load_element_t which contains:
 *                                        1) std::string name: the name of the file from which the (grouped) PETSc vector(s) will be loaded (within the folder)
 *                                        2) int DATA_SAMPLING: type of data-sampling, either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA or FACE_DATA
 *                                        3) u_int nvecs: the number of PETSc vector(s) within the (grouped) load (greater than or equal to 1)
 *                                        4) Vec* pointer_to_vecs: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be filled with the loaded vector(s)
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_load_forest_and_data(const MPI_Comm mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const char* forest_filename, const std::vector<save_or_load_element_t>& elements);

/*!
 * \brief my_p4est_load_forest_and_data loads an (augmented) p4est object from a file on-disk and PETSc vector(s) from files on disks. The vectors
 *        are automatically rescattered to the new grid partition (the p4est object can be reloaded with a different number of procs than when exported).
 * \param mpi_comm                [in]    MPI_Comm to which the newly loaded objects belong
 * \param absolute_path_to_folder [in]    path to the folder where the relevant exportation files (for the p4est and the PETSc vectors) have been stored (absolute path)
 * \param forest                  [inout] pointer to the loaded p4est object (the formerly pointed p4est is destroyed if not NULL on input)
 * \param conn                    [inout] pointer to the loaded connectivity (the formerly pointed connectivity is destroyed if not NULL on input)
 * \param expand_ghost            [in]    flag activating the expansion of the ghost layer if P4EST_TRUE
 * \param ghost                   [inout] pointer to the p4est_ghost_t object associated with the newly loaded and partitioned p4est object. The ghost are created using
 *                                        the P4EST_CONNECT_FULL protocol, and are expanded once if expand_ghost is P4EST_TRUE (the formerly pointed object is
 *                                        destroyed if not NULL on input)
 * \param nodes                   [inout] pointer to the p4est_nodes_t object associated with the newly loaded and partitioned p4est object and the (newly created)
 *                                        p4est_ghost_t objects (the formerly pointed object is destroyed if not NULL on input)
 * \param forest_filename         [in]    name to give to the file containing the exported (augmented) p4est object within the exportation folder
 * \param num_loads               [in]    number of loads of grouped PETSc vector(s) from the exportation folder
 * \param variable_list_of_args   [in, in, in, inout]
 *                                        variable list of arguments configuring the (grouped) loads, structured in the following way:
 *                                        for each load of (grouped) PETSc vector(s), the function expects
 *                                        1) const char* filename: the name of the file from which the (grouped) PETSc vector(s) will be loaded (within the folder)
 *                                        2) int dtype: type of data-sampling, either NODE_DATA, NODE_BLOCK_VECTOR_DATA, CELL_DATA (CANNOT be FACE_DATA in this case)
 *                                        3) u_int nvecs: the number of PETSc vector(s) within the (grouped) load (greater than or equal to 1)
 *                                        4) Vec* x: pointer to the Vec object (if nvecs = 1) or the array of Vec objects (if nvecs > 1) to be filled with the loaded vector(s)
 * Developer: Raphael Egan (raphaelegan@ucsb.edu)
 */
void my_p4est_load_forest_and_data(const MPI_Comm mpi_comm, const char* absolute_path_to_folder, p4est_t* &forest, p4est_connectivity_t* &conn,
                                   const p4est_bool_t expand_ghost, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
                                   const char* forest_filename, u_int num_loads, ...);

/*!
 * \brief my_p4est_recover_brick  reconstructs a my_p4est_brick_t object based on a p4est_connectivity_t structure.
 *                                This function does NOT assume that the trees described in the connectivity are all of the same size, to possibly investigate stretched
 *                                macro-mesh in the near future...
 * \param connectivity            [in]  p4est_connectivity_t structure describing the macro-mesh of a valid p4est_t object
 * \return  a pointer towards a valid my_p4est_brick_t object, consistent with the connectivity
 */
my_p4est_brick_t* my_p4est_recover_brick(const p4est_connectivity_t* connectivity);

#endif // MY_P4EST_SAVE_LOAD_H
