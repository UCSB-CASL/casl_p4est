#ifndef UTILS_H
#define UTILS_H

#include <src/casl_math.h>
#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_nodes.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_nodes.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <src/my_p4est_refine_coarsen.h>
#endif
#include <src/petsc_logging.h>
#include "petsc_compatibility.h"

#include <petsc.h>
#include <stdexcept>
#include <sstream>
#include <vector>

#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "unknown SIZE_MAX"
#endif




// forward declaration
class my_p4est_node_neighbors_t;
struct quad_neighbor_nodes_of_node_t;

namespace dir {
/* vertices directions */
enum {
  v_mmm = 0,
  v_pmm,
  v_mpm,
  v_ppm
#ifdef P4_TO_P8
  ,v_mmp,
  v_pmp,
  v_mpp,
  v_ppp
#endif
};
/* faces directions */
enum {
  f_m00 = 0,
  f_p00,
  f_0m0,
  f_0p0
#ifdef P4_TO_P8
  ,f_00m,
  f_00p
#endif
};
/* cartesian direction */
enum {
  x = 0,
  y
#ifdef P4_TO_P8
  ,z
#endif
};
}

enum interpolation_method{
  linear,
  quadratic,
  quadratic_non_oscillatory
};

class CF_1
{
public:
  double lip, t;
  virtual double operator()(double x) const=0 ;
  virtual ~CF_1() {}
};


class CF_2
{
public:
  double lip, t;
  virtual double operator()(double x, double y) const=0 ;
  double operator()(double *xyz) const {return this->operator()(xyz[0], xyz[1]); }
  virtual ~CF_2() {}
};

class CF_3
{
public:
  double lip, t;
  virtual double operator()(double x, double y,double z) const=0 ;
  double operator()(double *xyz) const {return this->operator()(xyz[0], xyz[1], xyz[2]); }
  virtual ~CF_3() {}
};

enum {
  WALL_m00 = -1,
  WALL_p00 = -2,
  WALL_0m0 = -3,
  WALL_0p0 = -4,
  WALL_00m = -5,
  WALL_00p = -6,
  INTERFACE = -7
};

inline int WALL_idx(unsigned char oriented_cart_dir)
{
  P4EST_ASSERT(oriented_cart_dir < P4EST_FACES);
  return (-1-((int) oriented_cart_dir));
}

typedef enum {
  DIRICHLET,
  NEUMANN,
  ROBIN,
  NOINTERFACE,
  MIXED,
  IGNORE
} BoundaryConditionType;

class mixed_interface
{
public:
  virtual BoundaryConditionType mixed_type(const double xyz_[]) const=0;
  virtual ~mixed_interface() {}
};

std::ostream& operator << (std::ostream& os, BoundaryConditionType  type);
std::istream& operator >> (std::istream& is, BoundaryConditionType& type);

class WallBC2D
{
public:
  virtual BoundaryConditionType operator()( double x, double y ) const=0 ;
  virtual ~WallBC2D() = 0;
};

class WallBC3D
{
public:
  virtual BoundaryConditionType operator()( double x, double y, double z ) const=0 ;
  virtual ~WallBC3D() = 0;
};


class BoundaryConditions2D
{
private:
  const WallBC2D* WallType_;
  BoundaryConditionType InterfaceType_;
  const mixed_interface* MixedInterface;

  const CF_2 *p_WallValue;
  const CF_2 *p_InterfaceValue;

public:
  BoundaryConditions2D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    MixedInterface = NULL;
    p_InterfaceValue = NULL;
  }

  inline void setWallTypes( const WallBC2D& w )
  {
    WallType_ = &w;
  }

  inline const WallBC2D& getWallType() const
  {
    return *WallType_;
  }

  inline void setWallValues( const CF_2& v ){
    p_WallValue = &v;
  }

  inline void setInterfaceType(BoundaryConditionType bc, const mixed_interface* obj_= NULL){
    InterfaceType_ = bc;
    if(InterfaceType_ == MIXED)
    {
      if(obj_ == NULL)
        throw std::invalid_argument("BoundaryConditions2D::setInterfaceType(): if the interface type is set to MIXED, a pointer to a class of abstract type mixed_interface MUST be provided as well!");
      MixedInterface = obj_;
    }
  }

  inline void setInterfaceValue(const CF_2& in){
    p_InterfaceValue = &in;
  }

  inline const CF_2& getInterfaceValue(){
    return *p_InterfaceValue;
  }

  inline BoundaryConditionType wallType( double x, double y ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y);
  }

  inline BoundaryConditionType wallType(const double xyz_[]) const
  {
    return wallType(xyz_[0], xyz_[1]);
  }

  inline BoundaryConditionType interfaceType() const
  {
    return InterfaceType_;
  }

  inline BoundaryConditionType interfaceType(const double* xyz) const
  {
    if(InterfaceType_ != MIXED)
      return interfaceType();
    return MixedInterface->mixed_type(xyz);
  }

  inline double wallValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y);
  }

  inline double wallValue(const double xyz_[]) const
  {
    return wallValue(xyz_[0], xyz_[1]);
  }

  inline double interfaceValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y);
  }
  inline double  interfaceValue(double xyz_[]) const
  {
    return interfaceValue(xyz_[0], xyz_[1]);
  }

};

class BoundaryConditions3D
{
private:
  const WallBC3D* WallType_;
  BoundaryConditionType InterfaceType_;
  const mixed_interface* MixedInterface;

  const CF_3 *p_WallValue;
  const CF_3 *p_InterfaceValue;

public:
  BoundaryConditions3D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    MixedInterface = NULL;
    p_InterfaceValue = NULL;
  }

  inline void setWallTypes( const WallBC3D& w )
  {
    WallType_ = &w;
  }

  inline const WallBC3D& getWallType() const
  {
    return *WallType_;
  }

  inline void setWallValues( const CF_3& v ){
    p_WallValue = &v;
  }

  inline void setInterfaceType(BoundaryConditionType bc, const mixed_interface* obj_= NULL){
    InterfaceType_ = bc;
    if(InterfaceType_ == MIXED)
    {
      if(obj_ == NULL)
        throw std::invalid_argument("BoundaryConditions3D::setInterfaceType(): if the interface type is set to MIXED, a pointer to a class of abstract type mixed_interface MUST be provided as well");
      MixedInterface = obj_;
    }
  }

  inline void setInterfaceValue(const CF_3& in){
    p_InterfaceValue = &in;
  }

  inline const CF_3& getInterfaceValue(){
    return *p_InterfaceValue;
  }

  inline BoundaryConditionType wallType( double x, double y, double z ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y,z);
  }

  inline BoundaryConditionType wallType(const double xyz_[]) const
  {
    return wallType(xyz_[0],xyz_[1],xyz_[2]);
  }

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline BoundaryConditionType interfaceType(const double* xyz) const
  {
    if(InterfaceType_ != MIXED)
      return interfaceType();
    return MixedInterface->mixed_type(xyz);
  }

  inline double wallValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y,z);
  }

  inline double wallValue(const double xyz_[]) const
  {
    return p_WallValue->operator ()(xyz_[0],xyz_[1],xyz_[2]);
  }

  inline double interfaceValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y,z);
  }

  inline double  interfaceValue(double xyz_[]) const
  {
    return interfaceValue(xyz_[0], xyz_[1], xyz_[2]);
  }

};

/*!
 * \brief index_of_node finds the (local) index of a node as defined within p4est, i.e. as a pest_quadrant_t structure whose level is P4EST_MAXLEVEL!
 *        The method uses a binary search through the provided nodes: its complexity is O(log(N_nodes)).
 *        The given node MUST MANDATORILY be canonicalized before being passed to this function to ensure consistency with the provided nodes: use
 *        p4est_node_canonicalize beforehand!
 * \param [in]    n node whose local index is queried!
 * \param [in]    nodes the nodes data structure
 * \param [inout] idx the local index of the node on output if found, undefined if not found (i.e. if the returned value is false)
 * \return true if the queried node exists and was found in the nodes (i.e. if the idx is valid), false otherwise.
 */
bool index_of_node(const p4est_quadrant_t *n, p4est_nodes_t* nodes, p4est_locidx_t& idx);

/*!
 * \brief linear_interpolation performs linear interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global, double *results, const unsigned int n_results);
double linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global);

/*!
 * \brief quadratic_non_oscillatory_interpolation performs non-oscilatory quadratic interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    Fdd a simple C-style array of size n_results*P4EST_CHILDREN*P4EST_DIM, containing the values of the second derivatives of the function(s) at the vertices of the quadrant
 *                Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] = value of the second derivative along dimension i, at quadrant's node j, of the kth function,
 *                0 <= i < P4EST_DIM, 0<= j < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results);
double quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);
/*!
 * \brief quadratic_interpolation performs quadratic interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    Fdd a simple C-style array of size n_results*P4EST_CHILDREN*P4EST_DIM, containing the values of the second derivatives of the function(s) at the vertices of the quadrant
 *                Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] = value of the second derivative along dimension i, at quadrant's node j, of the kth function,
 *                0 <= i < P4EST_DIM, 0<= j < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void quadratic_interpolation(const p4est_t* p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results);
double quadratic_interpolation(const p4est_t* p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);

p4est_bool_t nodes_are_equal(int mpi_size, p4est_nodes_t* nodes_1, p4est_nodes_t* nodes_2);

PetscErrorCode VecGetLocalAndGhostSizes(Vec& v, PetscInt& local_size, PetscInt& ghosted_size);

/*!
 * \brief vectorIsWellSetForNodes
 * (collective in debug mode)
 * \param v
 * \param nodes
 * \param mpicomm
 * \return
 */
bool vectorIsWellSetForNodes(Vec& v, const p4est_nodes_t* nodes, const MPI_Comm& mpicomm);

/*!
 * \brief VecCreateGhostNodes Creates a ghosted PETSc parallel vector on the nodes based on p4est node ordering
 * \param p4est [in]  the forest
 * \param nodes [in]  the nodes numbering data structure
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateGhostNodes(const p4est_t *p4est, p4est_nodes_t *nodes, Vec* v);

/*!
 * \brief VecCreateGhostNodesBlock Creates a ghosted block PETSc parallel vector on the nodes
 * \param p4est      [in]  p4est object
 * \param nodes      [in]  the nodes object
 * \param block_size [in]  block size of the vector
 * \param v          [out] PETSc vector
 * \return
 */
PetscErrorCode VecCreateGhostNodesBlock(const p4est_t *p4est, p4est_nodes_t *nodes, PetscInt block_size, Vec* v);

p4est_bool_t ghosts_are_equal(p4est_ghost_t* ghost_1, p4est_ghost_t* ghost_2);

/*!
 * \brief VecCreateGhostNodes Creates a ghosted PETSc parallel vector on the cells
 * \param p4est [in]  the forest
 * \param ghost [in]  the ghost cells
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateGhostCells(const p4est_t *p4est, p4est_ghost_t *ghost, Vec* v);

/*!
 * \brief VecCreateCellsNoGhost Creates a PETSc parallel vector on the cells
 * \param p4est [in]  the forest
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateCellsNoGhost(const p4est_t *p4est, Vec* v);

/*!
 * \brief VecCreateGhostNodesBlock Creates a ghosted block PETSc parallel vector
 * \param p4est      [in]  p4est object
 * \param ghost      [in]  the ghost cells
 * \param block_size [in]  block size of the vector
 * \param v          [out] PETSc vector
 * \return
 */
PetscErrorCode VecCreateGhostCellsBlock(const p4est_t *p4est, p4est_ghost_t *ghost, PetscInt block_size, Vec* v);

/*!
 * \brief VecScatterCreateChangeLayout Create a VecScatter context useful for changing the parallel layout of a vector
 * \param comm  [in]  MPI_Comm to which parallel vectors belong
 * \param from  [in]  input vector layout
 * \param to    [in]  output vector layout
 * \param ctx   [out] the created VecScatter context
 * \return
 */
PetscErrorCode VecScatterCreateChangeLayout(MPI_Comm comm, Vec from, Vec to, VecScatter *ctx);

/*!
 * \brief VecGhostChangeLayoutBegin Start changing the layout of a parallel vector. This potentially involves
 *  sending and receiving messages in a non-blocking mode
 * \param ctx   [in]  VecScatter context to initiate the transfer
 * \param from  [in]  input vector to the change the parallel layout
 * \param to    [out] output vector with the same global values but with a different parallel layout
 * \return
 */
PetscErrorCode VecGhostChangeLayoutBegin(VecScatter ctx, Vec from, Vec to);

/*!
 * \brief VecGhostChangeLayoutEnd Finish changing the layout of a parallel vector. This potentially involves
 *  sending and receiving messages in a non-blocking mode
 * \param ctx   [in]  VecScatter context to initiate the transfer
 * \param from  [in]  input vector to the change the parallel layout
 * \param to    [out] output vector with the same global values but with a different parallel layout
 * \return
 */
PetscErrorCode VecGhostChangeLayoutEnd(VecScatter ctx, Vec from, Vec to);

/*!
 * \brief is_folder returns true if the path points to an existing folder
 * does not use boost nor c++17 standard to maximize portability
 * \param path: path to be checked
 * \return true if the path points to a folder
 * [throws std::runtime_error if the path cannot be accessed]
 */
bool is_folder(const char* path);

/*!
 * \brief file_exists returns true if the path points to an existing file
 * does not use boost nor c++17 standard to maximize portability
 * \param path:path to be checked
 * \return true if there exists a file corresponding to the given path
 */
bool file_exists(const char* path);

/*!
 * \brief create_directory creates a folder indicated by the given path, permission rights: 755
 * does not use boost nor c++17 standard to maximize portability (parents are created as well)
 * \param path: path to the folder to be created
 * \param mpi_rank: rank of the calling process
 * \param comm: communicator
 * \return 0 if the creation was successful, non-0 otherwise
 * [the root process creates the folder, the operation is collective by MPI_Bcast on the result]
 */
int create_directory(const char* path, int mpi_rank, MPI_Comm comm=MPI_COMM_WORLD);

/*!
 * \brief delete_directory_recursive explores a directory then
 * - it deletes all regular files in the directrory;
 * - it goes through subdirectories and calls the same function on them;
 * - after recursive call returns the subdirectory is removed;
 * does not use boost nor c++17 standard to maximize portability
 * \param root_path: path to the root directory to be entirely deleted
 * \param mpi_rank: rank of the calling process
 * \param comm: communicator
 * \param non_collective: flag skipping the (collective) steps (for recursive calls on root process)
 * \return 0 if the deletion was successful, non-0 otherwise
 * [the root process deletes the content, the operation is collective by MPI_Bcast on the final result]
 * [throws std::invalid_argument if the root_path is NOT a directory]
 */
int delete_directory(const char* root_path, int mpi_rank, MPI_Comm comm=MPI_COMM_WORLD, bool non_collective=false);

int get_subdirectories_in(const char* root_path, std::vector<std::string>& subdirectories);

inline double int2double_coordinate_transform(p4est_qcoord_t a){
  return static_cast<double>(a)/static_cast<double>(P4EST_ROOT_LEN);
}

void dxyz_min(const p4est_t *p4est, double *dxyz);

void dxyz_quad(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz);

void xyz_min(const p4est_t *p4est, double *xyz_min_);

void xyz_max(const p4est_t *p4est, double *xyz_max_);

inline void xyz_min_max(const p4est_t *p4est, double *xyz_min_, double *xyz_max_){
  xyz_min(p4est, xyz_min_);
  xyz_max(p4est, xyz_max_);
}

inline double node_x_fr_n(const p4est_indep_t *ni){
  return ni->x == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->x)/static_cast<double>(P4EST_ROOT_LEN);
}

inline double node_y_fr_n(const p4est_indep_t *ni){
  return ni->y == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->y)/static_cast<double>(P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
inline double node_z_fr_n(const p4est_indep_t *ni){
  return ni->z == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->z)/static_cast<double>(P4EST_ROOT_LEN);
}
#endif

inline double node_x_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
}

inline double node_y_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
}

#ifdef P4_TO_P8
inline double node_z_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
}
#endif

inline void node_xyz_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes, double *xyz)
{
  xyz[0] = node_x_fr_n(n,p4est,nodes);
  xyz[1] = node_y_fr_n(n,p4est,nodes);
#ifdef P4_TO_P8
  xyz[2] = node_z_fr_n(n,p4est,nodes);
#endif
}

inline void p4est_dxyz_min(const p4est_t* p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN-1];
  const double *vert = p4est->connectivity->vertices;

  double h = 1.0 / (double) (1 << data->max_lvl);
  for (short i=0; i<P4EST_DIM; ++i)
    dxyz[i] = (vert[3*vp + i] - vert[3*vm + i]) * h;
}

inline void p4est_dxyz_max(const p4est_t* p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;
  p4est_topidx_t tr_idx = p4est->trees->elem_count - 1;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[tr_idx * P4EST_CHILDREN + P4EST_CHILDREN-1];
  const double *vert = p4est->connectivity->vertices;

  double h = 1.0 / (double) (1 << data->min_lvl);
  for (short i=0; i<P4EST_DIM; ++i)
    dxyz[i] = (vert[3*vp + i] - vert[3*vm + i]) * h;
}

inline double p4est_diag_min(const p4est_t* p4est) {
  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
#ifdef P4_TO_P8
  return sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
#else
  return sqrt(SQR(dx[0]) + SQR(dx[1]));
#endif
}

inline double p4est_diag_max(const p4est_t* p4est) {
  double dx[P4EST_DIM];
  p4est_dxyz_max(p4est, dx);
#ifdef P4_TO_P8
  return sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
#else
  return sqrt(SQR(dx[0]) + SQR(dx[1]));
#endif
}

/*!
 * \brief get the z-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_x_fr_i(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->x)/static_cast<double>(P4EST_ROOT_LEN);
}

/*!
 * \brief get the y-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_y_fr_j(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->y)/static_cast<double>(P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
/*!
 * \brief get the x-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_z_fr_k(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->z)/static_cast<double>(P4EST_ROOT_LEN);
}
#endif

inline p4est_tree_t* get_tree(p4est_topidx_t tr, p4est_t* p4est)
{
#ifdef CASL_THROWS
  if(tr < p4est->first_local_tree || tr > p4est->last_local_tree) {
    std::ostringstream oss;
    oss << "Tree with index " << tr << " is outside range. Processor " << p4est->mpirank
        << " inclusive range is [" << p4est->first_local_tree << ", " << p4est->last_local_tree << "]" << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_tree_t*)sc_array_index(p4est->trees, tr);
}

inline p4est_quadrant_t* get_quad(p4est_locidx_t q, p4est_tree_t* tree)
{
#ifdef CASL_THROWS
  if(q < 0 || q >= (p4est_locidx_t) tree->quadrants.elem_count) {
    std::ostringstream oss;
    oss << "Quad with index " << q << " is outside range of current tree. "
        << "Number of quadrants on this tree is " << tree->quadrants.elem_count << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
}

inline p4est_quadrant_t* get_quad(p4est_locidx_t q, p4est_ghost_t* ghost)
{
#ifdef CASL_THROWS
  if(q < 0 || q >= (p4est_locidx_t) ghost->ghosts.elem_count) {
    std::ostringstream oss;
    oss << "Quad with index " << q << " is outside range of ghost layer. "
        << "Size of ghost layer is " << ghost->ghosts.elem_count << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
}

inline p4est_indep_t* get_node(p4est_locidx_t n, p4est_nodes_t* nodes)
{
#ifdef CASL_THROWS
  if(n < 0 || n >= (p4est_locidx_t) nodes->indep_nodes.elem_count) {
    std::ostringstream oss;
    oss << "Node with index " << n << " is outside range of nodes." << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
}

/*!
 * \brief get the x-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_x_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*(quad_x_fr_i(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_xmin;
}

/*!
 * \brief quad_x        compute the x-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the x-coordinate
 */
inline double quad_x(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*(quad_x_fr_i(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_xmin;
}

/*!
 * \brief quad_dx     compute the dx size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dx
 */
inline double quad_dx(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];

  return (tree_xmax-tree_xmin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}

/*!
 * \brief get the y-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_y_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*(quad_y_fr_j(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_ymin;
}

/*!
 * \brief quad_y        compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the y-coordinate
 */
inline double quad_y(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*(quad_y_fr_j(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_ymin;
}

/*!
 * \brief quad_dy     compute the dy size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dy
 */
inline double quad_dy(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  return (tree_ymax-tree_ymin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
/*!
 * \brief get the z-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_z_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*(quad_z_fr_k(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_zmin;
}

/*!
 * \brief quad_z        compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the z-coordinate
 */
inline double quad_z(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*(quad_z_fr_k(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_zmin;
}

/*!
 * \brief quad_dz     compute the dz size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dz
 */
inline double quad_dz(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];

  return (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}
#endif


/*!
 * \brief get the xyz-coordinates of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline void quad_xyz_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost, double *xyz)
{
  xyz[0] = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
  xyz[1] = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
  xyz[2] = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif
}

/*!
 * \brief quad_z_fr_q   compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \param xyz   [out]   pointer to array of size P4EST_DIM to store xyz
 * \return  the z-coordinate
 */
inline void quad_xyz(const p4est_t *p4est, const p4est_quadrant_t *quad, double *xyz)
{
  xyz[0] = quad_x(p4est, quad);
  xyz[1] = quad_y(p4est, quad);
#ifdef P4_TO_P8
  xyz[2] = quad_z(p4est, quad);
#endif
}

/*!
 * \brief quad_dxyz   compute the dxyz sizes of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \param dxyz  [out]   pointer to array of size P4EST_DIM to store dxyz
 * \return  dy
 */
inline void quad_dxyz(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz)
{
  dxyz[0] = quad_dx(p4est, quad);
  dxyz[1] = quad_dy(p4est, quad);
#ifdef P4_TO_P8
  dxyz[2] = quad_dz(p4est, quad);
#endif
}

/*!
 * \brief compute the xyz_min of a given tree index
 * \param p4est the forest object
 * \param tr_idx index of the tree to find the xyz_min
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void tree_xyz_min(p4est_t* p4est, p4est_topidx_t tr_idx, double *xyz)
{
  p4est_topidx_t vtx = p4est->connectivity->tree_to_vertex[tr_idx*P4EST_CHILDREN];
  xyz[0] = p4est->connectivity->vertices[3*vtx + 0];
  xyz[1] = p4est->connectivity->vertices[3*vtx + 1];
#ifdef P4_TO_P8
  xyz[2] = p4est->connectivity->vertices[3*vtx + 2];
#endif
}

/*!
 * \brief compute the xyz_max of a given tree index
 * \param p4est the forest object
 * \param tr_idx index of the tree to find the xyz_max
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void tree_xyz_max(p4est_t* p4est, p4est_topidx_t tr_idx, double *xyz)
{
  p4est_topidx_t vtx = p4est->connectivity->tree_to_vertex[tr_idx*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  xyz[0] = p4est->connectivity->vertices[3*vtx + 0];
  xyz[1] = p4est->connectivity->vertices[3*vtx + 1];
#ifdef P4_TO_P8
  xyz[2] = p4est->connectivity->vertices[3*vtx + 2];
#endif
}

/*!
 * \brief computes the xyz_min of the entire forest
 * \param p4est teh forest object
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void p4est_xyz_min(p4est_t* p4est, double *xyz)
{
  tree_xyz_min(p4est, 0, xyz);
}

/*!
 * \brief computes the xyz_max of the entire forest
 * \param p4est teh forest object
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void p4est_xyz_max(p4est_t* p4est, double *xyz)
{
  tree_xyz_max(p4est, p4est->trees->elem_count - 1, xyz);
}

/*!
 * \brief integrate_over_negative_domain_in_one_quadrant
 */
double integrate_over_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_negative_domain integrate a quantity f over the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi
 * \param f the scalar to integrate
 * \return the integral of f over the phi<0 domain, \int_{\phi<0} f
 */
double integrate_over_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief area_in_negative_domain_in_one_quadrant
 */
double area_in_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

/*!
 * \brief area_in_negative_domain compute the area of the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \return the area in the negative phi domain, i.e. \int_{phi<0} 1
 */
double area_in_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi);

/*!
 * \brief integrate_over_interface_in_one_quadrant
 */
double integrate_over_interface_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_interface integrate a scalar f over the 0-contour of the level-set function phi.
 *        note: first order convergence only
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \param f the scalar to integrate
 * \return the integral of f over the contour defined by phi, i.e. \int_{phi=0} f
 */
double integrate_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief compute_mean_curvature computes the mean curvature using compact stencil k = -div(n)
 * \param qnnn neighborhood information for the point
 * \param phi pointer to the level set function
 * \param phi_x pointer to an array of size P4EST_DIM for the first derivatives of levelset. CANNOT be NULL.
 * \return mean curvature at a single point
 */
double compute_mean_curvature(const quad_neighbor_nodes_of_node_t& qnnn, double* phi, double* phi_x[P4EST_DIM]);

/*!
 * \brief compute_mean_curvature computes the mean curvature using divergence of normal k = -div(n)
 * \param qnnn neighborhood information for the point
 * \param normals pointer to an array of size P4EST_DIM of the normals. CANNOT be NULL.
 * \return mean curvature at a single point
 */
double compute_mean_curvature(const quad_neighbor_nodes_of_node_t& qnnn, double* normals[P4EST_DIM]);

/*!
 * \brief compute_mean_curvature computes the mean curvature in the entire domain k = -div(n)
 * \param neighbors the node neighborhood information
 * \param phi levelset function
 * \param phi_x an array of size P4EST_DIM representing the first derivative of levelset in the entire domain. CANNOT be NULL.
 * \param kappa curvature function in the entire domain
 */
void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec phi_x[P4EST_DIM], Vec kappa);

/*!
 * \brief compute_mean_curvature computes the mean curvature in the entire domain k = -div(n)
 * \param neighbors the node neighborhood information
 * \param normals pointer to an array of size P4EST_DIM for the normals. CANNOT be NULL.
 * \param kappa curvature function in the entire domain
 */
void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec normals[P4EST_DIM], Vec kappa);

/*!
 * \brief compute_normals computes the (scaled) normal to the surface
 * \param [in]  qnnn    neighborhood information for the point
 * \param [in]  phi     pointer to the levelset function
 * \param [out] normals array of size P4EST_DIM for the normals
 */
void compute_normals(const quad_neighbor_nodes_of_node_t& qnnn, double *phi, double normals[P4EST_DIM]);

/*!
 * \brief compute_normals computes the (scaled) normal to the surface for the entire grid
 * \param [in]  neighbors the neighborhood information
 * \param [in]  phi       PETSc vector of the levelset function
 * \param phi the level-set function
 * \param [out] normals   array of size P4EST_DIM of PETSc vectors to store the normal in the entire doamin
 */
void compute_normals(const my_p4est_node_neighbors_t& neighbors, Vec phi, Vec normals[P4EST_DIM]);

/*!
 * \brief interface_length_in_one_quadrant
 */
double interface_length_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

/*!
 * \brief interface_length
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \return the length (or area in 3D) of the contour defined by phi
 */
double interface_length(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi);

/*!
 * \brief is_node_xmWall checks if a node is on x^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the left domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_xpWall checks if a node is on x^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the right domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain bottom boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain top boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni);

#ifdef P4_TO_P8
/*!
 * \brief is_node_zmWall checks if a node is on z^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain back boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_zmWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_zpWall checks if a node is on z^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain front boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_zpWall(const p4est_t *p4est, const p4est_indep_t *ni);
#endif

/*!
 * \brief is_node_Wall checks if a node is on any of domain boundaries
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain boundary and p4est is _NOT_ periodic
 */
bool is_node_Wall  (const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_Wall checks if a node is on the oriented domain boundary pointed by oriented_dir
 * \param p4est         [in] p4est
 * \param ni            [in] pointer to the node structure
 * \param oriented_dir  [in] oriented direction (dir::f_m00, dir::f_p00, etc.)
 * \return true if the point is on the domain boundary and p4est is _NOT_ periodic
 */
bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni, const unsigned char oriented_dir);

/*!
 * \brief is_quad_xmWall checks if a quad is on x^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the left domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_xmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_xpWall checks if a quad is on x^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the right domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_xpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ymWall checks if a quad is on y^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the bottom domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_ymWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ypWall checks if a quad is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the top domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_ypWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zmWall checks if a quad is on z^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the back domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_zmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zpWall checks if a quad is on z^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the front domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_zpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_Wall checks if a quad is on the domain boundary in a given direction
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \param dir   [in] the direction to check, dir::f_m00, dir::f_p00, dir::f_0m0 ...
 * \return true if the quad is on the domain boundary in the direction dir and p4est is _NOT_ periodic
 */
bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi, int dir);

/*!
 * \brief is_quad_Wall checks if a quad is on any of domain boundaries
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the domain boundary and p4est is _NOT_ periodic
 */
bool is_quad_Wall  (const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_periodic checks if the forest is periodic in direction dir
 * \param p4est [in] the forest
 * \param dir   [in] the direction to check, 0 (x), 1 (y) or 2 (z, only in 3D)
 * \return true if the forest is periodic in direction dir, false otherwise
 */
inline bool is_periodic(const p4est_t *p4est, int dir)
{
  /* check whether there is not a boundary on the left side of first tree */
  P4EST_ASSERT (0 <= dir && dir < P4EST_DIM);

  const int face = 2 * dir;
  const p4est_topidx_t tfindex = 0 * P4EST_FACES + face;

  return !(p4est->connectivity->tree_to_tree[tfindex] == 0 &&
           p4est->connectivity->tree_to_face[tfindex] == face);
}

/*!
 * \brief find the owner rank of a ghost quadrant
 * \param ghost the ghost structure
 * \param ghost_idx the index of the ghost quadrant (between 0 and the number of ghost quadrants)
 * \return the rank who owns the ghost quadrant
 */
int quad_find_ghost_owner(const p4est_ghost_t *ghost, p4est_locidx_t ghost_idx);

/*!
 * \brief sample_cf_on_nodes samples a cf function on the nodes. both local and ghost poinst are considered
 * \param p4est [in] the p4est object
 * \param nodes [in] the nodes data structure
 * \param cf    [in] the cf function. It is assumed that the function can be evaluated at _ANY_ point, whether local or remote
 * \param f     [in, out] a PETSc Vec object to store the result. It is assumed that the vector is allocated. A check
 * is performed to ensure enough memory is available in the Vec object.
 */
#ifdef P4_TO_P8
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f);
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_3& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3* cf_array[], Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, std::vector<double>& f);
#else
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_2& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2* cf_array[], Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, std::vector<double>& f);
#endif

void write_comm_stats(const p4est_t *p4est, const p4est_ghost_t* ghost, const p4est_nodes_t *nodes,
                 const char* partition_name = NULL, const char* topology_name = NULL, const char* neighbors_name = NULL);

inline double ranged_rand(double a, double b, int seed = 0){
  if (seed) srand(seed);
  return (static_cast<double>(rand())/static_cast<double>(RAND_MAX) * (b-a) + a);
}

inline int ranged_rand(int a, int b, int seed = 0){
  if (seed) srand(seed);
  return (rand()%(b-a) + a);
}

inline int ranged_rand_inclusive(int a, int b, int seed = 0){
  if (seed) srand(seed);
  return (rand()%(b-a+1) + a);
}

// A Logger for interpolation function
struct InterpolatingFunctionLogEntry{
  int num_local_points, num_send_points, num_send_procs, num_recv_points, num_recv_procs;
};

class InterpolatingFunctionLogger{
  InterpolatingFunctionLogger() {}
  InterpolatingFunctionLogger(const InterpolatingFunctionLogger& ) {}
  static std::vector<InterpolatingFunctionLogEntry> entries;

public:
  inline static InterpolatingFunctionLogger& get_instance() {
    static InterpolatingFunctionLogger instance;
    return instance;
  }

  inline void log(const InterpolatingFunctionLogEntry& entry) {
    entries.push_back(entry);
  }

  inline void write(const std::string& filename) {
    for (size_t i = 0; i<entries.size();i++) {
      FILE *fp;
      std::ostringstream oss; oss << filename << "_" << i << ".dat";
      PetscFOpen(PETSC_COMM_WORLD, oss.str().c_str(), "w", &fp);
      PetscFPrintf(PETSC_COMM_WORLD, fp, "%% num_local_points | num_send_points | num_send_procs | num_recv_points | num_recv_procs \n");
      PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%7d \t %7d \t %4d \t %7d \t %4d \n", entries[i].num_local_points,
                                                                                           entries[i].num_send_points,
                                                                                           entries[i].num_send_procs,
                                                                                           entries[i].num_recv_points,
                                                                                           entries[i].num_recv_procs);
      PetscSynchronizedFlush(PETSC_COMM_WORLD, fp);
      PetscFClose(PETSC_COMM_WORLD, fp);
    }
    entries.clear();
  }
};

/*!
 * \brief prepares MPI, PETSc, p4est, and sc libraries
 */
class mpi_environment_t{
  PetscErrorCode ierr;
  MPI_Comm mpicomm;
  int mpirank;
  int mpisize;

public:
  ~mpi_environment_t(){
    ierr = PetscFinalize(); CHKERRXX(ierr);
    sc_finalize();
    MPI_Finalize();
  }

  void init(int argc, char **argv){
    mpicomm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(mpicomm, &mpisize);
    MPI_Comm_rank(mpicomm, &mpirank);

    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);

#ifdef DEBUG
    sc_init (mpicomm, P4EST_FALSE, P4EST_FALSE, NULL, SC_LP_DEFAULT); // to allow easy debugging --> backtracks the P4EST_ASSERTs!
#else
    sc_init (mpicomm, P4EST_FALSE, P4EST_FALSE, NULL, SC_LP_SILENT);
#endif
    p4est_init (NULL, SC_LP_SILENT);
#ifdef CASL_LOG_EVENTS
    register_petsc_logs();
#endif
  }

  inline const MPI_Comm& comm() const {return mpicomm;}
  inline const int& rank() const {return mpirank;}
  inline const int& size() const {return mpisize;}

};

class parStopWatch{
public:
  typedef enum{
    root_timings,
    all_timings
  } stopwatch_timing;

private:
  double ts, tf;
  MPI_Comm comm_;
  int mpirank;
  int mpisize;
  std::string msg_;
  stopwatch_timing timing_;
  std::vector<double> t;
  FILE *f_;

public:

  parStopWatch(stopwatch_timing timing = root_timings, FILE *f = stdout, MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm), timing_(timing), f_(f)
  {
    MPI_Comm_rank(comm_, &mpirank);
    MPI_Comm_size(comm_, &mpisize);
    t.resize(mpisize,0);
  }

  void start(const std::string& msg){
    msg_ = msg;
    if(msg_.length() > 0)
      PetscFPrintf(comm_, f_, "%s ... \n", msg.c_str());
    ts = MPI_Wtime();
  }

  void stop(){
    tf = MPI_Wtime();
  }

  double read_duration(){
    double elap = tf - ts;
    if (timing_ == all_timings)
      MPI_Gather(&elap, 1, MPI_DOUBLE, &t[0], 1, MPI_DOUBLE, 0, comm_);
    return elap;
  }

  void print_stats_only(){
    if(timing_ != all_timings)
    {
      PetscFPrintf(comm_, stderr, "parStopWatch::print_stats_only() can be called only in 'all_timing' mode.");
      return;
    }
    print_duration(true);
  }

  double print_duration(bool print_stats_only_ = false){
    double elap = read_duration();
    PetscPrintf(comm_, "%s ... done in \n", msg_.c_str());
    if (timing_ == all_timings){
      double tmax, tmin, tavg, tdev;
      tmax = tmin = elap;
      tavg = tdev = 0;
      if (mpirank == 0){
        if(!print_stats_only_)
          PetscFPrintf(comm_, f_, "t = [");
        for (size_t i=0; i<t.size()-1; i++){
          if(!print_stats_only_)
            PetscFPrintf(comm_, f_, "%.5lf, ", t[i]);
          tavg += t[i];
          tmax = MAX(tmax, t[i]);
          tmin = MIN(tmin, t[i]);
        }
        if(!print_stats_only_)
          PetscFPrintf(comm_, f_, "%.5lf];\n", t.back());

        tavg += t.back();
        tmax = MAX(tmax, t.back());
        tmin = MIN(tmin, t.back());

        tavg /= mpisize;

        for (size_t i=0; i<t.size(); i++){
          tdev += (t[i]-tavg)*(t[i]-tavg);
        }
        tdev = sqrt(tdev/mpisize);
      }

      PetscFPrintf(comm_, f_, " t_max = %.5lf (s), t_max/t_min = %.2lf, t_avg = %.5lf (s), t_dev/t_avg = %% %2.1lf, t_dev/(t_max-t_min) = %% %2.1lf\n\n", tmax, tmax/tmin, tavg, tdev/tavg*100, tdev/(tmax-tmin)*100);
    } else {
      PetscFPrintf(comm_, f_, " %.5lf secs. on process %d [Note: only showing root's timings]\n\n", elap, mpirank);
    }
    return elap;
  }
};

#endif // UTILS_H
