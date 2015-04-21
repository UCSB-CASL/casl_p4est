#ifndef UTILS_H
#define UTILS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <p8est_ghost.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <p4est_ghost.h>
#endif
#include <src/petsc_logging.h>
#include "petsc_compatibility.h"

#include <petsc.h>
#include <stdexcept>
#include <sstream>
#include <vector>

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
  double lip;
  virtual double operator()(double x) const=0 ;
  virtual ~CF_1() {}
};


class CF_2
{
public:
  double lip;
  virtual double operator()(double x, double y) const=0 ;
  virtual ~CF_2() {}
};

class CF_3
{
public:
  double lip;
  virtual double operator()(double x, double y,double z) const=0 ;
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

typedef enum {
  DIRICHLET,
  NEUMANN,
  ROBIN,
  NOINTERFACE,
  MIXED,
  IGNORE
} BoundaryConditionType;

std::ostream& operator << (std::ostream& os, BoundaryConditionType  type);
std::istream& operator >> (std::istream& is, BoundaryConditionType& type);

class WallBC2D
{
public:
  virtual BoundaryConditionType operator()( double x, double y ) const=0 ;
};

class WallBC3D
{
public:
  virtual BoundaryConditionType operator()( double x, double y, double z ) const=0 ;
};


class BoundaryConditions2D
{
private:
  const WallBC2D* WallType_;
  BoundaryConditionType InterfaceType_;

  const CF_2 *p_WallValue;
  const CF_2 *p_InterfaceValue;

public:
  BoundaryConditions2D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
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

  inline void setInterfaceType(BoundaryConditionType bc){
    InterfaceType_ = bc;
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

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline double wallValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y);
  }

  inline double interfaceValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y);
  }
};

class BoundaryConditions3D
{
private:
  const WallBC3D* WallType_;
  BoundaryConditionType InterfaceType_;

  const CF_3 *p_WallValue;
  const CF_3 *p_InterfaceValue;

public:
  BoundaryConditions3D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
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

  inline void setInterfaceType(BoundaryConditionType bc){
    InterfaceType_ = bc;
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

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline double wallValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y,z);
  }

  inline double interfaceValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y,z);
  }
};

// p4est boolean type
typedef int p4est_bool_t;
#define P4EST_TRUE  1
#define P4EST_FALSE 0



double linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global);

/*!
 * \brief non_oscilatory_quadratic_interpolation performs non-oscilatory quadratic interpolation for a point
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quarant
 * \param F a simple C-style array of size 4, containing the values of the function at the vertices of the quadrant. __MUST__ be z-ordered
 * \param Fxx a simple C-style array of size 4, containing the values of the xx derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param Fyy a simple C-style array of size 4, containing the values of the yy derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param x_global global x-coordinate ointerface_location_with_second_order_derivativef the point
 * \param y_global global y-coordinate of the point
 * \return interpolated value
 */
double quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);

/*!
 * \brief quadratic_interpolation performs quadratic interpolation for a point
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quarant
 * \param F a simple C-style array of size 4, containing the values of the function at the vertices of the quadrant. __MUST__ be z-ordered
 * \param Fxx a simple C-style array of size 4, containing the values of the xx derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param Fyy a simple C-style array of size 4, containing the values of the yy derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param x_global global x-coordinate of the point
 * \param y_global global y-coordinate of the point
 * \return interpolated value
 */
double quadratic_interpolation(const p4est_t* p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);

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

/*!
 * \brief VecCreateGhostNodes Creates a ghosted PETSc parallel vector on the cells
 * \param p4est [in]  the forest
 * \param ghost [in]  the ghost cells
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateGhostCells(const p4est_t *p4est, p4est_ghost_t *ghost, Vec* v);

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


inline double int2double_coordinate_transform(p4est_qcoord_t a){
  return static_cast<double>(a)/static_cast<double>(P4EST_ROOT_LEN);
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

inline double node_x_fr_n(p4est_locidx_t n, p4est_t *p4est, p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
  return node_x_fr_n(node) + tree_xmin;
}

inline double node_y_fr_n(p4est_locidx_t n, p4est_t *p4est, p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
  return node_y_fr_n(node) + tree_ymin;
}

#ifdef P4_TO_P8
inline double node_z_fr_n(p4est_locidx_t n, p4est_t *p4est, p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
  return node_z_fr_n(node) + tree_zmin;
}
#endif

inline double quad_x_fr_i(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->x)/static_cast<double>(P4EST_ROOT_LEN);
}

inline double quad_y_fr_j(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->y)/static_cast<double>(P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
inline double quad_z_fr_k(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->z)/static_cast<double>(P4EST_ROOT_LEN);
}
#endif

/*!
 * \brief integrate_over_negative_domain_in_one_quadrant
 */
double integrate_over_negative_domain_in_one_quadrant(const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

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
double area_in_negative_domain_in_one_quadrant(p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

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
double integrate_over_interface_in_one_quadrant(p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

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
 * \note: periodicity is not implemented
 */
bool is_node_Wall  (const p4est_t *p4est, const p4est_indep_t *ni);

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
 * \note: periodicity is not implemented
 */
bool is_quad_xpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ymWall checks if a quad is on y^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the bottom domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_ymWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ypWall checks if a quad is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the top domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_ypWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zmWall checks if a quad is on z^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the back domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_zmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zpWall checks if a quad is on z^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the front domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_zpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_Wall checks if a quad is on the domain boundary in a given direction
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \param dir   [in] the direction to check, dir::f_m00, dir::f_p00, dir::f_0m0 ...
 * \return true if the quad is on the domain boundary in the direction dir and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi, int dir);

/*!
 * \brief is_quad_Wall checks if a quad is on any of domain boundaries
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_Wall  (const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

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
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, std::vector<double>& f);
#else
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_2& cf, Vec f);
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
class Session{
  PetscLogStage stage;
  PetscErrorCode ierr;
public:
  ~Session(){
    ierr = PetscFinalize(); CHKERRXX(ierr);
  }

  void init(int argc, char **argv, MPI_Comm mpicomm){
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
    sc_init (mpicomm, P4EST_FALSE, P4EST_FALSE, NULL, SC_LP_SILENT);
    p4est_init (NULL, SC_LP_SILENT);
#ifdef CASL_LOG_EVENTS
    register_petsc_logs();
#endif
  }

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
    PetscFPrintf(comm_, f_, "%s ... \n", msg.c_str());
    ts = MPI_Wtime();
  }

  void stop(){
    tf = MPI_Wtime();
  }

  double read_duration(){
    double elap = tf - ts;

    PetscPrintf(comm_, "%s ... done in \n", msg_.c_str());
    if (timing_ == all_timings){
      MPI_Gather(&elap, 1, MPI_DOUBLE, &t[0], 1, MPI_DOUBLE, 0, comm_);
      double tmax, tmin, tavg, tdev;
      tmax = tmin = elap;
      tavg = tdev = 0;
      if (mpirank == 0){
        PetscFPrintf(comm_, f_, "t = [");
        for (size_t i=0; i<t.size()-1; i++)
          PetscFPrintf(comm_, f_, "%.5lf, ", t[i]);
        PetscFPrintf(comm_, f_, "%.5lf];\n", t.back());

        for (size_t i=0; i<t.size(); i++){
          tavg += t[i];
          tmax = MAX(tmax, t[i]);
          tmin = MIN(tmin, t[i]);
        }
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

typedef struct
{
  MPI_Comm            mpicomm;
  int                 mpisize;
  int                 mpirank;
}
mpi_context_t;

#endif // UTILS_H
