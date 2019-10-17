#ifndef MY_P4EST_FACES_H
#define MY_P4EST_FACES_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/casl_math.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/casl_math.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#endif

#include <vector>

using std::vector;

#define NO_VELOCITY -1

class my_p4est_faces_t
{
  friend class my_p4est_poisson_faces_t;
  friend class my_p4est_interpolation_faces_t;
  friend class my_p4est_navier_stokes_t;
private:

  typedef struct face_quad_ngbd
  {
    /* the indices of the neighbor quadrant.
     * - if the quadrant is local, store its index and the corresponding tree.
     *   Note that it is the local index in the forest (including ghosts), and not the index in the tree.
     * - if the quadrant is ghost, store its local index, i.e. local_num_quadrants + ghost_index.
     *
     * If a face has two well defined neighboring quadrants, the local one is prefered over the ghost one.
     * To find out which direction the quadrant is in, use the q2u_ structure to check which index matches.
     */
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    face_quad_ngbd() { quad_idx=-1; tree_idx=-1; }
  } face_quad_ngbd;

  typedef struct faces_comm_1
  {
    p4est_locidx_t local_num;
    int dir;
  } faces_comm_1_t;

  typedef struct faces_comm_2
  {
    p4est_locidx_t local_num[P4EST_FACES];
    int rank[P4EST_FACES];
  } faces_comm_2_t;

  const p4est_t *p4est;
  p4est_ghost_t *ghost;
  const my_p4est_brick_t *myb;
  my_p4est_cell_neighbors_t *ngbd_c;

  void init_faces();

public:
  /* the remote local number of the ghost velocities
   * ghost_local_num[P4EST_DIM]
   */
  vector<p4est_locidx_t> ghost_local_num[P4EST_DIM];

  /* q2f[P4EST_FACES][quad_idx] */
  vector<p4est_locidx_t> q2f_[P4EST_FACES];

  /* f2q[P4EST_DIM][u_idx]
   * e.g. u2q[1][12] is the quadrant whose face in the y direction has index 12
   */
  vector<my_p4est_faces_t::face_quad_ngbd> f2q_[P4EST_DIM];

  /* Store which process the ghost faces belong to.
   * Ghost y-face #i belongs to process nonlocal_ranks[0][i]
   */
  vector<int> nonlocal_ranks[P4EST_DIM];

  /* Store the number of owned faces for each rank.
   * Process #j owns global_owned_indeps[0][j] x-faces
   */
  vector<p4est_locidx_t> global_owned_indeps[P4EST_DIM];

  /* num_local[dir] contains the number of local faces in the direction "dir" */
  p4est_locidx_t num_local[P4EST_DIM];

  /* num_ghost[dir] contains the number of ghost faces in the direction "dir" */
  p4est_locidx_t num_ghost[P4EST_DIM];

  my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c);

  /*!
   * \brief q2f return the face of quadrant quad_idx in the direction dir
   * \param quad_idx the quadrant index in the local p4est
   * \param dir the direction of the face, dir::f_m00, dir::f_p00, dir::f_0m0 ...
   * \return the index of the face of quadrant quad_idx in direction dir, return NO_VELOCITY if there is many small neighbor quadrant in the direction dir
   */
  inline p4est_locidx_t q2f(p4est_locidx_t quad_idx, int dir) const
  {
    return q2f_[dir][quad_idx];
  }

  /*!
   * \brief get the local index of the neighbor cell
   * \param u_idx the index of the face
   * \param dir the cartesian direction of the face (dir::x, dir::y or dir::z)
   * \return the local index of the neighbor cell
   */
  inline void f2q(p4est_locidx_t f_idx, int dir, p4est_locidx_t& quad_idx, p4est_topidx_t& tree_idx) const
  {
    quad_idx = f2q_[dir][f_idx].quad_idx;
    tree_idx = f2q_[dir][f_idx].tree_idx;
  }

  /*!
   * \brief get the coordinates of the center of a face
   * \param f_idx the index of the face
   * \param dir the cartesian direction of the face (dir::x, dir::y or dir::z)
   * \return the coordinates of the center of face f_idx
   */
  double x_fr_f(p4est_locidx_t f_idx, int dir) const;
  double y_fr_f(p4est_locidx_t f_idx, int dir) const;
#ifdef P4_TO_P8
  double z_fr_f(p4est_locidx_t f_idx, int dir) const;
#endif

  void xyz_fr_f(p4est_locidx_t f_idx, int dir, double* xyz) const;
};

PetscErrorCode VecCreateGhostFaces     (const p4est_t *p4est, const my_p4est_faces_t *faces, Vec* v, int dir);
PetscErrorCode VecCreateGhostFacesBlock(const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, int dir);


/*!
 * \brief mark the faces that are well defined, i.e. that are solved for in an implicit poisson solve with irregular interface.
 *   For Dirichlet b.c. the condition is phi(face)<0. For Neumann, the control volume of the face must be at least partially in the negative domain.
 * \param p4est the forest
 * \param ngbd_n the node neighbors structure
 * \param faces the faces structure
 * \param dir the cartesian direction treated, dir::x, dir::y or dir::z
 * \param phi the level-set function
 * \param bc_type the type of boundary condition on the interface
 * \param is_well_defined a Vector the size of the number of faces in direction dir, to be filled
 */
void check_if_faces_are_well_defined(p4est_t *p4est, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, int dir,
                                     Vec phi, BoundaryConditionType bc_type, Vec is_well_defined);

#ifdef P4_TO_P8
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               p4est_locidx_t node_idx, Vec f, int dir,
                               Vec face_is_well_defined=NULL, int order=2, BoundaryConditions3D *bc=NULL);
#else
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               p4est_locidx_t node_idx, Vec f, int dir,
                               Vec face_is_well_defined=NULL, int order=2, BoundaryConditions2D *bc=NULL);
#endif

#endif /* MY_P4EST_FACES_H */
