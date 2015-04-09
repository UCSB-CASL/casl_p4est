#ifndef MY_P4EST_FACES_H
#define MY_P4EST_FACES_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/CASL_math.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_cell_neighbors.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/CASL_math.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_cell_neighbors.h>
#endif

#include <vector>

using std::vector;

#define NO_VELOCITY -1

class my_p4est_faces_t
{
  friend class PoissonSolverFaces;
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
  const my_p4est_cell_neighbors_t *ngbd_c;

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

#endif /* MY_P4EST_FACES_H */
