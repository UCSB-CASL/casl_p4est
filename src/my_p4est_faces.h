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

  /* q2u[P4EST_FACES][quad_idx] */
  vector<p4est_locidx_t> q2u_[P4EST_FACES];

  /* u2q[P4EST_DIM][u_idx]
   * e.g. u2q[1][12] is the quadrant whose face in the y direction has index 12
   */
  vector<my_p4est_faces_t::face_quad_ngbd> u2q_[P4EST_DIM];

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

  inline p4est_locidx_t q2u(p4est_locidx_t quad_idx, int dir) const
  {
    return q2u_[dir][quad_idx];
  }

  /*!
   * \brief get the local index of the neighbor cell
   * \param u_idx the index of the face
   * \return the local index of the neighbor cell
   */
  inline void u2q(p4est_locidx_t u_idx, p4est_locidx_t& quad_idx, p4est_topidx_t& tree_idx) const
  {
    quad_idx = u2q_[0][u_idx].quad_idx;
    tree_idx = u2q_[0][u_idx].tree_idx;
  }


  /*!
  * \brief get the local index of the neighbor cell
  * \param v_idx the index of the face
  * \return the local index of the neighbor cell
  */
  inline void v2q(p4est_locidx_t v_idx, p4est_locidx_t& quad_idx, p4est_topidx_t& tree_idx) const
  {
    quad_idx = u2q_[1][v_idx].quad_idx;
    tree_idx = u2q_[1][v_idx].tree_idx;
  }


  /*!
  * \brief get the local index of the neighbor cell
  * \param w_idx the index of the face
  * \return the local index of the neighbor cell
  */
  inline void w2q(p4est_locidx_t w_idx, p4est_locidx_t& quad_idx, p4est_topidx_t& tree_idx) const
  {
    quad_idx = u2q_[2][w_idx].quad_idx;
    tree_idx = u2q_[2][w_idx].tree_idx;
  }

  double x_fr_u(p4est_locidx_t u_idx) const;
  double y_fr_u(p4est_locidx_t u_idx) const;
#ifdef P4_TO_P8
  double z_fr_u(p4est_locidx_t u_idx) const;
#endif

  double x_fr_v(p4est_locidx_t v_idx) const;
  double y_fr_v(p4est_locidx_t v_idx) const;
#ifdef P4_TO_P8
  double z_fr_v(p4est_locidx_t v_idx) const;
#endif

#ifdef P4_TO_P8
  double x_fr_w(p4est_locidx_t w_idx) const;
  double y_fr_w(p4est_locidx_t w_idx) const;
  double z_fr_w(p4est_locidx_t w_idx) const;
#endif

  void xyz_fr_u(p4est_locidx_t u_idx, double* xyz) const;
  void xyz_fr_v(p4est_locidx_t v_idx, double* xyz) const;
#ifdef P4_TO_P8
  void xyz_fr_w(p4est_locidx_t w_idx, double* xyz) const;
#endif
};



PetscErrorCode VecCreateGhostFaces     (const p4est_t *p4est, const my_p4est_faces_t *faces, Vec* v, int dir);
PetscErrorCode VecCreateGhostFacesBlock(const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, int dir);

#endif /* MY_P4EST_FACES_H */
