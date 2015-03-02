#ifndef MY_P4EST_FACES_H
#define MY_P4EST_FACES_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_tools.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_utils.h>
#include <src/CASL_math.h>
#include <src/my_p4est_cell_neighbors.h>
#endif

#include <vector>

using std::vector;

#define NO_VELOCITY -1

class my_p4est_faces_t
{
private:
  typedef struct face_quad_ngbds
  {
    /* the indices of the neighbor quadrants. -1 if no neighbor in this direction.
     * otherwise, the local index of the quadrant is stored together with the index of the tree it belongs to.
     * Note that it is the local index in the forest (including ghosts), and not the index in the tree
     */
    p4est_locidx_t quad_idx[2];
    p4est_topidx_t tree_idx[2];
    face_quad_ngbds()
    {
      for(int i=0; i<P4EST_DIM; ++i)
      {
        quad_idx[i] = -1;
        tree_idx[i] = -1;
      }
    }
  } face_quad_ngbds;

  typedef struct faces_comm_1
  {
    p4est_locidx_t local_num;
    int dir;
  } faces_comm_1_t;

  typedef struct faces_comm_2
  {
    p4est_locidx_t local_num[P4EST_FACES];
  } faces_comm_2_t;

  const p4est_t *p4est;
  p4est_ghost_t *ghost;
  const my_p4est_brick_t *myb;
  const my_p4est_cell_neighbors_t *ngbd_c;

  size_t num_local_u;
  size_t num_ghost_u;

  size_t num_local_v;
  size_t num_ghost_v;

#ifdef P4_TO_P8
  size_t num_local_w;
  size_t num_ghost_w;
#endif

  /* the remote local number of the ghost velocities
   * ghost_local_num[P4EST_DIM]
   */
  vector<p4est_locidx_t> ghost_local_num[P4EST_DIM];

  /* q2u[P4EST_FACES][quad_idx] */
  vector<p4est_locidx_t> q2u_[P4EST_FACES];

  /* u2q[P4EST_DIM][u_idx].p/m
   * e.g. u2q[1][12].p is the top y face of quadrant 12
   */
  vector<my_p4est_faces_t::face_quad_ngbds> u2q_[P4EST_DIM];

  /* which processes do the ghost velocities belong to
   * proc_offsets[P4EST_DIM][mpisize+1]
   * proc_offsets[0][n] tells us the index where the x velocities (or u) of process n start.
   * note that proc_offsets[k][0] = 0, i.e. you have to add num_local_k to find the local index in the u2q list
   */
  vector< vector<p4est_locidx_t> > proc_offsets;

  void init_faces();

public:
  my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c);

  inline p4est_locidx_t q2u(p4est_locidx_t quad_idx, int dir)
  {
    return q2u_[dir][quad_idx];
  }

  /*!
   * \brief get the local index of the neighbor cell in the direction dir
   * \param u_idx the index of the face
   * \param dir 0 for the cell in the minus direction, 1 for the one in the plus direction
   * \return the local index of the neighbor cell in the direction dir
   */
  inline p4est_locidx_t u2q(p4est_locidx_t u_idx, bool dir)
  {
    return u2q_[0][u_idx].quad_idx[dir];
  }


  /*!
  * \brief get the local index of the neighbor cell in the direction dir
  * \param v_idx the index of the face
  * \param dir 0 for the cell in the minus direction, 1 for the one in the plus direction
  * \return the local index of the neighbor cell in the direction dir
  */
  inline p4est_locidx_t v2q(p4est_locidx_t v_idx, bool dir)
  {
    return u2q_[1][v_idx].quad_idx[dir];
  }
};



inline double face_x_fr_u(my_p4est_faces_t* faces, p4est_locidx_t u)
{
//  p4est_locidx_t qp = faces->u2q(u,1);
//  if(qp!=-1)
//  {
//    return quad_x_fr_q(qp, )
//  }
}


#endif /* MY_P4EST_FACES_H */
