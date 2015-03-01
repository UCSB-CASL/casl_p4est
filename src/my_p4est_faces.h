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
  const p4est_t *p4est;
  const p4est_ghost_t *ghost;
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

//  vector<p4est_locidx_t> um;
//  vector<p4est_locidx_t> up;
//  vector<p4est_locidx_t> vm;
//  vector<p4est_locidx_t> vp;
  vector< vector<p4est_locidx_t> > q2u_;

  vector< vector<p4est_locidx_t> > u2q_;

  void init_faces();

public:
  my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c);

  inline p4est_locidx_t q2u(p4est_locidx_t quad_idx, int dir)
  {
    return q2u_[dir][quad_idx];
  }
};





#endif /* MY_P4EST_FACES_H */
