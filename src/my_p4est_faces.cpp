#include "my_p4est_faces.h"

#include <algorithm>
#include <src/matrix.h>
#include <src/solve_lsqr.h>

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_faces_t;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif


my_p4est_faces_t::my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c)
{
  this->p4est = p4est;
  this->ghost = ghost;
  this->myb = myb;
  this->ngbd_c = ngbd_c;

  init_faces();
}

void my_p4est_faces_t::init_faces()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_faces_t, 0, 0, 0, 0); CHKERRXX(ierr);

  int mpiret;

  for(int d=0; d<P4EST_FACES; ++d)
    q2f_[d].resize(p4est->local_num_quadrants + ghost->ghosts.elem_count, NO_VELOCITY);

  for(int d=0; d<P4EST_DIM; ++d)
  {
    num_local[d] = 0;
    num_ghost[d] = 0;
    nonlocal_ranks[d].resize(0);
    ghost_local_num[d].resize(0);
  }

  vector<p4est_quadrant_t> ngbd;
  vector< vector<faces_comm_1_t> > buff_query1(p4est->mpisize);
  vector< vector<p4est_locidx_t> > map(p4est->mpisize);

  /* first process local velocities */
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;

      if(is_quad_xmWall(p4est, tree_idx, quad))
        q2f_[dir::f_m00][quad_idx] = num_local[0]++;
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, -1, 0, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, -1, 0);
#endif
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num <  p4est->local_num_quadrants && q2f_[dir::f_p00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_m00][quad_idx] = num_local[0]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_p00;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_m00][quad_idx] = q2f_[dir::f_p00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_xpWall(p4est, tree_idx, quad))
        q2f_[dir::f_p00][quad_idx] = num_local[0]++;
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
#endif
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2f_[dir::f_m00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_p00][quad_idx] = num_local[0]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_m00;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_p00][quad_idx] = q2f_[dir::f_m00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ymWall(p4est, tree_idx, quad))
        q2f_[dir::f_0m0][quad_idx] = num_local[1]++;
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, -1, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, -1);
#endif
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2f_[dir::f_0p0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_0m0][quad_idx] = num_local[1]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_0p0;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_0m0][quad_idx] = q2f_[dir::f_0p0][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ypWall(p4est, tree_idx, quad))
        q2f_[dir::f_0p0][quad_idx] = num_local[1]++;
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
#endif
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2f_[dir::f_0m0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_0p0][quad_idx] = num_local[1]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_0m0;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_0p0][quad_idx] = q2f_[dir::f_0m0][ngbd[0].p.piggy3.local_num];
        }
      }

#ifdef P4_TO_P8
      if(is_quad_zmWall(p4est, tree_idx, quad))
        q2f_[dir::f_00m][quad_idx] = num_local[2]++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2f_[dir::f_00p][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_00m][quad_idx] = num_local[2]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_00p;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_00m][quad_idx] = q2f_[dir::f_00p][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_zpWall(p4est, tree_idx, quad))
        q2f_[dir::f_00p][quad_idx] = num_local[2]++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level<quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2f_[dir::f_00m][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2f_[dir::f_00p][quad_idx] = num_local[2]++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_00m;
            buff_query1[r].push_back(c);
            map[r].push_back(quad_idx);
          }
          else
            q2f_[dir::f_00p][quad_idx] = q2f_[dir::f_00m][ngbd[0].p.piggy3.local_num];
        }
      }
#endif
    }
  }


  /* synchronize number of owned faces with the rest of the processes */
  for(int d=0; d<P4EST_DIM; ++d)
  {
    global_owned_indeps[d].resize(p4est->mpisize);
    global_owned_indeps[d][p4est->mpirank] = num_local[d];
    mpiret = MPI_Allgather(&num_local[d], 1, P4EST_MPI_LOCIDX, &global_owned_indeps[d][0], 1, P4EST_MPI_LOCIDX, p4est->mpicomm);
    SC_CHECK_MPI(mpiret);
  }


  /* send the queries to fill in the local information */
  vector<MPI_Request> req_query1(p4est->mpisize);
  for(int r=0; r<p4est->mpisize; ++r)
  {
    mpiret = MPI_Isend(&buff_query1[r][0], buff_query1[r].size()*sizeof(faces_comm_1_t), MPI_BYTE, r, 5, p4est->mpicomm, &req_query1[r]);
    SC_CHECK_MPI(mpiret);
  }

  vector<faces_comm_1_t> buff_recv_comm1;
  vector< vector<p4est_locidx_t> > buff_reply1_send(p4est->mpisize);
  vector<MPI_Request> req_reply1(p4est->mpisize);
  int nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 5, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    vec_size /= sizeof(faces_comm_1_t);
    int r = status.MPI_SOURCE;

    buff_recv_comm1.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_comm1[0], vec_size*sizeof(faces_comm_1_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    /* prepare the reply */
    buff_reply1_send[r].resize(buff_recv_comm1.size());
    for(unsigned int n=0; n<buff_recv_comm1.size(); ++n)
    {
      buff_reply1_send[r][n] = q2f_[buff_recv_comm1[n].dir][buff_recv_comm1[n].local_num];
    }

    /* send reply */
    mpiret = MPI_Isend(&buff_reply1_send[r][0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, 6, p4est->mpicomm, &req_reply1[r]);
    SC_CHECK_MPI(mpiret);

    nb_recv--;
  }
  buff_recv_comm1.clear();

  /* get the reply and fill in the missing local information */
  vector<p4est_locidx_t> buff_recv_locidx;
  nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 6, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    for(unsigned int n=0; n<buff_recv_locidx.size(); ++n)
    {
      switch(buff_query1[r][n].dir)
      {
      case dir::f_m00:
      case dir::f_p00:
        q2f_[buff_query1[r][n].dir==dir::f_m00 ? dir::f_p00 : dir::f_m00][map[r][n]] = num_ghost[0]+num_local[0];
        ghost_local_num[0].push_back(buff_recv_locidx[n]);
        nonlocal_ranks[0].push_back(r);
        num_ghost[0]++;
        break;
      case dir::f_0m0:
      case dir::f_0p0:
        q2f_[buff_query1[r][n].dir==dir::f_0m0 ? dir::f_0p0 : dir::f_0m0][map[r][n]] = num_ghost[1]+num_local[1];
        ghost_local_num[1].push_back(buff_recv_locidx[n]);
        nonlocal_ranks[1].push_back(r);
        num_ghost[1]++;
        break;
#ifdef P4_TO_P8
      case dir::f_00m:
      case dir::f_00p:
        q2f_[buff_query1[r][n].dir==dir::f_00m ? dir::f_00p : dir::f_00m][map[r][n]] = num_ghost[2]+num_local[2];
        ghost_local_num[2].push_back(buff_recv_locidx[n]);
        nonlocal_ranks[2].push_back(r);
        num_ghost[2]++;
        break;
#endif
      }
    }

    nb_recv--;
  }
  buff_recv_locidx.clear();

  mpiret = MPI_Waitall(req_query1.size(), &req_query1[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(req_reply1.size(), &req_reply1[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);


  /* now synchronize the ghost layer */
  vector< vector<p4est_locidx_t> > buff_query2(p4est->mpisize);
  for(int r=0; r<p4est->mpisize; ++r)
    map[r].clear();
  for(size_t ghost_idx=0; ghost_idx<ghost->ghosts.elem_count; ++ghost_idx)
  {
    int r = quad_find_ghost_owner(ghost, ghost_idx);
    p4est_quadrant_t *g = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
    buff_query2[r].push_back(g->p.piggy3.local_num);
    map[r].push_back(ghost_idx);
  }


  vector<MPI_Request> req_query2(p4est->mpisize);
  for(int r=0; r<p4est->mpisize; ++r)
  {
    mpiret = MPI_Isend(&buff_query2[r][0], buff_query2[r].size()*sizeof(p4est_locidx_t), MPI_BYTE, r, 7, p4est->mpicomm, &req_query2[r]);
    SC_CHECK_MPI(mpiret);
  }

  vector<MPI_Request> req_reply2(p4est->mpisize);
  vector< vector<faces_comm_2_t> > buff_reply2(p4est->mpisize);
  nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 7, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    for(unsigned int q=0; q<buff_recv_locidx.size(); ++q)
    {
      faces_comm_2_t c;
      for(int dir=0; dir<P4EST_FACES; dir++)
      {
        p4est_locidx_t u_tmp = q2f_[dir][buff_recv_locidx[q]];
        /* local value */
        if(u_tmp<num_local[dir/2])
        {
          c.rank[dir] = p4est->mpirank;
          c.local_num[dir] = u_tmp;
        }
        /* ghost value */
        else
        {
          c.rank[dir] = nonlocal_ranks[dir/2][u_tmp-num_local[dir/2]];
          c.local_num[dir] = ghost_local_num[dir/2][u_tmp-num_local[dir/2]];
        }
      }
      buff_reply2[r].push_back(c);
    }

    mpiret = MPI_Isend(&buff_reply2[r][0], buff_reply2[r].size()*sizeof(faces_comm_2_t), MPI_BYTE, r, 8, p4est->mpicomm, &req_reply2[r]);
    SC_CHECK_MPI(mpiret);

    nb_recv--;
  }


  /* receive the ghost information and fill in the local info */
  nb_recv = p4est->mpisize;
  vector<faces_comm_2_t> buff_recv_comm2;
  while(nb_recv>0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 8, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    vec_size /= sizeof(faces_comm_2_t);
    int r = status.MPI_SOURCE;

    buff_recv_comm2.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_comm2[0], vec_size*sizeof(faces_comm_2_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    /* FILL IN LOCAL INFO */
    for(int n=0; n<vec_size; ++n)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, map[r][n]);
      p4est_locidx_t quad_idx = map[r][n]+p4est->local_num_quadrants;
      p4est_topidx_t tree_idx = quad->p.piggy3.which_tree;

      if(is_quad_xmWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_m00][quad_idx] = num_local[0] + num_ghost[0];
        ghost_local_num[0].push_back(buff_recv_comm2[n].local_num[dir::f_m00]);
        nonlocal_ranks[0].push_back(buff_recv_comm2[n].rank[dir::f_m00]);
        num_ghost[0]++;
      }
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
#endif
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_m00]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_p00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_m00][quad_idx] = num_local[0] + num_ghost[0];
            ghost_local_num[0].push_back(buff_recv_comm2[n].local_num[dir::f_m00]);
            nonlocal_ranks[0].push_back(buff_recv_comm2[n].rank[dir::f_m00]);
            num_ghost[0]++;
          }
          else
          {
            q2f_[dir::f_m00][quad_idx] = q2f_[dir::f_p00][ngbd[0].p.piggy3.local_num];
          }
        }
      }

      if(is_quad_xpWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_p00][quad_idx] = num_local[0] + num_ghost[0];
        ghost_local_num[0].push_back(buff_recv_comm2[n].local_num[dir::f_p00]);
        nonlocal_ranks[0].push_back(buff_recv_comm2[n].rank[dir::f_p00]);
        num_ghost[0]++;
      }
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
#endif
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_p00]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_m00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_p00][quad_idx] = num_local[0] + num_ghost[0];
            ghost_local_num[0].push_back(buff_recv_comm2[n].local_num[dir::f_p00]);
            nonlocal_ranks[0].push_back(buff_recv_comm2[n].rank[dir::f_p00]);
            num_ghost[0]++;
          }
          else
          {
            q2f_[dir::f_p00][quad_idx] = q2f_[dir::f_m00][ngbd[0].p.piggy3.local_num];
          }
        }
      }

      if(is_quad_ymWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_0m0][quad_idx] = num_local[1] + num_ghost[1];
        ghost_local_num[1].push_back(buff_recv_comm2[n].local_num[dir::f_0m0]);
        nonlocal_ranks[1].push_back(buff_recv_comm2[n].rank[dir::f_0m0]);
        num_ghost[1]++;
      }
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1);
#endif
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_0m0]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_0p0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_0m0][quad_idx] = num_local[1] + num_ghost[1];
            ghost_local_num[1].push_back(buff_recv_comm2[n].local_num[dir::f_0m0]);
            nonlocal_ranks[1].push_back(buff_recv_comm2[n].rank[dir::f_0m0]);
            num_ghost[1]++;
          }
          else
          {
            q2f_[dir::f_0m0][quad_idx] = q2f_[dir::f_0p0][ngbd[0].p.piggy3.local_num];
          }
        }
      }

      if(is_quad_ypWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_0p0][quad_idx] = num_local[1] + num_ghost[1];
        ghost_local_num[1].push_back(buff_recv_comm2[n].local_num[dir::f_0p0]);
        nonlocal_ranks[1].push_back(buff_recv_comm2[n].rank[dir::f_0p0]);
        num_ghost[1]++;
      }
      else
      {
        ngbd.resize(0);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
#endif
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_0p0]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_0m0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_0p0][quad_idx] = num_local[1] + num_ghost[1];
            ghost_local_num[1].push_back(buff_recv_comm2[n].local_num[dir::f_0p0]);
            nonlocal_ranks[1].push_back(buff_recv_comm2[n].rank[dir::f_0p0]);
            num_ghost[1]++;
          }
          else
          {
            q2f_[dir::f_0p0][quad_idx] = q2f_[dir::f_0m0][ngbd[0].p.piggy3.local_num];
          }
        }
      }

#ifdef P4_TO_P8
      if(is_quad_zmWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_00m][quad_idx] = num_local[2] + num_ghost[2];
        ghost_local_num[2].push_back(buff_recv_comm2[n].local_num[dir::f_00m]);
        nonlocal_ranks[2].push_back(buff_recv_comm2[n].rank[dir::f_00m]);
        num_ghost[2]++;
      }
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_00m]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_00p][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_00m][quad_idx] = num_local[2] + num_ghost[2];
            ghost_local_num[2].push_back(buff_recv_comm2[n].local_num[dir::f_00m]);
            nonlocal_ranks[2].push_back(buff_recv_comm2[n].rank[dir::f_00m]);
            num_ghost[2]++;
          }
          else
          {
            q2f_[dir::f_00m][quad_idx] = q2f_[dir::f_00p][ngbd[0].p.piggy3.local_num];
          }
        }
      }

      if(is_quad_zpWall(p4est, tree_idx, quad))
      {
        q2f_[dir::f_00p][quad_idx] = num_local[2] + num_ghost[2];
        ghost_local_num[2].push_back(buff_recv_comm2[n].local_num[dir::f_00p]);
        nonlocal_ranks[2].push_back(buff_recv_comm2[n].rank[dir::f_00p]);
        num_ghost[2]++;
      }
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
        if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[dir::f_00p]!=NO_VELOCITY)
        {
          if(ngbd.size()==0 || q2f_[dir::f_00m][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
          {
            q2f_[dir::f_00p][quad_idx] = num_local[2] + num_ghost[2];
            ghost_local_num[2].push_back(buff_recv_comm2[n].local_num[dir::f_00p]);
            nonlocal_ranks[2].push_back(buff_recv_comm2[n].rank[dir::f_00p]);
            num_ghost[2]++;
          }
          else
          {
            q2f_[dir::f_00p][quad_idx] = q2f_[dir::f_00m][ngbd[0].p.piggy3.local_num];
          }
        }
      }
#endif

    }
    nb_recv--;
  }


  /* now construct the velocity to quadrant link */
  for(int d=0; d<P4EST_DIM; ++d)
    f2q_[d].resize(num_local[d] + num_ghost[d]);

  for(int dir=0; dir<P4EST_FACES; dir++)
  {
    for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for(size_t q=0; q<tree->quadrants.elem_count; ++q)
      {
        p4est_topidx_t quad_idx = q+tree->quadrants_offset;
        if(q2f_[dir][quad_idx] != NO_VELOCITY)
        {
          f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx = quad_idx;
          f2q_[dir/2][q2f_[dir][quad_idx]].tree_idx = tree_idx;
        }
      }
    }
  }

  for(int dir=0; dir<P4EST_FACES; dir++)
  {
    for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q+p4est->local_num_quadrants;
      if(q2f_[dir][quad_idx] != NO_VELOCITY && f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx == -1)
        f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx = quad_idx;
    }
  }

//  check data for debugging
//  for(int dir=0; dir<P4EST_DIM; ++dir)
//  {
//    for(unsigned int i=0; i<f2q_[dir].size(); ++i)
//    {
//      if(f2q_[dir][i].quad_idx==-1)                                                      std::cout << p4est->mpirank << " problem in " << dir << " !!" << std::endl;
//      if(f2q_[dir][i].quad_idx< p4est->local_num_quadrants && f2q_[dir][i].tree_idx==-1) std::cout << p4est->mpirank << " problem in " << dir << " local !!" << std::endl;
//      if(f2q_[dir][i].quad_idx>=p4est->local_num_quadrants && f2q_[dir][i].tree_idx!=-1) std::cout << p4est->mpirank << " problem in " << dir << " ghost !!" << std::endl;
//    }
//  }

  mpiret = MPI_Waitall(req_query2.size(), &req_query2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(req_reply2.size(), &req_reply2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  ierr = PetscLogEventEnd(log_my_p4est_faces_t, 0, 0, 0, 0); CHKERRXX(ierr);
}



double my_p4est_faces_t::x_fr_f(p4est_locidx_t f_idx, int dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx>=0);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
  {
    quad = (p4est_quadrant_t*) sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
  double x = quad_x_fr_i(quad) + tree_xmin;
  if(dir!=dir::x)
    x += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_p00)==f_idx)
    x += (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

  return x;
}


double my_p4est_faces_t::y_fr_f(p4est_locidx_t f_idx, int dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx>=0);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
  {
    quad = (p4est_quadrant_t*) sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
  double y = quad_y_fr_j(quad) + tree_ymin;
  if(dir!=dir::y)
    y += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_0p0)==f_idx)
    y += (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

  return y;
}


#ifdef P4_TO_P8
double my_p4est_faces_t::z_fr_f(p4est_locidx_t f_idx, int dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx>=0);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
  {
    quad = (p4est_quadrant_t*) sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
  double z = quad_z_fr_k(quad) + tree_zmin;
  if(dir!=dir::z)
    z += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_00p)==f_idx)
    z += (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

  return z;
}
#endif



void my_p4est_faces_t::xyz_fr_f(p4est_locidx_t f_idx, int dir, double* xyz) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx>=0);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
  {
    quad = (p4est_quadrant_t*) sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_xyz_min[P4EST_DIM];
  for(int i=0; i<P4EST_DIM; ++i)
    tree_xyz_min[i] = p4est->connectivity->vertices[3*v_mmm + i];

  xyz[0] = quad_x_fr_i(quad) + tree_xyz_min[0];
  xyz[1] = quad_y_fr_j(quad) + tree_xyz_min[1];
#ifdef P4_TO_P8
  xyz[2] = quad_z_fr_k(quad) + tree_xyz_min[2];
#endif

  if(dir!=dir::x)                           xyz[0] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_p00)==f_idx) xyz[0] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  if(dir!=dir::y)                           xyz[1] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_0p0)==f_idx) xyz[1] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
#ifdef P4_TO_P8
  if(dir!=dir::z)                           xyz[2] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else if(q2f(quad_idx, dir::f_00p)==f_idx) xyz[2] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
#endif
}



PetscErrorCode VecCreateGhostFaces(const p4est_t *p4est, const my_p4est_faces_t *faces, Vec* v, int dir)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = faces->num_local[dir];

  std::vector<PetscInt> ghost_faces(faces->num_ghost[dir], 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)faces->global_owned_indeps[dir][r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for(size_t i=0; i<ghost_faces.size(); ++i)
    ghost_faces[i] = faces->ghost_local_num[dir][i] + global_offset_sum[faces->nonlocal_ranks[dir][i]];

  ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                        ghost_faces.size(), (const PetscInt*)&ghost_faces[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode VecCreateGhostFacesBlock(const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, int dir)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = faces->num_local[dir];

  std::vector<PetscInt> ghost_faces(faces->num_ghost[dir], 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)faces->global_owned_indeps[dir][r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for(size_t i=0; i<ghost_faces.size(); ++i)
    ghost_faces[i] = faces->ghost_local_num[dir][i] + global_offset_sum[faces->nonlocal_ranks[dir][i]];

  ierr = VecCreateGhostBlock(p4est->mpicomm,
                             block_size, num_local*block_size, num_global*block_size,
                             ghost_faces.size(), (const PetscInt*)&ghost_faces[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}




#ifdef P4_TO_P8
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               Vec f, int dir, p4est_locidx_t node_idx,
                               Vec phi, BoundaryConditionType bc_type, BoundaryConditions3D *bc)
#else
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               Vec f, int dir, p4est_locidx_t node_idx,
                               Vec phi, BoundaryConditionType bc_type, BoundaryConditions2D *bc)
#endif
{
#ifdef CASL_THROWS
  if(node_idx>nodes->num_owned_indeps) throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node_n: cannot be called on a ghost node.");
#endif

  PetscErrorCode ierr;

  double xyz[P4EST_DIM];
  xyz[0] = node_x_fr_n(node_idx, p4est, nodes);
  xyz[1] = node_y_fr_n(node_idx, p4est, nodes);
#ifdef P4_TO_P8
  xyz[2] = node_z_fr_n(node_idx, p4est, nodes);
#endif

  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);

#ifdef P4_TO_P8
  if(bc!=NULL && is_node_Wall(p4est, node) && bc[dir].wallType(xyz[0],xyz[1],xyz[2])==DIRICHLET)
    return bc[dir].wallValue(xyz[0],xyz[1],xyz[2]);
#else
  if(bc!=NULL && is_node_Wall(p4est, node) && bc[dir].wallType(xyz[0],xyz[1])==DIRICHLET)
    return bc[dir].wallValue(xyz[0],xyz[1]);
#endif

  /* gather the neighborhood */
  vector<p4est_quadrant_t> ngbd;
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  double scaling = DBL_MAX;
  for(int i=-1; i<2; i+=2)
    for(int j=-1; j<2; j+=2)
#ifdef P4_TO_P8
      for(int k=-1; k<2; ++k)
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx, i, j, k, quad_idx, tree_idx);
#else
    {
      ngbd_n->find_neighbor_cell_of_node(node_idx, i, j, quad_idx, tree_idx);
#endif
      if(quad_idx!=-1)
      {
        p4est_quadrant_t quad;
        if(quad_idx<p4est->local_num_quadrants)
        {
          p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
          quad = *(p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
          quad.p.piggy3.which_tree = tree_idx;
          quad.p.piggy3.local_num = quad_idx;
        }
        else
        {
          quad = *(p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
          quad.p.piggy3.local_num = quad_idx;
        }

        scaling = MIN(scaling, (double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN);

        ngbd.push_back(quad);
#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, j, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, j, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, 0, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, j, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, j, k);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, j);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, j);
#endif
      }
    }

  double *f_p;
  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);


  vector<p4est_locidx_t> interp_points;
  matrix_t A;
#ifdef P4_TO_P8
  A.resize(1,10);
#else
  A.resize(1,6);
#endif
  vector<double> p;
  vector<double> nb[P4EST_DIM];

  scaling *= .5;
  double min_w = 1e-6;
  double inv_max_w = 1e-6;
  vector<p4est_quadrant_t> ngbd_tmp;

  for(unsigned int m=0; m<ngbd.size(); m++)
  {
    quad_idx = ngbd[m].p.piggy3.local_num;
    tree_idx = ngbd[m].p.piggy3.which_tree;
    double phi_tmp[P4EST_CHILDREN];
    for(int d=0; d<P4EST_CHILDREN; ++d)
      phi_tmp[d] = phi_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + d]];

    /* minus direction */
    p4est_locidx_t fm_idx = faces->q2f(quad_idx, 2*dir);
    if(fm_idx!=NO_VELOCITY)
    {
      /* for dirichlet interface */
      double phi_m;
      switch(dir)
      {
#ifdef P4_TO_P8
      case dir::x: phi_m = (phi_tmp[dir::v_mmm] + phi_tmp[dir::v_mpm] + phi_tmp[dir::v_mmp] + phi_tmp[dir::v_mpp]) / 4; break;
      case dir::y: phi_m = (phi_tmp[dir::v_mmm] + phi_tmp[dir::v_pmm] + phi_tmp[dir::v_mmp] + phi_tmp[dir::v_pmp]) / 4; break;
      case dir::z: phi_m = (phi_tmp[dir::v_mmm] + phi_tmp[dir::v_pmm] + phi_tmp[dir::v_mpm] + phi_tmp[dir::v_ppm]) / 4; break;
#else
      case dir::x: phi_m = (phi_tmp[dir::v_mmm] + phi_tmp[dir::v_mpm]) / 2; break;
      case dir::y: phi_m = (phi_tmp[dir::v_mmm] + phi_tmp[dir::v_pmm]) / 2; break;
#endif
      default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
      }

      /* for neumann interface */
      bool is_neg = true;
      if(bc_type==NEUMANN)
      {
        double phi_N[P4EST_CHILDREN];
        if(is_quad_Wall(p4est, tree_idx, &ngbd[m], 2*dir))
        {
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::x : phi_N[0] = phi_tmp[dir::v_mmm]; phi_N[1] = phi_tmp[dir::v_mpm]; phi_N[2] = phi_tmp[dir::v_mmp]; phi_N[3] = phi_tmp[dir::v_mpp]; break;
          case dir::y : phi_N[0] = phi_tmp[dir::v_mmm]; phi_N[1] = phi_tmp[dir::v_pmm]; phi_N[2] = phi_tmp[dir::v_mmp]; phi_N[3] = phi_tmp[dir::v_pmp]; break;
          case dir::z : phi_N[0] = phi_tmp[dir::v_mmm]; phi_N[1] = phi_tmp[dir::v_pmm]; phi_N[2] = phi_tmp[dir::v_mpm]; phi_N[3] = phi_tmp[dir::v_ppm]; break;
#else
          case dir::x : phi_N[0] = phi_tmp[dir::v_mmm]; phi_N[1] = phi_tmp[dir::v_mpm]; break;
          case dir::y : phi_N[0] = phi_tmp[dir::v_mmm]; phi_N[1] = phi_tmp[dir::v_pmm]; break;
#endif
          default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
          }
        }
        else
        {
          ngbd_tmp.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 2*dir);
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::x:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmm]] + phi_tmp[dir::v_mmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpm]] + phi_tmp[dir::v_mpm])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmp]] + phi_tmp[dir::v_mmp])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpp]] + phi_tmp[dir::v_mpp])/2;
            break;
          case dir::y:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmm]] + phi_tmp[dir::v_mmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmm]] + phi_tmp[dir::v_pmm])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmp]] + phi_tmp[dir::v_mmp])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmp]] + phi_tmp[dir::v_pmp])/2;
            break;
          case dir::z:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmm]] + phi_tmp[dir::v_mmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmm]] + phi_tmp[dir::v_pmm])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpm]] + phi_tmp[dir::v_mpm])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppm]] + phi_tmp[dir::v_ppm])/2;
            break;
#else
          case dir::x:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmm]] + phi_tmp[dir::v_mmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpm]] + phi_tmp[dir::v_mpm])/2;
            break;
          case dir::y:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmm]] + phi_tmp[dir::v_mmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmm]] + phi_tmp[dir::v_pmm])/2;
            break;
#endif
          default:
            throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
          }
        }

        switch(dir)
        {
#ifdef P4_TO_P8
        case dir::x:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_pmm])/2; phi_N[5] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_ppm])/2;
          phi_N[6] = (phi_tmp[dir::v_mmp]+phi_tmp[dir::v_pmp])/2; phi_N[7] = (phi_tmp[dir::v_mpp]+phi_tmp[dir::v_ppp])/2;
          break;
        case dir::y:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mpm])/2; phi_N[5] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_ppm])/2;
          phi_N[6] = (phi_tmp[dir::v_mmp]+phi_tmp[dir::v_mpp])/2; phi_N[7] = (phi_tmp[dir::v_pmp]+phi_tmp[dir::v_ppp])/2;
          break;
        case dir::z:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mmp])/2; phi_N[5] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_pmp])/2;
          phi_N[6] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_mpp])/2; phi_N[7] = (phi_tmp[dir::v_ppm]+phi_tmp[dir::v_ppp])/2;
          break;
#else
        case dir::x: phi_N[2] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_pmm])/2; phi_N[3] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_ppm])/2; break;
        case dir::y: phi_N[2] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mpm])/2; phi_N[3] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_ppm])/2; break;
#endif
        default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
        }

        is_neg = false;
        for(int i=0; i<P4EST_CHILDREN; ++i)
          is_neg = is_neg || (phi_N[i]<0);
      }

      if(( bc_type==NOINTERFACE ||
           (bc_type==DIRICHLET && phi_m<0) ||
           (bc_type==NEUMANN && is_neg) )
         && std::find(interp_points.begin(), interp_points.end(),fm_idx)==interp_points.end() )
      {
        double xyz_t[P4EST_DIM];
        faces->xyz_fr_f(fm_idx, dir, xyz_t);
        for(int i=0; i<P4EST_DIM; ++i)
          xyz_t[i] = (xyz[i] - xyz_t[i]) / scaling;

#ifdef P4_TO_P8
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif

#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[2]          * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 5, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 6, xyz_t[0]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 7, xyz_t[1]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 8, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 9, xyz_t[2]*xyz_t[2] * w);
#else
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 5, xyz_t[1]*xyz_t[1] * w);
#endif

        p.push_back(f_p[fm_idx] * w);

        for(int d=0; d<P4EST_DIM; ++d)
          if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
            nb[d].push_back(xyz_t[d]);

        interp_points.push_back(fm_idx);
      }
    }




    /* plus direction */
    fm_idx = faces->q2f(quad_idx, 2*dir+1);
    if(fm_idx!=NO_VELOCITY)
    {
      /* for dirichlet interface */
      double phi_m;
      switch(dir)
      {
#ifdef P4_TO_P8
      case dir::x: phi_m = (phi_tmp[dir::v_pmm] + phi_tmp[dir::v_ppm] + phi_tmp[dir::v_pmp] + phi_tmp[dir::v_ppp]) / 4; break;
      case dir::y: phi_m = (phi_tmp[dir::v_mpm] + phi_tmp[dir::v_ppm] + phi_tmp[dir::v_mpp] + phi_tmp[dir::v_ppp]) / 4; break;
      case dir::z: phi_m = (phi_tmp[dir::v_mmp] + phi_tmp[dir::v_pmp] + phi_tmp[dir::v_mpp] + phi_tmp[dir::v_ppp]) / 4; break;
#else
      case dir::x: phi_m = (phi_tmp[dir::v_pmm] + phi_tmp[dir::v_ppm]) / 2; break;
      case dir::y: phi_m = (phi_tmp[dir::v_mpm] + phi_tmp[dir::v_ppm]) / 2; break;
#endif
      default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
      }

      /* for neumann interface */
      bool is_neg = true;
      if(bc_type==NEUMANN)
      {
        double phi_N[P4EST_CHILDREN];
        if(is_quad_Wall(p4est, tree_idx, &ngbd[m], 2*dir+1))
        {
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::x : phi_N[0] = phi_tmp[dir::v_pmm]; phi_N[1] = phi_tmp[dir::v_ppm]; phi_N[2] = phi_tmp[dir::v_pmp]; phi_N[3] = phi_tmp[dir::v_ppp]; break;
          case dir::y : phi_N[0] = phi_tmp[dir::v_mpm]; phi_N[1] = phi_tmp[dir::v_ppm]; phi_N[2] = phi_tmp[dir::v_mpp]; phi_N[3] = phi_tmp[dir::v_ppp]; break;
          case dir::z : phi_N[0] = phi_tmp[dir::v_mmp]; phi_N[1] = phi_tmp[dir::v_pmp]; phi_N[2] = phi_tmp[dir::v_mpp]; phi_N[3] = phi_tmp[dir::v_ppp]; break;
#else
          case dir::x : phi_N[0] = phi_tmp[dir::v_pmm]; phi_N[1] = phi_tmp[dir::v_ppm]; break;
          case dir::y : phi_N[0] = phi_tmp[dir::v_mpm]; phi_N[1] = phi_tmp[dir::v_ppm]; break;
#endif
          default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
          }
        }
        else
        {
          ngbd_tmp.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 2*dir+1);
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::x:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmm]] + phi_tmp[dir::v_pmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppm]] + phi_tmp[dir::v_ppm])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmp]] + phi_tmp[dir::v_pmp])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppp]] + phi_tmp[dir::v_ppp])/2;
            break;
          case dir::y:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpm]] + phi_tmp[dir::v_mpm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppm]] + phi_tmp[dir::v_ppm])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpp]] + phi_tmp[dir::v_mpp])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppp]] + phi_tmp[dir::v_ppp])/2;
            break;
          case dir::z:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mmp]] + phi_tmp[dir::v_mmp])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmp]] + phi_tmp[dir::v_pmp])/2;
            phi_N[2] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpp]] + phi_tmp[dir::v_mpp])/2;
            phi_N[3] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppp]] + phi_tmp[dir::v_ppp])/2;
            break;
#else
          case dir::x:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_pmm]] + phi_tmp[dir::v_pmm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppm]] + phi_tmp[dir::v_ppm])/2;
            break;
          case dir::y:
            phi_N[0] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_mpm]] + phi_tmp[dir::v_mpm])/2;
            phi_N[1] = (phi_p[nodes->local_nodes[P4EST_CHILDREN*ngbd_tmp[0].p.piggy3.local_num + dir::v_ppm]] + phi_tmp[dir::v_ppm])/2;
            break;
#endif
          default:
            throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
          }
        }

        switch(dir)
        {
#ifdef P4_TO_P8
        case dir::x:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_pmm])/2; phi_N[5] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_ppm])/2;
          phi_N[6] = (phi_tmp[dir::v_mmp]+phi_tmp[dir::v_pmp])/2; phi_N[7] = (phi_tmp[dir::v_mpp]+phi_tmp[dir::v_ppp])/2;
          break;
        case dir::y:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mpm])/2; phi_N[5] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_ppm])/2;
          phi_N[6] = (phi_tmp[dir::v_mmp]+phi_tmp[dir::v_mpp])/2; phi_N[7] = (phi_tmp[dir::v_pmp]+phi_tmp[dir::v_ppp])/2;
          break;
        case dir::z:
          phi_N[4] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mmp])/2; phi_N[5] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_pmp])/2;
          phi_N[6] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_mpp])/2; phi_N[7] = (phi_tmp[dir::v_ppm]+phi_tmp[dir::v_ppp])/2;
          break;
#else
        case dir::x: phi_N[2] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_pmm])/2; phi_N[3] = (phi_tmp[dir::v_mpm]+phi_tmp[dir::v_ppm])/2; break;
        case dir::y: phi_N[2] = (phi_tmp[dir::v_mmm]+phi_tmp[dir::v_mpm])/2; phi_N[3] = (phi_tmp[dir::v_pmm]+phi_tmp[dir::v_ppm])/2; break;
#endif
        default: throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node: unknown direction.");
        }

        is_neg = false;
        for(int i=0; i<P4EST_CHILDREN; ++i)
          is_neg = is_neg || (phi_N[i]<0);
      }

      if(( bc_type==NOINTERFACE ||
           (bc_type==DIRICHLET && phi_m<0) ||
           (bc_type==NEUMANN && is_neg) )
         && std::find(interp_points.begin(), interp_points.end(),fm_idx)==interp_points.end() )
      {
        double xyz_t[P4EST_DIM];
        faces->xyz_fr_f(fm_idx, dir, xyz_t);
        for(int i=0; i<P4EST_DIM; ++i)
          xyz_t[i] = (xyz[i] - xyz_t[i]) / scaling;

#ifdef P4_TO_P8
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif

#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[2]          * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 5, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 6, xyz_t[0]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 7, xyz_t[1]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 8, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 9, xyz_t[2]*xyz_t[2] * w);
#else
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 5, xyz_t[1]*xyz_t[1] * w);
#endif

        p.push_back(f_p[fm_idx] * w);

        for(int d=0; d<P4EST_DIM; ++d)
          if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
            nb[d].push_back(xyz_t[d]);

        interp_points.push_back(fm_idx);
      }
    }
  }


  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if(interp_points.size()==0)
    return 0;

  A.scale_by_maxabs(p);

  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size());
}
