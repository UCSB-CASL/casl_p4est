#include "my_p4est_faces.h"


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
  for(int d=0; d<P4EST_FACES; ++d)
    q2u_[d].resize(p4est->local_num_quadrants + ghost->ghosts.elem_count, NO_VELOCITY);

  num_local_u = 0;
  num_local_v = 0;

  vector<p4est_quadrant_t> ngbd;
  vector< vector<faces_comm_1_t> > buff_query1(p4est->mpisize);
  vector< vector<p4est_locidx_t> > map1(p4est->mpisize);

  /* first process local velocities */
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;

      if(is_quad_xmWall(p4est, tree_idx, quad))
        q2u_[dir::f_m00][quad_idx] = num_local_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, -1, 0);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num <  p4est->local_num_quadrants && q2u_[dir::f_p00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_m00][quad_idx] = num_local_u++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_p00;
            buff_query1[r].push_back(c);
            map1[r].push_back(quad_idx);
          }
          else
            q2u_[dir::f_m00][quad_idx] = q2u_[dir::f_p00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_xpWall(p4est, tree_idx, quad))
        q2u_[dir::f_p00][quad_idx] = num_local_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2u_[dir::f_m00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_p00][quad_idx] = num_local_u++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_m00;
            buff_query1[r].push_back(c);
            map1[r].push_back(quad_idx);
          }
          else
            q2u_[dir::f_p00][quad_idx] = q2u_[dir::f_m00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ymWall(p4est, tree_idx, quad))
        q2u_[dir::f_0m0][quad_idx] = num_local_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, -1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2u_[dir::f_0p0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_0m0][quad_idx] = num_local_v++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_0p0;
            buff_query1[r].push_back(c);
            map1[r].push_back(quad_idx);
          }
          else
            q2u_[dir::f_0m0][quad_idx] = q2u_[dir::f_0p0][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ypWall(p4est, tree_idx, quad))
        q2u_[dir::f_0p0][quad_idx] = num_local_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level /* the neighbor is a bigger cell */
             || (ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants && q2u_[dir::f_0m0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
             || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_0p0][quad_idx] = num_local_v++;
          else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
          {
            p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
            int r = quad_find_ghost_owner(ghost, ghost_idx);
            const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
            faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = dir::f_0m0;
            buff_query1[r].push_back(c);
            map1[r].push_back(quad_idx);
          }
          else
            q2u_[dir::f_0p0][quad_idx] = q2u_[dir::f_0m0][ngbd[0].p.piggy3.local_num];
        }
      }
    }
  }

  /* send the queries to fill in the local information */
  vector<MPI_Request> req_query1(p4est->mpisize);
  for(int r=0; r<p4est->mpisize; ++r)
    MPI_Isend(&buff_query1[r][0], buff_query1[r].size()*sizeof(faces_comm_1_t), MPI_BYTE, r, 5, p4est->mpicomm, &req_query1[r]);

  vector<faces_comm_1_t> buff_recv_comm1;
  vector< vector<p4est_locidx_t> > buff_reply1_send(p4est->mpisize);
  vector<MPI_Request> req_reply1(p4est->mpisize);
  int nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 5, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(faces_comm_1_t);
    int r = status.MPI_SOURCE;

    buff_recv_comm1.resize(vec_size);
    MPI_Recv(&buff_recv_comm1[0], vec_size*sizeof(faces_comm_1_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);

    /* prepare the reply */
    buff_reply1_send[r].resize(buff_recv_comm1.size());
    for(unsigned int n=0; n<buff_recv_comm1.size(); ++n)
    {
      buff_reply1_send[r][n] = q2u_[buff_recv_comm1[n].dir][buff_recv_comm1[n].local_num];
    }

    /* send reply */
    MPI_Isend(&buff_reply1_send[r][0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, 6, p4est->mpicomm, &req_reply1[r]);

    nb_recv--;
  }
  buff_recv_comm1.clear();

  /* get the reply and fill in the missing local information */
  num_ghost_u = 0;
  num_ghost_v = 0;
  vector<p4est_locidx_t> buff_recv_locidx;
  nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 6, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);

    for(unsigned int n=0; n<buff_recv_locidx.size(); ++n)
    {
      switch(buff_query1[r][n].dir)
      {
      case dir::f_m00:
      case dir::f_p00:
        q2u_[buff_query1[r][n].dir==dir::f_m00 ? dir::f_p00 : dir::f_m00][map1[r][n]] = num_ghost_u+num_local_u;
        ghost_local_num[0].push_back(buff_recv_locidx[n]);
        num_ghost_u++;
        break;
      case dir::f_0m0:
      case dir::f_0p0:
        q2u_[buff_query1[r][n].dir==dir::f_0m0 ? dir::f_0p0 : dir::f_0m0][map1[r][n]] = num_ghost_v+num_local_v;
        ghost_local_num[1].push_back(buff_recv_locidx[n]);
        num_ghost_v++;
        break;
      }
    }

    nb_recv--;
  }
  buff_recv_locidx.clear();

  MPI_Waitall(req_query1.size(), &req_query1[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(req_reply1.size(), &req_reply1[0], MPI_STATUSES_IGNORE);






  /* now synchronize the ghost layer */
  vector< vector<p4est_locidx_t> > buff_query2(p4est->mpisize);
  for(size_t ghost_idx=0; ghost_idx<ghost->ghosts.elem_count; ++ghost_idx)
  {
    int r = quad_find_ghost_owner(ghost, ghost_idx);
    p4est_quadrant_t *g = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
    buff_query2[r].push_back(g->p.piggy3.local_num);
  }

  vector<MPI_Request> req_query2(p4est->mpisize);
  for(int r=0; r<p4est->mpisize; ++r)
  {
    MPI_Isend(&buff_query2[r][0], buff_query2[r].size()*sizeof(p4est_locidx_t), MPI_BYTE, r, 7, p4est->mpicomm, &req_query2[r]);
  }

  vector<MPI_Request> req_reply2(p4est->mpisize);
  vector< vector<faces_comm_2_t> > buff_reply2(p4est->mpisize);
  nb_recv = p4est->mpisize;
  while(nb_recv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 7, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);

    for(unsigned int q=0; q<buff_recv_locidx.size(); ++q)
    {
      faces_comm_2_t c;
      for(int dir=0; dir<P4EST_FACES; dir++)
        c.local_num[dir] = q2u_[dir][buff_recv_locidx[q]];
      buff_reply2[r].push_back(c);
    }

    MPI_Isend(&buff_reply2[r][0], buff_reply2[r].size()*sizeof(faces_comm_2_t), MPI_BYTE, r, 8, p4est->mpicomm, &req_reply2[r]);

    nb_recv--;
  }



  /* receive the ghost information and fill in the local info */
  nb_recv = p4est->mpisize;
  vector<faces_comm_2_t> buff_recv_comm2;
  while(nb_recv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 8, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(faces_comm_2_t);
    int r = status.MPI_SOURCE;

    buff_recv_comm2.resize(vec_size);
    MPI_Recv(&buff_recv_comm2[0], vec_size*sizeof(faces_comm_2_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);

    /* FILL IN LOCAL INFO */

    nb_recv--;
  }




  u2q_[0].resize(num_local_u + num_ghost_u);
  u2q_[1].resize(num_local_v + num_ghost_v);

  for(p4est_locidx_t quad_idx=0; quad_idx<p4est->local_num_quadrants; ++quad_idx)
  {
    if(q2u_[dir::f_m00][quad_idx] != NO_VELOCITY) u2q_[0][q2u_[dir::f_m00][quad_idx]].p = quad_idx;
    if(q2u_[dir::f_p00][quad_idx] != NO_VELOCITY) u2q_[0][q2u_[dir::f_p00][quad_idx]].m = quad_idx;
    if(q2u_[dir::f_0m0][quad_idx] != NO_VELOCITY) u2q_[1][q2u_[dir::f_0m0][quad_idx]].p = quad_idx;
    if(q2u_[dir::f_0p0][quad_idx] != NO_VELOCITY) u2q_[1][q2u_[dir::f_0p0][quad_idx]].m = quad_idx;
  }

//  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
//  {
//    p4est_locidx_t quad_idx = q+p4est->local_num_quadrants;

//    if(q2u_[dir::f_m00][quad_idx] != NO_VELOCITY) u2q_[0][q2u_[dir::f_m00][quad_idx]].p = quad_idx;
//    if(q2u_[dir::f_p00][quad_idx] != NO_VELOCITY) u2q_[0][q2u_[dir::f_p00][quad_idx]].m = quad_idx;
//    if(q2u_[dir::f_0m0][quad_idx] != NO_VELOCITY) u2q_[1][q2u_[dir::f_0m0][quad_idx]].p = quad_idx;
//    if(q2u_[dir::f_0p0][quad_idx] != NO_VELOCITY) u2q_[1][q2u_[dir::f_0p0][quad_idx]].m = quad_idx;
//  }
}

