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

  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_topidx_t quad_idx = q+tree->quadrants_offset;
      for(int dir=0; dir<P4EST_DIM; dir++)
      if(q2f_[dir][quad_idx] != NO_VELOCITY)
      {
        f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx = quad_idx;
        f2q_[dir/2][q2f_[dir][quad_idx]].tree_idx = tree_idx;
      }
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    p4est_locidx_t quad_idx = q+p4est->local_num_quadrants;
    for(int dir=0; dir<P4EST_DIM; dir++)
      if(q2f_[dir][quad_idx] != NO_VELOCITY && f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx == -1)
        f2q_[dir/2][q2f_[dir][quad_idx]].quad_idx = quad_idx;
  }




  // check data for debugging
//  for(unsigned int i=0; i<f2q_[0].size(); ++i)
//  {
//    if(f2q_[0][i].quad_idx==-1)                                                    std::cout << p4est->mpirank << " problem in u !!" << std::endl;
//    if(f2q_[0][i].quad_idx< p4est->local_num_quadrants && f2q_[0][i].tree_idx==-1) std::cout << p4est->mpirank << " problem in u local !!" << std::endl;
//    if(f2q_[0][i].quad_idx>=p4est->local_num_quadrants && f2q_[0][i].tree_idx!=-1) std::cout << p4est->mpirank << " problem in u ghost !!" << std::endl;
//  }
//  for(unsigned int i=0; i<f2q_[1].size(); ++i)
//  {
//    if(f2q_[1][i].quad_idx==-1)                                                    std::cout << p4est->mpirank << " problem in v !!" << std::endl;
//    if(f2q_[1][i].quad_idx< p4est->local_num_quadrants && f2q_[1][i].tree_idx==-1) std::cout << p4est->mpirank << " problem in v local !!" << std::endl;
//    if(f2q_[1][i].quad_idx>=p4est->local_num_quadrants && f2q_[1][i].tree_idx!=-1) std::cout << p4est->mpirank << " problem in v ghost !!" << std::endl;
//  }

  mpiret = MPI_Waitall(req_query2.size(), &req_query2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(req_reply2.size(), &req_reply2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
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

  if(dir==dir::x && q2f(quad_idx, dir::f_p00)==f_idx) xyz[0] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else                                                xyz[0] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  if(dir==dir::y && q2f(quad_idx, dir::f_0p0)==f_idx) xyz[1] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else                                                xyz[1] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
#ifdef P4_TO_P8
  if(dir==dir::z && q2f(quad_idx, dir::f_00p)==f_idx) xyz[2] +=    (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  else                                                xyz[2] += .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
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
