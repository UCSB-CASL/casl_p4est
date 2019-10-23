#include "my_p4est_faces.h"

#include <algorithm>
#include <src/matrix.h>

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_solve_lsqr.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/cube2.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_solve_lsqr.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <sc_notify.h>

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_faces_t;
extern PetscLogEvent log_my_p4est_faces_notify_t;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif


my_p4est_faces_t::my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c, bool initialize_neighborhoods_of_fine_faces):
  max_p4est_lvl(((splitting_criteria_t*) p4est->user_pointer)->max_lvl)
{
  this->p4est = p4est;
  this->ghost = ghost;
  this->myb = myb;
  this->ngbd_c = ngbd_c;
  dxyz_min(p4est, smallest_dxyz);
  finest_faces_neighborhoods_are_set = false;
  init_faces(initialize_neighborhoods_of_fine_faces);
}

void my_p4est_faces_t::init_faces(bool initialize_neighborhoods_of_fine_faces)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_faces_t, 0, 0, 0, 0); CHKERRXX(ierr);

  int mpiret;

  for(unsigned char d=0; d<P4EST_FACES; ++d)
    q2f_[d].resize(p4est->local_num_quadrants + ghost->ghosts.elem_count, NO_VELOCITY);

  for(unsigned char d=0; d<P4EST_DIM; ++d)
  {
    num_local[d] = 0;
    num_ghost[d] = 0;
    nonlocal_ranks[d].resize(0);
    ghost_local_num[d].resize(0);
    local_layer_face_index[d].resize(0);
    local_inner_face_index[d].resize(0);
    uniform_face_neighbors[d].clear();
  }

  vector<p4est_quadrant_t> ngbd;
  vector< vector<faces_comm_1_t> > buff_query1(p4est->mpisize); // buff_query[r][k] :: kth query to be sent to processor r (query == what is your local face number for the quadrant of queried local index in the queried face direction?)
  vector< vector<p4est_locidx_t> > map(p4est->mpisize); // map[r][k] :: local index of the locally owned quadrant associated with the kth query sent to proc r (i.e. buff_query[r][k])

  /* first process local velocities:
   * loop through all local quadrants, and process face by face in the following order f_m00, f_p00, f_0m0, f_0p0, f_00m and f_00p
   * */
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;
      for (int face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
      {

        /*
         * - If the face is a wall, it is owned by the current proc (and not shared by definition).
         * - Otherwise, find the neighboring cell(s) across the face,
         *   @ if more than one neighbor cell is found,
         *  ---> the face is left as NO_VELOCITY (i.e., not defined per se, since it is subrefined and owned from a smaller cell's perspective);
         *    @ if the unique neighbor cell is bigger
         *    @ or if the unique neighbor cell is of same size and local as well
         *    @ or if the unique neighbor cell is of same size but is a ghost owned by a higher-mpirank proc,
         *  ---> the face is owned by the current process;
         *    @ if the unique neighbor cell is of same size but is a ghost owned by a lower-mpirank proc,
         *  ---> prepare to send a query to the relevant proc to ask for their local index */
        if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
          q2f_[face_dir][quad_idx] = num_local[face_dir/2]++;
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, DIM(((face_dir/2==dir::x)?((face_dir%2==1)?+1:-1):0), ((face_dir/2==dir::y)?((face_dir%2==1)?+1:-1):0), ((face_dir/2==dir::z)?((face_dir%2==1)?+1:-1):0)));
          if(ngbd.size()==1)
          {
            P4EST_ASSERT(ngbd[0].level<=quad->level);
            if(ngbd[0].level<quad->level /* the neighbor is a (strictly) bigger cell */
               || (ngbd[0].p.piggy3.local_num <  p4est->local_num_quadrants && q2f_[((face_dir%2==0)?face_dir+1:face_dir-1)][ngbd[0].p.piggy3.local_num]==NO_VELOCITY) /* the shared face is local has not been indexed yet */
               || (ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank] ) ) /* ngbd is on process with larger index */
              q2f_[face_dir][quad_idx] = num_local[face_dir/2]++;
            else if(ngbd[0].p.piggy3.local_num >= p4est->local_num_quadrants && ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
            {
              p4est_locidx_t ghost_idx = ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants;
              int r = quad_find_ghost_owner(ghost, ghost_idx);
              const p4est_quadrant_t* g = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, ghost_idx);
              faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = ((face_dir%2==0)?face_dir+1:face_dir-1);
              buff_query1[r].push_back(c);
              map[r].push_back(quad_idx);
            }
            else
              q2f_[face_dir][quad_idx] = q2f_[((face_dir%2==0)?face_dir+1:face_dir-1)][ngbd[0].p.piggy3.local_num];
          }
        }
      }
    }
  }

  /* synchronize number of owned faces with the rest of the processes */
  vector<bool> face_has_already_been_visited[P4EST_DIM];
  for(unsigned char d=0; d<P4EST_DIM; ++d)
  {
    global_owned_indeps[d].resize(p4est->mpisize);
    global_owned_indeps[d][p4est->mpirank] = num_local[d];
    face_has_already_been_visited[d].resize(num_local[d], false);
    mpiret = MPI_Allgather(&num_local[d], 1, P4EST_MPI_LOCIDX, &global_owned_indeps[d][0], 1, P4EST_MPI_LOCIDX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  }

  /* initiate communications */
  vector<int> receivers_rank;
  for(int r=0; r<p4est->mpisize; ++r) // [Raphael:] probably not the best way to do it on large numbers of cores, but it'll do for now (consider using std::set otherwise)
  {
    if(buff_query1[r].size()>0)
    {
      P4EST_ASSERT(r != p4est->mpirank);
      receivers_rank.push_back(r);
    }
  }
  int num_receivers = receivers_rank.size();
  vector<int> senders_rank(p4est->mpisize);
  int num_senders;

  /* figure out the first communication pattern: notify the processes that this one is going to query and get to know which processes are going to query this one */
  ierr = PetscLogEventBegin(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);
  sc_notify(receivers_rank.data(), num_receivers, senders_rank.data(), &num_senders, p4est->mpicomm);
  ierr = PetscLogEventEnd(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);


  /* send the queries to fill in the local information */
  vector<MPI_Request> req_query1(num_receivers);
  for(int l=0; l<num_receivers; ++l)
  {
    mpiret = MPI_Isend(&buff_query1[receivers_rank[l]][0],
        buff_query1[receivers_rank[l]].size()*sizeof(faces_comm_1_t),
        MPI_BYTE, receivers_rank[l], 5, p4est->mpicomm, &req_query1[l]);
    SC_CHECK_MPI(mpiret);
  }

  // receive queries from other processes and send replies
  vector<faces_comm_1_t> buff_recv_comm1;
  vector< vector<p4est_locidx_t> > buff_reply1_send(num_senders);
  vector<MPI_Request> req_reply1(num_senders);
  for(int l=0; l<num_senders; ++l)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 5, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(vec_size%sizeof (faces_comm_1_t) == 0);
    vec_size /= sizeof(faces_comm_1_t);
    int r = status.MPI_SOURCE;

    buff_recv_comm1.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_comm1[0], vec_size*sizeof(faces_comm_1_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);

    /* prepare the reply */
    buff_reply1_send[l].resize(buff_recv_comm1.size());
    for(unsigned int n=0; n<buff_recv_comm1.size(); ++n)
    {
      p4est_locidx_t local_face_idx = q2f_[buff_recv_comm1[n].dir][buff_recv_comm1[n].local_num];
      buff_reply1_send[l][n] = local_face_idx;
      if(!face_has_already_been_visited[buff_recv_comm1[n].dir/2][local_face_idx]) // no check for NO_VELOCITY here because it SHOULD always be a velocity-face
      {
        local_layer_face_index[buff_recv_comm1[n].dir/2].push_back(local_face_idx);
        face_has_already_been_visited[buff_recv_comm1[n].dir/2][local_face_idx] = true;
      }
    }

    /* send reply */
    mpiret = MPI_Isend(&buff_reply1_send[l][0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, 6, p4est->mpicomm, &req_reply1[l]);
    SC_CHECK_MPI(mpiret);
  }
  buff_recv_comm1.clear();


  /* get the reply and fill in the missing local information */
  vector<p4est_locidx_t> buff_recv_locidx;
  for(int l=0; l<num_receivers; ++l)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 6, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(vec_size%sizeof (p4est_locidx_t) == 0);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    for(unsigned int n=0; n<buff_recv_locidx.size(); ++n)
    {
      unsigned int queried_face_dir = buff_query1[r][n].dir;
      q2f_[((queried_face_dir%2==0)? (queried_face_dir+1) : (queried_face_dir-1))][map[r][n]] = num_ghost[queried_face_dir/2]+num_local[queried_face_dir/2];
      ghost_local_num[queried_face_dir/2].push_back(buff_recv_locidx[n]);
      nonlocal_ranks[queried_face_dir/2].push_back(r);
      num_ghost[queried_face_dir/2]++;
    }
  }
  buff_recv_locidx.clear();

  mpiret = MPI_Waitall(num_receivers, &req_query1[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(num_senders  , &req_reply1[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

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

  /* figure out the second communication pattern */
  receivers_rank.clear();
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(buff_query2[r].size()>0)
    {
      P4EST_ASSERT(r != p4est->mpirank);
      receivers_rank.push_back(r);
    }
  }
  num_receivers = receivers_rank.size();
  ierr = PetscLogEventBegin(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);
  sc_notify(receivers_rank.data(), num_receivers, senders_rank.data(), &num_senders, p4est->mpicomm);
  ierr = PetscLogEventEnd(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);

  vector<MPI_Request> req_query2(num_receivers);
  for(int l=0; l<num_receivers; ++l)
  {
    mpiret = MPI_Isend(&buff_query2[receivers_rank[l]][0],
        buff_query2[receivers_rank[l]].size()*sizeof(p4est_locidx_t),
        MPI_BYTE, receivers_rank[l], 7, p4est->mpicomm, &req_query2[l]);
    SC_CHECK_MPI(mpiret);
  }

  vector<MPI_Request> req_reply2(num_senders);
  vector< vector<faces_comm_2_t> > buff_reply2(num_senders);
  for(int l=0; l<num_senders; ++l)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 7, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(vec_size%sizeof (p4est_locidx_t) == 0);
    vec_size /= sizeof(p4est_locidx_t);
    int r = status.MPI_SOURCE;

    buff_recv_locidx.resize(vec_size);
    mpiret = MPI_Recv(&buff_recv_locidx[0], vec_size*sizeof(p4est_locidx_t), MPI_BYTE, r, status.MPI_TAG, p4est->mpicomm, &status);
    SC_CHECK_MPI(mpiret);

    for(unsigned int q=0; q<buff_recv_locidx.size(); ++q)
    {
      faces_comm_2_t c;
      for(int face_dir=0; face_dir<P4EST_FACES; face_dir++)
      {
        p4est_locidx_t u_tmp = q2f_[face_dir][buff_recv_locidx[q]];
        /* local value */
        if(u_tmp<num_local[face_dir/2])
        {
          c.rank[face_dir] = p4est->mpirank;
          c.local_num[face_dir] = u_tmp;
          if(u_tmp!=NO_VELOCITY && !face_has_already_been_visited[face_dir/2][u_tmp])
          {
            local_layer_face_index[face_dir/2].push_back(u_tmp);
            face_has_already_been_visited[face_dir/2][u_tmp] = true;
          }
        }
        /* ghost value */
        else
        {
          c.rank[face_dir] = nonlocal_ranks[face_dir/2][u_tmp-num_local[face_dir/2]];
          c.local_num[face_dir] = ghost_local_num[face_dir/2][u_tmp-num_local[face_dir/2]];
        }
      }
      buff_reply2[l].push_back(c);
    }

    mpiret = MPI_Isend(&buff_reply2[l][0], buff_reply2[l].size()*sizeof(faces_comm_2_t), MPI_BYTE, r, 8, p4est->mpicomm, &req_reply2[l]);
    SC_CHECK_MPI(mpiret);
  }

  /* receive the ghost information and fill in the local info */
  vector<faces_comm_2_t> buff_recv_comm2;
  for(int l=0; l<num_receivers; ++l)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 8, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(vec_size%sizeof (faces_comm_2_t) == 0);
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

      for (int face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
        if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
        {
          q2f_[face_dir][quad_idx] = num_local[face_dir/2] + num_ghost[face_dir/2];
          ghost_local_num[face_dir/2].push_back(buff_recv_comm2[n].local_num[face_dir]);
          nonlocal_ranks[face_dir/2].push_back(buff_recv_comm2[n].rank[face_dir]);
          num_ghost[face_dir/2]++;
        }
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, DIM(((face_dir/2==dir::x)? ((face_dir%2==1)?+1:-1):0), ((face_dir/2==dir::y)? ((face_dir%2==1)?+1:-1):0), ((face_dir/2==dir::z)? ((face_dir%2==1)?+1:-1):0)));
          if((ngbd.size()==0 || ngbd.size()==1) && buff_recv_comm2[n].local_num[face_dir]!=NO_VELOCITY)
          {
            if(ngbd.size()==0 || q2f_[((face_dir%2==0)?(face_dir+1):(face_dir-1))][ngbd[0].p.piggy3.local_num]==NO_VELOCITY)
            {
              q2f_[face_dir][quad_idx] = num_local[face_dir/2] + num_ghost[face_dir/2];
              ghost_local_num[face_dir/2].push_back(buff_recv_comm2[n].local_num[face_dir]);
              nonlocal_ranks[face_dir/2].push_back(buff_recv_comm2[n].rank[face_dir]);
              num_ghost[face_dir/2]++;
            }
            else
              q2f_[face_dir][quad_idx] = q2f_[((face_dir%2==0)?(face_dir+1):(face_dir-1))][ngbd[0].p.piggy3.local_num];
          }
        }
      }
    }
  }

  /* now construct the velocity to quadrant link and complete the list of entirely local faces */
  int local_idx[P4EST_DIM];
  for(unsigned char d=0; d<P4EST_DIM; ++d)
  {
    f2q_[d].resize(num_local[d] + num_ghost[d]);
    local_inner_face_index[d].resize(num_local[d]-local_layer_face_index[d].size());
    local_idx[d] = 0;
  }
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;
      for(unsigned char face_dir=0; face_dir<P4EST_FACES; face_dir++)
      {
        if(q2f_[face_dir][quad_idx] != NO_VELOCITY)
        {
          p4est_locidx_t local_face_idx = q2f_[face_dir][quad_idx];
          f2q_[face_dir/2][local_face_idx].quad_idx = quad_idx;
          f2q_[face_dir/2][local_face_idx].tree_idx = tree_idx;
          if(local_face_idx < num_local[face_dir/2] && !face_has_already_been_visited[face_dir/2][local_face_idx]) // only local layer faces have been "visited" so far, the remaining local faces are all "inner" (i.e. non-shared)
          {
            local_inner_face_index[face_dir/2][local_idx[face_dir/2]++] = local_face_idx;
            face_has_already_been_visited[face_dir/2][local_face_idx] = true;
          }
          if(initialize_neighborhoods_of_fine_faces)
            find_fine_face_neighbors_and_store_it(tree_idx, quad_idx, tree, face_dir, local_face_idx);
        }
      }
    }
  }
#ifdef P4EST_DEBUG
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    P4EST_ASSERT(local_idx[dir] == num_local[dir]-((int) local_layer_face_index[dir].size()));
#endif

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t* ghost_quad = p4est_quadrant_array_index(&ghost->ghosts, q);
    p4est_locidx_t quad_idx = q+p4est->local_num_quadrants;
    for(unsigned char face_dir=0; face_dir<P4EST_FACES; face_dir++)
    {
      if(q2f_[face_dir][quad_idx] != NO_VELOCITY && f2q_[face_dir/2][q2f_[face_dir][quad_idx]].quad_idx == -1) // we do not overwrite f2q if already well-defined (i.e. not -1) to give precedence of local quadrants over ghosts
      {
        p4est_locidx_t local_face_idx = q2f_[face_dir][quad_idx];
        f2q_[face_dir/2][local_face_idx].quad_idx = quad_idx;
        f2q_[face_dir/2][local_face_idx].tree_idx = ghost_quad->p.piggy3.which_tree;
        if(initialize_neighborhoods_of_fine_faces)
          find_fine_face_neighbors_and_store_it(ghost_quad->p.piggy3.which_tree, quad_idx, NULL, face_dir, local_face_idx); // tree is irrelevant for ghost cells in the function
      }
    }
  }

  mpiret = MPI_Waitall(num_receivers, &req_query2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(num_senders  , &req_reply2[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  finest_faces_neighborhoods_are_set = initialize_neighborhoods_of_fine_faces;

#ifdef P4EST_DEBUG
  if(initialize_neighborhoods_of_fine_faces)
    P4EST_ASSERT(finest_face_neighborhoods_are_valid());
#endif

  ierr = PetscLogEventEnd(log_my_p4est_faces_t, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_faces_t::find_fine_face_neighbors_and_store_it(const p4est_topidx_t& tree_idx, const p4est_locidx_t& quad_idx, p4est_tree_t*tree,
                                                             const unsigned char& face_dir, const p4est_locidx_t& local_face_idx)
{
  P4EST_ASSERT((quad_idx >=0) && (quad_idx <p4est->local_num_quadrants + ((p4est_locidx_t)ghost->ghosts.elem_count)));
  P4EST_ASSERT(local_face_idx == q2f_[face_dir][quad_idx]);
  const p4est_quadrant_t* quad;
  if(quad_idx < p4est->local_num_quadrants)
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
  if(quad->level < max_p4est_lvl)
    return;

  if(!found_uniform_face_neighborhood(local_face_idx, face_dir/2)) // not in there yet
  {
    uniform_face_ngbd face_neighborhood;
    // ok, find neighboring faces, now
    bool add_to_map = true;
    for (unsigned char cart_dir = 0; add_to_map && (cart_dir < P4EST_FACES); ++cart_dir)
      add_to_map = add_to_map && found_finest_face_neighbor(quad, quad_idx, tree_idx, local_face_idx, face_dir/2, cart_dir, face_neighborhood.neighbor_face_idx[cart_dir]);
    if(add_to_map)
      uniform_face_neighbors[face_dir/2][local_face_idx] = face_neighborhood;
  }
}


//void my_p4est_faces_t::find_fine_face_neighbors_and_store_it(const p4est_topidx_t& tree_idx, const p4est_locidx_t& quad_idx, p4est_tree_t*tree,
//                                                             const unsigned char& face_dir, const p4est_locidx_t& local_face_idx)
//{
//  P4EST_ASSERT((quad_idx >=0) && (quad_idx <p4est->local_num_quadrants + ((p4est_locidx_t)ghost->ghosts.elem_count)));
//  P4EST_ASSERT(local_face_idx == q2f_[face_dir][quad_idx]);
//  const p4est_quadrant_t* quad;
//  if(quad_idx < p4est->local_num_quadrants)
//    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
//  else
//    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
//  if(quad->level < max_p4est_lvl)
//    return;

//  if(!found_uniform_face_neighborhood(local_face_idx, face_dir/2)) // not in there yet
//  {
//    uniform_face_ngbd face_neighborhood;
//    vector<p4est_quadrant_t> cell_neighbor(0);
//    p4est_locidx_t local_index_of_sharing_quad = -1;
//    p4est_topidx_t tree_idx_of_sharing_quad = -1;
//    // ok, find neighboring faces, now
//    bool do_not_add_to_map = false;
//    // in dual face_direction, first: use same quad
//    unsigned char dual_face_dir = ((face_dir%2==1)?(face_dir-1):(face_dir+1));
//    P4EST_ASSERT(q2f_[dual_face_dir][quad_idx] != NO_VELOCITY);
//    face_neighborhood.neighbor_face_idx[dual_face_dir] = q2f_[dual_face_dir][quad_idx];
//    // in face_dir, second: use sharing quad
//    if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
//      face_neighborhood.neighbor_face_idx[face_dir] = WALL_idx(face_dir);
//    else
//    {
//      cell_neighbor.clear();
//      ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, quad_idx, tree_idx, face_dir);
//      P4EST_ASSERT(cell_neighbor.size()<=1);
//      if((cell_neighbor.size()>0) && (cell_neighbor[0].level == max_p4est_lvl))
//      {
//        local_index_of_sharing_quad = cell_neighbor[0].p.piggy3.local_num;
//        tree_idx_of_sharing_quad    = cell_neighbor[0].p.piggy3.which_tree;
//        P4EST_ASSERT((q2f_[face_dir][local_index_of_sharing_quad] != NO_VELOCITY) && (q2f_[dual_face_dir][local_index_of_sharing_quad] == local_face_idx));
//        face_neighborhood.neighbor_face_idx[face_dir] = q2f_[face_dir][local_index_of_sharing_quad];
//      }
//      else
//        do_not_add_to_map = true;
//    }

//    for (unsigned char cart_dir = 0; !do_not_add_to_map && (cart_dir < P4EST_FACES); ++cart_dir) {
//      if((cart_dir == face_dir) || (cart_dir == dual_face_dir))
//        continue;
//      if(is_quad_Wall(p4est, tree_idx, quad, cart_dir))
//        face_neighborhood.neighbor_face_idx[cart_dir] = WALL_idx(cart_dir);
//      else
//      {
//        cell_neighbor.clear();
//        ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, quad_idx, tree_idx, cart_dir);
//        P4EST_ASSERT(cell_neighbor.size()<=1);
//        if((cell_neighbor.size()>0) && (cell_neighbor[0].level == max_p4est_lvl))
//        {
//          P4EST_ASSERT(q2f_[face_dir][cell_neighbor[0].p.piggy3.local_num] != NO_VELOCITY);
//          face_neighborhood.neighbor_face_idx[cart_dir] = q2f_[face_dir][cell_neighbor[0].p.piggy3.local_num];
//        }
//        else
//        {
//          P4EST_ASSERT((cell_neighbor.size() == 0) || (cell_neighbor[0].level < max_p4est_lvl));
//          cell_neighbor.clear();
//          ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, local_index_of_sharing_quad, tree_idx_of_sharing_quad, cart_dir);
//          if((cell_neighbor.size()>0) && (cell_neighbor[0].level == max_p4est_lvl))
//          {
//            P4EST_ASSERT(q2f_[dual_face_dir][cell_neighbor[0].p.piggy3.local_num] != NO_VELOCITY);
//            face_neighborhood.neighbor_face_idx[cart_dir] = q2f_[dual_face_dir][cell_neighbor[0].p.piggy3.local_num];
//          }
//          else
//            do_not_add_to_map = true;
//        }
//      }
//    }
//    if(!do_not_add_to_map)
//      uniform_face_neighbors[face_dir/2][local_face_idx] = face_neighborhood;
//  }
//}

void my_p4est_faces_t::set_finest_face_neighborhoods()
{
  if(finest_faces_neighborhoods_are_set)
    return;
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;
      for(unsigned char face_dir=0; face_dir<P4EST_FACES; face_dir++)
      {
        if(q2f_[face_dir][quad_idx] != NO_VELOCITY)
        {
          p4est_locidx_t local_face_idx = q2f_[face_dir][quad_idx];
          find_fine_face_neighbors_and_store_it(tree_idx, quad_idx, tree, face_dir, local_face_idx);
        }
      }
    }
  }
  finest_faces_neighborhoods_are_set = true;
  P4EST_ASSERT(finest_face_neighborhoods_are_valid());
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

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];

  p4est_qcoord_t xc = quad->x;
  if(dir!=dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00)==f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  return (tree_xmax-tree_xmin)*(double)xc/(double)P4EST_ROOT_LEN + tree_xmin;
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

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  p4est_qcoord_t yc = quad->y;
  if(dir!=dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0)==f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  return (tree_ymax-tree_ymin)*(double)yc/(double)P4EST_ROOT_LEN + tree_ymin;
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

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];

  p4est_qcoord_t zc = quad->z;
  if(dir!=dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p)==f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  return (tree_zmax-tree_zmin)*(double)zc/(double)P4EST_ROOT_LEN + tree_zmin;
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

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
  double tree_xyz_min[P4EST_DIM];
  double tree_xyz_max[P4EST_DIM];
  for(int i=0; i<P4EST_DIM; ++i)
  {
    tree_xyz_min[i] = p4est->connectivity->vertices[3*v_m + i];
    tree_xyz_max[i] = p4est->connectivity->vertices[3*v_p + i];
  }

  p4est_qcoord_t xc = quad->x;
  if(dir!=dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00)==f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[0] = (tree_xyz_max[0]-tree_xyz_min[0])*(double)xc/(double)P4EST_ROOT_LEN + tree_xyz_min[0];

  p4est_qcoord_t yc = quad->y;
  if(dir!=dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0)==f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[1] = (tree_xyz_max[1]-tree_xyz_min[1])*(double)yc/(double)P4EST_ROOT_LEN + tree_xyz_min[1];

#ifdef P4_TO_P8
  p4est_qcoord_t zc = quad->z;
  if(dir!=dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p)==f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[2] = (tree_xyz_max[2]-tree_xyz_min[2])*(double)zc/(double)P4EST_ROOT_LEN + tree_xyz_min[2];
#endif
}

void my_p4est_faces_t::rel_xyz_face_fr_node(const p4est_locidx_t& f_idx, const unsigned char& dir, double* xyz_rel, const double* xyz_node, const p4est_indep_t* node, const my_p4est_brick_t* brick,  __int64_t* logical_qcoord_diff) const
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

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + P4EST_CHILDREN-1];
  double tree_xyz_min[P4EST_DIM];
  double tree_xyz_max[P4EST_DIM];
  for(int i=0; i<P4EST_DIM; ++i)
  {
    tree_xyz_min[i]       = p4est->connectivity->vertices[3*v_m + i];
    tree_xyz_max[i]       = p4est->connectivity->vertices[3*v_p + i];
  }

  p4est_qcoord_t xc = quad->x;
  if(dir!=dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00)==f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[0] = (tree_xyz_max[0]-tree_xyz_min[0])*(double)xc/(double)P4EST_ROOT_LEN + tree_xyz_min[0] - xyz_node[0];
  double x_diff_tree = tree_xyz_min[0] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 0];
  p4est_topidx_t tree_x_diff = (p4est_topidx_t) round(x_diff_tree/(tree_xyz_max[0] - tree_xyz_min[0])); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[0] = tree_x_diff*P4EST_ROOT_LEN + xc - ((node->x != P4EST_ROOT_LEN-1)? node->x : P4EST_ROOT_LEN); // node might be clamped
  if(is_periodic(p4est, dir::x))
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[0] + ((double) i)*(brick->xyz_max[0] - brick->xyz_min[0])) < fabs(xyz_rel[0]))
        xyz_rel[0] += ((double) i)*(brick->xyz_max[0] - brick->xyz_min[0]);
      if(abs(logical_qcoord_diff[0] + i*brick->nxyztrees[0]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[0]))
        logical_qcoord_diff[0] += i*brick->nxyztrees[0]*P4EST_ROOT_LEN;
    }

  p4est_qcoord_t yc = quad->y;
  if(dir!=dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0)==f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[1] = (tree_xyz_max[1]-tree_xyz_min[1])*(double)yc/(double)P4EST_ROOT_LEN + tree_xyz_min[1] - xyz_node[1];
  double y_diff_tree = tree_xyz_min[1] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 1];
  p4est_topidx_t tree_y_diff = (p4est_topidx_t) round(y_diff_tree/(tree_xyz_max[1] - tree_xyz_min[1])); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[1] = tree_y_diff*P4EST_ROOT_LEN + yc - ((node->y != P4EST_ROOT_LEN-1)? node->y : P4EST_ROOT_LEN); // node might be clamped
  if(is_periodic(p4est, dir::y))
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[1] + ((double) i)*(brick->xyz_max[1] - brick->xyz_min[1])) < fabs(xyz_rel[1]))
        xyz_rel[1] += ((double) i)*(brick->xyz_max[1] - brick->xyz_min[1]);
      if(abs(logical_qcoord_diff[1] + i*brick->nxyztrees[1]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[1]))
        logical_qcoord_diff[1] += i*brick->nxyztrees[1]*P4EST_ROOT_LEN;
    }

#ifdef P4_TO_P8
  p4est_qcoord_t zc = quad->z;
  if(dir!=dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p)==f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[2] = (tree_xyz_max[2]-tree_xyz_min[2])*(double)zc/(double)P4EST_ROOT_LEN + tree_xyz_min[2] - xyz_node[2];
  double z_diff_tree = tree_xyz_min[2] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 2];
  p4est_topidx_t tree_z_diff = (p4est_topidx_t) round(z_diff_tree/(tree_xyz_max[2] - tree_xyz_min[2])); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[2] = tree_z_diff*P4EST_ROOT_LEN + zc - ((node->z != P4EST_ROOT_LEN-1)? node->z : P4EST_ROOT_LEN); // node might be clamped
  if(is_periodic(p4est, dir::z))
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[2] + ((double) i)*(brick->xyz_max[2] - brick->xyz_min[2])) < fabs(xyz_rel[2]))
        xyz_rel[2] += ((double) i)*(brick->xyz_max[2] - brick->xyz_min[2]);
      if(abs(logical_qcoord_diff[2] + i*brick->nxyztrees[2]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[2]))
        logical_qcoord_diff[2] += i*brick->nxyztrees[2]*P4EST_ROOT_LEN;
    }
#endif
}

#ifdef P4_TO_P8
double my_p4est_faces_t::face_area_in_negative_domain(p4est_locidx_t f_idx, int dir, const double *phi_p, const p4est_nodes_t* nodes) const
#else
double my_p4est_faces_t::face_area_in_negative_domain(p4est_locidx_t f_idx, int dir, const double *phi_p, const p4est_nodes_t* nodes, const double *phi_dd[]) const
#endif
{
#ifdef CASL_THROWS
  if((phi_p != NULL) && (nodes == NULL))
    throw std::invalid_argument("my_p4est_faces_t::face_area: if the node-sampled levelset function is provided, the nodes MUST be provided as well.");
#endif
  p4est_locidx_t  *t2v = p4est->connectivity->tree_to_vertex;
  double          *v2c = p4est->connectivity->vertices;

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

  int tmp = ((q2f(quad_idx, 2*dir)==f_idx)? 0 : 1);

  double area = 1.0;
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if(dim == dir)
      continue;
    area *= (v2c[3*t2v[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1] + dir]-v2c[3*t2v[P4EST_CHILDREN*tree_idx + 0] + dir])/((double) (1<<quad->level));
  }
  if(phi_p != NULL)
  {
    p4est_locidx_t node_indices[2*(P4EST_DIM-1)];
    short zzz, yyy, xxx;
    for (short first = 0; first < P4EST_DIM-1; ++first) {
      for (short second = 0; second < 2; ++second) {
#ifdef P4_TO_P8
        zzz = ((dir==dir::z)? tmp : first);
        yyy = ((dir==dir::y)? tmp : ((dir==dir::z)? first : second));
        xxx = ((dir==dir::x)? tmp : second);
#else
        zzz = first; // always 0...
        yyy = ((dir==dir::y)? tmp : second);
        xxx = ((dir==dir::x)? tmp : second);
#endif
        node_indices[2*first+second] = nodes->local_nodes[P4EST_CHILDREN*quad_idx+4*zzz+2*yyy+xxx];
      }
    }
    bool they_are_all_positive    = true;
    bool at_least_one_is_positive = false;
    for (short kk = 0; kk < 2*(P4EST_DIM-1); ++kk) {
      bool node_is_in_positive_domain = (phi_p[node_indices[kk]] > 0.0);
      they_are_all_positive     = they_are_all_positive && node_is_in_positive_domain;
      at_least_one_is_positive  = at_least_one_is_positive || (node_is_in_positive_domain);
    }
    if (they_are_all_positive)
      return 0.0;
    if (at_least_one_is_positive)
    {
#ifndef P4_TO_P8
      double h = area;
      if(phi_dd == NULL)
        area *= fraction_Interval_Covered_By_Irregular_Domain(phi_p[node_indices[0]], phi_p[node_indices[1]], h, h);
      else
        area *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[node_indices[0]], phi_p[node_indices[1]], phi_dd[(dir?0:1)][node_indices[0]], phi_dd[(dir?0:1)][node_indices[1]], h);
#else
      Cube2 my_face(0.0, 1.0, 0.0, 1.0);
      QuadValue ls_value(phi_p[node_indices[0]], phi_p[node_indices[1]], phi_p[node_indices[2]], phi_p[node_indices[3]]);
      area *= my_face.area_In_Negative_Domain(ls_value);
#endif
    }
  }
  return area;
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




void check_if_faces_are_well_defined(p4est_t *p4est, my_p4est_node_neighbors_t *ngbd_n,
                                     my_p4est_faces_t *faces, int dir,
                                     Vec phi, BoundaryConditionType bc_type, Vec face_is_well_defined)
{
  PetscErrorCode ierr;

  if(bc_type==NOINTERFACE)
  {
    Vec face_is_well_defined_loc;
    ierr = VecGhostGetLocalForm(face_is_well_defined, &face_is_well_defined_loc); CHKERRXX(ierr);
    ierr = VecSet(face_is_well_defined_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined, &face_is_well_defined_loc); CHKERRXX(ierr);
    return;
  }

  PetscScalar *face_is_well_defined_p;
  ierr = VecGetArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);
  interp.set_input(phi, linear);

  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = .5 * (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = .5 * (ymax-ymin) / pow(2.,(double) data->max_lvl);
#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = .5 * (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

  if(bc_type==DIRICHLET)
  {
    for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
    {
      double x = faces->x_fr_f(f_idx,dir);
      double y = faces->y_fr_f(f_idx,dir);
#ifdef P4_TO_P8
      double z = faces->z_fr_f(f_idx,dir);
      face_is_well_defined_p[f_idx] = interp(x,y,z)<=0;
#else
      face_is_well_defined_p[f_idx] = interp(x,y)<=0;
#endif
    }
  }
  else /* NEUMANN */
  {
    for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
    {
      double x = faces->x_fr_f(f_idx,dir);
      double y = faces->y_fr_f(f_idx,dir);
#ifdef P4_TO_P8
      double z = faces->z_fr_f(f_idx,dir);
      face_is_well_defined_p[f_idx] = ( interp(x-dx, y-dy, z-dz)<=0 || interp(x+dx, y-dy, z-dz)<=0 ||
                                        interp(x-dx, y-dy, z+dz)<=0 || interp(x+dx, y-dy, z+dz)<=0 ||
                                        interp(x-dx, y+dy, z-dz)<=0 || interp(x+dx, y+dy, z-dz)<=0 ||
                                        interp(x-dx, y+dy, z+dz)<=0 || interp(x+dx, y+dy, z+dz)<=0 );
#else
      face_is_well_defined_p[f_idx] = ( interp(x-dx, y-dy)<=0 || interp(x+dx, y-dy)<=0 ||
                                        interp(x-dx, y+dy)<=0 || interp(x+dx, y+dy)<=0 );
#endif
    }
  }

  ierr = VecRestoreArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}






#ifdef P4_TO_P8
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               p4est_locidx_t node_idx, Vec f, int dir,
                               Vec face_is_well_defined, int order, BoundaryConditions3D *bc)
#else
double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               p4est_locidx_t node_idx, Vec f, int dir,
                               Vec face_is_well_defined, int order, BoundaryConditions2D *bc)
#endif
{
  PetscErrorCode ierr;

  double xyz[P4EST_DIM];
  xyz[0] = node_x_fr_n(node_idx, p4est, nodes);
  xyz[1] = node_y_fr_n(node_idx, p4est, nodes);
#ifdef P4_TO_P8
  xyz[2] = node_z_fr_n(node_idx, p4est, nodes);
#endif

  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);

  if(bc!=NULL && is_node_Wall(p4est, node) && bc[dir].wallType(xyz)==DIRICHLET)
    return bc[dir].wallValue(xyz);

  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double xmin = v2c[3*t2v[0 + 0] + 0];
  double xmax = v2c[3*t2v[0 + P4EST_CHILDREN-1] + 0];
  double ymin = v2c[3*t2v[0 + 0] + 1];
  double ymax = v2c[3*t2v[0 + P4EST_CHILDREN-1] + 1];
#ifdef P4_TO_P8
  double zmin = v2c[3*t2v[0 + 0] + 2];
  double zmax = v2c[3*t2v[0 + P4EST_CHILDREN-1] + 2];
  double qh = MIN(xmax-xmin, ymax-ymin, zmax-zmin);
#else
  double qh = MIN(xmax-xmin, ymax-ymin);
#endif
  double domain_size[P4EST_DIM];
  if(is_periodic(p4est, dir::x) || is_periodic(p4est, dir::y)
   #ifdef P4_TO_P8
     || is_periodic(p4est, dir::z)
   #endif
     )
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      domain_size[dim] =
          p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1] + dim] -
          p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[0 + 0] + dim];


  /* gather the neighborhood */
#ifdef CASL_THROWS
  bool is_local = false;
#endif
  vector<p4est_quadrant_t> ngbd_tmp;
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  double scaling = DBL_MAX;
  for(int i=-1; i<2; i+=2)
    for(int j=-1; j<2; j+=2)
#ifdef P4_TO_P8
      for(int k=-1; k<2; k+=2)
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx, i, j, k, quad_idx, tree_idx);
#else
    {
      ngbd_n->find_neighbor_cell_of_node(node_idx, i, j, quad_idx, tree_idx);
#endif
      if(quad_idx!=NOT_A_VALID_QUADRANT)
      {
        p4est_quadrant_t quad;
        if(quad_idx<p4est->local_num_quadrants)
        {
          p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
          quad = *(p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
        }
        else
        {
          quad = *(p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
        }

        quad.p.piggy3.local_num = quad_idx;

#ifdef CASL_THROWS
        is_local = is_local || (quad_idx<p4est->local_num_quadrants);
#endif

        ngbd_tmp.push_back(quad);
        scaling = MIN(scaling, .5*qh*(double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN);

#ifdef P4_TO_P8
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 0, j, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 0, 0, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, 0, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 0, j, k);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j, k);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, 0, j);
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j);
#endif
      }
    }

#ifdef CASL_THROWS
  if(!is_local)
  {
    ierr = PetscPrintf(p4est->mpicomm, "Warning !! interpolation_f_at_node_n: the node is not local."); CHKERRXX(ierr);
//    throw std::invalid_argument("[CASL_ERROR]: interpolate_f_at_node_n: cannot be called on a ghost node.");
  }
#endif

  vector<p4est_locidx_t> ngbd;
  p4est_locidx_t f_tmp;
  for(unsigned int m=0; m<ngbd_tmp.size(); ++m)
  {
    f_tmp = faces->q2f(ngbd_tmp[m].p.piggy3.local_num, 2*dir  );
    if(f_tmp!=NO_VELOCITY && std::find(ngbd.begin(), ngbd.end(),f_tmp)==ngbd.end())
      ngbd.push_back(f_tmp);

    f_tmp = faces->q2f(ngbd_tmp[m].p.piggy3.local_num, 2*dir+1);
    if(f_tmp!=NO_VELOCITY && std::find(ngbd.begin(), ngbd.end(),f_tmp)==ngbd.end())
      ngbd.push_back(f_tmp);
  }

  double *f_p;
  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  vector<p4est_locidx_t> interp_points;
  matrix_t A;
  bool neumann_wall_x = (bc!=NULL && (is_node_xmWall(p4est, node) || is_node_xpWall(p4est, node)) && bc[dir].wallType(xyz)==NEUMANN);
  bool neumann_wall_y = (bc!=NULL && (is_node_ymWall(p4est, node) || is_node_ypWall(p4est, node)) && bc[dir].wallType(xyz)==NEUMANN);
#ifdef P4_TO_P8
  bool neumann_wall_z = (bc!=NULL && (is_node_zmWall(p4est, node) || is_node_zpWall(p4est, node)) && bc[dir].wallType(xyz)==NEUMANN);
  A.resize(1, (order>=2 ? 10 : 4) - (neumann_wall_x?1:0)- (neumann_wall_y?1:0) - (neumann_wall_z?1:0));
#else
  A.resize(1, (order>=2 ? 6 : 3) - (neumann_wall_x?1:0)- (neumann_wall_y?1:0));
#endif
  vector<double> p;
  vector<double> nb[P4EST_DIM];

  double min_w = 1e-6;
  double inv_max_w = 1e-6;

  const PetscScalar *face_is_well_defined_p;
  if(face_is_well_defined!=NULL)
    ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(unsigned int m=0; m<ngbd.size(); m++)
  {
    /* minus direction */
    p4est_locidx_t fm_idx = ngbd[m];
    if((face_is_well_defined==NULL || face_is_well_defined_p[fm_idx]) && std::find(interp_points.begin(), interp_points.end(),fm_idx)==interp_points.end() )
    {
      double xyz_t[P4EST_DIM];
      faces->xyz_fr_f(fm_idx, dir, xyz_t);
      for(int i=0; i<P4EST_DIM; ++i)
      {
        double rel_dist = (xyz[i] - xyz_t[i]);
        if(is_periodic(p4est, i))
          for (short cc = -1; cc < 2; cc+=2)
            if(fabs((xyz[i] - xyz_t[i] + ((double) cc)*domain_size[i])) < fabs(rel_dist))
              rel_dist = (xyz[i] - xyz_t[i] + ((double) cc)*domain_size[i]);
        xyz_t[i] = rel_dist / scaling;
      }

#ifdef P4_TO_P8
      double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
      double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif

#ifdef P4_TO_P8
      A.set_value(interp_points.size(), 0, 1                                                                                  * w);
      if(!neumann_wall_x)
        A.set_value(interp_points.size(), 1, xyz_t[0]                                                                         * w);
      if(!neumann_wall_y)
        A.set_value(interp_points.size(), 2-(neumann_wall_x?1:0), xyz_t[1]                                                    * w);
      if(!neumann_wall_z)
        A.set_value(interp_points.size(), 3-(neumann_wall_x?1:0)-(neumann_wall_y?1:0), xyz_t[2]                               * w);
      if(order>=2)
      {
        A.set_value(interp_points.size(), 4-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 5-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 6-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[0]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 7-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[1]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 8-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 9-(neumann_wall_x?1:0)-(neumann_wall_y?1:0)-(neumann_wall_z?1:0), xyz_t[2]*xyz_t[2] * w);
      }
#else
      A.set_value(interp_points.size(), 0, 1                                                              * w);
      if(!neumann_wall_x)
        A.set_value(interp_points.size(), 1, xyz_t[0]                                                     * w);
      if(!neumann_wall_y)
        A.set_value(interp_points.size(), 2-(neumann_wall_x?1:0), xyz_t[1]                                * w);
      if(order>=2)
      {
        A.set_value(interp_points.size(), 3-(neumann_wall_x?1:0)-(neumann_wall_y?1:0), xyz_t[0]*xyz_t[0]  * w);
        A.set_value(interp_points.size(), 4-(neumann_wall_x?1:0)-(neumann_wall_y?1:0), xyz_t[0]*xyz_t[1]  * w);
        A.set_value(interp_points.size(), 5-(neumann_wall_x?1:0)-(neumann_wall_y?1:0), xyz_t[1]*xyz_t[1]  * w);
      }
#endif

      p.push_back((f_p[fm_idx] + (neumann_wall_x? bc->wallValue(xyz)*xyz_t[0]*scaling: 0.0) + (neumann_wall_y? bc->wallValue(xyz)*xyz_t[1]*scaling: 0.0)
             #ifdef P4_TO_P8
                   + (neumann_wall_z? bc->wallValue(xyz)*xyz_t[2]*scaling: 0.0)
             #endif
                   ) * w);

      for(int d=0; d<P4EST_DIM; ++d)
        if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
          nb[d].push_back(xyz_t[d]);

      interp_points.push_back(fm_idx);
    }
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
  if(face_is_well_defined!=NULL)
    ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  if(interp_points.size()==0)
    return 0;

  A.scale_by_maxabs(p);

#ifdef P4_TO_P8
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size(), nb[2].size(), order);
#else
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size(), order);
#endif
}
