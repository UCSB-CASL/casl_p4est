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
extern PetscLogEvent log_my_p4est_faces_compute_voronoi_cell_t;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif


my_p4est_faces_t::my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c, bool initialize_neighborhoods_of_fine_faces):
  max_p4est_lvl(((splitting_criteria_t*) p4est->user_pointer)->max_lvl)
{
  this->p4est   = p4est;
  this->ghost   = ghost;
  this->myb     = myb;
  this->ngbd_c  = ngbd_c;
  // domain-size info
  const p4est_topidx_t  *t2v = p4est->connectivity->tree_to_vertex;
  const double          *v2c = p4est->connectivity->vertices;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    periodic[dir]         = is_periodic(p4est, dir);
    tree_dimensions[dir]  = v2c[3*t2v[P4EST_CHILDREN - 1] + dir] - v2c[3*t2v[0] + dir];
    smallest_dxyz[dir]    = tree_dimensions[dir]/(1 << (((splitting_criteria_t *) p4est->user_pointer)->max_lvl));
    xyz_max[dir]          = v2c[3*t2v[P4EST_CHILDREN*(p4est->trees->elem_count - 1) + P4EST_CHILDREN - 1] + dir];
    xyz_min[dir]          = v2c[3*t2v[0] + dir];
  }
  finest_faces_neighborhoods_are_set = false;
  init_faces(initialize_neighborhoods_of_fine_faces);
}

void my_p4est_faces_t::init_faces(bool initialize_neighborhoods_of_fine_faces)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_faces_t, 0, 0, 0, 0); CHKERRXX(ierr);

  int mpiret;

  for(unsigned char d = 0; d < P4EST_FACES; ++d)
    q2f_[d].resize(p4est->local_num_quadrants + ghost->ghosts.elem_count, NO_VELOCITY);

  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    num_local[d] = 0;
    num_ghost[d] = 0;
    nonlocal_ranks[d].resize(0);
    ghost_local_num[d].resize(0);
    local_layer_face_index[d].resize(0);
    local_inner_face_index[d].resize(0);
    uniform_face_neighbors[d].clear();
  }

  set_of_neighboring_quadrants ngbd;
  int min_ghost_owner_rank = 0;
  int max_ghost_owner_rank = 0;
  if(ghost->ghosts.elem_count > 0)
  {
    min_ghost_owner_rank = quad_find_ghost_owner(ghost, 0);
    max_ghost_owner_rank = quad_find_ghost_owner(ghost, ghost->ghosts.elem_count - 1);
  }

  std::set<int> set_of_ranks;
  vector< vector<faces_comm_1_t> > buff_query1(p4est->mpisize); // buff_query[r][k] :: kth query to be sent to processor r (query == what is your local face number for the quadrant of queried local index in the queried face direction?)
  vector< vector<p4est_locidx_t> > map(p4est->mpisize); // map[r][k] :: local index of the locally owned quadrant associated with the kth query sent to proc r (i.e. buff_query[r][k])

  /* first process local velocities:
   * loop through all local quadrants, and process face by face in the following order f_m00, f_p00, f_0m0, f_0p0, f_00m and f_00p
   * */
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
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
          ngbd.clear();
          char search[P4EST_DIM] = {DIM(0, 0, 0)}; search[face_dir/2] = (face_dir%2 == 1 ? 1 : -1);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, DIM(search[0], search[1], search[2]));
          if(ngbd.size() == 1)
          {
            P4EST_ASSERT(ngbd.begin()->level <= quad->level);
            if(ngbd.begin()->level < quad->level /* the neighbor is a (strictly) bigger cell */
               || (ngbd.begin()->p.piggy3.local_num <  p4est->local_num_quadrants && q2f_[(face_dir%2 == 0 ? face_dir + 1 : face_dir - 1)][ngbd.begin()->p.piggy3.local_num] == NO_VELOCITY) /* the shared face is local has not been indexed yet */
               || (ngbd.begin()->p.piggy3.local_num >= p4est->local_num_quadrants && ngbd.begin()->p.piggy3.local_num - p4est->local_num_quadrants >= ghost->proc_offsets[p4est->mpirank])) /* ngbd is on process with larger index */
              q2f_[face_dir][quad_idx] = num_local[face_dir/2]++;
            else if(ngbd.begin()->p.piggy3.local_num >= p4est->local_num_quadrants && ngbd.begin()->p.piggy3.local_num - p4est->local_num_quadrants < ghost->proc_offsets[p4est->mpirank])
            {
              p4est_locidx_t ghost_idx = ngbd.begin()->p.piggy3.local_num - p4est->local_num_quadrants;
              int r = quad_find_ghost_owner(ghost, ghost_idx, min_ghost_owner_rank, p4est->mpirank); P4EST_ASSERT(r < p4est->mpirank); // we know upper limit is mpirank (more restrictive than max_ghost_owner_rank)
              const p4est_quadrant_t* g = p4est_quadrant_array_index(&ghost->ghosts, ghost_idx);
              faces_comm_1_t c; c.local_num = g->p.piggy3.local_num; c.dir = (face_dir%2 == 0 ? face_dir + 1 : face_dir - 1);
              buff_query1[r].push_back(c);
              map[r].push_back(quad_idx);
              set_of_ranks.insert(r);
            }
            else
              q2f_[face_dir][quad_idx] = q2f_[(face_dir%2 == 0 ? face_dir + 1 : face_dir - 1)][ngbd.begin()->p.piggy3.local_num];
          }
        }
      }
    }
  }

  /* synchronize number of owned faces with the rest of the processes */
  vector<bool> face_has_already_been_visited[P4EST_DIM];
  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    global_owned_indeps[d].resize(p4est->mpisize);
    global_owned_indeps[d][p4est->mpirank] = num_local[d];
    face_has_already_been_visited[d].resize(num_local[d], false);
    mpiret = MPI_Allgather(&num_local[d], 1, P4EST_MPI_LOCIDX, &global_owned_indeps[d][0], 1, P4EST_MPI_LOCIDX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    proc_offset[d].resize(p4est->mpisize+1);
    proc_offset[d][0] = 0;
  }
  for (int r = 1; r <= p4est->mpisize; ++r)
    for (unsigned char d = 0; d < P4EST_DIM; ++d)
      proc_offset[d][r] = proc_offset[d][r-1] + global_owned_indeps[d][r-1];

  /* initiate communications */
  vector<int> receivers_rank;
  for (std::set<int>::const_iterator it = set_of_ranks.begin(); it != set_of_ranks.end(); ++it) {
    P4EST_ASSERT(*it < p4est->mpirank);
    receivers_rank.push_back(*it);
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
  for(int l = 0; l < num_receivers; ++l)
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
  for(int l = 0; l < num_senders; ++l)
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
    for(size_t n = 0; n < buff_recv_comm1.size(); ++n)
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
  for(int l = 0; l < num_receivers; ++l)
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

    for(size_t n = 0; n < buff_recv_locidx.size(); ++n)
    {
      unsigned char queried_face_dir = buff_query1[r][n].dir;
      q2f_[(queried_face_dir%2 == 0 ? queried_face_dir + 1 : queried_face_dir - 1)][map[r][n]] = num_ghost[queried_face_dir/2]+num_local[queried_face_dir/2];
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
  for (std::set<int>::const_iterator it = set_of_ranks.begin(); it != set_of_ranks.end(); ++it) {
    P4EST_ASSERT(*it < p4est->mpirank);
    map[*it].clear();
  }
  set_of_ranks.clear();
  int ghost_rank = min_ghost_owner_rank;
  for(p4est_locidx_t ghost_idx = 0; ghost_idx < (p4est_locidx_t) ghost->ghosts.elem_count; ++ghost_idx)
  {
    while (ghost_rank < max_ghost_owner_rank && ghost->proc_offsets[ghost_rank + 1] <= ghost_idx) { ghost_rank++; } // we loop in increasing ghost owner rank --> this is more efficient than binary search
    p4est_quadrant_t *g = p4est_quadrant_array_index(&ghost->ghosts, ghost_idx);
    buff_query2[ghost_rank].push_back(g->p.piggy3.local_num);
    map[ghost_rank].push_back(ghost_idx);
    set_of_ranks.insert(ghost_rank);
  }

  /* figure out the second communication pattern */
  receivers_rank.clear();
  for (std::set<int>::const_iterator it = set_of_ranks.begin(); it != set_of_ranks.end(); ++it) {
    P4EST_ASSERT(min_ghost_owner_rank <= *it && *it <= max_ghost_owner_rank);
    receivers_rank.push_back(*it);
  }
  num_receivers = receivers_rank.size();
  ierr = PetscLogEventBegin(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);
  sc_notify(receivers_rank.data(), num_receivers, senders_rank.data(), &num_senders, p4est->mpicomm);
  ierr = PetscLogEventEnd(log_my_p4est_faces_notify_t, 0, 0, 0, 0); CHKERRXX(ierr);

  vector<MPI_Request> req_query2(num_receivers);
  for(int l = 0; l < num_receivers; ++l)
  {
    mpiret = MPI_Isend(&buff_query2[receivers_rank[l]][0],
        buff_query2[receivers_rank[l]].size()*sizeof(p4est_locidx_t),
        MPI_BYTE, receivers_rank[l], 7, p4est->mpicomm, &req_query2[l]);
    SC_CHECK_MPI(mpiret);
  }

  vector<MPI_Request> req_reply2(num_senders);
  vector< vector<faces_comm_2_t> > buff_reply2(num_senders);
  for(int l = 0; l < num_senders; ++l)
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

    for(size_t q = 0; q < buff_recv_locidx.size(); ++q)
    {
      faces_comm_2_t c;
      for(unsigned char face_dir = 0; face_dir < P4EST_FACES; face_dir++)
      {
        p4est_locidx_t u_tmp = q2f_[face_dir][buff_recv_locidx[q]];
        /* local value */
        if(u_tmp < num_local[face_dir/2])
        {
          c.rank[face_dir] = p4est->mpirank;
          c.local_num[face_dir] = u_tmp;
          if(u_tmp != NO_VELOCITY && !face_has_already_been_visited[face_dir/2][u_tmp])
          {
            local_layer_face_index[face_dir/2].push_back(u_tmp);
            face_has_already_been_visited[face_dir/2][u_tmp] = true;
          }
        }
        /* ghost value */
        else
        {
          c.rank[face_dir] = nonlocal_ranks[face_dir/2][u_tmp - num_local[face_dir/2]];
          c.local_num[face_dir] = ghost_local_num[face_dir/2][u_tmp - num_local[face_dir/2]];
        }
      }
      buff_reply2[l].push_back(c);
    }

    mpiret = MPI_Isend(&buff_reply2[l][0], buff_reply2[l].size()*sizeof(faces_comm_2_t), MPI_BYTE, r, 8, p4est->mpicomm, &req_reply2[l]);
    SC_CHECK_MPI(mpiret);
  }

  /* receive the ghost information and fill in the local info */
  vector<faces_comm_2_t> buff_recv_comm2;
  for(int l = 0; l < num_receivers; ++l)
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
    for(int n = 0; n < vec_size; ++n)
    {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&ghost->ghosts, map[r][n]);
      p4est_locidx_t quad_idx = map[r][n]+p4est->local_num_quadrants;
      p4est_topidx_t tree_idx = quad->p.piggy3.which_tree;

      for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
        if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
        {
          q2f_[face_dir][quad_idx] = num_local[face_dir/2] + num_ghost[face_dir/2];
          ghost_local_num[face_dir/2].push_back(buff_recv_comm2[n].local_num[face_dir]);
          nonlocal_ranks[face_dir/2].push_back(buff_recv_comm2[n].rank[face_dir]);
          num_ghost[face_dir/2]++;
        }
        else
        {
          ngbd.clear();
          char search[P4EST_DIM] = {DIM(0, 0, 0)}; search[face_dir/2] = (face_dir%2 == 1 ? +1 : -1);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, DIM(search[0], search[1], search[2]));
          if((ngbd.size() == 0 || ngbd.size() == 1) && buff_recv_comm2[n].local_num[face_dir] != NO_VELOCITY)
          {
            if(ngbd.size() == 0 || q2f_[(face_dir%2 == 0 ? face_dir + 1 : face_dir - 1)][ngbd.begin()->p.piggy3.local_num] == NO_VELOCITY)
            {
              q2f_[face_dir][quad_idx] = num_local[face_dir/2] + num_ghost[face_dir/2];
              ghost_local_num[face_dir/2].push_back(buff_recv_comm2[n].local_num[face_dir]);
              nonlocal_ranks[face_dir/2].push_back(buff_recv_comm2[n].rank[face_dir]);
              num_ghost[face_dir/2]++;
            }
            else
              q2f_[face_dir][quad_idx] = q2f_[(face_dir%2 == 0 ? face_dir + 1 : face_dir - 1)][ngbd.begin()->p.piggy3.local_num];
          }
        }
      }
    }
  }

  /* now construct the velocity to quadrant link and complete the list of entirely local faces */
  int local_idx[P4EST_DIM];
  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    f2q_[d].resize(num_local[d] + num_ghost[d]);
    local_inner_face_index[d].resize(num_local[d]-local_layer_face_index[d].size());
    local_idx[d] = 0;
  }
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      for(unsigned char face_dir = 0; face_dir < P4EST_FACES; face_dir++)
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

  for(size_t q = 0; q < ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t* ghost_quad = p4est_quadrant_array_index(&ghost->ghosts, q);
    p4est_locidx_t quad_idx = q + p4est->local_num_quadrants;
    for(unsigned char face_dir = 0; face_dir < P4EST_FACES; face_dir++)
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
  if(finest_faces_neighborhoods_are_set)
    return;
  P4EST_ASSERT(quad_idx >= 0 && quad_idx < p4est->local_num_quadrants + (p4est_locidx_t) ghost->ghosts.elem_count);
  P4EST_ASSERT(local_face_idx == q2f_[face_dir][quad_idx]);
  const p4est_quadrant_t* quad;
  if(quad_idx < p4est->local_num_quadrants)
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
  if(quad->level < max_p4est_lvl)
    return;

  if(uniform_face_neighbors[face_dir/2].find(local_face_idx) == uniform_face_neighbors[face_dir/2].end()) // not in there yet
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

void my_p4est_faces_t::set_finest_face_neighborhoods()
{
  if(finest_faces_neighborhoods_are_set)
    return;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      for(unsigned char face_dir = 0; face_dir < P4EST_FACES; face_dir++)
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

double my_p4est_faces_t::x_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
  {
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_m  = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_xmin    = p4est->connectivity->vertices[3*v_m + 0];

  p4est_qcoord_t xc   = quad->x;
  if(dir != dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00) == f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  return tree_dimensions[0]*(double)xc/(double)P4EST_ROOT_LEN + tree_xmin;
}


double my_p4est_faces_t::y_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
  {
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];

  p4est_qcoord_t yc = quad->y;
  if(dir != dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0) == f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  return tree_dimensions[1]*(double)yc/(double)P4EST_ROOT_LEN + tree_ymin;
}


#ifdef P4_TO_P8
double my_p4est_faces_t::z_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
  {
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];

  p4est_qcoord_t zc = quad->z;
  if(dir != dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p) == f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  return tree_dimensions[2]*(double)zc/(double)P4EST_ROOT_LEN + tree_zmin;
}
#endif



void my_p4est_faces_t::xyz_fr_f(p4est_locidx_t f_idx, const unsigned char &dir, double* xyz) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
  {
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_xyz_min[P4EST_DIM];
  for(unsigned char i = 0; i < P4EST_DIM; ++i)
    tree_xyz_min[i] = p4est->connectivity->vertices[3*v_m + i];

  p4est_qcoord_t xc = quad->x;
  if(dir != dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00) == f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[0] = tree_dimensions[0]*(double)xc/(double)P4EST_ROOT_LEN + tree_xyz_min[0];

  p4est_qcoord_t yc = quad->y;
  if(dir != dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0) == f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[1] = tree_dimensions[1]*(double)yc/(double)P4EST_ROOT_LEN + tree_xyz_min[1];

#ifdef P4_TO_P8
  p4est_qcoord_t zc = quad->z;
  if(dir != dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p) == f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz[2] = tree_dimensions[2]*(double)zc/(double)P4EST_ROOT_LEN + tree_xyz_min[2];
#endif
}

void my_p4est_faces_t::rel_xyz_face_fr_node(const p4est_locidx_t& f_idx, const unsigned char& dir, double* xyz_rel, const double* xyz_node, const p4est_indep_t* node, const my_p4est_brick_t* brick,  __int64_t* logical_qcoord_diff) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
  {
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    tree_idx = quad->p.piggy3.which_tree;
  }

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx*P4EST_CHILDREN + 0];
  double tree_xyz_min[P4EST_DIM];
  for(unsigned char i = 0; i < P4EST_DIM; ++i)
    tree_xyz_min[i]       = p4est->connectivity->vertices[3*v_m + i];

  p4est_qcoord_t xc = quad->x;
  if(dir != dir::x)                           xc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_p00) == f_idx) xc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[0] = tree_dimensions[0]*(double)xc/(double)P4EST_ROOT_LEN + tree_xyz_min[0] - xyz_node[0];
  double x_diff_tree = tree_xyz_min[0] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 0];
  p4est_topidx_t tree_x_diff = (p4est_topidx_t) round(x_diff_tree/tree_dimensions[0]); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[0] = tree_x_diff*P4EST_ROOT_LEN + xc - (node->x != P4EST_ROOT_LEN - 1 ? node->x : P4EST_ROOT_LEN); // node might be clamped
  if(periodic[dir::x])
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[0] + i*(xyz_max[0] - xyz_min[0])) < fabs(xyz_rel[0]))
        xyz_rel[0] += i*(xyz_max[0] - xyz_min[0]);
      if(abs(logical_qcoord_diff[0] + i*brick->nxyztrees[0]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[0]))
        logical_qcoord_diff[0] += i*brick->nxyztrees[0]*P4EST_ROOT_LEN;
    }

  p4est_qcoord_t yc = quad->y;
  if(dir != dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0) == f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[1] = tree_dimensions[1]*(double)yc/(double)P4EST_ROOT_LEN + tree_xyz_min[1] - xyz_node[1];
  double y_diff_tree = tree_xyz_min[1] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 1];
  p4est_topidx_t tree_y_diff = (p4est_topidx_t) round(y_diff_tree/tree_dimensions[1]); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[1] = tree_y_diff*P4EST_ROOT_LEN + yc - (node->y != P4EST_ROOT_LEN - 1 ? node->y : P4EST_ROOT_LEN); // node might be clamped
  if(periodic[dir::y])
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[1] + i*(xyz_max[1] - xyz_min[1])) < fabs(xyz_rel[1]))
        xyz_rel[1] += i*(xyz_max[1] - xyz_min[1]);
      if(abs(logical_qcoord_diff[1] + i*brick->nxyztrees[1]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[1]))
        logical_qcoord_diff[1] += i*brick->nxyztrees[1]*P4EST_ROOT_LEN;
    }

#ifdef P4_TO_P8
  p4est_qcoord_t zc = quad->z;
  if(dir != dir::z)                           zc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_00p) == f_idx) zc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[2] = tree_dimensions[2]*(double)zc/(double)P4EST_ROOT_LEN + tree_xyz_min[2] - xyz_node[2];
  double z_diff_tree = tree_xyz_min[2] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 2];
  p4est_topidx_t tree_z_diff = (p4est_topidx_t) round(z_diff_tree/tree_dimensions[2]); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[2] = tree_z_diff*P4EST_ROOT_LEN + zc - (node->z != P4EST_ROOT_LEN - 1 ? node->z : P4EST_ROOT_LEN); // node might be clamped
  if(periodic[dir::z])
    for (char i = -1; i < 2; i+=2)
    {
      if(fabs(xyz_rel[2] + i*(xyz_max[2] - xyz_min[2])) < fabs(xyz_rel[2]))
        xyz_rel[2] += i*(xyz_max[2] - xyz_min[2]);
      if(abs(logical_qcoord_diff[2] + i*brick->nxyztrees[2]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[2]))
        logical_qcoord_diff[2] += i*brick->nxyztrees[2]*P4EST_ROOT_LEN;
    }
#endif
}

#ifdef P4_TO_P8
double my_p4est_faces_t::face_area_in_negative_domain(p4est_locidx_t f_idx, const unsigned char &dir, const double *phi_p, const p4est_nodes_t* nodes) const
#else
double my_p4est_faces_t::face_area_in_negative_domain(p4est_locidx_t f_idx, const unsigned char &dir, const double *phi_p, const p4est_nodes_t* nodes, const double *phi_dd[]) const
#endif
{
#ifdef CASL_THROWS
  if(phi_p != NULL && nodes == NULL)
    throw std::invalid_argument("my_p4est_faces_t::face_area: if the node-sampled levelset function is provided, the nodes MUST be provided as well.");
#endif

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    P4EST_ASSERT(tree_idx >= 0);
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

  const unsigned char tmp = (q2f(quad_idx, 2*dir) == f_idx ? 0 : 1);
  double area = 1.0;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    if(dim == dir)
      continue;
    area *= tree_dimensions[dir]/((double) (1 << quad->level));
  }
  if(phi_p != NULL)
  {
    p4est_locidx_t node_indices[2*(P4EST_DIM - 1)];
    unsigned char xxx, yyy;
#ifdef P4_TO_P8
    unsigned char zzz;
#endif
    for (unsigned char first = 0; first < P4EST_DIM - 1; ++first)
      for (unsigned char second = 0; second < 2; ++second) {
#ifdef P4_TO_P8
        zzz = (dir == dir::z ? tmp : first);
        yyy = (dir == dir::y ? tmp : (dir == dir::z ? first : second));
#else
        yyy = (dir == dir::y ? tmp : second);
#endif
        xxx = (dir == dir::x ? tmp : second);
        node_indices[2*first+second] = nodes->local_nodes[P4EST_CHILDREN*quad_idx + SUMD(xxx, 2*yyy, 4*zzz)];
      }
    bool they_are_all_positive    = true;
    bool at_least_one_is_positive = false;
    for (unsigned char kk = 0; kk < 2*(P4EST_DIM - 1); ++kk) {
      bool node_is_in_positive_domain = phi_p[node_indices[kk]] > 0.0;
      they_are_all_positive     = they_are_all_positive && node_is_in_positive_domain;
      at_least_one_is_positive  = at_least_one_is_positive || node_is_in_positive_domain;
    }
    if (they_are_all_positive)
      return 0.0;
    if (at_least_one_is_positive)
    {
#ifndef P4_TO_P8
      if(phi_dd == NULL)
        area *= fraction_Interval_Covered_By_Irregular_Domain(phi_p[node_indices[0]], phi_p[node_indices[1]], area, area);
      else
        area *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[node_indices[0]], phi_p[node_indices[1]], phi_dd[dir][node_indices[0]], phi_dd[dir][node_indices[1]], area);
#else
      Cube2 my_face(0.0, 1.0, 0.0, 1.0);
      QuadValue ls_value(phi_p[node_indices[0]], phi_p[node_indices[1]], phi_p[node_indices[2]], phi_p[node_indices[3]]);
      area *= my_face.area_In_Negative_Domain(ls_value);
#endif
    }
  }
  return area;
}

PetscErrorCode VecCreateGhostFacesBlock(const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, const unsigned char &dir)
{
  std::vector<PetscInt> ghost_faces(faces->num_ghost[dir], 0);
  for(size_t i = 0; i < ghost_faces.size(); ++i)
    ghost_faces[i] = faces->ghost_local_num[dir][i] + faces->proc_offset[dir][faces->nonlocal_ranks[dir][i]];

  PetscErrorCode ierr = 0;
  if(block_size > 1){
    ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, faces->num_local[dir]*block_size, faces->proc_offset[dir][p4est->mpisize]*block_size,
        ghost_faces.size(), (const PetscInt*)&ghost_faces[0], v); CHKERRQ(ierr);
  } else {
    ierr = VecCreateGhost(p4est->mpicomm, faces->num_local[dir], faces->proc_offset[dir][p4est->mpisize],
        ghost_faces.size(), (const PetscInt*)&ghost_faces[0], v); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateNoGhostFacesBlock(const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, const unsigned char &dir)
{
  PetscErrorCode ierr = 0;
  ierr = VecCreateMPI(p4est->mpicomm, faces->num_local[dir]*block_size, faces->proc_offset[dir][p4est->mpisize]*block_size, v); CHKERRQ(ierr);
  if(block_size > 1){
    ierr = VecSetBlockSize(*v, block_size); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

void check_if_faces_are_well_defined(my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, const unsigned char &dir,
                                     Vec phi, BoundaryConditionType interface_type, Vec face_is_well_defined)
{
  PetscErrorCode ierr;

  if(interface_type == NOINTERFACE)
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

  const double *dxyz = faces->get_smallest_dxyz();
  double xyz_face[P4EST_DIM];

  if(interface_type == DIRICHLET)
  {
    for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx)
    {
      faces->xyz_fr_f(f_idx, dir, xyz_face);
      face_is_well_defined_p[f_idx] = interp(xyz_face) <= 0.0;
    }
  }
  else /* NEUMANN */
  {
    for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx)
    {
      faces->xyz_fr_f(f_idx, dir, xyz_face);
      bool well_defined = false;
      for (char xxx = -1; xxx < 2 && !well_defined; xxx+=2)
        for (char yyy = -1; yyy < 2 && !well_defined; yyy+=2)
#ifdef P4_TO_P8
          for (char zzz = -1; zzz < 2 && !well_defined; zzz+=2)
#endif
          {
            double xyz_eval[P4EST_DIM] = {DIM(xyz_face[0] + xxx*0.5*dxyz[0], xyz_face[1] + yyy*0.5*dxyz[1], xyz_face[2] + zzz*0.5*dxyz[2])};
            well_defined = well_defined || interp(xyz_eval) <= 0.0;
          }
      face_is_well_defined_p[f_idx] = well_defined; // implicit conversion from bool to PetscScalar
    }
  }

  ierr = VecRestoreArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

double interpolate_f_at_node_n(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_faces_t *faces,
                               my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n,
                               p4est_locidx_t node_idx, Vec f, const unsigned char &dir,
                               Vec face_is_well_defined, int order, BoundaryConditionsDIM *bc, face_interpolator* interpolator_from_faces)
{
  PetscErrorCode ierr;

  double xyz[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est, nodes, xyz);

  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);

  if(bc!=NULL && is_node_Wall(p4est, node) && bc[dir].wallType(xyz)==DIRICHLET)
  {
    if(interpolator_from_faces!=NULL)
    {
      interpolator_from_faces->resize(1);
      interpolator_from_faces->at(0).face_idx = -1;
      interpolator_from_faces->at(0).weight   = +1.0;
    }
    return bc[dir].wallValue(xyz);
  }

  const double *dxyz    = faces->get_smallest_dxyz();
  const double *xyz_max = faces->get_xyz_max();
  const double *xyz_min = faces->get_xyz_min();
  const int8_t max_lvl  = ((splitting_criteria_t *) p4est->user_pointer)->max_lvl;
  const double qh = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));
  double domain_size[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    domain_size[dim] = xyz_max[dim] - xyz_min[dim];


  /* gather the neighborhood */
#ifdef CASL_THROWS
  bool is_local = false;
#endif
  set_of_neighboring_quadrants ngbd_tmp;
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  double scaling = DBL_MAX;
  for(char i = -1; i < 2; i += 2)
    for(char j = -1; j < 2; j += 2)
#ifdef P4_TO_P8
      for(char k = -1; k < 2; k += 2)
#endif
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx, DIM(i, j, k), quad_idx, tree_idx);
        if(quad_idx != NOT_A_VALID_QUADRANT)
        {
          p4est_quadrant_t quad;
          if(quad_idx < p4est->local_num_quadrants)
          {
            p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
            quad = *p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
          }
          else
            quad = *p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

          quad.p.piggy3.local_num = quad_idx;

#ifdef CASL_THROWS
          is_local = is_local || quad_idx < p4est->local_num_quadrants;
#endif

          ngbd_tmp.insert(quad);
          scaling = MIN(scaling, .5*qh*(double) (1 << (max_lvl - quad.level)));

          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, DIM(i, 0, 0));
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, DIM(0, j, 0));
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx,     0, 0, k );
#endif
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, DIM(i, j, 0));
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx,     i, 0, k );
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx,     0, j, k );
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx,     i, j, k );
#endif
        }
      }

#ifdef CASL_THROWS
  if(!is_local)
  {
    ierr = PetscPrintf(p4est->mpicomm, "Warning !! interpolate_f_at_node_n: the node is not local."); CHKERRXX(ierr);
  }
#endif

  std::set<indexed_and_located_face> face_ngbd;
  add_faces_to_set_and_clear_set_of_quad(faces, NO_VELOCITY, dir, face_ngbd, ngbd_tmp); // NO_VELOCITY for 2nd argument, because no "center_seed", we are not constructing a Voronoi cell --> bypass the check

  double *f_p;
  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  vector<p4est_locidx_t> interp_points;
  if(interpolator_from_faces != NULL)
    interpolator_from_faces->resize(0);
  matrix_t A;
  /* [Raphael Egan (01/22/2020): results seem more accurate (2nd order vs 1st order) when disregarding the Neumann wall boundary condition (from tests in main file for testing poisson_faces)...] */
  /* --> TO DO : test if SHS still behaves OK when not doing anyting for Neumann wall BC! */
//  bool neumann_wall_x = (bc != NULL && (is_node_xmWall(p4est, node) || is_node_xpWall(p4est, node)) && bc[dir].wallType(xyz) == NEUMANN);
//  bool neumann_wall_y = (bc != NULL && (is_node_ymWall(p4est, node) || is_node_ypWall(p4est, node)) && bc[dir].wallType(xyz) == NEUMANN);
//#ifdef P4_TO_P8
//  bool neumann_wall_z = (bc != NULL && (is_node_zmWall(p4est, node) || is_node_zpWall(p4est, node)) && bc[dir].wallType(xyz) == NEUMANN);
//#endif
  A.resize(1, (order >= 2 ? 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2 : 1 + P4EST_DIM)/* - SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/);
  vector<double> p;
  vector<double> nb[P4EST_DIM];

  double min_w = 1e-6;
  double inv_max_w = 1e-6;

  const PetscScalar *face_is_well_defined_p;
  if(face_is_well_defined != NULL)
    ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(std::set<indexed_and_located_face>::const_iterator it = face_ngbd.begin(); it != face_ngbd.end() ; it++)
  {
    /* minus direction */
    const indexed_and_located_face &neighbor_face = *it;
    if((face_is_well_defined == NULL || face_is_well_defined_p[neighbor_face.face_idx]) && std::find(interp_points.begin(), interp_points.end(), neighbor_face.face_idx) == interp_points.end())
    {
      if(interpolator_from_faces!=NULL)
      {
        face_interpolator_element new_element; new_element.face_idx = neighbor_face.face_idx;
        interpolator_from_faces->push_back(new_element);
      }
      double xyz_t[P4EST_DIM] = {DIM(neighbor_face.xyz_face[0], neighbor_face.xyz_face[1], neighbor_face.xyz_face[2])};
      for(unsigned char i = 0; i < P4EST_DIM; ++i)
      {
        double rel_dist = (xyz[i] - xyz_t[i]);
        if(faces->periodicity(i))
          for (char cc = -1; cc < 2; cc+=2)
            if(fabs(xyz[i] - xyz_t[i] + cc*domain_size[i]) < fabs(rel_dist))
              rel_dist = (xyz[i] - xyz_t[i] + cc*domain_size[i]);
        xyz_t[i] = rel_dist / scaling;
      }

      double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

      A.set_value(interp_points.size(), 0,                                                                                                          1.0               * w);
//      if(!neumann_wall_x)
      A.set_value(interp_points.size(), 1,                                                                                                          xyz_t[0]          * w);
//      if(!neumann_wall_y)
      A.set_value(interp_points.size(), 2 /*- (neumann_wall_x ? 1 : 0)*/,                                                                           xyz_t[1]          * w);
#ifdef P4_TO_P8
//      if(!neumann_wall_z)
      A.set_value(interp_points.size(), 3 /*- (neumann_wall_x ? 1 : 0) - (neumann_wall_y ? 1 : 0)*/,                                                xyz_t[2]          * w);
#endif
      if(order >= 2)
      {
        A.set_value(interp_points.size(),   1 + P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(),   2 + P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[0]*xyz_t[1] * w);
#ifdef P4_TO_P8
        A.set_value(interp_points.size(),   3 + P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[0]*xyz_t[2] * w);
#endif
        A.set_value(interp_points.size(), 1 + 2*P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[1]*xyz_t[1] * w);
#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 2 + 2*P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(),     3*P4EST_DIM /*- SUMD((neumann_wall_x ? 1 : 0), (neumann_wall_y ? 1 : 0), (neumann_wall_z ? 1 : 0))*/, xyz_t[2]*xyz_t[2] * w);
#endif
      }

//      const double neumann_term = SUMD((neumann_wall_x ? bc->wallValue(xyz)*xyz_t[0]*scaling : 0.0),
//                                       (neumann_wall_y ? bc->wallValue(xyz)*xyz_t[1]*scaling : 0.0),
//                                       (neumann_wall_z ? bc->wallValue(xyz)*xyz_t[2]*scaling : 0.0));
      p.push_back((f_p[neighbor_face.face_idx] /*+ neumann_term*/) * w);
      if(interpolator_from_faces != NULL)
        interpolator_from_faces->back().weight = w;
      // [Raphael:] note the sign used when defining xyz_t above, it is counter-intuitive, imo

      for(unsigned char d = 0; d < P4EST_DIM; ++d)
        if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
          nb[d].push_back(xyz_t[d]);

      interp_points.push_back(neighbor_face.face_idx);
    }
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
  if(face_is_well_defined != NULL)
    ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  if(interp_points.size() == 0)
  {
    if(interpolator_from_faces != NULL)
      interpolator_from_faces->resize(0);
    return 0;
  }

  double abs_max = A.scale_by_maxabs(p);
  std::vector<double>* interp_weights = NULL;
  if(interpolator_from_faces != NULL)
    interp_weights = new std::vector<double>(0);

  double value_to_return = solve_lsqr_system(A, p, DIM(nb[0].size(), nb[1].size(), nb[2].size()), order);

  if(interpolator_from_faces != NULL)
  {
    P4EST_ASSERT(interp_weights->size() <= interpolator_from_faces->size());
    interpolator_from_faces->resize(interp_weights->size());
    for (size_t k = 0; k < interpolator_from_faces->size(); k++)
      interpolator_from_faces->at(k).weight *= interp_weights->at(k)/abs_max;
  }
  if(interp_weights != NULL)
    delete interp_weights;

#ifdef DEBUG
  if(interpolator_from_faces != NULL)
  {
    double my_new_value = 0.0;
    const double *f_read_p;
    ierr = VecGetArrayRead(f, &f_read_p); CHKERRXX(ierr);
    for (size_t k = 0; k < interpolator_from_faces->size(); ++k)
      my_new_value += f_read_p[interpolator_from_faces->at(k).face_idx]*interpolator_from_faces->at(k).weight;
    ierr = VecRestoreArrayRead(f, &f_read_p); CHKERRXX(ierr);
    P4EST_ASSERT(fabs(my_new_value - value_to_return) < MAX(EPS, 1e-6*MAX(fabs(my_new_value), fabs(value_to_return))));
  }
#endif

  return value_to_return;
}

voro_cell_type compute_voronoi_cell(Voronoi_DIM &voronoi_cell, const my_p4est_faces_t* faces, const p4est_locidx_t &f_idx, const unsigned char &dir, const BoundaryConditionsDIM *bc, const PetscScalar *face_is_well_defined_p)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  p4est_t* p4est = faces->get_p4est();
#else
  const p4est_t* p4est = faces->get_p4est();
#endif
  const my_p4est_cell_neighbors_t *ngbd_c = faces->get_ngbd_c();
  const double *dxyz    = faces->get_smallest_dxyz();

  voronoi_cell.clear();
  double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz_face);
  voronoi_cell.set_center_point(ONLY3D(f_idx COMMA) xyz_face);

  // check first if the neighbors of the finest faces were stored during construction of faces
  // and if the face of interest is one of the finest quadrants' in the uniform region
  // in that case, the task is straightforward.
  const uniform_face_ngbd* face_neighbors;
  if(faces->found_uniform_face_neighborhood(f_idx, dir, face_neighbors) && no_wall_in_face_neighborhood(face_neighbors))
  {
#ifdef DEBUG
    p4est_quadrant_t qm, qp;
    faces->find_quads_touching_face(f_idx, dir, qm, qp);
    P4EST_ASSERT(qm.level == qp.level && qm.level == ((splitting_criteria_t*) (faces->get_p4est())->user_pointer)->max_lvl);
#endif
    vector<ngbdDIMseed> points(P4EST_FACES);
#ifdef P4_TO_P8
    vector<Point2> partition(P4EST_FACES);
#endif
    for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
#ifdef P4_TO_P8
      points[face_dir].n  = face_neighbors->neighbor_face_idx[face_dir];
      faces->point_fr_f(points[face_dir].n, dir, points[face_dir].p);
      points[face_dir].s  = (face_dir/2 == dir::x ? dxyz[1]*dxyz[2] : (face_dir/2 == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]));
#else
      unsigned char idx   = face_order_to_counterclock_cycle_order[face_dir];
      points[idx].n       = face_neighbors->neighbor_face_idx[face_dir];
      faces->point_fr_f(points[idx].n, dir, points[idx].p);
      points[idx].theta   = (face_dir/2)*M_PI_2 + (1.0 - face_dir%2)*M_PI;
      partition[idx].x    = points[idx].p.x + (0.5 - (face_dir%2))*dxyz[0];
      partition[idx].y    = points[idx].p.y + (1.0 - 2.0*(face_dir/2))*(face_dir%2 - 0.5)*dxyz[1];
#endif
    }
#ifdef P4_TO_P8
    voronoi_cell.set_cell(points, dxyz[0]*dxyz[1]*dxyz[2]);
#else
    voronoi_cell.set_neighbors_and_partition(points, partition, dxyz[0]*dxyz[1]);
#endif
    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    return parallelepiped_no_wall;
  }

  // check if well-defined (if using those tags) : if far in the positive domain and if not solving there, we don't need anything here
  if(face_is_well_defined_p != NULL && !face_is_well_defined_p[f_idx])
  {
    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    return not_well_defined;
  }

  p4est_quadrant_t qm, qp;
  faces->find_quads_touching_face(f_idx, dir, qm, qp);

  /* check for DIRICHLET wall faces */
  const bool is_wall_face = qm.p.piggy3.local_num == -1 || qp.p.piggy3.local_num == -1;
  if(is_wall_face && bc[dir].wallType(xyz_face) == DIRICHLET)
  {
    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    return dirichlet_wall_face;
  }

  /* find direct neighbors */
  /* Gather the neighbor cells to get the potential voronoi neighbors */
  set_of_neighboring_quadrants ngbd_m_[2*(P4EST_DIM - 1)]; // neighbors of qm in transverse cartesian direction
  set_of_neighboring_quadrants ngbd_p_[2*(P4EST_DIM - 1)]; // neighbors of qp in transverse cartesian direction
  P4EST_ASSERT(ORD(dir == dir::x, dir == dir::y, dir == dir::z));
  unsigned char ngbd_idx = 0;
  for (unsigned char neigbor_dir = 0; neigbor_dir < P4EST_FACES; ++neigbor_dir) {
    if(neigbor_dir/2 == dir)
      continue;
    char search[P4EST_DIM]  = {DIM(0, 0, 0)};
    search[neigbor_dir/2]   = (neigbor_dir%2 == 1 ?  1 : -1);
    if(qm.p.piggy3.local_num != -1)
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_[ngbd_idx], qm.p.piggy3.local_num, qm.p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
    if(qp.p.piggy3.local_num != -1)
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_[ngbd_idx], qp.p.piggy3.local_num, qp.p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
    ngbd_idx++;
  }
  P4EST_ASSERT(ngbd_idx == 2*(P4EST_DIM - 1));

  /* now gather the neighbor cells to get the potential voronoi neighbors */

  /* check for uniform case and/or wall in the neighborhood, if so build voronoi partition by hand */
  bool no_wall = (qp.p.piggy3.local_num != -1) && (qm.p.piggy3.local_num != -1);
  // if the face is wall itself, check that the other face across the cell is not subrefined
  bool has_uniform_ngbd =
      (qp.p.piggy3.local_num == -1 && faces->q2f(qm.p.piggy3.local_num, 2*dir) != NO_VELOCITY)
      || (qm.p.piggy3.local_num == -1 && faces->q2f(qp.p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY)
      || ((qp.level == qm.level) && faces->q2f(qm.p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(qp.p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY);
  for (unsigned char k = 0; k < 2*(P4EST_DIM - 1) && (has_uniform_ngbd || no_wall); ++k)
  {
    if(qp.p.piggy3.local_num != -1)
    {
      no_wall           = no_wall && ngbd_p_[k].size() > 0;
      has_uniform_ngbd  = has_uniform_ngbd && ngbd_p_[k].size() == 1 && ngbd_p_[k].begin()->level == qp.level
          && faces->q2f(ngbd_p_[k].begin()->p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(ngbd_p_[k].begin()->p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY;
    }
    if(qm.p.piggy3.local_num != -1)
    {
      no_wall           = no_wall && ngbd_m_[k].size() > 0;
      has_uniform_ngbd  = has_uniform_ngbd && ngbd_m_[k].size() == 1 && ngbd_m_[k].begin()->level == qm.level
          && faces->q2f(ngbd_m_[k].begin()->p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(ngbd_m_[k].begin()->p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY;
    }
  }
  if(has_uniform_ngbd && no_wall)
  {
    P4EST_ASSERT(qm.level <= ((splitting_criteria_t*) p4est->user_pointer)->max_lvl); // consistency check
#ifdef P4EST_DEBUG
    for (unsigned char k = 0; k < 2*(P4EST_DIM - 1); ++k)
      P4EST_ASSERT(faces->q2f(ngbd_m_[k].begin()->p.piggy3.local_num, 2*dir + 1) == faces->q2f(ngbd_p_[k].begin()->p.piggy3.local_num, 2*dir)); // consistency check
#endif
    const double cell_ratio = (double) (1 << (((splitting_criteria_t *) p4est->user_pointer)->max_lvl - qm.level));
    // neighbor faces in the direction of the face orientation, first:
#ifdef P4_TO_P8
    vector<ngbd3Dseed> points(P4EST_FACES);
    points[2*dir].n   = faces->q2f(qm.p.piggy3.local_num, 2*dir);
    faces->point_fr_f(points[2*dir].n, dir, points[2*dir].p);
    points[2*dir].s   = (dir == dir::x ? dxyz[1]*dxyz[2] : (dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);

    points[2*dir + 1].n = faces->q2f(qp.p.piggy3.local_num, 2*dir + 1);
    faces->point_fr_f(points[2*dir + 1].n, dir, points[2*dir + 1].p);
    points[2*dir + 1].s = (dir == dir::x ? dxyz[1]*dxyz[2] : (dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);
#else
    vector<ngbd2Dseed> points(P4EST_FACES);
    vector<Point2> partition(P4EST_FACES);
    unsigned char idx;
    idx               = face_order_to_counterclock_cycle_order[2*dir];
    points[idx].n     = faces->q2f(qm.p.piggy3.local_num, 2*dir);
    faces->point_fr_f(points[idx].n, dir, points[idx].p);
    points[idx].theta = dir*M_PI_2 + M_PI;
    partition[idx].x  = points[idx].p.x +         0.5*dxyz[0]*cell_ratio;
    partition[idx].y  = points[idx].p.y + (dir - 0.5)*dxyz[1]*cell_ratio;

    idx               = face_order_to_counterclock_cycle_order[2*dir + 1];
    points[idx].n     = faces->q2f(qp.p.piggy3.local_num, 2*dir + 1);
    faces->point_fr_f(points[idx].n, dir, points[idx].p);
    points[idx].theta = dir*M_PI_2;
    partition[idx].x  = points[idx].p.x -         0.5*dxyz[0]*cell_ratio;
    partition[idx].y  = points[idx].p.y + (0.5 - dir)*dxyz[1]*cell_ratio;
#endif
    // neighbor faces in the tranverse direction(s), then:
    unsigned char ngbd_idx = 0;
    for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
      if(face_dir/2 == dir)
        continue;
      else
      {
#ifdef P4_TO_P8
        points[face_dir].n  = faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir); // we loop through ngbd_p_ in the same order as when it was built
        faces->point_fr_f(points[face_dir].n, dir, points[face_dir].p);
        points[face_dir].s  = (face_dir == dir::x ? dxyz[1]*dxyz[2] : (face_dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);
#else
        idx = face_order_to_counterclock_cycle_order[face_dir];
        points[idx].n     = faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir); // we loop through ngbd_p_ in the same order as when it was built
        faces->point_fr_f(points[idx].n, dir, points[idx].p);
        points[idx].theta = (face_dir/2)*M_PI_2 + (1.0 - face_dir%2)*M_PI;
        partition[idx].x  = points[idx].p.x + (0.5 - face_dir%2)*dxyz[0]*cell_ratio;
        partition[idx].y  = points[idx].p.y + (1.0 - 2.0*(face_dir/2))*(face_dir%2 - 0.5)*dxyz[1]*cell_ratio;
#endif
        ngbd_idx++;
      }
    }
    P4EST_ASSERT(ngbd_idx == 2*(P4EST_DIM - 1));
#ifdef P4_TO_P8
    voronoi_cell.set_cell(points, dxyz[0]*dxyz[1]*dxyz[2]*pow(cell_ratio, 3.0));
#else
    voronoi_cell.set_neighbors_and_partition(points, partition, dxyz[0]*dxyz[1]*SQR(cell_ratio));
#endif
    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    return parallelepiped_no_wall;
  }
  /* otherwise, either
   * 1) the face is a non-Dirichlet wall face,
   * 2) there is a wall nearby,
   * 3) there is a T-junction and the grid is not uniform
   * --> need to compute the voronoi cell */
  else
  {
    /* gather neighbor cells:
     * find neighbor quadrants of touching quadrants in (all possible) transverse orientations + one more layer of such in the positive and negative face-normals
     */
    set_of_neighboring_quadrants ngbd; ngbd.clear();
    std::set<indexed_and_located_face> set_of_neighbor_faces; set_of_neighbor_faces.clear();
    // we add faces to the set as we find them and clear the list of neighbor quadrants after every search to avoid vectors growing very large and slowing down the O(n) searches implemented in find_neighbor_cells_of_cell
    for (char face_touch = -1; face_touch < 2; face_touch += 2) {
      p4est_locidx_t quadrant_touch_idx = (face_touch == -1 ? qm.p.piggy3.local_num : qp.p.piggy3.local_num);
      if(quadrant_touch_idx != -1)
      {
        p4est_topidx_t tree_touch_idx = (face_touch == -1 ? qm.p.piggy3.which_tree : qp.p.piggy3.which_tree);
        p4est_quadrant_t& quad_touch  = (face_touch == -1 ? qm : qp);
        const unsigned char dir_touch = 2*dir + (face_touch == -1 ? 0 : 1);
        ngbd.insert(quad_touch);

        // in face normal direction if needed
        if (faces->q2f(quadrant_touch_idx, dir_touch) == NO_VELOCITY)
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, dir_touch);
        // in all tranverse cartesian directions
        for (unsigned char k = 0; k < 2*(P4EST_DIM - 1); ++k)
          add_faces_to_set_and_clear_set_of_quad(faces, f_idx, dir, set_of_neighbor_faces, (face_touch == -1 ? ngbd_m_[k] : ngbd_p_[k]));
        char search[P4EST_DIM];
#ifdef P4_TO_P8
        // in all transverse "diagonal directions"
        search[dir] = 0;
        for (char iii = -1; iii < 2; iii += 2) {
          search[(dir + 1)%P4EST_DIM] = iii;
          for (char jjj = -1; jjj < 2; jjj += 2) {
            search[(dir + 2)%P4EST_DIM] = jjj;
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1], search[2]);
          }
        }
#endif
        // extra layer
        search[dir] = face_touch;
        for (char iii = -1; iii < 2; ++iii)
        {
#ifdef P4_TO_P8
          for (char jjj = -1; jjj < 2; ++jjj)
          {
            if(iii == 0 && jjj == 0)
              continue;
            search[(dir + 1)%P4EST_DIM] = iii;
            search[(dir + 2)%P4EST_DIM] = jjj;
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1], search[2]);
          }
#else
          if(iii == 0)
            continue;
          search[(dir + 1)%P4EST_DIM] = iii;
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1]);
#endif
        }
        add_faces_to_set_and_clear_set_of_quad(faces, f_idx, dir, set_of_neighbor_faces, ngbd);
      }
    }

    const bool *periodic  = faces->get_periodicity();
    const double *xyz_min = faces->get_xyz_min();
    const double *xyz_max = faces->get_xyz_max();
    voronoi_cell.assemble_from_set_of_faces(set_of_neighbor_faces, periodic, xyz_min, xyz_max);

    /* add the walls in 2d, note that they are dealt with by voro++ in 3D
     * This needs to be done AFTER assemble_from_set_of_faces because the latter starts by clearing the neighbor seeds
     * */
#ifndef P4_TO_P8
    const double cell_ratio = (double) (1 << (((splitting_criteria_t *) p4est->user_pointer)->max_lvl - MAX(qm.level, qp.level)));
    const unsigned char other_cartesian_dir = (dir == dir::x ? dir::y : dir::x);
    if(qm.p.piggy3.local_num == -1 && bc[dir].wallType(xyz_face) == NEUMANN)
      voronoi_cell.push(WALL_idx(2*dir),   xyz_face[0] - (dir == dir::x ? dxyz[0]*cell_ratio : 0.0), xyz_face[1] - (dir == dir::y ? dxyz[1]*cell_ratio : 0.0), periodic, xyz_min, xyz_max);
    if(qp.p.piggy3.local_num == -1 && bc[dir].wallType(xyz_face) == NEUMANN)
      voronoi_cell.push(WALL_idx(2*dir+1), xyz_face[0] + (dir == dir::x ? dxyz[0]*cell_ratio : 0.0), xyz_face[1] + (dir == dir::y ? dxyz[1]*cell_ratio : 0.0), periodic, xyz_min, xyz_max);
    if ((qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, 2*other_cartesian_dir)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, 2*other_cartesian_dir)))
      voronoi_cell.push(WALL_idx(2*other_cartesian_dir),      xyz_face[0] - (other_cartesian_dir == dir::x ? dxyz[0]*cell_ratio : 0.0), xyz_face[1] - (other_cartesian_dir == dir::y ? dxyz[1]*cell_ratio : 0.0), periodic, xyz_min, xyz_max);
    if ((qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, 2*other_cartesian_dir + 1)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, 2*other_cartesian_dir + 1)))
      voronoi_cell.push(WALL_idx(2*other_cartesian_dir + 1),  xyz_face[0] + (other_cartesian_dir == dir::x ? dxyz[0]*cell_ratio : 0.0), xyz_face[1] + (other_cartesian_dir == dir::y ? dxyz[1]*cell_ratio : 0.0), periodic, xyz_min, xyz_max);
#endif

#ifdef P4_TO_P8
    voronoi_cell.construct_partition(xyz_min, xyz_max, periodic);
#else
    voronoi_cell.construct_partition();
    voronoi_cell.compute_volume();
#endif
    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    if(has_uniform_ngbd)
      return parallelepiped_with_wall; // it for sure is a parallelepiped since it had a uniform neighborhood, but the face must have some wall neighbor(s), that's all
    else
      return nonuniform;
  }
}
