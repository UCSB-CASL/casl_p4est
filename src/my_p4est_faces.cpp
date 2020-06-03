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


my_p4est_faces_t::my_p4est_faces_t(p4est_t *p4est_, p4est_ghost_t *ghost_, const my_p4est_brick_t *myb_, my_p4est_cell_neighbors_t *ngbd_c_, bool initialize_neighborhoods_of_fine_faces)
  : max_p4est_lvl(((splitting_criteria_t*) p4est_->user_pointer)->max_lvl),
    smallest_dxyz{DIM(ngbd_c_->get_tree_dimensions()[0]/(1 << (((splitting_criteria_t*) p4est_->user_pointer)->max_lvl)),
                  ngbd_c_->get_tree_dimensions()[1]/(1 << (((splitting_criteria_t*) p4est_->user_pointer)->max_lvl)),
    ngbd_c_->get_tree_dimensions()[2]/(1 << (((splitting_criteria_t*) p4est_->user_pointer)->max_lvl)))}
{
  this->p4est   = p4est_;
  this->ghost   = ghost_;
  this->ngbd_c  = ngbd_c_;
  myb           = myb_;

  tree_dimensions   = ngbd_c->get_tree_dimensions();
  xyz_min           = myb->xyz_min;
  xyz_max           = myb->xyz_max;
  periodic          = ngbd_c->get_hierarchy()->get_periodicity();

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

void my_p4est_faces_t::rel_qxyz_face_fr_node(const p4est_locidx_t& f_idx, const unsigned char& dir, double* xyz_rel, const double* xyz_node, const p4est_indep_t* node, int64_t* logical_qcoord_diff) const
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
  xyz_rel[0] = tree_dimensions[0]*(double)xc/(double)P4EST_ROOT_LEN + tree_xyz_min[0] - xyz_node[0];
  double x_diff_tree = tree_xyz_min[0] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 0];
  p4est_topidx_t tree_x_diff = (p4est_topidx_t) round(x_diff_tree/tree_dimensions[0]); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[0] = tree_x_diff*P4EST_ROOT_LEN + xc - (node->x != P4EST_ROOT_LEN - 1 ? node->x : P4EST_ROOT_LEN); // node might be clamped
  if(periodic[dir::x])
    for (char i = -1; i < 2; i += 2)
    {
      if(fabs(xyz_rel[0] + i*(xyz_max[0] - xyz_min[0])) < fabs(xyz_rel[0]))
        xyz_rel[0] += i*(xyz_max[0] - xyz_min[0]);
      if(abs(logical_qcoord_diff[0] + i*myb->nxyztrees[0]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[0]))
        logical_qcoord_diff[0] += i*myb->nxyztrees[0]*P4EST_ROOT_LEN;
    }

  p4est_qcoord_t yc = quad->y;
  if(dir != dir::y)                           yc += .5*P4EST_QUADRANT_LEN(quad->level);
  else if(q2f(quad_idx, dir::f_0p0) == f_idx) yc +=    P4EST_QUADRANT_LEN(quad->level);
  xyz_rel[1] = tree_dimensions[1]*(double)yc/(double)P4EST_ROOT_LEN + tree_xyz_min[1] - xyz_node[1];
  double y_diff_tree = tree_xyz_min[1] - p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*node->p.which_tree + 0] + 1];
  p4est_topidx_t tree_y_diff = (p4est_topidx_t) round(y_diff_tree/tree_dimensions[1]); // assumes trees of constant size across the domain and cartesian block-structured
  logical_qcoord_diff[1] = tree_y_diff*P4EST_ROOT_LEN + yc - (node->y != P4EST_ROOT_LEN - 1 ? node->y : P4EST_ROOT_LEN); // node might be clamped
  if(periodic[dir::y])
    for (char i = -1; i < 2; i += 2)
    {
      if(fabs(xyz_rel[1] + i*(xyz_max[1] - xyz_min[1])) < fabs(xyz_rel[1]))
        xyz_rel[1] += i*(xyz_max[1] - xyz_min[1]);
      if(abs(logical_qcoord_diff[1] + i*myb->nxyztrees[1]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[1]))
        logical_qcoord_diff[1] += i*myb->nxyztrees[1]*P4EST_ROOT_LEN;
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
    for (char i = -1; i < 2; i += 2)
    {
      if(fabs(xyz_rel[2] + i*(xyz_max[2] - xyz_min[2])) < fabs(xyz_rel[2]))
        xyz_rel[2] += i*(xyz_max[2] - xyz_min[2]);
      if(abs(logical_qcoord_diff[2] + i*myb->nxyztrees[2]*P4EST_ROOT_LEN) < abs(logical_qcoord_diff[2]))
        logical_qcoord_diff[2] += i*myb->nxyztrees[2]*P4EST_ROOT_LEN;
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
    area *= tree_dimensions[dim]/((double) (1 << quad->level));
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

void check_if_faces_are_well_defined(const my_p4est_faces_t *faces, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_phi,
                                     const BoundaryConditionsDIM& bc, Vec face_is_well_defined)
{
  PetscErrorCode ierr;

  if(bc.interfaceType() == NOINTERFACE)
  {
    Vec face_is_well_defined_loc;
    ierr = VecGhostGetLocalForm(face_is_well_defined, &face_is_well_defined_loc); CHKERRXX(ierr);
    ierr = VecSet(face_is_well_defined_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined, &face_is_well_defined_loc); CHKERRXX(ierr);
    return;
  }

  PetscScalar *face_is_well_defined_p;
  ierr = VecGetArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(size_t k = 0; k < faces->get_layer_size(dir); ++k)
  {
    p4est_locidx_t f_idx = faces->get_layer_face(dir, k);
    face_is_well_defined_p[f_idx] = local_face_is_well_defined(f_idx, faces, interp_phi, dir, bc); // implicit conversion from bool to PetscScalar
  }
  ierr = VecGhostUpdateBegin(face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t k = 0; k < faces->get_local_size(dir); ++k)
  {
    p4est_locidx_t f_idx = faces->get_local_face(dir, k);
    face_is_well_defined_p[f_idx] = local_face_is_well_defined(f_idx, faces, interp_phi, dir, bc);
  }
  ierr = VecGhostUpdateEnd  (face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); // implicit conversion from bool to PetscScalar

  ierr = VecRestoreArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);
  return;
}

double interpolate_velocity_at_node_n(my_p4est_faces_t *faces, my_p4est_node_neighbors_t *ngbd_n, p4est_locidx_t node_idx, Vec velocity_component, const unsigned char &dir,
                                      Vec face_is_well_defined, int order, BoundaryConditionsDIM *bc, face_interpolator* interpolator_from_faces)
{
  PetscErrorCode ierr;

  const p4est_t* p4est        = faces->get_p4est();
  const p4est_nodes_t* nodes  = ngbd_n->get_nodes();

  double xyz[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est, nodes, xyz);

  /*
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);
  * [RAPHAEL:] substituted this latter line of code by the following to enforce and ensure const attribute in 'nodes' argument...
   */
  SC_ASSERT((size_t) node_idx < nodes->indep_nodes.elem_count);
  const p4est_indep_t* node = (p4est_indep_t*) (nodes->indep_nodes.array + ((size_t) node_idx)*nodes->indep_nodes.elem_size);

  if(bc != NULL && is_node_Wall(p4est, node) && bc[dir].wallType(xyz) == DIRICHLET)
  {
    if(interpolator_from_faces != NULL)
    {
      interpolator_from_faces->resize(1);
      interpolator_from_faces->at(0).face_idx = -1;
      interpolator_from_faces->at(0).weight   = +1.0;
    }
    return bc[dir].wallValue(xyz);
  }

  set_of_neighboring_quadrants cell_neighbors; cell_neighbors.clear();
  /* gather the neighborhood and get the (logical) size of the smallest nearby quadrant */
  const p4est_qcoord_t logical_size_smallest_nearby_cell = faces->get_ngbd_c()->gather_neighbor_cells_of_node(node_idx, nodes, cell_neighbors, true);
  const double* tree_dim = faces->get_tree_dimensions();
  const double scaling = 0.5*MIN(DIM(tree_dim[0], tree_dim[1], tree_dim[2]))*(double)logical_size_smallest_nearby_cell/(double) P4EST_ROOT_LEN;

  std::set<indexed_and_located_face> face_ngbd;
  add_faces_to_set_and_clear_set_of_quad(faces, NO_VELOCITY, dir, face_ngbd, cell_neighbors); // NO_VELOCITY for 2nd argument, because no "center_seed", we are not constructing a Voronoi cell --> bypass the check

  double *velocity_component_p;
  ierr = VecGetArray(velocity_component, &velocity_component_p); CHKERRXX(ierr);

  if(interpolator_from_faces != NULL)
    interpolator_from_faces->resize(0);
  matrix_t A;
  char neumann_wall[P4EST_DIM] = {DIM(0, 0, 0)};
  unsigned char nb_neumann_walls = 0;
  if(order >= 1 && bc != NULL && bc[dir].wallType(xyz) == NEUMANN)
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
    {
      neumann_wall[dd] = (is_node_Wall(p4est, node, 2*dd) ? -1 : (is_node_Wall(p4est, node, 2*dd + 1) ? +1 : 0));
      nb_neumann_walls += abs(neumann_wall[dd]);
    }
  A.resize(face_ngbd.size(), 1 + (order >= 1 ? P4EST_DIM - nb_neumann_walls : 0) + (order >= 2 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0));
  vector<double> p(face_ngbd.size());
  std::set<int64_t> nb[P4EST_DIM];

  const double min_w = 1e-6;
  const double inv_max_w = 1e-6;

  const PetscScalar *face_is_well_defined_p;
  if(face_is_well_defined != NULL)
    ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  unsigned int row_idx = 0;
  for(std::set<indexed_and_located_face>::const_iterator it = face_ngbd.begin(); it != face_ngbd.end() ; it++)
  {
    /* minus direction */
    const indexed_and_located_face &neighbor_face = *it;
    if(face_is_well_defined == NULL || face_is_well_defined_p[neighbor_face.face_idx])
    {
      if(interpolator_from_faces != NULL)
      {
        face_interpolator_element new_element; new_element.face_idx = neighbor_face.face_idx;
        interpolator_from_faces->push_back(new_element);
      }

      double xyz_t[P4EST_DIM];
      int64_t logical_qcoord_diff[P4EST_DIM];
      faces->rel_qxyz_face_fr_node(neighbor_face.face_idx, dir, xyz_t, xyz, node, logical_qcoord_diff);
      for(unsigned char i = 0; i < P4EST_DIM; ++i)
        xyz_t[i] /= scaling;

      double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

      unsigned char col_idx = 0;
      A.set_value(row_idx, col_idx++, w); // constant term --> what we are after in 99.99% of cases
      if(order >= 1)
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
          if(neumann_wall[comp] == 0)
            A.set_value(row_idx, col_idx++, xyz_t[comp]*w); // linear terms, first partial derivatives
      P4EST_ASSERT(col_idx == 1 + (order >= 1 ? P4EST_DIM - nb_neumann_walls : 0));
      if(order >= 2)
        for (unsigned char comp_1 = 0; comp_1 < P4EST_DIM; ++comp_1)
          for (unsigned char comp_2 = comp_1; comp_2 < P4EST_DIM; ++comp_2)
            A.set_value(row_idx, col_idx++, xyz_t[comp_1]*xyz_t[comp_2]*w); // quadratic terms, second (possibly crossed) partial derivatives
      P4EST_ASSERT(col_idx == 1 + (order >= 1 ? P4EST_DIM - nb_neumann_walls : 0) + (order >= 2 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0));

      const double neumann_term = (order >= 1 && nb_neumann_walls > 0 ? SUMD(neumann_wall[0]*bc[dir].wallValue(xyz)*xyz_t[0]*scaling, neumann_wall[1]*bc[dir].wallValue(xyz)*xyz_t[1]*scaling, neumann_wall[2]*bc[dir].wallValue(xyz)*xyz_t[2]*scaling) : 0.0);
      p[row_idx] = (velocity_component_p[neighbor_face.face_idx] - neumann_term) * w;

      if(interpolator_from_faces != NULL)
        interpolator_from_faces->back().weight = w;

      for(unsigned char d = 0; d < P4EST_DIM; ++d)
        nb[d].insert(logical_qcoord_diff[d]);

      row_idx++;
    }
  }

  ierr = VecRestoreArray(velocity_component, &velocity_component_p); CHKERRXX(ierr);
  if(face_is_well_defined != NULL)
    ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  if(row_idx == 0)
  {
    if(interpolator_from_faces != NULL)
      interpolator_from_faces->resize(0);
    return 0.0;
  }

  P4EST_ASSERT(row_idx <= (unsigned int) A.num_rows());
  if(row_idx < (unsigned int) A.num_rows())
  {
    A.resize(row_idx, A.num_cols());
    p.resize(row_idx);
  }

  double abs_max = A.scale_by_maxabs(p);
  std::vector<double>* interp_weights = NULL;
  if(interpolator_from_faces != NULL)
    interp_weights = new std::vector<double>(0);

  const double value_to_return = solve_lsqr_system(A, p, DIM(nb[0].size(), nb[1].size(), nb[2].size()), order, nb_neumann_walls, interp_weights);

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
    const double *velocity_component_read_p;
    ierr = VecGetArrayRead(velocity_component, &velocity_component_read_p); CHKERRXX(ierr);
    for (size_t k = 0; k < interpolator_from_faces->size(); ++k)
      my_new_value += velocity_component_read_p[interpolator_from_faces->at(k).face_idx]*interpolator_from_faces->at(k).weight;
    ierr = VecRestoreArrayRead(velocity_component, &velocity_component_read_p); CHKERRXX(ierr);
    P4EST_ASSERT(fabs(my_new_value - value_to_return) < MAX(EPS, 1e-6*MAX(fabs(my_new_value), fabs(value_to_return))));
  }
#endif

  return value_to_return;
}

voro_cell_type compute_voronoi_cell(Voronoi_DIM &voronoi_cell, const my_p4est_faces_t* faces, const p4est_locidx_t &f_idx, const unsigned char &dir, const BoundaryConditionsDIM *bc, const PetscScalar *face_is_well_defined_p)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);

  const p4est_t* p4est = faces->get_p4est();
  const my_p4est_cell_neighbors_t* ngbd_c = faces->get_ngbd_c();
  const double * dxyz = faces->get_smallest_dxyz();
  const double * tree_dim = faces->get_tree_dimensions();
  const int8_t& max_lvl = ((splitting_criteria_t*) p4est->user_pointer)->max_lvl;

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
    P4EST_ASSERT(qm.level == qp.level && qm.level == ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
#endif
    vector<ngbdDIMseed> points(P4EST_FACES);
#ifndef P4_TO_P8
    vector<Point2> partition(P4EST_FACES);
#endif
    for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
#ifdef P4_TO_P8
      unsigned char idx   = face_dir;
#else
      unsigned char idx   = face_order_to_counterclock_cycle_order[face_dir];
#endif
      points[idx].n  = face_neighbors->neighbor_face_idx[face_dir];
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        points[idx].p.xyz(dim) = xyz_face[dim] + (dim == face_dir/2 ? (face_dir%2 ? +1.0 : -1.0)*dxyz[face_dir/2]: 0.0);
#ifdef P4_TO_P8
      points[idx].s     = (face_dir/2 == dir::x ? dxyz[1]*dxyz[2] : (face_dir/2 == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]));
#else
      points[idx].theta = (face_dir/2)*M_PI_2 + (1.0 - face_dir%2)*M_PI;
      partition[idx].x  = points[idx].p.x + (0.5 - (face_dir%2))*dxyz[0];
      partition[idx].y  = points[idx].p.y + (1.0 - 2.0*(face_dir/2))*(face_dir%2 - 0.5)*dxyz[1];
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

  /* Gather the neighbor cells to get the potential voronoi neighbors:
   * find all neighbors of the quads touching the face, in any tranverse direction
   * */
  const unsigned int n_tranverse = (unsigned int) pow(3, P4EST_DIM - 1) - 1;
#ifdef P4_TO_P8
  const unsigned char first_trans_dir   = (dir == dir::y || dir == dir :: z ? dir::x : dir::y); P4EST_ASSERT(first_trans_dir != dir);
  const unsigned char second_trans_dir  = (dir == dir::x || dir == dir :: y ? dir::z : dir::y); P4EST_ASSERT(second_trans_dir != dir);
#else
  const unsigned char first_trans_dir   = (dir == dir::y ? dir::x : dir::y); P4EST_ASSERT(first_trans_dir != dir);
#endif
  unsigned char face_dir_to_set_idx[P4EST_FACES] = {DIM(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX), DIM(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX)};
  set_of_neighboring_quadrants ngbd_m_[n_tranverse]; // neighbors of qm in any transverse direction
  set_of_neighboring_quadrants ngbd_p_[n_tranverse]; // neighbors of qp in any transverse direction
  P4EST_ASSERT(ORD(dir == dir::x, dir == dir::y, dir == dir::z));
  unsigned char ngbd_idx = 0;
  char search[P4EST_DIM] = {DIM(0, 0, 0)};
  for (char tt = -1; tt < 2; ++tt)
#ifdef P4_TO_P8
    for (char vv = -1; vv < 2; ++vv)
#endif
    {
      if(tt == 0 ONLY3D(&& vv == 0))
        continue;
      search[first_trans_dir]   = tt;
#ifdef P4_TO_P8
      search[second_trans_dir]  = vv;
#endif
      if(qm.p.piggy3.local_num != -1)
        ngbd_c->find_neighbor_cells_of_cell(ngbd_m_[ngbd_idx], qm.p.piggy3.local_num, qm.p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
      if(qp.p.piggy3.local_num != -1)
        ngbd_c->find_neighbor_cells_of_cell(ngbd_p_[ngbd_idx], qp.p.piggy3.local_num, qp.p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
#ifdef P4_TO_P8
      if(tt == 0)
        face_dir_to_set_idx[2*second_trans_dir + (vv == 1)] = ngbd_idx;
      else if(vv == 0)
        face_dir_to_set_idx[2*first_trans_dir + (tt == 1)] = ngbd_idx;
#else
      face_dir_to_set_idx[2*first_trans_dir + (tt == 1)] = ngbd_idx;
#endif
      ngbd_idx++;
    }
  P4EST_ASSERT(ngbd_idx == n_tranverse);

  /* check for uniform case and/or wall in the neighborhood, if so build voronoi partition by hand */
  const bool extra_layer_in_trans_dir_may_be_required = extra_layer_in_tranverse_directon_may_be_required(dir, tree_dim);
  const bool need_to_look_over_sharing_quad           = check_past_sharing_quad_is_required(dir, tree_dim);
  bool no_wall = (qp.p.piggy3.local_num != -1 && qm.p.piggy3.local_num != -1);
  // if the face is wall itself, check that the other face across the cell is not subrefined
  bool has_uniform_ngbd = !need_to_look_over_sharing_quad || ((qp.p.piggy3.local_num == -1 || qp.level == max_lvl) && (qm.p.piggy3.local_num == -1 || qm.level == max_lvl));
  has_uniform_ngbd = has_uniform_ngbd && ((qp.p.piggy3.local_num == -1 && faces->q2f(qm.p.piggy3.local_num, 2*dir) != NO_VELOCITY)
                                          || (qm.p.piggy3.local_num == -1 && faces->q2f(qp.p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY)
                                          || (qp.level == qm.level && faces->q2f(qm.p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(qp.p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY));
  ngbd_idx = 0;
  for (char tt = -1; tt < 2 && (no_wall || has_uniform_ngbd); ++tt)
#ifdef P4_TO_P8
    for (char vv = -1; vv < 2 && (no_wall || has_uniform_ngbd); ++vv)
#endif
    {
      if(tt == 0 ONLY3D(&& vv == 0))
        continue;
      if(qp.p.piggy3.local_num != -1)
      {
        const bool local_wall = ngbd_p_[ngbd_idx].size() == 0;
        no_wall             = no_wall && !local_wall;
#ifdef P4_TO_P8
        if(tt == 0 || vv == 0) // cartesian tranverse direction
#endif
          has_uniform_ngbd  = has_uniform_ngbd && (local_wall || (ngbd_p_[ngbd_idx].size() == 1 && ngbd_p_[ngbd_idx].begin()->level == qp.level && faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY));
#ifdef P4_TO_P8
        else
          has_uniform_ngbd  = has_uniform_ngbd && (local_wall || (ngbd_p_[ngbd_idx].size() == 1 && ngbd_p_[ngbd_idx].begin()->level <= qp.level));
#endif
        if(has_uniform_ngbd && extra_layer_in_trans_dir_may_be_required && qp.level < ((splitting_criteria_t*) p4est->user_pointer)->max_lvl) // check that quadrants layering those neighbors are not finer because that would invalidate local uniform cells in case of large aspect ratios
        {
#ifdef P4_TO_P8
          if(tt != 0)
#endif
            has_uniform_ngbd = has_uniform_ngbd && (local_wall || (faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*first_trans_dir + (tt == 1)) != NO_VELOCITY));
#ifdef P4_TO_P8
          if(vv != 0)
            has_uniform_ngbd = has_uniform_ngbd && (local_wall || (faces->q2f(ngbd_p_[ngbd_idx].begin()->p.piggy3.local_num, 2*second_trans_dir + (vv == 1)) != NO_VELOCITY));
#endif
        }
      }
      if(qm.p.piggy3.local_num != -1)
      {
        const bool local_wall = ngbd_m_[ngbd_idx].size() == 0;
        no_wall             = no_wall && !local_wall;
#ifdef P4_TO_P8
        if(tt == 0 || vv == 0) // cartesian tranverse direction
#endif
          has_uniform_ngbd  = has_uniform_ngbd && (local_wall || (ngbd_m_[ngbd_idx].size() == 1 && ngbd_m_[ngbd_idx].begin()->level == qm.level && faces->q2f(ngbd_m_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir) != NO_VELOCITY && faces->q2f(ngbd_m_[ngbd_idx].begin()->p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY));
#ifdef P4_TO_P8
        else
          has_uniform_ngbd  = has_uniform_ngbd && (local_wall || (ngbd_m_[ngbd_idx].size() == 1 && ngbd_m_[ngbd_idx].begin()->level <= qm.level));
#endif
        if(has_uniform_ngbd && extra_layer_in_trans_dir_may_be_required && qm.level < ((splitting_criteria_t*) p4est->user_pointer)->max_lvl) // check that quadrants layering those neighbors are not finer because that would invalidate local uniform cells in case of large aspect ratios
        {
#ifdef P4_TO_P8
          if(tt != 0)
#endif
            has_uniform_ngbd = has_uniform_ngbd && (local_wall || (faces->q2f(ngbd_m_[ngbd_idx].begin()->p.piggy3.local_num, 2*first_trans_dir + (tt == 1)) != NO_VELOCITY));
#ifdef P4_TO_P8
          if(vv != 0)
            has_uniform_ngbd = has_uniform_ngbd && (local_wall || (faces->q2f(ngbd_m_[ngbd_idx].begin()->p.piggy3.local_num, 2*second_trans_dir + (vv == 1)) != NO_VELOCITY));
#endif
        }
      }
      ngbd_idx++;
    }
  P4EST_ASSERT((!no_wall && !has_uniform_ngbd) || ngbd_idx == n_tranverse);
  if(has_uniform_ngbd && no_wall)
  {
    P4EST_ASSERT(qm.level <= ((splitting_criteria_t*) p4est->user_pointer)->max_lvl); // consistency check
#ifdef P4EST_DEBUG
    for (char tt = -1; tt < 2; ++tt)
    {
      // consistency check only in perpendicular tranverse directions (not in diagonals, since they may be bigger cells, there)
      P4EST_ASSERT(faces->q2f(ngbd_m_[face_dir_to_set_idx[2*first_trans_dir + (tt == 1)]].begin()->p.piggy3.local_num, 2*dir + 1) == faces->q2f(ngbd_p_[face_dir_to_set_idx[2*first_trans_dir + (tt == 1)]].begin()->p.piggy3.local_num, 2*dir));
#ifdef P4_TO_P8
      P4EST_ASSERT(faces->q2f(ngbd_m_[face_dir_to_set_idx[2*second_trans_dir + (tt == 1)]].begin()->p.piggy3.local_num, 2*dir + 1) == faces->q2f(ngbd_p_[face_dir_to_set_idx[2*second_trans_dir + (tt == 1)]].begin()->p.piggy3.local_num, 2*dir));
#endif
    }
#endif
    const double cell_ratio = (double) (1 << (((splitting_criteria_t *) p4est->user_pointer)->max_lvl - qm.level));
    // neighbor faces in the direction of the face orientation, first:
    vector<ngbdDIMseed> points(P4EST_FACES);
#ifndef P4_TO_P8
    vector<Point2> partition(P4EST_FACES);
#endif
    unsigned char idx;

    // in direction 2*dir
#ifdef P4_TO_P8
    idx = 2*dir;
#else
    idx = face_order_to_counterclock_cycle_order[2*dir];
#endif
    points[idx].n = faces->q2f(qm.p.piggy3.local_num, 2*dir);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      points[idx].p.xyz(dim) = xyz_face[dim] + (dim == dir ? -dxyz[dir]*cell_ratio: 0.0); // safer than fetching the coordinates (if periodic)
#ifdef P4_TO_P8
    points[idx].s     = (dir == dir::x ? dxyz[1]*dxyz[2] : (dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);
#else
    points[idx].theta = dir*M_PI_2 + M_PI;
    partition[idx].x  = points[idx].p.x +         0.5*dxyz[0]*cell_ratio;
    partition[idx].y  = points[idx].p.y + (dir - 0.5)*dxyz[1]*cell_ratio;
#endif

    // in direction 2*dir + 1
#ifdef P4_TO_P8
    idx = 2*dir + 1;
#else
    idx = face_order_to_counterclock_cycle_order[2*dir + 1];
#endif
    points[idx].n = faces->q2f(qp.p.piggy3.local_num, 2*dir + 1);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      points[idx].p.xyz(dim) = xyz_face[dim] + (dim == dir ? dxyz[dir]*cell_ratio: 0.0); // safer than fetching the coordinates (if periodic)
#ifdef P4_TO_P8
    points[idx].s     = (dir == dir::x ? dxyz[1]*dxyz[2] : (dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);
#else
    points[idx].theta = dir*M_PI_2;
    partition[idx].x  = points[idx].p.x -         0.5*dxyz[0]*cell_ratio;
    partition[idx].y  = points[idx].p.y + (0.5 - dir)*dxyz[1]*cell_ratio;
#endif
    // neighbor faces in the tranverse direction(s), then:
    for (unsigned char tran_dir = 0; tran_dir < P4EST_DIM; ++tran_dir) {
      if(tran_dir == dir)
        continue;
      for (int tran_face_dir = 2*tran_dir; tran_face_dir < 2*tran_dir + 2; ++tran_face_dir) { // negative and positive face direction in tranverse Cartesian direction
#ifdef P4_TO_P8
        idx = tran_face_dir;
#else
        idx = face_order_to_counterclock_cycle_order[tran_face_dir];
#endif
        points[idx].n = faces->q2f(ngbd_p_[face_dir_to_set_idx[tran_face_dir]].begin()->p.piggy3.local_num, 2*dir); // we fetch the appropriate face through ngbd_p_
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          points[idx].p.xyz(dim) = xyz_face[dim] + (dim == tran_face_dir/2 ? (tran_face_dir%2 ? +1.0 : -1.0)*dxyz[tran_face_dir/2]*cell_ratio: 0.0); // safer than fetching the coordinates (if periodic)
#ifdef P4_TO_P8
        points[idx].s     = (tran_dir == dir::x ? dxyz[1]*dxyz[2] : (tran_dir == dir::y ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]))*SQR(cell_ratio);
#else
        points[idx].theta = (tran_face_dir/2)*M_PI_2 + (1.0 - tran_face_dir%2)*M_PI;
        partition[idx].x  = points[idx].p.x + (0.5 - tran_face_dir%2)*dxyz[0]*cell_ratio;
        partition[idx].y  = points[idx].p.y + (1.0 - 2.0*(tran_face_dir/2))*(tran_face_dir%2 - 0.5)*dxyz[1]*cell_ratio;
#endif
      }
    }
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
   * 4) the grid is locally uniform but very stretched and more neighbors are actually required!
   * --> need to compute the voronoi cell */
  else
  {
    set_of_neighboring_quadrants ngbd; ngbd.clear();
    set_of_neighboring_quadrants tmp_ngbd;
    std::set<indexed_and_located_face> set_of_neighbor_faces; set_of_neighbor_faces.clear();
    // we add faces to the set as we find them and clear the list of neighbor quadrants after every search to avoid vectors growing very large and slowing down the O(n) searches implemented in find_neighbor_cells_of_cell
    for (char face_touch = -1; face_touch < 2; face_touch += 2) {
      const p4est_quadrant_t& quad_touch = (face_touch == -1 ? qm : qp);
      if(quad_touch.p.piggy3.local_num != -1)
      {
        const p4est_quadrant_t& quad_touch    = (face_touch == -1 ? qm : qp);
        set_of_neighboring_quadrants* ngbd_touch_ = (face_touch == -1 ? ngbd_m_ : ngbd_p_);
        const unsigned char dir_touch = 2*dir + (face_touch == -1 ? 0 : 1);
        ngbd.insert(quad_touch);

        // in face normal direction if needed
        if (faces->q2f(quad_touch.p.piggy3.local_num, dir_touch) == NO_VELOCITY || need_to_look_over_sharing_quad)
        {
          tmp_ngbd.clear();
          tmp_ngbd.insert(quad_touch);
          if(need_to_look_over_sharing_quad)
            ngbd_c->find_neighbor_cells_of_cell(tmp_ngbd, quad_touch.p.piggy3.local_num, quad_touch.p.piggy3.which_tree, dir_touch);
          for (set_of_neighboring_quadrants::const_iterator it = tmp_ngbd.begin(); it != tmp_ngbd.end(); ++it)
          {
            ngbd.insert(*it);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, dir_touch);
          }
        }

        // fetch (all) the extra cells layering quad_touch in the face_touch direction
        // if the aspect ratio is weird, we also layer in the tranverse directions
        ngbd_idx = 0;
        for (char tt = -1; tt < 2; ++tt)
#ifdef P4_TO_P8
          for (char vv = -1; vv < 2; ++vv)
#endif
          {
            if(tt == 0 ONLY3D(&& vv == 0))
              continue;
            search[dir] = face_touch;
            search[first_trans_dir]   = tt;
#ifdef P4_TO_P8
            search[second_trans_dir]  = vv;
#endif
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_touch.p.piggy3.local_num, quad_touch.p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
            if(extra_layer_in_trans_dir_may_be_required)
            {
              search[dir] = 0;
#ifdef P4_TO_P8
              if(tt != 0)
#endif
              {
                search[first_trans_dir]   = tt;
#ifdef P4_TO_P8
                search[second_trans_dir]  = 0;
#endif
                for (set_of_neighboring_quadrants::const_iterator it = ngbd_touch_[ngbd_idx].begin(); it != ngbd_touch_[ngbd_idx].end(); ++it)
                {
                  tmp_ngbd.clear();
                  ngbd_c->find_neighbor_cells_of_cell(tmp_ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  for (set_of_neighboring_quadrants::const_iterator itt = tmp_ngbd.begin(); itt != tmp_ngbd.end(); ++itt) {
                    ngbd.insert(*itt);
                    if(faces->q2f(itt->p.piggy3.local_num, 2*first_trans_dir + (tt == 1)) == NO_VELOCITY)
                      ngbd_c->find_neighbor_cells_of_cell(ngbd, itt->p.piggy3.local_num, itt->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  }
                }
              }
#ifdef P4_TO_P8
              if(vv != 0)
              {
                search[first_trans_dir]   = 0;
                search[second_trans_dir]  = vv;
                for (set_of_neighboring_quadrants::const_iterator it = ngbd_touch_[ngbd_idx].begin(); it != ngbd_touch_[ngbd_idx].end(); ++it)
                {
                  tmp_ngbd.clear();
                  ngbd_c->find_neighbor_cells_of_cell(tmp_ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  for (set_of_neighboring_quadrants::const_iterator itt = tmp_ngbd.begin(); itt != tmp_ngbd.end(); ++itt) {
                    ngbd.insert(*itt);
                    if(faces->q2f(itt->p.piggy3.local_num, 2*second_trans_dir + (vv == 1)) == NO_VELOCITY)
                      ngbd_c->find_neighbor_cells_of_cell(ngbd, itt->p.piggy3.local_num, itt->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  }
                }
              }
              if(tt!= 0 && vv != 0)
              {
                search[first_trans_dir]   = tt;
                search[second_trans_dir]  = vv;
                for (set_of_neighboring_quadrants::const_iterator it = ngbd_touch_[ngbd_idx].begin(); it != ngbd_touch_[ngbd_idx].end(); ++it)
                {
                  tmp_ngbd.clear();
                  ngbd_c->find_neighbor_cells_of_cell(tmp_ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  for (set_of_neighboring_quadrants::const_iterator itt = tmp_ngbd.begin(); itt != tmp_ngbd.end(); ++itt) {
                    ngbd.insert(*itt);
                    if(faces->q2f(itt->p.piggy3.local_num, 2*first_trans_dir + (tt == 1)) == NO_VELOCITY || faces->q2f(itt->p.piggy3.local_num, 2*second_trans_dir + (vv == 1)) == NO_VELOCITY)
                      ngbd_c->find_neighbor_cells_of_cell(ngbd, itt->p.piggy3.local_num, itt->p.piggy3.which_tree, DIM(search[0], search[1], search[2]));
                  }
                }
              }
#endif
            }
            add_faces_to_set_and_clear_set_of_quad(faces, f_idx, dir, set_of_neighbor_faces, ngbd_touch_[ngbd_idx]);
            ngbd_idx++;
          }
        P4EST_ASSERT(ngbd_idx == n_tranverse);
        add_faces_to_set_and_clear_set_of_quad(faces, f_idx, dir, set_of_neighbor_faces, ngbd);
      }
    }

    const bool* periodic  = faces->get_periodicity();
    const double* xyz_min = faces->get_xyz_min();
    const double* xyz_max = faces->get_xyz_max();
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


    /*
     * in case of very stretched grids, problems might occur : a parallel wall may clip the cell --> needs to be added if not Neumann (Dirichlet) as it was not assumed not to happen here above
     * // Example:
     *
     * wall-wall-wall-wall-wall-wall-wall-
     * \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
     * |                |                |
     * |                |                |
     * |                |                |
     * |________________|___(x)__________|
     * |                |        |       |
     * |                |________|_______|
     * |                |        |       |
     * |________________|________|_______|
     *
     * --> one can show that the cell assoiated with (x) here above needs to be clipped by the above wall if dy/dx < sqrt(3.0)/4.0
     * [Raphael: attempt to use very stretched grids for SHS simulations]
     * */
#ifdef P4_TO_P8
    const bool might_need_more_care = !periodic[dir] && (dir == dir::x ? MIN(dxyz[0]/dxyz[1], dxyz[0]/dxyz[2]) < sqrt(3.0)/4.0 : (dir == dir::y ? MIN(dxyz[1]/dxyz[0], dxyz[1]/dxyz[2]) < sqrt(3.0)/4.0 : MIN(dxyz[2]/dxyz[0], dxyz[2]/dxyz[1]) < sqrt(3.0)/4.0));
#else
    const bool might_need_more_care = !periodic[dir] && (dir == dir::x ? dxyz[0]/dxyz[1] < sqrt(3.0)/4.0 : dxyz[1]/dxyz[0] < sqrt(3.0)/4.0);
#endif
    if(might_need_more_care)
    {
      const vector<ngbdDIMseed> *neighbor_seeds;
#ifdef P4_TO_P8
      char parallel_wall_m = (dir == dir::x ? WALL_m00 : (dir == dir::y ? WALL_0m0 : WALL_00m));
      char parallel_wall_p = (dir == dir::x ? WALL_p00 : (dir == dir::y ? WALL_0p0 : WALL_00p));
#else
      char parallel_wall_m = (dir == dir::x ? WALL_m00 : WALL_0m0);
      char parallel_wall_p = (dir == dir::x ? WALL_p00 : WALL_0p0);
#endif
      voronoi_cell.get_neighbor_seeds(neighbor_seeds);
      bool wall_added_manually = false;
      for (size_t m = 0; m < neighbor_seeds->size(); ++m) {
        if((*neighbor_seeds)[m].n == parallel_wall_m || (*neighbor_seeds)[m].n == parallel_wall_p)
        {
          try {
            double xyz_projected_point[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
            xyz_projected_point[dir] = ((*neighbor_seeds)[m].n == parallel_wall_m ? xyz_min[dir] : xyz_max[dir]);
            BoundaryConditionType bc_type_on_pojected_point = bc[dir].wallType(xyz_projected_point);
            if(bc_type_on_pojected_point == DIRICHLET) // if it was NEUMANN
              voronoi_cell.push(WALL_PARALLEL_TO_FACE, DIM(xyz_projected_point[0], xyz_projected_point[1], xyz_projected_point[2]), periodic, xyz_min, xyz_max);
          } catch (std::exception e) {
            throw std::runtime_error("my_p4est_faces_t::compute_voronoi_cell: the boundary condition type needs to be readable from everywhere in the domain when using such stretched grids and non-periodic wall conditions, sorry...");
          }
          wall_added_manually = true;
        }
      }
      if(wall_added_manually)
      {
#ifdef P4_TO_P8
        voronoi_cell.construct_partition(xyz_min, xyz_max, periodic);
#else
        voronoi_cell.construct_partition();
        voronoi_cell.compute_volume();
#endif
      }
    }

    ierr = PetscLogEventEnd(log_my_p4est_faces_compute_voronoi_cell_t, 0, 0, 0, 0); CHKERRXX(ierr);
    if(has_uniform_ngbd)
      return parallelepiped_with_wall; // it for sure is a parallelepiped since it had a uniform neighborhood, but the face must have some wall neighbor(s), that's all
    else
      return nonuniform;
  }
}
