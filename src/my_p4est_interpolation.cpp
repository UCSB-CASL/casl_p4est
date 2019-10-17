#ifdef P4_TO_P8
#include "my_p8est_interpolation.h"
#include <src/my_p8est_vtk.h>
#else
#include "my_p4est_interpolation.h"
#include <src/my_p4est_vtk.h>
#endif

#include <src/casl_math.h>

#include "petsc_compatibility.h"
#include <src/my_p4est_log_wrappers.h>
#include <src/ipm_logging.h>

#include <vector>
#include <mpi.h>
#include <assert.h>
#include <set>

// Logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegin
#undef  PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_interpolation_interpolate;
extern PetscLogEvent log_my_p4est_interpolation_process_local;
extern PetscLogEvent log_my_p4est_interpolation_process_queries;
extern PetscLogEvent log_my_p4est_interpolation_process_replies;
extern PetscLogEvent log_my_p4est_interpolation_all_reduce;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

const unsigned int my_p4est_interpolation_t::ALL_COMPONENTS; // definition of the static const unsigned int is required here

my_p4est_interpolation_t::my_p4est_interpolation_t(const my_p4est_node_neighbors_t* ngbd_n)
  : ngbd_n(ngbd_n), p4est(ngbd_n->p4est), ghost(ngbd_n->ghost), myb(ngbd_n->myb),
    Fi(vector<Vec>(1, NULL)), senders(p4est->mpisize, 0)
{
  // compute domain sizes
  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<P4EST_DIM; i++)
  {
    xyz_min[i]  = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
    xyz_max[i]  = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
    periodic[i] = is_periodic(p4est, i);
  }
  bs_f  = 0;
}

my_p4est_interpolation_t::~my_p4est_interpolation_t() {
  // make sure all messages are finsished sending
  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
}

void my_p4est_interpolation_t::clear()
{
  // make sure all messages are finsished sending
  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  /* reinitialize bs_f */
  bs_f = 0;
  /* clear the buffers */
  input_buffer.clear();
  local_buffer.clear();
  send_buffer.clear();
  query_req.clear();
  reply_req.clear();
  senders.clear(); senders.resize(p4est->mpisize, 0);
}

void my_p4est_interpolation_t::set_input(Vec *F, unsigned int n_vecs_, const unsigned int &block_size_f) {
  // before setting a new input, we need to make sure that any previous interpolation has fully completed,
  // otherwise the programe may crash
  if(query_req.size() > 0){
    mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret); }
  if(reply_req.size() > 0){
    mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret); }
  query_req.clear();
  reply_req.clear();
  send_buffer.clear(); // this is important in case of input reset when reusing the same node-interpolator at the same points, otherwise the buffer just keeps accumulating and communications get bigger and bigger
  // reset the input(s)
  P4EST_ASSERT(n_vecs_ > 0);
  P4EST_ASSERT(block_size_f>0);
  Fi.resize(n_vecs_);
  for (unsigned int k = 0; k < n_vecs_; ++k)
    Fi[k] = F[k];
  bs_f = block_size_f;
}

void my_p4est_interpolation_t::add_point(p4est_locidx_t locidx, const double *xyz)
{
  // first clip the coordinates
  double xyz_clip [P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz_clip[dir] = xyz[dir];
  clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);
  
  p4est_quadrant_t best_match;

  // find the quadrant -- Note point may become slightly purturbed after this call
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

  
  /* check who is going to own the quadrant.
   * we add the point to the local buffer if it is locally owned
   */
  if (rank_found == p4est->mpirank) { // local quadrant
    input_buffer[p4est->mpirank].push_back(locidx, xyz);
    local_buffer.push_back(best_match);
  } else {
    if (rank_found != -1) { // ghost quadrant
      input_buffer[rank_found].push_back(locidx, xyz);
      senders[rank_found] = 1;
    }

    // add all possible remote ranks
    std::set<int> remote_ranks;

    for (size_t i=0; i<remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);

    for(std::set<int>::const_iterator it = remote_ranks.begin(); it != remote_ranks.end(); ++it){
      int r = *it;
      if (r == rank_found) continue;
      input_buffer[r].push_back(locidx, xyz);
      senders[r] = 1;
    }
  }
}

void my_p4est_interpolation_t::interpolate(double * const *Fo_p, unsigned int n_functions, const unsigned int &comp) {
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(n_functions == n_vecs());
  P4EST_ASSERT(bs_f > 0);
  P4EST_ASSERT((comp==ALL_COMPONENTS) || (comp < bs_f));

  IPMLogRegionBegin("all_reduce");
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);

  /* determine how many processors will be communicating with this processor */
  int num_remaining_replies = 0, num_remote_points = 0;
  for (size_t i = 0; i<senders.size(); i++)
    num_remaining_replies += senders[i];
  
  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionEnd("all_reduce");

  // initiate sending points
  for (std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin();
       it != input_buffer.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    num_remote_points += xyz.size() / P4EST_DIM;

    MPI_Request req;
    mpiret = MPI_Isend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, query_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    query_req.push_back(req);
  }
  
  InterpolatingFunctionLogEntry log_entry = {0, 0, 0, 0, 0};
  log_entry.num_local_points = local_buffer.size();
  log_entry.num_send_points  = num_remote_points;
  log_entry.num_send_procs   = num_remaining_replies;
  
  // Begin main loop
  bool done = false;

  size_t it = 0, end = local_buffer.size();
  const input_buffer_t* input = &input_buffer[p4est->mpirank];
  MPI_Status status;

  const unsigned int nelements = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*n_functions : n_functions ;
  double results[nelements]; // serialized_results

  while (!done) {
    // interpolate local points
    if (it < end) {
//      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
//      IPMLogRegionBegin("process_local");

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      const p4est_quadrant_t &quad = local_buffer[it];

      p4est_locidx_t node_idx = input->node_idx[it];
      interpolate(quad, xyz, results, comp);
      //de-serialize
      if((comp==ALL_COMPONENTS) && (bs_f > 1))
        for (unsigned int k = 0; k < n_functions; ++k)
          for (unsigned int cc = 0; cc < bs_f; ++cc)
            Fo_p[k][bs_f*node_idx+cc] = results[bs_f*k+cc];
      else
        for (unsigned int k = 0; k < n_functions; ++k)
          Fo_p[k][node_idx] = results[k];

      it++;

//      IPMLogRegionEnd("process_local");
//      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }
    
    // probe for incoming queries
    if (num_remaining_queries > 0) {
//      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
//      IPMLogRegionBegin("process_queries");
      
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) { process_incoming_query(status, log_entry, comp); num_remaining_queries--; }
      
//      IPMLogRegionEnd("process_queries");
//      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
//      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
//      IPMLogRegionBegin("process_replies");

      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) { process_incoming_reply(status, Fo_p, comp); num_remaining_replies--; }
      
//      IPMLogRegionEnd("process_replies");
//      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    done = num_remaining_queries == 0 && num_remaining_replies == 0 && it == end;
  }

  //  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  //  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  //  query_req.clear();
  //  reply_req.clear();
  
  //  InterpolatingFunctionLogger& logger = InterpolatingFunctionLogger::get_instance();
  //  logger.log(log_entry);

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_interpolation_t::process_incoming_query(MPI_Status& status, InterpolatingFunctionLogEntry& entry, const unsigned int &comp)
{
  // receive incoming queries about points and send back the interpolated result
  int vec_size;
  mpiret = MPI_Get_count(&status, MPI_DOUBLE, &vec_size); SC_CHECK_MPI(mpiret);
  P4EST_ASSERT((vec_size%P4EST_DIM)==0);
  std::vector<double> xyz(vec_size);
  
  // log information
  entry.num_recv_points += vec_size / P4EST_DIM;
  entry.num_recv_procs++;

  mpiret = MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, query_tag, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);
  
  remote_buffer_t remote_data;

  double xyz_clip[P4EST_DIM];
  double xyz_tmp[P4EST_DIM];

  std::vector<remote_buffer_t>& buff = send_buffer[status.MPI_SOURCE];
  P4EST_ASSERT(buff.size() == 0);
  const unsigned int nfunctions = n_vecs();

  for (int i = 0; i<vec_size; i += P4EST_DIM) {
    // clip to bounding box
    for (unsigned char dir = 0; dir<P4EST_DIM; dir++){
      xyz_tmp[dir] = xyz[i+dir];
      xyz_clip[dir] = xyz[i+dir];
    }
    clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);

    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(&xyz_clip[0], best_match, remote_matches);

    /* only accept the quadrant if it is in our local part. If the point
     * is in our ghost it means another processor will eventually be able
     * to find it and send it back.
     *
     * Note that the point **CANNOT** be in remote matches as this means
     * the source processor has made a mistake when search for possible
     * remote candidates
     */
    if (rank_found == p4est->mpirank)
    {
      const unsigned int nelements = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*nfunctions : nfunctions ;
      double results[nelements]; // serialized_results
      interpolate(best_match, xyz_tmp, results, comp);
      remote_data.input_buffer_idx = i / P4EST_DIM;
      if((comp==ALL_COMPONENTS) && (bs_f > 1))
        for (unsigned int k = 0; k < nfunctions; ++k)
          for (unsigned int cc = 0; cc < bs_f; ++cc) {
            remote_data.value = results[bs_f*k+cc];
            buff.push_back(remote_data);
          }
      else
        for (unsigned int k = 0; k < nfunctions; ++k) {
          remote_data.value = results[k];
          buff.push_back(remote_data);
        }
    } else if (rank_found == -1) {
      /* this cannot happen as it means the source processor made a mistake in
       * calculating its remote matches.
       */
      throw std::runtime_error("[ERROR] A remote processor could not find a local or ghost quadrant "
                               "for a query point sent by the source processor.");
    }
  }

  // we are done, lets send the buffer back
  MPI_Request req;
  mpiret = MPI_Isend(buff.data(), buff.size()*sizeof(remote_buffer_t), MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
  reply_req.push_back(req);
}

void my_p4est_interpolation_t::process_incoming_reply(MPI_Status& status, double * const* Fo_p, const unsigned int &comp)
{
  // receive incoming reply we asked before and add it to the result
  int byte_count;
  mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
  std::vector<remote_buffer_t> reply_buffer (byte_count / sizeof(remote_buffer_t));

  mpiret = MPI_Recv(&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);

  // put the result in place
  const input_buffer_t& input = input_buffer[status.MPI_SOURCE];

  unsigned int nfunctions = n_vecs();
  const unsigned int nelements_per_point = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*nfunctions : nfunctions ;
  P4EST_ASSERT(reply_buffer.size()%nelements_per_point == 0);

  for (size_t i = 0; i<reply_buffer.size()/nelements_per_point; i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[nelements_per_point*i].input_buffer_idx];
    if((comp==ALL_COMPONENTS) && (bs_f > 1))
      for (unsigned int k = 0; k < nfunctions; ++k){
        for (unsigned int cc = 0; cc < bs_f; ++cc) {
          P4EST_ASSERT(node_idx == input.node_idx[reply_buffer[nfunctions*bs_f*i+k*bs_f+cc].input_buffer_idx]);
          Fo_p[k][bs_f*node_idx+cc] = reply_buffer[nfunctions*bs_f*i+k*bs_f+cc].value;
        }
      }
    else
      for (unsigned int k = 0; k < nfunctions; ++k){
        P4EST_ASSERT(node_idx == input.node_idx[reply_buffer[nfunctions*i+k].input_buffer_idx]);
        Fo_p[k][node_idx] = reply_buffer[nfunctions*i+k].value;
      }
  }
}

void my_p4est_interpolation_t::add_point_local(p4est_locidx_t locidx, const double *xyz)
{
  // first clip the coordinates
  double xyz_clip [] =
  {
    xyz[0], xyz[1]
  #ifdef P4_TO_P8
    , xyz[2]
  #endif
  };

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = is_periodic(p4est,i) ? xyz_clip[i]-(xyz_max[i]-xyz_min[i]) : xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = is_periodic(p4est,i) ? xyz_clip[i]+(xyz_max[i]-xyz_min[i]) : xyz_min[i];
  }

  p4est_quadrant_t best_match;

  // find the quadrant -- Note point may become slightly purturbed after this call
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

  /* check who is going to own the quadrant.
   * we add the point to the local buffer if it is locally owned
   */
  if (rank_found == p4est->mpirank) { // local quadrant
    input_buffer[p4est->mpirank].push_back(locidx, xyz);
    local_buffer.push_back(best_match);
  }
}


void my_p4est_interpolation_t::interpolate_local(double *Fo_p) {
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  InterpolatingFunctionLogEntry log_entry = {0, 0, 0, 0, 0};
  log_entry.num_local_points = local_buffer.size();

  // Begin main loop
  bool done = false;

  size_t it = 0, end = local_buffer.size();
  const input_buffer_t* input = &input_buffer[p4est->mpirank];

  while (!done) {
    // interpolate local points
    if (it < end)
    {
      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      const p4est_quadrant_t &quad = local_buffer[it];

      p4est_locidx_t node_idx = input->node_idx[it];
      Fo_p[node_idx] = interpolate(quad, xyz);
      it++;
    }

    done = it == end;
  }

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

