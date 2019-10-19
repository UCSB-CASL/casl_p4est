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
extern PetscLogEvent log_my_p4est_interpolation_all_reduce;
extern PetscLogEvent log_my_p4est_interpolation_interpolate;
extern PetscLogEvent log_my_p4est_interpolation_process_local;
extern PetscLogEvent log_my_p4est_interpolation_process_queries;
extern PetscLogEvent log_my_p4est_interpolation_process_replies;
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
  send_buffer.clear(); /* this is important in case of input reset when reusing the same
                          node-interpolator at the same points, otherwise the buffer just
                          keeps accumulating and communications get bigger and bigger */
  // reset the input(s)
  P4EST_ASSERT(n_vecs_ > 0);
  P4EST_ASSERT(block_size_f>0);
  Fi.resize(n_vecs_);
  for (unsigned int k = 0; k < n_vecs_; ++k)
    Fi[k] = F[k];
  bs_f = block_size_f;
}

void my_p4est_interpolation_t::add_point_general(const p4est_locidx_t &node_idx_on_output, const double *xyz, const bool &return_if_non_local)
{
  // first clip the coordinates
  double xyz_clip [P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz_clip[dir] = xyz[dir];
  clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);
  
  p4est_quadrant_t best_match;

  // find the quadrant
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

  
  /* check who is going to own the quadrant.
   * we add the best_match quadrant to the local buffer if it is locally owned
   */
  if (rank_found == p4est->mpirank) { // locally owned
    input_buffer[p4est->mpirank].push_back(node_idx_on_output, xyz);
    local_buffer.push_back(best_match);
  } else {
    if(return_if_non_local)
      return;
    if (rank_found != -1) { // not locally owned, but identified in a ghost quadrant
      input_buffer[rank_found].push_back(node_idx_on_output, xyz);
      senders[rank_found] = 1;
    }

    // add all possible remote ranks
    std::set<int> remote_ranks;

    for (size_t i=0; i<remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);

    for(std::set<int>::const_iterator it = remote_ranks.begin(); it != remote_ranks.end(); ++it){
      int r = *it;
      if (r == rank_found) continue;
      input_buffer[r].push_back(node_idx_on_output, xyz);
      senders[r] = 1;
    }
  }
}

void my_p4est_interpolation_t::interpolate(double * const *Fo_p, const unsigned int &comp) {
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  const unsigned int n_outputs = n_vecs();
  P4EST_ASSERT(bs_f > 0);
  P4EST_ASSERT((comp==ALL_COMPONENTS) || (comp < bs_f));

  ierr = PetscLogEventBegin(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);

  /* determine how many processors will be communicating with this processor */
  int num_remaining_replies = 0, num_remote_points = 0;
  for (size_t i = 0; i<senders.size(); i++)
    num_remaining_replies += senders[i];
  
  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);

  // initiate the queries: send remote points
  for (std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin();
       it != input_buffer.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    num_remote_points += xyz.size() / P4EST_DIM;
    P4EST_ASSERT(xyz.size()/P4EST_DIM>0);

    MPI_Request req;
    mpiret = MPI_Isend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, query_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    query_req.push_back(req);
  }
  

#ifdef CASL_LOG_EVENTS
  InterpolatingFunctionLogEntry log_entry = {0, 0, 0, 0, 0};
  log_entry.num_local_points = local_buffer.size();
  log_entry.num_send_points  = num_remote_points;
  log_entry.num_send_procs   = num_remaining_replies;
#endif
  
  // Begin main loop
  bool done = false;

  size_t it = 0, end = local_buffer.size();
  const input_buffer_t* input = &input_buffer[p4est->mpirank];
  MPI_Status status;

  const unsigned int nelements_per_point = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*n_outputs : n_outputs ;
  double results[nelements_per_point]; // serialized_results, for every locally owned point

  while (!done) {
    // interpolate local points
    if (it < end) {
      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      const p4est_quadrant_t &quad = local_buffer[it];

      p4est_locidx_t node_idx = input->node_idx[it];
      interpolate(quad, xyz, results, comp);
      //de-serialize
      if((comp==ALL_COMPONENTS) && (bs_f > 1))
        for (unsigned int k = 0; k < n_outputs; ++k)
          for (unsigned int cc = 0; cc < bs_f; ++cc)
            Fo_p[k][bs_f*node_idx+cc] = results[bs_f*k+cc];
      else
        for (unsigned int k = 0; k < n_outputs; ++k)
          Fo_p[k][node_idx] = results[k];

      it++;
      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }
    
    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
#ifdef CASL_LOG_EVENTS
        process_incoming_query(status, comp, log_entry);
#else
        process_incoming_query(status, comp);
#endif
        num_remaining_queries--;
      }
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) { process_incoming_reply(status, Fo_p, comp); num_remaining_replies--; }
    }

    done = num_remaining_queries == 0 && num_remaining_replies == 0 && it == end;
  }
  
#ifdef CASL_LOG_EVENTS
  InterpolatingFunctionLogger& logger = InterpolatingFunctionLogger::get_instance();
  logger.log(log_entry);
#endif

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef CASL_LOG_EVENTS
void my_p4est_interpolation_t::process_incoming_query(const MPI_Status& status, const unsigned int &comp, InterpolatingFunctionLogEntry& entry)
#else
void my_p4est_interpolation_t::process_incoming_query(const MPI_Status& status, const unsigned int &comp)
#endif
{
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  // receive incoming queries about points and send back the interpolated result
  int vec_size;
  mpiret = MPI_Get_count(&status, MPI_DOUBLE, &vec_size); SC_CHECK_MPI(mpiret);
  P4EST_ASSERT((vec_size%P4EST_DIM)==0);
  std::vector<double> xyz(vec_size);
  
#ifdef CASL_LOG_EVENTS
  // log information
  entry.num_recv_points += vec_size / P4EST_DIM;
  entry.num_recv_procs++;
#endif

  mpiret = MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, query_tag, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

  double xyz_clip[P4EST_DIM];

  const unsigned int nfunctions = n_vecs();
  const unsigned int nelements_per_point = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*nfunctions : nfunctions ;
  double results[nelements_per_point]; // serialized_results, for every point of interest

  std::vector<data_to_communicate>& buff = send_buffer[status.MPI_SOURCE];
  buff.reserve((vec_size/P4EST_DIM)*(nelements_per_point+1));
  P4EST_ASSERT(buff.size() == 0);

  for (int i = 0; i<vec_size; i += P4EST_DIM) {
    // clip to bounding box
    for (unsigned char dir = 0; dir<P4EST_DIM; dir++)
      xyz_clip[dir] = xyz[i+dir];
    clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);

    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

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
      interpolate(best_match, &xyz[i], results, comp);
      buff.push_back(int(i/P4EST_DIM));
      if((comp==ALL_COMPONENTS) && (bs_f > 1))
        for (unsigned int k = 0; k < nfunctions; ++k)
          for (unsigned int cc = 0; cc < bs_f; ++cc)
            buff.push_back(results[bs_f*k+cc]);
      else
        for (unsigned int k = 0; k < nfunctions; ++k)
          buff.push_back(results[k]);
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
  mpiret = MPI_Isend(buff.data(), buff.size()*sizeof (data_to_communicate), MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
  reply_req.push_back(req);
  // NOTE: it is absolutely possible to have messages of size 0 sent back from here, but we "send" them
  // anyways to comply with the expected number of replies on every process and ensure termination...
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_interpolation_t::process_incoming_reply(const MPI_Status& status, double * const* Fo_p, const unsigned int &comp)
{
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  // receive incoming reply we asked before, deserialize and add it to the result
  int byte_count;
  mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
  P4EST_ASSERT(byte_count%sizeof (data_to_communicate) == 0);
  const unsigned int nfunctions = n_vecs();
  const unsigned int nelements_per_point = ((comp==ALL_COMPONENTS) && (bs_f > 1))? bs_f*nfunctions : nfunctions ;
  std::vector<data_to_communicate> reply_buffer (byte_count/sizeof (data_to_communicate));
  P4EST_ASSERT(byte_count%(((nelements_per_point+1)*sizeof (data_to_communicate))) ==0);

  mpiret = MPI_Recv(&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);

  // put the result in place
  const input_buffer_t& input = input_buffer[status.MPI_SOURCE];
  for (size_t i = 0; i<reply_buffer.size()/(nelements_per_point+1); i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[(nelements_per_point+1)*i].index_in_input_buffer];
    size_t offset_in_reply = (nelements_per_point+1)*i+1;
    if((comp==ALL_COMPONENTS) && (bs_f > 1))
      for (unsigned int k = 0; k < nfunctions; ++k)
        for (unsigned int cc = 0; cc < bs_f; ++cc)
          Fo_p[k][bs_f*node_idx+cc] = reply_buffer[offset_in_reply+k*bs_f+cc].value;
    else
      for (unsigned int k = 0; k < nfunctions; ++k)
        Fo_p[k][node_idx] = reply_buffer[offset_in_reply+k].value;
  }
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_interpolation_t::interpolate_local(double *Fo_p) {
  ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(n_vecs()==1);
  P4EST_ASSERT(bs_f == 1);

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

