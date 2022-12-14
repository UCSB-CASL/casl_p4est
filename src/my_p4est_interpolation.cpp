#ifdef P4_TO_P8
#include "my_p8est_interpolation.h"
#else
#include "my_p4est_interpolation.h"
#endif


// Logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegin
#undef  PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
#include <src/petsc_logging.h>
extern PetscLogEvent log_my_p4est_interpolation_process_queries
extern PetscLogEvent log_my_p4est_interpolation_process_replies
extern PetscLogEvent log_my_p4est_interpolation_all_reduce;
extern PetscLogEvent log_my_p4est_interpolation_interpolate;
extern PetscLogEvent log_my_p4est_interpolation_process_local;
#endif

#include <vector>

const u_int my_p4est_interpolation_t::ALL_COMPONENTS; // definition of the static const u_int is required here

my_p4est_interpolation_t::my_p4est_interpolation_t(const my_p4est_node_neighbors_t* ngbd_n)
  : senders(ngbd_n->get_p4est()->mpisize, 0), ngbd_n(ngbd_n), p4est(ngbd_n->get_p4est()),
    ghost(ngbd_n->get_ghost()), myb(ngbd_n->get_brick()),
    Fi(vector<Vec>(1, NULL)), bs_f(0) { }

my_p4est_interpolation_t::~my_p4est_interpolation_t() {
  // make sure all messages are finsished sending
  complete_and_clear_communications();
}

void my_p4est_interpolation_t::clear()
{
  // make sure all messages are finsished sending
  complete_and_clear_communications();

  /* clear the buffers */
  input_buffer.clear();
  local_buffer.clear();
  send_buffer.clear();
  senders.clear(); senders.resize(p4est->mpisize, 0);
  return;
}

void my_p4est_interpolation_t::set_input_fields(const Vec *F, const size_t &n_vecs_, const u_int &block_size_f)
{
  // before setting a new input, we need to make sure that any previous interpolation has fully completed,
  // otherwise the programe may crash
  complete_and_clear_communications();
  send_buffer.clear(); /* this is important in case of input reset when reusing the same
                          node-interpolator at the same points, otherwise the buffer just
                          keeps accumulating and communications get bigger and bigger */
  // reset the input(s)
  P4EST_ASSERT(n_vecs_ > 0);
  P4EST_ASSERT(block_size_f > 0);
  Fi.resize(n_vecs_);
  for(u_int k = 0; k < n_vecs_; ++k)
    Fi[k] = F[k];
  bs_f = block_size_f;
  return;
}

void my_p4est_interpolation_t::add_point_general(const p4est_locidx_t &node_idx_on_output, const double *xyz, const bool &return_if_non_local)
{
  // first clip the coordinates
  double xyz_clip [P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
  clip_in_domain(xyz_clip, get_xyz_min(), get_xyz_max(), get_periodicity());

  p4est_quadrant_t best_match;

  // find the quadrant
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->get_hierarchy()->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches, false, true);

  /* check who is going to own the quadrant.
   * we add the best_match quadrant to the local buffer if it is locally owned
   */
  if (rank_found == p4est->mpirank) { // locally owned
    input_buffer[p4est->mpirank].push_back(node_idx_on_output, xyz);
    P4EST_ASSERT(best_match.p.piggy3.local_num < p4est->local_num_quadrants);
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

    for(size_t i = 0; i < remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);

    for(std::set<int>::const_iterator it = remote_ranks.begin(); it != remote_ranks.end(); ++it){
      int r = *it;
      if (r == rank_found) continue;
      input_buffer[r].push_back(node_idx_on_output, xyz);
      senders[r] = 1;
    }
  }
  return;
}

#ifdef CASL_LOG_EVENTS
void my_p4est_interpolation_t::process_incoming_query(const MPI_Status &status, const u_int &comp, InterpolatingFunctionLogEntry& entry)
#else
void my_p4est_interpolation_t::process_incoming_query(const MPI_Status &status, const u_int &comp)
#endif
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  // receive incoming queries about points and send back the interpolated result
  const size_t nfunctions = n_vecs();
  const size_t nelements_per_point = (comp == ALL_COMPONENTS && bs_f > 1 ? bs_f*nfunctions : nfunctions);
  std::vector<double> results(nelements_per_point); // serialized_results, for every point of interest
  std::vector<double> xyz(0);
#ifdef CASL_LOG_EVENTS
  receive_queried_coordinates_and_allocate_send_buffer_in_map(xyz, send_buffer, nelements_per_point, status, entry);
#else
  receive_queried_coordinates_and_allocate_send_buffer_in_map(xyz, send_buffer, nelements_per_point, status);
#endif

  std::vector<data_to_communicate<double> > &buff = send_buffer[status.MPI_SOURCE];
  double xyz_clip[P4EST_DIM];

  for(size_t i = 0; i < xyz.size(); i += P4EST_DIM) {
    // clip to bounding box
    for(u_char dim = 0; dim < P4EST_DIM; dim++)
      xyz_clip[dim] = xyz[i + dim];
    clip_in_domain(xyz_clip, get_xyz_min(), get_xyz_max(), get_periodicity()); // do we actually want that? Especially in case of non-periodicity??

    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->get_hierarchy()->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches, false, true);

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
      P4EST_ASSERT(best_match.p.piggy3.local_num < p4est->local_num_quadrants);
      interpolate(best_match, &xyz[i], results.data(), comp);
      buff.push_back(int(i/P4EST_DIM));
      if(comp == ALL_COMPONENTS && bs_f > 1)
        for(size_t k = 0; k < nfunctions; ++k)
          for(u_int cc = 0; cc < bs_f; ++cc)
            buff.push_back(results[bs_f*k + cc]);
      else
        for(size_t k = 0; k < nfunctions; ++k)
          buff.push_back(results[k]);
    } else if (rank_found == -1) {
      /* this cannot happen as it means the source processor made a mistake in
       * calculating its remote matches.
       */
      printf( "Relevant info: rank = %d \n"
                       "rank found = %d \n"
                        "(x, y) = (%0.4f, %0.4f) \n"
                                  "tree index = %d \n"
             "quad index = %d \n",p4est->mpirank ,rank_found, xyz[i], xyz[i+1],
                                  best_match.p.piggy3.which_tree,
                                  best_match.p.piggy3.local_num);

      throw std::runtime_error("[ERROR] process_incoming_query: A remote processor could not find a local or ghost quadrant "
                               "for a query point sent by the source processor. \n");
    }
  }
  // we are done, let's send the buffer back
  send_response_back_to_query(buff, status);
  // NOTE: it is absolutely possible to have messages of size 0 sent back from here, but we "send" them
  // anyways to comply with the expected number of replies on every process and ensure termination...
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

#ifdef CASL_LOG_EVENTS
void my_p4est_interpolation_t::process_incoming_query_interface_bc(const BoundaryConditionsDIM& bc_to_sample, std::map<int, std::vector<data_to_communicate<bc_sample> > > &map_of_send_buffers_for_bc_samples, const MPI_Status &status, InterpolatingFunctionLogEntry& entry)
#else
void my_p4est_interpolation_t::process_incoming_query_interface_bc(const BoundaryConditionsDIM& bc_to_sample, std::map<int, std::vector<data_to_communicate<bc_sample> > > &map_of_send_buffers_for_bc_samples, const MPI_Status &status)
#endif
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  // receive incoming queries about points and send back the sampled interface bc
  std::vector<double> xyz(0);
#ifdef CASL_LOG_EVENTS
  receive_queried_coordinates_and_allocate_send_buffer_in_map(xyz, map_of_send_buffers_for_bc_samples, 1, status, entry);
#else
  receive_queried_coordinates_and_allocate_send_buffer_in_map(xyz, map_of_send_buffers_for_bc_samples, 1, status);
#endif

  std::vector<data_to_communicate<bc_sample> > &buff = map_of_send_buffers_for_bc_samples[status.MPI_SOURCE];
  double xyz_clip[P4EST_DIM];

  for(size_t i = 0; i < xyz.size(); i += P4EST_DIM) {
    // clip to bounding box
    for(u_char dim = 0; dim < P4EST_DIM; dim++)
      xyz_clip[dim] = xyz[i + dim];
    clip_in_domain(xyz_clip, get_xyz_min(), get_xyz_max(), get_periodicity()); // do we actually want that? Especially in case of non-periodicity??

    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->get_hierarchy()->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches, false, true);

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
      P4EST_ASSERT(best_match.p.piggy3.local_num < p4est->local_num_quadrants);
      buff.push_back(int(i/P4EST_DIM));
      bc_sample local_interface_bc;
      local_interface_bc.type = bc_to_sample.interfaceType(xyz_clip);
      if(bc_to_sample.InterfaceValueIsDefined())
        local_interface_bc.value = bc_to_sample.interfaceValue(xyz_clip);
      buff.push_back(local_interface_bc);
    } else if (rank_found == -1) {
      /* this cannot happen as it means the source processor made a mistake in
       * calculating its remote matches.
       */
      PetscPrintf(p4est->mpicomm, "Relevant info: \n"
                                  "rank found = %d \n"
                                  "(x, y) = (%0.4f, %0.4f) \n"
                                  "tree index = %d \n"
                                  "quad index = %d \n", rank_found, xyz[i], xyz[i+1],
                  best_match.p.piggy3.which_tree,
                  best_match.p.piggy3.local_num);
      throw std::runtime_error("[ERROR] process_incoming_query_interface_bc: A remote processor could not find a local or ghost quadrant "
                               "for a query point sent by the source processor.");
    }
  }
  // we are done, let's send the buffer back
  send_response_back_to_query(buff, status);
  // NOTE: it is absolutely possible to have messages of size 0 sent back from here, but we "send" them
  // anyways to comply with the expected number of replies on every process and ensure termination...
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}


void my_p4est_interpolation_t::process_incoming_reply(const MPI_Status& status, double * const *Fo_p, const u_int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  const size_t nfunctions = n_vecs();
  const size_t nelements_per_point = (comp == ALL_COMPONENTS && bs_f > 1 ? bs_f*nfunctions : nfunctions);
  std::vector<data_to_communicate<double> > reply_buffer(0);
  receive_incoming_reply(reply_buffer, nelements_per_point, status);

  // deserialize and put the results in place
  const input_buffer_t& input = input_buffer.at(status.MPI_SOURCE);
  for(size_t i = 0; i < reply_buffer.size()/(nelements_per_point + 1); i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[(nelements_per_point + 1)*i].index_in_input_buffer];
    size_t offset_in_reply = (nelements_per_point + 1)*i + 1;
    if(comp == ALL_COMPONENTS && bs_f > 1)
      for(size_t k = 0; k < nfunctions; ++k)
        for(u_int cc = 0; cc < bs_f; ++cc)
          Fo_p[k][bs_f*node_idx + cc] = reply_buffer[offset_in_reply + k*bs_f + cc].value;
    else
      for(size_t k = 0; k < nfunctions; ++k)
        Fo_p[k][node_idx] = reply_buffer[offset_in_reply + k].value;
  }
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_interpolation_t::process_incoming_reply_interface_bc(const MPI_Status& status, bc_sample* interface_bc) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  const size_t nelements_per_point = 1;
  std::vector<data_to_communicate<bc_sample> > reply_buffer(0);
  receive_incoming_reply(reply_buffer, nelements_per_point, status);
  // deserialize and put the results in place
  const input_buffer_t& input = input_buffer.at(status.MPI_SOURCE);
  for(size_t i = 0; i < reply_buffer.size()/(nelements_per_point + 1); i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[(nelements_per_point + 1)*i].index_in_input_buffer];
    size_t offset_in_reply = (nelements_per_point + 1)*i + 1;
    interface_bc[node_idx] = reply_buffer[offset_in_reply].value;
  }
  ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_interpolation_t::determine_and_initiate_global_communications(int &num_remaining_replies, int &num_remote_points, int&num_remaining_queries)
{
  /* determine how many processors will be communicating with this processor and the communication pattern */
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);
  num_remaining_replies = num_remote_points = num_remaining_queries = 0;

  for(size_t i = 0; i < senders.size(); i++)
    num_remaining_replies += senders[i];

  std::vector<int> recvcount(p4est->mpisize, 1);
  int mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);

  // initiate the queries: send remote points
  for(std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin(); it != input_buffer.end(); ++it)
  {
    if (it->first == p4est->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    num_remote_points += xyz.size() / P4EST_DIM;
    P4EST_ASSERT(xyz.size()/P4EST_DIM > 0);

    MPI_Request req;
    mpiret = MPI_Isend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, query_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    query_req.push_back(req);
  }
  return;
}

void my_p4est_interpolation_t::interpolate(double * const *Fo_p, const u_int &comp, const bool &local_only)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  const size_t n_outputs = n_vecs();
  P4EST_ASSERT(bs_f > 0);
  P4EST_ASSERT(comp == ALL_COMPONENTS || comp < bs_f);


  int num_remaining_replies, num_remote_points, num_remaining_queries;
  if(!local_only)
    determine_and_initiate_global_communications(num_remaining_replies, num_remote_points, num_remaining_queries);
  else
    num_remaining_replies = num_remote_points = num_remaining_queries = 0;


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

  const size_t nelements_per_point = (comp == ALL_COMPONENTS && bs_f > 1 ? bs_f*n_outputs : n_outputs);
  std::vector<double> results(nelements_per_point); // serialized_results, for every locally owned point

  while (!done) {
    // interpolate local points
    if (it < end) {
      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      const p4est_quadrant_t &quad = local_buffer[it];
      P4EST_ASSERT(quad.p.piggy3.local_num < p4est->local_num_quadrants);

      const p4est_locidx_t &node_idx = input->node_idx[it];
      interpolate(quad, xyz, results.data(), comp);
      //de-serialize
      if(comp == ALL_COMPONENTS && bs_f > 1)
        for(size_t k = 0; k < n_outputs; ++k)
          for(u_int cc = 0; cc < bs_f; ++cc)
            Fo_p[k][bs_f*node_idx + cc] = results[bs_f*k + cc];
      else
        for(size_t k = 0; k < n_outputs; ++k)
          Fo_p[k][node_idx] = results[k];

      it++;
      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      int mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
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
      int mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) { process_incoming_reply(status, Fo_p, comp); num_remaining_replies--; }
    }

    done = num_remaining_queries == 0 && num_remaining_replies == 0 && it == end;
  }

#ifdef CASL_LOG_EVENTS
  InterpolatingFunctionLogger& logger = InterpolatingFunctionLogger::get_instance();
  logger.log(log_entry);
#endif

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_interpolation_t::evaluate_interface_bc(const BoundaryConditionsDIM &bc_to_sample, bc_sample *interface_bc)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(bs_f == 0); // the input should not be have been set, ever for using the object this way...

  int num_remaining_replies, num_remote_points, num_remaining_queries;
  determine_and_initiate_global_communications(num_remaining_replies, num_remote_points, num_remaining_queries);

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
  std::map<int, vector<data_to_communicate<bc_sample> > > map_of_send_buffers_for_bc_samples;

  while (!done) {
    // interpolate local points
    if (it < end) {
      ierr = PetscLogEventBegin(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
#ifdef P4EST_DEBUG
      const p4est_quadrant_t &quad = local_buffer[it];
      P4EST_ASSERT(quad.p.piggy3.local_num < p4est->local_num_quadrants); // make sure it's local
#endif

      const p4est_locidx_t &node_idx = input->node_idx[it];
      interface_bc[node_idx].type = bc_to_sample.interfaceType(xyz);
      if(bc_to_sample.InterfaceValueIsDefined())
        interface_bc[node_idx].value = bc_to_sample.interfaceValue(xyz);
      it++;
      ierr = PetscLogEventEnd(log_my_p4est_interpolation_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      int mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
#ifdef CASL_LOG_EVENTS
        process_incoming_query_interface_bc(bc_to_sample, map_of_send_buffers_for_bc_samples, status, log_entry);
#else
        process_incoming_query_interface_bc(bc_to_sample, map_of_send_buffers_for_bc_samples, status);
#endif
        num_remaining_queries--;
      }
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      int mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) { process_incoming_reply_interface_bc(status, interface_bc); num_remaining_replies--; }
    }

    done = num_remaining_queries == 0 && num_remaining_replies == 0 && it == end;
  }

#ifdef CASL_LOG_EVENTS
  InterpolatingFunctionLogger& logger = InterpolatingFunctionLogger::get_instance();
  logger.log(log_entry);
#endif

  ierr = PetscLogEventEnd(log_my_p4est_interpolation_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}
