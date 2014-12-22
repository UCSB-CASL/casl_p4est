#ifdef P4_TO_P8
#include "my_p8est_interpolating_function_nonblocking.h"
#include <src/my_p8est_vtk.h>
#else
#include "my_p4est_interpolating_function_nonblocking.h"
#include <src/my_p4est_vtk.h>
#endif

#include "petsc_compatibility.h"
#include <src/my_p4est_log_wrappers.h>
#include <src/ipm_logging.h>

#include <mpi.h>
#include <assert.h>
#include <set>

// Logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegintig
#undef  PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_InterpolatingFunctionNonblocking_interpolate;
extern PetscLogEvent log_InterpolatingFunctionNonblocking_process_local;
extern PetscLogEvent log_InterpolatingFunctionNonblocking_process_queries;
extern PetscLogEvent log_InterpolatingFunctionNonblocking_process_replies;
extern PetscLogEvent log_InterpolatingFunctionNonblocking_barrier;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

InterpolatingFunctionNodeBaseNonblocking::InterpolatingFunctionNodeBaseNonblocking(Vec F, const my_p4est_node_neighbors_t& neighbors, interpolation_method method)
  : neighbors_(neighbors), p4est_(neighbors_.p4est), nodes_(neighbors_.nodes), ghost_(neighbors_.ghost), myb_(neighbors_.myb),
    Fi(F),
    Fxx(NULL), Fyy(NULL),
#ifdef P4_TO_P8
    Fzz(NULL),
#endif
    method_(method),
    senders(p4est_->mpisize, 0)
{
  if (!(method_ == linear || method_ == quadratic || method_ == quadratic_non_oscillatory))
    throw std::invalid_argument("[ERROR]: interpolation method should be one of 'linear', 'quadratic', or 'quadratic_non_oscillatory'. ");

  // create a dedicated communication channel
  MPI_Comm_dup(p4est_->mpicomm, &comm);

  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;
  
  for (short i=0; i<3; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<3; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}

#ifdef P4_TO_P8
InterpolatingFunctionNodeBaseNonblocking::InterpolatingFunctionNodeBaseNonblocking(Vec F, Vec Fxx, Vec Fyy, Vec Fzz, const my_p4est_node_neighbors_t& neighbors, interpolation_method method)
#else
InterpolatingFunctionNodeBaseNonblocking::InterpolatingFunctionNodeBaseNonblocking(Vec F, Vec Fxx, Vec Fyy, const my_p4est_node_neighbors_t& neighbors, interpolation_method method)
#endif
  : neighbors_(neighbors), p4est_(neighbors_.p4est), nodes_(neighbors_.nodes), ghost_(neighbors_.ghost), myb_(neighbors_.myb),
    Fi(F),
    Fxx(Fxx), Fyy(Fyy),
#ifdef P4_TO_P8
    Fzz(Fzz),
#endif
    method_(method),
    senders(p4est_->mpisize, 0)
{
  if (!(method_ == linear || method_ == quadratic || method_ == quadratic_non_oscillatory))
    throw std::invalid_argument("[ERROR]: interpolation method should be one of 'linear', 'quadratic', or 'quadratic_non_oscillatory'. ");

  // create a dedicated communication channel
  MPI_Comm_dup(p4est_->mpicomm, &comm);

  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<3; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<3; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}


InterpolatingFunctionNodeBaseNonblocking::~InterpolatingFunctionNodeBaseNonblocking() {  
  // make sure all messages are finsished sending
  MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE);

  MPI_Comm_free(&comm);
}

void InterpolatingFunctionNodeBaseNonblocking::add_point(p4est_locidx_t node_locidx, const double *xyz)
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
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = xyz_min[i];
  }
  
  p4est_quadrant_t best_match;

  // find the quadrant -- Note point may become slightly purturbed after this call
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = neighbors_.hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  /* check who is going to own the quadrant.
   * we add the point to the local buffer if it is locally owned   
   */
  if (rank_found == p4est_->mpirank) { // local quadrant
    input_buffer[p4est_->mpirank].push_back(node_locidx, xyz);
    local_buffer.push_back(best_match);

  } else {
    if (rank_found != -1) { // ghost quadrant
      input_buffer[rank_found].push_back(node_locidx, xyz);
      senders[rank_found] = 1;
    }

    // add all possible remote ranks
    std::set<int> remote_ranks;

    for (size_t i=0; i<remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);

    for(std::set<int>::const_iterator it = remote_ranks.begin(); it != remote_ranks.end(); ++it){
      int r = *it;
      if (r == rank_found) continue;
      input_buffer[r].push_back(node_locidx, xyz);
      senders[r] = 1;
    }
  }
}

void InterpolatingFunctionNodeBaseNonblocking::interpolate(Vec Fo)
{
  double *Fo_p;
  ierr = VecGetArray(Fo, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(Fo, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseNonblocking::interpolate(double *Fo_p) {

  query_req.clear();
  reply_req.clear();

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_InterpolatingFunctionNonblocking_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  // initiate sending points
  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  const double *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL && Fzz != NULL;
  const double *Fzz_p;
#else
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL;
#endif

  if (use_precomputed_derivatives) {
    ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  int num_remaining_replies = 0, num_remote_points = 0;

  for (std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin();
       it != input_buffer.end(); ++it) {
    if (it->first == p4est_->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    num_remote_points += xyz.size() / P4EST_DIM;

    MPI_Request req;
    MPI_Issend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, query_tag, comm, &req); num_remaining_replies++;
    query_req.push_back(req);  
  }

  // get an instance of logger
  InterpolatingFunctionLogEntry log_entry = {0, 0, 0, 0, 0};
  log_entry.num_local_points = local_buffer.size();
  log_entry.num_send_points  = num_remote_points;
  log_entry.num_send_procs   = num_remaining_replies;

  // Begin main loop
  int done = false;

  size_t it = 0, end = local_buffer.size();
  const input_buffer_t* input = &input_buffer[p4est_->mpirank];
  MPI_Status status;

  int all_queries_processed = false;
  MPI_Request barrier_req;

  double f[P4EST_CHILDREN];
  double fdd[P4EST_CHILDREN*P4EST_DIM]; // fxx[0] = fdd[0], fyy[0] = fdd[1], fzz[0] = fdd[2], fxx[1] = fdd[3], ...

  while (!done) {
    // perform local interpolations
    if (it < end) {
      ierr = PetscLogEventBegin(log_InterpolatingFunctionNonblocking_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
      IPMLogRegionBegin("process_local");

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      const p4est_quadrant_t &quad = local_buffer[it];

      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, quad.p.piggy3.which_tree);
      p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
      for (short j = 0; j<P4EST_CHILDREN; j++) {
        p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];
        f[j] = Fi_p[node_idx];
      }

      // compute derivatives      
      if (method_ == quadratic || method_ == quadratic_non_oscillatory) {
        if (use_precomputed_derivatives) {
          for (short j = 0; j<P4EST_CHILDREN; j++) {
            p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];

            fdd[j*P4EST_DIM + 0] = Fxx_p[node_idx];
            fdd[j*P4EST_DIM + 1] = Fyy_p[node_idx];
#ifdef P4_TO_P8
            fdd[j*P4EST_DIM + 2] = Fzz_p[node_idx];
#endif          
          }
        } else {
          for (short j = 0; j<P4EST_CHILDREN; j++) {
            p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];
            const quad_neighbor_nodes_of_node_t& qnnn = neighbors_[node_idx];

            fdd[j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p);
            fdd[j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p);
#ifdef P4_TO_P8
            fdd[j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p);
#endif          
          }          
        }
      }

      p4est_locidx_t node_idx = input->node_idx[it];
      if (method_ == linear) {
        Fo_p[node_idx] = linear_interpolation(p4est_, quad.p.piggy3.which_tree, quad, f, xyz); it++;        
      } else if (method_ == quadratic) {
        Fo_p[node_idx] = quadratic_interpolation(p4est_, quad.p.piggy3.which_tree, quad, f, fdd, xyz); it++;        
      } else if (method_ == quadratic_non_oscillatory) {
        Fo_p[node_idx] = quadratic_non_oscillatory_interpolation(p4est_, quad.p.piggy3.which_tree, quad, f, fdd, xyz); it++;
      }

      IPMLogRegionEnd("process_local");
      ierr = PetscLogEventEnd(log_InterpolatingFunctionNonblocking_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // probe for incoming queries
    {          
      int is_msg_pending;
      MPI_Iprobe(MPI_ANY_SOURCE, query_tag, comm, &is_msg_pending, &status);
      if (is_msg_pending) { process_incoming_query(status, log_entry); }          
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, comm, &is_msg_pending, &status);
      if (is_msg_pending) { process_incoming_reply(status, Fo_p); num_remaining_replies--; }          
    }      

    if (all_queries_processed) {
        int is_barrier_set;
        MPI_Test(&barrier_req, &is_barrier_set, MPI_STATUS_IGNORE);
        done = is_barrier_set && num_remaining_replies == 0 && it == end;
    } else {
      MPI_Testall(query_req.size(), &query_req[0], &all_queries_processed, MPI_STATUSES_IGNORE);
      if (all_queries_processed)
        MPIX_Ibarrier(comm, &barrier_req);
    }
  }

  InterpolatingFunctionLogger& logger = InterpolatingFunctionLogger::get_instance();
  logger.log(log_entry);

  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  if (use_precomputed_derivatives) {
    ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }
  ierr = PetscLogEventEnd(log_InterpolatingFunctionNonblocking_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseNonblocking::process_incoming_query(MPI_Status& status, InterpolatingFunctionLogEntry& entry) {
  ierr = PetscLogEventBegin(log_InterpolatingFunctionNonblocking_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("process_queries");

  // receive incoming queries about points and send back the interpolated result
  int vec_size;
  MPI_Get_count(&status, MPI_DOUBLE, &vec_size);
  std::vector<double> xyz(vec_size);
  MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, query_tag, comm, MPI_STATUS_IGNORE);

  // log information
  entry.num_recv_points += vec_size / P4EST_DIM;
  entry.num_recv_procs++;

  // search for the quadrants and fill in the data
  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  
  remote_buffer_t remote_data;

  const double *Fi_p;
  PetscErrorCode ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  
  const double *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL && Fzz != NULL;
  const double *Fzz_p;
#else
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL;
#endif

  if (use_precomputed_derivatives) {
    ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  double f  [P4EST_CHILDREN];
  double fdd[P4EST_CHILDREN*P4EST_DIM];
  double xyz_clip[P4EST_DIM];

  std::vector<remote_buffer_t>& buff = send_buffer[status.MPI_SOURCE];

  for (int i = 0; i<vec_size; i += P4EST_DIM) {
    // clip to bounding box
    for (short j = 0; j<P4EST_DIM; j++){
      xyz_clip[j] = xyz[i+j];
      if (xyz[i+j] > xyz_max[j]) xyz_clip[j] = xyz_max[j];
      if (xyz[i+j] < xyz_min[j]) xyz_clip[j] = xyz_min[j];
    }

    int rank_found = neighbors_.hierarchy->find_smallest_quadrant_containing_point(&xyz_clip[0], best_match, remote_matches);

    /* only accept the quadrant if it is in our local part. If the point
     * is in our ghost it means another processor will eventually be able
     * to find it and send it back.
     *
     * Note that the point **CANNOT** be in remote matches as this means
     * the source processor has made a mistake when search for possible
     * remote candidates
     */
    if (rank_found == p4est_->mpirank) {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, best_match.p.piggy3.which_tree);
      p4est_locidx_t quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

      for (short j = 0; j<P4EST_CHILDREN; j++) {
        p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];
        f[j] = Fi_p[node_idx];
      }

      // compute derivatives      
      if (method_ == quadratic || method_ == quadratic_non_oscillatory) {
        if (use_precomputed_derivatives) {
          for (short j = 0; j<P4EST_CHILDREN; j++) {
            p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];

            fdd[j*P4EST_DIM + 0] = Fxx_p[node_idx];
            fdd[j*P4EST_DIM + 1] = Fyy_p[node_idx];
#ifdef P4_TO_P8
            fdd[j*P4EST_DIM + 2] = Fzz_p[node_idx];
#endif          
          }
        } else {
          for (short j = 0; j<P4EST_CHILDREN; j++) {
            p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];
            const quad_neighbor_nodes_of_node_t& qnnn = neighbors_[node_idx];

            fdd[j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p);
            fdd[j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p);
#ifdef P4_TO_P8
            fdd[j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p);
#endif          
          }          
        }
      }
      
      if (method_ == linear) {
        remote_data.value = linear_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, &xyz[i]);
      } else if (method_ == quadratic) {
        remote_data.value = quadratic_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, fdd, &xyz[i]);
      } else if (method_ == quadratic_non_oscillatory) {
        remote_data.value = quadratic_non_oscillatory_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, fdd, &xyz[i]);
      }
      remote_data.input_buffer_idx = i / P4EST_DIM;

      buff.push_back(remote_data);
    } else if (rank_found == -1) {
      /* this cannot happen as it means the source processor made a mistake in
       * calculating its remote matches.
       */
      throw std::runtime_error("[ERROR] A remote processor could not find a local or ghost quadrant "
                               "for a query point sent by the source processor.");
    }
  }

  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  if (use_precomputed_derivatives) {
    ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  // we are done, lets send the buffer back
  MPI_Request req;
  MPI_Isend(&buff[0], buff.size()*sizeof(remote_buffer_t), MPI_BYTE, status.MPI_SOURCE, reply_tag, comm, &req);
  reply_req.push_back(req);

  IPMLogRegionEnd("process_queries");
  ierr = PetscLogEventEnd(log_InterpolatingFunctionNonblocking_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);      
}

void InterpolatingFunctionNodeBaseNonblocking::process_incoming_reply(MPI_Status& status, double *Fo_p) {
  ierr = PetscLogEventBegin(log_InterpolatingFunctionNonblocking_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("process_replies");

  // receive incoming reply we asked before and add it to the result
  int byte_count;
  MPI_Get_count(&status, MPI_BYTE, &byte_count);
  std::vector<remote_buffer_t> reply_buffer (byte_count / sizeof(remote_buffer_t));

  MPI_Recv(&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, reply_tag, comm, MPI_STATUS_IGNORE);

  // put the result in place
  const input_buffer_t& input = input_buffer[status.MPI_SOURCE];

  for (size_t i = 0; i<reply_buffer.size(); i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[i].input_buffer_idx];
    Fo_p[node_idx] = reply_buffer[i].value;
  }

  IPMLogRegionEnd("process_replies");
  ierr = PetscLogEventEnd(log_InterpolatingFunctionNonblocking_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
double InterpolatingFunctionNodeBaseNonblocking::operator ()(double x, double y, double z) const
#else
double InterpolatingFunctionNodeBaseNonblocking::operator ()(double x, double y) const
#endif
{
  PetscErrorCode ierr;
  
  double xyz[] =
  {
    x, y
  #ifdef P4_TO_P8
    , z
  #endif
  };
  
  /* first clip the coordinates */
  double xyz_clip [] =
  {
    x, y
  #ifdef P4_TO_P8
    , z
  #endif
  };
  
  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = xyz_min[i];
  }
  
  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  const double *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL && Fzz != NULL;
  const double *Fzz_p;
#else
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL;
#endif

  if (use_precomputed_derivatives) {
    ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }
  
  double f  [P4EST_CHILDREN];
  double fdd[P4EST_CHILDREN*P4EST_DIM];
  
  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = neighbors_.hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  if (rank_found == p4est_->mpirank) { // local quadrant
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, best_match.p.piggy3.which_tree);
    p4est_locidx_t quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

    for (short i = 0; i<P4EST_CHILDREN; i++) {
      p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + i];
      f[i] = Fi_p[node_idx];
    }

      // compute derivatives
    if (method_ == quadratic || method_ == quadratic_non_oscillatory) {
      if (use_precomputed_derivatives) {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];

          fdd[j*P4EST_DIM + 0] = Fxx_p[node_idx];
          fdd[j*P4EST_DIM + 1] = Fyy_p[node_idx];
#ifdef P4_TO_P8
          fdd[j*P4EST_DIM + 2] = Fzz_p[node_idx];
#endif          
        }
      } else {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j];
          const quad_neighbor_nodes_of_node_t& qnnn = neighbors_[node_idx];

          fdd[j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p);
          fdd[j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p);
#ifdef P4_TO_P8
          fdd[j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p);
#endif          
        }          
      }
    }
     
    // restore arrays and release remote_maches
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  
    if (use_precomputed_derivatives) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }

    double value;
    if (method_ == linear) {
      value = linear_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, xyz);
    } else if (method_ == quadratic) {
      value = quadratic_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz);
    } else if (method_ == quadratic_non_oscillatory) {
      value = quadratic_non_oscillatory_interpolation(p4est_, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz);
    }

    return value;
    
  } else {
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
    if (use_precomputed_derivatives) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }

    std::ostringstream oss;
    oss << "[ERROR]: Point (" << x << "," << y <<
       #ifdef P4_TO_P8
           "," << z <<
       #endif
           ") is not locally owned by processor. "
           << p4est_->mpirank;
    if (rank_found != -1) {
      oss << "Remote owner's rank = " << rank_found << std::endl;
    } else {
      oss << "Possible remote owners are = ";
      for (size_t i = 0; i<remote_matches.size() - 1; i++) {
        oss << remote_matches[i].p.piggy1.owner_rank << ", ";
      }
      oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    }
    throw std::invalid_argument(oss.str());
  }
}
