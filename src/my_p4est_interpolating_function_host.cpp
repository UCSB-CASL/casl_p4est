#ifdef P4_TO_P8
#include "my_p8est_interpolating_function_host.h"
#include <src/my_p8est_vtk.h>
#else
#include "my_p4est_interpolating_function_host.h"
#include <src/my_p4est_vtk.h>
#endif

#include "petsc_compatibility.h"
#include <src/my_p4est_log_wrappers.h>
#include <src/ipm_logging.h>

#include <mpi.h>
#include <set>

// Logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegin
#undef  PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_InterpolatingFunctionHost_interpolate;
extern PetscLogEvent log_InterpolatingFunctionHost_process_local;
extern PetscLogEvent log_InterpolatingFunctionHost_process_queries;
extern PetscLogEvent log_InterpolatingFunctionHost_process_replies;
extern PetscLogEvent log_InterpolatingFunctionHost_all_reduce;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

InterpolatingFunctionNodeBaseHost::InterpolatingFunctionNodeBaseHost(Vec F, const my_p4est_node_neighbors_t *neighbors)
: neighbors_(neighbors), p4est_(neighbors_->p4est), nodes_(neighbors_->nodes), ghost_(neighbors_->ghost), myb_(neighbors_->myb),
Fi(F),
senders(p4est_->mpisize, 0)
{
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

InterpolatingFunctionNodeBaseHost::~InterpolatingFunctionNodeBaseHost() {  
  // make sure all messages are finsished sending
  MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE);
}

void InterpolatingFunctionNodeBaseHost::add_point(p4est_locidx_t node_locidx, const double *xyz)
{
  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

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
  int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

  input_buffer_t *local_input_buffer = &input_buffer[p4est_->mpirank];

  /* check who is going to own the quadrant.
   * we add the point to the local buffer if it is either owned
   * locally or belongs to the ghost layer.
   */
  if (rank_found == p4est_->mpirank) { // local quadrant
    local_input_buffer->push_back(node_locidx, xyz);
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, best_match.p.piggy3.which_tree);
    best_match.p.piggy3.local_num += tree->quadrants_offset;

    local_buffer.push_back(best_match);
  } else if (rank_found != -1 ) { // ghost quadrant
    local_input_buffer->push_back(node_locidx, xyz);
    best_match.p.piggy3.local_num += p4est_->local_num_quadrants;

    local_buffer.push_back(best_match);
  } else if ( remote_matches.size() != 0) {
    /* quadrant belongs to a remote processor and require explixit
     * communication. For this case we only buffer the input
     */

    // make sure remote ranks are unique
     std::set<int> remote_ranks;

     for (size_t i=0; i<remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);

    for(std::set<int>::const_iterator it = remote_ranks.begin(); it != remote_ranks.end(); ++it){
      int r = *it;
      input_buffer[r].push_back(node_locidx, xyz);
      senders[r] = 1;
    }

  } else {
    throw std::runtime_error("[ERROR_YOU_ARE_DOOMED]: InterpolatingFunction::add_point_to_buffer: no quadrant found ... auto-destruct initialized ....\n");
  }

  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseHost::interpolate(Vec Fo)
{
  double *Fo_p;
  ierr = VecGetArray(Fo, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(Fo, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseHost::interpolate(double *Fo_p) {
  ierr = PetscLogEventBegin(log_InterpolatingFunctionHost_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  IPMLogRegionBegin("all_reduce");
  ierr = PetscLogEventBegin(log_InterpolatingFunctionHost_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);

	// determine how many processors will be communicating with this processor
  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  int num_remaining_replies = 0;
  for (size_t i = 0; i<senders.size(); i++)
    num_remaining_replies += senders[i];
  
	// MPI_Allreduce(MPI_IN_PLACE, &senders[0], p4est_->mpisize, MPI_INT, MPI_SUM, p4est_->mpicomm);
  // num_remaining_msgs += senders[p4est_->mpirank];
  std::vector<int> recvcount(p4est_->mpisize, 1);
  int num_remaining_queries = 0; 
  MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est_->mpicomm);
  
  ierr = PetscLogEventEnd(log_InterpolatingFunctionHost_all_reduce, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionEnd("all_reduce");

  // initiate sending points
  for (std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin();
   it != input_buffer.end(); ++it) {
    if (it->first == p4est_->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    MPI_Request req;
    MPI_Isend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, query_tag, p4est_->mpicomm, &req);
    query_req.push_back(req);
  }

	// Begin main loop
  bool done = false;

  size_t it = 0, end = local_buffer.size();
  const input_buffer_t* input = &input_buffer[p4est_->mpirank];
  MPI_Status status;

  double f[P4EST_CHILDREN];
  while (!done) {
    // interpolate local points
    if (it < end) {
      ierr = PetscLogEventBegin(log_InterpolatingFunctionHost_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
      IPMLogRegionBegin("process_local");

      const double* xyz  = &(input->p_xyz[P4EST_DIM*it]);
      p4est_locidx_t node_idx = input->node_idx[it];
      const p4est_quadrant_t &quad = local_buffer[it];

      p4est_locidx_t quad_idx = quad.p.piggy3.local_num;

      for (short i = 0; i<P4EST_CHILDREN; i++)
        f[i] = Fi_p[nodes_->local_nodes[quad_idx*P4EST_CHILDREN + i]];
      
      Fo_p[node_idx] = linear_interpolation_local(xyz, quad, f);
      ++it;

      IPMLogRegionEnd("process_local");
      ierr = PetscLogEventEnd(log_InterpolatingFunctionHost_process_local, 0, 0, 0, 0); CHKERRXX(ierr);
    }
    
    // probe for incoming queries
    if (num_remaining_queries > 0) {
      ierr = PetscLogEventBegin(log_InterpolatingFunctionHost_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);
      IPMLogRegionBegin("process_queries");    
      
      int is_msg_pending;
      MPI_Iprobe(MPI_ANY_SOURCE, query_tag, p4est_->mpicomm, &is_msg_pending, &status);
      if (is_msg_pending) { process_incoming_query(status); num_remaining_queries--; }
      
      IPMLogRegionEnd("process_queries");
      ierr = PetscLogEventEnd(log_InterpolatingFunctionHost_process_queries, 0, 0, 0, 0); CHKERRXX(ierr);      
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      ierr = PetscLogEventBegin(log_InterpolatingFunctionHost_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
      IPMLogRegionBegin("process_replies");

      int is_msg_pending;
      MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, p4est_->mpicomm, &is_msg_pending, &status);
      if (is_msg_pending) { process_incoming_reply(status, Fo_p); num_remaining_replies--; }
      
      IPMLogRegionEnd("process_replies");
      ierr = PetscLogEventEnd(log_InterpolatingFunctionHost_process_replies, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    done = num_remaining_queries == 0 && num_remaining_replies == 0 && it == end;
  }
  
  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_InterpolatingFunctionHost_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseHost::process_incoming_query(MPI_Status& status) {
  // receive incoming queries about points and send back the interpolated result
  int vec_size;
  MPI_Get_count(&status, MPI_DOUBLE, &vec_size);
  std::vector<double> xyz(vec_size);
  MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, query_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

  // search for the quadrants and fill in the data
  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  
  remote_buffer_t remote_data;

  const double *Fi_p;
  PetscErrorCode ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  double f[P4EST_CHILDREN];
  double xyz_clip[P4EST_DIM];

  std::vector<remote_buffer_t>& buff = send_buffer[status.MPI_SOURCE];

  for (size_t i = 0; i<vec_size; i += P4EST_DIM) {
    // clip to bounding box
    for (short j = 0; j<P4EST_DIM; j++){
      xyz_clip[j] = xyz[i+j];
      if (xyz[i+j] > xyz_max[j]) xyz_clip[j] = xyz_max[j];
      if (xyz[i+j] < xyz_min[j]) xyz_clip[j] = xyz_min[j];
    }

    int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(&xyz_clip[0], best_match, remote_matches);

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
      p4est_locidx_t quad_idx = best_match.p.piggy3.local_num;

      for (short j = 0; j<P4EST_CHILDREN; j++)
        f[j] = Fi_p[nodes_->local_nodes[quad_idx*P4EST_CHILDREN + j]];

      remote_data.value = linear_interpolation_local(&xyz[i], best_match, f);
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

  // we are done, lets send the buffer back
  MPI_Request req;
  MPI_Isend(&buff[0], buff.size()*sizeof(remote_buffer_t), MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est_->mpicomm, &req);
  reply_req.push_back(req);  
}

void InterpolatingFunctionNodeBaseHost::process_incoming_reply(MPI_Status& status, double *Fo_p) {
  // receive incoming reply we asked before and add it to the result
  int byte_count;
  MPI_Get_count(&status, MPI_BYTE, &byte_count);
  std::vector<remote_buffer_t> reply_buffer (byte_count / sizeof(remote_buffer_t));

  MPI_Recv(&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

  // put the result in place
  const input_buffer_t& input = input_buffer[status.MPI_SOURCE];

  for (size_t i = 0; i<reply_buffer.size(); i++) {
    p4est_locidx_t node_idx = input.node_idx[reply_buffer[i].input_buffer_idx];
    Fo_p[node_idx] = reply_buffer[i].value;
  }
}

double InterpolatingFunctionNodeBaseHost::linear_interpolation_local(const double* xyz, const p4est_quadrant_t &quad, const double* f) {
  p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[quad.p.piggy3.which_tree*P4EST_CHILDREN + 0];

  double *tree_xyz  = &(p4est_->connectivity->vertices[3*v_mmm]);

  // transfer to the tree coordinate system
  double x = (xyz[0] - tree_xyz[0]);
  double y = (xyz[1] - tree_xyz[1]);
#ifdef P4_TO_P8
  double z = (xyz[2] - tree_xyz[2]);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double xmin = quad_x_fr_i(&quad);
  double ymin = quad_y_fr_j(&quad);
#ifdef P4_TO_P8
  double zmin = quad_z_fr_k(&quad);
#endif

  double d_m00 = x - xmin;
  double d_p00 = qh - d_m00;
  double d_0m0 = y - ymin;
  double d_0p0 = qh - d_0m0;
#ifdef P4_TO_P8
  double d_00m = z - zmin;
  double d_00p = qh - d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += f[j]*w_xyz[j];

#ifdef P4_TO_P8
  value /= qh*qh*qh;
#else
  value /= qh*qh;
#endif

  return value;
}

#ifdef P4_TO_P8
double InterpolatingFunctionNodeBaseHost::operator ()(double x, double y, double z) const
#else
double InterpolatingFunctionNodeBaseHost::operator ()(double x, double y) const
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
  
  double *Fi_p;
  ierr = VecGetArray(Fi, &Fi_p); CHKERRXX(ierr);
  
  static double f  [P4EST_CHILDREN];
  
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t best_match;
  
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  if (rank_found == p4est_->mpirank) { // local quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;
    
    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];
    
    // restore arrays and release remote_maches
    ierr = VecRestoreArray(Fi, &Fi_p); CHKERRXX(ierr);
    return linear_interpolation(p4est_, tree_idx, best_match, f, xyz);
    
  } else if ( rank_found != -1 ) { // ghost quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    quad_idx = best_match.p.piggy3.local_num + p4est_->local_num_quadrants;
    
    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];
    
    // get access to second derivatives only if needed

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(Fi, &Fi_p); CHKERRXX(ierr);
    return linear_interpolation(p4est_, tree_idx, best_match, f, xyz);

  } else {
    ierr = VecRestoreArray(Fi, &Fi_p); CHKERRXX(ierr);

    std::ostringstream oss;
    oss << "[ERROR]: Point (" << x << "," << y <<
       #ifdef P4_TO_P8
     "," << z <<
       #endif
     ") does not belong to "
"processor " << p4est_->mpirank << ". Found rank = " << rank_found <<
       #ifdef P4EST_POINT_LOOKUP
" and remote_macthes.size = " << remote_matches->elem_count << std::endl;
#else
" and remote_macthes.size = " << remote_matches.size() << std::endl;
#endif
throw std::invalid_argument(oss.str());
}
}
