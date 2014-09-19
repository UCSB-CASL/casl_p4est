#ifdef P4_TO_P8
#include "my_p8est_interpolating_function_balanced.h"
#include <src/my_p8est_vtk.h>
#else
#include "my_p4est_interpolating_function_balanced.h"
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
extern PetscLogEvent log_InterpolatingFunctionBalanced_interpolate;
extern PetscLogEvent log_InterpolatingFunctionBalanced_process_data;
extern PetscLogEvent log_InterpolatingFunctionBalanced_process_message;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

InterpolatingFunctionNodeBaseBalanced::InterpolatingFunctionNodeBaseBalanced(Vec F, const my_p4est_node_neighbors_t *neighbors)
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

InterpolatingFunctionNodeBaseBalanced::~InterpolatingFunctionNodeBaseBalanced() {  
  // make sure all messages are finsished sending
  MPI_Waitall(point_send_req.size(), &point_send_req[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(data_send_req.size(),  &data_send_req[0],  MPI_STATUSES_IGNORE);
}

void InterpolatingFunctionNodeBaseBalanced::add_point(p4est_locidx_t node_locidx, const double *xyz)
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
  static p4est_locidx_t local_input_buffer_idx = 0;

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank) {
    // added point belongs to a local quadrant

    local_input_buffer->push_back(node_locidx, xyz);

    // copy the important properties of quadrant into buffer
    cell_data_t data;
    data.q_xyz[0] = best_match.x;
    data.q_xyz[1] = best_match.y;
#ifdef P4_TO_P8
    data.q_xyz[2] = best_match.z;
#endif
    data.level    = best_match.level;
    data.tree_idx = best_match.p.piggy3.which_tree;
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, data.tree_idx);
    p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;
    for (short i = 0; i<P4EST_CHILDREN; i++)
      data.f[i] = Fi_p[nodes_->local_nodes[qu_locidx*P4EST_CHILDREN + i]];
    data.input_buffer_idx = local_input_buffer_idx++;
    data_buffer.push_back(data);

  } else if ( rank_found != -1 ) {
    /* added point belongs to ghost quadrant. Note, however, that since
     * we can still perform the interpolation locally, these points are
     * also added to the local part of the buffer
     */

    local_input_buffer->push_back(node_locidx, xyz);
    
    // copy the important properties of quadrant into buffer
    cell_data_t data;
    data.q_xyz[0] = best_match.x;
    data.q_xyz[1] = best_match.y;
#ifdef P4_TO_P8
    data.q_xyz[2] = best_match.z;
#endif
    data.level    = best_match.level;
    data.tree_idx = best_match.p.piggy3.which_tree;
    p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + p4est_->local_num_quadrants;
    for (short i = 0; i<P4EST_CHILDREN; i++)
      data.f[i] = Fi_p[nodes_->local_nodes[qu_locidx*P4EST_CHILDREN + i]];
    data.input_buffer_idx = local_input_buffer_idx++;
    data_buffer.push_back(data);

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

void InterpolatingFunctionNodeBaseBalanced::interpolate(Vec Fo)
{
  double *Fo_p;
  ierr = VecGetArray(Fo, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(Fo, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBaseBalanced::interpolate(double *Fo_p) {
  IPMLogRegionBegin("InterpolatingFunctionNodeBaseBalanced::interpolate");

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_InterpolatingFunctionBalanced_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  // initiate sending points
  for (std::map<int, input_buffer_t>::const_iterator it = input_buffer.begin();
       it != input_buffer.end(); ++it) {
    if (it->first == p4est_->mpirank)
      continue;

    const std::vector<double>& xyz = it->second.p_xyz;
    MPI_Request req;
    MPI_Isend((void*)&xyz[0], xyz.size(), MPI_DOUBLE, it->first, point_tag, p4est_->mpicomm, &req);
    point_send_req.push_back(req);
  }

  // determine how many processors will be communicating with this processor
  p4est_locidx_t num_remaining_msgs = 0;
  for (size_t i = 0; i<senders.size(); i++)
    num_remaining_msgs += senders[i];
  MPI_Allreduce(MPI_IN_PLACE, &senders[0], p4est_->mpisize, MPI_INT, MPI_SUM, p4est_->mpicomm);
  num_remaining_msgs += senders[p4est_->mpirank];

  bool done = false;
  std::queue<const input_buffer_t*> queue;

  size_t it = 0;
  const input_buffer_t* input = &input_buffer[p4est_->mpirank];
  MPI_Status status;

  while (!done) {
    if (it != data_buffer.size()) {
      process_data(input, data_buffer[it], Fo_p);
      ++it;
    } else if (!queue.empty()) {
      input = queue.front();
      queue.pop();
    }

    // probe for incoming messages
    int is_msg_pending;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, p4est_->mpicomm, &is_msg_pending, &status);

    // receive pending message based on its status
    if (is_msg_pending) {
      process_message(status, queue);
      num_remaining_msgs--;
    }

    done = num_remaining_msgs == 0 && it == data_buffer.size();
  }
  
  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_InterpolatingFunctionBalanced_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  IPMLogRegionEnd("InterpolatingFunctionNodeBaseBalanced::interpolate");
}

void InterpolatingFunctionNodeBaseBalanced::process_data(const input_buffer_t* input, const cell_data_t &data, double *Fo_p) {
  p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[data.tree_idx*P4EST_CHILDREN + 0];

  double *tree_xyz  = &(p4est_->connectivity->vertices[3*v_mmm]);
  const double *point_xyz = &(input->p_xyz[data.input_buffer_idx]);

  double x = (point_xyz[0] - tree_xyz[0]);
  double y = (point_xyz[1] - tree_xyz[0]);
#ifdef P4_TO_P8
  double z = (point_xyz[2] - tree_xyz[0]);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(data.level) / (double)(P4EST_ROOT_LEN);
  double xmin = (double)data.q_xyz[0] / (double)(P4EST_ROOT_LEN);
  double ymin = (double)data.q_xyz[1] / (double)(P4EST_ROOT_LEN);
#ifdef P4_TO_P8
  double zmin = (double)data.q_xyz[2] / (double)(P4EST_ROOT_LEN);
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
    value += data.f[j]*w_xyz[j];

#ifdef P4_TO_P8
  value /= qh*qh*qh;
#else
  value /= qh*qh;
#endif

  Fo_p[input->node_idx[data.input_buffer_idx]] = value;
}

void InterpolatingFunctionNodeBaseBalanced::process_message(MPI_Status& status, std::queue<const input_buffer_t*>& queue) {
  if (point_tag == status.MPI_TAG) {
    // receive incoming quqery about points and send back the results
    int vec_size;
    MPI_Get_count(&status, MPI_DOUBLE, &vec_size);
    std::vector<double> xyz(vec_size);
    MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);

    // search for the quadrants and fill in the data
    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    cell_data_t data;

    const double *Fi_p;
    PetscErrorCode ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

    std::vector<cell_data_t>& buff = send_buffer[status.MPI_SOURCE];
    for (size_t i = 0; i<vec_size; i += P4EST_DIM) {
      // clip to bounding box
      for (short j = 0; j<P4EST_DIM; j++){
        if (xyz[i+j] > xyz_max[j]) xyz[i+j] = xyz_max[j];
        if (xyz[i+j] < xyz_min[j]) xyz[i+j] = xyz_min[j];
      }

      int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(&xyz[i], best_match, remote_matches);

      /* only accept the quadrant if it is in our local part. If the point
       * is in our ghost it means another processor will eventually be able
       * to find it and send it back.
       *
       * Note that the point **CANNOT** be in remote matches as this means
       * the source processor has made a mistake when search for possible
       * remote candidates
       */
      if (rank_found == p4est_->mpirank) {
        // copy the important properties of quadrant into buffer
        data.q_xyz[0] = best_match.x;
        data.q_xyz[1] = best_match.y;
#ifdef P4_TO_P8
        data.q_xyz[2] = best_match.z;
#endif
        data.level    = best_match.level;
        data.tree_idx = best_match.p.piggy3.which_tree;
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, data.tree_idx);
        p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;
        for (short j = 0; j<P4EST_CHILDREN; j++)
          data.f[j] = Fi_p[nodes_->local_nodes[qu_locidx*P4EST_CHILDREN + j]];
        data.input_buffer_idx = i / P4EST_DIM;

        buff.push_back(data);
      } else if (rank_found == -1) {
        /* this cannot happen as it means the source processor made a mistake in
         * calculating its remote matches.
         */
        throw std::runtime_error("[ERROR] A remote processor could not find a local or ghost quadrant "
                                 "for a query point sent by the source processor.");
      }
    }

    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

    // we are done with searching, lets send the buffer back
    int tag = buff.size() == 0 ? ignore_tag:data_tag;
    MPI_Request req;
    MPI_Isend(&buff[0], buff.size()*sizeof(cell_data_t), MPI_BYTE, status.MPI_SOURCE, tag, p4est_->mpicomm, &req);
    data_send_req.push_back(req);

  } else if (data_tag == status.MPI_TAG) {
    // receive incoming data we asked before and add it to the queue
    int byte_count;
    MPI_Get_count(&status, MPI_BYTE, &byte_count);

    size_t old_size = data_buffer.size();
    size_t new_size = old_size + byte_count / sizeof(cell_data_t);
    data_buffer.resize(new_size);

    MPI_Recv(&data_buffer[old_size], byte_count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);

    // Add the new input to the queue
    queue.push(&input_buffer[status.MPI_SOURCE]);

  } else if (ignore_tag == status.MPI_TAG) {
    // ignore this message since the processor was not able to send any proper data
  } else {
    throw std::runtime_error("[ERROR] Received an MPI message with an unknown tag!");
  }

  // check if we can free data buffers
  int req_idx;
  int is_request_completed;
  MPI_Testany(data_send_req.size(), &data_send_req[0], &req_idx, &is_request_completed, &status);
  if (is_request_completed)
    send_buffer[status.MPI_SOURCE].clear();
}

#ifdef P4_TO_P8
double InterpolatingFunctionNodeBaseBalanced::operator ()(double x, double y, double z) const
#else
double InterpolatingFunctionNodeBaseBalanced::operator ()(double x, double y) const
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
