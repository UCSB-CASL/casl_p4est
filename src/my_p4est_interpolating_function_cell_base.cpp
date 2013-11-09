#ifdef P4_TO_P8
#include "my_p8est_interpolating_function_cell_base.h"
#include <src/my_p8est_vtk.h>
#include <src/point3.h>
#else
#include "my_p4est_interpolating_function_cell_base.h"
#include <src/my_p4est_vtk.h>
#include <src/point2.h>
#endif

#include "petsc_compatibility.h"
#include <sc_notify.h>
#include <mpi.h>
#include <src/CASL_math.h>

#include <fstream>
#include <set>

// Logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegin
#undef  PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_InterpolatingFunction_interpolate;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

#ifdef P4EST_POINT_LOOKUP
#ifdef P4_TO_P8
  #error "p4est point lookup is not available in 3D"
#endif
#endif

InterpolatingFunctionCellBase::InterpolatingFunctionCellBase(const my_p4est_cell_neighbors_t *cnnn)
  : p4est_(cnnn->p4est), ghost_(cnnn->ghost), myb_(cnnn->myb), cnnn_(cnnn),
    method_(IDW),
    remote_senders(p4est_->mpisize, -1),
    is_buffer_prepared(false)
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

InterpolatingFunctionCellBase::~InterpolatingFunctionCellBase()
{}

void InterpolatingFunctionCellBase::add_point_to_buffer(p4est_locidx_t node_locidx, const double *xyz)
{
  /* first clip the coordinates */
  double xyz_clip [] =
  {
    xyz[0], xyz[1]
#ifdef P4_TO_P8
    ,
    xyz[2]
#endif
  };

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = xyz_min[i];
  }

  p4est_quadrant_t best_match;

  // find the quadrant -- Note point may become slightly purturbed after this call
#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = cnnn_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank) { // local quadrant

    for (short i = 0; i<P4EST_DIM; i++)
      local_point_buffer.xyz.push_back(xyz[i]);

    local_point_buffer.quad.push_back(best_match);
    local_point_buffer.output_idx.push_back(node_locidx);

  } else if ( rank_found != -1 ) { /* point is found in the ghost layer */
    /*
     * unlike the case for node interpolation, there is no guarantee that
     * we can perform the interpolation for points that end up here. Here we
     * add the points to the list in the hope that interpolation will succeed.
     * Later, if the interpolation was not successful, we will throw an exception
     * and add the point to the remote list
     */

    for (short i = 0; i<P4EST_DIM; i++)
      ghost_point_buffer.xyz.push_back(xyz[i]);

    ghost_point_buffer.quad.push_back(best_match);
    ghost_point_buffer.rank.push_back(rank_found);
    ghost_point_buffer.output_idx.push_back(node_locidx);

#ifdef P4EST_POINT_LOOKUP
  } else if ( remote_matches->elem_count != 0 ) { /* quadrant belongs to a processor that is not included in the ghost layer */
#else
  } else if ( remote_matches.size() != 0) { /* quadrant belongs to a processor that is not included in the ghost layer */
#endif
    /*
     * HACK: This is a hack to take care of multiple remote matches! The real and
     * correct way of doing it is to ask each remote processor a status flag which
     * will determine if they can actually compute the correct value.
     *
     * The way this hack works is each of the possible remote processors will
     * try to look up a point. If they own the point, they will perform the
     * interpolation as intended.
     *
     * If they do not own the point, but the point ends up in the ghost layer,
     * this is a valid request that they cannot interpolate. To make things easier
     * they will return 0 which then will be added to the exiting value and thus
     * wont change the correct result.
     *
     * Finally, if they do not own the point and it is determind to be remote,
     * this would correspond to an invalid request that should not have been made
     * in the first place! Most probably this would be a bug!
     *
     * So, we will set the output vector to zero first for all remote matches. We
     * can do the same for essentially all querries (local, ghost, and remote) but
     * in the case of local and ghost this is not really needed since there will
     * always be a single valid value.
     */
    // make sure remote ranks are unique
    std::set<int> remote_ranks;

#ifdef P4EST_POINT_LOOKUP
    for (size_t i=0; i<remote_matches->elem_count; i++){
      p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(remote_matches, i);
      remote_ranks.insert(q->p.piggy1.owner_rank);
    }
#else
    for (size_t i=0; i<remote_matches.size(); i++)
      remote_ranks.insert(remote_matches[i].p.piggy1.owner_rank);
#endif

    std::set<int>::const_iterator it = remote_ranks.begin(),
        end = remote_ranks.end();
    for(; it != end; ++it){
      int r = *it;

      for (short i = 0; i<P4EST_DIM; i++)
        remote_send_buffer[r].push_back(xyz[i]);

      remote_node_index[r].push_back(node_locidx);
    }
  }
  else {
#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif
    throw std::runtime_error("[ERROR_YOU_ARE_DOOMED]: InterpolatingFunction::add_point_to_buffer: no quadrant found ... auto-destruct initialized ....\n");
  }

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif
  // set the flag to false so the prepare buffer method will be called
  is_buffer_prepared = false;
}

void InterpolatingFunctionCellBase::set_input_parameters(Vec input_vec, interpolation_method method)
{
  input_vec_ = input_vec;
  method_ = method;
}

void InterpolatingFunctionCellBase::interpolate(Vec output_vec)
{
  double *Fo_p;
  ierr = VecGetArray(output_vec, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(output_vec, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunctionCellBase::interpolate( double *output_vec )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  // Get a pointer to the data
  double *Fi_p, *Fo_p = output_vec;
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);

  // first try interpolating on the ghost cells
  for (size_t i=0; i<ghost_point_buffer.size(); ++i)
  {
    double *xyz = &ghost_point_buffer.xyz[P4EST_DIM*i];
    p4est_quadrant_t &quad = ghost_point_buffer.quad[i];
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + p4est_->local_num_quadrants;
    p4est_locidx_t out_idx = ghost_point_buffer.output_idx[i];

    try{
      if (method_ == linear)
        Fo_p[out_idx] = cell_based_linear_interpolation(quad, quad_idx, Fi_p, xyz);
      else if (method_ == IDW)
        Fo_p[out_idx] = cell_based_IDW_interpolation(quad, quad_idx, Fi_p, xyz);
      else
        Fo_p[out_idx] = cell_based_LSQR_interpolation(quad, quad_idx, Fi_p, xyz);
    } catch (const std::exception& e) {
      (void) e;
      /* if the interpolation failed locally, send it to the remote
       * Note that if this also fails on the remote processors, then
       * the exception is actually an error in which case the remote
       * processor will throw.
       *
       * Also note that we only add to the list if the buffer is still
       * not been sent otherwise we could duplicate points. This happens
       * if one decides to interpolate multiple values using the same set
       * of points.
       */
      if (!is_buffer_prepared){
        int rank = ghost_point_buffer.rank[i];
        for (short i = 0; i<P4EST_DIM; i++)
          remote_send_buffer[rank].push_back(xyz[i]);
        remote_node_index[rank].push_back(out_idx);
      }
    }
  }

  // begin sending point buffers
  if (!is_buffer_prepared)
    send_point_buffers_begin();

  // initialize the value for remote matches to zero
  {
    nonlocal_node_map::iterator it = remote_node_index.begin(),
        end = remote_node_index.end();

    for(; it != end; ++it){
      std::vector<p4est_locidx_t>& locidx = it->second;
      for (size_t i=0; i<locidx.size(); ++i)
        Fo_p[locidx[i]] = 0;
    }
  }

  // Do the interpolation for local points
  for (size_t i=0; i<local_point_buffer.size(); ++i)
  {
    double *xyz = &local_point_buffer.xyz[P4EST_DIM*i];
    p4est_quadrant_t &quad = local_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
    p4est_locidx_t out_idx = local_point_buffer.output_idx[i];

    if (method_ == linear)
      Fo_p[out_idx] = cell_based_linear_interpolation(quad, quad_idx, Fi_p, xyz);
    else if (method_ == IDW)
      Fo_p[out_idx] = cell_based_IDW_interpolation(quad, quad_idx, Fi_p, xyz);
    else
      Fo_p[out_idx] = cell_based_LSQR_interpolation(quad, quad_idx, Fi_p, xyz);
  }  

  // begin recieving point buffers
  if (!is_buffer_prepared)
    recv_point_buffers_begin();

  // begin data send/recv for remote
  typedef std::map<int, std::vector<double> > data_transfer_map;
  data_transfer_map f_remote_send;
  std::vector<MPI_Request> remote_data_send_req(remote_senders.size());

  // Do interpolation for remote points
  {
    remote_transfer_map::iterator it = remote_recv_buffer.begin(),
        end = remote_recv_buffer.end();

    int req_counter = 0;
    for (; it != end; ++it, ++req_counter)
    {
      int send_rank = it->first;

      // make sure we received the buffers for this process before accesing data
      if(!is_buffer_prepared)
        MPI_Wait(&remote_recv_req[req_counter], MPI_STATUS_IGNORE);

      std::vector<double>& xyz_recv = it->second;
      std::vector<double>& f_send = f_remote_send[send_rank];
      f_send.resize(xyz_recv.size()/P4EST_DIM);

      for (size_t i=0; i<xyz_recv.size()/P4EST_DIM; i++)
      {
        double *xyz = &xyz_recv[P4EST_DIM*i];

        /* first clip the coordinates */
        double xyz_clip [] =
        {
          xyz[0], xyz[1]
#ifdef P4_TO_P8
          , xyz[2]
#endif
        };

        // clip to bounding box
        for (short j=0; j<P4EST_DIM; j++){
          if (xyz_clip[j] > xyz_max[j]) xyz_clip[j] = xyz_max[j];
          if (xyz_clip[j] < xyz_min[j]) xyz_clip[j] = xyz_min[j];
        }

        // first find the quadrant for the remote points
        p4est_quadrant_t best_match;
#ifdef P4EST_POINT_LOOKUP
        sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
        int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
        std::vector<p4est_quadrant_t> remote_matches;
        int rank_found = cnnn_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif
        // make sure that the point belongs to us
        if (rank_found == p4est_->mpirank){ // if we own the point, interpolate
          // get the local index
          p4est_topidx_t tree_idx  = best_match.p.piggy3.which_tree;
          p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
          p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;

          if (method_ == linear)
            f_send[i] = cell_based_linear_interpolation(best_match, qu_locidx, Fi_p, xyz);
          else if (method_ == IDW)
            f_send[i] = cell_based_IDW_interpolation(best_match, qu_locidx, Fi_p, xyz);
          else
            f_send[i] = cell_based_LSQR_interpolation(best_match, qu_locidx, Fi_p, xyz);

        } else if ( rank_found != -1 ) {
          // if we don't own the point but its in the ghost layer return 0.
          f_send[i] = 0;
        } else { /* if we dont the own the point, and its not in the ghost layer
                  * this MUST be a bug or mistake so simply throw.
                  */
          std::ostringstream oss;
          oss << "[ERROR]: Point (" << xyz[0] << "," << xyz[1] <<
#ifdef P4_TO_P8
            xyz[2] <<
#endif
              ") was flagged as a remote point to either belong to processor "
              << p4est_->mpirank << " or be in its ghost layer, both of which"
                 " have failed. Found rank is = " << rank_found
              << " and remote_macthes->elem_count = "
#ifdef P4EST_POINT_LOOKUP
              << remote_matches->elem_count << ". This is most certainly a bug."
#else
              << remote_matches.size() << ". This is most certainly a bug."
#endif
              << std::endl;
          throw std::runtime_error(oss.str());
        }
#ifdef P4EST_POINT_LOOKUP
        sc_array_destroy(remote_matches);
#endif
      }

      // send the buffer
      MPI_Isend(&f_send[0], f_send.size(), MPI_DOUBLE, send_rank, remote_data_tag, p4est_->mpicomm, &remote_data_send_req[req_counter]);
    }
  }

  // Receive the interpolated remote data and put them in the correct position.
  {
    for (size_t i=0; i<remote_receivers.size(); ++i)
    {
      int recv_rank = remote_receivers[i];
      std::vector<p4est_locidx_t>& node_idx = remote_node_index[recv_rank];
      std::vector<double> f_recv(node_idx.size());
      MPI_Recv(&f_recv[0], f_recv.size(), MPI_DOUBLE, recv_rank, remote_data_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      for (size_t j=0; j<f_recv.size(); j++){
        Fo_p[node_idx[j]] += f_recv[j];
      }
    }
  }

  // wait for all buffer sends to finish
  if (!is_buffer_prepared){
    MPI_Waitall(remote_send_req.size(), &remote_send_req[0], MPI_STATUSES_IGNORE);

    /* set the buffer flag to true so that we wont wait for other interpolations
     * if the grid is unchanged
     */
    is_buffer_prepared = true;
  }

  // wait for all data send buffers to finish
  MPI_Waitall(remote_data_send_req.size(), &remote_data_send_req[0], MPI_STATUSES_IGNORE);

  // Restore the pointer
  ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
double InterpolatingFunctionCellBase::operator ()(double x, double y, double z) const
#else
double InterpolatingFunctionCellBase::operator ()(double x, double y) const
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
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  p4est_quadrant_t best_match;

#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = cnnn_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif

  if (rank_found == p4est_->mpirank) { // local quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

    if (method_ == linear)
      return cell_based_linear_interpolation(best_match, quad_idx, Fi_p, xyz);
    else if (method_ == IDW)
      return cell_based_IDW_interpolation(best_match, quad_idx, Fi_p, xyz);
    else
      return cell_based_LSQR_interpolation(best_match, quad_idx, Fi_p, xyz);

  } else if ( rank_found != -1 ) { // ghost quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    quad_idx = best_match.p.piggy3.local_num + p4est_->local_num_quadrants;

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

    if (method_ == linear)
      return cell_based_linear_interpolation(best_match, quad_idx, Fi_p, xyz);
    else if (method_ == IDW)
      return cell_based_IDW_interpolation(best_match, quad_idx, Fi_p, xyz);
    else
      return cell_based_LSQR_interpolation(best_match, quad_idx, Fi_p, xyz);

  } else {
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

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

void InterpolatingFunctionCellBase::send_point_buffers_begin()
{
  int req_counter = 0;

  remote_transfer_map::iterator it = remote_send_buffer.begin(), end = remote_send_buffer.end();
  for (;it != end; ++it)
    remote_receivers.push_back(it->first);

  // notify the other processors
  int num_senders;
  sc_notify(&remote_receivers[0], remote_receivers.size(), &remote_senders[0], &num_senders, p4est_->mpicomm);
  remote_senders.resize(num_senders);

  // Allocate enough requests slots
  remote_send_req.resize(remote_receivers.size());

  // Now that we know all sender/receiver pairs lets do MPI. We do blocking
  for (it = remote_send_buffer.begin(); it != end; ++it, ++req_counter){

    std::vector<double>& buff = it->second;
    int msg_size = buff.size();

    MPI_Isend(&buff[0], msg_size, MPI_DOUBLE, it->first, remote_point_tag, p4est_->mpicomm, &remote_send_req[req_counter]);
  }
}

void InterpolatingFunctionCellBase::recv_point_buffers_begin()
{
  // Allocate enough requests slots
  remote_recv_req.resize(remote_senders.size());

  // Now lets receive the stuff
  for (size_t i=0; i<remote_senders.size(); ++i){
    std::vector<double>& buff = remote_recv_buffer[remote_senders[i]];

    // Get the size and resize the recv buffer
    int msg_size;
    MPI_Status st;
    MPI_Probe(remote_senders[i], remote_point_tag, p4est_->mpicomm, &st);
    MPI_Get_count(&st, MPI_DOUBLE, &msg_size);
    buff.resize(msg_size);

    // Receive the data
    MPI_Irecv(&buff[0], msg_size, MPI_DOUBLE, remote_senders[i], remote_point_tag, p4est_->mpicomm, &remote_recv_req[i]);
  }
}

double InterpolatingFunctionCellBase::cell_based_linear_interpolation(const p4est_quadrant_t &quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const
{
  p4est_topidx_t tr_id = quad.p.piggy3.which_tree;
  p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[P4EST_CHILDREN*tr_id];
  double tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
  double tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tr_zmin = p4est_->connectivity->vertices[3*v_mmm + 2];
#endif

  double qh = (double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN;
  double xyz_C [] =
  {
    quad_x_fr_i(&quad) + 0.5*qh + tr_xmin,
    quad_y_fr_j(&quad) + 0.5*qh + tr_ymin
  #ifdef P4_TO_P8
    ,
    quad_z_fr_k(&quad) + 0.5*qh + tr_zmin
  #endif
  };

  double xyz_search [] =
  {
    xyz[0],
    xyz[1]
  #ifdef P4_TO_P8
    ,
    xyz[2]
  #endif
  };

  // correct for the search location if on the wall
  if (is_quad_xmWall(p4est_, tr_id, &quad) && xyz[0] < xyz_C[0])
    xyz_search[0] = xyz_C[0] + 0.5*qh;
  if (is_quad_xpWall(p4est_, tr_id, &quad) && xyz[0] > xyz_C[0])
    xyz_search[0] = xyz_C[0] - 0.5*qh;
  if (is_quad_ymWall(p4est_, tr_id, &quad) && xyz[1] < xyz_C[1])
    xyz_search[1] = xyz_C[1] + 0.5*qh;
  if (is_quad_ypWall(p4est_, tr_id, &quad) && xyz[1] > xyz_C[1])
    xyz_search[1] = xyz_C[1] - 0.5*qh;
#ifdef P4_TO_P8
  if (is_quad_zmWall(p4est_, tr_id, &quad) && xyz[2] < xyz_C[2])
    xyz_search[2] = xyz_C[2] + 0.5*qh;
  if (is_quad_zpWall(p4est_, tr_id, &quad) && xyz[2] > xyz_C[2])
    xyz_search[2] = xyz_C[2] - 0.5*qh;
#endif

  // check if the point is close to the center cell
#ifdef P4_TO_P8
  Point3 r_search(xyz_search[0] - xyz_C[0], xyz_search[1] - xyz_C[1], xyz_search[2] - xyz_C[2]);
  Point2 r(xyz[0] - xyz_C[0], xyz[1] - xyz_C[1], xyz[2] - xyz_C[2]);
#else
  Point2 r_search(xyz_search[0] - xyz_C[0], xyz_search[1] - xyz_C[1]);
  Point2 r(xyz[0] - xyz_C[0], xyz[1] - xyz_C[1]);
#endif

  if (r_search.norm_L2()/qh < EPS)
    return Fi_p[quad_idx];

  // decide where to look for the triangle based on the relative coordinate
  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = cnnn_->begin(quad_idx, i);
  cells[P4EST_FACES] = cnnn_->end(quad_idx, P4EST_FACES - 1);

  size_t num_ngbd_cells[P4EST_FACES];
  for (short i = 0; i<P4EST_FACES; i++)
    num_ngbd_cells[i] = cells[i+1] - cells[i];

  std::vector<const quad_info_t*> points;
  points.reserve(P4EST_DIM*(cells[P4EST_FACES] - cells[0])); // estimate

#ifdef P4_TO_P8
  throw std::runtime_error("[ERROR]: Not implemented");
#else
  /* add points to the list for testing. There are 8 cases in 2D:
   * 4 face directions.
   * 4 corners
   */

  // 0 - 3 first trying face directions
  for (short i = 0; i<P4EST_FACES; i++)
    for (const quad_info_t* it = cells[i]; it < cells[i+1] - 1; ++it){
      points.push_back(it);
      points.push_back(it+1);
    }

  /* corners */
  // 4 - mm corner
  if(num_ngbd_cells[dir::f_m00] != 0 && num_ngbd_cells[dir::f_0m0] != 0){
    points.push_back(cells[dir::f_m00]);
    points.push_back(cells[dir::f_0m0]);
  }

  // 5 - pm corner
  if(num_ngbd_cells[dir::f_p00] != 0 && num_ngbd_cells[dir::f_0m0] != 0){
    points.push_back(cells[dir::f_p00]);
    points.push_back(cells[dir::f_0p0] - 1);
  }

  // 6 - mp corner
  if(num_ngbd_cells[dir::f_m00] != 0 && num_ngbd_cells[dir::f_0p0] != 0){
    points.push_back(cells[dir::f_p00] - 1);
    points.push_back(cells[dir::f_0p0]);
  }

  // 7 - pp corner
  if(num_ngbd_cells[dir::f_p00] != 0 && num_ngbd_cells[dir::f_0p0] != 0){
    points.push_back(cells[dir::f_0m0] - 1);
    points.push_back(cells[P4EST_FACES] - 1);
  }

  // loop over cells and check if they contain our point
  for (size_t i = 0; i<points.size(); i += P4EST_DIM){
    const quad_info_t *it1 = points[i];
    const quad_info_t *it2 = points[i+1];

    // p1
    v_mmm = p4est_->connectivity->tree_to_vertex[P4EST_CHILDREN*it1->tree_idx];
    tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
    tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
    qh = (double)P4EST_QUADRANT_LEN(it1->level)/(double)P4EST_ROOT_LEN;

    Point2 r1;
    r1.x = quad_x_fr_i(it1->quad) + 0.5*qh + tr_xmin - xyz_C[0];
    r1.y = quad_y_fr_j(it1->quad) + 0.5*qh + tr_ymin - xyz_C[1];

    // p2
    v_mmm = p4est_->connectivity->tree_to_vertex[P4EST_CHILDREN*it2->tree_idx];
    tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
    tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
    qh = (double)P4EST_QUADRANT_LEN(it2->level)/(double)P4EST_ROOT_LEN;

    Point2 r2;
    r2.x = quad_x_fr_i(it2->quad) + 0.5*qh + tr_xmin - xyz_C[0];
    r2.y = quad_y_fr_j(it2->quad) + 0.5*qh + tr_ymin - xyz_C[1];

    // check if we have found the point
    double det = r1.cross(r2);
    double u1  =  r_search.cross(r2) / det;
    double u2  = -r_search.cross(r1) / det;

    if (u1 >= 0.0 && u2 >= 0.0 && u1+u2 <= 1.0){
      // use the actual point for interpolation
      u1  =  r.cross(r2) / det;
      u2  = -r.cross(r1) / det;
      return (Fi_p[quad_idx]*(1 - u1 - u2) + Fi_p[it1->locidx]*u1 + Fi_p[it2->locidx]*u2);
    }
  }
#endif
  throw std::runtime_error("[ERROR]: Could not find a suitable triangulation");
}

double InterpolatingFunctionCellBase::cell_based_IDW_interpolation(const p4est_quadrant_t &quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const
{
  p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[P4EST_CHILDREN*quad.p.piggy3.which_tree];
  double tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
  double tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tr_zmin = p4est_->connectivity->vertices[3*v_mmm + 2];
#endif

  double qh = (double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN;
  double xyz_C [] =
  {
    quad_x_fr_i(&quad) + 0.5*qh + tr_xmin,
    quad_y_fr_j(&quad) + 0.5*qh + tr_ymin
  #ifdef P4_TO_P8
    ,
    quad_z_fr_k(&quad) + 0.5*qh + tr_zmin
  #endif
  };

  /* loop over all quadrants and compute weights */
  const quad_info_t *begin = cnnn_->begin(quad_idx, 0);
  const quad_info_t *end   = cnnn_->end(quad_idx, P4EST_FACES - 1);

#ifdef P4_TO_P8
  double w = 1.0/(SQR(xyz_C[0] - xyz[0]) + SQR(xyz_C[1] - xyz[1]) + SQR(xyz_C[2]-xyz[2]));
#else
  double w = 1.0/(SQR(xyz_C[0] - xyz[0]) + SQR(xyz_C[1] - xyz[1]));
#endif
  double sum = w;
  double res = w*Fi_p[quad_idx];

  for (const quad_info_t *it = begin; it != end; ++it)
  {
    v_mmm = p4est_->connectivity->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];

    tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
    tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
  #ifdef P4_TO_P8
    tr_zmin = p4est_->connectivity->vertices[3*v_mmm + 2];
  #endif

    qh = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;
    xyz_C[0] = quad_x_fr_i(it->quad) + 0.5*qh + tr_xmin;
    xyz_C[1] = quad_y_fr_j(it->quad) + 0.5*qh + tr_ymin;
#ifdef P4_TO_P8
    xyz_C[2] = quad_z_fr_k(it->quad) + 0.5*qh + tr_zmin;
#endif

#ifdef P4_TO_P8
    w = 1.0/(SQR(xyz_C[0] - xyz[0]) + SQR(xyz_C[1] - xyz[1]) + SQR(xyz_C[2]-xyz[2]));
#else
    w = 1.0/(SQR(xyz_C[0] - xyz[0]) + SQR(xyz_C[1] - xyz[1]));
#endif
    sum += w;
    res += w*Fi_p[it->locidx];
  }

  return res/sum;
}

double InterpolatingFunctionCellBase::cell_based_LSQR_interpolation(const p4est_quadrant_t &quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const
{
  throw std::runtime_error("[ERROR]: Not implemented!");
}

