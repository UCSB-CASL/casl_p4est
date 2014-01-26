#ifdef P4_TO_P8
#include "my_p8est_interpolating_function.h"
#include <src/my_p8est_vtk.h>
#else
#include "my_p4est_interpolating_function.h"
#include <src/my_p4est_vtk.h>
#endif

#include "petsc_compatibility.h"
// #include <sc_notify.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/ipm_logging.h>
#include <mpi.h>
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
extern PetscLogEvent log_InterpolatingFunction_send_buffer;
extern PetscLogEvent log_InterpolatingFunction_recv_buffer;
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

InterpolatingFunctionNodeBase::InterpolatingFunctionNodeBase(p4est_t *p4est,
                                             p4est_nodes_t *nodes,
                                             p4est_ghost_t *ghost,
                                             my_p4est_brick_t *myb)
  : method_(linear),
    p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb), neighbors_(NULL),
    Fxx_(NULL), Fyy_(NULL),
#ifdef P4_TO_P8
    Fzz_(NULL),
#endif
    local_derivatives(false),
    is_buffer_prepared(false),
    notify_send_req(p4est->mpisize),
    notify_recv_req(p4est->mpisize),
    notify_send_buf(p4est->mpisize),
    notify_recv_buf(p4est->mpisize)
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

InterpolatingFunctionNodeBase::InterpolatingFunctionNodeBase(p4est_t *p4est,
                                             p4est_nodes_t *nodes,
                                             p4est_ghost_t *ghost,
                                             my_p4est_brick_t *myb,
                                             const my_p4est_node_neighbors_t *neighbors)
  : method_(quadratic_non_oscillatory),
    p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb), neighbors_(neighbors),
    Fxx_(NULL), Fyy_(NULL),
#ifdef P4_TO_P8
        Fzz_(NULL),
#endif
    local_derivatives(false),
    is_buffer_prepared(false),
    notify_send_req(p4est->mpisize),
    notify_recv_req(p4est->mpisize),
    notify_send_buf(p4est->mpisize),
    notify_recv_buf(p4est->mpisize)
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

InterpolatingFunctionNodeBase::~InterpolatingFunctionNodeBase()
{
  if ((method_ == quadratic || method_ == quadratic_non_oscillatory) && local_derivatives)
  {
    ierr = VecDestroy(Fxx_); CHKERRXX(ierr);
    ierr = VecDestroy(Fyy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(Fzz_); CHKERRXX(ierr);
#endif
  }
}

void InterpolatingFunctionNodeBase::add_point_to_buffer(p4est_locidx_t node_locidx, const double *xyz)
{
  /* first clip the coordinates */
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
#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank) { // local quadrant

    for (short i = 0; i<P4EST_DIM; i++)
      local_point_buffer.xyz.push_back(xyz[i]);

    local_point_buffer.quad.push_back(best_match);
    local_point_buffer.node_locidx.push_back(node_locidx);

  } else if ( rank_found != -1 ) { /* point is found in the ghost layer */
    /*
     * WARNING: This is (I think) a bug in p4est. In the documentation, it is
     * said that for ghost quadrants " ... piggy3 data member is filled with
     * their owner's tree and local number." (cf. p4est_ghost.h, line 34). This
     * implies that the numbering, i.e. p.piggy3.local_num field, stores the
     * local index w.r.t the tree it is located in, i.e. p.piggy3.which_tree.
     *
     * Looking at the implementation, however, it seems that the local numbering
     * is stored w.r.t all of quadrants, i.e. local_num + tree->quadrant_offsets
     * which counter-intutive and inconssitent with the rest of p4est code. We
     * take this into account anyway when trying to access the quadrant on the
     * remote processor. (cf. interpolate function below)
     */

#ifdef GHOST_REMOTE_INTERPOLATION
    for (short i = 0; i<P4EST_DIM; i++)
      remote_send_buffer[rank_found].push_back(xyz[i]);

    remote_node_index[rank_found].push_back(node_locidx);
#else
    for (short i = 0; i<P4EST_DIM; i++)
      ghost_point_buffer.xyz.push_back(xyz[i]);

    ghost_point_buffer.quad.push_back(best_match);
    ghost_point_buffer.node_locidx.push_back(node_locidx);
#endif

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

void InterpolatingFunctionNodeBase::set_input_parameters(Vec input_vec, interpolation_method method, Vec Fxx, Vec Fyy
#ifdef P4_TO_P8
                                                 , Vec Fzz
#endif
                                                 )
{
  method_ = method;

  if (neighbors_ == NULL && method != linear)
    throw std::invalid_argument("[ERROR]: a quadratic method requires a valid QNNN object but none was given");
  input_vec_ = input_vec;

  // compute the second derivates if necessary
  if (method_ == quadratic || method_ == quadratic_non_oscillatory){
    if (Fxx != NULL && Fyy != NULL
#ifdef P4_TO_P8
     && Fzz != NULL
#endif
       )
    {
       Fxx_ = Fxx;
       Fyy_ = Fyy;
#ifdef P4_TO_P8
       Fzz_ = Fzz;
#endif
       local_derivatives = false;
    } else {
      compute_second_derivatives();
    }
  }
}

void InterpolatingFunctionNodeBase::interpolate(Vec output_vec)
{
  double *Fo_p;
  ierr = VecGetArray(output_vec, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(output_vec, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBase::interpolate( double *output_vec )
{	
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);  

  // begin sending point buffers
  if (!is_buffer_prepared)
    send_point_buffers_begin();

  IPMLogRegionBegin("interpolate");

  // Get a pointer to the data
  double *Fi_p, *Fo_p = output_vec;
  double *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  double *Fzz_p;
#endif
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);

  if (method_ == linear){
    Fxx_p = Fyy_p = NULL;
#ifdef P4_TO_P8
    Fzz_p = NULL;
#endif
  } else {
    ierr = VecGetArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif
  }

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

  // Get access to the node numbering of all quadrants
  p4est_locidx_t *q2n = nodes_->local_nodes;
  double f  [P4EST_CHILDREN];
  double fdd[P4EST_DIM*P4EST_CHILDREN]; // fxx[0] = fdd[0], fyy[0] = fdd[1], fzz[0] = fdd[2], fxx[1] = fdd[3], ... 

  // Do the interpolation for local points
  for (size_t i=0; i<local_point_buffer.size(); ++i)
  {
    double *xyz = &local_point_buffer.xyz[P4EST_DIM*i];
    p4est_quadrant_t &quad = local_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
    p4est_locidx_t node_idx = local_point_buffer.node_locidx[i];

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = q2n[quad_idx*P4EST_CHILDREN+j];
        fdd[P4EST_DIM * j + 0] = Fxx_p[node_locidx];
        fdd[P4EST_DIM * j + 1] = Fyy_p[node_locidx];
#ifdef P4_TO_P8
        fdd[P4EST_DIM * j + 2] = Fzz_p[node_locidx];
#endif
      }
    }

    if (method_ == linear)
      Fo_p[node_idx] = linear_interpolation(p4est_, tree_idx, quad, f, xyz);
    else if (method_ == quadratic)
      Fo_p[node_idx] = quadratic_interpolation(p4est_, tree_idx, quad, f, fdd, xyz);
    else
      Fo_p[node_idx] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, quad, f, fdd, xyz);
  }

  // Do the interpolation for ghost points
  for (size_t i=0; i<ghost_point_buffer.size(); ++i)
  {
    double *xyz = &ghost_point_buffer.xyz[P4EST_DIM*i];
    p4est_quadrant_t &quad = ghost_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + p4est_->local_num_quadrants;
    p4est_locidx_t node_idx = ghost_point_buffer.node_locidx[i];

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = q2n[quad_idx*P4EST_CHILDREN+j];
        fdd[P4EST_DIM * j + 0] = Fxx_p[node_locidx];
        fdd[P4EST_DIM * j + 1] = Fyy_p[node_locidx];
#ifdef P4_TO_P8
        fdd[P4EST_DIM * j + 2] = Fzz_p[node_locidx];
#endif
      }
    }

    if (method_ == linear)
      Fo_p[node_idx] = linear_interpolation(p4est_, tree_idx, quad, f, xyz);
    else if (method_ == quadratic)
      Fo_p[node_idx] = quadratic_interpolation(p4est_, tree_idx, quad, f, fdd, xyz);
    else
      Fo_p[node_idx] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, quad, f, fdd, xyz);
  }

  IPMLogRegionEnd("interpolate");
  
  // begin recieving point buffers
  if (!is_buffer_prepared)
    recv_point_buffers_begin();

/*
	PetscSynchronizedPrintf(p4est_->mpicomm, "[%2d] %2d pairs (rank, elem): ", p4est_->mpirank, remote_senders.size());
	for (size_t i = 0; i<remote_senders.size(); i++)
		PetscSynchronizedPrintf(p4est_->mpicomm, "(%2d, %5d) ", remote_senders[i], remote_recv_buffer[remote_senders[i]].size());
	PetscSynchronizedPrintf(p4est_->mpicomm, "\n");
	PetscSynchronizedFlush(p4est_->mpicomm);
*/
	
  IPMLogRegionBegin("interpolate");

  // begin data send/recv for ghost and remote
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
        int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif
        // make sure that the point belongs to us
        if (rank_found == p4est_->mpirank){ // if we own the point, interpolate
          // get the local index
          p4est_topidx_t tree_idx  = best_match.p.piggy3.which_tree;
          p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
          p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;

          for (short j=0; j<P4EST_CHILDREN; j++)
            f[j] = Fi_p[q2n[qu_locidx*P4EST_CHILDREN+j]];

          // get access to second derivatives only if needed
          if (method_ == quadratic || method_ == quadratic_non_oscillatory)
          {
            for (short j=0; j<P4EST_CHILDREN; ++j)
            {
              p4est_locidx_t node_locidx = q2n[qu_locidx*P4EST_CHILDREN+j];
              fdd[P4EST_DIM * j + 0] = Fxx_p[node_locidx];
              fdd[P4EST_DIM * j + 1] = Fyy_p[node_locidx];
#ifdef P4_TO_P8
              fdd[P4EST_DIM * j + 2] = Fzz_p[node_locidx];
#endif
            }
          }

          if (method_ == linear)
            f_send[i] = linear_interpolation(p4est_, tree_idx, best_match, f, xyz);
          else if (method_ == quadratic)
            f_send[i] = quadratic_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);
          else
            f_send[i] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);

        } else if ( rank_found != -1 ) {
          // if we don't own the point but its in the ghost layer return 0.
          f_send[i] = 0;
        } else { /* if we dont the own the point, and its not in the ghost layer
                  * this MUST be a bug or mistake so simply throw.
                  */
          std::ostringstream oss;
          oss << "[ERROR]: Point (" << xyz[0] << "," << xyz[1] << 
#ifdef P4_TO_P8
            "," << xyz[2] <<
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

  if (method_ == quadratic || method_ == quadratic_non_oscillatory)
  {
    ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  IPMLogRegionEnd("interpolate");
  ierr = PetscLogEventEnd(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);	
}

#ifdef P4_TO_P8
double InterpolatingFunctionNodeBase::operator ()(double x, double y, double z) const
#else
double InterpolatingFunctionNodeBase::operator ()(double x, double y) const
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

  double *Fi_p, *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  double *Fzz_p;
#endif  

  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);
  if (method_ == linear){
    Fxx_p = Fyy_p = NULL;
#ifdef P4_TO_P8
    Fzz_p = NULL;
#endif
  } else {
    ierr = VecGetArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  static double f  [P4EST_CHILDREN];
  static double fdd[P4EST_DIM*P4EST_CHILDREN];

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t best_match;

#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
#endif

  if (rank_found == p4est_->mpirank) { // local quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = q2n[quad_idx*P4EST_CHILDREN+j];
        fdd[P4EST_DIM * j + 0] = Fxx_p[node_locidx];
        fdd[P4EST_DIM * j + 1] = Fyy_p[node_locidx];
#ifdef P4_TO_P8
        fdd[P4EST_DIM * j + 2] = Fzz_p[node_locidx];
#endif
      }
    }

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif
    }
#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif
    if (method_ == linear)
      return linear_interpolation(p4est_, tree_idx, best_match, f, xyz);
    else if (method_ == quadratic)
      return quadratic_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);
    else
      return quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);

  } else if ( rank_found != -1 ) { // ghost quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    quad_idx = best_match.p.piggy3.local_num + p4est_->local_num_quadrants;

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[q2n[quad_idx*P4EST_CHILDREN+j]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = q2n[quad_idx*P4EST_CHILDREN+j];
        fdd[P4EST_DIM * j + 0] = Fxx_p[node_locidx];
        fdd[P4EST_DIM * j + 1] = Fyy_p[node_locidx];
#ifdef P4_TO_P8
        fdd[P4EST_DIM * j + 2] = Fzz_p[node_locidx];
#endif
      }
    }
    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif
    }

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

    if (method_ == linear)
      return linear_interpolation(p4est_, tree_idx, best_match, f, xyz);
    else if (method_ == quadratic)
      return quadratic_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);
    else
      return quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fdd, xyz);

  } else {
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(Fzz_, &Fzz_p); CHKERRXX(ierr);
#endif

    }

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

void InterpolatingFunctionNodeBase::send_point_buffers_begin()
{
  ierr = PetscLogEventBegin(log_InterpolatingFunction_send_buffer, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("send_point_buffer");
  int req_counter = 0;

  remote_transfer_map::iterator it = remote_send_buffer.begin(), end = remote_send_buffer.end();
  for (;it != end; ++it)
    remote_receivers.push_back(it->first);

/*
  IPMLogRegionEnd("send_point_buffer");
  
  // notify the other processors
  IPMLogRegionBegin("notifyig_others");
  for (int r = 0; r<p4est_->mpisize; ++r)
    notify_send_buf[r] = P4EST_FALSE;

  for (size_t i=0; i<remote_receivers.size(); ++i)
    notify_send_buf[remote_receivers[i]] = P4EST_TRUE;

  // PetscSynchronizedPrintf(p4est_->mpicomm, "[%d]",p4est_->mpirank);
  // for (int r = 0; r<p4est_->mpisize; ++r)
  //   PetscSynchronizedPrintf(p4est_->mpicomm, "%d ",(int)notify_send_buf[r]);
  // PetscSynchronizedPrintf(p4est_->mpicomm, "\n",p4est_->mpirank);
  // PetscSynchronizedFlush(p4est_->mpicomm);

  for (int r = 0; r<p4est_->mpisize; ++r){
    MPI_Isend(&notify_send_buf[r], 1, MPI_CHAR, r, remote_notify_tag, p4est_->mpicomm, &notify_send_req[r]);
    MPI_Irecv(&notify_recv_buf[r], 1, MPI_CHAR, r, remote_notify_tag, p4est_->mpicomm, &notify_recv_req[r]);
  }
  IPMLogRegionEnd("notifyig_others");  
*/
  // PetscSynchronizedPrintf(p4est_->mpicomm, "[%d] send_done\n",p4est_->mpirank);
  // PetscSynchronizedFlush(p4est_->mpicomm);
///*
  int num_senders; 
	remote_senders.resize(p4est_->mpisize); 
  my_sc_notify(&remote_receivers[0], remote_receivers.size(), &remote_senders[0], &num_senders, p4est_->mpicomm);	
  remote_senders.resize(num_senders);
//*/	
  IPMLogRegionBegin("send_point_buffer");
  // Allocate enough requests slots
  remote_send_req.resize(remote_receivers.size());

  // Now that we know all sender/receiver pairs lets do MPI. We do blocking
  for (it = remote_send_buffer.begin(); it != end; ++it, ++req_counter){

    std::vector<double>& buff = it->second;
    int msg_size = buff.size();

    MPI_Isend(&buff[0], msg_size, MPI_DOUBLE, it->first, remote_point_tag, p4est_->mpicomm, &remote_send_req[req_counter]);
  }

	// Ensure everyone has called MPI_Isend
	MPI_Barrier(p4est_->mpicomm);
	
  IPMLogRegionEnd("send_point_buffer");
  ierr = PetscLogEventEnd(log_InterpolatingFunction_send_buffer, 0, 0, 0, 0); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBase::recv_point_buffers_begin()
{
  ierr = PetscLogEventBegin(log_InterpolatingFunction_recv_buffer, 0, 0, 0, 0); CHKERRXX(ierr);

/*
  // make sure we have been properly notified!
  IPMLogRegionBegin("notifyig_others");
  MPI_Waitall(p4est_->mpisize, &notify_send_req[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(p4est_->mpisize, &notify_recv_req[0], MPI_STATUSES_IGNORE);

  // figure out who to expect a message from
  for (int r = 0; r<p4est_->mpisize; ++r)
    if (P4EST_TRUE == notify_recv_buf[r])
      remote_senders.push_back(r);

  // PetscSynchronizedPrintf(p4est_->mpicomm, "[%d] recv_done\n", p4est_->mpirank);
  // PetscSynchronizedFlush(p4est_->mpicomm);  

  IPMLogRegionEnd("notifyig_others");
*/
  IPMLogRegionBegin("recv_point_buffer");
/*
	int is_msg_pending;
	MPI_Status st;
	MPI_Request req;
	MPI_Iprobe(MPI_ANY_SOURCE, remote_point_tag, p4est_->mpicomm, &is_msg_pending, &st);
	while(is_msg_pending){
		int sender = st.MPI_SOURCE;
    std::vector<double>& buff = remote_recv_buffer[sender];
		remote_senders.push_back(sender);

		// get the msg size
		int msg_size;
		MPI_Get_count(&st, MPI_DOUBLE, &msg_size);
    buff.resize(msg_size);

    // Receive the data
    MPI_Irecv(&buff[0], msg_size, MPI_DOUBLE, sender, remote_point_tag, p4est_->mpicomm, &req);
 		remote_recv_req.push_back(req);

		// probe for the next msg
		MPI_Iprobe(MPI_ANY_SOURCE, remote_point_tag, p4est_->mpicomm, &is_msg_pending, &st);
	}

*/
///*
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
//*/
  IPMLogRegionEnd("recv_point_buffer");
  ierr = PetscLogEventEnd(log_InterpolatingFunction_recv_buffer, 0, 0, 0, 0); CHKERRXX(ierr);
}

void InterpolatingFunctionNodeBase::compute_second_derivatives()
{
  IPMLogRegionBegin("compute_2nd_derv");
  // Allocate memory for second derivaties
#ifdef P4_TO_P8
  if (Fxx_ == NULL && Fyy_ == NULL && Fzz_ == NULL)
#else
  if (Fxx_ == NULL && Fyy_ == NULL)
#endif
  {
    ierr = VecCreateGhostNodes(p4est_, nodes_, &Fxx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_, nodes_, &Fyy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est_, nodes_, &Fzz_); CHKERRXX(ierr);
#endif
    local_derivatives = true;
  }
#ifdef P4_TO_P8
  neighbors_->second_derivatives_central(input_vec_, Fxx_, Fyy_, Fzz_);
#else
  neighbors_->second_derivatives_central(input_vec_, Fxx_, Fyy_);
#endif
  IPMLogRegionEnd("compute_2nd_derv");
}
