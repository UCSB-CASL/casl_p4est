#include "interpolating_function.h"
#include "petsc_compatibility.h"
#include <sc_notify.h>
#include <src/my_p4est_vtk.h>
#include <mpi.h>

#include <fstream>
#include <set>

// Logging variables -- defined in src/petsc_logging.cpp
extern PetscLogEvent log_InterpolatingFunction_interpolate;

#ifndef CASL_LOG_EVENTS
#undef  PetscLogEventBegin(e, o1, o2, o3, o4)
#undef  PetscLogEventEnd(e, o1, o2, o3, o4)
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops(n)
#define PetscLogFlops(n) 0
#endif

InterpolatingFunction::InterpolatingFunction(p4est_t *p4est,
                                             p4est_nodes_t *nodes,
                                             p4est_ghost_t *ghost,
                                             my_p4est_brick_t *myb)
  : method_(linear),
    p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb), qnnn_(NULL),
    Fxx_(NULL), Fyy_(NULL), local_derivatives(false),
    p4est2petsc(nodes->indep_nodes.elem_count),
    remote_senders(p4est->mpisize, -1),
    is_buffer_prepared(false)
{
  const size_t loc_begin = static_cast<size_t>(nodes_->offset_owned_indeps);
  const size_t loc_end   = static_cast<size_t>(nodes_->num_owned_indeps+nodes_->offset_owned_indeps);

  for (size_t i=0; i<p4est2petsc.size(); ++i)
  {
    if (i<loc_begin)
      p4est2petsc[i] = i + nodes_->num_owned_indeps;
    else if (i<loc_end)
      p4est2petsc[i] = i - nodes_->offset_owned_indeps;
    else
      p4est2petsc[i] = i;
  }

  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;

  xmin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 0];
  ymin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 1];
  xmax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 0];
  ymax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 1];
}

InterpolatingFunction::InterpolatingFunction(p4est_t *p4est,
                                             p4est_nodes_t *nodes,
                                             p4est_ghost_t *ghost,
                                             my_p4est_brick_t *myb,
                                             const my_p4est_node_neighbors_t *qnnn)
  : method_(quadratic_non_oscillatory),
    p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb), qnnn_(qnnn),
    Fxx_(NULL), Fyy_(NULL), local_derivatives(false),
    p4est2petsc(nodes->indep_nodes.elem_count),
    remote_senders(p4est->mpisize, -1),
    is_buffer_prepared(false)
{
  const size_t loc_begin = static_cast<size_t>(nodes_->offset_owned_indeps);
  const size_t loc_end   = static_cast<size_t>(nodes_->num_owned_indeps+nodes_->offset_owned_indeps);

  for (size_t i=0; i<p4est2petsc.size(); ++i)
  {
    if (i<loc_begin)
      p4est2petsc[i] = i + nodes_->num_owned_indeps;
    else if (i<loc_end)
      p4est2petsc[i] = i - nodes_->offset_owned_indeps;
    else
      p4est2petsc[i] = i;
  }

  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;

  xmin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 0];
  ymin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 1];
  xmax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 0];
  ymax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 1];
}

InterpolatingFunction::~InterpolatingFunction()
{
  if ((method_ == quadratic || method_ == quadratic_non_oscillatory) && local_derivatives)
  {
    ierr = VecDestroy(Fxx_); CHKERRXX(ierr);
    ierr = VecDestroy(Fyy_); CHKERRXX(ierr);
  }
}

void InterpolatingFunction::add_point_to_buffer(p4est_locidx_t node_locidx, double x, double y)
{
  // initialize data
  double xy [] = {x, y};
  p4est_quadrant_t best_match;

  /* first clip the coordinates */
  double xy_clip [] = {x, y};
  // bousouf
  if(x>xmax) xy_clip[0] = xmax;
  if(x<xmin) xy_clip[0] = xmin;
  if(y>ymax) xy_clip[1] = ymax;
  if(y<ymin) xy_clip[1] = ymin;

  // find the quadrant -- Note point may become slightly purturbed after this call
#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = qnnn_->hierarchy->find_smallest_quadrant_containing_point(xy_clip, best_match, remote_matches);
#endif

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank) { // local quadrant

    local_point_buffer.xy.push_back(xy[0]);
    local_point_buffer.xy.push_back(xy[1]);
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

    ghost_point_buffer.xy.push_back(xy[0]);
    ghost_point_buffer.xy.push_back(xy[1]);
    ghost_point_buffer.quad.push_back(best_match);
    ghost_point_buffer.node_locidx.push_back(node_locidx);

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
      remote_send_buffer[r].push_back(xy[0]);
      remote_send_buffer[r].push_back(xy[1]);
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

void InterpolatingFunction::set_input_parameters(Vec input_vec, interpolation_method method, Vec Fxx, Vec Fyy)
{
  method_ = method;

  if (qnnn_ == NULL && method != linear)
    throw std::invalid_argument("[ERROR]: a quadratic method requires a valid QNNN object but none was given");
  input_vec_ = input_vec;

  // compute the second derivates if necessary
  if (method_ == quadratic || method_ == quadratic_non_oscillatory){
    if (Fxx != NULL && Fyy != NULL){
       Fxx_ = Fxx;
       Fyy_ = Fyy;
       local_derivatives = false;
    } else {
      compute_second_derivatives();
    }
  }
}

void InterpolatingFunction::interpolate(Vec output_vec)
{
  double *Fo_p;
  ierr = VecGetArray(output_vec, &Fo_p); CHKERRXX(ierr);
  interpolate(Fo_p);
  ierr = VecRestoreArray(output_vec, &Fo_p); CHKERRXX(ierr);
}

void InterpolatingFunction::interpolate( double *output_vec )
{
  ierr = PetscLogEventBegin(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

  // begin sending point buffers
  if (!is_buffer_prepared)
    send_point_buffers_begin();

  PetscErrorCode ierr;

  // Get a pointer to the data
  double *Fi_p, *Fo_p = output_vec;
  double *Fxx_p, *Fyy_p;
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);

  if (method_ == linear)
    Fxx_p = Fyy_p = NULL;
  else
  {
    ierr = VecGetArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
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
  double fxx[P4EST_CHILDREN];
  double fyy[P4EST_CHILDREN];

  // Do the interpolation for local points
  for (size_t i=0; i<local_point_buffer.size(); ++i)
  {
    double *xy = &local_point_buffer.xy[2*i];
    p4est_quadrant_t &quad = local_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
    p4est_locidx_t node_idx = local_point_buffer.node_locidx[i];

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
        fxx[j] = Fxx_p[node_locidx];
        fyy[j] = Fyy_p[node_locidx];
      }
    }

    if (method_ == linear)
      Fo_p[node_idx] = bilinear_interpolation(p4est_, tree_idx, quad, f, xy);
    else if (method_ == quadratic)
      Fo_p[node_idx] = quadratic_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
    else
      Fo_p[node_idx] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
  }

  // Do the interpolation for ghost points
  for (size_t i=0; i<ghost_point_buffer.size(); ++i)
  {
    double *xy = &ghost_point_buffer.xy[2*i];
    p4est_quadrant_t &quad = ghost_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_locidx_t quad_idx = quad.p.piggy3.local_num + p4est_->local_num_quadrants;
    p4est_locidx_t node_idx = ghost_point_buffer.node_locidx[i];

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
        fxx[j] = Fxx_p[node_locidx];
        fyy[j] = Fyy_p[node_locidx];
      }
    }

    if (method_ == linear)
      Fo_p[node_idx] = bilinear_interpolation(p4est_, tree_idx, quad, f, xy);
    else if (method_ == quadratic)
      Fo_p[node_idx] = quadratic_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
    else
      Fo_p[node_idx] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
  }

  // begin recieving point buffers
  if (!is_buffer_prepared)
    recv_point_buffers_begin();

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

      std::vector<double>& xy_recv = it->second;
      std::vector<double>& f_send = f_remote_send[send_rank];
      f_send.resize(xy_recv.size()/2);

      for (size_t i=0; i<xy_recv.size()/2; i++)
      {
        double *xy = &xy_recv[2*i];

        /* first clip the coordinates */
        double xy_clip [] = {xy[0], xy[1]};
        // bousouf
        if(xy[0]>xmax) xy_clip[0] = xmax;
        if(xy[0]<xmin) xy_clip[0] = xmin;
        if(xy[1]>ymax) xy_clip[1] = ymax;
        if(xy[1]<ymin) xy_clip[1] = ymin;

        // first find the quadrant for the remote points
        p4est_quadrant_t best_match;
#ifdef P4EST_POINT_LOOKUP
        sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
        int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
        std::vector<p4est_quadrant_t> remote_matches;
        int rank_found = qnnn_->hierarchy->find_smallest_quadrant_containing_point(xy_clip, best_match, remote_matches);
#endif
        // make sure that the point belongs to us
        if (rank_found == p4est_->mpirank){ // if we own the point, interpolate
          // get the local index
          p4est_topidx_t tree_idx  = best_match.p.piggy3.which_tree;
          p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
          p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;

          for (short j=0; j<P4EST_CHILDREN; j++)
            f[j] = Fi_p[p4est2petsc[q2n[qu_locidx*P4EST_CHILDREN+j]]];

          // get access to second derivatives only if needed
          if (method_ == quadratic || method_ == quadratic_non_oscillatory)
          {
            for (short j=0; j<P4EST_CHILDREN; ++j)
            {
              p4est_locidx_t node_locidx = p4est2petsc[q2n[qu_locidx*P4EST_CHILDREN+j]];
              fxx[j] = Fxx_p[node_locidx];
              fyy[j] = Fyy_p[node_locidx];
            }
          }

          if (method_ == linear)
            f_send[i] = bilinear_interpolation(p4est_, tree_idx, best_match, f, xy);
          else if (method_ == quadratic)
            f_send[i] = quadratic_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
          else
            f_send[i] = quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);

        } else if ( rank_found != -1 ) {
          // if we don't own the point but its in the ghost layer return 0.
          f_send[i] = 0;
        } else { /* if we dont the own the point, and its not in the ghost layer
                  * this MUST be a bug or mistake so simply throw.
                  */
          std::ostringstream oss;
          oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") was flagged"
                 " as a remote point to either belong to processor "
              <<  p4est_->mpirank << " or be in its ghost layer, both of which"
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
  }

  ierr = PetscLogEventEnd(log_InterpolatingFunction_interpolate, 0, 0, 0, 0); CHKERRXX(ierr);

}

double InterpolatingFunction::operator ()(double x, double y) const {

  PetscErrorCode ierr;

  double xy[] = {x, y};
  double *Fi_p, *Fxx_p, *Fyy_p;

  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);
  if (method_ == linear)
    Fxx_p = Fyy_p = NULL;
  else
  {
    ierr = VecGetArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
  }

  static double f  [P4EST_CHILDREN];
  static double fxx[P4EST_CHILDREN];
  static double fyy[P4EST_CHILDREN];

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t best_match;

  /* first clip the coordinates */
  double xy_clip [] = {x, y};
  if(x>xmax) xy_clip[0] = xmax;
  if(x<xmin) xy_clip[0] = xmin;
  if(y>ymax) xy_clip[1] = ymax;
  if(y<ymin) xy_clip[1] = ymin;

#ifdef P4EST_POINT_LOOKUP
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_, xy_clip, &best_match, remote_matches);
#else
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = qnnn_->hierarchy->find_smallest_quadrant_containing_point(xy_clip, best_match, remote_matches);
#endif

  if (rank_found == p4est_->mpirank) { // local quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
        fxx[j] = Fxx_p[node_locidx];
        fyy[j] = Fyy_p[node_locidx];
      }
    }

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
    }
#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif
    if (method_ == linear)
      return bilinear_interpolation(p4est_, tree_idx, best_match, f, xy);
    else if (method_ == quadratic)
      return quadratic_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
    else
      return quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);

  } else if ( rank_found != -1 ) { // ghost quadrant
    tree_idx = best_match.p.piggy3.which_tree;
    quad_idx = best_match.p.piggy3.local_num + p4est_->local_num_quadrants;

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_p[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

    // get access to second derivatives only if needed
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      for (short j=0; j<P4EST_CHILDREN; ++j)
      {
        p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
        fxx[j] = Fxx_p[node_locidx];
        fyy[j] = Fyy_p[node_locidx];
      }
    }

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
    }

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

    if (method_ == linear)
      return bilinear_interpolation(p4est_, tree_idx, best_match, f, xy);
    else if (method_ == quadratic)
      return quadratic_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
    else
      return quadratic_non_oscillatory_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);

  } else {
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);

    // restore arrays and release remote_maches
    ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
    if (method_ == quadratic || method_ == quadratic_non_oscillatory)
    {
      ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);
    }

#ifdef P4EST_POINT_LOOKUP
    sc_array_destroy(remote_matches);
#endif

    std::ostringstream oss;
    oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") does not belong to "
           "processor " << p4est_->mpirank << ". Found rank = " << rank_found <<
#ifdef P4EST_POINT_LOOKUP
           " and remote_macthes.size = " << remote_matches->elem_count << std::endl;
#else
           " and remote_macthes.size = " << remote_matches.size() << std::endl;
#endif
    throw std::invalid_argument(oss.str());
  }
}

void InterpolatingFunction::send_point_buffers_begin()
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

void InterpolatingFunction::recv_point_buffers_begin()
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

void InterpolatingFunction::compute_second_derivatives()
{
  // Allocate memory for second derivaties
  if (Fxx_ == NULL && Fyy_ == NULL){
    ierr = VecCreateGhost(p4est_, nodes_, &Fxx_); CHKERRXX(ierr);
    ierr = VecDuplicate(Fxx_, &Fyy_); CHKERRXX(ierr);
    local_derivatives = true;
  }

  // Access internal data
  double *Fi_p, *Fxx_p, *Fyy_p;
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fxx_, &Fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fyy_, &Fyy_p); CHKERRXX(ierr);

  // Compute Fxx on local nodes
  for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
    Fxx_p[n] = (*qnnn_)[n].dxx_central(Fi_p);

  // Send ghost values for Fxx
  ierr = VecGhostUpdateBegin(Fxx_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Compute Fyy on local nodes
  for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
    Fyy_p[n] = (*qnnn_)[n].dyy_central(Fi_p);

  // receive the ghost values for Fxx
  ierr = VecGhostUpdateEnd(Fxx_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fxx_, &Fxx_p); CHKERRXX(ierr);

  // Send ghost values for Fyy and receive them
  ierr = VecGhostUpdateBegin(Fyy_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(Fyy_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // restore Fyy array
  ierr = VecRestoreArray(Fyy_, &Fyy_p); CHKERRXX(ierr);

  // restore input_vec_ array
  ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);  
}
