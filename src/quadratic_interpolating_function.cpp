#include "quadratic_interpolating_function.h"
#include "petsc_compatibility.h"
#include <sc_notify.h>
#include <src/my_p4est_vtk.h>

#include <fstream>
#include <set>

QuadraticInterpolatingFunction::QuadraticInterpolatingFunction(p4est_t *p4est,
                                                               p4est_nodes_t *nodes,
                                                               p4est_ghost_t *ghost,
                                                               my_p4est_node_neighbors_t *qnnn,
                                                               my_p4est_brick_t *myb)
  : inter_type_(nonoscilatory_interpolation),
    p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb), qnnn_(qnnn),
    p4est2petsc(nodes->indep_nodes.elem_count),
    ghost_senders(p4est->mpisize, -1),
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

  // Allocate memory for second derivaties
  ierr = VecCreateGhost(p4est_, nodes_, &Fxx); CHKERRXX(ierr);
  ierr = VecDuplicate(Fxx, &Fyy); CHKERRXX(ierr);

}

void QuadraticInterpolatingFunction::add_point_to_buffer(p4est_locidx_t node_locidx, double x, double y)
{
  // initialize data
  double xy [] = {x, y};
  p4est_quadrant_t best_match;
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

  // find the quadrant
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                               xy, &best_match, remote_matches);

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank){ // local quadrant
    local_point_buffer.xy.push_back(xy[0]);
    local_point_buffer.xy.push_back(xy[1]);
    local_point_buffer.quad.push_back(best_match);
    local_point_buffer.node_locidx.push_back(node_locidx);

  } else if (remote_matches->elem_count == 0) { /* point is found in the ghost
                                                 * layer
                                                 */
    size_t pos = (size_t)(best_match.p.piggy3.local_num);
    p4est_quadrant *q = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, pos);

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

    ghost_point_info tmp;
    tmp.xy[0] = xy[0]; tmp.xy[1] = xy[1];
    tmp.quad_locidx = q->p.piggy3.local_num;
    tmp.tree_idx    = q->p.piggy3.which_tree;

    ghost_send_buffer[rank_found].push_back(tmp);
    ghost_node_index[rank_found].push_back(node_locidx);

  } else { /* quadrant belongs to a processor that is not included in the ghost
            * layer
            */
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
    for (size_t i=0; i<remote_matches->elem_count; i++){
      p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(remote_matches, i);
      remote_ranks.insert(q->p.piggy1.owner_rank);
    }

    std::set<int>::const_iterator it = remote_ranks.begin(),
        end = remote_ranks.end();
    for(; it != end; ++it){
      int r = *it;
      remote_send_buffer[r].push_back(xy[0]);
      remote_send_buffer[r].push_back(xy[1]);
      remote_node_index[r].push_back(node_locidx);
    }
  }

  sc_array_destroy(remote_matches);

  // set the flag to false so the prepare buffer method will be called
  is_buffer_prepared = false;
}

void QuadraticInterpolatingFunction::update_grid(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t *qnnn)
{
  clear_buffer();

  p4est_ = p4est;
  nodes_ = nodes;
  ghost_ = ghost;
  qnnn_  = qnnn;

  ierr = VecCreateGhost(p4est_, nodes_, &Fxx); CHKERRXX(ierr);
  ierr = VecDuplicate(Fxx, &Fyy); CHKERRXX(ierr);

  ghost_senders.resize(p4est_->mpisize, -1);
  remote_senders.resize(p4est_->mpisize, -1);

  p4est2petsc.resize(nodes_->indep_nodes.elem_count);
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
}

void QuadraticInterpolatingFunction::set_input_vector(Vec &input_vec)
{
  input_vec_ = input_vec;
  compute_second_derivatives();
}

void QuadraticInterpolatingFunction::set_interpolation_type(interpolation_type type)
{
  inter_type_ = type;
}

void QuadraticInterpolatingFunction::interpolate(Vec& output_vec)
{
  // begin sending point buffers
  if (!is_buffer_prepared)
    send_point_buffers_begin();

  // Get a pointer to the data
  double *Fi_p,  *Fo_p;
  double *Fxx_p, *Fyy_p;
  ierr = VecGetArray(input_vec_, &Fi_p ); CHKERRXX(ierr);
  ierr = VecGetArray(output_vec, &Fo_p ); CHKERRXX(ierr);  
  ierr = VecGetArray(Fxx,        &Fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fyy,        &Fyy_p); CHKERRXX(ierr);

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

    for (short j=0; j<P4EST_CHILDREN; ++j){
      p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
      f[j]   = Fi_p [node_locidx];
      fxx[j] = Fxx_p[node_locidx];
      fyy[j] = Fyy_p[node_locidx];
    }

    if(inter_type_ == regular_interpolation)
      Fo_p[node_idx] = quadratic_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
    else if (inter_type_ == nonoscilatory_interpolation)
      Fo_p[node_idx] = quadratic_interpolation_nonoscilatory(p4est_, tree_idx, quad, f, fxx, fyy, xy);
    else {throw; /* this never happens */}
  }

  // begin recieving point buffers
  if (!is_buffer_prepared)
    recv_point_buffers_begin();

  // begin data send/recv for ghost and remote
  typedef std::map<int, std::vector<double> > data_transfer_map;
  data_transfer_map f_ghost_send, f_remote_send;
  std::vector<MPI_Request> ghost_data_send_req(ghost_senders.size());
  std::vector<MPI_Request> remote_data_send_req(remote_senders.size());

  // Do interpolation for ghost points and the send the results
  {
    ghost_transfer_map::iterator it = ghost_recv_buffer.begin(),
        end = ghost_recv_buffer.end();

    int req_counter = 0;
    for (; it != end; ++it, ++req_counter)
    {
      int send_rank = it->first;

      // make sure we received the buffers for this process before accesing data
      if(!is_buffer_prepared)
        MPI_Wait(&ghost_recv_req[req_counter], MPI_STATUS_IGNORE);

      std::vector<ghost_point_info>& recv_info = it->second;
      std::vector<double>& f_send = f_ghost_send[send_rank];
      f_send.resize(recv_info.size());

      for (size_t i=0; i<recv_info.size(); ++i){
        double *xy = &recv_info[i].xy[0];
        p4est_topidx_t tree_idx = recv_info[i].tree_idx;
        p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
        p4est_quadrant_t &quad = *(p4est_quadrant_t*)sc_array_index(&tree->quadrants, recv_info[i].quad_locidx-tree->quadrants_offset);
        p4est_locidx_t quad_idx = recv_info[i].quad_locidx;

        for (short j=0; j<P4EST_CHILDREN; ++j){
          p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
          f[j]   = Fi_p [node_locidx];
          fxx[j] = Fxx_p[node_locidx];
          fyy[j] = Fyy_p[node_locidx];
        }

        if(inter_type_ == regular_interpolation)
          f_send[i] = quadratic_interpolation(p4est_, tree_idx, quad, f, fxx, fyy, xy);
        else if (inter_type_ == nonoscilatory_interpolation)
          f_send[i] = quadratic_interpolation_nonoscilatory(p4est_, tree_idx, quad, f, fxx, fyy, xy);
        else {throw; /* this never happens */}
      }

      MPI_Isend(&f_send[0], f_send.size(), MPI_DOUBLE, send_rank, ghost_data_tag, p4est_->mpicomm, &ghost_data_send_req[req_counter]);
    }
  }

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

        // first find the quadrant for the remote points
        p4est_quadrant_t best_match;
        sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
        int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                                     xy, &best_match, remote_matches);

        // make sure that the point belongs to us
        if (rank_found == p4est_->mpirank){ // if we own the point, interpolate
          // get the local index
          p4est_topidx_t tree_idx  = best_match.p.piggy3.which_tree;
          p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
          p4est_locidx_t qu_locidx = best_match.p.piggy3.local_num + tree->quadrants_offset;

          for (short j=0; j<P4EST_CHILDREN; ++j){
            p4est_locidx_t node_locidx = p4est2petsc[q2n[qu_locidx*P4EST_CHILDREN+j]];
            f[j]   = Fi_p [node_locidx];
            fxx[j] = Fxx_p[node_locidx];
            fyy[j] = Fyy_p[node_locidx];
          }

          if(inter_type_ == regular_interpolation)
            f_send[i] = quadratic_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
          else if (inter_type_ == nonoscilatory_interpolation)
            f_send[i] = quadratic_interpolation_nonoscilatory(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
          else {throw; /* this never happens */}

        } else if (remote_matches->elem_count == 0){
          // if we don't own the point but its in the ghost layer return 0.
          f_send[i] = 0;
        } else { /* if we dont the own teh point, and its not in the ghost layer
                  * this MUST be a bug or mistake so simply throw.
                  */
          std::ostringstream oss;
          oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") was flagged"
                 " as a remote point to either belong to processor "
              <<  p4est_->mpirank << " or be in its ghost layer, both of which"
                  " have failed. Found rank is = " << rank_found
              << " and remote_macthes->elem_count = "
              << remote_matches->elem_count << ". This is most certainly a bug."
              << std::endl;
          throw std::runtime_error(oss.str());
        }
        sc_array_destroy(remote_matches);
      }

      // send the buffer
      MPI_Isend(&f_send[0], f_send.size(), MPI_DOUBLE, send_rank, remote_data_tag, p4est_->mpicomm, &remote_data_send_req[req_counter]);
    }
  }

  // Receive the interpolated ghost data and put it in the correct location
  {
    for (size_t i = 0; i < ghost_receivers.size(); ++i)
    {
      int recv_rank = ghost_receivers[i];
      std::vector<double> f_recv(ghost_send_buffer[recv_rank].size());
      std::vector<p4est_locidx_t>& node_idx = ghost_node_index[recv_rank];
      MPI_Recv(&f_recv[0], f_recv.size(), MPI_DOUBLE, recv_rank, ghost_data_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      for (size_t j=0; j<f_recv.size(); j++)
        Fo_p[node_idx[j]] = f_recv[j];
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
    MPI_Waitall(ghost_send_req.size() , &ghost_send_req[0] , MPI_STATUSES_IGNORE);
    MPI_Waitall(remote_send_req.size(), &remote_send_req[0], MPI_STATUSES_IGNORE);

    /* set the buffer flag to true so that we wont wait for other interpolations
     * if the grid is unchanged
     */
    is_buffer_prepared = true;
  }

  // wait for all data send buffers to finish
  MPI_Waitall(ghost_data_send_req.size() , &ghost_data_send_req[0] , MPI_STATUSES_IGNORE);
  MPI_Waitall(remote_data_send_req.size(), &remote_data_send_req[0], MPI_STATUSES_IGNORE);

  // Restore the pointer
  ierr = VecRestoreArray(input_vec_, &Fi_p ); CHKERRXX(ierr);
  ierr = VecRestoreArray(output_vec, &Fo_p ); CHKERRXX(ierr);  
  ierr = VecRestoreArray(Fxx,        &Fxx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fyy,        &Fyy_p); CHKERRXX(ierr);
}

double QuadraticInterpolatingFunction::operator ()(double x, double y) const {
  double xy[] = {x, y};

  // Access to internal PETSc data
  PetscErrorCode ierr;
  double *Fi_p, *Fxx_p, *Fyy_p;
  ierr = VecGetArray(input_vec_, &Fi_p ); CHKERRXX(ierr);
  ierr = VecGetArray(Fxx,        &Fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fyy,        &Fyy_p); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t best_match;
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                               xy, &best_match, remote_matches);
#ifdef CASL_THROWS
  if (rank_found != p4est_->mpirank){
    std::ostringstream oss;
    oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") does not belong to "
           "processor " << p4est_->mpirank << ". Found rank = " << rank_found <<
           " and remote_macthes.size = " << remote_matches->elem_count << std::endl;
    throw std::invalid_argument(oss.str());
  }
#endif

  tree_idx = best_match.p.piggy3.which_tree;
  p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
  quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;

  double f  [P4EST_CHILDREN];
  double fxx[P4EST_CHILDREN];
  double fyy[P4EST_CHILDREN];
  for (short j=0; j<P4EST_CHILDREN; ++j){
    p4est_locidx_t node_locidx = p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]];
    f[j]   = Fi_p[node_locidx];
    fxx[j] = Fyy_p[node_locidx];
    fyy[j] = Fyy_p[node_locidx];
  }

  ierr = VecRestoreArray(input_vec_, &Fi_p ); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fxx,        &Fxx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fyy,        &Fyy_p); CHKERRXX(ierr);

  if (inter_type_ == regular_interpolation)
    return quadratic_interpolation(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
  else if (inter_type_ == nonoscilatory_interpolation)
    return quadratic_interpolation_nonoscilatory(p4est_, tree_idx, best_match, f, fxx, fyy, xy);
  else {throw;/* this never happens -- just to prevent warnings */}
}


void QuadraticInterpolatingFunction::send_point_buffers_begin()
{
  /*
   * We will do this is two steps:
   * 1) Send the ghost buffers
   * 2) Send the remote buffers
   */

  // 1) Ghost buffers
  {
    int req_counter = 0;

    ghost_transfer_map::iterator it = ghost_send_buffer.begin(), end = ghost_send_buffer.end();
    for (;it != end; ++it)
      ghost_receivers.push_back(it->first);

    // notify the other processors
    int num_senders;
    sc_notify(&ghost_receivers[0], ghost_receivers.size(), &ghost_senders[0], &num_senders, p4est_->mpicomm);
    ghost_senders.resize(num_senders);

    // Allocate enough requests slots
    ghost_send_req.resize(ghost_receivers.size());

    // Now that we know all sender/receiver pairs lets do MPI. We do blocking
    for (it = ghost_send_buffer.begin(); it != end; ++it, ++req_counter){
      std::vector<ghost_point_info>& buff = it->second;

      int msg_size = buff.size()*sizeof(ghost_point_info);
      MPI_Isend(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_BYTE, it->first, ghost_point_tag, p4est_->mpicomm, &ghost_send_req[req_counter]);
    }
  }

  // 2) Remote buffers
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
}

void QuadraticInterpolatingFunction::recv_point_buffers_begin()
{
  /*
   * We will do this is two steps:
   * 1) Recv the ghost buffers
   * 2) Recv the remote buffers
   */

  // 1) Ghost buffers
  {
    // Allocate enough requests slots
    ghost_recv_req.resize(ghost_senders.size());

    // Now lets receive the stuff
    for (size_t i=0; i<ghost_senders.size(); ++i){
      std::vector<ghost_point_info>& buff = ghost_recv_buffer[ghost_senders[i]];

      /* Get the message size and resize the recv buffer
       * Note: We are receiving data as MPI_BYTE and thus we require to devide
       * by sizeof(ghost_point_info) to get the number of elements
       */
      int msg_size;
      MPI_Status st;
      MPI_Probe(ghost_senders[i], ghost_point_tag, p4est_->mpicomm, &st);
      MPI_Get_count(&st, MPI_BYTE, &msg_size);
      buff.resize(msg_size/sizeof(ghost_point_info));

      // Receive the data -- nonblocking
      MPI_Irecv(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_BYTE, ghost_senders[i], ghost_point_tag, p4est_->mpicomm, &ghost_recv_req[i]);
    }
  }

  // 2) Remote buffers
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
}

void QuadraticInterpolatingFunction::clear_buffer()
{
  local_point_buffer.xy.clear();
  local_point_buffer.quad.clear();
  local_point_buffer.node_locidx.clear();

  ghost_send_buffer.clear();
  ghost_recv_buffer.clear();
  ghost_node_index.clear();
  ghost_receivers.clear();

  remote_send_buffer.clear();
  remote_recv_buffer.clear();
  remote_node_index.clear();
  remote_receivers.clear();

  p4est2petsc.clear();

  ierr = VecDestroy(Fxx); CHKERRXX(ierr);
  ierr = VecDestroy(Fyy); CHKERRXX(ierr);

  is_buffer_prepared = false;
}

void QuadraticInterpolatingFunction::compute_second_derivatives()
{
  // Access internal data
  double *Fi_p, *Fxx_p, *Fyy_p;
  ierr = VecGetArray(input_vec_, &Fi_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fxx, &Fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(Fyy, &Fyy_p); CHKERRXX(ierr);

  // Compute Fxx on local nodes
  for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
    Fxx_p[n] = (*qnnn_)[n].dxx_central(Fi_p);

  // Send ghost values for Fxx
  ierr = VecGhostUpdateBegin(Fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Compute Fyy on local nodes
  for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; ++n)
    Fyy_p[n] = (*qnnn_)[n].dyy_central(Fi_p);

  // receive the ghost values for Fxx
  ierr = VecGhostUpdateEnd(Fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fxx, &Fxx_p); CHKERRXX(ierr);

  // Send ghost values for Fyy and receive them
  ierr = VecGhostUpdateBegin(Fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);  
  ierr = VecGhostUpdateEnd(Fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // restore Fyy array
  ierr = VecRestoreArray(Fyy, &Fyy_p); CHKERRXX(ierr);

  // restore input_vec_ array
  ierr = VecRestoreArray(input_vec_, &Fi_p); CHKERRXX(ierr);
}