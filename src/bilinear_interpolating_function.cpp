#include "bilinear_interpolating_function.h"
#include <sc_notify.h>
#include <fstream>

namespace parallel{
BilinearInterpolatingFunction::BilinearInterpolatingFunction(p4est_t *p4est,
                                                             p4est_nodes_t *nodes,
                                                             p4est_ghost_t *ghost,
                                                             my_p4est_brick_t *myb)
  : p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb),
    p4est2petsc(nodes->indep_nodes.elem_count),
    ghost_senders(p4est->mpisize, -1),
    remote_senders(p4est->mpisize, -1),
    is_buffer_prepared(false)
{
  for (int i=0; i<p4est2petsc.size(); ++i)
  {
    if (i<nodes_->offset_owned_indeps)
      p4est2petsc[i] = i + nodes_->num_owned_indeps;
    else if (i<nodes_->num_owned_indeps+nodes_->offset_owned_indeps)
      p4est2petsc[i] = i - nodes_->offset_owned_indeps;
    else
      p4est2petsc[i] = i;
  }
}

double BilinearInterpolatingFunction::operator ()(double x, double y) const {
  double xy[] = {x, y};
  double *Fi_ptr;
  PetscErrorCode ierr = VecGetArray(Fi_, &Fi_ptr); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx = 0;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t *best_match;
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                               xy, &best_match, remote_matches);
#ifdef CASL_THROWS
  if (rank_found != p4est_->mpirank){
    ostringstream oss;
    oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") does not belong to "
           "processor " << p4est_->mpirank << ". Found rank = " << rank_found << " and remote_macthes.size = " << remote_matches->elem_count << endl;
    throw invalid_argument(oss.str());
  }
#endif

  p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, best_match->p.piggy3.which_tree);
  quad_idx = best_match->p.piggy3.local_num + tree->quadrants_offset;

  double f [P4EST_CHILDREN];
  for (short j=0; j<P4EST_CHILDREN; ++j)
    f[j] = Fi_ptr[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

  ierr = VecRestoreArray(Fi_, &Fi_ptr); CHKERRXX(ierr);

  return bilinear_interpolation(p4est_, tree_idx, best_match, f, xy);
}

void BilinearInterpolatingFunction::add_point_to_buffer(p4est_locidx_t node_locidx, double x, double y)
{
  // initialize data
  double xy [] = {x, y};
  p4est_quadrant_t *best_match;
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

  // find the quadrant
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                               xy, &best_match, remote_matches);

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank){ // local quadrant
    local_point_buffer.xy.push_back(x);
    local_point_buffer.xy.push_back(y);
    local_point_buffer.quad.push_back(best_match);
    local_point_buffer.node_locidx.push_back(node_locidx);

  } else if (remote_matches->elem_count == 0) { // point is found in the ghost layer
    // retrive the correct quadrant from the ghost array
    p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, best_match->p.piggy3.local_num);

    ghost_point_info tmp;
    tmp.xy[0] = x; tmp.xy[1] = y;
    tmp.quad_locidx = q->p.piggy3.local_num;
    tmp.tree_idx    = q->p.piggy3.which_tree;

    ghost_send_buffer[rank_found].push_back(tmp);
    ghost_node_index[rank_found].push_back(node_locidx);

  } else { // quadrant belongs to a processor that is not included in the ghost list
    /*
     * OK. This makes things complicated and nasty. In general you should
     * consider the worst case scenario which is a point that matches to 4
     * remote quadrant. This would correspond to an actual node that is on the
     * boundary between 4 processors!
     *
     * This is a nasty case to consider since you have to send the information
     * to all 4 processors. Then, since they share the point, depending on the
     * size of quadrants (either all 4 the same size or all 4 different)
     * different number of processors will be eligible to do the interpolation.
     * Problem is, the current processor wont know which one is responsible to
     * do the interpolation so it sends the message to all 4. From here two
     * approaches are possible:
     *
     * 1) All 4 will respond back to the processor. In this case, 1, 2, 3, or 4
     * values will be correct depending on the how many processors could do the
     * interpolation and the rest will be junk. To differentiate, we could ask
     * processors that cannot do the interpolation to return 'nan' or 'inf' and
     * then sort things out on the root (processor who asked for the info in the
     * first place). Problem is, what if there is something else wrong in the
     * data? What if when get 'inf' or 'nan' due to some other invalid operation
     * on the responsible processor?
     *
     * 2) Second approach is to only ask the processor that does the correct
     * interpolation to respond to the message. In this case the number of bytes
     * that root will recieve will be different than the number we sent. So the
     * root has to create a mapping to handle cases from information is sent but
     * nothing is recieved so that it knows what indecies the recieved info will
     * belong to!
     *
     * Both cases are equally ugly with the first one being dangerous and the
     * second one being hard to implement and probably inefficient. Plus such a
     * case will most probably happen if the user is dumb enough to ask for the
     * interpolated at the boundary of 4 processors because otherwise the odds
     * of something like this happening is really really really small!
     *
     * Sooooo, what do we do here? Well only consider the case that there will
     * be just one remote match at most since this is the most relavant case. If
     * this condition is not satisfied, for the moment at least, we simply throw
     * an exception and ask the user to read this long explanation!
     */

#ifdef CASL_THROWS
//    {
//      vector<int> r(remote_matches->elem_count);
//      for (int i=0; i<r.size(); i++){
//        p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(remote_matches, i);
//        r[i] = q->p.piggy1.owner_rank;
//      }

//      for (int i=0; i<r.size()-1; i++){
//        if (r[i] != r[i+1]){
//          ostringstream oss;
//          oss << "[ERROR]:"
//              << " Was expecting all remote quadrants belong to the same processors, but they dont. "
//              << " This probably means you should not be doing the interpolation for the point (" << x << "," << y << ")."
//                 " Please consult the documentation in file " << __FILE__ << ". This error is generated from line " << __LINE__ << endl;
//          throw invalid_argument(oss.str());
//        }
//      }
//    }
#endif

    p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(remote_matches, 0);
    int remote_rank = q->p.piggy1.owner_rank;

    remote_send_buffer[remote_rank].push_back(x);
    remote_send_buffer[remote_rank].push_back(y);
    remote_node_index[remote_rank].push_back(node_locidx);
  }

  // set the flag to false so the prepare buffer method will be called
  is_buffer_prepared = false;
}

void BilinearInterpolatingFunction::prepare_buffer()
{
  /*
   * We will do this is two steps:
   * 1) Send/Recv the ghost buffers
   * 2) Send/Recv the remote buffers
   */

  // 1) Ghost buffers
  {
    ghost_transfer_map::iterator it = ghost_send_buffer.begin(), end = ghost_send_buffer.end();
    for (;it != end; ++it)
      ghost_recievers.push_back(it->first);

    // notify the other processors
    int num_senders;
    sc_notify(&ghost_recievers[0], ghost_recievers.size(), &ghost_senders[0], &num_senders, p4est_->mpicomm);
    ghost_senders.resize(num_senders);

    // Now that we know all sender/receiver pairs lets do MPI. We do blocking
    for (it = ghost_send_buffer.begin(); it != end; ++it){
      std::vector<ghost_point_info>& buff = it->second;

      int msg_size = buff.size()*sizeof(ghost_point_info);
      MPI_Send(&msg_size, 1, MPI_INT, it->first , size_tag, p4est_->mpicomm);
      MPI_Send(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_BYTE, it->first, ghost_point_tag, p4est_->mpicomm);
    }

    // Now lets receive the stuff
    for (int i=0; i<num_senders; ++i){
      std::vector<ghost_point_info>& buff = ghost_recv_buffer[ghost_senders[i]];

      // Get the size
      int msg_size;
      MPI_Recv(&msg_size, 1, MPI_INT, ghost_senders[i], size_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      /*
       * resize the buffer -- remember to rescale it with the size of
       * ghost_point_info since the data arrives in serialized
       */
      buff.resize(msg_size/sizeof(ghost_point_info));

      // Receive the data
      MPI_Recv(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_BYTE, ghost_senders[i], ghost_point_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);
    }
  }

  // 2) Remote buffers
  {
    remote_transfer_map::iterator it = remote_send_buffer.begin(), end = remote_send_buffer.end();
    for (;it != end; ++it)
      remote_recievers.push_back(it->first);

    // notify the other processors
    int num_senders;
    sc_notify(&remote_recievers[0], remote_recievers.size(), &remote_senders[0], &num_senders, p4est_->mpicomm);
    remote_senders.resize(num_senders);

    // Now that we know all sender/receiver pairs lets do MPI. We do blocking
    for (it = remote_send_buffer.begin(); it != end; ++it){

      std::vector<double>& buff = it->second;
      int msg_size = buff.size();

      cout << "[" << p4est_->mpirank << "] remote_rank = " << it->first << endl;
      cout << "points = ";

      ostringstream oss;
      oss << "points_dbg_" << p4est_->mpirank << ".py";
      ofstream py(oss.str().c_str());
      py << "try: paraview.simple" << endl;
      py << "except: from paraview.simple import * " << endl;
      py << "paraview.simple._DisableFirstRenderCameraReset() " << endl;
      py << "RenderView1 = CreateRenderView()" << endl;
      for (int i=0; i<buff.size()/2; i++){
        cout << "(" << buff[2*i] << "," << buff[2*i+1] << "), ";

        py << "PointSource" << i << " = PointSource(guiName=\"PointSource" << i << "\", Radius = 0.0, Center=[" << buff[2*i] << "," << buff[2*i+1] << ",0]"
               ", NumberOfPoints = 1)" << endl;
        py << "SetActiveSource(PointSource" << i << ")" << endl;
        py << "DataRepresentation" << i << " = Show()" << endl;
      }
      py.close();
      cout << endl;

      MPI_Send(&msg_size, 1, MPI_INT, it->first, size_tag, p4est_->mpicomm);
      MPI_Send(&buff[0], msg_size, MPI_DOUBLE, it->first, remote_point_tag, p4est_->mpicomm);
    }

    // Now lets receive the stuff
    for (int i=0; i<num_senders; ++i){
      std::vector<double>& buff = remote_recv_buffer[remote_senders[i]];

      // Get the size
      int msg_size;
      MPI_Recv(&msg_size, 1, MPI_INT, remote_senders[i], size_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      // resize the buffer
      buff.resize(msg_size);

      // Receive the data
      MPI_Recv(&buff[0], msg_size, MPI_DOUBLE, remote_senders[i], remote_point_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);
    }
  }

  is_buffer_prepared = true;
}

void BilinearInterpolatingFunction::interpolate(Vec& Fo)
{
  if (!is_buffer_prepared)
    prepare_buffer();

  PetscErrorCode ierr;

  // Get a pointer to the data
  double *Fi_ptr, *Fo_ptr;
  ierr = VecGetArray(Fi_, &Fi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Fo,  &Fo_ptr); CHKERRXX(ierr);

  // Get access to the node numbering of all quadrants
  p4est_locidx_t *q2n = nodes_->local_nodes;
  double f[P4EST_CHILDREN];

  // Do the interpolation for local points
  for (size_t i=0; i<local_point_buffer.size(); ++i)
  {
    double *xy = &local_point_buffer.xy[2*i];
    p4est_quadrant_t *quad = local_point_buffer.quad[i];
    p4est_topidx_t tree_idx = quad->p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    p4est_locidx_t quad_idx = quad->p.piggy3.local_num + tree->quadrants_offset;
    p4est_locidx_t node_idx = local_point_buffer.node_locidx[i];

    for (short j=0; j<P4EST_CHILDREN; ++j)
      f[j] = Fi_ptr[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

    Fo_ptr[node_idx] = bilinear_interpolation(p4est_, tree_idx, quad, f, xy);
  }

  // Do interpolation for ghost points and the send the results
  {
    ghost_transfer_map::iterator it = ghost_recv_buffer.begin(),
        end = ghost_recv_buffer.end();

    for (; it != end; ++it)
    {
      std::vector<ghost_point_info>& recv_info = it->second;
      std::vector<double> f_send;

      f_send.resize(recv_info.size());
      for (size_t i=0; i<recv_info.size(); ++i){
        double *xy = &recv_info[i].xy[0];
        p4est_topidx_t tree_idx = recv_info[i].tree_idx;
        p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, recv_info[i].quad_locidx);
        p4est_locidx_t quad_idx = recv_info[i].quad_locidx + tree->quadrants_offset;

        for (short j=0; j<P4EST_CHILDREN; ++j)
          f[j] = Fi_ptr[p4est2petsc[q2n[quad_idx*P4EST_CHILDREN+j]]];

        f_send[i] = bilinear_interpolation(p4est_, tree_idx, quad, f, xy);
      }

      MPI_Send(&f_send[0], f_send.size(), MPI_DOUBLE, it->first, ghost_data_tag, p4est_->mpicomm);
    }
  }

  // Receive the interpolated ghost data and put it in the correct location
  {
    for (size_t i = 0; i < ghost_recievers.size(); ++i)
    {
      int recv_rank = ghost_recievers[i];
      std::vector<double> f_recv(ghost_recv_buffer[recv_rank].size());
      std::vector<p4est_locidx_t>& node_idx = ghost_node_index[recv_rank];
      MPI_Recv(&f_recv[0], f_recv.size(), MPI_DOUBLE, recv_rank, ghost_data_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      for (size_t j=0; j<f_recv.size(); j++)
        Fo_ptr[node_idx[j]] = f_recv[j];
    }
  }

  // Do interpolation for remote points
  {
    remote_transfer_map::iterator it = remote_recv_buffer.begin(),
        end = remote_recv_buffer.end();

    for (; it != end; ++it )
    {
      std::vector<double>& xy_recv = it->second;
      std::vector<double> f_send(xy_recv.size()/2);

      for (size_t i=0; i<xy_recv.size()/2; i++)
      {
        double *xy = &xy_recv[2*i];

        // first find the quadrant for the remote points
        p4est_quadrant_t *best_match;
        sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
        int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                                     xy, &best_match, remote_matches);

        // make sure that the point belongs to us
        if (rank_found != p4est_->mpirank){
          ostringstream oss;
          oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") does not belong to "
                 "processor " << p4est_->mpirank << ". Found rank = " << rank_found << " and remote_macthes.size = " << remote_matches->elem_count << endl;
          throw runtime_error(oss.str());
        }
        sc_array_destroy(remote_matches);

        // get the local index
        p4est_topidx_t tree_idx  = best_match->p.piggy3.which_tree;
        p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
        p4est_locidx_t qu_locidx = best_match->p.piggy3.local_num + tree->quadrants_offset;

        for (short j=0; j<P4EST_CHILDREN; j++)
          f[j] = Fi_ptr[p4est2petsc[q2n[qu_locidx*P4EST_CHILDREN+j]]];

        f_send[i] = bilinear_interpolation(p4est_, tree_idx, best_match, f, xy);
      }

      // send the buffer
      MPI_Send(&f_send[0], f_send.size(), MPI_DOUBLE, it->first, remote_data_tag, p4est_->mpicomm);
    }

    // Receive the stuff and put them in the correct position.
    for (size_t i=0; i<remote_recievers.size(); ++i)
    {
      int recv_rank = remote_recievers[i];
      std::vector<double> f_recv(remote_send_buffer[recv_rank].size()/2);
      std::vector<p4est_locidx_t>& node_idx = remote_node_index[recv_rank];
      MPI_Recv(&f_recv[0], f_recv.size(), MPI_DOUBLE, recv_rank, remote_data_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      for (size_t j=0; j<f_recv.size(); j++)
        Fo_ptr[node_idx[j]] = f_recv[j];
    }
  }

  // Restore the pointer
  ierr = VecRestoreArray(Fi_, &Fi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Fo,  &Fo_ptr); CHKERRXX(ierr);
}

void BilinearInterpolatingFunction::update_vector(Vec &Fi)
{
  Fi_ = Fi;
}

void BilinearInterpolatingFunction::clear_buffer()
{
  local_point_buffer.xy.clear();
  local_point_buffer.quad.clear();
  local_point_buffer.node_locidx.clear();

  ghost_send_buffer.clear();
  ghost_recv_buffer.clear();
  ghost_node_index.clear();
  ghost_recievers.clear();

  remote_send_buffer.clear();
  remote_recv_buffer.clear();
  remote_node_index.clear();
  remote_recievers.clear();

  p4est2petsc.clear();

  is_buffer_prepared = false;
}

void BilinearInterpolatingFunction::update_grid(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost)
{
  clear_buffer();

  p4est_ = p4est;
  nodes_ = nodes;
  ghost_ = ghost;

  ghost_senders.resize(p4est_->mpisize, -1);
  remote_senders.resize(p4est_->mpisize, -1);

  p4est2petsc.resize(nodes_->indep_nodes.elem_count);
  for (int i=0; i<p4est2petsc.size(); ++i)
  {
    if (i<nodes_->offset_owned_indeps)
      p4est2petsc[i] = i + nodes_->num_owned_indeps;
    else if (i<nodes_->num_owned_indeps+nodes_->offset_owned_indeps)
      p4est2petsc[i] = i - nodes_->offset_owned_indeps;
    else
      p4est2petsc[i] = i;
  }
}
}
