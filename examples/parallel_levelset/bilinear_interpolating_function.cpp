#include "bilinear_interpolating_function.h"
#include <sc_notify.h>

namespace parallel{
BilinearInterpolatingFunction::BilinearInterpolatingFunction(p4est_t *p4est,
                                                             p4est_nodes_t *nodes,
                                                             p4est_ghost_t *ghost,
                                                             my_p4est_brick_t *myb)
  : p4est_(p4est), nodes_(nodes), ghost_(ghost), myb_(myb),
    is_buffer_prepared(false),
    ghost_senders(p4est->mpisize),
    remote_senders(p4est->mpisize)
{}

double BilinearInterpolatingFunction::operator ()(double x, double y) const {
  double xy[] = {x, y};
  return linear_interpolation(xy);
}

void BilinearInterpolatingFunction::add_point_to_buffer(double x, double y)
{
  // initialize data
  double xy [] = {x, y};
  p4est_quadrant_t *best_match;
  sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

  // find the quadrant
  int rank_found = my_p4est_brick_point_lookup(p4est_, ghost_, myb_,
                                               xy, best_match, remote_matches);

  // check who is going to own the quadrant
  if (rank_found == p4est_->mpirank){ // local quadrant
    local_point_buffer.xy.push_back(x);
    local_point_buffer.xy.push_back(y);
    local_point_buffer.quad.push_back(best_match);

  } else if (rank_found != -1) { // point is found in the ghost layer
    // retrive the correct quadrant from the ghost array
    p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, best_match->p.piggy3.local_num);

    ghost_point_info tmp;
    tmp.xy[0] = x; tmp.xy[1] = y;
    tmp.quad_locidx = q->p.piggy3.local_num;
    tmp.tree_idx    = q->p.piggy3.which_tree;

    ghost_point_send_buffer[rank_found].push_back(tmp);

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

    if (remote_matches->elem_count != 1){
      ostringstream oss;
      oss << "[ERROR]: was expecting only one remote match but recieved " << remote_matches->elem_count
          << " instead. This probably means you should not be doing the interpolation for the point (" << x << "," << y << ")."
             " Please consult the documentation in file " << __FILE__ << " line " << __LINE__ << endl;
      throw invalid_argument(oss.str());
    } else {
      p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(remote_matches, 0);

      remote_xy_send[q->p.piggy1.owner_rank].push_back(x);
      remote_xy_send[q->p.piggy1.owner_rank].push_back(y);
    }
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
    ghost_transfer_map it = ghost_point_send_buffer.begin(), end = ghost_point_send_buffer.end();
    for (;it != end; ++it)
      ghost_recievers.push_back(it->first);

    // notify the other processors
    int num_senders;
    sc_notify(&ghost_recievers[0], ghost_recievers.size(), &ghost_senders[0], &num_senders, p4est_->mpicomm);
    ghost_senders.resize(num_senders);

    // Now that we know all sender/receiver pairs lets do MPI. We do blocking
    for (it = ghost_point_send_buffer.begin(); it != end; ++it){
      std::vector<ghost_point_info>& buff = it->second;
      int msg_size = buff.size()*sizeof(ghost_point_info);

      MPI_Send(&msg_size, 1, MPI_INT, , size_tag, p4est_->mpicomm);
      MPI_Send(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_CHAR, ghost_point_tag, p4est_->mpicomm);
    }

    // Now lets receive the stuff
    for (int i=0; i<num_senders; ++i){
      std::vector<ghost_point_info>& buff = ghost_point_recv_buffer[ghost_senders[i]];

      // Get the size
      int msg_size;
      MPI_Recv(&msg_size, 1, MPI_INT, size_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      // resize the buffer
      buff.resize(msg_size/sizeof(ghost_point_info));

      // Receive the data
      MPI_Recv(reinterpret_cast<char*>(&buff[0]), msg_size, MPI_CHAR, ghost_point_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);
    }
  }

  // 2) Remote buffers
  {
    simple_transfer_map it = remote_xy_send.begin(), end = remote_xy_send.end();
    for (;it != end; ++it)
      remote_recievers.push_back(it->first);

    // notify the other processors
    int num_senders;
    sc_notify(&remote_recievers[0], remote_recievers.size(), &remote_senders[0], &num_senders, p4est_->mpicomm);
    remote_senders.resize(num_senders);

    // Now that we know all sender/receiver pairs lets do MPI. We do blocking
    for (it = remote_xy_send.begin(); it != end; ++it){
      std::vector<double>& buff = it->second;
      int msg_size = buff.size();

      MPI_Send(&msg_size, 1, MPI_INT, , size_tag, p4est_->mpicomm);
      MPI_Send(&buff[0], msg_size, MPI_DOUBLE, remote_xy_tag, p4est_->mpicomm);
    }

    // Now lets receive the stuff
    for (int i=0; i<num_senders; ++i){
      std::vector<double>& buff = remote_xy_recv[remote_senders[i]];

      // Get the size
      int msg_size;
      MPI_Recv(&msg_size, 1, MPI_INT, size_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);

      // resize the buffer
      buff.resize(msg_size);

      // Receive the data
      MPI_Recv(&buff[0], msg_size, MPI_DOUBLE, remote_xy_tag, p4est_->mpicomm, MPI_STATUS_IGNORE);
    }
  }

  is_buffer_prepared = true;
}

double BilinearInterpolatingFunction::linear_interpolation(const double xy[]) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx = 0;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t *quad;

  int found_rank = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                                        xy, &tree_idx, &quad_idx, &quad);
#ifdef CASL_THROWS
  if (p4est_->mpirank != found_rank){
    std::ostringstream oss; oss << "point (" << xy[0] << "," << xy[1] << ")"
                                   "does not belog to processor " << p4est_->mpirank;
    throw std::invalid_argument(oss.str());
  }
#endif

  p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
  quad_idx += tree->quadrants_offset;

  double f [] =
  {
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 0])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 1])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 2])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 3])]
  };

  return bilinear_interpolation(p4est_, tree_idx, quad, f, xy);

}

}
