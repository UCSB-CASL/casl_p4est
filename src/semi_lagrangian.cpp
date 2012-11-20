#include "semi_lagrangian.h"

semi_lagrangian::semi_lagrangian(const p4est_t *p4est_,
                                 p4est_point_process_lookup_t process_lookup_,
                                 p4est_point_quadrant_lookup_t quadrant_lookup_)
  : p4est(p4est_), process_lookup(process_lookup_), quadrant_lookup(quadrant_lookup_)
{
  is_processed.reallocate(local_num_nodes);
  is_processed = false;

  departure_point.reallocate(p4est->mpisize);
  departing_node.reallocate(p4est->mpisize);

}

void semi_lagrangian::advance(Vec velx, Vec vely, double dt, Vec phi)
{
  Vec phi_np1;
  ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

  // Loop over all local trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it <= p4est->last_local_tree; ++tr_it)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    // Loop over local quadrants
    for (p4est_locidx_t qu_it = 0; qu_it < tree->quadrants.elem_count; ++qu_it)
    {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);
      p4est_locidx_t quad_locidx = qu_it + tree->quadrants_offset;

      double qh = (double) P4EST_QUADRANT_LEN(quad->level);

      double velx_corner[4], vely_corner[4];
      ierr = VecGetValues(velx, P4EST_CHILDREN, e2n + quad_locidx*P4EST_CHILDREN, &velx_corner[0]); CHKERRXX(ierr);
      ierr = VecGetValues(vely, P4EST_CHILDREN, e2n + quad_locidx*P4EST_CHILDREN, &vely_corner[0]); CHKERRXX(ierr);

      // Loop over 4 corners
      for (unsigned short ci = 0; ci<2; ++ci)
      {
        for (unsigned short cj = 0; cj<2; ++cj)
        {
          p4est_gloidx_t node_gloidx = e2n[quad_locidx*P4EST_CHILDREN + ci + 2*cj];

          if (!is_processed(node_locidx))
          {
            double xd = (double)(quad->x + ci * qh) / (double) P4EST_ROOT_LEN;
            double yd = (double)(quad->y + cj * qh) / (double) P4EST_ROOT_LEN;

            xd -= dt*velx_corner[2*cj + ci];
            yd -= dt*vely_corner[2*cj + ci];

            int departure_point_rank = process_lookup(xd, yd);
            departing_node(departure_point_rank).push(node_locidx);
            departure_point(departure_point_rank).push(xd);
            departure_point(departure_point_rank).push(yd);

            is_processed(node_locidx) = true;
          }
        }
      }
    }
  }
  ArrayV<int> receivers;
  for (int r = 0; r<p4est->mpisize; ++r)
  {
    if (departing_node(r).size()!=0 && r != p4est->mpirank)
    {
      receivers.push(r);
    }
  }
  // TODO: build receivers, num_receivers;
  int *sender=new int[p4est->mpisize];
  int *buffer=(int*) receivers;
  int num_senders;
  sc_notify(buffer,receivers.size(),sender,&num_senders,p4est->mpicomm);

  // Send relavanty information to corresponding processors
  for (int r = 0; r<receivers.size(); ++r)
  {
      double *send_buffer = (double*)departure_point(r);
      int buffer_size = departure_point.size();

      MPI_Send((void*)&buffer_size, 1, MPI_INT, receivers(r), SIZE_TAG, p4est->mpicomm);
      MPI_Send((void*)send_buffer, departure_point(r).size(), MPI_DOUBLE, receivers(r), POINT_TAG, p4est->mpicomm);
  }

  for (int r = 0; r<num_senders; ++r)
  {
    ArrayV<double> received_departure_points;
    int buffer_size;
    MPI_Status st;
    MPI_Recv(&buffer_size, 1, MPI_INT, sender[r], SIZE_TAG, p4est->mpicomm, &st);

    received_departure_points.reallocate(buffer_size);
    double *buffer = (double*)received_departure_points;
    MPI_Recv(buffer, buffer_size, MPI_DOUBLE, sender[r], POINT_TAG, p4est->mpicomm, &st);

    ArrayV<double> phi_interpolated(buffer_size/2); phi_interpolated = 0;
    for (int i = 0; i<buffer_size/2; ++i)
    {
      p4est_topidx_t tree_id;
      double &xd_recv = received_departure_points(2*i);
      double &yd_recv = received_departure_points(2*i+1);
      p4est_quadrant_t *departure_quadrant = quadrant_lookup(xd_recv, yd_recv, &tree_id);

      // TODO: We need to get this thing form PETSc but that requires global numbers of the
      // 4 corners of the current cell.
      double phi_buffer[4] = {0, 0, 0, 0};
      phi_interpolated(i) = bilinear_interpolation(p4est, departure_quadrant, tree_id, phi_buffer, xd_recv, yd_recv);
    }

    double *phi_interpolated_buffer = (double*)phi_interpolated;
    MPI_Send(phi_interpolated_buffer, phi_interpolated.size(), MPI_DOUBLE, sender[r], LVLSET_TAG, p4est->mpicomm);
  }

  for (int r = 0; r<receivers.size(); ++r)
  {
    ArrayV<double> phi_interpolated_received(departing_node.size());
    double *phi_buffer = (double*)phi_interpolated_received;
    MPI_Recv(phi_buffer, phi_interpolated_received.size(), MPI_DOUBLE, receivers[r], LVLSET_TAG, p4est->mpicomm, &st);

    ierr = VecSetValues(phi_np1, phi_interpolated_received.size(), (p4est_locidx_t*)departing_node(r), phi_buffer, INSERT_VALUES); CHKERRXX(ierr);
  }

  // TODO: Now we need to update our local values at the departing nodes of the level-set function based on the local departure points
}
