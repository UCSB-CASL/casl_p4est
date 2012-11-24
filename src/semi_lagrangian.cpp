#include "semi_lagrangian.h"

semi_lagrangian::semi_lagrangian(p4est_t *p4est_, my_p4est_nodes_t *nodes_)
  : p4est(p4est_), nodes(nodes_)
{
  is_processed.reallocate(nodes->num_owned_indeps);
  is_processed = false;

  departure_point.reallocate(p4est->mpisize);
  departing_node.reallocate(p4est->mpisize);

  // TODO: e2n returns the local index of vertices of a cell.
  e2n = nodes->local_nodes;

}

void semi_lagrangian::advect(Vec velx, Vec vely, double dt, Vec& phi)
{
  Vec phi_np1;
  ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

  double *velx_val, *vely_val, *phi_val;

  ierr = VecGetArray(velx, &velx_val); CHKERRXX(ierr);
  ierr = VecGetArray(vely, &vely_val); CHKERRXX(ierr);
  ierr = VecGetArray(phi,  &phi_val);  CHKERRXX(ierr);

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double         *v2q = p4est->connectivity->vertices;

  p4est_topidx_t p4est_mm = t2v[0];
  p4est_topidx_t p4est_pp = t2v[p4est->connectivity->num_trees * P4EST_CHILDREN - 1];

  double domain_xmin = v2q[3*p4est_mm + 0];
  double domain_ymin = v2q[3*p4est_mm + 1];
  double domain_xmax = v2q[3*p4est_pp + 0];
  double domain_ymax = v2q[3*p4est_pp + 1];

  // Loop over all local trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it <= p4est->last_local_tree; ++tr_it)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    p4est_topidx_t v_mm = t2v[ tr_it   *P4EST_CHILDREN + 0];
    p4est_topidx_t v_pp = t2v[(tr_it+1)*P4EST_CHILDREN - 1];

    double tr_xmin = v2q[3*v_mm + 0];
    double tr_ymin = v2q[3*v_mm + 1];
    double tr_xmax = v2q[3*v_pp + 0];
    double tr_ymax = v2q[3*v_pp + 1];

    double tr_lx   = tr_xmax - tr_xmin;
    double tr_ly   = tr_ymax - tr_ymin;

    // Loop over local quadrants
    for (p4est_locidx_t qu_it = 0; qu_it < tree->quadrants.elem_count; ++qu_it)
    {
      p4est_locidx_t quad_locidx = qu_it + tree->quadrants_offset;

      // Loop over 4 corners
      for (unsigned short cj = 0; cj<2; ++cj)
      {
        for (unsigned short ci = 0; ci<2; ++ci)
        {
          p4est_locidx_t p4est_node_locidx = e2n[quad_locidx*P4EST_CHILDREN + 2*cj + ci];
          p4est_locidx_t petsc_node_locidx = p4est2petsc_local_numbering(nodes, p4est_node_locidx);

          if (petsc_node_locidx >= nodes->num_owned_indeps)
            continue;

          if (!is_processed(petsc_node_locidx))
          {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, p4est_node_locidx);
            double xy [] =
            {
              ((double)(node->x) / (double)P4EST_ROOT_LEN) * tr_lx + tr_xmin,
              ((double)(node->y) / (double)P4EST_ROOT_LEN) * tr_ly + tr_ymin
            };

            xy[0] -= dt*velx_val[petsc_node_locidx];
            xy[1] -= dt*vely_val[petsc_node_locidx];

            // clamp on the walls
            if (xy[0] < domain_xmin + EPS) xy[0] = domain_xmin + EPS;
            if (xy[0] > domain_xmax - EPS) xy[0] = domain_xmax - EPS;
            if (xy[1] < domain_ymin + EPS) xy[1] = domain_ymin + EPS;
            if (xy[1] > domain_ymax - EPS) xy[1] = domain_ymax - EPS;

            p4est_topidx_t which_tree = tr_it;
            int departure_point_rank = my_p4est_brick_point_lookup(p4est, xy, &which_tree, NULL, NULL);

            departing_node(departure_point_rank).push(petsc_node_locidx);
            departure_point(departure_point_rank).push(xy[0]);
            departure_point(departure_point_rank).push(xy[1]);

            is_processed(petsc_node_locidx) = true;
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

  ArrayV<int> senders(p4est->mpisize);
  int *sender_buffer   = (int*) senders;
  int *receiver_buffer = (int*) receivers;
  int num_senders;
  sc_notify(receiver_buffer, receivers.size(), sender_buffer, &num_senders, p4est->mpicomm);

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
    MPI_Recv(&buffer_size, 1, MPI_INT, senders(r), SIZE_TAG, p4est->mpicomm, &st);

    received_departure_points.reallocate(buffer_size);
    double *buffer = (double*)received_departure_points;
    MPI_Recv(buffer, buffer_size, MPI_DOUBLE, senders(r), POINT_TAG, p4est->mpicomm, &st);

    ArrayV<double> phi_interpolated(buffer_size/2);
    for (int i = 0; i<buffer_size/2; ++i)
    {
      p4est_topidx_t tree_id = 0;
      double xy [] = {received_departure_points(2*i), received_departure_points(2*i+1)};
      p4est_quadrant_t *departure_quadrant;
      p4est_locidx_t departure_quadrant_locidx;
      my_p4est_brick_point_lookup(p4est, xy, &tree_id, &departure_quadrant_locidx, &departure_quadrant);

      p4est_locidx_t nodes_locidx [] =
      {
        e2n[departure_quadrant_locidx*P4EST_CHILDREN + 0],
        e2n[departure_quadrant_locidx*P4EST_CHILDREN + 1],
        e2n[departure_quadrant_locidx*P4EST_CHILDREN + 2],
        e2n[departure_quadrant_locidx*P4EST_CHILDREN + 3]
      };

      for (int i = 0 ; i<4; ++i)
        nodes_locidx[i] = p4est2petsc_local_numbering(nodes, nodes_locidx[i]);

      double phi_buffer [] =
      {
        phi_val[nodes_locidx[0]],
        phi_val[nodes_locidx[1]],
        phi_val[nodes_locidx[2]],
        phi_val[nodes_locidx[3]]
      };

      phi_interpolated(i) = bilinear_interpolation(p4est, tree_id, departure_quadrant, phi_buffer, xy[0], xy[1]);
    }

    double *phi_interpolated_buffer = (double*)phi_interpolated;
    MPI_Send(phi_interpolated_buffer, phi_interpolated.size(), MPI_DOUBLE, senders(r), LVLSET_TAG, p4est->mpicomm);
  }

  for (int r = 0; r<receivers.size(); ++r)
  {
    ArrayV<double> phi_interpolated_received(departing_node.size());
    double *phi_buffer = (double*)phi_interpolated_received;
    MPI_Recv(phi_buffer, phi_interpolated_received.size(), MPI_DOUBLE, receivers(r), LVLSET_TAG, p4est->mpicomm, &st);

    ierr = VecSetValues(phi_np1, phi_interpolated_received.size(), (p4est_locidx_t*)departing_node(r), phi_buffer, INSERT_VALUES); CHKERRXX(ierr);
  }

  // TODO: Now we need to update our local values at the departing nodes of the level-set function based on the local departure points
  ArrayV<double> phi_interpolated(departing_node(p4est->mpirank).size());
  for (p4est_locidx_t n = 0; n<departing_node(p4est->mpirank).size(); ++n)
  {
    p4est_topidx_t tree_id = 0;
    double xy [] = {departure_point(p4est->mpirank)(2*n), departure_point(p4est->mpirank)(2*n+1)};
    p4est_quadrant_t *departure_quadrant;
    p4est_locidx_t departure_quadrant_locidx;

    my_p4est_brick_point_lookup(p4est, xy, &tree_id, &departure_quadrant_locidx, &departure_quadrant);
    p4est_tree_t *departure_tree = p4est_tree_array_index(p4est->trees, tree_id);
    departure_quadrant_locidx += departure_tree->quadrants_offset;

    p4est_locidx_t nodes_locidx [] =
    {
      e2n[departure_quadrant_locidx*P4EST_CHILDREN + 0],
      e2n[departure_quadrant_locidx*P4EST_CHILDREN + 1],
      e2n[departure_quadrant_locidx*P4EST_CHILDREN + 2],
      e2n[departure_quadrant_locidx*P4EST_CHILDREN + 3]
    };

    for (int i = 0 ; i<4; ++i)
      nodes_locidx[i] = p4est2petsc_local_numbering(nodes, nodes_locidx[i]);

    double phi_buffer [] =
    {
      phi_val[nodes_locidx[0]],
      phi_val[nodes_locidx[1]],
      phi_val[nodes_locidx[2]],
      phi_val[nodes_locidx[3]]
    };

    phi_interpolated(n) = bilinear_interpolation(p4est, tree_id, departure_quadrant, phi_buffer, xy[0], xy[1]);

  }

  double *phi_buffer = (double*)phi_interpolated;
  ierr =  VecSetValues(phi_np1, phi_interpolated.size(), (p4est_locidx_t*)departing_node(p4est->mpirank), phi_buffer, INSERT_VALUES); CHKERRXX(ierr);

  // Assemble the vector
  ierr = VecAssemblyBegin(phi_np1); CHKERRXX(ierr);
  ierr = VecAssemblyEnd(phi_np1); CHKERRXX(ierr);

  ierr = VecRestoreArray(velx, &velx_val); CHKERRXX(ierr);
  ierr = VecRestoreArray(vely, &vely_val); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi,  &phi_val);  CHKERRXX(ierr);

  ierr = VecDestroy(&phi); CHKERRXX(ierr);
  phi  = phi_np1;

}
