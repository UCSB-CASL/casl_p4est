#include "parallel_semi_lagrangian.h"
#include "bilinear_interpolating_function.h"
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <mpi/mpi.h>
#include <sc_notify.h>

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#define XY_TAG 0
#define PHI_TAG 1
#define SIZE_TAG 2

namespace parallel{

SemiLagrangian::SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, my_p4est_brick_t *myb)
  : p_p4est_(p4est), p4est_(*p4est),
    p_nodes_(nodes), nodes_(*nodes),
    myb_(myb)
{
  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->last_local_tree;

  xmin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 0];
  ymin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 1];
  xmax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 0];
  ymax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 1];
}

double SemiLagrangian::advect(const CF_2 &vx, const CF_2 &vy, Vec& phi){
  p4est_ghost_t *ghost = p4est_ghost_new(p4est_, P4EST_CONNECT_DEFAULT);

  std::map <int, non_local_point_buffer> send_buffer_map;      //Use a map to associate each rank to their respective struct of values to be sent using MPI SEND/RECV
  typedef std::map <int, non_local_point_buffer>::iterator send_it;

  std::map <int, vector<double> > xy_map;   //Map ranks to xy_buffers
  typedef std::map <int, vector<double> >::iterator recv_it;

  std::map <int, vector<double> > phi_send; //Map ranks to phi_send buffer to be sent via MPI_SEND/RECV
  typedef std::map <int, vector<double> >::iterator phi_send_it;

  double dt = compute_dt(vx, vy);
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_->connectivity->vertices; // coordinates of the vertices of a tree

  int rank_current = p4est_->mpirank, rank_found = 0;    //P: Used to check for ranks

  // Hold quadrant information for MPI_SEND/MPI_RECV. Changed to a struct of vectors instead of a vector of structs
  quad_information local_dep_points_info;

  // Create new Vec
  Vec phi_np1;
  ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

  // get access to the pointer
  double *phi_np1_ptr;
  ierr = VecGetArray(phi_np1, &phi_np1_ptr); CHKERRXX(ierr);

  double *phi_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

  // Loop through all nodes of local processor and separate nodes into local and non-local vectors
  // Local vectors will be used normally (serial).
  // Non-local vectors will need sent to non-local processors in order to compute data
  p4est_locidx_t ni_begin = nodes_->offset_owned_indeps;
  p4est_locidx_t ni_end   = nodes_->offset_owned_indeps + nodes_->num_owned_indeps;

  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;
    p4est_quadrant_t *quad;
    p4est_locidx_t quad_idx;
    sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];

    //Find initial xy points
    double xy[] =
    {
      int2double_coordinate_transform(indep_node->x) + tr_xmin,
      int2double_coordinate_transform(indep_node->y) + tr_ymin
    };

    // find the departure node via backtracing
    double xy_departure[] =
    {
      xy[0] - dt*vx(xy[0], xy[1]),
      xy[1] - dt*vy(xy[0], xy[1])
    };

    // make the domain periodic
    if (xy_departure[0] > xmax)
      xy_departure[0] -= xmax - xmin;
    else if (xy_departure[0] < xmin)
      xy_departure[0] += xmax - xmin;
    if (xy_departure[1] > ymax)
      xy_departure[1] -= ymax - ymin;
    else if (xy_departure[1] < ymin)
      xy_departure[1] += ymax - ymin;

    //Find quadrant of xy_departure point
    //Returns: quadrant, quadrant index, tree index, and processor rank of the backtraced xy coordinates
//    rank_found = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
//                                                      xy_departure,
//                                                      &tree_idx,
//                                                      &quad_idx,
//                                                      &quad);

    /*
     * We are using this function since the other one does not work in parallel
     * There are multiple cases that can happen. If the return is equal to the
     * current rank, we own the quadrant in our local list. If the return does
     * not match the current rank, then the quadrant is in the ghost layer.
     * finally if the return rank is -1, then not only we do not own the quad,
     * but its also not even in the ghost layer; its somewhere far away! In this
     * case, remote_matches array should be checked for the actual quadrant
     */
    rank_found = my_p4est_brick_point_lookup(p4est_, ghost, myb_,
                                             xy_departure, &quad, remote_matches);

    quad_idx = quad->p.piggy3.local_num;
    tree_idx = quad->p.piggy3.which_tree;

    // Check if ranks match (if backtraced point is within the current processor)
    if (rank_current == rank_found){
      local_dep_points_info.push(xy_departure, tree_idx, quad_idx/*quad->p.piggy3.local_num*/, quad, ni-nodes_->offset_owned_indeps);   //Push_back values for interpolation
    } else if (rank_found >= 0){ // in the ghost layer
      if (send_buffer_map.find(rank_found) == send_buffer_map.end()){  //Check if map has associated the found rank yet, if not, make it.
        send_buffer_map[rank_found] = non_local_point_buffer();    //Init new buffer
        send_buffer_map[rank_found].push(xy_departure, ni-nodes_->offset_owned_indeps);    //Init values
      }
      else  //If we found a point in a pair we already made, add it to that pair.
        send_buffer_map[rank_found].push(xy_departure, ni-nodes_->offset_owned_indeps);     //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
    } else {
/*      cout << __LINE__ << endl;
      p4est_quadrant_t *remote_quad = (p4est_quadrant_t*)sc_array_index(remote_matches, 0);
      int remote_rank = remote_quad->p.piggy1.owner_rank;
      if (send_buffer_map.find(remote_rank) == send_buffer_map.end()){  //Check if map has associated the found rank yet, if not, make it.
        send_buffer_map[remote_rank] = non_local_point_buffer();    //Init new buffer
        send_buffer_map[remote_rank].push(xy_departure, ni-nodes_->offset_owned_indeps);    //Init values
      }
      else  //If we found a point in a pair we already made, add it to that pair.
        send_buffer_map[remote_rank].push(xy_departure, ni-nodes_->offset_owned_indeps);  */   //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
    }

    sc_array_destroy(remote_matches);
  }

  //AT THIS POINT, WE NOW HAVE TWO VARIABLES: local_dep_points_info & send_buffer_map.
  // local_dep_points has all the points local to their respective processor.
  // send_buffer_map is a map where each element represents a rank that is associated to a buffer that will be sent later.
  // send_buffer_map[rank] = buffer;

  vector<int> receivers;        //Processors that are receiving data from us.
  vector<int> senders(p4est_->mpisize);          //Processors that are sending data to us.
  int num_senders = 0;          //Total number of processors we're sending data to.

  //Init the receivers buffer with the ranks of the processors from the send_buffer_map map.
  for (send_it it = send_buffer_map.begin(); it != send_buffer_map.end(); ++it){
    receivers.push_back(it->first);    //it->first holds the first value of the pair (the rank).
  }

  //Populate senders and num_senders from the number of receivers.
  sc_notify(&receivers[0], receivers.size(), &senders[0], &num_senders, p4est_->mpicomm);
  senders.resize(num_senders);

  //MPI_SEND to send data out to other processors from us.
  for (int it = 0; it < receivers.size(); ++it){
    vector<double>& xy_buf = send_buffer_map[receivers[it]].xy;    //Init xy send buffer from our map
    int xy_buf_size =  xy_buf.size();           //Init the buffer size

    MPI_Send(&xy_buf_size, 1, MPI_INT, receivers[it], SIZE_TAG, p4est_->mpicomm);   //Send the buffer size out
    MPI_Send(&xy_buf[0], xy_buf.size(), MPI_DOUBLE, receivers[it], XY_TAG, p4est_->mpicomm);  //Send the data (xy_buf) to all required processors
  }

  //MPI_RECV to receive data from all other processors to us.
  for (int it = 0; it < senders.size(); ++it){
    int xy_buf_size;      //Init buffer size
    MPI_Recv(&xy_buf_size, 1, MPI_INT, senders[it], SIZE_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);    //Receive the buffer size

    vector<double> xy_buf(xy_buf_size);   //Init the receive buffer
    MPI_Recv(&xy_buf[0], xy_buf_size, MPI_DOUBLE, senders[it], XY_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);   //Receive the data from all required processors

    xy_map[senders[it]] = xy_buf;     //Map the received processor with its buffer.
  }

  //AT THIS POINT
  // We now have: local_dep_points_info and non local points (in xy_map)
  // Local: Interpolation is done the same as serial.
  // Non-Local: Need to call brick_lookup and then do interpolation. After interpolation, we need to send the data BACK to
  //    their respective processors.

  // Loop over local points and interpolate them.
  for (p4est_locidx_t i = 0; i < local_dep_points_info.ni.size(); ++i){
    double xy[] = {
      local_dep_points_info.xy[(i * 2) + 0],
      local_dep_points_info.xy[(i * 2) + 1]
    };
    phi_np1_ptr[local_dep_points_info.ni[i]] = linear_interpolation(phi_ptr, xy, local_dep_points_info.tree_idx[i]);  //Interpolate
  }

  //Loop over received points and interpolate them.
  for (recv_it rank_it = xy_map.begin(); rank_it != xy_map.end(); ++rank_it){
    vector<double> &phi_temp = phi_send[rank_it->first];
    phi_temp.resize(rank_it->second.size() / 2);   //Temp phi values to be copied to phi_send map
    for (int vec_it = 0; vec_it < phi_temp.size(); ++vec_it){    //Loop over xy values
      p4est_topidx_t tree_idx = 0;
      p4est_locidx_t quad_idx;
      p4est_quadrant_t *quad;
      int rank_current = p4est_->mpirank, rank_found = 0;

      double xy_departure[] = {   //Setup xy_departure array for interpolation
                                  rank_it->second[(vec_it * 2) + 0],    //Set x value
                                  rank_it->second[(vec_it * 2) + 1]     //Set y value
                              };

      //Find quadrant of xy_departure point
      //Returns: quadrant, quadrant index, tree index, and processor rank of the backtraced xy coordinates
      rank_found = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                                        xy_departure,
                                                        &tree_idx,
                                                        &quad_idx,
                                                        &quad);
#ifdef CASL_THROWS
      if (rank_found != rank_current)
        throw std::runtime_error("Data has been sent to the wrong processor");
#endif
      phi_temp[vec_it] = linear_interpolation(phi_ptr, xy_departure, tree_idx);  //Interpolate, and put into temp phi vector
    }
  }

  //AT THIS POINT
  // We now have local points computed in vector<double> phi_np1.
  // We also have non local points computed in map<int, vector<double> > phi_send, organized with <rank, values>
  // Now we need to send phi_send to all their respective processors using MPI.

  //MPI_SEND to send calculated phi values from phi_buf back TO previous senders
  for (int it = 0; it < senders.size(); ++it){
    std::vector<double> &phi_buf = phi_send[senders[it]];    //Init send buffer from our map
    int phi_buf_size = phi_buf.size();    //Init the buffer size

    MPI_Send(&phi_buf_size, 1, MPI_INT,senders[it], SIZE_TAG, p4est_->mpicomm);    //Send the buffer size
    MPI_Send(&phi_buf[0], phi_buf_size, MPI_DOUBLE, senders[it], PHI_TAG, p4est_->mpicomm);    //Send the data to all designated processors
  }

  //MPI_RECV to receive calculated phi values FROM previous receivers
  for (int it = 0; it < receivers.size(); ++it){
    int phi_buf_size;    //Buffer size that we will receive
    MPI_Recv(&phi_buf_size, 1, MPI_INT, receivers[it], SIZE_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);   //Receive the buffer size

    std::vector<double> phi_temp_buf(phi_buf_size);      //Init a temp buffer that we will put received data in.
    MPI_Recv(&phi_temp_buf[0], phi_buf_size, MPI_DOUBLE, receivers[it], PHI_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);  //Receive the data from processors

    //Insert received calculated phi values back into new level set (phi_np1)
    for (int i = 0; i < phi_temp_buf.size(); ++i){
      phi_np1_ptr[send_buffer_map[receivers[it]].ni[i]] = phi_temp_buf[i];
    }
  }

  //AT THIS POINT
  // phi_np1 has all the computed local phi values. Just need to copy these values over old phi.

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_np1, &phi_np1_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Delete the old Vec and swap the pointers
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi  = phi_np1;

  // update the p4est
//  update_p4est(phi, ghost);
  p4est_ghost_destroy(ghost);
  return dt;
}

void SemiLagrangian::update_p4est(Vec &phi, p4est_ghost_t *ghost){
  std::map <int, non_local_point_buffer> send_buffer_map;      //Use a map to associate each rank to their respective struct of values to be sent using MPI SEND/RECV
  typedef std::map <int, non_local_point_buffer>::iterator send_it;

  std::map <int, vector<double> > xy_map;   //Map ranks to xy_buffers
  typedef std::map <int, vector<double> >::iterator recv_it;

  std::map <int, vector<double> > phi_send; //Map ranks to phi_send buffer to be sent via MPI_SEND/RECV
  typedef std::map <int, vector<double> >::iterator phi_send_it;

  quad_information local_dep_points_info;

  int rank_current = p4est_->mpirank;

  // define an interpolating function
  double *phi_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  BilinearInterpolatingFunction bif(phi_ptr, p4est_, nodes_);

  // make a new copy of p4est object -- we are going to modify p4est but we
  // still need the old one ...
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);

  // now refine/coarsen the new copy of p4est -- note that we need to swap
  // level-set function since it has moved
  cf_grid_data_t *data = (cf_grid_data_t*)(p4est_->user_pointer);
  data->phi = &bif;
  p4est_np1->user_pointer = p4est_->user_pointer;
  p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset, NULL);
  p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset, NULL);
  p4est_partition(p4est_np1, NULL);

  // now compute a new node data structure
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1);

  // update the values at the new time-step
  Vec phi_np1;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  double *phi_np1_ptr;
  ierr = VecGetArray(phi_np1, &phi_np1_ptr); CHKERRXX(ierr);

  /*
   * Now we need to transfer data from the old grid to the new grid. Since this
   * will generally involve interpolating from other processors, we need to put
   * things in groups just as we did before
   */
  p4est_locidx_t ni_begin = nodes_np1->offset_owned_indeps;
  p4est_locidx_t ni_end   = nodes_np1->offset_owned_indeps + nodes_np1->num_owned_indeps;
  for (p4est_locidx_t ni = ni_begin; ni<ni_end; ++ni){
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
    p4est_quadrant_t *quad;
    //    p4est_locidx_t quad_idx;
    sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));

    p4est_topidx_t v_mm = p4est_np1->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];

    double tree_xmin = p4est_np1->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est_np1->connectivity->vertices[3*v_mm + 1];

    double xy_node [] =
    {
      int2double_coordinate_transform(node->x) + tree_xmin,
      int2double_coordinate_transform(node->y) + tree_ymin
    };

    /* find the processor that can do the interpolation. Note that we will look
     * for the correct processoron the OLD grid
     */
    int rank_found = my_p4est_brick_point_lookup(p4est_, ghost, myb_,
                                             xy_node, &quad, remote_matches);

    // Check if ranks match (if backtraced point is within the current processor)
    if (rank_current == rank_found){
      local_dep_points_info.push(xy_node, tree_idx, quad->p.piggy3.local_num, quad, ni-nodes_->offset_owned_indeps);   //Push_back values for interpolation
    } else if (rank_found >= 0){ // in the ghost layer
      if (send_buffer_map.find(rank_found) == send_buffer_map.end()){  //Check if map has associated the found rank yet, if not, make it.
        send_buffer_map[rank_found] = non_local_point_buffer();    //Init new buffer
        send_buffer_map[rank_found].push(xy_node, ni-nodes_->offset_owned_indeps);    //Init values
      }
      else  //If we found a point in a pair we already made, add it to that pair.
        send_buffer_map[rank_found].push(xy_node, ni-nodes_->offset_owned_indeps);     //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
    } else {
      p4est_quadrant_t *remote_quad = (p4est_quadrant_t*)sc_array_index(remote_matches, 0);
      int remote_rank = remote_quad->p.piggy1.owner_rank;
      if (send_buffer_map.find(remote_rank) == send_buffer_map.end()){  //Check if map has associated the found rank yet, if not, make it.
        send_buffer_map[remote_rank] = non_local_point_buffer();    //Init new buffer
        send_buffer_map[remote_rank].push(xy_node, ni-nodes_->offset_owned_indeps);    //Init values
      }
      else  //If we found a point in a pair we already made, add it to that pair.
        send_buffer_map[remote_rank].push(xy_node, ni-nodes_->offset_owned_indeps);     //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
    }

    sc_array_destroy(remote_matches);
  }

  // Find the correct set of processors to send and receive information
  vector<int> receivers;        //Processors that are receiving data from us.
  vector<int> senders(p4est_->mpisize);          //Processors that are sending data to us.
  int num_senders = 0;          // # of processors that will actually send data to us

  //Init the receivers buffer with the ranks of the processors from the send_buffer_map map.
  for (send_it it = send_buffer_map.begin(); it != send_buffer_map.end(); ++it){
    receivers.push_back(it->first);    //it->first holds the first value of the pair (the rank).
  }

  //Populate senders and num_senders from the number of receivers.
  sc_notify(&receivers[0], receivers.size(), &senders[0], &num_senders, p4est_->mpicomm);
  senders.resize(num_senders);

  //MPI_SEND to send data out to other processors from us.
  for (int it = 0; it < receivers.size(); ++it){
    vector<double>& xy_buf = send_buffer_map[receivers[it]].xy;    //Init xy send buffer from our map
    int xy_buf_size =  xy_buf.size();           //Init the buffer size

    MPI_Send(&xy_buf_size, 1, MPI_INT, receivers[it], SIZE_TAG, p4est_->mpicomm);   //Send the buffer size out
    MPI_Send(&xy_buf[0], xy_buf.size(), MPI_DOUBLE, receivers[it], XY_TAG, p4est_->mpicomm);  //Send the data (xy_buf) to all required processors
  }

  //MPI_RECV to receive data from all other processors to us.
  for (int it = 0; it < senders.size(); ++it){
    int xy_buf_size;      //Init buffer size
    MPI_Recv(&xy_buf_size, 1, MPI_INT, senders[it], SIZE_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);    //Receive the buffer size

    vector<double> xy_buf(xy_buf_size);   //Init the receive buffer
    MPI_Recv(&xy_buf[0], xy_buf_size, MPI_DOUBLE, senders[it], XY_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);   //Receive the data from all required processors

    xy_map[senders[it]] = xy_buf;     //Map the received processor with its buffer.
  }

  // loop over local portion and do the interpolation
  for (size_t i=0; i<local_dep_points_info.ni.size(); ++i){
    double *xy = &local_dep_points_info.xy[2*i + 0];
    phi_np1_ptr[local_dep_points_info.ni[i]] = bif(xy[0], xy[1]);
  }

  // loop over nonlocal points and do the interpolation
  for (recv_it rank_it = xy_map.begin(); rank_it != xy_map.end(); ++rank_it){
    vector<double>& phi_temp = phi_send[rank_it->first];
    phi_temp.resize(rank_it->second.size() / 2);   //Temp phi values to be copied to phi_send map
    for (int vec_it = 0; vec_it < phi_temp.size(); ++vec_it){    //Loop over xy values
      double *xy = &rank_it->second[2*vec_it + 0];
      phi_temp[vec_it] = bif(xy[0], xy[1]);
    }
  }

  //MPI_SEND to send calculated phi values from phi_buf back TO previous senders
  for (int it = 0; it < senders.size(); ++it){
    std::vector<double> &phi_buf = phi_send[senders[it]];    //Init send buffer from our map
    int phi_buf_size = phi_buf.size();    //Init the buffer size

    MPI_Send(&phi_buf_size, 1, MPI_INT,senders[it], SIZE_TAG, p4est_->mpicomm);    //Send the buffer size
    MPI_Send(&phi_buf[0], phi_buf_size, MPI_DOUBLE, senders[it], PHI_TAG, p4est_->mpicomm);    //Send the data to all designated processors
  }

  //MPI_RECV to receive calculated phi values FROM previous receivers
  for (int it = 0; it < receivers.size(); ++it){
    int phi_buf_size;    //Buffer size that we will receive
    MPI_Recv(&phi_buf_size, 1, MPI_INT, receivers[it], SIZE_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);   //Receive the buffer size

    std::vector<double> phi_temp_buf(phi_buf_size);      //Init a temp buffer that we will put received data in.
    MPI_Recv(&phi_temp_buf[0], phi_buf_size, MPI_DOUBLE, receivers[it], PHI_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);  //Receive the data from processors

    //Insert received calculated phi values back into new level set (phi_np1)
    for (int i = 0; i < phi_temp_buf.size(); ++i){
      phi_np1_ptr[send_buffer_map[receivers[it]].ni[i]] = phi_temp_buf[i];
    }
  }

  // now that everything is updated, get rid of old stuff and swap them with
  // new ones
  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_np1, &phi_np1_ptr); CHKERRXX(ierr);

  // Update ghost values
  ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

}

double SemiLagrangian::linear_interpolation(const double *F, const double xy[], p4est_topidx_t tree_idx)
{
  p4est_locidx_t quad_idx;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t *quad;

  int found_rank = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                                        xy, &tree_idx, &quad_idx, &quad);

#ifdef CASL_THROWS
  if (p4est_->mpirank != found_rank){
    std::ostringstream oss; oss << "point (" << xy[0] << "," << xy[1] << ") "
                                   "does not belog to processor " << p4est_->mpirank;
    throw std::invalid_argument(oss.str());
  }
#endif

  p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
  quad_idx += tree->quadrants_offset;
  double f [] =
  {
    F[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 0])],
    F[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 1])],
    F[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 2])],
    F[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 3])]
  };

  return bilinear_interpolation(p4est_, tree_idx, quad, f, xy);
}

}
