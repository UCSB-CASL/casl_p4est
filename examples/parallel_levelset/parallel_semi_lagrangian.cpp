#include "parallel_semi_lagrangian.h"
#include "bilinear_interpolating_function.h"
#include <src/refine_coarsen.h>
#include <mpi/mpi.h>
#include <sc_notify.h>

#include <iostream>
#include <map>
#include <algorithm>

#define XY_TAG 0
#define PHI_TAG 1
#define SIZE_TAG 2

namespace parallel{

SemiLagrangian::SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes)
  : p_p4est_(p4est), p4est_(*p4est),
    p_nodes_(nodes), nodes_(*nodes)
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

double SemiLagrangian::advect(const CF_2 &vx, const CF_2 &vy, std::vector<double>& phi){
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

  // loop over all nodes
  std::vector<double> phi_np1(phi.size());

  // Loop through all nodes of local processor and separate nodes into local and non-local vectors
  // Local vectors will be used normally (serial).
  // Non-local vectors will need sent to non-local processors in order to compute data
  for (p4est_locidx_t ni = 0; ni < nodes_->num_owned_indeps; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;
    p4est_quadrant_t *quad;
    p4est_locidx_t quad_idx;

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
    rank_found = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                                      xy_departure,
                                                      &tree_idx,
                                                      &quad_idx,
                                                      &quad);

    // Check if ranks match (if backtraced point is within the current processor)
    if (rank_current == rank_found){     //Point is local to current processor
      local_dep_points_info.push(xy_departure, tree_idx, quad_idx, quad, ni);   //Push_back values for interpolation
    }
    else{                                 //Else, backtraced point is not in current processor.
      if (send_buffer_map.find(rank_found) == send_buffer_map.end()){  //Check if map has associated the found rank yet, if not, make it.
        send_buffer_map[rank_found] = non_local_point_buffer();    //Init new buffer
        send_buffer_map[rank_found].push(xy_departure, ni);    //Init values
      }
      else  //If we found a point in a pair we already made, add it to that pair.
        send_buffer_map[rank_found].push(xy_departure, ni);     //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
    }
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
    phi_np1[local_dep_points_info.ni[i]] = linear_interpolation(phi, xy, local_dep_points_info.tree_idx[i]);  //Interpolate
  }

  //Loop over received points and interpolate them.
  for (recv_it rank_it = xy_map.begin(); rank_it != xy_map.end(); ++rank_it){
    vector<double> phi_temp (rank_it->second.size() / 2);   //Temp phi values to be copied to phi_send map
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
      //Catch if rank_found is in current processor
      if (rank_found != rank_current){
        int abc;
        std::cout << "ERROR, SENT DATA TO WRONG PROCESSOR" << endl;
        std::cin >> abc;
      }
      else
        phi_temp[vec_it] = linear_interpolation(phi, xy_departure, tree_idx);  //Interpolate, and put into temp phi vector
    }
    phi_send[rank_it->first] = phi_temp;       //Copy values to phi_send map which will be used to send data back to other processors
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
    MPI_Recv(&phi_temp_buf, phi_buf_size, MPI_DOUBLE, receivers[it], PHI_TAG, p4est_->mpicomm, MPI_STATUS_IGNORE);  //Receive the data from processors

    //Insert received calculated phi values back into new level set (phi_np1)
    for (int it = 0; it < phi_temp_buf.size(); ++it){
      phi_np1[send_buffer_map[receivers[it]].ni[it]] = phi_temp_buf[it];
    }
  }

  //AT THIS POINT
  // phi_np1 has all the computed local phi values. Just need to copy these values over old phi.

  std::copy(phi_np1.begin(), phi_np1.end(), phi.begin());   //Copy new phi values to old phi values.

  // update the p4est
  update_p4est(phi);
  return dt;
}

void SemiLagrangian::update_p4est(std::vector<double>& phi){
  // define an interpolating function
  BilinearInterpolatingFunction bif(phi, p4est_, nodes_);

  // make a new copy of p4est object -- we are going to modify p4est but we
  // still need the old one ...
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);

  // now refine/coarsen the new copy of p4est -- note that we need to swap
  // level-set function since it has moved
  cf_grid_data_t *data = (cf_grid_data_t*)(p4est_->user_pointer);
  data->phi = &bif;
  p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset, NULL);
  p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset, NULL);
  p4est_partition(p4est_np1, NULL);

  // now compute a new node data structure
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1);

  // update the values at the new time-step
  vector<double> phi_np1(nodes_np1->num_owned_indeps);

  for (p4est_locidx_t ni = 0; ni<nodes_np1->num_owned_indeps; ++ni){
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = p4est_np1->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = p4est_np1->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est_np1->connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi_np1[ni] = bif(x,y);
  }

  // now that everything is updated, get rid of old stuff and swap them with
  // new ones
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  phi.resize(phi_np1.size()); copy(phi_np1.begin(), phi_np1.end(), phi.begin());
}

}
