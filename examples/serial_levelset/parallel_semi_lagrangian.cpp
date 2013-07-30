#include "parallel_semi_lagrangian.h"
#include "bilinear_interpolating_function_p.h"
#include <src/refine_coarsen.h>
#include <mpi/mpi.h>

#include <iostream>
#include <map>

namespace parallel{

std::map <int, non_local_point_buffer> rank_to_buffer;      //Use a map to associate each rank to their respective struct of values to be sent using MPI SEND/RECV


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
  double dt = compute_dt(vx, vy);
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_->connectivity->vertices; // coordinates of the vertices of a tree

  int rank_current = 0, rank_found = 0;    //P: Used to check for ranks
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_current); //Get current rank
  //std::cout << "Current Rank = " << rank_current << endl;

  // Buffers for local and non-local nodes for MPI_Send/MPI_RECV
  //std::vector<p4est_indep_t> local_dep_points, non_local_dep_points; May not need this.

  // Hold quadrant information for MPI_SEND/MPI_RECV. Changed to a struct of vectors instead of a vector of structs
  quad_information local_dep_points_info, non_local_dep_points_info;

  // Hold xy & rank & node inex for non-local points to be sent via MPI_SEND/MPI_RECEIVE
  //non_local_point_buffer non_local_send_buffer, non_local_receive_buffer;

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
      std::cout << "Rank Found = " << rank_found;
      std::cout << "Current Rank = " << rank_current;

      // Check if ranks match (if backtraced point is within the current processor)
      if (rank_current == rank_found){     //Point is local to current processor
          local_dep_points_info.push(xy_departure, tree_idx, quad_idx, quad, ni);   //Push_back values for interpolation
      }
      else{                                 //Else, backtraced point is not in current processor.
          if (rank_to_buffer.find(rank_found) == rank_to_buffer.end()){  //Check if map has associated the found rank yet, if not, make it.
              rank_to_buffer[rank_found] = non_local_point_buffer();    //Init new buffer
              rank_to_buffer[rank_found].push(xy_departure, ni);    //Init values
          }
          else  //If we found a point in a pair we already made, add it to that pair.
              rank_to_buffer[rank_found].push(xy_departure, ni);     //Need to send xy_departure to correct processor and recalculate tree_idx, quad_idx, and *quad.
      }
  }

  //AT THIS POINT, WE NOW HAVE TWO VARIABLES: local_dep_points_info & rank_to_buffer.
  // local_dep_points has all the points local to their respective processor.
  // rank_to_buffer is a map where each element represents a rank that is associated to a buffer that will be sent later.
  // rank_to_buffer[rank] = buffer;


  int abc = 0;
  std::cin >> abc;
  // Loop over local points and interpolate them.
  for (p4est_locidx_t ni = 0; ni < local_dep_points_info.ni.size(); ++ni){
    //Insert code to loop over local points and run linear_interpolation from parallel_semi_lagrangian.h
      double xy[] = {
                      local_dep_points_info.xy[(ni * 2) + 0],
                      local_dep_points_info.xy[(ni * 2) + 1]
                    };
      phi_np1[ni] = linear_interpolation(phi, xy, local_dep_points_info.tree_idx[ni]);  //Interpolate
  }



  //===================== UNSURE OF NON LOCAL POINTS RIGHT NOW ==========================

  //for (std::map <int, non_local_point_buffer>::iterator map_it = rank_to_buffer.begin; map_it != rank_to_buffer.end(); ++map_it){
      //*map_it
      //Use MPI_SEND & MPI_RECV to send and receive data to processors.
      //
  //}

  //Old serial code
/*
  for (p4est_locidx_t ni = 0; ni <nodes_->num_owned_indeps; ++ni){
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, ni);

    // find the global xy coordinate of a node among trees
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;
    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0]; // mm vertex of tree
    double tr_xmin = t2c[3*tr_mm + 0];
    double tr_ymin = t2c[3*tr_mm + 1];

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

//    // clip on the boundary
//    if (xy_departure[0] > xmax)
//      xy_departure[0] = xmax;
//    else if (xy_departure[0] < xmin)
//      xy_departure[0] = xmin;
//    if (xy_departure[1] > ymax)
//      xy_departure[1] = ymax;
//    else if (xy_departure[1] < ymin)
//      xy_departure[1] = ymin;

    // compute the new value of the level-set function via interpolation
    phi_np1[ni] = linear_interpolation(phi, xy_departure, tree_idx, quad_info);
  }

*/

  // copy the new values into the old vector
  //std::copy(phi_np1.begin(), phi_np1.end(), phi.begin());

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
