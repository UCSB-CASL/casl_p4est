#include "semi_lagrangian.h"
#include <src/bilinear_interpolating_function.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <mpi.h>
#include <sc_notify.h>

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

SemiLagrangian::SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, my_p4est_brick_t *myb)
  : p_p4est_(p4est), p4est_(*p4est),
    p_nodes_(nodes), nodes_(*nodes),
    myb_(myb)
{
  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;

  xmin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 0];
  ymin = v2c[3*t2v[P4EST_CHILDREN*first_tree + 0] + 1];
  xmax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 0];
  ymax = v2c[3*t2v[P4EST_CHILDREN*last_tree  + 3] + 1];
}

double SemiLagrangian::advect(const CF_2 &vx, const CF_2 &vy, Vec& phi){
  p4est_ghost_t *ghost = p4est_ghost_new(p4est_, P4EST_CONNECT_DEFAULT);

  BilinearInterpolatingFunction bif(p4est_, nodes_, ghost, myb_);
  bif.set_input_vector(phi);

  double dt = compute_dt(vx, vy);
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_->connectivity->vertices; // coordinates of the vertices of a tree

  // Create new Vec
  Vec phi_np1;
  ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

  // Loop through all nodes of local processor and separate nodes into local and non-local vectors
  // Local vectors will be used normally (serial).
  // Non-local vectors will need sent to non-local processors in order to compute data

  p4est_locidx_t ni_begin = 0;
  p4est_locidx_t ni_end   = nodes_->num_owned_indeps;

  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, ni+nodes_->offset_owned_indeps);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

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

    //Buffer the point for interpolation
    bif.add_point_to_buffer(ni, xy_departure[0], xy_departure[1]);
  }

  //interpolate from old vector into our output vector
  bif.interpolate(phi_np1);

  // update the p4est
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  update_p4est(phi, ghost);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  p4est_ghost_destroy(ghost);
  return dt;
}

void SemiLagrangian::update_p4est(Vec &phi, p4est_ghost_t *ghost){

  // make a new copy of p4est object -- we are going to modify p4est but we
  // still need the old one ...
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);//p4est_new(p4est_->mpicomm, conn, 0, NULL, NULL);

  // define an interpolating function
  BilinearInterpolatingFunction bif(p4est_, nodes_, ghost, myb_);
  bif.set_input_vector(phi);

  // now refine/coarsen the new copy of p4est -- note that we need to swap
  // level-set function since it has moved
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)(p4est_->user_pointer);
  data->phi = &bif;
  p4est_np1->user_pointer = data;
  p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_grid_transfer, NULL);
  p4est_refine(p4est_np1, P4EST_TRUE, refine_grid_transfer, NULL);
  p4est_partition(p4est_np1, NULL);

  // now compute a new node data structure
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
  //p4est_nodes_t *nodes_np1 = nodes_;

  // update the values at the new time-step
  Vec phi_np1;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

   // * Now we need to transfer data from the old grid to the new grid. Since this
   //* will generally involve interpolating from other processors, we need to put
   //* things in groups just as we did before

  p4est_locidx_t ni_begin = 0;
  p4est_locidx_t ni_end   = nodes_np1->num_owned_indeps;
  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni+nodes_np1->offset_owned_indeps);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;


    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];

    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];

    double xy_node [] =
    {
      int2double_coordinate_transform(indep_node->x) + tree_xmin,
      int2double_coordinate_transform(indep_node->y) + tree_ymin
    };

    bif.add_point_to_buffer(ni, xy_node[0], xy_node[1]);
  }

  // interpolate from old vector into our output vector
  bif.interpolate(phi_np1);

  // Update ghost values

  // now that everything is updated, get rid of old stuff and swap them with
  // new ones
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;
}

