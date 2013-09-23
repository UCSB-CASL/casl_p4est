#include "semi_lagrangian.h"
#include <src/interpolating_function.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <mpi.h>
#include <sc_notify.h>

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

SemiLagrangian::SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb)
  : p_p4est_(p4est), p4est_(*p4est),
    p_nodes_(nodes), nodes_(*nodes),
    p_ghost_(ghost), ghost_(*ghost),
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


double SemiLagrangian::compute_dt(const CF_2 &vx, const CF_2 &vy)
{
  double dt = DBL_MAX;

  // loop over trees
  for (p4est_topidx_t tr_it = p4est_->first_local_tree; tr_it <=p4est_->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tr_it);
    p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
    double *v2c = p4est_->connectivity->vertices;

    double tr_xmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 0];
    double tr_ymin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 1];

    // loop over quadrants
    for (size_t qu_it=0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      double dx = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
      double x  = int2double_coordinate_transform(quad->x) + 0.5*dx + tr_xmin;
      double y  = int2double_coordinate_transform(quad->y) + 0.5*dx + tr_ymin;
      double vn = sqrt(SQR(vx(x,y)) + SQR(vy(x,y)));
      dt = MIN(dt, dx/vn);
    }
  }

  double dt_min;
  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm);

  return dt_min;
}


double SemiLagrangian::advect(const CF_2 &vx, const CF_2 &vy, Vec& phi){

  // create hierarchy structure if you want to do quadratic interpolation
  my_p4est_hierarchy_t hierarchy(p4est_, ghost_);
  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);

  InterpolatingFunction bif(p4est_, nodes_, ghost_, myb_, &qnnn);
  bif.set_input_parameters(phi, quadratic_non_oscillatory);

  double dt = 5.*compute_dt(vx, vy);
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

    //Buffer the point for interpolation
    bif.add_point_to_buffer(ni, xy_departure[0], xy_departure[1]);
  }

  //interpolate from old vector into our output vector
  bif.interpolate(phi_np1);

  // update the p4est
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  update_p4est(phi, qnnn);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  return dt;
}



void SemiLagrangian::advect_from_n_to_np1(const CF_2& vx, const CF_2& vy, double dt, Vec &phi_n, std::vector<double> &phi_np1,
                                          p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, my_p4est_node_neighbors_t &qnnn)
{
  InterpolatingFunction interp(p4est_, nodes_, ghost_, myb_, &qnnn);
  interp.set_input_parameters(phi_n, quadratic_non_oscillatory);

  p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree

  p4est_locidx_t ni_begin = 0;
  p4est_locidx_t ni_end   = nodes_np1->indep_nodes.elem_count;

  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];

    /* Find initial xy points */
    double xy[] =
    {
      int2double_coordinate_transform(indep_node->x) + tr_xmin,
      int2double_coordinate_transform(indep_node->y) + tr_ymin
    };

    /* find the departure node via backtracing */

    double x_star = xy[0] - .5*dt*vx(xy[0], xy[1]);
    double y_star = xy[1] - .5*dt*vy(xy[0], xy[1]);

    double xy_departure[] =
    {
      xy[0] - dt*vx(x_star, y_star),
      xy[1] - dt*vy(x_star, y_star)
    };

    /* Buffer the point for interpolation */
    interp.add_point_to_buffer(ni, xy_departure[0], xy_departure[1]);
  }

  /* interpolate from old vector into our output vector */
  interp.interpolate(phi_np1.data());
}


double SemiLagrangian::update_p4est_intermediate_trees_no_ghost(const CF_2& vx, const CF_2& vy, Vec &phi, double dt)
{
  PetscErrorCode ierr;
  p4est *p4est_np1 = p4est_new(p4est_->mpicomm, p4est_->connectivity, 0, NULL, NULL);

  /* create hierarchy structure on p4est_n if you want to do quadratic interpolation */
  my_p4est_hierarchy_t hierarchy(p4est_, ghost_);
  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);
  std::vector<double> phi_tmp;

  //  double dt = 100*compute_dt(vx, vy);
  //  double dt = 1;

  int nb_iter = ((splitting_criteria_t*) (p4est_->user_pointer))->max_lvl - ((splitting_criteria_t*) (p4est_->user_pointer))->min_lvl;

  for( int iter = 0; iter < nb_iter; ++iter )
  {
    p4est_t       *p4est_tmp = p4est_copy(p4est_np1, P4EST_FALSE);
    p4est_nodes_t *nodes_tmp = my_p4est_nodes_new(p4est_tmp,NULL);

    /* compute phi_np1 on intermediate grid */
    phi_tmp.resize(nodes_tmp->indep_nodes.elem_count);
    advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp, p4est_tmp, nodes_tmp, qnnn);

    /* refine p4est_np1 */
    splitting_criteria_t *data = (splitting_criteria_t*) p4est_->user_pointer;
    splitting_criteria_update_t data_np1(1.2, data->min_lvl, data->max_lvl, &phi_tmp, myb_, p4est_tmp, NULL, nodes_tmp);
    p4est_np1->user_pointer = (void*) &data_np1;
    p4est_refine (p4est_np1, P4EST_FALSE, refine_criteria_sl , NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_destroy(p4est_tmp);
  }

  p4est_partition(p4est_np1, NULL);

  /* restore the user pointer in the p4est */
  p4est_np1->user_pointer = p4est_->user_pointer;

  /* compute new ghost layer */
  p4est_ghost_t *ghost_np1 = p4est_ghost_new(p4est_np1, P4EST_CONNECT_DEFAULT);

  /* compute the nodes structure */
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  /* update the values of phi at the new time-step */
  phi_tmp.resize(nodes_np1->indep_nodes.elem_count);
  advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp, p4est_np1, nodes_np1, qnnn);

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

  Vec phi_np1;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  double *f;
  ierr = VecGetArray(phi_np1, &f); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
    f[n] = phi_tmp[n+nodes_np1->offset_owned_indeps];
  ierr = VecRestoreArray(phi_np1, &f); CHKERRXX(ierr);

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  return dt;
}



double SemiLagrangian::update_p4est_intermediate_trees_with_ghost(const CF_2& vx, const CF_2& vy, Vec &phi)
{
  PetscErrorCode ierr;
  p4est *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);

  /* create hierarchy structure on p4est_n if you want to do quadratic interpolation */
  my_p4est_hierarchy_t hierarchy(p4est_, ghost_);
  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);
  std::vector<double> phi_tmp;

  double dt = 1;

  int nb_iter = ((splitting_criteria_t*) (p4est_->user_pointer))->max_lvl - ((splitting_criteria_t*) (p4est_->user_pointer))->min_lvl;

  for( int iter = 0; iter < nb_iter; ++iter )
  {
    p4est_t       *p4est_tmp = p4est_copy(p4est_np1, P4EST_FALSE);
    p4est_ghost_t *ghost_tmp = p4est_ghost_new(p4est_tmp, P4EST_CONNECT_DEFAULT);
    p4est_nodes_t *nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);

    /* compute phi_np1 on intermediate grid */
    phi_tmp.resize(nodes_tmp->indep_nodes.elem_count);
    advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp, p4est_tmp, nodes_tmp, qnnn);

    /* refine / coarsen p4est_np1 */
    splitting_criteria_t *data = (splitting_criteria_t*) p4est_->user_pointer;
    splitting_criteria_update_t data_np1(1.2, data->min_lvl, data->max_lvl, &phi_tmp, myb_, p4est_tmp, ghost_tmp, nodes_tmp);
    p4est_np1->user_pointer = (void*) &data_np1;

    p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_criteria_with_ghost_sl, NULL);
    throw std::invalid_argument("");
    p4est_refine (p4est_np1, P4EST_FALSE, refine_criteria_with_ghost_sl , NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_destroy(p4est_tmp);
  }

  p4est_partition(p4est_np1, NULL);

  /* restore the user pointer in the p4est */
  p4est_np1->user_pointer = p4est_->user_pointer;

  /* compute new ghost layer */
  p4est_ghost_t *ghost_np1 = p4est_ghost_new(p4est_np1, P4EST_CONNECT_DEFAULT);

  /* compute the nodes structure */
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  /* update the values of phi at the new time-step */
  phi_tmp.resize(nodes_np1->indep_nodes.elem_count);
  advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp, p4est_np1, nodes_np1, qnnn);

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

  Vec phi_np1;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  double *f;
  ierr = VecGetArray(phi_np1, &f); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
    f[n] = phi_tmp[n+nodes_np1->offset_owned_indeps];
  ierr = VecRestoreArray(phi_np1, &f); CHKERRXX(ierr);

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  return dt;
}





void SemiLagrangian::update_p4est(Vec &phi, my_p4est_node_neighbors_t& qnnn){

  // make a new copy of p4est object -- we are going to modify p4est but we
  // still need the old one ...
  p4est_connectivity_t *conn = p4est_->connectivity;
  p4est_t *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);

  // define an interpolating function
  InterpolatingFunction bif(p4est_, nodes_, ghost_, myb_, &qnnn);
  bif.set_input_parameters(phi, quadratic_non_oscillatory);

  // now refine/coarsen the new copy of p4est -- note that we need to swap
  // level-set function since it has moved
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)(p4est_->user_pointer);
  data->phi = &bif;
  p4est_np1->user_pointer = data;
  p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_cf, NULL);
  p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);
  p4est_partition(p4est_np1, NULL);

  // compute new ghost layer
  p4est_ghost_t *ghost_np1 = p4est_ghost_new(p4est_np1, P4EST_CONNECT_DEFAULT);

  // now compute a new node data structure
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

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

  // now that everything is updated, get rid of old stuff and swap them with
  // new ones
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;
}

