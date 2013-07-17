#include "serial_semi_lagrangian.h"
#include "bilinear_interpolating_function.h"
#include <src/refine_coarsen.h>

namespace serial{
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

  // loop over all nodes
  std::vector<double> phi_np1(phi.size());
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
    phi_np1[ni] = linear_interpolation(phi, xy_departure, tree_idx);
  }

  // copy the new values into the old vector
  std::copy(phi_np1.begin(), phi_np1.end(), phi.begin());

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
