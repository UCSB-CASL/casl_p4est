#ifdef P4_TO_P8
#include "my_p8est_node_neighbors.h"
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_node_neighbors.h"
#include <src/my_p4est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/ipm_logging.h>

// logging variable -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_node_neighbors_t;
extern PetscLogEvent log_my_p4est_node_neighbors_t_dd_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_block;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_1st_derivatives_central_block;
extern PetscLogEvent log_my_p4est_node_neighbors_t_1st_derivatives_central;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

/*void my_p4est_node_neighbors_t::init_neighbors(const bool &set_and_store_linear_interpolators, const bool &set_and_store_second_derivatives_operators,
                                               const bool &set_and_store_gradient_operator, const bool &set_and_store_quadratic_interpolators)*/
void my_p4est_node_neighbors_t::init_neighbors()
{
  if (is_initialized) return;

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t, 0, 0, 0, 0); CHKERRXX(ierr);
  neighbors.resize(nodes->indep_nodes.elem_count);

#ifdef CASL_THROWS
  is_qnnn_valid.resize(nodes->indep_nodes.elem_count);
#endif

  /* construct qnnn for ALL nodes. Note that we will not throw if a qnnn is not valid for
   * a node, e.g. a ghost node that is part of the last layer ghost cells. Instead, we set
   * a flag and postpone the actual throw if the user actually tries to access this qnnn.
   */
  for ( size_t n = 0; n < nodes->indep_nodes.elem_count; n++) {
#ifdef CASL_THROWS
    is_qnnn_valid[n] = !construct_neighbors(n, neighbors[n]);
    /* is_qnnn_valid[n] = !construct_neighbors(n, neighbors[n], set_and_store_linear_interpolators, set_and_store_second_derivatives_operators, set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
#else
    construct_neighbors(n, neighbors[n]);
    /* construct_neighbors(n, neighbors[n], set_and_store_linear_interpolators, set_and_store_second_derivatives_operators, set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
#endif
  }

  is_initialized = true;
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::clear_neighbors()
{
  neighbors.clear();
#ifdef CASL_THROWS
  is_qnnn_valid.clear();
#endif
  is_initialized = false;
}

/*void my_p4est_node_neighbors_t::update_all_but_hierarchy(p4est_t *p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, const bool &set_and_store_linear_interpolators, const bool &set_and_store_second_derivatives_operators,
                                                                                                const bool &set_and_store_gradient_operator, const bool &set_and_store_quadratic_interpolators)*/
void my_p4est_node_neighbors_t::update_all_but_hierarchy(p4est_t *p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_)
{
  c_ngbd.update(hierarchy);
  p4est = p4est_;
  ghost = ghost_;
  nodes = nodes_;

  if (is_initialized){
    clear_neighbors();
    init_neighbors();
    /*init_neighbors(set_and_store_linear_interpolators, set_and_store_second_derivatives_operators, set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
  }

  layer_nodes.clear();
  local_nodes.clear();

  set_layer_and_local_nodes();
}

/*void my_p4est_node_neighbors_t::update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_, const bool &set_and_store_linear_interpolators, const bool &set_and_store_second_derivatives_operators,
                                       const bool &set_and_store_gradient_operator, const bool &set_and_store_quadratic_interpolators)*/
void my_p4est_node_neighbors_t::update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
{
  hierarchy = hierarchy_;
  periodic = hierarchy_->get_periodicity();
  update_all_but_hierarchy(hierarchy_->p4est, hierarchy_->ghost, nodes_);
  /*update_all_but_hierarchy(p4est_, ghost_, nodes_, set_and_store_linear_interpolators, set_and_store_second_derivatives_operators, set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
}

/*void my_p4est_node_neighbors_t::update(p4est_t *p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, const bool &set_and_store_linear_interpolators, const bool &set_and_store_second_derivatives_operators,
                                       const bool &set_and_store_gradient_operator, const bool &set_and_store_quadratic_interpolators)*/
void my_p4est_node_neighbors_t::update(p4est_t *p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_)
{
  hierarchy->update(p4est_, ghost_);
  update_all_but_hierarchy(p4est_, ghost_, nodes_);
  /*update_all_but_hierarchy(p4est_, ghost_, nodes_, set_and_store_linear_interpolators, set_and_store_second_derivatives_operators, set_and_store_gradient_operator, set_and_store_quadratic_interpolators);*/
}

/*bool my_p4est_node_neighbors_t::construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t &qnnn, const bool &set_and_store_linear_interpolators, const bool &set_and_store_second_derivatives_operators,
                                                    const bool &set_and_store_gradient_operator, const bool &set_and_store_quadratic_interpolators) const*/
bool my_p4est_node_neighbors_t::construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t &qnnn) const
{
  p4est_connectivity_t *connectivity = p4est->connectivity;
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n);


  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double xmin = v2c[3*t2v[0 + 0] + 0];
  double ymin = v2c[3*t2v[0 + 0] + 1];
  double xmax = v2c[3*t2v[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1] + 0];
  double ymax = v2c[3*t2v[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1] + 1];
#ifdef P4_TO_P8
  double zmin = v2c[3*t2v[0 + 0] + 2];
  double zmax = v2c[3*t2v[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1] + 2];
#endif

  // need to unclamp the node to make sure we get the correct coordinate
  p4est_indep_t node_unclamped = *node;
  p4est_node_unclamp((p4est_quadrant_t*)&node_unclamped);

  p4est_topidx_t tree_id = node->p.piggy3.which_tree;
  p4est_topidx_t v_m = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

  double tree_xmin = connectivity->vertices[3*v_m + 0];
  double tree_ymin = connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
  double tree_zmin = connectivity->vertices[3*v_m + 2];
#endif

  double tree_xmax = connectivity->vertices[3*v_p + 0];
  double tree_ymax = connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmax = connectivity->vertices[3*v_p + 2];
#endif

  double x = (tree_xmax-tree_xmin)*(node_unclamped.x / (double) P4EST_ROOT_LEN) + tree_xmin;
  double y = (tree_ymax-tree_ymin)*(node_unclamped.y / (double) P4EST_ROOT_LEN) + tree_ymin;
#ifdef P4_TO_P8
  double z = (tree_zmax-tree_zmin)*(node_unclamped.z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif

  p4est_locidx_t quad_mmm_idx; p4est_topidx_t tree_mmm_idx;
  p4est_locidx_t quad_mpm_idx; p4est_topidx_t tree_mpm_idx;
  p4est_locidx_t quad_pmm_idx; p4est_topidx_t tree_pmm_idx;
  p4est_locidx_t quad_ppm_idx; p4est_topidx_t tree_ppm_idx;
#ifdef P4_TO_P8
  p4est_locidx_t quad_mmp_idx; p4est_topidx_t tree_mmp_idx;
  p4est_locidx_t quad_mpp_idx; p4est_topidx_t tree_mpp_idx;
  p4est_locidx_t quad_pmp_idx; p4est_topidx_t tree_pmp_idx;
  p4est_locidx_t quad_ppp_idx; p4est_topidx_t tree_ppp_idx;
#endif

  /* NOTE: The following function calls will throw if qnnn is not found. We will catch these
   * for ghost nodes inside the init method. For local nodes, we do not do anything and let the
   * higher level try-block handle the exception.
   */
  find_neighbor_cell_of_node(n, DIM(-1, -1, -1), quad_mmm_idx, tree_mmm_idx); if (quad_mmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, DIM(-1,  1, -1), quad_mpm_idx, tree_mpm_idx); if (quad_mpm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, DIM( 1, -1, -1), quad_pmm_idx, tree_pmm_idx); if (quad_pmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, DIM( 1,  1, -1), quad_ppm_idx, tree_ppm_idx); if (quad_ppm_idx == NOT_A_P4EST_QUADRANT) return true;
#ifdef P4_TO_P8
  find_neighbor_cell_of_node(n,     -1, -1,  1, quad_mmp_idx, tree_mmp_idx); if (quad_mmp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,     -1,  1,  1, quad_mpp_idx, tree_mpp_idx); if (quad_mpp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,      1, -1,  1, quad_pmp_idx, tree_pmp_idx); if (quad_pmp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,      1,  1,  1, quad_ppp_idx, tree_ppp_idx); if (quad_ppp_idx == NOT_A_P4EST_QUADRANT) return true;
#endif

  /* create dummy root quadrant */
  p4est_quadrant_t root;
  root.level = -1; root.x = 0; root.y = 0;
#ifdef P4_TO_P8
  root.z = 0;
#endif

  /* fetch the quadrants */
  p4est_quadrant_t *quad_mmm;
  p4est_quadrant_t *quad_mpm;
  p4est_quadrant_t *quad_pmm;
  p4est_quadrant_t *quad_ppm;
#ifdef P4_TO_P8
  p4est_quadrant_t *quad_mmp;
  p4est_quadrant_t *quad_mpp;
  p4est_quadrant_t *quad_pmp;
  p4est_quadrant_t *quad_ppp;
#endif

  if(quad_mmm_idx == NOT_A_VALID_QUADRANT)
    quad_mmm = &root;
  else
  {
    if(quad_mmm_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mmm_idx);
      quad_mmm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mmm_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_mmm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mmm_idx-p4est->local_num_quadrants);
  }

  if(quad_mpm_idx == NOT_A_VALID_QUADRANT)
    quad_mpm = &root;
  else
  {
    if(quad_mpm_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mpm_idx);
      quad_mpm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mpm_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_mpm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mpm_idx-p4est->local_num_quadrants);
  }

  if(quad_pmm_idx == NOT_A_VALID_QUADRANT)
    quad_pmm = &root;
  else
  {
    if(quad_pmm_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_pmm_idx);
      quad_pmm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_pmm_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_pmm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_pmm_idx-p4est->local_num_quadrants);
  }

  if(quad_ppm_idx == NOT_A_VALID_QUADRANT)
    quad_ppm = &root;
  else
  {
    if(quad_ppm_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_ppm_idx);
      quad_ppm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_ppm_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
    {
      quad_ppm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_ppm_idx-p4est->local_num_quadrants);
    }
  }
#ifdef P4_TO_P8
  if(quad_mmp_idx == NOT_A_VALID_QUADRANT)
    quad_mmp = &root;
  else
  {
    if(quad_mmp_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mmp_idx);
      quad_mmp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mmp_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_mmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mmp_idx-p4est->local_num_quadrants);
  }

  if(quad_mpp_idx == NOT_A_VALID_QUADRANT)
    quad_mpp = &root;
  else
  {
    if(quad_mpp_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mpp_idx);
      quad_mpp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mpp_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_mpp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mpp_idx-p4est->local_num_quadrants);
  }

  if(quad_pmp_idx == NOT_A_VALID_QUADRANT)
    quad_pmp = &root;
  else
  {
    if(quad_pmp_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_pmp_idx);
      quad_pmp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_pmp_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
      quad_pmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_pmp_idx-p4est->local_num_quadrants);
  }

  if(quad_ppp_idx == NOT_A_VALID_QUADRANT)
    quad_ppp = &root;
  else
  {
    if(quad_ppp_idx<p4est->local_num_quadrants)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_ppp_idx);
      quad_ppp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_ppp_idx-tree->quadrants_offset);
    }
    else /* in the ghost layer */
    {
      quad_ppp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_ppp_idx-p4est->local_num_quadrants);
    }
  }
#endif

  qnnn.nodes = nodes;
  qnnn.node_000 = n;

  /* m00 */
#ifdef P4_TO_P8
  if(quad_mmm != &root || quad_mpm != &root || quad_mmp != &root || quad_mpp != &root)
#else
  if(quad_mmm != &root || quad_mpm != &root)
#endif
  {
    p4est_quadrant_t *quad_m00  = quad_mmm;
    p4est_locidx_t quad_m00_idx = quad_mmm_idx;
    p4est_topidx_t tree_m00_idx = tree_mmm_idx;

    if (quad_m00->level < quad_mpm->level) { quad_m00 = quad_mpm; quad_m00_idx = quad_mpm_idx; tree_m00_idx = tree_mpm_idx; }
#ifdef P4_TO_P8
    if (quad_m00->level < quad_mmp->level) { quad_m00 = quad_mmp; quad_m00_idx = quad_mmp_idx; tree_m00_idx = tree_mmp_idx; }
    if (quad_m00->level < quad_mpp->level) { quad_m00 = quad_mpp; quad_m00_idx = quad_mpp_idx; tree_m00_idx = tree_mpp_idx; }
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_m00_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_m00_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_m00 = (tree_xmax-tree_xmin)*(P4EST_QUADRANT_LEN(quad_m00->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_m00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mmm];
    qnnn.node_m00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mpm];
#ifdef P4_TO_P8
    qnnn.node_m00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mmp];
    qnnn.node_m00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mpp];
#endif

    double qy = (tree_ymax-tree_ymin)*(quad_m00->y / (double) P4EST_ROOT_LEN) + tree_ymin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_m00->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_m00->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_m00_m0 = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_m00_m0 = ((fabs(qnnn.d_m00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_m00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_m00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_m00_0m = ((fabs(qnnn.d_m00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_m00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m): 0.0);
#endif
  }

  /* p00 */
#ifdef P4_TO_P8
  if(quad_pmm != &root || quad_ppm != &root || quad_pmp != &root || quad_ppp != &root)
#else
  if(quad_pmm != &root || quad_ppm != &root)
#endif
  {
    p4est_quadrant_t *quad_p00  = quad_pmm;
    p4est_locidx_t quad_p00_idx = quad_pmm_idx;
    p4est_topidx_t tree_p00_idx = tree_pmm_idx;

    if (quad_p00->level < quad_ppm->level) { quad_p00 = quad_ppm; quad_p00_idx = quad_ppm_idx; tree_p00_idx = tree_ppm_idx; }
#ifdef P4_TO_P8
    if (quad_p00->level < quad_pmp->level) { quad_p00 = quad_pmp; quad_p00_idx = quad_pmp_idx; tree_p00_idx = tree_pmp_idx; }
    if (quad_p00->level < quad_ppp->level) { quad_p00 = quad_ppp; quad_p00_idx = quad_ppp_idx; tree_p00_idx = tree_ppp_idx; }
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_p00_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_p00_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_p00 = (tree_xmax-tree_xmin)*(P4EST_QUADRANT_LEN(quad_p00->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_p00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_pmm];
    qnnn.node_p00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_p00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_pmp];
    qnnn.node_p00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_ppp];
#endif

    double qy = (tree_ymax-tree_ymin)*(quad_p00->y / (double) P4EST_ROOT_LEN) + tree_ymin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_p00->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_p00->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_p00_m0 = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_p00_m0 = ((fabs(qnnn.d_p00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_p00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_p00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_p00_0m = ((fabs(qnnn.d_p00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_p00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m): 0.0);
#endif
  }

  /* 0m0 */
#ifdef P4_TO_P8
  if(quad_mmm != &root || quad_pmm != &root || quad_mmp != &root || quad_pmp != &root)
#else
  if(quad_mmm != &root || quad_pmm != &root)
#endif
  {
    p4est_quadrant_t *quad_0m0  = quad_mmm;
    p4est_locidx_t quad_0m0_idx = quad_mmm_idx;
    p4est_topidx_t tree_0m0_idx = tree_mmm_idx;

    if (quad_0m0->level < quad_pmm->level) { quad_0m0 = quad_pmm; quad_0m0_idx = quad_pmm_idx; tree_0m0_idx = tree_pmm_idx; }
#ifdef P4_TO_P8
    if (quad_0m0->level < quad_mmp->level) { quad_0m0 = quad_mmp; quad_0m0_idx = quad_mmp_idx; tree_0m0_idx = tree_mmp_idx; }
    if (quad_0m0->level < quad_pmp->level) { quad_0m0 = quad_pmp; quad_0m0_idx = quad_pmp_idx; tree_0m0_idx = tree_pmp_idx; }
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0m0_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0m0_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_0m0 = (tree_ymax-tree_ymin)*(P4EST_QUADRANT_LEN(quad_0m0->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_0m0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_mmm];
    qnnn.node_0m0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_pmm];
#ifdef P4_TO_P8
    qnnn.node_0m0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_mmp];
    qnnn.node_0m0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_pmp];
#endif

    double qx = (tree_xmax-tree_xmin)*(quad_0m0->x / (double) P4EST_ROOT_LEN) + tree_xmin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_0m0->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_0m0->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_0m0_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0m0_m0 = ((fabs(qnnn.d_0m0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0m0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0m0_0m = ((fabs(qnnn.d_0m0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0m0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m): 0.0);
#endif

  }

  /* 0p0 */
#ifdef P4_TO_P8
  if(quad_mpm != &root || quad_ppm != &root || quad_mpp != &root || quad_ppp != &root)
#else
  if(quad_mpm != &root || quad_ppm != &root)
#endif
  {
    p4est_quadrant_t *quad_0p0  = quad_mpm;
    p4est_locidx_t quad_0p0_idx = quad_mpm_idx;
    p4est_topidx_t tree_0p0_idx = tree_mpm_idx;

    if (quad_0p0->level < quad_ppm->level) { quad_0p0 = quad_ppm; quad_0p0_idx = quad_ppm_idx; tree_0p0_idx = tree_ppm_idx; }
#ifdef P4_TO_P8
    if (quad_0p0->level < quad_mpp->level) { quad_0p0 = quad_mpp; quad_0p0_idx = quad_mpp_idx; tree_0p0_idx = tree_mpp_idx; }
    if (quad_0p0->level < quad_ppp->level) { quad_0p0 = quad_ppp; quad_0p0_idx = quad_ppp_idx; tree_0p0_idx = tree_ppp_idx; }
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0p0_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0p0_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_0p0 = (tree_ymax-tree_ymin)*(P4EST_QUADRANT_LEN(quad_0p0->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_0p0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_mpm];
    qnnn.node_0p0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_0p0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_mpp];
    qnnn.node_0p0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_ppp];
#endif

    double qx = (tree_xmax-tree_xmin)*(quad_0p0->x / (double) P4EST_ROOT_LEN) + tree_xmin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_0p0->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_0p0->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_0p0_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0p0_m0 = ((fabs(qnnn.d_0p0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0p0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0p0_0m = ((fabs(qnnn.d_0p0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0p0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m): 0.0);
#endif
  }

#ifdef P4_TO_P8
  /* 00m */
  if(quad_mmm != &root || quad_pmm != &root || quad_mpm != &root || quad_ppm != &root)
  {
    p4est_quadrant_t *quad_00m  = quad_mmm;
    p4est_locidx_t quad_00m_idx = quad_mmm_idx;
    p4est_topidx_t tree_00m_idx = tree_mmm_idx;

    if (quad_00m->level < quad_pmm->level) { quad_00m = quad_pmm; quad_00m_idx = quad_pmm_idx; tree_00m_idx = tree_pmm_idx; }
    if (quad_00m->level < quad_mpm->level) { quad_00m = quad_mpm; quad_00m_idx = quad_mpm_idx; tree_00m_idx = tree_mpm_idx; }
    if (quad_00m->level < quad_ppm->level) { quad_00m = quad_ppm; quad_00m_idx = quad_ppm_idx; tree_00m_idx = tree_ppm_idx; }

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00m_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00m_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];

    qnnn.d_00m = (tree_zmax-tree_zmin)*(P4EST_QUADRANT_LEN(quad_00m->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_00m_mm = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_mmm];
    qnnn.node_00m_pm = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_pmm];
    qnnn.node_00m_mp = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_mpm];
    qnnn.node_00m_pp = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_ppm];

    double qx = (tree_xmax-tree_xmin)*(quad_00m->x / (double) P4EST_ROOT_LEN) + tree_xmin;
    double qy = (tree_ymax-tree_ymin)*(quad_00m->y / (double) P4EST_ROOT_LEN) + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_00m->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00m_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00m_m0 = ((fabs(qnnn.d_00m_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00m_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0): 0.0);
    qnnn.d_00m_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00m_0m = ((fabs(qnnn.d_00m_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00m_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m): 0.0);
  }

  /* 00p */
  if(quad_mmp != &root || quad_pmp != &root || quad_mpp != &root || quad_ppp != &root)
  {
    p4est_quadrant_t *quad_00p  = quad_mmp;
    p4est_locidx_t quad_00p_idx = quad_mmp_idx;
    p4est_topidx_t tree_00p_idx = tree_mmp_idx;

    if (quad_00p->level < quad_pmp->level) { quad_00p = quad_pmp; quad_00p_idx = quad_pmp_idx; tree_00p_idx = tree_pmp_idx; }
    if (quad_00p->level < quad_mpp->level) { quad_00p = quad_mpp; quad_00p_idx = quad_mpp_idx; tree_00p_idx = tree_mpp_idx; }
    if (quad_00p->level < quad_ppp->level) { quad_00p = quad_ppp; quad_00p_idx = quad_ppp_idx; tree_00p_idx = tree_ppp_idx; }

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00p_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00p_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];

    qnnn.d_00p = (tree_zmax-tree_zmin)*(P4EST_QUADRANT_LEN(quad_00p->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_00p_mm = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_mmp];
    qnnn.node_00p_pm = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_pmp];
    qnnn.node_00p_mp = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_mpp];
    qnnn.node_00p_pp = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_ppp];

    double qx = (tree_xmax-tree_xmin)*(quad_00p->x / (double) P4EST_ROOT_LEN) + tree_xmin;
    double qy = (tree_ymax-tree_ymin)*(quad_00p->y / (double) P4EST_ROOT_LEN) + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_00p->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00p_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00p_m0 = ((fabs(qnnn.d_00p_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00p_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0):0.0);
    qnnn.d_00p_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00p_0m = ((fabs(qnnn.d_00p_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00p_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m): 0.0);
  }
#endif

  /* now do the special case when the node is on an edge of the domain, i.e. 2 roots in this direction */
  /* correcting for wall in the m00 direction */
#ifdef P4_TO_P8
  if(quad_mmm==&root && quad_mpm==&root && quad_mmp==&root && quad_mpp==&root)
#else
  if(quad_mmm==&root && quad_mpm==&root)
#endif
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = 1, cj = -1;
#ifdef P4_TO_P8
    short ck = -1;
#endif


    /* NOTE: First we find the smallest cell in the p00 direction that is neighbor
     * to this node. However, since we later need to fetch the second neighbors
     * of that quadrant (i.e. quad_min below) we have to require that quad_min
     * be local to the processor otherwise the second neighbor might not exist.
     * This can happen, for instance, if quad_min happens to be a ghost quadrant.
     * Unfortunately there is no way around this unless p4est provides us with
     * second layer of ghost cells. However, since this correction is only applied
     * at the walls when imposing neumann bc, the effects should really be minimal
     *
     */

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;
    if (quad_min->level < quad_pmm->level) {
      quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; cj = -1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
    if (quad_min->level < quad_ppm->level) {
      quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; cj =  1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
#ifdef P4_TO_P8
    if (quad_min->level < quad_pmp->level)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; cj = -1; ck =  1; }
    if (quad_min->level < quad_ppp->level)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; cj =  1; ck =  1; }
#endif
#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the p00 direction when correcting for wall in m00 direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif
    const bool di = 1;
    const bool dj = cj != 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, DIM(ci, cj, ck), quad_tmp_idx, tree_tmp_idx); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_m00 = - qnnn.d_p00 - (tree_xmax-tree_xmin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_m00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
    qnnn.node_m00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_m00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
    qnnn.node_m00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];
#endif

    double qy = (tree_ymax-tree_ymin)*(quad_tmp->y / (double) P4EST_ROOT_LEN) + tree_ymin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_tmp->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_m00_m0 = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_m00_m0 = ((fabs(qnnn.d_m00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_m00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_m00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_m00_0m = ((fabs(qnnn.d_m00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_m00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m): 0.0);
#endif
  }

  /* correcting for wall in the p00 direction */
#ifdef P4_TO_P8
  if(quad_pmm==&root && quad_ppm==&root && quad_pmp==&root && quad_ppp==&root)
#else
  if(quad_pmm==&root && quad_ppm==&root)
#endif
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1;
#ifdef P4_TO_P8
    short ck = -1;
#endif

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;

    if (quad_min->level < quad_mmm->level) {
      quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; cj = -1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
    if (quad_min->level < quad_mpm->level) {
      quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; cj =  1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mmp->level)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; cj = -1; ck =  1; }
    if (quad_min->level < quad_mpp->level)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; cj =  1; ck =  1; }
#endif
#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the m00 direction when correcting for wall in p00 direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif

    const bool di = 0;
    const bool dj = cj != 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, DIM(ci, cj, ck), quad_tmp_idx, tree_tmp_idx); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_p00 = - qnnn.d_m00 - (tree_xmax-tree_xmin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_p00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_p00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
#ifdef P4_TO_P8
    qnnn.node_p00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_p00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
#endif

    double qy = (tree_ymax-tree_ymin)*(quad_tmp->y / (double) P4EST_ROOT_LEN) + tree_ymin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_tmp->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_p00_m0 = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_p00_m0 = ((fabs(qnnn.d_p00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_p00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_p00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_p00_0m = ((fabs(qnnn.d_p00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_p00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m): 0.0);
#endif
  }

  /* correcting for wall in the 0m0 direction */
#ifdef P4_TO_P8
  if(quad_mmm==&root && quad_pmm==&root && quad_mmp==&root && quad_pmp==&root)
#else
  if(quad_mmm==&root && quad_pmm==&root)
#endif
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = 1;
#ifdef P4_TO_P8
    short ck = -1;
#endif

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;

    if (quad_min->level < quad_mpm->level) {
      quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; ci = -1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
    if (quad_min->level < quad_ppm->level) {
      quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; ci =  1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mpp->level)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; ci = -1; ck =  1; }
    if (quad_min->level < quad_ppp->level)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; ci =  1; ck =  1; }
#endif

#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the 0p0 direction when correcting for wall in 0m0 direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif

    const bool di = ci != 1;
    const bool dj = 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, DIM(ci, cj, ck), quad_tmp_idx, tree_tmp_idx); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_0m0 = - qnnn.d_0p0 - (tree_ymax-tree_ymin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_0m0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
    qnnn.node_0m0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_0m0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
    qnnn.node_0m0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];
#endif

    double qx = (tree_xmax-tree_xmin)*(quad_tmp->x / (double) P4EST_ROOT_LEN) + tree_xmin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_tmp->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_0m0_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0m0_m0 = ((fabs(qnnn.d_0m0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0m0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0):0.0);
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0m0_0m = ((fabs(qnnn.d_0m0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0m0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m): 0.0);
#endif
  }

  /* correcting for wall in the 0p0 direction */
#ifdef P4_TO_P8
  if(quad_mpm==&root && quad_ppm==&root && quad_mpp==&root && quad_ppp==&root)
#else
  if(quad_mpm==&root && quad_ppm==&root)
#endif
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1;
#ifdef P4_TO_P8
    short ck = -1;
#endif

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;

    if (quad_min->level < quad_mmm->level) {
      quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; ci = -1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
    if (quad_min->level < quad_pmm->level) {
      quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; ci =  1;
#ifdef P4_TO_P8
      ck = -1;
#endif
    }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mmp->level)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; ci = -1; ck =  1; }
    if (quad_min->level < quad_pmp->level)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; ci =  1; ck =  1; }
#endif

#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the 0m0 direction when correcting for wall in 0p0 direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif

    const bool di = ci != 1;
    const bool dj = 0;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, DIM(ci, cj, ck), quad_tmp_idx, tree_tmp_idx); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];
#endif

    qnnn.d_0p0 = - qnnn.d_0m0 - (tree_ymax-tree_ymin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_0p0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_0p0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
#ifdef P4_TO_P8
    qnnn.node_0p0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_0p0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
#endif

    double qx = (tree_xmax-tree_xmin)*(quad_tmp->x / (double) P4EST_ROOT_LEN) + tree_xmin;
#ifdef P4_TO_P8
    double qz = (tree_zmax-tree_zmin)*(quad_tmp->z / (double) P4EST_ROOT_LEN) + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_0p0_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0p0_m0 = ((fabs(qnnn.d_0p0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0p0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) : 0.0);
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(periodic[2] && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0p0_0m = ((fabs(qnnn.d_0p0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0p0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m): 0.0);
#endif
  }

#ifdef P4_TO_P8
  /* correcting for wall in the 00m direction  (Only in 3D) */
  if(quad_mmm==&root && quad_pmm==&root && quad_mpm==&root && quad_ppm==&root)
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = 1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;

    if (quad_min->level < quad_mmp->level)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; ci = -1; cj = -1; }
    if (quad_min->level < quad_pmp->level)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; ci =  1; cj = -1; }
    if (quad_min->level < quad_mpp->level)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; ci = -1; cj =  1; }
    if (quad_min->level < quad_ppp->level)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; ci =  1; cj =  1; }

#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the 00p direction when correcting for wall in 00m direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif

    const bool di = ci != 1;
    const bool dj = cj != 1;
    const bool dk = 1;

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];

    qnnn.d_00m = - qnnn.d_00p - (tree_zmax-tree_zmin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_00m_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_00m_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
    qnnn.node_00m_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
    qnnn.node_00m_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];

    double qx = (tree_xmax-tree_xmin)*(quad_tmp->x / (double) P4EST_ROOT_LEN) + tree_xmin;
    double qy = (tree_ymax-tree_ymin)*(quad_tmp->y / (double) P4EST_ROOT_LEN) + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00m_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00m_m0 = ((fabs(qnnn.d_00m_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00m_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) : 0.0);
    qnnn.d_00m_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00m_0m = ((fabs(qnnn.d_00m_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00m_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m) : 0.0);
  }

  /* correcting for wall in the 00p direction  (Only in 3D) */
  if(quad_mmp==&root && quad_pmp==&root && quad_mpp==&root && quad_ppp==&root)
  {
    /* fetch the second order neighbor to the right */
    p4est_locidx_t node_tmp_idx;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = -1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = NOT_A_VALID_QUADRANT;

    if (quad_min->level < quad_mmm->level)
    { quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; ci = -1; cj = -1; }
    if (quad_min->level < quad_pmm->level)
    { quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; ci =  1; cj = -1; }
    if (quad_min->level < quad_mpm->level)
    { quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; ci = -1; cj =  1; }
    if (quad_min->level < quad_ppm->level)
    { quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; ci =  1; cj =  1; }

#ifdef CASL_THROWS
    if (quad_min_idx == NOT_A_VALID_QUADRANT)
      throw std::runtime_error("[ERROR]: could not find a neighboring cell in the 00m direction when correcting for wall in 00p direction."
                               " This is either a bug in 'my_p4est_hierarchy_t' or that the entire p4est is a single cell!");
#endif

    const bool di = ci != 1;
    const bool dj = cj != 1;
    const bool dk = 0;

    node_tmp_idx = nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di];
    find_neighbor_cell_of_node(node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    p4est_topidx_t v_ppp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + P4EST_CHILDREN-1];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_xmax = connectivity->vertices[3*v_ppp + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double tree_ymax = connectivity->vertices[3*v_ppp + 1];
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double tree_zmax = connectivity->vertices[3*v_ppp + 2];

    qnnn.d_00p = - qnnn.d_00m - (tree_zmax-tree_zmin)*(P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN);
    qnnn.node_00p_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_00p_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
    qnnn.node_00p_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
    qnnn.node_00p_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];

    double qx = (tree_xmax-tree_xmin)*(quad_tmp->x / (double) P4EST_ROOT_LEN) + tree_xmin;
    double qy = (tree_ymax-tree_ymin)*(quad_tmp->y / (double) P4EST_ROOT_LEN) + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00p_m0 = ((fabs(x - qx) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (x - qx) : 0.0);
    if(periodic[0] && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00p_m0 = ((fabs(qnnn.d_00p_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00p_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0):0.0);
    qnnn.d_00p_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(periodic[1] && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00p_0m = ((fabs(qnnn.d_00p_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00p_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m): 0.0);
  }
#endif

  qnnn.inverse_d_max = MAX(qnnn.d_m00, qnnn.d_p00);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_0m0, qnnn.d_0p0);
#ifdef P4_TO_P8
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_00m, qnnn.d_00p);
#endif
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_m00_m0, qnnn.d_m00_p0);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_p00_m0, qnnn.d_p00_p0);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_0m0_m0, qnnn.d_0m0_p0);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_0p0_m0, qnnn.d_0p0_p0);
#ifdef P4_TO_P8
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_m00_0m, qnnn.d_m00_0p);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_p00_0m, qnnn.d_p00_0p);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_0m0_0m, qnnn.d_0m0_0p);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_0p0_0m, qnnn.d_0p0_0p);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_00m_m0, qnnn.d_00m_p0);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_00m_0m, qnnn.d_00p_0p);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_00p_m0, qnnn.d_00p_p0);
  qnnn.inverse_d_max = MAX(qnnn.inverse_d_max, qnnn.d_00p_0m, qnnn.d_00p_0p);
#endif
  qnnn.inverse_d_max = 1.0/qnnn.inverse_d_max;

  /*
  if(set_and_store_linear_interpolators)
    qnnn.set_and_store_linear_interpolators();
  if(set_and_store_second_derivatives_operators)
    qnnn.set_and_store_second_derivative_operators();
  if(set_and_store_gradient_operator)
    qnnn.set_and_store_gradient_operator();
  if(set_and_store_quadratic_interpolators)
    qnnn.set_and_store_quadratic_interpolators();
  */

  return false;
}

void my_p4est_node_neighbors_t::dd_central(const Vec f[], Vec fdd[], const unsigned int& n_vecs, const unsigned char& der) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_dd_central, f, fdd, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(n_vecs > 0);

  quad_neighbor_nodes_of_node_t::assert_fields_and_blocks( n_vecs );

#ifdef CASL_THROWS
  {
    Vec f_l, fdd_l;
    PetscInt f_size, fdd_size;


    for (unsigned int k = 0; k < n_vecs; ++k) {
      // Get local form
      ierr = VecGhostGetLocalForm(f[k],   &f_l  );  CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fdd[k], &fdd_l);  CHKERRXX(ierr);

      // Get sizes
      ierr = VecGetSize(f_l,   &f_size);            CHKERRXX(ierr);
      ierr = VecGetSize(fdd_l, &fdd_size);          CHKERRXX(ierr);

      if (f_size != fdd_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fdd_size = " << fdd_size << std::endl;

        throw std::invalid_argument(oss.str());
      }
      if(f_size != ((PetscInt)nodes->indep_nodes.elem_count))
      {
        std::ostringstream oss;
        oss << "[ERROR]: the local size of the ghosted vectors must be equald to the number of grid nodes (including ghosts)"
            << " f_size = " << f_size << " nodes->indep_nodes.elem_count = " << ((PetscInt)nodes->indep_nodes.elem_count) << std::endl;

        throw std::invalid_argument(oss.str());
      }

      // Restore local form
      ierr = VecGhostRestoreLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fdd[k], &fdd_l); CHKERRXX(ierr);
    }
  }
#endif

  // get access to the iternal data
  double *f_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fdd_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArray(f[k],   &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes
    for (int layer_node : layer_nodes)
      for (unsigned int k = 0; k < n_vecs; ++k)
        fdd_p[k][layer_node] = neighbors[layer_node].dd_central(der, f_p[k]);

    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (int local_node : local_nodes)
      for (unsigned int k = 0; k < n_vecs; ++k)
        fdd_p[k][local_node] = neighbors[local_node].dd_central(der, f_p[k]);
  } else {
    quad_neighbor_nodes_of_node_t qnnn{};

    // compute the derivatives on the boundary nodes
    for (int layer_node : layer_nodes){
      get_neighbors(layer_node, qnnn);
      for (unsigned int k = 0; k < n_vecs; ++k)
        fdd_p[k][layer_node] = qnnn.dd_central(der, f_p[k]);
    }

    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (int local_node : local_nodes){
      get_neighbors(local_node, qnnn);
      for (unsigned int k = 0; k < n_vecs; ++k)
        fdd_p[k][local_node] = qnnn.dd_central(der, f_p[k]);
    }
  }

  // restore internal data
  for (unsigned int k = 0; k < n_vecs; ++k){
    ierr = VecRestoreArray(f[k],   &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
  }

  // finish the ghost update process to ensure all values are updated
  for (unsigned int k = 0; k < n_vecs; ++k)
    ierr = VecGhostUpdateEnd(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_dd_central, f, fdd, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f[], Vec fdd[], const unsigned int& n_vecs, const unsigned int& bs_f) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central_block, f, fdd, 0, 0); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t::assert_fields_and_blocks( n_vecs, bs_f );

#ifdef CASL_THROWS
  {

    Vec f_l, fdd_l;
    PetscInt f_size, fdd_size, block_size;

    for (unsigned int k = 0; k < n_vecs; ++k) {
      // Get local form
      ierr = VecGhostGetLocalForm(f[k],   &f_l  );  CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fdd[k], &fdd_l);  CHKERRXX(ierr);

      // Get sizes
      ierr = VecGetSize(f_l,   &f_size);            CHKERRXX(ierr);
      ierr = VecGetSize(fdd_l, &fdd_size);          CHKERRXX(ierr);
      ierr = VecGetBlockSize(f[k], &block_size);    CHKERRXX(ierr);

      if (block_size != ((PetscInt) bs_f)){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in f does not match the given block size bs_f"
            << " block_size = " << block_size << " bs_f = " << bs_f << std::endl;

        throw std::invalid_argument(oss.str());
      }

      if (f_size*P4EST_DIM != fdd_size){
        std::ostringstream oss;
        oss << "[ERROR]: The vectors of derivatives must be P4EST_DIM times larger than the differentiated fields"
            << " P4EST_DIM*f_size = " << P4EST_DIM*f_size << " fdd_size = " << fdd_size << std::endl;

        throw std::invalid_argument(oss.str());
      }

      // Restore local form
      ierr = VecGhostRestoreLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fdd[k], &fdd_l); CHKERRXX(ierr);
    }
  }
#endif

  P4EST_ASSERT(bs_f > 0);
  // get access to the iternal data
  const double *f_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  double *fdd_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArrayRead(f[k], &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes
    for (int node_idx : layer_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs_f == 1)
        qnnn.laplace_insert_in_block_vectors(f_p, fdd_p, n_vecs);
      else
        qnnn.laplace_all_components_insert_in_block_vectors(f_p, fdd_p, n_vecs, bs_f);
    }

    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k) {
      ierr = VecGhostUpdateBegin(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs_f == 1)
        qnnn.laplace_insert_in_block_vectors(f_p, fdd_p, n_vecs);
      else
        qnnn.laplace_all_components_insert_in_block_vectors(f_p, fdd_p, n_vecs, bs_f);
    }
  } else {
    quad_neighbor_nodes_of_node_t qnnn{};

    // compute the derivatives on the boundary nodes
    for (int node_idx : layer_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs_f == 1)
        qnnn.laplace_insert_in_block_vectors(f_p, fdd_p, n_vecs);
      else
        qnnn.laplace_all_components_insert_in_block_vectors(f_p, fdd_p, n_vecs, bs_f);
    }

    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k) {
      ierr = VecGhostUpdateBegin(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    // compute the derivatives for all internal nodes
    for (int node_idx : local_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs_f == 1)
        qnnn.laplace_insert_in_block_vectors(f_p, fdd_p, n_vecs);
      else
        qnnn.laplace_all_components_insert_in_block_vectors(f_p, fdd_p, n_vecs, bs_f);
    }
  }

  for (unsigned int k = 0; k < n_vecs; ++k) {
    // restore internal data
    ierr = VecRestoreArrayRead(f[k], &f_p[k]  ); CHKERRXX(ierr);
    // finish the ghost update process to ensure all values are updated
    ierr = VecGhostUpdateEnd(fdd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    // restore internal data
    ierr = VecRestoreArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central_block, f, fdd, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f[], DIM(Vec fxx[], Vec fyy[], Vec fzz[]), const unsigned int& n_vecs, const unsigned int &bs) const
{ // this is being used rn
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("2nd_derivatives");

  quad_neighbor_nodes_of_node_t::assert_fields_and_blocks( n_vecs, bs );

#ifdef CASL_THROWS
  {
    Vec f_l, DIM(fxx_l, fyy_l, fzz_l);
    PetscInt f_size, DIM(fxx_size, fyy_size, fzz_size), bs_f, DIM(bs_xx, bs_yy, bs_zz);

    for (unsigned int k = 0; k < n_vecs; ++k) {
      // Get local form
      ierr = VecGhostGetLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fxx[k], &fxx_l); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fyy[k], &fyy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostGetLocalForm(fzz[k], &fzz_l); CHKERRXX(ierr);
#endif

      // Get sizes
      ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
      ierr = VecGetSize(fxx_l, &fxx_size); CHKERRXX(ierr);
      ierr = VecGetSize(fyy_l, &fyy_size); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetSize(fzz_l, &fzz_size); CHKERRXX(ierr);
#endif
      // Get block sizes
      ierr = VecGetBlockSize(f_l,   &bs_f);   CHKERRXX(ierr);
      ierr = VecGetBlockSize(fxx_l, &bs_xx); CHKERRXX(ierr);
      ierr = VecGetBlockSize(fyy_l, &bs_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetBlockSize(fzz_l, &bs_zz); CHKERRXX(ierr);
#endif

      if (f_size != fxx_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fxx_size = " << fxx_size << std::endl;

        throw std::invalid_argument(oss.str());
      }

      if (f_size != fyy_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fyy_size = " << fyy_size << std::endl;

        throw std::invalid_argument(oss.str());
      }

#ifdef P4_TO_P8
      if (f_size != fzz_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fzz_size = " << fzz_size << std::endl;

        throw std::invalid_argument(oss.str());
      }
#endif

      if (((PetscInt) bs) != bs_f){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in f does not match the given block size bs"
            << " bs_f = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

      if (((PetscInt) bs) != bs_xx){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fxx does not match the given block size bs"
            << " bs_xx = " << bs_xx << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

      if (((PetscInt) bs) != bs_yy){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fyy does not match the given block size bs"
            << " bs_yy = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

#ifdef P4_TO_P8
      if (((PetscInt) bs) != bs_zz){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fzz does not match the given block size bs"
            << " bs_zz = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }
#endif

      // Restore local form
      ierr = VecGhostRestoreLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fxx[k], &fxx_l); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fyy[k], &fyy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostRestoreLocalForm(fzz[k], &fzz_l); CHKERRXX(ierr);
#endif
    }
  }
#endif
  P4EST_ASSERT(bs > 0);
#ifdef DXX_USE_BLOCKS
  second_derivatives_central_using_block(f, DIM(fxx, fyy, fzz), n_vecs, bs);
#else // !DXX_USE_BLOCKS
  // get access to the internal data
  const double *f_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  double DIM(*fxx_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fyy_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fzz_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT]);
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArrayRead(f[k],  &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fxx[k],    &fxx_p[k]); CHKERRXX(ierr);
    ierr = VecGetArray(fyy[k],    &fyy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(fzz[k],    &fzz_p[k]); CHKERRXX(ierr);
#endif
  }
  if (is_initialized){
    // compute the derivatives on the boundary nodes -- fxx
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs == 1)
        qnnn.dxx_central_insert_in_vectors(f_p, fxx_p, n_vecs);
      else
        qnnn.dxx_central_all_components_insert_in_vectors(f_p, fxx_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fxx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fyy
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs == 1)
        qnnn.dyy_central_insert_in_vectors(f_p, fyy_p, n_vecs);
      else
        qnnn.dyy_central_all_components_insert_in_vectors(f_p, fyy_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fyy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

#ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fzz
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs == 1)
        qnnn.dzz_central_insert_in_vectors(f_p, fzz_p, n_vecs);
      else
        qnnn.dzz_central_all_components_insert_in_vectors(f_p, fzz_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fzz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs == 1)
        qnnn.laplace_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs);
      else
        qnnn.laplace_all_components_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs, bs);
    }
  }
  else {
    quad_neighbor_nodes_of_node_t qnnn{};

    // compute the derivatives on the boundary nodes -- fxx
    for (int node_idx : layer_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs == 1)
        qnnn.laplace_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs);
      else
        qnnn.laplace_all_components_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k) {
      ierr = VecGhostUpdateBegin(fxx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(fyy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostUpdateBegin(fzz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    }

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs == 1)
        qnnn.laplace_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs);
      else
        qnnn.laplace_all_components_insert_in_vectors(f_p, DIM(fxx_p, fyy_p, fzz_p), n_vecs, bs);
    }
  }

  // restore internal data
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecRestoreArrayRead(f[k], &f_p[k]  ); CHKERRXX(ierr);
    // finish the ghost update process to ensure all values are updated
    ierr = VecGhostUpdateEnd(fyy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(fxx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateEnd(fzz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(fxx[k], &fxx_p[k]); CHKERRXX(ierr);
    ierr = VecRestoreArray(fyy[k], &fyy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(fzz[k], &fzz_p[k]); CHKERRXX(ierr);
#endif
  }
#endif // !DXX_USE_BLOCKS


  IPMLogRegionEnd("2nd_derivatives");
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::first_derivatives_central(const Vec f[], Vec fd[], const unsigned int& n_vecs, const unsigned int& bs_f) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_1st_derivatives_central_block, f, fd, 0, 0); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t::assert_fields_and_blocks( n_vecs, bs_f );

  bool fd_is_ghosted[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  bool no_fd_is_ghosted = true;
  for (unsigned int k = 0; k < n_vecs; ++k)
  {
    Vec fdk_loc;
    ierr = VecGhostGetLocalForm(fd[k], &fdk_loc); CHKERRXX(ierr);
    fd_is_ghosted[k] = fdk_loc != nullptr;
    no_fd_is_ghosted = no_fd_is_ghosted && !fd_is_ghosted[k];
    ierr = VecGhostRestoreLocalForm(fd[k], &fdk_loc); CHKERRXX(ierr);
#ifdef P4EST_DEBUG
    // So,
    // vectors in f MUST be ghosted (otherwise, it's impossible to compute derivatives for layer nodes) :
    P4EST_ASSERT(VecIsSetForNodes(f[k], nodes, p4est->mpicomm, bs_f));
    // vectors in fd may or may not be ghosted but must be of blocksize P4EST_DIM*bs_f -- you may not have a local representation for that vector (if it's not ghosted) :
    P4EST_ASSERT(VecIsSetForNodes(fd[k], nodes, p4est->mpicomm, bs_f*P4EST_DIM, fdk_loc != NULL));
#endif
  }
  P4EST_ASSERT(bs_f > 0);

  // get access to the iternal data
  const double *f_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  double *fd_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArrayRead(f[k],  &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fd[k],     &fd_p[k]); CHKERRXX(ierr);
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes
    for (int node_idx : layer_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs_f == 1)
        qnnn.gradient_insert_in_block_vectors(f_p, fd_p, n_vecs);
      else
        qnnn.gradient_all_components_insert_in_block_vectors(f_p, fd_p, n_vecs, bs_f);
    }

    // start updating the ghost values
    if(!no_fd_is_ghosted)
      for (unsigned int k = 0; k < n_vecs; ++k)
        if(fd_is_ghosted[k]){
          ierr = VecGhostUpdateBegin(fd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      if(bs_f == 1)
        qnnn.gradient_insert_in_block_vectors(f_p, fd_p, n_vecs);
      else
        qnnn.gradient_all_components_insert_in_block_vectors(f_p, fd_p, n_vecs, bs_f);
    }
  } else {
    quad_neighbor_nodes_of_node_t qnnn{};

    // compute the derivatives on the boundary nodes
    for (int node_idx : layer_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs_f == 1)
        qnnn.gradient_insert_in_block_vectors(f_p, fd_p, n_vecs);
      else
        qnnn.gradient_all_components_insert_in_block_vectors(f_p, fd_p, n_vecs, bs_f);
    }

    // start updating the ghost values
    if(!no_fd_is_ghosted)
      for (unsigned int k = 0; k < n_vecs; ++k)
        if(fd_is_ghosted[k]){
          ierr = VecGhostUpdateBegin(fd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      get_neighbors(node_idx, qnnn);
      if(bs_f == 1)
        qnnn.gradient_insert_in_block_vectors(f_p, fd_p, n_vecs);
      else
        qnnn.gradient_all_components_insert_in_block_vectors(f_p, fd_p, n_vecs, bs_f);
    }
  }

  for (unsigned int k = 0; k < n_vecs; ++k) {
    // restore internal data
    ierr = VecRestoreArrayRead(f[k], &f_p[k]  ); CHKERRXX(ierr);
    // finish the ghost update process to ensure all values are updated
    if(fd_is_ghosted[k]){
      ierr = VecGhostUpdateEnd(fd[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
    // restore internal data
    ierr = VecRestoreArray(fd[k], &fd_p[k]); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central_block, f, fdd, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::first_derivatives_central(const Vec f[], DIM(Vec fx[], Vec fy[], Vec fz[]), const unsigned int& n_vecs, const unsigned int& bs) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_1st_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("1st_derivatives");

  if( n_vecs > CASL_NUM_SIMULTANEOUS_FIELD_COMPUT )
    throw std::runtime_error( "[CASL_ERROR] You're requesting more simultaneous fields than specified in macro CASL_NUM_SIMULTANEOUS_FIELD_COMPUT" );
  if( bs > CASL_NUM_SIMULTANEOUS_BLOCK_COMPUT )
	throw std::runtime_error( "[CASL_ERROR] You're requesting more simultaneous blocks than specified in macro CASL_NUM_SIMULTANEOUS_BLOCK_COMPUT" );

#ifdef CASL_THROWS
  {
    Vec f_l, fx_l, fy_l;
    PetscInt f_size, fx_size, fy_size, bs_f, bs_x, bs_y;
#ifdef P4_TO_P8
    Vec fz_l;
    PetscInt fz_size, bs_z;
#endif

    for (unsigned int k = 0; k < n_vecs; ++k) {
      // Get local form
      ierr = VecGhostGetLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fx[k], &fx_l); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(fy[k], &fy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostGetLocalForm(fz[k], &fz_l); CHKERRXX(ierr);
#endif

      // Get sizes
      ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
      ierr = VecGetSize(fx_l, &fx_size); CHKERRXX(ierr);
      ierr = VecGetSize(fy_l, &fy_size); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetSize(fz_l, &fz_size); CHKERRXX(ierr);
#endif
      // Get block sizes
      ierr = VecGetBlockSize(f_l,   &bs_f);   CHKERRXX(ierr);
      ierr = VecGetBlockSize(fx_l, &bs_x); CHKERRXX(ierr);
      ierr = VecGetBlockSize(fy_l, &bs_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetBlockSize(fz_l, &bs_z); CHKERRXX(ierr);
#endif

      if (f_size != fx_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fx_size = " << fx_size << std::endl;

        throw std::invalid_argument(oss.str());
      }

      if (f_size != fy_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fy_size = " << fy_size << std::endl;

        throw std::invalid_argument(oss.str());
      }

#ifdef P4_TO_P8
      if (f_size != fz_size){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fz_size = " << fz_size << std::endl;

        throw std::invalid_argument(oss.str());
      }
#endif

      if (((PetscInt) bs) != bs_f){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in f does not match the given block size bs"
            << " bs_f = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

      if (((PetscInt) bs) != bs_x){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fx does not match the given block size bs"
            << " bs_x = " << bs_x << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

      if (((PetscInt) bs) != bs_y){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fy does not match the given block size bs"
            << " bs_y = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }

#ifdef P4_TO_P8
      if (((PetscInt) bs) != bs_z){
        std::ostringstream oss;
        oss << "[ERROR]: the block size of a vector in fz does not match the given block size bs"
            << " bs_z = " << bs_f << " bs = " << bs << std::endl;
        throw std::invalid_argument(oss.str());
      }
#endif

      // Restore local form
      ierr = VecGhostRestoreLocalForm(f[k],   &f_l  ); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fx[k], &fx_l); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(fy[k], &fy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostRestoreLocalForm(fz[k], &fz_l); CHKERRXX(ierr);
#endif
    }
  }
#endif
  P4EST_ASSERT(bs > 0);

  // get access to the iternal data
  const double *f_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  double DIM(*fx_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fy_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], double *fz_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT]);
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArrayRead(f[k],  &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fx[k],     &fx_p[k]); CHKERRXX(ierr);
    ierr = VecGetArray(fy[k],     &fy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(fz[k],     &fz_p[k]); CHKERRXX(ierr);
#endif
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes -- fx
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      (bs == 1) ? qnnn.dx_central_insert_in_vectors(f_p, fx_p, n_vecs) : qnnn.dx_central_all_components_insert_in_vectors(f_p, fx_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fy
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      (bs == 1) ? qnnn.dy_central_insert_in_vectors(f_p, fy_p, n_vecs) : qnnn.dy_central_all_components_insert_in_vectors(f_p, fy_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

#ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fz
    for (int node_idx : layer_nodes)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      (bs == 1) ? qnnn.dz_central_insert_in_vectors(f_p, fz_p, n_vecs) : qnnn.dz_central_all_components_insert_in_vectors(f_p, fz_p, n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[node_idx];
      (bs == 1) ? qnnn.gradient_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs) : qnnn.gradient_all_components_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs, bs);
    }
  } else {

    quad_neighbor_nodes_of_node_t qnnn{};

    // compute the derivatives on the boundary nodes -- fxx
    for (int node_idx : layer_nodes){
      get_neighbors(node_idx, qnnn);
      (bs == 1) ? qnnn.gradient_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs) : qnnn.gradient_all_components_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs, bs);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k) {
      ierr = VecGhostUpdateBegin(fx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(fy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGhostUpdateBegin(fz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    }

    // compute the derivaties for all internal nodes
    for (int node_idx : local_nodes){
      get_neighbors(node_idx, qnnn);
      (bs == 1) ? qnnn.gradient_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs) : qnnn.gradient_all_components_insert_in_vectors(f_p, DIM(fx_p, fy_p, fz_p), n_vecs, bs);
    }
  }

  // restore internal data
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecRestoreArrayRead(f[k],  &f_p[k]  ); CHKERRXX(ierr);
    // finish the ghost update process to ensure all values are updated
    ierr = VecGhostUpdateEnd(fy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(fx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateEnd(fz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(fx[k], &fx_p[k]); CHKERRXX(ierr);
    ierr = VecRestoreArray(fy[k], &fy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(fz[k], &fz_p[k]); CHKERRXX(ierr);
#endif
  }

  IPMLogRegionEnd("1st_derivatives");
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_1st_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::second_derivatives_central_using_block(const Vec f[], DIM(Vec fxx[], Vec fyy[], Vec fzz[]), const unsigned int& n_vecs, const unsigned int &bs) const
{
  quad_neighbor_nodes_of_node_t::assert_fields_and_blocks( n_vecs, bs );

  // create temporary block vector
  PetscErrorCode ierr;
  Vec fdd[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT];
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecCreateGhostNodesBlock(p4est, nodes, int(bs)*P4EST_DIM, &fdd[k]); CHKERRXX(ierr);
  }

  // compute derivatives using block vector
  second_derivatives_central(f, fdd, n_vecs, bs);

  // copy data back into original vectors
  double *fdd_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT],
  	DIM(*fxx_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fyy_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT], *fzz_p[CASL_NUM_SIMULTANEOUS_FIELD_COMPUT]);
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
    ierr = VecGetArray(fxx[k], &fxx_p[k]); CHKERRXX(ierr);
    ierr = VecGetArray(fyy[k], &fyy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(fzz[k], &fzz_p[k]); CHKERRXX(ierr);
#endif
  }

  // compute the derivatives on the boundary nodes
  for (size_t i = 0; i < nodes->indep_nodes.elem_count; i++){
    for (unsigned int k = 0; k < n_vecs; ++k) {
      for (unsigned int comp = 0; comp < bs; ++comp) {
        fxx_p[k][bs*i+comp] = fdd_p[k][P4EST_DIM*(bs*i+comp) + 0];
        fyy_p[k][bs*i+comp] = fdd_p[k][P4EST_DIM*(bs*i+comp) + 1];
#ifdef P4_TO_P8
        fzz_p[k][bs*i+comp] = fdd_p[k][P4EST_DIM*(bs*i+comp) + 2];
#endif
      }
    }
  }

  // restore internal data
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecRestoreArray(fdd[k], &fdd_p[k]); CHKERRXX(ierr);
    ierr = VecRestoreArray(fxx[k], &fxx_p[k]); CHKERRXX(ierr);
    ierr = VecRestoreArray(fyy[k], &fyy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(fzz[k], &fzz_p[k]); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(fdd[k]); CHKERRXX(ierr);
  }
}

void my_p4est_node_neighbors_t::get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists) const
{
  for (int i = 0; i < num_neighbors_cube; i++) {
    neighbors[i] = -1;
  }

  // find neighboring quadrants
  p4est_locidx_t quad_mmm_idx; p4est_topidx_t tree_mmm_idx;
  p4est_locidx_t quad_mpm_idx; p4est_topidx_t tree_mpm_idx;
  p4est_locidx_t quad_pmm_idx; p4est_topidx_t tree_pmm_idx;
  p4est_locidx_t quad_ppm_idx; p4est_topidx_t tree_ppm_idx;
#ifdef P4_TO_P8
  p4est_locidx_t quad_mmp_idx; p4est_topidx_t tree_mmp_idx;
  p4est_locidx_t quad_mpp_idx; p4est_topidx_t tree_mpp_idx;
  p4est_locidx_t quad_pmp_idx; p4est_topidx_t tree_pmp_idx;
  p4est_locidx_t quad_ppp_idx; p4est_topidx_t tree_ppp_idx;
#endif

  find_neighbor_cell_of_node(n, DIM(-1, -1, -1), quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  find_neighbor_cell_of_node(n, DIM(-1,  1, -1), quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  find_neighbor_cell_of_node(n, DIM( 1, -1, -1), quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  find_neighbor_cell_of_node(n, DIM( 1,  1, -1), quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
#ifdef P4_TO_P8
  find_neighbor_cell_of_node(n,     -1, -1,  1, quad_mmp_idx, tree_mmp_idx); //nei_quads[dir::v_mmp] = quad_mmp_idx;
  find_neighbor_cell_of_node(n,     -1,  1,  1, quad_mpp_idx, tree_mpp_idx); //nei_quads[dir::v_mpp] = quad_mpp_idx;
  find_neighbor_cell_of_node(n,      1, -1,  1, quad_pmp_idx, tree_pmp_idx); //nei_quads[dir::v_pmp] = quad_pmp_idx;
  find_neighbor_cell_of_node(n,      1,  1,  1, quad_ppp_idx, tree_ppp_idx); //nei_quads[dir::v_ppp] = quad_ppp_idx;
#endif

  // find neighboring nodes
#ifdef P4_TO_P8
  neighbors[nn_000] = n;

  // m00
  if      (quad_mmm_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp];
  else if (quad_mpm_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmp];
  else if (quad_mmp_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
  else if (quad_mpp_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmm];

  // p00
  if      (quad_pmm_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppp];
  else if (quad_ppm_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmp];
  else if (quad_pmp_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppm];
  else if (quad_ppp_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

  // 0m0
  if      (quad_mmm_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp];
  else if (quad_pmm_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmp];
  else if (quad_mmp_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
  else if (quad_pmp_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmm];

  // 0p0
  if      (quad_mpm_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppp];
  else if (quad_ppm_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpp];
  else if (quad_mpp_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppm];
  else if (quad_ppp_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

  // 00m
  if      (quad_mmm_idx >= 0) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];
  else if (quad_pmm_idx >= 0) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mpm];
  else if (quad_mpm_idx >= 0) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_pmm];
  else if (quad_ppm_idx >= 0) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mmm];

  // 00p
  if      (quad_mmp_idx >= 0) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_ppp];
  else if (quad_pmp_idx >= 0) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mpp];
  else if (quad_mpp_idx >= 0) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_pmp];
  else if (quad_ppp_idx >= 0) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

  // 0mm
  if      (quad_mmm_idx >= 0) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx >= 0) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];
  // 0pm
  if      (quad_mpm_idx >= 0) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];
  else if (quad_ppm_idx >= 0) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  // 0mp
  if      (quad_mmp_idx >= 0) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
  else if (quad_pmp_idx >= 0) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmp];
  // 0pp
  if      (quad_mpp_idx >= 0) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppp];
  else if (quad_ppp_idx >= 0) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

  // m0m
  if      (quad_mmm_idx >= 0) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx >= 0) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];
  // p0m
  if      (quad_pmm_idx >= 0) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx >= 0) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];
  // m0p
  if      (quad_mmp_idx >= 0) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
  else if (quad_mpp_idx >= 0) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmp];
  // p0p
  if      (quad_pmp_idx >= 0) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppp];
  else if (quad_ppp_idx >= 0) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

  // mm0
  if      (quad_mmm_idx >= 0) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmp];
  else if (quad_mmp_idx >= 0) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
  // pm0
  if      (quad_pmm_idx >= 0) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmp];
  else if (quad_pmp_idx >= 0) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
  // mp0
  if      (quad_mpm_idx >= 0) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpp];
  else if (quad_mpp_idx >= 0) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
  // pp0
  if      (quad_ppm_idx >= 0) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppp];
  else if (quad_ppp_idx >= 0) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

  // mmm
  if      (quad_mmm_idx >= 0) neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  // pmm
  if      (quad_pmm_idx >= 0) neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  // mpm
  if      (quad_mpm_idx >= 0) neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  // ppm
  if      (quad_ppm_idx >= 0) neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

  // mmp
  if      (quad_mmp_idx >= 0) neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
  // pmp
  if      (quad_pmp_idx >= 0) neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
  // mpp
  if      (quad_mpp_idx >= 0) neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
  // ppp
  if      (quad_ppp_idx >= 0) neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];
#else
  neighbors[nn_000] = n;

  if      (quad_mmm_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx >= 0) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];

  if      (quad_pmm_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx >= 0) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

  if      (quad_mmm_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx >= 0) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];

  if      (quad_ppm_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  else if (quad_mpm_idx >= 0) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];

  if      (quad_mmm_idx >= 0) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  if      (quad_pmm_idx >= 0) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  if      (quad_mpm_idx >= 0) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  if      (quad_ppm_idx >= 0) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif

  if(neighbor_exists != NULL)
    for (int i = 0; i < num_neighbors_cube; i++)
      neighbor_exists[i] = (neighbors[i] >= 0);

  return;
}


void my_p4est_node_neighbors_t::get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors) const
{
  for (int i = 0; i < num_neighbors_cube; i++) {
    neighbors[i] = -1;
  }

  // find neighboring quadrants
  p4est_locidx_t quad_mmm_idx; p4est_topidx_t tree_mmm_idx;
  p4est_locidx_t quad_mpm_idx; p4est_topidx_t tree_mpm_idx;
  p4est_locidx_t quad_pmm_idx; p4est_topidx_t tree_pmm_idx;
  p4est_locidx_t quad_ppm_idx; p4est_topidx_t tree_ppm_idx;
#ifdef P4_TO_P8
  p4est_locidx_t quad_mmp_idx; p4est_topidx_t tree_mmp_idx;
  p4est_locidx_t quad_mpp_idx; p4est_topidx_t tree_mpp_idx;
  p4est_locidx_t quad_pmp_idx; p4est_topidx_t tree_pmp_idx;
  p4est_locidx_t quad_ppp_idx; p4est_topidx_t tree_ppp_idx;
#endif

#ifdef P4_TO_P8
  find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
  find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx); //nei_quads[dir::v_mmp] = quad_mmp_idx;
  find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx); //nei_quads[dir::v_mpp] = quad_mpp_idx;
  find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx); //nei_quads[dir::v_pmp] = quad_pmp_idx;
  find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx); //nei_quads[dir::v_ppp] = quad_ppp_idx;
#else
  find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  find_neighbor_cell_of_node(n, -1, +1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  find_neighbor_cell_of_node(n, +1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  find_neighbor_cell_of_node(n, +1, +1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
#endif
//  P4EST_ASSERT(quad_mmm_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_mpm_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_pmm_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_ppm_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_mmp_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_mpp_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_pmp_idx!=NOT_A_P4EST_QUADRANT);
//  P4EST_ASSERT(quad_ppp_idx!=NOT_A_P4EST_QUADRANT);


  // find neighboring nodes
#ifdef P4_TO_P8
  neighbors[nn_000] = n;

  // m00
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmm];

  // p00
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

  // 0m0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmm];

  // 0p0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

  // 00m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_pmm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mmm];

  // 00p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_ppp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_pmp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

  // 0mm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];
  // 0pm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  // 0mp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmp];
  // 0pp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

  // m0m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];
  // p0m
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];
  // m0p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmp];
  // p0p
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

  // mm0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
  // pm0
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
  // mp0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
  // pp0
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

  // mmm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  // pmm
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  // mpm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  // ppm
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

  // mmp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
  // pmp
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
  // mpp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
  // ppp
  if      (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];
#else
  neighbors[nn_000] = n;

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];

  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];

  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif
}

// Elyce and Rochi merge 11/22/21 -- keeping this function to preserve library functionality from old other classes, even though get_all_neighbors now handles this itself among other things
void my_p4est_node_neighbors_t::fetch_second_degree_node_neighbors_of_interpolation_node(const p4est_locidx_t& node_idx, set_of_local_node_index_t& second_degree_neighbor_nodes) const
{
  set_of_neighboring_quadrants quad_neighbors;
  c_ngbd.gather_neighbor_cells_of_node(node_idx, nodes, quad_neighbors, true);

  second_degree_neighbor_nodes.clear();
  for(set_of_neighboring_quadrants::const_iterator it = quad_neighbors.begin(); it != quad_neighbors.end(); it++)
  {
    const p4est_locidx_t quad_idx = it->p.piggy3.local_num;
    for(u_char vv = 0; vv < P4EST_CHILDREN; vv++)
    {
      p4est_locidx_t neighbor_node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx + vv];
      if(neighbor_node_idx != node_idx)
        second_degree_neighbor_nodes.insert(neighbor_node_idx);
    }
  }
  return;
}
