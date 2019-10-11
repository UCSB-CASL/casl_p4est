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
extern PetscLogEvent log_my_p4est_node_neighbors_t_dxx_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_dyy_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_dzz_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_1st_derivatives_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_above_threshold;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_block;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

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
#else
    construct_neighbors(n, neighbors[n]);
#endif
  }

  is_initialized = true;
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::clear_neighbors()
{
  neighbors.clear();
  is_qnnn_valid.clear();
  is_initialized = false;
}

void my_p4est_node_neighbors_t::update(my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
{
  hierarchy = hierarchy_;
  p4est = hierarchy_->p4est;
  ghost = hierarchy_->ghost;
  nodes = nodes_;

  if (is_initialized){
    clear_neighbors();
    init_neighbors();
  }

  layer_nodes.clear();
  local_nodes.clear();

  layer_nodes.reserve(nodes->num_owned_shared);
  local_nodes.reserve(nodes->num_owned_indeps - nodes->num_owned_shared);

  for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i){
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
    ni->pad8 == 0 ? local_nodes.push_back(i) : layer_nodes.push_back(i);
  }
}

void my_p4est_node_neighbors_t::update(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes)
{
  hierarchy->update(p4est, ghost);

  this->p4est = p4est;
  this->ghost = ghost;
  this->nodes = nodes;

  if (is_initialized){
    clear_neighbors();
    init_neighbors();
  }

  layer_nodes.clear();
  local_nodes.clear();

  layer_nodes.reserve(nodes->num_owned_shared);
  local_nodes.reserve(nodes->num_owned_indeps - nodes->num_owned_shared);

  for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i){
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
    ni->pad8 == 0 ? local_nodes.push_back(i) : layer_nodes.push_back(i);
  }
}

bool my_p4est_node_neighbors_t::construct_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t &qnnn) const
{
  bool p_x = is_periodic(p4est,0);
  bool p_y = is_periodic(p4est,1);
#ifdef P4_TO_P8
  bool p_z = is_periodic(p4est,2);
#endif

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
#ifdef P4_TO_P8
  find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx); if (quad_mmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx); if (quad_mpm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx); if (quad_pmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx); if (quad_ppm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx); if (quad_mmp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx); if (quad_mpp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx); if (quad_pmp_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx); if (quad_ppp_idx == NOT_A_P4EST_QUADRANT) return true;
#else
  find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx); if (quad_mmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n, -1,  1, quad_mpm_idx, tree_mpm_idx); if (quad_mpm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1, -1, quad_pmm_idx, tree_pmm_idx); if (quad_pmm_idx == NOT_A_P4EST_QUADRANT) return true;
  find_neighbor_cell_of_node(n,  1,  1, quad_ppm_idx, tree_ppm_idx); if (quad_ppm_idx == NOT_A_P4EST_QUADRANT) return true;
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
  if(quad_mmm!=&root || quad_mpm!=&root || quad_mmp!=&root || quad_mpp!=&root)
#else
  if(quad_mmm!=&root || quad_mpm!=&root)
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
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_m00_m0 = ((fabs(qnnn.d_m00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_m00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_m00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_m00_0m = ((fabs(qnnn.d_m00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_m00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_m00_0m): 0.0);
#endif
  }

  /* p00 */
#ifdef P4_TO_P8
  if(quad_pmm!=&root || quad_ppm!=&root || quad_pmp!=&root || quad_ppp!=&root)
#else
  if(quad_pmm!=&root || quad_ppm!=&root)
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
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_p00_m0 = ((fabs(qnnn.d_p00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_p00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_p00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_p00_0m = ((fabs(qnnn.d_p00_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_0m + (zmax-zmin)) : 0.0);
    qnnn.d_p00_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_p00_0m): 0.0);
#endif
  }

  /* 0m0 */
#ifdef P4_TO_P8
  if(quad_mmm!=&root || quad_pmm!=&root || quad_mmp!=&root || quad_pmp!=&root)
#else
  if(quad_mmm!=&root || quad_pmm!=&root)
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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0m0_m0 = ((fabs(qnnn.d_0m0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0m0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0m0_0m = ((fabs(qnnn.d_0m0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0m0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0m0_0m): 0.0);
#endif

  }

  /* 0p0 */
#ifdef P4_TO_P8
  if(quad_mpm!=&root || quad_ppm!=&root || quad_mpp!=&root || quad_ppp!=&root)
#else
  if(quad_mpm!=&root || quad_ppm!=&root)
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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0p0_m0 = ((fabs(qnnn.d_0p0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0p0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
      qnnn.d_0p0_0m = ((fabs(qnnn.d_0p0_0m + (zmax-zmin)) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_0m + (zmax-zmin)) : 0.0);
    qnnn.d_0p0_0p = ((fabs((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_zmax-tree_zmin)*qh - qnnn.d_0p0_0m): 0.0);
#endif
  }

#ifdef P4_TO_P8
  /* 00m */
  if(quad_mmm!=&root || quad_pmm!=&root || quad_mpm!=&root || quad_ppm!=&root)
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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00m_m0 = ((fabs(qnnn.d_00m_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00m_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0): 0.0);
    qnnn.d_00m_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00m_0m = ((fabs(qnnn.d_00m_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00m_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00m_0m): 0.0);
  }

  /* 00p */
  if(quad_mmp!=&root || quad_pmp!=&root || quad_mpp!=&root || quad_ppp!=&root)
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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00p_m0 = ((fabs(qnnn.d_00p_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00p_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0):0.0);
    qnnn.d_00p_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
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
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#else
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj,     quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#endif

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
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_m00_m0 = ((fabs(qnnn.d_m00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_m00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_m00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_m00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_m00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
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
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#else
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj,     quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#endif

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
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_p00_m0 = ((fabs(qnnn.d_p00_m0 + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_p00_m0 + (ymax-ymin)) : 0.0);
    qnnn.d_p00_p0 = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_p00_m0): 0.0);
#ifdef P4_TO_P8
    qnnn.d_p00_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
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
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#else
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj,     quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#endif

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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0m0_m0 = ((fabs(qnnn.d_0m0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0m0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0m0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0m0_m0):0.0);
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
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
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#else
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj,     quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;
#endif

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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_0p0_m0 = ((fabs(qnnn.d_0p0_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_0p0_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_0p0_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_0p0_m0) : 0.0);
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = ((fabs(z - qz) > (tree_zmax-tree_zmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (z - qz) : 0.0);
    if(p_z && z < qz-(tree_zmax-tree_zmin)*qh/4.0)
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
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00m_m0 = ((fabs(qnnn.d_00m_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00m_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00m_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00m_m0) : 0.0);
    qnnn.d_00m_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
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
    find_neighbor_cell_of_node( node_tmp_idx, ci, cj, ck, quad_tmp_idx, tree_tmp_idx ); if (quad_tmp_idx == NOT_A_P4EST_QUADRANT) return true;

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
    if(p_x && x < qx-(tree_xmax-tree_xmin)*qh/4.0)
      qnnn.d_00p_m0 = ((fabs(qnnn.d_00p_m0 + (xmax-xmin)) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_m0 + (xmax-xmin)) : 0.0);
    qnnn.d_00p_p0 = ((fabs((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0) > (tree_xmax-tree_xmin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_xmax-tree_xmin)*qh - qnnn.d_00p_m0):0.0);
    qnnn.d_00p_0m = ((fabs(y - qy) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (y - qy) : 0.0);
    if(p_y && y < qy-(tree_ymax-tree_ymin)*qh/4.0)
      qnnn.d_00p_0m = ((fabs(qnnn.d_00p_0m + (ymax-ymin)) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? (qnnn.d_00p_0m + (ymax-ymin)) : 0.0);
    qnnn.d_00p_0p = ((fabs((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m) > (tree_ymax-tree_ymin)*((double) P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))? ((tree_ymax-tree_ymin)*qh - qnnn.d_00p_0m): 0.0);
  }
#endif

  return false;
}

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::find_neighbor_cell_of_node( p4est_locidx_t n, char i, char j, char k, p4est_locidx_t& quad, p4est_topidx_t& nb_tree_idx ) const
{
  // make a local copy of the current node structure
  p4est_indep_t node_struct = *(p4est_indep_t *)sc_array_index(&nodes->indep_nodes, n);
  p4est_indep_t *node = &node_struct;
  p4est_node_unclamp((p4est_quadrant_t*)node);

  p4est_connectivity_t *connectivity = p4est->connectivity;

  p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
  nb_tree_idx = tree_idx;

  p4est_qcoord_t x_perturb = node->x+i;
  p4est_qcoord_t y_perturb = node->y+j;
  p4est_qcoord_t z_perturb = node->z+k;

  bool px = is_periodic(p4est, 0);
  bool py = is_periodic(p4est, 1);
  bool pz = is_periodic(p4est, 2);

  /* There are 26 special cases for a tree in 3D. These are:
   * 8  corners
   * 12 edges
   * 6  faces
   */

  /* corners: we could have done this in a smarter way but to keep things similar
   * to 2D we decompose the movement into three separate moves. For instance, if
   * one is interested in neighboring tree in the mpm direction, we first find the
   * neighborig in the m00, then 0p0, and finally 00m directions.
   */
  /* 0 - mmm */
  if(node->x == 0 && node->y == 0 && node->z == 0 &&
     i == -1      && j == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 1 - pmm */
  else if(node->x == P4EST_ROOT_LEN && node->y == 0 && node->z == 0 &&
          i ==  1                   && j == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 2 - mpm */
  else if(node->x == 0 && node->y == P4EST_ROOT_LEN && node->z == 0 &&
          i == -1      && j ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 3 - ppm */
  else if(node->x == P4EST_ROOT_LEN && node->y == P4EST_ROOT_LEN && node->z == 0 &&
          i ==  1                   && j ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 4 - mmp */
  else if(node->x == 0 && node->y == 0 && node->z == P4EST_ROOT_LEN &&
          i == -1      && j == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 5 - pmp */
  else if(node->x == P4EST_ROOT_LEN && node->y == 0 && node->z == P4EST_ROOT_LEN &&
          i ==  1                   && j == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 6 - mpp */
  else if(node->x == 0 && node->y == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          i == -1      && j ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
    z_perturb = 1;
  }
  /* 7 - ppp */
  else if(node->x == P4EST_ROOT_LEN && node->y == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          i ==  1                   && j ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[3];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = 1;
    z_perturb = 1;
  }

  /* Now check the edges of the tree where we use the same idea as before. Note
   * that here an edge is recognized by moving in only two directions. For instance
   * and edge movement of m0p requires movement first in m00 and next in the 00p
   * directions but is not concerned with movement in either 0m0 or 0p0 directions
   */
  /* 0 - mm0 */
  else if(node->x == 0 && node->y == 0 &&
          i == -1      && j == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 1 - pm0 */
  else if(node->x == P4EST_ROOT_LEN && node->y == 0 &&
          i ==  1                   && j == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 2 - mp0 */
  else if(node->x == 0 && node->y == P4EST_ROOT_LEN &&
          i == -1      && j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
  }
  /* 3 - pp0 */
  else if(node->x == P4EST_ROOT_LEN && node->y == P4EST_ROOT_LEN &&
          i ==  1                   && j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    y_perturb = 1;
  }
  /* 4 - m0m */
  else if(node->x == 0 && node->z == 0 &&
          i == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 5 - p0m */
  else if(node->x == P4EST_ROOT_LEN && node->z == 0 &&
          i ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 6 - m0p */
  else if(node->x == 0 && node->z == P4EST_ROOT_LEN &&
          i == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 7 - p0p */
  else if(node->x == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          i ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
    z_perturb = 1;
  }
  /* 8 - 0mm */
  else if(node->y == 0 && node->z == 0 &&
          j == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 9 - 0pm */
  else if(node->y == P4EST_ROOT_LEN && node->z == 0 &&
          j ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 10 - 0mp */
  else if(node->y == 0 && node->z == P4EST_ROOT_LEN &&
          j == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 11 - 0pp */
  else if(node->y == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          j ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = 1;
    z_perturb = 1;
  }

  /* finally faces. these are the easy single movement cases */
  /* 0 - m00 */
  else if(node->x == 0 &&
          i == -1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 1 - p00 */
  else if(node->x == P4EST_ROOT_LEN &&
          i ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(!px && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    x_perturb = 1;
  }
  /* 2 - 0m0 */
  else if(node->y == 0 &&
          j == -1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 3 - 0p0 */
  else if(node->y == P4EST_ROOT_LEN &&
          j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(!py && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    y_perturb = 1;
  }
  /* 4 - 00m */
  else if(node->z == 0 &&
          k == -1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_00m]; // 00m
    if(!pz && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 5 - 0p0 */
  else if(node->z == P4EST_ROOT_LEN &&
          k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_00p]; // 00p
    if(!pz && tmp_tree_idx[0] == tree_idx)        { quad = NOT_A_VALID_QUADRANT; return; }

    z_perturb = 1;
  }

  /* now find the the cell by searching the hierarchy */
  int ind = 0;
  while(hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF)
  {
    p4est_qcoord_t size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool ci = ( x_perturb >= hierarchy->trees[nb_tree_idx][ind].imin + size );
    bool cj = ( y_perturb >= hierarchy->trees[nb_tree_idx][ind].jmin + size );
    bool ck = ( z_perturb >= hierarchy->trees[nb_tree_idx][ind].kmin + size );
    ind = hierarchy->trees[nb_tree_idx][ind].child + 4*ck + 2*cj + ci;
  }

  quad = hierarchy->trees[nb_tree_idx][ind].quad;  
}

#else
void my_p4est_node_neighbors_t::find_neighbor_cell_of_node( p4est_locidx_t n, char i, char j, p4est_locidx_t& quad, p4est_topidx_t& nb_tree_idx ) const
{
  // make a copy of the current node
  p4est_indep_t node_struct = *(p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
  p4est_indep_t *node = &node_struct;
  p4est_node_unclamp((p4est_quadrant_t*)node);

  p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
  nb_tree_idx = tree_idx;
  p4est_connectivity_t *connectivity = p4est->connectivity;

  p4est_qcoord_t x_perturb = node->x+i;
  p4est_qcoord_t y_perturb = node->y+j;

  bool px = is_periodic(p4est, 0);
  bool py = is_periodic(p4est, 1);

  /* first check the corners of the tree */
  if(node->x==0 && node->y==0 && i==-1 && j==-1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(!px && tmp_tree_idx == tree_idx)    { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
    if(!py && nb_tree_idx == tmp_tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==P4EST_ROOT_LEN && node->y==0 && i== 1 && j==-1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(!px && tmp_tree_idx == tree_idx)    { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
    if(!py && nb_tree_idx == tmp_tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==0 && node->y==P4EST_ROOT_LEN && i==-1 && j== 1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(!px && tmp_tree_idx == tree_idx)    { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
    if(!py && nb_tree_idx == tmp_tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
  }
  else if(node->x==P4EST_ROOT_LEN && node->y==P4EST_ROOT_LEN && i== 1 && j== 1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(!px && tmp_tree_idx == tree_idx)    { quad = NOT_A_VALID_QUADRANT; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
    if(!py && nb_tree_idx == tmp_tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = 1;
    y_perturb = 1;
  }

  /* now check the edges of the tree */
  else if(node->x==0 && i==-1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(!px && nb_tree_idx == tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==P4EST_ROOT_LEN && i==1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(!px && nb_tree_idx == tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    x_perturb = 1;
  }
  else if(node->y==0 && j==-1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
    if(!py && nb_tree_idx == tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->y==P4EST_ROOT_LEN && j==1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
    if(!py && nb_tree_idx == tree_idx) { quad = NOT_A_VALID_QUADRANT; return; }
    y_perturb = 1;
  }

  int ind = 0;
  while(hierarchy->trees[nb_tree_idx][ind].child!=CELL_LEAF)
  {
    p4est_qcoord_t size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool dir_i = ( x_perturb >= hierarchy->trees[nb_tree_idx][ind].imin + size );
    bool dir_j = ( y_perturb >= hierarchy->trees[nb_tree_idx][ind].jmin + size );
    ind = hierarchy->trees[nb_tree_idx][ind].child + 2*dir_j + dir_i;
  }

  quad = hierarchy->trees[nb_tree_idx][ind].quad;
}
#endif

void my_p4est_node_neighbors_t::dxx_central(const Vec f, Vec fxx) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_dxx_central, f, fxx, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  {
    Vec f_l, fxx_l;
    PetscInt f_size, fxx_size;

    // Get local form
    ierr = VecGhostGetLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fxx, &fxx_l); CHKERRXX(ierr);

    // Get sizes
    ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
    ierr = VecGetSize(fxx_l, &fxx_size); CHKERRXX(ierr);

    if (f_size != fxx_size){
      std::ostringstream oss;
      oss << "[ERROR]: Vectors must be of same size when computing derivatives"
          << " f_size = " << f_size << " fxx_size = " << fxx_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fxx, &fxx_l); CHKERRXX(ierr);
  }
#endif

  // get access to the iternal data
  double *f_p, *fxx_p;
  ierr = VecGetArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(fxx, &fxx_p); CHKERRXX(ierr);

  if (is_initialized){
    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++)
      fxx_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dxx_central(f_p);

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++)
      fxx_p[local_nodes[i]] = neighbors[local_nodes[i]].dxx_central(f_p);
  } else {
    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      fxx_p[layer_nodes[i]] = qnnn.dxx_central(f_p);
    }

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);
      fxx_p[local_nodes[i]] = qnnn.dxx_central(f_p);
    }
  }

  // restore internal data
  ierr = VecRestoreArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fxx, &fxx_p); CHKERRXX(ierr);

  // finish the ghost update process to ensure all values are updated
  ierr = VecGhostUpdateEnd(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_dxx_central, f, fxx, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::dyy_central(const Vec f, Vec fyy) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_dyy_central, f, fyy, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  {
    Vec f_l, fyy_l;
    PetscInt f_size, fyy_size;

    // Get local form
    ierr = VecGhostGetLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fyy, &fyy_l); CHKERRXX(ierr);

    // Get sizes
    ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
    ierr = VecGetSize(fyy_l, &fyy_size); CHKERRXX(ierr);

    if (f_size != fyy_size){
      std::ostringstream oss;
      oss << "[ERROR]: Vectors must be of same size when computing derivatives"
          << " f_size = " << f_size << " fyy_size = " << fyy_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fyy, &fyy_l); CHKERRXX(ierr);
  }
#endif

  // get access to the iternal data
  double *f_p, *fyy_p;
  ierr = VecGetArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(fyy, &fyy_p); CHKERRXX(ierr);

  if(is_initialized){
    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++)
      fyy_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dyy_central(f_p);

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++)
      fyy_p[local_nodes[i]] = neighbors[local_nodes[i]].dyy_central(f_p);

  } else {
    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      fyy_p[layer_nodes[i]] = qnnn.dyy_central(f_p);
    }

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);
      fyy_p[local_nodes[i]] = qnnn.dyy_central(f_p);
    }
  }

  // restore internal data
  ierr = VecRestoreArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fyy, &fyy_p); CHKERRXX(ierr);

  // finish the ghost update process to ensure all values are updated
  ierr = VecGhostUpdateEnd(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_dyy_central, f, fyy, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::dzz_central(const Vec f, Vec fzz) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_dzz_central, f, fzz, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  {
    Vec f_l, fzz_l;
    PetscInt f_size, fzz_size;

    // Get local form
    ierr = VecGhostGetLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fzz, &fzz_l); CHKERRXX(ierr);

    // Get sizes
    ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
    ierr = VecGetSize(fzz_l, &fzz_size); CHKERRXX(ierr);

    if (f_size != fzz_size){
      std::ostringstream oss;
      oss << "[ERROR]: Vectors must be of same size when computing derivatives"
          << " f_size = " << f_size << " fzz_size = " << fzz_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fzz, &fzz_l); CHKERRXX(ierr);
  }
#endif

  // get access to the iternal data
  double *f_p, *fzz_p;
  ierr = VecGetArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(fzz, &fzz_p); CHKERRXX(ierr);

  if (is_initialized){
  // compute the derivatives on the boundary nodes
  for (size_t i=0; i<layer_nodes.size(); i++)
    fzz_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dzz_central(f_p);

  // start updating the ghost values
  ierr = VecGhostUpdateBegin(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // compute the derivaties for all internal nodes
  for (size_t i=0; i<local_nodes.size(); i++)
    fzz_p[local_nodes[i]] = neighbors[local_nodes[i]].dzz_central(f_p);

  } else {
    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      fzz_p[layer_nodes[i]] = qnnn.dzz_central(f_p);
    }

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);
      fzz_p[local_nodes[i]] = qnnn.dzz_central(f_p);
    }
  }

  // restore internal data
  ierr = VecRestoreArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fzz, &fzz_p); CHKERRXX(ierr);

  // finish the ghost update process to ensure all values are updated
  ierr = VecGhostUpdateEnd(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_dzz_central, f, fzz, 0, 0); CHKERRXX(ierr);
}
#endif

void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f, Vec fdd) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central_block, f, fdd, 0, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  {
    Vec f_l, fdd_l;
    PetscInt f_size, fdd_size, block_size;

    // Get local form
    ierr = VecGhostGetLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fdd, &fdd_l); CHKERRXX(ierr);

    // Get sizes
    ierr = VecGetSize(f_l,   &f_size);        CHKERRXX(ierr);
    ierr = VecGetSize(fdd_l, &fdd_size);      CHKERRXX(ierr);
    ierr = VecGetBlockSize(fdd, &block_size); CHKERRXX(ierr);

    if (f_size*block_size != fdd_size){
      std::ostringstream oss;
      oss << "[ERROR]: Vectors must be of same size when computing derivatives"
          << " f_size = " << f_size << " fdd_size = " << fdd_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    if (P4EST_DIM != block_size){
      std::ostringstream oss;
      oss << "[ERROR]: output vector 'fdd' must be a block vector os block size "
          << P4EST_DIM << " but block_size = " << block_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fdd, &fdd_l); CHKERRXX(ierr);
  }
#endif

  // get access to the iternal data
  double *f_p, *fdd_p;
  ierr = VecGetArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(fdd, &fdd_p); CHKERRXX(ierr);

  if (is_initialized){
    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      fdd_p[P4EST_DIM*layer_nodes[i] + 0] = neighbors[layer_nodes[i]].dxx_central(f_p); // fxx
      fdd_p[P4EST_DIM*layer_nodes[i] + 1] = neighbors[layer_nodes[i]].dyy_central(f_p); // fyy
  #ifdef P4_TO_P8
      fdd_p[P4EST_DIM*layer_nodes[i] + 2] = neighbors[layer_nodes[i]].dzz_central(f_p); // fzz
  #endif
    }

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fdd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      fdd_p[P4EST_DIM*local_nodes[i] + 0] = neighbors[local_nodes[i]].dxx_central(f_p); // fxx
      fdd_p[P4EST_DIM*local_nodes[i] + 1] = neighbors[local_nodes[i]].dyy_central(f_p); // fyy
  #ifdef P4_TO_P8
      fdd_p[P4EST_DIM*local_nodes[i] + 2] = neighbors[local_nodes[i]].dzz_central(f_p); // fzz
  #endif
    }

  } else {
    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);

      fdd_p[P4EST_DIM*layer_nodes[i] + 0] = qnnn.dxx_central(f_p); // fxx
      fdd_p[P4EST_DIM*layer_nodes[i] + 1] = qnnn.dyy_central(f_p); // fyy
  #ifdef P4_TO_P8
      fdd_p[P4EST_DIM*layer_nodes[i] + 2] = qnnn.dzz_central(f_p); // fzz
  #endif
    }

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fdd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);

      fdd_p[P4EST_DIM*local_nodes[i] + 0] = qnnn.dxx_central(f_p); // fxx
      fdd_p[P4EST_DIM*local_nodes[i] + 1] = qnnn.dyy_central(f_p); // fyy
  #ifdef P4_TO_P8
      fdd_p[P4EST_DIM*local_nodes[i] + 2] = qnnn.dzz_central(f_p); // fzz
  #endif
    }

  }

  // restore internal data
  ierr = VecRestoreArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fdd, &fdd_p); CHKERRXX(ierr);

  // finish the ghost update process to ensure all values are updated
  ierr = VecGhostUpdateEnd(fdd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central_block, f, fdd, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f[], Vec fxx[], Vec fyy[], Vec fzz[], unsigned int n_vecs) const
#else
void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f[], Vec fxx[], Vec fyy[], unsigned int n_vecs) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("2nd_derivatives");

#ifdef CASL_THROWS
  {
    Vec f_l, fxx_l, fyy_l;
    PetscInt f_size, fxx_size, fyy_size;
#ifdef P4_TO_P8
    Vec fzz_l;
    PetscInt fzz_size;
#endif

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

#ifdef DXX_USE_BLOCKS
#ifdef P4_TO_P8
  second_derivatives_central_using_block(f, fxx, fyy, fzz);
#else
  second_derivatives_central_using_block(f, fxx, fyy);
#endif
#else // !DXX_USE_BLOCKS
  // get access to the iternal data
  double *f_p[n_vecs], *fxx_p[n_vecs], *fyy_p[n_vecs];
#ifdef P4_TO_P8
  double *fzz_p[n_vecs];
#endif
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecGetArray(f[k],   &f_p[k]  ); CHKERRXX(ierr);
    ierr = VecGetArray(fxx[k], &fxx_p[k]); CHKERRXX(ierr);
    ierr = VecGetArray(fyy[k], &fyy_p[k]); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecGetArray(fzz[k], &fzz_p[k]); CHKERRXX(ierr);
  #endif
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes -- fxx
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[layer_nodes[i]];
      for (unsigned int k = 0; k < n_vecs; ++k)
        fxx_p[k][layer_nodes[i]] = qnnn.dxx_central(f_p[k]);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fxx[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fyy
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[layer_nodes[i]];
      for (unsigned int k = 0; k < n_vecs; ++k)
        fyy_p[k][layer_nodes[i]] = qnnn.dyy_central(f_p[k]);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fyy[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  #ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fzz
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[layer_nodes[i]];
      for (unsigned int k = 0; k < n_vecs; ++k)
        fzz_p[k][layer_nodes[i]] = qnnn.dzz_central(f_p[k]);
    }
    // start updating the ghost values
    for (unsigned int k = 0; k < n_vecs; ++k)
      ierr = VecGhostUpdateBegin(fzz[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  #endif

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[local_nodes[i]];
      for (unsigned int k = 0; k < n_vecs; ++k) {
        fxx_p[k][local_nodes[i]] = qnnn.dxx_central(f_p[k]);
        fyy_p[k][local_nodes[i]] = qnnn.dyy_central(f_p[k]);
#ifdef P4_TO_P8
        fzz_p[k][local_nodes[i]] = qnnn.dzz_central(f_p[k]);
#endif
      }
    }

  } else {

    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes -- fxx
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      for (unsigned int k = 0; k < n_vecs; ++k) {
        fxx_p[k][layer_nodes[i]] = qnnn.dxx_central(f_p[k]);
        fyy_p[k][layer_nodes[i]] = qnnn.dyy_central(f_p[k]);
#ifdef P4_TO_P8
        fzz_p[k][layer_nodes[i]] = qnnn.dzz_central(f_p[k]);
#endif
      }
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
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);
      for (unsigned int k = 0; k < n_vecs; ++k) {
        fxx_p[k][local_nodes[i]] = qnnn.dxx_central(f_p[k]);
        fyy_p[k][local_nodes[i]] = qnnn.dyy_central(f_p[k]);
#ifdef P4_TO_P8
        fzz_p[k][local_nodes[i]] = qnnn.dzz_central(f_p[k]);
#endif
      }
    }
  }

  // restore internal data
  for (unsigned int k = 0; k < n_vecs; ++k) {
    ierr = VecRestoreArray(f[k],   &f_p[k]  ); CHKERRXX(ierr);
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

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::second_derivatives_central_above_threshold(const Vec f, double thr, Vec fxx, Vec fyy, Vec fzz) const
#else
void my_p4est_node_neighbors_t::second_derivatives_central_above_threshold(const Vec f, double thr, Vec fxx, Vec fyy) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central_above_threshold, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("2nd_derivatives");

#ifdef CASL_THROWS
  {
    Vec f_l, fxx_l, fyy_l;
    PetscInt f_size, fxx_size, fyy_size;
#ifdef P4_TO_P8
    Vec fzz_l;
    PetscInt fzz_size;
#endif

    // Get local form
    ierr = VecGhostGetLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fxx, &fxx_l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(fyy, &fyy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostGetLocalForm(fzz, &fzz_l); CHKERRXX(ierr);
#endif

    // Get sizes
    ierr = VecGetSize(f_l,   &f_size);   CHKERRXX(ierr);
    ierr = VecGetSize(fxx_l, &fxx_size); CHKERRXX(ierr);
    ierr = VecGetSize(fyy_l, &fyy_size); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetSize(fzz_l, &fzz_size); CHKERRXX(ierr);
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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
          << " f_size = " << f_size << " fzz_size = " << fzz_size << std::endl;

      throw std::invalid_argument(oss.str());
    }
#endif

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f,   &f_l  ); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fxx, &fxx_l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(fyy, &fyy_l); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostRestoreLocalForm(fzz, &fzz_l); CHKERRXX(ierr);
#endif
  }
#endif

  // get access to the iternal data
  double *f_p, *fxx_p, *fyy_p;
  ierr = VecGetArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(fxx, &fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(fyy, &fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *fzz_p;
  ierr = VecGetArray(fzz, &fzz_p); CHKERRXX(ierr);
#endif

  if (is_initialized){
    // compute the derivatives on the boundary nodes -- fxx
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors.at(layer_nodes[i]);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
   #ifdef P4_TO_P8
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
   #endif
         )
        fxx_p[layer_nodes[i]] = qnnn.dxx_central(f_p);
    }
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fyy
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors.at(layer_nodes[i]);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
   #ifdef P4_TO_P8
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
   #endif
         )
        fyy_p[layer_nodes[i]] = qnnn.dyy_central(f_p);
    }
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  #ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fzz
    for (size_t i=0; i<layer_nodes.size(); i++)
    {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors.at(layer_nodes[i]);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
         )
        fzz_p[layer_nodes[i]] = qnnn.dzz_central(f_p);
    }
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  #endif

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors.at(local_nodes[i]);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
   #ifdef P4_TO_P8
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
   #endif
         )
      {
        fxx_p[local_nodes[i]] = qnnn.dxx_central(f_p);
        fyy_p[local_nodes[i]] = qnnn.dyy_central(f_p);
#ifdef P4_TO_P8
        fzz_p[local_nodes[i]] = qnnn.dzz_central(f_p);
#endif
      }
    }

  } else {

    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes -- fxx
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
   #ifdef P4_TO_P8
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
   #endif
         )
      {
        fxx_p[layer_nodes[i]] = qnnn.dxx_central(f_p);
        fyy_p[layer_nodes[i]] = qnnn.dyy_central(f_p);
#ifdef P4_TO_P8
        fzz_p[layer_nodes[i]] = qnnn.dzz_central(f_p);
#endif
      }
    }
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateBegin(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);
      if(f_p[qnnn.node_000] > thr-EPS
         || f_p[qnnn.node_m00_mm] > thr-EPS || f_p[qnnn.node_m00_pm] > thr-EPS
         || f_p[qnnn.node_p00_mm] > thr-EPS || f_p[qnnn.node_p00_pm] > thr-EPS
         || f_p[qnnn.node_0m0_mm] > thr-EPS || f_p[qnnn.node_0m0_pm] > thr-EPS
         || f_p[qnnn.node_0p0_mm] > thr-EPS || f_p[qnnn.node_0p0_pm] > thr-EPS
   #ifdef P4_TO_P8
         || f_p[qnnn.node_m00_mp] > thr-EPS || f_p[qnnn.node_m00_pp] > thr-EPS
         || f_p[qnnn.node_p00_mp] > thr-EPS || f_p[qnnn.node_p00_pp] > thr-EPS
         || f_p[qnnn.node_0m0_mp] > thr-EPS || f_p[qnnn.node_0m0_pp] > thr-EPS
         || f_p[qnnn.node_0p0_mp] > thr-EPS || f_p[qnnn.node_0p0_pp] > thr-EPS
         || f_p[qnnn.node_00m_mm] > thr-EPS || f_p[qnnn.node_00m_pm] > thr-EPS
         || f_p[qnnn.node_00m_mp] > thr-EPS || f_p[qnnn.node_00m_pp] > thr-EPS
         || f_p[qnnn.node_00p_mm] > thr-EPS || f_p[qnnn.node_00p_pm] > thr-EPS
         || f_p[qnnn.node_00p_mp] > thr-EPS || f_p[qnnn.node_00p_pp] > thr-EPS
   #endif
         )
      {
        fxx_p[local_nodes[i]] = qnnn.dxx_central(f_p);
        fyy_p[local_nodes[i]] = qnnn.dyy_central(f_p);
#ifdef P4_TO_P8
        fzz_p[local_nodes[i]] = qnnn.dzz_central(f_p);
#endif
      }
    }
  }

  // restore internal data
  ierr = VecRestoreArray(f,   &f_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fxx, &fxx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fyy, &fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(fzz, &fzz_p); CHKERRXX(ierr);
#endif

  // finish the ghost update process to ensure all values are updated
  ierr = VecGhostUpdateEnd(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateEnd(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

  IPMLogRegionEnd("2nd_derivatives");
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central_above_threshold, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::first_derivatives_central(const Vec f, Vec fx[P4EST_DIM]) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_1st_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("1st_derivatives");

#ifdef CASL_THROWS
  {
    Vec f_l, fx_l[P4EST_DIM];
    PetscInt f_size, fx_size[P4EST_DIM];

    // Get local form
    ierr = VecGhostGetLocalForm(f, &f_l); CHKERRXX(ierr);
    ierr = VecGetSize(f_l, &f_size);   CHKERRXX(ierr);
    for (short i=0; i<P4EST_DIM; i++){
      ierr = VecGhostGetLocalForm(fx[i], &fx_l[i]); CHKERRXX(ierr);
      ierr = VecGetSize(fx_l[i], &fx_size[i]); CHKERRXX(ierr);

      if (f_size != fx_size[i]){
        std::ostringstream oss;
        oss << "[ERROR]: Vectors must be of same size when computing derivatives"
            << " f_size = " << f_size << " fx_size[" << i << "] = " << fx_size[i] << std::endl;

        throw std::invalid_argument(oss.str());
      }
    }

    // Restore local form
    ierr = VecGhostRestoreLocalForm(f, &f_l); CHKERRXX(ierr);
    for (short i=0; i<P4EST_DIM; i++)
      ierr = VecGhostRestoreLocalForm(fx[i], &fx_l[i]); CHKERRXX(ierr);
  }
#endif

  // get access to the iternal data
  double *f_p, *fx_p[P4EST_DIM];
  ierr = VecGetArray(f,&f_p); CHKERRXX(ierr);
  foreach_dimension(dim) {
    ierr = VecGetArray(fx[dim], &fx_p[dim]); CHKERRXX(ierr);
  }

  if (is_initialized){
    // compute the derivatives on the boundary nodes -- fx
    for (size_t i=0; i<layer_nodes.size(); i++)
      fx_p[0][layer_nodes[i]] = neighbors[layer_nodes[i]].dx_central(f_p);

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fx[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fy
    for (size_t i=0; i<layer_nodes.size(); i++)
      fx_p[1][layer_nodes[i]] = neighbors[layer_nodes[i]].dy_central(f_p);

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fx[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  #ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fz
    for (size_t i=0; i<layer_nodes.size(); i++)
      fx_p[2][layer_nodes[i]] = neighbors[layer_nodes[i]].dz_central(f_p);

    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fx[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  #endif

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      fx_p[0][local_nodes[i]] = neighbors[local_nodes[i]].dx_central(f_p);
      fx_p[1][local_nodes[i]] = neighbors[local_nodes[i]].dy_central(f_p);
  #ifdef P4_TO_P8
      fx_p[2][local_nodes[i]] = neighbors[local_nodes[i]].dz_central(f_p);
  #endif
    }

    foreach_dimension(dim) {
      ierr = VecGhostUpdateEnd(fx[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

  } else {

    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      fx_p[0][layer_nodes[i]] = qnnn.dx_central(f_p);
      fx_p[1][layer_nodes[i]] = qnnn.dy_central(f_p);
#ifdef P4_TO_P8
      fx_p[2][layer_nodes[i]] = qnnn.dz_central(f_p);
#endif
    }
    // start updating the ghost values
    foreach_dimension(dim)
      ierr = VecGhostUpdateBegin(fx[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      get_neighbors(local_nodes[i], qnnn);

      fx_p[0][local_nodes[i]] = qnnn.dx_central(f_p);
      fx_p[1][local_nodes[i]] = qnnn.dy_central(f_p);
  #ifdef P4_TO_P8
      fx_p[2][local_nodes[i]] = qnnn.dz_central(f_p);
  #endif
    }

    // finish updating the ghost values
    foreach_dimension(dim) {
      ierr = VecGhostUpdateEnd(fx[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  // restore internal data
  ierr = VecRestoreArray(f,  &f_p  ); CHKERRXX(ierr);
  foreach_dimension(dim) {
    ierr = VecRestoreArray(fx[dim], &fx_p[dim]); CHKERRXX(ierr);
  }

  IPMLogRegionEnd("1st_derivatives");
  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_1st_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy, Vec fzz) const
#else
void my_p4est_node_neighbors_t::second_derivatives_central_using_block(const Vec f, Vec fxx, Vec fyy) const
#endif
{
  // create temporary block vector
  PetscErrorCode ierr;
  Vec fdd;
  ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &fdd); CHKERRXX(ierr);

  // compute derivatives using block vector
  second_derivatives_central(f, fdd);

  // copy data back into original vectors
  double *fdd_p, *fxx_p, *fyy_p;
  ierr = VecGetArray(fdd, &fdd_p); CHKERRXX(ierr);
  ierr = VecGetArray(fxx, &fxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(fyy, &fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *fzz_p;
  ierr = VecGetArray(fzz, &fzz_p); CHKERRXX(ierr);
#endif

  // compute the derivatives on the boundary nodes
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
    fxx_p[i] = fdd_p[P4EST_DIM*i + 0];
    fyy_p[i] = fdd_p[P4EST_DIM*i + 1];
#ifdef P4_TO_P8
    fzz_p[i] = fdd_p[P4EST_DIM*i + 2];
#endif
  }

  // restore internal data
  ierr = VecRestoreArray(fdd, &fdd_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fxx, &fxx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fyy, &fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(fzz, &fzz_p); CHKERRXX(ierr);
#endif

  // destroy temporary variable
  ierr = VecDestroy(fdd); CHKERRXX(ierr);
}


void my_p4est_node_neighbors_t::get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists) const
{
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est, ni);
  bool xp_wall = is_node_xpWall(p4est, ni);

  bool ym_wall = is_node_ymWall(p4est, ni);
  bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est, ni);
  bool zp_wall = is_node_zpWall(p4est, ni);
#endif

  // count neighbors
  for (int i = 0; i < num_neighbors_cube; i++) neighbor_exists[i] = true;

  if (xm_wall)
  {
    int i = 0;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
#endif
        neighbor_exists[i + j*3 CODE3D( + k*3*3 )] = false;
  }

  if (xp_wall)
  {
    int i = 2;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
#endif
        neighbor_exists[i + j*3 CODE3D( + k*3*3 )] = false;
  }

  if (ym_wall)
  {
    int j = 0;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
#endif
        neighbor_exists[i + j*3 CODE3D( + k*3*3 )] = false;
  }

  if (yp_wall)
  {
    int j = 2;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
#endif
        neighbor_exists[i + j*3 CODE3D( + k*3*3 )] = false;
  }

#ifdef P4_TO_P8
  if (zm_wall)
  {
    int k = 0;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }

  if (zp_wall)
  {
    int k = 2;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }
#endif

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
