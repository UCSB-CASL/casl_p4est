#ifdef P4_TO_P8
#include "my_p8est_node_neighbors.h"
#else
#include "my_p4est_node_neighbors.h"
#endif

#include <src/petsc_compatibility.h>

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
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central;
extern PetscLogEvent log_my_p4est_node_neighbors_t_2nd_derivatives_central_block;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

#include <assert.h>

#define sc_array_index(a,n) my_sc_array_index(a,n)

void* my_sc_array_index(sc_array_t* a, size_t n){
  assert(n<a->elem_count);
  return ((void *) (a->array + (a->elem_size * n)));
}

void my_p4est_node_neighbors_t::init_neighbors()
{
  if (is_initialized) return;

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t, 0, 0, 0, 0); CHKERRXX(ierr);
  neighbors.resize(nodes->num_owned_indeps);

  for( p4est_locidx_t n=0; n < nodes->num_owned_indeps; ++n)
    get_neighbors(n, neighbors[n]);

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_node_neighbors_t::get_neighbors(p4est_locidx_t n, quad_neighbor_nodes_of_node_t &qnnn) const
{
  p4est_connectivity_t *connectivity = p4est->connectivity;
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n);

  // need to unclamp the node to make sure we get the correct coordinate
  p4est_indep_t node_unclamped = *node;
  p4est_node_unclamp((p4est_quadrant_t*)&node_unclamped);

  p4est_topidx_t tree_id = node->p.piggy3.which_tree;
  p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

  double tree_xmin = connectivity->vertices[3*v_mmm + 0];
  double tree_ymin = connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tree_zmin = connectivity->vertices[3*v_mmm + 2];
#endif


  double x = node_unclamped.x / (double) P4EST_ROOT_LEN + tree_xmin;
  double y = node_unclamped.y / (double) P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
  double z = node_unclamped.z / (double) P4EST_ROOT_LEN + tree_zmin;
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

#ifdef P4_TO_P8
  find_neighbor_cell_of_node(node,-1,-1, -1, quad_mmm_idx, tree_mmm_idx);
  find_neighbor_cell_of_node(node,-1, 1, -1, quad_mpm_idx, tree_mpm_idx);
  find_neighbor_cell_of_node(node, 1,-1, -1, quad_pmm_idx, tree_pmm_idx);
  find_neighbor_cell_of_node(node, 1, 1, -1, quad_ppm_idx, tree_ppm_idx);
  find_neighbor_cell_of_node(node,-1,-1,  1, quad_mmp_idx, tree_mmp_idx);
  find_neighbor_cell_of_node(node,-1, 1,  1, quad_mpp_idx, tree_mpp_idx);
  find_neighbor_cell_of_node(node, 1,-1,  1, quad_pmp_idx, tree_pmp_idx);
  find_neighbor_cell_of_node(node, 1, 1,  1, quad_ppp_idx, tree_ppp_idx);
#else
  find_neighbor_cell_of_node(node,-1,-1, quad_mmm_idx, tree_mmm_idx);
  find_neighbor_cell_of_node(node,-1, 1, quad_mpm_idx, tree_mpm_idx);
  find_neighbor_cell_of_node(node, 1,-1, quad_pmm_idx, tree_pmm_idx);
  find_neighbor_cell_of_node(node, 1, 1, quad_ppm_idx, tree_ppm_idx);
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

  if(quad_mmm_idx == -1)
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

  if(quad_mpm_idx == -1)
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

  if(quad_pmm_idx == -1)
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

  if(quad_ppm_idx == -1)
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
  if(quad_mmp_idx == -1)
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

  if(quad_mpp_idx == -1)
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

  if(quad_pmp_idx == -1)
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

  if(quad_ppp_idx == -1)
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

    qnnn.d_m00 = P4EST_QUADRANT_LEN(quad_m00->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_m00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mmm];
    qnnn.node_m00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mpm];
#ifdef P4_TO_P8
    qnnn.node_m00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mmp];
    qnnn.node_m00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_m00_idx + dir::v_mpp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_m00_idx + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
#endif

    double qy = quad_m00->y / (double) P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double qz = quad_m00->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_m00->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_m00_m0 = y - qy;
    qnnn.d_m00_p0 = qh - qnnn.d_m00_m0;
#ifdef P4_TO_P8
    qnnn.d_m00_0m = z - qz;
    qnnn.d_m00_0p = qh - qnnn.d_m00_0m;
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

    qnnn.d_p00 = P4EST_QUADRANT_LEN(quad_p00->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_p00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_pmm];
    qnnn.node_p00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_p00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_pmp];
    qnnn.node_p00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_p00_idx + dir::v_ppp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_p00_idx + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
#endif

    double qy = quad_p00->y / (double) P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double qz = quad_p00->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_p00->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_p00_m0 = y - qy;
    qnnn.d_p00_p0 = qh - qnnn.d_p00_m0;
#ifdef P4_TO_P8
    qnnn.d_p00_0m = z - qz;
    qnnn.d_p00_0p = qh - qnnn.d_p00_0m;
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

    qnnn.d_0m0 = P4EST_QUADRANT_LEN(quad_0m0->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_0m0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_mmm];
    qnnn.node_0m0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_pmm];
#ifdef P4_TO_P8
    qnnn.node_0m0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_mmp];
    qnnn.node_0m0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_0m0_idx + dir::v_pmp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0m0_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
#endif

    double qx = quad_0m0->x / (double) P4EST_ROOT_LEN + tree_xmin;
#ifdef P4_TO_P8
    double qz = quad_0m0->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_0m0->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_0m0_m0 = x - qx;
    qnnn.d_0m0_p0 = qh - qnnn.d_0m0_m0;
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = z - qz;
    qnnn.d_0m0_0p = qh - qnnn.d_0m0_0m;
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

    qnnn.d_0p0 = P4EST_QUADRANT_LEN(quad_0p0->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_0p0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_mpm];
    qnnn.node_0p0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_0p0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_mpp];
    qnnn.node_0p0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_0p0_idx + dir::v_ppp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_0p0_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
#endif

    double qx = quad_0p0->x / (double) P4EST_ROOT_LEN + tree_xmin;
#ifdef P4_TO_P8
    double qz = quad_0p0->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_0p0->level) / (double) P4EST_ROOT_LEN;
    qnnn.d_0p0_m0 = x - qx;
    qnnn.d_0p0_p0 = qh - qnnn.d_0p0_m0;
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = z - qz;
    qnnn.d_0p0_0p = qh - qnnn.d_0p0_0m;
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

    qnnn.d_00m = P4EST_QUADRANT_LEN(quad_00m->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_00m_mm = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_mmm];
    qnnn.node_00m_pm = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_pmm];
    qnnn.node_00m_mp = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_mpm];
    qnnn.node_00m_pp = nodes->local_nodes[P4EST_CHILDREN*quad_00m_idx + dir::v_ppm];

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00m_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];

    double qx = quad_00m->x / (double) P4EST_ROOT_LEN + tree_xmin;
    double qy = quad_00m->y / (double) P4EST_ROOT_LEN + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_00m->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00m_m0 = x - qx;
    qnnn.d_00m_p0 = qh - qnnn.d_00m_m0;
    qnnn.d_00m_0m = y - qy;
    qnnn.d_00m_0p = qh - qnnn.d_00m_0m;
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

    qnnn.d_00p = P4EST_QUADRANT_LEN(quad_00p->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_00p_mm = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_mmp];
    qnnn.node_00p_pm = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_pmp];
    qnnn.node_00p_mp = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_mpp];
    qnnn.node_00p_pp = nodes->local_nodes[P4EST_CHILDREN*quad_00p_idx + dir::v_ppp];

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_00p_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];

    double qx = quad_00p->x / (double) P4EST_ROOT_LEN + tree_xmin;
    double qy = quad_00p->y / (double) P4EST_ROOT_LEN + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_00p->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00p_m0 = x - qx;
    qnnn.d_00p_p0 = qh - qnnn.d_00p_m0;
    qnnn.d_00p_0m = y - qy;
    qnnn.d_00p_0p = qh - qnnn.d_00p_0m;
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
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = 1, cj = -1, ck = -1;

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
    p4est_locidx_t    quad_min_idx = -1;
    if (quad_min->level < quad_pmm->level && quad_pmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; cj = -1; ck = -1; }
    if (quad_min->level < quad_ppm->level && quad_ppm_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; cj =  1; ck = -1; }
#ifdef P4_TO_P8
    if (quad_min->level < quad_pmp->level && quad_pmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; cj = -1; ck =  1; }
    if (quad_min->level < quad_ppp->level && quad_ppp_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; cj =  1; ck =  1; }
#endif
#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in m00."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif
    const bool di = 1;
    const bool dj = cj != 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );
#else
    find_neighbor_cell_of_node( node_tmp, ci, cj, quad_tmp_idx, tree_tmp_idx );
#endif

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in m00."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_m00 = - qnnn.d_p00 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_m00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
    qnnn.node_m00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_m00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
    qnnn.node_m00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double qy = quad_tmp->y / (double) P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double qz = quad_tmp->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_m00_m0 = y - qy;
    qnnn.d_m00_p0 = qh - qnnn.d_m00_m0;
#ifdef P4_TO_P8
    qnnn.d_m00_0m = z - qz;
    qnnn.d_m00_0p = qh - qnnn.d_m00_0m;
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
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = -1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = -1;

    if (quad_min->level < quad_mmm->level && quad_mmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; cj = -1; ck = -1; }
    if (quad_min->level < quad_mpm->level && quad_mpm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; cj =  1; ck = -1; }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mmp->level && quad_mmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; cj = -1; ck =  1; }
    if (quad_min->level < quad_mpp->level && quad_mpp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; cj =  1; ck =  1; }
#endif
#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in p00."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif
    const bool di = 0;
    const bool dj = cj != 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );
#else
    find_neighbor_cell_of_node( node_tmp, ci, cj, quad_tmp_idx, tree_tmp_idx );
#endif

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in p00."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_p00 = - qnnn.d_m00 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_p00_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_p00_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
#ifdef P4_TO_P8
    qnnn.node_p00_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_p00_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double qy = quad_tmp->y / (double) P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double qz = quad_tmp->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_p00_m0 = y - qy;
    qnnn.d_p00_p0 = qh - qnnn.d_p00_m0;
#ifdef P4_TO_P8
    qnnn.d_p00_0m = z - qz;
    qnnn.d_p00_0p = qh - qnnn.d_p00_0m;
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
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = 1, ck = -1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = -1;

    if (quad_min->level < quad_mpm->level && quad_mpm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; ci = -1; ck = -1; }
    if (quad_min->level < quad_ppm->level && quad_ppm_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; ci =  1; ck = -1; }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mpp->level && quad_mpp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; ci = -1; ck =  1; }
    if (quad_min->level < quad_ppp->level && quad_ppp_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; ci =  1; ck =  1; }
#endif

#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in 0m0."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif

    const bool di = ci != 1;
    const bool dj = 1;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );
#else
    find_neighbor_cell_of_node( node_tmp, ci, cj, quad_tmp_idx, tree_tmp_idx );
#endif

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in 0m0."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_0m0 = - qnnn.d_0p0 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_0m0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
    qnnn.node_0m0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];
#ifdef P4_TO_P8
    qnnn.node_0m0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
    qnnn.node_0m0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double qx = quad_tmp->x / (double) P4EST_ROOT_LEN + tree_xmin;
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double qz = quad_tmp->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_0m0_m0 = x - qx;
    qnnn.d_0m0_p0 = qh - qnnn.d_0m0_m0;
#ifdef P4_TO_P8
    qnnn.d_0m0_0m = z - qz;
    qnnn.d_0m0_0p = qh - qnnn.d_0m0_0m;
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
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = -1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = -1;

    if (quad_min->level < quad_mmm->level && quad_mmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; ci = -1; ck = -1; }
    if (quad_min->level < quad_pmm->level && quad_pmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; ci =  1; ck = -1; }
#ifdef P4_TO_P8
    if (quad_min->level < quad_mmp->level && quad_mmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; ci = -1; ck =  1; }
    if (quad_min->level < quad_pmp->level && quad_pmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; ci =  1; ck =  1; }
#endif

#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in 0p0."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif

    const bool di = ci != 1;
    const bool dj = 0;
#ifdef P4_TO_P8
    const bool dk = ck != 1;
#else
    const bool dk = 0;
#endif

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
#ifdef P4_TO_P8
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );
#else
    find_neighbor_cell_of_node( node_tmp, ci, cj, quad_tmp_idx, tree_tmp_idx );
#endif

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in 0p0."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_0p0 = - qnnn.d_0m0 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_0p0_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_0p0_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
#ifdef P4_TO_P8
    qnnn.node_0p0_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_0p0_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
#endif

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double qx = quad_tmp->x / (double) P4EST_ROOT_LEN + tree_xmin;
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mmm + 2];
    double qz = quad_tmp->z / (double) P4EST_ROOT_LEN + tree_zmin;
#endif
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_0p0_m0 = x - qx;
    qnnn.d_0p0_p0 = qh - qnnn.d_0p0_m0;
#ifdef P4_TO_P8
    qnnn.d_0p0_0m = z - qz;
    qnnn.d_0p0_0p = qh - qnnn.d_0p0_0m;
#endif
  }

#ifdef P4_TO_P8
  /* correcting for wall in the 00m direction  (Only in 3D) */
  if(quad_mmm==&root && quad_pmm==&root && quad_mpm==&root && quad_ppm==&root)
  {
    /* fetch the second order neighbor to the right */
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = 1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = -1;

    if (quad_min->level < quad_mmp->level && quad_mmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmp; quad_min_idx = quad_mmp_idx; ci = -1; cj = -1; }
    if (quad_min->level < quad_pmp->level && quad_pmp_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmp; quad_min_idx = quad_pmp_idx; ci =  1; cj = -1; }
    if (quad_min->level < quad_mpp->level && quad_mpp_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpp; quad_min_idx = quad_mpp_idx; ci = -1; cj =  1; }
    if (quad_min->level < quad_ppp->level && quad_ppp_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppp; quad_min_idx = quad_ppp_idx; ci =  1; cj =  1; }

#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in 00m."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif

    const bool di = ci != 1;
    const bool dj = cj != 1;
    const bool dk = 1;

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in 00m."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_00m = - qnnn.d_00p - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_00m_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmp];
    qnnn.node_00m_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmp];
    qnnn.node_00m_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpp];
    qnnn.node_00m_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppp];

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double qx = quad_tmp->x / (double) P4EST_ROOT_LEN + tree_xmin;
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double qy = quad_tmp->y / (double) P4EST_ROOT_LEN + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00m_m0 = x - qx;
    qnnn.d_00m_p0 = qh - qnnn.d_00m_m0;
    qnnn.d_00m_0m = y - qy;
    qnnn.d_00m_0p = qh - qnnn.d_00m_0m;
  }

  /* correcting for wall in the 00p direction  (Only in 3D) */
  if(quad_mmp==&root && quad_pmp==&root && quad_mpp==&root && quad_ppp==&root)
  {
    /* fetch the second order neighbor to the right */
    p4est_indep_t *node_tmp;
    p4est_locidx_t quad_tmp_idx;
    p4est_topidx_t tree_tmp_idx;
    short ci = -1, cj = -1, ck = -1;

    p4est_quadrant_t *quad_min     = &root;
    p4est_locidx_t    quad_min_idx = -1;

    if (quad_min->level < quad_mmm->level && quad_mmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mmm; quad_min_idx = quad_mmm_idx; ci = -1; cj = -1; }
    if (quad_min->level < quad_pmm->level && quad_pmm_idx < p4est->local_num_quadrants)
    { quad_min = quad_pmm; quad_min_idx = quad_pmm_idx; ci =  1; cj = -1; }
    if (quad_min->level < quad_mpm->level && quad_mpm_idx < p4est->local_num_quadrants)
    { quad_min = quad_mpm; quad_min_idx = quad_mpm_idx; ci = -1; cj =  1; }
    if (quad_min->level < quad_ppm->level && quad_ppm_idx < p4est->local_num_quadrants)
    { quad_min = quad_ppm; quad_min_idx = quad_ppm_idx; ci =  1; cj =  1; }

#ifdef CASL_THROWS
    if (quad_min_idx == -1)
        throw std::runtime_error("[ERROR]: could not find a neighboring cell when correcting for wall in 00p."
                                 "This is most probably a bug in my_p4est_hierarchy_t construction");
#endif

    const bool di = ci != 1;
    const bool dj = cj != 1;
    const bool dk = 0;

    node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_min_idx + 4*dk+2*dj+di] );
    find_neighbor_cell_of_node( node_tmp, ci, cj, ck, quad_tmp_idx, tree_tmp_idx );

#ifdef CASL_THROWS
    if (quad_tmp_idx == -1)
    throw std::runtime_error("[ERROR]: could not find a suitable second order neighbor when correcting for wall in 00p."
                             "This is most probably a bug in 'find_neighbor_cell_of_node' function.");
#endif

    p4est_quadrant_t *quad_tmp;
    if(quad_tmp_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
    }
    else
      quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

    qnnn.d_00p = - qnnn.d_00m - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
    qnnn.node_00p_mm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mmm];
    qnnn.node_00p_pm = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_pmm];
    qnnn.node_00p_mp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_mpm];
    qnnn.node_00p_pp = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + dir::v_ppm];

    p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0];
    double tree_xmin = connectivity->vertices[3*v_mmm + 0];
    double qx = quad_tmp->x / (double) P4EST_ROOT_LEN + tree_xmin;
    double tree_ymin = connectivity->vertices[3*v_mmm + 1];
    double qy = quad_tmp->y / (double) P4EST_ROOT_LEN + tree_ymin;
    double qh = P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;

    qnnn.d_00p_m0 = x - qx;
    qnnn.d_00p_p0 = qh - qnnn.d_00p_m0;
    qnnn.d_00p_0m = y - qy;
    qnnn.d_00p_0p = qh - qnnn.d_00p_0m;
  }
#endif
}

#ifdef P4_TO_P8
void my_p4est_node_neighbors_t::find_neighbor_cell_of_node( p4est_indep_t *node_, char i, char j, char k, p4est_locidx_t& quad, p4est_topidx_t& nb_tree_idx ) const
{
  p4est_indep_t node_struct = *node_;
  p4est_indep_t *node = &node_struct;
  p4est_node_unclamp((p4est_quadrant_t*)node);

  p4est_connectivity_t *connectivity = p4est->connectivity;

  p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
  nb_tree_idx = tree_idx;

  p4est_qcoord_t x_perturb = node->x+i;
  p4est_qcoord_t y_perturb = node->y+j;
  p4est_qcoord_t z_perturb = node->z+k;

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[2] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 1 - pm0 */
  else if(node->x == P4EST_ROOT_LEN && node->y == 0 &&
          i ==  1                   && j == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 2 - mp0 */
  else if(node->x == 0 && node->y == P4EST_ROOT_LEN &&
          i == -1      && j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
  }
  /* 3 - pp0 */
  else if(node->x == P4EST_ROOT_LEN && node->y == P4EST_ROOT_LEN &&
          i ==  1                   && j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = 1;
    y_perturb = 1;
  }
  /* 4 - m0m */
  else if(node->x == 0 && node->z == 0 &&
          i == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 5 - p0m */
  else if(node->x == P4EST_ROOT_LEN && node->z == 0 &&
          i ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 6 - m0p */
  else if(node->x == 0 && node->z == P4EST_ROOT_LEN &&
          i == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 7 - p0p */
  else if(node->x == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          i ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    x_perturb = 1;
    z_perturb = 1;
  }
  /* 8 - 0mm */
  else if(node->y == 0 && node->z == 0 &&
          j == -1      && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 9 - 0pm */
  else if(node->y == P4EST_ROOT_LEN && node->z == 0 &&
          j ==  1                   && k == -1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    y_perturb = 1;
    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 10 - 0mp */
  else if(node->y == 0 && node->z == P4EST_ROOT_LEN &&
          j == -1      && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
    z_perturb = 1;
  }
  /* 11 - 0pp */
  else if(node->y == P4EST_ROOT_LEN && node->z == P4EST_ROOT_LEN &&
          j ==  1                   && k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[2];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }
    nb_tree_idx = tmp_tree_idx[1] = connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) { quad = -1; return; }

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
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

    x_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 1 - p00 */
  else if(node->x == P4EST_ROOT_LEN &&
          i ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

    x_perturb = 1;
  }
  /* 2 - 0m0 */
  else if(node->y == 0 &&
          j == -1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

    y_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 3 - 0p0 */
  else if(node->y == P4EST_ROOT_LEN &&
          j ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

    y_perturb = 1;
  }
  /* 4 - 00m */
  else if(node->z == 0 &&
          k == -1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_00m]; // 00m
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

    z_perturb = P4EST_ROOT_LEN - 1;
  }
  /* 5 - 0p0 */
  else if(node->z == P4EST_ROOT_LEN &&
          k ==  1)
  {
    p4est_topidx_t tmp_tree_idx[1];
    nb_tree_idx = tmp_tree_idx[0] = connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_00p]; // 00p
    if(tmp_tree_idx[0] == tree_idx)        { quad = -1; return; }

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
void my_p4est_node_neighbors_t::find_neighbor_cell_of_node( p4est_indep_t *node_, char i, char j, p4est_locidx_t& quad, p4est_topidx_t& nb_tree_idx ) const
{
  p4est_indep_t node_struct = *node_;
  p4est_indep_t *node = &node_struct;
  p4est_node_unclamp((p4est_quadrant_t*)node);

  p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
  nb_tree_idx = tree_idx;
  p4est_connectivity_t *connectivity = p4est->connectivity;

  p4est_qcoord_t x_perturb = node->x+i;
  p4est_qcoord_t y_perturb = node->y+j;

  /* first check the corners of the tree */
  if(node->x==0 && node->y==0 && i==-1 && j==-1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(tmp_tree_idx == tree_idx) { quad = -1; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
    if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==P4EST_ROOT_LEN && node->y==0 && i== 1 && j==-1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(tmp_tree_idx == tree_idx) { quad = -1; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
    if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
    x_perturb = 1;
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==0 && node->y==P4EST_ROOT_LEN && i==-1 && j== 1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(tmp_tree_idx == tree_idx) { quad = -1; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
    if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
    y_perturb = 1;
  }
  else if(node->x==P4EST_ROOT_LEN && node->y==P4EST_ROOT_LEN && i== 1 && j== 1)
  {
    p4est_topidx_t tmp_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(tmp_tree_idx == tree_idx) { quad = -1; return; }
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
    if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
    x_perturb = 1;
    y_perturb = 1;
  }

  /* now check the edges of the tree */
  else if(node->x==0 && i==-1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
    if(nb_tree_idx == tree_idx) { quad = -1; return; }
    x_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->x==P4EST_ROOT_LEN && i==1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
    if(nb_tree_idx == tree_idx) { quad = -1; return; }
    x_perturb = 1;
  }
  else if(node->y==0 && j==-1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
    if(nb_tree_idx == tree_idx) { quad = -1; return; }
    y_perturb = P4EST_ROOT_LEN - 1;
  }
  else if(node->y==P4EST_ROOT_LEN && j==1)
  {
    nb_tree_idx = connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
    if(nb_tree_idx == tree_idx) { quad = -1; return; }
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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
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
void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f, Vec fxx, Vec fyy, Vec fzz) const
#else
void my_p4est_node_neighbors_t::second_derivatives_central(const Vec f, Vec fxx, Vec fyy) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_node_neighbors_t_2nd_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);

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
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
          << " f_size = " << f_size << " fxx_size = " << fxx_size << std::endl;

      throw std::invalid_argument(oss.str());
    }

    if (f_size != fyy_size){
      std::ostringstream oss;
      oss << "[ERROR]: Vectors must be of same size whe computing derivatives"
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

#ifdef DXX_USE_BLOCKS
#ifdef P4_TO_P8
  second_derivatives_central_using_block(f, fxx, fyy, fzz);
#else
  second_derivatives_central_using_block(f, fxx, fyy);
#endif
#else // !DXX_USE_BLOCKS
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
      fxx_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dxx_central(f_p);
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute the derivatives on the boundary nodes -- fyy
    for (size_t i=0; i<layer_nodes.size(); i++)
      fyy_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dyy_central(f_p);
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  #ifdef P4_TO_P8
    // compute the derivatives on the boundary nodes -- fzz
    for (size_t i=0; i<layer_nodes.size(); i++)
      fzz_p[layer_nodes[i]] = neighbors[layer_nodes[i]].dzz_central(f_p);
    // start updating the ghost values
    ierr = VecGhostUpdateBegin(fzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  #endif

    // compute the derivaties for all internal nodes
    for (size_t i=0; i<local_nodes.size(); i++){
      fxx_p[local_nodes[i]] = neighbors[local_nodes[i]].dxx_central(f_p);
      fyy_p[local_nodes[i]] = neighbors[local_nodes[i]].dyy_central(f_p);
  #ifdef P4_TO_P8
      fzz_p[local_nodes[i]] = neighbors[local_nodes[i]].dzz_central(f_p);
  #endif
    }

  } else {

    quad_neighbor_nodes_of_node_t qnnn;

    // compute the derivatives on the boundary nodes -- fxx
    for (size_t i=0; i<layer_nodes.size(); i++){
      get_neighbors(layer_nodes[i], qnnn);
      fxx_p[layer_nodes[i]] = qnnn.dxx_central(f_p);
      fyy_p[layer_nodes[i]] = qnnn.dyy_central(f_p);
#ifdef P4_TO_P8
      fzz_p[layer_nodes[i]] = qnnn.dzz_central(f_p);
#endif
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

      fxx_p[local_nodes[i]] = qnnn.dxx_central(f_p);
      fyy_p[local_nodes[i]] = qnnn.dyy_central(f_p);
  #ifdef P4_TO_P8
      fzz_p[local_nodes[i]] = qnnn.dzz_central(f_p);
  #endif
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
#endif // !DXX_USE_BLOCKS

  ierr = PetscLogEventEnd(log_my_p4est_node_neighbors_t_2nd_derivatives_central, 0, 0, 0, 0); CHKERRXX(ierr);
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
