#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#include "my_p8est_tools.h"
#include <p8est_connectivity.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include "cube3.h"
#else
#include "my_p4est_utils.h"
#include "my_p4est_tools.h"
#include <p4est_connectivity.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include "cube2.h"
#endif

#include "mpi.h"
#include <vector>
#include <set>
#include <sstream>
#include <petsclog.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <stack>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_TINY_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

std::vector<InterpolatingFunctionLogEntry> InterpolatingFunctionLogger::entries;

WallBC2D::~WallBC2D() {}
WallBC3D::~WallBC3D() {}

bool quadrant_value_is_well_defined(double &phi_q, const BoundaryConditionsDIM &bc_cell_field, const p4est_t* p4est, const p4est_ghost_t* ghost, const p4est_nodes_t* nodes,
                                    const p4est_locidx_t &quad_idx, const p4est_topidx_t &tree_idx, const double *node_sampled_phi_p)
{
  bool value_is_well_defined = bc_cell_field.interfaceType() == NOINTERFACE; // always well-defined if no interface (or, equivalently, if no node-sampled levelset is given)
  if(!value_is_well_defined)
  {
    /* check if quadrant is well defined */
    phi_q = 0.0;
    bool one_corner_in_neg_domain = false;
    for(u_char i = 0; i < P4EST_CHILDREN; ++i)
    {
      const double &tmp = node_sampled_phi_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + i]];
      one_corner_in_neg_domain = one_corner_in_neg_domain || tmp < 0.0;
      phi_q += tmp;
    }
    phi_q /= (double) P4EST_CHILDREN;
    // well defined if phi_q < 0.0 no matter which boundary condition is used
    // or if a corner value is in negative domain and the interface is (constant Neumann)
    value_is_well_defined = phi_q < 0.0 || (one_corner_in_neg_domain && bc_cell_field.interfaceType() == NEUMANN);
    if(!value_is_well_defined && one_corner_in_neg_domain && bc_cell_field.interfaceType() == MIXED)
    {
      // if mixed interface, phi_q non-negative, but one corner is in negative domain,
      // the value is well-defined if the local cell is marked "Neumann"
      double qxyz[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, qxyz);
      value_is_well_defined = bc_cell_field.interfaceType(qxyz) == NEUMANN;
    }
  }
  return value_is_well_defined;
}

bool index_of_node(const p4est_quadrant_t *n, const p4est_nodes_t* nodes, p4est_locidx_t& idx)
{
#ifdef P4EST_DEBUG
  int clamped = 1;
#endif
  P4EST_ASSERT(p4est_quadrant_is_node(n, clamped));
  size_t idx_l, idx_u, idx_m;
  const p4est_indep_t *node_l, *node_u, *node_m;
  // check if the candidate can be in the locally owned nodes first, or in the ghost ones
  idx_l   = 0;                              SC_ASSERT(idx_l < nodes->indep_nodes.elem_count);
  idx_u   = nodes->num_owned_indeps - 1;    SC_ASSERT(idx_u < nodes->indep_nodes.elem_count);

  node_l  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_l*nodes->indep_nodes.elem_size);
  node_u  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_u*nodes->indep_nodes.elem_size);
  if((p4est_quadrant_compare_piggy(node_l, n) > 0) || (p4est_quadrant_compare_piggy(node_u, n) < 0))
    goto lookup_in_ghost_nodes;
  while((p4est_quadrant_compare_piggy(node_l, n) <= 0) && (p4est_quadrant_compare_piggy(node_u, n) >= 0))
  {
    if(!p4est_quadrant_compare_piggy(node_l, n))
    {
      idx = idx_l;
      return true;
    }
    if(!p4est_quadrant_compare_piggy(node_u, n))
    {
      idx = idx_u;
      return true;
    }
    if(idx_u - idx_l == 1)
      break;
    idx_m   = (idx_l + idx_u)/2;  SC_ASSERT(idx_m < nodes->indep_nodes.elem_count);
    node_m  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_m*nodes->indep_nodes.elem_size);
    P4EST_ASSERT(p4est_quadrant_compare_piggy(node_l, node_m) <= 0 && p4est_quadrant_compare_piggy(node_u, node_m) >= 0);
    if(p4est_quadrant_compare_piggy(node_m, n) < 0)
    {
      idx_l   = idx_m;
      node_l  = node_m;
    }
    else if (p4est_quadrant_compare_piggy(node_m, n) > 0)
    {
      idx_u   = idx_m;
      node_u  = node_m;
    }
    else
    {
      P4EST_ASSERT(!p4est_quadrant_compare_piggy(node_m, n));
      idx = idx_m;
      return true;
    }
  }
  return false;
lookup_in_ghost_nodes:
  P4EST_ASSERT((p4est_quadrant_compare_piggy(node_l, n) > 0) || (p4est_quadrant_compare_piggy(node_u, n) < 0));

  idx_l   = nodes->num_owned_indeps;            SC_ASSERT(idx_l <= nodes->indep_nodes.elem_count); // '<=' because it could be '==' if there is no ghost node (or running on a single core for instance)
  idx_u   = nodes->indep_nodes.elem_count - 1;  SC_ASSERT(idx_u < nodes->indep_nodes.elem_count);
  if(idx_l <= idx_u) // do this only if there are ghost nodes!
  {
    node_l  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_l*nodes->indep_nodes.elem_size);
    node_u  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_u*nodes->indep_nodes.elem_size);
    while((p4est_quadrant_compare_piggy(node_l, n) <= 0) && (p4est_quadrant_compare_piggy(node_u, n) >= 0))
    {
      if(!p4est_quadrant_compare_piggy(node_l, n))
      {
        idx = idx_l;
        return true;
      }
      if(!p4est_quadrant_compare_piggy(node_u, n))
      {
        idx = idx_u;
        return true;
      }
      if(idx_u - idx_l == 1)
        break;
      idx_m   = (idx_l + idx_u)/2; SC_ASSERT(idx_m < nodes->indep_nodes.elem_count);
      node_m  = (const p4est_indep_t*) (nodes->indep_nodes.array + idx_m*nodes->indep_nodes.elem_size);
      P4EST_ASSERT((p4est_quadrant_compare_piggy(node_l, node_m) <= 0) && (p4est_quadrant_compare_piggy(node_u, node_m) >= 0));
      if(p4est_quadrant_compare_piggy(node_m, n) <0)
      {
        idx_l   = idx_m;
        node_l  = node_m;
      }
      else if (p4est_quadrant_compare_piggy(node_m, n) >0)
      {
        idx_u   = idx_m;
        node_u  = node_m;
      }
      else
      {
        P4EST_ASSERT(!p4est_quadrant_compare_piggy(node_m, n));
        idx = idx_m;
        return true;
      }
    }
  }
  return false;
}

p4est_gloidx_t compute_global_index_of_quad(const p4est_locidx_t& quad_local_idx, const p4est_t* p4est, const p4est_ghost_t* ghost)
{
  if(quad_local_idx < p4est->local_num_quadrants)
    return p4est->global_first_quadrant[p4est->mpirank] + quad_local_idx;

  const p4est_quadrant_t *quad = p4est_const_quadrant_array_index(&ghost->ghosts, quad_local_idx - p4est->local_num_quadrants);
  return p4est->global_first_quadrant[quad_find_ghost_owner(ghost, quad_local_idx - p4est->local_num_quadrants)] + quad->p.piggy3.local_num;
}

p4est_locidx_t find_local_index_of_quad(const p4est_gloidx_t& quad_global_idx, const p4est_t* p4est, const p4est_ghost_t* ghost)
{
  P4EST_ASSERT(0 <= quad_global_idx && quad_global_idx < p4est->global_num_quadrants);
  if(p4est->global_first_quadrant[p4est->mpirank] <= quad_global_idx && quad_global_idx < p4est->global_first_quadrant[p4est->mpirank + 1])
    return quad_global_idx - p4est->global_first_quadrant[p4est->mpirank];

  // search in ghost
  int owner_rank = 0;
  int rank_up   = p4est->mpisize;
  while(rank_up - owner_rank > 1)
  {
    const int r = (owner_rank + rank_up)/2;
    if(p4est->global_first_quadrant[r] <= quad_global_idx)
      owner_rank = r;
    else
      rank_up = r;
  }

  p4est_locidx_t ghost_idx    = ghost->proc_offsets[owner_rank];
  p4est_locidx_t ghost_idx_up = ghost->proc_offsets[owner_rank + 1];
  const p4est_quadrant_t* ghost_quad = p4est_const_quadrant_array_index(&ghost->ghosts, ghost_idx);
  while (p4est->global_first_quadrant[owner_rank] + ghost_quad->p.piggy3.local_num != quad_global_idx) {
    P4EST_ASSERT(ghost_idx_up - ghost_idx > 1);
    const p4est_locidx_t mid_ghost_idx = (ghost_idx + ghost_idx_up)/2;
    ghost_quad = p4est_const_quadrant_array_index(&ghost->ghosts, mid_ghost_idx);
    if(p4est->global_first_quadrant[owner_rank] + ghost_quad->p.piggy3.local_num <= quad_global_idx)
      ghost_idx = mid_ghost_idx;
    else
      ghost_idx_up = mid_ghost_idx;
  }

  return p4est->local_num_quadrants + ghost_idx;
}

p4est_topidx_t tree_index_of_quad(const p4est_locidx_t& quad_idx, const p4est_t* p4est, const p4est_ghost_t* ghost)
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants + (ghost != NULL ? (p4est_locidx_t) ghost->ghosts.elem_count : 0));
  if(quad_idx > p4est->local_num_quadrants)
  {
    if(ghost == NULL)
      throw std::runtime_error("my_p4est_utils::tree_index_of_quad called for a ghost quadrant but ghosts are not provided...");
    const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    return quad->p.piggy3.which_tree;
  }
  p4est_topidx_t tree_l = p4est->first_local_tree;
  p4est_topidx_t tree_u = p4est->last_local_tree + 1;
  while (tree_u - tree_l > 1) {
    p4est_topidx_t tree_m = (tree_l + tree_u)/2;
    p4est_locidx_t quad_offset_tree_m = p4est_tree_array_index(p4est->trees, tree_m)->quadrants_offset;
    if(quad_idx >= quad_offset_tree_m)
      tree_l = tree_m;
    else
      tree_u = tree_m;
  }
  return tree_l;
}

bool is_node_in_domain(p4est_indep_t& node, const my_p4est_brick_t* brick, const p4est_connectivity_t* connectivity)
{
  P4EST_ASSERT(p4est_quadrant_is_inside_3x3((const p4est_quadrant_t*) &node));
  /* list the coordinates that are past the tree borders
   * --> need to search across a corner if all true;
   * --> across an edge if 2 are true (in 3D only);
   * --> across a face if 1 only is true
   * --> the perturbed node is all good as it is if none is true
   */
  const int past_tree_border[P4EST_DIM] = {DIM(node.x < 0 || node.x > P4EST_ROOT_LEN, node.y < 0 || node.y > P4EST_ROOT_LEN, node.z < 0 || node.z > P4EST_ROOT_LEN)};

  if(ANDD(!past_tree_border[0], !past_tree_border[1], !past_tree_border[2]))
    return true; // we are still within the same tree, nothing else to do...

  // If some "past tree border" was found, we need to find the correct tree that owns that
  // perturbed point, if it exists (otherwise, we are searching past the domain's boundaries
  // and we gotta return 'false'). If a correct owning tree is found, we also correct the logical
  // coordinates of that perturbed node to match the appropriate description in that tree...

  // we copy the perturbed logical coordinates into an array first for ease of implementation, and we'll copy them back into the node structure thereafter
  p4est_topidx_t owning_tree_idx = node.p.which_tree;
  p4est_qcoord_t perturbed_qxyz[P4EST_DIM] = {DIM(node.x, node.y, node.z)};

  if(SUMD(past_tree_border[0], past_tree_border[1], past_tree_border[2]) == P4EST_DIM) // we have to look across a tree corner
  {
    if(ORD(!is_periodic(connectivity, dir::x) && brick->nxyztrees[dir::x] == 1,
           !is_periodic(connectivity, dir::y) && brick->nxyztrees[dir::y] == 1,
           !is_periodic(connectivity, dir::z) && brick->nxyztrees[dir::z] == 1))
      return false; // the perturbed node is past the domain's boundary

    const u_char local_corner_idx = SUMD((perturbed_qxyz[0] > P4EST_ROOT_LEN ? 1 : 0), (perturbed_qxyz[1] > P4EST_ROOT_LEN ? 2 : 0), (perturbed_qxyz[2] > P4EST_ROOT_LEN ? 4 : 0));
    P4EST_ASSERT(local_corner_idx < P4EST_CHILDREN);
    const p4est_topidx_t corner = connectivity->tree_to_corner[P4EST_CHILDREN*node.p.piggy3.which_tree + local_corner_idx];
    if(corner == -1)
      return false; // the corner exists indeed, but it does not connect with any other tree -> the perturbed node is past the domain's boundary

    const p4est_topidx_t offset = connectivity->ctt_offset[corner];
    owning_tree_idx = connectivity->corner_to_tree[offset + local_corner_idx];
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      perturbed_qxyz[dim] += (perturbed_qxyz[dim] > P4EST_ROOT_LEN ? -1 : +1)*P4EST_ROOT_LEN;
  }
#ifdef P4_TO_P8
  else if(SUMD(past_tree_border[0], past_tree_border[1], past_tree_border[2]) == 2) // across tree edge
  {
    // we go through two different trees in this case
    const u_char first_dim = (past_tree_border[dir::x] ? dir::x : dir::y); P4EST_ASSERT(past_tree_border[first_dim]);
    const u_char first_face_dir = 2*first_dim + (perturbed_qxyz[first_dim] > P4EST_ROOT_LEN ? 1 : 0);
    const p4est_topidx_t first_tree_idx = connectivity->tree_to_tree[P4EST_FACES*node.p.which_tree + first_face_dir];
    if(!is_periodic(connectivity, first_dim) && first_tree_idx == node.p.which_tree)
      return false; // no other tree there, so nothing to find
    perturbed_qxyz[first_dim] += (perturbed_qxyz[first_dim] > P4EST_ROOT_LEN ? -1 : +1)*P4EST_ROOT_LEN;

    // find the second direction
    const u_char second_dim = (past_tree_border[dir::z] ? dir::z : dir::y); P4EST_ASSERT(past_tree_border[second_dim]);
    const u_char second_face_dir = 2*second_dim + (perturbed_qxyz[second_dim] > P4EST_ROOT_LEN ? 1 : 0);
    owning_tree_idx = connectivity->tree_to_tree[P4EST_FACES*first_tree_idx + second_face_dir];
    if(!is_periodic(connectivity, second_dim) && owning_tree_idx == first_tree_idx)
      return false; // no other tree there, so nothing to find

    perturbed_qxyz[second_dim] += (perturbed_qxyz[second_dim] > P4EST_ROOT_LEN ? -1 : +1)*P4EST_ROOT_LEN;
  }
#endif
  else
  {
    P4EST_ASSERT(SUMD(past_tree_border[0], past_tree_border[1], past_tree_border[2]) == 1);
    const u_char dim = (past_tree_border[0] ? dir::x : ONLY3D( OPEN_PARENTHESIS past_tree_border[1] ?) dir::y ONLY3D( : dir::z CLOSE_PARENTHESIS)); P4EST_ASSERT(past_tree_border[dim]);
    const u_char face_dir = 2*dim + (perturbed_qxyz[dim] > P4EST_ROOT_LEN ? 1 : 0);
    owning_tree_idx = connectivity->tree_to_tree[P4EST_FACES*node.p.which_tree + face_dir];
    if(!is_periodic(connectivity, dim) && owning_tree_idx == node.p.which_tree)
      return false; // no other tree there, so nothing to find
    perturbed_qxyz[dim] += (perturbed_qxyz[dim] > P4EST_ROOT_LEN ? -1 : +1)*P4EST_ROOT_LEN;
  }

  node.x = perturbed_qxyz[0];
  node.y = perturbed_qxyz[1];
#ifdef P4_TO_P8
  node.z = perturbed_qxyz[2];
#endif
  node.p.which_tree = owning_tree_idx;
  return true;
}

bool logical_vertex_in_quad_is_fine_node(const p4est_t* fine_p4est, const p4est_nodes_t* fine_nodes,
                                         const p4est_quadrant_t &quad, const p4est_topidx_t& tree_idx, DIM(const char& vx, const char& vy, const char& vz),
                                         p4est_locidx_t& fine_vertex_idx)
{
  p4est_quadrant_t fine_node_to_fetch;
  fine_node_to_fetch.level = P4EST_MAXLEVEL; fine_node_to_fetch.p.which_tree = tree_idx;
  XCODE(fine_node_to_fetch.x = quad.x + (vx + 1)*P4EST_QUADRANT_LEN(quad.level + 1));
  YCODE(fine_node_to_fetch.y = quad.y + (vy + 1)*P4EST_QUADRANT_LEN(quad.level + 1));
  ZCODE(fine_node_to_fetch.z = quad.z + (vz + 1)*P4EST_QUADRANT_LEN(quad.level + 1));
  P4EST_ASSERT (p4est_quadrant_is_node (&fine_node_to_fetch, 0));
  const p4est_quadrant_t *tmp_ptr = &fine_node_to_fetch;
  // but if it lies on the edge of the tree, we need to canonicalize it first, or it won't be fetched correctly
  if(ORD(fine_node_to_fetch.x == 0 || fine_node_to_fetch.x == P4EST_ROOT_LEN, fine_node_to_fetch.y == 0 || fine_node_to_fetch.y == P4EST_ROOT_LEN, fine_node_to_fetch.z == 0 || fine_node_to_fetch.z == P4EST_ROOT_LEN))
  {
    p4est_quadrant_t n;
    p4est_node_canonicalize(fine_p4est, tree_idx, &fine_node_to_fetch, &n);
    tmp_ptr = &n;
  }

  return index_of_node(tmp_ptr, fine_nodes, fine_vertex_idx);
}

void rel_xyz_quad_fr_point(const p4est_t* p4est, const p4est_quadrant_t& quad, const double* xyz, const my_p4est_brick_t* brick, double *xyz_rel, int64_t* qcoord_quad)
{
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[quad.p.piggy3.which_tree*P4EST_CHILDREN + 0];
  const double* tree_xyz_min = p4est->connectivity->vertices + 3*v_m;
  const double tree_dimension[P4EST_DIM] = {DIM((brick->xyz_max[0] - brick->xyz_min[0])/brick->nxyztrees[0], (brick->xyz_max[1] - brick->xyz_min[1])/brick->nxyztrees[1], (brick->xyz_max[2] - brick->xyz_min[2])/brick->nxyztrees[2])};

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    p4est_qcoord_t quad_qxyz = P4EST_QUADRANT_LEN(quad.level + 1) + (dim == dir::x ? quad.x  : ONLY3D(OPEN_PARENTHESIS dim == dir::y ?) quad.y ONLY3D(: quad.z CLOSE_PARENTHESIS));
    qcoord_quad[dim] = ((p4est_topidx_t) round((tree_xyz_min[dim] - brick->xyz_min[dim])/tree_dimension[dim]))*P4EST_ROOT_LEN + quad_qxyz;
    xyz_rel[dim] = tree_dimension[dim]*(double)quad_qxyz/(double)P4EST_ROOT_LEN + tree_xyz_min[dim] - xyz[dim];
    if(is_periodic(p4est, dim))
    {
      const double pp = xyz_rel[dim]/(brick->xyz_max[dim] - brick->xyz_min[dim]);
      xyz_rel[dim] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(brick->xyz_max[dim] - brick->xyz_min[dim]);
    }
  }
}

void get_local_interpolation_weights(const p4est_t* p4est, const p4est_topidx_t& tree_id, const p4est_quadrant_t& quad, const double *xyz_global,
                                     double* linear_weight, double* second_derivative_weight)
{
  p4est_topidx_t v_m  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN - 1];

  const double* tree_xyz_min    = (p4est->connectivity->vertices + 3*v_m);
  const double* tree_xyz_max    = (p4est->connectivity->vertices + 3*v_p);

  const double qh = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  const double qxyz_min[P4EST_DIM] = {DIM((double) quad.x/(double) P4EST_ROOT_LEN, (double) quad.y/(double) P4EST_ROOT_LEN, (double) quad.z/(double) P4EST_ROOT_LEN)};

  double xyz[P4EST_DIM] = {DIM(xyz_global[0], xyz_global[1], xyz_global[2])};
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    xyz[dir] = (xyz[dir] - tree_xyz_min[dir])/(tree_xyz_max[dir] - tree_xyz_min[dir]);

  P4EST_ASSERT(ANDD(!is_periodic(p4est, dir::x) || (xyz[0] >= qxyz_min[0] - qh/10 && xyz[0] <= qxyz_min[0] + qh + qh/10),
      !is_periodic(p4est, dir::y) || (xyz[1] >= qxyz_min[1] - qh/10 && xyz[1] <= qxyz_min[1] + qh + qh/10),
      !is_periodic(p4est, dir::z) || (xyz[2] >= qxyz_min[2] - qh/10 && xyz[2] <= qxyz_min[2] + qh + qh/10)));

  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    xyz[dir] = (xyz[dir] - qxyz_min[dir])/qh;

  double d_[P4EST_DIM][2];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    d_[dim][0] = 1.0 - xyz[dim];
    d_[dim][1] = xyz[dim];
  }

#ifdef P4_TO_P8
  for (u_char inc_z = 0; inc_z < 2; ++inc_z)
#endif
    for (u_char inc_y = 0; inc_y < 2; ++inc_y)
      for (u_char inc_x = 0; inc_x < 2; ++inc_x)
        linear_weight[SUMD(inc_x, 2*inc_y, 4*inc_z)] = MULTD(d_[0][inc_x], d_[1][inc_y], d_[2][inc_z]);
  if(second_derivative_weight != NULL)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      second_derivative_weight[dim] = SQR((tree_xyz_max[dim] - tree_xyz_min[dim])*qh)*d_[dim][0]*d_[dim][1];

  return;
}

void linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global, double* results, const size_t &n_results )
{
  P4EST_ASSERT(n_results > 0);
  double linear_weight[P4EST_CHILDREN];
  get_local_interpolation_weights(p4est, tree_id, quad, xyz_global, linear_weight);

  for (unsigned int k = 0; k < n_results; ++k)
  {
    results[k] = 0.0;
    for (short j = 0; j<P4EST_CHILDREN; j++)
      results[k] += + F[P4EST_CHILDREN*k+j]*linear_weight[j];
  }

  PetscErrorCode ierr = PetscLogFlops(39); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, const size_t &n_results )
{
  P4EST_ASSERT(n_results > 0);
  double linear_weight[P4EST_CHILDREN], second_derivative_weight[P4EST_DIM];
  get_local_interpolation_weights(p4est, tree_id, quad, xyz_global, linear_weight, second_derivative_weight);

  double fdd[P4EST_DIM];
  for (unsigned int k = 0; k < n_results; ++k) {
    results[k] = 0.0;
    for (short j = 0; j < P4EST_CHILDREN; ++j) {
      for (short i = 0; i<P4EST_DIM; i++)
        fdd[i] = (j == 0? Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] : MINMOD(fdd[i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + i]));
      results[k] += F[k*P4EST_CHILDREN+j]*linear_weight[j];
    }
    results[k] -= 0.5*SUMD(second_derivative_weight[0]*fdd[0], second_derivative_weight[1]*fdd[1], second_derivative_weight[2]*fdd[2]);
  }

  PetscErrorCode ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_continuous_v1_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, const size_t &n_results )
{
  P4EST_ASSERT(n_results > 0);
  double linear_weight[P4EST_CHILDREN], second_derivative_weight[P4EST_DIM];
  get_local_interpolation_weights(p4est, tree_id, quad, xyz_global, linear_weight, second_derivative_weight);

  // First alternative scheme: first, minmod on every edge, then weight-average
  double fdd[P4EST_DIM];
  unsigned int i, jm, jp;
  for (unsigned int k = 0; k < n_results; ++k) {
    // set your fdd
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      fdd[dir] = 0.0;

    i = 0;
    jm = 0; jp = 1; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 2; jp = 3; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 5; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 6; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
#endif

    i = 1;
    jm = 0; jp = 2; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 1; jp = 3; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 6; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 5; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
#endif

#ifdef P4_TO_P8
    i = 2;
    jm = 0; jp = 4; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 1; jp = 5; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 2; jp = 6; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
    jm = 3; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(linear_weight[jm] + linear_weight[jp]);
#endif

    results[k] = 0.0;
    for (u_char j = 0; j < P4EST_CHILDREN; ++j)
      results[k] += F[k*P4EST_CHILDREN+j]*linear_weight[j];

    results[k] -= 0.5*SUMD(second_derivative_weight[0]*fdd[0], second_derivative_weight[1]*fdd[1], second_derivative_weight[2]*fdd[2]);
  }


  PetscErrorCode ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_continuous_v2_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, const size_t &n_results )
{
  P4EST_ASSERT(n_results > 0);
  double linear_weight[P4EST_CHILDREN], second_derivative_weight[P4EST_DIM];
  get_local_interpolation_weights(p4est, tree_id, quad, xyz_global, linear_weight, second_derivative_weight);

  // Second alternative scheme: first, weight-average in perpendicular plane, then minmod
  double fdd[P4EST_DIM];
  unsigned int i, jm, jp;
  double fdd_m, fdd_p;
  for (unsigned int k = 0; k < n_results; ++k) {
    // set your fdd
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      fdd[dir] = 0.0;

    i = 0;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 1; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 2; jp = 3; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 5; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 6; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
#endif
    fdd[i] = MINMOD(fdd_m, fdd_p);

    i = 1;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 2; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 1; jp = 3; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 6; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 5; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
#endif
    fdd[i] = MINMOD(fdd_m, fdd_p);

#ifdef P4_TO_P8
    i = 2;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 4; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 1; jp = 5; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 2; jp = 6; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    jm = 3; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jm*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM + jp*P4EST_DIM + i]*(linear_weight[jm] + linear_weight[jp]);
    fdd[i] = MINMOD(fdd_m, fdd_p);
#endif

    results[k] = 0.0;
    for (u_char j = 0; j < P4EST_CHILDREN; ++j)
      results[k] += F[k*P4EST_CHILDREN+j]*linear_weight[j];

    results[k] -= 0.5*SUMD(second_derivative_weight[0]*fdd[0], second_derivative_weight[1]*fdd[1], second_derivative_weight[2]*fdd[2]);
  }
  PetscErrorCode ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, const size_t &n_results )
{
  P4EST_ASSERT(n_results > 0);
  double linear_weight[P4EST_CHILDREN], second_derivative_weight[P4EST_DIM];
  get_local_interpolation_weights(p4est, tree_id, quad, xyz_global, linear_weight, second_derivative_weight);

  double fdd[P4EST_DIM];
  for (unsigned int k = 0; k < n_results; ++k)
  {
    results[k] = 0.0;
    for (short j=0; j<P4EST_CHILDREN; j++)
    {
      for (short i = 0; i<P4EST_DIM; i++)
        fdd[i] = ((j == 0)? 0.0 : fdd[i]) +Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + i] * linear_weight[j];
      results[k] += F[k*P4EST_CHILDREN+j]*linear_weight[j];
    }
    results[k] -= 0.5*SUMD(second_derivative_weight[0]*fdd[0], second_derivative_weight[1]*fdd[1], second_derivative_weight[2]*fdd[2]);
  }

  PetscErrorCode ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
}

void write_comm_stats(const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes, const char *partition_name, const char *topology_name, const char *neighbors_name)
{
  FILE *file;
  PetscErrorCode ierr;

  /* save partition information */
  if (partition_name) {
    ierr = PetscFOpen(p4est->mpicomm, partition_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  p4est_gloidx_t num_nodes = 0;
  for (int r =0; r<p4est->mpisize; r++)
    num_nodes += nodes->global_owned_indeps[r];

  PetscFPrintf(p4est->mpicomm, file, "%% global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
  PetscFPrintf(p4est->mpicomm, file, "%% mpi_rank | local_node_size | local_quad_size | ghost_node_size | ghost_quad_size\n");
  PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d, %7d, %7d, %5d, %5d\n",
                           p4est->mpirank, nodes->num_owned_indeps, p4est->local_num_quadrants, nodes->indep_nodes.elem_count-nodes->num_owned_indeps, ghost->ghosts.elem_count);
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (partition_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }

  /* save recv info based on the ghost nodes */
  if (topology_name){
    ierr = PetscFOpen(p4est->mpicomm, topology_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  PetscFPrintf(p4est->mpicomm, file, "%% Topology of ghost nodes based on how many ghost nodes belongs to a certain processor \n");
  PetscFPrintf(p4est->mpicomm, file, "%% this_rank | ghost_rank | ghost_node_size \n");
  std::vector<p4est_locidx_t> ghost_nodes(p4est->mpisize, 0);
  std::set<int> proc_neighbors;
  for (size_t i=0; i<nodes->indep_nodes.elem_count - nodes->num_owned_indeps; i++){
    int r = nodes->nonlocal_ranks[i];
    proc_neighbors.insert(r);
    ghost_nodes[r]++;
  }
  for (std::set<int>::const_iterator it = proc_neighbors.begin(); it != proc_neighbors.end(); ++it){
    int r = *it;
    PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d %4d %6d\n", p4est->mpirank, r, ghost_nodes[r]);
  }
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (topology_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }

  /* save recv info based on the ghost nodes */
  if (neighbors_name){
    ierr = PetscFOpen(p4est->mpicomm, neighbors_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  PetscFPrintf(p4est->mpicomm, file, "%% number of neighboring processors \n");
  PetscFPrintf(p4est->mpicomm, file, "%% this_rank | number_ghost_rank \n");
  PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d %4d\n", p4est->mpirank, proc_neighbors.size());
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (neighbors_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }
}


int8_t find_max_level(const p4est_t* p4est)
{
  int8_t max_lvl = 0;
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    max_lvl = MAX(max_lvl, tree->maxlevel);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_lvl, 1, MPI_INT8_T, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  return max_lvl;
}

p4est_bool_t nodes_are_equal(int mpi_size, p4est_nodes_t* nodes_1, p4est_nodes_t* nodes_2)
{
  if(nodes_1 == nodes_2)
    return P4EST_TRUE;
  const p4est_indep_t *node_1, *node_2;
  p4est_bool_t result = (nodes_1->indep_nodes.elem_count == nodes_2->indep_nodes.elem_count);
  result = result && (nodes_1->num_local_quadrants  == nodes_2->num_local_quadrants);
  result = result && (nodes_1->num_owned_indeps     == nodes_2->num_owned_indeps);
  result = result && (nodes_1->num_owned_shared     == nodes_2->num_owned_shared);
  result = result && (nodes_1->offset_owned_indeps == 0) && (nodes_2->offset_owned_indeps == 0);
  if(!result)
    goto return_time;
  for (int r = 0; r < mpi_size; ++r) {
    result = result && (nodes_1->global_owned_indeps[r] == nodes_2->global_owned_indeps[r]);
    if(!result)
      goto return_time;
  }
  // compare the raw nodes, one by one, first
  for (size_t k = 0; k < nodes_1->indep_nodes.elem_count; ++k) {
    node_1 = (const p4est_indep_t*) sc_array_index(&nodes_1->indep_nodes, k);
    node_2 = (const p4est_indep_t*) sc_array_index(&nodes_2->indep_nodes, k);
    result = result && (node_1->level == node_2->level);
    result = result && (node_1->x == node_2->x);
    result = result && (node_1->y == node_2->y);
#ifdef P4_TO_P8
    result = result && (node_1->z == node_2->z);
#endif
    result = result && (node_1->pad8 == node_2->pad8);
    // all nodes must have their p.piggy3 used, local or ghost...
    result = result && (node_1->p.piggy3.local_num  == node_2->p.piggy3.local_num);
    result = result && (node_1->p.piggy3.which_tree == node_2->p.piggy3.which_tree);
    if(k > ((size_t) nodes_1->num_owned_indeps))
      result = result && (nodes_1->nonlocal_ranks[k-nodes_1->num_owned_indeps] == nodes_2->nonlocal_ranks[k-nodes_2->num_owned_indeps]);
    if(!result)
      goto return_time;
  }
  // check that the local indices of points associated with local quadrants are equal
  for (p4est_locidx_t k = 0; k < nodes_1->num_local_quadrants; ++k) {
    for (short j = 0; j < P4EST_CHILDREN; ++j) {
      result = result && (nodes_1->local_nodes[P4EST_CHILDREN*k + j] == nodes_2->local_nodes[P4EST_CHILDREN*k + j]);
      if(!result)
        goto return_time;
    }
  }
return_time:
  return result;
}

p4est_bool_t ghosts_are_equal(const p4est_ghost_t* ghost_1, const p4est_ghost_t* ghost_2)
{
  if(ghost_1 == ghost_2)
    return P4EST_TRUE;

  const p4est_quadrant_t *quad_1, *quad_2;
  int mpisize = ghost_1->mpisize;
  p4est_bool_t result = ghost_2->mpisize == mpisize;
  result = result && ghost_1->ghosts.elem_count == ghost_2->ghosts.elem_count;
  result = result && ghost_1->num_trees == ghost_2->num_trees;
  result = result && ghost_1->btype == ghost_2->btype;
  if(!result)
    return result;
  for (size_t k = 0; k < ghost_1->ghosts.elem_count; ++k) {
    quad_1 = p4est_const_quadrant_array_index(&ghost_1->ghosts, k);
    quad_2 = p4est_const_quadrant_array_index(&ghost_2->ghosts, k);
    result = result && p4est_quadrant_is_equal(quad_1, quad_2);
    result = result && quad_1->p.piggy3.local_num == quad_2->p.piggy3.local_num;
    result = result && quad_1->p.piggy3.which_tree == quad_2->p.piggy3.which_tree;
    if(!result)
      return result;
  }
  for (int r = 0; r < mpisize + 1; ++r) {
    result = result && ghost_1->proc_offsets[r] == ghost_2->proc_offsets[r];
    if(!result)
      return result;
  }
  for (p4est_topidx_t tree_idx = 0; tree_idx < ghost_1->num_trees+1; ++tree_idx) {
    result = result && ghost_1->tree_offsets[tree_idx] == ghost_2->tree_offsets[tree_idx];
    if(!result)
      return result;
  }
  return result;
}

PetscErrorCode VecGetLocalAndGhostSizes(const Vec& v, PetscInt& local_size, PetscInt& ghosted_size, const bool &ghosted)
{
  PetscErrorCode ierr = 0;
  ierr = VecGetLocalSize(v, &local_size);       CHKERRQ(ierr);
  if(ghosted)
  {
    Vec v_loc;
    ierr = VecGhostGetLocalForm(v, &v_loc);     CHKERRQ(ierr);
    ierr = VecGetSize(v_loc, &ghosted_size);    CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(v, &v_loc); CHKERRQ(ierr);
  }
  return ierr;
}

bool VecIsSetForNodes(const Vec& v, const p4est_nodes_t* nodes, const MPI_Comm& mpicomm, const unsigned int& blocksize, const bool& ghosted)
{
  P4EST_ASSERT(v != NULL);
  P4EST_ASSERT(blocksize > 0);
  PetscInt local_size, ghosted_size;
  VecGetLocalAndGhostSizes(v, local_size, ghosted_size, ghosted);
  int my_test = (local_size == (PetscInt) (blocksize*nodes->num_owned_indeps) && (!ghosted || ghosted_size == (PetscInt) (blocksize*nodes->indep_nodes.elem_count))) ? 1 : 0;
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_test, 1, MPI_INT, MPI_LAND, mpicomm); SC_CHECK_MPI(mpiret);
  return my_test;
}

bool VecIsSetForCells(const Vec& v, const p4est_t* p4est, const p4est_ghost_t* ghost, const unsigned int &blocksize, const bool &ghosted)
{
  P4EST_ASSERT(v != NULL);
  P4EST_ASSERT(blocksize > 0);
  PetscInt local_size, ghosted_size;
  VecGetLocalAndGhostSizes(v, local_size, ghosted_size, ghosted);
  int my_test = (local_size == (PetscInt) (blocksize*p4est->local_num_quadrants) && (!ghosted || ghosted_size == (PetscInt) (blocksize*(p4est->local_num_quadrants + ghost->ghosts.elem_count)))) ? 1 : 0;
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_test, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  return my_test;
}

PetscErrorCode delete_and_nullify_vector(Vec& vv)
{
  PetscErrorCode ierr = 0;
  if(vv != NULL){
    ierr = VecDestroy(vv); CHKERRQ(ierr);
    vv = NULL;
  }
  return ierr;
}

PetscErrorCode VecCreateGhostNodesBlock(const p4est_t *p4est, const p4est_nodes_t *nodes, const PetscInt & block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;
  P4EST_ASSERT(block_size > 0);
  P4EST_ASSERT(nodes->indep_nodes.elem_count >= (size_t) num_local);
  P4EST_ASSERT(p4est->mpisize >= 0);
  std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for (size_t i = 0; i<ghost_nodes.size(); ++i)
  {
    /*
    p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+num_local);
     * [RAPHAEL:] substituted this latter line of code by the following to enforce and ensure const attribute in 'nodes' argument...
     */
    SC_ASSERT(((size_t) (i+num_local))<nodes->indep_nodes.elem_count);
    const p4est_indep_t* ni = (const p4est_indep_t*) (nodes->indep_nodes.array + (((size_t) (i+num_local))*nodes->indep_nodes.elem_size));

    ghost_nodes[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
  }

  if(block_size > 1){
    ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, num_local*block_size, num_global*block_size,
                               ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  } else{
    ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                          ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateNoGhostNodesBlock(const p4est_t *p4est, const p4est_nodes_t *nodes, const PetscInt & block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;
  P4EST_ASSERT(block_size > 0);

  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r < p4est->mpisize; ++r)
    global_offset_sum[r + 1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  ierr = VecCreateMPI(p4est->mpicomm, num_local*block_size, num_global*block_size, v); CHKERRQ(ierr);
  if(block_size > 1){
    ierr = VecSetBlockSize(*v, block_size); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecGhostCopy(Vec src, Vec dst)
{
  PetscErrorCode ierr;

  Vec src_l, dst_l;
  ierr = VecGhostGetLocalForm(src, &src_l); CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(dst, &dst_l); CHKERRQ(ierr);
  ierr = VecCopy(src_l, dst_l); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(src, &src_l); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(dst, &dst_l); CHKERRQ(ierr);

  return 0;
}

PetscErrorCode VecGhostSet(Vec x, double v)
{
  PetscErrorCode ierr;
  Vec x_l;

  ierr = VecGhostGetLocalForm(x, &x_l); CHKERRQ(ierr);
  ierr = VecSet(x, v); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(x, &x_l); CHKERRQ(ierr);

  return 0;
}

PetscErrorCode VecCreateGhostCellsBlock(const p4est_t *p4est, const p4est_ghost_t *ghost, const PetscInt & block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;
  P4EST_ASSERT(block_size > 0);

  std::vector<PetscInt> ghost_cells(ghost->ghosts.elem_count, 0);
  PetscInt num_global = p4est->global_num_quadrants;

  for (int r = 0; r<p4est->mpisize; ++r)
    for (p4est_locidx_t q = ghost->proc_offsets[r]; q < ghost->proc_offsets[r+1]; ++q)
    {
      /*
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
       * [RAPHAEL:] substituted this latter line of code by the following to enforce and ensure const attribute in 'nodes' argument...
       */
      SC_ASSERT(((size_t) q)<ghost->ghosts.elem_count);
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*) (ghost->ghosts.array + ((size_t) q)*ghost->ghosts.elem_size);

      ghost_cells[q] = (PetscInt)quad->p.piggy3.local_num + (PetscInt)p4est->global_first_quadrant[r];
    }

  if(block_size > 1){
    ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, num_local*block_size, num_global*block_size,
                             ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  } else {
    ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                          ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateNoGhostCellsBlock(const p4est_t *p4est, const PetscInt &block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;
  P4EST_ASSERT(block_size > 0);

  PetscInt num_global = p4est->global_num_quadrants;

  ierr = VecCreateMPI(p4est->mpicomm, num_local*block_size, num_global*block_size, v); CHKERRQ(ierr);
  if(block_size > 1){
    ierr = VecSetBlockSize(*v, block_size); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecScatterAllToSomeCreate(MPI_Comm comm, Vec origin_loc, Vec destination, const PetscInt &ndest_glo_idx, const PetscInt *dest_glo_idx, VecScatter *ctx)
{
  PetscErrorCode ierr;
  IS is_from, is_to;
  ierr    = ISCreateGeneral(comm, ndest_glo_idx, dest_glo_idx, PETSC_USE_POINTER, &is_to);    CHKERRQ(ierr);
  ierr    = ISCreateStride(comm, ndest_glo_idx, 0, 1, &is_from);                              CHKERRQ(ierr);
  ierr    = VecScatterCreate(origin_loc, is_from, destination, is_to, ctx);                   CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode VecScatterCreateChangeLayout(MPI_Comm comm, Vec from, Vec to, VecScatter *ctx)
{
  PetscErrorCode ierr = 0;
#ifdef CASL_THROWS
  PetscInt size_from, size_to;
  ierr = VecGetSize(from, &size_from); CHKERRXX(ierr);
  ierr = VecGetSize(to, &size_to); CHKERRXX(ierr);
  if (size_from != size_to)
    throw std::invalid_argument("[ERROR]: Change layout is only supported for vectors with the same global size");
#endif

  IS is_from, is_to;

  ISLocalToGlobalMapping l2g;
  ierr = VecGetLocalToGlobalMapping(to, &l2g); CHKERRXX(ierr);

  const PetscInt *idx;
  PetscInt l2g_size;
  ierr = ISLocalToGlobalMappingGetIndices(l2g, &idx); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingGetSize(l2g, &l2g_size); CHKERRXX(ierr);

  ierr = ISCreateStride(comm, l2g_size, 0, 1, &is_to); CHKERRXX(ierr);
  ierr = ISCreateGeneral(comm, l2g_size, idx, PETSC_USE_POINTER, &is_from); CHKERRXX(ierr);

  Vec to_l;
  ierr = VecGhostGetLocalForm(to, &to_l); CHKERRXX(ierr);
  ierr = VecScatterCreate(from, is_from, to_l, is_to, ctx); CHKERRXX(ierr);

  ierr = ISDestroy(is_from); CHKERRXX(ierr);
  ierr = ISDestroy(is_to); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(l2g, &idx); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l); CHKERRXX(ierr);

  return ierr;
}

PetscErrorCode VecGhostChangeLayoutBegin(VecScatter ctx, Vec from, Vec to)
{
  PetscErrorCode ierr;
  Vec to_l;

  ierr = VecGhostGetLocalForm(to, &to_l);
  ierr = VecScatterBegin(ctx, from, to_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l);

  return ierr;
}

PetscErrorCode VecGhostChangeLayoutEnd(VecScatter ctx, Vec from, Vec to)
{
  PetscErrorCode ierr;
  Vec to_l;

  ierr = VecGhostGetLocalForm(to, &to_l);
  ierr = VecScatterEnd(ctx, from, to_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l);

  return ierr;
}

bool is_folder(const char* path)
{
  struct stat info;
  if(stat(path, &info)!= 0 )
  {
#ifdef CASL_THROWS
    char error_message[1024];
    sprintf(error_message, "is_folder: could not access %s", path);
    throw std::runtime_error(error_message);
#else
    return false;
#endif
  }
  return (info.st_mode & S_IFDIR);
}

bool file_exists(const char* path)
{
  struct stat info;
  return ((stat(path, &info)== 0) && (info.st_mode & S_IFREG));
}


int create_directory(const char* path, int mpi_rank, MPI_Comm comm)
{
  int return_ = 1;
  if(mpi_rank == 0)
  {
    struct stat info;
    if((stat(path, &info) == 0) &&  (info.st_mode & S_IFDIR)) // if it already exists, no need to create it...
      return_ = 0;
    else
    {
      char tmp[PATH_MAX];
      snprintf(tmp, sizeof(tmp), "%s", path);
      size_t len = strlen(tmp);
      if(tmp[len-1] == '/')
        tmp[len-1] = 0;
      for (char* p = tmp+1; *p; p++){
        if(*p == '/'){
          *p = 0;
          if((stat(tmp, &info) == 0) &&  (info.st_mode & S_IFDIR)) // if it already exists, no need to create it...
            return_ = 0;
          else
            return_ = mkdir(tmp, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH); // permission = 755 like a regular mkdir in terminal
          *p = '/';
          if(return_)
            break;
        }
      }
      if(return_ == 0) // successfull up to here
        return_ = mkdir(path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH); // permission = 755 like a regular mkdir in terminal
    }
  }
  int mpiret = MPI_Bcast(&return_, 1, MPI_INT, 0, comm); SC_CHECK_MPI(mpiret);
  return return_;
}

int  get_subdirectories_in(const char* root_path, std::vector<std::string>& subdirectories)
{
  if(!is_folder(root_path))
    return 1;

  subdirectories.resize(0);

  DIR *dir = opendir(root_path);
  struct dirent *entry = readdir(dir);
  while (entry != NULL)
  {
    if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
      subdirectories.push_back(entry->d_name);
    entry = readdir(dir);
  }

  closedir(dir);

  return 0;
}

int delete_directory(const char* root_path, int mpi_rank, MPI_Comm comm, bool non_collective)
{
  if(!is_folder(root_path))
  {
    char error_message[1024];
    sprintf(error_message, "delete_directory: path %s is NOT a directory...", root_path);
    throw std::invalid_argument(error_message);
  }

  int return_ = 1;
  if(mpi_rank == 0)
  {
    std::vector<std::string> subdirectories; subdirectories.resize(0);
    std::vector<std::string> reg_files; reg_files.resize(0);

    DIR *dir = opendir(root_path);
    struct dirent *entry = readdir(dir);
    while (entry != NULL)
    {
      if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
      {
        if(entry->d_type == DT_DIR)
          subdirectories.push_back(entry->d_name);
        else if (entry->d_type == DT_REG)
          reg_files.push_back(entry->d_name);
        else
        {
          char path_to_weird_thing[PATH_MAX], error_msg[1024];
          sprintf(path_to_weird_thing, "%s/%s", root_path, entry->d_name);
          sprintf(error_msg, "delete_directory: a weird object has been encountered in %s: it is neither a folder nor a file, this function is not designed for that, use maybe 'rm -rf'", path_to_weird_thing);
          throw std::runtime_error(error_msg);
          return 1;
        }
      }
      entry = readdir(dir);
    }
    for (unsigned int idx = 0; idx < reg_files.size(); ++idx) {
      char path_to_file[PATH_MAX];
      sprintf(path_to_file, "%s/%s", root_path, reg_files[idx].c_str());
      remove(path_to_file);
    }
    for (unsigned int idx = 0; idx < subdirectories.size(); ++idx) {
      char path_to_subfolder[PATH_MAX];
      sprintf(path_to_subfolder, "%s/%s", root_path, subdirectories[idx].c_str());
      delete_directory(path_to_subfolder, mpi_rank, comm, true);
    }
    remove(root_path);
    if(non_collective)
      return 0;
    else
      return_ = 0;
  }
  if(!non_collective)
  {
    int mpiret = MPI_Bcast(&return_, 1, MPI_INT, 0, comm); SC_CHECK_MPI(mpiret);
  }
  return return_;
}

void dxyz_min(const p4est_t *p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  for(u_char dir = 0; dir < P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p + dir] - v[3*v_m + dir]) / (1<<data->max_lvl);
}

void get_dxyz_min(const p4est_t *p4est, double dxyz[], double *dxyz_min, double *diag_min)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  double dxyz_own[P4EST_DIM];
  if (dxyz == NULL) {
    dxyz = dxyz_own;
  }

  for(int dir=0; dir<P4EST_DIM; ++dir) {
    dxyz[dir] = (v[3*v_p + dir] - v[3*v_m + dir]) / (1<<data->max_lvl);

  }

  if (dxyz_min != NULL) {
    *dxyz_min = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));
  }

  if (diag_min != NULL) {
    *diag_min = ABSD(dxyz[0], dxyz[1], dxyz[2]);
  }
}

void dxyz_quad(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz)
{
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  double qh = P4EST_QUADRANT_LEN(quad->level) / (double) P4EST_ROOT_LEN;
  for(int dir=0; dir<P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p+dir]-v[3*v_m+dir]) * qh;
}

void xyz_min(const p4est_t *p4est, double *xyz_min_)
{
  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0;
  p4est_topidx_t first_vertex = 0;

  for (short i=0; i<3; i++)
    xyz_min_[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
}

void xyz_max(const p4est_t *p4est, double *xyz_max_)
{
  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<3; i++)
    xyz_max_[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}

#ifdef P4_TO_P8
void fill_quad_oct_values_from_node_sampled_vector(OctValue* quad_val, const p4est_locidx_t& quad_idx, const p4est_nodes_t* nodes, const double** node_sampled_values_p, const unsigned int n_fields = 1)
#else
void fill_quad_oct_values_from_node_sampled_vector(QuadValue* quad_val, const p4est_locidx_t& quad_idx, const p4est_nodes_t* nodes, const double** node_sampled_values_p, const unsigned int n_fields = 1)
#endif
{
  const p4est_locidx_t *q2n = nodes->local_nodes + P4EST_CHILDREN*quad_idx;

  for (u_char inc_x = 0; inc_x < 2; ++inc_x)
    for (u_char inc_y = 0; inc_y < 2; ++inc_y)
#ifdef P4_TO_P8
      for (u_char inc_z = 0; inc_z < 2; ++inc_z)
#endif
      {
        const u_char idx_in_quad_oct_value = SUMD((1 << (P4EST_DIM - 1))*inc_x, (1 << (P4EST_DIM - 2))*inc_y, inc_z);
        const p4est_locidx_t node_sub_idx_in_quad = q2n[SUMD(inc_x, 2*inc_y, 4*inc_z)];
        for (unsigned int ff = 0; ff < n_fields; ++ff)
          quad_val[ff].val[idx_in_quad_oct_value] = node_sampled_values_p[ff][node_sub_idx_in_quad];
      }
}

double integrate_over_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue  phi_and_function_quad_oct_values[2];
#else
  QuadValue phi_and_function_quad_oct_values[2];
#endif

  const double *node_sampled_phi_and_function_p[2];
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecGetArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);

  fill_quad_oct_values_from_node_sampled_vector(phi_and_function_quad_oct_values, quad_idx, nodes, node_sampled_phi_and_function_p, 2);

  ierr = VecRestoreArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);

  const p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  const p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  const double* tree_xyx_min = p4est->connectivity->vertices + 3*v_m;
  const double* tree_xyz_max = p4est->connectivity->vertices + 3*v_p;
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

#ifdef P4_TO_P8
  Cube3 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]), 0, dmin*(tree_xyz_max[2] - tree_xyx_min[2]));
#else
  Cube2 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]));
#endif

  return cube.integral(phi_and_function_quad_oct_values[1], phi_and_function_quad_oct_values[0]);
}


double integrate_over_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += integrate_over_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                            quad_idx + tree->quadrants_offset,
                                                            phi, f);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}


void integrate_over_negative_domain(int num, double *values, const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec map, Vec f)
{
  PetscErrorCode ierr;

  const double *map_ptr;
  ierr = VecGetArrayRead(map, &map_ptr); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
  for (int i = 0; i < num; ++i) values[i] = 0;

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      // count how many times each index appears in a quadrant
      std::vector<int> count(num, 0);
      for (int i = 0; i < P4EST_CHILDREN; ++i)
      {
        int loc_idx = int(map_ptr[q2n[quad_idx*P4EST_CHILDREN + i]]);
        if (loc_idx >= num) throw;
        if (loc_idx >= 0) count[loc_idx]++;
      }

      // select the most frequent one
      int idx       = 0;
      int max_count = count[0];
      for (int i = 1; i < num; ++i)
      {
        if (max_count < count[i])
        {
          max_count = count[i];
          idx = i;
        }
      }

      // add intergal to appropriate value
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      values[idx] += integrate_over_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                                    quad_idx + tree->quadrants_offset,
                                                                    phi, f);
    }
  }

  ierr = VecRestoreArrayRead(map, &map_ptr); CHKERRXX(ierr);

  /* compute global sum */
  ierr = MPI_Allreduce(MPI_IN_PLACE, values, num, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
}


double area_in_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{

#ifdef P4_TO_P8
  OctValue phi_values;
#else
  QuadValue phi_values;
#endif

  const double *P;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &P); CHKERRXX(ierr);
  fill_quad_oct_values_from_node_sampled_vector(&phi_values, quad_idx, nodes, &P);
  ierr = VecRestoreArrayRead(phi, &P); CHKERRXX(ierr);

  const p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  const p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  const double* tree_xyx_min = p4est->connectivity->vertices + 3*v_m;
  const double* tree_xyz_max = p4est->connectivity->vertices + 3*v_p;
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

#ifdef P4_TO_P8
  Cube3 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]), 0, dmin*(tree_xyz_max[2] - tree_xyx_min[2]));
  return cube.volume_In_Negative_Domain(phi_values);
#else
  Cube2 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]));
  return cube.area_In_Negative_Domain(phi_values);
#endif
}

double area_in_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += area_in_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                     quad_idx + tree->quadrants_offset,
                                                     phi);
    }
  }

  /* compute global sum */
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum;
}

void area_in_negative_domain(int num, double *values, const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec map)
{
  PetscErrorCode ierr;

  const double *map_ptr;
  ierr = VecGetArrayRead(map, &map_ptr); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
  for (int i = 0; i < num; ++i) values[i] = 0;

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      // count how many times each index appears in a quadrant
      std::vector<int> count(num, 0);
      for (int i = 0; i < P4EST_CHILDREN; ++i)
      {
        int loc_idx = int(map_ptr[q2n[quad_idx*P4EST_CHILDREN + i]]);
        if (loc_idx >= num) throw;
        if (loc_idx >= 0) count[loc_idx]++;
      }

      // select the most frequent one
      int idx = 0;
      int max_count = count[0];
      for (int i = 1; i < num; ++i)
      {
        if (max_count < count[i])
        {
          max_count = count[i];
          idx = i;
        }
      }

      // add intergal to appropriate value
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      values[idx] += area_in_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                     quad_idx + tree->quadrants_offset,
                                                     phi);
    }
  }

  ierr = VecRestoreArrayRead(map, &map_ptr); CHKERRXX(ierr);

  /* compute global sum */
  ierr = MPI_Allreduce(MPI_IN_PLACE, values, num, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
}

double integrate_over_interface_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue  phi_and_function_quad_oct_values[2];
#else
  QuadValue phi_and_function_quad_oct_values[2];
#endif

  const double *node_sampled_phi_and_function_p[2];
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecGetArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);

  fill_quad_oct_values_from_node_sampled_vector(phi_and_function_quad_oct_values, quad_idx, nodes, node_sampled_phi_and_function_p, 2);

  ierr = VecRestoreArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);

  const p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  const p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  const double* tree_xyx_min = p4est->connectivity->vertices + 3*v_m;
  const double* tree_xyz_max = p4est->connectivity->vertices + 3*v_p;
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

#ifdef P4_TO_P8
  Cube3 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]), 0, dmin*(tree_xyz_max[2] - tree_xyx_min[2]));
#else
  Cube2 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]));
#endif
  return cube.integrate_Over_Interface(phi_and_function_quad_oct_values[1], phi_and_function_quad_oct_values[0]);
}

double max_over_interface_in_one_quadrant(const p4est_nodes_t *nodes, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue  phi_and_function_quad_oct_values[2];
#else
  QuadValue phi_and_function_quad_oct_values[2];
#endif

  const double *node_sampled_phi_and_function_p[2];
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecGetArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);

  fill_quad_oct_values_from_node_sampled_vector(phi_and_function_quad_oct_values, quad_idx, nodes, node_sampled_phi_and_function_p, 2);

  ierr = VecRestoreArrayRead(phi, &node_sampled_phi_and_function_p[0]); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(f  , &node_sampled_phi_and_function_p[1]); CHKERRXX(ierr);


#ifdef P4_TO_P8
  Cube3 cube(0, 1, 0, 1, 0, 1);
#else
  Cube2 cube(0, 1, 0, 1);
#endif

  return cube.max_Over_Interface(phi_and_function_quad_oct_values[1], phi_and_function_quad_oct_values[0]);
}

double integrate_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);

      sum += integrate_over_interface_in_one_quadrant(p4est, nodes, quad,
                                                      quad_idx + tree->quadrants_offset,
                                                      phi, f);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

void integrate_over_interface(int num, double *values, const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec map, Vec f)
{
  PetscErrorCode ierr;

  const double *map_ptr;
  ierr = VecGetArrayRead(map, &map_ptr); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
  for (int i = 0; i < num; ++i) values[i] = 0;

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      // count how many times each index appears in a quadrant
      std::vector<int> count(num, 0);
      for (int i = 0; i < P4EST_CHILDREN; ++i)
      {
        int loc_idx = int(map_ptr[q2n[quad_idx*P4EST_CHILDREN + i]]);
        if (loc_idx >= num) throw;
        if (loc_idx >= 0) count[loc_idx]++;
      }

      // select the most frequent one
      int idx = 0;
      int max_count = count[0];
      for (int i = 1; i < num; ++i)
      {
        if (max_count < count[i])
        {
          max_count = count[i];
          idx = i;
        }
      }

      // add intergal to appropriate value
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      values[idx] += integrate_over_interface_in_one_quadrant(p4est, nodes, quad,
                                                              quad_idx + tree->quadrants_offset,
                                                              phi, f);
    }
  }

  ierr = VecRestoreArrayRead(map, &map_ptr); CHKERRXX(ierr);

  /* compute global sums */
  ierr = MPI_Allreduce(MPI_IN_PLACE, values, num, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
}

double max_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double max_over_interface = -DBL_MAX;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
      max_over_interface = MAX(max_over_interface, max_over_interface_in_one_quadrant(nodes, quad_idx + tree->quadrants_offset, phi, f));
  }

  /* compute global sum */
  double max_over_interface_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&max_over_interface, &max_over_interface_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);
  return max_over_interface_global;
}

void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec grad_phi[P4EST_DIM], Vec grad_phi_block, Vec phi_xxyyzz[P4EST_DIM], Vec phi_xxyyzz_block, Vec kappa)
{
#ifdef CASL_THROWS
  if(grad_phi == NULL && grad_phi_block == NULL)
    throw std::invalid_argument("compute_mean_curvature: grad phi must be provided (either by block or by component) when computing curvature.");
#endif

  PetscErrorCode ierr;
  const double *phi_p;
  const double *grad_phi_block_p = NULL;
  const double *phi_xxyyzz_block_p = NULL;
  const double **grad_phi_p   = (grad_phi == NULL   ? NULL : new const double* [P4EST_DIM]);
  const double **phi_xxyyzz_p = (phi_xxyyzz == NULL ? NULL : new const double* [P4EST_DIM]);
  double *kappa_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  if(grad_phi_block != NULL){
    ierr = VecGetArrayRead(grad_phi_block, &grad_phi_block_p); CHKERRXX(ierr); }
  if(phi_xxyyzz_block != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz_block, &phi_xxyyzz_block_p); CHKERRXX(ierr); }
  foreach_dimension(dim) {
    if(grad_phi != NULL){
      ierr = VecGetArrayRead(grad_phi[dim], &grad_phi_p[dim]); CHKERRXX(ierr); }
    if(phi_xxyyzz != NULL){
      ierr = VecGetArrayRead(phi_xxyyzz[dim], &phi_xxyyzz_p[dim]); CHKERRXX(ierr); }
  }

  // compute kappa on layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for (size_t i=0; i<neighbors.get_layer_size(); ++i) {
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = qnnn.get_curvature(phi_p, grad_phi_block_p, grad_phi_p, phi_xxyyzz_block_p, phi_xxyyzz_p);
  }

  // initiate communication
  ierr = VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // compute on local nodes
  for (size_t i=0; i<neighbors.get_local_size(); ++i) {
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = qnnn.get_curvature(phi_p, grad_phi_block_p, grad_phi_p, phi_xxyyzz_block_p, phi_xxyyzz_p);
  }

  // finish communication
  ierr = VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  if(grad_phi_block != NULL){
    ierr = VecRestoreArrayRead(grad_phi_block, &grad_phi_block_p); CHKERRXX(ierr); }
  if(phi_xxyyzz_block != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz_block, &phi_xxyyzz_block_p); CHKERRXX(ierr); }
  foreach_dimension(dim) {
    if(grad_phi != NULL){
      ierr = VecRestoreArrayRead(grad_phi[dim], &grad_phi_p[dim]); CHKERRXX(ierr); }
    if(phi_xxyyzz != NULL){
      ierr = VecRestoreArrayRead(phi_xxyyzz[dim], &phi_xxyyzz_p[dim]); CHKERRXX(ierr); }
  }

  if(grad_phi_p != NULL)
    delete[] grad_phi_p;
  if(phi_xxyyzz_p != NULL)
    delete[]  phi_xxyyzz_p;
  return;
}

void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec normals[], Vec kappa)
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals cannot be NULL when computing curvature.");
#endif

  const double *normals_p[P4EST_DIM];
  double *kappa_p;
  PetscErrorCode ierr;
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  foreach_dimension(dim) {ierr = VecGetArrayRead(normals[dim], &normals_p[dim]); CHKERRXX(ierr); }

  // compute kappa on layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for (size_t i=0; i<neighbors.get_layer_size(); ++i) {
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = qnnn.get_curvature(normals_p);
  }

  // initiate communication
  ierr = VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // compute on local nodes
  for (size_t i=0; i<neighbors.get_local_size(); ++i) {
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = qnnn.get_curvature(normals_p);
  }

  // finish communication
  ierr = VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  foreach_dimension(dim) { ierr = VecRestoreArrayRead(normals[dim], &normals_p[dim]); CHKERRXX(ierr); }
}

void compute_normals(const quad_neighbor_nodes_of_node_t &qnnn, double *phi, double normals[])
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  normals[0] = qnnn.dx_central(phi);
  normals[1] = qnnn.dy_central(phi);
#ifdef P4_TO_P8
  normals[2] = qnnn.dz_central(phi);
#endif
  double abs = sqrt(SUMD(SQR(normals[0]), SQR(normals[1]), SQR(normals[2])));
  if (abs < EPS)
    foreach_dimension(dim) normals[dim] = 0;
  else
    foreach_dimension(dim) normals[dim] /= abs;
}

void compute_normals(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec normals[])
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  neighbors.first_derivatives_central(phi, normals);
  double *normals_p[P4EST_DIM];
  foreach_dimension(dim) VecGetArray(normals[dim], &normals_p[dim]);

  foreach_node(n, neighbors.get_nodes()) {
    double abs = sqrt(SUMD(SQR(normals_p[0][n]), SQR(normals_p[1][n]), SQR(normals_p[2][n])));

    if (abs < EPS) {
      foreach_dimension(dim) normals_p[dim][n] = 0;
    } else {
      foreach_dimension(dim) normals_p[dim][n] /= abs;
    }
  }

  foreach_dimension(dim) VecRestoreArray(normals[dim], &normals_p[dim]);
}

void compute_normals(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec normals)
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  neighbors.first_derivatives_central(phi, normals);
  double *normals_p;
  PetscErrorCode ierr = VecGetArray(normals, &normals_p); CHKERRXX(ierr);

  foreach_node(n, neighbors.get_nodes()) {
    double abs = 0.0;
    foreach_dimension(dim) abs += SQR(normals_p[P4EST_DIM*n+dim]);
    abs = sqrt(abs);
    if(abs < EPS){
      foreach_dimension(dim) normals_p[P4EST_DIM*n+dim] = 0.0;
    } else{
      foreach_dimension(dim) normals_p[P4EST_DIM*n+dim] /= abs;
    }
  }
  ierr = VecRestoreArray(normals, &normals_p); CHKERRXX(ierr);
}

double interface_length_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{
#ifdef P4_TO_P8
  OctValue phi_values;
#else
  QuadValue phi_values;
#endif
  const double *P;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &P); CHKERRXX(ierr);
  fill_quad_oct_values_from_node_sampled_vector(&phi_values, quad_idx, nodes, &P);
  ierr = VecRestoreArrayRead(phi, &P); CHKERRXX(ierr);

  const p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  const p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  const double* tree_xyx_min = p4est->connectivity->vertices + 3*v_m;
  const double* tree_xyz_max = p4est->connectivity->vertices + 3*v_p;
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

#ifdef P4_TO_P8
  Cube3 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]), 0, dmin*(tree_xyz_max[2] - tree_xyx_min[2]));
  return cube.interface_Area_In_Cell(phi_values);
#else
  Cube2 cube(0, dmin*(tree_xyz_max[0] - tree_xyx_min[0]), 0, dmin*(tree_xyz_max[1] - tree_xyx_min[1]));
  return cube.interface_Length_In_Cell(phi_values);
#endif
}

double interface_length(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += interface_length_in_one_quadrant(p4est, nodes, quad,
                                              quad_idx + tree->quadrants_offset,
                                              phi);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_m00] != tr_it)
    return false;
  else if (ni->x == 0)
    return true;
  else
    return false;
}

bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_p00] != tr_it)
    return false;
  else if (ni->x == P4EST_ROOT_LEN - 1 || ni->x == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0m0] != tr_it)
    return false;
  else if (ni->y == 0)
    return true;
  else
    return false;
}

bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0p0] != tr_it)
    return false;
  else if (ni->y == P4EST_ROOT_LEN - 1 || ni->y == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

#ifdef P4_TO_P8
bool is_node_zmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00m] != tr_it)
    return false;
  else if (ni->z == 0)
    return true;
  else
    return false;
}

bool is_node_zpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00p] != tr_it)
    return false;
  else if (ni->z == P4EST_ROOT_LEN - 1 || ni->z == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}
#endif

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  return ( is_node_xmWall(p4est, ni) || is_node_xpWall(p4est, ni)
           || is_node_ymWall(p4est, ni) || is_node_ypWall(p4est, ni)
           ONLY3D(|| is_node_zmWall(p4est, ni) || is_node_zpWall(p4est, ni)));
}

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni, bool is_wall[])
{
  bool is_any = false;

  is_wall[dir::f_m00] = is_node_xmWall(p4est, ni); is_any = is_any || is_wall[dir::f_m00];
  is_wall[dir::f_p00] = is_node_xpWall(p4est, ni); is_any = is_any || is_wall[dir::f_p00];
  is_wall[dir::f_0m0] = is_node_ymWall(p4est, ni); is_any = is_any || is_wall[dir::f_0m0];
  is_wall[dir::f_0p0] = is_node_ypWall(p4est, ni); is_any = is_any || is_wall[dir::f_0p0];
#ifdef P4_TO_P8
  is_wall[dir::f_00m] = is_node_zmWall(p4est, ni); is_any = is_any || is_wall[dir::f_00m];
  is_wall[dir::f_00p] = is_node_zpWall(p4est, ni); is_any = is_any || is_wall[dir::f_00p];
#endif
  return is_any;
}

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni, const u_char& oriented_dir)
{
  switch(oriented_dir)
  {
  case dir::f_m00: return is_node_xmWall(p4est, ni);
  case dir::f_p00: return is_node_xpWall(p4est, ni);
  case dir::f_0m0: return is_node_ymWall(p4est, ni);
  case dir::f_0p0: return is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  case dir::f_00m: return is_node_zmWall(p4est, ni);
  case dir::f_00p: return is_node_zpWall(p4est, ni);
#endif
  default:
    throw std::invalid_argument("[CASL_ERROR]: is_node_wall: unknown direction.");
  }
}

bool is_quad_xmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_m00] != tr_it)
    return false;
  else if (qi->x == 0)
    return true;
  else
    return false;
}

bool is_quad_xpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_p00] != tr_it)
    return false;
  else if (qi->x == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}

bool is_quad_ymWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0m0] != tr_it)
    return false;
  else if (qi->y == 0)
    return true;
  else
    return false;
}

bool is_quad_ypWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_0p0] != tr_it)
    return false;
  else if (qi->y == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}

#ifdef P4_TO_P8
bool is_quad_zmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00m] != tr_it)
    return false;
  else if (qi->z == 0)
    return true;
  else
    return false;
}

bool is_quad_zpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_00p] != tr_it)
    return false;
  else if (qi->z == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}
#endif

bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi, int dir)
{
  switch(dir)
  {
  case dir::f_m00: return is_quad_xmWall(p4est, tr_it, qi);
  case dir::f_p00: return is_quad_xpWall(p4est, tr_it, qi);
  case dir::f_0m0: return is_quad_ymWall(p4est, tr_it, qi);
  case dir::f_0p0: return is_quad_ypWall(p4est, tr_it, qi);
#ifdef P4_TO_P8
  case dir::f_00m: return is_quad_zmWall(p4est, tr_it, qi);
  case dir::f_00p: return is_quad_zpWall(p4est, tr_it, qi);
#endif
  default:
    throw std::invalid_argument("[CASL_ERROR]: is_quad_wall: unknown direction.");
  }
}

bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  return ( is_quad_xmWall(p4est, tr_it, qi) || is_quad_xpWall(p4est, tr_it, qi)
           || is_quad_ymWall(p4est, tr_it, qi) || is_quad_ypWall(p4est, tr_it, qi)
           ONLY3D(|| is_quad_zmWall(p4est, tr_it, qi) || is_quad_zpWall(p4est, tr_it, qi)));
}

int quad_find_ghost_owner(const p4est_ghost_t *ghost, const p4est_locidx_t &ghost_idx, int r_down, int r_up)
{
  P4EST_ASSERT(ghost_idx < (p4est_locidx_t) ghost->ghosts.elem_count);
  P4EST_ASSERT(0 <= r_down && r_down < r_up && r_up <= ghost->mpisize);
  P4EST_ASSERT(ghost->proc_offsets[r_down] <= ghost_idx && ghost_idx < ghost->proc_offsets[r_up]);
  while(r_up - r_down > 1)
  {
    int r = (r_down + r_up)/2;
    if(ghost->proc_offsets[r] <= ghost_idx)
      r_down = r;
    else
      r_up = r;
  }
  return r_down;
}

void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_DIM& cf, Vec f)
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    PetscInt size;
    ierr = VecGetLocalSize(f, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->num_owned_indeps << ", " << nodes->indep_nodes.elem_count << ", "
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax - tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax - tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax - tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

    f_p[i] = cf(DIM(x, y, z));
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

void sample_cf_on_nodes(const p4est_t *p4est, const p4est_nodes_t *nodes, const CF_DIM& cf, Vec f)
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_const_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

    f_p[i] = cf(DIM(x, y, z));
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

void sample_cf_on_nodes(const p4est_t *p4est, const p4est_nodes_t *nodes, const CF_DIM* cf_array[], Vec f)
{
  double *f_p;
  PetscInt bs;
  PetscErrorCode ierr;
  ierr = VecGetBlockSize(f, &bs); CHKERRXX(ierr);

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->indep_nodes.elem_count * bs){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points x block_size."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " block_size = " << bs
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);

    for (PetscInt j = 0; j<bs; j++) {
      const CF_DIM& cf = *cf_array[j];
      f_p[i*bs + j] = cf(xyz);
    }
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}


void sample_cf_on_nodes(const p4est_t *p4est, const p4est_nodes_t *nodes, const CF_DIM& cf, std::vector<double>& f)
{
#ifdef CASL_THROWS
  {
    if ((PetscInt) f.size() != (PetscInt) nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " VecSize = " << f.size() << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_const_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

    f[i] = cf(DIM(x, y, z));
  }
}

void sample_cf_on_cells(const p4est_t *p4est, const p4est_ghost_t *ghost, const CF_DIM& cf, Vec f)
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    PetscInt num_local = (PetscInt)(p4est->local_num_quadrants + ghost->ghosts.elem_count);

    if (size != num_local){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             " p4est->local_num_quadrants + ghost->ghosts.elem_count = " << num_local
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  // sample on local quadrants
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      double x = quad_x_fr_q(quad_idx, tree_id, p4est, ghost);
      double y = quad_y_fr_q(quad_idx, tree_id, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(quad_idx, tree_id, p4est, ghost);
#endif

      f_p[quad_idx] = cf(DIM(x, y, z));
    }
  }

  // sample on ghost quadrants
  for (size_t q = 0; q < ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_const_array_index(&ghost->ghosts, q);
    p4est_topidx_t tree_id  = quad->p.piggy3.which_tree;
    p4est_locidx_t quad_idx = q + p4est->local_num_quadrants;

    double x = quad_x_fr_q(quad_idx, tree_id, p4est, ghost);
    double y = quad_y_fr_q(quad_idx, tree_id, p4est, ghost);
#ifdef P4_TO_P8
    double z = quad_z_fr_q(quad_idx, tree_id, p4est, ghost);
#endif

    f_p[quad_idx] = cf(DIM(x, y, z));
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_DIM& cf, std::vector<double>& f)
{
#ifdef CASL_THROWS
  {
    if (f.size() != nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " size() = " << f.size() << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif
  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    const p4est_indep_t *node = (const p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax - tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax - tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax - tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

    f[i] = cf(DIM(x, y, z));
  }
}

std::ostream& operator<< (std::ostream& os, BoundaryConditionType type)
{
  switch(type){
  case DIRICHLET:
    os << "Dirichlet";
    break;

  case NEUMANN:
    os << "Neumann";
    break;

  case ROBIN:
    os << "Robin";
    break;

  case NOINTERFACE:
    os << "No-Interface";
    break;

  case MIXED:
    os << "Mixed";
    break;

  default:
    os << "UNKNOWN";
    break;
  }

  return os;
}


std::istream& operator>> (std::istream& is, BoundaryConditionType& type)
{
  std::string str;
  is >> str;

  if (case_insenstive_str_compare(str, "dirichlet"))
    type = DIRICHLET;
  else if (case_insenstive_str_compare(str, "neumann"))
    type = NEUMANN;
  else if (case_insenstive_str_compare(str, "robin"))
    type = ROBIN;
  else if (case_insenstive_str_compare(str, "nointerface") || case_insenstive_str_compare(str, "no-interface"))
    type = NOINTERFACE;
  else if (case_insenstive_str_compare(str, "mixed"))
    type = MIXED;
  else
    throw std::invalid_argument("[ERROR]: Unknown BoundaryConditionType entered");

  return is;
}

std::ostream& operator << (std::ostream& os, interpolation_method method)
{
  switch(method){
  case linear:
    os << "linear interpolation";
    break;

  case quadratic:
    os << "quadratic interpolation";
    break;

  case quadratic_non_oscillatory:
    os << "quadratic non-oscillatory";
    break;

  case quadratic_non_oscillatory_continuous_v1:
    os << "quadratic non-oscillatory (version 1, by D. Bochkov)";
    break;

  case quadratic_non_oscillatory_continuous_v2:
      os << "quadratic non-oscillatory (version 2, by D. Bochkov)";
    break;

  default:
    os << "unknown interpolation technique";
    break;
  }

  return os;
}

std::istream& operator >> (std::istream& is, interpolation_method& method)
{
  std::string str;
  is >> str;

  if (case_insenstive_str_compare(str, "linear") || case_insenstive_str_compare(str, "lin") || case_insenstive_str_compare(str, "lin."))
    method = linear;
  else if (case_insenstive_str_compare(str, "quadratic") || case_insenstive_str_compare(str, "quad") || case_insenstive_str_compare(str, "quad."))
    method = quadratic;
  else if (case_insenstive_str_compare(str, "quadratic_non_oscillatory") || case_insenstive_str_compare(str, "quad_non_oscillatory")
           || case_insenstive_str_compare(str, "quadratic_minmod") || case_insenstive_str_compare(str, "quad_minmod") || case_insenstive_str_compare(str, "quad_min_mod"))
    method = quadratic_non_oscillatory;
  else if (case_insenstive_str_compare(str, "quadratic_non_oscillatory_continuous_v1") || case_insenstive_str_compare(str, "quad_non_oscillatory_continuous_v1")
           || case_insenstive_str_compare(str, "quadratic_minmod_continuous_v1") || case_insenstive_str_compare(str, "quad_minmod_continuous_v1") || case_insenstive_str_compare(str, "quad_min_mod_continuous_v1")
           || case_insenstive_str_compare(str, "quadratic_non_oscillatory_v1") || case_insenstive_str_compare(str, "quad_non_oscillatory_v1")
           || case_insenstive_str_compare(str, "quadratic_minmod_v1") || case_insenstive_str_compare(str, "quad_minmod_v1") || case_insenstive_str_compare(str, "quad_min_mod_v1"))
    method = quadratic_non_oscillatory_continuous_v1;
  else if (case_insenstive_str_compare(str, "quadratic_non_oscillatory_continuous_v2") || case_insenstive_str_compare(str, "quad_non_oscillatory_continuous_v2")
           || case_insenstive_str_compare(str, "quadratic_minmod_continuous_v2") || case_insenstive_str_compare(str, "quad_minmod_continuous_v2") || case_insenstive_str_compare(str, "quad_min_mod_continuous_v2")
           || case_insenstive_str_compare(str, "quadratic_non_oscillatory_v2") || case_insenstive_str_compare(str, "quad_non_oscillatory_v2")
           || case_insenstive_str_compare(str, "quadratic_minmod_v2") || case_insenstive_str_compare(str, "quad_minmod_v2") || case_insenstive_str_compare(str, "quad_min_mod_v2"))
    method = quadratic_non_oscillatory_continuous_v2;
  else
    throw std::invalid_argument("[ERROR]: unknown interpolation technique entered");

  return is;
}


std::string convert_to_string(const hodge_control& type)
{
  switch(type){
  case u_component:
    return std::string("u");
    break;
  case v_component:
    return std::string("v");
    break;
#ifdef P4_TO_P8
  case w_component:
    return std::string("w");
    break;
#endif
  case uvw_components:
    return std::string("uvw");
    break;
  case hodge_value:
    return std::string("value");
    break;
  default:
    return std::string("unknown type of hodge_control");
    break;
  }
}


std::ostream& operator<< (std::ostream& os, hodge_control type)
{
  switch(type){
  case u_component:
  case v_component:
#ifdef P4_TO_P8
  case w_component:
#endif
    os << convert_to_string(type) << " velocity component";
    break;
  case uvw_components:
    os << "all velocity components";
    break;
  case hodge_value:
    os << "value of Hodge variable";
    break;
  default:
    return os << "unknown type of hodge_control";
    break;
  }
  return os;
}

std::istream& operator >> (std::istream& is, hodge_control& type)
{
  std::string str;
  is >> str;

  if (str == "U" || str == "u" || str == "X" || str == "x")
    type = u_component;
  else if (str == "V" || str == "v" || str == "Y" || str == "y")
    type = v_component;
#ifdef P4_TO_P8
  else if (str == "W" || str == "w" || str == "Z" || str == "z")
    type = w_component;
#endif
  else if (str == "UVW" || str == "uvw" || str == "all" || str == "XYZ" || str == "xyz")
    type = uvw_components;
  else if (str == "VALUE" || str == "Value" || str == "value")
    type = hodge_value;
  else
    throw std::invalid_argument("[ERROR]: Unknown hodge_control entered");

  return is;
}

std::istream& operator >> (std::istream& is, jump_solver_tag& solver)
{
  std::string str;
  is >> str;

  std::vector<size_t> substr_found_at;
  case_insensitive_find_substr_in_str(str, "xGFM", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = xGFM;
    return is;
  }
  // xGFM not found, look for GFM
  case_insensitive_find_substr_in_str(str, "GFM", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = GFM;
    return is;
  }
  // nor xGFM nor GFM found, look for FV
  case_insensitive_find_substr_in_str(str, "FV", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = FV;
    return is;
  }
  throw std::runtime_error("unkonwn jump_solver_tag");
  return is;
}

double quadrant_interp_t::operator()(DIM(double x, double y, double z)) const
{
  double xyz_node[P4EST_DIM] = {DIM(x, y, z)};

#ifdef CASL_THROWS
  if (F_ == NULL) throw std::invalid_argument("[CASL_ERROR]: Values are not provided for interpolation.");
  if (Fdd_ == NULL && (method_ == quadratic || method_ == quadratic_non_oscillatory) ) throw std::invalid_argument("[CASL_ERROR]: Second order derivatives are not provided for quadratic interpolation.");
#endif

  switch (method_)
  {
    case linear:                    return linear_interpolation                   (p4est_, tree_idx_, *quad_, F_->data(),               xyz_node); break;
    case quadratic:                 return quadratic_interpolation                (p4est_, tree_idx_, *quad_, F_->data(), Fdd_->data(), xyz_node); break;
    case quadratic_non_oscillatory: return quadratic_non_oscillatory_interpolation(p4est_, tree_idx_, *quad_, F_->data(), Fdd_->data(), xyz_node); break;
    default: throw std::domain_error("Wrong type of interpolation\n");
  }
}

void copy_ghosted_vec(Vec input, Vec output)
{
  PetscErrorCode ierr = VecCopyGhost(input, output); CHKERRXX(ierr);
  return;
}

void set_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr = VecSetGhost(vec, scalar); CHKERRXX(ierr);
  return;
}

void shift_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr = VecShiftGhost(vec, scalar); CHKERRXX(ierr);
  return;
}

void scale_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr = VecScaleGhost(vec, scalar); CHKERRXX(ierr);
  return;
}

PetscErrorCode VecCopyGhost(Vec input, Vec output)
{
  PetscErrorCode ierr;
  Vec src, out;
  ierr = VecGhostGetLocalForm(input, &src);      if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(output, &out);     if (ierr != 0) return ierr;
  ierr = VecCopy(src, out);                      if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input, &src);  if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(output, &out); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecSetGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     if (ierr != 0) return ierr;
  ierr = VecSet(ptr, scalar);                 if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(vec, &ptr); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecShiftGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     if (ierr != 0) return ierr;
  ierr = VecShift(ptr, scalar);               if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(vec, &ptr); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecScaleGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     if (ierr != 0) return ierr;
  ierr = VecScale(ptr, scalar);               if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(vec, &ptr); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecPointwiseMultGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(input2, &in2);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(output, &out);     if (ierr != 0) return ierr;
  ierr = VecPointwiseMult(out, in1, in2);        if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input1, &in1); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input2, &in2); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(output, &out); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecAXPYGhost(Vec y, PetscScalar alpha, Vec x)
{
  PetscErrorCode ierr;
  Vec X, Y;
  ierr = VecGhostGetLocalForm(x, &X);     CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(y, &Y);     CHKERRQ(ierr);
  ierr = VecAXPY(Y, alpha, X);            CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(x, &X); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(y, &Y); CHKERRQ(ierr);
  return ierr;
}
PetscErrorCode VecAXPBYGhost(Vec y, PetscScalar alpha, PetscScalar beta, Vec x)
{
  PetscErrorCode ierr;
  Vec X, Y;
  ierr = VecGhostGetLocalForm(x, &X);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(y, &Y);     if (ierr != 0) return ierr;
  ierr = VecAXPBY(Y, alpha, beta, X);     if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(x, &X); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(y, &Y); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecPointwiseMinGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(input2, &in2);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(output, &out);     if (ierr != 0) return ierr;
  ierr = VecPointwiseMin(out, in1, in2);         if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input1, &in1); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input2, &in2); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(output, &out); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecPointwiseMaxGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(input2, &in2);     if (ierr != 0) return ierr;
  ierr = VecGhostGetLocalForm(output, &out);     if (ierr != 0) return ierr;
  ierr = VecPointwiseMax(out, in1, in2);         if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input1, &in1); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input2, &in2); if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(output, &out); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecReciprocalGhost(Vec input)
{
  PetscErrorCode ierr;
  Vec in;
  ierr = VecGhostGetLocalForm(input, &in);     if (ierr != 0) return ierr;
  ierr = VecReciprocal(in);                    if (ierr != 0) return ierr;
  ierr = VecGhostRestoreLocalForm(input, &in); if (ierr != 0) return ierr;
  return ierr;
}

PetscErrorCode VecGhostUpdate(Vec input, InsertMode insert_mode, ScatterMode scatter_mode)
{
  PetscErrorCode ierr;
  ierr = VecGhostUpdateBegin(input, insert_mode, scatter_mode); if (ierr != 0) return ierr;
  ierr = VecGhostUpdateEnd  (input, insert_mode, scatter_mode); if (ierr != 0) return ierr;
  return ierr;
}

void invert_phi(p4est_nodes_t *nodes, Vec phi)
{
  PetscErrorCode ierr;
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  for (size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
    phi_p[n] = -phi_p[n];

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
}


void normalize_gradient( Vec gradient[P4EST_DIM], const p4est_nodes_t *nodes )
{
	if( !gradient || ORD( !gradient[0], !gradient[1], !gradient[2] ) )
		throw std::invalid_argument( "[CASL_ERROR] normalize_gradient: The gradient or any of its components can't be null!" );

	double *normalPtr[P4EST_DIM];
	foreach_dimension( dim )
		CHKERRXX( VecGetArray(gradient[dim], &normalPtr[dim]) );

	foreach_node( n, nodes )
	{
		double norm = sqrt( SUMD( SQR( normalPtr[0][n] ), SQR( normalPtr[1][n] ), SQR( normalPtr[2][n] ) ) );
		for( auto& component : normalPtr )
			component[n] = norm < EPS ? 0 : component[n] / norm;
	}

	foreach_dimension( dim )
		CHKERRXX( VecRestoreArray( gradient[dim], &normalPtr[dim] ) );
}


void compute_normals_and_mean_curvature( const my_p4est_node_neighbors_t &neighbors, const Vec& phi, Vec normals[], Vec kappa )
{
  // Compute first derivatives.
  neighbors.first_derivatives_central( phi, normals );

  // Compute curvature using compact stencils (i.e., with non-normalized gradient).
  compute_mean_curvature( neighbors, phi, normals, kappa );

  // Compute normals (by normalizing the gradient).
  normalize_gradient( normals, neighbors.get_nodes() );
}


#ifdef P4_TO_P8
void compute_gaussian_curvature( const my_p4est_node_neighbors_t &ngbd, const Vec& phi, const Vec gradient[P4EST_DIM],
								 const Vec phi_xxyyzz[P4EST_DIM], Vec kappaG )
{
	bool allSecondDerivativesGiven = phi_xxyyzz && ANDD( phi_xxyyzz[0], phi_xxyyzz[1], phi_xxyyzz[2] );
	if( !gradient || !kappaG || (!phi && !allSecondDerivativesGiven) )
		throw std::invalid_argument( "[CASL_ERROR] Gradient and kappaG vectors and both phi and second derivatives "
									 "vectors can't be null!" );

	const double *phiReadPtr = nullptr;
	const double *gradientReadPtr[P4EST_DIM] = {nullptr, nullptr, nullptr};
	const double **phi_xxyyzzReadPtr = phi_xxyyzz? new const double *[P4EST_DIM] : nullptr;
	double *kappaGPtr = nullptr;
	if( phi )
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecGetArray( kappaG, &kappaGPtr ) );

	foreach_dimension( dim )
	{
		CHKERRXX( VecGetArrayRead( gradient[dim], &gradientReadPtr[dim] ) );
		if( allSecondDerivativesGiven )										// If second derivatives, all must
			VecGetArrayRead( phi_xxyyzz[dim], &phi_xxyyzzReadPtr[dim] );	// not be null.
	}

	// Compute kappaG on layer nodes.
	quad_neighbor_nodes_of_node_t qnnn{};
	for( size_t i = 0; i < ngbd.get_layer_size(); i++ )
	{
		p4est_locidx_t n = ngbd.get_layer_node( i );
		ngbd.get_neighbors( n, qnnn );
		kappaGPtr[n] = qnnn.get_gaussian_curvature( phiReadPtr, gradientReadPtr, phi_xxyyzzReadPtr );
	}

	// Initiate layer communication.
	CHKERRXX( VecGhostUpdateBegin( kappaG, INSERT_VALUES, SCATTER_FORWARD ) );

	// Compute on local nodes.
	for( size_t i = 0; i < ngbd.get_local_size(); i++ )
	{
		p4est_locidx_t n = ngbd.get_local_node( i );
		ngbd.get_neighbors( n, qnnn );
		kappaGPtr[n] = qnnn.get_gaussian_curvature( phiReadPtr, gradientReadPtr, phi_xxyyzzReadPtr );
	}

	// Finish layer communication.
	CHKERRXX( VecGhostUpdateEnd( kappaG, INSERT_VALUES, SCATTER_FORWARD ) );

	if( phi )
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecRestoreArray( kappaG, &kappaGPtr ) );
	foreach_dimension(dim)
	{
		CHKERRXX( VecRestoreArrayRead( gradient[dim], &gradientReadPtr[dim] ) );
		if( phi_xxyyzz )
			CHKERRXX( VecRestoreArrayRead( phi_xxyyzz[dim], &phi_xxyyzzReadPtr[dim] ) );
	}

	delete [] phi_xxyyzzReadPtr;
}


void compute_normals_and_curvatures( const my_p4est_node_neighbors_t& ngbd, const Vec& phi, Vec normals[P4EST_DIM],
									 Vec kappaM, Vec kappaG, Vec kappa12[2] )
{
	if( !phi || !normals || !kappaM || !kappaG || !kappa12 )	// Invalid vectors?
		throw std::invalid_argument( "[CASL_ERROR] compute_normals_and_curvatures: None of the input/output vectors can be null!" );

	const p4est_nodes_t *nodes = ngbd.get_nodes();
	ngbd.first_derivatives_central( phi, normals );				// Compute (non-normalized) gradient.

	// Reuse these derivatives to compute kappaG.
	compute_gaussian_curvature( ngbd, phi, normals, nullptr, kappaG );

	// Normalize gradient vector to return unit normals and compute more robust mean curvature.
	normalize_gradient( normals, nodes );

	// Compute the (doubled) mean curvature.
	compute_mean_curvature( ngbd, normals, kappaM );

	// Preparing principal curvature computation.
	double *kappaMPtr;
	CHKERRXX( VecGetArray( kappaM, &kappaMPtr ) );

	const double *kappaGReadPtr;								// Accessing Gaussian curvature.
	CHKERRXX( VecGetArrayRead( kappaG, &kappaGReadPtr ) );

	double *kappa12Ptr[2] = {nullptr, nullptr};					// Accessing principal curvatures vectors.
	for( int i = 0; i < 2; i++ )
		CHKERRXX( VecGetArray( kappa12[i], &kappa12Ptr[i] ) );

	auto computePrincipalCurvatures = [kappaMPtr, kappaGReadPtr, kappa12Ptr]( p4est_locidx_t n ){
		kappaMPtr[n] *= 0.5;				// Lets get the right value: kappaM = 0.5*div(n) = 0.5*(k1 + k2).
		double radical = sqrt( ABS( SQR( kappaMPtr[n] ) - kappaGReadPtr[n] ) );
		kappa12Ptr[0][n] = kappaMPtr[n] + radical;	// First principal curvature.
		kappa12Ptr[1][n] = kappaMPtr[n] - radical;	// Second principal curvature.
	};

	// Compute principal curvatures and *half* the *doubled* mean curvature on layer nodes.
	for( size_t i = 0; i < ngbd.get_layer_size(); i++ )
	{
		p4est_locidx_t n = ngbd.get_layer_node( i );
		computePrincipalCurvatures( n );
	}

	// Initiate layer communication.
	CHKERRXX( VecGhostUpdateBegin( kappa12[0], INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateBegin( kappa12[1], INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateBegin( kappaM, INSERT_VALUES, SCATTER_FORWARD ) );

	// Compute principal curvatures and *half* of the *doubled* mean curvature on local nodes.
	for( size_t i = 0; i < ngbd.get_local_size(); i++ )
	{
		p4est_locidx_t n = ngbd.get_local_node( i );
		computePrincipalCurvatures( n );
	}

	// Finish layer communication.
	CHKERRXX( VecGhostUpdateEnd( kappa12[0], INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( kappa12[1], INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( kappaM, INSERT_VALUES, SCATTER_FORWARD ) );

	// Clean up.
	for( int i = 0; i < 2; i++ )
		CHKERRXX( VecRestoreArray( kappa12[i], &kappa12Ptr[i] ) );
	CHKERRXX( VecRestoreArrayRead( kappaG, &kappaGReadPtr ) );
	CHKERRXX( VecRestoreArray( kappaM, &kappaMPtr ) );
}
#endif


void save_vector(const char *filename, const std::vector<double> &data, std::ios_base::openmode mode, char delim)
{
  std::ofstream ofs;
  ofs.open(filename, mode);

  for (unsigned int i = 0; i < data.size(); ++i)
  {
    if (i != 0) ofs << delim;
    ofs << data[i];
  }

  ofs << "\n";
}






void fill_island(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, int number, p4est_locidx_t n)
{
  const p4est_nodes_t *nodes = ngbd.get_nodes();

    std::stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        island_number_p[k] = number;
        const quad_neighbor_nodes_of_node_t& qnnn = ngbd[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && island_number_p[qnnn.node_m00_mm]<0) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && island_number_p[qnnn.node_m00_pm]<0) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && island_number_p[qnnn.node_p00_mm]<0) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && island_number_p[qnnn.node_p00_pm]<0) st.push(qnnn.node_p00_pm);

        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && island_number_p[qnnn.node_0m0_mm]<0) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && island_number_p[qnnn.node_0m0_pm]<0) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && island_number_p[qnnn.node_0p0_mm]<0) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && island_number_p[qnnn.node_0p0_pm]<0) st.push(qnnn.node_0p0_pm);
    }
}


void find_connected_ghost_islands(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited)
{
  const p4est_nodes_t *nodes = ngbd.get_nodes();

    std::stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        visited[k] = true;
        const quad_neighbor_nodes_of_node_t& qnnn = ngbd[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !visited[qnnn.node_m00_mm]) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !visited[qnnn.node_m00_pm]) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !visited[qnnn.node_p00_mm]) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !visited[qnnn.node_p00_pm]) st.push(qnnn.node_p00_pm);

        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !visited[qnnn.node_0m0_mm]) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !visited[qnnn.node_0m0_pm]) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !visited[qnnn.node_0p0_mm]) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !visited[qnnn.node_0p0_pm]) st.push(qnnn.node_0p0_pm);

        /* check connected ghost island and add to list if new */
        if(qnnn.node_m00_mm>=nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !contains(connected, island_number_p[qnnn.node_m00_mm])) connected.push_back(island_number_p[qnnn.node_m00_mm]);
        if(qnnn.node_m00_pm>=nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !contains(connected, island_number_p[qnnn.node_m00_pm])) connected.push_back(island_number_p[qnnn.node_m00_pm]);
        if(qnnn.node_p00_mm>=nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !contains(connected, island_number_p[qnnn.node_p00_mm])) connected.push_back(island_number_p[qnnn.node_p00_mm]);
        if(qnnn.node_p00_pm>=nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !contains(connected, island_number_p[qnnn.node_p00_pm])) connected.push_back(island_number_p[qnnn.node_p00_pm]);

        if(qnnn.node_0m0_mm>=nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_mm])) connected.push_back(island_number_p[qnnn.node_0m0_mm]);
        if(qnnn.node_0m0_pm>=nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_pm])) connected.push_back(island_number_p[qnnn.node_0m0_pm]);
        if(qnnn.node_0p0_mm>=nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_mm])) connected.push_back(island_number_p[qnnn.node_0p0_mm]);
        if(qnnn.node_0p0_pm>=nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_pm])) connected.push_back(island_number_p[qnnn.node_0p0_pm]);
    }
}


void compute_islands_numbers(const my_p4est_node_neighbors_t &ngbd, const Vec phi, int &nb_islands_total, Vec island_number)
{
  PetscErrorCode ierr;

  const p4est_t       *p4est = ngbd.get_p4est();
  const p4est_nodes_t *nodes = ngbd.get_nodes();

  nb_islands_total = 0;
  int proc_padding = 1e6;
//  return;

  Vec loc;
  ierr = VecGhostGetLocalForm(island_number, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(island_number, &loc); CHKERRXX(ierr);

  /* first everyone compute the local numbers */
  std::vector<int> nb_islands(p4est->mpisize);
  nb_islands[p4est->mpirank] = p4est->mpirank*proc_padding;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  double *island_number_p;
  ierr = VecGetArray(island_number, &island_number_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd.get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd.get_layer_node(i);
    if(phi_p[n]>0 && island_number_p[n]<0)
    {
      fill_island(ngbd, phi_p, island_number_p, nb_islands[p4est->mpirank], n);
      nb_islands[p4est->mpirank]++;
    }
  }
  ierr = VecGhostUpdateBegin(island_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd.get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd.get_local_node(i);
    if(phi_p[n]>0 && island_number_p[n]<0)
    {
      fill_island(ngbd, phi_p, island_number_p, nb_islands[p4est->mpirank], n);
      nb_islands[p4est->mpirank]++;
    }
  }
  ierr = VecGhostUpdateEnd(island_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* get remote number of islands to prepare graph communication structure */
  int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &nb_islands[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  /* compute offset for each process */
  std::vector<int> proc_offset(p4est->mpisize+1);
  proc_offset[0] = 0;
  for(int p=0; p<p4est->mpisize; ++p)
    proc_offset[p+1] = proc_offset[p] + (nb_islands[p]%proc_padding);

  /* build a local graph with
         *   - vertices = island number
         *   - edges    = connected islands
         * in order to simplify the communications, the graph is stored as a full matrix. Given the sparsity, this can be optimized ...
         */
  int nb_islands_g = proc_offset[p4est->mpisize];
  std::vector<int> graph(nb_islands_g*nb_islands_g, 0);
  /* note that the only reason this is double and not int is that Petsc works with doubles, can't do Vec of int ... */
  std::vector<double> connected;
  std::vector<bool> visited(nodes->num_owned_indeps, false);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(island_number_p[n]>=0 && !visited[n])
    {
      /* find the connected islands and add the connection information to the graph */
      find_connected_ghost_islands(ngbd, phi_p, island_number_p, n, connected, visited);
      for(unsigned int i=0; i<connected.size(); ++i)
      {
        int local_id = proc_offset[p4est->mpirank]+static_cast<int>(island_number_p[n])%proc_padding;
        int remote_id = proc_offset[static_cast<int>(connected[i])/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
        graph[nb_islands_g*local_id + remote_id] = 1;
      }

      connected.clear();
    }
  }

  std::vector<int> rcvcounts(p4est->mpisize);
  std::vector<int> displs(p4est->mpisize);
  for(int p=0; p<p4est->mpisize; ++p)
  {
    rcvcounts[p] = (nb_islands[p]%proc_padding) * nb_islands_g;
    displs[p] = proc_offset[p]*nb_islands_g;
  }

  mpiret = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &graph[0], &rcvcounts[0], &displs[0], MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  /* now we can color the graph connecting the islands, and thus obtain a unique numbering for all the islands */
  std::vector<int> graph_numbering(nb_islands_g,-1);
  std::stack<int> st;
  for(int i=0; i<nb_islands_g; ++i)
  {
    if(graph_numbering[i]==-1)
    {
      st.push(i);
      while(!st.empty())
      {
        int k = st.top();
        st.pop();
        graph_numbering[k] = nb_islands_total;
        for(int j=0; j<nb_islands_g; ++j)
        {
          int nj = k*nb_islands_g+j;
          if(graph[nj] && graph_numbering[j]==-1)
            st.push(j);
        }
      }
      nb_islands_total++;
    }
  }

  /* and finally assign the correct number to the islands of this level */
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if(island_number_p[n]>=0)
    {
      int index = proc_offset[static_cast<int>(island_number_p[n])/proc_padding] + (static_cast<int>(island_number_p[n])%proc_padding);
      island_number_p[n] = graph_numbering[index];
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(island_number, &island_number_p); CHKERRXX(ierr);
}

void compute_phi_eff(Vec phi_eff, p4est_nodes_t *nodes, std::vector<Vec> &phi, std::vector<mls_opn_t> &opn)
{
  PetscErrorCode ierr;
  double* phi_eff_ptr;
  ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

  std::vector<double *> phi_ptr(phi.size(), NULL);

  for (size_t i = 0; i < phi.size(); i++) { ierr = VecGetArray(phi.at(i), &phi_ptr[i]); CHKERRXX(ierr); }

  foreach_node(n, nodes)
  {
    double phi_total = -DBL_MAX;
    for (unsigned int i = 0; i < phi.size(); i++)
    {
      double phi_current = phi_ptr[i][n];

      if      (opn.at(i) == MLS_INTERSECTION) phi_total = MAX(phi_total, phi_current);
      else if (opn.at(i) == MLS_ADDITION)     phi_total = MIN(phi_total, phi_current);
    }
    phi_eff_ptr[n] = phi_total;
  }

//  if (refine_always != NULL)
//    foreach_node(n, nodes)
//      for (unsigned int i = 0; i < phi->size(); i++)
//        if (refine_always->at(i))
//          phi_eff_ptr[n] = MIN(phi_eff_ptr[n], fabs(phi_ptr[i][n]));

  for (size_t i = 0; i < phi.size(); i++) { ierr = VecRestoreArray(phi.at(i), &phi_ptr[i]); CHKERRXX(ierr); }
}

void compute_phi_eff(Vec phi_eff, p4est_nodes_t *nodes, int num_phi, ...)
{
  va_list ap;

  va_start(ap, num_phi);

  std::vector<Vec> phi;
  std::vector<mls_opn_t> opn;
  for (int i=0; i<num_phi; ++i) {
    Vec       P = va_arg(ap, Vec);              phi.push_back(P);
    mls_opn_t O = (mls_opn_t) va_arg(ap, int);  opn.push_back(O);
  }

  va_end(ap);

  compute_phi_eff(phi_eff, nodes, phi, opn);
}


void find_closest_interface_location(int &phi_idx, double &dist, double d, std::vector<mls_opn_t> opn,
                                     std::vector<double> &phi_a,
                                     std::vector<double> &phi_b,
                                     std::vector<double> &phi_a_xx,
                                     std::vector<double> &phi_b_xx)
{
  dist    = d;
  phi_idx =-1;

  for (size_t i = 0; i < opn.size(); ++i)
  {
    if (phi_a[i] > 0. && phi_b[i] > 0.)
    {
      if (opn[i] == MLS_INTERSECTION)
      {
        dist    =  0;
        phi_idx = -1;
      }
    } else if (phi_a[i] < 0. && phi_b[i] < 0.) {
      if (opn[i] == MLS_ADDITION)
      {
        dist    =  d;
        phi_idx = -1;
      }
    } else {
      double dist_new = interface_Location_With_Second_Order_Derivative(0., d, phi_a[i], phi_b[i], phi_a_xx[i], phi_b_xx[i]);

      switch (opn[i])
      {
      case MLS_INTERSECTION:
        if (phi_a[i] < 0.)
        {
          if (dist_new < dist)
          {
            dist    = dist_new;
            phi_idx = i;
          }
        } else {
          dist    =  0;
          phi_idx = -1;
        }
        break;
      case MLS_ADDITION:
        if (phi_a[i] < 0.)
        {
          if (dist_new > dist)
          {
            dist    = dist_new;
            phi_idx = i;
          }
        } else {
          if (dist_new < dist)
          {
            dist    =  d;
            phi_idx = -1;
          }
        }
        break;
      default:
#ifdef CASL_THROWS
        throw  std::runtime_error("find_closest_interface_location:: unknown MLS operation: only MLS_INTERSECTION and MLS_ADDITION implemented, currently.");
#endif
        break;
      }
    }
  }
}

void construct_finite_volume(my_p4est_finite_volume_t& fv,
                             const double* xyz_C, const double* dxyz, const bool* is_wall,
                             const std::vector<const CF_DIM *>& phi, const std::vector<mls_opn_t>& opn,
                             const int& order, const int& cube_refinement, const bool& compute_centroids, const double& perturb)
{
  const double scale = 1./MAX(DIM(dxyz[0], dxyz[1], dxyz[2]));
  const double diag  = sqrt(SUMD(SQR(dxyz[0]), SQR(dxyz[1]), SQR(dxyz[2])));

  // Reconstruct geometry
  double cube_xyz_min[] = { DIM( 0, 0, 0 ) };
  double cube_xyz_max[] = { DIM( 0, 0, 0 ) };
  int    cube_mnk[]     = { DIM( 0, 0, 0 ) };

  CODE2D( cube2_mls_t cube );
  CODE3D( cube3_mls_t cube );

  // determine dimensions of cube
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(is_wall == NULL || !is_wall[2*dim])    { cube_mnk[dim] += cube_refinement; cube_xyz_min[dim] -= .5*dxyz[dim]*scale;  }
    if(is_wall == NULL || !is_wall[2*dim + 1]){ cube_mnk[dim] += cube_refinement; cube_xyz_max[dim] += .5*dxyz[dim]*scale;  }
  }

  fv.full_cell_volume = MULTD( cube_xyz_max[0] - cube_xyz_min[0],
                               cube_xyz_max[1] - cube_xyz_min[1],
                               cube_xyz_max[2] - cube_xyz_min[2] ) / pow(scale, P4EST_DIM);

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    fv.full_face_area[2*dim] = fv.full_face_area[2*dim + 1] = (cube_xyz_max[(dim + 1)%P4EST_DIM] - cube_xyz_min[(dim + 1)%P4EST_DIM]) CODE3D(*(cube_xyz_max[(dim + 2)%P4EST_DIM] - cube_xyz_min[(dim + 2)%P4EST_DIM])) / pow(scale, P4EST_DIM - 1);

  if (cube_refinement == 0) cube_mnk[0] = cube_mnk[1] = CODE3D( cube_mnk[2] = ) 1;

  cube.initialize(cube_xyz_min, cube_xyz_max, cube_mnk, order);

  // get points at which values of level-set functions are needed
  XCODE( std::vector<double> x_grid; cube.get_x_coord(x_grid); );
  YCODE( std::vector<double> y_grid; cube.get_y_coord(y_grid); );
  ZCODE( std::vector<double> z_grid; cube.get_z_coord(z_grid); );

  int    points_total = x_grid.size();
  double num_phi      = phi.size();

  std::vector<double> phi_cube(num_phi*points_total, -1);

  // compute values of level-set functions at needed points
  for (int phi_idx=0; phi_idx<num_phi; ++phi_idx)
  {
    for (int i=0; i<points_total; ++i)
    {
      phi_cube[phi_idx*points_total + i] = (*phi[phi_idx])(DIM(xyz_C[0] + x_grid[i]/scale, xyz_C[1] + y_grid[i]/scale, xyz_C[2] + z_grid[i]/scale));

      // push interfaces inside the domain
      if (fabs(phi_cube[phi_idx*points_total + i]) < perturb*diag)
      {
        phi_cube[phi_idx*points_total + i] = perturb*diag;
      }
    }
  }

  std::vector<int> clr(num_phi);
  for (int i=0; i<num_phi; ++i) clr[i] = i;

  // reconstruct geometry
  reconstruct_cube(cube, phi_cube, opn, clr);

  // get quadrature points
  _CODE( std::vector<double> qp_w );
  XCODE( std::vector<double> qp_x );
  YCODE( std::vector<double> qp_y );
  ZCODE( std::vector<double> qp_z );

  cube.quadrature_over_domain(qp_w, DIM(qp_x, qp_y, qp_z));

  // compute cut-cell volume
  fv.volume = 0;
  for (size_t i=0; i<qp_w.size(); ++i)
    fv.volume += qp_w[i];

  fv.volume /= pow(scale, P4EST_DIM);

  // compute areas and centroinds of interfaces
  fv.interfaces.clear();

  for (int phi_idx=0; phi_idx<num_phi; ++phi_idx)
  {
    cube.quadrature_over_interface(phi_idx, qp_w, DIM(qp_x, qp_y, qp_z));
    if (qp_w.size() > 0)
    {
      interface_info_t data;

      _CODE( data.id          = phi_idx );
      _CODE( data.area        = 0 );
      XCODE( data.centroid[0] = 0 );
      YCODE( data.centroid[1] = 0 );
      ZCODE( data.centroid[2] = 0 );

      for (size_t i=0; i<qp_w.size(); ++i)
      {
        _CODE( data.area        += qp_w[i]         );
        XCODE( data.centroid[0] += qp_w[i]*qp_x[i] );
        YCODE( data.centroid[1] += qp_w[i]*qp_y[i] );
        ZCODE( data.centroid[2] += qp_w[i]*qp_z[i] );
      }

      XCODE( data.centroid[0] /= scale*data.area);
      YCODE( data.centroid[1] /= scale*data.area);
      ZCODE( data.centroid[2] /= scale*data.area);
      _CODE( data.area        /= pow(scale, P4EST_DIM-1) );

      fv.interfaces.push_back(data);
    }
  }

  // compute cut-face areas and their centroids
  for (int dir_idx=0; dir_idx<P4EST_FACES; ++dir_idx)
  {
    _CODE( fv.face_area      [dir_idx] = 0 );
    XCODE( fv.face_centroid_x[dir_idx] = 0 );
    YCODE( fv.face_centroid_y[dir_idx] = 0 );
    ZCODE( fv.face_centroid_z[dir_idx] = 0 );

    cube.quadrature_in_dir(dir_idx, qp_w, DIM(qp_x, qp_y, qp_z));
    if (qp_w.size() > 0)
    {
      for (size_t i=0; i<qp_w.size(); ++i) fv.face_area[dir_idx] += qp_w[i];

      if (compute_centroids)
      {
        for (size_t i=0; i<qp_w.size(); ++i)
        {
          XCODE( fv.face_centroid_x[dir_idx] += qp_w[i]*qp_x[i] );
          YCODE( fv.face_centroid_y[dir_idx] += qp_w[i]*qp_y[i] );
          ZCODE( fv.face_centroid_z[dir_idx] += qp_w[i]*qp_z[i] );
        }

        XCODE( fv.face_centroid_x[dir_idx] /= scale*fv.face_area[dir_idx] );
        YCODE( fv.face_centroid_y[dir_idx] /= scale*fv.face_area[dir_idx] );
        ZCODE( fv.face_centroid_z[dir_idx] /= scale*fv.face_area[dir_idx] );
      }

      fv.face_area[dir_idx] /= pow(scale, P4EST_DIM-1);
    }
  }

  XCODE( fv.face_centroid_x[dir::f_m00] = cube_xyz_min[0]/scale; fv.face_centroid_x[dir::f_p00] = cube_xyz_max[0]/scale );
  YCODE( fv.face_centroid_y[dir::f_0m0] = cube_xyz_min[1]/scale; fv.face_centroid_y[dir::f_0p0] = cube_xyz_max[1]/scale );
  ZCODE( fv.face_centroid_z[dir::f_00m] = cube_xyz_min[2]/scale; fv.face_centroid_z[dir::f_00p] = cube_xyz_max[2]/scale );
  return;
}


void compute_wall_normal(const int &dir, double normal[])
{
  switch (dir)
  {
    case dir::f_m00: normal[0] =-1; normal[1] = 0; CODE3D( normal[2] = 0;) break;
    case dir::f_p00: normal[0] = 1; normal[1] = 0; CODE3D( normal[2] = 0;) break;

    case dir::f_0m0: normal[0] = 0; normal[1] =-1; CODE3D( normal[2] = 0;) break;
    case dir::f_0p0: normal[0] = 0; normal[1] = 1; CODE3D( normal[2] = 0;) break;
#ifdef P4_TO_P8
    case dir::f_00m: normal[0] = 0; normal[1] = 0; CODE3D( normal[2] =-1;) break;
    case dir::f_00p: normal[0] = 0; normal[1] = 0; CODE3D( normal[2] = 1;) break;
#endif
    default:
      throw std::invalid_argument("Invalid direction\n");
  }
}

double interface_point_cartesian_t::interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr)
{
  const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

  p4est_locidx_t neigh = qnnn.neighbor(dir);
  double         h     = qnnn.distance(dir);

  return (ptr[n]*(h-dist) + ptr[neigh]*dist)/h;
}

double interface_point_cartesian_t::interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr, double *ptr_dd[P4EST_DIM])
{
  const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

  p4est_locidx_t neigh = qnnn.neighbor(dir);
  double         h     = qnnn.distance(dir);
  short          dim   = dir / 2;

  double p0  = ptr[n];
  double p1  = ptr[neigh];
  double pdd = MINMOD(ptr_dd[dim][n], ptr_dd[dim][neigh]);

  return .5*(p0+p1) + (p1-p0)*(dist/h-.5) + .5*pdd*(dist*dist-dist*h);
}

PetscErrorCode vec_and_ptr_t::ierr;
PetscErrorCode vec_and_ptr_cells_t::ierr;

PetscErrorCode vec_and_ptr_dim_t::ierr;
PetscErrorCode vec_and_ptr_array_t::ierr;

// Generalized smoothstep

// Returns binomial coefficient without explicit use of factorials,
// which can't be used with negative integers
double pascalTriangle(int a, int b) {
  double result = 1.;
  for (int i = 0; i < b; ++i)
    result *= double(a - i) / double(i + 1);
  return result;
}

double smoothstep(int N, double x) {
  if      (x <= 0) return 0;
  else if (x >= 1) return 1;
  else {
    double result = 0;
    for (int n = 0; n <= N; ++n)
    {
      result += pascalTriangle(-N - 1, n) *
                pascalTriangle(2 * N + 1, N - n) *
                pow(x, N + n + 1);
    }
    return result;
  }
}

void variable_step_BDF_implicit(const int order, std::vector<double> &dt, std::vector<double> &coeffs)
{
  coeffs.assign(order+1, 0);
  std::vector<double> r(order-1, 1.);

  if (dt.size() < (size_t) order) throw;

  switch (order)
  {
    case 1:
      coeffs[0] =  1;
      coeffs[1] = -1;
      break;
    case 2:
      r[0] = dt[0]/dt[1];

      coeffs[0] = (1.+2.*r[0])/(1.+r[0]);
      coeffs[1] = -(1.+r[0]);
      coeffs[2] = SQR(r[0])/(1.+r[0]);
      break;
    case 3:
      r[0] = dt[0]/dt[1];
      r[1] = dt[1]/dt[2];

      coeffs[0] = 1. + r[0]/(1.+r[0]) + r[1]*r[0]/(1.+r[1]*(1.+r[0]));
      coeffs[1] = -1. - r[0] - r[0]*r[1]*(1.+r[0])/(1.+r[1]);
      coeffs[2] = SQR(r[0])*(r[1] + 1./(1.+r[0]));
      coeffs[3] = - pow(r[1], 3.)*pow(r[0], 2)*(1.+r[0])/(1.+r[1])/(1.+r[1]+r[1]*r[0]);
      break;
    case 4:
    {
      r[0] = dt[0]/dt[1];
      r[1] = dt[1]/dt[2];
      r[2] = dt[2]/dt[3];

      double a1 = 1.+r[2]*(1.+r[1]);
      double a2 = 1.+r[1]*(1.+r[0]);
      double a3 = 1.+r[2]*a2;

      coeffs[0] = 1. + r[0]/(1.+r[0]) + r[1]*r[0]/a2 + r[2]*r[1]*r[0]/a3;
      coeffs[1] = -1.-r[0]*(1.+r[1]*(1.+r[0])/(1.+r[1])*(1.+r[2]*a2/a1));
      coeffs[2] = r[0]*(r[0]/(1.+r[0]) + r[1]*r[0]*(a3+r[2])/(1.+r[2]));
      coeffs[3] = -pow(r[1],3.)*pow(r[0],2.)*(1.+r[0])/(1.+r[1])*a3/a2;
      coeffs[4] = (1.+r[0])/(1+r[2])*a2/a1*pow(r[2],4.)*pow(r[1],3.)*pow(r[0],2.)/a3;
    }
      break;
    default:
      throw;
  }
}


void getStencil( const quad_neighbor_nodes_of_node_t *qnnnPtr, const double *f, double data[P4EST_DIM][2][2] )
{
	// Some convenient arrangement nodal solution values and their distances w.r.t. center node in its stencil of neighbors.
	data[0][0][0] = qnnnPtr->f_m00_linear( f ); data[0][0][1] = qnnnPtr->d_m00;			// Left.
	data[0][1][0] =	qnnnPtr->f_p00_linear( f ); data[0][1][1] = qnnnPtr->d_p00;			// Right.

	data[1][0][0] = qnnnPtr->f_0m0_linear( f ); data[1][0][1] = qnnnPtr->d_0m0;			// Bottom.
	data[1][1][0] = qnnnPtr->f_0p0_linear( f ); data[1][1][1] = qnnnPtr->d_0p0;			// Top.
#ifdef P4_TO_P8
	data[2][0][0] = qnnnPtr->f_00m_linear( f ); data[2][0][1] = qnnnPtr->d_00m;			// Back.
	data[2][1][0] = qnnnPtr->f_00p_linear( f ); data[2][1][1] = qnnnPtr->d_00p;			// Front.
#endif
}


void truncate_exportation_file_up_to_tstart(const double& tstart, const std::string &filename, const u_int& n_extra_values_exported_per_line)
{
  FILE* fp = fopen(filename.c_str(), "r+");
  char* read_line = NULL;
  size_t len = 0;
  ssize_t len_read;
  long size_to_keep = 0;
  if(((len_read = getline(&read_line, &len, fp)) != -1))
    size_to_keep += (long) len_read;
  else
    throw std::runtime_error("truncate_exportation_file_up_to_tstart: couldn't read the first header line of " + filename);
  std::string read_format = "%lg";
  for (u_int k = 0; k < n_extra_values_exported_per_line; ++k)
    read_format += " %*g";
  double time, time_nm1;
  double dt = 0.0;
  bool not_first_line = false;
  while ((len_read = getline(&read_line, &len, fp)) != -1) {
    if(not_first_line)
      time_nm1 = time;
    sscanf(read_line, read_format.c_str(), &time);
    if(not_first_line)
      dt = time - time_nm1;
    if(time <= tstart + 0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
      size_to_keep += (long) len_read;
    else
      break;
    not_first_line = true;
  }
  fclose(fp);
  if(read_line)
    free(read_line);
  if(truncate(filename.c_str(), size_to_keep))
    throw std::runtime_error("truncate_exportation_file_up_to_tstart: couldn't truncate " + filename);
  return;
}


