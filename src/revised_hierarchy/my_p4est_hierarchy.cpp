#ifdef P4_TO_P8
#include <p8est_communication.h>
#include <src/point3.h>
#include "my_p8est_hierarchy.h"
#else
#include <p4est_communication.h>
#include <src/point2.h>
#include "my_p4est_hierarchy.h"
#endif
#include "petsc_compatibility.h"

#include <src/types.h>
#include <src/casl_math.h>

#include <stdexcept>
#include <sstream>
#include <petsclog.h>

// logging variable -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_hierarchy_t;
#ifdef CASL_LOG_TINY_EVENTS
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
extern PetscLogEvent log_my_p4est_hierarchy_t_find_smallest_quad;
#endif
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_hierarchy_t::determine_local_and_layer_cells()
{
  PetscErrorCode ierr = PetscLogEventBegin(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);

  if(local_inner_quadrant.size() > 0)
    local_inner_quadrant.clear();
  if(local_layer_quadrant.size() > 0)
    local_layer_quadrant.clear();

  local_inner_quadrant.reserve(p4est->local_num_quadrants);
  local_layer_quadrant.reserve(p4est->local_num_quadrants);

  size_t mirror_idx = 0;
  const p4est_quadrant_t* mirror = NULL;
  if(ghost != NULL && mirror_idx < ghost->mirrors.elem_count)
    mirror = p4est_quadrant_array_index(&ghost->mirrors, mirror_idx++);
  /* loop on the local quadrants */
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = p4est_const_quadrant_array_index(&tree->quadrants, q);
      // mirrors and quadrant are stored using the same convention, parse both simultaneously for efficiency,
      // but do not use p4est_quadrant_is_equal_piggy(), since the p.piggy3 member is not filled for regular quadrants but only for ghosts and mirrors
      if(mirror != NULL && p4est_quadrant_is_equal(quad, mirror) && mirror->p.piggy3.which_tree == tree_idx)
      {
        local_layer_quadrant.push_back(local_and_tree_indices(q + tree->quadrants_offset, tree_idx));
        if(ghost != NULL && mirror_idx < ghost->mirrors.elem_count)
          mirror = p4est_quadrant_array_index(&ghost->mirrors, mirror_idx++);
      }
      else
        local_inner_quadrant.push_back(local_and_tree_indices(q + tree->quadrants_offset, tree_idx));
    }
  }

  local_inner_quadrant.shrink_to_fit();
  local_layer_quadrant.shrink_to_fit();

  P4EST_ASSERT(ghost == NULL || mirror_idx == ghost->mirrors.elem_count);
  P4EST_ASSERT(local_inner_quadrant.size() + local_layer_quadrant.size() == (size_t) p4est->local_num_quadrants);
  ierr = PetscLogEventEnd(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_hierarchy_t::update(p4est_t *p4est_, p4est_ghost_t *ghost_)
{
  p4est = p4est_;
  ghost = ghost_;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    if(periodic[dir] != is_periodic(p4est_, dir))
      throw std::invalid_argument("my_p4est_hierarchy_t::update : cannot update with a change of periodicity!");
  determine_local_and_layer_cells();
  return;
}

void my_p4est_hierarchy_t::get_all_quadrants_in(const p4est_quadrant_t* quad, const p4est_topidx_t& tree_idx, std::vector<p4est_locidx_t>& list_of_local_quad_idx) const
{
//  list_of_local_quad_idx.clear();

//  const HierarchyCell matching_cell = trees[tree_idx][get_index_of_hierarchy_cell_matching_or_containing_quad(quad, tree_idx)];
//  if(matching_cell.level == quad->level) // it is not a bigger hierarchy cell containing the quad of interest, but it's matching
//    matching_cell.add_all_inner_leaves_to(list_of_local_quad_idx, trees[tree_idx]);

  return;
}

int my_p4est_hierarchy_t::find_smallest_quadrant_containing_point(const double *xyz, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches,
                                                                  const bool &prioritize_local, const bool &set_cumulative_local_index_in_piggy3_of_best_match) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_hierarchy_t_find_smallest_quad, 0, 0, 0, 0); CHKERRXX(ierr);
#endif

  /* rescale coordinates to [0, nx] x [0, ny] x [0, nz] where nx, ny and nz are the numbers of
   * trees in the brick, i.e., the numbers of trees along the cartesian directions in the brick
   */
  const double *xyz_min = myb->xyz_min;
  const double *xyz_max = myb->xyz_max;
  const double tree_dimensions[P4EST_DIM] = {DIM((xyz_max[0] - xyz_min[0])/myb->nxyztrees[0], (xyz_max[1] - xyz_min[1])/myb->nxyztrees[1], (xyz_max[2] - xyz_min[2])/myb->nxyztrees[2])};

  // Calculate xyz - xyz_min, first
  double xyz_[P4EST_DIM] = {DIM(xyz[0] - xyz_min[0], xyz[1] - xyz_min[1], xyz[2] - xyz_min[2])};

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    // Wrap within the domain using periodicity if needed
    if (periodic[dim])
      xyz_[dim] -= floor(xyz_[dim]/(xyz_max[dim] - xyz_min[dim]))*(xyz_max[dim] - xyz_min[dim]);
    // at this point, xyz_ MUST be in [0 (xyz_max - xyz_min)]the computational domain --> critical check in DEBUG
    P4EST_ASSERT(0.0 <= xyz_[dim] && xyz_[dim] <= xyz_max[dim] - xyz_min[dim]);
    // scale by dimensions of the trees
    xyz_[dim] /= tree_dimensions[dim];
    // check that it is in [0 nxyz_trees] in DEBUG
    P4EST_ASSERT(0.0 <= xyz_[dim] && xyz_[dim] <= myb->nxyztrees[dim]);
  }

  int rank = -1; // initialize the return value --> this is what is returned if the quadrant of interest is remote
  P4EST_QUADRANT_INIT(&best_match);

  /*
   * At this stage, an integer value for xyz_[i], say "xyz_[i] == nn", theoretically means that the point of interest
   * lies exactly on the border between two trees of cartesian index (nn-1) and nn along cartesian direction i (if
   * both of these trees exist). Therefore, in such a case, one needs to perturb xyz_[i] by a small amount in the
   * positive and negative directions, and search then for both of these perturbed points in the respective trees.
   * However, such a test as "xyz_[i] == nn" will practically never return true since we are considering floating
   * point values for xyz_[i]. We need a finite, domain-independent, floating-point threshold value to determine
   * whether or not we consider that the point of coordinates xyz lies on the limit of a tree. Let us call that value
   * 'thresh' and replace the above test "xyz_[i] == nn" by "fabs(xyz[i]-nn) < thresh". Consistently, the small
   * amount by which we perturb xyz_[i] should be +/-thres, in that case.
   * ------------------------
   * Let's define 'threshold'
   * ------------------------
   * The most important constraint is to not miss the quadrant of interest, so we need to make sure that
   * 2*thresh < (logical) length of the smallest possible quadrant in a p4est grid, divided by P4EST_ROOT_LEN
   *
   * The (logical) length of the smallest quadrant ever possible is
   *      P4EST_QUADRANT_LEN(P4EST_QMAXLEVEL) = P4EST_QUADRANT_LEN(P4EST_MAXLEVEL - 1)
   * so let's define qeps as
   */
  const static double qeps = (double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL) / (double) P4EST_ROOT_LEN;
  /* so that qeps is half the logical length of the smallest possible quadrant as allowed by p4est (divided by P4EST_ROOT_LEN,
   * i.e., scaled down to a measure such that the scaled logical length of a root cell is 1.0)
   * Therefore, the smallest absolute difference between logical coordinate(s) of relevant grid-related data,
   * divided by P4EST_ROOT_LEN is qeps (e.g., difference between coordinates of a vertex and coordinates of
   * the center of the smallest possible quadrant).
   * --> qeps is a strict upper bound for thresh
   *
   * Given that we have 52 bits to represent the mantissa with 64-bit double values, as opposed to 32-bit integers
   * for the p4est_qcoord_t data type, we also have the following strict minimum bound for thresh
   * 2^(log2(max(number of trees along a cartesian direction))-20)*qeps
   * (any value smaller than that would possibly result in a comparison test equivalent to "xyz_[i] == nn").
   * Assuming that we can safely set log2(max(number of trees along a cartesian direction)) = 10,
   * this gives thresh > 0.001*qeps so I suggest to define thresh 10 times bigger, that is
   */
  const static double threshold = 0.01*(double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL); // == thresh*P4EST_ROOT_LEN

  /* In case of nonperiodic domain, we need to make sure that any point lying on the boundary of the domain is clearly
   * and unambiguously clipped inside, without changing the quadrant of interest, before we proceed further.
   * Otherwise, the routine will try to access a tree that does not exist...
   * Clearly and unambiguously clip the point inside the domain if not periodic
   */
  int tr_xyz_orig[P4EST_DIM];
  for (u_char dir = 0; dir < P4EST_DIM; ++dir){
    if (!periodic[dir])
      xyz_[dir] = MAX(qeps, MIN(xyz_[dir], myb->nxyztrees[dir] - qeps));
    tr_xyz_orig[dir] = (int)floor(xyz_[dir]);
  }
  const double ii = (xyz_[0] - tr_xyz_orig[0]) * P4EST_ROOT_LEN;
  const double jj = (xyz_[1] - tr_xyz_orig[1]) * P4EST_ROOT_LEN;
#ifdef P4_TO_P8
  const double kk = (xyz_[2] - tr_xyz_orig[2]) * P4EST_ROOT_LEN;
#endif

  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    if(periodic[dir])
      tr_xyz_orig[dir] = tr_xyz_orig[dir]%myb->nxyztrees[dir]; // important if the point is on the very last face along the periodic direction (ii = 0, but the tree index is off)

  const bool is_on_face_x = (fabs(ii - floor(ii)) < threshold || fabs(ceil(ii) - ii) < threshold);
  const bool is_on_face_y = (fabs(jj - floor(jj)) < threshold || fabs(ceil(jj) - jj) < threshold);
#ifdef P4_TO_P8
  const bool is_on_face_z = (fabs(kk - floor(kk)) < threshold || fabs(ceil(kk) - kk) < threshold);
#endif

  for (char i = (is_on_face_x ? -1 : 0); i < 2; i += 2)
    for (char j = (is_on_face_y ? -1 : 0); j < 2; j += 2)
#ifdef P4_TO_P8
      for (char k = (is_on_face_z ? -1 : 0);  k < 2; k += 2)
#endif
      {
        // perturb the point (note that i, j and/or k are 0 is no perturbation is required)
        double point[P4EST_DIM] = {DIM(ii + i*threshold, jj + j*threshold, kk + k*threshold)};
        find_quadrant_containing_point(tr_xyz_orig, point, rank, best_match, remote_matches, prioritize_local);
      }

  if(set_cumulative_local_index_in_piggy3_of_best_match && rank != -1)
  {
    if(rank == p4est->mpirank)
      best_match.p.piggy3.local_num += p4est_tree_array_index(p4est->trees, best_match.p.piggy3.which_tree)->quadrants_offset;
    else
      best_match.p.piggy3.local_num += p4est->local_num_quadrants;
  }

#ifdef CASL_LOG_TINY_EVENTS
  ierr = PetscLogEventEnd(log_my_p4est_hierarchy_t_find_smallest_quad, 0, 0, 0, 0); CHKERRXX(ierr);
#endif

  return rank;
}

void my_p4est_hierarchy_t::find_quadrant_containing_point(const int* tr_xyz_orig, double* xyz_point, int& current_rank, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches, const bool &prioritize_local) const
{
  int tr_xyz[P4EST_DIM] = {DIM(tr_xyz_orig[0], tr_xyz_orig[1], tr_xyz_orig[2])};

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    if (xyz_point[dir] < 0.0 || xyz_point[dir] > (double) P4EST_ROOT_LEN){
      const int ntree_to_slide = (int) ceil(xyz_point[dir]/((double) P4EST_ROOT_LEN)) - 1;
      P4EST_ASSERT((xyz_point[dir] < 0.0 && ntree_to_slide < 0) || (xyz_point[dir] > (double) P4EST_ROOT_LEN && ntree_to_slide > 0));
      xyz_point[dir] -= ((double) ntree_to_slide)*((double) P4EST_ROOT_LEN); // convert both terms to double BEFORE multiplying to avoid overflow in integer representation!
      tr_xyz[dir] = tr_xyz_orig[dir] + ntree_to_slide;
      if(periodic[dir])
        tr_xyz[dir] = mod(tr_xyz[dir], myb->nxyztrees[dir]);
    }
    P4EST_ASSERT(0 <= tr_xyz[dir] && tr_xyz[dir] < myb->nxyztrees[dir] && 0.0 <= xyz_point[dir] && xyz_point[dir] <= (double) P4EST_ROOT_LEN);
  }

  p4est_topidx_t tt = myb->nxyz_to_treeid[SUMD(tr_xyz[0], tr_xyz[1]*myb->nxyztrees[0], tr_xyz[2]*myb->nxyztrees[0]*myb->nxyztrees[1])];
  P4EST_ASSERT(0 <= tt && tt < p4est->connectivity->num_trees);

  p4est_quadrant_t pixel_quad;
  pixel_quad.level = P4EST_MAXLEVEL;
  pixel_quad.x = (p4est_qcoord_t) floor(xyz_point[0]); if(pixel_quad.x != P4EST_ROOT_LEN - 1) pixel_quad.x = pixel_quad.x & ~(smallest_logical_quad_size - 1); // this operation nullifies the last bit and ensures in the p4est_qcoord_t value, hence ensures divisibility by smallest_logical_quad_size
  pixel_quad.y = (p4est_qcoord_t) floor(xyz_point[1]); if(pixel_quad.y != P4EST_ROOT_LEN - 1) pixel_quad.y = pixel_quad.y & ~(smallest_logical_quad_size - 1);
#ifdef P4_TO_P8
  pixel_quad.z = (p4est_qcoord_t) floor(xyz_point[2]); if(pixel_quad.z != P4EST_ROOT_LEN - 1) pixel_quad.z = pixel_quad.z & ~(smallest_logical_quad_size - 1);
#endif

  bool is_a_local_quadrant;
  p4est_locidx_t pos;
  find_quad_owning_pixel_quad(pixel_quad, tt, pos, is_a_local_quadrant);

  if (pos >= 0) { // local or ghost quadrant
    const p4est_quadrant_t *tmp = p4est_const_quadrant_array_index((is_a_local_quadrant ? &(p4est_tree_array_index(p4est->trees, tt)->quadrants) : &ghost->ghosts), pos);
    // The quadrant was found, now check if it is better than the current candidate
    if (tmp->level > best_match.level || (prioritize_local && is_a_local_quadrant && tmp->level == best_match.level && current_rank != p4est->mpirank)) {
      // note the '(prioritize_local && tmp->level >= best_match.level && rank != p4est->mpirank)' here above
      // --> ensures that we pick a local quadrant over a ghost one, if we find one
      // --> useful for on-the-fly interpolations up to the very border of the local domain's partition!
      best_match = *tmp;
      best_match.p.piggy3.which_tree = tt;
      best_match.p.piggy3.local_num  = pos;
      if(is_a_local_quadrant)
        current_rank = p4est->mpirank;
      else
        current_rank = quad_find_ghost_owner(ghost, pos);
    }
  } else { // remote quadrant
    /* need to find the owner --> returned via piggy1! */
    pixel_quad.p.piggy1.which_tree = tt;
    pixel_quad.p.piggy1.owner_rank = p4est_comm_find_owner(p4est, tt, &pixel_quad, p4est->mpirank);
    remote_matches.push_back(pixel_quad);
  }
  return;
}

void my_p4est_hierarchy_t::find_quad_owning_pixel_quad(const p4est_quadrant_t& pixel_quad, const p4est_topidx_t& tree_idx,
                                                       p4est_locidx_t& pos, bool& is_a_local_quadrant) const
{
  P4EST_ASSERT(pixel_quad.level == P4EST_MAXLEVEL);

  const p4est_tree_t* tree  = ((p4est->first_local_tree <= tree_idx && tree_idx <= p4est->last_local_tree) ? p4est_tree_array_index(p4est->trees, tree_idx) : NULL);
  pos = -1;
  p4est_locidx_t pos_ub = -1;
  if(tree != NULL && p4est_quadrant_disjoint(&pixel_quad, p4est_const_quadrant_array_index(&tree->quadrants, tree->quadrants.elem_count - 1)) <= 0
     && p4est_quadrant_disjoint(p4est_const_quadrant_array_index(&tree->quadrants, 0), &pixel_quad) <= 0)
  {
    pos = 0;
    pos_ub  = tree->quadrants.elem_count;
    is_a_local_quadrant = true;
  }
  else
  {
    if(ghost->tree_offsets[tree_idx + 1] > ghost->tree_offsets[tree_idx])
    {
      pos    = ghost->tree_offsets[tree_idx];
      pos_ub = ghost->tree_offsets[tree_idx + 1];
    }
    is_a_local_quadrant = false;
  }

  if(pos_ub > pos)
  {
    // binary search
    p4est_locidx_t mid_pos = (pos + pos_ub)/2;
    const p4est_quadrant_t *mid_quad = p4est_const_quadrant_array_index((is_a_local_quadrant ? &tree->quadrants : &ghost->ghosts), mid_pos);
    int cmp = p4est_quadrant_disjoint(mid_quad, &pixel_quad);
    while(cmp != 0)
    {
      if(pos_ub <= pos + 1)
        break;
      if(cmp < 0)
        pos = mid_pos;
      else
        pos_ub = mid_pos;

      mid_pos = (pos + pos_ub)/2;
      mid_quad = p4est_const_quadrant_array_index((is_a_local_quadrant ? &tree->quadrants : &ghost->ghosts), mid_pos);
      cmp = p4est_quadrant_disjoint(mid_quad, &pixel_quad);
    }
    pos = (cmp == 0 ? mid_pos : -1); // if cmp != 0, the procedure looked into the ghost layer but did not find it (--> not known locally)
  }
  return;
}

void my_p4est_hierarchy_t::find_neighbor_cell_of_node(const p4est_locidx_t& node_idx, const p4est_nodes_t* nodes, DIM(const char& i, const char& j, const char& k),
                                                      p4est_locidx_t& quad_idx, p4est_topidx_t& owning_tree_idx) const
{
  // make a local copy of the current node structure
  p4est_indep_t node = *(p4est_indep_t *)sc_const_array_index(&nodes->indep_nodes, node_idx);
  // unclamp it
  p4est_node_unclamp((p4est_quadrant_t*) &node);

  P4EST_ASSERT(ANDD(abs(i) == 1, abs(j) == 1, abs(k) == 1));
  // perturb the copied unclamped node in the queried direction (by one logical coordinate unit)
  node.x += i; P4EST_ASSERT(node.x != 0 && node.x != P4EST_ROOT_LEN);
  node.y += j; P4EST_ASSERT(node.y != 0 && node.y != P4EST_ROOT_LEN);
#ifdef P4_TO_P8
  node.z += k; P4EST_ASSERT(node.z != 0 && node.z != P4EST_ROOT_LEN);
#endif
  // make sure it can be found
  if(!is_node_in_domain(node, myb, p4est->connectivity))
  {
    quad_idx = NOT_A_VALID_QUADRANT;
    return;
  }

  // Since it can be found, invoke the hierarchy with the appropriate logical coordinates of the
  // perturbed node and its correct owning tree index to find the (cumulative) index of the queried
  // quadrant
  owning_tree_idx = node.p.which_tree;
  if(node.x != P4EST_ROOT_LEN - 1) node.x = node.x & ~(smallest_logical_quad_size - 1);
  if(node.y != P4EST_ROOT_LEN - 1) node.y = node.y & ~(smallest_logical_quad_size - 1);
#ifdef P4_TO_P8
  if(node.z != P4EST_ROOT_LEN - 1) node.z = node.z & ~(smallest_logical_quad_size - 1);
#endif

  p4est_locidx_t pos;
  bool is_a_local_quad;
  find_quad_owning_pixel_quad((p4est_quadrant_t&)node, owning_tree_idx, pos, is_a_local_quad);
  if(pos >= 0)
  {
    if(is_a_local_quad)
      quad_idx = pos + p4est_tree_array_index(p4est->trees, owning_tree_idx)->quadrants_offset;
    else
      quad_idx = pos + p4est->local_num_quadrants;
  }
  else
    quad_idx = NOT_A_P4EST_QUADRANT;

  return;
}
