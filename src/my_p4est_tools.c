
#include <sc_search.h>
#include <p4est_bits.h>
#include <p4est_communication.h>
#include "my_p4est_tools.h"

int my_p4est_brick_point_lookup (p4est_t * p4est, const double * xy,
                                 p4est_topidx_t *which_tree,
                                 p4est_locidx_t *which_quad,
                                 p4est_quadrant_t **quad)
{
  const p4est_connectivity_t * conn = p4est->connectivity;
  const p4est_topidx_t num_trees = conn->num_trees;
  const p4est_topidx_t * ttv = conn->tree_to_vertex;
  const double * vv = conn->vertices;
  const int last = P4EST_CHILDREN - 1;
  double xyoffset[P4EST_DIM];
  int pp;
  size_t position;
  p4est_topidx_t tt = *which_tree;
  p4est_tree_t * tree;
  p4est_quadrant_t qq;

  P4EST_ASSERT (vv != NULL);
  P4EST_ASSERT (0 <= tt && tt < num_trees);


  /* Assuming a brick connectivity with no coordinate transformation */
  if ((xy[0] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 0]  &&
       xy[0] <  vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
      (xy[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
       xy[1] <  vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
    /* we have found the tree unchanged */
  }
  else {
    /* we need to search through all trees */
    /* this could be optimized for the brick */
    for (tt = 0; tt < num_trees; ++tt) {
      if (tt == *which_tree) {
        continue;
      }
      if ((xy[0] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 0]  &&
           xy[0] <  vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
          (xy[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
           xy[1] <  vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
        *which_tree = tt;
        break;
      }
    }
    P4EST_ASSERT (tt < num_trees);
  }
  xyoffset[0] = xy[0] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 0];
  P4EST_ASSERT (xyoffset[0] >= 0. && xyoffset[0] < 1.);
  xyoffset[1] = xy[1] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 1];
  P4EST_ASSERT (xyoffset[1] >= 0. && xyoffset[1] < 1.);

  /* construct the smallest quadrant containing xy */
  qq.x = (p4est_qcoord_t) (xyoffset[0] * P4EST_ROOT_LEN);
  qq.y = (p4est_qcoord_t) (xyoffset[1] * P4EST_ROOT_LEN);
  qq.level = P4EST_QMAXLEVEL;
  P4EST_ASSERT (p4est_quadrant_is_valid (&qq));
  pp = p4est_comm_find_owner (p4est, tt, &qq, p4est->mpirank);
  if (pp == p4est->mpirank) {
    /* this point is processor local */
    tree = p4est_tree_array_index (p4est->trees, tt);
    position = sc_bsearch_range (&qq, tree->quadrants.array,
                                 tree->quadrants.elem_count - 1,
                                 sizeof (p4est_quadrant_t),
                                 p4est_quadrant_compare);
    P4EST_ASSERT (position < tree->quadrants.elem_count);
    if (which_quad != NULL) {
      *which_quad = (p4est_locidx_t) position;
    }
    if (quad != NULL) {
      *quad = p4est_quadrant_array_index (&tree->quadrants, position);
    }
  }

  return pp;
}
