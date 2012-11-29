
#include <sc_search.h>
#include <p4est_bits.h>
#include <p4est_communication.h>
#include "my_p4est_tools.h"
#include "utils.h"

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

  p4est_topidx_t p4est_mm = ttv[0];
  p4est_topidx_t p4est_pp = ttv[num_trees * P4EST_CHILDREN - 1];

  double d_xmin = vv[3*p4est_mm + 0];
  double d_ymin = vv[3*p4est_mm + 1];
  double d_xmax = vv[3*p4est_pp + 0];
  double d_ymax = vv[3*p4est_pp + 1];

  double xy_tmp[] = {xy[0], xy[1]};

  if (xy_tmp[0] <= d_xmin) xy_tmp[0] = d_xmin;
  if (xy_tmp[0] >= d_xmax) xy_tmp[0] = d_xmax;
  if (xy_tmp[1] <= d_ymin) xy_tmp[1] = d_ymin;
  if (xy_tmp[1] >= d_ymax) xy_tmp[1] = d_ymax;

  /* Assuming a brick connectivity with no coordinate transformation */
  if ((xy_tmp[0] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 0]  &&
       xy_tmp[0] <= vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
      (xy_tmp[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
       xy_tmp[1] <= vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
    /* we have found the tree unchanged */
  }
  else {
    /* we need to search through all trees */
    /* this could be optimized for the brick */
    for (tt = 0; tt < num_trees; ++tt) {
      if (tt == *which_tree) {
        continue;
      }
      if ((xy_tmp[0] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 0]  &&
           xy_tmp[0] <= vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
          (xy_tmp[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
           xy_tmp[1] <= vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
        *which_tree = tt;
        break;
      }
    }
    P4EST_ASSERT (tt < num_trees);
  }

  xyoffset[0] = xy_tmp[0] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 0];
  P4EST_ASSERT (xyoffset[0] >= 0. && xyoffset[0] <= 1.);
  xyoffset[1] = xy_tmp[1] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 1];
  P4EST_ASSERT (xyoffset[1] >= 0. && xyoffset[1] <= 1.);

  const double rh = 1.0/(double)P4EST_ROOT_LEN;
  if (xyoffset[0]<=rh) xyoffset[0] = rh;
  if (xyoffset[0]>=1.0-rh) xyoffset[0] = 1.0 - rh;
  if (xyoffset[1]<=rh) xyoffset[1] = rh;
  if (xyoffset[1]>=1.0-rh) xyoffset[1] = 1.0 - rh;

  /* construct the smallest quadrant containing xy */
  qq.x = (p4est_qcoord_t) (xyoffset[0]*P4EST_ROOT_LEN);
  qq.y = (p4est_qcoord_t) (xyoffset[1]*P4EST_ROOT_LEN);
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

    if (which_quad != NULL) *which_quad = (p4est_locidx_t) position;
    if (quad != NULL) *quad = p4est_quadrant_array_index (&tree->quadrants, position);
  }

  return pp;
}

int my_p4est_brick_point_lookup_smallest(p4est_t * p4est, const double * xy,
                                         p4est_topidx_t *which_tree,
                                         p4est_locidx_t *which_quad,
                                         p4est_quadrant_t **quad)
{
  int rank, rank_tmp;
  p4est_topidx_t tr = 0, tr_tmp = 0;
  p4est_locidx_t qu, qu_tmp;
  p4est_quadrant_t* q, *q_tmp;

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double         *v2q = p4est->connectivity->vertices;

  p4est_topidx_t p4est_mm = t2v[0];
  p4est_topidx_t p4est_pp = t2v[p4est->connectivity->num_trees * P4EST_CHILDREN - 1];

  double domain_xmin = v2q[3*p4est_mm + 0];
  double domain_ymin = v2q[3*p4est_mm + 1];
  double domain_xmax = v2q[3*p4est_pp + 0];
  double domain_ymax = v2q[3*p4est_pp + 1];

  double xy_p[2];

  // + +
  xy_p[0] = xy[0] + EPS; xy_p[1] = xy[1] + EPS;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  rank = my_p4est_brick_point_lookup(p4est, xy_p, &tr, &qu, &q);

  // + -
  xy_p[0] = xy[0] + EPS; xy_p[1] = xy[1] - EPS;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, xy_p, &tr_tmp, &qu_tmp, &q_tmp);
  if (q_tmp->level > q->level){
    rank = rank_tmp;
    tr = tr_tmp;
    qu = qu_tmp;
    q  = q_tmp;
  }

  // - +
  xy_p[0] = xy[0] - EPS; xy_p[1] = xy[1] + EPS;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, xy_p, &tr_tmp, &qu_tmp, &q_tmp);
  if (q_tmp->level > q->level){
    rank = rank_tmp;
    tr = tr_tmp;
    qu = qu_tmp;
    q  = q_tmp;
  }

  // - -
  xy_p[0] = xy[0] - EPS; xy_p[1] = xy[1] - EPS;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, xy_p, &tr_tmp, &qu_tmp, &q_tmp);
  if (q_tmp->level > q->level){
    rank = rank_tmp;
    tr = tr_tmp;
    qu = qu_tmp;
    q  = q_tmp;
  }

  if (which_tree != NULL) *which_tree = tr;
  if (which_quad != NULL) *which_quad = qu;
  if (quad       != NULL) *quad       = q;

  return rank;
}

