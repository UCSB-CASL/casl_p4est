
#include <sc_search.h>
#include <p4est_bits.h>
#include <p4est_communication.h>
#include "my_p4est_tools.h"

static int
my_p4est_brick_point_lookup_real (p4est_t * p4est, p4est_ghost_t * ghost,
                                  const my_p4est_brick_t * myb,
                                  const double * xy,
                                  p4est_topidx_t *which_tree,
                                  p4est_locidx_t *which_quad,
                                  p4est_quadrant_t **quad);

p4est_connectivity_t *
my_p4est_brick_new (int nxtrees, int nytrees, my_p4est_brick_t *myb)
{
  int i, j;
  int nxytrees;
  double dii, djj;
  const double *vv;
  p4est_topidx_t tt, vindex;
  p4est_connectivity_t *conn;

  P4EST_ASSERT (0 < nxtrees && 0 < nytrees);
  conn = p4est_connectivity_new_brick (nxtrees, nytrees, 0, 0);
  vv = conn->vertices;
  P4EST_ASSERT (vv != NULL);

  myb->nxytrees[0] = nxtrees;
  myb->nxytrees[1] = nytrees;
  nxytrees = nxtrees * nytrees;
  myb->nxy_to_treeid = P4EST_ALLOC (p4est_topidx_t, nxytrees);
#ifdef P4EST_DEBUG
  memset (myb->nxy_to_treeid, -1, sizeof (p4est_topidx_t) * nxytrees);
#endif
  for (tt = 0; tt < conn->num_trees; ++tt) {
    /* build lookup structure from tree corner coordinates to treeid */
    vindex = conn->tree_to_vertex[P4EST_CHILDREN * tt + 0];
    P4EST_ASSERT (0 <= vindex && vindex < conn->num_vertices);
    dii = vv[3 * vindex + 0];
    P4EST_ASSERT (dii == floor (dii));
    i = (int) dii;
    P4EST_ASSERT (i >= 0 && i < nxtrees);
    djj = vv[3 * vindex + 1];
    P4EST_ASSERT (djj == floor (djj));
    j = (int) djj;
    P4EST_ASSERT (j >= 0 && j < nytrees);
    P4EST_ASSERT (vv[3 * vindex + 2] == 0.);
    P4EST_ASSERT (myb->nxy_to_treeid[nxtrees * j + i] == -1);
    myb->nxy_to_treeid[nxtrees * j + i] = tt;
  }

  return conn;
}

void
my_p4est_brick_destroy (p4est_connectivity_t *conn, my_p4est_brick_t * myb)
{
  P4EST_ASSERT (myb->nxy_to_treeid != NULL);
  P4EST_FREE (myb->nxy_to_treeid);
  myb->nxy_to_treeid = NULL;
  p4est_connectivity_destroy (conn);
}

int my_p4est_brick_point_lookup (p4est_t * p4est, p4est_ghost_t * ghost,
                                 const my_p4est_brick_t * myb,
                                 const double * xy,
                                 p4est_topidx_t *which_tree,
                                 p4est_locidx_t *which_quad,
                                 p4est_quadrant_t **quad)
{
  /* identify all possible quadrant queries */
  int i, ix, iy;
  int hit;
  int istreeboundary[P4EST_DIM];                /**< bool in each dimension */
  int maybequadboundary[P4EST_DIM];
  int integerstep[P4EST_DIM][2];
  const p4est_qcoord_t qlen = P4EST_QUADRANT_LEN (P4EST_QMAXLEVEL);
  p4est_qcoord_t qq, searchq[P4EST_DIM][2];
  /* p4est_topidx_t treeid[P4EST_CHILDREN]; */

  P4EST_LDEBUGF ("Looking up point %g %g\n", xy[0], xy[1]);

  for (i = 0; i < P4EST_DIM; ++i) {
    hit = (int) floor (xy[i]);
    P4EST_ASSERT (0 <= hit && hit <= myb->nxytrees[i]);
    
    /* check if the xy coordinates are on a tree boundary */
    if ((double) hit == xy[i]) {
      istreeboundary[i] = 1;
      maybequadboundary[i] = 1;
      integerstep[i][0] = hit - 1;      /* can be -1 which will be ignored */
      integerstep[i][1] = hit == myb->nxytrees[i] ? -1 : hit;   /* ditto */
      searchq[i][0] = P4EST_LAST_OFFSET (P4EST_QMAXLEVEL);
      searchq[i][1] = 0;
      P4EST_LDEBUGF ("Dimension %d hit tree %d %d\n", i,
                     integerstep[i][0], integerstep[i][1]);
    }
    else {
      istreeboundary[i] = 0;
      integerstep[i][0] = integerstep[i][1] = hit;
      searchq[i][1] = qq = (int) floor ((xy[i] - hit) * P4EST_ROOT_LEN);
      P4EST_ASSERT (0 <= qq && qq < P4EST_ROOT_LEN);
      qq &= ~(qlen - 1);                /* force qq to a quadrant multiple */

      /* check if the xy[i] coordinate is on a possible quadrant boundary */
      if ((double) qq == (xy[i] - hit) * P4EST_ROOT_LEN) {
        maybequadboundary[i] = 1;
        searchq[i][0] = qq - qlen;
        P4EST_ASSERT (searchq[i][0] >= 0);
      }
      else {
        maybequadboundary[i] = 0;
        integerstep[i][0] = -1;
        searchq[i][0] = -1;
      }
      P4EST_LDEBUGF ("Dimension %d non tree %d %d\n", i,
                     integerstep[i][0], integerstep[i][1]);
    }
  }

  /* go through all P4EST_CHILDREN possible quadrant searches */
  for (iy = 0; iy < 2; ++iy) {
    for (ix = 0; ix < 2; ++ix) {
    }
  }


  return my_p4est_brick_point_lookup_real (p4est, ghost, myb, xy,
                                           which_tree, which_quad, quad);
}

int my_p4est_brick_point_lookup_real (p4est_t * p4est, p4est_ghost_t * ghost,
                                      const my_p4est_brick_t * myb,
                                      const double * xy,
                                      p4est_topidx_t *which_tree,
                                      p4est_locidx_t *which_quad,
                                      p4est_quadrant_t **quad)
{
  const p4est_connectivity_t * conn = p4est->connectivity;
  const p4est_topidx_t num_trees = conn->num_trees;
  const p4est_topidx_t * ttv = conn->tree_to_vertex;
  const double * vv = conn->vertices;
  const int last = P4EST_CHILDREN - 1;
  const p4est_qcoord_t qlen = P4EST_QUADRANT_LEN (P4EST_QMAXLEVEL);
  const double halfqw = .5 / qlen;
  int pp;
  size_t position;
  double xyoffset[P4EST_DIM];
  p4est_topidx_t tt = *which_tree;
  p4est_tree_t * tree;
  p4est_quadrant_t qq;

  const p4est_topidx_t p4est_mm = ttv[0];
  const p4est_topidx_t p4est_pp = ttv[num_trees * P4EST_CHILDREN - 1];

  const double d_xmin = vv[3 * p4est_mm + 0];
  const double d_ymin = vv[3 * p4est_mm + 1];
  const double d_xmax = vv[3 * p4est_pp + 0];
  const double d_ymax = vv[3 * p4est_pp + 1];

  double xy_tmp[] = {xy[0], xy[1]};

  P4EST_ASSERT (vv != NULL);
  P4EST_ASSERT (0 <= tt && tt < num_trees);

  if (xy_tmp[0] < d_xmin) xy_tmp[0] = d_xmin;
  if (xy_tmp[0] >= d_xmax) xy_tmp[0] = d_xmax - halfqw;
  if (xy_tmp[1] < d_ymin) xy_tmp[1] = d_ymin;
  if (xy_tmp[1] >= d_ymax) xy_tmp[1] = d_ymax - halfqw;

  /* Assuming a brick connectivity with no coordinate transformation */
  if ((xy_tmp[0] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 0]  &&
       xy_tmp[0] < vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
      (xy_tmp[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
       xy_tmp[1] < vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
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
           xy_tmp[0] < vv[3 * ttv[P4EST_CHILDREN * tt + last] + 0]) &&
          (xy_tmp[1] >= vv[3 * ttv[P4EST_CHILDREN * tt + 0   ] + 1]  &&
           xy_tmp[1] < vv[3 * ttv[P4EST_CHILDREN * tt + last] + 1])) {
        *which_tree = tt;
        break;
      }
    }
    P4EST_ASSERT (tt < num_trees);
  }

  xyoffset[0] = xy_tmp[0] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 0];
  P4EST_ASSERT (xyoffset[0] >= 0. && xyoffset[0] < 1.);
  xyoffset[1] = xy_tmp[1] - vv[3 * ttv[P4EST_CHILDREN * tt + 0] + 1];
  P4EST_ASSERT (xyoffset[1] >= 0. && xyoffset[1] < 1.);

  /* construct the smallest quadrant containing xy */
  qq.x = (p4est_qcoord_t) (xyoffset[0] * P4EST_ROOT_LEN);
  qq.x &= ~(qlen - 1); 
  qq.y = (p4est_qcoord_t) (xyoffset[1] * P4EST_ROOT_LEN);
  qq.y &= ~(qlen - 1); 
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
    if (quad != NULL) *quad =
        p4est_quadrant_array_index (&tree->quadrants, position);
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
  const p4est_qcoord_t qlen = P4EST_QUADRANT_LEN (P4EST_QMAXLEVEL);
  const double halfqw = .5 / qlen;

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double         *v2q = p4est->connectivity->vertices;

  p4est_topidx_t p4est_mm = t2v[0];
  p4est_topidx_t p4est_pp = t2v[p4est->connectivity->num_trees *
                                P4EST_CHILDREN - 1];

  double domain_xmin = v2q[3*p4est_mm + 0];
  double domain_ymin = v2q[3*p4est_mm + 1];
  double domain_xmax = v2q[3*p4est_pp + 0];
  double domain_ymax = v2q[3*p4est_pp + 1];

  double xy_p[2];

  // + +
  xy_p[0] = xy[0] + halfqw; xy_p[1] = xy[1] + halfqw;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  rank = my_p4est_brick_point_lookup(p4est, NULL, NULL, xy_p, &tr, &qu, &q);

  // + -
  xy_p[0] = xy[0] + halfqw; xy_p[1] = xy[1] - halfqw;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, NULL, NULL,
                                         xy_p, &tr_tmp, &qu_tmp, &q_tmp);
  if (q_tmp->level > q->level){
    rank = rank_tmp;
    tr = tr_tmp;
    qu = qu_tmp;
    q  = q_tmp;
  }

  // - +
  xy_p[0] = xy[0] - halfqw; xy_p[1] = xy[1] + halfqw;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, NULL, NULL,
                                         xy_p, &tr_tmp, &qu_tmp, &q_tmp);
  if (q_tmp->level > q->level){
    rank = rank_tmp;
    tr = tr_tmp;
    qu = qu_tmp;
    q  = q_tmp;
  }

  // - -
  xy_p[0] = xy[0] - halfqw; xy_p[1] = xy[1] - halfqw;

  if (xy_p[0]<=domain_xmin) xy_p[0] = domain_xmin;
  if (xy_p[0]>=domain_xmax) xy_p[0] = domain_xmax;
  if (xy_p[1]<=domain_ymin) xy_p[1] = domain_ymin;
  if (xy_p[1]>=domain_ymax) xy_p[1] = domain_ymax;

  tr_tmp = tr;
  rank_tmp = my_p4est_brick_point_lookup(p4est, NULL, NULL,
                                         xy_p, &tr_tmp, &qu_tmp, &q_tmp);
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

