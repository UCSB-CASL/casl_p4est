/*
  This file is part of p4est.
  p4est is a C library to manage a collection (a forest) of multiple
  connected adaptive quadtrees or octrees in parallel.

  Copyright (C) 2010 The University of Texas System
  Written by Carsten Burstedde, Lucas C. Wilcox, and Tobin Isaac

  p4est is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  p4est is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with p4est; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include <sc_search.h>
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_communication.h>
#include "my_p8est_tools.h"
#else
#include <p4est_bits.h>
#include <p4est_communication.h>
#include "my_p4est_tools.h"
#endif
#include <assert.h>

p4est_connectivity_t *
#ifdef P4_TO_P8
my_p4est_brick_new (int nxtrees, int nytrees, int nztrees, my_p4est_brick_t * myb)
#else
my_p4est_brick_new (int nxtrees, int nytrees, my_p4est_brick_t * myb)
#endif
{
  int                 i, j;
  int                 nxyztrees;
  double              dii, djj;
#ifdef P4_TO_P8
  int k;
  double dkk;
#endif
  const double       *vv;
  p4est_topidx_t      tt, vindex;
  p4est_connectivity_t *conn;

  P4EST_ASSERT (0 < nxtrees && 0 < nytrees);
#ifdef P4_TO_P8
  P4EST_ASSERT (0 < nztrees);
#endif
#ifdef P4_TO_P8
  conn = p4est_connectivity_new_brick (nxtrees, nytrees, nztrees, 0, 0, 0);
#else
  conn = p4est_connectivity_new_brick (nxtrees, nytrees, 0, 0);
#endif
  vv = conn->vertices;
  P4EST_ASSERT (vv != NULL);

#ifdef P4_TO_P8
  myb->nxyztrees[0] = nxtrees;
  myb->nxyztrees[1] = nytrees;
  myb->nxyztrees[2] = nztrees;
#else
  myb->nxyztrees[0] = nxtrees;
  myb->nxyztrees[1] = nytrees;
  myb->nxyztrees[2] = 1;
#endif

  nxyztrees = myb->nxyztrees[0] * myb->nxyztrees[1] * myb->nxyztrees[2];
  myb->nxyz_to_treeid = P4EST_ALLOC (p4est_topidx_t, nxyztrees);

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

#ifdef P4_TO_P8
    dkk = vv[3 * vindex + 2];
    P4EST_ASSERT (dkk == floor (dkk));
    k = (int) dkk;
    P4EST_ASSERT (k >= 0 && k < nztrees);
#endif

#ifdef P4_TO_P8
    myb->nxyz_to_treeid[nxtrees*nytrees*k + nxtrees*j + i] = tt;
#else
    myb->nxyz_to_treeid[nxtrees*j + i] = tt;
#endif
  }

  return conn;
}

void
my_p4est_brick_destroy (p4est_connectivity_t * conn, my_p4est_brick_t * myb)
{
  P4EST_ASSERT (myb->nxyz_to_treeid != NULL);
  P4EST_FREE (myb->nxyz_to_treeid);
  myb->nxyz_to_treeid = NULL;
  p4est_connectivity_destroy (conn);
}

#ifndef P4_TO_P8
int
my_p4est_brick_point_lookup (p4est_t * p4est, p4est_ghost_t * ghost,
                             const my_p4est_brick_t * myb,
                             double *xy,
                             p4est_quadrant_t *best_match,
                             sc_array_t * remote_matches)
{
  /* identify all possible quadrant queries */
  int                 i, ix, iy;
  int                 pp, smallest_owner;
  int                 hit;
  /**< bool in each dimension */
  int                 integerstep[P4EST_DIM][2];/**< tree multiple ied. */
  int8_t              highest_level;
  size_t              position, smallest_pos;
  ssize_t             sgpos;
  const p4est_qcoord_t qlen = P4EST_QUADRANT_LEN (P4EST_QMAXLEVEL);
  p4est_qcoord_t      qq, searchq[P4EST_DIM][2];
  p4est_topidx_t      tt, treeid[P4EST_CHILDREN];
  p4est_topidx_t      smallest_tree;
  p4est_quadrant_t    sq, *qp, *found_quad;
  p4est_tree_t       *tree;

  // P4EST_LDEBUGF ("Looking up point %g %g\n", xy[0], xy[1]);

  P4EST_ASSERT (remote_matches->elem_size == sizeof (p4est_quadrant_t) &&
                remote_matches->elem_count == 0);

  /* We are looking for the smallest size quadrant that contains the point xy.
   * If there are multiple possibilities we choose the one on the lowest rank.
   * If there are still multiple possibilities we choose the smallest tree
   * and then the smallest quadrant number in this tree.
   *
   * It is possible that the point lies in remote quadrants outside of the
   * ghost layer, then we return the candidates for further examination. */

  /* gather information to construct search quadrants for all possibilities */

  /* BUG: to account for numerical rounding errors, we check if the coordinates given
   * are withing epsilon (chosen arbitrarily here .... to be improved) of an
   * integer. If so, round the coordinates. In general, using '==' on double is BAD!
   */
  if( fabs(round(xy[0])-xy[0]) < 1e-9 ) xy[0] = round(xy[0]);
  if( fabs(round(xy[1])-xy[1]) < 1e-9 ) xy[1] = round(xy[1]);

  for (i = 0; i < P4EST_DIM; ++i) {
    hit = (int) floor (xy[i]);
    P4EST_ASSERT (0 <= hit && hit <= myb->nxyztrees[i]);

    /* check if the xy[i] coordinate is on a tree boundary */
    if ((double) hit == xy[i]) {
      integerstep[i][0] = hit - 1;      /* can be -1 which will be ignored */
      integerstep[i][1] = hit == myb->nxyztrees[i] ? -1 : hit;   /* ditto */
      searchq[i][0] = P4EST_LAST_OFFSET (P4EST_QMAXLEVEL);
      searchq[i][1] = 0;
      // P4EST_LDEBUGF ("Dimension %d hit tree %d %d\n", i,
      //               integerstep[i][0], integerstep[i][1]);
    }
    else {
      integerstep[i][0] = integerstep[i][1] = hit;
      qq = (p4est_qcoord_t) floor ((xy[i] - hit) * P4EST_ROOT_LEN);
      P4EST_ASSERT (0 <= qq && qq < P4EST_ROOT_LEN);
      searchq[i][1] = qq &= ~(qlen - 1);        /* force quadrant multiple */

      /* check if the xy[i] coordinate is on a possible quadrant boundary */
      if ((double) qq == (xy[i] - hit) * P4EST_ROOT_LEN) {
        searchq[i][0] = qq - qlen;
        P4EST_ASSERT (searchq[i][0] >= 0);
      }
      else {
        integerstep[i][0] = -1;
        searchq[i][0] = -1;
      }
      // P4EST_LDEBUGF ("Dimension %d non tree %d %d\n", i,
      //               integerstep[i][0], integerstep[i][1]);
    }
  }

  /* go through all P4EST_CHILDREN possible quadrant searches */
  P4EST_QUADRANT_INIT (&sq);
  sq.level = P4EST_QMAXLEVEL;
  memset (treeid, -1, sizeof (p4est_topidx_t) * P4EST_CHILDREN);
  highest_level = -1;
  smallest_owner = p4est->mpisize;
  smallest_tree = p4est->connectivity->num_trees;
  smallest_pos = SIZE_MAX;
  found_quad = NULL;
  for (iy = 1; iy >= 0; --iy) {
    if (integerstep[1][iy] == -1) {
      continue;
    }
    for (ix = 1; ix >= 0; --ix) {
      if (integerstep[0][ix] == -1) {
        continue;
      }
      tt = treeid[2 * iy + ix] =
          myb->nxyz_to_treeid[myb->nxyztrees[0] * integerstep[1][iy] +
          integerstep[0][ix]];
      // P4EST_LDEBUGF ("Testing %d %d for tree %d\n", ix, iy, (int) tt);
      sq.x = searchq[0][ix];
      sq.y = searchq[1][iy];
      P4EST_ASSERT (p4est_quadrant_is_valid (&sq));
      pp = p4est_comm_find_owner (p4est, tt, &sq, p4est->mpirank);
      if (pp == p4est->mpirank) {
        /* this quadrant match is processor-local */
        P4EST_ASSERT (p4est->first_local_tree <= tt &&
                      tt <= p4est->last_local_tree);
        tree = p4est_tree_array_index (p4est->trees, tt);
        P4EST_ASSERT (tree->quadrants.elem_count > 0);
        position = sc_bsearch_range (&sq, tree->quadrants.array,
                                     tree->quadrants.elem_count - 1,
                                     sizeof (p4est_quadrant_t),
                                     p4est_quadrant_compare);
        qp = p4est_quadrant_array_index (&tree->quadrants, position);
        P4EST_ASSERT (p4est_quadrant_is_equal (qp, &sq) ||
                      p4est_quadrant_is_ancestor (qp, &sq));
        // P4EST_LDEBUGF ("Found local level %d\n", qp->level);
        if (qp->level > highest_level ||
            (qp->level == highest_level &&
             (pp < smallest_owner ||
              (pp == smallest_owner &&
               (tt < smallest_tree ||
                (tt == smallest_tree && position < smallest_pos)))))) {
          highest_level = qp->level;
          smallest_owner = pp;
          smallest_tree = tt;
          smallest_pos = position;
          found_quad = qp;
          // P4EST_LDEBUGF ("Current best guess at %lld %lld\n",
          //               (long long) smallest_tree, (long long) smallest_pos);
        }
        /* else no need to update our match */
      }
      else {
        sgpos = p4est_ghost_contains (ghost, pp, tt, &sq);
        if (sgpos >= 0) {
          /* this quadrant match is in the ghost layer */
          position = (size_t) sgpos;
          qp = p4est_quadrant_array_index (&ghost->ghosts, position);
          // P4EST_LDEBUGF ("Found ghost %d level %d\n", pp, qp->level);
          if (qp->level > highest_level ||
              (qp->level == highest_level &&
               (pp < smallest_owner ||
                (pp == smallest_owner &&
                 (tt < smallest_tree ||
                  (tt == smallest_tree && position < smallest_pos)))))) {
            highest_level = qp->level;
            smallest_owner = pp;
            smallest_tree = tt;
            smallest_pos = position;    /* we're indexing ghosts so this is
                                           partially redundant with tt */
            found_quad = qp;
            // P4EST_LDEBUGF ("Current best guess at %lld %lld\n",
            //               (long long) smallest_tree,
            //               (long long) smallest_pos);
          }
          /* else no need to update our match */
        }
        else {
          /* this quadrant is somewhere else on the remote processor */
          qp = (p4est_quadrant_t *) sc_array_push (remote_matches);
          *qp = sq;
          qp->p.piggy1.which_tree = tt;
          qp->p.piggy1.owner_rank = pp;
        }
      }
    }
  }

  if (smallest_owner < p4est->mpisize) {
    P4EST_ASSERT (found_quad != NULL);
    P4EST_ASSERT (found_quad->level == highest_level);
    *best_match = *found_quad;
    best_match->p.piggy3.which_tree = smallest_tree;
    best_match->p.piggy3.local_num = (p4est_locidx_t) smallest_pos;
    return smallest_owner;
  }

  return -1;
}
#endif
