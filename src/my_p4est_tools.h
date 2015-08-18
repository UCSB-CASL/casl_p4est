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

#ifndef MY_P4EST_TOOLS_H
#define MY_P4EST_TOOLS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#endif
#ifdef __cplusplus
extern              "C"
{
#if 0
}
#endif
#endif

typedef struct
{
  int                 nxyztrees[3];
  p4est_topidx_t     *nxyz_to_treeid;
}
my_p4est_brick_t;

/** Create a brick connectivity and tree lookup structure.
 * \param [in] nxtrees  Number of trees in x dimension.
 * \param [in] nytrees  Number of trees in y dimension.
 * \param [in,out] myb  Additional brick information will be populated.
 * \return              The brick connectivity structure.
 */
#ifdef P4_TO_P8
p4est_connectivity_t *my_p4est_brick_new (int nxtrees, int nytrees, int nztrees,
                                          double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                                          my_p4est_brick_t * myb);
#else
p4est_connectivity_t *my_p4est_brick_new (int nxtrees, int nytrees,
                                          double xmin, double xmax, double ymin, double ymax,
                                          my_p4est_brick_t * myb);
#endif

/** Free a brick connectivity and tree lookup structure.
 * \param [in] conn     The connectivity will be destroyed.
 * \param [in,out] myb  The dynamically allocated members will be freed.
 */
void                my_p4est_brick_destroy (p4est_connectivity_t * conn,
                                            my_p4est_brick_t * myb);

/** Find the owner processor and quadrant for a point in a brick domain.
 * A point may lie on a quadrant boundary so multiple matches are possible.
 * Identify the local or ghost quadrant with the smallest size as best match.
 * If there are multiple matches choose the smallest owner/tree/quadrant index.
 * Create a list of all possible remote matches outside of the ghost layer.
 * If this list is not empty, the decision about the best match is not final.
 * \param [in] p4est    The forest to be searched.
 * \param [in] ghost    A valid ghost layer.
 * \param [in] myb      Additional brick information.
 * \param [in] xy       The x and y coordinates of a point in the brick.
 *                      May lie on the brick boundary in any direction.
 * \param [in,out] best_match On output will contain quadrant and piggy3 data.
 *                      The local_num is relative to local tree or ghost array.
 * \param [in,out] remote_matches On input, empty quadrant array.  Remote matches
 *                      will be appended as needed including their piggy1 data.
 * \return              The processor number that owns the best match, or -1
 *                      if all possible matches are remote.
 */
int                 my_p4est_brick_point_lookup (p4est_t * p4est,
                                                 p4est_ghost_t * ghost,
                                                 const my_p4est_brick_t * myb,
                                                 double *xy,
                                                 p4est_quadrant_t *best_match,
                                                 sc_array_t * remote_matches);


#ifdef __cplusplus
#if 0
{
#endif
}
#endif

#endif /* !MY_P4EST_TOOLS_H */
