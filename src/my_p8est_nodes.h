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

#ifndef MY_P8EST_NODES_H
#define MY_P8EST_NODES_H

#include <p8est_nodes.h>

SC_EXTERN_C_BEGIN;

/** This structure holds complete parallel node information.
 *
 * Nodes are unique and considered independent.
 * Independent nodes store their owner's tree id in piggy3.which_tree.
 * The index in their owner's ordering is stored in piggy3.local_num.
 *
 * The local_nodes table is of dimension 8 * num_local_quadrants
 * and encodes the node indexes for all corners of all quadrants.
 *
 * The array shared_indeps holds lists of node sharers (not including rank).
 * The entry shared_indeps[i] is of type sc_recycle_array_t
 * and holds the list of nodes with i + 1 sharers.
 * For each independent node, its member pad8 holds the number of sharers
 * and its member pad16 holds the position in the assigned recycle array
 * if this number fits into an int16_t.  If this limit is exceeded, the
 * array shared_offsets is filled with these positions as one p4est_locidx_t
 * per independent node, and all pad16 members are set to -1.  To recognize
 * the latter situation you can check for shared_offsets != NULL.
 *
 * Each processor owns num_owned_indeps of the stored independent nodes.
 * The first independent owned node is at index offset_owned_indeps.
 * The table nonlocal_ranks contains the ranks of all stored non-owned nodes.
 * The table global_owned_indeps holds the number of owned nodes for each rank.
 */
typedef struct my_p8est_nodes
{
  p4est_locidx_t      num_local_quadrants;
  p4est_locidx_t      num_owned_indeps, num_owned_shared;
  p4est_locidx_t      offset_owned_indeps;
  sc_array_t          indep_nodes;
  p4est_locidx_t     *local_nodes;
  sc_array_t          shared_indeps;
  p4est_locidx_t     *shared_offsets;
  int                *nonlocal_ranks;
  p4est_locidx_t     *global_owned_indeps;
}
my_p8est_nodes_t;

/** Create node information.
 * \param [in] p4est    The forest.  Does not need to be balanced.
 * \param [in] ghost    Ghost layer.  If this is NULL, then only
 *                      processor-local nodes will be matched and
 *                      nodes->global_owned_indeps will be NULL.
 * \return              A fully populated my_p8est_nodes structure.
 */
my_p8est_nodes_t      *my_p8est_nodes_new (p8est_t * p8est,
                                           p8est_ghost_t * ghost);

/** Destroy node information. */
void                my_p8est_nodes_destroy (my_p8est_nodes_t * nodes);

/** Check node information for internal consistency. */
int                 my_p8est_nodes_is_valid (p8est_t * p8est,
                                             my_p8est_nodes_t * nodes);

SC_EXTERN_C_END;

#endif /* !MY_P8EST_NODES_H */
