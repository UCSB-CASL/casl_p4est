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

#ifndef P4_TO_P8
#include <p4est_algorithms.h>
#include <p4est_bits.h>
#include <p4est_communication.h>
#include "my_p4est_nodes.h"
#else
#include <p8est_algorithms.h>
#include <p8est_bits.h>
#include <p8est_communication.h>
#include "my_p8est_nodes.h"
#endif
#include <sc_notify.h>

#ifdef P4EST_MPI

typedef struct
{
  int                 expect_query, expect_reply;
  size_t              recv_offset;
  sc_array_t          send_first, send_second;
  sc_array_t          recv_first, recv_second;
}
p4est_node_peer_t;

#endif

/** Determine the owning tree for a node and clamp it inside the domain.
 *
 * If the node is on the boundary, assign the lowest tree to own it.
 * Clamp it inside the tree bounds if necessary.
 *
 * \param [in] p4est    The p4est to work on.
 * \param [in] treeid   Original tree index for this node.
 * \param [in] n        The node to work on.
 * \param [out] c       The clamped node in owning tree coordinates.
 *                      Its piggy data will be filled with owning tree id.
 */
static void
p4est_node_canonicalize (p4est_t * p4est, p4est_topidx_t treeid,
                         const p4est_quadrant_t * n, p4est_quadrant_t * c)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  int                 face_axis[3];     /* 3 not P4EST_DIM */
  int                 quad_contact[P4EST_FACES];
  int                 contacts, face, corner;
  int                 ftransform[P4EST_FTRANSFORM];
  size_t              ctreez;
  p4est_topidx_t      ntreeid, ntreeid2, lowest;
  p4est_quadrant_t    tmpq, o;
#ifdef P4_TO_P8
  int                 edge;
  size_t              etreez;
  p8est_edge_info_t   ei;
  p8est_edge_transform_t *et;
  sc_array_t         *eta;
#endif
  p4est_corner_info_t ci;
  p4est_corner_transform_t *ct;
  sc_array_t         *cta;

  P4EST_ASSERT (treeid >= 0 && treeid < conn->num_trees);
  P4EST_ASSERT (p4est_quadrant_is_node (n, 0));

  P4EST_QUADRANT_INIT (&tmpq);
  P4EST_QUADRANT_INIT (&o);

  lowest = treeid;
  p4est_node_clamp_inside (n, c);
  c->p.which_tree = -1;

  /* Check if the quadrant is inside the tree */
  quad_contact[0] = (n->x == 0);
  quad_contact[1] = (n->x == P4EST_ROOT_LEN);
  face_axis[0] = quad_contact[0] || quad_contact[1];
  quad_contact[2] = (n->y == 0);
  quad_contact[3] = (n->y == P4EST_ROOT_LEN);
  face_axis[1] = quad_contact[2] || quad_contact[3];
#ifndef P4_TO_P8
  face_axis[2] = 0;
#else
  quad_contact[4] = (n->z == 0);
  quad_contact[5] = (n->z == P4EST_ROOT_LEN);
  face_axis[2] = quad_contact[4] || quad_contact[5];
#endif
  contacts = face_axis[0] + face_axis[1] + face_axis[2];
  if (contacts == 0) {
    goto endfunction;
  }

  /* Check face neighbors */
#ifdef P4EST_DEBUG
  ntreeid = -1;
#endif
  for (face = 0; face < P4EST_FACES; ++face) {
    if (!quad_contact[face]) {
      /* The node is not touching this face */
      continue;
    }
    ntreeid = conn->tree_to_tree[P4EST_FACES * treeid + face];
    if (ntreeid == treeid
        && ((int) conn->tree_to_face[P4EST_FACES * treeid + face] == face)) {
      /* The node touches a face with no neighbor */
      continue;
    }
    if (ntreeid > lowest) {
      /* This neighbor tree is higher, so we keep the ownership */
      continue;
    }
    /* Transform the node into the other tree's coordinates */
    ntreeid2 = p4est_find_face_transform (conn, treeid, face, ftransform);
    P4EST_ASSERT (ntreeid2 == ntreeid);
    p4est_quadrant_transform_face (n, &o, ftransform);
    if (ntreeid < lowest) {
      /* we have found a new owning tree */
      p4est_node_clamp_inside (&o, c);
      lowest = ntreeid;
    }
    else {
      P4EST_ASSERT (lowest == ntreeid);
      p4est_node_clamp_inside (&o, &tmpq);
      if (p4est_quadrant_compare (&tmpq, c) < 0) {
        /* same tree (periodic) and the new position is lower than the old */
        *c = tmpq;
      }
    }
  }
  P4EST_ASSERT (ntreeid >= 0);
  if (contacts == 1) {
    goto endfunction;
  }

#ifdef P4_TO_P8
  P4EST_ASSERT (contacts >= 2);
  eta = &ei.edge_transforms;
  sc_array_init (eta, sizeof (p8est_edge_transform_t));
  for (edge = 0; edge < P8EST_EDGES; ++edge) {
    if (!(quad_contact[p8est_edge_faces[edge][0]] &&
          quad_contact[p8est_edge_faces[edge][1]])) {
      continue;
    }
    p8est_find_edge_transform (conn, treeid, edge, &ei);
    for (etreez = 0; etreez < eta->elem_count; ++etreez) {
      et = p8est_edge_array_index (eta, etreez);
      ntreeid = et->ntree;
      if (ntreeid > lowest) {
        /* This neighbor tree is higher, so we keep the ownership */
        continue;
      }
      p8est_quadrant_transform_edge (n, &o, &ei, et, 0);
      if (ntreeid < lowest) {
        p4est_node_clamp_inside (&o, c);
        lowest = ntreeid;
      }
      else {
        P4EST_ASSERT (lowest == ntreeid);
        p4est_node_clamp_inside (&o, &tmpq);
        if (p4est_quadrant_compare (&tmpq, c) < 0) {
          /* same tree (periodic) and the new position is lower than the old */
          *c = tmpq;
        }
      }
    }
  }
  sc_array_reset (eta);
  eta = NULL;
  et = NULL;
  if (contacts == 2) {
    goto endfunction;
  }
#endif

  P4EST_ASSERT (contacts == P4EST_DIM);
  cta = &ci.corner_transforms;
  sc_array_init (cta, sizeof (p4est_corner_transform_t));
  for (corner = 0; corner < P4EST_CHILDREN; ++corner) {
    if (!(quad_contact[p4est_corner_faces[corner][0]] &&
          quad_contact[p4est_corner_faces[corner][1]] &&
#ifdef P4_TO_P8
          quad_contact[p4est_corner_faces[corner][2]] &&
#endif
          1)) {
      continue;
    }
    p4est_find_corner_transform (conn, treeid, corner, &ci);
    for (ctreez = 0; ctreez < cta->elem_count; ++ctreez) {
      ct = p4est_corner_array_index (cta, ctreez);
      ntreeid = ct->ntree;
      if (ntreeid > lowest) {
        /* This neighbor tree is higher, so we keep the ownership */
        continue;
      }
      o.level = P4EST_MAXLEVEL;
      p4est_quadrant_transform_corner (&o, (int) ct->ncorner, 0);
      if (ntreeid < lowest) {
        p4est_node_clamp_inside (&o, c);
        lowest = ntreeid;
      }
      else {
        P4EST_ASSERT (lowest == ntreeid);
        p4est_node_clamp_inside (&o, &tmpq);
        if (p4est_quadrant_compare (&tmpq, c) < 0) {
          /* same tree (periodic) and the new position is lower than the old */
          *c = tmpq;
        }
      }
    }
  }
  sc_array_reset (cta);

endfunction:
  c->p.which_tree = lowest;

  P4EST_ASSERT (p4est_quadrant_is_node (c, 1));
  P4EST_ASSERT (c->p.which_tree >= 0 && c->p.which_tree < conn->num_trees);
}

static int
p4est_nodes_foreach (void **item, const void *u)
{
  const sc_hash_array_data_t *internal_data =
    (const sc_hash_array_data_t *) u;
  const p4est_locidx_t *new_node_number =
    (const p4est_locidx_t *) internal_data->user_data;

  *item = (void *) (long) new_node_number[(long) *item];

  return 1;
}

#ifdef P4EST_MPI

static p4est_locidx_t *
p4est_shared_offsets (sc_array_t * inda)
{
  p4est_locidx_t      il, num_indep_nodes;
  p4est_locidx_t     *shared_offsets;
  p4est_indep_t      *in;

  num_indep_nodes = (p4est_locidx_t) inda->elem_count;
  shared_offsets = P4EST_ALLOC (p4est_locidx_t, num_indep_nodes);

  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (inda, il);
    shared_offsets[il] = (p4est_locidx_t) in->pad16;
    in->pad16 = -1;
  }

  return shared_offsets;
}

#endif

my_p4est_nodes_t      *
my_p4est_nodes_new (p4est_t * p4est, p4est_ghost_t * ghost)
{
  const int           num_procs = p4est->mpisize;
  const int           rank = p4est->mpirank;
#ifdef P4EST_MPI
  int                 mpiret;
  int                 owner, prev;
  int                 num_send_queries, num_send_nonzero, num_recv_queries;
  int                 byte_count, elem_count;
  int                 local_send_count, local_recv_count;
  int                *procs;
  int                *old_sharers, *new_sharers;
  char               *this_base;
  int                 found;
  size_t              first_size, second_size, this_size;
  size_t              num_sharers, old_position, new_position;
  p4est_qcoord_t     *xyz;
  p4est_topidx_t     *ttt;
  p4est_locidx_t      end_owned_indeps;
  p4est_locidx_t     *node_number;
  p4est_node_peer_t  *peers, *peer;
  p4est_indep_t       inkey;
  sc_array_t          send_requests;
  sc_recycle_array_t *orarr, *nrarr;
  MPI_Request        *send_request;
  MPI_Status          probe_status, recv_status;
#endif
#if defined (P4EST_MPI) || defined (P4_TO_P8)
  int                 l;
#endif
  int                 k;
  int                *nonlocal_ranks;
  int                 clamped = 1;
  int                *sender_ranks, num_senders, num_receivers;
  void               *save_user_data;
  size_t              zz, position;
  p4est_topidx_t      jt;
  p4est_locidx_t      il;
  p4est_locidx_t      num_local_nodes;
  p4est_locidx_t      num_owned_shared, num_owned_indeps;
  p4est_locidx_t      offset_owned_indeps;
  p4est_locidx_t      num_indep_nodes, dup_indep_nodes;
  p4est_locidx_t     *local_nodes, *quad_nodes, *shared_offsets;
  p4est_locidx_t     *new_node_number;
  p4est_tree_t       *tree;
  my_p4est_nodes_t      *nodes;
  p4est_quadrant_t    c, n, p;
  p4est_quadrant_t   *q, *r;
  p4est_indep_t      *in;
  sc_array_t         receiver_ranks;
  sc_array_t         *quadrants;
  sc_array_t         *inda;
  sc_array_t         *shared_indeps;
  sc_hash_array_t    *indep_nodes;

  P4EST_GLOBAL_PRODUCTION ("Into my_" P4EST_STRING "_nodes_new\n");
  P4EST_ASSERT (p4est_is_valid (p4est));

  P4EST_QUADRANT_INIT (&c);
  P4EST_QUADRANT_INIT (&n);
  P4EST_QUADRANT_INIT (&p);

  /* allocate and initialize the node structure to return */
  nodes = P4EST_ALLOC (my_p4est_nodes_t, 1);
  memset (nodes, -1, sizeof (*nodes));
  shared_indeps = &nodes->shared_indeps;
  sc_array_init (shared_indeps, sizeof (sc_recycle_array_t));
  shared_offsets = nodes->shared_offsets = NULL;

  /* Compute number of local quadrant corners. */
  nodes->num_local_quadrants = p4est->local_num_quadrants;
  num_local_nodes =             /* same type */
    P4EST_CHILDREN * nodes->num_local_quadrants;

  /* Store the local node index for each corner of the elements. */
  nodes->local_nodes = local_nodes =
    P4EST_ALLOC (p4est_locidx_t, num_local_nodes);
  memset (local_nodes, -1, num_local_nodes * sizeof (*local_nodes));

  indep_nodes = sc_hash_array_new (sizeof (p4est_indep_t),
                                   p4est_node_hash_piggy_fn,
                                   p4est_node_equal_piggy_fn, &clamped);

  /* This loop will collect independent nodes relevant for the elements. */
  num_indep_nodes = dup_indep_nodes = 0;
  quad_nodes = local_nodes;
  for (jt = p4est->first_local_tree; jt <= p4est->last_local_tree; ++jt) {
    tree = p4est_tree_array_index (p4est->trees, jt);
    quadrants = &tree->quadrants;

    for (zz = 0; zz < quadrants->elem_count;
         quad_nodes += P4EST_CHILDREN, ++zz) {
      q = p4est_quadrant_array_index (quadrants, zz);

      /* collect all independent nodes related to the element */
      for (k = 0; k < P4EST_CHILDREN; ++k) {
        p4est_quadrant_corner_node (q, k, &n);
        p4est_node_canonicalize (p4est, jt, &n, &c);
        r =
          (p4est_quadrant_t *) sc_hash_array_insert_unique (indep_nodes, &c,
                                                            &position);
        if (r != NULL) {
          *r = c;
          P4EST_ASSERT (num_indep_nodes == (p4est_locidx_t) position);
          ++num_indep_nodes;
        }
        else {
          ++dup_indep_nodes;
        }
        P4EST_ASSERT ((p4est_locidx_t) position < num_indep_nodes);
        quad_nodes[k] = (p4est_locidx_t) position;
      }
    }
  }
  P4EST_ASSERT (num_indep_nodes + dup_indep_nodes == num_local_nodes);
  inda = &indep_nodes->a;
  P4EST_ASSERT (num_indep_nodes == (p4est_locidx_t) inda->elem_count);

  /* Reorder independent nodes by their global treeid and z-order index. */
  new_node_number = P4EST_ALLOC (p4est_locidx_t, num_indep_nodes);
  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (inda, (size_t) il);
    in->pad8 = 0;               /* shared by 0 other processors so far */
    in->pad16 = (int16_t) (-1);
    in->p.piggy3.local_num = il;
  }
  sc_array_sort (inda, p4est_quadrant_compare_piggy);
  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (inda, (size_t) il);
    new_node_number[in->p.piggy3.local_num] = il;
#ifndef P4EST_MPI
    in->p.piggy3.local_num = il;
#endif
  }

  /* Re-synchronize hash array and local nodes */
  save_user_data = indep_nodes->internal_data.user_data;
  indep_nodes->internal_data.user_data = new_node_number;
  sc_hash_foreach (indep_nodes->h, p4est_nodes_foreach);
  indep_nodes->internal_data.user_data = save_user_data;
  for (il = 0; il < num_local_nodes; ++il) {
    P4EST_ASSERT (local_nodes[il] >= 0 && local_nodes[il] < num_indep_nodes);
    local_nodes[il] = new_node_number[local_nodes[il]];
  }
#ifndef P4EST_MPI
  num_owned_indeps = num_indep_nodes;
  offset_owned_indeps = 0;
#else
  num_owned_indeps = 0;         /* will be computed below */
  offset_owned_indeps = -1;     /* will be computed below */
#endif
  num_owned_shared = 0;
  P4EST_FREE (new_node_number);

#ifdef P4EST_MPI
  /* Fill send buffers and number owned nodes. */
  first_size = P4EST_DIM * sizeof (p4est_qcoord_t) + sizeof (p4est_topidx_t);
  first_size = SC_MAX (first_size, sizeof (p4est_locidx_t));
  procs = P4EST_ALLOC_ZERO (int, num_procs);
  peers = P4EST_ALLOC (p4est_node_peer_t, num_procs);
  sc_array_init (&send_requests, sizeof (MPI_Request));
  for (k = 0; k < num_procs; ++k) {
    peer = peers + k;
    peer->expect_query = peer->expect_reply = 0;
    peer->recv_offset = 0;
    sc_array_init (&peer->send_first, first_size);
    sc_array_init (&peer->recv_first, first_size);
    sc_array_init (&peer->send_second, 1);
    sc_array_init (&peer->recv_second, 1);
  }
  prev = 0;
  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (inda, (size_t) il);
    owner = p4est_comm_find_owner (p4est, in->p.which_tree,
                                   (p4est_quadrant_t *) in, prev);
    if (owner != rank) {
      peer = peers + owner;
      xyz = (p4est_qcoord_t *) sc_array_push (&peer->send_first);
      xyz[0] = in->x;
      xyz[1] = in->y;
#ifdef P4_TO_P8
      xyz[2] = in->z;
#endif
      ttt = (p4est_topidx_t *) (&xyz[P4EST_DIM]);
      *ttt = in->p.which_tree;
      in->p.piggy1.owner_rank = owner;
      ++procs[owner];
    }
    else {
      if (offset_owned_indeps == -1) {
        offset_owned_indeps = il;
      }
      in->p.piggy3.local_num = num_owned_indeps++;
    }
    P4EST_ASSERT (prev <= owner);
    prev = owner;
  }
  if (offset_owned_indeps == -1) {
    P4EST_ASSERT (num_owned_indeps == 0);
    offset_owned_indeps = 0;
  }
  end_owned_indeps = offset_owned_indeps + num_owned_indeps;

  /* Distribute global information about who is sending to who. */
  sc_array_init (&receiver_ranks, sizeof (int));
  for (owner = 0; owner < num_procs; ++owner) {
    if (procs[owner] > 0) {
      P4EST_ASSERT (owner != rank);
      *(int *) sc_array_push (&receiver_ranks) = owner;
    }
  }
  num_receivers = (int) receiver_ranks.elem_count;
  sender_ranks = P4EST_ALLOC (int, num_procs);
  sc_notify ((int *) receiver_ranks.array, num_receivers,
             sender_ranks, &num_senders, p4est->mpicomm);
  P4EST_GLOBAL_LDEBUGF ("Num receivers %d senders %d\n",
                        num_receivers, num_senders);

  /* Send queries to the owners of the independent nodes that I share. */
  num_send_queries = num_send_nonzero = local_send_count = 0;
  for (l = 0; l < num_receivers; ++l) {
    k = *(int *) sc_array_index_int (&receiver_ranks, l);
    P4EST_ASSERT (k >= 0 && k < num_procs && k != rank);
    peer = peers + k;
    send_request = (MPI_Request *) sc_array_push (&send_requests);
    this_size = peer->send_first.elem_count * first_size;
    mpiret = MPI_Isend (peer->send_first.array, (int) this_size,
                        MPI_BYTE, k, P4EST_COMM_NODES_QUERY,
                        p4est->mpicomm, send_request);
    SC_CHECK_MPI (mpiret);
    local_send_count += (int) peer->send_first.elem_count;
    ++num_send_queries;
    P4EST_ASSERT (this_size > 0);
    ++num_send_nonzero;
    peer->expect_reply = 1;
  }
  sc_array_reset (&receiver_ranks);

  /* Prepare to receive queries */
  num_recv_queries = local_recv_count = 0;
  for (l = 0; l < num_senders; ++l) {
    k = sender_ranks[l];
    P4EST_ASSERT (k >= 0 && k < num_procs && k != rank);
    peers[k].expect_query = 1;
    ++num_recv_queries;
  }
  P4EST_FREE (sender_ranks);
  P4EST_VERBOSEF ("Node queries send %d nonz %d recv %d\n",
                  num_send_queries, num_send_nonzero, num_recv_queries);

  /* Receive queries and look up the reply information */
  P4EST_QUADRANT_INIT (&inkey);
  inkey.level = P4EST_MAXLEVEL;
  for (l = 0; l < num_recv_queries; ++l) {
    mpiret = MPI_Probe (MPI_ANY_SOURCE, P4EST_COMM_NODES_QUERY,
                        p4est->mpicomm, &probe_status);
    SC_CHECK_MPI (mpiret);
    k = probe_status.MPI_SOURCE;
    peer = peers + k;
    P4EST_ASSERT (k != rank && peer->expect_query);
    mpiret = MPI_Get_count (&probe_status, MPI_BYTE, &byte_count);
    SC_CHECK_MPI (mpiret);
    P4EST_ASSERT (byte_count % first_size == 0);
    elem_count = byte_count / (int) first_size;
    local_recv_count += elem_count;
    sc_array_resize (&peer->recv_first, (size_t) elem_count);
    mpiret = MPI_Recv (peer->recv_first.array, byte_count, MPI_BYTE,
                       k, P4EST_COMM_NODES_QUERY,
                       p4est->mpicomm, &recv_status);
    SC_CHECK_MPI (mpiret);
    peer->expect_query = 0;
    for (zz = 0; zz < peer->recv_first.elem_count; ++zz) {
      xyz = (p4est_qcoord_t *) sc_array_index (&peer->recv_first, zz);
      inkey.x = xyz[0];
      inkey.y = xyz[1];
#ifdef P4_TO_P8
      inkey.z = xyz[2];
#endif
      ttt = (p4est_topidx_t *) (&xyz[P4EST_DIM]);
      inkey.p.which_tree = *ttt;
      found = sc_hash_array_lookup (indep_nodes, &inkey, &position);
      P4EST_ASSERT (found);
      P4EST_ASSERT ((p4est_locidx_t) position >= offset_owned_indeps &&
                    (p4est_locidx_t) position < end_owned_indeps);
      node_number = (p4est_locidx_t *) xyz;
      *node_number = (p4est_locidx_t) position - offset_owned_indeps;
      in = (p4est_indep_t *) sc_array_index (inda, position);
      P4EST_ASSERT (p4est_node_equal_piggy_fn (&inkey, in, &clamped));
      P4EST_ASSERT (in->pad8 >= 0);
      num_sharers = (size_t) in->pad8;
      P4EST_ASSERT (num_sharers <= shared_indeps->elem_count);
      SC_CHECK_ABORT (num_sharers < (size_t) INT8_MAX,
                      "Max independent node sharer limit exceeded");
      if (num_sharers == shared_indeps->elem_count) {
        nrarr = (sc_recycle_array_t *) sc_array_push (shared_indeps);
        sc_recycle_array_init (nrarr, (num_sharers + 1) * sizeof (int));
      }
      else {
        nrarr =
          (sc_recycle_array_t *) sc_array_index (shared_indeps, num_sharers);
      }
      new_sharers = (int *) sc_recycle_array_insert (nrarr, &new_position);
      if (num_sharers > 0) {
        if (shared_offsets == NULL) {
          P4EST_ASSERT (in->pad16 >= 0);
          old_position = (size_t) in->pad16;
        }
        else {
          P4EST_ASSERT (in->pad16 == -1);
          old_position = (size_t) shared_offsets[position];
        }
        orarr =
          (sc_recycle_array_t *) sc_array_index (shared_indeps,
                                                 num_sharers - 1);
        old_sharers = (int *) sc_recycle_array_remove (orarr, old_position);
        memcpy (new_sharers, old_sharers, num_sharers * sizeof (int));
      }
      else {
        ++num_owned_shared;
      }
      new_sharers[num_sharers] = k;
      ++in->pad8;
      if (shared_offsets == NULL) {
        if (new_position > (size_t) INT16_MAX) {
          shared_offsets = p4est_shared_offsets (inda);
          shared_offsets[position] = (p4est_locidx_t) new_position;
        }
        else {
          in->pad16 = (int16_t) new_position;
        }
      }
      else {
        shared_offsets[position] = (p4est_locidx_t) new_position;
      }
    }
  }

  /* Assemble and send reply information.  This is variable size.
   * (p4est_locidx_t)      Node number in this processor's ordering
   * (int8_t)              Number of sharers (not including this processor)
   * num_sharers * (int)   The ranks of all sharers.
   */
  second_size = sizeof (p4est_locidx_t) + sizeof (int8_t);
  for (k = 0; k < num_procs; ++k) {
    peer = peers + k;
    if (peer->recv_first.elem_count == 0) {
      continue;
    }
    for (zz = 0; zz < peer->recv_first.elem_count; ++zz) {
      node_number = (p4est_locidx_t *) sc_array_index (&peer->recv_first, zz);
      position = (size_t) (*node_number + offset_owned_indeps);
      in = (p4est_indep_t *) sc_array_index (inda, position);
      P4EST_ASSERT (p4est_quadrant_is_node ((p4est_quadrant_t *) in, 1));
      P4EST_ASSERT (in->pad8 >= 0);
      num_sharers = (size_t) in->pad8;
      P4EST_ASSERT (num_sharers <= shared_indeps->elem_count);
      this_size = second_size + num_sharers * sizeof (int);
      this_base =
        (char *) sc_array_push_count (&peer->send_second, this_size);
      *(p4est_locidx_t *) this_base = *node_number;
      *(int8_t *) (this_base + sizeof (p4est_locidx_t)) = in->pad8;
      if (num_sharers > 0) {
        if (shared_offsets == NULL) {
          P4EST_ASSERT (in->pad16 >= 0);
          new_position = (size_t) in->pad16;
        }
        else {
          P4EST_ASSERT (in->pad16 == -1);
          new_position = (size_t) shared_offsets[position];
        }
        nrarr =
          (sc_recycle_array_t *) sc_array_index (shared_indeps,
                                                 num_sharers - 1);
        new_sharers = (int *) sc_array_index (&nrarr->a, new_position);
        memcpy (this_base + second_size, new_sharers,
                num_sharers * sizeof (int));
      }
    }
    send_request = (MPI_Request *) sc_array_push (&send_requests);
    mpiret = MPI_Isend (peer->send_second.array,
                        (int) peer->send_second.elem_count,
                        MPI_BYTE, k, P4EST_COMM_NODES_REPLY,
                        p4est->mpicomm, send_request);
    SC_CHECK_MPI (mpiret);
    sc_array_reset (&peer->recv_first);
  }
#endif /* P4EST_MPI */

  /* Allocate remaining output data structures */
  nodes->num_owned_indeps = num_owned_indeps;
  nodes->num_owned_shared = num_owned_shared;
  nodes->offset_owned_indeps = offset_owned_indeps;
  sc_hash_array_rip (indep_nodes, inda = &nodes->indep_nodes);
  nonlocal_ranks = nodes->nonlocal_ranks =
    P4EST_ALLOC (int, num_indep_nodes - num_owned_indeps);
  nodes->global_owned_indeps = P4EST_ALLOC (p4est_locidx_t, num_procs);
  nodes->global_owned_indeps[rank] = num_owned_indeps;
  indep_nodes = NULL;

#ifdef P4EST_MPI
  /* Receive the replies. */
  for (l = 0; l < num_send_nonzero; ++l) {
    mpiret = MPI_Probe (MPI_ANY_SOURCE, P4EST_COMM_NODES_REPLY,
                        p4est->mpicomm, &probe_status);
    SC_CHECK_MPI (mpiret);
    k = probe_status.MPI_SOURCE;
    peer = peers + k;
    P4EST_ASSERT (k != rank && peer->expect_reply);
    mpiret = MPI_Get_count (&probe_status, MPI_BYTE, &byte_count);
    SC_CHECK_MPI (mpiret);
    sc_array_resize (&peer->recv_second, (size_t) byte_count);
    mpiret = MPI_Recv (peer->recv_second.array, byte_count, MPI_BYTE,
                       k, P4EST_COMM_NODES_REPLY,
                       p4est->mpicomm, &recv_status);
    SC_CHECK_MPI (mpiret);
    peer->expect_reply = 0;
  }
#endif /* P4EST_MPI */

  /* Convert receive buffers into the output data structures and unclamp. */
  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (inda, (size_t) il);
    p4est_node_unclamp ((p4est_quadrant_t *) in);
#ifdef P4EST_MPI
    if (il >= offset_owned_indeps && il < end_owned_indeps) {
      continue;
    }
    k = in->p.piggy1.owner_rank;
    P4EST_ASSERT ((k < rank && il < offset_owned_indeps) ||
                  (k > rank && il >= end_owned_indeps));
    *nonlocal_ranks++ = k;
    peer = peers + k;
    P4EST_ASSERT (peer->recv_offset + second_size <=
                  peer->recv_second.elem_count);
    this_base =
      (char *) sc_array_index (&peer->recv_second, peer->recv_offset);
    in->p.piggy3.local_num = *(p4est_locidx_t *) this_base;
    num_sharers =
      (size_t) (*(int8_t *) (this_base + sizeof (p4est_locidx_t)));
    P4EST_ASSERT (num_sharers > 0);
    this_size = second_size + num_sharers * sizeof (int);
    P4EST_ASSERT (peer->recv_offset + this_size <=
                  peer->recv_second.elem_count);
    if (shared_indeps->elem_count < num_sharers) {
      nrarr = NULL;
      old_position = shared_indeps->elem_count;
      sc_array_resize (shared_indeps, num_sharers);
      for (zz = old_position; zz < num_sharers; ++zz) {
        nrarr = (sc_recycle_array_t *) sc_array_index (shared_indeps, zz);
        sc_recycle_array_init (nrarr, (zz + 1) * sizeof (int));
      }
    }
    else {
      nrarr =
        (sc_recycle_array_t *) sc_array_index (shared_indeps,
                                               num_sharers - 1);
    }
    new_sharers = (int *) sc_recycle_array_insert (nrarr, &new_position);
    memcpy (new_sharers, this_base + second_size, num_sharers * sizeof (int));
    for (zz = 0; zz < num_sharers; ++zz) {
      if (new_sharers[zz] == rank) {
        new_sharers[zz] = k;
        break;
      }
    }
    P4EST_ASSERT (zz < num_sharers);
    in->pad8 = (int8_t) num_sharers;
    if (shared_offsets == NULL) {
      if (new_position > (size_t) INT16_MAX) {
        shared_offsets = p4est_shared_offsets (inda);
        shared_offsets[il] = (p4est_locidx_t) new_position;
      }
      else {
        in->pad16 = (int16_t) new_position;
      }
    }
    else {
      shared_offsets[il] = (p4est_locidx_t) new_position;
    }
    peer->recv_offset += this_size;
#endif /* P4EST_MPI */
  }

#ifdef P4EST_MPI
  /* Wait and close all send requests. */
  if (send_requests.elem_count > 0) {
    mpiret = MPI_Waitall ((int) send_requests.elem_count,
                          (MPI_Request *) send_requests.array,
                          MPI_STATUSES_IGNORE);
    SC_CHECK_MPI (mpiret);
  }
  nodes->shared_offsets = shared_offsets;

  /* Clean up allocated communications memory. */
  sc_array_reset (&send_requests);
  for (k = 0; k < num_procs; ++k) {
    peer = peers + k;
    P4EST_ASSERT (peer->recv_offset == peer->recv_second.elem_count);
    sc_array_reset (&peer->send_first);
    /* peer->recv_first has been reset above */
    sc_array_reset (&peer->send_second);
    sc_array_reset (&peer->recv_second);
  }
  P4EST_FREE (peers);
  P4EST_FREE (procs);

  mpiret = MPI_Allgather (&num_owned_indeps, 1, P4EST_MPI_LOCIDX,
                          nodes->global_owned_indeps, 1, P4EST_MPI_LOCIDX,
                          p4est->mpicomm);
  SC_CHECK_MPI (mpiret);
#endif /* P4EST_MPI */

  /* Print some statistics and clean up. */
  P4EST_VERBOSEF ("Collected %lld independent nodes with %lld duplicates\n",
                  (long long) num_indep_nodes, (long long) dup_indep_nodes);
#ifdef P4EST_MPI
  P4EST_VERBOSEF ("Owned nodes %lld/%lld/%lld max sharer count %llu\n",
                  (long long) num_owned_shared,
                  (long long) num_owned_indeps,
                  (long long) num_indep_nodes,
                  (unsigned long long) shared_indeps->elem_count);
#endif

  P4EST_ASSERT (my_p4est_nodes_is_valid (p4est, nodes));
  P4EST_GLOBAL_PRODUCTION ("Done " P4EST_STRING "_nodes_new\n");

  return nodes;
}

void
my_p4est_nodes_destroy (my_p4est_nodes_t * nodes)
{
  size_t              zz;
  sc_recycle_array_t *rarr;

  sc_array_reset (&nodes->indep_nodes);
  P4EST_FREE (nodes->local_nodes);

  for (zz = 0; zz < nodes->shared_indeps.elem_count; ++zz) {
    rarr = (sc_recycle_array_t *) sc_array_index (&nodes->shared_indeps, zz);
    sc_recycle_array_reset (rarr);
  }
  sc_array_reset (&nodes->shared_indeps);
  P4EST_FREE (nodes->shared_offsets);
  P4EST_FREE (nodes->nonlocal_ranks);
  P4EST_FREE (nodes->global_owned_indeps);

  P4EST_FREE (nodes);
}

int
my_p4est_nodes_is_valid (p4est_t * p4est, my_p4est_nodes_t * nodes)
{
  const int           num_procs = p4est->mpisize;
  const int           rank = p4est->mpirank;
  int                 k, prev, owner, pshare, ocount;
  int                *sharers, *sorted;
  int                 failed, bad;
  size_t              zz, position, sharez, num_sharers, max_sharers;
  p4est_topidx_t      otree, ntree;
  p4est_locidx_t      il, num_indep_nodes, local_num;
  p4est_locidx_t      num_owned_indeps, offset_owned_indeps, end_owned_indeps;
  p4est_indep_t      *in;
  sc_recycle_array_t *rarr;

  failed = 0;
  sorted = P4EST_ALLOC (int, INT8_MAX);

  if (nodes->indep_nodes.elem_size != sizeof (p4est_indep_t) ||
      nodes->shared_indeps.elem_size != sizeof (sc_recycle_array_t)) {
    P4EST_NOTICE ("p4est nodes invalid array size\n");
    failed = 1;
    goto failtest;
  }

  if (nodes->num_local_quadrants != p4est->local_num_quadrants) {
    P4EST_NOTICE ("p4est nodes invalid quadrant count\n");
    failed = 1;
    goto failtest;
  }

  max_sharers = nodes->shared_indeps.elem_count;
  for (zz = 0; zz < max_sharers; ++zz) {
    rarr = (sc_recycle_array_t *) sc_array_index (&nodes->shared_indeps, zz);
    num_sharers = zz + 1;
    for (position = 0; position < rarr->a.elem_count; ++position) {
      sharers = (int *) sc_array_index (&rarr->a, position);
      memcpy (sorted, sharers, num_sharers * sizeof (int));
      qsort (sorted, num_sharers, sizeof (int), sc_int_compare);
      prev = -1;
      for (sharez = 0; sharez < num_sharers; ++sharez) {
        k = sorted[sharez];
        if (prev >= k || k == rank || k >= num_procs) {
          P4EST_NOTICE ("p4est nodes invalid sharers 1\n");
          failed = 1;
          goto failtest;
        }
        prev = k;
      }
    }
  }

  num_indep_nodes = (p4est_locidx_t) nodes->indep_nodes.elem_count;
  num_owned_indeps = nodes->num_owned_indeps;
  offset_owned_indeps = nodes->offset_owned_indeps;
  end_owned_indeps = offset_owned_indeps + num_owned_indeps;
  k = prev = 0;
  otree = 0;
  for (il = 0; il < num_indep_nodes; ++il) {
    in = (p4est_indep_t *) sc_array_index (&nodes->indep_nodes, (size_t) il);
    if (!p4est_quadrant_is_node ((p4est_quadrant_t *) in, 0)) {
      P4EST_NOTICE ("p4est nodes independent clamped\n");
      failed = 1;
      goto failtest;
    }
    ntree = in->p.piggy3.which_tree;
    local_num = in->p.piggy3.local_num;
    if (ntree < otree || ntree >= p4est->connectivity->num_trees) {
      P4EST_NOTICE ("p4est nodes invalid tree\n");
      failed = 1;
      goto failtest;
    }
    otree = ntree;
    if (in->pad8 < 0 || (num_sharers = (size_t) in->pad8) > max_sharers) {
      P4EST_NOTICE ("p4est nodes invalid sharer count\n");
      failed = 1;
      goto failtest;
    }
    if (il < offset_owned_indeps || il >= end_owned_indeps) {
      owner = nodes->nonlocal_ranks[k++];
      if (owner < prev || owner == rank || owner >= num_procs) {
        P4EST_NOTICE ("p4est nodes invalid owner\n");
        failed = 1;
        goto failtest;
      }
      if (local_num < 0 || local_num >= nodes->global_owned_indeps[owner]) {
        P4EST_NOTICE ("p4est nodes invalid non-owned index\n");
        failed = 1;
        goto failtest;
      }
      prev = owner;
      if (num_sharers < 1) {
        P4EST_NOTICE ("p4est nodes invalid non-owned sharing 1\n");
        failed = 1;
        goto failtest;
      }
    }
    else {
      if (local_num != il - offset_owned_indeps) {
        P4EST_NOTICE ("p4est nodes invalid owned index\n");
        failed = 1;
        goto failtest;
      }
      owner = prev = rank;
    }
    ocount = 0;
    if (num_sharers > 0) {
      rarr =
        (sc_recycle_array_t *) sc_array_index (&nodes->shared_indeps,
                                               num_sharers - 1);
      if (nodes->shared_offsets == NULL) {
        bad = (in->pad16 < 0);
        position = (size_t) in->pad16;
      }
      else {
        bad = (in->pad16 != -1);
        position = (size_t) nodes->shared_offsets[il];
      }
      if (bad || position >= rarr->a.elem_count) {
        P4EST_NOTICE ("p4est nodes invalid sharer position\n");
        failed = 1;
        goto failtest;
      }
      sharers = (int *) sc_array_index (&rarr->a, position);
      memcpy (sorted, sharers, num_sharers * sizeof (int));
      qsort (sorted, num_sharers, sizeof (int), sc_int_compare);
      pshare = -1;
      for (zz = 0; zz < num_sharers; ++zz) {
        if (sorted[zz] <= pshare || sorted[zz] == rank) {
          P4EST_NOTICE ("p4est nodes invalid sharers 2\n");
          failed = 1;
          goto failtest;
        }
        if (sorted[zz] == owner) {
          ++ocount;
        }
        pshare = sorted[zz];
      }
    }
    if (owner != rank && ocount != 1) {
      P4EST_NOTICE ("p4est nodes invalid non-owned sharing 2\n");
      failed = 1;
      goto failtest;
    }
  }

  /* TODO: Test hanging nodes and local corners. */

failtest:
  P4EST_FREE (sorted);

  return !p4est_comm_sync_flag (p4est, failed, MPI_BOR);
}
