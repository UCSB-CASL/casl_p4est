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

/** Create node information.
 * All nodes, including hanging nodes, are considered unique and independent.
 * Function is compatible with p8est_nodes_is_valid and p8est_nodes_destroy.
 * \param [in] p8est    The forest.  Does not need to be balanced.
 * \return              A fully allocated and valid p8est_nodes_t structure.
 *                      It does not consider any node as hanging.
 *                      See p8est_nodes.h for details.
 */
p8est_nodes_t      *my_p8est_nodes_new (p8est_t * p8est);

SC_EXTERN_C_END;

#endif /* !MY_P8EST_NODES_H */
