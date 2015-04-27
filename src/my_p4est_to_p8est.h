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

#ifndef MY_P4EST_TO_P8EST_H
#define MY_P4EST_TO_P8EST_H

#include <p4est_to_p8est.h>

/* BUG: Due to a bug in p4est, we need to rename the following variable
 * Note: I have no idea if p4est_ghost_contains would worl with p8est objects
 * but then again its only needed in the point lookup so who cares ...
 */
#undef p4est_ghost_contains
#define p4est_ghost_contains p4est_ghost_contains

// Our variables
#define my_p4est_nodes_t                my_p8est_nodes_t
#define my_p4est_nodes_new              my_p8est_nodes_new
#define my_p4est_brick_new              my_p8est_brick_new
#define my_p4est_brick_destroy          my_p8est_brick_destroy


#endif /* !MY_P4EST_TO_P8EST_H */
