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

#ifndef MY_P4EST_VTK_H
#define MY_P4EST_VTK_H

#ifdef P4_TO_P8
#include "my_p8est_nodes.h"
#include <p8est_bits.h>
#else
#include "my_p4est_nodes.h"
#include <p4est_bits.h>
#endif


/********************************************************************
 *                          IMPORTANT NOTE                          *
 *                                                                  *
 * The p4est_geometry interface will be removed shortly.            *
 * Please do NOT use this interface for newly written code.         *
 * It will be replaced with a generic transfinite blending scheme.  *
 ********************************************************************/

SC_EXTERN_C_BEGIN;

static const int VTK_POINT_DATA = 0;
static const int VTK_CELL_DATA  = 1;

#ifdef P4_TO_P8
#define P4EST_VTK_CELL_TYPE     11      /* VTK_VOXEL */
#else
#define P4EST_VTK_CELL_TYPE     8       /* VTK_PIXEL */
#endif

/** This writes out the p4est and any number of point fields in VTK format.
 *
 * This is a convenience function that will abort if there is a file error.
 *
 * \param [in] p4est    The p4est to be written.
 * \param [in] geom     A p4est_geometry_t structure or NULL for identity.
 * \param [in] scale    Double value between 0 and 1 to scale each quadrant.
 * \param filename      First part of the name, see p4est_vtk_write_file.
 * \param num_cell_scalars   Number of scalar fields (CELL_DATA) to write.
 * \param num_point_scalars   Number of scalar fields (POINT_DATA) to write.
 *
 * The variable arguments need to be pairs of (fieldname, fieldvalues)
 * where the scalars come first, then the vectors.
 */
void                my_p4est_vtk_write_all (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                            int write_rank, int write_tree,
                                            int num_point_scalars, int num_cell_scalars,
                                            const char *filename, ...);

/** This will write the header of the vtu file.
 *
 * Writing a VTK file is split into a couple of routines.
 * The allows there to be an arbitrary number of
 * fields.  The calling sequence would be something like
 *
 * \begincode
 * p4est_vtk_write_header(p4est, geom, 1., 1, 1, 0, "output");
 * p4est_vtk_write_point_scalar (...);
 * ...
 * p4est_vtk_write_footer(p4est, "output");
 * \endcode
 *
 * \param p4est     The p4est to be written.
 * \param geom      A p4est_geometry_t structure or NULL for identity.
 * \param scale     The relative length factor of the quadrants.
 *                  Use 1.0 to fit quadrants exactly, less to create gaps.
 * \param filename  The first part of the name which will have
 *                  the proc number appended to it (i.e., the
 *                  output file will be filename_procNum.vtu).
 *
 * \return          This returns 0 if no error and -1 if there is an error.
 */
int                 my_p4est_vtk_write_header (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                               const char *filename);

/** This will write a scalar field to the vtu file.
 *
 * It is good practice to make sure that the scalar field also
 * exists in the comma separated string \a point_scalars passed
 * to \c p4est_vtk_write_header.
 *
 * Writing a VTK file is split into a couple of routines.
 * The allows there to be an arbitrary number of fields.
 *
 * \param p4est     The p4est to be written.
 * \param geom      A p4est_geometry_t structure or NULL for identity.
 * \param filename  The first part of the name which will have
 *                  the proc number appended to it (i.e., the
 *                  output file will be filename_procNum.vtu).
 * \param scalar_name The name of the scalar field.
 * \param values    The point values that will be written.
 *
 * \return          This returns 0 if no error and -1 if there is an error.
 */

int                 my_p4est_vtk_write_point_scalar (p4est_t * p4est, p4est_nodes_t *nodes,
                                                     const char *filename,
                                                     const int num, const char *list_name, const char **scalar_names,
                                                     const double **values);
/** This will write a scalar field to the vtu file.
 *
 * It is good practice to make sure that the scalar field also
 * exists in the comma separated string \a point_scalars passed
 * to \c p4est_vtk_write_header.
 *
 * Writing a VTK file is split into a couple of routines.
 * The allows there to be an arbitrary number of fields.
 *
 * \param p4est     The p4est to be written.
 * \param geom      A p4est_geometry_t structure or NULL for identity.
 * \param filename  The first part of the name which will have
 *                  the proc number appended to it (i.e., the
 *                  output file will be filename_procNum.vtu).
 * \param scalar_name The name of the scalar field.
 * \param values    The point values that will be written.
 *
 * \return          This returns 0 if no error and -1 if there is an error.
 */
int                 my_p4est_vtk_write_cell_scalar (p4est_t * p4est, p4est_ghost_t *ghost,
                                                    int write_rank, int write_tree,
                                                    const char *filename, const int num,
                                                    const char *list_name, const char **scalar_names,
                                                    const double **values);


/** This will write the footer of the vtu file.
 *
 * Writing a VTK file is split into a couple of routines.
 * The allows there to be an arbitrary number of
 * fields.  To write out two fields the
 * calling sequence would be something like
 *
 * \begincode
 * p4est_vtk_write_header(p4est, ..., "output");
 * p4est_vtk_write_footer(p4est, "output");
 * \endcode
 *
 * \param p4est     The p4est to be written.
 * \param filename  The first part of the name which will have
 *                  the proc number appended to it (i.e., the
 *                  output file will be filename_procNum.vtu).
 *
 * \return          This returns 0 if no error and -1 if there is an error.
 */
int                 my_p4est_vtk_write_footer (p4est_t * p4est,
                                               const char *filename);

void my_p4est_vtk_write_ghost_layer(p4est_t *p4est, p4est_ghost_t *ghost);

SC_EXTERN_C_END;

#endif /* !MY_P4EST_VTK_H */
