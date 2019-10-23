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

static const int VTK_NODE_SCALAR = VTK_POINT_DATA;
static const int VTK_CELL_SCALAR = VTK_CELL_DATA;
static const int VTK_NODE_VECTOR_BY_COMPONENTS = 2;
static const int VTK_NODE_VECTOR_BLOCK = 3;
static const int VTK_CELL_VECTOR_BY_COMPONENTS = 4;
static const int VTK_CELL_VECTOR_BLOCK = 5;

#ifdef P4_TO_P8
#define P4EST_VTK_CELL_TYPE     11      /* VTK_VOXEL */
#else
#define P4EST_VTK_CELL_TYPE     8       /* VTK_PIXEL */
#endif

/*!
 * \brief my_p4est_vtk_write_all_general is the most general vtk exportation tool available to date [09/10/2019]
 * \param p4est                           the p4est to be exported
 * \param nodes                           the corresponding node structure
 * \param ghost                           the corresponding ghost layer
 * \param write_rank                      flag controlling the exportation of the process ranks (cell-sampled)
 * \param write_tree                      flag controlling the exportation of tree indices (cell-sampled)
 * \param num_node_scalars               Number of node-sampled scalar fields (VTK_POINT_DATA or VTK_NODE_SCALAR) to export.
 * \param num_node_vectors_by_component  Number of node-sampled vector fields given component by component (VTK_NODE_VECTOR_BY_COMPONENTS) to export.
 * \param num_node_vectors_block         Number of node-sampled vector fields given in a P4EST_DIM-block-structured way (VTK_NODE_VECTOR_BLOCK) to export.
 * \param num_cell_scalars                Number of cell-sampled scalar fields (VTK_CELL_DATA or VTK_CELL_SCALAR) to export.
 * \param num_cell_vectors_by_component   Number of cell-sampled vector fields given component by component (VTK_CELL_VECTOR_BY_COMPONENTS) to export.
 * \param num_cell_vectors_block          Number of cell-sampled vector fields given in a P4EST_DIM-block-structured way (VTK_CELL_VECTOR_BLOCK) to export.
 * \param filename                        (absolute) path of the exportation files, the created files are "filename.pvtu", "filename.visit" and as many
 *                                        "filename.vtu/%04d.vtu" as number of processes (where '%04d' is the proc's rank)
 * <\beginning of example\>
 * Example usage: say that you want to export a p4est structure with 2 node-sampled scalar fields (temperature "temp" and levelset "phi"), 1 block-structured
 * node-sampled vector field (say the normal vector, "normal"), 1 node-sampled vector field given component by component (say the velocity field "velocity"),
 * 1 cell-sampled scalar field (say the pressure, "pressure") and 1 cell-sampled vector field given component by component (say the gradieng of the hodge
 * variable "grad_hodge"). Also, say that you want the proc's ranks but not the tree idx. Then you would call the function in the following way:
 * my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
 *                                P4EST_TRUE, P4EST_TRUE,
 *                                2, // number of VTK_NODE_SCALAR
 *                                1, // number of VTK_NODE_VECTOR_BY_COMPONENTS
 *                                1, // number of VTK_NODE_VECTOR_BLOCK
 *                                1, // number of VTK_CELL_SCALAR
 *                                1, // number of VTK_CELL_VECTOR_BY_COMPONENTS
 *                                0, // number of VTK_CELL_VECTOR_BLOCK
 *                                name,
 *                                VTK_NODE_SCALAR, "phi", phi_p,
 *                                VTK_NODE_SCALAR, "temperature", phi_coarse_p,
 *                                VTK_NODE_VECTOR_BLOCK, "normal", normal_block_p,
 *                                VTK_NODE_VECTOR_BY_COMPONENTS, "velocity" , DIM(velocity_p[0], velocity_p[1], velocity_p[2]),
 *                                VTK_CELL_SCALAR, "pressure", pressure_p,
 *                                VTK_CELL_VECTOR_BY_COMPONENTS, "grad_hodge", DIM(grad_hodge_p[0], grad_hodge_p[1], grad_hodge_p[2]));
 * </end of example/>
 *
 * <\beginning of notes\>
 *  - VTK_NODE_SCALAR  is equivalent to (the former) VTK_POINT_DATA
 *  - VTK_CELL_SCALAR   is equivalent to (the former) VTK_CELL_DATA
 *  - calling the former (original) function my_p4est_vtk_write_all() is actually equivalent to calling this general function
 *    with num_node_vectors_by_component = num_node_vectors_block = num_cell_vectors_by_component =  num_cell_vectors_block = 0
 * </end of notes/>
 *
 * Original developper(s) unknown
 * Second developper: Raphael Egan (extension of the original features to node- and/or cell-sampled vector fields,
 * given either component by component or in a block-structured vector.
 */
void                my_p4est_vtk_write_all_general(p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                                   int write_rank, int write_tree,
                                                   int num_node_scalars, int num_node_vectors_by_component, int num_node_vectors_block,
                                                   int num_cell_scalars, int num_cell_vectors_by_component, int num_cell_vectors_block,
                                                   const char *filename, ...);
void                my_p4est_vtk_write_all (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                            int write_rank, int write_tree,
                                            int num_node_scalars, int num_cell_scalars,
                                            const char *filename, ...);
void                my_p4est_vtk_write_all_wrapper (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                                    int write_rank, int write_tree,
                                                    int num_node_scalars, int num_node_vectors_by_component, int num_node_vectors_block,
                                                    int num_cell_scalars, int num_cell_vectors_by_component, int num_cell_vectors_block,
                                                    const char *filename, va_list ap);

/*!
 * \brief my_p4est_vtk_write_header writes the headers of the "filename.pvtu" and of the "filename.vtu/%04d.vtu" files.
 * The headers of the "filename.vtu/%04d.vtu" files includes all geometry-related information: list of nodes and their
 * coordinates, list of cells and their corners (indexed as the latter list of nodes). Periodic conditions are exported
 * by duplicating periodically-mapped nodes in the exportation.
 * \param p4est           the p4est to be exported
 * \param nodes           the corresponding node structure (can be NULL)
 * \param ghost           the corresponding ghost layer (can be NULL)
 * \param filename        (absolute) path of the exportation files
 * \return                this returns 0 if no error and -1 if there is an error.
 */
int                 my_p4est_vtk_write_header (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                                               const char *filename);


/*!
 * \brief my_p4est_vtk_write_node_data writes node data to the relevant exportation files
 * \param p4est                             the p4est to be exported
 * \param nodes                             the corresponding node structure
 * \param ghost                             the corresponding ghost layer (can be NULL)
 * \param filename                          (absolute) path of the exportation files
 * \param num_scalar                        number of node-sampled scalar fields
 * \param num_vector_block                  number of node-sampled vector fields, P4EST_DIM-block-structured
 * \param num_vector_by_component           number of node-sampled vector fields given component by component
 * \param list_name_scalar                  chain of characters listing the names of the node-sampled scalar fields to be exported,
 *                                          separated by commas
 * \param list_name_vector_block            chain of characters listing the names of the node-sampled vector fields,
 *                                          P4EST_DIM-block-structured, to be exported, separated by commas
 * \param list_name_vector_by_component     chain of characters listing the names of the node-sampled vector fields,
 *                                          given component by component, to be exported, separated by commas
 * \param scalar_names                      array of pointers to the individual chains of characters for the names of the node-sampled
 *                                          scalar fields to be exported.
 * \param vector_block_names                array of pointers to the individual chains of characters for the names of the node-sampled
 *                                          vector fields, P4EST_DIM-block-structured, to be exported
 * \param vector_by_component_names         array of pointers to the individual chains of characters for the names of the node-sampled
 *                                          vector fields, given dimension by dimension, to be exported
 * \param scalar_values                     array of pointers to node-sampled data corresponding to the node-sampled scalar fields to be
 *                                          exported. scalar_values[i][il] is the value of the ith scalar field sampled at (local) node il
 * \param vector_block_values               array of pointers to node-sampled data corresponding to the node-sampled P4EST_DIM-block-structured
 *                                          vector fields to be exported. vector_block_values[i][P4EST_DIM*il+k] is the value of the kth component
 *                                          of the ith vector field that is block-structured, sampled at (local) node il
 * \param vector_by_component_values        array of P4EST_DIM arrays of pointers to node-sampled data corresponding to the node-sampled components
 *                                          of the vector fields to be exported. vector_by_component_values[k][i][il] is the value of the kth component
 *                                          of the ith vector field given component by component, sampled at (local) node il
 * \return                                  this returns 0 if no error and -1 if there is an error.
 *
 * Original developper(s) unknown
 * Second developper: Raphael Egan (extension of the original features to node- and/or cell-sampled vector fields,
 * given either component by component or in a block-structured vector.
 */
int                 my_p4est_vtk_write_node_data (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, const char *filename,
                                                   const int num_scalar, const int num_vector_block, const int num_vector_by_component,
                                                   const char* list_name_scalar, const char* list_name_vector_block, const char* list_name_vector_by_component,
                                                   const char **scalar_names, const char **vector_block_names, const char **vector_by_component_names,
                                                   const double **scalar_values, const double **vector_block_values, const double **vector_by_component_values[P4EST_DIM]);

/*!
 * \brief my_p4est_vtk_write_cell_data writes cell data to the relevant exportation files
 * \param p4est                             the p4est to be exported
 * \param ghost                             the corresponding ghost layer (can be NULL)
 * \param write_rank                        flag controlling the exportation of cell-sampled proc's rank (the procs' ranks are/are not exported if set to P4EST_TRUE/P4EST_FALSE)
 * \param write_tree                        flag controlling the exportation of cell-sampled tree index (the tree index is/is not exported if set to P4EST_TRUE/P4EST_FALSE)
 * \param filename                          (absolute) path of the exportation files
 * \param list_name_scalar                  chain of characters listing the names of the cell-sampled scalar fields to be exported,
 *                                          separated by commas
 * \param list_name_vector_block            chain of characters listing the names of the cell-sampled vector fields,
 *                                          P4EST_DIM-block-structured, to be exported, separated by commas
 * \param list_name_vector_by_component     chain of characters listing the names of the cell-sampled vector fields,
 *                                          given component by component, to be exported, separated by commas
 * \param scalar_names                      array of pointers to the individual chains of characters for the names of the cell-sampled
 *                                          scalar fields to be exported.
 * \param vector_block_names                array of pointers to the individual chains of characters for the names of the cell-sampled
 *                                          vector fields, P4EST_DIM-block-structured, to be exported
 * \param vector_by_component_names         array of pointers to the individual chains of characters for the names of the cell-sampled
 *                                          vector fields, given dimension by dimension, to be exported
 * \param scalar_values                     array of pointers to cell-sampled data corresponding to the cell-sampled scalar fields to be
 *                                          exported. scalar_values[i][il] is the value of the ith scalar field sampled at (local) node il
 * \param vector_block_values               array of pointers to cell-sampled data corresponding to the cell-sampled P4EST_DIM-block-structured
 *                                          vector fields to be exported. vector_block_values[i][P4EST_DIM*il+k] is the value of the kth component
 *                                          of the ith vector field that is block-structured, sampled at (local) node il
 * \param vector_by_component_values        array of P4EST_DIM arrays of pointers to cell-sampled data corresponding to the cell-sampled components
 *                                          of the vector fields to be exported. vector_by_component_values[k][i][il] is the value of the kth component
 *                                          of the ith vector field given component by component, sampled at (local) node il
 * \return                                  this returns 0 if no error and -1 if there is an error.
 *
 * Original developper(s) unknown
 * Second developper: Raphael Egan (extension of the original features to node- and/or cell-sampled vector fields,
 * given either component by component or in a block-structured vector.
 */
int                 my_p4est_vtk_write_cell_data (p4est_t * p4est, p4est_ghost_t *ghost,
                                                  int write_rank, int write_tree, const char *filename,
                                                  const int num_scalar, const int num_vector_block, const int num_vector_by_component,
                                                  const char* list_name_scalar, const char* list_name_vector_block, const char* list_name_vector_by_component,
                                                  const char **scalar_names, const char **vector_block_names, const char **vector_by_component_names,
                                                  const double **scalar_values, const double **vector_block_values, const double **vector_by_component_values[P4EST_DIM]);

/*!
 * \brief my_p4est_vtk_write_footer writes the footers of the "filename.pvtu" and of the "filename.vtu/%04d.vtu" files.
 * (this completes and ends the open sections of the xml-format files open for exportation)
 * \param p4est           the p4est to be exported
 * \param filename        (absolute) path of the exportation files
 * \return                this returns 0 if no error and -1 if there is an error.
 */
int                 my_p4est_vtk_write_footer (p4est_t * p4est,
                                               const char *filename);

void my_p4est_vtk_write_ghost_layer(p4est_t *p4est, p4est_ghost_t *ghost);

SC_EXTERN_C_END;

#endif /* !MY_P4EST_VTK_H */
