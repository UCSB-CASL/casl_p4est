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

/********************************************************************
 *                          IMPORTANT NOTE                          *
 *                                                                  *
 * The p4est_geometry interface will be removed shortly.            *
 * Please do NOT use this interface for newly written code.         *
 * It will be replaced with a generic transfinite blending scheme.  *
 ********************************************************************/

#ifdef P4_TO_P8
#include "my_p8est_vtk.h"
#include <p8est_nodes.h>
#define P4EST_VTK_CELL_TYPE     11      /* VTK_VOXEL */
#else
#include "my_p4est_vtk.h"
#include <p4est_nodes.h>
#define P4EST_VTK_CELL_TYPE      8      /* VTK_PIXEL */
#endif /* !P4_TO_P8 */

#include <sc_io.h>
#include <stdio.h>
#include <petsclog.h>

static const double p4est_vtk_scale = 1.0;
static const int    p4est_vtk_write_tree = 1;
static const int    p4est_vtk_write_rank = 1;
static const int    p4est_vtk_wrap_rank = 0;

#undef P4EST_VTK_COMPRESSION

#ifndef P4EST_VTK_DOUBLES
#define P4EST_VTK_FLOAT_NAME "Float32"
#define P4EST_VTK_FLOAT_TYPE float
#else
#define P4EST_VTK_FLOAT_NAME "Float64"
#define P4EST_VTK_FLOAT_TYPE double
#endif

#ifndef P4EST_VTK_BINARY
#define P4EST_VTK_ASCII 1
#define P4EST_VTK_FORMAT_STRING "ascii"
#else
#define P4EST_VTK_FORMAT_STRING "binary"

static int
my_p4est_vtk_write_binary (FILE * vtkfile, char *numeric_data,
                           size_t byte_length)
{
#ifndef P4EST_VTK_COMPRESSION
  return sc_vtk_write_binary (vtkfile, numeric_data, byte_length);
#else
  return sc_vtk_write_compressed(vtkfile, numeric_data, byte_length);
#endif /* P4EST_VTK_COMPRESSION */
}

#endif /* P4EST_VTK_BINARY */

// logging variable -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_vtk_write_all;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void
my_p4est_vtk_write_all (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                        int write_rank, int write_tree,
                        int num_point_scalars, int num_cell_scalars,
                        const char *filename, ...)
{
  PetscErrorCode ierr;
  int                 retval;
  int                 i, all_p, all_c;
  int                 cell_scalar_strlen, point_scalar_strlen;
  char                cell_scalars[BUFSIZ], point_scalars[BUFSIZ];
  const char         *cell_name, *point_name, **cell_names, **point_names;
  const double       **cell_values, **point_values;
  int            vtk_type;
  va_list             ap;

  P4EST_ASSERT (num_cell_scalars >= 0  && num_point_scalars >= 0 );

  // logging
  ierr = PetscLogEventBegin(log_my_p4est_vtk_write_all, 0, 0, 0, 0); CHKERRV(ierr);

  /* Allocate memory for the data and their names */
  cell_values  = P4EST_ALLOC(const double * , num_cell_scalars);
  point_values = P4EST_ALLOC(const double * , num_point_scalars);
  cell_names   = P4EST_ALLOC (const char *, num_cell_scalars);
  point_names  = P4EST_ALLOC (const char *, num_point_scalars);

  va_start (ap, filename);
  all_c = all_p = 0;
  cell_scalar_strlen = point_scalar_strlen = 0;
  cell_scalars[0] = point_scalars[0] = '\0';
  for (i = 0; i < num_point_scalars+num_cell_scalars; ++i) {
    /* first get the type */
    vtk_type = va_arg(ap, int);

    if (vtk_type == VTK_POINT_DATA){
      point_name = point_names[all_p] = va_arg(ap, const char*);
      retval = snprintf (point_scalars + point_scalar_strlen, BUFSIZ - point_scalar_strlen,
                         "%s%s", all_p == 0 ? "" : ", ", point_name);
      SC_CHECK_ABORT (retval > 0,
                      P4EST_STRING "_vtk: Error collecting point scalars");
      point_scalar_strlen += retval;

      /* now get the values */
      point_values[all_p] = va_arg(ap, double*);

      all_p++;
    } else if (vtk_type == VTK_CELL_DATA){
      cell_name = cell_names[all_c] = va_arg(ap, const char*);
      retval = snprintf (cell_scalars + cell_scalar_strlen, BUFSIZ - cell_scalar_strlen,
                         "%s%s", all_c == 0 ? "" : ", ", cell_name);
      SC_CHECK_ABORT (retval > 0,
                      P4EST_STRING "_vtk: Error collecting point scalars");
      cell_scalar_strlen += retval;

      /* get the values */
      cell_values[all_c] = va_arg(ap, double*);
      all_c++;
    }
  }

  va_end (ap);


  retval = my_p4est_vtk_write_header (p4est, nodes, ghost, filename);
  SC_CHECK_ABORT (!retval, P4EST_STRING "_vtk: Error writing header");

  /* now write the actual data */

  retval = my_p4est_vtk_write_point_scalar (p4est, nodes, filename,
                                            num_point_scalars, point_scalars, point_names, point_values);
  SC_CHECK_ABORT (!retval,
                  P4EST_STRING "_vtk: Error writing point scalars");

  retval = my_p4est_vtk_write_cell_scalar (p4est, ghost, write_rank, write_tree, filename,
                                           num_cell_scalars, cell_scalars, cell_names, cell_values);
  SC_CHECK_ABORT (!retval,
                  P4EST_STRING "_vtk: Error writing cell scalars");


  retval = my_p4est_vtk_write_footer (p4est, filename);
  SC_CHECK_ABORT (!retval, P4EST_STRING "_vtk: Error writing footer");

  P4EST_FREE (cell_values);
  P4EST_FREE (point_values);
  P4EST_FREE (cell_names);
  P4EST_FREE (point_names);

  ierr = PetscLogEventEnd(log_my_p4est_vtk_write_all, 0, 0, 0, 0); CHKERRV(ierr);
}

int
my_p4est_vtk_write_header (p4est_t * p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                           const char *filename)
{
  p4est_connectivity_t *connectivity = p4est->connectivity;
  sc_array_t         *trees = p4est->trees;
  const int           mpirank = p4est->mpirank;
  const double        intsize = 1.0 / P4EST_ROOT_LEN;
  const double       *v = connectivity->vertices;
  const p4est_topidx_t first_local_tree = p4est->first_local_tree;
  const p4est_topidx_t last_local_tree = p4est->last_local_tree;
  const p4est_topidx_t *tree_to_vertex = connectivity->tree_to_vertex;
  p4est_locidx_t Ncells = p4est->local_num_quadrants;
  if (ghost != NULL)
    Ncells += ghost->ghosts.elem_count;
  const p4est_locidx_t Ncorners = P4EST_CHILDREN * Ncells;
#ifdef P4EST_VTK_ASCII
  double              wx, wy, wz;
  p4est_locidx_t      sk;
#else
  int                 retval;
  uint8_t            *uint8_data;
  p4est_locidx_t     *locidx_data;
#endif
  int                 xi, yi, j, k;
#ifdef P4_TO_P8
  int                 zi;
  double              XYZ[3];   /* 3 not P4EST_DIM */
#endif
  double              h2, eta_x, eta_y, eta_z = 0.;
  double              xyz[3];   /* 3 not P4EST_DIM */
  size_t              num_quads, zz;
  p4est_topidx_t      jt;
  p4est_topidx_t      vt[P4EST_CHILDREN];
  p4est_locidx_t      quad_count, Ntotal;
  p4est_locidx_t      il;
  P4EST_VTK_FLOAT_TYPE *float_data;
  sc_array_t         *quadrants, *indeps;
  p4est_tree_t       *tree;
  p4est_quadrant_t   *quad;
  p4est_indep_t      *in;
  char                vtufilename[BUFSIZ];
  FILE               *vtufile;

  SC_CHECK_ABORT (p4est->connectivity->num_vertices > 0,
                  "Must provide connectivity with vertex information");

  P4EST_ASSERT (v != NULL && tree_to_vertex != NULL);

  if (nodes != NULL){
    indeps = &nodes->indep_nodes;
    Ntotal = indeps->elem_count;
  } else {
    indeps = NULL;
    Ntotal = Ncorners;
  }

  /* Have each proc write to its own file */
  snprintf (vtufilename, BUFSIZ, "%s_%04d.vtu", filename, mpirank);
  /* Use "w" for writing the initial part of the file.
   * For further parts, use "r+" and fseek so write_compressed succeeds.
   */
  vtufile = fopen (vtufilename, "wb");
  if (vtufile == NULL) {
    P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
    return -1;
  }

  fprintf (vtufile, "<?xml version=\"1.0\"?>\n");
  fprintf (vtufile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"");
#if defined P4EST_VTK_BINARY && defined P4EST_VTK_COMPRESSION
  fprintf (vtufile, " compressor=\"vtkZLibDataCompressor\"");
#endif
#ifdef SC_WORDS_BIGENDIAN
  fprintf (vtufile, " byte_order=\"BigEndian\">\n");
#else
  fprintf (vtufile, " byte_order=\"LittleEndian\">\n");
#endif
  fprintf (vtufile, "  <UnstructuredGrid>\n");
  fprintf (vtufile,
           "    <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n",
           (long long) Ntotal, (long long) Ncells);
  fprintf (vtufile, "      <Points>\n");

  float_data = P4EST_ALLOC (P4EST_VTK_FLOAT_TYPE, 3 * Ntotal);

  /* write point position data */
  fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"Position\""
           " NumberOfComponents=\"3\" format=\"%s\">\n",
           P4EST_VTK_FLOAT_NAME, P4EST_VTK_FORMAT_STRING);

  if (nodes == NULL) {
    /* loop over the trees */
    for (jt = first_local_tree, quad_count = 0; jt <= last_local_tree; ++jt) {
      tree = p4est_tree_array_index (trees, jt);
      quadrants = &tree->quadrants;
      num_quads = quadrants->elem_count;

      /* retrieve corners of the tree */
      for (k = 0; k < P4EST_CHILDREN; ++k)
        vt[k] = tree_to_vertex[jt * P4EST_CHILDREN + k];

      /* loop over the elements in tree and calculated vertex coordinates */
      for (zz = 0; zz < num_quads; ++zz, ++quad_count) {
        quad = p4est_quadrant_array_index (quadrants, zz);
        h2 = .5 * intsize * P4EST_QUADRANT_LEN (quad->level);
        k = 0;
#ifdef P4_TO_P8
        for (zi = 0; zi < 2; ++zi) {
#endif
          for (yi = 0; yi < 2; ++yi) {
            for (xi = 0; xi < 2; ++xi) {
              P4EST_ASSERT (0 <= k && k < P4EST_CHILDREN);
              eta_x = intsize * quad->x + h2 * (1. + (xi * 2 - 1));
              eta_y = intsize * quad->y + h2 * (1. + (yi * 2 - 1));
#ifdef P4_TO_P8
              eta_z = intsize * quad->z + h2 * (1. + (zi * 2 - 1));
#endif
              for (j = 0; j < 3; ++j) {
                /* *INDENT-OFF* */
                xyz[j] =
                    ((1. - eta_z) * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[0] + j] +
                     eta_x  * v[3 * vt[1] + j]) +
                    eta_y  * ((1. - eta_x) * v[3 * vt[2] + j] +
                    eta_x  * v[3 * vt[3] + j]))
    #ifdef P4_TO_P8
                    +     eta_z  * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[4] + j] +
                    eta_x  * v[3 * vt[5] + j]) +
                    eta_y  * ((1. - eta_x) * v[3 * vt[6] + j] +
                    eta_x  * v[3 * vt[7] + j]))
    #endif
                    );
                /* *INDENT-ON* */
              }

              for (j = 0; j < 3; ++j) {
                float_data[3 * (P4EST_CHILDREN * quad_count + k) +
                    j] = (P4EST_VTK_FLOAT_TYPE) xyz[j];
              }
              ++k;
            }
          }
#ifdef P4_TO_P8
        }
#endif
        P4EST_ASSERT (k == P4EST_CHILDREN);
      }
    }
    P4EST_ASSERT (P4EST_CHILDREN * quad_count == Ntotal);
  }
  else {
    for (zz = 0; zz < indeps->elem_count; ++zz) {
      in = (p4est_indep_t *) sc_array_index (indeps, zz);

      /* retrieve corners of the tree */
      jt = in->p.which_tree;
      for (k = 0; k < P4EST_CHILDREN; ++k)
        vt[k] = tree_to_vertex[jt * P4EST_CHILDREN + k];

      /* calculate vertex coordinates */
      eta_x = intsize * in->x;
      eta_y = intsize * in->y;
#ifdef P4_TO_P8
      eta_z = intsize * in->z;
#endif
      for (j = 0; j < 3; ++j) {
        /* *INDENT-OFF* */
        xyz[j] =
            ((1. - eta_z) * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[0] + j] +
             eta_x  * v[3 * vt[1] + j]) +
            eta_y  * ((1. - eta_x) * v[3 * vt[2] + j] +
            eta_x  * v[3 * vt[3] + j]))
    #ifdef P4_TO_P8
            +     eta_z  * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[4] + j] +
            eta_x  * v[3 * vt[5] + j]) +
            eta_y  * ((1. - eta_x) * v[3 * vt[6] + j] +
            eta_x  * v[3 * vt[7] + j]))
    #endif
            );
        /* *INDENT-ON* */
      }
      for (j = 0; j < 3; ++j) {
        float_data[3 * zz + j] = (P4EST_VTK_FLOAT_TYPE) xyz[j];
      }
    }
  }

#ifdef P4EST_VTK_ASCII
  for (il = 0; il < Ntotal; ++il) {
    wx = float_data[3 * il + 0];
    wy = float_data[3 * il + 1];
    wz = float_data[3 * il + 2];

#ifdef P4EST_VTK_DOUBLES
    fprintf (vtufile, "     %24.16e %24.16e %24.16e\n", wx, wy, wz);
#else
    fprintf (vtufile, "          %16.8e %16.8e %16.8e\n", wx, wy, wz);
#endif
  }
#else
  fprintf (vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = my_p4est_vtk_write_binary (vtufile, (char *) float_data,
                                      sizeof (*float_data) * 3 * Ntotal);
  fprintf (vtufile, "\n");
  if (retval) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error encoding points\n");
    fclose (vtufile);
    return -1;
  }
#endif
  P4EST_FREE (float_data);
  fprintf (vtufile, "        </DataArray>\n");
  fprintf (vtufile, "      </Points>\n");
  fprintf (vtufile, "      <Cells>\n");

  /* write connectivity data */
  fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"connectivity\""
           " format=\"%s\">\n", P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
#ifdef P4EST_VTK_ASCII
  for (sk = 0, il = 0; il < Ncells; ++il) {
    fprintf (vtufile, "         ");
    for (k = 0; k < P4EST_CHILDREN; ++sk, ++k) {
      fprintf (vtufile, " %lld", nodes == NULL ?
                 (long long) sk : (long long) nodes->local_nodes[sk]);
    }
    fprintf (vtufile, "\n");
  }
#else
  locidx_data = P4EST_ALLOC (p4est_locidx_t, Ncorners);
  fprintf (vtufile, "          ");
  if (nodes == NULL) {
    for (il = 0; il < Ncorners; ++il) {
      locidx_data[il] = il;
    }
    retval =
        my_p4est_vtk_write_binary (vtufile, (char *) locidx_data,
                                   sizeof (*locidx_data) * Ncorners);
  }
  else {
    retval =
        my_p4est_vtk_write_binary (vtufile, (char *) nodes->local_nodes,
                                   sizeof (*locidx_data) * Ncorners);
  }
  fprintf (vtufile, "\n");
  if (retval) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error encoding connectivity\n");
    fclose (vtufile);
    return -1;
  }
#endif
  fprintf (vtufile, "        </DataArray>\n");

  /* write offset data */
  fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"offsets\""
           " format=\"%s\">\n", P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
#ifdef P4EST_VTK_ASCII
  fprintf (vtufile, "         ");
  for (il = 1, sk = 1; il <= Ncells; ++il, ++sk) {
    fprintf (vtufile, " %lld", (long long) (P4EST_CHILDREN * il));
    if (!(sk % 8) && il != Ncells)
      fprintf (vtufile, "\n         ");
  }
  fprintf (vtufile, "\n");
#else
  for (il = 1; il <= Ncells; ++il)
    locidx_data[il - 1] = P4EST_CHILDREN * il;  /* same type */

  fprintf (vtufile, "          ");
  retval = my_p4est_vtk_write_binary (vtufile, (char *) locidx_data,
                                      sizeof (*locidx_data) * Ncells);
  fprintf (vtufile, "\n");
  if (retval) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error encoding offsets\n");
    fclose (vtufile);
    return -1;
  }
#endif
  fprintf (vtufile, "        </DataArray>\n");

  /* write type data */
  fprintf (vtufile, "        <DataArray type=\"UInt8\" Name=\"types\""
           " format=\"%s\">\n", P4EST_VTK_FORMAT_STRING);
#ifdef P4EST_VTK_ASCII
  fprintf (vtufile, "         ");
  for (il = 0, sk = 1; il < Ncells; ++il, ++sk) {
    fprintf (vtufile, " %d", P4EST_VTK_CELL_TYPE);
    if (!(sk % 20) && il != (Ncells - 1))
      fprintf (vtufile, "\n         ");
  }
  fprintf (vtufile, "\n");
#else
  uint8_data = P4EST_ALLOC (uint8_t, Ncells);
  for (il = 0; il < Ncells; ++il)
    uint8_data[il] = P4EST_VTK_CELL_TYPE;

  fprintf (vtufile, "          ");
  retval = my_p4est_vtk_write_binary (vtufile, (char *) uint8_data,
                                      sizeof (*uint8_data) * Ncells);
  P4EST_FREE (uint8_data);
  fprintf (vtufile, "\n");
  if (retval) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error encoding types\n");
    fclose (vtufile);
    return -1;
  }
#endif
  fprintf (vtufile, "        </DataArray>\n");
  fprintf (vtufile, "      </Cells>\n");

#ifndef P4EST_VTK_ASCII
  P4EST_FREE (locidx_data);
#endif

  if (ferror (vtufile)) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error writing header\n");
    fclose (vtufile);
    return -1;
  }
  if (fclose (vtufile)) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error closing header\n");
    return -1;
  }
  vtufile = NULL;

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    char                pvtufilename[BUFSIZ];
    FILE               *pvtufile;

    snprintf (pvtufilename, BUFSIZ, "%s.pvtu", filename);

    pvtufile = fopen (pvtufilename, "wb");
    if (!pvtufile) {
      P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
      return -1;
    }

    fprintf (pvtufile, "<?xml version=\"1.0\"?>\n");
    fprintf (pvtufile, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\"");
#if defined P4EST_VTK_BINARY && defined P4EST_VTK_COMPRESSION
    fprintf (pvtufile, " compressor=\"vtkZLibDataCompressor\"");
#endif
#ifdef SC_WORDS_BIGENDIAN
    fprintf (pvtufile, " byte_order=\"BigEndian\">\n");
#else
    fprintf (pvtufile, " byte_order=\"LittleEndian\">\n");
#endif

    fprintf (pvtufile, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    fprintf (pvtufile, "    <PPoints>\n");
    fprintf (pvtufile, "      <PDataArray type=\"%s\" Name=\"Position\""
             " NumberOfComponents=\"3\" format=\"%s\"/>\n",
             P4EST_VTK_FLOAT_NAME, P4EST_VTK_FORMAT_STRING);
    fprintf (pvtufile, "    </PPoints>\n");

    if (ferror (pvtufile)) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error writing parallel header\n");
      fclose (pvtufile);
      return -1;
    }
    if (fclose (pvtufile)) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error closing parallel header\n");
      return -1;
    }
  }

  return 0;
}

int
my_p4est_vtk_write_point_scalar (p4est_t * p4est, p4est_nodes_t *nodes,
                                 const char *filename,
                                 const int num, const char* list_name, const char **scalar_names, const double **values)
{
  const int           mpirank = p4est->mpirank;
  int                 retval;
  p4est_locidx_t      il;
#ifndef P4EST_VTK_ASCII
  P4EST_VTK_FLOAT_TYPE *float_data;
#endif
  char                vtufilename[BUFSIZ];
  FILE               *vtufile;

  p4est_locidx_t Ntotal;
  P4EST_ASSERT(nodes != NULL);
  Ntotal = nodes->indep_nodes.elem_count;

  /* Have each proc write to its own file */
  snprintf (vtufilename, BUFSIZ, "%s_%04d.vtu", filename, mpirank);
  /* To be able to fseek in a file you cannot open in append mode.
   * so you need to open with "r+" and fseek to SEEK_END.
   */

  vtufile = fopen (vtufilename, "rb+");
  if (vtufile == NULL) {
    P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
    return -1;
  }

  retval = fseek (vtufile, 0L, SEEK_END);
  if (retval) {
    P4EST_LERRORF ("Could not fseek %s for output\n", vtufilename);
    fclose (vtufile);
    return -1;
  }

  /* Point Data */
  fprintf (vtufile, "      <PointData");
  if (values != NULL)
    fprintf (vtufile, " Scalars=\"%s\"", list_name);
  fprintf (vtufile, ">\n");

  int i;
#ifndef P4EST_VTK_ASCII
  float_data = P4EST_ALLOC (P4EST_VTK_FLOAT_TYPE, Ntotal);
#endif
  for (i=0; i<num; ++i){
    /* write point position data */
    fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
             " format=\"%s\">\n",
             P4EST_VTK_FLOAT_NAME, scalar_names[i], P4EST_VTK_FORMAT_STRING);

#ifdef P4EST_VTK_ASCII
    for (il = 0; il < Ntotal; ++il) {
#ifdef P4EST_VTK_DOUBLES
      fprintf (vtufile, "     %24.16e\n", values[i][il]);
#else
      fprintf (vtufile, "          %16.8e\n", values[i][il]);
#endif
    }
#else

    for (il = 0; il<(p4est_locidx_t)nodes->indep_nodes.elem_count; ++il)
      float_data[il] = (P4EST_VTK_FLOAT_TYPE) values[i][il];

    fprintf (vtufile, "          ");
    /* TODO: Don't allocate the full size of the array, only allocate
     * the chunk that will be passed to zlib and do this a chunk
     * at a time.
     */

    retval = my_p4est_vtk_write_binary (vtufile, (char *) float_data,
                                        sizeof (*float_data) * Ntotal);
    fprintf (vtufile, "\n");


    if (retval) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error encoding points\n");
      fclose (vtufile);
      return -1;
    }


#endif

    fprintf (vtufile, "        </DataArray>\n");

    if (ferror (vtufile)) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error writing point scalar\n");
      fclose (vtufile);
      return -1;
    }
  }


  fprintf (vtufile, "      </PointData>\n");
  if (fclose (vtufile)) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error closing point scalar\n");
    return -1;
  }
  vtufile = NULL;
#ifndef P4EST_VTK_ASCII
  P4EST_FREE (float_data);
#endif

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    char                pvtufilename[BUFSIZ];
    FILE               *pvtufile;
    snprintf (pvtufilename, BUFSIZ, "%s.pvtu", filename);

    pvtufile = fopen (pvtufilename, "ab");
    if (!pvtufile) {
      P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
      return -1;
    }

    fprintf (pvtufile, "    <PPointData Scalars=\"%s\">\n",list_name);

    int i;
    for (i=0; i<num; ++i){
      fprintf (pvtufile, "      <PDataArray type=\"%s\" Name=\"%s\""
               " format=\"%s\"/>\n",
               P4EST_VTK_FLOAT_NAME, scalar_names[i], P4EST_VTK_FORMAT_STRING);

      if (ferror (pvtufile)) {
        P4EST_LERROR (P4EST_STRING
                      "_vtk: Error writing parallel point scalar\n");
        fclose (pvtufile);
        return -1;
      }
    }

    fprintf (pvtufile, "    </PPointData>\n");

    if (fclose (pvtufile)) {
      P4EST_LERROR (P4EST_STRING
                    "_vtk: Error closing parallel point scalar\n");
      return -1;
    }
  }

  return 0;
}

int
my_p4est_vtk_write_cell_scalar (p4est_t * p4est, p4est_ghost_t *ghost,
                                int write_rank, int write_tree,
                                const char *filename,
                                const int num, const char* list_name, const char **scalar_names, const double **values)
{
  const int           mpirank = p4est->mpirank;
  p4est_locidx_t Ncells = p4est->local_num_quadrants;
  if (ghost != NULL)
    Ncells += ghost->ghosts.elem_count;

  int                 retval;
  p4est_locidx_t      il, jt, zz;
#ifndef P4EST_VTK_ASCII
  P4EST_VTK_FLOAT_TYPE *float_data;
  p4est_locidx_t     *locidx_data;
#else
  p4est_locidx_t sk;
#endif
  char                vtufilename[BUFSIZ];
  FILE               *vtufile;

  /* Have each proc write to its own file */
  snprintf (vtufilename, BUFSIZ, "%s_%04d.vtu", filename, mpirank);
  /* To be able to fseek in a file you cannot open in append mode.
   * so you need to open with "r+" and fseek to SEEK_END.
   */
  vtufile = fopen (vtufilename, "rb+");
  if (vtufile == NULL) {
    P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
    return -1;
  }
  retval = fseek (vtufile, 0L, SEEK_END);
  if (retval) {
    P4EST_LERRORF ("Could not fseek %s for output\n", vtufilename);
    fclose (vtufile);
    return -1;
  }

  char rank_tree_name[BUFSIZ]; rank_tree_name[0]='\0';
  if (write_rank && write_tree)
    sprintf(rank_tree_name, "proc_rank, tree_idx, ");
  else if (write_rank)
    sprintf(rank_tree_name, "proc_rank, ");
  else if (write_tree)
    sprintf(rank_tree_name, "tree_idx, ");

  /* Cell Data */
  fprintf (vtufile, "      <CellData");
  if (values != NULL)
    fprintf (vtufile, " Scalars=\"%s%s\"", rank_tree_name, list_name);
  fprintf (vtufile, ">\n");

#ifndef P4EST_VTK_ASCII
  if (write_rank || write_tree)
    locidx_data = P4EST_ALLOC (p4est_locidx_t, Ncells);
  else
    locidx_data = NULL;
#endif

  if (write_rank) {

    fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"proc_rank\""
             " format=\"%s\">\n", P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
#ifdef P4EST_VTK_ASCII
    fprintf (vtufile, "         ");
    for (il = 0, sk = 1; il < p4est->local_num_quadrants; ++il, ++sk) {
      fprintf (vtufile, " %d", mpirank);
      if (!(sk % 20) && il != (Ncells - 1))
        fprintf (vtufile, "\n         ");
    }
    if (ghost != NULL){
      p4est_locidx_t *proc_offset = ghost->proc_offsets;
      p4est_locidx_t r;
      for (r = 0; r<p4est->mpisize; ++r)
        for(il = proc_offset[r]; il<proc_offset[r+1]; ++il){
          fprintf (vtufile, " %d", r);
          if (!(sk % 20) && il != (Ncells - 1))
            fprintf (vtufile, "\n         ");
          ++sk;
        }
    }
    fprintf (vtufile, "\n");
#else
    for (il = 0; il < p4est->local_num_quadrants; ++il)
      locidx_data[il] = (p4est_locidx_t) mpirank;

    if (ghost != NULL){
      p4est_locidx_t *proc_offset = ghost->proc_offsets;
      p4est_locidx_t r;
      for (r = 0; r<p4est->mpisize; ++r)
        for(il = proc_offset[r]; il<proc_offset[r+1]; ++il)
          locidx_data[il+p4est->local_num_quadrants] = r;
      il += p4est->local_num_quadrants;
    }

    fprintf (vtufile, "          ");
    retval = my_p4est_vtk_write_binary (vtufile, (char *) locidx_data,
                                        sizeof (*locidx_data) * Ncells);
    fprintf (vtufile, "\n");
    if (retval) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error encoding types\n");
      fclose (vtufile);
      return -1;
    }
#endif
    fprintf (vtufile, "        </DataArray>\n");
    P4EST_ASSERT(il == Ncells);
  }
  if (write_tree) {
    fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"tree_idx\""
             " format=\"%s\">\n", P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
#ifdef P4EST_VTK_ASCII
    fprintf (vtufile, "         ");
    for (il = 0, sk = 1, jt = p4est->first_local_tree; jt <= p4est->last_local_tree; ++jt) {
      p4est_tree_t *tree = p4est_tree_array_index (p4est->trees, jt);
      int num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz, ++sk, ++il) {
        fprintf (vtufile, " %lld", (long long) jt);
        if (!(sk % 20) && il != (Ncells - 1))
          fprintf (vtufile, "\n         ");
      }
    }
    if (ghost != NULL){
      p4est_locidx_t *tree_offset = ghost->tree_offsets;
      p4est_locidx_t tr;
      for (tr = 0; tr<p4est->connectivity->num_trees; ++tr)
        for(il = tree_offset[tr]; il<tree_offset[tr+1]; ++il){
          fprintf (vtufile, " %d", tr);
          if (!(sk % 20) && il != (Ncells - 1))
            fprintf (vtufile, "\n         ");
          ++sk;
        }
    }
    fprintf (vtufile, "\n");
#else
    for (il = 0, jt = p4est->first_local_tree; jt <= p4est->last_local_tree; ++jt) {
      p4est_tree_t *tree = p4est_tree_array_index (p4est->trees, jt);
      p4est_locidx_t num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz, ++il) {
        locidx_data[il] = (p4est_locidx_t) jt;
      }
    }
    if (ghost != NULL){
      p4est_locidx_t *tree_offset = ghost->tree_offsets;
      p4est_locidx_t tr;
      for (tr = 0; tr<p4est->connectivity->num_trees; ++tr)
        for(il = tree_offset[tr]; il<tree_offset[tr+1]; ++il)
          locidx_data[il+p4est->local_num_quadrants] = tr;
      il += p4est->local_num_quadrants;
    }
    fprintf (vtufile, "          ");
    retval = my_p4est_vtk_write_binary (vtufile, (char *) locidx_data,
                                        sizeof (*locidx_data) * Ncells);
    fprintf (vtufile, "\n");
    if (retval) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error encoding types\n");
      fclose (vtufile);
      return -1;
    }
#endif
    fprintf (vtufile, "        </DataArray>\n");
    P4EST_ASSERT (il == Ncells);
  }

#ifndef P4EST_VTK_ASCII
  if (locidx_data != NULL) P4EST_FREE (locidx_data);
#endif

  int i;
#ifndef P4EST_VTK_ASCII
  float_data = num >0 ? P4EST_ALLOC (P4EST_VTK_FLOAT_TYPE, Ncells):NULL;
#endif
  for (i=0; i<num; ++i){
    /* write cell-center position data */
    fprintf (vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
             " format=\"%s\">\n",
             P4EST_VTK_FLOAT_NAME, scalar_names[i], P4EST_VTK_FORMAT_STRING);

#ifdef P4EST_VTK_ASCII
    for (il = 0; il < Ncells; ++il) {
#ifdef P4EST_VTK_DOUBLES
      fprintf (vtufile, "     %24.16e\n", values[i][il]);
#else
      fprintf (vtufile, "          %16.8e\n", values[i][il]);
#endif
    }
#else
    for (il = 0; il < Ncells; ++il) {
      float_data[il] = (P4EST_VTK_FLOAT_TYPE) values[i][il];
    }

    fprintf (vtufile, "          ");
    /* TODO: Don't allocate the full size of the array, only allocate
     * the chunk that will be passed to zlib and do this a chunk
     * at a time.
     */
    retval = my_p4est_vtk_write_binary (vtufile, (char *) float_data,
                                        sizeof (*float_data) * Ncells);
    fprintf (vtufile, "\n");
    if (retval) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error encoding points\n");
      fclose (vtufile);
      return -1;
    }
#endif
    fprintf (vtufile, "        </DataArray>\n");

    if (ferror (vtufile)) {
      P4EST_LERROR (P4EST_STRING "_vtk: Error writing cell scalar\n");
      fclose (vtufile);
      return -1;
    }
  }

  fprintf (vtufile, "      </CellData>\n");

  if (fclose (vtufile)) {
    P4EST_LERROR (P4EST_STRING "_vtk: Error closing cell scalar\n");
    return -1;
  }
  vtufile = NULL;
#ifndef P4EST_VTK_ASCII
  if (float_data != NULL) P4EST_FREE (float_data);
#endif

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    char                pvtufilename[BUFSIZ];
    FILE               *pvtufile;
    snprintf (pvtufilename, BUFSIZ, "%s.pvtu", filename);

    pvtufile = fopen (pvtufilename, "ab");
    if (!pvtufile) {
      P4EST_LERRORF ("Could not open %s for output\n", vtufilename);
      return -1;
    }

    fprintf (pvtufile, "    <PCellData Scalars=\"%s%s\">\n",rank_tree_name, list_name);

    if (write_rank) {
      fprintf (pvtufile, "      "
               "<PDataArray type=\"%s\" Name=\"proc_rank\" format=\"%s\"/>\n",
               P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
    }
    if (write_tree) {
      fprintf (pvtufile, "      "
               "<PDataArray type=\"%s\" Name=\"tree_idx\" format=\"%s\"/>\n",
               P4EST_VTK_LOCIDX, P4EST_VTK_FORMAT_STRING);
    }
    int i;
    for (i=0; i<num; ++i){
      fprintf (pvtufile, "      <PDataArray type=\"%s\" Name=\"%s\""
               " format=\"%s\"/>\n",
               P4EST_VTK_FLOAT_NAME, scalar_names[i], P4EST_VTK_FORMAT_STRING);

      if (ferror (pvtufile)) {
        P4EST_LERROR (P4EST_STRING
                      "_vtk: Error writing parallel point scalar\n");
        fclose (pvtufile);
        return -1;
      }
    }

    fprintf (pvtufile, "    </PCellData>\n");

    if (fclose (pvtufile)) {
      P4EST_LERROR (P4EST_STRING
                    "_vtk: Error closing parallel point scalar\n");
      return -1;
    }
  }

  return 0;
}

int
my_p4est_vtk_write_footer (p4est_t * p4est, const char *filename)
{
  char                vtufilename[BUFSIZ];
  int                 p;
  int                 procRank = p4est->mpirank;
  int                 numProcs = p4est->mpisize;
  FILE               *vtufile;



  /* Have each proc write to its own file */
  snprintf (vtufilename, BUFSIZ, "%s_%04d.vtu", filename, procRank);
  vtufile = fopen (vtufilename, "ab");
  if (vtufile == NULL) {
    P4EST_LERRORF ("Could not open %s for output!\n", vtufilename);
    return -1;
  }

  fprintf (vtufile, "    </Piece>\n");
  fprintf (vtufile, "  </UnstructuredGrid>\n");
  fprintf (vtufile, "</VTKFile>\n");

  if (ferror (vtufile)) {
    P4EST_LERROR ("p4est_vtk: Error writing footer\n");
    fclose (vtufile);
    return -1;
  }
  if (fclose (vtufile)) {
    P4EST_LERROR ("p4est_vtk: Error closing footer\n");
    return -1;
  }
  vtufile = NULL;

  /* Only have the root write to the parallel vtk file */
  if (procRank == 0) {


    char                visitfilename[BUFSIZ];
    char                pvtufilename[BUFSIZ];
    FILE               *pvtufile, *visitfile;

    /* Reopen paraview master file for writing bottom half */
    snprintf (pvtufilename, BUFSIZ, "%s.pvtu", filename);
    pvtufile = fopen (pvtufilename, "ab");
    if (!pvtufile) {

      P4EST_LERRORF ("Could not open %s for output!\n", vtufilename);

      return -1;
    }



    /* Create a master file for visualization in Visit */
    snprintf (visitfilename, BUFSIZ, "%s.visit", filename);
    visitfile = fopen (visitfilename, "wb");
    if (!visitfile) {
      P4EST_LERRORF ("Could not open %s for output\n", visitfilename);
      fclose (pvtufile);
      return -1;
    }


    fprintf (visitfile, "!NBLOCKS %d\n", numProcs);

    /* Write data about the parallel pieces into both files */
    for (p = 0; p < numProcs; ++p) {
      fprintf (pvtufile,
               "    <Piece Source=\"%s_%04d.vtu\"/>\n", filename, p);
      fprintf (visitfile, "%s_%04d.vtu\n", filename, p);
    }
    fprintf (pvtufile, "  </PUnstructuredGrid>\n");
    fprintf (pvtufile, "</VTKFile>\n");

    /* Close paraview master file */
    if (ferror (pvtufile)) {
      P4EST_LERROR ("p4est_vtk: Error writing parallel footer\n");
      fclose (visitfile);
      fclose (pvtufile);
      return -1;
    }


    if (fclose (pvtufile)) {
      fclose (visitfile);
      P4EST_LERROR ("p4est_vtk: Error closing parallel footer\n");
      return -1;
    }


    /* Close visit master file */
    if (ferror (visitfile)) {
      P4EST_LERROR ("p4est_vtk: Error writing parallel footer\n");
      fclose (visitfile);
      return -1;
    }
    if (fclose (visitfile)) {
      P4EST_LERROR ("p4est_vtk: Error closing parallel footer\n");
      return -1;
    }
  }


  return 0;
}

void my_p4est_vtk_write_ghost_layer(p4est_t *p4est, p4est_ghost_t *ghost)
{
  char csvname[1024], vtkname[1024];
  sprintf(csvname, "ghost_layer_%04d.csv", p4est->mpirank);
  sprintf(vtkname, "ghost_layer_%04d.vtk", p4est->mpirank);

  FILE *csv = fopen(csvname, "w");
  FILE *vtk = fopen(vtkname, "w");

  double *x, *y;
  int num_quad = ghost->ghosts.elem_count;
  int xysize = P4EST_CHILDREN*num_quad;
  x = P4EST_ALLOC(double, xysize);
  y = P4EST_ALLOC(double, xysize);

  int i;
  short j, xj, yj;
  for (i=0; i<num_quad; i++)
  {
    p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, i);
    fprintf(csv, "%d,",q->p.piggy3.local_num);

    p4est_topidx_t tree_id = q->p.piggy3.which_tree;
    p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];

    double xq = (double)(q->x)/(double)(P4EST_ROOT_LEN) + tree_xmin;
    double yq = (double)(q->y)/(double)(P4EST_ROOT_LEN) + tree_ymin;
    double hq = (double)(P4EST_QUADRANT_LEN(q->level))/(double)(P4EST_ROOT_LEN);

    for (xj=0; xj<2; ++xj)
      for (yj=0; yj<2; ++yj){
        x[P4EST_CHILDREN*i+2*yj+xj] = xq + hq*xj;
        y[P4EST_CHILDREN*i+2*yj+xj] = yq + hq*yj;
      }
  }
  fclose(csv);

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Quadtree Mesh \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");
  fprintf(vtk, "POINTS %d double \n", xysize);
  for (i=0; i<xysize; i++)
    fprintf(vtk, "%lf %lf 0.0\n", x[i], y[i]);
  fflush(vtk);

  fprintf(vtk, "CELLS %d %d \n", num_quad, (1+P4EST_CHILDREN)*num_quad);
  for (i=0; i<num_quad; ++i)
  {
    fprintf(vtk, "%d ", P4EST_CHILDREN);
    for (j=0; j<P4EST_CHILDREN; ++j)
      fprintf(vtk, "%d ", P4EST_CHILDREN*i+j);
    fprintf(vtk,"\n");
  }
  fflush(vtk);

  fprintf(vtk, "CELL_TYPES %d\n", num_quad);
  for (i=0; i<num_quad; ++i)
    fprintf(vtk, "%d\n",P4EST_VTK_CELL_TYPE);
  fclose(vtk);

  P4EST_FREE(x);
  P4EST_FREE(y);
}
