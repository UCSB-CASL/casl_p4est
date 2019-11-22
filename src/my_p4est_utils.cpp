#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#include "my_p8est_tools.h"
#include <p8est_connectivity.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include "cube3.h"
#else
#include "my_p4est_utils.h"
#include "my_p4est_tools.h"
#include <p4est_connectivity.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include "cube2.h"
#endif

#include "mpi.h"
#include <vector>
#include <set>
#include <sstream>
#include <petsclog.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <stack>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_TINY_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

std::vector<InterpolatingFunctionLogEntry> InterpolatingFunctionLogger::entries;

WallBC2D::~WallBC2D() {};
WallBC3D::~WallBC3D() {};

bool index_of_node(const p4est_quadrant_t *n, p4est_nodes_t* nodes, p4est_locidx_t& idx)
{
#ifdef P4EST_DEBUG
  int clamped = 1;
#endif
  P4EST_ASSERT(p4est_quadrant_is_node(n, clamped));
  unsigned int idx_l, idx_u, idx_m;
  const p4est_indep_t *node_l, *node_u, *node_m;
  // check if the candidate can be in the locally owned nodes first, or in the ghost ones
  idx_l   = 0;
  idx_u   = nodes->num_owned_indeps-1;
  node_l  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_l);
  node_u  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_u);
  if((p4est_quadrant_compare_piggy(node_l, n) > 0) || (p4est_quadrant_compare_piggy(node_u, n) < 0))
    goto lookup_in_ghost_nodes;
  while((p4est_quadrant_compare_piggy(node_l, n) <= 0) && (p4est_quadrant_compare_piggy(node_u, n) >= 0))
  {
    if(!p4est_quadrant_compare_piggy(node_l, n))
    {
      idx = idx_l;
      return true;
    }
    if(!p4est_quadrant_compare_piggy(node_u, n))
    {
      idx = idx_u;
      return true;
    }
    if(idx_u-idx_l == 1)
      break;
    idx_m   = (idx_l + idx_u)/2;
    node_m  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_m);
    P4EST_ASSERT((p4est_quadrant_compare_piggy(node_l, node_m) <= 0) && (p4est_quadrant_compare_piggy(node_u, node_m) >= 0));
    if(p4est_quadrant_compare_piggy(node_m, n) <0)
    {
      idx_l   = idx_m;
      node_l  = node_m;
    }
    else if (p4est_quadrant_compare_piggy(node_m, n) >0)
    {
      idx_u   = idx_m;
      node_u  = node_m;
    }
    else
    {
      P4EST_ASSERT(!p4est_quadrant_compare_piggy(node_m, n));
      idx = idx_m;
      return true;
    }
  }
  return false;
lookup_in_ghost_nodes:
  P4EST_ASSERT((p4est_quadrant_compare_piggy(node_l, n) > 0) || (p4est_quadrant_compare_piggy(node_u, n) < 0));
  idx_l   = nodes->num_owned_indeps;
  idx_u   = nodes->indep_nodes.elem_count-1;
  if(idx_l <= idx_u) // do this only if there are ghost nodes!
  {
    node_l  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_l);
    node_u  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_u);
    while((p4est_quadrant_compare_piggy(node_l, n) <= 0) && (p4est_quadrant_compare_piggy(node_u, n) >= 0))
    {
      if(!p4est_quadrant_compare_piggy(node_l, n))
      {
        idx = idx_l;
        return true;
      }
      if(!p4est_quadrant_compare_piggy(node_u, n))
      {
        idx = idx_u;
        return true;
      }
      if(idx_u-idx_l == 1)
        break;
      idx_m   = (idx_l + idx_u)/2;
      node_m  = (const p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx_m);
      P4EST_ASSERT((p4est_quadrant_compare_piggy(node_l, node_m) <= 0) && (p4est_quadrant_compare_piggy(node_u, node_m) >= 0));
      if(p4est_quadrant_compare_piggy(node_m, n) <0)
      {
        idx_l   = idx_m;
        node_l  = node_m;
      }
      else if (p4est_quadrant_compare_piggy(node_m, n) >0)
      {
        idx_u   = idx_m;
        node_u  = node_m;
      }
      else
      {
        P4EST_ASSERT(!p4est_quadrant_compare_piggy(node_m, n));
        idx = idx_m;
        return true;
      }
    }
  }
  return false;
}

void linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global, double* results, unsigned int n_results)
{
  P4EST_ASSERT(n_results > 0);
  PetscErrorCode ierr;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN-1];

  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

  /* shift xyz to [0,1] */
  double x = (xyz_global[0] - tree_xmin)/(tree_xmax-tree_xmin);
  double y = (xyz_global[1] - tree_ymin)/(tree_ymax-tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin)/(tree_zmax-tree_zmin);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double xmin = (double)quad.x / (double)(P4EST_ROOT_LEN);
  double ymin = (double)quad.y / (double)(P4EST_ROOT_LEN);
#ifdef P4_TO_P8
  double zmin = (double)quad.z / (double)(P4EST_ROOT_LEN);
#endif

  double d_m00 = x - xmin;
  double d_p00 = qh - d_m00;
  double d_0m0 = y - ymin;
  double d_0p0 = qh - d_0m0;
#ifdef P4_TO_P8
  double d_00m = z - zmin;
  double d_00p = qh - d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  for (unsigned int k = 0; k < n_results; ++k)
  {
    results[k] = 0.0;
    for (short j = 0; j<P4EST_CHILDREN; j++)
      results[k] += + F[P4EST_CHILDREN*k+j]*w_xyz[j];
#ifdef P4_TO_P8
    results[k] /= qh*qh*qh;
#else
    results[k] /= qh*qh;
#endif
  }

  ierr = PetscLogFlops(39); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results)
{
  P4EST_ASSERT(n_results > 0);
  PetscErrorCode ierr;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN-1];

  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

  double x = (xyz_global[0] - tree_xmin)/(tree_xmax-tree_xmin);
  double y = (xyz_global[1] - tree_ymin)/(tree_ymax-tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin)/(tree_zmax-tree_zmin);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double xmin = quad_x_fr_i(&quad);
  double ymin = quad_y_fr_j(&quad);
#ifdef P4_TO_P8
  double zmin = quad_z_fr_k(&quad);
#endif

  x = (x-xmin) / qh;
  y = (y-ymin) / qh;
#ifdef P4_TO_P8
  z = (z-zmin) / qh;
#endif

  double d_m00 = x;
  double d_p00 = 1-x;
  double d_0m0 = y;
  double d_0p0 = 1-y;
#ifdef P4_TO_P8
  double d_00m = z;
  double d_00p = 1-z;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  double fdd[P4EST_DIM];
  double sx = (tree_xmax-tree_xmin)*qh;
  double sy = (tree_ymax-tree_ymin)*qh;
#ifdef P4_TO_P8
  double sz = (tree_zmax-tree_zmin)*qh;
#endif
  for (unsigned int k = 0; k < n_results; ++k) {
    results[k] = 0.0;
    for (short j = 0; j < P4EST_CHILDREN; ++j) {
      for (short i = 0; i<P4EST_DIM; i++)
        fdd[i] = ((j == 0)? Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] : MINMOD(fdd[i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + i]));
      results[k] += F[k*P4EST_CHILDREN+j]*w_xyz[j];
    }
#ifdef P4_TO_P8
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1] + sz*sz*d_00p*d_00m*fdd[2]);
#else
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1]);
#endif
  }

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_continuous_v1_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results)
{
  PetscErrorCode ierr;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN-1];

  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

  double x = (xyz_global[0] - tree_xmin)/(tree_xmax-tree_xmin);
  double y = (xyz_global[1] - tree_ymin)/(tree_ymax-tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin)/(tree_zmax-tree_zmin);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double xmin = quad_x_fr_i(&quad);
  double ymin = quad_y_fr_j(&quad);
#ifdef P4_TO_P8
  double zmin = quad_z_fr_k(&quad);
#endif

  x = (x-xmin) / qh;
  y = (y-ymin) / qh;
#ifdef P4_TO_P8
  z = (z-zmin) / qh;
#endif

  double d_m00 = x;
  double d_p00 = 1-x;
  double d_0m0 = y;
  double d_0p0 = 1-y;
#ifdef P4_TO_P8
  double d_00m = z;
  double d_00p = 1-z;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  // First alternative scheme: first, minmod on every edge, then weight-average
  double fdd[P4EST_DIM];
  unsigned int i, jm, jp;
  const double sx = (tree_xmax-tree_xmin)*qh;
  const double sy = (tree_ymax-tree_ymin)*qh;
#ifdef P4_TO_P8
  const double sz = (tree_zmax-tree_zmin)*qh;
#endif
  for (unsigned int k = 0; k < n_results; ++k) {
    // set your fdd
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      fdd[dir] = 0.0;

    i = 0;
    jm = 0; jp = 1; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 2; jp = 3; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 5; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 6; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
#endif

    i = 1;
    jm = 0; jp = 2; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 1; jp = 3; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 6; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 5; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
#endif

#ifdef P4_TO_P8
    i = 2;
    jm = 0; jp = 4; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 1; jp = 5; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 2; jp = 6; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
    jm = 3; jp = 7; fdd[i] += MINMOD(Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i], Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i])*(w_xyz[jm]+w_xyz[jp]);
#endif

    results[k] = 0.0;
    for (unsigned char j = 0; j < P4EST_CHILDREN; ++j)
      results[k] += F[k*P4EST_CHILDREN+j]*w_xyz[j];

#ifdef P4_TO_P8
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1] + sz*sz*d_00p*d_00m*fdd[2]);
#else
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1]);
#endif
  }

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}

void quadratic_non_oscillatory_continuous_v2_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results)
{
  PetscErrorCode ierr;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN-1];

  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

  double x = (xyz_global[0] - tree_xmin)/(tree_xmax-tree_xmin);
  double y = (xyz_global[1] - tree_ymin)/(tree_ymax-tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin)/(tree_zmax-tree_zmin);
#endif

  double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double xmin = quad_x_fr_i(&quad);
  double ymin = quad_y_fr_j(&quad);
#ifdef P4_TO_P8
  double zmin = quad_z_fr_k(&quad);
#endif

  x = (x-xmin) / qh;
  y = (y-ymin) / qh;
#ifdef P4_TO_P8
  z = (z-zmin) / qh;
#endif

  double d_m00 = x;
  double d_p00 = 1-x;
  double d_0m0 = y;
  double d_0p0 = 1-y;
#ifdef P4_TO_P8
  double d_00m = z;
  double d_00p = 1-z;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  // Second alternative scheme: first, weight-average in perpendicular plane, then minmod
  double fdd[P4EST_DIM];
  unsigned int i, jm, jp;
  const double sx = (tree_xmax-tree_xmin)*qh;
  const double sy = (tree_ymax-tree_ymin)*qh;
#ifdef P4_TO_P8
  const double sz = (tree_zmax-tree_zmin)*qh;
#endif
  double fdd_m, fdd_p;
  for (unsigned int k = 0; k < n_results; ++k) {
    // set your fdd
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      fdd[dir] = 0.0;

    i = 0;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 1; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 2; jp = 3; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 5; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 6; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
#endif
    fdd[i] = MINMOD(fdd_m, fdd_p);

    i = 1;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 2; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 1; jp = 3; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
    jm = 4; jp = 6; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 5; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
#endif
    fdd[i] = MINMOD(fdd_m, fdd_p);

#ifdef P4_TO_P8
    i = 2;
    fdd_m = 0;
    fdd_p = 0;
    jm = 0; jp = 4; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 1; jp = 5; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 2; jp = 6; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    jm = 3; jp = 7; fdd_m += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jm*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fdd[k*P4EST_CHILDREN*P4EST_DIM+jp*P4EST_DIM + i]*(w_xyz[jm]+w_xyz[jp]);
    fdd[i] = MINMOD(fdd_m, fdd_p);
#endif

    results[k] = 0.0;
    for (unsigned char j = 0; j < P4EST_CHILDREN; ++j)
      results[k] += F[k*P4EST_CHILDREN+j]*w_xyz[j];

#ifdef P4_TO_P8
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1] + sz*sz*d_00p*d_00m*fdd[2]);
#else
    results[k] -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1]);
#endif


  }
  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return;
}


void quadratic_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results)
{
  P4EST_ASSERT(n_results > 0);
  PetscErrorCode ierr;

  p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + 0];
  p4est_topidx_t v_pp = p4est->connectivity->tree_to_vertex[(p4est->trees->elem_count-1)*P4EST_CHILDREN + P4EST_CHILDREN-1];
  p4est_topidx_t v_m  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];
  p4est_topidx_t v_p  = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + P4EST_CHILDREN-1];

  const double* domain_xyz_min  = (p4est->connectivity->vertices+3*v_mm);
  const double* domain_xyz_max  = (p4est->connectivity->vertices+3*v_pp);
  const double* tree_xyz_min    = (p4est->connectivity->vertices+3*v_m);
  const double* tree_xyz_max    = (p4est->connectivity->vertices+3*v_p);

  const double qh   = (double)P4EST_QUADRANT_LEN(quad.level) / (double)(P4EST_ROOT_LEN);
  double qxyz_min[P4EST_DIM] = {DIM(quad_x_fr_i(&quad), quad_y_fr_j(&quad), quad_z_fr_k(&quad))};

  double xyz[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    xyz[dir] = xyz_global[dir];
    if(is_periodic(p4est, dir))
      while (fabs(xyz[dir] - (tree_xyz_min[dir] + (tree_xyz_max[dir]-tree_xyz_min[dir])*qxyz_min[dir])) >= (0.5+((double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL))/((double) P4EST_ROOT_LEN))*(domain_xyz_max[dir] - domain_xyz_min[dir]))
        xyz[dir] = xyz[dir] + ((xyz[dir] > (tree_xyz_min[dir] + (tree_xyz_max[dir]-tree_xyz_min[dir])*qxyz_min[dir]))? -1.0: +1.0)*(domain_xyz_max[dir]-domain_xyz_min[dir]);
    xyz[dir] = (xyz[dir] - tree_xyz_min[dir])/(tree_xyz_max[dir] - tree_xyz_min[dir]);
  }

  P4EST_ASSERT(ANDD(xyz[0]>=qxyz_min[0]-qh/10 && xyz[0]<=qxyz_min[0]+qh+qh/10, xyz[1]>=qxyz_min[1]-qh/10 && xyz[1]<=qxyz_min[1]+qh+qh/10, xyz[2]>=qxyz_min[2]-qh/10 && xyz[2]<=qxyz_min[2]+qh+qh/10));

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz[dir] = (xyz[dir]-qxyz_min[dir])/qh;

  double d_m00 = xyz[0];
  double d_p00 = 1-xyz[0];
  double d_0m0 = xyz[1];
  double d_0p0 = 1-xyz[1];
#ifdef P4_TO_P8
  double d_00m = xyz[2];
  double d_00p = 1-xyz[2];
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif


  double fdd[P4EST_DIM];
  const double sxyz[P4EST_DIM] = {DIM((tree_xyz_max[0] - tree_xyz_min[0])*qh, (tree_xyz_max[1] - tree_xyz_min[1])*qh, (tree_xyz_max[2] - tree_xyz_min[2])*qh)};
  for (unsigned int k = 0; k < n_results; ++k)
  {
    results[k] = 0.0;
    for (short j=0; j<P4EST_CHILDREN; j++)
    {
      for (short i = 0; i<P4EST_DIM; i++)
        fdd[i] = ((j == 0)? 0.0 : fdd[i]) +Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + i] * w_xyz[j];
      results[k] += F[k*P4EST_CHILDREN+j]*w_xyz[j];
    }
    results[k] -= 0.5*SUMD(sxyz[0]*sxyz[0]*d_p00*d_m00*fdd[0], sxyz[1]*sxyz[1]*d_0p0*d_0m0*fdd[1], sxyz[2]*sxyz[2]*d_00p*d_00m*fdd[2]);
  }

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
}

void write_comm_stats(const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes, const char *partition_name, const char *topology_name, const char *neighbors_name)
{
  FILE *file;
  PetscErrorCode ierr;

  /* save partition information */
  if (partition_name) {
    ierr = PetscFOpen(p4est->mpicomm, partition_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  p4est_gloidx_t num_nodes = 0;
  for (int r =0; r<p4est->mpisize; r++)
    num_nodes += nodes->global_owned_indeps[r];

  PetscFPrintf(p4est->mpicomm, file, "%% global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
  PetscFPrintf(p4est->mpicomm, file, "%% mpi_rank | local_node_size | local_quad_size | ghost_node_size | ghost_quad_size\n");
  PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d, %7d, %7d, %5d, %5d\n",
                           p4est->mpirank, nodes->num_owned_indeps, p4est->local_num_quadrants, nodes->indep_nodes.elem_count-nodes->num_owned_indeps, ghost->ghosts.elem_count);
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (partition_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }

  /* save recv info based on the ghost nodes */
  if (topology_name){
    ierr = PetscFOpen(p4est->mpicomm, topology_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  PetscFPrintf(p4est->mpicomm, file, "%% Topology of ghost nodes based on how many ghost nodes belongs to a certain processor \n");
  PetscFPrintf(p4est->mpicomm, file, "%% this_rank | ghost_rank | ghost_node_size \n");
  std::vector<p4est_locidx_t> ghost_nodes(p4est->mpisize, 0);
  std::set<int> proc_neighbors;
  for (size_t i=0; i<nodes->indep_nodes.elem_count - nodes->num_owned_indeps; i++){
    int r = nodes->nonlocal_ranks[i];
    proc_neighbors.insert(r);
    ghost_nodes[r]++;
  }
  for (std::set<int>::const_iterator it = proc_neighbors.begin(); it != proc_neighbors.end(); ++it){
    int r = *it;
    PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d %4d %6d\n", p4est->mpirank, r, ghost_nodes[r]);
  }
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (topology_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }

  /* save recv info based on the ghost nodes */
  if (neighbors_name){
    ierr = PetscFOpen(p4est->mpicomm, neighbors_name, "w", &file); CHKERRXX(ierr);
  } else {
    file = stdout;
  }

  PetscFPrintf(p4est->mpicomm, file, "%% number of neighboring processors \n");
  PetscFPrintf(p4est->mpicomm, file, "%% this_rank | number_ghost_rank \n");
  PetscSynchronizedFPrintf(p4est->mpicomm, file, "%4d %4d\n", p4est->mpirank, proc_neighbors.size());
  PetscSynchronizedFlush(p4est->mpicomm, stdout);

  if (neighbors_name){
    ierr = PetscFClose(p4est->mpicomm, file); CHKERRXX(ierr);
  }
}

p4est_bool_t nodes_are_equal(int mpi_size, p4est_nodes_t* nodes_1, p4est_nodes_t* nodes_2)
{
  if(nodes_1 == nodes_2)
    return P4EST_TRUE;
  const p4est_indep_t *node_1, *node_2;
  p4est_bool_t result = (nodes_1->indep_nodes.elem_count == nodes_2->indep_nodes.elem_count);
  result = result && (nodes_1->num_local_quadrants  == nodes_2->num_local_quadrants);
  result = result && (nodes_1->num_owned_indeps     == nodes_2->num_owned_indeps);
  result = result && (nodes_1->num_owned_shared     == nodes_2->num_owned_shared);
  result = result && (nodes_1->offset_owned_indeps == 0) && (nodes_2->offset_owned_indeps == 0);
  if(!result)
    goto return_time;
  for (int r = 0; r < mpi_size; ++r) {
    result = result && (nodes_1->global_owned_indeps[r] == nodes_2->global_owned_indeps[r]);
    if(!result)
      goto return_time;
  }
  // compare the raw nodes, one by one, first
  for (size_t k = 0; k < nodes_1->indep_nodes.elem_count; ++k) {
    node_1 = (const p4est_indep_t*) sc_array_index(&nodes_1->indep_nodes, k);
    node_2 = (const p4est_indep_t*) sc_array_index(&nodes_2->indep_nodes, k);
    result = result && (node_1->level == node_2->level);
    result = result && (node_1->x == node_2->x);
    result = result && (node_1->y == node_2->y);
#ifdef P4_TO_P8
    result = result && (node_1->z == node_2->z);
#endif
    result = result && (node_1->pad8 == node_2->pad8);
    // all nodes must have their p.piggy3 used, local or ghost...
    result = result && (node_1->p.piggy3.local_num  == node_2->p.piggy3.local_num);
    result = result && (node_1->p.piggy3.which_tree == node_2->p.piggy3.which_tree);
    if(k > ((size_t) nodes_1->num_owned_indeps))
      result = result && (nodes_1->nonlocal_ranks[k-nodes_1->num_owned_indeps] == nodes_2->nonlocal_ranks[k-nodes_2->num_owned_indeps]);
    if(!result)
      goto return_time;
  }
  // check that the local indices of points associated with local quadrants are equal
  for (p4est_locidx_t k = 0; k < nodes_1->num_local_quadrants; ++k) {
    for (short j = 0; j < P4EST_CHILDREN; ++j) {
      result = result && (nodes_1->local_nodes[P4EST_CHILDREN*k + j] == nodes_2->local_nodes[P4EST_CHILDREN*k + j]);
      if(!result)
        goto return_time;
    }
  }
return_time:
  return result;
}

p4est_bool_t ghosts_are_equal(p4est_ghost_t* ghost_1, p4est_ghost_t* ghost_2)
{
  if(ghost_1 == ghost_2)
    return P4EST_TRUE;

  const p4est_quadrant_t *quad_1, *quad_2;
  int mpisize = ghost_1->mpisize;
  p4est_bool_t result = (ghost_2->mpisize == mpisize);
  result = result && (ghost_1->ghosts.elem_count == ghost_2->ghosts.elem_count);
  result = result && (ghost_1->num_trees == ghost_2->num_trees);
  result = result && (ghost_1->btype == ghost_2->btype);
  if(!result)
    goto return_time;
  for (size_t k = 0; k < ghost_1->ghosts.elem_count; ++k) {
    quad_1 = p4est_quadrant_array_index(&ghost_1->ghosts, k);
    quad_2 = p4est_quadrant_array_index(&ghost_2->ghosts, k);
    result = result && p4est_quadrant_is_equal(quad_1, quad_2);
    result = result && (quad_1->p.piggy3.local_num == quad_2->p.piggy3.local_num);
    result = result && (quad_1->p.piggy3.which_tree == quad_2->p.piggy3.which_tree);
    if(!result)
      goto return_time;
  }
  for (int r = 0; r < mpisize+1; ++r) {
    result = result && (ghost_1->proc_offsets[r] == ghost_2->proc_offsets[r]);
    if(!result)
      goto return_time;
  }
  for (p4est_topidx_t tree_idx = 0; tree_idx < ghost_1->num_trees+1; ++tree_idx) {
    result = result && (ghost_1->tree_offsets[tree_idx] == ghost_2->tree_offsets[tree_idx]);
    if(!result)
       goto return_time;
  }
return_time:
  return result;
}

PetscErrorCode VecGetLocalAndGhostSizes(Vec& v, PetscInt& local_size, PetscInt& ghosted_size)
{
  PetscErrorCode ierr = 0;
  Vec v_loc;
  ierr = VecGetLocalSize(v, &local_size); CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(v, &v_loc); CHKERRQ(ierr);
  ierr = VecGetSize(v_loc, &ghosted_size); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(v, &v_loc); CHKERRQ(ierr);
  return ierr;
}

bool vectorIsWellSetForNodes(Vec& v, const p4est_nodes_t* nodes, const MPI_Comm& mpicomm, const unsigned int& blocksize)
{
  P4EST_ASSERT(v!=NULL);
  P4EST_ASSERT(blocksize>0);
  PetscInt local_size, ghosted_size;
  VecGetLocalAndGhostSizes(v, local_size, ghosted_size);
  int my_test = (local_size==((PetscInt)(blocksize*nodes->num_owned_indeps)) && ghosted_size!=((PetscInt)(blocksize*nodes->indep_nodes.elem_count)))?1:0;
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_test, 1, MPI_INT, MPI_LAND, mpicomm); SC_CHECK_MPI(mpiret);
  return my_test;
}

PetscErrorCode VecCreateGhostNodesBlock(const p4est_t *p4est, const p4est_nodes_t *nodes, const PetscInt & block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;
  P4EST_ASSERT(block_size > 0);

  std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for (size_t i = 0; i<ghost_nodes.size(); ++i)
  {
    /*
    p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+num_local);
     * [RAPHAEL:] substituted this latter line of code by the following to enforce and ensure const attribute in 'nodes' argument...
     */
    SC_ASSERT(((size_t) (i+num_local))<nodes->indep_nodes.elem_count);
    const p4est_indep_t* ni = (const p4est_indep_t*) (nodes->indep_nodes.array + (((size_t) (i+num_local))*nodes->indep_nodes.elem_size));

    ghost_nodes[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
  }

  if(block_size > 1){
    ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, num_local*block_size, num_global*block_size,
                               ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  } else{
    ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                          ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecGhostCopy(Vec src, Vec dst)
{
  PetscErrorCode ierr;

  Vec src_l, dst_l;
  ierr = VecGhostGetLocalForm(src, &src_l); CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(dst, &dst_l); CHKERRQ(ierr);
  ierr = VecCopy(src_l, dst_l); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(src, &src_l); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(dst, &dst_l); CHKERRQ(ierr);

  return 0;
}

PetscErrorCode VecGhostSet(Vec x, double v)
{
  PetscErrorCode ierr;
  Vec x_l;

  ierr = VecGhostGetLocalForm(x, &x_l); CHKERRQ(ierr);
  ierr = VecSet(x, v); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(x, &x_l); CHKERRQ(ierr);

  return 0;
}

PetscErrorCode VecCreateGhostCellsBlock(const p4est_t *p4est, const p4est_ghost_t *ghost, const PetscInt & block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;
  P4EST_ASSERT(block_size > 0);

  std::vector<PetscInt> ghost_cells(ghost->ghosts.elem_count, 0);
  PetscInt num_global = p4est->global_num_quadrants;

  for (int r = 0; r<p4est->mpisize; ++r)
    for (p4est_locidx_t q = ghost->proc_offsets[r]; q < ghost->proc_offsets[r+1]; ++q)
    {
      /*
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
       * [RAPHAEL:] substituted this latter line of code by the following to enforce and ensure const attribute in 'nodes' argument...
       */
      SC_ASSERT(((size_t) q)<ghost->ghosts.elem_count);
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*) (ghost->ghosts.array + ((size_t) q)*ghost->ghosts.elem_size);

      ghost_cells[q] = (PetscInt)quad->p.piggy3.local_num + (PetscInt)p4est->global_first_quadrant[r];
    }

  if(block_size > 1){
    ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, num_local*block_size, num_global*block_size,
                             ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  } else {
    ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                          ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateCellsBlockNoGhost(const p4est_t *p4est, const PetscInt &block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;
  P4EST_ASSERT(block_size > 0);

  PetscInt num_global = p4est->global_num_quadrants;

  ierr = VecCreateMPI(p4est->mpicomm, num_local*block_size, num_global*block_size, v); CHKERRQ(ierr);
  if(block_size > 1){
    ierr = VecSetBlockSize(*v, block_size); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecScatterAllToSomeCreate(MPI_Comm comm, Vec origin_loc, Vec destination, const PetscInt &ndest_glo_idx, const PetscInt *dest_glo_idx, VecScatter *ctx)
{
  PetscErrorCode ierr;
  IS is_from, is_to;
  ierr    = ISCreateGeneral(comm, ndest_glo_idx, dest_glo_idx, PETSC_USE_POINTER, &is_to);    CHKERRQ(ierr);
  ierr    = ISCreateStride(comm, ndest_glo_idx, 0, 1, &is_from);                              CHKERRQ(ierr);
  ierr    = VecScatterCreate(origin_loc, is_from, destination, is_to, ctx);                   CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode VecScatterCreateChangeLayout(MPI_Comm comm, Vec from, Vec to, VecScatter *ctx)
{
  PetscErrorCode ierr = 0;
#ifdef CASL_THROWS
  PetscInt size_from, size_to;
  ierr = VecGetSize(from, &size_from); CHKERRXX(ierr);
  ierr = VecGetSize(to, &size_to); CHKERRXX(ierr);
  if (size_from != size_to)
    throw std::invalid_argument("[ERROR]: Change layout is only supported for vectors with the same global size");
#endif

  IS is_from, is_to;

  ISLocalToGlobalMapping l2g;
  ierr = VecGetLocalToGlobalMapping(to, &l2g); CHKERRXX(ierr);

  const PetscInt *idx;
  PetscInt l2g_size;
  ierr = ISLocalToGlobalMappingGetIndices(l2g, &idx); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingGetSize(l2g, &l2g_size); CHKERRXX(ierr);

  ierr = ISCreateStride(comm, l2g_size, 0, 1, &is_to); CHKERRXX(ierr);
  ierr = ISCreateGeneral(comm, l2g_size, idx, PETSC_USE_POINTER, &is_from); CHKERRXX(ierr);

  Vec to_l;
  ierr = VecGhostGetLocalForm(to, &to_l); CHKERRXX(ierr);
  ierr = VecScatterCreate(from, is_from, to_l, is_to, ctx); CHKERRXX(ierr);

  ierr = ISDestroy(is_from); CHKERRXX(ierr);
  ierr = ISDestroy(is_to); CHKERRXX(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(l2g, &idx); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l); CHKERRXX(ierr);

  return ierr;
}

PetscErrorCode VecGhostChangeLayoutBegin(VecScatter ctx, Vec from, Vec to)
{
  PetscErrorCode ierr;
  Vec to_l;

  ierr = VecGhostGetLocalForm(to, &to_l);
  ierr = VecScatterBegin(ctx, from, to_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l);

  return ierr;
}

PetscErrorCode VecGhostChangeLayoutEnd(VecScatter ctx, Vec from, Vec to)
{
  PetscErrorCode ierr;
  Vec to_l;

  ierr = VecGhostGetLocalForm(to, &to_l);
  ierr = VecScatterEnd(ctx, from, to_l, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(to, &to_l);

  return ierr;
}

bool is_folder(const char* path)
{
  struct stat info;
  if(stat(path, &info)!= 0 )
  {
#ifdef CASL_THROWS
    char error_message[1024];
    sprintf(error_message, "is_folder: could not access %s", path);
    throw std::runtime_error(error_message);
#else
    return false;
#endif
  }
  return (info.st_mode & S_IFDIR);
}

bool file_exists(const char* path)
{
  struct stat info;
  return ((stat(path, &info)== 0) && (info.st_mode & S_IFREG));
}


int create_directory(const char* path, int mpi_rank, MPI_Comm comm)
{
  int return_ = 1;
  if(mpi_rank == 0)
  {
    struct stat info;
    if((stat(path, &info) == 0) &&  (info.st_mode & S_IFDIR)) // if it already exists, no need to create it...
      return_ = 0;
    else
    {
      char tmp[PATH_MAX];
      snprintf(tmp, sizeof(tmp), "%s", path);
      size_t len = strlen(tmp);
      if(tmp[len-1] == '/')
        tmp[len-1] = 0;
      for (char* p = tmp+1; *p; p++){
        if(*p == '/'){
          *p = 0;
          if((stat(tmp, &info) == 0) &&  (info.st_mode & S_IFDIR)) // if it already exists, no need to create it...
            return_ = 0;
          else
            return_ = mkdir(tmp, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH); // permission = 755 like a regular mkdir in terminal
          *p = '/';
          if(return_)
            break;
        }
      }
      if(return_ == 0) // successfull up to here
        return_ = mkdir(path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH); // permission = 755 like a regular mkdir in terminal
    }
  }
  int mpiret = MPI_Bcast(&return_, 1, MPI_INT, 0, comm); SC_CHECK_MPI(mpiret);
  return return_;
}

int  get_subdirectories_in(const char* root_path, std::vector<std::string>& subdirectories)
{
  if(!is_folder(root_path))
    return 1;

  subdirectories.resize(0);

  DIR *dir = opendir(root_path);
  struct dirent *entry = readdir(dir);
  while (entry != NULL)
  {
    if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
      subdirectories.push_back(entry->d_name);
    entry = readdir(dir);
  }

  closedir(dir);

  return 0;
}

int delete_directory(const char* root_path, int mpi_rank, MPI_Comm comm, bool non_collective)
{
  if(!is_folder(root_path))
  {
    char error_message[1024];
    sprintf(error_message, "delete_directory: path %s is NOT a directory...", root_path);
    throw std::invalid_argument(error_message);
  }

  int return_ = 1;
  if(mpi_rank == 0)
  {
    std::vector<std::string> subdirectories; subdirectories.resize(0);
    std::vector<std::string> reg_files; reg_files.resize(0);

    DIR *dir = opendir(root_path);
    struct dirent *entry = readdir(dir);
    while (entry != NULL)
    {
      if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
      {
        if(entry->d_type == DT_DIR)
          subdirectories.push_back(entry->d_name);
        else if (entry->d_type == DT_REG)
          reg_files.push_back(entry->d_name);
        else
        {
          char path_to_weird_thing[PATH_MAX], error_msg[1024];
          sprintf(path_to_weird_thing, "%s/%s", root_path, entry->d_name);
          sprintf(error_msg, "delete_directory: a weird object has been encountered in %s: it is neither a folder nor a file, this function is not designed for that, use maybe 'rm -rf'", path_to_weird_thing);
          throw std::runtime_error(error_msg);
          return 1;
        }
      }
      entry = readdir(dir);
    }
    for (unsigned int idx = 0; idx < reg_files.size(); ++idx) {
      char path_to_file[PATH_MAX];
      sprintf(path_to_file, "%s/%s", root_path, reg_files[idx].c_str());
      remove(path_to_file);
    }
    for (unsigned int idx = 0; idx < subdirectories.size(); ++idx) {
      char path_to_subfolder[PATH_MAX];
      sprintf(path_to_subfolder, "%s/%s", root_path, subdirectories[idx].c_str());
      delete_directory(path_to_subfolder, mpi_rank, comm, true);
    }
    remove(root_path);
    if(non_collective)
      return 0;
    else
      return_ = 0;
  }
  if(!non_collective)
  {
    int mpiret = MPI_Bcast(&return_, 1, MPI_INT, 0, comm); SC_CHECK_MPI(mpiret);
  }
  return return_;
}

void dxyz_min(const p4est_t *p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  for(int dir=0; dir<P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p + dir] - v[3*v_m + dir]) / (1<<data->max_lvl);
}

void get_dxyz_min(const p4est_t *p4est, double *dxyz, double &dxyz_min)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  for(int dir=0; dir<P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p + dir] - v[3*v_m + dir]) / (1<<data->max_lvl);

  dxyz_min = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));
}

void get_dxyz_min(const p4est_t *p4est, double *dxyz, double &dxyz_min, double &diag_min)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  for(int dir=0; dir<P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p + dir] - v[3*v_m + dir]) / (1<<data->max_lvl);

  dxyz_min = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));
  diag_min = sqrt(SUMD(SQR(dxyz[0]), SQR(dxyz[1]), SQR(dxyz[2])));
}

void dxyz_quad(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz)
{
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double *v = p4est->connectivity->vertices;

  double qh = P4EST_QUADRANT_LEN(quad->level) / (double) P4EST_ROOT_LEN;
  for(int dir=0; dir<P4EST_DIM; ++dir)
    dxyz[dir] = (v[3*v_p+dir]-v[3*v_m+dir]) * qh;
}

void xyz_min(const p4est_t *p4est, double *xyz_min_)
{
  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0;
  p4est_topidx_t first_vertex = 0;

  for (short i=0; i<3; i++)
    xyz_min_[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
}

void xyz_max(const p4est_t *p4est, double *xyz_max_)
{
  double *v2c = p4est->connectivity->vertices;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  p4est_topidx_t last_tree = p4est->trees->elem_count-1;
  p4est_topidx_t last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<3; i++)
    xyz_max_[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}

double integrate_over_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue phi_values;
  OctValue f_values;
#else
  QuadValue phi_values;
  QuadValue f_values;
#endif

  double *P, *F;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &F); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;

  // TODO: This is terrible! QuadValue, Cube2, Point2, etc should be templated classes!
#ifdef P4_TO_P8
  phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

  f_values.val000   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val100   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val010   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val110   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  f_values.val001   = F[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  f_values.val101   = F[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  f_values.val011   = F[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  f_values.val111   = F[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

#else
  phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];

  f_values.val00   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val10   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val01   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val11   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif

  ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &F); CHKERRXX(ierr);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (tree_xmax-tree_xmin)*dmin;
  double dy = (tree_ymax-tree_ymin)*dmin;
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  double dz = (tree_zmax-tree_zmin)*dmin;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
#else
  Cube2 cube(0, dx, 0, dy);
#endif

  return cube.integral(f_values,phi_values);
}


double integrate_over_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += integrate_over_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                            quad_idx + tree->quadrants_offset,
                                                            phi, f);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}


double area_in_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{

#ifdef P4_TO_P8
  OctValue phi_values;
#else
  QuadValue phi_values;
#endif

  double *P;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;

#ifdef P4_TO_P8
  phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];
#else
  phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif

  ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (tree_xmax-tree_xmin)*dmin;
  double dy = (tree_ymax-tree_ymin)*dmin;
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  double dz = (tree_zmax-tree_zmin)*dmin;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
  return cube.volume_In_Negative_Domain(phi_values);
#else
  Cube2 cube(0, dx, 0, dy);
  return cube.area_In_Negative_Domain(phi_values);
#endif
}

double area_in_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += area_in_negative_domain_in_one_quadrant(p4est, nodes, quad,
                                                     quad_idx + tree->quadrants_offset,
                                                     phi);
    }
  }

  /* compute global sum */
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum;
}

double integrate_over_interface_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue phi_values;
  OctValue f_values;
#else
  QuadValue phi_values;
  QuadValue f_values;
#endif
  double *P, *F;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &F); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
#ifdef P4_TO_P8
  phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

  f_values.val000   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val100   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val010   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val110   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  f_values.val001   = F[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  f_values.val101   = F[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  f_values.val011   = F[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  f_values.val111   = F[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

#else
  phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];

  f_values.val00   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val10   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val01   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val11   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif
  ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &F); CHKERRXX(ierr);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (tree_xmax-tree_xmin)*dmin;
  double dy = (tree_ymax-tree_ymin)*dmin;
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  double dz = (tree_zmax-tree_zmin)*dmin;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
#else
  Cube2 cube(0, dx, 0, dy);
#endif

  return cube.integrate_Over_Interface(f_values,phi_values);
}

double max_over_interface_in_one_quadrant(const p4est_nodes_t *nodes, p4est_locidx_t quad_idx, Vec phi, Vec f)
{
#ifdef P4_TO_P8
  OctValue phi_values;
  OctValue f_values;
#else
  QuadValue phi_values;
  QuadValue f_values;
#endif
  double *P, *F;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &F); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
#ifdef P4_TO_P8
  phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

  f_values.val000   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val100   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val010   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val110   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  f_values.val001   = F[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  f_values.val101   = F[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  f_values.val011   = F[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  f_values.val111   = F[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];

#else
  phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];

  f_values.val00   = F[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  f_values.val10   = F[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  f_values.val01   = F[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  f_values.val11   = F[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif
  ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &F); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Cube3 cube(0, 1, 0, 1, 0, 1);
#else
  Cube2 cube(0, 1, 0, 1);
#endif

  return cube.max_Over_Interface(f_values,phi_values);
}

double integrate_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += integrate_over_interface_in_one_quadrant(p4est, nodes, quad,
                                                      quad_idx + tree->quadrants_offset,
                                                      phi, f);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

double max_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double max_over_interface = -DBL_MAX;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
      max_over_interface = MAX(max_over_interface, max_over_interface_in_one_quadrant(nodes, quad_idx + tree->quadrants_offset, phi, f));
  }

  /* compute global sum */
  double max_over_interface_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&max_over_interface, &max_over_interface_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);
  return max_over_interface_global;
}

double compute_mean_curvature(const quad_neighbor_nodes_of_node_t &qnnn, double *phi, double* phi_x[])
{
#ifdef CASL_THROWS
  if(!phi_x)
    throw std::invalid_argument("phi_x cannot be NULL when computing curvature.");
#endif

  // compute first derivatives
  double dx = phi_x[0][qnnn.node_000];
  double dy = phi_x[1][qnnn.node_000];
#ifdef P4_TO_P8
  double dz = phi_x[2][qnnn.node_000];
#endif

  // compute second derivatives
  double dxx = qnnn.dxx_central(phi);
  double dyy = qnnn.dyy_central(phi);
  double dxy = qnnn.dy_central(phi_x[0]); // d/dy{d/dx}
#ifdef P4_TO_P8
  double dzz = qnnn.dzz_central(phi);
  double dxz = qnnn.dz_central(phi_x[0]); // d/dz{d/dx}
  double dyz = qnnn.dz_central(phi_x[1]); // d/dz{d/dy}
#endif

#ifdef P4_TO_P8
  double abs   = MAX(EPS, sqrt(SQR(dx)+SQR(dy)+SQR(dz)));
  double kappa = ((dyy+dzz)*SQR(dx) + (dxx+dzz)*SQR(dy) + (dxx+dyy)*SQR(dz) - 2*
                   (dx*dy*dxy + dx*dz*dxz + dy*dz*dyz)) / abs/abs/abs;
#else
  double abs   = MAX(EPS, sqrt(SQR(dx)+SQR(dy)));
  double kappa = (dxx*SQR(dy) - 2*dy*dx*dxy + dyy*SQR(dx)) / abs/abs/abs;
#endif
  return kappa;
}

double compute_mean_curvature(const quad_neighbor_nodes_of_node_t &qnnn, double *normals[])
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals cannot be NULL when computing curvature.");
#endif

#ifdef P4_TO_P8
  double kappa = qnnn.dx_central(normals[0]) + qnnn.dy_central(normals[1]) + qnnn.dz_central(normals[2]);
#else
  double kappa = qnnn.dx_central(normals[0]) + qnnn.dy_central(normals[1]);
#endif

  return kappa;
}

void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec phi_x[], Vec kappa)
{
#ifdef CASL_THROWS
  if(!phi_x)
    throw std::invalid_argument("phi_x cannot be NULL when computing curvature.");
#endif

  double *phi_p, *phi_x_p[P4EST_DIM], *kappa_p;
  VecGetArray(phi, &phi_p);
  VecGetArray(kappa, &kappa_p);
  foreach_dimension(dim) VecGetArray(phi_x[dim], &phi_x_p[dim]);

  // compute kappa on layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for (size_t i=0; i<neighbors.get_layer_size(); ++i) {
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = compute_mean_curvature(qnnn, phi_p, phi_x_p);
  }

  // initiate communication
  VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD);

  // compute on local nodes
  for (size_t i=0; i<neighbors.get_local_size(); ++i) {
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = compute_mean_curvature(qnnn, phi_p, phi_x_p);
  }

  // finish communication
  VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD);

  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(kappa, &kappa_p);
  foreach_dimension(dim) VecRestoreArray(phi_x[dim], &phi_x_p[dim]);
}

void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec normals[], Vec kappa)
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals cannot be NULL when computing curvature.");
#endif

  double *normals_p[P4EST_DIM], *kappa_p;
  VecGetArray(kappa, &kappa_p);
  foreach_dimension(dim) VecGetArray(normals[dim], &normals_p[dim]);

  // compute kappa on layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  for (size_t i=0; i<neighbors.get_layer_size(); ++i) {
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = compute_mean_curvature(qnnn, normals_p);
  }

  // initiate communication
  VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD);

  // compute on local nodes
  for (size_t i=0; i<neighbors.get_local_size(); ++i) {
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);

    kappa_p[n] = compute_mean_curvature(qnnn, normals_p);
  }

  // finish communication
  VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD);

  VecRestoreArray(kappa, &kappa_p);
  foreach_dimension(dim) VecRestoreArray(normals[dim], &normals_p[dim]);
}

void compute_normals(const quad_neighbor_nodes_of_node_t &qnnn, double *phi, double normals[])
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  normals[0] = qnnn.dx_central(phi);
  normals[1] = qnnn.dy_central(phi);
#ifdef P4_TO_P8
  normals[2] = qnnn.dz_central(phi);
  double abs = sqrt(SQR(normals[0]) + SQR(normals[1]) + SQR(normals[2]));
#else
  double abs = sqrt(SQR(normals[0]) + SQR(normals[1]));
#endif
  if (abs < EPS)
    foreach_dimension(dim) normals[dim] = 0;
  else
    foreach_dimension(dim) normals[dim] /= abs;
}

void compute_normals(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec normals[])
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  neighbors.first_derivatives_central(phi, normals);
  double *normals_p[P4EST_DIM];
  foreach_dimension(dim) VecGetArray(normals[dim], &normals_p[dim]);

  foreach_node(n, neighbors.get_nodes()) {
#ifdef P4_TO_P8
    double abs = sqrt(SQR(normals_p[0][n]) + SQR(normals_p[1][n]) + SQR(normals_p[2][n]));
#else
    double abs = sqrt(SQR(normals_p[0][n]) + SQR(normals_p[1][n]));
#endif

    if (abs < EPS) {
      foreach_dimension(dim) normals_p[dim][n] = 0;
    } else {
      foreach_dimension(dim) normals_p[dim][n] /= abs;
    }
  }

  foreach_dimension(dim) VecRestoreArray(normals[dim], &normals_p[dim]);
}

void compute_normals(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec normals)
{
#ifdef CASL_THROWS
  if(!normals)
    throw std::invalid_argument("normals array cannot be NULL.");
#endif

  neighbors.first_derivatives_central(phi, normals);
  double *normals_p;
  PetscErrorCode ierr = VecGetArray(normals, &normals_p); CHKERRXX(ierr);

  foreach_node(n, neighbors.get_nodes()) {
    double abs = 0.0;
    foreach_dimension(dim) abs += SQR(normals_p[P4EST_DIM*n+dim]);
    abs = sqrt(abs);
    if(abs < EPS){
      foreach_dimension(dim) normals_p[P4EST_DIM*n+dim] = 0.0;
    } else{
      foreach_dimension(dim) normals_p[P4EST_DIM*n+dim] /= abs;
    }
  }
  ierr = VecRestoreArray(normals, &normals_p); CHKERRXX(ierr);
}

double interface_length_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{
#ifdef P4_TO_P8
  OctValue phi_values;
#else
  QuadValue phi_values;
#endif
  double *P;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
#ifdef P4_TO_P8
  phi_values.val000 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val100 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val010 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val110 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
  phi_values.val001 = P[ q2n[ quad_idx*P4EST_CHILDREN + 4 ] ];
  phi_values.val101 = P[ q2n[ quad_idx*P4EST_CHILDREN + 5 ] ];
  phi_values.val011 = P[ q2n[ quad_idx*P4EST_CHILDREN + 6 ] ];
  phi_values.val111 = P[ q2n[ quad_idx*P4EST_CHILDREN + 7 ] ];
#else
  phi_values.val00 = P[ q2n[ quad_idx*P4EST_CHILDREN + 0 ] ];
  phi_values.val10 = P[ q2n[ quad_idx*P4EST_CHILDREN + 1 ] ];
  phi_values.val01 = P[ q2n[ quad_idx*P4EST_CHILDREN + 2 ] ];
  phi_values.val11 = P[ q2n[ quad_idx*P4EST_CHILDREN + 3 ] ];
#endif
  ierr = VecRestoreArray(phi, &P); CHKERRXX(ierr);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (tree_xmax-tree_xmin)*dmin;
  double dy = (tree_ymax-tree_ymin)*dmin;
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  double dz = (tree_zmax-tree_zmin)*dmin;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
  return cube.interface_Area_In_Cell(phi_values);
#else
  Cube2 cube(0, dx, 0, dy);
  return cube.interface_Length_In_Cell(phi_values);
#endif
}

double interface_length(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += interface_length_in_one_quadrant(p4est, nodes, quad,
                                              quad_idx + tree->quadrants_offset,
                                              phi);
    }
  }

  /* compute global sum */
  double sum_global;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
  return sum_global;
}

bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_m00] != tr_it)
    return false;
  else if (ni->x == 0)
    return true;
  else
    return false;
}

bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_p00] != tr_it)
    return false;
  else if (ni->x == P4EST_ROOT_LEN - 1 || ni->x == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0m0] != tr_it)
    return false;
  else if (ni->y == 0)
    return true;
  else
    return false;
}

bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0p0] != tr_it)
    return false;
  else if (ni->y == P4EST_ROOT_LEN - 1 || ni->y == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

#ifdef P4_TO_P8
bool is_node_zmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00m] != tr_it)
    return false;
  else if (ni->z == 0)
    return true;
  else
    return false;
}

bool is_node_zpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00p] != tr_it)
    return false;
  else if (ni->z == P4EST_ROOT_LEN - 1 || ni->z == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}
#endif

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni)
{
#ifdef P4_TO_P8
  return ( is_node_xmWall(p4est, ni) || is_node_xpWall(p4est, ni) ||
           is_node_ymWall(p4est, ni) || is_node_ypWall(p4est, ni) ||
           is_node_zmWall(p4est, ni) || is_node_zpWall(p4est, ni) );
#else
  return ( is_node_xmWall(p4est, ni) || is_node_xpWall(p4est, ni) ||
           is_node_ymWall(p4est, ni) || is_node_ypWall(p4est, ni) );
#endif
}

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni, bool is_wall[])
{
  bool is_any = false;

  is_wall[dir::f_m00] = is_node_xmWall(p4est, ni); is_any = is_any || is_wall[dir::f_m00];
  is_wall[dir::f_p00] = is_node_xpWall(p4est, ni); is_any = is_any || is_wall[dir::f_p00];
  is_wall[dir::f_0m0] = is_node_ymWall(p4est, ni); is_any = is_any || is_wall[dir::f_0m0];
  is_wall[dir::f_0p0] = is_node_ypWall(p4est, ni); is_any = is_any || is_wall[dir::f_0p0];
#ifdef P4_TO_P8
  is_wall[dir::f_00m] = is_node_zmWall(p4est, ni); is_any = is_any || is_wall[dir::f_00m];
  is_wall[dir::f_00p] = is_node_zpWall(p4est, ni); is_any = is_any || is_wall[dir::f_00p];
#endif
  return is_any;
}

bool is_node_Wall(const p4est_t *p4est, const p4est_indep_t *ni, const unsigned char oriented_dir)
{
  switch(oriented_dir)
  {
  case dir::f_m00: return is_node_xmWall(p4est, ni);
  case dir::f_p00: return is_node_xpWall(p4est, ni);
  case dir::f_0m0: return is_node_ymWall(p4est, ni);
  case dir::f_0p0: return is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  case dir::f_00m: return is_node_zmWall(p4est, ni);
  case dir::f_00p: return is_node_zpWall(p4est, ni);
#endif
  default:
    throw std::invalid_argument("[CASL_ERROR]: is_node_wall: unknown direction.");
  }
}

bool is_quad_xmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_m00] != tr_it)
    return false;
  else if (qi->x == 0)
    return true;
  else
    return false;
}

bool is_quad_xpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 0)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_p00] != tr_it)
    return false;
  else if (qi->x == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}

bool is_quad_ymWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_0m0] != tr_it)
    return false;
  else if (qi->y == 0)
    return true;
  else
    return false;
}

bool is_quad_ypWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 1)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_0p0] != tr_it)
    return false;
  else if (qi->y == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}

#ifdef P4_TO_P8
bool is_quad_zmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;

  if (t2t[P4EST_FACES*tr_it + dir::f_00m] != tr_it)
    return false;
  else if (qi->z == 0)
    return true;
  else
    return false;
}

bool is_quad_zpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
  if (is_periodic(p4est, 2)) return false;

  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(qi->level);

  if (t2t[P4EST_FACES*tr_it + dir::f_00p] != tr_it)
    return false;
  else if (qi->z == P4EST_ROOT_LEN - qh)
    return true;
  else
    return false;
}
#endif

bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi, int dir)
{
  switch(dir)
  {
  case dir::f_m00: return is_quad_xmWall(p4est, tr_it, qi);
  case dir::f_p00: return is_quad_xpWall(p4est, tr_it, qi);
  case dir::f_0m0: return is_quad_ymWall(p4est, tr_it, qi);
  case dir::f_0p0: return is_quad_ypWall(p4est, tr_it, qi);
#ifdef P4_TO_P8
  case dir::f_00m: return is_quad_zmWall(p4est, tr_it, qi);
  case dir::f_00p: return is_quad_zpWall(p4est, tr_it, qi);
#endif
  default:
    throw std::invalid_argument("[CASL_ERROR]: is_quad_wall: unknown direction.");
  }
}

bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
#ifdef P4_TO_P8
  return ( is_quad_xmWall(p4est, tr_it, qi) || is_quad_xpWall(p4est, tr_it, qi) ||
           is_quad_ymWall(p4est, tr_it, qi) || is_quad_ypWall(p4est, tr_it, qi) ||
           is_quad_zmWall(p4est, tr_it, qi) || is_quad_zpWall(p4est, tr_it, qi) );
#else
  return ( is_quad_xmWall(p4est, tr_it, qi) || is_quad_xpWall(p4est, tr_it, qi) ||
           is_quad_ymWall(p4est, tr_it, qi) || is_quad_ypWall(p4est, tr_it, qi) );
#endif
}

int quad_find_ghost_owner(const p4est_ghost_t *ghost, p4est_locidx_t ghost_idx)
{
  P4EST_ASSERT(ghost_idx<(p4est_locidx_t) ghost->ghosts.elem_count);
  int r=0;
  while(ghost->proc_offsets[r+1]<=ghost_idx) r++;
  return r;
}

#ifdef P4_TO_P8
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f)
#else
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f)
#endif
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    PetscInt size;
    ierr = VecGetLocalSize(f, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->num_owned_indeps << ", " << nodes->indep_nodes.elem_count << ", "
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

#ifdef P4_TO_P8
    f_p[i] = cf(x,y,z);
#else
    f_p[i] = cf(x,y);
#endif
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f)
#else
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f)
#endif
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

#ifdef P4_TO_P8
    f_p[i] = cf(x,y,z);
#else
    f_p[i] = cf(x,y);
#endif
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3* cf_array[], Vec f)
#else
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2* cf_array[], Vec f)
#endif
{
  double *f_p;
  PetscInt bs;
  PetscErrorCode ierr;
  ierr = VecGetBlockSize(f, &bs); CHKERRXX(ierr);

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    if (size != (PetscInt) nodes->indep_nodes.elem_count * bs){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points x block_size."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " block_size = " << bs
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);

    for (PetscInt j = 0; j<bs; j++) {
#ifdef P4_TO_P8
      const CF_3& cf = *cf_array[j];
      f_p[i*bs + j] = cf(xyz[0], xyz[1], xyz[2]);
#else
      const CF_2& cf = *cf_array[j];
      f_p[i*bs + j] = cf(xyz[0], xyz[1]);
#endif
    }
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}


#ifdef P4_TO_P8
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, std::vector<double>& f)
#else
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, std::vector<double>& f)
#endif
{
#ifdef CASL_THROWS
  {
    if ((PetscInt) f.size() != (PetscInt) nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " VecSize = " << f.size() << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif

  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

#ifdef P4_TO_P8
    f[i] = cf(x,y,z);
#else
    f[i] = cf(x,y);
#endif
  }
}

#ifdef P4_TO_P8
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_3& cf, Vec f)
#else
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_2& cf, Vec f)
#endif
{
  double *f_p;
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  {
    Vec local_form;
    ierr = VecGhostGetLocalForm(f, &local_form); CHKERRXX(ierr);
    PetscInt size;
    ierr = VecGetSize(local_form, &size); CHKERRXX(ierr);
    PetscInt num_local = (PetscInt)(p4est->local_num_quadrants + ghost->ghosts.elem_count);

    if (size != num_local){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             " p4est->local_num_quadrants + ghost->ghosts.elem_count = " << num_local
          << " VecSize = " << size << std::endl;

      throw std::invalid_argument(oss.str());
    }
    ierr = VecGhostRestoreLocalForm(f, &local_form); CHKERRXX(ierr);
  }
#endif

  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  // sample on local quadrants
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      double x = quad_x_fr_q(quad_idx, tree_id, p4est, ghost);
      double y = quad_y_fr_q(quad_idx, tree_id, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(quad_idx, tree_id, p4est, ghost);
#endif

#ifdef P4_TO_P8
      f_p[quad_idx] = cf(x,y,z);
#else
      f_p[quad_idx] = cf(x,y);
#endif
    }
  }

  // sample on ghost quadrants
  for (size_t q = 0; q < ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    p4est_topidx_t tree_id  = quad->p.piggy3.which_tree;
    p4est_locidx_t quad_idx = q + p4est->local_num_quadrants;

    double x = quad_x_fr_q(quad_idx, tree_id, p4est, ghost);
    double y = quad_y_fr_q(quad_idx, tree_id, p4est, ghost);
#ifdef P4_TO_P8
    double z = quad_z_fr_q(quad_idx, tree_id, p4est, ghost);
#endif

#ifdef P4_TO_P8
    f_p[quad_idx] = cf(x,y,z);
#else
    f_p[quad_idx] = cf(x,y);
#endif
  }

  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, std::vector<double>& f)
#else
void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, std::vector<double>& f)
#endif
{
#ifdef CASL_THROWS
  {
    if (f.size() != nodes->indep_nodes.elem_count){
      std::ostringstream oss;
      oss << "[ERROR]: size of the input vector must be equal to the total number of points."
             "nodes->indep_nodes.elem_count = " << nodes->indep_nodes.elem_count
          << " size() = " << f.size() << std::endl;

      throw std::invalid_argument(oss.str());
    }
  }
#endif
  const p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  const double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    const p4est_indep_t *node = (const p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_m = t2v[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_p = t2v[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];

    double tree_xmin = v2q[3*v_m + 0];
    double tree_xmax = v2q[3*v_p + 0];
    double tree_ymin = v2q[3*v_m + 1];
    double tree_ymax = v2q[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_m + 2];
    double tree_zmax = v2q[3*v_p + 2];
#endif

    double x = (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
    double y = (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
#endif

#ifdef P4_TO_P8
    f[i] = cf(x,y,z);
#else
    f[i] = cf(x,y);
#endif
  }
}

std::ostream& operator<< (std::ostream& os, BoundaryConditionType type)
{
  switch(type){
  case DIRICHLET:
    os << "Dirichlet";
    break;

  case NEUMANN:
    os << "Neumann";
    break;

  case ROBIN:
    os << "Robin";
    break;

  case NOINTERFACE:
    os << "No-Interface";
    break;

  case MIXED:
    os << "Mixed";
    break;

  default:
    os << "UNKNOWN";
    break;
  }

  return os;
}


std::istream& operator>> (std::istream& is, BoundaryConditionType& type)
{
  std::string str;
  is >> str;

  if (str == "DIRICHLET" || str == "Dirichlet" || str == "dirichlet")
    type = DIRICHLET;
  else if (str == "NEUMANN" || str == "Neumann" || str == "neumann")
    type = NEUMANN;
  else if (str == "ROBIN" || str == "Robin" || str == "robin")
    type = ROBIN;
  else if (str == "NOINTERFACE" || str == "Nointerface" || str == "No-Interface" || str == "nointerface" || str == "no-interface")
    type = NOINTERFACE;
  else if (str == "MIXED" || str == "Mixed" || str == "mixed")
    type = MIXED;
  else
    throw std::invalid_argument("[ERROR]: Unknown BoundaryConditionType entered");

  return is;
}


#ifdef P4_TO_P8
double quadrant_interp_t::operator()(double x, double y, double z) const
{
  double xyz_node[P4EST_DIM] = { x, y, z};
#else
double quadrant_interp_t::operator()(double x, double y) const
{
  double xyz_node[P4EST_DIM] = { x, y };
#endif

#ifdef CASL_THROWS
  if (F_ == NULL) throw std::invalid_argument("[CASL_ERROR]: Values are not provided for interpolation.");
  if (Fdd_ == NULL && (method_ == quadratic || method_ == quadratic_non_oscillatory) ) throw std::invalid_argument("[CASL_ERROR]: Second order derivatives are not provided for quadratic interpolation.");
#endif

  switch (method_)
  {
    case linear:                    return linear_interpolation                   (p4est_, tree_idx_, *quad_, F_->data(),               xyz_node); break;
    case quadratic:                 return quadratic_interpolation                (p4est_, tree_idx_, *quad_, F_->data(), Fdd_->data(), xyz_node); break;
    case quadratic_non_oscillatory: return quadratic_non_oscillatory_interpolation(p4est_, tree_idx_, *quad_, F_->data(), Fdd_->data(), xyz_node); break;
    default: throw std::domain_error("Wrong type of interpolation\n");
  }
}

void copy_ghosted_vec(Vec input, Vec output)
{
  PetscErrorCode ierr;
  Vec src, out;
  ierr = VecGhostGetLocalForm(input, &src);      CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(output, &out);     CHKERRXX(ierr);
  ierr = VecCopy(src, out);                      CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input, &src);  CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
}

void set_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecSet(ptr, scalar);                 CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
}

void shift_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecShift(ptr, scalar);               CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
}

void scale_ghosted_vec(Vec vec, double scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecScale(ptr, scalar);               CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
}

PetscErrorCode VecCopyGhost(Vec input, Vec output)
{
  PetscErrorCode ierr;
  Vec src, out;
  ierr = VecGhostGetLocalForm(input, &src);      CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(output, &out);     CHKERRXX(ierr);
  ierr = VecCopy(src, out);                      CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input, &src);  CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecSetGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecSet(ptr, scalar);                 CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecShiftGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecShift(ptr, scalar);               CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecScaleGhost(Vec vec, PetscScalar scalar)
{
  PetscErrorCode ierr;
  Vec ptr;
  ierr = VecGhostGetLocalForm(vec, &ptr);     CHKERRXX(ierr);
  ierr = VecScale(ptr, scalar);               CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(vec, &ptr); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecPointwiseMultGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(input2, &in2);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(output, &out);     CHKERRXX(ierr);
  ierr = VecPointwiseMult(out, in1, in2);        CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input1, &in1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input2, &in2); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecAXPBYGhost(Vec y, PetscScalar alpha, PetscScalar beta, Vec x)
{
  PetscErrorCode ierr;
  Vec X, Y;
  ierr = VecGhostGetLocalForm(x, &X);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(y, &Y);     CHKERRXX(ierr);
  ierr = VecAXPBY(Y, alpha, beta, X);     CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(x, &X); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(y, &Y); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecPointwiseMinGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(input2, &in2);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(output, &out);     CHKERRXX(ierr);
  ierr = VecPointwiseMin(out, in1, in2);        CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input1, &in1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input2, &in2); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecPointwiseMaxGhost(Vec output, Vec input1, Vec input2)
{
  PetscErrorCode ierr;
  Vec out, in1, in2;
  ierr = VecGhostGetLocalForm(input1, &in1);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(input2, &in2);     CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(output, &out);     CHKERRXX(ierr);
  ierr = VecPointwiseMax(out, in1, in2);        CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input1, &in1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input2, &in2); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  return ierr;
}

PetscErrorCode VecReciprocalGhost(Vec input)
{
  PetscErrorCode ierr;
  Vec in;
  ierr = VecGhostGetLocalForm(input, &in);     CHKERRXX(ierr);
  ierr = VecReciprocal(in);                    CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(input, &in); CHKERRXX(ierr);
  return ierr;
}

void invert_phi(p4est_nodes_t *nodes, Vec phi)
{
  PetscErrorCode ierr;
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  for (size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
    phi_p[n] = -phi_p[n];

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
}

void compute_normals_and_mean_curvature(const my_p4est_node_neighbors_t &neighbors, const Vec phi, Vec normals[], Vec kappa)
{
  PetscErrorCode ierr;
  const p4est_nodes_t *nodes = neighbors.get_nodes();

  /* compute first derivatives */
  neighbors.first_derivatives_central(phi, normals);

  /* compute curvature */
  compute_mean_curvature(neighbors, phi, normals, kappa);

  /* compute normals */
  double *normal_p[P4EST_DIM];
  foreach_dimension(dim) { ierr = VecGetArray(normals[dim], &normal_p[dim]); CHKERRXX(ierr); }

  foreach_node(n, nodes)
  {
#ifdef P4_TO_P8
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
    double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

    normal_p[0][n] = norm < EPS ? 0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm < EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
    normal_p[2][n] = norm < EPS ? 0 : normal_p[2][n]/norm;
#endif
  }

  foreach_dimension(dim) { ierr = VecRestoreArray(normals[dim], &normal_p[dim]); CHKERRXX(ierr); }
}

void save_vector(const char *filename, const std::vector<double> &data, std::ios_base::openmode mode, char delim)
{
  std::ofstream ofs;
  ofs.open(filename, mode);

  for (unsigned int i = 0; i < data.size(); ++i)
  {
    if (i != 0) ofs << delim;
    ofs << data[i];
  }

  ofs << "\n";
}






void fill_island(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, int number, p4est_locidx_t n)
{
  const p4est_nodes_t *nodes = ngbd.get_nodes();

    std::stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        island_number_p[k] = number;
        const quad_neighbor_nodes_of_node_t& qnnn = ngbd[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && island_number_p[qnnn.node_m00_mm]<0) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && island_number_p[qnnn.node_m00_pm]<0) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && island_number_p[qnnn.node_p00_mm]<0) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && island_number_p[qnnn.node_p00_pm]<0) st.push(qnnn.node_p00_pm);

        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && island_number_p[qnnn.node_0m0_mm]<0) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && island_number_p[qnnn.node_0m0_pm]<0) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && island_number_p[qnnn.node_0p0_mm]<0) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && island_number_p[qnnn.node_0p0_pm]<0) st.push(qnnn.node_0p0_pm);
    }
}


void find_connected_ghost_islands(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited)
{
  const p4est_nodes_t *nodes = ngbd.get_nodes();

    std::stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        visited[k] = true;
        const quad_neighbor_nodes_of_node_t& qnnn = ngbd[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !visited[qnnn.node_m00_mm]) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !visited[qnnn.node_m00_pm]) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !visited[qnnn.node_p00_mm]) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !visited[qnnn.node_p00_pm]) st.push(qnnn.node_p00_pm);

        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !visited[qnnn.node_0m0_mm]) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !visited[qnnn.node_0m0_pm]) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !visited[qnnn.node_0p0_mm]) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !visited[qnnn.node_0p0_pm]) st.push(qnnn.node_0p0_pm);

        /* check connected ghost island and add to list if new */
        if(qnnn.node_m00_mm>=nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]>0 && !contains(connected, island_number_p[qnnn.node_m00_mm])) connected.push_back(island_number_p[qnnn.node_m00_mm]);
        if(qnnn.node_m00_pm>=nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]>0 && !contains(connected, island_number_p[qnnn.node_m00_pm])) connected.push_back(island_number_p[qnnn.node_m00_pm]);
        if(qnnn.node_p00_mm>=nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]>0 && !contains(connected, island_number_p[qnnn.node_p00_mm])) connected.push_back(island_number_p[qnnn.node_p00_mm]);
        if(qnnn.node_p00_pm>=nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]>0 && !contains(connected, island_number_p[qnnn.node_p00_pm])) connected.push_back(island_number_p[qnnn.node_p00_pm]);

        if(qnnn.node_0m0_mm>=nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_mm])) connected.push_back(island_number_p[qnnn.node_0m0_mm]);
        if(qnnn.node_0m0_pm>=nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0m0_pm])) connected.push_back(island_number_p[qnnn.node_0m0_pm]);
        if(qnnn.node_0p0_mm>=nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_mm])) connected.push_back(island_number_p[qnnn.node_0p0_mm]);
        if(qnnn.node_0p0_pm>=nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]>0 && !contains(connected, island_number_p[qnnn.node_0p0_pm])) connected.push_back(island_number_p[qnnn.node_0p0_pm]);
    }
}


void compute_islands_numbers(const my_p4est_node_neighbors_t &ngbd, const Vec phi, int &nb_islands_total, Vec island_number)
{
  PetscErrorCode ierr;

  const p4est_t       *p4est = ngbd.get_p4est();
  const p4est_nodes_t *nodes = ngbd.get_nodes();

  nb_islands_total = 0;
  int proc_padding = 1e6;
//  return;

  Vec loc;
  ierr = VecGhostGetLocalForm(island_number, &loc); CHKERRXX(ierr);
  ierr = VecSet(loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(island_number, &loc); CHKERRXX(ierr);

  /* first everyone compute the local numbers */
  std::vector<int> nb_islands(p4est->mpisize);
  nb_islands[p4est->mpirank] = p4est->mpirank*proc_padding;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  double *island_number_p;
  ierr = VecGetArray(island_number, &island_number_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd.get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd.get_layer_node(i);
    if(phi_p[n]>0 && island_number_p[n]<0)
    {
      fill_island(ngbd, phi_p, island_number_p, nb_islands[p4est->mpirank], n);
      nb_islands[p4est->mpirank]++;
    }
  }
  ierr = VecGhostUpdateBegin(island_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd.get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd.get_local_node(i);
    if(phi_p[n]>0 && island_number_p[n]<0)
    {
      fill_island(ngbd, phi_p, island_number_p, nb_islands[p4est->mpirank], n);
      nb_islands[p4est->mpirank]++;
    }
  }
  ierr = VecGhostUpdateEnd(island_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* get remote number of islands to prepare graph communication structure */
  int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &nb_islands[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  /* compute offset for each process */
  std::vector<int> proc_offset(p4est->mpisize+1);
  proc_offset[0] = 0;
  for(int p=0; p<p4est->mpisize; ++p)
    proc_offset[p+1] = proc_offset[p] + (nb_islands[p]%proc_padding);

  /* build a local graph with
         *   - vertices = island number
         *   - edges    = connected islands
         * in order to simplify the communications, the graph is stored as a full matrix. Given the sparsity, this can be optimized ...
         */
  int nb_islands_g = proc_offset[p4est->mpisize];
  std::vector<int> graph(nb_islands_g*nb_islands_g, 0);
  /* note that the only reason this is double and not int is that Petsc works with doubles, can't do Vec of int ... */
  std::vector<double> connected;
  std::vector<bool> visited(nodes->num_owned_indeps, false);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(island_number_p[n]>=0 && !visited[n])
    {
      /* find the connected islands and add the connection information to the graph */
      find_connected_ghost_islands(ngbd, phi_p, island_number_p, n, connected, visited);
      for(unsigned int i=0; i<connected.size(); ++i)
      {
        int local_id = proc_offset[p4est->mpirank]+static_cast<int>(island_number_p[n])%proc_padding;
        int remote_id = proc_offset[static_cast<int>(connected[i])/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
        graph[nb_islands_g*local_id + remote_id] = 1;
      }

      connected.clear();
    }
  }

  std::vector<int> rcvcounts(p4est->mpisize);
  std::vector<int> displs(p4est->mpisize);
  for(int p=0; p<p4est->mpisize; ++p)
  {
    rcvcounts[p] = (nb_islands[p]%proc_padding) * nb_islands_g;
    displs[p] = proc_offset[p]*nb_islands_g;
  }

  mpiret = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &graph[0], &rcvcounts[0], &displs[0], MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  /* now we can color the graph connecting the islands, and thus obtain a unique numbering for all the islands */
  std::vector<int> graph_numbering(nb_islands_g,-1);
  std::stack<int> st;
  for(int i=0; i<nb_islands_g; ++i)
  {
    if(graph_numbering[i]==-1)
    {
      st.push(i);
      while(!st.empty())
      {
        int k = st.top();
        st.pop();
        graph_numbering[k] = nb_islands_total;
        for(int j=0; j<nb_islands_g; ++j)
        {
          int nj = k*nb_islands_g+j;
          if(graph[nj] && graph_numbering[j]==-1)
            st.push(j);
        }
      }
      nb_islands_total++;
    }
  }

  /* and finally assign the correct number to the islands of this level */
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if(island_number_p[n]>=0)
    {
      int index = proc_offset[static_cast<int>(island_number_p[n])/proc_padding] + (static_cast<int>(island_number_p[n])%proc_padding);
      island_number_p[n] = graph_numbering[index];
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(island_number, &island_number_p); CHKERRXX(ierr);
}

void compute_phi_eff(Vec phi_eff, p4est_nodes_t *nodes, std::vector<Vec> &phi, std::vector<mls_opn_t> &opn)
{
  PetscErrorCode ierr;
  double* phi_eff_ptr;
  ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

  std::vector<double *> phi_ptr(phi.size(), NULL);

  for (size_t i = 0; i < phi.size(); i++) { ierr = VecGetArray(phi.at(i), &phi_ptr[i]); CHKERRXX(ierr); }

  foreach_node(n, nodes)
  {
    double phi_total = -DBL_MAX;
    for (unsigned int i = 0; i < phi.size(); i++)
    {
      double phi_current = phi_ptr[i][n];

      if      (opn.at(i) == MLS_INTERSECTION) phi_total = MAX(phi_total, phi_current);
      else if (opn.at(i) == MLS_ADDITION)     phi_total = MIN(phi_total, phi_current);
    }
    phi_eff_ptr[n] = phi_total;
  }

//  if (refine_always != NULL)
//    foreach_node(n, nodes)
//      for (unsigned int i = 0; i < phi->size(); i++)
//        if (refine_always->at(i))
//          phi_eff_ptr[n] = MIN(phi_eff_ptr[n], fabs(phi_ptr[i][n]));

  for (size_t i = 0; i < phi.size(); i++) { ierr = VecRestoreArray(phi.at(i), &phi_ptr[i]); CHKERRXX(ierr); }
}

void compute_phi_eff(Vec phi_eff, p4est_nodes_t *nodes, int num_phi, ...)
{
  va_list ap;

  va_start(ap, num_phi);

  std::vector<Vec> phi;
  std::vector<mls_opn_t> opn;
  for (int i=0; i<num_phi; ++i) {
    Vec       P = va_arg(ap, Vec);              phi.push_back(P);
    mls_opn_t O = (mls_opn_t) va_arg(ap, int);  opn.push_back(O);
  }

  va_end(ap);

  compute_phi_eff(phi_eff, nodes, phi, opn);
}


void find_closest_interface_location(int &phi_idx, double &dist, double d, std::vector<mls_opn_t> opn,
                                     std::vector<double> &phi_a,
                                     std::vector<double> &phi_b,
                                     std::vector<double> &phi_a_xx,
                                     std::vector<double> &phi_b_xx)
{
  dist    = d;
  phi_idx =-1;

  for (size_t i = 0; i < opn.size(); ++i)
  {
    if (phi_a[i] > 0. && phi_b[i] > 0.)
    {
      if (opn[i] == MLS_INTERSECTION)
      {
        dist    =  0;
        phi_idx = -1;
      }
    } else if (phi_a[i] < 0. && phi_b[i] < 0.) {
      if (opn[i] == MLS_ADDITION)
      {
        dist    =  d;
        phi_idx = -1;
      }
    } else {
      double dist_new = interface_Location_With_Second_Order_Derivative(0., d, phi_a[i], phi_b[i], phi_a_xx[i], phi_b_xx[i]);

      switch (opn[i])
      {
      case MLS_INTERSECTION:
        if (phi_a[i] < 0.)
        {
          if (dist_new < dist)
          {
            dist    = dist_new;
            phi_idx = i;
          }
        } else {
          dist    =  0;
          phi_idx = -1;
        }
        break;
      case MLS_ADDITION:
        if (phi_a[i] < 0.)
        {
          if (dist_new > dist)
          {
            dist    = dist_new;
            phi_idx = i;
          }
        } else {
          if (dist_new < dist)
          {
            dist    =  d;
            phi_idx = -1;
          }
        }
        break;
      default:
#ifdef CASL_THROWS
        throw  std::runtime_error("find_closest_interface_location:: unknown MLS operation: only MLS_INTERSECTION and MLS_ADDITION implemented, currently.");
#endif
        break;
      }
    }
  }
}

void construct_finite_volume(my_p4est_finite_volume_t& fv, p4est_locidx_t n, p4est_t *p4est, p4est_nodes_t *nodes, std::vector<CF_DIM *> phi, std::vector<mls_opn_t> opn, int order, int cube_refinement, bool compute_centroids, double perturb)
{
  double xyz_C[P4EST_DIM];
  double dxyz [P4EST_DIM];

  node_xyz_fr_n(n, p4est, nodes, xyz_C);
  dxyz_min(p4est, dxyz);

  double scale = 1./MAX(DIM(dxyz[0], dxyz[1], dxyz[2]));
  double diag  = sqrt(SUMD(SQR(dxyz[0]), SQR(dxyz[1]), SQR(dxyz[2])));

  // Reconstruct geometry
  double cube_xyz_min[] = { DIM( 0, 0, 0 ) };
  double cube_xyz_max[] = { DIM( 0, 0, 0 ) };
  int    cube_mnk[]     = { DIM( 0, 0, 0 ) };

  CODE2D( cube2_mls_t cube );
  CODE3D( cube3_mls_t cube );

  // determine dimensions of cube
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

  if (!is_node_xmWall(p4est, ni)) { cube_mnk[0] += cube_refinement; cube_xyz_min[0] -= .5*dxyz[0]*scale; }
  if (!is_node_xpWall(p4est, ni)) { cube_mnk[0] += cube_refinement; cube_xyz_max[0] += .5*dxyz[0]*scale; }

  if (!is_node_ymWall(p4est, ni)) { cube_mnk[1] += cube_refinement; cube_xyz_min[1] -= .5*dxyz[1]*scale; }
  if (!is_node_ypWall(p4est, ni)) { cube_mnk[1] += cube_refinement; cube_xyz_max[1] += .5*dxyz[1]*scale; }
#ifdef P4_TO_P8
  if (!is_node_zmWall(p4est, ni)) { cube_mnk[2] += cube_refinement; cube_xyz_min[2] -= .5*dxyz[2]*scale; }
  if (!is_node_zpWall(p4est, ni)) { cube_mnk[2] += cube_refinement; cube_xyz_max[2] += .5*dxyz[2]*scale; }
#endif

  fv.full_cell_volume = MULTD( cube_xyz_max[0]-cube_xyz_min[0],
                               cube_xyz_max[1]-cube_xyz_min[1],
                               cube_xyz_max[2]-cube_xyz_min[2] ) / pow(scale, P4EST_DIM);

  fv.full_face_area[dir::f_m00] = (cube_xyz_max[1]-cube_xyz_min[1]) CODE3D(*(cube_xyz_max[2]-cube_xyz_min[2])) / pow(scale, P4EST_DIM-1);
  fv.full_face_area[dir::f_p00] = (cube_xyz_max[1]-cube_xyz_min[1]) CODE3D(*(cube_xyz_max[2]-cube_xyz_min[2])) / pow(scale, P4EST_DIM-1);

  fv.full_face_area[dir::f_0m0] = (cube_xyz_max[0]-cube_xyz_min[0]) CODE3D(*(cube_xyz_max[2]-cube_xyz_min[2])) / pow(scale, P4EST_DIM-1);
  fv.full_face_area[dir::f_0p0] = (cube_xyz_max[0]-cube_xyz_min[0]) CODE3D(*(cube_xyz_max[2]-cube_xyz_min[2])) / pow(scale, P4EST_DIM-1);
#ifdef P4_TO_P8
  fv.full_face_area[dir::f_00m] = (cube_xyz_max[0]-cube_xyz_min[0])*(cube_xyz_max[1]-cube_xyz_min[1]) / pow(scale, P4EST_DIM-1);
  fv.full_face_area[dir::f_00p] = (cube_xyz_max[0]-cube_xyz_min[0])*(cube_xyz_max[1]-cube_xyz_min[1]) / pow(scale, P4EST_DIM-1);
#endif

  if (cube_refinement == 0) cube_mnk[0] = cube_mnk[1] = CODE3D( cube_mnk[2] = ) 1;

  cube.initialize(cube_xyz_min, cube_xyz_max, cube_mnk, order);

  // get points at which values of level-set functions are needed
  XCODE( std::vector<double> x_grid; cube.get_x_coord(x_grid); );
  YCODE( std::vector<double> y_grid; cube.get_y_coord(y_grid); );
  ZCODE( std::vector<double> z_grid; cube.get_z_coord(z_grid); );

  int    points_total = x_grid.size();
  double num_phi      = phi.size();

  std::vector<double> phi_cube(num_phi*points_total,-1);

  // compute values of level-set functions at needed points
  for (int phi_idx=0; phi_idx<num_phi; ++phi_idx)
  {
    for (int i=0; i<points_total; ++i)
    {
      phi_cube[phi_idx*points_total + i] = (*phi[phi_idx])(DIM(xyz_C[0] + x_grid[i]/scale, xyz_C[1] + y_grid[i]/scale, xyz_C[2] + z_grid[i]/scale));

      // push interfaces inside the domain
      if (fabs(phi_cube[phi_idx*points_total + i]) < perturb*diag)
      {
        phi_cube[phi_idx*points_total + i] = perturb*diag;
      }
    }
  }

  std::vector<int> clr(num_phi);
  for (int i=0; i<num_phi; ++i) clr[i] = i;

  // reconstruct geometry
  reconstruct_cube(cube, phi_cube, opn, clr);

  // get quadrature points
  _CODE( std::vector<double> qp_w );
  XCODE( std::vector<double> qp_x );
  YCODE( std::vector<double> qp_y );
  ZCODE( std::vector<double> qp_z );

  cube.quadrature_over_domain(qp_w, DIM(qp_x, qp_y, qp_z));

  // compute cut-cell volume
  fv.volume = 0;
  for (size_t i=0; i<qp_w.size(); ++i)
    fv.volume += qp_w[i];

  fv.volume /= pow(scale, P4EST_DIM);

  // compute areas and centroinds of interfaces
  fv.interfaces.clear();

  for (int phi_idx=0; phi_idx<num_phi; ++phi_idx)
  {
    cube.quadrature_over_interface(phi_idx, qp_w, DIM(qp_x, qp_y, qp_z));
    if (qp_w.size() > 0)
    {
      interface_info_t data;

      _CODE( data.id          = phi_idx );
      _CODE( data.area        = 0 );
      XCODE( data.centroid[0] = 0 );
      YCODE( data.centroid[1] = 0 );
      ZCODE( data.centroid[2] = 0 );

      for (size_t i=0; i<qp_w.size(); ++i)
      {
        _CODE( data.area        += qp_w[i]         );
        XCODE( data.centroid[0] += qp_w[i]*qp_x[i] );
        YCODE( data.centroid[1] += qp_w[i]*qp_y[i] );
        ZCODE( data.centroid[2] += qp_w[i]*qp_z[i] );
      }

      XCODE( data.centroid[0] /= scale*data.area);
      YCODE( data.centroid[1] /= scale*data.area);
      ZCODE( data.centroid[2] /= scale*data.area);
      _CODE( data.area        /= pow(scale, P4EST_DIM-1) );

      fv.interfaces.push_back(data);
    }
  }

  // compute cut-face areas and their centroids
  for (int dir_idx=0; dir_idx<P4EST_FACES; ++dir_idx)
  {
    _CODE( fv.face_area      [dir_idx] = 0 );
    XCODE( fv.face_centroid_x[dir_idx] = 0 );
    YCODE( fv.face_centroid_y[dir_idx] = 0 );
    ZCODE( fv.face_centroid_z[dir_idx] = 0 );

    cube.quadrature_in_dir(dir_idx, qp_w, DIM(qp_x, qp_y, qp_z));
    if (qp_w.size() > 0)
    {
      for (size_t i=0; i<qp_w.size(); ++i) fv.face_area[dir_idx] += qp_w[i];

      if (compute_centroids)
      {
        for (size_t i=0; i<qp_w.size(); ++i)
        {
          XCODE( fv.face_centroid_x[dir_idx] += qp_w[i]*qp_x[i] );
          YCODE( fv.face_centroid_y[dir_idx] += qp_w[i]*qp_y[i] );
          ZCODE( fv.face_centroid_z[dir_idx] += qp_w[i]*qp_z[i] );
        }

        XCODE( fv.face_centroid_x[dir_idx] /= scale*fv.face_area[dir_idx] );
        YCODE( fv.face_centroid_y[dir_idx] /= scale*fv.face_area[dir_idx] );
        ZCODE( fv.face_centroid_z[dir_idx] /= scale*fv.face_area[dir_idx] );
      }

      fv.face_area[dir_idx] /= pow(scale, P4EST_DIM-1);
    }
  }

  XCODE( fv.face_centroid_x[dir::f_m00] = cube_xyz_min[0]/scale; fv.face_centroid_x[dir::f_p00] = cube_xyz_max[0]/scale );
  YCODE( fv.face_centroid_y[dir::f_0m0] = cube_xyz_min[1]/scale; fv.face_centroid_y[dir::f_0p0] = cube_xyz_max[1]/scale );
  ZCODE( fv.face_centroid_z[dir::f_00m] = cube_xyz_min[2]/scale; fv.face_centroid_z[dir::f_00p] = cube_xyz_max[2]/scale );
}

void compute_wall_normal(const int &dir, double normal[])
{
  switch (dir)
  {
    case dir::f_m00: normal[0] =-1; normal[1] = 0; CODE3D( normal[2] = 0;) break;
    case dir::f_p00: normal[0] = 1; normal[1] = 0; CODE3D( normal[2] = 0;) break;

    case dir::f_0m0: normal[0] = 0; normal[1] =-1; CODE3D( normal[2] = 0;) break;
    case dir::f_0p0: normal[0] = 0; normal[1] = 1; CODE3D( normal[2] = 0;) break;
#ifdef P4_TO_P8
    case dir::f_00m: normal[0] = 0; normal[1] = 0; CODE3D( normal[2] =-1;) break;
    case dir::f_00p: normal[0] = 0; normal[1] = 0; CODE3D( normal[2] = 1;) break;
#endif
    default:
      throw std::invalid_argument("Invalid direction\n");
  }
}

double interface_point_cartesian_t::interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr)
{
  const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

  p4est_locidx_t neigh = qnnn.neighbor(dir);
  double         h     = qnnn.distance(dir);

  return (ptr[n]*(h-dist) + ptr[neigh]*dist)/h;
}

double interface_point_cartesian_t::interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr, double *ptr_dd[P4EST_DIM])
{
  const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

  p4est_locidx_t neigh = qnnn.neighbor(dir);
  double         h     = qnnn.distance(dir);
  short          dim   = dir / 2;

  double p0  = ptr[n];
  double p1  = ptr[neigh];
  double pdd = MINMOD(ptr_dd[dim][n], ptr_dd[dim][neigh]);

  return .5*(p0+p1) + (p1-p0)*(dist/h-.5) + .5*pdd*(dist*dist-dist*h);
}

PetscErrorCode vec_and_ptr_t::ierr;
PetscErrorCode vec_and_ptr_dim_t::ierr;
PetscErrorCode vec_and_ptr_array_t::ierr;

// Generalized smoothstep

// Returns binomial coefficient without explicit use of factorials,
// which can't be used with negative integers
double pascalTriangle(int a, int b) {
  double result = 1.;
  for (int i = 0; i < b; ++i)
    result *= double(a - i) / double(i + 1);
  return result;
}

double clamp(double x, double lowerlimit, double upperlimit)
{
  if (x < lowerlimit) x = lowerlimit;
  if (x > upperlimit) x = upperlimit;
  return x;
}

double smoothstep(int N, double x) {
  x = clamp(x, 0, 1); // x must be equal to or between 0 and 1
  double result = 0;
  for (int n = 0; n <= N; ++n)
  {
    result += pascalTriangle(-N - 1, n) *
              pascalTriangle(2 * N + 1, N - n) *
              pow(x, N + n + 1);
  }
  return result;
}

void variable_step_BDF_implicit(const int order, std::vector<double> &dt, std::vector<double> &coeffs)
{
  coeffs.assign(order+1, 0);
  std::vector<double> r(order-1, 1.);

  if (dt.size() < (size_t) order) throw;

  switch (order)
  {
    case 1:
      coeffs[0] =  1;
      coeffs[1] = -1;
      break;
    case 2:
      r[0] = dt[0]/dt[1];

      coeffs[0] = (1.+2.*r[0])/(1.+r[0]);
      coeffs[1] = -(1.+r[0]);
      coeffs[2] = SQR(r[0])/(1.+r[0]);
      break;
    case 3:
      r[0] = dt[0]/dt[1];
      r[1] = dt[1]/dt[2];

      coeffs[0] = 1. + r[0]/(1.+r[0]) + r[1]*r[0]/(1.+r[1]*(1.+r[0]));
      coeffs[1] = -1. - r[0] - r[0]*r[1]*(1.+r[0])/(1.+r[1]);
      coeffs[2] = SQR(r[0])*(r[1] + 1./(1.+r[0]));
      coeffs[3] = - pow(r[1], 3.)*pow(r[0], 2)*(1.+r[0])/(1.+r[1])/(1.+r[1]+r[1]*r[0]);
      break;
    case 4:
    {
      r[0] = dt[0]/dt[1];
      r[1] = dt[1]/dt[2];
      r[2] = dt[2]/dt[3];

      double a1 = 1.+r[2]*(1.+r[1]);
      double a2 = 1.+r[1]*(1.+r[0]);
      double a3 = 1.+r[2]*a2;

      coeffs[0] = 1. + r[0]/(1.+r[0]) + r[1]*r[0]/a2 + r[2]*r[1]*r[0]/a3;
      coeffs[1] = -1.-r[0]*(1.+r[1]*(1.+r[0])/(1.+r[1])*(1.+r[2]*a2/a1));
      coeffs[2] = r[0]*(r[0]/(1.+r[0]) + r[1]*r[0]*(a3+r[2])/(1.+r[2]));
      coeffs[3] = -pow(r[1],3.)*pow(r[0],2.)*(1.+r[0])/(1.+r[1])*a3/a2;
      coeffs[4] = (1.+r[0])/(1+r[2])*a2/a1*pow(r[2],4.)*pow(r[1],3.)*pow(r[0],2.)/a3;
    }
      break;
    default:
      throw;
  }
}
