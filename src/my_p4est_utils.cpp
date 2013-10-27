#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#include "my_p8est_tools.h"
#include "cube3.h"
#else
#include "my_p4est_utils.h"
#include "my_p4est_tools.h"
#include "cube2.h"
#endif

#include "mpi.h"
#include <vector>
#include <petsclog.h>
#include <src/CASL_math.h>

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

double linear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global)
{  
  PetscErrorCode ierr;
  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];

  double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
#endif

  double x = (xyz_global[0] - tree_xmin);
  double y = (xyz_global[1] - tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin);
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

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value /= qh*qh*qh;
#else
  value /= qh*qh;
#endif

  ierr = PetscLogFlops(39); CHKERRXX(ierr); // number of flops in this event

  return value;
}

double quadratic_non_oscillatory_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global)
{
  PetscErrorCode ierr;
  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];

  double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
#endif

  double x = (xyz_global[0] - tree_xmin);
  double y = (xyz_global[1] - tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin);
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

  double fdd[P4EST_DIM];
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = Fdd[i];

  for (short j = 1; j<P4EST_CHILDREN; j++)
    for (short i = 0; i<P4EST_DIM; i++)
      fdd[i] = MINMOD(fdd[i], Fdd[j*P4EST_DIM + i]);

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value /= (qh*qh*qh);
  value -= 0.5*(d_p00*d_m00*fdd[0] + d_0p0*d_0m0*fdd[1] + d_00p*d_00m*fdd[2]);
#else
  value /= (qh*qh);
  value -= 0.5*(d_p00*d_m00*fdd[0] + d_0p0*d_0m0*fdd[1]);
#endif

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return value;
}

double quadratic_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global)
{
  PetscErrorCode ierr;
  p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[tree_id*P4EST_CHILDREN + 0];

  double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
#endif

  double x = (xyz_global[0] - tree_xmin);
  double y = (xyz_global[1] - tree_ymin);
#ifdef P4_TO_P8
  double z = (xyz_global[2] - tree_zmin);
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

  double fdd[P4EST_DIM];
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = 0;

  for (short j=0; j<P4EST_CHILDREN; j++)
    for (short i = 0; i<P4EST_DIM; i++)
      fdd[i] += Fdd[j*P4EST_DIM + i] * w_xyz[j];

  for (short i=0; i<P4EST_DIM; i++)
#ifdef P4_TO_P8
    fdd[i] /= qh*qh*qh;
#else
    fdd[i] /= qh*qh;
#endif

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value /= (qh*qh*qh);
  value -= 0.5*(d_p00*d_m00*fdd[0] + d_0p0*d_0m0*fdd[1] + d_00p*d_00m*fdd[2]);
#else
  value /= (qh*qh);
  value -= 0.5*(d_p00*d_m00*fdd[0] + d_0p0*d_0m0*fdd[1]);
#endif

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return value;
}

PetscErrorCode VecCreateGhost(p4est_t *p4est, p4est_nodes_t *nodes, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;

  std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for (size_t i = 0; i<ghost_nodes.size(); ++i)
  {
    p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+num_local);
    ghost_nodes[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
  }

  ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global, ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateGhostBlock(p4est_t *p4est, p4est_nodes_t *nodes, PetscInt block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;

  std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  for (size_t i = 0; i<ghost_nodes.size(); ++i)
  {
    p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+num_local);
    ghost_nodes[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
  }

  ierr = VecCreateGhostBlock(p4est->mpicomm, block_size, num_local*block_size, num_global*block_size, ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

double integrate_over_negative_domain_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
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

  p4est_locidx_t *q2n = nodes->local_nodes;

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

  double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dy = dx;
#ifdef P4_TO_P8
  double dz = dx;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
#else
  Cube2 cube(0, dx, 0, dy);
#endif

  return cube.integral(f_values,phi_values);
}


double integrate_over_negative_domain(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
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


double area_in_negative_domain_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi)
{

#ifdef P4_TO_P8
  OctValue phi_values;
#else
  QuadValue phi_values;
#endif

  double *P;
  PetscErrorCode ierr;
  ierr = VecGetArray(phi, &P); CHKERRXX(ierr);

  p4est_locidx_t *q2n = nodes->local_nodes;

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

  double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dy = dx;
#ifdef P4_TO_P8
  double dz = dx;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
  return cube.volume_In_Negative_Domain(phi_values);
#else
  Cube2 cube(0, dx, 0, dy);
  return cube.area_In_Negative_Domain(phi_values);
#endif
}

double area_in_negative_domain(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
      sum += area_in_negative_domain_in_one_quadrant(p4est, nodes, quad,
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

double integrate_over_interface_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f)
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

  p4est_locidx_t *q2n = nodes->local_nodes;
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

  double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dy = dx;
#ifdef P4_TO_P8
  double dz = dx;
#endif

#ifdef P4_TO_P8
  Cube3 cube(0, dx, 0, dy, 0, dz);
#else
  Cube2 cube(0, dx, 0, dy);
#endif

  return cube.integrate_Over_Interface(f_values,phi_values);
}

double integrate_over_interface(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec f)
{
  double sum = 0;
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx = 0; quad_idx < tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx);
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

bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 0] != tr_it)
    return false;
  else if (ni->x == 0)
    return true;
  else
    return false;
}

bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 1] != tr_it)
    return false;
  else if (ni->x == P4EST_ROOT_LEN - 1 || ni->x == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 2] != tr_it)
    return false;
  else if (ni->y == 0)
    return true;
  else
    return false;
}

bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 3] != tr_it)
    return false;
  else if (ni->y == P4EST_ROOT_LEN - 1 || ni->y == P4EST_ROOT_LEN) // nodes may be unclamped
    return true;
  else
    return false;
}

#ifdef P4_TO_P8
bool is_node_zmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 4] != tr_it)
    return false;
  else if (ni->z == 0)
    return true;
  else
    return false;
}

bool is_node_zpWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
  const p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_topidx_t tr_it = ni->p.piggy3.which_tree;

  if (t2t[P4EST_CHILDREN*tr_it + 5] != tr_it)
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
#ifdef P4_TO_P8
void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f)
#else
void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f)
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

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = t2v[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = v2q[3*v_mm + 0];
    double tree_ymin = v2q[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_mm + 2];
#endif

    double x = node->x != P4EST_ROOT_LEN - 1 ? (double)node->x/(double)P4EST_ROOT_LEN : 1.0;
    double y = node->y != P4EST_ROOT_LEN - 1 ? (double)node->y/(double)P4EST_ROOT_LEN : 1.0;
#ifdef P4_TO_P8
    double z = node->z != P4EST_ROOT_LEN - 1 ? (double)node->z/(double)P4EST_ROOT_LEN : 1.0;
#endif

    x += tree_xmin;
    y += tree_ymin;
#ifdef P4_TO_P8
    z += tree_zmin;
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
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = t2v[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = v2q[3*v_mm + 0];
    double tree_ymin = v2q[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*v_mm + 2];
#endif

    double x = node->x != P4EST_ROOT_LEN - 1 ? (double)node->x/(double)P4EST_ROOT_LEN : 1.0;
    double y = node->y != P4EST_ROOT_LEN - 1 ? (double)node->y/(double)P4EST_ROOT_LEN : 1.0;
#ifdef P4_TO_P8
    double z = node->z != P4EST_ROOT_LEN - 1 ? (double)node->z/(double)P4EST_ROOT_LEN : 1.0;
#endif

    x += tree_xmin;
    y += tree_ymin;
#ifdef P4_TO_P8
    z += tree_zmin;
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
  else if (str == "NOINTERFACE" || str == "Nointerface" || str == "No-Interface" || str == "nointerface" || str == "no-interface")
    type = NOINTERFACE;
  else if (str == "MIXED" || str == "Mixed" || str == "mixed")
    type = MIXED;
  else
    throw std::invalid_argument("[ERROR]: Unknown BoundaryConditionType entered");

  return is;
}
