#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#include "my_p8est_tools.h"
#include <p8est_connectivity.h>
#include "cube3.h"
#else
#include "my_p4est_utils.h"
#include "my_p4est_tools.h"
#include <p4est_connectivity.h>
#include "cube2.h"
#endif

#include "mpi.h"
#include <vector>
#include <set>
#include <sstream>
#include <petsclog.h>
#include <src/math.h>
#include <src/petsc_compatibility.h>

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

double linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global)
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

double quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global)
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

  double fdd[P4EST_DIM];
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = Fdd[i];

  for (short j = 1; j<P4EST_CHILDREN; j++)
    for (short i = 0; i<P4EST_DIM; i++)
      fdd[i] = MINMOD(fdd[i], Fdd[j*P4EST_DIM + i]);

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

  double sx = (tree_xmax-tree_xmin)*qh;
  double sy = (tree_ymax-tree_ymin)*qh;
#ifdef P4_TO_P8
  double sz = (tree_zmax-tree_zmin)*qh;
  value -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1] + sz*sz*d_00p*d_00m*fdd[2]);
#else
  value -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1]);
#endif

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return value;
}

double quadratic_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global)
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
  double qxmin = quad_x_fr_i(&quad);
  double qymin = quad_y_fr_j(&quad);
#ifdef P4_TO_P8
  double qzmin = quad_z_fr_k(&quad);
#endif

#ifdef CASL_THROWS
  if(x<qxmin-qh/10 || x>qxmin+qh+qh/10 || y<qymin-qh/10 || y>qymin+qh+qh/10)
  {
    std::cout << x << ", " << qxmin << ", " << qxmin+qh << std::endl;
    std::cout << y << ", " << qymin << ", " << qymin+qh << std::endl;
    std::cout << y-qymin << std::endl;
    throw std::invalid_argument("quadratic_interpolation: the point is not inside the quadrant.");
  }
#endif

  x = (x-qxmin) / qh;
  y = (y-qymin) / qh;
#ifdef P4_TO_P8
  z = (z-qzmin) / qh;
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
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = 0;

  for (short j=0; j<P4EST_CHILDREN; j++)
    for (short i = 0; i<P4EST_DIM; i++)
      fdd[i] += Fdd[j*P4EST_DIM + i] * w_xyz[j];

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

  double sx = (tree_xmax-tree_xmin)*qh;
  double sy = (tree_ymax-tree_ymin)*qh;
#ifdef P4_TO_P8
  double sz = (tree_zmax-tree_zmin)*qh;
  value -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1] + sz*sz*d_00p*d_00m*fdd[2]);
#else
  value -= 0.5*(sx*sx*d_p00*d_m00*fdd[0] + sy*sy*d_0p0*d_0m0*fdd[1]);
#endif

  ierr = PetscLogFlops(45); CHKERRXX(ierr); // number of flops in this event
  return value;
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

PetscErrorCode VecCreateGhostNodes(const p4est_t *p4est, p4est_nodes_t *nodes, Vec* v)
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

  ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global,
                        ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateGhostNodesBlock(const p4est_t *p4est, p4est_nodes_t *nodes, PetscInt block_size, Vec* v)
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

  ierr = VecCreateGhostBlock(p4est->mpicomm,
                             block_size, num_local*block_size, num_global*block_size,
                             ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateGhostCells(const p4est_t *p4est, p4est_ghost_t *ghost, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;

  std::vector<PetscInt> ghost_cells(ghost->ghosts.elem_count, 0);
  PetscInt num_global = p4est->global_num_quadrants;

  for (int r = 0; r<p4est->mpisize; ++r)
    for (p4est_locidx_t q = ghost->proc_offsets[r]; q < ghost->proc_offsets[r+1]; ++q)
    {
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
      ghost_cells[q] = (PetscInt)quad->p.piggy3.local_num + (PetscInt)p4est->global_first_quadrant[r];
    }

  ierr = VecCreateGhost(p4est->mpicomm,
                        num_local, num_global,
                        ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode VecCreateGhostCellsBlock(const p4est_t *p4est, p4est_ghost_t *ghost, PetscInt block_size, Vec* v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = p4est->local_num_quadrants;

  std::vector<PetscInt> ghost_cells(ghost->ghosts.elem_count, 0);
  PetscInt num_global = p4est->global_num_quadrants;

  for (int r = 0; r<p4est->mpisize; ++r)
    for (p4est_locidx_t q = ghost->proc_offsets[r]; q < ghost->proc_offsets[r+1]; ++q)
    {
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
      ghost_cells[q] = (PetscInt)quad->p.piggy3.local_num + (PetscInt)p4est->global_first_quadrant[r];
    }

  ierr = VecCreateGhostBlock(p4est->mpicomm,
                             block_size, num_local*block_size, num_global*block_size,
                             ghost_cells.size(), (const PetscInt*)&ghost_cells[0], v); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

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

bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni)
{
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

bool is_quad_xmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi)
{
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
