
/*
 * Test the face based p4est.
 * 1 - solve a poisson equation on an irregular domain (circle)
 * 2 - interpolate from faces to nodes
 * 3 - extrapolate faces over interface
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_faces.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin = -1;
double xmax =  1;
double ymin = -1;
double ymax =  1;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

using namespace std;

int lmin = 0;
int lmax = 4;

int nx = 2;
int ny = 2;
#ifdef P4_TO_P8
int nz = 2;
#endif

double mu = 1.;
double add_diagonal = 0;

#ifdef P4_TO_P8
double r0 = (double) MIN(xmax-xmin, ymax-ymin, zmax-zmin) / 4;
#else
double r0 = (double) MIN(xmax-xmin, ymax-ymin) / 4;
#endif



#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
  }
} level_set;

#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
  }
} level_set;

#endif


int main (int argc, char* argv[])
{
	PetscErrorCode ierr;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny, xmin, xmax, ymin, ymax, &brick);
#endif

  p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

  //    srand(1);
  //    splitting_criteria_random_t data(2, 7, 1000, 10000);
  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.6);
  p4est->user_pointer = (void*)(&data);

  //    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
  p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  ierr = PetscPrintf(mpi->mpicomm, "the tree has %d leaves\n", p4est->global_num_quadrants);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est, ghost);
	
	ierr = PetscPrintf(mpi->mpicomm, "ghost created\n"); CHKERRXX(ierr);

  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

	ierr = PetscPrintf(mpi->mpicomm, "nodes created\n"); CHKERRXX(ierr);

  my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
  my_p4est_cell_neighbors_t ngbd_c(&hierarchy);

	ierr = PetscPrintf(mpi->mpicomm, "ngbd_c created\n"); CHKERRXX(ierr);

  my_p4est_faces_t faces(p4est, ghost, &brick, &ngbd_c);

	ierr = PetscPrintf(mpi->mpicomm, "faces created\n"); CHKERRXX(ierr);

  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
