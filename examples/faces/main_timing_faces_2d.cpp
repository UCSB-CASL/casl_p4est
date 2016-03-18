
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

typedef enum
{
  WEAK = 0,
  STRONG = 1
} scaling_type_t;

std::ostream& operator<< (std::ostream& os, scaling_type_t type)
{
  switch(type){
  case WEAK:
    os << "Weak";
    break;
  case STRONG:
    os << "Strong";
    break;
  default:
    os << "UNKNOWN";
    break;
  }
  return os;
}

std::istream& operator>> (std::istream& is, scaling_type_t& type)
{
  std::string str;
  is >> str;

  if (str == "Weak" || str == "WEAK" || str == "weak")
    type = WEAK;
  else if (str == "Strong" || str == "STRONG" || str == "strong")
    type = STRONG;
  else
    throw std::invalid_argument("[ERROR]: Unknown scaling_type_t entered");

  return is;
}

/*
 * 0 - weak scaling
 * 1 - strong scaling
 */
scaling_type_t scaling_type = STRONG;



int nx = 2;
int ny = 2;
#ifdef P4_TO_P8
int nz = 2;
#endif

double mu = 1.;
double add_diagonal = 0;
double r0;

#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(scaling_type==STRONG)
    {
      return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
    }
    else
    {
      double d = DBL_MAX;
      for(int i=0; i<nx; ++i)
        for(int j=0; j<ny; ++j)
          for(int k=0; k<nz; ++k)
            d = MIN(d, sqrt(SQR(x-(xmin+4*r0*i+2*r0)) + SQR(y-(ymin+4*r0*j+2*r0)) + SQR(z-(zmin+4*r0*k+2*r0))) - r0);
      return d;
    }
  }
} level_set;

#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(scaling_type==STRONG)
    {
      return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
    }
    else
    {
      double d = DBL_MAX;
      for(int i=0; i<nx; ++i)
        for(int j=0; j<ny; ++j)
          d = MIN(d, sqrt(SQR(x-(xmin+4*r0*i+2*r0)) + SQR(y-(ymin+4*r0*j+2*r0))) - r0);
      return d;
    }
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
  cmd.add_option("scaling", "WEAK or STRONG scaling. If choosing weak scaling, you must choose a number of processes that is a perfect square (2D) or cube (3D)");
  cmd.add_option("save_vtk", "export vtk files");
	cmd.add_option("repeat", "number of times the whole procedure is repeated");
  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  scaling_type = cmd.get("scaling", scaling_type);
  bool save_vtk = cmd.get("save_vtk", 0);
	int repeat = cmd.get("repeat", 1);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  if(scaling_type==WEAK)
  {
#ifdef P4_TO_P8
    nx = cbrt(mpi->mpisize);
    ny = nx; nz = nx;
    if(nx*nx*nx != mpi->mpisize)
      throw std::invalid_argument("you must choose a number of processes that is a perfect cube for weak scaling in 3d.");
#else
    nx = sqrt(mpi->mpisize);
    ny = nx;
    if(nx*nx != mpi->mpisize)
      throw std::invalid_argument("you must choose a number of processes that is a perfect square for weak scaling in 2d.");
#endif
    r0 = (xmax-xmin)/(double)nx / 4.;
  }
  else
  {
    nx = 2;
    ny = 2;
#ifdef P4_TO_P8
    nz = 2;
    r0 = (double) MIN(xmax-xmin, ymax-ymin, zmax-zmin) / 4;
#else
    r0 = (double) MIN(xmax-xmin, ymax-ymin) / 4;
#endif
  }

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny, xmin, xmax, ymin, ymax, &brick);
#endif

	for(int iter=0; iter<repeat; ++iter)
	{
		p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

		splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);
		p4est->user_pointer = (void*)(&data);


    for(int l=0; l<lmax; ++l)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
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

		if(iter==0 && save_vtk)
		{
      Vec leaf_level;
      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      double *p;
      ierr = VecGetArray(leaf_level, &p); CHKERRXX(ierr);
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = q + tree->quadrants_offset;
          p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          p[quad_idx] = quad->level;
        }
      }

			char name[1000];
#ifdef P4_TO_P8
			sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/scaling/vtu/scaling_faces_3d_%d", mpi->mpisize);
#else
			sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/scaling/vtu/scaling_faces_2d_%d", mpi->mpisize);
#endif
      my_p4est_vtk_write_all(p4est, nodes, NULL, P4EST_FALSE, P4EST_FALSE, 0, 1, name, VTK_CELL_DATA, "leaf_level", p);
			ierr = PetscPrintf(mpi->mpicomm, "saved visuals in %s\n", name);

      ierr = VecRestoreArray(leaf_level, &p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
		}

		p4est_nodes_destroy(nodes);
		p4est_ghost_destroy(ghost);
		p4est_destroy      (p4est);
	}
  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
