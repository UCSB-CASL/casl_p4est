
/*
 * Advect the level_set (initially a circle using a second-order semi-Lagrangian scheme.
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
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
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

int lmin = 3;
int lmax = 6;
int nb_splits = 1;

int nx = 2;
int ny = 3;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = true;

/*
 * 0 - circle
 */
int interface_type = 0;

BoundaryConditionType bc_itype = NEUMANN;

double diag_add = 0;

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
    switch(interface_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class NO_INTERFACE_CF : public CF_3
{
public:
  double operator()(double, double, double) const
  {
    return -1;
  }
} no_interface_cf;


#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class NO_INTERFACE_CF : public CF_2
{
public:
  double operator()(double, double) const
  {
    return -1;
  }
} no_interface_cf;
#endif


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi,
              int split)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, "/home/fgibou/code/Output/p4est_levelset");
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/levelset_"
      << "proc="
      << p4est->mpisize << "_"
         << "brick="
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "_levels=(" <<lmin << "," << lmax << ")" <<
         ".split=" << split;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



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
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("save_vtk", "save the p4est in vtk format");

  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);

  bc_itype = cmd.get("bc_itype", bc_itype);

  save_vtk = cmd.get("save_vtk", save_vtk);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz,
                                    xmin, xmax, ymin, ymax, zmin, zmax,
                                    &brick, 0, 0, 0);
#else
  connectivity = my_p4est_brick_new(nx, ny,
                                    xmin, xmax, ymin, ymax,
                                    &brick, 0, 0);
#endif

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;


  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set, 1.2);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    if(bc_itype==NOINTERFACE)
      sample_cf_on_nodes(p4est, nodes, no_interface_cf, phi);
    else
      sample_cf_on_nodes(p4est, nodes, level_set, phi);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi, EPS);

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, phi, iter);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
