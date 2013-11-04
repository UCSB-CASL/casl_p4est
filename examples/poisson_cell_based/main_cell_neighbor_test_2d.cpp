// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_cell_neighbors.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_cell_neighbors.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

p4est_bool_t
refine_simple(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad)
{
  (void) which_tree;
  const splitting_criteria_t *data = (const splitting_criteria_t*)p4est->user_pointer;
  if (quad->level >= data->max_lvl)
    return P4EST_FALSE;

  if (quad->x == 0)
    return P4EST_TRUE;
  else
    return P4EST_FALSE;
}

#ifdef P4_TO_P8
static struct:CF_3{
  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
  double  x0, y0, z0, r;
} circle ;
#else
static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
  double  x0, y0, r;
} circle;
#endif

using namespace std;
int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode ierr;

  cmdParser cmd;
  cmd.add_option("lmin", "the min level of the tree");
  cmd.add_option("lmax", "the max level of the tree");
  cmd.parse(argc, argv);

#ifdef P4_TO_P8
  circle.update(1, 1, 1, .2);
  CF_3& cf = circle;
#else
  circle.update(1, 1, .2);
  CF_2& cf = circle;
#endif

  splitting_criteria_cf_t data(cmd.get("lmin", 0), cmd.get("lmax", 5), &cf, 1.2);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  w2.start("initializing the grid");

  /* create the macro mesh */
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(2, 2, 2, &brick);
#else
  connectivity = my_p4est_brick_new(2, 2, &brick);
#endif

  /* create the p4est */
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  /* partition the p4est */
  p4est_partition(p4est, NULL);

  /* create the ghost layer */
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  /* generate unique node indices */
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

  /* create the hierarchy structure */
  w2.start("construct the hierachy information");
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  w2.stop(); w2.read_duration();

  /* generate the cell neighborhood information */
  w2.start("construct the cell neighborhood information");
  my_p4est_cell_neighbors_t cell_neighbors(&hierarchy);

  FILE *pFile;
  ostringstream oss; oss << "cell_ngbd_" << p4est->mpirank << "_" << p4est->mpisize << ".dat";
  pFile = fopen(oss.str().c_str(), "w");
  for (size_t q = 0; q < p4est->local_num_quadrants + ghost->ghosts.elem_count; ++q)
    cell_neighbors.print_debug(q, pFile);
  fclose(pFile);

  w2.stop(); w2.read_duration();

  /* compute a function on the cells and save it as vtk */
  Vec phi;
  ierr = VecCreateGhostCells(p4est, ghost, &phi); CHKERRXX(ierr);
  sample_cf_on_cells(p4est, ghost, cf, phi);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* save the vtk file */
  oss.str(""); oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 1, oss.str().c_str(),
                         VTK_CELL_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  /* destroy PETSc vecs */
  ierr = VecDestroy(phi); CHKERRXX(ierr);

  /* destroy p4est objects */
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

