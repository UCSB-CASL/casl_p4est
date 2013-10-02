// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>

// casl_p4est
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <src/poisson_solver_node_base.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

#define DEBUG_TIMINGS

using namespace std;

static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
  double x0, y0, r;
} circle;

// Exact solution
static struct:CF_2{
    double operator()(double x, double y) const {
        return cos(2*M_PI*x)*cos(2*M_PI*y);
    }
} f_cf;

static struct:CF_2{
    double operator()(double x, double y) const {
        return -4*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
    }
} fxx_cf;

static struct:CF_2{
    double operator()(double x, double y) const {
        return -4*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
    }
} fyy_cf;

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  cmdParser cmd;
  cmd.add_option("min_level", "the min level of the tree");
  cmd.add_option("max_level", "the max level of the tree");
  cmd.add_option("nb_splits", "number of splits to apply to the min and max level");
  cmd.parse(argc, argv);

  int nb_splits, min_level, max_level;
  min_level = cmd.get("min_level", 0);
  max_level = cmd.get("max_level", 3);
  nb_splits = cmd.get("nb_splits", 0);

  circle.update(1, 1, .3);
  splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);DEBUG_TIMINGS
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

#ifdef DEBUG_TIMINGS
  w2.start("creating connectivity");
#endif
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating p4est");
#endif
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("refining p4est");
#endif
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("partitioning p4est object");
#endif
  p4est_partition(p4est, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("generating ghost data structure");
#endif
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating nodes data structure");
#endif
  nodes = my_p4est_nodes_new(p4est, ghost);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating ghosted vectors");
#endif
  Vec phi, f, fxx_ex, fyy_ex, fdd;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &f     ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &fxx_ex); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &fyy_ex); CHKERRXX(ierr);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("initializing vectors");
#endif
  sample_cf_on_nodes(p4est, nodes, circle, phi   );
  sample_cf_on_nodes(p4est, nodes, f_cf,   f     );
  sample_cf_on_nodes(p4est, nodes, fxx_cf, fxx_ex);
  sample_cf_on_nodes(p4est, nodes, fyy_cf, fyy_ex);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("constructing p4est hierarchy");
#endif
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("constructing node neighboring information");
#endif
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("Computing derivatives the new way");
#endif
  // compute the derivatives new way
  ierr = VecCreateGhostBlock(p4est, nodes, 2, &fdd); CHKERRXX(ierr);
  node_neighbors.dxx_and_dyy_central(f, fdd);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

  /* Elements in a block ghosted vectors are stored with a stride equal to
   * the block size. This means here items in fdd are stored as
   * fxx[0], fyy[0], fxx[1], fyy[1], ..., fxx[N], fyy[N]
   * This needed to ensure effcient data transfer and also improves cache
   * performance as most likely one needs both component of fdd at the same time
   * for a given index.
   *
   * However, currently this is not compatible with the vtk writer. We could
   * either change the vtk writer to directly work with PETSc objects or write
   * a wrapper vector that takes stride into account. The third way, which is
   * not optimal, is to copy stuff into a separate vector which we do here.
   */

  double *fdd_p;
  ierr = VecGetArray(fdd, &fdd_p); CHKERRXX(ierr);
  std::vector<double> fxx(nodes->indep_nodes.elem_count);
  std::vector<double> fyy(nodes->indep_nodes.elem_count);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
    fxx[i] = fdd_p[2*i+0];
    fyy[i] = fdd_p[2*i+1];
  }
  ierr = VecRestoreArray(fdd, &fdd_p); CHKERRXX(ierr);

  // done. lets write levelset and solutions
  double *phi_p, *f_p, *fxx_ex_p, *fyy_ex_p;
  ierr = VecGetArray(phi,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(f,      &f_p     ); CHKERRXX(ierr);
  ierr = VecGetArray(fxx_ex, &fxx_ex_p); CHKERRXX(ierr);
  ierr = VecGetArray(fyy_ex, &fyy_ex_p); CHKERRXX(ierr);

  std::ostringstream oss; oss << "derivatives_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         6, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi",    phi_p,
                         VTK_POINT_DATA, "f",      f_p  ,
                         VTK_POINT_DATA, "fxx",    &fxx[0],
                         VTK_POINT_DATA, "fyy",    &fyy[0],
                         VTK_POINT_DATA, "fxx_ex", fxx_ex_p,
                         VTK_POINT_DATA, "fyy_ex", fyy_ex_p);
  // restore pointers
  ierr = VecRestoreArray(phi,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(f,      &f_p     ); CHKERRXX(ierr);
  ierr = VecRestoreArray(fxx_ex, &fxx_ex_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fyy_ex, &fyy_ex_p); CHKERRXX(ierr);

  // finally, delete PETSc Vecs by calling 'VecDestroy' function
  ierr = VecDestroy(phi);    CHKERRXX(ierr);
  ierr = VecDestroy(f);      CHKERRXX(ierr);
  ierr = VecDestroy(fdd);    CHKERRXX(ierr);
  ierr = VecDestroy(fxx_ex); CHKERRXX(ierr);
  ierr = VecDestroy(fyy_ex); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();
  return 0;
}
