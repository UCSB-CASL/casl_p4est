// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// p4est Library
#include <p4est_extended.h>
#include <p4est_bits.h>
#include <p4est_nodes.h>

// casl_p4est
#include <src/my_p4est_vtk.h>
#include <src/poisson_solver.h>
#include <src/refine_coarsen.h>
#include <src/utils.h>
#include <src/petsc_compatibility.h>

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  circle circ(1, 1, .3);
  splitting_criteria_cf_t cf_data   = {&circ, 8, 4, 1};

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  // Create the connectivity object
  w2.start("connectivity");
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  // Now refine the tree
  w2.start("refine");
  p4est->user_pointer = (void*)(&cf_data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  w2.stop(); w2.read_duration();

  p4est_balance(p4est, P4EST_CONNECT_DEFAULT, NULL);

  // Finally re-partition
  w2.start("partition");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  w2.start("creating node structure");
  nodes = my_p4est_nodes_new(p4est);
  w2.stop(); w2.read_duration();

  // Now lets solve a poisson equation
  struct:CF_2{
    double operator()(double x, double y) const {
      return sin(2*M_PI*x)*sin(2*M_PI*y);
    }
  } uex;

  struct:CF_2{
    double operator()(double x, double y) const {
      return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);
    }
  } f;

  w2.start("petsc");
  Vec sol, sol_ex;
  PoissonSolver solver(p4est, uex, f);
  solver.setUpNegativeLaplaceSystem();
  solver.solve(sol, sol_ex);
  w2.stop(); w2.read_duration();

  // Get a pointer to the solution
  double *sol_ptr, *sol_ex_ptr;

  ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

  w2.start("vtk");
  ostringstream oss;
  oss << "poisson_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, 1.0,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 2, oss.str().c_str(),
                         VTK_CELL_DATA, "sol", sol_ptr,
                         VTK_CELL_DATA, "sol_ex", sol_ex_ptr);

  w2.stop(); w2.read_duration();

  ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

  // compute the maximum error and check convergence
  ierr = VecAXPY(sol, -1.0, sol_ex); CHKERRXX(ierr);
  double err_max;
  ierr = VecNorm(sol, NORM_INFINITY, &err_max); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi->mpicomm, "Maximum err = %1.4e\n", err_max); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy(nodes);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}
