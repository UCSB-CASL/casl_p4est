// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <p8est_communication.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/point3.h>
#include "charging_linear_implicit_3d.h"
#include "charging_linear_explicit_3d.h"
#include "charging_nonlinear_explicit_3d.h"
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_communication.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/point2.h>
#include "charging_linear_implicit_2d.h"
#include "charging_linear_explicit_2d.h"
#include "charging_nonlinear_explicit_2d.h"
#endif

#include <src/ipm_logging.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

using namespace std;

#ifdef P4_TO_P8
class Interface:public CF_3{
public:
  double operator ()(double x, double y, double z) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5) + SQR(z - 0.5));
  }
} sphere;
#else
class Interface:public CF_2{
public:
  double operator ()(double x, double y) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5));
  }
} sphere;
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

std::string output_dir;

#ifdef P4_TO_P8
class constant_cf: public CF_3{
  double c;
public:
  constant_cf(double c_): c(c_) {}
  inline void set(double c_) { c = c_; }
  double operator ()(double /* x */, double /* y */, double /* z */) const {
    return c;
  }
};
#else
class constant_cf: public CF_2{
  double c;
public:
  constant_cf(double c_): c(c_) {}
  inline void set(double c_) { c = c_; }
  double operator ()(double /* x */, double /* y */) const {
    return c;
  }
};
#endif

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("output-dir", "address of the output directory for all I/O");
    cmd.parse(argc, argv);
    cmd.print();

    output_dir       = cmd.get<std::string>("output-dir", ".");
    const int lmin   = cmd.get("lmin", 6);
    const int lmax   = cmd.get("lmax", 11);

    parStopWatch w1;//(parStopWatch::all_timings);
    parStopWatch w2;//(parStopWatch::all_timings);
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Print the SHA1 of the current commit
    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

    // print basic information
    PetscPrintf(mpi->mpicomm, "mpisize = %d\n", mpi->mpisize);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(4, 1, 1, brick);
#else
    connectivity = my_p4est_brick_new(4, 1, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    splitting_criteria_cf_t sp(lmin, lmax, &sphere, 1.2);
    p4est->user_pointer = &sp;
    for (int l=0; l<lmax; l++){
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_TRUE, NULL);
    }
    w2.stop(); w2.read_duration();

    w2.start("nodes and ghost construction");
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    // make the level-set signed distance
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, sphere, phi);
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, "grid",
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();

    ExplicitNonLinearChargingSolver solver(p4est, ghost, nodes, brick, &ngbd);
    solver.set_parameters(1e-5, 0.1, 1.0);
    solver.set_phi(phi);
    solver.init();

    for (int i=0; i<30000; i++){
      ostringstream oss;
      oss << "solving iteration " << i;
      w2.start(oss.str());

      solver.solve();
      oss.str(""); oss << output_dir + "/solution_explicit." << i;
      solver.write_vtk(oss.str());

      w2.stop(); w2.read_duration();
    }

    // free memory
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    p4est_destroy(p4est);
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}
