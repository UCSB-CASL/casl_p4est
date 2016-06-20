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
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/math.h>

using namespace std;

#ifdef P4_TO_P8
struct uexact_t:CF_3{
  double operator()(double x, double y, double z) const {
    return sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
  }
} uex;
#else
struct uexact_t:CF_2{
  double operator()(double x, double y) const {
    return sin(M_PI*x)*sin(M_PI*y);
  }
} uex;
#endif

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.parse(argc, argv);

    const int lmin = cmd.get("lmin", 0);
    const int lmax = cmd.get("lmax", 8);

    splitting_criteria_random_t cf_data(lmin, lmax, 0, 1000);

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, brick);
#else
    connectivity = my_p4est_brick_new(2, 2, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine and partition");
    p4est->user_pointer = (void*)(&cf_data);
    for (int l = 0; l < lmax; l++) {
      my_p4est_refine(p4est, P4EST_FALSE, refine_random, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    w2.stop(); w2.read_duration();

    // generate the node data structure    
    w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    w2.start("computing derivatives ");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    neighbors.init_neighbors();

    Vec u, uxx, uyy;
#ifdef P4_TO_P8
    Vec uzz;
#endif
    ierr = VecCreateGhostNodes(p4est, nodes, &u  ); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, uex, u);

    ierr = VecCreateGhostNodes(p4est, nodes, &uxx); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &uyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &uzz); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    neighbors.second_derivatives_central(u, uxx, uyy, uzz);
#else
    neighbors.second_derivatives_central(u, uxx, uyy);
#endif

    w2.stop(); w2.read_duration();

    // now check the correctness of
    double *u_p, *uxx_p, *uyy_p;
#ifdef P4_TO_P8
    double *uzz_p;
#endif
    ierr = VecGetArray(u,   &u_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(uxx, &uxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(uyy, &uyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(uzz, &uzz_p); CHKERRXX(ierr);
#endif

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, "gird",
                           VTK_POINT_DATA, "u", u_p);

    for (p4est_locidx_t i = 0; i < nodes->num_owned_indeps; i++) {
      const quad_neighbor_nodes_of_node_t& qnnn = neighbors[i];

      double err = 0;

      err = fabs(qnnn.f_m00_linear(uxx_p) - qnnn.dxx_central_on_m00(u_p, neighbors));
      if (err > EPS)
        cout << "err in the m00 direction! err = " << err << endl;

      err = fabs(qnnn.f_p00_linear(uxx_p) - qnnn.dxx_central_on_p00(u_p, neighbors));
      if (err > EPS)
        cout << "err in the p00 direction! err = " << err << endl;

      err = fabs(qnnn.f_0m0_linear(uyy_p) - qnnn.dyy_central_on_0m0(u_p, neighbors));
      if (err > EPS)
        cout << "err in the 0m0 direction! err = " << err << endl;

      err = fabs(qnnn.f_0p0_linear(uyy_p) - qnnn.dyy_central_on_0p0(u_p, neighbors));
      if (err > EPS)
        cout << "err in the 0p0 direction! err = " << err << endl;

#ifdef P4_TO_P8
      err = fabs(qnnn.f_00m_linear(uzz_p) - qnnn.dzz_central_on_00m(u_p, neighbors));
      if (err > EPS)
        cout << "err in the 00m direction! err = " << err << endl;

      err = fabs(qnnn.f_00p_linear(uzz_p) - qnnn.dzz_central_on_00p(u_p, neighbors));
      if (err > EPS)
        cout << "err in the 00p direction! err = " << err << endl;
#endif
    }

    ierr = VecRestoreArray(u,   &u_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(uxx, &uxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(uyy, &uyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(uzz, &uzz_p); CHKERRXX(ierr);
#endif

    // release memory
    ierr = VecDestroy(u);   CHKERRXX(ierr);
    ierr = VecDestroy(uxx); CHKERRXX(ierr);
    ierr = VecDestroy(uyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(uzz); CHKERRXX(ierr);
#endif

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

