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
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_log_wrappers.h>
#endif


#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

#ifdef P4_TO_P8
struct circle:CF_3{
  circle() {}
  double operator()(double x, double y, double z) const {
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    return r - sqrt(SQR(x-0.5) + SQR(y-0.5) + SQR(z-0.5));
  }
private:
  const static double r = 0.05;
};

struct circles:CF_3{
  circles(int n): x0(n), y0(n), z0(n), r(n)
  {
    for (int i=0; i<n; i++) {
      x0[i] = ranged_rand(0.0, 3.0);
      y0[i] = ranged_rand(0.0, 3.0);
      z0[i] = ranged_rand(0.0, 3.0);
      r[i]  = ranged_rand(0.0, 0.3);
    }
  }

  double operator()(double x, double y, double z) const {
    double f = -DBL_MAX;
    for (size_t i=0; i<r.size(); i++)
      f = MAX(f, r[i] - sqrt(SQR(x - x0[i]) + SQR(y - y0[i]) + SQR(z - z0[i])));
    return f;
  }
private:
  std::vector<double> x0, y0, z0, r;
};
#else
struct circle:CF_2{
  circle() {}
  double operator()(double x, double y) const {
    x -= floor(x);
    y -= floor(y);

    return r - sqrt(SQR(x-0.5) + SQR(y-0.5));
  }
private:
  const static double r = 0.05;
};

struct circles:CF_2{
  circles(int n): x0(n), y0(n), r(n)
  {
    for (int i=0; i<n; i++) {
      x0[i] = ranged_rand(0.0, 3.0);
      y0[i] = ranged_rand(0.0, 3.0);
      r[i]  = ranged_rand(0.0, 0.3);
    }
  }

  double operator()(double x, double y) const {
    double f = -DBL_MAX;
    for (size_t i=0; i<r.size(); i++)
      f = MAX(f, r[i] - sqrt(SQR(x - x0[i]) + SQR(y - y0[i])));
    return f;
  }
private:
  std::vector<double> x0, y0, r;
};
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

int main (int argc, char* argv[]){
  
	mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);
  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  try {
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    p4est_ghost_t      *ghost;
    PetscErrorCode ierr;
    cmdParser cmd;
    cmd.add_option("lmin", "min level");
    cmd.add_option("lmax", "max level");
    cmd.add_option("lip", "lip constant for refinement");
    cmd.add_option("test", "weak or strong scaling test");
    cmd.add_option("write-vtk", "pass this flag if interested in the vtk files");
    cmd.add_option("output-dir", "parent folder to save everythiong in");
		cmd.add_option("iter", "number of iterations for the reinitialization process");
    cmd.add_option("enable-qnnn-buffer", "if this flag is set qnnns are internally buffered");
    cmd.parse(argc, argv);
    cmd.print();

    const std::string foldername = cmd.get<std::string>("output-dir");
    const int lmin = cmd.get("lmin", 0);
    const int lmax = cmd.get("lmax", 7);
    const double lip = cmd.get("lip", 1.2);
    const bool write_vtk = cmd.contains("write-vtk");
    const string test = cmd.get<string>("test");
    mkdir(foldername.c_str(), 0777);

    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

#ifdef P4_TO_P8
    CF_3 *cf = NULL;
#else
    CF_2 *cf = NULL;
#endif

    circle w_cf;
    circles s_cf(100);
#ifdef P4_TO_P8
    int allowed_sizes [] = {16, 128, 432, 1024, 2000, 3456}; // 16*[1:6]^3
#else
    int allowed_sizes [] = {16, 64, 144, 256, 400, 576, 784, 1024, 1296, 1600, 1936, 2304, 2704, 3136, 3600, 4096}; // 16*[1:16]^2
#endif
    int nb = -1;

    if (test == "weak"){
      for (int i=0; i<sizeof(allowed_sizes)/sizeof(allowed_sizes[0]); i++){
        if (mpi->mpisize == allowed_sizes[i]){
          nb = i+1;
          break;
        }
      }
      if (nb == -1)
        throw invalid_argument("[ERROR]: not a valid mpi size");
      cf = &w_cf;
    } else if (test == "strong"){
      nb = 3;
      cf = &s_cf;
    } else {
      throw invalid_argument("test can only be 'weak' or 'strong'");
    }

    splitting_criteria_cf_t data(lmin, lmax, cf, lip);

    parStopWatch w1, w2;
    w1.start("total time");

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(nb, nb, nb, &brick);
#else
    connectivity = my_p4est_brick_new(nb, nb, &brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    // Note that we do non-recursive refine + partitioning to ensure that :
    // 1. the work is load-balanced (although this should not be a big deal here ...)
    // 2. and more importantly, we have enough memory for the global grid. If we do not do it this way, the code usually break around level 13 or so.
    w2.start("refine");
    p4est->user_pointer = (void*)(&data);
    for (int l=0; l<lmax; l++){
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, NULL);
    }
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    my_p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // create the ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // generate the node data structure
    nodes = my_p4est_nodes_new(p4est, ghost);

    w2.start("gather statistics");
    {
      p4est_gloidx_t num_nodes = 0;
      for (int r =0; r<p4est->mpisize; r++)
        num_nodes += nodes->global_owned_indeps[r];

      PetscPrintf(p4est->mpicomm, "%% global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
      PetscPrintf(p4est->mpicomm, "%% mpi_rank local_node_size local_quad_size ghost_node_size ghost_quad_size\n");
      PetscSynchronizedPrintf(p4est->mpicomm, "%4d, %7d, %7d, %5d, %5d\n",
                              p4est->mpirank, nodes->num_owned_indeps, p4est->local_num_quadrants, nodes->indep_nodes.elem_count-nodes->num_owned_indeps, ghost->ghosts.elem_count);
      PetscSynchronizedFlush(p4est->mpicomm);
    }
    w2.stop(); w2.read_duration();

    // Initialize the level-set function
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *cf, phi);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
		if(cmd.contains("enable-qnnn-buffer"))
      node_neighbors.init_neighbors();

    my_p4est_level_set level_set(&node_neighbors);

    w2.start("Reinit_1st_2nd");
    level_set.reinitialize_1st_order_time_2nd_order_space(phi, cmd.get("iter", 10));
    w2.stop(); w2.read_duration();

		// reset the level-set
		sample_cf_on_nodes(p4est, nodes, *cf, phi);
		w2.start("Reinit_2nd_2nd");
    level_set.reinitialize_2nd_order(phi, cmd.get("iter", 10));
    w2.stop(); w2.read_duration();

    if (write_vtk){
      w2.start("Saving vtk");
      std::ostringstream grid_name;
      grid_name << foldername << "/" << P4EST_DIM << "d_phi_" << p4est->mpisize << "_"
                << brick.nxyztrees[0] << "x"
                << brick.nxyztrees[1]
             #ifdef P4_TO_P8
                << "x" << brick.nxyztrees[2]
             #endif
                   ;

      double *phi_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             1, 0, grid_name.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p);
      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
      w2.stop(); w2.read_duration();
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_ghost_destroy(ghost);
    p4est_nodes_destroy(nodes);
    p4est_destroy(p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    PetscSynchronizedFPrintf(mpi->mpicomm, stderr, "[%d] %s\n", mpi->mpirank, e.what());
    PetscSynchronizedFlush(mpi->mpicomm);
    MPI_Abort(mpi->mpicomm, MPI_ERR_UNKNOWN);
  }

  return 0;
}
