// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>
using namespace std;

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  cmdParser cmd;
  cmd.add_option("lmin", "min level");
  cmd.add_option("lmax", "max level");
  cmd.parse(argc, argv);

  const int lmin = cmd.get("lmin", 0);
  const int lmax = cmd.get("lmax", 8);
  splitting_criteria_random_t sp_data(lmin, lmax, 100, 1000);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  // Create the connectivity object
  w2.start("connectivity");
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(1, 1, 1, &brick);
#else
  connectivity = my_p4est_brick_new(1, 1, &brick);
#endif
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  // Now refine the tree
  w2.start("refine and partition");
  p4est->user_pointer = (void*)(&sp_data);
  for (int i = 0; i < lmax; i++){
    my_p4est_refine(p4est, false, refine_random, NULL);
    my_p4est_partition(p4est, true, NULL);
  }
  w2.stop(); w2.read_duration();

  // generate the ghost data-structure
  w2.start("generating ghost data structure");
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  w2.start("creating nodes data structure");
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

  // write the grid
  std::ostringstream oss; oss << P4EST_DIM << "d_random_" << p4est->mpisize << "_0";
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, oss.str().c_str());

  // lets coarsen up
  for (int i = 0; i<lmax; i++) {
    my_p4est_coarsen(p4est, false, coarsen_every_cell, NULL);
    my_p4est_partition(p4est, true, NULL);
    p4est_ghost_destroy(ghost);
    p4est_nodes_destroy(nodes);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    nodes = my_p4est_nodes_new(p4est, ghost);

    std::ostringstream oss; oss << P4EST_DIM << "d_random_" << p4est->mpisize << "_" << i + 1;
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 0, oss.str().c_str());
  }

  // destroy the p4est, ghost, and nodes
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);

  // Now try refinement based on distance

  w2.start("refine and partition");
  p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  splitting_criteria_random_t sp_data2(lmin, lmax, 1000, 10000);
  p4est->user_pointer = (void*)(&sp_data2);
  for (int i = 0; i < lmax; i++){
    my_p4est_refine(p4est, false, refine_random, NULL);
    my_p4est_partition(p4est, true, NULL);
  }
  w2.stop(); w2.read_duration();

  w2.start("generating ghost data structure");
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  w2.start("creating nodes data structure");
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.25 - sqrt(SQR(x - 0.35) + SQR(y - 0.35) + SQR(z - 0.35));
    }
  } ls_cf;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.25 - sqrt(SQR(x - 0.35) + SQR(y - 0.35));
    }
  } ls_cf;
#endif

  Vec phi;
  double *phi_p;  

  int counter = 0;
  while (true) {
    ostringstream wss;
    wss << "refining iteration " << counter;
    w2.start(wss.str());

    VecCreateGhostNodes(p4est, nodes, &phi);
    sample_cf_on_nodes(p4est, nodes, ls_cf, phi);
    VecGetArray(phi, &phi_p);

    // write the vtk
    std::ostringstream oss; oss << P4EST_DIM << "d_tagged_" << p4est->mpisize << "_" << counter;
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);    

    // reset nodes and ghost    
    splitting_criteria_tag_t sp(lmin, lmax, 1.2);
    bool is_grid_changed = sp.refine_and_coarsen(p4est, nodes, phi_p);

    unsigned checksum = p4est_checksum(p4est);
    PetscPrintf(p4est->mpicomm, "Checksum = %u\n", checksum);

    VecRestoreArray(phi, &phi_p);

    if (is_grid_changed) {
      my_p4est_partition(p4est, true, NULL);
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);
      VecDestroy(phi);
    }

    p4est_gloidx_t num_nodes = 0;
      for (int r =0; r<p4est->mpisize; r++)
        num_nodes += nodes->global_owned_indeps[r];

    PetscPrintf(p4est->mpicomm, "global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
    counter++;
    w2.stop(); w2.read_duration();

    if (!is_grid_changed)
      break;
  }

  oss.str("");
  VecCreateGhostNodes(p4est, nodes, &phi);
  sample_cf_on_nodes(p4est, nodes, ls_cf, phi);
  VecGetArray(phi, &phi_p);
  oss << P4EST_DIM << "d_tagged_" << p4est->mpisize << "_" << counter;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);
  VecRestoreArray(phi, &phi_p);

  // free all memory
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

