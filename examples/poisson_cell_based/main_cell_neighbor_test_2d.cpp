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

using namespace std;
int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;

  cmdParser cmd;
  cmd.add_option("lmin", "the min level of the tree");
  cmd.add_option("lmax", "the max level of the tree");
  cmd.parse(argc, argv);

  splitting_criteria_t data = {cmd.get("lmax", 3), cmd.get("lmin", 0)};

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
  p4est_refine(p4est, P4EST_TRUE, refine_simple, NULL);

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

  for (size_t q = 0; q < p4est->local_num_quadrants + ghost->ghosts.elem_count; ++q){
    PetscSynchronizedPrintf(p4est->mpicomm, " ---------- %2d ---------- \n", q);
    for (short i = 0; i<P4EST_FACES; i++){
      const p4est_locidx_t *begin = cell_neighbors.begin(q, i);
      const p4est_locidx_t *end   = cell_neighbors.end(q, i);

      PetscSynchronizedPrintf(p4est->mpicomm, " dir = %d: ", i);
      for (const p4est_locidx_t *it = begin; it != end; ++it)
        PetscSynchronizedPrintf(p4est->mpicomm, " %2d, ", *it);
      PetscSynchronizedPrintf(p4est->mpicomm, "\n");
    }
  }
  PetscSynchronizedFlush(p4est->mpicomm);

  w2.stop(); w2.read_duration();

  /* save the vtk file */
  std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, oss.str().c_str());

  /* destroy p4est objects */
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

