// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double r, x0, y0;
};

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  circle circ(0.5, 0.5, .3);
  cf_grid_data_t data = {&circ, 7, 0, 1.0};

  Session::init(argc, argv, mpi->mpicomm);

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
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est);

  // Initialize the level-set function
  Vec phi_global;
  ierr = VecCreateGhost(p4est, nodes, &phi_global); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_global, &phi); CHKERRXX(ierr);

  // compute the level-set
  for (p4est_locidx_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi[p4est2petsc_local_numbering(nodes,i)] = circ(x,y);
  }

//  // lets save the processor rank and tree index for visualization purposes
//  std::vector<float> proc_ranks(p4est->local_num_quadrants, p4est->mpirank);
//  std::vector<float> tree_idxs(p4est->local_num_quadrants);
//  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it <= p4est->last_local_tree; ++tr_it)
//  {
//    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);
//    for (p4est_locidx_t qu_it = 0; qu_it<tree->quadrants.elem_count; ++qu_it)
//      tree_idxs[qu_it + tree->quadrants_offset]  = tr_it;
//  }

  // write the intial data to disk
//  my_p4est_vtk_write_all(p4est, nodes, 1.0,
//                         1, 2, "partition",
//                         VTK_POINT_DATA, "phi", phi,
//                         VTK_CELL_DATA,  "proc_rank", &proc_ranks[0],
//                         VTK_CELL_DATA,  "tree_idx",  &tree_idxs[0]);

  my_p4est_vtk_write_all(p4est, nodes, 1.0,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "partition",
                         VTK_POINT_DATA, "phi", phi);

  ierr = VecRestoreArray(phi_global, &phi); CHKERRXX(ierr);
  ierr = VecDestroy(phi_global); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  cout.flush();

  return 0;
}

