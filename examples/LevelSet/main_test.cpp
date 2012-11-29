// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/semi_lagrangian.h>
#include <src/refine_coarsen.h>

using namespace std;

int refine_dummy(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad){
  if ((quad->level == 0 && which_tree == 0) ||
      (quad->level == 0 && which_tree == 3) )

    return P4EST_TRUE;
  return P4EST_FALSE;
}

struct Circle:CF_2{
  Circle(double r_)
    : r(r_)
  {}
  void update(double r_){
    r = r_;
  }

  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-1.0) + SQR(y-1.0));
  }

private:
  double r;
};

int main (int argc, char* argv[]){

  mpi_context_t       mpi_context, *mpi = &mpi_context;
  mpi->mpicomm = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t            *p4est;

  Circle circle(0.5);
  refine_coarsen_data_t data = {&circle, 6, 0, 1.0};

  Session session(argc, argv);
  session.init(mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


  // Create the connectivity object
  w2.start("connectivity");
  connectivity = p4est_connectivity_new_brick (2, 2, 0, 0);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();
  p4est->user_pointer = (void*)(&data);

  w2.start("refinement/coarsening");
  //  p4est_refine (p4est, P4EST_TRUE, refine_dummy , NULL);
    p4est_refine (p4est, P4EST_TRUE, refine_levelset , NULL);
  //  p4est_coarsen(p4est, P4EST_TRUE, coarsen_levelset, NULL);
  w2.stop(); w2.read_duration();

  w2.start("grid partitioning");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  my_p4est_vtk_write_all(p4est, NULL, 1.0,
                         0, 0, "grid");

  // Now lets construct points and see if the function works correctly
  my_p4est_nodes_t *nodes = my_p4est_nodes_new(p4est);
  ArrayV<p4est_gloidx_t> gloidx_offset(p4est->mpisize + 1); gloidx_offset = 0;
  p4est_locidx_t *e2n = nodes->local_nodes;
  for (int n=0; n<p4est->mpisize; ++n)
    gloidx_offset(n+1) = gloidx_offset(n) + nodes->global_owned_indeps[n];

  ArrayV<p4est_gloidx_t> failed;
  ostringstream failed_oss;
  for (p4est_locidx_t ni = 0; ni<nodes->num_owned_indeps; ++ni){
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, ni + nodes->offset_owned_indeps);

    double xy [] =
    {
      (double)node->x/(double)P4EST_ROOT_LEN,
      (double)node->y/(double)P4EST_ROOT_LEN
    };
    c2p_coordinate_transform(p4est, node->p.piggy3.which_tree, &xy[0], &xy[1], NULL);

    // find the quadrant
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;
    p4est_locidx_t quad_locidx;
    p4est_quadrant_t *quad;
    if (p4est->mpirank != my_p4est_brick_point_lookup_smallest(p4est, xy, &tree_id, &quad_locidx, &quad))
      throw runtime_error("[CASL_ERROR]: Not a local cell!");

    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_id);
    quad_locidx += tree->quadrants_offset;

    double xyq [] =
    {
      (double)quad->x/(double)P4EST_ROOT_LEN,
      (double)quad->y/(double)P4EST_ROOT_LEN
    };
    c2p_coordinate_transform(p4est, tree_id, &xyq[0], &xyq[1], NULL);
    double dxyq[2];
    dx_dy_dz_quadrant(p4est, tree_id, quad, &dxyq[0], &dxyq[1], NULL);

    if (ni != e2n[quad_locidx*P4EST_CHILDREN + 0] - nodes->offset_owned_indeps &&
        ni != e2n[quad_locidx*P4EST_CHILDREN + 1] - nodes->offset_owned_indeps &&
        ni != e2n[quad_locidx*P4EST_CHILDREN + 2] - nodes->offset_owned_indeps &&
        ni != e2n[quad_locidx*P4EST_CHILDREN + 3] - nodes->offset_owned_indeps )
    {
      failed.push(gloidx_offset(p4est->mpirank) + ni);

      failed_oss << "node " << gloidx_offset(p4est->mpirank) + ni << " (" << xy[0] << "," << xy[1] << ") " << endl << "  ";
      failed_oss << "cell " << quad_locidx + p4est->global_first_quadrant[p4est->mpirank]
                 << " (" << xyq[0] << ", " << xyq[1] << ")" << endl << "  ";
      failed_oss << "nodes on this cell are: "
                 << e2n[quad_locidx*P4EST_CHILDREN + 0] - nodes->offset_owned_indeps + gloidx_offset(p4est->mpirank) << " (" << xyq[0]         << ", " << xyq[1]         << ") -- "
                 << e2n[quad_locidx*P4EST_CHILDREN + 1] - nodes->offset_owned_indeps + gloidx_offset(p4est->mpirank) << " (" << xyq[0]+dxyq[0] << ", " << xyq[1]         << ") -- "
                 << e2n[quad_locidx*P4EST_CHILDREN + 2] - nodes->offset_owned_indeps + gloidx_offset(p4est->mpirank) << " (" << xyq[0]         << ", " << xyq[1]+dxyq[1] << ") -- "
                 << e2n[quad_locidx*P4EST_CHILDREN + 3] - nodes->offset_owned_indeps + gloidx_offset(p4est->mpirank) << " (" << xyq[0]+dxyq[0] << ", " << xyq[1]+dxyq[1] << ") " << endl;
    }
  }

  if (failed.size() == 0)
    cout << endl << "All nodes passed the test!" << endl << endl;
  else
  {
    cout << endl << failed.size() << " nodes failed the test. These are:" << endl << "  ";
    for (int n= 0; n<failed.size(); n++)
      cout << failed(n) << ", ";
    cout << endl << "Below you can find more details:" << endl;

    cout << failed_oss.str() << endl;
  }

  // destroy the p4est and its connectivity structure
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);
  my_p4est_nodes_destroy(nodes);

  w1.stop(); w1.read_duration();

  return 0;
}


