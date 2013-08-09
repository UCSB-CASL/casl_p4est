// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>

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

void write_ghost_layer(p4est_t *p4est, p4est_ghost_t *ghost);

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;nodes;

    circle circ(1, 1, .3);
    //  rand_grid_data_t rand_data = {8, 4, 100, 20};
    cf_grid_data_t   cf_data   = {&circ, 8, 0, 1};

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
    p4est->user_pointer = (void*)(&cf_data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est);
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
    w2.stop(); w2.read_duration();

    write_ghost_layer(p4est, ghost);

    my_p4est_vtk_write_all(p4est, nodes, 1.0,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 0, "ghost_test");

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

void write_ghost_layer(p4est_t *p4est, p4est_ghost_t *ghost)
{
  stringstream filename;
  filename << "ghost_layer_p_" << p4est->mpirank << "_s_" << p4est->mpisize;

  ofstream csv ((filename.str() + ".csv").c_str());
  ofstream vtk ((filename.str() + ".vtk").c_str());

  vector<double> x(P4EST_CHILDREN*p4est->local_num_quadrants);
  vector<double> y(x);

  for (int i=0; i<ghost->ghosts.elem_count; i++)
  {
    p4est_quadrant_t *q = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, i);
    csv << q->p.piggy3.local_num << ",";

    p4est_topidx_t tree_id = q->p.piggy3.which_tree;
    p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];

    double xq = int2double_coordinate_transform(q->x) + tree_xmin;
    double yq = int2double_coordinate_transform(q->y) + tree_ymin;
    double hq = int2double_coordinate_transform(P4EST_QUADRANT_LEN(q->level));

    for (short xj=0; xj<2; ++xj)
      for (short yj=0; yj<2; ++yj){
        x[P4EST_CHILDREN*i+2*yj+xj] = xq + hq*xj;
        y[P4EST_CHILDREN*i+2*yj+xj] = yq + hq*yj;
      }
  }
  csv << endl;

  vtk << "# vtk DataFile Version 2.0 \n";
  vtk << "Quadtree Mesh \n";
  vtk << "ASCII \n";
  vtk << "DATASET UNSTRUCTURED_GRID \n" << endl;
  vtk << "POINTS " << x.size() << " double \n";
  for (int i=0; i<x.size(); i++)
    vtk << x[i] << " " << y[i] << " 0.0 \n";
  vtk << endl;

  vtk << "CELLS " << p4est->local_num_quadrants << " " << 5*p4est->local_num_quadrants << endl;
  for (int i=0; i<p4est->local_num_quadrants; ++i)
  {
    vtk << P4EST_CHILDREN << " ";
    for (short j=0; j<P4EST_CHILDREN; ++j)
      vtk << P4EST_CHILDREN*i+j << " ";
    vtk << "\n";
  }
  vtk << endl;

  vtk << "CELL_TYPES " << p4est->local_num_quadrants << endl;
  for (int i=0; i<p4est->local_num_quadrants; ++i)
    vtk << P4EST_VTK_CELL_TYPE << "\n";
  vtk << endl;

}

