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

//#include "serial_semi_lagrangian.h"
#include "parallel_semi_lagrangian.h"

using namespace std;

class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return -0.15*sin(M_PI*x/2)*sin(M_PI*x/2)*sin(2*M_PI*y/2);
  }
} vx_vortex;

class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  0.15*sin(M_PI*y/2)*sin(M_PI*y/2)*sin(2*M_PI*x/2);
  }
} vy_vortex;

struct:CF_2{
  double operator()(double x, double y) const {
    return 0.3;
  }
} vx_translate;

struct:CF_2{
  double operator()(double x, double y) const {
    return 0.3;
  }
} vy_translate;

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
  vector<double> phi(nodes->num_owned_indeps);
  vector<double> vx, vy;
  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi[i] = circ(x,y);
  }

  // write the intial data to disk
  my_p4est_vtk_write_all(p4est, NULL, 1.0,
                         1, 0, "init",
                         VTK_POINT_DATA, "phi", &phi[0]);

  // SemiLagrangian object
  //serial::SemiLagrangian sl(&p4est, &nodes);
  parallel::SemiLagrangian sl(&p4est, &nodes);

  // loop over time
  double tf = 10;
  int tc = 0;
  int save = 1;
  for (double t=0, dt=0; t<tf; t+=dt, tc++){
    if (tc % save == 0){
      // Save stuff
      std::ostringstream oss; oss << brick.nxytrees[0] << "x"
                                  << brick.nxytrees[1] << "." << tc/save;

      vx.resize(phi.size());
      vy.resize(phi.size());

      for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
      {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

        vx[i] = vx_vortex(x,y);
        vy[i] = vy_vortex(x,y);
      }

      my_p4est_vtk_write_all(p4est, NULL, 1.0,
                             3, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", &phi[0],
                             VTK_POINT_DATA, "vx", &vx[0],
                             VTK_POINT_DATA, "vy", &vy[0]);
    }

    // advect the function in time and get the computed time-step
    dt = sl.advect(vx_vortex, vy_vortex, phi);
  }
  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  return 0;
}
