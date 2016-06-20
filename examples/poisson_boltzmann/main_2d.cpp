/* 
 *
 * Title: poisson_boltzmann
 * Description:
 * Author: 
 * Date Created: 
 *
*/

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_boltzmann_nodes.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_boltzmann_nodes.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

using namespace std;

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: poisson_boltzmann");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz []      = {1, 1, 1};
  const double xyz_min [] = {-1, -1, -1};
  const double xyz_max [] = { 1,  1,  1};
  const int periodic []   = {0, 0, 0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to circle
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.30 - sqrt(SQR(x)+SQR(y)+SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.30 - sqrt(SQR(x)+SQR(y));
    }
  } circle;
#endif
  splitting_criteria_cf_t sp(3, 8, &circle);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // create neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);

  // define solution vectors
  Vec phi, sol;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecDuplicate(phi, &sol);
  sample_cf_on_nodes(p4est, nodes, circle, phi);

  // create the solver
  my_p4est_poisson_boltzmann_nodes_t pb(neighbors);
  pb.set_parameters(0.1, 5);
  pb.set_phi(phi);
  pb.solve_nonlinear(sol);

  // save the grid into vtk
  double *phi_p, *sol_p;
  VecGetArray(phi, &phi_p);
  VecGetArray(sol, &sol_p);
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         2, 0, "poisson_boltzmann",
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p);
  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(sol, &sol_p);

  // free vectors
  VecDestroy(phi);
  VecDestroy(sol);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

