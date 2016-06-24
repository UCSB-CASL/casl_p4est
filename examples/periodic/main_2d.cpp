/* 
 * Title: periodic
 * Description:
 * Author: Mohammad Mirzadeh
 * Date Created: 06-21-2016
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

using namespace std;

static p4est_bool_t
refine_periodic_test (p4est_t*, p4est_topidx_t, p4est_quadrant_t *quad) {
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);
  return quad->x + qh == P4EST_ROOT_LEN && quad->level < 3;
}

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: periodic");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz[]      = { 1,  1,  1};
  const double xyz_min[] = { 0,  0,  0};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 1,  1,  1};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); 
  my_p4est_refine(p4est, P4EST_TRUE, refine_periodic_test, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  if (is_periodic(p4est, 0)) std::cout << "X Periodic" << std::endl;
  if (is_periodic(p4est, 1)) std::cout << "Y Periodic" << std::endl;
#ifdef P4_TO_P8
  if (is_periodic(p4est, 2)) std::cout << "Z Periodic" << std::endl;
#endif

  // save the grid into vtk
  my_p4est_vtk_write_all(p4est, nodes, NULL,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, "periodic.1");

  std::cout << mpi.rank() << ": After " << std::endl;

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  std::cout << mpi.rank() << ": After2 " << std::endl;

  w.stop(); w.read_duration();
}

