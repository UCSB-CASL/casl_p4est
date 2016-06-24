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
#include <src/my_p4est_poisson_nodes.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_nodes.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

using namespace std;

static const splitting_criteria_t sp(1,5);

static p4est_bool_t
refine_periodic_wall (p4est_t*, p4est_topidx_t, p4est_quadrant_t *quad) {
  p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);
  return quad->x + qh == P4EST_ROOT_LEN && quad->level < sp.max_lvl;
}

static p4est_bool_t
refine_periodic_rand (p4est_t*, p4est_topidx_t, p4est_quadrant_t *quad) {
  return rand()%P4EST_CHILDREN == 0 && quad->level < sp.max_lvl;
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
  const int n_xyz[]      = { 3,  3,  3};
  const double xyz_min[] = {-1, -1, -1};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 0,  1,  1};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); 
  p4est->user_pointer = (void*)&sp;
  for (int it = 0; it < sp.min_lvl; it++)
    my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
  my_p4est_refine(p4est, P4EST_TRUE, refine_periodic_wall, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // solve for a poisson with dirichlet bc on x and periodic on y and z direction
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  my_p4est_poisson_nodes_t poisson(&neighbors);
#ifdef P4_TO_P8
  BoundaryConditions3D bc;
  struct:WallBC3D {
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;
  struct:CF_3 {
    double operator()(double x, double, double) const { return x; }
  } bc_wall_value;
  struct:CF_3 {
    double operator()(double, double, double) const { return 1.0; }
  } rhs;
#else
  BoundaryConditions2D bc;
  struct:WallBC2D {
    BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
  } bc_wall_type;
  struct:CF_2 {
    double operator()(double x, double) const { return fabs(x); }
  } bc_wall_value;
  struct:CF_2 {
    double operator()(double, double) const { return 1; }
  } rhs;
#endif
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  poisson.set_bc(bc);

  Vec u;
  VecCreateGhostNodes(p4est, nodes, &u);
  sample_cf_on_nodes(p4est, nodes, rhs, u);
  poisson.set_rhs(u);
  poisson.solve(u);

  // save the grid into vtk  
  double *u_p;
  VecGetArray(u, &u_p);
  my_p4est_vtk_write_all(p4est, nodes, NULL,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "periodic.1",
                         VTK_POINT_DATA, "u", u_p);
  VecRestoreArray(u, &u_p);

  // destroy the structures
  VecDestroy(u);
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

