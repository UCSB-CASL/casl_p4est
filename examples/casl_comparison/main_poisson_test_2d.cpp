/* 
 * Title: casl_test
 * Description:
 * Author: Mohammad Mirzadeh
 * Date Created: 01-11-2016
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

using namespace std;

int main(int argc, char** argv) {  
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser parser;
  parser.add_option("ns", "number of splits");
  parser.parse(argc, argv);

  const int ns = parser.get("ns", 0);

  // stopwatch
  parStopWatch w;
  w.start("Running example: casl_test");

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

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } interface;

  struct:CF_3{
    double operator() (double x, double y, double z) const {
      return cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
    }
  } uex;

  struct:CF_3{
    double operator() (double x, double y, double z) const {
      return 12*SQR(M_PI)*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
    }
  } fex;

  struct:CF_3{
    double operator() (double, double, double) const {
      return 1.0;
    }
  } mue_ex;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } interface;

  struct:CF_2{
    double operator() (double x, double y) const {
      return cos(2*M_PI*x)*cos(2*M_PI*y);
    }
  } uex;

  struct:CF_2{
    double operator() (double x, double y) const {
      return 8*SQR(M_PI)*cos(2*M_PI*x)*cos(2*M_PI*y);
    }
  } fex;

  struct:CF_2{
    double operator() (double, double) const {
      return 1.0;
    }
  } mue_ex;
#endif

  splitting_criteria_cf_t sp(3, 8, &interface);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  for (int n = 0; n<ns; n++)
    my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // create hierarchy and neighbors
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // initialize
  Vec phi, sol, mue, rhs;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecDuplicate(phi, &sol);
  VecDuplicate(phi, &mue);
  VecDuplicate(phi, &rhs);

  sample_cf_on_nodes(p4est, nodes, interface, phi);
  sample_cf_on_nodes(p4est, nodes, fex, rhs);
  sample_cf_on_nodes(p4est, nodes, mue_ex, mue);

  // boundary conditions
#ifdef P4_TO_P8
  BoundaryConditions3D bc;

  struct:WallBC3D {
    BoundaryConditionType operator() (double, double, double) const {
      return DIRICHLET;
    }
  } bc_wall_type;
#else
  BoundaryConditions2D bc;

  struct:WallBC2D {
    BoundaryConditionType operator() (double, double) const {
      return NEUMANN;
    }
  } bc_wall_type;

  struct:CF_2 {
    double operator()(double, double) const {return 0;}
  } bc_wall_value;
#endif

  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(uex);
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  // solve the poisson equation
  my_p4est_poisson_nodes_t poisson(&neighbors);
  poisson.set_bc(bc);
  poisson.set_phi(phi);
  poisson.set_mu(mue);
  poisson.set_rhs(rhs);
  poisson.solve(sol);

  // extend solution
  my_p4est_level_set_t ls(&neighbors);
  w.start("extrapolation");
  ls.extend_Over_Interface_TVD(phi, sol);
  w.stop(); w.read_duration();

  // save the grid into vtk
  double *sol_p, *phi_p, *rhs_p;
  VecGetArray(phi, &phi_p);
  VecGetArray(sol, &sol_p);
  VecGetArray(rhs, &rhs_p);

  double err = 0;
  double x[P4EST_DIM];
  double diag = p4est_diag_min(p4est);
  foreach_node (n, nodes) {    
    if (phi_p[n] > 0 && phi_p[n] < diag) {
      node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
      rhs_p[n] = fabs(sol_p[n] - uex(x[0], x[1], x[2]));
#else
      rhs_p[n] = fabs(sol_p[n] - uex(x[0], x[1]));
#endif
      err = MAX(err, rhs_p[n]);
    } else {
      rhs_p[n] = 0;
    }

  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         3, 0, "parcasl",
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", rhs_p,
                         VTK_POINT_DATA, "phi", phi_p);
  printf("err = %1.14e\n", err);

  // clear vectors
  VecDestroy(phi);
  VecDestroy(rhs);
  VecDestroy(sol);
  VecDestroy(mue);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

