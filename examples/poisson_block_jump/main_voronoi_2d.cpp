/*
 * Title: poisson_block_jump
 * Description:
 * Author: Mohammad Mirzadeh
 * Date Created: 03-25-2016
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
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#endif

#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: poisson_voronoi_jump");

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
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.35 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.35 - sqrt(SQR(x) + SQR(y));
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

  // compute levelset function
  Vec phi;
  VecCreateGhostNodes(p4est, nodes, &phi);
  sample_cf_on_nodes(p4est, nodes, circle, phi);

  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  node_neighbors.init_neighbors();

  my_p4est_cell_neighbors_t cell_neighbors(&hierarchy);
  cell_neighbors.init_neighbors();

  // set up the solver
#ifndef P4_TO_P8
  struct constant_cf_t:CF_2{
    constant_cf_t(double c): c(c) {}
    double operator() (double, double) const { return c; }
  private:
    double c;
  };

  struct:WallBC2D{
    BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
  } bc_wall_type;

  typedef CF_2 cf_t;
  typedef BoundaryConditions2D bc_t;
#else
  struct constant_cf_t:CF_3{
    constant_cf_t(double c): c(c) {}
    double operator() (double, double, double) { return c; }
  private:
    double c;
  };

  struct:WallBC2D{
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  typedef CF_3 cf_t;
  typedef BoundaryConditions3D bc_t;
#endif
  constant_cf_t bc_wall_value(0);

  bc_t bc;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  Vec rhs_m, rhs_p, mue_m, mue_p, jump_u, jump_du;
  VecCreateGhostNodes(p4est, nodes, &rhs_m);
  VecDuplicate(rhs_m, &rhs_p);
  VecDuplicate(rhs_m, &mue_m);
  VecDuplicate(rhs_m, &mue_p);
  VecDuplicate(rhs_m, &jump_u);
  VecDuplicate(rhs_m, &jump_du);

  Vec l;
  VecGhostGetLocalForm(mue_m, &l);
  VecSet(l, 1);
  VecGhostRestoreLocalForm(mue_m, &l);

  VecGhostGetLocalForm(mue_p, &l);
  VecSet(l, 1);
  VecGhostRestoreLocalForm(mue_p, &l);

  VecGhostGetLocalForm(jump_u, &l);
  VecSet(l, 1);
  VecGhostRestoreLocalForm(jump_u, &l);

  VecGhostGetLocalForm(jump_du, &l);
  VecSet(l, 2);
  VecGhostRestoreLocalForm(jump_du, &l);

  my_p4est_poisson_jump_nodes_voronoi_t jump_solver(&node_neighbors, &cell_neighbors);
  jump_solver.set_bc(bc);
  jump_solver.set_mu(mue_m, mue_p);
  jump_solver.set_u_jump(jump_u);
  jump_solver.set_mu_grad_u_jump(jump_du);
  jump_solver.set_phi(phi);
  jump_solver.set_rhs(rhs_m, rhs_p);

  Vec solution;
  VecCreateGhostNodes(p4est, nodes, &solution);
  jump_solver.solve(solution);

  // save the grid into vtk
  double *solution_p;
  VecGetArray(solution, &solution_p);
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "poisson_voronoi_jump",
                         VTK_POINT_DATA, "sol 1", solution_p);

  VecRestoreArray(solution, &solution_p);

  VecDestroy(solution);
  VecDestroy(rhs_p);
  VecDestroy(rhs_m);
  VecDestroy(mue_p);
  VecDestroy(mue_m);
  VecDestroy(jump_u);
  VecDestroy(jump_du);

  VecDestroy(phi);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}
