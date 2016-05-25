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
#include <src/my_p4est_poisson_jump_voronoi_block.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_jump_voronoi_block.h>
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
  w.start("Running example: poisson_block_jump");

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
  int periodic []   = {0, 0, 0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // define problem parameters
  static double kp [][2] =
  {
    {2, 5},
    {10, 1}
  };

  static double km [][2] =
  {
    {1, 0.2},
    {0.5, 3}
  };
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.35 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  static struct:CF_2{
    double operator()(double x, double y) const {
      return 0.35 - sqrt(SQR(x) + SQR(y));
    }
  } circle;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return cos(x)*cos(y);
    }

    double dn(double x, double y) const {
      double r = sqrt(SQR(x)+SQR(y));
      return -(-x*sin(x)*cos(y)-y*cos(x)*sin(y))/r;
    }

    double laplace(double x, double y) const {
      return -2*cos(x)*cos(y);
    }
  } sol_1_plus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return x*x*x+y*y*y;
    }

    double dn(double x, double y) const {
      double r = sqrt(SQR(x)+SQR(y));
      return -(3*x*x*x+3*y*y*y)/r;
    }

    double laplace(double x, double y) const {
      return 6*x+6*y;
    }
  } sol_2_plus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return exp(x+y);
    }

    double dn(double x, double y) const {
      double r = sqrt(SQR(x)+SQR(y));
      return -(x+y)*exp(x+y)/r;
    }

    double laplace(double x, double y) const {
      return 2*exp(x+y);
    }
  } sol_1_minus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return sin(x)+sin(y);
    }

    double dn(double x, double y) const {
      double r = sqrt(SQR(x)+SQR(y));
      return -(x*cos(x)+y*cos(y))/r;
    }

    double laplace(double x, double y) const {
      return -(sin(x)+sin(y));
    }
  } sol_2_minus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return sol_1_plus(x,y) - sol_1_minus(x,y);
    }
  } jump_sol_1;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return sol_2_plus(x,y) - sol_2_minus(x,y);
    }
  } jump_sol_2;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return (kp[0][0]*sol_1_plus.dn(x,y)  + kp[0][1]*sol_2_plus.dn(x,y)) -
             (km[0][0]*sol_1_minus.dn(x,y) + km[0][1]*sol_2_minus.dn(x,y));
    }
  } jump_grad_sol_1;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return (kp[1][0]*sol_1_plus.dn(x,y)  + kp[1][1]*sol_2_plus.dn(x,y)) -
             (km[1][0]*sol_1_minus.dn(x,y) + km[1][1]*sol_2_minus.dn(x,y));
    }
  } jump_grad_sol_2;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return -(kp[0][0]*sol_1_plus.laplace(x,y)+kp[0][1]*sol_2_plus.laplace(x,y));
    }
  } rhs_1_plus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return -(kp[1][0]*sol_1_plus.laplace(x,y)+kp[1][1]*sol_2_plus.laplace(x,y));
    }
  } rhs_2_plus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return -(km[0][0]*sol_1_minus.laplace(x,y)+km[0][1]*sol_2_minus.laplace(x,y));
    }
  } rhs_1_minus;

  static struct:CF_2{
    double operator()(double x, double y) const {
      return -(km[1][0]*sol_1_minus.laplace(x,y)+km[1][1]*sol_2_minus.laplace(x,y));
    }
  } rhs_2_minus;
#endif

  splitting_criteria_cf_t sp(5, 8, &circle);
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

  int bs = 2;
  vector<vector<cf_t*>> mue_m(bs, vector<cf_t*>(bs));
  mue_m[0][0] = new constant_cf_t(km[0][0]);
  mue_m[0][1] = new constant_cf_t(km[0][1]);
  mue_m[1][0] = new constant_cf_t(km[1][0]);
  mue_m[1][1] = new constant_cf_t(km[1][1]);

  vector<vector<cf_t*>> mue_p(bs, vector<cf_t*>(bs));
  mue_p[0][0] = new constant_cf_t(kp[0][0]);
  mue_p[0][1] = new constant_cf_t(kp[0][1]);
  mue_p[1][0] = new constant_cf_t(kp[1][0]);
  mue_p[1][1] = new constant_cf_t(kp[1][1]);

  vector<vector<cf_t*>> add(bs, vector<cf_t*>(bs));
  add[0][0] = new constant_cf_t(0);
  add[0][1] = new constant_cf_t(0);
  add[1][0] = new constant_cf_t(0);
  add[1][1] = new constant_cf_t(0);

  vector<cf_t*> jump_u(bs), jump_du(bs);
  jump_u[0]  = &jump_sol_1;
  jump_du[0] = &jump_grad_sol_1;
  jump_u[1]  = &jump_sol_2;
  jump_du[1] = &jump_grad_sol_2;

  vector<bc_t> bc(bs);
  bc[0].setWallTypes(bc_wall_type);
  bc[0].setWallValues(sol_1_minus);

  bc[1].setWallTypes(bc_wall_type);
  bc[1].setWallValues(sol_2_minus);

  Vec rhs_m[bs], rhs_p[bs];
  for (int i=0; i<bs; i++) {
    VecCreateGhostNodes(p4est, nodes, &rhs_m[i]);   
    VecCreateGhostNodes(p4est, nodes, &rhs_p[i]);
  }

  sample_cf_on_nodes(p4est, nodes, rhs_1_minus, rhs_m[0]);
  sample_cf_on_nodes(p4est, nodes, rhs_2_minus, rhs_m[1]);
  sample_cf_on_nodes(p4est, nodes, rhs_1_plus,  rhs_p[0]);
  sample_cf_on_nodes(p4est, nodes, rhs_2_plus,  rhs_p[1]);

  my_p4est_poisson_jump_voronoi_block_t jump_solver(bs, &node_neighbors, &cell_neighbors);
  jump_solver.set_bc(bc);
  jump_solver.set_diagonal(add);
  jump_solver.set_mu(mue_m, mue_p);
  jump_solver.set_u_jump(jump_u);
  jump_solver.set_mu_grad_u_jump(jump_du);
  jump_solver.set_phi(phi);
  jump_solver.set_rhs(rhs_m, rhs_p);

  Vec solution[bs];
  for (int i=0; i<bs; i++)
    VecCreateGhostNodes(p4est, nodes, &solution[i]);
  jump_solver.solve(solution);

  for (int i=0; i<bs; i++) {
    for (int j=0; j<bs; j++) {
      delete mue_m[i][j];
      delete mue_p[i][j];
      delete add[i][j];
    }
  }

  // save the grid into vtk
  double *solution_p[bs], *phi_p;
  for (int i=0; i<bs; i++)
    VecGetArray(solution[i], &solution_p[i]);
  VecGetArray(phi, &phi_p);

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         bs+1, 0, "poisson_block_jump",
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol 1", solution_p[0],
                         VTK_POINT_DATA, "sol 2", solution_p[1]);

  for (int i=0; i<bs; i++)
    VecRestoreArray(solution[i], &solution_p[i]);
  VecRestoreArray(phi, &phi_p);

  for (int i=0; i<bs; i++) {
    VecDestroy(solution[i]);
    VecDestroy(rhs_p[i]);
    VecDestroy(rhs_m[i]);
  }

  VecDestroy(phi);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

