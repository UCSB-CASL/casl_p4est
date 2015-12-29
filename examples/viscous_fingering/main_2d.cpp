/* 
 * Title: viscous_fingering
 * Description:
 * Solves the viscous fingering problem in a one-fluid configuration, i.e. we solve
 *                              div(k/mu grad(p)) = 0,                          (1)
 * where k is teh permeability function and mu is the viscosity. Equation (1) is
 * solved subjected to Dirichlet b.c on the interface, i.e.
 *                              p = gamma * kappa,                              (2)
 * where gamma is the surface tension and kappa = -div(n) is the curvature. The
 * interface is then advected with a normal velocity given by Darcy's law, i.e.
 *                              vn = -k/mu dp/dn.                               (3)
 *
 * Author: Mohammad Mirzadeh
 * Date Created: 12-22-2015
 */

#ifndef P4_TO_P8
#include "one_fluid_solver_2d.h"
#include <src/my_p4est_vtk.h>
#else
#include "one_fluid_solver_2d.h"
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_enviroment_t mpi;
  mpi.init(argc, argv);

  // get input parameters
  cmdParser cmd;
  cmd.add_option("lmin", "min level");
  cmd.add_option("lmax", "max level");
  cmd.add_option("g", "surface tension");
  cmd.add_option("iter", "number of iterations");
  cmd.add_option("cfl", "the CFL number");
  cmd.parse(argc, argv);

  const static int lmin = cmd.get("lmin", 2);
  const static int lmax = cmd.get("lmax", 8);
  const static int iter = cmd.get("iter", 100);
  const static double g = cmd.get("g", 0.01);
  const static double cfl = cmd.get("cfl", 5.0);

  // stopwatch
  parStopWatch w;
  w.start("Running example: viscous_fingering");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz []      = {10, 1, 1};
  const double xyz_min [] = {0, -0.5, -0.5};
  const double xyz_max [] = {10, 0.5,  0.5};
//  const int n_xyz []      = {1, 1, 1};
//  const double xyz_min [] = {-0.5, -0.5, -0.5};
//  const double xyz_max [] = { 0.5,  0.5,  0.5};

  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.05-x;
    }
  } interface;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.05 - x + 0.005*sin(2*M_PI*5*y);
//      return 0.05-x+0.005*sin(2*M_PI*10*y);
//      return 0.005 - sqrt(SQR(x)+SQR(y));
    }
  } interface;
#endif

  splitting_criteria_cf_t sp(lmin, lmax, &interface, 2.5);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // initialize variables
  Vec phi, pressure;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &pressure);
  sample_cf_on_nodes(p4est, nodes, interface, phi);

  // set up the solver
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double, double, double) const { return 1; }
  } K_D;

  struct:CF_3{
    double operator()(double, double, double) const { return g; }
  } gamma;

  struct:CF_3{
    double operator()(double, double, double) const { return 1; }
  } p_applied;
#else
  struct:CF_2{
    double operator()(double, double) const { return 1; }
  } K_D;

  struct:CF_2{
    double operator()(double, double) const { return g; }
  } gamma;

  struct:CF_2{
    double operator()(double, double) const { return 1; }
  } p_applied;
#endif

  one_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_properties(K_D, gamma, p_applied);

  const char* filename = "viscous_fingering";
  char vtk_name[FILENAME_MAX];

  double dt = 0;
  for(int i=0; i<iter; i++) {
    dt = solver.solve_one_step(phi, pressure, cfl);
    if (mpi.rank() == 0) std::cout << "i = " << i << " dt = " << dt << std::endl;

    // save vtk
    double *phi_p, *pressure_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(pressure, &pressure_p);
    sprintf(vtk_name, "%s_%dd.%04d", filename, P4EST_DIM, i);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "pressure", pressure_p);
    VecRestoreArray(phi, &phi_p);
    VecRestoreArray(pressure, &pressure_p);
  }

  // destroy vectors
  VecDestroy(phi);
  VecDestroy(pressure);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

