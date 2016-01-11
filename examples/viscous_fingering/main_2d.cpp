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
  cmd.add_option("lip", "Lipschitz constant used for grid generation");
  cmd.add_option("g", "surface tension");
  cmd.add_option("iter", "number of iterations");
  cmd.add_option("cfl", "the CFL number");
  cmd.add_option("method", "choose advection method");
  cmd.add_option("test", "Which test to run? Options are:"
                         "a) circle"
                         "b) plane");

  cmd.parse(argc, argv);

  const static int lmin = cmd.get("lmin", 3);
  const static int lmax = cmd.get("lmax", 10);
  const static int iter = cmd.get("iter", 100);
  const static double lip = cmd.get("lip", 1.2);
  const static double g = cmd.get("g", 1e-5);
  const static double cfl = cmd.get("cfl", 5.0);
  const static string method = cmd.get<string>("method", "semi_lagrangian");
  const static string test   = cmd.get<string>("test", "circle");

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
  static int n_tr [3]    = {10, 1, 1};
  static double xmin [3] = {0, -0.5, -0.5};
  static double xmax [3] = {10, 0.5,  0.5};

  if (test == "circle") {
    n_tr[0] = 1;
    xmin[0] = -0.5;
    xmax[0] =  0.5;
  } else if (test == "plane") {
    // default values
  } else {
    throw std::invalid_argument("Unknown test");
  }

  conn = my_p4est_brick_new(n_tr, xmin, xmax, &brick);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  typedef CF_3 cf_t;
  typedef WallBC3D wall_bc_t;
#else
  typedef CF_2 cf_t;
  typedef WallBC2D wall_bc_t;
#endif

#ifdef P4_TO_P8
  struct:cf_t{
    double operator()(double x, double, double) const {
      return 0.05-x;
    }
  } interface_palne;

  struct:cf_t{
    double operator()(double x, double y, double z) const {
      return return 0.01 - sqrt(SQR(x)+SQR(y)+SQR(z));
    }
  } interface_circle;
#else
  struct:cf_t{
    double operator()(double x, double y) const {
      return 0.01 - sqrt(SQR(x)+SQR(y));
    }
  } interface_circle;

  struct:cf_t{
    double operator()(double x, double) const {
      return 0.05 - x;
    }
  } interface_plane;
#endif

  cf_t *interface = NULL;
  if (test == "circle")
    interface = &interface_circle;
  else if (test == "plane")
    interface = &interface_plane;

  splitting_criteria_cf_t sp(lmin, lmax, interface, lip);
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
  sample_cf_on_nodes(p4est, nodes, *interface, phi);

  // set up the solver
#ifdef P4_TO_P8
  struct:cf_t{
    double operator()(double, double, double) const { return 1; }
  } K_D;

  struct:cf_t{
    double operator()(double, double, double) const { return g; }
  } gamma;

  struct:wall_bc_t {
    BoundaryConditionType operator()(double x, double, double) const {
      if (fabs(x-xmin[0]) < EPS || fabs(x-xmax[0]) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } bc_wall_type_plane;

  struct:wall_bc_t {
    BoundaryConditionType operator()(double, double, double) const {
      return DIRICHLET;
    }
  } bc_wall_type_circle;

  struct:CF_3 {
    double operator()(double, double, double) const {
      return 0;
    }
  } bc_wall_value;
#else
  struct:cf_t{
    double operator()(double, double) const { return 1; }
  } K_D;

  struct:cf_t{
    double operator()(double, double) const { return g; }
  } gamma;

  struct:wall_bc_t {
    BoundaryConditionType operator()(double x, double) const {
      if (fabs(x-xmin[0]) < EPS || fabs(x-xmax[0]) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } bc_wall_type_plane;

  struct:wall_bc_t {
    BoundaryConditionType operator()(double, double) const {
      return DIRICHLET;
    }
  } bc_wall_type_circle;

  struct:cf_t {
    double operator()(double, double) const {
      return 0;
    }
  } bc_wall_value;
#endif

  wall_bc_t *bc_wall_type = NULL;
  if (test == "circle")
    bc_wall_type = &bc_wall_type_circle;
  else if (test == "plane")
    bc_wall_type = &bc_wall_type_plane;

  one_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_properties(K_D, gamma);
  solver.set_bc_wall(*bc_wall_type, bc_wall_value);

  const char* filename = "viscous_fingering";
  char vtk_name[FILENAME_MAX];

  double dt = 0, t = 0;
  for(int i=0; i<iter; i++) {
    dt = solver.solve_one_step(phi, pressure, method, cfl);
    t += dt;

    p4est_gloidx_t num_nodes = 0;
    for (int r = 0; r<mpi.size(); r++)
      num_nodes += nodes->global_owned_indeps[r];
    PetscPrintf(mpi.comm(), "i = %04d n = %6d t = %1.5f dt = %1.5e\n", i, num_nodes, t, dt);

    // save vtk
    double *phi_p, *pressure_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(pressure, &pressure_p);
    sprintf(vtk_name, "%s_%s_%dd.%04d", filename, method.c_str(), P4EST_DIM, i);
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

