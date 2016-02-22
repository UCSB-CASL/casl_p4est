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

#ifdef P4_TO_P8
#error "Not fully implemented!"
typedef CF_3 cf_t;
typedef WallBC3D wall_bc_t;
#else
typedef CF_2 cf_t;
typedef WallBC2D wall_bc_t;
#endif

static struct {
  int lmin, lmax, iter;
  double lip, Ca, cfl, dts, dtmax, alpha;
  double xmin[3], xmax[3];
  int ntr[3];
  string method, test;

  cf_t *interface, *bc_wall_value, *K_D, *K_EO, *gamma;
  CF_1 *Q, *I;
  wall_bc_t *bc_wall_type;
} params;

void set_parameters(int argc, char **argv) {
  // parse input parameters
  cmdParser cmd;
  cmd.add_option("lmin", "min level");
  cmd.add_option("lmax", "max level");
  cmd.add_option("lip", "Lipschitz constant used for grid generation");
  cmd.add_option("Ca", "the capillary number");
  cmd.add_option("iter", "number of iterations");
  cmd.add_option("cfl", "the CFL number");
  cmd.add_option("dts", "dt for saving vtk files");
  cmd.add_option("dtmax", "max dt to use when solving");
  cmd.add_option("method", "choose advection method");
  cmd.add_option("alpha", "EO coupling strength");
  cmd.add_option("test", "Which test to run?, Options are:"
                         "circle\n"
                         "FastShelley04_Fig12\n");

  cmd.parse(argc, argv);

  params.test = cmd.get<string>("test", "FastShelley04_Fig12");

  // set default values
  params.ntr[0]  = params.ntr[1]  = params.ntr[2]  =  1;
  params.xmin[0] = params.xmin[1] = params.xmin[2] = -1;
  params.xmax[0] = params.xmax[1] = params.xmax[2] =  1;

  if (params.test == "circle") {
    // set interface
#ifdef P4_TO_P8
#else
    static struct:cf_t{
      double operator()(double, double) const { return 1.0/params.Ca; }
    } gamma;

    static struct:cf_t{
      double operator()(double, double) const { return 1.0; }
    } K_D;

    static struct:cf_t{
      double operator()(double, double) const { return 1.0; }
    } K_EO;

    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } Q;

    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } I;

    static struct:cf_t {
      double operator()(double x, double y) const {
        return 0.25 - sqrt(SQR(x)+SQR(y));
      }
    } interface; interface.lip = params.lip;

    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double, double) const { return 0; }
    } bc_wall_value;
#endif

    params.gamma         = &gamma;
    params.K_D           = &K_D;
    params.K_EO          = &K_EO;
    params.Q             = &Q;
    params.I             = &I;
    params.interface     = &interface;
    params.bc_wall_type  = &bc_wall_type;
    params.bc_wall_value = &bc_wall_value;
    params.dtmax         = 5e-3;
    params.dts           = 1e-1;
    params.alpha         = 0;

  } else if (params.test == "FastShelley04_Fig12") {

    params.xmin[0] = params.xmin[1] = params.xmin[2] = -10;
    params.xmax[0] = params.xmax[1] = params.xmax[2] =  10;

#ifdef P4_TO_P8
#else
    static struct:cf_t{
      double operator()(double, double) const { return 1.0/params.Ca; }
    } gamma;

    static struct:cf_t{
      double operator()(double, double) const { return 1.0; }
    } K_D;

    static struct:cf_t{
      double operator()(double, double) const { return 1.0; }
    } K_EO;

    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } Q;

    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } I;

    static struct:cf_t{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));

        return 1.0+0.1*(cos(3*theta)+sin(2*theta)) - r;
      }
    } interface; interface.lip = params.lip;

    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return NEUMANN; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double y) const {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));
        double ur    = -Q(t)/r;

        if (fabs(x-params.xmax[0]) < EPS || fabs(x - params.xmin[0]) < EPS)
          return x > 0 ? ur*cos(theta):-ur*cos(theta);
        else if (fabs(y-params.xmax[1]) < EPS || fabs(y - params.xmin[1]) < EPS)
          return y > 0 ? ur*sin(theta):-ur*sin(theta);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif

    // set parameters specific to this test
    params.gamma         = &gamma;
    params.K_D           = &K_D;
    params.K_EO          = &K_EO;
    params.Q             = &Q;
    params.I             = &I;
    params.interface     = &interface;
    params.bc_wall_type  = &bc_wall_type;
    params.bc_wall_value = &bc_wall_value;
    params.dtmax         = 5e-3;
    params.dts           = 1e-1;
    params.alpha         = 1;

  } else {
    throw std::invalid_argument("Unknown test");
  }

  // overwrite default values from stdin
  params.lmin   = cmd.get("lmin", 5);
  params.lmax   = cmd.get("lmax", 10);
  params.iter   = cmd.get("iter", INT_MAX);
  params.lip    = cmd.get("lip", 1.2);
  params.Ca     = cmd.get("Ca", 250);
  params.cfl    = cmd.get("cfl", 5.0);
  params.dts    = cmd.get("dts", params.dts);
  params.dtmax  = cmd.get("dtmax", params.dtmax);
  params.method = cmd.get<string>("method", "semi_lagrangian");
  params.alpha  = cmd.get("alpha", params.alpha);
}

int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: viscous_fingering");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // setup the parameters
  set_parameters(argc, argv);

  conn = my_p4est_brick_new(params.ntr, params.xmin, params.xmax, &brick);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
  splitting_criteria_cf_t sp(params.lmin, params.lmax, params.interface, params.lip);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // initialize variables
  Vec phi, pressure, potential;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &pressure);
  VecCreateGhostNodes(p4est, nodes, &potential);
  sample_cf_on_nodes(p4est, nodes, *params.interface, phi);

  // set up the solver
  one_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_injection_rates(*params.Q, *params.I, params.alpha);
  solver.set_properties(*params.K_D, *params.K_EO, *params.gamma);
  solver.set_bc_wall(*params.bc_wall_type, *params.bc_wall_value);

  const char* filename = params.test.c_str();
  char vtk_name[FILENAME_MAX];

  double dt = 0, t = 0;
  for(int i=0; i<params.iter; i++) {
    dt = solver.solve_one_step(t, phi, pressure, potential, params.method, params.cfl, params.dtmax);
    t += dt;

    p4est_gloidx_t num_nodes = 0;
    for (int r = 0; r<mpi.size(); r++)
      num_nodes += nodes->global_owned_indeps[r];
    PetscPrintf(mpi.comm(), "i = %04d n = %6d t = %1.5f dt = %1.5e\n", i, num_nodes, t, dt);

    // save vtk
    double *phi_p, *pressure_p, *potential_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(pressure, &pressure_p);
    VecGetArray(potential, &potential_p);
    sprintf(vtk_name, "%s_%s_%dd.%04d", filename, params.method.c_str(), P4EST_DIM, i);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, vtk_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "pressure", pressure_p,
                           VTK_POINT_DATA, "potential", potential_p);
    VecRestoreArray(phi, &phi_p);
    VecRestoreArray(pressure, &pressure_p);
    VecRestoreArray(potential, &potential_p);
  }

  // destroy vectors
  VecDestroy(phi);
  VecDestroy(pressure);
  VecDestroy(potential);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

