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
#include "two_fluid_solver_2d.h"
#include <src/my_p4est_vtk.h>
#else
#include "two_fluid_solver_2d.h"
#include <src/my_p4est_vtk.h>
#endif

#include <sys/stat.h>
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
  double lip, Ca, cfl, dts, dtmax, viscosity;
  double xmin[3], xmax[3];
  int ntr[3];
  string test, method;

  cf_t *interface, *bc_wall_value;
  CF_1 *Q;
  wall_bc_t *bc_wall_type;
} options;

void set_options(int argc, char **argv) {
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
  cmd.add_option("viscosity", "The viscosity ratio of inner to outer fluid");
  cmd.add_option("method", "method for solving the jump equation");
  cmd.add_option("test", "Which test to run?, Options are:\n"
                         "\tcircle\n"
                         "\tFastShelley04_Fig12\n");

  cmd.parse(argc, argv);

  options.test = cmd.get<string>("test", "FastShelley04_Fig12");
  options.method = cmd.get<string>("method", "voronoi");

  // set default values
  options.ntr[0]  = options.ntr[1]  = options.ntr[2]  =  1;
  options.xmin[0] = options.xmin[1] = options.xmin[2] = -1;
  options.xmax[0] = options.xmax[1] = options.xmax[2] =  1;

  if (options.test == "circle") {
    // set interface
#ifdef P4_TO_P8
#else        
    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } Q;

    static struct:cf_t {
      double operator()(double x, double y) const {
        return 0.25 - sqrt(SQR(x)+SQR(y));
      }
    } interface; interface.lip = options.lip;

    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double, double) const { return 0; }
    } bc_wall_value;
#endif

    options.Q             = &Q;
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;
    options.dtmax         = 5e-3;
    options.dts           = 1e-1;
    options.Ca            = 250;
    options.viscosity     = 1;

  } else if (options.test == "FastShelley04_Fig12") {

    options.xmin[0] = options.xmin[1] = options.xmin[2] = -10;
    options.xmax[0] = options.xmax[1] = options.xmax[2] =  10;

#ifdef P4_TO_P8
#else
    static struct:CF_1{
      double operator()(double t) const { return 1+t; }
    } Q;

    static struct:cf_t{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));

        return 1.0+0.1*(cos(3*theta)+sin(2*theta)) - r;
      }
    } interface; interface.lip = options.lip;

#if 0
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
#if 1
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double y) const {
        double r = sqrt(SQR(x)+SQR(y));
        return -Q(t)/(2*PI) * log(r);
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif

#endif

    // set parameters specific to this test
    options.Q              = &Q;
    options.interface      = &interface;
    options.bc_wall_type   = &bc_wall_type;
    options.bc_wall_value  = &bc_wall_value;
    options.dtmax          = 5e-3;
    options.dts            = 1e-1;
    options.Ca             = 250;
    options.viscosity      = 1;

  } else {
    throw std::invalid_argument("Unknown test");
  }

  // overwrite default values from stdin
  options.lmin      = cmd.get("lmin", 5);
  options.lmax      = cmd.get("lmax", 10);
  options.iter      = cmd.get("iter", INT_MAX);
  options.lip       = cmd.get("lip", 1.2);
  options.Ca        = cmd.get("Ca", options.Ca);
  options.cfl       = cmd.get("cfl", 5.0);
  options.dts       = cmd.get("dts", options.dts);
  options.dtmax     = cmd.get("dtmax", options.dtmax);
  options.viscosity = cmd.get("viscosity", options.viscosity);
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
  set_options(argc, argv);

  conn = my_p4est_brick_new(options.ntr, options.xmin, options.xmax, &brick);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
  splitting_criteria_cf_t sp(options.lmin, options.lmax, options.interface, options.lip);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // initialize variables
  Vec phi, press_m, press_p;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &press_m);
  VecCreateGhostNodes(p4est, nodes, &press_p);
  sample_cf_on_nodes(p4est, nodes, *options.interface, phi);

  // set up the solver
  two_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_properties(options.viscosity, options.Ca, *options.Q);
  solver.set_bc_wall(*options.bc_wall_type, *options.bc_wall_value);


  ostringstream folder;
  folder << options.test << "/two_fluid/"
         << "mue_"   << options.viscosity;
  system(("mkdir -p " + folder.str()).c_str());
  char vtk_name[FILENAME_MAX];

  double dt = 0, t = 0;
  for(int i=0; i<options.iter; i++) {
    dt = solver.solve_one_step(t, phi, press_m, press_p, options.cfl, options.dtmax, options.method);
    t += dt;

    p4est_gloidx_t num_nodes = 0;
    for (int r = 0; r<mpi.size(); r++)
      num_nodes += nodes->global_owned_indeps[r];
    PetscPrintf(mpi.comm(), "i = %04d n = %6d t = %1.5f dt = %1.5e\n", i, num_nodes, t, dt);

    // save vtk
    double *phi_p, *press_m_p, *press_p_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(press_m, &press_m_p);
    VecGetArray(press_p, &press_p_p);

    sprintf(vtk_name, "%s/%s_%dd.%04d", folder.str().c_str(), options.method.c_str(), P4EST_DIM, i);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, vtk_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "press_m", press_m_p,
                           VTK_POINT_DATA, "press_p", press_p_p);
    VecRestoreArray(phi, &phi_p);
    VecRestoreArray(press_m, &press_m_p);
    VecRestoreArray(press_p, &press_p_p);
  }

  // destroy vectors
  VecDestroy(phi);
  VecDestroy(press_m);
  VecDestroy(press_p);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

