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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_vtk.h>
#else
#include "one_fluid_solver_3d.h"
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_vtk.h>
#endif

#include <sys/stat.h>
#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

#ifdef P4_TO_P8
typedef CF_3 cf_t;
typedef WallBC3D wall_bc_t;
#else
typedef CF_2 cf_t;
typedef WallBC2D wall_bc_t;
#endif

static struct {
  int lmin, lmax, itmax;
  double lip, Ca, cfl, dts, dtmax, alpha;
  double xmin[3], xmax[3];
  int ntr[3];
  int periodic[3];
  double rot;
  double eps;
  int mode;
  string method, test;

  cf_t *interface, *bc_wall_value, *K_D, *K_EO, *gamma;
  CF_1 *Q, *I;
  wall_bc_t *bc_wall_type;
} options;

void set_parameters(int argc, char **argv) {
  // parse input parameters
  cmdParser cmd;
  cmd.add_option("lmin", "min level");
  cmd.add_option("lmax", "max level");
  cmd.add_option("lip", "Lipschitz constant used for grid generation");
  cmd.add_option("Ca", "the capillary number");
  cmd.add_option("itmax", "number of iterations");
  cmd.add_option("cfl", "the CFL number");
  cmd.add_option("dts", "dt for saving vtk files");
  cmd.add_option("dtmax", "max dt to use when solving");
  cmd.add_option("method", "choose advection method");
  cmd.add_option("alpha", "EO coupling strength");
  cmd.add_option("rot", "the angle to rotate the initial boundary");
  cmd.add_option("mode", "perturbation mode used for analysis");
  cmd.add_option("eps", "perturbation amplitude to be added to the interface");
  cmd.add_option("test", "Which test to run?, Options are:"
      "circle\n"
      "FastShelley04_Fig12\n");

  cmd.parse(argc, argv);

  options.test = cmd.get<string>("test", "FastShelley04_Fig12");

  // set default values
  options.ntr[0]  = options.ntr[1]  = options.ntr[2]  =  1;
  options.xmin[0] = options.xmin[1] = options.xmin[2] = -1;
  options.xmax[0] = options.xmax[1] = options.xmax[2] =  1;
  options.periodic[0] = options.periodic[1] = options.periodic[2] = false;
  options.lmin = 3; options.lmax = 10;
  options.rot  = 0;
  options.method = "semi_lagrangian";

  options.lip  = 1.2;
  options.cfl  = 2;
  options.itmax = numeric_limits<int>::max();

#ifdef P4_TO_P8
    static struct:cf_t{
      double operator()(double, double, double) const { return 1.0/options.Ca; }
    } gamma;

    static struct:cf_t{
      double operator()(double, double, double) const { return 1.0; }
    } K_D;

    static struct:cf_t{
      double operator()(double, double, double) const { return 1.0; }
    } K_EO;
#else
  static struct:cf_t{
    double operator()(double, double) const { return 1.0/options.Ca; }
  } gamma;

  static struct:cf_t{
    double operator()(double, double) const { return 1.0; }
  } K_D;

  static struct:cf_t{
    double operator()(double, double) const { return 1.0; }
  } K_EO;
#endif

  static struct:CF_1{
    double operator()(double t) const { return 1.0 + t; }
  } Q;

  static struct:CF_1{
    double operator()(double t) const { return 1.0 + t; }
  } I;

  options.K_D   = &K_D;
  options.K_EO  = &K_EO;
  options.gamma = &gamma;
  options.Q     = &Q;
  options.I     = &I;
  options.Ca    = 250;

  if (options.test == "circle") {
    // set interface
    options.xmin[0] = options.xmin[1] = options.xmin[2] = -10 + EPS;
    options.xmax[0] = options.xmax[1] = options.xmax[2] =  10;
    options.lmax    = 10;
    options.lmin    = 5;
    options.itmax   = 10;
    options.dtmax   = 1e-3;
    options.dts     = 1e-1;
    options.Ca      = 250;
    options.alpha   = 0;
    options.mode    = cmd.get("mode", 0);
    options.eps     = cmd.get("eps", 5e-3);
    options.method  = "semi_lagrangian";
    options.lip     = cmd.get("lip", options.lip);

    static struct:CF_2{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x);// - M_PI/180 * 45;
        double r     = sqrt(SQR(x)+SQR(y));
        return 1+options.eps*cos(options.mode*theta) - r;
      }
    } interface; interface.lip = options.lip;
    options.interface = &interface;

    static struct:WallBC2D{
      BoundaryConditionType operator()(double, double) const { return NEUMANN; }
    } bc_wall_type;

    static struct:CF_2{
      double operator()(double x, double y) const {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));
        double ur    = -(*options.Q)(t)/(r);

        if (fabs(x-options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
          return x > 0 ? ur*cos(theta):-ur*cos(theta);
        else if (fabs(y-options.xmax[1]) < EPS || fabs(y - options.xmin[1]) < EPS)
          return y > 0 ? ur*sin(theta):-ur*sin(theta);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;

//    static struct:WallBC2D {
//      BoundaryConditionType operator()(double, double) const {
//        return DIRICHLET;
//      }
//    } bc_wall_type;

//    static struct:CF_2 {
//      double operator()(double x, double y) const {
//        double r = sqrt(SQR(x)+SQR(y));
//        return - ( 1.0/(options.Ca*(1+t)) + (1+t) * log(r/(1+t)) );
//      }
//    } bc_wall_value; bc_wall_value.t = 0;

    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "FastShelley04_Fig12") {

    options.xmin[0] = options.xmin[1] = options.xmin[2] = -100;
    options.xmax[0] = options.xmax[1] = options.xmax[2] =  100;
    options.ntr[0]  = options.ntr[1]  = options.ntr[2]  = 10;
    options.cfl     = 2;
    options.dtmax   = 5e-3;
    options.dts     = 1e-1;
    options.alpha   = 0;

#ifdef P4_TO_P8
    static struct:cf_t{
      double operator()(double x, double y, double z) const  {
        double theta = atan2(y,x) - options.rot*PI/180;
        double r     = sqrt(SQR(x)+SQR(y)+SQR(z));
        double phi   = acos(z/MAX(r,1E-12));

        return 1.0+0.1*(cos(3*theta)+sin(2*theta))*pow(sin(2*phi),2) - r;
      }
    } interface; interface.lip = options.lip;

#if 1
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double, double) const { return NEUMANN; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double y, double z) const {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y)+SQR(z));
        double phi   = acos(z/MAX(r,1E-12));
        double ur    = -(*options.Q)(t)/(4*PI*r*r);

        if (fabs(x-options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
          return x > 0 ? ur*cos(theta)*sin(phi):-ur*cos(theta)*sin(phi);
        else if (fabs(y-options.xmax[1]) < EPS || fabs(y - options.xmin[1]) < EPS)
          return y > 0 ? ur*sin(theta)*sin(phi):-ur*sin(theta)*sin(phi);
        else if (fabs(z-options.xmax[2]) < EPS || fabs(z - options.xmin[2]) < EPS)
          return z > 0 ? ur*cos(phi):-ur*cos(phi);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 0
#if 0
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double y, double z) const {
        double r = sqrt(SQR(x)+SQR(y)+SQR(z));
        return (*options.Q)(t)/(4*PI*r);
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 1
#else // P4_TO_P8
    static struct:cf_t{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x) - options.rot*PI/180;
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
        double ur    = -(*options.Q)(t)/(2*PI*r);

        if (fabs(x-options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
          return x > 0 ? ur*cos(theta):-ur*cos(theta);
        else if (fabs(y-options.xmax[1]) < EPS || fabs(y - options.xmin[1]) < EPS)
          return y > 0 ? ur*sin(theta):-ur*sin(theta);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 0
#if 1
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double y) const {
        double r = sqrt(SQR(x)+SQR(y));
        return -(*options.Q)(t)/(2*PI) * log(r);
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 1
#endif // P4_TO_P8

    // set parameters specific to this test
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "Flat") {

    options.xmin[0]     =  0;
    options.xmin[1]     = options.xmin[2] = 0;
    options.xmax[0]     = 10;
    options.xmax[1]     = options.xmax[2] = 1;
    options.ntr[0]      = 10;
    options.ntr[1]      = options.ntr[2] = 1;
    options.periodic[0] = false;
    options.periodic[1] = options.periodic[2] = true;
    options.lmin        = 2;
    options.lmax        = 5;
    options.method      = "semi_lagrangian";
    options.cfl         = 2;
    options.dtmax       = 5e-3;
    options.dts         = 1e-1;
    options.alpha       = 0;

#ifdef P4_TO_P8
    static struct:cf_t{
      double operator()(double x, double y, double z) const  {
        return 0.1 + 0.01*cos(2*PI*y*4)*cos(2*PI*z*4) - x;
      }
    } interface; interface.lip = options.lip;

#if 1
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double, double) const { return NEUMANN; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double, double) const {
        if (fabs(x - options.xmax[0]) < EPS)
          return -(*options.Q)(t);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 0
#if 0
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double, double) const {
        if (fabs(x - options.xmax[0]) < EPS)
          return -(*options.Q)(t)*(options.xmax[0]-options.xmin[0]);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 1
#else // P4_TO_P8
    static struct:cf_t{
      double operator()(double x, double y) const  {
        return 0.1 + 0.05*cos(2*PI*y*4) - x;
      }
    } interface; interface.lip = options.lip;

#if 1
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return NEUMANN; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double) const {
        if (fabs(x - options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
          return -(*options.Q)(t);
        else
          return 0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 0
#if 0
    static struct:wall_bc_t{
      BoundaryConditionType operator()(double, double) const { return DIRICHLET; }
    } bc_wall_type;

    static struct:cf_t{
      double operator()(double x, double) const {
        return -(*options.Q)(t)*(x-options.xmin[0]);
      }
    } bc_wall_value; bc_wall_value.t = 0;
#endif // #if 1
#endif // P4_TO_P8

    // set parameters specific to this test
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else {
    throw std::invalid_argument("Unknown test");
  }

  // overwrite default values from stdin
  options.lmin   = cmd.get("lmin", options.lmin);
  options.lmax   = cmd.get("lmax", options.lmax);
  options.cfl    = cmd.get("cfl", options.cfl);
  options.dts    = cmd.get("dts", options.dts);
  options.dtmax  = cmd.get("dtmax", options.dtmax);
  options.method = cmd.get("method", options.method);
  options.alpha  = cmd.get("alpha", options.alpha);
  options.rot    = cmd.get("rot", options.rot);
  options.itmax  = cmd.get("itmax", options.itmax);
  options.Ca     = cmd.get("Ca", options.Ca);
}

int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: Ek_Fingering -- OneFluid Solver");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // setup the parameters
  set_parameters(argc, argv);

  conn = my_p4est_brick_new(options.ntr, options.xmin, options.xmax, &brick, options.periodic);

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
  Vec phi, pressure, potential;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &pressure);
  VecCreateGhostNodes(p4est, nodes, &potential);
  sample_cf_on_nodes(p4est, nodes, *options.interface, phi);

  // set up the solver
  one_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_injection_rates(*options.Q, *options.I, options.alpha);
  solver.set_properties(*options.K_D, *options.K_EO, *options.gamma);
  solver.set_bc_wall(*options.bc_wall_type, *options.bc_wall_value);

  const char* outdir = getenv("OUT_DIR");
  if (!outdir)
    throw std::runtime_error("You must set the $OUT_DIR enviroment variable");

  ostringstream oss;
  oss << outdir << "/one_fluid/" << options.test << "/" << options.method
      << "/" << mpi.size() << "p";
  const string folder = oss.str();
  ostringstream command;
  command << "mkdir -p " << folder;
  if (mpi.rank() == 0)
    system(command.str().c_str());
  MPI_Barrier(mpi.comm());

  char vtk_name[FILENAME_MAX];

  {
    double *phi_p, *pressure_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(pressure, &pressure_p);
    sprintf(vtk_name, "%s/%s_%dd.%04d", folder.c_str(), options.method.c_str(), P4EST_DIM, 0);
    PetscPrintf(mpi.comm(), "Saving file %s\n", vtk_name);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE, 2, 0, vtk_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "pressure", pressure_p);
    VecRestoreArray(phi, &phi_p);
    VecRestoreArray(pressure, &pressure_p);
  }

  double dt = 0, t = 0;
  int is = 1, it = 0;
  do {
    dt = solver.solve_one_step(t, phi, pressure, potential, options.method, options.cfl, options.dtmax);
    it++; t += dt;

    p4est_gloidx_t num_nodes = 0;
    for (int r = 0; r<mpi.size(); r++)
      num_nodes += nodes->global_owned_indeps[r];
    PetscPrintf(mpi.comm(), "i = %04d n = %6d t = %1.5f dt = %1.5e\n", it, num_nodes, t, dt);

    // save vtk
    if (t >= is*options.dts) {
      // reinitialize the solution before writing it
      {
        my_p4est_hierarchy_t h(p4est, ghost, &brick);
        my_p4est_node_neighbors_t ngbd(&h, nodes);
        ngbd.init_neighbors();

        my_p4est_level_set_t ls(&ngbd);
        ls.reinitialize_2nd_order(phi);
      }

      double *phi_p, *pressure_p;
      VecGetArray(phi, &phi_p);
      VecGetArray(pressure, &pressure_p);
      sprintf(vtk_name, "%s/%s_%dd.%04d", folder.c_str(), options.method.c_str(), P4EST_DIM, ++is);
      PetscPrintf(mpi.comm(), "Saving file %s\n", vtk_name);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE, 2, 0, vtk_name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "pressure", pressure_p);
      VecRestoreArray(phi, &phi_p);
      VecRestoreArray(pressure, &pressure_p);
    }

    // save the error if this is a modal analysis
    if (options.test == "circle") {
      my_p4est_hierarchy_t h(p4est, ghost, &brick);
      my_p4est_node_neighbors_t ngbd(&h, nodes);
      ngbd.init_neighbors();

      my_p4est_interpolation_nodes_t interp(&ngbd);
      interp.set_input(phi, quadratic);

      // we only ask the root to compute the interpolation
      int ntheta = mpi.rank() == 0 ? 3600:0;
      vector<double> err(ntheta);
      for (int n=0; n<ntheta; n++) {
        double r = 1+t;
        double theta = 2*PI*n/ntheta;
        double x[] = {r*cos(theta), r*sin(theta)};
        interp.add_point(n, x);
      }
      interp.interpolate(err.data());

      if (mpi.rank() == 0) {
        ostringstream filename;
        filename << folder
                 << "/err_" << options.lmax
                 << "_" << options.mode
                 << "_" << it << ".txt";
        FILE *file = fopen(filename.str().c_str(), "w");
        fprintf(file, "%% theta \t err\n");
        for (int n = 0; n<ntheta; n++) {
          double theta = 2*PI*n/ntheta;
          fprintf(file, "%1.6f % -1.12e\n", theta, err[n]);
        }
        fclose(file);
      }
    }
  } while (it < options.itmax);

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

