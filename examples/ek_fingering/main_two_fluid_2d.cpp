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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include "two_fluid_solver_3d.h"
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
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
  int lmin, lmax, iter, it_reinit;
  double lip, Ca, cfl, dts, dtmax, M, L, rot;
  double xmin[3], xmax[3];
  int ntr[3];
  int periodic[3];
  int mode;
  double eps;
  string test, method;
  BoundaryConditionType bcw;

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
  cmd.add_option("it_reinit", "number of ierations before reinit");
  cmd.add_option("cfl", "the CFL number");
  cmd.add_option("dts", "dt for saving vtk files");
  cmd.add_option("dtmax", "max dt to use when solving");
  cmd.add_option("L", "domain length");
  cmd.add_option("rot", "angle to rotate the interface about");
  cmd.add_option("M", "The viscosity ratio of inner to outer fluid");
  cmd.add_option("method", "method for solving the jump equation");
  cmd.add_option("mode", "perturbation mode used for analysis");
  cmd.add_option("eps", "perturbation amplitude to be added to the interface");
  cmd.add_option("bcw", "bc type on the wall");
  cmd.add_option("test", "Which test to run?, Options are:\n"
      "\tcircle\n"
      "\tflat\n"
      "\tFastShelley04_Fig12\n");

  cmd.parse(argc, argv);

  options.test = cmd.get<string>("test", "FastShelley04_Fig12");
  options.method = cmd.get<string>("method", "voronoi");
  options.bcw    = cmd.get("bcw", DIRICHLET);

  // set default values
  options.ntr[0]  = options.ntr[1]  = options.ntr[2]  =  1;
  options.xmin[0] = options.xmin[1] = options.xmin[2] = -1;
  options.xmax[0] = options.xmax[1] = options.xmax[2] =  1;
  options.periodic[0] = options.periodic[1] = options.periodic[2] = false;

  options.lip  = 1.2;
  options.cfl  = 1;
  options.it_reinit = 10;
  options.rot = cmd.get("rot", 0);
  options.iter = numeric_limits<int>::max();

  if (options.test == "circle") {
    // set interface
    options.L = cmd.get("L", 10);
    options.xmin[0] = options.xmin[1] = options.xmin[2] = -options.L + EPS;
    options.xmax[0] = options.xmax[1] = options.xmax[2] =  options.L;
    options.lmax    = 10;
    options.lmin    = 5;
    options.iter    = 10;
    options.dtmax   = 1e-3;
    options.dts     = 1e-1;
    options.Ca      = 250;
    options.mode    = cmd.get("mode", 0);
    options.eps     = cmd.get("eps", 1e-2);
    options.lip     = cmd.get("lip", options.lip);
    options.M       = 1;
    options.it_reinit = options.iter;

    static struct:CF_2{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x) - M_PI/180 * options.rot;
        double r     = sqrt(SQR(x)+SQR(y));
        return r - (1+options.eps*cos(options.mode*theta));
      }
    } interface; interface.lip = options.lip;
    options.interface = &interface;

    static struct:CF_1{
      double operator()(double t) const { return 2*PI*(1.0 + t); }
    } Q;

    static struct:WallBC2D{
      BoundaryConditionType operator()(double, double) const { return options.bcw; }
    } bc_wall_type;

    static struct:CF_2{
      double operator()(double x, double y) const {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));

        if (options.bcw == DIRICHLET) {
          return -(*options.Q)(t)/(2*PI)*log(r);

        } else if (options.bcw == NEUMANN) {
          double ur    = -(*options.Q)(t)/(2*PI*r);

          if (fabs(x-options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
            return x > 0 ? ur*cos(theta):-ur*cos(theta);
          else if (fabs(y-options.xmax[1]) < EPS || fabs(y - options.xmin[1]) < EPS)
            return y > 0 ? ur*sin(theta):-ur*sin(theta);
          else
            return 0;
        } else {
          throw runtime_error("Worng boundary condition");
        }
      }
    } bc_wall_value; bc_wall_value.t = 0;
    /*
       static struct:WallBC2D {
       BoundaryConditionType operator()(double, double) const {
       return DIRICHLET;
       }
       } bc_wall_type;

       static struct:CF_2 {
       double operator()(double x, double y) const {
       double r = sqrt(SQR(x)+SQR(y));
       return - ( 1.0/(options.Ca*(1+t)) + (1+t) * log(r/(1+t)) );
       }
       } bc_wall_value; bc_wall_value.t = 0;
     */

    options.Q             = &Q;
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "flat") {
    // set interface
    options.L = cmd.get("L", 5);
    options.xmin[0] = -options.L + EPS; options.xmin[1] = options.xmin[2] = -0.5;
    options.xmax[0] =  options.L;       options.xmax[1] = options.xmax[2] =  0.5;
    options.ntr[0]  = 2*options.L; options.ntr[1] = options.ntr[2] = 1;
    options.periodic[0] = false; options.periodic[1] = options.periodic[2] = false;
    options.lmax    = 7;
    options.lmin    = 2;
    options.iter    = 10;
    options.dtmax   = 1e-3;
    options.dts     = 1e-1;
    options.Ca      = 250;
    options.mode    = cmd.get("mode", 0);
    options.eps     = cmd.get("eps", 5e-3);
    options.lip     = cmd.get("lip", options.lip);
    options.M       = 1;

    static struct:CF_2{
      double operator()(double x, double y) const  {
        return x-options.eps*cos(2*M_PI*options.mode*y);
      }
    } interface; interface.lip = options.lip;
    options.interface = &interface;

    static struct:CF_1{
      double operator()(double) const { return 0; }
    } Q;

    static struct:WallBC2D{
      BoundaryConditionType operator()(double x, double y) const {
        if (fabs(y - options.xmin[1]) < EPS || fabs(y - options.xmax[1]) < EPS)
          return NEUMANN;
        else
          return x > 0 ? NEUMANN:DIRICHLET;
      }
    } bc_wall_type;

    static struct:CF_2{
      double operator()(double x, double y) const {
        if (fabs(y - options.xmin[1]) < EPS || fabs(y - options.xmax[1]) < EPS)
          return 0;
        else
          return x > 0 ? -1.0:0;
      }
    } bc_wall_value; bc_wall_value.t = 0;
    /*
       static struct:WallBC2D {
       BoundaryConditionType operator()(double, double) const {
       return DIRICHLET;
       }
       } bc_wall_type;

       static struct:CF_2 {
       double operator()(double x, double y) const {
       double r = sqrt(SQR(x)+SQR(y));
       return - ( 1.0/(options.Ca*(1+t)) + (1+t) * log(r/(1+t)) );
       }
       } bc_wall_value; bc_wall_value.t = 0;
     */

    options.Q             = &Q;
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "FastShelley04_Fig12") {
    options.L = cmd.get("L", 10);
    options.xmin[0]   = options.xmin[1] = options.xmin[2] = -options.L + EPS;
    options.xmax[0]   = options.xmax[1] = options.xmax[2] =  options.L;
    options.ntr[0]    = options.ntr[1]  = options.ntr[2]  = 1;
    options.method    = "voronoi";
    options.lmin      = 5;
    options.lmax      = 10;
    options.dtmax     = 5e-3;
    options.dts       = 1e-1;
    options.Ca        = 250;
    options.M         = 1e-4;

    static struct:CF_1{
      double operator()(double t) const { return 2*PI*(1+t); }
    } Q;

#ifdef P4_TO_P8
    static struct:cf_t{
      double operator()(double x, double y, double z) const  {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y)+SQR(z));
        double phi   = acos(z/MAX(r,1E-12));

        return r - ( 1.0+0.1*(cos(3*theta)+sin(2*theta))*pow(sin(2*phi),2) );
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
#else // 2D
    static struct:cf_t{
      double operator()(double x, double y) const  {
        double theta = atan2(y,x) - options.rot * PI/180;
        double r     = sqrt(SQR(x)+SQR(y));

        return r - ( 1.0+0.1*(cos(3*theta)+sin(2*theta)) );
      }
    } interface; interface.lip = options.lip;

    static struct:WallBC2D{
      BoundaryConditionType operator()(double, double) const { return options.bcw; }
    } bc_wall_type;

    static struct:CF_2{
      double operator()(double x, double y) const {
        double theta = atan2(y,x);
        double r     = sqrt(SQR(x)+SQR(y));

        if (options.bcw == DIRICHLET) {
          return -(*options.Q)(t)/(2*PI)*log(r);
        } else if (options.bcw == NEUMANN) {
          double ur    = -(*options.Q)(t)/(2*PI*r);

          if (fabs(x-options.xmax[0]) < EPS || fabs(x - options.xmin[0]) < EPS)
            return x > 0 ? ur*cos(theta):-ur*cos(theta);
          else if (fabs(y-options.xmax[1]) < EPS || fabs(y - options.xmin[1]) < EPS)
            return y > 0 ? ur*sin(theta):-ur*sin(theta);
          else
            return 0;
        } else {
          throw runtime_error("Worng boundary condition");
        }
      }
    } bc_wall_value; bc_wall_value.t = 0;

#endif // P4_TO_P8

    // set parameters specific to this test
    options.Q              = &Q;
    options.interface      = &interface;
    options.bc_wall_type   = &bc_wall_type;
    options.bc_wall_value  = &bc_wall_value;

  } else {
    throw std::invalid_argument("Unknown test");
  }

  // overwrite default values from stdin
  options.lmin      = cmd.get("lmin",      options.lmin);
  options.lmax      = cmd.get("lmax",      options.lmax);
  options.iter      = cmd.get("iter",      options.iter);
  options.lip       = cmd.get("lip",       options.lip);
  options.Ca        = cmd.get("Ca",        options.Ca);
  options.cfl       = cmd.get("cfl",       options.cfl);
  options.dts       = cmd.get("dts",       options.dts);
  options.dtmax     = cmd.get("dtmax",     options.dtmax);
  options.M         = cmd.get("M",         options.M);
  options.method    = cmd.get("method",    options.method);
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
  Vec phi, press_m, press_p;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &press_m);
  VecCreateGhostNodes(p4est, nodes, &press_p);
  sample_cf_on_nodes(p4est, nodes, *options.interface, phi);

  // set up the solver
  two_fluid_solver_t solver(p4est, ghost, nodes, brick);
  solver.set_properties(options.M, options.Ca, *options.Q);
  solver.set_bc_wall(*options.bc_wall_type, *options.bc_wall_value);


  const char* outdir = getenv("OUT_DIR");
  if (!outdir)
    throw std::runtime_error("You must set the $OUT_DIR enviroment variable");

  ostringstream folder;
  folder << outdir << "/two_fluid/" << options.test
         << "/mue_" << options.M
         << "/" << mpi.size() << "p";

  if (mpi.rank() == 0) system(("mkdir -p " + folder.str()).c_str());
  MPI_Barrier(mpi.comm());
  char vtk_name[FILENAME_MAX];

  {
    double *phi_p, *press_m_p, *press_p_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(press_m, &press_m_p);
    VecGetArray(press_p, &press_p_p);

    sprintf(vtk_name, "%s/two_fluid_%d_%d_%1.1f.%04d", folder.str().c_str(),
            options.lmin, options.lmax, options.lip, 0);
    PetscPrintf(mpi.comm(), "Saving %s\n", vtk_name);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, vtk_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "press_m", press_m_p,
                           VTK_POINT_DATA, "press_p", press_p_p);
    VecRestoreArray(phi, &phi_p);
  }

  double dt = 0, t = 0;
  int is = 1, it = 0;
  do {
    dt = solver.solve_one_step(t, phi, press_m, press_p,
                               options.cfl, options.dtmax, options.method);
    it++; t += dt;

    p4est_gloidx_t num_nodes = 0;
    for (int r = 0; r<mpi.size(); r++)
      num_nodes += nodes->global_owned_indeps[r];
    PetscPrintf(mpi.comm(), "it = %04d n = %6d t = %1.5f dt = %1.5e\n", it, num_nodes, t, dt);

    // reinitialize the solution before writing it
    if (it % options.it_reinit == 0){
      PetscPrintf(mpi.comm(), "Reinitalizing ... ");

      my_p4est_hierarchy_t h(p4est, ghost, &brick);
      my_p4est_node_neighbors_t ngbd(&h, nodes);
      ngbd.init_neighbors();

      my_p4est_level_set_t ls(&ngbd);
      ls.reinitialize_2nd_order(phi);

      PetscPrintf(mpi.comm(), "done!\n");
    }
    if (t >= is*options.dts) {

     // save vtk
      double *phi_p, *press_m_p, *press_p_p;
      VecGetArray(phi, &phi_p);
      VecGetArray(press_m, &press_m_p);
      VecGetArray(press_p, &press_p_p);

      sprintf(vtk_name, "%s/two_fluid_%d_%d_%1.1f.%04d", folder.str().c_str(),
              options.lmin, options.lmax, options.lip,
              is++);
      PetscPrintf(mpi.comm(), "Saving %s\n", vtk_name);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             3, 0, vtk_name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "press_m", press_m_p,
                             VTK_POINT_DATA, "press_p", press_p_p);
      VecRestoreArray(phi, &phi_p);
      foreach_node (n,nodes) {
        if (std::isnan(press_m_p[n])) cout << "nan in p^- for n = " << n << endl;
        if (std::isnan(press_p_p[n])) cout << "nan in p^+ for n = " << n << endl;
      }

      VecRestoreArray(press_m, &press_m_p);
      VecRestoreArray(press_p, &press_p_p);
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
        filename << folder.str() << "/err_" << options.lmax
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

    if (options.test == "flat") {
      my_p4est_hierarchy_t h(p4est, ghost, &brick);
      my_p4est_node_neighbors_t ngbd(&h, nodes);
      ngbd.init_neighbors();

      my_p4est_interpolation_nodes_t interp(&ngbd);
      interp.set_input(phi, quadratic);

      // we only ask the root to compute the interpolation
      int ntheta = mpi.rank() == 0 ? 3600:0;
      vector<double> err(ntheta);
      for (int n=0; n<ntheta; n++) {
        double x[] = {t, options.xmin[1] + (options.xmax[1]-options.xmin[1])/ntheta};
        interp.add_point(n, x);
      }
      interp.interpolate(err.data());

      if (mpi.rank() == 0) {
        ostringstream filename;
        filename << folder.str() << "/err_" << options.lmax
          << "_" << options.mode
          << "_" << it << ".txt";
        FILE *file = fopen(filename.str().c_str(), "w");
        fprintf(file, "%% theta \t err\n");
        for (int n = 0; n<ntheta; n++) {
          double y = options.xmin[1] + (options.xmax[1]-options.xmin[1])/ntheta;
          fprintf(file, "%1.6f % -1.12e\n", y, err[n]);
        }
        fclose(file);
      }
    }

  } while (it < options.iter);

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

