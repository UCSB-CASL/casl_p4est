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
  bool modal;
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
  cmd.add_option("modal", "is this a modal analysis?");
  cmd.add_option("test", "Which test to run?, Options are:\n"
                         "\tcircle\n"
                         "\tflat\n"
                         "\tFastShelley\n");

  cmd.parse(argc, argv);

  options.test = cmd.get<string>("test", "FastShelley");
  options.method = cmd.get<string>("method", "voronoi");
  options.bcw    = cmd.get("bcw", DIRICHLET);
  options.modal = cmd.contains("modal");

  // set default values
  options.ntr[0]  = options.ntr[1]  = options.ntr[2]  =  1;
  options.xmin[0] = options.xmin[1] = options.xmin[2] = -1;
  options.xmax[0] = options.xmax[1] = options.xmax[2] =  1;
  options.periodic[0] = options.periodic[1] = options.periodic[2] = false;

  options.lip       = 5;
  options.cfl       = 1;
  options.it_reinit = 50;
  options.dtmax     = 1e-3;
  options.dts       = 1e-1;
  options.Ca        = 250;
  options.M         = 1e-2;
  options.rot = cmd.get("rot", 0);
  options.iter = numeric_limits<int>::max();

  if (options.test == "circle") {
    // set interface
    options.L = cmd.get("L", 10);
    options.xmin[0] = options.xmin[1] = options.xmin[2] = -options.L + EPS;
    options.xmax[0] = options.xmax[1] = options.xmax[2] =  options.L;
    options.lmax    = 10;
    options.lmin    = 5;
    if (options.modal) options.iter = 10;
    options.mode    = cmd.get("mode", 5);
    options.eps     = cmd.get("eps", 1e-1);
    options.lip     = cmd.get("lip", 5);

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

    options.Q             = &Q;
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "flat") {
    // set interface
    options.L = cmd.get("L", 5);
    if (options.modal) {
      options.xmin[0] = -options.L*PI; options.xmin[1] = options.xmin[2] = -PI;
      options.xmax[0] =  options.L*PI; options.xmax[1] = options.xmax[2] =  PI;
      options.iter    = 10;
    } else {
      options.xmin[0] = -1; options.xmin[1] = options.xmin[2] = -PI;
      options.xmax[0] =  2*options.L*PI-1; options.xmax[1] = options.xmax[2] =  PI;
    }

    options.ntr[0]  = options.L; options.ntr[1] = options.ntr[2] = 1;
    options.periodic[0] = false; options.periodic[1] = options.periodic[2] = true;
    options.lmax    = 8;
    options.lmin    = 3;
    options.mode    = cmd.get("mode", 5);
    options.eps     = cmd.get("eps", 1e-1);
    options.lip     = cmd.get("lip", 5);

    static struct:CF_2{
      double operator()(double x, double y) const  {
        return EPS+x-options.eps*cos(options.mode*y);
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
          return x > 0 ? -1:0;
      }
    } bc_wall_value; bc_wall_value.t = 0;

    options.Q             = &Q;
    options.interface     = &interface;
    options.bc_wall_type  = &bc_wall_type;
    options.bc_wall_value = &bc_wall_value;

  } else if (options.test == "FastShelley") {
    options.L = cmd.get("L", 10);
    options.xmin[0]   = options.xmin[1] = options.xmin[2] = -options.L + EPS;
    options.xmax[0]   = options.xmax[1] = options.xmax[2] =  options.L;
    options.ntr[0]    = options.ntr[1]  = options.ntr[2]  = 1;
    options.lmin      = 5;
    options.lmax      = 10;

    static struct:CF_1{
      double operator()(double t) const { return 2*PI*(1+t); }
    } Q;

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
  options.it_reinit = cmd.get("it_reinit", options.it_reinit);
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
  my_p4est_ghost_expand(p4est, ghost);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // initialize variables
  Vec phi, press_m, press_p;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCreateGhostNodes(p4est, nodes, &press_m); VecSet(press_m, 0);
  VecCreateGhostNodes(p4est, nodes, &press_p); VecSet(press_p, 0);
  sample_cf_on_nodes(p4est, nodes, *options.interface, phi);

  // set up the solver
  auto solver = new two_fluid_solver_t(p4est, ghost, nodes, brick);
  solver->set_properties(options.M, options.Ca, *options.Q);
  solver->set_bc_wall(*options.bc_wall_type, *options.bc_wall_value);


  const char* outdir = getenv("OUT_DIR");
  if (!outdir)
    throw std::runtime_error("You must set the $OUT_DIR enviroment variable");

  ostringstream folder;
  folder << outdir << "/two_fluid" << (options.modal ? "/modal_":"/")
         << options.test
         << "/mue_" << options.M
         << "/" << mpi.size() << "p";

  if (mpi.rank() == 0) system(("mkdir -p " + folder.str()).c_str());
  MPI_Barrier(mpi.comm());
  char vtk_name[FILENAME_MAX];

  {
    PetscPrintf(mpi.comm(), "Reinitalizing ... ");

    my_p4est_hierarchy_t h(p4est, ghost, &brick);
    my_p4est_node_neighbors_t ngbd(&h, nodes);
    ngbd.init_neighbors();

    my_p4est_level_set_t ls(&ngbd);
    ls.reinitialize_2nd_order(phi);
    ls.perturb_level_set_function(phi, EPS);

    PetscPrintf(mpi.comm(), "done!\n");

    double *phi_p, *press_m_p, *press_p_p;
    VecGetArray(phi, &phi_p);
    VecGetArray(press_m, &press_m_p);
    VecGetArray(press_p, &press_p_p);    

    if (options.test == "circle" || options.test == "flat") {
      sprintf(vtk_name, "%s/mode_%d_%d_%d_%1.1f.%04d", folder.str().c_str(),
              options.mode, options.lmin, options.lmax, options.lip, 0);
    } else {
      sprintf(vtk_name, "%s/%d_%d_%1.1f.%04d", folder.str().c_str(),
              options.lmin, options.lmax, options.lip, 0);
    }

    PetscPrintf(mpi.comm(), "Saving %s\n", vtk_name);
    my_p4est_vtk_write_all(p4est, nodes, 0,
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
    dt = solver->solve_one_step(t, phi, press_m, press_p,
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

      if (options.test == "circle" && 0) {
        Vec phi_tmp;
        VecDuplicate(phi, &phi_tmp);
        double *phi_p, *phi_tmp_p;
        VecGetArray(phi, &phi_p);
        VecGetArray(phi_tmp, &phi_tmp_p);
        foreach_node(n,nodes) phi_tmp_p[n] = phi_p[n];

        my_p4est_interpolation_nodes_t interp(&ngbd);
        interp.set_input(phi, quadratic_non_oscillatory);

        double x[P4EST_DIM];
        vector<double> buff(nodes->indep_nodes.elem_count, 0);
        for (int m = 1; m < options.mode; m++) {
          double s = m*2*PI/(double)options.mode;
          foreach_node (n, nodes) {
            node_xyz_fr_n(n, p4est, nodes, x);
            double r = sqrt(SQR(x[0])+SQR(x[1]));
            double t = atan2(x[1], x[0]);

            x[0] = r*cos(t+s);
            x[1] = r*sin(t+s);
            interp.add_point(n, x);
          }
          interp.interpolate(buff.data());
          foreach_node(n, nodes) phi_tmp_p[n] += buff[n];
        }

        foreach_node(n, nodes) phi_p[n] = phi_tmp_p[n]/options.mode;
        VecRestoreArray(phi, &phi_p);
        VecRestoreArray(phi_tmp, &phi_tmp_p);
        VecDestroy(phi_tmp);
      }

      my_p4est_level_set_t ls(&ngbd);
      ls.reinitialize_2nd_order(phi);
      ls.perturb_level_set_function(phi, 1e-8);

      PetscPrintf(mpi.comm(), "done!\n");
    }

    if (t >= is*options.dts) {

      // save vtk
      double *phi_p, *press_m_p, *press_p_p;
      VecGetArray(phi, &phi_p);
      VecGetArray(press_m, &press_m_p);
      VecGetArray(press_p, &press_p_p);

      if (options.test == "circle" || options.test == "flat") {
        sprintf(vtk_name, "%s/mode_%d_%d_%d_%1.1f.%04d", folder.str().c_str(),
                options.mode, options.lmin, options.lmax, options.lip, is++);
      } else {
        sprintf(vtk_name, "%s/%d_%d_%1.1f.%04d", folder.str().c_str(),
                options.lmin, options.lmax, options.lip, is++);
      }
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
    if (options.modal) {
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
      } else if (options.test == "flat") {
        my_p4est_hierarchy_t h(p4est, ghost, &brick);
        my_p4est_node_neighbors_t ngbd(&h, nodes);
        ngbd.init_neighbors();

        my_p4est_interpolation_nodes_t interp(&ngbd);
        interp.set_input(phi, quadratic);

        // we only ask the root to compute the interpolation
        int ntheta = mpi.rank() == 0 ? 3600:0;
        vector<double> err(ntheta);
        for (int n=0; n<ntheta; n++) {
          double x[] = {t, options.xmin[1] + n*(options.xmax[1]-options.xmin[1])/ntheta};
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
            double y = options.xmin[1] + n*(options.xmax[1]-options.xmin[1])/ntheta;
            fprintf(file, "%1.6f % -1.12e\n", y, err[n]);
          }
          fclose(file);
        }
      } else {
        ostringstream oss;
        oss << "Test " << options.test << " does not have any modal analysis. "
            << "Consider removing '-moda' flag or run with a test which supports modal analysis"
            << endl;
        throw invalid_argument(oss.str());
      }
    }

  } while (it < options.iter);

  delete solver;

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

