/*
 * Title: extend
 * Description:
 * Author: bochkov.ds@gmail.com
 * Date Created: 10-30-2019
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
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_shapes.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_shapes.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>

using namespace std;

const static std::string main_description =
    "In this example, we illustrate extrpolation of fields. We consider both\n"
    "extrapolation over interface and extrapolation from interface.\n"
    "In the first procedure a field defined only in the negative domain\n"
    "is extrapolated (as a constant, linearly or quadratically) into the positive domain.\n"
    "In the second procedure a field defined everywhere is extrapolated as a constant\n"
    "in the normal direction from an interface.\n"
    "Developer: Daniil Bochkov (dbochkov@ucsb.edu), October 2019.\n";


// ---------------------------------------------------------
// define parameters
// ---------------------------------------------------------
param_list_t pl;

// grid parameters
param_t<int>    lmin       (pl, 4,   "lmin",       "Min level of refinement (default: 4)");
param_t<int>    lmax       (pl, 4,   "lmax",       "Max level of refinement (default: 4)");
param_t<double> lip        (pl, 1.2, "lip",        "Lipschitz constant (default: 1.2)");
param_t<int>    num_splits (pl, 5,   "num_splits", "Number of successive refinements (default: 5)");

// problem set-up (points of iterpolation and function to interpolate)
param_t<int> test_domain(pl, 0, "test_domain", "Test domain (default: 0):\n"
                                               "    0 - sphere \n"
                                               "    1 - flower shaped"
                                               "    2 - union of two spheres"
                                               "    3 - difference of two spheres");

param_t<int> test_function(pl, 0, "test_function", "Test function (default: 0):\n"
                                                   "    0 - sin(x)*cos(y)\n"
                                                   "    1 - ... (more to be added)");

// method set-up
param_t<bool>   show_convergence(pl, 0,  "show_convergence", "Show convergence as iterations performed (default: 0)");
param_t<bool>   use_full        (pl, 0,  "use_full"        , "Extend only normal derivatives (0) or all derivatives in Cartesian directions (1) (default: 0)");
param_t<int>    num_iterations  (pl, 50, "num_iterations"  , "Number of iterations (default: 50)\n");
param_t<double> band            (pl, 2,  "band"            , "Band around interface (in diagonal lengths) to check for accuracy (default: 2)\n");


// level-set function
#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_domain.val)
    {
      case 0: return -(0.501 - sqrt(SQR(x) + SQR(y) + SQR(z)));
      case 1:
        static flower_shaped_domain_t flower(0.501, .0, .0, .0, .15, 1);
        return flower.phi(x,y,z);
      case 2:
        return MIN(-(0.501 - sqrt(SQR(x+.1) + SQR(y+.3) + SQR(z+.2))),
                   -(0.401 - sqrt(SQR(x-.2) + SQR(y-.2) + SQR(z-.1))));
      case 3:
        return MAX(-(0.501 - sqrt(SQR(x+.0) + SQR(y+.0) + SQR(z+.0))),
                    (0.401 - sqrt(SQR(x-.4) + SQR(y-.3) + SQR(z-.2))));
      default:
        throw std::invalid_argument("Invalid domain");
    }
  }
} phi_cf;
#else
struct: CF_2 {
  double operator()(double x, double y) const {
    switch (test_domain.val)
    {
      case 0: return -(0.501 - sqrt(SQR(x) + SQR(y)));
      case 1:
        static flower_shaped_domain_t flower(0.501, .0, .0, .15, 1);
        return flower.phi(x,y);
      case 2:
        return MIN(-(0.501 - sqrt(SQR(x+.1) + SQR(y+.3))),
                   -(0.401 - sqrt(SQR(x-.2) + SQR(y-.2))));
      case 3:
        return MAX(-(0.501 - sqrt(SQR(x+.0) + SQR(y+.0))),
                    (0.401 - sqrt(SQR(x-.4) + SQR(y-.3))));
      default:
        throw std::invalid_argument("Invalid domain");
    }
  }
} phi_cf;
#endif

#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_function.val)
    {
      case 0: return sin(x)*cos(y)*exp(z);
      default:
        throw std::invalid_argument("Invalid test function");
    }
  }
} f_cf;
#else
struct: CF_2 {
  double operator()(double x, double y) const {
    switch (test_function.val)
    {
      case 0: return sin(x)*cos(y);
      default:
        throw std::invalid_argument("Invalid test function");
    }
  }
} f_cf;
#endif

int main(int argc, char** argv)
{
  PetscErrorCode ierr;

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // get parameter values from command line
  cmdParser cmd;
  pl.initialize_parser(cmd);
  if (cmd.parse(argc, argv, main_description)) return 0;
  pl.set_from_cmd_all(cmd);

  // stopwatch
  parStopWatch w;
  w.start("Running example: extension");

  // variables to store current and last max extrapolation errors
  double error_const_max_old = 0;
  double error_const_max_cur = 0;

  double error_linear_max_old = 0;
  double error_linear_max_cur = 0;

  double error_quadratic_max_old = 0;
  double error_quadratic_max_cur = 0;

  // ---------------------------------------------------------
  // loop through all grid resolutions
  // ---------------------------------------------------------
  for (int iter = 0; iter < num_splits(); ++iter)
  {
    // ---------------------------------------------------------
    // create grid
    // ---------------------------------------------------------
    // p4est variables
    p4est_t*              p4est;
    p4est_nodes_t*        nodes;
    p4est_ghost_t*        ghost;
    p4est_connectivity_t* conn;
    my_p4est_brick_t      brick;

    // domain size information
    const double xyz_min[]  = {-1, -1, -1};
    const double xyz_max[]  = { 1,  1,  1};
    const int    n_xyz[]    = { 1,  1,  1};
    const int    periodic[] = { 0,  0,  0};
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    // create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

    // refine based on distance to a level-set
    splitting_criteria_cf_t sp(lmin()+iter, lmax()+iter, &phi_cf, lip());
    p4est->user_pointer = &sp;
    for (int i = 0; i < lmax()+iter; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_TRUE, NULL);
    }

    // create ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // create node structure
    nodes = my_p4est_nodes_new(p4est, ghost);

    // create hierarchy
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);

    // create and initialize neighborhood information
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();

    // ---------------------------------------------------------
    // define variables
    // ---------------------------------------------------------
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    Vec phi;
    Vec f_exact;
    Vec f_const;
    Vec f_linear;
    Vec f_quadratic;
    Vec f_flat;
    Vec error_const;
    Vec error_linear;
    Vec error_quadratic;

    // double pointers to access Vec's data
    double *phi_ptr;
    double *f_exact_ptr;
    double *f_const_ptr;
    double *f_linear_ptr;
    double *f_quadratic_ptr;
    double *f_flat_ptr;
    double *error_const_ptr;
    double *error_linear_ptr;
    double *error_quadratic_ptr;

    // ---------------------------------------------------------
    // allocate memory for Vec's
    // ---------------------------------------------------------
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);

    ierr = VecDuplicate(phi, &f_exact);     CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &f_const);     CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &f_linear);    CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &f_quadratic); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &f_flat);      CHKERRXX(ierr);

    ierr = VecDuplicate(phi, &error_const);     CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &error_linear);    CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &error_quadratic); CHKERRXX(ierr);

    // ---------------------------------------------------------
    // sample level-set function and test function at grid points
    // ---------------------------------------------------------
    sample_cf_on_nodes(p4est, nodes, phi_cf, phi);
    sample_cf_on_nodes(p4est, nodes, f_cf, f_exact);
    VecCopyGhost(f_exact, f_const);
    VecCopyGhost(f_exact, f_linear);
    VecCopyGhost(f_exact, f_quadratic);


    // reset fields for phi > 0
    ierr = VecGetArray(phi,         &phi_ptr);
    ierr = VecGetArray(f_const,     &f_const_ptr);
    ierr = VecGetArray(f_linear,    &f_linear_ptr);
    ierr = VecGetArray(f_quadratic, &f_quadratic_ptr);

    foreach_local_node(n, nodes)
    {
      if (phi_ptr[n] > 0)
      {
        f_const_ptr    [n] = 0;
        f_linear_ptr   [n] = 0;
        f_quadratic_ptr[n] = 0;
      }
    }

    ierr = VecRestoreArray(phi,         &phi_ptr);
    ierr = VecRestoreArray(f_const,     &f_const_ptr);
    ierr = VecRestoreArray(f_linear,    &f_linear_ptr);
    ierr = VecRestoreArray(f_quadratic, &f_quadratic_ptr);


    // ---------------------------------------------------------
    // perform extrapolations
    // ---------------------------------------------------------
    my_p4est_level_set_t ls(&ngbd);

    if (show_convergence())
    {
      ls.set_show_convergence(1);
      ls.set_show_convergence_band(band()*diag_min);
    }

    if (use_full())
    {
      ls.extend_Over_Interface_TVD_Full(phi, f_const,     num_iterations(), 0); // constant extrapolation
      ls.extend_Over_Interface_TVD_Full(phi, f_linear,    num_iterations(), 1); // linear extrapolation
      ls.extend_Over_Interface_TVD_Full(phi, f_quadratic, num_iterations(), 2); // quadratic extrapolationn
    } else {
      ls.extend_Over_Interface_TVD(phi, f_const,     num_iterations(), 0); // constant extrapolation
      ls.extend_Over_Interface_TVD(phi, f_linear,    num_iterations(), 1); // linear extrapolation
      ls.extend_Over_Interface_TVD(phi, f_quadratic, num_iterations(), 2); // quadratic extrapolationn
    }

    ls.extend_from_interface_to_whole_domain_TVD(phi, f_exact, f_flat, num_iterations()); // constant extrapolation in both directions (flattening)

    // ---------------------------------------------------------
    // compute extension errors
    // ---------------------------------------------------------
    // get access to Vec's data
    ierr = VecGetArray(phi,             &phi_ptr);
    ierr = VecGetArray(f_exact,         &f_exact_ptr);
    ierr = VecGetArray(f_const,         &f_const_ptr);
    ierr = VecGetArray(f_linear,        &f_linear_ptr);
    ierr = VecGetArray(f_quadratic,     &f_quadratic_ptr);
    ierr = VecGetArray(error_const,     &error_const_ptr);
    ierr = VecGetArray(error_linear,    &error_linear_ptr);
    ierr = VecGetArray(error_quadratic, &error_quadratic_ptr);

    error_const_max_old = error_const_max_cur;
    error_const_max_cur = 0;

    error_linear_max_old = error_linear_max_cur;
    error_linear_max_cur = 0;

    error_quadratic_max_old = error_quadratic_max_cur;
    error_quadratic_max_cur = 0;

    // loop through local nodes
    foreach_local_node(n, nodes)
    {
      if (phi_ptr[n] > 0 &&
          phi_ptr[n] < band()*diag_min)
      {
        error_const_ptr    [n] = fabs(f_exact_ptr[n] - f_const_ptr    [n]);
        error_linear_ptr   [n] = fabs(f_exact_ptr[n] - f_linear_ptr   [n]);
        error_quadratic_ptr[n] = fabs(f_exact_ptr[n] - f_quadratic_ptr[n]);
      }

      // store max error
      error_const_max_cur     = MAX(error_const_max_cur,     error_const_ptr    [n]);
      error_linear_max_cur    = MAX(error_linear_max_cur,    error_linear_ptr   [n]);
      error_quadratic_max_cur = MAX(error_quadratic_max_cur, error_quadratic_ptr[n]);
    }

    // find max error among all processes
    int mpiret;
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_const_max_cur    , 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_linear_max_cur   , 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_quadratic_max_cur, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

    // rectore access to Vec's data
    ierr = VecRestoreArray(phi,             &phi_ptr);
    ierr = VecRestoreArray(f_exact,         &f_exact_ptr);
    ierr = VecRestoreArray(f_const,         &f_const_ptr);
    ierr = VecRestoreArray(f_linear,        &f_linear_ptr);
    ierr = VecRestoreArray(f_quadratic,     &f_quadratic_ptr);
    ierr = VecRestoreArray(error_const,     &error_const_ptr);
    ierr = VecRestoreArray(error_linear,    &error_linear_ptr);
    ierr = VecRestoreArray(error_quadratic, &error_quadratic_ptr);

    // print out max error
    ierr = PetscPrintf(mpi.comm(), "Grid levels: %2d / %2d, const: %1.2e / %1.2g, linear: %1.2e / %1.3g, quadratic: %1.2e / %1.3g\n", lmin()+iter, lmax()+iter,
                       error_const_max_cur,     log(error_const_max_old/error_const_max_cur)/log(2),
                       error_linear_max_cur,    log(error_linear_max_old/error_linear_max_cur)/log(2),
                       error_quadratic_max_cur, log(error_quadratic_max_old/error_quadratic_max_cur)/log(2)); CHKERRXX(ierr);

    // ---------------------------------------------------------
    // save the grid and data into vtk
    // ---------------------------------------------------------
    const char *out_dir = getenv("OUT_DIR");
    if (!out_dir) out_dir = ".";
    else
    {
      std::ostringstream command;
      command << "mkdir -p " << out_dir;
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
    }

    std::ostringstream oss;
    oss << out_dir
        << "/extend"
       #ifdef P4_TO_P8
        << "_3d"
       #else
        << "_2d"
       #endif
        << "_nprocs_" << p4est->mpisize
        << "_split_" << iter;

    ierr = VecGetArray(phi,             &phi_ptr);
    ierr = VecGetArray(f_exact,         &f_exact_ptr);
    ierr = VecGetArray(f_const,         &f_const_ptr);
    ierr = VecGetArray(f_linear,        &f_linear_ptr);
    ierr = VecGetArray(f_quadratic,     &f_quadratic_ptr);
    ierr = VecGetArray(f_flat,          &f_flat_ptr);
    ierr = VecGetArray(error_const,     &error_const_ptr);
    ierr = VecGetArray(error_linear,    &error_linear_ptr);
    ierr = VecGetArray(error_quadratic, &error_quadratic_ptr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           9, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi",             phi_ptr,
                           VTK_POINT_DATA, "f_exact",         f_exact_ptr,
                           VTK_POINT_DATA, "f_const",         f_const_ptr,
                           VTK_POINT_DATA, "f_linear",        f_linear_ptr,
                           VTK_POINT_DATA, "f_quaratic",      f_quadratic_ptr,
                           VTK_POINT_DATA, "f_flat",          f_flat_ptr,
                           VTK_POINT_DATA, "error_const",     error_const_ptr,
                           VTK_POINT_DATA, "error_linear",    error_linear_ptr,
                           VTK_POINT_DATA, "error_quadratic", error_quadratic_ptr);

    ierr = VecRestoreArray(phi,             &phi_ptr);
    ierr = VecRestoreArray(f_exact,         &f_exact_ptr);
    ierr = VecRestoreArray(f_const,         &f_const_ptr);
    ierr = VecRestoreArray(f_linear,        &f_linear_ptr);
    ierr = VecRestoreArray(f_quadratic,     &f_quadratic_ptr);
    ierr = VecRestoreArray(f_flat,          &f_flat_ptr);
    ierr = VecRestoreArray(error_const,     &error_const_ptr);
    ierr = VecRestoreArray(error_linear,    &error_linear_ptr);
    ierr = VecRestoreArray(error_quadratic, &error_quadratic_ptr);

    // ---------------------------------------------------------
    // destroy the structures and release allocated memory
    // ---------------------------------------------------------
    ierr = VecDestroy(phi);
    ierr = VecDestroy(f_exact);
    ierr = VecDestroy(f_const);
    ierr = VecDestroy(f_linear);
    ierr = VecDestroy(f_quadratic);
    ierr = VecDestroy(f_flat);
    ierr = VecDestroy(error_const);
    ierr = VecDestroy(error_linear);
    ierr = VecDestroy(error_quadratic);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(conn, &brick);
  }


  w.stop(); w.read_duration();
}
