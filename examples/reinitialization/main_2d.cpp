/*
 * Title: reinitialization
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
    "In this example, we illustrate reinitialization of level-set functions.\n"
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
param_t<int>    test_domain(pl, 1,   "test_domain", "Test domain (default: 0) (exact solutions available only for 0 and 1):\n"
                                                    "    0 - one sphere\n"
                                                    "    1 - three spheres\n"
                                                    "    2 - flower shaped");

// method set-up
param_t<bool>   show_convergence(pl, 0,  "show_convergence", "Show convergence as iterations performed (default: 0)");
param_t<int>    order_in_time   (pl, 2,  "order_in_time"   , "Order of accuracy in time: first (1) or second (2) (default: 2)\n");
param_t<int>    order_in_space  (pl, 2,  "order_in_space"  , "Order of accuracy in space: first (1) or second (2) (default: 2)\n");
param_t<int>    num_iterations  (pl, 50, "num_iterations"  , "Number of iterations  (default: 50)\n");
param_t<double> band            (pl, 3,  "band"            , "Band around interface (in diagonal lengths) to check for accuracy (default: 3)\n");


// level-set function
#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_domain.val)
    {
      case 0: return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
      case 1: return MAX(0.5 - sqrt(SQR(x-.3) + SQR(y-.3) + SQR(z-.1)),
                         0.2 - sqrt(SQR(x-.2) + SQR(y+.5) + SQR(z+.2)),
                         0.3 - sqrt(SQR(x+.5) + SQR(y+.3) + SQR(z+.3)));
      case 2:
        static flower_shaped_domain_t flower(0.5, .0, .0, .0, .15, 1);
        return flower.phi(x,y,z);
      default:
        throw std::invalid_argument("Invalid domain");
    }
  }
} phi_exact_cf;
#else
struct: CF_2 {
  double operator()(double x, double y) const {
    switch (test_domain.val)
    {
      case 0: return 0.5 - sqrt(SQR(x) + SQR(y));
      case 1: return MAX(0.5 - sqrt(SQR(x-.3) + SQR(y-.3)),
                         0.2 - sqrt(SQR(x-.2) + SQR(y+.5)),
                         0.3 - sqrt(SQR(x+.5) + SQR(y+.3)));
      case 2:
        static flower_shaped_domain_t flower(0.5, .0, .0, .15, 1);
        return flower.phi(x,y);
      default:
        throw std::invalid_argument("Invalid domain");
    }
  }
} phi_exact_cf;
#endif

#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    return phi_exact_cf(x,y,z)*(1.+0.5*sin(5.*(x+z))*cos(4.*(y-z)));
  }
} phi_initial_cf;
#else
struct: CF_2 {
  double operator()(double x, double y) const {
    return phi_exact_cf(x,y)*(1.+0.5*sin(5.*x)*cos(4.*y));
  }
} phi_initial_cf;
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
  w.start("Running example: interpolation_nodes");

  // variables to store current and last max interpoolation errors
  double error_max_old = 0;
  double error_max_cur = 0;

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
    splitting_criteria_cf_t sp(lmin()+iter, lmax()+iter, &phi_exact_cf, lip());
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

    Vec phi_initial;
    Vec phi_reinit;
    Vec phi_exact;
    Vec error;    // errors

    // double pointers to access Vec's data
    double *phi_initial_ptr;
    double *phi_reinit_ptr;
    double *phi_exact_ptr;
    double *error_ptr;

    // ---------------------------------------------------------
    // allocate memory for Vec's
    // ---------------------------------------------------------
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_initial); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_initial, &phi_reinit); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_initial, &phi_exact);  CHKERRXX(ierr);
    ierr = VecDuplicate(phi_initial, &error);      CHKERRXX(ierr);

    // ---------------------------------------------------------
    // sample function at grid points
    // ---------------------------------------------------------
    sample_cf_on_nodes(p4est, nodes, phi_initial_cf, phi_initial);
    sample_cf_on_nodes(p4est, nodes, phi_initial_cf, phi_reinit);
    sample_cf_on_nodes(p4est, nodes, phi_exact_cf, phi_exact);

    // ---------------------------------------------------------
    // reinitialize
    // ---------------------------------------------------------
    my_p4est_level_set_t ls(&ngbd);

    if (show_convergence())
    {
      ls.set_show_convergence(1);
      ls.set_show_convergence_band(band()*diag_min);
    }

    if      (order_in_time() == 1 && order_in_space() == 1) ls.reinitialize_1st_order(phi_reinit, num_iterations());
    else if (order_in_time() == 2 && order_in_space() == 2) ls.reinitialize_2nd_order(phi_reinit, num_iterations());
    else if (order_in_time() == 1 && order_in_space() == 2) ls.reinitialize_1st_order_time_2nd_order_space(phi_reinit, num_iterations());
    else if (order_in_time() == 2 && order_in_space() == 1) ls.reinitialize_2nd_order_time_1st_order_space(phi_reinit, num_iterations());
    else
    {
      throw std::invalid_argument("Invalid orders in time/space");
    }

    // ---------------------------------------------------------
    // compute reinitialization error
    // ---------------------------------------------------------
    // get access to Vec's data
    ierr = VecGetArray(phi_reinit, &phi_reinit_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_exact,  &phi_exact_ptr);  CHKERRXX(ierr);
    ierr = VecGetArray(error,      &error_ptr);    CHKERRXX(ierr);

    error_max_old = error_max_cur;
    error_max_cur = 0;

    // loop through local nodes
    foreach_local_node(n, nodes)
    {
      if ( fabs(phi_reinit_ptr[n]) < band()*diag_min)
      {
        error_ptr[n] = fabs(phi_reinit_ptr[n]-phi_exact_ptr[n]);
      }

      // store max error
      error_max_cur = MAX(error_max_cur, error_ptr[n]);
    }

    // find max error among all processes
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_max_cur, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

    // rectore access to Vec's data
    ierr = VecRestoreArray(phi_reinit, &phi_reinit_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_exact,  &phi_exact_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(error,    &error_ptr);    CHKERRXX(ierr);

    // print out max error
    ierr = PetscPrintf(mpi.comm(), "Grid levels: %2d / %2d, max reinitialization error: %1.2e, order: %1.3g\n", lmin()+iter, lmax()+iter, error_max_cur, log(error_max_old/error_max_cur)/log(2)); CHKERRXX(ierr);

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
        << "/reinitialization"
       #ifdef P4_TO_P8
        << "_3d"
       #else
        << "_2d"
       #endif
        << "_nprocs_" << p4est->mpisize
        << "_split_" << iter;

    ierr = VecGetArray(phi_initial, &phi_initial_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_reinit,  &phi_reinit_ptr);  CHKERRXX(ierr);
    ierr = VecGetArray(phi_exact,   &phi_exact_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(error,       &error_ptr);       CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi_initial", phi_initial_ptr,
                           VTK_POINT_DATA, "phi_reinit",  phi_reinit_ptr,
                           VTK_POINT_DATA, "phi_exact",   phi_exact_ptr,
                           VTK_POINT_DATA, "error",       error_ptr);

    ierr = VecRestoreArray(phi_initial, &phi_initial_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_reinit,  &phi_reinit_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_exact,   &phi_exact_ptr);   CHKERRXX(ierr);
    ierr = VecRestoreArray(error,       &error_ptr);       CHKERRXX(ierr);

    // ---------------------------------------------------------
    // destroy the structures and release allocated memory
    // ---------------------------------------------------------
    ierr = VecDestroy(phi_initial); CHKERRXX(ierr);
    ierr = VecDestroy(phi_reinit);  CHKERRXX(ierr);
    ierr = VecDestroy(phi_exact);  CHKERRXX(ierr);
    ierr = VecDestroy(error);    CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(conn, &brick);
  }


  w.stop(); w.read_duration();
}

