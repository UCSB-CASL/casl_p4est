/*
 * Title: interpolation_nodes_local
 * Description:
 * Author: bochkov.ds@gmail.com
 * Date Created: 10-17-2019
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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>

using namespace std;

const static std::string main_description =
    "In this example, we illustrate interpolation of node-sampled fields using\n"
    "optimized my_p4est_interpolation_nodes_local_t class. Specifically, we perform interpolation at \n"
    "the points of a local subgrid of size num_points^d spanning finite volume of a given grid node.\n"
    "Developer: Daniil Bochkov (dbochkov@ucsb.edu), October 2019.\n";

// ---------------------------------------------------------
// define parameters
// ---------------------------------------------------------
param_list_t pl;

// grid parameters
param_t<int>    lmin       (pl, 10,  "lmin",       "Min level of refinement (default: 4)");
param_t<int>    lmax       (pl, 10,  "lmin",       "Min level of refinement (default: 4)");
param_t<double> lip        (pl, 1.5, "lip",        "Lipschitz constant (default: 1.5)");
param_t<int>    num_splits (pl, 1,   "num_splits", "Number of successive refinements (default: 3)");

// problem set-up (points of iterpolation and function to interpolate)
param_t<int> num_points    (pl, 5, "num_points"   , "");
param_t<int> test_function (pl, 0, "test_function", "Test funciton to interpolate (default: 0):\n"
                                                    "    0 - sin(x)*cos(y)*exp(z)\n"
                                                    "    1 - ... (to be added)");

// method set-up
param_t<bool> local (pl, 1, "local", "Perform interpolation \"on-the-fly\" (1) or use buffering of points (0) (default: 1)");
param_t<int>  order (pl, 2, "order", "Order of interpolation (default: 1): \n"
                                     "    1 - bi-/tri-linear,\n"
                                     "    2 - bi-/tri-quadratic,\n"
                                     "    3 - bi-/tri-quadratic non-oscillatory,\n"
                                     "    4 - bi-/tri-quadratic non-oscillatory continuous (ver. 1),\n"
                                     "    5 - bi-/tri-quadratic non-oscillatory continuous (ver. 2)");

// test functions to interpolate
#ifdef P4_TO_P8
struct f_cf_t: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_function())
    {
      case 0: return sin(x)*cos(y)*exp(z);
      default: throw std::invalid_argument("Invalid test function number");
    }
  }
} f_cf;
#else
struct f_cf_t: CF_2 {
  double operator()(double x, double y) const {
    switch (test_function())
    {
      case 0: return sin(x)*cos(y);
      default: throw std::invalid_argument("Invalid test function number");
    }
  }
} f_cf;
#endif

// level-set function for refinement
#ifdef P4_TO_P8
struct phi_cf_t: CF_3 {
  double operator()(double x, double y, double z) const {
    return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
  }
} phi_cf;
#else
struct phi_cf_t : CF_2 {
  double operator()(double x, double y) const {
    return 0.5 - sqrt(SQR(x) + SQR(y));
  }
} phi_cf;
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
    splitting_criteria_cf_t sp(lmin()+iter, lmax()+iter, &phi_cf, lip());
    p4est->user_pointer = &sp;
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition the forest
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

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
    double dxyz[P4EST_DIM];    // dimensions of the smallest quadrants
    double dxyz_fv[P4EST_DIM]; // mesh size of the tiny local mesh
    double dxyz_min;           // minimum side length of the smallest quadrants
    double diag_min;           // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    dxyz_fv[0] = dxyz[0]/double(num_points()-1);
    dxyz_fv[1] = dxyz[1]/double(num_points()-1);
#ifdef P4_TO_P8
    dxyz_fv[2] = dxyz[2]/double(num_points()-1);
#endif

    Vec f;        // field to interpolate
    Vec fxx;      // its second derivative in x
    Vec fyy;      // its second derivative in y
#ifdef P4_TO_P8
    Vec fzz;      // its second derivative in z
#endif
    Vec f_interp; // interpolated values
    Vec f_exact;  // exact values
    Vec error;    // errors

    // double pointers to access Vec's data
    double *f_interp_ptr;
    double *f_exact_ptr;
    double *error_ptr;

    // ---------------------------------------------------------
    // allocate memory for Vec's
    // ---------------------------------------------------------
    ierr = VecCreateGhostNodes(p4est, nodes, &f);   CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &fxx); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &fyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &fzz); CHKERRXX(ierr);
#endif
    ierr = VecDuplicate(f, &f_interp); CHKERRXX(ierr);
    ierr = VecDuplicate(f, &f_exact);  CHKERRXX(ierr);
    ierr = VecDuplicate(f, &error);    CHKERRXX(ierr);

    // ---------------------------------------------------------
    // sample function at grid points
    // ---------------------------------------------------------
    sample_cf_on_nodes(p4est, nodes, f_cf, f);

    // ---------------------------------------------------------
    // create interpolation function
    // ---------------------------------------------------------
    my_p4est_interpolation_nodes_t       interp_g(&ngbd);
    my_p4est_interpolation_nodes_local_t interp_l(&ngbd);

    // ---------------------------------------------------------
    // set input for interpolation
    // ---------------------------------------------------------
#ifdef P4_TO_P8
    ngbd.second_derivatives_central(f, fxx, fyy, fzz);
    switch (order())
    {
      case 1:
        interp_g.set_input(f, linear);
        interp_l.set_input(f, linear); break;
      case 2:
        interp_g.set_input(f, fxx, fyy, fzz, quadratic);
        interp_l.set_input(f, fxx, fyy, fzz, quadratic); break;
      case 3:
        interp_g.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory);
        interp_l.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory); break;
      case 4:
        interp_g.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v1);
        interp_l.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v1); break;
      case 5:
        interp_g.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v2);
        interp_l.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v2); break;
      default:
        throw std::invalid_argument("Wrong order number");
    }
#else
    ngbd.second_derivatives_central(f, fxx, fyy);
    switch (order())
    {
      case 1:
        interp_g.set_input(f, linear);
        interp_l.set_input(f, linear); break;
      case 2:
        interp_g.set_input(f, fxx, fyy, quadratic);
        interp_l.set_input(f, fxx, fyy, quadratic); break;
      case 3:
        interp_g.set_input(f, fxx, fyy, quadratic_non_oscillatory);
        interp_l.set_input(f, fxx, fyy, quadratic_non_oscillatory); break;
      case 4:
        interp_g.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v1);
        interp_l.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v1); break;
      case 5:
        interp_g.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v2);
        interp_l.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v2); break;
      default:
        throw std::invalid_argument("Wrong order number");
    }
#endif

    // ---------------------------------------------------------
    // perform interpolation
    // ---------------------------------------------------------
    double xyz[P4EST_DIM];
    double xyz_fv[P4EST_DIM];

    // get access to Vec's data
    ierr = VecGetArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecGetArray(error,    &error_ptr);    CHKERRXX(ierr);

    parStopWatch timing;
    timing.start("Timing interpolation only");

    if (local())
    {
      // loop through all local nodes
      foreach_local_node(n, nodes)
      {
        interp_l.initialize(n);

        // get xyz coordinates of a node
        node_xyz_fr_n(n, p4est, nodes, xyz);

        xyz_fv[0] -= .5*dxyz[0];
        xyz_fv[1] -= .5*dxyz[1];
#ifdef P4_TO_P8
        xyz_fv[2] -= .5*dxyz[2];
#endif

        double max_error_local = 0;

#ifdef P4_TO_P8
        for (int k = 0; k < num_points(); ++k)
#endif
          for (int j = 0; j < num_points(); ++j)
            for (int i = 0; i < num_points(); ++i)
            {
              xyz_fv[0] = xyz[0]+double(i)*dxyz_fv[0]; xyz_fv[0] = MAX(xyz_fv[0], xyz_min[0]); xyz_fv[0] = MIN(xyz_fv[0], xyz_max[0]);
              xyz_fv[1] = xyz[1]+double(j)*dxyz_fv[1]; xyz_fv[1] = MAX(xyz_fv[1], xyz_min[1]); xyz_fv[1] = MIN(xyz_fv[1], xyz_max[1]);
#ifdef P4_TO_P8
              xyz_fv[2] = xyz[2]+double(k)*dxyz_fv[2]; xyz_fv[2] = MAX(xyz_fv[2], xyz_min[2]); xyz_fv[2] = MIN(xyz_fv[2], xyz_max[2]);
#endif

            // interpolate
#ifdef P4_TO_P8
            f_interp_ptr[n] = interp_l(xyz[0], xyz[1], xyz[2]);
            f_exact_ptr [n] = f_cf(xyz[0], xyz[1], xyz[2]);
#else
            f_interp_ptr[n] = interp_l(xyz[0], xyz[1]);
            f_exact_ptr [n] = f_cf(xyz[0], xyz[1]);
#endif
            max_error_local = MAX(max_error_local, fabs(f_interp_ptr[n]-f_exact_ptr [n]));
          }

        error_ptr[n] = max_error_local;
        error_max_cur = MAX(error_max_cur, max_error_local);
      }
    }
    else
    {
      // loop through all local nodes
      foreach_local_node(n, nodes)
      {
        // get xyz coordinates of a node
        node_xyz_fr_n(n, p4est, nodes, xyz);

        xyz_fv[0] -= .5*dxyz[0];
        xyz_fv[1] -= .5*dxyz[1];
#ifdef P4_TO_P8
        xyz_fv[2] -= .5*dxyz[2];
#endif

        double max_error_local = 0;

#ifdef P4_TO_P8
        for (int k = 0; k < num_points(); ++k)
#endif
          for (int j = 0; j < num_points(); ++j)
            for (int i = 0; i < num_points(); ++i)
            {
              xyz_fv[0] = xyz[0]+double(i)*dxyz_fv[0]; xyz_fv[0] = MAX(xyz_fv[0], xyz_min[0]); xyz_fv[0] = MIN(xyz_fv[0], xyz_max[0]);
              xyz_fv[1] = xyz[1]+double(j)*dxyz_fv[1]; xyz_fv[1] = MAX(xyz_fv[1], xyz_min[1]); xyz_fv[1] = MIN(xyz_fv[1], xyz_max[1]);
#ifdef P4_TO_P8
              xyz_fv[2] = xyz[2]+double(k)*dxyz_fv[2]; xyz_fv[2] = MAX(xyz_fv[2], xyz_min[2]); xyz_fv[2] = MIN(xyz_fv[2], xyz_max[2]);
#endif

            // interpolate
#ifdef P4_TO_P8
            f_interp_ptr[n] = interp_g(xyz[0], xyz[1], xyz[2]);
            f_exact_ptr [n] = f_cf(xyz[0], xyz[1], xyz[2]);
#else
            f_interp_ptr[n] = interp_g(xyz[0], xyz[1]);
            f_exact_ptr [n] = f_cf(xyz[0], xyz[1]);
#endif
            max_error_local = MAX(max_error_local, fabs(f_interp_ptr[n]-f_exact_ptr [n]));
          }

        error_ptr[n] = max_error_local;
        error_max_cur = MAX(error_max_cur, max_error_local);
      }
    }

    timing.stop(); timing.read_duration();

    // find max error among all processes
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_max_cur, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

    // rectore access to Vec's data
    ierr = VecRestoreArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(error,    &error_ptr);    CHKERRXX(ierr);

    // print out max error
    ierr = PetscPrintf(mpi.comm(), "Grid levels: %2d / %2d, max interpolation error: %e, order: %1.3g\n", lmin()+iter, lmax()+iter, error_max_cur, log(error_max_old/error_max_cur)/log(2)); CHKERRXX(ierr);

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
        << "/interpolation_nodes"
       #ifdef P4_TO_P8
        << "_3d"
       #else
        << "_2d"
       #endif
        << "_nprocs_" << p4est->mpisize
        << "_split_" << iter;

    ierr = VecGetArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecGetArray(error,    &error_ptr);    CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "interp", f_interp_ptr,
                           VTK_POINT_DATA, "exact",  f_exact_ptr,
                           VTK_POINT_DATA, "error",  error_ptr);

    ierr = VecRestoreArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(error,    &error_ptr);    CHKERRXX(ierr);

    // ---------------------------------------------------------
    // destroy the structures and release allocated memory
    // ---------------------------------------------------------
    ierr = VecDestroy(f);   CHKERRXX(ierr);
    ierr = VecDestroy(fxx); CHKERRXX(ierr);
    ierr = VecDestroy(fyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(fzz); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(f_interp); CHKERRXX(ierr);
    ierr = VecDestroy(f_exact);  CHKERRXX(ierr);
    ierr = VecDestroy(error);    CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(conn, &brick);
  }


  w.stop(); w.read_duration();
}

