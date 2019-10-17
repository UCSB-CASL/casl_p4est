/*
 * Title: interpolation_nodes
 * Description:
 * Author: bochkov.ds@gmail.com
 * Date Created: 10-15-2019
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
#endif

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

const static std::string main_description =
    "In this example, we illustrate interpolation of node-sampled fields using\n"
    "my_p4est_interpolation_nodes_t class. Specifically, we perform interpolation at points located\n"
    "displace*(smallest diagonal) away from grid nodes in the direction defined by vector (nx, ny, nz).\n"
    "One can choose between performing interpolation on the fly (that is, treating the interpolation\n"
    "object as a CF_2/CF_3 object) or using buffering of the points."
    "Developer: Daniil Bochkov (dbochkov@ucsb.edu), October 2019.\n";

int    lmin = 4;
int    lmax = 4;
int    num_ghost_layers = 1;
double lip  = 1.5;

int num_splits = 6;

int test_function = 0; // 0 - sin(x)*cos(y)
int order         = 4; // 1 - linear, 2 - quadratic, 3 - quadratic non-oscillatory, 4 - quadratic non-oscillatory continuous (ver. 1), 5 - quadratic non-oscillatory continuous (ver. 2)

bool on_the_fly = true;
bool use_precomputed_derivatives = true;

double nx       = 1;
double ny       = 1;
double nz       = 1;
double displace = 0.1;

#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_function)
    {
      case 0: return sin(x)*cos(y)*exp(z);
    }
  }
} f_cf;
#else
struct: CF_2 {
  double operator()(double x, double y) const {
    switch (test_function)
    {
      case 0: return sin(x)*cos(y);
    }
  }
} f_cf;
#endif

#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
  }
} phi_cf;
#else
struct: CF_2 {
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

  // get command line parameters
  cmdParser cmd;
  cmd.add_option("lmin","");
  cmd.add_option("lmax","");
  cmd.add_option("num_ghost_layers","");
  cmd.add_option("lip","");
  cmd.add_option("num_splits","");
  cmd.add_option("test_function","");
  cmd.add_option("order","");
  cmd.add_option("on_the_fly","");
  cmd.add_option("use_precomputed_derivatives","");
  cmd.add_option("nx","");
  cmd.add_option("ny","");
  cmd.add_option("nz","");
  cmd.add_option("displace","");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  // read user's input parameters
  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  num_ghost_layers = cmd.get("num_ghost_layers", num_ghost_layers);
  lip = cmd.get("lip", lip);
  num_splits = cmd.get("num_splits", num_splits);
  test_function = cmd.get("test_function", test_function);
  order = cmd.get("order", order);
  on_the_fly = cmd.get("on_the_fly", on_the_fly);
  use_precomputed_derivatives = cmd.get("use_precomputed_derivatives", use_precomputed_derivatives);
  nx = cmd.get("nx", nx);
  ny = cmd.get("ny", ny);
  nz = cmd.get("nz", nz);
  displace = cmd.get("displace", displace);

  // make sure (nx, ny, nz) is a unit vector
#ifdef P4_TO_P8
  double norm = sqrt(SQR(nx) + SQR(ny) + SQR(nz));
  nx /= norm;
  ny /= norm;
  nz /= norm;
#else
  double norm = sqrt(SQR(nx) + SQR(ny));
  nx /= norm;
  ny /= norm;
#endif

  // stopwatch
  parStopWatch w;
  w.start("Running example: interpolation_nodes");

  // variables to store current and last max interpoolation errors
  double error_max_old = 0;
  double error_max_cur = 0;

  for (int iter = 0; iter < num_splits; ++iter)
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
    splitting_criteria_cf_t sp(lmin+iter, lmax+iter, &phi_cf, lip);
    p4est->user_pointer = &sp;
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition the forest
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    // create ghost layer
    for (int i = 0; i < num_ghost_layers; ++i)
    {
      if (i==0) ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      else      my_p4est_ghost_expand(p4est, ghost);
    }

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
    my_p4est_interpolation_nodes_t interp(&ngbd);

    // ---------------------------------------------------------
    // set input for interpolation
    // ---------------------------------------------------------
    if (use_precomputed_derivatives)
    {
#ifdef P4_TO_P8
      ngbd.second_derivatives_central(f, fxx, fyy, fzz);
      switch (order)
      {
        case 1: interp.set_input(f, linear); break;
        case 2: interp.set_input(f, fxx, fyy, fzz, quadratic); break;
        case 3: interp.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory); break;
        case 4: interp.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v1); break;
        case 5: interp.set_input(f, fxx, fyy, fzz, quadratic_non_oscillatory_continuous_v2); break;
        default: throw std::invalid_argument("Wrong order number");
      }
#else
      ngbd.second_derivatives_central(f, fxx, fyy);
      switch (order)
      {
        case 1: interp.set_input(f, linear); break;
        case 2: interp.set_input(f, fxx, fyy, quadratic); break;
        case 3: interp.set_input(f, fxx, fyy, quadratic_non_oscillatory); break;
        case 4: interp.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v1); break;
        case 5: interp.set_input(f, fxx, fyy, quadratic_non_oscillatory_continuous_v2); break;
        default: throw std::invalid_argument("Wrong order number");
      }
#endif
    }
    else
    {
      switch (order)
      {
        case 1: interp.set_input(f, linear); break;
        case 2: interp.set_input(f, quadratic); break;
        case 3: interp.set_input(f, quadratic_non_oscillatory); break;
        case 4: interp.set_input(f, quadratic_non_oscillatory_continuous_v1); break;
        case 5: interp.set_input(f, quadratic_non_oscillatory_continuous_v2); break;
        default: throw std::invalid_argument("Wrong order number");
      }
    }

    // ---------------------------------------------------------
    // perform interpolation
    // ---------------------------------------------------------
    double xyz[P4EST_DIM];
    if (on_the_fly)
    {
      // get access to f_interp's data
      ierr = VecGetArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);

      // loop through all local nodes
      foreach_local_node(n, nodes)
      {
        // get xyz coordinates of a node
        node_xyz_fr_n(n, p4est, nodes, xyz);

        // find coordinates of the point where interpolation is needed
        xyz[0] += displace*nx*diag_min; xyz[0] = MAX(xyz[0], xyz_min[0]); xyz[0] = MIN(xyz[0], xyz_max[0]);
        xyz[1] += displace*ny*diag_min; xyz[1] = MAX(xyz[1], xyz_min[1]); xyz[1] = MIN(xyz[1], xyz_max[1]);
#ifdef P4_TO_P8
        xyz[2] += displace*nz*diag_min; xyz[2] = MAX(xyz[2], xyz_min[2]); xyz[2] = MIN(xyz[2], xyz_max[2]);
#endif

        // interpolate
#ifdef P4_TO_P8
        f_interp_ptr[n] = interp(xyz[0], xyz[1], xyz[2]);
#else
        f_interp_ptr[n] = interp(xyz[0], xyz[1]);
#endif
      }

      // restore access to f_interp's data
      ierr = VecRestoreArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    }
    else
    {
      // loop through all local nodes
      foreach_local_node(n, nodes)
      {
        // get xyz coordinates of a node
        node_xyz_fr_n(n, p4est, nodes, xyz);

        // find coordinates of the point where interpolation is needed
        xyz[0] += displace*nx*diag_min; xyz[0] = MAX(xyz[0], xyz_min[0]); xyz[0] = MIN(xyz[0], xyz_max[0]);
        xyz[1] += displace*ny*diag_min; xyz[1] = MAX(xyz[1], xyz_min[1]); xyz[1] = MIN(xyz[1], xyz_max[1]);
#ifdef P4_TO_P8
        xyz[2] += displace*nz*diag_min; xyz[2] = MAX(xyz[2], xyz_min[2]); xyz[2] = MIN(xyz[2], xyz_max[2]);
#endif

        // add coordinates into interpolation buffer
        interp.add_point(n, xyz);
      }

      // interpolate field f at buffered points
      interp.interpolate(f_interp);
    }

    // ---------------------------------------------------------
    // compute exact values and interpolation error
    // ---------------------------------------------------------
    // get access to Vec's data
    ierr = VecGetArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecGetArray(error,    &error_ptr);    CHKERRXX(ierr);

    error_max_old = error_max_cur;
    error_max_cur = 0;

    // loop through local nodes
    foreach_local_node(n, nodes)
    {
      // get xyz coordinates of a node
      node_xyz_fr_n(n, p4est, nodes, xyz);

      // find coordinates of the point where interpolation was performed
      xyz[0] += displace*nx*diag_min; xyz[0] = MAX(xyz[0], xyz_min[0]); xyz[0] = MIN(xyz[0], xyz_max[0]);
      xyz[1] += displace*ny*diag_min; xyz[1] = MAX(xyz[1], xyz_min[1]); xyz[1] = MIN(xyz[1], xyz_max[1]);
#ifdef P4_TO_P8
      xyz[2] += displace*nz*diag_min; xyz[2] = MAX(xyz[2], xyz_min[2]); xyz[2] = MIN(xyz[2], xyz_max[2]);
#endif

      // get exact value and compute error
      f_exact_ptr[n] = f_cf.value(xyz);
      error_ptr[n] = fabs(f_exact_ptr[n]-f_interp_ptr[n]);

      // store max error
      error_max_cur = MAX(error_max_cur, error_ptr[n]);
    }

    // find max error among all processes
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_max_cur, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

    // rectore access to Vec's data
    ierr = VecRestoreArray(f_interp, &f_interp_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(f_exact,  &f_exact_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(error,    &error_ptr);    CHKERRXX(ierr);

    // print out max error
    ierr = PetscPrintf(mpi.comm(), "Grid levels: %2d / %2d, max interpolation error: %e, order: %1.3g\n", lmin+iter, lmax+iter, error_max_cur, log(error_max_old/error_max_cur)/log(2)); CHKERRXX(ierr);

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

