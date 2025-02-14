/*
 * Title: integration_fast
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
#include <iomanip>

using namespace std;

const static std::string main_description =
    "In this example, we illustrate fast integration over separate parts of irregular geometries"
    "Developer: Daniil Bochkov (dbochkov@ucsb.edu), December 2019.\n";


// ---------------------------------------------------------
// define parameters
// ---------------------------------------------------------
param_list_t pl;

// grid parameters
param_t<int>    lmin                (pl,-1,   "lmin",         "Min level of refinement (default: 4)");
param_t<int>    lmax                (pl, 5,   "lmax",         "Max level of refinement (default: 4)");
param_t<double> lip                 (pl, 0.5, "lip",          "Lipschitz constant (characterize transition width between coarse and fine regions) (default: 1.2)");
param_t<double> uniform_band        (pl, 5,   "uniform_band", "Width of the uniform band around interface (in smallest quadrant lengths) (default: 5)");
param_t<int>    num_splits          (pl, 10,   "num_splits",   "Number of successive refinements (default: 5)");
param_t<int>    num_splits_per_split(pl, 1,   "num_splits_per_split",   "(default: 3)");


// problem set-up (points of iterpolation and function to interpolate)
param_t<int> geometry(pl, 1, "test_domain", "Test domain (default: 0):\n"
                                               "    0 - one sphere \n"
                                               "    1 - three spheres \n");

// method set-up
param_t<bool>   reinit_level_set(pl, 0,  "reinit_level_set", "Reinitialize level-set function before extension (helps to regularize normals in the presence of kinks) (default: 0)");

// output settings
param_t<bool>   save_vtk        (pl, 1,  "save_vtk",         "Save VTK into OUT_DIR/vtu (default: 1)");
param_t<bool>   save_convergence(pl, 1,  "save_convergence", "Save convergence data into OUT_DIR/convergence (default: 1)");


int num_spheres;
std::vector<double> r;
std::vector<double> xc;
std::vector<double> yc;
std::vector<double> zc;

void set_geometry()
{
  switch (geometry.val)
  {
    case 0:
      num_spheres = 1;
      r.push_back(0.54321); xc.push_back(0.00); yc.push_back(0.00); zc.push_back(0.00);
      break;
    case 1:
      num_spheres = 3;
      r.push_back(0.123); xc.push_back(0.39); yc.push_back(0.55); zc.push_back(0.44);
      r.push_back(0.231); xc.push_back(-.21); yc.push_back(0.34); zc.push_back(0.12);
      r.push_back(0.312); xc.push_back(0.13); yc.push_back(-.48); zc.push_back(-.33);
      break;
    default:
      throw;
  }
}

// level-set function
struct: CF_DIM {
  double operator()(DIM(double x, double y, double z)) const
  {
    int    idx_min = -1;
    double phi_min = DBL_MAX;

    for (int i = 0; i < r.size(); ++i)
    {
      double phi = ABSD(x-xc[i], y-yc[i], z-zc[i]) - r[i];

      if (phi_min > phi)
      {
        phi_min = phi;
        idx_min = i;
      }
    }

    if (idx_min == -1) throw;

    return phi_min;
  }
} phi_cf;

struct: CF_DIM {
  double operator()(DIM(double x, double y, double z)) const
  {
    int    idx_min = -1;
    double phi_min = DBL_MAX;

    for (int i = 0; i < r.size(); ++i)
    {
      double phi = ABSD(x-xc[i], y-yc[i], z-zc[i]) - r[i];

      if (phi_min > phi)
      {
        phi_min = phi;
        idx_min = i;
      }
    }

    if (idx_min == -1) throw;

    return idx_min;
  }
} idx_cf;

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

  set_geometry();

  // stopwatch
  parStopWatch w;
  w.start("Running example: extension");

  // arrays to store convergence data
  std::vector<double> lmin_all;
  std::vector<double> lmax_all;
  std::vector<double> hmin_all;
  std::vector<double> error_area_all;
  std::vector<double> error_volume_all;

  // domain size information
  const double xyz_min [] = {-1, -1, -1};
  const double xyz_max [] = { 1,  1,  1};
  const int    n_xyz   [] = { 1,  1,  1};
  const int    periodic[] = { 0,  0,  0};

  // ---------------------------------------------------------
  // loop through all grid resolutions
  // ---------------------------------------------------------
  int iteration_gl = 0;
  for (int iter = 0; iter < num_splits(); ++iter)
  {
    int num_sub_iter = (iter == 0 ? 1 : num_splits_per_split());
    for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
    {
      double xyz_min_alt[3];
      double xyz_max_alt[3];

      double scale = (double) (num_sub_iter-1-sub_iter) / (double) num_sub_iter;
      xyz_min_alt[0] = xyz_min[0] - .5*(pow(2.,scale)-1)*(xyz_max[0]-xyz_min[0]); xyz_max_alt[0] = xyz_max[0] + .5*(pow(2.,scale)-1)*(xyz_max[0]-xyz_min[0]);
      xyz_min_alt[1] = xyz_min[1] - .5*(pow(2.,scale)-1)*(xyz_max[1]-xyz_min[1]); xyz_max_alt[1] = xyz_max[1] + .5*(pow(2.,scale)-1)*(xyz_max[1]-xyz_min[1]);
      xyz_min_alt[2] = xyz_min[2] - .5*(pow(2.,scale)-1)*(xyz_max[2]-xyz_min[2]); xyz_max_alt[2] = xyz_max[2] + .5*(pow(2.,scale)-1)*(xyz_max[2]-xyz_min[2]);

      // ---------------------------------------------------------
      // create grid
      // ---------------------------------------------------------
      // p4est variables
      p4est_t*              p4est;
      p4est_nodes_t*        nodes;
      p4est_ghost_t*        ghost;
      p4est_connectivity_t* conn;
      my_p4est_brick_t      brick;

      conn = my_p4est_brick_new(n_xyz, xyz_min_alt, xyz_max_alt, &brick, periodic);

      // create the forest
      p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

      // refine based on distance to a level-set
      splitting_criteria_cf_t sp(lmin() == -1 ? 1 : lmin()+iter, lmax()+iter, &phi_cf, lip());
      p4est->user_pointer = &sp;
      for (int i = 0; i < lmax()+iter; ++i)
      {
        my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
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
      Vec idx;
      Vec ones;

      // double pointers to access Vec's data
      double *phi_ptr;
      double *idx_ptr;

      // ---------------------------------------------------------
      // allocate memory for Vec's
      // ---------------------------------------------------------
      ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
      ierr = VecDuplicate(phi, &idx);                 CHKERRXX(ierr);
      ierr = VecDuplicate(phi, &ones);                CHKERRXX(ierr);

      // ---------------------------------------------------------
      // sample level-set function and test function at grid points
      // ---------------------------------------------------------
      sample_cf_on_nodes(p4est, nodes, phi_cf, phi);
      sample_cf_on_nodes(p4est, nodes, idx_cf, idx);
      ierr = VecSetGhost(ones, 1.0); CHKERRXX(ierr);

      if (reinit_level_set())
      {
        my_p4est_level_set_t ls(&ngbd);
        ls.reinitialize_2nd_order(phi);
      }

      // ---------------------------------------------------------
      // compute integrals
      // ---------------------------------------------------------
      std::vector<double> volumes;
      std::vector<double> areas;

      integrate_over_negative_domain(num_spheres, volumes, p4est, nodes, phi, idx, ones);
      integrate_over_interface      (num_spheres, areas,   p4est, nodes, phi, idx, ones);

      // ---------------------------------------------------------
      // compute errors
      // ---------------------------------------------------------
      lmin_all        .push_back(sp.min_lvl);
      lmax_all        .push_back(sp.max_lvl);
      hmin_all        .push_back(dxyz_min);
      error_area_all  .push_back(0);
      error_volume_all.push_back(0);

      for (int i = 0; i < r.size(); ++i)
      {
#ifdef P4_TO_P8
        double area   = 4.*PI*SQR(r[i]);
        double volume = 4.*PI*pow(r[i],3.)/3.;
#else
        double area   = 2.*PI*r[i];
        double volume = PI*SQR(r[i]);
#endif
        error_area_all  .back() += fabs(areas  [i] - area);
        error_volume_all.back() += fabs(volumes[i] - volume);
      }

      // print out max error
      double log_error_area_ratio   = hmin_all.size() <= 1 ? 0 : log(error_area_all[hmin_all.size()-2]/error_area_all[hmin_all.size()-1]);
      double log_error_volume_ratio = hmin_all.size() <= 1 ? 0 : log(error_volume_all[hmin_all.size()-2]/error_volume_all[hmin_all.size()-1]);
      double logdx1dx2_inv = hmin_all.size() <= 1 ? 0 : 1./log(hmin_all[hmin_all.size()-2]/hmin_all[hmin_all.size()-1]);
      ierr = PetscPrintf(mpi.comm(), "Grid levels: %5.2f / %5.2f, total volume error: %1.2e / %+1.2f, total surface error: %1.2e / %+1.2f", lmin() == -1 ? 0 : lmin()+iter-scale, lmax()+iter-scale,
                         error_volume_all.back(), log_error_volume_ratio*logdx1dx2_inv,
                         error_area_all  .back(), log_error_area_ratio*logdx1dx2_inv); CHKERRXX(ierr);


      // ---------------------------------------------------------
      // save the grid and data into vtk
      // ---------------------------------------------------------
      if (save_vtk())
      {
        const char *out_dir = getenv("OUT_DIR");
        if (!out_dir) out_dir = ".";
        if (mpi.rank() == 0)
        {
          std::ostringstream command;
          command << "mkdir -p " << out_dir << "/vtu";
          int ret_sys = system(command.str().c_str());
          if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
        }

        std::ostringstream oss;
        oss << out_dir
            << "/vtu/integration_fast"
       #ifdef P4_TO_P8
            << "_3d"
       #else
            << "_2d"
       #endif
            << "_nprocs_" << p4est->mpisize
            << "_split_" << std::setfill('0') << std::setw(4)<< iteration_gl;

        ierr = VecGetArray(phi, &phi_ptr);
        ierr = VecGetArray(idx, &idx_ptr);

        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               2, 0, oss.str().c_str(),
                               VTK_POINT_DATA, "phi", phi_ptr,
                               VTK_POINT_DATA, "idx", idx_ptr);

        ierr = VecRestoreArray(phi, &phi_ptr);
        ierr = VecRestoreArray(idx, &idx_ptr);

        ierr = PetscPrintf(mpi.comm(), "VTK saved in %s", oss.str().c_str());
      }

      ierr = PetscPrintf(mpi.comm(), "\n");


      // ---------------------------------------------------------
      // destroy the structures and release allocated memory
      // ---------------------------------------------------------
      ierr = VecDestroy(ones);
      ierr = VecDestroy(idx);
      ierr = VecDestroy(phi);

      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_destroy      (p4est);
      my_p4est_brick_destroy(conn, &brick);

      iteration_gl++;
    }
  }

  // ---------------------------------------------------------
  // save convergence
  // ---------------------------------------------------------
  if (save_convergence() && mpi.rank() == 0)
  {
    const char *out_dir = getenv("OUT_DIR");
    if (!out_dir) out_dir = ".";
    if (mpi.rank() == 0)
    {
      std::ostringstream command;
      command << "mkdir -p " << out_dir << "/convergence";
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0) throw std::invalid_argument("could not create directory");
    }

    std::string filename;

    filename = out_dir; filename += "/convergence/lmin.txt";            save_vector(filename.c_str(), lmin_all);
    filename = out_dir; filename += "/convergence/lmax.txt";            save_vector(filename.c_str(), lmax_all);
    filename = out_dir; filename += "/convergence/hmin.txt";            save_vector(filename.c_str(), hmin_all);
    filename = out_dir; filename += "/convergence/error_area.txt";      save_vector(filename.c_str(), error_area_all);
    filename = out_dir; filename += "/convergence/error_volume.txt";    save_vector(filename.c_str(), error_volume_all);

    filename = out_dir; filename += "/convergence";
    ierr = PetscPrintf(mpi.comm(), "Convergence data saved in %s\n", filename.c_str());
  }



  w.stop(); w.read_duration();
}
