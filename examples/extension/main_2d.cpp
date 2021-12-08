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
#include <src/my_p4est_grid_aligned_extension.h>
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
//#include <src/my_p8est_grid_aligned_extension.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>
#include <iomanip>

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
param_t<int>    lmin                 (pl, 4,   "lmin",                 "Min level of refinement (can be negative -> will stay same for all refinements) (default: 4)");
param_t<int>    lmax                 (pl, 4,   "lmax",                 "Max level of refinement (can be negative -> will stay same for all refinements) (default: 4)");
param_t<double> lip                  (pl, 1.2, "lip",                  "Lipschitz constant (characterize transition width between coarse and fine regions) (default: 1.2)");
param_t<double> uniform_band         (pl, 5,   "uniform_band",         "Width of the uniform band around interface (in smallest quadrant lengths) (default: 5)");
param_t<int>    num_splits           (pl, 5,   "num_splits",           "Number of successive refinements (default: 5)");
param_t<int>    num_splits_per_split (pl, 1,   "num_splits_per_split", "Number of additional refinements (default: 1)");
param_t<bool>   aggressive_coarsening(pl, 1,   "aggressive_corsening", "Enfornce lip = 0 (i.e. no smooth transition from uniform band to coarse grid");


// problem set-up (points of iterpolation and function to interpolate)
param_t<int> test_domain(pl, 0, "test_domain", "Test domain (default: 0):\n"
                                               "    0 - sphere \n"
                                               "    1 - flower shaped"
                                               "    2 - union of two spheres"
                                               "    3 - difference of two spheres");

param_t<int> test_function(pl, 0, "test_function", "Test function (default: 0):\n"
                                                   "    0 - sin(pi*x)*cos(pi*y)\n"
                                                   "    1 - ... (more to be added)");

// method set-up
param_t<bool>   reinit_level_set(pl, 1,  "reinit_level_set", "Reinitialize level-set function before extension (helps to regularize normals in the presence of kinks) (default: 0)");
param_t<bool>   rerefine        (pl, 1,  "rerefine"        , "Refine according to signed distance (default: 0)");
param_t<bool>   show_convergence(pl, 0,  "show_convergence", "Show convergence as iterations performed (default: 0)");
param_t<bool>   use_full        (pl, 0,  "use_full"        , "Extend only normal derivatives (0) or all derivatives in Cartesian directions (1) (default: 0)");
param_t<int>    num_iterations  (pl, 51, "num_iterations"  , "Number of iterations (default: 50)\n");
param_t<double> band            (pl, 2,  "band"            , "Band around interface (in diagonal lengths) to check for accuracy (default: 2)\n");

// output settings
param_t<bool>   save_vtk        (pl, 1,  "save_vtk",         "Save VTK into OUT_DIR/vtu (default: 1)");
param_t<bool>   save_convergence(pl, 1,  "save_convergence", "Save convergence data into OUT_DIR/convergence (default: 1)");


// level-set function
#ifdef P4_TO_P8
struct: CF_3 {
  double operator()(double x, double y, double z) const {
    switch (test_domain.val)
    {
      case 0: return -(0.501 - sqrt(SQR(x) + SQR(y) + SQR(z)));
      case 1:
        static flower_shaped_domain_t flower(0.501, .0, .0, .0, .25, 1);
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
        static flower_shaped_domain_t flower(0.501, .0, .0, .25, 1);
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
      case 0: return sin(PI*x)*cos(PI*y)*exp(z);
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
      case 0: return sin(2.*PI*x)*cos(2.*PI*y);
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
  double error_const_max_old = 1;
  double error_const_max_cur = 1;

  double error_linear_max_old = 1;
  double error_linear_max_cur = 1;

  double error_quadratic_max_old = 1;
  double error_quadratic_max_cur = 1;

  // arrays to store convergence data
  std::vector<double> lmin_all;
  std::vector<double> lmax_all;
  std::vector<double> hmin_all;
  std::vector<double> error_const_all;
  std::vector<double> error_linear_all;
  std::vector<double> error_quadratic_all;

  // domain size information
  const double xyz_min[]  = {-1, -1, -1};
  const double xyz_max[]  = { 1,  1,  1};
  const int    n_xyz[]    = { 1,  1,  1};
  const int    periodic[] = { 0,  0,  0};

  // ---------------------------------------------------------
  // loop through all grid resolutions
  // ---------------------------------------------------------
  int iteration_gl = 0;
  for (int iter = 0; iter < num_splits(); ++iter)
  {
    // we get additional refinements (which correspond to fractional lmin and lmax) by increasing the computational domain size by two
    // and then shriking it back to its normal size
    int num_sub_iter = (iter == 0 ? 1 : num_splits_per_split());
    for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
    {
      // compute the modified size of the computational domain
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
      splitting_criteria_cf_t sp(lmin() < 0 ? ABS(lmin()) : lmin()+iter, lmax() < 0 ? ABS(lmax()) : lmax()+iter, &phi_cf, lip(), uniform_band());
      p4est->user_pointer = &sp;
      for (int i = 0; i < sp.max_lvl; ++i)
      {
        my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
        my_p4est_partition(p4est, P4EST_TRUE, NULL);
      }

      if (aggressive_coarsening())
      {
        sp.lip = 0;
        my_p4est_coarsen(p4est, P4EST_TRUE, coarsen_levelset_cf, NULL);
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

      // reinitialize and refine-coarsen again
      if (rerefine())
      {
        Vec phi_tmp;
        ierr = VecCreateGhostNodes(p4est, nodes, &phi_tmp); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, phi_cf, phi_tmp);

        my_p4est_level_set_t ls(&ngbd);
        ls.reinitialize_1st_order(phi_tmp, 100);

        p4est_t       *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
        p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        Vec phi_tmp_np1;
        ierr = VecDuplicate(phi_tmp, &phi_tmp_np1); CHKERRXX(ierr);
        ierr = VecCopyGhost(phi_tmp, phi_tmp_np1); CHKERRXX(ierr);

        bool is_grid_changing = true;
        while (is_grid_changing)
        {
          double *phi_tmp_ptr;
          ierr = VecGetArray(phi_tmp_np1, &phi_tmp_ptr); CHKERRXX(ierr);

          splitting_criteria_tag_t sp_new(sp.min_lvl, sp.max_lvl, 0*sp.lip, sp.band);
          is_grid_changing = sp_new.refine_and_coarsen(p4est_np1, nodes_np1, phi_tmp_ptr);

          ierr = VecRestoreArray(phi_tmp_np1, &phi_tmp_ptr); CHKERRXX(ierr);

          if (is_grid_changing)
          {
            // repartition p4est
            my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

            // reset nodes, ghost, and phi
            p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
            p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

            // interpolate data between grids
            my_p4est_interpolation_nodes_t interp(&ngbd);

            double xyz[P4EST_DIM];
            foreach_node(n, nodes_np1)
            {
              node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
              interp.add_point(n, xyz);
            }

            ierr = VecDestroy(phi_tmp_np1); CHKERRXX(ierr);
            ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_tmp_np1); CHKERRXX(ierr);

            interp.set_input(phi_tmp, linear);
            interp.interpolate(phi_tmp_np1);
          }
        }

        ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
        ierr = VecDestroy(phi_tmp_np1); CHKERRXX(ierr);

        // delete old p4est
        p4est_destroy(p4est);       p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;
        hierarchy.update(p4est, ghost);
        ngbd.update(&hierarchy, nodes);
      }

      // ---------------------------------------------------------
      // define variables
      // ---------------------------------------------------------
      double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
      double dxyz_min;        // minimum side length of the smallest quadrants
      double diag_min;        // diagonal length of the smallest quadrants
      get_dxyz_min(p4est, dxyz, &dxyz_min, &diag_min);

      Vec phi;
      Vec phi_reinit;
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
      double *phi_reinit_ptr;
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

      ierr = VecDuplicate(phi, &phi_reinit);  CHKERRXX(ierr);
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
      VecCopyGhost(phi, phi_reinit);
      sample_cf_on_nodes(p4est, nodes, f_cf, f_exact);
      VecCopyGhost(f_exact, f_const);
      VecCopyGhost(f_exact, f_linear);
      VecCopyGhost(f_exact, f_quadratic);


      // reset fields for phi > 0
      ierr = VecGetArray(phi,         &phi_ptr);
      ierr = VecGetArray(f_const,     &f_const_ptr);
      ierr = VecGetArray(f_linear,    &f_linear_ptr);
      ierr = VecGetArray(f_quadratic, &f_quadratic_ptr);

      foreach_node(n, nodes)
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
      ls.reinitialize_1st_order(phi_reinit);

      if (reinit_level_set()) ls.reinitialize_2nd_order(phi);

//      Vec phi_smoothed;

//      ierr = VecDuplicate( phi, &phi_smoothed ); CHKERRXX( ierr );
//      ierr = VecCopyGhost( phi, phi_smoothed ); CHKERRXX( ierr );

//      ierr = VecShiftGhost( phi_smoothed, 2.0*diag_min ); CHKERRXX( ierr );

//      ls.reinitialize_1st_order( phi_smoothed );
//      ierr = VecShiftGhost( phi_smoothed, -2.0*diag_min ); CHKERRXX( ierr );

//      Vec normal_smoothed[P4EST_DIM];

//      foreach_dimension(dim) {
//        ierr = VecCreateGhostNodes(p4est, nodes, &normal_smoothed[dim]); CHKERRXX(ierr);
//      }

//      compute_normals(ngbd, phi_smoothed, normal_smoothed);

      parStopWatch timer;
      double time_const;
      double time_linear;
      double time_quadratic;
      if (use_full())
      {
        w.start(); ls.extend_Over_Interface_TVD_Full(phi, f_const,     num_iterations(), 0); w.stop(); time_const     = w.get_duration_current(); // constant extrapolation
        w.start(); ls.extend_Over_Interface_TVD_Full(phi, f_linear,    num_iterations(), 1); w.stop(); time_linear    = w.get_duration_current(); // linear extrapolation
        w.start(); ls.extend_Over_Interface_TVD_Full(phi, f_quadratic, num_iterations(), 2); w.stop(); time_quadratic = w.get_duration_current(); // quadratic extrapolationn
      }
      else
      {
        w.start(); ls.extend_Over_Interface_TVD     (phi, f_const,     num_iterations(), 0); w.stop(); time_const     = w.get_duration_current(); // constant extrapolation
        w.start(); ls.extend_Over_Interface_TVD     (phi, f_linear,    num_iterations(), 1); w.stop(); time_linear    = w.get_duration_current(); // linear extrapolation
        w.start(); ls.extend_Over_Interface_TVD     (phi, f_quadratic, num_iterations(), 2); w.stop(); time_quadratic = w.get_duration_current(); // quadratic extrapolationn

//        my_p4est_grid_aligned_extension_t ext_c(&ngbd);
//        my_p4est_grid_aligned_extension_t ext_l(&ngbd);
//        my_p4est_grid_aligned_extension_t ext_q(&ngbd);
//        w.start(); ext_c.initialize(phi, 0, true, num_iterations(), band()+1, band(), normal_smoothed, NULL); ext_c.extend(1, &f_const);     time_const     = w.get_duration_current();
//        w.start(); ext_l.initialize(phi, 1, true, num_iterations(), band()+1, band(), normal_smoothed, NULL); ext_l.extend(1, &f_linear);    time_linear    = w.get_duration_current();
//        w.start(); ext_q.initialize(phi, 3, true, num_iterations(), band()+1, band(), normal_smoothed, NULL); ext_q.extend(1, &f_quadratic); time_quadratic = w.get_duration_current();
      }

      ls.extend_from_interface_to_whole_domain_TVD(phi, f_exact, f_flat, num_iterations()); // constant extrapolation in both directions (flattening)

      // ---------------------------------------------------------
      // compute extension errors
      // ---------------------------------------------------------
      // get access to Vec's data
      ierr = VecGetArray(phi_reinit,      &phi_ptr);
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
      foreach_node(n, nodes)
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
      ierr = VecRestoreArray(phi_reinit,      &phi_ptr);
      ierr = VecRestoreArray(f_exact,         &f_exact_ptr);
      ierr = VecRestoreArray(f_const,         &f_const_ptr);
      ierr = VecRestoreArray(f_linear,        &f_linear_ptr);
      ierr = VecRestoreArray(f_quadratic,     &f_quadratic_ptr);
      ierr = VecRestoreArray(error_const,     &error_const_ptr);
      ierr = VecRestoreArray(error_linear,    &error_linear_ptr);
      ierr = VecRestoreArray(error_quadratic, &error_quadratic_ptr);

      // save results
      lmin_all.push_back(sp.min_lvl-scale);
      lmax_all.push_back(sp.max_lvl-scale);
      hmin_all.push_back(dxyz_min);
      error_const_all.push_back(error_const_max_cur);
      error_linear_all.push_back(error_linear_max_cur);
      error_quadratic_all.push_back(error_quadratic_max_cur);

      // print out max error
      double logdx1dx2_inv = hmin_all.size() <= 1 ? 0 : 1./log(hmin_all[hmin_all.size()-2]/hmin_all[hmin_all.size()-1]);
      ierr = PetscPrintf(mpi.comm(), "Grid levels: %5.2f / %5.2f, const (%1.2e s): %1.2e / %+1.2f, linear (%1.2e s): %1.2e / %+1.2f, quadratic (%1.2e s): %1.2e / %+1.2f. ", sp.min_lvl-scale, sp.max_lvl-scale,
                         time_const,     error_const_max_cur,     log(error_const_max_old/error_const_max_cur)*logdx1dx2_inv,
                         time_linear,    error_linear_max_cur,    log(error_linear_max_old/error_linear_max_cur)*logdx1dx2_inv,
                         time_quadratic, error_quadratic_max_cur, log(error_quadratic_max_old/error_quadratic_max_cur)*logdx1dx2_inv); CHKERRXX(ierr);


      // ---------------------------------------------------------
      // save the grid and data into vtk
      // ---------------------------------------------------------
      if (save_vtk())
      {
        const char *out_dir = getenv("OUT_DIR");
        if (!out_dir) out_dir = ".";
        else if (mpi.rank() == 0)
        {
          std::ostringstream command;
          command << "mkdir -p " << out_dir << "/vtu";
          int ret_sys = system(command.str().c_str());
          if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
        }

        std::ostringstream oss;
        oss << out_dir
            << "/vtu/extend"
       #ifdef P4_TO_P8
            << "_3d"
       #else
            << "_2d"
       #endif
            << "_nprocs_" << p4est->mpisize
            << "_split_" << std::setfill('0') << std::setw(3)<< iteration_gl;

        /* save the size of the leaves */
        Vec leaf_level;
        ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
        double *l_p;
        ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

        for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
        {
          p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
          for( size_t q=0; q<tree->quadrants.elem_count; ++q)
          {
            const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            l_p[tree->quadrants_offset+q] = quad->level;
          }
        }

        for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
          l_p[p4est->local_num_quadrants+q] = quad->level;
        }

        ierr = VecGetArray(phi,             &phi_ptr);
//        ierr = VecGetArray(phi_smoothed,    &phi_ptr);
        ierr = VecGetArray(phi_reinit,      &phi_reinit_ptr);
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
                               10, 1, oss.str().c_str(),
                               VTK_POINT_DATA, "phi",             phi_ptr,
                               VTK_POINT_DATA, "phi_reinit",      phi_reinit_ptr,
                               VTK_POINT_DATA, "f_exact",         f_exact_ptr,
                               VTK_POINT_DATA, "f_const",         f_const_ptr,
                               VTK_POINT_DATA, "f_linear",        f_linear_ptr,
                               VTK_POINT_DATA, "f_quaratic",      f_quadratic_ptr,
                               VTK_POINT_DATA, "f_flat",          f_flat_ptr,
                               VTK_POINT_DATA, "error_const",     error_const_ptr,
                               VTK_POINT_DATA, "error_linear",    error_linear_ptr,
                               VTK_POINT_DATA, "error_quadratic", error_quadratic_ptr,
                               VTK_CELL_DATA, "level", l_p);

        ierr = VecRestoreArray(phi,             &phi_ptr);
//        ierr = VecRestoreArray(phi_smoothed,    &phi_ptr);
        ierr = VecRestoreArray(phi_reinit,      &phi_reinit_ptr);
        ierr = VecRestoreArray(f_exact,         &f_exact_ptr);
        ierr = VecRestoreArray(f_const,         &f_const_ptr);
        ierr = VecRestoreArray(f_linear,        &f_linear_ptr);
        ierr = VecRestoreArray(f_quadratic,     &f_quadratic_ptr);
        ierr = VecRestoreArray(f_flat,          &f_flat_ptr);
        ierr = VecRestoreArray(error_const,     &error_const_ptr);
        ierr = VecRestoreArray(error_linear,    &error_linear_ptr);
        ierr = VecRestoreArray(error_quadratic, &error_quadratic_ptr);

        ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
        ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

        ierr = PetscPrintf(mpi.comm(), "VTK saved in %s", oss.str().c_str());
      }

      ierr = PetscPrintf(mpi.comm(), "\n");

      // ---------------------------------------------------------
      // destroy the structures and release allocated memory
      // ---------------------------------------------------------
      ierr = VecDestroy(phi);
      ierr = VecDestroy(phi_reinit);
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
    filename = out_dir; filename += "/convergence/error_const.txt";     save_vector(filename.c_str(), error_const_all);
    filename = out_dir; filename += "/convergence/error_linear.txt";    save_vector(filename.c_str(), error_linear_all);
    filename = out_dir; filename += "/convergence/error_quadratic.txt"; save_vector(filename.c_str(), error_quadratic_all);

    filename = out_dir; filename += "/convergence";
    ierr = PetscPrintf(mpi.comm(), "Convergence data saved in %s\n", filename.c_str());
  }



  w.stop(); w.read_duration();
}
