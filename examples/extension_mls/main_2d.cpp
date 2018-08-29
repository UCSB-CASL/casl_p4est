
/*
 * Test the cell based multi level-set p4est.
 * Intersection of two circles
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_tools_mls.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_tools_mls.h>
#endif


#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#include "problem_case_0.h" // triangle (tetrahedron)
#include "problem_case_1.h" // two circles union
#include "problem_case_2.h" // two circles intersection
#include "problem_case_3.h" // two circles coloration
#include "problem_case_4.h" // four flowers
#include "problem_case_5.h" // two circles coloration (naive)
#include "problem_case_6.h" // one flower
#include "problem_case_7.h" // three flowers
#include "problem_case_8.h" // half-space
#include "problem_case_9.h" // angle
#include "problem_case_10.h" // angle 3d
#include "problem_case_11.h" // circular sector

#undef MIN
#undef MAX

using namespace std;

//-------------------------------------
// computational domain parameters
//-------------------------------------
const int periodicity[3] = {0, 0, 0};
const int num_trees[3]   = {1, 1, 1};
const double grid_xyz_min[3] = {-1.25, -1.25, -1.25};
const double grid_xyz_max[3] = { 1.25,  1.25,  1.25};

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
int lmin = 5;
int lmax = 5;
int num_splits = 3;
int num_splits_per_split = 1;
int num_shifts_x_dir = 5;
int num_shifts_y_dir = 5;
int num_shifts_z_dir = 5;
#else
int lmin = 6;
int lmax = 12;
int num_splits = 1;
int num_splits_per_split = 1;
int num_shifts_x_dir = 1;
int num_shifts_y_dir = 1;
int num_shifts_z_dir = 1;
#endif

double lip = 2;

int num_shifts_total = num_shifts_x_dir*
                       num_shifts_y_dir*
                       num_shifts_z_dir;

int num_resolutions = (num_splits-1)*num_splits_per_split + 1;
int num_iter_total  = num_resolutions*num_shifts_total;

int iter_start = 0; // is used to skip iterations and get to a problematic case

//-------------------------------------
// test solutions
//-------------------------------------
int n_geometry = 0;
int n_test     = 0;

//-------------------------------------
// output
//-------------------------------------
bool save_vtk           = 1;
bool save_domain        = 0;
bool save_convergence   = 1;

// EXACT SOLUTION
#include "exact_solutions.h"

#ifdef P4_TO_P8
std::vector<CF_3 *> phi_cf;
std::vector<CF_3 *> phi_x_cf, phi_y_cf, phi_z_cf;
std::vector<CF_3 *> bc_coeffs_cf;
#else
std::vector<CF_2 *> phi_cf;
std::vector<CF_2 *> phi_x_cf, phi_y_cf;
std::vector<CF_2 *> bc_coeffs_cf;
#endif

std::vector<action_t> action;
std::vector<int> color;

problem_case_0_t problem_case_0;
problem_case_1_t problem_case_1;
problem_case_2_t problem_case_2;
problem_case_3_t problem_case_3;
problem_case_4_t problem_case_4;
problem_case_5_t problem_case_5;
problem_case_6_t problem_case_6;
problem_case_7_t problem_case_7;
problem_case_8_t problem_case_8;
problem_case_9_t problem_case_9;
problem_case_10_t problem_case_10;
problem_case_11_t problem_case_11;

void set_parameters()
{
  switch (n_geometry)
  {
    case 0:
    {
      phi_cf        = problem_case_0.phi_cf;
      phi_x_cf      = problem_case_0.phi_x_cf;
      phi_y_cf      = problem_case_0.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_0.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_0.bc_coeffs_cf;
      action        = problem_case_0.action;
      color         = problem_case_0.color;
    } break;
    case 1:
    {
      phi_cf        = problem_case_1.phi_cf;
      phi_x_cf      = problem_case_1.phi_x_cf;
      phi_y_cf      = problem_case_1.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_1.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_1.bc_coeffs_cf;
      action        = problem_case_1.action;
      color         = problem_case_1.color;
    } break;
    case 2:
    {
      phi_cf        = problem_case_2.phi_cf;
      phi_x_cf      = problem_case_2.phi_x_cf;
      phi_y_cf      = problem_case_2.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_2.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_2.bc_coeffs_cf;
      action        = problem_case_2.action;
      color         = problem_case_2.color;
    } break;
    case 3:
    {
      phi_cf        = problem_case_3.phi_cf;
      phi_x_cf      = problem_case_3.phi_x_cf;
      phi_y_cf      = problem_case_3.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_3.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_3.bc_coeffs_cf;
      action        = problem_case_3.action;
      color         = problem_case_3.color;
    } break;
    case 4:
    {
      phi_cf        = problem_case_4.phi_cf;
      phi_x_cf      = problem_case_4.phi_x_cf;
      phi_y_cf      = problem_case_4.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_4.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_4.bc_coeffs_cf;
      action        = problem_case_4.action;
      color         = problem_case_4.color;
    } break;
    case 5:
    {
      phi_cf        = problem_case_5.phi_cf;
      phi_x_cf      = problem_case_5.phi_x_cf;
      phi_y_cf      = problem_case_5.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_5.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_5.bc_coeffs_cf;
      action        = problem_case_5.action;
      color         = problem_case_5.color;
    } break;
    case 6:
    {
      phi_cf        = problem_case_6.phi_cf;
      phi_x_cf      = problem_case_6.phi_x_cf;
      phi_y_cf      = problem_case_6.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_6.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_6.bc_coeffs_cf;
      action        = problem_case_6.action;
      color         = problem_case_6.color;
    } break;
    case 7:
    {
      phi_cf        = problem_case_7.phi_cf;
      phi_x_cf      = problem_case_7.phi_x_cf;
      phi_y_cf      = problem_case_7.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_7.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_7.bc_coeffs_cf;
      action        = problem_case_7.action;
      color         = problem_case_7.color;
    } break;
    case 8:
    {
      phi_cf        = problem_case_8.phi_cf;
      phi_x_cf      = problem_case_8.phi_x_cf;
      phi_y_cf      = problem_case_8.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_8.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_8.bc_coeffs_cf;
      action        = problem_case_8.action;
      color         = problem_case_8.color;
    } break;
    case 9:
    {
      phi_cf        = problem_case_9.phi_cf;
      phi_x_cf      = problem_case_9.phi_x_cf;
      phi_y_cf      = problem_case_9.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_9.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_9.bc_coeffs_cf;
      action        = problem_case_9.action;
      color         = problem_case_9.color;
    } break;
    case 10:
    {
      phi_cf        = problem_case_10.phi_cf;
      phi_x_cf      = problem_case_10.phi_x_cf;
      phi_y_cf      = problem_case_10.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_10.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_10.bc_coeffs_cf;
      action        = problem_case_10.action;
      color         = problem_case_10.color;
    } break;
    case 11:
    {
      phi_cf        = problem_case_11.phi_cf;
      phi_x_cf      = problem_case_11.phi_x_cf;
      phi_y_cf      = problem_case_11.phi_y_cf;
#ifdef P4_TO_P8
      phi_z_cf      = problem_case_11.phi_z_cf;
#endif
      bc_coeffs_cf  = problem_case_11.bc_coeffs_cf;
      action        = problem_case_11.action;
      color         = problem_case_11.color;
    } break;
  }
}

// BC VALUES
#ifdef P4_TO_P8
class bc_value_robin_t : public CF_3
{
  CF_3 *u, *ux, *uy, *uz;
  CF_3 *phi_x, *phi_y, *phi_z;
  CF_3 *kappa;
  CF_3 *mu;
public:
  bc_value_robin_t(CF_3 *u, CF_3 *ux, CF_3 *uy, CF_3 *uz, CF_3 *mu, CF_3 *phi_x, CF_3 *phi_y, CF_3 *phi_z, CF_3 *kappa) :
    u(u), ux(ux), uy(uy), uz(uz), mu(mu), phi_x(phi_x), phi_y(phi_y), phi_z(phi_z), kappa(kappa) {}
  double operator()(double x, double y, double z) const
  {
    double nx = (*phi_x)(x,y,z);
    double ny = (*phi_y)(x,y,z);
    double nz = (*phi_z)(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    return (*mu)(x,y,z)*(nx*(*ux)(x,y,z) + ny*(*uy)(x,y,z) + nz*(*uz)(x,y,z)) + (*kappa)(x,y,z)*(*u)(x,y,z);
  }
};
#else
class bc_value_robin_t : public CF_2
{
  CF_2 *u, *ux, *uy;
  CF_2 *mu;
  CF_2 *phi_x, *phi_y;
  CF_2 *kappa;
public:
  bc_value_robin_t(CF_2 *u, CF_2 *ux, CF_2 *uy, CF_2 *mu, CF_2 *phi_x, CF_2 *phi_y, CF_2 *kappa) :
    u(u), ux(ux), uy(uy), mu(mu), phi_x(phi_x), phi_y(phi_y), kappa(kappa) {}
  double operator()(double x, double y) const
  {
    double nx = (*phi_x)(x,y);
    double ny = (*phi_y)(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return (*mu)(x,y)*(nx*(*ux)(x,y) + ny*(*uy)(x,y)) + (*kappa)(x,y)*(*u)(x,y);
  }
};
#endif

// additional output functions
double compute_convergence_order(std::vector<double> &x, std::vector<double> &y);
void print_convergence_table(MPI_Comm mpi_comm,
                             std::vector<double> &level, std::vector<double> &h,
                             std::vector<double> &L_one, std::vector<double> &L_avg, std::vector<double> &L_dev, std::vector<double> &L_max,
                             std::vector<double> &Q_one, std::vector<double> &Q_avg, std::vector<double> &Q_dev, std::vector<double> &Q_max);

int main (int argc, char* argv[])
{
  // error variables
  PetscErrorCode ierr;
  int mpiret;

  // mpi
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  //-------------------------------------
  // refinement parameters
  //-------------------------------------
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");

  cmd.add_option("num_splits",           "number of recursive splits");
  cmd.add_option("num_splits_per_split", "number of additional resolutions");

  cmd.add_option("num_shifts_x_dir", "number of shifts in x-dir");
  cmd.add_option("num_shifts_y_dir", "number of shifts in y-dir");
  cmd.add_option("num_shifts_z_dir", "number of shifts in z-dir");

  //-------------------------------------
  // output
  //-------------------------------------
  cmd.add_option("save_vtk",           "Save the p4est in vtk format");
  cmd.add_option("save_domain",        "Save the reconstruction of an irregular domain (works only in serial!)");
  cmd.add_option("save_convergence",   "Save convergence results");

  //-------------------------------------
  // test solution
  //-------------------------------------
  cmd.add_option("n_test",     "Test function");
  cmd.add_option("n_geometry", "Irregular domain (0 - triangle/tetrahedron, 1 - union, 2 - difference, 7 - three flowers, 11 - circular sector)");

  //-------------------------------------
  // level-set representation parameters
  //-------------------------------------
  cmd.parse(argc, argv);

  //-------------------------------------
  // refinement parameters
  //-------------------------------------
  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);

  num_splits           = cmd.get("num_splits",           num_splits);
  num_splits_per_split = cmd.get("num_splits_per_split", num_splits_per_split);

  num_shifts_x_dir = cmd.get("num_shifts_x_dir", num_shifts_x_dir);
  num_shifts_y_dir = cmd.get("num_shifts_y_dir", num_shifts_y_dir);
  num_shifts_z_dir = cmd.get("num_shifts_z_dir", num_shifts_z_dir);

  //-------------------------------------
  // test solutions
  //-------------------------------------
  n_test     = cmd.get("n_test",     n_test);
  n_geometry = cmd.get("n_geometry", n_geometry);

  //-------------------------------------
  // output
  //-------------------------------------
  save_vtk           = cmd.get("save_vtk",           save_vtk);
  save_domain        = cmd.get("save_domain",        save_domain);
  save_convergence   = cmd.get("save_convergence",   save_convergence);

  // recalculate depending parameters
  num_shifts_total = num_shifts_x_dir*num_shifts_y_dir*num_shifts_z_dir;

  num_resolutions = (num_splits-1)*num_splits_per_split + 1;
  num_iter_total = num_resolutions*num_shifts_total;

  set_parameters();

  // prepare output directories
  const char* out_dir = getenv("OUT_DIR");

  if (!out_dir &&
      (save_vtk ||
       save_domain ||
       save_convergence))
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save results\n");
    return -1;
  }

  if (save_vtk)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  if (save_domain)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/geometry";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/geometry directory");
  }

  if (save_convergence)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/convergence directory");
  }

  // vectors to store convergence results
  vector<double> lvl_arr, h_arr;

  vector<double> error_ex_arr;

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  // the effective LSF
  level_set_tot_t level_set_tot_cf(&phi_cf, &action, &color);

  int iteration = -1;
  int file_idx  = -1;

  for(int iter=0; iter<num_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin+iter, lmax+iter); CHKERRXX(ierr);

    int num_sub_iter = (iter == 0 ? 1 : num_splits_per_split);

    for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
    {

      double grid_xyz_min_alt[3];
      double grid_xyz_max_alt[3];

      double scale = (double) (num_sub_iter-1-sub_iter) / (double) num_sub_iter;
      grid_xyz_min_alt[0] = grid_xyz_min[0] - .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]); grid_xyz_max_alt[0] = grid_xyz_max[0] + .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]);
      grid_xyz_min_alt[1] = grid_xyz_min[1] - .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]); grid_xyz_max_alt[1] = grid_xyz_max[1] + .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]);
      grid_xyz_min_alt[2] = grid_xyz_min[2] - .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]); grid_xyz_max_alt[2] = grid_xyz_max[2] + .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]);


      double dxyz[3] = { (grid_xyz_max_alt[0]-grid_xyz_min_alt[0])/pow(2., (double) lmax+iter),
                         (grid_xyz_max_alt[1]-grid_xyz_min_alt[1])/pow(2., (double) lmax+iter),
                         (grid_xyz_max_alt[2]-grid_xyz_min_alt[2])/pow(2., (double) lmax+iter) };

      double grid_xyz_min_shift[3];
      double grid_xyz_max_shift[3];

#ifdef P4_TO_P8
      double dxyz_m = MIN(dxyz[0],dxyz[1],dxyz[2]);
#else
      double dxyz_m = MIN(dxyz[0],dxyz[1]);
#endif

      h_arr.push_back(dxyz_m);
      lvl_arr.push_back(lmax+iter-scale);

      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f).\n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale); CHKERRXX(ierr);

#ifdef P4_TO_P8
      for (int k_shift = 0; k_shift < num_shifts_z_dir; ++k_shift)
      {
        grid_xyz_min_shift[2] = grid_xyz_min_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir) * dxyz[2];
        grid_xyz_max_shift[2] = grid_xyz_max_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir) * dxyz[2];
#endif
        for (int j_shift = 0; j_shift < num_shifts_y_dir; ++j_shift)
        {
          grid_xyz_min_shift[1] = grid_xyz_min_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir) * dxyz[1];
          grid_xyz_max_shift[1] = grid_xyz_max_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir) * dxyz[1];

          for (int i_shift = 0; i_shift < num_shifts_x_dir; ++i_shift)
          {
            grid_xyz_min_shift[0] = grid_xyz_min_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir) * dxyz[0];
            grid_xyz_max_shift[0] = grid_xyz_max_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir) * dxyz[0];

            iteration++;

            if (iteration < iter_start) continue;

            file_idx++;

            connectivity = my_p4est_brick_new(num_trees, grid_xyz_min_shift, grid_xyz_max_shift, &brick, periodicity);

            p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

            //    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

//            splitting_criteria_cf_t data_tmp(lmin, lmax, &level_set_tot_cf, lip);
//            p4est->user_pointer = (void*)(&data_tmp);

//            //    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
//            my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
//            my_p4est_partition(p4est, P4EST_FALSE, NULL);
//            for (int i = 0; i < iter; ++i)
//            {
//              my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
//              my_p4est_partition(p4est, P4EST_FALSE, NULL);
//            }

            splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot_cf, lip);
            p4est->user_pointer = (void*)(&data);

            my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
            my_p4est_partition(p4est, P4EST_FALSE, NULL);

            //    my_p4est_partition(p4est, P4EST_FALSE, NULL);
            //    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
            //    my_p4est_partition(p4est, P4EST_FALSE, NULL);

            ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
            //    my_p4est_ghost_expand(p4est, ghost);
            nodes = my_p4est_nodes_new(p4est, ghost);

            my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
            my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

            my_p4est_level_set_t ls(&ngbd_n);

            double dxyz[P4EST_DIM];
            dxyz_min(p4est, dxyz);

#ifdef P4_TO_P8
            double dxyz_max = MAX(dxyz[0], dxyz[1], dxyz[2]);
#else
            double dxyz_max = MAX(dxyz[0], dxyz[1]);
#endif

            unsigned int num_surfaces = phi_cf.size();

            // sample level-set functions
            std::vector<Vec> phi;
            for (unsigned int i = 0; i < num_surfaces; i++)
            {
              phi.push_back(Vec());
              ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, *phi_cf[i], phi.back());

//              if (reinit_level_set)
//                ls.reinitialize_1st_order_time_2nd_order_space(phi.back(),20);
            }

            Vec phi_eff;
            ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, level_set_tot_cf, phi_eff);

            Vec sol;
            ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, u_cf, sol);

            /* calculate errors */
            Vec vec_error_ex; double *vec_error_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex); CHKERRXX(ierr);

            //----------------------------------------------------------------------------------------------
            // calculate extrapolation error
            //----------------------------------------------------------------------------------------------
            // smoothed LSF
            //    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 25.*dxyz_max*dxyz_max);
            level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 9.*dxyz_max*dxyz_max);
//                level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 0.5*dxyz_max);
//                level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 0.01);

            Vec phi_smooth;
            double *phi_smooth_ptr;
            ierr = VecCreateGhostNodes(p4est, nodes, &phi_smooth); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, level_set_smooth_cf, phi_smooth);

//            if (do_extension)
            {
              ls.reinitialize_1st_order_time_2nd_order_space(phi_smooth);
              ls.reinitialize_1st_order_time_2nd_order_space(phi_eff);
//              ls.reinitialize_2nd_order(phi_eff);
//              ls.reinitialize_1st_order_time_2nd_order_space(phi_eff);
            }

            double band = 4.0;

            // copy solution into a new Vec
            Vec sol_ex; ierr = VecCreateGhostNodes(p4est, nodes, &sol_ex); CHKERRXX(ierr);

            double *sol_ptr;    ierr = VecGetArray(sol,    &sol_ptr);    CHKERRXX(ierr);
            double *sol_ex_ptr; ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
            double *phi_eff_ptr;
            ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

            for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
            {
              if (phi_eff_ptr[i] < 0)
                sol_ex_ptr[i] = sol_ptr[i];
              else                    sol_ex_ptr[i] = 0;
            }

            ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

            // extend
            //    ls.extend_Over_Interface_TVD(phi_smooth, sol_ex, 100); CHKERRXX(ierr);
            //    ls.extend_Over_Interface_TVD(phi_eff, sol_ex, 100); CHKERRXX(ierr);
//            ls.extend_Over_Interface_Iterative(phi_eff, phi_smooth, phi_eff, sol_ex, 10, 2, 5);
            for (int i = 0; i < 1; ++i)
            {
//              ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, sol_ex, 100, 2); CHKERRXX(ierr);
//              ls.extend_Over_Interface_Iterative(phi_eff, phi_smooth, phi_eff, sol_ex, 10, 2, 5);
              ls.extend_Over_Interface_TVD_full(phi_eff, phi_eff, sol_ex, 100, 2);
            }
//            for (int i = 0; i < 11; ++i)
//              ls.extend_Over_Interface(phi_smooth, sol_ex, 2, 10);
//                ls.extend_Over_Interface_TVD(phi_eff, phi_eff, sol_ex, 100); CHKERRXX(ierr);

            // calculate error
            ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);

            ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

            for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
            {
              if (phi_eff_ptr[n] > 0. && phi_eff_ptr[n] < band*dxyz_max)
//                if (phi_smooth_ptr[n] > 0. && phi_smooth_ptr[n] < band*dxyz_max)
              {
                double xyz[P4EST_DIM];
                node_xyz_fr_n(n, p4est, nodes, xyz);
#ifdef P4_TO_P8
                vec_error_ex_ptr[n] = ABS(sol_ex_ptr[n] - u_cf(xyz[0],xyz[1],xyz[2]));
#else
                vec_error_ex_ptr[n] = ABS(sol_ex_ptr[n] - u_cf(xyz[0],xyz[1]));
#endif
              }
              else
                vec_error_ex_ptr[n] = 0;
            }

            ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

            ierr = VecGhostUpdateBegin(vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateEnd  (vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

            // compute L-inf norm of errors
            double err_ex_max = 0.; VecMax(vec_error_ex, NULL, &err_ex_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

            // Store error values
            error_ex_arr.push_back(err_ex_max);

            // Print current errors
            if (iter > -1)
            {
              ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f). Iteration %6d / %6d \n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale, iteration, num_iter_total); CHKERRXX(ierr);
              ierr = PetscPrintf(mpi.comm(), "Error (ex): %g\n", err_ex_max); CHKERRXX(ierr);
            }

            if (save_vtk)
            {
              std::ostringstream oss;

              oss << out_dir
                  << "/vtu/nodes_"
                  << mpi.size() << "_"
                  << brick.nxyztrees[0] << "x"
                  << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
                     "x" << brick.nxyztrees[2] <<
       #endif
                     "." << file_idx;

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

              double *phi_eff_ptr;
              ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);

              my_p4est_vtk_write_all(p4est, nodes, ghost,
                                     P4EST_TRUE, P4EST_TRUE,
                                     5, 1, oss.str().c_str(),
                                     VTK_POINT_DATA, "phi", phi_eff_ptr,
                                     VTK_POINT_DATA, "phi_smooth", phi_smooth_ptr,
                                     VTK_POINT_DATA, "sol", sol_ptr,
                                     VTK_POINT_DATA, "sol_ex", sol_ex_ptr,
                                     VTK_POINT_DATA, "error_ex", vec_error_ex_ptr,
                                     VTK_CELL_DATA , "leaf_level", l_p);

              ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);

              ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
              ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

              PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
            }

            // destroy Vec's with errors
            ierr = VecDestroy(vec_error_ex); CHKERRXX(ierr);

            ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);
            ierr = VecDestroy(sol_ex); CHKERRXX(ierr);

            for (unsigned int i = 0; i < phi.size(); i++)
            {
              ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
            }

            ierr = VecDestroy(sol);         CHKERRXX(ierr);

            p4est_nodes_destroy(nodes);
            p4est_ghost_destroy(ghost);
            p4est_destroy      (p4est);
            my_p4est_brick_destroy(connectivity, &brick);

          }
        }
#ifdef P4_TO_P8
      }
#endif
    }
  }

  MPI_Barrier(mpi.comm());

  std::vector<double> error_ex_one(num_resolutions, 0), error_ex_avg(num_resolutions, 0), error_ex_max(num_resolutions, 0);


  // for each resolution compute max, mean and deviation
  for (int p = 0; p < num_resolutions; ++p)
  {
    // one
    error_ex_one[p] = error_ex_arr[p*num_shifts_total];

    // max
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_ex_max[p] = MAX(error_ex_max[p], error_ex_arr[p*num_shifts_total + s]);
    }

    // avg
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_ex_avg[p] += error_ex_arr[p*num_shifts_total + s];
    }

    error_ex_avg[p] /= num_shifts_total;
  }

  if (mpi.rank() == 0)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    std::string filename;

    // save level and resolution
    filename = out_dir; filename += "/convergence/lvl.txt";   save_vector(filename.c_str(), lvl_arr);
    filename = out_dir; filename += "/convergence/h_arr.txt"; save_vector(filename.c_str(), h_arr);

    filename = out_dir; filename += "/convergence/error_ex_all.txt"; save_vector(filename.c_str(), error_ex_arr);
    filename = out_dir; filename += "/convergence/error_ex_one.txt"; save_vector(filename.c_str(), error_ex_one);
    filename = out_dir; filename += "/convergence/error_ex_avg.txt"; save_vector(filename.c_str(), error_ex_avg);
    filename = out_dir; filename += "/convergence/error_ex_max.txt"; save_vector(filename.c_str(), error_ex_max);

  }

  w.stop(); w.read_duration();

  return 0;
}

double compute_convergence_order(std::vector<double> &x, std::vector<double> &y)
{
  if (x.size() != y.size())
  {
    std::cout << "[ERROR]: sizes of arrays do not coincide\n";
    return 0;
  }

  int n = x.size();

  double sumX  = 0;
  double sumY  = 0;
  double sumXY = 0;
  double sumXX = 0;

  for (int i = 0; i < n; ++i)
  {
    double logX = log(x[i]);
    double logY = log(y[i]);

    sumX  += logX;
    sumY  += logY;
    sumXY += logX*logY;
    sumXX += logX*logX;
  }

  return (sumXY - sumX*sumY/n)/(sumXX - sumX*sumX/n);
}
