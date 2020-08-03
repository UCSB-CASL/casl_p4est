
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
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_scft.h>
#include <src/my_p8est_save_load.h>
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
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_save_load.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

param_list_t pl;

using namespace std;

// comptational domain
param_t<double> xmin (pl, -12.5, "xmin", "xmin");
param_t<double> ymin (pl, -12.5, "ymin", "ymin");
param_t<double> zmin (pl, -12.5, "zmin", "zmin");

param_t<double> xmax (pl,  12.5, "xmax", "xmax");
param_t<double> ymax (pl,  12.5, "ymax", "ymax");
param_t<double> zmax (pl,  12.5, "zmax", "zmax");

param_t<int> nx (pl, 1, "nx", "number of trees in x-dimension");
param_t<int> ny (pl, 1, "ny", "number of trees in y-dimension");
param_t<int> nz (pl, 1, "nz", "number of trees in z-dimension");

// grid parameters
#ifdef P4_TO_P8
param_t<int> lmin (pl, 7, "lmin", "min level of trees");
param_t<int> lmax (pl, 7, "lmax", "max level of trees");
#else
param_t<int> lmin (pl, 8, "lmin", "min level of trees");
param_t<int> lmax (pl, 8, "lmax", "max level of trees");
#endif

param_t<double> lip (pl, 1.5, "lip", "Lipschitz constant");

// advection parameters
param_t<double> cfl               (pl, 0.5, "cfl", "");
param_t<int>    max_iterations    (pl, 1000, "max_iterations", "");
param_t<int>    scheme            (pl, 2, "scheme", "0 - pure curvature, 1 - Gaddiel's method, 2 - gradient-based method");
param_t<double> curvature_penalty (pl, 0.01*0, "curvature_penalty", "");

// scft parameters
param_t<int>    num_scft_subiters   (pl, 1, "num_scft_subiters",   "Maximum SCFT iterations");
param_t<int>    max_scft_iterations (pl, 200, "max_scft_iterations",   "Maximum SCFT iterations");
param_t<int>    bc_adjust_min       (pl, 5, "bc_adjust_min",     "Minimun SCFT steps between adjusting BC");
param_t<bool>   smooth_pressure     (pl, 1, "smooth_pressure",     "Smooth pressure after first BC adjustment 0/1");
param_t<double> scft_tol            (pl, 1.e-3, "scft_tol", "Tolerance for SCFT");
param_t<double> scft_bc_tol         (pl, 1.e-2, "scft_bc_tol", "Tolerance for adjusting BC");

// polymer
param_t<double> box_size (pl, 1, "box_size.val", "Box size in units of Rg");
param_t<double> f        (pl, 0.3, "f", "Fraction of polymer A");
param_t<double> XN       (pl, 30, "XN.val",  "Flory-Higgins interaction parameter");
param_t<int>    ns       (pl, 60, "ns",  "Discretization of polymer chain");

// output parameters
param_t<bool> save_vtk        (pl, 1, "save_vtk", "");
param_t<bool> save_p4est      (pl, 1, "save_p4est", "");
param_t<int>  save_data       (pl, 1, "save_data", "");
param_t<int>  save_parameters (pl, 1, "save_parameters", "");
param_t<int>  save_every_dn   (pl, 1, "save_every_dn", "");

// problem setting
param_t<int> num_target  (pl, 1, "num_target", "Target design: "
                                                   "0 - no target (for testing), "
                                                   "1 - one circle, "
                                                   "2 - two circles, "
                                                   "3 - three circles aligned, "
                                                   "4 - three circles triangle, "
                                                   "5 - six circles hexagonal, "
                                                   "6 - six circles v-shape, "
                                                   "7 - seven circles L-shape, ");
param_t<int> num_guess   (pl, 0, "num_guess", "Type of initial guess: 0 - target with margins, 1 - enclosing box");
param_t<int> num_seed    (pl, 0, "num_seed", "Seed for SCFT: 0 - target field, 1 - random");
param_t<int> num_example (pl, 0, "num_example", "");

param_t<bool> minimize (pl, 0, "minimize", "");

param_t<int> design_check_frequency  (pl, 0, "design_check_frequency", "");
param_t<int> design_check_iterations (pl, 200, "design_check_iterations", "");

// geometry parameters
param_t<double> r0               (pl, 1, "r0", "Radius of target wells");
param_t<double> guess_margin     (pl, 2.5, "guess_margin", "");
param_t<double> target_smoothing (pl, sqrt(XN.val), "target_smoothing", "");
param_t<double> mask_smoothing   (pl, 0.2, "mask_smoothing", "");

// surface tension
param_t<double> XN_wall_avg (pl, 0.0, "XN.val_wall_avg", "Polymer-air surface energy strength: average");
param_t<double> XN_wall_del (pl, 0.0, "XN.val_wall_del", "Polymer-air surface energy strength: difference");

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

param_t<double> dist  (pl, 3.3, "dist", "Characteristic distance between wells");
param_t<double> angle (pl, 120, "angle", "Characteristic distance between wells");

/* target morphology */
class phi_target_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_target.val)
    {
      case 0:
      return 0;
      case 1: // 1 cylinder
      {
        int num = 1;
        double xc[] = {0.0};
        double yc[] = {0.0};
        double rc[] = {r0.val};

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 2: // 2 cylinders
      {
        int num = 2;
        double xc[] = { -.5*dist.val, .5*dist.val };
        double yc[] = { 0.0, 0.0 };
        double rc[] = { r0.val, r0.val };

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 3: // 3 cylinders aligned
      {
        int num = 3;
        double xc[] = { -dist.val, 0, dist.val };
        double yc[] = { 0, 0, 0 };
        double rc[] = { r0.val, r0.val, r0.val };

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 4: // 3 cylinders triangle
      {
        double h = dist.val*cos(PI/6.);
        double xc0 = -.5*dist.val, yc0 = -.5*h;
        double xc1 =  .5*dist.val, yc1 = -.5*h;
        double xc2 =  0.0,  yc2 = .5*h;

        double phi0 = sqrt(SQR(x-xc0/box_size.val)+SQR(y-yc0/box_size.val)) - r0.val/box_size.val;
        double phi1 = sqrt(SQR(x-xc1/box_size.val)+SQR(y-yc1/box_size.val)) - r0.val/box_size.val;
        double phi2 = sqrt(SQR(x-xc2/box_size.val)+SQR(y-yc2/box_size.val)) - r0.val/box_size.val;

        return MIN(phi0, phi1, phi2);
      }
      case 5: // 7 cylinders hex
      {
        int num = 7;
        double h = dist.val*cos(PI/6.);
        double xc[] = { 0, -dist.val, dist.val, -.5*dist.val, .5*dist.val, -.5*dist.val, .5*dist.val };
        double yc[] = { 0, 0, 0, -h, -h, h, h };
        double rc[] = { r0.val, r0.val, r0.val, r0.val, r0.val, r0.val, r0.val };

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 6: // 6 cylinders V-shaped
      {
        int num = 6;
        double h = dist.val*cos(PI/4.);
        double xc[] = { -.55*dist.val - 2.*h, -.55*dist.val - 1.*h, -.55*dist.val,
                         .55*dist.val + 2.*h,  .55*dist.val + 1.*h,  .55*dist.val};
        double yc[] = { -h, 0, h, -h, 0, h, };
        double rc[] = { r0.val, r0.val, r0.val, r0.val, r0.val, r0.val };

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 7: // 5 cylinders L-shaped
      {
        const int num = 5;
        double xc[num], yc[num], rc[num];
        yc[0] = -dist.val; xc[0] = -2.*dist.val; rc[0] = r0.val;
        yc[1] = -dist.val; xc[1] = -1.*dist.val; rc[1] = r0.val;
        yc[2] = -dist.val; xc[2] = -0.*dist.val; rc[2] = r0.val;
        yc[3] = -dist.val + 1.*dist.val*sin(angle.val/180.*PI); xc[3] = -1.*dist.val*cos(angle.val/180.*PI); rc[3] = r0.val;
        yc[4] = -dist.val + 2.*dist.val*sin(angle.val/180.*PI); xc[4] = -2.*dist.val*cos(angle.val/180.*PI); rc[4] = r0.val;

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 8: // 3 cylinders L-shaped
      {
        int num = 3;
        double xc[] = { -0.5*dist.val, -0.5*dist.val,  0.5*dist.val };
        double yc[] = { -0.5*dist.val,  0.5*dist.val,  0.5*dist.val };
        double rc[] = { r0.val, r0.val, r0.val };

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;
      }
      case 9: // 9 cylinders grqid
      {
        const int num = 5;
        double xc[num], yc[num], rc[num];
        yc[0] = -dist.val; xc[0] = -2.*dist.val; rc[0] = r0.val;
        yc[1] = -dist.val; xc[1] = -1.*dist.val; rc[1] = r0.val;
        yc[2] = -dist.val; xc[2] = -0.*dist.val; rc[2] = r0.val;
        yc[3] = -dist.val + 1.*dist.val*sin(angle.val/180.*PI); xc[3] = -1.*dist.val*cos(angle.val/180.*PI); rc[3] = r0.val;
        yc[4] = -dist.val + 2.*dist.val*sin(angle.val/180.*PI); xc[4] = -2.*dist.val*cos(angle.val/180.*PI); rc[4] = r0.val;

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }

        return result;

        return result;
      }
      case 10: // 6 cylinders split
      {
        int num = 6;
        double xc[num], yc[num], rc[num];
        double dx = dist.val*cos(1.*PI/3.);
        double dy = dist.val*sin(1.*PI/3.);
        xc[0] = -1.*dist.val; yc[0] = 0; rc[0] = r0.val;
        xc[1] = -0.*dist.val; yc[1] = 0; rc[1] = r0.val;
        xc[2] = dx+1.*dist.val; yc[2] = dy; rc[2] = r0.val;
        xc[3] = dx+0.*dist.val; yc[3] = dy; rc[3] = r0.val;
        xc[4] = dx+1.*dist.val; yc[4] = -dy; rc[4] = r0.val;
        xc[5] = dx+0.*dist.val; yc[5] = -dy; rc[5] = r0.val;

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }
        return result;
      }
      case 11: // 7 cylinders H
      {
        int num = 7;
        double xc[num], yc[num], rc[num];
        xc[0] =  0.*dist.val; yc[0] =  0.*dist.val; rc[0] = r0.val;
        xc[1] = -1.*dist.val; yc[1] = -1.*dist.val; rc[1] = r0.val;
        xc[2] =  0.*dist.val; yc[2] = -1.*dist.val; rc[2] = r0.val;
        xc[3] =  1.*dist.val; yc[3] = -1.*dist.val; rc[3] = r0.val;
        xc[4] = -1.*dist.val; yc[4] =  1.*dist.val; rc[4] = r0.val;
        xc[5] =  0.*dist.val; yc[5] =  1.*dist.val; rc[5] = r0.val;
        xc[6] =  1.*dist.val; yc[6] =  1.*dist.val; rc[6] = r0.val;

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }
        return result;
      }
      case 12: // 6 cylinders connector
      {
        int num = 6;
        double xc[num], yc[num], rc[num];
        xc[0] =  0.*dist.val; yc[0] =  0.*dist.val; rc[0] = r0.val;
        xc[1] = -1.*dist.val; yc[1] =  0.*dist.val; rc[1] = r0.val;
        xc[2] = -2.*dist.val; yc[2] =  0.*dist.val; rc[2] = r0.val;
        xc[3] =  0.*dist.val; yc[3] =  1.*dist.val; rc[3] = r0.val;
        xc[4] =  1.*dist.val; yc[4] =  1.*dist.val; rc[4] = r0.val;
        xc[5] =  2.*dist.val; yc[5] =  1.*dist.val; rc[5] = r0.val;

        double result = 100;
        for (int i = 0; i < num; ++i) {
          double current = sqrt(SQR(x-xc[i]/box_size.val)+SQR(y-yc[i]/box_size.val)) - rc[i]/box_size.val;
          result = MIN(result, current);
        }
        return result;
      }
      default: throw std::invalid_argument("Choose a valid test number");
    }
  }
} phi_target_cf;

class target_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    double phi = phi_target_cf(DIM(x, y, z));
    double sign = -1;

    if      (phi > 0) sign = 1;
    else if (phi < 0) sign =-1;
    else              sign = 0;

    return 0.6*XN.val*(sign*(exp(-fabs(phi)*target_smoothing.val*box_size.val)-1.0));
  }
} target_cf;

/* guess */
class guess_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_guess.val)
    {
      case 0: return phi_target_cf(DIM(x, y, z)) - guess_margin.val/box_size.val;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} guess_cf;

/* seed */
class seed_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z) ) const
  {
    switch (num_seed.val)
    {
      case 0: return target_cf(DIM(x,y,z));
      case 1: return 0.0*0.5*XN.val*(1.-2.*double(rand())/double(RAND_MAX));
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} seed_cf;

/* surface energies */
class gamma_A_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    EXECD((void) x, (void) y, (void) z);
    return sqrt(XN_wall_avg.val + XN_wall_del.val);
  }
} gamma_A_cf;

class gamma_B_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    EXECD((void) x, (void) y, (void) z);
    return sqrt(XN_wall_avg.val - XN_wall_del.val);
  }
} gamma_B_cf;

class vx_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return -x*cos(5.*t);
  }
} vx_cf;

class vy_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return y*sin(5.*t);
  }
} vy_cf;

class random_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double, double, double)) const
  {
    return 0.001*0.5*XN.val*(1.-2.*double(rand())/double(RAND_MAX));
  }
} random_cf;

inline void interpolate_between_grids(my_p4est_interpolation_nodes_t &interp, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, Vec *vec, Vec parent=NULL, interpolation_method method=quadratic_non_oscillatory_continuous_v2)
{
  PetscErrorCode ierr;
  Vec tmp;

  if (parent == NULL) {
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &tmp); CHKERRXX(ierr);
  } else {
    ierr = VecDuplicate(parent, &tmp); CHKERRXX(ierr);
  }

  interp.set_input(*vec, method);
  interp.interpolate(tmp);

  ierr = VecDestroy(*vec); CHKERRXX(ierr);
  *vec = tmp;
}

PetscErrorCode ierr;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  /* initialize MPI */
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  /* create an output directory */
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
    return -1;
  }

  if (mpi.rank() == 0) {
    std::ostringstream command;

    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if(ret_sys < 0)
      throw std::invalid_argument("Could not create directory");
  }

  FILE *file_conv;
  char file_conv_name[10000];
  if (save_data())
  {
    sprintf(file_conv_name, "%s/data.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_conv_name, "w", &file_conv); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_conv, "iteration "
                                               "energy "
                                               "energy_change_predicted "
                                               "energy_change_effective\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
  }

  /* parse command line arguments */
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  if (mpi.rank() == 0 && save_parameters()) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  double scaling = 1./box_size.val;

  double xyz_min[] = { DIM(xmin.val, ymin.val, zmin.val) };
  double xyz_max[] = { DIM(xmax.val, ymax.val, zmax.val) };
  int nb_trees[] = { DIM(nx.val, ny.val, nz.val) };
  int periodic[] = { DIM(0, 0, 0) };

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin.val, lmax.val, &guess_cf, lip.val);
  data.set_refine_only_inside(1);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, &h, &diag);

  /* initialize geometry */
  Vec phi; // phi is the template for all other Vec's
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, guess_cf, phi);

  VecShiftGhost(phi, -mask_smoothing.val/box_size.val);
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_2nd_order(phi);
  VecShiftGhost(phi,  mask_smoothing.val/box_size.val);

  /* initialize potentials */
  Vec mu_m; ierr = VecDuplicate(phi, &mu_m); CHKERRXX(ierr);
  Vec mu_p; ierr = VecDuplicate(phi, &mu_p); CHKERRXX(ierr);
  Vec nu_m; ierr = VecDuplicate(phi, &nu_m); CHKERRXX(ierr);
  Vec nu_p; ierr = VecDuplicate(phi, &nu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, seed_cf, mu_m);
  VecSetGhost(mu_p, 0);
  VecSetGhost(nu_m, 0);
  VecSetGhost(nu_p, 0);

  double cost_function_nnn = 0;
  double cost_function_nm1 = 0;

  double rho_avg_old = 1;

  /* main loop */
  int iteration = 0;
  while (iteration < max_iterations.val)
  {
    std::vector<Vec> phi_all(1);
    std::vector<mls_opn_t> acn(1, MLS_INTERSECTION);
    std::vector<int> clr(1);

    phi_all[0] = phi; clr[0] = 0;

    Vec mu_t;
    ierr = VecDuplicate(phi, &mu_t); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, target_cf, mu_t);

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi_all, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi, normal, kappa);
    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, kappa, phi);

    if (design_check_frequency() > 0) {
      if (iteration%design_check_frequency() == 0) {

        my_p4est_scft_t scft(ngbd, ns.val);

        scft.add_boundary(phi, MLS_INT, gamma_A_cf, gamma_B_cf);

        scft.set_scaling(scaling);
        scft.set_polymer(f.val, XN.val);
        scft.set_rho_avg(rho_avg_old);

        Vec mu_m_tmp = scft.get_mu_m();
        Vec mu_p_tmp = scft.get_mu_p();

        sample_cf_on_nodes(p4est, nodes, random_cf, mu_m_tmp);
        VecSetGhost(mu_p_tmp, 0);

        scft.initialize_solvers();
        scft.initialize_bc_smart(iteration != 0);

        int scft_iteration = 0;
        double scft_error = 2.*scft_tol.val;
        bool adaptive = false;
        while (scft_iteration < design_check_iterations.val)
        {
          for (int i= num_scft_subiters.val;i--;) {
            scft.initialize_bc_smart(adaptive); adaptive = true;
            scft.solve_for_propogators();
            scft.calculate_densities();
            ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                               scft_iteration,
                               scft.get_energy(),
                               scft.get_pressure_force(),
                               scft.get_exchange_force()); CHKERRXX(ierr);
          }
          scft.solve_for_propogators();
          scft.calculate_densities();
          scft.update_potentials();
          scft_iteration++;

          scft.save_VTK(scft_iteration);
          ierr = PetscPrintf(mpi.comm(), "Desing check: %d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);
          scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        }
      }
    }

    /* get density field and non-curvature velocity */
    Vec shape_grad;
    ierr = VecDuplicate(phi, &shape_grad); CHKERRXX(ierr);

    bool adaptive = false;

    if (scheme.val == 2)
    {
      my_p4est_scft_t scft(ngbd, ns.val);

      scft.add_boundary(phi, MLS_INT, gamma_A_cf, gamma_B_cf);

      scft.set_scaling(scaling);
      scft.set_polymer(f.val, XN.val);
      scft.set_rho_avg(rho_avg_old);

      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);
      VecSetGhost(mu_p_tmp, 0);

      scft.initialize_solvers();
      scft.initialize_bc_smart(iteration != 0);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol.val;
      int bc_iters = 0;
      while (scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val)
      {
        for (int i= num_scft_subiters.val;i--;) {
          scft.initialize_bc_smart(adaptive); adaptive = true;
          scft.solve_for_propogators();
          scft.calculate_densities();
//          scft.save_VTK(scft_iteration++);
          ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                             scft_iteration,
                             scft.get_energy(),
                             scft.get_pressure_force(),
                             scft.get_exchange_force()); CHKERRXX(ierr);
        }
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();
        scft_iteration++;
        bc_iters++;

        scft.save_VTK(scft_iteration);
        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
      }

      rho_avg_old = scft.get_rho_avg();

      scft.sync_and_extend();
      scft.dsa_initialize();

      Vec mu_t_tmp = scft.dsa_get_mu_t();
      Vec nu_m_tmp = scft.dsa_get_nu_m();
      Vec nu_p_tmp = scft.dsa_get_nu_p();

      VecCopyGhost(mu_t, mu_t_tmp);
      VecCopyGhost(nu_m, nu_m_tmp);
      VecCopyGhost(nu_p, nu_p_tmp);
      VecSetGhost(nu_m_tmp, 0);
      VecSetGhost(nu_p_tmp, 0);

      scft_iteration = 0;
      scft_error = 2.*scft_tol.val;
      while (scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val)
      {
        scft.dsa_solve_for_propogators();
        scft.dsa_compute_densities();
        scft.dsa_update_potentials();
        ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e\n", scft.dsa_get_nu_0(), scft.dsa_get_pressure_force(), scft.dsa_get_exchange_force()); CHKERRXX(ierr);
        scft_error = fabs(scft.dsa_get_exchange_force());
        scft_iteration++;
      }

      scft.dsa_sync_and_extend();

      cost_function_nm1 = cost_function_nnn;
      cost_function_nnn = scft.dsa_get_cost_function();
      scft.dsa_compute_shape_gradient(0, shape_grad);

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, shape_grad, phi);

      VecScaleGhost(shape_grad, -1);

      VecCopyGhost(mu_m_tmp, mu_m);
      VecCopyGhost(mu_p_tmp, mu_p);

      VecCopyGhost(nu_m_tmp, nu_m);
      VecCopyGhost(nu_p_tmp, nu_p);
    }
    else if (scheme.val == 1)
    {
      sample_cf_on_nodes(p4est, nodes, target_cf, mu_m);

      VecSetGhost(nu_m, 0);
      VecSetGhost(nu_p, 0);

      my_p4est_scft_t scft(ngbd, ns.val);

      scft.set_scaling(scaling);
      scft.set_polymer(f.val, XN.val);
      scft.add_boundary(phi, MLS_INT, gamma_A_cf, gamma_B_cf);
      scft.set_rho_avg(1);

      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      scft.initialize_solvers();
      scft.initialize_bc_smart(true);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol.val;
      while (scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val)
      {
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials(false, true);
        scft_iteration++;

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = fabs(scft.get_pressure_force());
      }

      scft.sync_and_extend();

      cost_function_nnn = scft.dsa_get_cost_function();
      VecCopyGhost(mu_p_tmp, shape_grad);

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, shape_grad, phi);

//      VecScaleGhost(shape_grad, -1);

      VecCopyGhost(mu_m_tmp, mu_m);
      VecCopyGhost(mu_p_tmp, mu_p);

      VecSetGhost(nu_m, 0);
      VecSetGhost(nu_p, 0);
    }
    else if (scheme.val == 0)
    {
      sample_cf_on_nodes(p4est, nodes, seed_cf, mu_m);
      VecSetGhost(mu_p, 0);
      VecSetGhost(shape_grad, 0);

      VecSetGhost(nu_m, 0);
      VecSetGhost(nu_p, 0);
    }

    // -------------------------------------------
    // compute full shape gradient
    // -------------------------------------------
    Vec surf_tns;

    ierr = VecDuplicate(phi, &surf_tns); CHKERRXX(ierr);

    double *mu_m_ptr;
    double *mu_t_ptr;
    double *nu_m_ptr;
    double *surf_tns_ptr;

    ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes) {
      surf_tns_ptr[n] = curvature_penalty.val + 1*(sqrt(XN_wall_avg.val+XN_wall_del.val) - sqrt(XN_wall_avg.val-XN_wall_del.val))*(mu_m_ptr[n] - mu_t_ptr[n] + nu_m_ptr[n])/XN.val;
    }

    ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);

    Vec shape_grad_full;
    Vec surf_tns_d[P4EST_DIM];
    Vec tmp;

    ierr = VecDuplicate(phi, &shape_grad_full); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &surf_tns_d[dim]); CHKERRXX(ierr); }

    ls.extend_from_interface_to_whole_domain_TVD(phi, surf_tns, tmp);
    VecPointwiseMultGhost(shape_grad_full, kappa, tmp);

    ngbd->first_derivatives_central(surf_tns, surf_tns_d);
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(surf_tns_d[dim], surf_tns_d[dim], normal[dim]);
      ls.extend_from_interface_to_whole_domain_TVD(phi, surf_tns_d[dim], tmp);
      VecAXPBYGhost(shape_grad_full, 1, 1, tmp);
    }

    VecScaleGhost(shape_grad_full, -1);
    VecAXPBYGhost(shape_grad_full, 1, 1, shape_grad);

    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, shape_grad_full, phi);

    ierr = VecDestroy(tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDestroy(surf_tns_d[dim]); CHKERRXX(ierr); }

    // -------------------------------------------
    // select velocity
    // -------------------------------------------
    Vec velo;
    Vec velo_full;
    ierr = VecDuplicate(phi, &velo); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &velo_full); CHKERRXX(ierr);

    if (minimize.val) {
      VecCopyGhost(shape_grad,      velo);
      VecCopyGhost(shape_grad_full, velo_full);
    } else {
      double *normal_ptr[P4EST_DIM];
      double *velo_ptr;

      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
      foreach_dimension(dim) {
        ierr = VecGetArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
      }

      foreach_node(n, nodes) {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        velo_ptr[n] = SUMD(normal_ptr[0][n]*vx_cf.value(xyz), normal_ptr[1][n]*vy_cf.value(xyz), TODO);
      }

      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
      foreach_dimension(dim) {
        ierr = VecRestoreArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
      }

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, velo, phi);

      ierr = VecCopyGhost(velo, velo_full); CHKERRXX(ierr);
    }

    // -------------------------------------------
    // compute time step dt
    // -------------------------------------------
    double dt = DBL_MAX;
    double vmax = 0;

    double *velo_ptr;
    double *velo_full_ptr;

    ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);


    foreach_local_node(n, nodes) {
      const quad_neighbor_nodes_of_node_t &qnnn = (*ngbd)[n];

      double dx_min = DBL_MAX;
      foreach_direction(dir) {
        dx_min = MIN(dx_min, fabs(qnnn.distance(dir)));
      }

      dt = MIN(dt, cfl()*fabs(dx_min/velo_ptr[n]));
      dt = MIN(dt, cfl()*fabs(dx_min/velo_full_ptr[n]));
      vmax = MAX(vmax, fabs(velo_ptr[n]));
      vmax = MAX(vmax, fabs(velo_full_ptr[n]));
    }

    ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

    MPI_Allreduce(MPI_IN_PLACE, &dt,   1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);

//    ierr = PetscPrintf(mpi.comm(), "Max velo: %e, Time step: %e\n", vmax, dt);

    // ------------------------------------------------------------------------------------------------------------------
    // compute expected change in energy due to free surface
    // ------------------------------------------------------------------------------------------------------------------
    Vec integrand;
    ierr = VecDuplicate(phi, &integrand); CHKERRXX(ierr);
    ierr = VecPointwiseMultGhost(integrand, velo_full, shape_grad_full); CHKERRXX(ierr);
    double energy_change_predicted = -dt*integration.integrate_over_interface(0, integrand);
    ierr = VecDestroy(integrand); CHKERRXX(ierr);

    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e, Predicted: %e\n", cost_function_nnn, cost_function_nnn-cost_function_nm1, energy_change_predicted);

    // -------------------------------------------
    // save data
    // -------------------------------------------
    if (save_data())
    {
      ierr = PetscFOpen  (mpi.comm(), file_conv_name, "a", &file_conv); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e\n", (int) round(iteration), cost_function_nnn, energy_change_predicted, cost_function_nnn-cost_function_nm1); CHKERRXX(ierr);
      ierr = PetscFClose (mpi.comm(), file_conv); CHKERRXX(ierr);
    }

    if (save_vtk.val && iteration%save_every_dn.val == 0)
    {
      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nx.valyztrees[2] <<
       #endif
             "." << (int) round(iteration/save_every_dn.val);

      PetscPrintf(mpi.comm(), "VTK is being saved in %s\n", oss.str().c_str());

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

      double *phi_ptr;
      double *kappa_ptr;
      double *surf_tns_ptr;
      double *mu_t_ptr;
      double *mu_m_ptr;
      double *mu_p_ptr;
      double *nu_m_ptr;
      double *nu_p_ptr;
      double *velo_ptr;
      double *velo_full_ptr;

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             10, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_t", mu_t_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "nu_m", nu_m_ptr,
                             VTK_POINT_DATA, "nu_p", nu_p_ptr,
                             VTK_POINT_DATA, "velo", velo_ptr,
                             VTK_POINT_DATA, "velo_full", velo_full_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    if (save_p4est.val && iteration%save_every_dn.val == 0) {
      std::ostringstream p4est_dir;
      p4est_dir << out_dir << "/p4est";

      std::ostringstream grid_file; grid_file << "grid_" << (int) round(iteration/save_every_dn.val);
      std::ostringstream vecs_file; vecs_file << "vecs_" << (int) round(iteration/save_every_dn.val);

      vector<Vec> vecs_to_save;
      vecs_to_save.push_back(phi);
      vecs_to_save.push_back(mu_t);
      vecs_to_save.push_back(mu_m);
      vecs_to_save.push_back(mu_p);

      my_p4est_save_forest_and_data(p4est_dir.str().c_str(), p4est, nodes, grid_file.str().c_str(), 1,
                                    vecs_file.str().c_str(), vecs_to_save.size(), vecs_to_save.data());
    }

    // -------------------------------------------
    // advect interface and impose contact angle
    // -------------------------------------------
    if (minimize.val) {
      ls.set_use_neumann_for_contact_angle(0);
      ls.set_contact_angle_extension(0);
      ls.advect_in_normal_direction_with_contact_angle(velo, surf_tns, NULL, NULL, phi, dt);
    } else {
      ls.advect_in_normal_direction(velo, phi, dt);
      vx_cf.t += dt;
      vy_cf.t += dt;
    }

    ls.reinitialize_2nd_order(phi, 50);

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    ierr = VecDestroy(velo_full); CHKERRXX(ierr);
    ierr = VecDestroy(shape_grad); CHKERRXX(ierr);
    ierr = VecDestroy(shape_grad_full); CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);

    /* refine and coarsen grid */
    {
      double *phi_ptr;
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin.val, lmax.val, lip.val);
      sp.set_refine_only_inside(1);

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      bool is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_ptr);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);

      if (is_grid_changing)
      {
        // repartition p4est
        my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // interpolate data between grids
        my_p4est_interpolation_nodes_t interp(ngbd);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_np1)
        {
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          interp.add_point(n, xyz);
        }

        interpolate_between_grids(interp, p4est_np1, nodes_np1, &phi, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &mu_m, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &mu_p, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &nu_m, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &nu_p, phi, interpolation_between_grids);

        // delete old p4est
        p4est_destroy(p4est);       p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;
        hierarchy->update(p4est, ghost);
        ngbd->update(hierarchy, nodes);
      }
    }
    iteration++;

    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    ierr = VecDestroy(mu_t); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}

