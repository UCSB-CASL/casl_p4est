
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
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_poisson_nodes_mls_sc.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/simplex3_mls_vtk.h>
#include <src/simplex3_mls_quadratic_vtk.h>
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
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes_mls_sc.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/simplex2_mls_vtk.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_tools_mls.h>
#endif

#include <src/point3.h>
#include <tools/plotting.h>

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


#undef MIN
#undef MAX

using namespace std;

bool save_vtk = 1;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int nb_splits = 3;
//int lmin = 7;
//int lmax = 7;
//int nb_splits = 1;
int nb_splits_per_split = 10;
#else
int lmin = 5;
int lmax = 5;
int nb_splits = 3;
int nb_splits_per_split = 20;
#endif

//const int periodic[3] = {1, 1, 1};
const int periodic[3] = {0, 0, 0};
const int n_xyz[3] = {1, 1, 1};
const double p_xyz_min[3] = {-1, -1, -1};
const double p_xyz_max[3] = {1, 1, 1};
//const double p_xyz_min[3] = {-2, -2, -2};
//const double p_xyz_max[3] = {2, 2, 2};

/* Examples for Poisson paper
 * 0000
 * 1100
 * 2211
 * 3310
 * 4412
 * 7412
 */

int n_geometry = 7;
int n_test = 0;
int n_mu = 0;
int n_diag_add = 0;
//int n_test = 4;
//int n_mu = 1;
//int n_diag_add = 2;

bool reinitialize_lsfs = 0;
bool plot_convergence = 1;
bool save_domain_reconstruction = 0;
bool do_extension = 0;

bool sc_scheme = 1;


// EXACT SOLUTION
#include "exact_solutions.h"

// Diffusion coefficient
#ifdef P4_TO_P8
class MU_CF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_mu){
      case 0: return 1.;
      case 1: return 1+(0.2*sin(x)+0.3*cos(y))*cos(z);
    }
  }
} mu_cf;
class MUX: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_mu)
    {
      case 0: return 0.;
      case 1: return 0.2*cos(x)*cos(z);
    }
  }
} mux_cf;
class MUY: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_mu)
    {
      case 0: return 0.;
      case 1: return -0.3*sin(y)*cos(z);
    }
  }
} muy_cf;
class MUZ: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_mu)
    {
      case 0: return 0.;
      case 1: return -(0.2*sin(x)+0.3*cos(y))*sin(z);
    }
  }
} muz_cf;
#else
class MU_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_mu){
      case 0: return 1.;
      case 1: return 1+0.2*sin(x)+0.3*cos(y);
    }
  }
} mu_cf;
class MUX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_mu){
      case 0: return 0.;
      case 1: return .2*cos(x);
    }
  }
} mux_cf;
class MUY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_mu){
      case 0: return 0.;
      case 1: return -0.3*sin(y);
    }
  }
} muy_cf;
#endif

// Diagonal term
#ifdef P4_TO_P8
class DIAG_ADD_CF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_diag_add)
    {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x+z)*exp(y);
    }
  }
} diag_add_cf;
#else
class DIAG_ADD_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_diag_add)
    {
      case 0: return 0;
      case 1: return 1.;
      case 2: return sin(x)*exp(y);
    }
  }
} diag_add_cf;
#endif

// RHS
#ifdef P4_TO_P8
class RHS: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -mu_cf(x,y,z)*lap_u_cf(x,y,z) + diag_add_cf(x,y,z)*u_cf(x,y,z)
                    - mux_cf(x,y,z)*ux_cf(x,y,z) - muy_cf(x,y,z)*uy_cf(x,y,z) - muz_cf(x,y,z)*uz_cf(x,y,z);
  }
} rhs_cf;
#else
class RHS: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -mu_cf(x,y)*lap_u_cf(x,y) + diag_add_cf(x,y)*u_cf(x,y)
                    - mux_cf(x,y)*ux_cf(x,y) - muy_cf(x,y)*uy_cf(x,y);
  }
} rhs_cf;
#endif

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
  CF_2 *phi_x, *phi_y;
  CF_2 *kappa;
  CF_2 *mu;
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


#ifdef P4_TO_P8
class bc_wall_type_t : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type;
#else
class bc_wall_type_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type;
#endif


vector<double> level, h;

vector<double> error_sl_arr, error_sl_l1_arr;
vector<double> error_ex_arr, error_ex_l1_arr;
vector<double> error_dd_arr, error_dd_l1_arr;
vector<double> error_tr_arr, error_tr_l1_arr;
vector<double> error_gr_arr, error_gr_l1_arr;
vector<double> error_ge_arr, error_ge_l1_arr;


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_trunc, Vec err_grad,
              int compt);

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
//  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
//  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("reinit", "reinitialize level-set function");
  cmd.add_option("n_test", "test function");
  cmd.add_option("n_geometry", "geometry");
  cmd.add_option("n_mu", "diffusion coefficient");
  cmd.add_option("n_diag_add", "additional diagonal term");
  cmd.add_option("plot_convergence", "show convergence plots");
  cmd.add_option("save_domain_reconstruction", "save reconstruction of domain (works only in serial!)");
  cmd.add_option("do_extension", "extend solution after solving");
  cmd.add_option("sc_scheme", "use super convergent scheme");
  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  reinitialize_lsfs = cmd.get("reinit", reinitialize_lsfs);
  n_test = cmd.get("n_test", n_test);
  n_geometry = cmd.get("n_geometry", n_geometry);
  n_mu = cmd.get("n_mu", n_mu);
  plot_convergence = cmd.get("plot_convergence", plot_convergence);
  save_domain_reconstruction = cmd.get("save_domain_reconstruction", save_domain_reconstruction);
  do_extension = cmd.get("do_extension", do_extension);
  sc_scheme = cmd.get("sc_scheme", sc_scheme);


//  bc_wtype = cmd.get("bc_wtype", bc_wtype);
//  bc_itype = cmd.get("bc_itype", bc_itype);

  save_vtk = cmd.get("save_vtk", save_vtk);

  set_parameters();

  parStopWatch w;
  w.start("total time");

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;


  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  // an effective LSF
  level_set_tot_t level_set_tot_cf(&phi_cf, &action, &color);

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    for (int sub_iter = 0; sub_iter < nb_splits_per_split; ++sub_iter)
    {

//      double p_xyz_min_alt[3];
//      double p_xyz_max_alt[3];
//      int l_inc = 0;

//      double dxyz[3] = { (p_xyz_max[0]-p_xyz_min[0])/pow(2., (double) lmax+iter),
//                         (p_xyz_max[1]-p_xyz_min[1])/pow(2., (double) lmax+iter),
//                         (p_xyz_max[2]-p_xyz_min[2])/pow(2., (double) lmax+iter) };

//  #ifdef P4_TO_P8
//        for (int k = 0; k < nb_splits_per_split; ++k)
//        {
//          p_xyz_min_alt[2] = p_xyz_min[2] + (double) (k) / (double) (nb_splits_per_split) * dxyz[2];
//          p_xyz_max_alt[2] = p_xyz_max[2] + (double) (k) / (double) (nb_splits_per_split) * dxyz[2];
//  #endif
////          for (int j = 0; j < 1; ++j)
//            for (int j = 0; j < nb_splits_per_split; ++j)
//          {
//            p_xyz_min_alt[1] = p_xyz_min[1] + (double) (j) / (double) (nb_splits_per_split) * dxyz[1];
//            p_xyz_max_alt[1] = p_xyz_max[1] + (double) (j) / (double) (nb_splits_per_split) * dxyz[1];
//            for (int i = 0; i < nb_splits_per_split; ++i)
//            {
//              p_xyz_min_alt[0] = p_xyz_min[0] + (double) (i) / (double) (nb_splits_per_split) * dxyz[0];
//              p_xyz_max_alt[0] = p_xyz_max[0] + (double) (i) / (double) (nb_splits_per_split) * dxyz[0];

//#ifdef P4_TO_P8
//        int sub_iter = iter*nb_splits_per_split*nb_splits_per_split*nb_splits_per_split + k*nb_splits_per_split*nb_splits_per_split + j*nb_splits_per_split+i;
//#else
//        int sub_iter = iter*nb_splits_per_split*nb_splits_per_split + j*nb_splits_per_split+i;
//#endif

      ierr = PetscPrintf(mpi.comm(), "\t Sub split %d \n", sub_iter); CHKERRXX(ierr);
      double p_xyz_min_alt[3];
      double p_xyz_max_alt[3];
      int l_inc = 0;

      if (sub_iter == 0)
      {
        p_xyz_min_alt[0] = p_xyz_min[0]; p_xyz_max_alt[0] = p_xyz_max[0];
        p_xyz_min_alt[1] = p_xyz_min[1]; p_xyz_max_alt[1] = p_xyz_max[1];
        p_xyz_min_alt[2] = p_xyz_min[2]; p_xyz_max_alt[2] = p_xyz_max[2];
      } else {
        l_inc = 1;
        double scale = (double) (nb_splits_per_split-sub_iter) / (double) nb_splits_per_split;
        p_xyz_min_alt[0] = p_xyz_min[0] - .5*(pow(2.,scale)-1)*(p_xyz_max[0]-p_xyz_min[0]); p_xyz_max_alt[0] = p_xyz_max[0] + .5*(pow(2.,scale)-1)*(p_xyz_max[0]-p_xyz_min[0]);
        p_xyz_min_alt[1] = p_xyz_min[1] - .5*(pow(2.,scale)-1)*(p_xyz_max[1]-p_xyz_min[1]); p_xyz_max_alt[1] = p_xyz_max[1] + .5*(pow(2.,scale)-1)*(p_xyz_max[1]-p_xyz_min[1]);
        p_xyz_min_alt[2] = p_xyz_min[2] - .5*(pow(2.,scale)-1)*(p_xyz_max[2]-p_xyz_min[2]); p_xyz_max_alt[2] = p_xyz_max[2] + .5*(pow(2.,scale)-1)*(p_xyz_max[2]-p_xyz_min[2]);
      }


      connectivity = my_p4est_brick_new(n_xyz, p_xyz_min_alt, p_xyz_max_alt, &brick, periodic);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    splitting_criteria_cf_t data_tmp(lmin, lmax, &level_set_tot_cf, 1.4);
    p4est->user_pointer = (void*)(&data_tmp);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    for (int i = 0; i < iter+l_inc; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }

    splitting_criteria_cf_t data(lmin+iter+l_inc, lmax+iter+l_inc, &level_set_tot_cf, 1.4);
    p4est->user_pointer = (void*)(&data);

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
    double diag = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);
#else
    double dxyz_max = MAX(dxyz[0], dxyz[1]);
    double diag = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1]);
#endif

    int num_surfaces = phi_cf.size();

    // sample level-set functions
    std::vector<Vec> phi;
    for (int i = 0; i < num_surfaces; i++)
    {
      phi.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *phi_cf[i], phi.back());
      if (reinitialize_lsfs)
        ls.reinitialize_1st_order_time_2nd_order_space(phi.back(),20);
    }

        std::vector<BoundaryConditionType> bc_interface_type(num_surfaces, ROBIN);
//        std::vector<BoundaryConditionType> bc_interface_type(num_surfaces, DIRICHLET);
    //    std::vector<BoundaryConditionType> bc_interface_type(num_surfaces, NEUMANN);

    // sample boundary conditions
#ifdef P4_TO_P8
    std::vector<bc_value_robin_t *> bc_interface_value_(num_surfaces, NULL);
    std::vector<CF_3 *> bc_interface_value(num_surfaces, NULL);
#else
    std::vector<bc_value_robin_t *> bc_interface_value_(num_surfaces, NULL);
    std::vector<CF_2 *> bc_interface_value(num_surfaces, NULL);
#endif
    for (int i = 0; i < num_surfaces; i++)
    {
      if (bc_interface_type[i] == ROBIN || bc_interface_type[i] == NEUMANN)
      {
#ifdef P4_TO_P8
        bc_interface_value_[i] = new bc_value_robin_t(&u_cf, &ux_cf, &uy_cf, &uz_cf, &mu_cf, phi_x_cf[i], phi_y_cf[i], phi_z_cf[i], bc_coeffs_cf[i]);
#else
        bc_interface_value_[i] = new bc_value_robin_t(&u_cf, &ux_cf, &uy_cf, &mu_cf, phi_x_cf[i], phi_y_cf[i], bc_coeffs_cf[i]);
#endif
        bc_interface_value[i] = bc_interface_value_[i];
      } else {
        bc_interface_value[i] = &u_cf;
      }
    }

    Vec rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);

    Vec u_exact_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_cf, u_exact_vec);

    Vec mu;
    ierr = VecCreateGhostNodes(p4est, nodes, &mu); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_cf, mu);

    Vec diag_add;
    ierr = VecCreateGhostNodes(p4est, nodes, &diag_add); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, diag_add_cf, diag_add);


//    ierr = VecDestroy(rhs); CHKERRXX(ierr);
//    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Starting a solver\n"); CHKERRXX(ierr);

    Vec sol; double *sol_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
    std::vector<Vec> *phi_dd[P4EST_DIM];

    Vec phi_eff ;
    Vec mask;

    Mat A;
    std::vector<double> *scalling;
    Vec volumes;

    my_p4est_poisson_nodes_mls_sc_t solver_sc(&ngbd_n);
    my_p4est_poisson_nodes_mls_t    solver(&ngbd_n);

    if (sc_scheme)
    {
      solver_sc.set_geometry(num_surfaces, &action, &color, &phi);
      solver_sc.set_mu(mu);
      solver_sc.set_rhs(rhs);

      solver_sc.set_bc_wall_value(u_cf);
      solver_sc.set_bc_wall_type(bc_wall_type);
      solver_sc.set_bc_interface_type(bc_interface_type);
      solver_sc.set_bc_interface_coeff(bc_coeffs_cf);
      solver_sc.set_bc_interface_value(bc_interface_value);

      solver_sc.set_diag_add(diag_add);

      solver_sc.set_use_taylor_correction(1);
      solver_sc.set_keep_scalling(true);
      solver_sc.set_kink_treatment(1);

      solver_sc.solve(sol);

      solver_sc.get_phi_dd(phi_dd);

      phi_eff   = solver_sc.get_phi_eff();
      mask      = solver_sc.get_mask();
      A         = solver_sc.get_matrix();
      scalling  = solver_sc.get_scalling();
      volumes   = solver_sc.get_volumes();

    } else {

      solver.set_geometry(num_surfaces, &action, &color, &phi);
      solver.set_mu(mu);
      solver.set_rhs(rhs);

      solver.set_bc_wall_value(u_cf);
      solver.set_bc_wall_type(bc_wall_type);
      solver.set_bc_interface_type(bc_interface_type);
      solver.set_bc_interface_coeff(bc_coeffs_cf);
      solver.set_bc_interface_value(bc_interface_value);

      solver.set_diag_add(diag_add);

      solver.set_use_taylor_correction(1);
      solver.set_keep_scalling(true);
      solver.set_kink_treatment(1);

      solver.solve(sol);

      solver.get_phi_dd(phi_dd);

      phi_eff  = solver.get_phi_eff();
      mask     = solver.get_mask();
      A        = solver.get_matrix();
      scalling = solver.get_scalling();
      volumes  = solver.get_volumes();
    }


    my_p4est_integration_mls_t integrator(p4est, nodes);
#ifdef P4_TO_P8
    integrator.set_phi(phi, action, color);
#else
    integrator.set_phi(phi, action, color);
#endif
    if (save_vtk && save_domain_reconstruction)
    {
      const char* out_dir = getenv("OUT_DIR");
      if (!out_dir)
      {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
        return -1;
      }
      std::ostringstream command;
      command << "mkdir -p " << out_dir;
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0)
        throw std::invalid_argument("could not create directory");

      integrator.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integrator.cubes_linear.size(); k++)
        if (integrator.cubes_linear[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integrator.cubes_linear[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(out_dir), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(out_dir), to_string(iter));
#endif
    }

//#ifdef P4_TO_P8
//    integrator.set_phi(phi, *phi_dd[0], *phi_dd[1], *phi_dd[2], action, color);
//#else
//    integrator.set_phi(phi, *phi_dd[0], *phi_dd[1], action, color);
//#endif
//    if (save_vtk)
//    {
//      integrator.initialize();
//#ifdef P4_TO_P8
//      vector<simplex3_mls_quadratic_t *> simplices;
//      int n_sps = NTETS;
//#else
//      vector<simplex2_mls_t *> simplices;
//      int n_sps = 2;
//#endif

//      for (int k = 0; k < integrator.cubes_quadratic.size(); k++)
//        if (integrator.cubes_quadratic[k].loc == FCE)
//          for (int l = 0; l < n_sps; l++)
//            simplices.push_back(&integrator.cubes_quadratic[k].simplex[l]);

//#ifdef P4_TO_P8
//      simplex3_mls_quadratic_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#else
//      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#endif
//    }

    /* calculate errors */
    Vec vec_error_sl; double *vec_error_sl_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl); CHKERRXX(ierr);
    Vec vec_error_tr; double *vec_error_tr_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_tr); CHKERRXX(ierr);
    Vec vec_error_gr; double *vec_error_gr_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr); CHKERRXX(ierr);
    Vec vec_error_ex; double *vec_error_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex); CHKERRXX(ierr);
    Vec vec_error_dd; double *vec_error_dd_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd); CHKERRXX(ierr);
    Vec vec_error_ge; double *vec_error_ge_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ge); CHKERRXX(ierr);

    //----------------------------------------------------------------------------------------------
    // calculate error of solution
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);

    double *mask_ptr;
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (mask_ptr[n] < 0)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
  #ifdef P4_TO_P8
        vec_error_sl_ptr[n] = ABS(sol_ptr[n] - u_cf(xyz[0],xyz[1],xyz[2]));
  #else
        vec_error_sl_ptr[n] = ABS(sol_ptr[n] - u_cf(xyz[0],xyz[1]));
  #endif
      }
      else
        vec_error_sl_ptr[n] = 0;
    }

    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_sl, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_sl, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate error of gradients
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

#ifdef P4_TO_P8
      double ux_exact = ux_cf(xyz[0], xyz[1], xyz[2]);
      double uy_exact = uy_cf(xyz[0], xyz[1], xyz[2]);
      double uz_exact = uz_cf(xyz[0], xyz[1], xyz[2]);
#else
      double ux_exact = ux_cf(xyz[0], xyz[1]);
      double uy_exact = uy_cf(xyz[0], xyz[1]);
#endif

      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
      ngbd_n.get_neighbors(n, qnnn);
      if ( mask_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
           ( mask_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( mask_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( mask_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( mask_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( mask_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( mask_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        double ux_error = fabs(qnnn.dx_central(sol_ptr) - ux_exact);
        double uy_error = fabs(qnnn.dy_central(sol_ptr) - uy_exact);
#ifdef P4_TO_P8
        double uz_error = fabs(qnnn.dz_central(sol_ptr) - uz_exact);
        vec_error_gr_ptr[n] = sqrt(SQR(ux_error) + SQR(uy_error) + SQR(uz_error));
#else
        vec_error_gr_ptr[n] = sqrt(SQR(ux_error) + SQR(uy_error));
#endif
      } else {
        vec_error_gr_ptr[n] = 0;
      }
    }

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_gr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_gr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate truncation error
    //----------------------------------------------------------------------------------------------
//    sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);
//    solver.set_rhs(rhs);
//    solver.assemble_rhs_only();
    Vec vec_u_exact; ierr = VecCreateGhostNodes(p4est, nodes, &vec_u_exact);   CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, u_cf, vec_u_exact);

    ierr = MatMult(A, vec_u_exact, vec_error_tr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);
    double *rhs_ptr;
    ierr = VecGetArray(rhs, &rhs_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
    {
      if (mask_ptr[n] < 0)
      {
        vec_error_tr_ptr[n] = (vec_error_tr_ptr[n] - rhs_ptr[n])*scalling->at(n);
      } else {
        vec_error_tr_ptr[n] = 0;
      }
    }

    ierr = VecRestoreArray(vec_error_tr, &vec_error_tr_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_tr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_tr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(vec_u_exact); CHKERRXX(ierr);

    //----------------------------------------------------------------------------------------------
    // calculate extrapolation error
    //----------------------------------------------------------------------------------------------
    // smoothed LSF
//    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 25.*dxyz_max*dxyz_max);
    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 64.*dxyz_max*dxyz_max);
//    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 0.1*dxyz_max);
//    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 0.0025);

    Vec phi_smooth;
    double *phi_smooth_ptr;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_smooth); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_smooth_cf, phi_smooth);

    if (do_extension)
    {
      ls.reinitialize_1st_order_time_2nd_order_space(phi_smooth);
      ls.reinitialize_1st_order_time_2nd_order_space(phi_eff);
    }

    double band = 3.0;

    // copy solution into a new Vec
    Vec sol_ex; double *sol_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_ex); CHKERRXX(ierr);

    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
      sol_ex_ptr[i] = sol_ptr[i];

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

    // extend
//    ls.extend_Over_Interface_TVD(phi_smooth, sol_ex, 100); CHKERRXX(ierr);
//    ls.extend_Over_Interface_TVD(phi_eff, sol_ex, 100); CHKERRXX(ierr);
    if (do_extension)
      ls.extend_Over_Interface_TVD(phi_smooth, mask, sol_ex, 100, 2); CHKERRXX(ierr);
//    ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, sol_ex, 100); CHKERRXX(ierr);

    // calculate error
    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

    double *phi_eff_ptr;
    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
//      if (mask_ptr[n] > 0. && phi_smooth_ptr[n] < band*dxyz_max)
      if (mask_ptr[n] > 0. && phi_eff_ptr[n] < band*dxyz_max)
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
    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    //----------------------------------------------------------------------------------------------
    // calculate error of Laplacian
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

#ifdef P4_TO_P8
      double udd_exact = lap_u_cf(xyz[0], xyz[1], xyz[2]);
#else
      double udd_exact = lap_u_cf(xyz[0], xyz[1]);
#endif

      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
      ngbd_n.get_neighbors(n, qnnn);
      if ( mask_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
           ( mask_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( mask_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( mask_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( mask_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( mask_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( mask_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        double uxx = qnnn.dxx_central(sol_ptr);
        double uyy = qnnn.dyy_central(sol_ptr);
#ifdef P4_TO_P8
        double uzz = qnnn.dzz_central(sol_ptr);
        vec_error_dd_ptr[n] = fabs(udd_exact-uxx-uyy-uzz);
#else
        vec_error_dd_ptr[n] = fabs(udd_exact-uxx-uyy);
#endif
      } else {
        vec_error_dd_ptr[n] = 0;
      }
    }

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    Vec vec_uxx; double *vec_uxx_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uxx); CHKERRXX(ierr);
//    Vec vec_uyy; double *vec_uyy_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uyy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    Vec vec_uzz; double *vec_uzz_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uzz); CHKERRXX(ierr);
//    ngbd_n.second_derivatives_central(sol_ex, vec_uxx, vec_uyy, vec_uzz); CHKERRXX(ierr);
//#else
//    ngbd_n.second_derivatives_central(sol_ex, vec_uxx, vec_uyy); CHKERRXX(ierr);
//#endif

////    double phi_shift = 0*diag;

////    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
////    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
////    {
////      phi_eff_ptr[i] += phi_shift;
////    }
////    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

//////    ls.extend_Over_Interface_TVD(phi_smooth, vec_uxx); CHKERRXX(ierr);
//////    ls.extend_Over_Interface_TVD(phi_smooth, vec_uyy); CHKERRXX(ierr);
//////#ifdef P4_TO_P8
//////    ls.extend_Over_Interface_TVD(phi_smooth, vec_uzz); CHKERRXX(ierr);
//////#endif

////    ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_uxx, 20); CHKERRXX(ierr);
////    ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_uyy, 20); CHKERRXX(ierr);
////#ifdef P4_TO_P8
////    ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_uzz, 20); CHKERRXX(ierr);
////#endif

////    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
////    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
////    {
////      phi_eff_ptr[i] -= phi_shift;
////    }
////    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

//    ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);
//    ierr = VecGetArray(vec_uxx, &vec_uxx_ptr); CHKERRXX(ierr);
//    ierr = VecGetArray(vec_uyy, &vec_uyy_ptr); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecGetArray(vec_uzz, &vec_uzz_ptr); CHKERRXX(ierr);
//#endif

//    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

//    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//    {
//      if (mask_ptr[n] < 0)
//      {
//        double xyz[P4EST_DIM];
//        node_xyz_fr_n(n, p4est, nodes, xyz);
//#ifdef P4_TO_P8
//        vec_error_dd_ptr[n] = ABS(vec_uxx_ptr[n] + vec_uyy_ptr[n] + vec_uzz_ptr[n] - lap_u_cf(xyz[0],xyz[1],xyz[2]));
//#else
//        vec_error_dd_ptr[n] = ABS(vec_uxx_ptr[n] + vec_uyy_ptr[n] - lap_u_cf(xyz[0],xyz[1]));
//#endif
//      }
//      else
//        vec_error_dd_ptr[n] = 0;
//    }

//    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

//    ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);
//    ierr = VecRestoreArray(vec_uxx, &vec_uxx_ptr); CHKERRXX(ierr);
//    ierr = VecRestoreArray(vec_uyy, &vec_uyy_ptr); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecRestoreArray(vec_uzz, &vec_uzz_ptr); CHKERRXX(ierr);
//#endif

//    ierr = VecDestroy(vec_uxx); CHKERRXX(ierr);
//    ierr = VecDestroy(vec_uyy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecDestroy(vec_uzz); CHKERRXX(ierr);
//#endif

//    ierr = VecGhostUpdateBegin(vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd  (vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    //----------------------------------------------------------------------------------------------
//    // calculate error of gradient after extension
//    //----------------------------------------------------------------------------------------------
//    {
//      Vec vec_ux; double *vec_ux_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_ux); CHKERRXX(ierr);
//      Vec vec_uy; double *vec_uy_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      Vec vec_uz; double *vec_uz_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uz); CHKERRXX(ierr);
//      Vec vec_u_d[P4EST_DIM] = { vec_ux, vec_uy, vec_uz };
//#else
//      Vec vec_u_d[P4EST_DIM] = { vec_ux, vec_uy };
//#endif
//      ngbd_n.first_derivatives_central(sol_ex, vec_u_d); CHKERRXX(ierr);

//      double phi_shift = diag;

//      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
//      for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
//      {
//        phi_eff_ptr[i] += phi_shift;
//      }
//      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

//      //    ls.extend_Over_Interface_TVD(phi_smooth, vec_uxx); CHKERRXX(ierr);
//      //    ls.extend_Over_Interface_TVD(phi_smooth, vec_uyy); CHKERRXX(ierr);
//      //#ifdef P4_TO_P8
//      //    ls.extend_Over_Interface_TVD(phi_smooth, vec_uzz); CHKERRXX(ierr);
//      //#endif

//      ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_ux, 100); CHKERRXX(ierr);
//      ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_uy, 100); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ls.extend_Over_Interface_TVD(phi_smooth, phi_eff, vec_uz, 100); CHKERRXX(ierr);
//#endif

//      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
//      for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
//      {
//        phi_eff_ptr[i] -= phi_shift;
//      }
//      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

//      ierr = VecGetArray(vec_error_ge, &vec_error_ge_ptr); CHKERRXX(ierr);
//      ierr = VecGetArray(vec_ux, &vec_ux_ptr); CHKERRXX(ierr);
//      ierr = VecGetArray(vec_uy, &vec_uy_ptr); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecGetArray(vec_uz, &vec_uz_ptr); CHKERRXX(ierr);
//#endif

//      ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

//      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//      {
//        if (mask_ptr[n] < 0)
//        {
//          double xyz[P4EST_DIM];
//          node_xyz_fr_n(n, p4est, nodes, xyz);

//#ifdef P4_TO_P8
//          double ux_exact = ux_cf(xyz[0], xyz[1], xyz[2]);
//          double uy_exact = uy_cf(xyz[0], xyz[1], xyz[2]);
//          double uz_exact = uz_cf(xyz[0], xyz[1], xyz[2]);
//#else
//          double ux_exact = ux_cf(xyz[0], xyz[1]);
//          double uy_exact = uy_cf(xyz[0], xyz[1]);
//#endif

//#ifdef P4_TO_P8
//          vec_error_ge_ptr[n] = sqrt(SQR(vec_ux_ptr[n] - ux_exact) + SQR(vec_uy_ptr[n] - uy_exact) + SQR(vec_uz_ptr[n] - uz_exact));
//#else
//          vec_error_ge_ptr[n] = sqrt(SQR(vec_ux_ptr[n] - ux_exact) + SQR(vec_uy_ptr[n] - uy_exact));
//#endif
//        }
//        else
//          vec_error_ge_ptr[n] = 0;
//      }

//      ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

//      ierr = VecRestoreArray(vec_error_ge, &vec_error_ge_ptr); CHKERRXX(ierr);
//      ierr = VecRestoreArray(vec_ux, &vec_ux_ptr); CHKERRXX(ierr);
//      ierr = VecRestoreArray(vec_uy, &vec_uy_ptr); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecRestoreArray(vec_uz, &vec_uz_ptr); CHKERRXX(ierr);
//#endif

//      ierr = VecDestroy(vec_ux); CHKERRXX(ierr);
//      ierr = VecDestroy(vec_uy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecDestroy(vec_uz); CHKERRXX(ierr);
//#endif

//      ierr = VecGhostUpdateBegin(vec_error_ge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd  (vec_error_ge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

    // compute L-inf norm of errors
    double err_sl_max = 0.; VecMax(vec_error_sl, NULL, &err_sl_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_tr_max = 0.; VecMax(vec_error_tr, NULL, &err_tr_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_gr_max = 0.; VecMax(vec_error_gr, NULL, &err_gr_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_ex_max = 0.; VecMax(vec_error_ex, NULL, &err_ex_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_dd_max = 0.; VecMax(vec_error_dd, NULL, &err_dd_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_ge_max = 0.; VecMax(vec_error_ge, NULL, &err_ge_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ge_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    // compute L1 errors
    double measure_of_dom = integrator.measure_of_domain();
    error_sl_l1_arr.push_back(integrator.integrate_everywhere(vec_error_sl)/measure_of_dom);
    error_tr_l1_arr.push_back(integrator.integrate_everywhere(vec_error_tr)/measure_of_dom);
    error_gr_l1_arr.push_back(integrator.integrate_everywhere(vec_error_gr)/measure_of_dom);
    error_ex_l1_arr.push_back(integrator.integrate_everywhere(vec_error_ex)/measure_of_dom);
    error_dd_l1_arr.push_back(integrator.integrate_everywhere(vec_error_dd)/measure_of_dom);
    error_ge_l1_arr.push_back(integrator.integrate_everywhere(vec_error_ge)/measure_of_dom);

    // Store error values
    level.push_back(lmin+iter);
    h.push_back(dxyz_max*pow(2.,(double) data.max_lvl - data.min_lvl));

//#ifdef P4_TO_P8
//        h.push_back(iter*nb_splits_per_split*nb_splits_per_split*nb_splits_per_split + k*nb_splits_per_split*nb_splits_per_split + j*nb_splits_per_split+i);
//#else
//        h.push_back(iter*nb_splits_per_split*nb_splits_per_split + j*nb_splits_per_split+i);
//#endif

    error_sl_arr.push_back(err_sl_max);
    error_tr_arr.push_back(err_tr_max);
    error_gr_arr.push_back(err_gr_max);
    error_ex_arr.push_back(err_ex_max);
    error_dd_arr.push_back(err_dd_max);
    error_ge_arr.push_back(err_ge_max);

    // Print current errors
    if (iter > -1)
    {
      int step = iter*nb_splits_per_split + sub_iter;
      ierr = PetscPrintf(p4est->mpicomm, "Error (sl): %g, order = %g\n", err_sl_max, log(error_sl_arr[step-1]/error_sl_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (tr): %g, order = %g\n", err_tr_max, log(error_tr_arr[step-1]/error_tr_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (gr): %g, order = %g\n", err_gr_max, log(error_gr_arr[step-1]/error_gr_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (ex): %g, order = %g\n", err_ex_max, log(error_ex_arr[step-1]/error_ex_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (dd): %g, order = %g\n", err_dd_max, log(error_dd_arr[step-1]/error_dd_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (ge): %g, order = %g\n", err_ge_max, log(error_ge_arr[step-1]/error_ge_arr[step])/log(h[step-1]/h[step])); CHKERRXX(ierr);
    }

    if(save_vtk)
    {
//#ifdef STAMPEDE
//      char *out_dir;
//      out_dir = getenv("OUT_DIR");
//#else
//      char out_dir[10000];
//      sprintf(out_dir, OUTPUT_DIR);
//#endif

      const char* out_dir = getenv("OUT_DIR");
      if (!out_dir)
      {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
        return -1;
      }
      std::ostringstream command;
      command << "mkdir -p " << out_dir << "/vtu";
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0)
        throw std::invalid_argument("could not create directory");

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << p4est->mpisize << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
//             "." << iter;
             "." << iter*nb_splits_per_split + sub_iter;

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
      ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

      double *mask_ptr;
      ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

      double *volumes_ptr;
      ierr = VecGetArray(volumes, &volumes_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             11, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_eff_ptr,
                             VTK_POINT_DATA, "phi_smooth", phi_smooth_ptr,
                             VTK_POINT_DATA, "sol", sol_ptr,
                             VTK_POINT_DATA, "sol_ex", sol_ex_ptr,
                             VTK_POINT_DATA, "error_sl", vec_error_sl_ptr,
                             VTK_POINT_DATA, "error_tr", vec_error_tr_ptr,
                             VTK_POINT_DATA, "error_gr", vec_error_gr_ptr,
                             VTK_POINT_DATA, "error_ex", vec_error_ex_ptr,
                             VTK_POINT_DATA, "error_dd", vec_error_dd_ptr,
                             VTK_POINT_DATA, "mask", mask_ptr,
                             VTK_POINT_DATA, "volumes", volumes_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(volumes, &volumes_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    // destroy Vec's with errors
    ierr = VecDestroy(vec_error_sl); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_tr); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_gr); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_ex); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_dd); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_ge); CHKERRXX(ierr);

    ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);
    ierr = VecDestroy(sol_ex); CHKERRXX(ierr);

    for (int i = 0; i < phi.size(); i++)
    {
      ierr = VecDestroy(phi[i]);        CHKERRXX(ierr);
    }


    for (int i = 0; i < num_surfaces; i++)
    {
      if (bc_interface_type[i] == ROBIN || bc_interface_type[i] == NEUMANN)
      {
        delete bc_interface_value_[i];
      }
    }

    ierr = VecDestroy(sol);         CHKERRXX(ierr);
    ierr = VecDestroy(mu);          CHKERRXX(ierr);
    ierr = VecDestroy(rhs);         CHKERRXX(ierr);
    ierr = VecDestroy(diag_add);    CHKERRXX(ierr);
    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

//            }
//          }
//  #ifdef P4_TO_P8
//        }
//  #endif
    }
  }



  w.stop(); w.read_duration();

  if (mpi.rank() == 0 && plot_convergence)
  {
    Gnuplot graph;

    print_Table("Convergence", 0.0, level, h, "err sl", error_sl_arr, 1, &graph);
    print_Table("Convergence", 0.0, level, h, "err gr", error_gr_arr, 2, &graph);
    print_Table("Convergence", 0.0, level, h, "err tr", error_tr_arr, 3, &graph);
    print_Table("Convergence", 0.0, level, h, "err dd", error_dd_arr, 4, &graph);

//    print_Table("Convergence", 0.0, level, h, "err ex (max)", error_ex_arr,     4, &graph);

    cin.get();
  }


  if (mpi.rank() == 0)
  {
    // print all errors in compact form for plotting in matlab
    cout << "h";    for (int i = 0; i < h.size(); i++) { cout << ", " << h[i]; }   cout <<  ";" << endl;

    cout << "sl";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_sl_arr[i]); }   cout <<  ";" << endl;
    cout << "ex";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_ex_arr[i]); }   cout <<  ";" << endl;
    cout << "gr";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_gr_arr[i]); }   cout <<  ";" << endl;
    cout << "dd";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_dd_arr[i]); }   cout <<  ";" << endl;
    cout << "tr";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_tr_arr[i]); }   cout <<  ";" << endl;
  }

  return 0;
}
