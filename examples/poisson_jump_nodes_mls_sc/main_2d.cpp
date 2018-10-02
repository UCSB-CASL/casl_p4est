
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
#include <src/my_p8est_poisson_jump_nodes_mls_sc.h>
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
#include <src/my_p4est_poisson_jump_nodes_mls_sc.h>
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
#include "problem_case_10.h" // angle
#include "problem_case_11.h" // one circle

#undef MIN
#undef MAX

using namespace std;

bool save_vtk = 1;
bool reinitialize_lsfs = 0;

// Arthur's examples
// 11, 0, 0, 9, 8
// 11, 1, 0, 6, 7
// 11, 0, 1, 6, 7
// 6,  4, 5, 5, 11
// 11, 4, 5, 5, 6

int num_test_geometry = 0;

int num_test_mu_m = 5;
int num_test_mu_p = 4;

int num_test_um = 0;
int num_test_up = 1;

int num_test_diag_term_m = 0;
int num_test_diag_term_p = 0;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int nb_splits = 5;
int nb_splits_per_split = 1;
#else
int lmin = 4;
int lmax = 4;
int nb_splits = 6;
int nb_splits_per_split = 1;
#endif

bool use_sc_scheme = 1;
bool scale_errors = 1;
bool compute_grad_between = 1;

//const int periodic[3] = {1, 1, 1};
const int periodic[3] = {0, 0, 0};
const int n_xyz[3] = {1, 1, 1};
const double p_xyz_min[3] = {-1.0, -1.0, -1.0};
const double p_xyz_max[3] = { 1.0,  1.0,  1.0};


// DIFFUSION COEFFICIENTS
#include "diffusion_coeffs.h"

// EXACT SOLUTIONS
#include "exact_solutions.h"

// DIAGONAL TERMS
#include "diag_terms.h"

// RHS
#ifdef P4_TO_P8
class rhs_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -mu_p_cf(x,y,z)*lap_u_p_cf(x,y,z) + diag_term_p_cf(x,y,z)*u_p_cf(x,y,z)
                    - mux_p_cf(x,y,z)*ux_p_cf(x,y,z) - muy_p_cf(x,y,z)*uy_p_cf(x,y,z) - muz_p_cf(x,y,z)*uz_p_cf(x,y,z);
  }
} rhs_p_cf;
#else
class rhs_p_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -mu_p_cf(x,y)*lap_u_p_cf(x,y) + diag_term_p_cf(x,y)*u_p_cf(x,y)
                    - mux_p_cf(x,y)*ux_p_cf(x,y) - muy_p_cf(x,y)*uy_p_cf(x,y);
  }
} rhs_p_cf;
#endif

#ifdef P4_TO_P8
class rhs_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -mu_m_cf(x,y,z)*lap_u_m_cf(x,y,z) + diag_term_m_cf(x,y,z)*u_m_cf(x,y,z)
                    - mux_m_cf(x,y,z)*ux_m_cf(x,y,z) - muy_m_cf(x,y,z)*uy_m_cf(x,y,z) - muz_m_cf(x,y,z)*uz_m_cf(x,y,z);
  }
} rhs_m_cf;
#else
class rhs_m_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -mu_m_cf(x,y)*lap_u_m_cf(x,y) + diag_term_m_cf(x,y)*u_m_cf(x,y)
                    - mux_m_cf(x,y)*ux_m_cf(x,y) - muy_m_cf(x,y)*uy_m_cf(x,y);
  }
} rhs_m_cf;
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
problem_case_11_t problem_case_11;

void set_parameters()
{
  switch (num_test_geometry)
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

// JUMP CONDITIONS
#ifdef P4_TO_P8
class u_jump_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return u_p_cf(x,y,z) - u_m_cf(x,y,z);
  }
} u_jump_cf;
#else
class u_jump_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return 0;
    return u_p_cf(x,y) - u_m_cf(x,y);
  }
} u_jump_cf;
#endif

#ifdef P4_TO_P8
class mu_un_jump_t : public CF_3
{
  CF_3 *phi_x_, *phi_y_, *phi_z_;
public:
  mu_un_jump_t(CF_3 *phi_x, CF_3 *phi_y, CF_3 *phi_z) :
    phi_x_(phi_x), phi_y_(phi_y), phi_z_(phi_z) {}

  double operator()(double x, double y, double z) const
  {
    double nx = (*phi_x_)(x,y,z);
    double ny = (*phi_y_)(x,y,z);
    double nz = (*phi_z_)(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    nx /= norm; ny /= norm; nz /= norm;

    return mu_p_cf(x,y,z) * (ux_p_cf(x,y,z)*nx + uy_p_cf(x,y,z)*ny + uz_p_cf(x,y,z)*nz)
        -  mu_m_cf(x,y,z) * (ux_m_cf(x,y,z)*nx + uy_m_cf(x,y,z)*ny + uz_m_cf(x,y,z)*nz);
  }
};
#else
class mu_un_jump_t : public CF_2
{
  CF_2 *phi_x_, *phi_y_;
public:
  mu_un_jump_t(CF_2 *phi_x, CF_2 *phi_y) :
    phi_x_(phi_x), phi_y_(phi_y) {}

  double operator()(double x, double y) const
  {
    double nx = (*phi_x_)(x,y);
    double ny = (*phi_y_)(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;

//    return 2;
    return mu_p_cf(x,y) * (ux_p_cf(x,y)*nx + uy_p_cf(x,y)*ny)
        -  mu_m_cf(x,y) * (ux_m_cf(x,y)*nx + uy_m_cf(x,y)*ny);
  }
};

#endif

#ifdef P4_TO_P8
class u_cf_t : public CF_3
{
  CF_3 * phi_;
public:
  void set_phi(CF_3 &phi) { phi_ = &phi; }
  double operator()(double x, double y, double z) const
  {
    return (*phi_)(x,y,z) > 0 ? u_p_cf(x,y,z) : u_m_cf(x,y,z);
  }
} u_cf;
#else
class u_cf_t : public CF_2
{
  CF_2 * phi_;
public:
  void set_phi(CF_2 &phi) { phi_ = &phi; }
  double operator()(double x, double y) const
  {
    return (*phi_)(x,y) > 0 ? u_p_cf(x,y) : u_m_cf(x,y);
  }
} u_cf;
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

vector<double> error_sl_m_arr;
vector<double> error_ex_m_arr;
vector<double> error_dd_m_arr;
vector<double> error_gr_m_arr;
vector<double> error_tr_m_arr;

vector<double> error_sl_p_arr;
vector<double> error_ex_p_arr;
vector<double> error_dd_p_arr;
vector<double> error_gr_p_arr;
vector<double> error_tr_p_arr;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

//  cmdParser cmd;
//  cmd.add_option("lmin", "min level of the tree");
//  cmd.add_option("lmax", "max level of the tree");
//  cmd.add_option("nb_splits", "number of recursive splits");
//  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
//  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
//  cmd.add_option("save_vtk", "save the p4est in vtk format");
//#ifdef P4_TO_P8
//  cmd.add_option("test", "choose a test.\n\
//                 0 - x+y+z\n\
//                 1 - x*x + y*y + z*z\n\
//                 2 - sin(x)*cos(y)*exp(z)");
//#else
//  cmd.add_option("test", "choose a test.\n\
//                 0 - x+y\n\
//                 1 - x*x + y*y\n\
//                 2 - sin(x)*cos(y)");
//#endif
//  cmd.parse(argc, argv);

//  cmd.print();

//  lmin = cmd.get("lmin", lmin);
//  lmax = cmd.get("lmax", lmax);
//  nb_splits = cmd.get("nb_splits", nb_splits);
//  test_number = cmd.get("test", test_number);

//  bc_wtype = cmd.get("bc_wtype", bc_wtype);
//  bc_itype = cmd.get("bc_itype", bc_itype);

//  save_vtk = cmd.get("save_vtk", save_vtk);

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
  u_cf.set_phi(level_set_tot_cf);

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    for (int sub_iter = 0; sub_iter < nb_splits_per_split; ++sub_iter)
    {
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

    splitting_criteria_cf_t data_tmp(lmin, lmax, &level_set_tot_cf, 1.2);
    p4est->user_pointer = (void*)(&data_tmp);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    for (int i = 0; i < iter+l_inc; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }

    splitting_criteria_cf_t data(lmin+iter+l_inc, lmax+iter+l_inc, &level_set_tot_cf, 1.2);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_partition(p4est, P4EST_FALSE, NULL);
//    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
//    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
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


    // sample boundary conditions
#ifdef P4_TO_P8
    std::vector<mu_un_jump_t *> mu_un_jump(num_surfaces, NULL);
    std::vector<CF_3 *> mu_un_jump_cf(num_surfaces, NULL);
#else
    std::vector<mu_un_jump_t *> mu_un_jump(num_surfaces, NULL);
    std::vector<CF_2 *> mu_un_jump_cf(num_surfaces, NULL);
#endif
    for (int i = 0; i < num_surfaces; i++)
    {
#ifdef P4_TO_P8
      mu_un_jump[i] = new mu_un_jump_t(phi_x_cf[i], phi_y_cf[i], phi_z_cf[i]);
#else
      mu_un_jump[i] = new mu_un_jump_t(phi_x_cf[i], phi_y_cf[i]);
#endif
      mu_un_jump_cf[i] = mu_un_jump[i];
    }

    Vec rhs_m;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_m); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, rhs_m_cf, rhs_m);

    Vec rhs_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_p); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, rhs_p_cf, rhs_p);

//    Vec u_exact_vec;
//    ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est, nodes, u_cf, u_exact_vec);

    Vec mu_m;
    ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);

    Vec mu_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

    Vec diag_term_m;
    ierr = VecCreateGhostNodes(p4est, nodes, &diag_term_m); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, diag_term_m_cf, diag_term_m);

    Vec diag_term_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &diag_term_p); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, diag_term_p_cf, diag_term_p);

    ierr = PetscPrintf(p4est->mpicomm, "Starting a solver\n"); CHKERRXX(ierr);

    my_p4est_poisson_jump_nodes_mls_sc_t solver(&ngbd_n);
    solver.set_geometry(num_surfaces, &action, &color, &phi);
    solver.set_mu(mu_m, mu_p);
    solver.set_rhs(rhs_m, rhs_p);

    solver.set_bc_wall_value(u_cf);
    solver.set_bc_wall_type(bc_wall_type);

    solver.set_jumps(u_jump_cf, mu_un_jump_cf);

    solver.set_diag_add(diag_term_m, diag_term_p);

    solver.set_keep_scalling(true);
    solver.set_use_sc_scheme(use_sc_scheme);

//    solver.compute_volumes();

//    ierr = PetscPrintf(p4est->mpicomm, "Here\n"); CHKERRXX(ierr);

    Vec sol_m; double *sol_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_m); CHKERRXX(ierr);
    Vec sol_p; double *sol_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_p); CHKERRXX(ierr);

    solver.solve(sol_m, sol_p);

    std::vector<Vec> *phi_dd[P4EST_DIM];
    solver.get_phi_dd(phi_dd);

    Vec phi_eff = solver.get_phi_eff(); double *phi_eff_ptr;

    Vec mask_m = solver.get_mask_m();
    Vec mask_p = solver.get_mask_p();

    my_p4est_integration_mls_t integrator(p4est, nodes);
#ifdef P4_TO_P8
    integrator.set_phi(phi, action, color);
#else
    integrator.set_phi(phi, action, color);
#endif
//    if (save_vtk)
//    {
//      integrator.initialize();
//#ifdef P4_TO_P8
//      vector<simplex3_mls_t *> simplices;
//      int n_sps = NTETS;
//#else
//      vector<simplex2_mls_t *> simplices;
//      int n_sps = 2;
//#endif

//      for (int k = 0; k < integrator.cubes_linear.size(); k++)
//        if (integrator.cubes_linear[k].loc == FCE)
//          for (int l = 0; l < n_sps; l++)
//            simplices.push_back(&integrator.cubes_linear[k].simplex[l]);

//#ifdef P4_TO_P8
//      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#else
//      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#endif
//    }

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
    Vec vec_error_sl_m; double *vec_error_sl_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl_m); CHKERRXX(ierr);
    Vec vec_error_gr_m; double *vec_error_gr_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr_m); CHKERRXX(ierr);
    Vec vec_error_ex_m; double *vec_error_ex_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex_m); CHKERRXX(ierr);
    Vec vec_error_dd_m; double *vec_error_dd_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd_m); CHKERRXX(ierr);
    Vec vec_error_tr_m; double *vec_error_tr_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_tr_m); CHKERRXX(ierr);

    Vec vec_error_sl_p; double *vec_error_sl_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl_p); CHKERRXX(ierr);
    Vec vec_error_gr_p; double *vec_error_gr_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr_p); CHKERRXX(ierr);
    Vec vec_error_ex_p; double *vec_error_ex_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex_p); CHKERRXX(ierr);
    Vec vec_error_dd_p; double *vec_error_dd_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd_p); CHKERRXX(ierr);
    Vec vec_error_tr_p; double *vec_error_tr_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_tr_p); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate error of solution
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);

    double *mask_m_ptr;   ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    double *mask_p_ptr;   ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    double u_max = 0;

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      vec_error_sl_m_ptr[n] = mask_m_ptr[n] < 0 ? ABS(sol_m_ptr[n] - u_m_cf.value(xyz)) : 0;
      vec_error_sl_p_ptr[n] = mask_p_ptr[n] < 0 ? ABS(sol_p_ptr[n] - u_p_cf.value(xyz)) : 0;

      u_max = MAX(u_max, fabs(u_m_cf.value(xyz)), fabs(u_p_cf.value(xyz)));
    }

    ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_sl_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_sl_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_sl_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_sl_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate error of gradients
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;

    double gr_max = 0;

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

      if (!compute_grad_between)
      {
        ngbd_n.get_neighbors(n, qnnn);

        if ( mask_m_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
             ( mask_m_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( mask_m_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_m_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( mask_m_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
             ( mask_m_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
             ( mask_m_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( mask_m_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( mask_m_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )
        {
#ifdef P4_TO_P8
          double ux_m_exact = ux_m_cf(xyz[0], xyz[1], xyz[2]);
          double uy_m_exact = uy_m_cf(xyz[0], xyz[1], xyz[2]);
          double uz_m_exact = uz_m_cf(xyz[0], xyz[1], xyz[2]);
          gr_max = MAX(gr_max, sqrt(SQR(ux_m_exact) + SQR(uy_m_exact)+SQR(uz_m_exact)));
#else
          double ux_m_exact = ux_m_cf(xyz[0], xyz[1]);
          double uy_m_exact = uy_m_cf(xyz[0], xyz[1]);
          gr_max = MAX(gr_max, sqrt(SQR(ux_m_exact) + SQR(uy_m_exact)));
#endif
          double ux_m_error = fabs(qnnn.dx_central(sol_m_ptr) - ux_m_exact);
          double uy_m_error = fabs(qnnn.dy_central(sol_m_ptr) - uy_m_exact);
#ifdef P4_TO_P8
          double uz_m_error = fabs(qnnn.dz_central(sol_m_ptr) - uz_m_exact);
          vec_error_gr_m_ptr[n] = sqrt(SQR(ux_m_error) + SQR(uy_m_error) + SQR(uz_m_error));
#else
          vec_error_gr_m_ptr[n] = sqrt(SQR(ux_m_error) + SQR(uy_m_error));
#endif
        } else {
          vec_error_gr_m_ptr[n] = 0;
        }


        if ( mask_p_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
             ( mask_p_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( mask_p_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( mask_p_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
             ( mask_p_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
             ( mask_p_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( mask_p_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( mask_p_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )
        {
#ifdef P4_TO_P8
          double ux_p_exact = ux_p_cf(xyz[0], xyz[1], xyz[2]);
          double uy_p_exact = uy_p_cf(xyz[0], xyz[1], xyz[2]);
          double uz_p_exact = uz_p_cf(xyz[0], xyz[1], xyz[2]);
          gr_max = MAX(gr_max, sqrt(SQR(ux_p_exact) + SQR(uy_p_exact)+SQR(uz_p_exact)));
#else
          double ux_p_exact = ux_p_cf(xyz[0], xyz[1]);
          double uy_p_exact = uy_p_cf(xyz[0], xyz[1]);
          gr_max = MAX(gr_max, sqrt(SQR(ux_p_exact) + SQR(uy_p_exact)));
#endif
          double ux_p_error = fabs(qnnn.dx_central(sol_p_ptr) - ux_p_exact);
          double uy_p_error = fabs(qnnn.dy_central(sol_p_ptr) - uy_p_exact);
#ifdef P4_TO_P8
          double uz_p_error = fabs(qnnn.dz_central(sol_p_ptr) - uz_p_exact);
          vec_error_gr_p_ptr[n] = sqrt(SQR(ux_p_error) + SQR(uy_p_error) + SQR(uz_p_error));
#else
          vec_error_gr_p_ptr[n] = sqrt(SQR(ux_p_error) + SQR(uy_p_error));
#endif
        } else {
          vec_error_gr_p_ptr[n] = 0;
        }

      } else {
        p4est_locidx_t neighbors      [(int)pow(3, P4EST_DIM)];
        bool           neighbors_exist[(int)pow(3, P4EST_DIM)];

        double xyz_nei[P4EST_DIM];
        double xyz_mid[P4EST_DIM];
        double normal[P4EST_DIM];

        vec_error_gr_m_ptr[n] = 0;
        vec_error_gr_p_ptr[n] = 0;

        if (!is_node_Wall(p4est, ni))
        {
          solver.get_all_neighbors_(n, neighbors, neighbors_exist);
//          for (int j = 0; j < (int)pow(3, P4EST_DIM); ++j)
          for (int j = 1; j < (int)pow(3, P4EST_DIM); j+=2)
          {
            p4est_locidx_t n_nei = neighbors[j];
            double delta = 0;
            node_xyz_fr_n(n_nei, p4est, nodes, xyz_nei);

            for (int i = 0; i < P4EST_DIM; ++i)
            {
              xyz_mid[i] = .5*(xyz[i]+xyz_nei[i]);
              delta += SQR(xyz[i]-xyz_nei[i]);
              normal[i] = xyz_nei[i]-xyz[i];
            }
            delta = sqrt(delta);

            for (int i = 0; i < P4EST_DIM; ++i)
              normal[i] /= delta;

            if (mask_m_ptr[n] < 0)
              if (mask_m_ptr[n_nei] < 0)
              {
#ifdef P4_TO_P8
                double grad_exact = ux_m_cf.value(xyz_mid)*normal[0] + uy_m_cf.value(xyz_mid)*normal[1] + uz_m_cf.value(xyz_mid)*normal[2];
#else
                double grad_exact = ux_m_cf.value(xyz_mid)*normal[0] + uy_m_cf.value(xyz_mid)*normal[1];
#endif
                vec_error_gr_m_ptr[n] = MAX(vec_error_gr_m_ptr[n], fabs((sol_m_ptr[n_nei]-sol_m_ptr[n])/delta - grad_exact));

                gr_max = MAX(gr_max, fabs(grad_exact));
              }

            if (mask_p_ptr[n] < 0)
              if (mask_p_ptr[n_nei] < 0)
              {
#ifdef P4_TO_P8
                double grad_exact = ux_p_cf.value(xyz_mid)*normal[0] + uy_p_cf.value(xyz_mid)*normal[1] + uz_p_cf.value(xyz_mid)*normal[2];
#else
                double grad_exact = ux_p_cf.value(xyz_mid)*normal[0] + uy_p_cf.value(xyz_mid)*normal[1];
#endif
                vec_error_gr_p_ptr[n] = MAX(vec_error_gr_p_ptr[n], fabs((sol_p_ptr[n_nei]-sol_p_ptr[n])/delta - grad_exact));

                gr_max = MAX(gr_max, fabs(grad_exact));
              }
          }
        }
      }

    }

    ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_gr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_gr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_gr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_gr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate error of Laplacian
    //----------------------------------------------------------------------------------------------
    ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
      ngbd_n.get_neighbors(n, qnnn);

      if ( mask_m_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
           ( mask_m_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_m_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_m_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_m_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_m_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_m_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_m_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_m_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        double udd_exact = lap_u_m_cf.value(xyz);

        double uxx = qnnn.dxx_central(sol_m_ptr);
        double uyy = qnnn.dyy_central(sol_m_ptr);
#ifdef P4_TO_P8
        double uzz = qnnn.dzz_central(sol_m_ptr);
        vec_error_dd_m_ptr[n] = fabs(udd_exact-uxx-uyy-uzz);
#else
        vec_error_dd_m_ptr[n] = fabs(udd_exact-uxx-uyy);
#endif
      } else {
        vec_error_dd_m_ptr[n] = 0;
      }

      if ( mask_p_ptr[qnnn.node_000]<-EPS && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
           ( mask_p_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_p_ptr[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p_ptr[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_p_ptr[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_p_ptr[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_p_ptr[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_p_ptr[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_p_ptr[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        double udd_exact = lap_u_p_cf.value(xyz);

        double uxx = qnnn.dxx_central(sol_p_ptr);
        double uyy = qnnn.dyy_central(sol_p_ptr);
#ifdef P4_TO_P8
        double uzz = qnnn.dzz_central(sol_p_ptr);
        vec_error_dd_p_ptr[n] = fabs(udd_exact-uxx-uyy-uzz);
#else
        vec_error_dd_p_ptr[n] = fabs(udd_exact-uxx-uyy);
#endif
      } else {
        vec_error_dd_p_ptr[n] = 0;
      }
    }

    ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_dd_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_dd_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_dd_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_dd_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    //----------------------------------------------------------------------------------------------
    // calculate truncation error
    //----------------------------------------------------------------------------------------------
//    sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);
//    solver.set_rhs(rhs);
//    solver.assemble_rhs_only();

    Mat A = solver.get_matrix();
    std::vector<double> *scalling = solver.get_scalling();

    Vec vec_u_exact_block;    double *vec_u_exact_block_ptr;    ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &vec_u_exact_block);    CHKERRXX(ierr);
    Vec vec_error_tr_block;   double *vec_error_tr_block_ptr;   ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &vec_error_tr_block);   CHKERRXX(ierr);

    ierr = VecGetArray(vec_u_exact_block, &vec_u_exact_block_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(i, p4est, nodes, xyz);

      vec_u_exact_block_ptr[2*i  ] = mask_m_ptr[i] < 0 ? u_m_cf.value(xyz) : 0;
      vec_u_exact_block_ptr[2*i+1] = mask_p_ptr[i] < 0 ? u_p_cf.value(xyz) : 0;
    }

    ierr = VecRestoreArray(vec_u_exact_block, &vec_u_exact_block_ptr); CHKERRXX(ierr);

    ierr = MatMult(A, vec_u_exact_block, vec_error_tr_block); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_tr_block, &vec_error_tr_block_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_tr_m,     &vec_error_tr_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_tr_p,     &vec_error_tr_p_ptr); CHKERRXX(ierr);

    Vec rhs = solver.get_rhs();
    double *rhs_ptr;
    ierr = VecGetArray(rhs, &rhs_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
    {
      vec_error_tr_m_ptr[n] = (mask_m_ptr[n] < 0) ? (vec_error_tr_block_ptr[2*n  ] - rhs_ptr[2*n  ])*scalling->at(2*n  ) : 0;
      vec_error_tr_p_ptr[n] = (mask_p_ptr[n] < 0) ? (vec_error_tr_block_ptr[2*n+1] - rhs_ptr[2*n+1])*scalling->at(2*n+1) : 0;
    }

    ierr = VecRestoreArray(vec_error_tr_block, &vec_error_tr_block_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_tr_m,     &vec_error_tr_m_ptr);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_tr_p,     &vec_error_tr_p_ptr);  CHKERRXX(ierr);

    ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(rhs, &rhs_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_tr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_tr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateEnd  (vec_error_tr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_tr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(vec_u_exact_block); CHKERRXX(ierr);

    //----------------------------------------------------------------------------------------------
    // calculate extrapolation error
    //----------------------------------------------------------------------------------------------
    // smoothed LSF
    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 64.*dxyz_max*dxyz_max);

    Vec phi_smooth;   double *phi_smooth_ptr;   ierr = VecCreateGhostNodes(p4est, nodes, &phi_smooth); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, level_set_smooth_cf, phi_smooth);
//    ls.reinitialize_1st_order_time_2nd_order_space(phi_smooth);
//    ls.reinitialize_1st_order_time_2nd_order_space(phi_eff);

    double band = 3.0;

    // copy solution into a new Vec
    Vec sol_m_ex; double *sol_m_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_m_ex); CHKERRXX(ierr);
    Vec sol_p_ex; double *sol_p_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_p_ex); CHKERRXX(ierr);

    ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      sol_m_ex_ptr[i] = sol_m_ptr[i];
      sol_p_ex_ptr[i] = sol_p_ptr[i];
    }

    ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

    // extend
//    ls.extend_Over_Interface_TVD(phi_smooth, mask_m, sol_m_ex, 20, 2); CHKERRXX(ierr);

    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      phi_smooth_ptr[i] *= -1.;
    }
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

//    ls.extend_Over_Interface_TVD(phi_smooth, mask_p, sol_p_ex, 20, 2); CHKERRXX(ierr);

    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      phi_smooth_ptr[i] *= -1.;
    }
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    // calculate error
    ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      vec_error_ex_m_ptr[n] = (mask_m_ptr[n] > 0. && phi_eff_ptr[n] < band*dxyz_max) ? ABS(sol_m_ex_ptr[n] - u_m_cf.value(xyz)) : 0;
      vec_error_ex_p_ptr[n] = (mask_p_ptr[n] > 0. && phi_eff_ptr[n] >-band*dxyz_max) ? ABS(sol_p_ex_ptr[n] - u_p_cf.value(xyz)) : 0;
    }

    ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_ex_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_ex_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateEnd  (vec_error_ex_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_ex_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);








    // compute L-inf norm of errors
    double err_sl_m_max = 0.;   ierr = VecMax(vec_error_sl_m, NULL, &err_sl_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_tr_m_max = 0.;   ierr = VecMax(vec_error_tr_m, NULL, &err_tr_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tr_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_gr_m_max = 0.;   ierr = VecMax(vec_error_gr_m, NULL, &err_gr_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_ex_m_max = 0.;   ierr = VecMax(vec_error_ex_m, NULL, &err_ex_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_dd_m_max = 0.;   ierr = VecMax(vec_error_dd_m, NULL, &err_dd_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    double err_sl_p_max = 0.;   ierr = VecMax(vec_error_sl_p, NULL, &err_sl_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_tr_p_max = 0.;   ierr = VecMax(vec_error_tr_p, NULL, &err_tr_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tr_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_gr_p_max = 0.;   ierr = VecMax(vec_error_gr_p, NULL, &err_gr_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_ex_p_max = 0.;   ierr = VecMax(vec_error_ex_p, NULL, &err_ex_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_dd_p_max = 0.;   ierr = VecMax(vec_error_dd_p, NULL, &err_dd_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    if (scale_errors)
    {
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      err_sl_m_max /= u_max;
      err_sl_p_max /= u_max;

      mpiret = MPI_Allreduce(MPI_IN_PLACE, &gr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      err_gr_m_max /= gr_max;
      err_gr_p_max /= gr_max;
    }

    // Store error values
    level.push_back(lmin+iter);
    h.push_back(dxyz_max*pow(2.,(double) data.max_lvl - data.min_lvl));

    error_sl_m_arr.push_back(err_sl_m_max);
    error_tr_m_arr.push_back(err_tr_m_max);
    error_gr_m_arr.push_back(err_gr_m_max);
    error_ex_m_arr.push_back(err_ex_m_max);
    error_dd_m_arr.push_back(err_dd_m_max);

    error_sl_p_arr.push_back(err_sl_p_max);
    error_tr_p_arr.push_back(err_tr_p_max);
    error_gr_p_arr.push_back(err_gr_p_max);
    error_ex_p_arr.push_back(err_ex_p_max);
    error_dd_p_arr.push_back(err_dd_p_max);

    // Print current errors
    if (iter > -1)
    {
      ierr = PetscPrintf(p4est->mpicomm, "Error (sl_m): %g, order = %g\n", err_sl_m_max, log(error_sl_m_arr[iter-1]/error_sl_m_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (tr_m): %g, order = %g\n", err_tr_m_max, log(error_tr_m_arr[iter-1]/error_tr_m_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (gr_m): %g, order = %g\n", err_gr_m_max, log(error_gr_m_arr[iter-1]/error_gr_m_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (ex_m): %g, order = %g\n", err_ex_m_max, log(error_ex_m_arr[iter-1]/error_ex_m_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (dd_m): %g, order = %g\n", err_dd_m_max, log(error_dd_m_arr[iter-1]/error_dd_m_arr[iter])/log(2)); CHKERRXX(ierr);

      ierr = PetscPrintf(p4est->mpicomm, "Error (sl_p): %g, order = %g\n", err_sl_p_max, log(error_sl_p_arr[iter-1]/error_sl_p_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (tr_p): %g, order = %g\n", err_tr_p_max, log(error_tr_p_arr[iter-1]/error_tr_p_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (gr_p): %g, order = %g\n", err_gr_p_max, log(error_gr_p_arr[iter-1]/error_gr_p_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (ex_p): %g, order = %g\n", err_ex_p_max, log(error_ex_p_arr[iter-1]/error_ex_p_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (dd_p): %g, order = %g\n", err_dd_p_max, log(error_dd_p_arr[iter-1]/error_dd_p_arr[iter])/log(2)); CHKERRXX(ierr);
    }

    if(save_vtk)
    {
#ifdef STAMPEDE
      char *out_dir;
      out_dir = getenv("OUT_DIR");
#else
      char out_dir[10000];
      sprintf(out_dir, OUTPUT_DIR);
#endif

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << p4est->mpisize << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
             "." << iter;

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

      ierr = VecGetArray(phi_eff,    &phi_eff_ptr);    CHKERRXX(ierr);
      ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(sol_m,    &sol_m_ptr);    CHKERRXX(ierr);
      ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(sol_p,    &sol_p_ptr);    CHKERRXX(ierr);
      ierr = VecGetArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_tr_m, &vec_error_tr_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_tr_p, &vec_error_tr_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             18, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_eff_ptr,
                             VTK_POINT_DATA, "phi_smooth", phi_smooth_ptr,
                             VTK_POINT_DATA, "sol_m", sol_m_ptr,
                             VTK_POINT_DATA, "sol_p", sol_p_ptr,
                             VTK_POINT_DATA, "sol_m_ex", sol_m_ex_ptr,
                             VTK_POINT_DATA, "sol_p_ex", sol_p_ex_ptr,
                             VTK_POINT_DATA, "error_sl_m", vec_error_sl_m_ptr,
                             VTK_POINT_DATA, "error_tr_m", vec_error_tr_m_ptr,
                             VTK_POINT_DATA, "error_gr_m", vec_error_gr_m_ptr,
                             VTK_POINT_DATA, "error_ex_m", vec_error_ex_m_ptr,
                             VTK_POINT_DATA, "error_dd_m", vec_error_dd_m_ptr,
                             VTK_POINT_DATA, "error_sl_p", vec_error_sl_p_ptr,
                             VTK_POINT_DATA, "error_tr_p", vec_error_tr_p_ptr,
                             VTK_POINT_DATA, "error_gr_p", vec_error_gr_p_ptr,
                             VTK_POINT_DATA, "error_ex_p", vec_error_ex_p_ptr,
                             VTK_POINT_DATA, "error_dd_p", vec_error_dd_p_ptr,
                             VTK_POINT_DATA, "mask_m", mask_m_ptr,
                             VTK_POINT_DATA, "mask_p", mask_p_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_eff,    &phi_eff_ptr);    CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol_m,    &sol_m_ptr);    CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol_p,    &sol_p_ptr);    CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_tr_m, &vec_error_tr_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_tr_p, &vec_error_tr_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    // destroy Vec's with errors
    ierr = VecDestroy(vec_error_sl_m); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_tr_m); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_gr_m); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_ex_m); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_dd_m); CHKERRXX(ierr);

    ierr = VecDestroy(vec_error_sl_p); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_tr_p); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_gr_p); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_ex_p); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_dd_p); CHKERRXX(ierr);

    ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);

    ierr = VecDestroy(sol_m_ex); CHKERRXX(ierr);
    ierr = VecDestroy(sol_p_ex); CHKERRXX(ierr);

    for (int i = 0; i < phi.size(); i++)
    {
      ierr = VecDestroy(phi[i]);        CHKERRXX(ierr);
    }

    for (int i = 0; i < num_surfaces; i++)
    {
      delete mu_un_jump[i];
    }

    ierr = VecDestroy(sol_m);         CHKERRXX(ierr);
    ierr = VecDestroy(sol_p);         CHKERRXX(ierr);

    ierr = VecDestroy(mu_m);          CHKERRXX(ierr);
    ierr = VecDestroy(mu_p);          CHKERRXX(ierr);

    ierr = VecDestroy(rhs_m);         CHKERRXX(ierr);
    ierr = VecDestroy(rhs_p);         CHKERRXX(ierr);

    ierr = VecDestroy(diag_term_m);   CHKERRXX(ierr);
    ierr = VecDestroy(diag_term_p);   CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(connectivity, &brick);
    }
  }

  w.stop(); w.read_duration();

  if (mpi.rank() == 0)
  {
    Gnuplot graph_sl;
    Gnuplot graph_gr;
    Gnuplot graph_dd;

    print_Table("Solution", 0.0, level, h, "err_sl_m", error_sl_m_arr, 1, &graph_sl);
    print_Table("Solution", 0.0, level, h, "err_sl_p", error_sl_p_arr, 2, &graph_sl);
//    print_Table("Solution", 0.0, level, h, "err_ex_m", error_ex_m_arr, 2, &graph_sl);
//    print_Table("Solution", 0.0, level, h, "err_ex_p", error_ex_p_arr, 3, &graph_sl);

    print_Table("Gradient", 0.0, level, h, "err_gr_m", error_gr_m_arr, 1, &graph_gr);
    print_Table("Gradient", 0.0, level, h, "err_gr_p", error_gr_p_arr, 2, &graph_gr);

    print_Table("Second derivatives", 0.0, level, h, "err_dd_m", error_dd_m_arr, 1, &graph_dd);
    print_Table("Second derivatives", 0.0, level, h, "err_dd_p", error_dd_p_arr, 2, &graph_dd);
    print_Table("Second derivatives", 0.0, level, h, "err_tr_m", error_tr_m_arr, 1, &graph_dd);
    print_Table("Second derivatives", 0.0, level, h, "err_tr_p", error_tr_p_arr, 2, &graph_dd);

    // print all errors in compact form for plotting in matlab
    cout << "h";      for (int i = 0; i < h.size(); i++) { cout << ", " << h[i]; }   cout <<  ";" << endl;

    cout << "sl_m";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_sl_m_arr[i]); }   cout <<  ";" << endl;
    cout << "ex_m";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_ex_m_arr[i]); }   cout <<  ";" << endl;
    cout << "gr_m";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_gr_m_arr[i]); }   cout <<  ";" << endl;
    cout << "dd_m";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_dd_m_arr[i]); }   cout <<  ";" << endl;
    cout << "tr_m";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_tr_m_arr[i]); }   cout <<  ";" << endl;

    cout << "sl_p";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_sl_p_arr[i]); }   cout <<  ";" << endl;
    cout << "ex_p";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_ex_p_arr[i]); }   cout <<  ";" << endl;
    cout << "gr_p";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_gr_p_arr[i]); }   cout <<  ";" << endl;
    cout << "dd_p";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_dd_p_arr[i]); }   cout <<  ";" << endl;
    cout << "tr_p";   for (int i = 0; i < h.size(); i++) { cout << ", " << fabs(error_tr_p_arr[i]); }   cout <<  ";" << endl;

    cin.get();
  }

  return 0;
}
