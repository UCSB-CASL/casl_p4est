
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
#include <src/my_p8est_poisson_nodes_mls_sc.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_tools_mls.h>
#include <src/my_p8est_macros.h>
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
#include <src/my_p4est_poisson_nodes_mls_sc.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_tools_mls.h>
#include <src/my_p4est_macros.h>
#endif

#include <engine.h>

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
#include "problem_case_10.h" // angle
#include "problem_case_11.h" // one circle

#include "ii_problem_case_0.h" // triangle (tetrahedron)
#include "ii_problem_case_1.h" // two circles union
#include "ii_problem_case_2.h" // two circles intersection
#include "ii_problem_case_3.h" // two circles coloration
#include "ii_problem_case_4.h" // four flowers
#include "ii_problem_case_5.h" // two circles coloration (naive)
#include "ii_problem_case_6.h" // one flower
#include "ii_problem_case_7.h" // three flowers
#include "ii_problem_case_8.h" // half-space
#include "ii_problem_case_9.h" // angle
#include "ii_problem_case_10.h" // angle
#include "ii_problem_case_11.h" // one circle

#undef MIN
#undef MAX

#define ADD_OPTION(i, var, description) \
  i == 0 ? cmd.add_option(#var, description) : (void) (var = cmd.get(#var, var));

using namespace std;

//-------------------------------------
// computational domain parameters
//-------------------------------------
const int periodicity[3] = {0, 0, 0};
const int num_trees[3]   = {1, 1, 1};
const double grid_xyz_min[3] = {-1.5, -1.5, -1.5};
const double grid_xyz_max[3] = { 1.5,  1.5,  1.5};

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int num_splits = 5;
int num_splits_per_split = 1;
int num_shifts_x_dir = 1;
int num_shifts_y_dir = 1;
int num_shifts_z_dir = 1;
#else
int lmin = 5;
int lmax = 5;
int num_splits = 7;
int num_splits_per_split = 1;
int num_shifts_x_dir = 1;
int num_shifts_y_dir = 1;
int num_shifts_z_dir = 1;
#endif

int num_shifts_total = num_shifts_x_dir*
                       num_shifts_y_dir*
                       num_shifts_z_dir;

int num_resolutions = (num_splits-1)*num_splits_per_split + 1;
int num_iter_total  = num_resolutions*num_shifts_total;

int iter_start = 0; // is used to skip iterations and get to a problematic case

//-------------------------------------
// test solutions
//-------------------------------------

// Arthur's examples
// 11, 0, 0, 9, 8
// 11, 1, 0, 6, 7
// 11, 0, 1, 6, 7
// 6,  4, 5, 5, 11
// 11, 4, 5, 5, 6

int n_ii = 11;
int n_geometry = 6;

int n_mu_m = 0;
int n_mu_p = 2;

int n_um = 0;
int n_up = 1;

int n_diag_m = 0;
int n_diag_p = 0;

BoundaryConditionType bc_wtype = DIRICHLET;
BoundaryConditionType bc_itype = DIRICHLET;
//BoundaryConditionType bc_itype = ROBIN;

int jump_scheme = 0;

//-------------------------------------
// solver parameters
//-------------------------------------
int  integration_order = 2;
bool sc_scheme         = 0;

// for symmetric scheme:
bool taylor_correction      = 1;
bool kink_special_treatment = 1;

// for superconvergent scheme:
bool try_remove_hanging_cells = 0;

//-------------------------------------
// level-set representation parameters
//-------------------------------------
bool use_phi_cf       = 0;
bool reinit_level_set = 0;

// artificial perturbation of level-set values
int    domain_perturbation     = 0; // 0 - no, 1 - smooth, 2 - noisy
double domain_perturbation_mag = 0.1;
double domain_perturbation_pow = 2.;

//-------------------------------------
// convergence study parameters
//-------------------------------------
int    compute_cond_num = 0*num_splits;
bool   do_extension     = 0;
double mask_thresh      = 0;
bool   compute_grad_between = false;

//-------------------------------------
// output
//-------------------------------------
bool save_vtk           = 1;
bool save_domain        = 0;
bool save_matrix_ascii  = 0;
bool save_matrix_binary = 0;
bool save_convergence   = 0;

// DIFFUSION COEFFICIENTS
double mu_m_mag = 1;
double mu_p_mag = 1;

class mu_all_cf_t : public CF_DIM
{
public:
  int    *n;
  double *mag;
  enum value_type { VAL, DDX, DDY, DDZ, LAP } what;
  mu_all_cf_t(value_type what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: switch (what) {
          case VAL: return (*mag);
          case DDX: 0;
          case DDY: 0;
#ifdef P4_TO_P8
          case DDZ: 0;
#endif
        }
      case 1: switch (what) {
          case VAL: return  (*mag)*( 1. + (0.2*cos(x)+0.3*sin(y)) P8(*sin(z)) );
          case DDX: return -0.2*(*mag)*sin(x) P8(*sin(z));
          case DDY: return  0.3*(*mag)*cos(y) P8(*sin(z));
#ifdef P4_TO_P8
          case DDZ: return  (*mag)*(0.2*sin(x)+0.3*cos(y))*cos(z);
#endif
        }
      case 2: switch (what) {
          case VAL: return (*mag)*(y*y*log(x+2.) + 4.);
          case DDX: return (*mag)*y*y/(x+2.);
          case DDY: return (*mag)*2.*y*log(x+2.);
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
        }
      case 3: switch (what) {
          case VAL: return  (*mag)*(exp(-x));
          case DDX: return -(*mag)*(exp(-x));
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
        }
    }
  }
};

mu_all_cf_t mu_m_cf(mu_all_cf_t::VAL, n_mu_m, mu_m_mag), mu_p_cf(mu_all_cf_t::VAL, n_mu_p, mu_p_mag);
mu_all_cf_t mux_m_cf(mu_all_cf_t::DDX, n_mu_m, mu_m_mag), mux_p_cf(mu_all_cf_t::DDX, n_mu_p, mu_p_mag);
mu_all_cf_t muy_m_cf(mu_all_cf_t::DDY, n_mu_m, mu_m_mag), muy_p_cf(mu_all_cf_t::DDY, n_mu_p, mu_p_mag);
#ifdef P4_TO_P8
mu_all_cf_t muz_m_cf(mu_all_cf_t::DDZ, n_mu_m, mu_m_mag), muz_p_cf(mu_all_cf_t::DDZ, n_mu_p, mu_p_mag);
#endif

// EXACT SOLUTIONS
double mag_um = 1;
double mag_up = 1;
class u_pm_cf_t : public CF_DIM
{
public:
  int    *n;
  double *mag;
  enum value_type { VAL, DDX, DDY, DDZ, LAP } what;
  u_pm_cf_t(value_type what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(x)*cos(y)*exp(z);
          case DDX: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDY: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDZ: return  (*mag)*sin(x)*cos(y)*exp(z);
          case LAP: return -(*mag)*sin(x)*cos(y);
#else
          case VAL: return  (*mag)*sin(x)*cos(y);
          case DDX: return  (*mag)*cos(x)*cos(y);
          case DDY: return -(*mag)*sin(x)*sin(y);
          case LAP: return -(*mag)*2.*sin(x)*cos(y);
#endif
        }
      case 1: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*cos(x)*sin(y)*exp(z);
          case DDX: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDY: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDZ: return  (*mag)*cos(x)*sin(y)*exp(z);
          case LAP: return -(*mag)*cos(x)*sin(y);
#else
          case VAL: return  (*mag)*cos(x)*sin(y);
          case DDX: return -(*mag)*sin(x)*sin(y);
          case DDY: return  (*mag)*cos(x)*cos(y);
          case LAP: return -(*mag)*2.*cos(x)*sin(y);
#endif
        }
      case 2: switch (what) {
          case VAL: return (*mag)*exp(x);
          case DDX: return (*mag)*exp(x);
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return (*mag)*exp(x);
        }
      case 3: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*0.5*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDX: return (*mag)*0.5*( 1.0+2.0*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDY: return (*mag)*0.5*( 0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDZ: return (*mag)*0.5*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case LAP: return (*mag)*( 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
                -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.) );
#else
          case VAL: return (*mag)*0.5*log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDX: return (*mag)*(1.0*x+0.80*y+0.50)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDY: return (*mag)*(0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case LAP: {
              double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
              return (*mag)*( 1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C );
            };
#endif
        }
      case 4: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z) );
          case DDX: return (*mag)*( 1.0*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.) );
          case DDY: return (*mag)*( 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.)) );
          case DDZ: return (*mag)*( 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.)) );
          case LAP: return (*mag)*( ( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
                + ( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z) );
#else
          case VAL: return (*mag)*( log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y) );
          case DDX: return (*mag)*( ( 0.7/(0.7*x+3.) )*sin(x+0.5*y) + ( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case DDY: return (*mag)*( ( - 1./(y+3.) )*sin(x+0.5*y) + 0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case LAP: return (*mag)*( ( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
                + ( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y) );
#endif
        }
      case 5: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*exp(x+z-y*y)*(y+cos(x-z));
          case DDX: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
          case DDY: return (*mag)*exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
          case DDZ: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
          case LAP: return (*mag)*exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
#else
          case VAL: return (*mag)*exp(x-y*y)*(y+cos(x));
          case DDX: return (*mag)*exp(x-y*y)*(y+cos(x)-sin(x));
          case DDY: return (*mag)*exp(x-y*y)*(1.-2.*y*(y+cos(x)));
          case LAP: return (*mag)*exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
#endif
        }
      case 6: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5)) );
          case DDX: return (*mag)*( ( 1.0*cos(x+0.3*y)*cos(x-0.7*y) - 1.0*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5) );
          case DDY: return (*mag)*( ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5) );
          case DDZ: return (*mag)*( ( 1.0*cos(x-0.7*y)*sin(x+0.3*y)                                 )*exp(z) + 3.*z/(x*x+y*y+z*z+0.5) );
          case LAP: return (*mag)*( (-1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.) );
#else
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5)) );
          case DDX: return (*mag)*(  1.00*cos(x+0.3*y)*cos(x-0.7*y) - 1.00*sin(x+0.3*y)*sin(x-0.7*y) + 3.*x/(x*x+y*y+0.5) );
          case DDY: return (*mag)*(  0.30*cos(x+0.3*y)*cos(x-0.7*y) + 0.70*sin(x+0.3*y)*sin(x-0.7*y) + 3.*y/(x*x+y*y+0.5) );
          case LAP: return (*mag)*( -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y) + 3./pow(x*x+y*y+0.5, 2.) );
#endif
        }
      case 7: {
        double p = mu_p_cf(x,y)/mu_m_cf(x,y);
        double r = sqrt(x*x+y*y);
        double r0 = 0.5+EPS;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
          case DDZ: return 0;
          case LAP: return 0;
#else
          case VAL: return (*mag)*( x*(p+1.) - x*(p-1.)*r0*r0/r/r )        /( p+1.+r0*r0*(p-1.) );
          case DDX: return (*mag)*( p+1.-r0*r0*(p-1.)*(y*y-x*x)/pow(r,4.) )/( p+1.+r0*r0*(p-1.) );
          case DDY: return (*mag)*( (p-1.)*r0*r0*2.*x*y/pow(r,4.) )        /( p+1.+r0*r0*(p-1.) );
          case LAP: return 0;
#endif
        }}
      case 8: {
        double p = mu_p_cf(x,y)/mu_m_cf(x,y);
        double r = sqrt(x*x+y*y);
        if (r < EPS) r = EPS;
        double r0 = 0.5+EPS;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
          case DDZ: return 0;
          case LAP: return 0;
#else
          case VAL: return (*mag)*2.*x/( p+1.+r0*r0*(p-1.) );
          case DDX: return (*mag)*2.  /( p+1.+r0*r0*(p-1.) );
          case DDY: return 0;
          case LAP: return 0;
#endif
        }}
      case 9: {
#ifdef P4_TO_P8
        double r2 = x*x+y*y+z*z;
#else
        double r2 = x*x+y*y;
#endif
        if (r2 < EPS) r2 = EPS;
        switch (what) {
          case VAL: return (*mag)*(1+log(2.*sqrt(r2)));
          case DDX: return (*mag)*x/r2;
          case DDY: return (*mag)*y/r2;
#ifdef P4_TO_P8
          case DDZ: return (*mag)*z/r2;
#endif
          case LAP: return 0;
        }}
      case 10: {
        double phase_x =  0.13;
        double phase_y =  1.55;
        double phase_z =  0.7;
        double kx = 1;
        double ky = 1;
        double kz = 1;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDZ: return  (*mag)*PI*kz*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*cos(PI*kz*z+phase_z);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky+kz*kz)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
#else
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
#endif
        }}
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};

u_pm_cf_t u_m_cf(u_pm_cf_t::VAL, n_um, mag_um), u_p_cf(u_pm_cf_t::VAL, n_up, mag_up);
u_pm_cf_t ux_m_cf(u_pm_cf_t::DDX, n_um, mag_um), ux_p_cf(u_pm_cf_t::DDX, n_up, mag_up);
u_pm_cf_t uy_m_cf(u_pm_cf_t::DDY, n_um, mag_um), uy_p_cf(u_pm_cf_t::DDY, n_up, mag_up);
#ifdef P4_TO_P8
u_pm_cf_t uz_m_cf(u_pm_cf_t::DDZ, n_um, mag_um), uz_p_cf(u_pm_cf_t::DDZ, n_up, mag_up);
#endif
u_pm_cf_t ul_m_cf(u_pm_cf_t::LAP, n_um, mag_um), ul_p_cf(u_pm_cf_t::LAP, n_up, mag_up);

// DIAGONAL TERMS
double mag_diag_m = 1;
double mag_diag_p = 1;

class diag_cf_t: public CF_DIM
{
  int    *n;
  double *mag;
public:
  diag_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x P8(+z))*exp(y);
    }
  }
};

diag_cf_t diag_m_cf(n_diag_m);
diag_cf_t diag_p_cf(n_diag_p);

// ROBIN COEFFICIENTS
int n_bc_coeff_00 = 0; double mag_bc_coeff_00 = 1;
int n_bc_coeff_01 = 0; double mag_bc_coeff_01 = 1;
int n_bc_coeff_02 = 0; double mag_bc_coeff_02 = 1;
int n_bc_coeff_03 = 0; double mag_bc_coeff_03 = 1;

class bc_coeff_cf_t: public CF_DIM
{
  int    *n;
  double *mag;
public:
  bc_coeff_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: return (*mag);
      case 1: return (*mag)*(x+y);
      case 2: return (*mag)*(x-y + (x+y)*(x+y));
      case 3: return (*mag)*(sin(x+y)*cos(x-y) P8(*log(z+4.)) );
      case 4: return (*mag)*(cos(x+y)*sin(x-y) P8(*exp(z)) );
      case 5: return (*mag)*(1.0 + sin(x P8(+z)) P8(*cos(y P8(+z))) );
      case 6: return (*mag)*(exp(x+y P8(+z) ));
      case 7: return (*mag)*(1.0 + sin(x)*cos(y) P8(*exp(z)) );
    }
  }
};

bc_coeff_cf_t bc_coeff_cf_00(n_bc_coeff_00, mag_bc_coeff_00);
bc_coeff_cf_t bc_coeff_cf_01(n_bc_coeff_01, mag_bc_coeff_01);
bc_coeff_cf_t bc_coeff_cf_02(n_bc_coeff_02, mag_bc_coeff_02);
bc_coeff_cf_t bc_coeff_cf_03(n_bc_coeff_03, mag_bc_coeff_03);

// RHS
class rhs_m_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return diag_m_cf(DIM(x,y,z))*u_m_cf(DIM(x,y,z))
        - mu_m_cf(DIM(x,y,z))*ul_m_cf(DIM(x,y,z))
        - mux_m_cf(DIM(x,y,z))*ux_m_cf(DIM(x,y,z))
    #ifdef P4_TO_P8
        - muz_m_cf(DIM(x,y,z))*uz_m_cf(DIM(x,y,z))
    #endif
        - muy_m_cf(DIM(x,y,z))*uy_m_cf(DIM(x,y,z));
  }
} rhs_m_cf;

class rhs_p_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return diag_p_cf(DIM(x,y,z))*u_p_cf(DIM(x,y,z))
        - mu_p_cf(DIM(x,y,z))*ul_p_cf(DIM(x,y,z))
        - mux_p_cf(DIM(x,y,z))*ux_p_cf(DIM(x,y,z))
    #ifdef P4_TO_P8
        - muz_p_cf(DIM(x,y,z))*uz_p_cf(DIM(x,y,z))
    #endif
        - muy_p_cf(DIM(x,y,z))*uy_p_cf(DIM(x,y,z));
  }
} rhs_p_cf;


int n_phi_00 = 0;

class dom_phi_cf_t : public CF_DIM
{
public:
  int    *n;
  enum value_type { VAL, DDX, DDY, DDZ, LAP } what;
  dom_phi_cf_t(value_type what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: switch (what) {
          case VAL: return -1;
          case DDX: return  0;
          case DDY: return  0;
#ifdef P4_TO_P8
          case DDZ: return  0;
#endif
        }
      case 1: {
        double r = 0.711;
        flower_shaped_domain_t circle(r, DIM(0,0,0));
        switch (what) {
          case VAL: return circle.phi(DIM(x,y,z));
          case DDX: return circle.phi_x(DIM(x,y,z));
          case DDY: return circle.phi_y(DIM(x,y,z));
#ifdef P4_TO_P8
          case DDZ: return circle.phi_z(DIM(x,y,z));
#endif
        }}
    }
  }
};

dom_phi_cf_t dom_phi_cf_00(phi_cf_t::VAL, n_phi_00);
dom_phi_cf_t dom_phi_x_cf_00(phi_cf_t::DDX, n_phi_00);
dom_phi_cf_t dom_phi_y_cf_00(phi_cf_t::DDY, n_phi_00);
#ifdef P4_TO_P8
dom_phi_cf_t dom_phi_z_cf_00(phi_cf_t::DDZ, n_phi_00);
#endif

int n_dom_phi_00 = 0;

class dom_phi_cf_t : public CF_DIM
{
public:
  int    *n;
  enum value_type { VAL, DDX, DDY, DDZ, LAP } what;
  dom_phi_cf_t(value_type what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: switch (what) {
          case VAL: return -1;
          case DDX: return  0;
          case DDY: return  0;
#ifdef P4_TO_P8
          case DDZ: return  0;
#endif
        }
      case 1: {
        double r = 0.711;
        flower_shaped_domain_t circle(r, DIM(0,0,0));
        switch (what) {
          case VAL: return circle.phi(DIM(x,y,z));
          case DDX: return circle.phi_x(DIM(x,y,z));
          case DDY: return circle.phi_y(DIM(x,y,z));
#ifdef P4_TO_P8
          case DDZ: return circle.phi_z(DIM(x,y,z));
#endif
        }}
    }
  }
};

dom_phi_cf_t dom_phi_cf_00(phi_cf_t::VAL, n_phi_00);
dom_phi_cf_t dom_phi_x_cf_00(phi_cf_t::DDX, n_phi_00);
dom_phi_cf_t dom_phi_y_cf_00(phi_cf_t::DDY, n_phi_00);
#ifdef P4_TO_P8
dom_phi_cf_t dom_phi_z_cf_00(phi_cf_t::DDZ, n_phi_00);
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

#ifdef P4_TO_P8
std::vector<CF_3 *> ii_phi_cf;
std::vector<CF_3 *> ii_phi_x_cf, ii_phi_y_cf, ii_phi_z_cf;
std::vector<CF_3 *> ii_bc_coeffs_cf;
#else
std::vector<CF_2 *> ii_phi_cf;
std::vector<CF_2 *> ii_phi_x_cf, ii_phi_y_cf;
std::vector<CF_2 *> ii_bc_coeffs_cf;
#endif

std::vector<action_t> ii_action;
std::vector<int> ii_color;

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

ii_problem_case_0_t  ii_problem_case_0;
ii_problem_case_1_t  ii_problem_case_1;
ii_problem_case_2_t  ii_problem_case_2;
ii_problem_case_3_t  ii_problem_case_3;
ii_problem_case_4_t  ii_problem_case_4;
ii_problem_case_5_t  ii_problem_case_5;
ii_problem_case_6_t  ii_problem_case_6;
ii_problem_case_7_t  ii_problem_case_7;
ii_problem_case_8_t  ii_problem_case_8;
ii_problem_case_9_t  ii_problem_case_9;
ii_problem_case_10_t ii_problem_case_10;
ii_problem_case_11_t ii_problem_case_11;

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
  switch (n_ii)
  {
    case 0:
    {
      ii_phi_cf        = ii_problem_case_0.phi_cf;
      ii_phi_x_cf      = ii_problem_case_0.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_0.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_0.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_0.bc_coeffs_cf;
      ii_action        = ii_problem_case_0.action;
      ii_color         = ii_problem_case_0.color;
    } break;
    case 1:
    {
      ii_phi_cf        = ii_problem_case_1.phi_cf;
      ii_phi_x_cf      = ii_problem_case_1.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_1.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_1.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_1.bc_coeffs_cf;
      ii_action        = ii_problem_case_1.action;
      ii_color         = ii_problem_case_1.color;
    } break;
    case 2:
    {
      ii_phi_cf        = ii_problem_case_2.phi_cf;
      ii_phi_x_cf      = ii_problem_case_2.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_2.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_2.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_2.bc_coeffs_cf;
      ii_action        = ii_problem_case_2.action;
      ii_color         = ii_problem_case_2.color;
    } break;
    case 3:
    {
      ii_phi_cf        = ii_problem_case_3.phi_cf;
      ii_phi_x_cf      = ii_problem_case_3.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_3.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_3.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_3.bc_coeffs_cf;
      ii_action        = ii_problem_case_3.action;
      ii_color         = ii_problem_case_3.color;
    } break;
    case 4:
    {
      ii_phi_cf        = ii_problem_case_4.phi_cf;
      ii_phi_x_cf      = ii_problem_case_4.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_4.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_4.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_4.bc_coeffs_cf;
      ii_action        = ii_problem_case_4.action;
      ii_color         = ii_problem_case_4.color;
    } break;
    case 5:
    {
      ii_phi_cf        = ii_problem_case_5.phi_cf;
      ii_phi_x_cf      = ii_problem_case_5.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_5.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_5.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_5.bc_coeffs_cf;
      ii_action        = ii_problem_case_5.action;
      ii_color         = ii_problem_case_5.color;
    } break;
    case 6:
    {
      ii_phi_cf        = ii_problem_case_6.phi_cf;
      ii_phi_x_cf      = ii_problem_case_6.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_6.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_6.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_6.bc_coeffs_cf;
      ii_action        = ii_problem_case_6.action;
      ii_color         = ii_problem_case_6.color;
    } break;
    case 7:
    {
      ii_phi_cf        = ii_problem_case_7.phi_cf;
      ii_phi_x_cf      = ii_problem_case_7.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_7.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_7.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_7.bc_coeffs_cf;
      ii_action        = ii_problem_case_7.action;
      ii_color         = ii_problem_case_7.color;
    } break;
    case 8:
    {
      ii_phi_cf        = ii_problem_case_8.phi_cf;
      ii_phi_x_cf      = ii_problem_case_8.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_8.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_8.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_8.bc_coeffs_cf;
      ii_action        = ii_problem_case_8.action;
      ii_color         = ii_problem_case_8.color;
    } break;
    case 9:
    {
      ii_phi_cf        = ii_problem_case_9.phi_cf;
      ii_phi_x_cf      = ii_problem_case_9.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_9.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_9.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_9.bc_coeffs_cf;
      ii_action        = ii_problem_case_9.action;
      ii_color         = ii_problem_case_9.color;
    } break;
    case 10:
    {
      ii_phi_cf        = ii_problem_case_10.phi_cf;
      ii_phi_x_cf      = ii_problem_case_10.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_10.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_10.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_10.bc_coeffs_cf;
      ii_action        = ii_problem_case_10.action;
      ii_color         = ii_problem_case_10.color;
    } break;
    case 11:
    {
      ii_phi_cf        = ii_problem_case_11.phi_cf;
      ii_phi_x_cf      = ii_problem_case_11.phi_x_cf;
      ii_phi_y_cf      = ii_problem_case_11.phi_y_cf;
#ifdef P4_TO_P8
      ii_phi_z_cf      = ii_problem_case_11.phi_z_cf;
#endif
      ii_bc_coeffs_cf  = ii_problem_case_11.bc_coeffs_cf;
      ii_action        = ii_problem_case_11.action;
      ii_color         = ii_problem_case_11.color;
    } break;
  }
}

class mu_cf_t : public CF_DIM
{
  CF_DIM *phi_;
public:
  void set_phi(CF_DIM &phi) { phi_ = &phi; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return (*phi_)(DIM(x,y,z)) > 0 ? mu_p_cf(DIM(x,y,z)) : mu_m_cf(DIM(x,y,z));
  }
} mu_cf;
class u_cf_t : public CF_DIM
{
  CF_DIM *phi_;
public:
  void set_phi(CF_DIM &phi) { phi_ = &phi; }
  double operator()(DIM(double x, double y, double z)) const  {
    return (*phi_)(DIM(x,y,z)) > 0 ? u_p_cf(DIM(x,y,z)) : u_m_cf(DIM(x,y,z));
  }
} u_cf;
class ux_cf_t : public CF_DIM
{
  CF_DIM *phi_;
public:
  void set_phi(CF_DIM &phi) { phi_ = &phi; }
  double operator()(DIM(double x, double y, double z)) const  {
    return (*phi_)(DIM(x,y,z)) > 0 ? ux_p_cf(DIM(x,y,z)) : ux_m_cf(DIM(x,y,z));
  }
} ux_cf;
class uy_cf_t : public CF_DIM
{
  CF_DIM *phi_;
public:
  void set_phi(CF_DIM &phi) { phi_ = &phi; }
  double operator()(DIM(double x, double y, double z)) const  {
    return (*phi_)(DIM(x,y,z)) > 0 ? uy_p_cf(DIM(x,y,z)) : uy_m_cf(DIM(x,y,z));
  }
} uy_cf;
#ifdef P4_TO_P8
class uz_cf_t : public CF_DIM
{
  CF_DIM *phi_;
public:
  void set_phi(CF_DIM &phi) { phi_ = &phi; }
  double operator()(DIM(double x, double y, double z)) const  {
    return (*phi_)(DIM(x,y,z)) > 0 ? uz_p_cf(DIM(x,y,z)) : uz_m_cf(DIM(x,y,z));
  }
} uz_cf;
#endif

// BC VALUES
class bc_value_robin_t : public CF_DIM
{
  CF_DIM *u;
  CF_DIM *ux;
  CF_DIM *uy;
  CF_DIM *uz;
  CF_DIM *phi_x;
  CF_DIM *phi_y;
  CF_DIM *phi_z;
  CF_DIM *kappa;
  CF_DIM *mu;
public:
  bc_value_robin_t(CF_DIM *u,
                   DIM(CF_DIM *ux, CF_DIM *uy, CF_DIM *uz),
                   DIM(CF_DIM *phi_x, CF_DIM *phi_y, CF_DIM *phi_z),
                   CF_DIM *kappa,
                   CF_DIM *mu) :
    u(u), DIM(ux(ux), uy(uy), uz(uz)), mu(mu), DIM(phi_x(phi_x), phi_y(phi_y), phi_z(phi_z)), kappa(kappa) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    double nx = (*phi_x)(DIM(x,y,z));
    double ny = (*phi_y)(DIM(x,y,z));
#ifdef P4_TO_P8
    double nz = (*phi_z)(DIM(x,y,z));
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    nz /= norm;
#else
    double norm = sqrt(nx*nx+ny*ny);
#endif
    nx /= norm; ny /= norm;

    return (*mu)(DIM(x,y,z))*(nx*(*ux)(DIM(x,y,z)) + ny*(*uy)(DIM(x,y,z)) P8(+ nz*(*uz)(x,y,z))) + (*kappa)(DIM(x,y,z))*(*u)(DIM(x,y,z));
  }
};

// JUMP CONDITIONS
class u_jump_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return u_p_cf(DIM(x,y,z)) - u_m_cf(DIM(x,y,z));
  }
} u_jump_cf;

class mu_un_jump_t : public CF_DIM
{
  CF_DIM *phi_x_;
  CF_DIM *phi_y_;
  CF_DIM *phi_z_;
public:
  mu_un_jump_t(DIM(CF_DIM *phi_x, CF_DIM *phi_y, CF_DIM *phi_z)) :
    DIM(phi_x_(phi_x), phi_y_(phi_y), phi_z_(phi_z)) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    double nx = (*phi_x_)(DIM(x,y,z));
    double ny = (*phi_y_)(DIM(x,y,z));
#ifdef P4_TO_P8
    double nz = (*phi_z_)(DIM(x,y,z));
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
     nz /= norm;
#else
    double norm = sqrt(nx*nx+ny*ny);
#endif
    nx /= norm; ny /= norm;

    return mu_p_cf(DIM(x,y,z)) * (ux_p_cf(DIM(x,y,z))*nx + uy_p_cf(DIM(x,y,z))*ny P8(+ uz_p_cf(x,y,z)*nz) )
        -  mu_m_cf(DIM(x,y,z)) * (ux_m_cf(DIM(x,y,z))*nx + uy_m_cf(DIM(x,y,z))*ny P8(+ uz_m_cf(x,y,z)*nz) );
  }
};

class bc_wall_type_t : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return bc_wtype;
  }
} bc_wall_type;


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

  for (short i = 0; i < 2; ++i)
  {
    //-------------------------------------
    // refinement parameters
    //-------------------------------------
    ADD_OPTION(i, lmin, "min level of the tree");
    ADD_OPTION(i, lmax, "max level of the tree");

    ADD_OPTION(i, num_splits,           "number of recursive splits");
    ADD_OPTION(i, num_splits_per_split, "number of additional resolutions");

    ADD_OPTION(i, num_shifts_x_dir, "number of shifts in x-dir");
    ADD_OPTION(i, num_shifts_y_dir, "number of shifts in y-dir");
    ADD_OPTION(i, num_shifts_z_dir, "number of shifts in z-dir");

    //-------------------------------------
    // output
    //-------------------------------------
    ADD_OPTION(i, save_vtk,           "Save the p4est in vtk format");
    ADD_OPTION(i, save_domain,        "Save the reconstruction of an irregular domain (works only in serial!)");
    ADD_OPTION(i, save_convergence,   "Save convergence results");
    ADD_OPTION(i, save_matrix_ascii,  "Save the matrix in ASCII MATLAB format");
    ADD_OPTION(i, save_matrix_binary, "Save the matrix in BINARY MATLAB format");

    //-------------------------------------
    // test solution
    //-------------------------------------
    ADD_OPTION(i, n_geometry,    "n_geometry");
    ADD_OPTION(i, n_ii,          "n_ii");

    ADD_OPTION(i, n_mu_m,        "n_mu_m");
    ADD_OPTION(i, n_mu_p,        "n_mu_p");

    ADD_OPTION(i, n_um,          "n_um");
    ADD_OPTION(i, n_up,          "n_up");

    ADD_OPTION(i, n_diag_m, "n_diag_m");
    ADD_OPTION(i, n_diag_p, "n_diag_p");

    ADD_OPTION(i, bc_wtype, "Type of boundary conditions on the walls");
    ADD_OPTION(i, bc_itype, "Type of boundary conditions on the interface");

    //-------------------------------------
    // solver parameters
    //-------------------------------------
    ADD_OPTION(i, sc_scheme,         "Use super-convergent scheme");
    ADD_OPTION(i, integration_order, "Select integration order (1 - linear, 2 - quadratic)");

    ADD_OPTION(i, jump_scheme,       "Discretization scheme for interface conditions (0 - FVM, 1 - FDM)");

    // for symmetric scheme:
    ADD_OPTION(i, taylor_correction,      "Use Taylor correction to approximate Robin term (symmetric scheme)");
    ADD_OPTION(i, kink_special_treatment, "Use the special treatment for kinks (symmetric scheme)");

    // for superconvergent scheme:
    ADD_OPTION(i, try_remove_hanging_cells, "Ask solver to eliminate hanging cells");

    //-------------------------------------
    // convergence study options
    //-------------------------------------
    ADD_OPTION(i, compute_cond_num, "Estimate L1-norm condition number");
    ADD_OPTION(i, mask_thresh,      "Mask threshold for excluding points in convergence study");
    ADD_OPTION(i, do_extension,     "Extend solution after solving");
    ADD_OPTION(i, compute_grad_between, "compute_grad_between");

    //-------------------------------------
    // level-set representation parameters
    //-------------------------------------
    ADD_OPTION(i, use_phi_cf,       "Use analytical level-set functions");
    ADD_OPTION(i, reinit_level_set, "Reinitialize level-set function");

    // artificial perturbation of level-set values
    ADD_OPTION(i, domain_perturbation,     "Artificially pertub level-set functions (0 - no perturbation, 1 - smooth, 2 - noisy)");
    ADD_OPTION(i, domain_perturbation_mag, "Magnitude of level-set perturbations");
    ADD_OPTION(i, domain_perturbation_pow, "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

    if (i == 0) cmd.parse(argc, argv);
  }

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
       save_convergence ||
       save_matrix_ascii ||
       save_matrix_binary))
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

  if (save_matrix_ascii || save_matrix_binary)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/matrix";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/matrix directory");
  }

  // vectors to store convergence results
  vector<double> lvl_arr, h_arr;

  vector<double> error_sl_m_arr;
  vector<double> error_ex_m_arr;
  vector<double> error_dd_m_arr;
  vector<double> error_tr_m_arr;
  vector<double> error_gr_m_arr;
  vector<double> error_ge_m_arr;

  vector<double> error_sl_p_arr;
  vector<double> error_ex_p_arr;
  vector<double> error_dd_p_arr;
  vector<double> error_tr_p_arr;
  vector<double> error_gr_p_arr;
  vector<double> error_ge_p_arr;

  vector<double> cond_num_arr;

  // Start up a MATLAB Engine to calculate condidition number
  Engine *mengine = NULL;
  if (mpi.rank() == 0 && compute_cond_num)
  {
    mengine = engOpen("matlab -nodisplay -nojvm");
    if (mengine == NULL) throw std::runtime_error("Cannot start a MATLAB Engine session.\n");
  }

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  // the effective LSF
  level_set_tot_t level_set_tot_cf(&phi_cf, &action, &color);
  level_set_tot_t ii_level_set_tot_cf(&ii_phi_cf, &ii_action, &ii_color);
  mu_cf.set_phi(ii_level_set_tot_cf);
  u_cf.set_phi(ii_level_set_tot_cf);

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

            splitting_criteria_cf_t data_tmp(lmin, lmax, &level_set_tot_cf, 1.4);
            p4est->user_pointer = (void*)(&data_tmp);

            //    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
            my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
            my_p4est_partition(p4est, P4EST_FALSE, NULL);
            for (int i = 0; i < iter; ++i)
            {
              my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
              my_p4est_partition(p4est, P4EST_FALSE, NULL);
            }

            splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot_cf, 1.4);
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
//            double diag = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);
#else
            double dxyz_max = MAX(dxyz[0], dxyz[1]);
//            double diag = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1]);
#endif

            unsigned int num_surfaces = phi_cf.size();

            // sample level-set functions
            std::vector<Vec> phi;
            for (unsigned int i = 0; i < num_surfaces; i++)
            {
              phi.push_back(Vec());
              ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, *phi_cf[i], phi.back());

//              if (domain_perturbation)
//              {
//                double *phi_ptr;
//                ierr = VecGetArray(phi.back(), &phi_ptr); CHKERRXX(ierr);

//                for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
//                {
//                  double xyz[P4EST_DIM];
//                  node_xyz_fr_n(n, p4est, nodes, xyz);
//                  phi_ptr[n] += domain_perturbation_mag*perturb_cf.value(xyz)*pow(dxyz_m, domain_perturbation_pow);
//                }

//                ierr = VecRestoreArray(phi.back(), &phi_ptr); CHKERRXX(ierr);

//                ierr = VecGhostUpdateBegin(phi.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//                ierr = VecGhostUpdateEnd  (phi.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//              }

              if (reinit_level_set)
                ls.reinitialize_1st_order_time_2nd_order_space(phi.back(),20);
            }

            unsigned int ii_num_surfaces = ii_phi_cf.size();

            // sample level-set functions
            std::vector<Vec> ii_phi;
            for (unsigned int i = 0; i < ii_num_surfaces; i++)
            {
              ii_phi.push_back(Vec());
              ierr = VecCreateGhostNodes(p4est, nodes, &ii_phi.back()); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, *ii_phi_cf[i], ii_phi.back());

//              if (domain_perturbation)
//              {
//                double *phi_ptr;
//                ierr = VecGetArray(phi.back(), &phi_ptr); CHKERRXX(ierr);

//                for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
//                {
//                  double xyz[P4EST_DIM];
//                  node_xyz_fr_n(n, p4est, nodes, xyz);
//                  phi_ptr[n] += domain_perturbation_mag*perturb_cf.value(xyz)*pow(dxyz_m, domain_perturbation_pow);
//                }

//                ierr = VecRestoreArray(phi.back(), &phi_ptr); CHKERRXX(ierr);

//                ierr = VecGhostUpdateBegin(phi.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//                ierr = VecGhostUpdateEnd  (phi.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//              }

              if (reinit_level_set)
                ls.reinitialize_1st_order_time_2nd_order_space(ii_phi.back(),20);
            }

            std::vector<BoundaryConditionType> bc_interface_type(num_surfaces, bc_itype);

            // sample boundary conditions
#ifdef P4_TO_P8
            std::vector<bc_value_robin_t *> bc_interface_value_(num_surfaces, NULL);
            std::vector<CF_3 *> bc_interface_value(num_surfaces, NULL);
#else
            std::vector<bc_value_robin_t *> bc_interface_value_(num_surfaces, NULL);
            std::vector<CF_2 *> bc_interface_value(num_surfaces, NULL);
#endif
            for (unsigned int i = 0; i < num_surfaces; i++)
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

#ifdef P4_TO_P8
            std::vector<mu_un_jump_t *> mu_un_jump(ii_num_surfaces, NULL);
            std::vector<CF_3 *> mu_un_jump_cf(ii_num_surfaces, NULL);
#else
            std::vector<mu_un_jump_t *> mu_un_jump(ii_num_surfaces, NULL);
            std::vector<CF_2 *> mu_un_jump_cf(ii_num_surfaces, NULL);
#endif
            for (int i = 0; i < ii_num_surfaces; i++)
            {
#ifdef P4_TO_P8
              mu_un_jump[i] = new mu_un_jump_t(ii_phi_x_cf[i], ii_phi_y_cf[i], ii_phi_z_cf[i]);
#else
              mu_un_jump[i] = new mu_un_jump_t(ii_phi_x_cf[i], ii_phi_y_cf[i]);
#endif
              mu_un_jump_cf[i] = mu_un_jump[i];
            }


            Vec rhs_m;
            ierr = VecCreateGhostNodes(p4est, nodes, &rhs_m); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, rhs_m_cf, rhs_m);

            Vec rhs_p;
            ierr = VecCreateGhostNodes(p4est, nodes, &rhs_p); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, rhs_p_cf, rhs_p);

            Vec mu_m;
            ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);

            Vec mu_p;
            ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

            Vec diag_m;
            ierr = VecCreateGhostNodes(p4est, nodes, &diag_m); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, diag_m_cf, diag_m);

            Vec diag_p;
            ierr = VecCreateGhostNodes(p4est, nodes, &diag_p); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, diag_p_cf, diag_p);


//            ierr = PetscPrintf(mpi.comm(), "Starting a solver\n"); CHKERRXX(ierr);

            Vec sol; double *sol_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
            std::vector<Vec> *phi_dd[P4EST_DIM];

            Vec phi_eff; double *phi_eff_ptr;
            Vec ii_phi_eff; double *ii_phi_eff_ptr;
            Vec mask;

            Mat A;
            std::vector<double> *scalling;
            Vec areas;

            my_p4est_poisson_nodes_mls_sc_t solver(&ngbd_n);

            solver.set_jump_scheme(jump_scheme);
            solver.set_use_sc_scheme(sc_scheme);
            solver.set_integration_order(integration_order);

//            if (use_phi_cf) solver.set_phi_cf(phi_cf);

            solver.set_geometry(num_surfaces, &action, &color, &phi);
            solver.set_immersed_interface(ii_num_surfaces, &ii_action, &ii_color, &ii_phi);

            solver.set_mu2(mu_m, mu_p);
            solver.set_rhs(rhs_m, rhs_p);
//            solver.set_rhs(rhs);

            solver.set_bc_wall_value(u_cf);
            solver.set_bc_wall_type(bc_wall_type);

            solver.set_bc_interface_type(bc_interface_type);
            solver.set_bc_interface_coeff(bc_coeffs_cf);
            solver.set_bc_interface_value(bc_interface_value);

            solver.set_jump_conditions(u_jump_cf, mu_un_jump_cf);

            solver.set_diag_add(diag_m, diag_p);

            solver.set_use_taylor_correction(taylor_correction);
            solver.set_keep_scalling(1);
            solver.set_kink_treatment(kink_special_treatment);
            solver.set_try_remove_hanging_cells(try_remove_hanging_cells);

            solver.solve(sol);


            solver.get_phi_dd(phi_dd);

            phi_eff   = solver.get_phi_eff();
            ii_phi_eff= solver.get_immersed_phi_eff();
            mask      = solver.get_mask();
            A         = solver.get_matrix();
            scalling  = solver.get_scalling();
            areas     = solver.get_areas();

            if (save_matrix_ascii)
            {
              std::ostringstream oss; oss << out_dir << "/matrix/mat_" << file_idx << ".m";

              PetscViewer viewer;
              ierr = PetscViewerASCIIOpen(mpi.comm(), oss.str().c_str(), &viewer); CHKERRXX(ierr);
              ierr = PetscViewerPushFormat(viewer, 	PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);

              ierr = PetscObjectSetName((PetscObject)A, "mat");
              ierr = MatView(A, viewer); CHKERRXX(ierr);

              Vec lex_order;
              ierr = VecCreateGhostNodes(p4est, nodes, &lex_order); CHKERRXX(ierr);

              double *vec_ptr; ierr = VecGetArray(lex_order, &vec_ptr); CHKERRXX(ierr);

              int nx = round((grid_xyz_max_shift[0]-grid_xyz_min_shift[0])/dxyz[0] + 1);
#ifdef P4_TO_P8
              int ny = round((grid_xyz_max_shift[1]-grid_xyz_min_shift[1])/dxyz[1] + 1);
              int nz = round((grid_xyz_max_shift[2]-grid_xyz_min_shift[2])/dxyz[2] + 1);
#endif
              for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
              {
                double xyz[P4EST_DIM];
                node_xyz_fr_n(n, p4est, nodes, xyz);

                int ix = round((xyz[0]-grid_xyz_min_shift[0])/dxyz[0]);
                int iy = round((xyz[1]-grid_xyz_min_shift[1])/dxyz[1]);
#ifdef P4_TO_P8
                int iz = round((xyz[2]-grid_xyz_min_shift[2])/dxyz[2]);
                vec_ptr[n] = iz*nx*ny + iy*(nx) + ix + 1;
#else
                vec_ptr[n] = iy*(nx) + ix + 1;
#endif
              }

              ierr = VecRestoreArray(lex_order, &vec_ptr); CHKERRXX(ierr);

              ierr = PetscObjectSetName((PetscObject)lex_order, "vec");
              ierr = VecView(lex_order, viewer); CHKERRXX(ierr);

              ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
            }

            if (save_matrix_binary)
            {
              std::ostringstream oss; oss << out_dir << "/matrix/mat_" << file_idx << ".dat";

              PetscViewer viewer;
              ierr = PetscViewerBinaryOpen(mpi.comm(), oss.str().c_str(), FILE_MODE_WRITE, &viewer); CHKERRXX(ierr);
              ierr = PetscViewerPushFormat(viewer, 	PETSC_VIEWER_BINARY_MATLAB); CHKERRXX(ierr);
              ierr = MatView(A, viewer); CHKERRXX(ierr);
              ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
            }

            if (iter < compute_cond_num)
            {
              // Get the local AIJ representation of the matrix
              std::vector<double> aij;

              for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
              {
                int num_elem;
                const int *icol;
                const double *vals;

                PetscInt N = solver.get_global_idx(n);
                MatGetRow(A, N, &num_elem, &icol, &vals);
                for (int i = 0; i < num_elem; ++i)
                {
                  aij.push_back((double) (N+1));
                  aij.push_back((double) (icol[i]+1));
                  aij.push_back(vals[i]);
                }
                MatRestoreRow(A, N, &num_elem, &icol, &vals);
              }

              int num_local_entries = aij.size();

              // Collect all chucks of the matrix into global_aij on the 0-rank process
              std::vector<int> local_sizes(mpi.size(), 0);
              std::vector<int> displs(mpi.size(), 0);

              MPI_Gather(&num_local_entries, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, 0, mpi.comm());

              int num_total_entries = local_sizes[0];

              for (int i = 1; i < mpi.size(); ++i)
              {
                displs[i] = displs[i-1] + local_sizes[i-1];
                num_total_entries += local_sizes[i];
              }

              mxArray *mat = NULL;
              mxDouble *mat_data = NULL;

              if (mpi.rank() == 0)
              {
                mat = mxCreateDoubleMatrix(3, num_total_entries/3, mxREAL);
                mat_data = mxGetDoubles(mat);
              }

              MPI_Gatherv(aij.data(), aij.size(), MPI_DOUBLE, mat_data, local_sizes.data(), displs.data(), MPI_DOUBLE, 0, mpi.comm());

              aij.clear();

              // pass the matrix to MATLAB and ask to compute condition number
              if (mpi.rank() == 0)
              {
                // send the matrix to MATLAB
                engPutVariable(mengine, "AIJ", mat);
                mxDestroyArray(mat);

                // ask to compute condition number
                engEvalString(mengine, "cn = condest(spconvert(AIJ'));");

                // get the result
                mxArray *value = engGetVariable(mengine, "cn");
                double cn = *mxGetDoubles(value);
                mxDestroyArray(value);

                // store
                cond_num_arr.push_back(cn);
              } else {
                cond_num_arr.push_back(NAN);
              }
            } else {
              cond_num_arr.push_back(NAN);
            }

            my_p4est_integration_mls_t integrator(p4est, nodes);
#ifdef P4_TO_P8
            integrator.set_phi(phi, action, color);
#else
            integrator.set_phi(phi, action, color);
#endif

            /*
            if (save_domain)
            {
              std::ostringstream oss; oss << out_dir << "/geometry";

#ifdef P4_TO_P8
              vector<cube3_mls_t> cubes;
              unsigned int n_sps = 6;
#else
              vector<cube2_mls_t> cubes;
              unsigned int n_sps = 2;
#endif
              solver.reconstruct_domain(cubes);

              if (integration_order == 1)
              {
#ifdef P4_TO_P8
                vector<simplex3_mls_l_t *> simplices;
#else
                vector<simplex2_mls_l_t *> simplices;
#endif
                for (unsigned int k = 0; k < cubes.size(); k++)
                  for (unsigned int kk = 0; kk < cubes[k].cubes_l_.size(); kk++)
                    if (cubes[k].cubes_l_[kk]->loc == FCE)
                      for (unsigned int l = 0; l < n_sps; l++)
                        simplices.push_back(&cubes[k].cubes_l_[kk]->simplex[l]);

#ifdef P4_TO_P8
                simplex3_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#else
                simplex2_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#endif
              } else if (integration_order == 2) {

#ifdef P4_TO_P8
                vector<simplex3_mls_q_t *> simplices;
#else
                vector<simplex2_mls_q_t *> simplices;
#endif
                for (unsigned int k = 0; k < cubes.size(); k++)
                  for (unsigned int kk = 0; kk < cubes[k].cubes_q_.size(); kk++)
                    if (cubes[k].cubes_q_[kk]->loc == FCE)
                      for (unsigned int l = 0; l < n_sps; l++)
                        simplices.push_back(&cubes[k].cubes_q_[kk]->simplex[l]);

#ifdef P4_TO_P8
                simplex3_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#else
                simplex2_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#endif
              }

            }
            //*/

            Vec sol_m = sol; double *sol_m_ptr;
            Vec sol_p = sol; double *sol_p_ptr;

            Vec mask_m; double *mask_m_ptr;
            Vec mask_p; double *mask_p_ptr;

            ierr = VecDuplicate(mask, &mask_m); CHKERRXX(ierr);
            ierr = VecDuplicate(mask, &mask_p); CHKERRXX(ierr);

            copy_ghosted_vec(mask, mask_m);
            copy_ghosted_vec(mask, mask_p);

            ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(ii_phi_eff, &ii_phi_eff_ptr); CHKERRXX(ierr);

            foreach_node(n, nodes)
            {
              if (ii_phi_eff_ptr[n] < 0) mask_p_ptr[n] = 1;
              else                       mask_m_ptr[n] = 1;
            }

            ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);
            ierr = VecRestoreArray(ii_phi_eff, &ii_phi_eff_ptr); CHKERRXX(ierr);

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

            ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

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
                  solver.get_all_neighbors(n, neighbors, neighbors_exist);
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
                double udd_exact = ul_m_cf(DIM(xyz[0],xyz[1],xyz[2]));

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
                double udd_exact = ul_p_cf(DIM(xyz[0],xyz[1],xyz[2]));

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

            Vec vec_u_exact_block;    double *vec_u_exact_block_ptr;    ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &vec_u_exact_block);    CHKERRXX(ierr);
            Vec vec_error_tr_block;   double *vec_error_tr_block_ptr;   ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &vec_error_tr_block);   CHKERRXX(ierr);

//            ierr = VecGetArray(vec_u_exact_block, &vec_u_exact_block_ptr); CHKERRXX(ierr);

//            ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
//            ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

//            for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
//            {
//              double xyz[P4EST_DIM];
//              node_xyz_fr_n(i, p4est, nodes, xyz);

//              vec_u_exact_block_ptr[2*i  ] = mask_m_ptr[i] < 0 ? u_m_cf.value(xyz) : 0;
//              vec_u_exact_block_ptr[2*i+1] = mask_p_ptr[i] < 0 ? u_p_cf.value(xyz) : 0;
//            }

//            ierr = VecRestoreArray(vec_u_exact_block, &vec_u_exact_block_ptr); CHKERRXX(ierr);

////            ierr = MatMult(A, vec_u_exact_block, vec_error_tr_block); CHKERRXX(ierr);

//            ierr = VecGetArray(vec_error_tr_block, &vec_error_tr_block_ptr); CHKERRXX(ierr);
//            ierr = VecGetArray(vec_error_tr_m,     &vec_error_tr_m_ptr); CHKERRXX(ierr);
//            ierr = VecGetArray(vec_error_tr_p,     &vec_error_tr_p_ptr); CHKERRXX(ierr);

//            // FIX THIS
//            Vec rhs = sol;
//            double *rhs_ptr;
//            ierr = VecGetArray(rhs, &rhs_ptr); CHKERRXX(ierr);

//            for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
//            {
//              vec_error_tr_m_ptr[n] = (mask_m_ptr[n] < 0) ? (vec_error_tr_block_ptr[2*n  ] - rhs_ptr[2*n  ])*scalling->at(2*n  ) : 0;
//              vec_error_tr_p_ptr[n] = (mask_p_ptr[n] < 0) ? (vec_error_tr_block_ptr[2*n+1] - rhs_ptr[2*n+1])*scalling->at(2*n+1) : 0;
//            }

//            ierr = VecRestoreArray(vec_error_tr_block, &vec_error_tr_block_ptr);  CHKERRXX(ierr);
//            ierr = VecRestoreArray(vec_error_tr_m,     &vec_error_tr_m_ptr);  CHKERRXX(ierr);
//            ierr = VecRestoreArray(vec_error_tr_p,     &vec_error_tr_p_ptr);  CHKERRXX(ierr);

//            ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
//            ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

//            ierr = VecRestoreArray(rhs, &rhs_ptr); CHKERRXX(ierr);

//            ierr = VecGhostUpdateBegin(vec_error_tr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//            ierr = VecGhostUpdateBegin(vec_error_tr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//            ierr = VecGhostUpdateEnd  (vec_error_tr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//            ierr = VecGhostUpdateEnd  (vec_error_tr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//            ierr = VecDestroy(vec_u_exact_block); CHKERRXX(ierr);

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

//            if (scale_errors)
//            {
//              mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//              err_sl_m_max /= u_max;
//              err_sl_p_max /= u_max;

//              mpiret = MPI_Allreduce(MPI_IN_PLACE, &gr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
//              err_gr_m_max /= gr_max;
//              err_gr_p_max /= gr_max;
//            }

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
//              ierr = PetscPrintf(p4est->mpicomm, "Error (tr_m): %g, order = %g\n", err_tr_m_max, log(error_tr_m_arr[iter-1]/error_tr_m_arr[iter])/log(2)); CHKERRXX(ierr);
              ierr = PetscPrintf(p4est->mpicomm, "Error (gr_m): %g, order = %g\n", err_gr_m_max, log(error_gr_m_arr[iter-1]/error_gr_m_arr[iter])/log(2)); CHKERRXX(ierr);
//              ierr = PetscPrintf(p4est->mpicomm, "Error (ex_m): %g, order = %g\n", err_ex_m_max, log(error_ex_m_arr[iter-1]/error_ex_m_arr[iter])/log(2)); CHKERRXX(ierr);
//              ierr = PetscPrintf(p4est->mpicomm, "Error (dd_m): %g, order = %g\n", err_dd_m_max, log(error_dd_m_arr[iter-1]/error_dd_m_arr[iter])/log(2)); CHKERRXX(ierr);

              ierr = PetscPrintf(p4est->mpicomm, "Error (sl_p): %g, order = %g\n", err_sl_p_max, log(error_sl_p_arr[iter-1]/error_sl_p_arr[iter])/log(2)); CHKERRXX(ierr);
//              ierr = PetscPrintf(p4est->mpicomm, "Error (tr_p): %g, order = %g\n", err_tr_p_max, log(error_tr_p_arr[iter-1]/error_tr_p_arr[iter])/log(2)); CHKERRXX(ierr);
              ierr = PetscPrintf(p4est->mpicomm, "Error (gr_p): %g, order = %g\n", err_gr_p_max, log(error_gr_p_arr[iter-1]/error_gr_p_arr[iter])/log(2)); CHKERRXX(ierr);
//              ierr = PetscPrintf(p4est->mpicomm, "Error (ex_p): %g, order = %g\n", err_ex_p_max, log(error_ex_p_arr[iter-1]/error_ex_p_arr[iter])/log(2)); CHKERRXX(ierr);
//              ierr = PetscPrintf(p4est->mpicomm, "Error (dd_p): %g, order = %g\n", err_dd_p_max, log(error_dd_p_arr[iter-1]/error_dd_p_arr[iter])/log(2)); CHKERRXX(ierr);

              ierr = PetscPrintf(p4est->mpicomm, "Cond num: %e\n", cond_num_arr[iter]); CHKERRXX(ierr);
            }

            if(save_vtk)
            {
              char *out_dir;
              out_dir = getenv("OUT_DIR");

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

              ierr = VecGetArray(sol,    &sol_ptr);    CHKERRXX(ierr);

              ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
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
                                     4, 1, oss.str().c_str(),
                                     VTK_POINT_DATA, "phi", phi_eff_ptr,
//                                     VTK_POINT_DATA, "phi_smooth", phi_smooth_ptr,
                                     VTK_POINT_DATA, "sol", sol_ptr,
//                                     VTK_POINT_DATA, "sol_m_ex", sol_m_ex_ptr,
//                                     VTK_POINT_DATA, "sol_p_ex", sol_p_ex_ptr,
                                     VTK_POINT_DATA, "error_sl_m", vec_error_sl_m_ptr,
//                                     VTK_POINT_DATA, "error_tr_m", vec_error_tr_m_ptr,
//                                     VTK_POINT_DATA, "error_gr_m", vec_error_gr_m_ptr,
//                                     VTK_POINT_DATA, "error_ex_m", vec_error_ex_m_ptr,
//                                     VTK_POINT_DATA, "error_dd_m", vec_error_dd_m_ptr,
                                     VTK_POINT_DATA, "error_sl_p", vec_error_sl_p_ptr,
//                                     VTK_POINT_DATA, "error_tr_p", vec_error_tr_p_ptr,
//                                     VTK_POINT_DATA, "error_gr_p", vec_error_gr_p_ptr,
//                                     VTK_POINT_DATA, "error_ex_p", vec_error_ex_p_ptr,
//                                     VTK_POINT_DATA, "error_dd_p", vec_error_dd_p_ptr,
//                                     VTK_POINT_DATA, "mask_m", mask_m_ptr,
//                                     VTK_POINT_DATA, "mask_p", mask_p_ptr,
                                     VTK_CELL_DATA , "leaf_level", l_p);

              ierr = VecRestoreArray(phi_eff,    &phi_eff_ptr);    CHKERRXX(ierr);
              ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

              ierr = VecRestoreArray(sol,    &sol_ptr);    CHKERRXX(ierr);

              ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
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

            for (int i = 0; i < ii_phi.size(); i++)
            {
              ierr = VecDestroy(ii_phi[i]);        CHKERRXX(ierr);
            }

            for (int i = 0; i < num_surfaces; i++)
            {
              delete mu_un_jump[i];
            }

            ierr = VecDestroy(sol);           CHKERRXX(ierr);

            ierr = VecDestroy(mu_m);          CHKERRXX(ierr);
            ierr = VecDestroy(mu_p);          CHKERRXX(ierr);

            ierr = VecDestroy(rhs_m);         CHKERRXX(ierr);
            ierr = VecDestroy(rhs_p);         CHKERRXX(ierr);

            ierr = VecDestroy(diag_m);   CHKERRXX(ierr);
            ierr = VecDestroy(diag_p);   CHKERRXX(ierr);

//            // destroy Vec's with errors
//            ierr = VecDestroy(vec_error_sl); CHKERRXX(ierr);
//            ierr = VecDestroy(vec_error_tr); CHKERRXX(ierr);
//            ierr = VecDestroy(vec_error_gr); CHKERRXX(ierr);
//            ierr = VecDestroy(vec_error_ex); CHKERRXX(ierr);
//            ierr = VecDestroy(vec_error_dd); CHKERRXX(ierr);
//            ierr = VecDestroy(vec_error_ge); CHKERRXX(ierr);

            ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);
//            ierr = VecDestroy(sol_ex); CHKERRXX(ierr);

            for (unsigned int i = 0; i < phi.size(); i++)
            {
              ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
            }

            for (unsigned int i = 0; i < num_surfaces; i++)
            {
              if (bc_interface_type[i] == ROBIN || bc_interface_type[i] == NEUMANN)
              {
                delete bc_interface_value_[i];
              }
            }

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

  if (mpi.rank() == 0 && compute_cond_num)
  {
    engClose(mengine);
  }


  MPI_Barrier(mpi.comm());

  std::vector<double> error_m_sl_one(num_resolutions, 0), error_m_sl_avg(num_resolutions, 0), error_m_sl_max(num_resolutions, 0);
  std::vector<double> error_m_gr_one(num_resolutions, 0), error_m_gr_avg(num_resolutions, 0), error_m_gr_max(num_resolutions, 0);
  std::vector<double> error_m_ge_one(num_resolutions, 0), error_m_ge_avg(num_resolutions, 0), error_m_ge_max(num_resolutions, 0);
  std::vector<double> error_m_dd_one(num_resolutions, 0), error_m_dd_avg(num_resolutions, 0), error_m_dd_max(num_resolutions, 0);
  std::vector<double> error_m_tr_one(num_resolutions, 0), error_m_tr_avg(num_resolutions, 0), error_m_tr_max(num_resolutions, 0);
  std::vector<double> error_m_ex_one(num_resolutions, 0), error_m_ex_avg(num_resolutions, 0), error_m_ex_max(num_resolutions, 0);

  std::vector<double> error_p_sl_one(num_resolutions, 0), error_p_sl_avg(num_resolutions, 0), error_p_sl_max(num_resolutions, 0);
  std::vector<double> error_p_gr_one(num_resolutions, 0), error_p_gr_avg(num_resolutions, 0), error_p_gr_max(num_resolutions, 0);
  std::vector<double> error_p_ge_one(num_resolutions, 0), error_p_ge_avg(num_resolutions, 0), error_p_ge_max(num_resolutions, 0);
  std::vector<double> error_p_dd_one(num_resolutions, 0), error_p_dd_avg(num_resolutions, 0), error_p_dd_max(num_resolutions, 0);
  std::vector<double> error_p_tr_one(num_resolutions, 0), error_p_tr_avg(num_resolutions, 0), error_p_tr_max(num_resolutions, 0);
  std::vector<double> error_p_ex_one(num_resolutions, 0), error_p_ex_avg(num_resolutions, 0), error_p_ex_max(num_resolutions, 0);

  std::vector<double> cond_num_one(num_resolutions, 0), cond_num_avg(num_resolutions, 0), cond_num_max(num_resolutions, 0);

  error_ge_m_arr = error_sl_m_arr;
  error_dd_m_arr = error_sl_m_arr;
  error_tr_m_arr = error_sl_m_arr;
  error_ex_m_arr = error_sl_m_arr;
  error_ge_p_arr = error_sl_p_arr;
  error_dd_p_arr = error_sl_p_arr;
  error_tr_p_arr = error_sl_p_arr;
  error_ex_p_arr = error_sl_p_arr;

  // for each resolution compute max, mean and deviation
  for (int p = 0; p < num_resolutions; ++p)
  {
    // one
    error_m_sl_one[p] = error_sl_m_arr[p*num_shifts_total];
    error_m_gr_one[p] = error_gr_m_arr[p*num_shifts_total];
    error_m_ge_one[p] = error_ge_m_arr[p*num_shifts_total];
    error_m_dd_one[p] = error_dd_m_arr[p*num_shifts_total];
    error_m_tr_one[p] = error_tr_m_arr[p*num_shifts_total];
    error_m_ex_one[p] = error_ex_m_arr[p*num_shifts_total];

    error_p_sl_one[p] = error_sl_p_arr[p*num_shifts_total];
    error_p_gr_one[p] = error_gr_p_arr[p*num_shifts_total];
    error_p_ge_one[p] = error_ge_p_arr[p*num_shifts_total];
    error_p_dd_one[p] = error_dd_p_arr[p*num_shifts_total];
    error_p_tr_one[p] = error_tr_p_arr[p*num_shifts_total];
    error_p_ex_one[p] = error_ex_p_arr[p*num_shifts_total];

    cond_num_one[p] = cond_num_arr[p*num_shifts_total];

    // max
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_m_sl_max[p] = MAX(error_m_sl_max[p], error_sl_m_arr[p*num_shifts_total + s]);
      error_m_gr_max[p] = MAX(error_m_gr_max[p], error_gr_m_arr[p*num_shifts_total + s]);
      error_m_ge_max[p] = MAX(error_m_ge_max[p], error_ge_m_arr[p*num_shifts_total + s]);
      error_m_dd_max[p] = MAX(error_m_dd_max[p], error_dd_m_arr[p*num_shifts_total + s]);
      error_m_tr_max[p] = MAX(error_m_tr_max[p], error_tr_m_arr[p*num_shifts_total + s]);
      error_m_ex_max[p] = MAX(error_m_ex_max[p], error_ex_m_arr[p*num_shifts_total + s]);

      error_p_sl_max[p] = MAX(error_p_sl_max[p], error_sl_p_arr[p*num_shifts_total + s]);
      error_p_gr_max[p] = MAX(error_p_gr_max[p], error_gr_p_arr[p*num_shifts_total + s]);
      error_p_ge_max[p] = MAX(error_p_ge_max[p], error_ge_p_arr[p*num_shifts_total + s]);
      error_p_dd_max[p] = MAX(error_p_dd_max[p], error_dd_p_arr[p*num_shifts_total + s]);
      error_p_tr_max[p] = MAX(error_p_tr_max[p], error_tr_p_arr[p*num_shifts_total + s]);
      error_p_ex_max[p] = MAX(error_p_ex_max[p], error_ex_p_arr[p*num_shifts_total + s]);

      cond_num_max[p] = MAX(cond_num_max[p], cond_num_arr[p*num_shifts_total + s]);
    }

    // avg
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_m_sl_avg[p] += error_sl_m_arr[p*num_shifts_total + s];
      error_m_gr_avg[p] += error_gr_m_arr[p*num_shifts_total + s];
      error_m_ge_avg[p] += error_ge_m_arr[p*num_shifts_total + s];
      error_m_dd_avg[p] += error_dd_m_arr[p*num_shifts_total + s];
      error_m_tr_avg[p] += error_tr_m_arr[p*num_shifts_total + s];
      error_m_ex_avg[p] += error_ex_m_arr[p*num_shifts_total + s];

      error_p_sl_avg[p] += error_sl_p_arr[p*num_shifts_total + s];
      error_p_gr_avg[p] += error_gr_p_arr[p*num_shifts_total + s];
      error_p_ge_avg[p] += error_ge_p_arr[p*num_shifts_total + s];
      error_p_dd_avg[p] += error_dd_p_arr[p*num_shifts_total + s];
      error_p_tr_avg[p] += error_tr_p_arr[p*num_shifts_total + s];
      error_p_ex_avg[p] += error_ex_p_arr[p*num_shifts_total + s];

      cond_num_avg[p] += cond_num_arr[p*num_shifts_total + s];
    }

    error_m_sl_avg[p] /= num_shifts_total;
    error_m_gr_avg[p] /= num_shifts_total;
    error_m_ge_avg[p] /= num_shifts_total;
    error_m_dd_avg[p] /= num_shifts_total;
    error_m_tr_avg[p] /= num_shifts_total;
    error_m_ex_avg[p] /= num_shifts_total;

    error_p_sl_avg[p] /= num_shifts_total;
    error_p_gr_avg[p] /= num_shifts_total;
    error_p_ge_avg[p] /= num_shifts_total;
    error_p_dd_avg[p] /= num_shifts_total;
    error_p_tr_avg[p] /= num_shifts_total;
    error_p_ex_avg[p] /= num_shifts_total;

    cond_num_avg[p] /= num_shifts_total;
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

    filename = out_dir; filename += "/convergence/error_m_sl_all.txt"; save_vector(filename.c_str(), error_sl_m_arr);
    filename = out_dir; filename += "/convergence/error_m_gr_all.txt"; save_vector(filename.c_str(), error_gr_m_arr);
    filename = out_dir; filename += "/convergence/error_m_ge_all.txt"; save_vector(filename.c_str(), error_ge_m_arr);
    filename = out_dir; filename += "/convergence/error_m_dd_all.txt"; save_vector(filename.c_str(), error_dd_m_arr);
    filename = out_dir; filename += "/convergence/error_m_tr_all.txt"; save_vector(filename.c_str(), error_tr_m_arr);
    filename = out_dir; filename += "/convergence/error_m_ex_all.txt"; save_vector(filename.c_str(), error_ex_m_arr);

    filename = out_dir; filename += "/convergence/error_m_sl_one.txt"; save_vector(filename.c_str(), error_m_sl_one);
    filename = out_dir; filename += "/convergence/error_m_gr_one.txt"; save_vector(filename.c_str(), error_m_gr_one);
    filename = out_dir; filename += "/convergence/error_m_ge_one.txt"; save_vector(filename.c_str(), error_m_ge_one);
    filename = out_dir; filename += "/convergence/error_m_dd_one.txt"; save_vector(filename.c_str(), error_m_dd_one);
    filename = out_dir; filename += "/convergence/error_m_tr_one.txt"; save_vector(filename.c_str(), error_m_tr_one);
    filename = out_dir; filename += "/convergence/error_m_ex_one.txt"; save_vector(filename.c_str(), error_m_ex_one);

    filename = out_dir; filename += "/convergence/error_m_sl_avg.txt"; save_vector(filename.c_str(), error_m_sl_avg);
    filename = out_dir; filename += "/convergence/error_m_gr_avg.txt"; save_vector(filename.c_str(), error_m_gr_avg);
    filename = out_dir; filename += "/convergence/error_m_ge_avg.txt"; save_vector(filename.c_str(), error_m_ge_avg);
    filename = out_dir; filename += "/convergence/error_m_dd_avg.txt"; save_vector(filename.c_str(), error_m_dd_avg);
    filename = out_dir; filename += "/convergence/error_m_tr_avg.txt"; save_vector(filename.c_str(), error_m_tr_avg);
    filename = out_dir; filename += "/convergence/error_m_ex_avg.txt"; save_vector(filename.c_str(), error_m_ex_avg);

    filename = out_dir; filename += "/convergence/error_m_sl_max.txt"; save_vector(filename.c_str(), error_m_sl_max);
    filename = out_dir; filename += "/convergence/error_m_gr_max.txt"; save_vector(filename.c_str(), error_m_gr_max);
    filename = out_dir; filename += "/convergence/error_m_ge_max.txt"; save_vector(filename.c_str(), error_m_ge_max);
    filename = out_dir; filename += "/convergence/error_m_dd_max.txt"; save_vector(filename.c_str(), error_m_dd_max);
    filename = out_dir; filename += "/convergence/error_m_tr_max.txt"; save_vector(filename.c_str(), error_m_tr_max);
    filename = out_dir; filename += "/convergence/error_m_ex_max.txt"; save_vector(filename.c_str(), error_m_ex_max);

    filename = out_dir; filename += "/convergence/error_p_sl_all.txt"; save_vector(filename.c_str(), error_sl_p_arr);
    filename = out_dir; filename += "/convergence/error_p_gr_all.txt"; save_vector(filename.c_str(), error_gr_p_arr);
    filename = out_dir; filename += "/convergence/error_p_ge_all.txt"; save_vector(filename.c_str(), error_ge_p_arr);
    filename = out_dir; filename += "/convergence/error_p_dd_all.txt"; save_vector(filename.c_str(), error_dd_p_arr);
    filename = out_dir; filename += "/convergence/error_p_tr_all.txt"; save_vector(filename.c_str(), error_tr_p_arr);
    filename = out_dir; filename += "/convergence/error_p_ex_all.txt"; save_vector(filename.c_str(), error_ex_p_arr);

    filename = out_dir; filename += "/convergence/error_p_sl_one.txt"; save_vector(filename.c_str(), error_p_sl_one);
    filename = out_dir; filename += "/convergence/error_p_gr_one.txt"; save_vector(filename.c_str(), error_p_gr_one);
    filename = out_dir; filename += "/convergence/error_p_ge_one.txt"; save_vector(filename.c_str(), error_p_ge_one);
    filename = out_dir; filename += "/convergence/error_p_dd_one.txt"; save_vector(filename.c_str(), error_p_dd_one);
    filename = out_dir; filename += "/convergence/error_p_tr_one.txt"; save_vector(filename.c_str(), error_p_tr_one);
    filename = out_dir; filename += "/convergence/error_p_ex_one.txt"; save_vector(filename.c_str(), error_p_ex_one);

    filename = out_dir; filename += "/convergence/error_p_sl_avg.txt"; save_vector(filename.c_str(), error_p_sl_avg);
    filename = out_dir; filename += "/convergence/error_p_gr_avg.txt"; save_vector(filename.c_str(), error_p_gr_avg);
    filename = out_dir; filename += "/convergence/error_p_ge_avg.txt"; save_vector(filename.c_str(), error_p_ge_avg);
    filename = out_dir; filename += "/convergence/error_p_dd_avg.txt"; save_vector(filename.c_str(), error_p_dd_avg);
    filename = out_dir; filename += "/convergence/error_p_tr_avg.txt"; save_vector(filename.c_str(), error_p_tr_avg);
    filename = out_dir; filename += "/convergence/error_p_ex_avg.txt"; save_vector(filename.c_str(), error_p_ex_avg);

    filename = out_dir; filename += "/convergence/error_p_sl_max.txt"; save_vector(filename.c_str(), error_p_sl_max);
    filename = out_dir; filename += "/convergence/error_p_gr_max.txt"; save_vector(filename.c_str(), error_p_gr_max);
    filename = out_dir; filename += "/convergence/error_p_ge_max.txt"; save_vector(filename.c_str(), error_p_ge_max);
    filename = out_dir; filename += "/convergence/error_p_dd_max.txt"; save_vector(filename.c_str(), error_p_dd_max);
    filename = out_dir; filename += "/convergence/error_p_tr_max.txt"; save_vector(filename.c_str(), error_p_tr_max);
    filename = out_dir; filename += "/convergence/error_p_ex_max.txt"; save_vector(filename.c_str(), error_p_ex_max);

    filename = out_dir; filename += "/convergence/cond_num_all.txt"; save_vector(filename.c_str(), cond_num_arr);
    filename = out_dir; filename += "/convergence/cond_num_one.txt"; save_vector(filename.c_str(), cond_num_one);
    filename = out_dir; filename += "/convergence/cond_num_avg.txt"; save_vector(filename.c_str(), cond_num_avg);
    filename = out_dir; filename += "/convergence/cond_num_max.txt"; save_vector(filename.c_str(), cond_num_max);

  }

  w.stop(); w.read_duration();

  return 0;
}

void print_convergence_table(MPI_Comm mpi_comm,
                             std::vector<double> &level, std::vector<double> &h,
                             std::vector<double> &L_one, std::vector<double> &L_avg, std::vector<double> &L_dev, std::vector<double> &L_max,
                             std::vector<double> &Q_one, std::vector<double> &Q_avg, std::vector<double> &Q_dev, std::vector<double> &Q_max)
{
  PetscErrorCode ierr;
  double order;

  ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);

  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "                    |  Linear Integration                                                                     |  Quadratic Integration                                                                \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "                    |  Average                    |  One                        |  Max                        |  Average                    |  One                        |  Max                      \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "lvl  | Resolution   |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order  |  Error      ( Dev )  Order\n"); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);


  for (int i = 0; i < num_resolutions; ++i)
  {
    // lvl and h
    ierr = PetscPrintf(mpi_comm, "%.2f | %.5e", level[i], h[i]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);


    /* linear integration */
    // avg
    if (i == 0) order = compute_convergence_order(h, L_avg);
    else        order = log(L_avg[i-1]/L_avg[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_avg[i], 100.*L_dev[i]/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // one
    if (i == 0) order = compute_convergence_order(h, L_one);
    else        order = log(L_one[i-1]/L_one[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_one[i], 100.*fabs(L_one[i]-L_avg[i])/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // max
    if (i == 0) order = compute_convergence_order(h, L_max);
    else        order = log(L_max[i-1]/L_max[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", L_max[i], 100.*fabs(L_max[i]-L_avg[i])/L_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    /* quadratic integration */
    //avg
    if (i == 0) order = compute_convergence_order(h, Q_avg);
    else        order = log(Q_avg[i-1]/Q_avg[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_avg[i], 100.*Q_dev[i]/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // one
    if (i == 0) order = compute_convergence_order(h, Q_one);
    else        order = log(Q_one[i-1]/Q_one[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_one[i], 100.*fabs(Q_one[i]-Q_avg[i])/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "  |  "); CHKERRXX(ierr);

    // max
    if (i == 0) order = compute_convergence_order(h, Q_max);
    else        order = log(Q_max[i-1]/Q_max[i])/log(h[i-1]/h[i]);

    ierr = PetscPrintf(mpi_comm, "%.2e   (%5.1f) %6.2f", Q_max[i], 100.*fabs(Q_max[i]-Q_avg[i])/Q_avg[i], order); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);
  }
  ierr = PetscPrintf(mpi_comm, "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"); CHKERRXX(ierr);

  ierr = PetscPrintf(mpi_comm, "\n"); CHKERRXX(ierr);
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
