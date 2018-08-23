
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
#include <src/my_p8est_poisson_nonlinear_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
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
#include <src/my_p4est_poisson_nonlinear_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_tools_mls.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

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
const double grid_xyz_min[3] = {-1., -1., -1.};
const double grid_xyz_max[3] = { 1.,  1.,  1.};

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int num_splits = 3;
int num_splits_per_split = 1;
int num_shifts_x_dir = 1;
int num_shifts_y_dir = 1;
int num_shifts_z_dir = 1;
#else
int lmin = 4;
int lmax = 4;
int num_splits = 5;
int num_splits_per_split = 5;
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
// test
//-------------------------------------
int num_phi   = 0;
int num_sol   = 0;
int num_diag  = 0;
int num_mu    = 0;
int num_nlt   = 0;
int num_robin = 0;

BoundaryConditionType bc_wtype = DIRICHLET;
BoundaryConditionType bc_itype = ROBIN;

//-------------------------------------
// solver parameters
//-------------------------------------
int scheme_type = 3; // 0 - simple fixed-point, 1 - linearized fixed-point, 2 - simple gradient-based, 3 - linearized gradient-based
int max_iterations = 10;
double tolerance = 1.e-5;
bool use_non_zero_guess = 1;

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
bool use_phi_cf = 1;
bool reinit     = 0;

//-------------------------------------
// convergence study parameters
//-------------------------------------
bool do_extension = 0;

//-------------------------------------
// output
//-------------------------------------
bool save_vtk           = 0;
bool save_convergence   = 1;

// Level-set function
#ifdef P4_TO_P8
class phi_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return -1;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_cf;
class phi_x_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return 0;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_x_cf;
class phi_y_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return 0;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_y_cf;
class phi_z_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return 0;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_z_cf;
#else
class phi_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_phi)
    {
      case 0: return -1;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_cf;
class phi_x_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return 0;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_x_cf;
class phi_y_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_phi)
    {
      case 0: return 0;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} phi_y_cf;
#endif

// Exact solution
double phase_x =  0.13;
double phase_y =  1.55;
double phase_z =  0.7;

#ifdef P4_TO_P8
class u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_sol)
    {
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 2.*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))-3.;
      case 2: return 4.*log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(y+cos(x-z));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5));
      case 10: return sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} u_cf;
#else
class u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_sol)
    {
      case 0: return sin(x)*cos(y);
      case 1: return 2.*(log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 )-1.5);
      case 2: return 4.*log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5));
      case 10: return (sin(PI*x+phase_x)*sin(PI*y+phase_y));
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} u_cf;
#endif

// EXACT DERIVATIVES
#ifdef P4_TO_P8
class ux_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_sol){
      case 0: return cos(x)*cos(y)*exp(z);
      case 1: return 2.*(1.+2.*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 4.*( log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.) );
      case 3: return exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
      case 4: return ( cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5);
      case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} ux_cf;
class uy_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_sol)
    {
      case 0: return -sin(x)*sin(y)*exp(z);
      case 1: return 2.*(0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 4.*( 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.)) );
      case 3: return exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
      case 4: return ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5);
      case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y)*sin(PI*z+phase_z);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} uy_cf;
class uz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_sol)
    {
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 2.*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 4.*( 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.)) );
      case 3: return exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
      case 4: return cos(x-0.7*y)*sin(x+0.3*y)*exp(z) + 3.*z/(x*x+y*y+z*z+0.5);
      case 10: return PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*cos(PI*z+phase_z);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} uz_cf;
class lap_u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_sol)
    {
      case 0: return -1.0*sin(x)*cos(y)*exp(z);
      case 1: return 4.*( 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
                          -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.) );
      case 2: return 4.*( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
            + 4.*( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
      case 4: return -1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.);
      case 10: return -3.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} lap_u_cf;
#else
class ux_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_sol){
      case 0: return cos(x)*cos(y);
      case 1: return 2.*2.*(x+0.8*y+0.5)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return 4.*( 0.7/(0.7*x+3.) )*sin(x+0.5*y)
            + 4.*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x)-sin(x));
      case 4: return cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y)
            + 3.*x/(x*x+y*y+0.5);
      case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} ux_cf;
class uy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_sol){
      case 0: return -sin(x)*sin(y);
      case 1: return 2.*2.*(0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return 4.*( - 1./(y+3.) )*sin(x+0.5*y)
            + 4.*0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(1.-2.*y*(y+cos(x)));
      case 4: return 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y)
        + 3.*y/(x*x+y*y+0.5);
      case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} uy_cf;

class lap_u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_sol){
      case 0: return -2.0*sin(x)*cos(y);
      case 1: {
        double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
        return 2.*2.*(1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C);
      }
      case 2: return 4.*( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
            + 4.*( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
      case 4: return -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y)
        + 3./pow(x*x+y*y+0.5, 2.);
      case 10: return -2.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} lap_u_cf;
#endif

// Diffusion coefficient
#ifdef P4_TO_P8
class mu_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_mu){
      case 0: return 1.;
      case 1: return 1+(0.2*sin(x)+0.3*cos(y))*cos(z);
      case 2: return 1+0.2*sin(u);
    }
  }
} mu_cf;
class mux_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_mu)
    {
      case 0: return 0.;
      case 1: return 0.2*cos(x)*cos(z);
      case 2: return 0.;
    }
  }
} mux_cf;
class muy_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_mu)
    {
      case 0: return 0.;
      case 1: return -0.3*sin(y)*cos(z);
      case 2: return 0.;
    }
  }
} muy_cf;
class muz_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_mu)
    {
      case 0: return 0.;
      case 1: return -(0.2*sin(x)+0.3*cos(y))*sin(z);
      case 2: return 0.;
    }
  }
} muz_cf;
class mu_u_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_mu){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.2*cos(u);
    }
  }
} mu_u_cf;
#else
class mu_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_mu){
      case 0: return 1.;
      case 1: return 1+0.2*sin(x)+0.3*cos(y);
      case 2: return 1+0.2*sin(u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} mu_cf;
class mux_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_mu){
      case 0: return 0.;
      case 1: return .2*cos(x);
      case 2: return 0.;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} mux_cf;
class muy_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_mu){
      case 0: return 0.;
      case 1: return -0.3*sin(y);
      case 2: return 0.;
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} muy_cf;
class mu_u_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_mu){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.2*cos(u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} mu_u_cf;
#endif

// Diagonal term
#ifdef P4_TO_P8
class diag_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_diag)
    {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x+z)*exp(y);
      case 3: return sin(u);
    }
  }
} diag_cf;
class diag_u_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_diag)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.;
      case 3: return cos(u);
    }
  }
} diag_u_cf;
#else
class diag_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_diag)
    {
      case 0: return 0;
      case 1: return 1.;
      case 2: return sin(x)*exp(y);
      case 3: return sin(u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} diag_cf;
class diag_u_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_diag)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.;
      case 3: return cos(u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} diag_u_cf;
#endif

// Non-linear term
#ifdef P4_TO_P8
class nlt_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_nlt)
    {
      case 0: return 0.;
      case 1: return u*u;
      case 2: return sinh(u);
      case 3: return u/(1.+u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_cf;
class nlt_d_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_nlt)
    {
      case 0: return 0.;
      case 1: return 2.*u;
      case 2: return cosh(u);
      case 3: return 1./(1.+u) - u/pow(1.+u,2.);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_d_cf;
class nlt_dd_cf_t: public CF_4
{
public:
  double operator()(double u, double x, double y, double z) const
  {
    switch (num_nlt)
    {
      case 0: return 0.;
      case 1: return 2.;
      case 2: return sinh(u);
      case 3: return - 2./pow(1.+u,2.) + 2.*u/pow(1.+u,3.);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_dd_cf;
#else
class nlt_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_nlt)
    {
      case 0: return 0;
      case 1: return u*u;
      case 2: return sinh(u);
      case 3: return u/(1.+u);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_cf;
class nlt_d_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_nlt)
    {
      case 0: return 0.;
      case 1: return 2.*u;
      case 2: return cosh(u);
      case 3: return 1./(1.+u) - u/pow(1.+u,2.);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_d_cf;
class nlt_dd_cf_t: public CF_3
{
public:
  double operator()(double u, double x, double y) const
  {
    switch (num_nlt)
    {
      case 0: return 0.;
      case 1: return 2.;
      case 2: return sinh(u);
      case 3: return - 2./pow(1.+u,2.) + 2.*u/pow(1.+u,3.);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} nlt_dd_cf;
#endif

// Robin coeff
#ifdef P4_TO_P8
class bc_icoeff_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_robin)
    {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x+z)*exp(y);
    }
  }
} bc_icoeff_cf;
#else
class bc_icoeff_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_robin)
    {
      case 0: return 0;
      case 1: return 1.;
      case 2: return sin(x)*exp(y);
      default: throw std::invalid_argument("Invalid test number\n");
    }
  }
} bc_icoeff_cf;
#endif

// RHS
#ifdef P4_TO_P8
class rhs_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -mu_cf(x,y,z)*lap_u_cf(x,y,z) + diag_add_cf(x,y,z)*u_cf(x,y,z)
                    - mux_cf(x,y,z)*ux_cf(x,y,z) - muy_cf(x,y,z)*uy_cf(x,y,z) - muz_cf(x,y,z)*uz_cf(x,y,z) + nlt_cf(u_cf(x,y,z),x,y,z);
  }
} rhs_cf;
#else
class rhs_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -mu_cf(x,y)*lap_u_cf(x,y) + diag_add_cf(x,y)*u_cf(x,y)
                    - mux_cf(x,y)*ux_cf(x,y) - muy_cf(x,y)*uy_cf(x,y) + nlt_cf(u_cf(x,y),x,y);
  }
} rhs_cf;
#endif

// BC VALUES
#ifdef P4_TO_P8
class bc_ivalue_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (bc_itype == DIRICHLET)
    {
      return u_cf(x,y,z);
    } else if (bc_itype == NEUMANN || bc_itype == ROBIN) {
      double nx = phi_x_cf(x,y,z);
      double ny = phi_y_cf(x,y,z);
      double nz = phi_z_cf(x,y,z);
      double norm = sqrt(nx*nx+ny*ny+nz*nz);
      nx /= norm; ny /= norm; nz /= norm;
      return mu_cf(x,y)*(nx*ux_cf(x,y,z) +
                         ny*uy_cf(x,y,z) +
                         nz*uz_cf(x,y,z))
          + bc_icoeff_cf(x,y,z)*u_cf(x,y,z);
    }

    return 0;
  }
} bc_ivalue_cf;
#else
class bc_ivalue_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (bc_itype == DIRICHLET)
    {
      return u_cf(x,y);
    } else if (bc_itype == NEUMANN || bc_itype == ROBIN) {
      double nx = phi_x_cf(x,y);
      double ny = phi_y_cf(x,y);
      double norm = sqrt(nx*nx+ny*ny);
      nx /= norm; ny /= norm;
      return mu_cf(x,y)*(nx*ux_cf(x,y) + ny*uy_cf(x,y)) + bc_icoeff_cf(x,y)*u_cf(x,y);
    }

    return 0;
  }
} bc_ivalue_cf;
#endif

#ifdef P4_TO_P8
class bc_wvalue_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (bc_wtype == DIRICHLET)
    {
      return u_cf(x,y,z);
    } else if (bc_wtype == NEUMANN) {
      if (fabs(x-grid_xyz_min[0]) < EPS && !periodicity[0]) { return -mu_cf(x,y,z)*ux_cf(x,y,z); }
      if (fabs(x-grid_xyz_max[0]) < EPS && !periodicity[0]) { return  mu_cf(x,y,z)*ux_cf(x,y,z); }
      if (fabs(y-grid_xyz_min[1]) < EPS && !periodicity[1]) { return -mu_cf(x,y,z)*uy_cf(x,y,z); }
      if (fabs(y-grid_xyz_max[1]) < EPS && !periodicity[1]) { return  mu_cf(x,y,z)*uy_cf(x,y,z); }
      if (fabs(z-grid_xyz_min[2]) < EPS && !periodicity[2]) { return -mu_cf(x,y,z)*uz_cf(x,y,z); }
      if (fabs(z-grid_xyz_max[2]) < EPS && !periodicity[2]) { return  mu_cf(x,y,z)*uz_cf(x,y,z); }
    }

    return 0;
  }
} bc_wvalue_cf;
#else
class bc_wvalue_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (bc_wtype == DIRICHLET)
    {
      return u_cf(x,y);
    } else if (bc_wtype == NEUMANN) {
      if (fabs(x-grid_xyz_min[0]) < EPS && !periodicity[0]) { return -mu_cf(x,y)*ux_cf(x,y); }
      if (fabs(x-grid_xyz_max[0]) < EPS && !periodicity[0]) { return  mu_cf(x,y)*ux_cf(x,y); }
      if (fabs(y-grid_xyz_min[1]) < EPS && !periodicity[1]) { return -mu_cf(x,y)*uy_cf(x,y); }
      if (fabs(y-grid_xyz_max[1]) < EPS && !periodicity[1]) { return  mu_cf(x,y)*uy_cf(x,y); }
    }

    return 0;
  }
} bc_wvalue_cf;
#endif

#ifdef P4_TO_P8
class bc_wall_type_t : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;
#else
class bc_wall_type_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;
#endif

// additional output functions
double compute_convergence_order(std::vector<double> &x, std::vector<double> &y);

int main (int argc, char* argv[])
{
  // error variables
  PetscErrorCode ierr;
  int mpiret;

  // mpi
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  for (int i = 0; i < 2; i++)
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

    ADD_OPTION(i, iter_start, "skip n first iterations");

    //-------------------------------------
    // output
    //-------------------------------------
    ADD_OPTION(i, save_vtk,         "Save the p4est in vtk format");
    ADD_OPTION(i, save_convergence, "Save convergence results");

    //-------------------------------------
    // test solution
    //-------------------------------------
    ADD_OPTION(i, num_phi,   "");
    ADD_OPTION(i, num_sol,   "");
    ADD_OPTION(i, num_diag,  "");
    ADD_OPTION(i, num_mu,    "");
    ADD_OPTION(i, num_nlt,   "");
    ADD_OPTION(i, num_robin, "");

    ADD_OPTION(i, bc_wtype, "Type of boundary conditions on the walls");
    ADD_OPTION(i, bc_itype, "Type of boundary conditions on the interface");

    //-------------------------------------
    // solver parameters
    //-------------------------------------
    ADD_OPTION(i, sc_scheme,         "Use super-convergent scheme");
    ADD_OPTION(i, integration_order, "Select integration order (1 - linear, 2 - quadratic)");

    // for symmetric scheme:
    ADD_OPTION(i, taylor_correction,      "Use Taylor correction to approximate Robin term (symmetric scheme)");
    ADD_OPTION(i, kink_special_treatment, "Use the special treatment for kinks (symmetric scheme)");

    // for superconvergent scheme:
    ADD_OPTION(i, try_remove_hanging_cells, "Ask solver to eliminate hanging cells");

    //-------------------------------------
    // convergence study options
    //-------------------------------------
    ADD_OPTION(i, do_extension, "Extend solution after solving");

    //-------------------------------------
    // level-set representation parameters
    //-------------------------------------
    ADD_OPTION(i, use_phi_cf, "Use analytical level-set functions");
    ADD_OPTION(i, reinit,     "Reinitialize level-set function");

    if (i == 1) cmd.parse(argc, argv);
  }

  // recalculate depending parameters
  num_shifts_total = num_shifts_x_dir*num_shifts_y_dir*num_shifts_z_dir;

  num_resolutions = (num_splits-1)*num_splits_per_split + 1;
  num_iter_total = num_resolutions*num_shifts_total;

  // prepare output directories
  const char* out_dir = getenv("OUT_DIR");

  if (!out_dir &&
      (save_vtk ||
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

  vector<double> error_sl_arr;
  vector<double> error_gr_arr;
  vector<double> error_dd_arr;
  vector<double> error_ex_arr;

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

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

            splitting_criteria_cf_t data_tmp(lmin, lmax, &phi_cf, 1.4);
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

//            my_p4est_partition(p4est, P4EST_FALSE, NULL);
//            p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
//            my_p4est_partition(p4est, P4EST_FALSE, NULL);

            ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//            my_p4est_ghost_expand(p4est, ghost);
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

            // sample level-set function
            Vec phi;
            ierr = VecCreateGhostNodes(p4est, nodes, phi); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, phi_cf, phi);

            if (reinit_level_set)
              ls.reinitialize_1st_order_time_2nd_order_space(phi, 20);

            Vec rhs;
            ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);

//            Vec u_exact_vec;
//            ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
//            sample_cf_on_nodes(p4est, nodes, u_cf, u_exact_vec);

#ifdef P4_TO_P8
            BoundaryConditions3D bc;
#else
            BoundaryConditions2D bc;
#endif

            bc.setInterfaceType(bc_itype);
            bc.setInterfaceValue(bc_ivalue);
            bc.setInterfaceCoeff(bc_ivalue);
            bc.setWallTypes(bc_wtype);
            bc.setWallValues(bc_wvalue);

            ierr = PetscPrintf(mpi.comm(), "Starting a solver\n"); CHKERRXX(ierr);

            Vec sol; double *sol_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);

            my_p4est_poisson_nonlinear_nodes_t solver(&ngbd_n);

            solver.set_phi(phi);
            solver.set_rhs(rhs);
            solver.set_mu(mu_cf, mu_u_cf);
            solver.set_diag(diag_cf, diag_u_cf);
            solver.set_nlt(nlt_cf, nlt_u_cf, nlt_u_cf);
            solver.set_bc(bc);

            solver.set_scheme(scheme_type);
            solver.set_tolerance(tolerace, max_iterations);

            solver.set_use_sc_scheme(sc_scheme);
            solver.set_integration_order(integration_order);
            solver.set_use_taylor_correction(taylor_correction);
            solver.set_kink_treatment(kink_special_treatment);
            solver.set_try_remove_hanging_cells(try_remove_hanging_cells);

            solver.set_use_non_zero_guess(use_non_zero_guess);
            solver.solve(sol);

            Vec mask = solver.get_mask();
            Vec res  = solver.get_residual();

            /* calculate errors */
            Vec vec_error_sl; double *vec_error_sl_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl); CHKERRXX(ierr);
            Vec vec_error_gr; double *vec_error_gr_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr); CHKERRXX(ierr);
            Vec vec_error_ex; double *vec_error_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex); CHKERRXX(ierr);
            Vec vec_error_dd; double *vec_error_dd_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd); CHKERRXX(ierr);

            //----------------------------------------------------------------------------------------------
            // calculate error of solution
            //----------------------------------------------------------------------------------------------
            ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);

            double *mask_ptr;
            ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

            for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
            {
              if (mask_ptr[n] < mask_thresh)
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
              if ( mask_ptr[qnnn.node_000] < mask_thresh && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
                   ( mask_ptr[qnnn.node_m00_mm]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_mp]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pm]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pp]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mm]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mp]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pm]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pp]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mm]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mp]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pm]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pp]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mm]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mp]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pm]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pp]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_mm]<mask_thresh || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_mp]<mask_thresh || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_pm]<mask_thresh || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_pp]<mask_thresh || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_mm]<mask_thresh || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_mp]<mask_thresh || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_pm]<mask_thresh || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_pp]<mask_thresh || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
                   ( mask_ptr[qnnn.node_m00_mm]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pm]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mm]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pm]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mm]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pm]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mm]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pm]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS)
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
            // calculate extrapolation error
            //----------------------------------------------------------------------------------------------
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
            if (do_extension) { ls.extend_Over_Interface_TVD(phi, mask, sol_ex, 100, 2); CHKERRXX(ierr); }

            // calculate error
            ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
            ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

            double *phi_ptr;
            ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

            for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
            {
              if (mask_ptr[n] > mask_thresh && phi_ptr[n] < band*dxyz_max)
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
            ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);

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
              if ( mask_ptr[qnnn.node_000]<mask_thresh && !is_node_Wall(p4est, ni) &&
     #ifdef P4_TO_P8
                   ( mask_ptr[qnnn.node_m00_mm]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_mp]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pm]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pp]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mm]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mp]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pm]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pp]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mm]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mp]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pm]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pp]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mm]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mp]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pm]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pp]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_mm]<mask_thresh || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_mp]<mask_thresh || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_pm]<mask_thresh || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00m_pp]<mask_thresh || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_mm]<mask_thresh || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_mp]<mask_thresh || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_pm]<mask_thresh || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
                   ( mask_ptr[qnnn.node_00p_pp]<mask_thresh || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
                   ( mask_ptr[qnnn.node_m00_mm]<mask_thresh || fabs(qnnn.d_m00_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_m00_pm]<mask_thresh || fabs(qnnn.d_m00_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_mm]<mask_thresh || fabs(qnnn.d_p00_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_p00_pm]<mask_thresh || fabs(qnnn.d_p00_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_mm]<mask_thresh || fabs(qnnn.d_0m0_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_0m0_pm]<mask_thresh || fabs(qnnn.d_0m0_m0)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_mm]<mask_thresh || fabs(qnnn.d_0p0_p0)<EPS) &&
                   ( mask_ptr[qnnn.node_0p0_pm]<mask_thresh || fabs(qnnn.d_0p0_m0)<EPS)
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

            // compute L-inf norm of errors
            double err_sl_max = 0.; VecMax(vec_error_sl, NULL, &err_sl_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
            double err_gr_max = 0.; VecMax(vec_error_gr, NULL, &err_gr_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
            double err_ex_max = 0.; VecMax(vec_error_ex, NULL, &err_ex_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
            double err_dd_max = 0.; VecMax(vec_error_dd, NULL, &err_dd_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

            // Store error values
            error_sl_arr.push_back(err_sl_max);
            error_gr_arr.push_back(err_gr_max);
            error_ex_arr.push_back(err_ex_max);
            error_dd_arr.push_back(err_dd_max);

            // Print current errors
            if (iter > -1)
            {
              ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f). Iteration %6d / %6d \n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale, iteration, num_iter_total); CHKERRXX(ierr);
              ierr = PetscPrintf(mpi.comm(), "Error (sl): %g\n", err_sl_max); CHKERRXX(ierr);
              ierr = PetscPrintf(mpi.comm(), "Error (gr): %g\n", err_gr_max); CHKERRXX(ierr);
              ierr = PetscPrintf(mpi.comm(), "Error (dd): %g\n", err_dd_max); CHKERRXX(ierr);
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

              double *phi_ptr;
              ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
              ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

              double *mask_ptr;
              ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

              double *res_ptr;
              ierr = VecGetArray(res, &res_ptr); CHKERRXX(ierr);

              my_p4est_vtk_write_all(p4est, nodes, ghost,
                                     P4EST_TRUE, P4EST_TRUE,
                                     9, 1, oss.str().c_str(),
                                     VTK_POINT_DATA, "phi", phi_ptr,
                                     VTK_POINT_DATA, "sol", sol_ptr,
                                     VTK_POINT_DATA, "sol_ex", sol_ex_ptr,
                                     VTK_POINT_DATA, "error_sl", vec_error_sl_ptr,
                                     VTK_POINT_DATA, "error_gr", vec_error_gr_ptr,
                                     VTK_POINT_DATA, "error_ex", vec_error_ex_ptr,
                                     VTK_POINT_DATA, "error_dd", vec_error_dd_ptr,
                                     VTK_POINT_DATA, "mask", mask_ptr,
                                     VTK_POINT_DATA, "residual", res_ptr,
                                     VTK_CELL_DATA , "leaf_level", l_p);

              ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(vec_error_gr, &vec_error_gr_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

              ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
              ierr = VecRestoreArray(res, &res_ptr); CHKERRXX(ierr);

              ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
              ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

              PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
            }

            // destroy Vec's with errors
            ierr = VecDestroy(vec_error_sl); CHKERRXX(ierr);
            ierr = VecDestroy(vec_error_gr); CHKERRXX(ierr);
            ierr = VecDestroy(vec_error_ex); CHKERRXX(ierr);
            ierr = VecDestroy(vec_error_dd); CHKERRXX(ierr);

            ierr = VecDestroy(sol_ex); CHKERRXX(ierr);
            ierr = VecDestroy(phi); CHKERRXX(ierr);

            ierr = VecDestroy(sol);         CHKERRXX(ierr);
            ierr = VecDestroy(rhs);         CHKERRXX(ierr);

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

  std::vector<double> error_sl_one(num_resolutions, 0), error_sl_avg(num_resolutions, 0), error_sl_max(num_resolutions, 0);
  std::vector<double> error_gr_one(num_resolutions, 0), error_gr_avg(num_resolutions, 0), error_gr_max(num_resolutions, 0);
  std::vector<double> error_dd_one(num_resolutions, 0), error_dd_avg(num_resolutions, 0), error_dd_max(num_resolutions, 0);
  std::vector<double> error_ex_one(num_resolutions, 0), error_ex_avg(num_resolutions, 0), error_ex_max(num_resolutions, 0);

  // for each resolution compute max, mean and deviation
  for (int p = 0; p < num_resolutions; ++p)
  {
    // one
    error_sl_one[p] = error_sl_arr[p*num_shifts_total];
    error_gr_one[p] = error_gr_arr[p*num_shifts_total];
    error_dd_one[p] = error_dd_arr[p*num_shifts_total];
    error_ex_one[p] = error_ex_arr[p*num_shifts_total];

    // max
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_sl_max[p] = MAX(error_sl_max[p], error_sl_arr[p*num_shifts_total + s]);
      error_gr_max[p] = MAX(error_gr_max[p], error_gr_arr[p*num_shifts_total + s]);
      error_dd_max[p] = MAX(error_dd_max[p], error_dd_arr[p*num_shifts_total + s]);
      error_ex_max[p] = MAX(error_ex_max[p], error_ex_arr[p*num_shifts_total + s]);
    }

    // avg
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_sl_avg[p] += error_sl_arr[p*num_shifts_total + s];
      error_gr_avg[p] += error_gr_arr[p*num_shifts_total + s];
      error_dd_avg[p] += error_dd_arr[p*num_shifts_total + s];
      error_ex_avg[p] += error_ex_arr[p*num_shifts_total + s];
    }

    error_sl_avg[p] /= num_shifts_total;
    error_gr_avg[p] /= num_shifts_total;
    error_dd_avg[p] /= num_shifts_total;
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

    filename = out_dir; filename += "/convergence/error_sl_all.txt"; save_vector(filename.c_str(), error_sl_arr);
    filename = out_dir; filename += "/convergence/error_gr_all.txt"; save_vector(filename.c_str(), error_gr_arr);
    filename = out_dir; filename += "/convergence/error_dd_all.txt"; save_vector(filename.c_str(), error_dd_arr);
    filename = out_dir; filename += "/convergence/error_ex_all.txt"; save_vector(filename.c_str(), error_ex_arr);

    filename = out_dir; filename += "/convergence/error_sl_one.txt"; save_vector(filename.c_str(), error_sl_one);
    filename = out_dir; filename += "/convergence/error_gr_one.txt"; save_vector(filename.c_str(), error_gr_one);
    filename = out_dir; filename += "/convergence/error_dd_one.txt"; save_vector(filename.c_str(), error_dd_one);
    filename = out_dir; filename += "/convergence/error_ex_one.txt"; save_vector(filename.c_str(), error_ex_one);

    filename = out_dir; filename += "/convergence/error_sl_avg.txt"; save_vector(filename.c_str(), error_sl_avg);
    filename = out_dir; filename += "/convergence/error_gr_avg.txt"; save_vector(filename.c_str(), error_gr_avg);
    filename = out_dir; filename += "/convergence/error_dd_avg.txt"; save_vector(filename.c_str(), error_dd_avg);
    filename = out_dir; filename += "/convergence/error_ex_avg.txt"; save_vector(filename.c_str(), error_ex_avg);

    filename = out_dir; filename += "/convergence/error_sl_max.txt"; save_vector(filename.c_str(), error_sl_max);
    filename = out_dir; filename += "/convergence/error_gr_max.txt"; save_vector(filename.c_str(), error_gr_max);
    filename = out_dir; filename += "/convergence/error_dd_max.txt"; save_vector(filename.c_str(), error_dd_max);
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
