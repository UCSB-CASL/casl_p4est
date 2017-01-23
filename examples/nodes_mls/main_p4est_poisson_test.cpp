
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
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/simplex3_mls_vtk.h>
#include <src/my_p8est_semi_lagrangian.h>
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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/simplex2_mls_vtk.h>
#include <src/my_p4est_semi_lagrangian.h>
#endif

#include <src/point3.h>
#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#include "my_p4est_mls_tools.h"
#include "problem_case_0.h" // triangle (tetrahedron)
#include "problem_case_1.h" // two circles union
#include "problem_case_2.h" // two circles intersection
#include "problem_case_3.h" // two circles coloration
#include "problem_case_4.h" // four flowers
#include "problem_case_5.h" // two circles coloration (naive)
#include "problem_case_6.h" // one flower
#include "problem_case_7.h" // three flowers


#undef MIN
#undef MAX

using namespace std;

bool save_vtk = true;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 6;
int nb_splits = 4;
#else
int lmin = 4;
int lmax = 4;
int nb_splits = 7;
#endif

const int periodic[3] = {0, 0, 0};
const int n_xyz[3] = {1, 1, 1};
//const double p_xyz_min[3] = {-1, -1, -1};
//const double p_xyz_max[3] = {1, 1, 1};
const double p_xyz_min[3] = {-2, -2, -2};
const double p_xyz_max[3] = {2, 2, 2};

/* Examples for Poisson paper
 * 0000
 * 1100
 * 2211
 * 3310
 * 4412
 * 7412
 */

int n_geometry = 6;
int n_test = 0;
int n_mu = 0;
int n_diag_add = 0;

bool reinitialize_lsfs = true;


// EXACT SOLUTION
double phase_x =  0.13;
double phase_y =  1.55;
double phase_z =  0.7;
#ifdef P4_TO_P8
class u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 0.5*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(y+cos(x-z));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5));
      case 10: return sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} u_cf;
#else
class u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y);
      case 1: return 0.5*log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5));
      case 10: return (sin(PI*x+phase_x)*sin(PI*y+phase_y));
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
    switch (n_test){
      case 0: return cos(x)*cos(y)*exp(z);
      case 1: return 0.5*(1.+2.*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.);
      case 3: return exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
      case 4: return ( cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5);
    case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} ux_cf;
class uy_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return -sin(x)*sin(y)*exp(z);
      case 1: return 0.5*(0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.));
      case 3: return exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
      case 4: return ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5);
    case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} uy_cf;
class uz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 0.5*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.));
      case 3: return exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
      case 4: return cos(x-0.7*y)*sin(x+0.3*y)*exp(z) + 3.*z/(x*x+y*y+z*z+0.5);
    case 10: return PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*cos(PI*z+phase_z);
    }
  }
} uz_cf;
class lap_u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return -1.0*sin(x)*cos(y)*exp(z);
      case 1: return 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
            -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.);
      case 2: return ( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
            + ( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
      case 4: return -1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.);
    case 10: return -3.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} lap_u_cf;
#else
class ux_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return cos(x)*cos(y);
      case 1: return (x+0.8*y+0.5)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return ( 0.7/(0.7*x+3.) )*sin(x+0.5*y)
            + ( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x)-sin(x));
      case 4: return cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y)
            + 3.*x/(x*x+y*y+0.5);
      case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} ux_cf;
class uy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -sin(x)*sin(y);
      case 1: return (0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return ( - 1./(y+3.) )*sin(x+0.5*y)
            + 0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(1.-2.*y*(y+cos(x)));
      case 4: return 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y)
        + 3.*y/(x*x+y*y+0.5);
      case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y);
    }
  }
} uy_cf;

class lap_u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -2.0*sin(x)*cos(y);
      case 1: {
        double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
        return 1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C;
      }
      case 2: return ( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
            + ( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
      case 4: return -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y)
        + 3./pow(x*x+y*y+0.5, 2.);
      case 10: return -2.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} lap_u_cf;
#endif

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


vector<double> level, h;

vector<double> error_sl_arr, error_sl_l1_arr;
vector<double> error_ex_arr, error_ex_l1_arr;
vector<double> error_dd_arr, error_dd_l1_arr;
vector<double> error_tr_arr, error_tr_l1_arr;
vector<double> error_ux_arr, error_ux_l1_arr;
vector<double> error_uy_arr, error_uy_l1_arr;
#ifdef P4_TO_P8
vector<double> error_uz_arr, error_uz_l1_arr;
#endif


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_trunc, Vec err_grad,
              int compt);

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

  connectivity = my_p4est_brick_new(n_xyz, p_xyz_min, p_xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  // an effective LSF
  level_set_tot_t level_set_tot_cf(&phi_cf, &action, &color);

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot_cf, 1.4);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
//    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

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
    std::vector<Vec> bc_values;
    for (int i = 0; i < num_surfaces; i++)
    {
#ifdef P4_TO_P8
      bc_value_robin_t bc_val(&u_cf, &ux_cf, &uy_cf, &uz_cf, &mu_cf, phi_x_cf[i], phi_y_cf[i], phi_z_cf[i], bc_coeffs_cf[i]);
#else
      bc_value_robin_t bc_val(&u_cf, &ux_cf, &uy_cf, &mu_cf, phi_x_cf[i], phi_y_cf[i], bc_coeffs_cf[i]);
#endif

      bc_values.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &bc_values.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, bc_val, bc_values.back());
    }

    // sample Robin coefficients
    std::vector<Vec> bc_coeffs;
    for (int i = 0; i < num_surfaces; i++)
    {
      bc_coeffs.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *bc_coeffs_cf[i], bc_coeffs.back());
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

    std::vector<BoundaryConditionType> bc_types(num_surfaces, ROBIN);

//    ierr = VecDestroy(rhs); CHKERRXX(ierr);
//    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Starting a solver\n"); CHKERRXX(ierr);

    my_p4est_poisson_nodes_mls_t solver(&ngbd_n);
    solver.set_geometry(phi, action, color);
    solver.set_mu(mu);
    solver.set_rhs(rhs);
    solver.wall_value.set(u_exact_vec);
    solver.set_bc_type(bc_types);
    solver.set_diag_add(diag_add);
    solver.set_bc_coeffs(bc_coeffs);
    solver.set_bc_values(bc_values);
    solver.set_use_taylor_correction(true);
    solver.set_keep_scalling(true);
    solver.set_kinks_treatment(true);

    solver.compute_volumes();

//    ierr = PetscPrintf(p4est->mpicomm, "Here\n"); CHKERRXX(ierr);

    Vec sol; double *sol_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);

    solver.solve(sol);

    my_p4est_integration_mls_t integrator;
    integrator.set_p4est(p4est, nodes);
#ifdef P4_TO_P8
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, *solver.phi_zz, action, color);
#else
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, action, color);
#endif
    if (save_vtk)
    {
      integrator.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integrator.cubes.size(); k++)
        if (integrator.cubes[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integrator.cubes[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#endif
    }

    /* calculate errors */
    Vec vec_error_sl; double *vec_error_sl_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl); CHKERRXX(ierr);
    Vec vec_error_tr; double *vec_error_tr_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_tr); CHKERRXX(ierr);
    Vec vec_error_ux; double *vec_error_ux_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ux); CHKERRXX(ierr);
    Vec vec_error_uy; double *vec_error_uy_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_uy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec vec_error_uz; double *vec_error_uz_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_uz); CHKERRXX(ierr);
#endif
    Vec vec_error_ex; double *vec_error_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex); CHKERRXX(ierr);
    Vec vec_error_dd; double *vec_error_dd_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd); CHKERRXX(ierr);

    // calculate error of solution
    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (solver.is_calc(n))
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

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_sl, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_sl, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    // calculate error of gradients
    ierr = VecGetArray(sol, &sol_ptr);    CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_ux, &vec_error_ux_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_uy, &vec_error_uy_ptr); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecGetArray(vec_error_uz, &vec_error_uz_ptr); CHKERRXX(ierr);
  #endif

    double *node_vol_p; ierr = VecGetArray(solver.node_vol,  &node_vol_p); CHKERRXX(ierr);

    p4est_locidx_t neighbors[N_NBRS_MAX];
    p4est_locidx_t neighbors_of_n[N_NBRS_MAX];
    bool neighbor_exists[N_NBRS_MAX];

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
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

      bool nm1_available;
      bool np1_available;

      const quad_neighbor_nodes_of_node_t qnnn = ngbd_n.get_neighbors(n);

      if (!solver.is_calc(n))
      {
        vec_error_ux_ptr[n] = 0;
        vec_error_uy_ptr[n] = 0;
  #ifdef P4_TO_P8
        vec_error_uz_ptr[n] = 0;
  #endif
      } else if (solver.is_inside(n)) {

        vec_error_ux_ptr[n] = fabs(qnnn.dx_central(sol_ptr) - ux_exact);
        vec_error_uy_ptr[n] = fabs(qnnn.dy_central(sol_ptr) - uy_exact);
  #ifdef P4_TO_P8
        vec_error_uz_ptr[n] = fabs(qnnn.dz_central(sol_ptr) - uz_exact);
  #endif
      } else if (solver.is_calc(n)) {
        solver.get_all_neighbors(n, neighbors, neighbor_exists);

        // compute x-component
        if (neighbor_exists[nn_m00]) nm1_available = node_vol_p[neighbors[nn_m00]] > solver.eps_dom; else nm1_available = false;
        if (neighbor_exists[nn_p00]) np1_available = node_vol_p[neighbors[nn_p00]] > solver.eps_dom; else np1_available = false;

        if (nm1_available && np1_available) { // using central differences

          vec_error_ux_ptr[n] = fabs(qnnn.dx_central(sol_ptr) - ux_exact);

  //      } else if (node_vol_p[neighbors[nn_m00]] > eps_dom) { // using backward differences

  //        get_all_neighbors(neighbors[nn_m00], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_m00]] > eps_dom)
  //          err_ux_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_m00]] + 0.5*sol_p[neighbors_of_n[nn_m00]])/dx_min - ux_exact);
  //        else
  //          err_ux_p[n] = 0.;

  //      } else if (node_vol_p[neighbors[nn_p00]] > eps_dom) { // using forward differences

  //        get_all_neighbors(neighbors[nn_p00], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_p00]] > eps_dom)
  //          err_ux_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_p00]] - 0.5*sol_p[neighbors_of_n[nn_p00]])/dx_min - ux_exact);
  //        else
  //          err_ux_p[n] = 0.;

        } else vec_error_ux_ptr[n] = 0.;

        // compute y-component
        if (neighbor_exists[nn_0m0]) nm1_available = node_vol_p[neighbors[nn_0m0]] > solver.eps_dom; else nm1_available = false;
        if (neighbor_exists[nn_0p0]) np1_available = node_vol_p[neighbors[nn_0p0]] > solver.eps_dom; else np1_available = false;

        if (nm1_available && np1_available) { // using central differences

          vec_error_uy_ptr[n] = fabs(qnnn.dy_central(sol_ptr) - uy_exact);

  //      } else if (node_vol_p[neighbors[nn_0m0]] > eps_dom) { // using backward differences

  //        get_all_neighbors(neighbors[nn_0m0], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_0m0]] > eps_dom)
  //          err_uy_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_0m0]] + 0.5*sol_p[neighbors_of_n[nn_0m0]])/dy_min - uy_exact);
  //        else
  //          err_uy_p[n] = 0.;

  //      } else if (node_vol_p[neighbors[nn_0p0]] > eps_dom) { // using forward differences

  //        get_all_neighbors(neighbors[nn_0p0], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_0p0]] > eps_dom)
  //          err_uy_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_0p0]] - 0.5*sol_p[neighbors_of_n[nn_0p0]])/dy_min - uy_exact);
  //        else
  //          err_uy_p[n] = 0.;

        } else vec_error_uy_ptr[n] = 0;

  #ifdef P4_TO_P8
        // compute z-component
        if (neighbor_exists[nn_00m]) nm1_available = node_vol_p[neighbors[nn_00m]] > solver.eps_dom; else nm1_available = false;
        if (neighbor_exists[nn_00p]) np1_available = node_vol_p[neighbors[nn_00p]] > solver.eps_dom; else np1_available = false;

        if (nm1_available && np1_available) { // using central differences

          vec_error_uz_ptr[n] = fabs(qnnn.dz_central(sol_ptr) - uz_exact);

  //      } else if (node_vol_p[neighbors[nn_00m]] > eps_dom) { // using backward differences

  //        get_all_neighbors(neighbors[nn_00m], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_00m]] > eps_dom)
  //          err_uz_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_00m]] + 0.5*sol_p[neighbors_of_n[nn_00m]])/dz_min - uz_exact);
  //        else
  //          err_uz_p[n] = 0.;

  //      } else if (node_vol_p[neighbors[nn_00p]] > eps_dom) { // using forward differences

  //        get_all_neighbors(neighbors[nn_00p], neighbors_of_n, neighbor_exists);
  //        if (node_vol_p[neighbors_of_n[nn_00p]] > eps_dom)
  //          err_uz_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_00p]] - 0.5*sol_p[neighbors_of_n[nn_00p]])/dz_min - uz_exact);
  //        else
  //          err_uz_p[n] = 0.;

        } else vec_error_uz_ptr[n] = 0;
  #endif
      }
    }

    ierr = VecRestoreArray(sol, &sol_ptr);    CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_ux, &vec_error_ux_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_error_uy, &vec_error_uy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(vec_error_uz, &vec_error_uz_ptr); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(solver.node_vol,  &node_vol_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vec_error_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateBegin(vec_error_uz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
    ierr = VecGhostUpdateEnd  (vec_error_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGhostUpdateEnd  (vec_error_uz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

    // calculate truncation error
    Vec vec_u_exact; ierr = VecCreateGhostNodes(p4est, nodes, &vec_u_exact);   CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, u_cf, vec_u_exact);

    ierr = MatMult(solver.A, vec_u_exact, vec_error_tr); CHKERRXX(ierr);
    ierr = VecAXPY(vec_error_tr, -1., solver.rhs); CHKERRXX(ierr);
    ierr = VecPointwiseMult(vec_error_tr, vec_error_tr, solver.scalling); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
      if (!solver.is_calc(n)) vec_error_tr_ptr[n] = 0.;

    ierr = VecRestoreArray(vec_error_tr, &vec_error_tr_ptr);  CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_tr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_tr, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(vec_u_exact); CHKERRXX(ierr);

    // calculate extrapolation error

    // smoothed LSF
    level_set_smooth_t level_set_smooth_cf(&phi_cf, &action, &color, 14.*dxyz_max*dxyz_max);

    Vec phi_smooth; double *phi_smooth_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &phi_smooth); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_smooth_cf, phi_smooth);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_smooth);

    double band = 4.0;

    // copy solution into a new Vec
    Vec sol_ex; double *sol_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_ex); CHKERRXX(ierr);

    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
      sol_ex_ptr[i] = sol_ptr[i];

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

    // extend
    ls.extend_Over_Interface_TVD(phi_smooth, sol_ex); CHKERRXX(ierr);

    // calculate error
    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
//      if (!solver.is_calc(n) && phi_smooth_ptr[n] < band*dxyz_max)
      if (phi_smooth_ptr[n] > 0. && phi_smooth_ptr[n] < band*dxyz_max)
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

    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // calculate error of Laplacian
    Vec vec_uxx; double *vec_uxx_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uxx); CHKERRXX(ierr);
    Vec vec_uyy; double *vec_uyy_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec vec_uzz; double *vec_uzz_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_uzz); CHKERRXX(ierr);
    ngbd_n.second_derivatives_central(sol_ex, vec_uxx, vec_uyy, vec_uzz); CHKERRXX(ierr);
#else
    ngbd_n.second_derivatives_central(sol_ex, vec_uxx, vec_uyy); CHKERRXX(ierr);
#endif

    double phi_shift = diag;

    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      phi_smooth_ptr[i] += phi_shift;
    }
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    ls.extend_Over_Interface_TVD(phi_smooth, vec_uxx); CHKERRXX(ierr);
    ls.extend_Over_Interface_TVD(phi_smooth, vec_uyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ls.extend_Over_Interface_TVD(phi_smooth, vec_uzz); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      phi_smooth_ptr[i] -= phi_shift;
    }
    ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);

    ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_uxx, &vec_uxx_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(vec_uyy, &vec_uyy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(vec_uzz, &vec_uzz_ptr); CHKERRXX(ierr);
#endif

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (solver.is_calc(n))
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
#ifdef P4_TO_P8
        vec_error_dd_ptr[n] = ABS(vec_uxx_ptr[n] + vec_uyy_ptr[n] + vec_uzz_ptr[n] - lap_u_cf(xyz[0],xyz[1],xyz[2]));
#else
        vec_error_dd_ptr[n] = ABS(vec_uxx_ptr[n] + vec_uyy_ptr[n] - lap_u_cf(xyz[0],xyz[1]));
#endif
      }
      else
        vec_error_dd_ptr[n] = 0;
    }

    ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_uxx, &vec_uxx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vec_uyy, &vec_uyy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(vec_uzz, &vec_uzz_ptr); CHKERRXX(ierr);
#endif

    ierr = VecDestroy(vec_uxx); CHKERRXX(ierr);
    ierr = VecDestroy(vec_uyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(vec_uzz); CHKERRXX(ierr);
#endif

    ierr = VecGhostUpdateBegin(vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vec_error_dd, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // compute L-inf norm of errors
    double err_sl_max = 0.; VecMax(vec_error_sl, NULL, &err_sl_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_tr_max = 0.; VecMax(vec_error_tr, NULL, &err_tr_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_ux_max = 0.; VecMax(vec_error_ux, NULL, &err_ux_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ux_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_uy_max = 0.; VecMax(vec_error_uy, NULL, &err_uy_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_uy_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
#ifdef P4_TO_P8
    double err_uz_max = 0.; VecMax(vec_error_uz, NULL, &err_uz_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_uz_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
#endif
    double err_ex_max = 0.; VecMax(vec_error_ex, NULL, &err_ex_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double err_dd_max = 0.; VecMax(vec_error_dd, NULL, &err_dd_max); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    // compute L1 errors
    double measure_of_dom = integrator.measure_of_domain();
    error_sl_l1_arr.push_back(integrator.integrate_everywhere(vec_error_sl)/measure_of_dom);
    error_tr_l1_arr.push_back(integrator.integrate_everywhere(vec_error_tr)/measure_of_dom);
    error_ux_l1_arr.push_back(integrator.integrate_everywhere(vec_error_ux)/measure_of_dom);
    error_uy_l1_arr.push_back(integrator.integrate_everywhere(vec_error_uy)/measure_of_dom);
#ifdef P4_TO_P8
    error_uz_l1_arr.push_back(integrator.integrate_everywhere(vec_error_uz)/measure_of_dom);
#endif
    error_ex_l1_arr.push_back(integrator.integrate_everywhere(vec_error_ex)/measure_of_dom);
    error_dd_l1_arr.push_back(integrator.integrate_everywhere(vec_error_dd)/measure_of_dom);

    // Store error values
    level.push_back(lmin+iter);
    h.push_back(dxyz_max*pow(2.,(double) data.max_lvl - data.min_lvl));
    error_sl_arr.push_back(err_sl_max);
    error_tr_arr.push_back(err_tr_max);
    error_ux_arr.push_back(err_ux_max);
    error_uy_arr.push_back(err_uy_max);
#ifdef P4_TO_P8
    error_uz_arr.push_back(err_uz_max);
#endif
    error_ex_arr.push_back(err_ex_max);
    error_dd_arr.push_back(err_dd_max);

    // Print current errors
    if (iter > 0)
    {
      ierr = PetscPrintf(p4est->mpicomm, "Error (sl): %g, order = %g\n", err_sl_max, log(error_sl_arr[iter-1]/error_sl_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (tr): %g, order = %g\n", err_tr_max, log(error_tr_arr[iter-1]/error_tr_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (ux): %g, order = %g\n", err_ux_max, log(error_ux_arr[iter-1]/error_ux_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (uy): %g, order = %g\n", err_uy_max, log(error_uy_arr[iter-1]/error_uy_arr[iter])/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(p4est->mpicomm, "Error (uz): %g, order = %g\n", err_uz_max, log(error_uz_arr[iter-1]/error_uz_arr[iter])/log(2)); CHKERRXX(ierr);
#endif
      ierr = PetscPrintf(p4est->mpicomm, "Error (ex): %g, order = %g\n", err_ex_max, log(error_ex_arr[iter-1]/error_ex_arr[iter])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error (dd): %g, order = %g\n", err_dd_max, log(error_dd_arr[iter-1]/error_dd_arr[iter])/log(2)); CHKERRXX(ierr);
    }

    /* extrapolate the solution and check accuracy */

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

      /* mask for solution */
      Vec mask; double *mask_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &mask); CHKERRXX(ierr);

      ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if (solver.is_calc(n))  mask_ptr[n] = 1;
        else                    mask_ptr[n] = 0;
      }

      ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (mask, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

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
      ierr = VecGetArray(solver.phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_ux, &vec_error_ux_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_uy, &vec_error_uy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(vec_error_uz, &vec_error_uz_ptr); CHKERRXX(ierr);
#endif
      ierr = VecGetArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                       #ifdef P4_TO_P8
                             12, 1, oss.str().c_str(),
                       #else
                             11, 1, oss.str().c_str(),
                       #endif
                             VTK_POINT_DATA, "phi", phi_eff_ptr,
                             VTK_POINT_DATA, "phi_smooth", phi_smooth_ptr,
                             VTK_POINT_DATA, "sol", sol_ptr,
                             VTK_POINT_DATA, "sol_ex", sol_ex_ptr,
                             VTK_POINT_DATA, "error_sl", vec_error_sl_ptr,
                             VTK_POINT_DATA, "error_tr", vec_error_tr_ptr,
                             VTK_POINT_DATA, "error_ux", vec_error_ux_ptr,
                             VTK_POINT_DATA, "error_uy", vec_error_uy_ptr,
                       #ifdef P4_TO_P8
                             VTK_POINT_DATA, "error_uz", vec_error_uz_ptr,
                       #endif
                             VTK_POINT_DATA, "error_ex", vec_error_ex_ptr,
                             VTK_POINT_DATA, "error_dd", vec_error_dd_ptr,
                             VTK_POINT_DATA, "mask", mask_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(solver.phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_smooth, &phi_smooth_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_sl, &vec_error_sl_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_tr, &vec_error_tr_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_ux, &vec_error_ux_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_uy, &vec_error_uy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(vec_error_uz, &vec_error_uz_ptr); CHKERRXX(ierr);
#endif
      ierr = VecRestoreArray(vec_error_ex, &vec_error_ex_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_dd, &vec_error_dd_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    // destroy Vec's with errors
    ierr = VecDestroy(vec_error_sl); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_tr); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_ux); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_uy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(vec_error_uz); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(vec_error_ex); CHKERRXX(ierr);
    ierr = VecDestroy(vec_error_dd); CHKERRXX(ierr);

    ierr = VecDestroy(phi_smooth); CHKERRXX(ierr);
    ierr = VecDestroy(sol_ex); CHKERRXX(ierr);

    for (int i = 0; i < phi.size(); i++)
    {
      ierr = VecDestroy(phi[i]);        CHKERRXX(ierr);
      ierr = VecDestroy(bc_values[i]);  CHKERRXX(ierr);
      ierr = VecDestroy(bc_coeffs[i]);  CHKERRXX(ierr);
    }

    ierr = VecDestroy(sol);         CHKERRXX(ierr);
    ierr = VecDestroy(mu);          CHKERRXX(ierr);
    ierr = VecDestroy(rhs);         CHKERRXX(ierr);
    ierr = VecDestroy(diag_add);    CHKERRXX(ierr);
    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  if (mpi.rank() == 0)
  {
    Gnuplot graph;

    print_Table("Error", 0.0, level, h, "err sl (max)", error_sl_arr,     1, &graph);
    print_Table("Error", 0.0, level, h, "err sl (L1)",  error_sl_l1_arr,  2, &graph);

    print_Table("Solution error", 0.0, level, h, "err tr (max)", error_tr_arr,     3, &graph);
    print_Table("Solution error", 0.0, level, h, "err ex (max)", error_ex_arr,     4, &graph);
    print_Table("Solution error", 0.0, level, h, "err dd (max)", error_dd_arr,     5, &graph);
//    print_Table("Error", 0.0, level, h, "err tr (L1)",  error_tr_l1_arr,  4, &graph);

    Gnuplot graph_grad;

    print_Table("Error", 0.0, level, h, "err ux (max)", error_ux_arr,     1, &graph_grad);
    print_Table("Error", 0.0, level, h, "err ux (L1)",  error_ux_l1_arr,  2, &graph_grad);

    print_Table("Error", 0.0, level, h, "err uy (max)", error_uy_arr,     3, &graph_grad);
    print_Table("Error in gradients", 0.0, level, h, "err uy (L1)",  error_uy_l1_arr,  4, &graph_grad);
#ifdef P4_TO_P8
    print_Table("Error", 0.0, level, h, "err uz (max)", error_uz_arr,     5, &graph_grad);
    print_Table("Error", 0.0, level, h, "err uz (L1)",  error_uz_l1_arr,  6, &graph_grad);
#endif

    // print all errors in compact form for plotting in matlab
    // step sizes
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << h[i];
    }
    cout <<  ";" << endl;

    // Sol L-inf
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_sl_arr[i]);
    }
    cout <<  ";" << endl;

    // Sol L-1
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_sl_l1_arr[i]);
    }
    cout <<  ";" << endl;

    // Grad L-inf
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_ux_arr[i]);
    }
    cout <<  ";" << endl;
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_uy_arr[i]);
    }
    cout <<  ";" << endl;
#ifdef P4_TO_P8
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_uz_arr[i]);
    }
    cout <<  ";" << endl;
#endif

    // Grad L-1
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_ux_l1_arr[i]);
    }
    cout <<  ";" << endl;
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_uy_l1_arr[i]);
    }
    cout <<  ";" << endl;
#ifdef P4_TO_P8
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(error_uz_l1_arr[i]);
    }
    cout <<  ";" << endl;
#endif


    cin.get();
  }

  return 0;
}
