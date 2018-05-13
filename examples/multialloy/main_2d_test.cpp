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
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_multialloy_optimized.h>
#include <src/my_p8est_macros.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_multialloy_optimized.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

int lmin = 4;
int lmax = 6;
int save_every_n_iteration = 1;

int num_splits = 5;

int interface_type = 0;
int test_number = 0;

double bc_tolerance = 1.e-10;

double cfl_number = 0.1;
double phi_thresh = 1e-3;
double zero_negative_velocity = 0;
int max_iterations = 100;
int pin_every_n_steps = 100;

int max_total_iterations = INT_MAX;

double r_0 = 0.2;
double dr = 0.1;
double x_0 = 0.5+0.01;
double y_0 = 0.5-0.03;
double z_0 = 0.5+0.07;

double beta_0 = 0.0;
double dbeta  = 0.2;

double t_end = 1;

double time_limit = t_end;

double init_perturb = 0.00;

bool use_continuous_stencil    = 0;
bool use_one_sided_derivatives = 0;
bool use_points_on_interface   = 0;
bool update_c0_robin           = 0;

bool concentration_neumann = true;

// not implemented yet
bool use_superconvergent_robin = 0;
bool use_superconvergent_jump  = false;

double lip = 2;

using namespace std;

bool save_velocity = true;
bool save_vtk = true;

#ifdef P4_TO_P8
char direction = 'z';
#else
char direction = 'y';
#endif

double termination_length = 0.9;

/* 0 - NiCu
 * 1 - AlCu
 */
int alloy_type = 2;

//double box_size = 4e-2;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double box_size = 2e-1;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1/box_size;

double xmin = 0;
double ymin = 0;
double xmax = 1;
double ymax = 1;
#ifdef P4_TO_P8
double zmin = 0;
double zmax = 1;
int n_xyz[] = {1, 1, 1};
#else
int n_xyz[] = {1, 1};
#endif

double rho;                  /* density                                    - kg.cm-3      */
double heat_capacity;        /* c, heat capacity                           - J.kg-1.K-1   */
double Tm;                   /* melting temperature                        - K            */
double G;                    /* thermal gradient                           - k.cm-1       */
double V;                    /* cooling velocity                           - cm.s-1       */
double latent_heat;          /* L, latent heat                             - J.cm-3       */
double thermal_conductivity; /* k, thermal conductivity                    - W.cm-1.K-1   */
double lambda;               /* thermal diffusivity                        - cm2.s-1      */

double eps_c;                /* curvature undercooling coefficient         - cm.K         */
double eps_v;                /* kinetic undercooling coefficient           - s.K.cm-1     */
double eps_anisotropy;       /* anisotropy coefficient                                    */

double ml0;                   /* liquidus slope                             - K / at frac. */
double kp0;                   /* partition coefficient                                     */
double c00;                   /* initial concentration                      - at frac.     */
double Dl0;                   /* liquid concentration diffusion coefficient - cm2.s-1      */

double ml1;                   /* liquidus slope                             - K / at frac. */
double kp1;                   /* partition coefficient                                     */
double c01;                   /* initial concentration                      - at frac.     */
double Dl1;                   /* liquid concentration diffusion coefficient - cm2.s-1      */


void set_alloy_parameters()
{
  switch(alloy_type)
  {
    case 0:
      /* Ni - 0.2at%Cu - 0.2at%Cu */
      rho                  = 8.88e-3;        /* kg.cm-3    */
      heat_capacity        = 0.46e3;         /* J.kg-1.K-1 */
      Tm                   = 1728;           /* K           */
      G                    = 4e2;            /* k.cm-1      */
      V                    = 0.01;           /* cm.s-1      */
      latent_heat          = 2350;           /* J.cm-3      */
      thermal_conductivity = 6.07e-1;        /* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */

      eps_c                = 2.7207e-5;
      eps_v                = 2.27e-2;
      eps_anisotropy       = 0.05;

      ml0                   =-357;            /* K / at frac. - liquidous slope */
      kp0                   = 0.86;           /* partition coefficient */
      c00                   = 0.2;            /* at frac.    */
      Dl0                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      ml1                   =-357;            /* K / at frac. - liquidous slope */
      kp1                   = 0.86;           /* partition coefficient */
      c01                   = 0.2;            /* at frac.    */
      Dl1                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

//      box_size = 4e-2;

      break;
    case 1:
      /* Ni - 15.2wt%Al - 5.8wt%Ta  */
      rho            = 7.365e-3;  /* kg.cm-3    */
      heat_capacity  = 660;       /* J.kg-1.K-1 */
      Tm             = 1754;      /* K           */
      G              = 200;       /* K.cm-1      */
      V              = 0.01;      /* cm.s-1      */
      latent_heat    = 2136;      /* J.cm-3      */
      thermal_conductivity =  0.8;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 2.7207e-4;
      eps_v          = 2.27e-2;
      eps_anisotropy = 0.05;

      Dl0 = 5e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml0 =-255;       /* K / wt frac. - liquidous slope */
      c00 = 0.152;     /* wt frac.    */
      kp0 = 0.48;      /* partition coefficient */

      Dl1 = 5e-5;
      ml1 =-517;
      c01 = 0.058;
      kp1 = 0.54;

//      box_size = 2e-1;
      break;

    case 2:
      /* Co - 10.7at%W - 9.4at%Al  */
      rho            = 9.2392e-3;   /* kg.cm-3    */
      heat_capacity  = 356;         /* J.kg-1.K-1 */
      Tm             = 1996;        /* K           */
      G              = 5000;         /* K.cm-1      */
      V              = 0.005;        /* cm.s-1      */
      latent_heat    = 2588.7;      /* J.cm-3      */
      thermal_conductivity =  1.3;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 0*2.7207e-5;
      eps_v          = 0*2.27e-2;
      eps_anisotropy = 0.05;

      Dl0 = 1e-4;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml0 =-874;       /* K / wt frac. - liquidous slope */
      c00 = 0.107;     /* at frac.    */
      kp0 = 0.848;     /* partition coefficient */

      Dl1 = 5e-4;
      ml1 =-1378;
      c01 = 0.094;
      kp1 = 0.848;

      box_size = 10.0e-1;

      break;

    case 3:
      /* Co - 9.4at%Al - 10.7at%W  */
      rho            = 9.2392e-3;   /* kg.cm-3    */
      heat_capacity  = 356;         /* J.kg-1.K-1 */
      Tm             = 1996;        /* K           */
      G              = 5;         /* K.cm-1      */
      V              = 0.01;        /* cm.s-1      */
      latent_heat    = 2588.7;      /* J.cm-3      */
      thermal_conductivity =  1.3;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 2.7207e-5;
      eps_v          = 2.27e-2;
      eps_anisotropy = 0.05;

      Dl0 = 1e-5;
      ml0 =-1378;
      c00 = 0.094;
      kp0 = 0.848;

      Dl1 = 1e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml1 =-874;       /* K / wt frac. - liquidous slope */
      c01 = 0.107;     /* at frac.    */
      kp1 = 0.848;     /* partition coefficient */

      box_size = 0.5e-1;
      break;
  }
}

#ifdef P4_TO_P8
class eps_c_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_c*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_c_cf;

class eps_v_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_v*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_v_cf;
#else
class eps_c_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_c*(1.-15.*eps_anisotropy*cos(4.*theta));
  }
} eps_c_cf;

class eps_v_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_v*(1.-15.*eps_anisotropy*cos(4.*theta));
  }
} eps_v_cf;
#endif

#ifdef P4_TO_P8
class ZERO : public CF_3
{
public:
  double operator()(double, double, double) const
  {
    return 0;
  }
} zero_cf;
#else
class ZERO : public CF_2
{
public:
  double operator()(double, double) const
  {
    return 0;
  }
} zero_cf;
#endif

//------------------------------------------------------------
// Level-Set Function
//------------------------------------------------------------

#ifdef P4_TO_P8
class phi_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return r_0 - sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_cf;

class phi_x_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(x-x_0)/sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_x_cf;

class phi_y_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(y-y_0)/sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_y_cf;

class phi_z_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(z-z_0)/sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_z_cf;

class phi_xx_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(SQR(y-y_0)+SQR(z-z_0))/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xx_cf;

class phi_yy_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(SQR(x-x_0)+SQR(z-z_0))/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_yy_cf;

class phi_zz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return -(SQR(y-y_0)+SQR(x-x_0))/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_zz_cf;

class phi_xy_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return (x-x_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xy_cf;

class phi_yz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return (z-z_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_yz_cf;

class phi_zx_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
      case 0:
        return (x-x_0)*(z-z_0)/pow(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0) + EPS, 1.5);
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_zx_cf;
#else
class phi_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return r_0 + dr*SQR(t/t_end) - sqrt(SQR(x) + SQR(y))
            - (beta_0 + dbeta*(t/t_end))*(5.*pow(x,4.)*y - 10.*pow(x,2.)*pow(y,3.) + pow(y, 5.))/(3.*pow(x*x+y*y+EPS,2.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_cf;

class phi_x_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return -(x)/sqrt(SQR(x) + SQR(y) + EPS)
            - (beta_0 + dbeta*(t/t_end))*(-5.*x*y*(pow(x,4.)-10.*pow(x*y,2.)+5.*pow(y,4.)))/(3.*pow(x*x+y*y+EPS,3.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_x_cf;

class phi_y_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return -(y)/sqrt(SQR(x) + SQR(y) + EPS)
            - (beta_0 + dbeta*(t/t_end))*(5.*(pow(x,6.)-10.*pow(x*x*y,2.)+5.*pow(x*y*y,2.)))/(3.*pow(x*x+y*y+EPS,3.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_y_cf;

class phi_xx_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return -(y)*(y)/pow(SQR(x) + SQR(y) + EPS, 1.5)
            - (beta_0 + dbeta*(t/t_end))*(5.*y*(2.*pow(x,6.)-45.*pow(x*x*y,2.)+60.*pow(x*y*y,2.)-5.*pow(y,6.)))/(3.*pow(x*x+y*y+EPS,4.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xx_cf;

class phi_xy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return (x)*(y)/pow(SQR(x) + SQR(y) + EPS, 1.5)
            - (beta_0 + dbeta*(t/t_end))*(-5.*x*(pow(x,6.)-36.*pow(x*x*y,2.)+65.*pow(x*y*y,2.)-10.*pow(y,6.)))/(3.*pow(x*x+y*y+EPS,4.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xy_cf;

class phi_yy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return -(x)*(x)/pow(SQR(x) + SQR(y) + EPS, 1.5)
            - (beta_0 + dbeta*(t/t_end))*(-5.*x*x*y*(27.*pow(x,4.)-70.*pow(x*y,2.)+15.*pow(y,5.)))/(3.*pow(x*x+y*y+EPS,4.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_yy_cf;

class phi_t_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    x = x-x_0;
    y = y-y_0;
    switch(interface_type)
    {
      case 0:
        return 2.*dr*t/SQR(t_end)
            - dbeta/t_end*(5.*pow(x,4.)*y - 10.*pow(x,2.)*pow(y,3.) + pow(y, 5.))/(3.*pow(x*x+y*y+EPS,2.5));
      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_t_cf;
#endif

#ifdef P4_TO_P8
class kappa_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return (SQR(phi_x_cf(x,y,z))*phi_yy_cf(x,y,z) - 2.*phi_x_cf(x,y,z)*phi_y_cf(x,y,z)*phi_xy_cf(x,y,z) + SQR(phi_y_cf(x,y,z))*phi_xx_cf(x,y,z) +
            SQR(phi_y_cf(x,y,z))*phi_zz_cf(x,y,z) - 2.*phi_y_cf(x,y,z)*phi_z_cf(x,y,z)*phi_yz_cf(x,y,z) + SQR(phi_z_cf(x,y,z))*phi_yy_cf(x,y,z) +
            SQR(phi_x_cf(x,y,z))*phi_zz_cf(x,y,z) - 2.*phi_z_cf(x,y,z)*phi_x_cf(x,y,z)*phi_zx_cf(x,y,z) + SQR(phi_z_cf(x,y,z))*phi_xx_cf(x,y,z))
        /pow(SQR(phi_x_cf(x,y,z)) + SQR(phi_y_cf(x,y,z)) + SQR(phi_z_cf(x,y,z)) + EPS, 1.5);
  }
} kappa_cf;
#else
class kappa_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return (SQR(phi_x_cf(x,y))*phi_yy_cf(x,y) - 2.*phi_x_cf(x,y)*phi_y_cf(x,y)*phi_xy_cf(x,y) + SQR(phi_y_cf(x,y))*phi_xx_cf(x,y))/pow(SQR(phi_x_cf(x,y))+SQR(phi_y_cf(x,y)) + EPS, 1.5);
  }
} kappa_cf;
#endif

//------------------------------------------------------------
// Concentration 0
//------------------------------------------------------------
#ifdef P4_TO_P8
class c0_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*(1.1+sin(x)*cos(y))*exp(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_exact;

class c0_x_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*cos(x)*cos(y)*exp(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_x_exact;

class c0_y_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -0.5*sin(x)*sin(y)*exp(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_y_exact;

class c0_z_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*(1.1+sin(x)*cos(y))*exp(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_z_exact;

class c0_dd_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*0.5*sin(x)*cos(y)*exp(z) + 0.5*(1.1+sin(x)*cos(y))*exp(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_dd_exact;
#else
class c0_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*(1.1+sin(x)*cos(y))*(1.+0.2*cos(t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_exact;

class c0_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*cos(x)*cos(y)*(1.+0.2*cos(t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_x_exact;

class c0_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -0.5*sin(x)*sin(y)*(1.+0.2*cos(t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_y_exact;

class c0_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*0.5*sin(x)*cos(y)*(1.+0.2*cos(t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_dd_exact;

class c0_t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*(1.1+sin(x)*cos(y))*(-0.2*sin(t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_t_exact;
#endif

//------------------------------------------------------------
// Concentration 1
//------------------------------------------------------------
#ifdef P4_TO_P8
class c1_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*sin(y)*exp(-z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_exact;

class c1_x_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return -sin(x)*sin(y)*exp(-z);
      default: throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_x_exact;

class c1_y_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*cos(y)*exp(-z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_y_exact;

class c1_z_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -cos(x)*sin(y)*exp(-z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_z_exact;

class c1_dd_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -1.*cos(x)*sin(y)*exp(-z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_dd_exact;
#else
class c1_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*sin(y)*(1.-0.5*exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_exact;

class c1_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return -sin(x)*sin(y)*(1.-0.5*exp(-t));
      default: throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_x_exact;

class c1_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*cos(y)*(1.-0.5*exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_y_exact;

class c1_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*cos(x)*sin(y)*(1.-0.5*exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_dd_exact;

class c1_t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*sin(y)*0.5*exp(-t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_t_exact;

#endif

//------------------------------------------------------------
// Temperature
//------------------------------------------------------------
#ifdef P4_TO_P8
class tm_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        //        return sin(x)*(y+y*y);
        return (x*x + x*y + y*y)*sin(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_exact;

class tm_x_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        //        return cos(x)*(y+y*y);
        return (2.*x + y)*sin(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_x_exact;

class tm_y_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        //        return sin(x)*(1.+2.*y);
        return (x + 2.*y)*sin(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_y_exact;

class tm_z_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        //        return sin(x)*(1.+2.*y);
        return (x*x + x*y + y*y)*cos(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_z_exact;

class tm_dd_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        //        return -sin(x)*(y+y*y) + 2.*sin(x);
        return 4.*sin(z) - (x*x + x*y + y*y)*sin(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_dd_exact;
#else
class tm_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return x*x + x*y*sin(t) + y*y*cos(t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_exact;

class tm_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 2.*x + y*sin(t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_x_exact;

class tm_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return x*sin(t) + 2.*y*cos(t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_y_exact;

class tm_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 2.+2.*cos(t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_dd_exact;

class tm_t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return x*y*cos(t) - y*y*sin(t);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_t_exact;
#endif


#ifdef P4_TO_P8
class tp_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(y+y*y)*cos(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_exact;

class tp_x_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*(y+y*y)*cos(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_x_exact;

class tp_y_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(1.+2.*y)*cos(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_y_exact;

class tp_z_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -sin(x)*(y+y*y)*sin(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_z_exact;

class tp_dd_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*sin(x)*(y+y*y)*cos(z) + 2.*sin(x)*cos(z);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_dd_exact;
#else
class tp_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(y+y*y)*(1.+exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_exact;

class tp_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*(y+y*y)*(1.+exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_x_exact;

class tp_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(1.+2.*y)*(1.+exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_y_exact;

class tp_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -sin(x)*(y+y*y)*(1.+exp(-t)) + 2.*sin(x)*(1.+exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_dd_exact;

class tp_t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(y+y*y)*(-exp(-t));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_t_exact;
#endif

#ifdef P4_TO_P8
class t_exact_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (phi_cf(x,y,z) < 0)  return tm_exact(x,y,z);
    else                    return tp_exact(x,y,z);
  }
} t_exact;
#else
class t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (phi_cf(x,y) < 0)  return tm_exact(x,y);
    else                  return tp_exact(x,y);
  }
} t_exact;
#endif

//------------------------------------------------------------
// right-hand-sides
//------------------------------------------------------------
#ifdef P4_TO_P8
class rhs_c0_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return c0_t_exact(x,y,z) - Dl0*c0_dd_exact(x,y,z);
  }
} rhs_c0_cf;

class rhs_c1_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return c1_t_exact(x,y,z) - Dl1*c1_dd_exact(x,y,z);
  }
} rhs_c1_cf;

class rhs_tm_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return tm_t_exact(x,y,z) - lambda*tm_dd_exact(x,y,z);
  }
} rhs_tm_cf;

class rhs_tp_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return tp_t_exact(x,y,z) - lambda*tp_dd_exact(x,y,z);
  }
} rhs_tp_cf;
#else
class rhs_c0_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return c0_t_exact(x,y) - Dl0*c0_dd_exact(x,y);
  }
} rhs_c0_cf;

class rhs_c1_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return c1_t_exact(x,y) - Dl1*c1_dd_exact(x,y);
  }
} rhs_c1_cf;

class rhs_tm_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_t_exact(x,y) - lambda*tm_dd_exact(x,y);
  }
} rhs_tm_cf;

class rhs_tp_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tp_t_exact(x,y) - lambda*tp_dd_exact(x,y);
  }
} rhs_tp_cf;
#endif

#ifdef P4_TO_P8
class vn_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -phi_t_cf(x,y,z) / sqrt(SQR(phi_x_cf(x,y,z)) + SQR(phi_y_cf(x,y,z)) + SQR(phi_z_cf(x,y,z)) + EPS);
  }
} vn_cf;
#else
class vn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -phi_t_cf(x,y) / sqrt(SQR(phi_x_cf(x,y)) + SQR(phi_y_cf(x,y)) + EPS);
  }
} vn_cf;
#endif

//------------------------------------------------------------
// jumps in t
//------------------------------------------------------------
#ifdef P4_TO_P8
class jump_t_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return tm_exact(x,y,z) - tp_exact(x,y,z);
  }
} jump_t_cf;

class jump_tn_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double nx = phi_x_cf(x,y,z);
    double ny = phi_y_cf(x,y,z);
    double nz = phi_z_cf(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz)+EPS;
    nx /= norm; ny /= norm; nz /= norm;
    return (tm_x_exact(x,y,z) - tp_x_exact(x,y,z))*nx +
        (tm_y_exact(x,y,z) - tp_y_exact(x,y,z))*ny +
        (tm_z_exact(x,y,z) - tp_z_exact(x,y,z))*nz;
  }
} jump_tn_cf;
#else
class jump_t_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_exact(x,y) - tp_exact(x,y);
  }
} jump_t_cf;

class jump_tn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny)+EPS;
    nx /= norm; ny /= norm;
    return (tm_x_exact(x,y) - tp_x_exact(x,y))*nx + (tm_y_exact(x,y) - tp_y_exact(x,y))*ny + latent_heat/thermal_conductivity*vn_cf(x,y);
  }
} jump_tn_cf;
#endif

//------------------------------------------------------------
// bc for c1
//------------------------------------------------------------
#ifdef P4_TO_P8
class c1_robin_coef_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return (1.-kp1)/Dl1*vn_cf(x,y,z);
  }
} c1_robin_coef_cf;

class c1_interface_val_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double nx = phi_x_cf(x,y,z);
    double ny = phi_y_cf(x,y,z);
    double nz = phi_z_cf(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz)+EPS;
    nx /= norm; ny /= norm; nz /= norm;
    return c1_x_exact(x,y,z)*nx + c1_y_exact(x,y,z)*ny + c1_z_exact(x,y,z)*nz + c1_robin_coef_cf(x,y,z)*c1_exact(x,y,z);
  }
} c1_interface_val_cf;
#else
class c1_robin_coef_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return (1.-kp1)/Dl1*vn_cf(x,y);
  }
} c1_robin_coef_cf;

class c1_interface_val_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny)+EPS;
    nx /= norm; ny /= norm;
    return c1_x_exact(x,y)*nx + c1_y_exact(x,y)*ny + c1_robin_coef_cf(x,y)*c1_exact(x,y);
  }
} c1_interface_val_cf;
#endif

#ifdef P4_TO_P8
class bc_wall_type_t_t : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_t;

class bc_wall_type_c0_t : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c0;

class bc_wall_type_c1_t : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c1;
#else
class bc_wall_type_t_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_t;

class bc_wall_type_c0_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c0;

class bc_wall_type_c1_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c1;
#endif

#ifdef P4_TO_P8
class c0_interface_val_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double nx = phi_x_cf(x,y,z);
    double ny = phi_y_cf(x,y,z);
    double nz = phi_z_cf(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz)+EPS;
    nx /= norm; ny /= norm; nz /= norm;
    return c0_x_exact(x,y,z)*nx + c0_y_exact(x,y,z)*ny + c0_z_exact(x,y,z)*nz + (1.-kp0)/Dl0*vn_cf(x,y,z)*c0_exact(x,y,z);
  }
} c0_interface_val_cf;
#else
class c0_interface_val_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny)+EPS;
    nx /= norm; ny /= norm;
    return c0_x_exact(x,y)*nx + c0_y_exact(x,y)*ny + (1.-kp0)/Dl0*vn_cf(x,y)*c0_exact(x,y);
  }
} c0_interface_val_cf;
#endif

#ifdef P4_TO_P8
class c0_guess_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    //    return 1.;
    return c0_exact(x,y,z);
    //    return c0_exact(x,y) + 0.1;
  }
} c0_guess;
#else
class c0_guess_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    //    return 1.;
    return c0_exact(x,y);
    //    return c0_exact(x,y) + 0.1;
  }
} c0_guess;
#endif

#ifdef P4_TO_P8
class gibbs_thompson_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return tm_exact(x,y,z) - Tm - ml0*c0_exact(x,y,z) - ml1*c1_exact(x,y,z) + eps_c_cf(phi_x_cf(x,y,z), phi_y_cf(x,y,z), phi_z_cf(x,y,z))*kappa_cf(x,y,z) + eps_v_cf(phi_x_cf(x,y,z), phi_y_cf(x,y,z), phi_z_cf(x,y,z))*vn_cf(x,y,z);
  }
} gibbs_thompson;
#else
class gibbs_thompson_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_exact(x,y) - Tm - ml0*c0_exact(x,y) - ml1*c1_exact(x,y) + eps_c_cf(phi_x_cf(x,y), phi_y_cf(x,y))*kappa_cf(x,y) + eps_v_cf(phi_x_cf(x,y), phi_y_cf(x,y))*vn_cf(x,y);
  }
} gibbs_thompson;
#endif


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  cmd.add_option("xmin", "xmin");
  cmd.add_option("xmax", "xmax");
  cmd.add_option("ymin", "ymin");
  cmd.add_option("ymax", "ymax");
#ifdef P4_TO_P8
  cmd.add_option("zmin", "zmin");
  cmd.add_option("zmax", "zmax");
#endif

  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");

  cmd.add_option("nx", "number of blox in x-dimension");
  cmd.add_option("ny", "number of blox in y-dimension");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of blox in z-dimension");
#endif

  cmd.add_option("px", "periodicity in x-dimension 0/1");
  cmd.add_option("py", "periodicity in y-dimension 0/1");
#ifdef P4_TO_P8
  cmd.add_option("pz", "periodicity in z-dimension 0/1");
#endif
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_velo", "1 to save velocity of the interface, 0 otherwise");
  cmd.add_option("save_every_n", "save vtk every n iteration");
  cmd.add_option("write_stats", "write the statistics about the p4est");
  cmd.add_option("tf", "final time");
  cmd.add_option("L", "set latent heat");
  cmd.add_option("G", "set heat gradient");
  cmd.add_option("V", "set velocity");
  cmd.add_option("box_size", "set box_size");
  cmd.add_option("alloy", "choose the type of alloy. Default is 0.\n  0 - NiCuCu\n  1 - NiAlTa");
  cmd.add_option("direction", "direction of the crystal growth x/y");
  cmd.add_option("Dl0", "set the concentration diffusion coefficient in the liquid phase");
  cmd.add_option("Dl1", "set the concentration diffusion coefficient in the liquid phase");
  cmd.add_option("eps_c", "set the curvature undercooling coefficient");
  cmd.add_option("eps_v", "set the kinetic undercooling coefficient");

  cmd.add_option("bc_tolerance", "error tolerance for internal iterations");
  cmd.add_option("termination_length", "defines when a run will be stopped (fraction of box length, from 0 to 1)");
  cmd.add_option("lip", "set the lipschitz constant");
  cmd.add_option("cfl_number", "cfl_number");
  cmd.add_option("phi_thresh", "phi_thresh");
  cmd.add_option("zero_negative_velocity", "zero_negative_velocity");
  cmd.add_option("max_iterations", "max_iterations");
  cmd.add_option("pin_every_n_steps", "pin_every_n_steps");

  cmd.add_option("use_continuous_stencil"   , "use_continuous_stencil"   );
  cmd.add_option("use_one_sided_derivatives", "use_one_sided_derivatives");
  cmd.add_option("use_points_on_interface"  , "use_points_on_interface"  );
  cmd.add_option("update_c0_robin"          , "update_c0_robin"          );
  cmd.add_option("use_superconvergent_robin", "use_superconvergent_robin");
  cmd.add_option("use_superconvergent_jump" , "use_superconvergent_jump" );

  cmd.add_option("init_perturb", "init_perturb");

  cmd.add_option("concentration_neumann", "concentration_neumann");

  cmd.parse(argc, argv);

  alloy_type = cmd.get("alloy", alloy_type);
  set_alloy_parameters();

  xmin = cmd.get("xmin", xmin);
  xmax = cmd.get("xmax", xmax);
  ymin = cmd.get("ymin", ymin);
  ymax = cmd.get("ymax", ymax);
#ifdef P4_TO_P8
  zmin = cmd.get("zmin", zmin);
  zmax = cmd.get("zmax", zmax);
#endif

  save_vtk = cmd.get("save_vtk", save_vtk);
  save_velocity = cmd.get("save_velo", save_velocity);

  int periodic[P4EST_DIM];
  periodic[0] = cmd.get("px", (direction=='y' || direction=='z') ? 1 : 0);
  periodic[1] = cmd.get("py", (direction=='x' || direction=='z') ? 1 : 0);
#ifdef P4_TO_P8
  periodic[2] = cmd.get("pz", (direction=='x' || direction=='y') ? 1 : 0);
#endif

  periodic[0] = 0;
  periodic[1] = 0;
#ifdef P4_TO_P8
  periodic[2] = 0;
#endif


  n_xyz[0] = cmd.get("nx", n_xyz[0]);
  n_xyz[1] = cmd.get("ny", n_xyz[1]);
#ifdef P4_TO_P8
  n_xyz[2] = cmd.get("nz", n_xyz[2]);
#endif

#ifdef P4_TO_P8
  direction = cmd.get("direction", 'z');
#else
  direction = cmd.get("direction", 'y');
#endif

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

  PetscErrorCode ierr;

  save_every_n_iteration = cmd.get("save_every_n", save_every_n_iteration);
  latent_heat = cmd.get("L", latent_heat);
  G = cmd.get("G", G);
  V = cmd.get("V", V);
  box_size = cmd.get("box_size", box_size);
  Dl0 = cmd.get("Dl0", Dl0);
  Dl1 = cmd.get("Dl1", Dl1);
  eps_c = cmd.get("eps_c", eps_c);
  eps_v = cmd.get("eps_v", eps_v);

  termination_length        = cmd.get("termination_length", termination_length    );
  cfl_number                = cmd.get("cfl_number", cfl_number            );
  phi_thresh                = cmd.get("phi_thresh", phi_thresh            );
  zero_negative_velocity    = cmd.get("zero_negative_velocity", zero_negative_velocity);
  max_iterations            = cmd.get("max_iterations", max_iterations        );
  pin_every_n_steps         = cmd.get("pin_every_n_steps", pin_every_n_steps     );
  bc_tolerance              = cmd.get("bc_tolerance", bc_tolerance          );

  use_continuous_stencil    = cmd.get("use_continuous_stencil", use_continuous_stencil   );
  use_one_sided_derivatives = cmd.get("use_one_sided_derivatives", use_one_sided_derivatives);
  use_points_on_interface   = cmd.get("use_points_on_interface", use_points_on_interface  );
  update_c0_robin           = cmd.get("update_c0_robin", update_c0_robin          );
  use_superconvergent_robin = cmd.get("use_superconvergent_robin", use_superconvergent_robin);
  use_superconvergent_jump  = cmd.get("use_superconvergent_jump", use_superconvergent_jump );

  concentration_neumann     = cmd.get("concentration_neumann", concentration_neumann);
  init_perturb              = cmd.get("init_perturb", init_perturb);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  lip = cmd.get("lip", lip);

  double latent_heat_orig = latent_heat;
  double G_orig = G;
  double V_orig = V;

  scaling = 1/box_size;
  rho                  /= (scaling*scaling*scaling);
  thermal_conductivity /= scaling;
  G                    /= scaling;
  V                    *= scaling;
  latent_heat          /= (scaling*scaling*scaling);
  eps_c                *= scaling;
  eps_v                /= scaling;
  lambda                = thermal_conductivity/(rho*heat_capacity);

  Dl0                  *= (scaling*scaling);
  Dl1                  *= (scaling*scaling);

  parStopWatch w1;
  w1.start("total time");

  for (char iter = 0; iter < num_splits; ++iter)
  {
    PetscPrintf(mpi.comm(), "Resolution: %d / %d \n", lmin, lmax);

    double tn = 0;

    phi_cf.t = tn;
    phi_t_cf.t = tn;
    phi_x_cf.t = tn;
    phi_y_cf.t = tn;
    phi_xx_cf.t = tn;
    phi_xy_cf.t = tn;
    phi_yy_cf.t = tn;

    c0_exact.t = tn;
    c0_t_exact.t = tn;
    c0_x_exact.t = tn;
    c0_y_exact.t = tn;
    c0_dd_exact.t = tn;

    c1_exact.t = tn;
    c1_t_exact.t = tn;
    c1_x_exact.t = tn;
    c1_y_exact.t = tn;
    c1_dd_exact.t = tn;

    tm_exact.t = tn;
    tm_t_exact.t = tn;
    tm_x_exact.t = tn;
    tm_y_exact.t = tn;
    tm_dd_exact.t = tn;

    tp_exact.t = tn;
    tp_t_exact.t = tn;
    tp_x_exact.t = tn;
    tp_y_exact.t = tn;
    tp_dd_exact.t = tn;

  /* create the p4est */
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
#else
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
#endif
  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);


  splitting_criteria_cf_t data(lmin, lmax, &phi_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  if (use_continuous_stencil || use_one_sided_derivatives)
    my_p4est_ghost_expand(p4est, ghost);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  /* initialize the variables */
  Vec phi, tl, ts, c0, c1, normal_velocity;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &tl             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &ts             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c0             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c1             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &normal_velocity); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_cf, phi);
  sample_cf_on_nodes(p4est, nodes, tm_exact, tl);
  sample_cf_on_nodes(p4est, nodes, tp_exact, ts);
  sample_cf_on_nodes(p4est, nodes, c0_exact, c0);
  sample_cf_on_nodes(p4est, nodes, c1_exact, c1);
  sample_cf_on_nodes(p4est, nodes, vn_cf, normal_velocity);

  /* set initial time step */
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_tree = p4est->connectivity->vertices[3*vm + 0];
  double ymin_tree = p4est->connectivity->vertices[3*vm + 1];
  double xmax_tree = p4est->connectivity->vertices[3*vp + 0];
  double ymax_tree = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax_tree-xmin_tree) / pow(2., (double) data.max_lvl);
  double dy = (ymax_tree-ymin_tree) / pow(2., (double) data.max_lvl);
#ifdef P4_TO_P8
  double zmin_tree = p4est->connectivity->vertices[3*vm + 2];
  double zmax_tree = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax_tree-zmin_tree) / pow(2.,(double) data.max_lvl);
#endif

//#ifdef P4_TO_P8
//  double dt = cfl_number*MIN(dx,dy,dz)/r_dot;
//#else
//  double dt = cfl_number*MIN(dx,dy)/r_dot;
//#endif


//  double *phi_p;
//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

//  srand(mpi.rank());

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//  {
//    phi_p[n] += init_perturb*dx*(double)(rand()%1000)/1000.;
//  }

//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  /* perturb level set */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
//  ls.perturb_level_set_function(phi, EPS);

  /* initialize the solver */
  my_p4est_multialloy_t bas(ngbd);

  bas.set_parameters(latent_heat, thermal_conductivity, lambda,
                     V, Tm, scaling,
                     Dl0, kp0, c00, ml0,
                     Dl1, kp1, c01, ml1);
  bas.set_phi(phi);
  bas.set_bc(bc_wall_type_t,
             bc_wall_type_c0,
             t_exact,
             c0_exact,
             c1_exact);
  bas.set_temperature(tl, ts);
  bas.set_concentration(c0, c1);
  bas.set_normal_velocity(normal_velocity);
//  bas.set_dt(dt);

  bas.set_GT(gibbs_thompson);
  bas.set_jump_t(jump_t_cf);
  bas.set_jump_tn(jump_tn_cf);
  bas.set_flux_c(c0_interface_val_cf, c1_interface_val_cf);
  bas.set_undercoolings(eps_v_cf, eps_c_cf);

  bas.set_rhs(rhs_tm_cf, rhs_tp_cf, rhs_c0_cf, rhs_c1_cf);

  bas.set_bc_tolerance(bc_tolerance);
  bas.set_max_iterations(max_iterations);
  bas.set_pin_every_n_steps(pin_every_n_steps);
  bas.set_cfl(cfl_number);
  bas.set_phi_thresh(phi_thresh);

  bas.set_use_continuous_stencil   (use_continuous_stencil   );
  bas.set_use_one_sided_derivatives(use_one_sided_derivatives);
  bas.set_use_superconvergent_robin(use_superconvergent_robin);
  bas.set_use_superconvergent_jump (use_superconvergent_jump );
  bas.set_use_points_on_interface  (use_points_on_interface  );
  bas.set_update_c0_robin          (update_c0_robin          );
  bas.set_zero_negative_velocity   (zero_negative_velocity   );


//  bas.set_zero_negative_velocity(zero_negative_velocity);
//  bas.set_num_of_iterations_per_step(num_of_iters_per_step);

  bas.compute_velocity();
  bas.compute_dt();

  // loop over time
  int iteration = 0;

  FILE *fich;
  char name[10000];

  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
#ifdef P4_TO_P8
  sprintf(name, "%s/velo_%dx%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], n_xyz[2], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#else
  sprintf(name, "%s/velo_%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#endif

  if(save_velocity)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "time average_interface_velocity max_interface_velocity interface_length solid_phase_area time_elapsed iteration local_nodes ghost_nodes sub_iterations\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  bool keep_going = true;
  int sub_iterations = 0;

  while(keep_going)
//  while (iteration < 20)
  {
//    bas.set_dt(cfl_number*MIN(dx,dy)/r_dot);
    if (tn + bas.get_dt() > time_limit) { bas.set_dt(time_limit-tn); keep_going = false; }

    tn += bas.get_dt();

    phi_cf.t = tn;
    phi_t_cf.t = tn;
    phi_x_cf.t = tn;
    phi_y_cf.t = tn;
    phi_xx_cf.t = tn;
    phi_xy_cf.t = tn;
    phi_yy_cf.t = tn;

    c0_exact.t = tn;
    c0_t_exact.t = tn-.5*bas.get_dt();
    c0_x_exact.t = tn;
    c0_y_exact.t = tn;
    c0_dd_exact.t = tn-.5*bas.get_dt();

    c1_exact.t = tn;
    c1_t_exact.t = tn-.5*bas.get_dt();
    c1_x_exact.t = tn;
    c1_y_exact.t = tn;
    c1_dd_exact.t = tn-.5*bas.get_dt();

    tm_exact.t = tn;
    tm_t_exact.t = tn-.5*bas.get_dt();
    tm_x_exact.t = tn;
    tm_y_exact.t = tn;
    tm_dd_exact.t = tn-.5*bas.get_dt();

    tp_exact.t = tn;
    tp_t_exact.t = tn-.5*bas.get_dt();
    tp_x_exact.t = tn;
    tp_y_exact.t = tn;
    tp_dd_exact.t = tn;

    sub_iterations += bas.one_step();

    // save velocity, lenght of interface and area of solid phase in time
    if(save_velocity && iteration%save_every_n_iteration == 0)
    {
      p4est = bas.get_p4est();
      nodes = bas.get_nodes();
      phi = bas.get_phi();
      normal_velocity = bas.get_normal_velocity();

      Vec ones, tmp;
      ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(ones, &tmp); CHKERRXX(ierr);
      ierr = VecSet(tmp, 1.); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(ones, &tmp); CHKERRXX(ierr);

      // calculate the length of the interface and solid phase area
      double interface_length = integrate_over_interface(p4est, nodes, phi, ones);
      double solid_phase_area = 1.-integrate_over_negative_domain(p4est, nodes, phi, ones);

      double avg_velo = integrate_over_interface(p4est, nodes, phi, normal_velocity) / interface_length;

      ierr = VecDestroy(ones); CHKERRXX(ierr);

      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      double time_elapsed = w1.read_duration_current();

      int num_local_nodes = nodes->num_owned_indeps;
      int num_ghost_nodes = nodes->indep_nodes.elem_count - num_local_nodes;

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &num_local_nodes, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
          mpiret = MPI_Allreduce(MPI_IN_PLACE, &num_ghost_nodes, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e %e %d %d %d %d\n", tn, avg_velo/scaling, bas.get_max_interface_velocity()/scaling, interface_length, solid_phase_area, time_elapsed, iteration, num_local_nodes, num_ghost_nodes, sub_iterations);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved velocity in %s\n", name); CHKERRXX(ierr);
      sub_iterations = 0;
    }

    // save field data
    if(save_vtk && iteration%save_every_n_iteration == 0)
    {
      bas.save_VTK(iteration/save_every_n_iteration);
    }
    ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);

    keep_going = keep_going && (iteration < max_total_iterations);

    bas.update_grid();

    // check accuracy
    {
      p4est = bas.get_p4est();
      nodes = bas.get_nodes();
      ghost = bas.get_ghost();
      ngbd  = bas.get_ngbd();

      Vec tl = bas.get_tl();
      Vec ts = bas.get_ts();
      Vec c0 = bas.get_c0();
      Vec c1 = bas.get_c1();
      phi    = bas.get_phi();
      Vec *phi_dd = bas.get_phi_dd();

      double *tl_p; ierr = VecGetArray(tl, &tl_p);   CHKERRXX(ierr);
      double *ts_p; ierr = VecGetArray(ts, &ts_p);   CHKERRXX(ierr);
      double *c0_p; ierr = VecGetArray(c0, &c0_p); CHKERRXX(ierr);
      double *c1_p; ierr = VecGetArray(c1, &c1_p); CHKERRXX(ierr);
      double *phi_p; ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      Vec err_tm; ierr = VecDuplicate(phi, &err_tm); CHKERRXX(ierr);
      Vec err_tp; ierr = VecDuplicate(phi, &err_tp); CHKERRXX(ierr);
      Vec err_c0; ierr = VecDuplicate(phi, &err_c0); CHKERRXX(ierr);
      Vec err_c1; ierr = VecDuplicate(phi, &err_c1); CHKERRXX(ierr);
      Vec phi_ex; ierr = VecDuplicate(phi, &phi_ex); CHKERRXX(ierr);
      Vec err_ph; ierr = VecDuplicate(phi, &err_ph); CHKERRXX(ierr);

      sample_cf_on_nodes(p4est, nodes, phi_cf, phi_ex);

      double *err_tm_p; ierr = VecGetArray(err_tm, &err_tm_p); CHKERRXX(ierr);
      double *err_tp_p; ierr = VecGetArray(err_tp, &err_tp_p); CHKERRXX(ierr);
      double *err_c0_p; ierr = VecGetArray(err_c0, &err_c0_p); CHKERRXX(ierr);
      double *err_c1_p; ierr = VecGetArray(err_c1, &err_c1_p); CHKERRXX(ierr);
      double *phi_ex_p; ierr = VecGetArray(phi_ex, &phi_ex_p); CHKERRXX(ierr);
      double *err_ph_p; ierr = VecGetArray(err_ph, &err_ph_p); CHKERRXX(ierr);

      double xyz[P4EST_DIM];

      double error_tm = 0;
      double error_tp = 0;
      double error_c0 = 0;
      double error_c1 = 0;

      foreach_node(n, nodes)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
        if (phi_p[n] < 0)
        {
          err_tm_p[n] = fabs(tl_p[n] - tm_exact.value(xyz));
          err_tp_p[n] = 0;
          err_c0_p[n] = fabs(c0_p[n] - c0_exact.value(xyz));
          err_c1_p[n] = fabs(c1_p[n] - c1_exact.value(xyz));
        } else {
          err_tm_p[n] = 0;
          err_tp_p[n] = fabs(ts_p[n] - tp_exact.value(xyz));
          err_c0_p[n] = 0;
          err_c1_p[n] = 0;
        }

        error_tm = MAX(error_tm, err_tm_p[n]);
        error_tp = MAX(error_tp, err_tp_p[n]);
        error_c0 = MAX(error_c0, err_c0_p[n]);
        error_c1 = MAX(error_c1, err_c1_p[n]);
      }

      double error_ph = 0;

      double phi_000;
      double phi_m00;
      double phi_p00;
      double phi_0m0;
      double phi_0p0;
#ifdef P4_TO_P8
      double phi_00m;
      double phi_00p;
#endif

      double *phi_dd_p[P4EST_DIM];

      foreach_dimension(dim) { ierr = VecGetArray(phi_dd[dim], &phi_dd_p[dim]); CHKERRXX(ierr); }

      foreach_local_node(n, nodes)
      {
        err_ph_p[n] = 0;

        const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

#ifdef P4_TO_P8
        qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
        qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif

        for (char dir = 0; dir < P4EST_FACES; ++dir)
        {
          double p0 = phi_000, p1;

          switch (dir) {
            case 0: p1 = phi_m00; break;
            case 1: p1 = phi_p00; break;
            case 2: p1 = phi_0m0; break;
            case 3: p1 = phi_0p0; break;
#ifdef P4_TO_P8
            case 4: p1 = phi_00m; break;
            case 5: p1 = phi_00p; break;
#endif
          }

          if (p0*p1 < 0)
          {
            double pdd0=0, pdd1=0;

            switch (dir) {
              case 0: pdd0 = phi_dd_p[0][n]*dx*dx; pdd1 = phi_dd_p[0][qnnn.neighbor_m00()]*dx*dx; break;
              case 1: pdd0 = phi_dd_p[0][n]*dx*dx; pdd1 = phi_dd_p[0][qnnn.neighbor_p00()]*dx*dx; break;
              case 2: pdd0 = phi_dd_p[1][n]*dy*dy; pdd1 = phi_dd_p[1][qnnn.neighbor_0m0()]*dy*dy; break;
              case 3: pdd0 = phi_dd_p[1][n]*dy*dy; pdd1 = phi_dd_p[1][qnnn.neighbor_0p0()]*dy*dy; break;
#ifdef P4_TO_P8
              case 4: pdd0 = phi_dd_p[2][n]*dz*dz; pdd1 = phi_dd_p[2][qnnn.neighbor_00m()]*dz*dz; break;
              case 5: pdd0 = phi_dd_p[2][n]*dz*dz; pdd1 = phi_dd_p[2][qnnn.neighbor_00p()]*dz*dz; break;
#endif
            }

            double theta = interface_Location_With_Second_Order_Derivative(0., 1., p0, p1, pdd0, pdd1);

            node_xyz_fr_n(n, p4est, nodes, xyz);

            switch (dir) {
              case 0: xyz[0] -= theta*dx; break;
              case 1: xyz[0] += theta*dx; break;
              case 2: xyz[1] -= theta*dy; break;
              case 3: xyz[1] += theta*dy; break;
#ifdef P4_TO_P8
              case 4: xyz[2] -= theta*dz; break;
              case 5: xyz[2] += theta*dz; break;
#endif
            }

            err_ph_p[n] = MAX(err_ph_p[n], fabs(phi_cf.value(xyz)));
          }
        }

        error_ph = MAX(error_ph, err_ph_p[n]);
      }

      foreach_dimension(dim) { ierr = VecRestoreArray(phi_dd[dim], &phi_dd_p[dim]); CHKERRXX(ierr); }

      int mpiret;
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_tm, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_tp, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_c0, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_c1, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_ph, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);


      ierr = VecRestoreArray(tl, &tl_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(ts, &ts_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(c0, &c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(c1, &c1_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

      const char* out_dir = getenv("OUT_DIR");
      if (!out_dir)
      {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
      }

      char name[1000];
    #ifdef P4_TO_P8
      sprintf(name, "%s/vtu/bialloy_error_lvl_%d_%d_%d_%dx%dx%d.%05d", out_dir, lmin, lmax, p4est->mpisize, brick.nxyztrees[0], brick.nxyztrees[1], brick.nxyztrees[2], iteration/save_every_n_iteration);
    #else
      sprintf(name, "%s/vtu/bialloy_error_lvl_%d_%d_%d_%dx%d.%05d", out_dir, lmin, lmax, p4est->mpisize, brick.nxyztrees[0], brick.nxyztrees[1], iteration/save_every_n_iteration);
    #endif

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


      my_p4est_vtk_write_all(  p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                             #ifdef P4_TO_P8
                               6, 1, name,
                             #else
                               6, 1, name,
                             #endif
                               VTK_POINT_DATA, "phi", phi_ex_p,
                               VTK_POINT_DATA, "err_tm", err_tm_p,
                               VTK_POINT_DATA, "err_tp", err_tp_p,
                               VTK_POINT_DATA, "err_c0", err_c0_p,
                               VTK_POINT_DATA, "err_c1", err_c1_p,
                               VTK_POINT_DATA, "err_ph", err_ph_p,
                               VTK_CELL_DATA , "leaf_level", l_p);



      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "Error VTK saved in %s\n", name);

      ierr = VecRestoreArray(err_tm, &err_tm_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_tp, &err_tp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c0, &err_c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c1, &err_c1_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_ex, &phi_ex_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_ph, &err_ph_p); CHKERRXX(ierr);

      ierr = VecDestroy(err_tm); CHKERRXX(ierr);
      ierr = VecDestroy(err_tp); CHKERRXX(ierr);
      ierr = VecDestroy(err_c0); CHKERRXX(ierr);
      ierr = VecDestroy(err_c1); CHKERRXX(ierr);
      ierr = VecDestroy(phi_ex); CHKERRXX(ierr);
      ierr = VecDestroy(err_ph); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "Error in tm: %e\n", error_tm); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Error in tp: %e\n", error_tp); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Error in c0: %e\n", error_c0); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Error in c1: %e\n", error_c1); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Error in ph: %e\n", error_ph); CHKERRXX(ierr);
    }

    bas.compute_dt();

    iteration++;
  }
  lmin++;
  lmax++;

  }
  w1.stop(); w1.read_duration();

  return 0;
}
