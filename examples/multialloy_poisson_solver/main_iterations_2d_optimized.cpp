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
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_nodes_multialloy.h>
#include <src/my_p8est_interpolation_nodes.h>
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
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX
int lmin = 5;
int lmax = 10;
int nb_splits = 1;

bool use_continuous_stencil = 0;
bool use_one_sided_derivatives = false;
bool use_points_on_interface = 1;
bool update_c0_robin = 0;

bool use_superconvergent_robin = 1;

int pin_every_n_steps = 1000;
double bc_tolerance = 1.e-11;
int max_iterations = 1000;

double lip = 1.5;

using namespace std;

/* 0 - NiCu
 */
int alloy_type = 0;

double box_size = 10.0e-1;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1/box_size;

double xmin = 0;
double ymin = 0;
double zmin = 0;
double xmax = 1;
double ymax = 1;
double zmax = 1;
int n_xyz[] = {1, 1, 1};

double r_0 = 0.3;
double x_0 = 0.5+0.01;
double y_0 = 0.5-0.03;
double z_0 = 0.5+0.07;

const int num_comps = 2;
double rho;                  /* density                                    - kg.cm-3      */
double heat_capacity;        /* c, heat capacity                           - J.kg-1.K-1   */
double Tm;                   /* melting temperature                        - K            */
double G;                    /* thermal gradient                           - k.cm-1       */
double V;                    /* cooling velocity                           - cm.s-1       */
double latent_heat;          /* L, latent heat                             - J.cm-3       */
double thermal_conductivity; /* k, thermal conductivity                    - W.cm-1.K-1   */
double lambda;               /* thermal diffusivity                        - cm2.s-1      */

double ml[num_comps];         /* liquidus slope                             - K / at frac. */
double kp[num_comps];         /* partition coefficient                                     */
double c0[num_comps];         /* initial concentration                      - at frac.     */
double Dl[num_comps];         /* liquid concentration diffusion coefficient - cm2.s-1      */

double eps_c;                /* curvature undercooling coefficient         - cm.K         */
double eps_v;                /* kinetic undercooling coefficient           - s.K.cm-1     */
double eps_anisotropy;       /* anisotropy coefficient                                    */

double cfl_number = 0.1;
double dt = 0.333;

void set_alloy_parameters()
{
  switch(alloy_type)
  {
  case 0:
    /* those are the default parameters for Ni-0.25831at%Cu-0.15at%Cu = Ni-0.40831at%Cu */
    rho                  = 8.88e-3;        /* kg.cm-3    */
    heat_capacity        = 0.46e3;         /* J.kg-1.K-1 */
    Tm                   = 1728;           /* K           */
    G                    = 4e2;            /* k.cm-1      */
    V                    = 0.01;           /* cm.s-1      */
    latent_heat          = 2350;           /* J.cm-3      */
    thermal_conductivity = 6.07e-1;        /* W.cm-1.K-1  */
//    latent_heat          = 1;           /* J.cm-3      */
//    thermal_conductivity = 1;        /* W.cm-1.K-1  */
//    thermal_conductivity = 100;        /* W.cm-1.K-1  */
    lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
//    lambda = 1;
//    eps_c                = 2.7207e-5;
//    eps_v                = 2.27e-2;
//    eps_v                = 1;
//    eps_c                = 1;
    eps_c                = 0;
    eps_v                = 0;
    eps_anisotropy       = 0.05;

    Dl[0] = 1e-1;
//    Dl[0] = 1;
//    Dl[0] = 1;
    ml[0] =-357;
//    ml[0] =-1;
    c0[0] = 0.4;
    kp[0] = 0.86;

    Dl[1] = 2e-1;
//    Dl[1] = 1e-1;
//    Dl[1] = 1;
    ml[1] =-357;
//    ml[1] =-1;
    c0[1] = 0.4;
    kp[1] = 0.86;

    break;
  case 1:
      break;
  }
}

bool save_vtk = true;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 -
 */
int test_number = 0;


double diag_add = 1;

double solidus_value(double *c)
{
  return Tm;
}

void   solidus_slope(double *c, double *m)
{
}

//#ifdef P4_TO_P8
//class eps_v_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return eps_v*(1.0-15.0*eps_anisotropy*.5*(cos(4.0*x)+cos(4.0*y)));
//  }
//} eps_v_cf;

//class eps_c_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return eps_c*(1.0-15.0*eps_anisotropy*.5*(cos(4.0*x)+cos(4.0*y)));
//  }
//} eps_c_cf;
//#else
//class eps_v_cf_t : public CF_1
//{
//public:
//  double operator()(double x) const
//  {
//    return eps_v*(1.0-15.0*eps_anisotropy*cos(4.0*x));
//  }
//} eps_v_cf;

//class eps_c_cf_t : public CF_1
//{
//public:
//  double operator()(double x) const
//  {
//    return eps_c*(1.0-15.0*eps_anisotropy*cos(4.0*x));
//  }
//} eps_c_cf;
//#endif

#ifdef P4_TO_P8
class eps_v_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
//    double theta_xz = atan2(nz, nx);
//    double theta_yz = atan2(nz, ny);
//    return eps_v*(1.0-15.0*eps_anisotropy*.5*(cos(4.0*theta_xz)+cos(4.0*theta_yz)));
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_v*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_v_cf;

class eps_c_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
//    double theta_xz = atan2(nz, nx);
//    double theta_yz = atan2(nz, ny);
//    return eps_c*(1.0-15.0*eps_anisotropy*.5*(cos(4.0*theta_xz)+cos(4.0*theta_yz)));
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_c*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_c_cf;
#else
class eps_v_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_v*(1.0-15.0*eps_anisotropy*cos(4.0*theta));
  }
} eps_v_cf;

class eps_c_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_c*(1.0-15.0*eps_anisotropy*cos(4.0*theta));
  }
} eps_c_cf;
#endif


#ifdef P4_TO_P8
class zero_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return 0;
  }
} zero_cf;
#else
class zero_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
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
    switch(interface_type)
    {
    case 0:
      return r_0 - sqrt(SQR(x - x_0) + SQR(y - y_0));
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
    switch(interface_type)
    {
    case 0:
      return -(x-x_0)/sqrt(SQR(x - x_0) + SQR(y - y_0) + EPS);
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
    switch(interface_type)
    {
    case 0:
      return -(y-y_0)/sqrt(SQR(x - x_0) + SQR(y - y_0) + EPS);
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
    switch(interface_type)
    {
    case 0:
      return -(y-y_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0) + EPS, 1.5);
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
    switch(interface_type)
    {
    case 0:
        return (x-x_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0) + EPS, 1.5);
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
    switch(interface_type)
    {
    case 0:
        return -(x-x_0)*(x-x_0)/pow(SQR(x - x_0) + SQR(y - y_0) + EPS, 1.5);
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_yy_cf;
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


#ifdef P4_TO_P8
class theta_xz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double norm = sqrt(SQR(phi_x_cf(x,y,z)) +
                       SQR(phi_y_cf(x,y,z)) +
                       SQR(phi_z_cf(x,y,z))) + EPS;

    return atan2(phi_z_cf(x,y,z)/norm,
                 phi_x_cf(x,y,z)/norm);
  }
} theta_xz_cf;
class theta_yz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double norm = sqrt(SQR(phi_x_cf(x,y,z)) +
                       SQR(phi_y_cf(x,y,z)) +
                       SQR(phi_z_cf(x,y,z))) + EPS;

    return atan2(phi_z_cf(x,y,z)/norm,
                 phi_y_cf(x,y,z)/norm);
  }
} theta_yz_cf;
#else
class theta_xz_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double norm = sqrt(SQR(phi_x_cf(x,y)) + SQR(phi_y_cf(x,y))+EPS);
    return atan2(phi_y_cf(x,y)/norm, phi_x_cf(x,y)/norm);
  }
} theta_xz_cf;
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
        return 0.5*(1.1+sin(x)*cos(y));
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
        return 0.5*cos(x)*cos(y);
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
        return -0.5*sin(x)*sin(y);
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
        return -2.*0.5*sin(x)*cos(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_dd_exact;
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
        return cos(x)*sin(y);
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
      case 0: return -sin(x)*sin(y);
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
        return cos(x)*cos(y);
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
        return -2.*cos(x)*sin(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_dd_exact;
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
        return x*x + x*y + y*y;
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
        return 2.*x + y;
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
        return x + 2.*y;
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
        return 4.;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_dd_exact;
#endif


#ifdef P4_TO_P8
//class tm_exact_t : public CF_3
//{
//public:
//  double operator()(double x, double y, double z) const
//  {
//    switch(test_number)
//    {
//      case 0:
//        return 2.*sin(x)*(y+y*y)*cos(z)+3.;
//      default:
//        throw std::invalid_argument("Choose a valid test.");
//    }
//  }
//} tm_exact;

//class tm_x_exact_t : public CF_3
//{
//public:
//  double operator()(double x, double y, double z) const
//  {
//    switch(test_number)
//    {
//      case 0:
//        return 2.*cos(x)*(y+y*y)*cos(z);
//      default:
//        throw std::invalid_argument("Choose a valid test.");
//    }
//  }
//} tm_x_exact;

//class tm_y_exact_t : public CF_3
//{
//public:
//  double operator()(double x, double y, double z) const
//  {
//    switch(test_number)
//    {
//      case 0:
//        return 2.*sin(x)*(1.+2.*y)*cos(z);
//      default:
//        throw std::invalid_argument("Choose a valid test.");
//    }
//  }
//} tm_y_exact;

//class tm_z_exact_t : public CF_3
//{
//public:
//  double operator()(double x, double y, double z) const
//  {
//    switch(test_number)
//    {
//      case 0:
//        return -2.*sin(x)*(y+y*y)*sin(z);
//      default:
//        throw std::invalid_argument("Choose a valid test.");
//    }
//  }
//} tm_z_exact;

//class tm_dd_exact_t : public CF_3
//{
//public:
//  double operator()(double x, double y, double z) const
//  {
//    switch(test_number)
//    {
//      case 0:
//        return 2.*(-2.*sin(x)*(y+y*y)*cos(z) + 2.*sin(x)*cos(z));
//      default:
//        throw std::invalid_argument("Choose a valid test.");
//    }
//  }
//} tm_dd_exact;

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
        return sin(x)*(y+y*y);
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
        return cos(x)*(y+y*y);
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
        return sin(x)*(1.+2.*y);
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
        return -sin(x)*(y+y*y) + 2.*sin(x);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_dd_exact;
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
    return diag_add*c0_exact(x,y,z) - dt*Dl[0]*c0_dd_exact(x,y,z);
  }
} rhs_c0_cf;

class rhs_c1_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
//    if (use_superconvergent_robin)
//      return diag_add*c1_exact(x,y,z)/dt - Dl[1]*c1_dd_exact(x,y,z);
//    else
      return diag_add*c1_exact(x,y,z) - dt*Dl[1]*c1_dd_exact(x,y,z);
  }
} rhs_c1_cf;

class rhs_tm_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return diag_add*tm_exact(x,y,z) - dt*lambda*tm_dd_exact(x,y,z);
  }
} rhs_tm_cf;

class rhs_tp_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return diag_add*tp_exact(x,y,z) - dt*lambda*tp_dd_exact(x,y,z);
  }
} rhs_tp_cf;
#else
class rhs_c0_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*c0_exact(x,y) - dt*Dl[0]*c0_dd_exact(x,y);
  }
} rhs_c0_cf;

class rhs_c1_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    if (use_superconvergent_robin)
//      return diag_add*c1_exact(x,y)/dt - Dl[1]*c1_dd_exact(x,y);
//    else
      return diag_add*c1_exact(x,y) - dt*Dl[1]*c1_dd_exact(x,y);
  }
} rhs_c1_cf;

class rhs_tm_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*tm_exact(x,y) - dt*lambda*tm_dd_exact(x,y);
  }
} rhs_tm_cf;

class rhs_tp_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*tp_exact(x,y) - dt*lambda*tp_dd_exact(x,y);
  }
} rhs_tp_cf;
#endif

#ifdef P4_TO_P8
class vn_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double nx = phi_x_cf(x,y,z);
    double ny = phi_y_cf(x,y,z);
    double nz = phi_z_cf(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz)+EPS;
    nx /= norm; ny /= norm; nz /= norm;
    return (thermal_conductivity/latent_heat*((tm_x_exact(x,y,z) - tp_x_exact(x,y,z))*nx +
                                              (tm_y_exact(x,y,z) - tp_y_exact(x,y,z))*ny +
                                              (tm_z_exact(x,y,z) - tp_z_exact(x,y,z))*nz));
  }
} vn_cf;
#else
class vn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -0.01*cos(x)*sin(y);
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny)+EPS;
    nx /= norm; ny /= norm;
    return (thermal_conductivity/latent_heat*((tm_x_exact(x,y) - tp_x_exact(x,y))*nx +
                                              (tm_y_exact(x,y) - tp_y_exact(x,y))*ny));
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


//class jump_psi_tn_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return 1./ml[0]/lambda;
//  }
//} jump_psi_tn_cf;


//------------------------------------------------------------
// bc for c1
//------------------------------------------------------------
#ifdef P4_TO_P8
class c1_robin_coef_cf_t : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
//    return 1.0;
    return (1.-kp[1])/Dl[1]*vn_cf(x,y,z);
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
    if (use_superconvergent_robin)
      return dt*Dl[1]*(c1_x_exact(x,y,z)*nx + c1_y_exact(x,y,z)*ny + c1_z_exact(x,y,z)*nz + c1_robin_coef_cf(x,y,z)*c1_exact(x,y,z));
    else
      return c1_x_exact(x,y,z)*nx + c1_y_exact(x,y,z)*ny + c1_z_exact(x,y,z)*nz + c1_robin_coef_cf(x,y,z)*c1_exact(x,y,z);
  }
} c1_interface_val_cf;
#else
class c1_robin_coef_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return 1.0;
    return (1.-kp[1])/Dl[1]*vn_cf(x,y);
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
    if (use_superconvergent_robin)
      return dt*Dl[1]*(c1_x_exact(x,y)*nx + c1_y_exact(x,y)*ny + c1_robin_coef_cf(x,y)*c1_exact(x,y));
    else
      return c1_x_exact(x,y)*nx + c1_y_exact(x,y)*ny + c1_robin_coef_cf(x,y)*c1_exact(x,y);
  }
} c1_interface_val_cf;
#endif

//class psi_c1_interface_val_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return -ml[1]/ml[0]/Dl[1];
//  }
//} psi_c1_interface_val_cf;


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
    return c0_x_exact(x,y,z)*nx + c0_y_exact(x,y,z)*ny + c0_z_exact(x,y,z)*nz + (1.-kp[0])/Dl[0]*vn_cf(x,y,z)*c0_exact(x,y,z);
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
    return c0_x_exact(x,y)*nx + c0_y_exact(x,y)*ny + (1.-kp[0])/Dl[0]*vn_cf(x,y)*c0_exact(x,y);
  }
} c0_interface_val_cf;
#endif


//class vn_from_c0_t : public CF_2
//{
//  CF_2 *c0, *dc0dn;
//public:
//  void set_input(CF_2& c0_, CF_2& dc0dn_) {c0 = &c0_; dc0dn = &dc0dn_;}
//  double operator()(double x, double y) const
//  {
//    return Dl[0]/(1.-kp[0])*((*dc0dn)(x,y) - c0_interface_val_cf(x,y))/(*c0)(x,y);
//  }
//} vn_from_c0;

//class c1_robin_from_c0_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return -(1.-kp[1])/Dl[1]*vn_from_c0(x,y);
//  }
//} c1_robin_from_c0;

//class tn_jump_from_c0_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return latent_heat/thermal_conductivity*vn_from_c0(x,y);
//  }
//} tn_jump_from_c0;



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
    return c0_exact(x,y)+0.03*sin(5*y)*cos(5*x)-0.05;
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
    return tm_exact(x,y,z) - Tm - ml[0]*c0_exact(x,y,z) - ml[1]*c1_exact(x,y,z) - eps_c_cf(phi_x_cf(x,y,z), phi_y_cf(x,y,z), phi_z_cf(x,y,z))*kappa_cf(x,y,z) + eps_v_cf(phi_x_cf(x,y,z), phi_y_cf(x,y,z), phi_z_cf(x,y,z))*vn_cf(x,y,z);
  }
} gibbs_thompson;
#else
class gibbs_thompson_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_exact(x,y) - Tm - ml[0]*c0_exact(x,y) - ml[1]*c1_exact(x,y) - eps_c_cf(phi_x_cf(x,y), phi_y_cf(x,y))*kappa_cf(x,y) + eps_v_cf(phi_x_cf(x,y), phi_y_cf(x,y))*vn_cf(x,y);
  }
} gibbs_thompson;
#endif



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
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
//                 2 - sin(x)*cos(y)\n\
//                 3 - sin(x) + cos(y)");
//#endif
  cmd.parse(argc, argv);

//  lmin = cmd.get("lmin", lmin);
//  lmax = cmd.get("lmax", lmax);
//  nb_splits = cmd.get("nb_splits", nb_splits);
//  test_number = cmd.get("test", test_number);

//  bc_wtype = cmd.get("bc_wtype", bc_wtype);
//  bc_itype = cmd.get("bc_itype", bc_itype);

//  save_vtk = cmd.get("save_vtk", save_vtk);

  set_alloy_parameters();

  scaling = 1/box_size;
  rho                  /= (scaling*scaling*scaling);
  thermal_conductivity /= scaling;
  G                    /= scaling;
  V                    *= scaling;
  latent_heat          /= (scaling*scaling*scaling);
  eps_c                *= scaling;
  eps_v                /= scaling;
  lambda                = thermal_conductivity/(rho*heat_capacity);


  for (short i = 0; i < num_comps; ++i)
    Dl[i]              *= (scaling*scaling);

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

  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};
  const int periodic[] = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_tm_n, err_tm_nm1;
  double err_tp_n, err_tp_nm1;
  double err_c0_n, err_c0_nm1;
  double err_c1_n, err_c1_nm1;
  double err_vn_n, err_vn_nm1;
  double err_kappa_n, err_kappa_nm1;
  double err_theta_n, err_theta_nm1;

//  double err_ex_n;
//  double err_ex_nm1;

  vector<double> h, e_v, e_tm, e_tp, e_c0, e_c1, e_g, error_it, pdes_it;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    e_g.clear();

    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data_tmp(lmin, lmax, &phi_cf, lip);
    p4est->user_pointer = (void*)(&data_tmp);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    for (int i = 0; i < iter; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
//      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &phi_cf, lip);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    if (use_one_sided_derivatives || use_continuous_stencil)
      my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);
    ngbd_n.init_neighbors();

    my_p4est_level_set_t ls(&ngbd_n);

    /* find dx and dy smallest */
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
    double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    // compute dt
//    dt = cfl_number*diag/V;

    /* Initialize LSF */
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_cf, phi);

    if (0) {
      double xyz[P4EST_DIM];
      double *phi_ptr;
      srand(mpi.rank());

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);

        double d_phi = 0.000*dx*dx*dx*(double)(rand()%1000)/1000.;
//        double d_phi = 0.01*dx*dx*cos(2.*PI*data.max_lvl*xyz[0])*sin(2.*PI*data.max_lvl*xyz[1]);
        phi_ptr[n] += d_phi;
      }
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
      ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);
    }

//    ls.reinitialize_1st_order_time_2nd_order_space(phi, 100);
//    ls.reinitialize_1st_order(phi, 100);
//    ls.reinitialize_2nd_order(phi, 100);
//    ls.perturb_level_set_function(phi, EPS);

    Vec phi_dd[P4EST_DIM];

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_dd[dir]); CHKERRXX(ierr);
    }

    ngbd_n.second_derivatives_central(phi, phi_dd);

    // compute normal
    Vec normal[P4EST_DIM];
    Vec kappa; ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);
    Vec theta_xz; ierr = VecDuplicate(phi, &theta_xz); CHKERRXX(ierr);
    Vec theta_yz; ierr = VecDuplicate(phi, &theta_yz); CHKERRXX(ierr);

    double *normal_p[P4EST_DIM];
    {
      Vec kappa_tmp; ierr = VecDuplicate(phi, &kappa_tmp); CHKERRXX(ierr);
      const double *phi_p;

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &normal[dir]); CHKERRXX(ierr);
      }

      compute_normals_and_mean_curvature(ngbd_n, phi, normal, kappa_tmp);

      my_p4est_level_set_t ls(&ngbd_n);
//      ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
//      ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

            ierr = VecDestroy(kappa); CHKERRXX(ierr);
            kappa = kappa_tmp;

      //      sample_cf_on_nodes(p4est, nodes, kappa_cf, kappa);

      /* angle between normal and direction of growth */

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

#ifdef P4_TO_P8
      Vec theta_xz_tmp; double *theta_xz_tmp_p;
      Vec theta_yz_tmp; double *theta_yz_tmp_p;
      ierr = VecDuplicate(phi, &theta_xz_tmp); CHKERRXX(ierr);
      ierr = VecDuplicate(phi, &theta_yz_tmp); CHKERRXX(ierr);
      ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
      ierr = VecGetArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
#else
      Vec theta_xz_tmp; double *theta_xz_tmp_p;
      ierr = VecDuplicate(phi, &theta_xz_tmp); CHKERRXX(ierr);
      ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
#endif


      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
#ifdef P4_TO_P8
        theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
        theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
#else
        theta_xz_tmp_p[n] = atan2(normal_p[1][n], normal_p[0][n]);
#endif
      }

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

#ifdef P4_TO_P8
      //      ierr = VecRestoreArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
      //      ierr = VecRestoreArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
      //      ls.extend_from_interface_to_whole_domain_TVD(phi, theta_xz_tmp, theta_xz);
      //      ls.extend_from_interface_to_whole_domain_TVD(phi, theta_yz_tmp, theta_yz);
      //      ierr = VecDestroy(theta_xz_tmp); CHKERRXX(ierr);
      //      ierr = VecDestroy(theta_yz_tmp); CHKERRXX(ierr);
      ierr = VecDestroy(theta_xz); CHKERRXX(ierr);
      ierr = VecDestroy(theta_yz); CHKERRXX(ierr);
      theta_xz = theta_xz_tmp;
      theta_yz = theta_yz_tmp;
#else
      //  ierr = VecRestoreArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
      //  ls.extend_from_interface_to_whole_domain_TVD(phi, theta_tmp, theta);
      //  ierr = VecDestroy(theta_tmp); CHKERRXX(ierr);
      ierr = VecDestroy(theta_xz); CHKERRXX(ierr);
      theta_xz = theta_xz_tmp;
#endif

    }


    /* Sample RHS */
    Vec rhs_c0; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c0); CHKERRXX(ierr);
    Vec rhs_c1; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c1); CHKERRXX(ierr);
    Vec rhs_tm; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tm); CHKERRXX(ierr);
    Vec rhs_tp; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tp); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, rhs_c0_cf, rhs_c0);
    sample_cf_on_nodes(p4est, nodes, rhs_c1_cf, rhs_c1);
    sample_cf_on_nodes(p4est, nodes, rhs_tm_cf, rhs_tm);
    sample_cf_on_nodes(p4est, nodes, rhs_tp_cf, rhs_tp);

    /* set boundary conditions */
#ifdef P4_TO_P8
    BoundaryConditions3D bc_t;
    BoundaryConditions3D bc_c0;
    BoundaryConditions3D bc_c1;
#else
    BoundaryConditions2D bc_t;
    BoundaryConditions2D bc_c0;
    BoundaryConditions2D bc_c1;
#endif

    bc_t.setWallTypes(bc_wall_type_t);
    bc_t.setWallValues(t_exact);

    bc_c0.setWallTypes(bc_wall_type_c0);
    bc_c0.setWallValues(c0_exact);

    bc_c1.setWallTypes(bc_wall_type_c1);
    bc_c1.setWallValues(c1_exact);

    my_p4est_poisson_nodes_multialloy_t solver_all_in_one(&ngbd_n);

    solver_all_in_one.set_phi(phi, phi_dd, normal, kappa);
    solver_all_in_one.set_parameters(dt, lambda, thermal_conductivity, latent_heat, Tm, Dl[0], kp[0], ml[0], Dl[1], kp[1], ml[1]);
    solver_all_in_one.set_bc(bc_t, bc_c0, bc_c1);
    solver_all_in_one.set_GT(gibbs_thompson);
    solver_all_in_one.set_undercoolings(eps_v_cf, eps_c_cf);
    solver_all_in_one.set_pin_every_n_steps(pin_every_n_steps);
    solver_all_in_one.set_tolerance(bc_tolerance, max_iterations);
    solver_all_in_one.set_rhs(rhs_tm, rhs_tp, rhs_c0, rhs_c1);

    solver_all_in_one.set_use_continuous_stencil(use_continuous_stencil);
    solver_all_in_one.set_use_one_sided_derivatives(use_one_sided_derivatives);
    solver_all_in_one.set_use_points_on_interface(use_points_on_interface);
    solver_all_in_one.set_update_c0_robin(update_c0_robin);

    solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin);

    solver_all_in_one.set_jump_t(jump_t_cf);
    solver_all_in_one.set_jump_tn(jump_tn_cf);
    solver_all_in_one.set_flux_c(c0_interface_val_cf, c1_interface_val_cf);
    solver_all_in_one.set_c0_guess(c0_guess);

    solver_all_in_one.set_vn(vn_cf);

    Vec sol_tm; ierr = VecCreateGhostNodes(p4est, nodes, &sol_tm); CHKERRXX(ierr);
    Vec sol_tp; ierr = VecCreateGhostNodes(p4est, nodes, &sol_tp); CHKERRXX(ierr);
    Vec sol_c0; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c0); CHKERRXX(ierr);
    Vec sol_c1; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c1); CHKERRXX(ierr);

    Vec sol_tm_dd[P4EST_DIM];
    Vec sol_tp_dd[P4EST_DIM];
    Vec sol_c0_dd[P4EST_DIM];
    Vec sol_c1_dd[P4EST_DIM];
    Vec sol_c0n[P4EST_DIM];
    Vec bc_error;

    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDuplicate(phi_dd[dim], &sol_tm_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_tp_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_c0_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_c1_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_c0n[dim]); CHKERRXX(ierr);
    }

    ierr = VecDuplicate(phi, &bc_error); CHKERRXX(ierr);


    double bc_error_max = 0;
//    solver_all_in_one.solve(sol_t, sol_t_dd, sol_c0, sol_c0_dd, sol_c1, sol_c1_dd, bc_error_max, bc_error);
    solver_all_in_one.solve(sol_tm, sol_tp, sol_c0, sol_c1, sol_c0n, bc_error, bc_error_max, dt, 1.e10, false, &pdes_it, &error_it);


    /* check the error */
    my_p4est_poisson_nodes_t *solver_c0 = solver_all_in_one.get_solver_c0();
#ifdef P4_TO_P8
    CF_3 *vn = solver_all_in_one.get_vn();
#else
    CF_2 *vn = solver_all_in_one.get_vn();
#endif

    double vn_error = 0;
    double kappa_error = 0;
    double theta_xz_error = 0;

    double *kappa_p;    ierr = VecGetArray(kappa,    &kappa_p);    CHKERRXX(ierr);
    double *theta_xz_p; ierr = VecGetArray(theta_xz, &theta_xz_p); CHKERRXX(ierr);

    Vec err_vn;    ierr = VecDuplicate(phi,  &err_vn);    CHKERRXX(ierr);
    Vec err_kappa; ierr = VecDuplicate(phi,  &err_kappa); CHKERRXX(ierr);

    double *err_kappa_p; ierr = VecGetArray(err_kappa, &err_kappa_p); CHKERRXX(ierr);
    double *err_vn_p;    ierr = VecGetArray(err_vn,    &err_vn_p);    CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    }

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      err_kappa_p[n] = 0;
      err_vn_p[n] = 0;
      for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
      {
        double xyz[P4EST_DIM];
        solver_c0->get_xyz_interface_point(n, i, xyz);
        vn_error = MAX(vn_error, fabs(vn->value(xyz) - vn_cf.value(xyz)));
        kappa_error = MAX(kappa_error, fabs(kappa_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, kappa_p)));
//        theta_xz_error = MAX(theta_xz_error, fabs(theta_xz_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, theta_xz_p)));

        double nx = phi_x_cf.value(xyz);
        double ny = phi_y_cf.value(xyz);
#ifdef P4_TO_P8
        double nz = phi_z_cf.value(xyz);
        double norm = sqrt(nx*nx+ny*ny+nz*nz)+EPS;
#else
        double norm = sqrt(nx*nx+ny*ny)+EPS;
#endif

        double normal_error = sqrt(SQR(nx/norm - solver_c0->interpolate_at_interface_point(n, i, normal_p[0])) +
    #ifdef P4_TO_P8
                                   SQR(nz/norm - solver_c0->interpolate_at_interface_point(n, i, normal_p[2])) +
    #endif
                                   SQR(ny/norm - solver_c0->interpolate_at_interface_point(n, i, normal_p[1])));

        theta_xz_error = MAX(theta_xz_error, normal_error);

        err_kappa_p[n] = MAX(err_kappa_p[n], fabs(kappa_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, kappa_p)));
        err_vn_p[n] = MAX(err_vn_p[n], fabs(theta_xz_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, theta_xz_p)));
      }
    }
    ierr = VecRestoreArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_vn,  &err_vn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta_xz, &theta_xz_p); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    }

    err_kappa_nm1 = err_kappa_n; err_kappa_n = kappa_error;
    err_theta_nm1 = err_theta_n; err_theta_n = theta_xz_error;
    err_vn_nm1 = err_vn_n; err_vn_n = vn_error;
    err_tm_nm1 = err_tm_n; err_tm_n = 0;
    err_tp_nm1 = err_tp_n; err_tp_n = 0;
    err_c0_nm1 = err_c0_n; err_c0_n = 0;
    err_c1_nm1 = err_c1_n; err_c1_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    const double *sol_tm_p; ierr = VecGetArrayRead(sol_tm, &sol_tm_p); CHKERRXX(ierr);
    const double *sol_tp_p; ierr = VecGetArrayRead(sol_tp, &sol_tp_p); CHKERRXX(ierr);
    const double *sol_c0_p; ierr = VecGetArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    const double *sol_c1_p; ierr = VecGetArrayRead(sol_c1, &sol_c1_p); CHKERRXX(ierr);

    Vec err_tm_nodes; ierr = VecDuplicate(sol_tm, &err_tm_nodes); CHKERRXX(ierr);
    Vec err_tp_nodes; ierr = VecDuplicate(sol_tp, &err_tp_nodes); CHKERRXX(ierr);
    Vec err_c0_nodes; ierr = VecDuplicate(sol_c0, &err_c0_nodes); CHKERRXX(ierr);
    Vec err_c1_nodes; ierr = VecDuplicate(sol_c1, &err_c1_nodes); CHKERRXX(ierr);

    double *err_tm_p; ierr = VecGetArray(err_tm_nodes, &err_tm_p); CHKERRXX(ierr);
    double *err_tp_p; ierr = VecGetArray(err_tp_nodes, &err_tp_p); CHKERRXX(ierr);
    double *err_c0_p; ierr = VecGetArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
    double *err_c1_p; ierr = VecGetArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

    double xyz[P4EST_DIM];
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if(phi_p[n]<0)
      {
        err_tm_p[n] = fabs(sol_tm_p[n] - tm_exact.value(xyz));
        err_tp_p[n] = 0;
        err_c0_p[n] = fabs(sol_c0_p[n] - c0_exact.value(xyz));
        err_c1_p[n] = fabs(sol_c1_p[n] - c1_exact.value(xyz));
      } else {
        err_tm_p[n] = 0;
        err_tp_p[n] = fabs(sol_tp_p[n] - tp_exact.value(xyz));
        err_c0_p[n] = 0.;
        err_c1_p[n] = 0.;
      }

      err_tm_n = MAX(err_tm_n, err_tm_p[n]);
      err_tp_n = MAX(err_tp_n, err_tp_p[n]);
      err_c0_n = MAX(err_c0_n, err_c0_p[n]);
      err_c1_n = MAX(err_c1_n, err_c1_p[n]);
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_tm, &sol_tm_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_tp, &sol_tp_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_c1, &sol_c1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(err_tm_nodes, &err_tm_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_tp_nodes, &err_tp_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_tm_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_tm_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_tp_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_tp_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_c0_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c0_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_c1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_kappa_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_theta_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_vn_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tm_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tp_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c0_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c1_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    ierr = PetscPrintf(p4est->mpicomm, "Error in kappa on nodes : %g, order = %g\n", err_kappa_n, log(err_kappa_nm1/err_kappa_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in theta on nodes : %g, order = %g\n", err_theta_n, log(err_theta_nm1/err_theta_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in vn on nodes : %g, order = %g\n", err_vn_n, log(err_vn_nm1/err_vn_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in Tm on nodes : %g, order = %g\n", err_tm_n, log(err_tm_nm1 /err_tm_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in Tp on nodes : %g, order = %g\n", err_tp_n, log(err_tp_nm1 /err_tp_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C0 on nodes : %g, order = %g\n", err_c0_n, log(err_c0_nm1/err_c0_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C1 on nodes : %g, order = %g\n", err_c1_n, log(err_c1_nm1/err_c1_n)/log(2)); CHKERRXX(ierr);

    h.push_back(dx);
    e_v.push_back(err_vn_n);
    e_tm.push_back(err_tm_n);
    e_tp.push_back(err_tp_n);
    e_c0.push_back(err_c0_n);
    e_c1.push_back(err_c1_n);

    //-------------------------------------------------------------------------------------------
    // Save output
    //-------------------------------------------------------------------------------------------
    if(save_vtk)
    {
      PetscErrorCode ierr;
      const char *out_dir = getenv("OUT_DIR");
      if (!out_dir) {
        out_dir = "out_dir";
        system((string("mkdir -p ")+out_dir).c_str());
      }

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/multiallo_poisson_solver_"
          << "proc"
          << p4est->mpisize << "_"
             << "brick"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
//             "_levels=(" <<lmin << "," << lmax << ")" <<
             ".split" << iter;

      double *phi_p;
      double *sol_tm_p, *err_tm_p;
      double *sol_tp_p, *err_tp_p;
      double *sol_c0_p, *err_c0_p;
      double *sol_c1_p, *err_c1_p;

      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      ierr = VecGetArray(sol_tm, &sol_tm_p); CHKERRXX(ierr);
      ierr = VecGetArray(sol_tp, &sol_tp_p); CHKERRXX(ierr);
      ierr = VecGetArray(sol_c0, &sol_c0_p); CHKERRXX(ierr);
      ierr = VecGetArray(sol_c1, &sol_c1_p); CHKERRXX(ierr);

      ierr = VecGetArray(err_tm_nodes, &err_tm_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_tp_nodes, &err_tp_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

      ierr = VecGetArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_vn,  &err_vn_p); CHKERRXX(ierr);

      double *bc_error_p;

      ierr = VecGetArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//      double *err_ex_p;
//      ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

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

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             12, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "sol_tm", sol_tm_p,
                             VTK_POINT_DATA, "sol_tp", sol_tp_p,
                             VTK_POINT_DATA, "sol_c0", sol_c0_p,
                             VTK_POINT_DATA, "sol_c1", sol_c1_p,
                             VTK_POINT_DATA, "err_tm", err_tm_p,
                             VTK_POINT_DATA, "err_tp", err_tp_p,
                             VTK_POINT_DATA, "err_c0", err_c0_p,
                             VTK_POINT_DATA, "err_c1", err_c1_p,
                             VTK_POINT_DATA, "bc_error", bc_error_p,
                             VTK_POINT_DATA, "kappa_error", err_kappa_p,
                             VTK_POINT_DATA, "vn_error", err_vn_p,
//                             VTK_POINT_DATA, "err_ex", err_ex_p,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol_tm, &sol_tm_p);  CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_tp, &sol_tp_p);  CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_c0, &sol_c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_c1, &sol_c1_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(err_tm_nodes, &err_tm_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_tp_nodes, &err_tp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_vn,  &err_vn_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//      ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);


    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDestroy(sol_tm_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_tp_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_c0_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_c1_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_c0n[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(bc_error); CHKERRXX(ierr);

    ierr = VecDestroy(rhs_tm); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_tp); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c0); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c1); CHKERRXX(ierr);

    ierr = VecDestroy(sol_tm); CHKERRXX(ierr);
    ierr = VecDestroy(sol_tp); CHKERRXX(ierr);
    ierr = VecDestroy(sol_c0); CHKERRXX(ierr);
    ierr = VecDestroy(sol_c1); CHKERRXX(ierr);

    ierr = VecDestroy(err_tm_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_tp_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_c0_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_c1_nodes); CHKERRXX(ierr);

//    ierr = VecDestroy(err_ex); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  if (mpi.rank() == 0)
  {
    const char* out_dir = getenv("OUT_DIR");
    if (!out_dir)
    {
      ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save convergence results\n");
      return -1;
    }
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    std::string filename;

    // save level and resolution
    filename = out_dir; filename += "/convergence/h.txt";        save_vector(filename.c_str(), h);
    filename = out_dir; filename += "/convergence/error_v.txt";  save_vector(filename.c_str(), e_v);
    filename = out_dir; filename += "/convergence/error_tm.txt";  save_vector(filename.c_str(), e_tm);
    filename = out_dir; filename += "/convergence/error_tp.txt";  save_vector(filename.c_str(), e_tp);
    filename = out_dir; filename += "/convergence/error_c0.txt";  save_vector(filename.c_str(), e_c0);
    filename = out_dir; filename += "/convergence/error_c1.txt";  save_vector(filename.c_str(), e_c1);
    filename = out_dir; filename += "/convergence/error_pdes.txt";  save_vector(filename.c_str(), pdes_it);
    filename = out_dir; filename += "/convergence/error_error_it.txt";  save_vector(filename.c_str(), error_it);

    for (int i = 0; i < h.size(); ++i)
      std::cout << h[i] << " " << e_v[i] << " " << e_tm[i] << " " << e_tp[i] << " " << e_c0[i] << " " << e_c1[i] << "\n";
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
