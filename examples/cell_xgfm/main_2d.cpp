/*
 * Title: cell_xgfm
 * Description: xgfm solver at cell-centers
 * Author: Raphael Egan
 * Date Created: 10-05-2018
 */

/*
 * Test the cell based p4est_xgfm solver.
 * - solve a poisson equation with jump conditions across an irregular interface
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
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_xgfm_cells.h>
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
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_xgfm_cells.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#include <ctime>

#undef MIN
#undef MAX

struct box
{
  double xmin, xmax, ymin, ymax
#ifdef P4_TO_P8
  , zmin, zmax
#endif
  ;
};


using namespace std;

int lmin_ = 4;
int lmax_ = 4;

int ngrids_ = 4;
int ntree_ = 1;

BoundaryConditionType bc_wtype_ = DIRICHLET;
//BoundaryConditionType bc_wtype_ = NEUMANN;

bool use_second_order_theta_ = false;
//bool use_second_order_theta_ = true;

bool get_integral = true;
bool print_summary = false;

int test_number_ = 0;
/* run the program with the flag -help to know more about the various tests */

bool track_residuals_and_corrections = false;

class LEVEL_SET :
    #ifdef P4_TO_P8
    public CF_3
    #else
    public CF_2
    #endif
{
  box domain;
  int test_nb;
#ifndef P4_TO_P8
  const size_t n_bubbles = 15;
  vector<double> center_bubbles[P4EST_DIM];
  vector<double> radius_bubbles;
  vector<double> theta_bubbles;
  vector<double> dt_bubbles;
  const double min_bubble_radius = 0.005;
  const double max_bubble_radius = 0.02;
#endif
  double xc, yc, x_length, y_length;
#ifdef P4_TO_P8
  double zc, z_length;
#endif
  double xtheta(double xc_tmp, double t) const { return xc_tmp + 0.6*cos(t) - 0.3*cos(3.0*t); }
  double d_xtheta(double t) const { return -0.6*sin(t) + 0.3*3.0*sin(3.0*t); }
  double dd_xtheta(double t) const { return -0.6*cos(t) + 0.3*3.0*3.0*cos(3.0*t); }
  double ytheta(double yc_tmp, double t) const { return yc_tmp + 0.7*sin(t) - 0.07*sin(3.0*t) + 0.2*sin(7.0*t); }
  double d_ytheta(double t) const { return +0.7*cos(t) - 0.07*3.0*cos(3.0*t) + 0.2*7.0*cos(7.0*t); }
  double dd_ytheta(double t) const { return -0.7*sin(t) + 0.07*3.0*3.0*sin(3.0*t) - 0.2*7.0*7.0*sin(7.0*t); }

  double bone_shaped_ls(double xc_tmp, double yc_tmp, double x, double y) const
  {
    double x_tmp        = xc_tmp + fabs(x - xc_tmp);
    double y_tmp        = yc_tmp + fabs(y - yc_tmp);
    size_t n_sample     = 21;
    double theta_start  = 0.0;
    double theta_end    = 0.5*M_PI;
    double dist_min     = +DBL_MAX;
    double theta, theta_opt, distance; theta_opt = 0.0;
    while (theta_end - theta_start > 0.005*.5*M_PI) {
      for (size_t kk = 0; kk < n_sample; ++kk) {
        theta           = theta_start + ((double) kk)*(theta_end - theta_start)/((double) n_sample);
        distance        = sqrt(SQR(xtheta(xc_tmp, theta) - x_tmp) + SQR(ytheta(yc_tmp, theta) - y_tmp));
        if(distance <= dist_min)
        {
          theta_opt     = theta;
          dist_min      = distance;
        }
      }
      double dtheta     = (theta_end - theta_start)/((double) n_sample);
      theta_end         = theta_opt + dtheta;
      theta_start       = theta_opt - dtheta;
    }

    double corr = DBL_MAX;
    double xt, yt, d_xt, d_yt, dd_xt, dd_yt;
    uint counter = 0;
    while (abs(corr) > EPS*0.5*M_PI)
    {
      xt        = xtheta(xc_tmp, theta_opt);
      yt        = ytheta(yc_tmp, theta_opt);
      d_xt      = d_xtheta(theta_opt);
      dd_xt     = dd_xtheta(theta_opt);
      d_yt      = d_ytheta(theta_opt);
      dd_yt     = dd_ytheta(theta_opt);
      corr      = - ((xt - x_tmp)*d_xt + (yt - y_tmp)*d_yt)/
          (SQR(d_xt) + (xt - x_tmp)*dd_xt + SQR(d_yt) + (yt-y_tmp)*dd_yt);
      theta_opt += (((++counter>20) && (fabs(corr) < EPS*5.0*M_PI))?0.5:1.0)*corr; // relaxation needed on very fine grids (oscillatory behavior - and no convergence - observed on a 14/14 grid)
    }
    xt = xtheta(xc_tmp, theta_opt);
    yt = ytheta(yc_tmp, theta_opt);
    dist_min = sqrt(SQR(x_tmp - xt) + SQR(y_tmp - yt));

    bool is_in = false;
    if (x_tmp >(xc_tmp + sqrt(5.0/12.0)))
      is_in = false;
    else
    {
      double cosroot[3];
      for (short kk = 0; kk < 3; ++kk)
        cosroot[kk] = 2.0*sqrt(5.0/12.0)*cos((1.0/3.0)*acos(-(x_tmp - xc_tmp)*sqrt(12.0/5.0)) - 2.0*M_PI*((double) kk)/3.0);
      double cosroot_tmp;
      double root;
      double y_lim[2] = {yc_tmp, yc_tmp};
      int pp = 1;
      for (short kk = 0; kk < 3; ++kk) {
        for (short jj = kk+1; jj < 3; ++jj) {
          if(cosroot[jj] < cosroot[kk])
          {
            cosroot_tmp = cosroot[kk];
            cosroot[kk] = cosroot[jj];
            cosroot[jj] = cosroot_tmp;
          }
        }
        if((cosroot[kk] >= 0.0) && (cosroot[kk] <= 1.0) && pp >=0)
        {
          root = acos(cosroot[kk]);
          if(fabs(xtheta(xc_tmp, root) - x_tmp) > 10.0*EPS)
            std::cout << "this can't be..." << std::endl;
          y_lim[pp--] = ytheta(yc_tmp, root);
        }
      }
      is_in = ((y_tmp >= y_lim[0]) && (y_tmp <= y_lim[1]));
    }
    return ((is_in)?-1.0:1.0)*dist_min;
  }

public:
  LEVEL_SET(box domain_, int test_nb_) : domain(domain_), test_nb(test_nb_)
  {
    xc        = .5*(domain.xmin+domain.xmax);
    yc        = .5*(domain.ymin+domain.ymax);
    x_length  = (domain.xmax - domain.xmin);
    y_length  = (domain.ymax - domain.ymin);
#ifdef P4_TO_P8
    zc        = .5*(domain.zmin+domain.zmax);
    z_length  = (domain.zmax - domain.zmin);
    if(test_nb == 3)
    {
      xc      = domain.xmin + 0.15*x_length*.5*sqrt(2.0);
      yc      = domain.ymin + 0.15*y_length*.5*sqrt(2.0);
      zc      = domain.zmin + 0.20*z_length;
    }
#else
    if(test_nb == 8)
      xc      = domain.xmin + 0.15*x_length;
    if(test_nb == 9)
    {
      xc      = domain.xmin + 0.15*x_length;
      yc      = domain.ymin + 0.20*y_length;
    }
#endif

#ifndef P4_TO_P8
    if(test_nb == 7)
    {
      srand(time(0));
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        center_bubbles[dim].resize(n_bubbles);
      radius_bubbles.resize(n_bubbles);
      theta_bubbles.resize(n_bubbles);
      dt_bubbles.resize(n_bubbles);
      for (size_t k = 0; k < n_bubbles; ++k)
      {
        center_bubbles[0][k]  = domain.xmin + 0.03 + (x_length-2.0*0.03)*((double) rand() / RAND_MAX);
        center_bubbles[1][k]  = domain.ymin + 0.03 + (y_length-2.0*0.03)*((double) rand() / RAND_MAX);
        radius_bubbles[k]     = min_bubble_radius + (max_bubble_radius - min_bubble_radius)*((double) rand() / RAND_MAX);
        theta_bubbles[k]      = 2.0*M_PI*((double) rand() / RAND_MAX);
        dt_bubbles[k]         = 2.0*((double) rand() / RAND_MAX);
        if(k >=1)
        {
          bool intersect = false;
          for (size_t kk = 0; kk < k; ++kk)
            intersect = intersect || (sqrt(SQR(center_bubbles[0][k] - center_bubbles[0][kk]) + SQR(center_bubbles[1][k] - center_bubbles[1][kk])) <= sqrt(5.0)*(radius_bubbles[k] + radius_bubbles[kk]) + 0.001);
          while(intersect)
          {
            center_bubbles[0][k]  = domain.xmin + 0.03 + (x_length-2.0*0.03)*((double) rand() / RAND_MAX);
            center_bubbles[1][k]  = domain.ymin + 0.03 + (y_length-2.0*0.03)*((double) rand() / RAND_MAX);
            radius_bubbles[k]     = min_bubble_radius + (max_bubble_radius - min_bubble_radius)*((double) rand() / RAND_MAX);
            theta_bubbles[k]      = 2.0*M_PI*((double) rand() / RAND_MAX);
            dt_bubbles[k]         = 2.0*((double) rand() / RAND_MAX);
            intersect = false;
            for (size_t kk = 0; kk < k; ++kk)
              intersect = intersect || (sqrt(SQR(center_bubbles[0][k] - center_bubbles[0][kk]) + SQR(center_bubbles[1][k] - center_bubbles[1][kk])) <= sqrt(5.0)*(radius_bubbles[k] + radius_bubbles[kk]) + 0.001);
          }
        }
      }
    }
    else
    {
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        center_bubbles[dim].resize(0);
      radius_bubbles.resize(0);
      theta_bubbles.resize(0);
      dt_bubbles.resize(0);
    }
#endif
  }
  double operator()(double x, double y
                  #ifdef P4_TO_P8
                    , double z
                  #endif
                    ) const
  {
    switch(test_nb)
    {
#ifdef P4_TO_P8
    case 0:
      return sqrt(SQR(x - xc) + SQR(y - xc) + SQR(z - zc)) - x_length/4.0;
    case 1:
    {
      double phi, theta;
      if(fabs(sqrt(SQR(x - xc) + SQR(y - yc)) < EPS*MAX(x_length, y_length)))
        phi = M_PI_2;
      else
        phi = acos((x - xc)/sqrt(SQR(x - xc) + SQR(y - yc))) + ((y > yc)? 0.0 : M_PI);
      theta = ((sqrt(SQR(x - xc) + SQR(y- yc) + SQR(z - zc)) > EPS*MAX(x_length, y_length, z_length))? acos((z - zc)/sqrt(SQR(x - xc) + SQR(y- yc) + SQR(z - zc))) : 0.0);
      return SQR(x - xc) + SQR(y - yc) + SQR(z - zc) - SQR(1.25 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0-cos(6.0*theta)));
    }
    case 2:
    {
      double phi, theta;
      if(fabs(sqrt(SQR(x - xc) + SQR(y - yc)) < EPS*MAX(x_length, y_length)))
        phi = M_PI_2;
      else
        phi = acos((x - xc)/sqrt(SQR(x - xc) + SQR(y - yc))) + ((y > yc)? 0.0 : M_PI);
      theta = ((sqrt(SQR(x - xc) + SQR(y- yc) + SQR(z - zc)) > EPS*MAX(x_length, y_length, z_length))? acos((z - zc)/sqrt(SQR(x - xc) + SQR(y- yc) + SQR(z - zc))) : 0.0);
      return SQR(x - xc) + SQR(y - yc) + SQR(z - zc) - SQR(0.75 + 0.2*(1.0 - 0.6*cos(6.0*phi))*(1.0-cos(6.0*theta)));
    }
    case 3:
    {
      double phi = DBL_MAX;
      for (short ii = 0; ii < 2; ++ii)
        for (short jj = 0; jj < 2; ++jj)
        {
          double fake_x = sqrt(SQR(x - (xc + ((double) ii)*x_length)) + SQR(y - (yc + ((double) jj)*y_length)));
          for (short kk = 0; kk < 2; ++kk)
            phi = MIN(phi, bone_shaped_ls(0, zc + ((double) kk)*z_length, fake_x, z));
        }
      return phi;
    }
#else
    case 0:
    case 1:
    case 2:
    case 3:
      return sqrt(SQR(x - xc) + SQR(y - xc)) - x_length/4.0;
    case 4:
    {
      double alpha = 0.02*sqrt(5.0);
      double tt;
      if(fabs(x-xc-alpha) < EPS*x_length)
        tt = (y > yc + alpha + EPS*y_length) ? 0.5*M_PI: (((y < yc + alpha - EPS*y_length))? -0.5*M_PI : 0.0);
      else
        tt = (x > xc + alpha + EPS*x_length) ? atan((y-yc-alpha)/(x-xc-alpha)) : (( y >= yc+alpha)? (M_PI + atan((y-yc-alpha)/(x-xc-alpha))) : (-M_PI + atan((y-yc-alpha)/(x-xc-alpha))));
      return SQR(x-xc-alpha) + SQR(y-yc-alpha) - SQR(0.5 + 0.2*sin(5*tt));
    }
    case 5:
      return bone_shaped_ls(xc, yc, x, y);
    case 6:
    {
      double tt;
      if(fabs(x-xc) < EPS*x_length)
        tt = (y > yc + EPS*y_length) ? 0.5*M_PI: (((y < yc - EPS*y_length))? -0.5*M_PI : 0.0);
      else
        tt = (x > xc + EPS*x_length) ? atan((y-yc)/(x-xc)) : (( y >= yc)? (M_PI + atan((y-yc)/(x-xc))) : (-M_PI + atan((y-yc)/(x-xc))));
      return SQR(x-xc) + SQR(y-yc) - SQR(0.5 + 0.1*sin(5*tt));
    }
    case 7:
    {
      double phi = +DBL_MAX;
      for (size_t k = 0; k < n_bubbles; ++k)
      {
        double u      = (x - center_bubbles[0][k])*cos(theta_bubbles[k]) + (y - center_bubbles[1][k])*sin(theta_bubbles[k]);
        double v      = -(x - center_bubbles[0][k])*sin(theta_bubbles[k]) + (y - center_bubbles[1][k])*cos(theta_bubbles[k]);
        double vel_v  = (1.0 + SQR(v/max_bubble_radius))*radius_bubbles[k];
        double uc_adv = -radius_bubbles[k]*dt_bubbles[k];
        phi = MIN(phi, sqrt(SQR(u - vel_v*dt_bubbles[k] - uc_adv) + SQR(v)) - radius_bubbles[k]);
      }
      return phi;
    }
    case 8:
    {
      double phi = DBL_MAX;
      for (short ii = -1; ii < 2; ++ii)
        phi = MIN(phi, bone_shaped_ls(xc + ((double) ii)*(domain.xmax - domain.xmin), yc, x, y));
      return phi;
    }
    case 9:
    {
      double phi = DBL_MAX;
      for (short ii = -1; ii < 2; ++ii)
        for (short jj = -1; jj < 2; ++jj)
          phi = MIN(phi, bone_shaped_ls(xc + ((double) ii)*(domain.xmax - domain.xmin), yc + ((double) jj)*(domain.ymax - domain.ymin), x, y));
      return phi;
    }
#endif
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
};

double u_exact_m(int test_nb, double x, double y
                 #ifdef P4_TO_P8
                 , double z
                 #endif
                 )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return exp(-x*x-y*y-z*z);
  case 1:
    return 3.0+exp(.5*(x-z))*(x*sin(y) - cos(x+y)*atan(z))/500.0;
  case 2:
    return exp(.5*(x-z))*(x*sin(y) - cos(x+y)*atan(z));
  case 3:
  {
    double pp = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double qq = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    return atan(pp)*log(qq);
  }
#else
  case 0:
    return exp(-x*x-y*y);
  case 1:
    return 1.0;
  case 2:
    return exp(x)*cos(y);
  case 3:
    return x*x - y*y;
  case 4:
    return x*x + y*y;
  case 5:
    return exp(x)*(x*x*sin(y) + y*y);
  case 6:
    return exp(x)*(x*x*sin(y) + y*y)/10000.0;
  case 7:
    return (cos(2.0*M_PI*(x+3.0*y)/0.04) - sin(2.0*M_PI*(y - 2.0*x)/0.04))/1000.0;
  case 8:
    return cos((2.0*M_PI/3.0)*(x-tanh(y))) + exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))));
  case 9:
    return atan(sin((2.0*M_PI/3.0)*(2.0*x-y)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double d_u_exact_m_dx(int test_nb, double x, double y
                      #ifdef P4_TO_P8
                      , double z
                      #endif
                      )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return -2.0*x*exp(-x*x-y*y-z*z);
  case 1:
    return exp(.5*(x-z))*((1.0 + 0.5*x)*sin(y) + (sin(x+y) - 0.5*cos(x+y))*atan(z))/500.0;
  case 2:
    return exp(.5*(x-z))*((1.0 + 0.5*x)*sin(y) + (sin(x+y) - 0.5*cos(x+y))*atan(z));
  case 3:
  {
    double pp     = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double dpp_dx = cos((2.0*M_PI/3.0)*(2.0*x-y))*2.0*(2.0*M_PI/3.0);
    double qq     = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    return (dpp_dx/(1.0 + SQR(pp)))*log(qq);
  }
#else
  case 0:
    return -2.0*x*exp(-x*x-y*y);
  case 1:
    return 0.0;
  case 2:
    return exp(x)*cos(y);
  case 3:
  case 4:
    return 2.*x;
  case 5:
    return exp(x)*((x*x + 2.0*x)*sin(y) + y*y);
  case 6:
    return exp(x)*((x*x + 2.0*x)*sin(y) + y*y)/10000.0;
  case 7:
    return (-(2.0*M_PI/0.04)*sin(2.0*M_PI*(x+3.0*y)/0.04) + (2.0*M_PI*2.0/0.04)*cos(2.0*M_PI*(y - 2.0*x)/0.04))/1000.0;
  case 8:
    return -(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x-tanh(y))) - 2.0*(2.0*M_PI/3.0)*sin(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))));
  case 9:
    return (2.0*(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(2.0*x-y)))/(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x-y))));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_m_dxdx(int test_nb, double x, double y
                         #ifdef P4_TO_P8
                         , double z
                         #endif
                         )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return  (4.*x*x-2.)*exp(-x*x - y*y - z*z);
  case 1:
    return exp(.5*(x-z))*((1.0 + 0.25*x)*sin(y) + (sin(x+y) + 0.75*cos(x+y))*atan(z))/500.0;
  case 2:
    return exp(.5*(x-z))*((1.0 + 0.25*x)*sin(y) + (sin(x+y) + 0.75*cos(x+y))*atan(z));
  case 3:
  {
    double pp         = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double dpp_dx     = cos((2.0*M_PI/3.0)*(2.0*x-y))*2.0*(2.0*M_PI/3.0);
    double ddpp_dxdx  = -sin((2.0*M_PI/3.0)*(2.0*x-y))*SQR(2.0*(2.0*M_PI/3.0));
    double qq         = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    return ((ddpp_dxdx*(1.0 + SQR(pp)) - 2.0*pp*SQR(dpp_dx))/SQR(1.0 + SQR(pp)))*log(qq);
  }
#else
  case 0:
    return  (4.*x*x-2.)*exp(-x*x - y*y);
  case 1:
    return 0.0;
  case 2:
    return exp(x)*cos(y);
  case 3:
  case 4:
    return 2.0;
  case 5:
    return exp(x)*((x*x + 4.0*x + 2.0)*sin(y) + y*y);
  case 6:
    return exp(x)*((x*x + 4.0*x + 2.0)*sin(y) + y*y)/10000.0;
  case 7:
    return (-SQR(2.0*M_PI/0.04)*cos(2.0*M_PI*(x+3.0*y)/0.04) + SQR(2.0*M_PI*2.0/0.04)*sin(2.0*M_PI*(y - 2.0*x)/0.04))/1000.0;
  case 8:
    return -SQR(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(x-tanh(y))) - 2.0*SQR(2.0*2.0*M_PI/3.0)*cos(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y)))) + SQR(2.0*(2.0*M_PI/3.0)*sin(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y)))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))));
  case 9:
    return (-SQR(2.0*2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(2.0*x-y))*(1.0+SQR(sin((2.0*M_PI/3.0)*(2.0*x-y)))) - SQR(2.0*2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*(2.0*x-y))*SQR(cos((2.0*M_PI/3.0)*(2.0*x-y))))/SQR(1.0+SQR(sin((2.0*M_PI/3.0)*(2.0*x-y))));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double d_u_exact_m_dy(int test_nb, double x, double y
                      #ifdef P4_TO_P8
                      , double z
                      #endif
                      )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return -2.0*y*exp(-x*x-y*y-z*z);
  case 1:
    return exp(.5*(x-z))*(x*cos(y) + sin(x+y)*atan(z))/500.0;
  case 2:
    return exp(.5*(x-z))*(x*cos(y) + sin(x+y)*atan(z));
  case 3:
  {
    double pp     = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double dpp_dy = cos((2.0*M_PI/3.0)*(2.0*x-y))*(-2.0*M_PI/3.0);
    double qq     = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    double dqq_dy = -sin((2.0*M_PI/3.0)*(2.0*y-z))*(2.0*(2.0*M_PI/3.0));
    return (dpp_dy/(1.0 + SQR(pp)))*log(qq) + atan(pp)*dqq_dy/qq;
  }
#else
  case 0:
    return -2.0*y*exp(-x*x-y*y);
  case 1:
    return 0.0;
  case 2:
    return -exp(x)*sin(y);
  case 3:
    return -2.*y;
  case 4:
    return +2.*y;
  case 5:
    return exp(x)*(x*x*cos(y) + 2.0*y);
  case 6:
    return exp(x)*(x*x*cos(y) + 2.0*y)/10000.0;
  case 7:
    return (-(2.0*M_PI*3.0/0.04)*sin(2.0*M_PI*(x+3.0*y)/0.04) - (2.0*M_PI/0.04)*cos(2.0*M_PI*(y - 2.0*x)/0.04))/1000.0;
  case 8:
    return (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x-tanh(y)))*(1.0-SQR(tanh(y))) + (2.0*M_PI/3.0)*0.251*sin(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))));
  case 9:
    return (-(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(2.0*x-y)))/(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x-y))));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_m_dydy(int test_nb, double x, double y
                         #ifdef P4_TO_P8
                         , double z
                         #endif
                         )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return  (4.*y*y-2.)*exp(-x*x - y*y - z*z);
  case 1:
    return exp(.5*(x-z))*(-x*sin(y) + cos(x+y)*atan(z))/500.0;
  case 2:
    return exp(.5*(x-z))*(-x*sin(y) + cos(x+y)*atan(z));
  case 3:
  {
    double pp         = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double dpp_dy     = cos((2.0*M_PI/3.0)*(2.0*x-y))*(-2.0*M_PI/3.0);
    double ddpp_dydy  = -sin((2.0*M_PI/3.0)*(2.0*x-y))*SQR(-2.0*M_PI/3.0);
    double qq         = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    double dqq_dy     = -sin((2.0*M_PI/3.0)*(2.0*y-z))*(2.0*(2.0*M_PI/3.0));
    double ddqq_dydy  = -cos((2.0*M_PI/3.0)*(2.0*y-z))*SQR(2.0*(2.0*M_PI/3.0));
    return (ddpp_dydy*(1.0+SQR(pp)) - 2.0*pp*SQR(dpp_dy))/SQR(1.0+SQR(pp))*log(qq) + 2.0*(dpp_dy/(1.0 + SQR(pp)))*(dqq_dy/qq) + atan(pp)*(ddqq_dydy*qq-SQR(dqq_dy))/SQR(qq);
  }
#else
  case 0:
    return  (4.*y*y-2.)*exp(-x*x - y*y);
  case 1:
    return 0.0;
  case 2:
    return -exp(x)*cos(y);
  case 3:
    return -2.0;
  case 4:
    return +2.0;
  case 5:
    return exp(x)*(-x*x*sin(y) + 2.0);
  case 6:
    return exp(x)*(-x*x*sin(y) + 2.0)/10000.0;
  case 7:
    return (-SQR(2.0*M_PI*3.0/0.04)*cos(2.0*M_PI*(x+3.0*y)/0.04) + SQR(2.0*M_PI/0.04)*sin(2.0*M_PI*(y - 2.0*x)/0.04))/1000.0;
  case 8:
    return
        - SQR(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(x-tanh(y)))*SQR(1.0 - SQR(tanh(y)))
        + (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x-tanh(y)))*(-2.0*tanh(y)*(1.0 - SQR(tanh(y))))
        + (2.0*M_PI/3.0)*0.251*cos(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y))*(2.0*(2.0*M_PI/3.0)*(-0.251))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))))
        + SQR((2.0*M_PI/3.0)*0.251)*SQR(sin(2.0*(2.0*M_PI/3.0)*(2.0*x-0.251*y)))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y))));
  case 9:
    return (-SQR(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(2.0*x-y))*(1.0+SQR(sin((2.0*M_PI/3.0)*(2.0*x-y)))) - SQR(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*(2.0*x-y))*SQR(cos((2.0*M_PI/3.0)*(2.0*x-y))))/SQR(1.0+SQR(sin((2.0*M_PI/3.0)*(2.0*x-y))));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

#ifdef P4_TO_P8
double d_u_exact_m_dz(int test_nb, double x, double y, double z)
{
  switch(test_nb)
  {
  case 0:
    return -2.0*z*exp(-x*x-y*y-z*z);
  case 1:
    return -exp(.5*(x-z))*(.5*x*sin(y) + cos(x+y)*(1.0/(1.0 + SQR(z)) - .5*atan(z)))/500.0;
  case 2:
    return -exp(.5*(x-z))*(.5*x*sin(y) + cos(x+y)*(1.0/(1.0 + SQR(z)) - .5*atan(z)));
  case 3:
  {
    double pp         = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double qq         = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    double dqq_dz     = sin((2.0*M_PI/3.0)*(2.0*y-z))*(2.0*M_PI/3.0);
    return atan(pp)*dqq_dz/qq;
  }
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_m_dzdz(int test_nb, double x, double y, double z)
{
  switch(test_nb)
  {
  case 0:
    return  (4.*z*z-2.)*exp(-x*x - y*y - z*z);
  case 1:
    return exp(.5*(x-z))*(.25*x*sin(y) + cos(x+y)*(2.0*z/(SQR(1.0 + SQR(z))) + 1.0/(1.0 + SQR(z)) - .25*atan(z)))/500.0;
  case 2:
    return exp(.5*(x-z))*(.25*x*sin(y) + cos(x+y)*(2.0*z/(SQR(1.0 + SQR(z))) + 1.0/(1.0 + SQR(z)) - .25*atan(z)));
  case 3:
  {
    double pp         = sin((2.0*M_PI/3.0)*(2.0*x-y));
    double qq         = 1.5+cos((2.0*M_PI/3.0)*(2.0*y-z));
    double dqq_dz     = sin((2.0*M_PI/3.0)*(2.0*y-z))*(2.0*M_PI/3.0);
    double ddqq_dzdz  = -cos((2.0*M_PI/3.0)*(2.0*y-z))*SQR(2.0*M_PI/3.0);
    return atan(pp)*(ddqq_dzdz*qq - SQR(dqq_dz))/SQR(qq);
  }
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}
#endif

double laplacian_u_exact_m(int test_nb, double x, double y
                           #ifdef P4_TO_P8
                           , double z
                           #endif
                           )
{
#ifdef P4_TO_P8
  return (dd_u_exact_m_dxdx(test_nb, x, y, z) + dd_u_exact_m_dydy(test_nb, x, y, z) + dd_u_exact_m_dzdz(test_nb, x, y, z));
#else
  return (dd_u_exact_m_dxdx(test_nb, x, y) + dd_u_exact_m_dydy(test_nb, x, y));
#endif
}

double u_exact_p(int test_nb, double x, double y
                 #ifdef P4_TO_P8
                 , double z
                 #endif
                 )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    return -1.0 + atan(ff)/500.0;
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    return tanh(pp)*acos(qq);
  }
#else
  case 0:
  case 2:
  case 3:
    return 0.0;
  case 1:
    return 1.0 + log(2.0*sqrt(EPS + SQR(x) + SQR(y)));
  case 4:
    return 0.1*SQR(x*x + y*y) - 0.01*log(2.0*sqrt(EPS + x*x + y*y));
  case 5:
    return -x*x -y*y;
  case 6:
    return (0.5 + (cos(x)*(pow(y, 4.0) + sin(y*y - x*x))));
  case 7:
    return cos(x+y)*exp(-SQR(x*cos(y)));
  case 8:
    return tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y);
  case 9:
    return log(1.5+cos((2.0*M_PI/3.0)*(-x+3.0*y)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double d_u_exact_p_dx(int test_nb, double x, double y
                      #ifdef P4_TO_P8
                      , double z
                      #endif
                      )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(-sin(y)+2.0*z*sin(2.0*x));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dx = 3.0*0.1*x*x*y -y*cos(x+z);
    return (1.0/500.0)*(1.0/(1.0 + SQR(ff)))*dff_dx;
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double dpp_dx     = -sin((2.0*M_PI/3.0)*(2.0*x+y))*(2.0*2.0*M_PI/3.0);
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    double dqq_dx     = 0.5*cos((2.0*M_PI/3.0)*(2.0*z-x))*(-2.0*M_PI/3.0);
    return (1.0-SQR(tanh(pp)))*dpp_dx*acos(qq) + tanh(pp)*(-1.0/sqrt(1.0-SQR(qq)))*dqq_dx;
  }
#else
  case 0:
  case 2:
  case 3:
    return 0.0;
  case 1:
    return x/(EPS + SQR(x) + SQR(y));
  case 4:
    return 0.1*2.0*(x*x + y*y)*2.0*x - 0.01*x/(EPS + x*x + y*y);
  case 5:
    return -2.0*x;
  case 6:
    return (-sin(x)*(pow(y, 4.0) + sin(y*y - x*x)) - 2.0*x*cos(x)*cos(y*y-x*x));
  case 7:
    return -exp(-SQR(x*cos(y)))*(sin(x+y) + 2.0*x*cos(x+y)*SQR(cos(y)));
  case 8:
    return (1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*(-(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*2.0*x));
  case 9:
    return (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(3.0*y-x))/(1.5+cos((2.0*M_PI/3.0)*(3.0*y-x)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_p_dxdx(int test_nb, double x, double y
                         #ifdef P4_TO_P8
                         , double z
                         #endif
                         )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(SQR(-sin(y)+2.0*z*sin(2.0*x)) + 4.0*z*cos(2.0*x));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dx = 3.0*0.1*x*x*y -y*cos(x+z);
    double ddff_dxdx = 2.0*3.0*0.1*x*y + y*sin(x+z);
    return (1.0/500.0)*(ddff_dxdx*(1.0 + SQR(ff)) - 2.0*SQR(dff_dx)*ff)/SQR(1.0 + SQR(ff));
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double dpp_dx     = -sin((2.0*M_PI/3.0)*(2.0*x+y))*(2.0*2.0*M_PI/3.0);
    double ddpp_dxdx  = -cos((2.0*M_PI/3.0)*(2.0*x+y))*SQR(2.0*2.0*M_PI/3.0);
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    double dqq_dx     = 0.5*cos((2.0*M_PI/3.0)*(2.0*z-x))*(-2.0*M_PI/3.0);
    double ddqq_dxdx   = -0.5*sin((2.0*M_PI/3.0)*(2.0*z-x))*SQR(-2.0*M_PI/3.0);
    return (-2.0*tanh(pp)*(1.0 - SQR(tanh(pp)))*SQR(dpp_dx) + (1.0 - SQR(tanh(pp)))*ddpp_dxdx)*acos(qq) + 2.0*(1.0-SQR(tanh(pp)))*dpp_dx*(-1.0/sqrt(1.0-SQR(qq)))*dqq_dx - tanh(pp)*((ddqq_dxdx*sqrt(1.0-SQR(qq)) + SQR(dqq_dx)*qq/sqrt(1.0 - SQR(qq)))/(1.0 - SQR(qq)));
  }
#else
  case 0:
  case 2:
  case 3:
    return 0.0;
  case 1:
    return (y*y - x*x)/SQR(EPS + SQR(x) + SQR(y));
  case 4:
    return 0.1*2.*2.*(3.*x*x + y*y) - 0.01*(y*y - x*x)/(SQR(EPS + x*x + y*y));
  case 5:
    return -2.0;
  case 6:
    return (-cos(x)*(pow(y, 4.0) + sin(y*y - x*x)) + (4.0*x*sin(x) - 2.0*cos(x))*cos(y*y-x*x) - 4.0*x*x*cos(x)*sin(y*y - x*x));
  case 7:
    return exp(-SQR(x*cos(y)))*(4.0*x*x*cos(x+y)*SQR(SQR(cos(y))) + 4.0*x*SQR(cos(y))*sin(x+y) - cos(x+y) - 2.0*cos(x+y)*SQR(cos(y)));
  case 8:
    return
        - 2.0*tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*SQR(-(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*2.0*x))
        + (1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*(-SQR((2.0*M_PI/3.0)*2.0)*cos((2.0*M_PI/3.0)*2.0*x));
  case 9:
    return -SQR(2.0*M_PI/3.0)*(cos((2.0*M_PI/3.0)*(3.0*y-x))*(1.5 + cos((2.0*M_PI/3.0)*(3.0*y-x))) + SQR(sin((2.0*M_PI/3.0)*(3.0*y-x))))/SQR(1.5 + cos((2.0*M_PI/3.0)*(3.0*y-x)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double d_u_exact_p_dy(int test_nb, double x, double y
                      #ifdef P4_TO_P8
                      , double z
                      #endif
                      )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(-x*cos(y)-cos(z));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dy = 0.1*x*x*x - 2.0*z*sin(y) - sin(x+z);
    return (1.0/500.0)*(1.0/(1.0 + SQR(ff)))*dff_dy;
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double dpp_dy     = -sin((2.0*M_PI/3.0)*(2.0*x+y))*(2.0*M_PI/3.0);
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    return (1.0-SQR(tanh(pp)))*dpp_dy*acos(qq);
  };
#else
  case 0:
  case 2:
  case 3:
    return 0.0;
  case 1:
    return y/(EPS + SQR(x) + SQR(y));
  case 4:
    return 0.1*2.*(x*x + y*y)*2.*y - 0.01*y/(EPS + x*x + y*y);
  case 5:
    return -2.0*y;
  case 6:
    return (cos(x)*(4.0*pow(y, 3.0) + 2.0*y*cos(y*y - x*x)));
  case 7:
    return exp(-SQR(x*cos(y)))*(2.0*x*x*sin(y)*cos(y)*cos(x+y) - sin(x+y));
  case 8:
    return -0.24*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)));
  case 9:
    return -(3.0*(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(3.0*y-x)))/(1.5 + cos((2.0*M_PI/3.0)*(3.0*y-x)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_p_dydy(int test_nb, double x, double y
                         #ifdef P4_TO_P8
                         , double z
                         #endif
                         )
{
  switch(test_nb)
  {
#ifdef P4_TO_P8
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(SQR(-x*cos(y)-cos(z)) + x*sin(y));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dy = 0.1*x*x*x - 2.0*z*sin(y) - sin(x+z);
    double ddff_dydy = -2.0*z*cos(y);
    return (1.0/500.0)*(ddff_dydy*(1.0 + SQR(ff)) - 2.0*SQR(dff_dy)*ff)/SQR(1.0 + SQR(ff));
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double dpp_dy     = -sin((2.0*M_PI/3.0)*(2.0*x+y))*(2.0*M_PI/3.0);
    double ddpp_dydy  = -cos((2.0*M_PI/3.0)*(2.0*x+y))*SQR(2.0*M_PI/3.0);
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    return (-2.0*tanh(pp)*(1.0 - SQR(tanh(pp)))*SQR(dpp_dy) + (1.0 - SQR(tanh(pp)))*ddpp_dydy)*acos(qq);
  }
#else
  case 0:
  case 2:
  case 3:
    return 0.0;
  case 1:
    return (x*x - y*y)/SQR(EPS + SQR(x) + SQR(y));
  case 4:
    return 0.1*2.*2.*(x*x + 3.*y*y)- 0.01*(x*x - y*y)/(SQR(EPS + x*x + y*y));
  case 5:
    return -2.0;
  case 6:
    return (cos(x)*(12.0*SQR(y) + 2.0*cos(y*y - x*x) - 4.0*y*y*sin(y*y-x*x)));
  case 7:
    return exp(-SQR(x*cos(y)))*(4.0*SQR(x*x*cos(y)*sin(y))*cos(x+y) - 2.0*x*x*sin(2.0*y)*sin(x+y) +2.0*x*x*cos(2.0*y)*cos(x+y) -cos(x+y));
  case 8:
    return -2.0*SQR(0.24)*tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)));
  case 9:
    return -(SQR(3.0*(2.0*M_PI/3.0))*(cos((2.0*M_PI/3.0)*(3.0*y-x))*(1.5 + cos((2.0*M_PI/3.0)*(3.0*y-x))) + SQR(sin((2.0*M_PI/3.0)*(3.0*y-x)))))/SQR(1.5 + cos((2.0*M_PI/3.0)*(3.0*y-x)));
#endif
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

#ifdef P4_TO_P8
double d_u_exact_p_dz(int test_nb, double x, double y, double z)
{
  switch(test_nb)
  {
  case 0:
    return 0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(+y*sin(z)-cos(2.0*x));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dz = 2.0*cos(y) - y*cos(x+z);
    return (1.0/500.0)*(1.0/(1.0 + SQR(ff)))*dff_dz;
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    double dqq_dz     = 0.5*cos((2.0*M_PI/3.0)*(2.0*z-x))*(2.0*2.0*M_PI/3.0);
    return tanh(pp)*(-1.0/sqrt(1.0-SQR(qq)))*dqq_dz;
  }
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double dd_u_exact_p_dzdz(int test_nb, double x, double y, double z)
{
  switch(test_nb)
  {
  case 0:
    return  0.0;
  case 1:
    return exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x))*(SQR(+y*sin(z)-cos(2.0*x)) +y*cos(z));
  case 2:
  {
    double ff = 0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z);
    double dff_dz = 2.0*cos(y) - y*cos(x+z);
    double ddff_dzdz = +y*sin(x+z);
    return (1.0/500.0)*(ddff_dzdz*(1.0 + SQR(ff)) - 2.0*SQR(dff_dz)*ff)/SQR(1.0 + SQR(ff));
  }
  case 3:
  {
    double pp         = cos((2.0*M_PI/3.0)*(2.0*x+y));
    double qq         = 0.5*sin((2.0*M_PI/3.0)*(2.0*z-x));
    double dqq_dz     = 0.5*cos((2.0*M_PI/3.0)*(2.0*z-x))*(2.0*2.0*M_PI/3.0);
    double ddqq_dzdz   = -0.5*sin((2.0*M_PI/3.0)*(2.0*z-x))*SQR(2.0*2.0*M_PI/3.0);
    return -tanh(pp)*((ddqq_dzdz*sqrt(1.0-SQR(qq)) + SQR(dqq_dz)*qq/sqrt(1.0 - SQR(qq)))/(1.0 - SQR(qq)));
  }
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}
#endif

double laplacian_u_exact_p(int test_nb, double x, double y
                           #ifdef P4_TO_P8
                           , double z
                           #endif
                           )
{
#ifdef P4_TO_P8
  return (dd_u_exact_p_dxdx(test_nb, x, y, z) + dd_u_exact_p_dydy(test_nb, x, y, z) + dd_u_exact_p_dzdz(test_nb, x, y, z));
#else
  return (dd_u_exact_p_dxdx(test_nb, x, y) + dd_u_exact_p_dydy(test_nb, x, y));
#endif
}

class BCWALLTYPE :
    #ifdef P4_TO_P8
    public WallBC3D
    #else
    public WallBC2D
    #endif
{
  BoundaryConditionType bc_walltype;
public:
  BCWALLTYPE(BoundaryConditionType bc_walltype_): bc_walltype(bc_walltype_) {}
  BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                   , double
                                 #endif
                                   ) const
  {
    return bc_walltype;
  }
};

class BCWALLVAL :
    #ifdef P4_TO_P8
    public CF_3
    #else
    public CF_2
    #endif
{
  box domain;
  LEVEL_SET ls;
  BoundaryConditionType bc_wall_type;
  int test_nb;
public:
  BCWALLVAL(box domain_, LEVEL_SET ls_, BoundaryConditionType bc_wall_type_, int test_nb_) : domain(domain_), ls(ls_), bc_wall_type(bc_wall_type_), test_nb(test_nb_){}
  double operator()(double x, double y
                  #ifdef P4_TO_P8
                    , double z
                  #endif
                    ) const
  {
#ifdef P4_TO_P8
    if(bc_wall_type==DIRICHLET)
      return (ls(x,y,z) > 0)? u_exact_p(test_nb, x,y,z):u_exact_m(test_nb, x, y, z);
#else
    if(bc_wall_type==DIRICHLET)
      return (ls(x,y) > 0)? u_exact_p(test_nb, x,y):u_exact_m(test_nb, x, y);
#endif
    else
    {
      double dx = 0; dx = ((fabs(x-domain.xmin)<(domain.xmax - domain.xmin)*EPS) ? -1 :((fabs(x-domain.xmax)<(domain.xmax - domain.xmin)*EPS) ? 1 : 0));
      double dy = 0; dy = ((fabs(y-domain.ymin)<(domain.ymax - domain.ymin)*EPS) ? -1 :((fabs(y-domain.ymax)<(domain.ymax - domain.ymin)*EPS) ? 1 : 0));
#ifdef P4_TO_P8
      double dz = 0; dz = ((fabs(z-domain.zmin)<(domain.zmax - domain.zmin)*EPS) ? -1 :((fabs(z-domain.zmax)<(domain.zmax - domain.zmin)*EPS) ? 1 : 0));
      return (ls(x, y, z) > 0)? (dx*d_u_exact_p_dx(test_nb, x, y, z) + dy*d_u_exact_p_dy(test_nb, x, y, z) + dz*d_u_exact_p_dz(test_nb, x, y, z)):(dx*d_u_exact_m_dx(test_nb, x, y, z) + dy*d_u_exact_m_dy(test_nb, x, y, z) + dz*d_u_exact_m_dz(test_nb, x, y, z));
#else
      return (ls(x, y) > 0)? (dx*d_u_exact_p_dx(test_nb, x, y) + dy*d_u_exact_p_dy(test_nb, x, y)):(dx*d_u_exact_m_dx(test_nb, x, y) + dy*d_u_exact_m_dy(test_nb, x, y));
#endif
    }
  }
};


class JUMP_U:
    #ifdef P4_TO_P8
    public CF_3
    #else
    public CF_2
    #endif
{
  int test_nb;
public:
  JUMP_U(int test_nb_): test_nb(test_nb_){}
  double operator()(double x, double y
                  #ifdef P4_TO_P8
                    , double z
                  #endif
                    ) const
  {
#ifdef P4_TO_P8
    return u_exact_p(test_nb, x, y, z) - u_exact_m(test_nb, x, y, z);
#else
    return u_exact_p(test_nb, x, y) - u_exact_m(test_nb, x, y);
#endif
  }
};


p4est_bool_t
refine_levelset_cf_finest_in_negative (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

#ifdef P4_TO_P8
    double d = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double d = sqrt(dx*dx + dy*dy);
#endif

    double x = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

#ifdef P4_TO_P8
    CF_3&  phi = *(data->phi);
#else
    CF_2&  phi = *(data->phi);
#endif
    double lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
#ifdef P4_TO_P8
          f[4*ck+2*cj+ci] = phi(x+ci*dx, y+cj*dy, z+ck*dz);
          if (f[4*ck+2*cj+ci] <= 0.5*lip*d)
            return P4EST_TRUE;
#else
          f[2*cj+ci] = phi(x+ci*dx, y+cj*dy);
          if (f[2*cj+ci] <= 0.5*lip*d)
            return P4EST_TRUE;
#endif
        }

#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_TRUE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_TRUE;
#endif

    return P4EST_FALSE;
  }
}


void save_VTK(const string out_dir, int test_number,
              p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,
              p4est_t *p4est_fine, p4est_ghost_t *ghost_fine, p4est_nodes_t *nodes_fine,
              my_p4est_brick_t *brick,
              Vec phi, Vec normals[], Vec jump_u, Vec jump_normal_flux, Vec extended_field_fine_nodes_xgfm, Vec jump_mu_grad_u[2][P4EST_DIM], Vec correction_jump_mu_grad[P4EST_DIM],
              Vec sol_cells[2], Vec err_cells[2], Vec extension_xgfm
#ifndef P4_TO_P8
, Vec exact_msol_at_nodes, Vec exact_psol_at_nodes, Vec phi_coarse
#endif
)
{
  PetscErrorCode ierr;
  std::ostringstream oss;

  splitting_criteria_t* data = (splitting_criteria_t*) p4est->user_pointer;
  splitting_criteria_t* data_fine = (splitting_criteria_t*) p4est_fine->user_pointer;

#ifdef P4_TO_P8
  oss << out_dir << "/P8EST";
#else
  oss << out_dir << "/P4EST";
#endif

  oss << "/test_case_" << test_number << "/" << p4est->mpisize << "_procs/" <<
         brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "_macromesh/lvl_diff_" << data->max_lvl-data->min_lvl;

  ostringstream command;
  command << "mkdir -p " << oss.str().c_str();
  system(command.str().c_str());
  ostringstream oss_coarse;

  oss_coarse << oss.str() << "/computational_" << data->min_lvl;

  double *phi_p, *sol_cells_p[2], *err_cells_p[2], *extension_xgfm_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for (short flag = 0; flag < 2; ++flag) {
    ierr = VecGetArray(sol_cells[flag], &sol_cells_p[flag]); CHKERRXX(ierr);
    ierr = VecGetArray(err_cells[flag], &err_cells_p[flag]); CHKERRXX(ierr);
  }

  ierr = VecGetArray(extension_xgfm, &extension_xgfm_p); CHKERRXX(ierr);

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

#ifndef P4_TO_P8
  double *exact_msol_at_nodes_p, *exact_psol_at_nodes_p, *phi_coarse_p;
  ierr = VecGetArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
#endif

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                       #ifdef P4_TO_P8
                         0
                       #else
                         3
                       #endif
                         , 6, oss_coarse.str().c_str(),
                       #ifndef P4_TO_P8
                         VTK_POINT_DATA, "exact_sol_m", exact_msol_at_nodes_p,
                         VTK_POINT_DATA, "exact_sol_p", exact_psol_at_nodes_p,
                         VTK_POINT_DATA, "phi", phi_coarse_p,
                       #endif
                         VTK_CELL_DATA, "sol_gfm", sol_cells_p[0],
      VTK_CELL_DATA, "sol_xgfm", sol_cells_p[1],
      VTK_CELL_DATA, "err_gfm", err_cells_p[0],
      VTK_CELL_DATA, "err_xgfm", err_cells_p[1],
      VTK_CELL_DATA , "leaf_level", l_p,
      VTK_CELL_DATA, "extension_xgfm", extension_xgfm_p);

#ifndef P4_TO_P8
  ierr = VecRestoreArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
#endif



  std::ostringstream oss_fine;
  oss_fine << oss.str() << "/interface_capturing_" << data_fine->min_lvl;

  double *jump_u_p, *jump_normal_flux_p, *extended_field_fine_nodes_xgfm_p;
  double *normals_p[P4EST_DIM], *jump_mu_grad_u_p[2][P4EST_DIM], *correction_jump_mu_grad_p[P4EST_DIM];
  ierr = VecGetArray(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);

  ierr = VecGetArray(extended_field_fine_nodes_xgfm, &extended_field_fine_nodes_xgfm_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(normals[dim], &normals_p[dim]); CHKERRXX(ierr);
    for (short flag = 0; flag < 2; ++flag) {
      ierr = VecGetArray(jump_mu_grad_u[flag][dim], &jump_mu_grad_u_p[flag][dim]); CHKERRXX(ierr);
    }
    ierr = VecGetArray(correction_jump_mu_grad[dim], &correction_jump_mu_grad_p[dim]); CHKERRXX(ierr);
  }

  my_p4est_vtk_write_all(p4est_fine, nodes_fine, ghost_fine,
                         P4EST_TRUE, P4EST_TRUE,
                         4+4*P4EST_DIM, 0, oss_fine.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "jump", jump_u_p,
                         VTK_POINT_DATA, "jump_flux", jump_normal_flux_p,
                         VTK_POINT_DATA, "nx", normals_p[0],
      VTK_POINT_DATA, "ny", normals_p[1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "nz", normals_p[2],
    #endif
      VTK_POINT_DATA, "gfm_jump_mu_dxu", jump_mu_grad_u_p[0][0],
      VTK_POINT_DATA, "gfm_jump_mu_dyu", jump_mu_grad_u_p[0][1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "gfm_jump_mu_dzu", jump_mu_grad_u_p[0][2],
    #endif
      VTK_POINT_DATA, "xgfm_jump_mu_dxu", jump_mu_grad_u_p[1][0],
      VTK_POINT_DATA, "xgfm_jump_mu_dyu", jump_mu_grad_u_p[1][1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "xgfm_jump_mu_dzu", jump_mu_grad_u_p[1][2],
    #endif
      VTK_POINT_DATA, "corr_jump_mu_dxu", correction_jump_mu_grad_p[0],
      VTK_POINT_DATA, "corr_jump_mu_dyu", correction_jump_mu_grad_p[1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "corr_jump_mu_dzu", correction_jump_mu_grad_p[2],
    #endif
      VTK_POINT_DATA, "extension_xgfm", extended_field_fine_nodes_xgfm_p);


  for (short dim = 0; dim < P4EST_DIM ; ++dim) {
    ierr = VecRestoreArray(normals[dim], &normals_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(correction_jump_mu_grad[dim], &correction_jump_mu_grad_p[dim]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(extended_field_fine_nodes_xgfm, &extended_field_fine_nodes_xgfm_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_u, &jump_u_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  for (short flag = 0; flag < 2; ++flag) {
    ierr = VecRestoreArray(sol_cells[flag], &sol_cells_p[flag]); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_cells[flag], &err_cells_p[flag]); CHKERRXX(ierr);
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArray(jump_mu_grad_u[flag][dim], &jump_mu_grad_u_p[flag][dim]); CHKERRXX(ierr);
    }
  }

  ierr = VecRestoreArray(extension_xgfm, &extension_xgfm_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



void get_normals_and_flattened_jumps(p4est_t* p4est_fine, p4est_nodes_t* nodes_fine, my_p4est_node_neighbors_t& ngbd_n_fine, Vec phi, bool use_second_order_theta, int test_number, double mu_p, double mu_m, //input
                                     Vec& jump_u, Vec& jump_normal_flux, Vec normals[P4EST_DIM], Vec phi_xx, Vec phi_yy // output
                                     #ifdef P4_TO_P8
                                     , Vec phi_zz // output
                                     #endif
                                     )
{
  PetscErrorCode ierr;
  my_p4est_level_set_t ls(&ngbd_n_fine);
  JUMP_U jump_u_cf(test_number);
  sample_cf_on_nodes(p4est_fine, nodes_fine, jump_u_cf, jump_u);

  if(use_second_order_theta)
#ifdef P4_TO_P8
    ngbd_n_fine.second_derivatives_central(phi, phi_xx, phi_yy, phi_zz);
#else
    ngbd_n_fine.second_derivatives_central(phi, phi_xx, phi_yy);
#endif

  my_p4est_interpolation_nodes_t interp_phi(&ngbd_n_fine);
  interp_phi.set_input(phi, linear);
  compute_normals(ngbd_n_fine, phi, normals);

  double *jump_normal_flux_p, *normals_p[P4EST_DIM];
  ierr = VecGetArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(normals[dim], &normals_p[dim]); CHKERRXX(ierr);}

  double node_xyz[P4EST_DIM];
  for (p4est_locidx_t nn = 0; nn < nodes_fine->num_owned_indeps; ++nn) {
    node_xyz_fr_n(nn, p4est_fine, nodes_fine, node_xyz);
#ifdef P4_TO_P8
    jump_normal_flux_p[nn] =
        normals_p[0][nn]*(mu_p*d_u_exact_p_dx(test_number, node_xyz[0], node_xyz[1], node_xyz[2]) - mu_m*d_u_exact_m_dx(test_number, node_xyz[0], node_xyz[1], node_xyz[2]))
        + normals_p[1][nn]*(mu_p*d_u_exact_p_dy(test_number, node_xyz[0], node_xyz[1], node_xyz[2]) - mu_m*d_u_exact_m_dy(test_number, node_xyz[0], node_xyz[1], node_xyz[2]))
        + normals_p[2][nn]*(mu_p*d_u_exact_p_dz(test_number, node_xyz[0], node_xyz[1], node_xyz[2]) - mu_m*d_u_exact_m_dz(test_number, node_xyz[0], node_xyz[1], node_xyz[2]));
#else
    jump_normal_flux_p[nn] =
        normals_p[0][nn]*(mu_p*d_u_exact_p_dx(test_number, node_xyz[0], node_xyz[1]) - mu_m*d_u_exact_m_dx(test_number, node_xyz[0], node_xyz[1]))
        + normals_p[1][nn]*(mu_p*d_u_exact_p_dy(test_number, node_xyz[0], node_xyz[1]) - mu_m*d_u_exact_m_dy(test_number, node_xyz[0], node_xyz[1]));
#endif
  }
  ierr = VecGhostUpdateBegin(jump_normal_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(jump_normal_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(normals[dim], &normals_p[dim]); CHKERRXX(ierr);}

  ierr = VecRestoreArray(jump_normal_flux, &jump_normal_flux_p); CHKERRXX(ierr);

  Vec jump_u_flattened, jump_normal_flux_flattened;
  ierr = VecDuplicate(jump_u, &jump_u_flattened); CHKERRXX(ierr);
  ierr = VecDuplicate(jump_normal_flux, &jump_normal_flux_flattened); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi, jump_u, jump_u_flattened);
  ls.extend_from_interface_to_whole_domain_TVD(phi, jump_normal_flux, jump_normal_flux_flattened);
  ierr = VecDestroy(jump_u); CHKERRXX(ierr); jump_u = jump_u_flattened; jump_u_flattened = NULL;
  ierr = VecDestroy(jump_normal_flux); CHKERRXX(ierr); jump_normal_flux = jump_normal_flux_flattened; jump_normal_flux_flattened= NULL;
}

void get_sharp_rhs(const p4est_t* p4est, p4est_ghost_t* ghost, my_p4est_node_neighbors_t& ngbd_n_fine, Vec phi, int test_number, double mu_p, double mu_m, // inputs
                   Vec rhs) // output
{
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_phi(&ngbd_n_fine);
  interp_phi.set_input(phi, linear);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx=0; quad_idx<tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;
      double x = quad_x_fr_q(q_idx, tree_idx, p4est, ghost);
      double y = quad_y_fr_q(q_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(q_idx, tree_idx, p4est, ghost);
      rhs_p[q_idx] = (interp_phi(x, y, z) > 0)? (-mu_p*laplacian_u_exact_p(test_number, x, y, z)):  (-mu_m*laplacian_u_exact_m(test_number, x, y, z));
#else
      rhs_p[q_idx] = (interp_phi(x, y) > 0)?    (-mu_p*laplacian_u_exact_p(test_number, x, y)):     (-mu_m*laplacian_u_exact_m(test_number, x, y));
#endif
    }
  }
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

}

void measure_errors(p4est_t* p4est, p4est_ghost_t* ghost, my_p4est_node_neighbors_t& ngbd_n_fine, my_p4est_faces_t* faces, Vec phi, int test_number, double mu_p, double mu_m, Vec sol, Vec flux_components[P4EST_DIM],
                    Vec err_cells, double &err_n, double err_flux_components[P4EST_DIM], double err_derivatives_components[P4EST_DIM])
{
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_phi(&ngbd_n_fine);
  interp_phi.set_input(phi, linear);
  const double *sol_read_p, *flux_components_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(sol, &sol_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(flux_components[dim], &flux_components_read_p[dim]); CHKERRXX(ierr);}

  double *err_p;
  ierr = VecGetArray(err_cells, &err_p); CHKERRXX(ierr);

  err_n = 0.0;
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t quad_idx=0; quad_idx<tree->quadrants.elem_count; ++quad_idx)
    {
      p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;
      double quad_x = quad_x_fr_q(q_idx, tree_idx, p4est, ghost);
      double quad_y = quad_y_fr_q(q_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
      double quad_z = quad_z_fr_q(q_idx, tree_idx, p4est, ghost);
      err_p[q_idx] = (interp_phi(quad_x, quad_y, quad_z) > 0)?  fabs(sol_read_p[q_idx] - u_exact_p(test_number, quad_x, quad_y, quad_z)) : fabs(sol_read_p[q_idx] - u_exact_m(test_number, quad_x, quad_y, quad_z));
#else
      err_p[q_idx] = (interp_phi(quad_x, quad_y) > 0)?          fabs(sol_read_p[q_idx] - u_exact_p(test_number, quad_x, quad_y)) :         fabs(sol_read_p[q_idx] - u_exact_m(test_number, quad_x, quad_y));
#endif
      err_n = MAX(err_n, err_p[q_idx]);
    }
  }

  double xyz_face[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    err_flux_components[dim] = 0.0;
    err_derivatives_components[dim] = 0.0;
    for (p4est_locidx_t face_idx = 0; face_idx < faces->num_local[dim]; ++face_idx) {
      faces->xyz_fr_f(face_idx, dim, xyz_face);
      switch (dim) {
      case 0:
#ifdef P4_TO_P8
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_read_p[dim][face_idx] - ((interp_phi(xyz_face) > 0.0)? (mu_p*d_u_exact_p_dx(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (mu_m*d_u_exact_m_dx(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_read_p[dim][face_idx]/(((interp_phi(xyz_face) > 0.0)? mu_p : mu_m)) - ((interp_phi(xyz_face) > 0.0)? (d_u_exact_p_dx(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (d_u_exact_m_dx(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
#else
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_read_p[dim][face_idx] - ((interp_phi(xyz_face) > 0.0)? (mu_p*d_u_exact_p_dx(test_number, xyz_face[0], xyz_face[1])): (mu_m*d_u_exact_m_dx(test_number, xyz_face[0], xyz_face[1])))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_read_p[dim][face_idx]/(((interp_phi(xyz_face) > 0.0)? mu_p : mu_m)) - ((interp_phi(xyz_face) > 0.0)? (d_u_exact_p_dx(test_number, xyz_face[0], xyz_face[1])): (d_u_exact_m_dx(test_number, xyz_face[0], xyz_face[1])))));
#endif
        break;
      case 1:
#ifdef P4_TO_P8
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_read_p[dim][face_idx] - ((interp_phi(xyz_face) > 0.0)? (mu_p*d_u_exact_p_dy(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (mu_m*d_u_exact_m_dy(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_read_p[dim][face_idx]/(((interp_phi(xyz_face) > 0.0)? mu_p : mu_m)) - ((interp_phi(xyz_face) > 0.0)? (d_u_exact_p_dy(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (d_u_exact_m_dy(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
#else
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_read_p[dim][face_idx] - ((interp_phi(xyz_face) > 0.0)? (mu_p*d_u_exact_p_dy(test_number, xyz_face[0], xyz_face[1])): (mu_m*d_u_exact_m_dy(test_number, xyz_face[0], xyz_face[1])))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_read_p[dim][face_idx]/(((interp_phi(xyz_face) > 0.0)? mu_p : mu_m)) - ((interp_phi(xyz_face) > 0.0)? (d_u_exact_p_dy(test_number, xyz_face[0], xyz_face[1])): (d_u_exact_m_dy(test_number, xyz_face[0], xyz_face[1])))));
#endif
        break;
#ifdef P4_TO_P8
      case 2:
        err_flux_components[dim]        = MAX(err_flux_components[dim], fabs(flux_components_read_p[dim][face_idx] - ((interp_phi(xyz_face) > 0.0)? (mu_p*d_u_exact_p_dz(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (mu_m*d_u_exact_m_dz(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
        err_derivatives_components[dim] = MAX(err_derivatives_components[dim], fabs(flux_components_read_p[dim][face_idx]/(((interp_phi(xyz_face) > 0.0)? mu_p : mu_m)) - ((interp_phi(xyz_face) > 0.0)? (d_u_exact_p_dz(test_number, xyz_face[0], xyz_face[1], xyz_face[2])): (d_u_exact_m_dz(test_number, xyz_face[0], xyz_face[1], xyz_face[2])))));
        break;
#endif
      }
    }
  }

  ierr = VecRestoreArrayRead(sol, &sol_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(flux_components[dim], &flux_components_read_p[dim]); CHKERRXX(ierr);}
  ierr = VecRestoreArray(err_cells, &err_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_flux_components[0], P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_derivatives_components[0], P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("ngrids", "number of computational grids (increasing refinement levels)");
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall (0: Dirichlet wall, 1: Neumann Wall)");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("out_dir", "exportation directory for vtk files (required if save_vtk)");
  cmd.add_option("second_order_ls", "active second order interface localization");
  cmd.add_option("summary", "folder for summary file for convergence results");
  cmd.add_option("ntree", "number of trees in the macromesh along the smallest dimension of the computational domain");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
0: \n\
* domain = [0.0, 1.0] X [0.0, 1.0] X [0.0, 1.0] \n\
* interface = sphere of radius 1/4, centered in (0.5, 0.5, 0.5), negative inside, positive outside \n\
* mu_m = 2.0; \n\
* mu_p = 1.0; \n\
* u_m  = exp(-x*x-y*y-z*z); \n\
* u_p  = 0.0; \n\
* no periodicity \n\
Example 4 from Liu, Fedkiw, Kang 2000 \n\
1: \n\
* domain = [-2.0, 2.0] X [-2.0, 2.0] X [-2.0, 2.0] \n\
* interface = parameterized by (theta in [0.0, 2*pi[, phi in [0.0, pi[) \n\
r(theta, phi) = 1.25 + 0.2*(1.0 - 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0 - cos(6.0*theta)), spherical coordinates \n\
negative inside, positive outside \n\
* mu_m = 2000.0; \n\
* mu_p = 1.0; \n\
* u_m  = 3.0 + exp(.5*(x-z))*(x*sin(y) - cos(x+y)*atan(z))/500.0; \n\
* u_p  = exp(-x*sin(y)-y*cos(z)-z*cos(2.0*x)); \n\
* no periodicity \n\
Example by Raphael Egan for mildly convoluted 3D interface with large ratio of coefficients \n\
2: \n\
* domain = [-2.0, 2.0] X [-2.0, 2.0] X [-2.0, 2.0] \n\
* interface = parameterized by \n\
r(theta, phi) = 0.75 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0-cos(6.0*theta)) \n\
negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 1250.0; \n\
* u_m  = exp(.5*(x-z))*(x*sin(y) - cos(x+y)*atan(z)); \n\
* u_p  = -1.0 + atan(0.1*x*x*x*y + 2.0*z*cos(y)- y*sin(x+z))/500.0; \n\
* no periodicity \n\
Example by Raphael Egan for very convoluted 3D interface (AMR required) with large ratio of coefficients \n\
3: \n\
* domain = [-1.5, 1.5] X [-1.5, 1.5] X [-1.5, 1.5] \n\
* interface = revolution of the bone-shaped planar level-set around the z-axis, \n\
centered at (xmin + 0.15*.5*sqrt(2.0)*x_length, ymin + 0.15*.5*sqrt(2.0)*y_length, zmin + 0.20*z_length). \n\
The full periodicity is enforced.\n\
* mu_m = 1.0; \n\
* mu_p = 80.0; \n\
* u_m  = atan(sin((2.0*M_PI/3.0)*(2.0*x-y)))*log(1.5+cos((2.0*M_PI/3.0)*(2.0*y-z))); \n\
* u_p  = tanh(cos((2.0*M_PI/3.0)*(2.0*x+y)))*acos(0.5*sin((2.0*M_PI/3.0)*(2.0*x-x))); n\
* fully periodic \n\
Example by Raphael Egan for full periodicity.");
    #else
  cmd.add_option("test", "choose a test.\n\
0:\n\
* domain = [0, 1] X [0, 1] \n\
* interface = circle of radius 1/4, centered in (0.5, 0.5), negative inside, positive outside \n\
* mu_m = 2.0; \n\
* mu_p = 1.0; \n\
* u_m = exp(-x*x-y*y); \n\
* u_p = 0;\n\
* no periodicity \n\
Example 3 from Liu, Fedkiw, Kang 2000 \n\
1: \n\
* domain = [-1, 1] X [-1, 1] \n\
* interface = circle of radius 1/2, centered in (0, 0), negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 1.0; \n\
* u_m = 1.0; \n\
* u_p = 1.0 + log(2.0*sqrt(x*x + y*y)); \n\
* no periodicity \n\
Example 5 from Liu, Fedkiw, Kang 2000 \n\
2: \n\
* domain = [-1, 1] X [-1, 1] \n\
* interface = circle of radius 1/2, centered in (0, 0), negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 1.0; \n\
* u_m  = exp(x)*cos(y); \n\
* u_p  = 0.0; \n\
* no periodicity \n\
Example 6 from Liu, Fedkiw, Kang 2000 \n\
3: \n\
* domain = [-1, 1] X [-1, 1] \n\
* interface = circle of radius 1/2, centered in (0, 0), negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 1.0; \n\
* u_m  = x*x - y*y; \n\
* u_p  = 0.0; \n\
* no periodicity \n\
Example 7 from Liu, Fedkiw, Kang 2000 \n\
4: \n\
* domain = [-1, 1] X [-1, 1] \n\
* interface = curve parameterized by (t in [0, 2.0*PI[) \n\
@ x(t) = 0.02.0*sqrt(5) + (0.5 + 0.2*sin(5*t))*cos(t) \n\
@ y(t) = 0.02.0*sqrt(5) + (0.5 + 0.2*sin(5*t))*sin(t) \n\
negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 10.0; \n\
* u_m  = x*x + y*y; \n\
* u_p  = 0.1*(x*x + y*y)^2 - 0.01*log(2.0*sqrt(eps+x*x + y*y));\n\
* no periodicity \n\
Example 8 from Liu, Fedkiw, Kang 2000 \n\
5: \n\
* domain = [-1.5, 1.5] X [0.0, 3.0] \n\
* interface = curve parameterized by (t in [0, 2.0*PI[) \n\
@ x(t) = 0.6*cos(t) - 0.3*cos(3*t) \n\
@ y(t) = 1.5 + 0.7*sin(t) - 0.07*sin(3*t) + 0.2.0*sin(7*t) \n\
negative inside, positive outside \n\
* mu_m = 1.0; \n\
* mu_p = 10.0; \n\
* u_m  = exp(x)*(x*x*sin(y) + y*y); \n\
* u_p  = -x*x - y*y; \n\
* no periodicity \n\
Example 9 from Liu, Fedkiw, Kang 2000 \n\
6: \n\
* domain = [-1.0, 1.0] X [-1.0, 1.0] \n\
-interface = curve parameterized by (t in [0, 2.0*PI[) \n\
@ x(t) = (0.5 + 0.1*sin(5*t)).*cos(t) \n\
@ y(t) = (0.5 + 0.1*sin(5*t)).*cos(t) \n\
negative inside, positive outside \n\
* mu_m = 10000.0; \n\
* mu_p = 1.0; \n\
* u_m  = exp(x)*(x*x*sin(y) + y*y)/10000.0; \n\
* u_p  = 0.5 + (cos(x)*(y^4 + sin(y*y - x*x))); \n\
* no periodicity \n\
Example by Raphael Egan for large ratio of diffusion coefficient.\n\
7: \n\
* domain = [-1.0, 1.0] X [-1.0, 1.0] \n\
-interface = 15 small spherical bubbles in the domain, radius between 0.005 and 0.02 \n\
negative inside the bubbles, positive outside \n\
* mu_m = 1000.0; \n\
* mu_p = 1.0; \n\
* u_m  = (cos(200.0*M_PI*(x+3.0*y)) - sin(100.0*M_PI*(y - 2.0*x)))/10000.0; \n\
* u_p  = cos(x+y)*exp(-SQR(x*cos(y))); \n\
* no periodicity \n\
Example by Raphael Egan for adaptivity (can't be tested with Neumann boundary conditions). \n\
8: \n\
* domain = [-1.5, 1.5] X [-1.5, 1.5] \n\
* interface = curve parameterized by (t in [0, 2.0*PI[) \n\
@ x(t) = xmin + 0.15*(xmax-xmin) + 0.6*cos(t) - 0.3*cos(3*t) \n\
@ y(t) = 0.7*sin(t) - 0.07*sin(3*t) + 0.2*sin(7*t) \n\
negative inside, positive outside (periodicity along x enforced) \n\
* mu_m = 1.0; \n\
* mu_p = 10.0; \n\
* u_m  = cos((2.0*M_PI/3.0)*(x-tanh(y))) + exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x-0.251*y)))); \n\
* u_p  = tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y); \n\
* periodicity along x, no periodicity along y \n\
Example by Raphael Egan for periodicity in x.\n\
9: \n\
* domain = [-1.5, 1.5] X [-1.5, 1.5] \n\
* interface = curve parameterized by (t in [0, 2.0*PI[) \n\
@ x(t) = xmin + 0.15*(xmax-xmin) + 0.6*cos(t) - 0.3*cos(3*t) \n\
@ y(t) = ymin + 0.2*(ymax-ymin)  + 0.7*sin(t) - 0.07*sin(3*t) + 0.2*sin(7*t) \n\
negative inside, positive outside (periodicity along x and y enforced) \n\
* mu_m = 1.0; \n\
* mu_p = 100.0; \n\
* u_m  = atan(sin((2.0*M_PI/3.0)*(2.0*x-y))); \n\
* u_p  = log(1.5+cos((2.0*M_PI/3.0)*(-x+3.0*y))); \n\
* fully periodic \n\
Example by Raphael Egan for full periodicity.");
    #endif
      cmd.parse(argc, argv);
  cmd.print();



  int lmin = cmd.get<int>("lmin", lmin_);
  int lmax = cmd.get<int>("lmax", lmax_);
  int ngrids = cmd.get<int>("ngrids", ngrids_);
  int test_number = cmd.get<int>("test", test_number_);
  string out_dir = cmd.get<string>("out_dir", "/home/regan/workspace/projects/bubbles/cell_center_xgfm/output");
  BoundaryConditionType bc_wtype = cmd.get<BoundaryConditionType>("bc_wtype", bc_wtype_);
  int ntree = cmd.get<int>("ntree", ntree_);
  bool use_second_order_theta = cmd.get<bool>("second_order_ls", use_second_order_theta_);

#ifdef P4_TO_P8
  string summary_folder = cmd.get<string>("summary", "/home/regan/workspace/projects/bubbles/cell_center_xgfm/summaries/3D");
#else
  string summary_folder = cmd.get<string>("summary", "/home/regan/workspace/projects/bubbles/cell_center_xgfm/summaries/2D");
#endif
  string summary_file = summary_folder + "/summary_test" + to_string(test_number) + "_" + to_string(P4EST_DIM) + "D_lmin" + to_string(lmin) + "_lmax" + to_string(lmax) + "_ngrids" + to_string(ngrids) + "_ntree" + to_string(ntree) + "_accuracyls" + to_string(use_second_order_theta?2:1) + "_" + ((bc_wtype == DIRICHLET)? "dirichlet": "neumann") + ".dat";


  bool save_vtk = cmd.contains("save_vtk");

  parStopWatch watch, watch_global;
  watch_global.start("Total run time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  box domain;
  switch (test_number) {
#ifdef P4_TO_P8
  case 0:
    domain.xmin =  0.0;
    domain.xmax =  1.0;
    domain.ymin =  0.0;
    domain.ymax =  1.0;
    domain.zmin =  0.0;
    domain.zmax =  1.0;
    break;
  case 1:
  case 2:
    domain.xmin =  -2.0;
    domain.xmax =   2.0;
    domain.ymin =  -2.0;
    domain.ymax =   2.0;
    domain.zmin =  -2.0;
    domain.zmax =   2.0;
    break;
  case 3:
    domain.xmin =  -1.5;
    domain.xmax =   1.5;
    domain.ymin =  -1.5;
    domain.ymax =   1.5;
    domain.zmin =  -1.5;
    domain.zmax =   1.5;
    break;
#else
  case 0:
    domain.xmin =  0.0;
    domain.xmax =  1.0;
    domain.ymin =  0.0;
    domain.ymax =  1.0;
    break;
  case 1:
  case 2:
  case 3:
  case 4:
  case 6:
  case 7:
    domain.xmin =  -1.0;
    domain.xmax =  1.0;
    domain.ymin =  -1.0;
    domain.ymax =  1.0;
    break;
  case 5:
    domain.xmin =  -1.5;
    domain.xmax =  1.5;
    domain.ymin =  0.0;
    domain.ymax =  3.0;
    break;
  case 8:
  case 9:
    domain.xmin =  -1.5;
    domain.xmax =  1.5;
    domain.ymin =  -1.5;
    domain.ymax =  1.5;
    break;
#endif
  default:
    throw std::invalid_argument("invalid test number.");
  }
#ifdef P4_TO_P8
  int n_xyz [] = {ntree_, ntree_, ntree_};
  double xyz_min [] = {domain.xmin, domain.ymin, domain.zmin};
  double xyz_max [] = {domain.xmax, domain.ymax, domain.zmax};
#else
  int n_xyz [] = {ntree_, ntree_};
  double xyz_min [] = {domain.xmin, domain.ymin};
  double xyz_max [] = {domain.xmax, domain.ymax};
#endif

  int periodic[3];
  switch (test_number) {
#ifdef P4_TO_P8
  case 0:
  case 1:
  case 2:
    periodic[0] = periodic[1] = periodic[2] = 0;
    break;
  case 3:
    periodic[0] = periodic[1] = periodic[2] = 1;
    break;
#else
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
    periodic[0] = periodic[1] = periodic[2] = 0;
    break;
  case 8:
    periodic[0] = 1;
    periodic[1] = periodic[2] = 0;
    break;
  case 9:
    periodic[0] = periodic[1] = 1;
    periodic[2] = 0;
    break;
#endif
  default:
    throw std::invalid_argument("invalid test number.");
    break;
  }

  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est, *p4est_fine;
  p4est_nodes_t *nodes, *nodes_fine;
  p4est_ghost_t *ghost, *ghost_fine;

  double err[ngrids][2], err_flux_components[ngrids][2][P4EST_DIM], err_derivatives_components[ngrids][2][P4EST_DIM];

  double avg_exa = 0.0;
  if(bc_wtype==NEUMANN ||
   #ifdef P4_TO_P8
     test_number == 3
   #else
     test_number == 9
   #endif
     )
  {
    switch(test_number)
    {
#ifdef P4_TO_P8
    case 0: avg_exa = 0.030340553122903;  break; // using Richardson's extrapolation between uniform 512x512x512 and 1024x1024x1024 grids (assuming second-order accurate integration)
    case 1: avg_exa = 197.6819076161000;  break; // using Richardson's extrapolation between uniform 512x512x512 and 1024x1024x1024 grids (assuming second-order accurate integration)
    case 2: throw std::invalid_argument("Test case 2 cannot be validated with a non-empty nullspace (i.e. Neumann boundary condition). \n\
This test case is meant to check the AMR feature, hence the interface is supposedly captured with a very fine grid for which an accurate estimate of the mean value is unknown...");
    case 3: avg_exa = -0.164222868617700; break; // using Richardson's extrapolation between uniform 512x512x512 and 1024x1024x1024 grids (assuming second-order accurate integration)
#else
    case 0: avg_exa = 0.117241067253686032041502125837326856300828722393373744878; break; // calculated with Wolfram
    case 1: avg_exa =
          (double)(domain.xmax-domain.xmin)*(domain.ymax-domain.ymin)
          + 2.0*M_PI*(0.5*log(2.0)-0.25 + 0.25*0.5*0.5)
          + (-M_PI_2*2.0 - 6.0 + M_PI*2.0*log(2.0*sqrt(2.0)) + 2.0*2.0*acos(1.0/sqrt(2.0)) + 4.0*sqrt(2.0)*log(2.0*sqrt(2.0))*(1.0/sqrt(2.0) - sqrt(2.0)*acos(1.0/sqrt(2.0))) - 4.0*asin(1.0/sqrt(2.0)))
          - (-M_PI_2 + M_PI*log(2.0) - 2.0*M_PI); break; // analytically calculated (with Wolfram's help)
    case 2: avg_exa = 0.785398163397448309615660845819875721049292349843776455243; break; // calculated with Wolfram
    case 3: avg_exa = 0.0; break;
    case 4: avg_exa = 0.3782030713479; break; // calculated on a 14/14 grid (1X1 macromesh) on stampede...
    case 5: avg_exa = -25.59547830010; break; // calculated on a 14/14 grid (1X1 macromesh) on stampede...
    case 6: avg_exa = 2.415389999053; break; // calculated on a 14/14 grid (1X1 macromesh) on stampede...
    case 7: throw std::invalid_argument("Test case 7 cannot be validated with a non-empty nullspace (i.e. Neumann boundary condition) because of the random interface...");
    case 8: avg_exa = 0.6448672580288; break; // calculated on a 14/14 grid (1X1 macromesh) on stampede...
    case 9: avg_exa = 1.652618403615; break;  // calculated on a 14/14 grid (1X1 macromesh) on stampede...
#endif
    default: throw std::invalid_argument("invalid test number.");
    }

#ifdef P4_TO_P8
    avg_exa /= (domain.xmax-domain.xmin)*(domain.ymax-domain.ymin)*(domain.zmax-domain.zmin);
#else
    avg_exa /= (domain.xmax-domain.xmin)*(domain.ymax-domain.ymin);
#endif
  }

  LEVEL_SET levelset(domain, test_number);

  double mu_m, mu_p;
  switch (test_number) {
#ifdef P4_TO_P8
  case 0:
    mu_m = 2.0;
    mu_p = 1.0;
    break;
  case 1:
    mu_m = 2000.0;
    mu_p = 1.0;
    break;
  case 2:
    mu_m = 1.0;
    mu_p = 1250.0;
    break;
  case 3:
    mu_m = 1.0;
    mu_p = 80.0;
    break;
#else
  case 0:
    mu_m = 2.0;
    mu_p = 1.0;
    break;
  case 1:
  case 2:
  case 3:
    mu_m = 1.0;
    mu_p = 1.0;
    break;
  case 4:
  case 5:
    mu_m = 1.0;
    mu_p = 10.0;
    break;
  case 6:
    mu_m = 10000.0;
    mu_p = 1.0;
    break;
  case 7:
    mu_m = 1000.0;
    mu_p = 1.0;
    break;
  case 8:
    mu_m = 1.0;
    mu_p = 10.0;
    break;
  case 9:
    mu_m = 1.0;
    mu_p = 100.0;
    break;
#endif
  default:
    throw std::invalid_argument("set mus: unknown test number.");
  }

  for(int iter=0; iter<ngrids; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    /* build the computational grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods, its cell neighborhoods
     * the REINITIALIZED levelset on the computational grid
     */
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &levelset, 1.2);
    p4est->user_pointer = (void*)(&data);

    for (int i = 0; i < lmax+iter; ++i) {
#ifdef P4_TO_P8
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
#else
      if(test_number != 7)
        my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      else
        my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_finest_in_negative, NULL);
#endif
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    nodes = my_p4est_nodes_new(p4est, ghost);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes); ngbd_n.init_neighbors();
    Vec phi_coarse;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_coarse); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, levelset, phi_coarse);
    my_p4est_level_set_t ls_coarse(&ngbd_n);
    ls_coarse.reinitialize_2nd_order(phi_coarse);

    const double *phi_coarse_read_p;
    ierr = VecGetArrayRead(phi_coarse, &phi_coarse_read_p); CHKERRXX(ierr);
    splitting_criteria_tag_t data_tag(lmin+iter, lmax+iter, 1.2);
    p4est_t* new_p4est = p4est_copy(p4est, P4EST_FALSE);

    while(data_tag.refine_and_coarsen(new_p4est, nodes, phi_coarse_read_p, test_number==7))
    {
      ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_read_p); CHKERRXX(ierr);
      my_p4est_interpolation_nodes_t interp_nodes(&ngbd_n);
      interp_nodes.set_input(phi_coarse, linear);

      my_p4est_partition(new_p4est, P4EST_FALSE, NULL);
      p4est_ghost_t *new_ghost  = my_p4est_ghost_new(new_p4est, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(new_p4est, new_ghost);
      p4est_nodes_t *new_nodes  = my_p4est_nodes_new(new_p4est, new_ghost);
      Vec new_coarse_phi;
      ierr = VecCreateGhostNodes(new_p4est, new_nodes, &new_coarse_phi); CHKERRXX(ierr);
      for(size_t nn=0; nn<new_nodes->indep_nodes.elem_count; ++nn)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(nn, new_p4est, new_nodes, xyz);
        interp_nodes.add_point(nn, xyz);
      }
      interp_nodes.interpolate(new_coarse_phi);


      p4est_destroy(p4est); p4est = new_p4est; new_p4est = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_destroy(ghost); ghost = new_ghost;
      hierarchy.update(p4est, ghost);
      p4est_nodes_destroy(nodes); nodes = new_nodes;
      ngbd_n.update(&hierarchy, nodes);

      ierr = VecDestroy(phi_coarse); CHKERRXX(ierr); phi_coarse = new_coarse_phi;

      ierr = VecGetArrayRead(phi_coarse, &phi_coarse_read_p); CHKERRXX(ierr);
    }
    ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_read_p); CHKERRXX(ierr);
    p4est_destroy(new_p4est);
    my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
    my_p4est_faces_t faces(p4est, ghost, &brick, &ngbd_c);

    /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
     * the REINITIALIZED levelset on the interface-capturing grid
     */
    p4est_fine = p4est_copy(p4est, P4EST_FALSE);
    splitting_criteria_cf_t data_fine(lmin+iter, lmax+1+iter, &levelset, 1.2);
    p4est_fine->user_pointer = (void*)(&data_fine);
    my_p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);
    ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_fine, ghost_fine);
    my_p4est_hierarchy_t hierarchy_fine(p4est_fine, ghost_fine, &brick);
    nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
    my_p4est_node_neighbors_t ngbd_n_fine(&hierarchy_fine, nodes_fine); ngbd_n_fine.init_neighbors();
    Vec phi;
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_fine, nodes_fine, levelset, phi);
    my_p4est_level_set_t ls(&ngbd_n_fine);
    ls.reinitialize_2nd_order(phi);

    const double *phi_read_p;
    ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
    splitting_criteria_tag_t data_tag_fine(lmin+iter, lmax+1+iter, 1.2);
    p4est_t *new_p4est_fine = p4est_copy(p4est_fine, P4EST_FALSE);

    while(data_tag_fine.refine(new_p4est_fine, nodes_fine, phi_read_p)) // not refine_and_corsen, because we need the fine grid to be everywhere finer or as coarse as the coarse grid!
    {
      ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
      my_p4est_interpolation_nodes_t interp_nodes_fine(&ngbd_n_fine);
      interp_nodes_fine.set_input(phi, linear);

      p4est_ghost_t *new_ghost_fine = my_p4est_ghost_new(new_p4est_fine, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(new_p4est_fine, new_ghost_fine);
      p4est_nodes_t *new_nodes_fine  = my_p4est_nodes_new(new_p4est_fine, new_ghost_fine);
      Vec new_phi;
      ierr = VecCreateGhostNodes(new_p4est_fine, new_nodes_fine, &new_phi); CHKERRXX(ierr);
      for(size_t nn=0; nn<new_nodes_fine->indep_nodes.elem_count; ++nn)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(nn, new_p4est_fine, new_nodes_fine, xyz);
        interp_nodes_fine.add_point(nn, xyz);
      }
      interp_nodes_fine.interpolate(new_phi);


      p4est_destroy(p4est_fine); p4est_fine = new_p4est_fine; new_p4est_fine = p4est_copy(p4est_fine, P4EST_FALSE);
      p4est_ghost_destroy(ghost_fine); ghost_fine = new_ghost_fine;
      hierarchy_fine.update(p4est_fine, ghost_fine);
      p4est_nodes_destroy(nodes_fine); nodes_fine = new_nodes_fine;
      ngbd_n_fine.update(&hierarchy_fine, nodes_fine);
      ls.update(&ngbd_n_fine);

      ierr = VecDestroy(phi); CHKERRXX(ierr); phi = new_phi;

      ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
    }

    ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
    p4est_destroy(new_p4est_fine);

    /* Get the normals, the second derivatives of the levelset (if required) and the relevant flattened jumps
     */
    Vec jump_u, jump_normal_flux;
    Vec normals[P4EST_DIM];
    Vec phi_xx = NULL;
    Vec phi_yy = NULL;
#ifdef P4_TO_P8
    Vec phi_zz = NULL;
#endif
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &jump_u); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &jump_normal_flux); CHKERRXX(ierr);
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &normals[dim]); CHKERRXX(ierr);
    }
    if(use_second_order_theta){
      ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &phi_xx); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &phi_zz); CHKERRXX(ierr);
#endif
    }

    get_normals_and_flattened_jumps(p4est_fine, nodes_fine, ngbd_n_fine, phi, use_second_order_theta, test_number, mu_p, mu_m, //input
                          jump_u, jump_normal_flux, normals, phi_xx, phi_yy // output
                      #ifdef P4_TO_P8
                          , phi_zz // output
                      #endif
                          );

    /* TEST THE JUMP SOLVER */
#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    BCWALLTYPE bc_wall_type(bc_wtype);
    bc.setWallTypes(bc_wall_type);
    BCWALLVAL bc_wall_val(domain, levelset, bc_wtype, test_number);
    bc.setWallValues(bc_wall_val);

    Vec rhs_original;
    ierr = VecCreateCellsNoGhost(p4est, &rhs_original); CHKERRXX(ierr);
    get_sharp_rhs(p4est, ghost, ngbd_n_fine, phi, test_number, mu_p, mu_m, rhs_original);

    Vec sol[2], err_cells[2], extended_field_xgfm, extended_field_fine_nodes_xgfm;
    Vec exact_msol_at_nodes, exact_psol_at_nodes; // to enable illustration of exact solution with wrap-by-scalar in paraview
    Vec jump_mu_grad_u[2][P4EST_DIM];
    for (short flag = 0; flag < 2; ++flag) {
      my_p4est_xgfm_cells_t solver(&ngbd_c, &ngbd_n, &ngbd_n_fine, flag);
      if(use_second_order_theta)
  #ifdef P4_TO_P8
        solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
  #else
        solver.set_phi(phi, phi_xx, phi_yy);
  #endif
      else
        solver.set_phi(phi);
      solver.set_normals(normals);
      solver.set_mus(mu_m, mu_p);
      solver.set_jumps(jump_u, jump_normal_flux);
      solver.set_diagonals(0.0, 0.0);
      solver.set_bc(bc);
      Vec rhs;
      ierr = VecCreateCellsNoGhost(p4est, &rhs); CHKERRXX(ierr);
      ierr = VecCopy(rhs_original, rhs);
      solver.set_rhs(rhs);


      watch.start("Total time:");
      solver.solve();
      watch.stop(); watch.print_duration();

      if(flag)
        solver.get_extended_interface_values(extended_field_xgfm, extended_field_fine_nodes_xgfm);

      Vec flux[P4EST_DIM];
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecCreateGhostFaces(p4est, &faces, &flux[dim], dim); CHKERRXX(ierr);}
      solver.get_flux_components(flux, &faces);

      sol[flag] = solver.get_solution();
      solver.get_jump_mu_grad_u(jump_mu_grad_u[flag]);


      vector<double> max_corr = solver.get_max_corrections();
      vector<double> rel_res = solver.get_relative_residuals();
      vector<PetscInt> nb_iter = solver.get_numbers_of_ksp_iterations();

      if(mpi.rank() == 0)
      {
        PetscInt total_nb_iterations = nb_iter.at(0);
        for (size_t tt = 1; tt < max_corr.size(); ++tt){
          total_nb_iterations += nb_iter.at(tt);
          if(track_residuals_and_corrections)
            cout << "max corr " << tt << " = " << max_corr[tt] << " and rel residual " << tt << " = " << rel_res[tt] << " after " << nb_iter[tt] << " iterations." << endl;
        }
        cout << "The solver converged after a total of " << total_nb_iterations << " iterations." << std::endl;
      }

      if((save_vtk || get_integral) && flag)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &exact_msol_at_nodes); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &exact_psol_at_nodes); CHKERRXX(ierr);
        double *exact_msol_at_nodes_p, *exact_psol_at_nodes_p;
        ierr = VecGetArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
        ierr = VecGetArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
        for (size_t node_idx = 0; node_idx < nodes->indep_nodes.elem_count; ++node_idx) {
          double xyz_node[P4EST_DIM];
          node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
#ifdef P4_TO_P8
          exact_msol_at_nodes_p[node_idx] = u_exact_m(test_number, xyz_node[0], xyz_node[1], xyz_node[2]);
          exact_psol_at_nodes_p[node_idx] = u_exact_p(test_number, xyz_node[0], xyz_node[1], xyz_node[2]);
#else
          exact_msol_at_nodes_p[node_idx] = u_exact_m(test_number, xyz_node[0], xyz_node[1]);
          exact_psol_at_nodes_p[node_idx] = u_exact_p(test_number, xyz_node[0], xyz_node[1]);
#endif
        }
        ierr = VecRestoreArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);

        double integral_of_exact = 0.0;
        integral_of_exact += integrate_over_negative_domain(p4est, nodes, phi_coarse, exact_msol_at_nodes);
        if(ISNAN(integral_of_exact))
          std::cout << "the first integral part is nan" << std::endl;
        Vec phi_coarse_loc;
        ierr = VecGhostGetLocalForm(phi_coarse, &phi_coarse_loc); CHKERRXX(ierr);
        ierr = VecScale(phi_coarse_loc, -1.0); CHKERRXX(ierr);
        ierr = VecGhostRestoreLocalForm(phi_coarse, &phi_coarse_loc); CHKERRXX(ierr);
        integral_of_exact += integrate_over_negative_domain(p4est, nodes, phi_coarse, exact_psol_at_nodes);
        ierr = VecGhostGetLocalForm(phi_coarse, &phi_coarse_loc); CHKERRXX(ierr);
        ierr = VecScale(phi_coarse_loc, -1.0); CHKERRXX(ierr);
        ierr = VecGhostRestoreLocalForm(phi_coarse, &phi_coarse_loc); CHKERRXX(ierr);
        ierr = PetscPrintf(mpi.comm(), "The integral calculated with exact fields is %.12e \n", integral_of_exact); CHKERRXX(ierr);
      }


      /* if all NEUMANN boundary conditions, shift solution */
      if(solver.get_matrix_has_nullspace())
      {
        my_p4est_level_set_cells_t lsc(&ngbd_c, &ngbd_n);
        double avg_sol = lsc.integrate(phi_coarse, sol[flag]);
        Vec phi_loc;
        ierr = VecGhostGetLocalForm(phi_coarse, &phi_loc); CHKERRXX(ierr);
        ierr = VecScale(phi_loc, -1.0); CHKERRXX(ierr);
        ierr = VecGhostRestoreLocalForm(phi_coarse, &phi_loc); CHKERRXX(ierr);
        MPI_Barrier(p4est->mpicomm);
        avg_sol += lsc.integrate(phi_coarse, sol[flag]);
        ierr = VecGhostGetLocalForm(phi_coarse, &phi_loc); CHKERRXX(ierr);
        ierr = VecScale(phi_loc, -1.0); CHKERRXX(ierr);
        ierr = VecGhostRestoreLocalForm(phi_coarse, &phi_loc); CHKERRXX(ierr);
        avg_sol /=((domain.xmax-domain.xmin)*(domain.ymax-domain.ymin));

        double *sol_p;
        ierr = VecGetArray(sol[flag], &sol_p); CHKERRXX(ierr);

        for(p4est_locidx_t quad_idx=0; quad_idx<p4est->local_num_quadrants; ++quad_idx)
          sol_p[quad_idx] = sol_p[quad_idx] - avg_sol + avg_exa;

        for(size_t quad_idx=0; quad_idx<ghost->ghosts.elem_count; ++quad_idx)
          sol_p[quad_idx+p4est->local_num_quadrants] = sol_p[quad_idx+p4est->local_num_quadrants] - avg_sol + avg_exa;

        ierr = VecRestoreArray(sol[flag], &sol_p); CHKERRXX(ierr);
      }

      /* check the error */

      ierr = VecCreateGhostCells(p4est, ghost, &err_cells[flag]); CHKERRXX(ierr);
      measure_errors(p4est, ghost, ngbd_n_fine, &faces, phi, test_number, mu_p, mu_m, sol[flag], flux,
                     err_cells[flag], err[iter][flag], err_flux_components[iter][flag], err_derivatives_components[iter][flag]);
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecDestroy(flux[dim]); CHKERRXX(ierr);}
      ierr = VecDestroy(rhs); CHKERRXX(ierr);
    }

    Vec correction_jump_mu_grad[P4EST_DIM], loc_ghost_jump_mu_grad_u_gfm, loc_ghost_jump_mu_grad_u_xgfm, loc_ghost_correction;
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecDuplicate(jump_mu_grad_u[0][dim], &correction_jump_mu_grad[dim]); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(jump_mu_grad_u[0][dim], &loc_ghost_jump_mu_grad_u_gfm); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(jump_mu_grad_u[1][dim], &loc_ghost_jump_mu_grad_u_xgfm); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(correction_jump_mu_grad[dim], &loc_ghost_correction); CHKERRXX(ierr);
      ierr = VecCopy(loc_ghost_jump_mu_grad_u_xgfm, loc_ghost_correction); CHKERRXX(ierr);
      ierr = VecAXPY(loc_ghost_correction, -1.0, loc_ghost_jump_mu_grad_u_gfm); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(correction_jump_mu_grad[dim], &loc_ghost_correction); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(jump_mu_grad_u[1][dim], &loc_ghost_jump_mu_grad_u_xgfm); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(jump_mu_grad_u[0][dim], &loc_ghost_jump_mu_grad_u_gfm); CHKERRXX(ierr);
    }

    if(iter > 0){
      ierr = PetscPrintf(p4est->mpicomm, "Error on cells  for  gfm: %.5e, order = %g\n", err[iter][0], log(err[iter-1][0]/err[iter][0])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on cells  for xgfm: %.5e, order = %g\n", err[iter][1], log(err[iter-1][1]/err[iter][1])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-x for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][0], log(err_flux_components[iter-1][0][0]/err_flux_components[iter][0][0])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-x for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][0], log(err_flux_components[iter-1][1][0]/err_flux_components[iter][1][0])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-y for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][1], log(err_flux_components[iter-1][0][1]/err_flux_components[iter][0][1])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-y for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][1], log(err_flux_components[iter-1][1][1]/err_flux_components[iter][1][1])/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-z for  gfm: %.5e, order = %g\n", err_flux_components[iter][0][2], log(err_flux_components[iter-1][0][2]/err_flux_components[iter][0][2])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-z for xgfm: %.5e, order = %g\n", err_flux_components[iter][1][2], log(err_flux_components[iter-1][1][2]/err_flux_components[iter][1][2])/log(2)); CHKERRXX(ierr);
#endif
      ierr = PetscPrintf(p4est->mpicomm, "Error on x-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][0], log(err_derivatives_components[iter-1][0][0]/err_derivatives_components[iter][0][0])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on x-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][0], log(err_derivatives_components[iter-1][1][0]/err_derivatives_components[iter][1][0])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on y-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][1], log(err_derivatives_components[iter-1][0][1]/err_derivatives_components[iter][0][1])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on y-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][1], log(err_derivatives_components[iter-1][1][1]/err_derivatives_components[iter][1][1])/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(p4est->mpicomm, "Error on z-der  for  gfm: %.5e, order = %g\n", err_derivatives_components[iter][0][2], log(err_derivatives_components[iter-1][0][2]/err_derivatives_components[iter][0][2])/log(2)); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on z-der  for xgfm: %.5e, order = %g\n", err_derivatives_components[iter][1][2], log(err_derivatives_components[iter-1][1][2]/err_derivatives_components[iter][1][2])/log(2)); CHKERRXX(ierr);
#endif
    }
    else{
      ierr = PetscPrintf(p4est->mpicomm, "Error on cells  for  gfm: %.5e\n", err[iter][0]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on cells  for xgfm: %.5e\n", err[iter][1]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-x for  gfm: %.5e\n", err_flux_components[iter][0][0]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-x for xgfm: %.5e\n", err_flux_components[iter][1][0]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-y for  gfm: %.5e\n", err_flux_components[iter][0][1]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-y for xgfm: %.5e\n", err_flux_components[iter][1][1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-z for  gfm: %.5e\n", err_flux_components[iter][0][2]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on flux-z for xgfm: %.5e\n", err_flux_components[iter][1][2]); CHKERRXX(ierr);
#endif
      ierr = PetscPrintf(p4est->mpicomm, "Error on x-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][0]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on x-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][0]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on y-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][1]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on y-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(p4est->mpicomm, "Error on z-der  for  gfm: %.5e\n", err_derivatives_components[iter][0][2]); CHKERRXX(ierr);
      ierr = PetscPrintf(p4est->mpicomm, "Error on z-der  for xgfm: %.5e\n", err_derivatives_components[iter][1][2]); CHKERRXX(ierr);
#endif
    }

    if(save_vtk)
      save_VTK(out_dir, test_number,
               p4est, ghost, nodes,
               p4est_fine, ghost_fine, nodes_fine, &brick,
               phi, normals, jump_u, jump_normal_flux, extended_field_fine_nodes_xgfm, jump_mu_grad_u, correction_jump_mu_grad,
               sol, err_cells, extended_field_xgfm
         #ifndef P4_TO_P8
               , exact_msol_at_nodes, exact_psol_at_nodes, phi_coarse
         #endif
               );


    ierr = VecDestroy(phi_coarse); CHKERRXX(ierr);
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    if(use_second_order_theta)
    {
      ierr = VecDestroy(phi_xx); CHKERRXX(ierr);
      ierr = VecDestroy(phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecDestroy(phi_zz); CHKERRXX(ierr);
#endif
    }
    ierr = VecDestroy(jump_u); CHKERRXX(ierr);
    ierr = VecDestroy(jump_normal_flux); CHKERRXX(ierr);
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDestroy(normals[dim]); CHKERRXX(ierr);
      for (short flag = 0; flag < 2; ++flag) {
        ierr = VecDestroy(jump_mu_grad_u[flag][dim]); CHKERRXX(ierr);
      }
      ierr = VecDestroy(correction_jump_mu_grad[dim]); CHKERRXX(ierr);
    }
    ierr = VecDestroy(rhs_original); CHKERRXX(ierr);
    for (short flag = 0; flag < 2; ++flag)
    {
      ierr = VecDestroy(sol[flag]); CHKERRXX(ierr);
      ierr = VecDestroy(err_cells[flag]); CHKERRXX(ierr);
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecDestroy(jump_mu_grad_u[flag][dim]); CHKERRXX(ierr);}
    }

    if(save_vtk || get_integral)
    {
      ierr = VecDestroy(exact_msol_at_nodes); CHKERRXX(ierr);
      ierr = VecDestroy(exact_psol_at_nodes); CHKERRXX(ierr);
    }

    ierr = VecDestroy(extended_field_xgfm); CHKERRXX(ierr);
    ierr = VecDestroy(extended_field_fine_nodes_xgfm); CHKERRXX(ierr);


    p4est_nodes_destroy(nodes); p4est_nodes_destroy(nodes_fine);
    p4est_ghost_destroy(ghost); p4est_ghost_destroy(ghost_fine);
    p4est_destroy      (p4est); p4est_destroy(p4est_fine);
  }

  if(mpi.rank() == 0)
  {
    if(print_summary)
    {
      FILE *fid = fopen(summary_file.c_str(), "w");
      fprintf(fid, "=================================================================\n");
      fprintf(fid, "========================= SUMMARY ===============================\n");
      fprintf(fid, "=================================================================\n");
      fprintf(fid, "Test number %d in %d-D\n", test_number, P4EST_DIM);
      fprintf(fid, "lmin: %d\n", lmin);
      fprintf(fid, "lmax: %d\n", lmax);
      fprintf(fid, "Number of grids: %d\n", ngrids);
      fprintf(fid, "Number of trees along minimum dimension of domain in macromesh: %d\n", ntree);
      fprintf(fid, "Order of accuracy for interface localization: %d\n", (use_second_order_theta? 2:1));
      fprintf(fid, "Wall boundary condition: %s\n", ((bc_wtype == DIRICHLET)? "dirichlet" : "neumann"));
      fprintf(fid, "Resolution: " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%d/%d ", ntree_*(1<<(lmin+k)), ntree_*(1<<(lmax+k)));
        else
          fprintf(fid, "%d/%d\n", ntree_*(1<<(lmin+k)), ntree_*(1<<(lmax+k)));
      }
      fprintf(fid, "Error on solution (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err[k][0]);
        else
          fprintf(fid, "%.5e\n", err[k][0]);
      }
      fprintf(fid, "Error on solution (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err[k][1]);
        else
          fprintf(fid, "%.5e\n", err[k][1]);
      }

      fprintf(fid, "Error on x-derivative (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][0][0]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][0][0]);
      }
      fprintf(fid, "Error on x-derivative (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][1][0]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][1][0]);
      }

      fprintf(fid, "Error on y-derivative (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][0][1]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][0][1]);
      }
      fprintf(fid, "Error on y-derivative (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][1][1]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][1][1]);
      }
#ifdef P4_TO_P8
      fprintf(fid, "Error on z-derivative (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][0][2]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][0][2]);
      }
      fprintf(fid, "Error on z-derivative (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_derivatives_components[k][1][2]);
        else
          fprintf(fid, "%.5e\n", err_derivatives_components[k][1][2]);
      }
#endif

      fprintf(fid, "Error on x-flux (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][0][0]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][0][0]);
      }
      fprintf(fid, "Error on x-flux (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][1][0]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][1][0]);
      }

      fprintf(fid, "Error on y-flux (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][0][1]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][0][1]);
      }
      fprintf(fid, "Error on y-flux (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][1][1]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][1][1]);
      }
#ifdef P4_TO_P8
      fprintf(fid, "Error on z-flux (gfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][0][2]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][0][2]);
      }
      fprintf(fid, "Error on z-flux (xgfm): " );
      for (int k = 0; k < ngrids; ++k)
      {
        if(k!=ngrids-1)
          fprintf(fid, "%.5e ", err_flux_components[k][1][2]);
        else
          fprintf(fid, "%.5e\n", err_flux_components[k][1][2]);
      }
#endif

      fprintf(fid, "=================================================================\n");
      fprintf(fid, "===================== END OF SUMMARY ============================\n");
      fprintf(fid, "=================================================================");
      fclose(fid);
      printf("Summary file printed in %s\n", summary_file.c_str());
    }
  }

  my_p4est_brick_destroy(connectivity, &brick);

  watch_global.stop(); watch_global.print_duration();

  return 0;
}
