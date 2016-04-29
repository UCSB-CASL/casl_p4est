#include "grid_interpolation2.h"

double grid_interpolation2_t::linear(double *f, double x, double y)
{
  double eps = 1.e-12;

  // check if x and y within the grid
#ifdef CASL_THROWS
  if (x < xm || x > xp || y < ym || y > yp) throw std::invalid_argument("[CASL_ERROR]: Point of interpolation outside the grid.");
#endif

  // find quad
  int i = std::floor((x-xm)/dx);
  int j = std::floor((y-ym)/dy);

  // calculate relative distances to edges
  double d_m0 = (x-xm)/dx - (double)(i);
  double d_p0 = 1.-d_m0;

  double d_0m = (y-ym)/dy - (double)(j);
  double d_0p = 1.-d_0m;

  // check if a point is a corner point
  if (d_m0 < eps && d_0m < eps) return f[(j+0)*(nx+1) + (i+0)];
  if (d_p0 < eps && d_0m < eps) return f[(j+0)*(nx+1) + (i+1)];
  if (d_m0 < eps && d_0p < eps) return f[(j+1)*(nx+1) + (i+0)];
  if (d_p0 < eps && d_0p < eps) return f[(j+1)*(nx+1) + (i+1)];

  // do bilinear interpolation
  double w_mm = d_p0*d_0p;
  double w_pm = d_m0*d_0p;
  double w_mp = d_p0*d_0m;
  double w_pp = d_m0*d_0m;

  return  w_mm*f[(j+0)*(nx+1) + (i+0)] +
          w_pm*f[(j+0)*(nx+1) + (i+1)] +
          w_mp*f[(j+1)*(nx+1) + (i+0)] +
          w_pp*f[(j+1)*(nx+1) + (i+1)];
}

double grid_interpolation2_t::quadratic(double *f, double *f_xx, double *f_yy, double x, double y)
{
  double eps = 1.e-12;

  // check if x and y within the grid
#ifdef CASL_THROWS
  if (x < xm || x > xp || y < ym || y > yp) throw std::invalid_argument("[CASL_ERROR]: Point of interpolation outside the grid.");
#endif

  // find quad
  int i = std::floor((x-xm)/dx);
  int j = std::floor((y-ym)/dy);

  // calculate relative distances to edges
  double d_m0 = (x-xm)/dx - (double)(i);
  double d_p0 = 1.-d_m0;

  double d_0m = (y-ym)/dy - (double)(j);
  double d_0p = 1.-d_0m;

  // check if a point is a corner point
  if (d_m0 < eps && d_0m < eps) return f[(j+0)*(nx+1) + (i+0)];
  if (d_p0 < eps && d_0m < eps) return f[(j+0)*(nx+1) + (i+1)];
  if (d_m0 < eps && d_0p < eps) return f[(j+1)*(nx+1) + (i+0)];
  if (d_p0 < eps && d_0p < eps) return f[(j+1)*(nx+1) + (i+1)];

  // do quadratic interpolation
  double w_mm = d_p0*d_0p;
  double w_pm = d_m0*d_0p;
  double w_mp = d_p0*d_0m;
  double w_pp = d_m0*d_0m;

  double F = w_mm*f[(j+0)*(nx+1) + (i+0)] +
             w_pm*f[(j+0)*(nx+1) + (i+1)] +
             w_mp*f[(j+1)*(nx+1) + (i+0)] +
             w_pp*f[(j+1)*(nx+1) + (i+1)];

  double Fxx = w_mm*f_xx[(j+0)*(nx+1) + (i+0)] +
               w_pm*f_xx[(j+0)*(nx+1) + (i+1)] +
               w_mp*f_xx[(j+1)*(nx+1) + (i+0)] +
               w_pp*f_xx[(j+1)*(nx+1) + (i+1)];

  double Fyy = w_mm*f_yy[(j+0)*(nx+1) + (i+0)] +
               w_pm*f_yy[(j+0)*(nx+1) + (i+1)] +
               w_mp*f_yy[(j+1)*(nx+1) + (i+0)] +
               w_pp*f_yy[(j+1)*(nx+1) + (i+1)];

  F -= 0.5*(dx*dx*d_p0*d_m0*Fxx + dy*dy*d_0p*d_0m*Fyy);

  return F;
}
