#include "grid_interpolation3.h"

double grid_interpolation3_t::linear(double *f, double x, double y, double z)
{
  double eps = 1.e-12;

  // check if x and y within the grid
#ifdef CASL_THROWS
  if (x < xm-eps || x > xp+eps || y < ym-eps || y > yp+eps || z < zm-eps || z > zp+eps) throw std::invalid_argument("[CASL_ERROR]: Point of interpolation outside the grid.");
#endif

  // find quad
  int i = std::floor((x-xm)/dx);
  int j = std::floor((y-ym)/dy);
  int k = std::floor((z-zm)/dz);

  // calculate relative distances to edges
  double d_m00 = (x-xm)/dx - (double)(i);
  if (d_m00 < 0.) d_m00 = 0.;
  if (d_m00 > 1.) d_m00 = 1.;
  double d_p00 = 1.-d_m00;

  double d_0m0 = (y-ym)/dy - (double)(j);
  if (d_0m0 < 0.) d_0m0 = 0.;
  if (d_0m0 > 1.) d_0m0 = 1.;
  double d_0p0 = 1.-d_0m0;

  double d_00m = (z-zm)/dz - (double)(k);
  if (d_00m < 0.) d_00m = 0.;
  if (d_00m > 1.) d_00m = 1.;
  double d_00p = 1.-d_00m;

  // check if a point is a corner point
  if (d_m00 < eps && d_0m0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0m0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0p0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0p0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0m0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0m0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0p0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0p0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  // do trilinear interpolation
  double w_mmm = d_p00*d_0p0*d_00p;
  double w_pmm = d_m00*d_0p0*d_00p;
  double w_mpm = d_p00*d_0m0*d_00p;
  double w_ppm = d_m00*d_0m0*d_00p;
  double w_mmp = d_p00*d_0p0*d_00m;
  double w_pmp = d_m00*d_0p0*d_00m;
  double w_mpp = d_p00*d_0m0*d_00m;
  double w_ppp = d_m00*d_0m0*d_00m;

  return
      w_mmm*f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmm*f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpm*f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppm*f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)] +
      w_mmp*f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmp*f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpp*f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppp*f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];
}

double grid_interpolation3_t::quadratic(double *f, double *f_xx, double *f_yy, double *f_zz,
                                        double x, double y, double z)
{
  double eps = 1.e-12;

  // check if x and y within the grid
#ifdef CASL_THROWS
  if (x < xm-eps || x > xp+eps || y < ym-eps || y > yp+eps || z < zm-eps || z > zp+eps) throw std::invalid_argument("[CASL_ERROR]: Point of interpolation outside the grid.");
#endif

  // find quad
  int i = std::floor((x-xm)/dx);
  int j = std::floor((y-ym)/dy);
  int k = std::floor((z-zm)/dz);

  // calculate relative distances to edges
  double d_m00 = (x-xm)/dx - (double)(i);
  if (d_m00 < 0.) d_m00 = 0.;
  if (d_m00 > 1.) d_m00 = 1.;
  double d_p00 = 1.-d_m00;

  double d_0m0 = (y-ym)/dy - (double)(j);
  if (d_0m0 < 0.) d_0m0 = 0.;
  if (d_0m0 > 1.) d_0m0 = 1.;
  double d_0p0 = 1.-d_0m0;

  double d_00m = (z-zm)/dz - (double)(k);
  if (d_00m < 0.) d_00m = 0.;
  if (d_00m > 1.) d_00m = 1.;
  double d_00p = 1.-d_00m;

  // check if a point is a corner point
  if (d_m00 < eps && d_0m0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0m0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0p0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0p0 < eps && d_00m < eps) return f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0m0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0m0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)];
  if (d_m00 < eps && d_0p0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)];
  if (d_p00 < eps && d_0p0 < eps && d_00p < eps) return f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  double w_mmm = d_p00*d_0p0*d_00p;
  double w_pmm = d_m00*d_0p0*d_00p;
  double w_mpm = d_p00*d_0m0*d_00p;
  double w_ppm = d_m00*d_0m0*d_00p;
  double w_mmp = d_p00*d_0p0*d_00m;
  double w_pmp = d_m00*d_0p0*d_00m;
  double w_mpp = d_p00*d_0m0*d_00m;
  double w_ppp = d_m00*d_0m0*d_00m;

  double F =
      w_mmm*f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmm*f[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpm*f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppm*f[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)] +
      w_mmp*f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmp*f[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpp*f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppp*f[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  double Fxx =
      w_mmm*f_xx[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmm*f_xx[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpm*f_xx[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppm*f_xx[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)] +
      w_mmp*f_xx[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmp*f_xx[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpp*f_xx[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppp*f_xx[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  double Fyy =
      w_mmm*f_yy[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmm*f_yy[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpm*f_yy[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppm*f_yy[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)] +
      w_mmp*f_yy[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmp*f_yy[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpp*f_yy[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppp*f_yy[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  double Fzz =
      w_mmm*f_zz[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmm*f_zz[(k+0)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpm*f_zz[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppm*f_zz[(k+0)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)] +
      w_mmp*f_zz[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+0)] +
      w_pmp*f_zz[(k+1)*(nx+1)*(ny+1) + (j+0)*(nx+1) + (i+1)] +
      w_mpp*f_zz[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+0)] +
      w_ppp*f_zz[(k+1)*(nx+1)*(ny+1) + (j+1)*(nx+1) + (i+1)];

  F -= 0.5*(dx*dx*d_p00*d_m00*Fxx + dy*dy*d_0p0*d_0m0*Fyy + dz*dz*d_00p*d_00m*Fzz);

  return F;
}
