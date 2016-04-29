#ifndef GRID_INTERPOLATION3_H
#define GRID_INTERPOLATION3_H

#include <vector>
#include <cmath>
#include <stdexcept>

class grid_interpolation3_t
{
  int nx, ny, nz;
  double xm, xp, ym, yp, zm, zp;
  double dx, dy, dz;

public:

  grid_interpolation3_t(double xm = 0., double xp = 1.,
                        double ym = 0., double yp = 1.,
                        double zm = 0., double zp = 1.,
                        int nx = 1, int ny = 1, int nz = 1)
  {
    initialize(xm, xp, ym, yp, zm, zp, nx, ny, nz);
  }

  void initialize(double xm_ = 0., double xp_ = 1.,
                  double ym_ = 0., double yp_ = 1.,
                  double zm_ = 0., double zp_ = 1.,
                  int nx_ = 1, int ny_ = 1, int nz_ = 1)
  {
    xm = xm_; xp = xp_;
    ym = ym_; yp = yp_;
    zm = zm_; zp = zp_;
    nx = nx_; ny = ny_; nz = nz_;

    dx = (xp-xm)/(double)(nx);
    dy = (yp-ym)/(double)(ny);
    dz = (zp-zm)/(double)(nz);
  }

  double linear(double *f, double x, double y, double z);
  double quadratic(double *f, double *f_xx, double *f_yy, double *f_zz, double x, double y, double z);

  double linear(std::vector<double> &f, double x, double y, double z)
  {
    return linear(f.data(), x, y, z);
  }

  double quadratic(std::vector<double> &f,
                   std::vector<double> &f_xx, std::vector<double> &f_yy, std::vector<double> &f_zz,
                   double x, double y, double z)
  {
    return quadratic(f.data(), f_xx.data(), f_yy.data(), f_zz.data(), x, y, z);
  }

  int total_nodes() {return (nx+1)*(ny+1)*(nz+1);}

};

#endif // GRID_INTERPOLATION3_H
