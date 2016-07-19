#ifndef GRID_INTERPOLATION2_H
#define GRID_INTERPOLATION2_H

#include <vector>
#include <cmath>
#include <stdexcept>

class grid_interpolation2_t
{
  int nx, ny;
  double xm, ym, xp, yp;
  double dx, dy;

  double eps;

public:

  grid_interpolation2_t(double xm = 0., double xp = 1., double ym = 0., double yp = 1., int nx = 1, int ny = 1) : eps(1.0E-15)
  {
    initialize(xm, xp, ym, yp, nx, ny);
  }

  void initialize(double xm_ = 0., double xp_ = 1., double ym_ = 0., double yp_ = 1., int nx_ = 1, int ny_ = 1)
  {
    xm = xm_; xp = xp_;
    ym = ym_; yp = yp_;
    nx = nx_; ny = ny_;

    dx = (xp-xm)/(double)(nx);
    dy = (yp-ym)/(double)(ny);
  }

  double linear(double *f, double x, double y);
  double quadratic(double *f, double *f_xx, double *f_yy, double x, double y);

  inline double linear(std::vector<double> &f, double x, double y)
  {
    return linear(f.data(), x, y);
  }

  inline double quadratic(std::vector<double> &f, std::vector<double> &f_xx, std::vector<double> &f_yy, double x, double y)
  {
    return quadratic(f.data(), f_xx.data(), f_yy.data(), x, y);
  }

  int total_nodes() {return (nx+1)*(ny+1);}

  void set_eps(double eps_in) {eps = eps_in;}
};

#endif // GRID_INTERPOLATION2_H
