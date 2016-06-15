#ifndef CUBE2_MLS_H
#define CUBE2_MLS_H

/* Splitting of a cube into simplices
 *
 * all arrays should be in the order: 00, 10, 01, 11
 *                                     0,  1,  2,  3
 */

// Simplex #0
#define t0p0 0
#define t0p1 1
#define t0p2 3
// Simplex #1
#define t1p0 0
#define t1p1 2
#define t1p2 3

#include "simplex2_mls.h"
#include "vector"

class cube2_mls_t
{
public:
  double  x0, x1, y0, y1;
  loc_t   loc;
  int     num_non_trivial;

  std::vector<simplex2_mls_t> simplex;

  cube2_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1),
      phi(NULL), phi_x(NULL), phi_y(NULL), phi_xx(NULL), phi_yy(NULL) {}

  void construct_domain(std::vector<action_t> &action, std::vector<int> &color);

  std::vector<double> *phi;
  std::vector<double> *phi_x;
  std::vector<double> *phi_y;
  std::vector<double> *phi_xx;
  std::vector<double> *phi_yy;

  void set_phi    (std::vector<double> &phi_)    {phi = &phi_;}
  void set_phi_d  (std::vector<double> &phi_x_,
                   std::vector<double> &phi_y_)  {phi_x = &phi_x_;
                                                  phi_y = &phi_y_;}
  void set_phi_dd (std::vector<double> &phi_xx_,
                   std::vector<double> &phi_yy_) {phi_xx = &phi_xx_;
                                                  phi_yy = &phi_yy_;}

  double integrate_over_domain            (double *f);
  double integrate_over_interface         (double *f, int num);
  double integrate_over_intersection      (double *f, int num0, int num1);
  double integrate_in_dir                 (double *f, int dir);
  double integrate_in_non_cart_dir        (double *f, int dir);
  double integrate_over_colored_interface (double *f, int num0, int num1);

  double interpolate_linear(double *f, double x, double y);
  double interpolate_quadratic(double *f, double *fxx, double *fyy, double x, double y);
};

#endif // CUBE2_MLS_H
