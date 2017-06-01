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

#include "vector"
#include "simplex2_mls.h"
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

class cube2_mls_t
{
public:
  const static int n_nodes = 4;
  const static int n_nodes_simplex = 3;

  double  x0, x1, y0, y1;
  loc_t   loc;
  int     num_of_lsfs;
  bool    use_linear;

  std::vector<simplex2_mls_t> simplex;

  cube2_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), use_linear(true) {}

  void construct_domain(std::vector<CF_2 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  double integrate_over_domain            (CF_2 &f);
  double integrate_over_interface         (CF_2 &f, int num);
  double integrate_over_intersection      (CF_2 &f, int num0, int num1);
  double integrate_in_dir                 (CF_2 &f, int dir);
//  double integrate_in_non_cart_dir        (double *f, int dir);
  double integrate_over_colored_interface (CF_2 &f, int num0, int num1);

//  double measure_of_domain            ();
//  double measure_of_interface         (int num);
//  double measure_of_intersection      (int num0, int num1);
//  double measure_of_colored_interface (int num0, int num1);
//  double measure_in_dir               (int dir);

  void set_use_linear(bool val) { use_linear = val; }

//  double interpolate_linear(double *f, double x, double y);
//  double interpolate_quadratic(double *f, double *fxx, double *fyy, double x, double y);
};

#endif // CUBE2_MLS_H
