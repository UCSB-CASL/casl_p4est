#ifndef cube2_mls_l_H
#define cube2_mls_l_H

/* Splitting of a cube into simplices
 *
 * all arrays should be in the order: 00, 10, 01, 11
 *                                     0,  1,  2,  3
 */

// Simplex #0
#define l2_t0p0 0
#define l2_t0p1 1
#define l2_t0p2 3
// Simplex #1
#define l2_t1p0 0
#define l2_t1p1 2
#define l2_t1p2 3

#include "vector"
#include "simplex2_mls_l.h"

class cube2_mls_l_t
{
public:
  const static unsigned int n_nodes = 4;
  const static unsigned int n_nodes_simplex = 3;
  const static unsigned int n_nodes_dir = 2;

  double       x0, x1, y0, y1;
  loc_t        loc;
  unsigned int num_of_lsfs;
  bool         use_linear;

  std::vector<simplex2_mls_l_t> simplex;

  cube2_mls_l_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), use_linear(true) {}

  void construct_domain(const std::vector<double> &phi_all, const std::vector<action_t> &acn, const std::vector<int> &clr);

  void quadrature_over_domain      (                    std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface   (int num,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir           (int dir,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);

  void set_use_linear(bool val) { use_linear = val; }
};

#endif // cube2_mls_l_H
