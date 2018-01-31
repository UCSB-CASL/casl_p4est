#ifndef cube2_mls_q_H
#define cube2_mls_q_H

/* Values of LSFs and integrand should be in the following (z-) order
 *
 * 6--7--8
 * |  |  |
 * 3--4--5
 * |  |  |
 * 0--1--2
 */

/* Simplex 0
 *
 *     8
 *    /|
 *   4 5
 *  /  |
 * 0-1-2
 *
 */
#define t0p0 0
#define t0p1 2
#define t0p2 8
#define t0p3 1
#define t0p4 5
#define t0p5 4

/* Simplex 1
 *
 * 6-7-8
 * |  /
 * 3 4
 * |/
 * 0
 *
 */
//#define t1p0 0
//#define t1p1 8
//#define t1p2 6
//#define t1p3 4
//#define t1p4 7
//#define t1p5 3
#define t1p0 0
#define t1p1 6
#define t1p2 8
#define t1p3 3
#define t1p4 7
#define t1p5 4

#include "vector"
#include "simplex2_mls_q.h"

class cube2_mls_q_t
{
public:
  const static int n_nodes = 9;
  const static int n_nodes_simplex = 6;
  const static int n_nodes_dir = 3;

  double  x0, x1, y0, y1;
  loc_t   loc;
  int     num_of_lsfs;

  std::vector<simplex2_mls_q_t> simplex;

  cube2_mls_q_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1) {}

  void construct_domain(std::vector<double> &phi_all, std::vector<action_t> &acn, std::vector<int> &clr);

  void quadrature_over_domain      (                    std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface   (int num,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection(int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir           (int dir,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
};

#endif // cube2_mls_q_H
