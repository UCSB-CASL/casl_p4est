#ifndef cube3_mls_l_H
#define cube3_mls_l_H

/* Splitting of a cube into tetrahedra
 *
 * all arrays should be in the order: 000, 100, 010, 110, 001, 101, 011, 111
 *                                      0,   1,   2,   3,   4,   5,   6,   7
 */

#define cube3_mls_l_KUHN
#ifdef cube3_mls_l_MIDDLECUT // Middle-cut triangulation
#define NTETS 5
// Tetrahedron #0
#define l3_t0p0 0
#define l3_t0p1 1
#define l3_t0p2 2
#define l3_t0p3 4
// Tetrahedron #1
#define l3_t1p0 1
#define l3_t1p1 2
#define l3_t1p2 3
#define l3_t1p3 7
// Tetrahedron #2
#define l3_t2p0 1
#define l3_t2p1 2
#define l3_t2p2 4
#define l3_t2p3 7
// Tetrahedron #3
#define l3_t3p0 1
#define l3_t3p1 4
#define l3_t3p2 5
#define l3_t3p3 7
// Tetrahedron #4
#define l3_t4p0 2
#define l3_t4p1 4
#define l3_t4p2 6
#define l3_t4p3 7
#endif

#ifdef cube3_mls_l_KUHN // Kuhn triangulation

const int num_tetrs_l = 6;
const double tp_l[num_tetrs_l][4] = { { 0, 1, 3, 7},
                                    { 0, 2, 3, 7},
                                    { 0, 1, 5, 7},
                                    { 0, 2, 6, 7},
                                    { 0, 4, 5, 7},
                                    { 0, 4, 6, 7} };

#define NTETS 6
// Tetrahedron #0
#define l3_t0p0 0
#define l3_t0p1 1
#define l3_t0p2 3
#define l3_t0p3 7
// Tetrahedron #1
#define l3_t1p0 0
#define l3_t1p1 2
#define l3_t1p2 3
#define l3_t1p3 7
// Tetrahedron #2
#define l3_t2p0 0
#define l3_t2p1 1
#define l3_t2p2 5
#define l3_t2p3 7
// Tetrahedron #3
#define l3_t3p0 0
#define l3_t3p1 2
#define l3_t3p2 6
#define l3_t3p3 7
// Tetrahedron #4
#define l3_t4p0 0
#define l3_t4p1 4
#define l3_t4p2 5
#define l3_t4p3 7
// Tetrahedron #5
#define l3_t5p0 0
#define l3_t5p1 4
#define l3_t5p2 6
#define l3_t5p3 7
#endif

#include <vector>
#include "simplex3_mls_l.h"

class cube3_mls_l_t
{
public:
  const static unsigned int n_nodes = 8;
  const static unsigned int n_nodes_simplex = 4;
  const static unsigned int n_nodes_dir = 2;

  double  x0, x1, y0, y1, z0, z1, diag;
  loc_t   loc;
  unsigned int num_of_lsfs;

  double lip;

  std::vector<simplex3_mls_l_t> simplex;


  cube3_mls_l_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1), lip(2.)
  {
    diag = sqrt(pow(x1-x0, 2.)+
                pow(y1-y0, 2.)+
                pow(z1-z0, 2.));
  }

  void construct_domain(std::vector<double> &phi_all, std::vector<action_t> &acn, std::vector<int> &clr);

  void quadrature_over_domain      (                              std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_interface   (int num,                      std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection(int num0, int num1,           std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_in_dir           (int dir,                      std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
};

#endif // cube3_mls_l_H
