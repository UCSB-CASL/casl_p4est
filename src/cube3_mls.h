#ifndef CUBE3_MLS_H
#define CUBE3_MLS_H

/* Splitting of a cube into tetrahedra
 *
 * all arrays should be in the order: 000, 100, 010, 110, 001, 101, 011, 111
 *                                      0,   1,   2,   3,   4,   5,   6,   7
 */

#define CUBE3_MLS_KUHN
#ifdef CUBE3_MLS_MIDDLECUT // Middle-cut triangulation
#define NTETS 5
// Tetrahedron #0
#define t0p0 0
#define t0p1 1
#define t0p2 2
#define t0p3 4
// Tetrahedron #1
#define t1p0 1
#define t1p1 2
#define t1p2 3
#define t1p3 7
// Tetrahedron #2
#define t2p0 1
#define t2p1 2
#define t2p2 4
#define t2p3 7
// Tetrahedron #3
#define t3p0 1
#define t3p1 4
#define t3p2 5
#define t3p3 7
// Tetrahedron #4
#define t4p0 2
#define t4p1 4
#define t4p2 6
#define t4p3 7
#endif

#ifdef CUBE3_MLS_KUHN // Kuhn triangulation
#define NTETS 6
// Tetrahedron #0
#define t0p0 0
#define t0p1 1
#define t0p2 3
#define t0p3 7
// Tetrahedron #1
#define t1p0 0
#define t1p1 2
#define t1p2 3
#define t1p3 7
// Tetrahedron #2
#define t2p0 0
#define t2p1 1
#define t2p2 5
#define t2p3 7
// Tetrahedron #3
#define t3p0 0
#define t3p1 2
#define t3p2 6
#define t3p3 7
// Tetrahedron #4
#define t4p0 0
#define t4p1 4
#define t4p2 5
#define t4p3 7
// Tetrahedron #5
#define t5p0 0
#define t5p1 4
#define t5p2 6
#define t5p3 7
#endif

#include <vector>
#include "simplex3_mls.h"
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

class cube3_mls_t
{
public:
  const static int n_nodes = 8;
  const static int n_nodes_simplex = 4;
  const static int n_nodes_dir = 2;
  const double lip = 1.;

  double  x0, x1, y0, y1, z0, z1, diag;
  loc_t   loc;
  int     num_of_lsfs;

  std::vector<simplex3_mls_t> simplex;


  cube3_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1)
  {
    diag = sqrt(SQR(x1-x0)+SQR(y1-y0)+SQR(z1-z0));
  }

  void construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  double integrate_over_domain            (CF_3& f);
  double integrate_over_interface         (CF_3& f, int num);
  double integrate_over_colored_interface (CF_3& f, int num0, int num1);
  double integrate_over_intersection      (CF_3& f, int num0, int num1);
  double integrate_over_intersection      (CF_3& f, int num0, int num1, int num2);
  double integrate_in_dir                 (CF_3& f, int dir);

//  double measure_of_domain            ();
//  double measure_of_interface         (int num);
//  double measure_of_intersection      (int num0, int num1);
//  double measure_of_colored_interface (int num0, int num1);
//  double measure_in_dir               (int dir);

//  void interpolate_to_cube(double *in, double *out);

//  void set_use_linear(bool val) { use_linear = val; }

//  double interpolate_linear(double *f, double x, double y, double z);
//  double interpolate_quadratic(double *f, double *fxx, double *fyy, double *fzz, double x, double y, double z);

};

#endif // CUBE3_MLS_H
