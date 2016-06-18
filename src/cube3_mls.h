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

class cube3_mls_t
{
public:
  double  x0, x1, y0, y1, z0, z1;
  loc_t   loc;
  int     num_non_trivial;

  std::vector<simplex3_mls_t> simplex;

  std::vector< std::vector<double> > *phi;
  std::vector< std::vector<double> > *phi_xx;
  std::vector< std::vector<double> > *phi_yy;
  std::vector< std::vector<double> > *phi_zz;

  std::vector<action_t> *action;
  std::vector<int>      *color;

  cube3_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 1.)
    : x0(x0), y0(y0), z0(z0), x1(x1), y1(y1), z1(z1) {}


  void set_phi(std::vector< std::vector<double> > &phi_,
               std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = NULL; phi_yy = NULL; phi_zz = NULL;
    action  = &acn_;
    color   = &clr_;
  }

  void set_phi(std::vector< std::vector<double> > &phi_,
               std::vector< std::vector<double> > &phi_xx_,
               std::vector< std::vector<double> > &phi_yy_,
               std::vector< std::vector<double> > &phi_zz_,
               std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_; phi_zz = &phi_zz_;
    action  = &acn_;
    color   = &clr_;
  }

  void construct_domain();

  double integrate_over_domain            (double *f);
  double integrate_over_interface         (double *f, int num);
  double integrate_over_colored_interface (double *f, int num0, int num1);
  double integrate_over_intersection      (double *f, int num0, int num1);
  double integrate_over_intersection      (double *f, int num0, int num1, int num2);
  double integrate_in_dir                 (double *f, int dir);

  double interpolate_linear(double *f, double x, double y, double z);
  double interpolate_quadratic(double *f, double *fxx, double *fyy, double *fzz, double x, double y, double z);

};

#endif // CUBE3_MLS_H
