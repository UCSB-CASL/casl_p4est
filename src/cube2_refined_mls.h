#ifndef CUBE2_REFINED_MLS_H
#define CUBE2_REFINED_MLS_H

#include "cube2_mls.h"
#include "vector"

class cube2_refined_mls_t
{
public:
  double  x0, x1, y0, y1;
  loc_t   loc;
  int     num_non_trivial;
  int     n_cubes, n_nodes, nx, ny;
  int     v_mm, v_pm, v_mp, v_pp;

  std::vector<cube2_mls_t> cubes;

  cube2_refined_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1) {}

  void construct_domain(int nx_, int ny_, double *phi, std::vector<action_t> &action, std::vector<int> &color);

  double integrate_over_domain      (double *f);
  double integrate_over_interface   (double *f, int num);
  double integrate_over_intersection(double *f, int num0, int num1);
  double integrate_in_dir           (double *f, int dir);

  double integrate_over_colored_interface   (double *f, int num0, int num1);

  double measure_of_domain();
  double measure_of_interface(int num);
  double measure_of_colored_interface(int num0, int num1);
  double measure_in_dir(int dir);
};

#endif // CUBE2_REFINED_MLS_H
