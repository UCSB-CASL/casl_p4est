#ifndef CUBE3_REFINED_MLS_H
#define CUBE3_REFINED_MLS_H

#include "cube3_mls.h"
#include "vector"

class cube3_refined_mls_t
{
public:
  double  x0, x1, y0, y1, z0, z1;
  loc_t   loc;
  int     num_non_trivial;
  int     n_cubes, n_nodes, nx, ny, nz;
  int     v_mmm, v_pmm, v_mpm, v_ppm;
  int     v_mmp, v_pmp, v_mpp, v_ppp;

  std::vector<cube3_mls_t> cubes;

  cube3_refined_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 0.)
    : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {}

  void construct_domain(int nx_, int ny_, int nz_, double *phi, std::vector<action_t> &action, std::vector<int> &color);

  double integrate_over_domain      (double *f);
  double integrate_over_interface   (double *f, int num);
  double integrate_over_colored_interface   (double *f, int num0, int num1);

  double measure_of_domain();
  double measure_of_interface(int num);
  double measure_of_colored_interface(int num0, int num1);
  double measure_in_dir(int dir);
};

#endif // CUBE3_REFINED_MLS_H
