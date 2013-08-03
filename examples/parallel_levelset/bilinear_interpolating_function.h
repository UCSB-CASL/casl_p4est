#ifndef BILINEAR_INTERPOLATING_FUNCTION_H
#define BILINEAR_INTERPOLATING_FUNCTION_H

#include <vector>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>

namespace parallel{
class BilinearInterpolatingFunction: public CF_2
{
  const double* F_;
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;

  double linear_interpolation(const double xy[]) const;

public:
  BilinearInterpolatingFunction(const double *F, p4est_t *p4est, p4est_nodes_t *nodes);

  double operator()(double x, double y) const;
};
} // namepace parallel

#endif // BILINEAR_INTERPOLATING_FUNCTION_H
