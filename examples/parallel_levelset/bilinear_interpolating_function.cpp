#include "bilinear_interpolating_function.h"

namespace parallel{
BilinearInterpolatingFunction::BilinearInterpolatingFunction(const std::vector<double>&F,
                                                             p4est_t *p4est, p4est_nodes_t *nodes)
  : F_(F), p4est_(p4est), nodes_(nodes)
{}

double BilinearInterpolatingFunction::operator ()(double x, double y) const {
  double xy[] = {x, y};
  return linear_interpolation(F_, xy);
}
}
