#ifndef MY_P4EST_RBF_INTERPOLATING_FUNCITON_H
#define MY_P4EST_RBF_INTERPOLATING_FUNCITON_H

#include <vector>
#include <src/casl_types.h>
#include <src/Cholesky.h>

#ifdef P4_TO_P8
#include <src/point3.h>
#else
#include <src/point2.h>
#endif

#ifdef P4_TO_P8
class RBFInterpolatingFunciton: public CF_3
#else
class RBFInterpolatingFunciton: public CF_2
#endif
{
#ifdef P4_TO_P8
  typedef Point3 point_t;
#else
  typedef Point2 point_t;
#endif
  const std::vector<point_t>& p_;
  const std::vector<double>& f_;
  const CF_1& phi_;

  Cholesky chol;
  MatrixFull A;
  std::vector<double> w;

public:
  RBFInterpolatingFunciton(const std::vector<point_t>& p, const std::vector<double>& f, const CF_1& phi);

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif
};

#endif // MY_P4EST_RBF_INTERPOLATING_FUNCITON_H
