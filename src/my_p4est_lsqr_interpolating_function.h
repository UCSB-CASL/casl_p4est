#ifndef MY_P4EST_LSQR_INTERPOLATING_FUNCTION_H
#define MY_P4EST_LSQR_INTERPOLATING_FUNCTION_H

#include <vector>
#include <src/casl_types.h>
#include <src/Cholesky.h>

#ifdef P4_TO_P8
#include <src/point3.h>
#else
#include <src/point2.h>
#endif

enum LSQR_method{
  linear_LSQR,
  quadrantic_LSQR
};

#ifdef P4_TO_P8
class LSQRInterpolatingFunction: public CF_3
#else
class LSQRInterpolatingFunction: public CF_2
#endif
{
#ifdef P4_TO_P8
  typedef Point3 point_t;
#else
  typedef Point2 point_t;
#endif
  LSQR_method method_;

  std::vector<double> w;
public:
  LSQRInterpolatingFunction(const std::vector<point_t>& p, const std::vector<double>& f, LSQR_method method = linear_LSQR);

  inline const std::vector<double>& get_weights() const {return w;}
  inline size_t get_size() const {return w.size();}
  inline double operator[](size_t i) const {return w[i];}
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif
};

#endif // MY_P4EST_LSQR_INTERPOLATING_FUNCTION_H
