#include "my_p4est_lsqr_interpolating_function.h"

#ifdef P4_TO_P8
#define LSQR_NUM_WEIGHTS_LINEAR 4
#define LSQR_QUADRATIC_NUM_WEIGHTS 10
#else
#define LSQR_NUM_WEIGHTS_LINEAR 3
#define LSQR_NUM_WEIGHTS_QUADRATIC 6
#endif

LSQRInterpolatingFunction::LSQRInterpolatingFunction(const std::vector<point_t> &p, const std::vector<double> &f, LSQR_method method)
  : method_(method)
{
#ifdef CASL_THROWS
  if (p.size() != f.size())
    throw std::invalid_argument ("[ERROR]: points and values should have the same size");
  if (p.size() < LSQR_NUM_WEIGHTS_LINEAR)
    throw std::invalid_argument ("[ERROR] not enough points for linear LSQR interpolation");
  if (p.size() < LSQR_NUM_WEIGHTS_QUADRATIC && method_ == quadrantic_LSQR)
    throw std::invalid_argument ("[ERROR] not enough points for quadratic LSQR interpolation");
#endif

  DenseMatrix X;
  std::vector<double> b;

  if (method == linear_LSQR){
    b.resize(LSQR_NUM_WEIGHTS_LINEAR, 0);
    X.resize(f.size(), LSQR_NUM_WEIGHTS_LINEAR);
    for (size_t i = 0; i<f.size(); i++){
#ifdef P4_TO_P8
      double xyz [] = {1.0, p[i].x, p[i].y, p[i].z};
#else
      double xyz [] = {1.0, p[i].x, p[i].y};
#endif
      for (int j = 0; j<LSQR_NUM_WEIGHTS_LINEAR; j++){
        b[j] += f[i]*xyz[j];
        X.set_Value(i, j, xyz[j]);
      }
    }
  } else if (method == quadrantic_LSQR) {
    b.resize(LSQR_NUM_WEIGHTS_QUADRATIC, 0);
    X.resize(f.size(), LSQR_NUM_WEIGHTS_QUADRATIC);
    for (size_t i = 0; i<f.size(); i++){
#ifdef P4_TO_P8
      double xyz [] = {1.0, p[i].x, p[i].y, p[i].z,
                       p[i].x*p[i].y, p[i].y*p[i].z, p[i].z*p[i].x,
                       p[i].x*p[i].x, p[i].y*p[i].y, p[i].z*p[i].z };
#else
      double xyz [] = {1.0, p[i].x, p[i].y,
                       p[i].x*p[i].y, p[i].x*p[i].x, p[i].y*p[i].y};
#endif
      for (int j = 0; j<LSQR_NUM_WEIGHTS_QUADRATIC; j++){
        b[j] += f[i]*xyz[j];
        X.set_Value(i, j, xyz[j]);
      }
    }
  } else {
    throw std::invalid_argument("[ERROR] Unkown LSQR interpolation method");
  }

  DenseMatrix A;
  X.MtM_Product(A);

  Cholesky chol;
  w.resize(b.size());
  if (!chol.solve(A, b, w))
    throw std::runtime_error("[CASL_ERROR]: Could not invert the LSQR matrix");
}

#ifdef P4_TO_P8
double LSQRInterpolatingFunction::operator ()(double x, double y, double z) const
#else
double LSQRInterpolatingFunction::operator ()(double x, double y) const
#endif
{
  double res = 0;

  if (method_ == linear_LSQR) {
#ifdef P4_TO_P8
      double xyz [] = {1.0, x, y, z};
#else
      double xyz [] = {1.0, x, y};
#endif
      for (int i=0; i<LSQR_NUM_WEIGHTS_LINEAR; i++)
        res += w[i]*xyz[i];
  } else {
#ifdef P4_TO_P8
      double xyz [] = {1.0, x, y, z,
                       x*y, y*z, z*x,
                       x*x, y*y, z*z };
#else
      double xyz [] = {1.0, x, y,
                       x*y, x*x, y*y};
#endif
      for (int i=0; i<LSQR_NUM_WEIGHTS_QUADRATIC; i++)
        res += w[i]*xyz[i];
  }

  return res;
}
