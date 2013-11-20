#include "my_p4est_rbf_interpolating_funciton.h"

RBFInterpolatingFunciton::RBFInterpolatingFunciton(const std::vector<point_t> &p, const std::vector<double> &f, const CF_1 &phi)
  : p_(p), f_(f), phi_(phi),
    A(p.size(), p.size()),
    w(p.size())
{
  // Compute the RBF matrix
  for (size_t i = 0; i<p.size(); i++)
    for (size_t j = 0; j<p.size(); j++)
      A.set_Value(i, j, phi(p[i].distance(p[j])));

  for (size_t i = 0; i<w.size(); i++)
    std::cout << p_[i];
  std::cout << std::endl;

  // solve RBF system
  if (!chol.solve(A, f, w))
    throw std::runtime_error("[Error] could not invert the linear system for RBF interpolation");
}

#ifdef P4_TO_P8
double RBFInterpolatingFunciton::operator ()(double x, double y, double z) const
#else
double RBFInterpolatingFunciton::operator ()(double x, double y) const
#endif
{
  point_t px;
  px.x = x; px.y = y;
#ifdef P4_TO_P8
  px.z = z;
#endif



//  for (size_t i = 0; i<w.size(); i++)
//    std::cout << w[i] << " " ;
//  std::cout << std::endl;

  double res = 0;
  for (size_t i = 0; i<w.size(); i++)
    res += w[i]*phi_(p_[i].distance(px));

  return res;
}
