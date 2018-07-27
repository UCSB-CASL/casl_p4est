#ifndef PROBEM_CASE_5_H
#define PROBEM_CASE_5_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_shapes.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_shapes.h>
#endif

class problem_case_5_t
{
public:

  // Geometry
#ifdef P4_TO_P8
  std::vector<CF_3 *> phi_cf;
  std::vector<CF_3 *> phi_x_cf;
  std::vector<CF_3 *> phi_y_cf;
  std::vector<CF_3 *> phi_z_cf;
#else
  std::vector<CF_2 *> phi_cf;
  std::vector<CF_2 *> phi_x_cf;
  std::vector<CF_2 *> phi_y_cf;
#endif
  std::vector<action_t> action;
  std::vector<int> color;

  flower_shaped_domain_t domain0;
  flower_shaped_domain_t domain1;

  // Robin coefficients
#ifdef P4_TO_P8
  std::vector<CF_3 *> bc_coeffs_cf;
#else
  std::vector<CF_2 *> bc_coeffs_cf;
#endif

#ifdef P4_TO_P8
  class bc_coeff_0_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
          double r1 = 0.83, xc1 =-0.51, yc1 =-0.46, zc1 =-0.63;
          double phi = sqrt(pow(x-xc1, 2.)+pow(y-yc1, 2.)+pow(z-zc1, 2.)) - r1;
          if (phi <= 0) return -1;
          else          return 1;
          if (phi <= 0) return 1.0 + sin(x+z)*cos(y+z);
          else          return exp(x+y+z);
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
          return exp(x+y+z);
      }
  } bc_coeff_1;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          double r1 = 0.63, xc1 =-0.42, yc1 =-0.57;
          double phi = sqrt(pow(x-xc1, 2.)+pow(y-yc1, 2.)) - r1;

          if (phi <= 0) return -1;
          else          return 1;

//          if (phi <= 0) return 1.0 + sin(x)*cos(y);
//          else          return exp(x+y);
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          return exp(x+y);
      }
  } bc_coeff_1;
#endif

  problem_case_5_t()
  {
#ifdef P4_TO_P8
    double r0 = 0.76, xc0 = 0.08, yc0 = 0.01, zc0 = 0.03;
    double r1 = 0.83, xc1 =-0.51, yc1 =-0.46, zc1 =-0.63;

    domain0.set_params(r0, xc0, yc0, zc0, 0.1);
    domain1.set_params(r1, xc1, yc1, zc1);
#else
    double r0 = 0.62, xc0 = 0.03, yc0 = 0.02;

    domain0.set_params(r0, xc0, yc0, 0.2);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);

    phi_x_cf.push_back(&domain0.phi_x);

    phi_y_cf.push_back(&domain0.phi_y);

#ifdef P4_TO_P8
    phi_z_cf.push_back(&domain0.phi_z);
#endif

    bc_coeffs_cf.push_back(&bc_coeff_0);

  }

};

#endif // PROBEM_CASE_5_H
