#ifndef PROBEM_CASE_1_H
#define PROBEM_CASE_1_H
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

class problem_case_1_t
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
//        return 0.;
          return 1.0;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
//        return 0.;
          return sin(x+y)*cos(x-y)*log(z+4.);
      }
  } bc_coeff_1;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
//                return 0.;
          return 1.;
//          return 0.5;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
//                return 0.;
//          return 1.;
          return sin(x+y)*cos(x-y);
      }
  } bc_coeff_1;
#endif

  problem_case_1_t()
  {
#ifdef P4_TO_P8
    double r0 = 0.71, xc0 = 0.22, yc0 = 0.17, zc0 = 0.21;
    double r1 = 0.63, xc1 =-0.19, yc1 =-0.19, zc1 =-0.23;

//    double r1 = 0.71, xc1 = 0.22, yc1 = 0.17, zc1 = 0.21;
//    double r0 = 0.63, xc0 =-0.19, yc0 =-0.19, zc0 =-0.23;

    domain0.set_params(r0, xc0, yc0, zc0);
    domain1.set_params(r1, xc1, yc1, zc1);
#else
    double r0 = 0.77, xc0 = 0.13, yc0 = 0.21;
//    double r0 = 0.37, xc0 = 0.43, yc0 = 0.41;
    double r1 = 0.49, xc1 =-0.33, yc1 =-0.37;

//    double r1 = 0.77, xc1 = 0.13, yc1 = 0.21;
//    double r0 = 0.53, xc0 =-0.41, yc0 =-0.37;

    domain0.set_params(r0, xc0, yc0);
    domain1.set_params(r1, xc1, yc1);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    phi_cf.push_back(&domain1.phi); action.push_back(ADDITION); color.push_back(1);

    phi_x_cf.push_back(&domain0.phi_x);
    phi_x_cf.push_back(&domain1.phi_x);

    phi_y_cf.push_back(&domain0.phi_y);
    phi_y_cf.push_back(&domain1.phi_y);

#ifdef P4_TO_P8
    phi_z_cf.push_back(&domain0.phi_z);
    phi_z_cf.push_back(&domain1.phi_z);
#endif

    bc_coeffs_cf.push_back(&bc_coeff_0);
    bc_coeffs_cf.push_back(&bc_coeff_1);

  }

};

#endif // PROBEM_CASE_1_H
