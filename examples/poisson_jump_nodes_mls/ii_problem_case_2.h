#ifndef II_PROBEM_CASE_2_H
#define II_PROBEM_CASE_2_H
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

class ii_problem_case_2_t
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
      return cos(x+y)*sin(x-y)*exp(z);
    }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_3 {
  public:
    double operator()(double x, double y, double z) const
    {
      return sin(x+y)*cos(x-y)*log(z+4.);
    }
  } bc_coeff_1;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
    double operator()(double x, double y) const
    {
//      return 0.;
      return sin(x-y)*cos(x+y);
    }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
    double operator()(double x, double y) const
    {
//      return 0.;
      return sin(x+y)*cos(x-y);
    }
  } bc_coeff_1;
#endif

  ii_problem_case_2_t()
  {
#ifdef P4_TO_P8
    double r0 = 0.86, xc0 = 0.08, yc0 = 0.11, zc0 = 0.03;
    double r1 = 0.83, xc1 =-0.51, yc1 =-0.46, zc1 =-0.63;

    domain0.set_params(r0, xc0, yc0, zc0);
    domain1.set_params(r1, xc1, yc1, zc1, 0, -1);
#else
    double r0 = 0.84, xc0 = 0.03, yc0 = 0.04;
    double r1 = 0.63, xc1 =-0.42, yc1 =-0.37;

    domain0.set_params(r0, xc0, yc0);
    domain1.set_params(r1, xc1, yc1, 0, -1);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    phi_cf.push_back(&domain1.phi); action.push_back(INTERSECTION); color.push_back(1);

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

#endif // II_PROBEM_CASE_2_H
