#ifndef PROBEM_CASE_6_H
#define PROBEM_CASE_6_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class problem_case_6_t
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
        return .0;
      }
  } bc_coeff_0;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
        return .0;
      }
  } bc_coeff_0;
#endif

  problem_case_6_t()
  {
#ifdef P4_TO_P8
    double r0 = 0.5, xc0 = 0.08, yc0 = 0.01, zc0 = 0.03;

    domain0.set_params(r0, xc0, yc0, zc0, 0.1);
#else
    double r0 = 0.5, xc0 = 0.3, yc0 = -0.2;

    domain0.set_params(r0, xc0, yc0, 0.0, -1);
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

#endif // PROBEM_CASE_6_H
