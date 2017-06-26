#ifndef PROBEM_CASE_0_H
#define PROBEM_CASE_0_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class problem_case_0_t
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

  half_space_t domain0;
  half_space_t domain1;
  half_space_t domain2;
#ifdef P4_TO_P8
  half_space_t domain3;
#endif

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
//        return 0.0;
          return 1.0;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
//        return 0.0;
          return 0.5;
      }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
        return 0.0;
//        return 2.6;
          return x-y + (x+y)*(x+y);
      }
  } bc_coeff_2;

  class bc_coeff_3_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
        return 0.0;
//        return -1.1;
          return x+y;
      }
  } bc_coeff_3;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          return 1.0;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          return 0.0;
      }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
//        return 0.5;
          return x-y + (x+y)*(x+y);
      }
  } bc_coeff_2;
#endif

  problem_case_0_t()
  {

#ifdef P4_TO_P8
    double x0 =-0.86, y0 =-0.91, z0 =-0.83;
    double x1 = 0.88, y1 =-0.52, z1 = 0.63;
    double x2 = 0.67, y2 = 0.82, z2 =-0.87;
    double x3 =-0.93, y3 = 0.73, z3 = 0.85;

    domain0.set_params_points(x0, y0, z0, x2, y2, z2, x1, y1, z1);
    domain1.set_params_points(x1, y1, z1, x2, y2, z2, x3, y3, z3);
    domain2.set_params_points(x0, y0, z0, x3, y3, z3, x2, y2, z2);
    domain3.set_params_points(x0, y0, z0, x1, y1, z1, x3, y3, z3);
//    domain0.set_params(1.0, -0.4, -0.33, 0.9, 0.9, 0.9);
//    domain1.set_params(-0.31, 1.0, -0.29, 0.9, 0.9, 0.9);
//    domain2.set_params(-0.15, -0.19, 0.7, 0.9, 0.9, 0.9);
//    domain3.set_params(-1.0, -0.8, -0.7, -0.2, -0.3, -0.3);
#else
    double x0 =-0.74; double y0 =-0.89;
    double x1 = 0.83; double y1 =-0.11;
    double x2 =-0.37; double y2 = 0.87;
    domain0.set_params_points(x0, y0, x2, y2);
    domain1.set_params_points(x2, y2, x1, y1);
    domain2.set_params_points(x1, y1, x0, y0);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    phi_cf.push_back(&domain1.phi); action.push_back(INTERSECTION); color.push_back(1);
    phi_cf.push_back(&domain2.phi); action.push_back(INTERSECTION); color.push_back(2);
#ifdef P4_TO_P8
    phi_cf.push_back(&domain3.phi); action.push_back(INTERSECTION); color.push_back(3);
#endif

    phi_x_cf.push_back(&domain0.phi_x);
    phi_x_cf.push_back(&domain1.phi_x);
    phi_x_cf.push_back(&domain2.phi_x);

    phi_y_cf.push_back(&domain0.phi_y);
    phi_y_cf.push_back(&domain1.phi_y);
    phi_y_cf.push_back(&domain2.phi_y);

#ifdef P4_TO_P8
    phi_x_cf.push_back(&domain3.phi_x);
    phi_y_cf.push_back(&domain3.phi_y);

    phi_z_cf.push_back(&domain0.phi_z);
    phi_z_cf.push_back(&domain1.phi_z);
    phi_z_cf.push_back(&domain2.phi_z);
    phi_z_cf.push_back(&domain3.phi_z);
#endif

    bc_coeffs_cf.push_back(&bc_coeff_0);
    bc_coeffs_cf.push_back(&bc_coeff_1);
    bc_coeffs_cf.push_back(&bc_coeff_2);
#ifdef P4_TO_P8
    bc_coeffs_cf.push_back(&bc_coeff_3);
#endif

  }

};

#endif // PROBEM_CASE_0_H
