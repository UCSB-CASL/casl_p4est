#ifndef PROBEM_CASE_9_H
#define PROBEM_CASE_9_H
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

class problem_case_9_t
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
          return 0.0;
      }
  } bc_coeff_1;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          return -1.0;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
          return -1.0;
      }
  } bc_coeff_1;
#endif

  problem_case_9_t()
  {

#ifdef P4_TO_P8
    double x0 =-0.86, y0 =-0.91, z0 =-0.83;
    double x1 = 0.88, y1 =-0.52, z1 = 0.63;
    double x2 = 0.67, y2 = 0.82, z2 =-0.87;
    double x3 =-0.93, y3 = 0.73, z3 = 0.85;

    domain0.set_params_points(x0, y0, z0, x2, y2, z2, x1, y1, z1);
    domain1.set_params_points(x1, y1, z1, x2, y2, z2, x3, y3, z3);
#else
    double x0 =-0.740001; double y0 =-0.89;
    double x1 = 0.83; double y1 = 10.87;
    double x2 =-0.740001; double y2 = 10.87;
    domain0.set_params_points(x0, y0, x2, y2);
    domain1.set_params_points(x2, y2, x1, y1);
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

#endif // PROBEM_CASE_9_H
