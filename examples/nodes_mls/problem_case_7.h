#ifndef PROBEM_CASE_7_H
#define PROBEM_CASE_7_H
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

class problem_case_7_t
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
  std::vector<mls_opn_t> action;
  std::vector<int> color;

  flower_shaped_domain_t domain0;
  flower_shaped_domain_t domain1;
  flower_shaped_domain_t domain2;

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
          return 1.0 + sin(x)*cos(y)*exp(z);
//          return 1.0;
      }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
//        return 0.0;
          return exp(x+y+z);
//          return 1.0;
      }
  } bc_coeff_2;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
        (void) x; (void) y;
        return 1.0;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
//        return 1.0;
        return 1.0 + sin(x)*cos(y);
      }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_2 {
  public:
      double operator()(double x, double y) const
      {
//        return 1.0;
        return exp(x+y);
      }
  } bc_coeff_2;
#endif

  problem_case_7_t()
  {
#ifdef P4_TO_P8
    double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, zc0 = 0.19, nx0 = 1.0, ny0 = 1.0, nz0 = 1.0, theta0 = 0.3*PI, beta0 = 0.08, inside0 = 1;
    double r1 = 0.66, xc1 =-0.21, yc1 =-0.23, zc1 =-0.17, nx1 = 1.0, ny1 = 1.0, nz1 = 1.0, theta1 =-0.3*PI, beta1 =-0.08, inside1 = 1;
    double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, zc2 = 0.03, nx2 =-1.0, ny2 = 1.0, nz2 = 0.0, theta2 =-0.2*PI, beta2 =-0.08, inside2 =-1;

//    double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, zc0 = 0.19, nx0 = 1.0, ny0 = 1.0, nz0 = 1.0, theta0 = 0.3*PI, beta0 = 0.0, inside0 = 1;
//    double r1 = -0.66, xc1 =-0.21, yc1 =-0.23, zc1 =-0.17, nx1 = 1.0, ny1 = 1.0, nz1 = 1.0, theta1 =-0.3*PI, beta1 =-0.0, inside1 = 1;
//    double r2 = 0.39, xc2 = 0.45, yc2 =-0.53, zc2 = 0.03, nx2 =-1.0, ny2 = 1.0, nz2 = 0.0, theta2 =-0.2*PI, beta2 =-0.0, inside2 =-1;

    domain0.set_params(r0, xc0, yc0, zc0, beta0, inside0, nx0, ny0, nz0, theta0);
    domain1.set_params(r1, xc1, yc1, zc1, beta1, inside1, nx1, ny1, nz1, theta1);
    domain2.set_params(r2, xc2, yc2, zc2, beta2, inside2, nx2, ny2, nz2, theta2);
#else
    double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, theta0 = 0.1*PI, beta0 = 0.08, inside0 = 1;
    double r1 = 0.66, xc1 =-0.14, yc1 =-0.21, theta1 =-0.2*PI, beta1 =-0.08, inside1 = 1;
    double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, theta2 = 0.2*PI, beta2 =-0.08, inside2 =-1;

    domain0.set_params(r0, xc0, yc0, beta0, inside0, theta0);
    domain1.set_params(r1, xc1, yc1, beta1, inside1, theta1);
    domain2.set_params(r2, xc2, yc2, beta2, inside2, theta2);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(MLS_INTERSECTION); color.push_back(0);
    phi_cf.push_back(&domain1.phi); action.push_back(MLS_ADDITION); color.push_back(1);
    phi_cf.push_back(&domain2.phi); action.push_back(MLS_INTERSECTION); color.push_back(2);

    phi_x_cf.push_back(&domain0.phi_x);
    phi_x_cf.push_back(&domain1.phi_x);
    phi_x_cf.push_back(&domain2.phi_x);

    phi_y_cf.push_back(&domain0.phi_y);
    phi_y_cf.push_back(&domain1.phi_y);
    phi_y_cf.push_back(&domain2.phi_y);

#ifdef P4_TO_P8
    phi_z_cf.push_back(&domain0.phi_z);
    phi_z_cf.push_back(&domain1.phi_z);
    phi_z_cf.push_back(&domain2.phi_z);
#endif

    bc_coeffs_cf.push_back(&bc_coeff_0);
    bc_coeffs_cf.push_back(&bc_coeff_1);
    bc_coeffs_cf.push_back(&bc_coeff_2);

  }

};

#endif // PROBEM_CASE_7_H
