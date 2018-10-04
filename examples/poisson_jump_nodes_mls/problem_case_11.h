#ifndef PROBEM_CASE_11_H
#define PROBEM_CASE_11_H
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

// corner point
namespace p11
{
double x0 = .03, y0 = -.04;
// sector angle
double alpha = 1.25;
// rotation
double phase = 0.1*PI;
// radius
double R0 = 0.85;

double T_disc = -.5*(2.-alpha)*PI;

double k = 2.;
}

class problem_case_11_t
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
          return 0.;
      }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
          return 0.;
      }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_3 {
  public:
      double operator()(double x, double y, double z) const
      {
          return 1.;
      }
  } bc_coeff_2;
#else
  class bc_coeff_0_t : public CF_2 {
  public:
    double operator()(double x, double y) const
    {
      (void) x; (void) y;
      return 1.;
    }
  } bc_coeff_0;

  class bc_coeff_1_t : public CF_2 {
  public:
    double operator()(double x, double y) const
    {
      (void) x; (void) y;
      return 1.;
    }
  } bc_coeff_1;

  class bc_coeff_2_t : public CF_2 {
  public:
    double operator()(double x, double y) const
    {
      (void) x; (void) y;
      return 1.;
    }
  } bc_coeff_2;
#endif

  problem_case_11_t()
  {
#ifdef P4_TO_P8
#else
    domain0.set_params_points(p11::x0 + p11::R0*cos(p11::phase), p11::y0 + p11::R0*sin(p11::phase), p11::x0, p11::y0);
    domain1.set_params_points(p11::x0, p11::y0, p11::x0+p11::R0*cos(p11::alpha*PI+p11::phase), p11::y0+p11::R0*sin(p11::alpha*PI+p11::phase));
    domain2.set_params(p11::R0, p11::x0, p11::y0, 0, 1, 0);
#endif

    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    phi_cf.push_back(&domain1.phi); action.push_back(ADDITION); color.push_back(1);
    phi_cf.push_back(&domain2.phi); action.push_back(INTERSECTION); color.push_back(2);

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

#endif // PROBEM_CASE_11_H
