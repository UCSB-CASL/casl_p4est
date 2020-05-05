#ifndef SCALAR_TESTS_H
#define SCALAR_TESTS_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

struct domain_t {
  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  int periodicity[P4EST_DIM];
  double length() const { return xyz_max[0] - xyz_min[0]; }
  double height() const { return xyz_max[1] - xyz_min[1]; }
#ifdef P4_TO_P8
  double width() const  { return xyz_max[2] - xyz_min[2]; }
#endif
};

class test_case_for_scalar_jump_problem_t
{
protected:
  domain_t domain;
  std::string description;
  double mu_m, mu_p;
  const CF_DIM *level_set;
  double solution_integral;

public:
  ~test_case_for_scalar_jump_problem_t()
  {
    delete level_set;
  }

  virtual double solution_minus(DIM(const double &x, const double &y, const double &z)) const = 0;
  virtual double first_derivative_solution_minus(const unsigned char &der, DIM(const double &x, const double &y, const double &z)) const = 0;
  virtual double second_derivative_solution_minus(const unsigned char &der, DIM(const double &x, const double &y, const double &z)) const = 0;

  virtual double solution_plus(DIM(const double &x, const double &y, const double &z)) const = 0;
  virtual double first_derivative_solution_plus(const unsigned char &der, DIM(const double &x, const double &y, const double &z)) const = 0;
  virtual double second_derivative_solution_plus(const unsigned char &der, DIM(const double &x, const double &y, const double &z)) const = 0;

  inline double jump_in_solution(DIM(const double &x, const double &y, const double &z)) const
  {
    return solution_plus(DIM(x, y, z)) - solution_minus(DIM(x, y, z));
  }

  inline double jump_in_normal_flux(const double local_normal[P4EST_DIM], DIM(const double &x, const double &y, const double &z)) const
  {
    double jump_in_flux = 0.0;
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      jump_in_flux += local_normal[der]*(mu_p*first_derivative_solution_plus(der, DIM(x, y, z)) - mu_m*first_derivative_solution_minus(der, DIM(x, y, z)));
    return jump_in_flux;
  }

  inline double laplacian_u_minus(DIM(const double &x, const double &y, const double &z)) const
  {
    double laplacian = 0.0;
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      laplacian += second_derivative_solution_minus(der, DIM(x, y, z));
    return laplacian;
  }

  inline double laplacian_u_plus(DIM(const double &x, const double &y, const double &z)) const
  {
    double laplacian = 0.0;
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      laplacian += second_derivative_solution_plus(der, DIM(x, y, z));
    return laplacian;
  }

  const double *get_xyz_min() const           { return domain.xyz_min; }
  const double *get_xyz_max() const           { return domain.xyz_max; }
  const int *get_periodicity() const          { return domain.periodicity; }
  const std::string& get_description() const  { return description; }
  double get_avg_solution() const             { return avg_solution; }
  double get_mu_minus() const                 { return mu_m; }
  double get_mu_plus() const                  { return mu_p; }
};

class level_set_t : public CF_DIM
{
protected:
  const domain_t &domain;
  const double ls_center[P4EST_DIM];
  const bool neg_inside;
  virtual double elementary_function(const double xyz_c[P4EST_DIM], const double xyz[P4EST_DIM]) const = 0;

public:
  level_set_t(const domain_t &domain_, const double *center, const bool &negative_inside = true)
    : domain(domain_), ls_center{DIM(center[0], center[1], center[2])}, neg_inside(negative_inside) { }
  level_set_t(const domain_t &domain_, const bool &negative_inside = true)
    : domain(domain_), ls_center{DIM(NAN, NAN, NAN)}, neg_inside(negative_inside) { } // --> we put NANs to make sure that the user knows it is not required...


  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    double ls = elementary_function(ls_center, xyz);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      if(domain.periodicity[dim])
      {
        double ls_center_wrapped[P4EST_DIM] = {DIM(ls_center[0], ls_center[1], ls_center[2])};
        for (char ii = -1; ii < 2; ii += 2)
        {
          ls_center_wrapped[dim] = ls_center[dim] + ii*(domain.xyz_max[dim] - domain.xyz_min[dim]);
          if(neg_inside)
            ls = MIN(ls, elementary_function(ls_center_wrapped, xyz));
          else
            ls = MAX(ls, elementary_function(ls_center_wrapped, xyz));
        }
      }
    return ls;
  }
};

class level_set_sphere : public level_set_t
{
  const double radius;
  double elementary_function(const double xyz_c[P4EST_DIM], const double xyz[P4EST_DIM]) const
  {
    return (neg_inside ? +1.0 : -1.0)*(sqrt(SUMD(SQR(xyz[0] - xyz_c[0]), SQR(xyz[1] - xyz_c[1]), SQR(xyz[2] - xyz_c[2]))) - radius);
  }
public:
  level_set_sphere(const domain_t &domain_, const double *xyz_c, const double &radius_, const bool negative_inside = true)
    : level_set_t(domain_, xyz_c, negative_inside), radius(radius_) { }
};

class level_set_bone_shaped : public level_set_t
{
  const double alpha_x, beta_x;
  const double alpha_y, beta_y, gamma_y;

  double xtheta(double xc_tmp, double t) const { return xc_tmp + alpha_x*cos(t) - beta_x*cos(3.0*t); }
  double d_xtheta(double t) const { return -alpha_x*sin(t) + beta_x*3.0*sin(3.0*t); }
  double dd_xtheta(double t) const { return -alpha_x*cos(t) + beta_x*3.0*3.0*cos(3.0*t); }
  double ytheta(double yc_tmp, double t) const { return yc_tmp + alpha_y*sin(t) - beta_y*sin(3.0*t) + gamma_y*sin(7.0*t); }
  double d_ytheta(double t) const { return alpha_y*cos(t) - beta_y*3.0*cos(3.0*t) + gamma_y*7.0*cos(7.0*t); }
  double dd_ytheta(double t) const { return -alpha_y*sin(t) + beta_y*3.0*3.0*sin(3.0*t) - gamma_y*7.0*7.0*sin(7.0*t); }

  double elementary_function(const double xyz_c[P4EST_DIM], const double xyz[P4EST_DIM]) const
  {
#ifdef P4_TO_P8
    const double xc_tmp = 0.0;
    const double yc_tmp = xyz_c[2];
    double x_tmp        = xc_tmp + sqrt(SQR(xyz[0] - xyz_c[0]) + SQR(xyz[1] - xyz_c[1]));
    double y_tmp        = yc_tmp + fabs(xyz[2] - yc_tmp);
#else
    const double xc_tmp = xyz_c[0];
    const double yc_tmp = xyz_c[1];
    double x_tmp        = xc_tmp + fabs(xyz[0] - xc_tmp);
    double y_tmp        = yc_tmp + fabs(xyz[1] - yc_tmp);
#endif
    size_t n_sample     = 21;
    double theta_start  = 0.0;
    double theta_end    = 0.5*M_PI;
    double dist_min     = +DBL_MAX;
    double theta, theta_opt, distance; theta_opt = 0.0;
    while (theta_end - theta_start > 0.005*.5*M_PI) {
      for (size_t kk = 0; kk < n_sample; ++kk) {
        theta           = theta_start + ((double) kk)*(theta_end - theta_start)/((double) n_sample);
        distance        = sqrt(SQR(xtheta(xc_tmp, theta) - x_tmp) + SQR(ytheta(yc_tmp, theta) - y_tmp));
        if(distance <= dist_min)
        {
          theta_opt     = theta;
          dist_min      = distance;
        }
      }
      double dtheta     = (theta_end - theta_start)/((double) n_sample);
      theta_end         = theta_opt + dtheta;
      theta_start       = theta_opt - dtheta;
    }

    double corr = DBL_MAX;
    double xt, yt, d_xt, d_yt, dd_xt, dd_yt;
    uint counter = 0;
    while (abs(corr) > EPS*0.5*M_PI)
    {
      xt        = xtheta(xc_tmp, theta_opt);
      yt        = ytheta(yc_tmp, theta_opt);
      d_xt      = d_xtheta(theta_opt);
      dd_xt     = dd_xtheta(theta_opt);
      d_yt      = d_ytheta(theta_opt);
      dd_yt     = dd_ytheta(theta_opt);
      corr      = - ((xt - x_tmp)*d_xt + (yt - y_tmp)*d_yt)/
          (SQR(d_xt) + (xt - x_tmp)*dd_xt + SQR(d_yt) + (yt-y_tmp)*dd_yt);
      theta_opt += (((++counter>20) && (fabs(corr) < EPS*5.0*M_PI))?0.5:1.0)*corr; // relaxation needed on very fine grids (oscillatory behavior - and no convergence - observed on a 14/14 grid)
    }
    xt = xtheta(xc_tmp, theta_opt);
    yt = ytheta(yc_tmp, theta_opt);
    dist_min = sqrt(SQR(x_tmp - xt) + SQR(y_tmp - yt));

    bool is_in = false;
    if (x_tmp > xc_tmp + sqrt(5.0/12.0))
      is_in = false;
    else
    {
      double cosroot[3];
      for (unsigned char kk = 0; kk < 3; ++kk)
        cosroot[kk] = 2.0*sqrt(5.0/12.0)*cos((1.0/3.0)*acos(-(x_tmp - xc_tmp)*sqrt(12.0/5.0)) - 2.0*M_PI*((double) kk)/3.0);
      double cosroot_tmp;
      double root;
      double y_lim[2] = {yc_tmp, yc_tmp};
      int pp = 1;
      for (unsigned char kk = 0; kk < 3; ++kk) {
        for (unsigned char jj = kk+1; jj < 3; ++jj) {
          if(cosroot[jj] < cosroot[kk])
          {
            cosroot_tmp = cosroot[kk];
            cosroot[kk] = cosroot[jj];
            cosroot[jj] = cosroot_tmp;
          }
        }
        if((cosroot[kk] >= 0.0) && (cosroot[kk] <= 1.0) && pp >=0)
        {
          root = acos(cosroot[kk]);
          if(fabs(xtheta(xc_tmp, root) - x_tmp) > 10.0*EPS)
            std::cout << "level_set_bone_shaped::this can't be..." << std::endl;
          y_lim[pp--] = ytheta(yc_tmp, root);
        }
      }
      is_in = (y_tmp >= y_lim[0] && y_tmp <= y_lim[1]);
    }
    return (neg_inside == is_in ? -1.0 : 1.0)*dist_min;
  }

public:
  level_set_bone_shaped(const domain_t &domain_, const double *xyz_c, const bool negative_inside = true, const double &alpha_x_ = 0.6, const double &beta_x_ = 0.3, const double &alpha_y_ = 0.7, const double &beta_y_ = 0.07, const double &gamma_y_ = 0.2)
    : level_set_t(domain_, xyz_c, negative_inside), alpha_x(alpha_x_), beta_x(beta_x_), alpha_y(alpha_y_), beta_y(beta_y_), gamma_y(gamma_y_) {}
};

class level_set_flower : public level_set_t {
  const double radius, amplitude;
  const unsigned int nphi;
#ifdef P4_TO_P8
  const double phi_modulation;
  const unsigned int ntheta;
#endif
  double elementary_function(const double xyz_c[P4EST_DIM], const double xyz[P4EST_DIM]) const
  {
    double phi = 0.0;
    if (fabs(xyz[0] - xyz_c[0]) > EPS*domain.length() || fabs(xyz[1] - xyz_c[1]) > EPS*domain.height())
      phi = atan2(xyz[1] - xyz_c[1], xyz[0] - xyz_c[0]);
#ifdef P4_TO_P8
    double theta = 0.0;
    const double rr = sqrt(SQR(xyz[0] - xyz_c[0]) + SQR(xyz[1] - xyz_c[1]) + SQR(xyz[2] - xyz_c[2]));
    if (rr > EPS*MAX(domain.length(), domain.height(), domain.width()))
      theta = acos((xyz[2] - xyz_c[2])/rr);
    return (neg_inside ? +1.0 : -1.0)*(SQR(xyz[0] - xyz_c[0]) + SQR(xyz[1] - xyz_c[1]) + SQR(xyz[2] - xyz_c[2]) - SQR(radius + amplitude*(1.0 - phi_modulation*cos(nphi*phi))*(1.0 - cos(ntheta*theta))));
#else
    return (neg_inside ? +1.0 : -1.0)*(SQR(xyz[0] - xyz_c[0]) + SQR(xyz[1] - xyz_c[1]) - SQR(radius + amplitude*sin(nphi*phi)));
#endif
  }
public:
  level_set_flower(const domain_t &domain_, const double *xyz_c, const double &radius_, const double &amplitude_, const bool &negative_inside = true,
                 #ifndef P4_TO_P8
                   const unsigned int npetals_phi = 5
    #else
                   const unsigned int npetals_phi = 6, const double &phi_mod_ = 0.2, const unsigned int npetals_theta = 3
    #endif
      )
    : level_set_t(domain_, xyz_c, negative_inside), radius(radius_), amplitude(amplitude_), nphi(npetals_phi)
  #ifdef P4_TO_P8
    , phi_modulation(phi_mod_) , ntheta(2*npetals_theta)
  #endif
  {}
};

#ifndef P4_TO_P8
class level_set_bubbles : public level_set_t {
  std::vector<double> center_bubbles[P4EST_DIM];
  std::vector<double> radius_bubbles;
  std::vector<double> theta_bubbles;
  std::vector<double> dt_bubbles;
  const size_t n_bubbles;
  const double min_bubble_radius;
  const double max_bubble_radius;

  bool added_bubble_is_valid(const size_t & k)
  {
    center_bubbles[0][k]  = domain.xyz_min[0] + 1.5*max_bubble_radius + (domain.length() - 2.0*1.5*max_bubble_radius)*((double) rand() / RAND_MAX);
    center_bubbles[1][k]  = domain.xyz_min[1] + 1.5*max_bubble_radius + (domain.height() - 2.0*1.5*max_bubble_radius)*((double) rand() / RAND_MAX);
    radius_bubbles[k]     = min_bubble_radius + (max_bubble_radius - min_bubble_radius)*((double) rand() / RAND_MAX);
    theta_bubbles[k]      = 2.0*M_PI*((double) rand() / RAND_MAX);
    dt_bubbles[k]         = 2.0*((double) rand() / RAND_MAX);
    if(k == 0)
      return true;

    bool no_intersection = true;
    for (size_t kk = 0; kk < k; ++kk)
      no_intersection = no_intersection && (sqrt(SQR(center_bubbles[0][k] - center_bubbles[0][kk]) + SQR(center_bubbles[1][k] - center_bubbles[1][kk])) > sqrt(5.0)*(radius_bubbles[k] + radius_bubbles[kk]) + 0.001);
    return no_intersection;
  }

  double elementary_function(const double*, const double xyz[P4EST_DIM]) const // first argument is irrelevant in this case
  {
    double phi = +DBL_MAX;
    for (size_t k = 0; k < n_bubbles; ++k)
    {
      double u      = (xyz[0] - center_bubbles[0][k])*cos(theta_bubbles[k]) + (xyz[1] - center_bubbles[1][k])*sin(theta_bubbles[k]);
      double v      = -(xyz[0] - center_bubbles[0][k])*sin(theta_bubbles[k]) + (xyz[1] - center_bubbles[1][k])*cos(theta_bubbles[k]);
      double vel_v  = (1.0 + SQR(v/max_bubble_radius))*radius_bubbles[k];
      double uc_adv = -radius_bubbles[k]*dt_bubbles[k];
      if(neg_inside)
        phi = MIN(phi, sqrt(SQR(u - vel_v*dt_bubbles[k] - uc_adv) + SQR(v)) - radius_bubbles[k]);
      else
        phi = MAX(phi, radius_bubbles[k] - sqrt(SQR(u - vel_v*dt_bubbles[k] - uc_adv) + SQR(v)));
    }
    return phi;
  }

public:
  level_set_bubbles(const domain_t &domain_, const size_t &nbubbles = 15, const double &min_rad_ = 0.005, const double &max_rad_ = 0.02) : level_set_t(domain_),
    n_bubbles(nbubbles), min_bubble_radius(min_rad_), max_bubble_radius(max_rad_)
  {
    srand(time(0));
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      center_bubbles[dim].resize(n_bubbles);
    radius_bubbles.resize(n_bubbles);
    theta_bubbles.resize(n_bubbles);
    dt_bubbles.resize(n_bubbles);
    for (size_t k = 0; k < n_bubbles; ++k)
      while (added_bubble_is_valid(k)) { }
  }
};
#endif

#ifndef P4_TO_P8

static class GFM_example_3_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_3_t()
  {
    mu_m = 2.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = 0.0;
      domain.xyz_max[dim] = 1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.5, 0.5};
    level_set = new level_set_sphere(domain, center, 0.25, true);

    solution_integral = 0.117241067253686032041502125837326856300828722393373744878; // calculated with Wolfram

    description =
        std::string("* domain = [0.0, 1.0] X [0.0, 1.0] \n")
        + std::string("* interface = circle of radius 1/4, centered in (0.5, 0.5), negative inside, positive outside \n")
        + std::string("* mu_m = 2.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = exp(-x*x - y*y); \n")
        + std::string("* u_p  = 0.0; \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 3 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return exp(-x*x - y*y);
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = -2.0*x;
      break;
    case dir::y:
      to_return = -2.0*y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_3_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
    return to_return*solution_minus(x, y);
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = 4.0*x*x - 2.0;
      break;
    case dir::y:
      to_return = 4.0*y*y - 2.0;
      break;
    default:
      throw  std::invalid_argument("GFM_example_3_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
    return to_return*solution_minus(x, y);
  }

  double solution_plus(const double &, const double &) const
  {
    return 0.0;
  }

  double first_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }

  double second_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }
} GFM_example_3;

static class GFM_example_5_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_5_t()
  {
    mu_m = 1.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -1.0;
      domain.xyz_max[dim] = +1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.0, 0.0};
    level_set = new level_set_sphere(domain, center, 0.5, true);

    solution_integral = domain.length()*domain.height()
        + 2.0*M_PI*(0.5*log(2.0) - 0.25 + 0.25*0.5*0.5)
        + (-M_PI_2*2.0 - 6.0 + M_PI*2.0*log(2.0*sqrt(2.0)) + 2.0*2.0*acos(1.0/sqrt(2.0)) + 4.0*sqrt(2.0)*log(2.0*sqrt(2.0))*(1.0/sqrt(2.0) - sqrt(2.0)*acos(1.0/sqrt(2.0))) - 4.0*asin(1.0/sqrt(2.0)))
        - (-M_PI_2 + M_PI*log(2.0) - 2.0*M_PI); // analytically calculated (with Wolfram's help)

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("* interface = circle of radius 1/2, centered in (0.0, 0.0), negative inside, positive outside \n")
        + std::string("* mu_m = 2.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = 1.0; \n")
        + std::string("* u_p  = 1.0 + log(2.0*sqrt(x*x + y*y)); \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 5 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &, const double &) const
  {
    return 1.0;
  }

  double first_derivative_solution_minus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }

  double second_derivative_solution_minus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }

  double solution_plus(const double &x, const double &y) const
  {
    return 1.0 + log(2.0*sqrt(EPS + SQR(x) + SQR(y)));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = x;
      break;
    case dir::y:
      to_return = y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_5_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }

    return to_return/(EPS + SQR(x) + SQR(y));
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = y*y - x*x;
      break;
    case dir::y:
      to_return = x*x - y*y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_5_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
    return to_return/SQR(EPS + SQR(x) + SQR(y));
  }
} GFM_example_5;

static class GFM_example_6_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_6_t()
  {
    mu_m = 1.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -1.0;
      domain.xyz_max[dim] = +1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.0, 0.0};
    level_set = new level_set_sphere(domain, center, 0.5, true);
    solution_integral = 0.785398163397448309615660845819875721049292349843776455243; // calculated with Wolfram

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("* interface = circle of radius 1/2, centered in (0.0, 0.0), negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = exp(x)*cos(y); \n")
        + std::string("* u_p  = 0.0; \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 6 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return exp(x)*cos(y);
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return solution_minus(x, y);
      break;
    case dir::y:
      return -exp(x)*sin(y);
      break;
    default:
      throw  std::invalid_argument("GFM_example_6_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return solution_minus(x, y);
      break;
    case dir::y:
      return -solution_minus(x, y);
      break;
    default:
      throw  std::invalid_argument("GFM_example_6_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &, const double &) const
  {
    return 0.0;
  }

  double first_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }

  double second_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }
} GFM_example_6;

static class GFM_example_7_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_7_t()
  {
    mu_m = 1.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -1.0;
      domain.xyz_max[dim] = +1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.0, 0.0};
    level_set = new level_set_sphere(domain, center, 0.5, true);
    solution_integral = 0.0;

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("* interface = circle of radius 1/2, centered in (0.0, 0.0), negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = x*x - y*y; \n")
        + std::string("* u_p  = 0.0; \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 7 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return x*x - y*y;
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return 2.0*x;
      break;
    case dir::y:
      return -2.0*y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_7_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &, const double &) const
  {
    switch (der) {
    case dir::x:
      return +2.0;
      break;
    case dir::y:
      return -2.0;
      break;
    default:
      throw  std::invalid_argument("GFM_example_7_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &, const double &) const
  {
    return 0.0;
  }

  double first_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }

  double second_derivative_solution_plus(const unsigned char &, const double &, const double &) const
  {
    return 0.0;
  }
} GFM_example_7;

static class GFM_example_8_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_8_t()
  {
    mu_m = 1.0;
    mu_p = 10.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -1.0;
      domain.xyz_max[dim] = +1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.02*sqrt(5.0), 0.02*sqrt(5.0)};
    level_set = new level_set_flower(domain, center, 0.5, 0.2, true, 5);
    solution_integral = 0.3782030713479; // calculated on a 14/14 grid (1X1 macromesh) on stampede...

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("* interface = curve parameterized by (t in [0, 2.0*PI[) \n")
        + std::string("@ x(t) = 0.02*sqrt(5.0) + (0.5 + 0.2*sin(5*t))*cos(t) \n")
        + std::string("@ y(t) = 0.02*sqrt(5.0) + (0.5 + 0.2*sin(5*t))*sin(t) \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 10.0; \n")
        + std::string("* u_m  = x*x + y*y; \n")
        + std::string("* u_p  = 0.1*(x*x + y*y)^2 - 0.01*log(2.0*sqrt(EPS + x*x + y*y)); \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 8 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return x*x + y*y;
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return 2.0*x;
      break;
    case dir::y:
      return 2.0*y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_8_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &, const double &) const
  {
    switch (der) {
    case dir::x:
      return +2.0;
      break;
    case dir::y:
      return +2.0;
      break;
    default:
      throw  std::invalid_argument("GFM_example_8_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return 0.1*SQR(x*x + y*y) - 0.01*log(2.0*sqrt(EPS + x*x + y*y));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return 0.1*2.0*(x*x + y*y)*2.0*x - 0.01*x/(EPS + x*x + y*y);
      break;
    case dir::y:
      return 0.1*2.0*(x*x + y*y)*2.0*y - 0.01*y/(EPS + x*x + y*y);
      break;
    default:
      throw  std::invalid_argument("GFM_example_8_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return 0.1*2.0*2.0*(3.0*x*x + y*y) - 0.01*(y*y - x*x)/SQR(EPS + x*x + y*y);
      break;
    case dir::y:
      return 0.1*2.0*2.0*(x*x + 3.0*y*y) - 0.01*(x*x - y*y)/SQR(EPS + x*x + y*y);
      break;
    default:
      throw  std::invalid_argument("GFM_example_8_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} GFM_example_8;

static class GFM_example_9_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_9_t()
  {
    mu_m = 1.0;
    mu_p = 10.0;
    domain.xyz_min[0] = -1.5; domain.xyz_max[0] = +1.5; domain.periodicity[0] = 0;
    domain.xyz_min[1] =  0.0; domain.xyz_max[1] = +3.0; domain.periodicity[1] = 0;
    const double center[2] = {0.0, 1.5};
    level_set = new level_set_bone_shaped(domain, center, true, 0.6, 0.3, 0.7, 0.07, 0.2);
    solution_integral = -25.59547830010; // calculated on a 14/14 grid (1X1 macromesh) on stampede...

    description =
        std::string("* domain = [-1.5, 1.5] X [0.0, 3.0] \n")
        + std::string("* interface = curve parameterized by (t in [0, 2.0*PI[) \n")
        + std::string("@ x(t) = 0.6*cos(t) - 0.3*cos(3*t) \n")
        + std::string("@ y(t) = 1.5 + 0.7*sin(t) - 0.07*sin(3*t) + 0.2*sin(7*t) \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 10.0; \n")
        + std::string("* u_m  = exp(x)*(x*x*sin(y) + y*y); \n")
        + std::string("* u_p  =  -x*x - y*y;  \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 9 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return exp(x)*(x*x*sin(y) + y*y);
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return exp(x)*((x*x + 2.0*x)*sin(y) + y*y);
      break;
    case dir::y:
      return exp(x)*(x*x*cos(y) + 2.0*y);
      break;
    default:
      throw  std::invalid_argument("GFM_example_9_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return exp(x)*((x*x + 4.0*x + 2.0)*sin(y) + y*y);
      break;
    case dir::y:
      return exp(x)*(-x*x*sin(y) + 2.0);
      break;
    default:
      throw  std::invalid_argument("GFM_example_9_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return -x*x - y*y;
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -2.0*x;
      break;
    case dir::y:
      return -2.0*y;
      break;
    default:
      throw  std::invalid_argument("GFM_example_9_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &, const double &) const
  {
    switch (der) {
    case dir::x:
      return -2.0;
      break;
    case dir::y:
      return -2.0;
      break;
    default:
      throw  std::invalid_argument("GFM_example_9_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} GFM_example_9;

static class xGFM_example_large_ratio_moderate_flower_2D_t : public test_case_for_scalar_jump_problem_t
{
public:
  xGFM_example_large_ratio_moderate_flower_2D_t()
  {
    mu_m = 10000.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -1.0;
      domain.xyz_max[dim]     = +1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[2] = {0.0, 0.0};
    level_set = new level_set_flower(domain, center, 0.5, 0.1, true, 5);
    solution_integral = 2.415389999053; // calculated on a 14/14 grid (1X1 macromesh) on stampede...

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("* interface = curve parameterized by (t in [0, 2.0*PI[) \n")
        + std::string("@ x(t) = (0.5 + 0.1*sin(5*t)).*cos(t) \n")
        + std::string("@ y(t) = (0.5 + 0.1*sin(5*t)).*cos(t) \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 10000.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = exp(x)*(x*x*sin(y) + y*y)/mu_m; \n")
        + std::string("* u_p  = 0.5 + cos(x)*(y^4 + sin(y*y - x*x)); \n")
        + std::string("* no periodicity \n")
        + std::string("Example for large ratio of diffusion coefficient (example 4.6 of R. Egan, F. Gibou, JCP, May 2020)");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return exp(x)*(x*x*sin(y) + y*y)/mu_m;
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return exp(x)*((x*x + 2.0*x)*sin(y) + y*y)/mu_m;
      break;
    case dir::y:
      return exp(x)*(x*x*cos(y) + 2.0*y)/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_2D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return exp(x)*((x*x + 4.0*x + 2.0)*sin(y) + y*y)/mu_m;
      break;
    case dir::y:
      return exp(x)*(-x*x*sin(y) + 2.0)/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_2D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return 0.5 + cos(x)*(pow(y, 4.0) + sin(y*y - x*x));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -sin(x)*(pow(y, 4.0) + sin(y*y - x*x)) - 2.0*x*cos(x)*cos(y*y - x*x);
      break;
    case dir::y:
      return cos(x)*(4.0*pow(y, 3.0) + 2.0*y*cos(y*y - x*x));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_2D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -cos(x)*(pow(y, 4.0) + sin(y*y - x*x)) + (4.0*x*sin(x) - 2.0*cos(x))*cos(y*y - x*x) - 4.0*x*x*cos(x)*sin(y*y - x*x);
      break;
    case dir::y:
      return cos(x)*(12.0*SQR(y) + 2.0*cos(y*y - x*x) - 4.0*y*y*sin(y*y - x*x));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_2D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_large_ratio_moderate_flower_2D;

static class xGFM_example_random_bubbles_t : public test_case_for_scalar_jump_problem_t
{
public:
  xGFM_example_random_bubbles_t()
  {
    mu_m = 1000.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -1.0;
      domain.xyz_max[dim]     = +1.0;
      domain.periodicity[dim] = 0;
    }
    level_set = new level_set_bubbles(domain, 15, 0.005, 0.02);

    solution_integral = NAN; // undefined in this (random case)

    description =
        std::string("* domain = [-1.0, 1.0] X [-1.0, 1.0] \n")
        + std::string("-interface = 15 small spherical bubbles in the domain, radius between 0.005 and 0.02 \n")
        + std::string("negative inside the bubbles, positive outside \n")
        + std::string("* mu_m = 1000.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = (cos(2.0*M_PI*(x+3.0*y)/0.04) - sin(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m; \n")
        + std::string("* u_p  = cos(x + y)*exp(-SQR(x*cos(y))); \n")
        + std::string("* no periodicity \n")
        + std::string("Example for adaptivity and large ratio of coefficients (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return (cos(2.0*M_PI*(x + 3.0*y)/0.04) - sin(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m;
  }
  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (-(2.0*M_PI/0.04)*sin(2.0*M_PI*(x + 3.0*y)/0.04) + (2.0*M_PI*2.0/0.04)*cos(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m;
      break;
    case dir::y:
      return (-(2.0*M_PI*3.0/0.04)*sin(2.0*M_PI*(x + 3.0*y)/0.04) - (2.0*M_PI/0.04)*cos(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_bubbles_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (-SQR(2.0*M_PI/0.04)*cos(2.0*M_PI*(x + 3.0*y)/0.04) + SQR(2.0*M_PI*2.0/0.04)*sin(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m;
      break;
    case dir::y:
      return (-SQR(2.0*M_PI*3.0/0.04)*cos(2.0*M_PI*(x + 3.0*y)/0.04) + SQR(2.0*M_PI/0.04)*sin(2.0*M_PI*(y - 2.0*x)/0.04))/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_bubbles_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return cos(x + y)*exp(-SQR(x*cos(y)));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -exp(-SQR(x*cos(y)))*(sin(x + y) + 2.0*x*cos(x + y)*SQR(cos(y)));
      break;
    case dir::y:
      return exp(-SQR(x*cos(y)))*(2.0*x*x*sin(y)*cos(y)*cos(x + y) - sin(x + y));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_bubbles_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return exp(-SQR(x*cos(y)))*(4.0*x*x*cos(x + y)*SQR(SQR(cos(y))) + 4.0*x*SQR(cos(y))*sin(x + y) - cos(x + y) - 2.0*cos(x + y)*SQR(cos(y)));
      break;
    case dir::y:
      return exp(-SQR(x*cos(y)))*(4.0*SQR(x*x*cos(y)*sin(y))*cos(x + y) - 2.0*x*x*sin(2.0*y)*sin(x + y) +2.0*x*x*cos(2.0*y)*cos(x + y) - cos(x + y));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_bubbles_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_random_bubbles;

static class xGFM_example_x_periodic_2D_t : public test_case_for_scalar_jump_problem_t
{
public:
  xGFM_example_x_periodic_2D_t()
  {
    mu_m = 1.0;
    mu_p = 10.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -1.5;
      domain.xyz_max[dim]     = +1.5;
    }
    domain.periodicity[0]   = 1;
    domain.periodicity[1]   = 0;
    const double center[2]  = {0.85*domain.xyz_min[0] + 0.15*domain.xyz_max[0], 0.0};
    level_set = new level_set_bone_shaped(domain, center, true, 0.6, 0.3, 0.7, 0.07, 0.2);
    solution_integral = 0.6448672580288; // calculated on a 14/14 grid (1X1 macromesh) on stampede...

    description =
        std::string("* domain = [-1.5, 1.5] X [-1.5, 1.5] \n")
        + std::string("* interface = curve parameterized by (t in [0, 2.0*PI[) \n")
        + std::string("@ x(t) = 0.85*xmin + 0.15*xmax + 0.6*cos(t) - 0.3*cos(3*t) \n")
        + std::string("@ y(t) = 0.7*sin(t) - 0.07*sin(3*t) + 0.2*sin(7*t) \n")
        + std::string("negative inside, positive outside (periodicity along x enforced) \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 10.0; \n")
        + std::string("* u_m  = cos((2.0*M_PI/3.0)*(x - tanh(y))) + exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y)))); \n")
        + std::string("* u_p  = tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y); \n")
        + std::string("* Example for periodicity along x, no periodicity along y (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return cos((2.0*M_PI/3.0)*(x - tanh(y))) + exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))));
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x - tanh(y))) - 2.0*(2.0*M_PI/3.0)*sin(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))));
      break;
    case dir::y:
      return (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x - tanh(y)))*(1.0 - SQR(tanh(y))) + (2.0*M_PI/3.0)*0.251*sin(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_x_periodic_2D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -SQR(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(x - tanh(y))) - 2.0*SQR(2.0*2.0*M_PI/3.0)*cos(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y)))) + SQR(2.0*(2.0*M_PI/3.0)*sin(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y)))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))));
      break;
    case dir::y:
      return
          - SQR(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(x - tanh(y)))*SQR(1.0 - SQR(tanh(y)))
          + (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(x - tanh(y)))*(-2.0*tanh(y)*(1.0 - SQR(tanh(y))))
          + (2.0*M_PI/3.0)*0.251*cos(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y))*(2.0*(2.0*M_PI/3.0)*(-0.251))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))))
          + SQR((2.0*M_PI/3.0)*0.251)*SQR(sin(2.0*(2.0*M_PI/3.0)*(2.0*x - 0.251*y)))*exp(-SQR(sin((2.0*M_PI/3.0)*(2.0*x - 0.251*y))));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_x_periodic_2D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y);
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*(-(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*2.0*x));
      break;
    case dir::y:
      return -0.24*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_x_periodic_2D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return - 2.0*tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*SQR(-(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*2.0*x))
          + (1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)))*(-SQR((2.0*M_PI/3.0)*2.0)*cos((2.0*M_PI/3.0)*2.0*x));
      break;
    case dir::y:
      return -2.0*SQR(0.24)*tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)*(1.0 - SQR(tanh(cos((2.0*M_PI/3.0)*2.0*x) - 0.24*y)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_x_periodic_2D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_x_periodic_2D;

static class xGFM_example_full_periodic_2D_t : public test_case_for_scalar_jump_problem_t
{
public:
  xGFM_example_full_periodic_2D_t()
  {
    mu_m = 1.0;
    mu_p = 100.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -1.5;
      domain.xyz_max[dim]     = +1.5;
      domain.periodicity[dim] = 1;
    }
    const double center[2]  = {0.85*domain.xyz_min[0] + 0.15*domain.xyz_max[0],
                               0.8*domain.xyz_min[1] + 0.2*domain.xyz_max[1]};
    level_set = new level_set_bone_shaped(domain, center, true, 0.6, 0.3, 0.7, 0.07, 0.2);
    solution_integral = 1.652618403615; // calculated on a 14/14 grid (1X1 macromesh) on stampede...

    description =
        std::string("* domain = [-1.5, 1.5] X [-1.5, 1.5] \n")
        + std::string("* interface = curve parameterized by (t in [0, 2.0*PI[) \n")
        + std::string("@ x(t) = 0.85*xmin + 0.15*xmax + 0.6*cos(t) - 0.3*cos(3*t) \n")
        + std::string("@ y(t) = 0.8*ymin + 0.2*ymax + 0.7*sin(t) - 0.07*sin(3*t) + 0.2*sin(7*t) \n")
        + std::string("negative inside, positive outside (periodicity along x and y enforced) \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 100.0; \n")
        + std::string("* u_m  = atan(sin((2.0*M_PI/3.0)*(2.0*x - y))); \n")
        + std::string("* u_p  = log(1.5 + cos((2.0*M_PI/3.0)*(-x + 3.0*y))); \n")
        + std::string("* fully periodic \n")
        + std::string("Example for full periodicity (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y) const
  {
    return atan(sin((2.0*M_PI/3.0)*(2.0*x - y)));
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (2.0*(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(2.0*x - y)))/(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y))));
      break;
    case dir::y:
      return (-(2.0*M_PI/3.0)*cos((2.0*M_PI/3.0)*(2.0*x - y)))/(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y))));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_2D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (-SQR(2.0*2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(2.0*x - y))*(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y)))) - SQR(2.0*2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*(2.0*x - y))*SQR(cos((2.0*M_PI/3.0)*(2.0*x - y))))/SQR(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y))));
      break;
    case dir::y:
      return (-SQR(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(2.0*x - y))*(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y)))) - SQR(2.0*M_PI/3.0)*2.0*sin((2.0*M_PI/3.0)*(2.0*x - y))*SQR(cos((2.0*M_PI/3.0)*(2.0*x - y))))/SQR(1.0 + SQR(sin((2.0*M_PI/3.0)*(2.0*x - y))));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_2D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y) const
  {
    return log(1.5 + cos((2.0*M_PI/3.0)*(-x + 3.0*y)));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return (2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(3.0*y - x))/(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x)));
      break;
    case dir::y:
      return -(3.0*(2.0*M_PI/3.0)*sin((2.0*M_PI/3.0)*(3.0*y - x)))/(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_2D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y) const
  {
    switch (der) {
    case dir::x:
      return -SQR(2.0*M_PI/3.0)*(cos((2.0*M_PI/3.0)*(3.0*y - x))*(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x))) + SQR(sin((2.0*M_PI/3.0)*(3.0*y - x))))/SQR(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x)));
      break;
    case dir::y:
      return -(SQR(3.0*(2.0*M_PI/3.0))*(cos((2.0*M_PI/3.0)*(3.0*y - x))*(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x))) + SQR(sin((2.0*M_PI/3.0)*(3.0*y - x)))))/SQR(1.5 + cos((2.0*M_PI/3.0)*(3.0*y - x)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_2D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_full_periodic_2D;

#else

static class GFM_example_4_t : public test_case_for_scalar_jump_problem_t
{
public:
  GFM_example_4_t()
  {
    mu_m = 2.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = 0.0;
      domain.xyz_max[dim] = 1.0;
      domain.periodicity[dim] = 0;
    }
    const double center[3] = {0.5, 0.5, 0.5};
    level_set = new level_set_sphere(domain, center, 0.25, true);

    solution_integral = 0.030340552739300;  // using Richardson's extrapolation between uniform 1024x1024x1024 and 2048x2048x2048 grids (assuming second-order accurate integration)

    description =
        std::string("* domain = [0.0, 1.0] X [0.0, 1.0] X [0.0, 1.0] \n")
        + std::string("* interface = sphere of radius 1/4, centered in (0.5, 0.5, 0.5), negative inside, positive outside \n")
        + std::string("* mu_m = 2.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = exp(-x*x - y*y - z*z); \n")
        + std::string("* u_p  = 0.0; \n")
        + std::string("* no periodicity \n")
        + std::string("* Example 4 from Liu, Fedkiw, Kang 2000");
  }

  double solution_minus(const double &x, const double &y, const double &z) const
  {
    return exp(-x*x - y*y - z*z);
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = -2.0*x;
      break;
    case dir::y:
      to_return = -2.0*y;
      break;
    case dir::z:
      to_return = -2.0*z;
      break;
    default:
      throw  std::invalid_argument("GFM_example_4_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
    return to_return*solution_minus(x, y, z);
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    double to_return;
    switch (der) {
    case dir::x:
      to_return = 4.0*x*x - 2.0;
      break;
    case dir::y:
      to_return = 4.0*y*y - 2.0;
      break;
    case dir::z:
      to_return = 4.0*z*z - 2.0;
      break;
    default:
      throw  std::invalid_argument("GFM_example_4_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
    return to_return*solution_minus(x, y, z);
  }

  double solution_plus(const double &, const double &, const double &) const
  {
    return 0.0;
  }

  double first_derivative_solution_plus(const unsigned char &, const double &, const double &, const double &) const
  {
    return 0.0;
  }

  double second_derivative_solution_plus(const unsigned char &, const double &, const double &, const double &) const
  {
    return 0.0;
  }
} GFM_example_4;

static class xGFM_example_large_ratio_moderate_flower_3D_t : public test_case_for_scalar_jump_problem_t
{
public:
  xGFM_example_large_ratio_moderate_flower_3D_t()
  {
    mu_m = 2000.0;
    mu_p = 1.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -2.0;
      domain.xyz_max[dim] = +2.0;
      domain.periodicity[dim] = 0;
    }
    const double center[3] = {0., 0., 0.};
    level_set = new level_set_flower(domain, center, 1.25, 0.2, true, 6, 0.2, 3);

    solution_integral = 197.6819074552000; // using Richardson's extrapolation between uniform 1024x1024x1024 and 2048x2048x2048 grids (assuming second-order accurate integration)

    description =
        std::string("* domain = [-2.0, 2.0] X [-2.0, 2.0] X [-2.0, 2.0] \n")
        + std::string("* interface = parameterized by (theta in [0.0, pi[, phi in [0.0, 2*pi[) \n")
        + std::string("r(theta, phi) = 1.25 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0 - cos(6.0*theta)), spherical coordinates \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 2000.0; \n")
        + std::string("* mu_p = 1.0; \n")
        + std::string("* u_m  = 3.0 + exp(.5*(x - z))*(x*sin(y) - cos(x + y)*atan(z))*4.0/mu_m; \n")
        + std::string("* u_p  = exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x)); \n")
        + std::string("* no periodicity \n")
        + std::string("Example for mildly convoluted 3D interface with large ratio of coefficients (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y, const double &z) const
  {
    return 3.0 + exp(.5*(x - z))*(x*sin(y) - cos(x + y)*atan(z))*4.0/mu_m;
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(.5*(x - z))*((1.0 + 0.5*x)*sin(y) + (sin(x + y) - 0.5*cos(x + y))*atan(z))*4.0/mu_m;
      break;
    case dir::y:
      return exp(.5*(x - z))*(x*cos(y) + sin(x + y)*atan(z))*4.0/mu_m;
      break;
    case dir::z:
      return -exp(.5*(x - z))*(.5*x*sin(y) + cos(x + y)*(1.0/(1.0 + SQR(z)) - .5*atan(z)))*4.0/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_3D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(.5*(x - z))*((1.0 + 0.25*x)*sin(y) + (sin(x + y) + 0.75*cos(x + y))*atan(z))*4.0/mu_m;
      break;
    case dir::y:
      return exp(.5*(x - z))*(-x*sin(y) + cos(x + y)*atan(z))*4.0/mu_m;
      break;
    case dir::z:
      return exp(.5*(x - z))*(.25*x*sin(y) + cos(x + y)*(2.0*z/(SQR(1.0 + SQR(z))) + 1.0/(1.0 + SQR(z)) - .25*atan(z)))*4.0/mu_m;
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_3D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y, const double &z) const
  {
    return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(-sin(y) + 2.0*z*sin(2.0*x));
      break;
    case dir::y:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(-x*cos(y) - cos(z));
      break;
    case dir::z:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(+y*sin(z) - cos(2.0*x));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_3D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(SQR(-sin(y) + 2.0*z*sin(2.0*x)) + 4.0*z*cos(2.0*x));
      break;
    case dir::y:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(SQR(-x*cos(y) - cos(z)) + x*sin(y));
      break;
    case dir::z:
      return exp(-x*sin(y) - y*cos(z) - z*cos(2.0*x))*(SQR(+y*sin(z) - cos(2.0*x)) + y*cos(z));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_moderate_flower_3D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_large_ratio_moderate_flower_3D;

static class xGFM_example_large_ratio_severe_flower_3D_t : public test_case_for_scalar_jump_problem_t
{
  double ff(const double &x, const double &y, const double &z) const
  {
    return 0.1*x*x*x*y + 2.0*z*cos(y) - y*sin(x + z);
  }
  double dff_dx(const double &x, const double &y, const double &z) const
  {
    return 3.0*0.1*x*x*y - y*cos(x + z);
  }
  double ddff_dxdx(const double &x, const double &y, const double &z) const
  {
    return 2.0*3.0*0.1*x*y + y*sin(x + z);
  }

  double dff_dy(const double &x, const double &y, const double &z) const
  {
    return 0.1*x*x*x - 2.0*z*sin(y) - sin(x + z);
  }
  double ddff_dydy(const double &x, const double &y, const double &z) const
  {
    return -2.0*z*cos(y);
  }

  double dff_dz(const double &x, const double &y, const double &z) const
  {
    return 2.0*cos(y) - y*cos(x + z);
  }
  double ddff_dzdz(const double &x, const double &y, const double &z) const
  {
    return y*sin(x + z);
  }

public:
  xGFM_example_large_ratio_severe_flower_3D_t()
  {
    mu_m = 1.0;
    mu_p = 1250.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -2.0;
      domain.xyz_max[dim] = +2.0;
      domain.periodicity[dim] = 0;
    }
    const double center[3] = {0., 0., 0.};
    level_set = new level_set_flower(domain, center, 0.75, 0.2, true, 6, 0.6, 3);

    solution_integral = NAN; // not known, yet

    description =
        std::string("* domain = [-2.0, 2.0] X [-2.0, 2.0] X [-2.0, 2.0] \n")
        + std::string("* interface = parameterized by (theta in [0.0, pi[, phi in [0.0, 2*pi[) \n")
        + std::string("r(theta, phi) = 1.25 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0 - cos(6.0*theta)), spherical coordinates \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 1250.0; \n")
        + std::string("* u_m  = exp(.5*(x - z))*(x*sin(y) - cos(x + y)*atan(z)); \n")
        + std::string("* u_p  = -1.0 + atan(0.1*x*x*x*y + 2.0*z*cos(y) - y*sin(x + z))*2.5/mu_p; \n")
        + std::string("* no periodicity \n")
        + std::string("Example for very convoluted 3D interface with large ratio of coefficients (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y, const double &z) const
  {
    return exp(.5*(x - z))*(x*sin(y) - cos(x + y)*atan(z));
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(.5*(x - z))*((1.0 + 0.5*x)*sin(y) + (sin(x + y) - 0.5*cos(x + y))*atan(z));
      break;
    case dir::y:
      return exp(.5*(x - z))*(x*cos(y) + sin(x + y)*atan(z));
      break;
    case dir::z:
      return -exp(.5*(x - z))*(.5*x*sin(y) + cos(x + y)*(1.0/(1.0 + SQR(z)) - .5*atan(z)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_severe_flower_3D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return exp(.5*(x - z))*((1.0 + 0.25*x)*sin(y) + (sin(x + y) + 0.75*cos(x + y))*atan(z));
      break;
    case dir::y:
      return exp(.5*(x - z))*(-x*sin(y) + cos(x + y)*atan(z));
      break;
    case dir::z:
      return exp(.5*(x - z))*(.25*x*sin(y) + cos(x + y)*(2.0*z/(SQR(1.0 + SQR(z))) + 1.0/(1.0 + SQR(z)) - .25*atan(z)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_severe_flower_3D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y, const double &z) const
  {
    return -1.0 + atan(ff(x, y, z))*2.5/mu_p;
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (2.5/mu_p)*(1.0/(1.0 + SQR(ff(x, y, z))))*dff_dx(x, y, z);
      break;
    case dir::y:
      return (2.5/mu_p)*(1.0/(1.0 + SQR(ff(x, y, z))))*dff_dy(x, y, z);
      break;
    case dir::z:
      return (2.5/mu_p)*(1.0/(1.0 + SQR(ff(x, y, z))))*dff_dz(x, y, z);
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_severe_flower_3D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (2.5/mu_p)*(ddff_dxdx(x, y, z)*(1.0 + SQR(ff(x, y, z))) - 2.0*SQR(dff_dx(x, y, z))*ff(x, y, z))/SQR(1.0 + SQR(ff(x, y, z)));
      break;
    case dir::y:
      return (2.5/mu_p)*(ddff_dydy(x, y, z)*(1.0 + SQR(ff(x, y, z))) - 2.0*SQR(dff_dy(x, y, z))*ff(x, y, z))/SQR(1.0 + SQR(ff(x, y, z)));
      break;
    case dir::z:
      return (2.5/mu_p)*(ddff_dzdz(x, y, z)*(1.0 + SQR(ff(x, y, z))) - 2.0*SQR(dff_dz(x, y, z))*ff(x, y, z))/SQR(1.0 + SQR(ff(x, y, z)));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_large_ratio_severe_flower_3D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_large_ratio_severe_flower_3D;

static class xGFM_example_full_periodic_3D_t : public test_case_for_scalar_jump_problem_t
{
  double pp(const double &x, const double &y, const double &) const
  {
    return sin((2.0*M_PI/3.0)*(2.0*x - y));
  }
  double dpp_dx(const double &x, const double &y, const double &) const
  {
    return cos((2.0*M_PI/3.0)*(2.0*x - y))*2.0*(2.0*M_PI/3.0);
  }
  double ddpp_dxdx(const double &x, const double &y, const double &) const
  {
    return -sin((2.0*M_PI/3.0)*(2.0*x-y))*SQR(2.0*(2.0*M_PI/3.0));
  }
  double dpp_dy(const double &x, const double &y, const double &) const
  {
    return cos((2.0*M_PI/3.0)*(2.0*x - y))*(-2.0*M_PI/3.0);
  }
  double ddpp_dydy(const double &x, const double &y, const double &) const
  {
    return -sin((2.0*M_PI/3.0)*(2.0*x - y))*SQR(-2.0*M_PI/3.0);
  }
  double dpp_dz(const double &, const double &, const double &) const { return 0.0; }
  double ddpp_dzdz(const double &, const double &, const double &) const  { return 0.0; }

  double qq(const double &, const double &y, const double &z) const
  {
    return 1.5 + cos((2.0*M_PI/3.0)*(2.0*y - z));
  }
  double dqq_dx(const double &, const double &, const double &) const
  {
    return 0.0;
  }
  double ddqq_dxdx(const double &, const double &, const double &) const
  {
    return 0.0;
  }
  double dqq_dy(const double &, const double &y, const double &z) const
  {
    return -sin((2.0*M_PI/3.0)*(2.0*y - z))*(2.0*(2.0*M_PI/3.0));
  }
  double ddqq_dydy(const double &, const double &y, const double &z) const
  {
    return -cos((2.0*M_PI/3.0)*(2.0*y - z))*SQR(2.0*(2.0*M_PI/3.0));
  }
  double dqq_dz(const double &, const double &y, const double &z) const
  {
    return sin((2.0*M_PI/3.0)*(2.0*y - z))*(2.0*M_PI/3.0);
  }
  double ddqq_dzdz(const double &, const double &y, const double &z) const
  {
    return -cos((2.0*M_PI/3.0)*(2.0*y - z))*SQR(2.0*M_PI/3.0);
  }

  double rr(const double &x, const double &y, const double &) const
  {
    return cos((2.0*M_PI/3.0)*(2.0*x + y));
  }
  double drr_dx(const double &x, const double &y, const double &) const
  {
    return -sin((2.0*M_PI/3.0)*(2.0*x + y))*(2.0*2.0*M_PI/3.0);
  }
  double ddrr_dxdx(const double &x, const double &y, const double &) const
  {
    return -cos((2.0*M_PI/3.0)*(2.0*x + y))*SQR(2.0*2.0*M_PI/3.0);
  }
  double drr_dy(const double &x, const double &y, const double &) const
  {
    return -sin((2.0*M_PI/3.0)*(2.0*x + y))*(2.0*M_PI/3.0);
  }
  double ddrr_dydy(const double &x, const double &y, const double &) const
  {
    return -cos((2.0*M_PI/3.0)*(2.0*x + y))*SQR(2.0*M_PI/3.0);
  }
  double drr_dz(const double &, const double &, const double &) const { return 0.0; }
  double ddrr_dzdz(const double &, const double &, const double &) const { return 0.0; }

  double ss(const double &x, const double &, const double &z) const
  {
    return 0.5*sin((2.0*M_PI/3.0)*(2.0*z - x));
  }
  double dss_dx(const double &x, const double &, const double &z) const
  {
    return 0.5*cos((2.0*M_PI/3.0)*(2.0*z - x))*(-2.0*M_PI/3.0);
  }
  double ddss_dxdx(const double &x, const double &, const double &z) const
  {
    return -0.5*sin((2.0*M_PI/3.0)*(2.0*z - x))*SQR(-2.0*M_PI/3.0);
  }
  double dss_dy(const double &, const double &, const double &) const { return 0.0;  }
  double ddss_dydy(const double &x, const double &y, const double &) const { return 0.0; }
  double dss_dz(const double &x, const double &, const double &z) const
  {
    return 0.5*cos((2.0*M_PI/3.0)*(2.0*z - x))*(2.0*2.0*M_PI/3.0);
  }
  double ddss_dzdz(const double &x, const double &, const double &z) const
  {
    return -0.5*sin((2.0*M_PI/3.0)*(2.0*z - x))*SQR(2.0*2.0*M_PI/3.0);
  }

public:
  xGFM_example_full_periodic_3D_t()
  {
    mu_m = 1.0;
    mu_p = 80.0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -1.5;
      domain.xyz_max[dim] = +1.5;
      domain.periodicity[dim] = 1;
    }
    const double center[3] = {domain.xyz_min[0] + 0.15*.5*sqrt(2.0)*domain.length(),
                              domain.xyz_min[1] + 0.15*.5*sqrt(2.0)*domain.height(),
                              domain.xyz_min[2] + 0.2*domain.width()};
    level_set = new level_set_bone_shaped(domain, center, true, 0.6, 0.3, 0.7, 0.07, 0.2);

    solution_integral =  -0.164222868617700; // using Richardson's extrapolation between uniform 512x512x512 and 1024x1024x1024 grids (assuming second-order accurate integration)

    description =
        std::string("* domain = [-1.5, 1.5] X [-1.5, 1.5] X [-1.5, 1.5] \n")
        + std::string("* interface = revolution of the bone-shaped planar level-set around the z-axis, \n")
        + std::string("centered at (xmin + 0.15*.5*sqrt(2.0)*x_length, ymin + 0.15*.5*sqrt(2.0)*y_length, zmin + 0.20*z_length). \n")
        + std::string("negative inside, positive outside \n")
        + std::string("* mu_m = 1.0; \n")
        + std::string("* mu_p = 80.0; \n")
        + std::string("* u_m  = atan(sin((2.0*M_PI/3.0)*(2.0*x - y)))*log(1.5 + cos((2.0*M_PI/3.0)*(2.0*y - z))); \n")
        + std::string("* u_p  = tanh(cos((2.0*M_PI/3.0)*(2.0*x + y)))*acos(0.5*sin((2.0*M_PI/3.0)*(2.0*z - x))); \n")
        + std::string("* full periodicity is enforced \n")
        + std::string("Example for full periodicity in 3D (by R. Egan)");
  }

  double solution_minus(const double &x, const double &y, const double &z) const
  {
    return  atan(pp(x, y, z))*log(qq(x, y, z));
  }

  double first_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (dpp_dx(x, y, z)/(1.0 + SQR(pp(x, y, z))))*log(qq(x, y, z)) + atan(pp(x, y, z))*dqq_dx(x, y, z)/qq(x, y, z);
      break;
    case dir::y:
      return (dpp_dy(x, y, z)/(1.0 + SQR(pp(x, y, z))))*log(qq(x, y, z)) + atan(pp(x, y, z))*dqq_dy(x, y, z)/qq(x, y, z);
      break;
    case dir::z:
      return (dpp_dz(x, y, z)/(1.0 + SQR(pp(x, y, z))))*log(qq(x, y, z)) + atan(pp(x, y, z))*dqq_dz(x, y, z)/qq(x, y, z);
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_3D_t::first_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_minus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (ddpp_dxdx(x, y, z)*(1.0 + SQR(pp(x, y, z))) - 2.0*pp(x, y, z)*SQR(dpp_dx(x, y, z)))/SQR(1.0 + SQR(pp(x, y, z)))*log(qq(x, y, z)) + 2.0*(dpp_dx(x, y, z)/(1.0 + SQR(pp(x, y, z))))*(dqq_dx(x, y, z)/qq(x, y, z)) + atan(pp(x, y, z))*(ddqq_dxdx(x, y, z)*qq(x, y, z) - SQR(dqq_dx(x, y, z)))/SQR(qq(x, y, z));
      break;
    case dir::y:
      return (ddpp_dydy(x, y, z)*(1.0 + SQR(pp(x, y, z))) - 2.0*pp(x, y, z)*SQR(dpp_dy(x, y, z)))/SQR(1.0 + SQR(pp(x, y, z)))*log(qq(x, y, z)) + 2.0*(dpp_dy(x, y, z)/(1.0 + SQR(pp(x, y, z))))*(dqq_dy(x, y, z)/qq(x, y, z)) + atan(pp(x, y, z))*(ddqq_dydy(x, y, z)*qq(x, y, z) - SQR(dqq_dy(x, y, z)))/SQR(qq(x, y, z));
      break;
    case dir::z:
      return (ddpp_dzdz(x, y, z)*(1.0 + SQR(pp(x, y, z))) - 2.0*pp(x, y, z)*SQR(dpp_dz(x, y, z)))/SQR(1.0 + SQR(pp(x, y, z)))*log(qq(x, y, z)) + 2.0*(dpp_dz(x, y, z)/(1.0 + SQR(pp(x, y, z))))*(dqq_dz(x, y, z)/qq(x, y, z)) + atan(pp(x, y, z))*(ddqq_dzdz(x, y, z)*qq(x, y, z) - SQR(dqq_dz(x, y, z)))/SQR(qq(x, y, z));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_3D_t::second_derivative_solution_minus(): unknown differentiation direction");
      break;
    }
  }

  double solution_plus(const double &x, const double &y, const double &z) const
  {
    return tanh(rr(x, y, z))*acos(ss(x, y, z));
  }

  double first_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (1.0 - SQR(tanh(rr(x, y, z))))*drr_dx(x, y, z)*acos(ss(x, y, z)) + tanh(rr(x, y, z))*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dx(x, y, z);
      break;
    case dir::y:
      return (1.0 - SQR(tanh(rr(x, y, z))))*drr_dy(x, y, z)*acos(ss(x, y, z)) + tanh(rr(x, y, z))*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dy(x, y, z);
      break;
    case dir::z:
      return (1.0 - SQR(tanh(rr(x, y, z))))*drr_dz(x, y, z)*acos(ss(x, y, z)) + tanh(rr(x, y, z))*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dz(x, y, z);
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_3D_t::first_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }

  double second_derivative_solution_plus(const unsigned char &der, const double &x, const double &y, const double &z) const
  {
    switch (der) {
    case dir::x:
      return (-2.0*tanh(rr(x, y, z))*(1.0 - SQR(tanh(rr(x, y, z))))*SQR(drr_dx(x, y, z)) + (1.0 - SQR(tanh(rr(x, y, z))))*ddrr_dxdx(x, y, z))*acos(ss(x, y, z))
          + 2.0*(1.0 - SQR(tanh(rr(x, y, z))))*drr_dx(x, y, z)*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dx(x, y, z)
          - tanh(rr(x, y, z))*((ddss_dxdx(x, y, z)*sqrt(1.0 - SQR(ss(x, y, z))) + SQR(dss_dx(x, y, z))*ss(x, y, z)/sqrt(1.0 - SQR(ss(x, y, z))))/(1.0 - SQR(ss(x, y, z))));
      break;
    case dir::y:
      return (-2.0*tanh(rr(x, y, z))*(1.0 - SQR(tanh(rr(x, y, z))))*SQR(drr_dy(x, y, z)) + (1.0 - SQR(tanh(rr(x, y, z))))*ddrr_dydy(x, y, z))*acos(ss(x, y, z))
          + 2.0*(1.0 - SQR(tanh(rr(x, y, z))))*drr_dy(x, y, z)*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dy(x, y, z)
          - tanh(rr(x, y, z))*((ddss_dydy(x, y, z)*sqrt(1.0 - SQR(ss(x, y, z))) + SQR(dss_dy(x, y, z))*ss(x, y, z)/sqrt(1.0 - SQR(ss(x, y, z))))/(1.0 - SQR(ss(x, y, z))));
      break;
    case dir::z:
      return (-2.0*tanh(rr(x, y, z))*(1.0 - SQR(tanh(rr(x, y, z))))*SQR(drr_dz(x, y, z)) + (1.0 - SQR(tanh(rr(x, y, z))))*ddrr_dzdz(x, y, z))*acos(ss(x, y, z))
          + 2.0*(1.0 - SQR(tanh(rr(x, y, z))))*drr_dz(x, y, z)*(-1.0/sqrt(1.0 - SQR(ss(x, y, z))))*dss_dz(x, y, z)
          - tanh(rr(x, y, z))*((ddss_dzdz(x, y, z)*sqrt(1.0 - SQR(ss(x, y, z))) + SQR(dss_dz(x, y, z))*ss(x, y, z)/sqrt(1.0 - SQR(ss(x, y, z))))/(1.0 - SQR(ss(x, y, z))));
      break;
    default:
      throw  std::invalid_argument("xGFM_example_full_periodic_3D_t::second_derivative_solution_plus(): unknown differentiation direction");
      break;
    }
  }
} xGFM_example_full_periodic_3D;
#endif

#endif // SCALAR_TESTS_H
