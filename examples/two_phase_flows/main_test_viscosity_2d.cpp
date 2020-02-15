// System
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif


#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

static const double xmin = -1.25;
static const double xmax = +1.25;
static const double ymin = -1.25;
static const double ymax = +1.25;
#ifdef P4_TO_P8
static const double zmin = -1.25;
static const double zmax = +1.25;
#endif

static const double r0   = 0.5;
static const double uniform_band_factor = 0.15;

enum shape_t {
  CIRCLE,
  FLOWER
};

static const int lmin_  = 3;
static const int lmax_  = 5;
static const int nx_    = 1;
static const int ny_    = 1;
#ifdef P4_TO_P8
static const int nz_    = 1;
#endif
static const double rho             = 1.0;
static const double mu_minus_       = 1.0;
static const double mu_plus_        = 20.0;
static const bool implicit_         = true;
static const bool voro_fly_         = false;
static const unsigned int ngrids_   = 3;
static const bool save_vtk_         = true;
//static const bool save_vtk_         = false;
const static double duration_       = 1.0;
static const double vtk_dt_         = 0.05*duration_;
static const bool print_            = false;
static const shape_t shape_         = FLOWER;
static double tn;
static double dt;

class LEVEL_SET_GRID : public CF_DIM {
  shape_t shape;
public:
  LEVEL_SET_GRID() { lip = 1.2; shape = CIRCLE;}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (shape) {
    case CIRCLE:
      return r0 - sqrt(SUMD(SQR(x - (xmax + xmin)/2.0), SQR(y - (ymax + ymin)/2.0), SQR(z - (zmax + zmin)/2.0)));
      break;
    case FLOWER:
    {
#ifdef P4_TO_P8
      double phi, theta;
      if(fabs(sqrt(SQR(x) + SQR(y)) < EPS*MAX(xmax - xmin, ymax - ymin)))
        phi = M_PI_2;
      else
        phi = acos(x/sqrt(SQR(x) + SQR(y))) + (y > 0.0 ? 0.0 : M_PI);
      theta = (sqrt(SQR(x) + SQR(y) + SQR(z)) > EPS*MAX(xmax - xmin, ymax - ymin, zmax - zmin) ? acos(z/sqrt(SQR(x) + SQR(y) + SQR(z))) : 0.0);
      return SQR(x) + SQR(y) + SQR(z) - SQR(0.7 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0 - cos(6.0*theta)));
#else
      double alpha = 0.02*sqrt(5.0);
      double tt;
      if(fabs(x  -alpha) < EPS*(xmax - xmin))
        tt = (y > alpha + EPS*(ymax - ymin) ? 0.5*M_PI : (y < alpha - EPS*(ymax - ymin) ? -0.5*M_PI : 0.0));
      else
        tt = (x > alpha + EPS*(xmax - xmin) ? atan((y - alpha)/(x - alpha)) : ( y >= alpha ? M_PI + atan((y - alpha)/(x - alpha)) : -M_PI + atan((y - alpha)/(x - alpha))));
      return SQR(x - alpha) + SQR(y - alpha) - SQR(0.5 + 0.2*sin(5*tt));
#endif
      break;
    }
    default:
      throw std::invalid_argument("main_test_viscosity: choose a valid level set.");
      break;
    }
  }
  double operator()(const double *xyz) const
  {
    return operator()( DIM(xyz[0], xyz[1], xyz[2]));
  }

  inline void set_shape(const shape_t &shape_in) { shape = shape_in; }
  inline shape_t get_shape() const { return  shape; }

} level_set_grid;

class LEVEL_SET: public CF_DIM {
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return level_set_grid(DIM(x, y, z));
//    return -1.0;
  }
  double operator()(const double *xyz) const
  {
    return operator()( DIM(xyz[0], xyz[1], xyz[2]));
  }
} level_set;

class EXACT_SOLUTION{
  double TT;
  bool implicit;
  double mu_minus;
  double mu_plus;
public:
  inline double u_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*MULTD(sin(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double dt_u_minus(const double *xyz)
  {
    return -(2.0*M_PI/TT)*sin((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*MULTD(sin(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double dx_u_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*MULTD(cos(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double dy_u_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*MULTD(sin(xyz[0]), -sin(xyz[1]), sin(xyz[2]));
  }
#ifdef P4_TO_P8
  inline double dz_u_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*MULTD(sin(xyz[0]), cos(xyz[1]), cos(xyz[2]));
  }
#endif
  inline double laplace_u_minus(const double *xyz)
  {
    return -cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*((double) P4EST_DIM)*MULTD(sin(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double v_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*(3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5));
  }
  inline double dt_v_minus(const double *xyz)
  {
    return -(2.0*M_PI/TT)*sin((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*(3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5));
  }
  inline double dx_v_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*(9.0*SQR(xyz[0]) - 2.5*pow(xyz[0], 4))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5));
  }
  inline double dy_v_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*(3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*(2.0*xyz[1]/(1.0 + SQR(xyz[1])))ONLY3D(*atan(xyz[2]/2.5));
  }
#ifdef P4_TO_P8
  inline double dz_v_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*(3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*(1.0/2.5)/(1.0 + SQR(xyz[2]/2.5)));
  }
#endif
  inline double laplace_v_minus(const double *xyz)
  {
    return cos((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)*((18.0*xyz[0] - 10*pow(xyz[0], 3.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5))
        + (3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*(2.0*(1.0 - SQR(xyz[1]))/(SQR(1.0 + SQR(xyz[1]))))ONLY3D(*atan(xyz[2]/2.5))
        ONLY3D(+ (3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))*(-2.0*xyz[2]/(pow(2.5, 3.0)))/(SQR(1.0 + SQR(xyz[2]/2.5))))
        );
  }

#ifdef P4_TO_P8
  inline double w_minus(const double *xyz)
  {
    return (0.3 + 1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT))))*sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2]));
  }
  inline double dt_w_minus(const double *xyz)
  {
    return (1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT)))*sin(2.0*1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT)*1.4*2.0*M_PI/TT)*sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2]));
  }
  inline double dx_w_minus(const double *xyz)
  {
    return (0.3 + 1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT))))*(0.5*cos(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + sin(0.5*(xyz[0] - xyz[2]))*(sin(xyz[1]) + sin(xyz[0] + xyz[1])*atan(xyz[2])));
  }
  inline double dy_w_minus(const double *xyz)
  {
    return (0.3 + 1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT))))*sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*cos(xyz[1]) + sin(xyz[0] + xyz[1])*atan(xyz[2]));
  }
  inline double dz_w_minus(const double *xyz)
  {
    return (0.3 + 1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT))))*(-0.5*cos(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + sin(0.5*(xyz[0] - xyz[2]))*(-cos(xyz[0] + xyz[1])*(1.0/(1.0 + SQR(xyz[2])))));
  }
  inline double laplace_w_minus(const double *xyz)
  {
    return (0.3 + 1.7*exp(-SQR(cos(1.4*2.0*M_PI*(tn + (implicit ? dt : 0.0))/TT))))*(
          (-0.25*sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + cos(0.5*(xyz[0] - xyz[2]))*(sin(xyz[1]) + sin(xyz[0] + xyz[1])*atan(xyz[2])) + sin(0.5*(xyz[0] - xyz[2]))*cos(xyz[0] + xyz[1])*atan(xyz[2])) // dxx
        + (sin(0.5*(xyz[0] - xyz[2]))*(-xyz[0]*sin(xyz[1]) + cos(xyz[0] + xyz[1])*atan(xyz[2]))) // dyy
        + (-0.25*sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + cos(0.5*(xyz[0] - xyz[2]))*cos(xyz[0] + xyz[1])*(1.0/(1.0 + SQR(xyz[2]))) + sin(0.5*(xyz[0] - xyz[2]))*cos(xyz[0] + xyz[1])*(2.0*xyz[2]/SQR(1.0 + SQR(xyz[2]))))); // dzz
  }
#endif

  inline double u_plus(const double *xyz)
  {
    return (1.0 + exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT)))*(pow((xyz[1] - xyz[0])/2.5, 3.0) + cos(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(xyz[2] - 2.0*xyz[0] + cos(3.0*xyz[2] - xyz[1]))));
  }
  inline double dt_u_plus(const double *xyz)
  {
    return (exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT))*(-2.0*(tn + (implicit ? dt : 0.0) - 0.5*TT)/SQR(TT)))*(pow((xyz[1] - xyz[0])/2.5, 3.0) + cos(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(xyz[2] - 2.0*xyz[0] + cos(3.0*xyz[2] - xyz[1]))));
  }
  inline double dx_u_plus(const double *xyz)
  {
    return (1.0 + exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT)))*(-3.0*SQR((xyz[1] - xyz[0])/2.5)*(1.0/2.5) - 2.0*sin(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(-2.0)));
  }
  inline double dy_u_plus(const double *xyz)
  {
    return (1.0 + exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT)))*(3.0*SQR((xyz[1] - xyz[0])/2.5)*(1.0/2.5) + sin(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*sin(3.0*xyz[2] - xyz[1])));
  }
#ifdef P4_TO_P8
  inline double dz_u_plus(const double *xyz)
  {
    return (1.0 + exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT)))*(5.0/3.0)*(1.0 - 3.0*sin(3.0*xyz[2] - xyz[1]));
  }
#endif
  inline double laplace_u_plus(const double *xyz)
  {
    return (1.0 + exp(-SQR((tn + (implicit ? dt : 0.0) - 0.5*TT)/TT)))*(12.0*SQR(1.0/2.5)*((xyz[1] - xyz[0])/2.5) - (SQR(2.0) + SQR(1.0))*cos(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(-(SQR(3.0) + SQR(1.0))*cos(3.0*xyz[2] - xyz[1]))));
  }

  inline double v_plus(const double *xyz)
  {
    return (1.0 + 0.3*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(sin(3.0*xyz[0] - 2.0*xyz[1]) + log(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + cos(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dt_v_plus(const double *xyz)
  {
    return (-0.3*1.5*(2.0*M_PI/TT)*sin(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(sin(3.0*xyz[0] - 2.0*xyz[1]) + log(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + cos(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dx_v_plus(const double *xyz)
  {
    return (1.0 + 0.3*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(3.0*cos(3.0*xyz[0] - 2.0*xyz[1]) + (-2.0*1.2*(0.5*xyz[1] - 1.2*xyz[0]))/(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + 0.3*sin(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dy_v_plus(const double *xyz)
  {
    return (1.0 + 0.3*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(-2.0*cos(3.0*xyz[0] - 2.0*xyz[1]) + 2.0*0.5*(0.5*xyz[1] - 1.2*xyz[0])/(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])));
  }
#ifdef P4_TO_P8
  inline double dz_v_plus(const double *xyz)
  {
    return (1.0 + 0.3*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(- 1.7*sin(1.7*xyz[2] - 0.3*xyz[0]));
  }
#endif
  inline double laplace_v_plus(const double *xyz)
  {
    return (1.0 + 0.3*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*((-SQR(3.0) - SQR(-2.0))*sin(3.0*xyz[0] - 2.0*xyz[1]) +
        (2.0*(SQR(-1.2) + SQR(0.5))*(1.0 - SQR(0.5*xyz[1] - 1.2*xyz[0])))/(SQR(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])))
        ONLY3D(+ (-SQR(1.7) - SQR(0.3))*cos(1.7*xyz[2] - 0.3*xyz[0])));
  }

#ifdef P4_TO_P8
  inline double w_plus(const double *xyz)
  {
    return (1.0 + 0.7*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(0.1*xyz[0]*xyz[0]*xyz[0]*xyz[1] + 2.0*xyz[2]*cos(xyz[1]) - xyz[1]*sin(xyz[0] + xyz[2]));
  }
  inline double dt_w_plus(const double *xyz)
  {
    return (-0.7*1.5*(2.0*M_PI/TT)*sin(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(0.1*xyz[0]*xyz[0]*xyz[0]*xyz[1] + 2.0*xyz[2]*cos(xyz[1]) - xyz[1]*sin(xyz[0] + xyz[2]));
  }
  inline double dx_w_plus(const double *xyz)
  {
    return (1.0 + 0.7*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(0.3*xyz[0]*xyz[0]*xyz[1] - xyz[1]*cos(xyz[0] + xyz[2]));
  }
  inline double dy_w_plus(const double *xyz)
  {
    return (1.0 + 0.7*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(0.1*xyz[0]*xyz[0]*xyz[0] - 2.0*xyz[2]*sin(xyz[1]) - sin(xyz[0] + xyz[2]));
  }
  inline double dz_w_plus(const double *xyz)
  {
    return (1.0 + 0.7*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(2.0*cos(xyz[1]) - xyz[1]*cos(xyz[0] + xyz[2]));
  }
  inline double laplace_w_plus(const double *xyz)
  {
    return (1.0 + 0.7*cos(1.5*((tn + (implicit ? dt : 0.0))*2.0*M_PI/TT)))*(
          (0.6*xyz[0]*xyz[1] + xyz[1]*sin(xyz[0] + xyz[2])) // dxx
        + (-2.0*xyz[2]*cos(xyz[1])) // dyy
        + (xyz[1]*sin(xyz[0] + xyz[2])));
  }
#endif

  inline double source_term(const unsigned char &dir, const double *xyz)
  {
    switch (dir) {
    case dir::x:
      return (level_set(xyz) > 0.0 ? rho*dt_u_plus(xyz) - mu_plus*laplace_u_plus(xyz) : rho*dt_u_minus(xyz) - mu_minus*laplace_u_minus(xyz));
      break;
    case dir::y:
      return (level_set(xyz) > 0.0 ? rho*dt_v_plus(xyz) - mu_plus*laplace_v_plus(xyz) : rho*dt_v_minus(xyz) - mu_minus*laplace_v_minus(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (level_set(xyz) > 0.0 ? rho*dt_w_plus(xyz) - mu_plus*laplace_w_plus(xyz) : rho*dt_w_minus(xyz) - mu_minus*laplace_w_minus(xyz));
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::source_term: unknown cartesian direction");
    }
  }
  inline double jump_in_solution(const unsigned char &dir, const double *xyz)
  {
    switch (dir) {
    case dir::x:
      return (u_plus(xyz) - u_minus(xyz));
      break;
    case dir::y:
      return (v_plus(xyz) - v_minus(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (w_plus(xyz) - w_minus(xyz));
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::jump_in_solution: unknown cartesian direction");
    }
  }
  inline double jump_in_flux(const unsigned char &dir, const unsigned char &der, const double *xyz)
  {
    switch (dir) {
    case dir::x:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_u_plus(xyz) - mu_minus*dx_u_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_u_plus(xyz) - mu_minus*dy_u_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_u_plus(xyz) - mu_minus*dz_u_minus(xyz));
        break;
#endif
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
      }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_v_plus(xyz) - mu_minus*dx_v_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_v_plus(xyz) - mu_minus*dy_v_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_v_plus(xyz) - mu_minus*dz_v_minus(xyz));
        break;
#endif
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
      }
      break;
#ifdef P4_TO_P8
    case dir::z:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_w_plus(xyz) - mu_minus*dx_w_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_w_plus(xyz) - mu_minus*dy_w_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_w_plus(xyz) - mu_minus*dz_w_minus(xyz));
        break;
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
#endif
      }
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction");
    }
  }

  EXACT_SOLUTION() : implicit(false), mu_minus(1.0), mu_plus(1.0) {}

  inline void set_implicit() { implicit = true; }
  inline void set_viscosities(const double& mu_m_, const double& mu_p_) { mu_minus = mu_m_; mu_plus = mu_p_; }
  inline void set_time_scale(const double &TT_) { TT = TT_; return; }
  inline double get_time_scale() const { return TT; }

} exact_solution;

struct BCWALLTYPE_U : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

struct BCWALLVALUE_U : CF_DIM {
  double operator()(const double *xyz) const
  {
    return (level_set(xyz) <= 0.0 ? exact_solution.u_minus(xyz) : exact_solution.u_plus(xyz));
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return operator()(xyz);
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_DIM {
  double operator()(const double *xyz) const
  {
    return (level_set(xyz) <= 0.0 ? exact_solution.v_minus(xyz) : exact_solution.v_plus(xyz));
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return operator()(xyz);
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_3 {
  double operator()(const double *xyz) const
  {
    return (level_set(xyz) <= 0.0 ? exact_solution.w_minus(xyz) : exact_solution.w_plus(xyz));
  }
  double operator()(double x, double y, double z) const
  {
    const double xyz[P4EST_DIM] = {x, y, z};
    return operator()(xyz);
  }
} bc_wall_value_w;
#endif

void evaluate_errors(my_p4est_two_phase_flows_t *solver, double error_vnp1_minus[P4EST_DIM], double error_vnp1_plus[P4EST_DIM], double *error_at_faces_minus_p[P4EST_DIM], double *error_at_faces_plus_p[P4EST_DIM])
{
  PetscErrorCode ierr;
  Vec *vnp1_faces = solver->get_test_vnp1_faces();
  const double *vnp1_faces_p;
  double xyz_face[P4EST_DIM];
  my_p4est_faces_t* faces = solver->get_faces();
  my_p4est_interpolation_nodes_t* interp_phi = solver->get_interp_phi();
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    error_vnp1_minus[dir] = error_vnp1_plus[dir] = 0.0;
    ierr = VecGetArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
    for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx) {
      faces->xyz_fr_f(f_idx, dir, xyz_face);
      const double phi = (*interp_phi)(xyz_face);
      switch (dir) {
      case dir::x:
        if(phi <= 0.0)
        {
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.u_minus(xyz_face)));
          error_at_faces_minus_p[0][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.u_minus(xyz_face));
          error_at_faces_plus_p[0][f_idx] = 0.0;
        }
        else
        {
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.u_plus(xyz_face)));
          error_at_faces_minus_p[0][f_idx] = 0.0;
          error_at_faces_plus_p[0][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.u_plus(xyz_face));
        }
        break;
      case dir::y:
        if(phi <= 0.0)
        {
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.v_minus(xyz_face)));
          error_at_faces_minus_p[1][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.v_minus(xyz_face));
          error_at_faces_plus_p[1][f_idx] = 0.0;
        }
        else
        {
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.v_plus(xyz_face)));
          error_at_faces_minus_p[1][f_idx] = 0.0;
          error_at_faces_plus_p[1][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.v_plus(xyz_face));
        }
        break;
#ifdef P4_TO_P8
      case dir::z:
        if(phi <= 0.0)
        {
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.w_minus(xyz_face)));
          error_at_faces_minus_p[2][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.w_minus(xyz_face));
          error_at_faces_plus_p[2][f_idx] = 0.0;
        }
        else
        {
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.w_plus(xyz_face)));
          error_at_faces_minus_p[2][f_idx] = 0.0;
          error_at_faces_plus_p[2][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.w_plus(xyz_face));
        }
        break;
#endif
      default:
        throw std::runtime_error("evaluate_errors: unknown cartesian direction");
        break;
      }
    }
    ierr = VecRestoreArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
  }
  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_minus,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_plus,   P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
  return;
}

struct external_force_u_t : CF_DIM
{
  double operator ()(DIM(double x, double y, double z)) const
  {
    double xyz[P4EST_DIM] = { DIM(x, y, z) };
    return exact_solution.source_term(dir::x, xyz);
  }
} external_force_u;

struct external_force_v_t : CF_DIM
{
  double operator ()(DIM(double x, double y, double z)) const
  {
    double xyz[P4EST_DIM] = { DIM(x, y, z) };
    return exact_solution.source_term(dir::y, xyz);
  }
} external_force_v;

#ifdef P4_TO_P8
struct external_force_w_t : CF_DIM
{
  double operator ()(DIM(double x, double y, double z)) const
  {
    double xyz[P4EST_DIM] = { DIM(x, y, z) };
    return exact_solution.source_term(dir::z, xyz);
  }
} external_force_w;
#endif

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  // computational grid parameters
  cmd.add_option("lmin", "first min level of the trees, default is " + to_string(lmin_));
  cmd.add_option("lmax", "first max level of the trees, default is " + to_string(lmax_));
  cmd.add_option("nx", "number of trees in the x-direction. The default value is " + to_string(nx_) + " (length of domain is 2.5)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is " + to_string(nx_) + " (height of domain is 2.5)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is " + to_string(nx_) + " (width of domain is 2.5)");
#endif
  cmd.add_option("uniform_band_", "number of grid cells of uniform grid layering the interface on either side (default is such that the absolue value of phi is less than "  + to_string(uniform_band_factor*r0) + ")");
  // method/inner procedure control
  cmd.add_option("implicit",        "activates the implicit version if present (default is " + string(implicit_ ? "implicit)" : "explicit)"));
  cmd.add_option("explicit",        "deactivates the implicit version if present (default is " + string(implicit_ ? "implicit)" : "explicit)"));
  cmd.add_option("voro_on_the_fly", "activates the calculation of Voronoi cells on the fly (default is " + string(voro_fly_ ? "done on the fly)" : "stored in memory"));
  // physical parameters for the simulations
  cmd.add_option("mu_minus",  "viscosity coefficient in the negative domain (default is " + to_string(mu_minus_) + string(")"));
  cmd.add_option("mu_plus",   "viscosity coefficient in the positive domain (default is " + to_string(mu_plus_) + string(")"));
  cmd.add_option("duration",  "the duration of the simulation tfinal, tstart = 0.0, default duration is " + to_string(duration_) +".");
  cmd.add_option("shape",     "the shape of the interface (0: circle, 1: flower), default shape is " + string(shape_ == CIRCLE ? "circle)" : "flower.)" ));
  // exportation control
  cmd.add_option("save_vtk",      "saves vtk visualization files if present (default is " + string(save_vtk_ ? "" : "not ") + "saved)");
  cmd.add_option("no_save_vtk",   "does not save vtk visualization files if present (default is " + string(save_vtk_ ? "" : "not ") + "saved)");
  cmd.add_option("vtk_dt",        "time step between two vtk exportation, default duration is " + to_string(vtk_dt_));
  cmd.add_option("track_errors",  "prints the errors (sampled at faces) at every time step if present, only the final error is shown if not present)");
  cmd.add_option("ngrids",        "number of successively finer grids to consider (default is " + to_string(ngrids_) + ")");
  cmd.add_option("print",         "prints results and final convergence results in a file in the root exportation directory if present (default is "  + string(print_ ? "with" : "without") + " exportation)");

  string extra_info = "More details to come when I really have nothing better to do. If you really need it: reach out! (Raphael)";
  if(cmd.parse(argc, argv, extra_info))
    return 0;

  if(cmd.contains("explicit") && cmd.contains("implicit"))
    throw std::invalid_argument("Come on: choose between implicit and explicit, you dumbass user!");
  if(cmd.contains("save_vtk") && cmd.contains("no_save_vtk"))
    throw std::invalid_argument("Come on: do you want vtk exportation or not? Choose one!");
  if(cmd.contains("shape") && cmd.get<int>("shape") !=0 && cmd.get<int>("shape") !=1)
    throw std::invalid_argument("Come on: choose a valide shape for the interface...");


  int n_tree_xyz [P4EST_DIM];

  const bool implicit = !cmd.contains("explicit") && (implicit_ || cmd.contains("implicit"));
  if(implicit)
    exact_solution.set_implicit();
  string root_export_dir;
  if(getenv("OUTDIR") != NULL)
    root_export_dir = getenv("OUTDIR");
  else
#if defined(POD_CLUSTER)
    root_export_dir = "/scratch/regan/two_phase_flows/check_viscosity";
#elif defined(STAMPEDE)
    root_export_dir = "/scratch/04965/tg842642/two_phase_flows/check_viscosity";
#else
    root_export_dir = "/home/regan/workspace/projects/two_phase_flow/check_viscosity";
#endif
  root_export_dir += "/results_" + to_string(P4EST_DIM) + "d";

  const bool save_vtk                   = !cmd.contains("no_save_vtk") && (save_vtk_ || cmd.contains("save_vtk"));

  if(save_vtk && create_directory(root_export_dir.c_str(), mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
    sprintf(error_msg, "main_two_phase_flow_%dd: could not create the main exportation directory %s", P4EST_DIM, root_export_dir.c_str());
    throw std::runtime_error(error_msg);
  }

  double vtk_dt                         = -1.0;
  if(save_vtk)
  {
    vtk_dt = cmd.get<double>("vtk_dt", vtk_dt_);
    if(vtk_dt <= 0.0)
      throw std::invalid_argument("main_two_phase_flow_" + to_string(P4EST_DIM) + "d.cpp: the value of vtk_dt must be strictly positive.");
  }

  PetscErrorCode ierr;
  const double xyz_min [P4EST_DIM]  = { DIM(xmin, ymin, zmin) };
  const double xyz_max [P4EST_DIM]  = { DIM(xmax, ymax, zmax) };
  const double duration             = cmd.get<double>("duration", duration_);
  const bool track_errors           = cmd.contains("track_errors");
  const int periodic[P4EST_DIM]     = { DIM(0, 0, 0) };

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif

  int lmin              = cmd.get<int>("lmin", lmin_);
  int lmax              = cmd.get<int>("lmax", lmax_);
  n_tree_xyz[0]         = cmd.get<int>("nx", nx_);
  n_tree_xyz[1]         = cmd.get<int>("ny", ny_);
#ifdef P4_TO_P8
  n_tree_xyz[2]         = cmd.get<int>("nz", nz_);
#endif
  const double mu_minus = cmd.get<double>("mu_minus", mu_minus_);
  const double mu_plus  = cmd.get<double>("mu_plus", mu_plus_);
  exact_solution.set_viscosities(mu_minus, mu_plus);
  exact_solution.set_time_scale(0.5*duration);
  level_set_grid.set_shape((cmd.contains("shape") ? (cmd.get<int>("shape") == 0 ? CIRCLE : FLOWER) : shape_));

  const int ngrids = cmd.get<int>("ngrids", ngrids_);

  std::vector<double> convergence_error_minus_max_over_time[P4EST_DIM];
  std::vector<double> convergence_error_plus_max_over_time[P4EST_DIM];
  std::vector<double> convergence_error_minus_final_time[P4EST_DIM];
  std::vector<double> convergence_error_plus_final_time[P4EST_DIM];
  std::vector<double> computational_time;
  computational_time.resize(ngrids);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    convergence_error_minus_max_over_time[dir].resize(ngrids, 0.0);
    convergence_error_plus_max_over_time[dir].resize(ngrids, 0.0);
    convergence_error_minus_final_time[dir].resize(ngrids, 0.0);
    convergence_error_plus_final_time[dir].resize(ngrids, 0.0);
  }

  FILE *fp = NULL;
  const bool print_in_file = print_ || cmd.contains("print");
  if (!print_in_file)
    fp = stdout;
  else
  {
    if(create_directory(root_export_dir.c_str(), mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_two_phase_flow_%dd: could not create the main exportation directory %s", P4EST_DIM, root_export_dir.c_str());
      throw std::runtime_error(error_msg);
    }
    string level_set  = (level_set_grid.get_shape() == CIRCLE ? "circle" : "flower");
    string method     = (implicit ? "implicit" : "explicit");
    string filename = root_export_dir + string("/") + level_set + string("_") + method + string("_macromesh_") + to_string(n_tree_xyz[0]) + string("_") + to_string(n_tree_xyz[1]) ONLY3D(+ string("_") + to_string(n_tree_xyz[2])) + string("_lmin_") + to_string(lmin) + string("_lmax_") + to_string(lmax) + string("_ngrids_") + to_string(ngrids) + string("_mu_m_") + to_string(mu_minus) + string("_mu_p_") + to_string(mu_plus) + string(".dat") ;
    ierr = PetscFOpen(mpi.comm(), filename.c_str(), "w", &fp);
  }
  for (int k_grid = 0; k_grid < ngrids; ++k_grid) {
    if(k_grid > 0)
    {
      lmin++;
      lmax++;
    }

    my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
    my_p4est_brick_t* brick                           = NULL;
    p4est_connectivity_t *connectivity                = NULL;
    splitting_criteria_cf_and_uniform_band_t* data    = NULL;

    if(!implicit)
      dt = 0.5*0.5*SQR(MIN(DIM(xmax - xmin, ymax - ymin, zmax - zmin))/((double) (1<<lmax)))/MAX(mu_minus/rho, mu_plus/rho);
    else
      dt = 0.2*MIN(SQR(MIN(DIM(xmax - xmin, ymax - ymin, zmax - zmin)))*MIN(rho/mu_minus, rho/mu_plus), exact_solution.get_time_scale())*(1.0/((double) (1 << lmax))); // 0.2: arbitrary factor


    const double dxyzmin  = MAX(DIM((xmax - xmin)/(double)n_tree_xyz[0], (ymax - ymin)/(double)n_tree_xyz[1], (zmax - zmin)/(double)n_tree_xyz[2])) / (1 << lmax);
    double uniform_band   = .15*r0;
    uniform_band         /= dxyzmin;
    uniform_band          = cmd.get<double>("uniform_band_", uniform_band);

    brick = new my_p4est_brick_t;
    connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, brick, periodic);
    data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set_grid, uniform_band/*, 2.8*/);

    p4est_t* p4est_comp = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est_comp->user_pointer = (void*) data;

    for(int l = 0; l < lmax; ++l)
    {
      my_p4est_refine(p4est_comp, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
      my_p4est_partition(p4est_comp, P4EST_FALSE, NULL);
    }
    /* create the initial forest at time nm1 */
    p4est_balance(p4est_comp, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_comp, P4EST_FALSE, NULL);

    p4est_ghost_t *ghost_comp = my_p4est_ghost_new(p4est_comp, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_comp, ghost_comp);

    p4est_nodes_t *nodes_comp = my_p4est_nodes_new(p4est_comp, ghost_comp);
    my_p4est_hierarchy_t *hierarchy_comp = new my_p4est_hierarchy_t(p4est_comp, ghost_comp, brick);
    my_p4est_node_neighbors_t *ngbd_comp = new my_p4est_node_neighbors_t(hierarchy_comp, nodes_comp); ngbd_comp->init_neighbors();

    p4est_comp->user_pointer = (void*) data;
    my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_comp);
    my_p4est_faces_t *faces_comp = new my_p4est_faces_t(p4est_comp, ghost_comp, brick, ngbd_c, true);
    P4EST_ASSERT(faces_comp->finest_face_neighborhoods_are_valid());


    /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
     * the REINITIALIZED levelset on the interface-capturing grid
     */
    splitting_criteria_cf_t* data_fine = new splitting_criteria_cf_t(lmin, lmax + 1, &level_set_grid);
    p4est_t* p4est_fine = p4est_copy(p4est_comp, P4EST_FALSE);
    p4est_fine->user_pointer = (void*) data_fine;
    p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);
    p4est_ghost_t* ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_fine, ghost_fine);
    my_p4est_hierarchy_t* hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);
    p4est_nodes_t* nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
    my_p4est_node_neighbors_t* ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();

    Vec fine_phi;
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &fine_phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_fine, nodes_fine, level_set, fine_phi);
    if(level_set_grid.get_shape() == FLOWER)
    {
      my_p4est_level_set_t ls_fine(ngbd_n_fine);
      ls_fine.reinitialize_2nd_order(fine_phi);
    }

    two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_comp, ngbd_comp, faces_comp, ngbd_n_fine);
    bool second_order_phi = true;
    two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
    two_phase_flow_solver->set_dynamic_viscosities(mu_minus, mu_plus);
    two_phase_flow_solver->set_densities(rho, rho);
    two_phase_flow_solver->set_uniform_bands(uniform_band, uniform_band);
    two_phase_flow_solver->do_voronoi_computations_on_the_fly(voro_fly_ || cmd.contains("voro_on_the_fly"));
    CF_DIM *external_forces[P4EST_DIM] = { DIM(&external_force_u, &external_force_v, &external_force_w) };
    two_phase_flow_solver->set_external_forces(external_forces);
    two_phase_flow_solver->set_semi_lagrangian_order(2);

    // initialize face fields
    two_phase_flow_solver->initialize_viscosity_test_vectors();
    // time nm1

    tn = -(implicit ? 2.0 : 1.0)*dt; // -2.0 for "implicit" because the exact_solution has the +dt increment implemented (for correct jump evaluations thereafter)
    Vec* vnm1_faces = two_phase_flow_solver->get_test_vnm1_faces();
    double xyz_face[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      double *vnm1_faces_dir_p;
      ierr = VecGetArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
      for (p4est_locidx_t face_idx = 0; face_idx < faces_comp->num_local[dir] + faces_comp->num_ghost[dir]; ++face_idx) {
        faces_comp->xyz_fr_f(face_idx, dir, xyz_face);
        switch (dir) {
        case dir::x:
          vnm1_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.u_minus(xyz_face) : exact_solution.u_plus(xyz_face));
          break;
        case dir::y:
          vnm1_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.v_minus(xyz_face) : exact_solution.v_plus(xyz_face));
          break;
  #ifdef P4_TO_P8
        case dir::z:
          vnm1_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.w_minus(xyz_face) : exact_solution.w_plus(xyz_face));
          break;
  #endif
        default:
          throw std::runtime_error("main_test_viscosity: unknown directon for vnm1");
          break;
        }
      }
      ierr = VecRestoreArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
    }
    // time n
    tn += dt;
    Vec* vn_faces = two_phase_flow_solver->get_test_vn_faces();
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      double *vn_faces_dir_p;
      ierr = VecGetArray(vn_faces[dir], &vn_faces_dir_p); CHKERRXX(ierr);
      for (p4est_locidx_t face_idx = 0; face_idx < faces_comp->num_local[dir] + faces_comp->num_ghost[dir]; ++face_idx) {
        faces_comp->xyz_fr_f(face_idx, dir, xyz_face);
        switch (dir) {
        case dir::x:
          vn_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.u_minus(xyz_face) : exact_solution.u_plus(xyz_face));
          break;
        case dir::y:
          vn_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.v_minus(xyz_face) : exact_solution.v_plus(xyz_face));
          break;
  #ifdef P4_TO_P8
        case dir::z:
          vn_faces_dir_p[face_idx] = ((level_set(xyz_face) <= 0.0)? exact_solution.w_minus(xyz_face) : exact_solution.w_plus(xyz_face));
          break;
  #endif
        default:
          throw std::runtime_error("main_test_viscosity: unknown directon for vn");
          break;
        }
      }
      ierr = VecRestoreArray(vn_faces[dir], &vn_faces_dir_p); CHKERRXX(ierr);
    }

    if(implicit)
      tn += dt;

    two_phase_flow_solver->set_dt(dt, dt);
    two_phase_flow_solver->set_bc(bc_v, &bc_p);

    string export_dir = root_export_dir
        + string(level_set_grid.get_shape() == CIRCLE ? "/circle" : "/flower")
        + string(implicit ? "/implicit" : "/explicit")
        + "/macromesh_" + to_string(n_tree_xyz[0]) + "_" + to_string(n_tree_xyz[1]) ONLY3D(+ "_" + to_string(n_tree_xyz[2]))
        + "/mu_m_" + to_string(mu_minus) + "_mu_p_" + to_string(mu_plus)
        + "/lmin_" + to_string(lmin) + "_lmax_" + to_string(lmax);
    if(save_vtk && create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_two_phase_flow_%dd: could not create exportation directory %s", P4EST_DIM, export_dir.c_str());
      throw std::runtime_error(error_msg);
    }

    int iter = 0, iter_vtk = 0;

    // exportation stuff at nodes (if needed)
    Vec coarse_phi = NULL;
    const double *coarse_phi_p  = NULL;
    const double *fine_phi_p    = NULL;
    Vec exact_solution_minus  = NULL; double *exact_solution_minus_p  = NULL;
    Vec exact_solution_plus   = NULL; double *exact_solution_plus_p   = NULL;
    Vec error_at_node_minus[P4EST_DIM] = {DIM(NULL, NULL, NULL)}; double *error_at_node_minus_p[P4EST_DIM];
    Vec error_at_node_plus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};; double *error_at_node_plus_p[P4EST_DIM];
    if(save_vtk)
    {
      two_phase_flow_solver->interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, coarse_phi);
      ierr = VecCreateGhostNodesBlock(p4est_comp, nodes_comp, P4EST_DIM, &exact_solution_minus);  CHKERRXX(ierr);
      ierr = VecCreateGhostNodesBlock(p4est_comp, nodes_comp, P4EST_DIM, &exact_solution_plus);   CHKERRXX(ierr);
      ierr = VecGetArray(exact_solution_minus, &exact_solution_minus_p); CHKERRXX(ierr);
      ierr = VecGetArray(exact_solution_plus, &exact_solution_plus_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(coarse_phi, &coarse_phi_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        ierr = VecCreateGhostNodes(p4est_comp, nodes_comp, &error_at_node_minus[dir]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_comp, nodes_comp, &error_at_node_plus[dir]); CHKERRXX(ierr);
        ierr = VecGetArray(error_at_node_minus[dir], &error_at_node_minus_p[dir]); CHKERRXX(ierr);
        ierr = VecGetArray(error_at_node_plus[dir], &error_at_node_plus_p[dir]); CHKERRXX(ierr);
      }
    }

    // data-to-provide-to-the-solver stuff
    Vec fine_jump_mu_grad_v, fine_jump_u;
    double *fine_jump_mu_grad_v_p, *fine_jump_u_p;
    ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, SQR_P4EST_DIM, &fine_jump_mu_grad_v); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, P4EST_DIM, &fine_jump_u); CHKERRXX(ierr);
    two_phase_flow_solver->set_fine_jump_mu_grad_v(fine_jump_mu_grad_v);
    two_phase_flow_solver->set_fine_jump_velocity(fine_jump_u);
    double error_vnp1_minus[P4EST_DIM];
    double error_vnp1_plus[P4EST_DIM];
    double max_error_vnp1_minus[P4EST_DIM];
    double max_error_vnp1_plus[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      max_error_vnp1_minus[dir] = max_error_vnp1_plus[dir] = 0.0;

    // error measurement stuff
    Vec error_at_faces_minus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    Vec error_at_faces_plus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    double *error_at_faces_minus_p[P4EST_DIM], *error_at_faces_plus_p[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostFaces(p4est_comp, faces_comp, &error_at_faces_minus[dir], dir); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_comp, faces_comp, &error_at_faces_plus[dir], dir); CHKERRXX(ierr);
      ierr = VecGetArray(error_at_faces_minus[dir], &error_at_faces_minus_p[dir]);    CHKERRXX(ierr);
      ierr = VecGetArray(error_at_faces_plus[dir], &error_at_faces_plus_p[dir]);      CHKERRXX(ierr);
    }

    parStopWatch timer(parStopWatch::root_timings, fp);
    timer.start("Solving for grid " + to_string(lmin) + "/" + to_string(lmax) + " in " + to_string(P4EST_DIM) + " dimensions");
    while(tn + 0.01*dt < duration)
    {
      // set the jump conditions to what they are expected to be
      ierr = VecGetArray(two_phase_flow_solver->get_fine_jump_mu_grad_v(), &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
      ierr = VecGetArray(two_phase_flow_solver->get_fine_jump_velocity_test(), &fine_jump_u_p);     CHKERRXX(ierr);
      for (size_t fine_node_idx = 0; fine_node_idx < nodes_fine->indep_nodes.elem_count; ++fine_node_idx) {
        double xyz_node[P4EST_DIM]; node_xyz_fr_n(fine_node_idx, p4est_fine, nodes_fine, xyz_node);
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        {
          for (unsigned char der = 0; der < P4EST_DIM; ++der)
            fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_node_idx + P4EST_DIM*dir + der] = exact_solution.jump_in_flux(dir, der, xyz_node);
          fine_jump_u_p[P4EST_DIM*fine_node_idx + dir] = exact_solution.jump_in_solution(dir, xyz_node);
        }
      }
      ierr = VecRestoreArray(two_phase_flow_solver->get_fine_jump_mu_grad_v(), &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(two_phase_flow_solver->get_fine_jump_velocity_test(), &fine_jump_u_p);     CHKERRXX(ierr);

      if(!implicit)
      {
        two_phase_flow_solver->test_viscosity_explicit();
        tn += dt;
        // the bc objects were (and needed to be) set to tn, so the wall faces are not updated as they should. We fix that with
        two_phase_flow_solver->enforce_dirichlet_bc_on_test_vnp1_faces();
      }
      else
        two_phase_flow_solver->test_viscosity();

      evaluate_errors(two_phase_flow_solver, error_vnp1_minus, error_vnp1_plus, error_at_faces_minus_p, error_at_faces_plus_p);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        max_error_vnp1_minus[dir] = MAX(max_error_vnp1_minus[dir],  error_vnp1_minus[dir] );
        max_error_vnp1_plus[dir]  = MAX(max_error_vnp1_plus[dir],   error_vnp1_plus[dir]  );
      }

      if(implicit)
        tn += dt;

      two_phase_flow_solver->slide_face_fields();

      if(track_errors && (tn + 0.01*dt < duration))
      {
        ierr = PetscFPrintf(mpi.comm(), fp, "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t number of leaves = %d\n",
                            iter, tn, 100*tn/duration, two_phase_flow_solver->get_p4est()->global_num_quadrants); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_minus = %.5e \t error v_minus = %.5e \t error w_minus = %.5e \t \n",
                           error_vnp1_minus[0], error_vnp1_minus[1], error_vnp1_minus[2]); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_plus  = %.5e \t error v_plus  = %.5e \t error w_plus  = %.5e \t \n",
                           error_vnp1_plus[0], error_vnp1_plus[1], error_vnp1_plus[2]); CHKERRXX(ierr);
#else
        ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_minus = %.5e \t error v_minus = %.5e \t \n",
                           error_vnp1_minus[0], error_vnp1_minus[1]); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_plus  = %.5e \t error v_plus  = %.5e \t \n",
                           error_vnp1_plus[0], error_vnp1_plus[1]); CHKERRXX(ierr);
#endif
      }

      if(save_vtk && floor((tn + 0.01*dt)/vtk_dt) != iter_vtk)
      {
        iter_vtk = floor((tn + 0.01*dt)/vtk_dt);
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
          ierr = VecGhostUpdateBegin(error_at_faces_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(error_at_faces_plus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(error_at_faces_minus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(error_at_faces_plus[dir],    INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          for (size_t k = 0; k < ngbd_comp->get_layer_size(); ++k) {
            p4est_locidx_t node_idx = ngbd_comp->get_layer_node(k);
            error_at_node_minus_p[dir][node_idx] = interpolate_f_at_node_n(p4est_comp, ghost_comp, nodes_comp, faces_comp, ngbd_c, ngbd_comp, node_idx, error_at_faces_minus[dir], dir);
            error_at_node_plus_p[dir][node_idx] = interpolate_f_at_node_n(p4est_comp, ghost_comp, nodes_comp, faces_comp, ngbd_c, ngbd_comp, node_idx, error_at_faces_plus[dir], dir);
            double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_comp, nodes_comp, xyz_node);
            if(dir == 0)
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.u_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.u_plus(xyz_node);
            }
            else if(dir == 1)
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.v_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.v_plus(xyz_node);
            }
#ifdef P4_TO_P8
            else
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.w_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.w_plus(xyz_node);
            }
#endif
          }
          ierr = VecGhostUpdateBegin(error_at_node_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(error_at_node_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          for (size_t k = 0; k < ngbd_comp->get_local_size(); ++k) {
            p4est_locidx_t node_idx = ngbd_comp->get_local_node(k);
            error_at_node_minus_p[dir][node_idx] = interpolate_f_at_node_n(p4est_comp, ghost_comp, nodes_comp, faces_comp, ngbd_c, ngbd_comp, node_idx, error_at_faces_minus[dir], dir);
            error_at_node_plus_p[dir][node_idx] = interpolate_f_at_node_n(p4est_comp, ghost_comp, nodes_comp, faces_comp, ngbd_c, ngbd_comp, node_idx, error_at_faces_plus[dir], dir);
            double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_comp, nodes_comp, xyz_node);
            if(dir == 0)
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.u_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.u_plus(xyz_node);
            }
            else if(dir == 1)
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.v_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.v_plus(xyz_node);
            }
#ifdef P4_TO_P8
            else
            {
              exact_solution_minus_p[P4EST_DIM*node_idx + dir] = exact_solution.w_minus(xyz_node);
              exact_solution_plus_p[P4EST_DIM*node_idx + dir] = exact_solution.w_plus(xyz_node);
            }
#endif
          }
          ierr = VecGhostUpdateEnd(error_at_node_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(error_at_node_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
        ierr = VecGhostUpdateBegin(exact_solution_minus,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(exact_solution_plus,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(exact_solution_minus,    INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(exact_solution_plus,     INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        my_p4est_vtk_write_all_general(p4est_comp, nodes_comp, ghost_comp,
                                       P4EST_TRUE, P4EST_TRUE,
                                       1, 2, 2, 0, 0, 0, (export_dir + "/illustration_" + to_string(iter_vtk)).c_str(),
                                       VTK_NODE_SCALAR, "phi", coarse_phi_p,
                                       VTK_NODE_VECTOR_BY_COMPONENTS, "error_minus", DIM(error_at_node_minus_p[0], error_at_node_minus_p[1], error_at_node_minus_p[2]),
                                       VTK_NODE_VECTOR_BY_COMPONENTS, "error_plus", DIM(error_at_node_plus_p[0], error_at_node_plus_p[1], error_at_node_plus_p[2]),
                                       VTK_NODE_VECTOR_BLOCK, "u_minus", exact_solution_minus_p,
                                       VTK_NODE_VECTOR_BLOCK, "u_plus", exact_solution_plus_p);

        my_p4est_vtk_write_all(p4est_fine, nodes_fine, ghost_fine,
                               P4EST_TRUE, P4EST_TRUE,
                               1, 0, (export_dir + "/fine_illustration_" + to_string(iter_vtk)).c_str(),
                               VTK_NODE_SCALAR, "phi", fine_phi_p);
      }
      iter++;
    }
    timer.stop();
    computational_time[k_grid] = timer.read_duration();

    ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, "Final errors in infinity norm for the simulation with lmin/lmax = %d/%d \t (number of leaves = %d)\n",
                        lmin, lmax, two_phase_flow_solver->get_p4est()->global_num_quadrants);                                                                CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, "(error analysis in each subdomain in infinity norm;\n");                                                             CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, " maximum error over the full run of the simulation, i.e. max in time as well)\n");                                   CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_minus = %.5e \t error v_minus = %.5e \t error w_minus = %.5e \t(MAX over time)\n",
                       max_error_vnp1_minus[0], max_error_vnp1_minus[1], max_error_vnp1_minus[2]); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_plus  = %.5e \t error v_plus  = %.5e \t error w_plus  = %.5e \t(MAX over time)\n",
                       max_error_vnp1_plus[0], max_error_vnp1_plus[1], max_error_vnp1_plus[2]); CHKERRXX(ierr);
#else
    ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_minus = %.5e \t error v_minus = %.5e \t (MAX overall)\n",
                       max_error_vnp1_minus[0], max_error_vnp1_minus[1]); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fp, "\t error u_plus  = %.5e \t error v_plus  = %.5e \t (MAX overall)\n",
                       max_error_vnp1_plus[0], max_error_vnp1_plus[1]); CHKERRXX(ierr);
#endif


    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      convergence_error_minus_max_over_time[dir][k_grid] = max_error_vnp1_minus[dir];
      convergence_error_plus_max_over_time[dir][k_grid]  = max_error_vnp1_plus[dir];
      convergence_error_minus_final_time[dir][k_grid] = error_vnp1_minus[dir];
      convergence_error_plus_final_time[dir][k_grid]  = error_vnp1_plus[dir];
    }


    if(save_vtk)
    {
      ierr = VecRestoreArrayRead(coarse_phi, &coarse_phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(exact_solution_minus, &exact_solution_minus_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(exact_solution_plus, &exact_solution_plus_p); CHKERRXX(ierr);
      ierr = VecDestroy(exact_solution_minus); CHKERRXX(ierr);
      ierr = VecDestroy(exact_solution_plus); CHKERRXX(ierr);
      ierr = VecDestroy(coarse_phi); CHKERRXX(ierr);
    }
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(error_at_faces_minus[dir], &error_at_faces_minus_p[dir]);  CHKERRXX(ierr);
      ierr = VecRestoreArray(error_at_faces_plus[dir],  &error_at_faces_plus_p[dir]);   CHKERRXX(ierr);
      if(save_vtk)
      {
        ierr = VecRestoreArray(error_at_node_minus[dir],  &error_at_node_minus_p[dir]);   CHKERRXX(ierr);
        ierr = VecRestoreArray(error_at_node_plus[dir],   &error_at_node_plus_p[dir]);    CHKERRXX(ierr);
      }
      ierr = VecDestroy(error_at_faces_minus[dir]);                                     CHKERRXX(ierr);
      ierr = VecDestroy(error_at_faces_plus[dir]);                                      CHKERRXX(ierr);
      if(save_vtk)
      {
        ierr = VecDestroy(error_at_node_minus[dir]);                                      CHKERRXX(ierr);
        ierr = VecDestroy(error_at_node_plus[dir]);                                       CHKERRXX(ierr);
      }
    }


    delete two_phase_flow_solver;
    delete data;
    delete data_fine;
  }

  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "                                          CONVERGENCE SUMMARY                                                 \n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "                       (errors in each subdomain in infinity norm, max over time)                             \n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  lmin = cmd.get<int>("lmin", lmin_);
  lmax = cmd.get<int>("lmax", lmax_);
  ierr = PetscFPrintf(mpi.comm(), fp, "Grid levels (lmin/lmax): ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "      %d/%d  ", lmin+k_grid, lmax+k_grid);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in u_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_max_over_time[0][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in v_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_max_over_time[1][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in w_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_max_over_time[2][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#endif
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in u_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_max_over_time[0][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in v_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_max_over_time[1][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in w_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_max_over_time[2][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#endif

  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "                                          CONVERGENCE SUMMARY                                                 \n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "                    (errors in each subdomain in infinity norm, error at final time)                          \n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  lmin = cmd.get<int>("lmin", lmin_);
  lmax = cmd.get<int>("lmax", lmax_);
  ierr = PetscFPrintf(mpi.comm(), fp, "Grid levels (lmin/lmax): ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "      %d/%d  ", lmin+k_grid, lmax+k_grid);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in u_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_final_time[0][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in v_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_final_time[1][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in w_minus:  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_minus_final_time[2][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#endif
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in u_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_final_time[0][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in v_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_final_time[1][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscFPrintf(mpi.comm(), fp, "Max error in w_plus :  	 ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", convergence_error_plus_final_time[2][k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
#endif

  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "                                          COMPUTATIONAL TIME                                                  \n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "--------------------------------------------------------------------------------------------------------------\n");  CHKERRXX(ierr);
  lmin = cmd.get<int>("lmin", lmin_);
  lmax = cmd.get<int>("lmax", lmax_);
  ierr = PetscFPrintf(mpi.comm(), fp, "Grid levels (lmin/lmax): ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "      %d/%d  ", lmin+k_grid, lmax+k_grid);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);
  ierr = PetscFPrintf(mpi.comm(), fp, "Computational time (s):  ");  CHKERRXX(ierr);
  for (int k_grid = 0; k_grid < ngrids; ++k_grid){
    ierr = PetscFPrintf(mpi.comm(), fp, "%.3e  ", computational_time[k_grid]);  CHKERRXX(ierr);
  }
  ierr = PetscFPrintf(mpi.comm(), fp, "\n");  CHKERRXX(ierr);

  return 0;
}
