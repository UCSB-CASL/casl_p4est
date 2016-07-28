#include "shapes.h"

bool save_vtk = true;

double xmin = -1;
double xmax =  1;
double ymin = -1;
double ymax =  1;
double zmin = -1;
double zmax =  1;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int nb_splits = 3;
#else
int lmin = 4;
int lmax = 4;
int nb_splits = 5;
#endif

int nx = 1;
int ny = 1;
int nz = 1;

const int periodic[] = {0, 0, 0};

double mu = 1.;
double diag_add = 0.;

// GEOMETRY
#ifdef P4_TO_P8
double r0 =  0.757;
double r1 =  0.754;
double r2 =  0.631;
double r3 = -0.333;
double d  =  0.234;
#else
double r0 =  0.687;
double r1 =  0.594;
double r2 =  0.397;
double r3 = -0.333;
double d  =  0.134;
#endif

double theta0 = 0.682; double cosT0 = cos(theta0); double sinT0 = sin(theta0);
double theta1 = 0.323; double cosT1 = cos(theta1); double sinT1 = sin(theta1);

#ifdef P4_TO_P8
double x0 = -1.*d*cosT0*cosT1; double y0 =  1.*d*cosT0*cosT1; double z0 =  1.*d*sinT1;
double x1 =  1.*d*cosT0*cosT1; double y1 = -1.*d*cosT0*cosT1; double z1 = -1.*d*sinT1;
double x2 =  3.*d*cosT0*cosT1; double y2 =  3.*d*sinT0*cosT1; double z2 =  3.*d*sinT1;
double x3 = -4.*d*cosT0*cosT1; double y3 =  2.*d*cosT0*cosT1; double z3 =  1.*d*sinT1;
#else
double xc0 = -1.*d*sinT0; double yc0 =  1.*d*cosT0;
double xc1 =  1.*d*sinT0; double yc1 = -1.*d*cosT0;
double xc2 =  3.*d*cosT0; double yc2 =  3.*d*sinT0;
double xc3 = -4.*d*sinT0; double yc3 =  2.*d*cosT0;
#endif

double beta0 = 0.12; double inside0 = 1;
double beta1 = 0.12; double inside1 =  1;
double beta2 = 0.04; double inside2 = -1;
double beta3 = 0; double inside3 =  1;

flower_shaped_domain_t domain0(r0, xc0, yc0, beta0, inside0);
flower_shaped_domain_t domain1(r1, xc1, yc1, beta1, inside1);
flower_shaped_domain_t domain2(r2, xc2, yc2, beta2, inside2);
flower_shaped_domain_t domain3(r3, xc3, yc3, beta3, inside3);


// EXACT SOLUTION
int n_test = 0;

double phase_x =  0.4;
double phase_y = -0.3;
double phase_z =  0.7;

#ifdef P4_TO_P8
class U_EXACT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return sin(x)*cos(y)*exp(z);
    }
  }
} u_exact;
#else
class U_EXACT: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return (sin(PI*x+phase_x)*cos(PI*y+phase_y));
    }
  }
} u_exact;
#endif

// EXACT DERIVATIVES
#ifdef P4_TO_P8
class UX: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return cos(x)*cos(y)*exp(z);
    }
  }
} ux;
class UY: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return -sin(x)*sin(y)*exp(z);
    }
  }
} uy;
class UZ: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return sin(x)*cos(y)*exp(z);
    }
  }
} uz;
#else
class UX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return PI*cos(PI*x+phase_x)*cos(PI*y+phase_y);
    }
  }
} ux;
class UY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return -PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} uy;
#endif

// RHS
#ifdef P4_TO_P8
class RHS: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return mu*sin(x)*cos(y)*exp(z) + diag_add*u_exact(x,y,z);
    }
  }
} rhs_cf;
#else
class RHS: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return 2.0*PI*PI*mu*sin(PI*x+phase_x)*cos(PI*y+phase_y) + diag_add*u_exact(x,y);
    }
  }
} rhs_cf;
#endif

class double_to_cf_t : public CF_2
{
public:
  double (*func)(double, double);
  double_to_cf_t(double (*func)(double, double)) { this->func = func; }
  double operator()(double x, double y) const { return (*func)(x,y); }
};

// BC COEFFICIENTS
#ifdef P4_TO_P8
inline double kappa0(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff0(&kappa0);
inline double kappa1(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff1(&kappa1);
inline double kappa2(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff2(&kappa2);
inline double kappa3(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff3(&kappa3);
#else
inline double kappa0(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff0(&kappa0);
inline double kappa1(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff1(&kappa1);
inline double kappa2(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff2(&kappa2);
inline double kappa3(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff3(&kappa3);
#endif

// BC VALUES
class bc_value_robin_t : public CF_2
{
  CF_2 *u, *ux, *uy;
  CF_2 *phi_x, *phi_y;
  CF_2 *kappa;
public:
  bc_value_robin_t(CF_2 *u, CF_2 *ux, CF_2 *uy, CF_2 *phi_x, CF_2 *phi_y, CF_2 *kappa) :
    u(u), ux(ux), uy(uy), phi_x(phi_x), phi_y(phi_y), kappa(kappa) {}
  double operator()(double x, double y) const
  {
    double nx = (*phi_x)(x,y);
    double ny = (*phi_y)(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return nx*(*ux)(x,y) + ny*(*uy)(x,y) + (*kappa)(x,y)*(*u)(x,y);
  }
};

bc_value_robin_t bc_value0(&u_exact, &ux, &uy, &domain0.phi_x, &domain0.phi_y, &bc_coeff0);
bc_value_robin_t bc_value1(&u_exact, &ux, &uy, &domain1.phi_x, &domain1.phi_y, &bc_coeff1);
bc_value_robin_t bc_value2(&u_exact, &ux, &uy, &domain2.phi_x, &domain2.phi_y, &bc_coeff2);
bc_value_robin_t bc_value3(&u_exact, &ux, &uy, &domain0.phi_x, &domain0.phi_y, &bc_coeff3);

// GATHER EVERYTHING
class Problem {
public:
#ifdef P4_TO_P8
  std::vector<CF_3 *>   phi_cf;
#else
  std::vector<CF_2 *>   phi_cf;
#endif
  std::vector<action_t> action;
  std::vector<int>      color;

#ifdef P4_TO_P8
  std::vector<CF_3 *> bc_values;
  std::vector<CF_3 *> bc_coeffs;
#else
  std::vector<CF_2 *> bc_values;
  std::vector<CF_2 *> bc_coeffs;
#endif
  std::vector<BoundaryConditionType> bc_type;

  Problem()
  {
    // set geometry
    phi_cf.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(color.size());
    phi_cf.push_back(&domain1.phi); action.push_back(ADDITION);     color.push_back(color.size());
    phi_cf.push_back(&domain2.phi); action.push_back(INTERSECTION); color.push_back(color.size());
//    phi_cf.push_back(&domain3.phi); action.push_back(COLORATION);   color.push_back(color.size());

    // set BCs
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff0); bc_values.push_back(&bc_value0);
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff1); bc_values.push_back(&bc_value1);
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff2); bc_values.push_back(&bc_value2);
//    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff3); bc_values.push_back(&bc_value3);
  }
} problem;

class level_set_tot_t : public CF_2
{
  std::vector<CF_2 *>   *phi_cf;
  std::vector<action_t> *action;
  std::vector<int>      *color;
public:
  level_set_tot_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
    phi_cf(phi_cf), action(action), color(color) {}
  double operator()(double x, double y) const
  {
    double phi_total = -10;
    double phi_current = -10;
    for (short i = 0; i < color->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
        phi_current = (*phi_cf->at(i))(x,y);
        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
        phi_current = (*phi_cf->at(i))(x,y);
        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
  }
};

level_set_tot_t level_set_tot(&problem.phi_cf, &problem.action, &problem.color);
