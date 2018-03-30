#include "shapes.h"

bool save_vtk = true;
bool display_error = false;

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
int lmin = 5;
int lmax = 7;
int nb_splits = 3;
#endif

int nx = 1;
int ny = 1;
int nz = 1;

const int periodic[] = {0, 0, 0};

//double mu = 1.;
//double diag_add = 0.;

// GEOMETRY
#ifdef P4_TO_P8
double r0 =  0.7;
double r1 =  0.5;
double r2 =  0.6;
double r3 =  0.6;
double d  =  0.25;
#else
double r0 =  0.53;
double r1 =  0.4;
double r2 =  0.61;
double r3 =  0.5;
double d  =  0.25;
#endif

double theta0 = 0.882; double cosT0 = cos(theta0); double sinT0 = sin(theta0);
double theta1 = 0.623; double cosT1 = cos(theta1); double sinT1 = sin(theta1);

#ifdef P4_TO_P8
double xc0 = -1.*d*cosT0*cosT1; double yc0 =  1.*d*cosT0*cosT1; double zc0 =  1.*d*sinT1;
double xc1 = -3.*d*cosT0*cosT1; double yc1 =  3.*d*cosT0*cosT1; double zc1 =  3.*d*sinT1;
double xc2 =  1.*d*cosT0*cosT1; double yc2 = -1.*d*cosT0*cosT1; double zc2 = -1.*d*sinT1;
double xc3 =  3.*d*cosT0*cosT1; double yc3 =  3.*d*sinT0*cosT1; double zc3 =  3.*d*sinT1;
#else
double xc0 = -1.*d*sinT0; double yc0 =  1.*d*cosT0;
double xc1 = -3.*d*sinT0; double yc1 =  3.*d*cosT0;
double xc2 =  1.*d*sinT0; double yc2 = -1.*d*cosT0;
double xc3 =  3.*d*cosT0; double yc3 =  3.*d*sinT0;
#endif

double beta0 = 0.1; double inside0 =  1;
double beta1 = 0.1; double inside1 =  1;
double beta2 = 0.1; double inside2 =  1;
double beta3 = 0.13; double inside3 = -1;

double alpha0 = 0.3*PI;
double alpha1 = 0.1*PI;
double alpha2 = -0.14*PI;
double alpha3 = 0.3*PI;

double lx0 = 1, ly0 = 1, lz0 = 1;
double lx1 = -1, ly1 = 1, lz1 = 1;
double lx2 = 1, ly2 = -1, lz2 = 1;
double lx3 = 1, ly3 = 1, lz3 = -1;

#ifdef P4_TO_P8
flower_shaped_domain_t domain0(r0, xc0, yc0, zc0, beta0, inside0, lx0, ly0, lz0, alpha0);
flower_shaped_domain_t domain1(r1, xc1, yc1, zc1, beta1, inside1, lx1, ly1, lz1, alpha1);
flower_shaped_domain_t domain2(r2, xc2, yc2, zc2, beta2, inside2, lx2, ly2, lz2, alpha2);
flower_shaped_domain_t domain3(r3, xc3, yc3, zc3, beta3, inside3, lx3, ly3, lz3, alpha3);
#else
flower_shaped_domain_t domain0(r0, xc0, yc0, beta0, inside0, alpha0);
flower_shaped_domain_t domain1(r1, xc1, yc1, beta1, inside1, alpha1);
flower_shaped_domain_t domain2(r2, xc2, yc2, beta2, inside2, alpha2);
flower_shaped_domain_t domain3(r3, xc3, yc3, beta3, inside3, alpha3);
#endif

// cut corners
double r_corners    = 0.573;
double beta_corners = 0.04;

double delx =  0.079;
double dely = -0.123;
double delz =  0.057;

#ifdef P4_TO_P8
flower_shaped_domain_t corner_mmm(r_corners, xmin-delx, ymin-dely, zmin-delz, beta_corners, -1);
flower_shaped_domain_t corner_pmm(r_corners, xmax-delx, ymin-dely, zmin-delz, beta_corners, -1);
flower_shaped_domain_t corner_mpm(r_corners, xmin-delx, ymax-dely, zmin-delz, beta_corners, -1);
flower_shaped_domain_t corner_ppm(r_corners, xmax-delx, ymax-dely, zmin-delz, beta_corners, -1);
flower_shaped_domain_t corner_mmp(r_corners, xmin-delx, ymin-dely, zmax-delz, beta_corners, -1);
flower_shaped_domain_t corner_pmp(r_corners, xmax-delx, ymin-dely, zmax-delz, beta_corners, -1);
flower_shaped_domain_t corner_mpp(r_corners, xmin-delx, ymax-dely, zmax-delz, beta_corners, -1);
flower_shaped_domain_t corner_ppp(r_corners, xmax-delx, ymax-dely, zmax-delz, beta_corners, -1);

class phi_corners_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (x <= 0 && y <= 0 && z <= 0) return corner_mmm.phi(x,y,z);
    if (x >= 0 && y <= 0 && z <= 0) return corner_pmm.phi(x,y,z);
    if (x <= 0 && y >= 0 && z <= 0) return corner_mpm.phi(x,y,z);
    if (x >= 0 && y >= 0 && z <= 0) return corner_ppm.phi(x,y,z);
    if (x <= 0 && y <= 0 && z >= 0) return corner_mmp.phi(x,y,z);
    if (x >= 0 && y <= 0 && z >= 0) return corner_pmp.phi(x,y,z);
    if (x <= 0 && y >= 0 && z >= 0) return corner_mpp.phi(x,y,z);
    if (x >= 0 && y >= 0 && z >= 0) return corner_ppp.phi(x,y,z);
  }
} phi_corners;

class phi_x_corners_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (x <= 0 && y <= 0 && z <= 0) return corner_mmm.phi_x(x,y,z);
    if (x >= 0 && y <= 0 && z <= 0) return corner_pmm.phi_x(x,y,z);
    if (x <= 0 && y >= 0 && z <= 0) return corner_mpm.phi_x(x,y,z);
    if (x >= 0 && y >= 0 && z <= 0) return corner_ppm.phi_x(x,y,z);
    if (x <= 0 && y <= 0 && z >= 0) return corner_mmp.phi_x(x,y,z);
    if (x >= 0 && y <= 0 && z >= 0) return corner_pmp.phi_x(x,y,z);
    if (x <= 0 && y >= 0 && z >= 0) return corner_mpp.phi_x(x,y,z);
    if (x >= 0 && y >= 0 && z >= 0) return corner_ppp.phi_x(x,y,z);
  }
} phi_x_corners;

class phi_y_corners_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (x <= 0 && y <= 0 && z <= 0) return corner_mmm.phi_y(x,y,z);
    if (x >= 0 && y <= 0 && z <= 0) return corner_pmm.phi_y(x,y,z);
    if (x <= 0 && y >= 0 && z <= 0) return corner_mpm.phi_y(x,y,z);
    if (x >= 0 && y >= 0 && z <= 0) return corner_ppm.phi_y(x,y,z);
    if (x <= 0 && y <= 0 && z >= 0) return corner_mmp.phi_y(x,y,z);
    if (x >= 0 && y <= 0 && z >= 0) return corner_pmp.phi_y(x,y,z);
    if (x <= 0 && y >= 0 && z >= 0) return corner_mpp.phi_y(x,y,z);
    if (x >= 0 && y >= 0 && z >= 0) return corner_ppp.phi_y(x,y,z);
  }
} phi_y_corners;

class phi_z_corners_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (x <= 0 && y <= 0 && z <= 0) return corner_mmm.phi_z(x,y,z);
    if (x >= 0 && y <= 0 && z <= 0) return corner_pmm.phi_z(x,y,z);
    if (x <= 0 && y >= 0 && z <= 0) return corner_mpm.phi_z(x,y,z);
    if (x >= 0 && y >= 0 && z <= 0) return corner_ppm.phi_z(x,y,z);
    if (x <= 0 && y <= 0 && z >= 0) return corner_mmp.phi_z(x,y,z);
    if (x >= 0 && y <= 0 && z >= 0) return corner_pmp.phi_z(x,y,z);
    if (x <= 0 && y >= 0 && z >= 0) return corner_mpp.phi_z(x,y,z);
    if (x >= 0 && y >= 0 && z >= 0) return corner_ppp.phi_z(x,y,z);
  }
} phi_z_corners;
#else
flower_shaped_domain_t corner_mmm(r_corners, xmin-delx, ymin-dely, beta_corners, -1);
flower_shaped_domain_t corner_pmm(r_corners, xmax-delx, ymin-dely, beta_corners, -1);
flower_shaped_domain_t corner_mpm(r_corners, xmin-delx, ymax-dely, beta_corners, -1);
flower_shaped_domain_t corner_ppm(r_corners, xmax-delx, ymax-dely, beta_corners, -1);

class phi_corners_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (x <= 0 && y <= 0) return corner_mmm.phi(x,y);
    if (x >= 0 && y <= 0) return corner_pmm.phi(x,y);
    if (x <= 0 && y >= 0) return corner_mpm.phi(x,y);
    if (x >= 0 && y >= 0) return corner_ppm.phi(x,y);
  }
} phi_corners;

class phi_x_corners_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (x <= 0 && y <= 0) return corner_mmm.phi_x(x,y);
    if (x >= 0 && y <= 0) return corner_pmm.phi_x(x,y);
    if (x <= 0 && y >= 0) return corner_mpm.phi_x(x,y);
    if (x >= 0 && y >= 0) return corner_ppm.phi_x(x,y);
  }
} phi_x_corners;

class phi_y_corners_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (x <= 0 && y <= 0) return corner_mmm.phi_y(x,y);
    if (x >= 0 && y <= 0) return corner_pmm.phi_y(x,y);
    if (x <= 0 && y >= 0) return corner_mpm.phi_y(x,y);
    if (x >= 0 && y >= 0) return corner_ppm.phi_y(x,y);
  }
} phi_y_corners;

#endif



// EXACT SOLUTION
int n_test = 0;

double phase_x =  0.13;
double phase_y =  1.55;
double phase_z =  0.7;

#ifdef P4_TO_P8
class U_EXACT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
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
    case 0: return (sin(PI*x+phase_x)*sin(PI*y+phase_y));
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
    case 0: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} ux;
class UY: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} uy;
class UZ: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*cos(PI*z+phase_z);
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
      case 0: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} ux;
class UY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y);
    }
  }
} uy;

class UXX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} uxx;
class UYY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} uyy;
#endif

// Diffusion coefficient
#ifdef P4_TO_P8
class MU_CF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return 1;
    }
  }
} mu_cf;
#else
class MU_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return 1+0.2*sin(x)+0.3*cos(y);
    }
  }
} mu_cf;
#endif

#ifdef P4_TO_P8
class MUX: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return 0;
    }
  }
} mux;
class MUY: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return 0;
    }
  }
} muy;
class MUZ: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return 0;
    }
  }
} muz;
#else
class MUX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return .2*cos(x);
    }
  }
} mux;
class MUY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -0.3*sin(y);
    }
  }
} muy;
#endif

// Diagonal term
#ifdef P4_TO_P8
class DIAG_ADD_CF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return 0;
    }
  }
} diag_add_cf;
#else
class DIAG_ADD_CF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
        case 0: return 0;
//    case 0: return x*y+exp(y)*sin(x);
    }
  }
} diag_add_cf;
#endif

// RHS
#ifdef P4_TO_P8
class RHS: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return mu_cf(x,y,z)*3.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z) + diag_add_cf(x,y,z)*u_exact(x,y,z)
                    - mux(x,y,z)*ux(x,y,z) - muy(x,y,z)*uy(x,y,z) - muz(x,y,z)*uz(x,y,z);
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
    case 0: return mu_cf(x,y)*2.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y) + diag_add_cf(x,y)*u_exact(x,y)
                    - mux(x,y)*ux(x,y) - muy(x,y)*uy(x,y);
    }
  }
} rhs_cf;
#endif

#ifdef P4_TO_P8
class double_to_cf_t : public CF_3
{
public:
  double (*func)(double, double, double);
  double_to_cf_t(double (*func)(double, double, double)) { this->func = func; }
  double operator()(double x, double y, double z) const { return (*func)(x,y,z); }
};
#else
class double_to_cf_t : public CF_2
{
public:
  double (*func)(double, double);
  double_to_cf_t(double (*func)(double, double)) { this->func = func; }
  double operator()(double x, double y) const { return (*func)(x,y); }
};
#endif

// BC COEFFICIENTS
#ifdef P4_TO_P8
inline double kappa0(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y)*sin(z+0.33*PI);} double_to_cf_t bc_coeff0(&kappa0);
inline double kappa1(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y)*sin(z+0.33*PI);} double_to_cf_t bc_coeff1(&kappa1);
inline double kappa2(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y)*sin(z+0.33*PI);} double_to_cf_t bc_coeff2(&kappa2);
inline double kappa3(double x, double y, double z) { return 1.0 + .0*sin(x)*cos(y)*sin(z+0.33*PI);} double_to_cf_t bc_coeff3(&kappa3);
#else
inline double kappa0(double x, double y) { return .0 + 1.0*sin(x)*cos(y);} double_to_cf_t bc_coeff0(&kappa0);
inline double kappa1(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff1(&kappa1);
inline double kappa2(double x, double y) { return .0 + 1.0*cos(x)*sin(y);} double_to_cf_t bc_coeff2(&kappa2);
inline double kappa3(double x, double y) { return 1.0 + .0*sin(x)*cos(y);} double_to_cf_t bc_coeff3(&kappa3);
#endif

// BC VALUES
#ifdef P4_TO_P8
class bc_value_robin_t : public CF_3
{
  CF_3 *u, *ux, *uy, *uz;
  CF_3 *phi_x, *phi_y, *phi_z;
  CF_3 *kappa;
  CF_3 *mu;
public:
  bc_value_robin_t(CF_3 *u, CF_3 *ux, CF_3 *uy, CF_3 *uz, CF_3 *mu, CF_3 *phi_x, CF_3 *phi_y, CF_3 *phi_z, CF_3 *kappa) :
    u(u), ux(ux), uy(uy), uz(uz), mu(mu), phi_x(phi_x), phi_y(phi_y), phi_z(phi_z), kappa(kappa) {}
  double operator()(double x, double y, double z) const
  {
    double nx = (*phi_x)(x,y,z);
    double ny = (*phi_y)(x,y,z);
    double nz = (*phi_z)(x,y,z);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    return (*mu)(x,y,z)*(nx*(*ux)(x,y,z) + ny*(*uy)(x,y,z) + nz*(*uz)(x,y,z)) + (*kappa)(x,y,z)*(*u)(x,y,z);
  }
};
#else
class bc_value_robin_t : public CF_2
{
  CF_2 *u, *ux, *uy;
  CF_2 *phi_x, *phi_y;
  CF_2 *kappa;
  CF_2 *mu;
public:
  bc_value_robin_t(CF_2 *u, CF_2 *ux, CF_2 *uy, CF_2 *mu, CF_2 *phi_x, CF_2 *phi_y, CF_2 *kappa) :
    u(u), ux(ux), uy(uy), mu(mu), phi_x(phi_x), phi_y(phi_y), kappa(kappa) {}
  double operator()(double x, double y) const
  {
    double nx = (*phi_x)(x,y);
    double ny = (*phi_y)(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return (*mu)(x,y)*(nx*(*ux)(x,y) + ny*(*uy)(x,y)) + (*kappa)(x,y)*(*u)(x,y);
  }
};
#endif

#ifdef P4_TO_P8
bc_value_robin_t bc_value0(&u_exact, &ux, &uy, &uz, &mu_cf, &domain0.phi_x, &domain0.phi_y, &domain0.phi_z, &bc_coeff0);
bc_value_robin_t bc_value1(&u_exact, &ux, &uy, &uz, &mu_cf, &domain0.phi_x, &domain0.phi_y, &domain0.phi_z, &bc_coeff1);
bc_value_robin_t bc_value2(&u_exact, &ux, &uy, &uz, &mu_cf, &domain2.phi_x, &domain2.phi_y, &domain2.phi_z, &bc_coeff2);
bc_value_robin_t bc_value3(&u_exact, &ux, &uy, &uz, &mu_cf, &domain3.phi_x, &domain3.phi_y, &domain3.phi_z, &bc_coeff3);
bc_value_robin_t bc_value_corners(&u_exact, &ux, &uy, &uz, &mu_cf, &phi_x_corners, &phi_y_corners, &phi_z_corners, &bc_coeff0);
#else
bc_value_robin_t bc_value0(&u_exact, &ux, &uy, &mu_cf, &domain0.phi_x, &domain0.phi_y, &bc_coeff0);
bc_value_robin_t bc_value1(&u_exact, &ux, &uy, &mu_cf, &domain0.phi_x, &domain0.phi_y, &bc_coeff1);
bc_value_robin_t bc_value2(&u_exact, &ux, &uy, &mu_cf, &domain2.phi_x, &domain2.phi_y, &bc_coeff2);
bc_value_robin_t bc_value3(&u_exact, &ux, &uy, &mu_cf, &domain3.phi_x, &domain3.phi_y, &bc_coeff3);
bc_value_robin_t bc_value_corners(&u_exact, &ux, &uy, &mu_cf, &phi_x_corners, &phi_y_corners, &bc_coeff0);
#endif

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
    phi_cf.push_back(&domain1.phi); action.push_back(COLORATION);   color.push_back(color.size());
    phi_cf.push_back(&domain2.phi); action.push_back(ADDITION);     color.push_back(color.size());
    phi_cf.push_back(&domain3.phi); action.push_back(INTERSECTION); color.push_back(color.size());
//    phi_cf.push_back(&phi_corners); action.push_back(INTERSECTION); color.push_back(color.size());

    // set BCs
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff0); bc_values.push_back(&bc_value0);
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff1); bc_values.push_back(&bc_value1);
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff2); bc_values.push_back(&bc_value2);
    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff3); bc_values.push_back(&bc_value3);
//    bc_type.push_back(ROBIN); bc_coeffs.push_back(&bc_coeff0); bc_values.push_back(&bc_value_corners);
  }
} problem;

#ifdef P4_TO_P8
class level_set_tot_t : public CF_3
#else
class level_set_tot_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;

public:

#ifdef P4_TO_P8
  level_set_tot_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#else
  level_set_tot_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#endif
    phi_cf(phi_cf), action(action), color(color) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (short i = 0; i < color->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
    return phi_total;
  }
};

#ifdef P4_TO_P8
class level_set_tot_diff_t : public CF_3
#else
class level_set_tot_diff_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;
  double epsilon;

public:

#ifdef P4_TO_P8
  level_set_tot_diff_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, double epsilon) :
#else
  level_set_tot_diff_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, double epsilon) :
#endif
    phi_cf(phi_cf), action(action), color(color), epsilon(epsilon) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (short i = 0; i < color->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        phi_total = 0.5*(phi_total+phi_current+sqrt(SQR(phi_total-phi_current)+epsilon));
//        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        phi_total = 0.5*(phi_total+phi_current-(sqrt(SQR(phi_total-phi_current)+epsilon)-epsilon/sqrt(SQR(phi_total-phi_current)+epsilon)));
//        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
    return phi_total;//+epsilon;
  }
};

level_set_tot_t level_set_tot(&problem.phi_cf, &problem.action, &problem.color);
