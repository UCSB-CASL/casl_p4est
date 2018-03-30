#include "shapes.h"

/* discretization */
int lmin = 4;
int lmax = 4;
#ifdef P4_TO_P8
int nb_splits = 7;
#else
int nb_splits = 6;
#endif

int nx = 1;
int ny = 1;
int nz = 1;

bool save_vtk = false;

/* geometry */

double xmin = -1.00;
double xmax =  1.00;
double ymin = -1.00;
double ymax =  1.00;
double zmin = -1;
double zmax =  1;

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

#ifdef P4_TO_P8
class LS_REF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -1;
  }
} ls_ref;
#else
class LS_REF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -1;
  }
} ls_ref;
#endif

class Exact {
public:
  double ID;
  double IB;
  vector<double> ISB;
  vector<double> IX;
  vector<int> IXc0, IXc1;
  double alpha;

  double n_subs = 4;
  double n_Xs = 5;

  bool provided = false;

  Exact()
  {
    IXc0.push_back(0); IXc1.push_back(1);
    IXc0.push_back(0); IXc1.push_back(2);
    IXc0.push_back(0); IXc1.push_back(3);
    IXc0.push_back(1); IXc1.push_back(2);
    IXc0.push_back(2); IXc1.push_back(3);
  }
} exact;

class Geometry
{
public:
#ifdef P4_TO_P8
  vector<CF_3 *> LSF;
#else
  vector<CF_2 *> LSF;
#endif
  vector<action_t> action;
  vector<int> color;
  Geometry()
  {
    LSF.push_back(&domain0.phi); action.push_back(INTERSECTION);    color.push_back(0);
    LSF.push_back(&domain2.phi); action.push_back(ADDITION);        color.push_back(1);
    LSF.push_back(&domain3.phi); action.push_back(INTERSECTION);    color.push_back(2);
    LSF.push_back(&domain1.phi); action.push_back(COLORATION);      color.push_back(3);
#ifdef P4_TO_P8
    std::cout << "Level-set 0: " << xc0 << ", " << yc0 << ", " << zc0 << endl;
    std::cout << "Level-set 1: " << xc1 << ", " << yc1 << ", " << zc1 << endl;
#else
    std::cout << "Level-set 0: " << xc0 << ", " << yc0 << endl;
    std::cout << "Level-set 1: " << xc1 << ", " << yc1 << endl;
#endif
  }
} geometry;


