#ifndef GEOMETRY_ONE_CIRCLE_H
#define GEOMETRY_ONE_CIRCLE_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class geometry_one_circle_t
{
public:

  flower_shaped_domain_t domain0;

  struct exact_t {
    double ID;
    double IB;
    std::vector<double> ISB;
    std::vector<double> IX;
    std::vector<double> IX3;
  } exact0, exact1;


#ifdef P4_TO_P8
  std::vector<CF_3 *> LSF;
#else
  std::vector<CF_2 *> LSF;
#endif
  std::vector<action_t> action;
  std::vector<int> color;

  double n_subs;
  double n_Xs;

  std::vector<int> IXc0, IXc1;

  double n_X3s;
  std::vector<int> IX3c0, IX3c1, IX3c2;

  geometry_one_circle_t()
  {
//    double r0 = 0.814;
//    double d = 0.133;

//    double theta = 0.379;
//#ifdef P4_TO_P8
//    double phy = 0.312;
//#endif

//    double r0 = 0.711;
    double r1 = 0.639;
    double d = 0.35;

    double theta = 0.779;
#ifdef P4_TO_P8
    double phy = 0.312;
#endif

    double cosT = cos(theta);
    double sinT = sin(theta);
#ifdef P4_TO_P8
    double cosP = cos(phy);
    double sinP = sin(phy);
#endif

//#ifdef P4_TO_P8
//    double xc_0 = round((-d*sinT*cosP-0.02)*100.)/100.; double yc_0 = round(( d*cosT*cosP-0.07)*100.)/100.; double zc_0 = round(( d*sinP-0.03)*100.)/100.;
//#else
//    double xc_0 = round((-d*sinT+0.08)*100.)/100.; double yc_0 = round(( d*cosT-0.07)*100.)/100.;
//#endif

#ifdef P4_TO_P8
    double r0 = 0.77, xc_0 = 0, yc_0 = 0, zc_0 = 0;
//    double r0 = 0.71, xc_0 = 0.13, yc_0 = 0.27, zc_0 = 0.21;
//    double r0 = 0.5, xc_0 = 0.08, yc_0 = 0.01, zc_0 = 0.03;
//    double r0 = 0.63, xc_0 =-0.24, yc_0 =-0.16, zc_0 =-0.33;
    domain0.set_params(r0, xc_0, yc_0, zc_0, 0.0);
#else
//    double r0 = 0.77, xc_0 = 0.13, yc_0 = 0.21;
    double r0 = 0.749999999, xc_0 = 0., yc_0 = 0.;
    domain0.set_params(r0, xc_0, yc_0);
#endif

    LSF.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);

#ifdef P4_TO_P8
    std::cout << "Level-set 0: " << xc_0 << ", " << yc_0 << ", " << zc_0 << std::endl;
#else
    std::cout << "Level-set 0: " << xc_0 << ", " << yc_0 << std::endl;
#endif

    n_subs = 1;
    n_Xs = 0;

    n_X3s = 0;

#ifdef P4_TO_P8
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0 + zc_0*zc_0);

    double mu_sph_0 = 4./3.*PI*r0*r0*r0; double mi_sph_0 = mu_sph_0*(0.6*r0*r0+R0*R0);

    double mu_bnd_0 = 4.*PI*r0*r0; double mi_bnd_0 = mu_bnd_0*(r0*r0+R0*R0);

#else
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0);

    double mu_sph_0 = PI*r0*r0; double mi_sph_0 = PI*r0*r0*(0.5*r0*r0+R0*R0);

    double mu_bnd_0 = 2.0*PI*r0; double mi_bnd_0 = 2.0*PI*r0*(r0*r0+R0*R0);

#endif

    exact0.ID = mu_sph_0;
    exact1.ID = mi_sph_0;

    exact0.ISB.push_back(mu_bnd_0); exact1.ISB.push_back(mi_bnd_0);

  }

};

#endif // GEOMETRY_ONE_CIRCLE_H
