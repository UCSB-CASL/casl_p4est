#ifndef GEOMETRY_ROSE_H
#define GEOMETRY_ROSE_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class geometry_rose_t
{
public:

  flower_shaped_domain_t domain0;
  flower_shaped_domain_t domain1;
  flower_shaped_domain_t domain2;
  flower_shaped_domain_t domain3;

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

  geometry_rose_t()
  {
    double r0 = 0.6;
    double r1 = 0.5;
    double r2 = 0.4;
    double r3 = 0.3;
    double d = 0.25;

    double theta = 0.479;
#ifdef P4_TO_P8
    double phy = 0.312;
#endif

    double cosT = cos(theta);
    double sinT = sin(theta);
#ifdef P4_TO_P8
    double cosP = cos(phy);
    double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
    double xc_0 = 0.01; double yc_0 = -0.02; double zc_0 = 0.0;
    double xc_1 = 0.19; double yc_1 = -0.17; double zc_1 = 0.0;
    double xc_2 = round((r0*cosT*cosP)*1.e4)/1.e4; double yc_2 = round((r0*sinT*cosP)*1.e4)/1.e4; double zc_2 = round((r0*sinP)*1.e4)/1.e4-0.3;
    double xc_3 =-round((r0*cosT*cosP)*1.e4)/1.e4; double yc_3 =-round((r0*sinT*cosP)*1.e4)/1.e4; double zc_3 =-round((r0*sinP)*1.e4)/1.e4;
#else
    double xc_0 = 0.01; double yc_0 = -0.02;
    double xc_1 = 0.19; double yc_1 = -0.17;
    double xc_2 = round((r0*cosT-0.5)*1.e4)/1.e4; double yc_2 = round((r0*sinT+0.2)*1.e4)/1.e4;
    double xc_3 =-round((r0*cosT)*1.e4)/1.e4; double yc_3 =-round((r0*sinT)*1.e4)/1.e4;
#endif

#ifdef P4_TO_P8
    domain0.set_params(r0, xc_0, yc_0, zc_0, 0., 1.);
    domain1.set_params(r1, xc_1, yc_1, zc_1, 0.,-1.);
    domain2.set_params(r2, xc_2, yc_2, zc_2, 0., 1.);
    domain3.set_params(r3, xc_3, yc_3, zc_3, 0.,-1.);
#else
    domain0.set_params(r0, xc_0, yc_0, 0., 1.);
    domain1.set_params(r1, xc_1, yc_1, 0.,-1.);
    domain2.set_params(r2, xc_2, yc_2, 0., 1.);
    domain3.set_params(r3, xc_3, yc_3, 0.,-1.);
#endif

    LSF.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    LSF.push_back(&domain1.phi); action.push_back(INTERSECTION); color.push_back(1);
    LSF.push_back(&domain2.phi); action.push_back(ADDITION); color.push_back(2);
    LSF.push_back(&domain3.phi); action.push_back(INTERSECTION); color.push_back(3);

#ifdef P4_TO_P8
    std::cout << "Four circles 3d - shape parameters\n";
    std::cout << "Level-set 0: rad : " << r0 << "; center: " << xc_0 << ", " << yc_0 << ", " << zc_0 << std::endl;
    std::cout << "Level-set 1: rad : " << r1 << "; center: " << xc_1 << ", " << yc_1 << ", " << zc_1 << std::endl;
    std::cout << "Level-set 2: rad : " << r2 << "; center: " << xc_2 << ", " << yc_2 << ", " << zc_2 << std::endl;
    std::cout << "Level-set 3: rad : " << r3 << "; center: " << xc_3 << ", " << yc_3 << ", " << zc_3 << std::endl;
#else
    std::cout << "Four circles 2d - shape parameters\n";
    std::cout << "Level-set 0: rad : " << r0 << "; center: " << xc_0 << ", " << yc_0 << std::endl;
    std::cout << "Level-set 1: rad : " << r1 << "; center: " << xc_1 << ", " << yc_1 << std::endl;
    std::cout << "Level-set 2: rad : " << r2 << "; center: " << xc_2 << ", " << yc_2 << std::endl;
    std::cout << "Level-set 3: rad : " << r3 << "; center: " << xc_3 << ", " << yc_3 << std::endl;
#endif

    n_subs = 4;
    n_Xs = 5;
    IXc0.push_back(0); IXc1.push_back(1);
    IXc0.push_back(0); IXc1.push_back(2);
    IXc0.push_back(0); IXc1.push_back(3);
    IXc0.push_back(1); IXc1.push_back(2);
    IXc0.push_back(1); IXc1.push_back(3);


#ifdef P4_TO_P8
    n_X3s = 1;
    IX3c0.push_back(0); IX3c1.push_back(1); IX3c2.push_back(2);
#else
    n_X3s = 0;
#endif

  }

};

#endif // GEOMETRY_ROSE_H
