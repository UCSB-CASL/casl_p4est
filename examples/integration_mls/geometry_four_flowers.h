#ifndef GEOMETRY_FOUR_FLOWERS_H
#define GEOMETRY_FOUR_FLOWERS_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class geometry_four_flowers_t
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

  geometry_four_flowers_t()
  {
#ifdef P4_TO_P8
      double r0 =  0.7;
      double r1 =  0.6;
      double r2 =  0.6;
      double r3 =  0.6;
      double d  =  0.25;
#else
      double r0 =  0.53;
      double r1 =  0.5;
      double r2 =  0.61;
      double r3 =  0.5;
      double d  =  0.25;
#endif

      double theta0 = 0.882; double cosT0 = cos(theta0); double sinT0 = sin(theta0);
      double theta1 = 0.623; double cosT1 = cos(theta1); double sinT1 = sin(theta1);

#ifdef P4_TO_P8
      double xc0 = -round((1.*d*cosT0*cosT1)*1.e4)/1.e4; double yc0 =  round((1.*d*cosT0*cosT1)*1.e4)/1.e4; double zc0 =  round((1.*d*sinT1)*1.e4)/1.e4;
      double xc1 = -round((2.*d*cosT0*cosT1)*1.e4)/1.e4; double yc1 = -round((2.*d*cosT0*cosT1)*1.e4)/1.e4; double zc1 =  round((0.*d*sinT1)*1.e4)/1.e4;
      double xc2 =  round((1.*d*cosT0*cosT1)*1.e4)/1.e4; double yc2 = -round((1.*d*cosT0*cosT1)*1.e4)/1.e4; double zc2 = -round((1.*d*sinT1)*1.e4)/1.e4;
      double xc3 =  round((3.*d*cosT0*cosT1)*1.e4)/1.e4; double yc3 =  round((3.*d*sinT0*cosT1)*1.e4)/1.e4; double zc3 =  round((3.*d*sinT1)*1.e4)/1.e4;
#else
      double xc0 = -round((1.*d*sinT0)*1.e4)/1.e4; double yc0 =  round((1.*d*cosT0)*1.e4)/1.e4;
      double xc1 = -round((2.*d*sinT0)*1.e4)/1.e4; double yc1 = -round((2.*d*cosT0)*1.e4)/1.e4;
      double xc2 =  round((1.*d*sinT0)*1.e4)/1.e4; double yc2 = -round((1.*d*cosT0)*1.e4)/1.e4;
      double xc3 =  round((2.*d*cosT0)*1.e4)/1.e4; double yc3 =  round((2.*d*sinT0)*1.e4)/1.e4;
#endif

#ifdef P4_TO_P8
      double beta0 = 0.05; double inside0 =  1;
      double beta1 = 0.1; double inside1 =  1;
      double beta2 = -0.11; double inside2 =  1;
      double beta3 = 0.08; double inside3 = -1;
#else
      double beta0 = 0.05; double inside0 =  1;
      double beta1 = 0.1; double inside1 =  1;
      double beta2 = 0.11; double inside2 =  1;
      double beta3 = 0.08; double inside3 = -1;
#endif

//      double alpha0 = 0.3*PI;
//      double alpha1 = 0.1*PI;
//      double alpha2 =-0.2*PI;
//      double alpha3 = 0.3*PI;

      double alpha0 = 0.0*PI;
      double alpha1 = 0.0*PI;
      double alpha2 =-0.0*PI;
      double alpha3 = 0.0*PI;

      double lx0 = 1, ly0 = 1, lz0 = 1;
      double lx1 =-1, ly1 = 1, lz1 = 1;
      double lx2 = 1, ly2 =-1, lz2 = 1;
      double lx3 = 1, ly3 = 1, lz3 =-1;

#ifdef P4_TO_P8
      domain0.set_params(r0, xc0, yc0, zc0, beta0, inside0, lx0, ly0, lz0, alpha0);
      domain1.set_params(r1, xc1, yc1, zc1, beta1, inside1, lx1, ly1, lz1, alpha1);
      domain2.set_params(r2, xc2, yc2, zc2, beta2, inside2, lx2, ly2, lz2, alpha2);
      domain3.set_params(r3, xc3, yc3, zc3, beta3, inside3, lx3, ly3, lz3, alpha3);
#else
      domain0.set_params(r0, xc0, yc0, beta0, inside0, alpha0);
      domain1.set_params(r1, xc1, yc1, beta1, inside1, alpha1);
      domain2.set_params(r2, xc2, yc2, beta2, inside2, alpha2);
      domain3.set_params(r3, xc3, yc3, beta3, inside3, alpha3);
#endif

      LSF.push_back(&domain0.phi); action.push_back(INTERSECTION);    color.push_back(0);
      LSF.push_back(&domain2.phi); action.push_back(ADDITION);        color.push_back(1);
      LSF.push_back(&domain3.phi); action.push_back(INTERSECTION);    color.push_back(2);
      LSF.push_back(&domain1.phi); action.push_back(COLORATION);      color.push_back(3);

#ifdef P4_TO_P8
    std::cout << "Four flowers 3d - shape parameters\n";
    std::cout << "Level-set 0: rad : " << r0 << "; center: " << xc0 << ", " << yc0 << ", " << zc0 << std::endl;
    std::cout << "Level-set 1: rad : " << r1 << "; center: " << xc1 << ", " << yc1 << ", " << zc1 << std::endl;
    std::cout << "Level-set 2: rad : " << r2 << "; center: " << xc2 << ", " << yc2 << ", " << zc2 << std::endl;
    std::cout << "Level-set 3: rad : " << r3 << "; center: " << xc3 << ", " << yc3 << ", " << zc3 << std::endl;
#else
    std::cout << "Four flowers 2d - shape parameters\n";
    std::cout << "Level-set 0: rad : " << r0 << "; center: " << xc0 << ", " << yc0 << std::endl;
    std::cout << "Level-set 1: rad : " << r1 << "; center: " << xc1 << ", " << yc1 << std::endl;
    std::cout << "Level-set 2: rad : " << r2 << "; center: " << xc2 << ", " << yc2 << std::endl;
    std::cout << "Level-set 3: rad : " << r3 << "; center: " << xc3 << ", " << yc3 << std::endl;
#endif

    n_subs = 4;
    n_Xs = 5;
    IXc0.push_back(0); IXc1.push_back(2);
    IXc0.push_back(0); IXc1.push_back(3);
    IXc0.push_back(1); IXc1.push_back(2);
    IXc0.push_back(1); IXc1.push_back(3);
    IXc0.push_back(2); IXc1.push_back(3);

    n_X3s = 2;
    IX3c0.push_back(0); IX3c1.push_back(1); IX3c2.push_back(2);
    IX3c0.push_back(0); IX3c1.push_back(1); IX3c2.push_back(3);

  }

};

#endif // GEOMETRY_FOUR_FLOWERS_H
