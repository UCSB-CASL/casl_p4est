#ifndef GEOMETRY_TWO_CIRCLES_UNION_H
#define GEOMETRY_TWO_CIRCLES_UNION_H
#include <vector>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_integration_mls.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_integration_mls.h>
#endif

#include "shapes.h"

class geometry_two_circles_union_t
{
public:

  flower_shaped_domain_t domain0;
  flower_shaped_domain_t domain1;

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

  geometry_two_circles_union_t()
  {
//    double r0 = 0.711;
//    double r1 = 0.639;
//    double d = 0.253;

//    double theta = 0.779;
//#ifdef P4_TO_P8
//    double phy = 0.312;
//#endif

//    double cosT = cos(theta);
//    double sinT = sin(theta);
//#ifdef P4_TO_P8
//    double cosP = cos(phy);
//    double sinP = sin(phy);
//#endif

//#ifdef P4_TO_P8
//    double xc_0 = round((-d*sinT*cosP-0.02)*100.)/100.; double yc_0 = round(( d*cosT*cosP-0.07)*100.)/100.; double zc_0 = round(( d*sinP-0.03)*100.)/100.;
//    double xc_1 = round(( d*sinT*cosP-0.02)*100.)/100.; double yc_1 = round((-d*cosT*cosP-0.07)*100.)/100.; double zc_1 = round((-d*sinP-0.03)*100.)/100.;
//#else
//    double xc_0 = round((-d*sinT+0.08)*100.)/100.; double yc_0 = round(( d*cosT-0.07)*100.)/100.;
//    double xc_1 = round(( d*sinT+0.08)*100.)/100.; double yc_1 = round((-d*cosT-0.07)*100.)/100.;
//#endif
//#ifdef P4_TO_P8
//    domain0.set_params(r0, xc_0, yc_0, zc_0);
//    domain1.set_params(r1, xc_1, yc_1, zc_1);
//#else
//    domain0.set_params(r0, xc_0, yc_0);
//    domain1.set_params(r1, xc_1, yc_1);
//#endif

#ifdef P4_TO_P8
    double r0 = 0.71, xc_0 = 0.13, yc_0 = 0.27, zc_0 = 0.21;
    double r1 = 0.63, xc_1 =-0.24, yc_1 =-0.16, zc_1 =-0.33;

    domain0.set_params(r0, xc_0, yc_0, zc_0);
    domain1.set_params(r1, xc_1, yc_1, zc_1);
#else
    double r0 = 0.77, xc_0 = 0.13, yc_0 = 0.21;
    double r1 = 0.53, xc_1 =-0.41, yc_1 =-0.37;

    domain0.set_params(r0, xc_0, yc_0);
    domain1.set_params(r1, xc_1, yc_1);
#endif

    LSF.push_back(&domain0.phi); action.push_back(INTERSECTION); color.push_back(0);
    LSF.push_back(&domain1.phi); action.push_back(ADDITION); color.push_back(1);

//#ifdef P4_TO_P8
//    std::cout << "Level-set 0: " << xc_0 << ", " << yc_0 << ", " << zc_0 << std::endl;
//    std::cout << "Level-set 1: " << xc_1 << ", " << yc_1 << ", " << zc_1 << std::endl;
//#else
//    std::cout << "Level-set 0: " << xc_0 << ", " << yc_0 << std::endl;
//    std::cout << "Level-set 1: " << xc_1 << ", " << yc_1 << std::endl;
//#endif

    n_subs = 2;
    n_Xs = 1;
    IXc0.push_back(0); IXc1.push_back(1);

    n_X3s = 0;

#ifdef P4_TO_P8
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0 + zc_0*zc_0);
    double R1 = sqrt(xc_1*xc_1 + yc_1*yc_1 + zc_1*zc_1);

    double D = sqrt(pow(xc_1-xc_0,2.0)+pow(yc_1-yc_0,2.0)+pow(zc_1-zc_0,2.0));

    double d0 = (r0*r0+D*D-r1*r1)/(2.0*D);
    double d1 = D-d0;

    double d0x =  (xc_1-xc_0)/D*d0; double d0y =  (yc_1-yc_0)/D*d0; double d0z =  (zc_1-zc_0)/D*d0;
    double d1x = -(xc_1-xc_0)/D*d1; double d1y = -(yc_1-yc_0)/D*d1; double d1z = -(zc_1-zc_0)/D*d1;

    double r0d0 = xc_0*d0x + yc_0*d0y + zc_0*d0z;
    double r1d1 = xc_1*d1x + yc_1*d1y + zc_1*d1z;

    double alpha_0 = acos(d0/r0);
    double alpha_1 = acos(d1/r1);

    double mu_sph_0 = 4./3.*PI*r0*r0*r0; double mi_sph_0 = mu_sph_0*(0.6*r0*r0+R0*R0);
    double mu_sph_1 = 4./3.*PI*r1*r1*r1; double mi_sph_1 = mu_sph_1*(0.6*r1*r1+R1*R1);

    double mu_tri_0 = d0*PI*(r0*r0-d0*d0)/3.0; double mi_tri_0 = 0.3*mu_tri_0*(r0*r0+d0*d0) + mu_tri_0*(R0*R0+1.5*r0d0);
    double mu_tri_1 = d1*PI*(r1*r1-d1*d1)/3.0; double mi_tri_1 = 0.3*mu_tri_1*(r1*r1+d1*d1) + mu_tri_1*(R1*R1+1.5*r1d1);

    double mu_sec_0 = 2.*PI*r0*r0*(r0-d0)/3.; double mi_sec_0 = 0.4*PI*r0*r0*r0*r0*(r0-d0) + mu_sec_0*(R0*R0+0.75*r0d0*(r0/d0+1.));
    double mu_sec_1 = 2.*PI*r1*r1*(r1-d1)/3.; double mi_sec_1 = 0.4*PI*r1*r1*r1*r1*(r1-d1) + mu_sec_1*(R1*R1+0.75*r1d1*(r1/d1+1.));

    double mu_bnd_0 = 4.*PI*r0*r0; double mi_bnd_0 = mu_bnd_0*(r0*r0+R0*R0);
    double mu_bnd_1 = 4.*PI*r1*r1; double mi_bnd_1 = mu_bnd_1*(r1*r1+R1*R1);

    double mu_seg_0 = 2.*PI*r0*(r0-d0); double mi_seg_0 = 2.*PI*r0*r0*r0*(r0-d0) + mu_seg_0*(R0*R0+r0d0*(r0/d0+1.));
    double mu_seg_1 = 2.*PI*r1*(r1-d1); double mi_seg_1 = 2.*PI*r1*r1*r1*(r1-d1) + mu_seg_1*(R1*R1+r1d1*(r1/d1+1.));

#else
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0);
    double R1 = sqrt(xc_1*xc_1 + yc_1*yc_1);

    double D = sqrt(pow(xc_1-xc_0,2.0)+pow(yc_1-yc_0,2.0));

    double d0 = (r0*r0+D*D-r1*r1)/(2.0*D);
    double d1 = D-d0;

    double d0x =  (xc_1-xc_0)/D*d0; double d0y =  (yc_1-yc_0)/D*d0;
    double d1x = -(xc_1-xc_0)/D*d1; double d1y = -(yc_1-yc_0)/D*d1;

    double r0d0 = xc_0*d0x + yc_0*d0y;
    double r1d1 = xc_1*d1x + yc_1*d1y;

    double alpha_0 = acos(d0/r0);
    double alpha_1 = acos(d1/r1);

    double mu_sph_0 = PI*r0*r0; double mi_sph_0 = PI*r0*r0*(0.5*r0*r0+R0*R0);
    double mu_sph_1 = PI*r1*r1; double mi_sph_1 = PI*r1*r1*(0.5*r1*r1+R1*R1);

    double mu_tri_0 = d0*sqrt(r0*r0-d0*d0); double mi_tri_0 = d0*sqrt(r0*r0-d0*d0)*(r0*r0/6.0+d0*d0/3.0+R0*R0 + 4.0*r0d0/3.0);
    double mu_tri_1 = d1*sqrt(r1*r1-d1*d1); double mi_tri_1 = d1*sqrt(r1*r1-d1*d1)*(r1*r1/6.0+d1*d1/3.0+R1*R1 + 4.0*r1d1/3.0);

    double mu_sec_0 = alpha_0*r0*r0; double mi_sec_0 = alpha_0*r0*r0*(0.5*r0*r0+R0*R0+2.0*r0d0*2.0*r0*sin(alpha_0)/alpha_0/d0/3.0);
    double mu_sec_1 = alpha_1*r1*r1; double mi_sec_1 = alpha_1*r1*r1*(0.5*r1*r1+R1*R1+2.0*r1d1*2.0*r1*sin(alpha_1)/alpha_1/d1/3.0);

    double mu_bnd_0 = 2.0*PI*r0; double mi_bnd_0 = 2.0*PI*r0*(r0*r0+R0*R0);
    double mu_bnd_1 = 2.0*PI*r1; double mi_bnd_1 = 2.0*PI*r1*(r1*r1+R1*R1);

    double mu_seg_0 = 2.0*alpha_0*r0; double mi_seg_0 = 2.0*alpha_0*r0*(r0*r0+R0*R0+2.0*r0d0*r0*sin(alpha_0)/alpha_0/d0);
    double mu_seg_1 = 2.0*alpha_1*r1; double mi_seg_1 = 2.0*alpha_1*r1*(r1*r1+R1*R1+2.0*r1d1*r1*sin(alpha_1)/alpha_1/d1);
#endif

    exact0.ID = mu_sph_0+mu_sph_1 -mu_sec_0-mu_sec_1 +mu_tri_0+mu_tri_1;
    exact1.ID = mi_sph_0+mi_sph_1 -mi_sec_0-mi_sec_1 +mi_tri_0+mi_tri_1;

    exact0.ISB.push_back(mu_bnd_0-mu_seg_0); exact1.ISB.push_back(mi_bnd_0-mi_seg_0);
    exact0.ISB.push_back(mu_bnd_1-mu_seg_1); exact1.ISB.push_back(mi_bnd_1-mi_seg_1);

#ifdef P4_TO_P8
    exact0.IX.push_back(2.0*PI*sqrt(r0*r0-d0*d0)); exact1.IX.push_back(2.0*PI*sqrt(r0*r0-d0*d0)*(r0*r0+R0*R0+2.*r0d0));
#else
    exact0.IX.push_back(2.0); exact1.IX.push_back(2.0*(r0*r0+R0*R0+2.0*r0d0));
#endif

  }

};

#endif // GEOMETRY_TWO_CIRCLES_UNION_H
