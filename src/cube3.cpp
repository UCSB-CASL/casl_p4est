#include "cube3.h"

Cube3::Cube3()
{
  x0 = x1 = 0.;
  y0 = y1 = 0.;
  z0 = z1 = 0.;
  middlecut = false;
  num_tet = 6;
//  middlecut = true;
//  num_tet = 5;
}

Cube3::Cube3(double x0, double x1, double y0, double y1, double z0, double z1)
{
  this->x0 = x0; this->x1 = x1;
  this->y0 = y0; this->y1 = y1;
  this->z0 = z0; this->z1 = z1;
    middlecut = false;
    num_tet = 6;
//    middlecut = true;
//    num_tet = 5;
}

double Cube3::interface_Area_In_Cell( OctValue& level_set_values) const
{
  OctValue tmp(1.,1.,1.,1.,1.,1.,1.,1.);
  return integrate_Over_Interface(tmp, level_set_values);
}

double Cube3::volume_In_Negative_Domain( OctValue& level_set_values) const
{
  OctValue tmp(1.,1.,1.,1.,1.,1.,1.,1.);
  return integral(tmp,level_set_values);
}

//Finds the volume of domain inside each cell. Takes in values at imaginary points +/-dx/2, +/-dy/2, +/-dz/2 away from the actual node (those values have to be interpolated before the function is called)
double Cube3::integral(const OctValue &f, const OctValue &ls_values) const
{
  double sum=0;

  Point3 P000(x0,y0,z0);
  Point3 P001(x0,y0,z1);
  Point3 P010(x0,y1,z0);
  Point3 P011(x0,y1,z1);
  Point3 P100(x1,y0,z0);
  Point3 P101(x1,y0,z1);
  Point3 P110(x1,y1,z0);
  Point3 P111(x1,y1,z1);

  // simple cases
  if(  ls_values.val000<=0 && ls_values.val001<=0 &&
       ls_values.val010<=0 && ls_values.val011<=0 &&
       ls_values.val100<=0 && ls_values.val101<=0 &&
       ls_values.val110<=0 && ls_values.val111<=0 )
    return  (x1-x0)*(y1-y0)*(z1-z0)*(f.val000+f.val001+f.val010+f.val011+f.val100+f.val101+f.val110+f.val111)/8.;

  if(  ls_values.val000>0 && ls_values.val001>0 &&
       ls_values.val010>0 && ls_values.val011>0 &&
       ls_values.val100>0 && ls_values.val101>0 &&
       ls_values.val110>0 && ls_values.val111>0 ) return 0;

  // iteration on each simplex in the middle cut triangulation
  for(int n=0;n<num_tet;n++)
  {
    // Tetrahedron (P0,P1,P2,P3)
    Point3   P0,  P1,  P2,   P3;
    double   F0,  F1,  F2,   F3;
    double Phi0,Phi1,Phi2,Phi3;

    if (middlecut)
    {
      switch(n) {
      case 0:
        P0=              P000; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val000; F1=          f.val100; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      case 1:
        P0=              P110; P1=              P100; P2=              P010; P3=              P111;
        F0=          f.val110; F1=          f.val100; F2=          f.val010; F3=          f.val111;
        Phi0=ls_values.val110; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val111;
        break;
      case 2:
        P0=              P101; P1=              P100; P2=              P111; P3=              P001;
        F0=          f.val101; F1=          f.val100; F2=          f.val111; F3=          f.val001;
        Phi0=ls_values.val101; Phi1=ls_values.val100; Phi2=ls_values.val111; Phi3=ls_values.val001;
        break;
      case 3:
        P0=              P011; P1=              P111; P2=              P010; P3=              P001;
        F0=          f.val011; F1=          f.val111; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val011; Phi1=ls_values.val111; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      case 4:
        P0=              P111; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val111; F1=          f.val100; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val111; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
    } else {
      switch(n) {
      case 0:
        P0=              P000; P1=              P100; P2=              P110; P3=              P111;
        F0=          f.val000; F1=          f.val100; F2=          f.val110; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val110; Phi3=ls_values.val111;
        break;
      case 1:
        P0=              P000; P1=              P010; P2=              P110; P3=              P111;
        F0=          f.val000; F1=          f.val010; F2=          f.val110; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val110; Phi3=ls_values.val111;
        break;
      case 2:
        P0=              P000; P1=              P100; P2=              P101; P3=              P111;
        F0=          f.val000; F1=          f.val100; F2=          f.val101; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val101; Phi3=ls_values.val111;
        break;
      case 3:
        P0=              P000; P1=              P010; P2=              P011; P3=              P111;
        F0=          f.val000; F1=          f.val010; F2=          f.val011; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val011; Phi3=ls_values.val111;
        break;
      case 4:
        P0=              P000; P1=              P001; P2=              P101; P3=              P111;
        F0=          f.val000; F1=          f.val001; F2=          f.val101; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val101; Phi3=ls_values.val111;
        break;
      case 5:
        P0=              P000; P1=              P001; P2=              P011; P3=              P111;
        F0=          f.val000; F1=          f.val001; F2=          f.val011; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val011; Phi3=ls_values.val111;
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
    }

    // simple cases
    if(Phi0<=0 && Phi1<=0 && Phi2<=0 && Phi3<=0){ sum+=Point3::volume(P0,P1,P2,P3)*(F0+F1+F2+F3)/4.; continue;}
    if(Phi0>0 && Phi1>0 && Phi2>0 && Phi3>0){                                                    continue;}
//    if(Phi0==0 && Phi1==0 && Phi2==0 && Phi3<0) {return (F0+F1+F2+F3)/4.*Point3::volume(P0,P1,P2,P3);}
//    if(Phi0==0 && Phi1==0 && Phi2<0 && Phi3==0) {return (F0+F1+F2+F3)/4.*Point3::volume(P0,P1,P2,P3);}
//    if(Phi0==0 && Phi1<0 && Phi2==0 && Phi3==0) {return (F0+F1+F2+F3)/4.*Point3::volume(P0,P1,P2,P3);}
//    if(Phi0<0 && Phi1==0 && Phi2==0 && Phi3==0) {return (F0+F1+F2+F3)/4.*Point3::volume(P0,P1,P2,P3);}

    // sorting for simplication into two cases,
    if(Phi0>0 && Phi1<=0) swap(Phi0,Phi1,F0,F1,P0,P1);
    if(Phi0>0 && Phi2<=0) swap(Phi0,Phi2,F0,F2,P0,P2);
    if(Phi0>0 && Phi3<=0) swap(Phi0,Phi3,F0,F3,P0,P3);
    if(Phi1>0 && Phi2<=0) swap(Phi1,Phi2,F1,F2,P1,P2);
    if(Phi1>0 && Phi3<=0) swap(Phi1,Phi3,F1,F3,P1,P3);
    if(Phi2>0 && Phi3<=0) swap(Phi2,Phi3,F2,F3,P2,P3);

    // frustum of simplex (P0,P1,P2) cut by {Phi<=0}
    if(Phi0<=0 && Phi1>0 && Phi2>0 && Phi3>0) // type -+++
    {
      Point3 P01 = interpol_p(P0,Phi0,P1,Phi1);
      Point3 P02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P03 = interpol_p(P0,Phi0,P3,Phi3);

      double F01 = interpol_f(F0,Phi0,F1,Phi1);
      double F02 = interpol_f(F0,Phi0,F2,Phi2);
      double F03 = interpol_f(F0,Phi0,F3,Phi3);

      sum += Point3::volume(P0,P01,P02,P03)*(F0+F01+F02+F03)/4.;
    }
    else if(Phi0<=0 && Phi1<=0 && Phi2>0 && Phi3>0) // type --++
    {
      Point3 P02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P03 = interpol_p(P0,Phi0,P3,Phi3);
      Point3 P12 = interpol_p(P1,Phi1,P2,Phi2);
      Point3 P13 = interpol_p(P1,Phi1,P3,Phi3);

      double F02 = interpol_f(F0,Phi0,F2,Phi2);
      double F03 = interpol_f(F0,Phi0,F3,Phi3);
      double F12 = interpol_f(F1,Phi1,F2,Phi2);
      double F13 = interpol_f(F1,Phi1,F3,Phi3);

      sum += Point3::volume(P0 ,P1 ,P02,P13)*(F0 +F1 +F02+F13)/4.;
      sum += Point3::volume(P12,P1 ,P02,P13)*(F12+F1 +F02+F13)/4.;
      sum += Point3::volume(P0 ,P03,P02,P13)*(F0 +F03+F02+F13)/4.;
    }
    else // type ---+
    {
#ifdef CASL_THROWS
      if(Phi0>0 || Phi1>0 || Phi2>0 || Phi3<=0)
      {
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: wrong configuration.");
      }
#endif

      Point3 P03 = interpol_p(P0,Phi0,P3,Phi3);
      Point3 P13 = interpol_p(P1,Phi1,P3,Phi3);
      Point3 P23 = interpol_p(P2,Phi2,P3,Phi3);

      double F03 = interpol_f(F0,Phi0,F3,Phi3);
      double F13 = interpol_f(F1,Phi1,F3,Phi3);
      double F23 = interpol_f(F2,Phi2,F3,Phi3);

      sum += Point3::volume(P0 ,P1 ,P2 ,P13)*(F0 +F1 +F2+F13)/4.;
      sum += Point3::volume(P0 ,P03,P2 ,P13)*(F0 +F03+F2+F13)/4.;
      sum += Point3::volume(P23,P03,P2 ,P13)*(F23+F03+F2+F13)/4.;
    }
  }
  return sum;
}



double Cube3::integrate_Over_Interface(const OctValue &f, const OctValue &ls_values) const
{
  Point3 P000(x0,y0,z0);
  Point3 P001(x0,y0,z1);
  Point3 P010(x0,y1,z0);
  Point3 P011(x0,y1,z1);
  Point3 P100(x1,y0,z0);
  Point3 P101(x1,y0,z1);
  Point3 P110(x1,y1,z0);
  Point3 P111(x1,y1,z1);
  // [RAPHAEL:] I am ****PISSED****, I have wasted an entire day trying to find yet another bug due to sign
  // errors in this very function!
  //
  // I actually do not care mure about this one, except when it makes my whole set up crash on big grids on
  // Stampede and thus make things CRASH.
  //
  // I am going to introduce some *CLEAR* sign convention in here to fix my issue, I do not care if someone
  // changes it later on, but PLEASE, juste triple-check the consistency of your sign conventions!!!!!
  // Here below:
  // phi <= 0 --> negative domain
  // phi > 0 --> positive domain
  // WHATEVER CHANGE BROUGHT BY WHOMEVER CANNOT ALLOW ONE SINGLE VALUE TO BE CONSIDERED BOTH IN NEGATIVE AND
  // IN POSITIVE DOMAIN!!!

  // simple cases
  if(  ls_values.val000<=0.0 && ls_values.val001<=0.0 &&
       ls_values.val010<=0.0 && ls_values.val011<=0.0 &&
       ls_values.val100<=0.0 && ls_values.val101<=0.0 &&
       ls_values.val110<=0.0 && ls_values.val111<=0.0 ) return 0;

  if(  ls_values.val000>0.0 && ls_values.val001>0.0 &&
       ls_values.val010>0.0 && ls_values.val011>0.0 &&
       ls_values.val100>0.0 && ls_values.val101>0.0 &&
       ls_values.val110>0.0 && ls_values.val111>0.0 ) return 0;



  double sum=0;

  // iteration on each simplex in the middle cut triangulation
  for(int n=0;n<num_tet;n++)
  {
    // Tetrahedron (P0,P1,P2,P3)
    Point3   P0,  P1,  P2,   P3;
    double   F0,  F1,  F2,   F3;
    double Phi0,Phi1,Phi2,Phi3;

    if (middlecut)
    {
      switch(n) {
      case 0:
        P0=              P000; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val000; F1=          f.val100; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      case 1:
        P0=              P110; P1=              P100; P2=              P010; P3=              P111;
        F0=          f.val110; F1=          f.val100; F2=          f.val010; F3=          f.val111;
        Phi0=ls_values.val110; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val111;
        break;
      case 2:
        P0=              P101; P1=              P100; P2=              P111; P3=              P001;
        F0=          f.val101; F1=          f.val100; F2=          f.val111; F3=          f.val001;
        Phi0=ls_values.val101; Phi1=ls_values.val100; Phi2=ls_values.val111; Phi3=ls_values.val001;
        break;
      case 3:
        P0=              P011; P1=              P111; P2=              P010; P3=              P001;
        F0=          f.val011; F1=          f.val111; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val011; Phi1=ls_values.val111; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      case 4:
        P0=              P111; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val111; F1=          f.val100; F2=          f.val010; F3=          f.val001;
        Phi0=ls_values.val111; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
    } else {
      switch(n) {
      case 0:
        P0=              P000; P1=              P100; P2=              P110; P3=              P111;
        F0=          f.val000; F1=          f.val100; F2=          f.val110; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val110; Phi3=ls_values.val111;
        break;
      case 1:
        P0=              P000; P1=              P010; P2=              P110; P3=              P111;
        F0=          f.val000; F1=          f.val010; F2=          f.val110; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val110; Phi3=ls_values.val111;
        break;
      case 2:
        P0=              P000; P1=              P100; P2=              P101; P3=              P111;
        F0=          f.val000; F1=          f.val100; F2=          f.val101; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val101; Phi3=ls_values.val111;
        break;
      case 3:
        P0=              P000; P1=              P010; P2=              P011; P3=              P111;
        F0=          f.val000; F1=          f.val010; F2=          f.val011; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val011; Phi3=ls_values.val111;
        break;
      case 4:
        P0=              P000; P1=              P001; P2=              P101; P3=              P111;
        F0=          f.val000; F1=          f.val001; F2=          f.val101; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val101; Phi3=ls_values.val111;
        break;
      case 5:
        P0=              P000; P1=              P001; P2=              P011; P3=              P111;
        F0=          f.val000; F1=          f.val001; F2=          f.val011; F3=          f.val111;
        Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val011; Phi3=ls_values.val111;
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
    }

    // simple cases
    if(Phi0<=0.0 && Phi1<=0.0 && Phi2<=0.0 && Phi3<=0.0) continue;
    if(Phi0>0.0 && Phi1>0.0 && Phi2>0.0 && Phi3>0.0) continue;
    if(Phi0==0.0 && Phi1==0.0 && Phi2==0.0 && Phi3!=0.0) {return (F0+ F1+F2)/3.*Point3::area(P0,P1,P2);}
    if(Phi0==0.0 && Phi1==0.0 && Phi2!=0.0 && Phi3==0.0) {return (F0+ F1+F3)/3.*Point3::area(P0,P1,P3);}
    if(Phi0==0.0 && Phi1!=0.0 && Phi2==0.0 && Phi3==0.0) {return (F0+ F3+F2)/3.*Point3::area(P0,P3,P2);}
    if(Phi0!=0.0 && Phi1==0.0 && Phi2==0.0 && Phi3==0.0) {return (F3+ F1+F2)/3.*Point3::area(P3,P1,P2);}

    // number_of_negatives = 1,2,3
    int number_of_negatives = 0;

    if(Phi0<=0.0) number_of_negatives++;
    if(Phi1<=0.0) number_of_negatives++;
    if(Phi2<=0.0) number_of_negatives++;
    if(Phi3<=0.0) number_of_negatives++;

    if(number_of_negatives==3)
    {
      Phi0 *= -1;
      Phi1 *= -1;
      Phi2 *= -1;
      Phi3 *= -1;
    }

    // sorting for simplication into two cases,
    if(Phi0>0.0 && Phi1<=0.0) swap(Phi0,Phi1,F0,F1,P0,P1);
    if(Phi0>0.0 && Phi2<=0.0) swap(Phi0,Phi2,F0,F2,P0,P2);
    if(Phi0>0.0 && Phi3<=0.0) swap(Phi0,Phi3,F0,F3,P0,P3);
    if(Phi1>0.0 && Phi2<=0.0) swap(Phi1,Phi2,F1,F2,P1,P2);
    if(Phi1>0.0 && Phi3<=0.0) swap(Phi1,Phi3,F1,F3,P1,P3);
    if(Phi2>0.0 && Phi3<=0.0) swap(Phi2,Phi3,F2,F3,P2,P3);

    //
    if(Phi0<=0.0 && Phi1>0.0 && Phi2>0.0 && Phi3>0.0) // type -+++
    {
      Point3 P_btw_01 = interpol_p(P0,Phi0,P1,Phi1);
      Point3 P_btw_02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P_btw_03 = interpol_p(P0,Phi0,P3,Phi3);

      double F_btw_01 = interpol_f(F0,Phi0,F1,Phi1);
      double F_btw_02 = interpol_f(F0,Phi0,F2,Phi2);
      double F_btw_03 = interpol_f(F0,Phi0,F3,Phi3);

      sum += Point3::area(P_btw_01,P_btw_02,P_btw_03)*
             (F_btw_01+F_btw_02+F_btw_03)/3.;
    }
    else   // type --++ //if (Phi0<=0 && Phi1<=0 && Phi2>=0 && Phi3>=0)
    {
#ifdef CASL_THROWS
      if(Phi0>0.0 || Phi1>0.0 || Phi2<=0.0 || Phi3<=0.0)
        throw std::runtime_error("[CASL_ERROR]: Cube3->integrate_Over_Interface: wrong configuration.");
#endif

      Point3 P_btw_02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P_btw_03 = interpol_p(P0,Phi0,P3,Phi3);
      Point3 P_btw_12 = interpol_p(P1,Phi1,P2,Phi2);
      Point3 P_btw_13 = interpol_p(P1,Phi1,P3,Phi3);

      double F_btw_02 = interpol_f(F0,Phi0,F2,Phi2);
      double F_btw_03 = interpol_f(F0,Phi0,F3,Phi3);
      double F_btw_12 = interpol_f(F1,Phi1,F2,Phi2);
      double F_btw_13 = interpol_f(F1,Phi1,F3,Phi3);

      sum += Point3::area(P_btw_02,P_btw_03,P_btw_13)*
             (F_btw_02+F_btw_03+F_btw_13)/3.;
      sum += Point3::area(P_btw_02,P_btw_12,P_btw_13)*
             (F_btw_02+F_btw_12+F_btw_13)/3.;
    }
  }
  return sum;
}



double Cube3::integrate_Over_Interface(const CF_3 &f, const OctValue &ls_values) const
{
  Point3 P000(x0,y0,z0);
  Point3 P001(x0,y0,z1);
  Point3 P010(x0,y1,z0);
  Point3 P011(x0,y1,z1);
  Point3 P100(x1,y0,z0);
  Point3 P101(x1,y0,z1);
  Point3 P110(x1,y1,z0);
  Point3 P111(x1,y1,z1);

  // simple cases
  if(  ls_values.val000<=0 && ls_values.val001<=0 &&
       ls_values.val010<=0 && ls_values.val011<=0 &&
       ls_values.val100<=0 && ls_values.val101<=0 &&
       ls_values.val110<=0 && ls_values.val111<=0 ) return 0;

  if(  ls_values.val000>=0 && ls_values.val001>=0 &&
       ls_values.val010>=0 && ls_values.val011>=0 &&
       ls_values.val100>=0 && ls_values.val101>=0 &&
       ls_values.val110>=0 && ls_values.val111>=0 ) return 0;

  double sum=0;

  // iteration on each simplex in the middle cut triangulation
  for(int n=0;n<num_tet;n++)
  {
    // Tetrahedron (P0,P1,P2,P3)
    Point3   P0,  P1,  P2,   P3;
    double   F0,  F1,  F2,   F3;
    double Phi0,Phi1,Phi2,Phi3;

    if (middlecut)
    {
      switch(n) {
        case 0:
          P0=              P000; P1=              P100; P2=              P010; P3=              P001;
          Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
          break;
        case 1:
          P0=              P110; P1=              P100; P2=              P010; P3=              P111;
          Phi0=ls_values.val110; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val111;
          break;
        case 2:
          P0=              P101; P1=              P100; P2=              P111; P3=              P001;
          Phi0=ls_values.val101; Phi1=ls_values.val100; Phi2=ls_values.val111; Phi3=ls_values.val001;
          break;
        case 3:
          P0=              P011; P1=              P111; P2=              P010; P3=              P001;
          Phi0=ls_values.val011; Phi1=ls_values.val111; Phi2=ls_values.val010; Phi3=ls_values.val001;
          break;
        case 4:
          P0=              P111; P1=              P100; P2=              P010; P3=              P001;
          Phi0=ls_values.val111; Phi1=ls_values.val100; Phi2=ls_values.val010; Phi3=ls_values.val001;
          break;
        default:

#ifdef CASL_THROWS
          throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
          break;
      }
    } else {
      switch(n) {
        case 0:
          P0=              P000; P1=              P100; P2=              P110; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val110; Phi3=ls_values.val111;
          break;
        case 1:
          P0=              P000; P1=              P010; P2=              P110; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val110; Phi3=ls_values.val111;
          break;
        case 2:
          P0=              P000; P1=              P100; P2=              P101; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val100; Phi2=ls_values.val101; Phi3=ls_values.val111;
          break;
        case 3:
          P0=              P000; P1=              P010; P2=              P011; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val010; Phi2=ls_values.val011; Phi3=ls_values.val111;
          break;
        case 4:
          P0=              P000; P1=              P001; P2=              P101; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val101; Phi3=ls_values.val111;
          break;
        case 5:
          P0=              P000; P1=              P001; P2=              P011; P3=              P111;
          Phi0=ls_values.val000; Phi1=ls_values.val001; Phi2=ls_values.val011; Phi3=ls_values.val111;
          break;
        default:
#ifdef CASL_THROWS
          throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
          break;
      }
    }

    // simple cases
    if(Phi0<0 && Phi1<0 && Phi2<0 && Phi3<0) continue;
    if(Phi0>0 && Phi1>0 && Phi2>0 && Phi3>0) continue;
//    if(Phi0==0 && Phi1==0 && Phi2==0 && Phi3!=0) {return (F0+ F1+F2)/3.*Point3::area(P0,P1,P2);}
//    if(Phi0==0 && Phi1==0 && Phi2!=0 && Phi3==0) {return (F0+ F1+F3)/3.*Point3::area(P0,P1,P3);}
//    if(Phi0==0 && Phi1!=0 && Phi2==0 && Phi3==0) {return (F0+ F3+F2)/3.*Point3::area(P0,P3,P2);}
//    if(Phi0!=0 && Phi1==0 && Phi2==0 && Phi3==0) {return (F3+ F1+F2)/3.*Point3::area(P3,P1,P2);}

    // number_of_negatives = 1,2,3
    int number_of_negatives = 0;

    if(Phi0<0) number_of_negatives++;
    if(Phi1<0) number_of_negatives++;
    if(Phi2<0) number_of_negatives++;
    if(Phi3<0) number_of_negatives++;

    if(number_of_negatives==3)
    {
      Phi0 *= -1;
      Phi1 *= -1;
      Phi2 *= -1;
      Phi3 *= -1;
    }

    // sorting for simplication into two cases,
    if(Phi0>0 && Phi1<0) swap(Phi0,Phi1,F0,F1,P0,P1);
    if(Phi0>0 && Phi2<0) swap(Phi0,Phi2,F0,F2,P0,P2);
    if(Phi0>0 && Phi3<0) swap(Phi0,Phi3,F0,F3,P0,P3);
    if(Phi1>0 && Phi2<0) swap(Phi1,Phi2,F1,F2,P1,P2);
    if(Phi1>0 && Phi3<0) swap(Phi1,Phi3,F1,F3,P1,P3);
    if(Phi2>0 && Phi3<0) swap(Phi2,Phi3,F2,F3,P2,P3);

    //
    if(Phi0<=0 && Phi1>=0 && Phi2>=0 && Phi3>=0) // type -+++
    {
      Point3 P_btw_01 = interpol_p(P0,Phi0,P1,Phi1);
      Point3 P_btw_02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P_btw_03 = interpol_p(P0,Phi0,P3,Phi3);

      double F_btw_01 = f(P_btw_01.x, P_btw_01.y, P_btw_01.z);
      double F_btw_02 = f(P_btw_02.x, P_btw_02.y, P_btw_02.z);
      double F_btw_03 = f(P_btw_03.x, P_btw_03.y, P_btw_03.z);

      sum += Point3::area(P_btw_01,P_btw_02,P_btw_03)*
             (F_btw_01+F_btw_02+F_btw_03)/3.;
    }
    else   // type --++ //if (Phi0<=0 && Phi1<=0 && Phi2>=0 && Phi3>=0)
    {
#ifdef CASL_THROWS
      if(Phi0>0 || Phi1>0 || Phi2<0 || Phi3<0)
        throw std::runtime_error("[CASL_ERROR]: Cube3->integrate_Over_Interface: wrong configuration.");
#endif

      Point3 P_btw_02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P_btw_03 = interpol_p(P0,Phi0,P3,Phi3);
      Point3 P_btw_12 = interpol_p(P1,Phi1,P2,Phi2);
      Point3 P_btw_13 = interpol_p(P1,Phi1,P3,Phi3);

      double F_btw_02 = f(P_btw_02.x, P_btw_02.y, P_btw_02.z);
      double F_btw_03 = f(P_btw_03.x, P_btw_03.y, P_btw_03.z);
      double F_btw_12 = f(P_btw_12.x, P_btw_12.y, P_btw_12.z);
      double F_btw_13 = f(P_btw_13.x, P_btw_13.y, P_btw_13.z);

      sum += Point3::area(P_btw_02,P_btw_03,P_btw_13)*
             (F_btw_02+F_btw_03+F_btw_13)/3.;
      sum += Point3::area(P_btw_02,P_btw_12,P_btw_13)*
             (F_btw_02+F_btw_12+F_btw_13)/3.;
    }
  }
  return sum;
}
