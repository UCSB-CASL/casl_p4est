#include "cube3.h"

Cube3::Cube3()
{
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    xyz_mmm[dir] = xyz_ppp[dir] = 0.0;
  middlecut = false;
  num_tet = 6;
//  middlecut = true;
//  num_tet = 5;
}

Cube3::Cube3(double x0, double x1, double y0, double y1, double z0, double z1)
{
  this->xyz_mmm[0] = x0; this->xyz_ppp[0] = x1;
  this->xyz_mmm[1] = y0; this->xyz_ppp[1] = y1;
  this->xyz_mmm[2] = z0; this->xyz_ppp[2] = z1;
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

  Point3 P000(xyz_mmm[0], xyz_mmm[1], xyz_mmm[2]);
  Point3 P001(xyz_mmm[0], xyz_mmm[1], xyz_ppp[2]);
  Point3 P010(xyz_mmm[0], xyz_ppp[1], xyz_mmm[2]);
  Point3 P011(xyz_mmm[0], xyz_ppp[1], xyz_ppp[2]);
  Point3 P100(xyz_ppp[0], xyz_mmm[1], xyz_mmm[2]);
  Point3 P101(xyz_ppp[0], xyz_mmm[1], xyz_ppp[2]);
  Point3 P110(xyz_ppp[0], xyz_ppp[1], xyz_mmm[2]);
  Point3 P111(xyz_ppp[0], xyz_ppp[1], xyz_ppp[2]);

  // simple cases
  if(  ls_values.val[0] <= 0.0 && ls_values.val[1] <= 0.0 &&
       ls_values.val[2] <= 0.0 && ls_values.val[3] <= 0.0 &&
       ls_values.val[4] <= 0.0 && ls_values.val[5] <= 0.0 &&
       ls_values.val[6] <= 0.0 && ls_values.val[7] <= 0.0 )
    return  (xyz_ppp[0]-xyz_mmm[0])*(xyz_ppp[1]-xyz_mmm[1])*(xyz_ppp[2]-xyz_mmm[2])*(f.val[0]+f.val[1]+f.val[2]+f.val[3]+f.val[4]+f.val[5]+f.val[6]+f.val[7])/8.;

  if(  ls_values.val[0] >= 0.0 && ls_values.val[1] >= 0.0 &&
       ls_values.val[2] >= 0.0 && ls_values.val[3] >= 0.0 &&
       ls_values.val[4] >= 0.0 && ls_values.val[5] >= 0.0 &&
       ls_values.val[6] >= 0.0 && ls_values.val[7] >= 0.0 ) return 0.0;

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
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
        break;
      case 1:
        P0=              P110; P1=              P100; P2=              P010; P3=              P111;
        F0=          f.val[6]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[7];
        Phi0=ls_values.val[6]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[7];
        break;
      case 2:
        P0=              P101; P1=              P100; P2=              P111; P3=              P001;
        F0=          f.val[5]; F1=          f.val[4]; F2=          f.val[7]; F3=          f.val[1];
        Phi0=ls_values.val[5]; Phi1=ls_values.val[4]; Phi2=ls_values.val[7]; Phi3=ls_values.val[1];
        break;
      case 3:
        P0=              P011; P1=              P111; P2=              P010; P3=              P001;
        F0=          f.val[3]; F1=          f.val[7]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[3]; Phi1=ls_values.val[7]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
        break;
      case 4:
        P0=              P111; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val[7]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[7]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
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
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[6]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
        break;
      case 1:
        P0=              P000; P1=              P010; P2=              P110; P3=              P111;
        F0=          f.val[0]; F1=          f.val[2]; F2=          f.val[6]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
        break;
      case 2:
        P0=              P000; P1=              P100; P2=              P101; P3=              P111;
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[5]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
        break;
      case 3:
        P0=              P000; P1=              P010; P2=              P011; P3=              P111;
        F0=          f.val[0]; F1=          f.val[2]; F2=          f.val[3]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
        break;
      case 4:
        P0=              P000; P1=              P001; P2=              P101; P3=              P111;
        F0=          f.val[0]; F1=          f.val[1]; F2=          f.val[5]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
        break;
      case 5:
        P0=              P000; P1=              P001; P2=              P011; P3=              P111;
        F0=          f.val[0]; F1=          f.val[1]; F2=          f.val[3]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
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
    if(Phi0 <= 0.0 && Phi1 > 0.0 && Phi2 > 0.0 && Phi3 > 0.0) // type -+++
    {
      Point3 P01 = interpol_p(P0,Phi0,P1,Phi1);
      Point3 P02 = interpol_p(P0,Phi0,P2,Phi2);
      Point3 P03 = interpol_p(P0,Phi0,P3,Phi3);

      double F01 = interpol_f(F0,Phi0,F1,Phi1);
      double F02 = interpol_f(F0,Phi0,F2,Phi2);
      double F03 = interpol_f(F0,Phi0,F3,Phi3);

      sum += Point3::volume(P0,P01,P02,P03)*(F0+F01+F02+F03)/4.;
    }
    else if(Phi0 <= 0.0 && Phi1 <= 0.0 && Phi2 > 0.0 && Phi3 > 0.0) // type --++
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
      if(Phi0 > 0.0 || Phi1 > 0.0 || Phi2 > 0.0 || Phi3 <= 0.0)
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: wrong configuration.");
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
  Point3 P000(xyz_mmm[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P001(xyz_mmm[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P010(xyz_mmm[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P011(xyz_mmm[0],xyz_ppp[1],xyz_ppp[2]);
  Point3 P100(xyz_ppp[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P101(xyz_ppp[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P110(xyz_ppp[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P111(xyz_ppp[0],xyz_ppp[1],xyz_ppp[2]);
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
  if(  ls_values.val[0] <= 0.0 && ls_values.val[1] <= 0.0 &&
       ls_values.val[2] <= 0.0 && ls_values.val[3] <= 0.0 &&
       ls_values.val[4] <= 0.0 && ls_values.val[5] <= 0.0 &&
       ls_values.val[6] <= 0.0 && ls_values.val[7] <= 0.0 ) return 0;

  if(  ls_values.val[0] > 0.0 && ls_values.val[1] > 0.0 &&
       ls_values.val[2] > 0.0 && ls_values.val[3] > 0.0 &&
       ls_values.val[4] > 0.0 && ls_values.val[5] > 0.0 &&
       ls_values.val[6] > 0.0 && ls_values.val[7] > 0.0 ) return 0;



  double sum = 0.0;
  double cube_diag = (P111 - P000).norm_L2();

  // iteration on each simplex in the middle cut triangulation
  for(int n = 0; n < num_tet; n++)
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
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
        break;
      case 1:
        P0=              P110; P1=              P100; P2=              P010; P3=              P111;
        F0=          f.val[6]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[7];
        Phi0=ls_values.val[6]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[7];
        break;
      case 2:
        P0=              P101; P1=              P100; P2=              P111; P3=              P001;
        F0=          f.val[5]; F1=          f.val[4]; F2=          f.val[7]; F3=          f.val[1];
        Phi0=ls_values.val[5]; Phi1=ls_values.val[4]; Phi2=ls_values.val[7]; Phi3=ls_values.val[1];
        break;
      case 3:
        P0=              P011; P1=              P111; P2=              P010; P3=              P001;
        F0=          f.val[3]; F1=          f.val[7]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[3]; Phi1=ls_values.val[7]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
        break;
      case 4:
        P0=              P111; P1=              P100; P2=              P010; P3=              P001;
        F0=          f.val[7]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
        Phi0=ls_values.val[7]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
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
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[6]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
        break;
      case 1:
        P0=              P000; P1=              P010; P2=              P110; P3=              P111;
        F0=          f.val[0]; F1=          f.val[2]; F2=          f.val[6]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
        break;
      case 2:
        P0=              P000; P1=              P100; P2=              P101; P3=              P111;
        F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[5]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
        break;
      case 3:
        P0=              P000; P1=              P010; P2=              P011; P3=              P111;
        F0=          f.val[0]; F1=          f.val[2]; F2=          f.val[3]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
        break;
      case 4:
        P0=              P000; P1=              P001; P2=              P101; P3=              P111;
        F0=          f.val[0]; F1=          f.val[1]; F2=          f.val[5]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
        break;
      case 5:
        P0=              P000; P1=              P001; P2=              P011; P3=              P111;
        F0=          f.val[0]; F1=          f.val[1]; F2=          f.val[3]; F3=          f.val[7];
        Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
    }

    // simple cases
    if(Phi0 < 0.0 && Phi1 < 0.0 && Phi2 < 0.0 && Phi3 < 0.0) continue;
    if(Phi0 > 0.0 && Phi1 > 0.0 && Phi2 > 0.0 && Phi3 > 0.0) continue;
    if((Phi0 <= 0.0 && Phi1 <= 0.0 && Phi2 <= 0.0 && Phi3 <= 0.0) || (Phi0 >= 0.0 && Phi1 >= 0.0 && Phi2 >= 0.0 && Phi3 >= 0.0))
    {
      // all positive/negative but maybe a few 0
      // count the number of 0
      short nb_zeros = 0, non_zero_idx = 0;
      bool is_point0_zero = (fabs(Phi0) < EPS*cube_diag); nb_zeros += (is_point0_zero ? 1 : 0); non_zero_idx = (!is_point0_zero ? 0 : non_zero_idx);
      bool is_point1_zero = (fabs(Phi1) < EPS*cube_diag); nb_zeros += (is_point1_zero ? 1 : 0); non_zero_idx = (!is_point1_zero ? 1 : non_zero_idx);
      bool is_point2_zero = (fabs(Phi2) < EPS*cube_diag); nb_zeros += (is_point2_zero ? 1 : 0); non_zero_idx = (!is_point2_zero ? 2 : non_zero_idx);
      bool is_point3_zero = (fabs(Phi3) < EPS*cube_diag); nb_zeros += (is_point3_zero ? 1 : 0); non_zero_idx = (!is_point3_zero ? 3 : non_zero_idx);
      // nothing to do and continue if not 3 (either 0-measure subset of the face, or all 0 at the tetrahedron nodes...)
      if(nb_zeros !=3)
        continue;
      // there are exactly three points with phi == 0, --> a face of the tetrahedron is excalty on the interface
      double triangle_area;
      switch (non_zero_idx) {
      case 0:
        triangle_area = Point3::area(P1, P2, P3);
        break;
      case 1:
        triangle_area = Point3::area(P0, P2, P3);
        break;
      case 2:
        triangle_area = Point3::area(P0, P1, P3);
        break;
      case 3:
        triangle_area = Point3::area(P0, P1, P2);
        break;
      default:
#ifdef CASL_THROWS
        throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
        break;
      }
      sum += 0.5*triangle_area*((is_point0_zero ? F0 : 0.0) + (is_point1_zero ? F1 : 0.0) + (is_point2_zero ? F2 : 0.0) + (is_point3_zero ? F3 : 0.0))/3.0;
      // "0.5*" because the face is shared with another tetrahedron (either in this cube or another), which will add its own (same) contribution too...
      // FIXME: only wrong if the face in question is a non-periodic wall...
    }


    // number_of_negatives = 1,2,3
    int number_of_negatives = 0;

    if(Phi0<=0.0) number_of_negatives++;
    if(Phi1<=0.0) number_of_negatives++;
    if(Phi2<=0.0) number_of_negatives++;
    if(Phi3<=0.0) number_of_negatives++;

    if(number_of_negatives==3)
    {
      Phi0 *= -1.0;
      Phi1 *= -1.0;
      Phi2 *= -1.0;
      Phi3 *= -1.0;
    }

    // sorting for simplication into two cases,
    if(Phi0 > 0.0 && Phi1 <= 0.0) swap(Phi0, Phi1, F0, F1, P0, P1);
    if(Phi0 > 0.0 && Phi2 <= 0.0) swap(Phi0, Phi2, F0, F2, P0, P2);
    if(Phi0 > 0.0 && Phi3 <= 0.0) swap(Phi0, Phi3, F0, F3, P0, P3);
    if(Phi1 > 0.0 && Phi2 <= 0.0) swap(Phi1, Phi2, F1, F2, P1,  P2);
    if(Phi1 > 0.0 && Phi3 <= 0.0) swap(Phi1, Phi3, F1, F3, P1, P3);
    if(Phi2 > 0.0 && Phi3 <= 0.0) swap(Phi2, Phi3, F2, F3, P2, P3);

    //
    if(Phi0 <= 0.0 && Phi1 > 0.0 && Phi2 > 0.0 && Phi3 > 0.0) // type -+++
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
      if(Phi0 > 0.0 || Phi1 > 0.0 || Phi2 <= 0.0 || Phi3 <= 0.0)
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
  Point3 P000(xyz_mmm[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P001(xyz_mmm[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P010(xyz_mmm[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P011(xyz_mmm[0],xyz_ppp[1],xyz_ppp[2]);
  Point3 P100(xyz_ppp[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P101(xyz_ppp[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P110(xyz_ppp[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P111(xyz_ppp[0],xyz_ppp[1],xyz_ppp[2]);

  // simple cases
  if(  ls_values.val[0]<=0 && ls_values.val[1]<=0 &&
       ls_values.val[2]<=0 && ls_values.val[3]<=0 &&
       ls_values.val[4]<=0 && ls_values.val[5]<=0 &&
       ls_values.val[6]<=0 && ls_values.val[7]<=0 ) return 0;

  if(  ls_values.val[0]>=0 && ls_values.val[1]>=0 &&
       ls_values.val[2]>=0 && ls_values.val[3]>=0 &&
       ls_values.val[4]>=0 && ls_values.val[5]>=0 &&
       ls_values.val[6]>=0 && ls_values.val[7]>=0 ) return 0;

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
          Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
          break;
        case 1:
          P0=              P110; P1=              P100; P2=              P010; P3=              P111;
          Phi0=ls_values.val[6]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[7];
          break;
        case 2:
          P0=              P101; P1=              P100; P2=              P111; P3=              P001;
          Phi0=ls_values.val[5]; Phi1=ls_values.val[4]; Phi2=ls_values.val[7]; Phi3=ls_values.val[1];
          break;
        case 3:
          P0=              P011; P1=              P111; P2=              P010; P3=              P001;
          Phi0=ls_values.val[3]; Phi1=ls_values.val[7]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
          break;
        case 4:
          P0=              P111; P1=              P100; P2=              P010; P3=              P001;
          Phi0=ls_values.val[7]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
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
          Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
          break;
        case 1:
          P0=              P000; P1=              P010; P2=              P110; P3=              P111;
          Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[6]; Phi3=ls_values.val[7];
          break;
        case 2:
          P0=              P000; P1=              P100; P2=              P101; P3=              P111;
          Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
          break;
        case 3:
          P0=              P000; P1=              P010; P2=              P011; P3=              P111;
          Phi0=ls_values.val[0]; Phi1=ls_values.val[2]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
          break;
        case 4:
          P0=              P000; P1=              P001; P2=              P101; P3=              P111;
          Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[5]; Phi3=ls_values.val[7];
          break;
        case 5:
          P0=              P000; P1=              P001; P2=              P011; P3=              P111;
          Phi0=ls_values.val[0]; Phi1=ls_values.val[1]; Phi2=ls_values.val[3]; Phi3=ls_values.val[7];
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

double Cube3::max_Over_Interface(const OctValue &f, const OctValue &ls_values) const
{
  Point3 P000(xyz_mmm[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P001(xyz_mmm[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P010(xyz_mmm[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P011(xyz_mmm[0],xyz_ppp[1],xyz_ppp[2]);
  Point3 P100(xyz_ppp[0],xyz_mmm[1],xyz_mmm[2]);
  Point3 P101(xyz_ppp[0],xyz_mmm[1],xyz_ppp[2]);
  Point3 P110(xyz_ppp[0],xyz_ppp[1],xyz_mmm[2]);
  Point3 P111(xyz_ppp[0],xyz_ppp[1],xyz_ppp[2]);

  // simple cases
  if(  ls_values.val[0]<=0 && ls_values.val[1]<=0 &&
       ls_values.val[2]<=0 && ls_values.val[3]<=0 &&
       ls_values.val[4]<=0 && ls_values.val[5]<=0 &&
       ls_values.val[6]<=0 && ls_values.val[7]<=0 ) return -DBL_MAX;

  if(  ls_values.val[0]>=0 && ls_values.val[1]>=0 &&
       ls_values.val[2]>=0 && ls_values.val[3]>=0 &&
       ls_values.val[4]>=0 && ls_values.val[5]>=0 &&
       ls_values.val[6]>=0 && ls_values.val[7]>=0 ) return -DBL_MAX;

  double my_max = -DBL_MAX;
  double cube_diag = (P111-P000).norm_L2();

  // iteration on each simplex in the middle cut triangulation
  for(int n=0;n<5;n++)
  {
    // Tetrahedron (P0,P1,P2,P3)
    Point3   P0,  P1,  P2,   P3;
    double   F0,  F1,  F2,   F3;
    double Phi0,Phi1,Phi2,Phi3;

    switch(n) {
    case 0:
      P0=              P000; P1=              P100; P2=              P010; P3=              P001;
      F0=          f.val[0]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
      Phi0=ls_values.val[0]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
      break;
    case 1:
      P0=              P110; P1=              P100; P2=              P010; P3=              P111;
      F0=          f.val[6]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[7];
      Phi0=ls_values.val[6]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[7];
      break;
    case 2:
      P0=              P101; P1=              P100; P2=              P111; P3=              P001;
      F0=          f.val[5]; F1=          f.val[4]; F2=          f.val[7]; F3=          f.val[1];
      Phi0=ls_values.val[5]; Phi1=ls_values.val[4]; Phi2=ls_values.val[7]; Phi3=ls_values.val[1];
      break;
    case 3:
      P0=              P011; P1=              P111; P2=              P010; P3=              P001;
      F0=          f.val[3]; F1=          f.val[7]; F2=          f.val[2]; F3=          f.val[1];
      Phi0=ls_values.val[3]; Phi1=ls_values.val[7]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
      break;
    case 4:
      P0=              P111; P1=              P100; P2=              P010; P3=              P001;
      F0=          f.val[7]; F1=          f.val[4]; F2=          f.val[2]; F3=          f.val[1];
      Phi0=ls_values.val[7]; Phi1=ls_values.val[4]; Phi2=ls_values.val[2]; Phi3=ls_values.val[1];
      break;
    default:


#ifdef CASL_THROWS
      throw std::runtime_error("[CASL_ERROR]: Cube3->integral: error.");
#endif
      break;
    }

    // simple cases
    if(Phi0<0 && Phi1<0 && Phi2<0 && Phi3<0) continue;
    if(Phi0>0 && Phi1>0 && Phi2>0 && Phi3>0) continue;
    if((Phi0<=0 && Phi1<=0 && Phi2<=0 && Phi3<=0) || (Phi0>=0 && Phi1>=0 && Phi2>=0 && Phi3>=0))
    {
      // all positive/negative but maybe a few 0
      // count the number of 0
      short nb_zeros = 0, non_zero_idx = 0;
      bool is_point0_zero = (fabs(Phi0) < EPS*cube_diag); nb_zeros += (is_point0_zero?1:0); non_zero_idx = ((!is_point0_zero)?0:non_zero_idx);
      bool is_point1_zero = (fabs(Phi1) < EPS*cube_diag); nb_zeros += (is_point1_zero?1:0); non_zero_idx = ((!is_point1_zero)?1:non_zero_idx);
      bool is_point2_zero = (fabs(Phi2) < EPS*cube_diag); nb_zeros += (is_point2_zero?1:0); non_zero_idx = ((!is_point2_zero)?2:non_zero_idx);
      bool is_point3_zero = (fabs(Phi3) < EPS*cube_diag); nb_zeros += (is_point3_zero?1:0); non_zero_idx = ((!is_point3_zero)?3:non_zero_idx);
      // nothing to do and continue if not 3 (either 0-measure subset of the face, or all 0 at the tetrahedron nodes...)
      if(nb_zeros !=3)
        continue;
      // there are exactly three points with phi == 0, --> a face of the tetrahedron is excalty on the interface
      my_max = MAX(my_max, MAX((is_point0_zero?F0:-DBL_MAX), MAX((is_point1_zero?F1:-DBL_MAX), MAX((is_point2_zero?F2:-DBL_MAX), (is_point3_zero?F3:-DBL_MAX)))));
    }


    // number_of_negatives = 1,2,3
    int number_of_negatives = 0;

    if(Phi0<0) number_of_negatives++;
    if(Phi1<0) number_of_negatives++;
    if(Phi2<0) number_of_negatives++;
    if(Phi3<0) number_of_negatives++;

    if(number_of_negatives==3)
    {
      Phi0 *= -1.0;
      Phi1 *= -1.0;
      Phi2 *= -1.0;
      Phi3 *= -1.0;
    }

    // sorting for simplication into two cases,
    if(Phi0>=0 && Phi1<0) swap(Phi0,Phi1,F0,F1,P0,P1);
    if(Phi0>=0 && Phi2<0) swap(Phi0,Phi2,F0,F2,P0,P2);
    if(Phi0>=0 && Phi3<0) swap(Phi0,Phi3,F0,F3,P0,P3);
    if(Phi1>=0 && Phi2<0) swap(Phi1,Phi2,F1,F2,P1,P2);
    if(Phi1>=0 && Phi3<0) swap(Phi1,Phi3,F1,F3,P1,P3);
    if(Phi2>=0 && Phi3<0) swap(Phi2,Phi3,F2,F3,P2,P3);

    //
    if(Phi0<0 && Phi1>=0 && Phi2>=0 && Phi3>=0) // type -+++
    {
      double F_btw_01 = interpol_f(F0,Phi0,F1,Phi1);
      double F_btw_02 = interpol_f(F0,Phi0,F2,Phi2);
      double F_btw_03 = interpol_f(F0,Phi0,F3,Phi3);

      my_max = MAX(my_max, MAX(F_btw_01, MAX(F_btw_02, F_btw_03)));
    }
    else   // type --++ //if (Phi0<=0 && Phi1<=0 && Phi2>=0 && Phi3>=0)
    {
#ifdef CASL_THROWS
      if(Phi0>=0 || Phi1>=0 || Phi2<0 || Phi3<0)
        throw std::runtime_error("[CASL_ERROR]: Cube3->integrate_Over_Interface: wrong configuration.");
#endif

      double F_btw_02 = interpol_f(F0,Phi0,F2,Phi2);
      double F_btw_03 = interpol_f(F0,Phi0,F3,Phi3);
      double F_btw_12 = interpol_f(F1,Phi1,F2,Phi2);
      double F_btw_13 = interpol_f(F1,Phi1,F3,Phi3);

      my_max = MAX(my_max, MAX(F_btw_02, MAX(F_btw_03, MAX(F_btw_13, F_btw_12))));
    }
  }
  return my_max;
}
