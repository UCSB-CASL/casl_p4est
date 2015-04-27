#include "cube2.h"
#include <petsclog.h>

#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

Cube2::Cube2()
{
  x0=0.; x1=0.;
  y0=0.; y1=0.;
}

Cube2::Cube2(double x0, double x1, double y0, double y1)
{
  this->x0 = x0; this->x1 = x1;
  this->y0 = y0; this->y1 = y1;
}

void Cube2::kuhn_Triangulation(Simplex2& s1, Simplex2& s2 ) const
{
  s1.x0=x0 ; s1.x1=x1 ; s1.x2=x1;
  s1.y0=y0 ; s1.y1=y0 ; s1.y2=y1;

  s2.x0=x0 ; s2.x1=x0 ; s2.x2=x1;
  s2.y0=y0 ; s2.y1=y1 ; s2.y2=y1;
}

double Cube2::area_In_Negative_Domain( QuadValue& level_set_values) const
{
  QuadValue tmp(1.,1.,1.,1.);
  return integral(tmp,level_set_values);
}

double Cube2::integral( QuadValue f ) const
{
  PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);
  return (f.val00+f.val10+f.val01+f.val11)/4.*(x1-x0)*(y1-y0);
}

double Cube2::integral( const QuadValue& f, const QuadValue& level_set_values ) const
{
  if     (level_set_values.val00<=0 && level_set_values.val10<=0 && level_set_values.val01<=0 && level_set_values.val11<=0 ) return integral(f);
  else if(level_set_values.val00> 0 && level_set_values.val10> 0 && level_set_values.val01> 0 && level_set_values.val11> 0 ) return 0;
  else
  {
    Simplex2 S1,S2; kuhn_Triangulation(S1,S2);

    return S1.integral(f.val00,f.val10,f.val11,level_set_values.val00,level_set_values.val10,level_set_values.val11)
        + S2.integral(f.val00,f.val01,f.val11,level_set_values.val00,level_set_values.val01,level_set_values.val11);
  }
}

double Cube2::integrate_Over_Interface( const QuadValue& f, const QuadValue& level_set_values ) const
{
  double sum=0;

  Point2 p00(x0,y0); double f00 = f.val00; double phi00 = level_set_values.val00;
  Point2 p01(x0,y1); double f01 = f.val01; double phi01 = level_set_values.val01;
  Point2 p10(x1,y0); double f10 = f.val10; double phi10 = level_set_values.val10;
  Point2 p11(x1,y1); double f11 = f.val11; double phi11 = level_set_values.val11;

  // simple cases
  if(phi00<=0 && phi01<=0 && phi10<=0 && phi11<=0) return 0;
  if(phi00>=0 && phi01>=0 && phi10>=0 && phi11>=0) return 0;

  // iteration on each simplex in the Kuhn triangulation
  for(int n=0;n<2;n++)
  {
    Point2 p0=p00; double f0=f00; double phi0=phi00;
    Point2 p2=p11; double f2=f11; double phi2=phi11;

    // triangle (P0,P1,P2) with values (F0,F1,F2), (Phi0,Phi1,Phi2)
    Point2   p1 = (n==0) ?   p01 :   p10;
    double   f1 = (n==0) ?   f01 :   f10;
    double phi1 = (n==0) ? phi01 : phi10;

    // simple cases
    if(phi0<=0 && phi1<=0 && phi2<=0) continue;
    if(phi0>=0 && phi1>=0 && phi2>=0) continue;

    //
    int number_of_negatives = 0;

    if(phi0<0) number_of_negatives++;
    if(phi1<0) number_of_negatives++;
    if(phi2<0) number_of_negatives++;

#ifdef CASL_THROWS
    if(number_of_negatives!=1 && number_of_negatives!=2) throw std::runtime_error("[CASL_ERROR]: Wrong configuration.");
#endif

    if(number_of_negatives==2)
    {
      phi0*=-1;
      phi1*=-1;
      phi2*=-1;
    }

    // sorting for simplication into one case
    if(phi0>0 && phi1<0) swap(phi0,phi1,f0,f1,p0,p1);
    if(phi0>0 && phi2<0) swap(phi0,phi2,f0,f2,p0,p2);
    if(phi1>0 && phi2<0) swap(phi1,phi2,f1,f2,p1,p2);

    // type : (-++)
    Point2 p_btw_01 = interpol_p(p0,phi0,p1,phi1); Point2 p_btw_02 = interpol_p(p0,phi0,p2,phi2);
    double f_btw_01 = interpol_f(f0,phi0,f1,phi1); double f_btw_02 = interpol_f(f0,phi0,f2,phi2);

    double length_of_line_segment = (p_btw_02 - p_btw_01).norm_L2();

    sum += length_of_line_segment * (f_btw_02 + f_btw_01)/2.;

    PetscErrorCode ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }

  return sum;
}
