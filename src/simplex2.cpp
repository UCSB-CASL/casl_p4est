#include "math.h"
#ifdef P4_TO_P8
#include "simplex3.h"
#include "my_p8est_utils.h"
#else
#include "simplex2.h"
#include "my_p4est_utils.h"
#endif
#include <petsclog.h>
#include <src/math.h>

#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double Simplex2::area( double x0, double y0,
                       double x1, double y1,
                       double x2, double y2)
{
  double x10=x1-x0; double x20=x2-x0;
  double y10=y1-y0; double y20=y2-y0;

  double v = (x10*y20-x20*y10)*.5;

  PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);

  return (v>0) ? v : -v;
}

double Simplex2::area( const Point2& P0,
                       const Point2& P1,
                       const Point2& P2)
{
  double x10=P1.x-P0.x; double x20=P2.x-P0.x;
  double y10=P1.y-P0.y; double y20=P2.y-P0.y;

  double v = (x10*y20-x20*y10)*.5;

  PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);

  return (v>0) ? v : -v;
}

double Simplex2::area() const
{
  double x10=x1-x0; double x20=x2-x0;
  double y10=y1-y0; double y20=y2-y0;

  double v = (x10*y20-x20*y10)*.5;

  PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);

  return (v>0) ? v : -v;
}

double Simplex2::integral( double f0, double f1, double f2,
                           double p0, double p1, double p2)
{
  if(fabs(p0)<EPS) p0=1e-13;
  if(fabs(p1)<EPS) p1=1e-13;
  if(fabs(p2)<EPS) p2=1e-13;

  //---------------------------------------------------------------------
  // perturbation
  //---------------------------------------------------------------------
  if     (p0<0 && p1<0 && p2<0){ return area()*(f0+f1+f2)/3.; }
  else if(p0>0 && p1>0 && p2>0){ return 0.; }
  else
  {
    //---------------------------------------------------------------------
    // sorting to --+ or -++
    //---------------------------------------------------------------------
    double tmp;

    if(p0>p1)
    {
      tmp=x0; x0=x1; x1=tmp;
      tmp=y0; y0=y1; y1=tmp;
      tmp=p0; p0=p1; p1=tmp;
      tmp=f0; f0=f1; f1=tmp;
    }

    if(p0>p2)
    {
      tmp=x0; x0=x2; x2=tmp;
      tmp=y0; y0=y2; y2=tmp;
      tmp=p0; p0=p2; p2=tmp;
      tmp=f0; f0=f2; f2=tmp;
    }

    if(p1>p2)
    {
      tmp=x1; x1=x2; x2=tmp;
      tmp=y1; y1=y2; y2=tmp;
      tmp=p1; p1=p2; p2=tmp;
      tmp=f1; f1=f2; f2=tmp;
    }

    //---------------------------------------------------------------------
    // count number of negatives
    //---------------------------------------------------------------------
    int number_of_negatives = 0;
    if(p0<0) number_of_negatives++;
    if(p1<0) number_of_negatives++;
    if(p2<0) number_of_negatives++;

    //---------------------------------------------------------------------
    // -++ type
    //---------------------------------------------------------------------
    if(number_of_negatives==1)
    {
      //---------------------------------------------------------------------
      // interface points
      //---------------------------------------------------------------------
      double x01 = (x0*p1-x1*p0)/(p1-p0);
      double y01 = (y0*p1-y1*p0)/(p1-p0);
      double f01 = (f0*p1-f1*p0)/(p1-p0);

      double x02 = (x0*p2-x2*p0)/(p2-p0);
      double y02 = (y0*p2-y2*p0)/(p2-p0);
      double f02 = (f0*p2-f2*p0)/(p2-p0);

      Simplex2 S1;	S1.x0=x0; S1.x1=x01; S1.x2=x02;
      S1.y0=y0; S1.y1=y01; S1.y2=y02;

      PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);

      return S1.area()*(f0+f01+f02)/3.;
    }
    //---------------------------------------------------------------------
    // --+ type
    //---------------------------------------------------------------------
    else
    {
      //---------------------------------------------------------------------
      // interface points
      //---------------------------------------------------------------------
      double x12 = (x1*p2-x2*p1)/(p2-p1);
      double y12 = (y1*p2-y2*p1)/(p2-p1);
      double f12 = (f1*p2-f2*p1)/(p2-p1);

      double x02 = (x0*p2-x2*p0)/(p2-p0);
      double y02 = (y0*p2-y2*p0)/(p2-p0);
      double f02 = (f0*p2-f2*p0)/(p2-p0);

      Simplex2 S1; S1.x0=x0; S1.x1=x1; S1.x2=x12;
      S1.y0=y0; S1.y1=y1; S1.y2=y12;

      Simplex2 S2; S2.x0=x0; S2.x1=x02; S2.x2=x12;
      S2.y0=y0; S2.y1=y02; S2.y2=y12;

      PetscErrorCode ierr = PetscLogFlops(10); CHKERRXX(ierr);

      return S1.area()*(f0+f1 +f12)/3. +
          S2.area()*(f0+f12+f02)/3. ;
    }
  }
}


double Simplex2::integrate_Over_Interface( double f0, double f1, double f2,
                                           double p0, double p1, double p2)
{
  /* simple cases */
  if(p0<=0 && p1<=0 && p2<=0) return 0;
  if(p0>=0 && p1>=0 && p2>=0) return 0;


  int number_of_negatives = 0;
  if(p0<0) number_of_negatives++;
  if(p1<0) number_of_negatives++;
  if(p2<0) number_of_negatives++;

#ifdef CASL_THROWS
  if(number_of_negatives!=1 && number_of_negatives!=2) throw std::runtime_error("[CASL_ERROR]: Simplex2->integrate_Over_Interface: Wrong configuration.");
#endif

  if(number_of_negatives==2)
  {
    p0*=-1; p1*=-1; p2*=-1;
  }

  /* sorting for simplification into one case */
  if(p0>0 && p1<0) { swap(p0,p1); swap(f0,f1); swap(x0,x1); swap(y0,y1); }
  if(p0>0 && p2<0) { swap(p0,p2); swap(f0,f2); swap(x0,x2); swap(y0,y2); }
  if(p1>0 && p2<0) { swap(p1,p2); swap(f1,f2); swap(x1,x2); swap(y1,y2); }

  /* type : (-++) */
  Point2 p_btw_01 = interpol_p(x0, y0, p0, x1, y1, p1); Point2 p_btw_02 = interpol_p(x0, y0, p0, x2, y2, p2);
  double f_btw_01 = interpol_f(f0, p0, f1, p1); double f_btw_02 = interpol_f(f0, p0, f2, p2);

  double length_of_line_segment = (p_btw_02 - p_btw_01).norm_L2();

  return length_of_line_segment * (f_btw_02 + f_btw_01)/2.;
}




double Simplex2::integrate_Over_Interface( const CF_2& f,
                                           double p0, double p1, double p2)
{
  /* simple cases */
  if(p0<=0 && p1<=0 && p2<=0) return 0;
  if(p0>=0 && p1>=0 && p2>=0) return 0;


  int number_of_negatives = 0;
  if(p0<0) number_of_negatives++;
  if(p1<0) number_of_negatives++;
  if(p2<0) number_of_negatives++;

#ifdef CASL_THROWS
  if(number_of_negatives!=1 && number_of_negatives!=2) throw std::runtime_error("[CASL_ERROR]: Simplex2->integrate_Over_Interface: Wrong configuration.");
#endif

  if(number_of_negatives==2)
  {
    p0*=-1; p1*=-1; p2*=-1;
  }

  /* sorting for simplification into one case */
  if(p0>0 && p1<0) { swap(p0,p1); swap(x0,x1); swap(y0,y1); }
  if(p0>0 && p2<0) { swap(p0,p2); swap(x0,x2); swap(y0,y2); }
  if(p1>0 && p2<0) { swap(p1,p2); swap(x1,x2); swap(y1,y2); }

  /* type : (-++) */
  Point2 p_btw_01 = interpol_p(x0, y0, p0, x1, y1, p1); Point2 p_btw_02 = interpol_p(x0, y0, p0, x2, y2, p2);
  double f_btw_01 = f(p_btw_01.x, p_btw_01.y); double f_btw_02 = f(p_btw_02.x, p_btw_02.y);

  double length_of_line_segment = (p_btw_02 - p_btw_01).norm_L2();

  return length_of_line_segment * (f_btw_02 + f_btw_01)/2.;
}
