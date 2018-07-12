#ifndef MY_P4EST_MATH_H
#define MY_P4EST_MATH_H

#include <math.h>
#include <stdexcept>
#include <limits>

#include "petsc_compatibility.h"

#undef MIN
#undef MAX

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
/*!
 * \file math.h
 * \brief This file contains the basic mathematical functions used throughout the library
 */

// Some Macros
#define _CASL_EPS_ 1e-13

template <typename T>
inline T ABS(const T& val)
{
  return val>0 ? val : -val;
}

template <typename T>
inline T SQR(const T& val)
{
  return  val*val;
}

inline int mod(int a, int b)
{
#ifdef CASL_THROWS
  if (b==0) throw std::domain_error("[CASL_ERROR]: trying to take modulo (a,b) whith b=0.");
#endif

  int c = a%b;
  if(c<0) c+=b;
  return c;
}

inline double DELTA( double x, double h )
{
  if( x > h ) return 0;
  if( x <-h ) return 0;
  else      return (1+cos(M_PI*x/h))*0.5/h;
}

inline double HVY( double x, double h )
{
  if( x > h ) return 1;
  if( x <-h ) return 0;
  else      return (1+x/h+sin(M_PI*x/h)/M_PI)*0.5;
}

inline double SIGN(double a)
{
  return (a>0) ? 1:-1;
}

inline double MINMOD( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    if((fabs(a))<(fabs(b))) return a;
    else                    return b;
  }
}

inline double HARMOD( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    if(a<0) a=-a;
    if(b<0) b=-b;

    return 2*a*b/(a+b);
  }
}

inline double ENO2( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    if((ABS(a))<(ABS(b))) return a;
    else                  return b;
  }
}

inline double SUPERBEE( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    double theta = b/a;
    if(theta<0.5) return 2*b;
    if(theta<1.0) return a;
    if(theta<2.0) return b;
    else          return 2*a;
  }
}


inline double KOREN( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    double theta = b/a;
    if(theta<0.4) return 2.*b;
    if(theta<4.0) return (2*a+b)/3.;
    else          return 2.*a;
  }
}


inline double MC( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
//    double theta = b/a;
//    if(theta<1./3.) return 2.*b;
//    if(theta<3.0) return (a+b)/2.;
//    else          return 2.*a;

    if (3.*fabs(b) < fabs(a)) return 2.*b;
    if (fabs(b) < 3.*fabs(a)) return (a+b)/2.;
    else                      return 2.*a;
  }
}


inline double MC2( double a, double b )
{
  return (a+b)/2.;
  if(a*b<=0) return 0;
  else return (a+b)/2.;
}


template <typename T>
inline const T& MIN(const T& val1, const T& val2)
{
  return  val1 < val2 ? val1 : val2;
}

template <typename T>
inline const T& MIN(const T& val1, const T& val2, const T& val3)
{
  return  val1 < val2 ? MIN(val1,val3) : MIN(val2,val3);
}

template <typename T>
inline const T& MAX(const T& val1, const T& val2)
{
  return  val2 < val1 ? val1 : val2;
}

template <typename T>
inline const T& MAX(const T& val1, const T& val2, const T& val3)
{
  return  val2 < val1 ? MAX(val1,val3) : MAX(val2,val3);
}

template <typename T>
inline const T& CLAMP(const T& v, const T& vmin, const T& vmax)
{
  if (v > vmax)
    return vmax;
  else if (v < vmin)
    return vmin;
  else
    return v;
}

template <typename T>
inline bool ISNAN(T x)
{
  return x != x;
}

template <typename T>
inline bool ISINF(T x)
{
  return !ISNAN(x) && ISNAN(x - x);
}

/*!
* \brief returns the SGNMAX of the two parameters
* \note This is an operation used in Tsai 2003 (SIAM J. Num. Anal)
*/
template <typename T>
inline const T SGNMAX(T x, T y)
{
  T zero=0, xp=MAX(x,zero), ym=-MIN(y,zero);
  if(MAX(xp,ym)==xp)
    return xp;
  else
    return -ym;
}

/*!
 * \brief Check if the local data of a vector contains a NAN
 * \param v the vector to check
 * \return true if the vector contains a NAN, false otherwise
 */
bool VecIsNan(Vec v);

/*!
 * \brief Check if a ghosted vector contains a NAN
 * \param v the vector to check
 * \return true if the vector contains a NAN, false otherwise
 */
bool VecGhostIsNan(Vec v);

/*!
* \brief return the CuBi RooT of x
*/
template <class  T>
double CBRT(const T& x);

/*!
 * \brief compute the fraction of an interval covered by the irregular (\f$ \phi \leq 0 \f$) between two points
 * \param phi0 the level-set value at the first point
 * \param phi1 the level-set value at the second point
 * \param dx the minimum resolution in the first direction
 * \param dy the minimum resolution in the second direction
 * \return the fraction of the interval covered by the irregular domain
 */
double fraction_Interval_Covered_By_Irregular_Domain( double phi0, double phi1, double dx, double dy );

/*!
 * \brief compute the fraction of an interval covered by the irregular (\f$ \phi \leq 0 \f$) between two points using second order order derivatives of the level-set function
 * \param phi0 the level-set value at the first point
 * \param phi1 the level-set value at the second point
 * \param dx the minimum resolution in the first direction
 * \param dy the minimum resolution in the second direction
 * \param phi0xx the 2nd order derivative of the level-set at the first point
 * \param phi1xx the 2nd order derivative of the level-set at the second point
 * \return the fraction of the interval covered by the irregular domain
 */
double fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives( double phi0, double phi1, double phi0xx, double phi1xx, double dx);


/*!
 * \brief Get interface location on the interval [a,b] given one point on either side of the interface
 * \param a: location of first point
 * \param b: location of second point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \return interface location
 */
double interface_Location( double   a, double   b,
                           double  fa, double  fb );

/*!
 * \brief Get interface location on the interval [a,b] given two points on either side of interface
 * \param a: location of first point
 * \param b: location of second point
 * \param a: location of third point
 * \param b: location of fourth point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \param fa: value of the level set at third point
 * \param fb: value of the level set at the fourth point
 * \return interface location
 */
double interface_Location_Between_b_And_c( double   a, double   b, double  c, double  d,
                                           double  fa, double  fb, double fc, double fd );

/*!
 * \brief Get interface location on the interval [a,b] given two points on either side of interface
 * \param a: location of first point
 * \param b: location of second point
 * \param a: location of third point
 * \param b: location of fourth point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \param fa: value of the level set at third point
 * \param fb: value of the level set at the fourth point
 * \return interface location
 */
double interface_Location_Between_b_and_c_Minmod( double   a, double   b, double  c, double  d,
                                                  double  fa, double  fb, double fc, double fd );

/*!
 * \brief Finds the interface location between point b and c given two points on either side
 * \note This routine was implemented in Smereka's JCP delta function paper. It is not recommended to use. interface_Location_Between_b_And_c is more accurate and reliable. This is to try his test problems
 * \param a: location of point a
 * \param b: location of point b
 * \param c: location of point c
 * \param d: location of point d
 * \param fa: value of phi at point a
 * \param fb: value of phi at point a
 * \param fc: value of phi at point a
 * \param fd:value of phi at point a
 */
double find_Root_Between_b_And_c_Smereka(double  b, double  c, double fa, double fb, double fc, double fd);


/*!
 * \brief Get interface location on the interval [a,b] given one point on either side and the first derivative at a and b
 * \param a: location of first point
 * \param b: location of second point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \param fxa: value of the first derivative of the level set at first point
 * \param fxb: value of the first derivative of the level set at the second point
 * \return interface location
 */
double interface_Location_With_First_Order_Derivative(	double   a, double   b,
                                                        double  fa, double  fb,
                                                        double fxa, double fxb );

/*!
 * \brief Get interface location on the interval [a,b] given one point on either side and the second derivative at a and b
 * \param a: location of first point
 * \param b: location of second point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \param fxxa: value of the second derivative of the level set at first point
 * \param fxxb: value of the second derivative of the level set at the second point
 * \return interface location
 */
double interface_Location_With_Second_Order_Derivative(double    a, double    b,
                                                       double   fa, double   fb,
                                                       double fxxa, double fxxb );


/*!
 * \brief Get interface location on the interval [a,b] given one point on either side and the second derivative at a and b
 * \param a: location of first point
 * \param b: location of second point
 * \param fa: value of the level set at first point
 * \param fb: value of the level set at the second point
 * \param fxxa: value of the second derivative of the level set at first point
 * \param fxxb: value of the second derivative of the level set at the second point
 * \return interface location
 */
double interface_Location_With_Second_Order_Derivative(double    a, double    b,
                                                       double   fa, double   fb,
                                                       double fxxa, double fxxb );


/*!
 * \brief find zero of 1st degree polynomial
 * \param c: coefficient of the polynomial
 * \param s: zero of the polynomial
 * \return number of zeros
 */
int solve_Linear( double c[2], double s[1] );
/*!
 * \brief find zero of 2nd degree polynomial
 * \param c: coefficients of the polynomial
 * \param s: zeros of the polynomial
 * \return number of zeros
 */
int solve_Quadric( double c[3], double s[ 2 ]);
/*!
 * \brief find zero of 3rd degree polynomial
 * \param c: coefficients of the polynomial
 * \param s: zeros of the polynomial
 * \return number of zeros
 */
int solve_Cubic(double c[ 4 ], double s[ 3 ]);
/*!
 * \brief find zero of 4th degree polynomial
 * \param c: coefficients of the polynomial
 * \param s: zeros of the polynomial
 * \return number of zeros
 */
int solve_Quartic(double c[ 5 ], double s[ 4 ]);

/*!
* \brief fonction for the 4 above functions
*/
int is_Zero(double x);


#endif // MY_P4EST_MATH_H
