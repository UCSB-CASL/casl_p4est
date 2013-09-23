#ifndef __CASL_MATH_H__
#define __CASL_MATH_H__

#include <math.h>
#include <stdexcept>
#include <limits>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
/*!
 * \file CASL_math.h
 * \brief This file contains the basic mathematical functions used throughout the library
 */

template <typename T>
inline T mod(const T& a, const T& b)
{
#ifdef CASL_THROWS
  if(b==0) throw std::invalid_argument("[CASL_ERROR]: mod: cannot take the modulus with zero.");
#endif
  T c=a%b;
  return (c<0)? c+b:c;
}

double mod(double a, double b);

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

namespace constants{
const static double epsilon = std::numeric_limits<double>::epsilon();
const static double pi      = M_PI;
}


#endif // __CASL_MATH_H__
