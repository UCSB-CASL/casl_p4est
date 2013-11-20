#ifndef __POINT2_H__
#define __POINT2_H__

#include <iostream>
#include <stdexcept>
#include <src/CASL_math.h>

/*!
 * \file Point2.h
 * \brief Point or vector in two dimensions
 */

/*!
 * \class Point2
 *
 * The Point2 is a class for points and vectors in two dimensions
 */
class Point2
{
public:
  double x, y;

  /*!
     * \brief default constructor for Point2, the coordinates are set to 0
     */
  Point2();

  /*!
     * \brief constructor for Point2 with given coordinates
     * \param c1 the first coordinate
     * \param c2 the second coordinate
     */
  Point2(double c1, double c2);

  /*!
     * \brief copy constructor for Point2
     * \param pt the Point2 to copy
     */
  Point2(const Point2& pt);

  /*!
     * \brief overloading the = operator
     */
  void operator=(Point2 pt);

  /*!
     * \brief overloading the == operator
     */
  bool operator==(const Point2& pt)const;

  /*!
     * \brief overloading the - operator
     * \return a Point2 with opposite coordinate
     */
  Point2 operator-() const;

  /*!
     * \brief overloading the + operator
     * \param r the Point2 to add
     * \return the sum of the Point2 with the given Point2
     */
  Point2 operator+(const Point2& r) const;

  /*!
     * \brief overloading the - operator
     * \param r the Point2 to substract
     * \return the substraction of the given Point2 from the Point2
     */
  Point2 operator-(const Point2& r) const;

  /*!
     * \brief overloading the * operator
     * \param s the value to multiply by
     * \return the Point2 multiplicated with the given value
     */
  Point2 operator*(double s) const;

  /*!
     * \brief overloading the / operator
     * \param s the value to multiply by
     * \return the Point2 divided with the given value
     */
  Point2 operator/(double s) const;

  /*!
     * \brief overloading the += operator
     * \param r the Point2 to add
     */
  void operator+=(const Point2& r);

  /*!
     * \brief overloading the -= operator
     * \param r the Point2 to substract
     */
  void operator-=(const Point2& r);

  /*!
     * \brief overloading the *= operator
     * \param r the value to multiply by
     */
  void operator*=(double r);

  /*!
     * \brief overloading the /= operator
     * \param r the value to divide by
     */
  void operator/=(double r);

  /*!
     * \brief compute the dot product (or canonical scalar product) with another Point2
     * \param pt the Point2 for the dot product
     * \return the dot product (or canonical scalar product) of the Point2 with the given Point2
     */
  double dot(const Point2& pt) const;

  /*!
     * \brief compute the L2 norm of the Point2
     * \return the L2 norm of the Point2
     */
  double norm_L2() const;

  /*!
     * \brief compute the cross product with another Point2
     * \param pt the Point2 with which the cross product is to be computed
     * \return the cross product between the Point2 and the given Point2, i.e. (*this) x pt
     */
  double cross(const Point2& pt) const;

  /*!
     * \brief compute the square of the L2 norm of the Point2
     * \return the square of the L2 norm of the Point2
     */
  double sqr() const;

  /*!
     * \brief normalize a Point2 (i.e. transform to unitary norm)
     * \return the normalized Point2
     */
  Point2 normalize() const;

  /*!
     * \brief compute the L2 norm of a vector
     * \param P1 the beginning of the vector
     * \param P2 the end of the vector
     * \return the L2 norm of the vector P1-P2
     */
  static double norm_L2(const Point2& P1, const Point2& P2);

  /*!
     * \brief compute the curl between two Point2
     * \param P1 the first Point2
     * \param P2 the second Point2
     * \return the curl between the two Point2, i.e. P1xP2
     */
  static double curl(const Point2& P1, const Point2& P2);

  /*!
     * \brief compute the curl between two Point2 (vectors in two dimensions)
     * \param P1 the first Point2
     * \param P2 the second Point2
     * \return the curl between the two Point2
     */
  static double cross(const Point2& P1, const Point2& P2);

  /*!
     * \brief compute the area in a triangle
     * \param P1 the first vertex of the triangle
     * \param P2 the second vertex of the triangle
     * \param P3 the third vertex of the triangle
     * \return the area enclosed by the three Point2
     */
  static double area(const Point2& P1, const Point2& P2, const Point2& P3);

  /*!
   * \brief distance compute the Euclidean distance to another point
   * \param [in] p the point to compute the distance to
   * \return the Euclidean distance
   */
  inline double distance(const Point2& p) const { return sqrt(SQR(x-p.x) + SQR(y-p.y)); }

  /*!
     * \brief overload the << operator for ArrayV
     */
  friend std::ostream& operator<<(std::ostream& os, const Point2& p);
};

#endif // __POINT2_H__
