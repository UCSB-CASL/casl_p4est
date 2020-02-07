#ifndef MY_P4EST_POINT3_H
#define MY_P4EST_POINT3_H

#include <src/point2.h>

/*!
 * \file Point3.h
 * \brief Point or vector in three dimensions
 */

//-----------------------------------------------
// 
//  Point3 class
//
//-----------------------------------------------

/*!
 * \class Point3
 *
 * The Point3 is a class for points and vectors in three dimensions
 */
class Point3
{
public:
  double x, y, z;

  /*!
     * \brief default constructor for Point3, the coordinates are set to 0
     */
  Point3() ;

  /*!
     * \brief constructor for Point3 with given coordinates
     * \param c1 the first coordinate
     * \param c2 the second coordinate
     * \param c3 the third coordinate
     */
  Point3(double c1, double c2, double c3);

  /*!
     * \brief copy constructor for Point2
     * \param pt the Point2 to copy (z coordinate is set to 0)
     */
  Point3 (const Point2& pt);

  /*!
     * \brief overloading the = operator
     */
  void operator=(const Point3& P);

  /*!
     * \brief overloading the - operator
     * \return a Point3 with opposite coordinate
     */
  Point3 operator-() const;

  /*!
     * \brief overloading the + operator
     * \param r the Point3 to add
     * \return the sum of the Point2 with the given Point2
     */
  Point3 operator+(const Point3& r) const;

  /*!
     * \brief overloading the - operator
     * \param r the Point3 to substract
     * \return the substraction of the given Point3 from the Point3
     */
  Point3 operator-(const Point3& r) const;

  /*!
     * \brief overloading the * operator
     * \param s the value to multiply by
     * \return the Point3 multiplicated with the given value
     */
  Point3 operator*(double s) const;

  /*!
     * \brief overloading the / operator
     * \param s the value to multiply by
     * \return the Point3 divided with the given value
     */
  Point3 operator/(double s) const;

  /*!
     * \brief overloading the += operator
     * \param r the Point3 to add
     */
  void operator+=( const Point3& r );

  /*!
     * \brief overloading the -= operator
     * \param r the Point3 to substract
     */
  void operator-=( const Point3& r );

  /*!
     * \brief overloading the *= operator
     * \param r the value to multiply by
     */
  void operator*=( double r );

  /*!
     * \brief overloading the /= operator
     * \param r the value to divide by
     */
  void operator/=( double r );

  /*!
     * \brief compute the dot product (or canonical scalar product) with another Point3
     * \param pt the Point3 for the dot product
     * \return the dot product (or canonical scalar product) of the Point3 with the given Point3
     */
  double dot(const Point3& pt) const;

  /*!
     * \brief compute the L2 norm of the Point3
     * \return the L2 norm of the Point3
     */
  double norm_L2() const;

  /*!
     * \brief compute the cross product with another Point3
     * \param pt the Point3 with which the cross product is to be computed
     * \return the cross product between the Point3 and the given Point3, i.e. (*this) x r
     */
  Point3 cross(const Point3& r) const;

  /*!
     * \brief normalize a Point3 (i.e. transform to unitary norm)
     * \return the normalized Point3
     */
  Point3 normalize() const;

  inline double& xyz(const unsigned char &dir)
  {
#ifdef CASL_THROWS
    if(dir != 0 && dir != 1 && dir != 2)
      throw std::invalid_argument("[CASL_ERROR]: Point3::xyz(const unsigned char & dir) : dir must be either 0 or 1 or 2.");
#endif
    return (dir == 0 ? this->x : (dir == 1 ? this->y : this->z));
  }

  /*!
     * \brief compute the curl with another Point3
     * \param pt the Point3 with which the curl is to be computed
     * \return the curl between the Point3 and the given Point3, i.e. (*this) x r
     */
  Point3 curl(const Point3& r) const;

  /*!
     * \brief compute the curl between two Point3
     * \param l the first Point3
     * \param r the second Point3
     * \return the curl between the two Point3, i.e. lxr
     */
  static Point3 curl( const Point3& l,
                      const Point3& r );

  /*!
     * \brief compute the curl between two Point3
     * \param l the first Point3
     * \param r the second Point3
     * \param out the variable that is to contain the curl between the two Point3, i.e. out = lxr
     */
  static void   curl( const Point3& l,
                      const Point3& r,
                      Point3& out);

  /*!
     * \brief compute the area in a triangle
     * \param P0 the first vertex of the triangle
     * \param P1 the second vertex of the triangle
     * \param P2 the third vertex of the triangle
     * \return the area enclosed by the three Point3
     */
  static double area( const Point3& P0,
                      const Point3& P1,
                      const Point3& P2 );

  /*!
     * \brief compute the volume encolsed by three Point3
     * \param P0 the first vertex of the tetrahedron
     * \param P1 the second vertex of the tetrahedron
     * \param P2 the third vertex of the tetrahedron
     * \param P3 the fourth vertex of the tetrahedron
     * \return the volume enclosed by the four Point3
     */
  static double volume( const Point3& P0,
                        const Point3& P1,
                        const Point3& P2,
                        const Point3& P3 );

  /*!
     * \brief overload the << operator for ArrayV
     */
  friend std::ostream& operator<<(std::ostream& os, const Point3& p);

  /*!
     * \brief print the Point3 in the console
     */
  void print() const;
};
#endif // MY_P4EST_POINT3_H
