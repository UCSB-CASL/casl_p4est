#ifndef __CUBE2_H__
#define __CUBE2_H__

#include "point2.h"
#include "simplex2.h"

/*!
 * \file Cube2.h
 * \brief A two dimensional cube (i.e. rectangle) and the basic functions associated
 */

struct QuadValue {
  double val00;
  double val01;
  double val10;
  double val11;

  QuadValue()
  {
    val00 = val10 = val01 = val11 = 0;
  }

  QuadValue(double v00, double v01, double v10, double v11)
  {
    val00 = v00;
    val10 = v10;
    val01 = v01;
    val11 = v11;
  }

  /*!
     * \brief overload the << operator for QuadValue
     */
  friend std::ostream& operator<<(std::ostream& os, const QuadValue& q)
  {
    os << "val01 : " << q.val01 << ",\t" << "val11 : " << q.val11 << std::endl;
    os << "val00 : " << q.val00 << ",\t" << "val10 : " << q.val10 << std::endl;
    return os;
  }
};


/*!
 * \class Cube2
 *
 * A two dimensional cube (i.e. rectangle) and the basic functions associated. The algorithms are take from the JCP 190, 2003 paper
 * and the implementation was done by Chohong Min, 2006
 */

struct Cube2
{
private:
  inline void swap(double &phi1, double &phi2, double &f1, double &f2, Point2 &p1, Point2 &p2) const
  {
    double tmp_phi = phi1; phi1 = phi2; phi2 = tmp_phi;
    double tmp_f   = f1  ; f1   = f2  ; f2   = tmp_f;
    Point2 tmp_p   = p1  ; p1   = p2  ; p2   = tmp_p;
  }

  /*!
     * \brief interpolate a function f between two points
     * \param f1 the value of the function at the first point
     * \param phi1 the value of the level-set function at the first point
     * \param f2 the value of the function at the second point
     * \param phi2 the value of the level-set function at the second point
     * \return the interpolation of the function f weighed by the level-set values
     */
  inline double interpol_f(double f1, double phi1, double f2, double phi2) const
  {
    // NOTE: replace with an epsilon test
#ifdef CASL_THROWS
    if(phi2 == phi1) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
    return (f1*phi2 - f2*phi1)/(phi2-phi1);
  }

  /*!
     * \brief find the barycentre of two points
     * \param p1 the first point
     * \param phi1 the value of the level-set function at the first point
     * \param p2 the second point
     * \param phi2 the value of the level-set function at the second point
     * \return the barycentre of p1 and p2
     */
  inline Point2 interpol_p(Point2 p1, double phi1, Point2 p2, double phi2) const
  {
    // NOTE: replace with an epsilon test
#ifdef CASL_THROWS
    if(phi2 == phi1) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
    return (p1*phi2 - p2*phi1)/(phi2-phi1);
  }

public:
  double x0,x1; // nodes
  double y0,y1;

  /*!
     * \brief default constructor for Cube2, initialize the coordinates to zero
     */
  Cube2();

  /*!
     * \brief constructor for Cube2
     * \param x0
     * \param x1
     * \param y0
     * \param y1
     */
  Cube2(double x0, double x1, double y0, double y1);

  /*!
     * \brief compute the Kuhn triangulation of the rectangle
     * \param s1 the first triangle
     * \param s2 the second triangle
     */
  void kuhn_Triangulation( Simplex2& s1, Simplex2& s2 ) const;

  /*!
     * \brief compute the area of the Cube2 in the negative domain
     * \param level_set_values the values of the level-set function at the corners of the Cube2
     * \return the area of the Cube2 in the negative domain
     */
  double area_In_Negative_Domain( QuadValue& level_set_values) const;

  /*!
     * \brief integrate a quantity over the Cube2 using bilinear interpolation
     * \param f the values of f at the corners of the Cube2
     * \return the integral of the quantity over the Cube2
     */
  double integral( QuadValue f) const;

  /*!
     * \brief integrate a quantity over the negative domain in the Cube2
     * \param f the values of the quantity at the corners of the Cube2
     * \param p the values of the level-set at the corners of the Cube2
     * \return the integral of the quantity f over the negative domain (p<0) in the Cube2
     */
  double integral(const QuadValue &f, const QuadValue &level_set_values ) const;

  /*!
     * \brief integrate a quantity over the 0-level-set in a Cube2
     * \param f the values of the function to integrate at the corners of the Cube2
     * \param level_set_values the level-set values at the corners of the Cube2
     * \return the integral of f over the 0-level-set in the Cube2
     */
  double integrate_Over_Interface(const QuadValue &f, const QuadValue &level_set_values ) const;
};

#endif // __CUBE2_H__
