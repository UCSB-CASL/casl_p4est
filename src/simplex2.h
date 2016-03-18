#ifndef MY_P4EST_SIMPLEX2_H
#define MY_P4EST_SIMPLEX2_H

#include <src/point2.h>
#include <src/math.h>
#include <src/my_p4est_utils.h>
#include <math.h>

/*!
 * \file Simplex2.h
 * \brief A two dimentional simplex (i.e. triangle) and the basic functions associated
 */

/*!
 * \class Simplex2
 *
 * The Simplex2 class is a two dimentional simplex (i.e. triangle) with the basic functions associated
 */

class Simplex2
{
private:
  inline void swap(double &v1, double &v2)
  {
    double tmp = v1; v1 = v2; v2 = tmp;
  }

  inline double interpol_f(double f1, double phi1, double f2, double phi2) const
  {
#ifdef CASL_THROWS
      if(fabs(phi2-phi1)<EPS) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
      return (f1*phi2 - f2*phi1)/(phi2-phi1);
  }

  inline Point2 interpol_p(double x1, double y1, double phi1, double x2, double y2, double phi2) const
  {
#ifdef CASL_THROWS
      if(fabs(phi2-phi1)<EPS) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
      Point2 p;
      p.x = (x1*phi2 - x2*phi1)/(phi2-phi1);
      p.y = (y1*phi2 - y2*phi1)/(phi2-phi1);
      return p;
  }

public:

  double x0,x1,x2; // nodes
  double y0,y1,y2;

  /*!
     * \brief compute the area enclosed by three points
     * \param x0 the first coordinate of the first point
     * \param y0 the second coordinate of the first point
     * \param x1 the first coordinate of the second point
     * \param y1 the second coordinate of the second point
     * \param x2 the first coordinate of the third point
     * \param y2 the second coordinate of the third point
     * \return the area enclosed by the three points
     */
  static double area( double x0, double y0,
                      double x1, double y1,
                      double x2, double y2);

  /*!
     * \brief compute the area enclosed by three points
     * \param P0 the first point
     * \param P1 the second point
     * \param P2 the third point
     * \return the area enclosed by the three points
     */
  static double area( const Point2& P0,
                      const Point2& P1,
                      const Point2& P2);


  /*!
     * \brief compute the area enclosed by the simplex
     * \return the area of the simplex
     */
  double area() const;


  //---------------------------------------------------------------------
  // integral on the domain of P<0
  //---------------------------------------------------------------------
  /*!
     * \brief integrate a quantity on the negative domain of the simplex
     * \param f0 the value of the quantity at the first point
     * \param f1 the value of the quantity at the second point
     * \param f2 the value of the quantity at the third point
     * \param p0 the value of the level-set at the first point
     * \param p1 the value of the level-set at the second point
     * \param p2 the value of the level-set at the third point
     * \return the integral of f over the negative domain (p<0) of the simplex
     */
  double integral( double f0, double f1, double f2,
                   double p0, double p1, double p2);


  /*!
   * \brief integrate a quantity over the interface in the simplex
   * \param f0 the value of the quantity at the first point
   * \param f1 the value of the quantity at the second point
   * \param f2 the value of the quantity at the third point
   * \param p0 the value of the level-set at the first point
   * \param p1 the value of the level-set at the second point
   * \param p2 the value of the level-set at the third point
   * \return the integral of f over the interface in the simplex
   */
  double integrate_Over_Interface( double f0, double f1, double f2,
                                   double p0, double p1, double p2);

  /*!
   * \brief integrate a quantity over the interface in the simplex
   * \param f the continuous description of the quantity to integrate
   * \param p0 the value of the level-set at the first point
   * \param p1 the value of the level-set at the second point
   * \param p2 the value of the level-set at the third point
   * \return the integral of f over the interface in the simplex
   */
  double integrate_Over_Interface( const CF_2& f, double p0, double p1, double p2);
};

#endif // MY_P4EST_SIMPLEX2_H
