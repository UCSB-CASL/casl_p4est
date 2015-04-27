#ifndef __SIMPLEX2_H__
#define __SIMPLEX2_H__

#include "point2.h"

/*!
 * \file Simplex2.h
 * \brief A two dimentional simplex (i.e. triangle) and the basic functions associated
 */

/*!
 * \class Simplex2
 *
 * The Simplex2 class is a two dimentional simplex (i.e. triangle) with the basic functions associated
 */

struct Simplex2
{
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
};

#endif // __SIMPLEX2_H__
