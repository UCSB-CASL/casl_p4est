#ifndef MY_P4EST_CUBE2_H
#define MY_P4EST_CUBE2_H

#include "point2.h"
#include "simplex2.h"
#include "types.h"
#include <src/casl_geometry.h>
#include <unordered_map>

/*!
 * \file Cube2.h
 * \brief A two dimensional cube (i.e. rectangle) and the basic functions associated
 */


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

	/**
	 * Compute the minimum distance of the quad points to a 2D line segment.
	 * Update a distance map by keeping only the minimum distance.
	 * @param [in] allPoints Array of quad points' coordinates.
	 * @param [in] phiAndIdxQuadValues Level-set function values and p4est partition indices of quad corners.
	 * @param [in] v0 Pointer to first line segment vertex.
	 * @param [in] v1 Pointer to second line segment vertex.
	 * @param [out] distanceMap Hash map to hold the current miminum distance of points to \Gamma.
	 * @param [in] TOL Tolerance for zero-distance checking.
	 */
	static void _computeDistanceToLineSegment( const Point2 allPoints[], const QuadValueExtended& phiAndIdxQuadValues,
											   const Point2 *v0, const Point2 *v1,
											   std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL = EPS );

	/**
     * Update a distance map with the minimum distance value for a given grid node.
     * @param [in, out] distanceMap Distance hash map to update.
     * @param [in] n Grid node index in partition.
     * @param [in] d Distance.
     */
	static void _updateMinimumDistanceMap( std::unordered_map<p4est_locidx_t, double>& distanceMap, p4est_locidx_t n, double d );

public:
  double xyz_mmm[2], xyz_ppp[2]; // nodes

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
   * \brief interface_Length_In_Cell compute the length of the interface in the cell
   * \param level_set_values ...
   * \return length of the interface in the cell
   */
  double interface_Length_In_Cell(const QuadValue& level_set_values) const;

  /*!
     * \brief compute the area of the Cube2 in the negative domain
     * \param level_set_values the values of the level-set function at the corners of the Cube2
     * \return the area of the Cube2 in the negative domain
     */
  double area_In_Negative_Domain(const QuadValue& level_set_values) const;

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

  /*!
     * \brief integrate a quantity over the 0-level-set in a Cube2
     * \param f the values of the function to integrate at the corners of the Cube2
     * \param level_set_values the level-set values at the corners of the Cube2
     * \return the integral of f over the 0-level-set in the Cube2
     */
  double integrate_Over_Interface(const CF_2 &f, const QuadValue &level_set_values ) const;

  /*!
     * \brief calculate the maximum of a quantity over the 0-level-set in a Cube2
     * \param f the values of the function to at the corners of the Cube2
     * \param level_set_values the level-set values at the corners of the Cube2
     * \return the maximum of f over the 0-level-set in the Cube2
     */
  double max_Over_Interface( const QuadValue& f, const QuadValue& level_set_values ) const;

  	/**
  	 * Approximate the distance of the nodes in a quad to the interface.  Computations are based on quad's simplices that
  	 * are cut-out by the interface.  When this is true, a map of nodal indices to minimum distance is filled and
  	 * provided back to the caller function.  This map will be empty if no quad's simplex is cut by the interface.
  	 * @param [in] phiAndIdxQuadValues Container of level-set function values and indices associated to the quad nodes.
  	 * @param [out] distanceMap Minimum approximated distance from nodes belonging to at least one of the quad's simplices cut-out by \Gamma.
  	 * @param [in] TOL Distance tolerance for zero-checking.
  	 */
	void computeDistanceToInterface( const QuadValueExtended& phiAndIdxQuadValues,
			std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL = EPS ) const;
};

#endif // MY_P4EST_CUBE2_H
