#ifndef MY_P4EST_CUBE3_H
#define MY_P4EST_CUBE3_H

#include <src/types.h>
#include <src/point3.h>
#include <src/casl_geometry.h>
#include <unordered_map>
#include <vector>

/*!
 * \file Cube3.h
 * \brief A three dimensional cube and the basic functions associated
 */

/*!
 * \class Cube3
 *
 * A three dimensional cube and the basic functions associated. The algorithms are take from the JCP 190, 2003 paper
 * and the implementation was done by Chohong Min, 2006
 */

struct Cube3
{
  int num_tet;
  bool middlecut;
private:
  inline void swap(double &phi1, double &phi2, double &f1, double &f2, Point3 &p1, Point3 &p2) const
  {
    double tmp_phi = phi1; phi1 = phi2; phi2 = tmp_phi;
    double tmp_f   = f1  ; f1   = f2  ; f2   = tmp_f;
    Point3 tmp_p   = p1  ; p1   = p2  ; p2   = tmp_p;
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
  inline Point3 interpol_p(Point3 p1, double phi1, Point3 p2, double phi2) const
  {
    // NOTE: replace with an epsilon test
#ifdef CASL_THROWS
    if(phi2 == phi1) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
    return (p1*phi2 - p2*phi1)/(phi2-phi1);
  }

  /**
   * Update a distance map with the minimum distance value for a given grid node.
   * @param [in, out] distanceMap Distance hash map to update.
   * @param [in] n Grid node index in partition.
   * @param [in] d Distance.
   */
  static void _updateMinimumDistanceMap( std::unordered_map<p4est_locidx_t, double>& distanceMap, p4est_locidx_t n, double d );

  /**
   * Compute the minimum distance of the cube points to a 3D triangle.
   * Update a distance map by keeping only the minimum distance.
   * @param [in] allPoints Array of of cube points' coordinates.
   * @param [in] phiAndIdxOctValues Level-set function values and p4est partition indices of cube corners.
   * @param [in] v0 Pointer to first triangle vertex.
   * @param [in] v1 Pointer to second triangle vertex.
   * @param [in] v2 Pointer to third triangle vertex.
   * @param [out] distanceMap Hash map to hold the current minimum distance of points to \Gamma.
   * @param [in] TOL Tolerance for zero-distance checking.
   */
  static void _computeDistanceToTriangle( const Point3 allPoints[], const OctValueExtended& phiAndIdxOctValues,
  										  const Point3 *v0, const Point3 *v1, const Point3 *v2,
										  std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL = EPS );

  /**
   * Compute the minimum distance of the cube points to a 3D line segment.
   * Update a distance map by keeping only the minimum distance.
   * @param [in] allPoints Array of cube points' coordinates.
   * @param [in] phiAndIdxOctValues Level-set function values and p4est partition indices of cube corners.
   * @param [in] v0 Pointer to first line segment vertex.
   * @param [in] v1 Pointer to second line segment vertex.
   * @param [out] distanceMap Hash map to hold the current miminum distance of points to \Gamma.
   * @param [in] TOL Tolerance for zero-distance checking.
   */
  static void _computeDistanceToLineSegment( const Point3 allPoints[], const OctValueExtended& phiAndIdxOctValues,
											 const Point3 *v0, const Point3 *v1,
											 std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL = EPS );

public:
  double xyz_mmm[3], xyz_ppp[3]; // nodes

  /*!
       * \brief default constructor for Cube2, initialize the coordinates to zero
       */
  Cube3();

  /*!
       * \brief constructor for Cube2
       * \param x0
       * \param x1
       * \param y0
       * \param y1
       */
  Cube3(double x0, double x1, double y0, double y1, double z0, double z1);

    /*!
     * \brief interface_Area_In_Cell compute the are of the interface in the cube3
     * \param level_set_values ...
     * \return the are of the interface in the cell
     */
    double interface_Area_In_Cell(const OctValue& level_set_values) const;

    /*!
     * \brief compute the volume of the Cube3 in the negative domain
     * \param level_set_values the values of the level-set function at the corners of the Cube3
     * \return the volume of the Cube3 in the negative domain
     */
  double volume_In_Negative_Domain(const OctValue& level_set_values) const;

    /*!
     * \brief integrate a quantity over the negative domain in the Cube3
     * \param f the values of the quantity at the corners of the Cube3
     * \param p the values of the level-set at the corners of the Cube3
     * \return the integral of the quantity f over the negative domain (p<0) in the Cube3
     */
    double integral(const OctValue &f, const OctValue &level_set_values) const;


    double integrate_Over_Interface(const OctValue &f, const OctValue &level_set_values ) const;

    double integrate_Over_Interface(const CF_3 &f, const OctValue &level_set_values ) const;

    void set_middlecut(bool val)
    {
      if (val)  {middlecut = true;  num_tet = 5;}
      else      {middlecut = false; num_tet = 6;}
    }
    double max_Over_Interface(const OctValue &f, const OctValue &ls_values) const;

	/**
	 * Approximate the distance of the nodes in an octant to the interface.  Computations are based on oct's simplices that
	 * are cut-out by the interface.  When this is true, a map of nodal indices to minimum distance is filled and
	 * provided back to the caller function.  This map will be empty if no oct's simplex is cut by the interface.
	 * The cube is decomposed into five tetrahedra using the middle-cut algorithm.
	 * @param [in] phiAndIdxOctValues Container of level-set function values and indices associated to the oct nodes.
	 * @param [out] distanceMap Minimum approximated distance from nodes belonging to at least one of the octs's simplices cut-out by \Gamma.
	 * @param [in] TOL Distance tolerance for zero-checking.
	 */
	void computeDistanceToInterface( const OctValueExtended& phiAndIdxOctValues,
									 std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL = EPS ) const;
};
#endif // MY_P4EST_CUBE3_H
