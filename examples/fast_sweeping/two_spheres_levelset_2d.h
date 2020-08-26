#ifndef FAST_SWEEPING_TWO_SPHERES_LEVELSET_2D_H
#define FAST_SWEEPING_TWO_SPHERES_LEVELSET_2D_H

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <vector>
#include <utility>
#include <src/casl_geometry.h>

/**
 * Define a two-sphere level-set function that is NOT a signed-distance function.
 * Incorporate the functionality to retrieve the exact signed distance to the interface as well.
 * The level-set function is specified by a reference circle C1, centered at c1 = (x1, y1), and a second circle C2,
 * whose center, c2, is located d units from c1 in an angle, alpha in [-pi/2, +pi/2], with respect to the horizontal
 * line that emanates from c1 in the same direction of the positive x-axis.
 *
 * The information about C1 and C2 are used to define a local coordinate system whose origin coincides with c1 and whose
 * positive x-axis goes through c1 and c2.  This way, c1' = (0,0) and c2' = (d, 0).  The affine transformation that
 * aligns the world coordinate system to the local coordinate system is given by:
 *
 *      | x_world |   | 1  0  x1 |   | cos(alpha)  -sin(alpha)  0 |   | x_local |
 *      | y_world | = | 0  1  y1 | * | sin(alpha)   cos(alpha)  0 | * | y_local |
 *      |    1    |   | 0  0  1  |   |      0            0      1 |   |   1     |
 * or
 *        q = T(c1) * R(alpha) * p,        p, q in R^2
 * in homogeneous coordinates.
 *
 * For the level-set function to be valid, the inequality |r1 - r2| < d must always hold.
 *
 * Created on August 4, 2020, by Luis Ángel.
 */
class TwoSpheres : public CF_2
{
protected:
	Point2 _c1;				// Center coordinates of reference circle in world coordinates.
	Point2 _c2;				// Center coordinates of second circle in world coordinates.
	double _r1;				// Circle radii.
	double _r2;
	double _d;				// Distance between circles.
	double _alpha;			// Tilt angle of reference axis with respect to world's +x direction.

public:
	/**
	 * Constructor.
	 * @param [in] x1 Reference circle's center x-coordinate.
	 * @param [in] y1 Reference circle's center y-coordinate.
	 * @param [in] r1 Reference circle's radius.
	 * @param [in] r2 Second circle's radius.
	 * @param [in] d Distance between circles' centers.
	 * @param [in] alpha Tilt angle of reference line connecting c1 and c2 with respect to +x world axis.
	 * @throws Runtime exception if the correctness conditions on the radii, distance, and tilt angle are not met.
	 */
	explicit TwoSpheres( double x1, double y1, double r1, double r2, double d, double alpha )
	: _c1( x1, y1 )
	{
		// Check for correctness in radii, circles distance, and tilt angle.
		if( r1 <= EPS || r2 <= EPS )
			throw std::runtime_error( "[CASL_ERROR] TwoSphereNSD::TwoSphereNSD: Spheres' radii must be positive!" );
		_r1 = r1;
		_r2 = r2;

		if( d <= ABS( _r1 - _r2 ) )
			throw std::runtime_error( "[CASL_ERROR] TwoSphereNSD::TwoSphereNSD: A sphere cannot be contained in the other one!" );
		_d = d;

		if( alpha < -M_PI_2 || alpha > M_PI_2 )
			throw std::runtime_error( "[CASL_ERROR] TwoSphereNSD::TwoSphereNSD: Tilt angle must be in the range of [-pi/2, +pi/2]!" );
		_alpha = alpha;

		// Store the center coordinates of the second circle in world coordinates to facilitate computation of the
		// non-signed distance level-set function.
		_c2.x = _c1.x + _d * cos( _alpha );
		_c2.y = _c1.y + _d * sin( _alpha );
	}

	/**
	 * Transform a point from world coordinates to local coordinates.
	 * @param [in,out] x Coordinate in x-dimension.
	 * @param [in,out] y Coordinate in y-dimension.
	 */
	void toLocalCoordinates( double& x, double& y ) const
	{
		double x1 = ( x - _c1.x ) * cos( _alpha ) + ( y - _c1.y ) * sin( _alpha );
		double y1 = ( y - _c1.y ) * cos( _alpha ) - ( x - _c1.x ) * sin( _alpha );
		x = x1;
		y = y1;
	}

	/**
	 * Transform a point from local coordinates to world coordinates.
	 * @param [in,out] x Coordinate in x-direction in local coord. system.
	 * @param [in,out] y Coordinate in y-direction in local coord. system.
	 */
	void toWorldCoordinates( double& x, double& y ) const
	{
		double x0 = x * cos( _alpha ) - y * sin( _alpha ) + _c1.x;
		double y0 = x * sin( _alpha ) + y * cos( _alpha ) + _c1.y;
		x = x0;
		y = y0;
	}

	/**
	 * Non-signed level-set function evaluation at a given point in world coordinates.
	 * This function follows the definition of the boolean union operation between two level-set functions described in
	 * [7]: phi(x) = min( phi_1(x), phi_2(x) ).
	 * @param [in] x Point x-coordinate.
	 * @param [in] y Point y-coordinate.
	 * @return phi(x,y).
	 */
	[[nodiscard]] double operator()( double x, double y ) const override
	{
		double phi1 = SQR( _c1.x - x ) + SQR( _c1.y - y ) - SQR( _r1 );
		double phi2 = SQR( _c2.x - x ) + SQR( _c2.y - y ) - SQR( _r2 );
		return MIN( phi1, phi2 );
	}

	/**
	 * Signed-distance level-set function evaluation at a given point in world coordinates.
	 * Retrieve the curvature at the normal projection onto the closest interface point as well.
	 * @param [in] x Point x-coordinate.
	 * @param [in] y Point y-coordinate.
	 * @param [in] H Grid's cell minimum width.
	 * @param [out] hk Dimensionless curvature of closest point on compound Gamma.
	 * @param [out] who Set to -1 if hk is due to C1, +1 if hk is due to C2, and 0 if hk is in a discontinuous region.
	 * @return The signed distance from a point to the compound interface.
	 */
	[[nodiscard]] double getSignedDistance( double x, double y, const double H, double& hk, short& who ) const
	{
		double phi1 = sqrt( SQR( _c1.x - x ) + SQR( _c1.y - y ) ) - _r1;	// Respective signed-distances to the two
		double phi2 = sqrt( SQR( _c2.x - x ) + SQR( _c2.y - y ) ) - _r2;	// circles.

		if( _d > _r1 + _r2 || ( phi1 >= 0 && phi2 >= 0 ) )	// Are circles NOT intersecting? Or is the point exterior to intersecting circles?
		{
			if( phi1 < phi2 )			// Select the closest circular interface.
			{
				hk = H / _r1;
				who = -1;
				return phi1;
			}
			if( phi2 < phi1 )
			{
				hk = H / _r2;
				who = +1;
				return phi2;
			}
			hk = -1;					// Indicates a discontinuity in the level-set function as the point is equally
			who = 0;					// distant from both C1's and C2's interfaces.
			return phi1;
		}

		// Point lies inside of at least one of the two circles.  To facilitate calculations, we map the point from
		// world coordinates into local coordinates, where the reference circle is at the origin, and the second circle
		// sits to the right on the local +x axis.
		toLocalCoordinates( x, y );

		// Find the intersection point(s) between C1 and C2.
		double xi = (SQR( _d ) - SQR( _r2 ) + SQR( _r1 )) / (2 * _d);	// Point(s) (xi, ±yi) found with respect to C1.
		double yi = sqrt( 4 * SQR( _d * _r1 ) - SQR( SQR( _d ) - SQR( _r2 ) + SQR( _r1 ) ) ) / (2 * _d);

		double dp = sqrt( SQR( x - xi ) + SQR( y - yi ) );		// Corresponding distances to intersection points p
		double dq = sqrt( SQR( x - xi ) + SQR( y + yi ) );		// (above) and q (below).
		double minDPQ = MIN( dp, dq );							// The minimum distance to intersection points.

		// Check for domain error should we execute atan2: (0, 0).
		if( ABS( y ) < EPS )
		{
			if( ABS( x ) < EPS )			// At the center (x1, y1)?
			{
				hk = H / _r1;
				who = -1;
				return phi1;
			}
			if( ABS( x - _d ) < EPS )		// At the center (x2, y2)?
			{
				hk = H / _r2;
				who = +1;
				return phi2;
			}
		}

		// Query point doesn't match any of the two circles' centers: find angles.
		double psi1 = atan2( yi, xi );							// Psi1 is the angular range [-psi1, +psi1] that defines
																// the asymmetric lens from the intersection for C1.
		double psi2 = atan2( yi, xi - _d );						// Psi2 is the angular range [psi2, 2pi - psi2] for C2.
		double theta1 = atan2( y, x );							// Thus, for C1 we need theta in [-pi, pi), and for C2
		double theta2 = atan2( y, x - _d );						// theta in [0, 2pi).
		theta2 = ( theta2 < 0 )? 2 * M_PI + theta2 : theta2;

		// Computing the exact distance from (x, y) to the compound interface.
		if( phi1 >= 0 && phi2 < 0 )								// Inside C2 but outside C1?
		{
			if( theta2 < psi2 || theta2 > 2 * M_PI - psi2 )		// theta2 not in psi2 range?
			{
				hk = H / _r2;
				who = +1;
				return phi2;
			}
			hk = -1;											// Closest point on compound interface is (xi, ±yi).
			who = 0;
			return -minDPQ;										// theta2 in range of asymmetric quad.  Note the - sign.
		}
		if( phi1 < 0 && phi2 >= 0 )								// Inside C1 but outside C2?
		{
			if( ABS( theta1 ) > psi1 )							// theta1 not in psi1 range?
			{
				hk = H / _r1;
				who = -1;
				return phi1;
			}
			hk = -1;											// Closest point on compound interface is (xi, ±yi).
			who = 0;
			return -minDPQ;										// theta1 in range of asymmetric quad.
		}
		double phi = -PETSC_MAX_REAL;
		if( ABS( theta1 ) > psi1 )								// theta1 not in psi1 range?
		{
			phi = phi1;
			hk = H / _r1;
			who = -1;
		}
		if( theta2 < psi2 || theta2 > 2 * M_PI - psi2 )			// theta2 not in psi2 range?
		{
			if( phi2 > phi )
			{
				phi = phi2;
				hk = H / _r2;
				who = +1;
			}
			else
			{
				throw std::runtime_error( "[CASL_ERROR] TwoSphereNSD::getSignedDistance: A point cannot be outside of "
										  "the interior quad and be in the valid psi range for both C1 and C2!" );
			}
		}
		if( phi <= -PETSC_MAX_REAL )							// Point lies inside intersection quad?
		{
			phi = -minDPQ;
			hk = -1;											// Discontinuity in the level-set function.
			who = 0;
		}
		return phi;
	}
};

#endif //FAST_SWEEPING_TWO_SPHERES_LEVELSET_2D_H
