//
// Created by Im YoungMin on 4/30/20.
//

#ifndef FAST_SWEEPING_GEOMETRY_H
#define FAST_SWEEPING_GEOMETRY_H

#include <src/casl_math.h>
#include <src/point2.h>
#include <src/point3.h>

/**
 * A collection of geometric functions involving points, vectors, planes, polygons, etc.
 */
namespace geom
{
	/**
	 * Linearly interpolate a point based on level-set function values.
	 * @param [in] p1 First point.
	 * @param [in] phi1 Level-set function value at first point.
	 * @param [in] p2 Second point.
	 * @param [in] phi2 Level-set function value at second point.
	 * @param [in] TOL Distance zero-checking tolerance.
	 * @return Interpolated point between p1 and p2.
	 * @throws Zero division error if input level-set function values are (almost) equal.
	 */
	PointDIM interpolatePoint( const PointDIM *p1, double phi1, const PointDIM *p2, double phi2, double TOL = EPS )
	{
#ifdef CASL_THROWS
		if( ABS( phi2 - phi1 ) <= TOL )
			throw std::domain_error( "[CASL_ERROR]: geom::interpolatePoint - Division by zero." );
#endif
		return ( *p1 * phi2 - *p2 * phi1 ) / ( phi2 - phi1 );
	}

	/**
	 * Compute the closest point on a line segment to another query point, in any of 2 or 3D.
	 * Suppose the line segment, L, goes from the vertex v0 to v1.  Let v = v1 - v0, then a point Q on L can be obtained
	 * from the pametric equation:
	 *                          Q = v0 + tv,
	 * where t is the parameter.  Assuming that P is the normal projection of p onto L, then, the closest point on L to
	 * p is given by:
	 *                        |  v0,		if t <= 0
	 *                   P =  |  v0 + tv,	if 0 < t < 1
	 *                        |  v1,		if t >= 1
	 * @param [in] p Query point.
	 * @param [in] v0 First line segment's vertex.
	 * @param [in] v1 Second line segment's vertex.
	 * @return Closest point on the line segment v0v1.
	 */
	PointDIM findClosestPointOnLineSegmentToPoint( const PointDIM& p, const PointDIM& v0, const PointDIM& v1, double tol = EPS )
	{
		PointDIM v = v1 - v0;
		double denom = v.dot( v );
		if( sqrt( denom ) <= tol )						// Degenerate line segment?
			return v0;

		double t = ( p - v0 ).dot( v ) / denom;			// Parameter t in Q = v0 + tv.
		if( t <= 0 )
			return v0;
		if( t >= 1 )
			return v1;
		return v0 + v * t;
	}

	/**
	 * Project a 3D point on the plane subtended by a triangle given by its three input vertices.  If the projected
	 * point, P, falls within the triangle, the function returns true, and P's barycentric coordinates are given:
	 *                P = u * v0 + v * v1 + ( 1 - u - v ) * v2,   for 0 <= u, c <= 1
	 * If P falls outside of triangle, the function yields the end-points of the first line segment that is found to
	 * fail the inside/outside test.  This implies that there might still exist at most another side for which the same
	 * inside/outside test fails.
	 * Regardless of the point being or not within the triangle, the projection point P is always computed, unless the
	 * triangle is degenerate (i.e. colinear vertices).
	 * This function is an adaptation of the CG's ray-hitting-triangle algorithm provided in
	 * https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
	 * @param [in] p Query point.
	 * @param [in] v0 Triangle's first vertex.
	 * @param [in] v1 Triangle's second vertex.
	 * @param [in] v2 Triangle's third vertex.
	 * @param [out] u Produced barycentric coordinate of projected point within the triangle corresponding to v0.
	 * @param [out] v Produced barycentric coordinate of proejcted point within the triangle corresponding to v1.
	 * @param [out] P Normal projection of point p onto the triangle's plane.
	 * @param [out] x First end-point of first triangle's side to fail the inside/outside test.
	 * @param [out] y Second end-point of first triangle's side to fail the inside/outside test.
	 * @return True and valid barycentric coordinates if projected point falls within input triangle, false otherwise
	 * 		   and pointers to the end-points of first triangle's side to fail the inside/outside test.
	 * @throws Runtime exception if input triangle vertices are collinear.
	 */
	bool projectPointOnTriangleAndPlane( const Point3 *p, const Point3 *v0, const Point3 *v1, const Point3 *v2,
										 double& u, double& v, Point3& P, const Point3*& x, const Point3*& y )
	{
		// Compute triangle's subtended plane's normal vector.
		Point3 v0v1 = *v1 - *v0;
		Point3 v0v2 = *v2 - *v0;

		// No need to normalize.
		Point3 N = v0v1.cross( v0v2 );
		double denom = N.dot( N );

		if( sqrt( denom ) <= EPS )					// Check for colinear points.
			throw std::runtime_error( "[CASL_ERROR]: geom::projectPointOnTriangleAndPlane - Triangle's vertices are colinear!" );

		// Step 1: finding P, the projection of p onto the triangle's plane.
		double d = N.dot( *v0 );					// Compute the d parameter in plane's equation: ax + by + cz = d.
		double t = -( N.dot( *p ) + d ) / denom;	// Compute parameter t for ray's equation: r(t) = p + tN, where N is
													// both the plane's normal and ray's direction.
		P = *p + N * t;								// Intersection (projection) point.

		// Step 2: inside-outside test.
		// This test uses the cross product with a reference vector (i.e. a triangle's side).  When a point is outside
		// the triangle, we have at most two sides for which the inside/outside test fails:
		//         \                          \
		//          \   o Point                \
		//           \                    ------*.....
		//     -------*.....                     .  o Point
		//             .                          .
		//          (a)                       (b)
		// In case (a), there is just one side failing the inside/outside test.  In case (b), there are two.  In the
		// latter, just the first triangle's side to fail the test will be reported back.  However, in case (b) the
		// closest point on the triangle is the shared corner, here denoted as '*'.
		Point3 C; 									// Vector perpendicular to plane used for inside-outside test.
		x = y = nullptr;

		Point3 edge0 = *v1 - *v0;					// Edge 0.
		Point3 vp0 = P - *v0;
		C = edge0.cross( vp0 );
		if( N.dot( C ) < 0 )						// P is on the right side of edge 0, opposed to v2.
		{
			x = v0;
			y = v1;
			return false;
		}

		Point3 edge1 = *v2 - *v1;					// Edge 1.
		Point3 vp1 = P - *v1;
		C = edge1.cross( vp1 );
		if( ( u = N.dot( C ) ) < 0 )				// P is on the right side of edge 1, opposed to v0.
		{
			x = v1;
			y = v2;
			return false;
		}

		Point3 edge2 = *v0 - *v2;					// Edge 2.
		Point3 vp2 = P - *v2;
		C = edge2.cross( vp2 );
		if( ( v = N.dot( C ) ) < 0 )				// P is on the right side of edge 2, opposed to v1.
		{
			x = v2;
			y = v0;
			return false;
		}

		u /= denom;
		v /= denom;

		return true; 								// The projected point P falls within the triangle.
	}

	/**
	 * Some general purpose functions.
	 */
	namespace utils
	{
		/**
		 * Generic function to swap groups of 3 variables with different types.
		 * @tparam T1 Type for first and fourth values.
		 * @tparam T2 Type for second and fifth values.
		 * @tparam T3 Type for third and sixth values.
		 * @param [in, out] a First value to swap with fourth value.
		 * @param [in, out] b Second value to swap with fifth value.
		 * @param [in, out] c Third value to swap with sixth value.
		 * @param [in, out] u Fourth value to swap with first value.
		 * @param [in, out] v Fifth value to swap with second value.
		 * @param [in, out] w Sixth value to swap with third value.
		 */
		template<typename T1, typename T2, typename T3>
		void swapTriplet( T1& a, T2& b, T3& c, T1& u, T2& v, T3& w )
		{
			T1 tmpA = a; a = u; u = tmpA;
			T2 tmpB = b; b = v; v = tmpB;
			T3 tmpC = c; c = w; w = tmpC;
		}
	}
}

#endif //FAST_SWEEPING_GEOMETRY_H
