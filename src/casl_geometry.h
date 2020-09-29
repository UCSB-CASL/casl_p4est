//
// Created by Luis Ángel on 4/30/20.
// Modified on 9/29/2020.
//

#ifndef CASL_GEOMETRY_H
#define CASL_GEOMETRY_H

#include <src/casl_math.h>
#include <src/point2.h>
#include <src/point3.h>

/**
 * A collection of geometric functions and classes involving points, vectors, planes, polygons, etc.
 */
namespace geom
{
	//////////////////////////////////////////////// Level-Set Classes /////////////////////////////////////////////////

	/**
	 * Point at the origin.
	 */
	class APoint: public CF_DIM
	{
	public:
		double operator()( DIM( double x, double y, double z ) ) const override
		{
			return sqrt( SUMD( SQR( x ), SQR( y ), SQR( z ) ) );			// Distance to the origin.
		}
	};

	/**
	 * A sphere level-set function in two/three dimensions.
	 * This is a signed distance function version of the sphere.
	 */
	class Sphere: public CF_DIM
	{
	protected:
		double DIM( _x0, _y0, _z0 ), _r;

	public:
		explicit Sphere( DIM( double x0 = 0, double y0 = 0, double z0 = 0 ), double r = 1 ) :
			DIM( _x0( x0 ), _y0( y0 ), _z0( z0 ) ), _r( r )
		{}

		double operator()( DIM( double x, double y, double z ) ) const override
		{
			return sqrt( SUMD( SQR( x - _x0 ), SQR( y - _y0 ), SQR( z - _z0 ) ) ) - _r;
		}
	};

	/**
	 * A sphere level-set function in two/three dimensions.
	 * This is the *non* signed distance function version of the sphere.
	 */
	class SphereNSD: public Sphere
	{
	public:
		explicit SphereNSD( DIM( double x0 = 0, double y0 = 0, double z0 = 0 ), double r = 1 ) :
			Sphere( DIM( x0, y0, z0 ), r )
		{}

		double operator()( DIM( double x, double y, double z ) ) const override
		{
			return SUMD( SQR( x - _x0 ), SQR( y - _y0 ), SQR( z - _z0 ) ) - SQR( _r );
		}
	};

	/**
	 * A multidimensional plane defined by a normal vector and a point.  The plane extends infinitely in any direction.
	 * Not a signed distance function.
	 */
	class Plane: public CF_DIM
	{
	private:
		PointDIM _n;		// Unit normal vector to plane.
		PointDIM _p;		// Point on the plane.

	public:
		/**
		 * Constructor.
		 * @param [in] n Normal vector to plane.
		 * @param [in] p Point on plane.
		 */
		explicit Plane( const PointDIM& n, const PointDIM& p )
			: _n( n ), _p( p )
		{
#ifdef CASL_THROWS
			if( _n.norm_L2() <= EPS )
				throw std::domain_error( "[CASL_ERROR]: geom::Plane: Division by zero!" );
#endif
			_n = _n.normalize();
		}

		/**
		 * Evaluate phi(q), where q=[x,y,z]'.
		 * @param [in] x Query coordinate along x-axis.
		 * @param [in] y Query coordinate along y-axis.
		 * @param [in] z Query coordinate along z-axis.
		 * @return n'(q-p), where n and p are the normal vector and point on plane, respectively.
		 */
		double operator()( DIM( double x, double y, double z ) ) const override
		{
			PointDIM q( DIM( x, y, z ) );
			q = q - _p;
			return _n.dot( q );
		}
	};

	/**
 	 * Star-shaped interface centered at the origin.
 	 */
	class Star: public CF_2
	{
	private:
		double _a;			// Perturbation (petal extension).
		double _b;			// Base circle radius.
		int _p;				// Number of arms.
		double _d;			// Angle phase.

	public:
		/**
		 * Constructor
		 * @param [in] a Perturbation amplitude.
		 * @param [in] b Base circle radius.
		 * @param [in] p Number of arms.
		 * @param [in] d Angular phase.
		 */
		explicit Star( double a = -1.0, double b = 3.0, int p = 8, double d = 0 )
			: _a( a ), _b( b ), _p( p ), _d( d ) {}

		/**
		 * Level set evaluation at a given point.
		 * @param [in] x Point x-coordinate.
		 * @param [in] y Point y-coordinate.
		 * @return phi(x,y).
		 */
		double operator()( double x, double y ) const override
		{
			return -_a * cos( _p * atan2( y, x ) - _d ) + sqrt( SQR( x ) + SQR( y ) ) - _b;
		}

		/**
		 * Compute point on interface given an angular parameter.
		 * @param [in] theta Angular parameter.
		 * @return r(theta).
		 */
		[[nodiscard]] double r( double theta ) const
		{
			return _a * cos( _p * theta - _d ) + _b;
		}

		/**
		 * Compute first derivative of star at a given angle.
		 * @param theta Angular parameter.
		 * @return r'(theta).
		 */
		[[nodiscard]] double rPrime( double theta ) const
		{
			return -_a * _p * sin( _p * theta - _d );
		}

		/**
		 * Compute second derivative of star at a given angle.
		 * @param theta Angular parameter.
		 * @return r''(theta).
		 */
		[[nodiscard]] double rPrimePrime( double theta ) const
		{
			return -_a * SQR( _p ) * cos( _p * theta - _d );
		}

		/**
		 * Retrieve the side-length of a square containing the star.
		 * @return Inscribing square side-length.
		 */
		[[nodiscard]] double getInscribingSquareSideLength() const
		{
			double limitVal = MAX( ABS( _a + _b ), ABS( _a - _b ) );
			return limitVal * 2;
		}

		/**
		 * Compute curvature.
		 * @param [in] theta Angle parameter.
		 * @return kappa.
		 */
		[[nodiscard]] double curvature( double theta ) const
		{
			double r = this->r( theta );						// r(theta).
			double rPrime = this->rPrime( theta );				// r'(theta).
			double rPrimePrime = this->rPrimePrime( theta );	// r''(theta).
			return ( SQR( r ) + 2 * SQR( rPrime ) - r * rPrimePrime ) / pow( SQR( r ) + SQR( rPrime ), 1.5 );
		}

		/**
	 	 * Compute the spacing and the minimum value for the current star parameters given a desired resolution.
	 	 * @param [in] nGridPoints Number of uniformly distributed grid points along each dimension.
	 	 * @param [out] h Spacing (i.e. minimum cell width at the maximum refinement level).
	 	 * @param [out] minVal Minimum value in the active domain [minVal, -minVal]^2.
	 	 * @param [in] padding Number of uniform grid points to be used as padding around star (min value is 2).
	 	 */
		void getHAndMinVal( unsigned nGridPoints, double& h, double& minVal, unsigned padding = 2 ) const
		{
			if( nGridPoints < 16 )
			{
				nGridPoints = 16;
				std::cerr << "Number of grid points is too small, it'll be set to 16" << std::endl;
			}

			if( padding < 2 )
			{
				padding = 2;
				std::cerr << "Number of padding grid points used should be at least 2. It'll be set to 2" << std::endl;
			}

			double limitVal = MAX( ABS( _a + _b ), ABS( _a - _b ) );	// We have at least 2h as padding on each
			h = 2 * limitVal / ( nGridPoints - 1 - 2 * padding );		// direction from the limit val of the flower.
			minVal = -limitVal - padding * h;	// This is the lowest value we consider in the domain to create a grid.
		}

		/**
		 * Get the 'a' perturbation star's parameter.
		 * @return star.a.
		 */
		[[nodiscard]] double getA() const
		{
			return _a;
		}

		/**
		 * Get the 'b' base radius star's parameter.
		 * @return star.b.
		 */
		[[nodiscard]] double getB() const
		{
			return _b;
		}

		/**
		 * Get the number of arms 'p' in the star.
		 * @return star.p.
		 */
		[[nodiscard]] int getP() const
		{
			return _p;
		}

		/**
		 * Get the angular phase.
		 * @return star.d.
		 */
		[[nodiscard]] double getD() const
		{
			return _d;
		}
	};

	/////////////////////////////////////////////// Geometric functions ////////////////////////////////////////////////

	/**
	 * Linearly interpolate a point based on level-set function values.
	 * @param [in] p1 First point.
	 * @param [in] phi1 Level-set function value at first point.
	 * @param [in] p2 Second point.
	 * @param [in] phi2 Level-set function value at second point.
	 * @param [in] TOL Distance zero-checking tolerance.
	 * @return Interpolated point between p1 and p2.
	 * @tparam Point One of Point2 or Point3 types.
	 * @throws Zero division error if input level-set function values are (almost) equal.
	 */
	template<typename Point = Point2>
	inline Point interpolatePoint( const Point *p1, double phi1, const Point *p2, double phi2, double TOL = EPS )
	{
		static_assert( std::is_base_of<Point2, Point>::value || std::is_base_of<Point3, Point>::value,
					   "[CASL_ERROR]: geom::interpolatePoint: Function may be called only by Point2 or Point3 objects!" );

#ifdef CASL_THROWS
		if( ABS( phi2 - phi1 ) <= TOL )
			throw std::domain_error( "[CASL_ERROR]: geom::interpolatePoint: Division by zero!" );
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
	 * @param [in] TOL Tolerance for zero-distance checking.
	 * @tparam Point One of Point2 or Point3 types.
	 * @return Closest point on the line segment v0v1.
	 */
	template<typename Point = Point2>
	inline Point findClosestPointOnLineSegmentToPoint( const Point& p, const Point& v0, const Point& v1, double TOL = EPS )
	{
		static_assert( std::is_base_of<Point2, Point>::value || std::is_base_of<Point3, Point>::value,
					   "[CASL_ERROR]: geom::interpolatePoint: Function may be called only by Point2 or Point3 objects!" );

		Point v = v1 - v0;
		double denom = v.dot( v );
		if( sqrt( denom ) <= TOL )						// Degenerate line segment?
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
	inline bool projectPointOnTriangleAndPlane( const Point3 *p, const Point3 *v0, const Point3 *v1, const Point3 *v2,
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
		Point3 vp0 = ( *p - *v0 );
		P = *p - N * vp0.dot( N ) / denom;

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
	 * Find the closest point on a triangle to another query point in 3D.
	 * @param [in] p Query point pointer.
	 * @param [in] v0 Pointer to first triangle's vertex.
	 * @param [in] v1 Pointer to second triangle's vertex.
	 * @param [in] v2 Pointer to third triangle's vertex.
	 * @param [in] TOL Zero-checking tolerance.
	 * @return Closest point on triangle.
	 */
	inline Point3 findClosestPointOnTriangleToPoint( const Point3 *p, const Point3 *v0, const Point3 *v1, const Point3 *v2, double TOL = EPS )
	{
		double a, b;				// (Dummy) barycentric coordinates for projected point on triangle's plane.
		Point3 P;					// Projected point triangle's plane.
		const Point3 *u0, *u1;		// Pointers to vertices of line segment closest to P if the latter doesn't fall within the triangle.
		if( projectPointOnTriangleAndPlane( p, v0, v1, v2, a, b, P, u0, u1 ) )
			return P;
		else						// Find closest point from projected point to nearest triangle's segment that failed in/out test.
			return geom::findClosestPointOnLineSegmentToPoint( P, *u0, *u1, TOL );
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
		inline void swapTriplet( T1& a, T2& b, T3& c, T1& u, T2& v, T3& w )
		{
			T1 tmpA = a; a = u; u = tmpA;
			T2 tmpB = b; b = v; v = tmpB;
			T3 tmpC = c; c = w; w = tmpC;
		}
	}
}

#endif // CASL_GEOMETRY_H
