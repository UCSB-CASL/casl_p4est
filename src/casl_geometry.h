/**
 * A collection of geometric functions and classes involving points, vectors, planes, polygons, etc.
 * Developer: Luis Ángel.
 * Created: April 30, 2020.
 * Updated: March 8, 2022.
 */

#ifndef CASL_GEOMETRY_H
#define CASL_GEOMETRY_H

#include <iostream>
#include <src/types.h>
#include <src/my_p4est_macros.h>
#include <src/casl_math.h>
#include <src/point2.h>
#include <src/point3.h>
#include <random>
#include <unordered_set>
#include <unordered_map>

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

		void setCenter( DIM( double x0, double y0, double z0 ) )
		{
			this->_x0 = x0;
			this->_y0 = y0;
			ONLY3D( _z0 = z0 );
		}

		void setRadius( double r )
		{
			this->_r = r;
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

	/**
	 * Determines if a geometric shape (or point) lies inside or on the boundary of a relative area (like a limiting ellipse).
	 */
	enum RelPosType : unsigned char {UNDEFINED = 0, INTERIOR, BOUNDARY};

	///////////////////////////// A triangle class useful for triangulating Monge patches //////////////////////////////

	class Triangle
	{
	private:
		const Point3 *_v0, *_v1, *_v2;	// Pointers to triangle vertices to avoid redundancy with adjacent triangles.
		Point3 _n;						// (Non-normalized) normal vector to triangle's underlying plane.
		double _nNorm2;					// Norm^2 of normal vector.
		Point3 _edge0, _edge1, _edge2;	// The three sides of the triangle, being careful about the ordering.
		const RelPosType _relPosType;	// Determines relative location of traingle w.r.t. other enclosing shape (like a limiting ellipse).

		/**
		 * Project a 3D point on the plane subtended by the triangle.  If the projected point P falls within the
		 * triangle, the function returns true and sets P's barycentric coordinates u and v as follows:
		 *                P = u * v0 + v * v1 + ( 1 - u - v ) * v2,   for 0 <= u, v <= 1
		 * If P falls outside of triangle, the function yields the end-points of the first line segment that is found to
		 * fail the inside/outside test.  This implies that there might still exist at most another side for which the
		 * same inside/outside test fails.
		 * Regardless of the point being or not within the triangle, the projection point P is always computed.
		 * This function is an adaptation of the CG's ray-hitting-triangle algorithm provided in
		 * https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
		 * @param [in] p Query point.
		 * @param [out] u Produced barycentric coordinate of projected point within the triangle corresponding to v0.
		 * @param [out] v Produced barycentric coordinate of proejcted point within the triangle corresponding to v1.
		 * @param [out] P Normal projection of point p onto the triangle's plane.
		 * @param [out] x First end-point of first detected triangle's side to fail the inside/outside test.
		 * @param [out] y Second end-point of first detected triangle's side to fail the inside/outside test.
		 * @return True and valid barycentric coordinates if projected point falls within input triangle, false otherwise
		 * 		   and pointers to the end-points of first detected triangle's side to fail the inside/outside test.
		 */
		bool _projectPoint( const Point3 *p, double& u, double& v, Point3& P, const Point3*& x, const Point3*& y ) const
		{
			// Step 1: finding P, the projection of p onto the triangle's plane.
			Point3 vp0 = ( *p - *_v0 );
			P = *p - _n * vp0.dot( _n ) / _nNorm2;

			// Step 2: inside-outside test.
			// This test uses the cross product with a reference vector (i.e. a triangle's side).  When a point is
			// outside the triangle, we have at most two sides for which the inside/outside test fails:
			//         \                          \
			//          \   o Point                \
			//           \                    ------*.....
			//     -------*.....                     .  o Point
			//             .                          .
			//          (a)                       (b)
			// In case (a), there is just one side failing the inside/outside test.  In case (b), there are two.  In the
			// latter, just the first triangle's side to fail the test will be reported back.  However, in case (b) the
			// closest point on the triangle is the shared corner, here denoted as '*'.
			Point3 C; 							// Vector perpendicular to plane used for inside-outside test.
			x = y = nullptr;

			// Compare with edge 0.
			vp0 = P - *_v0;
			C = _edge0.cross( vp0 );
			if( _n.dot( C ) < 0 )				// P is on the right side of edge 0, opposed to v2.
			{
				x = _v0;
				y = _v1;
				return false;
			}

			// Compare with edge 1.
			Point3 vp1 = P - *_v1;
			C = _edge1.cross( vp1 );
			if( ( u = _n.dot( C ) ) < 0 )		// P is on the right side of edge 1, opposed to v0.
			{
				x = _v1;
				y = _v2;
				return false;
			}

			// Compare with edge 2.
			Point3 vp2 = P - *_v2;
			C = _edge2.cross( vp2 );
			if( ( v = _n.dot( C ) ) < 0 )		// P is on the right side of edge 2, opposed to v1.
			{
				x = _v2;
				y = _v0;
				return false;
			}

			u /= _nNorm2;
			v /= _nNorm2;

			return true; 						// The projected point P falls within the triangle.
		}

	public:
		/**
		 * Constructor.
		 * @param [in] v0 Pointer to triangle's first vertex.
		 * @param [in] v1 Pointer to triangle's second vertex.
		 * @param [in] v2 Pointer to triangle's third vertex.
		 * @param [in] TOL Optional tolerance to detect degenerate triangles by measuring collinearity.
		 * @param [in] relPosType Optional triangle type relative to some enclosing boundary.
		 */
		Triangle( const Point3 *v0, const Point3 *v1, const Point3 *v2,
				  const double& TOL=EPS, const RelPosType& relPosType=RelPosType::UNDEFINED )
				  : _v0( v0 ), _v1( v1 ), _v2( v2 ), _relPosType( relPosType )
		{
			// Compute triangle's subtended plane's normal vector.
			Point3 v0v1 = *v1 - *v0;
			Point3 v0v2 = *v2 - *v0;

			// No need to normalize, but we keep its norm^2.
			_n = v0v1.cross( v0v2 );
			_nNorm2 = _n.dot( _n );

			if( sqrt( _nNorm2 ) <= TOL )	// Check for collinear points.
				throw std::runtime_error( "[CASL_ERROR]: geom::Triangle::constructor: Triangle's vertices are collinear!" );

			// Let's compute the edges.
			_edge0 = *_v1 - *_v0;			// Defined in a circular fashion.
			_edge1 = *_v2 - *_v1;
			_edge2 = *_v0 - *_v2;
		}

		/**
		 * Find the closest point on triangle to another query point.
		 * @param [in] p Query point pointer.
		 * @param [in] TOL Zero-checking tolerance.
		 * @return Closest point on triangle.
		 */
		Point3 findClosestPointToQuery( const Point3 *p, double TOL=EPS ) const
		{
			double a, b;			// (Dummy) barycentric coordinates for projected point on triangle's plane.
			Point3 P;				// Projected point triangle's plane.
			const Point3 *u0, *u1;	// Pointers to vertices of line segment closest to P if the latter doesn't fall within the triangle.
			if( _projectPoint( p, a, b, P, u0, u1 ) )
				return P;
			else						// Find closest point from projected point to nearest triangle's segment that failed in/out test.
				return geom::findClosestPointOnLineSegmentToPoint( P, *u0, *u1, TOL );
		}

		/**
		 * Get one of triangles vertices.
		 * @param [in] i Index from 0 to 3.
		 * @return Pointer to requested vertex.
		 */
		const Point3 *getVertex( unsigned char i ) const
		{
			i = i % 3;
			switch( i )
			{
				case 0: return _v0;
				case 1: return _v1;
				default: return _v2;
			}
		}

		/**
		 * Retrieve the *unnormalized* normal vector to triangle.
		 * @return Pointer to normal vector.
		 */
		const Point3 *getNormal() const
		{
			return &_n;
		}

		/**
		 * Retrieve relative position state.
		 * @return Relative position state (any of the RelPosType enumeration).
		 */
		RelPosType getRelPosType() const
		{
			return _relPosType;
		}
	};

	/////////////////////////////////////// Balltree and related functionalities ///////////////////////////////////////

	struct BalltreeNode
	{
		int id = -1;							// Node identifier (mostly used for debugging).
		Point3 center;							// Ball center.
		double radius = 0;						// Ball radius.
		Point3 axis;							// Axis of maximum spread for children under this node.
		double m = 0;							// Decision boundary for left (< m) and right (>= m) projections onto the axis.
		BalltreeNode *leftChild = nullptr;		// Left an right children.  If node is a leaf, these should remain null.
		BalltreeNode *rightChild = nullptr;
		std::vector<const Point3 *> points;		// For leaf nodes, these is the list of (pointers to) actual points.
	};

	/**
	 * Balltree implementation as discussed in:
	 * @cite https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html
	 * @cite http://www.cs.cmu.edu/~agray/approxnn.pdf
	 *
	 * For the sklearn documentation of the balltree used in knn operations see:
	 * @cite https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
	 */
	class Balltree
	{
	private:
		const size_t _k;						// Minimum number of points to create a leaf.
		BalltreeNode *_root = nullptr;			// Tree root.
		std::vector<Point3> _points;			// Save a copy of all points (at the leaves) in tree.
		std::mt19937 _rng;						// Random generator.
		size_t _nodeCount;						// Used for node ids.
		bool _trace;							// Trace nodes visited during a knn query.

		/**
		 * Find the farthest point in a list S from a query point q.
		 * @param [in] q Query point.
		 * @param [in] S List of (pointers to) points to test.
		 * @return Pointer to farthest point.
		 */
		static const Point3 *_findFarthest( const Point3 *q, const std::vector<const Point3 *>& S )
		{
			double dFarthest = -1;
			const Point3 *farthest = nullptr;
			for( const auto& s : S )
			{
				double d = (*q - *s).norm_L2();
				if( d > dFarthest )
				{
					dFarthest = d;
					farthest = s;
				}
			}

			return farthest;
		}

		/**
		 * Find the k closest points in a list S to a query point q.
		 * @param [in] q Query point.
		 * @param [in] S List of (pointers) to test.
		 * @param [out] d Shortest distance with closest point.
		 * @param [in] k Maximum number of nearest neighbors to keep track of.
		 * @param [in] init Whether to discard the contents of xs and ds through initialization.
		 * @return Closest point.
		 */
		static void _findKClosest( const Point3& q, const std::vector<const Point3 *>& S, Point3 const **xs, double *ds,
								   const unsigned char& k=1, const bool& init=true )
		{
			// Initialize as needed.
			if( init )
			{
				for( int i = 0; i < k; i++ )
				{
					xs[i] = nullptr;
					ds[i] = DBL_MAX;
				}
			}

			// Search.
			for( const auto& s : S )
			{
				double dist = (q - *s).norm_L2();
				for( int i = 0; i < k; i++ )	// Find where to place the new closest point and distance.
				{
					if( dist < ds[i] )			// ith location must be updated; shift everything after that.
					{
						for( int j = k - 1; j > i; j-- )	// Shift data.
						{
							xs[j] = xs[j-1];
							ds[j] = ds[j-1];
						}
						xs[i] = s;
						ds[i] = dist;
						break;
					}
				}
			}
		}

		/**
		 * Construct tree and subtrees recursively.
		 * @param [in] S List of (pointers) to points.
		 * @return A pointer to a balltree node, which can be a leaf or an internal node with children.
		 */
		BalltreeNode *_buildTree( const std::vector<const Point3 *>& S )
		{
			// Populate (internal) node info.
			auto *node = new BalltreeNode;
			node->id = (int)_nodeCount;
			_nodeCount++;
			if( S.size() == 1 )					// Even for leaves, we must have a center and radius.
			{									// Here, we just speed out for the case of a single point.
				node->center = *S[0];
				node->radius = 0;
			}
			else
			{
				node->center = mean( S );
				const Point3 *xf = _findFarthest( &(node->center), S );
				node->radius = (*xf - node->center).norm_L2();
			}

			// Build (sub)tree rooted at node.
			if( S.size() <= _k )				// Got to a leaf?
			{
				node->points.reserve( _k );		// Store pointers to points to local storage.
				for( const auto& s : S )
					node->points.emplace_back( s );
			}
			else
			{
				// Find axis of maximum spread (approximately).
				std::uniform_int_distribution<size_t> dist( 0, S.size() - 1 );
				size_t x0Idx = dist( _rng );
				const Point3 *x1 = _findFarthest( S[x0Idx], S );
				const Point3 *x2 = _findFarthest( x1, S );
				node->axis = ( *x1 - *x2 );

				if( ANDD( ABS( node->axis.x ) <= PETSC_MACHINE_EPSILON, ABS( node->axis.y ) <= PETSC_MACHINE_EPSILON,
						  ABS( node->axis.z ) <= PETSC_MACHINE_EPSILON ) )
					throw std::runtime_error( "[CASL_ERROR]: geom::Balltree::_builTree: Singular axis!" );

				// Project data onto axis of maximum spread and classify them into left and right.
				Point3 midPoint = (*x1 + *x2) / 2;
				node->m = node->axis.dot( midPoint );		// This just an approximation to the median on the axis.
				std::vector<const Point3 *> Sl, Sr;
				Sl.reserve( 2 * S.size() / 3 );
				Sr.reserve( 2 * S.size() / 3 );
				for( const auto& s : S )
				{
					double z = node->axis.dot( *s );
					if( z < node->m )						// To the left or right of decision boundary?
						Sl.emplace_back( s );
					else
						Sr.emplace_back( s );
				}

				// Catch any possibility of having a degenerate tree where either the left or righ subtree don't exist.
				if( Sl.empty() || Sr.empty() )
					throw std::runtime_error( "[CASL_ERROR]: geom::Balltree::_buildTree: Either the left or right subtree is empty!" );
				node->leftChild = _buildTree( Sl );
				node->rightChild = _buildTree( Sr );
			}
			return node;
		}

		/**
		 * Deallocate (sub)trees recursively.  Upon exiting, the (sub)tree's root (n) becomes null.
		 * @param [in,out] n (Sub)tree root node.
		 */
		void _destroy( BalltreeNode *& n )
		{
			if( n->leftChild )
			{
				_destroy( n->leftChild );	// Destroy left subtree.
				n->leftChild = nullptr;
			}
			if( n->rightChild )
			{
				_destroy( n->rightChild );	// Destroy right subtree.
				n->rightChild = nullptr;
			}

			delete n;
			n = nullptr;
		}

		/**
		 * Explore the (sub)tree rooted at n recursively to find the k closest data points x (with distances d) to the
		 * query point q.
		 * @note Prior to calling this function for the first time, make sure the output arrays contain nullptr values.
		 * @param [in] q Query point.
		 * @param [out] xs Closest points to q in currently explored (sub)tree.
		 * @param [out] ds Distances from q to closest points x.
		 * @param [in] n (Sub)tree root or node to explore (if a leaf).
		 * @param [in] k Maximum number of closest points to keep track of.
		 */
		void _findKNearestNeighbors( const Point3& q, Point3 const **xs, double *ds, const BalltreeNode *n,
									 const unsigned char& k=1 ) const
		{
			if( _trace )
				std::cout << n->id << " ";

			if( !n->leftChild && !n->rightChild )					// If a leaf, compare against all nodes in it.
			{
				_findKClosest( q, n->points, xs, ds, k, false );	// Find k closest without overwriting current results.

				if( _trace )
					std::cout << "* tested leaf points" << std::endl;
			}
			else
			{
				if( _trace )
					std::cout << std::endl;

				double z = n->axis.dot( q );		// Project point on node's main axis and determine which child
				const BalltreeNode *first, *second;	// subtree to explore first.
				if( z < n->m )
				{
					first = n->leftChild;
					second = n->rightChild;
				}
				else
				{
					first = n->rightChild;
					second = n->leftChild;
				}

				if( (first->center - q).norm_L2() - first->radius < ds[k - 1] )	// Probe and prune left subtree if no
					_findKNearestNeighbors( q, xs, ds, first, k );				// member in it is within distance d from q.

				if( (second->center - q).norm_L2() - second->radius < ds[k - 1] )	// Probe and prune right subtree if no
					_findKNearestNeighbors( q, xs, ds, second, k );					// member in it is within distance d from q.
			}
		}

	public:
		/**
		 * Constructor.
		 * Default leaf size k based on value used in Python's scikit-learn module.
		 * @param [in] points List of points to arrange in the balltree.
		 * @param [in] copyPoints Whether to make local copies of input points.
		 * @param [in] k Maximum number of points in a leaf node.
		 * @param [in] trace Whether to trace nodes visited during a knn query.
		 */
		explicit Balltree( const std::vector<Point3>& points, const bool& copyPoints=true, const size_t& k=40,
						   const bool& trace=false )
						   : _k( k ), _rng( 0 ), _nodeCount( 0 ), _trace( trace )	// NOLINT.
		{
			// First, copy all points into local storage if user wants that.
			// At the same time, populate the list to pass to build function.
			std::vector<const Point3*> S;
			S.reserve( points.size() );
			if( copyPoints )
			{
				_points.reserve( points.size() );
				for( const Point3& point : points )
				{
					_points.emplace_back( point );
					S.emplace_back( &_points.back() );
				}
			}
			else
			{
				for( const Point3& point : points )		// If no copy is desired, user is responsible for not releasing
					S.emplace_back( &point );			// the list of points to avoid memory issues.
			}

			// Next, build the tree.
			_root = _buildTree( S );
		}

		/**
		 * Constructor.
		 * Default leaf size k based on value used in Python's scikit-learn module.
		 * @note The user is responsible for providing a list of existing points.
		 * @param [in] points List of pointers to point objects.
		 * @param [in] k Maximum number of points in a leaf node.
		 * @param [in] trace Whether to trace nodes visited during a knn query.
		 */
		explicit Balltree( const std::vector<const Point3 *>& points, const size_t& k=40, const bool& trace=false )
						   : _k( k ), _rng( 0 ), _nodeCount( 0 ), _trace( trace )
		{
			if( k <= 0 || points.empty() )
				throw std::runtime_error( "[CASL_ERROR] Balltree:constructor: k must be positive and points non empty!" );
			_root = _buildTree( points );
		}

		/**
		 * Find the k nearest neighbors to a query point q and send them back in distance-ascending order.
		 * @param [in] q Query point.
		 * @param [out] xs Nearest neighbor points.
		 * @param [out] ds Nearest neighbor distances.
		 * @param [in] k Maximum number of neighbors to keep track of.
		 * @param [in] init Whether to override xs and ds through initialization.
		 * @throws invalid_argument exception if k is not at least 1.
		 */
		void findKNearestNeighbors( const Point3& q, Point3 const **xs, double *ds, const unsigned char& k=1,
									const bool& init=true ) const
		{
			if( k == 0 )
				throw std::invalid_argument( "[CASL_ERROR] geom::Balltree::findKNearestNeighbors: k must be at least 1!" );

			if( init )	// Reset output arrays through initialization?
			{
				for( int i = 0; i < k; i++ )
				{
					xs[i] = nullptr;
					ds[i] = DBL_MAX;
				}
			}

			if( _trace )
			{
				std::cout << "Tracing query for [" << q.x << ", " << q.y << ", " << q.z << "]" << std::endl;
				std::cout << "--- Visited nodes ---" << std::endl;
			}

			_findKNearestNeighbors( q, xs, ds, _root, k );

			if( _trace )
				std::cout << "--- end ---" << std::endl;
		}

		/**
		 * Find the nearest point to a query point q.
		 * @param [in] q Query point.
		 * @param [out] d Shortest distance.
		 * @return Nearest neighbor.
		 */
		const Point3 *findNearestNeighbor( const Point3& q, double& d ) const
		{
			Point3 const* x;
			findKNearestNeighbors( q, &x, &d, 1, true );
			return x;
		}

		/**
		 * Destructor.
		 */
		~Balltree()
		{
			if( _root )
				_destroy( _root );

			_nodeCount = 0;		// Reset node counter.
		}
	};

	/**
	 * Abstract Monge patch function Q(u,v).
	 */
	class MongeFunction : public CF_2
	{
	public:
		virtual double meanCurvature( const double& u, const double& v ) const = 0;
		virtual double gaussianCurvature( const double& u, const double& v ) const = 0;
	};

	/**
	 * A two-dimensional Monge patch (surface) triangulated and discretized into a balltree for fast shortest-distance
	 * calculations.
	 * The class describes a Monge patch (x, y, f(x,y)) for (x,y) in a region R that is symmetric in every direction.
	 * The domain R is rectangular, with h = 2^{-L}, where L > 0.  This is similar to how we handle quadtrees, but here
	 * there are no intermediate cells.
	 */
	class DiscretizedMongePatch : public CF_3
	{
	protected:
		size_t _nPointsAlongU, _nPointsAlongV;	// How many points in each Cartesian direction.
		double _uMin, _vMin;					// Minimum coordinates (_uMin, _vMin) or the lower-left corner.
		double _h;								// "Mesh" size or the spacing between grid points on the uv plane.
		Balltree *_balltree;					// Underlying balltree organization of points in space.
		std::vector<Triangle> _triangles;		// List of triangles discretizing the Monge patch.
		const MongeFunction *_mongeFunction;	// Monge function to compute the "height" and curvature at any (x,y).
		std::vector<const Point3 *> _points;	// Points defining the grid (or nullptr if they lie outside a limiting ellipse).
		std::vector<RelPosType> _pointTypes;	// Determines whether created points are interior or boundary.
		std::vector<std::vector<const Triangle *>> _pointsToTriangles;	// Tracks which triangles each point is part of.
		mutable Point3 _lastNearestUVQ;					// Stores the u-v-q coords and the triangle of last
		mutable const Triangle *_lastNearestTriangle;	// nearest-point query

	public:
		/**
		 * Constructor.
		 * @note If a limiting ellipse is desired, both sau and sav must be positive and less than DBL_MAX.
		 * @param [in] ku Number of min cells in u half direction to define a symmetric domain.
		 * @param [in] kv Number of min cells in v half direction to define a symmetric domain.
		 * @param [in] L Number of refinement levels per unit length to define the cell width as H = 2^{-L}.
		 * @param [in] mongeFunction A function of the form f(u,v) --parametrized as [u,v,f(u,v)].
		 * @param [in] btKLeaf Maximum number of points in balltree leaf nodes.
		 * @param [in] sau2 Squared half-axis length on the u direction for the limiting ellipse on the uv plane.
		 * @param [in] sav2 Squared half-axis length on the v direction for the limiting ellipse on the uv plane.
		 */
		DiscretizedMongePatch( const size_t& ku, const size_t& kv, const size_t& L, const MongeFunction *mongeFunction,
							   const size_t& btKLeaf=40, const double& sau2=DBL_MAX, const double& sav2=DBL_MAX )
							   : _mongeFunction( mongeFunction ), _lastNearestTriangle( nullptr )
		{
			// Validate inputs.
			std::string errorPrefix = "[CASL_ERROR]: geom::DiscretizedMongePatch::constructor: ";
			if( !mongeFunction )
				throw std::runtime_error( errorPrefix + "Monge patch function can't be null!" );

			if( L == 0 )
				throw std::runtime_error( errorPrefix + "Monge patch function can't be null!" );

			if( ku == 0 || kv == 0 )
				throw std::runtime_error( errorPrefix + "Number of cells can't be zero!" );

			if( sau2 <= 0 || sav2 <= 0 )
				throw std::runtime_error( errorPrefix + "Limiting ellipse squared half-axis lengths must be positive!" );

			// Initializing space variables and domain.
			_h = 1. / (1 << L);								// Spacing.
			_uMin = -(double)ku * _h;						// Lower-left coordinate is at (_uMin, _vMin).
			_vMin = -(double)kv * _h;
			_nPointsAlongU = 2 * ku + 1;					// This is equivalent to (2|_dMin|/h + 1).
			_nPointsAlongV = 2 * kv + 1;
			_points.resize( _nPointsAlongU * _nPointsAlongV, nullptr );
			_pointTypes.resize( _nPointsAlongU * _nPointsAlongV, RelPosType::UNDEFINED );

			std::vector<const Point3*> balltreePoints;		// Array of (valid) point pointers to build the balltree.
			balltreePoints.reserve( _nPointsAlongU * _nPointsAlongV );

			// Lambda function for in/out test of bounding ellipse.
			auto _inOutTest = [&sau2, &sav2]( double& u, double& v ){
				return SQR( u ) / sau2 + SQR( v ) / sav2 <= 1;
			};

			// Let's create the points within ellipse and populate the list to pass on to the balltree.
			for( int j = 0; j < _nPointsAlongV; j++ )		// Rows, starting from the bottom-left corner.
			{
				for( int i = 0; i < _nPointsAlongU; i++ )	// Columns.
				{
					double x = _uMin + (double)i * _h;
					double y = _vMin + (double)j * _h;
					double z = _mongeFunction->operator()( x, y );
					size_t idx = _nPointsAlongU * j + i;

					if( sau2 == DBL_MAX || sav2 == DBL_MAX || _inOutTest( x, y ) )	// No limit or inside ellipse? Interior point.
					{
						_points[idx] = new Point3( x, y, z );
						_pointTypes[idx] = RelPosType::INTERIOR;
						balltreePoints.emplace_back( _points[idx] );
					}
					else	// Bring in any point outside ellipse if it has a neighbor inside.  If so, new point is a boundary point.
					{
						int neighbors[8][2] = {
							{i-1, j-1}, {i, j-1}, {i+1, j-1},	// Bottom neighbors.
							{i-1,   j}, {i+1, j}, 				// Same row neighbors.
							{i-1, j+1}, {i, j+1}, {i+1, j+1}	// Top neighbors.
						};

						for( const auto& neighbor : neighbors )	// See if there's at least one seed neighbor.
						{
							if( (neighbor[0] >= 0 && neighbor[0] < _nPointsAlongU) &&
								(neighbor[1] >= 0 && neighbor[1] < _nPointsAlongV) )
							{
								double xn = _uMin + (double)neighbor[0] * _h;
								double yn = _vMin + (double)neighbor[1] * _h;
								if( _inOutTest( xn, yn ) )
								{
									_points[idx] = new Point3( x, y, z );
									_pointTypes[idx] = RelPosType::BOUNDARY;
									balltreePoints.emplace_back( _points[idx] );
									break;						// Found seed neighbor -- halt loop.
								}
							}
						}
					}
				}
			}

			// Organize the points into a balltree for fast knn search.
			_balltree = new Balltree( balltreePoints, btKLeaf );

			// Triangulation.  The pattern is the following, starting from the bottom-left corner of the domain.
			//     :    :    :    :
			//   2 +----+----+----+····
			//     |  / |  / |  / |
			//     | /  | /  | /  |
			//   1 +----+----+----+····
			//     |  / |  / |  / |
			//     | /  | /  | /  |
			//     +----+----+----+····
			//   0      1    2    3
			_triangles.reserve( (_nPointsAlongU - 1) * (_nPointsAlongV - 1) * 2 );	// Here, we save the real triangles; then we use pointers to them.

			for( const auto& point : _points )				// Let's make space for the map of points to triangles.
			{
				_pointsToTriangles.emplace_back( std::vector<const Triangle *>() );
				if( point ) 								// Each valid point is part of at most 6 triangles under the above scheme plus 2 to
					_pointsToTriangles.back().reserve( 8 );	// complete the quads.  Edge points belong to 3+1, and corners belong to 1+1.
			}

			for( size_t j = 0; j < _nPointsAlongV - 1; j++ )		// Rows first (without getting to the very last).
			{
				for( size_t i = 0; i < _nPointsAlongU - 1; i++ )	// Columns next.
				{
					//                         idx3 +----+ idx2
					// Each quad has indices:       |  / |
					//                              | /  |
					//                         idx0 +----+ idx1
					// This way of enumerating the nodes in triangles sets their normal to vector 01×02 and 02×3 for low
					// and up triangles, resp.  If the surface is flat, this means the normals point up.
					size_t idx0 = _nPointsAlongU * j + i;		// Node indices in ccw direction.
					size_t idx1 = idx0 + 1;
					size_t idx2 = idx1 + _nPointsAlongU;
					size_t idx3 = idx0 + _nPointsAlongU;
					if( _points[idx0] && _points[idx1] && _points[idx2] )	// Can we create lower triangle?
					{
						RelPosType rpt = RelPosType::INTERIOR;
						if( _pointTypes[idx0] == RelPosType::BOUNDARY || 	// If at least one point is a boundary type,
							_pointTypes[idx1] == RelPosType::BOUNDARY || 	// triangle will be marked as boundary tye too.
							_pointTypes[idx2] == RelPosType::BOUNDARY )
							rpt = RelPosType::BOUNDARY;

						_triangles.emplace_back( _points[idx0], _points[idx1], _points[idx2], EPS, rpt );	// Build it and
						if( _points[idx0] ) _pointsToTriangles[idx0].push_back( &_triangles.back() );		// add pointers
						if( _points[idx1] ) _pointsToTriangles[idx1].push_back( &_triangles.back() );		// to it for
						if( _points[idx2] ) _pointsToTriangles[idx2].push_back( &_triangles.back() );		// defined quad
						if( _points[idx3] ) _pointsToTriangles[idx3].push_back( &_triangles.back() );		// points.
					}
					if( _points[idx0] && _points[idx2] && _points[idx3] )	// Can we create upper triangle?
					{
						RelPosType rpt = RelPosType::INTERIOR;
						if( _pointTypes[idx0] == RelPosType::BOUNDARY || 	// If at least one point is a boundary type,
							_pointTypes[idx2] == RelPosType::BOUNDARY || 	// triangle will be marked as boundary tye too.
							_pointTypes[idx3] == RelPosType::BOUNDARY )
							rpt = RelPosType::BOUNDARY;

						_triangles.emplace_back( _points[idx0], _points[idx2], _points[idx3], EPS, rpt );	// Build it and
						if( _points[idx0] ) _pointsToTriangles[idx0].push_back( &_triangles.back() );		// add pointers
						if( _points[idx2] ) _pointsToTriangles[idx2].push_back( &_triangles.back() );		// to it.
						if( _points[idx3] ) _pointsToTriangles[idx3].push_back( &_triangles.back() );
						if( _points[idx1] ) _pointsToTriangles[idx1].push_back( &_triangles.back() );
					}
				}
			}
		}

		/**
		 * Find nearest point to triangulated surface using a k-nearest-neighbor approach.
		 * @param [in] q Query point.
		 * @param [out] d Shortest distance to triangulated surface.
		 * @param [out] t Triangle where shortest distance was found.
		 * @param [in] k Maximum number of neighbors to find and pull triangles from.
		 * @return Nearest point.
		 * @throws invalid_argument exception if k == 0.
		 */
		Point3 findNearestPointFromKNN( const Point3& q, double& d, const Triangle *& t, const unsigned char& k ) const
		{
			if( k == 0 )
				throw std::invalid_argument( "[CASL_ERROR] geom::DiscretizedMongePatch::findKNearestPointFromKNN: k must be at least 1!" );

			// First, find the closest discrete points in the cloud.
			auto *ds = new double[k];
			auto const* *xs = new Point3 const*[k];
			_balltree->findKNearestNeighbors( q, xs, ds, k, true );		// This function also initializes vars to correct value.

			// Next, compute distance to triangles, and keep the minimum.
			int nn = 0;
			d = DBL_MAX;
			Point3 closestPoint;
			std::unordered_set<Triangle const*> visitedT( k * 8 );		// Memoization to avoid double-checking triangles.
			while( nn < k && xs[nn] )							// Check triangles attached to each found nearest point.
			{
				auto j = (size_t)((xs[nn]->y - _vMin) / _h);	// Row.
				auto i = (size_t)((xs[nn]->x - _uMin) / _h);	// Col.
				size_t idx = j * _nPointsAlongU + i;			// Node id.
				for( const auto& triangle : _pointsToTriangles[idx] )
				{
					if( visitedT.count( triangle ) )			// Skip triangles already visited.
						continue;
					visitedT.insert( triangle );

					Point3 p = triangle->findClosestPointToQuery( &q );
					double d1 = (p - q).norm_L2();
					if( d1 < d )								// Found a closer point?
					{
						closestPoint = p;
						d = d1;
						t = triangle;
					}
				}

				nn++;
			}

			// Clean up.
			delete [] ds;
			delete [] xs;

			return closestPoint;
		}

		/**
		 * Find nearest point to triangulated surface using a k-nearest-neighbor approach with k = 1.
		 * @param [in] q Query point.
		 * @param [out] d Shortest distance to triangulated surface.
		 * @param [out] t Triangle where shortest distance was found.
		 * @return Nearest point.
		 */
		Point3 findNearestPoint( const Point3& q, double& d, const Triangle *& t ) const
		{
			return findNearestPointFromKNN( q, d, t, 1 );
		}

		/**
		 * Implementation of query function for distance.  This is not "signed" distance though.
		 * Upon exiting, the function sets the variable _lastNearestUVQ with the coordinates of nearest point to query.
		 * @note A child class should re-implement this function accounting for the signed distance to the discretized
		 * interface.
		 * @warning Not a thread-safe method if you need to access to _lastNearestUVQ or _lastNearestTriangle; use the
		 * findNearestPoint() function instead.
		 * @param [in] x Coordinate in x.
		 * @param [in] y Coordinate in y.
		 * @param [in] z Coordinate in z.
		 * @return Distance to triangulated surface.
		 */
		double operator()( double x, double y, double z ) const override
		{
			Point3 q( x, y, z );
			double d = DBL_MAX;
			_lastNearestUVQ = findNearestPoint( q, d, _lastNearestTriangle );
			return  d;
		}

		/**
		 * Retrieve the coordinates of the nearest point on the triangulated surface to a query point performed in the
		 * ()-operator method.  Function useful for caching.
		 * @return Last nearest point from a query.
		 */
		Point3 getLastNearestUVQ() const
		{
			return _lastNearestUVQ;
		}

		/**
		 * Retrieve the nearest triangle from a query point after using the ()-operator method.  This function is useful
		 * to determine the sign of a grid point according to the linear reconstruction of the interface.
		 * @return Pointer to triangle holding the nearest point to query.
		 */
		const Triangle *getLastNearestTriangle() const
		{
			return _lastNearestTriangle;
		}

		/**
		 * Destructor.
		 */
		~DiscretizedMongePatch() override
		{
			delete _balltree;
			_lastNearestTriangle = nullptr;

			for( const auto &point : _points )	// Release points created dynamically.
				delete point;
		}

		/**
		 * Dump triangle vertices for debugging.
		 * @param [in,out] output File stream that has been already opened.
		 */
		void dumpTriangles( std::ofstream& output ) const
		{
			for( const auto& triangle : _triangles )
			{
				const Point3 *v;
				for( char unsigned i = 0; i < 2; i++ )
				{
					v = triangle.getVertex( i );
					output << v->x << "," << v->y << "," << v->z << ",";
				}
				v = triangle.getVertex( 2 );
				output << v->x << "," << v->y << "," << v->z << std::endl;
			}
		}
	};

	/**
	 * An abstract class for a level-set function whose zero isocontour is discretized into an affine-transformed Monge
	 * patch.
	 */
	class DiscretizedLevelSet : public geom::DiscretizedMongePatch
	{
	protected:
		__attribute__((unused)) const double _beta;	// Transformation parameters to vary canonical system w.r.t. world
		const Point3 _axis;							// coordinate system.  These include a rotation (unit) axis and
		const Point3 _trns;							// angle, and a translation vector that sets the origin of the
													// canonical system to any point in space.
		const double _c, _s;						// Since we use cosine and sine of beta a lot, let's precompute them.
		const double _one_m_c;						// (1-cos(beta)).

		bool _useCache = false;	// Computing distance to triangulated surface is expensive --let's cache distances and
		mutable std::unordered_map<std::string, std::pair<double, Point3>> _cache;	// nearest points.  Also, cache can-
		mutable std::unordered_map<std::string, Point3> _canonicalCoordsCache;		// onical coords for grid points.

		/**
		 * Find nearest point and signed distance to triangulated surface using knn search.
		 * @param [in] p Query point in canonical coords.
		 * @param [out] d Computed signed distance.
		 * @param [in] k Number of nearest neighbors to use.
		 * @return Nearest point.
		 */
		Point3 _findNearestPointAndSignedDistance( const Point3& p, double& d, const unsigned char& k=1 ) const
		{
			d = DBL_MAX;
			const geom::Triangle *nearestTriangle;
			Point3 nearestPoint;
			nearestPoint = findNearestPointFromKNN( p, d, nearestTriangle, k );		// Use knn for higher accuracy.

			// Fix sign: points above surface are negative and below are positive.  Because of the way we created the
			// triangles, their normals point up in the canonical coord system (i.e., into the negative region Omega-).
			Point3 w = p - *(nearestTriangle->getVertex(0));
			if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )	// In the direction of the normal?
				d *= -1;

			return nearestPoint;
		}

	public:
		/**
		 * Constructor.
		 * @param [in] trans Translation vector.
		 * @param [in] rotAxis Rotation axis (must be nonzero).
		 * @param [in] rotAngle Rotation angle about rotAxis.
		 * @param [in] ku Number of min cells in u half direction to define a symmetric domain.
		 * @param [in] kv Number of min cells in v half direction to define a symmetric domain.
		 * @param [in] L Number of refinement levels per unit length (so that h=2^{-L} is a power of two).
		 * @param [in] mongeFunction Monge patch in canonical coordinates.
		 * @param [in] sau2 Squared half-axis length on the u direction for the limiting ellipse on the uv plane.
		 * @param [in] sav2 Squared half-axis length on the v direction for the limiting ellipse on the uv plane.
		 * @param [in] btKLeaf Maximum number of points in balltree leaf nodes.
		 */
		DiscretizedLevelSet( const Point3& trans, const Point3& rotAxis, const double& rotAngle, const size_t& ku,
							 const size_t& kv, const size_t& L, const MongeFunction *mongeFunction,
							 const double& sau2=DBL_MAX, const double& sav2=DBL_MAX, const size_t& btKLeaf=40 )
							 : _trns( trans ), _axis( rotAxis.normalize() ), _beta( rotAngle),
							 _c( cos( rotAngle ) ), _s( sin( rotAngle ) ), _one_m_c( 1 - cos( rotAngle ) ),
							 DiscretizedMongePatch( ku, kv, L, mongeFunction, btKLeaf, sau2, sav2 )
		{
			const std::string errorPrefix = "[CASL_ERROR] DiscretizedLevelSet::constructor: ";

			if( !mongeFunction )
				throw std::invalid_argument( errorPrefix + "Sinusoid surface object can't be null!" );

			if( rotAxis.norm_L2() < EPS )		// Singular rotation axis?
				throw std::invalid_argument( errorPrefix + "Rotation axis shouldn't be 0!" );
		}

		/**
		 * Compute exact signed distance to Gamma.
		 * @param [in] x Query x-coordinate.
		 * @param [in] y Query y-coordinate.
		 * @param [in] z Query z-coordinate.
		 * @param [out] updated Status flag set to 1 if exact-distance computation succeeded, 2 otherwise.
		 * @return Shortest signed distance.
		 */
		virtual double computeExactSignedDistance( double x, double y, double z, unsigned char& updated ) const = 0;

		/**
		 * @see DiscretizedLevelSet::computeExactSignedDistance( double x, double y, double z, unsigned char& updated ).
		 * @param [in] xyz Query point in world coordinates.
		 * @param [out] updated Status flag set to 1 if exact-distance computation succeeded, 2 otherwise.
		 * @return Shortest signed distance.
		 */
		double computeExactSignedDistance( const double xyz[P4EST_DIM], unsigned char& updated ) const
		{
			return computeExactSignedDistance( xyz[0], xyz[1], xyz[2], updated );
		}

		/**
		 * Transform a point/vector in world coordinates to canonical coordinates using the tranformation info.
		 * @param [in] x x-coordinate.
		 * @param [in] y y-coordinate.
		 * @param [in] z z-coordinate.
		 * @param [in] isVector True if input is a vector (unnaffected by translation), false if input is a point.
		 * @return The coordinates of (x,y,z) in the representation of the canonical coordinate system.
		 */
		Point3 toCanonicalCoordinates( const double& x, const double& y, const double& z, const bool& isVector=false ) const
		{
			Point3 r;
			const double xmt = isVector? x : x - _trns.x;		// Displacements affect points only.
			const double ymt = isVector? y : y - _trns.y;
			const double zmt = isVector? z : z - _trns.z;
			r.x = (_c + _one_m_c*SQR(_axis.x))*xmt + (_one_m_c*_axis.x*_axis.y + _s*_axis.z)*ymt + (_one_m_c*_axis.x*_axis.z - _s*_axis.y)*zmt;
			r.y = (_one_m_c*_axis.y*_axis.x - _s*_axis.z)*xmt + (_c + _one_m_c*SQR(_axis.y))*ymt + (_one_m_c*_axis.y*_axis.z + _s*_axis.x)*zmt;
			r.z = (_one_m_c*_axis.z*_axis.x + _s*_axis.y)*xmt + (_one_m_c*_axis.z*_axis.y - _s*_axis.x)*ymt + (_c + _one_m_c*SQR(_axis.z))*zmt;

			return r;
		}

		/**
		 * Transform a point/vector from canonical coordinates to world coordinates using the transformation info.
		 * @param [in] x x-coordinate.
		 * @param [in] y y-coordinate.
		 * @param [in] z z-coordinate.
		 * @param [in] isVector True if input is a vector (unnaffected by translation), false if input is a point.
		 * @return The coordinates (x,y,z) in the representation of the world coordinate system.
		 */
		Point3 toWorldCoordinates( const double& x, const double& y, const double& z, const bool& isVector=false ) const
		{
			Point3 r;
			r.x = x*(_c + _one_m_c*SQR(_axis.x)) + y*(_one_m_c*_axis.y*_axis.x - _s*_axis.z) + z*(_one_m_c*_axis.z*_axis.x + _s*_axis.y);
			r.y = x*(_one_m_c*_axis.x*_axis.y + _s*_axis.z) + y*(_c + _one_m_c*SQR(_axis.y)) + z*(_one_m_c*_axis.z*_axis.y - _s*_axis.x);
			r.z = x*(_one_m_c*_axis.x*_axis.z - _s*_axis.y) + y*(_one_m_c*_axis.y*_axis.z + _s*_axis.x) + z*(_c + _one_m_c*SQR(_axis.z));

			if( !isVector )
			{
				r.x += _trns.x;
				r.y += _trns.y;
				r.z += _trns.z;
			}

			return r;
		}

		/**
		 * Retrieve discrete coordinates as a triplet normalized by h.
		 * @param [in] x x-coordinate.
		 * @param [in] y y-coordinate.
		 * @param [in] z z-coordinate.
		 * @return "i,j,k".
		 */
		std::string getDiscreteCoords( const double& x, const double& y, const double& z ) const
		{
			return std::to_string( (int)(x/_h) ) + "," + std::to_string( (int)(y/_h) ) + "," + std::to_string( (int)(z/_h) );
		}

		/**
		 * Dump triangles into a data file for debugging/visualizing.
		 * @param [in] filename Output file name.
		 * @throws runtime_error if file can't be opened.
		 */
		void dumpTriangles( const std::string& filename )
		{
			std::ofstream trianglesFile;				// Dumping triangles' vertices into a file for debugging/visualizing.
			trianglesFile.open( filename, std::ofstream::trunc );
			if( !trianglesFile.is_open() )
				throw std::runtime_error( filename + " couldn't be opened for dumping mesh!" );
			trianglesFile << R"("x0","y0","z0","x1","y1","z1","x2","y2","z2")" << std::endl;
			trianglesFile.precision( 15 );
			geom::DiscretizedMongePatch::dumpTriangles( trianglesFile );
			trianglesFile.close();
		}

		/**
		 * Turn on/off cache for faster distance retrieval.
		 * @param [in] useCache True to enable cache, false to disable it.
		 */
		void toggleCache( const bool& useCache )
		{
			_useCache = useCache;
		}

		/**
		 * Empty cache.
		 */
		void clearCache()
		{
			_cache.clear();
			_canonicalCoordsCache.clear();
		}

		/**
		 * Reserve space for cache.  Call this function, preferably, at the beginning of queries or octree construction.
		 * @param [in] n Reserve size.
		 */
		void reserveCache( const size_t& n )
		{
			_cache.reserve( n );
			_canonicalCoordsCache.reserve( n );
		}

		/**
		 * Retrieve the cache size.
		 * @return _cache (and _canonicalCoordsCache) size.
		 */
		size_t getCacheSize() const
		{
			return _cache.size();
		}
	};

}

#endif // CASL_GEOMETRY_H
