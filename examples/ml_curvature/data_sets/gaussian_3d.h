/**
 * A collection of classes and functions related to a Gaussian surface embedded in 3D.
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: February 10, 2022.
 */

#ifndef ML_CURVATURE_GAUSSIAN_3D_H
#define ML_CURVATURE_GAUSSIAN_3D_H

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/casl_geometry.h>
#include <dlib/optimization.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/math/tools/roots.hpp>

///////////////////////////////////////// Gaussian surface in canonical space //////////////////////////////////////////

/**
 * A 2D Gaussian in canonical space (i.e., a surface in 3D) modeling Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)), where A
 * is its height and su^2 and sv^2 are the variances.  By definition, our Gaussian has zero means mu and mv.
 */
class Gaussian : public geom::MongeFunction
{
private:
	const double _a;			// Height.
	const double _su2, _sv2;	// Variances (i.e., squared values).
	const double _su, _sv;		// Standard deviations.

public:
	/**
	 * Constructor.
	 * @param [in] a Max height.
	 * @param [in] su2 Variance along the u direction.
	 * @param [in] sv2 Variance along the v direction.
	 * @throws Runtime error if any of the shape parameters are zero.
	 */
	Gaussian( const double& a, const double& su2, const double& sv2 )
			: _a( ABS(a) ), _su2( ABS( su2 ) ), _sv2( ABS( sv2 ) ), _su( sqrt( _su2 ) ), _sv( sqrt( _sv2 ) )
	{
		if( _a == 0 || _su2 == 0 || _sv2 == 0 )
			throw std::runtime_error( "[CASL_ERROR] Gaussian::constructor: a, su, and sv must be nonzero!" );
	}

	/**
	 * Evaluate Gaussian Q(u,v).
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)).
	 */
	double operator()( double u, double v ) const override
	{
		return _a * exp( -0.5 * (SQR( u ) / _su2 + SQR( v ) / _sv2 ) );
	}

	/**
	 * First derivative w.r.t. u.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return dQdu(u, v).
	 */
	double dQdu( double u, double v, double Q=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -Q * u / _su2;
	}

	/**
	 * First derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return dQdv(u, v).
	 */
	double dQdv( double u, double v, double Q=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -Q * v / _sv2;
	}

	/**
	 * Second derivative w.r.t. u.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @param [in] Qu (Optional) Cached dQdu(u,v).
	 * @return d2Qdu2(u, v).
	 */
	double d2Qdu2( double u, double v, double Q=NAN, double Qu=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		Qu = (isnan( Qu )? dQdu( u, v, Q ) : Qu);
		return -(Qu * u + Q) / _su2;
	}

	/**
	 * Second derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @param [in] Qv (Optional) Cached dQdv(u,v).
	 * @return d2Qdv2(u, v).
	 */
	double d2Qdv2( double u, double v, double Q=NAN, double Qv=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		Qv = (isnan( Qv )? dQdv( u, v, Q ) : Qv);
		return -(Qv * v + Q) / _sv2;
	}

	/**
	 * Crossed second derivative.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return d2Qdudv(u, v).
	 */
	double d2Qdudv( double u, double v, double Q=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return Q * u * v / (_su2 * _sv2);
	}

	/**
	 * Compute mean curvature (which actually is twice that).
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return 2H, where H is the mean curvature at (u,v,Q(u,v)) on the surface.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v, Q );		// Sending in cached values to avoid double evaluations.
		double Qv = dQdv( u, v, Q );
		double Quu = d2Qdu2( u, v, Q, Qu );
		double Qvv = d2Qdv2( u, v, Q, Qv );
		double Quv = d2Qdudv( u, v, Q );
		double kappa = ((1 + SQR( Qv )) * Quu - 2 * Qu * Qv * Quv + (1 + SQR( Qu )) * Qvv)	// Numerator.
					   / pow( 1 + SQR( Qu ) + SQR( Qv ), 1.5 );								// Denominator.
		return kappa;
	}

	/**
	 * Retrieve the variance along the u direction.
	 * @return _su^2.
	 */
	double su2() const
	{
		return _su2;
	}

	/**
	 * Retrieve the variance along the v direction.
	 * @return _sv^2.
	 */
	double sv2() const
	{
		return _sv2;
	}

	/**
	 * Retrieve the standard deviation along the u direction.
	 * @return _su.
	 */
	double su() const
	{
		return _su;
	}

	/**
	 * Retrieve the standard deviation along the v direction.
	 * @return _sv.
	 */
	double sv() const
	{
		return _sv;
	}

	/**
	 * Find the axis value in the Gaussian's canonical coordinate system where curvature becomes 0 using
	 * Newton-Raphson's method.
	 * @param [in] h Mesh size.
	 * @param [in] dir Either 0 for u or 1 for v.
	 * @return The p value where kappa(p,0) or kappa(0,p) is zero (depending on the chosen direction).
	 * @throws runtime_error if dir is not 0 or 1, or if bracketing or Newton-Raphson's method fails to find the root.
	 */
	double findKappaZero( const double& h, const unsigned char& dir ) const
	{
		using namespace boost::math::tools;

		if( dir != 0 && dir != 1 )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Wrong direction!  Choose either 0 for u or 1 for v." );

		// Define parameters depending on direction: 0 for u, 1 for v.
		double s2 = (dir == 0? _su2 : _sv2);
		double t2 = (dir == 0? _sv2 : _su2);

		// Curvature with one of the directions set to zero.
		auto kappa = [this, &dir, &s2, &t2]( const double& p ){
			double q = (dir == 0? (*this)(p, 0) : (*this)(0, p));
			return SQR( q * p ) + SQR( s2 ) + s2 * t2 - SQR( p ) * t2;
		};

		// And this is the simplified expression to compute both kappa and its derivative kappa' with one of the dirs set to zero.
		auto kappaAndDKappa = [this, &kappa, &dir, &s2, &t2]( const double& p ){
			double k =  kappa( p );
			double q = (dir == 0? (*this)(p, 0) : (*this)(0, p));
			double dk = 2 * p * (SQR( q ) * (1 - SQR( p ) / s2) - t2);
			return std::make_pair( k, dk );
		};

		const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
		int getDigits = static_cast<int>( digits * 0.75 );		// Accuracy doubles with each step in Newton-Raphson's, so
		// stop when we have just over half the digits correct.
		boost::uintmax_t it = 0;
		const boost::uintmax_t MAX_IT = 10;						// Maximum number of iterations for bracketing and Newton-Raphson's.

		double s = (dir == 0? _su : _sv);
		double start = h;										// Determine the initial bracket with a sliding a window.
		double end = 2 * s;										// We need to find an interval with different kappa signs in its endpoints.
		while( it < MAX_IT && kappa( start ) * kappa( end ) > 0 )
		{
			end += 0.5 * s;
			start += 0.5 * s;
			it++;
		}

		if( kappa( start ) * kappa( end ) > 0 )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Failed to find a reliable bracket for " + std::to_string( dir ) + " direction!" );

		double root = (start + end) / 2;	// Initial guess.
		it = MAX_IT;						// Find zero with Newton-Raphson's.
		root = newton_raphson_iterate( kappaAndDKappa, root, start, end, getDigits, it );

		if( it >= MAX_IT )
			throw std::runtime_error( "[CASL_ERROR] findKappaZero: Couldn't find zero with Newton-Raphson's method for " + std::to_string( dir ) + " direction!" );

		return root;
	}
};

///////////////////////////////////// Distance from P to Gaussian as a model class /////////////////////////////////////

/**
 * A function model for the distance function to the Gaussian.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) = a*exp(-0.5*(u^su^2+v^2/sv^2)) for positive
 * a, su, and sv, and P is the query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
 */
class GaussianPointDistanceModel
{
public:
	typedef dlib::matrix<double,0,1> column_vector;
	typedef dlib::matrix<double> general_matrix;

private:
	const Point3 P;		// Fixed query point in canonical coordinates.
	const Gaussian& G;	// Reference to the Gaussian in canonical coordinates.

	/**
	 * Compute the distance from Gaussian surface to a fixed point.
	 * @param [in] m (u,v) parameters to evaluate in distance function.
	 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
	 */
	double _evalDistance( const column_vector& m ) const
	{
		const double u = m( 0 );
		const double v = m( 1 );

		return 0.5 * (SQR( u - P.x ) + SQR( v - P.y ) + SQR( G(u,v) - P.z ));
	}

	/**
	 * Gradient of the distance function.
	 * @param [in] m (u,v) parameters to evaluate in gradient of distance function.
	 * @return grad(D)(u,v).
	 */
	column_vector _evalGradient( const column_vector& m ) const
	{
		const double u = m( 0 );
		const double v = m( 1 );

		// Make a column vector of length 2 to hold the gradient.
		column_vector res( 2 );

		// Now, compute the gradient vector.
		double Q = G(u,v);
		double Qu = G.dQdu( u, v, Q );			// Sending cached Q too.
		double Qv = G.dQdv( u, v, Q );
		res( 0 ) = (u - P.x) + (Q - P.z) * Qu; 	// dD/du.
		res( 1 ) = (v - P.y) + (Q - P.z) * Qv; 	// dD/dv.
		return res;
	}

	/**
	 * The Hessian matrix for the distance function.
	 * @param [in] m (u,v) parameters to evaluate in Hessian of distance function.
	 * @return grad(grad(D))(u,v)
	 */
	dlib::matrix<double> _evalHessian ( const column_vector& m ) const
	{
		const double u = m(0);
		const double v = m(1);

		double Q = G( u, v );
		double Qu = G.dQdu( u, v, Q );			// Sending in cached Q too.
		double Qv = G.dQdv( u, v, Q );
		double Quu = G.d2Qdu2( u, v, Q, Qu );
		double Qvv = G.d2Qdv2( u, v, Q, Qv );
		double Quv = G.d2Qdudv( u, v, Q );

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		res( 0, 0 ) = 1 + (Q - P.z) * Quu + SQR( Qu );			// d/du(dD/du).
		res( 1, 0 ) = res( 0, 1 ) = (Q - P.z) * Quv + Qu * Qv;	// d/du(dD/dv) and d/dv(dD/du).
		res( 1, 1 ) = 1 + (Q - P.z) * Qvv + SQR( Qv );			// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Gaussian Monge patch object.
	 */
	GaussianPointDistanceModel( const Point3& p, const Gaussian& g ) : P( p ), G( g ) {}

	/**
	 * Interface for evaluating the Gaussian-point distance function.
	 * @param [in] x The (u,v) parameters to obtain the point on the Gaussian (u,v,Q(u,v)).
	 * @return D(x) = 0.5*norm(Q(x) - P)^2.
	 */
	double operator()( const column_vector& x ) const
	{
		return _evalDistance( x );
	}

	/**
	 * Compute gradient and Hessian of the Gaussian-point distance function.
	 * @note The function name and parameter order shouldn't change as this is the signature that dlib expects.
	 * @param [in] x The (u,v) parameters to calculate the point on the Gaussian (u,v,Q(u,v)).
	 * @param [out] grad Gradient of distance function.
	 * @param [out] H Hessian of distance function.
	 */
	__attribute__((unused))
	void get_derivative_and_hessian( const column_vector& x, column_vector& grad, general_matrix& H ) const
	{
		grad = _evalGradient( x );
		H = _evalHessian( x );
	}
};

/////////////////////////////// Signed distance function to a discretized Gaussian patch ///////////////////////////////

class GaussianLevelSet : public geom::DiscretizedMongePatch
{
private:
	__attribute__((unused)) const double _beta;	// Transformation parameters to vary canonical system w.r.t. world coor-
	const Point3 _axis;							// dinate system.  These include a rotation (unit) axis and angle, and a
	const Point3 _trns;							// translation vector that sets the origin of the canonical system to
												// any point in space.
	const double _c, _s;						// Since we use cosine and sine of beta a lot, let's precompute them.
	const double _one_m_c;						// (1-cos(beta)).

	double _deltaStop;							// Convergence for Newton's method for finding close-to-analytical
												// distance from a fixed point to Gaussian.

	const Gaussian *_gaussian;					// Gaussian surface in canonical coordinates.

	bool _useCache = false;						// Computing distance to triangulated surface is expensive --let's cache
	mutable std::unordered_map<std::string, std::pair<double, Point3>> _cache;		// distances and nearest points.
	mutable std::unordered_map<std::string, Point3> _canonicalCoordsCache;			// Also cache canonical coordinates.

	const double _ru2, _rv2;					// Squared half-axis lengths on the canonical uv plane for the bounding ellipse.
	const double _sdsu2, _sdsv2;				// Squared half-axis lengths on canonical uv plane for signed-distance-computation bounding ellipse.

	/**
	 * In/out test w.r.t. limiting ellipse.
	 * @param [in] u Canonical coordinate along u axis.
	 * @param [in] v Canonical coordinate along v axis.
	 * @return True if on or inside ellipse, false otherwise.
	 */
	bool _inOutTest( const double& u, const double& v ) const
	{
		return SQR( u ) / _ru2 + SQR( v ) / _rv2 <= 1;
	}

public:
	typedef dlib::matrix<double,0,1> column_vector;

	/**
	 * Constructor.
	 * @param [in] trans Translation vector.
	 * @param [in] rotAxis Rotation axis (must be nonzero).
	 * @param [in] rotAngle Rotation angle about rotAxis.
	 * @param [in] ku Number of min cells in u half direction to define a symmetric domain.
	 * @param [in] kv Number of min cells in v half direction to define a symmetric domain.
	 * @param [in] L Number of refinement levels per unit length (so that h=2^{-L} is a power of two).
	 * @param [in] gaussian Gaussian object in canonical coordinates.
	 * @param [in] ru2 Squared half-axis length on the u direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] rv2 Squared half-axis length on the v direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] btKLeaf (Optional) maximum number of points in balltree leaf nodes.
	 */
	GaussianLevelSet( const Point3& trans, const Point3& rotAxis, const double& rotAngle,
					  const size_t& ku, const size_t& kv, const size_t& L, const Gaussian *gaussian,
					  const double& ru2, const double& rv2, const size_t& btKLeaf=40 )
					  : _trns( trans ), _axis( rotAxis.normalize() ), _beta( rotAngle), _gaussian( gaussian ),
					  _c( cos( rotAngle ) ), _s( sin( rotAngle ) ), _one_m_c( 1 - cos( rotAngle ) ),
					  _ru2( ABS( ru2 ) ), _rv2( ABS( rv2 ) ), DiscretizedMongePatch( ku, kv, L, gaussian, btKLeaf, ru2, rv2 ),
					  _sdsu2( SQR( sqrt( _ru2 ) + 3 * _h ) ), _sdsv2( SQR( sqrt( _rv2 ) + 3 * _h ) )		// Notice the padding.
	{
		const std::string errorPrefix = "[CASL_ERROR] ParaboloidLevelSet::constructor: ";
		if( rotAxis.norm_L2() < EPS )		// Singular rotation axis?
			throw std::runtime_error( errorPrefix + "Rotation axis shouldn't be 0!" );
		_deltaStop = 1e-8 * _h;

		if( _ru2 == DBL_MAX || _rv2 == DBL_MAX )
			throw std::runtime_error( errorPrefix + "Squared half-axes must be smaller than DBL_MAX!" );
	}

	/**
	 * Transform a point/vector in world coordinates to canonical coordinates using the tranformation info.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @param [in] isVector True if input is a vector (unnaffected by translation), false if input is a point.
	 * @return The coordinates of (x,y,z) in the representation of the Gaussian canonical coordinate system.
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
	 * Get signed distance to discretized Gaussian (triangulated and with vertices structured into a balltree).
	 * @note You can speed up the process by caching grid point distance to the surface iff these map to integer-based
	 * coordinates.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return phi(x,y,z).
	 * @throws runtime error if signed distance doesn't match from using the triangle's normal and the Gaussian surface.
	 */
	double operator()( double x, double y, double z ) const override
	{
		std::string coords;
		if( _useCache )		// Use this only if you know that the coordinates normalized by h yield integers!
		{
			coords = getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			if( record != _cache.end() )
				return (record->second).first;
		}

		Point3 p = toCanonicalCoordinates( x, y, z );		// Transform query point to canonical coordinates.
		double d = DBL_MAX;
		const geom::Triangle *nearestTriangle;
		Point3 nearestPoint = findNearestPoint( p, d, nearestTriangle ); // Compute shortest distance.

		// Fix sign: points above Gaussian are negative and below are positive.  Because of the way we created the
		// triangles, their normal vector points up in the canonical coord. system (into the negative region Omega-).
		if( p.z > 0 )		// Save time, points above xy plane can have negative distance, but not those below.
		{
			Point3 w = p - *(nearestTriangle->getVertex(0));
			if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )			// In the direction of the normal?
				d *= -1;
		}

		// Let's handle distances for points beyond limiting ellipse or closer to boundary triangles.
		if( !_inOutTest( p.x, p.y ) || nearestTriangle->getRelPosType() == geom::RelPosType::BOUNDARY )
		{
			Point3 nearestPoint2( p.x, p.y, (*_gaussian)( p.x, p.y ) );	// Let's try this new point.
			double d2 = (p - nearestPoint2).norm_L2();
			if( d2 < ABS( d ) )		// When we are far from the sampling region, the distance to the Gaussian falls
			{						// back to the point's (canonical) z-coordinate.  Note the sign: above Gaussian
				d = d2 * (p.z > nearestPoint2.z? -1 : +1);				// is Omega^- and below is Omega^+.
				nearestPoint = nearestPoint2;
			}
		}

		if( _useCache )
		{
#pragma omp critical (update_gaussian_cache)
			{
				_cache[coords] = std::make_pair( d, nearestPoint );		// Cache shortest distance and nearest point.
				_canonicalCoordsCache[coords] = p;						// Cache canonical coords too.
			}
		}
		return d;
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
	 * Compute exact signed distance to Gaussian using Newton's method and trust region in dlib for points whose projec-
	 * tions fall within the limiting ellipse enlarged by 3h in the u and v directions.
	 * @param [in] x Query x-coordinate.
	 * @param [in] y Query y-coordinate.
	 * @param [in] z Query z-coordinate.
	 * @param [out] updated Set to true if exact distance was computed, false otherwise.
	 * @return Shortest distance.
	 * @throws runtime error if not using cache, if point wasn't located in cache, or if exact distance deviates by more
	 * 		   than 0.15h from linear estimation.
	 */
	double computeExactSignedDistance( double x, double y, double z, bool& updated ) const
	{
		if( _useCache )		// Use this only if you know that the coordinates normalized by h yield integers!
		{
			std::string coords = getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			if( record != _cache.end() )
			{
				double initialTrustRadius = MAX( _h, ABS( (record->second).first ) );
				column_vector initialPoint = {(record->second).second.x, (record->second).second.y};	// Initial (u,v) come from cached closest point.

				auto ccRecord = _canonicalCoordsCache.find( coords );
				if( ccRecord != _canonicalCoordsCache.end() )
				{
					// Let's compute exact distances only for points within the limiting ellipse (enlaged by a few H).
					Point3 p = ccRecord->second;
					updated = false;
					if( SQR( p.x ) / _sdsu2 + SQR( p.y ) / _sdsv2 <= 1 )
					{
						dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),			// Append .be_verbose() for debugging.
													 GaussianPointDistanceModel( p, *_gaussian ), initialPoint, initialTrustRadius );

						// Check if minimization produced a better d* distance from q to the Gaussian.
						Point3 q( initialPoint(0), initialPoint(1), (*_gaussian)( initialPoint(0), initialPoint(1) ) );
						double refSign = (p.z >= (*_gaussian)( p.x, p.y ))? -1 : 1;		// Exact sign for the distance to Gaussian.
						double dist = (p - q).norm_L2();

						if( dist >= ABS( record->second.first ) )		// If d* is smaller, it's fine.  The problem is if d*>d.
						{
							if( refSign * record->second.first < 0 )	// If both signs agree, it's fine.  If not, let's check.
							{
								if( ABS( dist - ABS( record->second.first ) ) > 0.15 * _h )
									throw std::runtime_error( "[CASL_ERROR] GaussianLevelSet::computeExactSignedDistance: "
															  "Distances disagree in sign and magnitude by more than 0.15h!" );
							}
						}

						record->second.first = refSign * dist;	// Update shortest distance and closest point on Gaussian by
						record->second.second = q;				// fixing the sign too (if needed).
						updated = true;
					}
					return record->second.first;
				}
				else
					throw std::runtime_error( "[CASL_ERROR] GaussianLevelSet::computeExactSignedDistance: Can't locate point in canonical coords cache!" );
			}
			else
				throw std::runtime_error( "[CASL_ERROR] GaussianLevelSet::computeExactSignedDistance: Can't locate point in cache!" );
		}
		else
			throw std::runtime_error( "[CASL_ERROR] GaussianLevelSet::computeExactSignedDistance: Method works only with cache enabled and nonempty!" );
	}

	/**
	 * @see computeExactSignedDistance( double x, double y, double z )
	 * @param [in] xyz Query point in world coordinates.
	 * @param [out] updated Set to true if exact distance was computed, false otherwise.
	 * @return Shortest distance to Gaussian.
	 */
	double computeExactSignedDistance( const double xyz[P4EST_DIM], bool& updated ) const
	{
		return computeExactSignedDistance( xyz[0], xyz[1], xyz[2], updated );
	}

	/**
	 * Evaluate Gaussian level-set function and compute "exact" signed distances for points whose projections lie within
	 * a 3h-enlarged limiting ellipse in the canonical coordinate system.
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes structure.
	 * @param [out] phi Parallel PETSc vector where to place (linearly approximated) level-set values and exact distances.
	 * @param [out] exactFlag Parallel PETSc vector to indicate if exact distance was computed (1) or not (0).
	 * @throws runtime_error if phi vector is null.
	 */
	void evaluate( const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec exactFlag )
	{
		if( !phi )
			throw std::runtime_error( "[CASL_ERROR] GaussianLevelSet::evaluate: Phi vector can't be null!" );

		double *phiPtr, *exactFlagPtr;
		CHKERRXX( VecGetArray( phi, &phiPtr ) );
		CHKERRXX( VecGetArray( exactFlag, &exactFlagPtr ) );

		std::vector<p4est_locidx_t> nodesForExactDist;
		nodesForExactDist.reserve( nodes->num_owned_indeps );

		const double H = _h;
		auto gls = [this](const double& x, const double& y, const double& z){
			return (*this)( x, y, z );
		};

		auto gdist = [this]( const double xyz[P4EST_DIM], bool& updated ){
			return (*this).computeExactSignedDistance( xyz, updated );
		};

//#pragma omp parallel for default( none ) shared( nodes, p4est, phiPtr, gls, H, nodesForExactDist )
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			phiPtr[n] = gls( xyz[0], xyz[1], xyz[2] );	// Retrieves (or sets) the value from the cache.

			// Points we are interested in lie within 3h away from Gamma (at least based on distance calculated from the triangulation).
			if( ABS( phiPtr[n] ) <= 3 * H )
			{
#pragma omp critical
				nodesForExactDist.emplace_back( n );
			}
		}

#pragma omp parallel for default( none ) num_threads( 4 ) \
		shared( nodes, p4est, nodesForExactDist, phiPtr, exactFlagPtr, std::cerr, gdist )
		for( int i = 0; i < nodesForExactDist.size(); i++ )		// NOLINT.  It can't be a range-based loop.
		{
			p4est_locidx_t n = nodesForExactDist[i];
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			try
			{
				bool updated;	// True if exact dist was computed for node within limit ellipse (enlarged by 3h in each dir).
				phiPtr[n] = gdist( xyz, updated );				// Also modifies the cache.
				exactFlagPtr[n] = updated;
			}
			catch( const std::exception &e )
			{
				std::cerr << e.what() << std::endl;
			}
		}

		CHKERRXX( VecRestoreArray( phi, &phiPtr ) );
		CHKERRXX( VecRestoreArray( exactFlag, &exactFlagPtr ) );

		// Synchronize exact distances to other ranks.
		CHKERRXX( VecGhostUpdateBegin( phi, INSERT_VALUES, SCATTER_FORWARD ) );
		CHKERRXX( VecGhostUpdateEnd( phi, INSERT_VALUES, SCATTER_FORWARD ) );

		CHKERRXX( VecGhostUpdateBegin( exactFlag, INSERT_VALUES, SCATTER_FORWARD ) );
		CHKERRXX( VecGhostUpdateEnd( exactFlag, INSERT_VALUES, SCATTER_FORWARD ) );
	}

	void generateSamples( const unsigned char& NumSamPerH2 )
	{
		const double STD_U = sqrt( _ru2 ) / 2;		// Use standard deviations to aim for 95% of data inside bounding ellipse.
		const double STD_V = sqrt( _rv2 ) / 2;
		const size_t N_SAMPLES = (size_t)round( M_PI * sqrt( _ru2 * _rv2 ) / SQR( _h ) ) * NumSamPerH2;	// We'll get *at most* these samples.

		// TODO: Pending...
	}

	/**
	 * Dump triangles into a data file for debugging/visualizing.
	 * @param [in] filename Output file.
	 * @throws Runtime error if file can't be opened.
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
	 * @param n
	 */
	void reserveCache( size_t n )
	{
		_cache.reserve( n );
		_canonicalCoordsCache.reserve( n );
	}
};

#endif //ML_CURVATURE_GAUSSIAN_3D_H
