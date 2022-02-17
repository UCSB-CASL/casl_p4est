/**
 * A collection of classes and functions related to a Gaussian surface embedded in 3D.
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: February 16, 2022.
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
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_curvature_ml.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <dlib/optimization.h>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/math/tools/roots.hpp>
#include <random>

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
	__attribute__((unused)) double su2() const
	{
		return _su2;
	}

	/**
	 * Retrieve the variance along the v direction.
	 * @return _sv^2.
	 */
	__attribute__((unused)) double sv2() const
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
	const mpi_environment_t *_mpi;				// MPI environment.

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
	 * @param [in] mpi MPI environment.
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
	GaussianLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
					  const size_t& ku, const size_t& kv, const size_t& L, const Gaussian *gaussian,
					  const double& ru2, const double& rv2, const size_t& btKLeaf=40 )
					  : _mpi( mpi ), _trns( trans ), _axis( rotAxis.normalize() ), _beta( rotAngle), _gaussian( gaussian ),
					  _c( cos( rotAngle ) ), _s( sin( rotAngle ) ), _one_m_c( 1 - cos( rotAngle ) ),
					  _ru2( ABS( ru2 ) ), _rv2( ABS( rv2 ) ), DiscretizedMongePatch( ku, kv, L, gaussian, btKLeaf, ru2, rv2 ),
					  _sdsu2( SQR( sqrt( _ru2 ) + 3 * _h ) ), _sdsv2( SQR( sqrt( _rv2 ) + 3 * _h ) )		// Notice the padding.
	{
		const std::string errorPrefix = "[CASL_ERROR] ParaboloidLevelSet::constructor: ";

		if( !gaussian )
			throw std::runtime_error( errorPrefix + "Gaussian surface object can't be null!" );

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

//#pragma omp parallel for default( none ) shared( nodes, p4est, phiPtr, gls, H, nodesForExactDist )	// Doesn't work with OpenMP.
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

	/**
	 * Collect samples within the limiting ellipse using an easing-off probability distribution based on curvature.
	 * @note Samples are not normalized in any way: not negative curvature or reoriented to first octant
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octreeMaxRL Effective octree maximum level of refinement (octree must have a length that is a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] samples Array of collected samples.
	 * @param [in,out] genP Random-number generator device to decide whether to take a sample or not.
	 * @param [in] easingOffMaxHK Upper bound max hk for easing-off probability.
	 * @param [in] easingOffProbMaxHK Probability for keeping points whose true |hk| is at least easigOffMaxHK.
	 * @param [in] minHK Minimum true hk.
	 * @param [in] probMinHK Probability for keeping points whose true |hk| is minHK.
	 * @param [out] sampledFlag Parallel vector with 1s for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [in] ru2 Override limiting ellipse u semiaxis squared just for sampling.
	 * @param [in] rv2 Override limiting ellipse v semiaxis squared just for sampling.
	 * @return Maximum error in dimensionless curvature (reduced across processes).
	 * @throws runtime_error if more than one node maps to the same discrete coordinates, or if cache is disabled or is
	 * 		   empty, or if we can't locate the nodes' exact nearest points on Gamma (which should be cached too), or if
	 * 		   the probability for lower bound max HK is invalid, or if the overriding ru2 and rv2 are non positive or
	 * 		   larger than the original _ru2 and _rv2.
	 */
	double collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
						   const Vec& phi, const unsigned char octreeMaxRL, const double xyzMin[P4EST_DIM],
						   const double xyzMax[P4EST_DIM], std::vector<std::vector<double>>& samples,
						   std::mt19937& genP, const double& easingOffMaxHK, const double& easingOffProbMaxHK=1.0,
						   const double& minHK=0.01, const double& probMinHK=0.01, Vec sampledFlag=nullptr,
						   double ru2=NAN, double rv2=NAN ) const
	{
		std::string errorPrefix = "[CASL_ERROR] GaussianLevelSet::collectSamples: ";
		if( !_useCache || _cache.empty() || _canonicalCoordsCache.empty() )
			throw std::runtime_error( errorPrefix + "Cache must be enabled and nonempty!" );

		if( !phi )
			throw std::runtime_error( errorPrefix + "Phi vector can't be null!" );

		if( easingOffProbMaxHK <= 0 || easingOffProbMaxHK > 1 )
			throw std::runtime_error( errorPrefix + "Easing-off max HK probability should be in the range of (0, 1]!" );

		if( !isnan( ru2 ) && (ru2 > _ru2 || ru2 <= 0) )
			throw std::runtime_error( errorPrefix + "Overriding ru2 can't be greater than " + std::to_string( _ru2 ) + " or non positive" );
		ru2 = (isnan( ru2 )? _ru2 : ru2);

		if( !isnan( rv2 ) && (rv2 > _rv2 || rv2 <= 0) )
			throw std::runtime_error( errorPrefix + "Overriding rv2 can't be greater than " + std::to_string( _rv2 ) + " or non positive" );
		rv2 = (isnan( rv2 )? _rv2 : rv2);

		// Get indices for candidate locally owned nodes next to Gamma.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)octreeMaxRL );
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		// Compute normal vectors and mean curvature.
		Vec normal[P4EST_DIM], kappa;
		for( auto& component : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &component ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappa ) );

		compute_normals( *ngbd, phi, normal );
		compute_mean_curvature( *ngbd, normal, kappa );

		const double *phiReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

		const double *normalReadPtr[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecGetArrayRead( normal[i], &normalReadPtr[i] ) );

		samples.clear();										// We'll get (possibly) as many as points next to Gamma
		samples.reserve( indices.size() );						// and within (poss. overriden) limiting ellipse.

		// Reset interface flag vector if given.
		double *sampledFlagPtr;
		if( sampledFlag )
		{
			CHKERRXX( VecGetArray( sampledFlag, &sampledFlagPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				sampledFlagPtr[n] = 0;
		}

		// Prepare curvature interpolation.
		my_p4est_interpolation_nodes_t kInterp( ngbd );
		kInterp.set_input( kappa, interpolation_method::linear );

		std::uniform_real_distribution<double> pDistribution;
		int outIdx = 0;											// Keeps track of interpolation (and sample) indices.
		double trackedMinHK = DBL_MAX, trackedMaxHK = 0;		// For debugging, track the min and max |hk*| and error.
		double trackedMaxHKError = 0;

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
#endif

		std::unordered_set<std::string> coordsSet;	// Validation set that stores coordinates of the form "i,j,k" for
		coordsSet.reserve( indices.size() );		// nodal indices, where i,j,k are H-based integers.

		for( const auto& n : indices )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * _h || ABS( xyz[0] - xyzMax[0] ) <= 4 * _h ||	// Skip nodes too close
				ABS( xyz[1] - xyzMin[1] ) <= 4 * _h || ABS( xyz[1] - xyzMax[1] ) <= 4 * _h ||	// to domain boundary.
				ABS( xyz[2] - xyzMin[2] ) <= 4 * _h || ABS( xyz[2] - xyzMax[2] ) <= 4 * _h )
				continue;

			std::string coords = getDiscreteCoords( xyz[0], xyz[1], xyz[2] );
			auto recordCCoords = _canonicalCoordsCache.find( coords );
			if( recordCCoords == _canonicalCoordsCache.end() )	// Canonical coords should be recorded.
				throw std::runtime_error( errorPrefix + "Couldn't locate node in cache!" );
			Point3 p = recordCCoords->second;

			if( SQR( p.x ) / ru2 + SQR( p.y ) / rv2 > 1 )		// First check: Skip nodes whose canonical projection
				continue;										// falls outside the (poss. overriden) limiting ellipse.

			if( coordsSet.find( coords ) != coordsSet.end() )	// Coords should be unique!
				throw std::runtime_error( errorPrefix + "Non-unique discrete node coords!" );
			coordsSet.insert( coords );

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Second check: Does it have a valid stencil?
					continue;

				auto record = _cache.find( coords );			// Use cache created during level-set computation: see
				if( record == _cache.end() )					// see computeExactSignedDistance() function above.
					throw std::invalid_argument( errorPrefix + "Point not found in the cache!" );	// Exception not captured!

				Point3 nearestPoint = record->second.second;
				double hk = _h * _gaussian->meanCurvature( nearestPoint.x, nearestPoint.y );
				if( ABS( hk ) < minHK )							// Third check: Target |hk*| must be >= minHK.
					continue;

				double prob = kml::utils::easingOffProbability( ABS( hk ), minHK, probMinHK, easingOffMaxHK, easingOffProbMaxHK );
				if( pDistribution( genP ) > prob )				// Fourth check: Use an easing-off prob to keep samples.
					continue;

				// Populate sample.
				std::vector<double> sample;
				sample.reserve( K_INPUT_SIZE + 1 );				// phi + normals + hk* + ihk.

				for( const auto& idx : stencil )				// First, phi values.
					sample.push_back( phiReadPtr[idx] );

#ifdef DEBUG
				// Verify that phi(center)'s sign differs with any of its irradiating neighbors.
							if( !NodesAlongInterface::isInterfaceStencil( sample ) )
								throw std::runtime_error( errorPrefix + "Detected a non-interface stencil!" );
#endif

				for( const auto &component : normalReadPtr)		// Next, normal components (in groups: First x, then y, etc.).
				{
					for( const auto& idx: stencil )
						sample.push_back( component[idx] );
				}

				sample.push_back( hk );							// Then, attach target hk*.

				for( int c = 0; c < P4EST_DIM; c++ )			// Finally, arrange for the location where to (linearly) interpolate num hk.
					xyz[c] -= phiReadPtr[n] * normalReadPtr[c][n];
				kInterp.add_point( outIdx, xyz );

				samples.push_back( sample );
				outIdx++;

				// Update flags.
				if( sampledFlag )
					sampledFlagPtr[n] = 1;						// Flag it as (valid) interface node.
			}
			catch( std::runtime_error &rt ) {}
			catch( std::invalid_argument &ia )
			{
				throw std::runtime_error( ia.what() );			// Raise again the exception for points not found in the cache.
			}
		}

		if( outIdx != samples.size() )
			throw std::runtime_error( errorPrefix + "Mismatch between nodes queued for interpolation and number of samples!" );

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " collected " << outIdx << " *unique* samples." << std::endl;
#endif

		// Perform bulk curvature interpolation at sampled nodes' nearest points on Gamma.
		auto *outKappa = new double[outIdx];
		kInterp.interpolate( outKappa );
		kInterp.clear();
		for( int i = 0; i < outIdx; i++ )						// Write ihk in sampled data.
		{
			samples[i].push_back( outKappa[i] * _h );
			trackedMinHK = MIN( trackedMinHK, ABS( samples[i][K_INPUT_SIZE - 1] ) );	// Collect stats.
			trackedMaxHK = MAX( trackedMaxHK, ABS( samples[i][K_INPUT_SIZE - 1] ) );
			double error = ABS( samples[i][K_INPUT_SIZE - 1] - samples[i][K_INPUT_SIZE] );
			trackedMaxHKError = MAX( trackedMaxHKError, error );
		}
		delete [] outKappa;

		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMinHK, 1, MPI_DOUBLE, MPI_MIN, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHK, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );

#ifdef DEBUG	// Printing the errors.
		CHKERRXX( PetscPrintf( _mpi->comm(), "Tracked HK in the range of [%f, %f]\n", trackedMinHK, trackedMaxHK ) );
		CHKERRXX( PetscPrintf( _mpi->comm(), "Tracked MAX HK Error = %f\n", trackedMaxHKError ) );
#endif

		if( sampledFlag )
		{
			CHKERRXX( VecRestoreArray( sampledFlag, &sampledFlagPtr ) );

			// Synchronize sampling flag among processes.
			CHKERRXX( VecGhostUpdateBegin( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		// Clean up.
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecRestoreArrayRead( normal[i], &normalReadPtr[i] ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		CHKERRXX( VecDestroy( kappa ) );
		for( auto& component : normal )
			CHKERRXX( VecDestroy( component ) );

		return trackedMaxHKError;
	}

	/**
	 * Dump triangles into a data file for debugging/visualizing.
	 * @param [in] filename Output file.
	 * @throws Runtime error if file can't be opened.
	 */
	__attribute__((unused)) void dumpTriangles( const std::string& filename )
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

	/**
	 * Retrieve the cache size.
	 * @return _cache (and _canonicalCoordsCache) size.
	 */
	__attribute__((unused)) size_t getCacheSize() const
	{
		return _cache.size();
	}
};

#endif //ML_CURVATURE_GAUSSIAN_3D_H
