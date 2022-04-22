/**
 * A collection of classes and functions related to a Gaussian surface embedded in 3d.
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: April 21, 2022.
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

/////////////////////////////////////////////////// Gaussian surface in canonical space ////////////////////////////////////////////////////

/**
 * A Gaussian in canonical space modeling Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)), where a is its height and su^2 and sv^2 are the var-
 * iances.  By definition, our Gaussian has zero means mu and mv.
 */
class Gaussian : public geom::MongeFunction
{
private:
	const double _a;			// Height.
	const double _su2, _sv2;	// Variances.
	const double _su, _sv;		// Standard deviations.

public:
	/**
	 * Constructor.
	 * @param [in] a Gaussian's height.
	 * @param [in] su2 Variance along the u direction.
	 * @param [in] sv2 Variance along the v direction.
	 * @throws invalid_argument error if any of the shape parameters are zero.
	 */
	Gaussian( const double& a, const double& su2, const double& sv2 )
			: _a( ABS(a) ), _su2( ABS( su2 ) ), _sv2( ABS( sv2 ) ), _su( sqrt( _su2 ) ), _sv( sqrt( _sv2 ) )
	{
		if( a <= 0 || su2 <= 0 || sv2 <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Gaussian::constructor: a, su, and sv must be positive!" );
	}

	/**
	 * Evaluate Q(u,v).
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)).
	 */
	double operator()( double u, double v ) const override
	{
		return _a * exp( -0.5 * (SQR( u ) / _su2 + SQR( v ) / _sv2) );
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
	 * Compute mean curvature.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return 2H, where H is the mean curvature at (u,v,Q(u,v)) on the surface.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v, Q );		// Sending in cached values to avoid duplicate evaluations.
		double Qv = dQdv( u, v, Q );
		double Quu = d2Qdu2( u, v, Q, Qu );
		double Qvv = d2Qdv2( u, v, Q, Qv );
		double Quv = d2Qdudv( u, v, Q );
		double kappa = ((1 + SQR(Qv)) * Quu - 2 * Qu * Qv * Quv + (1 + SQR(Qu)) * Qvv)
					   / (2.0 * pow( 1 + SQR(Qu) + SQR(Qv), 1.5 ));
		return kappa;
	}

	/**
	 * Compute Gaussian curvature.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return
	 */
	double gaussianCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v, Q );		// Sending in cached values to avoid duplicate evaluations.
		double Qv = dQdv( u, v, Q );
		double Quu = d2Qdu2( u, v, Q, Qu );
		double Qvv = d2Qdv2( u, v, Q, Qv );
		double Quv = d2Qdudv( u, v, Q );
		return (Quu * Qvv - SQR(Quv)) / SQR(1 + SQR(Qu) + SQR(Qv));
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
	 * Retrieve the Gaussian's height.
	 * @return _a.
	 */
	double a() const
	{
		return _a;
	}

	/**
	 * Find the axis value in the Gaussian's canonical coordinate system where mean curvature becomes 0 using Newton-Raphson's method.
	 * @param [in] h Mesh size.
	 * @param [in] dir Either 0 for u or 1 for v.
	 * @return The p value where kappa(p,0) or kappa(0,p) is zero (depending on the chosen direction).
	 * @throws invalid_argument if dir is not 0 or 1.
	 * 		   runtime_error if bracketing or Newton-Raphson's method fails to find the root.
	 */
	double findKappaZero( const double& h, const u_char& dir ) const
	{
		using namespace boost::math::tools;
		const std::string errorPrefix = "[CASL_ERROR] findKappaZero: ";

		if( dir != 0 && dir != 1 )
			throw std::invalid_argument( errorPrefix + "Wrong direction!  Choose either 0 for u or 1 for v." );

		// Define parameters depending on direction: 0 for u, 1 for v.
		double s2 = (dir == 0? _su2 : _sv2);
		double t2 = (dir == 0? _sv2 : _su2);

		// The function f(p) = 0, where the roots of f are the places where mean curvature is 0 with one of the u or v direction set to 0.
		auto f = [this, &dir, &s2, &t2]( const double& p ){
			double q = (dir == 0? (*this)(p, 0) : (*this)(0, p));
			return SQR( q * p ) + SQR( s2 ) + s2 * t2 - SQR( p ) * t2;
		};

		// And this is the simplified expression to compute both f(p) and its derivative.
		auto fAndDf = [this, &f, &dir, &s2, &t2]( const double& p ){
			double k =  f( p );
			double q = (dir == 0? (*this)(p, 0) : (*this)(0, p));
			double dk = 2 * p * (SQR( q ) * (1 - SQR( p ) / s2) - t2);
			return std::make_pair( k, dk );
		};

		const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
		int getDigits = static_cast<int>( digits * 0.75 );		// Accuracy doubles with each step in Newton-Raphson's, so stop when we have
																// just over half the digits correct.
		boost::uintmax_t it = 0;
		const boost::uintmax_t MAX_IT = 10;						// Maximum number of iterations for bracketing and Newton-Raphson's.

		double s = (dir == 0? _su : _sv);
		double start = h;										// Determine the initial bracket with a sliding a window.
		double end = 2 * s;										// We need to find an interval with different f() signs in its endpoints.
		while( it < MAX_IT && f( start ) * f( end ) > 0 )
		{
			end += 0.5 * s;
			start += 0.5 * s;
			it++;
		}

		if( f( start ) * f( end ) > 0 )
			throw std::runtime_error( errorPrefix + "Failed to find a reliable bracket for " + std::to_string( dir ) + " direction!" );

		double root = (start + end) / 2;	// Initial guess.
		it = MAX_IT;						// Find zero with Newton-Raphson's.
		root = newton_raphson_iterate( fAndDf, root, start, end, getDigits, it );

		if( it >= MAX_IT )
			throw std::runtime_error( errorPrefix + "Couldn't find zero with Newton-Raphson's method for " + std::to_string( dir ) + " direction!" );

		return root;
	}
};

/////////////////////////////////////////////// Distance from P to Gaussian as a model class ///////////////////////////////////////////////

/**
 * A function model for the distance function to the Gaussian surface.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) = a*exp(-0.5*(u^su^2+v^2/sv^2)) for positive a, su, and sv, and P
 * is the query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
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
		const double u = m(0);
		const double v = m(1);

		return 0.5 * (SQR( u - P.x ) + SQR( v - P.y ) + SQR( G(u,v) - P.z ));
	}

	/**
	 * Gradient of the distance function.
	 * @param [in] m (u,v) parameters to evaluate in gradient of distance function.
	 * @return grad(D)(u,v).
	 */
	column_vector _evalGradient( const column_vector& m ) const
	{
		const double u = m(0);
		const double v = m(1);

		// Make a column vector of length 2 to hold the gradient.
		column_vector res( 2 );

		// Now, compute the gradient vector.
		double Q = G(u, v);
		double Qu = G.dQdu( u, v, Q );			// Sending cached Q too.
		double Qv = G.dQdv( u, v, Q );
		res(0) = (u - P.x) + (Q - P.z) * Qu; 	// dD/du.
		res(1) = (v - P.y) + (Q - P.z) * Qv; 	// dD/dv.
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

		double Q = G(u, v);
		double Qu = G.dQdu( u, v, Q );			// Sending in cached Q too.
		double Qv = G.dQdv( u, v, Q );
		double Quu = G.d2Qdu2( u, v, Q, Qu );
		double Qvv = G.d2Qdv2( u, v, Q, Qv );
		double Quv = G.d2Qdudv( u, v, Q );

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		res(0, 0) = 1 + (Q - P.z) * Quu + SQR( Qu );		// d/du(dD/du).
		res(1, 0) = res(0, 1) = (Q - P.z) * Quv + Qu * Qv;	// d/du(dD/dv) and d/dv(dD/du).
		res(1, 1) = 1 + (Q - P.z) * Qvv + SQR( Qv );		// d/dv(dD/dv).
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
	 * @param [in] x The (u,v) parameters to obtain the point on the Gaussian Monge patch (u,v, Q(u,v)).
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

///////////////////////////////////// Signed distance function to a triangulated Gaussian Monge patch //////////////////////////////////////

class GaussianLevelSet : public geom::DiscretizedLevelSet
{
private:
	double _deltaStop;							// Convergence for Newton's method for finding close-to-analytical distance from a query
												// point P to the Gaussian surface.
	const mpi_environment_t *_mpi;				// MPI environment.

	const double _ru2, _rv2;					// Squared half-axis lengths on the canonical uv plane for the bounding ellipse.
	const double _sdsu2, _sdsv2;				// Squared half-axis lengths on canonical uv plane for signed-distance-computation bounding ellipse.

	const std::string _errorPrefix = "[CASL_ERROR] GaussianLevelSet::";

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
	 * @param [in] btKLeaf Maximum number of points in balltree leaves.
	 */
	GaussianLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
					  const size_t& ku, const size_t& kv, const size_t& L, const Gaussian *gaussian,
					  const double& ru2, const double& rv2, const size_t& btKLeaf=40 )
					  : _mpi( mpi ), DiscretizedLevelSet( trans, rotAxis, rotAngle, ku, kv, L, gaussian, ru2, rv2, btKLeaf ),
					  _ru2( ABS( ru2 ) ), _rv2( ABS( rv2 ) ),
					  _sdsu2( SQR( sqrt( _ru2 ) + 4 * _h ) ), _sdsv2( SQR( sqrt( _rv2 ) + 4 * _h ) )	// Notice the padding.
	{
		const std::string errorPrefix = _errorPrefix + "constructor: ";
		_deltaStop = 1e-8 * _h;

		if( _ru2 == DBL_MAX || _rv2 == DBL_MAX )
			throw std::runtime_error( errorPrefix + "Squared half-axes must be smaller than DBL_MAX!" );
	}

	/**
	 * Get signed distance to discretized Gaussian (triangulated and with vertices structured into a balltree).
	 * @note We speed up the process by caching grid point distance to the surface as long as these map to integer-based coordinates.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return phi(x,y,z).
	 * @throws runtime_error if signed distance doesn't match from using the triangle's normal and the Gaussian surface.
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

		// Fix sign: points above Gaussian are negative and below are positive.  Because of the way we created the triangles, their normal
		// vector points up in the canonical coords system (into the negative region Omega-).
		if( p.z > 0 )		// Save time, points above the xy plane can have negative distance, but not those below.
		{
			Point3 w = p - *(nearestTriangle->getVertex(0));
			if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )			// In the direction of the normal?
				d *= -1;
		}

		// Let's handle distances for points beyond limiting ellipse or closer to boundary triangles.
		if( !_inOutTest( p.x, p.y ) || nearestTriangle->getRelPosType() == geom::RelPosType::BOUNDARY )
		{
			Point3 nearestPoint2( p.x, p.y, (*_mongeFunction)( p.x, p.y ) );	// Let's try this new point.
			double d2 = (p - nearestPoint2).norm_L2();
			if( d2 < ABS( d ) )		// When we are far from the sampling region, the distance to the Gaussian falls back to the point's
			{						// (canonical) z-coordinate.  Note the sign: above Gaussian is Omega^- and below is Omega^+.
				d = d2 * (p.z >= nearestPoint2.z? -1 : +1);
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
	 * Compute exact signed distance to Gaussian using Newton's method and trust region in dlib for points whose projections fall within the
	 * limiting ellipse enlarged by 3h in the u and v directions.
	 * @param [in] x Query x-coordinate.
	 * @param [in] y Query y-coordinate.
	 * @param [in] z Query z-coordinate.
	 * @param [out] updated Set to 1 if exact distance was computed, 2 otherwise.
	 * @return Shortest distance.
	 * @throws runtime_error if not using cache, if point wasn't located in cache, or if exact distance deviates by more than 0.15h from linear estimation.
	 */
	double computeExactSignedDistance( double x, double y, double z, unsigned char& updated ) const override
	{
		const std::string errorPrefix = _errorPrefix + "computeExactSignedDistance: ";
		if( _useCache )		// Use this only if you know that the coordinates normalized by h yield integers!
		{
			std::string coords = getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			if( record != _cache.end() )
			{
				double initialTrustRadius = MAX( _h, ABS( (record->second).first ) );
				column_vector initialPoint = {(record->second).second.x, (record->second).second.y};	// Initial (u,v) comes from cached closest point.

				auto ccRecord = _canonicalCoordsCache.find( coords );
				if( ccRecord != _canonicalCoordsCache.end() )
				{
					// Let's compute exact distances only for points within the limiting ellipse (enlaged by a few h).
					Point3 p = ccRecord->second;
					updated = 2;
					if( SQR( p.x ) / _sdsu2 + SQR( p.y ) / _sdsv2 <= 1 )
					{
						dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),			// Append .be_verbose() for debugging.
													 GaussianPointDistanceModel( p, *((Gaussian*)_mongeFunction)), initialPoint, initialTrustRadius );

						// Check if minimization produced a better d* distance from query point p to the Gaussian.
						Point3 q( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) );
						double refSign = (p.z >= (*_mongeFunction)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to Gaussian.
						double dist = (p - q).norm_L2();

						if( dist >= ABS( record->second.first ) )		// If d* is smaller, it's fine.  The problem is if d*>d.
						{
							if( refSign * record->second.first < 0 )	// If both signs agree, it's fine.  If not, let's check.
							{
								if( ABS( dist - ABS( record->second.first ) ) > 0.15 * _h )
									throw std::runtime_error( errorPrefix + "Distances disagree in sign and magnitude by more than 0.15h!" );
							}
						}

						record->second.first = refSign * dist;	// Update shortest distance and closest point on Gaussian by fixing the sign
						record->second.second = q;				// too (if needed).
						updated = 1;
					}
					return record->second.first;
				}
				else
					throw std::runtime_error( errorPrefix + "Can't locate point in canonical coords cache!" );
			}
			else
				throw std::runtime_error( errorPrefix + "Can't locate point in cache!" );
		}
		else
			throw std::runtime_error( errorPrefix + "This method works only with cache enabled and nonempty!" );
	}

	/**
	 * Evaluate Gaussian level-set function and compute "exact" signed distances for points whose projections lie within a 3h-enlarged
	 * limiting ellipse in the canonical coordinate system.  We select these points by checking if their linear-reconstruction distance is
	 * less than 3h*sqrt(3) (i.e., within a shell around Gamma).
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes structure.
	 * @param [out] phi Parallel PETSc vector where to place (linearly approximated) level-set values and exact distances.
	 * @param [out] exactFlag Parallel PETSc vector to indicate if exact distance was computed (1) or not (0).
	 * @throws invalid_argument if phi vector is null.
	 */
	void evaluate( const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec exactFlag )
	{
		const double linearShellHalfWidth = 3 * _h * sqrt( 3 );

		if( !phi )
			throw std::invalid_argument( _errorPrefix + "evaluate: Phi vector can't be null!" );

		double *phiPtr, *exactFlagPtr;
		CHKERRXX( VecGetArray( phi, &phiPtr ) );
		CHKERRXX( VecGetArray( exactFlag, &exactFlagPtr ) );

		std::vector<p4est_locidx_t> nodesForExactDist;
		nodesForExactDist.reserve( nodes->num_owned_indeps );

		auto gdist = [this]( const double xyz[P4EST_DIM], u_char& updated ){
			return (*this).geom::DiscretizedLevelSet::computeExactSignedDistance( xyz, updated );
		};

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			phiPtr[n] = (*this)( xyz[0], xyz[1], xyz[2] );	// Retrieves (or sets) the value from the cache.

			// Points we are interested in lie close to Gamma (at least based on distance calculated from the triangulation).
			if( ABS( phiPtr[n] ) <= linearShellHalfWidth )
				nodesForExactDist.emplace_back( n );
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
				u_char updated;									// 1 if exact dist was computed for node within limiting ellipse.
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
	 * Collect samples within the limiting ellipse and safely away from the walls.
	 * @note Samples are not normalized in any way: not negative curvature or reoriented to first octant
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octMaxRL Effective octree maximum level of refinement (octree must have a length that is a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] trackedMaxErrors Max errors in dimensionless mean and Gaussian curvatures and phi for sampled nodes (reduced across processes).
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples.
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples.
	 * @param [out] samples Array of collected samples.
	 * @param [out] nNumericalSaddles Number of numerical saddles detected from sampled nodes (reduced across processes).
	 * @param [in] exactFlag Vector of flags where we computed exact signed distance functions (so that we can use those nodes to create samples).
	 * @param [out] sampledFlag Parallel vector with 1s for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [out] hkError Vector to hold absolute mean hk error for sampled nodes.
	 * @param [out] ihk Vector to hold linearly interpolated hk for sampled nodes.
	 * @param [out] h2kgError Vector to hold absolute Gaussian h^2*k error for sampled nodes.
	 * @param [out] ih2kg Vector to hold linearly interpolated Gaussian h^2*k for sampled nodes.
	 * @param [out] phiError Vector to hold phi error for sampled nodes.
	 * @param [in] ru2 Override limiting ellipse u semiaxis squared just for sampling.
	 * @param [in] rv2 Override limiting ellipse v semiaxis squared just for sampling.
	 * @return Maximum error in dimensionless curvature (reduced across processes).
	 * @throws invalid_argument exception if phi or exactFlag vector is null, or if overriding ru2 and rv2 are non positive or larger than _ru2 and _rv2.
	 * @throws runtime_error if the cache is disabled or empty, or if a candidate interface point is not in the cache.
	 */
	void collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi,
						 const u_char& octMaxRL, const double xyzMin[P4EST_DIM], const double xyzMax[P4EST_DIM],
						 double trackedMaxErrors[P4EST_DIM], double& trackedMinHK, double& trackedMaxHK,
						 std::vector<std::vector<double>>& samples, int& nNumericalSaddles, const Vec& exactFlag, Vec sampledFlag=nullptr,
						 Vec hkError=nullptr, Vec ihk=nullptr, Vec h2kgError=nullptr, Vec ih2kg=nullptr, Vec phiError=nullptr,
						 double ru2=NAN, double rv2=NAN ) const
	{
		std::string errorPrefix = _errorPrefix + "collectSamples: ";
		nNumericalSaddles = 0;

		if( !phi || !exactFlag )
			throw std::invalid_argument( errorPrefix + "phi and exactFlag vectors can't be null!" );

		if( !_useCache || _cache.empty() )
			throw std::runtime_error( errorPrefix + "Please enable the cache and make sure is non empty before collecting samples!" );

		if( !isnan( ru2 ) && (ru2 > _ru2 || ru2 <= 0) )
			throw std::invalid_argument( errorPrefix + "Overriding ru2 can't be greater than " + std::to_string( _ru2 ) + " or non positive" );
		ru2 = (isnan( ru2 )? _ru2 : ru2);

		if( !isnan( rv2 ) && (rv2 > _rv2 || rv2 <= 0) )
			throw std::invalid_argument( errorPrefix + "Overriding rv2 can't be greater than " + std::to_string( _rv2 ) + " or non positive" );
		rv2 = (isnan( rv2 )? _rv2 : rv2);

		// Get indices for locally owned candidate nodes next to Gamma.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)octMaxRL );
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		// Compute normal vectors and mean/Gaussian/principal curvatures.
		Vec normals[P4EST_DIM],	kappaMG[2], kappa12[2];
		for( auto& component : normals )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &component ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[0] ) );	// This is mean curvature, and
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[1] ) );	// this is Gaussian curvature.
		for( auto& pk : kappa12 )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &pk ) );

		compute_normals_and_curvatures( *ngbd, phi, normals, kappaMG[0], kappaMG[1], kappa12 );

		const double *phiReadPtr;								// We need access to phi to project points onto Gamma.
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

		const double *exactFlagReadPtr;							// Need to determine which nodes we can use for sampling.
		CHKERRXX( VecGetArrayRead( exactFlag, &exactFlagReadPtr ) );

		const double *normalsReadPtr[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );

		samples.clear();										// We'll get (possibly) as many as points there are next to Gamma
		samples.reserve( indices.size() );

		// Reset interface flag vector if given.
		double *sampledFlagPtr;
		if( sampledFlag )
		{
			CHKERRXX( VecGetArray( sampledFlag, &sampledFlagPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				sampledFlagPtr[n] = 0;
		}

		// Reset the mean curvature error vector if given.
		double *hkErrorPtr = nullptr;
		if( hkError )
		{
			CHKERRXX( VecGetArray( hkError, &hkErrorPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				hkErrorPtr[n] = 0;
		}

		// Reset mean ihk vector if given.
		double *ihkPtr = nullptr;
		if( ihk  )
		{
			CHKERRXX( VecGetArray( ihk, &ihkPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				ihkPtr[n] = 0;
		}

		// Reset the Gaussian curvature error vector if given.
		double *h2kgErrorPtr = nullptr;
		if( h2kgError )
		{
			CHKERRXX( VecGetArray( h2kgError, &h2kgErrorPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				h2kgErrorPtr[n] = 0;
		}

		// Reset the Gaussian ih2kg vector if given.
		double *ih2kgPtr;
		if( ih2kg )
		{
			CHKERRXX( VecGetArray( ih2kg, &ih2kgPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				ih2kgPtr[n] = 0;
		}

		// Reset the phi error vector if given.
		double *phiErrorPtr;
		if( phiError )
		{
			CHKERRXX( VecGetArray( phiError, &phiErrorPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				phiErrorPtr[n] = 0;
		}

		// Prepare mean and Gaussian curvature interpolation.
		my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
		kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );

		trackedMinHK = DBL_MAX, trackedMaxHK = 0;	// Track the min and max mean |hk*|
		double trackedMaxHKError = 0;				// and Gaussian curvature and phi errors.
		double trackedMaxH2KGError = 0;
		double trackedMaxPhiError = 0;

#ifdef DEBUG
		std::cout << "    Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
#endif

		int invalidNodes = 0;						// Count points we invalidate because they fail to have a uniform or signed-distance-computed stencil.
		for( const auto& n : indices )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * _h || ABS( xyz[0] - xyzMax[0] ) <= 4 * _h ||	// Skip nodes too close to domain boundary.
				ABS( xyz[1] - xyzMin[1] ) <= 4 * _h || ABS( xyz[1] - xyzMax[1] ) <= 4 * _h ||
				ABS( xyz[2] - xyzMin[2] ) <= 4 * _h || ABS( xyz[2] - xyzMax[2] ) <= 4 * _h )
				continue;

			std::string coords = getDiscreteCoords( xyz[0], xyz[1], xyz[2] );
			const auto record = _cache.find( coords );
			const auto recordCCoords = _canonicalCoordsCache.find( coords );
			if( record == _cache.end() && recordCCoords == _canonicalCoordsCache.end() )
			{
				std::stringstream msg;
				msg << errorPrefix << "Couldn't find [" << Point3( xyz ) << "] (or " << coords << ") in the cache!";
				throw std::runtime_error( msg.str() );
			}
			Point3 p = recordCCoords->second;					// Grid point in canonical coordinates.

			if( SQR( p.x ) / ru2 + SQR( p.y ) / rv2 > 1 )		// Skip nodes whose canonical projection falls outside the (possibly over-
				continue;										// riden) limiting ellipse.

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	//Does it have a valid stencil?
				{
					invalidNodes++;
					continue;
				}

				for( auto& s : stencil )						// Invalidate point if at least one stencil node is not exact-distance-flagged.
				{
					if( exactFlagReadPtr[s] != 1 )
						throw std::runtime_error( "Caught invalid point for " + std::to_string( n ) + "!" );
				}

				// Valid candidate grid node.  Get its exact distance to Gaussian and true curvatures at closest point.
				Point3 nearestPoint = record->second.second;
				double d = record->second.first;
				double hk = _h * _mongeFunction->meanCurvature( nearestPoint.x, nearestPoint.y );
				double h2kg = SQR( _h ) * _mongeFunction->gaussianCurvature( nearestPoint.x, nearestPoint.y );

				for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
					xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
				double kappaMGValues[2];
				kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
				double ihkVal = _h * kappaMGValues[0];
				double ih2kgVal = SQR( _h ) * kappaMGValues[1];

				// Populate sample features.
				samples.emplace_back();
				std::vector<double> *sample = &samples.back();	// Points to new sample in the appropriate array.
				sample->reserve( K_INPUT_SIZE_LEARN );			// phi + normals + hk* + ihk + h2kg* + ih2kg = 112 fields.

				for( const auto& idx : stencil )				// First, phi values.
					sample->push_back( phiReadPtr[idx] );

#ifdef DEBUG
				// Verify that phi(center)'s sign differs with any of its irradiating neighbors.
				if( !NodesAlongInterface::isInterfaceStencil( *sample ) )
					throw std::runtime_error( errorPrefix + "Detected a non-interface stencil!" );
#endif

				for( const auto &component : normalsReadPtr)	// Next, normal components (First x group, then y, then z).
				{
					for( const auto& idx: stencil )
						sample->push_back( component[idx] );
				}

				sample->push_back( hk );						// Then, attach target mean hk* and numerical ihk.
				sample->push_back( ihkVal );
				sample->push_back( h2kg );						// And, attach true Gaussian h^2*kg and ih2kg.
				sample->push_back( ih2kgVal );

				// Update flags: we should expect both non-saddles and saddles.
				if( sampledFlag )
				{
					if( ih2kgVal >= 0 )
						sampledFlagPtr[n] = 1;			// Non-saddle point: 1.
					else
					{
						sampledFlagPtr[n] = 2;			// Saddle point: 2.
						nNumericalSaddles++;
					}
				}

				// Update stats.
				trackedMinHK = MIN( trackedMinHK, ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );
				trackedMaxHK = MAX( trackedMaxHK, ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );

				double errorHK = ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] - (*sample)[K_INPUT_SIZE_LEARN - 3] );
				double errorH2KG = ABS( (*sample)[K_INPUT_SIZE_LEARN - 2] - (*sample)[K_INPUT_SIZE_LEARN - 1] );

				if( hkError )									// Are we also recording the hk and h2kg errors?
					hkErrorPtr[n] = errorHK;
				if( h2kgError )
					h2kgErrorPtr[n] = errorH2KG;
				if( ihk )										// Are we also recording ihk and ih2kg?
					ihkPtr[n] = (*sample)[K_INPUT_SIZE_LEARN - 3];
				if( ih2kg )
					ih2kgPtr[n] = (*sample)[K_INPUT_SIZE_LEARN - 1];
				double errorPhi = ABS( phiReadPtr[n] - d );
				if( phiError )									// What about the phi error for sampled nodes?
					phiErrorPtr[n] = errorPhi;

				trackedMaxPhiError = MAX( trackedMaxPhiError, errorPhi );
				trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
				trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );
			}
			catch( std::runtime_error &rt )
			{
#ifdef DEBUG
				std::cerr << rt.what() << std::endl;
#endif
				invalidNodes++;
			}
		}

#ifdef DEBUG
		std::cout << "    Rank " << _mpi->rank() << " collected " << samples.size() << " *unique* samples and discarded "
				  << invalidNodes << " invalid nodes" << std::endl;
#endif
		kappaMGInterp.clear();

		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMinHK, 1, MPI_DOUBLE, MPI_MIN, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHK, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxH2KGError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxPhiError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nNumericalSaddles, 1, MPI_INT, MPI_SUM, _mpi->comm() ) );

		// Scatter node info across processes.
		if( sampledFlag )
		{
			CHKERRXX( VecRestoreArray( sampledFlag, &sampledFlagPtr ) );

			// Synchronize sampling flag among processes.
			CHKERRXX( VecGhostUpdateBegin( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		if( hkError )
		{
			CHKERRXX( VecRestoreArray( hkError, &hkErrorPtr ) );

			// Synchronize sampling mean hk error among processes.
			CHKERRXX( VecGhostUpdateBegin( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		if( ihk )
		{
			CHKERRXX( VecRestoreArray( ihk, &ihkPtr ) );

			// Synchronize interpolated mean hk among processes.
			CHKERRXX( VecGhostUpdateBegin( ihk, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( ihk, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		if( h2kgError )
		{
			CHKERRXX( VecRestoreArray( h2kgError, &h2kgErrorPtr ) );

			// Synchronize sampling Gaussian h^2*k error among processes.
			CHKERRXX( VecGhostUpdateBegin( h2kgError, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( h2kgError, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		if( ih2kg )
		{
			CHKERRXX( VecRestoreArray( ih2kg, &ih2kgPtr ) );

			// Synchronize interpolated Gaussian h^2*k among processes.
			CHKERRXX( VecGhostUpdateBegin( ih2kg, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( ih2kg, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		if( phiError )
		{
			CHKERRXX( VecRestoreArray( phiError, &phiErrorPtr ) );

			// Synchronize phi error among processes.
			CHKERRXX( VecGhostUpdateBegin( phiError, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( phiError, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		// Clean up.
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecRestoreArrayRead( normals[i], &normalsReadPtr[i] ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( exactFlag, &exactFlagReadPtr ) );

		for( auto& pk : kappa12 )
			CHKERRXX( VecDestroy( pk ) );
		CHKERRXX( VecDestroy( kappaMG[0] ) );
		CHKERRXX( VecDestroy( kappaMG[1] ) );
		for( auto& component : normals )
			CHKERRXX( VecDestroy( component ) );

		trackedMaxErrors[0] = trackedMaxHKError;
		trackedMaxErrors[1] = trackedMaxH2KGError;
		trackedMaxErrors[2] = trackedMaxPhiError;
	}
};

#endif //ML_CURVATURE_GAUSSIAN_3D_H
