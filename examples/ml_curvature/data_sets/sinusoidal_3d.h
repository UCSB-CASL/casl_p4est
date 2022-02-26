/**
 * A collection of classes and functions related to a sinusoidal surface in 3D.
 * Developer: Luis Ángel.
 * Created: February 24, 2022.
 * Updated: February 25, 2022.
 */

#ifndef ML_CURVATURE_SINUSOIDAL_3D_H
#define ML_CURVATURE_SINUSOIDAL_3D_H

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

//////////////////////////////////////// Sinusoidal surface in canonical space /////////////////////////////////////////

/**
 * A sinusoidal surface in canonical space modeling Q(u,v) = a*sin(wu*u)*sin(wv*v), where a is the wave amplitude and wu
 * wv are the frequencies in the u and v directions.  For simplicity, assume a, wu, wv > 0.
 */
class Sinusoid : public geom::MongeFunction
{
private:
	const double _a;			// Amplitude.
	const double _wu, _wv;		// Frequencies in u and v (and their squares).
	const double _wu2, _wv2;

public:
	/**
	 * Constructor.
	 * @param [in] a Amplitude.
	 * @param [in] wu Frequency on the u direction.
	 * @param [in] wv Frequency on the v direction.
	 * @throws Runtime error if any of the shape parameters are non-positive.
	 */
	Sinusoid( const double& a, const double& wu, const double& wv )
			: _a( ABS(a) ), _wu( wu ), _wv( wv ), _wu2( SQR( wu ) ), _wv2( SQR( wv ) )
	{
		if( _a < 0 || _wu < 0 || _wv < 0 )
			throw std::runtime_error( "[CASL_ERROR] Sinusoid::constructor: a, wu, and wv must be positive!" );
	}

	/**
	 * Evaluate sinusoid Q(u,v).
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return Q(u,v) = a*sin(wu*u)*sin(wv*v).
	 */
	double operator()( double u, double v ) const override
	{
		return _a * sin( _wu * u ) * sin( _wv * v );
	}

	/**
	 * First derivative w.r.t. u.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return dQdu(u, v).
	 */
	double dQdu( double u, double v ) const
	{
		return _a * _wu * cos( _wu * u ) * sin ( _wv * v );
	}

	/**
	 * First derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return dQdv(u, v).
	 */
	double dQdv( double u, double v ) const
	{
		return _a * _wv * sin( _wu * u ) * cos( _wv * v );
	}

	/**
	 * Second derivative w.r.t. u.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return d2Qdu2(u, v).
	 */
	double d2Qdu2( double u, double v, double Q=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -_wu2 * Q;
	}

	/**
	 * Second derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return d2Qdv2(u, v).
	 */
	double d2Qdv2( double u, double v, double Q=NAN ) const
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -_wv2 * Q;
	}

	/**
	 * Mixed second derivative.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return d2Qdudv(u, v).
	 */
	double d2Qdudv( double u, double v ) const
	{
		return _a * _wu * _wv * cos( _wu * u ) * cos( _wv * v );
	}

	/**
	 * Compute mean curvature (which actually is twice that).
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return 2H, where H is the true mean curvature at (u,v,Q(u,v)) on the surface.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v );
		double Qv = dQdv( u, v );
		double Quu = d2Qdu2( u, v, Q );
		double Qvv = d2Qdv2( u, v, Q );
		double Quv = d2Qdudv( u, v );
		double kappa = ((1 + SQR( Qv )) * Quu - 2 * Qu * Qv * Quv + (1 + SQR( Qu )) * Qvv)
					   / pow( 1 + SQR( Qu ) + SQR( Qv ), 1.5 );
		return kappa;
	}

	/**
	 * Retrieve the frequency on the u direction.
	 * @return _wu.
	 */
	double wu() const
	{
		return _wu;
	}

	/**
	 * Retrieve the frequency on the v direction.
	 * @return _wv.
	 */
	double wv() const
	{
		return _wv;
	}

	/**
	 * Retrieve the squared frequency on the u direction.
	 * @return _wu2.
	 */
	double wu2() const
	{
		return _wu2;
	}

	/**
	 * Retrieve the squared frequency on the v direction.
	 * @return _wv2.
	 */
	double wv2() const
	{
		return _wv2;
	}
};

///////////////////////////////////// Distance from P to sinusoid as a model class /////////////////////////////////////

/**
 * A function model for the distance function to the sinusoid.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) = a*sin(wu*u)*sin(wv*v) for positive a, wu,
 * and wv, and P is the query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
 */
class SinusoidPointDistanceModel
{
public:
	typedef dlib::matrix<double,0,1> column_vector;
	typedef dlib::matrix<double> general_matrix;

private:
	const Point3 P;		// Fixed query point in canonical coordinates.
	const Sinusoid& S;	// Reference to the sinusoidal surface in canonical coordinates.

	/**
	 * Compute the distance from sinusoidal surface to a fixed point.
	 * @param [in] m (u,v) parameters to evaluate in distance function.
	 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
	 */
	double _evalDistance( const column_vector& m ) const
	{
		const double u = m( 0 );
		const double v = m( 1 );

		return 0.5 * (SQR( u - P.x ) + SQR( v - P.y ) + SQR( S(u,v) - P.z ));
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
		double Q = S(u,v);
		double Qu = S.dQdu( u, v );
		double Qv = S.dQdv( u, v );
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

		double Q = S( u, v );
		double Qu = S.dQdu( u, v );
		double Qv = S.dQdv( u, v );
		double Quv = S.d2Qdudv( u, v );

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		res( 0, 0 ) = 1 + SQR( Qu ) - (Q - P.z) * Q * S.wu2();	// d/du(dD/du).
		res( 1, 0 ) = res( 0, 1 ) = (2 * Q - P.z) * Quv;		// d/du(dD/dv) and d/dv(dD/du).
		res( 1, 1 ) = 1 + SQR( Qv ) - (Q - P.z) * Q * S.wv2();	// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] s Sinusoidal Monge patch object.
	 */
	SinusoidPointDistanceModel( const Point3& p, const Sinusoid& s ) : P( p ), S( s ) {}

	/**
	 * Interface for evaluating the sinusoid-point distance function.
	 * @param [in] x The (u,v) parameters to obtain the point on the sinusoid (u,v,Q(u,v)).
	 * @return D(x) = 0.5*norm(Q(x) - P)^2.
	 */
	double operator()( const column_vector& x ) const
	{
		return _evalDistance( x );
	}

	/**
	 * Compute gradient and Hessian of the sinusoid-point distance function.
	 * @note The function name and parameter order shouldn't change as this is the signature that dlib expects.
	 * @param [in] x The (u,v) parameters to calculate the point on the sinusoid, (u,v,Q(u,v)).
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

////////////////////////////// Signed distance function to a discretized sinusoidal patch //////////////////////////////

class SinusoidalLevelSet : public geom::DiscretizedMongePatch
{
private:
	__attribute__((unused)) const double _beta;	// Transformation parameters to vary canonical system w.r.t. world coor-
	const Point3 _axis;							// dinate system.  These include a rotation (unit) axis and angle, and a
	const Point3 _trns;							// translation vector that sets the origin of the canonical system to
												// any point in space.
	const double _c, _s;						// Since we use cosine and sine of beta a lot, let's precompute them.
	const double _one_m_c;						// (1-cos(beta)).

	double _deltaStop;							// Convergence for Newton's method for finding close-to-analytical
												// distance from a fixed point to sinusoidal surface.

	const Sinusoid *_sinusoid;					// Sinusoidal surface in canonical coordinates.
	const mpi_environment_t *_mpi;				// MPI environment.

	bool _useCache = false;						// Computing distance to triangulated surface is expensive --let's cache
	mutable std::unordered_map<std::string, std::pair<double, Point3>> _cache;	// distances and nearest points.  Also,
	mutable std::unordered_map<std::string, Point3> _canonicalCoordsCache;		// cache canonical coordinates for grid points.

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

		// Fix sign: points above sinusoid are negative and below are positive.  Because of the way we created the tri-
		// angles, their normal vectors point up in the canonical coord system (i.e., into the negative region Omega-).
		Point3 w = p - *(nearestTriangle->getVertex(0));
		if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )	// In the direction of the normal?
			d *= -1;

		return nearestPoint;
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
	 * @param [in] sinusoid Sinusoid surface object in canonical coordinates.
	 * @param [in] limR2 Squared limiting radius for triangulation.
	 * @param [in] btKLeaf (Optional) maximum number of points in balltree leaf nodes.
	 */
	SinusoidalLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis,
						const double& rotAngle, const size_t& ku, const size_t& kv, const size_t& L,
						const Sinusoid *sinusoid, const double& limR2, const size_t& btKLeaf=40 )
						: _mpi( mpi ), _trns( trans ), _axis( rotAxis.normalize() ), _beta( rotAngle),
						_sinusoid( sinusoid ), _c( cos( rotAngle ) ), _s( sin( rotAngle ) ),
						_one_m_c( 1 - cos( rotAngle ) ), DiscretizedMongePatch( ku, kv, L, sinusoid, btKLeaf, limR2, limR2 )
	{
		const std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::constructor: ";

		if( !sinusoid )
			throw std::runtime_error( errorPrefix + "Sinusoid surface object can't be null!" );

		if( rotAxis.norm_L2() < EPS )		// Singular rotation axis?
			throw std::runtime_error( errorPrefix + "Rotation axis shouldn't be 0!" );
		_deltaStop = 1e-8 * _h;
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
	 * Get signed distance to discretized sinusoid (triangulated and with vertices structured into a balltree).
	 * @note You can speed up the process by caching grid point distance to the surface iff these map to integer-based
	 * coordinates.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return phi(x,y,z).
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

		Point3 p = toCanonicalCoordinates( x, y, z );					// Transform query point to canonical coords to perform search.
		double d;
		Point3 nearestPoint = _findNearestPointAndSignedDistance( p, d );

		if( _useCache )
		{
#pragma omp critical (update_sinusoidal_cache)
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
	 * Compute exact signed distance to sinusoid using Newton's method and trust region in dlib.
	 * @param [in] x Query x-coordinate.
	 * @param [in] y Query y-coordinate.
	 * @param [in] z Query z-coordinate.
	 * @param [out] updated Set to 1 if exact-distance computation succeeded, 2 otherwise.
	 * @return Shortest distance.
	 * @throws runtime_error if not using cache, if point wasn't located in cache, or if estimated exact distance
	 * 		   deviates by more than 0.15h from linear estimation.
	 */
	double computeExactSignedDistance( double x, double y, double z, unsigned char& updated ) const
	{
		std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::computeExactSignedDistance: ";
		if( _useCache )		// Assume that the coordinates normalized by h yield integers!
		{
			std::string coords = getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			auto ccRecord = _canonicalCoordsCache.find( coords );
			if( record != _cache.end() && ccRecord != _canonicalCoordsCache.end() )
			{
				unsigned char k = 1;				// Tracks number of nearest neighbors to retrieve if tests fails.

				while( true )
				{
					double initialTrustRadius = MAX( _h, ABS( (record->second).first ) );
					column_vector initialPoint = {(record->second).second.x, (record->second).second.y};	// Initial (u,v) come from cached closest point.
					Point3 p = ccRecord->second;		// Canonical-coordinated point.

					dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),			// Append .be_verbose() for debugging.
												 SinusoidPointDistanceModel( p, *_sinusoid ), initialPoint, initialTrustRadius );

					// Check if minimization produced a better d* distance from q to the sinusoid.
					Point3 q( initialPoint(0), initialPoint(1), (*_sinusoid)( initialPoint(0), initialPoint(1) ) );
					double exactSign = (p.z >= (*_sinusoid)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to sinusoid.
					double dist = (p - q).norm_L2();

					if( dist > ABS( record->second.first ) )		// If d* is smaller, it's fine.  The problem is if d*>d.
					{
						if( exactSign * record->second.first >= 0 )	// Same sign?  Let's check reference point r.
						{
							Point3 r( (record->second).second.x,	// This reference point will tell us if we can accept a
									  (record->second).second.y, 	// larger distance than the one computed linearly.
									  (*_sinusoid)((record->second).second.x, (record->second).second.y));
							double dr = (p - r).norm_L2();

							if( dr < ABS( record->second.first ) )	// If reference point on surface is closer to query, q
							{										// should have been closer to surface since the sinusoid
								updated = 2;						// bends/bunches toward p.
								return record->second.first;		// Bad case: optimization failed!
							}
						}
						else	// If signs differ, p lies between a triangle and the surface.
						{
							if( ABS( dist - ABS( record->second.first ) ) > 0.15 * _h ) // Gap between linear approx and
							{															// surface shouldn't be large.
								updated = 2;					// Bad case: optimization failed!
								return record->second.first;
							}
						}
					}

					if( ABS( dist - ABS( record->second.first ) ) > 0.5 * _h ) 	// New distance shouldn't be too far from the linear approx-
					{															// imation anyways.  If so, repeat search with more neighbors.
						k++;
						if( k > 5 )								// Bad case: we shouldn't search indefinitely.
						{
							updated = 2;
							return record->second.first;
						}
						record->second.second = _findNearestPointAndSignedDistance( p, record->second.first, k );
					}
					else
					{
						updated = 1;
						record->second.first = exactSign * dist;	// Update shortest distance and closest point on
						record->second.second = q;					// sinusoid by fixing the sign too (if needed).
						return record->second.first;
					}
				}
			}
			else
				throw std::runtime_error( errorPrefix + "Can't locate point in cache!" );
		}
		else
			throw std::runtime_error( errorPrefix + "Method works only with cache enabled and nonempty!" );
	}

	/**
	 * @see computeExactSignedDistance( double x, double y, double z )
	 * @param [in] xyz Query point in world coordinates.
	 * @param [out] updated Set to 1 if exact-distance computation succeeded, 2 otherwise.
	 * @return Shortest distance to sinusoid.
	 */
	double computeExactSignedDistance( const double xyz[P4EST_DIM], unsigned char& updated ) const
	{
		return computeExactSignedDistance( xyz[0], xyz[1], xyz[2], updated );
	}

	/**
	 * Evaluate sinusoidal level-set function and compute "exact" signed distances for points in a (linearly reconstruc-
	 * ted shell) of half-width 3h around Gamma.
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes structure.
	 * @param [out] phi Parallel PETSc vector where to place (linearly approximated) level-set values and exact distances.
	 * @param [out] exactFlag Parallel PETSc vector to indicate if exact distance was computed (1) or not (0).
	 * @throws runtime_error if phi vector is null.
	 */
	void evaluate( const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec exactFlag )
	{
		if( !phi )
			throw std::runtime_error( "[CASL_ERROR] SinusoidalLevelSet::evaluate: Phi vector can't be null!" );

		double *phiPtr, *exactFlagPtr;
		CHKERRXX( VecGetArray( phi, &phiPtr ) );
		CHKERRXX( VecGetArray( exactFlag, &exactFlagPtr ) );

		std::vector<p4est_locidx_t> nodesForExactDist;
		nodesForExactDist.reserve( nodes->num_owned_indeps );

		const double H = _h;

		auto sdist = [this]( const double xyz[P4EST_DIM], unsigned char& updated ){
			return (*this).computeExactSignedDistance( xyz,updated );
		};

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			phiPtr[n] = (*this)( xyz[0], xyz[1], xyz[2] );	// Retrieves (or sets) the value from the cache.

			// Points we are interested in lie within 3h around Gamma (based on distance calculated from triangles).
			if( ABS( phiPtr[n] ) <= 3 * H )
				nodesForExactDist.emplace_back( n );
		}

#pragma omp parallel for default( none ) shared(nodes, p4est, nodesForExactDist, phiPtr, exactFlagPtr, std::cerr, sdist)
		for( int i = 0; i < nodesForExactDist.size(); i++ )	// NOLINT.  It can't be a range-based loop for OpenMP.
		{
			p4est_locidx_t n = nodesForExactDist[i];
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );

			unsigned char updated;
			phiPtr[n] = sdist( xyz, updated );				// Also modifies the cache.
			exactFlagPtr[n] = updated;
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
	 * Collect samples using an easing-off probability distribution based on curvature.
	 * @note Samples are not normalized in any way: not negative curvature nor reoriented to first octant
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octreeMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] samples Array of collected samples.
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples.
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples.
	 * @param [in,out] genP Random-number generator device to decide whether to take a sample or not.
	 * @param [in] easingOffMaxHK Upper bound max |hk| for easing-off probability.
	 * @param [in] easingOffProbMaxHK Probability for keeping points whose true |hk| is at least easigOffMaxHK.
	 * @param [in] minHK Minimum target |hk|.
	 * @param [in] probMinHK Probability for keeping points whose true |hk| is minHK.
	 * @param [out] sampledFlag Parallel vector with 1s for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [in] samR Sampling radius on the uv plane.
	 * @param [in] filter Filter vector with 1s for nodes we can sample and anything else for non-sampling nodes.
	 * @param [in] hkError Vector to hold |hk| error for sampled nodes.
	 * @return Maximum error in dimensionless curvature (reduced across processes).
	 * @throws runtime_error if more than one node maps to the same discrete coordinates, or if cache is disabled or is
	 * 		   empty, or if we can't locate the nodes' exact nearest points on Gamma (which should be cached too), or if
	 * 		   the probability for lower bound max HK is invalid, or if the sampling radius is less than 1.5h.
	 */
	double collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
						   const Vec& phi, const unsigned char octreeMaxRL, const double xyzMin[P4EST_DIM],
						   const double xyzMax[P4EST_DIM], std::vector<std::vector<double>>& samples,
						   double& trackedMinHK, double& trackedMaxHK, std::mt19937& genP,
						   const double& easingOffMaxHK, const double& easingOffProbMaxHK=1.0,
						   const double& minHK=0.01, const double& probMinHK=0.01, Vec sampledFlag=nullptr,
						   const double& samR=DBL_MAX, const Vec& filter=nullptr, Vec hkError=nullptr ) const
	{
		std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::collectSamples: ";
		if( !_useCache || _cache.empty() || _canonicalCoordsCache.empty() )
			throw std::runtime_error( errorPrefix + "Cache must be enabled and nonempty!" );

		if( !phi )
			throw std::runtime_error( errorPrefix + "Phi vector can't be null!" );

		if( easingOffProbMaxHK <= 0 || easingOffProbMaxHK > 1 )
			throw std::runtime_error( errorPrefix + "Easing-off max HK probability should be in the range of (0, 1]!" );

		if( samR <= 1.5 * _h )
			throw std::runtime_error( errorPrefix + "Sampling radius should be at least 1.5h!" );
		const double SAM_R2 = (samR == DBL_MAX? DBL_MAX : SQR( samR ) );	// Squared sampling radius.

		// Get indices for locally owned candidate nodes next to Gamma.
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
		samples.reserve( indices.size() );						// and within limiting sampling circle.

		// Reset interface flag vector if given.
		double *sampledFlagPtr;
		if( sampledFlag )
		{
			CHKERRXX( VecGetArray( sampledFlag, &sampledFlagPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				sampledFlagPtr[n] = 0;
		}

		// Load the filter vector if given.
		const double *filterReadPtr = nullptr;
		if( filter )
			CHKERRXX( VecGetArrayRead( filter, &filterReadPtr ) );

		// Reset the hk error vector if given.
		double *hkErrorPtr;
		if( hkError )
		{
			CHKERRXX( VecGetArray( hkError, &hkErrorPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				hkErrorPtr[n] = 0;
		}

		// Prepare curvature interpolation.
		my_p4est_interpolation_nodes_t kInterp( ngbd );
		kInterp.set_input( kappa, interpolation_method::linear );

		std::uniform_real_distribution<double> pDistribution;
		int outIdx = 0;											// Keeps track of interpolation (and sample) indices.
		trackedMinHK = DBL_MAX, trackedMaxHK = 0;				// For debugging and binning, track the min and max
		double trackedMaxHKError = 0;							// |hk*| and error.

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;

		std::unordered_set<std::string> coordsSet;	// Validation set that stores coordinates of the form "i,j,k" for
		coordsSet.reserve( indices.size() );		// nodal indices, where i,j,k are h-based integers.
#endif
		std::vector<p4est_locidx_t> outIdxToNodeIdx;			// Allows to know which node outIdx refers to.
		outIdxToNodeIdx.reserve( indices.size() );

		for( const auto& n : indices )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * _h || ABS( xyz[0] - xyzMax[0] ) <= 4 * _h ||	// Skip nodes too close
				ABS( xyz[1] - xyzMin[1] ) <= 4 * _h || ABS( xyz[1] - xyzMax[1] ) <= 4 * _h ||	// to domain boundary.
				ABS( xyz[2] - xyzMin[2] ) <= 4 * _h || ABS( xyz[2] - xyzMax[2] ) <= 4 * _h )
				continue;

			if( filter && filterReadPtr[n] != 1 )				// Skip also nodes marked as non-sampling.
				continue;

			std::string coords = getDiscreteCoords( xyz[0], xyz[1], xyz[2] );
			auto recordCCoords = _canonicalCoordsCache.find( coords );
			if( recordCCoords == _canonicalCoordsCache.end() )	// Canonical coords should be recorded.
				throw std::runtime_error( errorPrefix + "Couldn't locate node in cache!" );
			Point3 p = recordCCoords->second;

			if( SQR( p.x ) + SQR( p.y ) > SAM_R2 )				// First check: Skip nodes whose canonical projection
				continue;										// falls outside the sampling circle.

#ifdef DEBUG
			if( coordsSet.find( coords ) != coordsSet.end() )	// Coords should be unique!
				throw std::runtime_error( errorPrefix + "Non-unique discrete node coords!" );
			coordsSet.insert( coords );
#endif

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Second check: Does it have a valid stencil?
					continue;

				if( filter )
				{
					for( auto s : stencil )
					{
						if( filterReadPtr[s] != 1 )				// Check again we don't include a stencil with some
							throw std::runtime_error( "Skip" );	// invalid nodes: use a controlled exception.
					}
				}

				auto record = _cache.find( coords );			// Use cache created during level-set computation: see
				if( record == _cache.end() )					// see computeExactSignedDistance() function above.
					throw std::invalid_argument( errorPrefix + "Point not found in the cache!" );	// Exception not captured!

				Point3 nearestPoint = record->second.second;
				double hk = _h * _sinusoid->meanCurvature( nearestPoint.x, nearestPoint.y );
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

				for( const auto &component : normalReadPtr)		// Next, normal components (First x group, then y, then z).
				{
					for( const auto& idx: stencil )
						sample.push_back( component[idx] );
				}

				sample.push_back( hk );							// Then, attach target hk*.

				for( int c = 0; c < P4EST_DIM; c++ )			// Finally, find the location where to (linearly) interpolate num hk.
					xyz[c] -= phiReadPtr[n] * normalReadPtr[c][n];
				kInterp.add_point( outIdx, xyz );

				samples.push_back( sample );
				outIdxToNodeIdx.push_back( n );					// Keep track of node idx associated with this out idx.
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
			throw std::runtime_error( errorPrefix + "Mismatch between interpolation queued nodes and num of samples!" );

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
			if( hkError )										// Are we also recording the hk error?
				hkErrorPtr[outIdxToNodeIdx[i]] = error;
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

		if( hkError )
		{
			CHKERRXX( VecRestoreArray( hkError, &hkErrorPtr ) );

			// Synchronize sampling hk error among processes.
			CHKERRXX( VecGhostUpdateBegin( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		}

		// Clean up.
		if( filter )
			CHKERRXX( VecRestoreArrayRead( filter, &filterReadPtr ) );

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

#endif //ML_CURVATURE_SINUSOIDAL_3D_H
