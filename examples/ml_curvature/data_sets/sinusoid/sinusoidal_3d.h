/**
 * A collection of classes and functions related to a sinusoidal surface in 3D.
 * Developer: Luis Ángel.
 * Created: February 24, 2022.
 * Updated: May 23, 2022.
 */

#ifndef ML_CURVATURE_SINUSOIDAL_3D_H
#define ML_CURVATURE_SINUSOIDAL_3D_H

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include "../level_set_patch_3d.h"

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
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return dQdu(u, v).
	 */
	double dQdu( const double& u, const double& v, double Q ) const override
	{
		return _a * _wu * cos( _wu * u ) * sin ( _wv * v );
	}

	/**
	 * First derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return dQdv(u, v).
	 */
	double dQdv( const double& u, const double& v, double Q ) const override
	{
		return _a * _wv * sin( _wu * u ) * cos( _wv * v );
	}

	/**
	 * Second derivative w.r.t. u.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @param [in] Qu (Optional) Cached dQdu(u,v).
	 * @return d2Qdu2(u, v).
	 */
	double d2Qdu2( const double& u, const double& v, double Q, double Qu ) const override
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -_wu2 * Q;
	}

	/**
	 * Second derivative w.r.t. v.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @param [in] Qv (Optional) Cached dQdv(u,v).
	 * @return d2Qdv2(u, v).
	 */
	double d2Qdv2( const double& u, const double& v, double Q, double Qv ) const override
	{
		Q = (isnan( Q )? this->operator()( u, v ) : Q);
		return -_wv2 * Q;
	}

	/**
	 * Mixed second derivative.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return d2Qdudv(u, v).
	 */
	double d2Qdudv( const double& u, const double& v, double Q ) const override
	{
		return _a * _wu * _wv * cos( _wu * u ) * cos( _wv * v );
	}

	/**
	 * Compute mean curvature.
	 * @note It used to be twice the mean curvature for compatibility with other library functions, but we now return
	 * the true mean (i.e., 0.5*(k1+k2)) curvature.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return 2H, where H is the true mean curvature at (u,v,Q(u,v)) on the surface.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v, NAN );
		double Qv = dQdv( u, v, NAN );
		double Quu = d2Qdu2( u, v, Q, NAN );
		double Qvv = d2Qdv2( u, v, Q, NAN );
		double Quv = d2Qdudv( u, v, NAN );
		double kappa = ((1 + SQR( Qv )) * Quu - 2 * Qu * Qv * Quv + (1 + SQR( Qu )) * Qvv)
					   / (2 * pow( 1 + SQR( Qu ) + SQR( Qv ), 1.5 ) );
		return kappa;
	}

	/**
	 * Compute the Gaussian curvature.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @return
	 */
	double gaussianCurvature( const double& u, const double& v ) const override
	{
		double Q = this->operator()( u, v );
		double Qu = dQdu( u, v, NAN );
		double Qv = dQdv( u, v, NAN );
		double Quv = d2Qdudv( u, v, NAN );
		return (_wu2 * _wv2 * SQR( Q ) - SQR( Quv )) / SQR( 1 + SQR( Qu ) + SQR( Qv ) );
	}

	/**
	 * Retrieve the amplitude.
	 * @return _a.
	 */
	double A() const
	{
		return _a;
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
};

///////////////////////////////////// Distance from P to sinusoid as a model class /////////////////////////////////////

/**
 * A function model for the distance function to the sinusoid.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) = a*sin(wu*u)*sin(wv*v) for positive a, wu,
 * and wv, and P is the query (but fixed) point.  The goal is to find the closest point on canonical Q(u,v) to P.
 */
class SinusoidPointDistanceModel : public PointDistanceModel
{
private:
	/**
	 * Gradient of the distance function.
	 * @param [in] m (u,v) parameters to evaluate in gradient of distance function.
	 * @return grad(D)(u,v).
	 */
	column_vector _evalGradient( const column_vector& m ) const override
	{
		const double u = m(0);
		const double v = m(1);

		// Make a column vector of length 2 to hold the gradient.
		column_vector res( 2 );

		// Now, compute the gradient vector.
		double Q = F( u, v );
		double Qu = F.dQdu( u, v, NAN );
		double Qv = F.dQdv( u, v, NAN );
		res(0) = (u - P.x) + (Q - P.z) * Qu; 	// dD/du.
		res(1) = (v - P.y) + (Q - P.z) * Qv; 	// dD/dv.
		return res;
	}

	/**
	 * The Hessian matrix for the distance function.
	 * @param [in] m (u,v) parameters to evaluate in Hessian of distance function.
	 * @return grad(grad(D))(u,v)
	 */
	dlib::matrix<double> _evalHessian ( const column_vector& m ) const override
	{
		const double u = m(0);
		const double v = m(1);

		double Q = F( u, v );
		double Qu = F.dQdu( u, v, NAN );
		double Qv = F.dQdv( u, v, NAN );
		double Quv = F.d2Qdudv( u, v, NAN );

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		auto& S = (Sinusoid&)F;
		res(0, 0) = 1 + SQR( Qu ) - (Q - P.z) * Q * S.wu2();	// d/du(dD/du).
		res(1, 0) = res(0, 1) = (2 * Q - P.z) * Quv;			// d/du(dD/dv) and d/dv(dD/du).
		res(1, 1) = 1 + SQR( Qv ) - (Q - P.z) * Q * S.wv2();	// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] s Sinusoidal Monge patch object.
	 */
	SinusoidPointDistanceModel( const Point3& p, const Sinusoid& s ) : PointDistanceModel( p, s ) {}
};

////////////////////////////// Signed distance function to a discretized sinusoidal patch //////////////////////////////

class SinusoidalLevelSet : public geom::DiscretizedLevelSet
{
private:
	double _deltaStop;							// Convergence for Newton's method for finding close-to-analytical
												// distance from a fixed point to sinusoidal surface.

	const mpi_environment_t *_mpi;				// MPI environment.

	const double _samR2;						// Squared sampling radius.
	const double _sdR2;							// Sign-distance computation squared radius.

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
	 * @param [in] samR Sampling radius to exclude points lying outside canonical sphere.
	 * @param [in] btKLeaf Maximum number of points in balltree leaf nodes.
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface.
	 */
	SinusoidalLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
						const size_t& ku, const size_t& kv, const size_t& L, const Sinusoid *sinusoid,
						const double& limR2, const double& samR=DBL_MAX, const size_t& btKLeaf=40, const u_short& addToL=0 )
						: _mpi( mpi ), DiscretizedLevelSet( trans, rotAxis, rotAngle, ku, kv, L, sinusoid, limR2, limR2, btKLeaf, addToL ),
						_samR2( samR == DBL_MAX? samR : SQR( samR ) ),
						_sdR2( samR == DBL_MAX? samR : SQR( ABS( samR ) + 4 * _h ) )
	{
		const std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::constructor: ";
		_deltaStop = 1e-8 * _h;				// Stopping condition for trust-region minimizatin problem.

		if( samR < 1.5 * _h )				// Very small sampling radius?
			throw std::invalid_argument( errorPrefix + "Sampling radius must be at least 1.5h!" );
	}

	/**
	 * Get signed distance to discretized sinusoid (triangulated and with vertices structured into a balltree).
	 * @note To speed up the process, we cache grid point distances to Gamma since vertices map to integer-based coords.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return linearly approximated phi(x,y,z).
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

		Point3 p = toCanonicalCoordinates( x, y, z );	// Transform query point to canonical coords to perform search.
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
	 * Compute exact signed distance to sinusoid using Newton's method and trust region in dlib.
	 * @param [in] x Query x-coordinate.
	 * @param [in] y Query y-coordinate.
	 * @param [in] z Query z-coordinate.
	 * @param [out] updated Set to 1 if exact-distance computation succeeded, 2 otherwise.
	 * @return Shortest distance.
	 * @throws runtime_error if not using cache, if point wasn't located in cache, or if estimated exact distance
	 * 		   deviates by more than 0.15h from linear estimation.
	 */
	double computeExactSignedDistance( double x, double y, double z, unsigned char& updated ) const override
	{
		std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::computeExactSignedDistance: ";
		if( _useCache )		// Assume that the coordinates normalized by h yield integers!
		{
			std::string coords = getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			auto ccRecord = _canonicalCoordsCache.find( coords );
			if( record != _cache.end() && ccRecord != _canonicalCoordsCache.end() )
			{
				const Point3& p = ccRecord->second;		// Canonical-coordinated point.
				double exactSign = (p.z >= (*_mongeFunction)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to sinusoid.
				updated = 0;

				if( SQR( p.x ) + SQR( p.y ) <= _sdR2 )	// Skip points projected outside padded sampling circle on the uv plane.
				{
					unsigned char k = 1;				// Tracks number of nearest neighbors to retrieve if tests fails.

					while( true )
					{
						double initialTrustRadius = MAX( _h, ABS( (record->second).first ) );
						column_vector initialPoint = {(record->second).second.x, (record->second).second.y};	// Initial (u,v) come from cached closest point.

						dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),			// Append .be_verbose() for debugging.
													 SinusoidPointDistanceModel( p, *((Sinusoid*)_mongeFunction) ), initialPoint, initialTrustRadius );

						// Check if minimization produced a better d* distance from q to the sinusoid.
						Point3 q( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) );
						double dist = (p - q).norm_L2();

						if( dist > ABS( record->second.first ) )		// If d* is smaller, it's fine.  The problem is if d*>d.
						{
							if( exactSign * record->second.first >= 0 )	// Same sign?  Let's check reference point r.
							{
								Point3 r( (record->second).second.x,	// This reference point will tell us if we can accept a
										  (record->second).second.y, 	// larger distance than the one computed linearly.
										  (*_mongeFunction)((record->second).second.x, (record->second).second.y));
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
				return record->second.first;
			}
			else
				throw std::runtime_error( errorPrefix + "Can't locate point in cache!" );
		}
		else
			throw std::runtime_error( errorPrefix + "Method works only with a non-empty, enabled cache!" );
	}

	/**
	 * Evaluate sinusoidal level-set function and compute "exact" signed distances for points in a (linearly reconstruc-
	 * ted shell) around Gamma.
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes structure.
	 * @param [out] phi Parallel PETSc vector where to place (linearly approximated) level-set values and exact distances.
	 * @param [out] exactFlag Parallel PETSc vector to indicate if exact distance was computed (1) or not (0).
	 * @throws invalid_argument exception if phi is null.
	 */
	void evaluate( const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec exactFlag )
	{
		if( !phi )
			throw std::invalid_argument( "[CASL_ERROR] SinusoidalLevelSet::evaluate: Phi vector can't be null!" );

		double *phiPtr, *exactFlagPtr;
		CHKERRXX( VecGetArray( phi, &phiPtr ) );
		CHKERRXX( VecGetArray( exactFlag, &exactFlagPtr ) );

		std::vector<p4est_locidx_t> nodesForExactDist;
		nodesForExactDist.reserve( nodes->num_owned_indeps );

		auto sdist = [this]( const double xyz[P4EST_DIM], unsigned char& updated ){
			return (*this).geom::DiscretizedLevelSet::computeExactSignedDistance( xyz, updated );
		};

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			phiPtr[n] = (*this)( xyz[0], xyz[1], xyz[2] );	// Retrieves (or sets) the value from the cache.

			// Points we are interested in lie close to Gamma (based on distance calculated from triangles).
			if( ABS( phiPtr[n] ) <= 2 * _h * sqrt( 3 ) )
				nodesForExactDist.emplace_back( n );
		}

		int i;
#pragma omp parallel for default(none) private(i) shared(nodes, p4est, nodesForExactDist, phiPtr, exactFlagPtr, sdist)
		for( i = 0; i < nodesForExactDist.size(); i++ )	// NOLINT.  It can't be a range-based loop for OpenMP.
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
	 * Collect samples using an easing-off probability distribution based on mean curvature if the Gaussian curvature is
	 * positive.  For (numerical) negative-Gaussian-curvature points, we perform subsampling based on the numerically
	 * interpolated Gaussian curvature at Gamma and place samples in a separate array.
	 * @note Samples are not normalized in any way: not negative-mean-curvature nor gradient-reoriented to first octant.
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octreeMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples (non-saddle pos 0, saddle pos 1).
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples (non-saddle pos 0, saddle pos 1).
	 * @param [in] nonSaddleMinIH2KG Min numerical dimensionless Gaussian curvature (at Gamma) for numerical non-saddle points.
	 * @param [in,out] genP Random-number generator device to decide whether to take a sample or not.
	 * @param [out] nonSaddleSamples Array of collected samples for non-saddle regions (i.e., ih2kg > 0).
	 * @param [in] easingOffMaxHK Upper bound max |hk| for easing-off probability based on mean curvature for non-saddle points.
	 * @param [in] easingOffProbMaxHK Probability for keeping points whose true mean |hk*| is at least easigOffMaxHK.
	 * @param [in] minHK Minimum true mean |hk*| for non-saddle regions.
	 * @param [in] probMinHK Probability for keeping points whose true mean |hk*| is minHK.
	 * @param [out] saddleSamples Array of collected samples for saddle regions (i.e., ih2kg <= 0).
	 * @param [in] easingOffMaxIH2KG Upper bound max |ih2kg| for easing-off probability based on Gaussian curvature for saddle points.
	 * @param [in] easingOffProbMaxIH2KG Probability for keeping points whose |ih2kg| is at least easigOffMaxIH2KG.
	 * @param [in] minIH2KG Minimum (linearly interpolated) numerical |ih2kg| for saddle points.
	 * @param [in] probMinIH2KG Probability for keeping points whose numerical |ih2kg| is minIH2KG.
	 * @param [out] sampledFlag Parallel vector with 1s for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [in] samR Overriding sphere sampling radius on canonical space (to be used instead of the one provided during instantiation).
	 * @param [in] filter Vector with 1s for nodes we are allowed to sample from and anything else for non-sampling nodes.
	 * @param [out] hkError Vector to hold absolute mean hk error for sampled nodes.
	 * @param [out] ihk Vector to hold linearly interpolated hk for sampled nodes.
	 * @param [out] h2kgError Vector to hold absolute Gaussian h^2*k error for sampled nodes.
	 * @param [out] ih2kg Vector to hold linearly interpolated Gaussian h^2*k for sampled nodes.
	 * @return Maximum errors in dimensionless mean and Gaussian curvatures (reduced across processes).
	 * @throws runtime_error or invalid_argument if more than one node maps to the same discrete coordinates, or if
	 * 		   cache is disabled or empty, or if we can't locate the nodes' exact nearest points on Gamma (which should
	 * 		   be cached too), or if the probability for lower bound max HK is invalid, or if the overriding sampling
	 * 		   radius is less than 1.5h or bigger than local sampling radius.
	 */
	std::pair<double,double> collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
											 const Vec& phi, const unsigned char octreeMaxRL, const double xyzMin[P4EST_DIM],
											 const double xyzMax[P4EST_DIM], double trackedMinHK[SAMPLE_TYPES],
											 double trackedMaxHK[SAMPLE_TYPES], std::mt19937& genP, const double& nonSaddleMinIH2KG,
											 std::vector<std::vector<double>>& nonSaddleSamples,
											 const double& easingOffMaxHK, const double& easingOffProbMaxHK,	// These 4 params apply
											 const double& minHK, const double& probMinHK, 						// to non-saddle points.
											 std::vector<std::vector<double>>& saddleSamples,
											 const double& easingOffMaxIH2KG, const double& easingOffProbMaxIH2KG,	// And these 4 apply
											 const double& minIH2KG=0, const double& probMinIH2KG=0.01,				// to saddle points.
											 Vec sampledFlag=nullptr, const double& samR=NAN, const Vec& filter=nullptr,
											 Vec hkError=nullptr, Vec ihk=nullptr, Vec h2kgError=nullptr, Vec ih2kg=nullptr ) const
	{
		std::string errorPrefix = "[CASL_ERROR] SinusoidalLevelSet::collectSamples: ";
		if( !_useCache || _cache.empty() || _canonicalCoordsCache.empty() )
			throw std::runtime_error( errorPrefix + "Cache must be enabled and nonempty!" );

		if( !phi )
			throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

		if( easingOffProbMaxHK <= 0 || easingOffProbMaxHK > 1 )
			throw std::invalid_argument( errorPrefix + "Easing-off max HK probability should be in the range of (0, 1]!" );

		if( !isnan( samR ) && (SQR(samR) > _samR2 || samR < 1.5 * _h) )
			throw std::invalid_argument( errorPrefix + "Overriding sampling radius can't be larger than local sampling radius or less than 1.5h!" );
		const double SAM_R2 = (isnan( samR )? _samR2 : SQR( samR ));	// Squared sampling radius.

		// Get indices for locally owned candidate nodes next to Gamma.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)octreeMaxRL );
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

		const double *normalsReadPtr[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );

		nonSaddleSamples.clear();								// We'll get (possibly) as many as points next to Gamma
		nonSaddleSamples.reserve( indices.size() );				// and within limiting sampling circle.
		saddleSamples.clear();
		saddleSamples.reserve( indices.size() );

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

		// Prepare mean and Gaussian curvatures interpolation.
		my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
		kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );

		std::uniform_real_distribution<double> pDistribution;
		trackedMinHK[0] = DBL_MAX, trackedMinHK[1] = DBL_MAX, trackedMaxHK[0] = 0, trackedMaxHK[1] = 0;	// Track the min and max mean |hk*|
		double trackedMaxHKError = 0;																	// and Gaussian curvature errors.
		double trackedMaxH2KGError = 0;

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;

		std::unordered_set<std::string> coordsSet;	// Validation set that stores coordinates of the form "i,j,k" for
		coordsSet.reserve( indices.size() );		// nodal indices, where i,j,k are h-based integers.
#endif

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

			if( SQR( p.x ) + SQR( p.y ) + SQR( p.z ) > SAM_R2 )	// Skip nodes whose point on the surface lies outside the sampling sphere.
				continue;

#ifdef DEBUG
			if( coordsSet.find( coords ) != coordsSet.end() )	// Coords should be unique!
				throw std::runtime_error( errorPrefix + "Non-unique discrete node coords!" );
			coordsSet.insert( coords );
#endif

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Does it have a valid stencil?
					continue;

				if( filter )
				{
					for( auto s : stencil )
					{
						if( filterReadPtr[s] != 1 )				// Check we don't include a stencil with some invalid nodes.
							throw std::runtime_error( "Skip" );
					}
				}

				auto record = _cache.find( coords );			// Use cache created during level-set computation: see the
				if( record == _cache.end() )					// computeExactSignedDistance(...) function above.
					throw std::invalid_argument( errorPrefix + "Point not found in the cache!" );	// Exception not captured!

				// Starting at this point, we collect samples in two ways:
				// If the Gaussian curvature is < nonSaddleMinIH2KG at Gamma, this is a saddle-region point => subsample based on |ih2kg|.
				// If the Gaussian curvature is >= nonSaddleMinIH2KG at Gamma, this is not a saddle-region point => subsample based on |hk*|.
				Point3 nearestPoint = record->second.second;
				double hk = _h * _mongeFunction->meanCurvature( nearestPoint.x, nearestPoint.y );	// Mean hk at the *exact* projection onto Gamma.
				for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
					xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
				double kappaMGValues[2];
				kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
				double ihkVal = _h * kappaMGValues[0];
				double ih2kgVal = SQR( _h ) * kappaMGValues[1];

				bool isNonSaddle;
				if( ih2kgVal >= nonSaddleMinIH2KG )				// Not a saddle?  Continue filtering based on true mean hk.
				{
					if( ABS( hk ) < minHK )						// Target mean |hk*| must be >= minHK for non-saddle regions.
						continue;

					if( ABS( hk ) <= easingOffMaxHK )
					{
						double prob = kml::utils::easingOffProbability( ABS( hk ), minHK, probMinHK, easingOffMaxHK, easingOffProbMaxHK );
						if( pDistribution( genP ) > prob )		// Use an easing-off prob to keep samples in [minHK, easingOffMaxHK].
							continue;
					}
					else
					{
						double prob = kml::utils::easingOffProbability( ABS( hk ), easingOffMaxHK, easingOffProbMaxHK, 2./3, 0.75 );	// TODO: Set these as parameters.
						if( pDistribution( genP ) > prob )		// Use an easing-off prob to keep samples in [easingOffMaxHK, 2/3].
							continue;
					}

					isNonSaddle = true;							// Flag point as non-saddle.
				}
				else
				{
					if( ABS( ih2kgVal ) < minIH2KG )			// Interpolated |ihk2kg| must be >= minIH2KG for saddle regions.
						continue;

					double prob = kml::utils::easingOffProbability( ABS( ih2kgVal ), minIH2KG, probMinIH2KG, easingOffMaxIH2KG, easingOffProbMaxIH2KG );
					if( pDistribution( genP ) > prob )
						continue;								// Use an easing-off prob to keep samples.

					isNonSaddle = false;
				}

				// Up to this point, we got a good sample.  Populate its features.
				std::vector<double> *sample;					// Points to new sample in the appropriate array.
				if( isNonSaddle )
				{
					nonSaddleSamples.emplace_back();
					sample = &nonSaddleSamples.back();
				}
				else
				{
					saddleSamples.emplace_back();
					sample = &saddleSamples.back();
				}
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
				double h2kg = SQR( _h ) * _mongeFunction->gaussianCurvature( nearestPoint.x, nearestPoint.y );
				sample->push_back( h2kg );						// And, attach true Gaussian h^2*kg and ih2kg.
				sample->push_back( ih2kgVal );

				// Update flags.
				if( sampledFlag )
					sampledFlagPtr[n] = 1;						// Flag it as (valid) interface node.

				// Update stats.
				int which = isNonSaddle? 0 : 1;
				trackedMinHK[which] = MIN( trackedMinHK[which], ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );
				trackedMaxHK[which] = MAX( trackedMaxHK[which], ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );

				double errorHK = ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] - (*sample)[K_INPUT_SIZE_LEARN - 3] );
				double errorH2KG = ABS( (*sample)[K_INPUT_SIZE_LEARN - 2] - (*sample)[K_INPUT_SIZE_LEARN - 1] );

				if( hkError )									// Are we also recording the mean hk and Gaussian h^2*kg errors?
					hkErrorPtr[n] = errorHK;
				if( h2kgError )
					h2kgErrorPtr[n] = errorH2KG;
				if( ihk )										// Are we also recording ihk and ih2kg?
					ihkPtr[n] = (*sample)[K_INPUT_SIZE_LEARN - 3];
				if( ih2kg )
					ih2kgPtr[n] = (*sample)[K_INPUT_SIZE_LEARN - 1];

				trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
				trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );
			}
			catch( std::runtime_error &rt ) {}
			catch( std::invalid_argument &ia )
			{
				throw std::runtime_error( ia.what() );			// Raise again the exception for points not found in the cache.
			}
		}

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " collected " << nonSaddleSamples.size() + saddleSamples.size()
				  << " *unique* samples." << std::endl;
#endif
		kappaMGInterp.clear();

		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, trackedMinHK, SAMPLE_TYPES, MPI_DOUBLE, MPI_MIN, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, trackedMaxHK, SAMPLE_TYPES, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxH2KGError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );

#ifdef DEBUG	// Printing the errors.
		CHKERRXX( PetscPrintf( _mpi->comm(), "Tracked mean hk in the range of [%g, %g] for non-saddles and [%g, %g] for saddles\n",
							   trackedMinHK[0], trackedMaxHK[0], trackedMinHK[1], trackedMaxHK[1] ) );
		CHKERRXX( PetscPrintf( _mpi->comm(), "Tracked max mean hk error = %f\n", trackedMaxHKError ) );
		CHKERRXX( PetscPrintf( _mpi->comm(), "Tracked max Gaussian h^2k error = %f\n", trackedMaxH2KGError ) );
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

		// Clean up.
		if( filter )
			CHKERRXX( VecRestoreArrayRead( filter, &filterReadPtr ) );

		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecRestoreArrayRead( normals[i], &normalsReadPtr[i] ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		for( auto& pk : kappa12 )
			CHKERRXX( VecDestroy( pk ) );
		CHKERRXX( VecDestroy( kappaMG[0] ) );
		CHKERRXX( VecDestroy( kappaMG[1] ) );
		for( auto& component : normals )
			CHKERRXX( VecDestroy( component ) );

		return std::make_pair( trackedMaxHKError, trackedMaxH2KGError );
	}

	/**
	 * Set up the domain based on sinusoid shape parameters to ensure a good portion of the periodic surface resides inside Omega.
	 * @param [in] sinusoid Configured sinusoid function.
	 * @param [in] N_WAVES Desired number of full cycles for any direction.
	 * @param [in] h Mesh size.
	 * @param [in] MAX_A Maximum amplitude.  It's used to avoid unecessarily big domains by limiting the sampling radius.
	 * @param [in] MAX_RL Maximum level of refinement per unit octant (i.e., h = 2^{-MAX_RL}).
	 * @param [out] samRadius Sampling radius on the uv plane that resides fully within the domain.
	 * @param [out] octreeMaxRL Effective individual octree maximum level of refinement to achieve the desired h.
	 * @param [out] uvLim Limiting radius for triangulating sinusoid.
	 * @param [out] halfUV Number of h units (symmetrically) in the u and v direction to define the uv domain.
	 * @param [out] n_xyz Number of octrees in each direction with maximum level of refinement octreeMaxRL.
	 * @param [out] xyz_min Omega minimum dimensions.
	 * @param [out] xyz_max Omega maximum dimensions.
	 */
	static void setupDomain( const Sinusoid& sinusoid, const double& N_WAVES, const double& h, const double& MAX_A, const u_char& MAX_RL,
							 double& samRadius, u_char& octreeMaxRL, double& uvLim, size_t& halfUV, int n_xyz[P4EST_DIM],
							 double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] )
	{
		samRadius = N_WAVES * 2.0 * M_PI * MAX( 1/sinusoid.wu(), 1/sinusoid.wv() );	// Choose the sampling radius based on longer distance that contains N_WAVES full cycles.
		samRadius = MAX( samRadius, sinusoid.A() );					// Prevent the case of a very thin surface: we still want to sample the tips.
		samRadius = 6 * h + MIN( 1.5 * MAX_A, samRadius );			// Then, bound that radius with the largest amplitude.  Add enough padding (for uv plane).

		const double CUBE_SIDE_LEN = 2 * samRadius;					// We want a cubic domain with an effective, yet small size.
		const u_char OCTREE_RL_FOR_LEN = MAX( 0, MAX_RL - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		octreeMaxRL = MAX_RL - OCTREE_RL_FOR_LEN;					// Effective max refinement level to achieve desired h.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );		// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;		// Adjusted domain cube len as a multiple of h and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		const double D_CUBE_DIAG_LEN = sqrt( 3 ) * D_CUBE_SIDE_LEN;	// Use this diag to determine triangulated surface.
		uvLim = D_CUBE_DIAG_LEN / 2 + h;							// Notice the padding to account for the random shift in [-h/2,+h/2]^3.
		halfUV = ceil( uvLim / h );									// Half UV domain in h units.

		// Defining a symmetric cubic domain whose dimensions are multiples of h.
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			n_xyz[i] = N_TREES;
			xyz_min[i] = -HALF_D_CUBE_SIDE_LEN;
			xyz_max[i] = +HALF_D_CUBE_SIDE_LEN;
		}
	}
};

#endif //ML_CURVATURE_SINUSOIDAL_3D_H
