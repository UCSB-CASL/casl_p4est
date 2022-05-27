/**
 * Abstract class for a signed-distance level-set function that we wish to discretize as a Monge patch of the form Q(u,v) with selective
 * exact signed-distance computations for a shell of grid nodes around Gamma.
 *
 * Developer: Luis Ángel.
 * Created: April 25, 2022.
 * Updated: May 27, 2022.
 */

#ifndef ML_CURVATURE_LEVEL_SET_PATCH_3D_H
#define ML_CURVATURE_LEVEL_SET_PATCH_3D_H

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p8est_utils.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_curvature_ml.h>
#include <src/casl_geometry.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <vector>
#include <dlib/optimization.h>

/////////////////////////////////////////////// Distance from P to Gaussian as a model class ///////////////////////////////////////////////

/**
 * A function model for the distance function to a Monge function.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) is our Monge patch, and P is the query (but fixed) point.
 * The goal is to find the closest point on Q(u,v) to P.
 */
class PointDistanceModel
{
public:
	typedef dlib::matrix<double,0,1> column_vector;
	typedef dlib::matrix<double> general_matrix;

protected:
	const Point3 P;					// Fixed query point in canonical coordinates.
	const geom::MongeFunction& F;	// Reference to the Monge function in canonical coordinates.

	/**
	 * Compute the distance from Monge function to a fixed point.
	 * @param [in] m (u,v) parameters to evaluate in distance function.
	 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
	 */
	double _evalDistance( const column_vector& m ) const
	{
		const double u = m(0);
		const double v = m(1);

		return 0.5 * (SQR( u - P.x ) + SQR( v - P.y ) + SQR( F(u, v) - P.z ));
	}

	/**
	 * Gradient of the distance function.
	 * @param [in] m (u,v) parameters to evaluate in gradient of distance function.
	 * @return grad(D)(u,v).
	 */
	virtual column_vector _evalGradient( const column_vector& m ) const = 0;

	/**
	 * The Hessian matrix for the distance function.
	 * @param [in] m (u,v) parameters to evaluate in Hessian of distance function.
	 * @return grad(grad(D))(u,v)
	 */
	virtual dlib::matrix<double> _evalHessian ( const column_vector& m ) const = 0;

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Gaussian Monge patch object.
	 */
	PointDistanceModel( const Point3& p, const geom::MongeFunction& f ) : P( p ), F( f ) {}

	/**
	 * Interface for evaluating the surface-point distance function.
	 * @param [in] x The (u,v) parameters to obtain the point on the Monge patch (u,v, Q(u,v)).
	 * @return D(x) = 0.5*norm(Q(x) - P)^2.
	 */
	double operator()( const column_vector& x ) const
	{
		return _evalDistance( x );
	}

	/**
	 * Compute gradient and Hessian of the surface-point distance function.
	 * @note The function name and parameter order shouldn't change as this is the signature that dlib expects.
	 * @param [in] x The (u,v) parameters to calculate the point on the Monge patch (u,v,Q(u,v)).
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

////////////////////////////////////////// Signed distance function to a triangulated Monge patch //////////////////////////////////////////

class SignedDistanceLevelSet : public geom::DiscretizedLevelSet
{
protected:
	double _deltaStop;					// Convergence for Newton's method for finding close-to-analytical distance from a query
										// point P to the surface.
	const mpi_environment_t *_mpi;		// MPI environment.

	const double _ru2, _rv2;			// Squared half-axis lengths on the canonical uv plane for the bounding ellipse.
	const double _sdsu2, _sdsv2;		// Squared half-axis lengths on canonical uv plane for signed-distance-computation bounding ellipse.

	const std::string _errorPrefix = "[CASL_ERROR] SignedDistanceLevelSet::";

	/**
	 * In/out test w.r.t. an ellipse.
	 * @param [in] u Canonical coordinate along u axis.
	 * @param [in] v Canonical coordinate along v axis.
	 * @param [in] ru2 (Optional) Semi-axis length along the u direction; if not given, we'll use limiting ellipse _ru2.
	 * @param [in] rv2 (Optional) Semi-axis length along the v direction; if not given, we'll use limiting ellipse _rv2.
	 * @return True if on or inside ellipse, false otherwise.
	 */
	bool _inOutTest( const double& u, const double& v, double ru2=NAN, double rv2=NAN ) const
	{
		ru2 = isnan( ru2 )? _ru2 : ru2; assert( ru2 > 0 );
		rv2 = isnan( rv2 )? _rv2 : rv2; assert( rv2 > 0 );
		return SQR( u ) / ru2 + SQR( v ) / rv2 <= 1;
	}

	/**
	 * Fix sign of computed distance to triangulated surface by using the Monge function.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in] d Current POSITIVE distance to the triangulated surface.
	 * @return signed d.
	 */
	virtual double _fixSignOfDistance( const Point3& p, const geom::Triangle *nearestTriangle, const double& d ) const = 0;

	/**
	 * Handle the case of points whose projections onto the uv plane fall outside the limiting ellipse by possibly correcting their signed
	 * distance and currently set nearest point to triangulated surface.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in,out] d Current linearly computed signed distance.
	 * @param [in,out] nearestPoint Current linearly computed nearest point.
	 */
	virtual void _handlePointsBeyondLimitingEllipse( const Point3& p, const geom::Triangle *nearestTriangle, double& d, Point3& nearestPoint ) const = 0;

	/**
	 * Compute exact signed distance to surface using dlib's trust region method.
	 * @param [in] p Query point in canonical coordinates which we have corroborated that lies inside some limiting ellipse.
	 * @param [in,out] d Current linearly computed signed distance and then found to nearest point on Monge patch.
	 * @param [in,out] nearestPoint Current linearly computed nearest point to triangulated surface and then a more accurate version.
	 */
	virtual void _computeExactSignedDistance( const Point3& p, double& d, Point3& nearestPoint ) const = 0;

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
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface with balltree.
	 */
	SignedDistanceLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
							const size_t& ku, const size_t& kv, const u_short& L, const geom::MongeFunction *mongeFunction,
							const double& ru2, const double& rv2, const size_t& btKLeaf=40, const u_short& addToL=0 )
							: DiscretizedLevelSet( trans, rotAxis, rotAngle, ku, kv, L, mongeFunction, ru2, rv2, btKLeaf, addToL ),
							  _mpi( mpi ), _ru2( ABS( ru2 ) ), _rv2( ABS( rv2 ) ),
							  _sdsu2( SQR( sqrt( _ru2 ) + 4 * _h ) ), _sdsv2( SQR( sqrt( _rv2 ) + 4 * _h ) )	// Notice the padding.
	{
		const std::string errorPrefix = _errorPrefix + "constructor: ";
		_deltaStop = 1e-8 * _h;

		if( _ru2 == DBL_MAX || _rv2 == DBL_MAX )
			throw std::invalid_argument( errorPrefix + "Squared half-axes must be smaller than DBL_MAX!" );
	}

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
	 * @param [in] sdsu2 Squared half-axis length on the u direction for signed-distance computation on the canonical uv plane.
	 * @param [in] sdsv2 Squared half-axis length on the v direction for signed-distance computation on the canonical uv plane.
	 * @param [in] btKLeaf Maximum number of points in balltree leaves.
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface with balltree.
	 */
	SignedDistanceLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
							const size_t& ku, const size_t& kv, const u_short& L, const geom::MongeFunction *mongeFunction,
							const double& ru2, const double& rv2, const double& sdsu2, const double& sdsv2, const size_t& btKLeaf=40,
							const u_short& addToL=0 )
		: _mpi( mpi ), DiscretizedLevelSet( trans, rotAxis, rotAngle, ku, kv, L, mongeFunction, ru2, rv2, btKLeaf, addToL ),
		  _ru2( ABS( ru2 ) ), _rv2( ABS( rv2 ) ), _sdsu2( sdsu2 ), _sdsv2( sdsv2 )
	{
		const std::string errorPrefix = _errorPrefix + "constructor: ";
		_deltaStop = 1e-8 * _h;

		if( _ru2 == DBL_MAX || _rv2 == DBL_MAX )
			throw std::invalid_argument( errorPrefix + "Squared half-axes must be smaller than DBL_MAX!" );

		if( _sdsu2 <= SQR( _h ) || _sdsv2 <= SQR( _h ) )
			throw std::invalid_argument( errorPrefix + "Squared half-axes for signed-distance computation must be at least h^2." );
	}

	/**
	 * Get signed distance to discretized patch (triangulated and with vertices structured into a balltree).
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

		Point3 p = toCanonicalCoordinates( x, y, z );						// Transform query point to canonical coordinates.
		double d = DBL_MAX;
		const geom::Triangle *nearestTriangle;
		Point3 nearestPoint = findNearestPoint( p, d, nearestTriangle ); 	// Compute shortest distance.

		d = _fixSignOfDistance( p, nearestTriangle, d );
		_handlePointsBeyondLimitingEllipse( p, nearestTriangle, d, nearestPoint );

		if( _useCache )
		{
#pragma omp critical (update_gaussian_cache)
			{
				_cache[coords] = std::make_pair( d, nearestPoint );			// Cache shortest distance and nearest point.
				_canonicalCoordsCache[coords] = p;							// Cache canonical coords too.
			}
		}
		return d;
	}

	/**
	 * Compute exact signed distance to Gaussian using Newton's method and trust region in dlib for points whose projections fall within the
	 * limiting ellipse enlarged in the u and v directions.
	 * @param [in] x Query x-coordinate.
	 * @param [in] y Query y-coordinate.
	 * @param [in] z Query z-coordinate.
	 * @param [out] updated Set to 1 if exact distance was computed, 2 otherwise.
	 * @return Shortest distance.
	 * @throws runtime_error if not using cache, if point wasn't located in cache, or if exact signed distance computation fails.
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
				auto ccRecord = _canonicalCoordsCache.find( coords );
				if( ccRecord != _canonicalCoordsCache.end() )
				{
					// Let's compute exact distances only for points within the limiting ellipse (enlaged by a few h).
					Point3 p = ccRecord->second;
					updated = 2;
					if( _inOutTest( p.x, p.y, _sdsu2, _sdsv2 ) )
					{
						try
						{
							_computeExactSignedDistance( p, record->second.first, record->second.second );
						}
						catch( std::exception& e )	// Catch any possible exception and regenerate it with additional information.
						{
							throw std::runtime_error( errorPrefix + "Error for grid point " + coords + ": [" + e.what() + "]" );
						}
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
	 * Evaluate Gaussian level-set function and compute "exact" signed distances for points whose projections lie within an enlarged
	 * limiting ellipse in the canonical coordinate system.  We select these points by checking if their linear-reconstruction distance is
	 * less than numMinDiag * h*sqrt(3) (i.e., within a shell around Gamma).
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes structure.
	 * @param [out] phi Parallel PETSc vector where to place (linearly approximated) level-set values and exact distances.
	 * @param [out] exactFlag Parallel PETSc vector to indicate if exact distance was computed (1) or not (0).
	 * @param [in] numMinDiag Number of min diagonals (i.e., h*sqrt(3)) to take as half band width around linear shell for candidate nodes.
	 * @throws invalid_argument if phi vector is null.
	 */
	void evaluate( const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec exactFlag, const double& numMinDiag=3 )
	{
		const double linearShellHalfWidth = numMinDiag * _h * sqrt( 3 );

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
	 * @param [in] nonSaddleMinIH2KG Min numerical dimensionless Gaussian curvature (at Gamma) for numerical non-saddle samples.
	 * @param [in] trueHK Optional vector with true dimensionless mean curvature computed at Gamma for sampled nodes.
	 * @return Maximum error in dimensionless curvature (reduced across processes).
	 * @throws invalid_argument exception if phi or exactFlag vector is null, or if overriding ru2 and rv2 are non positive or larger than _ru2 and _rv2.
	 * @throws runtime_error if the cache is disabled or empty, or if a candidate interface point is not in the cache.
	 */
	void collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi,
						 const u_char& octMaxRL, const double xyzMin[P4EST_DIM], const double xyzMax[P4EST_DIM],
						 double trackedMaxErrors[P4EST_DIM], double& trackedMinHK, double& trackedMaxHK,
						 std::vector<std::vector<double>>& samples, int& nNumericalSaddles, const Vec& exactFlag, Vec sampledFlag=nullptr,
						 Vec hkError=nullptr, Vec ihk=nullptr, Vec h2kgError=nullptr, Vec ih2kg=nullptr, Vec phiError=nullptr,
						 double ru2=NAN, double rv2=NAN, const double& nonSaddleMinIH2KG=-7e-6, Vec trueHK=nullptr ) const
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
		double *ih2kgPtr = nullptr;
		if( ih2kg )
		{
			CHKERRXX( VecGetArray( ih2kg, &ih2kgPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				ih2kgPtr[n] = 0;
		}

		// Reset the phi error vector if given.
		double *phiErrorPtr = nullptr;
		if( phiError )
		{
			CHKERRXX( VecGetArray( phiError, &phiErrorPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				phiErrorPtr[n] = 0;
		}

		// Reset the true mean hk vector if given.
		double *trueHKPtr = nullptr;
		if( trueHK )
		{
			CHKERRXX( VecGetArray( trueHK, &trueHKPtr ) );
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
				trueHKPtr[n] = 0;
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

			if( !_inOutTest( p.x, p.y, ru2, rv2 ) )				// Skip nodes whose canonical projection falls outside the (possibly over-
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
					throw std::invalid_argument( errorPrefix + "Detected a non-interface stencil!" );
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

				// Update flags.
				if( sampledFlag )
				{
					if( ih2kgVal >= nonSaddleMinIH2KG )
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
				if( trueHK )
					trueHKPtr[n] = (*sample)[K_INPUT_SIZE_LEARN - 4];

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
			catch( std::invalid_argument &ia )
			{
				throw std::runtime_error( ia.what() );
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

		if( trueHK )
		{
			CHKERRXX( VecRestoreArray( trueHK, &trueHKPtr ) );

			// Synchronize true hk among processes.
			CHKERRXX( VecGhostUpdateBegin( trueHK, INSERT_VALUES, SCATTER_FORWARD ) );
			CHKERRXX( VecGhostUpdateEnd( trueHK, INSERT_VALUES, SCATTER_FORWARD ) );
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

/////////////////////////////////////////// Utility functions for discretized level-set functions //////////////////////////////////////////

/**
 * Save buffered samples to a file.  Upon exiting, the buffer will be emptied.
 * @param [in] mpi MPI environment.
 * @param [in,out] buffer Sample buffer.
 * @param [in,out] bufferSize Current buffer's size.
 * @param [in,out] file File where to write samples.
 * @return number of saved samples (already shared among processes).
 * @throws invalid_argument exception if buffer is empty.
 */
int saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>>& buffer, int& bufferSize, std::ofstream& file )
{
	int savedSamples = 0;

	if( bufferSize > 0 )
	{
		if( mpi.rank() == 0 )
		{
			int i;
			for( i = 0; i < bufferSize; i++ )
			{
				int j;
				for( j = 0; j < K_INPUT_SIZE_LEARN - 1; j++ )
					file << buffer[i][j] << ",";				// Inner elements.
				file << buffer[i][j] << std::endl;				// Last element is ihk in 2D or ih2kg in 3D.
			}
			savedSamples = i;

			buffer.clear();
		}
		bufferSize = 0;
	}
	else
		throw std::invalid_argument( "saveSamples: buffer is empty or there are less samples than intended number to save(?)!" );

	// Communicate to everyone the total number of saved samples.
	SC_CHECK_MPI( MPI_Bcast( &savedSamples, 1, MPI_INT, 0, mpi.comm() ) );

	return savedSamples;
}

#endif //ML_CURVATURE_LEVEL_SET_PATCH_3D_H
