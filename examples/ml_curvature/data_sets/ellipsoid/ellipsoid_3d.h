/**
 * A collection of classes and functions related to an ellipsoid in 3D.
 * Developer: Luis Ángel.
 * Created: April 5, 2022.
 * Updated: April 10, 2022.
 */

#ifndef ML_CURVATURE_ELLIPSOID_3D_H
#define ML_CURVATURE_ELLIPSOID_3D_H

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
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <dlib/matrix.h>

/////////////////////////////////////////////////////// Ellipsoid in canonical space ///////////////////////////////////////////////////////

/**
 * An ellipsoidal surface in canonical space modeling phi(x,y,z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1, where a, b, and c are positive shape
 * parameters.
 */
class Ellipsoid : public CF_DIM
{
private:
	const double _a, _b, _c;	// Semi-axes.
	const double _a2, _b2, _c2;	// Squared semi-axes.

	static std::string _errorPrefix() { return "[CASL_ERROR] Ellipsoid::"; }

public:

	/**
	 * Constructor.
	 * @param [in] a Length of the x semi-axis.
	 * @param [in] b Length of the y semi-axis.
	 * @param [in] c Length of the z semi-axis.
	 * @throws invalid_argument exception if any semi-axis length is non-positive.
	 */
	Ellipsoid( const double& a, const double& b, const double& c, const double& tol=FLT_EPSILON )
			: _a( a ), _b( b ), _c( c ), _a2( SQR( _a ) ), _b2( SQR( _b ) ), _c2( SQR( _c ) )
	{
		const std::string errorPrefix = _errorPrefix() + "constructor: ";

		if( a < 0 || b < 0 || c < 0 )
			throw std::invalid_argument( errorPrefix + "a, b, and c must be positive!" );
	}

	/**
	 * Evaluate implicit ellipsoid function.
	 * @param [in] x Value along the x-direction.
	 * @param [in] y Value along the y-direction.
	 * @param [in] z Value along the z-direction.
	 * @return phi(x,y,z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1.
	 */
	double operator()( double x, double y, double z ) const override
	{
		return SQR( x ) / _a2 + SQR( y ) / _b2 + SQR( z ) / _c2 - 1.0;
	}

	/**
	 * Evaluate implicit ellipsoid function.
	 * @param [in] p Query point.
	 * @return phi(p).
	 */
	double operator()( const Point3& p ) const
	{
		return (*this)( p.x, p.y, p.z );
	}

	/**
	 * Get Cartesian coordinates for the parametric angles theta and psi (i.e., longitude and latitude) on the surface.
	 * @note We use here the spherical parametrization of the ellipsoid to ensure we are computing curvatures *at* the surface, exactly:
	 *                      | a * cos(psi) * cos(theta) |
	 *      R(theta, psi) = | b * cos(psi) * sin(theta) |
	 *                      |       c * sin(psi)        |
	 * @see http://www.ma.ic.ac.uk/~rn/distance2ellipse.pdf to verify the above parametrization on (theta, psi), which measures from the
	 * center rather than a pole (https://en.wikipedia.org/wiki/Ellipsoid).
	 * @param [in] theta Longitude.
	 * @param [in] psi Reduced/parametric latitude or eccentric anomaly.
	 * @return Cartesian coordinates.
	 */
	Point3 getXYZFromAngularParams( const double& theta, const double& psi ) const
	{
		Point3 p;
		p.x = _a * cos( psi ) * cos( theta );
		p.y = _b * cos( psi ) * sin( theta );
		p.z = _c * sin ( psi );
		return p;
	}

	/**
	 * Compute mean and Gaussian curvatures.
	 * @param [in] theta Longitude.
	 * @param [in] psi Reduced/parametric latitude or eccentric anomaly.
	 * @param [out] H Mean curvature.
	 * @param [out] K Gaussian curvature.
	 */
	void getCurvatures( const double& theta, const double& psi, double& H, double& K ) const
	{
		Point3 p = getXYZFromAngularParams( theta, psi );
		double x2 = SQR( p.x );
		double y2 = SQR( p.y );
		double z2 = SQR( p.z );
		double x2a4_y2b4_z2c4 = x2 / SQR( _a2 ) + y2 / SQR( _b2 ) + z2 / SQR( _c2 );

		// Mean curvature.
		double num = -(x2/CUBE( _a2 ) + y2/CUBE( _b2 ) + z2/CUBE( _c2 )) + x2a4_y2b4_z2c4 * (1/SQR( _a2 ) + 1/SQR( _b2 ) + 1/SQR( _c2 ));
		double denom = 2 * pow( x2a4_y2b4_z2c4, 1.5 );
		H = num / denom;

		// Gaussian curvature.
		K = 1 / (_a2 * _b2 * _c2 * x2a4_y2b4_z2c4);
	}

	/**
	 * Find the angular parameters (theta, psi) for the nearest point on the surface to a query point p and its shortest distance using New-
	 * ton's method.
	 * @param [in] p Query point (must be in canonical coordinates).
	 * @param [out] theta Longitude.
	 * @param [out] psi Reduced/parametric latitude or eccentric anomaly.
	 * @param [in] tol Convergence tolerance for Newton's method.
	 * @param [in] maxIters Max number of iterations for Newton's root finding.
	 * @return Shortest distance from p to the ellipsoid.
	 * @throws invalid_argument exception if max number of iters is less than 1 or if tolerance is not in the range of (0,1).
	 * 		   runtime_error if Newton's method doesn't converge in the allowed iterations or if new shortest dist is larger than initial guess.
	 */
	double findNearestPointAngularParams( const Point3& p, double& theta, double& psi, const double& tol=1e-8, const u_int& maxIters=20 ) const
	{
		typedef dlib::matrix<double, 2, 1> column_vector_t;
		typedef dlib::matrix<double, 2, 2> matrix_t;

		const std::string errorPrefix = _errorPrefix() + "findNearestPointAngularParams: ";
		if( maxIters < 1 )
			throw std::invalid_argument( errorPrefix + "Max number of iterations must be at least 1." );

		if( tol <= 0 || tol >=1 )
			throw std::invalid_argument( errorPrefix + "Tolerance must be in the range of (0, 1)." );

		// Let's use Newton-Raphson's method to solve a system of non-linear equations (see http://www.ma.ic.ac.uk/~rn/distance2ellipse.pdf).
		column_vector_t xnp1 = {											// Initial guess:
			atan2( _a * p.y, _b * p.x ),									// theta(0)
			atan2( p.z, _c * sqrt( SQR( p.x ) / _a2 + SQR( p.y ) / _b2 ) )	// psi(0)
		};

		// We want to solve for delta = xnp1 - xn in the system DF(xn) * delta = -g(xn) where g(xn) is a column 2-vector and DF(xn) is a
		// 2x2 matrix.
		auto getGAndDF = [this, &p]( const column_vector_t& t, column_vector_t& g, matrix_t & DF ){
			double a = this->_a, a2 = this->_a2;			// Shortcuts.
			double b = this->_b, b2 = this->_b2;
			double c = this->_c, c2 = this->_c2;

			double cosT = cos( t(0) ), cosP = cos( t(1) );
			double sinT = sin( t(0) ), sinP = sin( t(1) );

			// The g(xn) vector.
			g(0) = (a2 - b2) * cosT * sinT * cosP - p.x * a * sinT + p.y * b * cosT;
			g(1) = (a2 * SQR( cosT ) + b2 * SQR( sinT ) - c2) * sinP * cosP - p.x * a * sinP * cosT - p.y * b * sinP * sinT + p.z * c * cosP;

			// And the Jacobian matrix.
			DF(0,0) = (a2 - b2) * (SQR( cosT ) - SQR( sinT )) * cosP - p.x * a * cosT - p.y * b * sinT;
			DF(0,1) = -(a2 - b2) * cosT * sinT * sinP;
			DF(1,0) = -2 * (a2 - b2) * cosT * sinT * sinP * cosP + p.x * a * sinP * sinT - p.y * b * sinP * cosT;
			DF(1,1) = (a2 * SQR( cosT ) + b2 * SQR( sinT ) - c2) * (SQR( cosP ) - SQR( sinP )) - p.x * a * cosP * cosT - p.y * b * cosP * sinT - p.z * c * sinP;
		};

		// Newton's method.
		double d0 = (getXYZFromAngularParams( xnp1(0), xnp1(1) ) - p).norm_L2();	// Distance to initial guess: we must improve this.
		double diffNorm = 1;
		u_int iter = 0;
		while( iter < maxIters && diffNorm > tol )
		{
			column_vector_t xn = xnp1;
			column_vector_t g;
			matrix_t DF;
			getGAndDF( xn, g, DF );						// Evaluate g(xn) and DF(xn).

			dlib::lu_decomposition<matrix_t> lu( DF );	// Use LU decomposition to solve system of equations.
			column_vector_t delta = lu.solve( -g );
			xnp1 = xn + delta;

			diffNorm = dlib::length( delta );
			iter++;
		}

		if( iter >= maxIters )
			throw std::runtime_error( errorPrefix + "Newton's method failed -- reached maximum number of iterations." );

		double d = (getXYZFromAngularParams( xnp1(0), xnp1(1) ) - p).norm_L2();		// Distance to nearest point as determined by Newton's method.
		if( d > d0 )
			throw std::runtime_error( errorPrefix + "Distance at nearest point is larger than distance at the initial guess." );

		return SIGN( operator()( p ) ) * d;
	}

	/**
	 * Compute the pairs (a, b) of semi-axes on the x and y directions that produce desired mean curvatures k_a and k_b while keeping c (the
	 * z-intercept) constant.  All computations consider the query values are in the representation of the ellipsoid's canonica frame.
	 * To find (a, b), we solve for these simultaneously in the equations:
	 * 		k_a = a*(b^2 + c^2)/(2*b^2*c^2)
	 * 		k_b = b*(a^2 + c^2)/(2*a^2*c^2).
	 * To this end, first we solve for y[=b] as the roots of the quartic polynomial:
	 * 							(4*k_a^2*c^2+1)*y^4 - (8*k_b*k_a^2*c^2)*y^3 + (2*c^2)y^2 + c^4 = 0
	 * by finding the real eigenvalues of the equivalent monic polynomial's companion matrix.  Then, we compute the corresponding x[=a] with
	 * 											x = (2*k_a*y^2*c^2)/(y^2+c^2).
	 * @param [in] k_a Desired (positive) mean curvature at the point (a, 0, 0).
	 * @param [in] k_b Desired (positive) mean curvature at the point (0, b, 0).
	 * @param [in] c Ellipsoid's (positive) z-direction semiaxis.
	 * @param [out] abTuples Vector of tuples (a, b) that meet the curvature and c-value conditions (to be clear to allocate results).
	 * @throws invalid_argument exception if ka or kb or c are no sufficiently positive.
	 * 		   runtime_error if it fails to find any tuples.
	 */
	static void findA_BParamsForDesiredKappaOnX_Y( const double& k_a, const double& k_b, const double& c,
												   std::vector<std::pair<double, double>>& abTuples )
	{
		typedef dlib::matrix<double, 4, 4> matrix_t;
		typedef dlib::matrix<double, 4, 4> column_vector_t;

		std::string errorPrefix = _errorPrefix() + "findA_BParamsForDesiredKappaOnX_Y: ";
		if( k_a <= FLT_EPSILON || k_b <= FLT_EPSILON )
			throw std::invalid_argument( errorPrefix + "Desired mean curvatures at (a,0,0) and (0,b,0) must be strictly positive!" );

		if( c <= FLT_EPSILON )
			throw std::invalid_argument( errorPrefix + "z-intercept (i.e., c) must be strictly positive!" );

		// First, define the companion matrix for the quartic monic polynomial:
		// p(y) = [c^4/(4*k_a^2*c^2+1)] + [  0 ]*y + [2*c^2/(4*k_a^2*c^2+1)]*y^2 + [-8*k_b*k_a^2*c^2/(4*k_a^2*c^2+1)]*y^3 + y^4
		//        <-------- m0 ------->   < m1 >     <--------- m2 -------->       <--------------- m3 ------------->
		const double denom = SQR( 2.0 * k_a * c ) + 1;
		const double m0 = SQR(SQR( c )) / denom;
		const double m1 = 0;
		const double m2 = 2.0 * SQR( c ) / denom;
		const double m3 = -8.0 * k_b * SQR( k_a * c );

		matrix_t C = {		// Polynomial's companion matrix whose roots of its characterictic polynomial are the roots of p(y) above.
			0, 0, 0, -m0,
			1, 0, 0, -m1,
			0, 1, 0, -m2,
			0, 0, 1, -m3
		};

		// Dlib's eigendecomposition: http://dlib.net/dlib/matrix/matrix_la_abstract.h.html#eigenvalue_decomposition.  We're interested only
		// on real eigenvalues.
		dlib::eigenvalue_decomposition<matrix_t> eigenDecomposition( C );
		const column_vector_t& evalsRealPart = eigenDecomposition.get_real_eigenvalues();
		const column_vector_t& evalsImagPart = eigenDecomposition.get_imag_eigenvalues();

		abTuples.clear();
		abTuples.reserve( eigenDecomposition.dim() );
		for( int i = 0; i < eigenDecomposition.dim(); i++ )
		{
			if( evalsImagPart(i) > EPS )		// Skip complex eigenvalues.
				continue;

			double y = evalsRealPart(i);
			double x = 2.0 * k_a * SQR( y * c ) / (SQR( y ) + SQR( c ));
			abTuples.emplace_back( x, y );
		}

		// Check everything went OK.
		// TODO: Check for negative a or b params?
		if( abTuples.empty() )
			throw std::runtime_error( errorPrefix + "Failed to find any tuple (a,b) for desired curvatures!" );
	}
};



////////////////////////////////////////// Level-set function for an affine-transformed ellipsoid //////////////////////////////////////////

class EllipsoidalLevelSet : public geom::AffineTransformedSpace, public CF_DIM
{
private:
	const Ellipsoid *_ellipsoid;			// Ellipsoid surface in its canonical coordinate system.
	const mpi_environment_t *_mpi;			// MPI environment.
	const double _h;						// Mesh size (finest cell width).

	const std::string _errorPrefix = "[CASL_ERROR] EllipsoidalLevelSet::";

public:
	/**
	 * Constructor.
	 * @param [in] mpi MPI environment.
	 * @param [in] trans Translation vector.
	 * @param [in] rotAxis Rotation axis (must be nonzero).
	 * @param [in] rotAngle Rotation angle about rotAxis.
	 * @param [in] ellipsoid Ellipsoidal object in canonical coordinates.
	 * @param [in] h Mesh size.
	 * @throws invalid_argument exception if mesh size is non positive.
	 */
	EllipsoidalLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
						 const Ellipsoid *ellipsoid, const double& h )
						 : _mpi( mpi ), AffineTransformedSpace( trans, rotAxis, rotAngle ), _h( h ), _ellipsoid( ellipsoid )
	{
		const std::string errorPrefix = _errorPrefix + "constructor: ";
		if( h <= 0 )						// Invalid mesh size?
			throw std::invalid_argument( errorPrefix + "Invalid mesh size!" );
	}

	/**
	 * Evaluate affine-transformed ellipsoidal level-set function.
	 * @param [in] x x-coordinate in the world frame.
	 * @param [in] y y-coordinate in the world frame.
	 * @param [in] z z-coordinate in the world frame.
	 */
	double operator()( double x, double y, double z ) const override
	{
		Point3 p = toCanonicalCoordinates( x, y, z );	// Transform query point to canonical coords to perform search.
		return (*_ellipsoid)( p );
	}

	/**
	 * Compute exact signed distance to affine-transformed ellipsoidal surface and retrieve the true mean and Gaussian curvatures evaluated
	 * at the nearest point.
	 * @param [in] xyz Query point in world coordinates.
	 * @param [out] H Mean curvature.
	 * @param [out] K Gaussian curvature.
	 * @return Exact signed distance.
	 */
	double computeExactSignedDistanceAndCurvatures( const double xyz[P4EST_DIM], double& H, double& K ) const
	{
		double theta, psi;		// The output angular parameters.
		Point3 p = toCanonicalCoordinates( xyz );
		double d = _ellipsoid->findNearestPointAngularParams( p, theta, psi );
		_ellipsoid->getCurvatures( theta, psi, H, K );
		return d;
	}

	/**
	 * Collect samples for valid grid points next to the interface
	 * @note Samples are not normalized in any way: not negative-mean-curvature nor gradient-reoriented to first octant.
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples.
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples.
	 * @param [out] samples Array of collected samples (one per interface node).
	 * @param [out] nNumericalSaddles Number of numerical saddles detected from sampled nodes (reduced across processes).
	 * @param [out] sampledFlag Parallel vector with 1s for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [out] hkError Vector to hold absolute mean hk error for sampled nodes.
	 * @param [out] ihk Vector to hold linearly interpolated hk for sampled nodes.
	 * @param [out] h2kgError Vector to hold absolute Gaussian h^2*k error for sampled nodes.
	 * @param [out] ih2kg Vector to hold linearly interpolated Gaussian h^2*k for sampled nodes.
	 * @param [out] phiError Vector to hold phi error for sampled nodes.
	 * @return Maximum errors in dimensionless mean and Gaussian curvatures (reduced across processes).
	 * @throws invalid_argument exception if phi vector.
	 */
	std::pair<double,double> collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
											 const Vec& phi, const u_char& octMaxRL, const double xyzMin[P4EST_DIM],
											 const double xyzMax[P4EST_DIM], double& trackedMinHK, double& trackedMaxHK,
											 std::vector<std::vector<double>>& samples, int& nNumericalSaddles, Vec sampledFlag=nullptr,
											 Vec hkError=nullptr, Vec ihk=nullptr, Vec h2kgError=nullptr, Vec ih2kg=nullptr,
											 Vec phiError=nullptr ) const
	{
		std::string errorPrefix = _errorPrefix + "collectSamples: ";
		nNumericalSaddles = 0;

		if( !phi )
			throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

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

		std::uniform_real_distribution<double> pDistribution;
		trackedMinHK = DBL_MAX, trackedMaxHK = 0;	// Track the min and max mean |hk*|
		double trackedMaxHKError = 0;				// and Gaussian curvature errors.
		double trackedMaxH2KGError = 0;

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
#endif

		int invalidNodes = 0;						// Let's count how many points we skipped.
		for( const auto& n : indices )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * _h || ABS( xyz[0] - xyzMax[0] ) <= 4 * _h ||	// Skip nodes too close
				ABS( xyz[1] - xyzMin[1] ) <= 4 * _h || ABS( xyz[1] - xyzMax[1] ) <= 4 * _h ||	// to domain boundary.
				ABS( xyz[2] - xyzMin[2] ) <= 4 * _h || ABS( xyz[2] - xyzMax[2] ) <= 4 * _h )
			{
				invalidNodes++;
				continue;
			}

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	//Does it have a valid stencil?
				{
					invalidNodes++;
					continue;
				}

				// Valid candidate grid node.  Get its distance to ellipsoid and true curvatures at closest point.
				double hk, h2kg;
				double d = computeExactSignedDistanceAndCurvatures( xyz, hk, h2kg );	// If this fails, invalidate point.
				hk *= _h; h2kg *= SQR( _h );

				for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
					xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
				double kappaMGValues[2];
				kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
				double ihkVal = _h * kappaMGValues[0];
				double ih2kgVal = SQR( _h ) * kappaMGValues[1];

				// Begin to populate its features.
				std::vector<double> *sample;					// Points to new sample in the appropriate array.
				samples.emplace_back();
				sample = &samples.back();
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

				// Update flags: we should expect only non-saddles, with ihk*hk >= 0.
				if( sampledFlag )
				{
					if( ih2kgVal >= 0 )
					{
						if( ihkVal * hk >= 0 )
							sampledFlagPtr[n] = 1;		// Valid candidate node: not a numerical saddle and hk and ihk have same sign.
						else
							sampledFlagPtr[n] = 2;		// hk and ihk flipped sign in a non-saddle point.
					}
					else
					{
						nNumericalSaddles++;
						if( ihkVal * hk >= 0 )
							sampledFlagPtr[n] = 3;		// Just a saddle point.
						else
							sampledFlagPtr[n] = 4;		// Both a saddle and flipped hk and ihk case.
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
				if( phiError )									// What about the phi error for sampled nodes?
					phiErrorPtr[n] = phiReadPtr[n] - d;

				trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
				trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );
			}
			catch( std::runtime_error &rt )
			{
				invalidNodes++;
			}
		}

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " collected " << samples.size() << " *unique* samples and discarded "
				  << invalidNodes << " invalid nodes" << std::endl;
#endif
		kappaMGInterp.clear();

		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMinHK, 1, MPI_DOUBLE, MPI_MIN, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHK, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxH2KGError, 1, MPI_DOUBLE, MPI_MAX, _mpi->comm() ) );
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

		for( auto& pk : kappa12 )
			CHKERRXX( VecDestroy( pk ) );
		CHKERRXX( VecDestroy( kappaMG[0] ) );
		CHKERRXX( VecDestroy( kappaMG[1] ) );
		for( auto& component : normals )
			CHKERRXX( VecDestroy( component ) );

		return std::make_pair( trackedMaxHKError, trackedMaxH2KGError );
	}
};

#endif // ML_CURVATURE_ELLIPSOID_3D_H
