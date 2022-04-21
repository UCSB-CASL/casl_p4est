/**
 * A collection of classes and functions related to an ellipsoid.
 * Developer: Luis Ángel.
 * Created: April 5, 2022.
 * Updated: April 15, 2022.
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
#include <dlib/optimization.h>

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

		if( a < FLT_EPSILON || b < FLT_EPSILON || c < FLT_EPSILON )
			throw std::invalid_argument( errorPrefix + "a, b, and c must be positive and at least the float32 eps!" );
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
		double num = -(x2/CUBE( _a2 ) + y2/CUBE( _b2 ) + z2/CUBE( _c2 )) + x2a4_y2b4_z2c4 * (1./_a2 + 1./_b2 + 1./_c2);
		double denom = 2 * pow( x2a4_y2b4_z2c4, 1.5 );
		H = num / denom;

		// Gaussian curvature.
		K = 1 / (_a2 * _b2 * _c2 * SQR( x2a4_y2b4_z2c4 ));
	}

	/**
	 * Compute the expected maximum mean curvatures, found at the ellipsoid's x-, y-, and z-intercepts.
	 * @param [out] k Maximum mean curvatures.
	 */
	void getMaxMeanCurvatures( double k[P4EST_DIM] ) const
	{
		double axs[P4EST_DIM] = {_a, _b, _c};
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			int j1 = (i + 1) % P4EST_DIM;	// The formulas are circular.  For x, for example, it's k_x = a (b^2 + c^2)/(2 b^2 c^2).
			int j2 = (i + 2) % P4EST_DIM;
			k[i] = axs[i] * (SQR(axs[j1]) + SQR(axs[j2])) / (2 * SQR(axs[j1] * axs[j2]));
		}
	}

	/**
	 * Accessors.
	 */
	double a()  const { return _a;  }
	double a2() const { return _a2; }
	double b()  const { return _b;  }
	double b2() const { return _b2; }
	double c()  const { return _c;  }
	double c2() const { return _c2; }
};


/////////////////////////////////// Distance from P to parametrized ellipsoid as a model class for dlib ////////////////////////////////////

/**
 * A function model for the distance function to the ellipsoid.  To be used with find_min_trust_region() from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q(u,v) is the parametrized surface in theta (u) and psi (v):
 *
 *                    | a * cos(v) * cos(u) |
 *	        Q(u, v) = | b * cos(v) * sin(u) |, where u in [0, 2*pi) and v in [-pi/2, pi/2],
 *                    |      c * sin(v)     |
 *
 * and P is the query (but fixed) point.  The goal is to find the closest point on canonical Q(u,v) to P.
 */
class EllipsoidPointDistanceModel
{
public:
	typedef dlib::matrix<double,0,1> column_vector;
	typedef dlib::matrix<double> general_matrix;

private:
	const Point3 P;		// Fixed query point in canonical ellipsoid's coordinates.
	const Ellipsoid& Q;	// Reference to the ellipsoidal surface in canonical coordinates.

	/**
	 * Compute the distance from ellipsoid to a fixed point.
	 * @param [in] m (u,v) parameters to evaluate in distance function.
	 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
	 */
	double _evalDistance( const column_vector& m ) const
	{
		const double u = m(0);
		const double v = m(1);

		return 0.5*(SQR(Q.a()*cos(u)*cos(v) - P.x) + SQR(Q.b()*cos(v)*sin(u) - P.y) + SQR(Q.c()*sin(v) - P.z));
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
		res(0) = Q.a()*cos(v)*sin(u)*(P.x - Q.a()*cos(u)*cos(v)) - Q.b()*cos(u)*cos(v)*(P.y - Q.b()*cos(v)*sin(u)); 									// dD/du.
		res(1) = Q.a()*cos(u)*sin(v)*(P.x - Q.a()*cos(u)*cos(v)) + Q.b()*sin(u)*sin(v)*(P.y - Q.b()*cos(v)*sin(u)) - Q.c()*cos(v)*(P.z - Q.c()*sin(v)); // dD/dv.
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

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		res(0, 0) = cos(v)*(Q.a2()*cos(v) - Q.b2()*cos(v) + Q.a()*P.x*cos(u) + Q.b()*P.y*sin(u) - 2*Q.a2()*SQR(cos(u))*cos(v) + 2*Q.b2()*SQR(cos(u))*cos(v));	// d/du(dD/du).
		res(1, 0) = res(0, 1) = sin(v)*(2*cos(u)*cos(v)*sin(u)*Q.a2() - P.x*sin(u)*Q.a() - 2*cos(u)*cos(v)*sin(u)*Q.b2() + P.y*cos(u)*Q.b());					// d/du(dD/dv) and d/dv(dD/du).
		res(1, 1) = Q.c2()*SQR(cos(v)) + Q.c()*sin(v)*(P.z - Q.c()*sin(v)) + Q.a2()*SQR(cos(u)*sin(v)) + Q.b2()*SQR(sin(u)*sin(v)) +
			Q.a()*cos(u)*cos(v)*(P.x - Q.a()*cos(u)*cos(v)) + Q.b()*cos(v)*sin(u)*(P.y - Q.b()*cos(v)*sin(u));													// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Ellipsoid object.
	 */
	EllipsoidPointDistanceModel( const Point3& p, const Ellipsoid& q ) : P( p ), Q( q ) {}

	/**
	 * Interface for evaluating the ellipsoid-point distance function.
	 * @param [in] x The (u,v) parameters to obtain the point on the sinusoid Q(u,v).
	 * @return D(x) = 0.5*norm(Q(x) - P)^2.
	 */
	double operator()( const column_vector& x ) const
	{
		return _evalDistance( x );
	}

	/**
	 * Compute gradient and Hessian of the ellipsoid-point distance function.
	 * @note The function name and parameter order shouldn't change as this is the signature that dlib expects.
	 * @param [in] x The (u,v) parameters to calculate the point on the ellipsoid, Q(u,v).
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


////////////////////////////////////////// Level-set function for an affine-transformed ellipsoid //////////////////////////////////////////

class EllipsoidalLevelSet : public geom::AffineTransformedSpace, public CF_DIM
{
private:
	double _deltaStop;						// Convergence for optimization method for finding close-to-analytical distance from a fixed
											// point to ellipsoid.
	const Ellipsoid *_ellipsoid;			// Ellipsoid surface in its canonical coordinate system.
	const mpi_environment_t *_mpi;			// MPI environment.
	const double _h;						// Mesh size (finest cell width).
	bool _useCache = false;					// Since we compute exact signed distance functions, let's cache responses for grid nodes.
	mutable std::unordered_map<std::string, std::pair<double, Point2>> _cache;  // Maps discrete coords to a pair with signed distance and (theta, psi).

	const std::string _errorPrefix = "[CASL_ERROR] EllipsoidalLevelSet::";

	/**
	 * Retrieve discrete coordinates as a triplet normalized by h.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return "i,j,k".
	 */
	std::string _getDiscreteCoords( const double& x, const double& y, const double& z ) const
	{
		return std::to_string( (int)(x/_h) ) + "," + std::to_string( (int)(y/_h) ) + "," + std::to_string( (int)(z/_h) );
	}

	/**
	 * Compute shortest signed distance and find the angular parameters (theta, psi) for the nearest point on the surface to a query point p
	 * using dlib.
	 * @param [in] p Query point (must be in canonical coordinates).
	 * @param [out] theta Longitude.
	 * @param [out] psi Reduced/parametric latitude or eccentric anomaly.
	 * @param [in] tol Convergence tolerance for Newton's method.
	 * @param [in] maxIters Max number of iterations for Newton's root finding.
	 * @return Shortest distance from p to the ellipsoid.
	 * @throws runtime_error if optimization method fails or if new shortest distance is larger than initial guess.
	 */
	double _computeShortestSignedDistance( const Point3& p, double& theta, double& psi ) const
	{
		const std::string errorPrefix = _errorPrefix + "_findNearestPointAngularParams: ";

		theta = psi = NAN;									// Nothing is settled now.
		double initialTrustRadius = MIN( 1.0, MIN( _ellipsoid->a(), _ellipsoid->b(), _ellipsoid->c() ) );
		column_vector initialPoint = {
			atan2( _ellipsoid->a() * p.y, _ellipsoid->b() * p.x ),													// theta
			atan2( p.z, _ellipsoid->c() * sqrt( SQR( p.x ) / _ellipsoid->a2() + SQR( p.y ) / _ellipsoid->b2() ) )	// psi
		};

		double d0 = (_ellipsoid->getXYZFromAngularParams( initialPoint(0), initialPoint(1) ) - p).norm_L2();	// Distance to initial guess: we must improve this.
		double D = dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),				// Append .be_verbose() for debugging.
									 EllipsoidPointDistanceModel( p, *(_ellipsoid) ), initialPoint, initialTrustRadius );
		double d = sqrt( 2 * D );		// D is the 0.5*||dist||^2.
		if( d > d0 )
			throw std::runtime_error( errorPrefix + "Distance at nearest point is larger than distance at the initial guess." );

		// To verify that the numerical method work, we can check that P-Q(theta,psi) is perpendicular to the tangen plane at Q(theta,psi).
		// @see https://www.ma.ic.ac.uk/~rn/distance2ellipse.pdf:
		// (P-Q)·dQ/dtheta = 0  and (P-Q)·dQ/dpsi = 0
		theta = initialPoint(0);
		psi = initialPoint(1);
		double cond1 = (_ellipsoid->a2()-_ellipsoid->b2())*cos(theta)*sin(theta)*cos(psi) - p.x*_ellipsoid->a()*sin(theta) + p.y*_ellipsoid->b()*cos(theta);
		double cond2 = (SQR(_ellipsoid->a()*cos(theta)) + SQR(_ellipsoid->b()*sin(theta)) - _ellipsoid->c2())*sin(psi)*cos(psi) - p.x*_ellipsoid->a()*sin(psi)*cos(theta)
			- p.y*_ellipsoid->b()*sin(psi)*sin(theta) + p.z*_ellipsoid->c()*cos(psi);
		if( ABS( cond1 ) > FLT_EPSILON || ABS( cond2 ) > FLT_EPSILON )
			throw std::runtime_error( errorPrefix + "Vector PQ is not (numerically) perpendicular to tangent plane at Q(theta,psi)." );
		return SIGN( (*_ellipsoid)( p ) ) * d;
	}

public:
	typedef dlib::matrix<double,0,1> column_vector;

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

		_deltaStop = 1e-8 * _h;				// Stopping condition for trust-region minimizatin problem.
	}

	/**
	 * Evaluate affine-transformed ellipsoidal level-set function by computing exact signed distances to the surface.
	 * @param [in] x x-coordinate in the world frame.
	 * @param [in] y y-coordinate in the world frame.
	 * @param [in] z z-coordinate in the world frame.
	 */
	double operator()( double x, double y, double z ) const override
	{
		std::string coords;
		if( _useCache )
		{
			coords = _getDiscreteCoords( x, y, z );
			auto record = _cache.find( coords );
			if( record != _cache.end() )
				return (record->second).first;
		}

		double theta, psi;
		Point3 p = toCanonicalCoordinates( x, y, z );
		double d = _computeShortestSignedDistance( p, theta, psi );

		if( _useCache )
			_cache[coords] = std::make_pair( d, Point2( theta, psi) );	// Cache shortest distance and angular params of nearest point.
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
	 * @param [out] trackedMaxErrors Maximum errors in dimensionless mean and Gaussian curvatures and phi for sampled nodes (reduced across processes).
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples.
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples.
	 * @param [out] samples Array of collected samples (one per interface node).
	 * @param [out] nNumericalSaddles Number of numerical saddles detected from sampled nodes (reduced across processes).
	 * @param [out] sampledFlag Parallel vector with >= 1 for sampled nodes (next to Gamma), 0s otherwise.
	 * @param [out] hkError Vector to hold absolute mean hk error for sampled nodes.
	 * @param [out] ihk Vector to hold linearly interpolated hk for sampled nodes.
	 * @param [out] h2kgError Vector to hold absolute Gaussian h^2*k error for sampled nodes.
	 * @param [out] ih2kg Vector to hold linearly interpolated Gaussian h^2*k for sampled nodes.
	 * @param [out] phiError Vector to hold phi error for sampled nodes.
	 * @throws invalid_argument exception if phi vector is null.
	 * 		   runtime_error if the cache is disabled or empty, or if a query point is not found in the cache.
	 */
	void collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi,
						 const u_char& octMaxRL, const double xyzMin[P4EST_DIM], const double xyzMax[P4EST_DIM],
						 double trackedMaxErrors[P4EST_DIM], double& trackedMinHK, double& trackedMaxHK,
						 std::vector<std::vector<double>>& samples, int& nNumericalSaddles, Vec sampledFlag=nullptr, Vec hkError=nullptr,
						 Vec ihk=nullptr, Vec h2kgError=nullptr, Vec ih2kg=nullptr, Vec phiError=nullptr ) const
	{
		std::string errorPrefix = _errorPrefix + "collectSamples: ";
		nNumericalSaddles = 0;

		if( !phi )
			throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

		if( !_useCache || _cache.empty() )
			throw std::runtime_error( errorPrefix + "Please enable the cache and make sure is non empty before collecting samples!" );

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

		trackedMinHK = DBL_MAX, trackedMaxHK = 0;	// Track the min and max mean |hk*|
		double trackedMaxHKError = 0;				// and Gaussian curvature and phi errors.
		double trackedMaxH2KGError = 0;
		double trackedMaxPhiError = 0;

#ifdef DEBUG
		std::cout << "    Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
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
				std::string coords = _getDiscreteCoords( xyz[0], xyz[1], xyz[2] );
				const auto record = _cache.find( coords );
				if( record == _cache.end() )
				{
					std::stringstream msg;
					msg << errorPrefix << "Couldn't find [" << Point3( xyz ) << "] (or " << coords << ") in the cache!";
					throw std::runtime_error( msg.str() );
				}

				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	//Does it have a valid stencil?
				{
					invalidNodes++;
					continue;
				}

				// Valid candidate grid node.  Get its exact distance to ellipsoid and true curvatures at closest point.
				double hk, h2kg;
				double d = record->second.first;
				_ellipsoid->getCurvatures( record->second.second.x, record->second.second.y, hk, h2kg );
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
							sampledFlagPtr[n] = 1;		// OK candidate node: not a numerical saddle and hk and ihk have same sign.
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
	}

	/**
	 * Reserve space for cache.  Call this function at the beginning of queries or octree construction.
	 * @param [in] n Reserve size.
	 */
	void reserveCache( const size_t& n )
	{
		_cache.reserve( n );
	}
};

#endif // ML_CURVATURE_ELLIPSOID_3D_H
