/**
 * A collection of classes and functions related to a hyperbolic paraboloid for generating saddle samples.
 *
 * Developer: Luis Ángel.
 * Created: May 4, 2022.
 * Updated: May 15, 2022.
 */

#ifndef ML_CURVATURE_HYP_PARABOLOID_3D_H
#define ML_CURVATURE_HYP_PARABOLOID_3D_H

#include "../level_set_patch_3d.h"

///////////////////////////////////////////////// Hyperbolic paraboloid in canonical space /////////////////////////////////////////////////

/**
 * A paraboloid in canonical space modeling the parametrized function Q(u,v) = a * u^2 - b * v^2, where a,b > 0.
 */
class HypParaboloid : public geom::MongeFunction
{
private:
	const double _a;		// Paraboloid a and b positive coefficients.
	const double _b;

public:
	/**
	 * Constructor.
	 * @param [in] a Hyp-paraboloid x-coefficient.
	 * @param [in] b Hyp-paraboloid y-coefficient.
	 * @throws invalid_argument error if a or b is not positive.
	 */
	HypParaboloid( const double& a, const double& b ) : _a( a ), _b( b )
	{
		if( a <= 0 || b <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] HypParaboloid::constructor: a and b must be positive!" );
	}

	/**
	 * Evaluate hyperboloic paraboloid Q(u,v).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return Q(u,v) = a * u^2 - b * v^2.
	 */
	double operator()( double u, double v ) const override
	{
		return _a * SQR( u ) - _b * SQR( v );
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
		return 2 * _a * u;
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
		return -2 * _b * v;
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
		return 2 * _a;
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
		return -2 * _b;
	}

	/**
	 * Crossed second derivative.
	 * @param [in] u Value along the u direction.
	 * @param [in] v Value along the v direction.
	 * @param [in] Q (Optional) Cached Q(u,v).
	 * @return d2Qdudv(u, v).
	 */
	double d2Qdudv( const double& u, const double& v, double Q ) const override
	{
		return 0;
	}

	/**
	 * Compute mean curvature (may be positive or negative).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return 0.5*(k1+k2), where k1 and k2 are the principal curvatures.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double num = _a * (1 + SQR( 2 * _b * v )) - _b * (1 + SQR( 2 * _a * u ));
		double den = pow( 1 + SQR( 2 * _a * u ) + SQR( 2 * _b * v ), 1.5 );
		return num / den;
	}

	/**
	 * Compute the Gaussian curvature (always negative because the hyperbolic paraboloid is a saddle surface).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return k1*k2, where k1 and k2 are the principal curvatures.
	 */
	double gaussianCurvature( const double& u, const double& v ) const override
	{
		return -4 * _a * _b / SQR( 1 + SQR(2 * _a * u) + SQR(2 * _b * v) );
	}

	/**
	 * Getter for parameter a.
	 * @return a.
	 */
	double a() const
	{
		return _a;
	}

	/**
	 * Getter for parameter b.
	 * @return
	 */
	double b() const
	{
		return _b;
	}

	/**
	 * Find the a and b parameters that yield the desired |maximum mean curvature| at the critical point(s) of the hyperboloic paraboloid.
	 * Compute also the sampling radius around the critical points on the uv plane.
	 * @param [in] r Ratio a/b if r>0, or b/a if r<0.
	 * @param [in] maxK Desired maximum curvature at any one of the critical points.
	 * @param [out] a Hyp-paraboloid a parameter.
	 * @param [out] b Hyp-paraboloid b parameter.
	 * @param [out] samRadius Output sampling radius.
	 * @param [in] h Mesh size.
	 * @param [in] minSamRadiusH How many h units we want around the critical points.
	 */
	static void findParamsAndSamRadius( double r, double maxK, double& a, double& b, double& samRadius, const double& h,
										const u_int& minSamRadiusH=16 )
	{
		double u, v;
		const std::string errorPrefix = "[CASL_ERROR] HypParaboloid::findParamsAndSamRadius: ";
		if( r < 0 )			// We have b=|r|a with |r| > 1; thus, k_max should be negative.
		{
			r = ABS( r );
			maxK = -ABS( maxK );
			v = 0;
			if( r < 3 )		// Bifurcation occurs at |r| = 3; before it, we have two u values where |k_max| is attained (with v=0).
			{
				a = -maxK / 2 * pow( 3 / r, 1.5 );
				u = sqrt( (3 / r - 1) / SQR( 2 * a ) );		// |k_max| occurs at (-u, 0) and (u, 0).
				if( 2 * u < 1.5 * h )
					throw std::runtime_error( errorPrefix + "Distance between (-u,0) and (u,0) is less than 1.5h: under-resolved curvature!" );
			}
			else			// |r| >= 3; |k_max| occurs at (0,0).
			{
				u = 0;
				a = maxK / (1 - r);
			}
			b = r * a;
		}
		else				// We have a=rb with r>=1; thus, k_max should be positive.
		{
			u = 0;
			maxK = ABS( maxK );
			if( r < 3 )		// Before bifurcation at r=3, we have two v values where k_max is attained (with u=0).
			{
				b = maxK / 2 * pow( 3 / r, 1.5 );
				v = sqrt( (3 / r - 1) / SQR( 2 * b ) );		// k_max occurs at (0, -v) and (0, v).
				if( 2 * v < 1.5 * h )
					throw std::runtime_error( errorPrefix + "Distance between (0,-v) and (0,v) is less than 1.5h: under-resolved curvature!" );
			}
			else			// r >= 3; k_max occurs at (0,0).
			{
				v = 0;
				b = maxK / (r - 1);
			}
			a = r * b;
		}

		samRadius = MAX(u, v) + h * minSamRadiusH;			// Give space around critical points on the uv plane.
	}
};

//////////////////////////////////////////// Distance from P to hyp-paraboloid as a model class ////////////////////////////////////////////

/**
 * A function model for the distance function to the hyperbolic paraboloid.  This can be used with delib's find_min_trust_region() routine.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q is the hyp-paraboloid Q(u,v) = a*u^2 + b*v^2 for a,b > 0, and P is the
 * query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
 */
class HypParaboloidPointDistanceModel : public PointDistanceModel
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

		// Make us a column vector of length 2 to hold the gradient.
		column_vector res( 2 );

		// Now, compute the gradient vector.
		auto& Q = (HypParaboloid&)F;
		res(0) = (u - P.x) + (Q(u,v) - P.z) * 2 * u * Q.a(); 	// dD/du.
		res(1) = (v - P.y) - (Q(u,v) - P.z) * 2 * v * Q.b(); 	// dD/dv.
		return res;
	}

	/**
	 * The Hessian matrix for the hyp-paraboloid-point distance function.
	 * @param [in] m (u,v) parameters to evaluate in Hessian of distance function.
	 * @return grad(grad(D))(u,v)
	 */
	dlib::matrix<double> _evalHessian ( const column_vector& m ) const override
	{
		const double u = m(0);
		const double v = m(1);

		// Make us a 2x2 matrix.
		dlib::matrix<double> res( 2, 2 );

		// Now, compute the second derivatives.
		auto& Q = (HypParaboloid&)F;
		double q = Q(u,v);
		res(0, 0) = 1 + 2 * Q.a() * (q - P.z) + SQR(2 * Q.a() * u);	// d/du(dD/du).
		res(1, 0) = res(0, 1) = -4 * Q.a() * Q.b() * u * v;			// d/du(dD/dv) and d/dv(dD/du).
		res(1, 1) = 1 - 2 * Q.b() * (q - P.z) + SQR(2 * Q.b() * v);	// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Paraboloid object.
	 */
	HypParaboloidPointDistanceModel( const Point3& p, const HypParaboloid& q ) : PointDistanceModel( p, q ) {}
};

///////////////////////////////////////// Signed distance function to a discretized hyp-paraboloid /////////////////////////////////////////

class HypParaboloidLevelSet : public SignedDistanceLevelSet
{
private:
	const std::string _errorPrefix = "[CASL_ERROR] HypParaboloidLevelSet::";

	/**
	 * Fix sign of computed distance to triangulated surface by using the Monge function.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in] d Current POSITIVE distance to the triangulated surface.
	 * @return signed d.
	 */
	double _fixSignOfDistance( const Point3& p, const geom::Triangle *nearestTriangle, const double& d ) const override
	{
		// Fix sign: points above hyp-paraboloid are negative and beow are positive.  Because of the way we created the triangles, their
		// normal vector points up in the canonical coord. system (into the negative region Omega-).
		Point3 w = p - *(nearestTriangle->getVertex(0));
		if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )			// In the direction of the normal?
			return d * -1;
		return d;
	}

	/**
	 * Handle the case of points whose projections onto the uv plane fall outside the limiting ellipse by possibly correcting their signed
	 * distance and currently set nearest point to triangulated surface.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in,out] d Current linearly computed signed distance.
	 * @param [in,out] nearestPoint Current linearly computed nearest point.
	 */
	void _handlePointsBeyondLimitingEllipse( const Point3& p, const geom::Triangle *nearestTriangle, double& d, Point3& nearestPoint ) const override
	{
		// Do nothing; we must ensure the triangulation ellipse (circle) "contains" the whole cubic domain.
	}

	/**
	 * Compute exact signed distance to surface using dlib's trust region method.
	 * @param [in] p Query point in canonical coordinates which we have corroborated that lies inside some limiting ellipse.
	 * @param [in,out] d Current linearly computed signed distance and then found to nearest point on Monge patch.
	 * @param [in,out] nearestPoint Current linearly computed nearest point to triangulated surface and then a more accurate version.
	 * @throws runtime_error if exact signed distance computation fails.
	 */
	void _computeExactSignedDistance( const Point3& p, double& d, Point3& nearestPoint ) const override
	{
		const std::string errorPrefix = _errorPrefix + "_computeExactSignedDistance: ";
		double initialTrustRadius = MAX( _h, ABS( d ) );
		column_vector initialPoint = {nearestPoint.x, nearestPoint.y};		// Initial (u,v) comes from cached closest point.

		double cond;			// Condition for iterating until we find the closest point.
		int iter = 0;
		double d1;				// Shortest distance found numerically.
		Point3 q;				// Potential new nearest point on hyp-paraboloid.
		do
		{
			double d0 = (Point3( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) ) - p).norm_L2();	// Distance to initial guess: we must improve this.
			double D = dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),	// Append .be_verbose() for debugging.
													HypParaboloidPointDistanceModel( p, *((HypParaboloid*)_mongeFunction) ), initialPoint, initialTrustRadius );
			d1 = sqrt( 2 * D );	// D is the 0.5*||dist||^2.
			if( d1 > d0 )
				throw std::runtime_error( errorPrefix + "Distance at nearest point is larger than distance at the initial guess." );

			// To verify that the numerical method work, we must verify that the normal vector at q (new nearest point) is parallel to vector r=pq.
			q = Point3( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) );
			Point3 r( p - q );
			Point3 g( (*_mongeFunction).dQdu( q.x, q.y, q.z ), (*_mongeFunction).dQdv( q.x, q.y, q.z ), -1 );	// Gradient at q.
			cond = ABS( r.dot( g ) / (r.norm_L2() * g.norm_L2()) ) - 1;		//  We expect cos(t) = r·g/(|r||g|) = +- 1.
			iter++;
		}
		while( iter < 5 && cond > FLT_EPSILON );

		if( cond > FLT_EPSILON )
			throw std::runtime_error( errorPrefix + "Vector PQ is not (numerically) perpendicular to tangent plane at nearest point on hyp-paraboloid." );

		double refSign = (p.z >= (*_mongeFunction)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to hyp-paraboloid.
		d = refSign * d1;			// Update shortest distance and closest point on gaussian by fixing the sign too (if needed).
		nearestPoint = q;
	}

public:
	/**
	 * Constructor.
	 * @param [in] mpi MPI environment.
	 * @param [in] trans Translation vector.
	 * @param [in] rotAxis Rotation axis (must be nonzero).
	 * @param [in] rotAngle Rotation angle about rotAxis.
	 * @param [in] ku Number of min cells in u half direction to define a symmetric domain.
	 * @param [in] kv Number of min cells in v half direction to define a symmetric domain.
	 * @param [in] L Number of refinement levels per unit length (so that h=2^{-L} is a power of two).
	 * @param [in] hypParaboloid Hyp-paraboloid Monge patch in canonical coordinates.
	 * @param [in] ru2 Squared half-axis length on the u direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] rv2 Squared half-axis length on the v direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] sdsu2 Squared half-axis length on the u direction for signed-distance computation on the canonical uv plane.
	 * @param [in] sdsv2 Squared half-axis length on the v direction for signed-distance computation on the canonical uv plane.
	 * @param [in] btKLeaf Maximum number of points in balltree leaves.
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface with balltree.
	 */
	HypParaboloidLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
						   const size_t& ku, const size_t& kv, const u_short& L, const HypParaboloid *hypParaboloid, const double& ru2,
						   const double& rv2, const double& sdsu2, const double& sdsv2, const size_t& btKLeaf=40, const u_short& addToL=0 )
						   : SignedDistanceLevelSet( mpi, trans, rotAxis, rotAngle, ku, kv, L, hypParaboloid, ru2, rv2, sdsu2, sdsv2,
													 btKLeaf, addToL ) {}

	/**
	 * Collect samples by splitting them into non-saddle and saddle types.
	 * @note Samples are not normalized in any way: not negative-mean-curvature nor gradient-reoriented to first octant.
	 * @param [in] p4est P4est data structure.
	 * @param [in] nodes Nodes data structure.
	 * @param [in] ngbd Nodes' neighborhood data structure.
	 * @param [in] phi Parallel vector with level-set values.
	 * @param [in] octMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
	 * @param [in] xyzMin Domain's minimum coordinates.
	 * @param [in] xyzMax Domain's maximum coordinates.
	 * @param [out] trackedMinHK Minimum |hk*| detected across processes for this batch of samples (non-saddle idx 0, saddle idx 1).
	 * @param [out] trackedMaxHK Maximum |hk*| detected across processes for this batch of samples (non-saddle idx 0, saddle idx 1).
	 * @param [in,out] genP Random engine for ease-off subsampling.
	 * @param [out] nonSaddleSamples Array of collected samples for non-saddle regions (i.e., ih2kg > nonSaddleMinH2KG).
	 * @param [in] minHK Minimum expected |hk*| for non-saddle samples.
	 * @param [out] saddleSamples Array of collected samples for saddle regions (i.e., ih2kg <= 0).
	 * @param [in] easeOffMaxIH2KG Max value for ease-off distribution according to ih2kg.
	 * @param [in] easeOffProbMaxIH2KG Probability associated with max value for ease-off distribution according to ih2kg.
	 * @param [in] easeOffProbMinIH2KG Probability associated with min value (i.e., nonSaddleMinIH2KG) for ease-off distribution according to ih2kg.
	 * @param [in] filter Vector with 1s for nodes we are allowed to sample from and anything else for non-sampling nodes.
	 * @param [in] maxHK Current achievable max |hk| for further subsampling.
	 * @param [in] samR2 Overriding sampling sphere radius^2 on canonical space (to be used instead of limiting circle used for triangulation).
	 * @param [in] nonSaddleMinIH2KG Min numerical dimensionless Gaussian curvature (at Gamma) for numerical non-saddle points.
	 * @return Maximum errors in dimensionless mean and Gaussian curvatures (reduced across processes).
	 * @throws runtime_error if cache is disabled or empty, or if we can't locate a node in the cache, or if sampling fails critically.
	 * @throws invalid_argument if phi or filter are null, or if sampling radius^2 is larger than limiting radius^2 or smaller than (1.5h)^2.
	 */
	std::pair<double,double> collectSamples( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
											 const Vec& phi, const unsigned char octMaxRL, const double xyzMin[P4EST_DIM],
											 const double xyzMax[P4EST_DIM], double trackedMinHK[SAMPLE_TYPES],
											 double trackedMaxHK[SAMPLE_TYPES], std::mt19937& genP,
											 std::vector<std::vector<double>>& nonSaddleSamples, const double& minHK, 			// Non-saddle params.
											 std::vector<std::vector<double>>& saddleSamples, const double& easeOffMaxIH2KG, 	// Saddle params.
											 const double& easeOffProbMaxIH2KG, const double& easeOffProbMinIH2KG,
											 Vec filter, const double& maxHK, double samR2=NAN, const double& nonSaddleMinIH2KG=-7e-6 ) const
	{
		std::string errorPrefix = _errorPrefix + "collectSamples: ";
		if( !_useCache || _cache.empty() || _canonicalCoordsCache.empty() )
			throw std::runtime_error( errorPrefix + "Cache must be enabled and nonempty!" );

		if( !phi )
			throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

		if( !filter )
			throw std::invalid_argument( errorPrefix + "filter vector can't be null!" );

		if( !isnan( samR2 ) && (samR2 > _ru2 || samR2 > _rv2 || samR2 < SQR(1.5 * _h)) )
			throw std::invalid_argument( errorPrefix + "Overriding sampling radius can't be larger than local sampling radius or less than 1.5h!" );
		samR2 = (isnan( samR2 )? _ru2 : samR2);

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

		const double *filterReadPtr = nullptr;					// Load the filter vector.
		CHKERRXX( VecGetArrayRead( filter, &filterReadPtr ) );

		nonSaddleSamples.clear();								// We'll get (possibly) as many as points next to Gamma
		nonSaddleSamples.reserve( indices.size() );				// and within limiting sampling circle.
		saddleSamples.clear();
		saddleSamples.reserve( indices.size() );

		// Prepare mean and Gaussian curvatures interpolation.
		my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
		kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );

		std::uniform_real_distribution<double> pDistribution;
		trackedMinHK[0] = DBL_MAX, trackedMinHK[1] = DBL_MAX, trackedMaxHK[0] = 0, trackedMaxHK[1] = 0;	// Track the min and max mean |hk*|
		double trackedMaxHKError = 0;																	// and Gaussian curvature errors.
		double trackedMaxH2KGError = 0;

#ifdef DEBUG
		std::cout << "Rank " << _mpi->rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
#endif

		for( const auto& n : indices )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * _h || ABS( xyz[0] - xyzMax[0] ) <= 4 * _h ||	// Skip nodes too close
				ABS( xyz[1] - xyzMin[1] ) <= 4 * _h || ABS( xyz[1] - xyzMax[1] ) <= 4 * _h ||	// to domain boundary.
				ABS( xyz[2] - xyzMin[2] ) <= 4 * _h || ABS( xyz[2] - xyzMax[2] ) <= 4 * _h )
				continue;

			std::string coords = getDiscreteCoords( xyz[0], xyz[1], xyz[2] );
			const auto record = _cache.find( coords );
			const auto recordCCoords = _canonicalCoordsCache.find( coords );
			if( record == _cache.end() && recordCCoords == _canonicalCoordsCache.end() )		// Canonical coords should be recorded.
			{
				std::stringstream msg;
				msg << errorPrefix << "Couldn't find [" << Point3( xyz ) << "] (or " << coords << ") in the cache!";
				throw std::runtime_error( msg.str() );
			}
			Point3 p = recordCCoords->second;

			if( SQR( p.x ) + SQR( p.y ) + SQR( p.z ) > samR2 )	// Skip nodes whose point on the surface lies outside the sampling sphere.
				continue;

			std::vector<p4est_locidx_t> stencil;
			try
			{
				if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Does it have a valid stencil?
					continue;

				for( auto& s : stencil )						// Invalidate point if at least one stencil node is not exact-distance-flagged.
				{
					if( filterReadPtr[s] != 1 )
						throw std::runtime_error( "Caught invalid point for " + std::to_string( n ) + "!" );
				}

				// Valid candidate grid node.  Get its exact distance to Gaussian and true curvatures at closest point.
				Point3 nearestPoint = record->second.second;
				double hk = _h * _mongeFunction->meanCurvature( nearestPoint.x, nearestPoint.y );
				double h2kg = SQR( _h ) * _mongeFunction->gaussianCurvature( nearestPoint.x, nearestPoint.y );

				for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
					xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
				double kappaMGValues[2];
				kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
				double ihkVal = _h * kappaMGValues[0];
				double ih2kgVal = SQR( _h ) * kappaMGValues[1];

				bool isNonSaddle;
				if( ih2kgVal >= nonSaddleMinIH2KG )				// Not a saddle?
				{
					if( ABS( hk ) < minHK )						// Target mean |hk*| must be >= minHK for non-saddle regions.
						continue;
					isNonSaddle = true;							// Flag point as non-saddle.
				}
				else
				{
					double prob = kml::utils::easingOffProbability( ABS( ih2kgVal ), nonSaddleMinIH2KG, easeOffProbMinIH2KG,
																	easeOffMaxIH2KG, easeOffProbMaxIH2KG );
					if( pDistribution( genP ) > prob )
						continue;								// Use an easing-off prob to keep samples.

					prob = kml::utils::easingOffProbability( ABS( hk ), 0, 0.0025, abs(maxHK)/2, 1 );
					if( pDistribution( genP ) > prob )
						continue;
                    
                    prob = kml::utils::easingOffProbability( ABS(hk - ihkVal), 0, 0.005, 0.1, 1 );
                    if( pDistribution( genP ) > prob )
                        continue;

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

				// Update stats.
				int which = isNonSaddle? 0 : 1;
				trackedMinHK[which] = MIN( trackedMinHK[which], ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );
				trackedMaxHK[which] = MAX( trackedMaxHK[which], ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] ) );

				double errorHK = ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] - (*sample)[K_INPUT_SIZE_LEARN - 3] );
				double errorH2KG = ABS( (*sample)[K_INPUT_SIZE_LEARN - 2] - (*sample)[K_INPUT_SIZE_LEARN - 1] );

				trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
				trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );

				// Saddle samples are biased towards negative mean hk.  Let's level up the situation by flipping randomly some of them.
//				if( !isNonSaddle && hk < 0 && pDistribution( genP ) <= 0.125 )
//					kml::utils::normalizeToNegativeCurvature( *sample, -hk, true );		// Note the -hk to force sign flipping.
			}
			catch( std::runtime_error &rt )
			{
#ifdef DEBUG
				std::cerr << rt.what() << std::endl;
#endif
			}
			catch( std::invalid_argument &ia )
			{
				throw std::runtime_error( ia.what() );			// Raise again the exception for critical errors.
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

		// Clean up.
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
};

////////////////////////////////////////// Utility functions for hyperbolic paraboloid data sets ///////////////////////////////////////////

/**
 * Set up domain and create a hyperbolic paboloid level-set function.
 * @param [in] mpi MPI environment object.
 * @param [in] hypParaboloid Paraboloid function in canonical space.
 * @param [in] h Mesh size.
 * @param [in] origin Paraboloid's local frame origin with respect to world coordinates.
 * @param [in] rotAngle Paraboloid's local frame angle of rotation about a unit axis.
 * @param [in] rotAxis The unit axis to rotate the frame about.
 * @param [in] maxRL Maximum level of refinement for the whole domain.
 * @param [out] octMaxRL Effective maximum refinement for each octree to achieve desired h.
 * @param [in] samRadius Requested sampling radius (containing the point(s) on the surface with highest curvature).
 * @param [out] n_xyz Number of octrees in each direction.
 * @param [out] xyz_min Mininum coordinates of computational domain.
 * @param [out] xyz_max Maximum coordinates of computational domain.
 * @return Dynamically allocated hyperbolic paraboloid level-set object.  You must delete it in caller function.
 */
HypParaboloidLevelSet *setupDomain( const mpi_environment_t& mpi, const HypParaboloid& hypParaboloid, const double& h,
									const double origin[P4EST_DIM], const double& rotAngle, const double rotAxis[P4EST_DIM],
									const u_char& maxRL, u_char& octMaxRL, const double& samRadius, int n_xyz[P4EST_DIM],
									double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] )
{
	double uvLim;
	u_int halfUVH;

	if( mpi.rank() == 0 )
	{
		// First, determine the bounds of the cylinder containing the surface.
		const double QTOP = MIN( 32 * h, MAX(
			MAX( hypParaboloid(-samRadius, 0), hypParaboloid(samRadius, 0) ),
			hypParaboloid(0, 0),
			MAX( hypParaboloid(0, -samRadius), hypParaboloid(0, samRadius) ) ) ) + 4 * h;		// Adding some padding at the top.
		const double QBOT = MAX( -32 * h, MIN(
			MIN( hypParaboloid(-samRadius, 0), hypParaboloid(samRadius, 0) ),
			hypParaboloid(0, 0),
			MIN( hypParaboloid(0, -samRadius), hypParaboloid(0, samRadius) ) ) ) - 4 * h;		// Adding some padding at the bottom.
		const double QCylCCoords[8][P4EST_DIM] = {			// Cylinder in canonical coords containing the desired surface.
			{-samRadius, 0, QTOP}, {+samRadius, 0, QTOP}, 	// Top coords (the four points lying on the same QTOP found above).
			{0, -samRadius, QTOP}, {0, +samRadius, QTOP},
			{-samRadius, 0, QBOT}, {+samRadius, 0, QBOT},	// Base coords (the four points lying on the same QBOT found above).
			{0, -samRadius, QBOT}, {0, +samRadius, QBOT}
		};

		// Finding the world coords of (canonical) cylinder containing Q(u,v).
		geom::AffineTransformedSpace ats( Point3( origin ), Point3( rotAxis ), rotAngle );
		double minQCylWCoords[P4EST_DIM] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};	// Min and max cylinder world coords.
		double maxQCylWCoords[P4EST_DIM] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
		for( const auto& cylCPoint : QCylCCoords )
		{
			Point3 cylWPoint = ats.toWorldCoordinates( cylCPoint[0], cylCPoint[1], cylCPoint[2] );
			for( int i = 0; i < P4EST_DIM; i++ )
			{
				minQCylWCoords[i] = MIN( minQCylWCoords[i], cylWPoint.xyz( i ) );
				maxQCylWCoords[i] = MAX( maxQCylWCoords[i], cylWPoint.xyz( i ) );
			}
		}

		// Use the cylinder x, y, z ranges to find the domain's length in each direction.
		double QCylWRange[P4EST_DIM];
		double WCentroid[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			QCylWRange[i] = maxQCylWCoords[i] - minQCylWCoords[i];
			WCentroid[i] = (minQCylWCoords[i] + maxQCylWCoords[i]) / 2;	// Raw centroid.
			WCentroid[i] = round( WCentroid[i] / h ) * h;				// Centroid as a multiple of h.
		}

		const double CUBE_SIDE_LEN = MAX( QCylWRange[0], MAX( QCylWRange[1], QCylWRange[2] ) );
		const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL - 5 );	// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		octMaxRL = maxRL - OCTREE_RL_FOR_LEN;							// Effective max refinement level to achieve desired h.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );			// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;			// Adjusted domain cube len as a multiple of *both* h and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		// Defining a symmetric cubic domain whose dimensions are multiples of h and contain surface and its limiting circle.
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			n_xyz[i] = N_TREES;
			xyz_min[i] = WCentroid[i] - HALF_D_CUBE_SIDE_LEN;
			xyz_max[i] = WCentroid[i] + HALF_D_CUBE_SIDE_LEN;
		}

		// Now that we know the domain, define the limiting ellipse to triangulate the hyp-paraboloid.
		const double D = D_CUBE_SIDE_LEN * sqrt( 3 );					// To do this, use the circumscribing sphere with this diameter.
		uvLim = D / 2;
		halfUVH = ceil( uvLim / h );									// Half axes in h units.

//		PetscPrintf( mpi.comm(), "D = %.6f\n", D );
	}

	// Sync vars.
	SC_CHECK_MPI( MPI_Bcast( &uvLim, 1, MPI_DOUBLE, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( &halfUVH, 1, MPI_UNSIGNED, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( &octMaxRL, 1, MPI_UNSIGNED_CHAR, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( n_xyz, P4EST_DIM, MPI_INT, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( xyz_min, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( xyz_max, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );

	// Defining the transformed level-set function.  This also discretizes the surface using a balltree to speed up queries during grid
	// refinement.  Note we're sending a signed-distance radius possibly way smaller than the triangulation limiting ellipse.  Also, the
	// last +1 tells the discretizer that we want to use 1 more level of refinement when triangulating the surface.
	return new HypParaboloidLevelSet( &mpi, Point3(origin), Point3(rotAxis), rotAngle, halfUVH, halfUVH, maxRL, &hypParaboloid,
									  SQR(uvLim), SQR(uvLim), SQR(samRadius + 4 * h), SQR(samRadius + 4 * h), 5, +1 );
}


#endif //ML_CURVATURE_HYP_PARABOLOID_3D_H
