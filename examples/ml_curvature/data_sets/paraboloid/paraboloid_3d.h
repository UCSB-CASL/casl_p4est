/**
 * A collection of classes and functions related to a paraboloid.
 *
 * Developer: Luis Ángel.
 * Created: November 20, 2021.
 * Updated: April 31, 2022.
 */

#ifndef ML_CURVATURE_PARABOLOID_3D_H
#define ML_CURVATURE_PARABOLOID_3D_H

#include "../level_set_patch_3d.h"

////////////////////////////////////////////////////// Paraboloid in canonical space ///////////////////////////////////////////////////////

/**
 * A paraboloid in canonical space modeling the parametrized function Q(u,v) = a * u^2 + b * v^2, where a and b are positive.
 */
class Paraboloid : public geom::MongeFunction
{
private:
	const double _a;		// Paraboloid a and b positive coefficients.
	const double _b;

public:
	/**
	 * Constructor.
	 * @param [in] a Paraboloid x-coefficient.
	 * @param [in] b Paraboloid y-coefficient.
	 * @throws invalid_argument error if a or b is not positive.
	 */
	Paraboloid( const double& a, const double& b ) : _a( a ), _b( b )
	{
		if( a <= 0 || b <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Paraboloid::constructor: a and b must be positive!" );
	}

	/**
	 * Evaluate paraboloid Q(u,v).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return Q(u,v) = a * u^2 + b * v^2.
	 */
	double operator()( double u, double v ) const override
	{
		return _a * SQR( u ) + _b * SQR( v );
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
		return 2 * _b * v;
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
		return 2 * _b;
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
	 * Compute mean curvature (always positive because the paraboloid is convex).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return 0.5*(k1+k2), where k1 and k2 are the principal curvatures.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		double num = _a * (1 + SQR( 2 * _b * v )) +  _b * (1 + SQR( 2 * _a * u ));
		double den = pow( 1 + SQR( 2 * _a * u ) + SQR( 2 * _b * v ), 1.5 );
		return num / den;
	}

	/**
	 * Compute the Gaussian curvature (always positive because the paraboloid has no saddle regions).
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return k1*k2, where k1 and k2 are the principal curvatures.
	 */
	double gaussianCurvature( const double& u, const double& v ) const override
	{
		return 4 * _a * _b / SQR( 1 + SQR(2 * _a * u) + SQR(2 * _b * v) );
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
};

////////////////////////////////////////////// Distance from P to paraboloid as a model class //////////////////////////////////////////////

/**
 * A function model for the distance function to the paraboloid.  This can be used with the find_min_trust_region() routine from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q is the paraboloid Q(u,v) = a*u^2 + b*v^2 for positive a and b, and P
 * is the query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
 */
class ParaboloidPointDistanceModel : public PointDistanceModel
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
		auto& Q = (Paraboloid&)F;
		res(0) = (u - P.x) + (Q(u,v) - P.z) * 2 * u * Q.a(); 	// dD/du.
		res(1) = (v - P.y) + (Q(u,v) - P.z) * 2 * v * Q.b(); 	// dD/dv.
		return res;
	}

	/**
	 * The Hessian matrix for the paraboloid-point distance function.
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
		auto& Q = (Paraboloid&)F;
		res(0, 0) = 1 + 6 * SQR(Q.a()) * u * u + 2 * Q.a() * Q.b() * v * v - 2 * Q.a() * P.z;	// d/du(dD/du).
		res(1, 0) = res(0, 1) = 4 * Q.a() * Q.b() * u * v;										// d/du(dD/dv) and d/dv(dD/du).
		res(1, 1) = 1 + 2 * Q.a() * Q.b() * u * u + 6 * SQR(Q.b()) * v * v - 2 * Q.b() * P.z;	// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Paraboloid object.
	 */
	ParaboloidPointDistanceModel( const Point3& p, const Paraboloid& q ) : PointDistanceModel( p, q ) {}
};

/////////////////////////////////////////// Signed distance function to a discretized paraboloid ///////////////////////////////////////////

class ParaboloidLevelSet : public SignedDistanceLevelSet
{
private:
	const std::string _errorPrefix = "[CASL_ERROR] ParaboloidLevelSet::";

	/**
	 * Fix sign of computed distance to triangulated surface by using the Monge function.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in] d Current POSITIVE distance to the triangulated surface.
	 * @return signed d.
	 */
	double _fixSignOfDistance( const Point3& p, const geom::Triangle *nearestTriangle, const double& d ) const override
	{
		// Fix sign: points inside paraboloid are negative and outside are positive.  Because of the way we created the triangles, their
		// normal vector points up in the canonical coord. system (into the negative region Omega-).
		if( p.z > 0 )		// Save time, points above the uv plane can have negative distance, but not those below.
		{
			Point3 w = p - *(nearestTriangle->getVertex(0));
			if( w.dot( *(nearestTriangle->getNormal()) ) >= 0 )			// In the direction of the normal?
				return d * -1;
		}
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
		// Do nothing...
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
		column_vector initialPoint = {nearestPoint.x, nearestPoint.y};					// Initial (u,v) comes from cached closest point.

		double cond;			// Condition for iterating until we find the closest point.
		int iter = 0;
		double d1;				// Shortest distance found numerically.
		Point3 q;				// Potential new nearest point on paraboloid.
		do
		{
			double d0 = (Point3( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) ) - p).norm_L2();	// Distance to initial guess: we must improve this.
			double D = dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),	// Append .be_verbose() for debugging.
													ParaboloidPointDistanceModel( p, *((Paraboloid*)_mongeFunction) ), initialPoint, initialTrustRadius );
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
			throw std::runtime_error( errorPrefix + "Vector PQ is not (numerically) perpendicular to tangent plane at nearest point on paraboloid." );

		double refSign = (p.z >= (*_mongeFunction)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to paraboloid.
		d = refSign * d1;			// Update shortest distance and closest point on parabolid by fixing the sign too (if needed).
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
	 * @param [in] paraboloid Paraboloid Monge patch in canonical coordinates.
	 * @param [in] ru2 Squared half-axis length on the u direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] rv2 Squared half-axis length on the v direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] btKLeaf Maximum number of points in balltree leaves.
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface with a balltree.
	 */
	ParaboloidLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle, const size_t& ku,
						const size_t& kv, const size_t& L, const Paraboloid *paraboloid, const double& ru2, const double& rv2,
						const size_t& btKLeaf=40, const u_short& addToL=0 )
						: SignedDistanceLevelSet( mpi, trans, rotAxis, rotAngle, ku, kv, L, paraboloid, ru2, rv2, btKLeaf, addToL ) {}
};


/**
 * Set up domain and create a paboloid level-set function.
 * @param [in] mpi MPI environment object.
 * @param [in] paraboloid Paraboloid function in canonical space.
 * @param [in] h Mesh size.
 * @param [in] origin Paraboloid's local frame origin with respect to world coordinates.
 * @param [in] rotAngle Paraboloid's local frame angle of rotation about a unit axis.
 * @param [in] rotAxis The unit axis to rotate the frame about.
 * @param [in] maxRL Maximum level of refinement for the whole domain.
 * @param [out] octMaxRL Effective maximum refinement for each octree to achieve desired h.
 * @param [in] c User-defined height to be contained inside the cubic domain.
 * @param [out] n_xyz Number of octrees in each direction.
 * @param [out] xyz_min Mininum coordinates of computational domain.
 * @param [out] xyz_max Maximum coordinates of computational domain.
 * @param [out] ru2 Semi-axis length on the u-direction for sampling (and reaching up to desired Q(u,v) = c).
 * @param [out] rv2 Semi-axis length on the v-direction for sampling (and reaching up to desired Q(u,v) = c).
 * @return Dynamically allocated paraboloid level-set object.  You must delete it in caller function.
 */
ParaboloidLevelSet *setupDomain( const mpi_environment_t& mpi, const Paraboloid& paraboloid, const double& h,
								 const double origin[P4EST_DIM], const double& rotAngle, const double rotAxis[P4EST_DIM],
								 const u_short& maxRL, u_short& octMaxRL, const double& c, int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM],
								 double xyz_max[P4EST_DIM], double& ru2, double& rv2 )
{
	// First, determine region on the uv plane that contains paraboloid up to Q(u,v) = c.
	ru2 = c / paraboloid.a();
	rv2 = c / paraboloid.b();
	const double U_C = sqrt( ru2 );
	const double V_C = sqrt( rv2 );
	const double QTOP = c + 4 * h;					// Adding some padding at top and bottom.
	const double QBOT = -4 * h;
	const double QCylCCoords[8][P4EST_DIM] = {		// Cylinder in canonical coords containing the desired paraboloid surface.
		{-U_C, 0, QTOP}, {+U_C, 0, QTOP}, 			// Top coords (the four points lying on the same QTOP found above).
		{0, -V_C, QTOP}, {0, +V_C, QTOP},
		{-U_C, 0, QBOT}, {+U_C, 0, QBOT},			// Base coords (the four points lying on the same QBOT found above).
		{0, -V_C, QBOT}, {0, +V_C, QBOT}
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

	// Defining a symmetric cubic domain whose dimensions are multiples of h and contain paraboloid and its limiting ellipse.
	for( int i = 0; i < P4EST_DIM; i++ )
	{
		n_xyz[i] = N_TREES;
		xyz_min[i] = WCentroid[i] - HALF_D_CUBE_SIDE_LEN;
		xyz_max[i] = WCentroid[i] + HALF_D_CUBE_SIDE_LEN;
	}

	// Now that we know the domain, define the limiting ellipse to triangulate the paraboloid.
	const double D = D_CUBE_SIDE_LEN * sqrt( 3 );					// To do this, use the circumscribing sphere with this diameter.
	const double ULIM = sqrt( D / paraboloid.a() );					// Limiting ellipse semi-axes for triangulation.
	const double VLIM = sqrt( D / paraboloid.b() );
	const size_t HALF_U_H = ceil( ULIM / h );						// Half axes in h units.
	const size_t HALF_V_H = ceil( VLIM / h );

	// Defining transformed paraboloid level-set function.  This also discretizes the surface using a balltree to speed up queries during
	// grid refinment.
	return new ParaboloidLevelSet( &mpi, Point3(origin), Point3(rotAxis), rotAngle, HALF_U_H, HALF_V_H, maxRL, &paraboloid, SQR(ULIM), SQR(VLIM), 5 );
}


#endif //ML_CURVATURE_PARABOLOID_3D_H
