/**
 * A collection of classes and functions related to a hyperbolic paraboloid for generating saddle samples.
 *
 * Developer: Luis Ángel.
 * Created: May 4, 2022.
 * Updated: May 5, 2022.
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
			maxK *= -ABS( maxK );
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
	 */
	HypParaboloidLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
						   const size_t& ku, const size_t& kv, const size_t& L, const HypParaboloid *hypParaboloid, const double& ru2,
						   const double& rv2, const double& sdsu2, const double& sdsv2, const size_t& btKLeaf=40 )
						   : SignedDistanceLevelSet( mpi, trans, rotAxis, rotAngle, ku, kv, L, hypParaboloid, ru2, rv2, sdsu2, sdsv2,
													 btKLeaf ) {}
};


#endif //ML_CURVATURE_HYP_PARABOLOID_3D_H
