/**
 * A collection of classes and functions related to a Gaussian surface embedded in 3d.
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: April 25, 2022.
 */

#ifndef ML_CURVATURE_GAUSSIAN_3D_H
#define ML_CURVATURE_GAUSSIAN_3D_H

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <boost/math/tools/roots.hpp>
#include "../level_set_patch_3d.h"

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
	double dQdu( const double& u, const double& v, double Q ) const override
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
	double dQdv( const double& u, const double& v, double Q ) const override
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
	double d2Qdu2( const double& u, const double& v, double Q, double Qu ) const override
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
	double d2Qdv2( const double& u, const double& v, double Q, double Qv ) const override
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
	double d2Qdudv( const double& u, const double& v, double Q ) const override
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
class GaussianPointDistanceModel : public PointDistanceModel
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
		double Q = F(u, v);
		double Qu = F.dQdu( u, v, Q );			// Sending cached Q too.
		double Qv = F.dQdv( u, v, Q );
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

		double Q = F(u, v);
		double Qu = F.dQdu( u, v, Q );			// Sending in cached Q too.
		double Qv = F.dQdv( u, v, Q );
		double Quu = F.d2Qdu2( u, v, Q, Qu );
		double Qvv = F.d2Qdv2( u, v, Q, Qv );
		double Quv = F.d2Qdudv( u, v, Q );

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
	GaussianPointDistanceModel( const Point3& p, const Gaussian& g ) : PointDistanceModel( p, g ) {}
};

///////////////////////////////////// Signed distance function to a triangulated Gaussian Monge patch //////////////////////////////////////

class GaussianLevelSet : public SignedDistanceLevelSet
{
private:
	const std::string _errorPrefix = "[CASL_ERROR] GaussianLevelSet::";

	/**
	 * Fix sign of computed distance to triangulated surface by using the Monge function.
	 * @param [in] p Query point in canonical coordinates.
	 * @param [in] nearestTriangle Nearest triangle found from querying the balltree.
	 * @param [in] d Current POSITIVE distance to the triangulated surface.
	 * @return signed d.
	 */
	double _fixSignOfDistance( const Point3& p, const geom::Triangle *nearestTriangle, const double& d ) const override
	{
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

		dlib::find_min_trust_region( dlib::objective_delta_stop_strategy( _deltaStop ),	// Append .be_verbose() for debugging.
									 GaussianPointDistanceModel( p, *((Gaussian*)_mongeFunction) ), initialPoint, initialTrustRadius );

		// Check if minimization produced a better d* distance from query point p to the Gaussian.
		Point3 q( initialPoint(0), initialPoint(1), (*_mongeFunction)( initialPoint(0), initialPoint(1) ) );
		double refSign = (p.z >= (*_mongeFunction)( p.x, p.y ))? -1 : 1;	// Exact sign for the distance to Gaussian.
		double dist = (p - q).norm_L2();

		if( dist >= ABS( d ) )							// If d* is smaller, it's fine.  The problem is if d*>d.
		{
			if( refSign * d < 0 )						// If both signs agree, it's fine.  If not, let's check.
			{
				if( ABS( dist - ABS( d ) ) > 0.15 * _h )
					throw std::runtime_error( errorPrefix + "Distances disagree in sign and magnitude by more than 0.15h!" );
			}
		}

		d = refSign * dist;	// Update shortest distance and closest point on Gaussian by fixing the sign too (if needed).
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
	 * @param [in] gaussian Gaussian object in canonical coordinates.
	 * @param [in] ru2 Squared half-axis length on the u direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] rv2 Squared half-axis length on the v direction for the limiting ellipse on the canonical uv plane.
	 * @param [in] btKLeaf Maximum number of points in balltree leaves.
	 * @param [in] addToL Adds more "levels of refinement" to triangulate the surface with a balltree.
	 */
	GaussianLevelSet( const mpi_environment_t *mpi, const Point3& trans, const Point3& rotAxis, const double& rotAngle,
					  const size_t& ku, const size_t& kv, const size_t& L, const Gaussian *gaussian,
					  const double& ru2, const double& rv2, const size_t& btKLeaf=40, const u_short& addToL=0 )
					  : SignedDistanceLevelSet( mpi, trans, rotAxis, rotAngle, ku, kv, L, gaussian, ru2, rv2, btKLeaf, addToL ) {}
};

#endif //ML_CURVATURE_GAUSSIAN_3D_H
