#ifndef ML_CURVATURE_PARABOLOID_3D_H
#define ML_CURVATURE_PARABOLOID_3D_H

#include <src/casl_geometry.h>
#include <dlib/optimization.h>

///////////////////////////////////////////// Paraboloid in canonical space ////////////////////////////////////////////

/**
 * A paraboloid in canonical space modeling the parametrized function Q(u,v) = a * u^2 + b * v^2, where a and b are
 * positive.
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
	 * @throws Runtime error if a or b is not positive.
	 */
	Paraboloid( const double& a, const double& b ) : _a( a ), _b( b )
	{
		if( a <= 0 || b <= 0 )
			throw std::runtime_error( "[CASL_ERROR] Paraboloid::constructor: a and b must be positive params!" );
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
	 * Compute mean curvature.
	 * @param [in] u x-coordinate value.
	 * @param [in] v y-coordinate value.
	 * @return
	 * TODO: Implement.
	 */
	double meanCurvature( const double& u, const double& v ) const override
	{
		return 0;
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

//////////////////////////////////// Distance from P to paraboloid as a model class ////////////////////////////////////

/**
 * A function model for the distance function to the paraboloid.  This can be used with the find_min_trust_region()
 * routine from dlib.
 * This class represents the function 0.5*norm(Q(u,v) - P)^2, where Q is the paraboloid Q(u,v) = a*u^2 + b*v^2 for posi-
 * tive a and b, and P is the query (but fixed) point.  The goal is to find the closest point on Q(u,v) to P.
 */
class ParaboloidPointDistanceModel
{
	typedef dlib::matrix<double,0,1> column_vector;
	typedef dlib::matrix<double> general_matrix;

private:
	const Point3 P;			// Fixed query point.
	const Paraboloid& Q;	// Reference to the paraboloid in canonical coordinates.

	/**
	 * Compute the distance from the paraboloid surface to a fixed point.
	 * @param [in] m (u,v) parameters to evaluate in distance function.
	 * @return D(u,v) = 0.5*norm(Q(u,v) - P)^2.
	 */
	double _evalDistance( const column_vector& m ) const
	{
		const double u = m( 0 );
		const double v = m( 1 );

		return 0.5 * (SQR( u - P.x ) + SQR( v - P.y ) + SQR( Q(u,v) - P.z ));
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

		// Make us a column vector of length 2 to hold the gradient.
		column_vector res( 2 );

		// Now, compute the gradient vector.
		res( 0 ) = (u - P.x) + (Q(u,v) - P.z) * 2 * u * Q.a(); 	// dD/du.
		res( 1 ) = (v - P.y) + (Q(u,v) - P.z) * 2 * v * Q.b(); 	// dD/dv.
		return res;
	}

	/**
	 * The Hessian matrix for the paraboloid-point distance function.
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
		res( 0, 0 ) = 1 + 6 * SQR(Q.a()) * u * u + 2 * Q.a() * Q.b() * v * v - 2 * Q.a() * P.z;	// d/du(dD/du).
		res( 1, 0 ) = res( 0, 1 ) = 4 * Q.a() * Q.b() * u * v;									// d/du(dD/dv) and d/dv(dD/du).
		res( 1, 1 ) = 1 + 2 * Q.a() * Q.b() * u * u + 6 * SQR(Q.b()) * v * v - 2 * Q.b() * P.z;	// d/dv(dD/dv).
		return res;
	}

public:
	/**
	 * Constructor.
	 * @param [in] p Query fixed point.
	 * @param [in] q Paraboloid object.
	 */
	ParaboloidPointDistanceModel( const Point3& p, const Paraboloid& q ) : P( p ), Q( q ) {}

	/**
	 * Interface for evaluating the paraboloid-point distance function.
	 * @param [in] x The (u,v) parameters to obtain the point on the paraboloid (u,v,Q(u,v)).
	 * @return D(x) = 0.5*norm(Q(x) - P)^2.
	 */
	double operator()( const column_vector& x ) const
	{
		return _evalDistance( x );
	}

	/**
	 * Compute gradient and Hessian of the paraboloid-point distance function.
	 * @note The function name and parameter order shouldn't change as this is the signature that dlib expects.
	 * @param [in] x The (u,v) parameters to calculate the point on the paraboloid (u,v,Q(u,v)).
	 * @param [out] grad Gradient of distance-point function.
	 * @param [out] H Hessian of distance-point function.
	 */
	void get_derivative_and_hessian( const column_vector& x, column_vector& grad, general_matrix& H ) const
	{
		grad = _evalGradient( x );
		H = _evalHessian( x );
	}
};

///////////////////////////////// Signed distance function to a discretized paraboloid /////////////////////////////////

class ParaboloidLevelSet : public geom::DiscretizedMongePatch
{
private:
	const double _beta;				// Transformation parameters to vary canonical system w.r.t. world coordinate system.
	const Point3 _axis;				// These include a rotation (unit) axis and angle, and a translation vector that sets
	const Point3 _trns;				// the origin of the canonical system to any point in space.
	const double _c, _s;			// Since we use cosine and sine of beta a lot, let's precompute them.
	const double _one_m_c;			// (1-cos(beta)).

	const Paraboloid *_paraboloid;	// Paraboloid in canonical coordinates.

public:
	/**
	 * Constructor.
	 * @param [in] k Number of halves to define a symmetric domain (i.e., domain is [-0.5k, 0.5k]^2)
	 * @param [in] L Number of refinement levels per unit length.
	 * @param [in] paraboloid Paraboloid object in canonical coordinates.
	 * @param [in] btKLeaf Maximum number of points in balltree leaf nodes.
	 */
	ParaboloidLevelSet( const Point3& trans, const Point3& rotAxis, const double& rotAngle,
						const size_t& k, const size_t& L, const Paraboloid *paraboloid, const size_t& btKLeaf=40 )
						: _trns( trans ), _axis( rotAxis ), _beta( rotAngle), _paraboloid( paraboloid ),
						_c( cos( rotAngle ) ), _s( sin( rotAngle ) ), _one_m_c( 1 - cos( rotAngle ) ),
						DiscretizedMongePatch( k, L, paraboloid, btKLeaf )
	{
		if( rotAxis.norm_L2() < EPS )		// Singular rotation axis?
			throw std::runtime_error( "[CASL_ERROR] ParaboloidLevelSet::constructor: Rotation axis shouldn't be 0!" );
	}

	/**
	 * Transform a point/vector in world coordinates to canonical coordinates using the tranformation info.
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @param [in] isVector True if input is a vector (unnaffected by translation), false if input is a point.
	 * @return The coordinates of (x,y,z) in the representation of the paraboloid canonical coordinate system.
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
	 * Transform a point/vector in canonical coordinates to world coordinates using the transformation info.
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
	 * Get signed distance to discretized paraboloid (triangulated and with vertices structured into a balltree).
	 * @param [in] x x-coordinate.
	 * @param [in] y y-coordinate.
	 * @param [in] z z-coordinate.
	 * @return phi(x,y,z).
	 */
	double operator()( double x, double y, double z ) const override
	{
		Point3 q = toCanonicalCoordinates( x, y, z );	// Transform query point to canonical coordinates.
		double d = geom::DiscretizedMongePatch::operator()( q.x, q.y, q.z );
		double refZ = (*_paraboloid)( q.x, q.y );

		// Fix sign: points inside paraboloid are negative and outside are positive.
		if( z > refZ )
			d *= -1;

		std::cout << "(" << x << ", " << y << ", " << z << "): " << d << std::endl;
		return d;
	}
};

#endif //ML_CURVATURE_PARABOLOID_3D_H
