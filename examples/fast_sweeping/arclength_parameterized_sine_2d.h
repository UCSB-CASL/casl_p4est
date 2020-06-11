//
// Created by Im YoungMin on 5/28/20.
//

#ifndef FAST_SWEEPING_ARCLENGTH_PARAMETERIZED_SINE_2D_H
#define FAST_SWEEPING_ARCLENGTH_PARAMETERIZED_SINE_2D_H

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <vector>
#include <utility>
#include <src/casl_geometry.h>
#include <boost/math/tools/roots.hpp>
#include <limits>

/**
 * Define a sinusoidal wave level-set function that is parameterized by arc-length as in [6] page 83, in order to
 * approximate the distance from points in \Omega to the \Gamma.
 * The sinusoidal wave is subjected to an afine transformation.  Given its parameter u \in [uBegin, uEnd], the interface
 * can be computed as:
 *
 *      | x |   | 1  0  Tx |   | cos(\theta)  -sin(\theta)  0 |   |          u            |
 *      | y | = | 0  1  Ty | * | sin(\theta)   cos(\theta)  0 | * | a * sin( \omega * u ) |
 *      | 1 |   | 0  0  1  |   |      0            0        1 |   |          1            |
 * or
 *        q = T(Tx, Ty) * R(\theta) * p(u),        p, q \in R^2
 *
 * in homogeneous coordinates, where [Tx, Ty] is the translation in x and y, and \theta is the rotation angle.
 */
class ArcLengthParameterizedSine : public CF_2
{
private:
	double _a;				// Amplitude.
	double _omega;			// Frequency.
	Point2 _trans;			// Translation (disturbance) from the origin.
	double _theta;			// Rotation angle around the z-axis with respect to horizontal x-axis.
	double _delta;			// Local step size that depends on grid minimum spacing.
	double _uBegin;			// Lower and upper bound for parameter u.
	double _uEnd;
	std::vector<std::pair<double, double>> _arcLengthTable;

	/**
	 * Create the arc length table which is used for parametrization.
	 */
	void _buildArcLengthTable()
	{
		const auto N = (size_t)ceil( ( _uEnd - _uBegin ) / _delta ) + 1;	// Number of points.
		_arcLengthTable.clear();
		_arcLengthTable.reserve( N );
		Point2 p;											// Keep track of last point visited.
		for( size_t i = 0; i < N; i++ )
		{
			double u = MIN( _uBegin + i * _delta, _uEnd );	// We clamp the last parameter u value to expected end.
			double d = 0;									// Distance from previous point.
			Point2 q( u, _a * sin( _omega * u ) );			// Calculations in canonical coordinate system.
			if( i > 0 )
				d = (q - p).norm_L2() + _arcLengthTable[i - 1].second;
			_arcLengthTable.emplace_back( u, d );
			p = q;
		}
	}

public:

	/**
	 * Constructor.
	 * @param [in] a Sine wave amplitude.
	 * @param [in] omega Sine wave frequency.
	 * @param [in] tx Translation of origin in the x-direction.
	 * @param [in] ty Translation of origin in the y-direction.
	 * @param [in] theta Angle of rotation around the z-axis, with respect to positive x-direction.
	 * @param [in] h Minimum space interval in domain.
	 * @param [in] halfAxisLen Half of horizontal axis length.  Parameter u will go from -halfAxisLen to +halfAxisLen.
	 * @throws Runtime exception if beginning and end values for parameter u are not a minimum grading spacing apart.
	 */
	explicit ArcLengthParameterizedSine( double a, double omega, double tx, double ty, double theta, double h,
										 double halfAxisLen )
		: _a( a ), _omega( omega ), _theta( theta ), _delta( h / 25 ), _uBegin( -halfAxisLen ), _uEnd( +halfAxisLen )
	{
#ifdef CASL_THROWS
		if( _uBegin + h >= _uEnd )
			throw std::runtime_error( "[CASL_ERROR] ArcLengthParameterizedSine::ArcLengthParameterizedSine: "
							 		  "Not room enough for parameter u!" );
#endif
		_trans = Point2( tx, ty );
		_buildArcLengthTable();
	}

	/**
	 * Level set evaluation at a given point.
	 * This method uses the arc-length table to find the closest point in the transformed sine wave to the query (x,y)
	 * coordinate.  To make the computation efficient, we "undo" the transformation of the input query point to align
	 * it to the canonical coordinate system of the sine wave function.  Then, distance is computed by linear search
	 * with equally-long steps along the sine wave (thus the arc-length table).
	 * @param [in] x: Point x-coordinate.
	 * @param [in] y: Point y-coordinate.
	 * @return phi(x,y).
	 */
	[[nodiscard]] double operator()( double x, double y ) const override
	{
		// Place query point in terms of the canonical coordinate system of the transformed sine wave.
		toCanonicalCoordinates( x, y );
		Point2 q( x, y );

		// Now, we can calculate distances to equally-spaced nodes along the (untransformed) sine wave.  We'll take
		// steps of size h, using interpolation to approximate the parameter u between nodes.
		const size_t LAST_IDX = _arcLengthTable.size() - 1;
		double minDistance = PETSC_MAX_INT;
		double minU = 0;
		Point2 minP;
		double d = 0;									// Start at the extreme _uBegin.
		size_t idx = 0;									// Index of closest parametric value in the _arcLength table.
		while( d < _arcLengthTable[LAST_IDX].second )
		{
			double segmentLength = _arcLengthTable[idx + 1].second - _arcLengthTable[idx].second;
			double fraction = ( d - _arcLengthTable[idx].second ) / segmentLength;
			double u = _arcLengthTable[idx].first + fraction * _delta;	// The difference between successive u's is _delta.

			Point2 p( u, _a * sin( _omega * u ) );		// Point at interpolated parameter u.
			double distance = ( p - q ).norm_L2();
			if( distance < minDistance )
			{
				minDistance = distance;
				minU = u;
				minP = p;
			}

			d += _delta;								// Move on curve one spacing step.
			if( d >= _arcLengthTable[idx + 1].second )
				idx++;
		}

		// Check last node on the sine wave.
		Point2 p( _arcLengthTable[LAST_IDX].first, _a * sin( _omega * _arcLengthTable[LAST_IDX].first ) );
		double distance = ( p - q ).norm_L2();
		if( distance < minDistance )
		{
			minDistance = distance;
			minU = _arcLengthTable[LAST_IDX].first;
			minP = p;
		}

		// Now that we know the parametric value u that yields the minimum distance, improve by looking at neighbors and
		// find the minimum distance to the left and right line segments.
		double u;
		Point2 closestP;
		if( minU > _arcLengthTable[0].first )			// Can we look at the left?
		{
			u = minU - _delta;
			Point2 pLeft( u, _a * sin( _omega * u) );
			closestP = geom::findClosestPointOnLineSegmentToPoint( q, pLeft, minP );
			distance = ( q - closestP ).norm_L2();
			if( distance < minDistance )
				minDistance = distance;
		}
		if( minU < _arcLengthTable[LAST_IDX].first )	// Can we look at the right?
		{
			u = minU + _delta;
			Point2 pRight( u, _a * sin( _omega * u ) );
			closestP = geom::findClosestPointOnLineSegmentToPoint( q, minP, pRight );
			distance = ( q - closestP ).norm_L2();
			if( distance < minDistance )
				minDistance = distance;
		}

		// Fix sign: points above sine wave are negative, points below are positive.
		double comparativeY = _a * sin( _omega * x );
		if( y > comparativeY )
			minDistance *= -1;

		return minDistance;
	}

	/**
	 * Transform a point in world coordinates to sine wave canonical system.
	 * @param [in,out] x Coordinate in x-dimension.
	 * @param [in,out] y Coordinate in y-dimension.
	 */
	void toCanonicalCoordinates( double& x, double& y ) const
	{
		double x1 = ( x - _trans.x ) * cos( _theta ) + ( y - _trans.y ) * sin( _theta );
		double y1 = ( y - _trans.y ) * cos( _theta ) - ( x - _trans.x ) * sin( _theta );
		x = x1;
		y = y1;
	}

	/**
	 * Transform a point in the canonical sine wave coordinate system to world coordinates.
	 * @param [in,out] x Coordinate in x-direction in canonical coord. system.
	 * @param [in,out] y Coordinate in y-direction in canonical coord. system.
	 */
	void toWorldCoordinates( double& x, double& y ) const
	{
		double x0 = x * cos( _theta ) - y * sin( _theta ) + _trans.x;		// Transformed point.
		double y0 = x * sin( _theta ) + y * cos( _theta ) + _trans.y;
		x = x0;
		y = y0;
	}

	/**
	 * Get a sinusoidal wave point at a given parametric value in the world coordinate system.
	 * @param [in] u Curve parameter.
	 * @return A two-dimensional point.
	 * @throws Runtime execption if u is outside of a valid range and macro CASL_THROWS is defined.
	 */
	[[nodiscard]] Point2 at( double u ) const
	{
#ifdef CASL_THROWS
		assert( u >= _uBegin && u <= _uEnd );	// Check for u \in [uBegin, uEnd].
#endif
		double x = u;							// Coordinates to be transformed.
		double y = _a * sin( _omega * u );
		toWorldCoordinates( x, y );
		return Point2( x, y );
	}

	/**
	 * Compute the curvature of the sinusoidal wave at a particular parametric value.
	 * @param [in] u Parameter.
	 * @return Curvature.
	 */
	[[nodiscard]] double curvature( double u ) const
	{
		return - _a * SQR( _omega ) * sin( _omega * u ) / pow( 1 + SQR( _a * _omega * cos( _omega * u ) ), 1.5 );
	}

	/**
	 * Retrieve sine wave amplitude.
	 * @return A.
	 */
	[[nodiscard]] double getA() const
	{
		return _a;
	}

	/**
	 * Retrieve sine wave frequency.
	 * @return \omega.
	 */
	[[nodiscard]] double getOmega() const
	{
		return _omega;
	}
};

///////////////////// Root finding procedures to compute curvature at nodes adjacent to interface //////////////////////

class DistThetaFunctorDerivative
{
private:
	const double _u, _v;		// We need the query point coordinates.
	const double _a;			// And the sine amplitude.

public:
	/**
	 * Constructor.
	 * @param [in] u Reference point's x-coordinate.
	 * @param [in] v Reference point's y-coordinate.
	 * @param [in] a Sine-wave amplitude.
	 */
	DistThetaFunctorDerivative( const double& u, const double& v, const double& a )
		: _u( u ), _v( v ), _a( a )
	{}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param [in] t Angle parameter.
	 * @return Function value.
	 */
	double operator()( const double& t ) const
	{
		return t - _u + _a * cos( t ) * ( _a * sin( t ) - _v );
	}

	/**
	 * The second derivative of the distance function to minimize.
	 * @param [in] t Angle parameter.
	 * @return Second derivative of distance function.
	 */
	double secondDerivative( const double& t ) const
	{
		return 1 - _a * sin( t ) * ( _a * sin( t ) - _v ) + SQR( _a * cos( t ) );
	}
};

/**
 * Class to find the parametric value for the projection of a node onto a sine-wave level-set function using Boost's
 * Newton-Rapson root finder.
 * The problem is simplified to finding the distance between (u,v) and f(x) = a * sin(x).  When calling using this
 * functor, make sure that the original sine frequency has been absorved into u, v, and a.  That is, the minimal
 * distance between (u,v) and f(x) = a*sin(w*t) is the same as 1/w times the minimal distance between (u*w, v*w) and
 * f(x') = w*a*sin(x').
 */
class DistThetaFunctorDerivativeNR
{
private:
	DistThetaFunctorDerivative _distThetaFunctorDerivative;
public:
	/**
	 * Constructor.
	 * @param distThetaFunctorDerivative The functor used for bisection.
	 */
	explicit DistThetaFunctorDerivativeNR( DistThetaFunctorDerivative distThetaFunctorDerivative )
		: _distThetaFunctorDerivative( distThetaFunctorDerivative )
	{}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param [in] t Angle parameter.
	 * @return A pair of a function evaluation and its derivative.
	 */
	std::pair<double, double> operator()( const double& t ) const
	{
		// Function which we need the roots of.
		double g = _distThetaFunctorDerivative( t );

		// And its derivative.
		double h = _distThetaFunctorDerivative.secondDerivative( t );
		return std::make_pair( g, h );
	}
};

/**
 * Obtain the parameter value that minimizes the distance between (u,v) and the sine wave interface using Newton-Raphson.
 * @param [in] u X-coordinate of query point.
 * @param [in] v Y-coordinate of query point.
 * @param [in] sine Reference to arclength-parameterized sine wave object.
 * @param [in] gen Random number generator.
 * @param [in] normalDistribution A normal random distribution generator.
 * @param [out] valOfDerivative Value of the derivative given the best theta angle (expected ~ 0).
 * @param [out] minDistance Minimum distance found between query point and sinusoid.
 * @param [in] verbose Whether to show debugging message or not.
 * @return Angle value that minimizes the distance from sine-wave and query point (u,v).
 */
double distThetaDerivative( p4est_locidx_t n, double u, double v, const ArcLengthParameterizedSine& sine,
	std::mt19937& gen, std::normal_distribution<double>& normalDistribution,
	double& valOfDerivative, double& minDistance, bool verbose = true )
{
	using namespace boost::math::tools;						// For bisect and newton_raphson_iterate.

	const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>( digits * 0.75 );    	// Accuracy doubles with each step, so stop when we have
															// just over half the digits correct.
	const boost::uintmax_t MAX_IT = 20;						// Maximum number of iterations for bracketing and root finding.
	boost::uintmax_t it = MAX_IT;
	const double A = sine.getOmega() * sine.getA();			// Convert constants for simpler f(x) = A * sin(x).
	const double U = u * sine.getOmega();
	const double V = v * sine.getOmega();
	DistThetaFunctorDerivative distThetaFunctorDerivative( U, V, A );							// Used for bisection (narrowing intervals).
	DistThetaFunctorDerivativeNR distThetaFunctorDerivativeNR( distThetaFunctorDerivative );	// Used for Newton-Raphson's method.
	eps_tolerance<double> epsToleranceFunctor( 10 );											// Used in bisection process (x bits).

	// Algorithm to find the parameter that minimizes the distance.  Based on the post made on Math Stack Exchange:
	// https://math.stackexchange.com/questions/2638377/distance-between-point-and-sine-again?noredirect=1&lq=1.
	const double SQRT_A_V = MAX( sqrt( SQR( ABS( A ) + ABS( V ) ) - SQR( ABS( A ) - ABS( V ) ) ), M_PI_2 );
	const double U_LOWER_B = U - SQRT_A_V;					// Lower and upper bound for U where
	const double U_UPPER_B = U + SQRT_A_V;					// solution is guaranteed.

	std::vector<double> arcSinValues;						// Solve for sinx: 1 - Asinx(Asinx - V) + (Acosx)^2.
	const double SQRT_A_V_SINX = sqrt( SQR( V * A ) + 8 * SQR( A ) * ( SQR( A ) + 1 ) );
	double sinx = ( -V * A + SQRT_A_V_SINX ) / ( -4 * SQR( A ) );
	if( ABS( sinx ) <= 1 )									// Given sinx = #, solve for x -> u value.
		arcSinValues.push_back( asin( sinx ) );
	sinx = ( -V * A - SQRT_A_V_SINX ) / ( -4 * SQR( A ) );
	if( ABS( sinx ) <= 1 )
		arcSinValues.push_back( asin( sinx ) );

	std::vector<double> checkpoints;						// U values/locations of extrema of first derivative.
	for( const double& u0 : arcSinValues )
	{
		int kMin = floor( ( U_LOWER_B - u0 ) / ( 2 * M_PI ) );	// Trying to find a range for u* = 2k\pi + u0, where
		int kMax = ceil( ( U_UPPER_B - u0 ) / ( 2 * M_PI ) );	// u* is the parameter that minimizes our function.
		for( int k = kMin; k <= kMax; k++ )
		{
			double u1 = 2 * M_PI * k + u0;					// The two paramters with the same sin(u_k) value.
			double u2 = 2 * M_PI * k + SIGN( u0 ) * M_PI - u0 ;
			if( u1 > U_LOWER_B && u1 < U_UPPER_B )			// Consider u values in the range of uMin and uMax,
				checkpoints.push_back( u1 );
			if( u2 > U_LOWER_B && u2 < U_UPPER_B )
				checkpoints.push_back( u2 );
		}
	}

	if( !checkpoints.empty() )								// Sort the list of u points in ascending order.
		std::sort( checkpoints.begin(), checkpoints.end() );
	checkpoints.insert( checkpoints.begin(), U_LOWER_B );
	checkpoints.push_back( U_UPPER_B );						// Add uMin and uMax at the front and end of list.

	std::vector<std::pair<int, int>> intervals;				// List of pairs of indices (i,j) referring to successive
	for( int i = 0; i < checkpoints.size() - 1; i++ )		// parameters stored in checkpoints, where g(checkpoints[i])
	{														// * g(checkpoints[j]) < 0, and g is the derivative of the
		if( distThetaFunctorDerivative( checkpoints[i] ) *	// distance function to minimize.
			distThetaFunctorDerivative( checkpoints[i+1] ) < 0 )
			intervals.emplace_back( i, i + 1 );
	}

	if( !intervals.empty() )								// Do we have parameters where the minimum is found?
	{
		double minD = PETSC_MAX_REAL;						// Keep track of min squared distance and its associated
		double minU = 0;									// parameter.

		for( const auto& interval : intervals )
		{
			double uStart = checkpoints[interval.first];	// Find the best parameter within the bracket first reducing
			double uEnd = checkpoints[interval.second];		// the interval with bisection, and then using Newton-Raphson's
															// to refine the root.
			int iterations = 0;
			it = MAX_IT;
			std::pair<double, double> uPair = bisect( distThetaFunctorDerivative, uStart, uEnd, epsToleranceFunctor, it );
			double uRange = uPair.second - uPair.first;
			double vOfD = 1;

			while( vOfD > 1e-8 )							// For standing interval, keep iterating until convergence.
			{												// Now that we have a narrow bracket, use Newton-Raphson's.
				double uResult = ( uPair.first + uPair.second ) / 2;		// Normally-randomize initial guess around
				uResult += normalDistribution( gen ) * ( uRange / 6 );		// midpoint.
				uResult = MAX( uPair.first, MIN( uResult, uPair.second ) );
				it = MAX_IT;
				uResult = newton_raphson_iterate( distThetaFunctorDerivativeNR, uResult, uPair.first, uPair.second, get_digits, it );
				vOfD = ABS( distThetaFunctorDerivative( uResult ) );

				if( verbose && it >= MAX_IT )
					std::cerr << "Node " << n << ":  Unable to locate solution in " << MAX_IT << " iterations!" << std::endl;

				double dx = U - uResult;
				double dy = V - A * sin( uResult );
				double d = SQR( dx ) + SQR( dy );
				if( d < minD )								// Don't discard the possibility that we may have distinct
				{											// valid intervals.  If so, get the one with the parameter
					minD = d;								// that yields the miminum distance among all.
					minU = uResult;
				}
				iterations++;
			}

			if( verbose && iterations > 20 )
				std::cerr << "Node " << n << ":  High number of convergence iterations " << iterations << std::endl;
		}

		valOfDerivative = distThetaFunctorDerivative( minU );
		minDistance = sqrt( minD ) / sine.getOmega();
		return minU / sine.getOmega();
	}
	else
	{
		throw std::runtime_error( "Node " + std::to_string( n ) + ":  Unable to find pairs of U values with derivative "
								  "values of distinct sign!" );
	}

	return PETSC_MAX_REAL;
}

#endif //FAST_SWEEPING_ARCLENGTH_PARAMETERIZED_SINE_2D_H
