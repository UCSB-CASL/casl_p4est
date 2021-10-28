//
// Created by Im YoungMin on 11/17/19.
//

#ifndef FAST_SWEEPING_STAR_THETA_ROOT_FINDING_H
#define FAST_SWEEPING_STAR_THETA_ROOT_FINDING_H

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <vector>
#include <utility>
#include <src/casl_geometry.h>
#include <boost/math/tools/roots.hpp>
#include <limits>
#include <random>


class DistThetaFunctorDerivative_Star
{
private:
	double _x, _y;					// Cartesian coordinates of query point.
	const geom::Star& _star;

public:
	/**
	 * Finding roots of derivative of norm of parameterized star and point (x, y).
	 * @param [in] x Reference point's x-coordinate.
	 * @param [in] y Reference point's y-coordinate.
	 * @param [in] star Star object.
	 */
	DistThetaFunctorDerivative_Star( const double& x, const double& y, const geom::Star& star )
		: _x( x ), _y( y ), _star( star )
	{}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param [in] theta Angle parameter.
	 * @return Function value.
	 */
	double operator()( const double& theta ) const
	{
		double r = _star.r( theta );
		double rPrime = _star.rPrime( theta );
		return _x * (r * sin( theta ) - rPrime * cos( theta )) - _y * (r * cos( theta ) + rPrime * sin( theta )) + r * rPrime;
	}

	/**
	 * The second derivative of the distance function to minimize.
	 * @param [in] t Angle parameter.
	 * @return Second derivative of distance function.
	 */
	double secondDerivative( const double& theta ) const
	{
		double r = _star.r( theta );
		double rPrime = _star.rPrime( theta );
		double rPrimePrime = _star.rPrimePrime( theta );

		return sin( theta ) * (_y * r + 2 * _x * rPrime - _y * rPrimePrime) +
			   cos( theta ) * (_x * r - 2 * _y * rPrime - _x * rPrimePrime) +
			   r * rPrimePrime + SQR( rPrime );
	}
};


/**
 * Class to find the angular value for the projection of a node onto a star-shaped level-set function using Boost's
 * Newton-Raphson root finder.
 */
class DistThetaFunctorDerivativeNR_Star
{
private:
	DistThetaFunctorDerivative_Star _distThetaFunctorDerivative;
public:
	/**
	 * Constructor.
	 * @param distThetaFunctorDerivative The functor used for bisection.
	 */
	explicit DistThetaFunctorDerivativeNR_Star( DistThetaFunctorDerivative_Star distThetaFunctorDerivative )
		: _distThetaFunctorDerivative( distThetaFunctorDerivative )
	{}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param [in] theta Angle parameter.
	 * @return A pair of a function evaluation and its derivative.
	 */
	std::pair<double, double> operator()( const double& theta ) const
	{
		// Function which we need the roots of.
		double g = _distThetaFunctorDerivative( theta );

		// And its derivative.
		double h = _distThetaFunctorDerivative.secondDerivative( theta );
		return std::make_pair( g, h );
	}
};


/**
 * Obtain the parameter value that minimizes the distance between (u,v) and the star-shaped interface using bisection
 * and Newton-Raphson.
 * @param [in] u X-coordinate of query point.
 * @param [in] v Y-coordinate of query point.
 * @param [in] star Reference to star object.
 * @param [in] INITIAL_THETA Initial angle to kick start the process.  Must be given in [0, 2*pi).
 * @param [in] H Spatial interval.
 * @param [in] gen Random number generator.
 * @param [in] normalDistribution A normal random distribution generator.
 * @param [out] valOfDerivative Value of the derivative given the best theta angle (expected ~ 0).
 * @param [out] minDistance Minimum distance found between query point and star-shaped interface.
 * @param [in] verbose Whether to show debugging message or not.
 * @return Angle value that minimizes the distance from query point (u,v) to star-shaped interface.
 */
double distThetaDerivative_Star( p4est_locidx_t n, double u, double v, const geom::Star& star,
							const double INITIAL_THETA, const double H,
							std::mt19937& gen, std::normal_distribution<double>& normalDistribution,
							double& valOfDerivative, double& minDistance, bool verbose = true )
{
	using namespace boost::math::tools;						// For bisect and newton_raphson_iterate.

	const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>( digits * 0.75 );		// Accuracy doubles with each step, so stop when we have
															// just over half the digits correct.
	const boost::uintmax_t MAX_IT = 30;						// Maximum number of iterations for bracketing and root finding.
	boost::uintmax_t it = MAX_IT;
	DistThetaFunctorDerivative_Star distThetaFunctorDerivative( u, v, star );    // Used for bisection (narrowing intervals).
	DistThetaFunctorDerivativeNR_Star distThetaFunctorDerivativeNR( distThetaFunctorDerivative );	// Used for Newton-Raphson's method.
	eps_tolerance<double> epsToleranceFunctor( 10 );		// Used in bisection process (x bits).

	// Algorithm to find the parameter that minimizes the distance.  Based on the post made on Math Stack Exchange:
	// https://math.stackexchange.com/questions/2638377/distance-between-point-and-sine-again?noredirect=1&lq=1, but
	// adapted for star.
	const double T_LOWER_B = INITIAL_THETA - M_PI_4 / 4;	// Lower and upper bound for theta where
	const double T_UPPER_B = INITIAL_THETA + M_PI_4 / 4 ;	// solution is guaranteed.

	// Find intervals where the function we want to find the roots of changes sign.  That means there's a root in there.
	std::vector<std::pair<double, double>> intervals;
	double t = T_LOWER_B;
	while( t < T_UPPER_B )
	{
		if( distThetaFunctorDerivative( t ) * distThetaFunctorDerivative( t + H ) <= 0 )
			intervals.emplace_back( t, t + H );
		t += H;
	}

	if( !intervals.empty() )						// Do we have parameters where a minimum is found?
	{
		double minD = PETSC_MAX_REAL;				// Keep track of min squared distance and its associated angle.
		double minT = 0;

		for( const auto &interval : intervals )
		{
			double tStart = interval.first;			// Find the best parameter within the bracket first reducing
			double tEnd = interval.second;			// the interval with bisection, and then using Newton-Raphson's
													// to refine the root.
			int iterations = 0;
			it = MAX_IT;
			std::pair<double, double> tPair = bisect( distThetaFunctorDerivative, tStart, tEnd, epsToleranceFunctor, it );
			double tRange = tPair.second - tPair.first;
			double vOfD = 1;
			double tResult;

			while( vOfD > 1e-8 )					// For standing interval, keep iterating until convergence.
			{										// Now that we have a narrow bracket, use Newton-Raphson's.
				tResult = (tPair.first + tPair.second) / 2;	// Normally-randomize initial guess around midpoint.
				tResult += normalDistribution( gen ) * (tRange / 4);
				tResult = MAX( tPair.first, MIN( tResult, tPair.second ) );
				it = MAX_IT;
				tResult = newton_raphson_iterate( distThetaFunctorDerivativeNR, tResult, tPair.first, tPair.second,
												  get_digits, it );
				vOfD = ABS( distThetaFunctorDerivative( tResult ) );

				if( verbose && it >= MAX_IT )
					std::cerr << "Node " << n << ":  Unable to locate solution in " << MAX_IT << " iterations!" << std::endl;
				iterations++;
			}

			double dx = u - star.r( tResult ) * cos( tResult );
			double dy = v - star.r( tResult ) * sin( tResult );
			double d = SQR( dx ) + SQR( dy );
			if( d < minD )							// Don't discard the possibility that we may have distinct
			{										// valid intervals.  If so, get the one with the parameter
				minD = d;							// that yields the miminum distance among all.
				minT = tResult;
			}

			if( verbose && iterations > 20 )
				std::cerr << "Node " << n << ":  High number of convergence iterations " << iterations << std::endl;
		}

		valOfDerivative = distThetaFunctorDerivative( minT );
		minDistance = sqrt( minD );
		return minT;
	}
	else
	{
		throw std::runtime_error( "Node " + std::to_string( n ) + ":  Unable to find pairs of U values with derivative "
																  "values of distinct sign!" );
	}
}

#endif //FAST_SWEEPING_STAR_THETA_ROOT_FINDING_H
