//
// Created by Im YoungMin on 11/17/19.
//

#ifndef FAST_SWEEPING_DISTTHETAROOTFINDING_H
#define FAST_SWEEPING_DISTTHETAROOTFINDING_H

#include "boost/math/tools/roots.hpp"
#include <limits>
#include <src/casl_geometry.h>

/**
 * Class to find the theta value for the projection of a node onto a 2D star using the Newton-Rapson root finder.
 */
class DistThetaFunctorDerivative
{
private:
	double _r;					// We need the query point in polar coordinates. r = radius,
	double _psi;				// and _psi is the angle with respect to horizontal.
	const geom::Star& _star;

public:
	/**
	 * Finding roots of derivative of norm of parameterized star and point (x, y).
	 * @param [in] x: Reference point's x-coordinate.
	 * @param [in] y: Reference point's y-coordinate.
	 * @param [in] star: Star object.
	 */
	DistThetaFunctorDerivative( const double& x, const double& y, const geom::Star& star )
		: _star( star )
	{
		_r = sqrt( SQR( x ) + SQR( y ) );
		_psi = atan2( y, x );
		if( _psi < 0 )
			_psi = 2 * M_PI + _psi;
	}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param [in] theta: Angle parameter.
	 * @return A pair of a function evaluation and its derivative.
	 */
	std::pair<double, double> operator()( const double& theta )
	{
		double r = _star.r( theta );
		double rPrime = -_star.getA() * _star.getP() * sin( _star.getP() * theta );
		double rPrimePrime = -_star.getA() * SQR( _star.getP() ) * cos( _star.getP() * theta );

		// Function which we need the roots of.
		double fTheta = rPrime * ( r - _r * cos( _psi - theta ) ) - r * _r * sin( _psi - theta );

		// And its derivative.
		double fPrimeTheta = r * rPrimePrime + SQR( rPrime ) + _r * ( r - rPrimePrime ) * cos( _psi - theta ) -
							 2 * _r * rPrime * sin( _psi - theta );
		return std::make_pair( fTheta, fPrimeTheta );
	}

	/**
	 * Given an angle, compute the value of the derivative of half distance square.
	 * @param [in] theta: Angle parameter.
	 * @return f(theta), the function which we want to find the roots of.
	 */
	[[nodiscard]] double valueOfDerivative( const double& theta ) const
	{
		double r = _star.r( theta );
		double rPrime = -_star.getA() * _star.getP() * sin( _star.getP() * theta );
		return rPrime * ( r - _r * cos( _psi - theta ) ) - r * _r * sin( _psi - theta );
	}
};

/**
 * Obtain the theta value that minimizes the distance between (x, y) and the star-shaped interface using Newton-Rapson.
 * @param [in] x: X-coordinate of query point.
 * @param [in] y: Y-coordinate of query point.
 * @param [in] star: Reference to star object.
 * @param [out] valOfDerivative: Value of the derivative given the best theta angle (expected = 0).
 * @param [in] initialGuess: Initial angular guess.
 * @param [in] minimum: Minimum value for search interval.
 * @param [in] maximum: Maximum value for search interval.
 * @return Angle value.
 */
double distThetaDerivative( double x, double y, const geom::Star& star, double& valOfDerivative,
							double initialGuess = 0, double minimum= -1, double maximum = +1 )
{
	using namespace boost::math::tools;						// For bracket_and_solve_root.

	const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>( digits * 0.75 );    	// Accuracy doubles with each step, so stop when we have
															// just over half the digits correct.
	const boost::uintmax_t maxit = 20;						// Maximum number of iterations.
	boost::uintmax_t it = maxit;
	DistThetaFunctorDerivative distThetaFunctorDerivative( x, y, star );
	double result = newton_raphson_iterate( distThetaFunctorDerivative, initialGuess, minimum, maximum, get_digits, it );

	if( it >= maxit )
		std::cerr << "Unable to locate solution in " << maxit << " iterations!" << std::endl;

	valOfDerivative = distThetaFunctorDerivative.valueOfDerivative( result );
	return result;
}

#endif //FAST_SWEEPING_DISTTHETAROOTFINDING_H
