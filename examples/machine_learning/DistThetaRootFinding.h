//
// Created by Im YoungMin on 11/17/19.
//

#ifndef MACHINE_LEARNING_DISTTHETAROOTFINDING_H
#define MACHINE_LEARNING_DISTTHETAROOTFINDING_H

#include "boost/math/tools/roots.hpp"
#include <limits>
#include "PointsOnCurves.h"

/**
 * Class to find the theta value for the projection of a grid point onto a flower using Newton-Rapson root finder.
 */
class DistThetaFunctorDerivative
{
private:
	double _qx;
	double _qy;
	const Flower& _flower;

public:
	/**
	 * Finding roots of derivative of norm of parameterized flower and point (x, y).
	 * @param x: Reference point's x-coordinate.
	 * @param y: Reference point's y-coordinate.
	 * @param flower: Flower object.
	 */
	DistThetaFunctorDerivative( const double& x, const double& y, const Flower& flower ) : _qx( x ), _qy( y ), _flower( flower )
	{}

	/**
	 * Evaluate functor at angle parameter value.
	 * @param theta: Angle parameter.
	 * @return A pair of function evaluation and its derivative.
	 */
	std::pair<double, double> operator()( const double& theta )
	{
		double r = _flower.r( theta );
		double rPrime = -_flower.getA() * _flower.getP() * sin( _flower.getP() * theta );
		double rPrimePrime = -_flower.getA() * _flower.getP() * _flower.getP() * cos( _flower.getP() * theta );
		double fTheta = rPrime * r + sin( theta ) * ( _qx * r - _qy * rPrime ) - cos( theta ) * ( _qx * rPrime + _qy * r );		// Function which we need the roots of.
		double fPrimeTheta = rPrime * rPrime + rPrime * rPrimePrime
							 + sin( theta ) * ( -_qy * rPrimePrime + 2 * _qx * rPrime + _qy * r )
							 + cos( theta ) * ( -_qx * rPrimePrime - 2 * _qy * rPrime + _qx * r );
		return std::make_pair( fTheta, fPrimeTheta );
	}

	/**
	 * Given a theta angle, compute the value of the derivative of half distance square.
	 * @param theta: Angle parameter.
	 * @return f(theta), the function for which we look for roots.
	 */
	[[nodiscard]] double valueOfDerivative( const double& theta ) const
	{
		double r = _flower.r( theta );
		double rPrime = -_flower.getA() * _flower.getP() * sin( _flower.getP() * theta );
		return rPrime * r + sin( theta ) * ( _qx * r - _qy * rPrime ) - cos( theta ) * ( _qx * rPrime + _qy * r );
	}
};

/**
 * Obtain the theta parameter value that minimizes the distance between (qx, qy) and the flower-shaped interface using Newton-Rapson method.
 * @param x: X-coordinate of query point.
 * @param y: Y-coordinate of query point.
 * @param flower: Reference to flower function.
 * @param valOfDerivative: Value of the derivative given the best theta angle (expected = 0).
 * @param initialGuess: Initial angular guess.
 * @param minimum: Minimum value for search interval.
 * @param maximum: Maximum value for search interval.
 * @return Angle value.
 */
double distThetaDerivative( double x, double y, const Flower& flower, double& valOfDerivative, double initialGuess = 0, double minimum= -1, double maximum = +1 )
{
	using namespace boost::math::tools;						// For bracket_and_solve_root.

	const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>( digits * 0.75 );    	// Accuracy doubles with each step, so stop when we have
	// just over half the digits correct.
	const boost::uintmax_t maxit = 20;						// Maximum number of iterations.
	boost::uintmax_t it = maxit;
	DistThetaFunctorDerivative distThetaFunctorDerivative( x, y, flower );
	double result = newton_raphson_iterate( distThetaFunctorDerivative, initialGuess, minimum, maximum, get_digits, it );

	if( it >= maxit )
		std::cerr << "Unable to locate solution in " << maxit << " iterations!" << std::endl;

	valOfDerivative = distThetaFunctorDerivative.valueOfDerivative( result );
	return result;
}

#endif //MACHINE_LEARNING_DISTTHETAROOTFINDING_H
