//
// Created by Im YoungMin on 11/4/19.
// Based on https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/root_finding_examples/cbrt_eg.html
//

#include "boost/math/tools/roots.hpp"
#include <iostream>
#include <limits>

const double _a = -0.4;
const double _b = M_PI;
const int _p = 8;

/**
 * Flower's r(theta) function.
 * @param theta: Angle parameter in [0, 2*pi].
 * @return r(theta).
 */
double _r( double theta )
{
	return _a * cos( _p * theta ) + _b;
}

struct DistThetaFunctorNoDerivative
{
	// Finding roots of derivative of norm of parameterized flower and point (x, y).
	DistThetaFunctorNoDerivative( const double& x, const double& y ) : qx( x ), qy( y )
	{}

	// Evaluate functor at angle parameter value.
	double operator()( const double& theta )
	{
		double r = _r( theta );
		double rPrime = -_a * _p * sin( _p * theta );
		return rPrime * r + sin( theta ) * ( qx * r - qy * rPrime ) - cos( theta ) * ( qx * rPrime + qy * r );
	}

private:
	double qx;
	double qy;
};

/**
 * Obtain the theta parameter value that minimizes the distance between (qx, qy) and the flower-shaped interface.
 * @param x: X-coordinate of query point.
 * @param y: Y-coordinate of query point.
 * @param initialGuess: Initial angular guess.
 * @return Angle value.
 */
double distThetaNoDerivative( double x, double y, double initialGuess = 0 )
{
	using namespace boost::math::tools;						// For bracket_and_solve_root.

	double factor = 2;										// How big steps to take when searching.

	const boost::uintmax_t maxit = 20;						// Limit to maximum iterations.
	boost::uintmax_t it = maxit;							// Initially our chosen max iterations, but updated with actual.
	bool isRising = true;									// If guess is too low, then try increasing guess.
	int digits = std::numeric_limits<float>::digits;		// Maximum possible binary digits accuracy for type double.
															// Some fraction of digits is used to control how accurate to try to make the result.
	int getDigits = digits - 3;								// We have to have a non-zero interval at each step, so
															// maximum accuracy is digits - 1.  But we also have to
															// allow for inaccuracy in f( qx, qy ), otherwise the last few
															// iterations just thrash around.
	eps_tolerance<float> tol( getDigits );					// Set the tolerance.
	std::pair<double, double> r = bracket_and_solve_root( DistThetaFunctorNoDerivative( x, y ), initialGuess, factor, isRising, tol, it );

	if( it >= maxit )
	{
		std::cerr << "Unable to locate solution in " << maxit << " iterations: "
				     "Current best guess is between " << r.first << " and " << r.second << std::endl;
	}
	else
		std::cout << "Converged after " << it << " (from maximum of " << maxit << " iterations)." << std::endl;

	return r.first + ( r.second - r.first ) / 2;      		// Midway between brackets is our result, if necessary we could
															// return the result as an interval here.
}

/**
 * How to find the roots of a function using the Boost library.
 */
int main( int argc, char **args )
{
	double q[] = {-3, 1};
	try
	{
		double theta = distThetaNoDerivative( q[0], q[1], M_PI_4 );
		std::cout << "Theta = " << theta << std::endl;
		double x = _r( theta ) * cos( theta );
		double y = _r( theta ) * sin( theta );
		std::cout << "Point = [" << x << "], [" << y << "]" << std::endl;
	}
	catch( const std::exception& e )
	{
		std::cerr << "Message from thrown exception was:\n" << e.what() << std::endl;
	}

	return 0;
}