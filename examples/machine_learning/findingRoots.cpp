//
// Created by Im YoungMin on 11/4/19.
// Based on https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/root_finding_examples/cbrt_eg.html
//

#include "boost/math/tools/roots.hpp"
#include <iostream>
#include <limits>

const double _a = -1;
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

struct DistThetaFunctorDerivative
{
	// Finding roots of derivative of norm of parameterized flower and point (x, y).
	DistThetaFunctorDerivative( const double& x, const double& y ) : qx( x ), qy( y )
	{}

	// Evaluate functor at angle parameter value.
	std::pair<double, double> operator()( const double& theta )
	{
		double r = _r( theta );
		double rPrime = -_a * _p * sin( _p * theta );
		double rPrimePrime = -_a * _p * _p * cos( _p * theta );
		double fTheta = rPrime * r + sin( theta ) * ( qx * r - qy * rPrime ) - cos( theta ) * ( qx * rPrime + qy * r );		// Function which we need the zeroes of.
		double fPrimeTheta = rPrime * rPrime + rPrime * rPrimePrime
				+ sin( theta ) * ( -qy * rPrimePrime + 2 * qx * rPrime + qy * r )
				+ cos( theta ) * ( -qx * rPrimePrime - 2 * qy * rPrime + qx * r );
		return std::make_pair( fTheta, fPrimeTheta );
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
 * Obtain the theta parameter value that minimizes the distance between (qx, qy) and the flower-shaped interface using Newton-Rapson method.
 * @param x: X-coordinate of query point.
 * @param y: Y-coordinate of query point.
 * @param initialGuess: Initial angular guess.
 * @param minimum: Minimum value for search interval.
 * @param maximum: Maximum value for search interval.
 * @return Angle value.
 */
double distThetaDerivative( double x, double y, double initialGuess = 0, double minimum= -1, double maximum = +1 )
{
	using namespace boost::math::tools;						// For bracket_and_solve_root.

	const int digits = std::numeric_limits<double>::digits;	// Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>(digits * 0.6);    	// Accuracy doubles with each step, so stop when we have
															// just over half the digits correct.
	const boost::uintmax_t maxit = 50;						// Maximum number of iterations.
	boost::uintmax_t it = maxit;
	double result = newton_raphson_iterate( DistThetaFunctorDerivative( x, y ), initialGuess, minimum, maximum, get_digits, it);

	if( it >= maxit )
		std::cerr << "Unable to locate solution in " << maxit << " iterations!" << std::endl;
	else
		std::cout << "Converged after " << it << " (from maximum of " << maxit << " iterations)." << std::endl;

	return result;
}

/**
 * How to find the roots of a function using the Boost library.
 */
int main( int argc, char **args )
{
	double q[] = {1.5, 1.4};
	try
	{
		// Find the line segment closest to query point.  This becomes the search interval.
		int M = 10000;
		double stepSize = 2 * M_PI / ( M - 1 );
		int minIdx = -1;
		double minDSquare = std::numeric_limits<double>::infinity();
		double R[M];
		double T[M];
		double D2[M];
		for( int i = 0; i < M - 1; i++ )
		{
			T[i] = i * stepSize;
			R[i] = _r( T[i] );
			double x = R[i] * cos( T[i] ),		// Coordinates of current point on interface.
				   y = R[i] * sin( T[i] );
			double dx = x - q[0],				// Difference between query point and point on interface.
				   dy = y - q[1];
			D2[i] = dx * dx + dy * dy;			// Save squared distance to query point.
			if( D2[i] < minDSquare )			// Point on interface closest to query point.
			{
				minDSquare = D2[i];
				minIdx = i;
			}
		}
		R[M-1] = R[0];							// Close curve.
		D2[M-1] = D2[0];
		T[M-1] = 2 * M_PI;

		int neighborIdx = ( minIdx - 1 < 0 )? M - 2: minIdx - 1;		// Find the other end point of closest line segment.
		if( D2[minIdx + 1] < D2[neighborIdx] )
			neighborIdx = minIdx + 1;

		double min = std::min( T[minIdx], T[neighborIdx] );
		double max = std::max( T[minIdx], T[neighborIdx] );
		double theta = distThetaDerivative( q[0], q[1], ( min + max ) / 2, min, max );
		std::cout << "Theta = " << theta << std::endl;
		double x = _r( theta ) * cos( theta );
		double y = _r( theta ) * sin( theta );
		std::cout << "Point = [" << x << "], [" << y << "]" << std::endl;
		DistThetaFunctorNoDerivative d( q[0], q[1] );
		std::cout << d( theta ) << std::endl;
	}
	catch( const std::exception& e )
	{
		std::cerr << "Message from thrown exception was:\n" << e.what() << std::endl;
	}

	return 0;
}