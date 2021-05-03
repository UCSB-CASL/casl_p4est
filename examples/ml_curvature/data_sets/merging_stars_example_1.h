//
// Created by Im YoungMin on 08/16/20.
//

#ifndef FAST_SWEEPING_MERGING_STARS_EXAMPLE_1_H
#define FAST_SWEEPING_MERGING_STARS_EXAMPLE_1_H

#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_macros.h>
#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <vector>
#include <utility>
#include <src/casl_geometry.h>
#include <boost/math/tools/roots.hpp>
#include <limits>
#include <random>


class DistThetaFunctorDerivative
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
	DistThetaFunctorDerivative( const double& x, const double& y, const geom::Star& star )
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
 * Level-set function formed by two 5-armed stars merging at two of their arms.  The left star is sharp while the right
 * one is smooth and smaller.  See ML_Curvature_3.0/MergingStarsExample1.ipynb for details:
 * Star 1) r1(t) = 0.12 * cos(5t) + 0.305
 * Star 2) r2(t) = 0.07 * cos(5t - pi) + 0.3
 */
class MergingStarsExample1 : public CF_2
{
private:
	const Point2 _c1;						// Star origins of their respective local coordinate systems.
	const Point2 _c2;
	const geom::Star _star1;				// The left and right stars.
	const geom::Star _star2;

	std::mt19937& _gen; 									// Standard mersenne_twister_engine.
	std::normal_distribution<double>& _normalDistribution;	// Normal distribution used for bracketing and root finding.
	const double _h;										// Minimum cell width.

	double _tIntervals1[2][2]{};			// Intersection t-value intervals with respect to left star.
	double _tIntervals2[2][2]{};			// Intersection t-value intervals with respect to right star.
	Point2 _intersectionCoords[2][2]{};		// Intersection points in world coordinates.

private:
	/**
	 * Wrap angle to [0, 2pi).
	 * @param [in] angle The angle in radians.
	 * @return Equivalent angle in the range [0, 2pi).
	 */
	static double _wrapAngle( double angle )
	{
		double twoPi = 2 * M_PI;
		return angle - twoPi * floor( angle / twoPi );
	}

	/**
	 * Obtain the parameter value that minimizes the distance between (u,v) and the star-shaped interface using
	 * bisection and Newton-Raphson.
	 * @param [in] n Node ID.
	 * @param [in] u X-coordinate of query point with respect to standing star.
	 * @param [in] v Y-coordinate of query point with respect to standing star.
	 * @param [in] star Reference to standing star object.
	 * @param [in] INITIAL_THETA Initial angle to kick start the process.  Must be given in [0, 2*pi).
	 * @param [out] valOfDerivative Value of the derivative given the best theta angle (expected ~ 0).
	 * @param [out] minDistance Minimum distance found between query point and star-shaped interface.
	 * @param [in] verbose Whether to show debugging message or not.
	 * @return Angle value that minimizes the distance from query point (u,v) to standing star-shaped interface.
	 */
	double _distThetaDerivative( p4est_locidx_t n, double u, double v, const geom::Star& star,
								const double INITIAL_THETA,
								double& valOfDerivative, double& minDistance, bool verbose = true ) const
	{
		using namespace boost::math::tools;						// For bisect and newton_raphson_iterate.

		const int digits = std::numeric_limits<float>::digits;	// Maximum possible binary digits accuracy for type T.
		int get_digits = static_cast<int>( digits * 0.75 );		// Accuracy doubles with each step, so stop when we have
																// just over half the digits correct.
		const boost::uintmax_t MAX_IT = 20;						// Maximum number of iterations for bracketing and root finding.
		boost::uintmax_t it = MAX_IT;
		DistThetaFunctorDerivative distThetaFunctorDerivative( u, v, star );    // Used for bisection (narrowing intervals).
		DistThetaFunctorDerivativeNR distThetaFunctorDerivativeNR( distThetaFunctorDerivative );	// Used for Newton-Raphson's method.
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
			if( distThetaFunctorDerivative( t ) * distThetaFunctorDerivative( t + _h ) <= 0 )
				intervals.emplace_back( t, t + _h );
			t += _h;
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
					tResult += _normalDistribution( _gen ) * (tRange / 6);
					tResult = MAX( tPair.first, MIN( tResult, tPair.second ) );
					it = MAX_IT;
					tResult = newton_raphson_iterate( distThetaFunctorDerivativeNR, tResult, tPair.first, tPair.second,
													  get_digits, it );
					vOfD = ABS( distThetaFunctorDerivative( tResult ) );

					if( verbose && it >= MAX_IT )
					{
						std::cerr << "Node " << n << ":  Unable to locate solution in " << MAX_IT << " iterations!"
								  << std::endl;
					}
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
			throw std::runtime_error( "Node " + std::to_string( n ) + ":  Unable to find pairs of U values with "
									  "derivative values of distinct sign!" );
		}
	}

	/**
	 * Compute the (positive) distance of a point to one of the merging star interfaces.
	 * @param [in] n Query node ID.
	 * @param [in] star Reference to standing star-shaped interface (left or right).
	 * @param [in] c World center coordinates for standing star.
	 * @param [in] tIntervals Array of intervals for t-values with respect to standing star.
	 * @param [in] q Query point world coordinates.
	 * @param [in] localX Local x-coordinate of query point.
	 * @param [in] localY Local y-coordinate of query point.
	 * @param [in] theta Initial angle to kick start the root finding process.
	 * @param [in,out] xOnGamma World x-coordinate of closest point to q on the star interface.
	 * @param [in,out] yOnGamma World y-coordinate of closest point to q on the star interface.
	 * @param [out] hk Dimensionless curvature.
	 * @param [in] altThetaRange Whether we decide to use the alternative angular range of [-pi, +pi).
	 * @return Positive distance to interface or to intersection points in one of the two blobs.
	 */
	[[nodiscard]] double _distanceToStar( const p4est_locidx_t n, const geom::Star& star, const Point2& c,
									   	  const double tIntervals[][2], const Point2& q,
									   	  const double localX, const double localY, double theta,
									   	  double& xOnGamma, double& yOnGamma, double& hk, bool altThetaRange=false ) const
	{
		double valOfDerivative = 1, distance = -1, d;
		theta = _distThetaDerivative( n, localX, localY, star, theta, valOfDerivative, distance );
		theta = _wrapAngle( theta );										// Ensuring angle is in [0, 2pi).
		theta = (altThetaRange && theta >= M_PI)? theta - 2 * M_PI : theta;	// Angle in [-pi, +pi) if alternative range flag is on.

		if( theta >= tIntervals[0][0] && theta <= tIntervals[0][1] )		// Is projection in range of first blob?
		{
			distance = (q - _intersectionCoords[0][0]).norm_L2();
			xOnGamma = _intersectionCoords[0][0].x; 						// Recall coords of closest point.
			yOnGamma = _intersectionCoords[0][0].y;
			d = (q - _intersectionCoords[0][1]).norm_L2();
			if( d < distance )
			{
				distance = d;
				xOnGamma = _intersectionCoords[0][1].x;
				yOnGamma = _intersectionCoords[0][1].y;
			}

			hk = -1;		// Indicating that point lies in a discontinuity region.
		}
		else if( theta >= tIntervals[1][0] && theta <= tIntervals[1][1] )	// Is projection in range of second blob?
		{
			distance = (q - _intersectionCoords[1][0]).norm_L2();
			xOnGamma = _intersectionCoords[1][0].x;							// Recall coords of closest point.
			yOnGamma = _intersectionCoords[1][0].y;
			d = (q - _intersectionCoords[1][1]).norm_L2();
			if( d < distance )
			{
				distance = d;
				xOnGamma = _intersectionCoords[1][1].x;
				yOnGamma = _intersectionCoords[1][1].y;
			}

			hk = -1;		// Indicating that point lies in a discontinuity region.
		}
		else			// Projection point doesn't fall in any of the bounds of the intersection points.
		{
			double r = star.r( theta );
			Point2 p( cos( theta ) * r + c.x, sin( theta ) * r + c.y );
			xOnGamma = p.x;													// Recall coords of closest point.
			yOnGamma = p.y;
			distance = (q - p).norm_L2();

			hk = _h * star.curvature( theta );
		}

		return distance;
	}

public:
	/**
	 * Constructor.
	 * @param [in] gen Random number generator.
	 * @param [in] normalDistribution A normal random distribution generator.
	 * @param [in] H Minimum cell width.
	 */
	MergingStarsExample1( std::mt19937& gen, std::normal_distribution<double>& normalDistribution, const double H )
	: _c1( -0.3, -0.15 ), _c2( 0.2, 0.2 ), _star1( 0.12, 0.305, 5 ), _star2( 0.07, 0.3, 5, M_PI ), _gen( gen ),
	_normalDistribution( normalDistribution ), _h( H )
	{
		// t-value intervals for left star. Valid t-values lie in the range [-pi, +pi).
		_tIntervals1[0][0] = 0.9824811;   _tIntervals1[0][1] = 1.21498458;		// First intersection blob.
		_tIntervals1[1][0] = -0.00650633; _tIntervals1[1][1] = 0.25939204;		// Second intersection blob.

		// t-value intervals for right star.  Valid t-values lie in the range [0, 2pi).
		_tIntervals2[0][0] = 3.01209996;		   _tIntervals2[0][1] = 2 * M_PI - 2.90532594;	// First intersection blob.
		_tIntervals2[1][0] = 2 * M_PI - 2.1540726; _tIntervals2[1][1] = 2 * M_PI - 1.78047999;	// Second intersection blob.

		// Intersection coordinates for two blobs.
		// Blob 1.
		_intersectionCoords[0][0] = Point2( -0.15285393939252187, 0.2459491253185989 );		// P1.
		_intersectionCoords[0][1] = Point2( -0.11750560363278095, 0.12355626080509488 );	// P2.
		// Blob 2.
		_intersectionCoords[1][0] = Point2( 0.026162131164882363, -0.06344613794240188 );	// P3.
		_intersectionCoords[1][1] = Point2( 0.12492751288926313, -0.15276475708496234 );	// P4.
	}

	/**
	 * Get a reference to center coordinates of one of the two stars.
	 * @param [in] i Star index: 0 left, 1 right.
	 * @return Center point.
	 */
	[[nodiscard]] const Point2& getCenter( short i ) const
	{
		switch( i )
		{
			case 0: return _c1;		// Left star.
			case 1: return _c2;		// Right star.
			default: throw std::runtime_error( "Valid indices are 0 and 1 only!" );
		}
	}

	/**
	 * Get a reference to one of the two stars.
	 * @param [in] i Star index: 0 left, 1 right.
	 * @return Star.
	 */
	[[nodiscard]] const geom::Star& getStar( short i ) const
	{
		switch( i )
		{
			case 0: return _star1;	// Left star.
			case 1: return _star2;	// Right star.
			default: throw std::runtime_error( "Valid indices are 0 and 1 only!" );
		}
	}

	/**
	 * Level-set evaluation at a given point in the 2D plane (e.g. world coordinates).
	 * This function follows the definition of the boolean union operation between two level-set functions described in
	 * [7]: phi(x) = min( phi_1(x), phi_2(x) ).
	 * @param [in] x Point x-coordinate.
	 * @param [in] y Point y-coordinate.
	 * @return phi(x,y).
	 */
	[[nodiscard]] double operator()( double x, double y ) const override
	{
		// phi_1(x): localizing point with respect to left (sharp) star.
		double x1 = x - _c1.x;
		double y1 = y - _c1.y;
		double phi1 = _star1( x1, y1 );

		// phi_2(x): localizing point with respect to right (smooth) star.
		double x2 = x - _c2.x;
		double y2 = y - _c2.y;
		double phi2 = _star2( x2, y2 );

		return MIN( phi1, phi2 );
	}

	/**
	 * Signed-distance level-set function evaluation at a given interface point in world coordinates.
	 * We mostly need the query points to be close to Gamma in order to reduce the possibility of divergence in the
	 * numerical process.
	 * Retrieve the curvature at the normal projection onto the closest interface point as well as which star this
	 * shortest distance and hk is due to: -1 for left star, 0 for both (discontinuity), +1 for right star.
	 * @param [in] n Node ID.
	 * @param [in] x Point x-coordinate.
	 * @param [in] y Point y-coordinate.
	 * @param [in,out] xOnGamma Approximated normal projection of x onto Gamma.
	 * @param [in,out] yOnGamma Approximated normal projection of y onto Gamma.
	 * @param [out] hk Dimensionless curvature of closest point on compound Gamma.
	 * @param [out] who -1 for left star, 0 for discontinuity, and +1 for right star.
	 * @return The signed distance from a point to the compound interface.
	 */
	[[nodiscard]] double getSignedDistance( p4est_locidx_t n, double x, double y, double& xOnGamma, double& yOnGamma,
										 	double& hk, short& who ) const
	{
		// First determine whether we need to compute exact distance with respect to one or both of the stars since the
		// operation is expensive and prone to diverge for points too far from any of the individual interfaces.
		double x1 = x - _c1.x;		// phi_1(x_Gamma): localizing projected point with respect to left (sharp) star.
		double y1 = y - _c1.y;
		double xOnGamma1 = xOnGamma - _c1.x;
		double yOnGamma1 = yOnGamma - _c1.y;
		double theta1 = _wrapAngle( atan2( yOnGamma1, xOnGamma1 ) );	// Initial projection angle.
		double phi1 = _star1( x1, y1 );

		double x2 = x - _c2.x;		// phi_2(x_Gamma): localizing projected point with respect to right (smooth) star.
		double y2 = y - _c2.y;
		double xOnGamma2 = xOnGamma - _c2.x;
		double yOnGamma2 = yOnGamma - _c2.y;
		double theta2 = _wrapAngle( atan2( yOnGamma2, xOnGamma2 ) );	// Initial projection angle.
		double phi2 = _star2( x2, y2 );

		double distance = PETSC_MAX_REAL;
		const Point2 q( x, y );

		// There are four scenarios we must evaluate.
		if( phi1 <= 0 && phi2 > 0 )				// Point lies inside left star only.
		{
			distance = _distanceToStar( n, _star1, _c1, _tIntervals1, q, x1, y1, theta1, xOnGamma, yOnGamma, hk, true );
			who = -1;
		}
		else if( phi2 <= 0 && phi1 > 0 )		// Point lies inside right star only.
		{
			distance = _distanceToStar( n, _star2, _c2, _tIntervals2, q, x2, y2, theta2, xOnGamma, yOnGamma, hk );
			who = +1;
		}
		else if( phi1 <= 0 && phi2 <= 0 )		// Point lies inside both interfaces: evaluate distances to intersection points only.
		{
			for( const auto& blob : _intersectionCoords)
			{
				for( const auto& point : blob )
				{
					double d = (point - q).norm_L2();	// Even though we iterate over all four intersection points,
					if( d < distance )					// it's cheaper to do so than using numerical methods to find
					{									// the optimal angle in each of the blobs and with respect to
						distance = d;					// each star.  We rely here on the fact that the blobs are too
						xOnGamma = point.x;				// far from one another so that the closest intersection point
						yOnGamma = point.y;				// can be retrieved with confidence.
					}
				}
			}

			hk = -1;							// Indicating the point lies in a discontinuity region.
			who = 0;
		}
		else									// Point is outside both stars: choose the minimum distance among all.
		{
			if( 3 * ABS( phi1 ) < ABS( phi2 )  )		// Make it cheap: only left star is needed.
			{
				distance = _distanceToStar( n, _star1, _c1, _tIntervals1, q, x1, y1, theta1, xOnGamma, yOnGamma, hk, true );
				who = -1;
			}
			else if( 3 * ABS( phi2 ) < ABS( phi1 ) )	// Make it cheap: only right star is needed.
			{
				distance = _distanceToStar( n, _star2, _c2, _tIntervals2, q, x2, y2, theta2, xOnGamma, yOnGamma, hk );
				who = +1;
			}
			else										// Point is closed to both stars from outside.
			{											// Note: I'm reusing xOnGamma# and yOnGamma# to avoid polluting global xOnGamma and yOnGamma.
				double hk1, hk2;
				double d1 = _distanceToStar( n, _star1, _c1, _tIntervals1, q, x1, y1, theta1, xOnGamma1, yOnGamma1, hk1, true );
				double d2 = _distanceToStar( n, _star2, _c2, _tIntervals2, q, x2, y2, theta2, xOnGamma2, yOnGamma2, hk2 );
				if( d1 < d2 )
				{
					distance = d1;
					xOnGamma = xOnGamma1;
					yOnGamma = yOnGamma1;
					hk = hk1;
					who = -1;
				}
				else if( d2 < d1 )
				{
					distance = d2;
					xOnGamma = xOnGamma2;
					yOnGamma = yOnGamma2;
					hk = hk2;
					who = +1;
				}
				else		// Exceptional case: Same distance from star1 and star2 (from outside)
				{
					distance = d1;
					xOnGamma = xOnGamma1;
					yOnGamma = yOnGamma1;
					hk = -1;
					who = 0;

					std::cerr << "[MergingStarsExample1::getSignedDistance] Exceptional case for node " << n << "!"
							  << std::endl;
				}
			}
		}

		who = (ABS( hk ) >= 1)? (short)0 : who;					// Note: If hk = -1, it means the point doesn't belong to a smooth stencil.
		return distance * ((phi1 <= 0 || phi2 <= 0)? -1 : 1);	// Fix the sign too.
	}
};

#endif //FAST_SWEEPING_MERGING_STARS_EXAMPLE_1_H
