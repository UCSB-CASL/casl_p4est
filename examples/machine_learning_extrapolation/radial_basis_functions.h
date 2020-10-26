//
// Created by Im YoungMin on 10/24/20.
//

#ifndef MACHINE_LEARNING_EXTRAPOLATION_RADIAL_BASIS_FUNCTIONS_H
#define MACHINE_LEARNING_EXTRAPOLATION_RADIAL_BASIS_FUNCTIONS_H

#include <src/casl_math.h>

/**
 * Abstract class for radial basis functions.
 */
class RBF
{
public:
	/**
	 * Evaluation of radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] virtual double operator()( double r ) const
	{
		return 0;
	}
};


/**
 * Gaussian RBF: rbf(r) = exp(-(ar)^2).
 */
class GaussianRBF : public RBF
{
private:
	double _a;		// Shape parameter.

public:
	/**
	 * Constructor.
	 * @param [in] a Positive shape parameter.
	 */
	explicit GaussianRBF( double a = 1 )
	{
		assert( a > 0 );
		_a = a;
	}

	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const override
	{
		return exp( -SQR( _a * r ) );
	}
};


/**
 * Biharmonic RBF: rbf(r) = r.
 */
class BiharmonicRBF : public RBF
{
public:
	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const override
	{
		return r;
	}
};


/**
 * Triharmonic RBF: rbf(r) = r^3.
 */
class TriharmonicRBF : public RBF
{
public:
	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const override
	{
		return pow( r, 3 );
	}
};


/**
 * Thin plate spline RBF: rbf(r) = r^2 * log( r ).
 */
class ThinPlateSplineRBF : public RBF
{
public:
	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const override
	{
		if( r < EPS )
			return 0;
		return SQR( r ) * log( r );
	}
};


/**
 * Multiquadric RBF given by rbf(r) = sqrt( 1 + (ar)^2 ), where r >= 0, and a > 0 is shape parameter.
 */
class MultiquadricRBF : public RBF
{
private:
	double _a;		// Shape parameter.

public:
	/**
	 * Constructor.
	 * @param [in] a Positive shape parameter.
	 */
	explicit MultiquadricRBF( double a = 0.001 )
	{
		assert( a > 0 );
		_a = a;
	}

	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const override
	{
		return sqrt( 1 + SQR( _a * r ) );
	}
};

#endif //MACHINE_LEARNING_EXTRAPOLATION_RADIAL_BASIS_FUNCTIONS_H
