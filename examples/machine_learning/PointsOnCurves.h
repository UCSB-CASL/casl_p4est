//
// Created by Im YoungMin on 10/7/19.
//

#ifndef MACHINE_LEARNING_POINTSONCURVES_H
#define MACHINE_LEARNING_POINTSONCURVES_H

#include <set>
#include <map>
#include <tuple>
#include <vector>
#include <iostream>
#include <src/casl_math.h>
#include <src/my_p4est_utils.h>

/**
 * Flower-shaped interface centered at the origin.
 */
class Flower: CF_2
{
private:
	double _a;			// Perturbation (petal extension).
	double _b;			// Base circle radius.
	int _p;				// Number of petals.

public:
	/**
	 * Constructor
	 * @param a: Perturbation amplitude.
	 * @param b: Base circle radius.
	 * @param p: Petal number.
	 */
	explicit Flower( double a = -1.0, double b = 3.0, int p = 8 );

	/**
	 * Level set evaluation at a given point.
	 * @param x: Point x-coordinate.
	 * @param y: Point y-coordinate.
	 * @return phi(x,y).
	 */
	double operator()( double x, double y ) const override;

	/**
	 * Compute point on interface given an angular parameter.
	 * @param theta: Angular parameter.
	 * @return r(theta).
	 */
	[[nodiscard]] double r( double theta ) const;

	/**
	 * Retrieve the grid point indices along the flower-shaped interface.  Also, compute h and the minVal for the active
	 * domain (i.e. minimum domain where it's safe to use grid points for the interface on the plane).
	 * @param nGridPoints: Number of grid points.
	 * @param h [out]: Computed spatial step size.
	 * @param minVal [out]: Minimum value in the active domain.
	 * @return A set of pairs of index coordinates for grid points, where (0,0) corresponds to (minVal, minVal) in cartesian coords.
	 */
	[[nodiscard]] std::set<std::tuple<int, int>> getPointIndicesAlongInterface( int nGridPoints, double& h, double& minVal ) const;

	/**
	 * Compute curvature.
	 * @param theta: Angle parameter.
	 * @return kappa.
	 */
	[[nodiscard]] double curvature( double theta ) const;

	/**
	 * Get the 'a' perturbation flower parameter.
	 * @return flower.a.
	 */
	[[nodiscard]] double getA() const;

	/**
	 * Get the 'b' base radius flower parameter.
	 * @return flower.b.
	 */
	[[nodiscard]] double getB() const;

	/**
	 * Get the number of petals 'p' in the flower.
	 * @return flower.p.
	 */
	[[nodiscard]] int getP() const;
};

/**
 * Finding the indices of nodes (i.e. grid points) along curves on regular, cartesian grids.
 */
class PointsOnCurves
{
public:
	/**
	 * Minimum and maximum allowed values for h.
	 */
	static const double MIN_H;
	static const double MAX_H;

	/**
	 * Retrieve the grid point indices along a circle.
	 * @param c: Circle center coordinates.
	 * @param r: Circle radius.
	 * @param h: Spatial step size in both x and y directions.
	 * @return A set of tuples with the discrete i and j node coordinates.
	 */
	[[nodiscard]] static std::set<std::tuple<int, int>> getPointsAlongCircle( double const* c, double r, double h );
};

#endif
