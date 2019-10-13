//
// Created by Im YoungMin on 10/7/19.
//

#ifndef MACHINE_LEARNING_LS_DATASETS_POINTSONCURVES_H
#define MACHINE_LEARNING_LS_DATASETS_POINTSONCURVES_H

#include <set>
#include <map>
#include <tuple>
#include <vector>
#include <iostream>
#include <src/casl_math.h>

/*!
 * Finding the indices of nodes (i.e. grid points) along curves on regular, cartesian grids.
 */
class PointsOnCurves
{
private:
	/*!
	 * Minimum and maximum allowed values for h.
	 */
	static const double _MIN_H;
	static const double _MAX_H;

public:

	/*!
	 * Retrieve the grid point indices along a circle.
	 * @param c: Circle center coordinates.
	 * @param r: Circle radius.
	 * @param h: Spatial step size in both x and y directions.
	 * @return A set of tuples with the discrete i and j node coordinates.
	 */
	static std::set<std::tuple<int, int>> getPointsAlongCircle( double const* c, double r, double h );
};


#endif //MACHINE_LEARNING_LS_DATASETS_POINTSONCURVES_H
