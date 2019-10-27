//
// Created by Im YoungMin on 10/7/19.
//

#include "PointsOnCurves.h"

const double PointsOnCurves::_MIN_H = 0.000001;
const double PointsOnCurves::_MAX_H = PI;

std::set<std::tuple<int, int>> PointsOnCurves::getPointsAlongCircle( double const* c, double r, double h )
{
	// Verify the provided h value is within a valid range.
	if( h < _MIN_H || h > _MAX_H )
	{
		h = std::min( std::max( _MIN_H, h ), _MAX_H );
		std::cerr << "The value of h is outside a valid range.  It'll be truncated to " << h << std::endl;
	}

	// Verify the provided radius is valid.
	if( r < h )
	{
		r = std::max( h, r );
		std::cerr << "The circle radius is below a valid limit.  It'll be reassigned to " << r << std::endl;
	}

	// Start by collecting cells, which we test whether they are crossed by the input circumference.
	std::set<std::tuple<int, int>> cells;
	std::vector<double> radii;
	for( int i = -1; i <= 1; i++ )		// Expand the radius inwards and outwards to capture more potential cells.
		radii.push_back( std::max( h, r + i * h / 2.0 ) );

	for( double radius : radii )
	{
		double theta = h / radius;
		double alpha = theta / 2.0;		// Angle step for moving along the circumference.
		double phi = 0;
		int i = 0;
		while( phi < 2.0 * PI )
		{
			double px = c[0] + radius * cos( phi ),		// Point in floating space.
				   py = c[1] + radius * sin( phi );
			int di = floor( px / h ),					// Cell where p lands on.
				dj = floor( py / h );
			cells.insert( std::make_tuple( di, dj ) );	// Keep unique cells.
			i++;
			phi = i * alpha;
		}
	}

	// Then, collect grid points by the circumference.
	std::set<std::tuple<int, int>> gridPoints;				// Resulting grid coordinates are now based on the lower left corner of a cell.
	std::map<std::tuple<int, int>, bool> visitedCorners;	// Avoid revisiting cell corners.
	for( auto& cell : cells )
	{
		// Consider cells whose grid points lie at opposite directions from the interface.
		// 3 -- 2
		// |    |   <-- Cell and its four corners.
		// 0 -- 1
		std::tuple<int, int> corners[] = {
				std::make_tuple( std::get<0>( cell ), std::get<1>( cell ) ),
				std::make_tuple( std::get<0>( cell ) + 1, std::get<1>( cell ) ),
				std::make_tuple( std::get<0>( cell ) + 1, std::get<1>( cell ) + 1 ),
				std::make_tuple( std::get<0>( cell ), std::get<1>( cell ) + 1 )};
		for( auto& corner : corners)
		{
			if( visitedCorners.find( corner ) != visitedCorners.end() )		// Check we haven't visited this corner (from a different cell)
				continue;													// because corner (values) are shared among cells.
			visitedCorners[corner] = true;

			double dx = std::get<0>( corner ) * h - c[0],
				   dy = std::get<1>( corner ) * h - c[1];
			double distSquare = SQR( dx ) + SQR( dy );
			double cornerVal = distSquare - SQR( r );						// Where the corner is located with respect to circumference.
			if( cornerVal == 0 )
			{
				gridPoints.insert( corner );
				continue;
			}

			int directions[4][2] = {{+1, 0}, {-1, 0}, {0, +1}, {0, -1}};	// Check out four directions with respect to current corner.
			for( auto& dir : directions )
			{
				int const* direction = &dir[0];
				int neighborX = std::get<0>( corner ) + direction[0],		// x and y value of neighboring grid point.
					neighborY = std::get<1>( corner ) + direction[1];
				dx = neighborX * h - c[0];
				dy = neighborY * h - c[1];
				distSquare = SQR( dx ) + SQR( dy );
				double neighborVal = distSquare - SQR( r );					// Where the neighbor is located with respect to circumference.
				if( neighborVal * cornerVal < 0 )
				{
					gridPoints.insert( corner );							// Interface intersects edge going from corner to neighbor grid point.
					break;
				}
			}
		}
	}

	return gridPoints;
}