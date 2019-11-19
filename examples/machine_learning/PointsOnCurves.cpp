//
// Created by Im YoungMin on 10/7/19.
//

#include "PointsOnCurves.h"

const double PointsOnCurves::MIN_H = 0.000001;
const double PointsOnCurves::MAX_H = M_PI;

std::set<std::tuple<int, int>> PointsOnCurves::getPointsAlongCircle( double const* c, double r, double h )
{
	// Verify the provided h value is within a valid range.
	if( h < MIN_H || h > MAX_H )
	{
		h = std::min( std::max( MIN_H, h ), MAX_H );
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

////////////////////////////////////////////// Flower functions ////////////////////////////////////////////////////////

Flower::Flower( double a, double b, int p ) : _a( a ), _b( b ), _p( p )
{}

double Flower::operator()( double x, double y ) const
{
	return -_a * cos( _p * atan2( y, x ) ) + sqrt( x * x + y * y ) - _b;
}

double Flower::r( double theta ) const
{
	return _a * cos( _p * theta ) + _b;
}

std::set<std::tuple<int, int>> Flower::getPointIndicesAlongInterface( unsigned nGridPoints, double &h, double &minVal ) const
{
	if( nGridPoints < 16 )							// Validation on number of grid points.
	{
		nGridPoints = 16;
		std::cerr << "Number of grid points is too small, it'll be set to 16" << std::endl;
	}

	double limitVal = std::max( abs( _a + _b ), abs( _a - _b ) );
	h = 2 * limitVal / ( nGridPoints - 5 );			// We have 2h as padding on each direction from the limit val of the flower.
	minVal = -limitVal - 2 * h;						// This is the lowest value we consider in the domain to create the grid.

	// Verify the computed h value is within a valid range.
	if( h < PointsOnCurves::MIN_H || h > PointsOnCurves::MAX_H )
	{
		h = std::min( std::max( PointsOnCurves::MIN_H, h ), PointsOnCurves::MAX_H );
		throw std::runtime_error( "The value of h = " + std::to_string( h ) + " is out of range!" );
		return std::set<std::tuple<int, int>>();	// Stop! There's an error.
	}

	// Start collecting cells.
	std::set<std::tuple<int, int>> cells;
	double alpha = h / limitVal / 2;				// Angle step along interface.
	double beta = 0;
	int idx = 0;
	while( beta < 2 * M_PI )
	{
		double rt = r( beta );
		double cpx = rt * cos( beta ), cpy = rt * sin( beta );		// Point cartesian coordinates.
		int dpi = floor( ( cpx - minVal ) / h );					// Cell where cartesian point lands on.
		int dpj = floor( ( cpy - minVal ) / h );
		cells.insert( std::make_tuple( dpi, dpj ) );				// Keep unique cells.
		idx++;
		beta = idx * alpha;
	}

	// Now, collect grid points whose any of their four edges are crossed by the interface.
	std::set<std::tuple<int, int>> gridPoints;				// Resulting grid coordinates are now based on the lower left corner of a cell.
	std::map<std::tuple<int, int>, bool> visitedCorners;	// Avoid revisiting cell corners.
	for( const auto& cell : cells )
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
		for( const auto& corner : corners)
		{
			if( visitedCorners.find( corner ) != visitedCorners.end() )		// Check we haven't visited this corner (from a different cell)
				continue;													// because corner (values) are shared among cells.
			visitedCorners[corner] = true;

			double x = std::get<0>( corner ) * h + minVal,
				   y = std::get<1>( corner ) * h + minVal;
			double cornerVal = this->operator()( x, y );					// Cartesian coordinates of circumference.
			if( cornerVal == 0 )
			{
				gridPoints.insert( corner );
				continue;
			}

			int directions[4][2] = {{+1, 0}, {-1, 0}, {0, +1}, {0, -1}};	// Check out four directions with respect to current corner.
			for( auto& dir : directions )
			{
				int const* direction = &dir[0];
				int neighborI = std::get<0>( corner ) + direction[0],		// Indices of neighboring grid point.
					neighborJ = std::get<1>( corner ) + direction[1];
				x = neighborI * h + minVal;
				y = neighborJ * h + minVal;
				double neighborVal = this->operator()( x, y );				// Where the neighbor is located with respect to circumference.
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

double Flower::curvature( double theta ) const
{
	double r = this->r( theta );								// r(theta).
	double rPrime = -_a * _p * sin( _p * theta );				// r'(theta).
	double rPrimePrime = -_a * _p * _p * cos( _p * theta );		// r''(theta).
	return ( r * r + 2 * rPrime * rPrime - r * rPrimePrime ) / pow( r * r + rPrime * rPrime, 3. / 2. );
}

double Flower::getA() const
{
	return _a;
}

double Flower::getB() const
{
	return _b;
}

int Flower::getP() const
{
	return _p;
}

void Flower::getHAndMinVal( unsigned nGridPoints, double &h, double &minVal, unsigned padding ) const
{
	if( nGridPoints < 16 )							// Validation on number of grid points.
	{
		nGridPoints = 16;
		std::cerr << "Number of grid points is too small, it'll be set to 16" << std::endl;
	}

	if( padding < 2 )
	{
		padding = 2;
		std::cerr << "Number of grid poits used for padding should be at least 2. It'll be set to 2" << std::endl;
	}

	double limitVal = std::max( abs( _a + _b ), abs( _a - _b ) );
	h = 2 * limitVal / ( nGridPoints - 1 - 2 * padding );	// We have at least 2h as padding on each direction from the limit val of the flower.
	minVal = -limitVal - padding * h;						// This is the lowest value we consider in the domain to create the grid.
}


