/**
 * Testing distance computation from a point to two-dimensional manifolds immersed in 3D and discretized by points and
 * triangles.
 *
 * Developer: Luis Ángel.
 * Created: November 14, 2021.
 * Updated: November 17, 2021.
 */
#include <src/casl_geometry.h>

struct MongeFunction1 : public geom::MongeFunction
{
	// Pyramidal function with appex at the origin.
	double operator()( double x, double y ) const override
	{
		return 1.0 - ABS( x ) - ABS( y );
	}

	// Dummy curvature function to pass the polymorphism test.
	double meanCurvature( const double& x, const double& y ) const override
	{
		return 0;
	}
};

int main()
{
	// Let's define a cloud of points.
	std::vector<Point3> points = {{2,3,0}, {2,4,0}, {3,3,0}, {3,5,0}, {3,6,0}, {3.5,5,0}, {4.5,5,0}, {4,6,0}, {5,5.5,0},
								  {6,5,0}, {7,6,0}, {7,4,0}, {6,7,0}, {5,7.5,0}, {6,8.5,0}, {6.5,8,0}, {7,9,0},
								  {7.5,8,0}, {9,7,0}, {10,7,0}, {10.5,6,0}, {11,5,0}, {11,7,0}, {11.5,7,0}, {12,7.5,0},
								  {11,8,0}, {13,6,0}, {13,8,0}, {14,7,0}, {14,7.5,0}, {13.5,8.5,0}, {14,9,0}, {16,8,0}};

	// Let's build a balltree to perform knn queries efficiently.
	geom::Balltree balltree( points, false, 5, true );

	// Let's perform knn search, for k=1.
	Point3 q0(4.5, 3, 0);
	double d0;
	const Point3 *nn0 = balltree.findNearestNeighbor( q0, d0 );
	std::cout << "Closest point: " << *nn0 << "Dist: " << d0 << std::endl << std::endl;

	Point3 q1( 8, 7, 0 );
	double d1;
	const Point3 *nn1 = balltree.findNearestNeighbor( q1, d1 );
	std::cout << "Closest point: " << *nn1 << "Dist: " << d1 << std::endl << std::endl;

	Point3 q2( 15, 2, 0 );
	double d2;
	const Point3 *nn2 = balltree.findNearestNeighbor( q2, d2 );
	std::cout << "Closest point: " << *nn2 << "Dist: " << d2 << std::endl << std::endl;

	// Testing distances from points in 3D to a triangle.
	Point3 vertices[] = {{0,0,0}, {2,0,0}, {0,3,0}};
	geom::Triangle triangle( &vertices[0], &vertices[1], &vertices[2] );

	Point3 queryPoints[] = {{1,0,1}, {1,1,1}, {0.5,2,-1}, {-1,2,2}, {3,0,-1}, {1.5,1.5,-2}, {1.5,-1,-1}, {-0.5,4,1}};
	Point3 closestPoints[] = {{1,0,0},{1,1,0}, {0.5,2,0}, {0,2,0}, {2,0,0}, {1.153846153846154,1.269230769230769,0}, {1.5,0,0}, {0,3,0}};
	double distances[] = {1, 1, 1, sqrt(5), M_SQRT2, 2.042811034598385, M_SQRT2, 1.5};

	for( int i = 0; i < 7; i++ )
	{
		std::cout << "Point " << i << "... ";
		Point3 P = triangle.findClosestPointToQuery( &queryPoints[i] );
		double d = (P - queryPoints[i]).norm_L2();
		assert( ABS( d - distances[i] ) < EPS );
		std::cout << "d ok, ";
		assert( (P - closestPoints[i]).norm_L2() < EPS );
		std::cout << "P ok." << std::endl;
	}
	std::cout << "All triangle tests succeeded!" << std::endl << std::endl;

	// Let's create Monge patch and test it.
	MongeFunction1 dummyFunction;
	geom::DiscretizedMongePatch discretizedMongePatch( 1, 2, &dummyFunction, 5 );

	std::ofstream trianglesFile;				// Dumping triangles' vertices into a file for debugging/visualizing.
	std::string rlsFileName = "triangles.csv";
	trianglesFile.open( rlsFileName, std::ofstream::trunc );
	if( !trianglesFile.is_open() )
		throw std::runtime_error( "Triangles file couldn't be opened for dumping mesh!" );
	trianglesFile << R"("x0","y0","z0","x1","y1","z1","x2","y2","z2")" << std::endl;
	discretizedMongePatch.dumpTriangles( trianglesFile );
	trianglesFile.precision( 15 );
	trianglesFile.close();

	q0 = Point3( 0.4, -0.2, 0.6 );
	Point3 r0 = discretizedMongePatch.findNearestPoint( q0, d0 );
	assert( (Point3( 1./3, -4./30, 16./30 ) - r0).norm_L2() < EPS );
	assert( ABS( d0 - 0.11547005383792514 ) < EPS );
	std::cout << "Fourth-quadrant test... ok." << std::endl;

	q1 = Point3( -0.2, -0.4, 0.8 );
	Point3 r1 = discretizedMongePatch.findNearestPoint( q1, d1 );
	assert( (Point3( -2./30, -8./30, 2./3 ) - r1).norm_L2() < EPS );
	assert( ABS( d1 - 0.23094010767585033 ) < EPS );
	std::cout << "Third-quadrant test... ok." << std::endl;

	q2 = Point3( 0.5, 0.5, 1 );
	Point3 r2 = discretizedMongePatch.findNearestPoint( q2, d2 );
	assert( (Point3( 5./30, 5./30, 2./3 ) - r2).norm_L2() < EPS );
	assert( ABS( d2 - 0.577350269189626 ) < EPS );
	std::cout << "First-quadrant test... ok." << std::endl;

	Point3 q3 = Point3( -0.125, 0.125, 1 );
	double d3;
	Point3 r3 = discretizedMongePatch.findNearestPoint( q3, d3 );
	assert( (Point3( -125./3000, 125./3000, 275./300 ) - r3).norm_L2() < EPS );
	assert( ABS( d3 - 0.14433756729740646 ) < EPS );
	std::cout << "Second-quadrant test... ok." << std::endl;

	std::cout << "All shortest distance tests succeeded!" << std::endl;

	return 0;
}