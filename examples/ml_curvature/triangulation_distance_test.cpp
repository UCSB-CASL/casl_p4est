/**
 * Testing distance computation from a point to two-dimensional manifolds immersed in 3D and discretized by points and
 * triangles.
 *
 * Developer: Luis Ángel.
 * Created: November 14, 2021.
 */
#include <src/casl_geometry.h>

int main()
{
	// Let's define a cloud of points.
	std::vector<Point3> points = {{2,3,0}, {2,4,0}, {3,3,0}, {3,5,0}, {3,6,0}, {3.5,5,0}, {4.5,5,0}, {4,6,0}, {5,5.5,0},
								  {6,5,0}, {7,6,0}, {7,4,0}, {6,7,0}, {5,7.5,0}, {6,8.5,0}, {6.5,8,0}, {7,9,0},
								  {7.5,8,0}, {9,7,0}, {10,7,0}, {10.5,6,0}, {11,5,0}, {11,7,0}, {11.5,7,0}, {12,7.5,0},
								  {11,8,0}, {13,6,0}, {13,8,0}, {14,7,0}, {14,7.5,0}, {13.5,8.5,0}, {14,9,0}, {16,8,0}};

	// Let's build a balltree to perform knn queries efficiently.
	geom::Balltree balltree( points, 5, true );

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

	return 0;
}