/*
 * Title: machine_learning
 * Description: Create datasets for evaluating a neural network on a non-signed distance function with circular level set.
 * Author: Luis Ángel (임 영민)
 * Date Created: 10-20-2019
 */

// System.
#include <stdexcept>
#include <iostream>

// casl_p4est
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>

#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <cassert>
#include "PointsOnCurves.h"

/*!
 * Circular *non* signed distance function.
 */
struct circle: CF_2
{
	/*!
	 * Circle constructor.
	 * @param x0_ Center x-coordinate.
	 * @param y0_ Center y-coordinate.
	 * @param r_ Circle radius.
	 */
	circle( double x0_, double y0_, double r_ ) : x0( x0_ ), y0( y0_ ), r( r_ )
	{}

	/*!
	 * Non-signed distance function evaluation at a point.
	 * @param x Point x-coordinate.
	 * @param y Point y-coordinate.
	 * @return 0 if point lies on circunference, < 0 if inside circle, > 0 if outside.
	 */
	double operator()( double x, double y ) const override
	{
		return SQR( x - x0 ) + SQR( y - y0 ) - SQR( r );
	}

private:
	double x0, y0, r;		// Center coordinates and circle radius.
};


/*!
 * Global variables.
 */
const std::string DATA_PATH = "/Users/youngmin/Documents/CS/CASL/LSCurvatureML/data/levelSet/";
const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "h"
};
const int NUM_COLUMNS = 11;
const int MIN_D = 0,								// Min and max dimension values.
		  MAX_D = 1;
const double R = 0.35 / 2.0;						// Evaluation interface radius.
const double C[] = {0.5, 0.5};						// Center (x,y) coordinates.

/*!
 * Generate and save evaluation datasets with a variable number or rows and 9+1+1 columns containing the renitialized phi
 * values for a non-distance function with circular interface.
 * @param nGridPoints: Number of grid points per axis (x and y).
 * @param iter: Number of iterations for level set reinitialization.
 * @param mpi: MPI object.
 */
void saveEvaluationReinitializedDataset( int nGridPoints, int iter, const mpi_environment_t& mpi ) noexcept( false );


int main ( int argc, char* argv[] )
{
	try
	{
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		int numberOfIterations = 5;											// Change it to 5, 10, 20, 50, 100.
		for( int resolution = 20; resolution <= 160; resolution += 10 )
			saveEvaluationReinitializedDataset( resolution, numberOfIterations, mpi );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}


void saveEvaluationReinitializedDataset( int nGridPoints, int iter, const mpi_environment_t& mpi ) noexcept(false)
{
	double h = 1. / ( nGridPoints - 1. );						// Spatial step size in both x and y directions.

	////////////////////////////////////////// Setting up the p4est structs ////////////////////////////////////////

	p4est_t *p4est;
	p4est_nodes_t *nodes;
	PetscErrorCode ierr;

	p4est_connectivity_t *connectivity;									// Create the connectivity object.  Our domain is [0,1]^2, and we want a cartesian grid.
	my_p4est_brick_t brick;
	int n_xyz[] = {nGridPoints - 1, nGridPoints - 1, nGridPoints - 1};	// Number of root cells in the macromesh along x, y, z
	double xyz_min[] = {MIN_D, MIN_D, MIN_D};							// Coordinates of the lower-left-back point.
	double xyz_max[] = {MAX_D, MAX_D, MAX_D};							// Coordinates of the front-right-top point.
	int periodic[] = {0, 0, 0};											// Whether the domain is periodic or not.

	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Now create the forest.
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );

	// Finally, re-partition.
	my_p4est_partition( p4est, P4EST_TRUE, nullptr );

	// Create the ghost structure.
	p4est_ghost_t *ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );

	// Generate the node data structure.
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize the neighbor nodes structure.
	my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
	my_p4est_node_neighbors_t node_neighbors( &hierarchy, nodes );

	///////////////////////////////////////// Generating the datasets //////////////////////////////////////////////////

	parStopWatch watch;

	printf( ">> Beginning to generate dataset for %i grid points, with %i iterations, and h = %f, in a [0,1]x[0,1] domain\n", nGridPoints, iter, h );
	watch.start();

	// Prepare file where to write samples.
	std::ofstream outputFile;
	std::string fileName = DATA_PATH + "m" + std::to_string( nGridPoints ) + "_iter" + std::to_string( iter ) +  ".csv";
	outputFile.open( fileName, std::ofstream::trunc );
	if( !outputFile.is_open() )
		throw std::runtime_error( "Output file " + fileName + " couldn't be opened!" );

	// Write output file header.
	std::ostringstream headerStream;
	for( int i = 0; i < NUM_COLUMNS - 1; i++ )
		headerStream << "\"" << COLUMN_NAMES[i] << "\",";
	headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
	outputFile << headerStream.str() << std::endl;

	outputFile.precision( 10 );									// Precision for floating point numbers.

	std::vector<std::vector<double>> CopyGrid( nGridPoints );	// Allocate space for regular grid of re-initialized phi values.
	for( int i = 0; i < nGridPoints; i++ )
		CopyGrid[i].resize( nGridPoints, 0 );

	// Create PETSc vector to hold phi values.
	Vec phi;
	ierr = VecCreateGhostNodes( p4est, nodes, &phi ); CHKERRXX( ierr );

	std::vector<std::vector<double>> samples;

	double hkappa = h / R;										// Target dimensionless curvature: h\kappa = h/r.
	circle circ( C[0], C[1], R );								// Non-signed distance function with circular interface.

	/////// Generate phi values ///////
	double* phi_p;

	ierr = VecGetArray( phi, &phi_p ); CHKERRXX( ierr );
	for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
	{
		double nodeCoords[P4EST_DIM];							// Nodes' coordinates in XY space.
		node_xyz_fr_n( i, p4est, nodes, nodeCoords );
		phi_p[i] = circ( nodeCoords[0], nodeCoords[1] );
	}

	ierr = VecRestoreArray( phi, &phi_p ); CHKERRXX( ierr );

	////// Reinitialize levelset and put values in a cartesian grid of easy access ///////
	my_p4est_level_set_t ls( &node_neighbors );
	ls.reinitialize_2nd_order( phi, iter );

	const double* phi_ptr;
	ierr = VecGetArrayRead( phi, &phi_ptr ); CHKERRXX( ierr );			// Get reinitialized values.

	// Create a regular grid (i.e. matrix) of reinitialized values.
	int totalGridPoints = 0;
	for( size_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
	{
		double nodeCoords[P4EST_DIM];										// Nodes' coordinates in XY space.
		node_xyz_fr_n( n, p4est, nodes, nodeCoords );
		int i = static_cast<int>( round( nodeCoords[0] * ( nGridPoints - 1 ) ) ),		// Indices from 0 to (nGridPoints - 1)
			j = static_cast<int>( round( nodeCoords[1] * ( nGridPoints - 1 ) ) );		// in both x- and y- directions.

		CopyGrid[i][j] = phi_ptr[n];	// Stores information in transposed format (i.e. ith row contains all y values for same x).
		totalGridPoints++;
	}

	ierr = VecRestoreArrayRead( phi, &phi_ptr ); CHKERRXX( ierr );

	////////////// Process grid points' along the circular level set for which we want to compute curvature ////////////

	std::set<std::tuple<int, int>> points = PointsOnCurves::getPointsAlongCircle( C, R, h );

	for( const auto& p : points )
	{
		int i = std::get<0>( p ),							// Returned indices.
			j = std::get<1>( p );
		int subgrid[9][P4EST_DIM] = {						// The nine grid points including the point of interest right in the middle.
				{i - 1, j + 1}, {i, j + 1}, {i + 1, j + 1},
				{i - 1,     j}, {i,     j}, {i + 1,     j},
				{i - 1, j - 1}, {i, j - 1}, {i + 1, j - 1}
		};

		std::vector<double> dataPve( NUM_COLUMNS, 0 );		// Phi, h and h*kappa results in positive form.

		int s;
		for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
		{
			const int* q = &subgrid[s][0];
			assert( q[0] >= 0 && q[0] < nGridPoints && q[1] >= 0 && q[1] < nGridPoints );

			dataPve[s] = CopyGrid[q[0]][q[1]];				// Store phi values.
		}

		dataPve[s] = hkappa;								// Second to last column holds h*\kappa.
		s++;
		dataPve[s] = h;										// Last column holds spatial scale h.
		samples.push_back( dataPve );						// Build nx(9+1+1) matrix of phi, h*kappa, and h values.
	}

	// Write to file the samples content.
	for( const std::vector<double>& row : samples )
	{
		std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
		outputFile << row.back() << std::endl;
	}

	printf( "<< Finished generating %lu samples in %f secs.\n", samples.size(), watch.get_duration_current() );

	// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
	ierr = VecDestroy( phi ); CHKERRXX( ierr );

	////////////////////////////////// Destroy the p4est and its connectivity structure ////////////////////////////////

	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	p4est_connectivity_destroy( connectivity );

	outputFile.close();
	watch.stop();
}