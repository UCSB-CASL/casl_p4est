//
// Created by Im YoungMin on 11/5/19.
//

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

// Root finding.
#include "DistThetaRootFinding.h"

/*!
 * Global variables.
 */
const std::string DATA_PATH = "/Users/youngmin/Documents/CS/CASL/LSCurvatureML/data/flower/";
const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "h"
};
const int NUM_COLUMNS = 11;

/*!
 * Generate and save a flower-shaped interface dataset with 9+1+1 columns containing the renitialized phi values.
 * @param flower: The flower interface.
 * @param nGridPoints: Number of grid points along each direction.
 * @param iter: Number of iterations for level set reinitialization.
 * @param mpi: MPI object.
 */
void generateReinitializedFlowerDataset( const Flower& flower, int nGridPoints, int iter, const mpi_environment_t& mpi )
{
	// We need to collect grid points along the flower interface and determine the value for h and minVal.
	double h, minVal;
	std::set<std::tuple<int, int>> points = flower.getPointIndicesAlongInterface( nGridPoints, h, minVal );

	//////////////////////////////////////////// Setting up the p4est structs //////////////////////////////////////////

	p4est_t *p4est;
	p4est_nodes_t *nodes;
	PetscErrorCode ierr;

	p4est_connectivity_t *connectivity;									// Create the connectivity object.
	my_p4est_brick_t brick;
	int n_xyz[] = {nGridPoints - 1, nGridPoints - 1, nGridPoints - 1};	// Number of root cells in the macromesh along x, y, [z].
	double xyz_min[] = {minVal, minVal, minVal};						// Coordinates of the lower-left-back point.
	double xyz_max[] = {-minVal, -minVal, -minVal};						// Coordinates of the front-right-top point.
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

	/////////////////////////////////////////// Generating the dataset /////////////////////////////////////////////////

	parStopWatch watch;

	printf( ">> Beginning to generate flower dataset for %i grid points, and h = %f, in a [%f,%f]^2 domain\n", nGridPoints, h, minVal, -minVal );
	watch.start();

	// Prepare file where to write sample phi values.
	std::ofstream outputFile;
	std::string fileName = DATA_PATH + "phi_iter" + std::to_string( iter ) + ".csv";
	outputFile.open( fileName, std::ofstream::trunc );
	if( !outputFile.is_open() )
		throw std::runtime_error( "Phi values output file " + fileName + " couldn't be opened!" );

	std::ostringstream headerStream;									// Write output file header.
	for( int i = 0; i < NUM_COLUMNS - 1; i++ )
		headerStream << "\"" << COLUMN_NAMES[i] << "\",";
	headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
	outputFile << headerStream.str() << std::endl;

	outputFile.precision( 10 );											// Precision for floating point numbers.

	// Prepare file where to write sample grid point cartesian coordinates.
	std::ofstream pointsFile;
	std::string pointsFileName = DATA_PATH + "points_iter" + std::to_string( iter ) + ".csv";
	pointsFile.open( pointsFileName, std::ofstream::trunc );
	if( !pointsFile.is_open() )
		throw std::runtime_error( "Point cartesian coordinates file " + pointsFileName + " couldn't be opened!" );

	std::ostringstream headerPointsStream;								// Write output file header.
	headerPointsStream << R"("x","y")";
	pointsFile << headerPointsStream.str() << std::endl;

	pointsFile.precision( 10 );

	// Prepare file where to write the params.
	std::ofstream paramsFile;
	std::string paramsFileName = DATA_PATH + "params_iter" + std::to_string( iter ) + ".csv";
	paramsFile.open( paramsFileName, std::ofstream::trunc );
	if( !paramsFile.is_open() )
		throw std::runtime_error( "Params file " + paramsFileName + " couldn't be opened!" );

	std::ostringstream headerParamsStream;
	headerParamsStream << R"("nGridPoints","h","minVal","a","b","p")";
	paramsFile << headerParamsStream.str() << std::endl;

	paramsFile.precision( 10 );
	paramsFile << nGridPoints << "," << h << "," << minVal << "," << flower.getA() << "," << flower.getB() << "," << flower.getP() << std::endl;

	// Prepare file where to write the angle parameter for corresponding points on interface.
	std::ofstream anglesFile;
	std::string anglesFileName = DATA_PATH + "angles_iter" + std::to_string( iter ) + ".csv";
	anglesFile.open( anglesFileName, std::ofstream::trunc );
	if( !anglesFile.is_open() )
		throw std::runtime_error( "Angles file " + anglesFileName + " couldn't be opened!" );

	anglesFile << R"("theta")" << std::endl;							// Write header.

	// Setting up the regular grid.
	std::vector<std::vector<double>> CopyGrid( nGridPoints );			// Allocate space for regular grid of re-initialized phi values.
	for( int i = 0; i < nGridPoints; i++ )
		CopyGrid[i].resize( nGridPoints, 0 );

	// Create PETSc vector to hold phi values.
	Vec phi;
	ierr = VecCreateGhostNodes( p4est, nodes, &phi ); CHKERRXX( ierr );

	/////// Generate phi values ///////
	double* phi_p;

	ierr = VecGetArray( phi, &phi_p ); CHKERRXX( ierr );
	for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
	{
		double nodeCoords[P4EST_DIM];									// Nodes' coordinates in cartesian coordinates.
		node_xyz_fr_n( i, p4est, nodes, nodeCoords );
		phi_p[i] = flower( nodeCoords[0], nodeCoords[1] );
	}

	ierr = VecRestoreArray( phi, &phi_p ); CHKERRXX( ierr );

	////// Reinitialize levelset and put values in a cartesian grid for easy access ///////
	my_p4est_level_set_t ls( &node_neighbors );
	ls.reinitialize_2nd_order( phi, iter );

	const double* phi_ptr;
	ierr = VecGetArrayRead( phi, &phi_ptr ); CHKERRXX( ierr );			// Get reinitialized values.

	// Create a regular grid (i.e. matrix) of reinitialized values.
	int totalGridPoints = 0;
	for( size_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
	{
		double nodeCoords[P4EST_DIM];									// Nodes' coordinates in XY space.
		node_xyz_fr_n( n, p4est, nodes, nodeCoords );
		int i = static_cast<int>( round( ( nodeCoords[0] - minVal ) / h ) ),	// Indices from 0 to (nGridPoints - 1)
			j = static_cast<int>( round( ( nodeCoords[1] - minVal ) / h ) );	// in both x- and y- directions.

		CopyGrid[i][j] = phi_ptr[n];	// Stores information in transposed format (i.e. ith row contains all y values for same x).
		totalGridPoints++;
	}

//	std::cout << "==================== Mesh grid ====================" << std::endl;
//	for( int i = 0; i < nGridPoints; i++ )
//	{
//		for( int j = 0; j < nGridPoints; j++ )
//			std::cout << CopyGrid[i][j] << ", ";
//		std::cout << ";" << std::endl;
//	}

	ierr = VecRestoreArrayRead( phi, &phi_ptr ); CHKERRXX( ierr );

	std::vector<std::vector<double>> samples;

	/////// Process grid points' along the flower levelset ///////
	for( const auto& p : points )
	{
		int i = std::get<0>( p ),							// Returned indices.
			j = std::get<1>( p );
		int subgrid[9][P4EST_DIM] = {						// The nine grid points including the point of interest right in the middle.
				{i - 1, j + 1}, {i, j + 1}, {i + 1, j + 1},
				{i - 1,     j}, {i,     j}, {i + 1,     j},
				{i - 1, j - 1}, {i, j - 1}, {i + 1, j - 1}
		};

		std::vector<double> sample( NUM_COLUMNS, 0 );		// Phi, h and h*kappa results.

		int s;
		double leftPhi, rightPhi, topPhi, bottomPhi;		// To compute grad(\phi_{i,j}).
		double centerPhi;									// \phi{i,j}.
		for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
		{
			const int* q = &subgrid[s][0];
			assert( q[0] >= 0 && q[0] < nGridPoints && q[1] >= 0 && q[1] < nGridPoints );
			sample[s] = CopyGrid[q[0]][q[1]];

			switch( s )
			{
				case 1: topPhi = sample[s]; break;			// \phi_{i,j+1}.
				case 3: leftPhi = sample[s]; break;			// \phi_{i-1,j}.
				case 4: centerPhi = sample[s]; break;		// Point's phi value.
				case 5: rightPhi = sample[s]; break;		// \phi_{i+1,j}.
				case 7: bottomPhi = sample[s]; break;		// \phi_{i,j-1}.
				default: ;
			}
		}

		// Computing the target curvature.
		double px = std::get<0>( p ) * h + minVal,				// Reference grid point in cartesian coordinates.
			   py = std::get<1>( p ) * h + minVal;
		double grad[] = {
				( rightPhi - leftPhi ) / ( 2 * h ),
				( topPhi - bottomPhi ) / ( 2 * h )
		};
		double gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );
		double pOnInterfaceX = px - grad[0] / gradNorm * centerPhi,			// Initial cartesian coordinates of projection of grid point on interface.
			   pOnInterfaceY = py - grad[1] / gradNorm * centerPhi;

		double thetaOnInterface = atan2( pOnInterfaceY, pOnInterfaceX );	// Initial guess for root finding method.
		thetaOnInterface = ( thetaOnInterface < 0 )? thetaOnInterface + 2 * M_PI : thetaOnInterface;
		double valOfDerivative;
		double newThetaOnInterface = distThetaDerivative( px, py, flower, valOfDerivative, thetaOnInterface, thetaOnInterface - 0.0001, thetaOnInterface + 0.0001 );
		double r = flower.r( thetaOnInterface );							// Recalculating closest point on interface.
		pOnInterfaceX = r * cos( newThetaOnInterface );
		pOnInterfaceY = r * sin( newThetaOnInterface );

//		std::cout << valOfDerivative << "; plot([" << px << "], [" << py << "], 'b.', [" << pOnInterfaceX << "], [" << pOnInterfaceY << "], 'm.');" << std::endl;

		double dx = px - pOnInterfaceX,							// Verify that point on interface is not far from corresponding grid point.
			   dy = py - pOnInterfaceY;
		if( dx * dx + dy * dy <= h * h )
			thetaOnInterface = newThetaOnInterface;				// Theta is OK.
		else
			std::cerr << "Minimization placed point on interface too far.  Reverting back to point on interface calculated with phi values" << std::endl;

		sample[s] = h * flower.curvature( thetaOnInterface );	// Second to last column holds h*\kappa.
		s++;
		sample[s] = h;											// Last column holds spatial scale h.
		samples.push_back( sample );							// Build nx(9+1+1) matrix of phi, h*kappa, and h values.

		// Write sample point cartesian coordinates.
		pointsFile << px << "," << py << std::endl;

		// Write angle parameter for projected point on interface.
		anglesFile << ( thetaOnInterface < 0 ? 2 * M_PI + thetaOnInterface : thetaOnInterface ) << std::endl;
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
	pointsFile.close();
	paramsFile.close();
	watch.stop();
}

/**
 * Main function.
 */
int main ( int argc, char* argv[] )
{
	try
	{
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		int numberOfIterations = 5;
		Flower flower( 0.05, 0.15, 3 );

		generateReinitializedFlowerDataset( flower, 107, numberOfIterations, mpi );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}