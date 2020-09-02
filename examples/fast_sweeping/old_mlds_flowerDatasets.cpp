//
// Created by Im YoungMin on 11/5/19.
// Note: This was adapted on 9/2/20 to allow for interpolated h*kappa at the interface because the results in the paper
// were collected, for the numerical method, at the stencil center rather than at gamma.  Thus, the comparison was
// unfair, in favor of the neural network (which produces h*kappa at the interface).
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
#include <src/casl_geometry.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>

// Root finding.
#include "star_theta_root_finding.h"
#include "local_utils.h"

/**
 * Generate the sample row of level-set function values and target h*kappa for a node that has been found next to the
 * star interface.  We assume that this query node is effectively adjacent to Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] neighbors Pointer to neighbors data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] star The level-set function with a star-shaped interface.
 * @param [in] gen Random-number generator device.
 * @param [in] uniformDistribution A uniform random-variable distribution.
 * @param [in/out] pointsFile Reference to file object where to write coordinates of nodes adjacent to Gamma.
 * @param [in/out] anglesFile Reference to file object where to write angles of normal projected points on Gamma.
 * @param [out] distances A vector of "true" distances from all of 9 stencil points to the star-shaped level-set.
 * @param [out] xOnGamma x-coordinate of normal projection of grid node onto interface.
 * @param [out] yOnGamma y-coordinate of normal projection of grid node onto interface.
 * @return Vector of sampled, reinitialized level-set function values for the stencil centered at the nodeIdx node.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
																 const double H,
																 const std::vector<p4est_locidx_t>& stencil,
																 const p4est_t *p4est, const p4est_nodes_t *nodes,
																 const my_p4est_node_neighbors_t *neighbors,
																 const double *phiReadPtr,
																 const geom::Star& star, std::mt19937& gen,
																 std::normal_distribution<double>& normalDistribution,
																 std::ofstream& pointsFile, std::ofstream& anglesFile,
																 std::vector<double>& distances,
																 double& xOnGamma, double& yOnGamma )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// Level-set function values and target h*kappa.
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// True distances and target h/kappa as well.

	int s;												// Index to fill in the sample vector.
	double grad[P4EST_DIM];
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double theta, r, valOfDerivative, centerTheta;
	double dx, dy, newDistance;
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample[s] = phiReadPtr[stencil[s]];				// This is the distance obtained after reinitialization.

		// To find the true distance we need the neighborhood of each stencil node.
		neighbors->get_neighbors( stencil[s], qnnnPtr );
		qnnnPtr->gradient( phiReadPtr, grad );
		gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );	// Get the unit gradient.

		// Approximate position of point projected on interface.
		node_xyz_fr_n( stencil[s], p4est, nodes, xyz );
		pOnInterfaceX = xyz[0] - grad[0] / gradNorm * sample[s];
		pOnInterfaceY = xyz[1] - grad[1] / gradNorm * sample[s];

		if( s == 4 )	// Rough estimation of point on interface, where curvature will be interpolated.
		{
			xOnGamma = pOnInterfaceX;
			yOnGamma = pOnInterfaceY;
		}

		// Get initial angle for polar approximation to point on star interface.
		theta = atan2( pOnInterfaceY, pOnInterfaceX );
		theta = ( theta < 0 )? theta + 2 * M_PI : theta;
		r = star.r( theta );
		pOnInterfaceX = r * cos( theta );
		pOnInterfaceY = r * sin( theta );				// Better approximation of projection of stencil point onto star.

		// Compute current distance to Gamma using the improved point on interface.
		dx = xyz[0] - pOnInterfaceX;
		dy = xyz[1] - pOnInterfaceY;
		distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

		// Find theta that yields "a" minimum distance between stencil point and star using Newton-Raphson's method.
		valOfDerivative = 1;
		theta = distThetaDerivative( stencil[s], xyz[0], xyz[1], star, theta, H, gen, normalDistribution,
									 valOfDerivative, newDistance );

		if( newDistance - distances[s] > EPS )			// Verify that new point is closest than previous approximmation.
		{
			std::ostringstream stream;
			stream << "Failure with node " << stencil[s] << ".  Val. of Der: " << std::scientific << valOfDerivative
				   << std::fixed << std::setprecision( 15 ) << ".  New dist: " << newDistance
				   << ".  Old dist: " << distances[s];
			throw std::runtime_error( stream.str() );
		}

		distances[s] = newDistance;						// Root finding was successful: keep minimum distance.

		if( star( xyz[0], xyz[1] ) < 0 )				// Fix sign.
			distances[s] *= -1;

		if( s == 4 )									// For center node we need theta to yield curvature.
			centerTheta = theta;
	}

	sample[s] = H * star.curvature( centerTheta );		// Last column holds h*kappa.
	distances.push_back( sample[s] );

	// Write center sample node index and coordinates.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	pointsFile << nodeIdx << "," << xyz[0] << "," << xyz[1] << std::endl;

	// Write angle parameter for projected point on interface.
	anglesFile << ( centerTheta < 0 ? 2 * M_PI + centerTheta : centerTheta ) << std::endl;

	return sample;
}

/*!
 * Generate and save a flower-shaped interface dataset with 9+1+1 columns containing the renitialized phi values, the
 * expected dimensionless curvature and the numerically interpolated h*kappa at the interface.
 * @param [in] star The flower interface.
 * @param [in] resolution Target unitary resolution (e.g. 256 grid points per unit length).
 * @param [in] nGridPoints Number of grid points along each direction.
 * @param [in] iter Number of iterations for level set reinitialization (e.g. 5, 10, 20).
 * @param [in] mpi MPI object.
 */
void generateReinitializedFlowerDataset( const geom::Star& star, int resolution, int nGridPoints, int iter,
										 const mpi_environment_t& mpi )
{
	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  							// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 							// Standard mersenne_twister_engine seeded with rd().
	std::normal_distribution<double> normalDistribution;				// Used for bracketing and root finding.

	// We need to collect grid points along the flower interface and determine the value for h and minVal.
	double H, minVal;
	star.getHAndMinVal( nGridPoints, H, minVal );

	//////////////////////////////////////////// Setting up the p4est structs //////////////////////////////////////////

	PetscErrorCode ierr;

	// Domain information.
	int n_xyz[] = {nGridPoints - 1, nGridPoints - 1, nGridPoints - 1};	// Number of root cells in the macromesh.
	double xyz_min[] = {minVal, minVal, minVal};						// Coordinates of the lower-left-back point.
	double xyz_max[] = {-minVal, -minVal, -minVal};						// Coordinates of the front-right-top point.
	int periodic[] = {0, 0, 0};											// Whether the domain is periodic or not.

	// p4est variables and data structures.
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	my_p4est_brick_t brick;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Definining the non-signed distance level-set function to be reinitialized.
	splitting_criteria_cf_and_uniform_band_t levelSetSC( 0, 0, (CF_2 *) &star, 2 );

	// Create the forest using a level set as refinement criterion.
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
	p4est->user_pointer = (void *) &levelSetSC;

	// Refine and recursively partition forest.
	my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
	my_p4est_partition( p4est, P4EST_TRUE, nullptr );

	// Create the ghost (cell) and node structures.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize the neighbor nodes structure.
	my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
	my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
	nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
										// how much we'll neeed the node neighbors.

	/////////////////////////////////////////// Generating the dataset /////////////////////////////////////////////////

	parStopWatch watch;

	printf( ">> Beginning to generate flower dataset for %i grid points, and h = %f, in a [%f,%f]^2 domain\n",
		 	nGridPoints, H, minVal, -minVal );
	watch.start();

	std::string DATA_PATH = "/Users/youngmin/Documents/CS/CASL/LSCurvatureML/data/updated_flower/";
	const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "ihk"		// Replaced old "h" by interpolated h*kappa in the last column.
	};
	const int NUM_COLUMNS = 11;

	DATA_PATH += std::to_string( resolution ) + "/";
	checkOrCreateDirectory( DATA_PATH );								// Resolution directory (256/, 266/, or 276/).

	char auxA[10];
	sprintf( auxA, "%.3f", star.getA() );
	DATA_PATH += "a_" + std::string( auxA ) + "/";
	checkOrCreateDirectory( DATA_PATH );								// Smoothness directory (a_0.050/ or a_0.075/).

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

	outputFile.precision( 15 );											// Precision for floating point numbers.

	// Prepare file where to write sample grid point cartesian coordinates.
	std::ofstream pointsFile;
	std::string pointsFileName = DATA_PATH + "points_iter" + std::to_string( iter ) + ".csv";
	pointsFile.open( pointsFileName, std::ofstream::trunc );
	if( !pointsFile.is_open() )
		throw std::runtime_error( "Point cartesian coordinates file " + pointsFileName + " couldn't be opened!" );

	std::ostringstream headerPointsStream;								// Write output file header.
	headerPointsStream << R"("x","y")";
	pointsFile << headerPointsStream.str() << std::endl;

	pointsFile.precision( 15 );

	// Prepare file where to write the params.
	std::ofstream paramsFile;
	std::string paramsFileName = DATA_PATH + "params_iter" + std::to_string( iter ) + ".csv";
	paramsFile.open( paramsFileName, std::ofstream::trunc );
	if( !paramsFile.is_open() )
		throw std::runtime_error( "Params file " + paramsFileName + " couldn't be opened!" );

	std::ostringstream headerParamsStream;
	headerParamsStream << R"("nGridPoints","h","minVal","a","b","p")";
	paramsFile << headerParamsStream.str() << std::endl;

	paramsFile.precision( 15 );
	paramsFile << nGridPoints << "," << H << "," << minVal << "," << star.getA() << "," << star.getB() << ","
			   << star.getP() << std::endl;

	// Prepare file where to write the angle parameter for corresponding points on interface.
	std::ofstream anglesFile;
	std::string anglesFileName = DATA_PATH + "angles_iter" + std::to_string( iter ) + ".csv";
	anglesFile.open( anglesFileName, std::ofstream::trunc );
	if( !anglesFile.is_open() )
		throw std::runtime_error( "Angles file " + anglesFileName + " couldn't be opened!" );

	anglesFile << R"("theta")" << std::endl;							// Write header.
	anglesFile.precision( 15 );

	// A ghosted parallel PETSc vector to store level-set function values.
	Vec phi;
	ierr = VecCreateGhostNodes( p4est, nodes, &phi );
	CHKERRXX( ierr );

	// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
	sample_cf_on_nodes( p4est, nodes, star, phi );

	// Prepare scaffold to collect nodal indices along the interface.  Also, collect indicators for those nodes that
	// are adjacent to the interface and have a full 9-point uniform neighborhood.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, 0 );

	// Curvature and normal ghosted parallel vectors.
	Vec curvature, normal[P4EST_DIM];
	ierr = VecDuplicate( phi, &curvature );
	CHKERRXX( ierr );
	for( auto& dim : normal )
	{
		VecCreateGhostNodes( p4est, nodes, &dim );
		CHKERRXX( ierr );
	}

	// Reinitialize level-set by solving the PDE equation using the given number of iterations.
	my_p4est_level_set_t ls( &nodeNeighbors );
	ls.reinitialize_2nd_order( phi, iter );

	// Compute curvature with reinitialized data, which will be interpolated at the interface.
	compute_normals( nodeNeighbors, phi, normal );
	compute_mean_curvature( nodeNeighbors, phi, normal, curvature );

	// Prepare interpolation.
	my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
	interpolation.set_input( curvature, linear );

	// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are the
	// points we'll use to create our sample files.
	std::vector<p4est_locidx_t> indices;
	nodesAlongInterface.getIndices( &phi, indices );

	// Getting the full uniform stencils of interface points.
	const double *phiReadPtr = nullptr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	// Now, collect samples with reinitialized level-set function values, target h*kappa, and numerically interpolated
	// h*kappa on Gamma.
	int nSamples = 0;
	double maxRE = 0;							// Max relative (absolute) error.
	std::vector<std::vector<double>> samples;

	for( auto n : indices )
	{
		std::vector<p4est_locidx_t> stencil;	// Contains 9 nodal indices in 2D, but order is different to what
		try										// is being used here
		{
			if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
			{
				double xOnGamma, yOnGamma;		// Approximated coords. of stencil center node projection onto Gamma.
				std::vector<double> distances;	// Holds the signed distances for error measuring.
				std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
																			nodes, &nodeNeighbors, phiReadPtr, star,
																			gen, normalDistribution,
																			pointsFile, anglesFile, distances,
																			xOnGamma, yOnGamma );
				sample[NUM_COLUMNS - 1] = H * interpolation( xOnGamma, yOnGamma );	// Attach interpolated h*kappa.
				distances.push_back( 0 );											// Dummy column (here unused).

				// Fixing the order of columns for backwards compatibility.
				// Expected: (i-1,j+1), (i,j+1), (i+1,j+1), (i-1,j), (i,j), (i+1,j), (i-1,j-1), (i,j-1), (i+1,j-1)
				//           mp       , 0p     , pp       , m0     , 00   , p0     , mm       , 0m     , pm
				// Obtained: mm       , m0     , mp       , 0m     , 00   , 0p     , pm       , p0     , pp
				// Also, check error the reinitialization method using the "true" distances from above.
				std::vector<double> copySample( sample );
				int j = 2;
				for( int i = 0; i < NUM_COLUMNS - 2; i++ )
				{
					double error = ( distances[i] - sample[i] ) / H;
					maxRE = MAX( maxRE, ABS( error ) );

					sample[i] = copySample[j % 10];
					j += 3;
				}

				samples.push_back( sample );
				nSamples++;
			}
		}
		catch( std::exception &e )
		{
			double xyz[P4EST_DIM];				// Position of node at the center of the stencil.
			node_xyz_fr_n( n, p4est, nodes, xyz );
			std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
		}
	}

	// Write to file the samples content.
	for( const std::vector<double>& row : samples )
	{
		std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
		outputFile << row.back() << std::endl;
	}

	// Write paraview file to visualize the star interface and nodes following along.
	std::ostringstream oss;
	oss << "old_mlds_star_" << resolution << "_a_" << std::string( auxA ) << "_iter_" << iter;
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							1, 0, oss.str().c_str(),
							VTK_POINT_DATA, "phi", phiReadPtr );
	my_p4est_vtk_write_ghost_layer( p4est, ghost );

	printf( "<< Finished generating %lu samples with maxRE = %f in %f secs.\n",
		 samples.size(), maxRE, watch.get_duration_current() );

	// Clean up.
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
	ierr = VecDestroy( phi );
	CHKERRXX( ierr );

	// Destroy the p4est and its connectivity structure.
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	p4est_connectivity_destroy( connectivity );

	// Close files.
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

		// 256 | Smooth = 107, Sharp = 120.
		// 266 | Smooth = 111, Sharp = 124.
		// 276 | Smooth = 114, Sharp = 129.
		int iterations = 5;					// Choose 5, 10, and 20 to test neural networks dictionary.
		int resolution = 256;				// Choose 256, 266, 276 as from above.
		int nGridNodes = 107;				// Choose 107 - 129 as from above.
		double a;
		if( nGridNodes == 107 || nGridNodes == 111 || nGridNodes == 114 )
			a = 0.05;
		else if( nGridNodes == 120 || nGridNodes == 124 || nGridNodes == 129 )
			a = 0.075;
		else
			throw std::runtime_error( "Wrong combination of parameters!" );

		geom::Star star( a, 0.15, 3 );		// Smooth a = 0.05. Sharp a = 0.075.

		generateReinitializedFlowerDataset( star, resolution, nGridNodes, iterations, mpi );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}