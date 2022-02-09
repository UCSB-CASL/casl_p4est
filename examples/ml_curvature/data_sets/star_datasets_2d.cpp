/**
 * Generate star-shaped interface data sets for evaluating our neural network offline, on the python project.
 * For each reinitialization scheme, generate files with reinitialized level-set function values and normal unit vector
 * components (where the negative-curvature-normalized stencil is reoriented in such a way that the gradient forms an
 * angle between 0 and pi/2 with respect to the horizontal), point coordinates, and projection angles of nodes detected
 * along the interface.
 * Along the way, we compute exact distances for error validation.  These distances are obtained by first using
 * bisection for bracketing and then Newton-Raphson's method to refine the zero of the derivative of the minimum
 * distance function betwen a point (i.e. nodes along Gamma) and the star-shaped interface.
 *
 * Note: Even though the stencils are reoriented after negative-curvature normalization, the ihk column in the data sets
 * is untouched so that we can recover the sign appropriately upon inference.  When inputing data to the neural network,
 * you must flip the ihk sign if it's positive, and there's no need to reorient the stencil (it's already in the
 * expected form).
 *
 * The output files generated are placed in the [output_dir]/L/star/a_X.XXX/b_Y.YYY, where L is the maximum level of
 * refinement, X.XXX is the arm amplitude, and Y.YYY is the base radius.  The [output_dir]/L/star directory must exist.
 * The rest of subdirectories are created if they don't exist.  Files are overwritten ([R] = number of reinitialization
 * iterations):
 * a) iter[R]_params.csv Star parameters.
 * b) iter[R]_angles.csv Angles of stencil center nodes' projected onto the interface.
 * c) iter[R]_points.csv Cartesian coordinates of stencil center nodes.
 * d) iter[R]_inputs.csv Level-set values, normal components, and numerical bilinearly interpolated hk at the interface.
 *
 * Developer: Luis Ángel.
 * Date: July 22, 2020.
 * Updated: November 11, 2021.
 *
 * [Update on May 3, 2021] Adapted code to handle data sets where the gradient of the negative-curvature stencil has an
 * angle in the range [0, 2pi].  That is, we collect samples where the gradient points towards the first quadrant of
 * the local coordinate system centered at the 00 node.
 * [Update on October 29, 2021].  Extended data set to include normal vector components.  Removed exact signed distance
 * values as we are following an error-correcting approach.
 */

// System.
#include <stdexcept>
#include <iostream>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_fast_sweeping.h>
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_fast_sweeping.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_curvature_ml.h>
#endif

#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include "local_utils.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<unsigned short> maxRL( pl, 7, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 7)" );
	param_t<unsigned int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<std::string> outputDir( pl, "/Volumes/YoungMinEXT/k_ecnet_data", "outputDir", "Path where files will be written (default: same folder as the executable)" );
	param_t<bool> verbose( pl, true, "verbose", "Show or not debugging messages (default: 1)" );
	param_t<bool> exportVTK( pl, true, "exportVTK", "Export VTK file (default: 1)" );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// sets to files.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating star datasets for offline inference on Python" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const int NUM_COLUMNS = (P4EST_DIM + 1) * num_neighbors_cube + 2;		// Number of columns in dataset (includes phi and normal components).
		std::string COLUMN_NAMES[NUM_COLUMNS];				// Column headers following the x-y truth table of 3-state variables.
		kml::utils::generateColumnHeaders( COLUMN_NAMES );

		// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
		std::mt19937 gen{}; 				// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
		std::normal_distribution<double> normalDistribution;					// Used for bracketing and root finding.

		const double H = 1. / (1 << maxRL());

		/////////////////////////////////////// Star-shaped interface parameters ///////////////////////////////////////

		// Steep curvature parameters: check the results folder in the python project for other smoother parameters.
//		A = 0.085, B = 0.300 for level 6.
//		A = 0.120, B = 0.305 for level 7.
//		A = 0.170, B = 0.330 for level 8.
//		A = 0.225, B = 0.355 for level 9.
//		A = 0.258, B = 0.356 for level 10.
//		A = 0.274, B = 0.345 for level 11.
		const double A = 0.120;
		const double B = 0.305;
		const int ARMS = 5;

		geom::Star star( A, B, ARMS );			// Define the star interface and determine domain bound based on it.
		const double STAR_SIDE_LENGTH = star.getInscribingSquareSideLength() + 4 * H;
		const double MIN_D = -ceil( STAR_SIDE_LENGTH / 0.5 ) * 0.5;
		const double MAX_D = -MIN_D;			// The canonical space is square and has integral side lengths.

		//////////////////////////////////////////// Prepare output files //////////////////////////////////////////////

		// Check for output directory, e.g. "[outputDir]/star/a_0.075/b_0.350/".  If it doesn't exist, create it.
		std::string DATA_PATH = outputDir() + "/" + std::to_string( maxRL() ) + "/star/";		// Destination folder.
		char auxA[10], auxB[10];
		sprintf( auxA, "%.3f", A );
		DATA_PATH += "a_" + std::string( auxA ) + "/";
		create_directory( DATA_PATH, mpi.rank() );			// First part of path: a.
		sprintf( auxB, "%.3f", B );
		DATA_PATH += "b_" + std::string( auxB ) + "/";
		create_directory( DATA_PATH, mpi.rank() );			// Second part of path: b.

		// Prepare files where to write samples.
		std::string prefix = "iter" + std::to_string( reinitNumIters() ) + "_";

		std::ofstream inputsFile;							// Inputs file.
		inputsFile.open( DATA_PATH + prefix + "inputs.csv", std::ofstream::trunc );
		if( !inputsFile.is_open() )
			throw std::runtime_error( "Input values output file couldn't be opened!" );

		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )			// Write header: phi values, normal components, hk, and ihk.
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		inputsFile << headerStream.str() << std::endl;
		inputsFile.precision( 15 );

		std::ofstream pointsFile;							// Point coordinates file.
		pointsFile.open( DATA_PATH + prefix + "points.csv", std::ofstream::trunc );
		if( !pointsFile.is_open() )
				throw std::runtime_error( "Point coordinates file couldn't be opened!" );

		pointsFile.precision( 15 );
		pointsFile << R"("i","x","y")" << std::endl;		// Write header: node index, x, and y coords.

		std::ofstream anglesFile;							// Angles file.
		anglesFile.open( DATA_PATH + prefix + "angles.csv", std::ofstream::trunc );
		if( !anglesFile.is_open() )
				throw std::runtime_error( "Angles file couldn't be opened!" );

		anglesFile.precision( 15 );
		anglesFile << R"("theta")" << std::endl;			// Write header: the theta polar coord.

		std::ofstream paramsFile;							// Params file.
		paramsFile.open( DATA_PATH + prefix + "params.csv", std::ofstream::trunc );
		if( !paramsFile.is_open() )
			throw std::runtime_error( "Params file couldn't be opened!" );

		std::ostringstream headerParamsStream;
		headerParamsStream << R"("minD","maxD","sideLength","a","b","p")";
		paramsFile << headerParamsStream.str() << std::endl;

		paramsFile.precision( 15 );
		paramsFile << MIN_D << "," << MAX_D << "," << STAR_SIDE_LENGTH << "," << star.getA() << "," << star.getB()
				   << "," << star.getP() << std::endl;

		double maxRE = 0;									// Track the (h-relative) maximum absolute error.

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began to generate datasets for a star-shaped interface with maximum refinement "
								 "level of %i and H = %f\n", maxRL(), H );

		parStopWatch watch;
		watch.start();

		// Domain information.
		const int N_TREES = (int)(MAX_D - MIN_D);
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};			// Trees per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain bounds.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the non-signed distance level-set function to be reinitialized.
		splitting_criteria_cf_and_uniform_band_t levelSetSC( MAX( 1, maxRL() - 5 ), maxRL(), &star, 7.0, 1.2 );

		// Create the forest using a level set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)(&levelSetSC);

		// Refine and recursively partition forest.
		for( int i = 0; i < maxRL(); i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
		nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
											// how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values to be reinitialized.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, star, phi );

		// Reinitialize level-set function.
		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, (int)reinitNumIters() );

		// Compute numerical curvature and normal unit vectors.
		Vec curvature, normal[P4EST_DIM], hk;
		CHKERRXX( VecDuplicate( phi, &curvature ) );
		CHKERRXX( VecDuplicate( curvature, &hk ) );
		for( auto& dim : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

		compute_normals( nodeNeighbors, phi, normal );
		compute_mean_curvature( nodeNeighbors, phi, normal, curvature );

		// Let's get the dimensionless curvature for all points.
		double *hkPtr;
		const double *curvatureReadPtr;
		CHKERRXX( VecGetArray( hk, &hkPtr ) );
		CHKERRXX( VecGetArrayRead( curvature, &curvatureReadPtr ) );

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
			hkPtr[n] = curvatureReadPtr[n] * H;

		Vec interfaceFlag;		// Flag vector gets initialized to zero automatically.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &interfaceFlag ) );

		// Prepare curvature bilinear interpolation at projected points on Gamma.
		my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
		interpolation.set_input( hk, interpolation_method::linear );

		// Prepare scaffold to collect nodal indices along the interface.  Also, collect indicators for
		// nodes adjacent to the interface having full 9-point uniform stencils.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, (char)maxRL() );
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		double *interfaceFlagPtr;
		CHKERRXX( VecGetArray( interfaceFlag, &interfaceFlagPtr ) );

		const double *phiReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

		const double *normalReadPtr[P4EST_DIM];
		for( int dim = 0; dim < P4EST_DIM; dim++ )
			CHKERRXX( VecGetArrayRead( normal[dim], &normalReadPtr[dim] ) );

		double minHK = std::numeric_limits<double>::max();		// Let's keep track of min and max expected dimensionless
		double maxHK = 0;										// curvatures (in absolute value).

		// Now, collect samples.
		int nSamples = 0;
		std::vector<std::vector<double>> samples;
		double pOnGamma[P4EST_DIM];
		std::unordered_map<p4est_locidx_t, Point2> visitedNodes( nodes->num_owned_indeps );	// Memoization.

		for( auto n : indices )
		{
			try
			{
				std::vector<p4est_locidx_t> stencil;			// Contains 9 nodal indices in 2D.
				if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
				{
					double tgtHK;
					std::vector<double> distances;	// Holds the signed distances for error measuring.
					std::vector<double> sample = kutils::sampleNodeNextToStarInterface( n, NUM_COLUMNS, H, stencil,
																						p4est, nodes, phiReadPtr, star,
																						gen, normalDistribution,
																						&pointsFile, &anglesFile,
																						distances, pOnGamma,
																						visitedNodes, normalReadPtr,
																						tgtHK, verbose() );

					for( const auto& dim : normalReadPtr )		// Attach the normal components too.
					{
						for( auto s: stencil )
							sample.push_back( dim[s] );
					}

					sample.push_back( tgtHK );
					sample.push_back( interpolation( DIM(pOnGamma[0], pOnGamma[1], pOnGamma[2]) ) );	// Attach numerical ihk.

					// Flip sign of stencil if interpolated curvature at the interface is positive.
					if( sample[NUM_COLUMNS - 1] > 0 )
					{
						for( int i = 0; i < NUM_COLUMNS - 2; i++ )	// Avoid flipping sign of target and interpolated k.
						{											// because ihk is still used during inference.
							sample[i] *= -1.0;
							distances[i] *= -1.0;
						}
					}

					// Error metric for validation.
					for( int i = 0; i < num_neighbors_cube; i++ )
					{
						double error = distances[i] - sample[i];
						maxRE = MAX( maxRE, ABS( error ) );
					}

					// Rotate stencil so that gradient at node 00 has an angle in first quadrant.
					kml::utils::rotateStencilToFirstQuadrant( sample );

					samples.push_back( sample );
					nSamples++;

					// Set valid node-along-interface and grad indicators just in iter10.
					interfaceFlagPtr[n] = 1;

					minHK = MIN( minHK, ABS( sample[NUM_COLUMNS - 2] ) );		// Min and max expected absolute
					maxHK = MAX( maxHK, ABS( sample[NUM_COLUMNS - 2] ) );		// curvatures.
				}
			}
			catch( std::exception &e )
			{
				double xyz[P4EST_DIM];				// Position of node at the center of the stencil.
				node_xyz_fr_n( n, p4est, nodes, xyz );
				std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
			}
		}

		// Write samples collected.
		for( const auto& row : samples )
		{
			std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( inputsFile, "," ) );		// Inner elements.
			inputsFile << row.back() << std::endl;
		}

		PetscPrintf( mpi.comm(), "   Generated %i samples\n"
								 "   Max h-normalized absolute error: %f\n"
								 "   Min hk = %f,    Max hk = %f\n"
								 "   Timing: %f\n",	nSamples, maxRE, minHK, maxHK, watch.get_duration_current() );
		watch.stop();

		PetscPrintf( mpi.comm(), "<< Done!\n" );

		// Write paraview file to visualize the star interface and nodes following it along.
		if( exportVTK() )
		{
			std::ostringstream oss;
			oss << "star_a_" << std::string( auxA ) << "_b_" << std::string( auxB );
			my_p4est_vtk_write_all( p4est, nodes, ghost,
									P4EST_TRUE, P4EST_TRUE,
									3, 0, oss.str().c_str(),
									VTK_POINT_DATA, "phi", phiReadPtr,
									VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr,
									VTK_POINT_DATA, "hk", hkPtr );
		}

		// Cleaning up.
		CHKERRXX( VecRestoreArrayRead( curvature, &curvatureReadPtr ) );
		CHKERRXX( VecRestoreArray( hk, &hkPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
		for( int dim = 0; dim < P4EST_DIM; dim++ )
			CHKERRXX( VecRestoreArrayRead( normal[dim], &normalReadPtr[dim] ) );
		CHKERRXX( VecRestoreArray( interfaceFlag, &interfaceFlagPtr ) );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		CHKERRXX( VecDestroy( phi ) );
		CHKERRXX( VecDestroy( interfaceFlag ) );
		CHKERRXX( VecDestroy( curvature ) );
		CHKERRXX( VecDestroy( hk ) );
		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );

		// Closing file objects that were open for input values, points, angles, and params.
		inputsFile.close();
		pointsFile.close();
		anglesFile.close();
		paramsFile.close();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}