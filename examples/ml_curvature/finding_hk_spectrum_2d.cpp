/**
 * Finding the bounds for |hk| to determine where the numerical method excels when computing curvature.  Then, we can
 * train a neural network only for samples where -2/3 <= -|hk| <= -|min_hk|.  These bounds are always negative because
 * we only deal with the negative curvature spectrum.  The maximum hk = 2/3 corresponds to a circle of radius 1.5h.
 * Beyond that, the computed value with the nnet is not guaranteed to be accurate due to under-resolution.
 *
 * To explore the hk spectrum, we run different combinations of star-shaped interface parameters and write a file with
 * exact and numerically estimated hk.
 *
 * Developer: Luis Ángel.
 * Created: October 22, 2021.
 * Updated: October 23, 2021.
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
	param_t<unsigned short> maxRL( pl, 10, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<unsigned int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<std::string> outputDir( pl, ".", "outputDir", "Path where file will be written (default: same folder as the executable)" );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// set files.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Finding hk spectrum" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double MIN_D = -1, MAX_D = -MIN_D;			// The canonical space is [-1, 1]^{P4EST_DIM}.
		const int NUM_COLUMNS = num_neighbors_cube + 2;		// Number of columns in a curvature sample (including numerical and expected hk).
		std::string COLUMN_NAMES[NUM_COLUMNS];				// Column headers following the x-y truth table of 3-state variables.
		kutils::generateColumnHeaders( COLUMN_NAMES );

		// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
		std::mt19937 gen{}; 				// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
		std::normal_distribution<double> normalDistribution;					// Used for bracketing and root finding.

		const double H = 1. / (1 << maxRL());

		/////////////////////////////////////// Star-shaped interface parameters ///////////////////////////////////////

		const int NUM_STAR_PARAM_PAIRS = 3;
//		const double A[NUM_STAR_PARAM_PAIRS] = {0.075, 0.111, 0.120};	// For level 6, 7, and 8.
//		const double B[NUM_STAR_PARAM_PAIRS] = {0.350, 0.314, 0.305};
		const double A[NUM_STAR_PARAM_PAIRS] = {0.120, 0.2, 0.225};		// For level 9, 10.
		const double B[NUM_STAR_PARAM_PAIRS] = {0.305, 0.35, 0.355};
		const int ARMS[] = {5};
		const double PHASE_STEP = M_PI / 3;
		const int NUM_PHASE_STEPS = (int)(2 * M_PI / PHASE_STEP);

		///////////////////////////////////////////// Prepare output files /////////////////////////////////////////////

		std::ofstream outputFile;
		outputFile.open( outputDir() + "/exploring_hk_spectrum.csv", std::ofstream::trunc );
		if( !outputFile.is_open() )
			throw std::runtime_error( "Output file couldn't be opened!" );

		std::ostringstream headerStream;							// Write output file header.
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		outputFile << headerStream.str() << std::endl;
		outputFile.precision( 15 );									// Precision for floating point numbers.

		std::ofstream anglesFile;									// Here we store the projection angle onto star shape.
		anglesFile.open( outputDir() + "/exploring_hk_angles.csv", std::ofstream::trunc );
		if( !anglesFile.is_open() )
			throw std::runtime_error( "Angles file couldn't be opened!" );

		anglesFile.precision( 15 );
		anglesFile << R"("theta")" << std::endl;					// Write header: the theta polar coord.

		/////////////////////////////////////////////// Begin exploration //////////////////////////////////////////////

		// Domain information constant thoughout simulations.
		int n_xyz[] = {2, 2, 2};									// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};					// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};									// Non-periodic.


		PetscPrintf( mpi.comm(), ">> Began sampling star-shaped interfaces with maximum refinement level %i and H = %f\n",
					 maxRL(), H );

		for( int paramIdx = 0; paramIdx < NUM_STAR_PARAM_PAIRS; paramIdx++ )	// Varying base parameters.
		{
			for( const auto& nArms : ARMS )										// Varying number of arms.
			{
				parStopWatch watch;
				watch.start();

				ierr = PetscPrintf( mpi.comm(), ":: Gathering data for A = %.3f, B = %.3f, nArms = %i\n",
									A[paramIdx], B[paramIdx], nArms );
				CHKERRXX( ierr );

				double minK = std::numeric_limits<double>::max();		// Let's keep track of min and max expected
				double maxK = 0;										// curvatures (in absolute value).
				int nSamples = 0;

				for( int phaseIdx = 0; phaseIdx < NUM_PHASE_STEPS; phaseIdx++ )	// Varying angular phase.
				{
					// Star-shaped interface.
					geom::Star star( A[paramIdx], B[paramIdx], nArms, phaseIdx * PHASE_STEP );
					const double STAR_SIDE_LENGTH = star.getInscribingSquareSideLength();
					if( STAR_SIDE_LENGTH > MAX_D - MIN_D - 4 * H )				// Consider also 2H padding on each side.
						throw std::runtime_error( "Star exceeds allowed space in domain!" );

					// p4est variables and data structures.
					p4est_t *p4est;
					p4est_nodes_t *nodes;
					my_p4est_brick_t brick;
					p4est_ghost_t *ghost;
					p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

					// Definining the non-signed distance level-set function to be reinitialized.
					splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, maxRL(), &star, 4 );

					// Create the forest using a level set as refinement criterion.
					p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
					p4est->user_pointer = (void *)(&levelSetSC);

					// Refine and recursively partition forest.
					my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );
					my_p4est_partition( p4est, P4EST_TRUE, nullptr );

					// Create the ghost (cell) and node structures.
					ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
					nodes = my_p4est_nodes_new( p4est, ghost );

					// Initialize the neighbor nodes structure.
					my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
					my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
					nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
														// how much we'll neeed the node neighbors.

					// A ghosted parallel PETSc vector to store level-set function values.
					Vec phi;
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

					// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
					sample_cf_on_nodes( p4est, nodes, star, phi );

					// Reinitialize level-set function.
					my_p4est_level_set_t ls( &nodeNeighbors );
					ls.reinitialize_2nd_order( phi, (int)reinitNumIters() );

					// Compute numerical curvature and normal unit vectors.
					Vec curvature, normal[P4EST_DIM];
					CHKERRXX( VecDuplicate( phi, &curvature ) );
					for( auto& dim : normal )
						CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

					compute_normals( nodeNeighbors, phi, normal );
					compute_mean_curvature( nodeNeighbors, phi, normal, curvature );

					// Prepare curvature bilinear interpolation at projected points on Gamma.
					my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
					interpolation.set_input( curvature, interpolation_method::linear );

					// Prepare scaffold to collect nodal indices along the interface.  Also, collect indicators for
					// nodes adjacent to the interface having full 9-point uniform stencils.
					NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, (signed char)maxRL() );
					std::vector<p4est_locidx_t> indices;
					nodesAlongInterface.getIndices( &phi, indices );

					// Getting the full uniform stencils of interface points.
					const double *phiReadPtr = nullptr;
					CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

					const double *normalReadPtr[P4EST_DIM];
					for( int dim = 0; dim < P4EST_DIM; dim++ )
						CHKERRXX( VecGetArrayRead( normal[dim], &normalReadPtr[dim] ) );

					std::vector<std::vector<double>> samples;
					std::unordered_map<p4est_locidx_t, Point2> visitedNodes( nodes->num_owned_indeps );	// Memoization.

					for( const auto& n : indices )
					{
						std::vector<p4est_locidx_t> stencil;	// Contains 9 nodal indices in 2D.
						try
						{
							if( nodesAlongInterface.getFullStencilOfNode( n, stencil ) )
							{
								double pOnGamma[P4EST_DIM];
								double gradient[P4EST_DIM] = {DIM( normalReadPtr[0][n], normalReadPtr[1][n], normalReadPtr[2][n] )};
								std::vector<double> distances;	// Not doing anything here with exact distances.
								std::vector<double> sample = kutils::sampleNodeNextToStarInterface( n, NUM_COLUMNS, H, stencil, p4est, nodes,
																									phiReadPtr, star, gen, normalDistribution,
																									nullptr, &anglesFile, distances, pOnGamma,
																									visitedNodes, normalReadPtr );
								sample[NUM_COLUMNS - 1] = H * interpolation( DIM( pOnGamma[0], pOnGamma[1], pOnGamma[2] ) );	// Attach interpolated h*kappa.
								distances.push_back( 0 );																		// Dummy column.

								// Flip sign of stencil if interpolated curvature at the interface is positive.
								if( sample[NUM_COLUMNS - 1] > 0 )
								{
									for( int i = 0; i < NUM_COLUMNS - 2; i++ )	// Avoid flipping sign of target and interpolated k.
									{											// because ihk is still used during inference.
										sample[i] *= -1.0;
										distances[i] *= -1.0;
									}

									for( auto& component : gradient )			// Flip sign of gradient too.
										component *= -1.0;
								}

								// Rotate stencil so that gradient at node 00 has an angle in first quadrant.
								kutils::rotateStencilToFirstQuadrant( sample, gradient );
								kutils::rotateStencilToFirstQuadrant( distances, gradient );

								samples.push_back( sample );
								nSamples++;

								minK = MIN( minK, ABS( sample[NUM_COLUMNS - 2] ) / H );		// Min and max expected
								maxK = MAX( maxK, ABS( sample[NUM_COLUMNS - 2] ) / H );		// absolute curvatures.
							}
						}
						catch( std::exception &e ) {}
					}

					// Write samples.
					for( const auto& row : samples )
					{
						std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
						outputFile << row.back() << std::endl;
					}

					// Restore access.
					CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
					for( int dim = 0; dim < P4EST_DIM; dim++ )
						CHKERRXX( VecRestoreArrayRead( normal[dim], &normalReadPtr[dim] ) );

					// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
					CHKERRXX( VecDestroy( phi ) );
					CHKERRXX( VecDestroy( curvature ) );
					for( auto& dim : normal )
						CHKERRXX( VecDestroy( dim ) );

					// Destroy the p4est and its connectivity structure.
					p4est_nodes_destroy( nodes );
					p4est_ghost_destroy( ghost );
					p4est_destroy( p4est );
					my_p4est_brick_destroy( connectivity, &brick );
				}

				PetscPrintf( mpi.comm(), "   Generated %d samples\n", nSamples );
				PetscPrintf( mpi.comm(), "   Min curvature = %f,    Max curvature = %f\n", minK, maxK );
				PetscPrintf( mpi.comm(), "   Timing = %f\n", watch.get_duration_current() );
				watch.stop();
			}
		}

		// Closing file objects.
		anglesFile.close();
		outputFile.close();

		PetscPrintf( mpi.comm(), "<<< Done!\n" );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}