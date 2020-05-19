/**
 * Generate datasets for training a feedforward neural network on spherical interfaces using samples
 * from a reinitialized level-set function with the fast sweeping method.
 * NOTE: This hasn't been tested on 3D, but the code is prepared for the 3D scenario.
 *
 * To collect samples for a signed distance function with spherical interface set the variable SIGN_DIST_FUNC to true.
 * To collect samples for a reinitialized level-set function with the fast sweeping method set SIGN_DIST_FUNC to false.
 *
 * Developer: Luis Ángel.
 * Date: May 12, 2020.
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
//#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_fast_sweeping.h>
#include <src/my_p4est_nodes_along_interface.h>
//#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iterator>
#include <fstream>

/**
 * Generate the column headers following the truth-table order with x changing slowly, then y changing faster than x,
 * and finally z changing faster than y.  Each dimension has three states: m, 0, and p (minus, center, plus).  For
 * example, in 2D, the columns that are generated are:
 * 	   Acronym      Meaning
 *		"mm"  =>  (i-1, j-1)
 *		"m0"  =>  (i-1, j  )
 *		"mp"  =>  (i-1, j+1)
 *		"0m"  =>  (  i, j-1)
 *		"00"  =>  (  i,   j)
 *		"0p"  =>  (  i, j+1)
 *		"pm"  =>  (i+1, j-1)
 *		"p0"  =>  (i+1,   j)
 *		"pp"  =>  (i+1, j+1)
 *		"hk"  =>  h * \kappa
 * @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
 */
void generateColumnHeaders( std::string header[] )
{
	const int STEPS = 3;
	std::string states[] = {"m", "0", "p"};			// States for x, y, and z directions.
	int i = 0;
	for( int x = 0; x < STEPS; x++ )
		for( int y = 0; y < STEPS; y++ )
#ifdef P4_TO_P8
			for( int z = 0; z < STEPS; z++ )
#endif
		{
			i = SUMD( x * (int)pow( STEPS, P4EST_DIM - 1 ), y * (int)pow( STEPS, P4EST_DIM - 2 ), z );
			header[i] = SUMD( states[x], states[y], states[z] );
		}
	header[i+1] = "hk";								// Don't forget the h*\kappa column!
}

int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = 0, MAX_D = 1;										// The canonical space is [0,1]^{P4EST_DIM}.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.

	const int NUM_DIFF_CIRCLES_SAME_RAD = 5;								// How many different circles of same radius we generate.
	const int NUM_CIRCLES = (int)pow( 2, MAX_REFINEMENT_LEVEL ) - 5;		// Number of circles is proportional to finest resolution.
																			// and ensures at least 2 circles per finest quad/oct.
	const double MIN_RADIUS = 1.5 * H;			// Ensures at least 4 nodes inside smallest circle.
	const double MAX_RADIUS = HALF_D - 2 * H;	// Prevents sampling interface nodes with invalid full uniform stencils.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/fsm/data/";			// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 					// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// Check again.  To generate datasets we don't admit more than a single process to avoid race conditions when
		// writing datasets to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Determining if we want samples from a signed distance function or from a reinitialized level-set.
		int answer;
		std::cout << ":: Do you want samples from a signed distance function [1]? Or from a reinitialized level-set function [0]? ";
		do
		{
			answer = getchar();
		}
		while( answer == 10 || answer == 13 );

		const bool SIGN_DIST_FUNC = answer == '1';
		if( SIGN_DIST_FUNC )
			std::cout << "Collecting samples from a signed distance function..." << std::endl;
		else
			std::cout << "Collecting samples from a reinitialized level-set function using fast sweeping method..." << std::endl;

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i spheres with maximum refinement level of %i and finest h = %f\n",
			NUM_CIRCLES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples file: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream outputFile;
		std::string fileName = DATA_PATH + ( SIGN_DIST_FUNC? "sdf_" : "rls_" ) + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		outputFile.open( fileName, std::ofstream::trunc );
		if( !outputFile.is_open() )
			throw std::runtime_error( "Output file " + fileName + " couldn't be opened!" );

		// Write column headers: enforcing strings by adding quotes around them.
		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		outputFile << headerStream.str() << std::endl;

		outputFile.precision( 15 );							// Precision for floating point numbers.

		// Variables to control the spread of circles' radii, which must vary uniformly from MIN_RADIUS to MAX_RADIUS.
		double distance = MAX_RADIUS - MIN_RADIUS;			// Circles' radii are in [1.5*H, 0.5-2H], inclusive.
		double linspace[NUM_CIRCLES];
		for( int i = 0; i < NUM_CIRCLES; i++ )				// Uniform linear space from 0 to 1, with NUM_CIRCLES steps.
			linspace[i] = (double)( i ) / ( NUM_CIRCLES - 1.0 );

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[] = {1, 1, 1};									// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};					// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};									// Non-periodic domain.

		int nSamples = 0;
		int nc = 0;								// Keeps track of number of circles whose samples has been collected.
		while( nc < NUM_CIRCLES )
		{
			const double R = MIN_RADIUS + linspace[nc] * distance;	// Circle radius to be evaluated.
			const double H_KAPPA = H / R;							// Expected dimensionless curvature: h\kappa = h/r.
			std::vector<std::vector<double>> samples;

			// Generate a given number of randomly centered circles with the same radius.
			int randomnessCount = 0;
			double maxRE = 0;										// Maximum relative error.
			while( randomnessCount < NUM_DIFF_CIRCLES_SAME_RAD )
			{
				const double C[] = {
					DIM( HALF_D + uniformDistribution( gen ),		// Center coords are randomly chosen
						 HALF_D + uniformDistribution( gen ),		// around the center of the grid.
						 HALF_D + uniformDistribution( gen ) )
				 };

				// p4est variables and data structures: these change with every single circle because we must refine the
				// trees according to the new circle's center and radius.
				p4est_t *p4est;
				p4est_nodes_t *nodes;
				my_p4est_brick_t brick;
				p4est_ghost_t *ghost;
				p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

				// Definining the non-signed distance level-set function to be reinitialized.  Remains unused if
				// SIGN_DIST_FUNC is true.
				geom::SphereNSD sphereNsd( DIM( C[0], C[1], C[2] ), R );
				geom::Sphere sphere( DIM( C[0], C[1], C[2] ), R );	// Signed-distance level-set function for error checking.
				splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, SIGN_DIST_FUNC? &sphere : &sphereNsd );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = ( void * ) ( &levelSetSC );

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

				// A ghosted parallel PETSc vector to store level-set function values.
				Vec phi;
				ierr = VecCreateGhostNodes( p4est, nodes, &phi );
				CHKERRXX( ierr );

				// Calculate the level-set function values for each independent node (i.e. locally owned and ghost nodes).
				double *phiPtr;
				ierr = VecGetArray( phi, &phiPtr );
				CHKERRXX( ierr );
				for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
				{
					double xyz[P4EST_DIM];
					node_xyz_fr_n( i, p4est, nodes, xyz );
					if( SIGN_DIST_FUNC )
						phiPtr[i] = sphere( DIM( xyz[0], xyz[1], xyz[2] ) );	// Using the signed distance function.
					else
						phiPtr[i] = sphereNsd( DIM( xyz[0], xyz[1], xyz[2] ) );	// Using the non-signed distance function.
				}
				ierr = VecRestoreArray( phi, &phiPtr );
				CHKERRXX( ierr );

				// Reinitialize level-set function using the fast sweeping method.
				if( !SIGN_DIST_FUNC )
				{
					FastSweeping fsm;
					fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
					fsm.reinitializeLevelSetFunction( &phi, 8 );
//					my_p4est_level_set_t ls( &nodeNeighbors );
//					ls.reinitialize_2nd_order( phi, 5 );
				}

				// Once the level-set function is reinitialized (if user chose to), collect nodes on or adjacent to the
				// interface; these are the point we'll use to create our sample files.
				NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
				std::vector<p4est_locidx_t> indices;
				nodesAlongInterface.getIndices( &phi, indices );

				// Getting the full uniform stencils of interface points.
				const double *phiReadPtr;
				ierr = VecGetArrayRead( phi, &phiReadPtr );
				CHKERRXX( ierr );

				for( auto n : indices )
				{
					std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D, 27 values in 3D.
					std::vector<double> dataPve;			// Phi and h*kappa results in positive and negative form.
					std::vector<double> dataNve;
					dataPve.reserve( NUM_COLUMNS );			// Efficientize containers.
					dataNve.reserve( NUM_COLUMNS );
					try
					{
						if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
						{
							for( auto s : stencil )
							{
								dataPve.push_back( +phiReadPtr[s] );		// Positive curvature phi values.
								dataNve.push_back( -phiReadPtr[s] );		// Negative curvature phi values.

								double xyz[P4EST_DIM];						// Error checking.
								node_xyz_fr_n( s, p4est, nodes, xyz );
								double error = ABS( sphere( DIM( xyz[0], xyz[1], xyz[2] ) ) - phiReadPtr[s] ) / H;
								maxRE = MAX( maxRE, error );
							}

							// Appending the target h*\kappa.
							dataPve.push_back( +H_KAPPA );
							dataNve.push_back( -H_KAPPA );

							// Accumulating samples.
							samples.push_back( dataPve );
							samples.push_back( dataNve );
						}
					}
					catch( std::exception &e )
					{
						std::cerr << e.what() << std::endl;
					}
				}

				randomnessCount++;

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
			}

			// Write all samples collected for all circles with the same radius but randomized center content to file.
			for( const auto& row : samples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
				outputFile << row.back() << std::endl;
			}

			nc++;
			nSamples += samples.size();

			std::cout << "     (" << nc << ") Done with radius = " << R << ".  Maximum relative error = " << maxRE << std::endl;

			if( nc % 10 == 0 )
				printf( "   [%i circle groups evaluated after %f secs.]\n", nc, watch.get_duration_current() );
		}

		printf( "<< Finished generating %i circles and %i samples in %f secs.\n", nc, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}