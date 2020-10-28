/**
 * Generate datasets for training a feedforward neural network on spherical interfaces using samples from a signed
 * distance function and from a reinitialized level-set function.
 * NOTE: This hasn't been tested on 3D, but the code is prepared for the 3D scenario.
 *
 * The samples collected for signed and reinitialized level-set functions have one-to-one correlation.  That is, the nth
 * sample (i.e. nth row in both files) correspond to the same standing point adjacent to the interface and its 9-stencil
 * neighborhood.  This correspondance can be used, eventually, to train denoising-like autoencoders.
 *
 * The files generated are named sphere_[X]_Y.csv, where [X] is one of "sdf" or "rls", for signed distance function and
 * reinitialized level-set, respectively, and Y is the highest level of tree resolution.
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
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/casl_geometry.h>
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>


int main ( int argc, char* argv[] )
{
	////////////////////////////////////////////////////// Options /////////////////////////////////////////////////////

	const int N_GRID_POINTS = 266;			// Pick 256, 266, or 276.

	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = 0, MAX_D = 1;										// The canonical space is [0,1]^{P4EST_DIM}.
	const double HALF_D = (MAX_D - MIN_D) / 2;								// Half domain.
	const double H = (MAX_D - MIN_D) / (double)(N_GRID_POINTS - 1);			// Highest spatial resolution in x/y directions.

	const double MIN_RADIUS = 1.6 * H;										// Ensures at least 4 nodes inside smallest circle.
	const double MAX_RADIUS = HALF_D - 2 * H;
	const int N_CIRCLES = (int)(ceil( (N_GRID_POINTS - 8.2) / 2 ) + 1);		// Number of circles is proportional to finest resolution.
	const int TOTAL_RANDOMNESS = 5;											// How many different circles of same radius we generate.
	const int ITERS[] = { 0, 5, 10, 15, 20 };								// 0 iterations means use the exact signed distance function.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/LSCurvatureML/updated_data_ihk/";
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;					// Number of columns in resulting dataset.
	const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "ihk"		// Replaced old "h" by interpolated h*kappa in the last column.
	};

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// sets to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Collect samples from signed distance function and from reinitialized level-set.
		std::cout << "Collecting samples from level-set function with spherical interface..." << std::endl;

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[] = { N_GRID_POINTS - 1, N_GRID_POINTS - 1 };	// Number of cells in the macromesh per dimension.
		double xyz_min[] = { MIN_D, MIN_D };					// Square domain.
		double xyz_max[] = { MAX_D, MAX_D };
		int periodic[] = { 0, 0 };								// Non-periodic domain.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Create the forest.  (No need to assign p4est->user_pointer).
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );

		// Partition forest.  (No need for my_p4est_refine()).
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
		nodeNeighbors.init_neighbors();

		// Variables to control the spread of circles' radii, which must vary uniformly from MIN_RADIUS to MAX_RADIUS.
		double distance = MAX_RADIUS - MIN_RADIUS;			// Circles' radii are in [1.6*h, 0.5-2h].
		double spread[N_CIRCLES];
		for( int i = 0; i < N_CIRCLES; i++ )
			spread[i] = (double)( i ) / (N_CIRCLES - 1);	// Uniform distribution from 0 to 1, with N_CIRCLES steps, inclusive, to spread distances.

		for( const int& nIters : ITERS )	// Collect samples for several number of iterations.
		{
			parStopWatch watch;
			printf( ">> Beginning to generate dataset for %i circles, %i grid points, %i iterations, and h = %f, in a [0,1]^2 domain\n",
					N_CIRCLES, N_GRID_POINTS,  nIters, H );
			watch.start();

			// Standard mersenne_twister_engine with same seed for each iteration.
			std::mt19937 gen{};
			std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

			// Prepare samples file.
			std::ofstream outFile;
			std::string outFileName;
			if( nIters <= 0 )
				outFileName = DATA_PATH + "Distance Function/circlesDataset_" + std::to_string( N_GRID_POINTS ) + ".csv";
			else
				outFileName = DATA_PATH + "Non Distance Function/iter" + std::to_string( nIters ) + "/reinitDataset_m" + std::to_string( N_GRID_POINTS ) + ".csv";

			outFile.open( outFileName, std::ofstream::trunc );
			if( !outFile.is_open() )
				throw std::runtime_error( "Output file " + outFileName + " couldn't be opened!" );
			outFile.precision( 15 );							// Precision for floating point numbers.

			// Write column headers: enforcing strings by adding quotes around them.
			std::ostringstream headerStream;
			for( int i = 0; i < NUM_COLUMNS - 1; i++ )
				headerStream << "\"" << COLUMN_NAMES[i] << "\",";
			headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
			outFile << headerStream.str() << std::endl;

			int nSamples = 0;
			int nc = 0;					// Keeps track of number of circles whose samples has been collected.
			while( nc < N_CIRCLES )
			{
				const double R = MIN_RADIUS + spread[nc] * distance;	// Circle radius to be evaluated.
				const double H_KAPPA = H / R;							// Expected dimensionless curvature: h*kappa = h/r.
				std::vector<std::vector<double>> samples;

				// Generate a given number of randomly centered circles with the same radius and accumulate samples until we
				// reach a given maximum.
				double maxRE = 0;							// Maximum relative error.
				int randomnessCount = 0;
				while( randomnessCount < TOTAL_RANDOMNESS )	// Generate various randomly centered circles with same radius.
				{
					const double C[] = {
						HALF_D + uniformDistribution( gen ),	// Center coords are randomly chosen
						HALF_D + uniformDistribution( gen )		// around the center of the grid.
					};

					// Definining the non-signed distance level-set function to be reinitialized if needed, and always
					// the exact signed distance function to check errors.  (No need to create a splitting_criteria_cf_t).
					geom::SphereNSD sphereNsd( C[0], C[1], R );
					geom::Sphere sphere( C[0], C[1], R );

					// Ghosted parallel PETSc vectors to store level-set function values.  If nIters <= 0, rlsPhi = sdfPhi.
					Vec sdfPhi, rlsPhi;
					ierr = VecCreateGhostNodes( p4est, nodes, &sdfPhi );
					CHKERRXX( ierr );

					ierr = VecDuplicate( sdfPhi, &rlsPhi );
					CHKERRXX( ierr );

					Vec curvature, normal[P4EST_DIM];
					ierr = VecDuplicate( rlsPhi, &curvature );
					CHKERRXX( ierr );
					for( auto& dim : normal )
					{
						VecCreateGhostNodes( p4est, nodes, &dim );
						CHKERRXX( ierr );
					}

					// Calculate the level-set function values for all independent nodes.
					sample_cf_on_nodes( p4est, nodes, sphere, sdfPhi );
					if( nIters <= 0 )
						sample_cf_on_nodes( p4est, nodes, sphere, rlsPhi );
					else
						sample_cf_on_nodes( p4est, nodes, sphereNsd, rlsPhi );

					// Reinitialize level-set function using PDE equation only if we are not basing calculations on the
					// exact signed distance function.
					if( nIters > 0 )
					{
						my_p4est_level_set_t ls( &nodeNeighbors );
						ls.reinitialize_2nd_order( rlsPhi, nIters );
					}

					// Compute curvature with reinitialized data, which will be interpolated at the interface.
					compute_normals( nodeNeighbors, rlsPhi, normal );
					compute_mean_curvature( nodeNeighbors, rlsPhi, normal, curvature );

					// Prepare interpolation.
					my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
					interpolation.set_input( curvature, linear );

					// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these
					// are the points we'll use to create our sample files and compare with the signed distance function.
					NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, 0 );
					std::vector<p4est_locidx_t> indices;
					nodesAlongInterface.getIndices( &rlsPhi, indices );

					// Getting the full uniform stencils of interface points.
					const double *sdfPhiReadPtr, *rlsPhiReadPtr;
					ierr = VecGetArrayRead( sdfPhi, &sdfPhiReadPtr );
					CHKERRXX( ierr );
					ierr = VecGetArrayRead( rlsPhi, &rlsPhiReadPtr );
					CHKERRXX( ierr );

					for( auto n : indices )
					{
						std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
						std::vector<double> rlsDataPve;			// Phi and h*kappa results in positive form.
						rlsDataPve.reserve( NUM_COLUMNS );		// Efficientize container.
						try
						{
							if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
							{
								for( auto s : stencil )
								{
									rlsDataPve.push_back( rlsPhiReadPtr[s] );

									// Error.
									double error = ABS( sdfPhiReadPtr[s] - rlsPhiReadPtr[s] ) / H;
									maxRE = MAX( maxRE, error );	// Should be zero when rlsPhi == sdfPhi.
								}

								// Fixing the order of columns for backwards compatibility.
								// Expected: (i-1,j+1), (i,j+1), (i+1,j+1), (i-1,j), (i,j), (i+1,j), (i-1,j-1), (i,j-1), (i+1,j-1)
								//           mp       , 0p     , pp       , m0     , 00   , p0     , mm       , 0m     , pm
								// Obtained: mm       , m0     , mp       , 0m     , 00   , 0p     , pm       , p0     , pp
								std::vector<double> copyRlsDataPve( rlsDataPve );
								int j = 2;
								for( int i = 0; i < NUM_COLUMNS - 2; i++ )
								{
									rlsDataPve[i] = copyRlsDataPve[j % 10];
									j += 3;
								}

								// Appending the target h*kappa.
								rlsDataPve.push_back( H_KAPPA );

								// Appending the interpolated h*kappa.
								double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
								node_xyz_fr_n( n, p4est, nodes, xyz );
								double grad[P4EST_DIM];					// Getting its gradient (i.e. normal).
								const quad_neighbor_nodes_of_node_t *qnnnPtr;
								nodeNeighbors.get_neighbors( n, qnnnPtr );
								qnnnPtr->gradient( rlsPhiReadPtr, grad );
								double gradNorm = sqrt( SUMD( SQR( grad[0] ), SQR( grad[1] ), SQR( grad[2] ) ) );	// Get the unit gradient.

								for( int i = 0; i < P4EST_DIM; i++ )					// Translation: this is the location where
									xyz[i] -= grad[i] / gradNorm * rlsPhiReadPtr[n];	// we need to interpolate numerical curvature.

								double iHKappa = H * interpolation( DIM( xyz[0], xyz[1], xyz[2] ) );
								rlsDataPve.push_back( iHKappa );	// Attach interpolated h*kappa.

								// Getting negative version of samples.
								std::vector<double> rlsDataNve( NUM_COLUMNS );
								for( int i = 0; i < NUM_COLUMNS; i++ )
									rlsDataNve[i] = -rlsDataPve[i];

								// Accumulating samples.
								samples.push_back( rlsDataPve );
								samples.push_back( rlsDataNve );
							}
						}
						catch( std::exception &e )
						{
							std::cerr << e.what() << std::endl;
						}
					}

					randomnessCount++;

					// Cleaning up.
					ierr = VecRestoreArrayRead( sdfPhi, &sdfPhiReadPtr );
					CHKERRXX( ierr );

					ierr = VecRestoreArrayRead( rlsPhi, &rlsPhiReadPtr );
					CHKERRXX( ierr );

					// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
					ierr = VecDestroy( sdfPhi );
					CHKERRXX( ierr );

					ierr = VecDestroy( rlsPhi );
					CHKERRXX( ierr );

					ierr = VecDestroy( curvature );
					CHKERRXX( ierr );

					for( auto& dim : normal )
					{
						ierr = VecDestroy( dim );
						CHKERRXX( ierr );
					}
				}

				// Write all samples collected for all circles with the same radius but randomized center content to files.
				for( const auto& row : samples )
				{
					std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outFile, "," ) );	// Inner elements.
					outFile << row.back() << std::endl;
				}

				nc++;
				nSamples += samples.size();

				std::cout << "     (" << nc << ") Done with radius = " << R
						  << ".  Maximum relative error = " << maxRE
						  << ".  Samples = " << samples.size() << ";" << std::endl;

				if( nc % 10 == 0 )
					printf( "   [%i radii evaluated after %f secs.]\n", nc, watch.get_duration_current() );
			}

			outFile.close();

			printf( "<< Finished generating %i circles and %i samples in %f secs.\n", nc, nSamples, watch.get_duration_current() );
			watch.stop();
		}

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}