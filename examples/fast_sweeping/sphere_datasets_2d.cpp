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
#include <unordered_map>
#include "local_utils.h"


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const int MAX_REFINEMENT_LEVEL = 5;						// Always take H from unit squares with this maximum
	const double H = 1. / pow( 2, MAX_REFINEMENT_LEVEL );	// level of refinement.
	const int NUM_REINIT_ITERS = 10;						// Number of iterations for PDE reintialization.
	const double FLAT_LIM_HK = 0.04;						// Flatness limit for dimensionless curvature.
	const double MIN_RADIUS = 1.5 * H;						// Ensures at least 4 nodes inside smallest circle.
	const double MAX_RADIUS = H / FLAT_LIM_HK;				// Ensures we can cover h*kappa up to 0.04.
	const double DIM = ceil( MAX_RADIUS + 2 * H );			// Symmetric units around origin: domain is [-DIM, +DIM]^{P4EST_DIM}
	const int NUM_CIRCLES = (int)(3 * ((MAX_RADIUS - MIN_RADIUS) / H + 1));		// Number of circles is proportional to radii difference.
																				// Originally, 2 circles per finest quad/oct.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-1120/data-" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";		// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;				// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::mt19937 gen{}; 			// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
	std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

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

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i spheres with maximum refinement level of %i and finest h = %g\n",
			NUM_CIRCLES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples files: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream sdfFile;
		std::string sdfFileName = DATA_PATH + "sphere_sdf_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		sdfFile.open( sdfFileName, std::ofstream::trunc );
		if( !sdfFile.is_open() )
			throw std::runtime_error( "Output file " + sdfFileName + " couldn't be opened!" );

		std::ofstream rlsFile;
		std::string rlsFileName = DATA_PATH + "sphere_rls_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		rlsFile.open( rlsFileName, std::ofstream::trunc );
		if( !rlsFile.is_open() )
			throw std::runtime_error( "Output file " + rlsFileName + " couldn't be opened!" );

		// Write column headers: enforcing strings by adding quotes around them.
		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		sdfFile << headerStream.str() << std::endl;
		rlsFile << headerStream.str() << std::endl;

		sdfFile.precision( 15 );							// Precision for floating point numbers.
		rlsFile.precision( 15 );

		// Variables to control the spread of circles' radii, which must vary uniformly from H/MAX_RADIUS to H/MIN_RADIUS.
		double kappaDistance = 1 / MAX_RADIUS - 1 / MIN_RADIUS;		// Circles' radii are in [1.5*H, H/0.04], inclusive.
		double linspace[NUM_CIRCLES];
		for( int i = 0; i < NUM_CIRCLES; i++ )				// Uniform linear space from 0 to 1, with NUM_CIRCLES steps.
			linspace[i] = (double)( i ) / ( NUM_CIRCLES - 1.0 );

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[] = { 2 * (int)DIM, 2 * (int)DIM, 2 * (int)DIM };	// One tree per dimension.
		double xyz_min[] = { -DIM, -DIM, -DIM };			// Squared domain.
		double xyz_max[] = { DIM, DIM, DIM };
		int periodic[] = { 0, 0, 0 };						// Non-periodic domain.

		int nSamples = 0;
		int nc = 0;							// Keeps track of number of circles whose samples have been collected.
											// Number of samples per radius is approximated by 5 times 2 samples per
											// h^2, which comes from the area difference of largest circle and second
											// to largest circle.
		const int MAX_SAMPLES_PER_RADIUS = (int)( 10 * M_PI / SQR( H ) * (SQR( MAX_RADIUS ) - SQR( MAX_RADIUS - H )) );
		while( nc < NUM_CIRCLES )
		{
			const double KAPPA = 1 / MIN_RADIUS + linspace[nc] * kappaDistance;
			const double R = 1 / KAPPA;			// Circle radius to be evaluated.
			const double H_KAPPA = H * KAPPA;	// Expected dimensionless curvature: h*kappa = h/r.
			std::vector<std::vector<double>> rlsSamples;
			std::vector<std::vector<double>> sdfSamples;

			// Generate a given number of randomly centered circles with the same radius and accumulate samples until we
			// reach a given maximum.
			double maxRE = 0;							// Maximum relative error.
			int nSamplesForSameRadius = 0;
			while( nSamplesForSameRadius < MAX_SAMPLES_PER_RADIUS )
			{
				const double C[] = {
					DIM( uniformDistribution( gen ),	// Center coords are randomly chosen
						 uniformDistribution( gen ),	// around the origin of the grid.
						 uniformDistribution( gen ) )
				 };

				// p4est variables and data structures: these change with every single circle because we must refine the
				// trees according to the new circle's center and radius.
				p4est_t *p4est;
				p4est_nodes_t *nodes;
				my_p4est_brick_t brick;
				p4est_ghost_t *ghost;
				p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

				// Definining the non-signed distance level-set function to be reinitialized and the exact signed
				// distance function.  The non-signed distance function is used for partitioning and refinement.
				geom::SphereNSD sphereNsd( DIM( C[0], C[1], C[2] ), R );
				geom::Sphere sphere( DIM( C[0], C[1], C[2] ), R );
				splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sphereNsd );

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

				// Validation.
				double dxyz[P4EST_DIM]; 			// Dimensions of the smallest quadrants.
				double dxyz_min;        			// Minimum side length of the smallest quadrants.
				double diag_min;        			// Diagonal length of the smallest quadrants.
				get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );
				assert( ABS( dxyz_min - H ) <= EPS );

				// Ghosted parallel PETSc vectors to store level-set function values.
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
				sample_cf_on_nodes( p4est, nodes, sphereNsd, rlsPhi );

				// Reinitialize level-set function using PDE equation.
				my_p4est_level_set_t ls( &nodeNeighbors );
				ls.reinitialize_2nd_order( rlsPhi, NUM_REINIT_ITERS );

				// Compute curvature with reinitialized data, which will be interpolated at the interface.
				compute_normals( nodeNeighbors, rlsPhi, normal );
				compute_mean_curvature( nodeNeighbors, rlsPhi, normal, curvature );

				// Prepare interpolation.
				my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
				interpolation.set_input( curvature, linear );

				// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these
				// are the points we'll use to create our sample files and compare with the signed distance function.
				NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
				std::vector<p4est_locidx_t> indices;
				nodesAlongInterface.getIndices( &rlsPhi, indices );

				// Getting the full uniform stencils of interface points.
				const double * sdfPhiReadPtr, *rlsPhiReadPtr;
				ierr = VecGetArrayRead( sdfPhi, &sdfPhiReadPtr );
				CHKERRXX( ierr );
				ierr = VecGetArrayRead( rlsPhi, &rlsPhiReadPtr );
				CHKERRXX( ierr );

				for( auto n : indices )
				{
					std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D, 27 values in 3D.
					std::vector<double> sdfDataNve;			// Phi and h*kappa results in negative form only.
					std::vector<double> rlsDataNve;
					sdfDataNve.reserve( NUM_COLUMNS );		// Efficientize containers.
					rlsDataNve.reserve( NUM_COLUMNS );
					try
					{
						if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
						{
							for( auto s : stencil )
							{
								// First the signed distance function.
								sdfDataNve.push_back( -sdfPhiReadPtr[s] );		// Negative curvature phi values.

								// Then the reinitialized data.
								rlsDataNve.push_back( -rlsPhiReadPtr[s] );

								// Error.
								double error = ABS( sdfPhiReadPtr[s] - rlsPhiReadPtr[s] ) / H;
								maxRE = MAX( maxRE, error );
							}

							// Appending the target h*kappa.
							sdfDataNve.push_back( -H_KAPPA );
							rlsDataNve.push_back( -H_KAPPA );

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
							rlsDataNve.push_back( -iHKappa );		// Attach interpolated h*kappa to reinit. data only.
							sdfDataNve.push_back( -0 );				// For signed distance function data, add dummy -0's.

							// Accumulating samples.
							sdfSamples.push_back( sdfDataNve );
							rlsSamples.push_back( rlsDataNve );

							// Counting samples.
							nSamplesForSameRadius++;				// Negatives only for a given interface node.
							if( nSamplesForSameRadius >= MAX_SAMPLES_PER_RADIUS )
								break;
						}
					}
					catch( std::exception &e )
					{
						std::cerr << e.what() << std::endl;
					}
				}

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

				// Destroy the p4est and its connectivity structure.
				p4est_nodes_destroy( nodes );
				p4est_ghost_destroy( ghost );
				p4est_destroy( p4est );
				p4est_connectivity_destroy( connectivity );
			}

			// Write all samples collected for all circles with the same radius but randomized center content to files.
			for( const auto& row : sdfSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( sdfFile, "," ) );	// Inner elements.
				sdfFile << row.back() << std::endl;
			}

			for( const auto& row : rlsSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( rlsFile, "," ) );	// Inner elements.
				rlsFile << row.back() << std::endl;
			}

			nc++;
			nSamples += sdfSamples.size();

			std::cout << "     (" << nc << ") Done with radius = " << R
					  << ".  Maximum relative error = " << maxRE
					  << ".  Samples = " << sdfSamples.size() << ";" << std::endl;

			if( nc % 10 == 0 )
				printf( "   [%i radii evaluated after %f secs.]\n", nc, watch.get_duration_current() );
		}

		sdfFile.close();
		rlsFile.close();

		printf( "<< Finished generating %i circles and %i samples in %f secs.\n", nc, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}