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
 * Updated: May 3, 2021.
 *
 * [Update on May 3, 2020] Adapted code to handle data sets where the gradient of the negative-curvature stencil has an
 * angle in the range [0, pi/2].  That is, we collect samples where the gradient points towards the first quadrant of
 * the local coordinate system centered at the 00 node.  This tries to simplify the architecture of the neural network.
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
	// Modified to allow comparison between different levels of refinement.  Instead of considering h*kappa, we need
	// to base our computations on kappa only.
	// We used to depart from our previous calculations with h = 1 / 2^7.

	const int MAX_REFINEMENT_LEVEL = 6;						// Always take H from unit squares with this maximum
	const double H = 1. / pow( 2, MAX_REFINEMENT_LEVEL );	// level of refinement.
	const double H_BASE = 1. / pow( 2, MIN( MAX_REFINEMENT_LEVEL, 7 ) );	// To avoid data explosion, the number of samples depends on
															// a base value for mesh size (equiv. to max. lvl. at most 7).
	const int NUM_REINIT_ITERS = 10;						// Number of iterations for PDE reintialization.
	const double MIN_KAPPA = 0.5;							// Minimum and maximum curvature values.  Computed by
	const double MAX_KAPPA = 85. + 1 / 3.;					// considering H_BASE: hk_max = h/(1.5h), and hk_min = 0.5h.
	const double MIN_RADIUS = 1. / MAX_KAPPA;				// All resolutions must meet these radius constraints.
	const double MAX_RADIUS = 1. / MIN_KAPPA;
	const double FLAT_LIM_KAPPA = 2.5;						// Flatness limit for triggering hybrid method using a multiple of this (used to be 5, use 2.5 for lvl 6).
	const double FLAT_LIM_RADIUS = 1. / FLAT_LIM_KAPPA;		// All resolutions adhere to this constraint.  Then, the
															// hybrid model is used whenever:
															// 1 * h7 * kLim  = 0.0390625,  (for max lvl of ref = 7),
															// 2 * h8 * kLim  = 0.0390625,
															// 4 * h9 * kLim  = 0.0390625,
															// 8 * h10 * klim = 0.0390625.
	const double DIM = ceil( MAX_RADIUS + 2 * H );			// Symmetric units around origin: [-DIM, +DIM]^{P4EST_DIM}.

	// Number of circles is proportional to radii difference and to H_BASE ratio to H.
	// Originally, 2 circles per finest quad/oct.
	const int NUM_CIRCLES = ceil( 2 * ((MAX_RADIUS - MIN_RADIUS) / H_BASE + 1) * (log2( H_BASE / H ) + 1) );

	// Expected number of samples per distinct radius.
	// First, we allow to generate a tentative number of samples.  Then, we randomly collect only the expected number
	// that we would get if we had used H_BASE instead.  This allows varying the origin of the circles, and then pick
	// a smaller subset that encompasses samples from several configurations.
	// Number of samples per radius is approximated by 5 times 1 sample per h^2, which comes from the area difference of
	// the largest circle and the second to that circle.  This ensures that each radius gets the same number of samples.
	// By doing this, we are reducing the data sets for very small spacing.
	const int SAMPLES_PER_RADIUS = ceil( 5 * M_PI / SQR( H ) * (SQR( FLAT_LIM_RADIUS ) - SQR( FLAT_LIM_RADIUS - H )) );
	const int MAX_SAMPLES_PER_RADIUS = ceil( 5 * M_PI / SQR( H_BASE ) * (SQR( FLAT_LIM_RADIUS ) - SQR( FLAT_LIM_RADIUS - H_BASE )) );

	// Destination folder.
	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-0521/data/" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";
	const int NUM_COLUMNS = num_neighbors_cube + 2;			// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];					// Column headers following the x-y truth table of
	kutils::generateColumnHeaders( COLUMN_NAMES );			// 3-state variables.

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::mt19937 gen{}; 			// NOLINT Standard mersenne_twister_engine with default seed for reproducibility.
	std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// sets to files.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		/////////////////////////////////////////// Preparing data set files ///////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i circles with maximum level of refinement = %i and finest h = %g\n",
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

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		// Variables to control the spread of circles' radii.
		// These must vary depending on the uniform spread of curvature.
		double kappaDistance = MIN_KAPPA - MAX_KAPPA;		// Circles' radii are in [1/MAX_KAPPA, 1/MIN_KAPPA].
		double linspace[NUM_CIRCLES];
		for( int i = 0; i < NUM_CIRCLES; i++ )				// Uniform linear space from 0 to 1, with NUM_CIRCLES steps.
			linspace[i] = (double)( i ) / ( NUM_CIRCLES - 1.0 );

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[] = { 2 * (int)DIM, 2 * (int)DIM, 2 * (int)DIM };		// Symmetric num. of trees in +ve and -ve axes.
		double xyz_min[] = { -DIM, -DIM, -DIM };			// Squared domain.
		double xyz_max[] = { DIM, DIM, DIM };
		int periodic[] = { 0, 0, 0 };						// Non-periodic domain.

		int nSamples = 0;
		int nc = 0;							// Keeps track of number of circles whose samples have been collected.
		while( nc < NUM_CIRCLES )
		{
			const double KAPPA = MAX_KAPPA + linspace[nc] * kappaDistance;
			const double R = 1 / KAPPA;						// Circle radius to be evaluated.
			const double H_KAPPA = H * KAPPA;				// Expected dimensionless curvature: h*kappa = h/r.
			std::vector<std::vector<double>> rlsSamples;
			std::vector<std::vector<double>> sdfSamples;

			// Generate a given number of randomly centered circles with the same radius and accumulate samples until we
			// reach a given maximum for current H.  Then, filter out samples by using the expected max number from
			// using H_BASE.
			double maxRE = 0;								// Maximum relative error.
			int nSamplesForSameRadius = 0;
			while( nSamplesForSameRadius < SAMPLES_PER_RADIUS )
			{
				const double C[] = {
					DIM( uniformDistribution( gen ),		// Center coords are randomly chosen around the origin.
						 uniformDistribution( gen ),
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
				// distance function.  The signed distance function is used for partitioning and refinement and get a
				// band of uniform cells around Gamma.  The latter only works if the level-set function is close to
				// an exact signed distance function.  If no analytic form is given, we must work around it by reinitia-
				// lizing a non-signed distance function, and then using interpolation.  Here, there's no need for the
				// this because we have an analytic form of the signed distance function with a circular inteface.  This
				// is also way faster.
				geom::SphereNSD sphereNsd( DIM( C[0], C[1], C[2] ), R );
				geom::Sphere sphere( DIM( C[0], C[1], C[2] ), R );
				splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sphere, 6.0 );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = (void *)( &levelSetSC );

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
				const double *sdfPhiReadPtr, *rlsPhiReadPtr;
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

							for( int i = 0; i < P4EST_DIM; i++ )	// Translation: this is where we need to interpolate
							{										// the numerical curvature.
								xyz[i] -= grad[i] / gradNorm * rlsPhiReadPtr[n];
								grad[i] *= -1.0;					// After using the gradient, let's negate it to use it below.
							}

							double iHKappa = H * interpolation( DIM( xyz[0], xyz[1], xyz[2] ) );
							rlsDataNve.push_back( -iHKappa );		// Attach interpolated h*kappa to reinit. data only.
							sdfDataNve.push_back( -0 );				// For signed distance function data, add dummy -0's.

							// Rotate stencil so that (negated) gradient at node 00 has an angle in first quadrant.

							kutils::rotateStencilToFirstQuadrant( rlsDataNve, grad );
							kutils::rotateStencilToFirstQuadrant( sdfDataNve, grad );

							// Accumulating samples.
							sdfSamples.push_back( sdfDataNve );
							rlsSamples.push_back( rlsDataNve );

							// Counting samples.
							nSamplesForSameRadius++;				// Negatives only for a given interface node.
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

			// Collect randomly as many samples as we need.
			if( nSamplesForSameRadius > MAX_SAMPLES_PER_RADIUS )
			{
				std::mt19937 gen2{}; 			// NOLINT In order to keep correlated sdf and rls samples, use the same
												// sampling generator and reset its seed.
				std::shuffle( sdfSamples.begin(), sdfSamples.end(), gen2 );
				gen2.seed();					// NOLINT.
				std::shuffle( rlsSamples.begin(), rlsSamples.end(), gen2 );
			}

			// Write all samples collected for all circles with the same radius but randomized center content to files.
			nSamplesForSameRadius = 0;
			for( const auto& row : sdfSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( sdfFile, "," ) );	// Inner elements.
				sdfFile << row.back() << std::endl;

				if( ++nSamplesForSameRadius >= MAX_SAMPLES_PER_RADIUS )
					break;
			}

			nSamplesForSameRadius = 0;
			for( const auto& row : rlsSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( rlsFile, "," ) );	// Inner elements.
				rlsFile << row.back() << std::endl;

				if( ++nSamplesForSameRadius >= MAX_SAMPLES_PER_RADIUS )
					break;
			}

			nc++;
			nSamples += nSamplesForSameRadius;

			std::cout << "     (" << nc << ") Done with radius = " << R
					  << ".  Maximum relative error = " << maxRE
					  << ".  Samples = " << nSamplesForSameRadius << ";" << std::endl;

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
}