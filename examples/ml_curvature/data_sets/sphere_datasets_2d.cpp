/**
 * Generate datasets for training a neural network on circular interfaces using samples from a signed distance function
 * and from a reinitialized level-set function.
 *
 * The samples collected for signed and reinitialized level-set functions have one-to-one correlation.  That is, the nth
 * sample (i.e. nth row in both files) correspond to the same standing point adjacent to the interface and its 9-stencil
 * neighborhood.  This correspondance can be used, eventually, to train denoising-like autoencoders if you wish.
 *
 * The files generated are named sphere_[X]_Y.csv, where [X] is one of "sdf" or "rls", for signed distance function and
 * reinitialized level-set, respectively, and Y is the highest level of tree resolution.
 *
 * Developer: Luis Ángel.
 * Date: May 12, 2020.
 * Updated: October 24, 2021.
 *
 * [Update on May 3, 2021] Adapted code to handle data sets where the gradient of the negative-curvature stencil has an
 * angle in the range [0, pi/2].  That is, we collect samples where the gradient points towards the first quadrant of
 * the local coordinate system centered at the 00 node.  This tries to simplify the architecture of the neural network.
 * [Update on October 23, 2021] Data sets include normal unit vector components as additional training cues.
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
#include <src/parameter_list.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include "local_utils.h"


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<unsigned short> maxRL( pl, 10, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 10)" );
	param_t<double> minHK( pl, 0.005, "minHK", "Minimum dimensionless curvature (default: 0.005)" );
	param_t<double> maxHK( pl, 2./3, "maxHK", "Maximum dimensionless curvature (default: 2./3)" );
	param_t<int> circlesPerH( pl, 2, "circlesPerH", "How many circle radii to roughly fit in a cell (default: 2)" );
	param_t<int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<int> skipEveryXSamples( pl, 4, "skipEveryXSamples", "Skip every x samples next to Gamma randomly (default: 4)" );
	param_t<std::string> outputDir( pl, "/Volumes/YoungMinEXT/k_ecnet_data", "outputDir", "Path where file will be written (default: same folder as the executable)" );

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
		if( cmd.parse( argc, argv, "Generating sphere data sets for curvature inference" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double ONE_OVER_H = 1 << maxRL();
		const double H = 1. / ONE_OVER_H;					// Maximum level of refinement.
		const double MAX_KAPPA = maxHK() * ONE_OVER_H;		// Steepest curvature (by default, (2/3)/H).
		const double MIN_KAPPA = minHK() * ONE_OVER_H;		// Flattest curvature.
		const double MIN_RADIUS = 1. / MAX_KAPPA;
		const double MAX_RADIUS = 1. / MIN_KAPPA;
		const double D_DIM = ceil( MAX_RADIUS + 4 * H );	// Symmetric units around origin: [-DIM, +DIM]^{P4EST_DIM}.
		const int NUM_CIRCLES = ceil( circlesPerH() * ((MAX_RADIUS - MIN_RADIUS) * ONE_OVER_H + 1) );	// Number of circles is proportional to radii difference and H.

		// Expected number of samples per distinct radius.
		// First, we allow to generate a tentative number of samples.  Then, we randomly subsample.  This allows varying
		// the origin of the circles, and then pick a smaller subset with samples from several configurations.
		// Number of samples per radius is approximated by 5 times 1 sample per H^2, which comes from the area
		// difference of the average circle and the second to that circle.  This ensures that each radius gets the same
		// number of samples.
		// If user wants it, skip every x samples randomly to reduce data set size.
		const double AVG_RADIUS = (MAX_RADIUS - MIN_RADIUS) / 2.;
		const int SAMPLES_PER_RADIUS = (int)ceil( 5 * M_PI / SQR( H ) * (SQR( AVG_RADIUS ) - SQR( AVG_RADIUS - H )) ) / skipEveryXSamples();

		// Destination folder.
		const std::string DATA_PATH = outputDir() + "/" + std::to_string( maxRL() ) + "/";
		const int NUM_COLUMNS = (P4EST_DIM + 1) * num_neighbors_cube + 2;		// Number of columns in dataset.
		std::string COLUMN_NAMES[NUM_COLUMNS];				// Column headers following the x-y truth table of 3-state
		kutils::generateColumnHeaders( COLUMN_NAMES );		// variables: includes phi values and normal components.

		// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
		std::mt19937 gen{}; 		// NOLINT Standard mersenne_twister_engine with default seed for reproducibility.
		std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

		std::mt19937 genSkip{};		// NOLINT.
		std::uniform_real_distribution<double> skipDist;

		/////////////////////////////////////////// Preparing data set files ///////////////////////////////////////////

		parStopWatch watch;
		PetscPrintf( mpi.comm(), ">> Began to generate datasets for %i circles with maximum level of refinement = %i "
								 "and finest h = %g\n", NUM_CIRCLES, maxRL(), H );
		watch.start();

		// Prepare samples files: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream sdfFile;
		std::string sdfFileName = DATA_PATH + "sphere_sdf_" + std::to_string( maxRL() ) +  ".csv";
		sdfFile.open( sdfFileName, std::ofstream::trunc );
		if( !sdfFile.is_open() )
			throw std::runtime_error( "Output file " + sdfFileName + " couldn't be opened!" );

		std::ofstream rlsFile;
		std::string rlsFileName = DATA_PATH + "sphere_rls_" + std::to_string( maxRL() ) +  ".csv";
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
		int n_xyz[] = { 2 * (int)D_DIM, 2 * (int)D_DIM, 2 * (int)D_DIM };	// Symmetric num. of trees in +ve and -ve axes.
		double xyz_min[] = { -D_DIM, -D_DIM, -D_DIM };		// Squared domain.
		double xyz_max[] = { D_DIM, D_DIM, D_DIM };
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
			// reach a given maximum for current H.  Then, perform random subsampling.
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
				splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, maxRL(), &sphere, 4.0 );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = (void *)( &levelSetSC );

				// Refine and partition forest.
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

				// Validation.
				double dxyz[P4EST_DIM]; 			// Dimensions of the smallest quadrants.
				double dxyz_min;        			// Minimum side length of the smallest quadrants.
				double diag_min;        			// Diagonal length of the smallest quadrants.
				get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );
				assert( ABS( dxyz_min - H ) <= EPS );

				// Ghosted parallel PETSc vectors to store level-set function values.
				Vec sdfPhi, rlsPhi;
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sdfPhi ) );
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &rlsPhi ) );

				Vec curvature, rlsNormal[P4EST_DIM], sdfNormal[P4EST_DIM];
				CHKERRXX( VecDuplicate( rlsPhi, &curvature ) );
				for( int dim = 0; dim <P4EST_DIM; dim++ )
				{
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sdfNormal[dim] ) );
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &rlsNormal[dim] ) );
				}

				// Calculate the level-set function values for all independent nodes.
				sample_cf_on_nodes( p4est, nodes, sphere, sdfPhi );
				sample_cf_on_nodes( p4est, nodes, sphereNsd, rlsPhi );

				// Reinitialize level-set function.
				my_p4est_level_set_t ls( &nodeNeighbors );
				ls.reinitialize_2nd_order( rlsPhi, reinitNumIters() );

				// Compute numerical curvature with reinitialized data, which will be interpolated at the interface.
				// Also need normal vectors with both level-set functions.
				compute_normals( nodeNeighbors, rlsPhi, rlsNormal );
				compute_normals( nodeNeighbors, sdfPhi, sdfNormal );
				compute_mean_curvature( nodeNeighbors, rlsPhi, rlsNormal, curvature );

				// Prepare interpolation.
				my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
				interpolation.set_input( curvature, interpolation_method::linear );

				// Once the level-set function is reinitialized, collect nodes on or next to the interface; these are
				// the points we'll use to create our sample files and compare with the signed distance function.
				NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, (char)maxRL() );
				std::vector<p4est_locidx_t> indices;
				nodesAlongInterface.getIndices( &rlsPhi, indices );

				// Getting the full uniform stencils of interface points.
				const double *sdfPhiReadPtr, *rlsPhiReadPtr;
				CHKERRXX( VecGetArrayRead( sdfPhi, &sdfPhiReadPtr ) );
				CHKERRXX( VecGetArrayRead( rlsPhi, &rlsPhiReadPtr ) );

				const double *rlsNormalReadPtr[P4EST_DIM], *sdfNormalReadPtr[P4EST_DIM];
				for( int dim = 0; dim < P4EST_DIM; dim++ )
				{
					CHKERRXX( VecGetArrayRead( rlsNormal[dim], &rlsNormalReadPtr[dim] ) );
					CHKERRXX( VecGetArrayRead( sdfNormal[dim], &sdfNormalReadPtr[dim] ) );
				}

				for( auto n : indices )
				{
					if( skipEveryXSamples() > 1 && skipDist( genSkip ) <= 1. / skipEveryXSamples() )	// Premature subsampling.
						continue;

					std::vector<p4est_locidx_t> stencil;	// Contains 9*3 values in 2D, 27*4 values in 3D.
					std::vector<double> sdfDataNve;			// Phi and h*kappa results in negative form only.
					std::vector<double> rlsDataNve;
					sdfDataNve.reserve( NUM_COLUMNS );		// Efficientize containers.
					rlsDataNve.reserve( NUM_COLUMNS );
					try
					{
						if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
						{
							// First the phi-values.
							for( auto s : stencil )
							{
								// First the signed distance function.
								sdfDataNve.push_back( -sdfPhiReadPtr[s] * ONE_OVER_H );		// H-normalized negative-curvature phi values.

								// Then the reinitialized data.
								rlsDataNve.push_back( -rlsPhiReadPtr[s] * ONE_OVER_H );

								// Error.
								double error = ABS( sdfPhiReadPtr[s] - rlsPhiReadPtr[s] ) * ONE_OVER_H;
								maxRE = MAX( maxRE, error );
							}

							// Now the normal unit vectors' components.
							for( int dim = 0; dim < P4EST_DIM; dim++ )
							{
								for( auto s: stencil )
								{
									sdfDataNve.push_back( -sdfNormalReadPtr[dim][s] );	// Flip gradients.
									rlsDataNve.push_back( -rlsNormalReadPtr[dim][s] );
								}
							}

							// Appending the target hk.
							sdfDataNve.push_back( -H_KAPPA );
							rlsDataNve.push_back( -H_KAPPA );

							// Appending the interpolated hk.
							double xyz[P4EST_DIM];							// Position of node at the center of the stencil.
							node_xyz_fr_n( n, p4est, nodes, xyz );
							double nveGrad[P4EST_DIM];
							for( int dim = 0; dim < P4EST_DIM; dim++ )		// Get negative gradient and make sure no component is exactly zero.
								nveGrad[dim] = (rlsNormalReadPtr[dim][n] == 0)? -EPS : -rlsNormalReadPtr[dim][n];

							for( int i = 0; i < P4EST_DIM; i++ )			// Translation: this is where we need to
								xyz[i] += nveGrad[i] * rlsPhiReadPtr[n];	// interpolate the numerical curvature.

							double iHKappa = H * interpolation( DIM( xyz[0], xyz[1], xyz[2] ) );
							rlsDataNve.push_back( -iHKappa );		// Attach interpolated hk to reinitialized data only.
							sdfDataNve.push_back( -H_KAPPA );		// For signed distance function data, the exact hk.

							// Reorienting stencil so that (negated) gradient at node 00 has an angle in first quadrant.
							kutils::rotateStencilToFirstQuadrant( rlsDataNve, nveGrad );
							kutils::rotateStencilToFirstQuadrant( sdfDataNve, nveGrad );

							// Accumulating samples.
							sdfSamples.push_back( sdfDataNve );
							rlsSamples.push_back( rlsDataNve );

							// Data augmentation.
							kutils::reflectStencil_yEqx( rlsDataNve );
							kutils::reflectStencil_yEqx( sdfDataNve );

							// Accumulating reflected samples.
							sdfSamples.push_back( sdfDataNve );
							rlsSamples.push_back( rlsDataNve );

							// Counting samples.
							nSamplesForSameRadius++;
						}
					}
					catch( std::exception &e )
					{
						std::cerr << e.what() << std::endl;
					}
				}

				// Cleaning up.
				CHKERRXX( VecRestoreArrayRead( sdfPhi, &sdfPhiReadPtr ) );
				CHKERRXX( VecRestoreArrayRead( rlsPhi, &rlsPhiReadPtr ) );
				for( int dim = 0; dim < P4EST_DIM; dim++ )
				{
					CHKERRXX( VecRestoreArrayRead( rlsNormal[dim], &rlsNormalReadPtr[dim] ) );
					CHKERRXX( VecRestoreArrayRead( sdfNormal[dim], &sdfNormalReadPtr[dim] ) );
				}

				// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
				CHKERRXX( VecDestroy( sdfPhi ) );
				CHKERRXX( VecDestroy( rlsPhi ) );
				CHKERRXX( VecDestroy( curvature ) );

				for( int dim = 0; dim < P4EST_DIM; dim++ )
				{
					CHKERRXX( VecDestroy( rlsNormal[dim] ) );
					CHKERRXX( VecDestroy( sdfNormal[dim] ) );
				}

				// Destroy the p4est and its connectivity structure.
				p4est_nodes_destroy( nodes );
				p4est_ghost_destroy( ghost );
				p4est_destroy( p4est );
				my_p4est_brick_destroy( connectivity, &brick );
			}

			// Collect randomly as many samples as we need.
			if( nSamplesForSameRadius > SAMPLES_PER_RADIUS )
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

				if( ++nSamplesForSameRadius >= SAMPLES_PER_RADIUS )
					break;
			}

			nSamplesForSameRadius = 0;
			for( const auto& row : rlsSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( rlsFile, "," ) );	// Inner elements.
				rlsFile << row.back() << std::endl;

				if( ++nSamplesForSameRadius >= SAMPLES_PER_RADIUS )
					break;
			}

			nc++;
			nSamples += nSamplesForSameRadius;

			PetscPrintf( mpi.comm(), "     (%d) Done with radius = %f.  Maximum relative error = %f.  Samples = %d;\n",
						 nc, R, maxRE, nSamplesForSameRadius );

			if( nc % 10 == 0 )
				PetscPrintf( mpi.comm(), "   [%i radii evaluated after %f secs.]\n", nc, watch.get_duration_current() );
		}

		sdfFile.close();
		rlsFile.close();

		PetscPrintf( mpi.comm(), "<< Finished generating %i circles and %i samples in %f secs.\n",
					 nc, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}