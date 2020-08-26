/**
 * Generate datasets for training a feedforward neural network on merging spherical interfaces using samples from a
 * signed distance function and from a reinitialized level-set function using the PDE equation with 10 iterations.
 *
 * The samples collected for signed and reinitialized level-set functions have one-to-one correlation.  That is, the nth
 * sample (i.e. nth row in both files) correspond to the same standing point adjacent to the interface and its 9-stencil
 * neighborhood.  This correspondance can be used, eventually, to train denoising-like autoencoders.
 *
 * The files generated are named merging_spheres_[X]_[Y].csv, where [X] is one of "sdf" or "rls", for signed distance
 * function and reinitialized level-set, respectively, and [Y] is the highest level of tree resolution.
 *
 * Developer: Luis Ángel.
 * Date: August 14, 2020.
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
#include "two_spheres_levelset_2d.h"


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -1, MAX_D = 1;						// The canonical space is [-1,1]^{P4EST_DIM} so that there's
	const double HALF_D = 0.5;								// space for each circle to the left and right of the origin.
	const int MAX_REFINEMENT_LEVEL = 7;						// Maximum level of refinement for each tree.
	const double H = 1 / pow( 2, MAX_REFINEMENT_LEVEL );	// Highest spatial resolution in x/y directions.
	const int STENCIL_SIZE = (int)pow( 3, P4EST_DIM );		// Stencil size.

	const double MIN_RADIUS = 1.5 * H;						// Ensures at least 4 nodes inside smallest circle.
	const double MAX_RADIUS = HALF_D - 2 * H;				// Prevents falling off the domain.
	const int NUM_CIRCLES = (int)ceil( (MAX_RADIUS - MIN_RADIUS) / H + 1 );	// Number of circles is proportional to finest resolution.

	const int NUM_THETAS = (int)pow( 2, MAX_REFINEMENT_LEVEL - 5 ) + 1;
	const double DELTA_THETA = M_PI_2 / NUM_THETAS;			// Angular step.
	const double MIN_THETA = DELTA_THETA - M_PI_2;			// We vary the rotation of the circles' axis with respect to
	const double MAX_THETA = 0;								// the world +x-axis so that we cover the whole space.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde/data-merging/";		// Destination folder.
	const int NUM_COLUMNS = STENCIL_SIZE + 2;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::mt19937 gen( 5489u ); 					// Standard mersenne_twister_engine with (default) constant seed.
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );

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

		// Collect samples from signed distance function and reinitialized level-set using PDE equation with 10 iterations.
		std::cout << "Collecting samples from merging-circles level-set function..." << std::endl;

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i circles with maximum refinement level of %i, finest h = %f, and "
		  "%d tilts from -pi/8 to +pi/8.\n", NUM_CIRCLES, MAX_REFINEMENT_LEVEL, H, NUM_THETAS );
		watch.start();

		// Prepare samples files: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream sdfFile;
		std::string sdfFileName = DATA_PATH + "merging_sphere_sdf_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		sdfFile.open( sdfFileName, std::ofstream::trunc );
		if( !sdfFile.is_open() )
			throw std::runtime_error( "Output file " + sdfFileName + " couldn't be opened!" );

		std::ofstream rlsFile;
		std::string rlsFileName = DATA_PATH + "merging_sphere_rls_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
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
		double kappaDistance = 1 / MAX_RADIUS - 1 / MIN_RADIUS;		// Circles' radii are in [1.5*H, 0.5-2H], inclusive.
		double linspace[NUM_CIRCLES];
		for( int i = 0; i < NUM_CIRCLES; i++ )				// Uniform linear space from 0 to 1, with NUM_CIRCLES steps.
			linspace[i] = (double)( i ) / (NUM_CIRCLES - 1.0);

		// Variables to control the spread of ratation angles, which must vary uniformly from MIN_THETA to MAX_THETA, in
		// a finite number of steps.
		const double THETA_DIST = MAX_THETA - MIN_THETA;	// As defined above, in [-pi/8, +pi/8].
		double linspaceTheta[NUM_THETAS];
		for( int i = 0; i < NUM_THETAS; i++ )				// Uniform linear space from 0 to 1, with NUM_TETHAS steps.
			linspaceTheta[i] = (double)( i ) / (NUM_THETAS - 1.0);

		const double HALF_AXIS_LEN = 0.5;

		// Domain information, applicable to all merging circular interfaces.
		int n_xyz[] = {2, 2, 2};							// One tree per unit square of the domain.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		// Printing header for log.
		std::cout << "Theta Val, Circle 1, Max Rel Error, Num Samples/Evaluated, Time" << std::endl;

		// Varying the tilt of the merging circle's main axis.
		int nSamples = 0;
		for( int nt = 0; nt < NUM_THETAS; nt++ )
		{
			const double THETA = MIN_THETA + linspaceTheta[nt] * THETA_DIST;

			// Varying the reference (i.e. left) circle radius.
			for( int nc1 = 0; nc1 < NUM_CIRCLES; nc1++ )
			{
				const double KAPPA1 = 1 / MIN_RADIUS + linspace[nc1] * kappaDistance;
				const double R1 = 1 / KAPPA1;				// Circle radius to be evaluated.
				double maxRE = 0;							// Maximum absolute error, relative to H.
				int totalInclusiveSamples = 0;				// A debugging var to check how many total samples are evaluated.

				std::vector<std::vector<double>> rlsSamples;
				std::vector<std::vector<double>> sdfSamples;

				const double T[] = {
					uniformDistributionH_2( gen ),		// Translate center coords by a randomly chosen
					uniformDistributionH_2( gen )		// perturbation from the grid's midpoint.
				};

				const double X1 = -(HALF_AXIS_LEN * cos( -THETA) + T[0]);	// Center coordinates for reference
				const double Y1 = HALF_AXIS_LEN * sin( -THETA ) + T[1];		// circle in world coordinate system.

				// Varying radii of second circle (i.e. to the right), which can get as big as the reference circle.
				for( int nc2 = 0; nc2 <= nc1; nc2++ )
				{
					const double KAPPA2 = 1 / MIN_RADIUS + linspace[nc2] * kappaDistance;
					const double R2 = 1 / KAPPA2;

					// Now that we have both circle's, it remains to vary their distance from |R1 - R2| + H
					// to R1 + R2 + 3H, in steps of 3H/2.
					const double MIN_DIST = ABS( R1 - R2 ) + H;
					const double MAX_DIST = R1 + R2 + 3 * H;
					const int NUM_DIST_STEPS = (int)floor( (MAX_DIST - MIN_DIST) / (1.5 * H) ) + 1;

					for( int nd = 0; nd < NUM_DIST_STEPS; nd++ )
					{
						const double D = MIN_DIST + (double)(nd) / (NUM_DIST_STEPS - 1) * (MAX_DIST - MIN_DIST);

						// p4est variables and data structures: these change with every new merging-circles configuration.
						p4est_t *p4est;
						p4est_nodes_t *nodes;
						my_p4est_brick_t brick;
						p4est_ghost_t *ghost;
						p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

						// Defining the level-set function to be reinitialized.
						TwoSpheres twoSpheres( X1, Y1, R1, R2, D, THETA );
						splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &twoSpheres );

						// Create the forest using a level-set as refinement criterion.
						p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
						p4est->user_pointer = (void *) ( &levelSetSC );

						// Refine and recursively partition forest.
						my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
						my_p4est_partition( p4est, P4EST_TRUE, nullptr );

						// Create the ghost (cell) and node structures.
						ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
						nodes = my_p4est_nodes_new( p4est, ghost );

						// Initialize the neighbor nodes structure.
						my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
						my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
						nodeNeighbors.init_neighbors(); // This is not mandatory, but it can only help performance
														// given how much we'll neeed the node neighbors.

						// A ghosted parallel PETSc vector to store level-set function values.
						Vec phi;
						ierr = VecCreateGhostNodes( p4est, nodes, &phi );
						CHKERRXX( ierr );

						// Calculate the level-set function values for each independent node
						// (i.e. locally owned and ghost nodes).
						sample_cf_on_nodes( p4est, nodes, twoSpheres, phi );

						// Vectors for curvature and normals.
						Vec curvature, normal[P4EST_DIM];
						ierr = VecDuplicate( phi, &curvature );
						CHKERRXX( ierr );
						for( auto& dim : normal )
						{
							VecCreateGhostNodes( p4est, nodes, &dim );
							CHKERRXX( ierr );
						}

						// A vector to store the signed distance function for all nodes.
						Vec distPhi;
						ierr = VecDuplicate( phi, &distPhi );
						CHKERRXX( ierr );

						double *distPhiPtr;
						ierr = VecGetArray( distPhi, &distPhiPtr );
						CHKERRXX( ierr );

						// A vector to store which circle produces the closest distance to each node.
						// This is later used to find troubling nodes.
						Vec who;
						ierr = VecDuplicate( phi, &who );
						CHKERRXX( ierr );

						double *whoPtr;
						ierr = VecGetArray( who, &whoPtr );
						CHKERRXX( ierr );

						// A vector to store the target dimensionless curvature at nodes along the interface.
						Vec hKappa;
						ierr = VecDuplicate( phi, &hKappa );
						CHKERRXX( ierr );

						double *hKappaPtr;
						ierr = VecGetArray( hKappa, &hKappaPtr );
						CHKERRXX( ierr );

						// Calculating the signed-distance and dimensionless curvature.
						for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
						{
							double xyz[P4EST_DIM];
							node_xyz_fr_n( i, p4est, nodes, xyz );
							distPhiPtr[i] = twoSpheres.getSignedDistance( xyz[0], xyz[1], H, hKappaPtr[i],
																		  reinterpret_cast<short &>(whoPtr[i]));
						}

						// Reinitialize level-set function.
						my_p4est_level_set_t ls( &nodeNeighbors );
						ls.reinitialize_2nd_order( phi, 10 );

						// Compute numerical curvature with reinitialized data, which will be interpolated at the
						// interface for comparison purposes.
						compute_normals( nodeNeighbors, phi, normal );
						compute_mean_curvature( nodeNeighbors, phi, normal, curvature );

						// Prepare interpolation.
						my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
						interpolation.set_input( curvature, linear );

						// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are
						// the points we'll use to create our sample files.
						NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

						// Getting the full uniform stencils of interface points.
						std::vector<p4est_locidx_t> indices;
						nodesAlongInterface.getIndices( &phi, indices );

						const double *phiReadPtr;
						ierr = VecGetArrayRead( phi, &phiReadPtr );
						CHKERRXX( ierr );

						// Now, collect samples with reinitialized level-set function values and target h*kappa.
						for( auto n : indices )
						{
							double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
							node_xyz_fr_n( n, p4est, nodes, xyz );

							std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
							try
							{
								if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
								{
									std::vector<double> sdfData;	// Signed-distance function values.
									std::vector<double> rlsData;	// Reinitialized level-set function values.
									totalInclusiveSamples++;

									// A troubling node is one that lies in a discontinuity region or owns at least one
									// of its 9-point stencil neighbors lying on a discontinuity or with a phi value
									// that is due to a circle different to n's.
									bool troublingFlag = whoPtr[n] == 0;
									double sampleError = 0;
									for( auto s : stencil )
									{
										sdfData.push_back( -distPhiPtr[s] );
										rlsData.push_back( -phiReadPtr[s] );

										// Error for standing stencil.
										double error = ABS( sdfData.back() - rlsData.back() ) / H;
										sampleError = MAX( sampleError, error );

										// Checking for trouble.
										if( whoPtr[s] != whoPtr[n] )
											troublingFlag = true;
									}

									if( !troublingFlag )		// Skip samples that are not troubling.
										continue;

									maxRE = MAX( sampleError, maxRE );

									// Appending target dimensionless curvature.
									sdfData.push_back( -hKappaPtr[n] );
									rlsData.push_back( -hKappaPtr[n] );

									// Computing the gradient.
									double grad[P4EST_DIM];					// Getting its gradient (i.e. normal).
									const quad_neighbor_nodes_of_node_t *qnnnPtr;
									nodeNeighbors.get_neighbors( n, qnnnPtr );
									qnnnPtr->gradient( phiReadPtr, grad );
									double gradNorm = sqrt( SQR( grad[0] ) + SQR( grad[1] ) );

									// Normalize gradient and translate center node to interface for interpolation.
									for( int i = 0; i < P4EST_DIM; i++ )				// Translation: this is the location where
										xyz[i] -= grad[i] / gradNorm * phiReadPtr[n];	// we need to interpolate numerical curvature.

									double iHKappa = H * interpolation( xyz[0], xyz[1] );
									sdfData.push_back( 0 );
									rlsData.push_back( -iHKappa );			// Append interpolated h*kappa.

									// Add troubling samples to global samples vectors using data augmentation.
									for( int i = 0; i < 4; i++ )	// Data augmentation by rotating samples 90 degrees
									{								// three times.
										rlsSamples.push_back( rlsData );
										sdfSamples.push_back( sdfData );

										rotatePhiValues90( rlsData, NUM_COLUMNS );
										rotatePhiValues90( sdfData, NUM_COLUMNS );
									}
								}
							}
							catch( std::exception &e )
							{
								std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
							}
						}

						// Cleaning up.
						ierr = VecRestoreArray( who, &whoPtr );
						CHKERRXX( ierr );

						ierr = VecRestoreArray( hKappa, &hKappaPtr );
						CHKERRXX( ierr );

						ierr = VecRestoreArray( distPhi, &distPhiPtr );
						CHKERRXX( ierr );

						ierr = VecRestoreArrayRead( phi, &phiReadPtr );
						CHKERRXX( ierr );

						// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
						ierr = VecDestroy( who );
						CHKERRXX( ierr );

						ierr = VecDestroy( hKappa );
						CHKERRXX( ierr );

						ierr = VecDestroy( distPhi );
						CHKERRXX( ierr );

						ierr = VecDestroy( phi );
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

				nSamples += rlsSamples.size();

				// Log output.
				std::cout << nt + 1 << ", " << nc1 + 1 << ", " << maxRE << ", " << rlsSamples.size() << "/"
						  << totalInclusiveSamples * 4 << ", "					// The 4 is for the number of rotations per collected sample.
				 		  << watch.get_duration_current() << ";" << std::endl;
			}
		}

		sdfFile.close();
		rlsFile.close();

		printf( "<< Finished generating %d samples in %f secs.\n", nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}