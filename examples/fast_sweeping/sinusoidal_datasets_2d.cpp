/**
 * Generate datasets for training a feedforward neural network on level-set functions with sinusoidal interfaces.
 * The level-set is implemented as the signed-distance function to a parameterized sine wave that is subject to an
 * affine transformation to allow for pattern variations.  See arclength_parameterized_sin_2d.h for more details.
 *
 * To avoid a disproportionate ratio of h*kappa ~ 0 samples, we collect samples that are close to zero using a
 * probabilistic approach.  Seek the [SAMPLING] subsection in this file.
 *
 * This approach is currently implemented only for 2D level-set functions.
 *
 * Developer: Luis Ángel.
 * Date: May 28, 2020.
 * Updated: November 7, 2020.
 *
 * [¡Important Note!]  This used to sample the level-set function from a look-up table.  However, that turned out being
 * impractical for domains where the maximum level of refinement was greater than 8.  An important observation that led
 * us to the current implementation was that even if we had the exact signed distance function, reinitializing such a
 * level-set introduced noise into the exact values of phi.  We thus took advantage of this extraneous noise for
 * generating our "reinitialized" samples.  In reality, using bisection and Newton Raphson with unbounded iterations
 * was way more efficient than the look-up process that we initially planned.  The arclength_parameterized_sine_2d.h
 * file has been modified to reflect these changes.
 *
 * TODO: Source code needs further refactoring now that we discarded the look-up table approach.
 * TODO: Improve sampleNodeAdjacentToInterface.
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
//#include <src/my_p8est_level_set.h>
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

#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include "arclength_parameterized_sine_2d.h"
#include "local_utils.h"
#include <unordered_map>


/**
 * Generate the sample row of level-set function values and target h*kappa for a node that has been found next to the
 * sine wave interface.  We assume that this query node is effectively adjacent to Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] neighbors Pointer to neighbors data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] sine The level-set function with a sinusoidal interface.
 * @param [in] gen Random number generator.
 * @param [in] normalDistribution A standard normal random distribution generator.
 * @param [out] distances True normal distances from full neighborhood to sine wave using Newton-Raphson's root-finding.
 * @param [out] xOnGamma x-coordinate of normal projection of grid node onto interface.
 * @param [out] yOnGamma y-coordinate of normal projection of grid node onto interface.
 * @param [in,out] visitedNodes Hash map functioning as a memoization mechanism to speed up access to visited nodes.
 * @return Vector with sampled phi values and target dimensionless curvature.
 * @throws runtime exception if Newton-Raphson's didn't converge to a global minimum.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const my_p4est_node_neighbors_t *neighbors, const double *phiReadPtr, const ArcLengthParameterizedSine& sine,
	std::mt19937& gen, std::normal_distribution<double>& normalDistribution, std::vector<double>& distances,
	double& xOnGamma, double& yOnGamma, std::unordered_map<p4est_locidx_t, Point2>& visitedNodes )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h*kappa.
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// Include h*kappa as well.

	int s;												// Index to fill in the sample vector.
	double grad[P4EST_DIM];
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double u, valOfDerivative, centerU;
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

		// Transform point on interface to sine-wave canonical coordinates.
		sine.toCanonicalCoordinates( xyz[0], xyz[1] );
		sine.toCanonicalCoordinates( pOnInterfaceX, pOnInterfaceY );
		pOnInterfaceY = sine.getA() * sin( sine.getOmega() * pOnInterfaceX );	// Better approximation to y on Gamma.

		// Compute current distance to Gamma using the improved y.
		dx = xyz[0] - pOnInterfaceX;
		dy = xyz[1] - pOnInterfaceY;
		distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

		// Find parameter u that yields "a" minimum distance between point and sine-wave using Newton-Raphson's method.
		if( visitedNodes.find( stencil[s] ) != visitedNodes.end() )		// Speed up queries.
		{
			u = visitedNodes[stencil[s]].x;				// First component is the parameter u.
			newDistance = visitedNodes[stencil[s]].y;	// Second component is the distance to Gamma.
		}
		else
		{
			valOfDerivative = 1;
			u = distThetaDerivative( stencil[s], xyz[0], xyz[1], sine, gen, normalDistribution, valOfDerivative, newDistance );
			visitedNodes[stencil[s]] = Point2( u, newDistance );		// Memorize information for visited node.

//			if( s == 4 )
//			{
//				double v = sine.getA() * sin( sine.getOmega() * u );	// Recalculating point on interface (still in canonical coords).
//			}

			if( newDistance - distances[s] > EPS )
			{
				std::ostringstream stream;
				stream << "Failure with node " << stencil[s] << " in stencil of " << nodeIdx
					   << ".  Val. of Der: " << std::scientific << valOfDerivative
					   << std::fixed << std::setprecision( 15 ) << ".  New dist: " << newDistance
					   << ".  Old dist: " << distances[s];
				throw std::runtime_error( stream.str() );
			}
		}

		distances[s] = newDistance;						// Root finding was successful: keep minimum distance.

		if( sample[s] < 0 )								// Fix sign.
			distances[s] *= -1;

		if( s == 4 )									// For center node we need the parameter u to yield curvature.
			centerU = u;
	}

	sample[s] = H * sine.curvature( centerU );			// Last column holds h*kappa.
	distances.push_back( sample[s] );

	return sample;
}


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const int NUM_REINIT_ITERS = 10;			// Number of iterations for PDE reintialization.
	const double MIN_D = -0.5, MAX_D = -MIN_D;								// The canonical space is [-1/2, +1/2]^2.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const double FLAT_LIM_HK = 0.004;										// Flatness limit for dimensionless curvature.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.
	const int NUM_AMPLITUDES = 33;				// Originally: (int)pow( 2, 5 ) + 1; tumber different sine wave amplitudes.

	const double MIN_A = 1.5 * H;				// An almost flat wave.
	const double MAX_A = HALF_D / 2;			// Tallest wave amplitude.
	const double MAX_HKAPPA_LB = 1.0 / 6.0;		// Lower and upper bounds for maximum h*kappa (used for discriminating
	const double MAX_HKAPPA_UB = 2.0 / 3.0;		// which samples to keep -- see below for details).
	const double MAX_HKAPPA_MIDPOINT = ( MAX_HKAPPA_LB + MAX_HKAPPA_UB ) / 2;

	const double HALF_AXIS_LEN = ( MAX_D - MIN_D ) * M_SQRT2 / 2 + 2 * H;	// Adding some padding of 2H to wave main axis.

	const double MIN_THETA = -M_PI_4;			// For each amplitude, we vary the rotation of the wave with respect
	const double MAX_THETA = +M_PI_4;			// to the horizontal axis from -pi/4 to +pi/4, without the end point.
	const int NUM_THETAS = 34;					// Originally: (int)pow( 2, MAX_REFINEMENT_LEVEL - 2 ) + 2;
												// where the last 2 is to account for skipping +pi/4.

	char strFlatLimHk[10];
	sprintf( strFlatLimHk, "%.3f", FLAT_LIM_HK );
	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-1120/data-" + std::string( strFlatLimHk ) + "/"
								  + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";		// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::mt19937 gen{}; 				// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );
	std::uniform_real_distribution<double> uniformDistribution;				// Used for collecting low h*kappa values.
	std::normal_distribution<double> normalDistribution;					// Used for bracketing and root finding.

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing datasets
		// to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Collecting samples from a sinusoidal wave function..." << std::endl;

		//////////////////////////////////////////// Generating the datasets ///////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i distinct amplitudes with maximum refinement level of %i and finest h = %g\n",
			NUM_AMPLITUDES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples files: sine_rls_X.csv for reinitialized level-set, sine_sdf_X.csv for signed-distance function values.
		std::ofstream rlsFile;
		std::string rlsFileName = DATA_PATH + "sine_rls_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		rlsFile.open( rlsFileName, std::ofstream::trunc );
		if( !rlsFile.is_open() )
			throw std::runtime_error( "Output file " + rlsFileName + " couldn't be opened!" );

		std::ofstream sdfFile;
		std::string sdfFileName = DATA_PATH + "sine_sdf_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		sdfFile.open( sdfFileName, std::ofstream::trunc );
		if( !sdfFile.is_open() )
			throw std::runtime_error( "Output file " + sdfFileName + " couldn't be opened!" );

		// Write column headers: enforcing strings by adding quotes around them.
		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		rlsFile << headerStream.str() << std::endl;
		sdfFile << headerStream.str() << std::endl;

		rlsFile.precision( 15 );							// Precision for floating point numbers.
		sdfFile.precision( 15 );

		// Variables to control the spread of sine waves' amplitudes, which must vary uniformly from MIN_A to MAX_A.
		double A_DIST = MAX_A - MIN_A;						// Amplitudes are in [1.5H, 0.5-2H], inclusive.
		double linspaceA[NUM_AMPLITUDES];
		for( int i = 0; i < NUM_AMPLITUDES; i++ )			// Uniform linear space from 0 to 1, with NUM_AMPLITUDES steps.
			linspaceA[i] = (double)( i ) / (NUM_AMPLITUDES - 1.0);

		// Variables to control the spread of ratation angles per amplitude, which must vary uniformly from MIN_THETA to
		// MAX_THETA, in a finite number of steps.
		const double THETA_DIST = MAX_THETA - MIN_THETA;	// As defined above, in [-pi/4, +pi/4).
		double linspaceTheta[NUM_THETAS];
		for( int i = 0; i < NUM_THETAS; i++ )				// Uniform linear space from 0 to 1, with NUM_TETHAS steps.
			linspaceTheta[i] = (double)( i ) / (NUM_THETAS - 1.0);

		// Domain information, applicable to all sinusoidal interfaces.
		int n_xyz[] = {1, 1, 1};							// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		// Printing header for log.
		std::cout << "Amplitude Idx, Omega Idx, Amplitude Val, Omega Val, Max Rel Error, Min Rel Error, Num Samples, Time" << std::endl;

		int nSamples = 0;
		int na = 0;
		for( ; na < NUM_AMPLITUDES; na++ )					// Go through all wave amplitudes evaluated.
		{
			const double A = MIN_A + linspaceA[na] * A_DIST;			// Amplitude to be evaluated.

			const double MIN_OMEGA = sqrt( MAX_HKAPPA_LB / ( H * A ) );	// Range of frequencies to ensure that the max
			const double MAX_OMEGA = sqrt( MAX_HKAPPA_UB / ( H * A ) );	// h*kappa is in the range of [1/6, 2/3].
			const double OMEGA_DIST = MAX_OMEGA - MIN_OMEGA;
			const double OMEGA_PEAK_DIST = M_PI_2 * ( 1 / MIN_OMEGA - 1 / MAX_OMEGA );	// Distance between u-values with highest peaks.
			const int NUM_OMEGAS = (int)ceil( OMEGA_PEAK_DIST / H ) + 1;				// Num. of omegas per amplitude.
			double linspaceOmega[NUM_OMEGAS];
			for( int i = 0; i < NUM_OMEGAS; i++ )			// Uniform linear space from 0 to 1, with NUM_OMEGA steps.
				linspaceOmega[i] = (double)( i ) / ( NUM_OMEGAS - 1.0 );

			for( int no = 0; no < NUM_OMEGAS; no++ )		// Evaluate all frequencies for the same amplitude.
			{
				std::vector<std::vector<double>> rlsSamples;	// Reinitialized level-set function samples.
				std::vector<std::vector<double>> sdfSamples;	// Exact signed-distance function samples.
				double maxRE = 0;								// Maximum relative error for verification.
				double minRE = PETSC_MAX_REAL;					// Minimum relative error.

				const double OMEGA = MIN_OMEGA + linspaceOmega[no] * OMEGA_DIST;
				for( int nt = 0; nt < NUM_THETAS - 1; nt++ )	// Various rotation angles for same amplitude and frequency
				{												// (skipping last endpoint because we do augmentation).
					const double THETA = MIN_THETA + linspaceTheta[nt] * THETA_DIST;	// Rotation of main sine axis.
					const double T[] = {
						( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen ),	// Translate origin coords by a random
						( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen )	// perturbation from grid's midpoint.
					};

					// Level-set function with a sinusoidal interface.
					ArcLengthParameterizedSine sine( A, OMEGA, T[0], T[1], THETA, HALF_AXIS_LEN, gen, normalDistribution );

					// p4est variables and data structures: these change with every sine wave because we must refine the
					// trees according to the new waves's origin and amplitude.
					p4est_t *p4est;
					p4est_nodes_t *nodes;
					my_p4est_brick_t brick;
					p4est_ghost_t *ghost;
					p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

					// Splitting criterion: notice the fine resolution.
					splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sine );

					// Create the forest using interpolation-based sinusoid-interface level-set as refinement criterion.
					p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
					p4est->user_pointer = (void *)( &levelSetSC );

					// Refine and recursively partition forest.
//					double timing = watch.get_duration_current();
					my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
					my_p4est_partition( p4est, P4EST_TRUE, nullptr );
//					std::cout << "Refinement and partition: " << watch.get_duration_current() - timing << std::endl;

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

					Vec curvature, normal[P4EST_DIM];
					ierr = VecDuplicate( phi, &curvature );
					CHKERRXX( ierr );
					for( auto& dim : normal )
					{
						VecCreateGhostNodes( p4est, nodes, &dim );
						CHKERRXX( ierr );
					}

					// Calculate the level-set function values for each independent node (i.e. locally owned and ghost nodes).
//					timing = watch.get_duration_current();
					sample_cf_on_nodes( p4est, nodes, sine, phi );
//					std::cout << "Sampling level-set functon: " << watch.get_duration_current() - timing << std::endl;

					// Reinitialize level-set function.
					my_p4est_level_set_t ls( &nodeNeighbors );
					ls.reinitialize_2nd_order( phi, NUM_REINIT_ITERS );

					// Compute curvature with reinitialized data, which will be interpolated at the interface.
					compute_normals( nodeNeighbors, phi, normal );
					compute_mean_curvature( nodeNeighbors, normal, curvature );

					// Prepare interpolation.
					my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
					interpolation.set_input( curvature, linear );

					// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface;
					// these are the points we'll use to create our sample files.
					std::vector<p4est_locidx_t> indices;
					NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
					nodesAlongInterface.getIndices( &phi, indices );

					// Getting the full uniform stencils of interface points.
					const double *phiReadPtr;
					ierr = VecGetArrayRead( phi, &phiReadPtr );
					CHKERRXX( ierr );

					// [SAMPLING] Now, collect samples with reinitialized level-set function values and target h*kappa.
//					timing = watch.get_duration_current();
					std::unordered_map<p4est_locidx_t, Point2> visitedNodes( nodes->num_owned_indeps );	// Memoization.
					for( auto n : indices )
					{
						double xyz[P4EST_DIM];						// Position of node at the center of the stencil.
						node_xyz_fr_n( n, p4est, nodes, xyz );
						if( ABS( xyz[0] - MIN_D ) <= 4 * H || ABS( xyz[0] - MAX_D ) <= 4 * H ||
							ABS( xyz[1] - MIN_D ) <= 4 * H || ABS( xyz[1] - MAX_D ) <= 4 * H )
							continue;								// Skip conflicting samples.

						std::vector<p4est_locidx_t> stencil;		// Contains 9 values in 2D.

						try
						{
							// Randomly deciding if we proceed with these node or not.  Otherwise, we'll get enourmous
							// data sets for resolutions higher than the base of max refinement level of 7.
							if( uniformDistribution( gen ) > 128. * H )
								continue;

							if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
							{
								double xOnGamma, yOnGamma;
								std::vector<double> distances;		// Holds the signed distances.
								std::vector<double> data = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil,
									p4est, nodes, &nodeNeighbors, phiReadPtr, sine, gen, normalDistribution, distances,
									xOnGamma, yOnGamma, visitedNodes );

								if( ABS( data[NUM_COLUMNS - 2] ) < FLAT_LIM_HK )	// Skip flat surfaces.
									continue;

								// Accumulating samples: we always take samples with h*kappa > midpoint; for those with
								// h*kappa <= midpoint, we take them with an easing-off probability from 1 to 0.05,
								// where Pr(h*kappa = midpoint) = 1 and Pr(h*kappa = 0) = 0.05.
								if( ABS( data[NUM_COLUMNS - 2] ) > MAX_HKAPPA_MIDPOINT ||
									uniformDistribution( gen ) <= 0.05 + ( sin( -M_PI_2 + ABS( data[NUM_COLUMNS - 2] )
									* M_PI / MAX_HKAPPA_MIDPOINT ) + 1 ) * 0.95 / 2  )
								{
									data[NUM_COLUMNS - 1] = H * interpolation( xOnGamma, yOnGamma );	// Attach interpolated h*kappa.
									distances.push_back( 0 );											// Dummy column.

									if( data[NUM_COLUMNS - 2] > 0 )	// Flip sign for positive samples.
									{
										for( int i = 0; i < NUM_COLUMNS; i++ )
										{
											data[i] *= -1.0;
											distances[i] *= -1.0;
										}
									}

									for( int i = 0; i < 4; i++ )	// Data augmentation by rotating samples 90 degrees
									{								// three times.
										rlsSamples.push_back( data );
										sdfSamples.push_back( distances );

										rotatePhiValues90( data, NUM_COLUMNS );
										rotatePhiValues90( distances, NUM_COLUMNS );
									}

									// Error metric for validation.
									for( int i = 0; i < NUM_COLUMNS - 2; i++ )
									{
										double error = ( distances[i] - data[i] ) / H;
										maxRE = MAX( maxRE, ABS( error ) );
										minRE = MIN( minRE, ABS( error ) );
									}
								}
							}
						}
						catch( std::exception &e )
						{
							std::cerr << "Node " << n << ".  Omega #" << no << ".  Theta #" << nt << ": \n    "
									  << e.what() << std::endl;
						}
					}
//					std::cout << "Collecting training samples: " << watch.get_duration_current() - timing << std::endl;

					ierr = VecRestoreArrayRead( phi, &phiReadPtr );
					CHKERRXX( ierr );

					// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
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

				// Write to file samples collected for all sines with the same amplitude and same frequency but
				// randomized origin and for all rotations of main axis.
				for( const auto& row : rlsSamples )
				{
					std::copy( row.begin(), row.end() - 2, std::ostream_iterator<double>( rlsFile, "," ) );		// Inner elements.
					rlsFile << std::setprecision( 8 ) << row[NUM_COLUMNS - 2] << "," << row.back()
							<< std::setprecision( 15 ) << std::endl;
				}

				// Same for signed distance function.
				for( const auto& row : sdfSamples )
				{
					std::copy( row.begin(), row.end() - 2, std::ostream_iterator<double>( sdfFile, "," ) );		// Inner elements.
					sdfFile << std::setprecision( 8 ) << row[NUM_COLUMNS - 2] << "," << row.back()
							<< std::setprecision( 15 ) << std::endl;
				}

				nSamples += rlsSamples.size();

				// Log output.
				std::cout << na + 1 << ", " << no + 1 << ", " << A << ", " << OMEGA << ", " << maxRE << ", "
						  << minRE << ", " << rlsSamples.size() << ", " << watch.get_duration_current() << ";"
						  << std::endl;
			}
		}

		// Close files.
		rlsFile.close();
		sdfFile.close();

		printf( "<< Finished generating %i distinct amplitudes and %i samples in %f secs.\n",
			na, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}