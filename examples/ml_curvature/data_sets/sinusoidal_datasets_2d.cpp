/**
 * Generate datasets for training a feedforward neural network on level-set functions with sinusoidal interfaces.
 * The level-set is implemented as the signed-distance function to a parameterized sine wave that is subject to an
 * affine transformation to allow for pattern variations.  See arclength_parameterized_sine_2d.h for more details.
 *
 * To avoid a disproportionate ratio of h*kappa ~ 0 samples, we collect samples that are close to zero using a
 * probabilistic approach.  Seek the [SAMPLING] subsection in this file.
 *
 * This approach is currently implemented only for 2D level-set functions.
 *
 * Developer: Luis Ángel.
 * Date: May 28, 2020.
 * Updated: May 3, 2021.
 *
 * [¡Important Note!]  This used to sample the level-set function from a look-up table.  However, that turned out being
 * impractical for domains where the maximum level of refinement was greater than 8.  An important observation that led
 * us to the current implementation was that even if we had the exact signed distance function, reinitializing such a
 * level-set introduced noise into the exact values of phi.  We thus took advantage of this extraneous noise for
 * generating our "reinitialized" samples.  In reality, using bisection and Newton Raphson with unbounded iterations
 * was way more efficient than the look-up process that we initially planned.  The arclength_parameterized_sine_2d.h
 * file has been modified to reflect these changes.
 *
 * [Update on May 3, 2020] Adapted code to handle data sets where the gradient of the negative-curvature stencil has an
 * angle in the range [0, pi/2].  That is, we collect samples where the gradient points towards the first quadrant of
 * the local coordinate system centered at the 00 node.  This tries to simplify the architecture of the neural network.
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
 * @param [out] distances True normal distances from full neighborhood to sine wave using Newton-Raphson's root-finding.
 * @param [out] xOnGamma x-coordinate of normal projection of grid node onto interface.
 * @param [out] yOnGamma y-coordinate of normal projection of grid node onto interface.
 * @param [in,out] visitedNodes Hash map functioning as a memoization mechanism to speed up access to visited nodes.
 * @param [out] grad Gradient at center node.
 * @return Vector with sampled phi values and target dimensionless curvature.
 * @throws runtime exception if Newton-Raphson's didn't converge to a global minimum.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const my_p4est_node_neighbors_t *neighbors, const double *phiReadPtr, const ArcLengthParameterizedSine& sine,
	std::vector<double>& distances, double& xOnGamma, double& yOnGamma,
	std::unordered_map<p4est_locidx_t, Point2>& visitedNodes, double grad[P4EST_DIM] )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h*kappa.
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// Include h*kappa as well.

	int s;												// Index to fill in the sample vector.
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double u, centerU;
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
		// We should have this information in the memoization map.
		assert( visitedNodes.find( stencil[s] ) != visitedNodes.end() );

		u = visitedNodes[stencil[s]].x;				// First component is the parameter u (in local coordinate system).
		newDistance = visitedNodes[stencil[s]].y;	// Second component is the signed distance to Gamma.

//		if( s == 4 )
//		{
//			double v = sine.getA() * sin( sine.getOmega() * u );	// Recalculating point on interface (still in canonical coords).
//		}

		double relDiff = (ABS( newDistance ) - distances[s]) / distances[s];
		if( relDiff > 1e-4  )							// Verify that new point is closer than previous approximation.
		{
			std::ostringstream stream;
			stream << "Failure with node " << stencil[s] << " in stencil of " << nodeIdx
				   << std::scientific << std::setprecision( 15 ) << ".  New dist: " << ABS( newDistance )
				   << ".  Old dist: " << distances[s]
				   << ".  Rel diff: " << relDiff;
			throw std::runtime_error( stream.str() );
		}

		distances[s] = newDistance;						// Root finding was successful: keep minimum distance.

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
	const int MAX_REFINEMENT_LEVEL = 6;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes/dim.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Mesh size.
	const double FLAT_LIM_HK = 0.5 * H;			// Flatness limit for h*kappa (note it depends now on H).
	const int NUM_AMPLITUDES = 33;				// Originally: (int)pow( 2, 5 ) + 1; number of distinct sine amplitudes.
	const double H_BASE = 1. / pow( 2, MIN( MAX_REFINEMENT_LEVEL, 7 ) );	// Base sampling on the mesh size for a max lvl of ref = 6 (used to be 7).

	const double MIN_A = 1.5 * H_BASE;			// An almost flat wave.
	const double MAX_A = HALF_D / 2;			// Tallest wave amplitude.
	const double MAX_HKAPPA_LB = H * (21. + 1 / 3.);	// Lower and upper bounds for maximum h*kappa (used for
	const double MAX_HKAPPA_UB = H * (85. + 1 / 3.);	// discriminating samples --see below for details).
	const double MAX_HKAPPA_MIDPOINT = (MAX_HKAPPA_LB + MAX_HKAPPA_UB) / 2;

	const double HALF_AXIS_LEN = (MAX_D - MIN_D) * M_SQRT2 / 2 + 2 * H;	// Adding some padding of 2H to wave main axis.

	const double MIN_THETA = -M_PI_2;			// For each amplitude, we vary the rotation of the wave with respect
	const double MAX_THETA = +M_PI_2;			// to the horizontal axis from -pi/2 to +pi/2, without the end point.
	const int NUM_THETAS = 34;					// Originally: (int)pow( 2, MAX_REFINEMENT_LEVEL - 2 ) + 2;
												// where the last 2 is to account for skipping +pi/4 (in -pi/4 to pi/4).

	// Destination folder.
	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-0521/data/" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";
	const int NUM_COLUMNS = num_neighbors_cube + 2;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	kutils::generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::mt19937 gen{}; 				// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );
	std::uniform_real_distribution<double> uniformDistribution;

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing datasets
		// to files.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Collecting samples from a sinusoidal wave function..." << std::endl;

		//////////////////////////////////////////// Generating the datasets ///////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i distinct amplitudes with maximum refinement level of %i and finest h = %g\n",
			NUM_AMPLITUDES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples files: sine_rls_X.csv for reinitialized level-set function and sine_sdf_X.csv for
		// signed distance function values.
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
		double A_DIST = MAX_A - MIN_A;						// Amplitudes are in [1.5*H_BASE, 0.5], inclusive.
		double linspaceA[NUM_AMPLITUDES];
		for( int i = 0; i < NUM_AMPLITUDES; i++ )			// Uniform linear space from 0 to 1, with NUM_AMPLITUDES steps.
			linspaceA[i] = (double)( i ) / (NUM_AMPLITUDES - 1.0);

		// Variables to control the spread of rotation angles per amplitude, which must vary uniformly from MIN_THETA to
		// MAX_THETA, in a finite number of steps.
		const double THETA_DIST = MAX_THETA - MIN_THETA;	// As defined above, in [-pi/2, +pi/2).
		double linspaceTheta[NUM_THETAS];
		for( int i = 0; i < NUM_THETAS; i++ )				// Uniform linear space from 0 to 1, with NUM_TETHAS steps.
			linspaceTheta[i] = (double)( i ) / (NUM_THETAS - 1.0);

		// Domain information, applicable to all sinusoidal interfaces.
		int n_xyz[] = { 1, 1, 1 };							// One tree per dimension.
		double xyz_min[] = { MIN_D, MIN_D, MIN_D };			// Square domain.
		double xyz_max[] = { MAX_D, MAX_D, MAX_D };
		int periodic[] = { 0, 0, 0 };						// Non-periodic domain.

		// Printing header for log.
		std::cout << "Amplitude Idx, Omega Idx, Amplitude Val, Omega Val, Max Rel Error, Min Rel Error, Num Samples, Time" << std::endl;

		unsigned long nSamples = 0;
		int na = 0;
		double maxRelAbsError = 0;
		for( ; na < NUM_AMPLITUDES; na++ )					// Go through all wave amplitudes evaluated.
		{
			const double A = MIN_A + linspaceA[na] * A_DIST;			// Amplitude to be evaluated.

			const double MIN_OMEGA = sqrt( MAX_HKAPPA_LB / (H * A) );	// Range of frequencies to ensure that the max
			const double MAX_OMEGA = sqrt( MAX_HKAPPA_UB / (H * A) );	// kappa is in the range of [21+1/3, 85+1/3].
			const double OMEGA_DIST = MAX_OMEGA - MIN_OMEGA;
			const double OMEGA_PEAK_DIST = M_PI_2 * (1 / MIN_OMEGA - 1 / MAX_OMEGA);	// Shortest u-distance between crests obtained with omega max and min.
			const double LIN_PROPORTION = 1. + log2( H_BASE / H ) / 3.;					// Linearly from 1 to 2 as max lvl of ref goes from 7 to 10.
			const int NUM_OMEGAS = (int)ceil( OMEGA_PEAK_DIST / H_BASE * LIN_PROPORTION ) + 1;	// Num. of omegas per amplitude.
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
					ArcLengthParameterizedSine sine( A, OMEGA, T[0], T[1], THETA, HALF_AXIS_LEN, gen, uniformDistribution );

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

					Vec curvature, normal[P4EST_DIM];
					ierr = VecDuplicate( phi, &curvature );
					CHKERRXX( ierr );
					for( auto& dim : normal )
					{
						VecCreateGhostNodes( p4est, nodes, &dim );
						CHKERRXX( ierr );
					}

					// Calculate the level-set function values for each independent node (i.e. locally owned and ghost
					// nodes).  Save the exact signed distance to the sine interface at the same time to avoid double
					// work (wasted iterations for bisection/Newton-Raphson).
					double *phiPtr;
					ierr = VecGetArray( phi, &phiPtr );
					CHKERRXX( ierr );

					std::unordered_map<p4est_locidx_t, Point2> visitedNodes( nodes->num_owned_indeps );	// Memoization.
					for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
					{
						double xyz[P4EST_DIM];
						node_xyz_fr_n( n, p4est, nodes, xyz );
						sine.toCanonicalCoordinates( xyz[0], xyz[1] );		// Change of coordinates.
						double valOfDerivative = 1, distance;
						double u = distThetaDerivative( n, xyz[0], xyz[1], sine, gen, uniformDistribution, valOfDerivative, distance );
						double comparativeY = sine.getA() * sin( sine.getOmega() * xyz[0] );

						// Fix sign: points above sine wave are negative, points below are positive.
						if( xyz[1] > comparativeY )
							distance *= -1;

						// Save values.
						phiPtr[n] = distance;						// To be reinitialized.
						visitedNodes[n] = Point2( u, distance );	// Memorize information for visited node.
					}

					ierr = VecRestoreArray( phi, &phiPtr );
					CHKERRXX( ierr );

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
							// data sets for resolutions higher than the base of max refinement level of MIN( lvl, 7 ).
							if( uniformDistribution( gen ) > H / H_BASE )
								continue;

							if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
							{
								double xOnGamma, yOnGamma, gradient[P4EST_DIM];
								std::vector<double> distances;						// Holds the signed distances.
								std::vector<double> data = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil,
									p4est, nodes, &nodeNeighbors, phiReadPtr, sine, distances, xOnGamma, yOnGamma,
									visitedNodes, gradient );

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

										for( auto& component : gradient )	// Flip sign of gradient too.
											component *= -1.0;
									}

									// Rotate stencil so that gradient at node 00 has an angle in first quadrant.
									kutils::rotateStencilToFirstQuadrant( data, gradient );
									kutils::rotateStencilToFirstQuadrant( distances, gradient );

									rlsSamples.push_back( data );			// Store original sample.
									sdfSamples.push_back( distances );

									// Data augmentation by reflection along y=x line.
									kutils::reflectStencil_yEqx( data );
									kutils::reflectStencil_yEqx( distances );
									rlsSamples.push_back( data );			// Store augmented sample too.
									sdfSamples.push_back( distances );

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

				maxRelAbsError = MAX( maxRE, maxRelAbsError );
			}
		}

		// Close files.
		rlsFile.close();
		sdfFile.close();

		printf( "<< Finished generating %i distinct amplitudes and %lu samples with max rel error %f in %f secs.\n",
			na, nSamples, maxRelAbsError, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}