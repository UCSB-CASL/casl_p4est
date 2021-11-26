/**
 * Generate datasets for training a neural network on level-set functions with sinusoidal interfaces.
 * The level-set is implemented as the signed-distance function to a parameterized sine wave that is subject to an
 * affine transformation to allow for pattern variations.
 *
 * @see arclength_parameterized_sine_2d.h
 *
 * To avoid a disproportionate ratio of hk ~ 0 samples, we use an easing-off probability distribution for subsampling.
 * Seek the [SAMPLING] subsection in this file.
 *
 * Developer: Luis Ángel.
 * Date: May 28, 2020.
 * Updated: November 25, 2021.
 *
 * [Update on May 3, 2020] Adapted code to handle data sets where the gradient of the negative-curvature stencil has an
 * angle in the range [0, pi/2].  That is, we collect samples where the gradient points towards the first quadrant of
 * the local coordinate system centered at the 00 node.  This tries to simplify the architecture of the neural network.
 * [Update on October 27, 2021] Data sets include normal unit vector components as additional training cues.
 */

// System.
#include <stdexcept>
#include <iostream>

#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_curvature_ml.h>

#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include "arclength_parameterized_sine_2d.h"
#include <unordered_map>


/**
 * Generate the sample row of h-normalized level-set function values and compute target hk for a node next to the sine
 * wave interface.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] rlsPhiReadPtr Pointer to reinitialized level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] sine The level-set function with a sinusoidal interface.
 * @param [out] distances True normal distances from full neighborhood to sine wave using Newton-Raphson's root-finding.
 * @param [out] pOnGamma Normal projection of grid node onto interface based on reinitialized level-set function.
 * @param [in,out] visitedNodes Hash map functioning as a memoization mechanism to speed up access to visited nodes.
 * @param [in] rlsNormalReadPtr Pointer to normal unit vector components backed by PETSc parallel vectors.
 * @param [out] hk Target dimensionless curvature.
 * @return Vector with sampled phi values and target dimensionless curvature.
 * @throws runtime exception if Newton-Raphson's didn't converge to a global minimum.
 */
std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS, const double H,
												   const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est,
												   const p4est_nodes_t *nodes, const double *rlsPhiReadPtr,
												   const ArcLengthParameterizedSine& sine,
												   std::vector<double>& distances, double pOnGamma[P4EST_DIM],
												   std::unordered_map<p4est_locidx_t, Point2>& visitedNodes,
												   const double *rlsNormalReadPtr[P4EST_DIM], double& hk )
{
	std::vector<double> sample;							// (Reinitialized) level-set function values and target hk.
	sample.reserve( NUM_COLUMNS );
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// Include target hk as well.

	int s;												// Index to fill in the sample vector.
	double xyz[P4EST_DIM], pOnInterface[P4EST_DIM];
	double u, centerU;
	double dx, dy, newDistance;
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample.push_back( rlsPhiReadPtr[stencil[s]] );	// This is the distance obtained after reinitialization.

		// Approximate position of point projected on interface.
		const double grad[P4EST_DIM] = {DIM( rlsNormalReadPtr[0][stencil[s]], rlsNormalReadPtr[1][stencil[s]], rlsNormalReadPtr[2][stencil[s]] )};
		node_xyz_fr_n( stencil[s], p4est, nodes, xyz );
		for( int dim = 0; dim < P4EST_DIM; dim++ )
			pOnInterface[dim] = xyz[dim] - grad[dim] * sample[s];

		if( s == 4 )	// Rough estimation of point on interface, where curvature will be interpolated.
		{
			for( int dim = 0; dim < P4EST_DIM; dim++ )
				pOnGamma[dim] = pOnInterface[dim];
		}

		// Transform point on interface to sine-wave canonical coordinates.
		sine.toCanonicalCoordinates( xyz[0], xyz[1] );
		sine.toCanonicalCoordinates( pOnInterface[0], pOnInterface[1] );
		pOnInterface[1] = sine.getA() * sin( sine.getOmega() * pOnInterface[0] );	// Better approximation to y on Gamma.

		// Compute current distance to Gamma using the improved y.
		dx = xyz[0] - pOnInterface[0];
		dy = xyz[1] - pOnInterface[1];
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

		double relDiff = (ABS( newDistance ) - distances[s]) / H;
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

		// Normalize by H.
		sample[s] /= H;
		distances[s] /= H;
	}

	hk = H * sine.curvature( centerU );					// Return target dimensionless curvature
	return sample;										// and sample of h-normalized level-set values.
}


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<unsigned short> maxRL( pl, 8, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 8)" );
	param_t<double> minHK( pl, 0.004, "minHK", "Minimum dimensionless curvature (default: 0.004)" );
	param_t<double> maxHK( pl, 2./3, "maxHK", "Maximum dimensionless curvature (default: 2./3)" );
	param_t<double> easeOffMaxProb( pl, 0.4, "easeOffMaxProb", "Easing-off distribution max probability to keep midpoint hk (default: 0.4)" );
	param_t<int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<std::string> outputDir( pl, "/Volumes/YoungMinEXT/k_ecnet_data", "outputDir", "Path where files will be written (default: same folder as the executable)" );
	param_t<bool> writeSDF( pl, true, "writeSDF", "Write signed distance function samples (default: 1)" );
	param_t<bool> verbose( pl, false, "verbose", "Show or not debugging messages (default: 0)" );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing datasets
		// to files.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Collecting samples from a sinusoidal wave function..." << std::endl;

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating sphere data sets for curvature inference" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////
		// We must set all bounds and metrics relative to hk, otherwise we'll experience data explosion very easily with
		// high resolutions.

		const double ONE_OVER_H = 1 << maxRL();
		const double H = 1. / ONE_OVER_H;				// Mesh size.
		const double MAX_KAPPA = maxHK() * ONE_OVER_H;	// Steepest curvature.
		const double MIN_KAPPA = minHK() * ONE_OVER_H;	// Flattest curvature.
		const double MIN_A = 4. / MAX_KAPPA;			// Slightly flat wave (proportional to the minimum radius in circular data sets).
		const double MAX_A = 1. / MIN_KAPPA;			// Tallest wave amplitude (proportional to the maximum radius in circular data sets).

		// Let's define a bounding box for sampling.  Then, define the whole domain with integer-side lengths based on it.
		const double SAMPLING_BOX_HALF_SIDE_LEN = 2 * MAX_A;
		const double MIN_D = -ceil( (SAMPLING_BOX_HALF_SIDE_LEN + 4 * H) / 0.5 ) * 0.5;
		const double MAX_D = -MIN_D;					// The canonical space is square and has integral side lengths.
		const double HALF_D = (MAX_D - MIN_D) / 2;		// Half domain length.
		const int NUM_AMPLITUDES = (int)((MAX_A - MIN_A) * ONE_OVER_H / 7);	// Number of distinct sine amplitudes (~34 for all resolutions).

		const double MAX_HKAPPA_LB = maxHK() / 2;		// Lower and upper bounds for maximum hk.  These define the
		const double MAX_HKAPPA_UB = maxHK();			// frequencies based on amplitude.
		const double MAX_HKAPPA_MIDPOINT = (MAX_HKAPPA_LB + MAX_HKAPPA_UB) / 2;

		const double HALF_AXIS_LEN = HALF_D*M_SQRT2 + 2*H;	// Adding some padding of 2h to wave main axis.

		const double MIN_THETA = -M_PI_2;		// For each amplitude, we vary the rotation of the wave with respect
		const double MAX_THETA = +M_PI_2;		// to the horizontal axis from -pi/2 to +pi/2, without the end point.
		const int NUM_THETAS = M_PI * SAMPLING_BOX_HALF_SIDE_LEN * ONE_OVER_H / 41;		// Must be ~38 for all resolutions: half circunference / (29h).

		// Destination folder.
		const std::string DATA_PATH = outputDir() + "/" + std::to_string( maxRL() ) + "/";
		const int NUM_COLUMNS = (P4EST_DIM + 1) * num_neighbors_cube + 2;		// Number of columns in dataset.
		std::string COLUMN_NAMES[NUM_COLUMNS];				// Column headers following the x-y truth table of 3-state
		kml::utils::generateColumnHeaders( COLUMN_NAMES );	// variables: includes phi values and normal components.

		// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
		std::mt19937 gen{}; 			// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
		std::uniform_real_distribution<double> uniformDistributionH_2( -H/2, +H/2 );	// For tranlating origin of sine wave coord. syst.
		std::uniform_real_distribution<double> uniformDistribution;						// For subsampling due to target hk.
		std::uniform_real_distribution<double> skipDist;								// For randomly skipping points.

		//////////////////////////////////////////// Generating the datasets ///////////////////////////////////////////

		parStopWatch watch;
		PetscPrintf( mpi.comm(), ">> Began to generate datasets for %i distinct amplitudes with max refinement level of %i and finest h = %g\n",
					 NUM_AMPLITUDES, maxRL(), H );
		watch.start();

		// Prepare samples files: sine_rls_X.csv for reinitialized level-set function and sine_sdf_X.csv for
		// signed distance function values.
		std::ofstream rlsFile;
		std::string rlsFileName = DATA_PATH + "sine_rls_" + std::to_string( maxRL() ) +  ".csv";
		rlsFile.open( rlsFileName, std::ofstream::trunc );
		if( !rlsFile.is_open() )
			throw std::runtime_error( "Output file " + rlsFileName + " couldn't be opened!" );

		std::ofstream sdfFile;
		if( writeSDF() )
		{
			std::string sdfFileName = DATA_PATH + "sine_sdf_" + std::to_string( maxRL()) + ".csv";
			sdfFile.open( sdfFileName, std::ofstream::trunc );
			if( !sdfFile.is_open())
				throw std::runtime_error( "Output file " + sdfFileName + " couldn't be opened!" );
		}

		// Write column headers: enforcing strings by adding quotes around them.
		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		rlsFile << headerStream.str() << std::endl;
		rlsFile.precision( 15 );

		if( writeSDF() )
		{
			sdfFile << headerStream.str() << std::endl;
			sdfFile.precision( 15 );
		}

		// Variables to control the spread of sine waves' amplitudes, which must vary uniformly from MIN_A to MAX_A.
		std::vector<double> linspaceA;
		linspace( MIN_A, MAX_A, NUM_AMPLITUDES, linspaceA );

		// Variables to control the spread of rotation angles per amplitude, which must vary uniformly from MIN_THETA to
		// MAX_THETA, in a finite number of steps.
		std::vector<double> linspaceTheta;
		linspace( MIN_THETA, MAX_THETA, NUM_THETAS, linspaceTheta );

		// Domain information, applicable to all sinusoidal interfaces.
		const int N_TREES = (int)(MAX_D - MIN_D);
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};
		double xyz_min[] = { MIN_D, MIN_D, MIN_D };			// Square domain.
		double xyz_max[] = { MAX_D, MAX_D, MAX_D };
		int periodic[] = { 0, 0, 0 };						// Non-periodic domain.

		// Printing header for log.
		PetscPrintf( mpi.comm(), "Amplitude Idx, Omega Idx, Amplitude Val, Omega Val, Max Rel Error, Min Rel Error, Num Samples, Time\n" );

		unsigned long nSamples = 0;
		int na = 0;
		double maxRelAbsError = 0;
		for( ; na < NUM_AMPLITUDES; na++ )					// Go through all wave amplitudes evaluated.
		{
			const double A = linspaceA[na];					// Amplitude to be evaluated.

			const double MIN_OMEGA = sqrt( MAX_HKAPPA_LB / (H * A) );	// Range of frequencies to ensure that the max
			const double MAX_OMEGA = sqrt( MAX_HKAPPA_UB / (H * A) );	// kappa is in the range of [MAX_KAPPA/2, MAX_KAPPA].
			const double OMEGA_PEAK_DIST = M_PI_2 * (1 / MIN_OMEGA - 1 / MAX_OMEGA);	// Shortest u-distance between crests obtained with omega max and min.
			const int NUM_OMEGAS = (int)ceil( OMEGA_PEAK_DIST * ONE_OVER_H ) + 1;		// Num. of omegas per amplitude.
			std::vector<double> linspaceOmega;
			linspace( MIN_OMEGA, MAX_OMEGA, NUM_OMEGAS, linspaceOmega );

			for( int no = 0; no < NUM_OMEGAS; no++ )			// Evaluate all frequencies for the same amplitude.
			{
				std::vector<std::vector<double>> rlsSamples;	// Reinitialized level-set function samples.
				std::vector<std::vector<double>> sdfSamples;	// Exact signed-distance function samples.
				double maxRE = 0;								// Maximum relative error for verification.
				double minRE = PETSC_MAX_REAL;					// Minimum relative error.

				const double OMEGA = linspaceOmega[no];
				for( int nt = 0; nt < NUM_THETAS - 1; nt++ )	// Various rotation angles for same amplitude and frequency
				{												// (skipping last endpoint because we do augmentation).
					const double THETA = linspaceTheta[nt];		// Rotation of main wave axis.
					const double T[] = {
						(MIN_D + MAX_D) / 2 + uniformDistributionH_2( gen ),	// Translate origin coords by a random
						(MIN_D + MAX_D) / 2 + uniformDistributionH_2( gen )		// perturbation from grid's midpoint.
					};

					// Level-set function with a sinusoidal interface.
					ArcLengthParameterizedSine sine( A, OMEGA, T[0], T[1], THETA, HALF_AXIS_LEN, gen, uniformDistribution, verbose() );

					// p4est variables and data structures: these change with every sine wave because we must refine the
					// trees according to the new waves's origin and amplitude.
					p4est_t *p4est;
					p4est_nodes_t *nodes;
					my_p4est_brick_t brick;
					p4est_ghost_t *ghost;
					p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

					// Splitting criterion.
					splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, maxRL(), &sine, 2.0 );

					// Create the forest using interpolation-based sinusoid-interface level-set as refinement criterion.
					p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
					p4est->user_pointer = (void *)( &levelSetSC );

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

					// A ghosted parallel PETSc vector to store level-set function values.
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

					// Calculate the level-set function values for each independent node (i.e., locally owned and ghost
					// nodes).  Save the exact signed distance to the sine interface at the same time to avoid double
					// work (wasted iterations for bisection/Newton-Raphson).
					double *rlsPhiPtr, *sdfPhiPtr;
					CHKERRXX( VecGetArray( rlsPhi, &rlsPhiPtr ) );
					CHKERRXX( VecGetArray( sdfPhi, &sdfPhiPtr ) );

					std::unordered_map<p4est_locidx_t, Point2> visitedNodes( nodes->num_owned_indeps );	// Memoization.
					for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
					{
						double xyz[P4EST_DIM];
						node_xyz_fr_n( n, p4est, nodes, xyz );
						sine.toCanonicalCoordinates( xyz[0], xyz[1] );		// Change of coordinates.
						double valOfDerivative = 1, distance;
						double u = distThetaDerivative( n, xyz[0], xyz[1], sine, gen, uniformDistribution, valOfDerivative, distance, verbose() );
						double comparativeY = sine.getA() * sin( sine.getOmega() * xyz[0] );

						// Fix sign: points above sine wave are negative, points below are positive.
						if( xyz[1] > comparativeY )
							distance *= -1;

						// Save values.
						rlsPhiPtr[n] = distance;					// To be reinitialized.
						sdfPhiPtr[n] = distance;					// Exact distance.
						visitedNodes[n] = Point2( u, distance );	// Memorize information for visited node.
					}

					CHKERRXX( VecRestoreArray( rlsPhi, &rlsPhiPtr ) );
					CHKERRXX( VecRestoreArray( sdfPhi, &sdfPhiPtr ) );

					// Reinitialize level-set function.
					my_p4est_level_set_t ls( &nodeNeighbors );
					ls.reinitialize_2nd_order( rlsPhi, reinitNumIters() );

					// Compute curvature with reinitialized data, which will be interpolated at the interface.
					compute_normals( nodeNeighbors, rlsPhi, rlsNormal );
					compute_normals( nodeNeighbors, sdfPhi, sdfNormal );
					compute_mean_curvature( nodeNeighbors, rlsPhi, rlsNormal, curvature );

					// Prepare interpolation.
					my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
					interpolation.set_input( curvature, interpolation_method::linear );

					// Once the level-set function is reinitialized, collect nodes on or next to the interface; these
					// are the points we'll use to create our sample files.
					std::vector<p4est_locidx_t> indices;
					NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, (char)maxRL() );
					nodesAlongInterface.getIndices( &rlsPhi, indices );

					// Getting the full uniform stencils of interface points.
					const double *rlsPhiReadPtr;
					CHKERRXX( VecGetArrayRead( rlsPhi, &rlsPhiReadPtr ) );

					const double *rlsNormalReadPtr[P4EST_DIM], *sdfNormalReadPtr[P4EST_DIM];
					for( int dim = 0; dim < P4EST_DIM; dim++ )
					{
						CHKERRXX( VecGetArrayRead( rlsNormal[dim], &rlsNormalReadPtr[dim] ) );
						CHKERRXX( VecGetArrayRead( sdfNormal[dim], &sdfNormalReadPtr[dim] ) );
					}

					// [SAMPLING] Now, collect samples with reinitialized and exact level-set function values and target hk.
					for( auto n : indices )
					{
						double xyz[P4EST_DIM];						// Position of node at the center of the stencil.
						node_xyz_fr_n( n, p4est, nodes, xyz );
						double r = sqrt( SQR(xyz[0] - T[0]) + SQR(xyz[1] - T[1]) );
						if( r > SAMPLING_BOX_HALF_SIDE_LEN )
							continue;								// Skip samples outside of sampling (circular) space.

						std::vector<p4est_locidx_t> stencil;		// Contains 9 values in 2D.
						try
						{
							if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
							{
								double pOnGamma[P4EST_DIM], tgtHK;
								std::vector<double> distances;						// Holds the exact signed distances.
								std::vector<double> data = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil,
									p4est, nodes, rlsPhiReadPtr, sine, distances, pOnGamma, visitedNodes, rlsNormalReadPtr, tgtHK );

								if( ABS( tgtHK ) < minHK() )		// Skip flat surfaces.
									continue;

								// Accumulating samples: we always take samples with hk > midpoint; for those with
								// hk <= midpoint, we take them with an easing-off probability, where
								// Pr(hk = midpoint) = easeOffMaxProb param and Pr(hk = minHK) = 0.01.
								double p = MIN( 1.0, (ABS( tgtHK ) - minHK()) / (MAX_HKAPPA_MIDPOINT - minHK()) );
								if( uniformDistribution( gen ) <= 0.01 + (sin( -M_PI_2 + p * M_PI ) + 1) * (easeOffMaxProb() - 0.01) / 2  )
								{
									for( int dim = 0; dim < P4EST_DIM; dim++ )		// Attach the normal components.
									{
										for( auto s: stencil )
										{
											distances.push_back( sdfNormalReadPtr[dim][s] );
											data.push_back( rlsNormalReadPtr[dim][s] );
										}
									}

									data.push_back( tgtHK );			// Attach target hk.
									distances.push_back( tgtHK );

									data.push_back( H * interpolation( pOnGamma[0], pOnGamma[1] ) );	// Attach interpolated ihk.
									distances.push_back( tgtHK );										// And exact hk for sdf.

									double rlsGrad[P4EST_DIM], sdfGrad[P4EST_DIM];
									for( int dim = 0; dim < P4EST_DIM; dim++ )		// Let's pick a numerically good gradient.
									{
										rlsGrad[dim] = (rlsNormalReadPtr[dim][n] == 0)? EPS : rlsNormalReadPtr[dim][n];
										sdfGrad[dim] = (sdfNormalReadPtr[dim][n] == 0)? EPS : sdfNormalReadPtr[dim][n];
									}

									// Error metric for validation.  Before negative-curvature normalization and
									// reorientation because the signs might be opposite or the sdf gradient might
									// rotate or not the sdf stencil in comparison to the rls stencil.
									for( int i = 0; i < num_neighbors_cube; i++ )
									{
										double error = distances[i] - data[i];
										maxRE = MAX( maxRE, ABS( error ) );
										minRE = MIN( minRE, ABS( error ) );
									}

									// Flip sign for positive samples: treat reinitialized and signed distance samples separately.
									if( data.back() > 0 )
									{
										for( int i = 0; i < NUM_COLUMNS; i++ )
											data[i] *= -1.0;

										for( double& dim : rlsGrad )	// Flip sign of gradient too.
											dim *= -1.0;
									}

									if( distances.back() > 0 )
									{
										for( int i = 0; i < NUM_COLUMNS; i++ )
											distances[i] *= -1.0;

										for( double& dim : sdfGrad )	// Flip sign of gradient too.
											dim *= -1.0;
									}

									// Rotate stencil so that gradient at node 00 has an angle in first quadrant.
									kml::utils::rotateStencilToFirstQuadrant( data, rlsGrad );
									kml::utils::rotateStencilToFirstQuadrant( distances, sdfGrad );

									rlsSamples.push_back( data );			// Store original sample.
									if( writeSDF() )
										sdfSamples.push_back( distances );

									// Data augmentation by reflection along y=x line.
									kml::utils::reflectStencil_yEqx( data );
									kml::utils::reflectStencil_yEqx( distances );
									rlsSamples.push_back( data );			// Store augmented sample too.
									if( writeSDF() )
										sdfSamples.push_back( distances );
								}
							}
						}
						catch( std::exception &e )
						{
							std::cerr << "Node " << n << ".  Omega #" << no << ".  Theta #" << nt << ": \n    "
									  << e.what() << std::endl;
						}
					}

					// Restore access.
					for( int dim = 0; dim < P4EST_DIM; dim++ )
					{
						CHKERRXX( VecRestoreArrayRead( rlsNormal[dim], &rlsNormalReadPtr[dim] ) );
						CHKERRXX( VecRestoreArrayRead( sdfNormal[dim], &sdfNormalReadPtr[dim] ) );
					}
					CHKERRXX( VecGetArrayRead( rlsPhi, &rlsPhiReadPtr ) );

					// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
					for( int dim = 0; dim <P4EST_DIM; dim++ )
					{
						CHKERRXX( VecDestroy( sdfNormal[dim] ) );
						CHKERRXX( VecDestroy( rlsNormal[dim] ) );
					}
					CHKERRXX( VecDestroy( curvature ) );
					CHKERRXX( VecDestroy( sdfPhi ) );
					CHKERRXX( VecDestroy( rlsPhi ) );

					// Destroy the p4est and its connectivity structure.
					p4est_nodes_destroy( nodes );
					p4est_ghost_destroy( ghost );
					p4est_destroy( p4est );
					my_p4est_brick_destroy( connectivity, &brick );
				}

				// Write to file samples collected for all sines with the same amplitude and same frequency but
				// randomized origin and for all rotations of main axis.
				for( const auto& row : rlsSamples )
				{
					std::copy( row.begin(), row.end() - 2,
							   std::ostream_iterator<double>( rlsFile, "," ) );		// Inner elements.
					rlsFile << std::setprecision( 8 ) << row[NUM_COLUMNS - 2] << "," << row.back()
							<< std::setprecision( 15 ) << std::endl;
				}

				// Same for signed distance function.
				if( writeSDF() )
				{
					for( const auto &row: sdfSamples )
					{
						std::copy( row.begin(), row.end() - 2,
								   std::ostream_iterator<double>( sdfFile, "," ));	// Inner elements.
						sdfFile << std::setprecision( 8 ) << row[NUM_COLUMNS - 2] << "," << row.back()
								<< std::setprecision( 15 ) << std::endl;
					}
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

		PetscPrintf( mpi.comm(), "<< Finished generating %i distinct amplitudes and %lu samples with max rel error %f in %f secs.\n",
					 na, nSamples, maxRelAbsError, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}