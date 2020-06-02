/**
 * Generate datasets for training a feedforward neural network on sinusoidal interfaces using samples from a
 * reinitialized level-set function with the fast sweeping method.
 * The level-set function is implemented as an arc-length parameterized sine wave function that is transformed with an
 * affine transformation to allow for pattern variations.  See arclength_parameterized_sin_2d.h for more details.
 *
 * To avoid a disproportionate ratio of h\kappa ~ 0 samples, we collect samples that are close to zero using a
 * probabilistic approach.  Seek the [SAMPLING] subsection in this file.
 *
 * Developer: Luis Ángel.
 * Date: May 28, 2020.
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
#include <iterator>
#include <fstream>
#include "arclength_parameterized_sine_2d.h"

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


/**
 * Generate the sample row of level-set function values and target h\kappa for a node that has been found next to the
 * sine wave interface.  We assume that this query node is effectively adjacent to \Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] sine The level-set function with a sinusoidal interface.
 * @param [out] distance Found normal distance from node to sine wave using Newton-Raphson's root-finding.
 * @return Vector with sampled phi values and target dimensionless curvature.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const double *phiReadPtr, const ArcLengthParameterizedSine& sine, double& distance )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h\kappa.

	int s;												// Index to fill in the sample vector.
	double leftPhi, rightPhi, topPhi, bottomPhi;		// To compute grad(\phi_{i,j}).
	double centerPhi;									// \phi{i,j}.
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample[s] = phiReadPtr[stencil[s]];

		switch( s )
		{
			case 5: topPhi = sample[s]; break;			// \phi_{i,j+1}.
			case 1: leftPhi = sample[s]; break;			// \phi_{i-1,j}.
			case 4: centerPhi = sample[s]; break;		// Point's phi value.
			case 7: rightPhi = sample[s]; break;		// \phi_{i+1,j}.
			case 3: bottomPhi = sample[s]; break;		// \phi_{i,j-1}.
			default: ;
		}
	}

	// Computing the target curvature.
	double xyz[P4EST_DIM];								// Position of node at the center of the stencil in world coords.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	double grad[] = {									// Gradient at center point.
		( rightPhi - leftPhi ) / ( 2 * H ),				// Using central differences.
		( topPhi - bottomPhi ) / ( 2 * H )
	};
	double gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );
	double pOnInterfaceX = xyz[0] - grad[0] / gradNorm * centerPhi,		// Coordinates of projection of grid point on
		   pOnInterfaceY = xyz[1] - grad[1] / gradNorm * centerPhi;		// interface in world coords.

   // Transform node position and approximated point on the interface to sine wave canonical coordinate system to
   // simplify computations with minimization process.
   sine.toCanonicalCoordinates( xyz[0], xyz[1] );
   sine.toCanonicalCoordinates( pOnInterfaceX, pOnInterfaceY );

	double u = pOnInterfaceX;							// Initial parameter guess for root finding method.
	double valOfDerivative;
	double newU = distThetaDerivative( nodeIdx, xyz[0], xyz[1], sine, valOfDerivative, u, u - H, u + H );

	pOnInterfaceX = newU;								// Recalculating point on interface (still in canonical coords).
	pOnInterfaceY = sine.getA() * sin( sine.getOmega() * newU );

	double dx = xyz[0] - pOnInterfaceX,					// Verify that point on interface is not far from grid point.
		   dy = xyz[1] - pOnInterfaceY;
	distance = sqrt( dx * dx + dy * dy );
	if( distance <= H && ABS( valOfDerivative ) <= 1e-8 )
		u = newU;										// Parameter is OK.
	else
	{
		pOnInterfaceX = u;
		pOnInterfaceY = sine.getA() * sin( sine.getOmega() * u );		// More acurately recover point on interface
		dx = xyz[0] - pOnInterfaceX;									// in canonical coordinates.
		dy = xyz[1] - pOnInterfaceY;
		distance = sqrt( dx * dx + dy * dy );
		sine.toWorldCoordinates( xyz[0], xyz[1] );
		sine.toWorldCoordinates( pOnInterfaceX, pOnInterfaceY );
		std::cerr << "Node " << nodeIdx << ": Minimization placed node on interface too far.  Reverting back to point "
				  << "on interface calculated with phi values.  \n     "
				  << distance << "; " << H * sine.curvature( u ) << "; "
				  << valOfDerivative << "; plot([" << xyz[0] << "], [" << xyz[1] << "], 'b.', ["
				  << pOnInterfaceX << "], [" << pOnInterfaceY << "], 'ko');" << std::endl;
	}

	sample[s] = H * sine.curvature( u );				// Last column holds h\kappa.
	if( centerPhi < 0 )									// Fix sign of found "exact" distance.
		distance *= -1;
//	std::cout << u << ", " << sample[s] << ";" << std::endl;

	return sample;
}


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -0.5, MAX_D = -MIN_D;								// The canonical space is [-1/2, +1/2]^2.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.
	const int NUM_AMPLITUDES = (int)pow( 2, MAX_REFINEMENT_LEVEL - 2 ) + 1;	// Number different sine wave amplitudes.

	const double MIN_A = 1.5 * H;				// An almost flat wave.
	const double MAX_A = HALF_D / 2;			// Tallest wave amplitude.
	const double MAX_HKAPPA_LB = 1.0 / 6.0;		// Lower and upper bounds for maximum h\kappa (used for discriminating
	const double MAX_HKAPPA_UB = 2.0 / 3.0;		// which samples to keep -- see below for details).
	const double MAX_HKAPPA_MIDPOINT = ( MAX_HKAPPA_LB + MAX_HKAPPA_UB ) / 2;

	const double HALF_AXIS_LEN = ( MAX_D - MIN_D ) * M_SQRT2 / 2 + 2 * H;	// Adding some padding of 2H to wave main axis.

	const double MIN_THETA = -M_PI_4;			// For each amplitude, we vary the rotation of the wave with respect
	const double MAX_THETA = +M_PI_4;			// to the horizontal axis from -pi/4 to +pi/4.
	const int NUM_THETAS = (int)pow( 2, MAX_REFINEMENT_LEVEL - 2 ) + 1;

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/fsm/data/";			// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 					// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );
	std::uniform_real_distribution<double> uniformDistribution;				// Used for collecting low h\kappa values.

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
		printf( ">> Began to generate datasets for %i distinct amplitudes with maximum refinement level of %i and finest h = %f\n",
			NUM_AMPLITUDES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples file: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream outputFile;
		std::string fileName = DATA_PATH + "sine_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
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

		// Variables to control the spread of sine waves' amplitudes, which must vary uniformly from MIN_A to MAX_A.
		double A_DIST = MAX_A - MIN_A;						// Amplitudes are in [1.5H, 0.5-2H], inclusive.
		double linspaceA[NUM_AMPLITUDES];
		for( int i = 0; i < NUM_AMPLITUDES; i++ )			// Uniform linear space from 0 to 1, with NUM_AMPLITUDES steps.
			linspaceA[i] = (double)( i ) / ( NUM_AMPLITUDES - 1.0 );

		// Variables to control the spread of ratation angles per amplitude, which must vary uniformly from MIN_THETA to
		// MAX_THETA, in a finite number of steps.
		const double THETA_DIST = MAX_THETA - MIN_THETA;	// As defined above, in [-pi/4, +pi/4].
		double linspaceTheta[NUM_THETAS];
		for( int i = 0; i < NUM_THETAS; i++ )				// Uniform linear space from 0 to 1, with NUM_TETHAS steps.
			linspaceTheta[i] = (double)( i ) / ( NUM_THETAS - 1.0 );

		// Domain information, applicable to all sinusoidal interfaces.
		int n_xyz[] = {1, 1, 1};							// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		// Printing header for log.
		std::cout << "Amplitude Idx, Omega Idx, Amplitude Val, Omega Val, Max Rel Error, Num Samples, Time" << std::endl;

		int nSamples = 0;
		int na = 0;
		for( ; na < NUM_AMPLITUDES; na++ )					// Go through all wave amplitudes evaluated.
		{
			const double A = MIN_A + linspaceA[na] * A_DIST;			// Amplitude to be evaluated.

			const double MIN_OMEGA = sqrt( MAX_HKAPPA_LB / ( H * A ) );	// Range of frequencies to ensure that the max
			const double MAX_OMEGA = sqrt( MAX_HKAPPA_UB / ( H * A ) );	// h\kappa is in the range of [1/6, 2/3].
			const double OMEGA_DIST = MAX_OMEGA - MIN_OMEGA;
			const double OMEGA_PEAK_DIST = M_PI_2 * ( 1 / MIN_OMEGA - 1 / MAX_OMEGA );	// Distance between u-values with highest peaks.
			const int NUM_OMEGAS = (int)ceil( OMEGA_PEAK_DIST / H ) + 1;				// Num. of omegas per amplitude.
			double linspaceOmega[NUM_OMEGAS];
			for( int i = 0; i < NUM_OMEGAS; i++ )			// Uniform linear space from 0 to 1, with NUM_OMEGA steps.
				linspaceOmega[i] = (double)( i ) / ( NUM_OMEGAS - 1.0 );

			for( int no = 0; no < NUM_OMEGAS; no++ )		// Evaluate all frequencies for the same amplitude.
			{
				std::vector<std::vector<double>> samples;
				double maxRE = 0;							// Maximum relative error for verification.

				const double OMEGA = MIN_OMEGA + linspaceOmega[no] * OMEGA_DIST;
				for( int nt = 0; nt < NUM_THETAS; nt++ )	// Various rotation angles for same amplitude and frequency.
				{
					const double THETA = MIN_THETA + linspaceTheta[nt] * THETA_DIST;	// Rotation of main sine axis.
					const double T[] = {
						( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen ),	// Translate origin coords by a random
						( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen )	// perturbation from grid's midpoint.
					};

					// p4est variables and data structures: these change with every sine wave because we must refine the
					// trees according to the new waves's origin and amplitude.
					p4est_t *p4est;
					p4est_nodes_t *nodes;
					my_p4est_brick_t brick;
					p4est_ghost_t *ghost;
					p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

					// Definining the level-set function to be reinitialized.
					ArcLengthParameterizedSine sine( A, OMEGA, T[0], T[1], THETA, H, HALF_AXIS_LEN );
					splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sine );

					// Create the forest using a level-set as refinement criterion.
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
					sample_cf_on_nodes( p4est, nodes, sine, phi );

					// Reinitialize level-set function using the fast sweeping method.
					FastSweeping fsm;
					fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
					fsm.reinitializeLevelSetFunction( &phi, 8 );
//					my_p4est_level_set_t ls( &nodeNeighbors );
//					ls.reinitialize_2nd_order( phi, 5 );

					// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface;
					// these are the points we'll use to create our sample files.
					std::vector<p4est_locidx_t> indices;
					NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
					nodesAlongInterface.getIndices( &phi, indices );

					// Getting the full uniform stencils of interface points.
					const double *phiReadPtr;
					ierr = VecGetArrayRead( phi, &phiReadPtr );
					CHKERRXX( ierr );

					// [SAMPLING] Now, collect samples with reinitialized level-set function values and target h\kappa.
					for( auto n : indices )
					{
						std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.

						try
						{
							if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
							{
								double distance;
								std::vector<double> data = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
									nodes, phiReadPtr, sine, distance );

								// Accumulating samples: we always take samples with h\kappa > midpoint; for those with
								// h\kappa <= midpoint, we take them with an easing-off-dropping probability from 1 to
								// 0.015, where Pr(h\kappa = midpoint) = 1 and Pr(h\kappa = 0) = 0.015.
								if( ABS( data[NUM_COLUMNS - 1] ) > MAX_HKAPPA_MIDPOINT ||
									uniformDistribution( gen ) <= 0.015 + ( sin( -M_PI_2 + ABS( data[NUM_COLUMNS - 1] )
									* M_PI / MAX_HKAPPA_MIDPOINT ) + 1 ) * 0.985 / 2  )
								{
									samples.push_back( data );

									// Error metric for validation.
									double error = ABS( distance - phiReadPtr[n] ) / H;
									maxRE = MAX( maxRE, error );
								}
							}
						}
						catch( std::exception &e )
						{
//							double xyz[P4EST_DIM];				// Position of node at the center of the stencil.
//							node_xyz_fr_n( n, p4est, nodes, xyz );
//							std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
						}
					}

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

				// Write to file samples collected for all sines with the same amplitude and same frequency but
				// randomized origin and for all rotations of main axis.
				for( const auto& row : samples )
				{
					std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
					outputFile << row.back() << std::endl;
				}

				nSamples += samples.size();

				// Output for log.
				std::cout << na + 1 << ", " << no + 1 << ", " << A << ", " << OMEGA << ", " << maxRE << ", "
						  << samples.size() << ", " << watch.get_duration_current() << ";" << std::endl;
			}
		}

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