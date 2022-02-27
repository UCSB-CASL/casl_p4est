/**
 * Generating samples using a sinusoidal surface in 3D.  The idea is to vary the shape parameters in the sinusoidal
 * Monge patch given by
 *                                      Q(u,v) = A * sin(wu*u) * sin(wv*v),
 *
 * where A, wu, and wv specify the surface in its canonical space.  Also, by applying rotations and random translations,
 * we inject further pattern variations.  All collected samples are negative-curvature normalized, reoriented, and refl-
 * ection-based augmented so that the (possibly negated) gradient at the stencil's center node has all its components
 * non-negative.
 *
 * @note Negative-curvature normalization depends on the sign of the linearly interpolated hk at the interface.  In the
 * case of saddle points, this numerical estimation is often wrong.  For this reason, we must discard samples with posi-
 * tive true hk* as we want the neural network to only understand the negative spectrum.
 *
 * Files written are of the form "#/sinusoid_$.csv", where # is the unit-cube maximum level of refinement and $ is the
 * sinusoidal amplitude index (i.e., 0, 1,... NUM_A-1).
 *
 * Developer: Luis Ángel.
 * Created: February 26, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "sinusoidal_3d.h"
#include <src/parameter_list.h>


void printLogHeader( const mpi_environment_t& mpi );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>         minHK( pl,  0.01, "minHK"			, "Minimum mean dimensionless curvature (default: 0.01 = twice 0.005 from 2D)" );
	param_t<double>         maxHK( pl,  4./3, "maxHK"			, "Maximum mean dimensionless curvature (default: 4/3 = twice 2/3 from 2D)" );
	param_t<u_char>         maxRL( pl,     6, "maxRL"			, "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<int>      reinitIters( pl,    10, "reinitIters"		, "Number of iterations for reinitialization (default: 10)" );
	param_t<double>  probMidMaxHK( pl,   1.0, "probMidMaxHK"	, "Easing-off max probability for mean max HK (default: 1.0)" );
	param_t<u_short>    startAIdx( pl,     0, "startAIdx"		, "Start index for sinusoidal amplitude (default: 0)" );
	param_t<float> histMedianFrac( pl,   0.3, "histMedianFrac"	, "Histogram subsampling median fraction (default: 0.3)" );
	param_t<float>    histMinFold( pl,   2.0, "histMinFold"		, "Histogram subsampling min count fold (default: 2.0)" );
	param_t<u_short>    nHistBins( pl,   100, "nHistBins"		, "Number of bins in histogram (default: 100)" );
	param_t<std::string>   outDir( pl,   ".", "outDir"			, "Path where files will be written to (default: build folder)" );
	param_t<size_t> bufferMinSize( pl, 1.5e5, "bufferMinSize"	, "Buffer minimum size to trigger histogram-based subsampling (default: 150,000" );

	std::mt19937 genProb{};		// NOLINT Random engine for probability when choosing candidate nodes.
	std::mt19937 genTrans{};	// NOLINT This engine is used for the random shift of the sinusoid's canonical frame.

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating three-dimensional sinusoidal data set" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		CHKERRXX( PetscPrintf( mpi.comm(), "\n**************** Generating a sinusoidal data set in 3D ****************\n" ) );

		parStopWatch watch;
		watch.start();

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double H = 1. / (1 << maxRL());					// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / H;						// Curvature bounds.
		const double MAX_K = maxHK() / H;
		const double MAX_A = 2 / MIN_K / 2;						// Amplitude bounds: MAX_A, which is half the max sphere radius.
		const double MIN_A = 10 / MAX_K;						// MIN_A = 5*(min radius).
		const auto NUM_A = (u_short)((MAX_A - MIN_A) / H / 7);	// Number of distinct amplitudes.
		const double HK_MAX_LO = maxHK() / 2;					// Maximum HK bounds at the peaks.
		const double HK_MAX_UP = maxHK();
		const double MID_HK_MAX = (HK_MAX_LO + HK_MAX_UP) / 2;
		const int NUM_HK_MAX = (int)ceil( (HK_MAX_UP - HK_MAX_LO) / (3 * H) );

		// Affine transformation parameters.
		const int NUM_AXES = P4EST_DIM;
		const Point3 ROT_AXES[NUM_AXES] = {{1,0,0}, {0,1,0}, {0,0,1}};	// Let's use Euler angles; the rotation axes.
		const double MIN_THETA = -M_PI_2;								// For each axis, we vary the angle from -pi/2
		const double MAX_THETA = +M_PI_2;								// +pi/2 without the end point.
		const int NUM_THETAS = 38;
		std::uniform_real_distribution<double> uniformDistributionH_2( -H/2, +H/2 );	// Random translation.

		PetscPrintf( mpi.comm(), ">> Began to generate dataset for %i distinct amplitudes, starting at A index %i, "
								 "with MaxRL=%i and H=%g\n", NUM_A, startAIdx(), maxRL(), H );

		std::vector<double> linspaceA;						// Amplitude values.
		linspace( MIN_A, MAX_A, NUM_A, linspaceA );

		std::vector<double> linspaceHK_MAX;					// HK_MAX values (i.e., the desired max hk at the peaks).
		linspace( HK_MAX_LO, HK_MAX_UP, NUM_HK_MAX, linspaceHK_MAX );

		std::vector<double> linspaceTheta;					// Angular values for each standard axis.
		linspace( MIN_THETA, MAX_THETA, NUM_THETAS, linspaceTheta );

		if( startAIdx() >= NUM_A )
			throw std::invalid_argument( "[CASL_ERROR] Initial amplitude index is invalid!" );

		///////////////////// Setting the limits for both triangulation and common physical domain /////////////////////

		const double SAM_RADIUS = MAX_A + 6 * H;			// Sampling radius (with enough padding) on the canonical uv plane.

		const double CUBE_SIDE_LEN = 2 * SAM_RADIUS;						// We want a cubic domain with an effective, yet small size.
		const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL() - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		const unsigned char OCTREE_MAX_RL = maxRL() - OCTREE_RL_FOR_LEN;	// Effective max refinement level to achieve desired H.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );				// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;				// Adjusted domain cube len as a multiple of H and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		const double D_CUBE_DIAG_LEN = sqrt( 3 ) * D_CUBE_SIDE_LEN;			// Use this diag to determine triangulated surface.
		const double UVLIM = D_CUBE_DIAG_LEN / 2 + H;						// Notice the padding to account for the random shift.
		const size_t halfUV = ceil( UVLIM / H );							// Half UV domain in H units.

		// Defining a symmetric cubic domain whose dimensions are multiples of H.
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};
		double xyz_min[] = {-HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN};
		double xyz_max[] = {+HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN};
		int periodic[] = {0, 0, 0};											// Non-periodic domain.

		///////////////////////////////////////////// Data-production loop /////////////////////////////////////////////

		const size_t TOT_ITERS = 3 * NUM_HK_MAX * (NUM_HK_MAX + 1) / 2;		// Num of axes times num of pairs of hk_max for each A.
		size_t step = 0;
		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Buffer of accumulated (normalized and augmented) samples.
		if( mpi.rank() == 0 )								// Only rank 0 controls the buffer.
			buffer.reserve( bufferMinSize() );
		int bufferSize = 0;									// But everyone knows the current buffer size to keep them in sync.
		double trackedMinHK = DBL_MAX, trackedMaxHK = 0;	// We want to track the min and max |hk*| from processed
		SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );			// samples until we save the buffered feature vectors.

		// Logging header.
		CHKERRXX( PetscPrintf( mpi.comm(), "Expecting %u iterations per amplitude and %d hk_max steps\n\n",
							   TOT_ITERS, NUM_HK_MAX ) );

		for( u_short a = startAIdx(); a < NUM_A; a++ )		// For each amplitude, vary the u and v freqs to achieve a
		{													// desired maximum curvature at the peak.
			const double A = linspaceA[a];

			// Preping the samples' file.  Notice we are no longer interested on exact-signed distance functions, only
			// reinitialized data.  Only rank 0 writes the samples to a file.
			const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
			std::ofstream file;
			std::string fileName = "sinusoid_" + std::to_string( a ) + ".csv";
			kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

			printLogHeader( mpi );
			size_t iters = 0;

			for( int s = 0; s < NUM_HK_MAX; s++ )			// Define a starting HK_MAX to define u-frequency: wu.
			{
				const double START_MAX_K = linspaceHK_MAX[s] / H;	// Init max desired kappa; recall hk_max^up = 4/3
				const double WU = sqrt( START_MAX_K / (2 * A) );	// and hk_max^low = 2/3 (2/3 and 1/3 in 2D).

				for( int t = s; t < NUM_HK_MAX; t++ )				// Then, define HK_MAX at the peak to find wv.
				{													// Intuitively, we start with circular peaks, and
					const double END_MAX_K = linspaceHK_MAX[t] / H;	// we transition to elliptical in uniform steps.
					const double WV = sqrt( END_MAX_K / A - SQR( WU ) );

					Sinusoid sinusoid( A, WU, WV );					// Sinusoidal Monge patch: Q(u,v) = A * sin(wu*u) * sin(wv*v).

					for( int axisIdx = 0; axisIdx < P4EST_DIM; axisIdx++ )	// Use Euler angles to rotate canonical coord system.
					{
						const Point3 ROT_AXIS = ROT_AXES[axisIdx];
						double maxHKError = 0;							// Tracking the maximum error and number of samples
						size_t loggedSamples = 0;						// collectively shared across processes for this rot axis.

						for( int nt = 0; nt < NUM_THETAS - 1; nt++ )	// Various rotation angles for same axis (skip last one).
						{
							////////////////// Defining the transformed sinusoidal level-set function //////////////////

							const double THETA = linspaceTheta[nt];
							double TRANS[P4EST_DIM];
							if( mpi.rank() == 0 )						// Only rank 0 determines the random shift and
							{											// then broadcasts it.
								for( auto& dim : TRANS )
									dim = uniformDistributionH_2( genTrans );
							}
							SC_CHECK_MPI( MPI_Bcast( TRANS, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same shift.

							// Also discretizes the surface using a balltree to speed up queries during grid refinment.
							SinusoidalLevelSet sLS( &mpi, Point3( TRANS ), ROT_AXIS.normalize(), THETA, halfUV, halfUV,
													maxRL(), &sinusoid, SQR( UVLIM ) );

							// Macromesh variables and data structures.
							p4est_t *p4est;
							p4est_nodes_t *nodes;
							my_p4est_brick_t brick;
							p4est_ghost_t *ghost;
							p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

							//////////////////// Let's now discretize the domain and collect samples ///////////////////

							// Create the forest using the sinusoidal level-set as a refinement criterion.
							splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, OCTREE_MAX_RL, &sLS, 2.0 );
							p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
							p4est->user_pointer = (void *)( &splittingCriterion );

							// Refine and partition forest.
							sLS.toggleCache( true );	// Turn on cache to speed up repeated signed distance comput.
							sLS.reserveCache( (size_t)pow( 0.75 * HALF_D_CUBE_SIDE_LEN / H, 3 ) );
							for( int i = 0; i < OCTREE_MAX_RL; i++ )
							{
								my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
								my_p4est_partition( p4est, P4EST_FALSE, nullptr );
							}

							// Create the ghost (cell) and node structures.
							ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
							nodes = my_p4est_nodes_new( p4est, ghost );

							// Initialize the neighbor nodes structure.
							auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
							auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
							ngbd->init_neighbors();

							// Verify mesh size.
							double dxyz[P4EST_DIM];
							get_dxyz_min( p4est, dxyz );
							assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == H );

							// A ghosted parallel PETSc vector to store level-set function values and a couple of flags.
							Vec phi = nullptr, exactFlag = nullptr, sampledFlag = nullptr;
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );

							// Populate phi and compute exact distance for vertices within a (linearly estimated) shell
							// around Gamma.  Reinitialization perturbs the otherwise calculated exact distances. exact-
							// Flag vector holds nodes status: only those with 1's can be used when sampling.
							sLS.evaluate( p4est, nodes, phi, exactFlag );

							// Reinitialize level-set function.
							my_p4est_level_set_t ls( ngbd );
							ls.reinitialize_2nd_order( phi, reinitIters() );

							std::vector<std::vector<double>> samples;
							double minHKInBatch, maxHKInBatch;
							double hkError = sLS.collectSamples( p4est, nodes, ngbd, phi, OCTREE_MAX_RL, xyz_min,
																 xyz_max, samples, minHKInBatch, maxHKInBatch, genProb,
																 MID_HK_MAX, probMidMaxHK(), minHK(), 0.01, sampledFlag,
																 SAM_RADIUS, exactFlag );

							maxHKError = MAX( maxHKError, hkError );
							trackedMinHK = MIN( minHKInBatch, trackedMinHK );	// Update the tracked |hk*| bounds.
							trackedMaxHK = MAX( maxHKInBatch, trackedMaxHK );	// These are shared across processes.

							// Accumulate samples in the buffer.
							int batchSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, H );
							loggedSamples += batchSize;
							bufferSize += batchSize;

							sLS.toggleCache( false );		// Done with cache.
							sLS.clearCache();

							CHKERRXX( VecDestroy( sampledFlag ) );
							CHKERRXX( VecDestroy( exactFlag ) );
							CHKERRXX( VecDestroy( phi ) );

							// Destroy the p4est and its connectivity structure.
							delete ngbd;
							delete hierarchy;
							p4est_nodes_destroy( nodes );
							p4est_ghost_destroy( ghost );
							p4est_destroy( p4est );
							my_p4est_brick_destroy( connectivity, &brick );

							// Synchronize.
							SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
						}

						// Logging stats.
						iters++;
						CHKERRXX( PetscPrintf( mpi.comm(), "[%6d] \t%.8f \t(%2d) %.8f \t(%2d) %.8f \t%i \t%.8f \t\t%u \t\t(%7.3f%%) \t%g\n",
											   step, A, s, H * START_MAX_K, t, H * END_MAX_K, axisIdx, maxHKError,
											   loggedSamples, (100.0 * iters / TOT_ITERS), watch.get_duration_current() ) );

						// Is it time to save to file?
						if( bufferSize >= bufferMinSize() )
						{
							int savedSamples = kml::utils::histSubSamplingAndSaveToFile( mpi, buffer, file,
																						 (FDEEP_FLOAT_TYPE)minHK(), (FDEEP_FLOAT_TYPE)maxHK(),
																						 nHistBins(), histMedianFrac(), histMinFold() );
							CHKERRXX( PetscPrintf( mpi.comm(), "[*] Saved %d out of %d samples to output file %s, with |hk*| in the range of [%f, %f].\n",
												   savedSamples, bufferSize, fileName.c_str(), trackedMinHK, trackedMaxHK ) );
							printLogHeader( mpi );

							buffer.clear();							// Reset control variables.
							if( mpi.rank() == 0 )
								buffer.reserve( bufferMinSize() );
							trackedMinHK = DBL_MAX;
							trackedMaxHK = 0;
							bufferSize = 0;
							SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
						}

						step++;
					}
				}
			}

			// Save any samples left in the buffer.
			if( bufferSize > 0 )
			{
				int savedSamples = kml::utils::histSubSamplingAndSaveToFile( mpi, buffer, file,
																			 (FDEEP_FLOAT_TYPE)minHK(), (FDEEP_FLOAT_TYPE)maxHK(),
																			 nHistBins(), histMedianFrac(), histMinFold() );
				CHKERRXX( PetscPrintf( mpi.comm(), "[*] Saved %d out of %d samples to output file %s, with |hk*| in the range of [%f, %f].\n",
									   savedSamples, bufferSize, fileName.c_str(), trackedMinHK, trackedMaxHK ) );
				buffer.clear();
			}

			if( mpi.rank() == 0 )
				file.close();

			CHKERRXX( PetscPrintf( mpi.comm(), "<<< Done with A = %f, index %d\n", A, a ) );
			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
		}

		watch.read_duration_current( true );
		watch.stop();

	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}

/**
 * Print the log header.
 * @param [in] mpi MPI environment.
 */
void printLogHeader( const mpi_environment_t& mpi )
{
	CHKERRXX( PetscPrintf( mpi.comm(), "_____________________________________________________________________________________________________________________\n") );
	CHKERRXX( PetscPrintf( mpi.comm(), "[Step  ] \tAmplitude \t(ss) Start_MaxHK \t(tt) End_MaxHK \tAxis \tMaxHK_Error \tNum_Samples\t(%%_Done) \tTime\n" ) );
}