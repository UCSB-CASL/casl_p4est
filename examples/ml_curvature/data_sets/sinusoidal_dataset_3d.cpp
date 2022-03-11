/**
 * Generating samples using a sinusoidal surface in 3D.  The idea is to vary the shape parameters in the sinusoidal
 * Monge patch given by
 *                                      Q(u,v) = A * sin(wu*u) * sin(wv*v),
 *
 * where A, wu, and wv specify the surface in its canonical space.  Also, by applying rotations and random translations,
 * we inject further pattern variations.  We classify samples into two types: fron non-saddle regions and from saddle
 * regions.  To classify samples into these two types, we use the Gaussian curvature linearly interpolated at the normal
 * projection onto Gamma.  Tests on Q(u,v) have shown that samples from non-saddle points are more well-behaved than
 * saddle samples.  In particular, non-saddle samples have both true mean hk and interpolated ihk with the same sign,
 * and, consequently, we can apply negative-mean-curvature normalization, as we did in 2d.  Furthermore, we can rely on
 * ihk to decide when to use or not the neural correction (with linear blending).  On the other hand, saddle samples are
 * not that consistent, and, for them, we do not apply negative normalization.  For these reasons, we need to train two
 * modes: one for saddle points and another for non-saddle (more reliable) regions.  To this end, this source code gene-
 * rates two separate files for saddle/non-saddle regions.  However, in all cases we do gradient-based normalization,
 * which entails reorienting the stencil so that nabla phi at the center point has all its components positive.  Simi-
 * larly, we perform sample augmentation by reflecting stencils about the x - y = 0 plane (which preserves mean and
 * Gaussian curvature).
 *
 * Negative-curvature normalization depends on the sign of the linearly interpolated mean ihk at the interface.
 * As for the Gaussian curvature, we normalize it by scaling it with h^2 ---which leads to the true h2kg and the linear-
 * ly interpolated ih2kg values.
 *
 * Files written are of the form "#/non_saddle_sinusoid_$.csv" and "#/saddle_sinusoid_$.csv", where # is the unit-cube
 * maximum level of refinement and $ is the sinusoidal amplitude index (i.e., 0, 1,... NUM_A-1).
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition
 * of mean curvature; that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: February 26, 2022.
 * Updated: March 10, 2022.
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
void saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES],
				  int bufferSize[SAMPLE_TYPES], std::ofstream file[SAMPLE_TYPES], double trackedMinHK[SAMPLE_TYPES],
				  double trackedMaxHK[SAMPLE_TYPES], const std::string fileName[SAMPLE_TYPES], const size_t& bufferMinSize,
				  const u_short& nHistBins, const float& histMedianFrac, const float& histMinFold, const bool& force=false );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>               minHK( pl, 0.005, "minHK"					, "Minimum mean dimensionless curvature for non-saddle points (default: 0.005)" );
	param_t<double>               maxHK( pl,  2./3, "maxHK"					, "Maximum mean dimensionless curvature (default: 2/3)" );
	param_t<u_char>               maxRL( pl,     6, "maxRL"					, "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<u_short>        reinitIters( pl,    10, "reinitIters"			, "Number of iterations for reinitialization (default: 10)" );
	param_t<double>    easeOffProbMaxHK( pl,   0.4, "easeOffProbMaxHK"		, "Easing-off probability for |hk*| upper bound for subsampling non-saddle points (default: 0.4)" );
	param_t<double>    easeOffProbMinHK( pl,  0.01, "easeOffProbMinHK"		, "Easing-off probability for |hk*| lower bound for subsampling non-saddle points (default: 0.01)" );
	param_t<double> easeOffProbMaxIH2KG( pl,   0.4, "easeOffProbMaxIH2KG"	, "Easing-off probability for |ih2kg| upper bound for subsampling saddle points (default: 0.4)" );
	param_t<double> easeOffProbMinIH2KG( pl,  0.01, "easeOffProbMinIH2KG"	, "Easing-off probability for |ih2kg| lower bound for subsampling saddle points (default: 0.01)" );
	param_t<u_short>          startAIdx( pl,     0, "startAIdx"				, "Start index for sinusoidal amplitude (default: 0)" );
	param_t<float>       histMedianFrac( pl,  1./3, "histMedianFrac"		, "Post-histogram subsampling median fraction for non-saddle points (default: 1/3)" );
	param_t<float>          histMinFold( pl,   1.5, "histMinFold"			, "Post-histogram subsampling min count fold for non-saddle points (default: 1.5)" );
	param_t<u_short>          nHistBins( pl,   100, "nHistBins"				, "Number of bins in histogram (default: 100)" );
	param_t<std::string>         outDir( pl,   ".", "outDir"				, "Path where files will be written to (default: build folder)" );
	param_t<size_t>       bufferMinSize( pl, 1.5e5, "bufferMinSize"			, "Buffer minimum size to trigger histogram-based subsampling for both saddle/non-saddle points (default: 150K)" );
	param_t<u_short>      numHKMaxSteps( pl,     7, "numHKMaxSteps" 		, "Number of steps to vary target max hk (default: 7)" );
	param_t<u_short>          numThetas( pl,     8, "numThetas"				, "Number of angular steps from -pi/2 to +pi/2 (inclusive) (default: 8)" );
	param_t<u_short>      numAmplitudes( pl,     9, "numAmplitudes"			, "Number of amplitude steps (default: 9)" );

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
		if( cmd.parse( argc, argv, "Generating three-dimensional sinusoidal data set with saddle/non-saddle points" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		CHKERRXX( PetscPrintf( mpi.comm(), "\n**************** Generating a sinusoidal data set in 3D ****************\n" ) );

		parStopWatch watch;
		watch.start();

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double h = 1. / (1 << maxRL());					// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / h;						// Target mean curvature bounds.
		const double MAX_K = maxHK() / h;
		const double MAX_A = 1 / MIN_K / 2;						// Amplitude bounds: MAX_A, which is half the max sphere radius.
		const double MIN_A = 5 / MAX_K;							// MIN_A = 5*(min radius).
		const double HK_MAX_LO = maxHK() / 2;					// Maximum HK bounds at the peaks.
		const double HK_MAX_UP = maxHK();

		// Affine transformation parameters.
		const int NUM_AXES = P4EST_DIM;
		const Point3 ROT_AXES[NUM_AXES] = {{1,0,0}, {0,1,0}, {0,0,1}};	// Use Euler angles; these are the rotation axes.
		const double MIN_THETA = -M_PI_2;								// For each axis, we vary the angle from -pi/2
		const double MAX_THETA = +M_PI_2;								// +pi/2, excluding the right-end point.
		std::uniform_real_distribution<double> uniformDistributionH_2( -h/2, +h/2 );	// Random translation.

		// Parameter validation.

		if( numAmplitudes() < 2 )
			throw std::invalid_argument( "[CASL_ERROR] There must be at least 2 amplitude steps!" );

		if( startAIdx() >= numAmplitudes() )
			throw std::invalid_argument( "[CASL_ERROR] Initial amplitude index is invalid!" );

		if( numHKMaxSteps() < 2 )
			throw std::invalid_argument( "[CASL_ERROR] There should be at least two steps for hk max!" );

		if( numThetas() < 2 )
			throw std::invalid_argument( "[CASL_ERROR] There should be at least two angular steps!" );

		if( easeOffProbMinHK() < 0 || easeOffProbMinHK() >= 1 ||
			easeOffProbMaxHK() <= 0 || easeOffProbMaxHK() > 1 ||
			easeOffProbMinHK() > easeOffProbMaxHK() )
			throw std::invalid_argument( "[CASL_ERROR] Invalid probabilities! We expect easeOffProbMinHK in [0, 1), "
										 "easeOffProbMaxHK in (0, 1], and easeOffProbMinHK < easeOffProbMaxHK." );

		if( easeOffProbMinIH2KG() < 0 || easeOffProbMinIH2KG() >= 1 ||
			easeOffProbMaxIH2KG() <= 0 || easeOffProbMaxIH2KG() > 1 ||
			easeOffProbMinIH2KG() > easeOffProbMaxIH2KG() )
			throw std::invalid_argument( "[CASL_ERROR] Invalid probabilities! We expect easeOffProbMinIH2KG in [0, 1), "
										 "easeOffProbMaxIH2KG in (0, 1], and easeOffProbMinIH2KG < easeOffProbMaxIH2KG." );

		PetscPrintf( mpi.comm(), ">> Began to generate dataset for %i distinct amplitudes, starting at A index %i, "
								 "with MaxRL = %i and h = %g\n", numAmplitudes(), startAIdx(), maxRL(), h );

		std::vector<double> linspaceA;						// Amplitude values.
		linspace( MIN_A, MAX_A, numAmplitudes(), linspaceA );

		std::vector<double> linspaceHK_MAX;					// HK_MAX values (i.e., the desired max hk at the peaks).
		linspace( HK_MAX_LO, HK_MAX_UP, numHKMaxSteps(), linspaceHK_MAX );

		std::vector<double> linspaceTheta;					// Angular values for each standard axis.
		linspace( MIN_THETA, MAX_THETA, numThetas(), linspaceTheta );

		///////////////////// Setting the limits for both triangulation and common physical domain /////////////////////

		const double SAM_RADIUS = MAX_A + 6 * h;			// Sampling radius (with enough padding) on the canonical uv plane.

		const double CUBE_SIDE_LEN = 2 * SAM_RADIUS;						// We want a cubic domain with an effective, yet small size.
		const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL() - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		const unsigned char OCTREE_MAX_RL = maxRL() - OCTREE_RL_FOR_LEN;	// Effective max refinement level to achieve desired h.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );				// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;				// Adjusted domain cube len as a multiple of h and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		const double D_CUBE_DIAG_LEN = sqrt( 3 ) * D_CUBE_SIDE_LEN;			// Use this diag to determine triangulated surface.
		const double UVLIM = D_CUBE_DIAG_LEN / 2 + h;						// Notice the padding to account for the random shift.
		const size_t halfUV = ceil( UVLIM / h );							// Half UV domain in h units.

		// Defining a symmetric cubic domain whose dimensions are multiples of h.
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};
		double xyz_min[] = {-HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN};
		double xyz_max[] = {+HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN};
		int periodic[] = {0, 0, 0};											// Non-periodic domain.

		///////////////////////////////////////////// Data-production loop /////////////////////////////////////////////

		const size_t TOT_ITERS = 3 * numHKMaxSteps() * (numHKMaxSteps() + 1) / 2;	// Num of axes times num of pairs of hk_max for each A.
		size_t step = 0;
		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES];	// Buffer of accumulated (normalized and augmented)
																			// samples for non-saddle (0) and saddle (1) points.
		if( mpi.rank() == 0 )												// Only rank 0 controls the buffers.
		{
			for( auto& b : buffer )
				b.reserve( bufferMinSize() );
		}
		int bufferSize[SAMPLE_TYPES] = {0, 0};			// Everyone knows the current buffer sizes to keep them in sync.
		double trackedMinHK[SAMPLE_TYPES] = {DBL_MAX, DBL_MAX};
		double trackedMaxHK[SAMPLE_TYPES] = {0, 0};		// We want to track min and max mean |hk*| (for saddle: 0, non-saddle: 1)
		SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );		// from processed samples until we save the buffered feature vectors.

		// Logging header.
		CHKERRXX( PetscPrintf( mpi.comm(), "Expecting %u iterations per amplitude and %d hk_max steps\n\n",
							   TOT_ITERS, numHKMaxSteps() ) );

		for( u_short a = startAIdx(); a < numAmplitudes(); a++ )	// For each amplitude, vary the u and v freqs to
		{															// achieve a desired maximum curvature at the peak.
			const double A = linspaceA[a];

			// Prepping the samples' files.  Notice we are no longer interested on exact-signed distance functions, only
			// reinitialized data.  Only rank 0 writes the samples to a file.
			const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
			std::ofstream file[SAMPLE_TYPES];
			std::string fileName[SAMPLE_TYPES] = {
				"non_saddle_sinusoid_" + std::to_string( a ) + ".csv",
				"saddle_sinusoid_" + std::to_string( a ) + ".csv"
			};

			for( int i = 0; i < SAMPLE_TYPES; i++ )
				kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName[i], file[i] );

			printLogHeader( mpi );
			size_t iters = 0;

			for( int s = 0; s < numHKMaxSteps(); s++ )		// Define a starting mean HK_MAX to define u-frequency wu.
			{
				const double START_MAX_K = linspaceHK_MAX[s] / h;	// Init mean max desired kappa; recall hk_max^up=2/3
				const double WU = sqrt( START_MAX_K / A );			// and hk_max^low=1/3.

				for( int t = s; t < numHKMaxSteps(); t++ )			// Then, define mean HK_MAX at the peak to find wv.
				{													// Intuitively, we start with circular peaks, and
					const double END_MAX_K = linspaceHK_MAX[t] / h;	// we transition to elliptical in uniform steps.
					const double WV = sqrt( 2 * END_MAX_K / A - SQR( WU ) );

					Sinusoid sinusoid( A, WU, WV );					// Sinusoid: Q(u,v) = A * sin(wu*u) * sin(wv*v).

					for( int axisIdx = 0; axisIdx < P4EST_DIM; axisIdx++ )	// Use Euler angles to rotate canonical coord system.
					{
						const Point3 ROT_AXIS = ROT_AXES[axisIdx];
						double maxHKError = 0, maxIH2KGError = 0;		// Tracking the maximum error and number of samples
						size_t loggedSamples[SAMPLE_TYPES] = {0, 0};	// collectively shared across processes for this rot axis.

						for( int nt = 0; nt < numThetas() - 1; nt++ )	// numThetas rotation angles for same axis (skip last one).
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
													maxRL(), &sinusoid, SQR( UVLIM ), SAM_RADIUS );

							// Macromesh variables and data structures.
							p4est_t *p4est;
							p4est_nodes_t *nodes;
							my_p4est_brick_t brick;
							p4est_ghost_t *ghost;
							p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

							//////////////////// Let's now discretize the domain and collect samples ///////////////////

							// Create the forest using the sinusoidal level-set as a refinement criterion.
							splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, OCTREE_MAX_RL, &sLS, 3.0 );
							p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
							p4est->user_pointer = (void *)( &splittingCriterion );

							// Refine and partition forest.
							sLS.toggleCache( true );	// Turn on cache to speed up repeated signed distance comput.
							sLS.reserveCache( (size_t)pow( 0.75 * HALF_D_CUBE_SIDE_LEN / h, 3 ) );
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
							assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == h );

							// A ghosted parallel PETSc vector to store level-set function values and a couple of flags.
							Vec phi = nullptr, exactFlag = nullptr, sampledFlag = nullptr;
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );

							// Populate phi and compute exact distance for vertices within a (linearly estimated) shell
							// around Gamma.  Reinitialization perturbs the otherwise calculated exact distances. exact-
							// Flag vector holds nodes status: only those with 1's can be used for sampling.
							sLS.evaluate( p4est, nodes, phi, exactFlag );

							// Reinitialize level-set function.
							my_p4est_level_set_t ls( ngbd );
							ls.reinitialize_2nd_order( phi, reinitIters() );

							std::vector<std::vector<double>> samples[SAMPLE_TYPES];
							std::pair<double, double> maxErrors;
							double minHKInBatch[2], maxHKInBatch[SAMPLE_TYPES];		// 0 for non-saddles, 1 for saddles.
							maxErrors = sLS.collectSamples( p4est, nodes, ngbd, phi, OCTREE_MAX_RL, xyz_min, xyz_max,
															minHKInBatch, maxHKInBatch, genProb,
															samples[0], HK_MAX_LO, easeOffProbMaxHK(), minHK(), easeOffProbMinHK(),	// Non-saddle params.
															samples[1], 5e-3, easeOffProbMaxIH2KG(), 0, easeOffProbMinIH2KG(),		// Saddle params.
															sampledFlag, NAN, exactFlag );

							maxHKError = MAX( maxHKError, maxErrors.first );
							maxIH2KGError = MAX( maxIH2KGError, maxErrors.second );
							for( int i = 0; i < SAMPLE_TYPES; i++ )
							{
								trackedMinHK[i] = MIN( minHKInBatch[i], trackedMinHK[i] );	// Update the tracked mean |hk*| bounds.
								trackedMaxHK[i] = MAX( maxHKInBatch[i], trackedMaxHK[i] );	// These are shared across processes.

								// Accumulate samples in the buffers; apply negative-mean-curvature normalization only to non-saddle samples.
								int batchSize = kml::utils::processSamplesAndAccumulate( mpi, samples[i], buffer[i], h, i == 0 );

								loggedSamples[i] += batchSize;
								bufferSize[i] += batchSize;
							}

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
											   step, A, s, h * START_MAX_K, t, h * END_MAX_K, axisIdx, maxHKError,
											   loggedSamples[0] + loggedSamples[1], (100.0 * iters / TOT_ITERS), watch.get_duration_current() ) );

						// Save samples if it's time.
						saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, fileName,
									 bufferMinSize(), nHistBins(), histMedianFrac(), histMinFold(), true );

						step++;
						return 0;	// TODO: Remove.
					}
				}
			}

			// Save any samples left in the buffer (by forcing the process) and start afresh for next A value.
			saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, fileName,
						 bufferMinSize(), nHistBins(), histMedianFrac(), histMinFold(), true );

			if( mpi.rank() == 0 )
			{
				for( auto& f : file )
					f.close();
			}

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

/**
 * Save samples in buffers if it's time or if the user forces the process (i.e., if corresponding buffer has overflowed
 * the user-defined min size or we have finished but there are samples left in the buffers).
 * Upon exiting, the buffer will be emptied and re-reserved, and the tracked min HK, max HK, and buffer size variables
 * will be reset if buffer was saved to a file.
 * @param [in] mpi MPI environment.
 * @param [in,out] buffer Sample buffers for non-saddle and saddle points.
 * @param [in,out] bufferSize Current buffers' size.
 * @param [in,out] file Files where to write samples.
 * @param [in,out] trackedMinHK Currently tracked minimum true |hk*| for non-saddles and saddles.
 * @param [in,out] trackedMaxHK Currently tracked maximum true |hk*| for non-saddles and saddles.
 * @param [in] fileName File names array.
 * @param [in] bufferMinSize Predefined minimum size to trigger file saving (same value for non-saddles and saddles).
 * @param [in] nHistBins Number of bins for histogram-based subsampling (applicable to non-saddles only).
 * @param [in] histMedianFrac Median scaling factor for histogram-based subsampling (applicable to non-saddles only).
 * @param [in] histMinFold Fold factor for minimum non-zero count in histogram-based subsampling (applicable to non-saddles only).
 * @param [in] force Set to true if we want to bypass the overflow condition (i.e., if we're done but there are samples left in the buffers).
 */
void saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES],
				  int bufferSize[SAMPLE_TYPES], std::ofstream file[SAMPLE_TYPES], double trackedMinHK[SAMPLE_TYPES],
				  double trackedMaxHK[SAMPLE_TYPES], const std::string fileName[SAMPLE_TYPES], const size_t& bufferMinSize,
				  const u_short& nHistBins, const float& histMedianFrac, const float& histMinFold, const bool& force )
{
	for( int i = 0; i < SAMPLE_TYPES; i++ )				// Do this for 0: non-saddle points and 1: saddle points.
	{
		if( force || bufferSize[i] >= bufferMinSize )	// Check if it's time to save samples.
		{
			int savedSamples;
			if( i == 0 )		// Do histogram-based subsampling only for non-saddle points.
			{
				savedSamples = kml::utils::histSubSamplingAndSaveToFile( mpi, buffer[i], file[i],
																		 (FDEEP_FLOAT_TYPE)trackedMinHK[i],
																		 (FDEEP_FLOAT_TYPE)trackedMaxHK[i],
																		 nHistBins, histMedianFrac, histMinFold );
			}
			else
			{
				savedSamples = kml::utils::saveSamplesBufferToFile( mpi, file[i], buffer[i] );
			}

			CHKERRXX( PetscPrintf( mpi.comm(),
								   "[*] Saved %d out of %d samples to output file %s, with |hk*| in the range of [%f, %f].\n",
								   savedSamples, bufferSize[i], fileName[i].c_str(), trackedMinHK[i], trackedMaxHK[i] ) );
			printLogHeader( mpi );

			buffer[i].clear();							// Reset control variables.
			if( mpi.rank() == 0 )
				buffer[i].reserve( bufferMinSize );
			trackedMinHK[i] = DBL_MAX;
			trackedMaxHK[i] = 0;
			bufferSize[i] = 0;

			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
		}
	}
}