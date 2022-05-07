/**
 * Generating samples using a sinusoidal surface in 3D.  The idea is to vary the shape parameters in the sinusoidal Monge path given by
 *                                      Q(u,v) = A * sin(wu*u) * sin(wv*v),
 *
 * where A, wu, and wv specify the surface in its canonical space.  Also, by applying rotations and random translations, we inject further
 * pattern variations.  We classify samples into two types: from non-saddle regions and from saddle regions, for a better analysis.  To
 * classify samples, we use the (dimensionless) Gaussian curvature linearly interpolated at the normal projection onto Gamma.  Tests on
 * Q(u,v) have shown that samples for which ih2kg > -7e-6 (i.e., the numerical estimation of h^2 times the Gaussian curvature at Gamma),
 * the mean curvature error increases as |ihk| -> infty (i.e., h times the mean curvature at Gamma).  On the other hand, saddle samples are
 * not that consistent, and, because of them, we can't always simplify the problem by normalizing to the negative mean curvature spectrum
 * (as in the 2d case).  However, we can still reorient all stencils so that the gradient at the center node has all its components non-
 * negative.  Similarly, we can perform sample augmentation by reflecting stencils about the x-y = 0 plane (which preserves mean and
 * Gaussian curvature).  Finally, histogram subsampling helps keep well-balanced data sets (regarding mean |hk*|) as much as possible.
 *
 * Files written are of the form "#/non_saddle_sinusoid_$.csv" and "#/saddle_sinusoid_$.csv", where # is the unit-cube max level of
 * refinement and $ is the sinusoidal amplitude index (i.e., 0, 1,... NUM_A-1, with NUM_A being the number of distinct amplitudes).
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.  To avoid confusion, we don't refer to the mean curvature as H --we
 * prefer hk for dimensionless mean curvature, and h2kg for dimensionless Gaussian curvature.
 *
 * Developer: Luis Ángel.
 * Created: February 26, 2022.
 * Updated: May 6, 2022.
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

bool saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES], int bufferSize[SAMPLE_TYPES],
				  std::ofstream file[SAMPLE_TYPES], double trackedMinHK[SAMPLE_TYPES], double trackedMaxHK[SAMPLE_TYPES],
				  const double& hkDist, const std::string fileName[SAMPLE_TYPES], const size_t& bufferMinSize, const u_short& nHistBins,
				  const float& histMedianFrac, const float& histMinFold, const bool& force );

void setupDomain( const Sinusoid& sinusoid, const double& N_WAVES, const double& h, const double& MAX_A, const u_char& MAX_RL,
				  double& samRadius, u_char& octreeMaxRL, double& uvLim, size_t& halfUV, int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM],
				  double xyz_max[P4EST_DIM] );

void uniformRandomSpace( const mpi_environment_t& mpi, const double& start, const double& end, const int& n, std::vector<double>& values,
						 std::mt19937& gen );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>   nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG"	 , "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																		   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>               minHK( pl, 0.004, "minHK"				 , "Min dimensionless mean curvature for numerical non-saddle "
																		   "samples (i.e., where ih2kg >= nonSaddleMinIH2KG) "
																		   "(default: 0.004)" );
	param_t<double>               maxHK( pl,  2./3, "maxHK"				 , "Max dimensionless mean curvature (default: 2/3)" );
	param_t<u_char>               maxRL( pl,     6, "maxRL"				 , "Max level of refinement per unit-cube octree (default: 6)" );
	param_t<u_short>        reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<double>    easeOffProbMaxHK( pl,  0.25, "easeOffProbMaxHK"	 , "Easing-off prob for |hk*| upper bound for subsampling numerical"
																		   " non-saddle samples (default: 0.25)" );
	param_t<double>    easeOffProbMinHK( pl, 0.005, "easeOffProbMinHK"	 , "Easing-off prob for |hk*| lower bound for subsampling numerical"
																			  "numerical non-saddle points (default: 0.005)" );
	param_t<double> easeOffProbMaxIH2KG( pl, 0.075, "easeOffProbMaxIH2KG", "Easing-off prob for |ih2kg| upper bound for subsampling saddle "
																		   "samples (default: 0.075)" );
	param_t<double> easeOffProbMinIH2KG( pl,0.0025, "easeOffProbMinIH2KG", "Easing-off prob for |ih2kg| lower bound for subsampling saddle "
																		   "samples (default: 0.0025)" );
	param_t<u_short>          startAIdx( pl,     0, "startAIdx"			 , "Start index for sinusoidal amplitude (default: 0)" );
	param_t<float>       histMedianFrac( pl,  1./3, "histMedianFrac"	 , "Post-histogram subsampling median fraction (default: 1/3)" );
	param_t<float>          histMinFold( pl,   1.5, "histMinFold"		 , "Post-histogram subsampling min count fold (default: 1.5)" );
	param_t<u_short>          nHistBins( pl,   100, "nHistBins"			 , "Number of bins in |hk*| histogram (default: 100)" );
	param_t<std::string>         outDir( pl,   ".", "outDir"			 , "Path where to write data files (default: build folder)" );
	param_t<size_t>       bufferMinSize( pl,   3e5, "bufferMinSize"		 , "Buffer minimum overflow size to trigger histogram-based "
																		   "subsampling and storage (default: 300K)" );
	param_t<u_short>      numHKMaxSteps( pl,     7, "numHKMaxSteps" 	 , "Number of steps to vary target max hk (default: 7)" );
	param_t<u_short>          numThetas( pl,    10, "numThetas"			 , "Number of angular steps from -pi/2 to +pi/2 (inclusive) "
																		   "(default: 10)" );
	param_t<u_short>      numAmplitudes( pl,    13, "numAmplitudes"		 , "Number of amplitude steps (default: 13)" );
	param_t<double>        numFullWaves( pl,   2.0, "numFullWaves"       , "How many full sinusoidal cycles to have inside the domain "
																		   "(default: 2.0)" );
	param_t<double>         randomNoise( pl,  1e-4, "randomNoise"		 , "How much uniform random noise to add to phi(x) as "
																		   "[+/-]h*randomNoise; use 0 or a negative value to disable "
																		   "(default: 1e-4)" );
	param_t<bool>        useNegCurvNorm( pl,  true, "useNegCurvNorm"	 , "Whether we want to apply negative-mean-curvature normalization "
																		   "for numerical non-saddle samples (default: true)" );

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

		CHKERRXX( PetscPrintf( mpi.comm(), "\n*********************** Generating a sinusoidal data set in 3D ***********************\n" ) );

		parStopWatch watch;
		watch.start();

		///////////////////////////////////////////////////////// Parameter setup //////////////////////////////////////////////////////////

		const double h = 1. / (1 << maxRL());					// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / h;						// Target mean curvature bounds.
		const double MAX_K = maxHK() / h;
		const double MAX_A = 1 / MIN_K / 2;						// Amplitude bounds: MAX_A is half the max sphere radius.
		const double MIN_A = 5 / MAX_K;							// MIN_A = 5*(min radius).
		const double HK_MAX_LO = maxHK() / 2;					// Maximum HK bounds at the peaks.
		const double HK_MAX_UP = maxHK();

		std::mt19937 genProb( mpi.rank() );	// Random engine for probability when choosing candidate nodes.
		std::mt19937 gen{};					// NOLINT Rank 0 uses this for random domain perturbations and spacing out amplitudes, hk_max values, and angles.

		// Affine transformation parameters.
		const double MIN_THETA = -M_PI_2;						// For each random basis vector, we vary the angle from -pi/2
		const double MAX_THETA = +M_PI_2;						// +pi/2, excluding the right-end point.
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
			throw std::invalid_argument( "[CASL_ERROR] Invalid probabilities! We expect easeOffProbMinHK in [0, 1), easeOffProbMaxHK in "
										 "(0, 1], and easeOffProbMinHK < easeOffProbMaxHK." );

		if( easeOffProbMinIH2KG() < 0 || easeOffProbMinIH2KG() >= 1 ||
			easeOffProbMaxIH2KG() <= 0 || easeOffProbMaxIH2KG() > 1 ||
			easeOffProbMinIH2KG() > easeOffProbMaxIH2KG() )
			throw std::invalid_argument( "[CASL_ERROR] Invalid probabilities! We expect easeOffProbMinIH2KG in [0, 1), easeOffProbMaxIH2KG "
										 "in (0, 1], and easeOffProbMinIH2KG < easeOffProbMaxIH2KG." );

		if( numFullWaves() < 1 )
			throw std::invalid_argument( "[CASL_ERROR] Choose at least one full cycle for sampling!" );

		if( randomNoise() < 0 || randomNoise() >= 1 )
			throw std::invalid_argument( "[CASL_ERROR] Uniform random noise factor can only be in the range [0, 1)." );

		PetscPrintf( mpi.comm(), ">> Began to generate dataset for %i amplitudes, starting at A index %i, with MaxRL = %i and h = %g\n",
					 numAmplitudes(), startAIdx(), maxRL(), h );

		std::vector<double> linspaceA;						// Random amplitude values from MIN_A to MAX_A.
		uniformRandomSpace( mpi, MIN_A, MAX_A, numAmplitudes(), linspaceA, gen );

		std::uniform_real_distribution<double> randomNoiseDist( -h * randomNoise(), +h * randomNoise() );
		std::mt19937 genNoise( mpi.rank() );				// A separate see for each rank: to be used only for noise, if requested.

		/////////////////////////////////////////////////////// Data-production loop ///////////////////////////////////////////////////////

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
		CHKERRXX( PetscPrintf( mpi.comm(), "Expecting %u iterations per amplitude and %d hk_max steps\n\n", TOT_ITERS, numHKMaxSteps() ) );

		for( u_short a = startAIdx(); a < numAmplitudes(); a++ )	// For each amplitude, vary the u and v freqs to achieve a desired max
		{															// curvature at the peak.
			const double A = linspaceA[a];

			// Prepping the samples' files.  Notice we are no longer interested on exact-signed distance functions, only reinitialized data.
			// Only rank 0 writes the samples to a file.
			const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
			std::ofstream file[SAMPLE_TYPES];
			std::string fileName[SAMPLE_TYPES] = {
				"non_saddle_sinusoid_" + std::to_string( a ) + ".csv",
				"saddle_sinusoid_" + std::to_string( a ) + ".csv"
			};

			for( int i = 0; i < SAMPLE_TYPES; i++ )
				kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName[i], file[i] );

			std::vector<double> linspaceHK_MAX;	// Random HK_MAX values for current A (i.e., the desired max hk at the peaks).
			uniformRandomSpace( mpi, HK_MAX_LO, HK_MAX_UP, numHKMaxSteps(), linspaceHK_MAX, gen );

			printLogHeader( mpi );
			size_t iters = 0;

			for( int s = 0; s < numHKMaxSteps(); s++ )		// Define a starting mean HK_MAX to define u-frequency wu.
			{
				const double START_MAX_K = linspaceHK_MAX[s] / h;	// Init mean max desired kappa; recall hk_max^up=2/3 and hk_max^low=1/3.
				const double WU = sqrt( START_MAX_K / A );

				for( int t = s; t < numHKMaxSteps(); t++ )				// Then, define mean HK_MAX at the peak to find wv.
				{														// Intuitively, we start with circular peaks and transition to
					const double END_MAX_K = linspaceHK_MAX[t] / h;		// elliptical in uniform steps.
					const double WV = sqrt( 2 * END_MAX_K / A - SQR( WU ) );

					Sinusoid sinusoid( A, WU, WV );						// Sinusoid: Q(u,v) = A * sin(wu*u) * sin(wv*v).

					///////////////////////////////// Setting up domain for current sinusoid configuration /////////////////////////////////

					double samRadius;									// Sampling radius on uv plane.
					u_char octMaxRL;									// Effective max ref lvl to achieve desired h.
					double uvLim;										// Limiting radius for triangulation.
					size_t halfUV;										// Half UV domain in h units.
					int n_xyz[P4EST_DIM];								// Number of trees in each direction and domain
					double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];		// min and max coords.
					int periodic[P4EST_DIM] = {0, 0, 0};				// Non-periodic domain.

					setupDomain( sinusoid, numFullWaves(), h, MAX_A, maxRL(), samRadius, octMaxRL, uvLim, halfUV, n_xyz, xyz_min, xyz_max );

					double rotAxes[P4EST_DIM][P4EST_DIM] = {{1,0,0}, {0,1,0}, {0,0,1}};					// Orthornal random basis vectors.
					if( mpi.rank() == 0 )
					{
						std::vector<Point3> basis;
						geom::buildRandomBasis( basis, gen );
						for( int i = 0; i < P4EST_DIM; i++ )
						{
							rotAxes[i][0] = basis[i].x;
							rotAxes[i][1] = basis[i].y;
							rotAxes[i][2] = basis[i].z;
						}
					}
					for( auto& rotAxis : rotAxes )
						SC_CHECK_MPI( MPI_Bcast( &rotAxis, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// Everyone uses same random basis.

					for( int axisIdx = 0; axisIdx < P4EST_DIM; axisIdx++ )	// Use random basis to rotate canonical coord system.
					{
						const Point3 ROT_AXIS( rotAxes[axisIdx] );
						double maxHKError = 0, maxIH2KGError = 0;		// Tracking the maximum error and number of samples collectively
						size_t loggedSamples[SAMPLE_TYPES] = {0, 0};	// shared across processes for this rot axis.

						std::vector<double> linspaceTheta;				// Random angular values for each random unit axis.
						uniformRandomSpace( mpi, MIN_THETA, MAX_THETA, numThetas(), linspaceTheta, gen );

						for( int nt = 0; nt < numThetas() - 1; nt++ )	// numThetas rotation angles for same axis (skipping last one).
						{
							//////////////////////////// Defining the transformed sinusoidal level-set function ////////////////////////////

							const double THETA = linspaceTheta[nt];
							double TRANS[P4EST_DIM];
							if( mpi.rank() == 0 )						// Only rank 0 determines the random shift and then broadcasts it.
							{
								for( auto& dim : TRANS )
									dim = uniformDistributionH_2( gen );
							}
							SC_CHECK_MPI( MPI_Bcast( TRANS, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// Everyone uses same random shift.

							// Create level-set and discretize the surface using a balltree to speed up queries during grid refinment.
							SinusoidalLevelSet sLS( &mpi, Point3(TRANS), ROT_AXIS, THETA, halfUV, halfUV, maxRL(), &sinusoid, SQR(uvLim), samRadius );

							// Macromesh variables and data structures.
							p4est_t *p4est;
							p4est_nodes_t *nodes;
							my_p4est_brick_t brick;
							p4est_ghost_t *ghost;
							p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

							////////////////////////////////// Discretize the domain and collect samples ///////////////////////////////////

							// Create the forest using the sinusoidal level-set as a refinement criterion.
							splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, octMaxRL, &sLS, 3.0 );
							p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
							p4est->user_pointer = (void *)( &splittingCriterion );

							// Refine and partition forest.
							sLS.toggleCache( true );			// Turn on cache to speed up repeated signed distance computations.
							sLS.reserveCache( (size_t)pow( 0.75 * xyz_max[0] / h, 3 ) );
							for( int i = 0; i < octMaxRL; i++ )
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

							// Populate phi and compute exact distance for vertices within a (linearly estimated) shell around Gamma.
							// Reinitialization perturbs the otherwise calculated exact distances.  The exactFlag vector holds nodes'
							// status: only those with 1's can be used for sampling.
							sLS.evaluate( p4est, nodes, phi, exactFlag );

							// Add random noise if requested.
							if( randomNoise() > 0 )
							{
								double *phiPtr;
								CHKERRXX( VecGetArray( phi, &phiPtr ) );
								foreach_node( n, nodes )
									phiPtr[n] += randomNoiseDist( genNoise );
								CHKERRXX( VecRestoreArray( phi, &phiPtr ) );
							}

							// Reinitialize level-set function.
							my_p4est_level_set_t ls( ngbd );
							ls.reinitialize_2nd_order( phi, reinitIters() );

							std::vector<std::vector<double>> samples[SAMPLE_TYPES];
							std::pair<double, double> maxErrors;
							double minHKInBatch[SAMPLE_TYPES], maxHKInBatch[SAMPLE_TYPES];	// 0 for non-saddles, 1 for saddles.
							maxErrors = sLS.collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max,
															minHKInBatch, maxHKInBatch, genProb, nonSaddleMinIH2KG(),
															samples[0], HK_MAX_LO, easeOffProbMaxHK(), minHK(), easeOffProbMinHK(),	// Non-saddle params.
															samples[1], 1e-2, easeOffProbMaxIH2KG(), 0, easeOffProbMinIH2KG(),		// Saddle params.
															sampledFlag, NAN, exactFlag );

							maxHKError = MAX( maxHKError, maxErrors.first );
							maxIH2KGError = MAX( maxIH2KGError, maxErrors.second );
							for( int i = 0; i < SAMPLE_TYPES; i++ )
							{
								trackedMinHK[i] = MIN( minHKInBatch[i], trackedMinHK[i] );	// Update the tracked mean |hk*| bounds.
								trackedMaxHK[i] = MAX( maxHKInBatch[i], trackedMaxHK[i] );	// These are shared across processes.

								// Accumulate samples in buffers; apply negative-mean-curvature normalization only if requested.
								int batchSize = kml::utils::processSamplesAndAccumulate( mpi, samples[i], buffer[i], h, useNegCurvNorm()? (i == 0? 1 : 0) : 0 );

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
											   loggedSamples[0]+loggedSamples[1], (100.0*iters/TOT_ITERS), watch.get_duration_current() ) );

						// Save samples if it's time.
						saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, ABS( maxHK() - minHK() ), fileName,
									 bufferMinSize(), nHistBins(), histMedianFrac(), histMinFold(), false );

						step++;
					}
				}
			}

			// Save any samples left in the buffers (by forcing the process) and start afresh for next A value.
			saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, ABS( maxHK() - minHK() ), fileName, bufferMinSize(),
						 nHistBins(), histMedianFrac(), histMinFold(), true );

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
 * Save samples in buffers if it's time or if the user forces the process (i.e., if corresponding buffer has overflowed the user-defined min
 * size or we have finished but there are samples left in the buffers).  Upon exiting, the buffer will be emptied and re-reserved, and the
 * tracked min HK, max HK, and buffer size variable will be reset if buffer was saved to a file.
 * @param [in] mpi MPI environment.
 * @param [in,out] buffer Sample buffers for non-saddle and saddle points.
 * @param [in,out] bufferSize Current buffers' size.
 * @param [in,out] file Files where to write samples.
 * @param [in,out] trackedMinHK Currently tracked minimum true |hk*| for non-saddles and saddles.
 * @param [in,out] trackedMaxHK Currently tracked maximum true |hk*| for non-saddles and saddles.
 * @param [in] hkDist Distance between min and max |hk*| one would expect (i.e., 100).
 * @param [in] fileName File names array.
 * @param [in] bufferMinSize Predefined minimum size to trigger file saving (same value for non-saddles and saddles).
 * @param [in] nHistBins Number of bins one would expect for histogram-based subsampling.
 * @param [in] histMedianFrac Median scaling factor for histogram-based subsampling.
 * @param [in] histMinFold Fold factor for minimum non-zero count in histogram-based subsampling.
 * @param [in] force Set it to true if you want to bypass the overflow condition (i.e., if there are samples left in the buffers).
 * @return true if wrote any type of samples, false otherwise.
 */
bool saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES], int bufferSize[SAMPLE_TYPES],
				  std::ofstream file[SAMPLE_TYPES], double trackedMinHK[SAMPLE_TYPES], double trackedMaxHK[SAMPLE_TYPES],
				  const double& hkDist, const std::string fileName[SAMPLE_TYPES], const size_t& bufferMinSize, const u_short& nHistBins,
				  const float& histMedianFrac, const float& histMinFold, const bool& force )
{
	bool wroteSamples = false;
	for( int i = 0; i < SAMPLE_TYPES; i++ )				// Do this for 0: non-saddle points and 1: saddle points.
	{
		if( bufferSize[i] > 0 && (force || bufferSize[i] >= bufferMinSize) )	// Check if it's time to save samples.
		{
			// Effective number of bins is proportional to the difference between tracked min and max mean |hk*|, but not less than 50 and more than nHistBins.
			const u_short nBins = MAX( (u_short)50, MIN( (u_short)ceil(nHistBins * (trackedMaxHK[i] - trackedMinHK[i]) / hkDist), nHistBins ) );
			int savedSamples = kml::utils::histSubSamplingAndSaveToFile( mpi, buffer[i], file[i],
																		 (FDEEP_FLOAT_TYPE) trackedMinHK[i],
																		 (FDEEP_FLOAT_TYPE) trackedMaxHK[i],
																		 nBins, histMedianFrac, histMinFold );

			CHKERRXX( PetscPrintf( mpi.comm(),
								   "[*] Saved %d out of %d samples to output file %s, with |hk*| in the range of [%f, %f] using %i bins.\n",
								   savedSamples, bufferSize[i], fileName[i].c_str(), trackedMinHK[i], trackedMaxHK[i], nBins ) );
			wroteSamples = true;

			buffer[i].clear();							// Reset control variables.
			if( mpi.rank() == 0 )
				buffer[i].reserve( bufferMinSize );
			trackedMinHK[i] = DBL_MAX;
			trackedMaxHK[i] = 0;
			bufferSize[i] = 0;

			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
		}
	}

	if( wroteSamples )
		printLogHeader( mpi );

	return wroteSamples;
}

/**
 * Space out values in the range [start, end] uniformly using a random distribution that includes the end points.
 * @note Compare this function with linspace.
 * @param [in] mpi MPI environment.
 * @param [in] start Initial value.
 * @param [in] end End value.
 * @param [in] n Number of values (including the end points).
 * @param [out] values Vector of values.
 * @param [in,out] gen Random generator.
 */
void uniformRandomSpace( const mpi_environment_t& mpi, const double& start, const double& end, const int& n, std::vector<double>& values,
						 std::mt19937& gen )
{
	if( n < 2 )
		throw std::invalid_argument( "uniformRandomSpace: n must be at least 2!" );

	if( start >= end )
		throw std::invalid_argument( "uniformRandomSpace: start must be strictly less than end!" );

	values.resize( n );
	if( mpi.rank() == 0 )
	{
		std::uniform_real_distribution<double> uniformDist( start, end );
		for( int i = 0; i < n; i++ )						// Uniform random dist in [start, end] with n steps to be
			values[i] = uniformDist( gen );					// shared among processes.
		values[0] = start;									// Make sure we include the end points.
		values[n - 1] = end;
		std::sort( values.begin(), values.end() );
	}
	SC_CHECK_MPI( MPI_Bcast( values.data(), n, MPI_DOUBLE, 0, mpi.comm() ) );
}

/**
 * Set up the domain based on sinusoid shape parameters to ensure a good portion of the periodic surface resides inside Omega.
 * @param [in] sinusoid Configured sinusoid function.
 * @param [in] N_WAVES Desired number of full cycles for any direction.
 * @param [in] h Mesh size.
 * @param [in] MAX_A Maximum amplitude.  It's used to avoid unecessarily big domains by limiting the sampling radius.
 * @param [in] MAX_RL Maximum level of refinement per unit octant (i.e., h = 2^{-MAX_RL}).
 * @param [out] samRadius Sampling radius on the uv plane that resides fully within the domain.
 * @param [out] octreeMaxRL Effective individual octree maximum level of refinement to achieve the desired h.
 * @param [out] uvLim Limiting radius for triangulating sinusoid.
 * @param [out] halfUV Number of h units (symmetrically) in the u and v direction to define the uv domain.
 * @param [out] n_xyz Number of octrees in each direction with maximum level of refinement octreeMaxRL.
 * @param [out] xyz_min Omega minimum dimensions.
 * @param [out] xyz_max Omega maximum dimensions.
 */
void setupDomain( const Sinusoid& sinusoid, const double& N_WAVES, const double& h, const double& MAX_A, const u_char& MAX_RL,
				  double& samRadius, u_char& octreeMaxRL, double& uvLim, size_t& halfUV, int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM],
				  double xyz_max[P4EST_DIM] )
{
	samRadius = N_WAVES * 2.0 * M_PI * MAX( 1/sinusoid.wu(), 1/sinusoid.wv() );	// Choose the sampling radius based on longer distance that contains N_WAVES full cycles.
	samRadius = MAX( samRadius, sinusoid.A() );					// Prevent the case of a very thin surface: we still want to sample the tips.
	samRadius = 6 * h + MIN( 1.5 * MAX_A, samRadius );			// Then, bound that radius with the largest amplitude.  Add enough padding (for uv plane).

	const double CUBE_SIDE_LEN = 2 * samRadius;					// We want a cubic domain with an effective, yet small size.
	const u_char OCTREE_RL_FOR_LEN = MAX( 0, MAX_RL - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
	const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
	octreeMaxRL = MAX_RL - OCTREE_RL_FOR_LEN;					// Effective max refinement level to achieve desired h.
	const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );		// Number of trees in each dimension.
	const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;		// Adjusted domain cube len as a multiple of h and octree len.
	const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

	const double D_CUBE_DIAG_LEN = sqrt( 3 ) * D_CUBE_SIDE_LEN;	// Use this diag to determine triangulated surface.
	uvLim = D_CUBE_DIAG_LEN / 2 + h;							// Notice the padding to account for the random shift in [-h/2,+h/2]^3.
	halfUV = ceil( uvLim / h );									// Half UV domain in h units.

	// Defining a symmetric cubic domain whose dimensions are multiples of h.
	for( int i = 0; i < P4EST_DIM; i++ )
	{
		n_xyz[i] = N_TREES;
		xyz_min[i] = -HALF_D_CUBE_SIDE_LEN;
		xyz_max[i] = +HALF_D_CUBE_SIDE_LEN;
	}
}