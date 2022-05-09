/**
 * Generate samples from a hyperbolic paraboloid surfaces for training mainly on saddle sampling.  The canonical surface is the Monge patch
 *
 *                                                    Q(u,v) = a*u^2 - b*v^2,
 *
 * where a > 0 and b > 0.  As for the level-set function whose Gamma = Q(u,v) = 0, phi < 0 for points above Q(u,v), and phi > 0 for points
 * below Q(u,v).  We simplify calculations by expressing query points in terms of the canonical frame, which can be affected by rigid-body
 * transformations (i.e., a translation and rotation).
 *
 * Theoretically, the hyperbolic paraboloid Gaussian curvature is always negative (never 0).  Thus, its data set contains samples only
 * saddle regions (i.e., h2kg < 0).  If requested, we can apply negative-mean-curvature normalization selectively for each numerical
 * non-saddle sample (say we found some point for which ih2kg >= nonSaddleMinIH2KG).  In any case, every sample is reoriented by rotating
 * the stencil so that the gradient at the center node has all its components non-negative.  Finally, we reflect the data packet about the
 * y-x = 0 plane, and we produce two samples for each interface point.  At inference time, both outputs are averaged to improve accuracy.
 *
 * To inject variations, we vary the ratio between a and b.  The ratio a/b or b/a is r.  We change r between 1 and 10 and are careful to
 * avoid hitting an r that yields hk maxima whose critical points are less than 1.5h apart; we consider those values for r as under-resolved
 * for computing a reliable cuvature.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: May 8, 2022.
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
#include "hyp_paraboloid_3d.h"
#include <src/parameter_list.h>
#include <cassert>


void randomizeRatios( const mpi_environment_t& mpi, const double& start, const double& end, const int& n, std::vector<double>& ratios,
					  std::mt19937& gen );
void printLogHeader( const mpi_environment_t& mpi );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>             minHK( pl, 0.004, "minHK"			 , "Min dimensionless mean curvature for numerical non-saddle samples "
																	   "(i.e., where ih2kg >= nonSaddleMinIH2KG) (default: 0.004)" );
	param_t<double>           maxHKLB( pl, 1./15, "maxHKLB"			 , "Max abs dimensionless mean curvature lower bound (default: 1/15)" );
	param_t<double>             maxHK( pl,  2./3, "maxHK"			 , "Max abs dimensionless mean curvature (default: 2/3)" );
	param_t<u_char>             maxRL( pl,     6, "maxRL"		 	 , "Max level of refinement per unit-cube octree (default: 6)" );
	param_t<int>          reinitIters( pl,    10, "reinitIters"	 	 , "Number of iterations for reinitialization (default: 10)" );
	param_t<u_short>    minSamRadiusH( pl,    16, "minSamRadiusH"	 , "Min sampling radius in h units on the uv plane.  Must be in the "
																	   "range of [16, 32] (default 16)" );
	param_t<double>        minABRatio( pl,     1, "minABRatio"		 , "Min a/b or b/a |ratio| so that |ratio| in (minABRatio, maxABRatio) "
																	   "(default: 1)" );
	param_t<double>        maxABRatio( pl,    10, "maxABRatio"		 , "Max a/b or b/a |ratio| so that |ratio| in (minABRatio, maxABRatio) "
																	   "(default: 10)" );
	param_t<u_int>        randomState( pl,    11, "randomState"	 	 , "Seed for random perturbations of canonical frame (default: 11)" );
	param_t<std::string>       outDir( pl,   ".", "outDir"		 	 , "Path where files will be written to (default: build folder)" );
	param_t<size_t>     bufferMinSize( pl,   3e5, "bufferMinSize"	 , "Buffer minimum overflow size to trigger histogram-based "
																	   "subsampling and storage (default: 300K)" );
	param_t<float>     histMedianFrac( pl,  1./3, "histMedianFrac"	 , "Post-histogram subsampling median fraction (default: 1/3)" );
	param_t<float>        histMinFold( pl,   1.5, "histMinFold"		 , "Post-histogram subsampling min count fold (default: 1.5)" );
	param_t<u_short>        nHistBins( pl,   100, "nHistBins"		 , "Max number of bins in |hk*| histogram (default: 100)" );
	param_t<u_short>    numMaxHKSteps( pl,     7, "numMaxHKSteps" 	 , "Number of steps to vary target max |hk| (default: 7)" );
	param_t<u_short>      numABRatios( pl,    10, "numABRatios"		 , "Number of random a/b (for r>0) or b/a (for r<0) ratios in [1, 10] "
																	   "or [-10, -1) for each max |hk| value (default: 10)" );
	param_t<u_short>     numRotations( pl,    10, "numRotations"	 , "Number of rotations around random axes and by random angles for each"
																	   "ratio (default: 10)" );
	param_t<bool>      useNegCurvNorm( pl,  true, "useNegCurvNorm"	 , "Whether to apply negative-mean-curvature normalization for non-"
																	   "saddle samples (default: true)" );
	param_t<double>       randomNoise( pl,  1e-4, "randomNoise"		 , "How much random noise to add to phi(x) as [+/-]h*randomNoise.  Use "
																	   "a negative value or 0 to disable this feature (default: 1e-4)" );

	try
	{
		///////////////////////////////////////////////////////// Parameter setup //////////////////////////////////////////////////////////

		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating hyperbolic paraboloid data set for training a error-correcting neural network" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		if( !(0 < maxHKLB() && maxHKLB() < maxHK()) )
			throw std::invalid_argument( "[CASL_ERROR] Note that 0 < maxHKLB < maxHK." );

		if( minABRatio() < 1 || minABRatio() >= maxABRatio() )
			throw std::invalid_argument( "[CASL_ERROR] The a/b or b/a minABRatio must be at least 1 and no larger than maxABRatio." );

		if( minSamRadiusH() < 16 || minSamRadiusH() > 32 )
			throw std::invalid_argument( "[CASL_ERROR] Desired sampling radius in h units must be in the range of [16, 32]." );

		if( numMaxHKSteps() < 2 )
			throw std::invalid_argument( "[CASL_ERROR] You must ask for at least two different values of max |hk|." );

		if( numABRatios() < 2 )
			throw std::invalid_argument( "[CASL_ERROR] You must ask for at least two different ratios a/b or b/a." );

		if( numRotations() < 1 )
			throw std::invalid_argument( "[CASL_ERROR] We expect at least one random rotation." );

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.

		std::mt19937 gen( randomState() );					// Engine used for random perturbations and spacing out max |hk|, etc.
		std::uniform_real_distribution<double> uniformDistH_2( -h/2, +h/2 );
		std::uniform_real_distribution<double> uniformDist;

		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );

		///////////////////////////////////////////////////////// Data production //////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began generating hyperbolic paraboloid data set for max |hk| in [%.6f, %.6f], and h = %g (level %i)\n",
					 maxHKLB(), maxHK(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Prepping the samples' files.  Notice we are no longer interested on exact-signed distance functions, only reinitialized data.
		// Only rank 0 writes the samples to a file.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
		std::ofstream file[SAMPLE_TYPES];
		std::string fileName[SAMPLE_TYPES] = {"non_saddle_hyp_paraboloid.csv", "saddle_hyp_paraboloid.csv"};
		for( int i = 0; i < SAMPLE_TYPES; i++ )
			kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName[i], file[i] );

		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer[SAMPLE_TYPES];	// Buffer of accumulated (normalized and augmented) samples for
																			// non-saddle (0) and saddle (1) points.
		if( mpi.rank() == 0 )								// Only rank 0 controls the buffers.
		{
			for( auto& b : buffer )
				b.reserve( bufferMinSize() );
		}
		int bufferSize[SAMPLE_TYPES] = {0, 0};				// Everyone knows the current buffer sizes to keep them in sync.
		double trackedMinHK[SAMPLE_TYPES] = {DBL_MAX, DBL_MAX};
		double trackedMaxHK[SAMPLE_TYPES] = {0, 0};			// We want to track min and max mean |hk*| (for saddle: 0, non-saddle: 1) from
		SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );			// processed samples until we save the buffered feature vectors.

		std::vector<double> linspaceMaxHK;					// Random target max |hk| values from maxHKLB to maxHK, inclusive.
		kml::utils::uniformRandomSpace( mpi, maxHKLB(), maxHK(), numMaxHKSteps(), linspaceMaxHK, gen );

		printLogHeader( mpi );

		int hkIdx = -1;
		for( const auto& maxHKVal : linspaceMaxHK )
		{
			hkIdx++;
			std::vector<double> ratios;
			randomizeRatios( mpi, minABRatio(), maxABRatio(), numABRatios(), ratios, gen );

			int rIdx = -1;
			for( const auto& ratio : ratios )				// Process every ratio for current max |hk|
			{
				const double MAX_K = maxHKVal / h;
				double A, B, samRadius;
				rIdx++;

				try 										// Not all |ratios| between 1 and 3 are possible because the critical points may
				{											// be too close (at a distance less than 1.5h on the uv plane).  Skeep those.
					HypParaboloid::findParamsAndSamRadius( ratio, MAX_K, A, B, samRadius, h, minSamRadiusH() );
				}
				catch( std::exception& e )
				{
					CHKERRXX( PetscPrintf( mpi.comm(), "  -- Skipped ratio %03i [%.8g] for |hk_max| %03i = [%.8g]\n", rIdx, ratio, hkIdx, maxHKVal ) );
					continue;
				}

				HypParaboloid hypParaboloid( A, B );		// All is good, create the canonical surface.

				double maxHKError = 0, maxIH2KGError = 0;			// Tracking the maximum error and number of samples collectively shared
				size_t loggedSamples[SAMPLE_TYPES] = {0, 0};		// across processes for this (max |hk|, ratio) pair.

				for( int nRot = 0; nRot < numRotations(); nRot++ )	// Perform as many as numRotations affine transformations.
				{
					//////////////////////////////////////// Defining the transformation parameters ////////////////////////////////////////

					double origin[P4EST_DIM] = {0, 0, 0};			// Hyp-paraboloid's perturbed frame.
					double rotAxis[P4EST_DIM] = {0, 0, 1};			// Rotation axis for possible random rotation.
					double rotAngle = 0;

					if( mpi.rank() == 0 )
					{
						for( auto& dim : origin )					// Random translation.
							dim = uniformDistH_2( gen );

						rotAngle = 2 * M_PI * uniformDist( gen );	// Random rotation axis and angle.
						double azimuthAngle = uniformDist( gen ) * 2 * M_PI;
						double polarAngle = acos( 2 * uniformDist( gen ) - 1 );
						rotAxis[0] = cos( azimuthAngle ) * sin( polarAngle );
						rotAxis[1] = sin( azimuthAngle ) * sin( polarAngle );
						rotAxis[2] = cos( polarAngle );
					}
					SC_CHECK_MPI( MPI_Bcast( origin, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// Share affine transformation across cores.
					SC_CHECK_MPI( MPI_Bcast( &rotAngle, 1, MPI_DOUBLE, 0, mpi.comm() ) );
					SC_CHECK_MPI( MPI_Bcast( rotAxis, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );

					///////////////////////////////// Setting up domain for current surface configuration //////////////////////////////////

					u_char octMaxRL;
					int n_xyz[P4EST_DIM];
					double xyz_min[P4EST_DIM];
					double xyz_max[P4EST_DIM];
					int periodic[P4EST_DIM] = {0, 0, 0};
					HypParaboloidLevelSet *pLS = setupDomain( mpi, hypParaboloid, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL,
															  samRadius, n_xyz, xyz_min, xyz_max );

					// p4est variables and data structures.
					p4est_t *p4est;
					p4est_nodes_t *nodes;
					my_p4est_brick_t brick;
					p4est_ghost_t *ghost;
					p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

					////////////////////////////////////// Discretize the domain and collect samples ///////////////////////////////////////

					// Create the forest using the hyp-paraboloid level-set as a refinement criterion.
					splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, octMaxRL, pLS, 3.0 );
					p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
					p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

					// Refine and partition forest.
					pLS->toggleCache( true );		// Turn on cache to speed up repeated distance computations.
					pLS->reserveCache( (size_t)pow( 0.75 * (xyz_max[0] - xyz_min[0]) / h, 3 ) );
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

					// A ghosted parallel PETSc vector to store level-set function values and where we computed exact signed distances.
					Vec phi = nullptr, exactFlag = nullptr;
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );

					// Populate phi and compute exact distance for vertices within a (linearly estimated) shell around Gamma.
					// Reinitialization perturbs the otherwise calculated exact distances.  The exactFlag vector holds nodes' status: only
					// those with 1's can be used for sampling.
					pLS->evaluate( p4est, nodes, phi, exactFlag, 5 );

					// Add random noise if requested.
					if( randomNoise() > 0 )
						addRandomNoiseToLSFunction( phi, nodes, genNoise, randomNoiseDist );

					my_p4est_level_set_t ls( ngbd );
					ls.reinitialize_2nd_order( phi, reinitIters() );

					std::vector<std::vector<double>> samples[SAMPLE_TYPES];
					std::pair<double, double> maxErrors;
					double minHKInBatch[SAMPLE_TYPES], maxHKInBatch[SAMPLE_TYPES];	// 0 for non-saddles, 1 for saddles.
					maxErrors = pLS->collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, minHKInBatch, maxHKInBatch,
													 samples[0], minHK(), samples[1], exactFlag, SQR( samRadius ), nonSaddleMinIH2KG() );

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

					pLS->clearCache();
					pLS->toggleCache( false );		// Done with cache.
					delete pLS;

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
				CHKERRXX( PetscPrintf( mpi.comm(), "[  %03i] %8.6f  ( %03i) %8.6f  %10.6f  %12.6f  %7i  %8.3f\n", hkIdx, maxHKVal, rIdx, ratio,
									   maxHKError, maxIH2KGError, loggedSamples[0]+loggedSamples[1], watch.get_duration_current() ) );

				// Save samples if it's time.
				if( kml::utils::saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, ABS( maxHK() - minHK() ), fileName,
											 bufferMinSize(), nHistBins(), histMedianFrac(), histMinFold(), false ) )
					printLogHeader( mpi );
			}
		}

		// Save any samples left in the buffers (by forcing the process).
		if( kml::utils::saveSamples( mpi, buffer, bufferSize, file, trackedMinHK, trackedMaxHK, ABS( maxHK() - minHK() ), fileName,
									 bufferMinSize(), nHistBins(), histMedianFrac(), histMinFold(), true ) )
			printLogHeader( mpi );

		if( mpi.rank() == 0 )
		{
			for( auto& f : file )
				f.close();
		}

		SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "<<< Done after %.3f seconds\n", watch.get_duration_current() ) );
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
	CHKERRXX( PetscPrintf( mpi.comm(), "______________________________________________________________________________\n") );
	CHKERRXX( PetscPrintf( mpi.comm(), "[hkIdx] hk_max    (rIdx) ratio     hk_max_err  h2kg_max_err  samples  time    \n" ) );
}

/**
 * Get randomized ratios between a given range (excluding ending points).
 * @param [in] mpi MPI environment.
 * @param [in] start Left-most ending point.
 * @param [in] end Right-most ending point.
 * @param [in] n Number of ratios whose absolute value lies in (start, end).
 * @param [out] ratios Array of values.
 * @param [in,out] gen Random engine.
 */
void randomizeRatios( const mpi_environment_t& mpi, const double& start, const double& end, const int& n, std::vector<double>& ratios,
					  std::mt19937& gen )
{
	kml::utils::uniformRandomSpace( mpi, start, end, n, ratios, gen, false, false );	// Don't include start and end in the linspace.
	if( mpi.rank() == 0 )
	{
		std::uniform_real_distribution<double> coin;		// Change ratio's signs randomly.
		for( int i = 0; i < n; i++ )
			ratios[i] *= (coin( gen ) < 0.5)? +1 : -1;		// Max hk will be negative if ratio < 0 and positive otherwise.
	}
	SC_CHECK_MPI( MPI_Bcast( ratios.data(), n, MPI_DOUBLE, 0, mpi.comm() ) );
}
