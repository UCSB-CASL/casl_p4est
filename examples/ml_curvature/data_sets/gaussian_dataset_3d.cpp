/**
 * Generating samples using a 2D Gaussian surface embedded in 3D.
 *
 * Files written to file are of the form "#/gaussian_$.csv", where # is the unit-cube maximum level of refinement and $
 * is the Gaussian height index (i.e., 0, 1,... ).
 *
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: February 17, 2022.
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
#include "gaussian_3d.h"
#include <src/parameter_list.h>
#include <cassert>


void printLogHeader( const mpi_environment_t& mpi );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>         minHK( pl, 0.01, "minHK", "Minimum mean dimensionless curvature (default: 0.01 = twice 0.005 from 2D)" );
	param_t<double>         maxHK( pl, 4./3, "maxHK", "Maximum mean dimensionless curvature (default: 4/3 = twice 2/3 from 2D)" );
	param_t<u_char>         maxRL( pl,    6, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<int>      reinitIters( pl,   10, "reinitIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<double>   probMaxHKLB( pl,  0.9, "probMaxHKLB", "Easing-off max probability for lower bound max HK (default: 0.9)" );
	param_t<u_short>    startAIdx( pl,    0, "startAIdx", "Start index for Gaussian height (default: 0)" );
	param_t<float> histMedianFrac( pl,  0.4, "histMedianFrac", "Histogram subsampling median fraction (default: 0.4)" );
	param_t<std::string>   outDir( pl, "/Volumes/YoungMinEXT/k_ecnet_3d", "outDir", "Path where files will be written to (default: build folder)" );

	// These random generators are initialized to the same seed across processes.
	std::mt19937 genProb{};		// NOLINT Random engine for probability when choosing candidate nodes.
	std::mt19937 genTrans{};	// NOLINT This engine is used for the random shift of the Gaussian's canonical frame.

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating a Gaussian data set" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		CHKERRXX( PetscPrintf( mpi.comm(), "\n**************** Generating a Gaussian data set in 3D ****************\n" ) );

		parStopWatch watch;
		watch.start();

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double H = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / H;					// Curvature bounds.
		const double MAX_K = maxHK() / H;
		const double MAX_A = 2 / MIN_K;						// Height bounds: MAX_A, which is also the max sphere radius and
		const double MIN_A = 8 / MAX_K;						// MIN_A = 4*(min radius).
		const int NUM_A = (int)((MAX_A - MIN_A) / H / 7);	// Number of distinct heights.
		const double HK_MAX_LO = maxHK() / 2;				// Maximum HK bounds at the peak.
		const double HK_MAX_UP = maxHK();
		const int NUM_HK_MAX = (int)ceil( (HK_MAX_UP - HK_MAX_LO) / (3 * H) );

		// Affine transformation parameters.
		const Point3 ROT_AXES[P4EST_DIM] = {{1,0,0}, {0,1,0}, {0,0,1}};	// Let's use Euler angles; here, the rotation axes.
		const double MIN_THETA = -M_PI_2;								// For each axis, we vary the angle from -pi/2
		const double MAX_THETA = +M_PI_2;								// +pi/2 without the end point.
		const int NUM_THETAS = 38;
		std::uniform_real_distribution<double> uniformDistributionH_2( -H/2, +H/2 );	// Random translation.

		PetscPrintf( mpi.comm(), ">> Began to generate dataset for %i distinct heights, MaxRL=%i, H=%g\n", NUM_A, maxRL(), H );

		std::vector<double> linspaceA;						// Height spread.
		linspace( MIN_A, MAX_A, NUM_A, linspaceA );

		std::vector<double> linspaceHK_MAX;					// HK_MAX spread (i.e., the max hk at the peak).
		linspace( HK_MAX_LO, HK_MAX_UP, NUM_HK_MAX, linspaceHK_MAX );

		std::vector<double> linspaceTheta;					// Angular spread for each standard axis.
		linspace( MIN_THETA, MAX_THETA, NUM_THETAS, linspaceTheta );

		///////////////////////////////////////////// Data-production loop /////////////////////////////////////////////

		const size_t TOT_ITERS = 3 * NUM_HK_MAX * (NUM_HK_MAX + 1) / 2;	// Num of axes times num of pairs of hk_max.
		size_t step = 0;
		const int BUFFER_MIN_SIZE = 100000;
		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Buffer of accumulated (normalized and augmented) samples.
		if( mpi.rank() == 0 )								// Only rank 0 controls the buffer.
			buffer.reserve( BUFFER_MIN_SIZE );
		int bufferSize = 0;									// But everyone knows the buffer size to keep them in sync.
		double trackedMinHK = DBL_MAX, trackedMaxHK = 0;	// We need to track the min and max |hk*| from processed
		SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );			// samples until we save the buffered feature vectors.

		// Logging header.
		CHKERRXX( PetscPrintf( mpi.comm(), "Expecting %u iterations per height and %d hk_max steps\n\n", TOT_ITERS, NUM_HK_MAX ) );

		for( u_short a = startAIdx(); a < NUM_A; a++ )		// For each height, vary the u and v variances to achieve a
		{													// maximum curvature at the peak.
			const double A = linspaceA[a];

			// Preping the samples' file.  Notice we are no longer interested on exact-signed distance functions, only re-
			// initialized data.  File name is gaussian.csv; only rank 0 writes the samples to a file.
			const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
			std::ofstream file;
			std::string fileName = "gaussian_" + std::to_string( a ) + ".csv";
			kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

			printLogHeader( mpi );

			size_t iters = 0;

			for( int s = 0; s < NUM_HK_MAX; s++ )			// Define a starting HK_MAX to define u-variance: SU2.
			{
				const double START_MAX_K = linspaceHK_MAX[s] / H;
				const double SU2 = 2 * A / START_MAX_K;

				for( int t = s; t < NUM_HK_MAX; t++ )				// Then, define HK_MAX at the peak to find SV2.
				{													// Intuitively, we start with a circular Gaussian,
					const double END_MAX_K = linspaceHK_MAX[t] / H;	// and we transition to an elliptical little by little.
					const double denom = END_MAX_K / A * SU2 - 1;
					assert( denom > 0 );
					const double SV2 = SU2 / denom;					// v-variance: SV2.

					Gaussian gaussian( A, SU2, SV2 );				// Gaussian: Q(u,v) = A * exp(-0.5*(u^2/su^2 + v^2/sv^2)).

					// Finding how far to go in the limiting ellipse half-axes.  We'll tringulate surface only within
					// this region.  Note: this Gaussian surface will remain constant throughout level-set variations.
					const double U_ZERO = gaussian.findKappaZero( H, dir::x );
					const double V_ZERO = gaussian.findKappaZero( H, dir::y );
					const double ULIM = U_ZERO + gaussian.su();		// Limiting ellipse semi-axes for triangulation.
					const double VLIM = V_ZERO + gaussian.sv();
					const double QTOP = A + 4 * H;					// Adding some padding so that we can sample points correctly at the tip.
					double quZero = gaussian( U_ZERO, 0 );			// Let's find the lowest Q.
					double qvZero = gaussian( 0, V_ZERO );
					const double QBOT = MAX( 0., MIN( quZero, qvZero ) - 4 * H );
					const size_t HALF_U_H = ceil(ULIM / H);			// Half u axis in H units.
					const size_t HALF_V_H = ceil(VLIM / H);			// Half v axis in H units.

					const double QCylCCoords[8][P4EST_DIM] = {		// Cylinder in canonical coords containing the Gaussian surface.
						{-U_ZERO, 0, QTOP}, {+U_ZERO, 0, QTOP}, 	// Top coords (the four points lying on the same QTOP found above).
						{0, -V_ZERO, QTOP}, {0, +V_ZERO, QTOP},
						{-U_ZERO, 0, QBOT}, {+U_ZERO, 0, QBOT},		// Base coords (the four points lying on the same QBOT found above).
						{0, -V_ZERO, QBOT}, {0, +V_ZERO, QBOT}
					};

					for( int axisIdx = 0; axisIdx < P4EST_DIM; axisIdx++ )	// Use Euler angles to rotate canonical coord system.
					{
						const Point3 ROT_AXIS = ROT_AXES[axisIdx];
						double maxHKError = 0;							// Tracking the maximum error and number of samples
						size_t loggedSamples = 0;						// collectively shared across processes for this rot axis.

						for( int nt = 0; nt < NUM_THETAS - 1; nt++ )	// Various rotation angles for same axis (skip last one).
						{

							/////////////////// Defining the transformed Gaussian level-set function ///////////////////

							const double THETA = linspaceTheta[nt];
							const Point3 TRANS(
								uniformDistributionH_2( genTrans ),		// Translate canonical origin coords by a random
								uniformDistributionH_2( genTrans ),		// perturbation from world's origin.
								uniformDistributionH_2( genTrans )
							);

							// Also discretizes the surface using a balltree to speed up queries during grid refinment.
							GaussianLevelSet gLS( &mpi, TRANS, ROT_AXIS.normalize(), THETA, HALF_U_H, HALF_V_H, maxRL(),
												  &gaussian, SQR( ULIM ), SQR( VLIM ), 5 );

							//////////// Finding the world coords of (canonical) cylinder containing Q(u,v) ////////////

							double minQCylWCoords[P4EST_DIM] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};	// Min and max cylinder
							double maxQCylWCoords[P4EST_DIM] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};	// world coordinates.
							for( const auto& cylCPoint : QCylCCoords )
							{
								Point3 cylWPoint = gLS.toWorldCoordinates( cylCPoint[0], cylCPoint[1], cylCPoint[2] );
								for( int i = 0; i < P4EST_DIM; i++ )
								{
									minQCylWCoords[i] = MIN( minQCylWCoords[i], cylWPoint.xyz( i ) );
									maxQCylWCoords[i] = MAX( maxQCylWCoords[i], cylWPoint.xyz( i ) );
								}
							}

							//////////// Use the x,y,z ranges to find the domain's length in each direction ////////////

							double QCylWRange[P4EST_DIM];
							double WCentroid[P4EST_DIM];
							for( int i = 0; i < P4EST_DIM; i++ )
							{
								QCylWRange[i] = maxQCylWCoords[i] - minQCylWCoords[i];
								WCentroid[i] = (minQCylWCoords[i] + maxQCylWCoords[i]) / 2;	// Raw centroid.
								WCentroid[i] = round( WCentroid[i] / H ) * H;				// Centroid as a multiple of H.
							}

							const double CUBE_SIDE_LEN = MAX( QCylWRange[0], MAX( QCylWRange[1], QCylWRange[2] ) );
							const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL() - 5 );	// Defines the log2 of octree's len (i.e., octree's len is a power of two).
							const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
							const unsigned char OCTREE_MAX_RL = maxRL() - OCTREE_RL_FOR_LEN;// Effective max refinement level to achieve desired H.
							const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );			// Number of trees in each dimension.
							const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;			// Adjusted domain cube len as a multiple of *both* H and octree len.
							const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

							// Defining a cubic domain that contains at least the Gaussian and its limiting ellipse.
							int n_xyz[] = {N_TREES, N_TREES, N_TREES};
							double xyz_min[] = {
								WCentroid[0] - HALF_D_CUBE_SIDE_LEN,
								WCentroid[1] - HALF_D_CUBE_SIDE_LEN,
								WCentroid[2] - HALF_D_CUBE_SIDE_LEN
							};
							double xyz_max[] = {
								xyz_min[0] + D_CUBE_SIDE_LEN,
								xyz_min[1] + D_CUBE_SIDE_LEN,
								xyz_min[2] + D_CUBE_SIDE_LEN
							};
							int periodic[] = {0, 0, 0};				// Non-periodic domain.

							p4est_t *p4est;							// p4est variables and data structures.
							p4est_nodes_t *nodes;
							my_p4est_brick_t brick;
							p4est_ghost_t *ghost;
							p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

							//////////////////// Let's now discretize the domain and collect samples ///////////////////

							// Create the forest using the Gaussian level-set as a refinement criterion.
							splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, OCTREE_MAX_RL, &gLS, 2.0 );
							p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
							p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

							// Refine and partition forest.
							gLS.toggleCache( true );		// Turn on cache to speed up repeated distance computations.
							gLS.reserveCache( (size_t)pow( 0.75 * HALF_D_CUBE_SIDE_LEN / H, 3 ) );	// Reserve space in cache to improve hashing.
							for( int i = 0; i < OCTREE_MAX_RL; i++ )								// queries for grid points.
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

							// Verify mesh size is H, as expected.
							double dxyz[P4EST_DIM];
							get_dxyz_min( p4est, dxyz );
							assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == H );

							// A ghosted parallel PETSc vector to store level-set function values and a couple of flags.
							Vec phi = nullptr, exactFlag = nullptr, sampledFlag = nullptr;
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );

							// Populate phi and compute exact distance for vertices within a (linearly estimated) shell
							// around Gamma.  Reinitialization perturbs the otherwise calculated exact distances.
							gLS.evaluate( p4est, nodes, phi, exactFlag );

							// Reinitialize level-set function.
							my_p4est_level_set_t ls( ngbd );
							ls.reinitialize_2nd_order( phi, reinitIters() );

							// Collect samples using an overriden limiting ellipse on the canonical uv plane.
							std::vector<std::vector<double>> samples;
							double minHKInBatch, maxHKInBatch;
							double hkError = gLS.collectSamples( p4est, nodes, ngbd, phi, OCTREE_MAX_RL, xyz_min,
																 xyz_max, samples, minHKInBatch, maxHKInBatch, genProb,
																 HK_MAX_LO, probMaxHKLB(), minHK(), 0.05, sampledFlag,
																 SQR( U_ZERO ), SQR( V_ZERO ) );
							maxHKError = MAX( maxHKError, hkError );
							trackedMinHK = MIN( minHKInBatch, trackedMinHK );	// Update the tracked |hk*| bounds.
							trackedMaxHK = MAX( maxHKInBatch, trackedMaxHK );	// These are shared across processes.

							// Save samples to dataset file.
							int batchSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, H );
							loggedSamples += batchSize;
							bufferSize += batchSize;

							gLS.toggleCache( false );		// Done with cache.
							gLS.clearCache();

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
						if( bufferSize >= BUFFER_MIN_SIZE )
						{
							int savedSamples = kml::utils::histSubSamplingAndSaveToFile( mpi, buffer, file,
																						 (FDEEP_FLOAT_TYPE)minHK(),
																						 (FDEEP_FLOAT_TYPE)maxHK(),
																						 100, histMedianFrac() );
							CHKERRXX( PetscPrintf( mpi.comm(), "[*] Saved %d out of %d samples to output file %s, with |hk*| in the range of [%f, %f].\n",
												   savedSamples, bufferSize, fileName.c_str(), trackedMinHK, trackedMaxHK ) );
							printLogHeader( mpi );

							buffer.clear();							// Reset control variables.
							if( mpi.rank() == 0 )
								buffer.reserve( BUFFER_MIN_SIZE );
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
																			 (FDEEP_FLOAT_TYPE)minHK(),
																			 (FDEEP_FLOAT_TYPE)maxHK(),
																			 100, histMedianFrac() );
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


void printLogHeader( const mpi_environment_t& mpi )
{
	CHKERRXX( PetscPrintf( mpi.comm(), "_____________________________________________________________________________________________________________________\n") );
	CHKERRXX( PetscPrintf( mpi.comm(), "[Step  ] \tHeight \t\t(ss) Start_MaxHK \t(tt) End_MaxHK \tAxis \tMaxHK_Error \tNum_Samples\t(%%_Done) \tTime\n" ) );
}
