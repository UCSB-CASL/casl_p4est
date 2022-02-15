/**
 * Generating samples using a 2D Gaussian surface embedded in 3D.
 *
 * @note Only supporting multiprocess in the same machine so far.
 *
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: February 14, 2022.
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
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "gaussian_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>        minHK( pl, 0.01, "minHK", "Minimum mean dimensionless curvature (default: 0.01 = twice 0.005 from 2D)" );
	param_t<double>        maxHK( pl, 4./3, "maxHK", "Maximum mean dimensionless curvature (default: 4/3 = twice 2/3 from 2D)" );
	param_t<u_char>        maxRL( pl,    6, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<int>     reinitIters( pl,   10, "reinitIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<double> probMidMaxHK( pl,  1.0, "probMidMaxHK", "Easing-off distribution max probability to keep midpoint max HK (default: 1.0)" );
	param_t<std::string>  outDir( pl, "/Volumes/YoungMinEXT/k_ecnet_3d", "outDir", "Path where files will be written to (default: build folder)" );

	// These random generators are initialized to the same seed across processes; and that's fine -- we need that to
	// replicate the normal samples during data collection.  Unlike the 2D case, here we are not sweeping all the candi-
	// date nodes, only those selected by another bivariate normal distribution.
	std::mt19937 genNormal{};	// NOLINT Standard mersenne_twister_engine for bivariate normal sampling inside limiting ellipse.
	std::mt19937 genProb{};		// NOLINT Random engine for probability when choosing candidate nodes and to avoid clusters.
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

		// Preping the samples' file.  Notice we are no longer interested on exact-signed distance functions, only re-
		// initialized data.  File name is gaussian.csv; only rank 0 writes the samples to a file.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
		std::ofstream file;
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, "gaussian", file );

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
		const double MID_MAX_HKAPPA = (HK_MAX_LO + HK_MAX_UP) / 2;
		const int NUM_HK_MAX = (int)ceil( (HK_MAX_UP - HK_MAX_LO) / (2 * H) );

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

		// Logging header.
		CHKERRXX( PetscPrintf( mpi.comm(), "Height \tStart_MaxK \tEnd_MaxK \tMaxK_Error \tNum_Samples \tTime\n" ) );

		for( const auto& A : linspaceA )					// For each height, vary the u and v variances to achieve a
		{													// maximum curvature at the peak.
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
					const double ULIM = U_ZERO + gaussian.su();		// Limiting ellipse semi-axes.
					const double VLIM = V_ZERO + gaussian.sv();
					const double QLIM = A + 6 * H;					// Adding some padding so that we can sample points correctly at the tip.
					const size_t HALF_U_H = ceil(ULIM / H);			// Half u axis in H units.
					const size_t HALF_V_H = ceil(VLIM / H);			// Half v axis in H units.

					const double QCylCCoords[8][P4EST_DIM] = {		// Cylinder in canonical coords containing the Gaussian surface.
						{-ULIM, 0, QLIM}, {+ULIM, 0, QLIM}, 		// Top coords (the four points lying on the same QLIM found above).
						{0, -VLIM, QLIM}, {0, +VLIM, QLIM},
						{-ULIM, 0,    0}, {+ULIM, 0,    0},			// Base coords (the four points lying on the uv-plane).
						{0, -VLIM,    0}, {0, +VLIM,    0}
					};

					double maxHKError = 0;							// Tracking the maximum error and number of samples
					size_t numSamples = 0;							// collectively shared across processes.

					for( const auto& ROT_AXIS : ROT_AXES )	// Using Euler angles for the rotation transformation.
					{
						for( int nt = 0; nt < NUM_THETAS - 1; nt++ )	// Various rotation angles for same axis (skip-
						{												// (ping last endpoint because of augmentation).

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
							my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
							my_p4est_node_neighbors_t ngbd( &hierarchy, nodes );
							ngbd.init_neighbors();

							// Verify mesh size is H, as expected.
							double dxyz[P4EST_DIM];
							get_dxyz_min( p4est, dxyz );
							assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == H );

							// A ghosted parallel PETSc vector to store level-set function values and exact flag.
							Vec phi, exactFlag;
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );

							// Populate phi and compute exact distance for vertices within a (linearly estimated) shell
							// around Gamma.  Reinitialization perturbs the otherwise calculated exact distances.
							gLS.evaluate( p4est, nodes, phi, exactFlag );

							// Reinitialize level-set function.
							my_p4est_level_set_t ls( &ngbd );
							ls.reinitialize_2nd_order( phi, reinitIters() );

							// Sample nodes with a normal distribuction aligned with Gaussian.
							Vec sampledFlag;							// An flag vector to distinguish sampled nodes along the interface.
							CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );

							std::vector<std::vector<double>> samples;
							double hkError = gLS.collectSamples( p4est, nodes, &ngbd, phi, OCTREE_MAX_RL, xyz_min,
																 xyz_max, samples, genNormal, genProb, MID_MAX_HKAPPA,
																 probMidMaxHK(), minHK(), 0.05, sampledFlag, 1.0 );
							maxHKError = MAX( maxHKError, hkError );

							// Save samples to dataset file.
							numSamples += kml::utils::processSamplesAndSaveToFile( mpi, samples, file, H );

							// Create a VTK visualization when debugging.
							const double *phiReadPtr, *exactFlagReadPtr, *sampledFlagReadPtr;
							CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
							CHKERRXX( VecGetArrayRead( exactFlag, &exactFlagReadPtr ) );
							CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );

							std::ostringstream oss;
							oss << "gaussian_ds_test";
							my_p4est_vtk_write_all( p4est, nodes, ghost,
													P4EST_TRUE, P4EST_TRUE,
													3, 0, oss.str().c_str(),
													VTK_POINT_DATA, "phi", phiReadPtr,
													VTK_POINT_DATA, "sampledFlag", sampledFlagReadPtr,
													VTK_POINT_DATA, "exactFlag", exactFlagReadPtr );

							// Clean up.
							gLS.toggleCache( false );		// Done with cache.
							gLS.clearCache();

							CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
							CHKERRXX( VecRestoreArrayRead( exactFlag, &exactFlagReadPtr ) );
							CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

							CHKERRXX( VecDestroy( exactFlag ) );
							CHKERRXX( VecDestroy( sampledFlag ) );
							CHKERRXX( VecDestroy( phi ) );

							// Destroy the p4est and its connectivity structure.
							p4est_nodes_destroy( nodes );
							p4est_ghost_destroy( ghost );
							p4est_destroy( p4est );
							my_p4est_brick_destroy( connectivity, &brick );

							// Synchronize.
							SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );

							break; // TODO: remove...
						}

						break; // TODO: remove...
					}

					// Logging stats
					CHKERRXX( PetscPrintf( mpi.comm(), "%g \t%g \t%g \t%g \t%u \t%g\n",
										   A, START_MAX_K, END_MAX_K, maxHKError, numSamples, watch.get_duration_current() ) );

					break; // TODO: remove...
				}

				break; // TODO: remove...
			}

			break; // TODO: remove...
		}

		if( mpi.rank() == 0 )
			file.close();

		watch.read_duration_current( true );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}