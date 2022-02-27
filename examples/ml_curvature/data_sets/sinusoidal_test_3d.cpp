/**
 * Testing distance computation from a point to sinusoidal manifold immersed in 3D and triangulated into a cloud of
 * points organized into a balltree for fast querying.
 *
 * Developer: Luis Ángel.
 * Created: February 24, 2022.
 * Updated: February 25, 2022.
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
#include "sinusoidal_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>         minHK( pl, 0.01, "minHK", "Minimum mean dimensionless curvature (default: 0.01 = twice 0.005 from 2D)" );
	param_t<double>         maxHK( pl, 4./3, "maxHK", "Maximum mean dimensionless curvature (default: 4/3 = twice 2/3 from 2D)" );
	param_t<unsigned short> maxRL( pl, 6, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<double> probMaxHKLB( pl, 0.9, "probMaxHKLB", "Easing-off max probability for lower bound max HK (default: 0.9)" );
	param_t<std::string> outputDir( pl, ".", "outputDir", "Path where files will be written to (default: build folder)" );

	std::mt19937 genProb{};	// NOLINT Random engine for probability for choosing whether to sample a grid point or not.

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Initializing OpenMP.
		int nThreads = 0;
#pragma omp parallel reduction( + : nThreads ) default( none )
			nThreads += 1;
		std::cout << "\n:: OpenMP :: Process " << mpi.rank() << " can spawn " << nThreads << " thread(s)" << std::endl << std::endl;

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Three-dimensional sinusoidal data set test" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		std::cout << "Testing sinusoidal level-set function in 3D" << std::endl;

		// Preping the samples' file.  Notice we are no longer interested on exact-signed distance functions, only re-
		// initialized data.  File name is sinusoid.csv; only rank 0 writes the samples to a file.
		const std::string DATA_PATH = outputDir() + "/" + std::to_string( maxRL() );
		std::ofstream file;
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, "sinusoid.csv", file );

		parStopWatch watch( parStopWatch::all_timings );

		////////////////////////// 1) Defining the sinusoidal surface and its shape parameters /////////////////////////

		const double H = 1. / (1 << maxRL());				// Highest spatial resolution.
//		const double MAX_K = maxHK() / H;					// Steepest curvature.
		const double MIN_K = minHK() / H;					// Flattest curvature.
		const double MAX_A = 2 / MIN_K / 2;					// Height bounds: MAX_A, which is half the max sphere radius and
//		const double MIN_A = 10 / MAX_K;					// MIN_A = 5*(min radius).
		const double A = MAX_A;								// Sinusoid amplitude.
		const double start_k_max = 2 / (3 * H);				// Starting max desired curvature; hk_max^up = 4/3  and  hk_max^low = 2/3 (2/3 and 1/3 in 2D).
		const double K_MAX = 2 * start_k_max;				// Max curvature at the peak (here, I chose the maximum possible).
		const double WU = sqrt( start_k_max / (2 * A) );	// Frequencies along u and v directions that yield this curvature.
		const double WV = sqrt( K_MAX / A - SQR( WU ) );

		Sinusoid sinusoid( A, WU, WV );						// Sinusoidal surface: Q(u,v) = A * sin(wu*u) * sin(wv*v).

		/////////////////////// 2) Finding the limits for both triangulation and physical domain ///////////////////////

		const double SAM_RADIUS = MAX_A + 6 * H;			// Sampling radius (with enough padding).

		const double CUBE_SIDE_LEN = 2 * SAM_RADIUS;						// We want a cubic domain with an effective, yet small size.
		const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL() - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		const unsigned char OCTREE_MAX_RL = maxRL() - OCTREE_RL_FOR_LEN;	// Effective max refinement level to achieve desired H.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );				// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;				// Adjusted domain cube len as a multiple of H and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		const double D_CUBE_DIAG_LEN = sqrt( 3 ) * D_CUBE_SIDE_LEN;			// Use this diag to determine triangulated surface.
		const double UVLIM = D_CUBE_DIAG_LEN / 2 + H;						// Notice the padding to account for the random shift below.
		const size_t halfUV = ceil( UVLIM / H );							// Half UV domain in H units.

		////////////////////////// 3) Defining the transformed sinusoidal level-set function ///////////////////////////

		const Point3 trans = {-0.125, 0.125, -0.125};		// Translation of canonical coordinate system.
		const Point3 rotAxis = {1, -1, 0};					// Axis of rotation (normalized when constructing level-set).
		const double rotAngle = 11 * M_PI / 36;				// Rotation angle about rotAxis.

		watch.start();
		PetscPrintf( mpi.comm(), "Creating balltree" );
		SinusoidalLevelSet sinusoidalLevelSet( &mpi, trans, rotAxis.normalize(), rotAngle, halfUV, halfUV, maxRL(),
											   &sinusoid, SQR( UVLIM ), SAM_RADIUS );
		watch.read_duration_current( true );

		sinusoidalLevelSet.dumpTriangles( "sinusoidal_triangles.csv" );

		////////////////////////////////////////////// 4) Set up macromesh /////////////////////////////////////////////

		// Defining a symmetric cubic domain whose dimensions are multiples of H.
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};
		double xyz_min[] = {-HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN, -HALF_D_CUBE_SIDE_LEN};
		double xyz_max[] = {+HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN, +HALF_D_CUBE_SIDE_LEN};
		int periodic[] = {0, 0, 0};											// Non-periodic domain.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		////////////////////////////////// 5) Proceed with discretization and sampling /////////////////////////////////

		// Create the forest using the sinusoidal level-set as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, OCTREE_MAX_RL, &sinusoidalLevelSet, 2.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		watch.start();
		PetscPrintf( mpi.comm(), "Refining/coarsening and partitioning" );
		sinusoidalLevelSet.toggleCache( true );				// Turn on cache to speed up repeated signed distance
		sinusoidalLevelSet.reserveCache( (size_t)pow( 0.75 * HALF_D_CUBE_SIDE_LEN / H, 3 ) );	// Reserve space in cache to improve hashing.
		for( int i = 0; i < OCTREE_MAX_RL; i++ )												// queries for grid points.
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}
		watch.read_duration_current( true );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t ngbd( &hierarchy, nodes );
		ngbd.init_neighbors();

		// Verify mesh size.
		double dxyz[P4EST_DIM];
		get_dxyz_min( p4est, dxyz );
		assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == H );

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// A ghosted parallel vector to keep track of nodes where we computed exact signed distances to Gamma.
		Vec exactFlag;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );

		// Populate phi values and compute the exact distance for vertices within a (linearly estimated) shell around Gamma.
		watch.start();
		PetscPrintf( mpi.comm(), "Query processing" );
		sinusoidalLevelSet.evaluate( p4est, nodes, phi, exactFlag );
		watch.read_duration_current( true );

		// Reinitialize level-set function.
		watch.start();
		PetscPrintf( mpi.comm(), "Reinitialization" );
		my_p4est_level_set_t ls( &ngbd );
		ls.reinitialize_2nd_order( phi, reinitNumIters() );
		watch.read_duration_current( true );

		// Once the level-set function is reinitialized, sample nodes next to Gamma.
		watch.start();
		PetscPrintf( mpi.comm(), "Collecting samples" );
		Vec sampledFlag;							// A flag vector to distinguish sampled nodes along the interface.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
		Vec hkError;								// A vector with sampled |hk| error.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );

		std::vector<std::vector<double>> samples;
		double trackedMinHK, trackedMaxHK;
		double maxHKError = sinusoidalLevelSet.collectSamples( p4est, nodes, &ngbd, phi, OCTREE_MAX_RL, xyz_min, xyz_max,
															   samples, trackedMinHK, trackedMaxHK, genProb,
															   H * K_MAX / 2, probMaxHKLB(), minHK(), 0.01, sampledFlag,
															   NAN, exactFlag, hkError );
		PetscPrintf( mpi.comm(), " with a max hk error of %g", maxHKError );
		watch.read_duration_current( true );

		watch.start();
		PetscPrintf( mpi.comm(), "Saving samples to a file; " );
		size_t numSamples = kml::utils::processSamplesAndSaveToFile( mpi, samples, file, H );
		PetscPrintf( mpi.comm(), " %u samples in total", numSamples );

		if( mpi.rank() == 0 )
			file.close();

		watch.read_duration_current( true );
		watch.stop();

		const double *phiReadPtr, *exactFlagReadPtr, *sampledFlagReadPtr, *hkErrorReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecGetArrayRead( exactFlag, &exactFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( hkError, &hkErrorReadPtr ) );

		std::ostringstream oss;
		oss << "sinusoid_test";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								4, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "sampledFlag", sampledFlagReadPtr,
								VTK_POINT_DATA, "exactFlag", exactFlagReadPtr,
								VTK_POINT_DATA, "hkError", hkErrorReadPtr );

		// Clean up.
		sinusoidalLevelSet.toggleCache( false );		// Done with cache.
		sinusoidalLevelSet.clearCache();

		CHKERRXX( VecRestoreArrayRead( hkError, &hkErrorReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( exactFlag, &exactFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		CHKERRXX( VecDestroy( hkError ) );
		CHKERRXX( VecDestroy( exactFlag ) );
		CHKERRXX( VecDestroy( sampledFlag ) );
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}