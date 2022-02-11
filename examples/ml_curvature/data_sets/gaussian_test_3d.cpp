/**
 * Testing distance computation from a point to 2D Gaussian manifold immersed in 3D and triangulated into a cloud of
 * points organized into a balltree for fast querying.
 *
 * Based on matlab/gaussian_3d_adjusted_domain.m, steps 1 through 4.
 *
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: February 10, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_level_set.h>
#endif

#include "gaussian_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<unsigned short> maxRL( pl, 6, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Initializing OpenMP.
		int nThreads = 0;
#pragma omp parallel reduction( + : nThreads ) default( none )
			nThreads += 1;
		std::cout << "\n:: OpenMP :: Process can spawn " << nThreads << " thread(s)" << std::endl << std::endl;

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Gaussian data set test" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		std::cout << "Testing Gaussian level-set function in 3D" << std::endl;

		parStopWatch watch;
		watch.start();

		/////////////////////////// 1) Defining the Gaussian surface and its shape parameters //////////////////////////

		const double H = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		const double A = 200 *  H;							// Gaussian height.
		const double start_k_max = 2 / (3 * H);				// Starting max desired curvature; hk_max^up = 4/3  and  hk_max^low = 2/3 (2/3 and 1/3 in 2D).
		const double K_MAX = 2 * start_k_max;				// Max curvature at the peak.
		const double SU2 = 2 * A / start_k_max;				// Variances along u and v directions that yield this curvature.
		const double denom = K_MAX / A * SU2 - 1;
		assert( denom > 0 );
		const double SV2 = SU2 / denom;

		Gaussian gaussian( A, SU2, SV2 );					// Gaussian surface: Q(u,v) = A * exp(-0.5*(u^2/su^2 + v^2/sv^2)).

		// Finding how far to go in the limiting ellipse half-axes.  We'll tringulate surface only within this region.
		const double U_ZERO = gaussian.findKappaZero( H, dir::x );
		const double V_ZERO = gaussian.findKappaZero( H, dir::y );
		const double ULIM = U_ZERO + gaussian.su();			// Limiting ellipse semi-axes.
		const double VLIM = V_ZERO + gaussian.sv();
		const double QLIM = A + 6 * H;						// Adding some padding so that we can sample points correctly at the tip.
		const size_t halfU = ceil(ULIM / H);				// Half u axis in H units.
		const size_t halfV = ceil(VLIM / H);				// Half v axis in H units.

		//////////////////////////// 2) Defining the transformed Gaussian level-set function ///////////////////////////

		const Point3 trans = {-0.125, 0.125, -0.125};		// Translation of canonical coordinate system.
		const Point3 rotAxis = {1, -1, 0};					// Axis of rotation (normalized when constructing level-set).
		const double rotAngle = 11 * M_PI / 36;				// Rotation angle about rotAxis.

		double timeCreate = watch.get_duration_current();
		GaussianLevelSet gaussianLevelSet( trans, rotAxis.normalize(), rotAngle, halfU, halfV, maxRL(), &gaussian, SQR( ULIM ), SQR( VLIM ), 5 );
		std::cout << "Created balltree in " << watch.get_duration_current() - timeCreate << " secs." << std::endl;
		gaussianLevelSet.dumpTriangles( "gaussian_triangles.csv" );

		//////////////////// 3) Finding the world coords of (canonical) cylinder containing Q(u,v) /////////////////////

		const double QCylCCoords[8][P4EST_DIM] = {
			{-ULIM, 0, QLIM}, {+ULIM, 0, QLIM}, {0, -VLIM, QLIM}, {0, +VLIM, QLIM},	// Top coords (the four points lying on the same QLIM found above).
			{-ULIM, 0,    0}, {+ULIM, 0,    0}, {0, -VLIM,    0}, {0, +VLIM,    0}	// Base coords (the four points lying on the uv-plane).
		};

		double minQCylWCoords[P4EST_DIM] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};		// Hold the minimum and maximum cylinder
		double maxQCylWCoords[P4EST_DIM] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};		// world coordinates.
		for( const auto& cylCPoint : QCylCCoords )
		{
			Point3 cylWPoint = gaussianLevelSet.toWorldCoordinates( cylCPoint[0], cylCPoint[1], cylCPoint[2] );
			for( int i = 0; i < P4EST_DIM; i++ )
			{
				minQCylWCoords[i] = MIN( minQCylWCoords[i], cylWPoint.xyz( i ) );
				maxQCylWCoords[i] = MAX( maxQCylWCoords[i], cylWPoint.xyz( i ) );
			}
		}

		//////////////////// 4) Use the x,y,z ranges to find the domain's length in each direction /////////////////////

		double QCylWRange[P4EST_DIM];
		double WCentroid[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			QCylWRange[i] = maxQCylWCoords[i] - minQCylWCoords[i];
			WCentroid[i] = (minQCylWCoords[i] + maxQCylWCoords[i]) / 2;		// Raw centroid.
			WCentroid[i] = round( WCentroid[i] / H ) * H;					// A numerically good centroid as a multiple of H.
		}

		const double CUBE_SIDE_LEN = MAX( QCylWRange[0], MAX( QCylWRange[1], QCylWRange[2] ) );
		const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL() - 5 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		const unsigned char OCTREE_MAX_RL = maxRL() - OCTREE_RL_FOR_LEN;	// Effective max refinement level to achieve desired H.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );				// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;				// Adjusted domain cube len as a multiple of H and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		// Defining a cubic domain that contains at least the Gaussian and its limiting ellipse.
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};
		double xyz_min[] = {WCentroid[0] - HALF_D_CUBE_SIDE_LEN, WCentroid[1] - HALF_D_CUBE_SIDE_LEN, WCentroid[2] - HALF_D_CUBE_SIDE_LEN};
		double xyz_max[] = {  xyz_min[0] + D_CUBE_SIDE_LEN     ,   xyz_min[1] + D_CUBE_SIDE_LEN     ,   xyz_min[2] + D_CUBE_SIDE_LEN     };
		int periodic[] = {0, 0, 0};											// Non-periodic domain.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		////////////////////////////////// 5) Proceed with discretization and sampling /////////////////////////////////

		// Create the forest using the Gaussian level-set as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, OCTREE_MAX_RL, &gaussianLevelSet, 2.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		double startTimePartitioningGrid = watch.get_duration_current();
		gaussianLevelSet.toggleCache( true );				// Turn on cache to speed up repeated signed distance
		gaussianLevelSet.reserveCache( (size_t)pow( 0.75 * HALF_D_CUBE_SIDE_LEN / H, 3 ) );	// Reserve space in cache to improve hashing.
		for( int i = 0; i < OCTREE_MAX_RL; i++ )											// queries for grid points.
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}
		std::cout << "Refining/coarsening and partitioning duration: " << watch.get_duration_current() - startTimePartitioningGrid << std::endl;

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
		double startTimeProcessingQueries = watch.get_duration_current();
		gaussianLevelSet.evaluate( p4est, nodes, phi, exactFlag );
		std::cout << "Query processing duration: " << watch.get_duration_current() - startTimeProcessingQueries << std::endl;

		gaussianLevelSet.toggleCache( false );		// Done with cache: clear it on exit.
		gaussianLevelSet.clearCache();

		// Reinitialize level-set function.
		double startTimeReinitialization = watch.get_duration_current();
		my_p4est_level_set_t ls( &ngbd );
		ls.reinitialize_2nd_order( phi, reinitNumIters() );
		std::cout << "Reinitialization duration: " << watch.get_duration_current() - startTimeReinitialization << std::endl;

		// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface.
		double startTimeCollecting = watch.get_duration_current();
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &ngbd, (char)OCTREE_MAX_RL );

		// An interface flag vector to distinguish nodes along the interface with full uniform neighborhoods.
		// By default, its values are set to 0.
		Vec interfaceFlag;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &interfaceFlag ) );

		double *interfaceFlagPtr;
		CHKERRXX( VecGetArray( interfaceFlag, &interfaceFlagPtr ) );

		// Getting the full uniform stencils of interface points.
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		const double *phiReadPtr, *exactFlagReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecGetArrayRead( exactFlag, &exactFlagReadPtr ) );

		// Now, check nodes next to Gamma.
		for( auto n : indices )
		{
			double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
			node_xyz_fr_n( n, p4est, nodes, xyz );	// Skip nodes very close to the boundary.
			if( ABS( xyz[0] - xyz_min[0] ) <= 4 * H || ABS( xyz[0] - xyz_max[0] ) <= 4 * H ||
				ABS( xyz[1] - xyz_min[1] ) <= 4 * H || ABS( xyz[1] - xyz_max[1] ) <= 4 * H ||
				ABS( xyz[2] - xyz_min[2] ) <= 4 * H || ABS( xyz[2] - xyz_max[2] ) <= 4 * H )
				continue;

			std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D. // TODO: Get full stencils on demand for select vertices.
			try
			{
				if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					interfaceFlagPtr[n] = 1;
			}
			catch( std::exception &e )
			{
				std::cerr << "Node (" << n << "): " << e.what() << std::endl;
			}
		}

		std::cout << "Collecting nodes duration: " << watch.get_duration_current() - startTimeCollecting << std::endl;
		watch.stop();

		std::ostringstream oss;
		oss << "gaussian_test";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								3, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr,
								VTK_POINT_DATA, "exactFlag", exactFlagReadPtr );

		// Clean up.
		CHKERRXX( VecRestoreArray( interfaceFlag, &interfaceFlagPtr ) );
		CHKERRXX( VecRestoreArrayRead( exactFlag, &exactFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		CHKERRXX( VecDestroy( exactFlag ) );
		CHKERRXX( VecDestroy( interfaceFlag ) );
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