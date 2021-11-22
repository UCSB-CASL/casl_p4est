/**
 * Testing distance computation from a point to a paraboloid two-dimensional manifolds immersed in 3D and discretized
 * by points and triangles organized into a balltree for fast querying.
 *
 * Developer: Luis Ángel.
 * Created: November 14, 2021.
 * Updated: November 21, 2021.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

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

#include "paraboloid_3d.h"
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
		PetscErrorCode ierr;

		if( mpi.size() > 1 )					// To test we don't admit more than a single process.
			throw std::runtime_error( "Only a single process is allowed!" );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Paraboloid data set test" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		std::cout << "Testing paraboloid level-set function in 3D" << std::endl;

		parStopWatch watch;
		watch.start();

		// Domain information.
		const double MIN_D = -0.5, MAX_D = -MIN_D;			// The canonical space is [-0.5, +0.5]^3.
		const double H = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		int n_xyz[] = {1, 1, 1};							// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the level-set function to be reinitialized.
		const double A = 4;									// Paraboloid params: Q(u,v) = A*u^2 + B*v^2.
		const double B = 1;
		Paraboloid paraboloid( A, B );
		const Point3 translation = {0, 0, 0};				// Translation of canonical coordinate system.
		const Point3 rotationAxis = {0, 0, 1};				// Axis of rotation.
		const double beta = 0;								// Rotation angle about RotAxis.

		// Finding how far to go in the half-axes to get a lower bound on the maximum height in Q(u,v) = A*u^2 + B*v^2.
		const double R = MAX_D * sqrt( 3 );					// Radius of circumscribing circle (i.e., containing the domain cube).
		const double hiQU = 0.5 * (-1/A + sqrt(1./SQR(A) + 4*SQR(R)));	// Lower bound for Q along u axis (for v=0).
		const size_t halfU = ceil(sqrt( hiQU / A ) / H);				// Half u axis in terms of H.
		const double hiQV = 0.5 * (-1/B + sqrt(1./SQR(B) + 4*SQR(R)));	// Lower bound for Q along v axis (for u=0).
		const size_t halfV = ceil(sqrt( hiQV / B ) / H);				// Half v axis in terms of H.

		double timeCreate = watch.get_duration_current();
		ParaboloidLevelSet paraboloidLevelSet( translation, rotationAxis, beta, halfU, halfV, maxRL(), &paraboloid );
		std::cout << "Created balltree in " << watch.get_duration_current() - timeCreate << " secs." << std::endl;
		paraboloidLevelSet.dumpTriangles( "paraboloid_triangles.csv" );
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( MAX( 1, maxRL() - 5 ), maxRL(), &paraboloidLevelSet, 2.0 );

		// Create the forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		double timeProcessingQueries = watch.get_duration_current();
		paraboloidLevelSet.toggleCache( true );				// Turn on cache to speed up repeated signed distance
		for( int i = 0; i < maxRL(); i++ )					// queries for grid points.
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
		ngbd.init_neighbors(); 				// This is not mandatory, but it can only help performance given
											// how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// Calculate the level-set function values for each independent node.
		double *phiPtr;
		CHKERRXX( VecGetArray( phi, &phiPtr ) );

		// TODO: Here, we should recompute the exact distance for vertices within a shell around Gamma.
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			phiPtr[n] = paraboloidLevelSet( xyz[0], xyz[1], xyz[2] );
		}

		CHKERRXX( VecRestoreArray( phi, &phiPtr ) );
		paraboloidLevelSet.toggleCache( false );		// Done with cache: clear it on exit.
		paraboloidLevelSet.clearCache();

		// Reinitialize level-set function.
		my_p4est_level_set_t ls( &ngbd );
		ls.reinitialize_2nd_order( phi, reinitNumIters() );

		// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &ngbd, (char)maxRL() );

		// An interface flag vector to distinguish nodes along the interface with full uniform neighborhoods.
		// By default, its values are set to 0.
		Vec interfaceFlag;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &interfaceFlag ) );

		double *interfaceFlagPtr;
		CHKERRXX( VecGetArray( interfaceFlag, &interfaceFlagPtr ) );

		// Getting the full uniform stencils of interface points.
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Now, check nodes next to Gamma.
		for( auto n : indices )
		{
			double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
			node_xyz_fr_n( n, p4est, nodes, xyz );	// Skip nodes very close to the boundary.
			if( ABS( xyz[0] - MIN_D ) <= 4 * H || ABS( xyz[0] - MAX_D ) <= 4 * H ||
				ABS( xyz[1] - MIN_D ) <= 4 * H || ABS( xyz[1] - MAX_D ) <= 4 * H ||
				ABS( xyz[2] - MIN_D ) <= 4 * H || ABS( xyz[2] - MAX_D ) <= 4 * H )
				continue;

			std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
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

		std::cout << "Query processing duration: " << watch.get_duration_current() - timeProcessingQueries << std::endl;
		watch.stop();

		std::ostringstream oss;
		oss << "paraboloid_test";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								2, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr );

		// Clean up.
		CHKERRXX( VecRestoreArray( interfaceFlag, &interfaceFlagPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

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