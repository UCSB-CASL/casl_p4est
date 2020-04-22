// System
#include <stdexcept>
#include <iostream>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/FastSweeping.h>

using namespace std;


class Circle: public CF_2
{
private:
	double _x0, _y0, _r;

public:
	Circle( double x0, double y0, double r ) : _x0( x0 ), _y0( y0 ), _r( r )
	{}

	double operator()( double x, double y ) const override
	{
		return sqrt( SQR( x - _x0 ) + SQR( y - _y0 ) ) - _r;
	}
};


int main( int argc, char* argv[] )
{
	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// p4est variables.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		p4est_connectivity_t *connectivity;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;

		// Domain information.
		int n_xyz[] = {1, 1, 1};
		double xyz_min[] = {-1, -1, -1};
		double xyz_max[] = {+1, +1, +1};
		int periodic[] = {0, 0, 0};
		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		cmdParser cmd;
		cmd.add_option( "lmin", "min level for refinement" );
		cmd.add_option( "lmax", "max level for refinement" );
		cmd.parse( argc, argv );

		Circle circle( 0, 0, 0.3 );
		splitting_criteria_cf_t levelSetSC( cmd.get( "lmin", 1 ), cmd.get( "lmax", 2 ), &circle );

		parStopWatch w;
		w.start( "total time" );

		// Create the forest using a level set as refinement criterion.
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
		nodeNeighbors.init_neighbors(); // This is not mandatory, but it can only help performance given how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );

		// Calculate the level-set function values for each independent node (i.e. locally owned and ghost nodes).
		double *phiPtr;
		ierr = VecGetArray( phi, &phiPtr );
		CHKERRXX( ierr );
		for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( i, p4est, nodes, xyz );
			phiPtr[i] = circle( xyz[0], xyz[1] );
		}
		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );

		/// Testing the fast sweeping algorithm ///
		FastSweeping fsm{};

		Vec orderings[4];
		ierr = VecCreateGhostNodes( p4est, nodes, &orderings[0] );			// Verifying the orderings.
		CHKERRXX( ierr );
		ierr = VecDuplicate( orderings[0], &orderings[1] );
		CHKERRXX( ierr );
		ierr = VecDuplicate( orderings[0], &orderings[2] );
		CHKERRXX( ierr );
		ierr = VecDuplicate( orderings[0], &orderings[3] );
		CHKERRXX( ierr );

		p4est_locidx_t **orderingsSrc = fsm.prepare( p4est, ghost, nodes, &nodeNeighbors, xyz_min, xyz_max );
//		fsm.reinitializeLevelSetFunction( phi, xyz_min, xyz_max, l1Norm_1 );

		double *orderingsPtr[4];
		for( int i = 0; i < 4; i++ )
		{
			ierr = VecGetArray( orderings[i], &orderingsPtr[i] );
			CHKERRXX( ierr );

			for( int j = 0; j < nodes->num_owned_indeps; j++ )
				orderingsPtr[i][j] = orderingsSrc[i][j];

			ierr = VecGhostUpdateBegin( orderings[i], INSERT_VALUES, SCATTER_FORWARD );		// After we are done with the locally
			CHKERRXX( ierr );																// owned nodes, scatter them onto the
			ierr = VecGhostUpdateEnd( orderings[i], INSERT_VALUES, SCATTER_FORWARD ); 		// ghost nodes.
			CHKERRXX( ierr );
		}

		// Reinitialize the level-set function values.
		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, 100 );

		std::ostringstream oss;
		oss << "fsm_" << mpi.size() << "_" << P4EST_DIM;
		ierr = VecGetArray( phi, &phiPtr );
		CHKERRXX( ierr );
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								5, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiPtr,
								VTK_POINT_DATA, "orderings0", orderingsPtr[0],
								VTK_POINT_DATA, "orderings1", orderingsPtr[1],
								VTK_POINT_DATA, "orderings2", orderingsPtr[2],
								VTK_POINT_DATA, "orderings3", orderingsPtr[3] );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );

		for( int i = 0; i < 4; i++ )
		{
			ierr = VecRestoreArray( orderings[i], &orderingsPtr[i] );
			CHKERRXX( ierr );

			ierr = VecDestroy( orderings[i] );
			CHKERRXX( ierr );
		}

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );

		w.stop();
		w.read_duration();
	}
	catch( const std::exception &e )
	{
		cerr << e.what() << endl;
	}

	return 0;
}

