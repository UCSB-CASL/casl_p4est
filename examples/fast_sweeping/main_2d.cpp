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
		splitting_criteria_cf_t levelSetSC( cmd.get( "lmin", 2 ), cmd.get( "lmax", 4 ), &circle );

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

		// Calculate L^1 norm from each independent node (i.e. locally owned and ghost nodes) to the bottom left reference node.
		Vec l1Norm_1;
		ierr = VecCreateGhostNodes( p4est, nodes, &l1Norm_1 );
		CHKERRXX( ierr );

		double *l1NormPtr_1;
		ierr = VecGetArray( l1Norm_1, &l1NormPtr_1 );
		CHKERRXX( ierr );

		for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( i, p4est, nodes, xyz );
			double diff[P4EST_DIM] = { xyz[0] - xyz_min[0], xyz[1] - xyz_min[1] };
			l1NormPtr_1[i] = compute_L1_norm( diff, P4EST_DIM );
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
								2, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiPtr,
								VTK_POINT_DATA, "L1_1", l1NormPtr_1 );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( l1Norm_1, &l1NormPtr_1 );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( l1Norm_1 );
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

