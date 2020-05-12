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
#include <src/my_p8est_fast_sweeping.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_fast_sweeping.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

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

		Sphere sphere( DIM( 0, 0, 0 ), 0.3 );
		splitting_criteria_cf_t levelSetSC( cmd.get( "lmin", 1 ), cmd.get( "lmax", 5 ), &sphere );

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
		nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given how much
											// we'll neeed the node neighbors.

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
			phiPtr[i] = sphere( DIM( xyz[0], xyz[1], xyz[2] ) );
		}
		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );

		// Determining the node types.
		Vec nodeType;
		ierr = VecCreateGhostNodes( p4est, nodes, &nodeType );
		CHKERRXX( ierr );
		double *nodeTypePtr;
		ierr = VecGetArray( nodeType, &nodeTypePtr );
		CHKERRXX( ierr );
		for( size_t i = 0; i < nodes->num_owned_indeps; i++ )
			nodeTypePtr[i] = 0.0;
		for( size_t g = 0; g < ( nodes->indep_nodes.elem_count - nodes->num_owned_indeps); g++ )
    		nodeTypePtr[nodes->num_owned_indeps + g] = 1.0 * pow( 3, mpi.rank() );

		VecGhostUpdateBegin( nodeType, ADD_VALUES, SCATTER_REVERSE );
		VecGhostUpdateEnd( nodeType, ADD_VALUES, SCATTER_REVERSE );
		VecGhostUpdateBegin( nodeType, INSERT_VALUES, SCATTER_FORWARD );
		VecGhostUpdateEnd( nodeType, INSERT_VALUES, SCATTER_FORWARD );

		PetscSynchronizedPrintf( mpi.comm(), "Process %d indep_nodes = %d, num_owned_indeps = %d, num_owned_shared = %d\n",
				mpi.rank(), nodes->indep_nodes.elem_count, nodes->num_owned_indeps, nodes->num_owned_shared );
		PetscSynchronizedFlush( mpi.comm(), PETSC_STDOUT );

		// Finding out which ghost nodes should be discarded given that they do not have a well defined neighborhood to
		// compute their finite differences.
		Vec badNode;
		ierr = VecCreateGhostNodes( p4est, nodes, &badNode );
		CHKERRXX( ierr );
		double *badNodePtr;
		ierr = VecGetArray( badNode, &badNodePtr );
		CHKERRXX( ierr );
		for( size_t i = 0; i < nodes->num_owned_indeps; i++ )
			badNodePtr[i] = 0;
		for( size_t g = 0; g < ( nodes->indep_nodes.elem_count - nodes->num_owned_indeps); g++ )
		{
			try
			{
				const quad_neighbor_nodes_of_node_t *qnnnPtr;							// Works only in DEBUG mode; detecting ghost nodes without
				nodeNeighbors.get_neighbors( nodes->num_owned_indeps + g, qnnnPtr );	// a well-defined neighborhood.
				badNodePtr[nodes->num_owned_indeps + g] = 0;
			}
			catch( std::exception& e )
			{
				badNodePtr[nodes->num_owned_indeps + g] = 1.0 * pow( 3, mpi.rank() );
			}
		}

		VecGhostUpdateBegin( badNode, ADD_VALUES, SCATTER_REVERSE );
		VecGhostUpdateEnd( badNode, ADD_VALUES, SCATTER_REVERSE );
		VecGhostUpdateBegin( badNode, INSERT_VALUES, SCATTER_FORWARD );
		VecGhostUpdateEnd( badNode, INSERT_VALUES, SCATTER_FORWARD );

		// Determining the process that owns each independent node.
		Vec process;
		ierr = VecCreateGhostNodes( p4est, nodes, &process );
		CHKERRXX( ierr );
		double *processPtr;
		ierr = VecGetArray( process, &processPtr );
		CHKERRXX( ierr );
		for( size_t i = 0; i < nodes->num_owned_indeps; i++ )
			processPtr[i] = mpi.rank();
		VecGhostUpdateBegin( process, INSERT_VALUES, SCATTER_FORWARD );
		VecGhostUpdateEnd( process, INSERT_VALUES, SCATTER_FORWARD );

		/// Testing the fast sweeping algorithm ///
		double *fsmPhiPtr;
		const double *phiReadPtr;

		Vec fsmPhi;													// Save here the solution from FSM.
		ierr = VecCreateGhostNodes( p4est, nodes, &fsmPhi );
		CHKERRXX( ierr );

		ierr = VecGetArray( fsmPhi, &fsmPhiPtr );					// Copy current phi values into FSM parallel vector.
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		std::copy( phiReadPtr, phiReadPtr + nodes->indep_nodes.elem_count, fsmPhiPtr );
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( fsmPhi, &fsmPhiPtr );
		CHKERRXX( ierr );

		parStopWatch fsmReinitTimer;
		fsmReinitTimer.start( "Fast sweeping reinitialization" );

		FastSweeping fsm;
		fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
		fsm.reinitializeLevelSetFunction( &fsmPhi, 4 );

		fsmReinitTimer.stop();
		fsmReinitTimer.read_duration();

		const double *fsmPhiReadPtr;
		ierr = VecGetArrayRead( fsmPhi, &fsmPhiReadPtr );
		CHKERRXX( ierr );

/*		if( p4est->mpirank == 0 )
		{
			for( size_t i= 0; i < nodes->indep_nodes.elem_count; i++ )		// Check that we got all data.
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( i, p4est, nodes, xyz );
				cout << i << ": (" << xyz[0] << ", " << xyz[1] << ")  " << fsmPhiReadPtr[i] << endl;
			}
		}*/

		/// Collect the absolute error between fast sweeping reinitialization and exact distance ///
		Vec fsmError;
		ierr = VecDuplicate( fsmPhi, &fsmError );
		CHKERRXX( ierr );

		double *fsmErrorPtr;
		ierr = VecGetArray( fsmError, &fsmErrorPtr );
		CHKERRXX( ierr );
		for( size_t i = 0; i < nodes->num_owned_indeps; i++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( i, p4est, nodes, xyz );
			fsmErrorPtr[i] = ABS( sphere( DIM( xyz[0], xyz[1], xyz[2] ) ) - fsmPhiReadPtr[i] );
		}
		VecGhostUpdateBegin( fsmError, INSERT_VALUES, SCATTER_FORWARD );
		VecGhostUpdateEnd( fsmError, INSERT_VALUES, SCATTER_FORWARD );

		/// Reinitialize the level-set function values using the transient pseudo-temporal equation ///
		parStopWatch temporalReinitTimer;
		temporalReinitTimer.start( "Temporal reinitialization" );

		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, 5 );			// Using x iterations.

		temporalReinitTimer.stop();
		temporalReinitTimer.read_duration();

		std::ostringstream oss;
		oss << "fsm_" << mpi.size() << "_" << P4EST_DIM;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								6, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "fsmPhi", fsmPhiReadPtr,
								VTK_POINT_DATA, "fsmError", fsmErrorPtr,
								VTK_POINT_DATA, "nodeType", nodeTypePtr,
								VTK_POINT_DATA, "badNode", badNodePtr,
								VTK_POINT_DATA, "process", processPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArrayRead( fsmPhi, &fsmPhiReadPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( fsmError, &fsmErrorPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( nodeType, &nodeTypePtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( badNode,&badNodePtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( process, &processPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( fsmPhi );
		CHKERRXX( ierr );

		ierr = VecDestroy( fsmError );
		CHKERRXX( ierr );

		ierr = VecDestroy( nodeType );
		CHKERRXX( ierr );

		ierr = VecDestroy( badNode );
		CHKERRXX( ierr );

		ierr = VecDestroy( process );
		CHKERRXX( ierr );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );
	}
	catch( const std::exception &e )
	{
		cerr << e.what() << endl;
	}

	return 0;
}

