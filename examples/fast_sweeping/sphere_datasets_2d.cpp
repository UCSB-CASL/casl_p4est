/**
 * Generate datasets for training a feedforward neural network on a two-dimensional circular interface.
 *
 * Developer: Luis Ángel.
 * Date: May 12, 2020.
 */

// System.
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
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <cassert>

int main ( int argc, char* argv[] )
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
		const unsigned short MIN_LEVEL = 1;							// Minimum level of tree refinement.
		const unsigned short MAX_LEVEL = 5;							// Maximum level of tree refinement.
		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the level-set function and its signed-distance function version for error testing.
		const double RADIUS = 0.3;
		const double X0 = 0, Y0 = 0, Z0 = 0;
		geom::Sphere sphere( DIM( X0, Y0, Z0 ), RADIUS );			// Signed distance function.
		geom::SphereNSD sphereNsd( DIM( X0, Y0, Z0 ), RADIUS );		// Non-signed distance function to reinitialize.
		splitting_criteria_cf_t levelSetSC( MIN_LEVEL, MAX_LEVEL, &sphereNsd );

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
			phiPtr[i] = sphereNsd( DIM( xyz[0], xyz[1], xyz[2] ) );		// Using the non-signed distance function.
		}
		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );

		PetscSynchronizedPrintf( mpi.comm(), ">> Process %d indep_nodes = %d, num_owned_indeps = %d, num_owned_shared = %d\n",
								 mpi.rank(), nodes->indep_nodes.elem_count, nodes->num_owned_indeps, nodes->num_owned_shared );
		PetscSynchronizedFlush( mpi.comm(), PETSC_STDOUT );

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
		fsmReinitTimer.start( "FSM reinitialization" );

		FastSweeping fsm;
		fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
		fsm.reinitializeLevelSetFunction( &fsmPhi, 8 );

		fsmReinitTimer.stop();
		fsmReinitTimer.read_duration();

		const double *fsmPhiReadPtr;
		ierr = VecGetArrayRead( fsmPhi, &fsmPhiReadPtr );
		CHKERRXX( ierr );

		/// Obtaining location of locally owned nodes for debugging ///
		std::vector<p4est_locidx_t> stencil;			// In 3D, node 4723 has full uniform stencil.  In 2D: 291.
		getFullStencilOfNode( 291, &nodeNeighbors, nodes, stencil, 2. / ( 1u << MAX_LEVEL ) );

		for( p4est_locidx_t n : stencil )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			std::cout << n << ": (" << xyz[0] << ", " << xyz[1] ONLY3D( << ", " << xyz[2] ) << ")" << std::endl;
		}

		/// Reinitialize the level-set function values using the transient PDE-based equation ///
		parStopWatch pdeReinitTimer;
		pdeReinitTimer.start( "PDE reinitialization" );

		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, 5 );			// Using x iterations.

		pdeReinitTimer.stop();
		pdeReinitTimer.read_duration();

		/// Collect the absolute error between PDE-based reinitialization and exact distance ///
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		double pdeMAE = 0;								// Mean absolute error for PDE-based reinit for current partition.
		for( size_t i = 0; i < nodes->num_owned_indeps; i++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( i, p4est, nodes, xyz );
			pdeMAE += ABS( sphere( DIM( xyz[0], xyz[1], xyz[2] ) ) - phiReadPtr[i] );
		}
		pdeMAE /= nodes->num_owned_indeps;

		PetscSynchronizedPrintf( mpi.comm(), ">> Process %d: PDE MAE = %f\n",
								 mpi.rank(), pdeMAE );
		PetscSynchronizedFlush( mpi.comm(), PETSC_STDOUT );

		std::ostringstream oss;
		oss << "fsm_" << mpi.size() << "_" << P4EST_DIM;
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								3, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "fsmPhi", fsmPhiReadPtr,
								VTK_POINT_DATA, "process", processPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArrayRead( fsmPhi, &fsmPhiReadPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( process, &processPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( fsmPhi );
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
		std::cerr << e.what() << std::endl;
	}

	return 0;
}