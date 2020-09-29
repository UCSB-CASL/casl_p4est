/**
 * Title: machine_learning_extrapolation
 * Description:
 * Author: Luis Ángel (임 영민)
 * Date Created: 09-29-2020
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#endif

#include <src/casl_geometry.h>

using namespace std;

int main(int argc, char** argv)
{
	// Prepare parallel enviroment.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );

	// Stopwatch.
	parStopWatch w;
	w.start( "Running example: machine_learning_extrapolation" );

	// p4est variables.
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *connectivity;
	my_p4est_brick_t brick;

	// Domain information.
	const int n_xyz[] = {1, 1, 1};
	const double xyz_min[] = {-1, -1, -1};
	const double xyz_max[] = {1, 1, 1};
	const int periodic[] = {0, 0, 0};
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Definining the non-signed distance level-set function to be reinitialized.
	geom::Plane plane( PointDIM( DIM( 1, 1, 1 ) ), PointDIM( DIM( 0.25, 0.25, 0.25 ) ) );
	splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, 6, &plane, 5 );

	// Create the forest using a level set as refinement criterion.
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
	p4est->user_pointer = (void *)( &levelSetSC );

	// Partition and refine forest.
	my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );
	my_p4est_partition( p4est, P4EST_TRUE, nullptr );

	// Create the ghost (cell) and node structures.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize the neighbor nodes structure.
	my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
	my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
	nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
										// how much we'll need the node neighbors.

	// A ghosted parallel PETSc vector to store level-set function values.
	Vec phi;
	PetscErrorCode ierr = VecCreateGhostNodes( p4est, nodes, &phi );
	CHKERRXX( ierr );

	// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
	sample_cf_on_nodes( p4est, nodes, plane, phi );

	// Reinitialize level-set using PDE-based approach.
	my_p4est_level_set_t ls( &nodeNeighbors );
	ls.reinitialize_2nd_order( phi, 10 );

	const double *phiReadPtr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	// save the grid into vtk
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							1, 0, "machine_learning_extrapolation",
							VTK_POINT_DATA, "phi", phiReadPtr );

	// Cleaning up.
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	ierr = VecDestroy( phi );
	CHKERRXX( ierr );

	// destroy the structures
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	my_p4est_brick_destroy( connectivity, &brick );

	w.stop();
	w.read_duration();
}

