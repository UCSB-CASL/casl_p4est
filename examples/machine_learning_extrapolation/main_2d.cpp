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

/**
 * Scalar field to extend over the interface and into Omega+.
 * Notice that it has the same structure as the level-set function because we want to evaluate at the nodes.
 */
class Field: public CF_2
{
public:
	/**
	 * The scalar function to extend: f(x,y).
	 * @param [in] x Query point x-coordinate.
	 * @param [in] y Query point y-coordinate.
	 * @return f(x,y).
	 */
	double operator()( double x, double y ) const override
	{
		return sin( M_PI * x ) * cos( M_PI * y );
	}
};

int main(int argc, char** argv)
{
	const double MIN_D = -1;				// Minimum value for domain (in x, y, and z).  Domain is symmetric.
	const int NUM_TREES_PER_DIM = 2;		// Number of trees per dimension.
	const int REFINEMENT_MAX_LEVEL = 6;		// Maximum level of refinement.
	const int REFINEMENT_BAND_WIDTH = 5;	// Band around interface for grid refinement.
	const int REINIT_NUM_ITER = 10;			// Number of iterations to solve PDE for reinitialization.
	const int EXTENSION_NUM_ITER = 50;		// Number of iterations to solve PDE for extrapolation.
	const int EXTENSION_ORDER = 2;			// Order of extrapolation (0: constant, 1: linear, 2: quadratic).
	const int EXTENSION_BAND_WIDTH = 2;		// Band around interface (in diagonal lengths) to check for extension accuracy.

	// Prepare parallel enviroment, although we enforce just a single processor to avoid race conditions when generating
	// datasets.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );
	if( mpi.rank() > 1 )
		throw std::runtime_error( "Only a single process is allowed!" );

	// Stopwatch.
	parStopWatch watch;
	watch.start();

	// p4est variables.
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *connectivity;
	my_p4est_brick_t brick;

	// Domain information.
	const int n_xyz[] = { NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM };
	const double xyz_min[] = { MIN_D, MIN_D, MIN_D };
	const double xyz_max[] = { -MIN_D, -MIN_D, -MIN_D };
	const int periodic[] = {0, 0, 0};
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Definining the non-signed distance level-set function to be reinitialized.
	geom::Plane plane( PointDIM( DIM( 1, 1, 1 ) ), PointDIM( DIM( 0.25, 0.25, 0.25 ) ) );
	splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, REFINEMENT_MAX_LEVEL, &plane, REFINEMENT_BAND_WIDTH );

	// Scalar field to extend.
	Field field;

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

	// Smallest quadrant features.
	double dxyz[P4EST_DIM]; 			// Dimensions.
	double dxyzMin;        				// Minimum side length of the smallest quadrant.
	double diagMin;        				// Diagonal length of the smallest quadrant.
	get_dxyz_min( p4est, dxyz, dxyzMin, diagMin );

	// A ghosted parallel PETSc vector to store level-set function values.
	Vec phi;
	PetscErrorCode ierr = VecCreateGhostNodes( p4est, nodes, &phi );
	CHKERRXX( ierr );

	// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
	sample_cf_on_nodes( p4est, nodes, plane, phi );

	// Reinitialize level-set using PDE-based approach.
	my_p4est_level_set_t ls( &nodeNeighbors );
	ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

	const double *phiReadPtr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	// Vectors to store the scalar function exactly and numerically extrapolated.
	Vec extField, exactField;
	ierr = VecDuplicate( phi, &extField );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi, &exactField );
	CHKERRXX( ierr );

	sample_cf_on_nodes( p4est, nodes, field, exactField );		// Now we have the field evaluated exactly at each node.
	ierr = VecCopyGhost( exactField, extField );
	CHKERRXX( ierr );

	// Reset field for phi > 0 (i.e. for nodes in Omega+).
	double *extFieldPtr;
	ierr = VecGetArray( extField, &extFieldPtr );
	CHKERRXX( ierr );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
	{
		if( phiReadPtr[n] > 0 )
			extFieldPtr[n] = 0;
	}

	// Perform extrapolation using all derivatives (from Daniil's paper).
	ls.extend_Over_Interface_TVD_Full( phi, extField, EXTENSION_NUM_ITER, EXTENSION_ORDER );

	// Calculate extrapolation errors in a band around Gamma, within Omega+.
	Vec errorExtField;			// Extension error parallel vector.
	ierr = VecDuplicate( phi, &errorExtField );
	CHKERRXX( ierr );

	double *errorExtFieldPtr;
	ierr = VecGetArray( errorExtField, &errorExtFieldPtr );
	CHKERRXX( ierr );

	const double *exactFieldReadPtr;
	ierr = VecGetArrayRead( exactField, &exactFieldReadPtr );
	CHKERRXX( ierr );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
	{
		if( phiReadPtr[n] > 0 && phiReadPtr[n] < EXTENSION_BAND_WIDTH * diagMin )
			errorExtFieldPtr[n] = ABS( exactFieldReadPtr[n] - extFieldPtr[n] );
	}

	// Save the grid into vtk
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							4, 0, "machine_learning_extrapolation",
							VTK_POINT_DATA, "phi", phiReadPtr,
							VTK_POINT_DATA, "exactField", exactFieldReadPtr,
							VTK_POINT_DATA, "extField", extFieldPtr,
							VTK_POINT_DATA, "errorExtField", errorExtFieldPtr );

	// Cleaning up.
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );					// Restoring vector pointers.
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( exactField, &exactFieldReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArray( extField, &extFieldPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArray( errorExtField, &errorExtFieldPtr );
	CHKERRXX( ierr );

	ierr = VecDestroy( phi );										// Freeing memory.
	CHKERRXX( ierr );
	ierr = VecDestroy( exactField );
	CHKERRXX( ierr );
	ierr = VecDestroy( extField );
	CHKERRXX( ierr );
	ierr = VecDestroy( errorExtField );
	CHKERRXX( ierr );

	// Destroy the structures.
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	my_p4est_brick_destroy( connectivity, &brick );

	watch.stop();
	watch.read_duration();
}

