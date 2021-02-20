//
// Created by Im YoungMin on 2/19/21.
//

#ifndef ML_MASS_CONSERVATION_COARSEGRID_H
#define ML_MASS_CONSERVATION_COARSEGRID_H

#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_semi_lagrangian_ml.h>

/**
 * Auxiliary class to handle all procedures involving the coarse grid, which is to be sampled and then updated by using
 * a finer grid from where its values are drawn by interpolation.
 */
class CoarseGrid
{
public:
	my_p4est_brick_t brick{};					// Brick and connectivity objects.
	p4est_connectivity_t *connectivity;
	p4est_t *p4est;								// Forest variables.
	p4est_ghost_t *ghost;
	p4est_nodes_t *nodes;
	my_p4est_hierarchy_t *hierarchy;			// Neighborhood-node structures.
	my_p4est_node_neighbors_t *nodeNeighbors;

	const double BAND;							// Minimum band around the interface.
	const int MAX_RL;							// Maximum refinement level.

	Vec phi = nullptr;							// Level-set function values parallel vector.

	splitting_criteria_cf_and_uniform_band_t *lsSplittingCriteria;		// Criteria created dynamically in constructor.

	/**
	 * Construct the grid and, optionally, sample the level-set field into the parallel internal phi vector.
	 * @param [in] mpi Parallel MPI object reference.
	 * @param [in] nTreesPerDim Number of trees per dimension.
	 * @param [in] xyzMin Domain minimum coordinates.
	 * @param [in] xyzMax Domain maximum coordinates.
	 * @param [in] periodic Domain periodicity.
	 * @param [in] band Minimum number of (min) cells around the interface.
	 * @param [in] maxRL Maximum refinement level.
	 * @param [in] initialInterface Object to define the initial interface and the initial splitting criteria.
	 * @param [in] samplePhi Whether or not sample the level-set function on the initial interface.
	 */
	CoarseGrid( const mpi_environment_t& mpi, const int nTreesPerDim[], const double xyzMin[], const double xyzMax[],
			 	const int periodic[], double band, int maxRL, CF_DIM *initialInterface, bool samplePhi=true )
			 	: BAND( band ), MAX_RL( maxRL )
	{

		// Init macromesh via the brick and connectivity objects.
		connectivity = my_p4est_brick_new( nTreesPerDim, xyzMin, xyzMax, &brick, periodic );

		// Create the forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		lsSplittingCriteria = new splitting_criteria_cf_and_uniform_band_t( 1, MAX_RL, initialInterface, BAND );
		p4est->user_pointer = lsSplittingCriteria;		// Don't forget to delete object manually.

		// Refine and partition forest (according to the 'grid_update' example, I shouldn't use recursive refinement).
		for( int i = 0; i < MAX_RL; i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor node structure.
		hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );		// Don't forget to delete objects manually.
		nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
		nodeNeighbors->init_neighbors();

		// Allocate parallel vector for level-set function values.
		PetscErrorCode ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );

		// Finish up by sampling the level-set function at all independent nodes.
		if( samplePhi )
			sample_cf_on_nodes( p4est, nodes, *initialInterface, phi );
	}

	/**
	 * End-of life function to release resources and destroy parallel objects created by this class instance.
	 */
	void destroy()
	{
		PetscErrorCode ierr;

		if( phi )
		{
			ierr = VecDestroy( phi );
			CHKERRXX( ierr );
			phi = nullptr;			// Taking precautions to nullify anything that is dynamically created/deleted.
		}

		if( lsSplittingCriteria )
		{
			delete lsSplittingCriteria;
			lsSplittingCriteria = nullptr;
		}

		if( hierarchy )
		{
			delete hierarchy;
			hierarchy = nullptr;
		}

		if( nodeNeighbors )
		{
			delete nodeNeighbors;
			nodeNeighbors = nullptr;
		}

		if( nodes )
		{
			p4est_nodes_destroy( nodes );
			nodes = nullptr;
		}

		if( ghost )
		{
			p4est_ghost_destroy( ghost );
			ghost = nullptr;
		}

		if( p4est )
		{
			p4est_destroy( p4est );
			p4est = nullptr;
		}

		// Destroy the dynamically allocated brick and connectivity structures.  Connectivity and Brick objects are the
		// only ones that are not re-created in every iteration of semi-Lagrangian advection.
		if( connectivity )
		{
			my_p4est_brick_destroy( connectivity, &brick );
			connectivity = nullptr;
		}
	}

};


#endif //ML_MASS_CONSERVATION_COARSEGRID_H
