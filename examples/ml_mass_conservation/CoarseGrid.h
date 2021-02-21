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

	double minCellWidth = 0;					// Minimum cell width.
	double minCellDiag = 0;						// Minimum call diagonal length.

	const double BAND;							// Minimum band around the interface.
	const int MAX_RL;							// Maximum refinement level.

	Vec phi = nullptr;							// Level-set function values parallel vector.
	Vec vel[P4EST_DIM] = {DIM( nullptr, nullptr, nullptr )};			// Velocity field parallel vectors.

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
	 * @param [in] velocityField Velocity object with as many components as velocities.
	 * @param [in] sampleVecs Whether or not to sample the level-set function and velocity field on the initial grid.
	 */
	CoarseGrid( const mpi_environment_t& mpi, const int nTreesPerDim[], const double xyzMin[], const double xyzMax[],
			 	const int periodic[], double band, int maxRL, CF_DIM *initialInterface,
			 	const CF_2 *velocityField[P4EST_DIM], bool sampleVecs=true )
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

		// Allocate parallel vectors for velocity field.
		for( auto& dir : vel )
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &dir );
			CHKERRXX( ierr );
		}

		// Retrieve grid size data.
		double dxyz[P4EST_DIM];
		get_dxyz_min( p4est, dxyz, minCellWidth, minCellDiag );

		// Finish up by sampling the level-set function at all independent nodes.
		if( sampleVecs )
		{
			// Sampling the level-set function.
			sample_cf_on_nodes( p4est, nodes, *initialInterface, phi );

			// Sampling the velocity field.
			for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
				sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );
		}
	}

	/**
	 * Use the FINE grid to "advect" or update the COARSE grid.  To achieve this, we use interpolation.  The COARSE grid
	 * is upated until convergence.
	 * @param [in] ngbd_f Pointer to FINE grid node neighborhood struct.
	 * @param [in] phi_f Reference to parallel level-set function value vector for FINE grid.
	 */
	void fitToFineGrid( const my_p4est_node_neighbors_t *ngbd_f, const Vec& phi_f )
	{
		assert( phi );			// Check we have a well defined coarse grid.
		PetscErrorCode ierr;

		///////////////////////////////////////////////// Preparation //////////////////////////////////////////////////

		// Compute second derivatives of FINE level-set function. We need these for quadratic interpolation its phi
		// values to draw the corresponding phi values in the COARSE grid.
		Vec phi_f_xx[P4EST_DIM];
		for( auto& derivative : phi_f_xx )
		{
			ierr = VecCreateGhostNodes( ngbd_f->get_p4est(), ngbd_f->get_nodes(), &derivative );
			CHKERRXX( ierr );
		}
		ngbd_f->second_derivatives_central( phi_f, DIM( phi_f_xx[0], phi_f_xx[1], phi_f_xx[2] ) );

		// Save the old splitting criteria information; we need to restore it once the COARSE grid converges.
		auto *oldSplittingCriteria = (splitting_criteria_t*) p4est->user_pointer;

		// Define splitting criteria.
		auto *splittingCriteriaBandPtr = new splitting_criteria_band_t( oldSplittingCriteria->min_lvl,
																  		oldSplittingCriteria->max_lvl,
																  		oldSplittingCriteria->lip, BAND );

		// New grid level-set values: start from current COARSE grid structure.
		Vec phiNew;
		ierr = VecCreateGhostNodes( p4est, nodes, &phiNew );	// Notice p4est and nodes are the grid at time n so far.
		CHKERRXX( ierr );

		///////////////////////////////////////// Update grid until convergence ////////////////////////////////////////

		bool isGridChanging = true;
		while( isGridChanging )
		{
			double *phiNewPtr;
			ierr = VecGetArray( phiNew, &phiNewPtr );
			CHKERRXX( ierr );

			// Use FINE grid to "advect" COARSE grid by using interpolation.
			// Interpolation object based on FINE grid.  It's used to find the values at the nodes of COARSE grid.
			my_p4est_interpolation_nodes_t interp( ngbd_f );
			for( size_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( n, p4est, nodes, xyz );
				interp.add_point( n, xyz );
			}
			interp.set_input( phi_f, DIM( phi_f_xx[0], phi_f_xx[1], phi_f_xx[2] ), interpolation_method::quadratic );
			interp.interpolate( phiNewPtr );
			interp.clear();

			// Refine an coarsen COARSE grid; detect if it changes from previous coarsening/refinement operation.
			isGridChanging = splittingCriteriaBandPtr->refine_and_coarsen_with_band( p4est, nodes, phiNewPtr );

			ierr = VecRestoreArray( phiNew, &phiNewPtr );
			CHKERRXX( ierr );

			if( isGridChanging )
			{
				// Repartition grid as it changed.
				my_p4est_partition( p4est, P4EST_TRUE, nullptr );

				// Reset nodes, ghost, and phi.
				p4est_ghost_destroy( ghost );
				ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
				p4est_nodes_destroy( nodes );
				nodes = my_p4est_nodes_new( p4est, ghost );

				// Allocate vector for new level-set values for most up-to-date grid.
				ierr = VecDestroy( phiNew );
				CHKERRXX( ierr );
				ierr = VecCreateGhostNodes( p4est, nodes, &phiNew );
				CHKERRXX( ierr );
			}
		}

		///////////////////////////////////////////////// Finishing up /////////////////////////////////////////////////

		p4est->user_pointer = (void *) oldSplittingCriteria;

		ierr = VecDestroy( phi );	// Update old COARSE phi to new level-set values.
		CHKERRXX( ierr );
		phi = phiNew;

		// Free vectors with second derivatives.
		for( auto& derivative : phi_f_xx )
		{
			ierr = VecDestroy( derivative );
			CHKERRXX( ierr );
		}

		delete splittingCriteriaBandPtr;

		// Rebuilding COARSE hierarchy and neighborhoods from updated p4est, nodes, and ghost structs.
		delete hierarchy;
		delete nodeNeighbors;
		hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
		nodeNeighbors->init_neighbors();
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

		for( auto& dir : vel )
		{
			if( dir )
			{
				ierr = VecDestroy( dir );
				CHKERRXX( ierr );
				dir = nullptr;
			}
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
