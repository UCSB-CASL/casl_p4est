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
#include <src/my_p4est_semi_lagrangian_ml.h>

/**
 * Auxiliary class to handle all procedures involving the coarse grid, which is to be sampled and then updated by using
 * a finer grid from where its values are drawn by interpolation.
 */
class CoarseGrid
{
public:
	const mpi_environment_t& mpi;				// Reference to MPI environment.
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
	Vec gammaFlag = nullptr;					// A flag vector that stores 1s in nodes adjacent to Gamma, 0 otherwise.

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
	 * @param [in] velocityField Velocity field given as P4EST_DIM CF_DIM components (different to RandomVelocityField).
	 * @param [in] sampleVecs Whether or not to sample the level-set function and velocity field on the initial grid.
	 */
	CoarseGrid( const mpi_environment_t& mpi, const int nTreesPerDim[], const double xyzMin[], const double xyzMax[],
			 	const int periodic[], double band, int maxRL, CF_DIM *initialInterface,
			 	const CF_2 *velocityField[P4EST_DIM]=nullptr, bool sampleVecs=true )
			 	: BAND( band ), MAX_RL( maxRL ), mpi( mpi )
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

			// Sampling the velocity field if given.
			if( velocityField )
			{
				for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
					sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );
			}
		}
	}

	/**
	 * Use the FINE grid to "advect" or update the COARSE grid.  To achieve this, we use interpolation.  The COARSE grid
	 * is upated until convergence.
	 * @note You must collect samples before "advecting" COARSE grid.
	 * @param [in] ngbd_f Pointer to FINE grid node neighborhood struct.
	 * @param [in] phi_f Reference to parallel level-set function value vector for FINE grid.
	 * @param [in] velocityField Velocity field given as P4EST_DIM CF_DIM components (different to RandomVelocityField).
	 */
	void fitToFineGrid( const my_p4est_node_neighbors_t *ngbd_f, const Vec& phi_f,
					 	const CF_2 *velocityField[P4EST_DIM]=nullptr )
	{
		assert( phi );			// Check we have a well defined coarse grid.
		PetscErrorCode ierr;

		// Invalidate the flag vector.  You must set re-allocate the vector when sampling.
		if( gammaFlag )
		{
			ierr = VecDestroy( gammaFlag );
			CHKERRXX( ierr );
			gammaFlag = nullptr;
		}

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

		// Reconstruct the COARSE velocity vectors on new grid.  Resample if CF_DIM objects were given.
		for( int dir = 0; dir < P4EST_DIM; dir++ )
		{
			ierr = VecDestroy( vel[dir] );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &vel[dir] );
			CHKERRXX( ierr );

			if( velocityField )
				sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );
		}

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
	 * Collect sample from COARSE grid points next to the interface (all independent nodes).  This function must be
	 * called *after* advecting the FINE grid an adequate number of times and *before* using the FINE grid to "advect"
	 * the COARSE grid.
	 * @param [in] ngbd_f Advected fine grid neighborhood struct.
	 * @param [in] phi_f Advected fine grid level-set values vector.
	 * @param [in] dt Coarse grid step size.
	 * @return true if all nodes along the interface were backtracked within the domain; false otherwise.
	 */
	bool collectSamples( const my_p4est_node_neighbors_t *ngbd_f, const Vec& phi_f, const double& dt )
	{
		assert( phi );	// Check we have a well defined coarse grid.
		PetscErrorCode ierr;

		///////////////////////////////////////////////// Preparation //////////////////////////////////////////////////

		// Invalidate the flag vector and reallocate it with the current grid status.
		if( gammaFlag )
		{
			ierr = VecDestroy( gammaFlag );
			CHKERRXX( ierr );
			gammaFlag = nullptr;
		}
		ierr = VecCreateGhostNodes( p4est, nodes, &gammaFlag );
		CHKERRXX( ierr );

		// Compute second derivatives of FINE level-set function. We need these for quadratic interpolation of its phi
		// values to draw the corresponding target phi values in the COARSE grid.
		Vec phi_f_xx[P4EST_DIM];
		for( auto& derivative : phi_f_xx )
		{
			ierr = VecCreateGhostNodes( ngbd_f->get_p4est(), ngbd_f->get_nodes(), &derivative );
			CHKERRXX( ierr );
		}
		ngbd_f->second_derivatives_central( phi_f, DIM( phi_f_xx[0], phi_f_xx[1], phi_f_xx[2] ) );

		my_p4est_interpolation_nodes_t interp( ngbd_f );	// Interpolation object on FINE grid phi values.

		//////////////////////////////////////////////// Data collection ///////////////////////////////////////////////

		char msg[1024];
		slml::SemiLagrangian semiLagrangianML( &p4est, &nodes, &ghost, nodeNeighbors );		// Use a semi-Lagrangian
		std::vector<slml::DataPacket *> dataPackets;										// scheme with a single vel
		bool allInside = semiLagrangianML.collectSamples( vel, dt, phi, dataPackets );		// step for backtracing to
		sprintf( msg, "* %lu received packets!\n", dataPackets.size() );					// retrieve samples.
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

		// Finding the target phi values via interpolation from FINE grid.
		double targetPhi[dataPackets.size()];
		size_t outIndex;
		for( outIndex = 0; outIndex < dataPackets.size(); outIndex++ )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( dataPackets[outIndex]->nodeIdx, p4est, nodes, xyz );
			interp.add_point( outIndex, xyz );
		}
		interp.set_input( phi_f, DIM( phi_f_xx[0], phi_f_xx[1], phi_f_xx[2] ), interpolation_method::quadratic );
		interp.interpolate( targetPhi );
		interp.clear();

		// TODO: Go through collected nodes and add the target value.  Populate the flag vector at the same time.
		double *gammaFlagPtr;
		ierr = VecGetArray( gammaFlag, &gammaFlagPtr );
		CHKERRXX( ierr );
		for( auto dataPacket : dataPackets )
			gammaFlagPtr[dataPacket->nodeIdx] = 1.0;			// Turn on "bit" for node next to Gamma.
		ierr = VecRestoreArray( gammaFlag, &gammaFlagPtr );
		CHKERRXX( ierr );

		// Let's synchronize the flag vector among all processes.  This way, we know which nodes will be updated using
		// machine learning even at processes that do not own those nodes.  If I don't do this, the flag set in one pro-
		// cess can be reset to 0 by another (e.g., if the node is in the ghost layer of a process and, according to it,
		// the node isn't next to the interface, while the owner process has determined that the node is next to Gamma).
		ierr = VecGhostUpdateBegin( gammaFlag, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
		VecGhostUpdateEnd( gammaFlag, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );

		///////////////////////////////////////////////// Finishing up /////////////////////////////////////////////////

		slml::SemiLagrangian::freeDataPacketArray( dataPackets );

		// Free vectors with second derivatives for FINE phi.
		for( auto& derivative : phi_f_xx )
		{
			ierr = VecDestroy( derivative );
			CHKERRXX( ierr );
		}

		return allInside;	// True if all backtracked points along the interface fell within domain.
	}

	/**
	 * Write VTK files for prior or post advection.  Prior saves the flagged nodes along Gamma, post saves the exact
	 * solution phi.
	 * @param [in] vtkIdx File index.
	 * @param [in] phiExact Exact phi parallel vector.  If given, post advection is saved, otherwise, prior advection is saved.
 	 */
	void writeVTK( int vtkIdx, Vec phiExact=nullptr )
	{
		char name[1024];
		PetscErrorCode ierr;

		const double *phiReadPtr, *phiExactOrFlagReadPtr, *velReadPtr[2];		// Pointers to Vec contents.

		// Depending on whether the flag vector is set (after sampling) or not (after fitting COARSE to FINE grid), we
		// save different information.
		// Prior: Means that we have not advected the COARSE grid.  So, we can show the flagged nodes.
		// Post: Means that we have advected the coarse grid.  We can show the exact solution (as it counts on t=tn+1).
		sprintf( name, (!phiExact)? "visualization_prior_%d" : "visualization_post_%d", vtkIdx );
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( (!phiExact)? gammaFlag : phiExact, &phiExactOrFlagReadPtr );
		CHKERRXX( ierr );
		for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecGetArrayRead( vel[dir], &velReadPtr[dir] );
			CHKERRXX( ierr );
		}
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								2 + P4EST_DIM, 0, name,
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, (!phiExact)? "flag" : "phiExact", phiExactOrFlagReadPtr,
								VTK_POINT_DATA, "vel_x", velReadPtr[0],
								VTK_POINT_DATA, "vel_y", velReadPtr[1]
		);
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( (!phiExact)? gammaFlag : phiExact, &phiExactOrFlagReadPtr );
		CHKERRXX( ierr );
		for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecRestoreArrayRead( vel[dir], &velReadPtr[dir] );
			CHKERRXX( ierr );
		}

		char msg[1024];
		sprintf( msg, " -> Saving vtu files in %s.vtu\n", name );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );
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

		if( gammaFlag )
		{
			ierr = VecDestroy( gammaFlag );
			CHKERRXX( ierr );
			gammaFlag = nullptr;
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
