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

#include <random>

/**
 * Auxiliary class to handle all procedures involving the coarse grid, which is to be sampled and then updated by using
 * a reference finer grid from where its values are drawn by interpolation.
 *
 * Updated: September 30, 2021.
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
	double minCoords[P4EST_DIM] = {};			// Minimum coordinates of computational domain.
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
	 * @param [in] maxRL Maximum refinement level.
	 * @param [in] initialInterface Object to define the initial interface and the initial splitting criteria.
	 * @param [in] velocityField Velocity field given as P4EST_DIM CF_DIM components (different to RandomVelocityField).
	 * @param [in] sampleVecs Whether or not to sample the level-set function and velocity field on the initial grid.
	 */
	CoarseGrid( const mpi_environment_t& mpi, const int nTreesPerDim[], const double xyzMin[], const double xyzMax[],
			 	const int periodic[], int maxRL, CF_DIM *initialInterface,
			 	const CF_2 *velocityField[P4EST_DIM]=nullptr, bool sampleVecs=true )
			 	: MAX_RL( maxRL ), mpi( mpi )
	{

		// Init macromesh via the brick and connectivity objects.
		connectivity = my_p4est_brick_new( nTreesPerDim, xyzMin, xyzMax, &brick, periodic );

		// Create the forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		lsSplittingCriteria = new splitting_criteria_cf_and_uniform_band_t( 1, MAX_RL, initialInterface, MASS_BAND_HALF_WIDTH );
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

		// Minimum coordinates of computational domain.
		minCoords[0] = xyzMin[0];
		minCoords[1] = xyzMin[1];
		ONLY3D(minCoords[2] = xyzMin[2]);

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
	 * @note You must collect samples before "advecting" the COARSE grid.
	 * @param [in] ngbd_f Pointer to FINE grid node neighborhood struct.
	 * @param [in] phi_f Reference to parallel level-set function value vector for FINE grid.
	 * @param [in] velocityField Optional velocity field given as P4EST_DIM CF_DIM components (different to RandomVelocityField).
	 */
	void fitToFineGrid( const my_p4est_node_neighbors_t *ngbd_f, const Vec& phi_f,
					 	const CF_2 *velocityField[P4EST_DIM]=nullptr )
	{
		assert( phi );			// Check we have a well defined coarse grid.
		PetscErrorCode ierr;

		///////////////////////////////////////////////// Preparation //////////////////////////////////////////////////

		// Compute second derivatives of FINE level-set function. We need these for quadratically interpolating its phi
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
																  		oldSplittingCriteria->lip, MASS_BAND_HALF_WIDTH );

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
			for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
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
	}

	/**
	 * Advect coarse grid numerically, following the semi-Lagrangian method with a band and second-order accurate loca-
	 * tion of departure points.
	 * @note You must collect samples before "advecting" the COARSE grid.
	 * @param [in] dt Timestep.
	 * @param [in] velocityField Optional velocity field given as P4EST_DIM CF_DIM components (different to RandomVelocityField).
	 */
	void updateP4EST( const double& dt, const CF_2 *velocityField[P4EST_DIM]=nullptr )
	{
		assert( phi );			// Check we have a well defined coarse grid.
		PetscErrorCode ierr;

		// Declare auxiliary COARSE p4est objects; they will be updated during the semi-Lagrangian advection step.
		p4est_t *p4est_np1 = p4est_copy( p4est, P4EST_FALSE );
		p4est_ghost_t *ghost_np1 = my_p4est_ghost_new( p4est_np1, P4EST_CONNECT_FULL );
		p4est_nodes_t *nodes_np1 = my_p4est_nodes_new( p4est_np1, ghost_np1 );

		// Create COARSE semi-lagrangian object and set up to using quadratic interpolation for velocity and phi.
		my_p4est_semi_lagrangian_t semiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors );
		semiLagrangian.set_phi_interpolation( interpolation_method::quadratic );
		semiLagrangian.set_velo_interpolation( interpolation_method::quadratic );

		// Advect the COARSE level-set function one step, then update the grid.
		semiLagrangian.update_p4est( vel, dt, phi, nullptr, nullptr, MASS_BAND_HALF_WIDTH );

		// Destroy old COARSE forest and create new structures.
		p4est_destroy( p4est );
		p4est = p4est_np1;
		p4est_ghost_destroy( ghost );
		ghost = ghost_np1;
		p4est_nodes_destroy( nodes );
		nodes = nodes_np1;

		// Rebuilding COARSE hierarchy and neighborhoods from updated p4est, nodes, and ghost structs.
		delete hierarchy;
		delete nodeNeighbors;
		hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
		nodeNeighbors->init_neighbors();

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
	}

	/**
	 * Collect samples from locally owned and valid COARSE grid points next to the interface.  This function must be
	 * called *after* advecting the FINE grid an adequate number of times and *before* using the FINE grid to "advect"
	 * the COARSE grid.  The function populates an array of pointers to data packets allocated dynamically.
	 * @note The caller function is responsible for deallocating the data packets by using the static utility function
	 * slml::SemiLagrangian::freeDataPacketArray.
	 * @param [in] ngbd_f Advected fine grid neighborhood struct.
	 * @param [in] phi_f Advected fine grid level-set values vector.
	 * @param [in] dt Coarse grid step size.
	 * @param [out] dataPackets Array of data packets received from the semi-Lagrangian sampler.
	 * @param [out] maxRelError Maximum relative error of phi at departure point (w.r.t. minimum cell width).
	 * @param [out] flaggedCoords Map of integer-valued coordinates for locally owned nodes whose samples were collected
	 * 							  (a.k.a valid nodes next to Gamma^n with h-uniform stencils and meeting the angular
	 * 							  criterion between their phi-signed normal and midpoint velocity).
	 * @param [in] debug True if you're debugging and want to store errors and angles in data packets, false otherwise.
	 * @return false if some nodes were backtracked within the domain (although they're not included in dataPackets); true otherwise.
	 */
	bool collectSamples( const my_p4est_node_neighbors_t *ngbd_f, const Vec& phi_f, const double& dt,
					  	 std::vector<slml::DataPacket *>& dataPackets,
					  	 double& maxRelError, std::unordered_map<std::string, double>& locallyOwnedFlaggedCoords )
	{
		assert( phi );	// Check we have a well defined coarse grid.
		PetscErrorCode ierr;
		maxRelError = 0;

		///////////////////////////////////////////////// Preparation //////////////////////////////////////////////////

		// Clear array of locally owned flagged node coords.
		locallyOwnedFlaggedCoords.clear();

		// Allocate PETSc vectors for normals and curvature.
		Vec curvature, normal[P4EST_DIM];
		ierr = VecDuplicate( phi, &curvature );
		CHKERRXX( ierr );
		for( auto& dim : normal )
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &dim );
			CHKERRXX( ierr );
		}

		// Compute curvature, which will be (linearly) interpolated at the interface.
		compute_normals( *nodeNeighbors, phi, normal );
		compute_mean_curvature( *nodeNeighbors, normal, curvature );

		// Prepare curvature interpolation.
		my_p4est_interpolation_nodes_t kappaInterp( nodeNeighbors );
		kappaInterp.set_input( curvature, interpolation_method::linear );

		// Also need read access to phi and normal vectors.
		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		const double *normalReadPtr[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			ierr = VecGetArrayRead( normal[i], &normalReadPtr[i] );
			CHKERRXX( ierr );
		}

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

		//////////////// Compute second derivatives of coarse velocity field: vel_xx, vel_yy[, vel_zz] /////////////////
		// We need these to interpolate the velocity and retrieve the backtracked departure point with second-order
		// accuracy.
		Vec *vel_c_xx[P4EST_DIM];
		for( int dir = 0; dir < P4EST_DIM; dir++ )	// Choose velocity component: u, v, or w.
		{
			vel_c_xx[dir] = new Vec[P4EST_DIM];		// For each velocity component, we need the derivatives w.r.t. x, y, z.
			if( dir == 0 )
			{
				for( int dd = 0; dd < P4EST_DIM; dd++ )
				{
					ierr = VecCreateGhostNodes( p4est, nodes, &vel_c_xx[dir][dd] );
					CHKERRXX( ierr );
				}
			}
			else
			{
				for( int dd = 0; dd < P4EST_DIM; dd++ )
				{
					ierr = VecDuplicate( vel_c_xx[0][dd], &vel_c_xx[dir][dd] );
					CHKERRXX( ierr );
				}
			}
			nodeNeighbors->second_derivatives_central( vel[dir], DIM( vel_c_xx[dir][0], vel_c_xx[dir][1], vel_c_xx[dir][2] ) );
		}

		////////// Compute second spatial derivatives of coarse level-set function: phi_xx, phi_yy[, phi_zz] ///////////
		// We need these to compute numerical level-set value at the departure point with quadratic interpolation.
		// These are also used as part of the samples retrieved for building the training data sets.
		Vec phi_c_xx[P4EST_DIM];
		for( auto & dir : phi_c_xx )
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &dir );
			CHKERRXX( ierr );
		}
		nodeNeighbors->second_derivatives_central( phi, DIM( phi_c_xx[0], phi_c_xx[1], phi_c_xx[2] ) );

		//////////////////////////////////////////////// Data collection ///////////////////////////////////////////////

		// Use a semi-Lagrangian scheme with a single vel step for backtracking to retrieve samples.
		char msg[1024];
		slml::SemiLagrangian semiLagrangianML( &p4est, &nodes, &ghost, nodeNeighbors, phi, MASS_BAND_HALF_WIDTH );
		bool allInside = semiLagrangianML.collectSamples( vel, vel_c_xx, dt, phi, phi_c_xx, dataPackets );

		// Continue process if all samples lie within computational domain.
		if( allInside )
		{
			// Finding the target phi values via quadratic interpolation from FINE grid.  Actually, it doesn't matter
			// the interpolation order because COARSE grid nodes match grid point in FINE grid.
			double targetPhi[dataPackets.size()];
			for( p4est_locidx_t outIndex = 0; outIndex < dataPackets.size(); outIndex++ )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( dataPackets[outIndex]->nodeIdx, p4est, nodes, xyz );
				interp.add_point( outIndex, xyz );
			}
			interp.set_input( phi_f, DIM( phi_f_xx[0], phi_f_xx[1], phi_f_xx[2] ), interpolation_method::quadratic );
			interp.interpolate( targetPhi );
			interp.clear();

			// Go through collected nodes and add the target value.  Populate the flag vector at the same time.
			// Note: Let's collect the nine-point stencils of locally owned nodes as well, and keep only those nodes
			// that have such full stencils.  These stencils will be used to compute kappa with the curvature nnets.
			// Even though retrieving stencils will discard ghost nodes, in a real advection scenario, we must obtain
			// the level-set value at departure points for locally owned nodes with full stencils, set their gammaFlag,
			// and then scatter forward both the flag and the level-set at departure point (computed neurally).
			// Afterward, numerical advection is used in the rest of the nodes that are not flagged (including ghost
			// nodes), and we proceed with the refinement/coarsening process.
			auto *splittingCriteria = (splitting_criteria_t*) p4est->user_pointer;
			NodesAlongInterface nodesAlongInterface( p4est, nodes, nodeNeighbors, (signed char)splittingCriteria->max_lvl );

			for( size_t i = 0; i < dataPackets.size(); i++ )
			{
				slml::DataPacket *dataPacket = dataPackets[i];
				dataPacket->targetPhi_d = targetPhi[i];				// Populate expected phi value.

				double xyz[P4EST_DIM];								// Populate dimensionless curvature at Gamma.
				node_xyz_fr_n( dataPacket->nodeIdx, p4est, nodes, xyz );
				double p = phiReadPtr[dataPacket->nodeIdx];
				dataPacket->hk_a = kappaInterp( DIM( xyz[0] - p * normalReadPtr[0][dataPacket->nodeIdx],
										 			 xyz[1] - p * normalReadPtr[1][dataPacket->nodeIdx],
										 			 xyz[2] - p * normalReadPtr[2][dataPacket->nodeIdx] ) );
				dataPacket->hk_a *= minCellWidth;

				double relError = ABS( targetPhi[i] - dataPacket->numBacktrackedPhi_d ) / minCellWidth;
				maxRelError = MAX( maxRelError, relError );

				std::vector<p4est_locidx_t> stencilIndices( num_neighbors_cube );
				if( nodesAlongInterface.getFullStencilOfNode( dataPacket->nodeIdx, stencilIndices ) )
				{
					// Insert integer-based coordinates into map of flagged coords with its corresponding phi_d^*.
					std::stringstream intCoords;
					for( int j = 0; j < P4EST_DIM; j++ )
						intCoords << long( (xyz[j] - minCoords[j]) / minCellWidth ) << ",";
					locallyOwnedFlaggedCoords[intCoords.str()] = dataPacket->targetPhi_d;
				}
			}

			sprintf( msg, "%3lu   ", dataPackets.size() );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );
		}
		else
		{
			// Don't forget to destroy dynamic objects even if all points we backtracked fell outside the domain.
			slml::SemiLagrangian::freeDataPacketArray( dataPackets );
		}

		///////////////////////////////////////////////// Finishing up /////////////////////////////////////////////////

		// Restore read access.
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		for( int dim = 0; dim < P4EST_DIM; dim++ )
		{
			ierr = VecRestoreArrayRead( normal[dim], &normalReadPtr[dim] );
			CHKERRXX( ierr );
		}

		// Destroy parallel vectors for second derivatives in the COARSE grid.
		for( auto& dir : phi_c_xx )
		{
			ierr = VecDestroy( dir );
			CHKERRXX( ierr );
		}

		for( auto& dir : vel_c_xx )
		{
			for( unsigned char dd = 0; dd < P4EST_DIM; ++dd )
			{
				ierr = VecDestroy( dir[dd] );
				CHKERRXX( ierr );
			}
			delete[] dir;
		}

		// Free vectors with second derivatives for FINE phi.
		for( auto& derivative : phi_f_xx )
		{
			ierr = VecDestroy( derivative );
			CHKERRXX( ierr );
		}

		// Free vectors for normals and curvature.
		ierr = VecDestroy( curvature );
		CHKERRXX( ierr );

		for( auto& dim : normal )
		{
			ierr = VecDestroy( dim );
			CHKERRXX( ierr );
		}

		return allInside;	// True if all backtracked points along the interface fell within domain.
	}

	/**
	 * Write VTK files for debugging.
	 * @note Call this function at the end of each iteration to see how the interface is evolving.
	 * @param [in] vtkIdx File index.
	 * @param [in] howUpdated How was every node updated: 0 numerically, 1 numerically but used to have target level-set
	 * 			   value, 2 preserved its target level-set value.
 	 */
	void writeVTK( int vtkIdx, Vec howUpdated=nullptr ) const
	{
		char name[1024];
		PetscErrorCode ierr;

		const double *phiReadPtr, *howUpdatedReadPtr, *velReadPtr[2];

		bool createdLocalHowUpdated = false;
		if( !howUpdated )	// In the first iteration, we don't have any values for howUpdated Vec.  Create a dummy one.
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated );
			CHKERRXX( ierr );
			createdLocalHowUpdated = true;
		}

		sprintf( name, "ds_debugging_%d", vtkIdx );
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( howUpdated, &howUpdatedReadPtr );
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
								VTK_POINT_DATA, "howUpdated", howUpdatedReadPtr,
								VTK_POINT_DATA, "vel_x", velReadPtr[0],
								VTK_POINT_DATA, "vel_y", velReadPtr[1]
		);
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( howUpdated, &howUpdatedReadPtr );
		CHKERRXX( ierr );
		for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecRestoreArrayRead( vel[dir], &velReadPtr[dir] );
			CHKERRXX( ierr );
		}

		if( createdLocalHowUpdated )
		{
			ierr = VecDestroy( howUpdated );
			CHKERRXX( ierr );
		}
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
