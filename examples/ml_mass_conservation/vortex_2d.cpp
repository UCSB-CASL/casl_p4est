/**
 * Testing semi-Lagrangian error-correction neural networks with a vortex velocity field.
 * The test consists of a deforming transformation of a circular-interface level-set function located at the top
 * center of a unit-square domain.  The level-set is advected using a velocity field that switches direction at half the
 * total time of the simulation.  In the end, the initial circular interface should be recovered.
 *
 * Code is based on examples/level_set_advection/main_2d.cpp
 *
 * @cite C. Min and F. Gibou, A second order accurate level set method on non-graded adaptive cartesian grids, J.
 * 		 Comput. Phys., 225:300-321, 2007.  Vortex test appears on p. 310.
 *
 * Author: Luis Ángel (임 영민)
 * Created: May 22, 2021.
 * Updated: July 11, 2021.
 */

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian_ml.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_geometry.h>
#include <src/parameter_list.h>

///////////////////////////////////// Vortex velocity field (also divergence free) /////////////////////////////////////

class UComponent : public CF_2
{
private:
	double sign;

public:
	UComponent()
	{ sign = 1.0; }

	void switch_direction()
	{ sign *= -1.0; }

	double operator()( double x, double y ) const override
	{
		return -SQR( sin( M_PI * x ) ) * sin( 2 * M_PI * y ) * sign;
	}
};

class VComponent : public CF_2
{
private:
	double sign;

public:
	VComponent()
	{ sign = 1.0; }

	void switch_direction()
	{ sign *= -1.0; }

	double operator()( double x, double y ) const override
	{
		return SQR( sin( M_PI * y ) ) * sin( 2 * M_PI * x ) * sign;
	}
};

////////////////////////////////////////////////// Auxiliary functions /////////////////////////////////////////////////

void writeVTK( int vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec phiExact, Vec hk,
			   Vec uniformFlag, Vec howUpdated )
{
	char name[1024];
	PetscErrorCode ierr;

	const double *phiReadPtr, *phiExactReadPtr, *hkReadPtr, *uniformFlagReadPtr, *howUpdatedReadPtr;	// Pointers to Vec contents.

	sprintf( name, "vortex_%d", vtkIdx );
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( phiExact, &phiExactReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( uniformFlag, &uniformFlagReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( howUpdated, &howUpdatedReadPtr );
	CHKERRXX( ierr );
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							5, 0, name,
							VTK_POINT_DATA, "phi", phiReadPtr,
							VTK_POINT_DATA, "phiExact", phiExactReadPtr,
							VTK_POINT_DATA, "hk", hkReadPtr,
							VTK_POINT_DATA, "uniformFlag", uniformFlagReadPtr,
							VTK_POINT_DATA, "howUpdated", howUpdatedReadPtr
	);
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( phiExact, &phiExactReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( uniformFlag, &uniformFlagReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( howUpdated, &howUpdatedReadPtr );
	CHKERRXX( ierr );

	PetscPrintf( p4est->mpicomm, ":: Saved vtk files with index %02d ::\n", vtkIdx );
}

void computeHK( const double& h, const my_p4est_node_neighbors_t *nbgd, const Vec& phi, Vec& hk, Vec& uniformFlag,
				std::unordered_set<p4est_locidx_t>& localUniformIndices )
{
	PetscErrorCode ierr;

	// Shortcuts.
	const p4est_t *p4est = nbgd->get_p4est();
	const p4est_nodes_t *nodes = nbgd->get_nodes();

	// Prepare output parallel vector with dimensionless curvature values.  Will store interface attribute only for
	// nodes that lies next to Gamma, regardless of their stencils uniformity.  Indices of vertices with uniform
	// stencils next to the interface will be returned in the indices out vector.
	ierr = hk? VecDestroy( hk ) : 0;
	CHKERRXX( ierr );
	ierr = VecCreateGhostNodes( p4est, nodes, &hk );			// By default, all values are zero.
	CHKERRXX( ierr );

	double *hkPtr;
	ierr = VecGetArray( hk, &hkPtr );
	CHKERRXX( ierr );

	// Allocate flag parallel vector to indicate nodes with uniform stencils.  These were susceptible to machine
	// learning improvement.  Downstream processing in machine learning semi-Lagrangian advection will use this flag as
	// an indicator that curvature computation is relieable.  SLML then considers only the intersection of locally
	// owned nodes with those flagged here.
	ierr = uniformFlag? VecDestroy( uniformFlag ) : 0;
	CHKERRXX( ierr );
	ierr = VecCreateGhostNodes( p4est, nodes, &uniformFlag );	// By default, all values are zero.
	CHKERRXX( ierr );

	double *uniformFlagPtr;
	ierr = VecGetArray( uniformFlag, &uniformFlagPtr );
	CHKERRXX( ierr );

	// Allocate temporary PETSc vectors for normals and curvature.
	Vec curvature, normal[P4EST_DIM];
	ierr = VecCreateGhostNodes( p4est, nodes, &curvature );
	CHKERRXX( ierr );
	for( auto& dim : normal )
	{
		ierr = VecCreateGhostNodes( p4est, nodes, &dim );
		CHKERRXX( ierr );
	}

	// First, compute curvature, which will be interpolated at the interface and scaled by h for points next to Gamma.
	compute_normals( *nbgd, phi, normal );
	compute_mean_curvature( *nbgd, normal, curvature );

	// Prepare curvature interpolation.
	my_p4est_interpolation_nodes_t kappaInterp( nbgd );
	kappaInterp.set_input( curvature, interpolation_method::linear );

	// Also need read access to phi and normals to compute closest point on Gamma.
	const double *phiReadPtr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	const double *normalReadPtr[P4EST_DIM];
	for( int i = 0; i < P4EST_DIM; i++ )
	{
		ierr = VecGetArrayRead( normal[i], &normalReadPtr[i] );
		CHKERRXX( ierr );
	}

	// Compute dimensionless curvature for points next to the interface.
	auto *splittingCriteria = (splitting_criteria_t*) p4est->user_pointer;
	std::vector<p4est_locidx_t> indices;

	// Collect *locally owned* grid points next to the interface, although these might have a nonuniform stencil.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, nbgd, (signed char)splittingCriteria->max_lvl );
	nodesAlongInterface.getIndices( &phi, indices );
	localUniformIndices.clear();
	localUniformIndices.reserve( indices.size() );	// Return this to caller with effective local indices of nodes next
													// to Gamma with uniform stencils.
	for( auto nodeIdx : indices )
	{
		// Compute numerical hk for ALL points next to Gamma, regardless their stencil uniformity.  Those are further
		// processed below.
		double xyz[P4EST_DIM];
		node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
		double p = phiReadPtr[nodeIdx];
		hkPtr[nodeIdx] = kappaInterp( DIM( xyz[0] - p * normalReadPtr[0][nodeIdx],
									 	   xyz[1] - p * normalReadPtr[1][nodeIdx],
									 	   xyz[2] - p * normalReadPtr[2][nodeIdx] ) );
		hkPtr[nodeIdx] *= h;

		try
		{
			std::vector<p4est_locidx_t> stencilIndices( num_neighbors_cube );
			if( nodesAlongInterface.getFullStencilOfNode( nodeIdx, stencilIndices ) )	// Uniform stencil?
			{
				uniformFlagPtr[nodeIdx] = p4est->mpirank + 1;		// Flag node.
				localUniformIndices.insert( nodeIdx );

				// TODO: Use hybrid inference system to improve (if needed) curvature computation.
			}
		}
		catch( const std::exception &exception )
		{
			std::cerr << "[" << nodeIdx << "]" << std::endl;
		}
	}

	// Scatter forward dimensionless curvature and uniform-flag values to synchronize info.
	ierr = VecGhostUpdateBegin( hk, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );
	VecGhostUpdateEnd( hk, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );

	ierr = VecGhostUpdateBegin( uniformFlag, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );
	VecGhostUpdateEnd( uniformFlag, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );

	// Restore array accessors.
	for( int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = VecRestoreArrayRead( normal[dim], &normalReadPtr[dim] );
		CHKERRXX( ierr );
	}

	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArray( uniformFlag, &uniformFlagPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArray( hk, &hkPtr );
	CHKERRXX( ierr );

	// Destroy temporary parallel normals and curvature vectors.
	ierr = VecDestroy( curvature );
	CHKERRXX( ierr );
	for( auto& dim : normal )
	{
		ierr = VecDestroy( dim );
		CHKERRXX( ierr );
	}
}

//////////////////////////////////////////////////// Main function /////////////////////////////////////////////////////

/**
 * Main function.
 * @param argc Number of input arguments.
 * @param argv Actual arguments.
 * @return 0 if process finished successfully, nonzero otherwise.
 */
int main( int argc, char** argv )
{
	// Main global variables.
	const double DURATION = 1.0;		// Duration of the simulation.
	const int MAX_RL = 6;				// Grid's maximum refinement level.
	const int REINIT_NUM_ITER = 10;		// Number of iterations for level-set renitialization.
	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.

	const int MIN_D = 0;				// Domain minimum and maximum values for each dimension.
	const int MAX_D = 1;
	const int NUM_TREES_PER_DIM = 1;	// Number of macro cells per dimension.
	const int PERIODICITY = 0;			// Domain periodicity.

	const int NUM_ITER_VTK = 4;			// Save VTK files every NUM_ITER_VTK iterations.

	const double BAND = 2; 				// Minimum number of cells around interface.  Must match what was used in training.

	char msg[1024];						// Some string to write messages to standard ouput.

	// Setting up parameters from command line.
	param_list_t pl;
	param_t<int> mode ( pl, 1, "mode", "Execution mode: 0 - numerical, 1 - nnet (default: 1)");
	param_t<int> exportAllVTK (pl, 1, "exportAllVTK", "Export all VTK files: 0 - no (only first and last), 1 - yes (default: 1)" );

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Vortex test" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// OpenMP verification.
		int nThreads = 0;
#pragma omp parallel reduction( + : nThreads ) default( none )
		nThreads += 1;

		std::cout << "Rank " << mpi.rank() << " can spawn " << nThreads << " thread(s)\n\n";

		// Loading semi-Lagrangian error-correction neural network if user has selected its option.
		const slml::NeuralNetwork *nnet = nullptr;
		if( mode() )
		{
			nnet = new slml::NeuralNetwork( "/Users/youngmin/nnets", 1. / (1 << MAX_RL), false );

			const int N_SAMPLES = 2;
			double inputs[N_SAMPLES][MASS_INPUT_SIZE] = {
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4},
				{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6}
			};
			double outputs[N_SAMPLES];
			int j;
#pragma omp parallel for default( none ) schedule( static ) \
		shared( N_SAMPLES, nnet, inputs, outputs ) \
		private( j )
			for( j = 0; j < N_SAMPLES; j++ )
			{
				nnet->predict( &inputs[j], &outputs[j], 1 );
				printf( "Thread %i took care of sample %i\n", omp_get_thread_num(), j );
			}

			std::cout << std::setprecision( 8 );
			std::cout << outputs[0] << std::endl;
			std::cout << outputs[1] << std::endl;
		}

		// Let's continue with numerical computations.

		parStopWatch watch;
		watch.start();

		sprintf( msg, ">> Began 2D vortex test with MAX_RL = %d in %s mode\n", MAX_RL, mode()? "NNET" : "NUMERICAL" );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

		// Define the velocity field arrays.
		UComponent uComponent;
		VComponent vComponent;
		const CF_DIM *velocityField[P4EST_DIM] = {&uComponent, &vComponent};

		// Domain information: a square with the same number of trees per dimension.
		const int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
		const double xyz_min[] = {MIN_D, MIN_D, MIN_D};
		const double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		const int periodic[] = {PERIODICITY, PERIODICITY, PERIODICITY};

		// Define the initial interfaces: exact and non-signed distance function.
		const double CENTER[P4EST_DIM] = {DIM( 0.5, 0.75, 0.0 )};
		const double RADIUS = 0.15;
		geom::SphereNSD sphereNsd( DIM( CENTER[0], CENTER[1], CENTER[2] ), RADIUS );
		geom::Sphere sphere( DIM( CENTER[0], CENTER[1], CENTER[2] ), RADIUS );

		// Macromesh declaration via the brick and connectivity objects.
		my_p4est_brick_t brick;
		p4est_connectivity_t *connectivity;
		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Pointers to p4est variables.
		p4est_t *p4est;
		p4est_ghost_t *ghost;
		p4est_nodes_t *nodes;

		// Create forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		splitting_criteria_cf_and_uniform_band_t lsSplittingCriterion( 1, MAX_RL, &sphereNsd, BAND );
		p4est->user_pointer = &lsSplittingCriterion;

		// Refine and partition forest.
		for( int i = 0; i < MAX_RL; i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize neighbor node structure and hierarchy.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
		nodeNeighbors->init_neighbors();

		// Retrieve grid size data.
		double dxyz[P4EST_DIM];
		double dxyz_min;
		double diag_min;
		get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );

		// Declare data vectors and pointers for read/write.
		Vec phi;							// Level-set function values (subject to reinitialization).
		Vec phiExact;						// Exact level-set function values.
		const double *phiReadPtr, *phiExactReadPtr;

		Vec vel[P4EST_DIM];					// Veloctiy field.

		// Allocate memory for parallel vectors.
		ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( p4est, nodes, &phiExact );
		CHKERRXX( ierr );
		for( auto& dir : vel )
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &dir );
			CHKERRXX( ierr );
		}

		// Sample the level-set functions at t = 0 at all independent nodes.
		sample_cf_on_nodes( p4est, nodes, sphereNsd, phi );
		sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

		// Sample the velocity field at t = 0 at all independent nodes.
		for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
			sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );

		// Reinitialize grid before we start advection as we are using a non-signed distance level-set function.
		my_p4est_level_set_t levelSet( nodeNeighbors );
		levelSet.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

		// Computing curvature and flagging nodes next to Gamma that have uniform stencils.
		Vec hk = nullptr, uniformFlag = nullptr;
		std::unordered_set<p4est_locidx_t> localUniformIndices;
		computeHK( dxyz_min, nodeNeighbors, phi, hk, uniformFlag, localUniformIndices );

		// Let's use a debugging vector to detect how were grid points updated at time tnp1: 0 if numerically, 1 if
		// numerically but around a band of location of new interface, and 2 if using neural inference system.
		Vec howUpdated;
		ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated );
		CHKERRXX( ierr );

		// Save the initial grid and fields into vtk (regardless of input command choice).
		writeVTK( 0, p4est, nodes, ghost, phi, phiExact, hk, uniformFlag, howUpdated );

		// Define time stepping variables.
		double tn = 0;								// Current time.
		bool hasVelSwitched = false;
		int iter = 0;
		int vtkIdx = 1;								// Index for post VTK files.
		const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm is known analitically.
		double dt = CFL * dxyz_min / MAX_VEL_NORM;	// deltaT knowing that the CFL condition is c*dt/dx <= CFLN.

		// Advection loop.
		while( tn < DURATION )
		{
			// Clip step if it's going to go over the final time.
			if( tn + dt > DURATION )
				dt = DURATION - tn;

			// Clip time step if it's going to go over half time.
			if( tn + dt >= DURATION / 2.0 && !hasVelSwitched )
			{
				dt = (DURATION / 2.0) - tn;
				uComponent.switch_direction();
				vComponent.switch_direction();
				hasVelSwitched = true;
				ierr = PetscPrintf( mpi.comm(), "*** Switching Velocity ***\n" );
				CHKERRXX( ierr );
			}

			// p4est objects at time tnp1; they will be updated during the semi-Lagrangian advection step.
			p4est_t *p4est_np1 = p4est_copy( p4est, P4EST_FALSE );
			p4est_ghost_t *ghost_np1 = my_p4est_ghost_new( p4est_np1, P4EST_CONNECT_FULL );
			p4est_nodes_t *nodes_np1 = my_p4est_nodes_new( p4est_np1, ghost_np1 );

			// Create semi-Lagrangian object in machine learning module: linear interp. for phi, quadratic for velocity.
			slml::SemiLagrangian *mlSemiLagrangian;
			my_p4est_semi_lagrangian_t *numSemiLagrangian;
			if( mode() )
			{
				mlSemiLagrangian = new slml::SemiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors,
															 &localUniformIndices, nnet, BAND, iter );
			}
			else
			{
				numSemiLagrangian = new my_p4est_semi_lagrangian_t( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors );
				numSemiLagrangian->set_phi_interpolation( interpolation_method::quadratic );
				numSemiLagrangian->set_velo_interpolation( interpolation_method::quadratic );
			}

			// Advect level-set function one step, then update the grid.
			if( mode() )
				mlSemiLagrangian->updateP4EST( vel, dt, &phi, hk, &howUpdated );
			else
				numSemiLagrangian->update_p4est( vel, dt, phi, nullptr, nullptr, BAND );

			// Destroy old forest and create new structures.
			p4est_destroy( p4est );
			p4est = p4est_np1;
			p4est_ghost_destroy( ghost );
			ghost = ghost_np1;
			p4est_nodes_destroy( nodes );
			nodes = nodes_np1;

			delete hierarchy;
			delete nodeNeighbors;
			hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
			nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
			nodeNeighbors->init_neighbors();

			// Reinitialize level-set function.
			my_p4est_level_set_t ls( nodeNeighbors );
			if( mode() )
			{
				// Selective reinitialization of level-set function: affect only those nodes that were not updated with nnet.
				Vec mask;
				ierr = VecCreateGhostNodes( p4est, nodes, &mask );		// Mask vector to flag updatable nodes.
				CHKERRXX( ierr );

				const double *howUpdatedReadPtr;
				ierr = VecGetArrayRead( howUpdated, &howUpdatedReadPtr );
				CHKERRXX( ierr );

				double *maskPtr;
				ierr = VecGetArray( mask, &maskPtr );
				CHKERRXX( ierr );

				int numMaskedNodes = 0;
				for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )	// No need to check all independent nodes.
				{
					if( howUpdatedReadPtr[n] == 2 )		// Masked node? Nonupdatable?
					{
						numMaskedNodes++;
						maskPtr[n] = 0;					// 0 => nonupdatable.
					}
					else								// Updatable?
						maskPtr[n] = 1;					// 1 => updatable.
				}

				ierr = VecRestoreArray( mask, &maskPtr );
				CHKERRXX( ierr );

				ierr = VecRestoreArrayRead( howUpdated, &howUpdatedReadPtr );
				CHKERRXX( ierr );

				ls.reinitialize_2nd_order_with_mask( phi, mask, numMaskedNodes, REINIT_NUM_ITER );

				ierr = VecDestroy( mask );
				CHKERRXX( ierr );
			}
			else
			{
				ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );
			}

			// Advance time.
			tn += dt;
			dt = CFL * dxyz_min / MAX_VEL_NORM;						// Restore time step size.
			iter++;

			// Re-sample the velocity field on new grid.
			for( int dir = 0; dir < P4EST_DIM; dir++ )
			{
				ierr = VecDestroy( vel[dir] );
				CHKERRXX( ierr );
				ierr = VecCreateGhostNodes( p4est, nodes, &vel[dir] );
				CHKERRXX( ierr );
				sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );
			}

			// Re-sample the exact initial level-set function.
			ierr = VecDestroy( phiExact );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &phiExact );
			CHKERRXX( ierr );
			sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

			// Recompute dimensionless curvature and uniform flag.
			computeHK( dxyz_min, nodeNeighbors, phi, hk, uniformFlag, localUniformIndices );

			// Display iteration message.
			sprintf( msg, "\tIteration %04d: t = %1.4f \n", iter, tn );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Save to vtk format (last file is always written, the others are written if exportAllVTK is true).
			if( ABS( tn - DURATION ) <= PETSC_MACHINE_EPSILON ||
				(exportAllVTK() && (ABS( tn - DURATION / 2.0 ) <=PETSC_MACHINE_EPSILON || iter % NUM_ITER_VTK == 0)) )
			{
				writeVTK( vtkIdx, p4est, nodes, ghost, phi, phiExact, hk, uniformFlag, howUpdated );
				vtkIdx++;
			}

			// Destroy semi-Lagrangian objects.
			if( mode() )
				delete mlSemiLagrangian;
			else
				delete numSemiLagrangian;
		}

		// Compute error L-1 and L-inf norms.
		int numPoints = 0;
		double cumulativeError = 0;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phiExact, &phiExactReadPtr );
		CHKERRXX( ierr );
		double maxError = 0;
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			if( ABS( phiReadPtr[n] ) < diag_min )
			{
				double error = ABS( phiReadPtr[n] - phiExactReadPtr[n] );
				maxError = MAX( maxError, error );
				numPoints++;
				cumulativeError += error;
			}
		}
		int mpiret = MPI_Allreduce( MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() );	// Max abs error.
		SC_CHECK_MPI( mpiret );

		mpiret = MPI_Allreduce( MPI_IN_PLACE, &numPoints, 1, MPI_INT, MPI_SUM, mpi.comm() );		// Total error.
		SC_CHECK_MPI( mpiret );
		mpiret = MPI_Allreduce( MPI_IN_PLACE, &cumulativeError, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() );
		SC_CHECK_MPI( mpiret );

		double l1Error = cumulativeError / numPoints;

		double area = area_in_negative_domain( p4est, nodes, phi );
		double expectedArea = M_PI * SQR( RADIUS );
		double massLossPercentage = (1.0 - area / expectedArea) * 100.0;

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( phiExact, &phiExactReadPtr );
		CHKERRXX( ierr );

		// Destroy parallel vectors.
		ierr = VecDestroy( howUpdated );
		CHKERRXX( ierr );

		ierr = VecDestroy( hk );
		CHKERRXX( ierr );

		ierr = VecDestroy( uniformFlag );
		CHKERRXX( ierr );

		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		for( auto& dir : vel )
		{
			ierr = VecDestroy( dir );
			CHKERRXX( ierr );
		}
		ierr = VecDestroy( phiExact );
		CHKERRXX( ierr );

		// Destroy p4est and my_p4est structures.
		delete hierarchy;
		delete nodeNeighbors;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );

		// Destroy the dynamically allocated FINE brick and connectivity structures.
		// Connectivity and Brick objects are the only ones that are not re-created in every iteration of
		// semi-Lagrangian advection.
		my_p4est_brick_destroy( connectivity, &brick );

		// Destroy neural network.
		delete nnet;

		sprintf( msg, "<< Finished after %.3f secs with:\n   mean abs error %.3e\n   max abs error %.3e\n   area %.3e (expected %.3e, loss %.2f%%%%)",
				 watch.get_duration_current(), l1Error, maxError, area, expectedArea, massLossPercentage );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );
		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}

	return 0;
}