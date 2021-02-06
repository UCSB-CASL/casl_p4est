/**
 * Title: ml_mass_conservation
 * Description: Data set generation for training a neural network that corrects the semi-Lagrangian scheme for simple
 * advection.  We assume that all considered velocity fields are divergence-free.  To generate these velocity fields, we
 * obtain the skew gradient of random gaussians.
 * Author: Luis Ángel (임 영민)
 * Date Created: 01-20-2021
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#endif

#include <iostream>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <src/casl_geometry.h>

// Velocity field.
struct UComponent : CF_2
{
private:
	double sign;
public:
	UComponent() : sign( 1.0 ){}

	void switch_direction()
	{ sign *= -1.0; }

	double operator()( double x, double y ) const override
	{
		return -SQR( sin( PI * x )) * sin( 2 * PI * y ) * sign;
	}
};

struct VComponent : CF_2
{
private:
	double sign;
public:
	VComponent() : sign( 1.0 ){}

	void switch_direction()
	{ sign *= -1.0; }

	double operator()( double x, double y ) const override
	{
		return SQR( sin( PI * y )) * sin( 2 * PI * x ) * sign;
	}
};

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
	const int COARSE_MAX_RL = 6;		// Maximum refinement levels for coarse and fine grids.
	const int FINE_MAX_RL = 7;

	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.
	const double LIP = 1.2;				// Lipschitz constant.
	const auto PHI_INTERP_MTHD = interpolation_method::linear;		// Phi interpolation method.
	const auto VEL_INTERP_MTHD = interpolation_method::linear;		// Velocity interpolation method.

	const double MIN_D = -1.0;			// Domain minimum value for each dimension.
	const int NUM_TREES_PER_DIM = 2;	// Number of macro cells per dimension.

	const int NUM_ITER_VTK = 8;			// Save VTK files every NUM_ITER_VTK iterations.

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		// To generate data sets we don't admit more than a single process to avoid race conditions.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		parStopWatch watch;
		printf( ">> Began to generate data sets for MAX_RL_COARSE = %d and MAX_RL_FINE = %d\n",
		  	COARSE_MAX_RL, FINE_MAX_RL );
		watch.start();

		// Domain information: a square with the same number of trees per dimension.
		const int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
		const double xyz_min[] = {MIN_D, MIN_D, MIN_D};
		const double xyz_max[] = {-MIN_D, -MIN_D, -MIN_D};
		const int periodic[] = {0, 0, 0};

		// Define the initial interface.
		geom::Sphere sphere( DIM( 0.5, 0.75, 0.0 ), 0.15 );

		// Declaration of the macromesh via the brick and connectivity objects.
		my_p4est_brick_t brick;
		p4est_connectivity_t *connectivity;
		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Define the velocity field array.
		UComponent uComponent;
		VComponent vComponent;
		const CF_2 *velocityField[P4EST_DIM] = {&uComponent, &vComponent};

		// Pointers to p4est variables.
		p4est_t *p4est;
		p4est_ghost_t *ghost;
		p4est_nodes_t *nodes;

		// Create the forest.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );

		// Refine based on distance to the interface.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 1, COARSE_MAX_RL, &sphere, 6.0 );
		p4est->user_pointer = &levelSetSplittingCriterion;

		// Create the forest using a level set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and recursively partition forest.
		my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor node structure.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
		nodeNeighbors->init_neighbors();

		// Compute grid size data.
		double dxyz[P4EST_DIM];
		double dxyz_min;
		double diag_min;
		get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );

		// Declare data vectors and pointers for read/write.
		Vec phi, phiExact;					// Evolving level-set function.
		double *phiPtr, *phiExactPtr;

		Vec vel[P4EST_DIM];					// Veloctiy field.
		double *velPtr[P4EST_DIM];

		// Allocate memory for the Vecs.
		ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( p4est, nodes, &phiExact );
		CHKERRXX( ierr );
		for( auto& dir : vel )
		{
			ierr = VecCreateGhostNodes( p4est, nodes, &dir );
			CHKERRXX( ierr );
		}

		// Sample the level-set function at t = 0 at all independent nodes.
		sample_cf_on_nodes( p4est, nodes, sphere, phi );
		sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

		// Sample the velocity field at t = 0 at all independent nodes.
		for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
			sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );

		// Save the initial grid and fields into vtk.
		char name[1024];
		sprintf( name, "visualization_%d", 0 );
		ierr = VecGetArray( phi, &phiPtr );
		CHKERRXX( ierr );
		ierr = VecGetArray( phiExact, &phiExactPtr );
		CHKERRXX( ierr );
		for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecGetArray( vel[dir], &velPtr[dir] );
			CHKERRXX( ierr );
		}
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								2 + P4EST_DIM, 0, name,
								VTK_POINT_DATA, "phi", phiPtr,
								VTK_POINT_DATA, "phiExact", phiExactPtr,
								VTK_POINT_DATA, "vel_x", velPtr[0],
								VTK_POINT_DATA, "vel_y", velPtr[1]
		);
		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( phiExact, &phiExactPtr );
		CHKERRXX( ierr );
		for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
		{
			ierr = VecRestoreArray( vel[dir], &velPtr[dir] );
			CHKERRXX( ierr );
		}
		char msg[1024];
		sprintf( msg, " -> Saving vtu files in %s.vtu\n", name );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

		// Define time stepping variables.
		double tn = 0;								// Initial time.
		bool hasVelSwitched = false;
		int iter = 0;
		int vtkIdx = 1;								// Index for VTK files.
		const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm known analitically.
		double dt = CFL * dxyz_min / MAX_VEL_NORM;	// This is deltaT knowing that the CFL condition is (c * deltaT)/deltaX <= CFLN.

		// Advection loop.
		while( tn + 0.1 * dt < DURATION )
		{
			// Clip time step if it's going to go over the final time.
			if( tn + dt > DURATION )
				dt = DURATION - tn;

			// Clip time step if it's going to go over the half time.
			if( tn + dt >= DURATION / 2.0 && !hasVelSwitched )
			{
				if( tn + dt > DURATION / 2.0 )
					dt = (DURATION / 2.0) - tn;
				uComponent.switch_direction();
				vComponent.switch_direction();
				hasVelSwitched = true;
			}

			// Declare auxiliary p4est objects: these will be updated during the semi-Lagrangian advection step.
			p4est_t *p4est_1 = p4est_copy( p4est, P4EST_FALSE );
			p4est_ghost_t *ghost_1 = my_p4est_ghost_new( p4est_1, P4EST_CONNECT_FULL );
			p4est_nodes_t *nodes_1 = my_p4est_nodes_new( p4est_1, ghost_1 );

			// Create semi-lagrangian object.
			my_p4est_semi_lagrangian_t semiLagrangian( &p4est_1, &nodes_1, &ghost_1, nodeNeighbors );
			semiLagrangian.set_phi_interpolation( PHI_INTERP_MTHD );
			semiLagrangian.set_velo_interpolation( VEL_INTERP_MTHD );

			// Advect the level-set function one step, then update the grid.
			semiLagrangian.update_p4est( vel, dt, phi );

			// Destroy old forest and create new structures.
			p4est_destroy( p4est );
			p4est = p4est_1;
			p4est_ghost_destroy( ghost );
			ghost = ghost_1;
			p4est_nodes_destroy( nodes );
			nodes = nodes_1;

			delete hierarchy;
			hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
			delete nodeNeighbors;
			nodeNeighbors = new my_p4est_node_neighbors_t( hierarchy, nodes );
			nodeNeighbors->init_neighbors();

			// Reinitialize level-set function.
			my_p4est_level_set_t levelSet( nodeNeighbors );
			levelSet.reinitialize_2nd_order( phi );

			// Advance time step and iteration counter.
			tn += dt;
			iter++;
			if( tn == DURATION / 2.0 )
				dt = CFL * dxyz_min / MAX_VEL_NORM;		// Restore time step to original definition.

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

			// Display iteration message.
			sprintf( msg, " Iteration %04d: t = %1.4f \n", iter, tn );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Save to vtk format.
			if( iter >= vtkIdx * NUM_ITER_VTK || tn == DURATION )
			{
				sprintf( name, "visualization_%d", vtkIdx );
				ierr = VecGetArray( phi, &phiPtr );
				CHKERRXX( ierr );
				ierr = VecGetArray( phiExact, &phiExactPtr );
				CHKERRXX( ierr );
				for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
				{
					ierr = VecGetArray( vel[dir], &velPtr[dir] );
					CHKERRXX( ierr );
				}
				my_p4est_vtk_write_all( p4est, nodes, ghost,
										P4EST_TRUE, P4EST_TRUE,
										2 + P4EST_DIM, 0, name,
										VTK_POINT_DATA, "phi", phiPtr,
										VTK_POINT_DATA, "phiExact", phiExactPtr,
										VTK_POINT_DATA, "vel_x", velPtr[0],
										VTK_POINT_DATA, "vel_y", velPtr[1]
				);
				ierr = VecRestoreArray( phi, &phiPtr );
				CHKERRXX( ierr );
				ierr = VecRestoreArray( phiExact, &phiExactPtr );
				CHKERRXX( ierr );
				for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
				{
					ierr = VecRestoreArray( vel[dir], &velPtr[dir] );
					CHKERRXX( ierr );
				}

				sprintf( msg, " -> Saving vtu files in %s.vtu\n", name );
				ierr = PetscPrintf( mpi.comm(), msg );
				CHKERRXX( ierr );
				vtkIdx++;
			}
		}

		// Compute error L-inf norm and store it once we finished a simulation split.
		ierr = VecGetArray( phi, &phiPtr );
		CHKERRXX( ierr );
		ierr = VecGetArray( phiExact, &phiExactPtr );
		CHKERRXX( ierr );
		double error = 0;
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			if( fabs( phiPtr[n] ) < 4.0 * diag_min )
				error = MAX( error, fabs( phiPtr[n] - phiExactPtr[n] ) );
		}
		int mpiret = MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() );
		SC_CHECK_MPI( mpiret );
		ierr = VecRestoreArray( phi, &phiPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( phiExact, &phiExactPtr );
		CHKERRXX( ierr );

		// Destroy the dynamically allocated Vecs.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );
		ierr = VecDestroy( phiExact );
		CHKERRXX( ierr );
		for( auto& dir : vel )
		{
			ierr = VecDestroy( dir );
			CHKERRXX( ierr );
		}

		// Destroy the dynamically allocated p4est and my_p4est structures.
		delete nodeNeighbors;
		delete hierarchy;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );

		// Destroy the dynamically allocated brick and connectivity structures.  Connectivity and Brick objects are the
		// only ones that are not re-created in every iteration of semi-Lagrangian advection.
		my_p4est_brick_destroy( connectivity, &brick );

		printf( "<< Finished data set generation after %f secs.\n", watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}
}

