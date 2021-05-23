/**
 * Testing semi-Lagrangian error-correction neural networks with a vortex velocity field.
 * The test consists of a deforming transformation of a circular-interface level-set function located at the top
 * center of a unit-square domain.  The level-set is advected using a velocity field that switches direction at half the
 * total time of the simulation.  In the end, the initial circular interface should be recovered.
 *
 * Code is based on examples/level_set_advection/main_2d.cpp
 *
 * @cite C. Min and F. Gibou, A second order accurate level set method on non-graded adaptive cartesian grids, J.
 * 		 Comput. Phys., 225:300-321, 2007.
 *
 * Author: Luis Ángel (임 영민)
 * Date Created: 05-22-2021
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
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
#include "data_sets/CoarseGrid.h"

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

void writeVTK( int vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec vel[P4EST_DIM],
			   Vec phiExact )
{
	char name[1024];
	PetscErrorCode ierr;

	const double *phiReadPtr, *phiExactReadPtr, *velReadPtr[2];		// Pointers to Vec contents.

	sprintf( name, "vortex_%d", vtkIdx );
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( phiExact, &phiExactReadPtr );
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
							VTK_POINT_DATA, "phiExact", phiExactReadPtr,
							VTK_POINT_DATA, "vel_x", velReadPtr[0],
							VTK_POINT_DATA, "vel_y", velReadPtr[1]
	);
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( phiExact, &phiExactReadPtr );
	CHKERRXX( ierr );
	for( unsigned int dir = 0; dir < P4EST_DIM; ++dir )
	{
		ierr = VecRestoreArrayRead( vel[dir], &velReadPtr[dir] );
		CHKERRXX( ierr );
	}

	PetscPrintf( p4est->mpicomm, ":: Saved vtk files with index %02d ::\n", vtkIdx );
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
	const int REINIT_NUM_ITER = 20;		// Number of iterations for level-set renitialization.

	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.
	const auto PHI_INTERP_MTHD = interpolation_method::linear;		// Phi interpolation method.
	const auto VEL_INTERP_MTHD = interpolation_method::quadratic;	// Velocity interpolation method.

	const int MIN_D = 0;				// Domain minimum and maximum values for each dimension.
	const int MAX_D = 1;
	const int NUM_TREES_PER_DIM = 1;	// Number of macro cells per dimension.
	const int PERIODICITY = 0;			// Domain periodicity.

	const int NUM_ITER_VTK = 8;			// Save VTK files every NUM_ITER_VTK iterations.

	const double BAND = 2; 				// Minimum number of cells around interface.  Must match what was used in training.

	char msg[1024];						// Some string to write messages to standard ouput.

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		parStopWatch watch;
		watch.start();

		sprintf( msg, ">> Began 2D vortex test with MAX_RL = %d\n", MAX_RL );
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

		// Save the initial grid and fields into vtk.
		writeVTK( 0, p4est, nodes, ghost, phi, vel, phiExact );

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

			// Create semi-lagrangian object.
			my_p4est_semi_lagrangian_t semiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors );
			semiLagrangian.set_phi_interpolation( PHI_INTERP_MTHD );
			semiLagrangian.set_velo_interpolation( VEL_INTERP_MTHD );

			// Advect level-set function one step, then update the grid.
			semiLagrangian.update_p4est_one_vel_step( vel, dt, phi, BAND );

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
			ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

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

			// Display iteration message.
			sprintf( msg, "\tIteration %04d: t = %1.4f \n", iter, tn );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Save to vtk format.
			if( iter % NUM_ITER_VTK == 0 || tn == DURATION || tn == DURATION / 2.0 )
			{
				writeVTK( vtkIdx, p4est, nodes, ghost, phi, vel, phiExact );
				vtkIdx++;
			}
		}

		// Compute error L-inf norm.
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phiExact, &phiExactReadPtr );
		CHKERRXX( ierr );
		double error = 0;
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			if( ABS( phiReadPtr[n] ) < 4.0 * diag_min )
				error = MAX( error, ABS( phiReadPtr[n] - phiExactReadPtr[n] ) );
		}
		int mpiret = MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() );
		SC_CHECK_MPI( mpiret );
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( phiExact, &phiExactReadPtr );
		CHKERRXX( ierr );

		// Destroy parallel vectors.
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

		sprintf( msg, "<< Finished test after %f secs with error %.3e.\n", watch.get_duration_current(), error );
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