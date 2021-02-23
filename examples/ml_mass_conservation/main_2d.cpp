/**
 * Title: ml_mass_conservation
 *
 * Description: Data set generation for training a neural network that corrects the semi-Lagrangian scheme for simple
 * advection.  We assume that all considered velocity fields are divergence-free.  To generate these velocity fields, we
 * obtain the skew gradient of random Gaussians.
 * @note Not yet tested on 3D.
 *
 * Author: Luis Ángel (임 영민)
 * Date Created: 01-20-2021
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

#include <iostream>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <src/casl_geometry.h>
#include "CoarseGrid.h"

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
	const int FINE_MAX_RL = 8;

	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.
	const auto PHI_INTERP_MTHD = interpolation_method::quadratic;		// Phi interpolation method.
	const auto VEL_INTERP_MTHD = interpolation_method::quadratic;		// Velocity interpolation method.

	const double MIN_D = 0;				// Domain minimum and maximum values for each dimension.
	const double MAX_D = 1;
	const int NUM_TREES_PER_DIM = 1;	// Number of macro cells per dimension.

	const int NUM_ITER_VTK = 8;			// Save VTK files every NUM_ITER_VTK iterations.

	const double BAND_C = 2; 			// Minimum number of cells around interface in both grids.
	const double BAND_F = BAND_C * (1u << (FINE_MAX_RL - COARSE_MAX_RL));

	char msg[1024];						// Some string to write messages to standard ouput.

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		// To generate data sets we don't admit more than a single process to avoid race conditions.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		parStopWatch watch;
		watch.start();

		sprintf( msg, ">> Began to generate data sets for MAX_RL_COARSE = %d and MAX_RL_FINE = %d\n", COARSE_MAX_RL, FINE_MAX_RL );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

		// Define the velocity field arrays (valid for COARSE and FINE grids).
		UComponent uComponent;
		VComponent vComponent;
		const CF_DIM *velocityField[P4EST_DIM] = {&uComponent, &vComponent};

		// Domain information: a square with the same number of trees per dimension.
		const int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
		const double xyz_min[] = {MIN_D, MIN_D, MIN_D};
		const double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		const int periodic[] = {0, 0, 0};

		// Define the initial interface (valid for COARSE and FINE grids).
		geom::Sphere sphere( DIM( 0.5, 0.75, 0.0 ), 0.15 );

		// Declaration of the FINE macromesh via the brick and connectivity objects.
		my_p4est_brick_t brick_f;
		p4est_connectivity_t *connectivity_f;
		connectivity_f = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick_f, periodic );

		// Pointers to FINE p4est variables.
		p4est_t *p4est_f;
		p4est_ghost_t *ghost_f;
		p4est_nodes_t *nodes_f;

		// Create the FINE forest using a level-set as refinement criterion.
		p4est_f = my_p4est_new( mpi.comm(), connectivity_f, 0, nullptr, nullptr );
		splitting_criteria_cf_and_uniform_band_t lsSplittingCriterion_f( 1, FINE_MAX_RL, &sphere, BAND_F );
		p4est_f->user_pointer = &lsSplittingCriterion_f;

		// Refine and partition the FINE forest.
		for( int i = 0; i < FINE_MAX_RL; i++ )
		{
			my_p4est_refine( p4est_f, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est_f, P4EST_FALSE, nullptr );
		}

		// Create the FINE ghost (cell) and node structures.
		ghost_f = my_p4est_ghost_new( p4est_f, P4EST_CONNECT_FULL );
		nodes_f = my_p4est_nodes_new( p4est_f, ghost_f );

		// Initialize the FINE neighbor node structure.
		auto *hierarchy_f = new my_p4est_hierarchy_t( p4est_f, ghost_f, &brick_f );
		auto *nodeNeighbors_f = new my_p4est_node_neighbors_t( hierarchy_f, nodes_f );
		nodeNeighbors_f->init_neighbors();

		CoarseGrid coarseGrid( mpi, n_xyz, xyz_min, xyz_max, periodic, BAND_C, COARSE_MAX_RL, &sphere, velocityField );

		// Retrieve FINE grid size data.
		double dxyz_f[P4EST_DIM];
		double dxyz_min_f;
		double diag_min_f;
		get_dxyz_min( p4est_f, dxyz_f, dxyz_min_f, diag_min_f );

		// Declare data vectors and pointers for read/write.
		Vec phi_f;							// Level-set function function values for FINE grid.
		Vec phiExact_c;						// Exact level-set function values for COARSE grid.
		const double *phiReadPtr_c, *phiExactReadPtr_c;

		Vec vel_f[P4EST_DIM];				// Veloctiy field for FINE grid.

		// Allocate memory for FINE and COARSE Vecs.
		ierr = VecCreateGhostNodes( p4est_f, nodes_f, &phi_f );
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( coarseGrid.p4est, coarseGrid.nodes, &phiExact_c );
		CHKERRXX( ierr );
		for( auto& dir : vel_f )
		{
			ierr = VecCreateGhostNodes( p4est_f, nodes_f, &dir );
			CHKERRXX( ierr );
		}

		// Sample the level-set function at t = 0 at all independent nodes of FINE grid.
		sample_cf_on_nodes( p4est_f, nodes_f, sphere, phi_f );

		// Sample the exact level-set function at t = 0 at all independent nodes of COARSE grid.
		sample_cf_on_nodes( coarseGrid.p4est, coarseGrid.nodes, sphere, phiExact_c );

		// Sample the velocity field at t = 0 at all independent nodes of FINE grid.
		for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
			sample_cf_on_nodes( p4est_f, nodes_f, *velocityField[dir], vel_f[dir] );

		// Save the initial grid and fields into vtk.
		coarseGrid.writeVTK( 0, phiExact_c );

		// Define time stepping variables.
		double tn_c = 0;							// Current time for COARSE grid.
		double tn_f = 0;							// Current time for FINE grid.
		bool hasVelSwitched = false;
		int iter = 0;
		int vtkIdx = 1;								// Index for post VTK files.
		int vtkIdxPrior = 0;						// Index for prior VTK files.
		const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm known analitically.
		double dt_c = CFL * coarseGrid.minCellWidth / MAX_VEL_NORM;		// deltaT for COARSE grid.
		double dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;					// FINE deltaT knowing that the CFL condition is
																		// (c * deltaT)/deltaX <= CFLN.

		// Advection loop.
		// For each COARSE step, there are 2^(FINE_MAX_RL - COARSE_MAX_RL) FINE steps.
		const int N_FINE_STEPS_PER_COARSE_STEP = 1u << (FINE_MAX_RL - COARSE_MAX_RL);
		while( tn_c + 0.1 * dt_c < DURATION )
		{
			// Clip leading COARSE time step if it's going to go over the final time.
			if( tn_c + dt_c > DURATION )
				dt_c = DURATION - tn_c;

			// Clip leading COARSE time step if it's going to go over half time.
			if( tn_c + dt_c > DURATION / 2.0 && !hasVelSwitched )
				dt_c = (DURATION / 2.0) - tn_c;

			// Update up to N_FINE_STEPS_PER_COARSE_STEP times the FINE grid.
			for( int step = 0; step < N_FINE_STEPS_PER_COARSE_STEP; step++ )
			{
				if( step == 3 )		// In last step, stretch the FINE step so that tn_f matches tn_c + dt_c.
				{
					dt_f = tn_c + dt_c - tn_f;
				}
				else if( tn_f + dt_f > tn_c + dt_c )		// Is FINE step going over COARSE step?
				{
					dt_f = tn_c + dt_c - tn_f;
					step = 3;
				}

				// Declare auxiliary FINE p4est objects; they will be updated during the semi-Lagrangian advection step.
				p4est_t *p4est_f1 = p4est_copy( p4est_f, P4EST_FALSE );
				p4est_ghost_t *ghost_f1 = my_p4est_ghost_new( p4est_f1, P4EST_CONNECT_FULL );
				p4est_nodes_t *nodes_f1 = my_p4est_nodes_new( p4est_f1, ghost_f1 );

				// Create FINE semi-lagrangian object.
				my_p4est_semi_lagrangian_t semiLagrangian_f( &p4est_f1, &nodes_f1, &ghost_f1, nodeNeighbors_f );
				semiLagrangian_f.set_phi_interpolation( PHI_INTERP_MTHD );
				semiLagrangian_f.set_velo_interpolation( VEL_INTERP_MTHD );

				// Advect the FINE level-set function one step, then update the grid.
				semiLagrangian_f.update_p4est_one_vel_step( vel_f, dt_f, phi_f, BAND_F );

				// Destroy old FINE forest and create new structures.
				p4est_destroy( p4est_f );
				p4est_f = p4est_f1;
				p4est_ghost_destroy( ghost_f );
				ghost_f = ghost_f1;
				p4est_nodes_destroy( nodes_f );
				nodes_f = nodes_f1;

				delete hierarchy_f;
				delete nodeNeighbors_f;
				hierarchy_f = new my_p4est_hierarchy_t( p4est_f, ghost_f, &brick_f );
				nodeNeighbors_f = new my_p4est_node_neighbors_t( hierarchy_f, nodes_f );
				nodeNeighbors_f->init_neighbors();

				// Reinitialize FINE level-set function.
				my_p4est_level_set_t levelSet_f( nodeNeighbors_f );
				levelSet_f.reinitialize_2nd_order( phi_f );

				// Advance FINE time.
				tn_f += dt_f;

				// Re-sample the FINE velocity field on new grid.
				for( int dir = 0; dir < P4EST_DIM; dir++ )
				{
					ierr = VecDestroy( vel_f[dir] );
					CHKERRXX( ierr );
					ierr = VecCreateGhostNodes( p4est_f, nodes_f, &vel_f[dir] );
					CHKERRXX( ierr );
					sample_cf_on_nodes( p4est_f, nodes_f, *velocityField[dir], vel_f[dir] );
				}
			}

			// Restore FINE time step size.
			dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;

			// Collect samples from COARSE grid: must be done before "advecting" it.
			coarseGrid.collectSamples( nodeNeighbors_f, phi_f, dt_c ); 	// Also allocates flagged nodes.
			if( iter >= vtkIdxPrior * NUM_ITER_VTK )
			{
				coarseGrid.writeVTK( vtkIdxPrior );						// Prior-advection is one vtk index behind.
				vtkIdxPrior++;
			}

			// "Advecting" COARSE grid by using the FINE grid as reference: updates internal phi and velocity vectors.
			coarseGrid.fitToFineGrid( nodeNeighbors_f, phi_f, velocityField );

			// Advance COARSE time step.
			tn_c += dt_c;
			tn_f = tn_c;												// Synchronize COARSE and FINE times.
			iter++;
			if( tn_c >= DURATION / 2.0 && !hasVelSwitched )
			{
				dt_c = CFL * coarseGrid.minCellWidth / MAX_VEL_NORM; // Restore COARSE time step to original definition.
				uComponent.switch_direction();
				vComponent.switch_direction();
				hasVelSwitched = true;
				ierr = PetscPrintf( mpi.comm(), "*** Velocity switched ***\n" );
				CHKERRXX( ierr );
			}

			// Re-sample the exact initial COARSE level-set function.
			ierr = VecDestroy( phiExact_c );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( coarseGrid.p4est, coarseGrid.nodes, &phiExact_c );
			CHKERRXX( ierr );
			sample_cf_on_nodes( coarseGrid.p4est, coarseGrid.nodes, sphere, phiExact_c );

			// Display iteration message.
			sprintf( msg, "    Iteration %04d: t = %1.4f \n", iter, tn_c );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Save to vtk format.
			if( iter >= vtkIdx * NUM_ITER_VTK || tn_c == DURATION )
			{
				coarseGrid.writeVTK( vtkIdx, phiExact_c );
				vtkIdx++;
			}
		}

		// Compute error L-inf norm on FINE grid.
		ierr = VecGetArrayRead( coarseGrid.phi, &phiReadPtr_c );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phiExact_c, &phiExactReadPtr_c );
		CHKERRXX( ierr );
		double error = 0;
		for( p4est_locidx_t n = 0; n < coarseGrid.nodes->num_owned_indeps; n++ )
		{
			if( fabs( phiReadPtr_c[n] ) < 4.0 * coarseGrid.minCellDiag )
				error = MAX( error, fabs( phiReadPtr_c[n] - phiExactReadPtr_c[n] ) );
		}
		int mpiret = MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() );
		SC_CHECK_MPI( mpiret );
		ierr = VecRestoreArrayRead( coarseGrid.phi, &phiReadPtr_c );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( phiExact_c, &phiExactReadPtr_c );
		CHKERRXX( ierr );

		// Destroy the dynamically allocated Vecs for FINE grid.
		ierr = VecDestroy( phi_f );
		CHKERRXX( ierr );
		for( auto& dir : vel_f )
		{
			ierr = VecDestroy( dir );
			CHKERRXX( ierr );
		}

		// Destroy the dynamically allocated Vecs for COARSE grid.
		ierr = VecDestroy( phiExact_c );
		CHKERRXX( ierr );

		// Destroy the dynamically allocated FINE p4est and my_p4est structures.
		delete hierarchy_f;
		delete nodeNeighbors_f;
		p4est_nodes_destroy( nodes_f );
		p4est_ghost_destroy( ghost_f );
		p4est_destroy( p4est_f );

		// Destroy the dynamically allocated FINE brick and connectivity structures.
		// Connectivity and Brick objects are the only ones that are not re-created in every iteration of
		// semi-Lagrangian advection.
		my_p4est_brick_destroy( connectivity_f, &brick_f );

		coarseGrid.destroy();

		sprintf( msg, "<< Finished data set generation after %f secs with error %f.\n", watch.get_duration_current(), error );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );
		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}
}