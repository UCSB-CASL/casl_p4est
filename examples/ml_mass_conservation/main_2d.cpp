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
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_semi_lagrangian_ml.h>
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
#include "CoarseGrid.h"

/**
 * Write VTK files.
 * @param [in] vtkIdx File index.
 * @param [in] p4est Pointer to p4est object.
 * @param [in] nodes Pointer to nodes object.
 * @param [in] ghost Pointer to ghost struct.
 * @param [in] phi Phi parallel vector.
 * @param [in] phiExact Exact phi parallel vector.
 * @param [in] vel Array of two-dimensional velocity components.
 */
void writeVTK( int vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
				 const Vec& phi, const Vec& phiExact, const Vec vel[2] )
{
	char name[1024];
	PetscErrorCode ierr;

	const double *phiReadPtr, *phiExactReadPtr, *velReadPtr[2];		// Pointers to Vec contents.

	sprintf( name, "visualization_%d", vtkIdx );
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
							VTK_POINT_DATA, "phi_f", phiReadPtr,
							VTK_POINT_DATA, "phiExact_f", phiExactReadPtr,
							VTK_POINT_DATA, "vel_x_f", velReadPtr[0],
							VTK_POINT_DATA, "vel_y_f", velReadPtr[1]
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

	std::cout << " -> Saving vtu files in " + std::string( name ) + ".vtu" << std::endl;
}

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
		printf( ">> Began to generate data sets for MAX_RL_COARSE = %d and MAX_RL_FINE = %d\n",
		  	COARSE_MAX_RL, FINE_MAX_RL );
		watch.start();

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

//		CoarseGrid coarseGrid( mpi, n_xyz, xyz_min, xyz_max, periodic, BAND_C, COARSE_MAX_RL, &sphere, velocityField );

		// Retrieve FINE grid size data.
		double dxyz_f[P4EST_DIM];
		double dxyz_min_f;
		double diag_min_f;
		get_dxyz_min( p4est_f, dxyz_f, dxyz_min_f, diag_min_f );

		// Declare FINE data vectors and pointers for read/write.
		Vec phi_f, phiExact_f;				// Level-set function function values for FINE grid.
		const double *phiReadPtr_f, *phiExactReadPtr_f;

		Vec vel_f[P4EST_DIM];				// Veloctiy field for FINE grid.

		// Allocate memory for FINE the Vecs.
		ierr = VecCreateGhostNodes( p4est_f, nodes_f, &phi_f );
		CHKERRXX( ierr );
		ierr = VecCreateGhostNodes( p4est_f, nodes_f, &phiExact_f );
		CHKERRXX( ierr );
		for( auto& dir : vel_f )
		{
			ierr = VecCreateGhostNodes( p4est_f, nodes_f, &dir );
			CHKERRXX( ierr );
		}

		// Sample the level-set function at t = 0 at all independent nodes of FINE grid.
		sample_cf_on_nodes( p4est_f, nodes_f, sphere, phi_f );
		sample_cf_on_nodes( p4est_f, nodes_f, sphere, phiExact_f );

		// Sample the velocity field at t = 0 at all independent nodes of FINE grid.
		for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
			sample_cf_on_nodes( p4est_f, nodes_f, *velocityField[dir], vel_f[dir] );

		// Save the initial grid and fields into vtk.
		writeVTK( 0, p4est_f, nodes_f, ghost_f, phi_f, phiExact_f, vel_f );

		// Define time stepping variables.
		double tn = 0;								// Current time.
		bool hasVelSwitched = false;
		int iter = 0;
		int vtkIdx = 1;								// Index for VTK files.
		const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm known analitically.
		double dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;	// FINE deltaT knowing that the CFL condition is (c * deltaT)/deltaX <= CFLN.

		// Advection loop.
		while( tn + 0.1 * dt_f < DURATION )
		{
			// Clip time step if it's going to go over the final time.
			if( tn + dt_f > DURATION )
				dt_f = DURATION - tn;

			// Clip time step if it's going to go over the half time.
			if( tn + dt_f >= DURATION / 2.0 && !hasVelSwitched )
			{
				if( tn + dt_f > DURATION / 2.0 )
					dt_f = (DURATION / 2.0) - tn;
				uComponent.switch_direction();
				vComponent.switch_direction();
				hasVelSwitched = true;
			}

			// Declare auxiliary FINE p4est objects: these will be updated during the semi-Lagrangian advection step.
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

			// Advance time step and iteration counter.
			tn += dt_f;
			iter++;
			if( tn == DURATION / 2.0 )
				dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;		// Restore time step to original definition.

			// Re-sample the FINE velocity field on new grid.
			for( int dir = 0; dir < P4EST_DIM; dir++ )
			{
				ierr = VecDestroy( vel_f[dir] );
				CHKERRXX( ierr );
				ierr = VecCreateGhostNodes( p4est_f, nodes_f, &vel_f[dir] );
				CHKERRXX( ierr );
				sample_cf_on_nodes( p4est_f, nodes_f, *velocityField[dir], vel_f[dir] );
			}

			// Re-sample the exact initial FINE level-set function.
			ierr = VecDestroy( phiExact_f );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est_f, nodes_f, &phiExact_f );
			CHKERRXX( ierr );
			sample_cf_on_nodes( p4est_f, nodes_f, sphere, phiExact_f );

			// Display iteration message.
			char msg[1024];
			sprintf( msg, " Iteration %04d: t = %1.4f \n", iter, tn );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Save to vtk format.
			if( iter >= vtkIdx * NUM_ITER_VTK || tn == DURATION )
			{
				writeVTK( vtkIdx, p4est_f, nodes_f, ghost_f, phi_f, phiExact_f, vel_f );
				vtkIdx++;
			}
		}

		// Compute error L-inf norm on FINE grid and store it once we finished a simulation split.
		ierr = VecGetArrayRead( phi_f, &phiReadPtr_f );
		CHKERRXX( ierr );
		ierr = VecGetArrayRead( phiExact_f, &phiExactReadPtr_f );
		CHKERRXX( ierr );
		double error = 0;
		for( p4est_locidx_t n = 0; n < nodes_f->num_owned_indeps; n++ )
		{
			if( fabs( phiReadPtr_f[n] ) < 4.0 * diag_min_f )
				error = MAX( error, fabs( phiReadPtr_f[n] - phiExactReadPtr_f[n] ) );
		}
		int mpiret = MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() );
		SC_CHECK_MPI( mpiret );
		ierr = VecRestoreArrayRead( phi_f, &phiReadPtr_f );
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( phiExact_f, &phiExactReadPtr_f );
		CHKERRXX( ierr );

		// Destroy the dynamically allocated Vecs for FINE grid.
		ierr = VecDestroy( phi_f );
		CHKERRXX( ierr );
		ierr = VecDestroy( phiExact_f );
		CHKERRXX( ierr );
		for( auto& dir : vel_f )
		{
			ierr = VecDestroy( dir );
			CHKERRXX( ierr );
		}

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

//		coarseGrid.destroy();

		printf( "<< Finished data set generation after %f secs with error %f.\n", watch.get_duration_current(), error );
		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}
}

