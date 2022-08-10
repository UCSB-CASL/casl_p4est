/**
 * Testing error-correcting neural network for semi-Lagrangian advection for a circular rotation patch.
 * The test consists of a rigid-body rotation inside a disk located at the center of a [-1,+1]^2 domain and no flow outside of the circle.
 * The exact solution is that the circle should not deform.  This test was suggested by one of the reviewers of the paper.
 * The velocity field inside the circle of radius 0.5 is given by U(u,v) = (1/0.5)*(-y, x), and max|U| = 1.  k internal revolutions are
 * completed at t = k*2*pi*r = k*pi, where r = 0.5 is the original max internal velocity magnitude, and k is a positive integer.
 *
 * Code is based on rotation_2d.cpp
 *
 * Author: Luis Ángel (임 영민)
 * Created: August 10, 2022.
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_node_neighbors.h>
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

#include <src/casl_geometry.h>
#include <src/parameter_list.h>

/////////////////////////////////////////////////// Velocity field (also divergence free) //////////////////////////////////////////////////

class UComponent : public CF_2
{
private:
	const double R;		// Circular patch radius; also, the normalizing factor.

public:
	explicit UComponent( double r ): R( r ) {}

	double operator()( double x, double y ) const override
	{
		if( SQR( x ) + SQR( y ) >= SQR( R ) )		// No flow outside circle.
			return 0;
		return -y / R;
	}
};

class VComponent : public CF_2
{
private:
	const double R;		// Circular patch radius; also, the normalizing factor.

public:
	explicit VComponent( double r ): R( r ) {}

	double operator()( double x, double y ) const override
	{
		if( SQR( x ) + SQR( y ) >= SQR( R ) )		// No flow outside circle.
			return 0;
		return x / R;
	}
};

//////////////////////////////////////////////////////////// Auxiliary functions ///////////////////////////////////////////////////////////

void writeVTK( int vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec phiExact, Vec hk, Vec howUpdated )
{
	char name[1024];
	PetscErrorCode ierr;

	const double *phiReadPtr, *phiExactReadPtr, *hkReadPtr, *howUpdatedReadPtr;	// Pointers to Vec contents.

	sprintf( name, "rotation_vortex_patch_%d", vtkIdx );
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( phiExact, &phiExactReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );
	ierr = VecGetArrayRead( howUpdated, &howUpdatedReadPtr );
	CHKERRXX( ierr );
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							4, 0, name,
							VTK_POINT_DATA, "phi", phiReadPtr,
							VTK_POINT_DATA, "phiExact", phiExactReadPtr,
							VTK_POINT_DATA, "hk", hkReadPtr,
							VTK_POINT_DATA, "howUpdated", howUpdatedReadPtr
	);
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( phiExact, &phiExactReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( howUpdated, &howUpdatedReadPtr );
	CHKERRXX( ierr );

	PetscPrintf( p4est->mpicomm, ":: Saved vtk files with index %02d ::\n", vtkIdx );
}

void computeHKAndNormals( const double& h, const my_p4est_node_neighbors_t *nbgd, const Vec& phi, Vec& hk, Vec normal[P4EST_DIM] )
{
	PetscErrorCode ierr;

	// Shortcuts.
	const p4est_t *p4est = nbgd->get_p4est();
	const p4est_nodes_t *nodes = nbgd->get_nodes();

	// Prepare output parallel vector with dimensionless curvature values.
	ierr = hk? VecDestroy( hk ) : 0;
	CHKERRXX( ierr );
	ierr = VecCreateGhostNodes( p4est, nodes, &hk );			// By default, all values are zero.
	CHKERRXX( ierr );

	for(int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = normal[dim]? VecDestroy( normal[dim] ) : 0;
		CHKERRXX( ierr );

		ierr = VecCreateGhostNodes( p4est, nodes, &normal[dim] );
		CHKERRXX( ierr );
	}

	// Compute curvature and normals for all points.  It'll be interpolated on the interface for valid points later.
	// The normals are computed for locally owned points only (no scattered) while curvature is scattered forward.
	compute_normals( *nbgd, phi, normal );
	compute_mean_curvature( *nbgd, normal, hk );

	double *hkPtr;
	ierr = VecGetArray( hk, &hkPtr );
	CHKERRXX( ierr );

	// Scaling curvature by h.
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
		hkPtr[n] *= h;

	ierr = VecRestoreArray( hk, &hkPtr );
	CHKERRXX( ierr );
}

/////////////////////////////////////////////////////////////// Main function //////////////////////////////////////////////////////////////

/**
 * Main function.
 * @param argc Number of input arguments.
 * @param argv Actual arguments.
 * @return 0 if process finished successfully, nonzero otherwise.
 */
int main( int argc, char** argv )
{
	// Main global variables.
	const int MAX_RL = 6;				// Grid's maximum refinement level.
	const int REINIT_NUM_ITER = 10;		// Number of iterations for level-set renitialization.
	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.

	const int MIN_D = -1;				// Domain minimum and maximum values for each dimension.
	const int MAX_D = 1;
	const int NUM_TREES_PER_DIM = 2;	// Number of macro cells per dimension.
	const int PERIODICITY = 0;			// Domain periodicity.
	const double VEL_NORM_FACTOR = 0.5;

	const int NUM_ITER_VTK = 1;			// Save VTK files every NUM_ITER_VTK iterations.

	char msg[1024];						// Some string to write messages to standard ouput.

	// Setting up parameters from command line.
	param_list_t pl;
	param_t<int> mode( pl, 1, "mode", "Execution mode: 0 - numerical, 1 - nnet (default: 1)");
	param_t<bool> exportAllVTK( pl, false, "exportAllVTK", "Export all VTK files (default: false)" );
	param_t<double> rotations( pl, 1, "nRotations", "Number of rotations (default: 1.0)" );

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Circular vortex patch test" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		const double DURATION = rotations() * (2 * M_PI * VEL_NORM_FACTOR);	// Duration of the simulation: one full cycle completes at 2*pi*r.

		// OpenMP verification.
		int nThreads = 0;
#pragma omp parallel reduction( + : nThreads ) default( none )
		nThreads += 1;

		std::cout << "Rank " << mpi.rank() << " can spawn " << nThreads << " thread(s)\n\n";

		// Loading error-correcting neural network if user has selected its option.
		const slml::NeuralNetwork *nnet = nullptr;
		if( mode() )
			nnet = new slml::NeuralNetwork( "/Users/youngmin/nnets", 1. / (1 << MAX_RL), false );

		// Let's continue with numerical computations.
		parStopWatch watch;
		watch.start();

		sprintf( msg, ">> Began 2D circular vortex patch test with MAX_RL = %d in %s mode\n", MAX_RL, mode()? "NNET" : "NUMERICAL" );
		CHKERRXX( PetscPrintf( mpi.comm(), msg ) );

		// Define the velocity field arrays.
		UComponent uComponent( VEL_NORM_FACTOR );
		VComponent vComponent( VEL_NORM_FACTOR );
		const CF_DIM *velocityField[P4EST_DIM] = {&uComponent, &vComponent};

		// Domain information: a square with the same number of trees per dimension.
		const int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
		const double xyz_min[] = {MIN_D, MIN_D, MIN_D};
		const double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		const int periodic[] = {PERIODICITY, PERIODICITY, PERIODICITY};

		// Define the initial interfaces: exact and non-signed distance function.
		const double CENTER[P4EST_DIM] = {DIM( 0, 0, 0 )};
		const double RADIUS = VEL_NORM_FACTOR;
		geom::SphereNSD sphereNsd( DIM( CENTER[0], CENTER[1], 0 ), RADIUS );
		geom::Sphere sphere( DIM( CENTER[0], CENTER[1], 0 ), RADIUS );

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
		splitting_criteria_cf_and_uniform_band_t lsSplittingCriterion( 1, MAX_RL, &sphereNsd, MASS_BAND_HALF_WIDTH );
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
		get_dxyz_min( p4est, dxyz, &dxyz_min, &diag_min );

		// Declare data vectors and pointers for read/write.
		Vec phi;							// Level-set function values (subject to reinitialization).
		Vec phiExact;						// Exact level-set function values.
		Vec vel[P4EST_DIM];					// Veloctiy field.

		// Allocate memory for parallel vectors.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phiExact ) );
		for( auto& dir : vel )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dir ) );

		// Sample the level-set functions at t = 0 at all independent nodes.
		sample_cf_on_nodes( p4est, nodes, sphereNsd, phi );
		sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

		// Sample the velocity field at t = 0 at all independent nodes.
		for( unsigned int dir = 0; dir < P4EST_DIM; dir++ )
			sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );

		// Reinitialize grid before we start advection as we are using a non-signed distance level-set function.
		my_p4est_level_set_t levelSet( nodeNeighbors );
		levelSet.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

		// Computing dimensionless curvature and normal vectors.
		Vec hk = nullptr;
		Vec normal[P4EST_DIM] = {nullptr, nullptr};
		computeHKAndNormals( dxyz_min, nodeNeighbors, phi, hk, normal );

		// Let's use a debugging vector to detect how were grid points updated at time tnp1: 0 if numerically and 1 if nnet was used.
		// We'll add another state: 2, if vertex state is 1 and is protected in selective reinit.
		Vec howUpdated;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &howUpdated ) );

		// Save the initial grid and fields into vtk (regardless of input command choice).
		writeVTK( 0, p4est, nodes, ghost, phi, phiExact, hk, howUpdated );

		// Define time stepping variables.
		double tn = 0;								// Current time.
		int iter = 0;
		int vtkIdx = 1;								// Index for post VTK files.
		const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm is known beforehand.
		double dt = CFL * dxyz_min / MAX_VEL_NORM;	// deltaT knowing that the CFL condition is c*dt/dx <= CFLN.

		// Advection loop.
		while( tn < DURATION )
		{
			// Clip step if it's going to go over the final time.
			if( tn + dt > DURATION )
				dt = DURATION - tn;

			// p4est objects at time tnp1; they will be updated during the semi-Lagrangian advection step.
			p4est_t *p4est_np1 = p4est_copy( p4est, P4EST_FALSE );
			p4est_ghost_t *ghost_np1 = my_p4est_ghost_new( p4est_np1, P4EST_CONNECT_FULL );
			p4est_nodes_t *nodes_np1 = my_p4est_nodes_new( p4est_np1, ghost_np1 );

			// Parallel vector flagging nodes in the (opposite) flow direction that we want to protect.
			const short int flipFlow = -1;
			Vec withTheFlow;
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &withTheFlow ) );

			// Create semi-Lagrangian object and advect.
			slml::SemiLagrangian *mlSemiLagrangian = nullptr;
			my_p4est_semi_lagrangian_t *numSemiLagrangian = nullptr;
			if( mode() && ABS(dt - dxyz_min) <= PETSC_MACHINE_EPSILON && !(iter % 2) )	// Use neural network in an alternate schedule, only if dt=dx.
			{
				mlSemiLagrangian = new slml::SemiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors, phi, false, nnet, iter );
				mlSemiLagrangian->updateP4EST( vel, dt, &phi, hk, normal, &howUpdated, &withTheFlow, flipFlow );
			}
			else
			{
				numSemiLagrangian = new my_p4est_semi_lagrangian_t( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors );
				numSemiLagrangian->set_phi_interpolation( interpolation_method::quadratic );
				numSemiLagrangian->set_velo_interpolation( interpolation_method::quadratic );
				numSemiLagrangian->update_p4est( vel, dt, phi, nullptr, nullptr, MASS_BAND_HALF_WIDTH );

				CHKERRXX( VecDestroy( howUpdated ) );		// For numerical method, howUpdated flag is zero everywhere.
				CHKERRXX( VecCreateGhostNodes( p4est_np1, nodes_np1, &howUpdated ) );
			}

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
			if( mode() && ABS(dt - dxyz_min) <= PETSC_MACHINE_EPSILON && !(iter % 2) )
			{
				const double *withTheFlowReadPtr;
				CHKERRXX( VecGetArrayRead( withTheFlow, &withTheFlowReadPtr ) );

				// Selective reinitialization of level-set function: protect nodes updated with the nnet in the opposite direction to the
				// flow (i.e.,lagging behind).
				Vec mask;
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &mask ) );		// Mask vector to flag updatable nodes.

				double *howUpdatedPtr;
				CHKERRXX( VecGetArray( howUpdated, &howUpdatedPtr ) );

				double *maskPtr;
				CHKERRXX( VecGetArray( mask, &maskPtr ) );

				int numMaskedNodes = 0;
				for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )	// No need to check all independent nodes.
				{
					if( howUpdatedPtr[n] == 1 && withTheFlowReadPtr[n] == 1 )
					{
						numMaskedNodes++;
						maskPtr[n] = 0;					// 0 => nonupdatable.
						howUpdatedPtr[n] = 2;
					}
					else
						maskPtr[n] = 1;					// 1 => updatable.
				}

				CHKERRXX( VecRestoreArray( mask, &maskPtr ) );
				CHKERRXX( VecRestoreArray( howUpdated, &howUpdatedPtr ) );
				CHKERRXX( VecRestoreArrayRead( withTheFlow, &withTheFlowReadPtr ) );

				ls.reinitialize_2nd_order_with_mask( phi, mask, numMaskedNodes, REINIT_NUM_ITER );

				CHKERRXX( VecDestroy( mask ) );
			}
			else
			{
				ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );
			}

			// Destroy semi-Lagrangian objects.
			if( mode() && ABS(dt - dxyz_min) <= PETSC_MACHINE_EPSILON && !(iter % 2) )
				delete mlSemiLagrangian;
			else
				delete numSemiLagrangian;

			CHKERRXX( VecDestroy( withTheFlow ) );

			// Advance time.
			tn += dt;
			dt = CFL * dxyz_min / MAX_VEL_NORM;						// Restore time step size.
			iter++;

			// Re-sample the velocity field on new grid.
			for( int dir = 0; dir < P4EST_DIM; dir++ )
			{
				CHKERRXX( VecDestroy( vel[dir] ) );
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &vel[dir] ) );
				sample_cf_on_nodes( p4est, nodes, *velocityField[dir], vel[dir] );
			}

			// Re-sample the exact level-set function.
			CHKERRXX( VecDestroy( phiExact ) );
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phiExact ) );
			sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

			// Recompute dimensionless curvature and normal vectors.
			computeHKAndNormals( dxyz_min, nodeNeighbors, phi, hk, normal );

			// Display iteration message.
			sprintf( msg, "\tIteration %04d: t = %1.4f \n", iter, tn );
			CHKERRXX( PetscPrintf( mpi.comm(), msg ) );

			// Save to vtk format (last file is always written, the others are written if exportAllVTK is true).
			if( ABS( tn - DURATION ) <= PETSC_MACHINE_EPSILON ||
				(exportAllVTK() && iter % NUM_ITER_VTK == 0) )
			{
				writeVTK( vtkIdx, p4est, nodes, ghost, phi, phiExact, hk, howUpdated );
				vtkIdx++;
			}
		}

		// Compute error L-1 and L-inf norms.
		const double *phiReadPtr, *phiExactReadPtr;
		int numPoints = 0;
		double cumulativeError = 0;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecGetArrayRead( phiExact, &phiExactReadPtr ) );
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

		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phiExact, &phiExactReadPtr ) );

		// Destroy parallel vectors.
		CHKERRXX( VecDestroy( howUpdated ) );
		CHKERRXX( VecDestroy( hk ) );
		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );
		CHKERRXX( VecDestroy( phi ) );
		for( auto& dir : vel )
			CHKERRXX( VecDestroy( dir ) );
		CHKERRXX( VecDestroy( phiExact ) );

		// Destroy p4est and my_p4est structures.
		delete hierarchy;
		delete nodeNeighbors;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );

		// Destroy the dynamically allocated brick and connectivity structures.
		// Connectivity and Brick objects are the only ones that are not re-created in every iteration of semi-Lagrangian advection.
		my_p4est_brick_destroy( connectivity, &brick );

		// Destroy neural network.
		delete nnet;

		sprintf( msg, "<< Finished after %.3f secs with:\n   mean abs error %.3e\n   max abs error %.3e\n   area %.3e (expected %.3e, loss %.2f%%%%)",
				 watch.get_duration_current(), l1Error, maxError, area, expectedArea, massLossPercentage );
		CHKERRXX( PetscPrintf( mpi.comm(), msg ) );
		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}

	return 0;
}