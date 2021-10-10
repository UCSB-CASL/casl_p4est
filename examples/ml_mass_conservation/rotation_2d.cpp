/**
 * Testing error-correcting neural network for semi-Lagrangian advection for rotation.
 * The test consists of a rigid-body rotation of a disk located at the top center of a [-1,+1]^2 domain.  The level-set
 * is advected using a velocity field that has been normalized so that its maximum magnitude is one within Omega.
 * In the end, the initial circular interface should be recovered when t = k*2*pi*r, where r is the original max
 * velocity magnitude and k is a positive integer denoting the number of rotations.
 *
 * Code is based on examples/level_set_advection/main_2d.cpp
 *
 * @cite C. Min and F. Gibou, A second order accurate level set method on non-graded adaptive cartesian grids, J.
 * 		 Comput. Phys., 225:300-321, 2007.  Rotation test appears on p. 310.
 *
 * Author: Luis Ángel (임 영민)
 * Created: June 29, 2021.
 * Updated: October 9, 2021.
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

///////////////////////////////////////// Velocity field (also divergence free) ////////////////////////////////////////

class UComponent : public CF_2
{
private:
	const double R;		// Normalizing factor.

public:
	explicit UComponent( double r ): R( r ) {}

	double operator()( double x, double y ) const override
	{
		return -y / R;
	}
};

class VComponent : public CF_2
{
private:
	const double R;		// Normalizing factor.

public:
	explicit VComponent( double r ): R( r ) {}

	double operator()( double x, double y ) const override
	{
		return x / R;
	}
};

////////////////////////////////////////////////// Auxiliary functions /////////////////////////////////////////////////

void writeVTK( int vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec phiExact, Vec hk, Vec howUpdated )
{
	char name[1024];
	PetscErrorCode ierr;

	const double *phiReadPtr, *phiExactReadPtr, *hkReadPtr, *howUpdatedReadPtr;	// Pointers to Vec contents.

	sprintf( name, "rotation_%d", vtkIdx );
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
	const int MAX_RL = 6;				// Grid's maximum refinement level.
	const int REINIT_NUM_ITER = 10;		// Number of iterations for level-set renitialization.
	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.

	const int MIN_D = -1;				// Domain minimum and maximum values for each dimension.
	const int MAX_D = 1;
	const int NUM_TREES_PER_DIM = 2;	// Number of macro cells per dimension.
	const int PERIODICITY = 0;			// Domain periodicity.
	const double VEL_NORM_FACTOR = M_SQRT2;

	const int NUM_ITER_VTK = 1;			// Save VTK files every NUM_ITER_VTK iterations.

	char msg[1024];						// Some string to write messages to standard ouput.

	// Setting up parameters from command line.
	param_list_t pl;
	param_t<int> mode ( pl, 1, "mode", "Execution mode: 0 - numerical, 1 - nnet (default: 1)");
	param_t<int> exportAllVTK (pl, 0, "exportAllVTK", "Export all VTK files: 0 - no (only first and last), 1 - yes (default: 0)" );
	param_t<double> rotations (pl, 1, "nRotations", "Number of rotations (default: 1.0)" );

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Rotation test" ) )
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

		sprintf( msg, ">> Began 2D rotation test with MAX_RL = %d in %s mode\n", MAX_RL, mode()? "NNET" : "NUMERICAL" );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

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
		const double CENTER[P4EST_DIM] = {DIM( 0, 0.75, 0 )};
		const double RADIUS = 0.15;
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
		get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );

		// Declare data vectors and pointers for read/write.
		Vec phi;							// Level-set function values (subject to reinitialization).
		Vec phiExact;						// Exact level-set function values.
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

		// Computing dimensionless curvature and normal vectors.
		Vec hk = nullptr;
		Vec normal[P4EST_DIM] = {nullptr, nullptr};
		computeHKAndNormals( dxyz_min, nodeNeighbors, phi, hk, normal );

		// Let's use a debugging vector to detect how were grid points updated at time tnp1: 0 if numerically and 1 if
		// nnet was used.  We'll add another state: 2, if vertex state is 1 and is protected in selective reinit.
		Vec howUpdated;
		ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated );
		CHKERRXX( ierr );

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

			// Create semi-Lagrangian object and advect.
			slml::SemiLagrangian *mlSemiLagrangian;
			my_p4est_semi_lagrangian_t *numSemiLagrangian;
			if( mode() && ABS(dt - dxyz_min) <= PETSC_MACHINE_EPSILON && !(iter % 2) )		// Use neural network in an alternate schedule and only if dt = dx.
			{
				mlSemiLagrangian = new slml::SemiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors, phi, false, nnet, iter );
				mlSemiLagrangian->updateP4EST( vel, dt, &phi, hk, normal, &howUpdated );
			}
			else
			{
				numSemiLagrangian = new my_p4est_semi_lagrangian_t( &p4est_np1, &nodes_np1, &ghost_np1, nodeNeighbors );
				numSemiLagrangian->set_phi_interpolation( interpolation_method::quadratic );
				numSemiLagrangian->set_velo_interpolation( interpolation_method::quadratic );
				numSemiLagrangian->update_p4est( vel, dt, phi, nullptr, nullptr, MASS_BAND_HALF_WIDTH );

				ierr = VecDestroy( howUpdated );		// For numerical method, howUpdated flag is zero everywhere.
				CHKERRXX( ierr );
				ierr = VecCreateGhostNodes( p4est_np1, nodes_np1, &howUpdated );
				CHKERRXX( ierr );
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
				const double *phiReadPtr;
				ierr = VecGetArrayRead( phi, &phiReadPtr );
				CHKERRXX( ierr );

				// Selective reinitialization of level-set function: protect nodes updated with the nnet whose level-set
				// value is negative and are immediately next to Gamma^np1.
				Vec mask;
				ierr = VecCreateGhostNodes( p4est, nodes, &mask );		// Mask vector to flag updatable nodes.
				CHKERRXX( ierr );

				double *howUpdatedPtr;
				ierr = VecGetArray( howUpdated, &howUpdatedPtr );
				CHKERRXX( ierr );

				double *maskPtr;
				ierr = VecGetArray( mask, &maskPtr );
				CHKERRXX( ierr );

				for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )	// No need to check all independent nodes.
					maskPtr[n] = 1;						// Initially, all are 1 => updatable.

				NodesAlongInterface nodesAlongInterface( p4est, nodes, nodeNeighbors, MAX_RL );
				std::vector<p4est_locidx_t> indices;
				nodesAlongInterface.getIndices( &phi, indices );

				int numMaskedNodes = 0;
				for( const auto& n : indices )			// Now, check only points next to Gamma^np1.
				{
					if( howUpdatedPtr[n] == 1 && phiReadPtr[n] <= 0 )
					{
						numMaskedNodes++;
						maskPtr[n] = 0;					// 0 => nonupdatable.
						howUpdatedPtr[n] = 2;
					}
				}

				ierr = VecRestoreArray( mask, &maskPtr );
				CHKERRXX( ierr );

				ierr = VecRestoreArray( howUpdated, &howUpdatedPtr );
				CHKERRXX( ierr );

				ierr = VecRestoreArrayRead( phi, &phiReadPtr );
				CHKERRXX( ierr );

				ls.reinitialize_2nd_order_with_mask( phi, mask, numMaskedNodes, REINIT_NUM_ITER );

				ierr = VecDestroy( mask );
				CHKERRXX( ierr );
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

			// Re-sample the exact level-set function: actually, it follows the path defined by the solution of the
			// equivalent ODE describing the velocity field.
			ierr = VecDestroy( phiExact );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &phiExact );
			CHKERRXX( ierr );
			double x0 = CENTER[0] * cos( tn / VEL_NORM_FACTOR ) - CENTER[1] * sin( tn / VEL_NORM_FACTOR );
			double y0 = CENTER[0] * sin( tn / VEL_NORM_FACTOR ) + CENTER[1] * cos( tn / VEL_NORM_FACTOR );
			sphere.setCenter( DIM( x0, y0, 0 ) );
			sample_cf_on_nodes( p4est, nodes, sphere, phiExact );

			// Recompute dimensionless curvature and normal vectors.
			computeHKAndNormals( dxyz_min, nodeNeighbors, phi, hk, normal );

			// Display iteration message.
			sprintf( msg, "\tIteration %04d: t = %1.4f \n", iter, tn );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

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

		for( auto& dim : normal )
		{
			ierr = VecDestroy( dim );
			CHKERRXX( ierr );
		}

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

		// Destroy the dynamically allocated brick and connectivity structures.
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