/**
 * Compute all the numerical curvatures in 3D: mean, Gaussian, and principal curvatures, for a spherical interface.
 * For a sphere of radius R, the numerical method computes 2*K_M = 2/R, where K_M is the mean curvature.  Also, the num-
 * erical method computes K_G = 1/R^2, where K_G is the Gaussian curvature.  To relate these with the principal curvatu-
 * res, we know that K_M = 0.5(k1 + k2) and K_G = k1*k2, where k1 and k2 are the principal curvatures.  In the case of
 * the sphere, k1 = k2 = 1/R.
 *
 * Based on example in curvature/main_2d.cpp and the draft in matlab/principal_curvatues.m
 *
 * Developer: Luis Ángel.
 * Created: March 3, 2022
 */

#include <src/my_p4est_to_p8est.h>

#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#include <src/casl_geometry.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>

template<typename M, typename G>
void computeCurvaturesErrors( const my_p4est_node_neighbors_t *ngbd, Vec phi,
							  Vec kappaM, const M& exactMeanK, Vec errorKappaM, double& maxErrorKappaM,
							  Vec kappaG, const G& exactGaussK, Vec errorKappaG, double& maxErrorKappaG )
{
	const double *kappaMReadPtr, *kappaGReadPtr, *phiReadPtr;
	double *errorKappaMPtr, *errorKappaGPtr;
	CHKERRXX( VecGetArrayRead( kappaM, &kappaMReadPtr ) );
	CHKERRXX( VecGetArrayRead( kappaG, &kappaGReadPtr ) );
	CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecGetArray( errorKappaM, &errorKappaMPtr ) );
	CHKERRXX( VecGetArray( errorKappaG, &errorKappaGPtr ) );

	double diag_min = p4est_diag_min( ngbd->get_p4est() );
	maxErrorKappaM = 0;
	maxErrorKappaG = 0;
	double xyz[P4EST_DIM];
	foreach_node( n, ngbd->get_nodes() ) 							// Checks *all independent* nodes.
	{
		if( ABS( phiReadPtr[n] ) < diag_min )						// Look only immediately next to Gamma.
		{
			node_xyz_fr_n( n, ngbd->get_p4est(), ngbd->get_nodes(), xyz );
			const double kM = exactMeanK( xyz );					// Exact (doubled) mean curvature.
			const double kG = exactGaussK( xyz );					// Exact Gaussian curvature.
			errorKappaMPtr[n] = ABS( kappaMReadPtr[n] - kM );
			errorKappaGPtr[n] = ABS( kappaGReadPtr[n] - kG );
			maxErrorKappaM = MAX( maxErrorKappaM, errorKappaMPtr[n] );
			maxErrorKappaG = MAX( maxErrorKappaG, errorKappaGPtr[n] );
		}
	}

	// Get max errors across processes.
	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &maxErrorKappaM, 1, MPI_DOUBLE, MPI_MAX, ngbd->get_p4est()->mpicomm ) );
	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &maxErrorKappaG, 1, MPI_DOUBLE, MPI_MAX, ngbd->get_p4est()->mpicomm ) );

	// Clean up.
	CHKERRXX( VecRestoreArray( errorKappaG, &errorKappaGPtr ) );
	CHKERRXX( VecRestoreArray( errorKappaM, &errorKappaMPtr ) );
	CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( kappaG, &kappaGReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( kappaM, &kappaMReadPtr ) );
}


int main( int argc, char** argv )
{
	// Prepare parallel enviroment.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );

	// Setting up parameters from command line.
	param_list_t pl;
	param_t<u_char>        minRL( pl,     1, "minRL"		, "Minimum level of refinement (default: 1)" );
	param_t<u_char>        maxRL( pl,     6, "maxRL"		, "Maximum level of refinement (default: 6)" );
	param_t<u_char>          nsp( pl,     4, "nsp"			, "Number of splits (default: 4)" );
	param_t<u_short> reinitIters( pl,    10, "reinitIters"	, "Number of iterations for reinitialization (default: 10)" );
	param_t<bool>            vtk( pl, false, "vtk"			, "Activate visualization exports (default: 0)" );

	// Loading parameters from command line.
	cmdParser cmd;
	pl.initialize_parser( cmd );
	if( cmd.parse( argc, argv, "Testing mean, Gaussian, and principal curvatures on a spherical interface" ) )
		return 0;
	pl.set_from_cmd_all( cmd );

	// Timer.
	parStopWatch w;
	w.start( "Running example: evaluation of all curvatures in three dimensions" );

	// p4est variables.
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *connectivity;
	my_p4est_brick_t brick;

	// Domain size information.
	const int n_xyz [P4EST_DIM] = {1, 1, 1};
	const double xyz_min [P4EST_DIM] = {-1, -1, -1};
	const double xyz_max [P4EST_DIM] = {1,  1,  1};
	const int periodic [P4EST_DIM] = {0, 0, 0};
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Refine based on distance to a spherical-interface level-set function.
	geom::Sphere sphere( 0, 0, 0, 0.5 );

	auto exactMeanK = [](const double xyz[P4EST_DIM]){		// Mean curvature lambda function (actually, doubled meanK).
		return 2. / sqrt( SUMD( SQR( xyz[0] ), SQR( xyz[1] ), SQR( xyz[2] ) ) );
	};

	auto exactGaussK = [](const double xyz[P4EST_DIM]){		// Gaussian curvature lambda function.
		return 1. / SUMD( SQR( xyz[0] ), SQR( xyz[1] ), SQR( xyz[2] ) );
	};

	double err[2][nsp()];
	for( int s = 0; s < nsp(); s++)
	{
		// Create, refine, and partition the forest.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		splitting_criteria_cf_t sp( minRL() + s, maxRL() + s, &sphere, 2.0 );
		p4est->user_pointer = &sp;
		my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create ghost layer and node structure.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Create neighbor node structure.
		auto hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
		ngbd->init_neighbors();

		Vec phi, kappaM, errorKappaM, kappaG, errorKappaG, kappa12[2];
		Vec normal[P4EST_DIM];
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaM ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &errorKappaM ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaG ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &errorKappaG ) );
		for( auto& kappa : kappa12 )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappa ) );
		for( auto& dim : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

		// Compute level-set values.
		sample_cf_on_nodes( p4est, nodes, sphere, phi );

		// Reinitialize.
		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi, reinitIters() );

		// Before, I was using compute_normals(), and then using these unit-length normals to compute curvature with
		// compute_mean_curvature( ngbd, phi, normal, kappa ) <- this yielded higher error than nonnormalized gradient!
		// Instead, use compute_mean_curvature(ngbd, normal, kappa) to get mean curvature as the divergence of the unit
		// normals.
		compute_normals_and_curvatures( *ngbd, phi, normal, kappaM, kappaG, kappa12 );
		computeCurvaturesErrors( ngbd, phi,
								 kappaM, exactMeanK, errorKappaM, err[0][s],
								 kappaG, exactGaussK, errorKappaG, err[1][s] );

		if( vtk() )
		{
			const double *phiReadPtr, *errorKappaMReadPtr;
			CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
			CHKERRXX( VecGetArrayRead( errorKappaM, &errorKappaMReadPtr ) );

			char filename[FILENAME_MAX];
			sprintf( filename, "all_curvatures_errors.%d", s );
			my_p4est_vtk_write_all( p4est, nodes, ghost, P4EST_TRUE, P4EST_TRUE, 2, 0, filename,
									VTK_POINT_DATA, "phi", phiReadPtr,
									VTK_POINT_DATA, "errorKappaM", errorKappaM );

			CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
			CHKERRXX( VecRestoreArrayRead( errorKappaM, &errorKappaMReadPtr ) );
		}

		CHKERRXX( PetscPrintf( mpi.comm(), "Resolution: (%d,%d)\n", minRL() + s, maxRL() + s ) );
		if(s > 0)
		{
			CHKERRXX( PetscPrintf( mpi.comm(), "Error K_M = %e, order = %f\n", err[0][s], log2( err[0][s-1] / err[0][s] ) ) );
		}
		else
		{
			CHKERRXX( PetscPrintf( mpi.comm(), "Error K_M = %e\n", err[0][s] ) );
		}

		CHKERRXX( PetscPrintf( mpi.comm(), "\n" ) );

		// Clean up.
		CHKERRXX( VecDestroy( phi ) );
		CHKERRXX( VecDestroy( kappaM ) );
		CHKERRXX( VecDestroy( errorKappaM ) );
		CHKERRXX( VecDestroy( kappaG ) );
		CHKERRXX( VecDestroy( errorKappaG ) );

		for( auto& kappa : kappa12 )
			CHKERRXX( VecDestroy( kappa ) );

		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );

		// Destroy the structures.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		delete ngbd;
		delete hierarchy;
	}

	my_p4est_brick_destroy( connectivity, &brick );
	w.stop(); w.read_duration();
}

