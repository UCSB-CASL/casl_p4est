/**
 * Convergence test for non-saddle and saddle neural networks on a sphere of fixed radius.  The surface is represented implicitly by
 *
 *                                           phi(x,y,z) = x^2 + y^2 + z^2 - R^2,
 *
 * where R is constant across grid resolutions.  In this test, we use the domain [-1, 1]^3 and a uniform grid whose resolution is not a
 * power of 2.
 *
 * @note Metrics are reported without scaling by h, i.e., plain curvature.
 *
 * This is the test on section 5.3 of "Second-order accurate computation of curvatures in a level set framework using novel high-order
 * reinitialization schemes" by Du Chéné, Min, and Gibou.  We compute the mean and maximum absolute errors on nodes next to Gamma.
 *
 * Developer: Luis Ángel.
 * Created: June 10, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <random>
#include "test_utils_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<u_short>  numCellsPerSide( pl,    19, "numCellsPerSide"	 , "Number of cells per side (default: 19)" );
	param_t<u_short>        numSplits( pl,     3, "numSplits"		 , "Number of splits for convergence (i.e., how many times to dividy "
																	   "nCellsPerSide by 2 (default: 3)" );
	param_t<double>                 R( pl,0.2222, "R"				 , "Radius (default: 0.2222) " );
	param_t<u_short>      reinitIters( pl,    80, "reinitIters"		 , "Number of iterations for reinitialization (default: 80)" );
	param_t<std::string>     nnetsDir( pl,   ".", "nnetsDir"		 , "Folder where nnets are found (default: build directory)" );
	param_t<u_short>       nnetsMaxRL( pl,     6, "nnetsMaxRL"		 , "Maximum level of refinement used for trained nnets (default: 6)" );
	param_t<bool>       perturbCenter( pl,  true, "perturbCenter"	 , "Whether to perturb the sphere center randomly (default: true)" );
	param_t<u_int>        randomState( pl,    11, "randomState"		 , "Seed for random perturbations of the centers (default: 11)" );

	try
	{
		////////////////////////////////////////////////////////// Parameter setup /////////////////////////////////////////////////////////

		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Convergence test on a sphere of constant radius" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( numCellsPerSide() < 19 )
			throw std::invalid_argument( "[CASL_ERROR] Number of cells per side must be at least 19." );
		if( numSplits() < 1 )
			throw std::invalid_argument( "[CASL_ERROR] Number of splits must be at least 1." );
		if( reinitIters() < 1 )
			throw std::invalid_argument( "[CASL_ERROR] Number of redistancing steps must be at least 1." );
		if( nnetsMaxRL() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] MaxRL for neural network must be at least 1." );

		double h = 2. / numCellsPerSide();					// Initial mesh size.
		if( R() < 1.5 * h )
			throw std::invalid_argument( "[CASL_ERROR] Radius must be at least 1.5h." );

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began convergence test using the hybrid curvature routine on a sphere with radius %.6e, for %d spits, "
								 "starting at %d cells per side\n", R(), numSplits(), numCellsPerSide() );

		// Load the neural network.
		const std::string ROOT = nnetsDir() + "/3d/" + std::to_string( nnetsMaxRL() );
		const double nnetsH = 1. / (1 << nnetsMaxRL());								// Highest spatial resolution for neural network.
		kml::NeuralNetwork nnetNS( ROOT + "/non-saddle", nnetsH, false );			// Load non-saddle neural network.
		kml::NeuralNetwork nnetSD( ROOT + "/saddle", nnetsH, false );				// Let's load the saddle neural network, too.

		// Domain information.
		double xyz_min[P4EST_DIM] = {-1, -1, -1};
		double xyz_max[P4EST_DIM] = {+1, +1, +1};
		int n_xyz[P4EST_DIM] = {numCellsPerSide(), numCellsPerSide(), numCellsPerSide()};
		int periodic[P4EST_DIM] = {0, 0, 0};
		std::mt19937 gen( randomState() );					// Engine used for random perturbation of the centers (only rank 0 uses it).

		// The errors.
		double prevNumL1Norm = 0, prevNumLInftyNorm = 0;	// To compute the order of convergence, we need the errors from previous split.
		double prevHybL1Norm = 0, prevHybLInftyNorm = 0;

		for( u_short split = 0; split <= numSplits(); split++ )		// Subdivide mesh as many as numSplits() times.
		{
			parStopWatch watch;
			watch.start();
			CHKERRXX( PetscPrintf( mpi.comm(), "\n-----------------------------[ Split %d ]---------------------------\n", split ) );

			h = 2. / ((1 << split) * numCellsPerSide());							// Effective mesh size for current grid resolution.
			std::uniform_real_distribution<double> uniformDistTrans( -h/2, +h/2 );	// Random translation.
			const double trueHK = h / R();

			// Metrics we'll track for all processes.
			int nGridPoints = 0;
			int nNumericalSaddles = 0;
			int nHybNodes = 0;								// In how many points we have used the hybrid approach.
			double numL1Norm = 0, numLInftyNorm = 0;
			double hybL1Norm = 0, hybLInftyNorm = 0;

			// p4est variables and data structures.
			p4est_t *p4est;
			p4est_nodes_t *nodes;
			my_p4est_brick_t brick;
			p4est_ghost_t *ghost;
			p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

			// Generate a randomly centered sphere.
			double C[P4EST_DIM] = {0, 0, 0};
			if( perturbCenter() && mpi.rank() == 0 )
			{
				for( auto& dim : C )
					dim = uniformDistTrans( gen );
			}
			SC_CHECK_MPI( MPI_Bcast( C, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same center.

			// Define a level-set function with spherical interface.
			geom::Sphere sphere( C[0], C[1], C[2], R() );							// This one is used only for refinement.
			geom::SphereNSD sphereNsd( C[0], C[1], C[2], R() );						// And this one for the level-set function.
			splitting_criteria_cf_and_uniform_band_t splittingCriterion( split, split, &sphere, 3.0 );

			// Create the forest using a level set as refinement criterion.
			p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
			p4est->user_pointer = (void *)( &splittingCriterion );

			// Refine and partition forest.
			for( int i = 0; i < split; i++ )
			{
				my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
				my_p4est_partition( p4est, P4EST_FALSE, nullptr );
			}

			// Create the ghost (cell) and node structures.
			ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
			my_p4est_ghost_expand( p4est, ghost );
			nodes = my_p4est_nodes_new( p4est, ghost );

			// Initialize the neighbor nodes structure.
			auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
			auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
			ngbd->init_neighbors();

			// Verify mesh size.
			double dxyz[P4EST_DIM];
			get_dxyz_min( p4est, dxyz );
			assert( ABS( dxyz[0] - dxyz[1] ) <= EPS && ABS( dxyz[1] - dxyz[2] ) <= EPS && ABS( dxyz[2] - h ) <= EPS );

			// Ghosted parallel PETSc vector to store level-set values.
			Vec phi;
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

			// Evaluate level-set function and reinitialize it.
			sample_cf_on_nodes( p4est, nodes, sphereNsd, phi );

			my_p4est_level_set_t ls( ngbd );
			ls.reinitialize_2nd_order( phi, reinitIters() );

			// Acumulate numerical statistics and flag grid points next to Gamma where we want to use the hybrid approach.
			Vec sampledFlag;				// This vector will be used as filter.
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
			nGridPoints += test_utils::sphere::getNumStatsAndFlag( p4est, nodes, ngbd, phi, h, split, xyz_min, xyz_max, trueHK, numL1Norm,
																   numLInftyNorm, nNumericalSaddles, sampledFlag, nonSaddleMinIH2KG(),
																   false, false );

			// Accumulate hybrid statistics.
			test_utils::sphere::getHybStats( p4est, nodes, ngbd, phi, h, trueHK, hybL1Norm, hybLInftyNorm, nHybNodes, &nnetNS, &nnetSD,
											 sampledFlag, nonSaddleMinIH2KG(), nnetsMaxRL(), false, false );

			if( nGridPoints != nHybNodes )
				throw std::runtime_error( "Number of grid points next to Gamma and number of hybrid nodes has diverged!" );

			// Clean up.
			CHKERRXX( VecDestroy( sampledFlag ) );
			CHKERRXX( VecDestroy( phi ) );

			// Destroy the p4est and its connectivity structure.
			delete ngbd;
			delete hierarchy;
			p4est_nodes_destroy( nodes );
			p4est_ghost_destroy( ghost );
			p4est_destroy( p4est );
			my_p4est_brick_destroy( connectivity, &brick );

			// Show results for current grid: first numerical baseline.
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nGridPoints, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nNumericalSaddles, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numL1Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			numL1Norm = (numL1Norm / nGridPoints) / h;	// Report results without scaling by h.
			numLInftyNorm = numLInftyNorm / h;
			double L1Order = 0, LInftyOrder = 0;
			if( split > 0 )
			{
				L1Order = log( prevNumL1Norm / numL1Norm ) / log( 2 );
				LInftyOrder = log( prevNumLInftyNorm / numLInftyNorm ) / log( 2 );
			}
			CHKERRXX( PetscPrintf( mpi.comm(), "** Number of grid points = %i (%i saddles)\n", nGridPoints, nNumericalSaddles ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "** h = %.6e\n", h ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   Numerical baseline:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - L1 error norm         = %.6e\t(%.2f)\n", numL1Norm, L1Order ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - LInfty error norm     = %.6e\t(%.2f)\n", numLInftyNorm, LInftyOrder ) );

			// ...then the hybrid approach.
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nHybNodes, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybL1Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			hybL1Norm = (hybL1Norm / nHybNodes) / h;
			hybLInftyNorm = hybLInftyNorm / h;
			if( split > 0 )
			{
				L1Order = log( prevHybL1Norm / hybL1Norm ) / log( 2 );
				LInftyOrder = log( prevHybLInftyNorm / hybLInftyNorm ) / log( 2 );
			}
			CHKERRXX( PetscPrintf( mpi.comm(), "   Hybrid approach:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - L1 error norm         = %.6e\t(%.2f)\n", hybL1Norm, L1Order ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - LInfty error norm     = %.6e\t(%.2f)\n", hybLInftyNorm, LInftyOrder ) );

			CHKERRXX( PetscPrintf( mpi.comm(), "** Completed split %i after %.2f secs.\n", split, watch.get_duration_current() ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "-------------------------------------------------------------------\n" ) );
			watch.stop();

			// Save stats for next round.
			prevNumL1Norm = numL1Norm;
			prevNumLInftyNorm = numLInftyNorm;
			prevHybL1Norm = hybL1Norm;
			prevHybLInftyNorm = hybLInftyNorm;
		}
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}
