/**
 * Convergence test for non-saddle and saddle neural networks on a sphere of fixed radius.  The surface is represented implicitly by
 *
 *                                           phi(x,y,z) = sqrt(x^2 + y^2 + z^2) - R,
 *
 * where R is constant across grid resolutions.
 *
 * This is the test on section 3.1 of "A hybrid particle volume-of-fluid method for curvature estimation in multiphase flows" by Karnakov,
 * Litvinov, and Koumoutsakos.  We compute the (optional) relative mean and maximum absolute errors on nodes next to Gamma for 100 random
 * centers.
 *
 * Developer: Luis Ángel.
 * Created: June 9, 2022.
 * Updated: June 10, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <random>
#include "../data_sets/ellipsoid/ellipsoid_3d.h"
#include "test_utils_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<u_short>          loMaxRL( pl,     7, "loMaxRL"			 , "Lower-bound maximum level of refinement per unit-cube octree "
																	   "(default: 7)" );
	param_t<u_short>          upMaxRL( pl,    11, "upMaxRL"			 , "Upper-bound maximum level of refinement per unit-cube octree "
																	   "(default: 11)" );
	param_t<double>                 R( pl,2./128, "R"				 , "Radius (default: 2/128) " );
	param_t<bool>      relativeErrors( pl,  true, "relativeErrors"	 , "Whether to compute relative errors or absolute (default: true)" );
	param_t<u_short>      reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<u_short>       numSpheres( pl,   100, "numSpheres"		 , "Number of spheres to test per grid resolution (default: 100)" );
	param_t<u_int>        randomState( pl,    11, "randomState"		 , "Seed for random perturbations of the centers (default: 11)" );
	param_t<std::string>     nnetsDir( pl,   ".", "nnetsDir"		 , "Folder where nnets are found (default: build directory)" );
	param_t<u_int>         nnetsMaxRL( pl,     6, "nnetsMaxRL"		 , "Maximum level of refinement used for trained nnets; use <= 0 to "
																	   "use the nnets correspoding to each resolution (default: 6)" );
	param_t<double>       randomNoise( pl,     0, "randomNoise"		 , "How much random noise to add to phi(x) as [+/-]h*randomNoise.  Use "
																	   "a negative value or 0 to disable this feature (default: 0)" );

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
		if( loMaxRL() < 6 )
			throw std::invalid_argument( "[CASL_ERROR] Lower-bound maximum level of refinement must be at least 6." );
		if( upMaxRL() > 11 )
			throw std::invalid_argument( "[CASL_ERROR] Upper-bound maximum level of refinement must be at most 11." );
		if( loMaxRL() > upMaxRL() )
			throw std::invalid_argument( "[CASL_ERROR] Lower-bound maximum level of refinement must not exceed the upper-bound." );
		if( numSpheres() < 1 )
			throw std::invalid_argument( "[CASL_ERROR] Number of spheres per grid resolution must be at least 1." );
//		if( R() < 2. / 64 )
//			throw std::invalid_argument( "[CASL_ERROR] The radius should be at least 2/64" );

		///////////////////////////////////////////////////// Defining shape parameters ////////////////////////////////////////////////////

		std::mt19937 gen( randomState() );					// Engine used for random perturbation of the centers (only rank 0 uses it).
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		double origin[P4EST_DIM] = {0, 0, 0};				// Local frame origin (possible perturbed).

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began convergence test using the hybrid curvature routine on %i spheres with radius %.6e, for max "
								 "levels of refinement between %i and %i\n", numSpheres(), R(), loMaxRL(), upMaxRL() );

		for( u_short maxRL = loMaxRL(); maxRL <= upMaxRL(); maxRL++ )	// Go through each grid resolution.
		{
			parStopWatch watch;
			watch.start();
			char header[100];
			sprintf( header, "\n-----------------------------[ Resolution %2d ]---------------------------\n", maxRL );
			CHKERRXX( PetscPrintf( mpi.comm(), header ) );
			int dot = 0;
			double dotdt = numSpheres() / (double)(strlen( header ) - 2);

			// Load the neural network.
			const std::string ROOT = nnetsDir() + "/3d/" + std::to_string( nnetsMaxRL() > 0? nnetsMaxRL() : maxRL );
			const double nnetsH = 1. / (1 << (nnetsMaxRL() > 0? nnetsMaxRL() : maxRL));	// Highest spatial resolution for neural network.
			kml::NeuralNetwork nnetNS( ROOT + "/non-saddle", nnetsH, false );			// Load non-saddle neural network.
			kml::NeuralNetwork nnetSD( ROOT + "/saddle", nnetsH, false );				// Let's load the saddle neural network, too.

			// Domain information.
			u_short octMaxRL;
			int n_xyz[P4EST_DIM];
			double xyz_min[P4EST_DIM];
			double xyz_max[P4EST_DIM];
			int periodic[P4EST_DIM] = {0, 0, 0};

			const double h = 1. / (1 << maxRL);															// Effective mesh size for current grid resolution.
			const double trueHK = h / R();
			setupDomain( mpi, origin, R(), R(), R(), h, maxRL, octMaxRL, n_xyz, xyz_min, xyz_max );		// Using the ellipsoid domain function.
			std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );	// Optional random noise for phi(x) in multiples of h.
			std::uniform_real_distribution<double> uniformDistTrans( -h/2, +h/2 );						// Random translation.

			// Relative metrics we'll track over the numSpheres() centers for all processes.
			int nGridPoints = 0;
			double numRelL2Norm = 0;		// For numerical baseline.
			double numRelLInftyNorm = 0;
			double hybRelL2Norm = 0;		// For hybrid approach.
			double hybRelLInftyNorm = 0;
			int nNumericalSaddles = 0;
			int nHybNodes = 0;				// In how many points we have used the hybrid approach.
			for( u_short c = 0; c < numSpheres(); c++ )		// Evaluate interface points from numSpheres() spheres for each grid resolution.
			{
				// p4est variables and data structures.
				p4est_t *p4est;
				p4est_nodes_t *nodes;
				my_p4est_brick_t brick;
				p4est_ghost_t *ghost;
				p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

				// Generate a randomly centered sphere.
				double C[P4EST_DIM] = { uniformDistTrans( gen ), uniformDistTrans( gen ), uniformDistTrans( gen ) };
				SC_CHECK_MPI( MPI_Bcast( C, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift.

				// Define a level-set function with spherical interface.
				geom::Sphere sphere( C[0], C[1], C[2], R() );
				splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, octMaxRL, &sphere, 3.0 );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = (void *)( &splittingCriterion );

				// Refine and partition forest.
				for( int i = 0; i < octMaxRL; i++ )
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
				assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == h );

				// Ghosted parallel PETSc vector to store level-set values.
				Vec phi;
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

				// Evaluate level-set function, reinitialize it, and add noise if requested.
				sample_cf_on_nodes( p4est, nodes, sphere, phi );	// This computes exact-signed distances and populates nearest points (needed to know true hk).

				if( randomNoise() > 0 )			// Optional noise.
					addRandomNoiseToLSFunction( phi, nodes, genNoise, randomNoiseDist );

				my_p4est_level_set_t ls( ngbd );
				ls.reinitialize_2nd_order( phi, reinitIters() );

				// Acumulate numerical statistics and flag grid points next to Gamma where we want to use the hybrid approach.
				Vec sampledFlag;				// This vector will be used as filter.
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
				nGridPoints += test_utils::sphere::getNumStatsAndFlag( p4est, nodes, ngbd, phi, h, octMaxRL, xyz_min, xyz_max, trueHK,
																	   numRelL2Norm, numRelLInftyNorm, nNumericalSaddles, sampledFlag,
																	   nonSaddleMinIH2KG(), relativeErrors() );

				// Accumulate hybrid statistics.
				test_utils::sphere::getHybStats( p4est, nodes, ngbd, phi, h, trueHK, hybRelL2Norm, hybRelLInftyNorm, nHybNodes, &nnetNS,
												 &nnetSD, sampledFlag, nonSaddleMinIH2KG(), nnetsMaxRL(), relativeErrors() );

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

				SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
				if( floor( (c + 1) / dotdt ) > dot )
				{
					CHKERRXX( PetscPrintf( mpi.comm(), "." ) );
					dot++;
				}
				if( (c + 1) % numSpheres() == 0 )
					CHKERRXX( PetscPrintf( mpi.comm(), "\n" ) );
			}

			// Show results for current grid resolution: first numerical baseline.
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nGridPoints, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nNumericalSaddles, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numRelLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numRelL2Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			numRelL2Norm = sqrt( numRelL2Norm / nGridPoints );
			CHKERRXX( PetscPrintf( mpi.comm(), "** R/h = %f\n", R() / h ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "** Number of grid points = %i (%i saddles)\n", nGridPoints, nNumericalSaddles ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   Numerical baseline:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - %sL2 error norm         = %.6e\n", relativeErrors()? "Rel " : "", numRelL2Norm ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - %sLInfty error norm     = %.6e\n", relativeErrors()? "Rel " : "", numRelLInftyNorm ) );

			// ...then the hybrid approach.
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nHybNodes, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybRelLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybRelL2Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			hybRelL2Norm = sqrt( hybRelL2Norm / nHybNodes );
			CHKERRXX( PetscPrintf( mpi.comm(), "   Hybrid approach:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - %sL2 error norm         = %.6e\n", relativeErrors()? "Rel " : "", hybRelL2Norm ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - %sLInfty error norm     = %.6e\n", relativeErrors()? "Rel " : "", hybRelLInftyNorm ) );

			CHKERRXX( PetscPrintf( mpi.comm(), "   Completed %i resolution after %.2f secs:\n", maxRL, watch.get_duration_current() ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "-------------------------------------------------------------------------\n" ) );
			watch.stop();
		}
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}