/**
 * Convergence test for non-saddle and saddle neural networks on a sphere of fixed radius.  The surface is represented implicitly by
 *
 *                                           phi(x,y,z) = sqrt(x^2 + y^2 + z^2) - R^2,
 *
 * where R is constant across grid resolutions.
 *
 * This is the test on section 3.1 of "A hybrid particle volume-of-fluid method for curvature estimation in multiphase flows" by Karnakov,
 * Litvinov, and Koumoutsakos.  We compute the relative mean and maximum absolute errors on nodes next to Gamma for 100 random centers.
 *
 * TODO: Increase test to level 11, add option to compute errors relative or absolute (i.e., without dividing by hk*).
 *
 * Developer: Luis Ángel.
 * Created: June 9, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <random>
#include "../data_sets/ellipsoid/ellipsoid_3d.h"
#include "test_utils_3d.h"
#include <src/parameter_list.h>


int getNumStatsAndFlag( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi,
						const double& h, const u_short& octMaxRL, const double xyzMin[P4EST_DIM], const double xyzMax[P4EST_DIM],
						const double& trueHK, double& numRelL2Norm, double& numRelLInftyNorm, int& nNumericalSaddles, Vec sampledFlag,
						const double& nonSaddleMinIH2KG=-7e-6 );
void getHybStats( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const double& h,
				  const double& trueHK, double& hybRelL2Norm, double& hybRelLInftyNorm, int& nHybNodes, const kml::NeuralNetwork *nnetNS,
				  const kml::NeuralNetwork *nnetSD, Vec sampledFlag, const double& nonSaddleMinIH2KG=-7e-6, const u_short& nnetsMaxRL=0 );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<u_short>          loMaxRL( pl,     7, "loMaxRL"			 , "Lower-bound maximum level of refinement per unit-cube octree "
																	   "(default: 7)" );
	param_t<u_short>          upMaxRL( pl,    10, "upMaxRL"			 , "Upper-bound maximum level of refinement per unit-cube octree "
																	   "(default: 10)" );
	param_t<double>                 R( pl,2./128, "R"				 , "Radius (default: 2/128) " );
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
			CHKERRXX( PetscPrintf( mpi.comm(), header, maxRL ) );
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
				nGridPoints += getNumStatsAndFlag( p4est, nodes, ngbd, phi, h, octMaxRL, xyz_min, xyz_max, trueHK, numRelL2Norm,
												   numRelLInftyNorm, nNumericalSaddles, sampledFlag, nonSaddleMinIH2KG() );

				// Accumulate hybrid statistics.
				getHybStats( p4est, nodes, ngbd, phi, h, trueHK, hybRelL2Norm, hybRelLInftyNorm, nHybNodes, &nnetNS, &nnetSD, sampledFlag,
							 nonSaddleMinIH2KG(), nnetsMaxRL() );

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
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numRelLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &numRelL2Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			numRelL2Norm = sqrt( numRelL2Norm / nGridPoints );
			CHKERRXX( PetscPrintf( mpi.comm(), "   Numerical baseline:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i\n", nGridPoints ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Rel L2 error norm         = %.6e\n", numRelL2Norm ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Rel LInfty error norm     = %.6e\n", numRelLInftyNorm ) );

			// ...then the hybrid approach.
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nHybNodes, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nNumericalSaddles, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybRelLInftyNorm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
			SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &hybRelL2Norm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
			hybRelL2Norm = sqrt( hybRelL2Norm / nHybNodes );
			CHKERRXX( PetscPrintf( mpi.comm(), "   Hybrid approach:\n" ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i\n", nHybNodes ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of num saddles     = %i\n", nNumericalSaddles ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Rel L2 error norm         = %.6e\n", hybRelL2Norm ) );
			CHKERRXX( PetscPrintf( mpi.comm(), "   - Rel LInfty error norm     = %.6e\n", hybRelLInftyNorm ) );

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


/**
 * Accumulate stats for hybrid approach.
 * @note No metric reduction across processes occurs here --everything is local.
 * @param [in] p4est P4est structure.
 * @param [in] nodes Nodes structure.
 * @param [in] ngbd Neighborhood structure.
 * @param [in] phi Nodal level-set values.
 * @param [in] h Mesh size.
 * @param [in] trueHK Expected dimensionless mean curvature.
 * @param [in,out] hybRelL2Norm Cumulative sum of ((k - k*)/k*)^2 terms to compute relative L2 norm.
 * @param [in,out] hybRelLInftyNorm Running value of maximum |k - k*|/|k*| error.
 * @param [in,out] nHybNodes Number of nodes where we used the hybrid approach.
 * @param [in] nnetNS Neural network for non-saddle samples.
 * @param [in] nnetSD Neural network for saddle samples.
 * @param [in] sampledFlag Vector of >= 1 values for nodes where we'll use the hybrid approach.
 * @param [in] nonSaddleMinIH2KG Min numerical dimensionless Gaussian curvature (at Gamma) for numerical non-saddle samples.
 * @param [in] nnetsMaxRL Set to <= 0 to use nnets corresponding to each max RL; if > 0, use a single nnet for all grid resolutions.
 */
void getHybStats( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const double& h,
				  const double& trueHK, double& hybRelL2Norm, double& hybRelLInftyNorm, int& nHybNodes, const kml::NeuralNetwork *nnetNS,
				  const kml::NeuralNetwork *nnetSD, Vec sampledFlag, const double& nonSaddleMinIH2KG, const u_short& nnetsMaxRL )
{
	// Hybrid curvature vectors.
	Vec numCurvature, hybHK, hybFlag, normal[P4EST_DIM];
	CHKERRXX( VecDuplicate( phi, &numCurvature ) );	// Numerical mean curvature at the nodes.
	CHKERRXX( VecDuplicate( phi, &hybHK ) );		// Hybrid curvature at normal projections of interface nodes (masked with sampledFlag).
	CHKERRXX( VecDuplicate( phi, &hybFlag ) );		// Where we used the hybrid approach (should match sampledFlag).
	for( auto& dim : normal )
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

	kml::Curvature mlCurvature( nnetNS, nnetSD, h, 0.004, 0.007, nonSaddleMinIH2KG, nnetsMaxRL > 0 );
	mlCurvature.compute( *ngbd, phi, normal, numCurvature, hybHK, hybFlag, true, nullptr, sampledFlag );

	// Compute statistics.
	const double *hybHKReadPtr, *sampledFlagReadPtr, *hybFlagReadPtr;
	CHKERRXX( VecGetArrayRead( hybHK, &hybHKReadPtr ) );
	CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
	CHKERRXX( VecGetArrayRead( hybFlag, &hybFlagReadPtr ) );
	foreach_local_node( n, nodes )
	{
		if( hybFlagReadPtr[n] == 1 )
		{
			if( sampledFlagReadPtr[n] != 0 )		// Filter did work.
			{
				double relHKError = (trueHK - hybHKReadPtr[n]) / trueHK;
				hybRelL2Norm += SQR( relHKError );
				hybRelLInftyNorm = MAX( hybRelLInftyNorm, ABS( relHKError ) );
				nHybNodes++;
			}
			else
				std::cerr << "Error!!! Did you just compute the hybrid curvature for non-sampled node " << n << "?!" << std::endl;
		}
		else
		{
			if( sampledFlagReadPtr[n] != 0 )
				std::cerr << "Error!!! Node " << n << " was supposed to be considered for hybrid computation!" << std::endl;
		}
	}

	CHKERRXX( VecRestoreArrayRead( hybHK, &hybHKReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( hybFlag, &hybFlagReadPtr ) );

	// Clean up.
	CHKERRXX( VecDestroy( numCurvature ) );
	CHKERRXX( VecDestroy( hybHK ) );
	CHKERRXX( VecDestroy( hybFlag ) );
	for( auto& dim : normal )
		CHKERRXX( VecDestroy( dim ) );
}


/**
 * Collect numerical statistics and populate the sampledFlag vector to filter out invalid nodes during the hybrid computation of curvature.
 * @note No reduction across processes occurs for error stats within this function --everything is local.
 * @param [in] p4est P4est data structure.
 * @param [in] nodes Nodes data structure.
 * @param [in] ngbd Nodes' neighborhood data structure.
 * @param [in] phi Parallel vector with level-set values.
 * @param [in] h Mesh size.
 * @param [in] octMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
 * @param [in] xyzMin Domain's minimum coordinates.
 * @param [in] xyzMax Domain's maximum coordinates.
 * @param [in] trueHK Expected dimensionless mean curvature.
 * @param [in,out] numRelL2Norm Cumulative sum of ((k - k*)/k*)^2 terms to compute the relative L2 error norm.
 * @param [in,out] numRelLInftyNorm Running relative L^infty norm of |k - k*|/|k*| terms.
 * @param [in,out] nNumericalSaddles Running number of numerical saddles.
 * @param [out] sampledFlag Parallel vector with >= 1 for valid nodes (next to Gamma), 0s otherwise.
 * @param [in] nonSaddleMinIH2KG Min numerical dimensionless Gaussian curvature (at Gamma) for numerical non-saddle samples.
 * @returns Number of valid grid points next to Gamma.
 * @throws invalid_argument exception if the phi or sampledFlag vector is null.
 */
int getNumStatsAndFlag( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, const Vec& phi,
						const double& h, const u_short& octMaxRL, const double xyzMin[P4EST_DIM], const double xyzMax[P4EST_DIM],
						const double& trueHK, double& numRelL2Norm, double& numRelLInftyNorm, int& nNumericalSaddles, Vec sampledFlag,
						const double& nonSaddleMinIH2KG )
{
	if( !phi || !sampledFlag )
		throw std::invalid_argument( "getNumStatsAndFlag: phi and sampledFlag vectors can't be null!" );

	// Get indices for locally owned candidate nodes next to Gamma.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)octMaxRL );
	std::vector<p4est_locidx_t> indices;
	nodesAlongInterface.getIndices( &phi, indices );

	// Compute normal vectors and mean/Gaussian curvatures.
	Vec normals[P4EST_DIM],	kappaMG[2];
	for( auto& component : normals )
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &component ) );
	CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[0] ) );	// This is mean curvature, and
	CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[1] ) );	// this is Gaussian curvature.

	compute_normals_and_curvatures( *ngbd, phi, normals, kappaMG[0], kappaMG[1] );

	const double *phiReadPtr;								// We need access to phi to project points onto Gamma.
	CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

	const double *normalsReadPtr[P4EST_DIM];
	for( int i = 0; i < P4EST_DIM; i++ )
		CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );

	// Reset interface flag.
	double *sampledFlagPtr;
	CHKERRXX( VecGetArray( sampledFlag, &sampledFlagPtr ) );
	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		sampledFlagPtr[n] = 0;

	// Prepare mean and Gaussian curvature interpolation.
	my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
	kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );
	int nGridPoints = 0;
	for( const auto& n : indices )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est, nodes, xyz );
		if( ABS( xyz[0] - xyzMin[0] ) <= 4 * h || ABS( xyz[0] - xyzMax[0] ) <= 4 * h ||	// Skip nodes too close
			ABS( xyz[1] - xyzMin[1] ) <= 4 * h || ABS( xyz[1] - xyzMax[1] ) <= 4 * h ||	// to domain boundary.
			ABS( xyz[2] - xyzMin[2] ) <= 4 * h || ABS( xyz[2] - xyzMax[2] ) <= 4 * h )
			continue;

		std::vector<p4est_locidx_t> stencil;
		try
		{
			if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	//Does it have a valid stencil?
				continue;

			// Valid candidate grid node.
			for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
				xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
			double kappaMGValues[2];
			kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
			double ihkVal = h * kappaMGValues[0];
			double ih2kgVal = SQR( h ) * kappaMGValues[1];

			// Compute relative errors.
			double relHKError = (ihkVal - trueHK) / trueHK;
			numRelL2Norm += SQR( relHKError );
			numRelLInftyNorm = MAX( numRelLInftyNorm, ABS( relHKError ) );

			// Update flags: we should expect only non-saddles.
			if( ih2kgVal >= nonSaddleMinIH2KG )
				sampledFlagPtr[n] = 1;
			else
			{
				nNumericalSaddles++;
				sampledFlagPtr[n] = 2;
			}
			nGridPoints++;
		}
		catch( std::runtime_error &rt )
		{
			std::cerr << rt.what() << std::endl;
		}
		catch( std::exception &e )
		{
			std::cerr << e.what() << std::endl;
			throw std::runtime_error( e.what() );
		}
	}
	kappaMGInterp.clear();

	// Scatter node info across processes.
	CHKERRXX( VecRestoreArray( sampledFlag, &sampledFlagPtr ) );
	CHKERRXX( VecGhostUpdateBegin( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( sampledFlag, INSERT_VALUES, SCATTER_FORWARD ) );

	// Clean up.
	for( int i = 0; i < P4EST_DIM; i++ )
		CHKERRXX( VecRestoreArrayRead( normals[i], &normalsReadPtr[i] ) );
	CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

	CHKERRXX( VecDestroy( kappaMG[0] ) );
	CHKERRXX( VecDestroy( kappaMG[1] ) );
	for( auto& component : normals )
		CHKERRXX( VecDestroy( component ) );

	return nGridPoints;
}
