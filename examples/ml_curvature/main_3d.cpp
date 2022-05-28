/**
 * Test non-saddle and saddle neural networks on a Gaussian (i.e., *online inference*).  The canonical surface is a Monge patch
 *
 *                                           Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)),
 *
 * with zero means mu=mv=0, variances su^2 and sv^2, and height a.  As for the level-set function whose Gamma=Q(u,v)=0, phi < 0 for points
 * below Q(u,v), and phi > 0 for points above the Gaussian.  We simplify calculations by expressing query points in terms of the canonical
 * frame, which can be affected by a rigid-body transformation (i.e., translation and rotation).
 *
 * Theoretically, the Gaussian curvature can be positive and negative.  Thus, this data set can be used to test the neural network ability
 * on saddle samples.  In any case, if a point belongs to a non-saddle region (i.e., ih2kg>=C), it'll be negative-mean-curvature normalized
 * by taking the sign of ihk ONLY if requested.  On the other hand, if the point belongs to a (numerical) saddle region, its sample will
 * never be negative-mean-curvature-normalized.  In any case, we extract six samples per interface node by applying a sequence of
 * reorientations and reflections.  These make the center node's gradient components non-negative.  At inference  time, all six outputs are
 * averaged to improve accuracy.
 *
 * Negative-mean-curvature normalization, when applicable, depends on the sign of the linearly interpolated mean ihk at the interface.
 *
 * We export VTK data for visualization and compute mean and maximum absolute errors for validation on nodes next to Gamma.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: May 27, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "data_sets/gaussian/gaussian_3d.h"
#include <src/parameter_list.h>
#include <cassert>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>             maxHK( pl,   0.6, "maxHK"			 , "Desired max (absolute) dimensionless mean curvature at the peak. "
																	   "Must be in the open interval of (1/3, 2/3) (default: 0.6)" );
	param_t<u_char>             maxRL( pl,     6, "maxRL"			 , "Maximum level of refinement per unit-cube octree (default: 6)" );
	param_t<int>          reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<double>                 a( pl,   1.0, "a"				 , "Gaussian height (i.e., Q(0,0)) in the range of [16h, 64h] "
																	   "(default 1)" );
	param_t<double>         susvRatio( pl,   3.0, "susvRatio"	 	 , "The ratio su/sv in the range of [1, 3] (default: 3)" );
	param_t<bool>  		perturbOrigin( pl,  true, "perturbOrigin"	 , "Whether to perturb the Gaussian's frame randomly in [-h/2,+h/2]^3 "
																	   "(default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,     7, "randomState"	 	 , "Seed for canonical frame's random perturbations (default: 7)" );
	param_t<std::string>     nnetsDir( pl,   ".", "nnetsDir"		 , "Folder where nnets are found (default: build directory)" );
	param_t<double>       randomNoise( pl,  1e-4, "randomNoise"		 , "How much random noise to add to phi(x) as [+/-]h*randomNoise.  Use "
																	   "a negative value or 0 to disable this feature (default: 1e-4)" );

	try
	{
		////////////////////////////////////////////////////////// Parameter setup /////////////////////////////////////////////////////////

		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating Gaussian data set for offline evaluation of a trained error-correcting neural network" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		if( maxHK() <= 1./3 || maxHK() >= 2./3 )
			throw std::invalid_argument( "[CASL_ERROR] Desired max hk must be in the range of (1/3, 2/3)." );

		if( susvRatio() < 1 || susvRatio() > 3 )
			throw std::invalid_argument( "[CASL_ERROR] The ratio su/sv must be in the range of [1, 3]." );

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.

		if( a() < 16 * h || a() > 64 * h )
			throw std::invalid_argument( "[CASL_ERROR] Gaussian amplitude must be in the range of [16h, 64h] "
										 "= [" + std::to_string( 16 * h ) + ", " + std::to_string( 64 * h ) + "]." );

		////////////////////////// First, checking that we can load the neural networks and scalers appropriately //////////////////////////

		const std::string ROOT = nnetsDir() + "/3d/" + std::to_string( maxRL() );
#ifdef DEBUG
		const int N_TEST_SAMPLES = 1;
		FDEEP_FLOAT_TYPE testSamples[N_TEST_SAMPLES][K_INPUT_SIZE];		// Notice: using singles instead of doubles.
		for( int i = 0; i < K_INPUT_SIZE; i++ )
			testSamples[0][i] = (FDEEP_FLOAT_TYPE)i / 100;

		// Check first the scalers.
		kml::StandardScaler standardScaler( ROOT + "/non-saddle/k_std_scaler.json", false );
		standardScaler.transform( testSamples, N_TEST_SAMPLES );
		CHKERRXX( PetscPrintf( mpi.comm(), "\nStd-scaled data as float32:\n" ) );
		if( mpi.rank() == 0 )
		{
			std::cout << std::setprecision( 8 );
			for( int i = 0; i < K_INPUT_SIZE; i++ )
			{
				std::cout << testSamples[0][i] << ",\t";
				if( (i + 1) % 5 == 0 )
					std::cout << std::endl;
			}
		}

		kml::PCAScaler pcaScaler( ROOT + "/non-saddle/k_pca_scaler.json", false );
		pcaScaler.transform( testSamples, N_TEST_SAMPLES );
		CHKERRXX( PetscPrintf( mpi.comm(), "\nPCA-scaled data as float32:\n" ) );
		if( mpi.rank() == 0 )
		{
			std::cout << std::setprecision( 8 );
			for( int i = 0; i < K_INPUT_SIZE; i++ )
			{
				std::cout << testSamples[0][i] << ",\t";
				if( (i + 1) % 4 == 0 )
					std::cout << std::endl;
			}
		}
#endif
		// Load non-saddle neural network.
		kml::NeuralNetwork nnetNS( ROOT + "/non-saddle", h, false );
#ifdef DEBUG
		for( int i = 0; i < K_INPUT_SIZE; i++ )
			testSamples[0][i] = (FDEEP_FLOAT_TYPE)i / 100;				// Back to the initial test data.
		FDEEP_FLOAT_TYPE outputs[N_TEST_SAMPLES];
		nnetNS.predict( testSamples, outputs, N_TEST_SAMPLES );			// If there's an error, it'll be thrown here.
		CHKERRXX( PetscPrintf( mpi.comm(), "\nTest prediction: %.7f\n", outputs[0] ) );
#endif
		// Let's load the saddle neural network, too.
		kml::NeuralNetwork nnetSD( ROOT + "/saddle", h, false );

		///////////////////////////////////////////////////// Defining shape parameters ////////////////////////////////////////////////////

		std::mt19937 gen( randomState() );					// Engine used for random perturbations of the canonical frame: rotation and shift.
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );
		const double MAX_K = maxHK() / h;					// Now that we know the parameters are valid, find max hk and variances.
		const double SV2 = a() * (1 + SQR( susvRatio() )) / (2 * SQR( susvRatio() ) * MAX_K);
		const double SU2 = SQR( susvRatio() ) * SV2;

		Gaussian gaussian( a(), SU2, SV2 );

		double origin[P4EST_DIM] = {0, 0, 0};				// Gaussian's frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;

		double trackedMinHK = DBL_MAX;						// We want to track min and max mean |hk*| for debugging.
		double trackedMaxHK = 0;
		if( mpi.rank() == 0 )			// Only rank 0 perturbs the Gaussian's frame to create an affine-transformed level-set function.
		{
			if( perturbOrigin() )
			{
				std::uniform_real_distribution<double> uniformDistributionH_2( -h/2, +h/2 );
				for( auto& dim : origin )
					dim = uniformDistributionH_2( gen );
			}

			if( randomRotation() )		// Generate a random unit axis and its rotation angle.
			{
				std::uniform_real_distribution<double> uniformDist;
				rotAngle = 2 * M_PI * uniformDist( gen );
				double azimuthAngle = uniformDist( gen ) * 2 * M_PI;
				double polarAngle = acos( 2 * uniformDist( gen ) - 1 );
				rotAxis[0] = cos( azimuthAngle ) * sin( polarAngle );
				rotAxis[1] = sin( azimuthAngle ) * sin( polarAngle );
				rotAxis[2] = cos( polarAngle );
			}
		}
		SC_CHECK_MPI( MPI_Bcast( origin, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift and rotation.
		SC_CHECK_MPI( MPI_Bcast( &rotAngle, 1, MPI_DOUBLE, 0, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Bcast( rotAxis, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began testing hybrid curvature on a Gaussian surface online with a = %g, su^2 = %g, sv^2 = %g, max "
								 "|hk| = %g, and h = %g (level %i)\n", a(), SU2, SV2, maxHK(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Domain information.
		u_char octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		int periodic[P4EST_DIM] = {0, 0, 0};
		GaussianLevelSet *gLS = setupDomain( mpi, gaussian, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max );

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Create the forest using the Gaussian level-set as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, octMaxRL, gLS, 3.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		gLS->toggleCache( true );		// Turn on cache to speed up repeated distance computations.
		gLS->reserveCache( (size_t)pow( 0.75 * (xyz_max[0] - xyz_min[0]) / h, 3 ) );	// Reserve space in cache to improve hashing.
		for( int i = 0; i < octMaxRL; i++ )												// queries for grid points.
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
		ngbd->init_neighbors();

		// Verify mesh size.
		double dxyz[P4EST_DIM];
		get_dxyz_min( p4est, dxyz );
		assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == h );

		// A ghosted parallel PETSc vector to store level-set function values and where we computed exact signed distances.
		Vec phi = nullptr, exactFlag = nullptr;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );

		// Populate phi and compute exact distance for vertices within a (linearly estimated) shell around Gamma.  Reinitialization perturbs
		// the otherwise calculated exact distances.  Add noise if requested.
		gLS->evaluate( p4est, nodes, phi, exactFlag );

		if( randomNoise() > 0 )
			addRandomNoiseToLSFunction( phi, nodes, genNoise, randomNoiseDist );

		// Reinitialization.
		double reinitStartTime = watch.get_duration_current();
		CHKERRXX( PetscPrintf( mpi.comm(), "* Reinitializing...  " ) );
		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi, reinitIters() );
		double reinitTime = watch.get_duration_current() - reinitStartTime;
		CHKERRXX( PetscPrintf( mpi.comm(), "done after %.6f secs.\n", reinitTime ) );

		Vec sampledFlag, trueHK, hkError;	// These vectors distinguish sampled nodes along the interface and store the true hk on Gamma and its error.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &trueHK ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );

		// We won't use these samples -- we really just want to get the sampledFlag to filter nodes in the hybrid curvature computation.
		std::vector<std::vector<double>> samples;
		double trackedMaxErrors[P4EST_DIM];
		int nNumericalSaddles;
		gLS->collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
							 nNumericalSaddles, exactFlag, sampledFlag, nullptr, nullptr, nullptr, nullptr, nullptr, NAN, NAN,
							 nonSaddleMinIH2KG(), trueHK );
		gLS->clearCache();
		gLS->toggleCache( false );
		samples.clear();
		delete gLS;

		// Collecting error stats and performance metrics from numerical baseline curvature computation.
		double numTime, numMaxAbsError, numMeanAbsError;
		int numGridPoints;
		CHKERRXX( PetscPrintf( mpi.comm(), "* Evaluating numerical baseline...  " ) );
		numTime = numericalBaselineComputation( mpi, *ngbd, h, phi, trueHK, &watch, sampledFlag, numMaxAbsError, numMeanAbsError, numGridPoints );
		CHKERRXX( PetscPrintf( mpi.comm(), "done with the following stats:\n" ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Time (in secs)            = %.6f\n", numTime + reinitTime ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i\n", numGridPoints ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Mean absolute error       = %.6e\n", numMeanAbsError ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Maximum absolute error    = %.6e\n", numMaxAbsError ) );

		// Hybrid curvature vectors.
		Vec numCurvature, hybHK, hybFlag, normal[P4EST_DIM];
		CHKERRXX( VecDuplicate( phi, &numCurvature ) );	// Numerical mean curvature at the nodes.
		CHKERRXX( VecDuplicate( phi, &hybHK ) );		// Hybrid curvature at normal projections of interface nodes (masked with sampledFlag).
		CHKERRXX( VecDuplicate( phi, &hybFlag ) );		// Where we used the hybrid approach (should match sampledFlag).
		for( auto& dim : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

		// Compute hybrid (dimensionless) mean curvature.
		CHKERRXX( PetscPrintf( mpi.comm(), "* Computing hybrid mean curvature... " ) );
		kml::Curvature mlCurvature( &nnetNS, &nnetSD, h, 0.004, 0.007, nonSaddleMinIH2KG() );
		std::pair<double, double> durations = mlCurvature.compute( *ngbd, phi, normal, numCurvature, hybHK, hybFlag, true, &watch, sampledFlag );

		// Compute statistics.
		int nHybNodes = 0;								// In how many nodes did we use the hybrid approach?
		double maxAbsError = 0;
		double meanAbsError = 0;
		double *hkErrorPtr;
		CHKERRXX( VecGetArray( hkError, &hkErrorPtr ) );
		const double *hybHKReadPtr, *trueHKReadPtr, *sampledFlagReadPtr, *hybFlagReadPtr;
		CHKERRXX( VecGetArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecGetArrayRead( trueHK, &trueHKReadPtr ) );
		CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( hybFlag, &hybFlagReadPtr ) );
		foreach_local_node( n, nodes )
		{
			if( hybFlagReadPtr[n] == 1 )
			{
				if( sampledFlagReadPtr[n] != 0 )		// Filter did work.
				{
					hkErrorPtr[n] = ABS( trueHKReadPtr[n] - hybHKReadPtr[n] );
					meanAbsError += hkErrorPtr[n];
					maxAbsError = MAX( maxAbsError, hkErrorPtr[n] );
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

		// Reduce stats across processes.
		CHKERRXX( VecGhostUpdateBegin( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		CHKERRXX( VecGhostUpdateEnd( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nHybNodes, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &maxAbsError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &meanAbsError, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
		meanAbsError /= nHybNodes;

		CHKERRXX( PetscPrintf( mpi.comm(), "done with the following stats:\n" ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Time (in secs)            = %.6f\n", durations.second + reinitTime ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i (%i saddles)\n", nHybNodes, nNumericalSaddles ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Mean absolute error       = %.6e\n", meanAbsError ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Maximum absolute error    = %.6e\n", maxAbsError ) );

		// Export visual data.
		const double *phiReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		std::ostringstream oss;
		oss << "gaussian_online_test_lvl" << (int)maxRL();
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								5, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "hybHK", hybHKReadPtr,
								VTK_POINT_DATA, "trueHK", trueHKReadPtr,
								VTK_POINT_DATA, "hybFlag", hybFlagReadPtr,
								VTK_POINT_DATA, "hkError", hkErrorPtr );

		CHKERRXX( VecRestoreArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( trueHK, &trueHKReadPtr ) );
		CHKERRXX( VecRestoreArray( hkError, &hkErrorPtr ) );
		CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( hybFlag, &hybFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		// Clean up.
		CHKERRXX( VecDestroy( numCurvature ) );
		CHKERRXX( VecDestroy( hybHK ) );
		CHKERRXX( VecDestroy( hybFlag ) );
		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );
		CHKERRXX( VecDestroy( hkError ) );
		CHKERRXX( VecDestroy( trueHK ) );
		CHKERRXX( VecDestroy( sampledFlag ) );
		CHKERRXX( VecDestroy( exactFlag ) );
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		delete ngbd;
		delete hierarchy;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );

		CHKERRXX( PetscPrintf( mpi.comm(), "<< Done after %.2f secs.\n", watch.get_duration_current() ) );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}
