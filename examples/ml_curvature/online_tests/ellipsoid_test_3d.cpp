/**
 * Test non-saddle and saddle neural networks on an ellipsoid (i.e., *online inference*).  The surface is represented implicitly by
 *
 *                                           phi(x,y,z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1,
 *
 * where phi<0 for points inside the ellipsoid, phi>0 for points outside, and phi=0 for Gamma.  We simplify calculations by expressing
 * query points in terms of the canonical frame, which can be affected by a rigid transformation (i.e., translation and rotation).
 *
 * Theoretically, the Gaussian curvature should be positive everywhere on the surface, but because of numerical inaccuracies, we can get
 * negative Gaussian curvature samples.  In any case, if a point belongs to non-saddle region (i.e., ih2kg >= C), it'll be negative-mean-
 * curvature normalized by taking the sign of ihk.  On the other hand, if the point belongs to a (numerical) saddle region, its sample won't
 * be negative-mean-curvature normalized.  Here, we extract six samples per interface node by applying a sequence of reorientations and
 * reflections.  These make the center node' gradient components non-negative.  At inference time, all six outputs are averaged to improve
 * accuracy.
 *
 * Negative-mean-curvature normalization, when applicable, depends on the sign of the linearly interpolated mean ihk at the interface.
 *
 * We compute mean and maximum absolute errors for validation on nodes next to Gamma.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: May 29, 2022.
 * Updated: May 31, 2022.
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
	param_t<bool>   useExactSignedLSF( pl,  true, "useExactSignedLSF", "Whether to use an exact signed level-set function to populate the "
																	   "nodal phi vector (default: true" );
	param_t<u_short>            maxRL( pl,     6, "maxRL"			 , "Maximum level of refinement per unit-cube octree (default: 6)" );
	param_t<u_short>      reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<double>             maxHK( pl,  2./3, "maxHK"			 , "Expected maximum dimensionless mean curvature (default: 2/3)" );
	param_t<double>                 a( pl,  1.65, "a"				 , "Ellipsoid's x-semiaxis (default: 4)" );
	param_t<double>                 b( pl,  0.75, "b"				 , "Ellipsoid's y-semiaxis (default: 2.5)" );
	param_t<double>                 c( pl,   0.2, "c"				 , "Ellipsoid's z-semiaxis (default: 0.25)" );
	param_t<bool>       perturbOrigin( pl,  true, "perturbOrigin"	 , "Whether to perturb the ellipsoid's center randomly in [-h/2,+h/2]^3"
																	   " (default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,    11, "randomState"		 , "Seed for random perturbations of the canonical frame (default: 11)" );
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
		if( cmd.parse( argc, argv, "Online evaluation of a trained error-correcting neural network on an ellipsoidal surface" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		const double h = 1. / (1 << maxRL());							// Highest spatial resolution in x/y directions.

		if( a() < 1.5 * h || b() < 1.5 * h || c() < 1.5 * h )
			throw std::invalid_argument( "[CASL_ERROR] Any semiaxis must be larger than 1.5h" );

		//////////////////////////////////////////////// First, loading the neural networks ////////////////////////////////////////////////

		const std::string ROOT = nnetsDir() + "/3d/" + std::to_string( maxRL() );
		kml::NeuralNetwork nnetNS( ROOT + "/non-saddle", h, false );	// Load non-saddle neural network.
		kml::NeuralNetwork nnetSD( ROOT + "/saddle", h, false );		// Let's load the saddle neural network, too.

		///////////////////////////////////////////////////// Defining shape parameters ////////////////////////////////////////////////////

		std::mt19937 gen( randomState() );					// Engine used for random perturbations.
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );

		Ellipsoid ellipsoid( a(), b(), c() );				// An ellipsoid implicit function in canonical coords (i.e., untransformed).
		double maxK[P4EST_DIM];
		ellipsoid.getMaxMeanCurvatures( maxK );
		for( const auto& k : maxK )
		{
			if( h * k > maxHK() )
				throw std::invalid_argument( "[CASL_ERROR] One of the ellipsoid's max hk exceeds the expected max hk." );
		}

		double origin[P4EST_DIM] = {0, 0, 0};				// Local frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;
		test_utils::computeAffineTransformationParameters( mpi, origin, rotAxis, rotAngle, perturbOrigin(), randomRotation(), h, gen );

		const double maxHK_a = maxK[0] * h;
		const double maxHK_b = maxK[1] * h;
		const double maxHK_c = maxK[2] * h;

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began testing hybrid curvature on an ellipsoid data with hk_a = %g (a = %g), hk_b = %g (b = %g),"
								 " hk_c = %g (c = %g), and h = %g (level %i)\n", maxHK_a, a(), maxHK_b, b(), maxHK_c, c(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Domain information.
		u_short octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		int periodic[P4EST_DIM] = {0, 0, 0};
		setupDomain( mpi, origin, a(), b(), c(), h, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max );

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Define a level-set function with ellipsoidal interface.
		EllipsoidalLevelSet levelSet( &mpi, Point3( origin ), Point3( rotAxis ), rotAngle, &ellipsoid, h );
		splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, octMaxRL, &levelSet, 3.0 );

		// Enable and reserve space for caching.
		levelSet.toggleCache( true );
		auto discreteVolumeDiff = (size_t)(3. * 4./3 * M_PI * ((a() + 2*h)*(b() + 2*h)*(c() + 2*h) - (a() - 2*h)*(b() - 2*h)*(c() - 2*h))
										   / CUBE( h ) / mpi.size());
		levelSet.reserveCache( discreteVolumeDiff );

		// Create the forest using ellipsoidal level-set function as a refinement criterion.
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
		sample_cf_on_nodes( p4est, nodes, levelSet, phi );	// This computes exact-signed distances and populates nearest points (needed to know true hk).
		if( !useExactSignedLSF() )
			levelSet.evaluateNS( p4est, nodes, phi );		// This overwrites exact signed distances found before.

		if( randomNoise() > 0 )
			addRandomNoiseToLSFunction( phi, nodes, genNoise, randomNoiseDist );

		// Reinitialization.
		double reinitStartTime = watch.get_duration_current();
		CHKERRXX( PetscPrintf( mpi.comm(), "* Reinitializing...  " ) );
		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi, reinitIters() );
		double reinitTime = watch.get_duration_current() - reinitStartTime;
		CHKERRXX( PetscPrintf( mpi.comm(), "done after %.6f secs.\n", reinitTime ) );

		Vec sampledFlag, trueHK;	// These vectors distinguish sampled nodes along the interface and store the true hk on Gamma.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &trueHK ) );

		std::vector<std::vector<double>> samples;
		double trackedMaxErrors[P4EST_DIM];
		int nNumericalSaddles;
		double trackedMinHK = DBL_MAX;
		double trackedMaxHK = 0;
		levelSet.collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
								 nNumericalSaddles, sampledFlag, nullptr, nullptr, nullptr, nullptr, nullptr, nonSaddleMinIH2KG(), trueHK );
		levelSet.clearCache();
		levelSet.toggleCache( false );
		samples.clear();

		test_utils::collectErrorStats( mpi, h, maxRL(), watch, p4est, nodes, ngbd, ghost, phi, sampledFlag, trueHK, nonSaddleMinIH2KG(),
									   nNumericalSaddles, &nnetNS, &nnetSD, reinitTime );

		// Clean up.
		CHKERRXX( VecDestroy( trueHK ) );
		CHKERRXX( VecDestroy( sampledFlag ) );
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
