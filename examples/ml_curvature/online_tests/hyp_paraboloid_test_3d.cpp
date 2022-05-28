/**
 * Test non-saddle and saddle neural networks on a hyperbolic paraboloid (i.e., *online inference*).  The canonical surface is a Monge patch
 *
 *                                           Q(u,v) = a*u^2 - b*v^2,
 *
 * where a > 0 and b > 0.  As for the level-set function whose Gamma = Q(u,v) = 0, phi < 0 for points above Q(u,v), and phi > 0 for points
 * below Q(u,v).  We simplify calculations by expressing query points in terms of the canonical frame, which can be affected by rigid-body
 * transformations (i.e., a translation and rotation).
 *
 * Theoretically, the hyperbolic paraboloid Gaussian curvature is always negative (never 0).  Thus, its data set contains samples only
 * saddle regions (i.e., h2kg < 0).  If requested, we can apply negative-mean-curvature normalization selectively for each numerical
 * non-saddle sample (say we found some point for which ih2kg >= nonSaddleMinIH2KG).  In any case, we extract six samples per interface node
 * by applying a sequence of reorientations and reflections.  These make the center node's gradient components non-negative.  At inference
 * time, all six outputs are averaged to improve accuracy.
 *
 * The desired maximum |hk| can occur at two places if the parameter 1 <= r < 3, and a=rb or b=ra.  If the user picks an r factor that
 * yields maxima whose critical points are less than 1.5h apart, we abort the program as curvature becomes "under-resolved".
 *
 * Negative-mean-curvature normalization, when applicable, depends on the sign of the linearly interpolated mean ihk at the interface.
 *
 * We compute mean and maximum absolute errors for validation on nodes next to Gamma.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: May 27, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <random>
#include "../data_sets/hyp_paraboloid/hyp_paraboloid_3d.h"
#include "test_utils_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>             maxHK( pl,  0.35, "maxHK"		 	 , "Desired max (absolute) dimensionless mean curvature at the critical"
																	   " points.  Must be in the interval of [1/15, 2/3] (default: 0.35)" );
	param_t<u_char>             maxRL( pl,     6, "maxRL"		 	 , "Max level of refinement per unit-cube octree (default: 6)" );
	param_t<int>          reinitIters( pl,    10, "reinitIters"	 	 , "Number of iterations for reinitialization (default: 10)" );
	param_t<u_short>    minSamRadiusH( pl,    16, "minSamRadiusH"	 , "Min sampling radius in h units on the uv plane.  Must be in the "
																	   "range of [16, 32] (default 16)" );
	param_t<double>                 r( pl,     3, "r"		     	 , "The ratio a/b in the range of [-10, -1) union [1, 10].  If it's "
																	   "negative, then b=|r|a; otherwise, a=rb (default: 3)" );
	param_t<bool>       perturbOrigin( pl,  true, "perturbFrame"	 , "Whether to perturb the surface's frame randomly in [-h/2,+h/2]^3 "
																	   "(default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,    11, "randomState"	 	 , "Seed for random perturbations of canonical frame (default: 11)" );
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
		if( cmd.parse( argc, argv, "Online evaluation of a trained error-correcting neural network on a hyp-paraboloidal surface" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		if( maxHK() < 1./15 || maxHK() > 2./3 )
			throw std::invalid_argument( "[CASL_ERROR] Desired max hk must be in the range of [1/15, 2/3]." );

		if( r() < -10 || (r() >= -1 && r() < 1) || r() > 10 )
			throw std::invalid_argument( "[CASL_ERROR] The ratio a/b must be in the range of [-10, -1) union [1, 10]." );

		if( minSamRadiusH() < 16 || minSamRadiusH() > 32 )
			throw std::invalid_argument( "[CASL_ERROR] Desired sampling radius in h units must be in the range of [16, 32]." );

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.

		//////////////////////////////////////////////// First, loading the neural networks ////////////////////////////////////////////////

		const std::string ROOT = nnetsDir() + "/3d/" + std::to_string( maxRL() );
		kml::NeuralNetwork nnetNS( ROOT + "/non-saddle", h, false );	// Load non-saddle neural network.
		kml::NeuralNetwork nnetSD( ROOT + "/saddle", h, false );		// Let's load the saddle neural network, too.

		///////////////////////////////////////////////////// Defining shape parameters ////////////////////////////////////////////////////

		std::mt19937 gen( randomState() );					// Engine used for random perturbations.
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );

		double MAX_K = maxHK() / h;
		double A, B, samRadius;
		HypParaboloid::findParamsAndSamRadius( r(), MAX_K, A, B, samRadius, h, minSamRadiusH() );
		HypParaboloid hypParaboloid( A, B );				// The surface in canonical space.

		double origin[P4EST_DIM] = {0, 0, 0};				// Local frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;
		test_utils::computeAffineTransformationParameters( mpi, origin, rotAxis, rotAngle, perturbOrigin(), randomRotation(), h, gen );

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began testing hybrid curvature on a hyperbolic paraboloid with a = %g, b = %g, max |hk| = %g, and "
								 "h = %g (level %i)\n", A, B, maxHK(), h, maxRL() );

		// Domain information.
		u_char octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		HypParaboloidLevelSet *pLS = setupDomain( mpi, hypParaboloid, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL, samRadius, n_xyz,
												  xyz_min, xyz_max );

		test_utils::evaluatePerformance( mpi, h, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max, reinitIters(), pLS, nonSaddleMinIH2KG(),
										 SQR( samRadius ), SQR( samRadius ), &nnetNS, &nnetSD, genNoise,
										 (randomNoise() > 0? &randomNoiseDist : nullptr ) );
		delete pLS;
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}
