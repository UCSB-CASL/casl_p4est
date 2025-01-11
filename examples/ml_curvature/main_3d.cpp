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
 * Created: May 31, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <random>
#include "data_sets/gaussian/gaussian_3d.h"
#include "online_tests/test_utils_3d.h"
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>             maxHK( pl,   0.6, "maxHK"			 , "Desired max (absolute) dimensionless mean curvature at the peak. "
																	   "Must be in the open interval of (1/3, 2/3) (default: 0.6)" );
	param_t<u_short>            maxRL( pl,     6, "maxRL"			 , "Maximum level of refinement per unit-cube octree (default: 6)" );
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

		double origin[P4EST_DIM] = {0, 0, 0};				// Local frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;
		test_utils::computeAffineTransformationParameters( mpi, origin, rotAxis, rotAngle, perturbOrigin(), randomRotation(), h, gen );

		///////////////////////////////////////////////////////////// Testing //////////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began testing hybrid curvature on a Gaussian surface online with a = %g, su^2 = %g, sv^2 = %g, max "
								 "|hk| = %g, and h = %g (level %i)\n", a(), SU2, SV2, maxHK(), h, maxRL() );

		// Domain information.
		u_short octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		GaussianLevelSet *gLS = setupDomain( mpi, gaussian, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max );

		test_utils::evaluatePerformance( mpi, h, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max, reinitIters(), gLS, nonSaddleMinIH2KG(),
										 NAN, NAN, &nnetNS, &nnetSD, genNoise, (randomNoise() > 0? &randomNoiseDist : nullptr ) );
		delete gLS;
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}
