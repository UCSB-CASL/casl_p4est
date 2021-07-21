/**
 * OpenBlas benchmark: deciding whether this optimized library is faster than inference using frugally-deep.
 *
 * @cite https://github.com/xianyi/OpenBLAS
 * @cite https://software.intel.com/content/www/us/en/develop/documentation/onemkl-tutorial-c/top/multiplying-matrices-using-dgemm.html
 *
 * Author: Luis Ángel (임 영민)
 * Created: July 20, 2021.
 * Updated: July 21, 2021.
 */

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p4est_utils.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian_ml.h>
#include <src/petsc_compatibility.h>

#include <cblas.h>

//////////////////////////////////////////////////// Main function /////////////////////////////////////////////////////

/**
 * Main function.
 * @param argc Number of input arguments.
 * @param argv Actual arguments.
 * @return 0 if process finished successfully, nonzero otherwise.
 */
int main( int argc, char** argv )
{
	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// OpenMP verification.
		int nThreads = 0;
#pragma omp parallel reduction( + : nThreads ) default( none )
		nThreads += 1;

		std::cout << "Rank " << mpi.rank() << " can spawn " << nThreads << " thread(s)\n\n";

		// Preparing input.
		srand( 37 );					// NOLINT.
		const int N_SAMPLES = 100;
		const int N_FEATURES = 17;
		double inputs1[N_SAMPLES][N_FEATURES];
		double inputs2[N_SAMPLES];		// Second part of inputs is a copy of last column in inputs1.
		float outputs[N_SAMPLES];

		for( int i = 0; i < N_SAMPLES; i++ )	// Random values in inputs1 with N_SAMPLES x N_FEATURES elements.
		{
			for( auto& feature : inputs1[i] )
				feature = rand() / double( RAND_MAX );	// NOLINT.
			inputs2[i] = inputs1[i][N_FEATURES-1];
		}

		parStopWatch watch;
		watch.start();				// Start up timer.

		/////////////////////////////////////// Performance using neural network ///////////////////////////////////////

		// Load neural network that uses frugally-deep library.
		const fdeep::model model = fdeep::load_model( "/Users/youngmin/nnets/fdeep_mass_nnet.json", true, fdeep::dev_null_logger );

		std::cout << ">> Began testing neural network..." << std::endl;

		// Proceed with predictions, one input at a time.
		double cumulativeTime = 0;
		for( int iter = 0; iter < 11; iter++ )
		{
			double start = watch.get_duration_current();
			for( int i = 0; i < N_SAMPLES; i++ )
			{
				// Build two-part input.
				std::vector<FDEEP_FLOAT_TYPE> input1( &inputs1[i][0], &inputs1[i][0] + N_FEATURES );
				std::vector<FDEEP_FLOAT_TYPE> input2( &inputs2[i], &inputs2[i] + 1 );

				// Add split inputs as independent entries to a two-element tensor.
				std::vector<fdeep::tensor> inputTensors = {
					fdeep::tensor( fdeep::tensor_shape( N_FEATURES ), input1 ),
					fdeep::tensor( fdeep::tensor_shape( 1 ), input2 )
				};

				// Predict and undo normalization, too.
				outputs[i] = model.predict_single_output( inputTensors );
			}
			if( iter != 0 )
			{
				double end = watch.get_duration_current();
				cumulativeTime += end - start;
				std::cout << end - start << std::endl;
			}
		}

		std::cout << "<< Average timing " << cumulativeTime / 10. << " secs." << std::endl;

		////////////////////////////////////////// Performance using OpenBlas //////////////////////////////////////////

		// Adding bias entry to inputs1 and rearrange so that each column is a sample (rather than a row).
		float inputs1b[(N_FEATURES + 1) * N_SAMPLES];
		for( int j = 0; j < N_FEATURES; j++ )
		{
			for( int i = 0; i < N_SAMPLES; i++ )
				inputs1b[j * N_SAMPLES + i] = float( inputs1[i][j] );
		}
		for( int i = 0; i < N_SAMPLES; i++ )
			inputs1b[N_FEATURES * N_SAMPLES + i] = 1;

		// Loading weights: just random values.
		const int LAYER_SIZE = 130;							// Hidden plus output layers.
		std::vector<std::vector<float>> W;
		const int N_LAYERS = 5;
		int sizes[][3] = {
			{LAYER_SIZE, N_FEATURES + 1, N_SAMPLES},		// W0 x I0.
			{LAYER_SIZE, LAYER_SIZE + 1, N_SAMPLES},		// W1 x f(W0 x I0).
			{LAYER_SIZE, LAYER_SIZE + 1, N_SAMPLES},		// W2 x f(W1 x f(W0 x I0)).
			{LAYER_SIZE, LAYER_SIZE + 1, N_SAMPLES},		// W3 x f(W2 x f(W1 x f(W0 x I0))).
			{         1, LAYER_SIZE + 1, N_SAMPLES},		// W4 x f(W3 x f(W2 x f(W1 x f(W0 x I0)))).
		};

		for( int i = 0; i < N_LAYERS; i++ )
		{
			const int N_WEIGHTS = sizes[i][0] * sizes[i][1];
			W.emplace_back( N_WEIGHTS );
			for( auto& w : W[i] )
				w = rand() / float( RAND_MAX ) * (rand() / float( RAND_MAX ) > 0.5? -1 : 1);	// NOLINT.
		}							// When this loop ends, we have weights and bias all in row-majored weight matrices.

		// Allocating outputs.
		std::vector<std::vector<float>> O;
		for( int i = 0; i < N_LAYERS; i++ )
		{
			const int N_OUTPUTS = (sizes[i][0] + (i == N_LAYERS - 1? 0 : 1)) * sizes[i][2];
			O.emplace_back( N_OUTPUTS, 1 );		// Adding the one for the bias too.
		}

		goto_set_num_threads( 1 );		// Single-thread execution.
		openblas_set_num_threads( 1 );

		std::cout << ">> Began testing OpenBlas-based inference..." << std::endl;

		cumulativeTime = 0;
		for( int iter = 0; iter < 11; iter++ )
		{
			double start = watch.get_duration_current();

			// Inference.
			for( int i = 0; i < N_LAYERS; i++ )
			{
				const float *input = inputs1b;
				if( i > 0 )
					input = O[i - 1].data();
				cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, sizes[i][0], sizes[i][2], sizes[i][1], 1,
							 W[i].data(), sizes[i][1], input, sizes[i][2], 0, O[i].data(), sizes[i][2] );

				// Apply ReLU activation function to all hidden layers.
				if( i < N_LAYERS - 1 )
				{
					for( int j = 0; j < sizes[i][0] * sizes[i][2]; j++ )    // Activation function doesn't affect the bias.
						O[i][j] = MAX( 0.0f, O[i][j] );
				}
			}

			// Add inputs2 to error-correcting output.
			for( int i = 0; i < N_SAMPLES; i++ )
				outputs[i] = O[N_LAYERS - 1][i] + float( inputs2[i] );

			if( iter != 0 )
			{
				double end = watch.get_duration_current();
				cumulativeTime += end - start;
				std::cout << end - start << std::endl;
			}
		}

		std::cout << "<< Average timing " << cumulativeTime / 10. << " secs." << std::endl;

		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}

	return 0;
}