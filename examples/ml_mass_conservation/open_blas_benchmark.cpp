/**
 * OpenBlas benchmark: deciding whether this optimized library is faster than inference using frugally-deep.
 *
 * @cite https://github.com/xianyi/OpenBLAS
 * @cite https://software.intel.com/content/www/us/en/develop/documentation/onemkl-tutorial-c/top/multiplying-matrices-using-dgemm.html
 *
 * Author: Luis Ángel (임 영민)
 * Created: July 20, 2021.
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
		double outputs[N_SAMPLES];

		for( int i = 0; i < N_SAMPLES; i++ )	// Random values in inputs1 with N_SAMPLES x N_FEATURES elements.
		{
			for( auto& feature : inputs1[i] )
				feature = rand() / double( RAND_MAX );	// NOLINT.
			inputs2[i] = inputs1[i][N_FEATURES-1];
		}

		/////////////////////////////////////// Performance using neural network ///////////////////////////////////////

		// Load neural network that uses frugally-deep library.
		const fdeep::model model = fdeep::load_model( "/Users/youngmin/nnets/fdeep_mass_nnet.json", true, fdeep::dev_null_logger );

		parStopWatch watch;				// Start evaluation after instantiating the nnet.
		watch.start();

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

		// Adding bias to inputs1.
		double inputs1b[N_SAMPLES][N_FEATURES + 1];
		for( int i = 0; i < N_SAMPLES; i++ )
		{
			int j;
			for( j = 0; j < N_FEATURES; j++ )
				inputs1b[i][j] = inputs1[i][j];
			inputs1b[i][j] = 1.;
		}

		// Loading weights: just random values.
		std::vector<std::vector<float>> W;
		int sizes[][3] = {
			{130,  18, 100},		// W0 x I0.
			{130, 131, 100},		// W1 x f(W0 x I0).
			{130, 131, 100},		// W2 x f(W1 x f(W0 x I0)).
			{130, 131, 100},		// W3 x f(W2 x f(W1 x f(W0 x I0))).
			{  1, 131, 100},		// W4 x f(W3 x f(W2 x f(W1 x f(W0 x I0)))).
		};

		goto_set_num_threads( 1 );		// Single-thread execution.
		openblas_set_num_threads( 1 );

		int m = 4, k = 3, n = 2;
		float A[] = {
			 1,  2,  3,
			 4,  5,  6,
			 7,  8,  9,
			10, 11, 12
		};
		float B[] = {
			3, 4, 2,
			1, 5, 7
		};
		float C[m * n];
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n );
		for( int i = 0; i < m * n; i++ )
			std::cout << C[i] << ((i + 1) % n == 0? "\n" : "\t");



		watch.stop();
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}

	return 0;
}