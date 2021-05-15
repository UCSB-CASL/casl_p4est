/**
 * Testing frugally deep library v0.15.2-p0 (02/23/2021) for loading tensorflow+keras models into C++.
 * See https://github.com/Dobiasd/frugally-deep for details on this project.
 * The main dependencies to convert models are python 3.7, tensorflow 2.4.1, and C++14.
 * No Lambda layers supported, I must strip those layers from the curvature models.
 */
#define FDEEP_FLOAT_TYPE double
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
#include <fdeep/fdeep.hpp>
#include <chrono>
#include <random>

/**
 * Get the number of threads in a sequential region.
 * @return Number of threads.
 */
int omp_thread_count()
{
	int n = 0;
#pragma omp parallel reduction( + : n ) default( none )
	n += 1;
	return n;
}

int main()
{
	// Using OpenMP for fast inference.
	std::cout << "Using OpenMP with " << omp_thread_count() << " threads for fast inference." << std::endl;

	const auto model = fdeep::load_model( "/Users/youngmin/fdeep_model.json" );

	// Performance test: consider one thousand random samples.
	const int NUM_SAMPLES = 1000;
	const int SAMPLE_WIDTH = 9;
	std::vector<FDEEP_FLOAT_TYPE> samples[NUM_SAMPLES];
	std::mt19937 gen{}; 				// NOLINT.
	std::uniform_real_distribution<FDEEP_FLOAT_TYPE> dist;
	for( auto& sample : samples )				// Populate samples' matrix.
	{
		sample.resize( SAMPLE_WIDTH );
		for( auto& feature : sample )
			feature = dist( gen );
	}
	FDEEP_FLOAT_TYPE predictions[NUM_SAMPLES];

	// The neural network for curvature computation receives nine-element inputs and produces a single value.
	// Note we can't send a batch of inputs.  Inputs is a vector, but the elements depend on the model definition, which
	// can receive many input groups.
	// Timing prediction.
	auto start = std::chrono::high_resolution_clock::now();
	int i;
#pragma omp parallel for default( none ) schedule( static ) \
		shared( SAMPLE_WIDTH, NUM_SAMPLES, samples, predictions, model ) \
		private( i )
	for( i = 0; i < NUM_SAMPLES; i++ )
	{
		std::vector<fdeep::tensor> inputs = {
			fdeep::tensor( fdeep::tensor_shape( SAMPLE_WIDTH ), samples[i] )
		};

		predictions[i] = model.predict_single_output( inputs );
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) * 1e-6;
	// End of timing prediction.

	std::cout << std::setprecision( 8 );
	std::cout << "Timing: " << duration << " seconds" << std::endl;
}