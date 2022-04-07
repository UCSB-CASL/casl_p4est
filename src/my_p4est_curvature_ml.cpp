#include "my_p4est_curvature_ml.h"

#include <random>

//////////////////////////////////////////////// Scaler Abstract Class /////////////////////////////////////////////////

#ifdef P4_TO_P8
const int kml::Scaler::PHI_COLS[K_INPUT_PHI_SIZE] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,	// Phi column indices.
													  9, 10, 11, 12, 13, 14, 15, 16, 17, 	// Not defining the indices for
													 18, 19, 20, 21, 22, 23, 24, 25, 26};	// normals and ihk since they're
																							// not needed.
#else
const int    kml::Scaler::PHI_COLS[K_INPUT_PHI_SIZE   ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8};	// Phi column indices.
const int kml::Scaler::NORMAL_COLS[K_INPUT_NORMAL_SIZE] = { 9,10,11,12,13,14,15,16,17,	// Normal components: x first,
											  			   18,19,20,21,22,23,24,25,26};	// then y.
const int     kml::Scaler::HK_COLS[K_INPUT_HK_SIZE    ] = {27};							// Numerical hk column index.
#endif

void kml::Scaler::_printParams( const kml::Scaler::json& params, const std::string& paramsFileName )
{
	std::cout << "===----------------------------- Loaded parameters -----------------------------===" << std::endl;
	std::cout << "Source JSON file: '" << paramsFileName << "'" << std::endl;
	std::stringstream o;
	o << std::setw( 4 ) << params;
	std::cout << o.str() << std::endl;
	std::cout << "===-----------------------------------------------------------------------------===" << std::endl;
}


////////////////////////////////////////////////////// PCAScaler ///////////////////////////////////////////////////////

kml::PCAScaler::PCAScaler( const std::string& paramsFileName, const bool& printLoadedParams )
{
	// Load parameters.
	std::ifstream input( paramsFileName );
	json params;
	input >> params;

	// Assign parameters to internal variables.
	const std::string errorPrefix = "[CASL_ERROR] kml::PCAScaler: ";
	if( params.contains( "components" ) )			// Loading the components.
	{
		const auto& param = params["components"];
		_nComponents = param.size();
		for( const auto& component : param )
		{
			_components.emplace_back();
			for( const auto& val : component )
				_components.back().push_back( val.get<double>() );
		}
	}
	else
		throw std::runtime_error( errorPrefix + " components are missing!" );

	if( params.contains( "mean" ) )					// Loading the mean vector, which must have K_INPUT_SIZE elements.
	{
		const auto& means = params["mean"];
		if( means.size() == K_INPUT_SIZE )
		{
			for( const auto& mean : means )
				_means.push_back( mean.get<double>() );
		}
		else
			throw std::runtime_error( errorPrefix + " invalid number of mean values!" );
	}
	else
		throw std::runtime_error( errorPrefix + " mean values are missing!" );

	if( params.contains( "variance" ) )			// Loading the variance vector, which must have _nComponents elements.
	{
		const auto& variances = params["variance"];
		if( variances.size() == _nComponents )
		{
			for( const auto& variance : variances )
				_stds.push_back(  sqrt( variance.get<double>() ) );
		}
		else
			throw std::runtime_error( errorPrefix + " invalid number of variances!" );
	}

	if( params.contains( "whiten" ) )		// Using whitening?
		_whiten = params["whiten"].get<bool>();
	else
		_whiten = true;						// By default, use whitening.

	if( printLoadedParams )
		_printParams( params, paramsFileName );
}


void kml::PCAScaler::transform( double samples[][K_INPUT_SIZE], const int &nSamples ) const
{
	// To transform in Python: ((Y-pcaw.mean_)@pcaw.components_.T)/np.sqrt(pcaw.explained_variance_)
	for( int i = 0; i < nSamples; i++ )
	{
		// First, copy sample and subtract the mean.
		double sample[K_INPUT_SIZE];
		for( int j = 0; j < K_INPUT_SIZE; j++ )
			sample[j] = samples[i][j] - _means[j];

		// Second, project onto principal components.
		auto *projected = new double[_nComponents];
		for( int c = 0; c < _nComponents; c++ )
		{
			projected[c] = 0;
			for( int j = 0; j < K_INPUT_SIZE; j++ )
				projected[c] += sample[j] * _components[c][j];
		}

		// Third, divide by explained standard deviation while writing to output array if using whitening.
		for( int j = 0; j < K_INPUT_SIZE; j++ )
		{
			if( j < _nComponents )
				samples[i][j] = (_whiten? projected[j] / _stds[j] : projected[j]);
			else
				samples[i][j] = 0;			// Fill missing slots (i.e., belonging to no component) with zeros.
		}

		delete [] projected;
	}
}


unsigned long kml::PCAScaler::getNComponents() const
{
	return _nComponents;
}


//////////////////////////////////////////////////// StandardScaler ////////////////////////////////////////////////////

kml::StandardScaler::StandardScaler( const std::string& paramsFileName, const bool& printLoadedParams )
{
	// Load parameters.
	std::ifstream input( paramsFileName );
	json params;
	input >> params;

	const std::string errorPrefix = "[CASL_ERROR] kml::StandardScaler Constructor: ";

	// Assign parameters to internal variables.
	if( params.size() != 2 )
		throw std::runtime_error( errorPrefix + R"(expecting 2 groups of parameters: "mean" and "std"!)" );

	_loadParams( "mean", params, _mean );
	_loadParams( "std", params, _std );

	if( printLoadedParams )
		_printParams( params, paramsFileName );
}


void kml::StandardScaler::_loadParams( const std::string& inName, const json& params, double outFeature[K_INPUT_SIZE] )
{
	const std::string errorPrefix = "[CASL_ERROR] kml::StandardScaler::_loadParams: ";
	if( params.contains( inName ) )
	{
		const auto& param = params[inName];
		if( param.size() != K_INPUT_SIZE )
			throw std::runtime_error( errorPrefix + inName + " wrong number of values.  Expected " + std::to_string( K_INPUT_SIZE ) + "!" );

		for( int i = 0; i < K_INPUT_SIZE; i++ )
			outFeature[i] = param[i].get<double>();
	}
	else
		throw std::runtime_error( errorPrefix + inName + " parameter is missing!" );
}


void kml::StandardScaler::transform( double samples[][K_INPUT_SIZE], const int& nSamples ) const
{
	for( int i = 0; i < nSamples; i++ )
	{
		for( int j = 0; j < K_INPUT_SIZE; j++ )
			samples[i][j] = (samples[i][j] - _mean[j]) / _std[j];
	}
}


//////////////////////////////////////////////////// NeuralNetwork /////////////////////////////////////////////////////

kml::NeuralNetwork::NeuralNetwork( const std::string& folder, const double& h, const bool& verbose )
								   : _h( h ), _pcaScaler( folder + "/k_pca_scaler.json", verbose ),
								   _stdScaler( folder + "/k_std_scaler.json", verbose )
{
	const std::string errorPrefix = "[CASL_ERROR] kml::NeuralNetwork::Constructor: ";

	// Let's load the nnet params JSON file.
	std::ifstream in( folder + "/k_nnet.json" );
	json nnet;
	in >> nnet;

	if( _pcaScaler.getNComponents() != nnet["input_shape"][0].get<int>() )	// Verify that the number of true features corresponds to input size.
		throw std::runtime_error( errorPrefix + "Input shape mismatch!" );

	N_LAYERS = nnet["hidden_layers"].size() + 1;							// Count the output layer too.
	_inputSize = _pcaScaler.getNComponents();

	// Fill up the weight matrix sizes (accounting for bias).
	for( int i = 0; i < N_LAYERS - 1; i++ )
	{
		auto shape = nnet["hidden_layers"][i]["shape"].get<std::vector<int>>();
		_sizes.emplace_back( std::vector<int>{shape[0], shape[1]} );		// m-by-k matrix.
	}

	auto shape = nnet["output"]["shape"].get<std::vector<int>>();
	if( shape[0] != 1 )														// Expecting a single output neuron.
		throw std::runtime_error( errorPrefix + "Expected a single ouput neuron but more were detected!" );
	_sizes.emplace_back( std::vector<int>{shape[0], shape[1]} );			// 1-by-k matrix.

	// Allocating weights: hidden and ouput layers.
	for( int i = 0; i < N_LAYERS; i++ )
	{
		const int N_WEIGHTS = _sizes[i][0] * _sizes[i][1];
		W.emplace_back( N_WEIGHTS );
		auto layer = (i < N_LAYERS - 1? nnet["hidden_layers"][i] : nnet["output"] );
		auto weights = fdeep::internal::decode_floats( layer["weights"] );	// Taking advantage of function within frugally deep library.
		for( int j = 0; j < N_WEIGHTS; j++ )
			W[i][j] = weights[j];
	}	// When this loop ends, we have weights and bias all in row-major (hidden and output) weight matrices.

	// Enforce single-thread OpenBlas execution.
	goto_set_num_threads( 1 );
	openblas_set_num_threads( 1 );
}


void kml::NeuralNetwork::predict( double inputs[][K_INPUT_SIZE], double outputs[], const int& nSamples, const bool& hNormalize ) const
{
	// Normalize phi values by h (if needed) and leave mean hk as a negative value.
	for( int i = 0; i < nSamples; i++ )
	{
		if( hNormalize )
		{
			for( const int& j : Scaler::PHI_COLS )
				inputs[i][j] /= _h;
		}

		inputs[i][K_INPUT_SIZE - (P4EST_DIM - 1)] = -ABS( inputs[i][K_INPUT_SIZE - (P4EST_DIM - 1)] );
	}

	// Second part of inputs is the numerical dimensionless mean curvature.  Note we cast to float type (not double).
	auto *inputsPt2 = new FDEEP_FLOAT_TYPE[nSamples];
	for( int i = 0; i < nSamples; i++ )
		inputsPt2[i] = static_cast<FDEEP_FLOAT_TYPE>( inputs[i][K_INPUT_SIZE - (P4EST_DIM - 1)] );

	// First, preprocess inputs in part 1 (still doubles).
	// TODO: In 3d, the inputs will be in single precision but the scalers in double: must cast floats to doubles, transform, and cast back.
	_stdScaler.transform( inputs, nSamples );
	_pcaScaler.transform( inputs, nSamples );

	// Next, adding bias entry to transformed inputs1 and rearrange them so that each column is a sample (rather than a
	// row), and we agree to the expected float type (not double).
	auto *inputs1b = new FDEEP_FLOAT_TYPE[(_inputSize + 1) * nSamples];
	for( int j = 0; j < _inputSize; j++ )
	{
		for( int i = 0; i < nSamples; i++ )
			inputs1b[j * nSamples + i] = FDEEP_FLOAT_TYPE( inputs[i][j] );
	}
	for( int i = 0; i < nSamples; i++ )
		inputs1b[_inputSize * nSamples + i] = 1;		// The last row of nSamples 1's.

	// Allocate layer outputs.  Intermediate layers have an additional row of nSamples 1's for the bias.  The last
	// (output) doesn't have any row of 1's.
	std::vector<std::vector<FDEEP_FLOAT_TYPE>> O;
	for( int i = 0; i < N_LAYERS; i++ )
	{
		const int N_OUTPUTS = (_sizes[i][0] + (i == N_LAYERS - 1? 0 : 1)) * nSamples;
		O.emplace_back( N_OUTPUTS, 1 );		// Adding the one for the bias too.
	}

	// Inference: composite evaluation using OpenBlas single general matrix multiplication.
	for( int i = 0; i < N_LAYERS; i++ )
	{
		const FDEEP_FLOAT_TYPE *input = inputs1b;
		if( i > 0 )
			input = O[i - 1].data();

		// OpenBlas multiplication C = a*A*B + b*C
		//                               No transposing         m: out size   n: batch size  k: features   a
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, _sizes[i][0], nSamples,      _sizes[i][1], 1,
					 W[i].data(), _sizes[i][1], input, nSamples, 0, O[i].data(), nSamples );
		//           A mat        lda: k        B mat  ldb: n    b  C mat        ldc: n
		// The multiplication should leave the row of 1's in intermediate ouput matrices untouched.

		// Apply activation function to all hidden layers' outputs.
		if( i < N_LAYERS - 1 )
		{
			for( int j = 0; j < _sizes[i][0] * nSamples; j++ )    // Activation function doesn't touch the bias inputs.
				O[i][j] = _reLU( O[i][j] );
		}
	}

	// Add numerical hk to error-correcting output.
	for( int i = 0; i < nSamples; i++ )
		outputs[i] = O[N_LAYERS - 1][i] + inputsPt2[i];

	// Cleaning up.
	delete [] inputsPt2;
	delete [] inputs1b;
}


double kml::NeuralNetwork::getH() const
{
	return _h;
}


FDEEP_FLOAT_TYPE kml::NeuralNetwork::_reLU( const FDEEP_FLOAT_TYPE& x )
{
	return MAX( static_cast<FDEEP_FLOAT_TYPE>( 0.0 ), x );
}


///////////////////////////////////////////////////// Utilities //////////////////////////////////////////////////////

void kml::utils::generateColumnHeaders( std::string header[], const bool& includeTrueCurvatures )
{
	const int STEPS = 3;
	std::string states[] = {"m", "0", "p"};			// States for x, y, and z directions.
	int i = 0;
	for( int x = 0; x < STEPS; x++ )
		for( int y = 0; y < STEPS; y++ )
#ifdef P4_TO_P8
			for( int z = 0; z < STEPS; z++ )
#endif
		{
			i = SUMD( x * (int)pow( STEPS, P4EST_DIM - 1 ), y * (int)pow( STEPS, P4EST_DIM - 2 ), z );
			header[i] = SUMD( states[x], states[y], states[z] );

			header[i+1*num_neighbors_cube] = "nx_" + header[i];
			header[i+2*num_neighbors_cube] = "ny_" + header[i];
#ifdef P4_TO_P8
			header[i+3*num_neighbors_cube] = "nz_" + header[i];
#endif
		}
	i += P4EST_DIM * num_neighbors_cube;
	if( includeTrueCurvatures )						// Don't forget the true mean curvature column if requested!
		header[++i] = "hk";
	header[++i] = "ihk";
#ifdef P4_TO_P8
	if( includeTrueCurvatures )						// Don't forget the true Gaussian curvature column if requested!
		header[++i] = "h2kg";
	header[++i] = "ih2kg";
#endif
}


void kml::utils::rotateStencil90z( double stencil[], const int& dir )
{
	// Lambda function to perform vector rotation about z axis (this works for 2 and 3D).
	auto _rotate90 = (dir > 0)? []( double& x, double& y ){
		std::swap( x, (y *= -1) );			// Rotate by positive 90 degrees (counterclockwise).
	} : []( double& x, double& y ){
		std::swap( (x *= -1), y );			// Rotate by negative 90 degrees (clockwise).
	};

	// Lambda function to rotate features.
	auto _rotateFeatures = (dir > 0)? []( double f[] ){
#ifdef P4_TO_P8
		double c[] = {f[6], f[7], f[8], f[15], f[16], f[17], f[24], f[25], f[26],
					  f[3], f[4], f[5], f[12], f[13], f[14], f[21], f[22], f[23],
					  f[0], f[1], f[2],  f[9], f[10], f[11], f[18], f[19], f[20]};
#else
		double c[] = {f[2], f[5], f[8], f[1], f[4], f[7], f[0], f[3], f[6]};
#endif
		for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by +90 degrees (counterclokwise).
			f[i] = c[i];
	} : []( double f[] ){
#ifdef P4_TO_P8
		double c[] = {f[18], f[19], f[20],  f[9], f[10], f[11], f[0], f[1], f[2],
					  f[21], f[22], f[23], f[12], f[13], f[14], f[3], f[4], f[5],
					  f[24], f[25], f[26], f[15], f[16], f[17], f[6], f[7], f[8]};
#else
		double c[] = {f[6], f[3], f[0], f[7], f[4], f[1], f[8], f[5], f[2]};
#endif
		for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by -90 degrees (clockwise).
			f[i] = c[i];
	};

	// Rotate features.
	_rotateFeatures( &stencil[0] );						// Level-set values.
	_rotateFeatures( &stencil[num_neighbors_cube] );	// Normal x components.
	_rotateFeatures( &stencil[2*num_neighbors_cube] );	// Normal y components.
#ifdef P4_TO_P8
	_rotateFeatures( &stencil[3*num_neighbors_cube] );	// Normal z components.
#endif

	// Rotate actual vectors (z component is left unchanged in 3D).
	for( int i = num_neighbors_cube; i < 2 * num_neighbors_cube; i++ )
		_rotate90( stencil[i], stencil[i + num_neighbors_cube] );
}


#ifdef P4_TO_P8
void kml::utils::rotateStencil90y( double *stencil, const int& dir )
{
	// Lambda function to perform vector rotation about y axis (this works only on 3D).
	auto _rotate90 = (dir > 0)? []( double& x, double& z ){
		std::swap( (x *= -1), z );			// Rotate by positive 90 degrees (counterclockwise) w.r.t. +z axis.
	} : []( double& x, double& z ){
		std::swap( x, (z *= -1) );			// Rotate by negative 90 degrees (clockwise).
	};

	// Lambda function to rotate features.
	auto _rotateFeatures = (dir > 0)? []( double f[] ){
		double c[] = {f[18],  f[9], f[0], f[21], f[12], f[3], f[24], f[15], f[6],
					  f[19], f[10], f[1], f[22], f[13], f[4], f[25], f[16], f[7],
					  f[20], f[11], f[2], f[23], f[14], f[5], f[26], f[17], f[8]};
		for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by +90 degrees (counterclokwise).
			f[i] = c[i];
	} : []( double f[] ){
		double c[] = {f[2], f[11], f[20], f[5], f[14], f[23], f[8], f[17], f[26],
					  f[1], f[10], f[19], f[4], f[13], f[22], f[7], f[16], f[25],
					  f[0],  f[9], f[18], f[3], f[12], f[21], f[6], f[15], f[24]};
		for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by -90 degrees (clockwise).
			f[i] = c[i];
	};

	// Rotate features.
	_rotateFeatures( &stencil[0] );						// Level-set values.
	_rotateFeatures( &stencil[num_neighbors_cube] );	// Normal x components.
	_rotateFeatures( &stencil[2*num_neighbors_cube] );	// Normal y components.
	_rotateFeatures( &stencil[3*num_neighbors_cube] );	// Normal z components.

	// Rotate actual vectors (y component is left unchaged).
	for( int i = num_neighbors_cube; i < 2 * num_neighbors_cube; i++ )
		_rotate90( stencil[i], stencil[i + 2*num_neighbors_cube] );
}
#endif


void kml::utils::reflectStencil_yEqx( double stencil[] )
{
	// First the swap all features from one vertex to another.
	for( int i = 0; i < 1 + P4EST_DIM; i++ )
	{
		int offset = num_neighbors_cube * i;
#ifdef P4_TO_P8
		std::swap( stencil[ 3 + offset], stencil[ 9 + offset] );
		std::swap( stencil[ 4 + offset], stencil[10 + offset] );
		std::swap( stencil[ 5 + offset], stencil[11 + offset] );
		std::swap( stencil[ 6 + offset], stencil[18 + offset] );
		std::swap( stencil[ 7 + offset], stencil[19 + offset] );
		std::swap( stencil[ 8 + offset], stencil[20 + offset] );
		std::swap( stencil[15 + offset], stencil[21 + offset] );
		std::swap( stencil[16 + offset], stencil[22 + offset] );
		std::swap( stencil[17 + offset], stencil[23 + offset] );
#else
		std::swap( stencil[1 + offset], stencil[3 + offset] );
		std::swap( stencil[2 + offset], stencil[6 + offset] );
		std::swap( stencil[5 + offset], stencil[7 + offset] );
#endif
	}

	// Then, swap normal components: x<->y.
	for( int i = 0; i < num_neighbors_cube; i++ )
		std::swap( stencil[num_neighbors_cube + i], stencil[2*num_neighbors_cube + i] );
}


#ifdef P4_TO_P8
void kml::utils::rotateStencilToFirstOctant( double stencil[] )
#else
void kml::utils::rotateStencilToFirstQuadrant( double stencil[] )
#endif
{
	const int CENTER_IDX = floor(num_neighbors_cube / 2);	// Must be 4 in 2D and 13 in 3D.
	const double grad1[P4EST_DIM] = {DIM( stencil[num_neighbors_cube*1 + CENTER_IDX],
									 	  stencil[num_neighbors_cube*2 + CENTER_IDX],
										  stencil[num_neighbors_cube*3 + CENTER_IDX] )};
	double theta = atan2( grad1[dir::y], grad1[dir::x] );
	const double TWO_PI = 2. * M_PI;
	theta = (theta < 0)? TWO_PI + theta : theta;		// Make sure current angle lies in [0, 2pi].

	// Rotate only if theta not in [0, pi/2].
	if( theta > M_PI_2 )
	{
		if( theta <= M_PI )								// Quadrant/octant II?
		{
			rotateStencil90z( stencil, -1 );			// Rotate by -pi/2.
		}
		else if( theta < M_PI_2 * 3 )					// Quadrant/octant III?
		{
			rotateStencil90z( stencil );				// Rotate by pi.
			rotateStencil90z( stencil );
		}
		else											// Quadrant/octant IV?
		{
			rotateStencil90z( stencil );				// Rotate by pi/2.
		}
	}

#ifdef P4_TO_P8
	// Now rotate with respect to y, assuming that the previous step put the gradient's projection onto xy plane's first
	// quadrant.
	const double grad2[P4EST_DIM] = {stencil[num_neighbors_cube*1 + CENTER_IDX],
									 stencil[num_neighbors_cube*2 + CENTER_IDX],
									 stencil[num_neighbors_cube*3 + CENTER_IDX]};
	theta = atan2( grad2[dir::x], grad2[dir::z] );		// To rotate about y, the angle begins at +z axis.
	theta = (theta < 0)? TWO_PI + theta : theta;		// Make sure current angle lies in [0, 2pi].

	// Rotate only if theta not in [0, pi/2].
	if( theta > M_PI_2 )
	{
		if( theta <= M_PI )								// Octant V?
		{
			rotateStencil90y( stencil, -1 );			// Rotate by -pi/2.
		}
		else											// Because of the first transformation, we should not have an
		{												// angle w.r.t. +z greater than pi.
			throw std::runtime_error( "[CASL_ERROR] kml::utils::rotateStencilToFirstOctant: Wrong angle configuration!" );
		}
	}
#endif

	const double gradNew[P4EST_DIM] = {DIM( stencil[num_neighbors_cube*1 + CENTER_IDX],
									   		stencil[num_neighbors_cube*2 + CENTER_IDX],
									   		stencil[num_neighbors_cube*3 + CENTER_IDX] )};
	if( ORD( gradNew[0] < 0, gradNew[1] < 0, gradNew[2] < 0 ) )
		throw std::runtime_error( "[CASL_ERROR] kml::utils::rotateStencilToFirstOctant: One or more reoriented gradient"
								  " components is negative!" );
}


void kml::utils::normalizeToNegativeCurvature( std::vector<double>& stencil, const double& refHK, const bool& learning )
{
	if( refHK <= 0 )
		return;

	const int ELMTS_TO_CHANGE = K_INPUT_SIZE - (P4EST_DIM - 2) + learning;
	for( int i = 0; i < ELMTS_TO_CHANGE; i++ )	// Flip sign of level-set, gradient, and mean curvature data
		stencil[i] *= -1;						// (the last position(s) are left unchanged for Gaussian K).
}


void kml::utils::prepareSamplesFile( const mpi_environment_t& mpi, const std::string& directory,
									 const std::string& fileName, std::ofstream& file )
{
	std::string errorPrefix = "[CASL_ERROR] kml::utils::prepareSamplesFile: ";
	std::string fullFileName = directory + "/" + fileName;

	if( create_directory( directory, mpi.rank(), mpi.comm() ) )
		throw std::runtime_error( errorPrefix + "Couldn't create directory: " + directory );

	if( mpi.rank() == 0 )
	{
		const int NUM_COLUMNS = K_INPUT_SIZE_LEARN;					// We need to include the true curvatures too.
		std::string COLUMN_NAMES[NUM_COLUMNS];						// Headers follow the xy[z] truth table of 3-state
		kml::utils::generateColumnHeaders( COLUMN_NAMES, true );	// variables: phi + normal + hk + ihk + hkg + ihkg.
		file.open( fullFileName, std::ofstream::trunc );
		if( !file.is_open() )
			throw std::runtime_error( "[CASL_ERROR] kml::utils::prepareSamplesFile: Output file " + fullFileName + " couldn't be opened!" );

		std::ostringstream headerStream;					// Write column headers: enforcing strings by adding quotes.
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		file << headerStream.str() << std::endl;
		file.precision( 8 );								// Write data to preserve single precision.
	}

	CHKERRXX( PetscPrintf( mpi.comm(), "Rank %d successfully created samples file '%s'\n", mpi.rank(), fullFileName.c_str() ) );
	SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );				// Wait here until rank 0 is done.
}


int kml::utils::processSamplesAndAccumulate( const mpi_environment_t& mpi, std::vector<std::vector<double>>& samples,
											 std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer, const double& h,
											 const bool& negMeanKNormalize )
{
	// Let's reduce precision from 64b to 32b and normalize phi by h; Tensorflow trains on single precision anyways.
	// So, why bother to keep samples in double?
	const int RANK_TOTAL_SAMPLES = (int)samples.size() * 2;
	auto D = new FDEEP_FLOAT_TYPE [RANK_TOTAL_SAMPLES * K_INPUT_SIZE_LEARN];	// Let's put everying in a long array.

//#pragma omp parallel for default( none ) num_threads( 4 ) shared( samples, h, D ) schedule(static, 500)
	for( size_t i = 0; i < samples.size(); i++ )
	{
		// Use numerical mean ihk to determine sign: ihk comes after true hk.  Normalize only if requested.
		if( negMeanKNormalize )
			normalizeToNegativeCurvature( samples[i], samples[i][K_INPUT_SIZE - (P4EST_DIM - 2)], true );
#ifdef P4_TO_P8
		rotateStencilToFirstOctant( samples[i] );								// Reorientation.
#else
		rotateStencilToFirstQuadrant( samples[i] );								// Reorientation.
#endif

		// Normalize phi: Mainly to avoid losing precision when we move to floats.
		for( size_t j : Scaler::PHI_COLS )
			samples[i][j] /= h;

		// First copy: reoriented data packet.
		for( size_t j = 0; j < K_INPUT_SIZE_LEARN; j++ )
			D[i*2*(K_INPUT_SIZE_LEARN) + j] = (FDEEP_FLOAT_TYPE)samples[i][j];

		reflectStencil_yEqx( samples[i] );										// Reflect about plane y - x = 0.

		// Second copy: reflected data packet.
		for( size_t j = 0; j < K_INPUT_SIZE_LEARN; j++ )
			D[(i*2 + 1)*(K_INPUT_SIZE_LEARN) + j] = (FDEEP_FLOAT_TYPE)samples[i][j];
	}

	// First, rank 0 gathers the number of effective samples processed by each rank (including itself).
	int *totalValuesPerRank = nullptr;
	int *displacements = nullptr;			// Indicates where to place rank i data relative to beginning of recvbuf.
	float *allData = nullptr;				// Receive data from all processes here.
	if( mpi.rank() == 0 )
	{
		totalValuesPerRank = new int[mpi.size()];
		displacements = new int[mpi.size()];
	}

	SC_CHECK_MPI( MPI_Gather( &RANK_TOTAL_SAMPLES, 1, MPI_INT, totalValuesPerRank, 1, MPI_INT, 0, mpi.comm() ) );

	// Then, modify the totalValuesPerRank to contain the actual number of floats (not just the samples).
	int allRankTotalSamples = 0;
	if( mpi.rank() == 0 )
	{
		int beginning = 0;
#ifdef DEBUG
		CHKERRXX( PetscPrintf( mpi.comm(), "\nThe total number of values (samples x features) per rank is:\n" ) );
#endif
		for( int i = 0; i < mpi.size(); i++ )
		{
			displacements[i] = beginning;			// We need to know where each rank must start placing it data.
			totalValuesPerRank[i] *= K_INPUT_SIZE_LEARN;
			beginning += totalValuesPerRank[i];
#ifdef DEBUG
			CHKERRXX( PetscPrintf( mpi.comm(), "* Rank %d: %d samples (%d values), starting at index %d\n",
								   i, totalValuesPerRank[i] / (K_INPUT_SIZE_LEARN), totalValuesPerRank[i], displacements[i] ) );
#endif
		}

		allData = new FDEEP_FLOAT_TYPE[beginning];
		allRankTotalSamples = beginning / K_INPUT_SIZE_LEARN;
	}

	// Gather samples data.
	SC_CHECK_MPI( MPI_Gatherv( D, RANK_TOTAL_SAMPLES * K_INPUT_SIZE_LEARN, MPI_FLOAT, allData, totalValuesPerRank,
							   displacements, MPI_FLOAT, 0, mpi.comm() ) );

	// Accumulate samples.  Only rank 0 does this.
	if( mpi.rank() == 0 )
	{
		for( int i = 0; i < allRankTotalSamples; i++ )
		{
			buffer.emplace_back( K_INPUT_SIZE_LEARN );
			for( int j = 0; j < K_INPUT_SIZE_LEARN; j++ )
				buffer.back()[j] = allData[i*K_INPUT_SIZE_LEARN + j];
		}
	}

	// Cleaning up.
	delete [] allData;
	delete [] displacements;
	delete [] totalValuesPerRank;
	delete [] D;

	// Comunicate to everyone the total number of samples across processes.
	SC_CHECK_MPI( MPI_Bcast( &allRankTotalSamples, 1, MPI_INT, 0, mpi.comm() ) );	// Acts as an MPI_Barrier, too.
	return allRankTotalSamples;
}


int kml::utils::saveSamplesBufferToFile( const mpi_environment_t& mpi, std::ofstream& file,
										 const std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer,
										 const size_t& nSamplesToSave )
{
	int savedSamples;
	if( nSamplesToSave < 0 || nSamplesToSave > buffer.size() )
		throw std::invalid_argument( "[CASL_ERROR] kml::utils::saveSamplesBufferToFile: nSamplesToSave must be "
									 "non-negative and at most buffer current size!" );
	if( mpi.rank() == 0 )
	{
		int i;
		size_t numSamplesToSave = nSamplesToSave == 0? buffer.size() : nSamplesToSave;
		for( i = 0; i < numSamplesToSave; i++ )
		{
			int j;
			for( j = 0; j < K_INPUT_SIZE_LEARN - 1; j++ )
				file << buffer[i][j] << ",";		// Inner elements.
			file << buffer[i][j] << std::endl;		// Last element is ihk in 2D or ih2kg in 3D.
		}
		savedSamples = i;
	}

	// Comunicate to everyone the total number of saved samples.
	SC_CHECK_MPI( MPI_Bcast( &savedSamples, 1, MPI_INT, 0, mpi.comm() ) );	// Acts as an MPI_Barrier, too.
	return savedSamples;
}


int kml::utils::processSamplesAndSaveToFile( const mpi_environment_t& mpi, std::vector<std::vector<double>>& samples,
											 std::ofstream& file, const double& h, const bool& negMeanKNormalize,
											 const int& preAllocateSize )
{
	std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;
	if( mpi.rank() == 0 )
		buffer.reserve( preAllocateSize );
	processSamplesAndAccumulate( mpi, samples, buffer, h, negMeanKNormalize );
	return saveSamplesBufferToFile( mpi, file, buffer );
}


int kml::utils::histSubSamplingAndSaveToFile( const mpi_environment_t& mpi,
											  const std::vector<std::vector<FDEEP_FLOAT_TYPE>>& buffer,
											  std::ofstream& file, FDEEP_FLOAT_TYPE minHK, FDEEP_FLOAT_TYPE maxHK,
											  const unsigned short& nbins, const FDEEP_FLOAT_TYPE& frac,
											  const FDEEP_FLOAT_TYPE& minFold)
{
	const std::string errorPrefix = "[CASL_ERROR] kml::utils::histSubSamplingAndSaveToFile: ";

	if( minHK >= maxHK )
		throw std::invalid_argument( errorPrefix + "minHK must be smaller than maxHK!" );

	if( minFold < 1 )
		throw std::invalid_argument( errorPrefix + "minFold must be at least 1!" );

	if( frac <= 0 )
		throw std::invalid_argument( errorPrefix + "frac must be strictly positive!" );

	if( nbins < 10 )
		throw std::invalid_argument( errorPrefix + "nbins must be at least 10!" );

	int savedSamples = 0;
	if( mpi.rank() == 0 )
	{
		if( !buffer.empty() )
		{
			// Begin by finding the range of mean |hk*| if not given.
			if( isnan( minHK ) || isnan( maxHK ) )
			{
				minHK = FLT_MAX;
				maxHK = 0;
				for( const auto& sample : buffer )
				{
					minHK = MIN( minHK, ABS( sample[K_INPUT_SIZE - (P4EST_DIM - 1)] ) );
					maxHK = MAX( maxHK, ABS( sample[K_INPUT_SIZE - (P4EST_DIM - 1)] ) );
				}
			}

			minHK -= FLT_EPSILON;		// Some padding to the histogran end points.
			maxHK += FLT_EPSILON;

			// Build the histogram.
			std::vector<FDEEP_FLOAT_TYPE> limits;										// Bin i specifies the semi-open interval [limits[i], limits[i+1]),
			const FDEEP_FLOAT_TYPE dx = linspace( minHK, maxHK, nbins + 1, limits );	// except the end points, which extend numerically to -inf and +inf.

			std::vector<std::vector<int>> bins( nbins, std::vector<int>() );	// Bins act like buckets holding sample indices.
			std::vector<int>counts( nbins, 0 );									// Keeps track of how many samples lie in each bin.
			for( int i = 0; i < buffer.size(); i++ )
			{
				FDEEP_FLOAT_TYPE hk = ABS( buffer[i][K_INPUT_SIZE - (P4EST_DIM - 1)] );	// Mean |hk*|
				int idx = (int)MAX( 0.0f, MIN( floor( (hk - minHK) / dx ), (FDEEP_FLOAT_TYPE)nbins - 1.0f ) );
				if( !(hk >= limits[idx] && hk < limits[idx+1]) )				// Check that |hk| does fall in the range.
				{
					if( idx > 0 && (hk >= limits[idx-1] && hk < limits[idx]) )					// Does it fit to the left?
						idx--;
					else if( idx < nbins - 1 && (hk >= limits[idx+1] && hk < limits[idx+2]) )	// Does it fit to the right?
						idx++;
					else
						throw std::runtime_error( errorPrefix + "Wrong histogram configuration!" );
				}

				if( bins[idx].empty() )														// First sample arriving to idx bin?
					bins[idx].reserve( (size_t)(1.25 * (double)buffer.size() / nbins) );	// Pre-allocate size.

				bins[idx].push_back( i );
				counts[idx]++;
			}

			// Find the median.
			std::sort( counts.begin(), counts.end() );
			int mIdx = floor( nbins / 2.0 );
			auto median = (nbins % 2 == 0)? (FDEEP_FLOAT_TYPE)ceil( (counts[mIdx] + counts[mIdx - 1]) / 2.0 ) : (FDEEP_FLOAT_TYPE)counts[mIdx];

			// Find the smallest non-zero bin count.
			int minCount = 0;
			int b = 0;
			while( b < nbins && (minCount = counts[b]) == 0 )
				b++;

			if( minCount == 0 )
				throw std::runtime_error( errorPrefix + "Min count is zero?!" );

			// Probabilistic subsampling by shuffling the indices within overpopulated bins.  Then, writing samples to file.
			int cap = (int)MAX( ceil( frac * median ), minFold * (FDEEP_FLOAT_TYPE)minCount );

			CHKERRXX( PetscPrintf( mpi.comm(), "[-] Nonzero min count = %d, median = %.2f, cap = %d\n", minCount, median, cap ) );

			for( b = 0; b < nbins; b++ )
			{
				if( bins[b].size() > cap )
					std::shuffle( bins[b].begin(), bins[b].end(), std::mt19937( b ) );

				for( int i = 0; i < MIN( (int)bins[b].size(), cap ); i++ )
				{
					int j;
					int idx = bins[b][i];
					for( j = 0; j < K_INPUT_SIZE_LEARN - 1; j++ )
						file << buffer[idx][j] << ",";		// Inner elements.
					file << buffer[idx][j] << std::endl;	// Last element is ihk in 2D or ih2kg in 3D.
					savedSamples++;
				}
			}
		}
	}

	// Communicate to everyone the total number of saved samples.
	SC_CHECK_MPI( MPI_Bcast( &savedSamples, 1, MPI_INT, 0, mpi.comm() ) );	// Acts as an MPI_Barrier, too.
	return savedSamples;
}


double kml::utils::easingOffProbability( double x, const double& lowVal, const double& lowProb, const double& upVal,
										 const double& upProb )
{
	// Some checks.
	if( lowVal >= upVal )
		throw std::invalid_argument( "kml::utils::easingOffProbability: lowVal must be strictly less than upVal!" );

	if( lowProb < 0 || lowProb > 1 || upProb < 0 || upProb > 1 || lowProb >= upProb )
		throw std::invalid_argument( "kml::utils::easingOffProbability: lowProb must be strictly less than upProb and "
									 "both must be in the range of [0,1]!" );

	if( x <= lowVal )		// Lower-bound edge case.
		return lowProb;

	if( x >= upVal )		// Upper-bound edge case.
		return upProb;

	// Compute the probability with help of the sinusoidal function.
	x = (x - lowVal) / (upVal - lowVal);		// Normalize between 0 and 1.
	return lowProb + (sin( -M_PI_2 + x * M_PI ) + 1) * (upProb - lowProb) / 2;
}


////////////////////////////////////////////////////// Curvature ///////////////////////////////////////////////////////

kml::Curvature::Curvature( const NeuralNetwork *nnet, const double& h, const double& loMinHK, const double& upMinHK )
						   : _h( h ), LO_MIN_HK( loMinHK ), UP_MIN_HK( upMinHK ), _nnet( nnet )
{
	if( ABS( _h - _nnet->getH() ) > PETSC_MACHINE_EPSILON )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::Constructor: Neural network and current spacing are incompatible!" );

	if( loMinHK >= upMinHK || loMinHK < 0 )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::Constructor: minHK lower- and upper-bound must be positive!" );
}


void kml::Curvature::_collectSamples( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec numMeanK,
									  std::vector<std::vector<double>>& samples, std::vector<p4est_locidx_t>& indices ) const
{
	// Data accessors.
	const p4est_nodes_t *nodes = ngbd.get_nodes();
	const p4est_t *p4est = ngbd.get_p4est();
	const auto *splittingCriteria = (splitting_criteria_t*) p4est->user_pointer;
	const int maxRL = splittingCriteria->max_lvl;

	const double *phiReadPtr;
	CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

	const double *normalReadPtr[P4EST_DIM];
	for( int dim = 0; dim < P4EST_DIM; dim++ )
		CHKERRXX( VecGetArrayRead( normal[dim], &normalReadPtr[dim] ) );

	// We'll interpolate numerical hk at the interface linearly: prepare structure.
	my_p4est_interpolation_nodes_t interp( &ngbd );
	interp.set_input( numMeanK, interpolation_method::linear );

	// Collect samples: one per valid (i.e., with full h-uniform stencil) locally owned node next to Gamma.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, &ngbd, (char)maxRL );
	std::vector<p4est_locidx_t> ngIndices;
	nodesAlongInterface.getIndices( &phi, ngIndices );

	samples.clear();
	indices.clear();
	samples.reserve( nodes->num_owned_indeps );
	indices.reserve( nodes->num_owned_indeps );
	double xyz[P4EST_DIM];
	const double ONE_OVER_H = 1. / _h;

	for( auto n : ngIndices )
	{
		std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D, 27 values in 3D.
		std::vector<double> sample;
		sample.reserve( K_INPUT_SIZE );
		try
		{
			if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
			{
				for( auto s : stencil )								// First, add the the phi-values.
					sample.push_back( phiReadPtr[s] * ONE_OVER_H );	// H-normalized phi values.

				for( auto& dim : normalReadPtr)						// Now, the normal unit vectors' components.
				{
					for( auto s: stencil )							// First u, then v [, then w].
						sample.push_back( dim[s] );
				}

				node_xyz_fr_n( n, p4est, nodes, xyz );				// Finally, the interpolated numerical hk on Gamma.
				for( int dim = 0; dim < P4EST_DIM; dim++ )
					xyz[dim] -= normalReadPtr[dim][n] * phiReadPtr[n];
				sample.push_back( interp( xyz ) * _h );

				samples.push_back( sample );
				indices.push_back( n );		// Keep track of which locally owned nodes we are looking at.
			}
		}
		catch( std::exception &e ) {}
	}

	// Cleaning up.
	for( int dim = 0; dim < P4EST_DIM; dim++ )
		CHKERRXX( VecRestoreArrayRead( normal[dim], &normalReadPtr[dim] )  );
	CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
}


void kml::Curvature::_computeHybridHK( const std::vector<std::vector<double>>& samples, std::vector<double>& hybMeanHK ) const
{
	hybMeanHK.clear();
	hybMeanHK.resize( samples.size() );

	// Since we receive samples with phi h-normalization, let's put them into negative-mean-curvature form and reorient them.
	int outIdx = 0;			// To avoid unnecessary neural evaluations, skip samples for which mean |ihk| < LO_MIN_HK.
	std::vector<int> outIdxToSampleIdx;
	outIdxToSampleIdx.reserve( samples.size() );
	std::vector<std::vector<double>> negSamples;
	negSamples.reserve( samples.size() );
	for( int i = 0; i < samples.size(); i++ )
	{
		if( ABS( samples[i][K_INPUT_SIZE - (P4EST_DIM - 1)] ) < LO_MIN_HK )		// Skip well resolved stencils; use the numerical mean kappa estimation.
		{
			hybMeanHK[i] = samples[i][K_INPUT_SIZE - (P4EST_DIM - 1)];
			continue;
		}

		negSamples.emplace_back( std::vector<double>( samples[i] ) );
		utils::normalizeToNegativeCurvature( negSamples.back(), negSamples.back()[K_INPUT_SIZE - (P4EST_DIM - 1)] );

#ifdef P4_TO_P8
		utils::rotateStencilToFirstOctant( negSamples.back() );
#else
		utils::rotateStencilToFirstQuadrant( negSamples.back() );
#endif
		outIdxToSampleIdx.push_back( i );
		outIdx++;
	}

	// Build inputs array to evaluate neural network.
	const int N_INPUTS_PER_SAMPLE = 2;
	const int N_INPUTS = outIdx * N_INPUTS_PER_SAMPLE;
	auto inputs = new double[N_INPUTS][K_INPUT_SIZE];
	auto *outputs = new double[N_INPUTS];
	for( int i = 0; i < outIdx; i++ )
	{
		int idx = i * N_INPUTS_PER_SAMPLE;
		for( int j = 0; j < K_INPUT_SIZE; j++ )			// We'll give it two takes: original.
			inputs[idx + 0][j] = negSamples[i][j];

		utils::reflectStencil_yEqx( negSamples[i] );	// And reflected about x - y = 0 plane.
		for( int j = 0; j < K_INPUT_SIZE; j++ )
			inputs[idx + 1][j] = negSamples[i][j];
	}

	// Execute inference on batch: ihk in original samples preserves its sign (to be used below).
	_nnet->predict( inputs, outputs, N_INPUTS, false );

	// Collect outputs.
	for( int i = 0; i < outIdx; i++ )
	{
		int idx = i * N_INPUTS_PER_SAMPLE;
		double hk = (outputs[idx + 0] + outputs[idx + 1]) / 2.0;	// Average predictions produces a better one.

		// Blend with numerical mean ihk within the range of [0.004, 0.007] (using ihk as the indicator).
		double ihk = samples[outIdxToSampleIdx[i]][K_INPUT_SIZE - (P4EST_DIM - 1)];
		if( ABS( ihk ) <= UP_MIN_HK )
		{
			double lam = (UP_MIN_HK - ABS( ihk )) / (UP_MIN_HK - LO_MIN_HK);
			hk = (1 - lam) * hk + lam * -ABS( ihk );
		}

		// Fix sign according to (untouched) mean curvature.
		hybMeanHK[outIdxToSampleIdx[i]] = SIGN( ihk ) * ABS( hk );
	}

	// Clean up.
	delete [] inputs;
	delete [] outputs;
}


std::pair<double, double> kml::Curvature::compute( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM],
												   Vec numMeanK, Vec hybMeanK, Vec hybFlag, const bool& dimensionless,
												   parStopWatch *watch ) const
{
	if( !phi || !normal || !numMeanK || !hybMeanK || !hybFlag )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::compute: One of the provided vectors is null!" );

	// Data accessors.
	const p4est_nodes_t *nodes = ngbd.get_nodes();

	// We start by computing the unit normals and numerical mean curvature (as a byproduct of calling this function).
	// TODO: Need to retrain 2d k_ecnets using compute_mean_curvature( ngbd, normal, numCurvature ) for compatibility with 3D.
	double startTime = watch? watch->get_duration_current() : 0;
	compute_normals( ngbd, phi, normal );
	compute_mean_curvature( ngbd, phi, normal, numMeanK );
	double totalNumericalTime = watch? watch->get_duration_current() - startTime : -1;

	// Collect samples.
	std::vector<std::vector<double>> samples;
	std::vector<p4est_locidx_t> indices;
	_collectSamples( ngbd, phi, normal, numMeanK, samples, indices );

	// Compute hybrid (dimensionless) mean curvature.
	std::vector<double> hybMeanHK;
	_computeHybridHK( samples, hybMeanHK );

	// Copy solution to parallel vectors.
	double *hybMeanKPtr, *hybFlagPtr;
	CHKERRXX( VecGetArray( hybFlag, &hybFlagPtr ) );
	CHKERRXX( VecGetArray( hybMeanK, &hybMeanKPtr ) );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )		// Initialization.
		hybMeanKPtr[n] = hybFlagPtr[n] = 0;

	for( int i = 0; i < indices.size(); i++ )	// Go through the nodes where we computed the hybrid solution.
	{
		p4est_locidx_t idx = indices[i];
		hybMeanKPtr[idx] = dimensionless? hybMeanHK[i] : hybMeanHK[i] / _h;
		hybFlagPtr[idx] = 1;					// Signal that node idx contains mean kappa "at" the interface.
	}

	// Cleaning up.
	CHKERRXX( VecRestoreArray( hybMeanK, &hybMeanKPtr ) );
	CHKERRXX( VecRestoreArray( hybFlag, &hybFlagPtr ) );

	// Let's synchronize the machine learning flag vector among all processes.
	CHKERRXX( VecGhostUpdateBegin( hybFlag, INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( hybFlag, INSERT_VALUES, SCATTER_FORWARD ) );

	// Let's synchronize the curvature values among all processes.
	CHKERRXX( VecGhostUpdateBegin( hybMeanK, INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( hybMeanK, INSERT_VALUES, SCATTER_FORWARD ) );

	double totalHybridTime = watch? watch->get_duration_current() - startTime : -1;
	return std::make_pair( totalNumericalTime, totalHybridTime );
}
