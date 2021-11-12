#include "my_p4est_curvature_ml.h"

//////////////////////////////////////////////// Scaler Abstract Class /////////////////////////////////////////////////

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
								   : H( h ), _pcaScaler( folder + "/k_pca_scaler.json", verbose ),
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
	// Normalize phi values by h (if needed) and leave hk as a negative value.
	for( int i = 0; i < nSamples; i++ )
	{
		if( hNormalize )
		{
			for( const int& j : _pcaScaler.PHI_COLS )
				inputs[i][j] /= H;
		}

		inputs[i][K_INPUT_SIZE-1] = -ABS( inputs[i][K_INPUT_SIZE-1] );
	}

	// Second part of inputs is the numerical dimensionless curvature.  Note that we cast to float type (not double).
	auto *inputsPt2 = new FDEEP_FLOAT_TYPE[nSamples];
	for( int i = 0; i < nSamples; i++ )
		inputsPt2[i] = static_cast<FDEEP_FLOAT_TYPE>( inputs[i][K_INPUT_SIZE-1] );

	// First, preprocess inputs in part 1 (still doubles).
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
	return H;
}


FDEEP_FLOAT_TYPE kml::NeuralNetwork::_reLU( const FDEEP_FLOAT_TYPE& x )
{
	return MAX( static_cast<FDEEP_FLOAT_TYPE>( 0.0 ), x );
}


///////////////////////////////////////////////////// Utilities //////////////////////////////////////////////////////

void kml::utils::generateColumnHeaders( std::string header[], const bool& includeTargetHK )
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
		}
	i += 2*num_neighbors_cube;
	if( includeTargetHK )
		header[++i] = "hk";							// Don't forget the target hk column if requested!
	header[++i] = "ihk";
}


void kml::utils::rotateStencil90( double stencil[], const int& dir )
{
	// Lambda function to perform rotation of a 2D vector.
	auto _rotate90 = (dir >= 1)? []( double& x, double& y ){
		std::swap( x, (y *= -1) );			// Rotate by positive 90 degrees (counterclockwise).
	} : []( double& x, double& y ){
		std::swap( (x *= -1), y );			// Rotate by negative 90 degrees (clockwise).
	};

	// Lambda function to rotate features.
	auto _rotateFeatures = (dir >= 1)? []( double f[] ){
		double c[] = {f[2], f[5], f[8], f[1], f[4], f[7], f[0], f[3], f[6]};
		for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by positive 90 degrees (counterclokwise).
			f[i] = c[i];
	} : []( double f[] ){
		double c[] = {f[6], f[3], f[0], f[7], f[4], f[1], f[8], f[5], f[2]};
		for( int i = 0; i < num_neighbors_cube; i++ )		// Rotate by negative 90 degrees.
			f[i] = c[i];
	};

	// Rotate features.
	_rotateFeatures( &stencil[0] );						// Level-set values.
	_rotateFeatures( &stencil[num_neighbors_cube] );	// Normal x-components.
	_rotateFeatures( &stencil[2*num_neighbors_cube] );	// Normal y-components.

	// Rotate actual vectors.
	for( int i = num_neighbors_cube; i < 2 * num_neighbors_cube; i++ )
		_rotate90( stencil[i], stencil[i + num_neighbors_cube] );
}


void kml::utils::reflectStencil_yEqx( double stencil[] )
{
	// First the swap all features from one vertex to another.
	for( int i = 0; i < 3; i++ )
	{
		int offset = num_neighbors_cube*i;
		std::swap( stencil[1 + offset], stencil[3 + offset] );
		std::swap( stencil[2 + offset], stencil[6 + offset] );
		std::swap( stencil[5 + offset], stencil[7 + offset] );
	}

	// Then, swap normal components: x<->y.
	for( int i = 0; i < num_neighbors_cube; i++ )
		std::swap( stencil[num_neighbors_cube + i], stencil[2*num_neighbors_cube + i] );
}


void kml::utils::rotateStencilToFirstQuadrant( double stencil[], const double gradient[P4EST_DIM] )
{
	double theta = atan2( gradient[1], gradient[0] );
	const double TWO_PI = 2. * M_PI;
	theta = (theta < 0)? TWO_PI + theta : theta;		// Make sure current angle lies in [0, 2pi].

	// Rotate only if theta not in [0, pi/2].
	if( theta > M_PI_2 )
	{
		if( theta <= M_PI )								// Quadrant II?
		{
			rotateStencil90( stencil, -1 );				// Rotate by -pi/2.
		}
		else if( theta < M_PI_2 * 3 )					// Quadrant III?
		{
			rotateStencil90( stencil );					// Rotate by pi.
			rotateStencil90( stencil );
		}
		else											// Quadrant IV?
		{
			rotateStencil90( stencil );					// Rotate by pi/2.
		}
	}
}


////////////////////////////////////////////////////// Curvature ///////////////////////////////////////////////////////

kml::Curvature::Curvature( const NeuralNetwork *nnet, const double& h, const double& loMinHK, const double& upMinHK )
						   : H( h ), LO_MIN_HK( loMinHK ), UP_MIN_HK( upMinHK ), _nnet( nnet )
{
	if( ABS( H - _nnet->getH() ) > PETSC_MACHINE_EPSILON )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::Constructor: Neural network and current spacing are incompatible!" );

	if( loMinHK >= upMinHK || loMinHK < 0 )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::Constructor: HK lower- and upper-bound must be positive!" );
}


void kml::Curvature::_collectSamples( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec hk,
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
	interp.set_input( hk, interpolation_method::linear );

	// Collect samples: one per valid (i.e., with full h-uniform stencil) locally owned node next to Gamma.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, &ngbd, (char)maxRL );
	std::vector<p4est_locidx_t> ngIndices;
	nodesAlongInterface.getIndices( &phi, ngIndices );

	samples.clear();
	indices.clear();
	samples.reserve( nodes->num_owned_indeps );
	indices.reserve( nodes->num_owned_indeps );
	double xyz[P4EST_DIM];
	const double ONE_OVER_H = 1. / H;

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
				sample.push_back( interp( xyz ) );

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


void kml::Curvature::_computeHybridHK( const std::vector<std::vector<double>>& samples, std::vector<double>& hybHK ) const
{
	hybHK.clear();
	hybHK.resize( samples.size() );

	// Since we receive samples with phi h-normalization, let's put them into negative-curvature form and reorient them.
	int outIdx = 0;				// To avoid unnecessary neural evaluations, skip samples for which |ihk| < LO_MIN_HK.
	std::vector<int> outIdxToSampleIdx;
	outIdxToSampleIdx.reserve( samples.size() );
	std::vector<std::vector<double>> negSamples;
	negSamples.reserve( samples.size() );
	double grad[P4EST_DIM];
	int gradIdx = num_neighbors_cube / 2 + num_neighbors_cube;
	for( int i = 0; i < samples.size(); i++ )
	{
		if( ABS( samples[i][K_INPUT_SIZE-1] ) < LO_MIN_HK )		// Skip well resolved stencils; use the numerical estimation.
		{
			hybHK[i] = samples[i][K_INPUT_SIZE-1];
			continue;
		}

		negSamples.emplace_back( std::vector<double>( K_INPUT_SIZE ) );
		for( int j = 0; j < K_INPUT_SIZE; j++ )
			negSamples.back()[j] = (samples[i][K_INPUT_SIZE-1] > 0)? -samples[i][j] : samples[i][j];

		for( int dim = 0; dim < P4EST_DIM; dim++ )	// Let's pick a numerically good gradient for sample reorientation.
			grad[dim] = (negSamples.back()[gradIdx + dim * num_neighbors_cube] == 0)? EPS : negSamples.back()[gradIdx + dim * num_neighbors_cube];

		utils::rotateStencilToFirstQuadrant( negSamples.back(), grad );
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

		utils::reflectStencil_yEqx( negSamples[i] );	// And reflected about y=x.
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

		// Blend with numerical ihk within the range of [0.004, 0.007] (using ihk as the indicator).
		double ihk = samples[outIdxToSampleIdx[i]][K_INPUT_SIZE-1];
		if( ABS( ihk ) <= UP_MIN_HK )
		{
			double lam = (UP_MIN_HK - ABS( ihk )) / (UP_MIN_HK - LO_MIN_HK);
			hk = (1 - lam) * hk + lam * -ABS( ihk );
		}

		// Fix sign according to (untouched) curvature.
		hybHK[outIdxToSampleIdx[i]] = SIGN( ihk ) * ABS( hk );
	}

	// Clean up.
	delete [] inputs;
	delete [] outputs;
}


void kml::Curvature::compute( const my_p4est_node_neighbors_t& ngbd, Vec phi, Vec normal[P4EST_DIM], Vec numCurvature,
							  Vec hybCurvature, Vec hybFlag ) const
{
	if( !phi || !normal || !numCurvature || !hybCurvature || !hybFlag )
		throw std::runtime_error( "[CASL_ERROR] kml::Curvature::compute: One of the provided vectors is null!" );

	// Data accessors.
	const p4est_nodes_t *nodes = ngbd.get_nodes();

	// We start by computing the numerical mean curvature (as a byproduct of calling this function).
	compute_mean_curvature( ngbd, phi, normal, numCurvature );

	// Numerical dimensionless curvature *at* the grid points.
	Vec hk;
	CHKERRXX( VecDuplicate( numCurvature, &hk ) );
	CHKERRXX( VecCopy( numCurvature, hk ) );

	double *hkPtr;
	CHKERRXX( VecGetArray( hk, &hkPtr ) );
	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		hkPtr[n] *= H;
	CHKERRXX( VecRestoreArray( hk, &hkPtr ) );

	// Collect samples.
	std::vector<std::vector<double>> samples;
	std::vector<p4est_locidx_t> indices;
	_collectSamples( ngbd, phi, normal, hk, samples, indices );

	// Compute hybrid (dimensionless) curvature.
	std::vector<double> hybHK;
	_computeHybridHK( samples, hybHK );

	// Copy solution to parallel vectors.
	double *hybCurvaturePtr, *hybFlagPtr;
	CHKERRXX( VecGetArray( hybFlag, &hybFlagPtr ) );
	CHKERRXX( VecGetArray( hybCurvature, &hybCurvaturePtr ) );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )		// Initialization.
		hybCurvaturePtr[n] = hybFlagPtr[n] = 0;

	for( int i = 0; i < indices.size(); i++ )	// Go through the nodes where we computed the hybrid solution.
	{
		p4est_locidx_t idx = indices[i];
		hybCurvaturePtr[idx] = hybHK[i] / H;
		hybFlagPtr[idx] = 1;					// Signal that node idx contains kappa "at" the interface.
	}

	// Cleaning up.
	CHKERRXX( VecRestoreArray( hybCurvature, &hybCurvaturePtr ) );
	CHKERRXX( VecRestoreArray( hybFlag, &hybFlagPtr ) );
	CHKERRXX( VecDestroy( hk ) );

	// Let's synchronize the machine learning flag vector among all processes.
	CHKERRXX( VecGhostUpdateBegin( hybFlag, INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( hybFlag, INSERT_VALUES, SCATTER_FORWARD ) );

	// Let's synchronize the curvature values among all processes.
	CHKERRXX( VecGhostUpdateBegin( hybCurvature, INSERT_VALUES, SCATTER_FORWARD ) );
	CHKERRXX( VecGhostUpdateEnd( hybCurvature, INSERT_VALUES, SCATTER_FORWARD ) );
}
