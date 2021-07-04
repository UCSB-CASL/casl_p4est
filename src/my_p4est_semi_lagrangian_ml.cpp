#include "my_p4est_semi_lagrangian_ml.h"

//////////////////////////////////////////////// Scaler Abstract Class /////////////////////////////////////////////////

void slml::Scaler::_printParams( const slml::Scaler::json& params, const std::string& paramsFileName )
{
	std::cout << "===----------------------------- Loaded parameters -----------------------------===" << std::endl;
	std::cout << "Source JSON file: '" << paramsFileName << "'" << std::endl;
	std::stringstream o;
	o << std::setw( 4 ) << params;
	std::cout << o.str() << std::endl;
	std::cout << "===-----------------------------------------------------------------------------===" << std::endl;
}


////////////////////////////////////////////////////// PCAScaler ///////////////////////////////////////////////////////

slml::PCAScaler::PCAScaler( const std::string& paramsFileName, const bool& printLoadedParams )
{
	// Load parameters.
	std::ifstream input( paramsFileName );
	json params;
	input >> params;

	// Assign parameters to internal variables.
	const std::string errorPrefix = "[CASL_ERROR] slml::PCAScaler: ";
	if( params.contains( "components" ) )					// Loading the components.
	{
		const auto& param = params["components"];
		for( const auto& component : param )
		{
			_components.emplace_back();
			for( const auto& val : component )
				_components.back().push_back( val.get<double>() );
		}
	}
	else
		throw std::runtime_error( errorPrefix + " components are missing!" );

	if( params.contains( "mean" ) )							// Loading the mean vector, which must have MASS_INPUT_SIZE elements.
	{
		const auto& means = params["mean"];
		if( means.size() == MASS_INPUT_SIZE )
		{
			for( const auto& mean : means )
				_means.push_back( mean.get<double>() );
		}
		else
			throw std::runtime_error( errorPrefix + " invalid number of mean values!" );
	}
	else
		throw std::runtime_error( errorPrefix + " mean values are missing!" );

	if( params.contains( "variance" ) )					// Loading the variance vector, which must have MASS_N_COMPONENTS elements.
	{
		const auto& variances = params["variance"];
		if( variances.size() == MASS_N_COMPONENTS )
		{
			for( const auto& variance : variances )
				_stds.push_back(  sqrt( variance.get<double>() ) );
		}
		else
			throw std::runtime_error( errorPrefix + " invalid number of variances!" );
	}

	if( printLoadedParams )
		_printParams( params, paramsFileName );
}


void slml::PCAScaler::transform( double samples[][MASS_INPUT_SIZE], const int &nSamples ) const
{
	// To transform in Python: ((Y-pcaw.mean_)@pcaw.components_.T)/np.sqrt(pcaw.explained_variance_)
	for( int i = 0; i < nSamples; i++ )
	{
		// First, copy sample and subtract the mean.
		double sample[MASS_INPUT_SIZE];
		for( int j = 0; j < MASS_INPUT_SIZE; j++ )
			sample[j] = samples[i][j] - _means[j];

		// Second, project onto principal components.
		double projected[MASS_N_COMPONENTS];
		for( int c = 0; c < MASS_N_COMPONENTS; c++ )
		{
			projected[c] = 0;
			for( int j = 0; j < MASS_INPUT_SIZE; j++ )
				projected[c] += sample[j] * _components[c][j];
		}

		// Third, divide by explained standard deviation while writing to output array.
		for( int j = 0; j < MASS_INPUT_SIZE; j++ )
		{
			if( j < MASS_N_COMPONENTS )
				samples[i][j] = projected[j] / _stds[j];
			else
				samples[i][j] = 0;			// Fill missing slots (i.e., belonging to no component) with zeros.
		}
	}
}


//////////////////////////////////////////////////// StandardScaler ////////////////////////////////////////////////////

slml::StandardScaler::StandardScaler( const std::string& paramsFileName, const bool& printLoadedParams )
{
	// Load parameters.
	std::ifstream input( paramsFileName );
	json params;
	input >> params;

	const std::string errorPrefix = "[CASL_ERROR] slml::StandardScaler Constructor: ";

	// Assign parameters to internal variables.
	if( params.size() < 4 )
		throw std::runtime_error( errorPrefix + "expecting at least 4 groups of parameters!" );

	_loadParams( "coord", params, _meanCoord, _stdCoord );
	_loadParams( "dist", params, _meanDist, _stdDist );
	_loadParams( "phi", params, _meanPhi, _stdPhi );
	_loadParams( "vel", params, _meanVel, _stdVel );
	_loadParams( "hk", params, _meanHK, _stdHK );
	_loadParams( "h2_phi_xx", params, _meanH2Phi_xx, _stdH2Phi_xx );

	if( printLoadedParams )
		_printParams( params, paramsFileName );
}


void slml::StandardScaler::_loadParams( const std::string& inName, const json& params, double& outMean, double& outStd )
{
	const std::string errorPrefix = "[CASL_ERROR] slml::StandardScaler::_loadParams: ";
	if( params.contains( inName ) )
	{
		const auto& param = params[inName];

		if( param.contains( "mean" ) )			// Loading mean value of input parameter.
			outMean = param["mean"].get<double>();
		else
			throw std::runtime_error( errorPrefix + inName + " mean is missing!" );

		if( param.contains( "std" ) )			// Loading standard deviation.
			outStd = param["std"].get<double>();
		else
			throw std::runtime_error( errorPrefix + inName + " standard deviation is missing!" );
	}
	else
		throw std::runtime_error( errorPrefix + inName + " parameter is missing!" );
}


void slml::StandardScaler::transform( double samples[][MASS_INPUT_SIZE], const int& nSamples ) const
{
	for( int i = 0; i < nSamples; i++ )
	{
		// Scaling phi values.
		for( const auto& j : PHI_COLS )
			samples[i][j] = (samples[i][j] - _meanPhi) / _stdPhi;

		// Scaling velocity components.
		for( const auto& j : VEL_COLS )
			samples[i][j] = (samples[i][j] - _meanVel) / _stdVel;

		// Scaling distance variable.
		for( const auto& j : DIST_COLS )
			samples[i][j] = (samples[i][j] - _meanDist) / _stdDist;

		// Scaling departure point coords.
		for( const auto& j : COORDS_COLS )
			samples[i][j] = (samples[i][j] - _meanCoord) / _stdCoord;

		// Scaling dimensionless curvature for arrival point.
		for( const auto& j : HK_COLS )
			samples[i][j] = (samples[i][j] - _meanHK) / _stdHK;

		// Scaling second spatial derivatives of phi at departure point.
		for( const auto& j : H2_PHI_XX_COLS )
			samples[i][j] = (samples[i][j] - _meanH2Phi_xx) / _stdH2Phi_xx;
	}
}


void slml::StandardScaler::untransformPhi( double phi[], const int& nValues ) const
{
	for( int i = 0; i < nValues; i++ )
		phi[i] = phi[i] * _stdPhi + _meanPhi;
}


//////////////////////////////////////////////////// NeuralNetwork /////////////////////////////////////////////////////

slml::NeuralNetwork::NeuralNetwork( const std::string& modelPath, const std::string& transformerPath, const double& h,
									const bool& verbose )
									: H( h ), _model( fdeep::load_model( modelPath, true,
								    verbose? fdeep::cout_logger: fdeep::dev_null_logger ) ),
									_pcaScaler( transformerPath, verbose ){}


void slml::NeuralNetwork::predict( double inputs[][MASS_INPUT_SIZE], double outputs[], const int& nSamples ) const
{
	// Normalize backtracked phi_d by h and leave hk as a negative value.
	for( int i = 0; i < nSamples; i++ )
	{
		inputs[i][MASS_INPUT_SIZE-1] /= H;
		inputs[i][MASS_INPUT_SIZE-2] = -ABS( inputs[i][MASS_INPUT_SIZE-2] );
	}

	// Second part of inputs is the normalized backtracked phi_d.
	double inputsPt2[nSamples];
	for( int i = 0; i < nSamples; i++ )
		inputsPt2[i] = inputs[i][MASS_INPUT_SIZE-1];

	// First, preprocess inputs in part 1.
	_pcaScaler.transform( inputs, nSamples );

	// Proceed with predictions, one input at a time.
	for( int i = 0; i < nSamples; i++ )
	{
		// Build two-part input.
		std::vector<FDEEP_FLOAT_TYPE> input1( &inputs[i][0], &inputs[i][0] + INPUT_WIDTH_PT1 );
		std::vector<FDEEP_FLOAT_TYPE> input2( &inputsPt2[i], &inputsPt2[i] + INPUT_WDITH_PT2 );

		// Add split inputs as independent entries to a two-element tensor.
		std::vector<fdeep::tensor> inputTensors = {
			fdeep::tensor( fdeep::tensor_shape( INPUT_WIDTH_PT1 ), input1 ),
			fdeep::tensor( fdeep::tensor_shape( INPUT_WDITH_PT2 ), input2 )
		};

		// Predict and undo normalization, too.
		outputs[i] = _model.predict_single_output( inputTensors ) * H;
	}
}


///////////////////////////////////////////////////// DataFetcher //////////////////////////////////////////////////////

slml::DataFetcher::DataFetcher( const my_p4est_node_neighbors_t *ngbd )
								: my_p4est_interpolation_t( ngbd ), _nodes( ngbd_n->get_nodes() ){}


void slml::DataFetcher::setInput( Vec fields[], const int& nFields )
{
	set_input_fields( fields, nFields, 1 );
}


void slml::DataFetcher::interpolate( const p4est_quadrant_t &quad, const double *xyz, double *results,
									 const unsigned int &comp ) const
{
	PetscErrorCode ierr;
	const p4est_topidx_t &treeIdx = quad.p.piggy3.which_tree;
	const p4est_locidx_t &quadIdx = quad.p.piggy3.local_num;

	const size_t N_FIELDS = n_vecs();		// Number of input fields.
	P4EST_ASSERT( N_FIELDS > 0 );
	P4EST_ASSERT( bs_f == 1 );				// Here, we only allow one-block vectors.
	P4EST_ASSERT( comp == ALL_COMPONENTS );	// Here, we only allow ALL_COMPONENTS.

	// Acquire read-only access to input fields.
	const double *fieldsReadPtr[N_FIELDS];
	for( int k = 0; k < N_FIELDS; ++k )
	{
		ierr = VecGetArrayRead( Fi[k], &fieldsReadPtr[k] );
		CHKERRXX( ierr );
	}
	const size_t N_ELEMS_PER_NODE = N_FIELDS;		// We return as many elements as input fields.
	double f[N_ELEMS_PER_NODE * P4EST_CHILDREN]; 	// f[k*P4EST_CHILDREN+i] = value of the kth field at child node i.

	// Serialize all input fields for all quad child nodes.
	// Suppose N_FIELDS = 2, and P4EST_CHILDREN = 4.  Field 1 => a's, Field 2 => b's.
	//       0  1  2  3  4  5  6  7
	// f = [a0 a1 a2 a3 b0 b1 b2 b3]
	for( int i = 0; i < P4EST_CHILDREN; i++ )
	{
		p4est_locidx_t nodeIdx = _nodes->local_nodes[quadIdx * P4EST_CHILDREN + i];
		for( size_t k = 0; k < N_FIELDS; ++k )
			f[k * P4EST_CHILDREN + i] = fieldsReadPtr[k][nodeIdx];
	}

	for( size_t k = 0; k < N_FIELDS; ++k )
	{
		ierr = VecRestoreArrayRead( Fi[k], &fieldsReadPtr[k] );
		CHKERRXX( ierr );
	}

	// Fetch data.
	_fetch( p4est, treeIdx, quad, f, xyz, results, N_ELEMS_PER_NODE );
}


void slml::DataFetcher::_fetch( const p4est_t *p4est, p4est_topidx_t treeId, const p4est_quadrant_t& quad,
								const double *fields, const double xyz[P4EST_DIM], double *results,
								const size_t& nResults )
{
	P4EST_ASSERT( nResults > 0 );

	// Let's initialize the results array.
	for( int i = 0; i < nResults; i++ )
		results[i] = 0;

	// Check that the query point lies in a quad with the highest resolution level.  If not, we can't use the
	// quad information because it's inconsistent with the machine learning assumptions (i.e., all training quads are
	// the highest resolution possible).
	auto* p4estUserData = (splitting_criteria_t*)p4est->user_pointer;
	if( quad.level != p4estUserData->max_lvl )
	{
		results[0] = -1;		// Error: we can't continue processing this query point.
		return;
	}

	int outOffset = 1;			// This will be indicating where to start populating the output results.

	// Retrieve normalized point coordinates w.r.t. quad/oct min corner.
	double normalizedXYZ[P4EST_DIM];
	_getNormalizedCoords( p4est, treeId, quad, xyz, normalizedXYZ );
	for( int i = 0; i < P4EST_DIM; i++ )
		results[outOffset + i] = normalizedXYZ[i];
	outOffset += P4EST_DIM;

	// Next, retrieve the field values at the 4 (8) corners of the quad (oct).  We have 1 + 2 * P4EST_DIM input fields.
	// Also, recall that children (e.g., nodes in a quad) appear as zyx (z being the slowest changing coordinate).
	// We need them the other way around in the output: xyz (x being the slowest changing).
	int fieldIdx;		// Note: The spatial 2nd derivatives of phi are processed differently -- see below.
	for( fieldIdx = InputFields::PHI; fieldIdx != InputFields::PHI_XX; fieldIdx++ )
	{
		for( int x = 0; x < 2; x++ )				// Truth table for output format, with x changing slowly, y changing
			for( int y = 0; y < 2; y++ )			// faster than x, and z changing faster than y.
#ifdef P4_TO_P8
				for( int z = 0; z < 2; z++ )
#endif
			{
				int outIdx = SUMD( (1u << (unsigned)( P4EST_DIM - 1 )) * x,
								   (1u << (unsigned)( P4EST_DIM - 2 )) * y,
								   z );
				int childIdxInQuad = SUMD( x, 2 * y, 4 * z );
				results[outOffset + outIdx] = fields[P4EST_CHILDREN * fieldIdx + childIdxInQuad];
			}
		outOffset += P4EST_CHILDREN;
	}

	// Use bilinear interpolation to compute phi_xx, phi_yy[, phi_zz] at departure point (using its normalized coords).
	linear_interpolation( p4est, treeId, quad, &fields[P4EST_CHILDREN * fieldIdx], xyz, &results[outOffset], P4EST_DIM );
	outOffset += P4EST_DIM;

	// Verify everything went right.
	if( nResults != outOffset )
	{
		throw std::runtime_error( "[CASL_ERROR]: slml::DataFetcher::_fetch: Number of fields and ouputs mismatch!" );
	}
}


void slml::DataFetcher::_getNormalizedCoords( const p4est_t *p4est, const p4est_topidx_t& treeId,
											  const p4est_quadrant_t& quad, const double xyz[P4EST_DIM],
											  double normalizedXYZ[P4EST_DIM] )
{
	// First and last vertex in tree.  The convention for enumerating nodes in the tree and quad is opposite to what I
	// had thought and based my code.  In this case, child nodes appear depending on the truth table of 3 variables, in
	// that order: z, y, x, where x is the fastest changing state and z is the slowest changing var.  This can be seen
	// in the last section of my_p4est_utils::get_local_interpolation_weights.  The location of a child node is computed
	// as SUMD(x, 2*y, 4*z), with the state vars taking the values of 0 and 1.
	p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[treeId*P4EST_CHILDREN + 0];
	p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[treeId*P4EST_CHILDREN + P4EST_CHILDREN - 1];

	const double *treeXYZ_min = (p4est->connectivity->vertices + 3 * v_m);		// Coordinates of first and last tree
	const double *treeXYZ_max = (p4est->connectivity->vertices + 3 * v_p);		// vertices.

	const double qh = (double)P4EST_QUADRANT_LEN( quad.level ) / (double)(P4EST_ROOT_LEN);	// Cell width.
	double qxyz_min[P4EST_DIM];
	quad_xyz_fr_ijk( &quad, qxyz_min );		// Coordinates of bottom left quad corner in the tree cordinate system.

	double xyzLocal[P4EST_DIM] = {DIM( xyz[0], xyz[1], xyz[2] )};	// Coordinates of query point w.r.t. tree system.
	for( int dir = 0; dir < P4EST_DIM; dir++ )
		xyzLocal[dir] = (xyzLocal[dir] - treeXYZ_min[dir]) / (treeXYZ_max[dir] - treeXYZ_min[dir]);

	P4EST_ASSERT(ANDD(!is_periodic(p4est, dir::x) || (xyzLocal[0] >= qxyz_min[0] - qh/10 && xyzLocal[0] <= qxyz_min[0] + qh + qh/10),
					  !is_periodic(p4est, dir::y) || (xyzLocal[1] >= qxyz_min[1] - qh/10 && xyzLocal[1] <= qxyz_min[1] + qh + qh/10),
					  !is_periodic(p4est, dir::z) || (xyzLocal[2] >= qxyz_min[2] - qh/10 && xyzLocal[2] <= qxyz_min[2] + qh + qh/10)));

	// Compute normalized coords w.r.t. quad/oct coordinate system.  The results lie always between 0 and 1.
	for( int dir = 0; dir < P4EST_DIM; dir++ )
		normalizedXYZ[dir] = (xyzLocal[dir] - qxyz_min[dir]) / qh;
}


void slml::DataFetcher::operator()( const double *xyz, double *results ) const
{
	throw std::runtime_error( "[CASL_ERROR] slml::DataFetcher::operator(): Not implemented yet!" );
}


///////////////////////////////////////////////////// Cache Fetcher ////////////////////////////////////////////////////

const int slml::Cache::N_FIELDS = 2;

slml::Cache::Cache( const my_p4est_node_neighbors_t *ngbd )
					: my_p4est_interpolation_t( ngbd ), _nodes( ngbd_n->get_nodes() ){}


void slml::Cache::setInput( Vec fields[], const int& nFields )
{
	set_input_fields( fields, nFields, 1 );
}


void slml::Cache::interpolate( const p4est_quadrant_t &quad, const double *xyz, double *results,
									  const unsigned int &comp ) const
{
	PetscErrorCode ierr;
	const p4est_topidx_t &treeIdx = quad.p.piggy3.which_tree;
	const p4est_locidx_t &quadIdx = quad.p.piggy3.local_num;

	P4EST_ASSERT( n_vecs() == N_FIELDS );
	P4EST_ASSERT( N_FIELDS > 0 );
	P4EST_ASSERT( bs_f == 1 );				// Here, we only allow one-block vectors.
	P4EST_ASSERT( comp == ALL_COMPONENTS );	// Here, we only allow ALL_COMPONENTS.

	// Acquire read-only access to input fields.
	const double *fieldsReadPtr[N_FIELDS];
	for( int k = 0; k < N_FIELDS; ++k )
	{
		ierr = VecGetArrayRead( Fi[k], &fieldsReadPtr[k] );
		CHKERRXX( ierr );
	}

	// Go through all quad's children and check which one corresponds to queried node global coords.
	results[Fields::PHI] = 0;
	results[Fields::FLAG] = 0;
	for( int i = 0; i < P4EST_CHILDREN; i++ )
	{
		p4est_locidx_t nodeIdx = _nodes->local_nodes[quadIdx * P4EST_CHILDREN + i];
		double childCoords[P4EST_DIM];
		node_xyz_fr_n( nodeIdx, p4est, _nodes, childCoords );
		if( ANDD( ABS( childCoords[0] - xyz[0] ) < PETSC_MACHINE_EPSILON,
				  ABS( childCoords[1] - xyz[1] ) < PETSC_MACHINE_EPSILON,
				  ABS( childCoords[2] - xyz[2] ) < PETSC_MACHINE_EPSILON )
			  && fieldsReadPtr[Fields::FLAG][nodeIdx] > 0 )	// A match?
		{
			results[Fields::PHI] = fieldsReadPtr[Fields::PHI][nodeIdx];
			results[Fields::FLAG] = fieldsReadPtr[Fields::FLAG][nodeIdx];
			break;
		}
	}

	// Restore array read access.
	for( size_t k = 0; k < N_FIELDS; ++k )
	{
		ierr = VecRestoreArrayRead( Fi[k], &fieldsReadPtr[k] );
		CHKERRXX( ierr );
	}
}


void slml::Cache::operator()( const double *xyz, double *results ) const
{
	throw std::runtime_error( "[CASL_ERROR] sml::CacheFetcher::operator(): Not implemented yet!" );
}


//////////////////////////////////////////////////// SemiLagrangian ////////////////////////////////////////////////////

slml::SemiLagrangian::SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
									  my_p4est_node_neighbors_t *ngbdN,
									  const std::unordered_set<p4est_locidx_t> *localUniformIndices,
									  const NeuralNetwork * const nnet,
									  const double& band, const unsigned long& iteration )
									  : BAND( MAX( 2.0, band ) ), 		// Minimum bandwidth of 2 to give enough space.
									  ITERATION( iteration ),
									  VEL_INTERP_MTHD( interpolation_method::quadratic ),
									  PHI_INTERP_MTHD( interpolation_method::quadratic ),
									  _nnet( nnet ),
									  my_p4est_semi_lagrangian_t( p4estNp1, nodesNp1, ghostNp1, ngbdN )
{
	if( band < 2 )
		throw std::runtime_error( "[CASL_ERROR] slml::SemiLagrangian Constructor: band must be at least 2!" );

	_localUniformIndicesPtr = localUniformIndices;

	velo_interpolation = VEL_INTERP_MTHD;
	phi_interpolation = PHI_INTERP_MTHD;
}


slml::SemiLagrangian::SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
									  my_p4est_node_neighbors_t *ngbdN, Vec phi, const double& band,
									  const unsigned long& iteration )
									  : BAND( MAX( 2.0, band ) ), 		// Minimum bandwidth of 2 to give enough space.
										ITERATION( iteration ),
										VEL_INTERP_MTHD( interpolation_method::quadratic ),
										PHI_INTERP_MTHD( interpolation_method::quadratic ),
										_nnet( nullptr ),
										my_p4est_semi_lagrangian_t( p4estNp1, nodesNp1, ghostNp1, ngbdN )
{
	if( band < 2 )
		throw std::runtime_error( "[CASL_ERROR] slml::SemiLagrangian Constructor: band must be at least 2!" );

	_localUniformIndicesPtr = _computeLocalUniformIndices( ngbdN, phi );

	velo_interpolation = VEL_INTERP_MTHD;
	phi_interpolation = PHI_INTERP_MTHD;
}


const std::unordered_set<p4est_locidx_t>* slml::SemiLagrangian::_computeLocalUniformIndices( const my_p4est_node_neighbors_t *ngbdN, Vec phi )
{
	if( !phi )
		throw std::runtime_error( "[CASL_ERROR] slml::SemiLagrangian Constructor: phi values must be supplied!" );

	// Some pointers to structs for the the grid/forest at time tn.
	p4est_t const *p4estN = ngbdN->get_p4est();
	p4est_nodes_t const *nodesN = ngbdN->get_nodes();

	_localUniformIndices.clear();
	_localUniformIndices.reserve( nodesN->num_owned_indeps );

	// We need the indices for nodes next to the interface (i.e., one of their irradiating edges is crossed by Gamma).
	// Since curvature is used to flip samples, we need only locally owned nodes because we can use the curvature
	// nnet with them as long as they have a uniform stencil.  We collect this indices once per semi-Lagrangian step
	// only at the beginning of the process to save time.
	auto* p4estUserData = (splitting_criteria_t*)p4estN->user_pointer;
	NodesAlongInterface nodesAlongInterface( p4estN, nodesN, ngbdN, (signed char)p4estUserData->max_lvl );
	vector<p4est_locidx_t> indices;
	nodesAlongInterface.getIndices( &phi, indices );

	// Check each node if it has a full uniform stencil.
	for( auto nodeIdx : indices )
	{
		try
		{
			std::vector<p4est_locidx_t> stencilIndices( num_neighbors_cube );
			if( nodesAlongInterface.getFullStencilOfNode( nodeIdx, stencilIndices ) )
				_localUniformIndices.insert( nodeIdx );
		}
		catch( const std::exception &exception ) {}
	}

	return &_localUniformIndices;
}


size_t slml::SemiLagrangian::freeDataPacketArray( vector<DataPacket *>& dataPackets )
{
	size_t count = 0;
	for( auto& dataPacket : dataPackets )
	{
		delete dataPacket;
		dataPacket = nullptr;		// Taking precautions.
		count++;
	}
	dataPackets.clear();			// More precautions.

	return count;
}


bool slml::SemiLagrangian::collectSamples( Vec vel[P4EST_DIM], Vec *vel_xx[P4EST_DIM], const double& dt,
										   Vec phi, Vec phi_xx[P4EST_DIM], std::vector<DataPacket *>& dataPackets ) const
{
	PetscErrorCode ierr;
	ierr = PetscLogEventBegin( log_SemiLagrangianML_collectSamples, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	// Some pointers to structs for the the grid/forest at time tn.
	p4est_t const *p4est_n = ngbd_n->get_p4est();
	p4est_nodes_t const *nodes_n = ngbd_n->get_nodes();

	// Initialize output array.
	freeDataPacketArray( dataPackets );		// Just in case, free and clear whatever is left in output array.
	dataPackets.reserve( _localUniformIndicesPtr->size() );

	// Domain features.
	const double *xyz_min = get_xyz_min();
	const double *xyz_max = get_xyz_max();
	const bool *periodicity = get_periodicity();

	// Retrieve grid size data.
	double dxyz[P4EST_DIM];
	double dxyz_min;			// Minimum cell width for current macromesh.  Use this to normalize distances below.
	get_dxyz_min( p4est_n, dxyz, dxyz_min );

	// Getting read access to phi and velocity parallel PETSc vectors.  These will be our input fields.
	const double *phiReadPtr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	const double *velReadPtr[P4EST_DIM];
	for( int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = VecGetArrayRead( vel[dim], &velReadPtr[dim] );
		CHKERRXX( ierr );
	}

	// To backtrack points, we need to compute velStar, which is computed as a midpoint approximation.  This comes from:
	// x^* = x^np1 - dt/2 * u^n(x^np1),
	// x_d = x^np1 - dt * u^n(x^*).
	// This approximation holds because u^n is the same as u^np1 (that's the assumption).  Further, we assume G^n to be
	// G^np1 in the first step.
	my_p4est_interpolation_nodes_t velInterp( ngbd_n );

	Vec xx_v_derivatives[P4EST_DIM] = {DIM( vel_xx[0][0], vel_xx[1][0], vel_xx[2][0] )};	// Reorganize spatial 2nd derivatives
	Vec yy_v_derivatives[P4EST_DIM] = {DIM( vel_xx[0][1], vel_xx[1][1], vel_xx[2][1] )};	// of velocity field for interpolation.
#ifdef P4_TO_P8
	Vec zz_v_derivatives[P4EST_DIM] = {vel_xx[0][2], vel_xx[1][2], vel_xx[2][2]};
#endif

	double *velStar[P4EST_DIM] = {nullptr, nullptr};
	for( auto& dir : velStar )		// To allocate mid velocity it is safe to use only nodes we are interested in.
		dir = new double[_localUniformIndicesPtr->size()];

	// Now, find velStar.
	std::unordered_map<p4est_locidx_t, int> velStarNodeIdxToOutIdxMap;	// We need to map node indices to out indices.
	velStarNodeIdxToOutIdxMap.reserve( _localUniformIndicesPtr->size() );
	int velStarOutIdx = 0;
	for( const auto& n : *_localUniformIndicesPtr )
	{
		double xyzStar[P4EST_DIM];
		node_xyz_fr_n( n, p4est_n, nodes_n, xyzStar );
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			xyzStar[dir] -= 0.5 * dt * velReadPtr[dir][n];
		clip_in_domain( xyzStar, xyz_min, xyz_max, periodicity );

		velInterp.add_point( velStarOutIdx, xyzStar );
		velStarNodeIdxToOutIdxMap[n] = velStarOutIdx;
		velStarOutIdx++;
	}
	velInterp.set_input( vel, DIM( xx_v_derivatives, yy_v_derivatives, zz_v_derivatives ), VEL_INTERP_MTHD, P4EST_DIM );
	velInterp.interpolate( velStar );
	velInterp.clear();				// At this point, we can use velStar to find backtracked departure points.

	// Prepare an interpolation object on the current grid.  We need this to retrieve the numerical solution to the
	// advection problem.  This numerical approximation would then be compared to the target value (collected with a
	// finer grid.
	my_p4est_interpolation_nodes_t numericalInterp( ngbd_n );

	// Initialize infrastructure.  We'll use the interpolation across processes infraestructure and hack the interpolate
	// method.  To do this, we need to instantiate dummy vectors that will pretend to work as empty "fields".  That way,
	// upon return, the multi-process interpolation will send back the data for the four (eight) quad (oct) corners:
	// In 2D: exit_code + normalized(x_d) + 4 phi values + 4*2 velocity components + 2 phi 2nd derivatives = 17 values per "interpolation".
	// In 3D: exit_code + normalized(x_d) + 8 phi values + 8*3 velocity components + 3 phi 2nd derivatives = 39 values per "interpolation".
	// Note that for phi_xx, we don't return them for each child in the quad; instead, we use bilinear interpolation at
	// x_d, as it is done in my_p4est_utils.cpp#quadratic_interpolation.
	// When setting up the input fields (in that order):
	// In 2D: phi + vel_u + vel_v + phi_xx + phi_yy [+ 12 dummy fields].
	// In 3D: phi + vel_u + vel_v + vel_w + phi_xx + phi_yy + phi_zz [+ 32 dummy fields].
	//
	// The exit_code above is used to determined if data collection finished successfully.  A possible error situation
	// arises, for example, if the quadrant owning the query point is not at the maximum level of refinement.  We need
	// to collect data from uniform quadrants at the maximum level of refinement only for training consistency.
	DataFetcher dataFetcher( ngbd_n );

	const int N_FIELDS = 1 + 2 * P4EST_DIM + (1 + P4EST_DIM) * P4EST_CHILDREN;	// Number of fields to receive.
	const int N_GIVEN_INPUT_FIELDS = 1 + 2 * P4EST_DIM;	// Actual number of true fields to send (phi, vel, and phi_dd).

	// Collect nodes along the interface and add their backtracked departure points to "interpolation" buffer.
	// Filter out any node whose backtracked departure point falls outside of domain (if no periodicity is allowed in
	// that direction).
	double xyz_d[P4EST_DIM];						// Departure point.
	double xyz_a[P4EST_DIM];						// Arrival point.
	std::vector<p4est_locidx_t> outIdxToNodeIdx;	// This is a map from out index to actual node index in PETSc Vecs.
	outIdxToNodeIdx.reserve( _localUniformIndicesPtr->size() );
	std::vector<double> distances;					// Vector to store the distance between arrival and departure points.
	distances.reserve( _localUniformIndicesPtr->size() );
	int outIdx = 0;									// Index in "interpolation" output.
	bool allInside = true;							// Warning flag: false if at least one point along the interface is
													// backtracked outside the domain.
													// Helps prevent inconsistent training samples.
	for( const auto& nodeIdx : *_localUniformIndicesPtr )
	{
		// Let's skip nodes for which the velocity at it or the midpoint velocity is practically zero.
		double velMagnitude = 0, velStarMagnitude = 0;
		for( int dir = 0; dir < P4EST_DIM; dir++ )
		{
			velMagnitude += SQR( velReadPtr[dir][nodeIdx] );
			velStarMagnitude += SQR( velStar[dir][velStarNodeIdxToOutIdxMap.at( nodeIdx )] );
		}
		if( sqrt( velMagnitude ) <= PETSC_MACHINE_EPSILON || sqrt( velStarMagnitude ) <= PETSC_MACHINE_EPSILON )
			continue;

		// Backtracking the point using a midpoint velocity step in the negative direction (2nd-order accurate).
		node_xyz_fr_n( nodeIdx, p4est_n, nodes_n, xyz_a );
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			xyz_d[dir] = xyz_a[dir] - dt * velStar[dir][velStarNodeIdxToOutIdxMap.at( nodeIdx )];

		// Check if departure point falls within the domain.
		// We won't admit truncated backtracked points to avoid inconsistency in the training patterns.
		if( !clip_in_domain_with_check( xyz_d, xyz_min, xyz_max, periodicity ) )
		{
			allInside = false;
			continue;
		}

		// Euclidean distance between arrival and backtracked departure point.
		double d = 0;
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			d += SQR( xyz_d[dir] - xyz_a[dir] );
		d = sqrt( d );

		// Add departure point to buffer and to vector map to node indices.
		dataFetcher.add_point( outIdx, xyz_d );
		outIdxToNodeIdx.push_back( nodeIdx );

		// Add the normalized distance from x_d to x_a.
		distances.push_back( d / dxyz_min );

		// Add request for numerical interpolation.
		numericalInterp.add_point( outIdx, xyz_d );

		outIdx++;
	}

	// At this point, outIdx is also equal to total number of valid nodes along the interface.  However, I still need to
	// set up and trigger interpolation even if the number of points expected in this process is zero.
	// Not doing so will enter the program into a deadlock because others need current process to have interpolation
	// set up.
	Vec fields[N_FIELDS] = {phi, DIM( vel[0], vel[1], vel[2] ), DIM( phi_xx[0], phi_xx[1], phi_xx[2] )};	// Input fields.
	for( int i = N_GIVEN_INPUT_FIELDS; i < N_FIELDS; i++ )		// Dummy fields are initialized with zeros.
	{
		ierr = VecCreateGhostNodes( p4est_n, nodes_n, &fields[i] );
		CHKERRXX( ierr );
	}

	// Allocate output vectors with as many elements as valid indices we collected in previous loop.  Set up fetch
	// inputs and launch data collection.
	double *output[N_FIELDS] = {nullptr, nullptr};
	for( auto& fOutput : output )
		fOutput = new double[outIdx];
	dataFetcher.setInput( fields, N_FIELDS );
	dataFetcher.interpolate( output );
	dataFetcher.clear();

	// Allocate output array for computed, numerically approximated phi value at departure point.
	auto *outputDepartureNumericalPhi = new double[outIdx];
	numericalInterp.set_input( phi, DIM( phi_xx[0], phi_xx[1], phi_xx[2] ), PHI_INTERP_MTHD );
	numericalInterp.interpolate( outputDepartureNumericalPhi );
	numericalInterp.clear();

	// Go through the output fields to build the data packet objects that will be returned to caller.
	// This will be a semi deserialization by splitting the output contents into the different attributes of the
	// data packet objects.
	for( int i = 0; i < outIdx; i++ )
	{
		if( output[0][i] == 0 )								// Success for ith node?
		{
			p4est_locidx_t nodeIdx = outIdxToNodeIdx[i];	// Actual node index in PETSc vector.

			// Allocate new packet for current ith node.
			auto *dataPacket = new DataPacket;
			dataPacket->nodeIdx = nodeIdx;
			dataPacket->phi_a = phiReadPtr[nodeIdx];		// Phi and (mid point) velocity at arrival point.
			for( int dim = 0; dim < P4EST_DIM; dim++ )
				dataPacket->vel_a[dim] = velStar[dim][velStarNodeIdxToOutIdxMap.at( nodeIdx )];
			dataPacket->distance = distances[i];			// Normalized distance from departure to arrival point.

			dataPacket->numBacktrackedPhi_d = outputDepartureNumericalPhi[i];	// Numerically backtracked phi.

			// Splitting serialized info in the output fields.
			int fIndex = 1;									// Field index.
			for( double& dim : dataPacket->xyz_d )			// Normalized departure coordinates.
			{
				dim = output[fIndex][i];
				fIndex++;
			}
			for( double& child : dataPacket->phi_d )		// Level-set values of corners of departure point owner.
			{
				child = output[fIndex][i];
				fIndex++;
			}
			for( double& velCompChild : dataPacket->vel_d )	// Serialized velocity at corners of departure
			{												// point owner.  Order is vel_u, vel_v, vel_z.  For each
				velCompChild = output[fIndex][i];			// component we have P4EST_CHILDREN values.
				fIndex++;
			}
			for( double& phi_xxComp : dataPacket->h2_phi_xx_d ) 		// Serialized, abs. value of h^2 * 2nd spatial
			{															// phi derivative bilinearly interpolated at the
				phi_xxComp = SQR( dxyz_min ) * ABS( output[fIndex][i] );// dep. point.  Order is phi_xx, phi_yy, phi_zz.
				fIndex++;
			}

			// Add the newly populated data packet to the output array.
			dataPackets.push_back( dataPacket );
		}
	}

	// Free memory for output fields.
	for( auto& fOutput : output )
		delete [] fOutput;
	delete [] outputDepartureNumericalPhi;

	// Cleaning up.
	for( int i = N_GIVEN_INPUT_FIELDS; i < N_FIELDS; i++ )
	{
		ierr = VecDestroy( fields[i] );
		CHKERRXX( ierr );
	}

	for( auto& dir : velStar )
		delete [] dir;

	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	for( int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = VecRestoreArrayRead( vel[dim], &velReadPtr[dim] );
		CHKERRXX( ierr );
	}

	ierr = PetscLogEventEnd( log_SemiLagrangianML_collectSamples, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	return allInside;
}


void slml::SemiLagrangian::_computeMLSolution( Vec vel[], Vec *vel_xx[P4EST_DIM], const double& dt, Vec phi,
											   Vec phi_xx[P4EST_DIM], Vec hk, const double& h )
{
	PetscErrorCode ierr;

	// Some pointers to structs for the grid at time tn.  You should NOT use this function after the grid has entered
	// the loop for refining and coarsening during an advection step.
	p4est_t const *p4est_n = ngbd_n->get_p4est();
	p4est_nodes_t const *nodes_n = ngbd_n->get_nodes();

	// Invalidate the flag vector and reallocate it with Gn.
	if( _mlFlag )
	{
		ierr = VecDestroy( _mlFlag );
		CHKERRXX( ierr );
		_mlFlag = nullptr;
	}
	ierr = VecCreateGhostNodes( p4est_n, nodes_n, &_mlFlag );
	CHKERRXX( ierr );

	// Invalidate the machine learning solution phi vector.  Reallocate it using the struct of Gn.
	if( _mlPhi )
	{
		ierr = VecDestroy( _mlPhi );
		CHKERRXX( ierr );
		_mlPhi = nullptr;
	}
	ierr = VecCreateGhostNodes( p4est_n, nodes_n, &_mlPhi );
	CHKERRXX( ierr );

	// Collect data packets for locally owned nodes (no ghost nodes).  Also, skip nodes whose backtracked point falls
	// outside the domain (even if periodicity is enabled).  This makes the runtime process compatible with training.
	std::vector<DataPacket *> dataPackets;
	collectSamples( vel, vel_xx, dt, phi, phi_xx, dataPackets );

	// Go through collected nodes and add the target value.  Populate the flag vector at the same time.
	// Note: During advection and when using the neural network, we concern only about locally owned nodes at this first
	// step because curvature should have been computed using the hybrid inference system only for those vertices with
	// full stencils.  For them, we set their _mlFlag value and then scatter forward both the flag and the level-set at
	// departure point (computed with nnet).  Afterward, numerical advection is used in the rest of the nodes that are
	// not flagged (including ghost nodes), and we proceed with the refinement/coarsening process.  Notice we still need
	// to flip inferred level-set values depending on hk sign (i.e., we deal only with the negative curvature spectrum).
	const double *hkReadPtr;
	ierr = VecGetArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );

	double *_mlFlagPtr;
	ierr = VecGetArray( _mlFlag, &_mlFlagPtr );
	CHKERRXX( ierr );

	double *_mlPhiPtr;
	ierr = VecGetArray( _mlPhi, &_mlPhiPtr );
	CHKERRXX( ierr );

	for( auto dataPacket : dataPackets )
	{
		dataPacket->hk_a = hkReadPtr[dataPacket->nodeIdx];	// hk at the interface: could've been computed numerically
															// or with nnet.  It can be positive or negative.

		// Grab only locally owned nodes at maximum level of refinement and having a valid (uniform) neighborhood.
		// We can access these through the pointer _localUniformIndicesPtr.
		if( _localUniformIndicesPtr->find( dataPacket->nodeIdx ) != _localUniformIndicesPtr->end() )
		{
			_mlFlagPtr[dataPacket->nodeIdx] = p4est_n->mpirank + 1;	// Set value for valid node next to Gamma: rank+1.

			// Let's normalize data the way the nnet understands it.
			if( dataPacket->hk_a > 0 )					// Working on the negative curvature spectrum.
			{											// Flipping signs accordingly except for hk_a: that one
				dataPacket->phi_a *= -1;				// statys the way it is to fip back predictions.
				for( auto& phi_d : dataPacket->phi_d )
					phi_d *= -1;
				dataPacket->numBacktrackedPhi_d *= -1;
			}

			// Normalize semi-Lagrangian data so that -v_a has an angle in the range of [0, pi/2].
			dataPacket->rotateToFirstQuadrant();
		}
	}

	// Evaluate neural network to fix numBacktrackedPhi.
	const int N_SAMPLES_PER_PACKET = 2;
	int i;
//#pragma omp parallel for default( none ) schedule( static ) \
//		shared( N_SAMPLES_PER_PACKET, dataPackets, _nnet, _mlFlagPtr, _mlPhiPtr, h ) \
//		private( i )
	for( i = 0; i < dataPackets.size(); i++ )
	{
		std::vector<double> sample1, sample2;		// We'll give it two takes: original and reflected about y=x.
		sample1.reserve( MASS_INPUT_SIZE );
		sample2.reserve( MASS_INPUT_SIZE );
		dataPackets[i]->serialize( sample1, false, true, true, false );	// Include curvature too.
		dataPackets[i]->reflect_yEqx();
		dataPackets[i]->serialize( sample2, false, true, true, false );

		double inputs[N_SAMPLES_PER_PACKET][MASS_INPUT_SIZE];		// Populate inputs for nnet.
		double outputs[N_SAMPLES_PER_PACKET];
		for( int j = 0; j < MASS_INPUT_SIZE; j++ )
		{
			inputs[0][j] = sample1[j];
			inputs[1][j] = sample2[j];
		}

		for( auto& input : inputs )				// Reorder input: swap backtracked phi_d with hk
			std::swap( input[MASS_INPUT_SIZE-1], input[MASS_INPUT_SIZE-2] );

		_nnet->predict( inputs, outputs, N_SAMPLES_PER_PACKET );	// Gather predictions: they've been already denormalized.
		double phi_d = (outputs[0] + outputs[1]) / 2.0;				// Average predictions for a better one.

		double absRelDiff = ABS( phi_d - dataPackets[i]->numBacktrackedPhi_d ) / h;	// Sometimes the nnet fails awfully.
		if( absRelDiff > 0.1 || ABS( phi_d ) >= 2.0 * h  )							// To catch those cases, we test
		{																			// against numerical phi_d and check
			phi_d = dataPackets[i]->numBacktrackedPhi_d;							// if predicted value is >= 2H.
		}

		_mlPhiPtr[dataPackets[i]->nodeIdx] = phi_d * (dataPackets[i]->hk_a > 0? -1 : 1);	// Fix sign according to curvature.
	}

	// Restore access.
	ierr = VecRestoreArray( _mlPhi, &_mlPhiPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArray( _mlFlag, &_mlFlagPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArrayRead( hk, &hkReadPtr );
	CHKERRXX( ierr );

	// Let's synchronize the machine learning flag vector among all processes.
	ierr = VecGhostUpdateBegin( _mlFlag, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );
	VecGhostUpdateEnd( _mlFlag, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );

	// Let's synchronize the nnet phi vector among all processes.
	ierr = VecGhostUpdateBegin( _mlPhi, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );
	VecGhostUpdateEnd( _mlPhi, INSERT_VALUES, SCATTER_FORWARD );
	CHKERRXX( ierr );

	// We're done with data packets.
	freeDataPacketArray( dataPackets );
}


void slml::SemiLagrangian::updateP4EST( Vec vel[], const double& dt, Vec *phi, Vec hk, Vec *howUpdated )
{
	if( _used )				// One-time usage check.
	{
		std::string msg = "[CASL_ERROR] slml::SemiLagrangian::updateP4EST: Instance for iteration "
			+ std::to_string( ITERATION ) + " has been used already to advect the level-set function!";
		throw std::runtime_error( msg );
	}

	_used = true;			// From now on, we can't reuse this object to advect other level-set functions.

	PetscErrorCode ierr;
	ierr = PetscLogEventBegin( log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	// Some pointers to structs for the grid at time tn.
	p4est_t const *p4est_n = ngbd_n->get_p4est();
	p4est_nodes_t const *nodes_n = ngbd_n->get_nodes();

	// Retrieve grid size data and validate we are working on a domain with square cells.
	double dxyz[P4EST_DIM];
	double dxyz_min;
	get_dxyz_min( p4est_n, dxyz, dxyz_min );
	if( ABS( dxyz[0] - dxyz[1] ) > PETSC_MACHINE_EPSILON ONLY3D( || ABS( dxyz[1] - dxyz[2] ) > PETSC_MACHINE_EPSILON ) )
		throw std::runtime_error( "[CASL_ERROR] slml::SemiLagrangian::updateP4EST: Cells must be square!" );

	////////////////////// Compute second derivatives of velocity field: vel_xx, vel_yy[, vel_zz] //////////////////////
	// We need these to interpolate the velocity at updated grid Gnp1 and to retrieve the backtracked departure point
	// with second-order accuracy.
	Vec *vel_xx[P4EST_DIM];
	for( int dir = 0; dir < P4EST_DIM; dir++ )	// Choose velocity component: u, v, or w.
	{
		vel_xx[dir] = new Vec[P4EST_DIM];		// For each velocity component, we need the derivatives w.r.t. x, y, z.
		if( dir == 0 )
		{
			for( int dd = 0; dd < P4EST_DIM; dd++ )
			{
				ierr = VecCreateGhostNodes( p4est_n, nodes_n, &vel_xx[dir][dd] );
				CHKERRXX( ierr );
			}
		}
		else
		{
			for( int dd = 0; dd < P4EST_DIM; dd++ )
			{
				ierr = VecDuplicate( vel_xx[0][dd], &vel_xx[dir][dd] );
				CHKERRXX( ierr );
			}
		}
		ngbd_n->second_derivatives_central( vel[dir], DIM( vel_xx[dir][0], vel_xx[dir][1], vel_xx[dir][2] ) );
	}

	//////////////// Compute second spatial derivatives of level-set function: phi_xx, phi_yy[, phi_zz] ////////////////
	// We need these when collecting samples along the interface for computing the machine learning solution.
	Vec phi_xx[P4EST_DIM];
	for( auto & dir : phi_xx )
	{
		ierr = VecCreateGhostNodes( p4est_n, nodes_n, &dir );
		CHKERRXX( ierr );
	}
	ngbd_n->second_derivatives_central( *phi, DIM( phi_xx[0], phi_xx[1], phi_xx[2] ) );

	//////////////// Compute level-set values for points next to Gamma using the machine learning model ////////////////
	// As the grid converges below, we'll query if the values for grid points have been computed with the neural model.
	// This also serves as a cache to avoid costly nnet evaluation every time that the new grid iterates.
	// Note: the cache is computed off the grid (Gn or ngbd_n) at tn.  _mlFlag and _mlPhi should be invalided upon
	// exiting the current function (see the very last part of this procedure).
	_computeMLSolution( vel, vel_xx, dt, *phi, phi_xx, hk, dxyz_min );

	///////////////////////////////////////// Preparing coarse-refine process //////////////////////////////////////////

	// Save the old splitting criteria information.
	auto *oldSplittingCriteria = (splitting_criteria_t*) p4est->user_pointer;

	// Define splitting criteria with an explicit band of min diags around Gamma.
	auto splittingCriteriaBandPtr = new splitting_criteria_band_t( oldSplittingCriteria->min_lvl,
																   oldSplittingCriteria->max_lvl,
																   oldSplittingCriteria->lip, BAND );

	// New grid level-set values: start from current grid values.
	Vec phi_np1;
	ierr = VecCreateGhostNodes( p4est, nodes, &phi_np1 );	// Notice p4est and nodes are the grid at time n so far.
	CHKERRXX( ierr );

	// See how nodes were updated; later, this is used for selective reinitialization.
	Vec howUpdated_np1;
	ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated_np1 );
	CHKERRXX( ierr );

	/////////////////////////////////////////// Update grid until convergence //////////////////////////////////////////
	// Main loop in Algorithm 3 in reference [*].
	bool isGridChanging = true;
	int counter = 0;
	while( isGridChanging )
	{
		ierr = PetscLogEventBegin( log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0 );
		CHKERRXX( ierr );

		// Advect from Gn to Gnp1 to enable refinement.
		double *phi_np1Ptr;
		ierr = VecGetArray( phi_np1, &phi_np1Ptr );
		CHKERRXX( ierr );

		double *howUpdated_np1Ptr;
		ierr = VecGetArray( howUpdated_np1, &howUpdated_np1Ptr );
		CHKERRXX( ierr );

		// Perform advection with second-order semi-Lagrangian step.
		_advectFromNToNp1( dt, dxyz_min, vel, vel_xx, *phi, phi_xx, phi_np1Ptr, howUpdated_np1Ptr );

		// Refine an coarsen grid; detect if it changes from previous coarsening/refinement operation.
		isGridChanging = splittingCriteriaBandPtr->refine_and_coarsen_with_band( p4est, nodes, phi_np1Ptr );

		ierr = VecRestoreArray( howUpdated_np1, &howUpdated_np1Ptr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( phi_np1, &phi_np1Ptr );
		CHKERRXX( ierr );

		if( isGridChanging )
		{
			// Repartition grid as it changed.
			my_p4est_partition( p4est, P4EST_TRUE, nullptr );

			// Reset nodes, ghost, and phi.
			p4est_ghost_destroy( ghost );
			ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
			p4est_nodes_destroy( nodes );
			nodes = my_p4est_nodes_new( p4est, ghost );

			// Allocate vector for new level-set values for most up-to-date grid.
			ierr = VecDestroy( phi_np1 );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &phi_np1 );
			CHKERRXX( ierr );

			// Reallocate the how-updated debugging vector.
			ierr = VecDestroy( howUpdated_np1 );
			CHKERRXX( ierr );
			ierr = VecCreateGhostNodes( p4est, nodes, &howUpdated_np1 );
			CHKERRXX( ierr );
		}

		ierr = PetscLogEventEnd( log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0 );
		CHKERRXX( ierr );
		counter++;
	}

	p4est->user_pointer = (void *) oldSplittingCriteria;
	*p_p4est = p4est;	// I still don't understand the use of these pointers if they are never returned to caller.
	*p_nodes = nodes;
	*p_ghost = ghost;

	// Finishing up.
	ierr = VecDestroy( *phi );
	CHKERRXX( ierr );
	*phi = phi_np1;

	if( howUpdated )
	{
		ierr = VecDestroy( *howUpdated );
		CHKERRXX( ierr );
		*howUpdated = howUpdated_np1;
	}
	else		// If caller doesn't want the howUpdated flag, just destroy it the one we computed here.
	{
		ierr = VecDestroy( howUpdated_np1 );
		CHKERRXX( ierr );
	}

	for( auto& dir : phi_xx )
	{
		ierr = VecDestroy( dir );
		CHKERRXX( ierr );
	}

	for( auto& dir : vel_xx )
	{
		for( unsigned char dd = 0; dd < P4EST_DIM; ++dd )
		{
			ierr = VecDestroy( dir[dd] );
			CHKERRXX( ierr );
		}
		delete[] dir;
	}

	ierr = PetscLogEventEnd( log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	delete splittingCriteriaBandPtr;

	// Invalidating and clearing internal cache parallel vectors.
	ierr = VecDestroy( _mlFlag );
	CHKERRXX( ierr );
	_mlFlag = nullptr;

	ierr = VecDestroy( _mlPhi );
	CHKERRXX( ierr );
	_mlPhi = nullptr;
}


void slml::SemiLagrangian::_advectFromNToNp1( const double& dt, const double& h, Vec vel[P4EST_DIM],
											  Vec *vel_xx[P4EST_DIM], Vec phi, Vec phi_xx[P4EST_DIM],
											  double *phi_np1Ptr, double *howUpdated_np1Ptr )
{
	PetscErrorCode ierr;
	ierr = PetscLogEventBegin( log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	my_p4est_interpolation_nodes_t interp( ngbd_n );			// These node neighborhoods are the ones that preserve
	my_p4est_interpolation_nodes_t interp_phi( ngbd_phi );		// the grid structure at time n.

	double *interpOutput[P4EST_DIM];							// Where interpolated velocity at time tnp1 is stored.

	Vec xx_v_derivatives[P4EST_DIM] = {DIM( vel_xx[0][0], vel_xx[1][0], vel_xx[2][0] )};	// Reorganize velocity derivatives
	Vec yy_v_derivatives[P4EST_DIM] = {DIM( vel_xx[0][1], vel_xx[1][1], vel_xx[2][1] )};	// by derivative direction.
#ifdef P4_TO_P8
	Vec zz_v_derivatives[P4EST_DIM] = {vel_xx[0][2], vel_xx[1][2], vel_xx[2][2]};
#endif

	// Domain features.
	const double *xyz_min = get_xyz_min();
	const double *xyz_max = get_xyz_max();
	const bool *periodicity = get_periodicity();

	///////////////////////////////////////////// Finding velocity at Gnp1 /////////////////////////////////////////////
	// Using quadratic interpolation to find unp1 from Gn.
	std::vector<double> vel_np1[P4EST_DIM];
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )	// Notice nodes struct changes with advection.
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est, nodes, xyz );
		interp.add_point( n, xyz );										// Defining points Xnp1 where we need vnp1.
	}

	for( int dir = 0; dir < P4EST_DIM; dir++ )
	{
		vel_np1[dir].resize( nodes->indep_nodes.elem_count );
		interpOutput[dir] = vel_np1[dir].data();
	}
	interp.set_input( vel, DIM( xx_v_derivatives, yy_v_derivatives, zz_v_derivatives ), VEL_INTERP_MTHD, P4EST_DIM );
	interp.interpolate( interpOutput );				// Interpolate velocities.  Save these in vel_np1 vector.
	interp.clear();

	//////////////////////////////////////// Finding midpoint velocity at Gnp1 /////////////////////////////////////////
	// Used for secon-order accurate location of departure points for Gnp1.
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
	{
		double xyz_star[P4EST_DIM];				// Midway point.
		node_xyz_fr_n( n, p4est, nodes, xyz_star );
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			xyz_star[dir] -= 0.5 * dt * vel_np1[dir][n];
		clip_in_domain( xyz_star, xyz_min, xyz_max, periodicity );

		interp.add_point(n, xyz_star);
	}

	for( int dir = 0; dir < P4EST_DIM; dir++ )		// Replace the vel_np1 by the midpoint velocity which we use below
		interpOutput[dir] = vel_np1[dir].data();	// for backtracking points.
	interp.set_input( vel, DIM(xx_v_derivatives, yy_v_derivatives, zz_v_derivatives), VEL_INTERP_MTHD, P4EST_DIM );
	interp.interpolate( interpOutput );
	interp.clear();

	/////////////////////////////////////////////// Find departure points //////////////////////////////////////////////
	// Using the user-defined interpolation for phi.  Make sure the process is compatible with nnet training.
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )	// Notice nodes struct changes with advection.
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est, nodes, xyz );
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			xyz[dir] -= dt * vel_np1[dir][n];
		clip_in_domain_with_check( xyz, xyz_min, xyz_max, periodicity );	// TODO: Consider what happens if point is invalid.

		interp_phi.add_point( n, xyz );
		howUpdated_np1Ptr[n] = HowUpdated::NUM;
	}
	interp_phi.set_input( phi, DIM( phi_xx[0], phi_xx[1], phi_xx[2]), PHI_INTERP_MTHD );
	interp_phi.interpolate( phi_np1Ptr );			// New phi values at time tnp1 based off Gn.

	////////// Load cached values computed with neural network for points in a band around new Gamma location //////////
	Cache cache( ngbd_n );							// Fake cache using interpolation infraestructure.
	std::vector<p4est_locidx_t> outIdxToNodeIdx;	// This is a map from out index to actual node index in PETSc Vecs.
	outIdxToNodeIdx.reserve( nodes->indep_nodes.elem_count );
	int outIdx = 0;
	for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
	{
		if( ABS( phi_np1Ptr[n] ) <= 2.0 * M_SQRT2 * h )	// This band ensures we see which points were updated with nnet.
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			cache.add_point( outIdx, xyz );
			outIdxToNodeIdx.push_back( n );			// Keep track of correspondence between node index and output index.
			outIdx++;
		}
	}

	// Note: Must perform interpolation even if outIdx is zero (i.e., no independent nodes next to Gamma for this
	// process).  Not doing so, will enter processes into deadlock, waiting for each other to finish their interp. task.
	// Allocate output vectors with as many elements as valid indices we collected in previous loop.  Set up cache
	// fetcher inputs and launch data collection.
	Vec fields[Cache::N_FIELDS] = {_mlPhi, _mlFlag};	// Input fields we built with the neural network.
	double *output[Cache::N_FIELDS];
	for( auto& fOutput : output )
		fOutput = new double[outIdx];
	cache.setInput( fields, Cache::N_FIELDS );
	cache.interpolate( output );						// Read cache for points along band of new Gamma location.
	cache.clear();

	// Process requests and match them with cache.
	for( int i = 0; i < outIdx; i++ )
	{
		p4est_locidx_t nodeIdx = outIdxToNodeIdx[i];	// Actual node index in PETSc vector.
		if( output[Cache::Fields::FLAG][i] > 0 )		// Computed with neural network?  Flagged rank + 1.
		{
			phi_np1Ptr[nodeIdx] = output[Cache::Fields::PHI][i];
			howUpdated_np1Ptr[nodeIdx] = HowUpdated::NNET;
		}
		else											// Point within band but not computed with nnet.
			howUpdated_np1Ptr[nodeIdx] = HowUpdated::NUM_BAND;
	}

	// Free memory for output fields.
	for( auto& fOutput : output )
		delete[] fOutput;

	ierr = PetscLogEventEnd( log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0 );
	CHKERRXX( ierr );
}
