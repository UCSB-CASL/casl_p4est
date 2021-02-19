#include "my_p4est_semi_lagrangian_ml.h"

///////////////////////////////////////////////////// DataFetcher //////////////////////////////////////////////////////

slml::DataFetcher::DataFetcher( const my_p4est_node_neighbors_t *ngbd )
								: my_p4est_interpolation_t( ngbd ), _nodes(ngbd_n->get_nodes()){}


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

	// Next, retrieve the field values at the 4 (8) corners of the quad (oct).  We have 1 + P4EST_DIM input fields.
	// Also, recall that children (e.g., nodes in a quad) appear as zyx (z being the slowest changing coordinate).
	// We need them the other way around in the output: xyz (x being the slowest changing).
	for( int fieldIdx = InputFields::PHI; fieldIdx != InputFields::LAST; fieldIdx++ )
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
	throw std::runtime_error( "[CASL_ERROR]: slml::DataFetcher::operator(): Not implemented yet!" );
}


//////////////////////////////////////////////////// SemiLagrangian ////////////////////////////////////////////////////

slml::SemiLagrangian::SemiLagrangian( p4est_t **p4estNp1, p4est_nodes_t **nodesNp1, p4est_ghost_t **ghostNp1,
									  my_p4est_node_neighbors_t *ngbdN )
									  : my_p4est_semi_lagrangian_t( p4estNp1, nodesNp1, ghostNp1, ngbdN ){}


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


void slml::SemiLagrangian::collectSamples( Vec vel[P4EST_DIM], double dt, Vec phi,
										   std::vector<DataPacket *>& dataPackets ) const
{
	PetscErrorCode ierr;
	ierr = PetscLogEventBegin( log_SemiLagrangianML_collectSamples, 0, 0, 0, 0 );
	CHKERRXX( ierr );

	// Initialize output array.
	freeDataPacketArray( dataPackets );		// Just in case, free and clear whatever is left in output array.
	dataPackets.reserve( nodes->indep_nodes.elem_count );

	// We need the indices for nodes next to the interface (i.e., one of their irradiating edges is crossed by Gamma).
	// Since we don't care about the full stencils of nodes, we use the alternative method from NodesAlongInterface to
	// retrieve all independent nodes (local and ghost) that the partition is aware of.
	auto* p4estUserData = (splitting_criteria_t*)p4est->user_pointer;
	NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd_phi, (signed char)p4estUserData->max_lvl );
	std::unordered_set<p4est_locidx_t> indices;
	nodesAlongInterface.getIndepIndices( &phi, ghost, indices );

	// Domain features.
	const double *xyz_min = get_xyz_min();
	const double *xyz_max = get_xyz_max();
	const bool *periodicity = get_periodicity();

	// Retrieve grid size data.
	double dxyz[P4EST_DIM];
	double dxyz_min;			// Minimum cell width for current macromesh.  Use this to normalize distances below.
	get_dxyz_min( p4est, dxyz, dxyz_min );

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

	// Initialize infrastructure.  We'll use the interpolation across processes infraestructure and hack the interpolate
	// method.  To do this, we need to instantiate dummy vectors that will pretend to work as empty "fields".  That way,
	// upon return, the multi-process interpolation will send back the data for the four (eight) quad (oct) corners:
	// In 2D: exit_code + normalized(x_d) + 4 phi values + 4*2 velocity components = 15 values per "interpolation".
	// In 3D: exit_code + normalized(x_d) + 8 phi values + 8*3 velocity components = 36 values per "interpolation".
	// When setting up the input fields (in that order):
	// In 2D: phi + vel_u + vel_v [+ 12 dummy fields].
	// In 3D: phi + vel_u + vel_v + vel_w [+ 32 dummy fields].
	//
	// The exit_code above is used to determined if data collection finished successfully.  A possible error situation
	// arises, for example, if the quadrant owning the query point is not at the maximum level of refinement.  We need
	// to collect data from uniform quadrants at the maximum level of refinement only for training consistency.
	DataFetcher dataFetcher( ngbd_n );

	const int N_FIELDS = 1 + P4EST_DIM + (1 + P4EST_DIM) * P4EST_CHILDREN;	// Number of fields to receive.
	const int N_GIVEN_INPUT_FIELDS = 1 + P4EST_DIM;				// Actual number of true fields to send (phi and vel).

	// Collect nodes along the interface and add their backtraced departure points to "interpolation" buffer.
	// Filter out any node whose backtraced departure point fall outside of domain (if no periodicity is allowed in that
	// direction).
	double xyz_d[P4EST_DIM];						// Departure point.
	double xyz_a[P4EST_DIM];						// Arrival point.
	std::vector<p4est_locidx_t> outIdxToNodeIdx;	// This is a map from out index to actual node index in PETSc Vecs.
	outIdxToNodeIdx.reserve( indices.size() );
	std::vector<double> distances;					// Vector to store the distance between arrival and departure points.
	distances.reserve( indices.size() );
	int outIdx = 0;									// Index in "interpolation" output.
	for( const auto nodeIdx : indices )
	{
		// Backtracing the point using one step in the negative velocity direction.
		node_xyz_fr_n( nodeIdx, p4est, nodes, xyz_a );
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			xyz_d[dir] = xyz_a[dir] - dt * velReadPtr[dir][nodeIdx];

		// Check if node falls within the domain (or if it's correctly circled due to periodicity in any direction).
		// We don't admit truncated backtraced points to avoid inconsistency in the training patterns.
		if( !clip_in_domain_with_check( xyz_d, xyz_min, xyz_max, periodicity ) )
			continue;

		// Euclidean distance between arrival and backtraced departure point.
		double d = 0;
		for( int dir = 0; dir < P4EST_DIM; dir++ )
			d += SQR( xyz_d[dir] - xyz_a[dir] );
		d = sqrt( d );

		// Add departure point to buffer and to vector map to node indices.
		dataFetcher.add_point( outIdx, xyz_d );
		outIdxToNodeIdx.push_back( nodeIdx );

		// Add the normalized distance from x_d to x_a.
		distances.push_back( d / dxyz_min );

		outIdx++;
	}

	// Do we actually have points for which to collect data?  outIdx is also equal to total number of valid nodes along
	// the interface.
	if( outIdx > 0 )
	{
		Vec fields[N_FIELDS] = {phi, DIM(vel[0], vel[1], vel[2])};	// Input fields.
		for( int i = N_GIVEN_INPUT_FIELDS; i < N_FIELDS; i++ )		// Dummy fields are initialized with zeros.
		{
			ierr = VecCreateGhostNodes( ngbd_phi->get_p4est(), ngbd_phi->get_nodes(), &fields[i] );
			CHKERRXX( ierr );
		}

		// Allocate output vectors with as many elements as valid indices we collected in previous loop.  Set up fetch
		// inputs and launch data collection.
		double *output[N_FIELDS];
		for( auto& fOutput : output )
			fOutput = new double[outIdx];
		dataFetcher.setInput( fields, N_FIELDS );
		dataFetcher.interpolate( output );
		dataFetcher.clear();

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
				dataPacket->phi_a = phiReadPtr[nodeIdx];		// Phi and velocity at arrival point.
				for( int dim = 0; dim < P4EST_DIM; dim++ )
					dataPacket->vel_a[dim] = velReadPtr[dim][nodeIdx];
				dataPacket->distance = distances[i];			// Normalized distance from departure to arrival point.

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

				// Add the newly populated data packet to the output array.
				dataPackets.push_back( dataPacket );
			}
		}

		// Free memory for output fields.
		for( auto& fOutput : output )
			delete[] fOutput;

		// Cleaning up, part 1.
		for( int i = N_GIVEN_INPUT_FIELDS; i < N_FIELDS; i++ )
		{
			ierr = VecDestroy( fields[i] );
			CHKERRXX( ierr );
		}
	}

	// Cleaning up, part 2.
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	for( int dim = 0; dim < P4EST_DIM; dim++ )
	{
		ierr = VecRestoreArrayRead( vel[dim], &velReadPtr[dim] );
		CHKERRXX( ierr );
	}

	ierr = PetscLogEventEnd( log_SemiLagrangianML_collectSamples, 0, 0, 0, 0 );
	CHKERRXX( ierr );
}
