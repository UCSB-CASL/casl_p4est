//
// Created by Im YoungMin on 4/21/20.
//

#include "my_p4est_fast_sweeping.h"

/////////////////////////////////////////////// Private member functions ///////////////////////////////////////////////

void FastSweeping::_clearScaffoldingData()
{
	if( _orderings )			// Free memory if there are orderings previously loaded.
	{
		for( size_t i = 0; i < N_ORDERINGS; i++ )
			delete [] _orderings[i];
		delete [] _orderings;
	}

	// Reset critical variables.
	_orderings = nullptr;
}

void FastSweeping::_clearSolutionData()
{
	delete [] _uOld;			// Free memory if there were any previously stored old solution values,
	delete [] _uCpy;			// if we copied any input solution values,
	delete [] _seedStates;		// or if we stored the nodes viability as seed nodes.

	// Reset critical variables.
	_u = nullptr;
	_uPtr = nullptr;
	_uOld = nullptr;
	_uCpy = nullptr;
	_rhs = nullptr;
	_rhsPtr = nullptr;
	_seedStates = nullptr;
}

FastSweeping::SEED_STATE FastSweeping::_checkSeedState( p4est_locidx_t n )
{
	// If we already determined that the nth node is a viable/non-viable seed point, just return that.
	if( _seedStates[n] == SEED_STATE::UNDEFINED )
	{
		// At this point, the seed state is UNDEFINED: we need to verify whether the point is a viable seed node or not by
		// checking if it lies exactly on the interface or at least one of its outgoing edges is crossed by \Gamma.
		if( _neighbors->has_valid_qnnn( n ) )
		{
			if( ABS( _uCpy[n] ) <= _zeroDistanceThreshold )		// Point lies in on the interface.  It's then a valid
				_seedStates[n] = SEED_STATE::VALID;				// seed node.
			else
			{
				const quad_neighbor_nodes_of_node_t *qnnnPtr;	// Evaluate neighborhood.
				_neighbors->get_neighbors( n, qnnnPtr );

				double data[P4EST_DIM][2][2];
				_getStencil( qnnnPtr, _uCpy, data );			// Retrieve the 2 or 3 dimensional stencil.
				const short F = 0, S = 1;						// Meaning: function index, distance index.
				bool crossed = false;

				for( const auto& dim : data )					// Check irradiating edges in each dimension.
				{
					for( const auto& dir : dim )				// And in each direction: left/bottom/back - right/top/front
					{
						if( dir[S] < 0 )						// Is current node on a wall?
							continue;
						if( dir[F] * _uCpy[n] <= 0 )			// Crossed by interface?
						{
							crossed = true;
							break;
						}
					}

					if( crossed )								// Just find a single edge crossed by \Gamma.
						break;
				}

				if( crossed )
					_seedStates[n] = SEED_STATE::VALID;
				else
					_seedStates[n] = SEED_STATE::INVALID;
			}
		}
		else
		{
			_seedStates[n] = SEED_STATE::INVALID;		// Point is a ghost node with no well defined neighborhood.
		}
	}

	return _seedStates[n];
}

void FastSweeping::_defineHamiltonianConstants( const quad_neighbor_nodes_of_node_t *qnnnPtr, double a[], double h[] )
{
	// Some convenient arrangement nodal solution values and their distances w.r.t. center node in its stencil of neighbors.
	double data[P4EST_DIM][2][2];
	_getStencil( qnnnPtr, _uPtr, data );

	for( size_t i = 0; i < P4EST_DIM; i++ )		// Choose a and h for x, y [and z].
	{
		const double *m = data[i][0], *p = data[i][1];

		if( m[1] < 0.0 )						// On the left/bottom/back wall?
		{
			a[i] = p[0];
			h[i] = p[1];
		}
		else if( p[1] < 0 )						// On the right/top/front wall?
		{
			a[i] = m[0];
			h[i] = m[1];
		}
		else									// Determining from both neighbors.
		{
			if( m[0] < p[0] )					// Choose the minimum according to [1].
			{
				a[i] = m[0];
				h[i] = m[1];
			}
			else
			{
				a[i] = p[0];
				h[i] = p[1];
			}
		}
		a[i] = MIN( a[i], PETSC_INFINITY );		// Shouldn't be larger than local infinity.
	}
}

void FastSweeping::_sortHamiltonianConstants( double a[], double h[] )
{
	double auxA, auxH;

#ifdef P4_TO_P8
	// Bubble sort, but only for up to arrays of 3 elements (e.g. in 3D).
	for( size_t i = 0; i < P4EST_DIM - 1; i++ )
	{
		for( size_t j = 0; j < P4EST_DIM - 1 - i; j++ )
		{
			if( a[j] > a[j+1] )					// Swap?
			{
				auxA = a[j]; auxH = h[j];
				a[j] = a[j+1]; h[j] = h[j+1];
				a[j+1] = auxA; h[j+1] = auxH;
			}
		}
	}
#else
	if( a[0] > a[1] )					// Swap?
	{
		auxA = a[0]; auxH = h[0];
		a[0] = a[1]; h[0] = h[1];
		a[1] = auxA; h[1] = auxH;
	}
#endif

	// Finally, set a_dim = infinity.
	a[P4EST_DIM] = PETSC_INFINITY;
	h[P4EST_DIM] = -1;					// The h value is irrelevant.
}

void FastSweeping::_getStencil( const quad_neighbor_nodes_of_node_t *qnnnPtr, const double *f, double data[P4EST_DIM][2][2] )
{
	// Some convenient arrangement nodal solution values and their distances w.r.t. center node in its stencil of neighbors.
	data[0][0][0] = qnnnPtr->f_m00_linear( f ); data[0][0][1] = qnnnPtr->d_m00;			// Left.
	data[0][1][0] =	qnnnPtr->f_p00_linear( f ); data[0][1][1] = qnnnPtr->d_p00;			// Right.

	data[1][0][0] = qnnnPtr->f_0m0_linear( f ); data[1][0][1] = qnnnPtr->d_0m0;			// Bottom.
	data[1][1][0] = qnnnPtr->f_0p0_linear( f ); data[1][1][1] = qnnnPtr->d_0p0;			// Top.
#ifdef P4_TO_P8
	data[2][0][0] = qnnnPtr->f_00m_linear( f ); data[2][0][1] = qnnnPtr->d_00m;			// Back.
	data[2][1][0] = qnnnPtr->f_00p_linear( f ); data[2][1][1] = qnnnPtr->d_00p;			// Front.
#endif
}

double FastSweeping::_computeNewUAtNode( p4est_locidx_t n )
{
	if( _rhsPtr[n] >= PETSC_INFINITY )	// Is this an adjacent node to the interface (i.e. non-updatable)?
		return _uPtr[n];

	// Dealing with nodes off the interface.  We solve the Hamiltonian of the Eikonal equation given by:
	//      | (u(x) - a_0)+ |2          | (u(x) - a_2)+ |2
	//      | ------------- |  + ... +  | ------------- |  = f^2,   for i = 0, 1 in 2D, and i = 0, 1, 2 in 3D.
	//      |      h_0      |           |       h_2     |
	//
	// In our case, f^2 = 1 for updatable nodes and infinite for fixed nodes.
	// a_i = u_{x_i min} = min( u_{x - h_{i-}}, u{x + h_{i+}} ), for the ith dimension.  See [1].
	// Now, access the neighborhood of requested node.  Assume that the neighborhood is well defined.
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	_neighbors->get_neighbors( n, qnnnPtr );

	// Determining a_i and h_i constants for Hamiltonian in above equation.
	double a[P4EST_DIM + 1];			// The additional space is reserved for an infinite value used for breaking the
	double h[P4EST_DIM + 1];			// loop below.
	_defineHamiltonianConstants( qnnnPtr, a, h );

	// If node n is surrounded by local infinite values, it then should be given a local infinite value.
	if( MIN( DIM( a[0], a[1], a[2] ) ) >= PETSC_INFINITY )
		return PETSC_INFINITY;

	// Solve for u(x) (i.e. new approximation to u) in the equation above using the method in [1] and [2].
	// 1) Sort a_i from least to greatest (h_i will not be in any particular order).
	_sortHamiltonianConstants( a, h );

	// 2) Begin solution.
	double uHat, numerator, h0Sqrd, h1Sqrd, h2Sqrd, hSqrdSum;
	double fSqrd = SQR( _rhsPtr[n] );
	for( size_t m = 0; m < P4EST_DIM; m++ )
	{
		// 3) Solving \sum_{j=0}^{m} ( (u - a_j)^+ / h_j )^2 = f^2 for u.
		if( m == 0 )
		{
			uHat = a[0] + h[0] * _rhsPtr[n];
		}
		else if( m == 1 )
		{
			h0Sqrd = SQR( h[0] );
			h1Sqrd = SQR( h[1] );
			hSqrdSum = h0Sqrd + h1Sqrd;
			numerator = ( a[0] * h1Sqrd + a[1] * h0Sqrd ) + h[0] * h[1] * sqrt( fSqrd * hSqrdSum - SQR( a[0] - a[1] ) );
			uHat = numerator / hSqrdSum;
		}
		else if( m == 2 )					// Only for 3D.
		{
			h0Sqrd = SQR( h[0] );
			h1Sqrd = SQR( h[1] );
			h2Sqrd = SQR( h[2] );
			hSqrdSum = h1Sqrd * h2Sqrd + h0Sqrd * h2Sqrd + h0Sqrd * h1Sqrd;
			numerator = a[0] * h1Sqrd * h2Sqrd + a[1] * h0Sqrd * h2Sqrd + a[2] * h0Sqrd * h1Sqrd;
			numerator += h[0] * h[1] * h[2] * sqrt( fSqrd * hSqrdSum - h2Sqrd * SQR( a[0] - a[1] ) - h1Sqrd * SQR( a[0] - a[2] ) - h0Sqrd * SQR( a[1] - a[2] ) );
			uHat = numerator / hSqrdSum;
		}

		// 4) If uHat <= a_{m+1}, we are done and u(x) = uHat.
		if( uHat <= a[m+1] )
			break;
	}

	return uHat;
}

#ifdef P4_TO_P8
void FastSweeping::_fillQuadOctValuesFromNodeSampledVector( OctValueExtended *quadValPtr, const p4est_locidx_t& quadIdx, const double *nodeSampledValuesPtr )
#else
void FastSweeping::_fillQuadOctValuesFromNodeSampledVector( QuadValueExtended *quadValPtr, const p4est_locidx_t& quadIdx, const double *nodeSampledValuesPtr )
#endif
{
	// Load nodal function values and their respective indices.
	const p4est_locidx_t *q2n = _nodes->local_nodes + P4EST_CHILDREN * quadIdx;
	for( unsigned char incX = 0; incX < 2; incX++ )
		for( unsigned char incY = 0; incY < 2; incY++ )
#ifdef P4_TO_P8
			for( unsigned char incZ = 0; incZ < 2; incZ++ )
#endif
		{
			unsigned idxInQuadOctValue = SUMD( ( 1u << (unsigned)( P4EST_DIM - 1 ) ) * incX, ( 1u << (unsigned)( P4EST_DIM - 2 ) ) * incY, incZ );
			p4est_locidx_t nodeSubIdxInQuad = q2n[SUMD( incX, 2 * incY, 4 * incZ )];
			quadValPtr->val[idxInQuadOctValue] = nodeSampledValuesPtr[nodeSubIdxInQuad];	// Store nodal value and their
			quadValPtr->indices[idxInQuadOctValue] = nodeSubIdxInQuad;						// indices.
		}
}

void FastSweeping::_processQuadOct( const p4est_quadrant_t *quad, p4est_locidx_t quadIdx )
{
#ifdef P4_TO_P8
	OctValueExtended  phiAndIdxQuadOctValues;
#else
	QuadValueExtended phiAndIdxQuadOctValues;
#endif

	// Populate a quad/oct struct with node values and corresponding node indices belonging to each of them.
	_fillQuadOctValuesFromNodeSampledVector( &phiAndIdxQuadOctValues, quadIdx, _uCpy );

	p4est_topidx_t vm = _p4est->connectivity->tree_to_vertex[0];
	p4est_topidx_t vp = _p4est->connectivity->tree_to_vertex[P4EST_CHILDREN-1];
	const double* tree_xyz_min = _p4est->connectivity->vertices + 3 * vm;
	const double* tree_xyz_max = _p4est->connectivity->vertices + 3 * vp;
	double dmin = (double)P4EST_QUADRANT_LEN( quad->level ) / (double)P4EST_ROOT_LEN;	// Side length of current cell.

	// Distance in each Cartesian direction, assuming we start at (0, 0[, 0]) and end at (dx, dy[, dz]).
	double dx = dmin * ( tree_xyz_max[0] - tree_xyz_min[0] );
	double dy = dmin * ( tree_xyz_max[1] - tree_xyz_min[1] );
#ifdef P4_TO_P8
	double dz = dmin * ( tree_xyz_max[2] - tree_xyz_min[2] );
	Cube3 cell( 0, dx, 0, dy, 0, dz );
#else
	Cube2 cell( 0, dx, 0, dy );
#endif

	// Now, approximate interface (if any) within current quad/oct, and compute initial distance to its nodes.
	// We do this using the methodology explained in ref [5].
	std::unordered_map<p4est_locidx_t, double> distanceMap;
	cell.computeDistanceToInterface( phiAndIdxQuadOctValues, distanceMap, _zeroDistanceThreshold );
	for( const auto& pair : distanceMap )
	{
		// Determine viability of seed node if it has a valid neighborhood and its outgoing edges are crossed by \Gamma.
		// Note that we must not check for *all* nodes in the current partition; only for those whose quad/oct has at
		// least one corner on \Gamma, or if the quad/oct is cut-out by the interface, which at this point has been
		// already accounted for by ignoring distanceMaps that are empty.
		if( _checkSeedState( pair.first ) == SEED_STATE::VALID )
		{
			_uPtr[pair.first] = MIN( pair.second, _uPtr[pair.first] );		// Define seed point by keeping the minimum distance.
			_rhsPtr[pair.first] = PETSC_INFINITY;							// A seed point is *not* updatable.
		}
	}
}

void FastSweeping::_approximateInterfaceAndSeedNodes()
{
	// Given two nodes, n1 and n2, connected by an edge e, the interface is located between n1 and n2 iff phi(n1) * phi(n2) <= 0.
	// Furthermore, if |phi(ni)| <= scaledEPS, then ni lies on the interface.
	// To approximate the interface location, we rely on the quads/octants that make up the trees in the p4est struct.
	// Thus, we check which quads/octs are cut by the interface and approximate the interface so that we can compute the
	// distance to the (seed) nodes that belong to that cut-out cell.  We base this computations on simplices in 2 and 3D.

	// Attention!  We must use a copy of the original nodal values or else we'll be affecting the seed determination as
	// we progress in our initialization of u.
	// Notice too that we need to consider the ghost quads too.  At the end we must *gather* from foreign ghosts to locally
	// owned nodes using the MIN operation, and then we must *scatter* forward from local to foreign nodes.

	// Use the u copy to initialize the solution pointer _uPtr.  Start with all nodes being infinitely far and updatable.
	// Also begin with undefined seed state for nodes.
	for( p4est_locidx_t n = 0; n < _nodes->num_owned_indeps; n++ )
	{
		_uPtr[n] = PETSC_INFINITY;
		_rhsPtr[n] = 1;
		_seedStates[n] = SEED_STATE::UNDEFINED;		// Not yet known if the node will be used as seed point or not.
	}

	// Go through each quad/octant and check if it's crossed by the interface.  If so, update its nodes that can be used
	// as seed points by approximating their shortest distance to the interface using a piece-wise linear reconstruction.
	// Since a node may belong to several quads/octs, we keep the minimum as long as it is a valid seed point.
	for(p4est_topidx_t treeIdx = _p4est->first_local_tree; treeIdx <= _p4est->last_local_tree; treeIdx++ )
	{
		auto *tree = (p4est_tree_t*)sc_array_index( _p4est->trees, treeIdx );			// Check all local trees.
		for( size_t quadIdx = 0; quadIdx < tree->quadrants.elem_count; quadIdx++ )		// Check each quadrant in local trees.
		{
			auto *quad = (const p4est_quadrant_t*)sc_array_index( &tree->quadrants, quadIdx );
			_processQuadOct( quad, quadIdx + tree->quadrants_offset );
		}
	}

	// Gather and scatter seed nodes and init state across processes.
	PetscErrorCode ierr;
	if( _p4est->mpisize > 1 )
	{
		ierr = VecGhostUpdateBegin( *_u, MIN_VALUES, SCATTER_REVERSE );			// Gather minimum value for u from foreign ghost nodes.
		CHKERRXX( ierr );
		ierr = VecGhostUpdateEnd( *_u, MIN_VALUES, SCATTER_REVERSE );
		CHKERRXX( ierr );
		ierr = VecGhostUpdateBegin( _rhs, MAX_VALUES, SCATTER_REVERSE );		// Gather maximum value for rhs from foreign ghost nodes.
		CHKERRXX( ierr );
		ierr = VecGhostUpdateEnd( _rhs, MAX_VALUES, SCATTER_REVERSE );
		CHKERRXX( ierr );

		ierr = VecGhostUpdateBegin( *_u, INSERT_VALUES, SCATTER_FORWARD );		// Scatter seed nodes solution onto ghost nodes.
		CHKERRXX( ierr );
		VecGhostUpdateEnd( *_u, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
		ierr = VecGhostUpdateBegin( _rhs, INSERT_VALUES, SCATTER_FORWARD );		// Scatter inverse speed onto ghost nodes.
		CHKERRXX( ierr );
		VecGhostUpdateEnd( _rhs, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
	}
}

void FastSweeping::_fixSolutionSign()
{
	// Take care of the negative solution reinitialized values only on the locally owned nodes.
	size_t updateCount = 0;
	for( p4est_locidx_t i = 0; i < _nodes->num_owned_indeps; i++ )
	{
		if( _uCpy[i] < -EPS )
		{
			_uPtr[i] *= -1;
			updateCount++;
		}
	}

	// Scatter sign-corrected updated solution onto foreign ghost nodes only if there were any updates.
	if( updateCount )
	{
		PetscErrorCode ierr = VecGhostUpdateBegin( *_u, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
		ierr = VecGhostUpdateEnd(*_u, INSERT_VALUES, SCATTER_FORWARD );
		CHKERRXX( ierr );
	}
}

/////////////////////////////////////////////// Public member functions ////////////////////////////////////////////////

FastSweeping::FastSweeping(): N_ORDERINGS( static_cast<size_t>( pow( 2, P4EST_DIM ) ) ){}

FastSweeping::~FastSweeping()
{
	_clearScaffoldingData();
	_clearSolutionData();
}

void FastSweeping::prepare( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *neighbors,
	const double xyzMin[], const double xyzMax[] )
{
	// Start afresh; setting pointers to internal pointers.
	_clearScaffoldingData();
	_p4est = p4est;
	_nodes = nodes;
	_neighbors = neighbors;

	// Establish the zero distance threshold as a scaled version of EPS that depends on the local domain resolution.
	_zeroDistanceThreshold = EPS * MIN( DIM( ( _neighbors->myb->xyz_max[0] - _neighbors->myb->xyz_min[0] ) / _neighbors->myb->nxyztrees[0],
			( _neighbors->myb->xyz_max[1] - _neighbors->myb->xyz_min[1] ) / _neighbors->myb->nxyztrees[1],
			( _neighbors->myb->xyz_max[2] - _neighbors->myb->xyz_min[2] ) / _neighbors->myb->nxyztrees[2] ) );

	// Determine the sweep orderings by picking non-colinear reference points and sorting locally owned indep nodes with
	// respect to those points.  In 2D we need 2 points, in 3D 4 points.  For each reference point we obtain an ordering
	// based on ascent and descent ordering based on the Manhattan distance.  @ indicates reference points.
	//         2D			          3D
	//   |            |		  |  |         |  |
	//   |            |		  |  @---------|--@
	// y |            |		y | /          | /
	//   |            |		  |/           |/ z
	//   @------------@		  @------------@
	//         x			        x
#ifdef P4_TO_P8
	double referencePoints[][P4EST_DIM] = { { xyzMin[0], xyzMin[1], xyzMin[2] },
											{ xyzMin[0], xyzMin[1], xyzMax[2] },
											{ xyzMax[0], xyzMin[1], xyzMin[2] },
											{ xyzMax[0], xyzMin[1], xyzMax[2] }};
#else
	double referencePoints[][P4EST_DIM] = { { xyzMin[0], xyzMin[1] },
											{ xyzMax[0], xyzMin[1] } };
#endif
	_orderings = new p4est_locidx_t *[N_ORDERINGS];
	const p4est_locidx_t N_INDEP_NODES = _nodes->indep_nodes.elem_count;	// We need to account for ghost nodes too.

	double nodePositions[N_INDEP_NODES][P4EST_DIM];
	for( p4est_locidx_t i = 0; i < N_INDEP_NODES; i++ )				// Cache the physical position of each indep node.
		node_xyz_fr_n( i, _p4est, _nodes, nodePositions[i] );

	for( size_t i = 0; i < N_ORDERINGS; i += 2 )					// Each reference point generates 2 orderings: ascent and descent.
	{
		_orderings[i] = new p4est_locidx_t[N_INDEP_NODES];			// Allocate ascent and descent direction orderings.
		_orderings[i+1] = new p4est_locidx_t[N_INDEP_NODES];

		const double *refPoint = referencePoints[i / 2];			// Current reference point.
		std::vector<NodePairL1> nodePairs( N_INDEP_NODES );
		for( p4est_locidx_t j = 0; j < N_INDEP_NODES; j++ )			// Compute the Manhattan distance from partition nodes to current reference point.
		{
#ifdef P4_TO_P8
			double diff[P4EST_DIM] = { refPoint[0] - nodePositions[j][0],
							  		   refPoint[1] - nodePositions[j][1],
							  		   refPoint[2] - nodePositions[j][2] };
#else
			double diff[P4EST_DIM] = { refPoint[0] - nodePositions[j][0],
							  		   refPoint[1] - nodePositions[j][1] };
#endif
			nodePairs[j] = { j, compute_L1_norm( diff, P4EST_DIM ) };
		}

		// Sort indices in list of node pairs based on L1 norm.
		std::sort( nodePairs.begin(), nodePairs.end(), _comparator );
		p4est_locidx_t f = 0, b = N_INDEP_NODES - 1;
		for( const auto& nodePair : nodePairs )
		{
			_orderings[i][f] = nodePair.index;						// Ascent direction.
			_orderings[i+1][b] = nodePair.index;					// Descent direction.
			f++;
			b--;
		}
	}
}

void FastSweeping::reinitializeLevelSetFunction( Vec *u, unsigned maxIter )
{
	const p4est_locidx_t N_INDEP_NODES = _nodes->indep_nodes.elem_count;	// We need to account for ghost nodes too.
	maxIter = MAX( 1u, maxIter );

	// Start afresh with the solution data structures.
	_clearSolutionData();
	_u = u;
	PetscErrorCode ierr = VecDuplicate( *u, &_rhs );
	CHKERRXX( ierr );

	// Getting access to the memory in the solution parallel vector and in Eikonal equation's rhs inverse speed.
	ierr = VecGetArray( *_u, &_uPtr );
	CHKERRXX( ierr );
	ierr = VecGetArray( _rhs, &_rhsPtr );
	CHKERRXX( ierr );

	// Allocate the old solution container.  Notice that it stores the solution for all independent nodes (including all ghosts).
	// Also, make a copy of the original solution to be reinitialized.  This information is used for seeding and sign-fix.
	_uOld = new double[N_INDEP_NODES];
	_uCpy = new double[N_INDEP_NODES];
	_seedStates = new SEED_STATE[N_INDEP_NODES];
	std::copy( _uPtr, _uPtr + N_INDEP_NODES, _uCpy );

	// Approximate location of interface by defining seed nodes.  Also, initialize the inverse speed for each partition
	// node by setting it as INF for seed nodes and 1 for interface-non-adjacent nodes.
	// Note: from this point on, we compute *positive* normal distances to interface.  Almost at the end we restore the
	// sign of the solution; this is the reason why we must keep a copy of the original solution signal.
	_approximateInterfaceAndSeedNodes();

	double relDiffAll = 1;													// Buffer to collect relative difference across processes.
	double relDiff = relDiffAll;
	unsigned iter = 0;
	while( relDiff > EPS && iter < maxIter )
	{
		std::copy( _uPtr, _uPtr + N_INDEP_NODES, _uOld );					// u_old = u.

		for( size_t i = 0; i < N_ORDERINGS; i++ )							// The 2^d sweep orderings.
		{
			for( size_t j = 0; j < N_INDEP_NODES; j++ )						// Update each valid node in the order given by curren sweep.
			{
				p4est_locidx_t n = _orderings[i][j];
				if( _neighbors->has_valid_qnnn( n ) )						// Valid locally ownded or ghost node?
					_uPtr[n] = MIN( _uPtr[n], _computeNewUAtNode( n ) );
			}
		}

		// Gather updated "common boundary" ghost nodes solution onto remotely owned nodes.
		if( _p4est->mpisize > 1 )			// Must check this or Petsc fails in the case of a single-process run.
		{
			ierr = VecGhostUpdateBegin( *_u, MIN_VALUES, SCATTER_REVERSE );
			CHKERRXX( ierr );
			ierr = VecGhostUpdateEnd( *_u, MIN_VALUES, SCATTER_REVERSE );
			CHKERRXX( ierr );

			// Scatter minimum solution values from local onto remote ghost nodes.
			ierr = VecGhostUpdateBegin( *_u, INSERT_VALUES, SCATTER_FORWARD );
			CHKERRXX( ierr );
			VecGhostUpdateEnd( *_u, INSERT_VALUES, SCATTER_FORWARD );
			CHKERRXX( ierr );
		}

		// Update local relative difference using the solution in this partition.
		relDiff = 0;
		for( size_t i = 0; i < N_INDEP_NODES; i++ )
			relDiff += fabs( _uPtr[i] - _uOld[i] );
		relDiff /= N_INDEP_NODES;

		// Broadcast relDiff to all processes and collect the max of them.  We must wait until all partitions converge.
		MPI_Allreduce( &relDiff, &relDiffAll, 1, MPI_DOUBLE, MPI_MAX, _p4est->mpicomm );
		relDiff = relDiffAll;
		iter++;
	}

	// Fix sign of expected negative solution nodal values.
	_fixSolutionSign();

	// Cleaning up.
	ierr = VecRestoreArray( *_u, &_uPtr );
	CHKERRXX( ierr );

	ierr = VecRestoreArray( _rhs, &_rhsPtr );
	CHKERRXX( ierr );
	ierr = VecDestroy( _rhs );
	CHKERRXX( ierr );

	_clearSolutionData();
}