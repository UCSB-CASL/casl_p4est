//
// Created by Im YoungMin on 5/17/20.
//

#include "my_p4est_nodes_along_interface.h"

/////////////////////////////////////////////////// Private methods ////////////////////////////////////////////////////

#ifdef P4_TO_P8
void NodesAlongInterface::_compareNodes( const OctValueExtended& phiAndIdxValues, int i, int j,
										 std::unordered_set<p4est_locidx_t>& indices )
#else
void NodesAlongInterface::_compareNodes( const QuadValueExtended& phiAndIdxValues, int i, int j,
										 std::unordered_set<p4est_locidx_t>& indices )
#endif
{
	if( phiAndIdxValues.val[i] * phiAndIdxValues.val[j] <= 0 )
	{
		indices.insert( phiAndIdxValues.indices[i] );
		indices.insert( phiAndIdxValues.indices[j] );
	}
}


#ifdef P4_TO_P8
bool NodesAlongInterface::_organizeNodalInfo( p4est_locidx_t quadIdx, OctValueExtended& phiAndIdxValues ) const
#else
bool NodesAlongInterface::_organizeNodalInfo( p4est_locidx_t quadIdx, QuadValueExtended& phiAndIdxValues ) const
#endif
{
	// Get node indices belonging to current quadrant.
	const p4est_locidx_t *q2n = _nodes->local_nodes + P4EST_CHILDREN * quadIdx;
	for( unsigned char incX = 0; incX < 2; incX++ )			// Truth table with x changing slowly, y changing faster
		for( unsigned char incY = 0; incY < 2; incY++ )		// than x, and z changing faster than y.
#ifdef P4_TO_P8
			for( unsigned char incZ = 0; incZ < 2; incZ++ )
#endif
		{
			unsigned idxInQuadOctValue = SUMD( ( 1u << (unsigned)( P4EST_DIM - 1 ) ) * incX,
											   ( 1u << (unsigned)( P4EST_DIM - 2 ) ) * incY,
											   incZ );
			p4est_locidx_t nodeSubIdxInQuad = q2n[SUMD( incX, 2 * incY, 4 * incZ )];
			phiAndIdxValues.val[idxInQuadOctValue] = _phiPtr[nodeSubIdxInQuad];		// Store nodal value and their
			phiAndIdxValues.indices[idxInQuadOctValue] = nodeSubIdxInQuad;			// indices.
		}

	// Discard quad/oct if not crossed by Gamma.
	if( phiAndIdxValues.val[0] <= 0 && phiAndIdxValues.val[1] <= 0 &&
		phiAndIdxValues.val[2] <= 0 && phiAndIdxValues.val[3] <= 0
		ONLY3D( && phiAndIdxValues.val[4] <= 0 && phiAndIdxValues.val[5] <= 0 &&
					phiAndIdxValues.val[6] <= 0 && phiAndIdxValues.val[7] <= 0 ) )
		return false;

	if( phiAndIdxValues.val[0] > 0 && phiAndIdxValues.val[1] > 0 &&
		phiAndIdxValues.val[2] > 0 && phiAndIdxValues.val[3] > 0
		ONLY3D( && phiAndIdxValues.val[4] > 0 && phiAndIdxValues.val[5] > 0 &&
					phiAndIdxValues.val[6] > 0 && phiAndIdxValues.val[7] > 0 ) )
		return false;

	return true;		// Quad/oct is crossed by Gamma.
}


void NodesAlongInterface::_processIndepQuadOct( p4est_locidx_t quadIdx, std::unordered_set<p4est_locidx_t> &indices ) const
{
#ifdef P4_TO_P8
	OctValueExtended  phiAndIdxValues;
#else
	QuadValueExtended phiAndIdxValues;
#endif

	// Get quad/oct data.
	if( !_organizeNodalInfo( quadIdx, phiAndIdxValues ) )	// Stop processing quad/oct if not crossed by Gamma.
		return;

	// Gamma crosses the quad/oct.  Find the pairs of affected vertices.
	// We consider the fact that if one of the quad corners falls exactly on the interface (i.e., phi = 0), all of its
	// neighboring points will be included in the indices set.  This is different than _processQuadOct, where we don't
	// add any of the neigobors of a node sitting exactly on the interface.
	// Based on types.h:
	//		 2D                      3D                   #vertex  #loc
	//	v01      v11			010      110               	000		0
	//	 *--------*				 *--------*					001		1
	//	 |        |				/.   111 /|					010		2
	// 	 |        |		   011 *--------* |					011		3
	//	 *--------*			   | *......|.*					100		4
	//	v00      v10		   |· 000   |/ 100				101		5
	//		 				   *--------*					110		6
	//	   y|				  001      101	   y|			111		7
	//		+---								+--- x
	//		  x								   /z
	for( int i = 0; i < P4EST_DIM - 1; i++ )	// Loop covers the quad case and the left and right faces of octant.
	{
		int stride = i * 4;
		_compareNodes( phiAndIdxValues, 0 + stride, 1 + stride, indices );	// Notice we don't compare nodes across
		_compareNodes( phiAndIdxValues, 0 + stride, 2 + stride, indices );	// the quad's diagonal.
		_compareNodes( phiAndIdxValues, 3 + stride, 1 + stride, indices );
		_compareNodes( phiAndIdxValues, 3 + stride, 2 + stride, indices );
	}

#ifdef P4_TO_P8
	// In 3D, we have four more edeges to check, parallel to x axis.
	for( int i = 0; i < 4; i++ )
		_compareNodes( phiAndIdxValues, i, i + 4, indices );
#endif
}


void NodesAlongInterface::_processQuadOct( p4est_locidx_t quadIdx, std::vector<p4est_locidx_t>& indices )
{
#ifdef P4_TO_P8
	OctValueExtended  phiAndIdxValues;
#else
	QuadValueExtended phiAndIdxValues;
#endif

	// Get quad/oct data.
	if( !_organizeNodalInfo( quadIdx, phiAndIdxValues ) )	// Stop processing quad/oct if not crossed by Gamma.
		return;

	// Check each non-visited node with an irradiating edge being crossed by Gamma.
	double data[P4EST_DIM][2][2];
	const short F = 0, S = 1;								// Meaning: function index, distance index.
	for( auto n : phiAndIdxValues.indices )
	{
		if( n >= _nodes->num_owned_indeps || _visited[n] )
			continue;

		if( ABS( _phiPtr[n] ) <= _zEPS )					// Point lies on the interface.
			indices.push_back( n );
		else
		{
			const quad_neighbor_nodes_of_node_t *qnnnPtr;	// Evaluate neighborhood.
			_neighbors->get_neighbors( n, qnnnPtr );

			getStencil( qnnnPtr, _phiPtr, data );			// Retrieve the 2 or 3 dimensional stencil.
			bool crossed = false;

			for( const auto& dim : data )					// Check irradiating edges in each dimension.
			{
				for( const auto& dir : dim )				// And in each direction: left/bottom/back - right/top/front
				{
					if( dir[S] < 0 )						// Is current node on a wall?
						continue;
					if( dir[F] * _phiPtr[n] < 0 )			// Crossed by interface? Unlike in FastSweeping, we don't
					{										// check for <= 0 because it then will output points that
						crossed = true;						// are neighbors to nodes that fall exactly on Gamma.
						break;
					}
				}

				if( crossed )								// Just find a single edge crossed by Gamma.
					break;
			}

			if( crossed )
				indices.push_back( n );
		}

		_visited[n] = true;
	}
}

bool NodesAlongInterface::_verifyNodesOnPrimaryDirection( int outIdx, int matchOutIdx, int inIdx, int matchInIdx,
														  p4est_locidx_t outNodeIdx, p4est_locidx_t inNodeIdx )
{
	if( outIdx == matchOutIdx && inIdx == matchInIdx )		// Primary direction? (left, right, top, left, front, back).
	{
		if( outNodeIdx != inNodeIdx )						// For a uniform stencil these indices should match.
		{
#ifdef CASL_THROWS
			throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: Non-uniform stencil!" );
#endif
			return false;
		}
	}

	return true;		// Diagonal directions are skipped (they evaluate to true).
}

/////////////////////////////////////////////// Public interface methods ///////////////////////////////////////////////

NodesAlongInterface::NodesAlongInterface( const p4est_t *p4est, const p4est_nodes_t *nodes,
										  const my_p4est_node_neighbors_t *neighbors, signed char maxLevelOfRefinement )
										  : _p4est( p4est ), _nodes( nodes ), _neighbors( neighbors ),
											_maxLevelOfRefinement( maxLevelOfRefinement )
{
	// Establish the zero distance threshold as a scaled version of EPS that depends on the local domain resolution.
	_zEPS = EPS * MIN( DIM( ( _neighbors->myb->xyz_max[0] - _neighbors->myb->xyz_min[0] ) / _neighbors->myb->nxyztrees[0],
							( _neighbors->myb->xyz_max[1] - _neighbors->myb->xyz_min[1] ) / _neighbors->myb->nxyztrees[1],
							( _neighbors->myb->xyz_max[2] - _neighbors->myb->xyz_min[2] ) / _neighbors->myb->nxyztrees[2] ) );

	// Set the minimum spacing for cells crossed by the interface.  We require the minimum spacing to be uniform for
	// all directions.
	p4est_topidx_t vm = _p4est->connectivity->tree_to_vertex[0];
	p4est_topidx_t vp = _p4est->connectivity->tree_to_vertex[P4EST_CHILDREN-1];
	const double* tree_xyz_min = _p4est->connectivity->vertices + 3 * vm;
	const double* tree_xyz_max = _p4est->connectivity->vertices + 3 * vp;
	double dmin = (double)P4EST_QUADRANT_LEN( _maxLevelOfRefinement ) / (double)P4EST_ROOT_LEN;	// Side length proportion of smallest quad/oct.

	// Minimum cell width in any tree.
	double dx = dmin * ( tree_xyz_max[0] - tree_xyz_min[0] );
	double dy = dmin * ( tree_xyz_max[1] - tree_xyz_min[1] );
#ifdef P4_TO_P8
	double dz = dmin * ( tree_xyz_max[2] - tree_xyz_min[2] );
#endif

	// Do not accept any adaptive grid with non-square quads/octs.
	if( ABS( dx - dy ) <= _zEPS ONLY3D( && ABS( dx - dz ) <= _zEPS ) )
		_h = dx;
	else
		throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::FullUniformStencil: smallest quad is not uniform!" );
}

void NodesAlongInterface::getIndices( const Vec *phi, std::vector<p4est_locidx_t>& indices )
{
	PetscErrorCode ierr;
	ierr = VecGetArrayRead( *phi, &_phiPtr );		// Access the parallel vector with level-set function values.
	CHKERRXX( ierr );

	// Reserve space only for locally-owned nodes.
	indices.clear();								// Clean the output vector of indices.
	indices.reserve( _nodes->num_owned_indeps );
	_visited.clear();								// Clean visited status caching vector and reset it to false.
	_visited.resize( _nodes->num_owned_indeps );
	for( auto&& i : _visited)
		i = false;

	// Go through each quad/octant and check if it's crossed by the interface.  As a shortcut, we can skip quads/octs
	// that are not at the max level of refinement since that means they are not detailed enough to contain Gamma.
	for( p4est_topidx_t treeIdx = _p4est->first_local_tree; treeIdx <= _p4est->last_local_tree; treeIdx++ )
	{
		auto *tree = (p4est_tree_t*)sc_array_index( _p4est->trees, treeIdx );	// Check all local trees that posses local
		if( tree->maxlevel == _maxLevelOfRefinement )							// quads/octs with max level of refinement.
		{
			// Check each quadrant in local trees.
			for( size_t quadIdx = 0; quadIdx < tree->quadrants.elem_count; quadIdx++ )
			{
				auto *quad = (const p4est_quadrant_t*)sc_array_index( &tree->quadrants, quadIdx );
				if( quad->level == _maxLevelOfRefinement )	// Is this quad/oct potentially crossed by the interface?
					_processQuadOct( quadIdx + tree->quadrants_offset, indices );
			}
		}
	}

	ierr = VecRestoreArrayRead( *phi, &_phiPtr );	// Cleaning up.
	CHKERRXX( ierr );
}

bool NodesAlongInterface::getFullStencilOfNode( const p4est_locidx_t nodeIdx, std::vector<p4est_locidx_t>& stencil )
{
	// The stencil is valid if all neighboring nodes are at the same distance in each Cartesian direction, if the
	// center node is locally owned and is *NOT* a wall node.
	const int STEPS = 3;
	const int TOTAL_NEIGHBORS = (int)pow( STEPS, P4EST_DIM );
	stencil.clear();
	stencil.resize( TOTAL_NEIGHBORS, -1 );
	if( nodeIdx < 0 || nodeIdx >= _nodes->num_owned_indeps )
	{
#ifdef CASL_THROWS
		throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: "
								  "Node " + std::to_string( nodeIdx ) + " is not locally owned!" );
#endif
		return false;
	}

	const quad_neighbor_nodes_of_node_t& qnnn = _neighbors->get_neighbors( nodeIdx );	// Quad neighborhood of node.
	const double LOWER_B = _h - _zEPS;					// Range for distance between nodes in each Cartesian direction.
	const double UPPER_B = _h + _zEPS;
	for( int dir = 0; dir < P4EST_DIM * 2; dir++ )		// Verify that query node is not a wall node and that its
	{													// Cartesian neighbors are at the minimum distance h.
		if( qnnn.neighbor( dir ) == -1 )
		{
#ifdef CASL_THROWS
			throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: "
									  "Node " + std::to_string( nodeIdx ) + " is missing a uniform neighbor in some "
									  "Cartesian direction!" );
#endif
			return false;
		}

		if( qnnn.distance( dir ) < LOWER_B || qnnn.distance( dir ) > UPPER_B )
		{
#ifdef CASL_THROWS
			throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: "
									  "Node " + std::to_string( nodeIdx ) + " is part of a quad/oct that is not at "
									  "the maximum level of refinement!" );
#endif
			return false;
		}
	}

	// At this point, we know the query point is not on the wall.  Now, collect all neighbors, possibly with nodes in
	// quads/octs that are not all uniform.  We'll verify those and reorder the results to match x-y-z convention.
	bool neighborExists[TOTAL_NEIGHBORS];
	p4est_locidx_t neighborsOfNode[TOTAL_NEIGHBORS];
	_neighbors->get_all_neighbors( nodeIdx, neighborsOfNode, neighborExists );

	// Check uniformity of neighbors.
	double xyzCenter[P4EST_DIM];								// Retrieve location of center node.
	node_xyz_fr_n( nodeIdx, _p4est, _nodes, xyzCenter );
	const double FLAT_DIAG_LOWER_B = _h * M_SQRT2 - _zEPS;		// Diagonal neighbors on the same facet/plane: lower and
	const double FLAT_DIAG_UPPER_B = _h * M_SQRT2 + _zEPS;		// upper bounds.
#ifdef P4_TO_P8
	const double CUBE_DIAG_LOWER_B = _h * sqrt( 3 ) - _zEPS;	// Diagonal neighbors on different plane: lower and
	const double CUBE_DIAG_UPPER_B = _h * sqrt( 3 ) + _zEPS;	// upper bounds.
#endif
	for( int xStep = 0; xStep < STEPS; xStep++ )
		for( int yStep = 0; yStep < STEPS; yStep++ )
#ifdef P4_TO_P8
			for( int zStep = 0; zStep < STEPS; zStep++ )
#endif
		{
			// Using this index to store information as a truth table with 3 states per dimension: m (minus),
			// 0 (center), and p (plus), so that the most significan "bit" is x, then y [, then z].
			int outIdx = SUMD( xStep * (int)pow( STEPS, P4EST_DIM - 1 ), yStep * (int)pow( STEPS, P4EST_DIM - 2 ), zStep );

#ifdef P4_TO_P8
			// Incoming index from neighbors function.
			int inIdx = SUMD( xStep, yStep * (int)pow( STEPS, P4EST_DIM - 2 ), zStep * (int)pow( STEPS, P4EST_DIM - 1 ) );

			// Checking that the 6 primary direction nodes match those provided by qnnn.
			if( !_verifyNodesOnPrimaryDirection( outIdx, 4, inIdx, 12, qnnn.neighbor_m00(), neighborsOfNode[inIdx] ) ||		// Left?
			    !_verifyNodesOnPrimaryDirection( outIdx, 22, inIdx, 14, qnnn.neighbor_p00(), neighborsOfNode[inIdx] ) ||	// Right?
			    !_verifyNodesOnPrimaryDirection( outIdx, 10, inIdx, 10, qnnn.neighbor_0m0(), neighborsOfNode[inIdx] ) ||	// Bottom?
			    !_verifyNodesOnPrimaryDirection( outIdx, 16, inIdx, 16, qnnn.neighbor_0p0(), neighborsOfNode[inIdx] ) ||	// Top?
			    !_verifyNodesOnPrimaryDirection( outIdx, 12, inIdx, 4, qnnn.neighbor_00m(), neighborsOfNode[inIdx] ) ||		// Back?
			    !_verifyNodesOnPrimaryDirection( outIdx, 14, inIdx, 22, qnnn.neighbor_00p(), neighborsOfNode[inIdx] ) )		// Front?
				return false;
#else
			// Incoming index from neighbors function.
			int inIdx = xStep + yStep * (int)pow( STEPS, P4EST_DIM - 1 );

			// Checking that the 4 primary direction nodes match those provided by qnnn.
			if( !_verifyNodesOnPrimaryDirection( outIdx, 1, inIdx, 3, qnnn.neighbor_m00(), neighborsOfNode[inIdx] ) ||	// Left?
				!_verifyNodesOnPrimaryDirection( outIdx, 7, inIdx, 5, qnnn.neighbor_p00(), neighborsOfNode[inIdx] ) ||	// Right?
				!_verifyNodesOnPrimaryDirection( outIdx, 3, inIdx, 1, qnnn.neighbor_0m0(), neighborsOfNode[inIdx] ) ||	// Bottom?
				!_verifyNodesOnPrimaryDirection( outIdx, 5, inIdx, 7, qnnn.neighbor_0p0(), neighborsOfNode[inIdx] ) )	// Top?
				return false;
#endif
			// Checking the diagonal neighbors on the same horizontal or vertical plane as the center node.
			// Sometimes the previous test passes, even though a diagonal neighbor is not part of a uniform neighborhood.
#ifdef P4_TO_P8
			if( outIdx == 1 || outIdx == 19 || outIdx == 7 || outIdx == 25 ||
				outIdx == 11 || outIdx == 9 || outIdx == 17 || outIdx == 15 )
#else
			if( outIdx == 0 || outIdx == 2 || outIdx == 6 || outIdx == 8 )
#endif
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( neighborsOfNode[inIdx], _p4est, _nodes, xyz );
				double DIM( dx = xyz[0] - xyzCenter[0], dy = xyz[1] - xyzCenter[1], dz = xyz[2] - xyzCenter[2] );
				double distance = sqrt( SUMD( SQR( dx ), SQR( dy ), SQR( dz ) ) );
				if( distance < FLAT_DIAG_LOWER_B || distance > FLAT_DIAG_UPPER_B )
				{
#ifdef CASL_THROWS
					throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: "
											  "Node " + std::to_string( nodeIdx ) + " has an unexpectedly far "
										      "flat-diagonal neighbor!" );
#endif
					return false;
				}
			}

#ifdef P4_TO_P8
			// Checking the diagonal neighbors in a different horizontal or vertical plane than the center node.
			// Cube-diagonal elements only happen in 3D.
			if( outIdx == 2 || outIdx == 20 || outIdx == 8 || outIdx == 26 ||
				outIdx == 0 || outIdx == 18 || outIdx == 6 || outIdx == 24 )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( neighborsOfNode[inIdx], _p4est, _nodes, xyz );
				double DIM( dx = xyz[0] - xyzCenter[0], dy = xyz[1] - xyzCenter[1], dz = xyz[2] - xyzCenter[2] );
				double distance = sqrt( SUMD( SQR( dx ), SQR( dy ), SQR( dz ) ) );
				if( distance < CUBE_DIAG_LOWER_B || distance > CUBE_DIAG_UPPER_B )
				{
#ifdef CASL_THROWS
					throw std::runtime_error( "[CASL_ERROR]: NodesAlongInterface::getFullStencilOfNode: "
											  "Node " + std::to_string( nodeIdx ) + " has an unexpectedly far "
										      "cube-diagonal neighbor!" );
#endif
					return false;
				}
			}
#endif
			stencil[outIdx] = neighborsOfNode[inIdx];
		}

	return true;
}


void NodesAlongInterface::getIndepIndices( const Vec *phi, const p4est_ghost_t *ghost,
										 std::unordered_set<p4est_locidx_t> &indices )
{
	PetscErrorCode ierr;
	ierr = VecGetArrayRead( *phi, &_phiPtr );		// Access the parallel vector with level-set function values.
	CHKERRXX( ierr );

	// Reserve space for all independent nodes.
	indices.clear();								// Start afresh.
	indices.reserve( _nodes->indep_nodes.elem_count );

	// First, go through each locally owned quad/octant and check if it's crossed by the interface.  Skip quads/octs
	// that are not at the max level of refinement since that means they are not detailed enough to contain Gamma.
	for( p4est_topidx_t treeIdx = _p4est->first_local_tree; treeIdx <= _p4est->last_local_tree; treeIdx++ )
	{
		auto *tree = (p4est_tree_t*)sc_array_index( _p4est->trees, treeIdx );	// Check all local trees that posses local
		if( tree->maxlevel == _maxLevelOfRefinement )							// quads/octs with max level of refinement.
		{
			// Check each quadrant in local trees.
			for( size_t quadIdx = 0; quadIdx < tree->quadrants.elem_count; quadIdx++ )
			{
				auto *quad = (const p4est_quadrant_t*)sc_array_index( &tree->quadrants, quadIdx );
				if( quad->level == _maxLevelOfRefinement )	// Is this quad/oct potentially crossed by the interface?
					_processIndepQuadOct( quadIdx + tree->quadrants_offset, indices );
			}
		}
	}

	// Second, go through ghost quadrants and repeat the previous process.
	for( p4est_topidx_t ghostIdx = 0; ghostIdx < ghost->ghosts.elem_count; ghostIdx++ )
	{
		auto *quad = (const p4est_quadrant_t*)p4est_quadrant_array_index( &(const_cast<p4est_ghost_t*>(ghost)->ghosts), ghostIdx );
		if( quad->level == _maxLevelOfRefinement )			// Is this quad/oct potentially crossed by the interface?
			_processIndepQuadOct( _p4est->local_num_quadrants + ghostIdx, indices );
	}

	ierr = VecRestoreArrayRead( *phi, &_phiPtr );	// Cleaning up.
	CHKERRXX( ierr );
}


