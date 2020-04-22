//
// Created by Im YoungMin on 4/21/20.
//

#include "FastSweeping.h"


/////////////////////////////////////////////// Private member functions ///////////////////////////////////////////////

void FastSweeping::_clearOrderings()
{
	if( _orderings )			// Free memory if there are orderings previously loaded.
	{
		for( size_t i = 0; i < N_ORDERINGS; i++ )
			delete [] _orderings[i];
		delete [] _orderings;
	}
}

/////////////////////////////////////////////// Public member functions ////////////////////////////////////////////////

FastSweeping::FastSweeping(): N_ORDERINGS( static_cast<size_t>( pow( 2, P4EST_DIM ) ) ){}

FastSweeping::~FastSweeping()
{
	_clearOrderings();
}

p4est_locidx_t ** FastSweeping::prepare( const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes,
							const my_p4est_node_neighbors_t *neighbors, const double xyzMin[], const double xyzMax[] )
{
	// Start afresh; setting pointers to internal pointers.
	_p4est = p4est;
	_ghost = ghost;
	_nodes = nodes;
	_neighbors = neighbors;
	_clearOrderings();

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
	const p4est_locidx_t N_LOCAL_NODES = _nodes->num_owned_indeps;

	double nodePositions[N_LOCAL_NODES][P4EST_DIM];
	for( p4est_locidx_t i = 0; i < N_LOCAL_NODES; i++ )				// Cache the physical position of each locally owned indep node.
		node_xyz_fr_n( i, _p4est, _nodes, nodePositions[i] );

	for( size_t i = 0; i < N_ORDERINGS; i += 2 )					// Each reference point generates 2 orderings: ascent and descent.
	{
		_orderings[i] = new p4est_locidx_t[N_LOCAL_NODES];			// Allocate ascent and descent direction orderings.
		_orderings[i+1] = new p4est_locidx_t[N_LOCAL_NODES];

		const double *refPoint = referencePoints[i / 2];			// Current reference point.
		std::vector<NodePairL1> nodePairs( N_LOCAL_NODES );
		for( p4est_locidx_t j = 0; j < N_LOCAL_NODES; j++ )			// Compute the Manhattan distance from grid nodes to current reference point.
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
		p4est_locidx_t f = 0, b = N_LOCAL_NODES - 1;
		for( const auto& nodePair : nodePairs )
		{
			_orderings[i][f] = nodePair.index;						// Ascent direction.
			_orderings[i+1][b] = nodePair.index;					// Descent direction.
			f++;
			b--;
		}
	}

	return _orderings;
	// TODO: Determine the interface and the initial signed distance to adjacent nodes.
}

void FastSweeping::reinitializeLevelSetFunction( Vec& phi, Vec& l1Norm_1 )
{
	PetscErrorCode ierr;

	// Calculate L^1 norm from each locally owned node to the bottom left reference node.
//	Vec l1Norm_1;
//	ierr = VecCreateGhostNodes( _p4est, _nodes, &l1Norm_1 );
//	CHKERRXX( ierr );
/*
	double *l1NormPtr_1;
	ierr = VecGetArray( l1Norm_1, &l1NormPtr_1 );
	CHKERRXX( ierr );

	for( size_t i = 0; i < _nodes->num_owned_indeps; i++ )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( i, _p4est, _nodes, xyz );
		double diff[P4EST_DIM] = { xyz[0] - xyzMin[0], xyz[1] - xyzMin[1] };
		l1NormPtr_1[i] = compute_L1_norm( diff, P4EST_DIM );
	}

	ierr = VecGhostUpdateBegin( l1Norm_1, INSERT_VALUES, SCATTER_FORWARD );		// After we are done with the locally
	CHKERRXX( ierr );															// owned nodes, scatter them onto the
	ierr = VecGhostUpdateEnd( l1Norm_1, INSERT_VALUES, SCATTER_FORWARD ); 		// ghost nodes.
	CHKERRXX( ierr );

	ierr = VecRestoreArray( l1Norm_1, &l1NormPtr_1 );
	CHKERRXX( ierr );
*/
//	ierr = VecDestroy( l1Norm_1 );
//	CHKERRXX( ierr );
}

//////////////////////////////////////////////////////// Setters ///////////////////////////////////////////////////////

void FastSweeping::setP4est( const p4est_t *p4est )
{
	_p4est = p4est;
}

void FastSweeping::setGhost( const p4est_ghost_t *ghost )
{
	_ghost = ghost;
}

void FastSweeping::setNodes( const p4est_nodes_t *nodes )
{
	_nodes = nodes;
}

void FastSweeping::setNeighbors( const my_p4est_node_neighbors_t *neighbors )
{
	_neighbors = neighbors;
}
