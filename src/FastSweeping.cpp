//
// Created by Im YoungMin on 4/21/20.
//

#include "FastSweeping.h"


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
	delete [] _uOld;			// Free memory if there were any previously stored old solution values.
	delete [] _rhs;				// and Eikonal right-hand-side (inverse speed) values.

	// Reset critical variables.
	_u = nullptr;
	_uPtr = nullptr;
	_uOld = nullptr;
}

void FastSweeping::_copySolutionIntoOldU()
{
	for( p4est_locidx_t i = 0; i < _nodes->indep_nodes.elem_count; i++ )
		_uOld[i] = _uPtr[i];
}

double FastSweeping::_computeNewUAtNode( p4est_locidx_t n )
{
	// Access the neighborhood of requested node.  Assume that the neighborhood is well defined.
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	_neighbors->get_neighbors( n, qnnnPtr );

	return 0;
}

/////////////////////////////////////////////// Public member functions ////////////////////////////////////////////////

FastSweeping::FastSweeping(): N_ORDERINGS( static_cast<size_t>( pow( 2, P4EST_DIM ) ) ){}

FastSweeping::~FastSweeping()
{
	_clearScaffoldingData();
	_clearSolutionData();
}

void FastSweeping::prepare( const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes,
							const my_p4est_node_neighbors_t *neighbors, const double xyzMin[], const double xyzMax[] )
{
	// Start afresh; setting pointers to internal pointers.
	_clearScaffoldingData();
	_p4est = p4est;
	_ghost = ghost;
	_nodes = nodes;
	_neighbors = neighbors;

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

void FastSweeping::reinitializeLevelSetFunction( Vec *u )
{
	const p4est_locidx_t N_INDEP_NODES = _nodes->indep_nodes.elem_count;	// We need to account for ghost nodes too.

	// Start afresh with the solution data structures.
	_clearSolutionData();
	_u = u;

	// Getting access to the memory in the parallel vector.
	PetscErrorCode ierr = VecGetArray( *_u, &_uPtr );
	CHKERRXX( ierr );

	// Allocate the old solution container.  Notice that it stores the solution for all independent nodes (including all ghosts).
	// Also, make room for Eikonal equation's rhs inverse speed.
	_uOld = new double[N_INDEP_NODES];
	_rhs = new double[N_INDEP_NODES];

	// TODO: Determine the interface and the initial signed distance to adjacent nodes.
	// TODO: For this, update _rhs only in locally owned nodes and scatter forward to remote ghost nodes.

	double relDiffAll = 1;													// Buffer to collect relative difference across processes.
	double relDiff = relDiffAll;
	while( relDiff > EPS )
	{
		_copySolutionIntoOldU();											// u_old = u.

		for( size_t i = 0; i < N_ORDERINGS; i++ )							// The 2^d sweep orderings.
		{
			for( size_t j = 0; j < N_INDEP_NODES; j++ )						// Update each valid node in the order given by curren sweep.
			{
				p4est_locidx_t n = _orderings[i][j];
				if( _neighbors->has_valid_qnnn( n ) )						// Valid locally ownded or ghost node?
					_uPtr[n] = fmin( _uPtr[n], _computeNewUAtNode( n ) );
			}
		}

		// Gather updated "common boundary" ghost nodes solution onto remotely owned nodes.
		VecGhostUpdateBegin( *_u, MIN_VALUES, SCATTER_REVERSE );
		VecGhostUpdateEnd( *_u, MIN_VALUES, SCATTER_REVERSE );

		// Scatter minimum solution values from local onto remote ghost nodes.
		VecGhostUpdateBegin( *_u, INSERT_VALUES, SCATTER_FORWARD );
		VecGhostUpdateEnd( *_u, INSERT_VALUES, SCATTER_FORWARD );

		// Update local relative difference using the solution in this partition.
		relDiff = 0;
		for( size_t i = 0; i < N_INDEP_NODES; i++ )
			relDiff += fabs( _uPtr[i] - _uOld[i] );
		relDiff /= N_INDEP_NODES;

		// Broadcast relDiff to all processes and collect the max of them.  We must wait until all partitions converge.
		MPI_Allreduce( &relDiff, &relDiffAll, 1, MPI_DOUBLE, MPI_MAX, _p4est->mpicomm );
		relDiff = relDiffAll;
	}

	// Cleaning up.
	ierr = VecRestoreArray( *_u, &_uPtr );
	CHKERRXX( ierr );
}
