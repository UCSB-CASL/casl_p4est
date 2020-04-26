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

void FastSweeping::_defineHamiltonianConstants( const quad_neighbor_nodes_of_node_t *qnnnPtr, double a[], double h[] )
{
	// Some convenient arrangement nodal solution values and their distances w.r.t. center node in its stencil of neighbors.
	const double data[P4EST_DIM][2][2] = {
			{
				{ qnnnPtr->f_m00_linear( _uPtr ), qnnnPtr->d_m00 },			// Left.
				{ qnnnPtr->f_p00_linear( _uPtr ), qnnnPtr->d_p00 }			// Right.
			},
			{
				{ qnnnPtr->f_0m0_linear( _uPtr ), qnnnPtr->d_0m0 },			// Bottom.
				{ qnnnPtr->f_0p0_linear( _uPtr ), qnnnPtr->d_0p0 }			// Top.
			},
#ifdef P4_TO_P8
			{
				{ qnnnPtr->f_00m_linear( _uPtr ), qnnnPtr->d_00m },			// Back.
				{ qnnnPtr->f_00p_linear( _uPtr ), qnnnPtr->d_00p }			// Front.
			}
#endif
	};

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

double FastSweeping::_computeNewUAtNode( p4est_locidx_t n )
{
	if( _rhs[n] >= PETSC_INFINITY )		// Is this an adjacent node to the interface (i.e. non-updatable)?
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
	if( MIN( DIM( a[0], a[1], a[2] ) ) == PETSC_INFINITY )
		return PETSC_INFINITY;

	// Solve for u(x) (i.e. new approximation to u) in the equation above using the method in [1] and [2].
	// 1) Sort a_i from least to greatest (h_i will not be in any particular order).
	_sortHamiltonianConstants( a, h );

	// 2) Begin solution.
	double uHat, numerator, h0Sqrd, h1Sqrd, h2Sqrd, hSqrdSum;
	double fSqrd = SQR( _rhs[n] );
	for( size_t m = 0; m < P4EST_DIM; m++ )
	{
		// 3) Solving \sum_{j=0}^{m} ( (u - a_j)^+ / h_j )^2 = f^2 for u.
		if( m == 0 )
		{
			uHat = a[0] + h[0] * _rhs[n];
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

	// Getting access to the memory in the solution parallel vector.
	PetscErrorCode ierr = VecGetArray( *_u, &_uPtr );
	CHKERRXX( ierr );

	// Allocate the old solution container.  Notice that it stores the solution for all independent nodes (including all ghosts).
	// Also, make room for Eikonal equation's rhs inverse speed.
	_uOld = new double[N_INDEP_NODES];
	_rhs = new double[N_INDEP_NODES];

	// TODO: Determine the interface and the initial signed distance to adjacent nodes.
	// TODO: For this, update also _rhs (which will include ghost nodes).
	// TODO: This is an example using a point at the origin.
	for( p4est_locidx_t i = 0; i < N_INDEP_NODES; i++ )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( i, _p4est, _nodes, xyz );
		if( sqrt( SQR( xyz[0] ) + SQR( xyz[1] ) ) < EPS )	// Interface?
		{
			_uPtr[i] = 0;
			_rhs[i] = PETSC_INFINITY;						// Fixed point.
		}
		else
		{
			_uPtr[i] = PETSC_INFINITY;						// Updatable point.
			_rhs[i] = 1.0;
		}
	}

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
					_uPtr[n] = MIN( _uPtr[n], _computeNewUAtNode( n ) );
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
