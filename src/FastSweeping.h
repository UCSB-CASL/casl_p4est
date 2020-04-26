//
// Created by Im YoungMin on 4/21/20.
//

#ifndef FAST_SWEEPING_FASTSWEEPING_H
#define FAST_SWEEPING_FASTSWEEPING_H

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <algorithm>
#include <vector>

// TODO: Relyng on accessing is_qnnn_valid[.] from my_p4est_node_neighbors.

/**
 * Point at the origin.
 */
class APoint: public CF_2
{
public:
	double operator()( double x, double y ) const override
	{
		return sqrt( SQR( x ) + SQR( y ) );			// Distance to the origin.
	}
};

/**
 * Implementation of the Fast Sweeping Method in a parallel distributed environment using quadtrees (resp. octrees) to
 * solve the Eikonal equation $|\nabla u| = 1$, which is effectively the process of building a signed distance function u.
 * Algorithm is adapted from the "Domain Decomposition Parallel FSM" presented in [4].
 * Orderings are based on grid node distances to reference points as described in [3].
 * Updates to the nodal solution, u, of the Eikonal equation are based on the methods given in [1] and [2].
 */
class FastSweeping
{
private:
	/**
	 * Auxiliary struct to hold locally owned nodes' indexes and their distances to a reference point.
	 * Used to compute sweep orderings by sorting points according to their Manhattan distance to a reference point.
	 */
	struct NodePairL1
	{
		p4est_locidx_t index;								// From 0 to number of owned indep. nodes - 1.
		double distance;									// L1-norm distance to a reference point.
	};

	const p4est_t *_p4est = nullptr;						// Pointer to the parallel p4est data.
	const p4est_ghost_t *_ghost = nullptr;					// Pointer to the quadrants that neighbor the local domain.
	const p4est_nodes_t *_nodes = nullptr;					// Pointer to parallel nodes' information.
	const my_p4est_node_neighbors_t *_neighbors = nullptr;	// Pointer to nodes' neighborhood information.
	const size_t N_ORDERINGS;								// Number of orderings (depends on number of spatial dimensions).

	p4est_locidx_t **_orderings = nullptr;					// Sweep orderings: 2^d, for d dimensions.
	Vec *_u = nullptr;										// Parallel PETSc vector to hold solution data.
	double *_uPtr = nullptr;								// Pointer to solution, which is backed by the _u parallel PETSc vector.
	double *_uOld = nullptr;								// Dynamic array to store old values of solution.  Used for convergence checks.
	double *_rhs = nullptr;									// Right-hand-side of Eikonal equation: 1 for updatable node, INF otherwise.

	/**
	 * Clear and drop supporting data structures such as the matrix of orderings and valid node indices.
	 * This function just resets structures that directly depend on the tree definition.
	 */
	void _clearScaffoldingData();

	/**
	 * Clear and drop data structures associated with solutions that were computed on top of the tree definition.
	 */
	void _clearSolutionData();

	/**
	 * Copy current solution vector into old u.
	 */
	void _copySolutionIntoOldU();

	/**
	 * Helper function to compute the constants a and h in Hamiltonian discretization.
	 * @param [in] qnnnPtr Pointer to center node's quad node neighborhood.
	 * @param [out] a Minimum neighboring nodal approximation to u in each cartesian direction.
	 * @param [out] h Distance between center node and the chosen minimum neighboring nodal approximation given in a.
	 */
	void _defineHamiltonianConstants( const quad_neighbor_nodes_of_node_t *qnnnPtr, double a[], double h[] );

	/**
	 * Sort Hamiltonian constants in ascending order with respect to a values.
	 * @param [in/out] a Minimum neighboring nodal approximation to u in each cartesian direction.
	 * @param [in/out] h Distance between center node and the chosen minimum neighboring nodal approximation given in a.
	 */
	static void _sortHamiltonianConstants( double a[], double h[] );

	/**
	 * Compute new value of solution u at a partition fsm node.
	 * A partition fsm node is defined as an independent node that is either locally owned (shared and inner) or ghost,
	 * as long as it has a well defined quad neighborhood.
	 * @param [in] n Valid node index (i.e. that exists in _nodeIndices).
	 * @return New value for u, which must be later compared with current u value and choose the minimum.
	 */
	double _computeNewUAtNode( p4est_locidx_t n );

	/**
	 * Comparator function for sorting using std::sort function on NodeParL1 objects.
	 * @param [in] o1 First object.
	 * @param [out] o2 Second object.
	 * @return True is first point has shorter distance than second, false otherwise.
	 */
	inline static bool _comparator( const NodePairL1& o1, const NodePairL1& o2 )
	{
		return o1.distance < o2.distance;
	}

public:
	/**
	 * Constructor.
	 */
	FastSweeping();

	/**
	 * Cleaning destructor.
	 */
	~FastSweeping();

	/**
	 * Prepare fast sweeping process.
	 * @param [in] p4est Pointer to a p4est object.
	 * @param [in] ghost Pointer to ghost data structure.
	 * @param [in] nodes Pointer to nodes data structure.
	 * @param [in] neighbors Pointer to neighbors data structure.
	 * @param [in] xyzMin Lower-left limits for physical domain.
	 * @param [in] xyzMax Upper-right limits for physical domain.
	 */
	void prepare( const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes,
			const my_p4est_node_neighbors_t *neighbors, const double xyzMin[], const double xyzMax[] );

	/**
	 * Reinitialize a level-set function using the parallel fast sweeping method.
	 * @param [in/out] u Pointer to external PETSc parallel vector solution of the Eikonal equation.
	 */
	void reinitializeLevelSetFunction( Vec *u );
};


#endif //FAST_SWEEPING_FASTSWEEPING_H
