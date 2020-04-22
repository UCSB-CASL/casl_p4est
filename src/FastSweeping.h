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

/**
 * Implementation of the Fast Sweeping Method in a parallel distributed environment using quadtrees (resp. octrees).
 * Algorithm is adapted from the Domain Decomposition Parallel FSM presented in "Hybrid massively parallel fast sweeping
 * method for static Hamilton-Jacobi equations" by M. Detrixhe and F. Gibou, 2016 [1].
 * Orderings are based on grid node distances to reference points as described in "Fast Sweeping Methods for Eikonal
 * Equations on Triangular Meshes" by J. Qian, Y. Zhang, and H. Zhao, 2007 [2].
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

	void _clearOrderings();

	/**
	 * Comparator function for sorting using std::sort function on NodeParL1 objects.
	 * @param other Reference to other object of same kind.
	 * @return True is this point has shorter distance, false otherwise.
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
	 * @param p4est Pointer to a p4est object.
	 * @param ghost Pointer to ghost data structure.
	 * @param nodes Pointer to nodes data structure.
	 * @param neighbors Pointer to neighbors data structure.
	 */
	p4est_locidx_t ** prepare( const p4est_t *p4est, const p4est_ghost_t *ghost, const p4est_nodes_t *nodes,
			const my_p4est_node_neighbors_t *neighbors, const double xyzMin[], const double xyzMax[] );

	void reinitializeLevelSetFunction( Vec& phi, Vec& l1Norm_1 );

	////////////////////////////////////////////////////// Setters /////////////////////////////////////////////////////

	/**
	 * Set the p4est internal pointer.
	 * @param p4est Pointer to a p4est object.
	 */
	void setP4est( const p4est_t *p4est );

	/**
	 * Set the internal pointer to quadrants that neighbor local domain.
	 * @param ghost Pointer to ghost data structure.
	 */
	void setGhost( const p4est_ghost_t *ghost );

	/**
	 * Set the internal pointer to parallel nodes' information data structure.
	 * @param nodes Pointer to nodes data structure.
	 */
	void setNodes( const p4est_nodes_t *nodes );

	/**
	 * Set the internal pointer to nodes' neighborhood information.
	 * @param neighbors Pointer to neighbors data structure.
	 */
	 void setNeighbors( const my_p4est_node_neighbors_t *neighbors );
};


#endif //FAST_SWEEPING_FASTSWEEPING_H
