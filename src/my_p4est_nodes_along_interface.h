//
// Created by Im YoungMin on 5/17/20.
//

#ifndef FAST_SWEEPING_MY_P4EST_NODES_ALONG_INTERFACE_H
#define FAST_SWEEPING_MY_P4EST_NODES_ALONG_INTERFACE_H

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
#include <vector>

/**
 * Class to hold methods related to retrieving and processing nodes along the interface, such as collecting their and
 * full stencils with uniform distancing along the Cartesian directions.
 */
class NodesAlongInterface
{
private:

	const p4est_t *_p4est = nullptr;							// Pointer to the parallel p4est data structure.
	const p4est_nodes_t *_nodes = nullptr;						// Pointer to parallel nodes' information.
	const my_p4est_node_neighbors_t *_neighbors = nullptr;		// Pointer to nodes' neighborhood information.
	const signed char _maxLevelOfRefinement = P4EST_MAXLEVEL;	// Expected maximum level of refinement, possibly different
																// to actual maximum level of refinement reached by trees.
	double _h = 0.0;											// Uniform minimum spacing for quads/octs crossed by \Gamma.
	double _zEPS = EPS;											// To be a scaled version of EPS, based on my_p4est_level_set.h.
	const double *_phiPtr = nullptr;							// Read-pointer for a queried level-set function.
	std::vector<bool> _visited;									// Cache vector of visited nodes when collecting points
																// by the interface.

	/**
	 * Process a quadrant/octant and obtain the indices of nodes that lie on or next to the interface.  That is, check
	 * for the nodes in the quad/oct and verify if there are some whose irradiating edges are crossed by \Gamma.  If so,
	 * place their indices in an output dynamic vector.
	 * @param [in] quad Pointer to quadrant object to analyze.
	 * @param [in] quadIdx Absolute index of quadrant in macromesh.
	 * @param [out] indices Vector of nodal indices to be populated.
	 */
	void _processQuadOct( const p4est_quadrant_t *quad, p4est_locidx_t quadIdx, std::vector<p4est_locidx_t>& indices );

	/**
	 * Shortcut function to compare the nodal indices comming from the my_p4est_node_neighbors_t::get_all_neighbors with
	 * the nodal indices obtained from the qnnn.neighbor(.) function.  For this, we first match the order-based index,
	 * and then we compare the actual nodal indices.  If there's a mismatch, it means that the quadrants/octants are
	 * not uniform.
	 * @param [in] outIdx Currently evaluated order index (e.g. 0 - 9 for 2D) based on the sequencing x-y-z.
	 * @param [in matchOutIdx Which value should outIdx be tested against to proceed with nodal indices comparisons.
	 * @param [in] inIdx Currently evaluated order index (e.g. 0 - 9 for 2D) based on the sequencing z-y-x.
	 * @param [in] matchInIdx Which value should inIdx be tested against to proceed with nodal indices comparisons.
	 * @param [in] outNodeIdx Nodal index as reported by qnnn.neighbor(.).
	 * @param [in] inNodeIdx Nodal index as reported by neighbors.get_all_neighbors(.).
	 * @return True if outIdx or inIdx do not represent primary directions (e.g. they are diagonals), or if outIdx and
	 * inIdx represent primary directions and the nodal indices associated to them match.  False otherwise.
	 * @throws Runtime execption if CASL_THROWS is defined and the compared primary nodal indices do not match (e.g.
	 * a non-uniform stencil is detected).
	 */
	static bool _verifyNodesOnPrimaryDirection( int outIdx, int matchOutIdx, int inIdx, int matchInIdx,
												p4est_locidx_t outNodeIdx, p4est_locidx_t inNodeIdx );

public:

	/**
	 * Constructor.
	 * @param [in] p4est Pointer to the p4est data structure.
	 * @param [in] nodes Pointer to the nodes data structure.
	 * @param [in] neighbors Pointer to neighbors data structure.
	 * @param [in] maxLevelOfRefinement Expected maximum level of refinement in any tree in the forest.
	 * @throws Runtime exception if expected smallest quad/oct is not square.
	 */
	NodesAlongInterface( const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *neighbors,
						signed char maxLevelOfRefinement );

	/**
	 * Retrieve indices of locally-owned nodes that are on or adjacent to the interface.
	 * Warning! This function returns indices of nodes that belong to at least one quad/oct at the maximum level of
	 * refinement AND at least one of their irradiating edges is crossed by \Gamma.  The function can still return the
	 * index of a node that meets the previous condition, yet it is a neighbor to a node that is not at the same
	 * distance than the other neighbors.  The function `getFullStencilOfNode(.)` will discard these nodes because their
	 * neighborhood is not uniform.
	 * @param [in] phi A PETSc parallel vector with nodal level-set function values.
	 * @param [out] indices Dynamic vector of nodal indices on or adjacent to \Gamma.
	 */
	void getIndices( const Vec *phi, std::vector<p4est_locidx_t>& indices );

	/**
	 * Retrieve the full stencil of neighbor-node indices to a locally owned node if and only if all of them have uniform
	 * distances in all Cartesian direction, the query node is not on a wall, and the distance to its 4 (resp. 6)
	 * primary neighbors is the actual shortest distance determined by the expected maximum level of refinement of all
	 * trees.  That is, for a well-defined behavior, this function should be called on nodes that are known to be on or
	 * adjacent to the interface.  (See function getIndices to fulfill the current function's preconditions).
	 * When successful, the produced full stencil of neighbor-nodes to the query node are laid out as follows:
	 *
	 *        Two-dimensional                                    Three-dimensional
	 *     [2]------[5]------[8]  p                    [06]    ····     [15]    ····     [24]  p
	 *      |        |        |                       / |              / |              / |    |
	 *      |        |        |                    [07] |           [16] |           [25] |    |
	 *     [1]------[4]------[7]  0 y             / | [03]         / | [12]         / | [21]   0 y
	 *      |        |        |                [08] | / |       [17] | / |       [26] | / |    |
	 *      |        |        |                  | [04] |         | [13] |         | [22] |    |
	 *     [0]------[3]------[6]  m              | /| [00]        | /| [09]        | /| [18]   m
	 *      m        0        p                [05] | /         [14] | /         [23] | /     /
	 *               x                           | [01]           | [10]           | [19]   0  z
	 *                                           | /              | /              | /     /
	 *                                          [02]    ····    [11]    ····    [20]    p
	 *                                           m                0               p
	 *                                                            x
	 *
	 * The order of the nodes is depicted on the 0-based indices in each of the diagrams above.  The stencil is returned to
	 * the caller in that particular order if the stencil is well defined.  Assuming that each Cartesian dimension is a
	 * 3-state variable, the returned data is organized as a "truth" table, where the columns are provided, strictly, as
	 * x-y-z:  X changing slowly, Y changing faster than X, and Z changing faster than Y.  That is, if {m,0,p} are the three
	 * states for each dimension variable, then,
	 *
	 *              Two-dimensional                  Three-dimensional                  States:
	 *                # | x | y                        # | x | y | z                   m: Left/Bottom/Back
	 *               ---+---+---                      ---+---+---+---                  0: Center
	 *                0 | m | m                        0 | m | m | m                   p: Right/Top/Front
	 *                1 | m | 0                        1 | m | m | 0
	 *                2 | m | p                        2 | m | m | p                   (This is based on how quadrant or
	 *                3 | 0 | m                        3 | m | 0 | m                   octants are filled up with function
	 *                4 | 0 | 0                        4 | m | 0 | 0                   values.  See types.h).
	 *                5 | 0 | p                        5 | m | 0 | p
	 *                6 | p | m                        6 | m | p | m
	 *                7 | p | 0                        7 | m | p | 0
	 *                8 | p | p                        8 | m | p | p
	 *               ---+---+---                       9 | 0 | m | m
	 *                                                 : | : | : | :
	 *                                                18 | p | m | m
	 *                                                 : | : | : | :
	 *                                                26 | p | p | p
	 *                                                ---+---+---+---
	 * @param [in] nodeIdx Query node index, which must be locally owned by partition.
	 * @param [out] stencil Output vector with node indices sorted as above (in x-y[-z]).  Invalid if function returns false.
	 * @return True if uniform stencil is well defined, false otherwise.
	 * @throws Runtime exception if stencil cannot be defined for input node index and if CASL_THROWS macro is defined.
	 */
	bool getFullStencilOfNode( p4est_locidx_t nodeIdx, std::vector<p4est_locidx_t>& stencil );
};


#endif //FAST_SWEEPING_MY_P4EST_NODES_ALONG_INTERFACE_H
