#ifndef MY_P4EST_NODE_NEIGHBORS_H
#define MY_P4EST_NODE_NEIGHBORS_H

#include <p4est.h>
#include <p4est_ghost.h>

#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <p4est_bits.h>
#include <src/my_p4est_quad_neighbor_nodes_of_node.h>
#include <src/my_p4est_hierarchy.h>

#include <vector>


class my_p4est_node_neighbors_t {

private:
    my_p4est_hierarchy_t *hierarchy;
    p4est_t *p4est;
    p4est_ghost_t *ghost;
    p4est_nodes_t *nodes;

    std::vector< quad_neighbor_nodes_of_node_t > neighbors;

    /**
     * This function is finds the neighboring cell of a node in the given (i,j) direction. The direction must be diagonal
     * for the function to work ! (e.g. (-1,1) ... no cartesian direction!).
     * \param [in] node          a pointer to the node whose neighboring cells are looked for
     * \param [in] i             the x search direction, -1 or 1
     * \param [in] j             the y search direction, -1 or 1
     * \param [out] quad         the index of the found quadrant, in mpirank numbering. To fetch this quadrant from its corresponding tree
     *                           you need to substract the tree quadrant offset. If no quadrant was found, this is set to -1 (e.g. edge of domain)
     * \param [out] nb_tree_idx  the index of the tree in which the quadrant was found
     *
     */
    void find_neighbor_cell_of_node( p4est_indep_t *node, char i, char j, p4est_locidx_t& quad_idx, p4est_topidx_t& nb_tree_idx );

    /**
     * Initialize the QuadNeighborNodeOfNode information
     */
    void init_neighbors();
public:

    my_p4est_node_neighbors_t( my_p4est_hierarchy_t *hierarchy_, p4est_nodes_t *nodes_)
        : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), nodes(nodes_), neighbors(nodes->num_owned_indeps)
    {
        init_neighbors();
    }

    inline const quad_neighbor_nodes_of_node_t& operator[]( p4est_locidx_t n ) const { return neighbors[n]; }
};

#endif /* !MY_P4EST_NODE_NEIGHBORS_H */
