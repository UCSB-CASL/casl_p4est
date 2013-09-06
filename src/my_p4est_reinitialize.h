#include <src/CASL_math.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>

#include <vector>

void reinitialize_One_Iteration( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                                         double *p0, double *pn, double *pnp1, double limit );

void reinitialize( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                   Vec& phi_petsc, int number_of_iteration=100, double limit=DBL_MAX );


class my_p4est_level_set {

private:
    p4est_t *p4est;
    p4est_nodes_t *nodes;
    my_p4est_node_neighbors_t *ngbd;

    /* order the nodes based on whether they are in another mpirank's ghost layer or not */
    std::vector<p4est_locidx_t> layer_nodes;
    std::vector<p4est_locidx_t> local_nodes;

    void reinitialize_One_Iteration( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit );
public:
    my_p4est_level_set( p4est_t *p4est_, p4est_nodes_t *nodes_, my_p4est_node_neighbors_t *ngbd_ )
        : p4est(p4est_), nodes(nodes_), ngbd(ngbd_)
    {
        for( p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n )
        {
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n+nodes->offset_owned_indeps);
            p4est_locidx_t quad_mm_idx; p4est_locidx_t quad_mp_idx; p4est_locidx_t quad_pm_idx; p4est_locidx_t quad_pp_idx;
            p4est_topidx_t tree_mm_idx; p4est_topidx_t tree_mp_idx; p4est_topidx_t tree_pm_idx; p4est_topidx_t tree_pp_idx;

            /* find the neighbor cells of the node */
            ngbd->find_neighbor_cell_of_node( node, -1, -1, quad_mm_idx, tree_mm_idx);
            ngbd->find_neighbor_cell_of_node( node, -1,  1, quad_mp_idx, tree_mp_idx);
            ngbd->find_neighbor_cell_of_node( node,  1, -1, quad_pm_idx, tree_pm_idx);
            ngbd->find_neighbor_cell_of_node( node,  1,  1, quad_pp_idx, tree_pp_idx);

            /* check if node is in the ghost layer */
            if(  quad_mm_idx >= p4est->local_num_quadrants || quad_mp_idx >= p4est->local_num_quadrants ||
                 quad_pm_idx >= p4est->local_num_quadrants || quad_pp_idx >= p4est->local_num_quadrants )
//                 (*ngbd)[n].node_m0_m < nodes->offset_owned_indeps || (*ngbd)[n].node_m0_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_m0_p < nodes->offset_owned_indeps || (*ngbd)[n].node_m0_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_p0_m < nodes->offset_owned_indeps || (*ngbd)[n].node_p0_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_p0_p < nodes->offset_owned_indeps || (*ngbd)[n].node_p0_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_0m_m < nodes->offset_owned_indeps || (*ngbd)[n].node_0m_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_0m_p < nodes->offset_owned_indeps || (*ngbd)[n].node_0m_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_0p_m < nodes->offset_owned_indeps || (*ngbd)[n].node_0p_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
//                 (*ngbd)[n].node_0p_p < nodes->offset_owned_indeps || (*ngbd)[n].node_0p_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps )
                layer_nodes.push_back(n);
            else
                local_nodes.push_back(n);
        }
    }

    void reinitialize_1st_order( Vec &phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );

    void reinitialize_2nd_order( Vec &phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );
};
