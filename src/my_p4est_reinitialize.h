#include <src/CASL_math.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>

#include <vector>

class my_p4est_level_set {

private:
    p4est_t *p4est;
    p4est_nodes_t *nodes;
    my_p4est_node_neighbors_t *ngbd;

    /* order the nodes based on whether they are in another mpirank's ghost layer or not */
    std::vector<p4est_locidx_t> layer_nodes;
    std::vector<p4est_locidx_t> local_nodes;

    void reinitialize_One_Iteration( std::vector<p4est_locidx_t>& map, const double *dx, const double *dy, double *p0, double *pn, double *pnp1, double limit );
public:
    my_p4est_level_set( p4est_t *p4est_, p4est_nodes_t *nodes_, my_p4est_node_neighbors_t *ngbd_ )
        : p4est(p4est_), nodes(nodes_), ngbd(ngbd_)
    {
        for( p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n )
        {
            if(  (*ngbd)[n].node_m0_m < nodes->offset_owned_indeps || (*ngbd)[n].node_m0_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_m0_p < nodes->offset_owned_indeps || (*ngbd)[n].node_m0_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_p0_m < nodes->offset_owned_indeps || (*ngbd)[n].node_p0_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_p0_p < nodes->offset_owned_indeps || (*ngbd)[n].node_p0_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_0m_m < nodes->offset_owned_indeps || (*ngbd)[n].node_0m_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_0m_p < nodes->offset_owned_indeps || (*ngbd)[n].node_0m_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_0p_m < nodes->offset_owned_indeps || (*ngbd)[n].node_0p_m >= nodes->offset_owned_indeps+nodes->num_owned_indeps ||
                 (*ngbd)[n].node_0p_p < nodes->offset_owned_indeps || (*ngbd)[n].node_0p_p >= nodes->offset_owned_indeps+nodes->num_owned_indeps )
                layer_nodes.push_back(n);
            else
                local_nodes.push_back(n);
        }
    }

    /* 1st order in time, 1st order in space */
    void reinitialize_1st_order( Vec &phi_petsc, int number_of_iteration=20, double limit=DBL_MAX );

    /* 2nd order in time, 1st order in space */
    void reinitialize_2nd_order( Vec &phi_petsc, int number_of_iteration=20, double limit=DBL_MAX );
};
