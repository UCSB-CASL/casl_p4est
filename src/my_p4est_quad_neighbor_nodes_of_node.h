#ifndef MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H
#define MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H

#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <petscvec.h>

//---------------------------------------------------------------------
//
// quad_neighbor_nodes_of_nodes_t : neighborhood of a node in Quadtree
//
//        node_m0_p                         node_p0_p
//             |                                 |
//      d_m0_p |                                 | d_p0_p
//             |                                 |
//             --- d_m0 ----- node_00 --- d_p0 ---
//             |                                 |
//      d_m0_m |                                 | d_p0_m
//             |                                 |
//        node_m0_m                         node_p0_m
//
//---------------------------------------------------------------------

struct quad_neighbor_nodes_of_node_t {
    p4est_nodes_t *nodes;

    /* store the local index for the neighbor nodes
     * if a node is local, then its index is between offset_owned_indeps  and  offset_owned_indeps + num_owned_indeps.
     */
    p4est_locidx_t node_00;
    p4est_locidx_t node_m0_m; p4est_locidx_t node_m0_p;
    p4est_locidx_t node_p0_m; p4est_locidx_t node_p0_p;
    p4est_locidx_t node_0m_m; p4est_locidx_t node_0m_p;
    p4est_locidx_t node_0p_m; p4est_locidx_t node_0p_p;

    double d_m0; double d_m0_m; double d_m0_p;
    double d_p0; double d_p0_m; double d_p0_p;
    double d_0m; double d_0m_m; double d_0m_p;
    double d_0p; double d_0p_m; double d_0p_p;

    double f_m0_linear( const double *f ) const;
    double f_p0_linear( const double *f ) const;
    double f_0m_linear( const double *f ) const;
    double f_0p_linear( const double *f ) const;

    void ngbd_with_quadratic_interpolation( const double *f, double& f_00,
                                            double& f_m0,
                                            double& f_p0,
                                            double& f_0m,
                                            double& f_0p) const;

    void x_ngbd_with_quadratic_interpolation( const double *f, double& f_m0,
                                              double& f_00,
                                              double& f_p0) const;

    void y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m,
                                              double& f_00,
                                              double& f_0p) const;

    double dx_central ( const double *f ) const;
    double dy_central ( const double *f ) const;

    double dx_forward_linear ( const double *f ) const;
    double dx_backward_linear( const double *f ) const;
    double dy_forward_linear ( const double *f ) const;
    double dy_backward_linear( const double *f ) const;

    void blah(p4est_nodes_t *nodes) const
    {
        p4est_indep_t *n_m0_m = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m0_m);
        p4est_indep_t *n_m0_p = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m0_p);
        p4est_indep_t *n_p0_m = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p0_m);
        p4est_indep_t *n_p0_p = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p0_p);
        p4est_indep_t *n_0m_m = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m_m);
        p4est_indep_t *n_0m_p = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m_p);
        p4est_indep_t *n_0p_m = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p_m);
        p4est_indep_t *n_0p_p = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p_p);
        printf("------------- Printing QNNN for node %d ------------\n",node_00);
        printf("node_m0_m : %d - ( %f , %f )  -  %f\n",node_m0_m,n_m0_m->x / (double) P4EST_ROOT_LEN,n_m0_m->y / (double) P4EST_ROOT_LEN,d_m0_m);
        printf("node_m0_p : %d - ( %f , %f )  -  %f\n",node_m0_p,n_m0_p->x / (double) P4EST_ROOT_LEN,n_m0_p->y / (double) P4EST_ROOT_LEN,d_m0_p);
        printf("node_p0_m : %d - ( %f , %f )  -  %f\n",node_p0_m,n_p0_m->x / (double) P4EST_ROOT_LEN,n_p0_m->y / (double) P4EST_ROOT_LEN,d_p0_m);
        printf("node_p0_p : %d - ( %f , %f )  -  %f\n",node_p0_p,n_p0_p->x / (double) P4EST_ROOT_LEN,n_p0_p->y / (double) P4EST_ROOT_LEN,d_p0_p);
        printf("node_0m_m : %d - ( %f , %f )  -  %f\n",node_0m_m,n_0m_m->x / (double) P4EST_ROOT_LEN,n_0m_m->y / (double) P4EST_ROOT_LEN,d_0m_m);
        printf("node_0m_p : %d - ( %f , %f )  -  %f\n",node_0m_p,n_0m_p->x / (double) P4EST_ROOT_LEN,n_0m_p->y / (double) P4EST_ROOT_LEN,d_0m_p);
        printf("node_0p_m : %d - ( %f , %f )  -  %f\n",node_0p_m,n_0p_m->x / (double) P4EST_ROOT_LEN,n_0p_m->y / (double) P4EST_ROOT_LEN,d_0p_m);
        printf("node_0p_p : %d - ( %f , %f )  -  %f\n",node_0p_p,n_0p_p->x / (double) P4EST_ROOT_LEN,n_0p_p->y / (double) P4EST_ROOT_LEN,d_0p_p);
        printf("d_m0 : %f\nd_p0 : %f\nd_0m : %f\nd_0p : %f\n",d_m0,d_p0,d_0m,d_0p);
    }
};

#endif /* !MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H */
