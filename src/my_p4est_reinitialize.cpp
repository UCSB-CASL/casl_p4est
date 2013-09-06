#include "my_p4est_reinitialize.h"

void reinitialize_One_Iteration( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                                         double *p0, double *pn, double *pnp1, double limit )
{
    for( p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
    {
        p4est_locidx_t n_petsc = p4est2petsc_local_numbering(nodes,n+nodes->offset_owned_indeps);
        if(fabs(p0[n_petsc]) <= EPSILON)
            pnp1[n_petsc] = 0;
        else if(fabs(p0[n_petsc]) <= limit)
        {
            double p0_00, p0_m0, p0_p0, p0_0m, p0_0p ;
            double p_00 , p_m0 , p_p0 , p_0m , p_0p  ;
            (*node_neighbors)[n].ngbd_with_quadratic_interpolation(p0, p0_00, p0_m0, p0_p0, p0_0m, p0_0p);
            (*node_neighbors)[n].ngbd_with_quadratic_interpolation(pn, p_00 , p_m0 , p_p0 , p_0m , p_0p );
            double s_p0 = (*node_neighbors)[n].d_p0; double s_m0 = (*node_neighbors)[n].d_m0;
            double s_0p = (*node_neighbors)[n].d_0p; double s_0m = (*node_neighbors)[n].d_0m;

            //---------------------------------------------------------------------
            // check if the node is near interface
            //---------------------------------------------------------------------
            if ( (p0_00*p0_m0<0) || (p0_00*p0_p0<0) || (p0_00*p0_0m<0) || (p0_00*p0_0p<0) )
            {
                if(p0_00*p0_m0<0) { s_m0 = -interface_Location(-s_m0, 0, p0_m0, p0_00); p_m0 = 0; }
                if(p0_00*p0_p0<0) { s_p0 =  interface_Location( s_p0, 0, p0_p0, p0_00); p_p0 = 0; }
                if(p0_00*p0_0m<0) { s_0m = -interface_Location(-s_0m, 0, p0_0m, p0_00); p_0m = 0; }
                if(p0_00*p0_0p<0) { s_0p =  interface_Location( s_0p, 0, p0_0p, p0_00); p_0p = 0; }

                //                double p0x_00 = (*node_neighbors)[n].dx_central        (p0);
                //                double p0y_00 = (*node_neighbors)[n].dy_central        (p0);
                //                double p0x_m0 = (*node_neighbors)[n].dx_backward_linear(p0);
                //                double p0x_p0 = (*node_neighbors)[n].dx_forward_linear (p0);
                //                double p0y_0m = (*node_neighbors)[n].dy_backward_linear(p0);
                //                double p0y_0p = (*node_neighbors)[n].dy_forward_linear (p0);

                //                if(p0_00*p0_m0<0) { s_m0 = -interface_Location_With_First_Order_Derivative(-s_m0, 0, p0_m0, p0_00, p0x_m0, p0x_00); p_m0 = 0; }
                //                if(p0_00*p0_p0<0) { s_p0 =  interface_Location_With_First_Order_Derivative( s_p0, 0, p0_p0, p0_00, p0x_p0, p0x_00); p_p0 = 0; }
                //                if(p0_00*p0_0m<0) { s_0m = -interface_Location_With_First_Order_Derivative(-s_0m, 0, p0_0m, p0_00, p0y_0m, p0y_00); p_0m = 0; }
                //                if(p0_00*p0_0p<0) { s_0p =  interface_Location_With_First_Order_Derivative( s_0p, 0, p0_0p, p0_00, p0y_0p, p0y_00); p_0p = 0; }

                s_m0 = MAX(s_m0,EPSILON);
                s_p0 = MAX(s_p0,EPSILON);
                s_0m = MAX(s_0m,EPSILON);
                s_0p = MAX(s_0p,EPSILON);
            }

            //---------------------------------------------------------------------
            // Neumann boundary condition on the walls
            //---------------------------------------------------------------------
            /* first unclamp the node */
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n+nodes->offset_owned_indeps);
            p4est_indep_t unclamped_node = *node;
            p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

            /* wall in the x direction */
            if(unclamped_node.x==0)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
                if(nb_tree_idx == tree_idx) { s_m0 = s_p0; p_m0 = p_p0; }
            }
            else if(unclamped_node.x==P4EST_ROOT_LEN)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
                if(nb_tree_idx == tree_idx) { s_p0 = s_m0; p_p0 = p_m0; }
            }

            /* wall in the y direction */
            if(unclamped_node.y==0)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
                if(nb_tree_idx == tree_idx) { s_0m = s_0p; p_0m = p_0p; }
            }
            else if(unclamped_node.y==P4EST_ROOT_LEN)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
                if(nb_tree_idx == tree_idx) { s_0p = s_0m; p_0p = p_0m; }
            }

            //---------------------------------------------------------------------
            // First Order One-Sided Differecing
            //---------------------------------------------------------------------
            double px_p0 = (p_p0-p_00)/s_p0; double px_m0 = (p_00-p_m0)/s_m0;
            double py_0p = (p_0p-p_00)/s_0p; double py_0m = (p_00-p_0m)/s_0m;

            double sgn = (p0_00>0) ? 1 : -1;

            //---------------------------------------------------------------------
            // Upwind Scheme
            //---------------------------------------------------------------------
            double dt = MIN(s_m0,s_p0);
            dt = MIN(dt,s_0m);
            dt = MIN(dt,s_0p);
            dt = dt/2.;

            if(sgn>0) { if(px_p0>0) px_p0 = 0; if(px_m0<0) px_m0 = 0; if(py_0p>0) py_0p = 0; if(py_0m<0) py_0m = 0;}
            else      { if(px_p0<0) px_p0 = 0; if(px_m0>0) px_m0 = 0; if(py_0p<0) py_0p = 0; if(py_0m>0) py_0m = 0;}

            pnp1[n_petsc] = p_00 - dt*sgn*(sqrt( MAX(px_p0*px_p0 , px_m0*px_m0) + MAX( py_0p*py_0p , py_0m*py_0m ) ) - 1.);
            if(p0_00*pnp1[n_petsc]<0) pnp1[n_petsc] *= -1;
        }
        /* else : far away from the interface and not in the band ... nothing to do */
        else
            pnp1[n_petsc] = p0[n_petsc];
    }
}


void reinitialize( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                   Vec& phi_petsc, int number_of_iteration, double limit )
{
    PetscErrorCode ierr;
    double *phi;
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

    double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        p0[n] = phi[n];

    double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
    double *p2 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

    for(int i=0; i<number_of_iteration; i++)
    {

        if(i!=0) ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

        reinitialize_One_Iteration( p4est, nodes, node_neighbors, p0, phi, p1, limit);

        /* now copy the ghost layer */
        for( size_t n=0; n < nodes->indep_nodes.elem_count - nodes->num_owned_indeps; ++n)
            p1[n+nodes->num_owned_indeps] = phi[n+nodes->num_owned_indeps];

        reinitialize_One_Iteration( p4est, nodes, node_neighbors, p0, p1 , p2, limit);

        for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
        {
            p4est_locidx_t n_petsc = p4est2petsc_local_numbering(nodes,n+nodes->offset_owned_indeps);
            phi[n_petsc]=.5*(phi[n_petsc]+p2[n_petsc]);
        }

        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd  (phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    free(p0);
    free(p1);
    free(p2);
}



void my_p4est_level_set::reinitialize_One_Iteration( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit )
{
    for( size_t n_map=0; n_map<map.size(); ++n_map)
    {
        p4est_locidx_t n = map[n_map];

        p4est_locidx_t n_petsc = p4est2petsc_local_numbering(nodes, n + nodes->offset_owned_indeps);
        if(fabs(p0[n_petsc]) <= EPSILON)
            pnp1[n_petsc] = 0;
        else if(fabs(p0[n_petsc]) <= limit)
        {
            double p0_00, p0_m0, p0_p0, p0_0m, p0_0p ;
            double p_00 , p_m0 , p_p0 , p_0m , p_0p  ;
            (*ngbd)[n].ngbd_with_quadratic_interpolation(p0, p0_00, p0_m0, p0_p0, p0_0m, p0_0p);
            (*ngbd)[n].ngbd_with_quadratic_interpolation(pn, p_00 , p_m0 , p_p0 , p_0m , p_0p );
            double s_p0 = (*ngbd)[n].d_p0; double s_m0 = (*ngbd)[n].d_m0;
            double s_0p = (*ngbd)[n].d_0p; double s_0m = (*ngbd)[n].d_0m;

            //---------------------------------------------------------------------
            // check if the node is near interface
            //---------------------------------------------------------------------
            if ( (p0_00*p0_m0<0) || (p0_00*p0_p0<0) || (p0_00*p0_0m<0) || (p0_00*p0_0p<0) )
            {
                if(p0_00*p0_m0<0) { s_m0 = -interface_Location(-s_m0, 0, p0_m0, p0_00); p_m0 = 0; }
                if(p0_00*p0_p0<0) { s_p0 =  interface_Location( s_p0, 0, p0_p0, p0_00); p_p0 = 0; }
                if(p0_00*p0_0m<0) { s_0m = -interface_Location(-s_0m, 0, p0_0m, p0_00); p_0m = 0; }
                if(p0_00*p0_0p<0) { s_0p =  interface_Location( s_0p, 0, p0_0p, p0_00); p_0p = 0; }

                //                double p0x_00 = (*node_neighbors)[n].dx_central        (p0);
                //                double p0y_00 = (*node_neighbors)[n].dy_central        (p0);
                //                double p0x_m0 = (*node_neighbors)[n].dx_backward_linear(p0);
                //                double p0x_p0 = (*node_neighbors)[n].dx_forward_linear (p0);
                //                double p0y_0m = (*node_neighbors)[n].dy_backward_linear(p0);
                //                double p0y_0p = (*node_neighbors)[n].dy_forward_linear (p0);

                //                if(p0_00*p0_m0<0) { s_m0 = -interface_Location_With_First_Order_Derivative(-s_m0, 0, p0_m0, p0_00, p0x_m0, p0x_00); p_m0 = 0; }
                //                if(p0_00*p0_p0<0) { s_p0 =  interface_Location_With_First_Order_Derivative( s_p0, 0, p0_p0, p0_00, p0x_p0, p0x_00); p_p0 = 0; }
                //                if(p0_00*p0_0m<0) { s_0m = -interface_Location_With_First_Order_Derivative(-s_0m, 0, p0_0m, p0_00, p0y_0m, p0y_00); p_0m = 0; }
                //                if(p0_00*p0_0p<0) { s_0p =  interface_Location_With_First_Order_Derivative( s_0p, 0, p0_0p, p0_00, p0y_0p, p0y_00); p_0p = 0; }

                s_m0 = MAX(s_m0,EPSILON);
                s_p0 = MAX(s_p0,EPSILON);
                s_0m = MAX(s_0m,EPSILON);
                s_0p = MAX(s_0p,EPSILON);
            }

            //---------------------------------------------------------------------
            // Neumann boundary condition on the walls
            //---------------------------------------------------------------------
            /* first unclamp the node */
            p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n+nodes->offset_owned_indeps);
            p4est_indep_t unclamped_node = *node;
            p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

            /* wall in the x direction */
            if(unclamped_node.x==0)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
                if(nb_tree_idx == tree_idx) { s_m0 = s_p0; p_m0 = p_p0; }
            }
            else if(unclamped_node.x==P4EST_ROOT_LEN)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
                if(nb_tree_idx == tree_idx) { s_p0 = s_m0; p_p0 = p_m0; }
            }

            /* wall in the y direction */
            if(unclamped_node.y==0)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
                if(nb_tree_idx == tree_idx) { s_0m = s_0p; p_0m = p_0p; }
            }
            else if(unclamped_node.y==P4EST_ROOT_LEN)
            {
                p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
                p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
                if(nb_tree_idx == tree_idx) { s_0p = s_0m; p_0p = p_0m; }
            }

            //---------------------------------------------------------------------
            // First Order One-Sided Differecing
            //---------------------------------------------------------------------
            double px_p0 = (p_p0-p_00)/s_p0; double px_m0 = (p_00-p_m0)/s_m0;
            double py_0p = (p_0p-p_00)/s_0p; double py_0m = (p_00-p_0m)/s_0m;

            double sgn = (p0_00>0) ? 1 : -1;

            //---------------------------------------------------------------------
            // Upwind Scheme
            //---------------------------------------------------------------------
            double dt = MIN(s_m0,s_p0);
            dt = MIN(dt,s_0m);
            dt = MIN(dt,s_0p);
            dt = dt/2.;

            if(sgn>0) { if(px_p0>0) px_p0 = 0; if(px_m0<0) px_m0 = 0; if(py_0p>0) py_0p = 0; if(py_0m<0) py_0m = 0;}
            else      { if(px_p0<0) px_p0 = 0; if(px_m0>0) px_m0 = 0; if(py_0p<0) py_0p = 0; if(py_0m>0) py_0m = 0;}

            pnp1[n_petsc] = p_00 - dt*sgn*(sqrt( MAX(px_p0*px_p0 , px_m0*px_m0) + MAX( py_0p*py_0p , py_0m*py_0m ) ) - 1.);
            if(p0_00*pnp1[n_petsc]<0) pnp1[n_petsc] *= -1;
        }
        /* else : far away from the interface and not in the band ... nothing to do */
        else
            pnp1[n_petsc] = p0[n_petsc];
    }
}


void my_p4est_level_set::reinitialize_1st_order( Vec &phi_petsc, int number_of_iteration, double limit )
{
    PetscErrorCode ierr;
    double *phi;

    double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
    double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        p0[n] = phi[n];
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    for(int i=0; i<number_of_iteration; i++)
    {
        /* process the local nodes */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        reinitialize_One_Iteration( local_nodes, p0, phi, p1, limit);
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        /* finish receiving the ghost layer */
        if(i!=0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

        /* processes the layer nodes */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        reinitialize_One_Iteration( layer_nodes, p0, phi, p1, limit);
        for(size_t n=0; n<nodes->num_owned_indeps; ++n)
            phi[n] = p1[n];
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);


        /* initiate the communication for the ghost layer */
        ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    /* synchronize */
    ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    free(p0);
    free(p1);
}






void my_p4est_level_set::reinitialize_2nd_order( Vec &phi_petsc, int number_of_iteration, double limit )
{
    /* let's call
     * Ln the local nodes at time n
     * Bnp1 the boundary nodes at time np1
     * Gnm1 the ghost nodes at time nm1
     */
    PetscErrorCode ierr;
    double *phi;

    double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
    double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
    double *p2 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        p0[n] = phi[n];
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    for(int i=0; i<number_of_iteration; i++)
    {
        /* p1(Ln, Gn, Gnm1) */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
            p1[n] = phi[n];

        /* process the local nodes */
        /* phi(Lnp1, Bn, Gnm1) */
        reinitialize_One_Iteration( local_nodes, p0, p1, phi, limit);
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        /* finish receiving the ghost layer */
        /* phi(Lnp1, Bn, Gn) */
        if(i!=0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

        /* processes the layer nodes */
        /* p1(Ln, Bn, Gn) */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        for(size_t n=nodes->num_owned_indeps; n<nodes->indep_nodes.elem_count; ++n)
            p1[n] = phi[n];
        /* phi(Lnp1, Bnp1, Gn) */
        reinitialize_One_Iteration( layer_nodes, p0, p1, phi, limit);
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        /* initiate the communication for the ghost layer */
        ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);




        /* process the local nodes */
        /* p2(Lnp2, Bnp1, Gn) */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        reinitialize_One_Iteration( local_nodes, p0, phi, p2, limit);
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        /* finish receiving the ghost layer */
        /* phi(Lnp1, Bnp1, Gnp1) */
        ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        /* process the layer nodes */
        ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
        /* p2(Lnp2, Bnp2, Gn) */
        reinitialize_One_Iteration( layer_nodes, p0, phi, p2, limit);

        /* update phi */
        for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
            phi[n] = .5 * (p1[n] + p2[n]);
        ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

        /* initiate the communication for the ghost layer */
        ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    /* synchronize */
    if(number_of_iteration>0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    free(p0);
    free(p1);
    free(p2);
}
