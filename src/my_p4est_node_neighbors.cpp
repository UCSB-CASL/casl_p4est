#include "my_p4est_node_neighbors.h"



void my_p4est_node_neighbors_t::init_neighbors()
{
    for( p4est_locidx_t n=0; n < nodes->num_owned_indeps; ++n)
    {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,nodes->offset_owned_indeps+n);

        double x = node->x / (double) P4EST_ROOT_LEN;
        double y = node->y / (double) P4EST_ROOT_LEN;
        c2p_coordinate_transform( p4est, node->p.piggy3.which_tree, &x, &y, NULL );

        p4est_locidx_t quad_mm_idx; p4est_topidx_t tree_mm_idx; find_neighbor_cell_of_node(node,-1,-1, quad_mm_idx, tree_mm_idx);
        p4est_locidx_t quad_mp_idx; p4est_topidx_t tree_mp_idx; find_neighbor_cell_of_node(node,-1, 1, quad_mp_idx, tree_mp_idx);
        p4est_locidx_t quad_pm_idx; p4est_topidx_t tree_pm_idx; find_neighbor_cell_of_node(node, 1,-1, quad_pm_idx, tree_pm_idx);
        p4est_locidx_t quad_pp_idx; p4est_topidx_t tree_pp_idx; find_neighbor_cell_of_node(node, 1, 1, quad_pp_idx, tree_pp_idx);

        /* create dummy root quadrant */
        p4est_quadrant_t root;
        root.level = -1; root.x = 0; root.y = 0;

        /* fetch the quadrants */
        p4est_quadrant_t *quad_mm;
        p4est_quadrant_t *quad_mp;
        p4est_quadrant_t *quad_pm;
        p4est_quadrant_t *quad_pp;

        if(quad_mm_idx == -1)
            quad_mm = &root;
        else
        {
            if(quad_mm_idx<p4est->local_num_quadrants)
            {
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mm_idx);
                quad_mm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mm_idx-tree->quadrants_offset);
            }
            else /* in the ghost layer */
                quad_mm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mm_idx-p4est->local_num_quadrants);
        }

        if(quad_mp_idx == -1)
            quad_mp = &root;
        else
        {
            if(quad_mp_idx<p4est->local_num_quadrants)
            {
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_mp_idx);
                quad_mp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_mp_idx-tree->quadrants_offset);
            }
            else /* in the ghost layer */
                quad_mp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_mp_idx-p4est->local_num_quadrants);
        }

        if(quad_pm_idx == -1)
            quad_pm = &root;
        else
        {
            if(quad_pm_idx<p4est->local_num_quadrants)
            {
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_pm_idx);
                quad_pm = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_pm_idx-tree->quadrants_offset);
            }
            else /* in the ghost layer */
                quad_pm = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_pm_idx-p4est->local_num_quadrants);
        }

        if(quad_pp_idx == -1)
            quad_pp = &root;
        else
        {
            if(quad_pp_idx<p4est->local_num_quadrants)
            {
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tree_pp_idx);
                quad_pp = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,quad_pp_idx-tree->quadrants_offset);
            }
            else /* in the ghost layer */
            {
                quad_pp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_pp_idx-p4est->local_num_quadrants);
            }
        }

        neighbors[n].nodes = nodes;
        neighbors[n].node_00 = n + nodes->offset_owned_indeps;

        /* m0 */
        if(quad_mm!=&root || quad_mp!=&root)
        {
            p4est_quadrant_t *quad_m0;
            p4est_locidx_t quad_m0_idx;
            p4est_topidx_t tree_m0_idx;
            if(quad_mm->level < quad_mp->level) { quad_m0_idx = quad_mp_idx; quad_m0 = quad_mp; tree_m0_idx = tree_mp_idx; }
            else                                { quad_m0_idx = quad_mm_idx; quad_m0 = quad_mm; tree_m0_idx = tree_mm_idx; }
            neighbors[n].d_m0 = P4EST_QUADRANT_LEN(quad_m0->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_m0_m = nodes->local_nodes[P4EST_CHILDREN*quad_m0_idx + 0];
            neighbors[n].node_m0_p = nodes->local_nodes[P4EST_CHILDREN*quad_m0_idx + 2];

            double qx = quad_m0->x / (double) P4EST_ROOT_LEN;
            double qy = quad_m0->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_m0_idx, &qx, &qy, NULL );
            neighbors[n].d_m0_m = y - qy;

            qx = quad_m0->x / (double) P4EST_ROOT_LEN;
            qy = ( quad_m0->y + P4EST_QUADRANT_LEN(quad_m0->level) ) / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_m0_idx, &qx, &qy, NULL );
            neighbors[n].d_m0_p = qy - y;
        }

        /* p0 */
        if(quad_pm!=&root || quad_pp!=&root)
        {
            p4est_quadrant_t *quad_p0;
            p4est_locidx_t quad_p0_idx;
            p4est_topidx_t tree_p0_idx;
            if(quad_pm->level < quad_pp->level) { quad_p0_idx = quad_pp_idx; quad_p0 = quad_pp; tree_p0_idx = tree_pp_idx; }
            else                                { quad_p0_idx = quad_pm_idx; quad_p0 = quad_pm; tree_p0_idx = tree_pm_idx; }
            neighbors[n].d_p0 = P4EST_QUADRANT_LEN(quad_p0->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_p0_m = nodes->local_nodes[P4EST_CHILDREN*quad_p0_idx + 1];
            neighbors[n].node_p0_p = nodes->local_nodes[P4EST_CHILDREN*quad_p0_idx + 3];

            double qx = quad_p0->x / (double) P4EST_ROOT_LEN;
            double qy = quad_p0->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_p0_idx, &qx, &qy, NULL );
            neighbors[n].d_p0_m = y - qy;

            qx = quad_p0->x / (double) P4EST_ROOT_LEN;
            qy = ( quad_p0->y + P4EST_QUADRANT_LEN(quad_p0->level) ) / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_p0_idx, &qx, &qy, NULL );
            neighbors[n].d_p0_p = qy - y;
        }

        /* 0m */
        if(quad_mm!=&root || quad_pm!=&root)
        {
            p4est_quadrant_t *quad_0m;
            p4est_locidx_t quad_0m_idx;
            p4est_topidx_t tree_0m_idx;
            if(quad_mm->level < quad_pm->level) { quad_0m_idx = quad_pm_idx; quad_0m = quad_pm; tree_0m_idx = tree_pm_idx; }
            else                                { quad_0m_idx = quad_mm_idx; quad_0m = quad_mm; tree_0m_idx = tree_mm_idx; }
            neighbors[n].d_0m = P4EST_QUADRANT_LEN(quad_0m->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_0m_m = nodes->local_nodes[P4EST_CHILDREN*quad_0m_idx + 0];
            neighbors[n].node_0m_p = nodes->local_nodes[P4EST_CHILDREN*quad_0m_idx + 1];

            double qx = quad_0m->x / (double) P4EST_ROOT_LEN;
            double qy = quad_0m->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_0m_idx, &qx, &qy, NULL );
            neighbors[n].d_0m_m = x - qx;

            qx = ( quad_0m->x + P4EST_QUADRANT_LEN(quad_0m->level) ) / (double) P4EST_ROOT_LEN;
            qy = quad_0m->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_0m_idx, &qx, &qy, NULL );
            neighbors[n].d_0m_p = qx - x;
        }

        /* 0p */
        if(quad_mp!=&root || quad_pp!=&root)
        {
            p4est_quadrant_t *quad_0p;
            p4est_locidx_t quad_0p_idx;
            p4est_topidx_t tree_0p_idx;
            if(quad_mp->level < quad_pp->level) { quad_0p_idx = quad_pp_idx; quad_0p = quad_pp; tree_0p_idx = tree_pp_idx; }
            else                                { quad_0p_idx = quad_mp_idx; quad_0p = quad_mp; tree_0p_idx = tree_mp_idx; }
            neighbors[n].d_0p = P4EST_QUADRANT_LEN(quad_0p->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_0p_m = nodes->local_nodes[P4EST_CHILDREN*quad_0p_idx + 2];
            neighbors[n].node_0p_p = nodes->local_nodes[P4EST_CHILDREN*quad_0p_idx + 3];

            double qx = quad_0p->x / (double) P4EST_ROOT_LEN;
            double qy = quad_0p->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_0p_idx, &qx, &qy, NULL );
            neighbors[n].d_0p_m = x - qx;

            qx = ( quad_0p->x + P4EST_QUADRANT_LEN(quad_0p->level) ) / (double) P4EST_ROOT_LEN;
            qy = quad_0p->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_0p_idx, &qx, &qy, NULL );
            neighbors[n].d_0p_p = qx - x;
        }

        /* now do the special case when the node is on an edge of the domain, i.e. 2 roots in this direction */
        if(quad_mm==&root && quad_mp==&root)
        {
            /* fetch the second order neighbor to the right */
            p4est_indep_t *node_tmp;
            p4est_locidx_t quad_tmp_idx;
            p4est_topidx_t tree_tmp_idx;
            if(quad_pm->level < quad_pp->level)
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_pp_idx + 1] );
                find_neighbor_cell_of_node( node_tmp, 1, 1, quad_tmp_idx, tree_tmp_idx );
            }
            else
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_pm_idx + 3] );
                find_neighbor_cell_of_node( node_tmp, 1, -1, quad_tmp_idx, tree_tmp_idx );
            }

            p4est_quadrant_t *quad_tmp;
            if(quad_tmp_idx < p4est->local_num_quadrants)
            {
                p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
            }
            else
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

            neighbors[n].d_m0 = - neighbors[n].d_p0 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_m0_m = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 1];
            neighbors[n].node_m0_p = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 3];

            double qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            double qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_m0_m = y - qy;

            qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            qy = ( quad_tmp->y + P4EST_QUADRANT_LEN(quad_tmp->level) ) / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_m0_p = qy - y;
        }

        if(quad_pm==&root && quad_pp==&root)
        {
            /* fetch the second order neighbor to the right */
            p4est_indep_t *node_tmp;
            p4est_locidx_t quad_tmp_idx;
            p4est_topidx_t tree_tmp_idx;
            if(quad_mm->level < quad_mp->level)
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_mp_idx + 0] );
                find_neighbor_cell_of_node( node_tmp, -1, 1, quad_tmp_idx, tree_tmp_idx );
            }
            else
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_mm_idx + 2] );
                find_neighbor_cell_of_node( node_tmp, -1, -1, quad_tmp_idx, tree_tmp_idx );
            }

            p4est_quadrant_t *quad_tmp;
            if(quad_tmp_idx < p4est->local_num_quadrants)
            {
                p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
            }
            else
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

            neighbors[n].d_p0 = - neighbors[n].d_m0 - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_p0_m = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 0];
            neighbors[n].node_p0_p = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 2];

            double qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            double qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_p0_m = y - qy;

            qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            qy = ( quad_tmp->y + P4EST_QUADRANT_LEN(quad_tmp->level) ) / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_p0_p = qy - y;
        }

        if(quad_mm==&root && quad_pm==&root)
        {
            /* fetch the second order neighbor to the right */
            p4est_indep_t *node_tmp;
            p4est_locidx_t quad_tmp_idx;
            p4est_topidx_t tree_tmp_idx;
            if(quad_mp->level < quad_pp->level)
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_pp_idx + 2] );
                find_neighbor_cell_of_node( node_tmp, 1, 1, quad_tmp_idx, tree_tmp_idx );
            }
            else
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_mp_idx + 3] );
                find_neighbor_cell_of_node( node_tmp, -1, 1, quad_tmp_idx, tree_tmp_idx );
            }

            p4est_quadrant_t *quad_tmp;
            if(quad_tmp_idx < p4est->local_num_quadrants)
            {
                p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
            }
            else
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

            neighbors[n].d_0m = - neighbors[n].d_0p - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_0m_m = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 2];
            neighbors[n].node_0m_p = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 3];

            double qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            double qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_0m_m = x - qx;

            qx = ( quad_tmp->x + P4EST_QUADRANT_LEN(quad_tmp->level) ) / (double) P4EST_ROOT_LEN;
            qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_0m_p = qx - x;
        }

        if(quad_mp==&root && quad_pp==&root)
        {
            /* fetch the second order neighbor to the right */
            p4est_indep_t *node_tmp;
            p4est_locidx_t quad_tmp_idx;
            p4est_topidx_t tree_tmp_idx;
            if(quad_mm->level < quad_pm->level)
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_pm_idx + 0] );
                find_neighbor_cell_of_node( node_tmp, 1, -1, quad_tmp_idx, tree_tmp_idx );
            }
            else
            {
                node_tmp = (p4est_indep_t*)sc_array_index( &nodes->indep_nodes, nodes->local_nodes[P4EST_CHILDREN*quad_mm_idx + 1] );
                find_neighbor_cell_of_node( node_tmp, -1, -1, quad_tmp_idx, tree_tmp_idx );
            }

            p4est_quadrant_t *quad_tmp;
            if(quad_tmp_idx < p4est->local_num_quadrants)
            {
                p4est_tree_t *tree_tmp = (p4est_tree_t*)sc_array_index(p4est->trees,tree_tmp_idx);
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&tree_tmp->quadrants,quad_tmp_idx-tree_tmp->quadrants_offset);
            }
            else
                quad_tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts,quad_tmp_idx-p4est->local_num_quadrants);

            neighbors[n].d_0p = - neighbors[n].d_0m - P4EST_QUADRANT_LEN(quad_tmp->level) / (double) P4EST_ROOT_LEN;
            neighbors[n].node_0p_m = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 0];
            neighbors[n].node_0p_p = nodes->local_nodes[P4EST_CHILDREN*quad_tmp_idx + 1];

            double qx = quad_tmp->x / (double) P4EST_ROOT_LEN;
            double qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_0p_m = x - qx;

            qx = ( quad_tmp->x  + P4EST_QUADRANT_LEN(quad_tmp->level) ) / (double) P4EST_ROOT_LEN;
            qy = quad_tmp->y / (double) P4EST_ROOT_LEN;
            c2p_coordinate_transform( p4est, tree_tmp_idx, &qx, &qy, NULL );
            neighbors[n].d_0p_p = qx - x;
        }
    }

}


void my_p4est_node_neighbors_t::find_neighbor_cell_of_node( p4est_indep_t *node_, char i, char j, p4est_locidx_t& quad, p4est_topidx_t& nb_tree_idx )
{
    p4est_indep_t node_struct = *node_;
    p4est_indep_t *node = &node_struct;
    p4est_node_unclamp((p4est_quadrant_t*)node);

    p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
    nb_tree_idx = tree_idx;

    p4est_qcoord_t x_perturb = node->x+i;
    p4est_qcoord_t y_perturb = node->y+j;

    /* first check the corners of the tree */
    if(node->x==0 && node->y==0 && i==-1 && j==-1)
    {
        p4est_topidx_t tmp_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(tmp_tree_idx == tree_idx) { quad = -1; return; }
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
        if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
        x_perturb = P4EST_ROOT_LEN - 1;
        y_perturb = P4EST_ROOT_LEN - 1;
    }
    else if(node->x==P4EST_ROOT_LEN && node->y==0 && i== 1 && j==-1)
    {
        p4est_topidx_t tmp_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(tmp_tree_idx == tree_idx) { quad = -1; return; }
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 2];
        if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
        x_perturb = 1;
        y_perturb = P4EST_ROOT_LEN - 1;
    }
    else if(node->x==0 && node->y==P4EST_ROOT_LEN && i==-1 && j== 1)
    {
        p4est_topidx_t tmp_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(tmp_tree_idx == tree_idx) { quad = -1; return; }
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
        if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
        x_perturb = P4EST_ROOT_LEN - 1;
        y_perturb = 1;
    }
    else if(node->x==P4EST_ROOT_LEN && node->y==P4EST_ROOT_LEN && i== 1 && j== 1)
    {
        p4est_topidx_t tmp_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(tmp_tree_idx == tree_idx) { quad = -1; return; }
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tmp_tree_idx + 3];
        if(nb_tree_idx == tmp_tree_idx) { quad = -1; return; }
        x_perturb = 1;
        y_perturb = 1;
    }

    /* now check the edges of the tree */
    else if(node->x==0 && i==-1)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(nb_tree_idx == tree_idx) { quad = -1; return; }
        x_perturb = P4EST_ROOT_LEN - 1;
    }
    else if(node->x==P4EST_ROOT_LEN && i==1)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(nb_tree_idx == tree_idx) { quad = -1; return; }
        x_perturb = 1;
    }
    else if(node->y==0 && j==-1)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
        if(nb_tree_idx == tree_idx) { quad = -1; return; }
        y_perturb = P4EST_ROOT_LEN - 1;
    }
    else if(node->y==P4EST_ROOT_LEN && j==1)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
        if(nb_tree_idx == tree_idx) { quad = -1; return; }
        y_perturb = 1;
    }

    int ind = 0;
    while(hierarchy->trees[nb_tree_idx][ind].child!=CELL_LEAF)
    {
        p4est_qcoord_t size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
        bool dir_i = ( x_perturb >= hierarchy->trees[nb_tree_idx][ind].imin + size );
        bool dir_j = ( y_perturb >= hierarchy->trees[nb_tree_idx][ind].jmin + size );
        ind = hierarchy->trees[nb_tree_idx][ind].child + 2*dir_j + dir_i;
    }

    quad = hierarchy->trees[nb_tree_idx][ind].quad;
}
