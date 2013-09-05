#include "my_p4est_cell_neighbors.h"

void my_p4est_cell_neighbors_t::initialize_neighbors()
{
    for( p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx )
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q )
            for(int dir = 0; dir<2*P4EST_DIM; ++dir )
                find_neighbor_cells_of_cell(q,tree_idx, dir);
    }
}


void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell( p4est_locidx_t q, p4est_topidx_t tree_idx, int dir )
{
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);

    /* construct the coordinate of the neighbor cell of the same size in the given direction */
    p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

    p4est_topidx_t nb_tree_idx = tree_idx;
    p4est_qcoord_t i_nb = quad->x + ( dir==0 ? -size : ( dir==1 ? size : 0) );
    p4est_qcoord_t j_nb = quad->y + ( dir==2 ? -size : ( dir==3 ? size : 0) );

    /* check if quadrant is on a boundary */
    if(quad->x==0 && dir==0)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(nb_tree_idx==tree_idx) return;
        i_nb = P4EST_ROOT_LEN - size;
    }
    else if(quad->x+size==P4EST_ROOT_LEN && dir==1)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(nb_tree_idx==tree_idx) return;
        i_nb = 0;
    }
    else if(quad->y==0 && dir==2)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
        if(nb_tree_idx==tree_idx) return;
        j_nb = P4EST_ROOT_LEN - size;
    }
    else if(quad->y+size==P4EST_ROOT_LEN && dir==3)
    {
        nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
        if(nb_tree_idx==tree_idx) return;
        j_nb = 0;
    }

    /* find the constructed neighbor cell of the same size */
    int ind = 0;
    while( hierarchy->trees[nb_tree_idx][ind].level != quad->level && hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF )
    {
        p4est_qcoord_t half_size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
        bool i_search = ( i_nb >= hierarchy->trees[nb_tree_idx][ind].imin + half_size );
        bool j_search = ( j_nb >= hierarchy->trees[nb_tree_idx][ind].jmin + half_size );
        ind = hierarchy->trees[nb_tree_idx][ind].child + 2*j_search + i_search;
    }

    /* now find the children of this constructed cell in the desired direction and add them to the list */
    find_neighbor_cells_of_cell_recursive( tree->quadrants_offset+q, nb_tree_idx, ind, dir );
}


void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive( p4est_locidx_t q, p4est_topidx_t tr, int ind, int dir )
{

    if (hierarchy->trees[tr][ind].child==CELL_LEAF)
    {
        neighbor_cells[dir][q].push_back( hierarchy->trees[tr][ind].quad );
        return ;
    }

    switch(dir)
    {
    case 0:
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 1, dir);
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 3, dir);
        break;
    case 1:
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 0, dir);
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 2, dir);
        break;
    case 2:
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 2, dir);
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 3, dir);
        break;
    case 3:
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 0, dir);
        find_neighbor_cells_of_cell_recursive(q, tr, hierarchy->trees[tr][ind].child + 1, dir);
        break;
    }
}
