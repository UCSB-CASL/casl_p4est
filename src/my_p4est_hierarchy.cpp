#include "my_p4est_hierarchy.h"

void my_p4est_hierarchy_t::split( int tree_idx, int ind )
{
    trees[tree_idx][ind].child = trees[tree_idx].size();

    p4est_qcoord_t size = P4EST_QUADRANT_LEN(trees[tree_idx][ind].level) / 2;
    for(int j=0; j<2; ++j) {
        for(int i=0; i<2; ++i) {
            struct HierarchyCell child = { CELL_LEAF, NOT_A_P4EST_QUADRANT,
                        trees[tree_idx][ind].imin + i*size,
                        trees[tree_idx][ind].jmin + j*size,
                        trees[tree_idx][ind].level+1};
            trees[tree_idx].push_back(child);
        }
    }
}


int my_p4est_hierarchy_t::update_tree( int tree_idx, p4est_quadrant_t *quad )
{
    int ind = 0;
    while( trees[tree_idx][ind].level != quad->level )
    {
        if(trees[tree_idx][ind].child == CELL_LEAF)
            split(tree_idx, ind);

        /* now the intermediate cell is split, select the correct child */
        p4est_qcoord_t size = P4EST_QUADRANT_LEN(trees[tree_idx][ind].level) / 2;
        bool i = ( quad->x >= trees[tree_idx][ind].imin + size );
        bool j = ( quad->y >= trees[tree_idx][ind].jmin + size );

        ind = trees[tree_idx][ind].child + 2*j + i;
    }
    return ind;
}

void my_p4est_hierarchy_t::construct_tree() {

    /* loop on the quadrants */
    for( p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
            p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            int ind = update_tree(tree_idx, quad);

            /* the cell corresponding to the quadrant has been found, associate it to the quadrant */
            trees[tree_idx][ind].quad = tree->quadrants_offset + q;
        }
    }

    /* loop on the ghosts */
    for( size_t g=0; g<ghost->ghosts.elem_count; ++g)
    {
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, g);
        int ind = update_tree(quad->p.piggy3.which_tree, quad);

        /* the cell corresponding to the quadrant has been found, associate it to the quadrant */
        trees[quad->p.piggy3.which_tree][ind].quad = p4est->local_num_quadrants + g;
    }
}

void my_p4est_hierarchy_t::write_vtk(const char* filename) const
{
    p4est_connectivity_t* connectivity = p4est->connectivity;

    /* filename */
    char vtkname[1024];
    sprintf(vtkname, "%s_%04d.vtk", filename, p4est->mpirank);

    FILE *vtk = fopen(vtkname, "w");

    fprintf(vtk, "# vtk DataFile Version 2.0 \n");
    fprintf(vtk, "Quadtree Mesh \n");
    fprintf(vtk, "ASCII \n");
    fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

    size_t num_quads = 0;
    for (size_t i=0; i<trees.size(); ++i){
        for (size_t j=0; j<trees[i].size(); j++){
            if (trees[i][j].child == CELL_LEAF)
                num_quads++;
        }
    }

    fprintf(vtk, "POINTS %ld double \n", P4EST_CHILDREN*num_quads);

    for (size_t i=0; i<trees.size(); ++i){
        p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*i + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

        for (size_t j=0; j<trees[i].size(); j++){
            const HierarchyCell& cell = trees[i][j];
            if (cell.child == CELL_LEAF){
                double h = (double) P4EST_QUADRANT_LEN(cell.level) / P4EST_ROOT_LEN;

                for (short xj=0; xj<2; xj++)
                    for (short xi=0; xi<2; xi++){
                        double x = (double) cell.imin / P4EST_ROOT_LEN + xi*h + tree_xmin;
                        double y = (double) cell.jmin / P4EST_ROOT_LEN + xj*h + tree_ymin;

                        fprintf(vtk, "%lf %lf 0.0\n", x, y);
                    }
            }
        }
    }

    fprintf(vtk, "CELLS %ld %ld \n", num_quads, (1+P4EST_CHILDREN)*num_quads);
    for (size_t i=0; i<num_quads; ++i)
    {
        fprintf(vtk, "%d ", P4EST_CHILDREN);
        for (short j=0; j<P4EST_CHILDREN; ++j)
            fprintf(vtk, "%ld ", P4EST_CHILDREN*i+j);
        fprintf(vtk,"\n");
    }

    fprintf(vtk, "CELL_TYPES %ld\n", num_quads);
    for (size_t i=0; i<num_quads; ++i)
        fprintf(vtk, "%d\n",P4EST_VTK_CELL_TYPE);
    fclose(vtk);
}
