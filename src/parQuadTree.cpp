#include "parQuadTree.h"

parQuadTree::parQuadTree(const p4est_t* p4est_)
  : p4est(p4est_)
{
  num_of_trees = p4est->last_local_tree - p4est->first_local_tree + 1;
  tr_array.reallocate(num_of_trees);
}

void parQuadTree::copyFromP4est()
{
  for (p4est_topidx_t tr_it = 0; tr_it < num_of_trees; ++tr_it)
  {
    p4est_topidx_t tr_id = tr_it+p4est->first_local_tree;

    // Get a referrence to current tree
    QuadTree& tr = tr_array(tr_it);

    // Set the computational domain boundaries
    double         *v   = p4est->connectivity->vertices;
    p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
    tr.set_grid(v[3*t2v[tr_id*P4EST_CHILDREN + 0] + 0], v[3*t2v[tr_id*P4EST_CHILDREN + 1] + 0],
                v[3*t2v[tr_id*P4EST_CHILDREN + 0] + 1], v[3*t2v[tr_id*P4EST_CHILDREN + 2] + 1]);

    p4est_tree_t *tree    = p4est_tree_array_index(p4est->trees, tr_id);
    sc_array_t *quadrants = &tree->quadrants;

    // Loop over local quadrants
    for (p4est_locidx_t qu = 0; qu<tree->quadrants.elem_count; ++qu){
      p4est_quadrant_t *quad = p4est_quadrant_array_index(quadrants, qu);

      double qh  = (double) P4EST_QUADRANT_LEN(quad->level);
      double qxc = ((double)quad->x + 0.5*qh)/((double) P4EST_ROOT_LEN);
      double qyc = ((double)quad->y + 0.5*qh)/((double) P4EST_ROOT_LEN);

      const QuadCell *cells = tr.get_cells();
      CaslInt c = 0;
      unsigned short level = 0;
      while (level++ < quad->level)
      {
        if (cells[c].is_leaf())
        {
          tr.split_cell(c);
          cells = tr.get_cells();
        }

        double xc = ((double) cells[c].icenter())/((double) MAX_NUMBER_OF_NODES_IN_ONE_DIRECTION);
        double yc = ((double) cells[c].jcenter())/((double) MAX_NUMBER_OF_NODES_IN_ONE_DIRECTION);

        unsigned short ci = qxc<xc ? 0:1;
        unsigned short cj = qyc<yc ? 0:1;

        c = cells[c].child_index(ci, cj);
      }
    }
  }
}

void parQuadTree::print()
{
  for (p4est_topidx_t tr_it = 0; tr_it < num_of_trees; ++tr_it)
  {
    ostringstream oss;
    oss << "Proc_" << p4est->mpirank;
    oss << "_tree_" << tr_it+p4est->first_local_tree;

    xdmfWriter xmf(oss.str());
    xmf.openTimeStep(0);
    xmf.write(tr_array(tr_it), oss.str());
    xmf.closeTimeStep();
  }
}
