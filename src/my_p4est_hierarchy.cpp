#include "my_p4est_hierarchy.h"
#include <p4est_communication.h>
#include <stdexcept>
#include <petsclog.h>
#include <sstream>

// logging variable -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin //(e, o1, o2, o3, o4)
#undef PetscLogEventEnd //(e, o1, o2, o3, o4)
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_hierarchy_t;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops //(n)
#define PetscLogFlops(n) 0
#endif

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

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);


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

  ierr = PetscLogEventEnd(log_my_p4est_hierarchy_t, 0, 0, 0, 0); CHKERRXX(ierr);

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

int my_p4est_hierarchy_t::find_smallest_quadrant_containing_point(const double *xy, p4est_quadrant_t &best_match, std::vector<p4est_quadrant_t> &remote_matches)
{
#ifdef CASL_THROWS
  if (xy[0] < 0 || xy[0] > myb->nxytrees[0] || xy[1] < 0 || xy[1] > myb->nxytrees[1])
  {
    std::ostringstream oss;
    oss << "[ERROR]: Point (" << xy[0] << "," << xy[1] << ") is outside computational domain" << std::endl;
    throw std::invalid_argument(oss.str());
  }
#endif

  int found_rank, rank=-1;
  p4est_quadrant_t sq;
  P4EST_QUADRANT_INIT(&sq);
  sq.level = P4EST_QMAXLEVEL;

  P4EST_QUADRANT_INIT(&best_match);

  // a quadrant length at most will be P4EST_QMAXLEVEL = P4EST_MAXLEVEL - 1
  const static double qeps = (double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL)  / (double)P4EST_ROOT_LEN;
  const static p4est_qcoord_t qh = P4EST_QUADRANT_LEN(P4EST_QMAXLEVEL);

  // perturb in 4 directions
  for (short i = -1; i<2; i += 2)
    for (short j = -1; j<2; j += 2)
    {
      // perturb the point
      double xy_perturb [] = {xy[0] + i*qeps, xy[1] + j*qeps};

      /* clip to the boundary
       * TODO: this wont work with periodic. Need to add something in myb
       * to indicate if the p4est is periodic
       */
      if      (xy_perturb[0] < qeps)                    xy_perturb[0] = qeps;
      else if (xy_perturb[0] > myb->nxytrees[0] - qeps) xy_perturb[0] = myb->nxytrees[0] - qeps;
      if      (xy_perturb[1] < qeps)                    xy_perturb[1] = qeps;
      else if (xy_perturb[1] > myb->nxytrees[1] - qeps) xy_perturb[1] = myb->nxytrees[1] - qeps;

      // first locate the correct tree
      int tr_xy [] = {(int)floor(xy_perturb[0]), (int)floor(xy_perturb[1])};
      p4est_topidx_t tt = myb->nxy_to_treeid[tr_xy[0] + tr_xy[1]*myb->nxytrees[0]];
      p4est_tree_t *p4est_tr = (p4est_tree_t*)sc_array_index(p4est->trees, tt);
      const std::vector<HierarchyCell>& h_tr = trees[tt];

      // check who is the owner
      p4est_qcoord_t sqx = (p4est_qcoord_t)((xy_perturb[0] - tr_xy[0]) * P4EST_ROOT_LEN);
      p4est_qcoord_t sqy = (p4est_qcoord_t)((xy_perturb[1] - tr_xy[1]) * P4EST_ROOT_LEN);

      // ensure that quadrant is a multiple of qh, otherwise p4est function will freak out!
      sq.x = sqx & ~(qh - 1);
      sq.y = sqy & ~(qh - 1);

      found_rank = p4est_comm_find_owner(p4est, tt, &sq, p4est->mpirank);

      // return to original value otherwise a point could be placed on the boundary of childs
      // and we get biased results
      sq.x = sqx;
      sq.y = sqy;

      const HierarchyCell *it, *begin; begin = it = &h_tr[0];
      while(CELL_LEAF != it->child){
        p4est_qcoord_t half_h = P4EST_QUADRANT_LEN(it->level) / 2;
        short cj = (it->jmin + half_h < sq.y);
        short ci = (it->imin + half_h < sq.x);

        it = begin + it->child + 2*cj + ci;
      }

      if (found_rank == p4est->mpirank) {
        if (it->quad >= 0 && it->quad < p4est->local_num_quadrants) { // local quadrant
          p4est_locidx_t pos = it->quad - p4est_tr->quadrants_offset;
          p4est_quadrant_t *tmp = (p4est_quadrant_t*)sc_array_index(&p4est_tr->quadrants, pos);
          if (tmp->level > best_match.level) {
            best_match = *tmp;
            best_match.p.piggy3.which_tree = tt;
            best_match.p.piggy3.local_num  = pos;
            rank = found_rank;
          }
        } else {
#ifdef CASL_THROWS
          std::ostringstream oss;
          oss << "[ERROR]: point = (" << xy[0] << "," << xy[1] << "). found_rank = " << found_rank << " quad = " << it->quad << std::endl <<
                 " p4est suggests point should be owned locally but it's not found in the local part of hierarchy" << std::endl;
          throw std::runtime_error(oss.str());
#endif
        }
      } else {
        if (it->quad >= p4est->local_num_quadrants && it->quad < p4est->local_num_quadrants + (p4est_locidx_t)ghost->ghosts.elem_count) { // ghost quadrant
          p4est_locidx_t pos = it->quad - p4est->local_num_quadrants;
          p4est_quadrant_t *tmp = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, pos);
          if (tmp->level > best_match.level) {
            best_match = *tmp;
            best_match.p.piggy3.which_tree = tt;
            best_match.p.piggy3.local_num  = pos;
            rank = found_rank;
          }
        } else if (NOT_A_P4EST_QUADRANT == it->quad) { // remote quadrant
          sq.p.piggy1.which_tree = tt;
          sq.p.piggy1.owner_rank = found_rank;

          remote_matches.push_back(sq);
          rank = -1;
        } else {
#ifdef CASL_THROWS
          std::ostringstream oss;
          oss << "[ERROR]: point = (" << xy[0] << "," << xy[1] << "). found_rank = " << found_rank << " quad = " << it->quad << std::endl <<
                 " Could not find any point in the hierarchy for given point. This is a bug!" << std::endl;
          throw std::runtime_error(oss.str());
#endif
        }
      }
    }

  return rank;
}
