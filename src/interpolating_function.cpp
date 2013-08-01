#include "interpolating_function.h"

BilinearInterpolatingFunction::BilinearInterpolatingFunction(p4est_t *p4est_, p4est_nodes_t *nodes_, Vec F_)
  : p4est(p4est_), nodes(nodes_), F(F_)
{}

void BilinearInterpolatingFunction::update(p4est_t *p4est_, p4est_nodes_t *nodes_, Vec F_)
{
  p4est = p4est_;
  nodes = nodes_;
  F = F_;
}

double BilinearInterpolatingFunction::operator ()(double x, double y) const
{
  p4est_locidx_t quad_locidx;
  p4est_quadrant_t *quad;
  p4est_topidx_t quad_tree = 0;
  int quad_mpirank;

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double         *v2q = p4est->connectivity->vertices;

  p4est_topidx_t p4est_mm = t2v[0];
  p4est_topidx_t p4est_pp = t2v[p4est->connectivity->num_trees * P4EST_CHILDREN - 1];

  double domain_xmin = v2q[3*p4est_mm + 0];
  double domain_ymin = v2q[3*p4est_mm + 1];
  double domain_xmax = v2q[3*p4est_pp + 0];
  double domain_ymax = v2q[3*p4est_pp + 1];

  double xy [] = {x, y};

  if (xy[0]<=domain_xmin) xy[0] = domain_xmin;
  if (xy[0]>=domain_xmax) xy[0] = domain_xmax;
  if (xy[1]<=domain_ymin) xy[1] = domain_ymin;
  if (xy[1]>=domain_ymax) xy[1] = domain_ymax;

  quad_mpirank = my_p4est_brick_point_lookup_smallest(p4est, NULL, NULL, xy, &quad_tree, &quad_locidx, &quad);

  if (quad_mpirank != p4est->mpirank)
    throw std::runtime_error("[CASL_ERROR]: This point does not belog to this processor");

  p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, quad_tree);
  quad_locidx += tree->quadrants_offset;

  p4est_locidx_t *e2n = nodes->local_nodes;

  p4est_locidx_t nodes_locidx [] =
  {
    e2n[quad_locidx*P4EST_CHILDREN + 0],
    e2n[quad_locidx*P4EST_CHILDREN + 1],
    e2n[quad_locidx*P4EST_CHILDREN + 2],
    e2n[quad_locidx*P4EST_CHILDREN + 3]
  };

  for (int i = 0; i<4; ++i)
    nodes_locidx[i] = p4est2petsc_local_numbering(nodes, nodes_locidx[i]);

  double *F_val;
  PetscErrorCode ierr;
  ierr = VecGetArray(F, &F_val); CHKERRXX(ierr);
  double F_inter [] =
  {
    F_val[nodes_locidx[0]],
    F_val[nodes_locidx[1]],
    F_val[nodes_locidx[2]],
    F_val[nodes_locidx[3]]
  };

  double val = bilinear_interpolation(p4est, quad_tree, quad, F_inter, xy[0], xy[1]);
  ierr = VecRestoreArray(F, &F_val); CHKERRXX(ierr);

  return val;
}

void BilinearInterpolatingFunction::interpolateValuesToNewForest(p4est_t *p4est_new, p4est_nodes_t *nodes_new, Vec *F_new)
{
  // First create a fector long enough to hold new values
  PetscErrorCode ierr;
  ierr = VecCreateGhost(p4est_new, nodes_new, F_new); CHKERRXX(ierr);

  p4est_locidx_t *e2n_new = nodes_new->local_nodes;
  p4est_locidx_t *e2n_old = nodes->local_nodes;
  double *F_val_old, *F_val_new;

  ierr = VecGetArray( F    , &F_val_old); CHKERRXX(ierr);
  ierr = VecGetArray(*F_new, &F_val_new); CHKERRXX(ierr);

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double         *v2q = p4est->connectivity->vertices;

  p4est_topidx_t p4est_mm = t2v[0];
  p4est_topidx_t p4est_pp = t2v[p4est->connectivity->num_trees * P4EST_CHILDREN - 1];

  double domain_xmin = v2q[3*p4est_mm + 0];
  double domain_ymin = v2q[3*p4est_mm + 1];
  double domain_xmax = v2q[3*p4est_pp + 0];
  double domain_ymax = v2q[3*p4est_pp + 1];

  ArrayV<bool> is_processed(nodes_new->num_owned_indeps); is_processed = false;

  for (p4est_topidx_t tr_it = p4est_new->first_local_tree; tr_it <= p4est_new->last_local_tree; ++tr_it)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est_new->trees, tr_it);
    for (p4est_locidx_t qu = 0; qu < tree->quadrants.elem_count; ++qu)
    {
      p4est_locidx_t qu_locidx = qu + tree->quadrants_offset;
      for (unsigned short i = 0; i<P4EST_CHILDREN; ++i)
      {
        p4est_locidx_t p4est_node_locidx = e2n_new[qu_locidx*P4EST_CHILDREN + i];
        p4est_locidx_t petsc_node_locidx = p4est2petsc_local_numbering(nodes_new, p4est_node_locidx);

        if (petsc_node_locidx >= nodes_new->num_owned_indeps)
          continue;

        if (!is_processed(petsc_node_locidx))
        {
          // Get the coordinates of this node
          p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_new->indep_nodes, p4est_node_locidx);

          double xy [] =
          {
            (double)node->x/(double)P4EST_ROOT_LEN,
            (double)node->y/(double)P4EST_ROOT_LEN
          };
          c2p_coordinate_transform(p4est_new, tr_it, &xy[0], &xy[1], NULL);

          if (xy[0]<=domain_xmin) xy[0] = domain_xmin;
          if (xy[0]>=domain_xmax) xy[0] = domain_xmax;
          if (xy[1]<=domain_ymin) xy[1] = domain_ymin;
          if (xy[1]>=domain_ymax) xy[1] = domain_ymax;

          // we need to find the quadrant that owns this new node
          p4est_topidx_t tr_it_old = tr_it;
          p4est_quadrant_t *quad_old;
          p4est_locidx_t qu_locidx_old;
          if (p4est_new->mpirank != my_p4est_brick_point_lookup_smallest(p4est, NULL, NULL, xy, &tr_it_old, &qu_locidx_old, &quad_old))
            throw std::runtime_error("[CASL_ERROR]: Currently cannot interpolate from a point if it belongs to another processor on the old mesh");

          p4est_tree_t *tree_old = p4est_tree_array_index(p4est->trees, tr_it_old);
          qu_locidx_old += tree_old->quadrants_offset;

          p4est_locidx_t nodes_locidx_old[P4EST_CHILDREN];
          double F_nodes_old[P4EST_CHILDREN];
          for (unsigned short j = 0; j<P4EST_CHILDREN; ++j)
          {
            nodes_locidx_old[j] = p4est2petsc_local_numbering(nodes, e2n_old[qu_locidx_old*P4EST_CHILDREN + j]);
            F_nodes_old[j] = F_val_old[nodes_locidx_old[j]];
          }

          F_val_new[petsc_node_locidx] = bilinear_interpolation(p4est, tr_it_old, quad_old, F_nodes_old, xy[0], xy[1]);
          is_processed(petsc_node_locidx) = true;
        }
      }
    }
  }

  ierr = VecRestoreArray( F    , &F_val_old); CHKERRXX(ierr);
  ierr = VecRestoreArray(*F_new, &F_val_new); CHKERRXX(ierr);

  // Now broadcast to ghost values
  ierr = VecGhostUpdateBegin(*F_new, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (*F_new, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}
