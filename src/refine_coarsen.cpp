#include "refine_coarsen.h"

int
refine_levelset_continous (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  grid_continous_data_t *data = (grid_continous_data_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    double dx, dy;
    dx_dy_dz_quadrant(p4est, which_tree, quad, &dx, &dy, NULL);
    double d = sqrt(dx*dx + dy*dy);

    double x = (double)quad->x/(double)P4EST_ROOT_LEN;
    double y = (double)quad->y/(double)P4EST_ROOT_LEN;

    c2p_coordinate_transform(p4est, which_tree, &x, &y, NULL);

    CF_2&  phi = *(data->phi);
    double lip = data->lip;

    if (fabs(phi(x+0.5*dx, y+0.5*dy)) <= lip * 0.5*d)
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

int
coarsen_levelset_continous (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  grid_continous_data_t *data = (grid_continous_data_t*)p4est->user_pointer;

  if (quad[0]->level <= data->min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level > data->max_lvl)
    return P4EST_TRUE;
  else
  {
    double dx, dy;
    dx_dy_dz_quadrant(p4est, which_tree, quad[0], &dx, &dy, NULL);
    dx *= 2;
    dy *= 2;
    double d = sqrt(dx*dx + dy*dy);

    double x = (double)quad[0]->x/(double)P4EST_ROOT_LEN;
    double y = (double)quad[0]->y/(double)P4EST_ROOT_LEN;

    c2p_coordinate_transform(p4est, which_tree, &x, &y, NULL);

    CF_2  &phi = *(data->phi);
    double lip = data->lip;

    if (fabs(phi(x+0.5*dx, y+0.5*dy)) > lip * 0.5*d)
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

int
refine_levelset_discrete(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  grid_discrete_data_t *data = (grid_discrete_data_t*)p4est->user_pointer;
  my_p4est_nodes_t *old_nodes = data->nodes;
  p4est_t *old_p4est = data->p4est;
  p4est_locidx_t *e2n = old_nodes->local_nodes;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    double dx, dy;
    dx_dy_dz_quadrant(p4est, which_tree, quad, &dx, &dy, NULL);
    double d = sqrt(dx*dx + dy*dy);

    double xy [] =
    {
      (double)quad->x/(double)P4EST_ROOT_LEN,
      (double)quad->y/(double)P4EST_ROOT_LEN,
    };
    c2p_coordinate_transform(p4est, which_tree, &xy[0], &xy[1], NULL);

    xy[0] += 0.5*dx;
    xy[1] += 0.5*dy;

    // Find this point in the old forest
    p4est_quadrant_t *old_quad;
    p4est_locidx_t old_quad_locidx;
    p4est_tree_t *old_tree;

    if (p4est->mpirank != my_p4est_brick_point_lookup(old_p4est, xy, &which_tree, &old_quad_locidx, &old_quad))
      throw std::runtime_error("[CASL_ERROR]: Currently can only interpolate from parts of the old tree that belong to the same processor");

    old_tree = p4est_tree_array_index(old_p4est->trees, which_tree);
    old_quad_locidx += old_tree->quadrants_offset;

    double *phi = data->phi;
    double lip  = data->lip;

    double F[P4EST_CHILDREN];
    for (unsigned short i = 0; i<P4EST_CHILDREN; ++i)
      F[i] = phi[p4est2petsc_local_numbering(old_nodes, e2n[old_quad_locidx*P4EST_CHILDREN + i])];

    double phi_c = bilinear_interpolation(old_p4est, which_tree, old_quad, F, xy[0], xy[1]);

    if (fabs(phi_c) <= lip *0.5*d)
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

int
coarsen_levelset_discrete(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  grid_discrete_data_t *data = (grid_discrete_data_t*)p4est->user_pointer;
  my_p4est_nodes_t *old_nodes = data->nodes;
  p4est_t *old_p4est = data->p4est;
  p4est_locidx_t *e2n = old_nodes->local_nodes;


  if (quad[0]->level <= data->min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level > data->max_lvl)
    return P4EST_TRUE;
  else
  {
    double dx, dy;
    dx_dy_dz_quadrant(p4est, which_tree, quad[0], &dx, &dy, NULL);
    dx *= 2;
    dy *= 2;
    double d = sqrt(dx*dx + dy*dy);

    double xy [] =
    {
      (double)quad[0]->x/(double)P4EST_ROOT_LEN,
      (double)quad[0]->y/(double)P4EST_ROOT_LEN,
    };
    c2p_coordinate_transform(p4est, which_tree, &xy[0], &xy[1], NULL);

    xy[0] += 0.5*dx;
    xy[1] += 0.5*dy;

    // Find this point in the old forest
    p4est_quadrant_t *old_quad;
    p4est_locidx_t old_quad_locidx;

    if (p4est->mpirank != my_p4est_brick_point_lookup(old_p4est, xy, &which_tree, &old_quad_locidx, &old_quad))
      throw std::runtime_error("[CASL_ERROR]: Currently can only interpolate from parts of the old tree that belong to the same processor");

    p4est_tree_t *old_tree = p4est_tree_array_index(old_p4est->trees, which_tree);
    old_quad_locidx += old_tree->quadrants_offset;

    double lip  = data->lip;
    double *phi = data->phi;

    double F[P4EST_CHILDREN];
    for (unsigned short i = 0; i<P4EST_CHILDREN; ++i)
      F[i] = phi[p4est2petsc_local_numbering(old_nodes, e2n[old_quad_locidx*P4EST_CHILDREN + i])];

    double phi_c = bilinear_interpolation(old_p4est, which_tree, old_quad, F, xy[0], xy[1]);

    if (fabs(phi_c) > lip * 0.5*d)
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}
