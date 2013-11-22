#ifdef P4_TO_P8
#include "my_p8est_refine_coarsen.h"
#include <p8est_bits.h>
#else
#include "my_p4est_refine_coarsen.h"
#include <p4est_bits.h>
#endif
#include <sc_search.h>

p4est_bool_t
refine_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dy = dx;
  #ifdef P4_TO_P8
    double dz = dx;
  #endif

#ifdef P4_TO_P8
    double d = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double d = sqrt(dx*dx + dy*dy);
#endif

    p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];

    double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
#endif

    double x = (double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

#ifdef P4_TO_P8
    CF_3&  phi = *(data->phi);
#else
    CF_2&  phi = *(data->phi);
#endif
    double lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
    for (unsigned short cj = 0; cj<2; ++cj)
      for (unsigned short ci = 0; ci <2; ++ci){
#ifdef P4_TO_P8
        f[4*ck+2*cj+ci] = phi(x+ci*dx, y+cj*dy, z+ck*dz);
        if (fabs(f[4*ck+2*cj+ci]) <= 0.5*lip*d)
          return P4EST_TRUE;
#else
        f[2*cj+ci] = phi(x+ci*dx, y+cj*dy);
        if (fabs(f[2*cj+ci]) <= 0.5*lip*d)
          return P4EST_TRUE;
#endif
      }

#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_TRUE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_TRUE;
#endif

    return P4EST_FALSE;
  }
}

p4est_bool_t
coarsen_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)p4est->user_pointer;

  if (quad[0]->level <= data->min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level > data->max_lvl)
    return P4EST_TRUE;
  else
  {
    double dx = 2*(double)P4EST_QUADRANT_LEN((*quad)->level)/(double)P4EST_ROOT_LEN;
    double dy = dx;
  #ifdef P4_TO_P8
    double dz = dx;
  #endif

#ifdef P4_TO_P8
    double d = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double d = sqrt(dx*dx + dy*dy);
#endif

    p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];

    double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
#endif

    double x = (double)((*quad)->x)/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (double)((*quad)->y)/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (double)((*quad)->z)/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

#ifdef P4_TO_P8
    CF_3&  phi = *(data->phi);
#else
    CF_2&  phi = *(data->phi);
#endif
    double lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
    for (unsigned short cj = 0; cj<2; ++cj)
      for (unsigned short ci = 0; ci <2; ++ci){
#ifdef P4_TO_P8
        f[4*ck+2*cj+ci] = phi(x+ci*dx, y+cj*dy, z+ck*dz);
        if (fabs(f[4*ck+2*cj+ci]) <= 0.5*lip*d)
          return P4EST_FALSE;
#else
        f[2*cj+ci] = phi(x+ci*dx, y+cj*dy);
        if (fabs(f[2*cj+ci]) <= 0.5*lip*d)
          return P4EST_FALSE;
#endif
      }

#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_FALSE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_FALSE;
#endif

    return P4EST_TRUE;
  }
}

p4est_locidx_t splitting_criteria_random_t::counter = 0;
p4est_bool_t
refine_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) which_tree;
  splitting_criteria_random_t *data = (splitting_criteria_random_t*) p4est->user_pointer;

  p4est_bool_t refine;
  if (p4est->global_num_quadrants + splitting_criteria_random_t::counter >= data->max_quads)
    refine = P4EST_FALSE;
  else if (quad->level < data->min_lvl)
    refine = P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    refine = P4EST_FALSE;
  else
  {
    if (rand()%2)
      refine = P4EST_TRUE;
    else
      refine = P4EST_FALSE;
  }

  if (refine) splitting_criteria_random_t::counter += P4EST_CHILDREN -1;
  return refine;
}

p4est_bool_t
coarsen_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  (void) which_tree;
  splitting_criteria_random_t *data = (splitting_criteria_random_t*) p4est->user_pointer;

  p4est_bool_t coarsen;
  if (p4est->global_num_quadrants - splitting_criteria_random_t::counter<= data->min_quads)
    coarsen = P4EST_FALSE;
  else if (quad[0]->level <= data->min_lvl)
    coarsen = P4EST_FALSE;
  else if (quad[0]->level >  data->max_lvl)
    coarsen = P4EST_TRUE;
  else
  {
    if (rand()%2)
      coarsen = P4EST_TRUE;
    else
      coarsen = P4EST_FALSE;
  }

  if (coarsen) splitting_criteria_random_t::counter += P4EST_CHILDREN - 1;
  return coarsen;
}

p4est_bool_t
refine_every_cell(p4est_t *p4est, p4est_topidx_t tr, p4est_quadrant_t *quad)
{
  (void) p4est; (void) tr; (void) quad;
  return P4EST_TRUE;
}

p4est_bool_t
coarsen_every_cell(p4est_t *p4est, p4est_topidx_t tr, p4est_quadrant_t **quad)
{
  (void) p4est; (void) tr; (void) quad;
  return P4EST_TRUE;
}
