#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_macros.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_macros.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#endif
#include <sc_search.h>
#include <iostream>
#include <cmath>

p4est_bool_t
refine_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl && !data->refine_only_inside)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
    double smallest_dxyz_max = MAX((tree_xmax-tree_xmin), (tree_ymax-tree_ymin))*((double)P4EST_QUADRANT_LEN(data->max_lvl))/((double)P4EST_ROOT_LEN);
#ifdef P4_TO_P8
    smallest_dxyz_max = MAX(smallest_dxyz_max, (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(data->max_lvl))/((double)P4EST_ROOT_LEN));
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    double x = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    CF_DIM &phi  = *(data->phi);
    double  lip  = data->lip;
    double  band = data->band*smallest_dxyz_max;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
          f[SUMD(ci, 2*cj, 4*ck)] = phi(DIM(x+ci*dx, y+cj*dy, z+ck*dz));
          if (fabs(f[SUMD(ci, 2*cj, 4*ck)])-band <= 0.5*lip*d)
            return P4EST_TRUE;
        }

#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_TRUE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_TRUE;
#endif

    if (data->refine_only_inside && f[0] <= 0 && quad->level < data->min_lvl)
      return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

p4est_bool_t
refine_levelset_cf_and_uniform_band (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_cf_and_uniform_band_t *data = (splitting_criteria_cf_and_uniform_band_t*)p4est->user_pointer;
  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif
    double smallest_dxyz_max = MAX(DIM((tree_xmax-tree_xmin), (tree_ymax-tree_ymin), (tree_zmax-tree_zmin)))*((double)P4EST_QUADRANT_LEN(data->max_lvl))/((double)P4EST_ROOT_LEN);

    double x = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    const CF_DIM&  phi = *(data->phi);

    double f;
    bool vmmm_is_neg = phi(DIM(x, y, z)) <= 0.0;
    bool is_crossed = false;
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<=2; ++ck)
#endif
      for (unsigned short cj = 0; cj<=2; ++cj)
        for (unsigned short ci = 0; ci <=2; ++ci){
          f = phi(DIM(x+ci*0.5*dx, y+cj*0.5*dy, z+ck*0.5*dz));
          is_crossed = is_crossed || (vmmm_is_neg != (f <= 0.0));
          if(fabs(f) < data->uniform_band*smallest_dxyz_max || is_crossed)
            return P4EST_TRUE;
        }

  }
  return refine_levelset_cf(p4est, which_tree, quad);
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
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = 2*(double)P4EST_QUADRANT_LEN(quad[0]->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
    double smallest_dxyz_max = MAX((tree_xmax-tree_xmin), (tree_ymax-tree_ymin))*((double)P4EST_QUADRANT_LEN(data->max_lvl))/((double)P4EST_ROOT_LEN);
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
    smallest_dxyz_max = MAX(smallest_dxyz_max, (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(data->max_lvl))/((double)P4EST_ROOT_LEN));
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    double x = (tree_xmax-tree_xmin)*(double)quad[0]->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad[0]->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad[0]->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    CF_DIM &phi  = *(data->phi);
    double  lip  = data->lip;
    double  band = data->band*smallest_dxyz_max;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
          f[SUMD(ci, 2*cj, 4*ck)] = phi(DIM(x+ci*dx, y+cj*dy, z+ck*dz));
          if (fabs(f[SUMD(ci, 2*cj, 4*ck)])-band <= 0.5*lip*d)
            return P4EST_FALSE;
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

p4est_bool_t
refine_levelset_thresh (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  splitting_criteria_thresh_t *data = (splitting_criteria_thresh_t*)p4est->user_pointer;

  if (quad->level < data->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double x = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    const CF_DIM& f = *(data->f);
    double thresh = data->thresh;

#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci)
          if(f(DIM(x+ci*dx, y+cj*dy, z+ck*dz))>thresh)
            return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

p4est_bool_t
coarsen_levelset_thresh (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  splitting_criteria_thresh_t *data = (splitting_criteria_thresh_t*)p4est->user_pointer;

  if (quad[0]->level <= data->min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level > data->max_lvl)
    return P4EST_TRUE;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = 2*(double)P4EST_QUADRANT_LEN(quad[0]->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double x = (tree_xmax-tree_xmin)*(double)quad[0]->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad[0]->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad[0]->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    const CF_DIM& f = *(data->f);
    double thresh = data->thresh;

#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci)
          if(f(DIM(x+ci*dx, y+cj*dy, z+ck*dz))>thresh)
            return P4EST_FALSE;

    return P4EST_TRUE;
  }
}

p4est_bool_t
refine_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) which_tree;
  splitting_criteria_random_t *data = (splitting_criteria_random_t*) p4est->user_pointer;

  // if (data->num_quads >= (p4est_gloidx_t) ((double)data->max_quads/(double)p4est->mpisize))
  if (data->num_quads >= data->max_quads)
    return P4EST_FALSE;
  else if (quad->level < data->min_lvl)
  { data->num_quads += P4EST_CHILDREN - 1; return P4EST_TRUE; }
  else if (quad->level >= data->max_lvl)
    return P4EST_FALSE;
  else
  {
    if (rand()%2)
    { data->num_quads += P4EST_CHILDREN - 1; return P4EST_TRUE;}
    else
      return P4EST_FALSE;
  }
}

p4est_bool_t
coarsen_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  (void) which_tree;
  splitting_criteria_random_t *data = (splitting_criteria_random_t*) p4est->user_pointer;

  // if (data->num_quads <= (p4est_gloidx_t) ((double)data->min_quads/(double)p4est->mpisize))
  if (data->num_quads <= data->min_quads)
    return P4EST_FALSE;
  else if (quad[0]->level <= data->min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level >  data->max_lvl)
  { data->num_quads -= P4EST_CHILDREN - 1; return P4EST_TRUE; }
  else
  {
    if (rand()%2)
    { data->num_quads -= P4EST_CHILDREN - 1; return P4EST_TRUE; }
    else
      return P4EST_FALSE;
  }
}

p4est_bool_t
refine_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) p4est;
  (void) which_tree;
  (void) quad;

  return P4EST_TRUE;
}

p4est_bool_t
coarsen_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  (void) p4est;
  (void) which_tree;
  (void) quad;

  return P4EST_TRUE;
}

p4est_bool_t
refine_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) which_tree;
  const splitting_criteria_marker_t& marker = *(splitting_criteria_marker_t*)(p4est->user_pointer);
  if (quad->level < marker.min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= marker.max_lvl)
    return P4EST_FALSE;
  else
    return (*(p4est_bool_t*)quad->p.user_data);
}

p4est_bool_t
coarsen_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  (void) which_tree;
  const splitting_criteria_marker_t& marker = *(splitting_criteria_marker_t*)(p4est->user_pointer);
  if (quad[0]->level < marker.min_lvl)
    return P4EST_FALSE;
  else if (quad[0]->level >= marker.max_lvl)
    return P4EST_TRUE;
  else
    for (short i=0; i<P4EST_CHILDREN; i++)
      if (*(p4est_bool_t*)quad[i]->p.user_data) return P4EST_TRUE;

  return P4EST_FALSE;
}

void splitting_criteria_tag_t::tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t which_tree, const double* f, bool finest_in_negative_flag) {
  if (quad->level < min_lvl) {
    quad->p.user_int = REFINE_QUADRANT;

  } else if (quad->level > max_lvl) {
    quad->p.user_int = COARSEN_QUADRANT;

  } else {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
    double smallest_dxyz_max = MAX((tree_xmax-tree_xmin), (tree_ymax-tree_ymin))*((double)P4EST_QUADRANT_LEN(max_lvl))/((double)P4EST_ROOT_LEN);
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
    smallest_dxyz_max = MAX(smallest_dxyz_max, (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(max_lvl))/((double)P4EST_ROOT_LEN));
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));
    double band_real = band*smallest_dxyz_max;

    // refinement based on distance
    bool refine = false, coarsen = true;

    if(finest_in_negative_flag)
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || (quad->level < max_lvl && (f[i]-band_real <= 0.5*lip*d || (i == 0 ? false : ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && quad->level > min_lvl && f[i]-band_real >= 1.0*lip*d && (i == 0 ? true : ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
      }
    else
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || (quad->level < max_lvl && (fabs(f[i])-band_real <= 0.5*lip*d || (i == 0 ? false : ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && quad->level > min_lvl && fabs(f[i])-band_real >= 1.0*lip*d && (i == 0 ? true : ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
      }

    if (refine) {
      quad->p.user_int = REFINE_QUADRANT;
    } else if (coarsen) {
      quad->p.user_int = COARSEN_QUADRANT;
    } else {
      quad->p.user_int = SKIP_QUADRANT;
    }
  }
}

void splitting_criteria_tag_t::tag_quadrant_inside(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t which_tree, const double* f) {
  if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;
  else
  {
    p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0];
    p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + P4EST_CHILDREN-1];

    double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
#endif

    double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
    double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
    double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (tree_xmax-tree_xmin) * dmin;
    double dy = (tree_ymax-tree_ymin) * dmin;
    double smallest_dxyz_max = MAX((tree_xmax-tree_xmin), (tree_ymax-tree_ymin))*((double)P4EST_QUADRANT_LEN(max_lvl))/((double)P4EST_ROOT_LEN);
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
    smallest_dxyz_max = MAX(smallest_dxyz_max, (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(max_lvl))/((double)P4EST_ROOT_LEN));
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));
    double min_diag = d/((double) (1 << (max_lvl - quad->level)));

    double band_real = band*smallest_dxyz_max;

    // refinement based on distance
    bool refine = false, coarsen = true;

    for (short i = 0; i < P4EST_CHILDREN; i++) {
      refine  = refine  || (fabs(f[i])-band_real <= 0.5*lip*d );
      coarsen = coarsen && (fabs(f[i])-band_real >= 1.0*lip*d );
    }

    if (refine && quad->level >= max_lvl) refine = false;

    bool one_negative = false;
    for (short i = 0; i < P4EST_CHILDREN; i++) { one_negative = one_negative || f[i] < 0; }
    if (quad->level < min_lvl && one_negative) { refine = true;      }

    if (quad->level < min_lvl && one_negative) { refine = true;      }

    if (coarsen && quad->level <= min_lvl && one_negative) coarsen = false;

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;
    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;
    else
      quad->p.user_int = SKIP_QUADRANT;
  }
}

// ELYCE TRYING SOMETHING --------:

void splitting_criteria_tag_t::tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t tree_idx, p4est_locidx_t quad_idx,p4est_nodes_t *nodes,
                                            const double* phi_p, const int num_fields,
                                            bool use_block, bool enforce_uniform_band,
                                            double refine_band,double coarsen_band,
                                            const double** fields,const double* fields_block,
                                            std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn,
                                            std::vector<int> lmax_custom){

  // WARNING: This function has not yet been validated in 3d

  // Option lists are provided in the following format:
  // opn = {coarsen_field_1, refine_field_1, coarsen_field_2, refine_field_2, ......, coarsen_field_n, refine_field_n}, where n = num_fields
  // Thus the length of the list is 2*num_fields
  // Coarsen options are accessed as opn[2*i], for i = 0,..., num_fields
  // Refine options are accessed as opn[2*i + 1]
  // Compare option:
  // - 0 --> less than
  // - 1 --> greater than

  // Diag option:
  // - 0 --> divide criteria by diag
  // - 1 --> multiply criteria by diag
  // - 2 --> Neither, compare values
  if (quad->level < min_lvl) {
    quad->p.user_int = REFINE_QUADRANT;

  } else if (quad->level > max_lvl) {
    quad->p.user_int = COARSEN_QUADRANT;
  }
  else{
      p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
      p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];

      const double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
      const double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  #ifdef P4_TO_P8
      const double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  #endif

      const double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
      const double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  #ifdef P4_TO_P8
      double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  #endif

      const double dxyz_min_overall = (double)P4EST_QUADRANT_LEN((int8_t) max_lvl)/(double)P4EST_ROOT_LEN; // Gives min dimension overall --> used for uniform band implementation
      const double dlvl = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dlvl -- the size of the current quad (as a fraction of root length)
      const double dx = (tree_xmax-tree_xmin) * dlvl;
      const double dy = (tree_ymax-tree_ymin) * dlvl;

  #ifdef P4_TO_P8
      double dz = (tree_zmax-tree_zmin) * dlvl;

  #endif

      const double quad_diag      = sqrt(SUMD(dx*dx, dy*dy, dz*dz));
      const double max_quad_size  = MAX(DIM(dx, dy, dx)); // max_quad_size gives the largest size of the quad
      double max_tree_dim = MAX((tree_xmax - tree_xmin),(tree_ymax - tree_ymin));

      CODE3D(max_tree_dim = MAX(max_tree_dim,(tree_zmax - tree_zmin)));
      double dxyz_smallest = max_tree_dim*dxyz_min_overall;

      // Check possibility of refining or coarsening:
      bool coarsen = (quad->level > min_lvl); // Will return true if possible to coarsen
      bool refine = (quad->level < max_lvl);  // Will return true if possible to refine

      // Initialize booleans to check if the LSF changes sign within the quadrant --> if not all pos and not all neg, tag for refinement
      bool all_pos = true;
      bool all_neg = true;

      // Define the size of the uniform band to refine and coarsen by, determined by the user (default is set to 0 if user does not select this option)
      double ref_band, coars_band;
      ref_band = enforce_uniform_band? dxyz_smallest*refine_band : 0.0;
      coars_band = enforce_uniform_band? dxyz_smallest*coarsen_band : 0.0;


      // First now, check coarsening arguments if coarsening is possible:
      if(coarsen){
          // Initialize node index:
          p4est_locidx_t node_idx;

          // TO-CLEAN: Initiliaze booleans to help keep track in case we are looking for a sign change for a field
          bool checking_sign_change = false;
          bool field_all_pos[num_fields];
          bool field_all_neg[num_fields];
          bool below_threshold[num_fields]; // to allow for coarsening, all nodes in the cell for the given field must be below the specified threshold

          // Initialize the checks for all positive or all negative:
          for(int n=0; n<num_fields;n++){
              field_all_pos[n] = true;
              field_all_neg[n] = true;
              below_threshold[n] = true;
          }

          for(unsigned short k=0;k<P4EST_CHILDREN;k++){
              // Get the appropriate node index:
              node_idx    = nodes->local_nodes[P4EST_CHILDREN*quad_idx+k];


              // Initialize doubles to hold field value and coarsening criteria:
              double field_val;
              double criteria_coarsen = -1.; // Initialized to negative 1 so we can check if they are set -- assert this is positive

              // Now consider the LSF coarsening criteria:
              coarsen = coarsen && ((fabs(phi_p[node_idx]) - coars_band) >= 1.0*lip*quad_diag);


              /*Now, if coarsening is still allowed by LSF, check the fields --
               * if LSF does not allow coarsening, don't bother looping over fields or checking that info unnecessarily*/
              if(coarsen){
                  for(unsigned short n = 0; n<num_fields;n++){
                      if(!coarsen) break; // if coarsen becomes false, no point in checking rest of conditions

                      // Access the field value, depending on whether the user has specified block structure or array of vectors
                      if(use_block){
                          // Get the field value from the block vector ptr
                          field_val = fields_block[num_fields*node_idx + n];
                      }
                      else{
                          // Get the field value from the array of PETSc vector ptrs
                          field_val = fields[n][node_idx];
                      }

                      // Get the coarsening criteria value:
                      criteria_coarsen = criteria[2*n];

                      // Now we enter switch case of possible scenarii (depending on what user has specified)
                      P4EST_ASSERT(criteria_coarsen>=0.); // Make sure criteria has been defined before continuing
                      switch(diag_opn[2*n]){
                      case DIVIDE_BY:{
                          switch(compare_opn[2*n]){
                          case GREATER_THAN:{
                              coarsen = coarsen && ((fabs(field_val)) > criteria_coarsen/max_quad_size);
//                              if(n==2) printf("COARSEN IF GREATER THAN %d \n AT LEVEL %d \n",criteria_coarsen/d,quad->level);
                              break;
                          }
                          case LESS_THAN:{
                              coarsen = coarsen && ((fabs(field_val)) < criteria_coarsen/max_quad_size);
                              //                                  if(coarsen && n==0){printf("Coarsened at level %d based on vorticity = %0.4f on rank %d \n \n",quad->level,field_val,p4est->mpirank);}
                              break;
                          }
                          case SIGN_CHANGE:{
                              checking_sign_change = true;

                              field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                              field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                              below_threshold[n] = (fabs(field_val)<criteria_coarsen/max_quad_size) && below_threshold[n];
//                                  PetscPrintf(p4est->mpicomm,"CHECKING SIGN CHANGE \n field_all_pos: %s field_all_neg: %s \n",field_all_pos[n]?"true":"false",field_all_neg[n]?"true":"false");


                              break;
                          }
                          default:{
                              throw std::invalid_argument("blah");
                          }
                        } // end of switch on compare option
                          break;
                      } // end of case : divide_by
                      case MULTIPLY_BY:{
                          switch(compare_opn[2*n]){
                          case GREATER_THAN:{
                              coarsen = coarsen && (fabs(field_val) > criteria_coarsen*max_quad_size);
                              break;
                          }
                          case LESS_THAN:{
                              coarsen = coarsen && (fabs(field_val) < criteria_coarsen*max_quad_size);
                              break;
                          }
                          case SIGN_CHANGE:{
                              checking_sign_change = true;

                              field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                              field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                              below_threshold[n] = (fabs(field_val)<criteria_coarsen*max_quad_size) && below_threshold[n];

//                                  PetscPrintf(p4est->mpicomm,"CHECKING SIGN CHANGE \n field_all_pos: %s field_all_neg: %s \n",field_all_pos[n]?"true":"false",field_all_neg[n]?"true":"false");


                              break;
                          }

                          default:{
                              throw std::invalid_argument("blah");
                          }
                        } // end of switch on compare option
                          break;
                      } // end of case: multiply_by
                      case ABSOLUTE:{
                          switch(compare_opn[2*n]){
                          case GREATER_THAN:{
                              coarsen = coarsen && (fabs(field_val) > criteria_coarsen);
                              break;
                          }
                          case LESS_THAN:{
                              coarsen = coarsen && (fabs(field_val) < criteria_coarsen);
                              break;
                          }
                          case SIGN_CHANGE:{
                              checking_sign_change = true;

                              field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                              field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                              below_threshold[n] = (fabs(field_val)<criteria_coarsen) && below_threshold[n];

//                                  PetscPrintf(p4est->mpicomm,"CHECKING SIGN CHANGE \n field_all_pos: %s field_all_neg: %s \n",field_all_pos[n]?"true":"false",field_all_neg[n]?"true":"false");


                              break;
                          }
                          case NO_CHECK:{
                              continue;
                          }
                          default:{
                              throw std::invalid_argument("blah");
                          }

                        } // end of switch on compare option

                          break;
                      } // end of case: absolute
                      } // End of switch case on diagonal comparison option
                  } // end of loop over number of fields
              } // end of if coarsen still allowed within "if coarsen"
          } // end of loop over children


          // If we had a sign change, check if that happened:
          if(checking_sign_change){
  //            PetscPrintf(p4est->mpicomm,"CHECKING FOR A SIGN CHANGE COARSEN CASE: \n");
              for(int n=0;n<num_fields;n++){
                  bool sign_change = !field_all_pos[n] && !field_all_neg[n];

                  coarsen = coarsen && (!sign_change && below_threshold[n]);
              }

          }
      } // end of "if coarsen"


      // Next we check the refinement arguments if refining is possible:

      if(refine){

        // Now that we are inside refine, set refine to false:

        /*Now tht we are inside refine, set refine to false:
         *
         * This is because cell will be refined if ANY of the refinement conditions come back as true
         * So we take refine = refine || (other refinement conditions)
         * Thus we need to initialize it to false, otherwise any time refinement is possible, cell will be refined
         *
         * */
        refine = false;

        // Initialize holder for node index in question
        p4est_locidx_t node_idx;

        /*Note: looping over children will look different here than for coarsening case,
         * because we are searching for possible neighbor nodes at higher levels of refinement to provide us some extra info*/

         /*We loop over the children of the quadrant: -- will check criteria of each field at each node child of the quadrant in question
         We loop over i,j,k coordinates of the cell where the indices correspond to children
         In 2D, this looks like:
         */

        /*
         *    (i=0,j=2)o--------(i=1,j=2)x---------o(i=2,j=2)
         *             |                           |
         *             |                           |
         *             |                           |
         *             |                           |
         *             |                           |
         *    (i=0,j=1)x---------------------------x(i=2,j=1)
         *             |                           |
         *             |                           |
         *             |                           |
         *             |                           |
         *             |                           |
         *    (i=0.j=0)o-------(i=1,j=0)x----------o(i=2,j=0)
         *
         *
         * The purpose of doing this is to check for T-junction points which may not be owned
         * by the quadrant we are considering, but may be present nonetheless and can provide
         * more accurate information about the fields that we may want to be refining by
         *
         * In this illustration, the "o" points are nodes that are owned by the quadrant that we
         * are considering, and "x" points are locations of *possible* neighboring points at a higher
         * level of refinement
         * */

        // Initialize boolean that keeps track of if a node exists or not at each child case we iterate through
        bool node_found;

        // Initialize boolean to keep track of if we found a higher refinement level neighbor point to use or not
        bool we_had_neighbor_point = false;

        // Initiliaze booleans to help keep track in case we are looking for a sign change for a field
        bool checking_sign_change = false;
        bool above_threshold[num_fields]; // To refine in the sign change case, all node values for the given field must be above the specified threshold (this avoids refining around sign changes on order of machine error)
        bool field_all_pos[num_fields];
        bool field_all_neg[num_fields];

        // Initialize the checks for all positive or all negative:
        for(int n=0; n<num_fields;n++){
            field_all_pos[n] = true;
            field_all_neg[n] = true;
            above_threshold[n] = true; // will be set to true if we are checking sign change for this field
        }


        const p4est_qcoord_t mid_qh = P4EST_QUADRANT_LEN(quad->level + 1);
        for(unsigned short i=0; i<3; i++){
            for (unsigned short j=0;j<3;j++){
//                for (unsigned short k=0; k<3;k++){} TO-DO: NOT YET IMPLEMENTED IN 3D

                // Search for finer points if they exist:
                if((i==1) && (j==1)){ // This corresponds to the cell center, so ignore this case and keep looking
                    continue;
                }
                else if (((i==0) || (i==2)) && ((j==0) || (j==2))){
                    // This corresponds to the 4 nodes which make up the quadrant, which we know exist and know the indices
                    node_found = true;
                    node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx + 2*(j/2) + (i/2)];
                }
                else {
                    // This corresponds to the case of potential finer points owned by neighboring quadrants
                    // We will check whether or not these nodes exist, and if so, we will take the node into our consideration
                    // NOTE: this only looks t the midpoint, so technically we are assuming we are working on a balanced grid (or just only looking one level of refinement higher)

                    // First, define two "quadrants" at the max refinement level -- r we will define ourselves, c we will canonicalize using info from r
                    // These are defined as a quadrant existing at the maximum p4est level -- this is how the node index function takes the node input to see if it exists
                    p4est_quadrant_t r,c;
                    r.level = P4EST_MAXLEVEL;
                    r.x = quad->x + i*mid_qh;
                    r.y = quad->y + j*mid_qh;

                    P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
                    p4est_node_canonicalize(p4est,tree_idx,&r,&c);
                    node_found = index_of_node(&c,nodes,node_idx);

                    if(node_found) we_had_neighbor_point=true;

                }

                if(node_found){
                    P4EST_ASSERT(node_idx < ((p4est_locidx_t) nodes->indep_nodes.elem_count));

                    // Now, check refinement conditions on the LSF:
                    refine = refine || ((fabs(phi_p[node_idx]) - ref_band) <= 0.5*lip*quad_diag);

                    // Additionally, check if there is a sign change across the quadrant in LSF (if so, we will refine)
                    all_pos = all_pos && (phi_p[node_idx]>0);
                    all_neg = all_neg && (phi_p[node_idx]<0);

                    // One of the above statements must be true to proceed because:
                    // --> If any coarsen statement is false, we cannot coarsen, so no point continuing
                    // --> If refinement is possible, we must check all the conditions for possibility of refinement
                    // Note: I check this to save computation time -- no sense looping over all the fields if there is nothing to gain from it

                    double field_val;
                    double criteria_refine = -1.; // Initialize these values to a negative number(which they cannot be), so if they aren't assigned, we can check if they are still less than zero.

                    for(unsigned short n = 0; n<num_fields;n++){
                        if(refine){ // If refine is ever true, we can stop checking and mark the quad for refinement
                            goto end_of_function;
                        }
                        // Check if we are allowed to refine for this specific field,
                        //according to the user-specified custom lmax:
                        bool field_refine_allowed = true; // assume refinement is allowed for each field, then check condition
                        field_refine_allowed = (quad->level < lmax_custom[n]);

                        // Get field val and criteria (same as in coarsen case)
                        if(use_block){
                            // Get the field value from the block vector ptr
                            field_val = fields_block[num_fields*node_idx + n];
                        }
                        else{
                            // Get the field value from the std::vector of PETSc vector ptrs
                            field_val = fields[n][node_idx];
                        }

                        criteria_refine = criteria[2*n + 1];

                        P4EST_ASSERT(criteria_refine>=0.); // Make sure criteria has been defined before continuing
                        switch(diag_opn[2*n + 1]){
                        case DIVIDE_BY:{
                            switch(compare_opn[2*n + 1]){
                            case GREATER_THAN:{
                                //                                    if((fabs(field_val) > criteria_refine/d) && n==0){
                                //                                        printf("Refined at level %d based on vorticity = %0.4f on rank %d \n \n",quad->level,field_val,p4est->mpirank);}
                                refine = refine || ((fabs(field_val) > criteria_refine/max_quad_size) && field_refine_allowed);
                                break;
                            }
                            case LESS_THAN:{
                                refine = refine || ((fabs(field_val) < criteria_refine/max_quad_size) && field_refine_allowed);
                                break;
                            }
                            case SIGN_CHANGE:{
                                checking_sign_change = true;

                                field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                                field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                                above_threshold[n] = (fabs(field_val)>criteria_refine/max_quad_size) && above_threshold[n];

                                break;
                            }
                            default:{
                                throw std::invalid_argument("blah");
                            }
                            } // end of case : divide by
                            break;
                        }
                        case MULTIPLY_BY:{
                            switch(compare_opn[2*n + 1]){
                            case GREATER_THAN:{
                                refine = refine || ((fabs(field_val) > criteria_refine*max_quad_size) && field_refine_allowed);
                                break;
                            }
                            case LESS_THAN:{
                                refine = refine || ((fabs(field_val) < criteria_refine*max_quad_size) && field_refine_allowed);
                                break;
                            }
                            case SIGN_CHANGE:{
                                checking_sign_change = true;

                                field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                                field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                                above_threshold[n] = (fabs(field_val)>criteria_refine*max_quad_size) && above_threshold[n];

                                break;
                            }
                            default:{
                                throw std::invalid_argument("blah");
                            }
                            }
                            break;
                        } // end of case : multiply by
                        case ABSOLUTE:{
                            switch(compare_opn[2*n + 1]){
                            case GREATER_THAN:{
                                refine = refine || ((fabs(field_val) > criteria_refine) && field_refine_allowed);
                                break;
                            }
                            case LESS_THAN:{
                                refine = refine || ((fabs(field_val) < criteria_refine) && field_refine_allowed);
                                break;
                            }
                            case SIGN_CHANGE:{
                                checking_sign_change = true;

                                field_all_pos[n] = field_all_pos[n] && (field_val>0.);
                                field_all_neg[n] = field_all_neg[n] && (field_val<0.);

                                above_threshold[n] = (fabs(field_val)>criteria_refine) && above_threshold[n];

                                break;
                            }
                            case NO_CHECK:{
                                continue;

                            }
                            default:{
                                throw std::invalid_argument("blah");
                            }
                            }
                            break;
                        } // end of case: absolute
                        } // end of switch case on diagonal option
                    } // End of loop over n fields
                } // end of if node found
            } // end of loop over j

          } // End of loop over i

        if((!all_pos && !all_neg)){ // if nodes of the quad have different signs of LSF after checking each node --> interface crosses quad, we should refine
            refine = true;

          }
        // If we had a sign change, check if that happened:
        if(checking_sign_change){
//            PetscPrintf(p4est->mpicomm,"CHECKING FOR A SIGN CHANGE REFINE CASE: \n");
            for(unsigned short n=0;n<num_fields;n++){
                bool field_refine_allowed = true; // assume refinement is allowed for each field, then check condition
                field_refine_allowed = (quad->level < lmax_custom[n]);

                bool sign_change = (!field_all_pos[n] && !field_all_neg[n]) && field_refine_allowed;

                if(sign_change && we_had_neighbor_point && above_threshold[n]){
                    refine = (refine || sign_change) ;
                }
            }
        } // end of checking sign change

        } // End of if refine possible


end_of_function:
      // Now --> Apply the results of the check:
      if(refine){
          quad->p.user_int = REFINE_QUADRANT;
        }
      else if (coarsen){
          quad->p.user_int = COARSEN_QUADRANT;

        }
      else {
          quad->p.user_int = SKIP_QUADRANT;
        }
    } // End of if statement to check for refine and coarsening
} // end of function

// END : ELYCE TRYING SOMETHING --------:

int splitting_criteria_tag_t::refine_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad) {
  (void) p4est;
  (void) which_tree;
  return quad->p.user_int == REFINE_QUADRANT;
}

int splitting_criteria_tag_t::coarsen_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad) {
  (void) p4est;
  (void) which_tree;

  int coarsen = quad[0]->p.user_int == COARSEN_QUADRANT;
  for (short i = 1; i<P4EST_CHILDREN; i++)
    coarsen = coarsen && (quad[i]->p.user_int == COARSEN_QUADRANT);
  return coarsen;
}

void splitting_criteria_tag_t::init_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad) {
	(void) p4est;
	(void) which_tree;
  quad->p.user_int = NEW_QUADRANT;
}

bool splitting_criteria_tag_t::refine_and_coarsen(p4est_t* p4est, const p4est_nodes_t* nodes, const double *phi, bool finest_in_negative_flag) {

  double f[P4EST_CHILDREN];
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      p4est_locidx_t qu_idx  = q + tree->quadrants_offset;

      for (short i = 0; i<P4EST_CHILDREN; i++)
        f[i] = phi[nodes->local_nodes[qu_idx*P4EST_CHILDREN + i]];
      if(refine_only_inside)  tag_quadrant_inside(p4est, quad, it, f);
      else                    tag_quadrant(p4est, quad, it, f, finest_in_negative_flag);
    }
  }

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_tag_t::coarsen_fn, splitting_criteria_tag_t::init_fn);
  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_tag_t::refine_fn,  splitting_criteria_tag_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      if (quad->p.user_int == NEW_QUADRANT) {
        is_grid_changed = true;
        goto function_end;
      }
    }
  }

function_end:
  MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);

  return is_grid_changed;
}

bool splitting_criteria_tag_t::refine(p4est_t* p4est, const p4est_nodes_t* nodes, const double *phi, bool finest_in_negative_flag) {

  double f[P4EST_CHILDREN];
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t qu_idx  = q + tree->quadrants_offset;

      for (short i = 0; i<P4EST_CHILDREN; i++)
        f[i] = phi[nodes->local_nodes[qu_idx*P4EST_CHILDREN + i]];
      if(refine_only_inside)  tag_quadrant_inside(p4est, quad, it, f);
      else                    tag_quadrant(p4est, quad, it, f, finest_in_negative_flag);
    }
  }

  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_tag_t::refine_fn,  splitting_criteria_tag_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      if (quad->p.user_int == NEW_QUADRANT) {
        is_grid_changed = true;
        goto function_end;
      }
    }
  }

function_end:
  MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);

  return is_grid_changed;
}

// ELYCE TRYING SOMETHING -------:

bool splitting_criteria_tag_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi,
                                                  const unsigned int num_fields, bool use_block, bool enforce_uniform_band,
                                                  double refine_band,double coarsen_band,
                                                  Vec *fields,Vec fields_block,
                                                  std::vector<double> criteria, std::vector<compare_option_t> compare_opn,
                                                  std::vector<compare_diagonal_option_t> diag_opn,std::vector<int> lmax_custom){
  PetscErrorCode ierr;
  bool is_grid_changed;
 // note to self: consider changing the double array fields_p to a standard vector instead for consistent coding practices
  const double* phi_p;
 const double* fields_p[num_fields];
  const double* fields_block_p;

  // Get appropriate arrays -- phi, and either PETSc block vector of fields, or vector of PETSC Vector fields
  // Also -- check our assumptions, make sure everything is provided correctly
  ierr = VecGetArrayRead(phi,&phi_p);
  if(use_block){
      ierr = VecGetArrayRead(fields_block,&fields_block_p); CHKERRXX(ierr);

      // Make sure other option is set to NULL, since at this point we assume we are using block -- if other option isn't NULL, will get an error when we call the next refine and coarsen function, because the object type of fields won't be a vector of doubles anymore
      for(unsigned int i =0; i < num_fields; i++){
          fields[i] = NULL;
        }
  }// end of "if use block /else" statement -- if portion
  else{
      for (unsigned int i = 0; i < num_fields; i++){
          ierr = VecGetArrayRead(fields[i],&fields_p[i]);
        }
      //P4EST_ASSERT(fields_block == NULL); // if we are using list of fields, then block fields should be set to NULL, otherwise will get an error when we call the next refine and coarsen function bc of a datatype mismatch
    } // end of "if use block /else" statement -- else portion


  // Call inner function which uses the pointers:
  is_grid_changed = refine_and_coarsen(p4est,nodes,phi_p,num_fields,use_block,enforce_uniform_band,refine_band,coarsen_band,fields_p,fields_block_p,criteria,compare_opn,diag_opn,lmax_custom);


  // Restore appropriate arrays:
  ierr = VecRestoreArrayRead(phi,&phi_p);
  if(use_block){
      ierr = VecRestoreArrayRead(fields_block,&fields_block_p);CHKERRXX(ierr);
    }
  else{
      for (unsigned int i = 0; i<num_fields; i++){
          ierr = VecRestoreArrayRead(fields[i],&fields_p[i]);
        }
    }
  return is_grid_changed;
}

bool splitting_criteria_tag_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes,
                                                  const double *phi_p, const unsigned int num_fields,
                                                  bool use_block,bool enforce_uniform_band,double refine_band,double coarsen_band,
                                                  const double** fields,const double* fields_block,
                                                  std::vector<double> criteria, std::vector<compare_option_t> compare_opn,
                                                  std::vector<compare_diagonal_option_t> diag_opn,std::vector<int> lmax_custom){
  // WARNING: This function has not yet been validated in 3d

  // Option lists are provided in the following format:
  // opn = {coarsen_field_1, refine_field_1, coarsen_field_2, refine_field_2, ......, coarsen_field_n, refine_field_n}, where n = num_fields
  // Thus the length of the list is 2*num_fields
  // Coarsen options are accessed as opn[2*i], for i = 0,..., num_fields
  // Refine options are accessed as opn[2*i + 1]
  // Compare option:
  // - 0 --> less than
  // - 1 --> greater than

  // Diag option:
  // - 0 --> divide criteria by diag
  // - 1 --> multiply criteria by diag
  // - 2 --> Neither, compare values

  //compare_opn_array = compare_opn.data(); // access the array , technically points to the first element in the array, so compare_opn_array = compare_opn[0], compare_opn_array[1] = compare_opn[1]

  // Assert that provided vectors are the appropriate lengths:
  P4EST_ASSERT(2*num_fields == criteria.size());
  P4EST_ASSERT(2*num_fields == compare_opn.size());
  P4EST_ASSERT(2*num_fields == diag_opn.size());

  if(use_block){
    }
  else {
      //P4EST_ASSERT(num_fields == fields.size());
    }

  // First, loop over all the quadrants and tag them for possible refinement/coarsening
  foreach_tree(tr,p4est){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tr);

    foreach_local_quad(q,tree){
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      tag_quadrant(p4est,quad,tr,quad_idx,nodes,phi_p,num_fields,use_block,enforce_uniform_band, refine_band,coarsen_band,fields,fields_block,criteria,compare_opn,diag_opn,lmax_custom);
    }
  }

  // Now that quadrants are tagged, call the refine and coarsen functions:
  my_p4est_coarsen(p4est,P4EST_FALSE,splitting_criteria_tag_t::coarsen_fn,splitting_criteria_tag_t::init_fn);
  my_p4est_refine(p4est,P4EST_FALSE,splitting_criteria_tag_t::refine_fn,splitting_criteria_tag_t::init_fn);

  int is_grid_changed = false;

  // Now check to see if there are any new quadrants to see if the grid is changed:
  foreach_tree(tr,p4est){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tr);
    foreach_local_quad(q,tree){
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants,q);

      if(quad->p.user_int == NEW_QUADRANT){
          is_grid_changed = true;
          goto function_end;
        }
    }
  }
  function_end:
    // Check how the grid has changed across all the processes:
    int global_is_grid_changed = false;
    MPI_Allreduce(&is_grid_changed,&global_is_grid_changed,1,MPI_INT,MPI_LOR,p4est->mpicomm);
    return global_is_grid_changed;
}

// END: ELYCE TRYING SOMETHING--------------

p4est_bool_t
refine_grad_cf(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) which_tree;

  splitting_criteria_grad_t *sp = (splitting_criteria_grad_t*)p4est->user_pointer;
  if (quad->level < sp->min_lvl)
    return P4EST_TRUE;
  else if (quad->level >= sp->max_lvl)
    return P4EST_FALSE;
  else
  {
    const CF_DIM& cf = *sp->cf;

    double x[P4EST_DIM], dx[P4EST_DIM];
    quad_xyz(p4est, quad, x);
    dxyz_quad(p4est, quad, dx);

#ifdef P4_TO_P8
    double fx = (cf(x[0] + 0.5*dx[0], x[1], x[2]) - cf(x[0] - 0.5*dx[0], x[1], x[2]))/dx[0];
    double fy = (cf(x[0], x[1] + 0.5*dx[1], x[2]) - cf(x[0], x[1] - 0.5*dx[1], x[2]))/dx[1];
    double fz = (cf(x[0], x[1], x[2] + 0.5*dx[2]) - cf(x[0], x[1], x[2] - 0.5*dx[2]))/dx[2];

    return MIN(dx[0], dx[1], dx[2]) * sqrt(SQR(fx)+SQR(fy)+SQR(fz))/sp->fmax > sp->tol;
#else
    double f[] = {
      cf(x[0] - 0.5*dx[0], x[1] - 0.5*dx[1]),
      cf(x[0] + 0.5*dx[0], x[1] - 0.5*dx[1]),
      cf(x[0] - 0.5*dx[0], x[1] + 0.5*dx[1]),
      cf(x[0] + 0.5*dx[0], x[1] + 0.5*dx[1]),
    };
    double fx = 0.5*((f[1]+f[3]) - (f[0]+f[2]))/dx[0];
    double fy = 0.5*((f[2]+f[3]) - (f[0]+f[1]))/dx[1];
    return (MIN(dx[0], dx[1])*sqrt(SQR(fx)+SQR(fy))/sp->fmax) >= sp->tol;
//    double diag = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
//    double fx1 = (f[1] - f[0])/dx[0];
//    double fy1 = (f[3] - f[2])/dx[1];
//    double fx2 = (f[3] - f[0])/diag;
//    double fy2 = (f[2] - f[1])/diag;

//    return (MIN(dx[0], dx[1])*0.5*(sqrt(SQR(fx1)+SQR(fy1))+sqrt(SQR(fx2)+SQR(fy2)))/sp->fmax) >= sp->tol;
#endif
  }
}

p4est_bool_t
coarsen_down_to_lmax (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  (void) which_tree;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  if (quad->level > data->max_lvl)
    return P4EST_TRUE;
  else
    return P4EST_FALSE;
}

/////////////////////////////////////////// Splitting criteria band methods ////////////////////////////////////////////

void splitting_criteria_band_t::tag_quadrant( p4est_t const *p4est, p4est_locidx_t quadIdx, p4est_topidx_t treeIdx,
											  p4est_nodes_t const *nodes, double const *phiReadPtr )
{
	auto *tree = (p4est_tree_t*)sc_array_index( p4est->trees, treeIdx );	// Which tree does this quadrant belong to?
	auto *quad = (p4est_quadrant_t*)sc_array_index( &tree->quadrants, quadIdx - tree->quadrants_offset );	// Quadrant.

	if( quad->level < min_lvl )
		quad->p.user_int = REFINE_QUADRANT; 	// If coarser than min refinement level, mark for refinement.
	else if( quad->level > max_lvl )
		quad->p.user_int = COARSEN_QUADRANT; 	// If finer than max refinement level, mark for coarsening.
	else
	{											// Decide if quadrant should be tag for refinement or coarsening.
		double dxyz[P4EST_DIM];
		double dxyzMin, diagMin;
    get_dxyz_min( p4est, dxyz, &dxyzMin, &diagMin );	// Domain metrics.
		const double quadDiag = diagMin * (1 << (max_lvl - quad->level));

		bool coarsen = (quad->level > min_lvl);	// Beging by deciding if quadrant must be coarsened.
		if( coarsen )
		{
			bool coarsenInterface = true;		// Coarsening due to distance to interface?
			bool coarsenBand = true;			// Coarsening due to lying within band around the interface?
			p4est_locidx_t nodeIdx;

			// Check the P4EST_CHILDREN vertices of the quadrant and verify all conditions in each.
			for( int v = 0; v < P4EST_CHILDREN; v++ )
			{
				nodeIdx = nodes->local_nodes[P4EST_CHILDREN * quadIdx + v];

				coarsenInterface = coarsenInterface && (ABS( phiReadPtr[nodeIdx] ) >= lip * 2.0 * quadDiag);
				coarsenBand = coarsenBand && (ABS( phiReadPtr[nodeIdx] ) > MAX( 1.0, _bandWidth ) * diagMin);

				// Need ALL of the coarsening conditions satisfied to coarsen the quadrant.
				coarsen = coarsenInterface && coarsenBand;
				if( !coarsen )
					break;
			}
		}

		bool refine = (quad->level < max_lvl);	// Check now if quadrant must be refined.
		if( refine )
		{
			bool refineInterface = false;		// Refining due to distance to interface?
			bool refineBand = false;			// Refining due to lying within band around the interface?
			p4est_locidx_t nodeIdx;

			// Check the P4EST_CHILDREN vertices of the quadrant and verify all conditions in each.
			for( int v = 0; v < P4EST_CHILDREN; v++ )
			{
				nodeIdx = nodes->local_nodes[P4EST_CHILDREN * quadIdx + v];

				refineInterface = refineInterface || (ABS( phiReadPtr[nodeIdx] ) <= lip * quadDiag);
				refineBand = refineBand || (ABS( phiReadPtr[nodeIdx] ) < MAX( 1.0, _bandWidth ) * diagMin);

				// Need AT LEAST ONE of the refining conditions satisfied to refine the quadrant.
				refine = refineInterface || refineBand;
				if( refine )
					break;
			}
		}

		if( refine )
			quad->p.user_int = REFINE_QUADRANT;
		else if( coarsen )
			quad->p.user_int = COARSEN_QUADRANT;
		else
			quad->p.user_int = SKIP_QUADRANT;
	}
}

bool splitting_criteria_band_t::refine_and_coarsen_with_band( p4est_t *p4est, p4est_nodes_t const *nodes, const double *phiReadPtr )
{
	// Tag the quadrants that need to be refined or coarsened.
	for( p4est_topidx_t treeIdx = p4est->first_local_tree; treeIdx <= p4est->last_local_tree; treeIdx++ )
	{
		auto *tree = (p4est_tree_t *)sc_array_index( p4est->trees, treeIdx );
		for( p4est_locidx_t q = 0; q < tree->quadrants.elem_count; q++ )
		{
			p4est_locidx_t quadIdx = q + tree->quadrants_offset;
			tag_quadrant( p4est, quadIdx, treeIdx, nodes, phiReadPtr );
		}
	}

	my_p4est_coarsen( p4est, P4EST_FALSE, splitting_criteria_band_t::coarsen_fn, splitting_criteria_band_t::init_fn );
	my_p4est_refine( p4est, P4EST_FALSE, splitting_criteria_band_t::refine_fn, splitting_criteria_band_t::init_fn );

	int hasGridchanged = false;
	for( p4est_topidx_t treeIdx = p4est->first_local_tree; !hasGridchanged && treeIdx <= p4est->last_local_tree; treeIdx++ )
	{
		auto *tree = (p4est_tree_t *)sc_array_index( p4est->trees, treeIdx );
		for( p4est_locidx_t q = 0; !hasGridchanged && q < tree->quadrants.elem_count; q++ )
		{
			auto *quad = (p4est_quadrant_t *)sc_array_index( &tree->quadrants, q );
			if( quad->p.user_int == NEW_QUADRANT )
				hasGridchanged = true;
		}
	}

	int mpiret = MPI_Allreduce( MPI_IN_PLACE, &hasGridchanged, 1, MPI_INT, MPI_LOR, p4est->mpicomm );
	SC_CHECK_MPI( mpiret );

	return hasGridchanged;
}


p4est_bool_t refine_levelset_cf_and_uniform_band_shs( p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad )
{
	auto *data = (splitting_criteria_cf_and_uniform_band_shs_t *) p4est->user_pointer;
	if( data->SPECIAL_REFINEMENT )
		throw std::runtime_error( "refine_levelset_cf_and_uniform_band_shs function is not allowed when special refinement option is enabled!" );

	if( quad->level < data->min_lvl )		// Refine until we get to the desired min level.
		return P4EST_TRUE;
	else if( quad->level >= data->max_lvl )	// Stop refining beyond max level.
		return P4EST_FALSE;
	else
	{
		p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN * which_tree + 0];
		p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN * which_tree + P4EST_CHILDREN - 1];

		double tree_xmin = p4est->connectivity->vertices[3 * v_m + 0];
		double tree_ymin = p4est->connectivity->vertices[3 * v_m + 1];
#ifdef P4_TO_P8
		double tree_zmin = p4est->connectivity->vertices[3 * v_m + 2];
#endif

		double tree_xmax = p4est->connectivity->vertices[3 * v_p + 0];
		double tree_ymax = p4est->connectivity->vertices[3 * v_p + 1];
#ifdef P4_TO_P8
		double tree_zmax = p4est->connectivity->vertices[3 * v_p + 2];
#endif

		double dmin = (double)P4EST_QUADRANT_LEN( quad->level ) / (double)P4EST_ROOT_LEN;
		double dx = (tree_xmax - tree_xmin) * dmin;
		double dy = (tree_ymax - tree_ymin) * dmin;
#ifdef P4_TO_P8
		double dz = (tree_zmax - tree_zmin) * dmin;
#endif
		double smallest_dy = (tree_ymax - tree_ymin) * (double)P4EST_QUADRANT_LEN( data->max_lvl ) / P4EST_ROOT_LEN;

		// Use smallest_dy to define the limits for mid-level-cell refinement.
		std::vector<double> midBounds;
		bool midLvlCellsOK = data->getBandedBounds( smallest_dy, midBounds );
		const int NUM_MID_LEVELS = (int)midBounds.size();

		double x = (tree_xmax - tree_xmin) * (double)quad->x / (double)P4EST_ROOT_LEN + tree_xmin;
		double y = (tree_ymax - tree_ymin) * (double)quad->y / (double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
		double z = (tree_zmax - tree_zmin) * (double)quad->z / (double)P4EST_ROOT_LEN + tree_zmin;
#endif
		const CF_DIM& phi = *(data->phi);
		double f;
		bool vmmm_is_neg = phi( DIM( x, y, z ) ) <= 0.0;
		bool is_crossed = false;
#ifdef P4_TO_P8
		for( unsigned short ck = 0; ck <= 2; ++ck )
#endif
			for( unsigned short cj = 0; cj <= 2; ++cj )
				for( unsigned short ci = 0; ci <= 2; ++ci )
				{
					double xyz[P4EST_DIM] = {DIM( x + ci * 0.5 * dx, y + cj * 0.5 * dy, z + ck * 0.5 * dz )};
					f = phi( DIM( xyz[0], xyz[1], xyz[2] ) );
					is_crossed = is_crossed || (vmmm_is_neg != (f <= 0.0));
					if( fabs( f ) < data->uniformBand() * smallest_dy || is_crossed )
						return P4EST_TRUE;

					// Check if any of quad's corners/midpoints are in the negative domain.
					if( f <= 0 )
					{
						// If after the above condition we didn't refine a quad, we need to check cells next to the
						// air interface (which is not considered by the solid-ridge-based level-set function).
						double minDistToWall = MIN( ABS( xyz[1] + data->DELTA ), ABS( xyz[1] - data->DELTA ) );
						if( minDistToWall < data->uniformBand() * smallest_dy )
							return P4EST_TRUE;	// Enforce uniform band along the wall, regardless of interface type.

						// Check the mid-level cells (and their bands) only if requested and valid.
						if( NUM_MID_LEVELS > 0 && midLvlCellsOK )
						{
							int boundIdx = MAX( 0, MIN( (data->max_lvl - 1) - (quad->level + 1), NUM_MID_LEVELS - 1 ) );	// We want to check if we can go one level up.
							if( minDistToWall < midBounds[boundIdx] && quad->level < data->max_lvl - boundIdx - 1 )
								return P4EST_TRUE;
						}
					}
				}
	}
	return refine_levelset_cf( p4est, which_tree, quad );
}

int splitting_criteria_cf_and_uniform_band_shs_t::refine_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad) {
	(void) p4est;
	(void) which_tree;
	return quad->p.user_int == REFINE_QUADRANT;
}

int splitting_criteria_cf_and_uniform_band_shs_t::coarsen_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad) {
	(void) p4est;
	(void) which_tree;

	int coarsen = quad[0]->p.user_int == COARSEN_QUADRANT;
	for (short i = 1; i<P4EST_CHILDREN; i++)
		coarsen = coarsen && (quad[i]->p.user_int == COARSEN_QUADRANT);
	return coarsen;
}

void splitting_criteria_cf_and_uniform_band_shs_t::init_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad) {
	(void) p4est;
	(void) which_tree;
	quad->p.user_int = NEW_QUADRANT;
}

bool splitting_criteria_cf_and_uniform_band_shs_t::refine_and_coarsen( p4est_t* p4est, p4est_nodes_t* nodes, Vec phi )
{
	if( state < STATE::COARSEN_AND_REFINE_MAX_LVL || (state == STATE::REFINE_MAX_LVL_PLASTRON && !SPECIAL_REFINEMENT) || state > STATE::REFINE_MID_BANDS )
		throw std::runtime_error( "[CASL_ERROR] splitting_criteria_cf_and_uniform_band_shs_t::refine_and_coarsen: Invalid state!" );

	const double *phi_p;
	phi_p = nullptr;
	CHKERRXX( VecGetArrayRead( phi, &phi_p ) );

	double tree_dimensions[P4EST_DIM];		// Cell dimensions in each direction.
	p4est_locidx_t *t2v = p4est->connectivity->tree_to_vertex;
	double *v2c = p4est->connectivity->vertices;

	// Tag the quadrants that need to be refined or coarsened.
	for( p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx )
	{
		p4est_tree_t *tree = p4est_tree_array_index( p4est->trees, tree_idx );
		for( u_char dir = 0; dir < P4EST_DIM; ++dir )
			tree_dimensions[dir] = v2c[3 * t2v[P4EST_CHILDREN * tree_idx + P4EST_CHILDREN - 1] + dir] - v2c[3 * t2v[P4EST_CHILDREN * tree_idx + 0] + dir];

		const double plastron_smallest_dy = tree_dimensions[1] * (((double) P4EST_QUADRANT_LEN((int8_t) PLASTRON_MAX_LVL))/((double) P4EST_ROOT_LEN));
		std::vector<double> midBounds;
		getBandedBounds( plastron_smallest_dy, midBounds );

		for( p4est_locidx_t q = 0; q < tree->quadrants.elem_count; ++q )
		{
			p4est_locidx_t quad_idx = q + tree->quadrants_offset;
			tag_quadrant( p4est, quad_idx, tree_idx, nodes, tree_dimensions, phi_p, midBounds );
		}
	}

	my_p4est_coarsen( p4est, P4EST_FALSE, splitting_criteria_cf_and_uniform_band_shs_t::coarsen_fn, splitting_criteria_cf_and_uniform_band_shs_t::init_fn );
	my_p4est_refine( p4est, P4EST_FALSE, splitting_criteria_cf_and_uniform_band_shs_t::refine_fn, splitting_criteria_cf_and_uniform_band_shs_t::init_fn );

	int is_grid_changed = false;
	CHKERRXX( PetscPrintf( p4est->mpicomm, ">> Refining/coarsening%s... ", (SPECIAL_REFINEMENT? " with special refinement" : "") ) );
	for( p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it )
	{
		p4est_tree_t *tree = p4est_tree_array_index( p4est->trees, it );
		for( size_t q = 0; q < tree->quadrants.elem_count; ++q )
		{
			p4est_quadrant_t *quad = p4est_quadrant_array_index( &tree->quadrants, q );
			if( quad->p.user_int == NEW_QUADRANT )
			{
				is_grid_changed = true;
				goto function_end;
			}
		}
	}

function_end:
	MPI_Allreduce( MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm );
	CHKERRXX( PetscPrintf( p4est->mpicomm, "done!\n" ) );

	CHKERRXX( VecRestoreArrayRead( phi, &phi_p ) );
	return is_grid_changed;
}

#ifdef P4_TO_P8
double splitting_criteria_cf_and_uniform_band_shs_t::_normalize_z( const double &z ) const
{
	return my_fmod( z + XYZ_MAX[2], P );
}
#endif

double splitting_criteria_cf_and_uniform_band_shs_t::_offset() const
{
#ifdef P4_TO_P8
	if( !SPANWISE )
		return 0.1 * XYZ_DIM[2] / (N_TREES[2] * (1 << max_lvl));
#endif
	return 0.5 * XYZ_DIM[0] / (N_TREES[0] * (1 << max_lvl));
}

bool splitting_criteria_cf_and_uniform_band_shs_t::is_ridge( const double xyz[P4EST_DIM] ) const
{
#ifdef P4_TO_P8
	if( !SPANWISE )
		return _offset() >= _normalize_z( xyz[2] ) || _normalize_z( xyz[2] ) >= P * GF - _offset();
#endif
	return my_fmod( xyz[0] - XYZ_MIN[0] - _offset(), P ) / P >= GF;
}

void splitting_criteria_cf_and_uniform_band_shs_t::tag_quadrant( p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx,
																 p4est_nodes_t *nodes, const double *tree_dimensions, const double *phi_p,
																 const std::vector<double>& midBounds )
{
	p4est_tree_t *tree = p4est_tree_array_index( p4est->trees, tree_idx );
	p4est_quadrant_t *quad = p4est_quadrant_array_index( &tree->quadrants, quad_idx - tree->quadrants_offset );
	const int NUM_MID_LEVELS = (int)midBounds.size();
	const double R = (1 - GF) * P;			// Ridge width.

	if( quad->level < min_lvl )
		quad->p.user_int = REFINE_QUADRANT;
	else if( quad->level > max_lvl )
		quad->p.user_int = COARSEN_QUADRANT;
	else
	{
//		const double quad_denom = (( double ) P4EST_QUADRANT_LEN( quad->level )) / (( double ) P4EST_ROOT_LEN);
//		const double quad_diag = sqrt(SUMD( SQR( tree_dimensions[0] ), SQR( tree_dimensions[1] ), SQR( tree_dimensions[2] ))) * quad_denom;
		const double plastron_smallest_dy = tree_dimensions[1] * (((double) P4EST_QUADRANT_LEN((int8_t) PLASTRON_MAX_LVL))/((double) P4EST_ROOT_LEN));

		const double h0 = uniform_band * plastron_smallest_dy * 0.1;
		auto wave0 = [&](const double& t) -> double {
			return h0 * pow( cos( M_PI * (t + R/2) / P) , 6);
		};

		const double h1 = uniform_band * plastron_smallest_dy * 0.5;		// Height of max level wave in specially refined grid.  Note that comparison is made with respect to plastron's band.
		auto wave1 = [&](const double& t) -> double {				// First wave for special refinement (closest to wall).
			double ret = h1 * 2.0;//* pow( cos( M_PI * (t + R/2) / P) , 6);
			return WALL_REFINEMENT ? MAX( ret, wave0( t ) ) : ret;
		};

		bool coarsen = false;
		if( state == STATE::COARSEN_AND_REFINE_MAX_LVL )			// We coarsen only in the first round, otherwise the grid never stabilizes.
		{
			coarsen = (quad->level > min_lvl);
			if( coarsen )
			{
				bool cor_band;
				p4est_locidx_t node_idx;

				for( unsigned char k = 0; k < P4EST_CHILDREN; ++k )
				{
					node_idx = nodes->local_nodes[P4EST_CHILDREN * quad_idx + k];
					double xyz[P4EST_DIM];
					node_xyz_fr_n( node_idx, p4est, nodes, xyz );
					double minDistToWall = MIN( ABS( xyz[1] + DELTA ), ABS( xyz[1] - DELTA ) );

					// Coarsening if we are out of the (possibly wavy) uniform band?
					if( SPECIAL_REFINEMENT)
						cor_band = minDistToWall > wave1( xyz[2] );
					else
						cor_band = minDistToWall > h1;

					coarsen = cor_band;
					if( !coarsen )		// Mark cell for coarsening if all cell nodes are marked for coarsening.
						break;
				}
			}
		}

		const double h2 = SPECIAL_REFINEMENT ? uniform_band * plastron_smallest_dy / 2 : 0;	// Height of second wave (at max lvl of refinement on plastron).
		double o2 = NUM_MID_LEVELS > 0 ? midBounds[0] : DELTA * 0.325;
		auto wave2 = [&](const double& t) -> double {										// Second wave for special refinement lies above wave1.
			double ret = h1 + (o2 - uniform_band * plastron_smallest_dy) / 2 + h2 / 2 * cos( 2 * M_PI * (t + R/2) / P);
			return MAX( ret, wave1( t ) );
		};

		bool refine = quad->level < max_lvl - (state > 0? int( SPECIAL_REFINEMENT ) : 0);
		double xyz[P4EST_DIM];
		if( refine )
		{
			refine = false;
			p4est_locidx_t node_idx;
			bool node_found;
			// check possibly finer points
			const p4est_qcoord_t mid_qh = P4EST_QUADRANT_LEN( quad->level + 1 );
#ifdef P4_TO_P8
			for( unsigned char k = 0; k < 3; ++k )
#endif
				for( unsigned char j = 0; j < 3; ++j )
					for( unsigned char i = 0; i < 3; ++i )
					{
						if( ANDD( i == 1, j == 1, k == 1 ))
							continue;
						if( ANDD( i == 0 || i == 2, j == 0 || j == 2, k == 0 || k == 2 ))
						{
							node_found = true;
							node_idx = nodes->local_nodes[P4EST_CHILDREN * quad_idx +
														  SUMD( i / 2, 2 * (j / 2), 4 * (k / 2))]; // integer divisions!
						}
						else
						{
							p4est_quadrant_t r, c;
							r.level = P4EST_MAXLEVEL;
							r.x = quad->x + i * mid_qh;
							r.y = quad->y + j * mid_qh;
#ifdef P4_TO_P8
							r.z = quad->z + k * mid_qh;
#endif
							P4EST_ASSERT ( p4est_quadrant_is_node( &r, 0 ));
							p4est_node_canonicalize( p4est, tree_idx, &r, &c );
							node_found = index_of_node( &c, nodes, node_idx );
						}
						if( node_found )
						{
							P4EST_ASSERT( node_idx < (( p4est_locidx_t ) nodes->indep_nodes.elem_count));
							node_xyz_fr_n( node_idx, p4est, nodes, xyz );
							double minDistToWall = MIN( ABS( xyz[1] + DELTA ), ABS( xyz[1] - DELTA ) );

							if( state == STATE::COARSEN_AND_REFINE_MAX_LVL )	// Are we refining for max lvl next to wall?
								if( SPECIAL_REFINEMENT )
									if( WALL_REFINEMENT )
										refine = minDistToWall <= ((quad->level < max_lvl - 1) ? wave1( xyz[2] ) : wave0( xyz[2] ));
									else
										refine = minDistToWall <=  wave1( xyz[2] );
								else
									refine = minDistToWall <= h1;
							else if( state == STATE::REFINE_MAX_LVL_PLASTRON )	// Are we refining for the second wave and can refine?
								refine = quad->level < PLASTRON_MAX_LVL && minDistToWall <= wave2( xyz[2] );
							else if( NUM_MID_LEVELS > 0 )						// Are we now refining the usual straight mid-level bands.
							{
								int boundIdx = MAX( 0, MIN( (PLASTRON_MAX_LVL - 1) - (quad->level + 1), NUM_MID_LEVELS - 1 ) );	// Can go one level up?
								if( minDistToWall < midBounds[boundIdx] && quad->level < PLASTRON_MAX_LVL - boundIdx - 1 )
									refine = true;
							}

							if( refine )		// Refine if at least one grid point is marked for refinement.
								goto end_of_function;
						}
					}
		}

end_of_function:
#ifdef DEBUG
		if( coarsen && refine )
			std::cerr << "Octant with point [" << xyz[0] << ", " << xyz[1] <<
					  ONLY3D( "," << xyz[2] <<) "] is marked for coarsening and refinement!" << std::endl;
#endif

		if( refine )
			quad->p.user_int = REFINE_QUADRANT;
		else if( coarsen )
			quad->p.user_int = COARSEN_QUADRANT;
		else
			quad->p.user_int = SKIP_QUADRANT;
	}
}

bool splitting_criteria_cf_and_uniform_band_shs_t::getBandedBounds( const double& plastronSmallest_dy, std::vector<double>&midBounds ) const
{
	// Use plastron smallest dy to define the limits for mid-level-cell refinement.  In special refinement, max_lvl is 1 above max lvl on plastron.
	const int NUM_MID_LEVELS = PLASTRON_MAX_LVL - min_lvl - 1;
	double midEffectiveDist = DELTA * LMID_DELTA_PERCENT - uniformBand() * plastronSmallest_dy;
	bool midLvlCellsOK = false;
	midBounds.clear();
	if( NUM_MID_LEVELS > 0 && LMID_DELTA_PERCENT > 0 )	// Use option only if user allowed it with a percent > 0.
	{
		if( midEffectiveDist <= 0 )		// Check mid-level cells have space to be placed; if not, don't enforce anything.
		{
			std::cerr << "[CASL_WARNING] splitting_criteria_cf_and_uniform_band_shs_t::getBandedBounds: The uniform band of finest cells "
					  << "extends beyond the requested space for mid-level cells!  Check your calculations..." << std::endl;
		}
		else
		{
			midBounds.reserve( NUM_MID_LEVELS );						// Mid bands are spaced like (b, 2b, 4b,..., 2^{n-1}b).
			double b = midEffectiveDist / ((1<<NUM_MID_LEVELS) - 1);	// where n is number of mid levels.
			for( int i = 0; i < NUM_MID_LEVELS; i++ )
			{
				if( i == 0 )
					midBounds.push_back( uniformBand() * plastronSmallest_dy + (1 << i) * b );
				else
					midBounds.push_back( midBounds.back() + (1 << i) * b );
			}
			midLvlCellsOK = true;
		}
	}

	return midLvlCellsOK;
}
