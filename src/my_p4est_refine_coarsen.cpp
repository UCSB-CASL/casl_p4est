#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_utils.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_utils.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#endif
#include <sc_search.h>
#include <iostream>

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
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    double x = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    CF_DIM& phi = *(data->phi);

    double lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
          f[SUMD(ci, 2*cj, 4*ck)] = phi(DIM(x+ci*dx, y+cj*dy, z+ck*dz));
          if (fabs(f[SUMD(ci, 2*cj, 4*ck)]) <= 0.5*lip*d)
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

    CF_DIM&  phi = *(data->phi);

    double f;
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
          f = phi(DIM(x+ci*dx, y+cj*dy, z+ck*dz));
          if(fabs(f) < data->uniform_band*smallest_dxyz_max)
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
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    double x = (tree_xmax-tree_xmin)*(double)quad[0]->x/(double)P4EST_ROOT_LEN + tree_xmin;
    double y = (tree_ymax-tree_ymin)*(double)quad[0]->y/(double)P4EST_ROOT_LEN + tree_ymin;
#ifdef P4_TO_P8
    double z = (tree_zmax-tree_zmin)*(double)quad[0]->z/(double)P4EST_ROOT_LEN + tree_zmin;
#endif

    CF_DIM &phi = *(data->phi);
    double lip = data->lip;

    double f[P4EST_CHILDREN];
#ifdef P4_TO_P8
    for (unsigned short ck = 0; ck<2; ++ck)
#endif
      for (unsigned short cj = 0; cj<2; ++cj)
        for (unsigned short ci = 0; ci <2; ++ci){
          f[SUMD(ci, 2*cj, 4*ck)] = phi(DIM(x+ci*dx, y+cj*dy, z+ck*dz));
          if (fabs(f[SUMD(ci, 2*cj, 4*ck)]) <= 0.5*lip*d)
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

    CF_DIM& f = *(data->f);
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

    CF_DIM& f = *(data->f);
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

void splitting_criteria_tag_t::tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t which_tree, const double *f, bool finest_in_negative_flag){
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
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    // refinement based on distance
    bool refine = false, coarsen = true;

    if(finest_in_negative_flag)
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || ((quad->level < max_lvl) && ((f[i] <= 0.5*lip*d) || ((i==0)? false: ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && (quad->level > min_lvl) && (f[i] >= 1.0*lip*d) && ((i==0)? true: ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
      }
    else
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || ((quad->level < max_lvl) && ((fabs(f[i]) <= 0.5*lip*d) || ((i==0)? false: ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && (quad->level > min_lvl) && (fabs(f[i]) >= 1.0*lip*d) && ((i==0)? true: ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
      }

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;
    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;
    else
      quad->p.user_int = SKIP_QUADRANT;
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
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    // refinement based on distance
    bool refine = false, coarsen = true;

    for (short i = 0; i < P4EST_CHILDREN; i++) {
      //                        refine  = refine  || (fabs(f[i]) <= 0.5*lip*d && quad->level < max_lvl);
      //                        coarsen = coarsen && (fabs(f[i]) >= 1.0*lip*d && quad->level > min_lvl);
      refine  = refine  || (fabs(f[i]) <= 0.5*lip*d );
      coarsen = coarsen && (fabs(f[i]) >= 1.0*lip*d );
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

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_tag_t::coarsen_fn, splitting_criteria_tag_t::init_fn);
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


bool splitting_criteria_tag_t::refine(p4est_t *p4est, const p4est_nodes_t *nodes, const double *phi, bool finest_in_negative_flag) {

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
    CF_DIM& cf = *sp->cf;

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
