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
    bool vmmm_is_neg = phi(DIM(x, y, z)) <= 0.0;
    bool is_crossed = false;;
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
#ifdef P4_TO_P8
    double dz = (tree_zmax-tree_zmin) * dmin;
#endif

    double d = sqrt(SUMD(dx*dx, dy*dy, dz*dz));

    // refinement based on distance
                bool refine = false, coarsen = true;

    if(finest_in_negative_flag)
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || (quad->level < max_lvl && (f[i] <= 0.5*lip*d || (i == 0 ? false : ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && quad->level > min_lvl && f[i] >= 1.0*lip*d && (i == 0 ? true : ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
      }
    else
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        refine  = refine  || (quad->level < max_lvl && (fabs(f[i]) <= 0.5*lip*d || (i == 0 ? false : ((f[i] > 0.0 && f[0] <= 0.0) || (f[i] <= 0.0 && f[0] > 0.0)))));
        coarsen = coarsen && quad->level > min_lvl && fabs(f[i]) >= 1.0*lip*d && (i == 0 ? true : ((f[i] > 0.0 && f[0] > 0.0) || (f[i] <= 0.0 && f[0] <= 0.0)));
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

// ELYCE TRYING SOMETHING --------:

void splitting_criteria_tag_t::tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t tree_idx, p4est_locidx_t quad_idx,p4est_nodes_t *nodes, const double* phi_p, const int num_fields,bool use_block, bool enforce_uniform_band,double refine_band,double coarsen_band,const double** fields,const double* fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn){

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
      const double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // Gives min dimension of the quad--> used to get quad_diag
      const double dx = (tree_xmax-tree_xmin) * dmin;
      const double dy = (tree_ymax-tree_ymin) * dmin;


  #ifdef P4_TO_P8
      double dz = (tree_zmax-tree_zmin) * dmin;

  #endif

  #ifdef P4_TO_P8
      double d = sqrt(dx*dx + dy*dy + dz*dz);
  #else
      const double d = sqrt(dx*dx + dy*dy); // This gives the quad diagonal
  #endif
      double max_tree_dim = MAX((tree_xmax - tree_xmin),(tree_ymax - tree_ymin));

      CODE3D(max_tree_dim = MAX(max_tree_dim,(tree_zmax - tree_zmin)));
      double dxyz_smallest = max_tree_dim*dxyz_min_overall;

      // Check possibility of refining or coarsening:
      bool coarsen_possible = (quad->level > min_lvl); // Will return true if possible to coarsen
      bool refine_possible = (quad->level < max_lvl);  // Will return true if possible to refine

      // Initialize booleans which will track if any refinement or coarsening is allowed -- which is different than possible
      // Only one refinement condition must be true to tag the cell for refining
      // All coarsening conditions must be true to tag the cell for coarsening
      bool coarsen = false;
      bool refine = false;

      // Initialize booleans to check if the LSF changes sign within the quadrant --> if not all pos and not all neg, tag for refinement
      bool all_pos = true;
      bool all_neg = true;

      double ref_band,coars_band;
      ref_band = enforce_uniform_band? dxyz_smallest*refine_band : 0.0;
      coars_band = enforce_uniform_band? dxyz_smallest*coarsen_band : 0.0;

      if(refine_possible || coarsen_possible){
        // Initialize holder for node index in question
        p4est_locidx_t node_idx;

        // Now, loop over the children of the quadrant: -- will check criteria of each field at each node child of the quadrant in question
        for(unsigned short i=0; i<P4EST_CHILDREN; i++){
            node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx + i];

            // First, check conditions on the LSF: -- If LSF won't allow for coarsening, there is no point in checking for more coarsening conditions
            coarsen = coarsen_possible && ((fabs(phi_p[node_idx]) - coars_band) > 1.0*lip*d);

/*            if(coarsen_possible) {
                coarsen = coarsen && ; // ELYCE DEBUGGING
//                if(coarsen) PetscPrintf(p4est->mpicomm,"COARSEN ALLOWED\n");
              } // a*/ //t this point, coarsen is a more restrictive flag than coarsen_possible
//            if (refine_possible) {
//                refine = ((fabs(phi_p[node_idx]) - ref_band) <= 0.5*lip*d);
//              }
            refine = refine_possible && ((fabs(phi_p[node_idx]) - ref_band) <= 0.5*lip*d);

            all_pos = all_pos && (phi_p[node_idx]>0);
            all_neg = all_neg && (phi_p[node_idx]<0);

            // Loop over the number of fields we are taking into consideration: (if still possible to refine or coarsen)
            if(coarsen || refine_possible){
                // One of the above statements must be true to proceed because:
                // --> If any coarsen statement is false, we cannot coarsen, so no point continuing
                // --> If refinement is possible, we must check all the conditions for possibility of refinement
                // Note: I check this to save computation time -- no sense looping over all the fields if there is nothing to gain from it

                double field_val;
                double criteria_coarsen = DBL_MAX;
                double criteria_refine = DBL_MAX;

                for(unsigned short n = 0; n<num_fields;n++){
//                    if(refine){ // If refine is ever true, we can stop checking and mark the quad for refinement
//                        PetscPrintf(p4est->mpicomm,"--> Going to end of function to refine \n");
//                        goto end_of_function;
//                      }
                    // Get the value of the field and store it as a double -- so we don't keep accessing the vector over and over
                    // Need to get it either from a block vector ptr, or from a vector of PETSc vector ptrs, depending on what the user has specified and provided

                    if(use_block){
                        // Get the field value from the block vector ptr
                        field_val = fields_block[num_fields*node_idx + n];
                      }
                    else{
                        // Get the field value from the std::vector of PETSc vector ptrs
                        field_val = fields[n][node_idx];
                      }

                    if(coarsen) criteria_coarsen = criteria[2*n];
                    if(refine_possible) criteria_refine = criteria[2*n + 1];

                    // Switch over the cases for different comparision options to check the refinement and coarsening criteria:
                    if(coarsen){
                        P4EST_ASSERT(criteria_coarsen < DBL_MAX); // Make sure criteria has been defined before continuing
                        switch(diag_opn[2*n]){
                          case DIVIDE_BY:{
                              switch(compare_opn[2*n]){
                                case GREATER_THAN:
                                  coarsen = coarsen && ((fabs(field_val)) > criteria_coarsen/d);
                                  break;
                                case LESS_THAN:
                                  coarsen = coarsen && ((fabs(field_val)) < criteria_coarsen/d);
                                  break;
                                default:
                                  throw std::invalid_argument("blah");
                                }
                            break;
                            } // end of case : divide_by
                          case MULTIPLY_BY:{
                              switch(compare_opn[2*n]){
                                case GREATER_THAN:
                                  coarsen = coarsen && (fabs(field_val) > criteria_coarsen*d);
                                  break;
                                case LESS_THAN:
                                  coarsen = coarsen && (fabs(field_val) < criteria_coarsen*d);
                                  break;
                                default:
                                  throw std::invalid_argument("blah");
                                }
                            break;
                            } // end of case: multiply_by
                          case ABSOLUTE:{
                              switch(compare_opn[2*n]){
                                case GREATER_THAN:
                                  coarsen = coarsen && (fabs(field_val) > criteria_coarsen);

                                  break;
                                case LESS_THAN:
                                  coarsen = coarsen && (fabs(field_val) < criteria_coarsen);

                                  break;
                                default:
                                  throw std::invalid_argument("blah");
                                }

                            break;
                            } // end of case: absolute
                          } // End of switch case on diagonal comparison option
                      } // End of if (coarsen)
                    if(refine_possible){
                        P4EST_ASSERT(criteria_refine < DBL_MAX); // Make sure criteria has been defined before continuing
                        switch(diag_opn[2*n + 1]){
                          case DIVIDE_BY:{
                              switch(compare_opn[2*n + 1]){
                                case GREATER_THAN:{
                                  refine = refine || (fabs(field_val) > criteria_refine/d);
                                  break;
                                  }
                                case LESS_THAN:{
                                  refine = refine || (fabs(field_val) < criteria_refine/d);
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
                                  refine = refine || (fabs(field_val) > criteria_refine*d);
                                  break;
                                  }
                                case LESS_THAN:{
                                  refine = refine || (fabs(field_val) < criteria_refine*d);
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
                                  refine = refine || (fabs(field_val) > criteria_refine);
                                  break;
                                  }
                                case LESS_THAN:{
                                  refine = refine || (fabs(field_val) < criteria_refine);
                                  break;
                                  }
                                default:{
                                  throw std::invalid_argument("blah");
                                  }
                                }
                            break;
                            } // end of case: absolute
                          } // end of switch case on diagonal option
                      } // end of if(refine_possible)
                  } // End of loop over n fields
              } // End if(coarsen OR refine_possible)
          } // End of loop over quadrant children


        } // End of refine possible OR coarsen possible

      if(refine_possible && (!all_pos && !all_neg)){ // if nodes of the quad have different signs after checking each node --> interface crosses quad
          refine = true;
        }
//end_of_function:
      // Now --> Apply the results of the check:
      if(refine){
          quad->p.user_int = REFINE_QUADRANT;
//          PetscPrintf(p4est->mpicomm,"Quadrant is marked for refining \n");
        }
      else if (coarsen){
          quad->p.user_int = COARSEN_QUADRANT;
//          PetscPrintf(p4est->mpicomm,"QUAD COARSENED \n");

        }
      else {
          quad->p.user_int = SKIP_QUADRANT;
//          PetscPrintf(p4est->mpicomm,"QUAD SKIPPED \n");

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

bool splitting_criteria_tag_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, const unsigned int num_fields, bool use_block, bool enforce_uniform_band,double refine_band,double coarsen_band,Vec *fields,Vec fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn){
  PetscErrorCode ierr;
  bool is_grid_changed;

  const double* phi_p;
  std::vector<const double*> fields_p(num_fields);
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
  is_grid_changed = refine_and_coarsen(p4est,nodes,phi_p,num_fields,use_block,enforce_uniform_band,refine_band,coarsen_band,fields_p.data(),fields_block_p,criteria,compare_opn,diag_opn);


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

bool splitting_criteria_tag_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, const double *phi_p, const unsigned int num_fields, bool use_block,bool enforce_uniform_band,double refine_band,double coarsen_band, const double** fields,const double* fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn){
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

      tag_quadrant(p4est,quad,tr,quad_idx,nodes,phi_p,num_fields,use_block,enforce_uniform_band, refine_band,coarsen_band,fields,fields_block,criteria,compare_opn,diag_opn);
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
