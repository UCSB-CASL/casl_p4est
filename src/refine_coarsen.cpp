#include "refine_coarsen.h"
#include <sc_search.h>
#include <p4est_bits.h>

int
refine_levelset (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  refine_coarsen_data_t *data = (refine_coarsen_data_t*)p4est->user_pointer;

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

    for (unsigned short cj = 0; cj<2; ++cj)
      for (unsigned short ci = 0; ci <2; ++ci)
        if (fabs(phi(x+ci*dx, y+cj*dy)) <= 0.5*lip*d)
          return P4EST_TRUE;

    return P4EST_FALSE;
  }
}

int
coarsen_levelset (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
{
  refine_coarsen_data_t *data = (refine_coarsen_data_t*)p4est->user_pointer;

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

    for (unsigned short cj = 0; cj<2; ++cj)
      for (unsigned short ci = 0; ci <2; ++ci)
          if(fabs(phi(x+ci*dx, y+cj*dy)) <= 0.5*lip*d)
            return P4EST_FALSE;

    return P4EST_TRUE;
  }
}
