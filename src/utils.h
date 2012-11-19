#ifndef UTILS_H
#define UTILS_H

#include <p4est.h>
#include <stdexcept>

void c2p_coordinate_transform(p4est_t *p4est, p4est_topidx_t tree_id, double *x, double *y, double *z);

void dx_dy_dz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *dx, double *dy, double *dz);

void xyz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *x, double *y, double *z);

#endif // UTILS_H
