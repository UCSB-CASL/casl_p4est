#ifndef SEMI_LAGRANGIAN_H
#define SEMI_LAGRANGIAN_H

#include <p4est.h>
#include <src/ArrayV.h>
#include <petsc.h>
#include <map>
#include <sc_notify.h>
#include <src/utils.h>

#define POINT_TAG 0
#define LVLSET_TAG 1
#define SIZE_TAG 2

typedef int (*p4est_point_process_lookup_t)(double, double) ;
typedef p4est_quadrant_t* (*p4est_point_quadrant_lookup_t)(double, double, p4est_topidx_t*);

struct point{
  double x, y;
};

class semi_lagrangian
{
  const p4est_t *p4est;
  ArrayV<bool> is_processed;
  p4est_locidx_t local_num_nodes;
  p4est_locidx_t ghost_num_nodes;

  p4est_gloidx_t *e2n;

  p4est_point_process_lookup_t process_lookup;
  p4est_point_quadrant_lookup_t quadrant_lookup;
  PetscErrorCode ierr;

  ArrayV<ArrayV<double> > departure_point;
  ArrayV<ArrayV<p4est_locidx_t> > departing_node;

public:
  semi_lagrangian(const p4est_t *p4est_, p4est_point_process_lookup_t process_lookup_, p4est_point_quadrant_lookup_t quadrant_lookup_);

  void advance(Vec velx, Vec vely, double dt, Vec phi);
};

#endif // SEMI_LAGRANGIAN_H
