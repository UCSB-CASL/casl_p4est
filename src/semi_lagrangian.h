#ifndef SEMI_LAGRANGIAN_H
#define SEMI_LAGRANGIAN_H

#include <p4est.h>
#include <src/ArrayV.h>
#include <petsc.h>
#include <map>
#include <sc_notify.h>
#include <src/utils.h>
#include <src/my_p4est_tools.h>

#define POINT_TAG 0
#define LVLSET_TAG 1
#define SIZE_TAG 2

struct point{
  double x, y;
};

class semi_lagrangian
{
  p4est_t *p4est;
  my_p4est_nodes *nodes;
  ArrayV<bool> is_processed;

  p4est_locidx_t *e2n;

  PetscErrorCode ierr;

  ArrayV<ArrayV<double> > departure_point;
  ArrayV<ArrayV<p4est_locidx_t> > departing_node;

  MPI_Status st;

  // Disable copy constructor and assigment
  semi_lagrangian(const semi_lagrangian& other);
  semi_lagrangian& operator=(const semi_lagrangian& other);

public:
  semi_lagrangian(p4est_t *p4est_, my_p4est_nodes_t *nodes_);

  void advect(Vec velx, Vec vely, double dt, Vec &phi);
};

#endif // SEMI_LAGRANGIAN_H
