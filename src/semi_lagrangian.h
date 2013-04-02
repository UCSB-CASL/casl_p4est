#ifndef SEMI_LAGRANGIAN_H
#define SEMI_LAGRANGIAN_H

#include <p4est.h>
#include <petsc.h>
#include <map>
#include <sc_notify.h>
#include <src/ArrayV.h>
#include <src/utils.h>
#include <src/my_p4est_tools.h>
#include <src/petsc_compatibility.h>

#define POINT_TAG 0
#define LVLSET_TAG 1
#define SIZE_TAG 2

struct point{
  double x, y;
};

class SemiLagrangian
{
  p4est_t *p4est;
  p4est_nodes *nodes;
  ArrayV<bool> is_processed;

  p4est_locidx_t *e2n;

  PetscErrorCode ierr;

  ArrayV<ArrayV<double> > departure_point;
  ArrayV<ArrayV<p4est_locidx_t> > departing_node;

  MPI_Status st;

  // Disable copy constructor and assigment
  SemiLagrangian(const SemiLagrangian& other);
  SemiLagrangian& operator=(const SemiLagrangian& other);

  int tc;

public:
  SemiLagrangian(p4est_t *p4est_, p4est_nodes_t *nodes_);

  void update(p4est_t *p4est_, p4est_nodes_t *nodes_);
  void advect(CF_2& velx, CF_2& vely, double dt, Vec &phi);
};

#endif // SEMI_LAGRANGIAN_H
