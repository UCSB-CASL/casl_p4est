#ifndef INTERPOLATING_FUNCTION_H
#define INTERPOLATING_FUNCTION_H

#include <petsc.h>
#include <p4est.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/utils.h>

class BilinearInterpolatingFunction: public CF_2{
  Vec F;
  p4est_t *p4est;
  my_p4est_nodes_t *nodes;

public:
  BilinearInterpolatingFunction(p4est_t *p4est_, my_p4est_nodes_t *nodes_, Vec F_);
  void update(p4est_t *p4est_, my_p4est_nodes_t *nodes_, Vec F_);
  void interpolateValuesToNewForest(p4est_t *new_p4est, my_p4est_nodes_t *new_nodes, Vec *new_F);
  double operator()(double x, double y) const;
};

#endif // INTERPOLATING_FUNCTION_H
