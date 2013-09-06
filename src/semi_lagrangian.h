#ifndef PARALLEL_SEMI_LAGRANGIAN_H
#define PARALLEL_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <iostream>

class SemiLagrangian
{
  p4est_t **p_p4est_, *p4est_;
  p4est_nodes_t **p_nodes_, *nodes_;
  p4est_ghost_t **p_ghost_, *ghost_;
  my_p4est_brick_t *myb_;

  double xmin, xmax, ymin, ymax;

  std::vector<double> local_xy_departure_dep, non_local_xy_departure_dep;   //Buffers to hold local and non-local departure points
  PetscErrorCode ierr;

  inline double compute_dt(const CF_2& vx, const CF_2& vy){
    double dt = DBL_MAX;

    // loop over trees
    for (p4est_topidx_t tr_it = p4est_->first_local_tree; tr_it <=p4est_->last_local_tree; ++tr_it){
      p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tr_it);
      p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
      double *v2c = p4est_->connectivity->vertices;

      double tr_xmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 0];
      double tr_ymin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 1];

      // loop over quadrants
      for (size_t qu_it=0; qu_it<tree->quadrants.elem_count; ++qu_it){
        p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

        double dx = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
        double x  = int2double_coordinate_transform(quad->x) + 0.5*dx + tr_xmin;
        double y  = int2double_coordinate_transform(quad->y) + 0.5*dx + tr_ymin;
        double vn = sqrt(SQR(vx(x,y)) + SQR(vy(x,y)));
        dt = MIN(dt, dx/vn);
      }
    }

    double dt_min;
    MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm);

    return dt_min;
  }

  double linear_interpolation(const double *F, const double xy[], p4est_topidx_t tree_idx = 0);
  void update_p4est(Vec& phi);

public:
  SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb);

  double advect(const CF_2& vx, const CF_2& vy, Vec &phi);
};

#endif // PARALLEL_SEMI_LAGRANGIAN_H
