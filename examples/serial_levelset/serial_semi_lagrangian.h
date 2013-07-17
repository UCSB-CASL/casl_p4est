#ifndef SERIAL_SEMI_LAGRANGIAN_H
#define SERIAL_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>

namespace serial{
class SemiLagrangian
{
  p4est_t **p_p4est_, *p4est_;
  p4est_nodes_t **p_nodes_, *nodes_;

  double xmin, xmax, ymin, ymax;

  inline double compute_dt(const CF_2& vx, const CF_2& vy){
    double dt = 1000;

    // loop over trees
    for (p4est_topidx_t tr_it = p4est_->first_local_tree; tr_it <=p4est_->last_local_tree; ++tr_it){
      p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tr_it);
      p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
      double *v2c = p4est_->connectivity->vertices;

      double tr_xmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 0];
      double tr_ymin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 1];

      // loop over quadrants
      for (p4est_locidx_t qu_it=0; qu_it<tree->quadrants.elem_count; ++qu_it){
        p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

        double dx = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
        double x  = int2double_coordinate_transform(quad->x) + 0.5*dx + tr_xmin;
        double y  = int2double_coordinate_transform(quad->y) + 0.5*dx + tr_ymin;
        double vn = SQRT(SQR(vx(x,y)) + SQR(vy(x,y)));
        dt = MIN(dt, dx/vn);
      }
    }

    return dt;
  }

  inline double linear_interpolation(const std::vector<double>& F, const double xy[], p4est_topidx_t tree_idx = 0){
    p4est_locidx_t quad_idx;
    p4est_locidx_t *q2n = nodes_->local_nodes;
    p4est_quadrant_t *quad;

    my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                         xy,
                                         &tree_idx, &quad_idx, &quad);
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
    quad_idx += tree->quadrants_offset;
    double f [] =
    {
      F[q2n[P4EST_CHILDREN*quad_idx + 0]],
      F[q2n[P4EST_CHILDREN*quad_idx + 1]],
      F[q2n[P4EST_CHILDREN*quad_idx + 2]],
      F[q2n[P4EST_CHILDREN*quad_idx + 3]]
    };

    return bilinear_interpolation(p4est_, tree_idx, quad, f, xy[0], xy[1]);
  }

  void update_p4est(std::vector<double>& phi);

public:
  SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes);

  double advect(const CF_2& vx, const CF_2& vy, std::vector<double> &phi);
};
} // namespace serial
#endif // SERIAL_SEMI_LAGRANGIAN_H
