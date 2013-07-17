#ifndef BILINEAR_INTERPOLATING_FUNCTION_H
#define BILINEAR_INTERPOLATING_FUNCTION_H

#include <vector>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>

namespace serial{
class BilinearInterpolatingFunction: public CF_2
{
  const std::vector<double>& F_;
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;

  inline double linear_interpolation(const std::vector<double>& F, const double xy[]) const{
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx = 0;
    p4est_locidx_t *q2n = nodes_->local_nodes;
    p4est_quadrant_t *quad;

    my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                         xy, &tree_idx, &quad_idx, &quad);

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

public:
  BilinearInterpolatingFunction(const std::vector<double>& F, p4est_t *p4est, p4est_nodes_t *nodes);

  double operator()(double x, double y) const;
};
} // namepace serial
#endif // BILINEAR_INTERPOLATING_FUNCTION_H
