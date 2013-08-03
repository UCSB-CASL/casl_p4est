#include "bilinear_interpolating_function.h"

namespace parallel{
BilinearInterpolatingFunction::BilinearInterpolatingFunction(const double *F,
                                                             p4est_t *p4est, p4est_nodes_t *nodes)
  : F_(F), p4est_(p4est), nodes_(nodes)
{}

double BilinearInterpolatingFunction::operator ()(double x, double y) const {
  double xy[] = {x, y};
  return linear_interpolation(xy);
}

double BilinearInterpolatingFunction::linear_interpolation(const double xy[]) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx = 0;
  p4est_locidx_t *q2n = nodes_->local_nodes;
  p4est_quadrant_t *quad;

  int found_rank = my_p4est_brick_point_lookup_smallest(p4est_, NULL, NULL,
                                                        xy, &tree_idx, &quad_idx, &quad);
#ifdef CASL_THROWS
  if (p4est_->mpirank != found_rank){
    std::ostringstream oss; oss << "point (" << xy[0] << "," << xy[1] << ")"
                                   "does not belog to processor " << p4est_->mpirank;
    throw std::invalid_argument(oss.str());
  }
#endif

  p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tree_idx);
  quad_idx += tree->quadrants_offset;

  double f [] =
  {
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 0])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 1])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 2])],
    F_[p4est2petsc_local_numbering(nodes_, q2n[P4EST_CHILDREN*quad_idx + 3])]
  };

  return bilinear_interpolation(p4est_, tree_idx, quad, f, xy);

}

}
