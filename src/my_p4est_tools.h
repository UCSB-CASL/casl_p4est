
#ifndef MY_P4EST_TOOLS_H
#define MY_P4EST_TOOLS_H

#include <p4est.h>
#include <p4est_ghost.h>

#ifdef __cplusplus
extern "C" {
#if 0
}
#endif
#endif

/** Find the owner processor for a point in a brick domain.
 * For remote points, only the owner's rank is returned.
 * \param [in] p4est    The forest to be searched.
 * \param [in] xy       The x and y coordinates of a point in the brick.
 * \param [in,out] which_tree   On input, a guess for the tree.
 *                      For a local point, on output its tree id.
 * \param [out] which_quad      For a local point, the quadrant index
 *                      relative to its tree if !NULL.
 * \param [out] quad    For a local point, the containing quadrant if !NULL.
 * \return              The processor number that owns the point xy.
 */
int my_p4est_brick_point_lookup (p4est_t * p4est, const double * xy,
                                 p4est_topidx_t *which_tree,
                                 p4est_locidx_t *which_quad,
                                 p4est_quadrant_t **quad);

/*!
 * \brief my_p4est_brick_point_lookup_smallest same as above except returns the smallest possible cell
 * \param p4est
 * \param xy
 * \param which_tree
 * \param which_quad
 * \param quad
 * \return
 */
int my_p4est_brick_point_lookup_smallest (p4est_t * p4est, const double * xy,
                                          p4est_topidx_t *which_tree,
                                          p4est_locidx_t *which_quad,
                                          p4est_quadrant_t **quad);

#ifdef __cplusplus
#if 0
{
#endif
}
#endif

#endif /* !MY_P4EST_TOOLS_H */
