#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#include <p4est.h>

#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>

struct cf_grid_data_t {
  CF_2 *phi;
  int max_lvl, min_lvl;
  double lip;
};

struct rand_grid_data_t {
  int max_lvl, min_lvl;
  p4est_locidx_t max_quads, min_quads;
  static p4est_locidx_t counter;
};


/*!
 * \brief refine_levelset
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
p4est_bool_t
refine_levelset (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
p4est_bool_t
coarsen_levelset (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \brief refine_random a random refinement method
 * \param p4est      [in] forest object to consider
 * \param which_tree [in] current tree to which the quadrant belongs
 * \param quad       [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_random a method to randomly coarsen a forest
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);


#endif // REFINE_COARSEN_H
