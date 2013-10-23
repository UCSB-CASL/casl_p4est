#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_utils.h>
#endif

struct splitting_criteria_t {
  int max_lvl, min_lvl;
};

struct splitting_criteria_cf_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *phi;
#else
  CF_2 *phi;
#endif
  double lip;
#ifdef P4_TO_P8
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_3 *phi, double lip)
#else
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_2 *phi, double lip)
#endif
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;
    this->phi = phi;
    this->lip = lip;
  }
};

struct splitting_criteria_random_t : splitting_criteria_t {
  p4est_locidx_t max_quads, min_quads;
  static p4est_locidx_t counter;
  splitting_criteria_random_t(int min_lvl, int max_lvl, p4est_locidx_t min_quads, p4est_locidx_t max_quads)
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;
    this->min_quads = min_quads;
    this->max_quads = max_quads;
  }
};

/*!
 * \brief refine_levelset
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
p4est_bool_t
refine_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
p4est_bool_t
coarsen_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \brief refine_levelset_discrete
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
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
