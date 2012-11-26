#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#include <p4est.h>

#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>


typedef struct {
  p4est_t *p4est;
  my_p4est_nodes_t *nodes;
  double *phi;
  int max_lvl, min_lvl;
  double lip;
} grid_discrete_data_t;

typedef struct {
  CF_2 *phi;
  int max_lvl, min_lvl;
  double lip;
} grid_continous_data_t;

/*!
 * \brief refine_levelset_continous
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
int
refine_levelset_continous (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset_continous
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
int
coarsen_levelset_continous (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \brief refine_levelset_discrete
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
int
refine_levelset_discrete(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset_discrete
 * \param p4est
 * \param which_tree
 * \param quad
 * \return
 */
int
coarsen_levelset_discrete(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);


#endif // REFINE_COARSEN_H
