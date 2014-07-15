#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_utils.h>
#include <p8est.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_utils.h>
#include <p4est.h>
#endif

#include <set>
#include <deque>

struct splitting_criteria_t {
  splitting_criteria_t(int min_lvl = 0, int max_lvl = 0, double lip = 1.2)
  {
    this->max_lvl = max_lvl;
    this->min_lvl = min_lvl;
    this->lip     = lip;
  }

  int max_lvl, min_lvl;
  double lip;
};

struct splitting_criteria_cf_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *phi;
#else
  CF_2 *phi;
#endif
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

struct splitting_criteria_threshold_cf_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *phi;
#else
  CF_2 *phi;
#endif
  double min_thr, max_thr;
#ifdef P4_TO_P8
  splitting_criteria_threshold_cf_t(int min_lvl, int max_lvl, double min_thr, double max_thr, CF_3 *phi, double lip)
#else
  splitting_criteria_threshold_cf_t(int min_lvl, int max_lvl, double min_thr, double max_thr, CF_2 *phi, double lip)
#endif
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;
    this->min_lvl = min_thr;
    this->max_thr = max_thr;
    this->phi = phi;
    this->lip = lip;
  }
};


struct splitting_criteria_random_t : splitting_criteria_t {
  p4est_gloidx_t max_quads, min_quads, num_quads;
  splitting_criteria_random_t(int min_lvl, int max_lvl, p4est_gloidx_t min_quads, p4est_gloidx_t max_quads)
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;
    this->min_quads = min_quads;
    this->max_quads = max_quads;
    num_quads = 0;
  }
};

class splitting_criteria_marker_t: public splitting_criteria_t {  
protected:
  p4est_t *p4est;
  std::deque<p4est_bool_t> markers;

public:
  splitting_criteria_marker_t(p4est_t *p4est, int min_lvl, int max_lvl, double lip)
    : p4est(p4est), markers(p4est->local_num_quadrants, P4EST_FALSE)
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;
    this->lip     = lip;
  }

  inline bool is_empty() {return markers.empty();}
  inline p4est_bool_t pop_front(){
    p4est_bool_t pop = markers.front();
    markers.pop_front();
    return pop;
  }
  inline p4est_bool_t& operator[](p4est_locidx_t q) {return markers[q];}
  inline const p4est_bool_t& operator[](p4est_locidx_t q) const {return markers[q];}
};

class splitting_criteria_discrete_t : public splitting_criteria_marker_t {
public:
  splitting_criteria_discrete_t(p4est_t *p4est, int min_lvl, int max_lvl, double lip)
    : splitting_criteria_marker_t(p4est, min_lvl, max_lvl, lip)
  {}

  void mark_cells_for_refinement(p4est_nodes_t *nodes, const double *phi);
  void mark_cells_for_coarsening(p4est_nodes_t *nodes, const double *phi);
};


/*!
 * \brief refine_levelset_cf refine based on distance to a cf levelset
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset coarsen based on distance of a cf function
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \brief refine_threshold_cf refine based on a threshold on a cf function
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_threshold_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_threshold_cf coarsen based on threshold on a cf function
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_threshold_cf(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);


/*!
 * \brief refine_random a random refinement method
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
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

/*!
 * \brief refine_every_cell refines all the cell in the p4est
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_every_cell coarsens all the cells in the p4est
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \brief refine_marked_quadrants refines quadrants that have been explicitly marked for refinement
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                 a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_marked_quadrants coarsens quadrants that have been explicitly marked for coarsening
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

#endif // REFINE_COARSEN_H
