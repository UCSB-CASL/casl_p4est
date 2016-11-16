#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#include <p8est.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_log_wrappers.h>
#include <p4est.h>
#endif

#include <set>
#include <vector>
#include <stdexcept>

#define SKIP_QUADRANT		 0
#define REFINE_QUADRANT  1
#define COARSEN_QUADRANT 2
#define NEW_QUADRANT     3

// p4est boolean type
typedef int p4est_bool_t;
#define P4EST_TRUE  1
#define P4EST_FALSE 0

// forward declaration
class CF_3;
class CF_2;

struct splitting_criteria_t {
  splitting_criteria_t(int min_lvl = 0, int max_lvl = 0, double lip = 1.2)
  {
    if(min_lvl>max_lvl)
      throw std::invalid_argument("[ERROR]: you cannot choose a min level larger than the max level.");
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
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_3 *phi, double lip=1.2)
#else
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_2 *phi, double lip=1.2)
#endif
    : splitting_criteria_t(min_lvl, max_lvl, lip)
  {
    this->phi = phi;
  }
};

struct splitting_criteria_thresh_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *f;
#else
  CF_2 *f;
#endif
  double thresh;
#ifdef P4_TO_P8
  splitting_criteria_thresh_t(int min_lvl, int max_lvl, CF_3 *f, double thresh)
#else
  splitting_criteria_thresh_t(int min_lvl, int max_lvl, CF_2 *f, double thresh)
#endif
    :splitting_criteria_t(min_lvl, max_lvl)
  {
    this->f = f;
    this->thresh = thresh;
  }
};

struct splitting_criteria_random_t : splitting_criteria_t {
  p4est_gloidx_t max_quads, min_quads, num_quads;
  splitting_criteria_random_t(int min_lvl, int max_lvl, p4est_gloidx_t min_quads, p4est_gloidx_t max_quads)
    : splitting_criteria_t(min_lvl, max_lvl)
  {
    this->min_quads = min_quads;
    this->max_quads = max_quads;
    num_quads = 0;
  }
};

class splitting_criteria_marker_t: public splitting_criteria_t {
  std::vector<p4est_bool_t> markers;
public:
  splitting_criteria_marker_t(p4est_t *p4est, int min_lvl, int max_lvl, double lip=1.2)
    : splitting_criteria_t(min_lvl, max_lvl, lip), markers(p4est->local_num_quadrants, P4EST_FALSE)
  {
    // Associate each marker with a quadrant
    for (p4est_topidx_t tr = p4est->first_local_tree; tr <= p4est->last_local_tree; tr++){
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      for (size_t qu = 0; qu < tree->quadrants.elem_count; qu++){
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
        p4est_locidx_t q = qu + tree->quadrants_offset;

        quad->p.user_data = &markers[q];
      }
    }
  }

  inline p4est_bool_t& operator[](p4est_locidx_t q) {return markers[q];}
  inline const p4est_bool_t& operator[](p4est_locidx_t q) const {return markers[q];}
};

class splitting_criteria_tag_t: public splitting_criteria_t {
protected:
	static void init_fn   (p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t*  quad);
	static int  refine_fn (p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t*  quad);
	static int  coarsen_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t** quad);
	
  void tag_quadrant(p4est_t* p4est, p4est_quadrant_t* quad, p4est_topidx_t which_tree, const double* f);
  void tag_quadrant_inside(p4est_t* p4est, p4est_quadrant_t* quad, p4est_topidx_t which_tree, const double* f);
  bool refine_only_inside;
public:
  splitting_criteria_tag_t(int min_lvl, int max_lvl, double lip=1.2)
    : splitting_criteria_t(min_lvl, max_lvl, lip), refine_only_inside(false)
  {
  }

  bool refine_and_coarsen(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);
  bool refine(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);

  void set_refine_only_inside(bool val) { refine_only_inside = true; }
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
 * \brief refine_levelset_cf refine based on the threshold of a continuous function
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_levelset_thresh(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset coarsen based on the threshold of a continuous function
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_levelset_thresh(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

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

/*!
 * \brief refine_levelset_cf refine based on distance to a cf levelset
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_inside_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \brief coarsen_levelset coarsen based on distance of a cf function
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_inside_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

#endif // REFINE_COARSEN_H
