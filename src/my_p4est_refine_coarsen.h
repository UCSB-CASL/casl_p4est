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

/*!
 * \class splitting_criteria_t
 * \brief Basic grid refinement class. Not very useful by itself, but all of the
 *        refinement classes used in practice (see below) are inherited from it.
 */
struct splitting_criteria_t {
  splitting_criteria_t(int min_lvl = 0, int max_lvl = 0, double lip = 1.2)
  {
    if(min_lvl>max_lvl)
      throw std::invalid_argument("[ERROR]: you cannot choose a min level larger than the max level.");
    this->max_lvl = max_lvl;
    this->min_lvl = min_lvl;
    this->lip     = lip;
  }

  int max_lvl, min_lvl; /*! Maximum and minimum levels of refinement.*/
  double lip;           /*! Lipschitz constant for refinement with the distance to an interface.*/
};

/*!
 * \class splitting_criteria_cf_t
 * \brief Class for refinement based on the distance to an interface. The level-set
 *        function representing the interface is provided as a continuous function.
 */
struct splitting_criteria_cf_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *phi;               /*! Pointer to continuous function object representing the level-set function.*/
#else
  CF_2 *phi;               /*! Pointer to continuous function object representing the level-set function.*/
#endif
  bool refine_only_inside; /*! If true, enforces refinement only where the l-s function is negative.*/
#ifdef P4_TO_P8
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_3 *phi, double lip=1.2)
#else
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_2 *phi, double lip=1.2)
#endif
    : splitting_criteria_t(min_lvl, max_lvl, lip), refine_only_inside(false)
  {
    this->phi = phi;
  }
  void set_refine_only_inside(bool val) { refine_only_inside = val; }
};

/*!
 * \class splitting_criteria_cf_and_uniform_band_t
 * \brief Class for refinement based on the distance to an interface, additionally
 *        enforcing a band of uniform cells around it. The level-set function
 *        representing the interface is provided as a continuous function.
 */
struct splitting_criteria_cf_and_uniform_band_t : splitting_criteria_cf_t {
  const double uniform_band; /*! Thickness of the band, expressed as the number smallest edges of the smallest cell ( i.e. thickness=uniform_band*min(dx_min,dy_min,dz_min) ).*/
#ifdef P4_TO_P8
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, CF_3 *phi_, double uniform_band_, double lip=1.2)
#else
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, CF_2 *phi_, double uniform_band_, double lip=1.2)
#endif
    : splitting_criteria_cf_t (min_lvl, max_lvl, phi_, lip), uniform_band(uniform_band_) { }
};

/*!
 * \class splitting_criteria_thresh_t
 * \brief Class for refinement based on the threshold of a function. The function
 *        of interest is provided as a continuous function.
 */
struct splitting_criteria_thresh_t : splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3 *f; /*! Pointer to continuous function object representing the function of interest.*/
#else
  CF_2 *f; /*! Pointer to continuous function object representing the function of interest.*/
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

/*!
 * \class splitting_criteria_random_t
 * \brief Class for random refinement.
 */
struct splitting_criteria_random_t : splitting_criteria_t {
  p4est_gloidx_t max_quads, min_quads, num_quads;
  splitting_criteria_random_t(int min_lvl, int max_lvl, p4est_gloidx_t min_quads, p4est_gloidx_t max_quads)
    : splitting_criteria_t(min_lvl, max_lvl)
  {
    this->min_quads = min_quads; /*! Minimum number of quadrants to be refined.*/
    this->max_quads = max_quads; /*! Maximum number of quadrants to be refined.*/
    num_quads = 0;               /*! Quadrant counter dummy variable.*/
  }
};

/*!
 * \class splitting_criteria_marker_t
 * \brief Class for refinement based on custom markers for each individual quadrant.
 */
class splitting_criteria_marker_t: public splitting_criteria_t {
  std::vector<p4est_bool_t> markers; /*! Vector of refinement markers, one per quadrant.*/
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

/*!
 * \class splitting_criteria_tag_t
 * \brief Class for refinement based on the distance to an interface. The level-set
 *        function representing the interface is provided as data sampled at grid nodes.
 */
class splitting_criteria_tag_t: public splitting_criteria_t {
protected:
	static void init_fn   (p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t*  quad);
	static int  refine_fn (p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t*  quad);
	static int  coarsen_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t** quad);

  /*!
   * \fn    tag_quadrant
   * \brief Tags an individual quadrant for refinement, coarsening or no-change depending on its
   *        distance to the interface. The version with '_inside' will only enforce this criterion
   *        in quadrants where the level-set function is negative.
   * \param p4est       [in] forest object
   * \param quad        [in] a pointer to the quadrant of interest
   * \param which_tree  [in] current tree to which the quadrant of interest belongs
   * \param f           [in] a pointer to data stored in a Vec containing the sampled level-set function on the grid
   */
  void tag_quadrant(p4est_t* p4est, p4est_quadrant_t* quad, p4est_topidx_t which_tree, const double* f);
  void tag_quadrant_inside(p4est_t* p4est, p4est_quadrant_t* quad, p4est_topidx_t which_tree, const double* f);
  bool refine_only_inside; /*! If true, enforces refinement only where the l-s function is negative.*/
public:
  splitting_criteria_tag_t(int min_lvl, int max_lvl, double lip=1.2)
    : splitting_criteria_t(min_lvl, max_lvl, lip), refine_only_inside(false)
  {
  }

  /*!
   * \fn    refine_and_coarsen
   * \brief Loops through all the quadrants in the grid, and tags them for refinement/coarsening using 'tag_quadrant' or
   *        'tag_quadrant_inside'. Then, it refines and coarsens the whole grid according to the tagging. The version
   *        without '_and_coarsen' only enforces refinement, not coarsening.
   * \param p4est       [in] forest object
   * \param nodes       [in] nodes object
   * \param phi         [in] a pointer to data stored in a Vec containing the sampled level-set function on the grid
   * \return            a boolean (0/1) set as true if at least one quadrant of the grid has been marked for refinement or coarsening
   */
  bool refine_and_coarsen(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);
  bool refine(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);

  void set_refine_only_inside(bool val) { refine_only_inside = val; }
};

/*!
 * \class splitting_criteria_grad_t
 * \brief Class for refinement based on the gradient of a function. The function
 *        of interest is provided as data sampled at grid nodes.
 */
struct splitting_criteria_grad_t: public splitting_criteria_t {
#ifdef P4_TO_P8
  CF_3* cf;
#else
  CF_2* cf;
#endif
  double fmax, tol;
#ifdef P4_TO_P8

  splitting_criteria_grad_t(int min_lvl, int max_lvl, CF_3* cf, double fmax, double tol = 1e-2)
#else
  splitting_criteria_grad_t(int min_lvl, int max_lvl, CF_2* cf, double fmax, double tol = 1e-2)
#endif
  : splitting_criteria_t(min_lvl, max_lvl), cf(cf), fmax(fmax), tol(tol)
  {}
};

/*!
 * \fn    refine_levelset_cf
 * \brief Refine based on distance to a cf level-set function.
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_levelset_cf
 * \brief Coarsen based on distance of a cf level-set function.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_levelset_cf (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    refine_levelset_cf_and_uniform_band
 * \brief Refine based on distance to a cf levelset and
 *        impose a band of uniform cells around it.
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_levelset_cf_and_uniform_band (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    refine_levelset_thres
 * \brief Refine based on the threshold of a continuous function.
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_levelset_thresh(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_levelset_thresh
 * \brief Coarsen based on the threshold of a continuous function.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_levelset_thresh(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    refine_random
 * \brief A random refinement method.
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_random
 * \brief A method to randomly coarsen a forest.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_random(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    refine_every_cell
 * \brief Refines all the cell in the p4est.
 * \param p4est       [in] forest object to consider
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_every_cell
 * \brief Coarsens all the cells in the p4est.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_every_cell(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    refine_marked_quadrants
 * \brief Refines quadrants that have been explicitly marked for refinement.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] pointer to the current quadrant
 * \return                 a boolean (0/1) describing if refinement is needed
 */
p4est_bool_t
refine_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_marked_quadrants
 * \brief Coarsens quadrants that have been explicitly marked for coarsening.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    refine_grad_cf
 * \brief Refinement based on gradient indicator.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
refine_grad_cf(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

/*!
 * \fn    coarsen_grad_cf
 * \brief Coarsening based on gradient indicator.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_grad_cf(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

/*!
 * \fn    coarsen_down_to_lmax
 * \brief A dumb coarsening down to lmax.
 * \param p4est       [in] forest object
 * \param which_tree  [in] current tree to which the quadrant belongs
 * \param quad        [in] a pointer to a list of quadrant to be coarsened
 * \return                 a boolean (0/1) describing if a set of quadrants need to be coarsened
 */
p4est_bool_t
coarsen_down_to_lmax (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

#endif // REFINE_COARSEN_H
