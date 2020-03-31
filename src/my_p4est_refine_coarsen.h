#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#include <p8est.h>
#else
//#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_log_wrappers.h>
#include <p4est.h>
#endif

#include <set>
#include <vector>
#include <stdexcept>

#define SKIP_QUADRANT	 0
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

// Define options for one of the refinement tools:
// Elyce trying something: // for use in refine and coarsen
enum compare_option_t {LESS_THAN = 0, GREATER_THAN = 1, SIGN_CHANGE =2, NO_CHECK = 3};
// sign change indicates a search for change in sign of a field, no check indicates you don't want to check that particular field for either refining or coarsening
enum compare_diagonal_option_t {DIVIDE_BY=0, MULTIPLY_BY = 1, ABSOLUTE = 2};

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
  bool refine_only_inside;
#ifdef P4_TO_P8
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_3 *phi, double lip=1.2)
#else
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_2 *phi, double lip=1.2)
#endif
    : splitting_criteria_t(min_lvl, max_lvl, lip)
  {
    this->phi = phi;
  }
  void set_refine_only_inside(bool val) { refine_only_inside = val; }
};

struct splitting_criteria_cf_and_uniform_band_t : splitting_criteria_cf_t {
  const double uniform_band;
#ifdef P4_TO_P8
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, CF_3 *phi_, double uniform_band_, double lip=1.2)
#else
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, CF_2 *phi_, double uniform_band_, double lip=1.2)
#endif
    : splitting_criteria_cf_t (min_lvl, max_lvl, phi_, lip), uniform_band(uniform_band_) { }
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
  // ELYCE TRYING SOMETHING:
  /*!
   * \brief tag_quadrant
   * This function tags a quadrant for either refinement or coarsening, depending on a variable number of fields and criteria which are provided by the user.
   * Please see the documentation on refine_and_coarsen with corresponding inputs for more information on the exact usage of the fields and criteria.
   * \param p4est [inout] the grid you want to refine and coarsen
   * \param quad  [in] The current quadrant we are evaluating for refinement or coarsening
   * \param tree_idx [in] Tree index of current tree we are considering
   * \param quad_idx [in] Quad index of current quadrant we are considering
   * \param nodes [in] nodes of the grid we want to refine and coarsen
   * \param phi_p [in] double pointer to the level set function we are considering
   * \param num_fields [in] *please see refine_and_coarsen documentation
   * \param use_block [in] *please see refine_and_coarsen documentation
   * \param enforce_uniform_band [in] *please see refine_and_coarsen documentation
   * \param refine_band [in] *please see refine_and_coarsen documentation
   * \param coarsen_band [in] *please see refine_and_coarsen documentation
   * \param fields [in] *please see refine_and_coarsen documentation
   * \param fields_block [in] *please see refine_and_coarsen documentation
   * \param criteria [in] *please see refine_and_coarsen documentation
   * \param compare_opn [in] *please see refine_and_coarsen documentation
   * \param diag_opn [in] *please see refine_and_coarsen documentation
   * Developer: Elyce Bayat, ebayat@ucsb.edu
   * Last modified: 3/30/2020
   * WARNING: This function has not yet been fully validated in 2d
   * WARNING: This function has not been implemented or validated in 3d
   */


  void tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t tree_idx, p4est_locidx_t quad_idx,p4est_nodes_t *nodes, const double* phi_p, const int num_fields,bool use_block,bool enforce_uniform_band,double refine_band,double coarsen_band, const double** fields,const double* fields_block,std::vector<double> criteria, std::vector<compare_option_t> compare_opn,std::vector<compare_diagonal_option_t> diag_opn);

  bool refine_only_inside;
public:
  splitting_criteria_tag_t(int min_lvl, int max_lvl, double lip=1.2)
    : splitting_criteria_t(min_lvl, max_lvl, lip), refine_only_inside(false)
  {
  }

  bool refine_and_coarsen(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);
  // ELYCE TRYING SOMETHING:

  bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, const int num_fields, bool use_block,bool enforce_uniform_band,double refine_band, double coarsen_band, Vec* fields,Vec fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn);
  /*!
   * \brief refine_and_coarsen
   * \param p4est           [inout] the grid you want to refine and coarsen
   * \param nodes           [inout] nodes of the grid you want to refine and coarsen
   * \param phi_p           [inout] a PETSC Vector pointer (object Vec) to the LSF (or effective LSF) that we want to refine and coarsen around
   * \param num_fields      [in] int, number of fields to refine and coarsen by
   * \param use_block       [in] boolean, describing whether to use a PETSc block vector to access fields to refine by, or not.
   *                                True = use user provided double pointer to PETSc block vector with block size = num_fields.
   *                                False = use std::vector of double pointers for num_fields number of PETSc Vectors
   *
   * \param enforce_uniform_band [in] boolean, describing whether or not a uniform band will be enforced around the interface
   * \param refine_band     [in] double, Size of refined band around the interface we want to enforce. ie. refine_band = 2.0 --> band of 2 smallest grid cells will be enforced around the interface
   * \param coarsen_band    [in] double, Size of coarsened band around the interface that we allow coarsening for. ie. coarsen_band = 4.0 --> coarsening is not allowed around the interface for less than 4 grid cells from interface, but greater than 4 coarsening is allowed
   * \param fields          [in] an array of PETSC vector pointer (object Vec) which point to the fields we want to refine by
   *
   * \param fields_block    [in] a PETSC Vector pointer (object Vec) to a PETSc block vector of fields to refine by
   * \param criteria        [in] a std::vector of criteria to coarsen and refine by
   *                            - The order of the criteria list should be as follows:
   *                                criteria = {criteria_coarsen_field_1, criteria_refine_field_1, ....., criteria_coarsen_field_n,
   *                                            criteria_refine_field_n}, for n = 1, ..., num_fields
   *                            - Therefore, the total length of criteria should be 2*n (one coarsen condition and one refine condition for each field we are considering)
   *
   * \param compare_opn     [in] a std::vector of comparison options to refine and coarsen by, with same ordering as criteria above (see below for more information)
   * \param diag_opn        [in] a std::vector of diagonal comparison options to refine and coarsen by, with same ordering as criteria above (see below for more information)
   *
   * MORE INFORMATION FROM THE DEVELOPER:
   * Refinement for LSF:
   *     - LSF is evaluated for refinement and/or coarsening by
   *        (a) comparing LSF*(lipschitz coeff) to 0
   *        (b) checking for sign changes in LSF across a grid cell
   *
   * Looking for more refined neighbors:
   *    - The refinement procedure considers not only the current quadrant, but also searches for the existence of neighbors of the current cell which have higher levels of refinement.
   *      If they exist, we use the data from these neighbors as well in our evaluation.
   *
   *
   * Info about using compare_opn : there are 3 comparison options currently: LESS_THAN, GREATER_THAN, and SIGN_CHANGE
   *                        - GREATER_THAN evaluates case where your field is greater than the specified criteria
   *                        - LESS_THAN    is true for case where your field is less than the specified criteria
   *                        - SIGN_CHANGE evaluates cases for the change in sign of a provided field (so long as values are above the specified criteria)
   *                                --> NOTE: sign change refines in the case where there is a sign change across the cell AND the cell has a more refined neighbor node on one or more sides AND the (abs(field value)>provided criteria for all nodes in cell)
   *                                        sign change coarsens in the case where there is EITHER (NO sign change across cell) OR (abs(field value)< provided criteria for all nodes in cell)
   *
   * Info about using diag_opn : there are 3 diagonal comparsion options currently: ABSOLUTE, DIVIDE_BY, and MULTIPLY_BY
   *                        - ABSOLUTE: compares the field value to the criteria provided
   *                        - DIVIDE_BY: compares the field value to (criteria provided)/(cell size)
   *                        - MULTIPLY_BY: compares the field value to (criteria provided)*(cell size)
   * The various selections for compare_opn and diag_opn are combined to provide 9 total possible ways to compare a field for refinement or coarsening
   *
   * Developer: Elyce Bayat, ebayat@ucsb.edu
   * Last modified: 3/30/2020
   * WARNING: This function has not yet been fully validated in 2d
   * WARNING: This function has not been implemented or validated in 3d
   * \return
   */
  bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, const double *phi_p,
                          const int num_fields, bool use_block,bool enforce_uniform_band,double refine_band, double coarsen_band, const double** fields, const double* fields_block, std::vector<double> criteria,
                          std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn);

  bool refine(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi);

  void set_refine_only_inside(bool val) { refine_only_inside = val; }
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

p4est_bool_t
refine_levelset_cf_and_uniform_band (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

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

p4est_bool_t
coarsen_down_to_lmax (p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

#endif // REFINE_COARSEN_H
