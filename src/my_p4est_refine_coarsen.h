#ifndef REFINE_COARSEN_H
#define REFINE_COARSEN_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_macros.h>
#include <p8est.h>
#else
//#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_macros.h>
#include <p4est.h>
#endif

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


/*!
 * \class splitting_criteria_t
 * \brief Basic grid refinement class. Not very useful by itself, but all of the
 *        refinement classes used in practice (see below) are inherited from it.
 */
struct splitting_criteria_t {
  splitting_criteria_t(int min_lvl = 0, int max_lvl = 0, double lip = 1.2, double band = 0)
  {
    if(min_lvl>max_lvl)
      throw std::invalid_argument("[ERROR]: you cannot choose a min level larger than the max level.");
    this->max_lvl            = max_lvl;
    this->min_lvl            = min_lvl;
    this->lip                = lip;
    this->band               = band;
    this->refine_only_inside = false;
  }

  void set_refine_only_inside(bool val) { refine_only_inside = val; }

  int    max_lvl, min_lvl;   /*! Maximum and minimum levels of refinement.*/
  double lip;                /*! Lipschitz constant for refinement with the distance to an interface.*/
  double band;               /*! Uniform band around an interface.*/
  bool   refine_only_inside; /*! If true, enforces refinement only where the l-s function is negative.*/
};

/*!
 * \class splitting_criteria_cf_t
 * \brief Class for refinement based on the distance to an interface. The level-set
 *        function representing the interface is provided as a continuous function.
 */
struct splitting_criteria_cf_t : splitting_criteria_t {
  CF_DIM *phi;             /*! Pointer to continuous function object representing the level-set function.*/
  splitting_criteria_cf_t(int min_lvl, int max_lvl, CF_DIM *phi, double lip=1.2, double band = 0)
    : splitting_criteria_t(min_lvl, max_lvl, lip, band)
  {
    this->phi = phi;
  }
  splitting_criteria_cf_t(int min_lvl, int max_lvl, const CF_DIM *phi, double lip=1.2, double band = 0):splitting_criteria_t(min_lvl, max_lvl, lip, band)
  {
      this->phi =const_cast<CF_DIM*>(phi);
      // this one did not work: Elyce and Rochi merge ~12/3/21 //this->phi = std::remove_const<typename std::remove_pointer<const CF_DIM*>::type>::type* CF_DIM(phi);
  }
};

/*!
 * \class splitting_criteria_cf_and_uniform_band_t
 * \brief Class for refinement based on the distance to an interface, additionally
 *        enforcing a band of uniform cells around it. The level-set function
 *        representing the interface is provided as a continuous function.
 */
struct splitting_criteria_cf_and_uniform_band_t : splitting_criteria_cf_t {
  const double uniform_band;
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, CF_DIM *phi_, double uniform_band_, double lip=1.2)
    : splitting_criteria_cf_t (min_lvl, max_lvl, phi_, lip), uniform_band(uniform_band_) { }
  splitting_criteria_cf_and_uniform_band_t(int min_lvl, int max_lvl, const CF_DIM *phi_, double uniform_band_, double lip=1.2)
    : splitting_criteria_cf_t (min_lvl, max_lvl, phi_, lip), uniform_band(uniform_band_) { }
};

/*!
 * \class splitting_criteria_thresh_t
 * \brief Class for refinement based on the threshold of a function. The function
 *        of interest is provided as a continuous function.
 */
struct splitting_criteria_thresh_t : splitting_criteria_t {
  const CF_DIM *f;
  double thresh;
  splitting_criteria_thresh_t(int min_lvl, int max_lvl, const CF_DIM *f, double thresh)
    : splitting_criteria_t(min_lvl, max_lvl)
  {
    this->f = f;
    this->thresh = thresh;
  }
  virtual ~splitting_criteria_thresh_t() {};
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
  virtual ~splitting_criteria_random_t() {};
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
  inline p4est_bool_t& operator[](p4est_locidx_t q) { return markers[q]; }
  inline const p4est_bool_t& operator[](p4est_locidx_t q) const { return markers[q]; }
  virtual ~splitting_criteria_marker_t() {};
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

  void tag_quadrant(p4est_t* p4est, p4est_quadrant_t* quad, p4est_topidx_t which_tree, const double* f, bool finest_in_negative_flag);
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
   * WARNING: This function has not been implemented or validated in 3d
   */


  void tag_quadrant(p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t tree_idx, p4est_locidx_t quad_idx,p4est_nodes_t *nodes, const double* phi_p, const int num_fields,bool use_block,bool enforce_uniform_band,double refine_band,double coarsen_band, const double** fields,const double* fields_block,std::vector<double> criteria, std::vector<compare_option_t> compare_opn,std::vector<compare_diagonal_option_t> diag_opn,std::vector<int> lmax_custom);

  double uniform_band;
  bool refine_only_inside;
public:
  splitting_criteria_tag_t(int min_lvl, int max_lvl, double lip=1.2, double uniform_band_ = -1.0)
    : splitting_criteria_t(min_lvl, max_lvl, lip), uniform_band(uniform_band_), refine_only_inside(false)
  {
  }
  splitting_criteria_tag_t(const splitting_criteria_t* splitting_criteria_, double uniform_band_ = -1.0)
    : splitting_criteria_t(*splitting_criteria_), uniform_band(uniform_band_), refine_only_inside(false)
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
  bool refine_and_coarsen(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi, bool finest_in_negative_flag = false);
  bool refine(p4est_t* p4est, const p4est_nodes_t* nodes, const double* phi, bool finest_in_negative_flag = false);
  inline bool refine(p4est_t* p4est, const p4est_nodes_t* nodes, Vec phi, bool finest_in_negative_flag = false)
  {
    const double *phi_p;
    PetscErrorCode ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    const bool to_return = refine(p4est, nodes, phi_p, finest_in_negative_flag);
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    return to_return;
  }

  // ELYCE TRYING SOMETHING:
  /*!
   * \brief refine_and_coarsen
   * This function evaluates whether quadrants in the grid are eligible for refinement/coarsening based on a variable number of fields with user-selected refinement and coarsen criteria for each field. This function returns a boolean which is true if the grid has changed and false if the grid has not changed.
   * Note: This particular function is a wrapper for the refine_coarsen function below whose arguments are identical except for the fields to refine/coarsen around, which are constant double pointers instead of provided PETSC Vec.
   * This particular function:
   *      (a) Gets the associated double pointer with the given PETSC Vec field being considered for refinement/coarsening criteria
   *      (b) Passes that field and criteria to the refine_and_coarsen function listed below, which will then check the refine/coarsen condition for all quadrants in the forest.
   * \param p4est           [inout] the grid you want to refine and coarsen
   * \param nodes           [in] nodes of the grid you want to refine and coarsen
   * \param phi             [in] a PETSC Vec for the LSF (or effective LSF) that we want to refine and coarsen around
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
  bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi,
                          const unsigned int num_fields, bool use_block,bool enforce_uniform_band,
                          double refine_band, double coarsen_band,
                          Vec* fields,Vec fields_block,
                          std::vector<double> criteria, std::vector<compare_option_t> compare_opn,
                          std::vector<compare_diagonal_option_t> diag_opn,std::vector<int> lmax_custom);

  /*!
   * \brief refine_and_coarsen
   * This function loops through each quadrant in the grid and evaluates whether the quadrant is eligible for refinement or coarsening depending on the provided user criteria. It then returns a boolean which is true if the grid has changed and false if the grid has not changed.
   * The complete information from the developer is described in the function listed above.
   * The refine_and_coarsen function listed above wraps around this function, and serves to get PETSC Vec fields as pointers to then pass to this function. This function then:
   *      (a) Loops through each quadrant in the grid
   *      (b) Evaluates whether or not refinement/coarsening can be applied to that grid, and calls the function to tag the quadrants appropriately
   *      (c) Returns whether or not the grid has been changed
   * \param p4est [inout] See above description
   * \param nodes [in] " "
   * \param phi_p [in] a double pointer for the LSF (or effective LSF) that we want to refine and coarsen around
   * \param num_fields [in] See above description
   * \param use_block [in] " "
   * \param enforce_uniform_band [in] " "
   * \param refine_band [in] " "
   * \param coarsen_band [in] " "
   * \param fields [in] a double pointer to the fields for which we want to evaluate refinement/coarsening criteria
   * \param fields_block [in] a double pointer to the fields (provided in block vector format) for which we want to evaluate refinement/coarsening criteria
   * \param criteria [in] See above function description
   * \param compare_opn [in] See above function description
   * \param diag_opn [in] See above function description
   * \param lmax_custom [in] See above function description
   * \return
   */
  bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, const double *phi_p,
                          const unsigned int num_fields, bool use_block,
                          bool enforce_uniform_band,
                          double refine_band, double coarsen_band,
                          const double** fields, const double* fields_block,
                          std::vector<double> criteria, std::vector<compare_option_t> compare_opn,
                          std::vector<compare_diagonal_option_t> diag_opn,std::vector<int> lmax_custom);

  void set_refine_only_inside(bool val) { refine_only_inside = val; }

  virtual ~ splitting_criteria_tag_t() {};
};

/*!
 * \class splitting_criteria_grad_t
 * \brief Class for refinement based on the gradient of a function. The function
 *        of interest is provided as data sampled at grid nodes.
 */
struct splitting_criteria_grad_t: public splitting_criteria_t {
  const CF_DIM* cf;
  double fmax, tol;

  splitting_criteria_grad_t(int min_lvl, int max_lvl, const CF_DIM* cf, double fmax, double tol = 1e-2)
  : splitting_criteria_t(min_lvl, max_lvl), cf(cf), fmax(fmax), tol(tol)
  {}
  virtual ~splitting_criteria_grad_t() {};
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

/**
 * Refine the grid based on distances to the walls on the y-axis.
 * This function should be used when special refinement is disabled.
 * @note This is an especialization of the refine_levelset_cf_and_uniform_band function.
 * @param [in] p4est Forest object.
 * @param [in] which_tree Current tree to which the quadrant belongs.
 * @param [in] quad Current quadrant.
 * @return 0/1 describing if refinement is needed.
 */
p4est_bool_t refine_levelset_cf_and_uniform_band_shs( p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad );

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Splitting-criteria class based on usual distance to interface and explicit narrow band around interface.
 * Code based on the 'grid_update' example.
 */
class splitting_criteria_band_t : public splitting_criteria_tag_t
{
private:
	double _bandWidth;		// Band width around the interface (measured in min cell diagonal).

	/**
	 * Tag quadrants for coarsening or refinement depending on whether their phi value meets the usual splitting crite-
	 * rion or if they are within some band (in min diagonals) from the interface.
	 * @param [in] p4est Pointer to p4est object.
	 * @param [in] quadIdx Quadrant index.
	 * @param [in] treeIdx Tree index which the quadrant belongs to.
	 * @param [in] nodes Pointer to nodes object.
	 * @param [in] phiReadPtr Pointer to level-set function values vector.
	 */
	void tag_quadrant( p4est_t const *p4est, p4est_locidx_t quadIdx, p4est_topidx_t treeIdx, p4est_nodes_t const *nodes,
					   double const *phiReadPtr );
public:
	splitting_criteria_band_t( int minLvl, int maxLvl, double lip, double bandWidth=0 )
	: splitting_criteria_tag_t( minLvl, maxLvl, lip ), _bandWidth( bandWidth ) {}

	/**
	 * Refine or coarsen a grid after tagging the quadrants appropriately.
	 * @note I had to rename this function from refine_and_coarsen to refine_and_coarsen because I cannot declare the
	 * parent's class function virtual --it has parameters with default values, which are forbidden.
	 * @param [in,out] p4est Pointer to p4est object.
	 * @param [in] nodes Pointer to nodes object.
	 * @param [in] phiReadPtr Pointer to level-set function values vector.
	 * @return True if grid changed, false otherwise.
	 */
	bool refine_and_coarsen_with_band( p4est_t *p4est, p4est_nodes_t const *nodes, double const *phiReadPtr );
};

/**
 * Class for refining an superhydrophic-surfaced channel where the solid ridges and air interface lie at y=+-DELTA, and
 * DELTA is half the channel height.  The goal of this refining criteria is to use uniform bands for the distinct levels
 * of refinements based on cell distances to the solid ridges.  Since the traditional approach doesn't enforce uniform
 * refinement along the air interfaces, here we add conditions (independent of the Lipschitz condition) so that the
 * Voronio tesellation doesn't fail in later computations.
 * @note This class is to be used with the refine_levelset_cf_and_uniform_band_shs() method only.
 */
class splitting_criteria_cf_and_uniform_band_shs_t : public splitting_criteria_cf_and_uniform_band_t
{
private:
	// These functions are identical to those from splitting_criteria_tag_t.  I needed to copy them because extending the
	// splitting_criteria_tag_t class didn't cast to splitting_criteria_t correctly when reading the max level of refinement in
	// my_p4est_navier_stokes_t::get_lmax().
	static void init_fn( p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quad );
	static int  refine_fn (p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t*  quad);
	static int  coarsen_fn(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t** quad);

	/**
	 * Tag a quadrant for coarsening, refinement, or leave as is.
	 * @param [in] p4est P4est structure.
	 * @param [in] quad_idx Quad index.
	 * @param [in] tree_idx Tree index.
	 * @param [in] nodes Nodes structure.
	 * @param [in] tree_dimensions Owning tree dimensions.
	 * @param [in] phi_p Pointer to a double array of level-set values.
	 * @param [in] midBounds Array of max height from wall for mid layers.  If no mid layers exist or lmid_delta_percent is invalid,
	 * 			   midBounds is empty.  In special refinement, this array applies to the plastron.
	 * @param [in] midBoundsRidge Array of max height from wall for mid layers for grid points lying on or above solid ridges.  Only valid
	 * 			   if special refinement is enabled.
	 */
	void tag_quadrant( p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
					   const double *tree_dimensions, const double *phi_p, const std::vector<double>& midBounds,
					   const std::vector<double>& midBoundsRidge );

#ifdef P4_TO_P8
	/**
	 * Normalize a z-value to poitch coordinates (i.e., in the range of [0, P)).
	 * @param [in] z Input val.
	 * @return Normalized value between 0 and P, excluding P.
	 */
	double _normalize_z( const double& z ) const;
#endif

	/**
	 * Numerical offset for gratings (similarly done in my_p4est_shs_channel.h class).
	 * @return Grating offset along the z-direction.
	 */
	double _offset() const;

public:
	const double DELTA;					// Channel half-height (on the y-axis).
	const double LMID_DELTA_PERCENT;	// How far to extend mid-level cells (use 0 to disable this option).
	const bool SPECIAL_REFINEMENT;		// Whether we'll use separate special refinement for plastrons and ridges.
	const int PLASTRON_MAX_LVL;			// Maximum level of refinement for plastron.
	const double GF;					// Gas fraction.
	const double P;						// Pitch.
	const double XYZ_DIM[P4EST_DIM];	// Channel dimensions.
	const double XYZ_MIN[P4EST_DIM];	// Min coords of channel.
	const double XYZ_MAX[P4EST_DIM];	// Max coords of channel.
	const int N_TREES[P4EST_DIM];		// Number of trees in each direction.
#ifdef P4_TO_P8
	const bool SPANWISE;				// Whether the gratings are perpendicular to the flow.
#endif

	/**
	 * Constructor.
	 * @param [in] minLvl Quad/Octree minimum level of refinement.
	 * @param [in] maxLvl Quad/Octree maximum level of refinement.
	 * @param [in] phi Level-set object.
	 * @param [in] uniformBand Desired uniform band next to the wall, using as a reference dy_min over the plastron.  If special refinement
	 * 			   is enabled, the uniform band over the ridge will have approximately the same half width as the plastron's uniform band.
	 * @param [in] delta Channel half-height.
	 * @param [in] lmidDeltaPercent How far to extend mid-level cells away from the wall.  Value must be in [0,1), where 0 disables the option.
	 * @param [in] lip Lipschitz constant.
	 * @param [in] gf Gas fraction in the interval (0, 1).
	 * @param [in] pitch Gratings length (ridge + plastron).
	 * @param [in] xyzDim Channel dimensions (not normalized by delta).
	 * @param [in] nTrees Number of trees in each direction.
	 * @param [in] spRef Whether to use special refinement for plastrons and ridges.
	 * @param [in] spanwise Whether the gratings are perpendicular to the flow (option available only in 3D).
	 */
	splitting_criteria_cf_and_uniform_band_shs_t( const int& minLvl, const int& maxLvl, const CF_DIM *phi, const double& uniformBand,
												  const double& delta, const double& lmidDeltaPercent, const double& lip, const double& gf,
												  const double& pitch, const double xyzDim[P4EST_DIM], const int nTrees[P4EST_DIM],
												  const bool& spRef=false ONLY3D(COMMA const bool& spanwise=false) ) :
		splitting_criteria_cf_and_uniform_band_t( minLvl, maxLvl, phi, uniformBand, lip ),
		DELTA( delta ), LMID_DELTA_PERCENT( lmidDeltaPercent ), GF( gf ),  P( pitch ), XYZ_DIM{DIM( xyzDim[0], xyzDim[1], xyzDim[2] )},
		XYZ_MIN{DIM( -xyzDim[0]/2, -xyzDim[1]/2, -xyzDim[2]/2 )}, XYZ_MAX{DIM( xyzDim[0]/2, xyzDim[1]/2, xyzDim[2]/2 )},
		N_TREES{DIM( nTrees[0], nTrees[1], nTrees[2] )}, SPECIAL_REFINEMENT( spRef ), PLASTRON_MAX_LVL( spRef? (maxLvl - 1) : maxLvl )
		ONLY3D(COMMA SPANWISE( spanwise ))
	{
		std::string errorPrefix = "[CASL_ERROR] splitting_criteria_cf_and_uniform_band_shs_t::constructor: ";

		if( minLvl < 0 || minLvl > maxLvl )
			throw std::invalid_argument( errorPrefix + "Invalid min and max levels of refinement!" );

		// Validate only if we expect mid-level cells and user wants to use mid-level cell percentage option.
		if( maxLvl > minLvl && (lmidDeltaPercent < 0 || lmidDeltaPercent >= 1) )
			throw std::invalid_argument( errorPrefix + "Mid-level-cell extension must be non-negative and no more than 1*delta away from "
													   "the wall." );

		// For special refinement, we'll use one layer of the smallest dy for the whole wall, and then the next level of refinement for the
		// plastron will be maxLvl - 2 <---this is considered the max lvl of ref for the plastron, whereas the ridge will undergo a banded
		// discretization starting next to the wall at the user-defined max lvl of ref.
		if( SPECIAL_REFINEMENT && maxLvl - minLvl < 3 )
		{
			// Require at least 2 different lvls of ref over plastron: one for the uniform band and other for discretizing the rest.
			throw std::invalid_argument( errorPrefix + "The difference between min and max levels of refinement must be at least 3 "
													   "for the special discretization!" );
		}

		if( GF <= PETSC_MACHINE_EPSILON || GF >= 1 - PETSC_MACHINE_EPSILON )
			throw std::invalid_argument( errorPrefix + "Gas fraction must be in the range of (0, 1), excluding the end points!" );
	}

	/**
	 * Refining/coarsening function to update the underlying grid.
	 * @param [in,out] p4est P4est structure.
	 * @param [in] nodes Nodes structure (these don't change here -- you need to update them in calling function).
	 * @param [in] phi Nodal level-set values.
	 * @return True if grid has changed; false othewise.
	 */
	bool refine_and_coarsen( p4est_t* p4est, p4est_nodes_t* nodes, Vec phi );

	/**
	 * Check if point is on or hovers a solid ridge.
	 * @param [in] xyz Point coords.
	 * @return True is on/above ridge, false otherwise.
	 */
	bool is_ridge( const double xyz[P4EST_DIM] ) const;

	/**
	 * Retrieve the uniform band property.
	 * @return uniform band half width.
	 */
	double uniformBand() const
	{
		return uniform_band;
	}

	/**
	 * Calculate the limits for each band a different levels.  These bands take place above the height determined by the uniform band as
	 * computed on the plastron.
	 * @param [in] plastronSmallest_dy Smallest mesh dy for plastron, used to compute the uniform band half width on the whole wall.
	 * @param [out] midBounds The limits for each intermediate refinement level (i.e., betwen min_lvl and max_lvl, excluding the ends).
	 * @param [in] isRidge Whether calculations should be performed for the solid ridge or the plastron.
	 * @return True if computations succeeded; false otherwise.  You can also check for success if midBounds is non-empty.
	 */
	bool getBandedBounds( const double& plastronSmallest_dy, std::vector<double>& midBounds, const bool& isRidge=false ) const;
};

#endif // REFINE_COARSEN_H
