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
#include <vector>

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

//struct splitting_criteria_random_t : splitting_criteria_t {
//  splitting_criteria_random_t(p4est_t *p4est, int min_lvl, int max_lvl)
//    : marked(p4est->local_num_quadrants, false)
//  {
//    this->min_lvl = min_lvl;
//    this->max_lvl = max_lvl;

//    std::vector<double> s(max_lvl - min_lvl + 1);
//    double sum = 0;
//    for (int l=0; l<max_lvl-min_lvl+1; l++) {
//      s[l] = 1.0/sqrt(l+1.0);
//      sum += s[l];
//    }

//    for (int l=0; l<max_lvl-min_lvl+1; l++)
//      s[l] /= sum;

//    volatile u_int8_t refine; // prevent compiler to optimize the loop
//    for (p4est_gloidx_t i = 0; i<p4est->global_first_quadrant[p4est->mpirank]; i++)
//      refine = ranged_rand(0.,1.) < 0.5;
//    for (p4est_topidx_t tr = p4est->first_local_tree; tr <= p4est->last_local_tree; tr++){
//      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
//      for (size_t qu = 0; qu < tree->quadrants.elem_count; qu++){
//        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
//        p4est_locidx_t q = qu + tree->quadrants_offset;

//        if (quad->level < min_lvl)
//          marked[q] = 1;
//        else if (quad->level > max_lvl)
//          marked[q] = 0;
//        else
//          marked[q] = ranged_rand(0.,1.) < s[quad->level - min_lvl];

//        quad->p.user_data = &marked[qu+tree->quadrants_offset];
//      }
//    }
//    for (p4est_gloidx_t i = p4est->global_first_quadrant[p4est->mpirank+1]; i<p4est->global_num_quadrants; i++)
//      refine = ranged_rand(0.,1.) < 0.5;
//  }

//private:
//  std::vector<u_int8_t> marked;
//};

class splitting_criteria_marker_t: public splitting_criteria_t {
  std::vector<p4est_bool_t> markers;
public:
  splitting_criteria_marker_t(p4est_t *p4est, int min_lvl, int max_lvl)
    : markers(p4est->local_num_quadrants, P4EST_FALSE)
  {
    this->min_lvl = min_lvl;
    this->max_lvl = max_lvl;

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

/*!
 * \brief refine_every_cell refines all the cell in the p4est
 * \param p4est      [in] forest object to consider
 * \param which_tree [in] current tree to which the quadrant belongs
 * \param quad       [in] pointer to the current quadrant
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

p4est_bool_t
refine_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad);

p4est_bool_t
coarsen_marked_quadrants(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad);

void
my_p4est_refine_quadrant(p4est_t *p4est, p4est_topidx_t which_tree, p4est_locidx_t which_quad);


#endif // REFINE_COARSEN_H
