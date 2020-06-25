#ifndef MY_P4EST_POISSON_JUMP_CELLS_FV_H
#define MY_P4EST_POISSON_JUMP_CELLS_FV_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_cells.h>
#else
#include <src/my_p4est_poisson_jump_cells.h>
#endif

class my_p4est_poisson_jump_cells_fv : public my_p4est_poisson_jump_cells_t
{
  union global_correction_function_elementary_data_t
  {
    p4est_gloidx_t  quad_global_idx;
    double          jump_dependent_terms;
    size_t          n_solution_dependent_terms;
    p4est_gloidx_t  solution_dependent_term_global_index;
    double          solution_dependent_term_weight;
  };

  // arbitrary-defined tag used to label the communications between processes related to correction function data
  const static int correction_function_communication_tag = 14789632;

  struct correction_function_t {
    double                      jump_dependent_terms;
    linear_combination_of_dof_t solution_dependent_terms;
    double operator()(const double* sharp_solution_p) const
    {
      return jump_dependent_terms + solution_dependent_terms(sharp_solution_p);
    }
    correction_function_t() { solution_dependent_terms.clear(); }
  };


#if __cplusplus >= 201103L
  typedef std::unordered_map<p4est_locidx_t, correction_function_t> map_of_correction_functions_t;
  typedef std::unordered_map<p4est_locidx_t, my_p4est_finite_volume_t> map_of_finite_volume_t;
#else
  typedef std::map<p4est_locidx_t, correction_function_t> map_of_correction_functions_t;
  typedef std::map<p4est_locidx_t, my_p4est_finite_volume_t> map_of_finite_volume_t;
#endif

  map_of_correction_functions_t correction_function_for_quad; //
  map_of_finite_volume_t        finite_volume_data_for_quad;  // only required in local quadrants
  bool                          are_required_finite_volumes_and_correction_functions;

  void build_finite_volumes_and_correction_functions();

  void build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx);

  bool is_point_in_slow_side(const double* xyz) const { return mus_are_equal() || ((mu_minus < mu_plus) == (interface_manager->phi_at_point(xyz) <= 0.0)); }


  void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const;

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL);



public:
  my_p4est_poisson_jump_cells_fv(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_poisson_jump_cells_fv();

};

#endif // MY_P4EST_POISSON_JUMP_CELLS_FV_H
