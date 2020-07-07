#ifndef MY_P4EST_POISSON_JUMP_CELLS_FV_H
#define MY_P4EST_POISSON_JUMP_CELLS_FV_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_cells.h>
#else
#include <src/my_p4est_poisson_jump_cells.h>
#endif

class my_p4est_poisson_jump_cells_fv_t : public my_p4est_poisson_jump_cells_t
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
  const int correction_function_communication_tag = 1493; // Don't go with crazy large values (except if you want to spend days figuring out why it fails on several nodes...)

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
  bool                          are_required_finite_volumes_and_correction_functions_known;

  void build_finite_volumes_and_correction_functions();

  void build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx);

  bool is_point_in_slow_side(const char& sgn_point) const { return mus_are_equal() || ((mu_minus < mu_plus) == (sgn_point < 0)); }


  void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const;

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL);

  double get_sharp_flux_component_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, char& sgn_face) const;

public:
  my_p4est_poisson_jump_cells_fv_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_poisson_jump_cells_fv_t() {}; // no extra data allocated dynamically

  void solve_for_sharp_solution(const KSPType &ksp = KSPBCGS, const PCType& pc = PCHYPRE);

  inline double get_sharp_integral_solution() const
  {
    PetscErrorCode ierr;
    P4EST_ASSERT(solution != NULL);
    double *sol_p;
    ierr = VecGetArray(solution, &sol_p); CHKERRXX(ierr);

    double sharp_integral_solution = 0.0;
    const double *tree_xyz_min, *tree_xyz_max;
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
      const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        const p4est_quadrant_t* quad;
        fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
        const double cell_volume = MULTD((tree_xyz_max[0] - tree_xyz_min[0])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN),
            (tree_xyz_max[1] - tree_xyz_min[1])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN),
            (tree_xyz_max[2] - tree_xyz_min[2])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN));
        sharp_integral_solution += sol_p[quad_idx]*cell_volume;
        map_of_finite_volume_t::const_iterator it_fv = finite_volume_data_for_quad.find(quad_idx);
        if(it_fv != finite_volume_data_for_quad.end())
        {
          const my_p4est_finite_volume_t& finite_volume = it_fv->second;
          double xyz_quad[P4EST_DIM]; xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad);
          const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);
          map_of_correction_functions_t::const_iterator it_corr_fun = correction_function_for_quad.find(quad_idx);
          if(it_corr_fun == correction_function_for_quad.end())
            throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::get_sharp_integral_solution() : couldn't find the correction function for local quadrant " + std::to_string(quad_idx));
          const correction_function_t& correction_function = correction_function_for_quad.at(quad_idx);
          sharp_integral_solution -= sgn_quad*(sgn_quad < 0 ? finite_volume.volume_in_positive_domain() : finite_volume.volume_in_negative_domain())*correction_function(sol_p);
        }
      }
    }
    ierr = VecRestoreArray(solution, &sol_p); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &sharp_integral_solution, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    return sharp_integral_solution;
  }

  inline void print_solve_info() const
  {
    PetscInt nksp_iterations;
    PetscErrorCode ierr = KSPGetIterationNumber(ksp, &nksp_iterations); CHKERRXX(ierr);
    if(p4est->mpirank == 0)
      std::cout << "The solver converged after a total of " << nksp_iterations << " iterations." << std::endl;
  }

};

#endif // MY_P4EST_POISSON_JUMP_CELLS_FV_H
