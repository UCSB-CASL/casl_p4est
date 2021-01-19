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
    bool            using_fast_side;
    size_t          local_corr_fun_idx;
  };

  // arbitrary-defined tag used to label the communications between processes related to correction function data
  const int correction_function_communication_tag = 1493; // Don't go with crazy large values (except if you want to spend days figuring out why it fails when using several computer nodes...)

  struct correction_function_t {
    double                      jump_dependent_terms;
    linear_combination_of_dof_t solution_dependent_terms;
    double operator()(const double* sharp_solution_p) const
    {
      return jump_dependent_terms + solution_dependent_terms(sharp_solution_p);
    }
    bool not_reliable_for_extrapolation;
    correction_function_t() {
      solution_dependent_terms.clear();
      not_reliable_for_extrapolation = false;
    }
  };

#if __cplusplus >= 201103L
  typedef std::unordered_map<p4est_locidx_t, correction_function_t> map_of_correction_functions_t;
  typedef std::unordered_map<p4est_locidx_t, my_p4est_finite_volume_t> map_of_finite_volume_t;
  typedef std::unordered_map<p4est_locidx_t, size_t> map_of_local_quad_to_corr_fun_t;
#else
  typedef std::map<p4est_locidx_t, correction_function_t> map_of_correction_functions_t;
  typedef std::map<p4est_locidx_t, my_p4est_finite_volume_t> map_of_finite_volume_t;
  typedef std::map<p4est_locidx_t, size_t> map_of_local_quad_to_corr_fun_t;
#endif

  map_of_local_quad_to_corr_fun_t local_corr_fun_for_layer_quad;
  map_of_local_quad_to_corr_fun_t local_corr_fun_for_inner_quad;
  map_of_local_quad_to_corr_fun_t local_corr_fun_for_ghost_quad;
  vector<int> offset_corr_fun_on_proc;
  vector<PetscInt> global_idx_of_ghost_corr_fun;
  inline PetscErrorCode VecCreateGhostCellCorrFun(Vec *vv) const
  {
    PetscErrorCode ierr;
    ierr = VecCreateGhost(p4est->mpicomm, local_corr_fun_for_inner_quad.size() + local_corr_fun_for_layer_quad.size(),
                          offset_corr_fun_on_proc[p4est->mpisize], global_idx_of_ghost_corr_fun.size(), global_idx_of_ghost_corr_fun.data(), vv); CHKERRQ(ierr);
    return ierr;
  }
  Vec jump_terms_in_corr_fun;

  map_of_correction_functions_t correction_function_for_quad;
  map_of_finite_volume_t        finite_volume_data_for_quad;  // only required in local quadrants
  bool                          are_required_finite_volumes_and_correction_functions_known;
  double                        interface_relative_threshold;
  double                        threshold_volume_ratio_for_extrapolation;
  double                        reference_face_area;
  bool                          pin_normal_derivative_for_correction_functions;

  void build_finite_volumes_and_correction_functions();

  void build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, map_of_local_quad_to_corr_fun_t* map_quad_to_cf = NULL);

  bool is_point_in_slow_side(const char& sgn_point) const { return mus_are_equal() || ((mu_minus < mu_plus) == (sgn_point < 0)); }


  void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const;

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL);

  void local_projection_for_face(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces,
                                 double* flux_component_minus_p[P4EST_DIM], double* flux_component_plus_p[P4EST_DIM],
                                 double* face_velocity_minus_p[P4EST_DIM], double* face_velocity_plus_p[P4EST_DIM]) const;

  void initialize_extrapolation_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                      double* extrapolation_minus_p, double* extrapolation_plus_p,
                                      double* normal_derivative_of_solution_minus_p, double* normal_derivative_of_solution_plus_p, const u_char& degree);

  void extrapolate_solution_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* /*dummy argument for this solver*/,
                                  double* tmp_minus_p, double* tmp_plus_p,
                                  const double* extrapolation_minus_p, const double* extrapolation_plus_p,
                                  const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p);

  void clear_node_sampled_jumps();

public:
  my_p4est_poisson_jump_cells_fv_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_poisson_jump_cells_fv_t();

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

  /*!
   * \brief set_interface_relative_threshold sets a new threshold value for considering that the face area
   * connecting two (actual or ghost) degrees of freedom is not zero: the face-connection is disregarded if
   * the corresponding face area connecting the degrees of freedom is smaller than
   *                       interface_relative_threshold*pow(diag_min, P4EST_DIM - 1).
   * The default value for interface_relative_threshold is 1.0e-11.
   * \param threshold_value [in] new value to be set for interface_relative_threshold.
   */
  inline void set_interface_relative_threshold(const double& threshold_value) { P4EST_ASSERT(threshold_value >= 0.0); interface_relative_threshold = threshold_value; }

  /*!
   * \brief set_threshold_volume_ratio_for_extrapolation sets a new threshold for the ratio of volumes in cut cells
   * that invalidates the use of correction function when determining extrapolated values from one side of the
   * interface to the other.
   * The correction function is used to determine the extrapolated value from the other side in cells that are cut
   * by the interface if and only if
   *
   * 1) the correction function actually uses the "slow side" to evaluate the normal derivative at the
   * interface;
   *
   * 2) the volume that lies across the interafce in the cut is larger
   *                   threshold_volume_ratio_for_extrapolation*(full volume of the cell)
   * \param desired_threshold_volume_ratio [in] new value to be set for threshold_volume_ratio.
   *
   * The default value for threshold_volume_ratio_for_extrapolation is 0.01: this value value was found to be good
   * enough for stable two-phase flow simulations (in the very first early stable runs). If such a criterion was
   * disregarded (i.e., if threshold_volume_ratio_for_extrapolation <-- 0.00), the error associated with
   * correction-function-defined ghost value was found to spike in cells crossed by tiny volumes across the interface,
   * therefore creating large gradient errors and inappropriately large velocity values leading to unstable runs.
   * Note that a value larger than or equal to 1.0 would invalidate the use of ANY correction functions for
   * extrapolation purposes: only inner cell values would be used. While this probably sounds like the "safest" approach,
   * error analyses for sharp flux components revealed that
   * 1) their rate of convergence was more erratic;
   * 2) the error could be the one order of magnitude larger close to the interface;
   * if disabling the use of ANY correction function.
   * (In conclusion, this control parameter is probably worth a much more thorough analysis in order to be defined in a
   * less arbitrary fashion, but this was deemed "not important" by authorities and instructions to move on were given)
   */
  inline void set_threshold_volume_ratio(const double& desired_threshold_volume_ratio) { P4EST_ASSERT(desired_threshold_volume_ratio >= 0.0); threshold_volume_ratio_for_extrapolation = desired_threshold_volume_ratio; }

  /*!
   * \brief set_pinning_for_normal_derivatives_in_correction_functions sets the internal flag controlling the use of
   * quad-center pinning when building the normal derivatives required for the correction functions
   * \param do_the_pinning [in] action desired by the user;
   */
  inline void set_pinning_for_normal_derivatives_in_correction_functions(const bool& do_the_pinning) { pin_normal_derivative_for_correction_functions = do_the_pinning;}

};

#endif // MY_P4EST_POISSON_JUMP_CELLS_FV_H
