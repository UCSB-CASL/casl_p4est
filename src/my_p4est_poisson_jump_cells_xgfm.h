#ifndef MY_P4EST_POISSON_JUMP_CELLS_XGFM_H
#define MY_P4EST_POISSON_JUMP_CELLS_XGFM_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_cells.h>
#else
#include <src/my_p4est_poisson_jump_cells.h>
#endif

typedef struct {
  double jump_field;
  double known_jump_flux_component;
  linear_combination_of_dof_t xgfm_jump_flux_component_correction;
  inline double jump_flux_component(const double* extension_p = NULL) const { return known_jump_flux_component + (extension_p != NULL  ? xgfm_jump_flux_component_correction(extension_p) : 0.0); }
} scalar_field_xgfm_jump;

#if __cplusplus >= 201103L
typedef std::unordered_map<couple_of_dofs, scalar_field_xgfm_jump, hash_functor> map_of_scalar_field_xgfm_jumps_t;
typedef std::unordered_map<couple_of_dofs, differential_operators_on_face_sampled_field, hash_functor> map_of_face_operators_for_jumps_t;
#else
typedef std::map<couple_of_dofs, scalar_field_xgfm_jump> map_of_scalar_field_xgfm_jumps_t;
typedef std::map<couple_of_dofs, differential_operators_on_face_sampled_field> map_of_face_operators_for_jumps_t;
#endif

class my_p4est_poisson_jump_cells_xgfm_t : public my_p4est_poisson_jump_cells_t
{
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec residual;   // cell-sampled, residual = A*solution - rhs(jump_u, jump_normal_flux_u, extension_of_interface_defined_values)
  Vec extension;  // cell-sampled, extension of interface-defined values
  Vec grad_jump;  // node-sampled, P4EST_DIM block-structure, gradient of jump_u, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  my_p4est_interpolation_nodes_t * interp_grad_jump;
  bool activate_xGFM, print_residuals_and_corrections_with_solve_info;
  double xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual;

  class solver_monitor_t {
    friend class my_p4est_poisson_jump_cells_xgfm_t;
    typedef struct
    {
      PetscInt n_ksp_iterations;
      PetscReal L2_norm_residual;
      PetscReal L2_norm_rhs;
      double max_correction;
    } solver_iteration_log;
    std::vector<solver_iteration_log> logger;
  public:
    void clear() { logger.clear(); }
    void log_iteration(const double& max_correction, const my_p4est_poisson_jump_cells_xgfm_t* solver)
    {
      PetscErrorCode ierr;
      solver_iteration_log log_entry;
      ierr = KSPGetIterationNumber(solver->ksp, &log_entry.n_ksp_iterations); CHKERRXX(ierr);
      if(solver->residual != NULL){
        ierr = VecNorm(solver->residual, NORM_2, &log_entry.L2_norm_residual); CHKERRXX(ierr);
      }
      else
        log_entry.L2_norm_residual = NAN; // can't be computed (should happen only when logging the only iteration in standard GFM use)
      ierr = VecNorm(solver->rhs, NORM_2, &log_entry.L2_norm_rhs); CHKERRXX(ierr);
      log_entry.max_correction = max_correction;
      logger.push_back(log_entry);
    }
    size_t nsteps() const { return logger.size(); }
    size_t last_step() const { P4EST_ASSERT(nsteps() > 0); return nsteps() - 1; }
    double relative_residual(const size_t& k) const { return logger[k].L2_norm_residual/logger[k].L2_norm_rhs; }
    double latest_L2_norm_of_residual() const { return logger[last_step()].L2_norm_residual; }
    double latest_relative_residual() const { return relative_residual(last_step()); }
    size_t get_number_of_xGFM_corrections() const { return nsteps() - 1; }
    std::vector<PetscInt> get_n_ksp_iterations() const {
      std::vector<PetscInt> nksp_iter(nsteps());
      for (size_t k = 0; k < nsteps(); ++k)
        nksp_iter[k] = logger[k].n_ksp_iterations;
      return nksp_iter;
    }
    std::vector<double> get_max_corrections() const
    {
      std::vector<double> max_corrections(nsteps());
      for (size_t k = 0; k < nsteps(); ++k)
        max_corrections[k] = logger[k].max_correction;
      return max_corrections;
    }
    std::vector<double> get_relative_residuals() const {
      std::vector<double> relative_residuals(nsteps());
      for (size_t k = 0; k < nsteps(); ++k)
        relative_residuals[k] = relative_residual(k);
      return relative_residuals;
    }

    bool reached_convergence_within_desired_bounds(const double& absolute_accuracy_threshold, const double& tolerance_on_rel_residual) const
    {
      const size_t last_step_idx = last_step();
      return logger[last_step_idx].max_correction < absolute_accuracy_threshold && // the latest max_correction must be below the desired absolute accuracy requirement AND
          (relative_residual(last_step_idx) < tolerance_on_rel_residual || // either the latest relative residual is below the desired threshold as well OR
           (last_step_idx != 0 && fabs(relative_residual(last_step_idx) - relative_residual(last_step_idx - 1)) < 1.0e-6*MAX(relative_residual(last_step_idx), relative_residual(last_step_idx - 1)))); // or we have done at least two solves and we have reached a fixed-point for which the relative residual is above the desired threshold but can't really be made any smaller, apparently
    }
  } solver_monitor;

  // Extension-related objects (extension operators are memorized)
  inline bool extend_negative_interface_values() const { return mu_minus >= mu_plus; }
  struct interface_extension_neighbor
  {
    double weight;
    p4est_locidx_t neighbor_quad_idx_across;
    u_char oriented_dir;
  };
  struct extension_increment_operator
  {
    p4est_locidx_t quad_idx;
    linear_combination_of_dof_t               regular_terms;
    std::vector<interface_extension_neighbor> interface_terms;
    double dtau;
    bool in_band, in_positive_domain;
    inline void clear()
    {
      regular_terms.clear();
      interface_terms.clear();
    }

    inline double operator()(const double* extension_n_p, // input for regular neighbor terms (same side of the interface)
                             const double* solution_p, const double* current_extension_p, my_p4est_poisson_jump_cells_xgfm_t& solver, // input for evaluating interface-defined values
                             const bool& fetch_positive_interface_values, double& max_correction_in_band, // inout control parameter
                             const double *normal_derivative_p = NULL) const  // for 1st-degree extrapolation
    {
      double increment = regular_terms(extension_n_p);
      if(interface_terms.size() > 0)
      {
        const double& mu_this_side  = (in_positive_domain ? solver.mu_plus   : solver.mu_minus);
        const double& mu_across     = (in_positive_domain ? solver.mu_minus  : solver.mu_plus);
        for (size_t k = 0; k < interface_terms.size(); ++k)
        {
          const scalar_field_xgfm_jump& jump_info = solver.get_xgfm_jump_between_quads(quad_idx, interface_terms[k].neighbor_quad_idx_across, interface_terms[k].oriented_dir);
          const FD_interface_neighbor& FD_interface_neighbor = solver.interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, interface_terms[k].neighbor_quad_idx_across, interface_terms[k].oriented_dir);
          increment += interface_terms[k].weight*FD_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, interface_terms[k].oriented_dir, in_positive_domain, fetch_positive_interface_values,
                                                                                           solution_p[quad_idx], solution_p[interface_terms[k].neighbor_quad_idx_across], jump_info.jump_field, jump_info.jump_flux_component(current_extension_p), solver.dxyz_min[interface_terms[k].oriented_dir/2]);
        }
      }

      if(normal_derivative_p != NULL)
        increment += dtau*normal_derivative_p[quad_idx];

      if(in_band)
        max_correction_in_band = MAX(fabs(increment), max_correction_in_band);

      return increment;
    }
  };
  std::vector<extension_increment_operator> pseudo_time_step_increment_operator;
  const extension_increment_operator& get_extension_increment_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double& control_band);
  bool extension_operators_are_stored_and_set;

  // Memorized jump information for interface-point between quadrants
  map_of_scalar_field_xgfm_jumps_t xgfm_jump_between_quads;
  map_of_face_operators_for_jumps_t jump_operators_for_viscous_terms_between_quads;
  linear_combination_of_dof_t build_xgfm_jump_flux_correction_operator_at_point(const double* xyz, const double* normal,
                                                                                const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& flux_component) const;
  const scalar_field_xgfm_jump& get_xgfm_jump_between_quads(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir);
  const differential_operators_on_face_sampled_field& get_differential_operators_for_viscous_jump_terms(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir);


  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_cells_xgfm_t(const my_p4est_poisson_jump_cells_xgfm_t& other);
  my_p4est_poisson_jump_cells_xgfm_t& operator=(const my_p4est_poisson_jump_cells_xgfm_t& other);

  // preallocation-related
  void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const;

  // methods involved in the xgfm iterative procedure, minimizing the residual
  /*!
   * \brief update_solution saves the currently known solution and solves the current linear system with the best available initial guess
   * \param [out] former_solution   : solution as known by the solver before this call
   * \return true if there was indeed an update in the solution (otherwise, it means that a fix-point was found)
   */
  bool update_solution(Vec &former_solution);

  /*!
   * \brief update_extension_of_interface_values computes the new extension of relevant interface-defined values, which are function of the current
   * solution and jump conditions (which depend on the currently known extension). Then the currently known extension is saved and return to calling
   * procedure via "former_extension" and the internal "extension" is updated with the newly calculated results.
   * The extension is Aslam's PDE extension, with first order in pseudo-time forward Euler integration and subcell resolution close to the interface.
   * The derivatives of the cell-sampled extension are evaluated consistently with the derivatives defined for stable projection operators, away from
   * the interface. (The result of the extension must be a linear function of the interface-defined values to ensure convergence of the xgfm iterative
   * procedure)
   * \param [out] former_extension  : extension of relevant interface-defined values, as known by the solver before this call
   * \param [in] threshold          : [optional] absolute threshold value to stop the pseudo-time procedure: if the interface-extended values do
   *                                  not change by more than this threshold over one pseudo-time step in a band of 3 diag from the interface, it is
   *                                  assumed that convergence is reached. Default value is 1e-10
   * \param [in] niter_max          : [optional] maximum number of pseudo time steps for the extension. Default value is 20.
   */
  void update_extension_of_interface_values(Vec &former_extension, const double& threshold = 1.0e-10, const uint& niter_max = 20);

  /*!
   * \brief update_rhs_and_residual saves the currently known (discretized) rhs and residual and updates them thereafter (given the current "extension"
   * and the associated Cartesian jump conditions)
   * \param [out] former_rhs        : discretized rhs, as known by the solver before this call
   * \param [out] former_residual   : residual (Ax - b), as known by the solver before this call
   */
  void update_rhs_and_residual(Vec &former_rhs, Vec &former_residual);

  /*!
   * \brief set_solver_state_minimizing_L2_norm_of_residual linearly combines the former solver's state (provided by the user)
   * with the current one (understood as the result of a fix-point update) in such a way that the linearly combined states
   * minimize the L2 norm of the residual. If no former state is actually known (i.e., if it is the very first pass through the
   * procedure, meaning that former_residual, former_solution and former_extension are all NULL), then the current state is left
   * unchanged.
   * \param [in] former_solution  : solution as known by the solver before the fixpoint update
   * \param [in] former_extension : extension of the relevant interface-defined values, as known by the solver before the fixpoint update
   * \param [in] former_rhs       : discretized rhs, as known by the solver before the fixpoint update
   * \param [in] former_residual  : (fixpoint) residual of the targeted jump problem, as known by the solver before the fixpoint update
   * \return the maximum of the absolute value of the correction to "former_solution" defining the new solver's solution (if a
   * former_solution was provided, 0.0 otherwise)
   */
  double set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension, Vec former_rhs, Vec former_residual);

  void initialize_extension(Vec cell_sampled_extension);
  void initialize_extension_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                  const double* solution_p, double* extension_p) const;

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL);

  inline void make_sure_extensions_are_defined()
  {
    if(solution == NULL)
      solve_for_sharp_solution();
    if(extension == NULL)
    {
      P4EST_ASSERT(!activate_xGFM || mus_are_equal()); // those are the (only) conditions under which the extension on cells can possibly be not defined
      Vec dummy = NULL;
      update_extension_of_interface_values(dummy); // argument is dummy in that case...
    }
    return;
  }

  void local_projection_for_face(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces,
                                 double* flux_component_minus_p[P4EST_DIM], double* flux_component_plus_p[P4EST_DIM],
                                 const double* face_velocity_star_minus_kp1_p[P4EST_DIM], const double* face_velocity_star_plus_kp1_p[P4EST_DIM],
                                 double* divergence_free_velocity_minus_p[P4EST_DIM], double* divergence_free_velocity_plus_p[P4EST_DIM]);

  void initialize_extrapolation_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                      double* extrapolation_minus_p, double* extrapolation_plus_p,
                                      double* normal_derivative_of_solution_minus_p, double* normal_derivative_of_solution_plus_p, const u_char& degree);

  void extrapolate_solution_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                  double* tmp_minus_p, double* tmp_plus_p,
                                  const double* extrapolation_minus_p, const double* extrapolation_plus_p,
                                  const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p);

  void clear_node_sampled_jumps();
  void update_jump_terms_for_projection();

public:
  my_p4est_poisson_jump_cells_xgfm_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_poisson_jump_cells_xgfm_t();

  inline void set_jumps(Vec jump_u_, Vec jump_normal_flux_u_)
  {
    my_p4est_poisson_jump_cells_t::set_jumps(jump_u_, jump_normal_flux_u_);

    if(activate_xGFM)
    {
      if(jump_u != NULL)
      {
        const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
        if(grad_jump == NULL){
          PetscErrorCode ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), P4EST_DIM, &grad_jump); CHKERRXX(ierr); }
        interface_capturing_ngbd_n.first_derivatives_central(jump_u, grad_jump);
        if(interp_grad_jump == NULL)
          interp_grad_jump = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);

        interp_grad_jump->set_input(grad_jump, linear, P4EST_DIM);
      }
      else
      {
        if(grad_jump != NULL){
          PetscErrorCode ierr = VecDestroy(grad_jump); CHKERRXX(ierr);
          grad_jump = NULL;
        }
        if(interp_grad_jump != NULL){
          delete interp_grad_jump;
          interp_grad_jump = NULL;
        }
      }
    }
  }

  inline Vec get_extended_interface_values()                          { make_sure_extensions_are_defined();     return extension; }
  inline int get_number_of_xGFM_corrections()                   const { return solver_monitor.get_number_of_xGFM_corrections();   }
  inline std::vector<PetscInt> get_numbers_of_ksp_iterations()  const { return solver_monitor.get_n_ksp_iterations();             }
  inline std::vector<double> get_max_corrections()              const { return solver_monitor.get_max_corrections();              }
  inline std::vector<double> get_relative_residuals()           const { return solver_monitor.get_relative_residuals();           }
  inline bool is_using_xGFM()                                   const { return activate_xGFM;                                     }

  /* Benchmark tests revealed that PCHYPRE is MUCH faster than PCSOR as PCType!
   * The linear system is supposed to be symmetric positive (semi-) definite, so KSPCG is ok as KSPType
   * Note: a low threshold for tolerance_on_rel_residual is critical to ensure accuracy in cases with large differences in diffusion coefficients!
   * */
  void solve_for_sharp_solution(const KSPType& ksp_type = KSPCG, const PCType& pc_type = PCHYPRE);

  inline void set_xGFM_absolute_value_threshold(const double& abs_thresh)              { P4EST_ASSERT(abs_thresh > 0.0);           xGFM_absolute_accuracy_threshold  = abs_thresh; }
  inline void set_xGFM_relative_residual_threshold(const double& rel_residual_thresh)  { P4EST_ASSERT(rel_residual_thresh > 0.0);  xGFM_tolerance_on_rel_residual    = rel_residual_thresh; }

  inline double get_sharp_integral_solution() const
  {
    PetscErrorCode ierr;
    P4EST_ASSERT(solution != NULL);
    double *sol_p;
    ierr = VecGetArray(solution, &sol_p); CHKERRXX(ierr);

    double sharp_integral_solution = 0.0;
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
      const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        double negative_volume, positive_volume;
        interface_manager->compute_subvolumes_in_cell(quad_idx, tree_idx, negative_volume, positive_volume);

        double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
        // crude estimate but whatever, it's mostly to get closer to what we expect...
        sharp_integral_solution += sol_p[quad_idx]*(negative_volume + positive_volume);
        if(interp_jump_u != NULL)
          sharp_integral_solution += (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? positive_volume : -negative_volume)*(*interp_jump_u)(xyz_quad);
      }
    }
    ierr = VecRestoreArray(solution, &sol_p); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &sharp_integral_solution, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    return sharp_integral_solution;
  }

  inline void print_solve_info() const
  {
    if(p4est->mpirank == 0)
    {
      PetscInt total_nb_iterations = solver_monitor.logger[0].n_ksp_iterations;
      for(size_t tt = 1; tt < solver_monitor.logger.size(); ++tt){
        total_nb_iterations += solver_monitor.logger[tt].n_ksp_iterations;
        if(print_residuals_and_corrections_with_solve_info)
          std::cout << "After iterative step " << tt << "(" << solver_monitor.logger[tt].n_ksp_iterations << " iterations): " <<std::endl
                    << " \t\t max correction = " << solver_monitor.logger[tt].max_correction << std::endl
                    << " \t\t relative residual = " << solver_monitor.relative_residual(tt) << std::endl;
      }
      std::cout << "The solver converged after a total of " << total_nb_iterations << " iterations." << std::endl;
    }
  }

  inline void activate_xGFM_corrections(const bool& flag_, const bool& print_xGFM_residuals_and_corrections = false)
  {
    activate_xGFM = flag_;
    print_residuals_and_corrections_with_solve_info = activate_xGFM && print_xGFM_residuals_and_corrections;
  }

  inline bool uses_xGFM_corrections() const { return activate_xGFM; }

};

#endif // MY_P4EST_POISSON_JUMP_CELLS_XGFM_H

