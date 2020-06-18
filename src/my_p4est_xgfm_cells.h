#ifndef MY_P4EST_XGFM_CELLS_H
#define MY_P4EST_XGFM_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_cells.h>
#else
#include <src/my_p4est_poisson_jump_cells.h>
#endif

const static double xgfm_threshold_cond_number_lsqr = 1.0e4;

class my_p4est_xgfm_cells_t : public my_p4est_poisson_jump_cells_t
{
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec residual;   // cell-sampled, residual = A*solution - rhs(jump_u, jump_normal_flux_u, extension_of_interface_defined_values)
  Vec extension;  // cell-sampled, extension of interface-defined values
  Vec grad_jump;  // node-sampled, P4EST_DIM block-structure, gradient of jump_u, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  my_p4est_interpolation_nodes_t * interp_grad_jump;
  bool activate_xGFM;
  double xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual;

  class solver_monitor_t {
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
    void log_iteration(const double& max_correction, const my_p4est_xgfm_cells_t* solver)
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

  // (possibly memorized) extension operators for interface-defined values
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
    bool in_band, in_positive_domain;
    inline void clear()
    {
      regular_terms.clear();
      interface_terms.clear();
    }

    inline double operator()(const double* extension_n_p, // input for regular neighbor terms (same side of the interface)
                             const double* solution_p, const double* current_extension_p, my_p4est_xgfm_cells_t& solver, // input for evaluating interface-defined values
                             double& max_correction_in_band) const // inout control parameter
    {
      double increment = regular_terms(extension_n_p);
      if(interface_terms.size() > 0)
      {
        const double& mu_this_side  = (in_positive_domain ? solver.mu_plus   : solver.mu_minus);
        const double& mu_across     = (in_positive_domain ? solver.mu_minus  : solver.mu_plus);
        const bool extending_positive_values = !solver.extend_negative_interface_values();
        for (size_t k = 0; k < interface_terms.size(); ++k)
        {
          const xgfm_jump& jump_info = solver.get_xgfm_jump_between_quads(quad_idx, interface_terms[k].neighbor_quad_idx_across, interface_terms[k].oriented_dir);
          const FD_interface_neighbor& FD_interface_neighbor = solver.interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, interface_terms[k].neighbor_quad_idx_across, interface_terms[k].oriented_dir);
          increment += interface_terms[k].weight*FD_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, interface_terms[k].oriented_dir, in_positive_domain, extending_positive_values,
                                                                                           solution_p[quad_idx], solution_p[interface_terms[k].neighbor_quad_idx_across], jump_info.jump_field, jump_info.jump_flux_component(current_extension_p), solver.dxyz_min);
        }
      }
      if(in_band)
        max_correction_in_band = MAX(fabs(increment), max_correction_in_band);

      return increment;
    }
  };
  std::vector<extension_increment_operator> pseudo_time_step_increment_operator;
  const extension_increment_operator& get_extension_increment_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double& control_band);
  bool extension_operators_are_stored_and_set;

  // memorized jump information for interface-point between quadrants
  map_of_xgfm_jumps_t xgfm_jump_between_quads;
  linear_combination_of_dof_t build_xgfm_jump_flux_correction_operator_at_point(const double* xyz, const double* normal,
                                                                                const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& flux_component) const;
  const xgfm_jump& get_xgfm_jump_between_quads(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir);


  // disallow copy ctr and copy assignment
  my_p4est_xgfm_cells_t(const my_p4est_xgfm_cells_t& other);
  my_p4est_xgfm_cells_t& operator=(const my_p4est_xgfm_cells_t& other);

  // internal procedures
  void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const;
  bool solve_for_fixpoint_solution(Vec& former_solution);

  // using PDE extrapolation : uses the current solution results and jump conditions to extend the appropriate
  // interface-defined values in the normal directions, using ASLAM's PDE-based extrapolation on the cells
  void extend_interface_values(Vec &former_extension, const double& threshold = 1.0e-10, const uint& niter_max = 20);
  // updates the right-hand side terms for cells involving jump terms (after extension has been updated)
  void update_rhs_in_relevant_cells_only();
  void update_rhs_and_residual(Vec& former_rhs, Vec& former_residual);
  double set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension, Vec former_rhs, Vec former_residual);

  void initialize_extension(Vec cell_sampled_extension);
  void initialize_extension_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                  const double* solution_p, double* extension_p) const;

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL);

  inline void make_sure_solution_is_set()
  {
    if(solution == NULL)
      solve_for_sharp_solution();
    P4EST_ASSERT(solution != NULL);
  }

  inline void make_sure_extensions_are_defined()
  {
    make_sure_solution_is_set();
    if(extension == NULL)
    {
      P4EST_ASSERT(!activate_xGFM || mus_are_equal()); // those are the (only) conditions under which the extension on cells can possibly be not defined
      extend_interface_values(extension);
    }
    P4EST_ASSERT(extension != NULL);
    return;
  }

  double get_sharp_flux_component_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, double& phi_face) const;

public:
  my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_xgfm_cells_t();

  inline void set_jumps(Vec jump_u_, Vec jump_normal_flux_u_)
  {
    my_p4est_poisson_jump_cells_t::set_jumps(jump_u_, jump_normal_flux_u_);

    if(activate_xGFM)
    {
      const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
      if(grad_jump == NULL){
        PetscErrorCode ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), P4EST_DIM, &grad_jump); CHKERRXX(ierr); }
      interface_capturing_ngbd_n.first_derivatives_central(jump_u, grad_jump);
      if(interp_grad_jump == NULL)
        interp_grad_jump = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);

      interp_grad_jump->set_input(grad_jump, linear, P4EST_DIM);
    }
  }

  inline Vec get_extended_interface_values()                          { make_sure_extensions_are_defined();     return extension; }
  inline Vec get_solution()                                           { make_sure_solution_is_set();            return solution;  }
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

  void set_xGFM_absolute_value_threshold(const double& abs_thresh)              { P4EST_ASSERT(abs_thresh > 0.0);           xGFM_absolute_accuracy_threshold  = abs_thresh; }
  void set_xGFM_relative_residual_threshold(const double& rel_residual_thresh)  { P4EST_ASSERT(rel_residual_thresh > 0.0);  xGFM_tolerance_on_rel_residual    = rel_residual_thresh; }

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
        sharp_integral_solution += (interface_manager->phi(xyz_quad) <= 0.0 ? positive_volume : -negative_volume)*(*interp_jump_u)(xyz_quad);
      }
    }
    ierr = VecRestoreArray(solution, &sol_p); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &sharp_integral_solution, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    return sharp_integral_solution;
  }

  void inline activate_xGFM_corrections(const bool flag_) { activate_xGFM = flag_; }

};

#endif // MY_P4EST_XGFM_CELLS_H

