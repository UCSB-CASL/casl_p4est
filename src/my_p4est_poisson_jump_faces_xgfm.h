#ifndef MY_P4EST_POISSON_JUMP_FACES_XGFM_H
#define MY_P4EST_POISSON_JUMP_FACES_XGFM_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_faces.h>
#else
#include <src/my_p4est_poisson_jump_faces.h>
#endif

typedef struct {
  double jump_component;
  double known_jump_flux;
  linear_combination_of_dof_t xgfm_jump_flux_tangential_correction[P4EST_DIM];  // works on extensions
  linear_combination_of_dof_t xgfm_jump_flux_normal_correction[P4EST_DIM];      // works on extrapolations
  inline double jump_flux_component(const double* extension_p[P4EST_DIM], const double* extrapolation_p[P4EST_DIM]) const
  {
    return known_jump_flux
        + (ANDD(extension_p[0] != NULL,     extension_p[1] != NULL,     extension_p[2] != NULL)     ? SUMD(xgfm_jump_flux_tangential_correction[0](extension_p[0]), xgfm_jump_flux_tangential_correction[1](extension_p[1]),  xgfm_jump_flux_tangential_correction[2](extension_p[2])) : 0.0)
        + (ANDD(extrapolation_p[0] != NULL, extrapolation_p[1] != NULL, extrapolation_p[2] != NULL) ? SUMD(xgfm_jump_flux_normal_correction[0](extrapolation_p[0]), xgfm_jump_flux_normal_correction[1](extrapolation_p[1]),  xgfm_jump_flux_normal_correction[2](extrapolation_p[2])) : 0.0);
  }
} vector_field_component_xgfm_jump;

#if __cplusplus >= 201103L
typedef std::unordered_map<couple_of_dofs, vector_field_component_xgfm_jump, hash_functor> map_of_vector_field_component_xgfm_jumps_t;
#else
typedef std::map<couple_of_dofs, vector_field_component_xgfm_jump> map_of_vector_field_component_xgfm_jumps_t;
#endif

class my_p4est_poisson_jump_faces_xgfm_t : public my_p4est_poisson_jump_faces_t
{
  /* NOTE : Voronoi finite-volume treatment far away from the interface, (x)gfm treatment of interface jump
   * conditions. (No Shortley-Weller kind of treatment in order to keep the systems symmetric).
   */

  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec residual[P4EST_DIM];  // face-sampled, residual = A*solution - rhs(jump_u, jump_normal_flux_u, extension_of_interface_defined_values)
  Vec grad_jump_u_dot_n;    // node-sampled, P4EST_DIM block-structure, gradient of jump_u_dot_n, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  Vec extension[P4EST_DIM]; // face-sampled, extension of interface-defined component values

  my_p4est_interpolation_nodes_t *interp_grad_jump_u_dot_n;
  bool activate_xGFM, print_residuals_and_corrections_with_solve_info, use_face_dofs_only_in_extrapolations;
  double xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual;

  // - BEGIN validation data only -
  Vec validation_jump_u;          // node-sampled, P4EST_DIM block-structure, jump in every component of the solution defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  Vec validation_jump_mu_grad_u;  // node-sampled, P4EST_DIM block-structure, jump in every component of mu*grad(u) defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  my_p4est_interpolation_nodes_t *interp_validation_jump_u;
  my_p4est_interpolation_nodes_t *interp_validation_jump_mu_grad_u;
  bool set_for_testing_backbone;
  // - END validation data only -

  class solver_monitor_t {
    friend class my_p4est_poisson_jump_faces_xgfm_t;
    typedef struct
    {
      PetscInt n_ksp_iterations;
      PetscReal L2_norm_residual;
      PetscReal L2_norm_rhs;
      double max_correction[P4EST_DIM];
    } solver_iteration_log;
    std::vector<solver_iteration_log> logger;
  public:
    void clear() { logger.clear(); }
    void log_iteration(const double max_correction[P4EST_DIM], const my_p4est_poisson_jump_faces_xgfm_t* solver)
    {
      PetscErrorCode ierr;
      solver_iteration_log log_entry;
      log_entry.n_ksp_iterations  = 0;
      log_entry.L2_norm_residual  = 0.0;
      log_entry.L2_norm_rhs       = 0.0;
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        PetscInt latest_nksp_dim;
        PetscReal L2_norm_latest_residual_dim, L2_norm_latest_rhs;
        ierr = KSPGetIterationNumber(solver->ksp[dim], &latest_nksp_dim); CHKERRXX(ierr);
        if(solver->residual[dim] != NULL){
          ierr = VecNorm(solver->residual[dim], NORM_2, &L2_norm_latest_residual_dim); CHKERRXX(ierr);
          log_entry.L2_norm_residual += SQR(L2_norm_latest_residual_dim);
        }
        else
          log_entry.L2_norm_residual = NAN; // can't be computed (should happen only when logging the only iteration in standard GFM use)
        ierr = VecNorm(solver->rhs[dim], NORM_2, &L2_norm_latest_rhs); CHKERRXX(ierr);

        log_entry.L2_norm_rhs += SQR(L2_norm_latest_rhs);

        log_entry.n_ksp_iterations += latest_nksp_dim;
        log_entry.max_correction[dim] = max_correction[dim];
      }
      if(!ISNAN(log_entry.L2_norm_residual))
        log_entry.L2_norm_residual = sqrt(log_entry.L2_norm_residual);
      log_entry.L2_norm_rhs = sqrt(log_entry.L2_norm_rhs);

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
      std::vector<double> max_corrections(P4EST_DIM*nsteps());
      for (size_t k = 0; k < nsteps(); ++k)
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          max_corrections[P4EST_DIM*k + dim] = logger[k].max_correction[dim];
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
      return ANDD(logger[last_step_idx].max_correction[0] < absolute_accuracy_threshold, logger[last_step_idx].max_correction[1] < absolute_accuracy_threshold, logger[last_step_idx].max_correction[2] < absolute_accuracy_threshold) && // the latest max_correction must be below the desired absolute accuracy requirement AND
          (relative_residual(last_step_idx) < tolerance_on_rel_residual || // either the latest relative residual is below the desired threshold as well OR
          (last_step_idx != 0 && fabs(relative_residual(last_step_idx) - relative_residual(last_step_idx - 1)) < 1.0e-6*MAX(relative_residual(last_step_idx), relative_residual(last_step_idx - 1)))); // or we have done at least two solves and we have reached a fixed-point for which the relative residual is above the desired threshold but can't really be made any smaller, apparently
    }
  } solver_monitor;

  // Extension-related objects (extension operators are memorized)
  inline bool extend_negative_interface_values() const { return mu_minus >= mu_plus; }
  struct interface_extension_neighbor
  {
    double weight;
    p4est_locidx_t neighbor_face_idx_across;
    u_char oriented_dir;
  };
  struct extension_increment_operator
  {
    p4est_locidx_t face_idx;
    linear_combination_of_dof_t               regular_terms;
    std::vector<interface_extension_neighbor> interface_terms;
    double dtau;
    bool in_band, in_positive_domain;
    inline void clear()
    {
      regular_terms.clear();
      interface_terms.clear();
    }

    inline double operator()(const u_char& dim, // face orientation == vector component of interest
                             const double* extension_n_p[P4EST_DIM], // input for regular neighbor terms (same side of the interface)
                             const double* solution_p[P4EST_DIM], const double* current_extension_p[P4EST_DIM], const double* current_viscous_extrapolation_p[P4EST_DIM], my_p4est_poisson_jump_faces_xgfm_t& solver, // input for evaluating interface-defined values
                             const bool& fetch_positive_interface_values, double& max_correction_in_band, // inout control parameter
                             const double *normal_derivative_p[P4EST_DIM] = NULL) const  // for 1st-degree extrapolation (avoiding redundant storage of operators, if possible)
    {
      double increment = regular_terms(extension_n_p[dim]);
      if(interface_terms.size() > 0)
      {
        const double& mu_this_side  = (in_positive_domain ? solver.mu_plus   : solver.mu_minus);
        const double& mu_across     = (in_positive_domain ? solver.mu_minus  : solver.mu_plus);
        for (size_t k = 0; k < interface_terms.size(); ++k)
        {
          const vector_field_component_xgfm_jump& jump_info = solver.get_xgfm_jump_between_faces(dim, face_idx, interface_terms[k].neighbor_face_idx_across, interface_terms[k].oriented_dir);
          const FD_interface_neighbor& FD_interface_neighbor = solver.interface_manager->get_face_FD_interface_neighbor_for(face_idx, interface_terms[k].neighbor_face_idx_across, dim, interface_terms[k].oriented_dir);
          increment += interface_terms[k].weight*FD_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, interface_terms[k].oriented_dir, in_positive_domain, fetch_positive_interface_values,
                                                                                           solution_p[dim][face_idx], solution_p[dim][interface_terms[k].neighbor_face_idx_across], jump_info.jump_component, jump_info.jump_flux_component(current_extension_p, current_viscous_extrapolation_p), solver.dxyz_min[interface_terms[k].oriented_dir/2]);
        }
      }

      if(normal_derivative_p != NULL)
        increment += dtau*normal_derivative_p[dim][face_idx];

      if(in_band)
        max_correction_in_band = MAX(fabs(increment), max_correction_in_band);

      return increment;
    }
  };
  std::vector<extension_increment_operator> pseudo_time_step_increment_operator[P4EST_DIM];
  const extension_increment_operator& get_extension_increment_operator_for(const u_char& dim, const p4est_locidx_t& face_idx, const double& control_band);
  bool extension_operators_are_stored_and_set[P4EST_DIM];

  // Memorized jump information for interface-point between quadrants
  map_of_vector_field_component_xgfm_jumps_t xgfm_jump_between_faces[P4EST_DIM];
  void build_xgfm_jump_flux_correction_operators_at_point(vector_field_component_xgfm_jump& xgfm_jump_data,
                                                          const double* xyz, const double* normal,
                                                          const u_char& dim, const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir) const;
  const vector_field_component_xgfm_jump& get_xgfm_jump_between_faces(const u_char& dir, const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_faces_xgfm_t(const my_p4est_poisson_jump_faces_xgfm_t& other);
  my_p4est_poisson_jump_faces_xgfm_t& operator=(const my_p4est_poisson_jump_faces_xgfm_t& other);

  // preallocation-related
  void get_numbers_of_faces_involved_in_equation_for_face(const u_char& dir, const p4est_locidx_t& face_idx,
                                                          PetscInt& number_of_local_faces_involved, PetscInt& number_of_ghost_faces_involved);

  // methods involved in the xgfm iterative procedure, minimizing the residual
  /*!
   * \brief update_solution saves the currently known solution and solves the current linear system with the best available initial guess
   * \param [out] former_solution : solution as known by the solver before this call
   * \return true if there was indeed an update in the solution (otherwise, it means that a fix-point was found)
   */
  bool update_solution(Vec former_solution[P4EST_DIM]);

  /*!
   * \brief update_extensions_and_extrapolations computes the new
   * 1) extension of relevant interface-defined values (function of the current solution and jump conditions);
   * 2) extrapolations of the solution from either side to the other.
   * The currently known extensions and extrapolations are saved and returned to calling procedure via "former_extensions",
   * "former_extrapolation_minus" and "former_extrapolation_plus" and the internal equivalent are updated with the newly calculated results.
   * The extension and extrapolation are Aslam's PDE extension, with first order in pseudo-time forward Euler integration and subcell
   * resolution close to the interface.
   * (The result of the extension must be a linear function of the interface-defined values to ensure convergence of the xgfm iterative
   * procedure)
   * \param [out] former_extension  : extension of relevant interface-defined values, as known by the solver before this call
   * \param [out] former_extrapolation_minus : extrapolation of the soution from the minus side, as known by the solver before this call
   * \param [out] former_extrapolation_plus : extrapolation of the soution from the plus side, as known by the solver before this call
   * \param [in] threshold  : [optional] absolute threshold value to stop the pseudo-time procedure(s): if the extended/extrapolated values do
   *                           not change by more than this threshold over one pseudo-time step in a band of 3 diag from the interface, it is
   *                           assumed that convergence is reached. Default value is 1e-10
   * \param [in] niter_max  : [optional] maximum number of pseudo time steps for the extension. Default value is 20.
   */
  void update_extensions_and_extrapolations(Vec former_extension[P4EST_DIM], Vec former_extrapolation_minus[P4EST_DIM], Vec former_extrapolation_plus[P4EST_DIM],
                                            const double& threshold = 1.0e-10, const uint& niter_max = 20);

  /*!
   * \brief update_rhs_and_residual saves the currently known (discretized) rhs and residual and updates them thereafter (given the current "extension"
   * and the associated Cartesian jump conditions)
   * \param [out] former_rhs        : discretized rhs, as known by the solver before this call
   * \param [out] former_residual   : residual (Ax - b), as known by the solver before this call
   */
  void update_rhs_and_residual(Vec former_rhs[P4EST_DIM], Vec former_residual[P4EST_DIM]);

  /*!
   * \brief set_solver_state_minimizing_L2_norm_of_residual linearly combines the former solver's state (provided by the user)
   * with the current one (understood as the result of a fix-point update) in such a way that the linearly combined states
   * minimize the L2 norm of the residual. If no former state is actually known (i.e., if it is the very first pass through the
   * procedure, meaning that former_residual, former_solution and former_extension are all NULL), then the current state is left
   * unchanged.
   * \param [in]  former_solution   : solution as known by the solver before the fixpoint update
   * \param [in]  former_extension  : extension of the relevant interface-defined values, as known by the solver before the fixpoint update
   * \param [in]  former_extrapolation_minus : (linear) extrapolation of the solution from the negative domain, as known by the solver before the fixpoint update
   * \param [in]  former_extrapolation_plus  : (linear) extrapolation of the solution from the positive domain, as known by the solver before the fixpoint update
   * \param [in]  former_rhs        : discretized rhs, as known by the solver before the fixpoint update
   * \param [in]  former_residual   : (fixpoint) residual of the targeted jump problem, as known by the solver before the fixpoint update
   * \param [out] max_correction    : maximum absolute values of the corrections to "former_solution" (by component) defining the new solver's
   *                                  solution (if a former_solution was provided, 0.0 otherwise)
   */
  void set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution[P4EST_DIM],
                                                       Vec former_extension[P4EST_DIM], Vec former_extrapoltion_minus[P4EST_DIM], Vec former_extrapoltion_plus[P4EST_DIM],
                                                       Vec former_rhs[P4EST_DIM], Vec former_residual[P4EST_DIM],
                                                       double max_correction[P4EST_DIM]);

  void initialize_extensions_and_extrapolations(Vec new_extension[P4EST_DIM], Vec new_extrapolation_minus[P4EST_DIM], Vec new_extrapolation_plus[P4EST_DIM],
                                                Vec new_normal_derivative_minus[P4EST_DIM], Vec new_normal_derivative_plus[P4EST_DIM]);

  void build_discretization_for_face(const u_char& dir, const p4est_locidx_t& face_idx, int *nullspace_contains_constant_vector = NULL);

  void initialize_extrapolation_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                      double* extrapolation_minus_p[P4EST_DIM], double* extrapolation_plus_p[P4EST_DIM],
                                      double* normal_derivative_of_solution_minus_p[P4EST_DIM], double* normal_derivative_of_solution_plus_p[P4EST_DIM], const u_char& degree,
                                      double* sharp_max_component);

  void extrapolate_solution_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                  double* tmp_minus_p[P4EST_DIM], double* tmp_plus_p[P4EST_DIM],
                                  const double* extrapolation_minus_p[P4EST_DIM], const double* extrapolation_plus_p[P4EST_DIM],
                                  const double* normal_derivative_of_solution_minus_p[P4EST_DIM], const double* normal_derivative_of_solution_plus_p[P4EST_DIM]);

public:
  my_p4est_poisson_jump_faces_xgfm_t(const my_p4est_faces_t *faces_, const p4est_nodes_t* nodes_);
  ~my_p4est_poisson_jump_faces_xgfm_t();

  inline void set_jumps(Vec jump_u_dot_n_, Vec jump_tangential_stress_)
  {
    my_p4est_poisson_jump_faces_t::set_jumps(jump_u_dot_n_, jump_tangential_stress_);

    if(activate_xGFM)
    {
      if(jump_u_dot_n != NULL)
      {
        if(!interface_manager->is_curvature_set())
          interface_manager->set_curvature();
        if(!interface_manager->is_gradient_of_normal_set())
          interface_manager->set_gradient_of_normal();

        const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
        if(grad_jump_u_dot_n == NULL){
          PetscErrorCode ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), P4EST_DIM, &grad_jump_u_dot_n); CHKERRXX(ierr); }
        interface_capturing_ngbd_n.first_derivatives_central(jump_u_dot_n, grad_jump_u_dot_n);
        if(interp_grad_jump_u_dot_n == NULL)
          interp_grad_jump_u_dot_n = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);

        interp_grad_jump_u_dot_n->set_input(grad_jump_u_dot_n, linear, P4EST_DIM);
      }
      else
      {
        if(grad_jump_u_dot_n != NULL){
          PetscErrorCode ierr = delete_and_nullify_vector(grad_jump_u_dot_n); CHKERRXX(ierr); }
        if(interp_grad_jump_u_dot_n != NULL){
          delete interp_grad_jump_u_dot_n;
          interp_grad_jump_u_dot_n = NULL;
        }
      }
    }
  }

  inline void set_jumps_for_validation(Vec validation_jump_u_, Vec validation_jump_mu_grad_u_)
  {
    PetscErrorCode ierr;
    // make sure there is no interference with jumps set another way, first
    if(jump_u_dot_n != NULL)
      jump_u_dot_n = NULL;
    if(jump_tangential_stress != NULL)
      jump_tangential_stress = NULL;
    if(grad_jump_u_dot_n != NULL) {
      ierr = delete_and_nullify_vector(grad_jump_u_dot_n); CHKERRXX(ierr); }

    if(interp_jump_u_dot_n != NULL)
      delete interp_jump_u_dot_n;
    if(interp_grad_jump_u_dot_n != NULL)
      delete interp_grad_jump_u_dot_n;
    if(interp_jump_tangential_stress != NULL)
      delete interp_jump_tangential_stress;

    if(!interface_is_set())
      throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm_t::set_jumps_for_validation(): the interface manager must be set before the jumps");
    const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
#ifdef P4EST_DEBUG
    P4EST_ASSERT(validation_jump_u_         == NULL || VecIsSetForNodes(validation_jump_u_,         interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, P4EST_DIM));
    P4EST_ASSERT(validation_jump_mu_grad_u_ == NULL || VecIsSetForNodes(validation_jump_mu_grad_u_, interface_capturing_ngbd_n.get_nodes(), interface_capturing_ngbd_n.get_p4est()->mpicomm, SQR_P4EST_DIM));
#endif

    // fetch the validation input data and set the solver for validation purposes
    validation_jump_u = validation_jump_u_;
    validation_jump_mu_grad_u = validation_jump_mu_grad_u_;

    if(interp_validation_jump_u != NULL && validation_jump_u == NULL){
      delete interp_validation_jump_u;
      interp_validation_jump_u = NULL;
    }
    if(interp_validation_jump_u == NULL && validation_jump_u != NULL){
      interp_validation_jump_u = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
    }
    if(validation_jump_u != NULL)
      interp_validation_jump_u->set_input(validation_jump_u, linear, P4EST_DIM);

    if(interp_validation_jump_mu_grad_u != NULL && validation_jump_mu_grad_u == NULL){
      delete interp_validation_jump_mu_grad_u;
      interp_validation_jump_mu_grad_u = NULL;
    }
    if(interp_validation_jump_mu_grad_u == NULL && validation_jump_mu_grad_u != NULL){
      interp_validation_jump_mu_grad_u = new my_p4est_interpolation_nodes_t(&interface_capturing_ngbd_n);
    }
    if(validation_jump_mu_grad_u != NULL)
      interp_validation_jump_mu_grad_u->set_input(validation_jump_mu_grad_u, linear, SQR_P4EST_DIM);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      rhs_is_set[dim] = false;

    niter_extrapolations_done = 0;
    set_for_testing_backbone = true;

    return;
  }

//  inline Vec* get_extended_interface_values()                         { make_sure_extensions_are_defined();     return extension; }
//  inline int get_number_of_xGFM_corrections()                   const { return solver_monitor.get_number_of_xGFM_corrections();   }
//  inline std::vector<PetscInt> get_numbers_of_ksp_iterations()  const { return solver_monitor.get_n_ksp_iterations();             }
//  inline std::vector<double> get_max_corrections()              const { return solver_monitor.get_max_corrections();              }
//  inline std::vector<double> get_relative_residuals()           const { return solver_monitor.get_relative_residuals();           }
  inline bool is_using_xGFM()                                   const { return activate_xGFM;                                     }

  void solve_for_sharp_solution(const KSPType& ksp_type = KSPCG, const PCType& pc_type = PCHYPRE);
  inline void set_xGFM_absolute_value_threshold(const double& abs_thresh)              { P4EST_ASSERT(abs_thresh > 0.0);           xGFM_absolute_accuracy_threshold  = abs_thresh; }
  inline void set_xGFM_relative_residual_threshold(const double& rel_residual_thresh)  { P4EST_ASSERT(rel_residual_thresh > 0.0);  xGFM_tolerance_on_rel_residual    = rel_residual_thresh; }

  inline void activate_xGFM_corrections(const bool& flag_, const bool& print_xGFM_residuals_and_corrections = false)
  {
    activate_xGFM = flag_;
    print_residuals_and_corrections_with_solve_info = activate_xGFM && print_xGFM_residuals_and_corrections;
    if(!activate_xGFM)
      max_iter = 0;
  }

  inline void set_validity_of_interface_neighbors_for_extrapolation(const bool& interface_neighbors_are_valid) { use_face_dofs_only_in_extrapolations = !interface_neighbors_are_valid; }

  inline bool uses_xGFM_corrections() const { return activate_xGFM; }

  inline Vec get_validation_jump() const { return validation_jump_u; }
  inline Vec get_validation_jump_mu_grad_u() const { return validation_jump_mu_grad_u; }

};

#endif // MY_P4EST_POISSON_JUMP_FACES_XGFM_H
