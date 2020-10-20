#ifndef MY_P4EST_POISSON_JUMP_FACES_XGFM_H
#define MY_P4EST_POISSON_JUMP_FACES_XGFM_H

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_jump_faces.h>
#else
#include <src/my_p4est_poisson_jump_faces.h>
#endif

typedef struct {
  double jump;
  double known_jump_flux;
  linear_combination_of_dof_t xgfm_jump_flux_correction[P4EST_DIM];
  inline double jump_flux_component(const double* extension_p[P4EST_DIM] = NULL) const
  {
    return known_jump_flux + (extension_p != NULL  ? SUMD(xgfm_jump_flux_correction[0](extension_p[0]), xgfm_jump_flux_correction[1](extension_p[1]), xgfm_jump_flux_correction[2](extension_p[2])) : 0.0);
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

  inline bool extend_negative_interface_values() const { return mu_minus >= mu_plus; }

  // Memorized jump information for interface-point between quadrants
  map_of_vector_field_component_xgfm_jumps_t xgfm_jump_between_faces[P4EST_DIM];
  void build_xgfm_jump_flux_correction_operators_at_point(const double* xyz, const double* normal,
                                                          const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& flux_orientation) const;
  const vector_field_component_xgfm_jump& get_xgfm_jump_between_faces(const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_faces_xgfm_t(const my_p4est_poisson_jump_faces_xgfm_t& other);
  my_p4est_poisson_jump_faces_xgfm_t& operator=(const my_p4est_poisson_jump_faces_xgfm_t& other);

  // preallocation-related
  void get_numbers_of_faces_involved_in_equation_for_face(const u_char& dir, const p4est_locidx_t& face_idx,
                                                          PetscInt& number_of_local_faces_involved, PetscInt& number_of_ghost_faces_involved);

  // methods involved in the xgfm iterative procedure, minimizing the residual
  /*!
   * \brief update_solution saves the currently known solution and solves the current linear system with the best available initial guess
   * \param [out] former_solution   : solution as known by the solver before this call
   * \return true if there was indeed an update in the solution (otherwise, it means that a fix-point was found)
   */
  bool update_solution(Vec former_solution[P4EST_DIM]);

  /*!
   * \brief update_extension_of_interface_values computes the new extension of relevant interface-defined values, which are function of the current
   * solution and jump conditions (which depend on the currently known extension). Then the currently known extension is saved and return to calling
   * procedure via "former_extension" and the internal "extension" is updated with the newly calculated results.
   * The extension is Aslam's PDE extendion, with first order in pseudo-time forward Euler integration and subcell resolution close to the interface.
   * The derivatives of the face-sampled extension are evaluated consistently with the derivatives defined for stable projection operators, away from
   * the interface. (The result of the extension must be a linear function of the interface-defined values to ensure convergence of the xgfm iterative
   * procedure)
   * \param [out] former_extension  : extension of relevant interface-defined values, as known by the solver before this call
   * \param [in] threshold          : [optional] absolute threshold value to abort stop the pseudo-time procedure: if the interface-extended values do
   *                                  not change by more than this threshold over one pseudo-time step in a band of 3 diag from the interface, it is
   *                                  assumed that convergence is reached. Default value is 1e-10
   * \param [in] niter_max          : [optional] maximum number of pseudo time steps for the extension. Default value is 20.
   */
  void update_extension_of_interface_values(Vec former_extension[P4EST_DIM], const double& threshold = 1.0e-10, const uint& niter_max = 20);

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
   * \param [in] former_solution  : solution as known by the solver before the fixpoint update
   * \param [in] former_extension : extension of the relevant interface-defined values, as known by the solver before the fixpoint update
   * \param [in] former_rhs       : discretized rhs, as known by the solver before the fixpoint update
   * \param [in] former_residual  : (fixpoint) residual of the targeted jump problem, as known by the solver before the fixpoint update
   * \return the maximum of the absolute value of the correction to "former_solution" defining the new solver's solution (if a
   * former_solution was provided, 0.0 otherwise)
   */
  double set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution[P4EST_DIM], Vec former_extension[P4EST_DIM], Vec former_rhs[P4EST_DIM], Vec former_residual[P4EST_DIM]);

  void initialize_extension(Vec face_sampled_extension[P4EST_DIM]);
  void initialize_extension_local(const u_char& dir, const p4est_locidx_t& face_idx,
                                  const double* solution_p[P4EST_DIM], double* extension_p[P4EST_DIM]) const;

  void build_discretization_for_face(const u_char& dir, const p4est_locidx_t& face_idx, int *nullspace_contains_constant_vector = NULL);

  inline void make_sure_extensions_are_defined()
  {
    if(ANDD(solution[0], solution[1], solution[2]))
      solve_for_sharp_solution();
    if(ANDD(extension[0] == NULL, extension[1] == NULL, extension[2] == NULL))
    {
      P4EST_ASSERT(!activate_xGFM || mus_are_equal()); // those are the (only) conditions under which the extension on cells can possibly be not defined
      Vec dummy[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
      update_extension_of_interface_values(dummy); // argument is dummy in that case...
    }
    return;
  }

  void initialize_extrapolation_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                      double* extrapolation_minus_p[P4EST_DIM], double* extrapolation_plus_p[P4EST_DIM],
                                      double* normal_derivative_of_solution_minus_p[P4EST_DIM], double* normal_derivative_of_solution_plus_p[P4EST_DIM], const u_char& degree);

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

    extrapolations_are_set = false;
    set_for_testing_backbone = true;

    return;
  }

  inline Vec* get_extended_interface_values()                         { make_sure_extensions_are_defined();     return extension; }
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
  }

  inline void set_validity_of_interface_neighbors_for_extrapolation(const bool& interface_neighbors_are_valid) { use_face_dofs_only_in_extrapolations = !interface_neighbors_are_valid; }

  inline bool uses_xGFM_corrections() const { return activate_xGFM; }

  inline Vec get_validation_jump() const { return validation_jump_u; }

};

#endif // MY_P4EST_POISSON_JUMP_FACES_XGFM_H
