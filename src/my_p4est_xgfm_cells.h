#ifndef MY_P4EST_XGFM_CELLS_H
#define MY_P4EST_XGFM_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#else
#include <src/my_p4est_interface_manager.h>
#endif

const static double xgfm_threshold_cond_number_lsqr = 1.0e4;

class my_p4est_xgfm_cells_t
{
  // data related to the computational grid
  const my_p4est_cell_neighbors_t *cell_ngbd;
  const p4est_t                   *p4est;
  const p4est_ghost_t             *ghost;
  const p4est_nodes_t             *nodes;
  // computational domain parameters (fetched from the above objects at construction)
  const double *const xyz_min;
  const double *const xyz_max;
  const double *const tree_dimensions;
  const bool *const periodicity;
  // elementary computational grid parameters
  double dxyz_min[P4EST_DIM];
  inline double diag_min() const { return sqrt(SUMD(SQR(dxyz_min[0]), SQR(dxyz_min[1]), SQR(dxyz_min[2]))); }

  // this solver needs an interface manager (and may help build it)
  my_p4est_interface_manager_t* interface_manager;

  // equation parameters
  double mu_minus, mu_plus;
  double add_diag_minus, add_diag_plus;

  // Petsc vectors vectors of cell-centered values
  /* ---- NOT OWNED BY THE SOLVER ---- (hence not destroyed at solver's destruction) */
  Vec user_rhs_minus, user_rhs_plus;  // cell-sampled rhs of the continuum-level problem
  Vec jump_u, jump_normal_flux_u;     // node-sampled, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  my_p4est_interpolation_nodes_t* interp_jump_u; // we may need to interpolate the jumps pretty much anywhere
  inline bool interface_is_set()    const { return interface_manager != NULL; }
  inline bool jumps_have_been_set() const { return jump_u != NULL && jump_normal_flux_u != NULL; }
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction, except if returned before-hand) */
  Vec rhs;                  // cell-sampled, discretized rhs
  Vec residual;             // cell-sampled, residual residual = A*solution - rhs(jump_u, jump_normal_flux_u, extension_on_nodes)
  Vec solution;             // cell-sampled, sharp
  Vec extension_on_cells;   // cell-sampled, extension of interface-defined values
  Vec extension_on_nodes;   // node-sampled, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  Vec jump_flux;            // node-sampled, P4EST_DIM block-structure, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  /* ---- other PETSc objects ---- */
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  /* ---- Pointer to boundary condition (wall only) ---- */
  const BoundaryConditionsDIM *bc;
  /* ---- Control flags ---- */
  bool matrix_is_set, rhs_is_set;
  bool activate_xGFM;

  inline bool extend_negative_interface_values()      const { return mu_minus >= mu_plus; }
  inline bool mus_are_equal()                         const { return fabs(mu_minus - mu_plus) < EPS*MAX(fabs(mu_minus), fabs(mu_plus)); }
  inline bool diffusion_coefficients_have_been_set()  const { return mu_minus > 0.0 && mu_plus > 0.0; }
  inline double get_jump_in_mu()                      const { return (mu_plus - mu_minus); }

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

    bool reached_converged_within_desired_bounds(const double& absolute_accuracy_threshold, const double& tolerance_on_rel_residual) const
    {
      const size_t last_step_idx = last_step();
      return logger[last_step_idx].max_correction < absolute_accuracy_threshold && // the latest max_correction must be below the desired absolute accuracy requirement AND
          (relative_residual(last_step_idx) < tolerance_on_rel_residual || // either the latest relative residual is below the desired threshold as well OR
           (last_step_idx != 0 && fabs(relative_residual(last_step_idx) - relative_residual(last_step_idx - 1)) < 1.0e-6*MAX(relative_residual(last_step_idx), relative_residual(last_step_idx - 1)))); // or we have done at least two solves and we have reached a fixed-point for which the relative residual is above the desired threshold but can't really be made any smaller, apparently
    }
  } solver_monitor;

  // (possibly memorized) extension operator for interface-defined values
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

    inline double operator()(const double* extension_on_cells_p, // input for regular neighbor terms (same side of the interface)
                             const double* solution_p, const double* jump_u_p, const double* jump_flux_p, const my_p4est_xgfm_cells_t& solver, // input for evaluating interface-defined values
                             double& max_correction_in_band) const // inout control parameter
    {
      double increment = regular_terms(extension_on_cells_p);
      if(interface_terms.size() > 0)
      {
        const double& mu_this_side    = (in_positive_domain ? solver.mu_plus   : solver.mu_minus);
        const double& mu_across       = (in_positive_domain ? solver.mu_minus  : solver.mu_plus);
        const bool extending_positive_values = !solver.extend_negative_interface_values();
        for (size_t k = 0; k < interface_terms.size(); ++k)
          increment += interface_terms[k].weight*solver.interface_manager->GFM_interface_value_between_cells(quad_idx, interface_terms[k].neighbor_quad_idx_across, interface_terms[k].oriented_dir,
                                                                                                             mu_this_side, mu_across, in_positive_domain, extending_positive_values, solution_p, jump_u_p, jump_flux_p);
      }
      if(in_band)
        max_correction_in_band = MAX(fabs(increment), max_correction_in_band);

      return increment;
    }
  };
  std::vector<extension_increment_operator> pseudo_time_step_increment_operator;
  const extension_increment_operator& get_extension_increment_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double& control_band);
  bool extension_operators_are_stored_and_set;

  // memorized local interpolation operators
  std::vector<linear_combination_of_dof_t> local_interpolators;
  bool local_interpolators_are_stored_and_set;

  // disallow copy ctr and copy assignment
  my_p4est_xgfm_cells_t(const my_p4est_xgfm_cells_t& other);
  my_p4est_xgfm_cells_t& operator=(const my_p4est_xgfm_cells_t& other);

  // internal procedures
  void preallocate_matrix();
  void setup_linear_system();
  PetscErrorCode setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type, const double &tolerance_on_rel_residual) const;
  KSPConvergedReason solve_linear_system();
  bool solve_for_fixpoint_solution(Vec& former_solution);
  inline void reset_rhs()           { rhs_is_set = false;                 setup_linear_system(); }
  inline void reset_matrix()        { matrix_is_set = false;              setup_linear_system(); }
  inline void reset_linear_system() { matrix_is_set = rhs_is_set = false; setup_linear_system(); }

  inline p4est_gloidx_t compute_global_index(const p4est_locidx_t &quad_idx) const
  {
    if(quad_idx < p4est->local_num_quadrants)
      return p4est->global_first_quadrant[p4est->mpirank] + quad_idx;

    const p4est_quadrant_t *quad = p4est_const_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
    return p4est->global_first_quadrant[quad_find_ghost_owner(ghost, quad_idx - p4est->local_num_quadrants)] + quad->p.piggy3.local_num;
  }

  /*!
   * \brief interpolate_cell_field_at_local_interface_capturing_node computes the interpolation of a cell-sampled field
   * at a grid node of the interface-capturing grid. If the local interpolators are stored and set, they are used to
   * calculate the results, right away; otherwise the hardwork calculation is done and the interpolators are built and
   * stored internally, in order to shortcut subsequent local interpolation calls.
   * \param [in] node_idx     : local index of the (possibly subrefined) node where the interpolated value is desired
   * \param [in] cell_field_p : pointer to the cell-sampled data field to interpolate (sampled on the computational grid)
   * \return value of the interpolated field at the desired node.
   *
   * Details of implementation:
   * 1) the probed grid node exists on the computational grid (always the case if no subrefinement is used)
   * --> the (first-degree only) cell neighbors of the node are sought and least-square interpolation is used
   * (in case of locally uniform cell neighborhood, the results is identical to an arithmetic average between
   * the P4EST_CHILDREN neighbor cells)
   * 2) if the probed node does not exist on the computational grid (i.e. it is either a cell-, face- or edge-
   * center on the computational grid) then
   * 2a) if it is a cell-center on the computational grid, the corresponding cell-value is returned;
   * 2b) otherwise, the computational cells sharing the point are fetched (2 cells if face center, possibly 4 if
   * edge-center in 3D) and, if all those cells exist (i.e. no wall) and if they are all as fine as possible on
   * the computatinal grid, the arithmetic average is returned; otherwise, their cell neighbors in tranverse
   * Cartesian directions are fetched and least-square interpolation is used.
   */
  double interpolate_cell_field_at_local_interface_capturing_node(const p4est_locidx_t &node_idx, const my_p4est_node_neighbors_t& interface_capturing_ngbd_n, const double *cell_field_p);
  /*!
   * \brief interpolate_cell_extension_to_interface_capturing_nodes interpolates the cell-sampled field of the
   * appropriate interface-defined values, i.e., from extension_on_cells, to all nodes of the interface-capturing,
   * i.e., to extension_on_nodes. This function will do the hardwork on the very first call, but will
   * store the relevant interpolation data internally to shortcut the task thereafter and optimize execution.
   */
  void interpolate_cell_extension_to_interface_capturing_nodes();

  // using PDE extrapolation : uses the current solution results and jump conditions to extend the appropriate
  // interface-defined values in the normal directions, using ASLAM's PDE-based extrapolation on the cells
  void extend_interface_values(Vec &former_extension_on_cells, Vec &former_extension_on_nodes, const double& threshold = 1.0e-10, const uint& niter_max = 20);
  // updates the right-hand side terms for cells involving jump terms (after extension_on_nodes has been updated)
  void update_rhs_in_relevant_cells_only();
  void update_rhs_and_residual(Vec& former_rhs, Vec& former_residual);
  double set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension_on_cells, Vec former_extension_on_nodes,
                                                         Vec former_rhs, Vec former_residual);

  void compute_jumps_in_flux_components_at_all_interface_capturing_nodes() const;
  void compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only() const;
  void compute_jumps_in_flux_components_for_interface_capturing_node(const p4est_locidx_t& node_idx, double *jump_flux_p,
                                                                     const double *jump_u_p, const double *jump_normal_flux_p, const double *grad_phi_p,
                                                                     const double *extension_on_nodes_p) const;

  void initialize_extension_on_cells();
  void initialize_extension_on_cells_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                           const double* const &solution_p, double* const &extension_on_cells_p) const;

  void cell_TVD_extension_of_interface_values(const double& threshold, const uint& niter_max);

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double *user_rhs_minus_p, const double *user_rhs_plus_p,
                                     const double *jump_u_p, const double *jump_flux_p,
                                     double* rhs_p, int *nullspace_contains_constant_vector = NULL);

  inline void make_sure_solution_is_set()
  {
    if(solution == NULL)
      solve();
    P4EST_ASSERT(solution != NULL);
  }

  inline void make_sure_extensions_are_defined()
  {
    make_sure_solution_is_set();
    if(extension_on_cells == NULL)
    {
      P4EST_ASSERT(jump_flux != NULL);
      P4EST_ASSERT(extension_on_nodes == NULL); // those can't be set if extenson_on_cells is not set
      P4EST_ASSERT(!activate_xGFM || mus_are_equal()); // those are the (only) conditions under which the extension on cells can possibly be not defined
      extend_interface_values(extension_on_cells, extension_on_nodes);
    }
    P4EST_ASSERT(extension_on_cells != NULL && extension_on_nodes != NULL);
    return;
  }

  inline void make_sure_jumps_in_flux_are_defined(const bool& at_all_nodes)
  {
    if(activate_xGFM)
    {
      make_sure_solution_is_set(); // the (accurate) jumps in flux components are functions of the solution (through the extended interface-values) in case of xGFM so make sure it is defined, first
      P4EST_ASSERT(extension_on_nodes != NULL);
    }
    if(at_all_nodes)
      compute_jumps_in_flux_components_at_all_interface_capturing_nodes();
    else
      compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only();
    P4EST_ASSERT(jump_flux != NULL);
  }

  linear_combination_of_dof_t stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& oriented_dir,
                                                                            set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side) const;

  void get_flux_components_and_subtract_them_from_velocities_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, const double* solution_p,
                                                                   const double* jump_u_p, const double* jump_flux_p, const my_p4est_interpolation_nodes_t& interp_jump_flux,
                                                                   double* flux_dir_p, const double* vstar_dir_p, double* vnp1_plus_dir_p, double* vnp1_minus_dir_p);

public:

  my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  ~my_p4est_xgfm_cells_t();

  void set_interface(my_p4est_interface_manager_t* interface_manager_);

  /*!
   * \brief set_jumps sets the jump in solution and in its normal flux, sampled on the nodes of the interpolation_node_ngbd
   * of the interface manager (important if using subrefinement)
   * \param [in] jump_u            : node-sampled values of [u] = u^+ - u^-;
   * \param [in] jump_normal_flux  : node-sampled values of [mu*dot(n, grad u)] = mu^+*dot(n, grad u^+) - mu^-*dot(n, grad u^-).
   */
  void set_jumps(Vec jump_u_, Vec jump_normal_flux_u_);

  inline void set_bc(const BoundaryConditionsDIM& bc_)
  {
    bc = &bc_;
    // we can't really check for unchanged behavior in this cass, --> play it safe
    matrix_is_set = false;
    rhs_is_set    = false;
  }

  inline void set_mus(const double& mu_minus_, const double& mu_plus_)
  {
    const bool mus_unchanged = (fabs(mu_minus_ - mu_minus) < EPS*MAX(mu_minus_, mu_minus) && fabs(mu_plus_ - mu_plus) < EPS*MAX(mu_plus_, mu_plus));
    matrix_is_set = matrix_is_set && mus_unchanged;
    rhs_is_set    = rhs_is_set    && mus_unchanged;
    if(!mus_unchanged)
    {
      mu_minus = mu_minus_;
      mu_plus = mu_plus_;
    }
    P4EST_ASSERT(diffusion_coefficients_have_been_set()); // must be both strictly positive
  }

  inline void set_diagonals(const double& add_diag_minus_, const double& add_diag_plus_)
  {
    const bool diags_unchanged = (fabs(add_diag_minus_ - add_diag_minus) < EPS*MAX(add_diag_minus_, add_diag_minus) && fabs(add_diag_plus_ - add_diag_plus) < EPS*MAX(add_diag_plus_, add_diag_plus));
    matrix_is_set = matrix_is_set && diags_unchanged;
    rhs_is_set    = rhs_is_set    && diags_unchanged;
    if(!diags_unchanged)
    {
      add_diag_minus = add_diag_minus_;
      add_diag_plus = add_diag_plus_;
    }
  }

  inline void set_rhs(Vec user_sharp_rhs) { set_rhs(user_sharp_rhs, user_sharp_rhs); }
  inline void set_rhs(Vec user_rhs_minus_, Vec user_rhs_plus_)
  {
    P4EST_ASSERT(VecIsSetForCells(user_rhs_minus_, p4est, ghost, 1, false) && VecIsSetForCells(user_rhs_plus_, p4est, ghost, 1, false));
    user_rhs_minus  = user_rhs_minus_;
    user_rhs_plus   = user_rhs_plus_;
    rhs_is_set = false;
  }

  /* Benchmark tests revealed that PCHYPRE is MUCH faster than PCSOR as PCType!
   * The linear systme is supposed to be symmetric positive (semi-) definite, so KSPCG is ok as KSPType
   * Note: a low threshold for tolerance_on_rel_residual is critical to ensure accuracy in cases with large differences in diffusion coefficients!
   * */
  void solve(KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE, double absolute_accuracy_threshold = 1e-8, double tolerance_on_rel_residual = 1e-12);

  inline Vec get_extended_interface_values()                                { make_sure_extensions_are_defined();               return extension_on_cells;  }
  inline Vec get_extended_interface_values_on_interface_capturing_nodes()   { make_sure_extensions_are_defined();               return extension_on_nodes;  }
  inline Vec get_jump_in_flux(const bool& everywhere = true)                { make_sure_jumps_in_flux_are_defined(everywhere);  return jump_flux;           }
  inline Vec get_solution()                                                 { make_sure_solution_is_set();                      return solution;            }
  inline int get_number_of_xGFM_corrections()                         const { return solver_monitor.get_number_of_xGFM_corrections();                       }
  inline std::vector<PetscInt> get_numbers_of_ksp_iterations()        const { return solver_monitor.get_n_ksp_iterations();                                 }
  inline std::vector<double> get_max_corrections()                    const { return solver_monitor.get_max_corrections();                                  }
  inline std::vector<double> get_relative_residuals()                 const { return solver_monitor.get_relative_residuals();                               }
  inline bool is_using_xGFM()                                         const { return activate_xGFM;                                                         }
  inline bool get_matrix_has_nullspace()                              const { return A_null_space != NULL;                                                  }
  inline const p4est_t* get_p4est()                                   const { return p4est;                                                                 }
  inline const p4est_ghost_t* get_ghost()                             const { return ghost;                                                                 }
  inline const p4est_nodes_t* get_nodes()                             const { return nodes;                                                                 }
  inline const my_p4est_cell_neighbors_t* get_cell_ngbd()             const { return cell_ngbd;                                                             }
  inline const my_p4est_hierarchy_t* get_hierarchy()                  const { return cell_ngbd->get_hierarchy();                                            }
  inline const double* get_smallest_dxyz()                            const { return dxyz_min;                                                              }
  inline Vec get_jump()                                               const { return jump_u;                                                                }
  inline Vec get_jump_in_normal_flux()                                const { return jump_normal_flux_u;                                                    }
  inline my_p4est_interface_manager_t* get_interface_manager()        const { return interface_manager;                                                     }

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

  void get_flux_components_and_subtract_them_from_velocities(Vec flux[P4EST_DIM], my_p4est_faces_t *faces, Vec vstar[P4EST_DIM] = NULL, Vec vnp1_minus[P4EST_DIM] = NULL, Vec vnp1_plus[P4EST_DIM] = NULL);
  inline void get_flux_components(Vec flux[P4EST_DIM], my_p4est_faces_t* faces)
  {
    get_flux_components_and_subtract_them_from_velocities(flux, faces);
  }

  inline Vec return_ownership_of_solution()
  {
    make_sure_solution_is_set();
    Vec to_return = solution;
    solution = NULL; // will be handled by user from now on, hopefully!
    return to_return;
  }

  /*!
   * \brief set_initial_guess self-explanatory
   * \param initial_guess self-explanatory
   */
  inline void set_initial_guess(Vec initial_guess)
  {
    P4EST_ASSERT(VecIsSetForCells(initial_guess, p4est, ghost, 1));
    PetscErrorCode ierr;
    if(solution != NULL && !VecIsSetForCells(solution, p4est, ghost, 1)){
      ierr = VecDestroy(solution); CHKERRXX(ierr);
      solution = NULL;
    }
    if(solution == NULL){
      ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); }

    ierr = VecCopyGhost(initial_guess, solution); CHKERRXX(ierr); // set the solution to the given guess
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr); // activates initial guess
    return;
  }

  void inline activate_xGFM_corrections(const bool flag_) { activate_xGFM = flag_; }

};

#endif // MY_P4EST_XGFM_CELLS_H

