#ifndef MY_P4EST_POISSON_JUMP_CELLS_H
#define MY_P4EST_POISSON_JUMP_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#else
#include <src/my_p4est_interface_manager.h>
#endif

// struct of differential operators for capturing appropriate
// viscous terms in jump of normal stress in two-phase flows
struct differential_operators_on_face_sampled_field {
  double scaling;
  linear_combination_of_dof_t n_dot_grad_dot_n_operator[P4EST_DIM];
  linear_combination_of_dof_t div_term[P4EST_DIM];

  differential_operators_on_face_sampled_field() : scaling(1.0) {}

  inline double n_dot_grad_dot_n(const double *face_sampled_field_p[P4EST_DIM]) const
  {
    return (ANDD(face_sampled_field_p[0] != NULL, face_sampled_field_p[1] != NULL, face_sampled_field_p[2] != NULL) ?
        SUMD(n_dot_grad_dot_n_operator[0](face_sampled_field_p[0]), n_dot_grad_dot_n_operator[1](face_sampled_field_p[1]), n_dot_grad_dot_n_operator[2](face_sampled_field_p[2]))
        : 0.0);
  }

  inline double divergence(const double *face_sampled_field_p[P4EST_DIM]) const
  {
    return (ANDD(face_sampled_field_p[0] != NULL, face_sampled_field_p[1] != NULL, face_sampled_field_p[2] != NULL) ?
        SUMD(div_term[0](face_sampled_field_p[0]), div_term[1](face_sampled_field_p[1]), div_term[2](face_sampled_field_p[2]))
        : 0.0);
  }


  inline double jump_viscous_terms(const bool for_projection_steps,
                                   const double& dt_over_alpha, const double& shear_viscosity_plus, const double& shear_viscosity_minus,
                                   const double *face_velocity_plus_km1_p[P4EST_DIM], const double *face_velocity_minus_km1_p[P4EST_DIM],
                                   const double *face_velocity_plus_k_p[P4EST_DIM], const double *face_velocity_minus_k_p[P4EST_DIM]) const
  {
    if(!for_projection_steps)
      return scaling*(shear_viscosity_plus*n_dot_grad_dot_n(face_velocity_plus_km1_p) - shear_viscosity_minus*n_dot_grad_dot_n(face_velocity_minus_km1_p));
    else
    {
      double ans  = dt_over_alpha*shear_viscosity_plus  *(n_dot_grad_dot_n(face_velocity_plus_k_p)  - n_dot_grad_dot_n(face_velocity_plus_km1_p)  + divergence(face_velocity_plus_k_p));
      ans        -= dt_over_alpha*shear_viscosity_minus *(n_dot_grad_dot_n(face_velocity_minus_k_p) - n_dot_grad_dot_n(face_velocity_minus_km1_p) + divergence(face_velocity_minus_k_p));
      return scaling*ans;
    }
  }
};

class my_p4est_poisson_jump_cells_t
{
protected:
  // data related to the computational grid
  const my_p4est_cell_neighbors_t *cell_ngbd;
  const p4est_t       *p4est;
  const p4est_ghost_t *ghost;
  const p4est_nodes_t *nodes;
  // computational domain parameters (fetched from the above objects at construction)
  const double *const xyz_min;
  const double *const xyz_max;
  const double *const tree_dimensions;
  const bool *const periodicity;
  // elementary computational grid parameters
  double dxyz_min[P4EST_DIM];
  inline double diag_min() const { return sqrt(ABSD(dxyz_min[0], dxyz_min[1], dxyz_min[2])); }

  // this solver needs an interface manager (and may contribute to building some of its interface cell-specific maps)
  my_p4est_interface_manager_t* interface_manager;

  // equation parameters
  double mu_minus, mu_plus;
  double add_diag_minus, add_diag_plus;
  // for coupling with two-phase flow solver (evaluation of normal component of normal viscous stress balance across the interface)
  double shear_viscosity_minus, shear_viscosity_plus;
  Vec *face_velocity_minus_km1, *face_velocity_plus_km1; // required/used for jump in pressure guess AND for jump corrections in subsequent projection steps
  double dt_over_BDF_alpha;
  bool set_for_projection_steps;
  inline bool is_coupled_to_two_phase_flow() const
  {
    return (fabs(shear_viscosity_minus) > 0.0) && (fabs(shear_viscosity_plus) > 0.0);
  }

  // Petsc tolerance parameters
  PetscInt    max_ksp_iterations;
  PetscScalar relative_tolerance, absolute_tolerance, divergence_tolerance;

  // Petsc vectors of cell-sampled values
  /* ---- NOT OWNED BY THE SOLVER ---- (hence not destroyed at solver's destruction) */
  // one needs to provide the rhs of the problem either as
  // - sharp, cell-sampled value of the continuum rhs
  Vec user_rhs_minus, user_rhs_plus;        // cell-sampled rhs of the continuum-level problem --> sharp, cell-sampled value of the continuum value of f defining the rhs as diag*u - div(mu*grad(u)) = f
  // - or sharp, face-sampled values of the two-phase velocity field that needs to be made divergence-free
  Vec *face_velocity_minus, *face_velocity_plus;  // face-sampled rhs of the two-phase velocity field that needs to be made divergence-free --> sharp, face-sampled values of v_star defining the rhs as diag*u - div(mu*grad(u)) = -div(v_star)
  const CF_DIM* interp_jump_normal_velocity; // interpolator to the jump in normal velocity value --> considered to be 0.0 if not provided
  Vec jump_u, jump_normal_flux_u;     // node-sampled, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  inline bool interface_is_set()    const { return interface_manager != NULL; }
  Vec user_initial_guess;
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec solution;   // cell-sampled, sharp
  Vec extrapolation_minus, extrapolation_plus;  // cell-sampled, self-explanatory
  /*!
   * \brief extrapolation_operator_minus and extrapolation_operator_plus store the (upwind) operators
   * for evaluating the relevant "n_dot_grad" (along with the pseudo-time step dtau), pertaining to
   * the (appropriate) discretization of the pseudo-time PDE-based extrapolations.
   * For the quadrant of local index quad_idx, if the quad center is in the negative domain,
   * extrapolation_operator_plus will always be defined and
   * stored therein; extrapolation_operator_minus may or may not be --> it may be defined for
   * quadrants close to the interface without enough well-defined cartesian neighbors)
   * (defined and constructed in the "initialization" stage of the abstract extrapolation procedure!)
   */
  std::map<p4est_locidx_t, extrapolation_operator_t> extrapolation_operator_minus, extrapolation_operator_plus;
  Vec rhs; // cell-sampled, discretized rhs of the linear system to invert
  my_p4est_interpolation_nodes_t *interp_jump_u, *interp_jump_normal_flux; // we may need to interpolate the jumps pretty much anywhere
  /* ---- other PETSc objects ---- */
  Mat A;
  Vec sqrt_reciprocal_diagonal;
  Vec my_own_nullspace_vector;
  MatNullSpace A_null_space;
  KSP ksp;
  /* ---- Pointer to boundary condition (wall only) ---- */
  const BoundaryConditionsDIM *bc;
  /* ---- Control flags ---- */
  bool matrix_is_set, rhs_is_set;
  bool scale_system_by_diagonals;
  bool extrapolations_are_set;
  bool use_extrapolations_in_sharp_flux_calculations;

  inline bool mus_are_equal()                         const { return fabs(mu_minus - mu_plus) < EPS*MAX(fabs(mu_minus), fabs(mu_plus)); }
  inline bool diffusion_coefficients_have_been_set()  const { return mu_minus > 0.0 && mu_plus > 0.0; }
  inline double get_jump_in_mu()                      const { return (mu_plus - mu_minus); }

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_cells_t(const my_p4est_poisson_jump_cells_t& other);
  my_p4est_poisson_jump_cells_t& operator=(const my_p4est_poisson_jump_cells_t& other);

  // internal procedures
  virtual void get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                  PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const = 0;

  void preallocate_matrix();
  /*!
   * \brief setup_linear_solver sets the Krylov solver for the linear system of equations to be inverted
   * \param [in] ksp_type                   solver type desired by the user
   *                                        IMPORTANT NOTE: PetSc recommends using GMRES for singular problems
   *                                        --> GMRES is enforces in that case and the provided ksp_type is irrelevant then (i.e. if A_null_space != NULL)
   * \param [in] pc_type                    preconditioner type desired by the user
   * \return a PetscError code to check if anything went wrong
   */
  PetscErrorCode setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type) const;
  void solve_linear_system();
  inline void reset_rhs()           { rhs_is_set = false;                 setup_linear_system(); }
  inline void reset_matrix()        { matrix_is_set = false;              setup_linear_system(); }
  inline void reset_linear_system() { matrix_is_set = rhs_is_set = false; setup_linear_system(); }

  inline p4est_gloidx_t compute_global_index(const p4est_locidx_t &quad_idx) const
  {
    return compute_global_index_of_quad(quad_idx, p4est, ghost);
  }

  linear_combination_of_dof_t stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& oriented_dir,
                                                                            set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side,
                                                                            linear_combination_of_dof_t *vstar_on_face_for_stable_projection = NULL) const;

  virtual void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL) = 0;
  void setup_linear_system();


  virtual void local_projection_for_face(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces,
                                         double* flux_component_minus_p[P4EST_DIM], double* flux_component_plus_p[P4EST_DIM],
                                         double* face_velocity_minus_p[P4EST_DIM], double* face_velocity_plus_p[P4EST_DIM]) const = 0;

  virtual void initialize_extrapolation_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                              double* extrapolation_minus_p, double* extrapolation_plus_p,
                                              double* normal_derivative_of_solution_minus_p, double* normal_derivative_of_solution_plus_p, const u_char& degree) = 0;

  void extrapolate_normal_derivatives_local(const p4est_locidx_t& quad_idx,
                                            double* tmp_minus_p, double* tmp_plus_p,
                                            const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p) const;

  virtual void extrapolate_solution_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                          double* tmp_minus_p, double* tmp_plus_p,
                                          const double* extrapolation_minus_p, const double* extrapolation_plus_p,
                                          const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p) = 0;

  void pointwise_operation_with_sqrt_of_diag(size_t num_vectors, ...) const;

  virtual void clear_node_sampled_jumps();
  virtual void update_jump_terms_for_projection() = 0;

public:
  my_p4est_poisson_jump_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  virtual ~my_p4est_poisson_jump_cells_t();

  void set_interface(my_p4est_interface_manager_t* interface_manager_);

  /*!
   * \brief set_jumps sets the jump in solution and in its normal flux, sampled on the nodes of the interpolation_node_ngbd
   * of the interface manager (important if using subrefinement)
   * \param [in] jump_u            : node-sampled values of [u] = u^+ - u^-;
   * \param [in] jump_normal_flux  : node-sampled values of [mu*dot(n, grad u)] = mu^+*dot(n, grad u^+) - mu^-*dot(n, grad u^-).
   */
  virtual void set_jumps(Vec jump_u_, Vec jump_normal_flux_u_);

  inline void set_tolerances(const double& rel_tol, const int& max_ksp_iter = PETSC_DEFAULT, const double& abs_tol = PETSC_DEFAULT, const double& div_tol = PETSC_DEFAULT)
  {
    relative_tolerance    = rel_tol;
    absolute_tolerance    = abs_tol;
    divergence_tolerance  = div_tol;
    max_ksp_iterations    = max_ksp_iter;
  }

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

  inline void set_shear_viscosities(const double& shear_viscosity_minus_, const double& shear_viscosity_plus_)
  {
    shear_viscosity_minus = shear_viscosity_minus_;
    shear_viscosity_plus = shear_viscosity_plus_;
    P4EST_ASSERT(shear_viscosity_minus > 0.0 && shear_viscosity_plus_ > 0.0); // must be both strictly positive
  }

  inline void set_diagonals(const double& add_diag_minus_, const double& add_diag_plus_)
  {
    const bool diags_unchanged = (fabs(add_diag_minus_ - add_diag_minus) < EPS*MAX(add_diag_minus_, add_diag_minus) && fabs(add_diag_plus_ - add_diag_plus) < EPS*MAX(add_diag_plus_, add_diag_plus));
    matrix_is_set = matrix_is_set && diags_unchanged;
    if(!diags_unchanged)
    {
      add_diag_minus = add_diag_minus_;
      add_diag_plus = add_diag_plus_;
    }
  }

  inline void set_rhs(Vec user_rhs_minus_, Vec user_rhs_plus_)
  {
    P4EST_ASSERT((user_rhs_minus_ == NULL && user_rhs_plus_ == NULL)
                 || (VecIsSetForCells(user_rhs_minus_, p4est, ghost, 1, false) && VecIsSetForCells(user_rhs_plus_, p4est, ghost, 1, false)));
    user_rhs_minus  = user_rhs_minus_;
    user_rhs_plus   = user_rhs_plus_;
    rhs_is_set = false;
  }

  inline void set_initial_guess(Vec user_initial_guess_)
  {
    user_initial_guess = user_initial_guess_;
  }


  inline void set_for_subsequent_projection_steps(const double &dt_over_BDF_alpha_)
  {
    P4EST_ASSERT(is_coupled_to_two_phase_flow());
    dt_over_BDF_alpha = dt_over_BDF_alpha_;
    set_for_projection_steps = true;
    add_diag_minus = add_diag_plus = 0.0;
    clear_node_sampled_jumps();
    set_rhs(NULL, NULL);
    set_initial_guess(NULL);
    PetscErrorCode ierr;
    ierr = delete_and_nullify_vector(solution); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_minus); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_plus); CHKERRXX(ierr);
  }

  inline void set_face_velocities_km1(Vec *face_velocity_minus_km1_, Vec *face_velocity_plus_km1_)
  {
    P4EST_ASSERT(!interface_is_set() ||
                 (VecsAreSetForFaces(face_velocity_minus_km1_, interface_manager->get_faces(), 1) && VecsAreSetForFaces(face_velocity_plus_km1_, interface_manager->get_faces(), 1)));
    face_velocity_minus_km1 = face_velocity_minus_km1_;
    face_velocity_plus_km1  = face_velocity_plus_km1_;
  }

  inline void set_velocity_on_faces(Vec* face_velocity_minus_, Vec* face_velocity_plus_, const CF_DIM* interp_jump_normal_velocity_ = NULL)
  {
    P4EST_ASSERT(!interface_is_set() ||
                 (VecsAreSetForFaces(face_velocity_minus_, interface_manager->get_faces(), 1) && VecsAreSetForFaces(face_velocity_plus_, interface_manager->get_faces(), 1)));
    face_velocity_minus = face_velocity_minus_;
    face_velocity_plus  = face_velocity_plus_;
    interp_jump_normal_velocity = interp_jump_normal_velocity_;
    if(is_coupled_to_two_phase_flow() && set_for_projection_steps)
      update_jump_terms_for_projection();

    rhs_is_set = false; // but the matrix is fine...
  }

  virtual void solve_for_sharp_solution(const KSPType &ksp_type, const PCType& pc_type) = 0;
  inline void solve(const KSPType& ksp_type, const PCType& pc_type = PCHYPRE, Vec initial_guess_ = NULL)
  {
    P4EST_ASSERT(initial_guess_ == NULL || VecIsSetForCells(initial_guess_, p4est, ghost, 1));
    PetscErrorCode ierr;
    user_initial_guess = initial_guess_;
    if(user_initial_guess != NULL){ // activates initial guess
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr); }

    solve_for_sharp_solution(ksp_type, pc_type);
    return;
  }

  inline bool get_matrix_has_nullspace()                        const { return A_null_space != NULL;        }
  inline const p4est_t* get_p4est()                             const { return p4est;                       }
  inline const p4est_ghost_t* get_ghost()                       const { return ghost;                       }
  inline const p4est_nodes_t* get_nodes()                       const { return nodes;                       }
  inline const my_p4est_cell_neighbors_t* get_cell_ngbd()       const { return cell_ngbd;                   }
  inline const my_p4est_hierarchy_t* get_hierarchy()            const { return cell_ngbd->get_hierarchy();  }
  inline const double* get_smallest_dxyz()                      const { return dxyz_min;                    }
  inline Vec get_solution()                                     const { return solution;                    }
  inline Vec get_rhs()                                          const { return rhs;                         }
  inline Vec get_jump()                                         const { return jump_u;                      }
  inline Vec get_jump_in_normal_flux()                          const { return jump_normal_flux_u;          }
  inline my_p4est_interface_manager_t* get_interface_manager()  const { return interface_manager;           }
  inline Vec get_extrapolated_solution_minus()                  const { return extrapolation_minus;         }
  inline Vec get_extrapolated_solution_plus()                   const { return extrapolation_plus;          }

  inline Vec return_extrapolated_solution_minus()
  {
    Vec tmp = extrapolation_minus;
    PetscErrorCode ierr = delete_and_nullify_vector(extrapolation_minus); CHKERRXX(ierr);
    return tmp;
  }
  inline Vec return_extrapolated_solution_plus()
  {
    Vec tmp = extrapolation_plus;
    PetscErrorCode ierr = delete_and_nullify_vector(extrapolation_plus); CHKERRXX(ierr);
    return tmp;
  }

  /*!
   * \brief project_face_velocities calculates the sharp flux components of the solution (at faces of the
   * computational grids) and subtract them from the (corresponding) user-defined velocity components, if
   * they're provided --> this function executes the "two-phase projection step".
   * \param [in] faces      : pointer to the face data structure for the computational grid under consideration
   * \param [in] flux_minus : (optional) array of P4EST_DIM parallel vectors for face-sampled vector fields
   * \param [in] flux_plus  : (optional) array of P4EST_DIM parallel vectors for face-sampled vector fields
   * (A) If the solver is aware of face-sampled velocity components (and used them in order to define and determine
   * the divergence-free projection problem to solve) this function will operate on the user-provided velocity and
   * subtract the relevant flux components;
   * (B) If flux_minus and flux_plus are NULL, the actual flux components are _not_ returned to the user in any way
   * (C) If flux_minus and flux_plus are not NULL but equal (thus pointing to the same array), the local sharp flux
   * components are returned (e.g. if phi_face <= 0.0, only the negative flux component is returned)
   * (D) If flux_minus and flux_plus are not NULL and pointing to separate arrays, the corresponding flux components
   * are returned in the corresponding arrays wherever they can be evaluated (the value is unmodified otherwise)
   *
   * If the solver is aware of the extrapolated solutions from either side of the domain, it will give them precedence
   * in the calculation of the flux components and either (sharp) flux component can then be evaluated everywhere in
   * the domain (and the projection is actually executed for extrapolated velocity components as well, if the user has
   * set the velocity components).
   * Otherwise, only the sharp solution and the jump conditions are used to determine the sharp flux components and,
   * therefore, the flux components from one side cannot be determine far in the other side. In such a case, the
   * velocity components are set to DBL_MAX (if known to the solver) as a way to tag them "unknown" and prevent their
   * use thereafter...
   */
  void project_face_velocities(const my_p4est_faces_t *faces, Vec* flux_minus = NULL, Vec* flux_plus = NULL);
  /*!
   * \brief get_sharp_flux_components returns the local sharp flux component corresponding to the local sign of the
   * face (only the sharp locally defined flux component)
   * --> more details in project_face_velocities
   */
  inline void get_sharp_flux_components(Vec flux[P4EST_DIM], const my_p4est_faces_t* faces)
  {
    project_face_velocities(faces, flux, flux);
  }
  /*!
   * \brief get_flux_components returns the flux components from either side wherever they can be evaluated
   * --> more details in project_face_velocities
   */
  inline void get_flux_components(Vec flux_minus[P4EST_DIM], Vec flux_plus[P4EST_DIM], const my_p4est_faces_t* faces)
  {
    project_face_velocities(faces, flux_minus, flux_plus);
  }

  inline void extrapolate_before_calculating_fluxes(const bool& desired_action) { use_extrapolations_in_sharp_flux_calculations = desired_action; }

  void extrapolate_solution_from_either_side_to_the_other(const u_int& n_pseudo_time_iterations, const u_char& degree = 1);

  virtual double get_sharp_integral_solution() const = 0;

  virtual void print_solve_info() const = 0;

  /*!
   * \brief set_scale_by_diagonal sets the internal flag for controlling the (symmetric) scaling of the linear
   * system by the diagonal. If set to true, the solver does not call the KSP solver on
   *                                        A*x           = b,
   * but on
   *                    (D^{-1/2}*A*D^{-1/2})*(D^{1/2}*x) = (D^{-1/2}*b)
   * instead (every diagonal element in D^{-1/2}*A*D^{-1/2} is 1).
   * [NOTE:] we do not use the built-in Petsc function KSPSetDiagonalScale which does the very same job on paper
   * because that would not allow us to easily get the original rhs back (b instead of (D^{-1/2}*b)) without
   * using KSPSetDiagonalScaleFix which also "unscales" (D^{-1/2}*A*D^{-1/2}) back to A (costly operation)/
   * \param do_the_scaling [in] action desired by the user;
   */
  inline void set_scale_by_diagonal(const bool& do_the_scaling) { scale_system_by_diagonals = do_the_scaling; }
};

#endif // MY_P4EST_POISSON_JUMP_CELLS_H
