#ifndef MY_P4EST_POISSON_JUMP_CELLS_H
#define MY_P4EST_POISSON_JUMP_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#else
#include <src/my_p4est_interface_manager.h>
#endif

class my_p4est_poisson_jump_cells_t
{
protected:
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

  // this solver needs an interface manager (and may contribute to building some of its interface cell-specific maps)
  my_p4est_interface_manager_t* interface_manager;

  // equation parameters
  double mu_minus, mu_plus;
  double add_diag_minus, add_diag_plus;

  // Petsc vectors vectors of cell-centered values
  /* ---- NOT OWNED BY THE SOLVER ---- (hence not destroyed at solver's destruction) */
  // one needs to provide the rhs of the problem either as
  // - sharp, cell-sampled value of the continuum rhs
  Vec user_rhs_minus, user_rhs_plus;        // cell-sampled rhs of the continuum-level problem --> sharp, cell-sampled value of the continuum value of f defining the rhs as diag*u - div(mu*grad(u)) = f
  // - or sharp, face-sampled values of the two-phase velocity field that needs to be made divergence-free
  Vec *user_vstar_minus, *user_vstar_plus;  // face-sampled rhs of the two-phase velocity field that needs to be made divergence-free --> sharp, face-sampled values of v_star defining the rhs as diag*u - div(mu*grad(u)) = -div(v_star)
  Vec jump_u, jump_normal_flux_u;     // node-sampled, defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  my_p4est_interpolation_nodes_t *interp_jump_u, *interp_jump_normal_flux; // we may need to interpolate the jumps pretty much anywhere
  inline bool interface_is_set()    const { return interface_manager != NULL; }
  inline bool jumps_have_been_set() const { return jump_u != NULL && jump_normal_flux_u != NULL; }
  Vec user_initial_guess;
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec solution;   // cell-sampled, sharp
  Vec rhs;        // cell-sampled, discretized rhs of the linear system to invert
  /* ---- other PETSc objects ---- */
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  /* ---- Pointer to boundary condition (wall only) ---- */
  const BoundaryConditionsDIM *bc;
  /* ---- Control flags ---- */
  bool matrix_is_set, rhs_is_set;

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
  PetscErrorCode setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type, const double &tolerance_on_rel_residual) const;
  void solve_linear_system();
  inline void reset_rhs()           { rhs_is_set = false;                 setup_linear_system(); }
  inline void reset_matrix()        { matrix_is_set = false;              setup_linear_system(); }
  inline void reset_linear_system() { matrix_is_set = rhs_is_set = false; setup_linear_system(); }

  inline p4est_gloidx_t compute_global_index(const p4est_locidx_t &quad_idx) const
  {
    return compute_global_index_of_quad(quad_idx, p4est, ghost);
  }

  linear_combination_of_dof_t stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& oriented_dir,
                                                                            set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side) const;

  virtual void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector = NULL) = 0;
  void setup_linear_system();


  virtual double get_sharp_flux_component_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, double& phi_face) const = 0;

public:

  my_p4est_poisson_jump_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_);
  virtual ~my_p4est_poisson_jump_cells_t();

  virtual void set_interface(my_p4est_interface_manager_t* interface_manager_);

  /*!
   * \brief set_jumps sets the jump in solution and in its normal flux, sampled on the nodes of the interpolation_node_ngbd
   * of the interface manager (important if using subrefinement)
   * \param [in] jump_u            : node-sampled values of [u] = u^+ - u^-;
   * \param [in] jump_normal_flux  : node-sampled values of [mu*dot(n, grad u)] = mu^+*dot(n, grad u^+) - mu^-*dot(n, grad u^-).
   */
  virtual void set_jumps(Vec jump_u_, Vec jump_normal_flux_u_);

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
    if(!diags_unchanged)
    {
      add_diag_minus = add_diag_minus_;
      add_diag_plus = add_diag_plus_;
    }
  }

  inline void set_rhs(Vec user_rhs_minus_, Vec user_rhs_plus_)
  {
    P4EST_ASSERT(VecIsSetForCells(user_rhs_minus_, p4est, ghost, 1, false) && VecIsSetForCells(user_rhs_plus_, p4est, ghost, 1, false));
    user_rhs_minus  = user_rhs_minus_;
    user_rhs_plus   = user_rhs_plus_;
    rhs_is_set = false;
  }

  virtual void solve_for_sharp_solution(const KSPType &ksp, const PCType& pc) = 0;
  inline void solve(const KSPType& ksp_type = KSPCG, const PCType& pc_type = PCHYPRE, Vec initial_guess_ = NULL)
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
  inline Vec get_jump()                                         const { return jump_u;                      }
  inline Vec get_jump_in_normal_flux()                          const { return jump_normal_flux_u;          }
  inline my_p4est_interface_manager_t* get_interface_manager()  const { return interface_manager;           }

  void get_sharp_flux_components_and_subtract_them_from_velocities(Vec sharp_flux[P4EST_DIM], my_p4est_faces_t *faces,
                                                             Vec vstar_minus[P4EST_DIM], Vec vstar_plus[P4EST_DIM], Vec sharp_vnp1[P4EST_DIM]) const;
  inline void get_sharp_flux_components(Vec flux[P4EST_DIM], my_p4est_faces_t* faces) const
  {
    get_sharp_flux_components_and_subtract_them_from_velocities(flux, faces, NULL, NULL, NULL);
  }
};

#endif // MY_P4EST_POISSON_JUMP_CELLS_H
