#ifndef MY_P4EST_POISSON_JUMP_FACES_H
#define MY_P4EST_POISSON_JUMP_FACES_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#else
#include <src/my_p4est_interface_manager.h>
#endif

class my_p4est_poisson_jump_faces_t
{
  friend class my_p4est_two_phase_flows_t;
protected:
  // data related to the computational grid
  const my_p4est_faces_t  *faces;
  const p4est_t           *p4est;
  const p4est_ghost_t     *ghost;
  const p4est_nodes_t     *nodes;
  // computational domain parameters (fetched from the above objects at construction)
  const double *const xyz_min;
  const double *const xyz_max;
  const double *const tree_dimensions;
  const bool *const periodicity;
  // elementary computational grid parameters
  double dxyz_min[P4EST_DIM];
  inline double diag_min() const { return sqrt(ABSD(dxyz_min[0], dxyz_min[1], dxyz_min[2])); }

  // this solver needs an interface manager (and may contribute to building some of its interface face-specific maps)
  my_p4est_interface_manager_t* interface_manager;

  // equation parameters
  double mu_minus, mu_plus;
  double add_diag_minus, add_diag_plus;

  // Petsc tolerance parameters
  PetscInt    max_ksp_iterations;
  PetscScalar relative_tolerance, absolute_tolerance, divergence_tolerance;

  // Petsc vectors of face-sampled values
  /* ---- NOT OWNED BY THE SOLVER ---- (hence not destroyed at solver's destruction) */
  // one needs to provide the rhs of the problem either
  // sharp, face-sampled value of the continuum rhs
  Vec *user_rhs_minus, *user_rhs_plus;  // pointers to P4EST_DIM face-sampled rhs of the continuum-level problem --> sharp, face-sampled value of the continuum value of f defining the rhs as diag*u[dir] - div(mu*grad(u[dir])) = f[dir]
  // for relevant applications (two-phase flows)
  Vec jump_u_dot_n, jump_tangential_stress; // node-sampled (resp. blocksize 1 and P4EST_DIM), defined on the nodes of the interpolation_node_ngbd of the interface manager (important if using subrefinement)
  inline bool interface_is_set() const { return interface_manager != NULL; }
  Vec *user_initial_guess; // pointers to P4EST_DIM face-sampled initial guesses

  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction) */
  Vec solution[P4EST_DIM]; // face-sampled, sharp
  Vec extrapolation_minus[P4EST_DIM], extrapolation_plus[P4EST_DIM]; // face-sampled, self-explanatory
  /*!
   * \brief extrapolation_operator_minus and extrapolation_operator_plus store the (upwind) operators
   * for evaluating the relevant "n_dot_grad" (along with the pseudo-time step dtau), pertaining to
   * the (appropriate) discretization of the pseudo-time PDE-based extrapolations.
   * For the face of orientation dim and local index face_idx, if the face center is in the negative
   * domain, extrapolation_operator_plus[dim][face_idx] will always be defined and stored therein;
   * extrapolation_operator_minus may or may not be --> it may be defined for faces close to the
   * interface without enough well-defined cartesian neighbors)
   * (defined and constructed in the "initialization" stage of the abstract extrapolation procedure!)
   */
  std::map<p4est_locidx_t, extrapolation_operator_t> extrapolation_operator_minus[P4EST_DIM], extrapolation_operator_plus[P4EST_DIM];
  // we may need to interpolate the jumps pretty much anywhere
  my_p4est_interpolation_nodes_t *interp_jump_u_dot_n, *interp_jump_tangential_stress;
  Mat matrix[P4EST_DIM];
  Vec sqrt_reciprocal_diagonal[P4EST_DIM];
  Vec my_own_nullspace_vector[P4EST_DIM];
  MatNullSpace null_space[P4EST_DIM];
  KSP ksp[P4EST_DIM];
  Vec rhs[P4EST_DIM]; // face-sampled, discretized rhs's of the linear systems to invert
  /* ---- Pointer to boundary condition (wall only) ---- */
  const BoundaryConditionsDIM *bc; // array of P4EST_DIM boundary conditions
  /* ---- Control flags ---- */
  bool matrix_is_set[P4EST_DIM], rhs_is_set[P4EST_DIM];
  bool scale_systems_by_diagonals;
  bool extrapolations_are_set;
  /* voronoi tesselation data */
  bool voronoi_on_the_fly, all_voronoi_cells_are_set[P4EST_DIM];
  // If voronoi_on_the_fly is true, voronoi_cell[dim] is of size 1 and voronoi_cell[dim][0] is the face being considered (of cartesian orientation 'dim')
  // Otherwise, voronoi_cell[dim] is of size faces->num_local(dim) and voronoi_cell[dim][face_idx] is the voronoi cell associated with the face of orientation dim and of local index face_idx
  vector<Voronoi_DIM> voronoi_cell[P4EST_DIM];

  inline bool mus_are_equal()                         const { return fabs(mu_minus - mu_plus) < EPS*MAX(fabs(mu_minus), fabs(mu_plus)); }
  inline bool diffusion_coefficients_have_been_set()  const { return mu_minus > 0.0 && mu_plus > 0.0; }
  inline double get_jump_in_mu()                      const { return (mu_plus - mu_minus); }

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_faces_t(const my_p4est_poisson_jump_faces_t& other);
  my_p4est_poisson_jump_faces_t& operator=(const my_p4est_poisson_jump_faces_t& other);

  // internal procedures
  virtual void get_numbers_of_faces_involved_in_equation_for_face(const u_char& dim, const p4est_locidx_t& face_idx,
                                                                  PetscInt& number_of_local_faces_involved, PetscInt& number_of_ghost_faces_involved) = 0;

  void preallocate_matrix(const u_char& dim);
  /*!
   * \brief setup_linear_solver sets the Krylov solver for the linear system of equations to be inverted
   * \param [in] dir      Cartesian dimension for the face of interest
   * \param [in] ksp_type solver type desired by the user
   *                      IMPORTANT NOTE: PetSc recommends using GMRES for singular problems
   *                      --> GMRES is enforces in that case and the provided ksp_type is irrelevant then (i.e. if null_space[dir] != NULL)
   * \param [in] pc_type  preconditioner type desired by the user
   * \return a PetscError code to check if anything went wrong
   */
  PetscErrorCode setup_linear_solver(const u_char& dim, const KSPType& ksp_type, const PCType& pc_type) const;
  void solve_linear_systems();
  inline void reset_rhs()
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      rhs_is_set[dim] = false;
      setup_linear_system(dim);
    }
  }
  inline void reset_matrices()
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      matrix_is_set[dim] = false;
      setup_linear_system(dim);
    }
  }
  inline void reset_linear_systems()
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      matrix_is_set[dim] = rhs_is_set[dim] = false;
      setup_linear_system(dim);
    }
  }

  // (for two-phase flows applications)
  // NOTE:
  // Most (in fact, ideally all) of your faces having a neighbor across the interface *SHOULD* be of type parallelepiped and
  // associated with the finest level of refinement available. If living in an ideal world, that could be enforced at grid updates
  // and one would make sure that that assumption is never invalidated. HOWEVER, such a grid check would required costly
  // inter-processor communications. So, one should try to make the solver compliant to slightly "nonuniform" types having one
  // neighbor face across the interface...
  // For instance,
  //
  // ___.2________3_____________4_________
  // |   .    |        |                 |
  // |    .   |        |                 |
  // |     .  |        |                 |
  // |___0_.__|___1____|                 |
  // |     .  |        |                 |
  // |     .  |        |                 |
  // |    .   |        |                 |
  // |   .    |        |                 |
  // ---.6--------7-------------5---------
  //
  // Face 1 here above has neighbor face 0 across the interface, but is not locally uniform. because of
  // neighbors 4 and 5. Guaranteed robustness (in the sense of alleviating random crashes) requires such
  // cases to be considered when building the discretization.
  virtual void build_discretization_for_face(const u_char& dim, const p4est_locidx_t& face_idx, int *nullspace_contains_constant_vector = NULL) = 0;
  void setup_linear_system(const u_char& dim);

  virtual void initialize_extrapolation_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                              double* extrapolation_minus_p[P4EST_DIM], double* extrapolation_plus_p[P4EST_DIM],
                                              double* normal_derivative_of_solution_minus_p[P4EST_DIM], double* normal_derivative_of_solution_plus_p[P4EST_DIM], const u_char& degree) = 0;

  void extrapolate_normal_derivatives_local(const u_char& dim, const p4est_locidx_t& face_idx,
                                            double* tmp_minus_p[P4EST_DIM], double* tmp_plus_p[P4EST_DIM],
                                            const double* normal_derivative_of_solution_minus_p[P4EST_DIM], const double* normal_derivative_of_solution_plus_p[P4EST_DIM]) const;

  virtual void extrapolate_solution_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                          double* tmp_minus_p[P4EST_DIM], double* tmp_plus_p[P4EST_DIM],
                                          const double* extrapolation_minus_p[P4EST_DIM], const double* extrapolation_plus_p[P4EST_DIM],
                                          const double* normal_derivative_of_solution_minus_p[P4EST_DIM], const double* normal_derivative_of_solution_plus_p[P4EST_DIM]) = 0;

  void pointwise_operation_with_sqrt_of_diag(const u_char& dim, size_t num_vectors, ...) const;


  inline const Voronoi_DIM& get_voronoi_cell(const p4est_locidx_t &face_idx, const u_char &dir)
  {
    Voronoi_DIM& my_cell = (voronoi_on_the_fly ? voronoi_cell[dir][0] : voronoi_cell[dir][face_idx]);
    if(!voronoi_on_the_fly && all_voronoi_cells_are_set[dir])
      return my_cell;

    compute_voronoi_cell(my_cell, faces, face_idx, dir, bc, NULL);
    P4EST_ASSERT(my_cell.get_type() != not_well_defined && my_cell.get_type() != unknown); // prohibited in this kind of applications: all Voronoi cell must be defined because all are used (in the current context, at least)...

#ifndef P4_TO_P8
    clip_voronoi_cell_by_parallel_walls(my_cell, dir); // if the face is a wall face with nonuniform neighborhood (for instance), the construction in 2D gaps over the border --> needs clipping
#endif
    return my_cell;
  }

#ifndef P4_TO_P8
  inline void clip_voronoi_cell_by_parallel_walls(Voronoi2D &voro_cell, const u_char &dir)
  {
    const vector<ngbd2Dseed> *points;
    vector<Point2> *partition;

    voro_cell.get_neighbor_seeds(points);
    voro_cell.get_partition(partition);

    /* clip the voronoi partition at the boundary of the domain */
    const u_char other_dir = (dir + 1)%P4EST_DIM;
    for(size_t m = 0; m < points->size(); m++)
      if((*points)[m].n < 0 &&  (-1 - (*points)[m].n)/2 == dir) // there is a "wall" point added because of a parallel wall
      {
        const char which_wall = (-1 - (*points)[m].n)%2;
        for(char i = -1; i < 1; ++i)
        {
          size_t k                        = mod(m + i                 , partition->size());
          size_t tmp                      = mod(k + (i == -1 ? -1 : 1), partition->size());
          Point2 segment                  = (*partition)[tmp] - (*partition)[k];
          double lambda                   = ((which_wall == 0 ? xyz_min[dir] : xyz_max[dir]) - (*partition)[k].xyz(dir))/segment.xyz(dir);
          (*partition)[k].xyz(dir)        = (which_wall == 0 ? xyz_min[dir] : xyz_max[dir]);
          (*partition)[k].xyz(other_dir)  = (*partition)[k].xyz(other_dir) + lambda*segment.xyz(other_dir);
        }
      }
    voro_cell.compute_volume();
  }
#endif

  inline bool face_is_dirichlet_wall(const p4est_locidx_t& face_idx, const u_char& dir) const
  {
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    faces->f2q(face_idx, dir, quad_idx, tree_idx);
    u_char loc_face_idx = (faces->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
    P4EST_ASSERT(faces->q2f(quad_idx, loc_face_idx) == face_idx);
    const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est, ghost);

    if(is_quad_Wall(p4est, tree_idx, quad, loc_face_idx))
    {
      double xyz_face[P4EST_DIM];
      faces->xyz_fr_f(face_idx, dir, xyz_face);
      return (bc[dir].wallType(xyz_face) == DIRICHLET);
    }

    return false;
  }

public:
  my_p4est_poisson_jump_faces_t(const my_p4est_faces_t *faces_, const p4est_nodes_t* nodes_);
  virtual ~my_p4est_poisson_jump_faces_t();

  void set_interface(my_p4est_interface_manager_t* interface_manager_);

  /*!
   * \brief set_jumps sets the jump(s) in the normal component of the solution and in the tangential component of
   * [mu (grad(u) + (grad(u)^{T}))\cdot n]
   * \param [in] jump_u_dot_n_            : node-sampled values of n \cdot [u];
   * \param [in] jump_tangential_stress_  : node-sampled values of (I - nn)\cdot[mu*(grad(u) + (grad(u)^{T}))\cdot n].
   */
  virtual void set_jumps(Vec jump_u_dot_n_, Vec jump_tangential_stress_);

  inline void set_tolerances(const double& rel_tol, const int& max_ksp_iter = PETSC_DEFAULT, const double& abs_tol = PETSC_DEFAULT, const double& div_tol = PETSC_DEFAULT)
  {
    relative_tolerance    = rel_tol;
    absolute_tolerance    = abs_tol;
    divergence_tolerance  = div_tol;
    max_ksp_iterations    = max_ksp_iter;
  }

  /*!
   * \brief set_bc sets the boundary conditions for the different components of the vector fields to be solved for
   * \param bc_ pointer to P4EST_DIM BoundaryConditionsDIM objects representing the boundary conditions for the
   * respective components of the vector field to be solved.
   * NOTE 1: only wall Dirichlet and Neumann are considered
   * NOTE 2: for Neumann wall boundary conditions, the prescribed wall value is assumed to be the normal _derivative_
   * of the solution (thus not the normal _flux_ of the solution)
   */
  inline void set_bc(const BoundaryConditionsDIM *bc_)
  {
    bc = bc_;
    // we can't really check for unchanged behavior in this cass, --> play it safe
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      matrix_is_set[dim]  = false;
      rhs_is_set[dim]     = false;
    }
  }

  inline void set_mus(const double& mu_minus_, const double& mu_plus_)
  {
    const bool mus_unchanged = (fabs(mu_minus_ - mu_minus) < EPS*MAX(mu_minus_, mu_minus) && fabs(mu_plus_ - mu_plus) < EPS*MAX(mu_plus_, mu_plus));
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      matrix_is_set[dim] = matrix_is_set[dim] && mus_unchanged;
      rhs_is_set[dim]    = rhs_is_set[dim]    && mus_unchanged;
    }
    if(!mus_unchanged)
    {
      mu_minus  = mu_minus_;
      mu_plus   = mu_plus_;
    }
    P4EST_ASSERT(diffusion_coefficients_have_been_set()); // must be both strictly positive
  }

  inline void set_diagonals(const double& add_diag_minus_, const double& add_diag_plus_)
  {
    const bool diags_unchanged = (fabs(add_diag_minus_ - add_diag_minus) < EPS*MAX(add_diag_minus_, add_diag_minus) && fabs(add_diag_plus_ - add_diag_plus) < EPS*MAX(add_diag_plus_, add_diag_plus));
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      matrix_is_set[dim] = matrix_is_set[dim] && diags_unchanged;

    if(!diags_unchanged)
    {
      add_diag_minus = add_diag_minus_;
      add_diag_plus = add_diag_plus_;
    }
  }

  inline void set_rhs(Vec* user_rhs_minus_, Vec* user_rhs_plus_)
  {
    P4EST_ASSERT(VecsAreSetForFaces(user_rhs_minus_, faces, 1, false) && VecsAreSetForFaces(user_rhs_plus_, faces, 1, false));
    user_rhs_minus  = user_rhs_minus_;
    user_rhs_plus   = user_rhs_plus_;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      rhs_is_set[dim] = false;
  }

  virtual void solve_for_sharp_solution(const KSPType &ksp_type, const PCType& pc_type) = 0;

  inline void solve(const KSPType& ksp_type, const PCType& pc_type = PCHYPRE, Vec* initial_guess_ = NULL)
  {
    P4EST_ASSERT(initial_guess_ == NULL || VecsAreSetForFaces(initial_guess_, faces, 1));
    PetscErrorCode ierr;
    user_initial_guess = initial_guess_;
    if(user_initial_guess != NULL)
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = KSPSetInitialGuessNonzero(ksp[dim], PETSC_TRUE); CHKERRXX(ierr); // activates initial guess
      }

    solve_for_sharp_solution(ksp_type, pc_type);
    return;
  }

  inline bool get_matrix_has_nullspace(const u_char& dim)       const { return null_space[dim] != NULL;               }
  inline const p4est_t* get_p4est()                             const { return p4est;                                 }
  inline const p4est_ghost_t* get_ghost()                       const { return ghost;                                 }
  inline const p4est_nodes_t* get_nodes()                       const { return nodes;                                 }
  inline const my_p4est_faces_t* get_faces()                    const { return faces;                                 }
  inline const my_p4est_cell_neighbors_t* get_cell_ngbd()       const { return faces->get_ngbd_c();                   }
  inline const my_p4est_hierarchy_t* get_hierarchy()            const { return faces->get_ngbd_c()->get_hierarchy();  }
  inline const double* get_smallest_dxyz()                      const { return dxyz_min;                              }
  inline Vec const* get_solution()                              const { return solution;                              }
  inline Vec const* get_rhs()                                   const { return rhs;                                   }
  inline Vec get_jump_u_dot_n()                                 const { return jump_u_dot_n;                          }
  inline Vec get_jump_in_tangential_stress()                    const { return jump_tangential_stress;                }
  inline my_p4est_interface_manager_t* get_interface_manager()  const { return interface_manager;                     }
  inline Vec const* get_extrapolated_solution_minus()           const { return extrapolation_minus;                   }
  inline Vec const* get_extrapolated_solution_plus()            const { return extrapolation_plus;                    }
  inline void return_ownership_of_extrapolations(Vec extrapolation_minus_out[P4EST_DIM], Vec extrapolation_plus_out[P4EST_DIM])
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      // we swap pointers to alleviate memory leaks if the "out" arguments are not NULL
      std::swap(extrapolation_minus_out[dim], extrapolation_minus[dim]);
      std::swap(extrapolation_plus_out[dim],  extrapolation_plus[dim]);
    }
    return;
  }

  void extrapolate_solution_from_either_side_to_the_other(const u_int& n_pseudo_time_iterations, const u_char& degree = 1);

  /*!
   * \brief set_scale_by_diagonal sets the internal flag for controlling the (symmetric) scaling of the linear
   * system by the diagonal. If set to true, the solver does not call the KSP solvers on
   *                                        A*x           = b,
   * but on
   *                    (D^{-1/2}*A*D^{-1/2})*(D^{1/2}*x) = (D^{-1/2}*b)
   * instead (every diagonal element in D^{-1/2}*A*D^{-1/2} is 1).
   * [NOTE:] we do not use the built-in Petsc function KSPSetDiagonalScale which does the very same job on paper
   * because that would not allow us to easily get the original rhs back (b instead of (D^{-1/2}*b)) without
   * using KSPSetDiagonalScaleFix which also "unscales" (D^{-1/2}*A*D^{-1/2}) back to A (costly operation)/
   * \param do_the_scaling [in] action desired by the user;
   */
  inline void set_scale_by_diagonal(const bool& do_the_scaling) { scale_systems_by_diagonals = do_the_scaling; }

  inline void set_compute_partition_on_the_fly(const bool& do_it_on_the_fly)
  {
    voronoi_on_the_fly = do_it_on_the_fly;
    if(voronoi_on_the_fly)
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      {
        all_voronoi_cells_are_set[dim] = false;
        voronoi_cell[dim].resize(1);
      }
    }
  }

  void print_partition_VTK(const char *file, const u_char &dir);

};

#endif // MY_P4EST_POISSON_JUMP_FACES_H
