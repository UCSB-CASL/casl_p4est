#ifndef MY_P4EST_XGFM_CELLS_H
#define MY_P4EST_XGFM_CELLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_faces.h>
#include <p8est_nodes.h>
#include <src/my_p8est_solve_lsqr.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_faces.h>
#include <p4est_nodes.h>
#include <src/my_p4est_solve_lsqr.h>
#endif

#include <src/matrix.h>
#include <algorithm>
#include <map>

const static double xgfm_threshold_cond_number_lsqr = 1.0e4;
static const double value_not_needed = NAN;

class my_p4est_xgfm_cells_t
{
  // data related to the computational grid
  const my_p4est_cell_neighbors_t *cell_ngbd;
  const my_p4est_node_neighbors_t *node_ngbd;
#ifdef DEBUG
  p4est_t             *p4est; // I loose the const qualifier on this one in DEBUG because some of p4est's debug check functions can't take const p4est objects in...
#else
  const p4est_t       *p4est;
#endif
  const p4est_nodes_t *nodes;
  const p4est_ghost_t *ghost;
  // computational domain parameters
  const double *const xyz_min;
  const double *const xyz_max;
  const double *const tree_dimensions;
  const bool *const periodicity;
  // elementary computational grid parameters
  double dxyz_min[P4EST_DIM];
  inline double diag_min() const { return sqrt(SUMD(SQR(dxyz_min[0]), SQR(dxyz_min[1]), SQR(dxyz_min[2]))); }

  // equation parameters
  double mu_m, mu_p, add_diag_m, add_diag_p;

#ifdef WITH_SUBREFINEMENT
  // data related to the (subrefined) interface-capturing grid, if present
  const p4est_t                   *fine_p4est;
  const p4est_nodes_t             *fine_nodes;
  const p4est_ghost_t             *fine_ghost;
  const my_p4est_node_neighbors_t *fine_node_ngbd;
  // elementary interface-capturing grid parameters
  double dxyz_min_fine[P4EST_DIM];
#endif

  // Petsc vectors vectors of cell-centered values
  /* ---- NOT OWNED BY THE SOLVER ---- (hence not destroyed at solver's destruction) */
  Vec user_rhs;                   // cell-sampled rhs of the continuum-level problem
#ifdef WITH_SUBREFINEMENT
  Vec phi, normals, phi_xxyyzz;   // node-sampled on fine nodes, if using subrefinement
  Vec jump_u, jump_normal_flux_u; // node-sampled on fine nodes, if using subrefinement
  inline bool levelset_has_been_set() const { return phi != NULL; }
  inline bool normals_have_been_set() const { return normals != NULL; }
  inline bool jumps_have_been_set() const   { return jump_u != NULL && jump_normal_flux_u != NULL; }
  my_p4est_interpolation_nodes_t interp_subrefined_phi, interp_subrefined_normals, interp_subrefined_jump_u;
#else
  const my_p4est_interpolation_nodes_t *interp_phi, *interp_normals;
  const my_p4est_interpolation_nodes_t *interp_jump_u, *interp_jump_normal_flux_u;
  inline bool levelset_has_been_set() const { return interp_phi != NULL; }
  inline bool normals_have_been_set() const { return interp_normals != NULL; }
  inline bool jumps_have_been_set() const   { return interp_jump_u != NULL && interp_jump_normal_flux_u != NULL; }
#endif
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction, except if returned before-hand) */
  Vec rhs;                  // cell-sampled, discretized rhs
  Vec residual;             // cell-sampled, vector of the residual r_k = A*solution_k - rhs(jump_u, jump_normal_flux_u, extension_on_nodes_k)
  Vec solution;             // cell-sampled
  Vec extension_on_cells;   // cell-sampled
  Vec extension_on_nodes;   // node-sampled (fine nodes if subrefined)
  Vec jump_flux;            // node-sampled, P4EST_DIM block-structure (fine nodes if subrefined)
  /* ---- other PETSc objects ---- */
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;

  inline bool mu_m_is_larger()                        const { return mu_m >= mu_p; }
  inline bool mus_are_equal()                         const { return fabs(mu_m - mu_p) < EPS*MAX(fabs(mu_m), fabs(mu_p)); }
  inline bool diffusion_coefficients_have_been_set()  const { return mu_m > 0.0 && mu_p > 0.0; }
  inline double get_smaller_mu()                      const { return (mu_m_is_larger() ? mu_p : mu_m); }
  inline double get_jump_in_mu()                      const { return (mu_p - mu_m); }

  const BoundaryConditionsDIM *bc;

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
      return logger[last_step()].max_correction < absolute_accuracy_threshold && // the latest max_correction must be below the desired absolute accuracy requirement AND
          (relative_residual(last_step()) < tolerance_on_rel_residual || // either the latest relative residual is below the desired threshold as well OR
           (last_step() != 0 && fabs(relative_residual(last_step()) - relative_residual(last_step() - 1)) < 1.0e-6*MAX(relative_residual(last_step()), relative_residual(last_step() - 1)))); // or we have done at least two solves and we have reached a fixed-point for which the relative residual is above the desired threshold but can't really be made any smaller, apparently
    }
  } solver_monitor;

  // flags
  bool matrix_is_set, rhs_is_set;
  const bool activate_xGFM;

  /*!
   * \brief The interface_neighbor struct contains all relevant data regarding interface-neighbor,
   * i.e., intersection between the interface and the segment joining the current cell of interest
   * and its neighbor cell, on the computational grid.
   * phi_q    : value of the level-set at the center of the cell of interest;
   * phi_nb   : value of the level-set at the center of the neighbor cell (i.e. across the interface);
   * theta    : fraction of the grid spacing covered by the domain in which the cell of interest is;
   * neighbor_quad_idx      : local index of the neighbor cell in the computational grid (across the interface);
   * mid_point_fine_node_idx: local index of the grid node in between those two cells on the
   *                          interface-capturing grid (if using subrefinement);
   * quad_fine_node_idx     : local index of the grid node that coincides with the center of the cell
   *                          of interest, on the interface-capturing grid (if using subrefinement);
   * tmp_fine_node_idx      : local index of the grid node that coincides with the center of the neighbor
   *                          cell across the interface, on the interface-capturing grid (if using subrefinement).
   */
  struct interface_neighbor
  {
    double  phi_q;
    double  phi_nb;
    double  theta;
    p4est_locidx_t neighbor_quad_idx;
#ifdef WITH_SUBREFINEMENT
    p4est_locidx_t mid_point_fine_node_idx;
    p4est_locidx_t quad_fine_node_idx;
    p4est_locidx_t nb_fine_node_idx;
#endif
#ifdef DEBUG
    inline bool is_consistent_with_neighbor_across(const interface_neighbor& nb_across) const
    {
      return ((phi_q > 0.0) != (nb_across.phi_q > 0.0))
          && fabs(phi_q - nb_across.phi_nb) < EPS*MAX(fabs(phi_q), fabs(phi_nb))
          && fabs(phi_nb - nb_across.phi_q) < EPS*MAX(fabs(phi_q), fabs(phi_nb))
          && fabs(theta + nb_across.theta - 1.0) < EPS;
    }
#endif

    inline void get_GFM_jump_data(double& jump_field, double& jump_flux_component, const double* jump_p, const double* jump_flux_p, const u_char& dir) const
    {
      P4EST_ASSERT(mid_point_fine_node_idx >= 0);
      const bool past_mid_point = theta >= 0.5;
      const double theta_between_fine_nodes     = 2.0*theta - (past_mid_point ? 1.0 : 0.0);
      const p4est_locidx_t &fine_node_this_side = (past_mid_point ? mid_point_fine_node_idx  : quad_fine_node_idx);
      const p4est_locidx_t &fine_node_across    = (past_mid_point ? nb_fine_node_idx         : mid_point_fine_node_idx);
      jump_field          = theta_between_fine_nodes*jump_p[fine_node_across]                         + (1.0 - theta_between_fine_nodes)*jump_p[fine_node_this_side];
      jump_flux_component = theta_between_fine_nodes*jump_flux_p[P4EST_DIM*fine_node_across + dir/2]  + (1.0 - theta_between_fine_nodes)*jump_flux_p[P4EST_DIM*fine_node_this_side + dir/2];
      return;
    }

    inline double GFM_mu_tilde(const double& mu_this_side, const double& mu_across) const
    {
      return (1.0 - theta)*mu_this_side + theta*mu_across;
    }

    inline double GFM_mu_jump(const double& mu_this_side, const double& mu_across) const
    {
      return mu_this_side*mu_across/GFM_mu_tilde(mu_this_side, mu_across);
    }

    inline double GFM_jump_terms_for_flux_component(const double& mu_this_side, const double& mu_across, const u_char& dir, const bool &this_side_is_in_positive_domain,
                                                    const double* jump_p, const double* jump_flux_p,
                                                    const double* dxyz, const bool& evaluate_flux_on_this_side) const
    {
      double jump_field, jump_flux_component;
      get_GFM_jump_data(jump_field, jump_flux_component, jump_p, jump_flux_p, dir);

      return GFM_mu_jump(mu_this_side, mu_across)*(this_side_is_in_positive_domain ? +1.0 : -1.0)*
          (jump_flux_component*(evaluate_flux_on_this_side ? (1 - theta)/mu_across : -theta/mu_this_side) + (dir%2 == 1 ? +1.0 : -1.0)*jump_field/dxyz[dir/2]);
    }

    inline double GFM_flux_component(const double& mu_this_side, const double& mu_across, const u_char& dir, const bool &this_side_is_in_positive_domain,
                                     const double& solution_this_side, const double& solution_across,
                                     const double* jump_p, const double* jump_flux_p,
                                     const double* dxyz, const bool& evaluate_flux_on_this_side) const
    {
      return (dir%2 == 1 ? +1.0 : -1.0)*GFM_mu_jump(mu_this_side, mu_across)*(solution_across - solution_this_side)/dxyz[dir/2]
          + GFM_jump_terms_for_flux_component(mu_this_side, mu_across, dir, this_side_is_in_positive_domain, jump_p, jump_flux_p, dxyz, evaluate_flux_on_this_side);
    }

    inline double GFM_interface_defined_value(const double& mu_this_side, const double& mu_across, const u_char& dir, const bool &this_side_is_in_positive_domain, const bool &extending_positive_interface_values,
                                              const double& solution_this_side, const double& solution_across,
                                              const double* jump_p, const double* jump_flux_p,
                                              const double* dxyz) const
    {
      double jump_field, jump_flux_component;
      get_GFM_jump_data(jump_field, jump_flux_component, jump_p, jump_flux_p, dir);

      return ((1.0 - theta)*mu_this_side*(solution_this_side  + (this_side_is_in_positive_domain != extending_positive_interface_values ? (this_side_is_in_positive_domain ? -1.0 : +1.0)*jump_field : 0.0))
              +      theta *mu_across   *(solution_across     + (this_side_is_in_positive_domain == extending_positive_interface_values ? (this_side_is_in_positive_domain ? +1.0 : -1.0)*jump_field : 0.0))
              + (this_side_is_in_positive_domain ? +1.0 : -1.0)*(dir%2 == 1 ? +1.0 : -1.0)*theta*(1.0 - theta)*dxyz[dir/2]*jump_flux_component)/GFM_mu_tilde(mu_this_side, mu_across);
    }
  };

  class interface_manager_t {
    my_p4est_xgfm_cells_t& solver;

    struct which_interface_neighbor_t
    {
      p4est_locidx_t loc_idx;
      u_char dir;

      inline bool operator==(const which_interface_neighbor_t& other) const { return (this->loc_idx == other.loc_idx && this->dir == other.dir); } // equality comparator
#if __cplusplus < 201103L
      inline bool operator<(const which_interface_neighbor_t& other) const { return (this->loc_idx < other.loc_idx || (this->loc_idx == other.loc_idx && this->dir < other.dir)); } // comparison operator for storing in ordered map
#endif
    };

#if __cplusplus >= 201103L
    struct hash_functor{
      size_t operator()(const which_interface_neighbor_t& key) const { return P4EST_DIM*key.loc_idx + key.dir; } // hash value for unordered map keys
    };
    typedef std::unordered_map<which_interface_neighbor_t, interface_neighbor, hash_functor> map_of_interface_neighbors_t;
#else
    typedef std::map<which_interface_neighbor_t, interface_neighbor> map_of_interface_neighbors_t;
#endif
    map_of_interface_neighbors_t interface_data;
    map_of_interface_neighbors_t::const_iterator current_interface_data;

    inline map_of_interface_neighbors_t::const_iterator find_interface_neighbor_in_map(const p4est_locidx_t& quad_idx, const u_char& dir) const
    {
      P4EST_ASSERT(0 <= quad_idx && quad_idx < solver.p4est->local_num_quadrants && dir < P4EST_FACES);
      const which_interface_neighbor_t which_one = {quad_idx, dir};
      return interface_data.find(which_one);
    }

    void clear() {
      interface_data.clear();
      current_interface_data = interface_data.end();
    }

    inline bool current_interface_point_is_set_for(const p4est_locidx_t& quad_idx, const u_char& dir) const
    {
      const which_interface_neighbor_t which_one = {quad_idx, dir};
      return (current_interface_data != interface_data.end() && current_interface_data->first == which_one);
    }

    void set_current_interface_point_for(const p4est_locidx_t& quad_idx, const u_char& dir);

  public:
    interface_manager_t(my_p4est_xgfm_cells_t& parent_solver) : solver(parent_solver) { clear(); }

    void update_jumps_in_flux_at_all_relevant_nodes() const;
    void update_rhs_in_relevant_cells_only() const;

    const interface_neighbor get_interface_neighbor(const p4est_locidx_t& quad_idx, const u_char& dir);

    inline double GFM_mu_jump(const p4est_locidx_t& quad_idx, const u_char& dir, const double& mu_this_side, const double& mu_across)
    {
      if(!current_interface_point_is_set_for(quad_idx, dir))
        set_current_interface_point_for(quad_idx, dir);
      return current_interface_data->second.GFM_mu_jump(mu_this_side, mu_across);
    }

    inline double GFM_jump_terms_for_flux_component(const p4est_locidx_t& quad_idx, const u_char& dir,
                                                    const double& mu_this_side, const double& mu_across, const bool& in_positive_domain,
                                                    const double* jump_field_p, const double* jump_flux_p)
    {
      if(!current_interface_point_is_set_for(quad_idx, dir))
        set_current_interface_point_for(quad_idx, dir);
      return current_interface_data->second.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, dir, in_positive_domain, jump_field_p, jump_flux_p, solver.dxyz_min, true);
    }

    inline double interface_value(const p4est_locidx_t& quad_idx, const u_char& dir, const double* solution_p, const double* jump_field_p, const double* jump_flux_p)
    {
      if(!current_interface_point_is_set_for(quad_idx, dir))
        set_current_interface_point_for(quad_idx, dir);


      const double &mu_this_side  = (current_interface_data->second.phi_q   > 0.0 ? solver.mu_p : solver.mu_m);
      const double &mu_across     = (current_interface_data->second.phi_nb  > 0.0 ? solver.mu_p : solver.mu_m);

      const bool in_positive_domain = current_interface_data->second.phi_q > 0.0;
      const bool extending_positive_values = !solver.mu_m_is_larger();
      return current_interface_data->second.GFM_interface_defined_value(mu_this_side, mu_across, dir, in_positive_domain, extending_positive_values, solution_p[quad_idx], solution_p[current_interface_data->second.neighbor_quad_idx], jump_field_p, jump_flux_p, solver.dxyz_min);
    }

    inline double GFM_flux_at_center_face(const p4est_locidx_t& quad_idx, const u_char& dir, const double& mu_this_side, const double mu_across,
                                          const bool& in_positive_domain, const bool face_is_on_this_side, const double& solution_quadrant, const double& solution_neighbor_quad,
                                          const double* jump_field_p, const double* jump_flux_p)
    {
      if(!current_interface_point_is_set_for(quad_idx, dir))
        set_current_interface_point_for(quad_idx, dir);
      return current_interface_data->second.GFM_flux_component(mu_this_side, mu_across, dir, in_positive_domain, solution_quadrant, solution_neighbor_quad, jump_field_p, jump_flux_p, solver.dxyz_min, face_is_on_this_side);
    }


#ifdef DEBUG
  int is_map_consistent();
#endif
  } interface_manager;

  class cell_TVD_extension_operator_t
  {
    class local_cell_TVD_extension_operator
    {
      friend class cell_TVD_extension_operator_t;
      struct off_diag_entry{
        double coeff;
        virtual double neighbor_value(const double* extension_on_cells_p, interface_manager_t& interface_manager,
                                      const p4est_locidx_t& quad_idx, const double* solution_p, const double* jump_u_p, const double* jump_flux_p) const = 0;
        inline double contribution_to_negative_normal_derivative(const double* extension_on_cells_p, interface_manager_t& interface_manager,
                                                                 const p4est_locidx_t& quad_idx, const double* solution_p, const double* jump_u_p, const double* jump_flux_p) const
        {
          return coeff*neighbor_value(extension_on_cells_p, interface_manager, quad_idx, solution_p, jump_u_p, jump_flux_p);
        }
        virtual ~off_diag_entry(){};
      };
      struct regular_quad_entry : off_diag_entry
      {
        p4est_locidx_t loc_idx;
        inline double neighbor_value(const double* extension_on_cells_p, interface_manager_t&,
                                     const p4est_locidx_t&, const double*, const double*, const double*) const
        {
          return extension_on_cells_p[loc_idx];
        }
        inline ~regular_quad_entry(){}
      };
      struct interface_entry : off_diag_entry
      {
        u_char dir;
        inline double neighbor_value(const double*, interface_manager_t& interface_manager,
                                     const p4est_locidx_t& quad_idx, const double* solution_p, const double* jump_u_p, const double* jump_flux_p) const
        {
          return interface_manager.interface_value(quad_idx, dir, solution_p, jump_u_p, jump_flux_p);
        }
        inline ~interface_entry(){}
      };

      void clear_extension_entries() {
        for (size_t k = 0; k < extension_entries.size(); ++k)
          delete extension_entries[k];
        extension_entries.clear();
      }

      bool too_close;
      u_char forced_interface_value_dir;
      double diag_entry, dtau, phi_q;
      std::vector<off_diag_entry*> extension_entries;

    public:
      local_cell_TVD_extension_operator() {
        extension_entries.resize(0);
        too_close = false;
        diag_entry = 0.0;
        dtau = DBL_MAX;
        forced_interface_value_dir = UCHAR_MAX;
      }
      ~local_cell_TVD_extension_operator(){ clear_extension_entries(); }

      void add_interface_neighbor(const double* signed_normal, const double* dxyz_min, const interface_neighbor& neighbor, const u_char& direction);

//      void add_one_sided_derivative(const p4est_locidx_t& quad_idx, const double* signed_normal, const u_char& dir,
//                                    const linear_combination_of_dof_t& one_sided_derivative_operator, const double& discretization_distance);

      void add_major_quad_and_direct_neighbors(const p4est_locidx_t& quad_idx, const double* signed_normal, const double* tree_dimensions, const u_char& dir,
                                               const p4est_quadrant_t* major_quad, const p4est_locidx_t& major_quad_idx,
                                               const set_of_neighboring_quadrants& neighbors_across_face, const bool& major_quad_is_leading);
    };

    my_p4est_xgfm_cells_t& solver;
    std::vector<local_cell_TVD_extension_operator> my_local_operators;

  public:
    bool is_set;
    cell_TVD_extension_operator_t(my_p4est_xgfm_cells_t& parent_solver) : solver(parent_solver), is_set(false) {
      my_local_operators.resize(solver.p4est->local_num_quadrants);
    }

    inline double advance_one_pseudo_time_step(const p4est_locidx_t& quad_idx, const double* extension_on_cells_p, const double* solution_p, const double* jump_u_p, const double* jump_flux_p,
                                               double& max_correction_in_band, const double& band_to_diag_ratio) const;

    void build_local_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx);
  } cell_TVD_extension_operator;

  // memorized local interpolation operators
  std::vector<linear_combination_of_dof_t> local_interpolators;
  bool local_interpolators_are_set;

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
   * \brief interpolate_cell_field_at_local_node computes the interpolation of a cell-sampled field at a (subrefined,
   * if using subrefinement) grid node. If the local interpolators are set, they are used to calculate the results,
   * right away; otherwise the hardwork calculation is done and the interpolators are built and stored internally, in
   * order to shortcut subsequent local interpolation calls.
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
  double interpolate_cell_field_at_local_node(const p4est_locidx_t &node_idx, const double *cell_field_p);
  /*!
   * \brief interpolate_cell_extension_to_nodes interpolates the cell-sampled field of the appropriate interface-
   * defined values, i.e., from extension_on_cells, to all nodes (of the interface-capturing grid if using sub-
   * refinement), i.e., to extension_on_nodes. This function will do the hardwork on the very first call, but will
   * store the relevant interpolation data internally to shortcut the task thereafter and optimize execution.
   */
  void interpolate_cell_extension_to_nodes();

  // using PDE extrapolation : uses the current solution results and jump conditions to extend the appropriate
  // interface-defined values in the normal directions, using ASLAM's PDE-based extrapolation on the cells
  void extend_interface_values(Vec &former_extension_on_cells, Vec &former_extension_on_nodes, const double& threshold = 1.0e-10, const uint& niter_max = 20);
  // updates the right-hand side terms for cells involving jump terms (after extension_on_nodes has been updated)
  void update_rhs_and_residual(Vec& former_rhs, Vec& former_residual);
  double set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension_on_cells, Vec former_extension_on_nodes,
                                                         Vec former_rhs, Vec former_residual);

  void compute_jumps_in_flux_components_at_all_nodes();
  void compute_jumps_in_flux_components_for_node(const p4est_locidx_t& node_idx, double *jump_flux_p, const double *jump_normal_flux_p, const double *normals_p, const double *jump_u_p, const double *extension_on_nodes_p) const;

  void initialize_extension_on_cells();
  void initialize_extension_on_cells_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                           const double* const &solution_p, double* const &extension_on_cells_p) const;

  void cell_TVD_extension_of_interface_values(const double& threshold, const uint& niter_max);

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                     const double *user_rhs_p, const double *jump_u_p, const double *jump_flux_p,
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
      compute_jumps_in_flux_components_at_all_nodes();
    else
      interface_manager.update_jumps_in_flux_at_all_relevant_nodes();
    P4EST_ASSERT(jump_flux != NULL);
  }

  void compute_subvolumes_in_computational_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const;

  linear_combination_of_dof_t stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& dir,
                                                                            set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side,
                                                                            double* discretization_distance_out = NULL) const;

public:

  my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t *ngbd_n, const my_p4est_node_neighbors_t *fine_ngbd_n, const bool &activate_xGFM_ = true);
  ~my_p4est_xgfm_cells_t();

#ifdef WITH_SUBREFINEMENT
  /*!
   * \brief set_phi sets the levelset function. Those vectors *MUST* be sampled at the nodes of the interface-capturing grid.
   * Providing the second derivatives of phi is optional. If they're provided, the intersection between the interface and
   * cartesian grid lines are defined as roots of quadratic local interpolant of phi between the appropriate grid nodes of
   * the interface-capturing grid, otherwise linear local interpolants are used.
   * \param [in] node_sampled_phi         : node-sampled levelset values of the levelset function, sampled on the
   *                                        interface-capturing grid.
   * \param [in] node_sampled_phi_xxyyzz  : (optional) node-sampled values of the second derivatives of the levelset
   *                                        function, on the interface-capturing grid. This vector must be block-structured,
   *                                        of blocksize P4EST_DIM
   */
  void set_phi(Vec node_sampled_phi, Vec node_sampled_phi_xxyyzz = NULL);

  /*!
   * \brief set_normals sets the local interface-normal vectors (gradient of the levelset function), sampled on the nodes
   * of the interface-capturing grid.
   * IMPORTANT NOTE : the normals are _not_ assumed to be normalized beforehand : this class always normalizes the normal
   * vector whenever used (after possible interpolation). If the norm of the vector is too small before normalization, the
   * vector is considered locally ill-defined and a zero vector is used instead.
   * \param [in] node_sampled_normals : node-sampled values of the components of the interface-normal vector, sampled on the
   *                                    nodes of the interface-capturing grid
   */
  void set_normals(Vec node_sampled_normals);

  /*!
   * \brief set_jumps sets the jump in solution and in its normal flux, sampled on the nodes of the interface-capturing grid.
   * \param [in] node_sampled_jump_u            : node-sampled values of [u] = u^+ - u^-, sampled on the interface-capturing grid;
   * \param [in] node_sampled_jump_normal_flux  : node-sampled values of [mu*dot(n, grad u)] = mu^+*dot(n, grad u^+) - mu^-*dot(n, grad u^-),
   *                                              sampled on the interface-capturing grid.
   */
  void set_jumps(Vec node_sampled_jump_u, Vec node_sampled_jump_normal_flux);
#else
#endif

  inline void set_bc(const BoundaryConditionsDIM& bc_)
  {
    bc = &bc_;
    // we can't really check for unchanged behavior in this cass, --> play it safe
    matrix_is_set = false;
    rhs_is_set    = false;
  }

  inline void set_mus(const double& mu_m_, const double& mu_p_)
  {
    const bool mus_unchanged = (fabs(mu_m_ - mu_m) < EPS*MAX(mu_m_, mu_m)) && (fabs(mu_p_ - mu_p) < EPS*MAX(mu_p_, mu_p));
    matrix_is_set = matrix_is_set && mus_unchanged;
    rhs_is_set    = rhs_is_set    && mus_unchanged;
    if(!mus_unchanged)
    {
      mu_m = mu_m_;
      mu_p = mu_p_;
    }
    P4EST_ASSERT(diffusion_coefficients_have_been_set()); // must be both strictly positive
  }

  inline void set_diagonals(const double& add_m, const double& add_p)
  {
    const bool diags_unchanged = (fabs(add_m - add_diag_m) < EPS*MAX(add_m, add_diag_m)) && (fabs(add_p - add_diag_p) < EPS*MAX(add_p, add_diag_p));
    matrix_is_set = matrix_is_set && diags_unchanged;
    rhs_is_set    = rhs_is_set    && diags_unchanged;
    if(!diags_unchanged)
    {
      add_diag_m = add_m;
      add_diag_p = add_p;
    }
  }
  inline void set_rhs(Vec user_rhs_)
  {
    P4EST_ASSERT(VecIsSetForCells(user_rhs_, p4est, ghost, 1, false));
    user_rhs = user_rhs_;
    rhs_is_set = false;
  }

  /* Benchmark tests revealed that PCHYPRE is MUCH faster than PCSOR as PCType!
   * The linear systme is supposed to be symmetric positive (semi-) definite, so KSPCG is ok as KSPType
   * Note: a low threshold for tolerance_on_rel_residual is critical to ensure accuracy in cases with large differences in diffusion coefficients!
   * */
  void solve(KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE, double absolute_accuracy_threshold = 1e-8, double tolerance_on_rel_residual = 1e-12);

  inline Vec get_extended_interface_values()                                        { make_sure_extensions_are_defined();               return extension_on_cells;  }
  inline Vec get_extended_interface_values_interpolated_on_nodes()                  { make_sure_extensions_are_defined();               return extension_on_nodes;  }
  inline Vec get_jump_in_flux(const bool& everywhere = true)                        { make_sure_jumps_in_flux_are_defined(everywhere);  return jump_flux;           }
  inline Vec get_solution()                                                         { make_sure_solution_is_set();                      return solution;            }
  inline int get_number_of_xGFM_corrections()                                 const { return solver_monitor.get_number_of_xGFM_corrections();                       }
  inline std::vector<PetscInt> get_numbers_of_ksp_iterations()                const { return solver_monitor.get_n_ksp_iterations();                                 }
  inline std::vector<double> get_max_corrections()                            const { return solver_monitor.get_max_corrections();                                  }
  inline std::vector<double> get_relative_residuals()                         const { return solver_monitor.get_relative_residuals();                               }
  inline bool is_using_xGFM()                                                 const { return activate_xGFM;                                                         }
  inline bool get_matrix_has_nullspace()                                      const { return A_null_space != NULL;                                                  }
  inline const p4est_t* get_computational_p4est()                             const { return p4est;                                                                 }
  inline const p4est_ghost_t* get_computational_ghost()                       const { return ghost;                                                                 }
  inline const p4est_nodes_t* get_computational_nodes()                       const { return nodes;                                                                 }
  inline const my_p4est_hierarchy_t* get_computational_hierarchy()            const { return cell_ngbd->get_hierarchy();                                            }
  inline const my_p4est_node_neighbors_t* get_computational_node_neighbors()  const { return node_ngbd;                                                             }
#ifdef WITH_SUBREFINEMENT
  inline Vec get_subrefined_phi()                                             const { return phi;                                                                   }
  inline Vec get_subrefined_normals()                                         const { return normals;                                                               }
  inline Vec get_subrefined_jump()                                            const { return jump_u;                                                                }
  inline Vec get_subrefined_jump_in_normal_flux()                             const { return jump_normal_flux_u;                                                    }
  inline const my_p4est_node_neighbors_t* get_subrefined_node_neighbors()     const { return fine_node_ngbd;                                                        }
  inline const p4est_t* get_subrefined_p4est()                                const { return fine_p4est;                                                            }
  inline const p4est_ghost_t* get_subrefined_ghost()                          const { return fine_ghost;                                                            }
  inline const p4est_nodes_t* get_subrefined_nodes()                          const { return fine_nodes;                                                            }
  inline const my_p4est_hierarchy_t* get_subrefined_hierarchy()               const { return fine_node_ngbd->get_hierarchy();                                       }
  inline const my_p4est_interpolation_nodes_t& get_interp_phi()               const { return interp_subrefined_phi;                                                 }
#else
  inline const my_p4est_interpolation_nodes_t& get_interp_phi()               const { return *interp_phi;                                                           }
#endif

  inline double get_sharp_integral_solution() const
  {
    PetscErrorCode ierr;
    P4EST_ASSERT(solution != NULL);
    double *sol_p;
    ierr = VecGetArray(solution, &sol_p); CHKERRXX(ierr);
#ifdef WITH_SUBREFINEMENT
    my_p4est_interpolation_nodes_t interp_phi(fine_node_ngbd); interp_phi.set_input(phi, linear);
    my_p4est_interpolation_nodes_t interp_jump(fine_node_ngbd); interp_jump.set_input(jump_u, linear);
#endif

    double sharp_integral_solution = 0.0;
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
      const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        double negative_volume, positive_volume;
        compute_subvolumes_in_computational_cell(quad_idx, tree_idx, negative_volume, positive_volume);

        double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
        // crude estimate but whatever, it's mostly to get closer to what we expect...
        sharp_integral_solution += sol_p[quad_idx]*(negative_volume + positive_volume);
        sharp_integral_solution += (interp_phi(xyz_quad) <= 0.0 ? positive_volume : -negative_volume)*interp_jump(xyz_quad);
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

};

#endif // MY_P4EST_XGFM_CELLS_H

