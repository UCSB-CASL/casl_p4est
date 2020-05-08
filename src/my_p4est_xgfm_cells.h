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

inline bool signs_of_phi_are_different(const double& phi_0, const double& phi_1)
{
  return (phi_0 > 0.0) != (phi_1 > 0.0);
}

/*!
 * \brief The interface_neighbor struct contains all relevant data regarding interface-neighbor,
 * i.e., intersection between the interface and the segment joining the current cell of interest
 * and its neighbor cell, on the computational grid.
 * phi_q    : value of the level-set at the center of the cell of interest;
 * phi_nb   : value of the level-set at the center of the neighbor cell (i.e. across the interface);
 * theta    : fraction of the grid spacing covered by the domain in which the cell of interest is;
 * mu_this_side : value of the diffusion coefficient as seen from the the cell of interest;
 * mu_other_side: value of the diffusion coefficient across the interface (as seen from the neighbor cell);
 * int_value    : value of u^- (or u^+) at the interface point;
 * quad_nb_idx  : local index of the neighbor cell in the computational grid (across the interface);
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
  double  int_value;
  p4est_locidx_t quad_nb_idx;
#ifdef SUBREFINED
  p4est_locidx_t mid_point_fine_node_idx;
  p4est_locidx_t quad_fine_node_idx;
  p4est_locidx_t nb_fine_node_idx;
#endif
#ifdef DEBUG
  bool is_consistent_with_neighbor_across(const interface_neighbor nb_across) const
  {
    return ((phi_q > 0.0) != (nb_across.phi_q > 0.0))
        && fabs(phi_q - nb_across.phi_nb) < EPS*MAX(fabs(phi_q), fabs(phi_nb))
        && fabs(phi_nb - nb_across.phi_q) < EPS*MAX(fabs(phi_q), fabs(phi_nb))
        && fabs(theta + nb_across.theta - 1.0) < EPS
        && fabs(int_value - nb_across.int_value) < 0.000001*MAX(fabs(int_value), 1.0);
  }
#endif
};

#ifdef DEBUG
struct which_interface_nb
{
  p4est_locidx_t loc_idx;
  u_char dir;
};
#endif

struct extension_matrix_entry
{
  p4est_locidx_t loc_idx;
  double coeff;
};
struct extension_interface_value_entry
{
  int dir;
  double coeff;
};

struct extension_affine_map
{
  bool too_close;
  int forced_interface_value_dir;
  double diag_entry, dtau, phi_q;
  std::vector<extension_interface_value_entry> interface_entries;
  std::vector<extension_matrix_entry> quad_entries;
  extension_affine_map() {
    interface_entries.resize(0);
    quad_entries.resize(0);
    too_close = false;
    diag_entry = 0.0;
    forced_interface_value_dir = -1;
  }
};

class my_p4est_xgfm_cells_t
{
  // data related to the computational grid
  const my_p4est_cell_neighbors_t *cell_ngbd;
  const my_p4est_node_neighbors_t *node_ngbd;
#ifdef DEBUG
  p4est_t             *p4est; // I loose the const qualifier on this one in DEBUG because of some of p4est's debug check functions that can't take const p4est objects in
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

#ifdef SUBREFINED
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
#ifdef SUBREFINED
  Vec phi, normals, phi_xxyyzz;   // node-sampled on fine nodes, if using subrefinement
  Vec jump_u, jump_normal_flux_u; // node-sampled on fine nodes, if using subrefinement
  inline bool levelset_has_been_set() const { return phi != NULL; }
  inline bool normals_have_been_set() const { return normals != NULL; }
  inline bool jumps_have_been_set() const   { return jump_u != NULL && jump_normal_flux_u != NULL; }
#else
  const my_p4est_interpolation_nodes_t *interp_phi, *interp_normals;
  const my_p4est_interpolation_nodes_t *interp_jump_u, *interp_jump_normal_flux_u;
  inline bool levelset_has_been_set() const { return interp_phi != NULL; }
  inline bool normals_have_been_set() const { return interp_normals != NULL; }
  inline bool jumps_have_been_set() const   { return interp_jump_u != NULL && interp_jump_normal_flux_u != NULL; }
#endif
  /* ---- OWNED BY THE SOLVER ---- (therefore destroyed at solver's destruction, except if returned before-hand) */
  Vec rhs;                  // cell-sampled, discretized rhs
//  Vec corrected_rhs;        // cell-sampled, discretized rhs
  Vec solution;             // cell-sampled
  Vec extension_on_cells;   // cell-sampled
  Vec extension_on_nodes;   // node-sampled (fine nodes if subrefined)
  Vec jump_flux;            // node-sampled, P4EST_DIM block-structure (fine nodes if subrefined)
  /* ---- other PETSc objects ---- */
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  PetscErrorCode ierr;

  const BoundaryConditionsDIM *bc;

  // solver monitoring
  std::vector<PetscInt> numbers_of_ksp_iterations;
  std::vector<double> max_corrections, relative_residuals;

  // flags
  bool matrix_is_preallocated, matrix_is_set, rhs_is_set;
  bool interface_values_are_set;
  bool solution_is_set, use_initial_guess;
  const bool activate_xGFM;

  // interface_jump_info:
  // key    = local index of the considered quadrant
  // value  = another map such that
  //      - key   = (oriented) Cartesian direction in which the jump info is sought;
  //      - value = structure encapsulating the jump info.
//  std::map<p4est_locidx_t, std::map<u_char, jump_data> > interface_jump_data;
  std::map<p4est_locidx_t, std::map<u_char, interface_neighbor> > map_of_interface_neighbors;

  // memorized local extension operators
  std::vector<extension_affine_map> extension_entries;
  bool extension_entries_are_set;
  // memorized local interpolation operators
  std::vector<cell_field_interpolator_t> local_interpolator;
  bool local_interpolators_are_set;

  // disallow copy ctr and copy assignment
  my_p4est_xgfm_cells_t(const my_p4est_xgfm_cells_t& other);
  my_p4est_xgfm_cells_t& operator=(const my_p4est_xgfm_cells_t& other);

  // internal procedures
  void preallocate_matrix();
  void setup_linear_system();
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
   * \brief interpolate_cell_field_to_nodes interpolates the cell-sampled field (sampled on the cells of computational
   * grid) at all nodes (of the interface-capturing grid if using subrefinement). This function will do the
   * hardwork on the very first call, but will store the relevant interpolation data internally to shortcut the
   * task thereafter.
   * \param [in]  cell_field_p            : pointer to the cell-sampled data field to interpolate (sampled on the
   *                                        computational grid)
   * \param [out] interpolated_node_field : node-sampled vector containing the result of the interpolation operation
   *                                        on output (sampled on the nodes of the computational grid if not using
   *                                        subrefinement, on the nodes of the interface-capturing grid otherwise)
   */
  void interpolate_cell_field_to_nodes(const double *cell_field_p, Vec interpolated_node_field);

  // using PDE extrapolation
  void extend_interface_values(const double *solution_p, Vec new_cell_extension, const double* extension_on_fine_nodes_p, double threshold = 1.0e-10, uint niter_max = 20);
  // get the correction jump terms
  void get_corrected_rhs(Vec corrected_rhs, const double *fine_extension_interface_values_p);

#ifdef DEBUG
  int is_map_consistent() const;
#endif

  void compute_jumps_in_flux_components();
  void compute_jumps_in_flux_components_for_node(const p4est_locidx_t& node_idx, double *jump_flux_p,
                                                 const double *jump_normal_flux_p, const double *normals_p, const double *jump_u_p, const double *extension_on_nodes_p);

  bool interface_neighbor_is_found(const p4est_locidx_t& quad_idx, const u_char& dir, interface_neighbor& int_nb) const ;
  interface_neighbor get_interface_neighbor(const p4est_locidx_t& quad_idx, const u_char& dir, const p4est_locidx_t& nb_quad_idx,
                                            const p4est_locidx_t& quad_fine_node_idx, const p4est_locidx_t& nb_fine_node_idx,
                                            const double *phi_p, const double *phi_xxyyzz_p);
//  bool jump_data_is_found(const p4est_locidx_t& quad_idx, const u_char& dir, jump_data& int_nb) const ;
//  jump_data get_jump_data(const p4est_quadrant_t& quad, const u_char& dir, const p4est_quadrant_t& nb_quad,
//                          const p4est_locidx_t& quad_fine_node_idx, const p4est_locidx_t& nb_fine_node_idx,
//                          const double *phi_p, const double *phi_xxyyzz_p, const double *jump_u_p, const double *jump_flux_p);

  void update_interface_values(Vec new_cell_extension, const double *solution_p, const double *extension_on_fine_nodes_p);
  void cell_TVD_extension_of_interface_values(Vec new_cell_extension, const double& threshold, const uint& niter_max);

  inline bool quad_center_is_fine_node(const p4est_quadrant_t &quad, const p4est_locidx_t &tree_idx, p4est_locidx_t& fine_node_idx_of_quad_center) const
  {
#ifdef DEBUG
    fine_node_idx_of_quad_center = -1;
#endif
    bool to_return = logical_vertex_in_quad_is_fine_node(fine_p4est, fine_nodes, quad, tree_idx, DIM(0, 0, 0), fine_node_idx_of_quad_center);
    P4EST_ASSERT(!to_return || fine_node_idx_of_quad_center >= 0);
    return to_return;
  }

  inline bool face_in_quad_is_fine_node(const p4est_quadrant_t &quad, const p4est_locidx_t &tree_idx, const u_char& face_dir, p4est_locidx_t& fine_node_idx_for_face) const
  {
    char logical_vertex_in_quad[P4EST_DIM] = {DIM(0, 0, 0)}; logical_vertex_in_quad[face_dir/2] = (face_dir%2 == 1 ? 1 : -1);
#ifdef DEBUG
    fine_node_idx_for_face = -1;
#endif
    bool to_return = logical_vertex_in_quad_is_fine_node(fine_p4est, fine_nodes, quad, tree_idx, DIM(logical_vertex_in_quad[0], logical_vertex_in_quad[1], logical_vertex_in_quad[2]), fine_node_idx_for_face);
    P4EST_ASSERT(!to_return || fine_node_idx_for_face >= 0);
    return to_return;
  }

  void build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                     const double *phi_p, const double* phi_xxyyzz_p, const my_p4est_interpolation_nodes_t& interp_phi,
                                     const double *user_rhs_p, const double *jump_u_p, const double *jump_flux_p,
                                     int &nullspace_contains_constant_vector, double* rhs_p);

public:

  my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t *ngbd_n, const my_p4est_node_neighbors_t *fine_ngbd_n, const bool &activate_xGFM_ = true);
  ~my_p4est_xgfm_cells_t();

#ifdef SUBREFINED
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
    // we can't really check for unchanged behavior in this cass, --> play safe
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
  inline bool mu_m_is_larger() const { return mu_m >= mu_p; }
  inline bool mus_are_equal() const { return fabs(mu_m - mu_p) < EPS*MAX(fabs(mu_m), fabs(mu_p)); }
  inline bool diffusion_coefficients_have_been_set() const { return mu_m > 0.0 && mu_p > 0.0; }
  inline double get_smaller_mu() const { return (mu_m_is_larger() ? mu_p : mu_m); }
  inline double get_jump_in_mu() const { return (mu_p - mu_m); }

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
  inline bool get_matrix_has_nullspace() const
  {
    return A_null_space != NULL;
  }

  /* Benchmark tests revealed that PCHYPRE is MUCH faster than PCSOR as PCType!
   * The linear systme is supposed to be symmetric positive (semi-) definite, so KSPCG is ok as KSPType
   * Note: a low threshold for tolerance_on_rel_residual is critical to ensure accuracy in cases with large differences in diffusion coefficients!
   * */
  void solve(KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE, double absolute_accuracy_threshold = 1e-8, double tolerance_on_rel_residual = 1e-12);

  inline void get_extended_interface_values(Vec& cell_centered_extension, Vec& fine_node_sampled_extension)
  {
    P4EST_ASSERT(!solution_is_set || solution != NULL || extension_on_cells != NULL); // the extended interface values cannot be calculated if the solution has been returned to the user beforehand
    if(!solution_is_set)
      solve();
    if(extension_on_cells == NULL)
    {
      P4EST_ASSERT(jump_flux != NULL); // the extended interface values cannot be calculated if the jumps in flux components have been returned to the used beforehand
      P4EST_ASSERT((!activate_xGFM || mus_are_equal()) && extension_on_nodes == NULL); // those are the (only) conditions under which the extension on cells can possibly be not defined

      const double *solution_p;
      ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
      ierr = VecCreateGhostCells(p4est, ghost, &extension_on_cells); CHKERRXX(ierr);
      extend_interface_values(solution_p, extension_on_cells, NULL);
      const double *extension_cell_values_p;
      ierr = VecGetArrayRead(extension_on_cells, &extension_cell_values_p); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_nodes); CHKERRXX(ierr);
      interpolate_cell_field_to_nodes(extension_cell_values_p, extension_on_nodes);
      ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extension_on_cells, &extension_cell_values_p); CHKERRXX(ierr);
    }
    cell_centered_extension = extension_on_cells;
    extension_on_cells = NULL; // will be handled by the new owner (hopefully :-P)...
    fine_node_sampled_extension = extension_on_nodes;
    extension_on_nodes = NULL; // will be handled by the new owner (hopefully :-P)...
  }

  inline void get_jump_flux(Vec& to_return)
  {
    if(activate_xGFM && !solution_is_set)
      solve();
    to_return = jump_flux;
    jump_flux = NULL; // will be handled by the new owner (hopefully :-P)...
  };

  int get_number_of_corrections() const {return numbers_of_ksp_iterations.size()-1; }
  std::vector<PetscInt> get_numbers_of_ksp_iterations() const {return numbers_of_ksp_iterations; }
  std::vector<double> get_max_corrections() const {return max_corrections; }
  std::vector<double> get_relative_residuals() const {return relative_residuals; }

  void get_flux_components_and_subtract_them_from_velocities(Vec flux[P4EST_DIM], my_p4est_faces_t *faces, Vec vstar[P4EST_DIM] = NULL, Vec vnp1_minus[P4EST_DIM] = NULL, Vec vnp1_plus[P4EST_DIM] = NULL);
  inline void get_flux_components(Vec flux[P4EST_DIM], my_p4est_faces_t* faces)
  {
    get_flux_components_and_subtract_them_from_velocities(flux, faces);
  }

  inline Vec get_solution()
  {
    if(!solution_is_set)
      solve();
    Vec to_return = solution;
    solution = NULL; // will be handled by user, hopefully!
    return to_return;
  }

  /*!
   * \brief set_initial_guess self-explanatory, the user loses ownership of the object 'initial_guess'
   * \param initial_guess self-explanatory
   */
  inline void set_initial_guess(Vec& initial_guess)
  {
    P4EST_ASSERT(VecIsSetForCells(initial_guess, p4est, ghost, 1));
    if(solution != NULL){
      ierr = VecDestroy(solution); CHKERRXX(ierr); } // make sure we don't have a memory leak here
    solution          = initial_guess;  // --> the solver gets the ownership of the object
    initial_guess     = NULL;           // --> the user loses the ownership of the object
    use_initial_guess = true;           // --> the solver will use this object as its initial guess!
  }

};

#endif // MY_P4EST_XGFM_CELLS_H

