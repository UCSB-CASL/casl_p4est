#ifndef MY_P4EST_INTERFACE_MANAGER_H
#define MY_P4EST_INTERFACE_MANAGER_H

#ifdef P4_TO_P8
#include <src/my_p8est_faces.h>
#else
#include <src/my_p4est_faces.h>
#endif

#if __cplusplus >= 201103L
#include <unordered_map>
#else
#include <map>
#endif

struct GFM_jump_info
{
  double jump_field;
  double jump_flux_component;
};

struct which_interface_neighbor_t
{
  p4est_locidx_t local_dof_idx;
  p4est_locidx_t neighbor_dof_idx;

  inline bool operator==(const which_interface_neighbor_t& other) const // equality comparator
  {
    return (this->local_dof_idx == other.local_dof_idx && this->neighbor_dof_idx == other.neighbor_dof_idx)
        || (this->local_dof_idx == other.neighbor_dof_idx && this->neighbor_dof_idx == other.local_dof_idx);
  }

  inline bool operator<(const which_interface_neighbor_t& other) const // comparison operator for storing in (standard) ordered map
  {
    return (MIN(this->local_dof_idx, this->neighbor_dof_idx) < MIN(other.local_dof_idx, other.neighbor_dof_idx)
            || (MIN(this->local_dof_idx, this->neighbor_dof_idx) == MIN(other.local_dof_idx, other.neighbor_dof_idx)
                && MAX(this->local_dof_idx, this->neighbor_dof_idx) < MAX(other.local_dof_idx, other.neighbor_dof_idx)));
  }
};

/*!
 * \brief The FD_interface_data struct contains the geometric-related data pertaining to finite-difference
 * interface neighbor points, i.e., intersection between the interface and the (cartesian) grid line joining
 * the degrees of freedom of interest, on the computational grid.
 * theta            : fraction of the grid spacing covered by the domain in which the cell of interest is;
 * node_interpolant : basic operator required to interpolate data (linear interpolation) from the nodes of the
 *                    interpolation_node_ngbd to the relevant interface point.
 */
struct FD_interface_data
{
  double theta;
  linear_combination_of_dof_t node_interpolant;
  bool swapped;

  inline double GFM_mu_tilde(const double& mu_this_side, const double& mu_across) const
  {
    return (1.0 - theta)*mu_this_side + theta*mu_across;
  }

  inline double GFM_mu_jump(const double& mu_this_side, const double& mu_across) const
  {
    return mu_this_side*mu_across/GFM_mu_tilde(mu_this_side, mu_across);
  }

  inline GFM_jump_info get_GFM_jump_data(const double* jump_p, const double* jump_flux_p, const u_char& flux_component) const
  {
    return {node_interpolant(jump_p), node_interpolant(jump_flux_p, flux_component, P4EST_DIM)};
  }

  inline double GFM_jump_terms_for_flux_component(const double& mu_this_side, const double& mu_across, const u_char& oriented_dir, const bool &this_side_is_in_positive_domain,
                                                  const double* jump_p, const double* jump_flux_p, const double& xgfm_flux_correction, const double* dxyz) const
  {
    GFM_jump_info jump_info = get_GFM_jump_data(jump_p, jump_flux_p, oriented_dir/2);

    return GFM_mu_jump(mu_this_side, mu_across)*(this_side_is_in_positive_domain ? +1.0 : -1.0)*
        ((jump_info.jump_flux_component + xgfm_flux_correction)*(1 - theta)/mu_across + (oriented_dir%2 == 1 ? +1.0 : -1.0)*jump_info.jump_field/dxyz[oriented_dir/2]);
  }

  inline double GFM_flux_component_this_side(const double& mu_this_side, const double& mu_across, const u_char& oriented_dir, const bool &this_side_is_in_positive_domain,
                                             const double& solution_this_side, const double& solution_across,
                                             const double* jump_p, const double* jump_flux_p, const double& xgfm_flux_correction, const double* dxyz) const
  {
    return (oriented_dir%2 == 1 ? +1.0 : -1.0)*GFM_mu_jump(mu_this_side, mu_across)*(solution_across - solution_this_side)/dxyz[oriented_dir/2]
        + GFM_jump_terms_for_flux_component(mu_this_side, mu_across, oriented_dir, this_side_is_in_positive_domain, jump_p, jump_flux_p, xgfm_flux_correction, dxyz);
  }

  inline double GFM_interface_defined_value(const double& mu_this_side, const double& mu_across, const u_char& oriented_dir, const bool &this_side_is_in_positive_domain, const bool& get_positive_interface_value,
                                            const double& solution_this_side, const double& solution_across,
                                            const double* jump_p, const double* jump_flux_p, const double& xgfm_flux_correction, const double* dxyz) const
  {
    GFM_jump_info jump_info = get_GFM_jump_data(jump_p, jump_flux_p, oriented_dir/2);

    return ((1.0 - theta)*mu_this_side*(solution_this_side  + (this_side_is_in_positive_domain != get_positive_interface_value ? (this_side_is_in_positive_domain ? -1.0 : +1.0)*jump_info.jump_field : 0.0))
            +      theta *mu_across   *(solution_across     + (this_side_is_in_positive_domain == get_positive_interface_value ? (this_side_is_in_positive_domain ? +1.0 : -1.0)*jump_info.jump_field : 0.0))
            + (this_side_is_in_positive_domain ? +1.0 : -1.0)*(oriented_dir%2 == 1 ? +1.0 : -1.0)*theta*(1.0 - theta)*dxyz[oriented_dir/2]*(jump_info.jump_flux_component + xgfm_flux_correction))/GFM_mu_tilde(mu_this_side, mu_across);
  }
};

#if __cplusplus >= 201103L
// hash value for unordered map keys
struct hash_functor{
  inline size_t operator()(const which_interface_neighbor_t& key) const
  {
    return ((size_t) MIN(key.local_dof_idx, key.neighbor_dof_idx) << 8*sizeof (p4est_locidx_t)) + MAX(key.local_dof_idx, key.neighbor_dof_idx);
  }
};
typedef std::unordered_map<which_interface_neighbor_t, FD_interface_data, hash_functor> map_of_interface_neighbors_t;
#else
typedef std::map<which_interface_neighbor_t, interface_neighbor> map_of_interface_neighbors_t;
#endif

class my_p4est_interface_manager_t
{
  // computational grid data
  const my_p4est_faces_t          *faces;
  const my_p4est_cell_neighbors_t *c_ngbd;
  const p4est_t                   *p4est;
  const p4est_ghost_t             *ghost;
  const p4est_nodes_t             *nodes;
  const double                    *dxyz_min;
  Vec                             phi_on_computational_nodes;
  // object related to possibly subresolved interface-capturing feature
  const my_p4est_node_neighbors_t *interpolation_node_ngbd;
  my_p4est_interpolation_nodes_t  interp_phi;
  my_p4est_interpolation_nodes_t  *interp_grad_phi;
  my_p4est_interpolation_nodes_t  *interp_phi_xxyyzz;
  Vec                             grad_phi_local;
  const int                       max_level_p4est;
  const int                       max_level_interpolation_p4est;
  bool                            use_second_derivative_when_computing_FD_theta;

  FD_interface_data *tmp_FD_interface_data; // unique element to be used at first construction/pass through map or if maps are not used at all (pointer so that I can keep most methods const hereunder);
  map_of_interface_neighbors_t *cell_FD_interface_data;
  map_of_interface_neighbors_t *face_FD_interface_data[P4EST_DIM];

  inline void clear_cell_FD_interface_data() {
    if(cell_FD_interface_data != NULL)
      cell_FD_interface_data->clear();
  }
  inline void clear_face_FD_interface_data(const u_char& dim) {
    if(face_FD_interface_data[dim] != NULL)
      face_FD_interface_data[dim]->clear();
  }

  const FD_interface_data& get_cell_FD_interface_data_for(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const;

  void build_grad_phi_locally();

  // disallow copy ctr and copy assignment
  my_p4est_interface_manager_t(const my_p4est_interface_manager_t& other);
  my_p4est_interface_manager_t& operator=(const my_p4est_interface_manager_t& other);

public:
  /*!
   * \brief my_p4est_interface_manager_t constructor of the interface manager object
   * \param [in] faces_ pointer to a constant face structure for your computational grid
   *             (all the grid-related information is fetched from there, except for the nodes)
   * \param [in] nodes_ pointer to a constant node structure for your computational grid
   * \param [in] interpolation_node_ngbd_ pointer to a constant node-neighborhood information
   *             for your interface-capturing grid. Your interface-capturing grid may be identical
   *             to your computational grid or subresolved close to the interface, but if it MUST
   *             NOT be coarser than your computational grid!
   *             (if you're smart enough, you'll have initialized that neighborhood beforehand,
   *              because it can't be done thereafter given the internal 'const' qualifier...)
   * - This constructor creates relevant (empty) maps for storing finite-difference related data for
   * cell and faces dofs. The default behavior of the object is to store that information internally
   * (and without duplicates) to accelerate interface-specific treatment of any type thereafter.
   * If you want this object to _NOT_ store any such information in memory, call the
   * "do_not_store_*" routines after construction...
   */
  my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const p4est_nodes_t* nodes_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_);
  ~my_p4est_interface_manager_t();

  /*!
   * \brief set_levelset sets the levelset values, which *MUST* be sampled on the nodes of the interface-capturing grid, i.e.,
   * on the nodes of the interpolation_node_ngbd.
   * \param [in] phi                            vector of node-sampled levelset values, must be sampled on the nodes of the
   *                                            interface-capturing grid, i.e., on the nodes of interpolation_node_ngbd_n.
   * \param [in] method_interp_phi              [optional] desired interpolation method for evaluating your levelset function
   *                                            at points, deault is linear interpolation.
   *                                            If your interface-capturing grid is subresolving, this parameter should be
   *                                            rather irrelevant regarding the subsequent behavior of (jump) solver(s) for
   *                                            cell- and face-sampled dofs, since the relevant value of the levelset should
   *                                            actually be sampled at those points in that case.
   *                                            However, this should definitely be specified by the user if not using a
   *                                            subresolving interface-capturing grid grid .
   * \param [in] phi_xxyyzz                     [optional] second derivatives of your levelset function, must be
   *                                            P4EST_DIM block-structured, and sampled on the nodes of the interface-capturing
   *                                            grid, i.e., on the nodes of interpolation_node_ngbd_n. This input is optional
   *                                            but it is advised and/or required to provide them in the following cases
   *                                            a)  if you're not using subresolution (i.e. if your interface-capturing grid is
   *                                                identical to your computational grid) but you are _NOT_ using linear
   *                                                interpolation for your levelset values --> it is advised to provide them in
   *                                                that case (for speedup)
   *                                            b)  if you want the finite difference theta values to be evaluated as the roots
   *                                                of local quadratic interpolation polynomials along grid lines (which is the
   *                                                default behavior of this object when the second derivatives are available).
   *                                                If the second derivatives are not provided in this call, this object will
   *                                                _NOT_ calculate them and the aforementioned values of theta will be evaluated
   *                                                as roots of linear interpolants along grid lines.
   *                                            If available, the second derivatives of phi will be interpolated with linear inter-
   *                                            polation, where needed (other methods of interpolation would required 4th order
   *                                            derivatives of phi, which would probably be as good as a white noise signal input,
   *                                            anyways)
   * \param [in] build_and_set_grad_phi_locally [optional] flag activating the calculation and configuration of the internal inter-
   *                                            -polation tool for evaluating gradients of the levelset function (the gradient of
   *                                            the levelset function is computed internally and owned by this object in that case)
   * NOTE : if _NOT_ using a subresolving interpolation_node_ngbd, this method will also set phi as the "phi_on_computational_nodes"
   * (otherwise, the user is advised have to set it themselves using "set_under_resolved_levelset" if needed/desired)
   */
  void set_levelset(Vec phi, const interpolation_method& method_interp_phi = linear, Vec phi_xxyyzz = NULL, const bool& build_and_set_grad_phi_locally = false);

  /*!
   * \brief set_grad_phi sets the vector of node-sampled gradient values of the interface-capturing levelset function.
   * If provided by the user, this object doesn't take ownership, otherwise, it will build it and compute it internally.
   * \param [in] grad_phi_in                    [option] first derivatives of your levelset function or NULL pointer. If not NULL,
   *                                            the vector must be P4EST_DIM block-structured, and sampled on the nodes of the
   *                                            interface-capturing grid, i.e., on the nodes of interpolation_node_ngbd_n. If NULL
   *                                            (or disregarded), this method will calculate them internally (and own them).
   * NOTE : the gradient of the levelset function will be interpolated with linear interpolation, where needed (other methods of
   * interpolation would required 3rd order derivatives of phi, which would probably be as good as a white noise signal input, anyways)
   */
  void set_grad_phi(Vec grad_phi_in = NULL);
  inline bool is_grad_phi_set() const { return interp_grad_phi != NULL; }

  /*!
   * \brief set_under_resolved_levelset self-explanatory. The values of the levelset function on the nodes of your computational grids
   * are only available via interpolation if not invoked and if using a subresolving interpolation_node_ngbd. If not using a subresolving
   * interpolation_node_ngbd, this method will also set the internal levelset interpolation tool if it is not done yet (with a default linear
   * interpolation method)
   * \param [in] phi_on_computational_nodes_    vector of node-sampled levelset values, sampled on the nodes of your computational grids, i.e.,
   *                                            on "nodes" (and _NOT_ on interpolation_node_ngbd.get_nodes() if they're different).
   */
  void set_under_resolved_levelset(Vec phi_on_computational_nodes_);

  /*!
   * \brief do_not_store_cell_FD_interface_data deactivates the internal storage of
   * finite-difference interface data between cell dofs
   */
  void do_not_store_cell_FD_interface_data();

  /*!
   * \brief do_not_store_face_FD_interface_data deactivates the internal storage of
   * finite-difference interface data between face dofs
   */
  void do_not_store_face_FD_interface_data();

  /*!
   * \brief clear_all_FD_interface_data clears the content  of all finite-difference
   * data stored internally (between cells as well as between faces)
   */
  void clear_all_FD_interface_data();

  /*!
   * \brief is_storing_cell_FD_interface_data self_explanatory
   * \return true if storing finite-difference data between cell dofs internally
   */
  inline bool is_storing_cell_FD_interface_data() const { return cell_FD_interface_data != NULL; }

  /*!
   * \brief is_storing_face_FD_interface_data self_explanatory
   * \return true if storing finite-difference data between face dofs internally
   */
  inline bool is_storing_face_FD_interface_data() const { return ANDD(face_FD_interface_data[0] != NULL, face_FD_interface_data[1] != NULL, face_FD_interface_data[2] != NULL); }

  /*!
   * \brief evaluate_FD_theta_with_quadratics_if_second_derivatives_are_available self-explanatory
   * \param [in] flag : the user's choice
   * NOTE: (default flag value is set to true internally if this function is never called)
   */
  inline void evaluate_FD_theta_with_quadratics_if_second_derivatives_are_available(const bool& flag) { use_second_derivative_when_computing_FD_theta = flag; }

  /*!
   * \brief get_FD_theta_between_cells self-explanatory
   * \param [in] quad_idx           local index of the cell of interest (cumulative over the local trees) [must be a local cell]
   * \param [in] neighbor_quad_idx  local index of its neighbor cell across the interface (cumulative over the local trees) [may be a ghost cell]
   * \param [in] oriented_dir       oriented cartesian direction in which the neighbor cell is, as seen from the cell of interest
   * \return the infamous finite-difference theta value, AS SEEN FROM THE CELL OF INTEREST, i.e., as seen from the cell of local index quad_idx.
   * [NOTE :] This routine will fetch the data from its appropriate map, if using it and if found in there. Otherwise, the interface data will be
   * built on-the-fly. If the object is storing such data in maps, it will be added to it to accelerate access thereafter
   */
  inline double get_FD_theta_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.theta;
  }

  void get_coordinates_of_FD_interface_point_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, double *xyz) const;

  /*!
   * \brief GFM_mu_jump_between_cells self-explanatory calculates the "effective diffusion coefficient"
   * \param [in] quad_idx           local index of the cell of interest (cumulative over the local trees) [must be a local cell]
   * \param [in] neighbor_quad_idx  local index of its neighbor cell across the interface (cumulative over the local trees) [may be a ghost cell]
   * \param [in] oriented_dir       oriented cartesian direction in which the neighbor cell is, as seen from the cell of interest
   * \param [in] mu_this_side       value of the diffusion coefficient as seen from the cell of interest
   * \param [in] mu_across          value of the diffusion coefficient as seen from the neighbor cell acoss the interface
   * \return the value of the "effective diffusion coefficient", i.e. the value \hat{mu}, as defined in the standard Ghost Fluid Method
   * (as defined in "A Boundary Condition Capturing Method for Poisson's Equation on Irregular Domains", JCP, 160(1):151-178, Liu, Fedkiw, Kand, 2000);
   */
  inline double GFM_mu_jump_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, const double& mu_this_side, const double& mu_across) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_mu_jump(mu_this_side, mu_across);
  }

  /*!
   * \brief GFM_jump_terms_for_flux_component_between_cells evaluates the jump terms contributing to the (one-sided) flux components close to the interface
   * \param [in] quad_idx           local index of the cell of interest (cumulative over the local trees) [must be a local cell]
   * \param [in] neighbor_quad_idx  local index of its neighbor cell across the interface (cumulative over the local trees) [may be a ghost cell]
   * \param [in] oriented_dir       oriented cartesian direction in which the neighbor cell is, as seen from the cell of interest
   * \param [in] mu_this_side       value of the diffusion coefficient as seen from the cell of interest
   * \param [in] mu_across          value of the diffusion coefficient as seen from the neighbor cell acoss the interface
   * \param [in] in_positive_domain flag indicating if the cell of interest is in the positive domain (true) or in the negative domain (false)
   * \param [in] jump_field_p       pointer to (constant) node-sampled values of the jump on the field of interest
   *                                (must be sampled on the nodes of the interpolation_node_ngbd)
   * \param [in] jump_flux_p        pointer to (constant) node-sampled values of the jump in flux components for the field of interest
   *                                (must be P4EST_DIM-block structued and sampled on the nodes of the interpolation_node_ngbd)
   * \return the value of jump_terms such that the relevant flux component between the cell of interest and its neighbor cell (therefore along
   *          the cartesian direction oriented_dir/2), as seen from the cell of interest (not across the interface) can be evaluated as
   *  flux component = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(\hat{mu}*(value on neighbor_cell - value on cell of interest)/dxyz_min[oriented_dir/2] + jump_terms)
   * --> relevant for adding to RHS when assembling linear systems...
   */
  inline double GFM_jump_terms_for_flux_component_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir,
                                                                const double& mu_this_side, const double& mu_across, const bool& in_positive_domain,
                                                                const double* jump_field_p, const double* jump_flux_p, const double& xgfm_flux_correction) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, oriented_dir, in_positive_domain, jump_field_p, jump_flux_p, xgfm_flux_correction, dxyz_min);
  }

  /*!
   * \brief GFM_interface_value_between_cells evaluates the GFM-consistently interface-defined value given a cell-sampled solution field
   * and jump conditions
   * \param [in] quad_idx                     local index of the cell of interest (cumulative over the local trees) [must be a local cell]
   * \param [in] neighbor_quad_idx            local index of its neighbor cell across the interface (cumulative over the local trees) [may be a ghost cell]
   * \param [in] oriented_dir                 oriented cartesian direction in which the neighbor cell is, as seen from the cell of interest
   * \param [in] mu_this_side                 value of the diffusion coefficient as seen from the cell of interest
   * \param [in] mu_across                    value of the diffusion coefficient as seen from the neighbor cell acoss the interface
   * \param [in] in_positive_domain           flag indicating if the cell of interest is in the positive domain (true) or in the negative domain (false)
   * \param [in] get_positive_interface_value flag indicating if the user wants the positive or negative interface-defined value
   * \param [in] solution_p                   a pointer to the (constant) cell-sampled solution field
   * \param [in] jump_field_p                 pointer to (constant) node-sampled values of the jump on the field of interest
   *                                          (must be sampled on the nodes of the interpolation_node_ngbd)
   * \param [in] jump_flux_p                  pointer to (constant) node-sampled values of the jump in flux components for the field of interest
   *                                          (must be P4EST_DIM-block structued and sampled on the nodes of the interpolation_node_ngbd)
   * \return the desired interface-defined value
   * --> essential, key concept for xGFM strategy
   */
  inline double GFM_interface_value_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir,
                                                  const double& mu_this_side, const double mu_across, const bool& in_positive_domain, const bool& get_positive_interface_value,
                                                  const double* solution_p, const double* jump_field_p, const double* jump_flux_p, const double& xgfm_flux_correction) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_interface_defined_value(mu_this_side, mu_across, oriented_dir, in_positive_domain, get_positive_interface_value, solution_p[quad_idx], solution_p[neighbor_quad_idx], jump_field_p, jump_flux_p, xgfm_flux_correction, dxyz_min);
  }

  /*!
   * \brief GFM_flux_at_face_between_cells evaluate the appropriate, one-sided relevant flux component at the face separating two cells
   * across the interface
   * \param [in] quad_idx                     local index of the cell of interest (cumulative over the local trees) [must be a local cell]
   * \param [in] neighbor_quad_idx            local index of its neighbor cell across the interface (cumulative over the local trees) [may be a ghost cell]
   * \param [in] oriented_dir                 oriented cartesian direction in which the neighbor cell is, as seen from the cell of interest
   * \param [in] mu_this_side                 value of the diffusion coefficient as seen from the cell of interest
   * \param [in] mu_across                    value of the diffusion coefficient as seen from the neighbor cell acoss the interface
   * \param [in] in_positive_domain           flag indicating if the cell of interest is in the positive domain (true) or in the negative domain (false)
   * \param [in] face_is_on_this_side         flag indicating if the face of interest is on the same side of (true) or acrpss (false) the interface as the cell of interest
   * \param [in] solution_p                   a pointer to the (constant) cell-sampled solution field
   * \param [in] jump_field_p                 pointer to (constant) node-sampled values of the jump on the field of interest
   *                                          (must be sampled on the nodes of the interpolation_node_ngbd)
   * \param [in] jump_flux_p                  pointer to (constant) node-sampled values of the jump in flux components for the field of interest
   *                                          (must be P4EST_DIM-block structued and sampled on the nodes of the interpolation_node_ngbd)
   * \return the desired, one-sided, flux component, on the side of the interface where the face lies!
   */
  inline double GFM_flux_at_face_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, const double& mu_this_side, const double mu_across,
                                               const bool& in_positive_domain, const bool face_is_on_this_side, const double* solution_p,
                                               const double* jump_field_p, const double* jump_flux_p, const double& xgfm_flux_correction) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_flux_component_this_side(mu_this_side, mu_across, oriented_dir, in_positive_domain, solution_p[quad_idx], solution_p[neighbor_quad_idx],jump_field_p, jump_flux_p, xgfm_flux_correction, dxyz_min)
        + (face_is_on_this_side ? 0.0 : (in_positive_domain ? -1.0 : +1.0)*(interface_point.node_interpolant(jump_flux_p, oriented_dir/2, P4EST_DIM) + xgfm_flux_correction));
  }

  /*!
   * \brief get_cell_FD_interface_data self-explanatory
   * \return
   */
  inline const map_of_interface_neighbors_t& get_cell_FD_interface_data() const
  {
#ifdef CASL_THROWS
    if(cell_FD_interface_data == NULL)
      throw std::runtime_error("my_p4est_interface_manager_t::get_cell_FD_interface_data() called but the corresponding data is not stored...");
#endif
    return *cell_FD_interface_data;
  }

  /*!
   * \brief compute_subvolumes_in_cell self-explanatory
   * \param [in] quad_idx         local index of the quadrant of interest (cumulative over trees) [must be a local cell]
   * \param [in] tree_idx         index of the tree owning the quadrant of interest
   * \param [out] negative_volume surface, resp. volume, of the quadrant that is in the negative domain in 2D (resp. 3D)
   * \param [out] positive_volume surface, resp. volume, of the quadrant that is in the positive domain in 2D (resp. 3D) --> complement of the above
   * [Integration in subrefining quadrants is executed if subresolving interface-capturing is used]
   */
  void compute_subvolumes_in_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const;

  /*!
   * \brief phi evaluates the levelset value at a given point
   * \param [in] xyz coordinates of the point where the levelset value is desired
   * \return the levelset value
   */
  inline double phi(const double *xyz) const { return interp_phi(xyz); }

  /*!
   * \brief grad_phi evaluates the gradient of the levelset function at a given point
   * \param [in] xyz coordinates of the point where the gradient of the levelset function is desired
   * \param [out] grad_phi the computed gradient at xyz
   * [set_grad_phi, must have been called first!]
   */
  inline void grad_phi(const double *xyz, double* grad_phi) const
  {
#ifdef CASL_THROWS
    if(interp_grad_phi == NULL)
      throw std::runtime_error("my_p4est_interface_manager_t::grad_phi() called but interp_grad_phi is not available...");
#endif
    (*interp_grad_phi)(xyz, grad_phi);
    return;
  }

  /*!
   * \brief get_interface_capturing_ngbd_n self-explanatory
   * \return the interface-captruing node neighborgood info
   */
  inline const my_p4est_node_neighbors_t& get_interface_capturing_ngbd_n() const { return *interpolation_node_ngbd; }

  /*!
   * \brief get_grad_phi self-explanatory
   * \return the P4EST_DIM block-structured vector of node-sampled gradients of phi, sampled on the nodes of the interface-capturing grid
   * [set_grad_phi, must have been called first!]
   */
  inline Vec get_grad_phi() const
  {
#ifdef CASL_THROWS
    if(interp_grad_phi == NULL)
      throw std::runtime_error("my_p4est_interface_manager_t::get_grad_phi() called but interp_grad_phi is not available...");
#endif
    return interp_grad_phi->get_input_fields()[0];
  }

  /*!
   * \brief get_phi self-explanatory
   * \return the vector of node-sampled values of phi, sampled on the nodes of the interface-capturing grid
   */
  inline Vec get_phi() const
  {
#ifdef CASL_THROWS
    if(interp_phi.get_input_fields().size() < 1)
      throw std::runtime_error("my_p4est_interface_manager_t::get_phi() called but interp_phi is not set yet...");
#endif
    return interp_phi.get_input_fields()[0];
  }

  /*!
   * \brief get_phi_on_computational_nodes
   * \return the vector of node-sampled values of phi, sampled on the nodes of the computational grid (if available, returning NULL, otherwise)
   */
  inline Vec get_phi_on_computational_nodes() const
  {
    if(phi_on_computational_nodes != NULL)
      return phi_on_computational_nodes;
#ifdef CASL_THROWS
    std::cerr << "my_p4est_interface_manager_t::get_phi_on_computational_nodes() called but phi is not available on the computational nodes, returning NULL..." << std::endl;
#endif

    return NULL;
  }

  /*!
   * \brief subcell_resolution
   * \return the subresolution level, i.e., the difference in maximum refinement levle between the inteface-capturing grid and the computational grid
   */
  inline int subcell_resolution() const { return max_level_interpolation_p4est - max_level_p4est; }

#ifdef DEBUG
  int cell_FD_map_is_consistent_across_procs();
#endif
};

#endif // MY_P4EST_INTERFACE_MANAGER_H
