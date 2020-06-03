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
 * theta                  : fraction of the grid spacing covered by the domain in which the cell of interest is;
 * neighbor_quad_idx      : local index of the neighbor cell in the computational grid (across the interface);
 * mid_point_fine_node_idx: local index of the grid node in between those two cells on the
 *                          interface-capturing grid (if using subrefinement);
 * quad_fine_node_idx     : local index of the grid node that coincides with the center of the cell
 *                          of interest, on the interface-capturing grid (if using subrefinement);
 * neighbor_fine_node_idx : local index of the grid node that coincides with the center of the neighbor
 *                          cell across the interface, on the interface-capturing grid (if using subrefinement).
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
                                                  const double* jump_p, const double* jump_flux_p, const double* dxyz, const bool& evaluate_flux_on_this_side) const
  {
    GFM_jump_info jump_info = get_GFM_jump_data(jump_p, jump_flux_p, oriented_dir/2);

    return GFM_mu_jump(mu_this_side, mu_across)*(this_side_is_in_positive_domain ? +1.0 : -1.0)*
        (jump_info.jump_flux_component*(evaluate_flux_on_this_side ? (1 - theta)/mu_across : -theta/mu_this_side) + (oriented_dir%2 == 1 ? +1.0 : -1.0)*jump_info.jump_field/dxyz[oriented_dir/2]);
  }

  inline double GFM_flux_component(const double& mu_this_side, const double& mu_across, const u_char& oriented_dir, const bool &this_side_is_in_positive_domain,
                                   const double& solution_this_side, const double& solution_across,
                                   const double* jump_p, const double* jump_flux_p, const double* dxyz,
                                   const bool& evaluate_flux_on_this_side) const
  {
    return (oriented_dir%2 == 1 ? +1.0 : -1.0)*GFM_mu_jump(mu_this_side, mu_across)*(solution_across - solution_this_side)/dxyz[oriented_dir/2]
        + GFM_jump_terms_for_flux_component(mu_this_side, mu_across, oriented_dir, this_side_is_in_positive_domain, jump_p, jump_flux_p, dxyz, evaluate_flux_on_this_side);
  }

  inline double GFM_interface_defined_value(const double& mu_this_side, const double& mu_across, const u_char& oriented_dir, const bool &this_side_is_in_positive_domain, const bool& get_positive_interface_value,
                                            const double& solution_this_side, const double& solution_across,
                                            const double* jump_p, const double* jump_flux_p, const double* dxyz) const
  {
    GFM_jump_info jump_info = get_GFM_jump_data(jump_p, jump_flux_p, oriented_dir/2);

    return ((1.0 - theta)*mu_this_side*(solution_this_side  + (this_side_is_in_positive_domain != get_positive_interface_value ? (this_side_is_in_positive_domain ? -1.0 : +1.0)*jump_info.jump_field : 0.0))
            +      theta *mu_across   *(solution_across     + (this_side_is_in_positive_domain == get_positive_interface_value ? (this_side_is_in_positive_domain ? +1.0 : -1.0)*jump_info.jump_field : 0.0))
            + (this_side_is_in_positive_domain ? +1.0 : -1.0)*(oriented_dir%2 == 1 ? +1.0 : -1.0)*theta*(1.0 - theta)*dxyz[oriented_dir/2]*jump_info.jump_flux_component)/GFM_mu_tilde(mu_this_side, mu_across);
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
  const my_p4est_cell_neighbors_t *c_ngbd;
  const my_p4est_faces_t          *faces;
  const p4est_t                   *p4est;
  const p4est_ghost_t             *ghost;
  const double                    *dxyz_min;
  const my_p4est_node_neighbors_t *interpolation_node_ngbd;
  my_p4est_interpolation_nodes_t  interp_phi;
  my_p4est_interpolation_nodes_t  *interp_grad_phi;
  my_p4est_interpolation_nodes_t  *interp_phi_xxyyzz;
  Vec                             grad_phi_local;

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

  inline void clear() {
    clear_cell_FD_interface_data();
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      clear_face_FD_interface_data(dim);
  }

  const FD_interface_data& get_cell_FD_interface_data_for(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const;

public:
  my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const my_p4est_cell_neighbors_t* cell_ngbd, const double* dxyz_min_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_);

  inline my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_)
    : my_p4est_interface_manager_t(faces_, faces_->get_ngbd_c(), faces_->get_smallest_dxyz(), interpolation_node_ngbd_) { }

  ~my_p4est_interface_manager_t();

  inline void do_not_store_cell_FD_interface_data()
  {
    if(cell_FD_interface_data != NULL){
      delete cell_FD_interface_data;
      cell_FD_interface_data = NULL;
    }
    return;
  }

  inline void do_not_store_face_FD_interface_data()
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(face_FD_interface_data[dim] != NULL){
        delete face_FD_interface_data[dim];
        face_FD_interface_data[dim] = NULL;
      }
    }
    return;
  }

  inline bool is_storing_cell_FD_interface_data() const { return cell_FD_interface_data != NULL; }
  inline bool is_storing_face_FD_interface_data() const { return ANDD(face_FD_interface_data[0] != NULL, face_FD_interface_data[1] != NULL, face_FD_interface_data[2] != NULL); }

  // the interpolation method for phi is (or at least should be) irrelevant in presence of subrefinement, since all relevant points are sampled
  void set_levelset(Vec phi, const interpolation_method& method_interp_phi ONLY_WITH_SUBREFINEMENT(= linear), Vec phi_xxyyzz = NULL);
  void set_grad_phi();

  inline double get_FD_theta_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.theta;
  }

  inline double GFM_mu_jump_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, const double& mu_this_side, const double& mu_across) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_mu_jump(mu_this_side, mu_across);
  }

  inline double GFM_jump_terms_for_flux_component_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir,
                                                                const double& mu_this_side, const double& mu_across, const bool& in_positive_domain,
                                                                const double* jump_field_p, const double* jump_flux_p) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, oriented_dir, in_positive_domain, jump_field_p, jump_flux_p, dxyz_min, true);
  }

  inline double GFM_interface_value_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir,
                                                  const double& mu_this_side, const double mu_across, const bool& in_positive_domain, const bool& get_positive_interface_value,
                                                  const double* solution_p, const double* jump_field_p, const double* jump_flux_p) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_interface_defined_value(mu_this_side, mu_across, oriented_dir, in_positive_domain, get_positive_interface_value, solution_p[quad_idx], solution_p[neighbor_quad_idx], jump_field_p, jump_flux_p, dxyz_min);
  }

  inline double GFM_flux_at_face_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, const double& mu_this_side, const double mu_across,
                                               const bool& in_positive_domain, const bool face_is_on_this_side, const double& solution_quadrant, const double& solution_neighbor_quad,
                                               const double* jump_field_p, const double* jump_flux_p) const
  {
    const FD_interface_data& interface_point = get_cell_FD_interface_data_for(quad_idx, neighbor_quad_idx, oriented_dir);
    return interface_point.GFM_flux_component(mu_this_side, mu_across, oriented_dir, in_positive_domain, solution_quadrant, solution_neighbor_quad, jump_field_p, jump_flux_p, dxyz_min, face_is_on_this_side);
  }

  inline const map_of_interface_neighbors_t& get_cell_FD_interface_data() const { P4EST_ASSERT(cell_FD_interface_data != NULL); return *cell_FD_interface_data; }

  void compute_subvolumes_in_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const;

#ifdef DEBUG
  int cell_FD_map_is_consistent_across_procs();
#endif
};

#endif // MY_P4EST_INTERFACE_MANAGER_H
