#include "my_p4est_poisson_jump_cells_fv.h"

my_p4est_poisson_jump_cells_fv::my_p4est_poisson_jump_cells_fv(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_)
  : my_p4est_poisson_jump_cells_t(ngbd_c, nodes_)
{
//  phi_quad = NULL;
  correction_function_for_quad.clear();
  finite_volume_data_for_quad.clear();
  finite_volumes_and_correction_functions_are_known = false;
}

void my_p4est_poisson_jump_cells_fv::build_finite_volumes_and_correction_functions()
{
  if(finite_volumes_and_correction_functions_are_known)
    return;
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k) {

  }








  finite_volumes_and_correction_functions_are_known = true;
}

void my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx)
{
  if(correction_function_for_quad.find(quad_idx) != correction_function_for_quad.end() &&
     finite_volume_data_for_quad.find(quad_idx) != finite_volume_data_for_quad.end())
    return;

#ifdef CASL_THROWS
  if(quad_idx < 0 || quad_idx > p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed() cannot be called for nonlocal quadrants");
#endif

  // construct finite volume info
  my_p4est_finite_volume_t fv_to_build;
  const int order = (interface_manager->get_interpolation_method_for_phi() == linear ? 1 : 2);
  construct_finite_volume(fv_to_build, quad_idx, tree_idx, p4est, interface_manager->get_phi_as_local_cf(), order, interface_manager->subcell_resolution(), true);
  finite_volume_data_for_quad.insert(std::pair<p4est_locidx_t, my_p4est_finite_volume_t>(quad_idx, fv_to_build));

  // construct correction function
  const p4est_quadrant_t* quad;
  const double *tree_xyz_min, *tree_xyz_max;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
#ifdef CASL_THROWS
  if(quad->level != interface_manager->get_max_level_computational_grid())
    throw std::logic_error("my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed() : called on a quadrant that is not as fine as expected.");
#endif

  // get projected point onto the interface, signed distance, normal vector at projected point
  double xyz_quad[P4EST_DIM], xyz_quad_projected[P4EST_DIM], normal_at_projected_point[P4EST_DIM];
  xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad);
  const double signed_distance = interface_manager->signed_distance_at_point(xyz_quad);
  interface_manager->projection_onto_interface_of_point(xyz_quad, xyz_quad_projected);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(fabs(xyz_quad[dim] - xyz_quad_projected[dim]) > 1.5*dxyz_min[dim]) // this is considered an essential runtime error --> throw even if CASL_THROWS is not defined...
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed() : the projection of a quad center onto th interface lies out of a 3 by 3 (by 3) surrounding box");
    clip_in_domain(xyz_quad_projected, p4est);
  }

  linear_combination_of_dof_t* normal_derivative_on_slow_side_at_projected_point = (mus_are_equal() ? NULL : new linear_combination_of_dof_t); // we need that only if there is indeed a jump in mu

  if (normal_derivative_on_slow_side_at_projected_point != NULL)
  {
    interface_manager->normal_vector_at_point(xyz_quad_projected, normal_at_projected_point);
    // fetch relevant neighbors to build slow-diffusion-sided derivatives
    p4est_quadrant_t quad_with_piggy3 = *quad;
    quad_with_piggy3.p.piggy3.local_num = quad_idx; quad_with_piggy3.p.piggy3.which_tree = tree_idx;
    set_of_neighboring_quadrants all_first_degree_neighbors;
    p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = cell_ngbd->gather_neighbor_cells_of_cell(quad_with_piggy3, all_first_degree_neighbors);

    set_of_neighboring_quadrants first_degree_neighbors_in_slow_side;
    first_degree_neighbors_in_slow_side.insert(quad_with_piggy3); // this one is double-valued --> always add it to the stencil
    for (set_of_neighboring_quadrants::const_iterator it = all_first_degree_neighbors.begin(); it != all_first_degree_neighbors.end(); ++it) {
      if(it->p.piggy3.local_num == quad_idx)
        continue;

      double xyz_neighbor_quad[P4EST_DIM]; quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_neighbor_quad);
      if(is_point_in_slow_side(xyz_neighbor_quad)) // add it to stencil only if in slow side, for sure
        first_degree_neighbors_in_slow_side.insert(*it);
    }

    if(first_degree_neighbors_in_slow_side.size() < 1 + P4EST_DIM)
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed() not enough neighbor in the slow side");

    const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;

    linear_combination_of_dof_t lsqr_cell_grad_operator_on_slow_side_at_projected_point[P4EST_DIM];
    get_lsqr_cell_gradient_operator_at_point(xyz_quad_projected, cell_ngbd, first_degree_neighbors_in_slow_side,  scaling_distance, lsqr_cell_grad_operator_on_slow_side_at_projected_point);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      normal_derivative_on_slow_side_at_projected_point->add_operator_on_same_dofs(lsqr_cell_grad_operator_on_slow_side_at_projected_point[dim], normal_at_projected_point[dim]);
  }

  const double jump_field       = (*interp_jump_u)(xyz_quad_projected);
  const double jump_normal_flux = (*interp_jump_normal_flux)(xyz_quad_projected);
  const double &fast_mu         = (mu_minus > mu_plus ? mu_minus  : mu_plus);

  double scaling_factor = 1.0;  // either 1.0 (i.e. no scaling required) if quad center is in slow side or if jump_mu is 0.0
                                // otherwise, inverse of (1.0 +/- signed_distance*mu_jump*(weight of the normal_derivative_on_slow_side_at_projected_point operator, for the quad of interest)/fast_mu)
  if(!mus_are_equal() && !is_point_in_slow_side(xyz_quad))
  {
    P4EST_ASSERT(normal_derivative_on_slow_side_at_projected_point != NULL);
    double* quad_weight_for_normal_derivative_on_slow_side_at_projected_point = NULL;
    for (size_t k = 0; k < normal_derivative_on_slow_side_at_projected_point->size(); ++k)
      if((*normal_derivative_on_slow_side_at_projected_point)[k].dof_idx == quad_idx)
      {
        quad_weight_for_normal_derivative_on_slow_side_at_projected_point = &(*normal_derivative_on_slow_side_at_projected_point)[k].weight;
        break;
      }
    P4EST_ASSERT(quad_weight_for_normal_derivative_on_slow_side_at_projected_point != NULL);
    const double lhs_coeff = (1.0 + (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? +1.0 : -1.0)*signed_distance*get_jump_in_mu()*(*quad_weight_for_normal_derivative_on_slow_side_at_projected_point)/fast_mu);
#ifdef CASL_THROWS
    if(fabs(lhs_coeff) < EPS)
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv::build_and_store_double_valued_info_for_quad_if_needed() : the denominator is ill-defined in the definition of on correction function");
#endif
    scaling_factor = 1.0/lhs_coeff;
  }

  correction_function_t correction_function_to_build;
  correction_function_to_build.jump_dependent_terms = scaling_factor*(jump_field + signed_distance*jump_normal_flux/fast_mu);
  if(normal_derivative_on_slow_side_at_projected_point != NULL)
    for (size_t k = 0; k < normal_derivative_on_slow_side_at_projected_point->size(); ++k)
      correction_function_to_build.solution_dependent_terms.add_term((*normal_derivative_on_slow_side_at_projected_point)[k].dof_idx,
                                                                     -scaling_factor*(signed_distance*get_jump_in_mu()/fast_mu)*(*normal_derivative_on_slow_side_at_projected_point)[k].weight);

  correction_function_for_quad.insert(std::pair<p4est_locidx_t, correction_function_t>(quad_idx, correction_function_to_build));

  if(normal_derivative_on_slow_side_at_projected_point != NULL)
    delete normal_derivative_on_slow_side_at_projected_point;

  return;
}

