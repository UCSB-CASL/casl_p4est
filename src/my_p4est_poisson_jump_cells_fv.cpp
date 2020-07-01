#ifdef P4_TO_P8
#include "my_p8est_poisson_jump_cells_fv.h"
#else
#include "my_p4est_poisson_jump_cells_fv.h"
#endif


// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_jump_cells_fv_solve_for_sharp_solution;
#endif


my_p4est_poisson_jump_cells_fv_t::my_p4est_poisson_jump_cells_fv_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t *nodes_)
  : my_p4est_poisson_jump_cells_t(ngbd_c, nodes_)
{
  correction_function_for_quad.clear();
  finite_volume_data_for_quad.clear();
  are_required_finite_volumes_and_correction_functions_known = false;
}

void my_p4est_poisson_jump_cells_fv_t::build_finite_volumes_and_correction_functions()
{
  if(are_required_finite_volumes_and_correction_functions_known)
    return;

  std::map<int, std::vector<global_correction_function_elementary_data_t> > serialized_global_correction_functions_to_send_to;
  serialized_global_correction_functions_to_send_to.clear();
  std::set<int> ranks_to_receive_correction_functions_from; ranks_to_receive_correction_functions_from.clear();
  int mpiret;
  std::vector<MPI_Request> nonblocking_send_requests;

  // loop through layer quadrants
  // 1) build correction functions where needed (i.e., if quadrant is crossed by the interface)
  // 2) check direct neighbors and
  //    a) check if any direct neighbor is ghost and required the specific correction function associated with this cell
  //       --> make the correction function global (local -> global indices), and serialize it for communication
  //    b) check if any direct neighbor is ghost and not on the same side of the domain
  //       --> expect a message from the process owning that ghost cell in such a case
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k) {
    const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k);
    const p4est_topidx_t tree_idx = cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k);

    bool is_face_crossed[P4EST_FACES];
    const bool is_quad_double_valued = interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, is_face_crossed);

#ifdef CASL_THROWS
    if(is_quad_double_valued && ANDD(!is_face_crossed[0] && !is_face_crossed[1], !is_face_crossed[2] && !is_face_crossed[3], !is_face_crossed[4] && !is_face_crossed[5]))
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_finite_volumes_and_correction_functions() : you're playing with fire here, a cell contains an enclosed region of the interface but none of its faces is actually crossed by the interface.");
#endif

    if(is_quad_double_valued)
      build_and_store_double_valued_info_for_quad_if_needed(quad_idx, tree_idx);

    // figure out if required by another process and or if this process expects communications from another one
    std::set<int> ranks_to_communicate_correction_function_with; ranks_to_communicate_correction_function_with.clear();
    const double *tree_xyz_min, *tree_xyz_max;
    const p4est_quadrant_t* quad;
    fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
    double xyz_quad[P4EST_DIM], dxyz_quad[P4EST_DIM];
    xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad, dxyz_quad);
    const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);
    for (u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
    {
      if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
        continue;
      set_of_neighboring_quadrants neighbors_across_face;
      cell_ngbd->find_neighbor_cells_of_cell(neighbors_across_face, quad_idx, tree_idx, oriented_dir);
      P4EST_ASSERT(neighbors_across_face.size() > 0);
      for (set_of_neighboring_quadrants::const_iterator it = neighbors_across_face.begin(); it != neighbors_across_face.end(); ++it) {
        const p4est_quadrant_t& neighbor_across_face = *it;
        if(neighbor_across_face.p.piggy3.local_num >= p4est->local_num_quadrants) // the neighbor is a ghost quad
        {
          double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[oriented_dir/2] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*dxyz_quad[oriented_dir/2];
          double xyz_neighbor[P4EST_DIM]; quad_xyz_fr_q(neighbor_across_face.p.piggy3.local_num, neighbor_across_face.p.piggy3.which_tree, p4est, ghost, xyz_neighbor);
          const char sgn_face     = (interface_manager->phi_at_point(xyz_face)      <= 0.0 ? -1 : +1);
          const char sgn_neighbor = (interface_manager->phi_at_point(xyz_neighbor)  <= 0.0 ? -1 : +1);
          const int rank_owning_neighbor = quad_find_ghost_owner(ghost, neighbor_across_face.p.piggy3.local_num - p4est->local_num_quadrants);
          if(is_quad_double_valued && (is_face_crossed[oriented_dir] || signs_of_phi_are_different(sgn_face, sgn_quad))) // you need to send the correction function you have just constructed to that one
            ranks_to_communicate_correction_function_with.insert(rank_owning_neighbor);
          if(is_face_crossed[oriented_dir] || signs_of_phi_are_different(sgn_face, sgn_neighbor)) // you expect a correction function for the ghost quad
            ranks_to_receive_correction_functions_from.insert(rank_owning_neighbor);
        }
      }
    }
    P4EST_ASSERT(is_quad_double_valued || ranks_to_communicate_correction_function_with.size() == 0); // nothing to send if you haven't constructed anything --> sanity check
    if(is_quad_double_valued && ranks_to_communicate_correction_function_with.size() > 0) // you gotta send stuff
    {
      P4EST_ASSERT(correction_function_for_quad.find(quad_idx) != correction_function_for_quad.end()); // make sure it is known and accessible
      // serialize your message(s)
      const correction_function_t& correction_function = correction_function_for_quad[quad_idx];
      global_correction_function_elementary_data_t data_to_send;
      for (std::set<int>::const_iterator it = ranks_to_communicate_correction_function_with.begin(); it != ranks_to_communicate_correction_function_with.end(); ++it)
      {
        const int& receiver_rank = *it;
        data_to_send.quad_global_idx            = p4est->global_first_quadrant[p4est->mpirank] + quad_idx;  serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
        data_to_send.jump_dependent_terms       = correction_function.jump_dependent_terms;                 serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
        data_to_send.n_solution_dependent_terms = correction_function.solution_dependent_terms.size();      serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
        for (size_t k = 0; k < correction_function.solution_dependent_terms.size(); ++k) {
          data_to_send.solution_dependent_term_global_index = compute_global_index_of_quad(correction_function.solution_dependent_terms[k].dof_idx, p4est, ghost);
          serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
          data_to_send.solution_dependent_term_weight       = correction_function.solution_dependent_terms[k].weight;
          serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
        }
      }
    }
  }

  // Send (nonblocking) the (global) serialized correction functions to the relevant neighbor processes
  for (std::map<int, std::vector<global_correction_function_elementary_data_t> >::const_iterator it = serialized_global_correction_functions_to_send_to.begin();
       it != serialized_global_correction_functions_to_send_to.end(); ++it){
    const int& rank = it->first;
    MPI_Request req;
    mpiret = MPI_Isend(serialized_global_correction_functions_to_send_to[rank].data(),
                       serialized_global_correction_functions_to_send_to[rank].size()*sizeof (global_correction_function_elementary_data_t),
                       MPI_BYTE, rank, correction_function_communication_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    nonblocking_send_requests.push_back(req);
  }

  // loop through inner quadrants, build correction functions where needed (nothing else to do there)
  for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k) {
    const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k);
    const p4est_topidx_t tree_idx = cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k);
    bool is_face_crossed[P4EST_FACES];
    if(interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, is_face_crossed))
    {
#ifdef CASL_THROWS
      if(ANDD(!is_face_crossed[0] && !is_face_crossed[1], !is_face_crossed[2] && !is_face_crossed[3], !is_face_crossed[4] && !is_face_crossed[5]))
        throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_finite_volumes_and_correction_functions() : you're playing with fire here, a cell contains an enclosed region of the interface but none of its faces is actually crossed by the interface.");
#endif
      build_and_store_double_valued_info_for_quad_if_needed(quad_idx, tree_idx);
    }
  }

  // Receive (blocking) the serialized (global) correction functions from the relevant neighbors
  // 1) deserialize the message;
  // 2) make the correction functions local (global -> local indices)
  // 3) insert in map
  while (ranks_to_receive_correction_functions_from.size() > 0) {
    int is_msg_pending;
    MPI_Status status;
    mpiret = MPI_Iprobe(MPI_ANY_SOURCE, correction_function_communication_tag, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
    if(is_msg_pending)
    {
      P4EST_ASSERT(ranks_to_receive_correction_functions_from.find(status.MPI_SOURCE) != ranks_to_receive_correction_functions_from.end());
      int byte_count;
      mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
      P4EST_ASSERT(byte_count%sizeof (global_correction_function_elementary_data_t) == 0);

      std::vector<global_correction_function_elementary_data_t> received_serialized_global_correction_functions;
      received_serialized_global_correction_functions.resize(byte_count/sizeof (global_correction_function_elementary_data_t));
      mpiret = MPI_Recv(received_serialized_global_correction_functions.data(),
                        byte_count, MPI_BYTE, status.MPI_SOURCE, correction_function_communication_tag, p4est->mpicomm, MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

      // deserialize the message and add it to the local map of correction functions
      size_t running_idx = 0;
      while (running_idx < received_serialized_global_correction_functions.size()) {
        const p4est_locidx_t local_quad_idx = find_local_index_of_quad(received_serialized_global_correction_functions[running_idx++].quad_global_idx, p4est, ghost);
        P4EST_ASSERT(local_quad_idx >= p4est->local_num_quadrants); // must be a ghost quadrant!
        P4EST_ASSERT(correction_function_for_quad.find(local_quad_idx) == correction_function_for_quad.end()); // must not be in there yet
        correction_function_t correction_function_for_ghost_quad;
        correction_function_for_ghost_quad.jump_dependent_terms = received_serialized_global_correction_functions[running_idx++].jump_dependent_terms;
        const size_t n_solution_dependent_terms = received_serialized_global_correction_functions[running_idx++].n_solution_dependent_terms;
        for (size_t k = 0; k < n_solution_dependent_terms; ++k){
          const p4est_locidx_t local_idx_of_term = find_local_index_of_quad(received_serialized_global_correction_functions[running_idx++].solution_dependent_term_global_index, p4est, ghost);
          const double weight_for_term = received_serialized_global_correction_functions[running_idx++].solution_dependent_term_weight;
          correction_function_for_ghost_quad.solution_dependent_terms.add_term(local_idx_of_term, weight_for_term);
        }
        correction_function_for_quad.insert(std::pair<p4est_locidx_t, correction_function_t>(local_quad_idx, correction_function_for_ghost_quad));
      }
      // remove the source from the set of expected messengers
      ranks_to_receive_correction_functions_from.erase(status.MPI_SOURCE);
    }
  }

  mpiret = MPI_Waitall(nonblocking_send_requests.size(), nonblocking_send_requests.data(), MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  are_required_finite_volumes_and_correction_functions_known = true;
}

void my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx)
{
  if(correction_function_for_quad.find(quad_idx) != correction_function_for_quad.end() &&
     finite_volume_data_for_quad.find(quad_idx) != finite_volume_data_for_quad.end())
    return;

#ifdef CASL_THROWS
  if(quad_idx < 0 || quad_idx > p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() cannot be called for nonlocal quadrants");
#endif

  // construct finite volume info
  finite_volume_data_for_quad.insert(std::pair<p4est_locidx_t, my_p4est_finite_volume_t>(quad_idx, interface_manager->get_finite_volume_for_quad(quad_idx, tree_idx)));

  // construct correction function
  const p4est_quadrant_t* quad;
  const double *tree_xyz_min, *tree_xyz_max;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
#ifdef CASL_THROWS
  if(quad->level != interface_manager->get_max_level_computational_grid())
    throw std::logic_error("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() : called on a quadrant that is not as fine as expected.");
#endif

  // get projected point onto the interface, signed distance, normal vector at projected point
  double xyz_quad[P4EST_DIM], xyz_quad_projected[P4EST_DIM], normal_at_projected_point[P4EST_DIM];
  xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad);
  const double signed_distance = interface_manager->signed_distance_at_point(xyz_quad);
  const char sgn_quad = (signed_distance <= 0.0 ? -1  :+1);
  interface_manager->projection_onto_interface_of_point(xyz_quad, xyz_quad_projected);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(fabs(xyz_quad[dim] - xyz_quad_projected[dim]) > 1.5*dxyz_min[dim]) // this is considered an essential runtime error --> throw even if CASL_THROWS is not defined...
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() : the projection of a quad center onto the interface lies out of a 3 by 3 (by 3) surrounding box");

  clip_in_domain(xyz_quad_projected, xyz_min, xyz_max, periodicity); // --> clip it back in domain (periodic wrapping if periodicity allows it)

  const double jump_field       = (*interp_jump_u)(xyz_quad_projected);
  const double jump_normal_flux = (*interp_jump_normal_flux)(xyz_quad_projected);
  const double &mu_fast         = (mu_minus > mu_plus ? mu_minus  : mu_plus);
  const double &mu_slow         = (mu_minus > mu_plus ? mu_plus   : mu_minus);
  double mu_across_normal_derivative = mu_fast;

  double scaling_factor = 1.0;  // either 1.0 (i.e. no scaling required) if quad center is on the same side as the evaluatio of normal derivative or if jump in mu is 0.0
                                // otherwise, inverse of (1.0 +/- signed_distance*mu_jump*(weight of the normal_derivative_at_projected_point operator, for the quad of interest)/mu_across_normal_derivative)

  linear_combination_of_dof_t* one_sided_normal_derivative_at_projected_point = (mus_are_equal() ? NULL : new linear_combination_of_dof_t); // we need that only if there is a nonzero jump in mu

  if (!mus_are_equal())
  {
    P4EST_ASSERT(one_sided_normal_derivative_at_projected_point != NULL);
    interface_manager->normal_vector_at_point(xyz_quad_projected, normal_at_projected_point);
    // fetch relevant neighbors to build slow-diffusion-sided derivatives
    p4est_quadrant_t quad_with_piggy3 = *quad;
    quad_with_piggy3.p.piggy3.local_num = quad_idx; quad_with_piggy3.p.piggy3.which_tree = tree_idx;
    set_of_neighboring_quadrants all_first_degree_neighbors;
    p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = cell_ngbd->gather_neighbor_cells_of_cell(quad_with_piggy3, all_first_degree_neighbors);

    set_of_neighboring_quadrants first_degree_neighbors_in_slow_side, first_degree_neighbors_in_fast_side;
    first_degree_neighbors_in_slow_side.insert(quad_with_piggy3); // this one is double-valued --> always add it to the stencil
    first_degree_neighbors_in_fast_side.insert(quad_with_piggy3); // this one is double-valued --> always add it to the stencil
    for (set_of_neighboring_quadrants::const_iterator it = all_first_degree_neighbors.begin(); it != all_first_degree_neighbors.end(); ++it) {
      if(it->p.piggy3.local_num == quad_idx)
        continue;

      double xyz_neighbor_quad[P4EST_DIM]; quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_neighbor_quad);
      const char sgn_neighbor_quad = (interface_manager->phi_at_point(xyz_neighbor_quad) <= 0.0 ? -1 : +1);
      if(is_point_in_slow_side(sgn_neighbor_quad))
        first_degree_neighbors_in_slow_side.insert(*it);
      else
        first_degree_neighbors_in_fast_side.insert(*it);
    }

    const bool use_slow_side = (first_degree_neighbors_in_slow_side.size() >= 1 + P4EST_DIM); // --> take the slow side if possible
    mu_across_normal_derivative                                     = (use_slow_side ? mu_fast : mu_slow);
    const set_of_neighboring_quadrants& one_sided_neighbors_to_use  = (use_slow_side ? first_degree_neighbors_in_slow_side : first_degree_neighbors_in_fast_side);
    P4EST_ASSERT(one_sided_neighbors_to_use.size() >= 1 + P4EST_DIM);

    const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;

    linear_combination_of_dof_t lsqr_cell_grad_operator_on_slow_side_at_projected_point[P4EST_DIM];
    get_lsqr_cell_gradient_operator_at_point(xyz_quad_projected, cell_ngbd, one_sided_neighbors_to_use, scaling_distance, lsqr_cell_grad_operator_on_slow_side_at_projected_point);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      one_sided_normal_derivative_at_projected_point->add_operator_on_same_dofs(lsqr_cell_grad_operator_on_slow_side_at_projected_point[dim], normal_at_projected_point[dim]);


    if(use_slow_side != is_point_in_slow_side(sgn_quad))
    {
      double* quad_weight_for_normal_derivative_on_slow_side_at_projected_point = NULL;
      for (size_t k = 0; k < one_sided_normal_derivative_at_projected_point->size(); ++k)
        if((*one_sided_normal_derivative_at_projected_point)[k].dof_idx == quad_idx)
        {
          quad_weight_for_normal_derivative_on_slow_side_at_projected_point = &(*one_sided_normal_derivative_at_projected_point)[k].weight;
          break;
        }
      P4EST_ASSERT(quad_weight_for_normal_derivative_on_slow_side_at_projected_point != NULL);
      const double lhs_coeff = (1.0 + (sgn_quad <= 0 ? +1.0 : -1.0)*signed_distance*get_jump_in_mu()*(*quad_weight_for_normal_derivative_on_slow_side_at_projected_point)/mu_across_normal_derivative);
#ifdef CASL_THROWS
      if(fabs(lhs_coeff) < EPS)
        throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() : the denominator is ill-defined in the definition of on correction function");
#endif
      scaling_factor = 1.0/lhs_coeff;
    }
  }

  correction_function_t correction_function_to_build;
  correction_function_to_build.jump_dependent_terms = scaling_factor*(jump_field + signed_distance*jump_normal_flux/mu_across_normal_derivative);
  if(one_sided_normal_derivative_at_projected_point != NULL)
    for (size_t k = 0; k < one_sided_normal_derivative_at_projected_point->size(); ++k)
      correction_function_to_build.solution_dependent_terms.add_term((*one_sided_normal_derivative_at_projected_point)[k].dof_idx,
                                                                     -scaling_factor*(signed_distance*get_jump_in_mu()/mu_across_normal_derivative)*(*one_sided_normal_derivative_at_projected_point)[k].weight);

  correction_function_for_quad.insert(std::pair<p4est_locidx_t, correction_function_t>(quad_idx, correction_function_to_build));

  if(one_sided_normal_derivative_at_projected_point != NULL)
    delete one_sided_normal_derivative_at_projected_point;

  return;
}

void my_p4est_poisson_jump_cells_fv_t::get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                                        PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const
{
  const p4est_quadrant_t *quad;
  const double *tree_xyz_min, *tree_xyz_max;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
  double xyz_quad[P4EST_DIM], cell_dxyz[P4EST_DIM];
  xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad, cell_dxyz);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);

  std::set<p4est_locidx_t> local_quad_indices_involved;
  local_quad_indices_involved.insert(quad_idx); // this quad goes in, for sure
  bool is_face_crossed[P4EST_FACES];
  const bool is_quad_crossed = interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, is_face_crossed);
  if(is_quad_crossed)
  {
#ifdef CASL_THROWS
    if(correction_function_for_quad.find(quad_idx) == correction_function_for_quad.end())
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::get_numbers_of_cells_involved_in_equation_for_quad() \n couldn't find the correction function for local quad "
                               + std::to_string(quad_idx) + " on proc " + std::to_string(p4est->mpirank) + ", located at (" + std::to_string(xyz_quad[0]) + ", " + std::to_string(xyz_quad[1]) ONLY3D(+ ", " + std::to_string(xyz_quad[1])) + ")");
#endif
    const correction_function_t& correction_function = correction_function_for_quad.at(quad_idx);
    for (size_t k = 0; k < correction_function.solution_dependent_terms.size(); ++k)
      local_quad_indices_involved.insert(correction_function.solution_dependent_terms[k].dof_idx);
  }

  for(u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
  {
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
      continue;
    double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[oriented_dir/2] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[oriented_dir/2];
    const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : 1);

    set_of_neighboring_quadrants direct_neighbors;
    bool are_all_cell_centers_on_same_side;
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, are_all_cell_centers_on_same_side);
    if(is_face_crossed[oriented_dir] || (!are_all_cell_centers_on_same_side && !signs_of_phi_are_different(sgn_face, sgn_quad))) // the neighbor's correction function kicks in
    {
      P4EST_ASSERT(direct_neighbors.size() == 1);
      const p4est_quadrant_t& direct_neighbor = *direct_neighbors.begin();
      P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid() && quad->level == direct_neighbor.level);
#ifdef CASL_THROWS
      if(correction_function_for_quad.find(direct_neighbor.p.piggy3.local_num) == correction_function_for_quad.end())
        throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::get_numbers_of_cells_involved_in_equation_for_quad() \n couldn't find the correction function for direct neighbor quad " + std::to_string(direct_neighbor.p.piggy3.local_num)
                                 + ", neighbor in direction " + std::to_string(oriented_dir) + " of quad " + std::to_string(quad_idx)  + " located at (" + std::to_string(xyz_quad[0]) +  ", " + std::to_string(xyz_quad[1]) ONLY3D( + ", " + std::to_string(xyz_quad[2])) + "),  on proc " + std::to_string(p4est->mpirank) + " (local partition has " + std::to_string(p4est->local_num_quadrants) + " quadrants).");
#endif

      local_quad_indices_involved.insert(direct_neighbor.p.piggy3.local_num);
      const correction_function_t& correction_function_of_direct_neighbor = correction_function_for_quad.at(direct_neighbor.p.piggy3.local_num);
      for (size_t k = 0; k < correction_function_of_direct_neighbor.solution_dependent_terms.size(); ++k)
        local_quad_indices_involved.insert(correction_function_of_direct_neighbor.solution_dependent_terms[k].dof_idx);
    }
    else
      for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k)
        local_quad_indices_involved.insert(stable_projection_derivative_operator[k].dof_idx);
  }

  number_of_local_cells_involved = 0;
  number_of_ghost_cells_involved = 0;
  for (std::set<p4est_locidx_t>::const_iterator it = local_quad_indices_involved.begin(); it != local_quad_indices_involved.end(); ++it) {
    if(*it < p4est->local_num_quadrants)  // locally owned
      number_of_local_cells_involved++;
    else                                  // ghost
      number_of_ghost_cells_involved++;
  }

  P4EST_ASSERT(number_of_local_cells_involved > 0); // should always contain the current cell at least!

  return;
}

void my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int *nullspace_contains_constant_vector)
{
  PetscErrorCode ierr;
  const double *user_rhs_minus_p  = NULL;
  const double *user_rhs_plus_p   = NULL;
  double       *rhs_p             = NULL;

  if(!rhs_is_set)
  {
    if(user_rhs_minus != NULL){
      ierr = VecGetArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr); }
    if(user_rhs_plus != NULL){
      ierr = VecGetArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr); }
    if(user_vstar_minus != NULL || user_vstar_plus != NULL)
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad : not able to handle vstar as rhs, yet --> implement that now, please");
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  }

  const p4est_quadrant_t *quad;
  const double *tree_xyz_min, *tree_xyz_max;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
  double xyz_quad[P4EST_DIM], cell_dxyz[P4EST_DIM];
  xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad, cell_dxyz);
  const double cell_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);
  const double &mu_this_side = (sgn_quad > 0 ? mu_plus : mu_minus);
  const PetscInt quad_gloidx = compute_global_index(quad_idx);

  const my_p4est_finite_volume_t* finite_volume_of_quad                   = NULL;
  const correction_function_t*    correction_function_of_quad             = NULL;
  bool is_face_crossed[P4EST_FACES];
  const bool is_quad_crossed = interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, is_face_crossed);
  if(is_quad_crossed)
  {
    if(ANDD(!is_face_crossed[0] && !is_face_crossed[1], !is_face_crossed[2] && !is_face_crossed[3], !&it_corr_fun->seconis_face_crossed[4] && !is_face_crossed[5]))
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() : you're playing with fire here, a cell contains an enclosed region of the interface but none of its faces is actually crossed by the interface.");

#ifdef CASL_THROWS
    if(correction_function_for_quad.find(quad_idx) == correction_function_for_quad.end())
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() couldn't find the correction function for local quad " + std::to_string(quad_idx) + " on proc " + std::to_string(p4est->mpirank));
    if(finite_volume_data_for_quad.find(quad_idx) == finite_volume_data_for_quad.end())
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() couldn't find the finite volume data for local quad " + std::to_string(quad_idx) + " on proc " + std::to_string(p4est->mpirank));
#endif
    correction_function_of_quad = &correction_function_for_quad.at(quad_idx);
    finite_volume_of_quad       = &finite_volume_data_for_quad.at(quad_idx);
  }

  /* First add the diagonal terms */
  const bool nonzero_diag_term = (is_quad_crossed ? MAX(fabs(add_diag_minus), fabs(add_diag_plus)) : (sgn_quad > 0 ? fabs(add_diag_plus) : fabs(add_diag_minus))) > EPS;
  if(!matrix_is_set && nonzero_diag_term)
  {
    if(is_quad_crossed){
      ierr = MatSetValue(A, quad_gloidx, quad_gloidx,
                         finite_volume_of_quad->volume_in_negative_domain()*add_diag_minus + finite_volume_of_quad->volume_in_positive_domain()*add_diag_plus, ADD_VALUES); CHKERRXX(ierr); }
    else {
      ierr = MatSetValue(A, quad_gloidx, quad_gloidx,
                         cell_volume*(sgn_quad > 0 ? add_diag_plus : add_diag_minus), ADD_VALUES); CHKERRXX(ierr); }
    if(nullspace_contains_constant_vector != NULL)
      *nullspace_contains_constant_vector = 0;
  }
  if(!rhs_is_set)
  {
    if(is_quad_crossed)
    {
      P4EST_ASSERT(finite_volume_of_quad->interfaces.size() <= 1); // could be 0.0 if only one point is 0.0 and the rest are > 0.0
      rhs_p[quad_idx] = finite_volume_of_quad->volume_in_negative_domain()*user_rhs_minus_p[quad_idx] + finite_volume_of_quad->volume_in_positive_domain()*user_rhs_plus_p[quad_idx];
      for (size_t k = 0; k < finite_volume_of_quad->interfaces.size(); ++k)
      {
        const double xyz_interface_quadrature[P4EST_DIM] = {DIM(xyz_quad[0] + finite_volume_of_quad->interfaces[k].centroid[0], xyz_quad[1] + finite_volume_of_quad->interfaces[k].centroid[1], xyz_quad[2] + finite_volume_of_quad->interfaces[k].centroid[2])};
        rhs_p[quad_idx] -= finite_volume_of_quad->interfaces[k].area*(*interp_jump_normal_flux)(xyz_interface_quadrature);
      }
    }
    else
      rhs_p[quad_idx] = (sgn_quad < 0 ? user_rhs_minus_p[quad_idx] : user_rhs_plus_p[quad_idx])*cell_volume;
  }

  for(u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
  {
    const double full_face_area = cell_volume/cell_dxyz[oriented_dir/2];
    double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[oriented_dir/2] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[oriented_dir/2];
    const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : 1);

    /* First check if the cell is a wall.
     * We will assume that walls are not crossed by the interface and are on the same side, in a first attempt! */
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
    {
#ifdef CASL_THROWS
      if(is_face_crossed[oriented_dir] || signs_of_phi_are_different(sgn_quad, sgn_quad))
        throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() : a wall-face is crossed by the interface, this is not handled yet...");
#endif
      switch(bc->wallType(xyz_face))
      {
      case DIRICHLET:
      {
        if(!matrix_is_set)
        {
          if(nullspace_contains_constant_vector != NULL)
            *nullspace_contains_constant_vector = 0;
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu_this_side*full_face_area/cell_dxyz[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
          rhs_p[quad_idx]  += 2.0*mu_this_side*full_face_area*bc->wallValue(xyz_face)/cell_dxyz[oriented_dir/2];
      }
        break;
      case NEUMANN:
      {
        if(!rhs_is_set)
          rhs_p[quad_idx]  += mu_this_side*full_face_area*bc->wallValue(xyz_face);
      }
        break;
      default:
        throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() : unknown boundary condition on a wall.");
      }
      continue;
    }
    // not a wall in that direction
    set_of_neighboring_quadrants direct_neighbors;
    bool are_all_cell_centers_on_same_side;
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, are_all_cell_centers_on_same_side);
    /* Two possible scenarii :
     * 1) two local equations to consider (and sum up), which happens if
     *    a) the face is crossed --> no trivial (i.e. identically 0.0) term in either equation to consider;
     *    b) the face is not crossed but at least one of its neighbor centers has a different sign --> only one equation has nontrivial (i.e. nonzero) terms, but at least one correction function comes into play
     *    --> this could be situations of this kind        OR        of this kind
     *                   (I)                                              (II)
     *      |     \   |       |        |                     |     \   |    |  |        |
     *      |------\--|-------|--------|                     |------\--|----|--|--------|
     *      |       | |       |        |                     |       | |+  /   |        |
     *      |    - /  |   +   |   +    |                     |    - /  |+ /-   |   -    |
     *      |     /   |       |        |                     |     /   |+|     |        |
     *      |----/----|-------|--------|                     |----/----|--\----|--------|
     *      |    |    |                |                     |    |    |   \   |        |
     *      |    | <- interface                                   |         \
     *           |                                                |<- interface
     * 2) regular discretization : the face is not crossed, all relevant cell centers are on the same side, and so is the relevant face
     *    --> no correction function comes into play
     * */
    if(is_face_crossed[oriented_dir]                      // case 1) a)
       || !are_all_cell_centers_on_same_side              // case 1) b) I)
       || signs_of_phi_are_different(sgn_quad, sgn_face)) // case 1) b) II)
    {
      // some correction function(s) must kick in!
      if(direct_neighbors.size() > 1 || direct_neighbors.begin()->level < quad->level)
      {
        // this would happen in cases like the following --> while it might be possible to address such issues immediately in 2D,
        // more work would be required to address it generally in 3D (the correction function of the neighbor may not be available, as such)
        // |     \   |                |
        // |------\--|----------------|
        // |       | |                |
        // |    - /  |                |
        // |     /   |                |
        // |----/----|       +        |
        // |   /     |                |
        // |  | +    |                |
        // |  /      |                |
        // |-/-------|----------------|
        // | |       |                |
        //   | <- interface
        throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() : the stable-projection derivative operator for quad "
                                 + std::to_string(quad_idx) + " at (probably uncrossed but shared) face of orientation " + std::to_string(oriented_dir) +
                                 " on proc " + std::to_string(p4est->mpirank) + " involves quadrants lying on the other side of the domain.");
      }
      P4EST_ASSERT(direct_neighbors.size() == 1);
      const p4est_quadrant_t& direct_neighbor = *direct_neighbors.begin();
      P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid() && quad->level == direct_neighbor.level);
      const char sgn_direct_neighbor = (are_all_cell_centers_on_same_side ? +1 : -1)*sgn_quad;

      const correction_function_t* correction_function_of_direct_neighbor = NULL;
      if(is_face_crossed[oriented_dir] || signs_of_phi_are_different(sgn_face, sgn_direct_neighbor))
      {
#ifdef CASL_THROWS
        if(correction_function_for_quad.find(direct_neighbor.p.piggy3.local_num) == correction_function_for_quad.end())
          throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad() couldn't find the correction function for direct neighbor quad " + std::to_string(direct_neighbor.p.piggy3.local_num)
                                   + ", neighbor of quad " + std::to_string(quad_idx) + " in direction " + std::to_string(oriented_dir)
                                   + " on proc " + std::to_string(p4est->mpirank) + " (local partition has " + std::to_string(p4est->local_num_quadrants) + " quadrants).");
#endif
        correction_function_of_direct_neighbor = &correction_function_for_quad.at(direct_neighbor.p.piggy3.local_num);
      }
      // make sure we have access to all we need right away:
      P4EST_ASSERT((is_face_crossed[oriented_dir] && finite_volume_of_quad != NULL && correction_function_of_quad != NULL && correction_function_of_direct_neighbor != NULL)
                   || (!is_face_crossed[oriented_dir] && (!signs_of_phi_are_different(sgn_quad, sgn_face) || correction_function_of_quad != NULL) && (!signs_of_phi_are_different(sgn_face, sgn_direct_neighbor) || correction_function_of_direct_neighbor != NULL)));
      for (char sgn_eqn = (is_face_crossed[oriented_dir] ? -1 : sgn_face); sgn_eqn < (is_face_crossed[oriented_dir] ? 1 : sgn_face) + 1; sgn_eqn += 2) { // sum up all nontrivial equation terms
        const double face_area = (is_face_crossed[oriented_dir] ? (sgn_eqn < 0 ? finite_volume_of_quad->face_area_in_negative_domain(oriented_dir) : finite_volume_of_quad->face_area_in_positive_domain(oriented_dir)) : full_face_area);

        // loop over the two terms, play with references, +/-1 prefactor for this quad/direct neighbor and that's it
        for (u_char dof = 0; dof < 2; ++dof) { // dof == 0 : this quadrant; dof == 1 : the direct neighbor
          const double coeff_dof                = (dof == 0 ? +1.0 : -1.0)*(sgn_eqn < 0 ? mu_minus : mu_plus)*face_area/dxyz_min[oriented_dir/2];
          const p4est_gloidx_t& global_idx_dof  = (dof == 0 ? quad_gloidx : compute_global_index(direct_neighbor.p.piggy3.local_num));
          const char& sgn_dof                   = (dof == 0 ? sgn_quad : sgn_direct_neighbor);

          if(!matrix_is_set){
            ierr = MatSetValue(A, quad_gloidx, global_idx_dof, coeff_dof, ADD_VALUES); CHKERRXX(ierr); }

          if(signs_of_phi_are_different(sgn_dof, sgn_eqn)) // the terms of correction_function_of_quad must kick in
          {
            const correction_function_t* &correction_function_dof = (dof == 0 ? correction_function_of_quad : correction_function_of_direct_neighbor);
            P4EST_ASSERT(correction_function_dof != NULL);
            if(!rhs_is_set)
              rhs_p[quad_idx] -= coeff_dof*sgn_eqn*correction_function_dof->jump_dependent_terms;
            if(!matrix_is_set)
              for (size_t k = 0; k < correction_function_dof->solution_dependent_terms.size(); ++k) {
                ierr = MatSetValue(A, quad_gloidx, compute_global_index(correction_function_dof->solution_dependent_terms[k].dof_idx),
                                   +sgn_eqn*coeff_dof*correction_function_dof->solution_dependent_terms[k].weight, ADD_VALUES); CHKERRXX(ierr); }
          }
        }
      }
    }
    else
    {
      P4EST_ASSERT(!is_face_crossed[oriented_dir] && are_all_cell_centers_on_same_side && !signs_of_phi_are_different(sgn_face, sgn_quad)); // --> standard discretization
      if(!matrix_is_set)
        for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k) {
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(stable_projection_derivative_operator[k].dof_idx),
                             (oriented_dir%2 == 1 ? -1.0 : +1.0)*mu_this_side*full_face_area*stable_projection_derivative_operator[k].weight, ADD_VALUES); CHKERRXX(ierr);
        }
    }
  }

  if(!rhs_is_set)
  {
    if(user_rhs_minus_p != NULL){
      ierr = VecRestoreArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr); }
    if(user_rhs_plus_p != NULL){
      ierr = VecRestoreArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr); }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  }

  return;
}


double my_p4est_poisson_jump_cells_fv_t::get_sharp_flux_component_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, char& sgn_face) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(f_idx, dim, quad_idx, tree_idx);
  const u_char oriented_dir = 2*dim + (faces->q2f(quad_idx, 2*dim) == f_idx ? 0 : + 1);
  const double *tree_xyz_min, *tree_xyz_max;
  const p4est_quadrant_t* quad;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
  double xyz_quad[P4EST_DIM], cell_dxyz[P4EST_DIM];
  xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad, cell_dxyz);
  double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[dim] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[dim];
  sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : 1);
  const char sgn_q      = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);
  const double mu_face  = (sgn_face > 0 ? mu_plus : mu_minus);
  PetscErrorCode ierr;
  const double *solution_p;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);

  double sharp_flux_component;

  if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
  {
    P4EST_ASSERT(f_idx != NO_VELOCITY);
#ifdef CASL_THROWS
    if(signs_of_phi_are_different(sgn_q, sgn_face ))
      throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::get_sharp_flux_component_local(): a wall-cell is crossed by the interface, this is not handled yet...");
#endif
    switch(bc->wallType(xyz_face))
    {
    case DIRICHLET:
      sharp_flux_component = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(2.0*mu_face*(bc->wallValue(xyz_face) - solution_p[quad_idx])/cell_dxyz[dim]);
      break;
    case NEUMANN:
      sharp_flux_component = (oriented_dir%2 == 1 ? +1.0 : -1.0)*mu_face*bc->wallValue(xyz_face);
      break;
    default:
      throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::get_flux_components_and_subtract_them_from_velocities_local(): unknown boundary condition on a wall.");
    }
  }
  else
  {
    set_of_neighboring_quadrants direct_neighbors;
    bool one_sided;
    linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, one_sided);

    if(one_sided && !signs_of_phi_are_different(sgn_q, sgn_face))
      sharp_flux_component = mu_face*stable_projection_derivative(solution_p);
    else
    {
      P4EST_ASSERT(direct_neighbors.size() == 1);
      const p4est_quadrant_t& direct_neighbor = *direct_neighbors.begin();
      P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid() && quad->level == direct_neighbor.level);
      const double &mu_face           = (sgn_face  < 0 ? mu_minus  : mu_plus);
      const char sgn_direct_neighbor  = (one_sided ? 1 : -1)*sgn_q;

      sharp_flux_component = 0.0;
      for (u_char dof = 0; dof < 2; ++dof) { // dof == 0 : this quadrant; dof == 1 : the direct neighbor
        const double coeff_dof        = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(dof == 0 ? -1.0 : +1.0)*mu_face/dxyz_min[oriented_dir/2];
        const p4est_locidx_t& idx_dof = (dof == 0 ? quad_idx  : direct_neighbor.p.piggy3.local_num);
        const char& sgn_dof           = (dof == 0 ? sgn_q     : sgn_direct_neighbor);

        sharp_flux_component += coeff_dof*solution_p[idx_dof];
        if(signs_of_phi_are_different(sgn_face, sgn_dof))
        {
#ifdef CASL_THROWS
          if(correction_function_for_quad.find(idx_dof) == correction_function_for_quad.end())
            throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::get_sharp_flux_component_local couldn't find the correction function for quad " + std::to_string(idx_dof)
                                     + ", required for the calculation of flux at face " + std::to_string(f_idx) + " of Cartesian normal " + std::to_string(dim) + ", found between quads "
                                     + std::to_string(quad_idx) + " and " + std::to_string(direct_neighbor.p.piggy3.local_num) + " on proc " + std::to_string(p4est->mpirank)
                                     + " (local partition has " + std::to_string(p4est->local_num_quadrants) + " quadrants).");
#endif
          const correction_function_t& correction_function_dof = correction_function_for_quad.at(idx_dof);
          sharp_flux_component += coeff_dof*(sgn_face > 0 ? +1.0 : -1.0)*correction_function_dof(solution_p);
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);

  return sharp_flux_component;
}

void my_p4est_poisson_jump_cells_fv_t::solve_for_sharp_solution(const KSPType &ksp_type, const PCType& pc_type)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_fv_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  // make sure the problem is fully defined
  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2])); // boundary conditions
  P4EST_ASSERT(diffusion_coefficients_have_been_set() && interface_is_set());       // essential parameters
  P4EST_ASSERT(((user_rhs_minus != NULL && user_rhs_plus != NULL) || (user_vstar_minus != NULL && user_vstar_plus != NULL)) && jumps_have_been_set()); // rhs fully determined

  // build required finite volumes and correction functions
  if(!are_required_finite_volumes_and_correction_functions_known)
    build_finite_volumes_and_correction_functions();

  /* Set the linear system, the linear solver and solve it */
  setup_linear_system();
  ierr = setup_linear_solver(ksp_type, pc_type); CHKERRXX(ierr);
  solve_linear_system();
  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_fv_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  return;

}
