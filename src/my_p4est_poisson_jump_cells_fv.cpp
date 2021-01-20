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
  jump_operators_for_viscous_terms_on_quad.clear();
  are_required_finite_volumes_and_correction_functions_known = false;
  interface_relative_threshold = +1.0e-11;
  threshold_volume_ratio_for_extrapolation = 0.01;
  reference_face_area = pow(ABSD(dxyz_min[0], dxyz_min[1], dxyz_min[2]), P4EST_DIM - 1);
  pin_normal_derivative_for_correction_functions = false;
  scale_system_by_diagonals = true;

  local_corr_fun_for_layer_quad.clear();
  local_corr_fun_for_inner_quad.clear();
  local_corr_fun_for_ghost_quad.clear();
  offset_corr_fun_on_proc.assign(p4est->mpisize + 1, 0);
  global_idx_of_ghost_corr_fun.clear();
  jump_dependent_terms_in_corr_fun = NULL;
}

my_p4est_poisson_jump_cells_fv_t::~my_p4est_poisson_jump_cells_fv_t()
{
  PetscErrorCode ierr = delete_and_nullify_vector(jump_dependent_terms_in_corr_fun); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_cells_fv_t::clear_node_sampled_jumps()
{
  my_p4est_poisson_jump_cells_t::clear_node_sampled_jumps();
  // clearing the jumps nullifies the current solution (done at virtual parent level),
  // its extensions, the current residual and the xgfm jump values:
  for(map_of_correction_functions_t::iterator it = correction_function_for_quad.begin();
      it != correction_function_for_quad.end(); it++)
    it->second.jump_dependent_terms = 0.0;
}

void my_p4est_poisson_jump_cells_fv_t::update_jump_terms_for_projection()
{
  PetscErrorCode ierr;
  P4EST_ASSERT(set_for_projection_steps);
  if(jump_dependent_terms_in_corr_fun == NULL)
  {
    ierr = VecCreateGhostCellCorrFun(&jump_dependent_terms_in_corr_fun); CHKERRXX(ierr);
  }
  double *jump_dependent_terms_in_corr_fun_p;
  ierr = VecGetArray(jump_dependent_terms_in_corr_fun, &jump_dependent_terms_in_corr_fun_p); CHKERRXX(ierr);

  const double *face_velocity_plus_km1_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};  P4EST_ASSERT(face_velocity_plus_km1 == NULL   || (ANDD(face_velocity_plus_km1[0]  != NULL,  face_velocity_plus_km1[1]  != NULL, face_velocity_plus_km1[2]  != NULL)));
  const double *face_velocity_minus_km1_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};  P4EST_ASSERT(face_velocity_minus_km1 == NULL  || (ANDD(face_velocity_minus_km1[0] != NULL,  face_velocity_minus_km1[1] != NULL, face_velocity_minus_km1[2] != NULL)));
  const double *face_velocity_plus_p[P4EST_DIM]       = {DIM(NULL, NULL, NULL)};  P4EST_ASSERT(face_velocity_plus == NULL       || (ANDD(face_velocity_plus[0] != NULL,       face_velocity_plus[1] != NULL,      face_velocity_plus[2] != NULL     )));
  const double *face_velocity_minus_p[P4EST_DIM]      = {DIM(NULL, NULL, NULL)};  P4EST_ASSERT(face_velocity_minus == NULL      || (ANDD(face_velocity_minus[0] != NULL,      face_velocity_minus[1] != NULL,     face_velocity_minus[2] != NULL    )));
  for(u_char dim = 0; dim < P4EST_DIM; dim++)
  {
    if(face_velocity_plus_km1 != NULL){
      ierr = VecGetArrayRead(face_velocity_plus_km1[dim], &face_velocity_plus_km1_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_minus_km1 != NULL){
      ierr = VecGetArrayRead(face_velocity_minus_km1[dim], &face_velocity_minus_km1_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_plus != NULL){
      ierr = VecGetArrayRead(face_velocity_plus[dim], &face_velocity_plus_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_minus != NULL){
      ierr = VecGetArrayRead(face_velocity_minus[dim], &face_velocity_minus_p[dim]); CHKERRXX(ierr);
    }
  }

  for(map_of_local_quad_to_corr_fun_t::const_iterator it = local_corr_fun_for_layer_quad.begin();
      it != local_corr_fun_for_layer_quad.end(); it++)
  {
    P4EST_ASSERT(jump_operators_for_viscous_terms_on_quad.find(it->first) != jump_operators_for_viscous_terms_on_quad.end());
    const differential_operators_on_face_sampled_field& viscous_term_operators = jump_operators_for_viscous_terms_on_quad.at(it->first);

    double new_jump  = dt_over_BDF_alpha*shear_viscosity_plus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_km1_p) + viscous_term_operators.divergence(face_velocity_plus_p));
    new_jump        -= dt_over_BDF_alpha*shear_viscosity_minus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_km1_p) + viscous_term_operators.divergence(face_velocity_minus_p));
    new_jump *= viscous_term_operators.FV_corr_function_scaling_for_jump_dependent_terms;

    jump_dependent_terms_in_corr_fun_p[it->second] = new_jump;
    correction_function_for_quad[it->first].jump_dependent_terms = new_jump;
  }
  ierr = VecGhostUpdateBegin(jump_dependent_terms_in_corr_fun, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(map_of_local_quad_to_corr_fun_t::const_iterator it = local_corr_fun_for_inner_quad.begin();
      it != local_corr_fun_for_inner_quad.end(); it++)
  {
    P4EST_ASSERT(jump_operators_for_viscous_terms_on_quad.find(it->first) != jump_operators_for_viscous_terms_on_quad.end());
    const differential_operators_on_face_sampled_field& viscous_term_operators = jump_operators_for_viscous_terms_on_quad.at(it->first);

    double new_jump  = dt_over_BDF_alpha*shear_viscosity_plus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_km1_p) + viscous_term_operators.divergence(face_velocity_plus_p));
    new_jump        -= dt_over_BDF_alpha*shear_viscosity_minus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_km1_p) + viscous_term_operators.divergence(face_velocity_minus_p));
    new_jump *= viscous_term_operators.FV_corr_function_scaling_for_jump_dependent_terms;

    jump_dependent_terms_in_corr_fun_p[it->second] = new_jump;
    correction_function_for_quad[it->first].jump_dependent_terms = new_jump;
  }
  ierr = VecGhostUpdateEnd(jump_dependent_terms_in_corr_fun, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(u_char dim = 0; dim < P4EST_DIM; dim++)
  {
    if(face_velocity_plus_km1 != NULL){
      ierr = VecRestoreArrayRead(face_velocity_plus_km1[dim], &face_velocity_plus_km1_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_minus_km1 != NULL){
      ierr = VecRestoreArrayRead(face_velocity_minus_km1[dim], &face_velocity_minus_km1_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_plus != NULL){
      ierr = VecRestoreArrayRead(face_velocity_plus[dim], &face_velocity_plus_p[dim]); CHKERRXX(ierr);
    }
    if(face_velocity_minus != NULL){
      ierr = VecRestoreArrayRead(face_velocity_minus[dim], &face_velocity_minus_p[dim]); CHKERRXX(ierr);
    }
  }

  // now that ghost values have been synchronized, update the jump terms in correction functions associated with ghost cells as well:
  for(map_of_local_quad_to_corr_fun_t::const_iterator it = local_corr_fun_for_ghost_quad.begin();
      it != local_corr_fun_for_ghost_quad.end(); it++)
  {
    P4EST_ASSERT(correction_function_for_quad.find(it->first) != correction_function_for_quad.end());
    correction_function_for_quad[it->first].jump_dependent_terms = jump_dependent_terms_in_corr_fun_p[it->second];
  }

  ierr = VecRestoreArray(jump_dependent_terms_in_corr_fun, &jump_dependent_terms_in_corr_fun_p); CHKERRXX(ierr);
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
      build_and_store_double_valued_info_for_quad_if_needed(quad_idx, tree_idx, &local_corr_fun_for_layer_quad);

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
        data_to_send.using_fast_side = correction_function.not_reliable_for_extrapolation; serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
        data_to_send.local_corr_fun_idx = local_corr_fun_for_layer_quad[quad_idx]; serialized_global_correction_functions_to_send_to[receiver_rank].push_back(data_to_send);
      }
    }
  }

  // Send (nonblocking) the (global) serialized correction functions to the relevant neighbor processes
  for (std::map<int, std::vector<global_correction_function_elementary_data_t> >::const_iterator it = serialized_global_correction_functions_to_send_to.begin();
       it != serialized_global_correction_functions_to_send_to.end(); ++it){
    const int& rank = it->first;
    MPI_Request req;
    mpiret = MPI_Isend(it->second.data(), it->second.size()*sizeof (global_correction_function_elementary_data_t),
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
      build_and_store_double_valued_info_for_quad_if_needed(quad_idx, tree_idx, &local_corr_fun_for_inner_quad);
    }
  }
  P4EST_ASSERT(local_corr_fun_for_inner_quad.size() + local_corr_fun_for_layer_quad.size() == correction_function_for_quad.size()); // make sure we didn't miss one in there
  // get global offsets per processor for correction functions
  int n_corr_fun = correction_function_for_quad.size();
  mpiret = MPI_Allgather(&n_corr_fun, 1, MPI_INT, (void *) (&offset_corr_fun_on_proc[1]), 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret); // offet of 1, this is normal!
  for(int r = 1; r < p4est->mpisize + 1; r++)
    offset_corr_fun_on_proc[r] += offset_corr_fun_on_proc[r - 1];

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
        correction_function_for_ghost_quad.not_reliable_for_extrapolation = received_serialized_global_correction_functions[running_idx++].using_fast_side;

        global_idx_of_ghost_corr_fun.push_back(received_serialized_global_correction_functions[running_idx++].local_corr_fun_idx + offset_corr_fun_on_proc[status.MPI_SOURCE]);
        local_corr_fun_for_ghost_quad.insert(std::pair<p4est_locidx_t, size_t>(local_quad_idx, correction_function_for_quad.size()));
        correction_function_for_quad.insert(std::pair<p4est_locidx_t, correction_function_t>(local_quad_idx, correction_function_for_ghost_quad));
      }
      P4EST_ASSERT(global_idx_of_ghost_corr_fun.size() == local_corr_fun_for_ghost_quad.size()); // make sure we didn't miss one
      // remove the source from the set of expected messengers
      ranks_to_receive_correction_functions_from.erase(status.MPI_SOURCE);
    }
  }

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &use_extrapolations_in_sharp_flux_calculations, 1, MPI_CXX_BOOL, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(nonblocking_send_requests.size(), nonblocking_send_requests.data(), MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  are_required_finite_volumes_and_correction_functions_known = true;
}

void my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, map_of_local_quad_to_corr_fun_t* map_quad_to_cf)
{
  if(correction_function_for_quad.find(quad_idx) != correction_function_for_quad.end() &&
     finite_volume_data_for_quad.find(quad_idx) != finite_volume_data_for_quad.end())
    return;

#ifdef CASL_THROWS
  if(quad_idx < 0 || quad_idx > p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() cannot be called for nonlocal quadrants");
#endif

  // construct finite volume info
  finite_volume_data_for_quad[quad_idx] = interface_manager->get_finite_volume_for_quad(quad_idx, tree_idx);

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
  const double signed_distance = interface_manager->signed_distance_and_projection_onto_interface_of_point(xyz_quad, xyz_quad_projected);
  const char sgn_quad = (signed_distance <= 0.0 ? -1  :+1);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    xyz_quad_projected[dim] = MAX(xyz_quad[dim] - 1.5*dxyz_min[dim], MIN(xyz_quad[dim] + 1.5*dxyz_min[dim], xyz_quad_projected[dim])); // we need to be able to process this locally...

  clip_in_domain(xyz_quad_projected, xyz_min, xyz_max, periodicity); // --> clip it back in domain (periodic wrapping if periodicity allows it)

  const double jump_field       = (interp_jump_u != NULL           ? (*interp_jump_u)(xyz_quad_projected)           : 0.0);
  const double jump_normal_flux = (interp_jump_normal_flux != NULL ? (*interp_jump_normal_flux)(xyz_quad_projected) : 0.0);
  const double &mu_fast         = (mu_minus > mu_plus ? mu_minus  : mu_plus);
  const double &mu_slow         = (mu_minus > mu_plus ? mu_plus   : mu_minus);
  double mu_across_normal_derivative = mu_fast;

  double scaling_factor = 1.0;  // either 1.0 (i.e. no scaling required) if quad center is on the same side as the evaluatio of normal derivative or if jump in mu is 0.0
                                // otherwise, inverse of (1.0 +/- signed_distance*mu_jump*(weight of the normal_derivative_at_projected_point operator, for the quad of interest)/mu_across_normal_derivative)

  linear_combination_of_dof_t* one_sided_normal_derivative_at_projected_point = (mus_are_equal() ? NULL : new linear_combination_of_dof_t); // we need that only if there is a nonzero jump in mu
  correction_function_t correction_function_to_build;
  correction_function_to_build.not_reliable_for_extrapolation = ((sgn_quad < 0 ? finite_volume_data_for_quad[quad_idx].volume_in_positive_domain() : finite_volume_data_for_quad[quad_idx].volume_in_negative_domain()) < threshold_volume_ratio_for_extrapolation*finite_volume_data_for_quad[quad_idx].full_cell_volume);
  if(correction_function_to_build.not_reliable_for_extrapolation)
    use_extrapolations_in_sharp_flux_calculations = true;

  if (!mus_are_equal())
  {
    P4EST_ASSERT(one_sided_normal_derivative_at_projected_point != NULL);
    const double *xyz_for_normal_derivative = (pin_normal_derivative_for_correction_functions ? xyz_quad : xyz_quad_projected);
    interface_manager->normal_vector_at_point(xyz_for_normal_derivative, normal_at_projected_point);
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

    bool use_slow_side = true; // --> take the slow side if possible
    const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
    linear_combination_of_dof_t lsqr_cell_grad_operator_on_slow_side_at_projected_point[P4EST_DIM];
    try {
      get_lsqr_cell_gradient_operator_at_point(xyz_for_normal_derivative, cell_ngbd, first_degree_neighbors_in_slow_side, scaling_distance, lsqr_cell_grad_operator_on_slow_side_at_projected_point, pin_normal_derivative_for_correction_functions, quad_idx);
    } catch (std::exception e) { // it couldn't be done on the slow side
      use_slow_side = false;
      correction_function_to_build.not_reliable_for_extrapolation = true;
      use_extrapolations_in_sharp_flux_calculations = true;
      get_lsqr_cell_gradient_operator_at_point(xyz_for_normal_derivative, cell_ngbd, first_degree_neighbors_in_fast_side, scaling_distance, lsqr_cell_grad_operator_on_slow_side_at_projected_point, pin_normal_derivative_for_correction_functions, quad_idx); // this should work if the other didn't (hopefully, otherwise, we're screwed)
    }

    mu_across_normal_derivative = (use_slow_side ? mu_fast : mu_slow);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      one_sided_normal_derivative_at_projected_point->add_operator_on_same_dofs(lsqr_cell_grad_operator_on_slow_side_at_projected_point[dim], normal_at_projected_point[dim]);

    if(use_slow_side != is_point_in_slow_side(sgn_quad))
    {
      const double* quad_weight_for_normal_derivative_at_projected_point = NULL;
      for (size_t k = 0; k < one_sided_normal_derivative_at_projected_point->size(); ++k)
        if((*one_sided_normal_derivative_at_projected_point)[k].dof_idx == quad_idx)
        {
          quad_weight_for_normal_derivative_at_projected_point = &(*one_sided_normal_derivative_at_projected_point)[k].weight;
          break;
        }
      P4EST_ASSERT(quad_weight_for_normal_derivative_at_projected_point != NULL);
      const double lhs_coeff = (1.0 + (sgn_quad <= 0 ? +1.0 : -1.0)*signed_distance*get_jump_in_mu()*(*quad_weight_for_normal_derivative_at_projected_point)/mu_across_normal_derivative);
#ifdef CASL_THROWS
      if(fabs(lhs_coeff) < EPS)
        throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::build_and_store_double_valued_info_for_quad_if_needed() : the denominator is ill-defined in the definition of on correction function");
#endif
      scaling_factor = 1.0/lhs_coeff;
    }
  }

  correction_function_to_build.jump_dependent_terms = scaling_factor*(jump_field + signed_distance*jump_normal_flux/mu_across_normal_derivative);
  if(one_sided_normal_derivative_at_projected_point != NULL)
    for (size_t k = 0; k < one_sided_normal_derivative_at_projected_point->size(); ++k)
      correction_function_to_build.solution_dependent_terms.add_term((*one_sided_normal_derivative_at_projected_point)[k].dof_idx,
                                                                     -scaling_factor*(signed_distance*get_jump_in_mu()/mu_across_normal_derivative)*(*one_sided_normal_derivative_at_projected_point)[k].weight);

  if(is_coupled_to_two_phase_flow())
  {
    if(jump_operators_for_viscous_terms_on_quad.find(quad_idx) == jump_operators_for_viscous_terms_on_quad.end())
    {
      // not found in map --> build it and insert in map
      differential_operators_on_face_sampled_field to_insert_in_map;
      const my_p4est_faces_t* faces = interface_manager->get_faces();

      set_of_neighboring_quadrants nb_quads;
      p4est_quadrant_t qq = *quad;
      qq.p.piggy3.which_tree = tree_idx; qq.p.piggy3.local_num = quad_idx;
      p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = cell_ngbd->gather_neighbor_cells_of_cell(qq, nb_quads, true); // include second-degree neighbors, because YOLO!
      logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, cell_ngbd->gather_neighbor_cells_of_cell(qq, nb_quads));
      const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
      std::set<indexed_and_located_face> set_of_neighbor_faces[P4EST_DIM];
      add_all_faces_to_sets_and_clear_set_of_quad(faces, set_of_neighbor_faces, nb_quads);
      const bool grad_is_known[P4EST_DIM] = {DIM(false, false, false)};
      linear_combination_of_dof_t gradient_of_face_sampled_field[P4EST_DIM][P4EST_DIM]; // we need the gradient of all vector components
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        get_lsqr_face_gradient_at_point(xyz_quad_projected, faces, set_of_neighbor_faces[comp], scaling_distance, gradient_of_face_sampled_field[comp], NULL, grad_is_known);

      if(mus_are_equal()) // normal_at_projected_point is not known yet if mus are equal...
      {
        const double *xyz_for_normal_derivative = (pin_normal_derivative_for_correction_functions ? xyz_quad : xyz_quad_projected);
        interface_manager->normal_vector_at_point(xyz_for_normal_derivative, normal_at_projected_point);
      }

      for(u_char comp = 0; comp < P4EST_DIM; comp++)
      {
        for(u_char der = 0; der < P4EST_DIM; der++)
          to_insert_in_map.n_dot_grad_dot_n_operator[comp].add_operator_on_same_dofs(gradient_of_face_sampled_field[comp][der], normal_at_projected_point[comp]*normal_at_projected_point[der]);

        // the divergence operator is build at the cell center, as per in the stable projection in this case
        // (the correction function is associated with a specific quadrant in this approach)
        to_insert_in_map.div_term[comp].clear();
        const double dx = (tree_xyz_max[comp] - tree_xyz_min[comp])*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));

        for(u_char face = 0; face < 2; face++)
        {
          set_of_neighboring_quadrants direct_neighbor;
          cell_ngbd->find_neighbor_cells_of_cell(direct_neighbor, quad_idx, tree_idx, 2*comp + face);
          P4EST_ASSERT(direct_neighbor.size() <= 1); // otherwise it means the current quadrant is not as fine as it should be, it's under-refined...
          if(direct_neighbor.size() == 1 && direct_neighbor.begin()->level < quad->level) // should not happen, hopefully, but who knows, we'd better be careful in this world...
          {
            set_of_neighboring_quadrants minor_quads;
            cell_ngbd->find_neighbor_cells_of_cell(minor_quads, direct_neighbor.begin()->p.piggy3.local_num, direct_neighbor.begin()->p.piggy3.which_tree, 2*comp + (face == 0 ? 1 : 0));
            P4EST_ASSERT(minor_quads.size() > 1);
            for(const p4est_quadrant_t& minor_quad : minor_quads)
              to_insert_in_map.div_term[comp].add_term(faces->q2f(minor_quad.p.piggy3.local_num, 2*comp + face),
                                                       (face == 0 ? -1.0 : +1.0)*pow(2.0, (double)(direct_neighbor.begin()->level - minor_quad.level)*(P4EST_DIM - 1))/dx);
          }
          else
            to_insert_in_map.div_term[comp].add_term(faces->q2f(quad_idx, 2*comp + face), (face == 0 ? -1.0 : +1.0)/dx);
        }
      }
      to_insert_in_map.FV_corr_function_scaling_for_jump_dependent_terms = scaling_factor; // VERY IMPORTANT
      jump_operators_for_viscous_terms_on_quad.insert(std::pair<p4est_locidx_t, differential_operators_on_face_sampled_field>(quad_idx, to_insert_in_map));
    }
    const differential_operators_on_face_sampled_field& viscous_term_operators = jump_operators_for_viscous_terms_on_quad.at(quad_idx);

    PetscErrorCode ierr;
    const double *face_velocity_plus_km1_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
    const double *face_velocity_minus_km1_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
    const double *face_velocity_plus_p[P4EST_DIM]       = {DIM(NULL, NULL, NULL)};
    const double *face_velocity_minus_p[P4EST_DIM]      = {DIM(NULL, NULL, NULL)};
    for(u_char dim = 0; dim < P4EST_DIM; dim++)
    {
      if(face_velocity_plus_km1 != NULL){
        ierr = VecGetArrayRead(face_velocity_plus_km1[dim], &face_velocity_plus_km1_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_minus_km1 != NULL){
        ierr = VecGetArrayRead(face_velocity_minus_km1[dim], &face_velocity_minus_km1_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_plus != NULL && set_for_projection_steps){
        ierr = VecGetArrayRead(face_velocity_plus[dim], &face_velocity_plus_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_minus != NULL && set_for_projection_steps){
        ierr = VecGetArrayRead(face_velocity_minus[dim], &face_velocity_minus_p[dim]); CHKERRXX(ierr);
      }
    }

    if(!set_for_projection_steps)
      correction_function_to_build.jump_dependent_terms += scaling_factor*(shear_viscosity_plus*viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_km1_p) - shear_viscosity_minus*viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_km1_p));
    else
    {
      correction_function_to_build.jump_dependent_terms += scaling_factor*dt_over_BDF_alpha*shear_viscosity_plus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_plus_km1_p) + viscous_term_operators.divergence(face_velocity_plus_p));
      correction_function_to_build.jump_dependent_terms -= scaling_factor*dt_over_BDF_alpha*shear_viscosity_minus*(viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_p) - viscous_term_operators.n_dot_grad_dot_n(face_velocity_minus_km1_p) + viscous_term_operators.divergence(face_velocity_minus_p));
    }

    for(u_char dim = 0; dim < P4EST_DIM; dim++)
    {
      if(face_velocity_plus_km1 != NULL){
        ierr = VecRestoreArrayRead(face_velocity_plus_km1[dim], &face_velocity_plus_km1_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_minus_km1 != NULL){
        ierr = VecRestoreArrayRead(face_velocity_minus_km1[dim], &face_velocity_minus_km1_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_plus != NULL && set_for_projection_steps){
        ierr = VecRestoreArrayRead(face_velocity_plus[dim], &face_velocity_plus_p[dim]); CHKERRXX(ierr);
      }
      if(face_velocity_minus != NULL && set_for_projection_steps){
        ierr = VecRestoreArrayRead(face_velocity_minus[dim], &face_velocity_minus_p[dim]); CHKERRXX(ierr);
      }
    }
  }

  if(map_quad_to_cf != NULL)
    map_quad_to_cf->insert(std::pair<p4est_locidx_t, size_t>(quad_idx, correction_function_for_quad.size()));
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
  const my_p4est_finite_volume_t* fv_quad = (is_quad_crossed ? &finite_volume_data_for_quad.at(quad_idx) : NULL);

  if(is_quad_crossed)
  {
#ifdef CASL_THROWS
    if(correction_function_for_quad.find(quad_idx) == correction_function_for_quad.end() || fv_quad == NULL)
      throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::get_numbers_of_cells_involved_in_equation_for_quad() \n couldn't find the correction function or the finite-volume info for local quad "
                               + std::to_string(quad_idx) + " on proc " + std::to_string(p4est->mpirank) + ", located at (" + std::to_string(xyz_quad[0]) + ", " + std::to_string(xyz_quad[1]) ONLY3D(+ ", " + std::to_string(xyz_quad[1])) + ")");
#endif
    const correction_function_t& correction_function = correction_function_for_quad.at(quad_idx);

    double max_relevant_face_for_ghost = 0.0;
    for (u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
      max_relevant_face_for_ghost = MAX(max_relevant_face_for_ghost, (sgn_quad < 0 ? fv_quad->face_area_in_positive_domain(oriented_dir) : fv_quad->face_area_in_negative_domain(oriented_dir)));

    if(max_relevant_face_for_ghost >= interface_relative_threshold*reference_face_area)
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
    if(is_face_crossed[oriented_dir]
       || (!are_all_cell_centers_on_same_side && !signs_of_phi_are_different(sgn_face, sgn_quad))
       || (are_all_cell_centers_on_same_side  &&  signs_of_phi_are_different(sgn_face, sgn_quad))) // the neighbor's correction function kicks in (--> see build_discretization_for_quad)
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
      const char sgn_ghost_value_in_neighbor_cell = (are_all_cell_centers_on_same_side ? -sgn_quad : +sgn_quad);
      if(!is_face_crossed[oriented_dir] || (sgn_ghost_value_in_neighbor_cell < 0 ? fv_quad->face_area_in_negative_domain(oriented_dir) : fv_quad->face_area_in_positive_domain(oriented_dir)) >= interface_relative_threshold*reference_face_area)
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
  const double *vstar_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *vstar_plus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double       *rhs_p             = NULL;

  P4EST_ASSERT((user_rhs_minus == NULL && user_rhs_plus == NULL) || (user_rhs_minus != NULL && user_rhs_plus != NULL));                     // can't have one provided but not the other...
  P4EST_ASSERT((face_velocity_minus == NULL && face_velocity_plus == NULL) || (face_velocity_minus != NULL && face_velocity_plus != NULL)); // can't have one provided but not the other...
  const bool with_cell_sampled_rhs  = user_rhs_minus != NULL && user_rhs_plus != NULL;
  const bool is_projecting_vstar = face_velocity_minus != NULL && face_velocity_plus != NULL;
#ifdef CASL_THROWS
  if(with_cell_sampled_rhs && is_projecting_vstar)
    std::cerr << "my_p4est_poisson_jump_cells_fv_t::build_discretization_for_quad(): [WARNING] the solver is configured with both cell-sampled rhs's and face-sampled velocity fields..." << std::endl;
#endif
  linear_combination_of_dof_t vstar_on_face;
  P4EST_ASSERT(!is_projecting_vstar || (ANDD(face_velocity_minus[0] != NULL, face_velocity_minus[1] != NULL, face_velocity_minus[2] != NULL) && ANDD(face_velocity_plus[0] != NULL, face_velocity_plus[1] != NULL, face_velocity_plus[2] != NULL)));

  if(!rhs_is_set)
  {
    if(with_cell_sampled_rhs)
    {
      ierr = VecGetArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);
    }
    if(is_projecting_vstar)
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecGetArrayRead(face_velocity_minus[dim], &vstar_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArrayRead(face_velocity_plus[dim], &vstar_plus_p[dim]); CHKERRXX(ierr);
      }
    }
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    // initialize the discretized rhs
    rhs_p[quad_idx] = 0.0;
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

  const my_p4est_finite_volume_t* finite_volume_of_quad       = NULL;
  const correction_function_t*    correction_function_of_quad = NULL;
  bool is_face_crossed[P4EST_FACES];
  const bool is_quad_crossed = interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, is_face_crossed);
  if(is_quad_crossed)
  {
    if(ANDD(!is_face_crossed[0] && !is_face_crossed[1], !is_face_crossed[2] && !is_face_crossed[3], !is_face_crossed[4] && !is_face_crossed[5]))
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
  P4EST_ASSERT(!is_quad_crossed || (correction_function_of_quad != NULL && finite_volume_of_quad != NULL)); // if the quadrant is crossed, we *MUST* have access to those

  /* First add the diagonal terms */
  const bool nonzero_diag_term = (is_quad_crossed ? MAX(fabs(add_diag_minus), fabs(add_diag_plus)) : (sgn_quad > 0 ? fabs(add_diag_plus) : fabs(add_diag_minus))) > EPS;
  if(!matrix_is_set && nonzero_diag_term)
  {
    if(is_quad_crossed){
      ierr = MatSetValue(A, quad_gloidx, quad_gloidx,
                         finite_volume_of_quad->volume_in_negative_domain()*add_diag_minus + finite_volume_of_quad->volume_in_positive_domain()*add_diag_plus, ADD_VALUES); CHKERRXX(ierr);
      P4EST_ASSERT(correction_function_of_quad != NULL);
      for (size_t k = 0; k < correction_function_of_quad->solution_dependent_terms.size(); ++k) {
        ierr = MatSetValue(A, quad_gloidx, compute_global_index(correction_function_of_quad->solution_dependent_terms[k].dof_idx),
                           (sgn_quad > 0 ? -finite_volume_of_quad->volume_in_negative_domain()*add_diag_minus : +finite_volume_of_quad->volume_in_positive_domain()*add_diag_plus)*correction_function_of_quad->solution_dependent_terms[k].weight, ADD_VALUES); CHKERRXX(ierr); }
    }
    else {
      ierr = MatSetValue(A, quad_gloidx, quad_gloidx,
                         cell_volume*(sgn_quad > 0 ? add_diag_plus : add_diag_minus), ADD_VALUES); CHKERRXX(ierr); }
    if(nullspace_contains_constant_vector != NULL)
      *nullspace_contains_constant_vector = 0;
  }
  if(!rhs_is_set && nonzero_diag_term && is_quad_crossed)
    rhs_p[quad_idx] -= (sgn_quad > 0 ? -finite_volume_of_quad->volume_in_negative_domain()*add_diag_minus : +finite_volume_of_quad->volume_in_positive_domain()*add_diag_plus)*correction_function_of_quad->jump_dependent_terms;

  // bulk terms (volumetric terms and integral of fluxes across the interface) coming into the discretized rhs
  if(!rhs_is_set)
  {
    if(with_cell_sampled_rhs)
    {
      if(is_quad_crossed)
        rhs_p[quad_idx] += finite_volume_of_quad->volume_in_negative_domain()*user_rhs_minus_p[quad_idx] + finite_volume_of_quad->volume_in_positive_domain()*user_rhs_plus_p[quad_idx];
      else
        rhs_p[quad_idx] += (sgn_quad < 0 ? user_rhs_minus_p[quad_idx] : user_rhs_plus_p[quad_idx])*cell_volume;
    }
    if(is_quad_crossed)
    {
      P4EST_ASSERT(finite_volume_of_quad->interfaces.size() <= 1); // could be 0 if only one point is 0.0 and the rest are > 0.0, but we are not dealing yet with > 1 interfaces...
      for (size_t k = 0; k < finite_volume_of_quad->interfaces.size(); ++k)
      {
        const double xyz_interface_quadrature[P4EST_DIM] = {DIM(xyz_quad[0] + finite_volume_of_quad->interfaces[k].centroid[0], xyz_quad[1] + finite_volume_of_quad->interfaces[k].centroid[1], xyz_quad[2] + finite_volume_of_quad->interfaces[k].centroid[2])};
        if(interp_jump_normal_flux != NULL)
          rhs_p[quad_idx] -= finite_volume_of_quad->interfaces[k].area*(*interp_jump_normal_flux)(xyz_interface_quadrature);
        if(interp_jump_normal_velocity != NULL) // --> 0.0 if not given
          rhs_p[quad_idx] += finite_volume_of_quad->interfaces[k].area*((*interp_jump_normal_velocity)(xyz_interface_quadrature));
      }
    }
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
      if(!rhs_is_set && is_projecting_vstar)
      {
        const p4est_locidx_t f_idx = interface_manager->get_faces()->q2f(quad_idx, oriented_dir);
        const double* &vstar_to_consider = (sgn_quad < 0 ? vstar_minus_p[oriented_dir/2] : vstar_plus_p[oriented_dir/2]);
        rhs_p[quad_idx] += (oriented_dir%2 == 1 ? -1.0 : +1.0)*full_face_area*vstar_to_consider[f_idx];
      }

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
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, are_all_cell_centers_on_same_side, (is_projecting_vstar ? &vstar_on_face : NULL));
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
      const p4est_locidx_t face_idx = interface_manager->get_faces()->q2f(quad_idx, oriented_dir);
      for (char sgn_eqn = (is_face_crossed[oriented_dir] ? -1 : sgn_face); sgn_eqn < (is_face_crossed[oriented_dir] ? 1 : sgn_face) + 1; sgn_eqn += 2) { // sum up all nontrivial equation terms
        const double face_area = (is_face_crossed[oriented_dir] ? (sgn_eqn < 0 ? finite_volume_of_quad->face_area_in_negative_domain(oriented_dir) : finite_volume_of_quad->face_area_in_positive_domain(oriented_dir)) : full_face_area);

        if(face_area < interface_relative_threshold*reference_face_area)
          continue;

        if(!rhs_is_set && is_projecting_vstar)
        {
          const double* &vstar_to_consider = (sgn_eqn < 0 ? vstar_minus_p[oriented_dir/2] : vstar_plus_p[oriented_dir/2]);
          rhs_p[quad_idx] += (oriented_dir%2 == 1 ? -1.0 : +1.0)*vstar_to_consider[face_idx]*face_area;
        }

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
      if(!rhs_is_set && is_projecting_vstar)
      {
        const double* &vstar_dir_to_consider = (sgn_quad < 0 ? vstar_minus_p[oriented_dir/2] : vstar_plus_p[oriented_dir/2]);
        rhs_p[quad_idx] += (oriented_dir%2 == 1 ? -1.0 : +1.0)*vstar_on_face(vstar_dir_to_consider)*full_face_area;
      }
    }
  }

  if(!rhs_is_set)
  {
    if(with_cell_sampled_rhs)
    {
      ierr = VecRestoreArrayRead(user_rhs_minus, &user_rhs_minus_p);  CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(user_rhs_plus, &user_rhs_plus_p);    CHKERRXX(ierr);
    }
    if(is_projecting_vstar)
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecRestoreArrayRead(face_velocity_minus[dim], &vstar_minus_p[dim]);  CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(face_velocity_plus[dim], &vstar_plus_p[dim]);    CHKERRXX(ierr);
      }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  }

  return;
}

void my_p4est_poisson_jump_cells_fv_t::local_projection_for_face(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces,
                                                                 double* flux_component_minus_p[P4EST_DIM], double* flux_component_plus_p[P4EST_DIM],
                                                                 double* face_velocity_minus_p[P4EST_DIM], double* face_velocity_plus_p[P4EST_DIM]) const
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
  const char sgn_face   = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : 1);
  const char sgn_q      = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);
  const bool is_face_crossed = interface_manager->is_face_crossed_by_interface(f_idx, dim);

  PetscErrorCode ierr;
  const double *extrapolation_minus_p = NULL, *extrapolation_plus_p = NULL;
  const double *solution_p = NULL;
  bool using_extrapolations = false;
  if(extrapolation_minus != NULL && extrapolation_plus != NULL)
  {
    ierr = VecGetArrayRead(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
    using_extrapolations = true;
  }
  else
  {
    ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  }
  P4EST_ASSERT((using_extrapolations && extrapolation_minus_p != NULL && extrapolation_plus_p != NULL) || (!using_extrapolations && solution_p != NULL));

  double flux_component_minus, flux_component_plus;
  flux_component_minus = flux_component_plus = NAN;
  double *flux_at_face = (sgn_face < 0 ? &flux_component_minus : &flux_component_plus);

  if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
  {
    P4EST_ASSERT(f_idx != NO_VELOCITY);
#ifdef CASL_THROWS
    if(signs_of_phi_are_different(sgn_q, sgn_face) || is_face_crossed)
      throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::local_projection_for_face(): a wall-cell is crossed by the interface, this is not handled yet...");
#endif
    switch(bc->wallType(xyz_face))
    {
    case DIRICHLET:
      if(using_extrapolations)
      {
        flux_component_minus  = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(2.0*mu_minus *(bc->wallValue(xyz_face) - extrapolation_minus_p[quad_idx])/cell_dxyz[dim]);
        flux_component_plus   = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(2.0*mu_plus  *(bc->wallValue(xyz_face) - extrapolation_plus_p[quad_idx])/cell_dxyz[dim]);
      }
      else
        *flux_at_face = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(2.0*(sgn_face > 0 ? mu_plus : mu_minus)*(bc->wallValue(xyz_face) - solution_p[quad_idx])/cell_dxyz[dim]);
      break;
    case NEUMANN:
    {
      flux_component_minus  = (oriented_dir%2 == 1 ? +1.0 : -1.0)*mu_minus*bc->wallValue(xyz_face);
      flux_component_plus   = (oriented_dir%2 == 1 ? +1.0 : -1.0)*mu_plus*bc->wallValue(xyz_face);
      break;
    }
    default:
      throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::local_projection_for_face(): unknown boundary condition on a wall.");
    }
  }
  else
  {
    set_of_neighboring_quadrants direct_neighbors;
    bool one_sided;
    linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, one_sided);

    if(using_extrapolations)
    {
      flux_component_minus  = mu_minus*stable_projection_derivative(extrapolation_minus_p);
      flux_component_plus   = mu_plus *stable_projection_derivative(extrapolation_plus_p);
    }
    else
    {
      if(one_sided && !signs_of_phi_are_different(sgn_q, sgn_face) && !is_face_crossed)
        *flux_at_face = (sgn_face > 0 ? mu_plus : mu_minus)*stable_projection_derivative(solution_p);
      else
      {
        P4EST_ASSERT(direct_neighbors.size() == 1);
        const p4est_quadrant_t& direct_neighbor = *direct_neighbors.begin();
        P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid() && quad->level == direct_neighbor.level);
        const char sgn_direct_neighbor  = (one_sided ? 1 : -1)*sgn_q;

        for (char sgn_flux = (is_face_crossed ? -1 : sgn_face); sgn_flux <= (is_face_crossed ? +1 : sgn_face); sgn_flux += 2) {
          const double &mu_flux = (sgn_flux < 0 ? mu_minus  : mu_plus);
          flux_at_face    = (sgn_flux < 0 ? &flux_component_minus : &flux_component_plus);
          *flux_at_face   = 0.0; // initialization

          for (u_char dof = 0; dof < 2; ++dof) // dof == 0 : this quadrant; dof == 1 : the direct neighbor
          {
            const double coeff_dof        = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(dof == 0 ? -1.0 : +1.0)*mu_flux/dxyz_min[oriented_dir/2];
            const p4est_locidx_t& idx_dof = (dof == 0 ? quad_idx  : direct_neighbor.p.piggy3.local_num);
            const char& sgn_dof           = (dof == 0 ? sgn_q     : sgn_direct_neighbor);

            *flux_at_face += coeff_dof*solution_p[idx_dof];
            if(signs_of_phi_are_different(sgn_flux, sgn_dof))
            {
  #ifdef CASL_THROWS
              if(correction_function_for_quad.find(idx_dof) == correction_function_for_quad.end())
                throw std::runtime_error("my_p4est_poisson_jump_cells_fv_t::local_projection_for_face couldn't find the correction function for quad " + std::to_string(idx_dof)
                                         + ", required for the calculation of a one-sided flux at face " + std::to_string(f_idx) + " of Cartesian normal " + std::to_string(dim) + ", found between quads "
                                         + std::to_string(quad_idx) + " and " + std::to_string(direct_neighbor.p.piggy3.local_num) + " on proc " + std::to_string(p4est->mpirank)
                                         + " (local partition has " + std::to_string(p4est->local_num_quadrants) + " quadrants).");
  #endif
              const correction_function_t& correction_function_dof = correction_function_for_quad.at(idx_dof);
              *flux_at_face += coeff_dof*(sgn_flux > 0 ? +1.0 : -1.0)*correction_function_dof(solution_p);
            }
          }
        }
      }
    }
  }
  if(extrapolation_minus_p != NULL && extrapolation_plus_p != NULL)
  {
    ierr = VecRestoreArrayRead(extrapolation_minus, &extrapolation_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(extrapolation_plus, &extrapolation_plus_p); CHKERRXX(ierr);
  }
  else
  {
    ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  }

  // If the user needs the flux components back, return them hereunder
  // If the user needs the sharp flux components back (i.e., if flux_component_minus and flux_component_plus are pointing
  // to the same vectors), return the component corresponding to the face of interest only.
  if(flux_component_minus_p[dim] != NULL && (sgn_face < 0 || flux_component_minus_p[dim] != flux_component_plus_p[dim]))
  {
    flux_component_minus_p[dim][f_idx] = flux_component_minus;
    P4EST_ASSERT(sgn_face > 0 || !ISNAN(flux_component_minus_p[dim][f_idx]));
  }
  if(flux_component_plus_p[dim] != NULL && (sgn_face > 0 || flux_component_plus_p[dim] != flux_component_minus_p[dim]))
  {
    flux_component_plus_p[dim][f_idx] = flux_component_plus;
    P4EST_ASSERT(sgn_face < 0 || !ISNAN(flux_component_plus_p[dim][f_idx]));
  }

  if(face_velocity_plus_p[dim] != NULL && face_velocity_minus_p[dim] != NULL)
  {
    P4EST_ASSERT((sgn_face > 0 && !is_face_crossed) || !ISNAN(flux_component_minus));
    P4EST_ASSERT((sgn_face < 0 && !is_face_crossed) || !ISNAN(flux_component_plus ));
    if(ISNAN(flux_component_minus))
      face_velocity_minus_p[dim][f_idx] = DBL_MAX;
    else
      face_velocity_minus_p[dim][f_idx] -= flux_component_minus;

    if(ISNAN(flux_component_plus))
      face_velocity_plus_p[dim][f_idx] = DBL_MAX;
    else
      face_velocity_plus_p[dim][f_idx] -= flux_component_plus;
  }

  return;
}

void my_p4est_poisson_jump_cells_fv_t::solve_for_sharp_solution(const KSPType &ksp_type, const PCType& pc_type)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_fv_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  // make sure the problem is fully defined
  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2])); // boundary conditions
  P4EST_ASSERT(diffusion_coefficients_have_been_set() && interface_is_set());       // essential parameters

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

void my_p4est_poisson_jump_cells_fv_t::initialize_extrapolation_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                                                      double* extrapolation_minus_p, double* extrapolation_plus_p,
                                                                      double* normal_derivative_of_solution_minus_p, double* normal_derivative_of_solution_plus_p, const u_char& degree)
{
  const p4est_quadrant_t* quad;
  const double *xyz_tree_min, *xyz_tree_max;
  double xyz_quad[P4EST_DIM], dxyz_quad[P4EST_DIM];
  fetch_quad_and_tree_coordinates(quad, xyz_tree_min, xyz_tree_max, quad_idx, tree_idx, p4est, ghost);
  xyz_of_quad_center(quad, xyz_tree_min, xyz_tree_max, xyz_quad, dxyz_quad);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);

  map_of_correction_functions_t::const_iterator it = correction_function_for_quad.find(quad_idx);
  const correction_function_t* corr_fun = (it != correction_function_for_quad.end() ? &it->second : NULL);

  if(sgn_quad < 0)
  {
    extrapolation_minus_p[quad_idx] = sharp_solution_p[quad_idx];
    if(corr_fun != NULL && !corr_fun->not_reliable_for_extrapolation)
      extrapolation_plus_p[quad_idx]  = sharp_solution_p[quad_idx] + (*corr_fun)(sharp_solution_p);
    else
      extrapolation_plus_p[quad_idx]  = sharp_solution_p[quad_idx] + (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0); // rough initialization to (hopefully) speed up the convergence in pseudo-time
  }
  else
  {
    extrapolation_plus_p[quad_idx] = sharp_solution_p[quad_idx];
    if(corr_fun != NULL && !corr_fun->not_reliable_for_extrapolation)
      extrapolation_minus_p[quad_idx] = sharp_solution_p[quad_idx] - (*corr_fun)(sharp_solution_p);
    else
      extrapolation_minus_p[quad_idx] = sharp_solution_p[quad_idx] - (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0); // rough initialization to (hopefully) speed up the convergence in pseudo-time
  }

  double oriented_normal[P4EST_DIM]; // to calculate the normal derivative of the solution (in the local subdomain) --> the opposite of that vector is required when extrapolating the solution from across the interface
  interface_manager->normal_vector_at_point(xyz_quad, oriented_normal, (sgn_quad < 0 ? +1.0 : -1.0));

  double diagonal_coeff_for_n_dot_grad_this_side = 0.0, diagonal_coeff_for_n_dot_grad_across = 0.0;
  extrapolation_operator_t* extrapolation_operator_across     = NULL;
  extrapolation_operator_t* extrapolation_operator_this_side  = NULL; // ("_this_side" may not be required)
  if(extrapolation_operator_minus.find(quad_idx) == extrapolation_operator_minus.end() && extrapolation_operator_plus.find(quad_idx) == extrapolation_operator_plus.end()) // no operator constructed yet
  {
    extrapolation_operator_across     = new extrapolation_operator_t;
    extrapolation_operator_this_side  = new extrapolation_operator_t; // ("_this_side" may not be required)
  }

  double n_dot_grad_u = 0.0; // let's evaluate this term on the side of quad, set the corresponding value across to 0.0

  bool un_is_well_defined = true;

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    double sharp_derivative_m, sharp_derivative_p;
    double dist_m = dxyz_quad[dim], dist_p = dxyz_quad[dim]; // relevant only in case of dirichlet wall BC
    for (u_char orientation = 0; orientation < 2; ++orientation) {
      double &oriented_sharp_derivative = (orientation == 1 ? sharp_derivative_p  : sharp_derivative_m);
      double &oriented_dist             = (orientation == 1 ? dist_p              : dist_m);
      if(is_quad_Wall(p4est, tree_idx, quad, 2*dim + orientation))
      {
        double xyz_wall[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_wall[dim] += (orientation == 1 ? +0.5 : -0.5)*dxyz_quad[dim];
#ifdef CASL_THROWS
        const char sgn_wall = (interface_manager->phi_at_point(xyz_wall) <= 0.0 ? -1 : +1);
        if(signs_of_phi_are_different(sgn_quad, sgn_wall))
          throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::initialize_extrapolation_local(): a wall-cell is crossed by the interface, this is not handled yet...");
#endif
        switch(bc->wallType(xyz_wall))
        {
        case DIRICHLET:
          oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*(2.0*(bc->wallValue(xyz_wall) - sharp_solution_p[quad_idx])/dxyz_quad[dim]);
          oriented_dist             = 0.5*dxyz_quad[dim];
          break;
        case NEUMANN:
          oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*bc->wallValue(xyz_wall);
          break;
        default:
          throw std::invalid_argument("my_p4est_poisson_jump_cells_fv_t::initialize_extrapolation_local(): unknown boundary condition on a wall.");
        }
        // no term in the operator(s) <--> "Neumann condition" for extrapolation purposes (for now, at least, if you have a better idea, go ahead, be my guest!)
      }
      else
      {
        set_of_neighboring_quadrants direct_neighbors;
        bool one_sided;
        linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, 2*dim + orientation, direct_neighbors, one_sided);

        if(one_sided)
          oriented_sharp_derivative   = stable_projection_derivative(sharp_solution_p);
        else
        {
          P4EST_ASSERT(direct_neighbors.size() == 1);
          const p4est_quadrant_t& direct_neighbor = *direct_neighbors.begin();
          P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid() && quad->level == direct_neighbor.level);

          map_of_correction_functions_t::const_iterator it_neighbor = correction_function_for_quad.find(direct_neighbor.p.piggy3.local_num);
          if(it_neighbor != correction_function_for_quad.end() && !it_neighbor->second.not_reliable_for_extrapolation)
            oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*(sharp_solution_p[direct_neighbor.p.piggy3.local_num] + (sgn_quad > 0 ? +1.0 : -1.0)*it_neighbor->second(sharp_solution_p) - sharp_solution_p[quad_idx])/dxyz_min[dim];
          else
            un_is_well_defined = false;
        }

        if(extrapolation_operator_across != NULL && extrapolation_operator_this_side != NULL)
        {
          // add the (regular, i.e. without interface-fetching) derivative term(s) to the relevant extrapolation operator (for extrapolating normal derivatives, for instance)
          double discretization_distance = 0.0;
          const bool derivative_is_relevant_for_extrapolation_of_this_side = (oriented_normal[dim] <= 0.0 && orientation == 1) || (oriented_normal[dim] > 0.0 && orientation == 0);
          const double relevant_normal_component = (derivative_is_relevant_for_extrapolation_of_this_side ? +1.0 : -1.0)*oriented_normal[dim];
          extrapolation_operator_t* relevant_operator = (derivative_is_relevant_for_extrapolation_of_this_side ? extrapolation_operator_this_side : extrapolation_operator_across);
          double& relevant_diagonal_term = (derivative_is_relevant_for_extrapolation_of_this_side ? diagonal_coeff_for_n_dot_grad_this_side : diagonal_coeff_for_n_dot_grad_across);

          for (size_t k = 0; k < stable_projection_derivative.size(); ++k) {
            const dof_weighted_term& derivative_term = stable_projection_derivative[k];
            if(derivative_term.dof_idx == quad_idx)
              relevant_diagonal_term += derivative_term.weight*relevant_normal_component;
            else
              relevant_operator->n_dot_grad.add_term(derivative_term.dof_idx, relevant_normal_component*derivative_term.weight);
            discretization_distance = MAX(discretization_distance, fabs(1.0/derivative_term.weight));
          }

          relevant_operator->dtau = MIN(relevant_operator->dtau, discretization_distance/(double) P4EST_DIM);
        }
      }
    }

    // the following is equivalent to FD evaluation using the interface-fetched point(s)
    double sharp_derivative_quad_center = (dist_p*sharp_derivative_m + dist_m*sharp_derivative_p)/(dist_p + dist_m);
    if(dist_m + dist_p < 0.1*pow(2.0, -interface_manager->get_max_level_computational_grid())*dxyz_min[dim]) // "0.1" <--> minimum for coarse grid.
      sharp_derivative_quad_center = 0.5*(sharp_derivative_m + sharp_derivative_p); // it was an underresolved case, the above operation is too risky...

    n_dot_grad_u += oriented_normal[dim]*sharp_derivative_quad_center;
  }

  if(extrapolation_operator_across != NULL && extrapolation_operator_this_side != NULL)
  {
    // complete the extrapolation operators
    extrapolation_operator_across->n_dot_grad.add_term(quad_idx, diagonal_coeff_for_n_dot_grad_across);
    extrapolation_operator_this_side->n_dot_grad.add_term(quad_idx, diagonal_coeff_for_n_dot_grad_this_side);
  }

  if(degree > 0)
  {
    if(sgn_quad < 0)
    {
      normal_derivative_of_solution_minus_p[quad_idx] = (un_is_well_defined ? n_dot_grad_u : 0.0); // if not well-defined, will be estimated via extrapolation
      normal_derivative_of_solution_plus_p[quad_idx]  = 0.0; // to be calculated later on in actual extrapolation
    }
    else
    {
      normal_derivative_of_solution_minus_p[quad_idx] = 0.0; // to be calculated later on in actual extrapolation
      normal_derivative_of_solution_plus_p[quad_idx]  = (un_is_well_defined ? n_dot_grad_u : 0.0); // if not well-defined, will be estimated via extrapolation
    }
  }


  if(extrapolation_operator_across != NULL && extrapolation_operator_this_side != NULL)
  {
    if(sgn_quad < 0)
      extrapolation_operator_plus[quad_idx] = *extrapolation_operator_across;
    else
      extrapolation_operator_minus[quad_idx] = *extrapolation_operator_across;
    if(!un_is_well_defined)
    {
      if(sgn_quad < 0)
        extrapolation_operator_minus[quad_idx] = *extrapolation_operator_this_side;
      else
        extrapolation_operator_plus[quad_idx] = *extrapolation_operator_this_side;
    }

    // delete what you built locally (map insertion creates a copy)
    delete extrapolation_operator_across;
    delete extrapolation_operator_this_side;
  }

  return;
}


void my_p4est_poisson_jump_cells_fv_t::extrapolate_solution_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double*,
                                                                  double* tmp_minus_p, double* tmp_plus_p,
                                                                  const double* extrapolation_minus_p, const double* extrapolation_plus_p,
                                                                  const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p)
{
  tmp_minus_p[quad_idx] = extrapolation_minus_p[quad_idx];
  tmp_plus_p[quad_idx] = extrapolation_plus_p[quad_idx];

  if(correction_function_for_quad.find(quad_idx) != correction_function_for_quad.end() && !correction_function_for_quad[quad_idx].not_reliable_for_extrapolation)
    return; // the ghost value is safely defined intrinsically, no need to solve for the pseudo-time step equation here

  // no (safe to use) correction function was defined, extrapolation is required
  double xyz_quad[P4EST_DIM];
  quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);

  double* extrapolation_np1_p       = (sgn_quad < 0 ? tmp_plus_p : tmp_minus_p);
  const double* extrapolation_n_p   = (sgn_quad < 0 ? extrapolation_plus_p : extrapolation_minus_p);
  const double* normal_derivative_p = (sgn_quad < 0 ? normal_derivative_of_solution_plus_p : normal_derivative_of_solution_minus_p);
  const extrapolation_operator_t& extrapolation_operator = (sgn_quad < 0 ? extrapolation_operator_plus.at(quad_idx) : extrapolation_operator_minus.at(quad_idx));
  extrapolation_np1_p[quad_idx] -= extrapolation_operator.dtau*(extrapolation_operator.n_dot_grad(extrapolation_n_p) - (normal_derivative_p != NULL ? normal_derivative_p[quad_idx] : 0.0));

  return;
}



