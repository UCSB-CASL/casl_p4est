#ifdef P4_TO_P8
#include "my_p8est_xgfm_cells.h"
#else
#include "my_p4est_xgfm_cells.h"
#endif

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_xgfm_cells_solve_for_sharp_solution;
extern PetscLogEvent log_my_p4est_xgfm_cells_extend_interface_values;
extern PetscLogEvent log_my_p4est_xgfm_cells_interpolate_cell_extension_to_interface_capturing_nodes;
extern PetscLogEvent log_my_p4est_xgfm_cells_update_rhs_and_residual;
#endif

my_p4est_xgfm_cells_t::my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t* nodes_)
  : my_p4est_poisson_jump_cells_t (ngbd_c, nodes_), activate_xGFM(true) // default behavior is to activate the xGFM corrections
{
  xGFM_absolute_accuracy_threshold  = 1e-8;   // default value
  xGFM_tolerance_on_rel_residual    = 1e-12;  // default value

  residual = solution = extension_on_cells = extension_on_nodes = jump_flux = NULL;
  local_interpolators_are_stored_and_set = false;
  pseudo_time_step_increment_operator.resize(p4est->local_num_quadrants);
  extension_operators_are_stored_and_set = false;
}

my_p4est_xgfm_cells_t::~my_p4est_xgfm_cells_t()
{
  PetscErrorCode ierr;
  if (extension_on_cells  != NULL)  { ierr = VecDestroy(extension_on_cells);  CHKERRXX(ierr); }
  if (extension_on_nodes  != NULL)  { ierr = VecDestroy(extension_on_nodes);  CHKERRXX(ierr); }
  if (residual            != NULL)  { ierr = VecDestroy(residual);            CHKERRXX(ierr); }
  if (solution            != NULL)  { ierr = VecDestroy(solution);            CHKERRXX(ierr); }
  if (jump_flux           != NULL)  { ierr = VecDestroy(jump_flux);           CHKERRXX(ierr); }
}

void my_p4est_xgfm_cells_t::set_interface(my_p4est_interface_manager_t* interface_manager_)
{
  my_p4est_poisson_jump_cells_t::set_interface(interface_manager_);

  local_interpolators.resize(interface_manager->get_interface_capturing_ngbd_n().get_nodes()->num_owned_indeps);
  return;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_at_all_interface_capturing_nodes() const
{
  PetscErrorCode ierr;
  P4EST_ASSERT(jump_flux != NULL);
  const double *jump_u_p, *jump_normal_flux_u_p, *grad_phi_p;
  const double *extension_on_nodes_p = NULL;
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecGetArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }
  double *jump_flux_p;
  ierr = VecGetArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_layer_size(); ++k)
    compute_jumps_in_flux_components_for_interface_capturing_node(interface_capturing_ngbd_n.get_layer_node(k), jump_flux_p,
                                                                  jump_u_p, jump_normal_flux_u_p, grad_phi_p, extension_on_nodes_p);
  ierr = VecGhostUpdateBegin(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_local_size(); ++k)
    compute_jumps_in_flux_components_for_interface_capturing_node(interface_capturing_ngbd_n.get_local_node(k), jump_flux_p,
                                                                  jump_u_p, jump_normal_flux_u_p, grad_phi_p, extension_on_nodes_p);
  ierr = VecGhostUpdateEnd(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecRestoreArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }

  return;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only() const
{
  if(!interface_manager->is_storing_cell_FD_interface_data())
  {
    compute_jumps_in_flux_components_at_all_interface_capturing_nodes();
    return;
  }

  PetscErrorCode ierr;
  P4EST_ASSERT(jump_flux != NULL);
  const double *jump_u_p, *jump_normal_flux_u_p, *grad_phi_p;
  const double *extension_on_nodes_p = NULL;
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecGetArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }
  double *jump_flux_p;
  ierr = VecGetArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  const map_of_interface_neighbors_t& cell_FD_interface_data = interface_manager->get_cell_FD_interface_data();
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
  for (map_of_interface_neighbors_t::const_iterator it = cell_FD_interface_data.begin(); it != cell_FD_interface_data.end(); ++it) {
    const linear_combination_of_dof_t& node_interpolant = it->second.node_interpolant;
    for (size_t k = 0; k < node_interpolant.size(); ++k)
      if(node_interpolant[k].dof_idx < interface_capturing_ngbd_n.get_nodes()->num_owned_indeps)
        compute_jumps_in_flux_components_for_interface_capturing_node(node_interpolant[k].dof_idx, jump_flux_p,
                                                                      jump_u_p, jump_normal_flux_u_p, grad_phi_p, extension_on_nodes_p);
  }

  ierr = VecGhostUpdateBegin(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecRestoreArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }

  return;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_for_interface_capturing_node(const p4est_locidx_t& node_idx, double *jump_flux_p,
                                                                                          const double *jump_u_p, const double *jump_normal_flux_p, const double *grad_phi_p,
                                                                                          const double *extension_on_nodes_p) const
{
  double grad_jump_u[P4EST_DIM];
  double grad_extension[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)}; // set to 0.0 to avoid undefined behavior if extension_on_nodes_p == NULL

  const quad_neighbor_nodes_of_node_t *qnnn;
  if(activate_xGFM)
  {
    interface_manager->get_interface_capturing_ngbd_n().get_neighbors(node_idx, qnnn);
    if(extension_on_nodes_p != NULL)
    {
      const double *inputs[2] = {jump_u_p,    extension_on_nodes_p};
      double *outputs[2]      = {grad_jump_u, grad_extension};
      qnnn->gradient(inputs, outputs, 2);
    }
    else
      qnnn->gradient(jump_u_p, grad_jump_u);
  }

  const double *grad_phi = grad_phi_p + P4EST_DIM*node_idx;
  const double mag_grad_phi = sqrt(SUMD(SQR(grad_phi[0]), SQR(grad_phi[1]), SQR(grad_phi[2])));
  const double normal[P4EST_DIM] = {DIM((mag_grad_phi > EPS ? grad_phi[0]/mag_grad_phi : 0.0),
                                    (mag_grad_phi > EPS ? grad_phi[1]/mag_grad_phi : 0.0),
                                    (mag_grad_phi > EPS ? grad_phi[2]/mag_grad_phi : 0.0))};

  const double grad_jump_u_cdot_normal    = SUMD(normal[0]*grad_jump_u[0], normal[1]*grad_jump_u[1], normal[2]*grad_jump_u[2]);
  const double grad_extension_cdot_normal = SUMD(normal[0]*grad_extension[0], normal[1]*grad_extension[1], normal[2]*grad_extension[2]);

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    jump_flux_p[P4EST_DIM*node_idx + dim] = jump_normal_flux_p[node_idx]*normal[dim];
    if(activate_xGFM)
    {
      jump_flux_p[P4EST_DIM*node_idx + dim] += (extend_negative_interface_values() ? mu_plus : mu_minus)*(grad_jump_u[dim] - grad_jump_u_cdot_normal*normal[dim]);
      if(extension_on_nodes_p != NULL)
        jump_flux_p[P4EST_DIM*node_idx + dim] += get_jump_in_mu()*(grad_extension[dim] - grad_extension_cdot_normal*normal[dim]);
    }
  }

  return;
}

void my_p4est_xgfm_cells_t::get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                               PetscInt& number_of_local_cells_involved, PetscInt& number_of_ghost_cells_involved) const
{
  const p4est_tree_t *tree  = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t *quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  set_of_neighboring_quadrants neighbor_quads_involved;
  for(u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
  {
    set_of_neighboring_quadrants direct_neighbors;
    cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, oriented_dir);

    for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
      neighbor_quads_involved.insert(*it);

    if(direct_neighbors.size() == 1 && direct_neighbors.begin()->level < quad->level)
      cell_ngbd->find_neighbor_cells_of_cell(neighbor_quads_involved, direct_neighbors.begin()->p.piggy3.local_num, direct_neighbors.begin()->p.piggy3.which_tree, oriented_dir%2 == 0 ? oriented_dir + 1 : oriented_dir - 1);
  }

  number_of_local_cells_involved = 1; // will always have the current cell
  number_of_ghost_cells_involved = 0;
  for (set_of_neighboring_quadrants::const_iterator it = neighbor_quads_involved.begin(); it != neighbor_quads_involved.end(); ++it) {
    if(it->p.piggy3.local_num < p4est->local_num_quadrants && it->p.piggy3.local_num != quad_idx)
      number_of_local_cells_involved++;
    else
      number_of_ghost_cells_involved++;
  }

  return;
}

void my_p4est_xgfm_cells_t::build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int* nullspace_contains_constant_vector)
{
  PetscErrorCode ierr;
  const double *user_rhs_minus_p  = NULL;
  const double *user_rhs_plus_p   = NULL;
  const double *jump_u_p          = NULL;
  const double *jump_flux_p       = NULL;
  double       *rhs_p             = NULL;
  if(!rhs_is_set)
  {
    if(user_rhs_minus != NULL){
      ierr = VecGetArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr); }
    if(user_rhs_plus != NULL){
      ierr = VecGetArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr); }
    if(user_vstar_minus != NULL || user_vstar_plus != NULL)
      throw std::runtime_error("my_p4est_xgfm_cells_t::build_discretization_for_quad : not able to handle vstar, yet --> implement that, please");
    P4EST_ASSERT(jump_u != NULL && jump_flux != NULL);
    ierr = VecGetArrayRead(jump_u, &jump_u_p);        CHKERRXX(ierr);
    ierr = VecGetArrayRead(jump_flux, &jump_flux_p);  CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p);                  CHKERRXX(ierr);
  }

  const PetscInt quad_gloidx = compute_global_index(quad_idx);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double cell_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_quad = interface_manager->phi(xyz_quad);
  const double &mu_here = (phi_quad > 0.0 ? mu_plus : mu_minus);

  /* First add the diagonal term */
  const double& add_diag = (phi_quad > 0.0 ? add_diag_plus : add_diag_minus);
  if(!matrix_is_set && fabs(add_diag) > EPS)
  {
    ierr = MatSetValue(A, quad_gloidx, quad_gloidx, cell_volume*add_diag, ADD_VALUES); CHKERRXX(ierr);
    if(nullspace_contains_constant_vector != NULL)
      *nullspace_contains_constant_vector = 0;
  }
  if(!rhs_is_set)
    rhs_p[quad_idx] = (phi_quad <= 0.0 ? user_rhs_minus_p[quad_idx] : user_rhs_plus_p[quad_idx])*cell_volume;

  for(u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
  {
    const double face_area = cell_volume/cell_dxyz[oriented_dir/2];

    /* first check if the cell is a wall
     * We will assume that walls are not crossed by the interface, in a first attempt! */
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
    {
      double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])};
      xyz_face[oriented_dir/2] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[oriented_dir/2];
#ifdef CASL_THROWS
      const double phi_face = interface_manager->phi(xyz_face);
      if(signs_of_phi_are_different(phi_quad, phi_face))
        throw std::invalid_argument("my_p4est_xgfm_cells_t::build_discretization_for_quad() : a wall-cell is crossed by the interface, this is not handled yet...");
#endif
      switch(bc->wallType(xyz_face))
      {
      case DIRICHLET:
      {
        if(!matrix_is_set)
        {
          if(nullspace_contains_constant_vector != NULL)
            *nullspace_contains_constant_vector = 0;
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu_here*face_area/cell_dxyz[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
          rhs_p[quad_idx]  += 2.0*mu_here*face_area*bc->wallValue(xyz_face)/cell_dxyz[oriented_dir/2];
      }
        break;
      case NEUMANN:
      {
        if(!rhs_is_set)
          rhs_p[quad_idx]  += mu_here*face_area*bc->wallValue(xyz_face);
      }
        break;
      default:
        throw std::invalid_argument("my_p4est_xgfm_cells_t::build_discretization_for_quad() : unknown boundary condition on a wall.");
      }
      continue;
    }

    set_of_neighboring_quadrants direct_neighbors;
    bool operator_is_one_sided;
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, operator_is_one_sided);
    if(operator_is_one_sided)
    {
      if(!matrix_is_set)
        for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k) {
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(stable_projection_derivative_operator[k].dof_idx),
                             (oriented_dir%2 == 1 ? -1.0 : +1.0)*mu_here*face_area*stable_projection_derivative_operator[k].weight, ADD_VALUES); CHKERRXX(ierr);
        }
    }
    else
    {
      /* If no one-side, we assume that the interface is tesselated with uniform finest grid level */
      if(direct_neighbors.size() != 1)
        throw std::runtime_error("my_p4est_xgfm_cells_t::build_discretization_for_quad(): did not find one single direct neighbor for a cell center across the interface. \n Is your grid uniform across the interface?");
      if(quad->level != ((splitting_criteria_t*) p4est->user_pointer)->max_lvl || quad->level != direct_neighbors.begin()->level)
        throw std::runtime_error("my_p4est_xgfm_cells_t::build_discretization_for_quad(): the interface crosses two cells that are either not of the same size or bigger than expected.");
      const p4est_quadrant_t& neighbor_quad = *direct_neighbors.begin();
      const double& mu_across = (phi_quad > 0.0 ? mu_minus : mu_plus);
      if(!matrix_is_set)
      {
        const double mu_jump = interface_manager->GFM_mu_jump_between_cells(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir, mu_here, mu_across);
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx,                                             mu_jump * face_area/dxyz_min[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, quad_gloidx, compute_global_index(neighbor_quad.p.piggy3.local_num), -mu_jump * face_area/dxyz_min[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
      }
      if(!rhs_is_set)
        rhs_p[quad_idx] += face_area*(oriented_dir%2 == 1 ? +1.0 : -1.0)*interface_manager->GFM_jump_terms_for_flux_component_between_cells(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir, mu_here, mu_across, (phi_quad > 0.0), jump_u_p, jump_flux_p);
    }
  }

  if(!rhs_is_set)
  {
    if(user_rhs_minus_p != NULL){
      ierr = VecRestoreArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr); }
    if(user_rhs_plus_p != NULL){
      ierr = VecRestoreArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr); }

    ierr = VecRestoreArrayRead(jump_u, &jump_u_p);        CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p);                  CHKERRXX(ierr);
  }

  return;
}

void my_p4est_xgfm_cells_t::solve_for_sharp_solution(const KSPType& ksp_type , const PCType& pc_type)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2]));   // make sure we have wall boundary conditions if we need them
  P4EST_ASSERT(interface_is_set() && jumps_have_been_set());                          // make sure the problem is fully defined

  PetscBool saved_ksp_original_guess_flag;
  ierr = KSPGetInitialGuessNonzero(ksp, &saved_ksp_original_guess_flag); // we'll change that one to true internally, but we want to set it back to whatever it originally was

  /* clear the solver monitoring */
  solver_monitor.clear();

  /* Set the linear system, the linear solver and solve it (regular GFM, i.e., "Boundary Condition-Capturing scheme...") */
  setup_linear_system();
  ierr = setup_linear_solver(ksp_type, pc_type, xGFM_tolerance_on_rel_residual); CHKERRXX(ierr);
  KSPConvergedReason termination_reason = solve_linear_system();
  if(termination_reason <= 0)
    throw std::runtime_error("my_p4est_xgfm_cells_t::solve() the Krylov solver failed to converge for the very first linear system to solve, the KSPConvergedReason code is " + std::to_string(termination_reason)); // collective runtime_error throw
  if(!activate_xGFM || mus_are_equal())
    solver_monitor.log_iteration(0.0, this); // we just want to log the number of ksp iterations in this case, 0.0 because no correction yet
  else
  {
    Vec former_extension_on_cells = extension_on_cells; // both should be NULL at this stage, this is the generalized usage for fix-point update
    Vec former_extension_on_nodes = extension_on_nodes; // both should be NULL at this stage, this is the generalized usage for fix-point update
    Vec former_residual           = residual;           // both should be NULL at this stage, this is the generalized usage for fix-point update
    Vec former_rhs                = rhs;
    Vec former_solution           = solution;
    if(extension_on_nodes != NULL) // should not be, ever, but if not NULL for whatever reason, we need this here for consistency
      compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only(); // those jumps are function of extension_on_nodes so they need to be updated first if known for whatever reason
    // fix-point update
    extend_interface_values(former_extension_on_cells, former_extension_on_nodes);
    update_rhs_and_residual(former_rhs, former_residual);
    solver_monitor.log_iteration(0.0, this);

    while (!solver_monitor.reached_convergence_within_desired_bounds(xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual)
           && !solve_for_fixpoint_solution(former_solution))
    {
      // we need to keep going : find the next fix-point rhs and fix-point residual based,
      // and linearly combine the two last states in order to minimize the next residual
      extend_interface_values(former_extension_on_cells, former_extension_on_nodes);
      update_rhs_and_residual(former_rhs, former_residual);

      // linear combination of last two solver's states that minimizes the next residual
      const double max_correction = set_solver_state_minimizing_L2_norm_of_residual(former_solution, former_extension_on_cells, former_extension_on_nodes,
                                                                                    former_rhs, former_residual);
      solver_monitor.log_iteration(max_correction, this);
    }

    P4EST_ASSERT(former_solution != solution);
    ierr = VecDestroy(former_solution); CHKERRXX(ierr);
    if(former_extension_on_cells != extension_on_cells){
      ierr = VecDestroy(former_extension_on_cells); CHKERRXX(ierr); }
    if(former_extension_on_nodes != extension_on_nodes){
      ierr = VecDestroy(former_extension_on_nodes); CHKERRXX(ierr); }
    if(former_rhs != rhs){
      ierr = VecDestroy(former_rhs); CHKERRXX(ierr); }
    if(former_residual != residual){
      ierr = VecDestroy(former_residual); CHKERRXX(ierr); }
  }
  ierr = KSPSetInitialGuessNonzero(ksp, saved_ksp_original_guess_flag); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  return;
}

// this function assumes that the (relevant) jumps_in_flux_components are up-to-date and will reset the extension of the appropriate interface-defined value
void my_p4est_xgfm_cells_t::extend_interface_values(Vec &former_extension_on_cells, Vec &former_extension_on_nodes, const double& threshold, const uint& niter_max)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_extend_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(interface_is_set() && jumps_have_been_set() && threshold > EPS && niter_max > 0);

  // save the current cell-sampled extension
  std::swap(former_extension_on_cells, extension_on_cells);
  // create a new vector if needed and build an educated guess for the TVD extension to come:
  if(extension_on_cells == NULL || extension_on_cells == former_extension_on_cells){
    ierr = VecCreateGhostCells(p4est, ghost, &extension_on_cells); CHKERRXX(ierr); } // we need a new vector for the fix-point iteration step with the currently known extension_on_nodes

  if(former_extension_on_cells == NULL)
    initialize_extension_on_cells(); // we start from scratch, build an educated guess using jump_u where needed
  else {
    ierr = VecCopyGhost(former_extension_on_cells, extension_on_cells); CHKERRXX(ierr); }  // we knew something already, use it as initial guess...

  // now do the real TVD extension task on the cell-sampled solution field with the currently known extension_on_nodes
  cell_TVD_extension_of_interface_values(threshold, niter_max);

  // redefine the extension_on_nodes by interpolating the newly defined cell-sampled interface values
  // save the current node-sampled extension
  std::swap(former_extension_on_nodes, extension_on_nodes);
  // create a new vector if needed
  if(extension_on_nodes == NULL || extension_on_nodes == former_extension_on_nodes){
    const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
    ierr = VecCreateGhostNodes(interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), &extension_on_nodes); CHKERRXX(ierr); // we need a new vector to store the interpolated result
  }
  interpolate_cell_extension_to_interface_capturing_nodes();

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_extend_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_xgfm_cells_t::update_rhs_in_relevant_cells_only()
{
  PetscErrorCode ierr;
  // update the rhs values associated with cells involving jump conditions (fix-point update)
  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  const double *jump_u_p, *jump_flux_p, *user_rhs_minus_p, *user_rhs_plus_p;
  ierr = VecGetArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  rhs_is_set = false; // lower this flag in order to update the rhs terms appropriately
  if(interface_manager->is_storing_cell_FD_interface_data())
  {
    std::set<p4est_locidx_t> already_done; already_done.clear();
    const map_of_interface_neighbors_t& cell_FD_interface_data = interface_manager->get_cell_FD_interface_data();
    for (map_of_interface_neighbors_t::const_iterator it = cell_FD_interface_data.begin(); it != cell_FD_interface_data.end(); ++it)
    {
      if(it->first.local_dof_idx < p4est->local_num_quadrants && already_done.find(it->first.local_dof_idx) == already_done.end())
      {
        // find tree_idx
        const p4est_topidx_t tree_idx  = tree_index_of_quad(it->first.local_dof_idx, p4est, ghost);
        build_discretization_for_quad(it->first.local_dof_idx, tree_idx);
        already_done.insert(it->first.local_dof_idx);
      }
      if(it->first.neighbor_dof_idx < p4est->local_num_quadrants && already_done.find(it->first.neighbor_dof_idx) == already_done.end())
      {
        // find tree_idx
        const p4est_topidx_t tree_idx  = tree_index_of_quad(it->first.neighbor_dof_idx, p4est, ghost);
        build_discretization_for_quad(it->first.neighbor_dof_idx, tree_idx);
        already_done.insert(it->first.neighbor_dof_idx);
      }
    }
  }
  else
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
      const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
        build_discretization_for_quad(q + tree->quadrants_offset, tree_idx);
    }
  rhs_is_set = true;

  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  return;
}

// this function recalculates the (relevant) jumps in flux components given the currently known node-sampled extension of interface-defined values
// updates the discretized rhs accordingly and calculates the updated residual of the system based on the current solution and those newly updated jump
// conditions. If a fix-point was reached, this residual should be (close to) 0.0 basically...
void my_p4est_xgfm_cells_t::update_rhs_and_residual(Vec& former_rhs, Vec& former_residual)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(rhs != NULL);

  // update the relevant jumps in flux components according to the newly defined node-sampled extension
  compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only();
  // update the rhs consistently with those new jumps in flux components
  // save the rhs first
  std::swap(former_rhs, rhs);
  // create a new vector if needed
  if(rhs == former_rhs){
    ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr);
    ierr = VecCopy(former_rhs, rhs); CHKERRXX(ierr);
  }

  update_rhs_in_relevant_cells_only();

  // save the current residual
  std::swap(former_residual, residual);
  // create a new vector if needed:
  if(residual == NULL || residual == former_residual){
    ierr = VecCreateNoGhostCells(p4est, &residual); CHKERRXX(ierr); }

  // calculate the fix-point residual
  ierr = VecAXPBY(residual, -1.0, 0.0, rhs); CHKERRXX(ierr);
  ierr = MatMultAdd(A, solution, residual, residual); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

// linearly combines the last known solver's state (provided by the user) with the current state in such a way that
// the linearly combined states minimize the L2 norm of the residual
double my_p4est_xgfm_cells_t::set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension_on_cells, Vec former_extension_on_nodes,
                                                                              Vec former_rhs, Vec former_residual)
{
  PetscErrorCode ierr;
  PetscReal former_residual_dot_residual, L2_norm_residual;
  ierr = VecDot(former_residual, residual, &former_residual_dot_residual); CHKERRXX(ierr);
  ierr = VecNorm(residual, NORM_2, &L2_norm_residual); CHKERRXX(ierr);
  const double step_size = (SQR(solver_monitor.latest_L2_norm_of_residual()) - former_residual_dot_residual)/(SQR(solver_monitor.latest_L2_norm_of_residual()) - 2.0*former_residual_dot_residual + SQR(L2_norm_residual));

  // doing the state update of relevant internal variable all at once and knowingly avoiding separate Petsc operations that would multiply the number of such loops
  const double *former_extension_on_nodes_p, *former_rhs_p, *former_extension_on_cells_p, *former_solution_p, *former_residual_p;
  double *extension_on_nodes_p, *rhs_p, *extension_on_cells_p, *solution_p, *residual_p;
  ierr = VecGetArrayRead(former_extension_on_nodes, &former_extension_on_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_rhs, &former_rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_extension_on_cells, &former_extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_residual, &former_residual_p); CHKERRXX(ierr);
  ierr = VecGetArray(residual, &residual_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_solution, &former_solution_p); CHKERRXX(ierr);
  ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);
  const p4est_nodes_t* interface_capturing_nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
  double max_correction = 0.0;
  for (size_t idx = 0; idx < MAX(p4est->local_num_quadrants + ghost->ghosts.elem_count, interface_capturing_nodes->indep_nodes.elem_count); ++idx) {
    if(idx < (size_t) p4est->local_num_quadrants) // cell-sampled field without ghosts
    {
      max_correction            = MAX(max_correction, fabs(step_size*(solution_p[idx] - former_solution_p[idx])));
      residual_p[idx]           = (1.0 - step_size)*former_residual_p[idx] + step_size*residual_p[idx];
      rhs_p[idx]                = (1.0 - step_size)*former_rhs_p[idx] + step_size*rhs_p[idx];
    }
    if(idx < p4est->local_num_quadrants + ghost->ghosts.elem_count)
    {
      solution_p[idx]           = (1.0 - step_size)*former_solution_p[idx] + step_size*solution_p[idx];
      extension_on_cells_p[idx] = (1.0 - step_size)*former_extension_on_cells_p[idx] + step_size*extension_on_cells_p[idx];
    }
    if(idx < interface_capturing_nodes->indep_nodes.elem_count)
      extension_on_nodes_p[idx] = (1.0 - step_size)*former_extension_on_nodes_p[idx] + step_size*extension_on_nodes_p[idx];
  }
  ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_solution, &former_solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(residual, &residual_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_residual, &former_residual_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_extension_on_cells, &former_extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_rhs, &former_rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_extension_on_nodes, &former_extension_on_nodes_p); CHKERRXX(ierr);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_correction, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  // the jumps in flux component are multilinear in extension_on_nodes
  // --> need to be updated as well for future consistent extension, since extension_on_nodes and rhs were updated!
  compute_jumps_in_flux_components_at_relevant_interface_capturing_nodes_only();

  return max_correction;
}

bool my_p4est_xgfm_cells_t::solve_for_fixpoint_solution(Vec& former_solution)
{
  P4EST_ASSERT(solution != NULL);
  // save current solution
  std::swap(former_solution, solution);
  PetscErrorCode ierr;
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  // create a new vector if needed
  if(solution == former_solution){
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); } // we need a new one for fixpoint iteration

  ierr = VecCopyGhost(former_solution, solution); CHKERRXX(ierr); // update solution <= former_solution to have a good initial guess for next KSP solve

  KSPConvergedReason termination_reason = solve_linear_system(); CHKERRXX(ierr);
  if(termination_reason <= 0)
    throw std::runtime_error("my_p4est_xgfm_cells_t::solve_for_fixpoint_solution() the Krylov solver failed to converge for one of the subsequent linear systems to solve, the KSPConvergedReason code is " + std::to_string(termination_reason)); // collective runtime_error throw

  PetscInt nksp_iteration;
  ierr = KSPGetIterationNumber(ksp, &nksp_iteration); CHKERRXX(ierr);

  return (termination_reason > 0 && nksp_iteration == 0);
}

void my_p4est_xgfm_cells_t::initialize_extension_on_cells()
{
  P4EST_ASSERT(interface_is_set() && jumps_have_been_set());
  PetscErrorCode ierr;
  P4EST_ASSERT(extension_on_cells != NULL && VecIsSetForCells(extension_on_cells, p4est, ghost, 1));

  const double *solution_p;
  double *extension_on_cells_p;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);

  const my_p4est_hierarchy_t* hierarchy = cell_ngbd->get_hierarchy();
  for (size_t k = 0; k < hierarchy->get_layer_size(); ++k) {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_layer_quadrant(k);
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_layer_quadrant(k);
    initialize_extension_on_cells_local(quad_idx, tree_idx, solution_p, extension_on_cells_p);
  }
  ierr = VecGhostUpdateBegin(extension_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < hierarchy->get_inner_size(); ++k) {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_inner_quadrant(k);
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_inner_quadrant(k);
    initialize_extension_on_cells_local(quad_idx, tree_idx, solution_p, extension_on_cells_p);
  }
  ierr = VecGhostUpdateEnd(extension_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(interface_manager->is_storing_cell_FD_interface_data())
    P4EST_ASSERT(interface_manager->cell_FD_map_is_consistent_across_procs());
#endif

  return;
}

void my_p4est_xgfm_cells_t::initialize_extension_on_cells_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                const double* const &solution_p, double* const &extension_on_cells_p) const
{
  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_q = interface_manager->phi(xyz_quad);

  // build the educated initial guess : we extend u^{-}_{interface} if mu_m is larger, u^{+}_{interface} otherwise
  extension_on_cells_p[quad_idx] = solution_p[quad_idx];
  if(extend_negative_interface_values() && phi_q > 0.0)
    extension_on_cells_p[quad_idx] -= (*interp_jump_u)(xyz_quad);
  else if(!extend_negative_interface_values() && phi_q <= 0.0)
    extension_on_cells_p[quad_idx] += (*interp_jump_u)(xyz_quad);

  return;
}

const my_p4est_xgfm_cells_t::extension_increment_operator&
my_p4est_xgfm_cells_t::get_extension_increment_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double& control_band)
{
  if(extension_operators_are_stored_and_set)
    return pseudo_time_step_increment_operator[quad_idx];

  extension_increment_operator& pseudo_time_step_operator = pseudo_time_step_increment_operator[quad_idx];
  // we'll build the cell-sampled interface-value extension operators, here!
  // the extension procedure, can be formally written as
  // pseudo_time_increment = dtau*(A*cell_values + B*interface_values)
  // where A and B are appropriate (sparse) matrices: here below, we build the entries of A and B.
  double diagonal_coefficient = 0.0;
  double dtau = DBL_MAX;
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_quad = interface_manager->phi(xyz_quad);
  pseudo_time_step_operator.quad_idx            = quad_idx;
  pseudo_time_step_operator.in_band             = fabs(phi_quad) < control_band;
  pseudo_time_step_operator.in_positive_domain  = phi_quad > 0.0;
  double grad_phi[P4EST_DIM]; interface_manager->grad_phi(xyz_quad, grad_phi);
  const double mag_grad_phi = sqrt(SUMD(SQR(grad_phi[0]), SQR(grad_phi[1]), SQR(grad_phi[2])));
  const double signed_normal[P4EST_DIM] = {DIM((mag_grad_phi > EPS ? (phi_quad <= 0.0 ? -1.0 : +1.0)*grad_phi[0]/mag_grad_phi : 0.0),
                                   (mag_grad_phi > EPS ? (phi_quad <= 0.0 ? -1.0 : +1.0)*grad_phi[1]/mag_grad_phi : 0.0),
                                   (mag_grad_phi > EPS ? (phi_quad <= 0.0 ? -1.0 : +1.0)*grad_phi[2]/mag_grad_phi : 0.0))};
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    const u_char oriented_dir = 2*dim + (signed_normal[dim] > 0.0 ? 0 : 1);
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
      continue; // homogeneous Neumann boundary condition on walls (for now, at least, if you have a better idea, go ahead!)

    set_of_neighboring_quadrants neighbor_cells;
    bool derivative_is_one_side;
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, neighbor_cells, derivative_is_one_side);

    if(derivative_is_one_side)
    {
      double discretization_distance = 0.0;
      for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k) {
        const dof_weighted_term& derivative_term = stable_projection_derivative_operator[k];
        if(derivative_term.dof_idx == quad_idx)
          diagonal_coefficient += -derivative_term.weight*signed_normal[oriented_dir/2];
        else
          pseudo_time_step_operator.regular_terms.add_term(derivative_term.dof_idx, -signed_normal[oriented_dir/2]*derivative_term.weight);
        discretization_distance = MAX(discretization_distance, fabs(1.0/derivative_term.weight));
      }
      dtau = MIN(dtau, discretization_distance/(double) P4EST_DIM);
    }
    else
    {
      const double theta = interface_manager->get_FD_theta_between_cells(quad_idx, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir);
      if(theta < EPS) // too close --> force the interface value!
      {
        pseudo_time_step_operator.regular_terms.clear();
        pseudo_time_step_operator.regular_terms.add_term(quad_idx, -1.0);
        pseudo_time_step_operator.interface_terms.clear();
        pseudo_time_step_operator.interface_terms.push_back({+1.0, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir});
        return pseudo_time_step_operator;
      }
      const double coeff = signed_normal[oriented_dir/2]*(oriented_dir%2 == 0 ? +1.0 : -1.0)/(theta*dxyz_min[oriented_dir/2]);
      pseudo_time_step_operator.interface_terms.push_back({coeff, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir});
      diagonal_coefficient -= coeff;
      dtau = MIN(dtau, theta*dxyz_min[oriented_dir/2]/(double) P4EST_DIM);
    }
  }
  pseudo_time_step_operator.regular_terms.add_term(quad_idx, diagonal_coefficient);
  pseudo_time_step_operator.regular_terms *= dtau;
  for (size_t k = 0; k < pseudo_time_step_operator.interface_terms.size(); ++k)
    pseudo_time_step_operator.interface_terms[k].weight *= dtau;
  return pseudo_time_step_operator;
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_of_interface_values(const double& threshold, const uint& niter_max)
{
  P4EST_ASSERT(interface_is_set() && threshold > EPS && niter_max > 0);
  const double *solution_p, *jump_u_p, *jump_flux_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  Vec extension_on_cells_np1; // (next pseudo-time step values)
  ierr = VecCreateGhostCells(p4est, ghost, &extension_on_cells_np1); CHKERRXX(ierr);

  const double control_band = 3.0*diag_min();
  double max_increment_in_band = 10.0*threshold;
  uint iter = 0;
  while (max_increment_in_band > threshold && iter < niter_max)
  {
    double *extension_on_cells_n_p;
    double *extension_on_cells_np1_p;
    ierr = VecGetArray(extension_on_cells, &extension_on_cells_n_p); CHKERRXX(ierr);
    ierr = VecGetArray(extension_on_cells_np1, &extension_on_cells_np1_p); CHKERRXX(ierr);
    max_increment_in_band = 0.0; // reset the measure;
    /* Main loop over all local quadrant */
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k) {
      const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k);
      const extension_increment_operator& extension_increment = get_extension_increment_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k), control_band);
      extension_on_cells_np1_p[quad_idx] = extension_on_cells_n_p[quad_idx]
          + extension_increment(extension_on_cells_n_p, solution_p, jump_u_p, jump_flux_p, *this, max_increment_in_band);
    }
    ierr = VecGhostUpdateBegin(extension_on_cells_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k) {
      const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k);
      const extension_increment_operator& extension_increment = get_extension_increment_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k), control_band);
      extension_on_cells_np1_p[quad_idx] = extension_on_cells_n_p[quad_idx]
          + extension_increment(extension_on_cells_n_p, solution_p, jump_u_p, jump_flux_p, *this, max_increment_in_band);
    }
    if(!extension_operators_are_stored_and_set)
      extension_operators_are_stored_and_set = true;

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_increment_in_band, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = VecGhostUpdateEnd(extension_on_cells_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // swap the vectors before moving on
    Vec tmp = extension_on_cells; extension_on_cells = extension_on_cells_np1; extension_on_cells_np1 = tmp;

    iter++;
  }

  ierr = VecDestroy(extension_on_cells_np1); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  return;
}

void my_p4est_xgfm_cells_t::interpolate_cell_extension_to_interface_capturing_nodes()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_interpolate_cell_extension_to_interface_capturing_nodes, 0, 0, 0, 0); CHKERRXX(ierr);

  if(interface_manager->subcell_resolution() > 1)
    throw std::runtime_error("my_p4est_xgfm_cells_t::interpolate_cell_extension_to_interface_capturing_nodes() is not implemented yet for a subcell resolution greater than 1");

  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();

  const double *extension_on_cells_p;
  double *extension_on_nodes_p;
  ierr = VecGetArrayRead(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);

  for (size_t nnn = 0; nnn < interface_capturing_ngbd_n.get_layer_size(); ++nnn)
  {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_layer_node(nnn);
    extension_on_nodes_p[node_idx] = interpolate_cell_field_at_local_interface_capturing_node(node_idx, interface_capturing_ngbd_n, extension_on_cells_p);
  }
  ierr = VecGhostUpdateBegin(extension_on_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t nnn = 0; nnn < interface_capturing_ngbd_n.get_local_size(); ++nnn)
  {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_local_node(nnn);
    extension_on_nodes_p[node_idx] = interpolate_cell_field_at_local_interface_capturing_node(node_idx, interface_capturing_ngbd_n, extension_on_cells_p);
  }
  ierr = VecGhostUpdateEnd(extension_on_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(!local_interpolators_are_stored_and_set) // if they were not already set, now they are!
    local_interpolators_are_stored_and_set = true;

  ierr = VecRestoreArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_interpolate_cell_extension_to_interface_capturing_nodes, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

double my_p4est_xgfm_cells_t::interpolate_cell_field_at_local_interface_capturing_node(const p4est_locidx_t &node_idx, const my_p4est_node_neighbors_t &interface_capturing_ngbd_n, const double *cell_field_p)
{
  linear_combination_of_dof_t& cell_interpolator = local_interpolators[node_idx];

  if(local_interpolators_are_stored_and_set) // no need for the hardwork if we have already done and memorized it
    return cell_interpolator(cell_field_p);

  cell_interpolator.clear();
  double to_return;
  double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), xyz_node);
  const p4est_indep_t* ni = (p4est_indep_t*) sc_const_array_index(&interface_capturing_ngbd_n.get_nodes()->indep_nodes, node_idx);

  p4est_locidx_t node_idx_on_computational_grid = (interface_capturing_ngbd_n.get_nodes() == nodes ? node_idx : -1);

  if(node_idx_on_computational_grid >= 0 || index_of_node((const p4est_quadrant_t*) ni, nodes, node_idx_on_computational_grid))
  {
    P4EST_ASSERT(node_idx_on_computational_grid >= 0);
    set_of_neighboring_quadrants nearby_cell_neighbors;
    const p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = cell_ngbd->gather_neighbor_cells_of_node(node_idx_on_computational_grid, nodes, nearby_cell_neighbors);
    const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
    to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, nearby_cell_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
    // note : if locally uniform the result of the above is the expected arithmetic average of neighbor cells
  }
  else
  {
    // find smallest (coarse) quad containing point
    p4est_quadrant_t coarse_quad; std::vector<p4est_quadrant_t> remote_matches; remote_matches.resize(0);
    int rank = cell_ngbd->get_hierarchy()->find_smallest_quadrant_containing_point(xyz_node, coarse_quad, remote_matches, false, true); P4EST_ASSERT(rank != -1); (void) rank;
    double quad_xyz[P4EST_DIM]; quad_xyz_fr_q(coarse_quad.p.piggy3.local_num, coarse_quad.p.piggy3.which_tree, p4est, ghost, quad_xyz);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      if(periodicity[dim])
      {
        const double pp = (xyz_node[dim] - quad_xyz[dim])/(xyz_max[dim] - xyz_min[dim]);
        xyz_node[dim] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[dim] - xyz_min[dim]);
      }
    char local_orientation_in_coarse_cell[P4EST_DIM];
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      local_orientation_in_coarse_cell[dim] = (fabs(quad_xyz[dim] - xyz_node[dim]) < EPS*tree_dimensions[dim] ? 0 : (xyz_node[dim] < quad_xyz[dim] ? -1 : +1));

    if(ANDD(local_orientation_in_coarse_cell[0] == 0, local_orientation_in_coarse_cell[1] == 0, local_orientation_in_coarse_cell[1] == 0))
    {
      to_return = cell_field_p[coarse_quad.p.piggy3.local_num];
      cell_interpolator.add_term(coarse_quad.p.piggy3.local_num, 1.0);
      P4EST_ASSERT(cell_interpolator.size() == 1);
    }
    else
    {
      if(SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])) >= P4EST_DIM)
      {
        node_xyz_fr_n(node_idx, interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), xyz_node);
        std::cout << "investigating what can go wrong with this node : " << xyz_node[0] << ", " << xyz_node[1] << std::endl;
        std::cout << "quad center is : " << quad_xyz[0] << ", " << quad_xyz[1] << std::endl;
      }
      P4EST_ASSERT(SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])) < P4EST_DIM); // because if not, it is a vertex on the coarse grid and it should have been found...
      set_of_neighboring_quadrants quads_sharing_subrefined_node;
      quads_sharing_subrefined_node.insert(coarse_quad);
      int8_t max_coarse_level = ((splitting_criteria_t*) p4est->user_pointer)->max_lvl;
      bool fine_uniform_neighborhood = coarse_quad.level == max_coarse_level;
      for (char ii = MIN(local_orientation_in_coarse_cell[0], (char) 0); ii <= MAX(local_orientation_in_coarse_cell[0], (char) 0); ++ii)
        for (char jj = MIN(local_orientation_in_coarse_cell[1], (char) 0); jj <= MAX(local_orientation_in_coarse_cell[1], (char) 0); ++jj)
#ifdef P4_TO_P8
          for (char kk = MIN(local_orientation_in_coarse_cell[2], (char) 0); kk <= MAX(local_orientation_in_coarse_cell[2], (char) 0); ++kk)
#endif
          {
            if(ANDD(ii == 0, jj == 0, kk == 0))
              continue;
            if(ORD(ii != 0 && is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::x + (ii == 1 ? 1 : 0)),
                   jj != 0 && is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::y + (jj == 1 ? 1 : 0)),
                   kk != 0 && is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::z + (kk == 1 ? 1 : 0)))) // it's a wall, nothing to find there
              fine_uniform_neighborhood = false;
            else
            {
              set_of_neighboring_quadrants tmp;
              cell_ngbd->find_neighbor_cells_of_cell(tmp, coarse_quad.p.piggy3.local_num, coarse_quad.p.piggy3.which_tree, DIM(ii, jj, kk));
              P4EST_ASSERT(tmp.size() == 1); // must have found *one* neighbor (or the hierarchy didn't do its job right in the first place)
              fine_uniform_neighborhood = fine_uniform_neighborhood && tmp.begin()->level == max_coarse_level;
              for (set_of_neighboring_quadrants::const_iterator it = tmp.begin(); it != tmp.end(); ++it)
                quads_sharing_subrefined_node.insert(*it);
            }
          }
      P4EST_ASSERT(!fine_uniform_neighborhood || quads_sharing_subrefined_node.size() == (size_t) 2*SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])));
      fine_uniform_neighborhood = fine_uniform_neighborhood && quads_sharing_subrefined_node.size() == (size_t) 2*SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])); // should be useless but let's play safe in release...
      if(fine_uniform_neighborhood) // avoid lsqr in that case, keep it simple
      {
        to_return = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = quads_sharing_subrefined_node.begin(); it != quads_sharing_subrefined_node.end(); ++it)
        {
          to_return += cell_field_p[it->p.piggy3.local_num]/(double) quads_sharing_subrefined_node.size();
          cell_interpolator.add_term(it->p.piggy3.local_num, 1.0/(double) quads_sharing_subrefined_node.size());
        }
        P4EST_ASSERT(cell_interpolator.size() == quads_sharing_subrefined_node.size());
      }
      else
      {
        // we'll build the lsqr interpolant based on the quads sharing the point and their tranverse neighbors
        p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = P4EST_ROOT_LEN;
        // we won't search (extra) neighbors past in the directions where we already found the quads sharing the point (except if the cell is a wall)
        const bool no_search[P4EST_DIM] = {DIM(local_orientation_in_coarse_cell[0] != 0 || is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::x + (local_orientation_in_coarse_cell[0] == 1 ? 1 : 0)),
                                           local_orientation_in_coarse_cell[1] != 0 || is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::y + (local_orientation_in_coarse_cell[1] == 1 ? 1 : 0)),
                                           local_orientation_in_coarse_cell[2] != 0 || is_quad_Wall(p4est, coarse_quad.p.piggy3.which_tree, &coarse_quad, 2*dir::z + (local_orientation_in_coarse_cell[2] == 1 ? 1 : 0)))};
        set_of_neighboring_quadrants lsqr_neighbors;
        for (set_of_neighboring_quadrants::const_iterator it = quads_sharing_subrefined_node.begin(); it != quads_sharing_subrefined_node.end(); ++it)
          logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, cell_ngbd->gather_neighbor_cells_of_cell(*it, lsqr_neighbors, false, no_search));
        const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
        to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, lsqr_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
      }
    }
  }

#ifdef DEBUG
  const double value_check = cell_interpolator(cell_field_p);
  if(ISNAN(value_check))
    std::cout << "my_p4est_xgfm_cells_t::interpolate_cell_field_at_local_node(): tracking a NAN : the local_interpolator returns a NAN for node_idx = " << node_idx
              << " (on the nodes of the interface-capturing grid) on proc " << p4est->mpirank << std::endl;
  P4EST_ASSERT(fabs(value_check - to_return) < MAX(EPS, 0.000001*MAX(fabs(value_check), fabs(to_return))));
#endif

  return to_return;
}

double my_p4est_xgfm_cells_t::get_sharp_flux_component_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, double& phi_face) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(f_idx, dim, quad_idx, tree_idx);
  const u_char oriented_dir = 2*dim + (faces->q2f(quad_idx, 2*dim) == f_idx ? 0 : + 1);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_quad_size = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_quad_size, tree_dimensions[1]*logical_quad_size, tree_dimensions[2]*logical_quad_size)};

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[dim] += (oriented_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[dim];
  phi_face = interface_manager->phi(xyz_face);
  const double phi_q    = interface_manager->phi(xyz_quad);
  const double mu_face  = (phi_face > 0.0 ? mu_plus : mu_minus);
  PetscErrorCode ierr;
  const double *solution_p, *jump_u_p, *jump_flux_p;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  double sharp_flux_component;

  if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
  {
    P4EST_ASSERT(f_idx != NO_VELOCITY);
#ifdef CASL_THROWS
    if(signs_of_phi_are_different(phi_q, phi_face))
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_sharp_flux_component_local(): a wall-cell is crossed by the interface, this is not handled yet...");
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
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities_local(): unknown boundary condition on a wall.");
    }
  }
  else
  {
    set_of_neighboring_quadrants direct_neighbors;
    bool one_sided;
    linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, one_sided);

    if(one_sided)
    {
      if(signs_of_phi_are_different(phi_q, phi_face)) // can be under-resolved :  +-+ or -+- --> derivative operator may be seen as one sided but the face is actually across the interface
      {
        const double flux_component_across = (phi_face > 0.0 ? mu_minus : mu_plus)*stable_projection_derivative(solution_p);

        const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
        p4est_quadrant_t best_match;
        std::vector<p4est_quadrant_t> remote_matches;
        int rank_owner = interface_capturing_ngbd_n.get_hierarchy()->find_smallest_quadrant_containing_point(xyz_face, best_match, remote_matches, true, true);
        P4EST_ASSERT(rank_owner == p4est->mpirank); (void) rank_owner;
        double xyz_q[P4EST_DIM]; // needed only in case of periodicity
        if(ORD(periodicity[0], periodicity[1], periodicity[2]))
          quad_xyz_fr_q(best_match.p.piggy3.local_num, best_match.p.piggy3.which_tree, interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_ghost(), xyz_q);
        for(u_char dim = 0; dim < P4EST_DIM; ++dim)
          if(periodicity[dim])
          {
            const double pp = (xyz_face[dim] - xyz_q[dim])/(xyz_max[dim] - xyz_min[dim]);
            xyz_face[dim] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[dim] - xyz_min[dim]);
          }

        double linear_interpolation_weights[P4EST_CHILDREN];
        get_local_interpolation_weights(interface_capturing_ngbd_n.get_p4est(), best_match.p.piggy3.which_tree, best_match, xyz_face, linear_interpolation_weights);
        double flux_jump_face = 0.0;
        const p4est_nodes_t *interface_capturing_nodes = interface_capturing_ngbd_n.get_nodes();
        for (u_char k = 0; k < P4EST_CHILDREN; ++k)
          flux_jump_face += linear_interpolation_weights[k]*jump_flux_p[P4EST_DIM*interface_capturing_nodes->local_nodes[P4EST_CHILDREN*best_match.p.piggy3.local_num + k] + dim];



        std::cout << "yo man, you're here!" << std::endl;


        sharp_flux_component = flux_component_across + (phi_face > 0.0 ? +1.0 : -1.0)*flux_jump_face;
      }
      else
        sharp_flux_component = mu_face*stable_projection_derivative(solution_p);
    }
    else
    {
      const double &mu_this_side    = (phi_q <= 0.0 ? mu_minus  : mu_plus);
      const double &mu_across       = (phi_q <= 0.0 ? mu_plus   : mu_minus);
      const bool in_positive_domain = (phi_q > 0.0);
      const p4est_quadrant_t& neighbor_quad = *direct_neighbors.begin();
      sharp_flux_component = interface_manager->GFM_flux_at_face_between_cells(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir, mu_this_side, mu_across, in_positive_domain, !signs_of_phi_are_different(phi_q, phi_face), solution_p, jump_u_p, jump_flux_p);
    }
  }
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  return sharp_flux_component;
}

