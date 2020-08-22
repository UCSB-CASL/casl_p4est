#ifdef P4_TO_P8
#include "my_p8est_poisson_jump_cells_xgfm.h"
#else
#include "my_p4est_poisson_jump_cells_xgfm.h"
#endif

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_jump_cells_xgfm_solve_for_sharp_solution;
extern PetscLogEvent log_my_p4est_poisson_jump_cells_xgfm_update_extension_of_interface_values;
extern PetscLogEvent log_my_p4est_poisson_jump_cells_xgfm_update_rhs_and_residual;
#endif

my_p4est_poisson_jump_cells_xgfm_t::my_p4est_poisson_jump_cells_xgfm_t(const my_p4est_cell_neighbors_t *ngbd_c, const p4est_nodes_t* nodes_)
  : my_p4est_poisson_jump_cells_t (ngbd_c, nodes_), activate_xGFM(true) // default behavior is to activate the xGFM corrections
{
  xGFM_absolute_accuracy_threshold  = 1e-8;   // default value
  xGFM_tolerance_on_rel_residual    = 1e-12;  // default value

  residual = extension = grad_jump = NULL;
  interp_grad_jump = NULL;
  pseudo_time_step_increment_operator.resize(p4est->local_num_quadrants);
  extension_operators_are_stored_and_set = false;
  xgfm_jump_between_quads.clear();
  solver_monitor.clear();
  print_residuals_and_corrections_with_solve_info = false;
}

my_p4est_poisson_jump_cells_xgfm_t::~my_p4est_poisson_jump_cells_xgfm_t()
{
  PetscErrorCode ierr;
  if (extension != NULL)  { ierr = VecDestroy(extension); CHKERRXX(ierr); }
  if (residual  != NULL)  { ierr = VecDestroy(residual);  CHKERRXX(ierr); }
  if (solution  != NULL)  { ierr = VecDestroy(solution);  CHKERRXX(ierr); }
  if (grad_jump != NULL)  { ierr = VecDestroy(grad_jump); CHKERRXX(ierr); }
  if (interp_grad_jump != NULL)
    delete interp_grad_jump;
}

void my_p4est_poisson_jump_cells_xgfm_t::get_numbers_of_cells_involved_in_equation_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
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

void my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, int* nullspace_contains_constant_vector)
{
  PetscErrorCode ierr;
  const double *user_rhs_minus_p  = NULL;
  const double *user_rhs_plus_p   = NULL;
  const double *extension_p       = NULL;
  const double *vstar_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *vstar_plus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double       *rhs_p             = NULL;

  P4EST_ASSERT((user_rhs_minus == NULL && user_rhs_plus == NULL) || (user_rhs_minus != NULL && user_rhs_plus != NULL));                     // can't have one provided but not the other...
  P4EST_ASSERT((face_velocity_minus == NULL && face_velocity_plus == NULL) || (face_velocity_minus != NULL && face_velocity_plus != NULL)); // can't have one provided but not the other...
  const bool with_cell_sampled_rhs  = user_rhs_minus != NULL && user_rhs_plus != NULL;
  const bool is_projecting_vstar = face_velocity_minus != NULL && face_velocity_plus != NULL;
#ifdef CASL_THROWS
  if(with_cell_sampled_rhs && is_projecting_vstar)
    std::cerr << "my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad(): [WARNING] the solver is configured with both cell-sampled rhs's and face-sampled velocity fields..." << std::endl;
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
    if(extension != NULL) {
      ierr = VecGetArrayRead(extension, &extension_p); CHKERRXX(ierr); }
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    // initialize the discretized rhs
    rhs_p[quad_idx] = 0.0;
  }

  const PetscInt quad_gloidx = compute_global_index(quad_idx);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double cell_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : 1);
  const double &mu_this_side = (sgn_quad > 0 ? mu_plus : mu_minus);

  /* First add the diagonal term */
  const double& add_diag = (sgn_quad > 0 ? add_diag_plus : add_diag_minus);
  if(!matrix_is_set && fabs(add_diag) > EPS)
  {
    ierr = MatSetValue(A, quad_gloidx, quad_gloidx, cell_volume*add_diag, ADD_VALUES); CHKERRXX(ierr);
    if(nullspace_contains_constant_vector != NULL)
      *nullspace_contains_constant_vector = 0;
  }
  if(!rhs_is_set)
  {
    if(with_cell_sampled_rhs)
      rhs_p[quad_idx] += (sgn_quad < 0 ? user_rhs_minus_p[quad_idx] : user_rhs_plus_p[quad_idx])*cell_volume;
  }

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
      const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : 1);
      if(signs_of_phi_are_different(sgn_quad, sgn_face))
        throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad() : a wall-cell is crossed by the interface, this is not handled yet...");
#endif
      if(!rhs_is_set && is_projecting_vstar)
      {
        const p4est_locidx_t f_idx = interface_manager->get_faces()->q2f(quad_idx, oriented_dir);
        const double* &vstar_to_consider = (sgn_quad < 0 ? vstar_minus_p[oriented_dir/2] : vstar_plus_p[oriented_dir/2]);
        rhs_p[quad_idx] += (oriented_dir%2 == 1 ? -1.0 : +1.0)*face_area*vstar_to_consider[f_idx];
      }
      switch(bc->wallType(xyz_face))
      {
      case DIRICHLET:
      {
        if(!matrix_is_set)
        {
          if(nullspace_contains_constant_vector != NULL)
            *nullspace_contains_constant_vector = 0;
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu_this_side*face_area/cell_dxyz[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
          rhs_p[quad_idx]  += 2.0*mu_this_side*face_area*bc->wallValue(xyz_face)/cell_dxyz[oriented_dir/2];
      }
        break;
      case NEUMANN:
      {
        if(!rhs_is_set)
          rhs_p[quad_idx]  += mu_this_side*face_area*bc->wallValue(xyz_face);
      }
        break;
      default:
        throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad() : unknown boundary condition on a wall.");
      }
      continue;
    }

    set_of_neighboring_quadrants direct_neighbors;
    bool operator_is_one_sided;
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, operator_is_one_sided, (is_projecting_vstar ? &vstar_on_face : NULL));
    if(!rhs_is_set && is_projecting_vstar)
    {
      const double* &vstar_dir_to_consider = (sgn_quad < 0 ? vstar_minus_p[oriented_dir/2] : vstar_plus_p[oriented_dir/2]);
      rhs_p[quad_idx] += (oriented_dir%2 == 1 ? -1.0 : +1.0)*vstar_on_face(vstar_dir_to_consider)*face_area;
    }
    if(operator_is_one_sided)
    {
      if(!matrix_is_set)
        for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k) {
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(stable_projection_derivative_operator[k].dof_idx),
                             (oriented_dir%2 == 1 ? -1.0 : +1.0)*mu_this_side*face_area*stable_projection_derivative_operator[k].weight, ADD_VALUES); CHKERRXX(ierr);
        }
    }
    else
    {
      /* If no one-side, we assume that the interface is tesselated with uniform finest grid level */
      if(direct_neighbors.size() != 1)
        throw std::runtime_error("my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad(): did not find one single direct neighbor for a cell center across the interface. \n Is your grid uniform across the interface?");
      if(quad->level != ((splitting_criteria_t*) p4est->user_pointer)->max_lvl || quad->level != direct_neighbors.begin()->level)
        throw std::runtime_error("my_p4est_poisson_jump_cells_xgfm_t::build_discretization_for_quad(): the interface crosses two cells that are either not of the same size or bigger than expected.");
      const p4est_quadrant_t& neighbor_quad = *direct_neighbors.begin();
      const FD_interface_neighbor& cell_interface_neighbor = interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir);
      const double& mu_across = (sgn_quad > 0 ? mu_minus : mu_plus);
      if(!matrix_is_set)
      {
        const double mu_jump = cell_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across);
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx,                                             mu_jump * face_area/dxyz_min[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, quad_gloidx, compute_global_index(neighbor_quad.p.piggy3.local_num), -mu_jump * face_area/dxyz_min[oriented_dir/2], ADD_VALUES); CHKERRXX(ierr);
      }
      if(!rhs_is_set)
      {
        const xgfm_jump& jump_info = get_xgfm_jump_between_quads(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir);
        rhs_p[quad_idx] += face_area*(oriented_dir%2 == 1 ? +1.0 : -1.0)*cell_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, oriented_dir, (sgn_quad > 0),
                                                                                                                                   jump_info.jump_field, jump_info.jump_flux_component(extension_p), dxyz_min[oriented_dir/2]);
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

    if(extension_p != NULL) {
      ierr = VecRestoreArrayRead(extension, &extension_p); CHKERRXX(ierr); }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  }

  return;
}

const xgfm_jump& my_p4est_poisson_jump_cells_xgfm_t::get_xgfm_jump_between_quads(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir)
{
  couple_of_dofs quad_couple({quad_idx, neighbor_quad_idx});
  map_of_xgfm_jumps_t::const_iterator it = xgfm_jump_between_quads.find(quad_couple);
  if(it != xgfm_jump_between_quads.end())
    return it->second;

  // not found in map --> build it and insert in map
  xgfm_jump to_insert_in_map;
  double xyz_interface_point[P4EST_DIM];
  double normal[P4EST_DIM];
  interface_manager->get_coordinates_of_FD_interface_point_between_cells(quad_idx, neighbor_quad_idx, oriented_dir, xyz_interface_point);
  interface_manager->normal_vector_at_point(xyz_interface_point, normal);

  to_insert_in_map.jump_field                 = (interp_jump_u != NULL            ? (*interp_jump_u)(xyz_interface_point) : 0.0);
  to_insert_in_map.known_jump_flux_component  = (interp_jump_normal_flux != NULL  ? (*interp_jump_normal_flux)(xyz_interface_point)*normal[oriented_dir/2] : 0.0);
  if(activate_xGFM)
  {
    if(interp_grad_jump != NULL)
    {
      double local_grad_jump[P4EST_DIM];
      (*interp_grad_jump)(xyz_interface_point, local_grad_jump);
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
        to_insert_in_map.known_jump_flux_component += (extend_negative_interface_values() ? mu_plus : mu_minus)*((dim == oriented_dir/2 ? 1.0 : 0.0) - normal[oriented_dir/2]*normal[dim])*local_grad_jump[dim];
    }

    if(!mus_are_equal())
      to_insert_in_map.xgfm_jump_flux_component_correction = build_xgfm_jump_flux_correction_operator_at_point(xyz_interface_point, normal, quad_idx, neighbor_quad_idx, oriented_dir/2);
  }

  xgfm_jump_between_quads.insert(std::pair<couple_of_dofs, xgfm_jump>(quad_couple, to_insert_in_map));
  return xgfm_jump_between_quads.at(quad_couple);
}

linear_combination_of_dof_t my_p4est_poisson_jump_cells_xgfm_t::build_xgfm_jump_flux_correction_operator_at_point(const double* xyz, const double* normal,
                                                                                                     const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& flux_component) const
{
  const p4est_quadrant_t quad           = get_quad(quad_idx,          p4est, ghost, true);
  const p4est_quadrant_t neighbor_quad  = get_quad(neighbor_quad_idx, p4est, ghost, true);
  set_of_neighboring_quadrants nearby_cell_neighbors;
  p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = P4EST_ROOT_LEN;
  logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, cell_ngbd->gather_neighbor_cells_of_cell(quad, nearby_cell_neighbors));
  logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, cell_ngbd->gather_neighbor_cells_of_cell(neighbor_quad, nearby_cell_neighbors));
  const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;

  linear_combination_of_dof_t lsqr_cell_grad_operator[P4EST_DIM];
  get_lsqr_cell_gradient_operator_at_point(xyz, cell_ngbd, nearby_cell_neighbors, scaling_distance, lsqr_cell_grad_operator);

  linear_combination_of_dof_t xgfm_flux_correction;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    xgfm_flux_correction.add_operator_on_same_dofs(lsqr_cell_grad_operator[dim], get_jump_in_mu()*((flux_component == dim ? 1.0 : 0.0) - normal[flux_component]*normal[dim]));

  return xgfm_flux_correction;
}

void my_p4est_poisson_jump_cells_xgfm_t::solve_for_sharp_solution(const KSPType& ksp_type, const PCType& pc_type)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_xgfm_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  // make sure the problem is fully defined
  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2])); // boundary conditions
  P4EST_ASSERT(diffusion_coefficients_have_been_set() && interface_is_set());       // essential parameters

  PetscBool saved_ksp_original_guess_flag;
  ierr = KSPGetInitialGuessNonzero(ksp, &saved_ksp_original_guess_flag); // we'll change that one to true internally, but we want to set it back to whatever it originally was

  /* Set the linear system, the linear solver and solve it */
  setup_linear_system();
  ierr = setup_linear_solver(ksp_type, pc_type, xGFM_tolerance_on_rel_residual); CHKERRXX(ierr);

  if(!activate_xGFM || mus_are_equal())
  {
    solve_linear_system();
    solver_monitor.log_iteration(0.0, this); // we just want to log the number of ksp iterations in this case, 0.0 because no correction yet
  }
  else
  {
    // We will need to memorize former solver's states for the xgfm iterative procedure
    Vec former_rhs, former_solution, former_extension, former_residual;
    former_rhs = former_solution = former_extension = former_residual = NULL; // the procedure will adequately determine/create them

    while(update_solution(former_solution))
    {
      // fix-point update
      update_extension_of_interface_values(former_extension);
      update_rhs_and_residual(former_rhs, former_residual);
      // linear combination of the last two solver's states to minimize minimize L2 residual:
      const double max_correction = set_solver_state_minimizing_L2_norm_of_residual(former_solution, former_extension, former_rhs, former_residual);
      solver_monitor.log_iteration(max_correction, this);
      // check if good enough, yet
      if(solver_monitor.reached_convergence_within_desired_bounds(xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual))
        break;
    }
    if(former_solution != NULL){
      ierr = VecDestroy(former_solution); CHKERRXX(ierr); }
    if(former_extension != NULL){
      ierr = VecDestroy(former_extension); CHKERRXX(ierr); }
    if(former_rhs != NULL){
      ierr = VecDestroy(former_rhs); CHKERRXX(ierr); }
    if(former_residual != NULL){
      ierr = VecDestroy(former_residual); CHKERRXX(ierr); }
  }

  ierr = KSPSetInitialGuessNonzero(ksp, saved_ksp_original_guess_flag); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_xgfm_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  return;
}

bool my_p4est_poisson_jump_cells_xgfm_t::update_solution(Vec &former_solution)
{
  PetscErrorCode ierr;

  // save current solution needed by xgfm iterative procedure
  std::swap(former_solution, solution); // save former solution
  if(solution == NULL){
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); }
  if(former_solution != NULL || user_initial_guess != NULL){
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
    if(former_solution != NULL){ // former solution has precedence as initial guess, if defined
      ierr = VecCopyGhost(former_solution, solution); CHKERRXX(ierr); }
    else{ // copy the given initial guess
      ierr = VecCopyGhost(user_initial_guess, solution); CHKERRXX(ierr); }
  }
  solve_linear_system();

  PetscInt nksp_iteration;
  ierr = KSPGetIterationNumber(ksp, &nksp_iteration); CHKERRXX(ierr);

  return nksp_iteration != 0; // if the ksp solver did at least one iteration, there was an update
}

void my_p4est_poisson_jump_cells_xgfm_t::update_extension_of_interface_values(Vec& former_extension, const double& threshold, const uint& niter_max)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_xgfm_update_extension_of_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(interface_is_set() && threshold > EPS && niter_max > 0);

  const double *solution_p;
  const double *current_extension_p = NULL; // --> this feeds the jump conditions that were used to define "solution" so it's required to determine the interface-defined values
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  Vec extension_n, extension_np1; // (extensions at pseudo times n and np1)
  ierr = VecCreateGhostCells(p4est, ghost, &extension_n); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est, ghost, &extension_np1); CHKERRXX(ierr);
  if(extension != NULL){
    ierr = VecCopyGhost(extension, extension_n); CHKERRXX(ierr);
    ierr = VecGetArrayRead(extension, &current_extension_p); CHKERRXX(ierr);
  }
  else
    initialize_extension(extension_n);

  const double control_band = 3.0*diag_min();
  const bool extend_positive_interface_values = !extend_negative_interface_values();
  double max_increment_in_band = 10.0*threshold;
  uint iter = 0;
  while (max_increment_in_band > threshold && iter < niter_max)
  {
    double *extension_n_p, *extension_np1_p;
    ierr = VecGetArray(extension_n,    &extension_n_p); CHKERRXX(ierr);
    ierr = VecGetArray(extension_np1,  &extension_np1_p); CHKERRXX(ierr);
    max_increment_in_band = 0.0; // reset the measure;
    /* Main loop over all local quadrant */
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_layer_size(); ++k) {
      const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_layer_quadrant(k);
      const extension_increment_operator& extension_increment = get_extension_increment_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k), control_band);
      extension_np1_p[quad_idx] = extension_n_p[quad_idx]
          + extension_increment(extension_n_p, solution_p, current_extension_p, *this, extend_positive_interface_values, max_increment_in_band);
    }
    ierr = VecGhostUpdateBegin(extension_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k) {
      const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k);
      const extension_increment_operator& extension_increment = get_extension_increment_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k), control_band);
      extension_np1_p[quad_idx] = extension_n_p[quad_idx]
          + extension_increment(extension_n_p, solution_p, current_extension_p, *this, extend_positive_interface_values, max_increment_in_band);
    }
    if(!extension_operators_are_stored_and_set)
      extension_operators_are_stored_and_set = true;

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_increment_in_band, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = VecGhostUpdateEnd(extension_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(extension_n,    &extension_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(extension_np1,  &extension_np1_p); CHKERRXX(ierr);

    // swap the vectors before moving on
    std::swap(extension_n, extension_np1);

    iter++;
  }

  // restore pointers
  if(current_extension_p != NULL){
    ierr = VecRestoreArrayRead(extension, &current_extension_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);

  // destroy what needs be
  // extension_n is the most advanced in pseudo-time at this point (because of the final "swap" in the loop here above) --> destroy extension_np1
  ierr = VecDestroy(extension_np1);  CHKERRXX(ierr);

  // avoid memory leak, destroy former_extension if needed before making it point to the former extension
  if(former_extension != NULL){
    ierr = VecDestroy(former_extension); CHKERRXX(ierr); }

  former_extension  = extension;
  extension         = extension_n;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_xgfm_update_extension_of_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_poisson_jump_cells_xgfm_t::update_rhs_and_residual(Vec &former_rhs, Vec &former_residual)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_xgfm_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(solution != NULL && rhs != NULL);

  // update the rhs in the cells that are affected by the new jumps in flux components (because of new extension)
  // save the current rhs first
  std::swap(former_rhs, rhs);
  // create a new vector if needed
  if(rhs == NULL){
    ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr);
    ierr = VecCopy(former_rhs, rhs); CHKERRXX(ierr);
  }

  rhs_is_set = false; // lower this flag in order to update the rhs terms appropriately
  std::set<p4est_locidx_t> already_done; already_done.clear();
  for (map_of_xgfm_jumps_t::const_iterator it = xgfm_jump_between_quads.begin(); it != xgfm_jump_between_quads.end(); ++it)
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
  rhs_is_set = true; // rise the flag up again since you're done

  // save the current residual
  std::swap(former_residual, residual);
  // create a new vector if needed:
  if(residual == NULL){
    ierr = VecCreateNoGhostCells(p4est, &residual); CHKERRXX(ierr); }

  // calculate the fix-point residual
  if(scale_system_by_diagonals)
    pointwise_operation_with_sqrt_of_diag(2, solution, multiply_by_sqrt_D, rhs, divide_by_sqrt_D);
  ierr = VecAXPBY(residual, -1.0, 0.0, rhs); CHKERRXX(ierr);
  ierr = MatMultAdd(A, solution, residual, residual); CHKERRXX(ierr);
  if(scale_system_by_diagonals)
    pointwise_operation_with_sqrt_of_diag(3, solution, divide_by_sqrt_D, rhs, multiply_by_sqrt_D, residual, multiply_by_sqrt_D);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_xgfm_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

double my_p4est_poisson_jump_cells_xgfm_t::set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution, Vec former_extension, Vec former_rhs, Vec former_residual)
{
  P4EST_ASSERT(solution != NULL && extension != NULL && rhs !=NULL && residual != NULL);
  if(former_residual == NULL)
  {
    P4EST_ASSERT(former_solution == NULL && former_extension == NULL); // otherwise, something went wrong...
    return 0.0; // if the former residual is not known (0th step), we can't do anything and we need to leave the solver's state as is (0.0 returned because no actual "correction")
  }

  PetscErrorCode ierr;
  PetscReal former_residual_dot_residual, L2_norm_residual;
  ierr = VecDot(former_residual, residual, &former_residual_dot_residual); CHKERRXX(ierr);
  ierr = VecNorm(residual, NORM_2, &L2_norm_residual); CHKERRXX(ierr);
  const double step_size = (SQR(solver_monitor.latest_L2_norm_of_residual()) - former_residual_dot_residual)/(SQR(solver_monitor.latest_L2_norm_of_residual()) - 2.0*former_residual_dot_residual + SQR(L2_norm_residual));

  // doing the state update of relevant internal variable all at once and knowingly avoiding separate Petsc operations that would multiply the number of such loops
  const double *former_rhs_p, *former_extension_p, *former_solution_p, *former_residual_p;
  double *rhs_p, *extension_p, *solution_p, *residual_p;
  ierr = VecGetArrayRead(former_rhs, &former_rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_extension, &former_extension_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension, &extension_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_residual, &former_residual_p); CHKERRXX(ierr);
  ierr = VecGetArray(residual, &residual_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(former_solution, &former_solution_p); CHKERRXX(ierr);
  ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);
  const p4est_nodes_t* interface_capturing_nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
  double max_correction = 0.0;
  for (size_t idx = 0; idx < MAX(p4est->local_num_quadrants + ghost->ghosts.elem_count, interface_capturing_nodes->indep_nodes.elem_count); ++idx) {
    if(idx < (size_t) p4est->local_num_quadrants) // cell-sampled field without ghosts
    {
      max_correction    = MAX(max_correction, fabs(step_size*(solution_p[idx] - former_solution_p[idx])));
      residual_p[idx]   = (1.0 - step_size)*former_residual_p[idx] + step_size*residual_p[idx];
      rhs_p[idx]        = (1.0 - step_size)*former_rhs_p[idx] + step_size*rhs_p[idx];
    }
    if(idx < p4est->local_num_quadrants + ghost->ghosts.elem_count)
    {
      solution_p[idx]   = (1.0 - step_size)*former_solution_p[idx] + step_size*solution_p[idx];
      extension_p[idx]  = (1.0 - step_size)*former_extension_p[idx] + step_size*extension_p[idx];
    }
  }
  ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_solution, &former_solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(residual, &residual_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_residual, &former_residual_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(extension, &extension_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_extension, &former_extension_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(former_rhs, &former_rhs_p); CHKERRXX(ierr);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_correction, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  return max_correction;
}

void my_p4est_poisson_jump_cells_xgfm_t::initialize_extension(Vec cell_sampled_extension)
{
  P4EST_ASSERT(interface_is_set());
  PetscErrorCode ierr;
  P4EST_ASSERT(cell_sampled_extension != NULL && VecIsSetForCells(cell_sampled_extension, p4est, ghost, 1));

  const double *solution_p;
  double *cell_sampled_extension_p;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArray(cell_sampled_extension, &cell_sampled_extension_p); CHKERRXX(ierr);

  const my_p4est_hierarchy_t* hierarchy = cell_ngbd->get_hierarchy();
  for (size_t k = 0; k < hierarchy->get_layer_size(); ++k) {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_layer_quadrant(k);
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_layer_quadrant(k);
    initialize_extension_local(quad_idx, tree_idx, solution_p, cell_sampled_extension_p);
  }
  ierr = VecGhostUpdateBegin(cell_sampled_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < hierarchy->get_inner_size(); ++k) {
    const p4est_topidx_t tree_idx = hierarchy->get_tree_index_of_inner_quadrant(k);
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_inner_quadrant(k);
    initialize_extension_local(quad_idx, tree_idx, solution_p, cell_sampled_extension_p);
  }
  ierr = VecGhostUpdateEnd(cell_sampled_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(cell_sampled_extension, &cell_sampled_extension_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(interface_manager->is_storing_cell_FD_interface_neighbors())
    P4EST_ASSERT(interface_manager->cell_FD_map_is_consistent_across_procs());
#endif

  return;
}

void my_p4est_poisson_jump_cells_xgfm_t::initialize_extension_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                       const double* solution_p, double* extension_p) const
{
  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_q = interface_manager->phi_at_point(xyz_quad);

  // build the educated initial guess : we extend u^{-}_{interface} if mu_m is larger, u^{+}_{interface} otherwise
  extension_p[quad_idx] = solution_p[quad_idx];
  if(extend_negative_interface_values() && phi_q > 0.0)
    extension_p[quad_idx] -= (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0);
  else if(!extend_negative_interface_values() && phi_q <= 0.0)
    extension_p[quad_idx] += (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0);

  return;
}

const my_p4est_poisson_jump_cells_xgfm_t::extension_increment_operator&
my_p4est_poisson_jump_cells_xgfm_t::get_extension_increment_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double& control_band)
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
  const double phi_quad = interface_manager->phi_at_point(xyz_quad);
  pseudo_time_step_operator.quad_idx            = quad_idx;
  pseudo_time_step_operator.in_band             = fabs(phi_quad) < control_band;
  pseudo_time_step_operator.in_positive_domain  = phi_quad > 0.0;
  double signed_normal[P4EST_DIM];
  interface_manager->normal_vector_at_point(xyz_quad, signed_normal, (phi_quad <= 0.0 ? -1.0 : +1.0));
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    const u_char oriented_dir = 2*dim + (signed_normal[dim] > 0.0 ? 0 : 1);
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
      continue; // homogeneous Neumann boundary condition on walls (for now, at least, if you have a better idea, go ahead, be my guest!)

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
      const FD_interface_neighbor& interface_neighbor = interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir);
      if(interface_neighbor.theta < EPS) // too close --> force the interface value!
      {
        pseudo_time_step_operator.regular_terms.clear();
        pseudo_time_step_operator.regular_terms.add_term(quad_idx, -1.0);
        pseudo_time_step_operator.dtau = 0.0; // we force the interface value if too close --> normal derivatives are irrelevant in that case
        pseudo_time_step_operator.interface_terms.clear();
        pseudo_time_step_operator.interface_terms.push_back({+1.0, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir});
        return pseudo_time_step_operator;
      }
      const double coeff = signed_normal[oriented_dir/2]*(oriented_dir%2 == 0 ? +1.0 : -1.0)/(interface_neighbor.theta*dxyz_min[oriented_dir/2]);
      pseudo_time_step_operator.interface_terms.push_back({coeff, neighbor_cells.begin()->p.piggy3.local_num, oriented_dir});
      diagonal_coefficient -= coeff;
      dtau = MIN(dtau, interface_neighbor.theta*dxyz_min[oriented_dir/2]/(double) P4EST_DIM);
    }
  }
  pseudo_time_step_operator.regular_terms.add_term(quad_idx, diagonal_coefficient);
  pseudo_time_step_operator.regular_terms *= dtau;
  for (size_t k = 0; k < pseudo_time_step_operator.interface_terms.size(); ++k)
    pseudo_time_step_operator.interface_terms[k].weight *= dtau;
  pseudo_time_step_operator.dtau = dtau;
  return pseudo_time_step_operator;
}

void my_p4est_poisson_jump_cells_xgfm_t::local_projection_for_face(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces,
                                                                   double* flux_component_minus_p[P4EST_DIM], double* flux_component_plus_p[P4EST_DIM],
                                                                   double* face_velocity_minus_p[P4EST_DIM], double* face_velocity_plus_p[P4EST_DIM]) const
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
  const char sgn_face   = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
  const char sgn_q      = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);
  PetscErrorCode ierr;
  const double *solution_p;
  const double *extension_p = NULL;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  if(extension != NULL){
    ierr = VecGetArrayRead(extension, &extension_p); CHKERRXX(ierr);
  }

  double flux_component_minus, flux_component_plus;
  flux_component_minus = flux_component_plus = NAN;
  double &sharp_flux_at_face  = (sgn_face < 0 ? flux_component_minus : flux_component_plus);
  double &sharp_flux_across   = (sgn_face < 0 ? flux_component_plus  : flux_component_minus);

  if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
  {
    P4EST_ASSERT(f_idx != NO_VELOCITY);
#ifdef CASL_THROWS
    if(signs_of_phi_are_different(sgn_q, sgn_face))
      throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::local_projection_for_face(): a wall-cell is crossed by the interface, this is not handled yet...");
#endif
    switch(bc->wallType(xyz_face))
    {
    case DIRICHLET:
      sharp_flux_at_face = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(2.0*(sgn_face > 0 ? mu_plus : mu_minus)*(bc->wallValue(xyz_face) - solution_p[quad_idx])/cell_dxyz[dim]);
      break;
    case NEUMANN:
      sharp_flux_at_face = (oriented_dir%2 == 1 ? +1.0 : -1.0)*(sgn_face > 0 ? mu_plus : mu_minus)*bc->wallValue(xyz_face);
      break;
    default:
      throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::local_projection_for_face(): unknown boundary condition on a wall.");
    }
  }
  else
  {
    set_of_neighboring_quadrants direct_neighbors;
    bool one_sided;
    linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, oriented_dir, direct_neighbors, one_sided);

    if(one_sided)
    {
      if(signs_of_phi_are_different(sgn_q, sgn_face)) // can be under-resolved :  +-+ or -+- --> derivative operator may be seen as one sided but the face is actually across the interface
      {
        // calculate the flux component as seen from the other side
        sharp_flux_across = (sgn_face > 0 ? mu_minus : mu_plus)*stable_projection_derivative(solution_p);

        // evaluate the jump in flux component as defined consistently with the logic for regular interface point
        const double jump_normal_flux = (interp_jump_normal_flux != NULL ? (*interp_jump_normal_flux)(xyz_face) : 0.0);
        double normal[P4EST_DIM];
        interface_manager->normal_vector_at_point(xyz_face, normal);

        double jump_in_flux_component = jump_normal_flux*normal[dim];
        if(activate_xGFM)
        {
          if(interp_grad_jump != NULL)
          {
            double local_grad_jump[P4EST_DIM];
            (*interp_grad_jump)(xyz_face, local_grad_jump);
            for (u_char dd = 0; dd < P4EST_DIM; ++dd)
              jump_in_flux_component += (extend_negative_interface_values() ? mu_plus : mu_minus)*((dd == dim ? 1.0 : 0.0) - normal[dim]*normal[dd])*local_grad_jump[dd];
          }

          if(!mus_are_equal() && extension_p != NULL)
          {
            linear_combination_of_dof_t xgfm_jump_flux_component_correction = build_xgfm_jump_flux_correction_operator_at_point(xyz_face, normal, quad_idx, direct_neighbors.begin()->p.piggy3.local_num, dim);
            jump_in_flux_component += xgfm_jump_flux_component_correction(extension_p);
          }
        }

        sharp_flux_at_face = sharp_flux_across + (sgn_face > 0 ? +1.0 : -1.0)*jump_in_flux_component;
      }
      else
        sharp_flux_at_face = (sgn_face > 0 ? mu_plus : mu_minus)*stable_projection_derivative(solution_p);
    }
    else
    {
      const double &mu_this_side    = (sgn_q < 0 ? mu_minus  : mu_plus);
      const double &mu_across       = (sgn_q < 0 ? mu_plus   : mu_minus);
      const bool in_positive_domain = (sgn_q > 0);
      P4EST_ASSERT(direct_neighbors.size() == 1);
      const p4est_quadrant_t& neighbor_quad = *direct_neighbors.begin();

      const FD_interface_neighbor interface_neighbor = interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, neighbor_quad.p.piggy3.local_num, oriented_dir);
      const couple_of_dofs quad_couple({quad_idx, neighbor_quad.p.piggy3.local_num});
      map_of_xgfm_jumps_t::const_iterator it = xgfm_jump_between_quads.find(quad_couple);
      if(it == xgfm_jump_between_quads.end())
        throw std::runtime_error("my_p4est_poisson_jump_cells_xgfm_t::local_projection_for_face(): found an interface neighbor that was not stored internally by the solver... Have you called solve()?");

      const xgfm_jump& jump_info = it->second;

      double& flux_quad_side = (sgn_q < 0 ? flux_component_minus : flux_component_plus);
      flux_quad_side = interface_neighbor.GFM_flux_component(mu_this_side, mu_across, oriented_dir, in_positive_domain, solution_p[quad_idx], solution_p[neighbor_quad.p.piggy3.local_num],
          jump_info.jump_field, jump_info.jump_flux_component(extension_p), dxyz_min[oriented_dir/2]);

      if(sgn_q < 0)
        flux_component_plus   = flux_component_minus  + jump_info.jump_flux_component(extension_p);
      else
        flux_component_minus  = flux_component_plus   - jump_info.jump_flux_component(extension_p);
    }
  }
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  if(extension_p != NULL){
    ierr = VecRestoreArrayRead(extension, &extension_p); CHKERRXX(ierr); }

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


void my_p4est_poisson_jump_cells_xgfm_t::initialize_extrapolation_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                                                        double* extrapolation_minus_p, double* extrapolation_plus_p,
                                                                        double* normal_derivative_of_solution_minus_p, double* normal_derivative_of_solution_plus_p, const u_char& degree)
{
  const p4est_quadrant_t* quad;
  const double *xyz_tree_min, *xyz_tree_max;
  double xyz_quad[P4EST_DIM], dxyz_quad[P4EST_DIM];
  fetch_quad_and_tree_coordinates(quad, xyz_tree_min, xyz_tree_max, quad_idx, tree_idx, p4est, ghost);
  xyz_of_quad_center(quad, xyz_tree_min, xyz_tree_max, xyz_quad, dxyz_quad);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);

  if(sgn_quad < 0)
  {
    extrapolation_minus_p[quad_idx] = sharp_solution_p[quad_idx];
    extrapolation_plus_p[quad_idx]  = sharp_solution_p[quad_idx] + (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0); // rough initialization to (hopefully) speed up the convergence in pseudo-time
  }
  else
  {
    extrapolation_plus_p[quad_idx] = sharp_solution_p[quad_idx];
    extrapolation_minus_p[quad_idx] = sharp_solution_p[quad_idx] - (interp_jump_u != NULL ? (*interp_jump_u)(xyz_quad) : 0.0); // rough initialization to (hopefully) speed up the convergence in pseudo-time
  }

  double oriented_normal[P4EST_DIM]; // to calculate the normal derivative of the solution (in the local subdomain) --> the opposite of that vector is required when extrapolating the solution from across the interface
  interface_manager->normal_vector_at_point(xyz_quad, oriented_normal, (sgn_quad < 0 ? +1.0 : -1.0));

  double diagonal_coeff_for_n_dot_grad_this_side = 0.0, diagonal_coeff_for_n_dot_grad_across = 0.0;
  extrapolation_operator_t extrapolation_operator_across, extrapolation_operator_this_side; // ("_this_side" may not be required)

  double n_dot_grad_u = 0.0; // let's evaluate this term on the side of quad, set the corresponding value across to 0.0

  PetscErrorCode ierr;
  const double *extension_p = NULL;
  if(extension != NULL){
    ierr = VecGetArrayRead(extension, &extension_p); CHKERRXX(ierr);
  }

  bool un_is_well_defined = true;

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    double sharp_derivative_m, sharp_derivative_p;
    double dist_m = dxyz_quad[dim], dist_p = dxyz_quad[dim]; // relevant only if fetching some interface neighbor/dirichlet wall BC (i.e., to handle subcell resolution)
    for (u_char orientation = 0; orientation < 2; ++orientation) {
      double &oriented_sharp_derivative = (orientation == 1 ? sharp_derivative_p  : sharp_derivative_m);
      double &oriented_dist             = (orientation == 1 ? dist_p              : dist_m);
      if(is_quad_Wall(p4est, tree_idx, quad, 2*dim + orientation))
      {
        double xyz_wall[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_wall[dim] += (orientation == 1 ? +0.5 : -0.5)*dxyz_quad[dim];
#ifdef CASL_THROWS
        const char sgn_wall = (interface_manager->phi_at_point(xyz_wall) <= 0.0 ? -1 : +1);
        if(signs_of_phi_are_different(sgn_quad, sgn_wall))
          throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::initialize_extrapolation_local(): a wall-cell is crossed by the interface, this is not handled yet...");
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
          throw std::invalid_argument("my_p4est_poisson_jump_cells_xgfm_t::initialize_extrapolation_local(): unknown boundary condition on a wall.");
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
          if(!activate_xGFM)
            un_is_well_defined = false; // interface-fetched values are not accurate in this case --> not reliable to use subcell resolution for initializing n_dot_grad_u

          const double &mu_this_side    = (sgn_quad < 0 ? mu_minus  : mu_plus);
          const double &mu_across       = (sgn_quad < 0 ? mu_plus   : mu_minus);
          const bool in_positive_domain = (sgn_quad > 0);
          P4EST_ASSERT(direct_neighbors.size() == 1);
          const p4est_quadrant_t& neighbor_quad = *direct_neighbors.begin();

          const FD_interface_neighbor interface_neighbor = interface_manager->get_cell_FD_interface_neighbor_for(quad_idx, neighbor_quad.p.piggy3.local_num, 2*dim + orientation);
          const couple_of_dofs quad_couple({quad_idx, neighbor_quad.p.piggy3.local_num});
          map_of_xgfm_jumps_t::const_iterator it = xgfm_jump_between_quads.find(quad_couple);
          if(it == xgfm_jump_between_quads.end())
            throw std::runtime_error("my_p4est_poisson_jump_cells_xgfm_t::initialize_extrapolation_local(): found an interface neighbor that was not stored internally by the solver... Have you called solve()?");

          const xgfm_jump& jump_info = it->second;

          oriented_sharp_derivative = interface_neighbor.GFM_flux_component(mu_this_side, mu_across, 2*dim + orientation, in_positive_domain, sharp_solution_p[quad_idx], sharp_solution_p[neighbor_quad.p.piggy3.local_num],
              jump_info.jump_field, jump_info.jump_flux_component(extension_p), dxyz_min[dim])/mu_this_side; // actually equal to fetching the interface value and using subcell resolution
          oriented_dist = interface_neighbor.theta*dxyz_min[dim];
        }


        // add the (regular, i.e. without interface-fetching) derivative term(s) to the relevant extrapolation operator (for extrapolating normal derivatives, for instance)
        double discretization_distance = 0.0;
        const bool derivative_is_relevant_for_extrapolation_of_this_side = (oriented_normal[dim] <= 0.0 && orientation == 1) || (oriented_normal[dim] > 0.0 && orientation == 0);
        const double relevant_normal_component = (derivative_is_relevant_for_extrapolation_of_this_side ? +1.0 : -1.0)*oriented_normal[dim];
        extrapolation_operator_t& relevant_operator = (derivative_is_relevant_for_extrapolation_of_this_side ? extrapolation_operator_this_side : extrapolation_operator_across);
        double& relevant_diagonal_term = (derivative_is_relevant_for_extrapolation_of_this_side ? diagonal_coeff_for_n_dot_grad_this_side : diagonal_coeff_for_n_dot_grad_across);

        for (size_t k = 0; k < stable_projection_derivative.size(); ++k) {
          const dof_weighted_term& derivative_term = stable_projection_derivative[k];
          if(derivative_term.dof_idx == quad_idx)
            relevant_diagonal_term += derivative_term.weight*relevant_normal_component;
          else
            relevant_operator.n_dot_grad.add_term(derivative_term.dof_idx, relevant_normal_component*derivative_term.weight);
          discretization_distance = MAX(discretization_distance, fabs(1.0/derivative_term.weight));
        }

        relevant_operator.dtau = MIN(relevant_operator.dtau, discretization_distance/(double) P4EST_DIM);
      }
    }

    // the following is equivalent to FD evaluation using the interface-fetched point(s)
    double sharp_derivative_quad_center = (dist_p*sharp_derivative_m + dist_m*sharp_derivative_p)/(dist_p + dist_m);
    if(dist_m + dist_p < 0.1*pow(2.0, -interface_manager->get_max_level_computational_grid())*dxyz_min[dim]) // "0.1" <--> minimum for coarse grid.
      sharp_derivative_quad_center = 0.5*(sharp_derivative_m + sharp_derivative_p); // it was an underresolved case, the above operation is too risky...

    n_dot_grad_u += oriented_normal[dim]*sharp_derivative_quad_center;
  }

  // complete the extrapolation operators
  extrapolation_operator_across.n_dot_grad.add_term(quad_idx, diagonal_coeff_for_n_dot_grad_across);
  extrapolation_operator_this_side.n_dot_grad.add_term(quad_idx, diagonal_coeff_for_n_dot_grad_this_side);

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

  if(sgn_quad < 0)
    extrapolation_operator_plus[quad_idx] = extrapolation_operator_across;
  else
    extrapolation_operator_minus[quad_idx] = extrapolation_operator_across;
  if(!un_is_well_defined)
  {
    if(sgn_quad < 0)
      extrapolation_operator_minus[quad_idx] = extrapolation_operator_this_side;
    else
      extrapolation_operator_plus[quad_idx] = extrapolation_operator_this_side;
  }

  if(extension != NULL){
    ierr = VecRestoreArrayRead(extension, &extension_p); CHKERRXX(ierr);
  }

  return;
}

void my_p4est_poisson_jump_cells_xgfm_t::extrapolate_solution_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double* sharp_solution_p,
                                                                    double* tmp_minus_p, double* tmp_plus_p,
                                                                    const double* extrapolation_minus_p, const double* extrapolation_plus_p,
                                                                    const double* normal_derivative_of_solution_minus_p, const double* normal_derivative_of_solution_plus_p)
{
  tmp_minus_p[quad_idx] = extrapolation_minus_p[quad_idx];
  tmp_plus_p[quad_idx] = extrapolation_plus_p[quad_idx];

  double xyz_quad[P4EST_DIM];
  quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const char sgn_quad = (interface_manager->phi_at_point(xyz_quad) <= 0.0 ? -1 : +1);

  double* extrapolation_np1_p       = (sgn_quad < 0 ? tmp_plus_p : tmp_minus_p);
  const double* extrapolation_n_p   = (sgn_quad < 0 ? extrapolation_plus_p : extrapolation_minus_p);
  const double* normal_derivative_p = (sgn_quad < 0 ? normal_derivative_of_solution_plus_p : normal_derivative_of_solution_minus_p);

  if(activate_xGFM)
  {
    const extension_increment_operator& xgfm_extension_operator = get_extension_increment_operator_for(quad_idx, tree_idx, DBL_MAX);
    const bool fetch_positive_interface_values = (sgn_quad < 0);
    const double *extension_p = NULL;
    PetscErrorCode ierr;
    if(extension != NULL){
      ierr = VecGetArrayRead(extension, &extension_p); CHKERRXX(ierr); }
    double dummy;

    extrapolation_np1_p[quad_idx] = extrapolation_n_p[quad_idx] + xgfm_extension_operator(extrapolation_n_p, sharp_solution_p, extension_p, *this, fetch_positive_interface_values, dummy, normal_derivative_p);

    if(extension != NULL){
      ierr = VecRestoreArrayRead(extension, &extension_p); CHKERRXX(ierr); }
  }
  else
  {
    const extrapolation_operator_t& extrapolation_operator = (sgn_quad < 0 ? extrapolation_operator_plus.at(quad_idx) : extrapolation_operator_minus.at(quad_idx));
    extrapolation_np1_p[quad_idx] -= extrapolation_operator.dtau*(extrapolation_operator.n_dot_grad(extrapolation_n_p) - (normal_derivative_p != NULL ? normal_derivative_p[quad_idx] : 0.0));
  }

  return;
}


