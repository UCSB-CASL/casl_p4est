#ifdef P4_TO_P8
#include "my_p8est_xgfm_cells.h"
#include <p8est_communication.h>
#else
#include "my_p4est_xgfm_cells.h"
#include <p4est_communication.h>
#endif

#ifdef CASL_THROWS
#ifdef P4_TO_P8
#include <p8est_algorithms.h>
#else
#include <p4est_algorithms.h>
#endif
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_xgfm_cells_matrix_preallocation;
extern PetscLogEvent log_my_p4est_xgfm_cells_setup_linear_system;
extern PetscLogEvent log_my_p4est_xgfm_cells_solve;
extern PetscLogEvent log_my_p4est_xgfm_cells_KSPSolve;
extern PetscLogEvent log_my_p4est_xgfm_cells_extend_interface_values;
extern PetscLogEvent log_my_p4est_xgfm_cells_interpolate_cell_extension_to_nodes;
extern PetscLogEvent log_my_p4est_xgfm_cells_update_rhs_and_residual;
#endif

my_p4est_xgfm_cells_t::my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c,
                                             const my_p4est_node_neighbors_t *ngbd_n,
                                             const my_p4est_node_neighbors_t *fine_ngbd_n,
                                             const bool &activate_xGFM_)
  : cell_ngbd(ngbd_c), node_ngbd(ngbd_n),
    p4est(ngbd_c->p4est), nodes(ngbd_n->get_nodes()), ghost(ngbd_c->ghost),
    xyz_min(ngbd_c->p4est->connectivity->vertices + 3*ngbd_c->p4est->connectivity->tree_to_vertex[0]),
    xyz_max(ngbd_c->p4est->connectivity->vertices + 3*ngbd_c->p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(ngbd_c->p4est->trees->elem_count - 1) + P4EST_CHILDREN - 1]),
    tree_dimensions(ngbd_c->get_tree_dimensions()),
    periodicity(ngbd_c->get_hierarchy()->get_periodicity()),
    #ifdef WITH_SUBREFINEMENT
    fine_p4est(fine_ngbd_n->get_p4est()), fine_nodes(fine_ngbd_n->get_nodes()), fine_ghost(fine_ngbd_n->get_ghost()), fine_node_ngbd(fine_ngbd_n),
    interp_subrefined_phi(fine_ngbd_n), interp_subrefined_normals(fine_ngbd_n), interp_subrefined_jump_u(fine_ngbd_n),
    #endif
    activate_xGFM(activate_xGFM_), interface_manager(*this), cell_TVD_extension_operator(*this)
{
  // set up the KSP solver
  PetscErrorCode ierr;
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  mu_minus = mu_plus = -1.0;
  add_diag_minus = add_diag_plus = 0.0;
  user_rhs_minus = user_rhs_plus = rhs = residual = solution = extension_on_cells = extension_on_nodes = jump_flux = NULL;
#ifdef WITH_SUBREFINEMENT
  phi = normals = phi_xxyyzz = NULL;
  jump_u = jump_normal_flux_u = NULL;
#else
  interp_phi = interp_normals = NULL;
  interp_jump_u = interp_jump_normal_flux_u = NULL;
#endif

  A = NULL;
  A_null_space = NULL;
  bc = NULL;
  matrix_is_set = rhs_is_set = false;

  splitting_criteria_t *data      = (splitting_criteria_t*) p4est->user_pointer;
#ifdef WITH_SUBREFINEMENT
  splitting_criteria_t *data_fine = (splitting_criteria_t*) fine_p4est->user_pointer;
#endif

  // Domain and grid parameters
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    dxyz_min[dim]       = tree_dimensions[dim] / (double) (1 << data->max_lvl);
#ifdef WITH_SUBREFINEMENT
    dxyz_min_fine[dim]  = tree_dimensions[dim] / (double) (1 << data_fine->max_lvl);
#endif
  }

#ifdef WITH_SUBREFINEMENT
  local_interpolators.resize(fine_nodes->num_owned_indeps);
#else
  local_interpolators.resize(nodes->num_owned_indeps);
#endif
  local_interpolators_are_set     = false;

  return;
}

my_p4est_xgfm_cells_t::~my_p4est_xgfm_cells_t()
{
  PetscErrorCode ierr;
  if (A                   != NULL)  { ierr = MatDestroy(A);                       CHKERRXX(ierr); }
  if (A_null_space        != NULL)  { ierr = MatNullSpaceDestroy (A_null_space);  CHKERRXX(ierr); }
  if (ksp                 != NULL)  { ierr = KSPDestroy(ksp);                     CHKERRXX(ierr); }
  if (extension_on_cells  != NULL)  { ierr = VecDestroy(extension_on_cells);      CHKERRXX(ierr); }
  if (extension_on_nodes  != NULL)  { ierr = VecDestroy(extension_on_nodes);      CHKERRXX(ierr); }
  if (rhs                 != NULL)  { ierr = VecDestroy(rhs);                     CHKERRXX(ierr); }
  if (residual            != NULL)  { ierr = VecDestroy(residual);                CHKERRXX(ierr); }
  if (solution            != NULL)  { ierr = VecDestroy(solution);                CHKERRXX(ierr); }
  if (jump_flux           != NULL)  { ierr = VecDestroy(jump_flux);               CHKERRXX(ierr); }

  return;
}

void my_p4est_xgfm_cells_t::set_phi(Vec node_sampled_phi, Vec node_sampled_phi_xxyyzz)
{
  P4EST_ASSERT(VecIsSetForNodes(node_sampled_phi, fine_nodes, fine_p4est->mpicomm, 1));
  if(node_sampled_phi_xxyyzz != NULL)
  {
    P4EST_ASSERT(VecIsSetForNodes(node_sampled_phi_xxyyzz, fine_nodes, fine_p4est->mpicomm, P4EST_DIM));
    phi_xxyyzz = node_sampled_phi_xxyyzz;
  }
  phi = node_sampled_phi;
  interp_subrefined_phi.set_input(node_sampled_phi, linear); // in case of subrefinement, we have more node-sampled data everywhere it is required so juste use it like that

  matrix_is_set = rhs_is_set = false;
  return;
}

void my_p4est_xgfm_cells_t::set_normals(Vec node_sampled_normals)
{
  P4EST_ASSERT(node_sampled_normals != NULL && VecIsSetForNodes(node_sampled_normals, fine_nodes, fine_p4est->mpicomm, P4EST_DIM));
  normals = node_sampled_normals;
  interp_subrefined_normals.set_input(normals, linear, P4EST_DIM);

  rhs_is_set = false;
  return;
}

void my_p4est_xgfm_cells_t::set_jumps(Vec node_sampled_jump_u, Vec node_sampled_jump_normal_flux)
{
  P4EST_ASSERT(node_sampled_jump_u != NULL);
  P4EST_ASSERT(node_sampled_jump_normal_flux != NULL);
  P4EST_ASSERT(VecIsSetForNodes(node_sampled_jump_u,            fine_nodes, fine_p4est->mpicomm, 1));
  P4EST_ASSERT(VecIsSetForNodes(node_sampled_jump_normal_flux,  fine_nodes, fine_p4est->mpicomm, 1));
  jump_u              = node_sampled_jump_u;
  jump_normal_flux_u  = node_sampled_jump_normal_flux;

  interp_subrefined_jump_u.set_input(jump_u, linear);

  rhs_is_set = false;
  return;
}

void my_p4est_xgfm_cells_t::compute_subvolumes_in_computational_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const
{
  if(quad_idx >= p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::compute_subvolumes_in_computational_cell(): cannot be called on ghost cells");
  const p4est_tree* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double quad_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

#ifdef WITH_SUBREFINEMENT
  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  if(quadrant_if_subrefined(fine_p4est, fine_nodes, *quad, tree_idx)) // the quadrant is subrefined, find all subrefining quads and do the calculations therein
  {
    std::vector<p4est_locidx_t> indices_of_subrefining_quads;
    fine_node_ngbd->get_hierarchy()->get_all_quadrants_in(quad, tree_idx, indices_of_subrefining_quads);
    P4EST_ASSERT(indices_of_subrefining_quads.size() > 0);
    negative_volume = 0.0;
    const p4est_tree* fine_tree = p4est_tree_array_index(fine_p4est->trees, tree_idx);
    for (size_t k = 0; k < indices_of_subrefining_quads.size(); ++k)
    {
      const p4est_locidx_t& fine_quad_idx = indices_of_subrefining_quads[k];
      const p4est_quadrant_t* fine_quad = p4est_const_quadrant_array_index(&fine_tree->quadrants, fine_quad_idx - fine_tree->quadrants_offset);
      negative_volume += area_in_negative_domain_in_one_quadrant(fine_p4est, fine_nodes, fine_quad, fine_quad_idx, phi);
    }
  }
  else
  {
    const double phi_q = interp_subrefined_phi(xyz_quad);
#ifdef P4EST_DEBUG
    for (char kx = -1; kx < 2; kx += 2)
      for (char ky = -1; ky < 2; ky += 2)
        for (char kz = -1; kz < 2; kz += 2)
        {
          double xyz_vertex[P4EST_DIM] = {DIM(xyz_quad[0] + kx*0.5*cell_dxyz[0], xyz_quad[1] + ky*0.5*cell_dxyz[1], xyz_quad[2] + kz*0.5*cell_dxyz[2])};
          P4EST_ASSERT(!signs_of_phi_are_different(phi_q, interp_subrefined_phi(xyz_vertex)));
        }
#endif
    negative_volume = (phi_q <= 0.0 ? quad_volume : 0.0);
  }
#else
  throw std::runtime_error("my_p4est_xgfm_cells_t::compute_subvolumes_in_computational_cell(): not implemented, yet");
#endif
  P4EST_ASSERT(0.0 <= negative_volume && negative_volume <= quad_volume);
  positive_volume = MAX(0.0, MIN(quad_volume, quad_volume - negative_volume));

  return;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_at_all_nodes()
{
  PetscErrorCode ierr;
  P4EST_ASSERT(jump_flux != NULL);
  const double *jump_u_p, *jump_normal_flux_u_p, *normals_p;
  const double *extension_on_nodes_p = NULL;
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normals, &normals_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecGetArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }
  double *jump_flux_p;
  ierr = VecGetArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);

  for (size_t k = 0; k < fine_node_ngbd->get_layer_size(); ++k)
    compute_jumps_in_flux_components_for_node(fine_node_ngbd->get_layer_node(k), jump_flux_p, jump_normal_flux_u_p, normals_p, jump_u_p, extension_on_nodes_p);
  ierr = VecGhostUpdateBegin(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < fine_node_ngbd->get_local_size(); ++k)
    compute_jumps_in_flux_components_for_node(fine_node_ngbd->get_local_node(k), jump_flux_p, jump_normal_flux_u_p, normals_p, jump_u_p, extension_on_nodes_p);
  ierr = VecGhostUpdateEnd(jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normals, &normals_p); CHKERRXX(ierr);
  if(extension_on_nodes != NULL){
    ierr = VecRestoreArrayRead(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }

  return;
}

void my_p4est_xgfm_cells_t::interface_manager_t::update_jumps_in_flux_at_all_relevant_nodes() const
{
  PetscErrorCode ierr;
  P4EST_ASSERT(solver.jump_flux != NULL);
  const double *jump_u_p, *jump_normal_flux_u_p, *normals_p;
  const double *extension_on_nodes_p = NULL;
  ierr = VecGetArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.normals, &normals_p); CHKERRXX(ierr);
  if(solver.extension_on_nodes != NULL){
    ierr = VecGetArrayRead(solver.extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }
  double *jump_flux_p;
  ierr = VecGetArray(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);

  const my_p4est_node_neighbors_t* ngbd_n = solver.fine_node_ngbd;

  p4est_locidx_t former_quad_idx = -1;
  for (map_of_interface_neighbors_t::const_iterator it = interface_data.begin(); it != interface_data.end(); ++it) {
    const p4est_locidx_t& quad_idx = it->first.loc_idx;
    const bool center_fine_node_alread_done = (quad_idx == former_quad_idx);

    const interface_neighbor& int_nb = it->second;
    if(int_nb.quad_fine_node_idx < ngbd_n->get_nodes()->num_owned_indeps && !center_fine_node_alread_done)
      solver.compute_jumps_in_flux_components_for_node(int_nb.quad_fine_node_idx, jump_flux_p, jump_normal_flux_u_p, normals_p, jump_u_p, extension_on_nodes_p);
    if(int_nb.mid_point_fine_node_idx < ngbd_n->get_nodes()->num_owned_indeps)
      solver.compute_jumps_in_flux_components_for_node(int_nb.mid_point_fine_node_idx, jump_flux_p, jump_normal_flux_u_p, normals_p, jump_u_p, extension_on_nodes_p);
    if(int_nb.neighbor_fine_node_idx < ngbd_n->get_nodes()->num_owned_indeps)
      solver.compute_jumps_in_flux_components_for_node(int_nb.neighbor_fine_node_idx, jump_flux_p, jump_normal_flux_u_p, normals_p, jump_u_p, extension_on_nodes_p);
    former_quad_idx = quad_idx;
  }

  ierr = VecGhostUpdateBegin(solver.jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solver.jump_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.jump_normal_flux_u, &jump_normal_flux_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.normals, &normals_p); CHKERRXX(ierr);
  if(solver.extension_on_nodes != NULL){
    ierr = VecRestoreArrayRead(solver.extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr); }

  return;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_for_node(const p4est_locidx_t& node_idx, double *jump_flux_p,
                                                                      const double *jump_normal_flux_p, const double *normals_p, const double *jump_u_p, const double *extension_on_nodes_p) const
{
  double grad_jump_u[P4EST_DIM];
  double grad_extension[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)}; // set to 0.0 to avoid undefined behavior if extension_on_nodes_p == NULL

  const quad_neighbor_nodes_of_node_t *qnnn;
  if(activate_xGFM)
  {
    fine_node_ngbd->get_neighbors(node_idx, qnnn);
    if(extension_on_nodes_p != NULL)
    {
      const double *inputs[2] = {jump_u_p,    extension_on_nodes_p};
      double *outputs[2]      = {grad_jump_u, grad_extension};
      qnnn->gradient(inputs, outputs, 2);
    }
    else
      qnnn->gradient(jump_u_p, grad_jump_u);
  }

  const double *grad_phi = normals_p + P4EST_DIM*node_idx;
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
      jump_flux_p[P4EST_DIM*node_idx + dim] += get_smaller_mu()*(grad_jump_u[dim] - grad_jump_u_cdot_normal*normal[dim]);
      if(extension_on_nodes_p != NULL)
        jump_flux_p[P4EST_DIM*node_idx + dim] += get_jump_in_mu()*(grad_extension[dim] - grad_extension_cdot_normal*normal[dim]);
    }
  }

  return;
}

linear_combination_of_dof_t my_p4est_xgfm_cells_t::stable_projection_derivative_operator_at_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const u_char& face_direction,
                                                                                                 set_of_neighboring_quadrants &direct_neighbors, bool& all_cell_centers_on_same_side) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  P4EST_ASSERT(!is_quad_Wall(p4est, tree_idx, quad, face_direction)); // we do not allow for wall faces in here
  double xyz_[P4EST_DIM];

  direct_neighbors.clear();
  cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, face_direction);
  P4EST_ASSERT(direct_neighbors.size() > 0);

  const bool quad_is_major            = (direct_neighbors.size() > 1 || (direct_neighbors.size() == 1 && direct_neighbors.begin()->level == quad->level));
  const bool major_quad_is_leading    = (face_direction%2 == (quad_is_major ?  0 : 1));
  const p4est_quadrant_t& major_quad  = (quad_is_major ? *quad : *direct_neighbors.begin());
  quad_xyz_fr_q((quad_is_major ? quad_idx : major_quad.p.piggy3.local_num), (quad_is_major ? tree_idx : major_quad.p.piggy3.which_tree), p4est, ghost, xyz_);
  const double phi_major_quad = interp_subrefined_phi(xyz_);
  all_cell_centers_on_same_side = true;

  set_of_neighboring_quadrants *minor_quads = NULL;
  if(quad_is_major)
    minor_quads = &direct_neighbors;
  else
  {
    minor_quads = new set_of_neighboring_quadrants;
    cell_ngbd->find_neighbor_cells_of_cell(*minor_quads, major_quad.p.piggy3.local_num, major_quad.p.piggy3.which_tree, face_direction + (face_direction%2 == 0 ? 1 : -1)); // find all other quads sharing the face
  }

  const double shared_surface = pow((double) P4EST_QUADRANT_LEN(major_quad.level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1); // (logical) shared surface
  double discretization_distance = 0.0;
#ifdef DEBUG
  double split_face_check = 0.0; bool quad_is_among_sharers = quad_is_major;
#endif
  for (set_of_neighboring_quadrants::const_iterator it = minor_quads->begin(); it != minor_quads->end(); ++it)
  {
    const double surface_ratio = pow((double) P4EST_QUADRANT_LEN(it->level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1)/shared_surface;
    discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(major_quad.level) + P4EST_QUADRANT_LEN(it->level))/(double) P4EST_ROOT_LEN;
#ifdef DEBUG
    split_face_check += surface_ratio; quad_is_among_sharers = quad_is_among_sharers || it->p.piggy3.local_num == quad_idx;
#endif
    quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_);
    all_cell_centers_on_same_side = all_cell_centers_on_same_side && !signs_of_phi_are_different(phi_major_quad, interp_subrefined_phi(xyz_));
  }
  P4EST_ASSERT(quad_is_among_sharers && fabs(split_face_check - 1.0) < EPS);
  discretization_distance *= tree_dimensions[face_direction/2];

  linear_combination_of_dof_t local_derivative_operator;
  for (set_of_neighboring_quadrants::const_iterator it = minor_quads->begin(); it != minor_quads->end(); ++it)
  {
    const double surface_ratio = pow((double) P4EST_QUADRANT_LEN(it->level)/(double) P4EST_ROOT_LEN, (double) P4EST_DIM - 1)/shared_surface;
    local_derivative_operator.add_term(it->p.piggy3.local_num, (major_quad_is_leading ? -1.0 : +1.0)*surface_ratio/discretization_distance);
  }
  local_derivative_operator.add_term((quad_is_major ? quad_idx : major_quad.p.piggy3.local_num), (major_quad_is_leading ? +1.0 : -1.0)/discretization_distance);

  if(minor_quads != &direct_neighbors)
    delete minor_quads;

  return local_derivative_operator;
}

void my_p4est_xgfm_cells_t::preallocate_matrix()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(!matrix_is_set);

  if (A != NULL){
    ierr = MatDestroy(A); CHKERRXX(ierr); }

  PetscInt num_owned_global = p4est->global_num_quadrants;
  PetscInt num_owned_local  = p4est->local_num_quadrants;

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; q++)
    {
      set_of_neighboring_quadrants neighbor_quads_involved;
      const p4est_quadrant_t *quad  = p4est_const_quadrant_array_index(&tree->quadrants, q);
      const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      for(u_char face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
      {
        set_of_neighboring_quadrants direct_neighbors;
        cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, face_dir);

        for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
          neighbor_quads_involved.insert(*it);

        if(direct_neighbors.size() == 1 && direct_neighbors.begin()->level < quad->level)
          cell_ngbd->find_neighbor_cells_of_cell(neighbor_quads_involved, direct_neighbors.begin()->p.piggy3.local_num, direct_neighbors.begin()->p.piggy3.which_tree, face_dir%2 == 0 ? face_dir + 1 : face_dir - 1);
      }

      for (set_of_neighboring_quadrants::const_iterator it = neighbor_quads_involved.begin(); it != neighbor_quads_involved.end(); ++it) {
        if(it->p.piggy3.local_num < num_owned_local)
          d_nnz[quad_idx]++;
        else
          o_nnz[quad_idx]++;
      }
    }
  }

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_xgfm_cells_t::build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const double *user_rhs_minus_p, const double *user_rhs_plus_p,
                                                          const double *jump_u_p, const double *jump_flux_p,
                                                          double* rhs_p, int* nullspace_contains_constant_vector)
{
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double cell_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

  const PetscInt quad_gloidx = compute_global_index(quad_idx);
  PetscErrorCode ierr;

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_quad = interp_subrefined_phi(xyz_quad);
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

  for(u_char face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
  {
    const double face_area = cell_volume/cell_dxyz[face_dir/2];

    /* first check if the cell is a wall
     * We will assume that walls are not crossed by the interface, in a first attempt! */
    if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
    {
      double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])};
      xyz_face[face_dir/2] += (face_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[face_dir/2];
#ifdef CASL_THROWS
      const double phi_face = interp_subrefined_phi(xyz_face);
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
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu_here*face_area/cell_dxyz[face_dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
          rhs_p[quad_idx]  += 2.0*mu_here*face_area*bc->wallValue(xyz_face)/cell_dxyz[face_dir/2];
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
    linear_combination_of_dof_t stable_projection_derivative_operator = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, face_dir, direct_neighbors, operator_is_one_sided);
    if(operator_is_one_sided)
    {
      if(!matrix_is_set)
        for (size_t k = 0; k < stable_projection_derivative_operator.size(); ++k) {
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(stable_projection_derivative_operator[k].dof_idx),
                             (face_dir%2 == 1 ? -1.0 : +1.0)*mu_here*face_area*stable_projection_derivative_operator[k].weight, ADD_VALUES); CHKERRXX(ierr);
        }
    }
    else
    {
      /* If no one-side, we assume that the interface is tesselated with uniform finest grid level */
      if(direct_neighbors.size() != 1)
        throw std::runtime_error("my_p4est_xgfm_cells_t::build_discretization_for_quad(): did not find one single direct neighbor for a cell center across the interface. \n Is your grid uniform across the interface?");
      if(quad->level != ((splitting_criteria_t*) p4est->user_pointer)->max_lvl || quad->level != direct_neighbors.begin()->level)
        throw std::runtime_error("my_p4est_xgfm_cells_t::build_discretization_for_quad(): the interface crosses two cells that are either not of the same size or bigger than expected.");
      const double& mu_across = (phi_quad > 0.0 ? mu_minus : mu_plus);
      if(!matrix_is_set)
      {
        const double mu_jump = interface_manager.GFM_mu_jump(quad_idx, face_dir, mu_here, mu_across);
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx,                                                         mu_jump * face_area/dxyz_min[face_dir/2], ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, quad_gloidx, compute_global_index(direct_neighbors.begin()->p.piggy3.local_num), -mu_jump * face_area/dxyz_min[face_dir/2], ADD_VALUES); CHKERRXX(ierr);
      }
      if(!rhs_is_set)
        rhs_p[quad_idx] += face_area*(face_dir%2 == 1 ? +1.0 : -1.0)*interface_manager.GFM_jump_terms_for_flux_component(quad_idx, face_dir, mu_here, mu_across, (phi_quad > 0.0), jump_u_p, jump_flux_p);
    }
  }

  return;
}

void my_p4est_xgfm_cells_t::setup_linear_system()
{
  if(matrix_is_set && rhs_is_set)
    return;

  if(!matrix_is_set)
    preallocate_matrix();

  PetscErrorCode ierr;
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  const double *phi_p;
  const double *jump_u_p     = NULL;
  const double *jump_flux_p  = NULL;
  const double *phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }

  double *rhs_p = NULL;
  const double *user_rhs_minus_p = NULL;
  const double *user_rhs_plus_p = NULL;
  if(!rhs_is_set)
  {
    if(jump_flux == NULL)
    {
      if(!normals_have_been_set() || !diffusion_coefficients_have_been_set() || !jumps_have_been_set())
        throw std::runtime_error("my_p4est_xgfm_cells_t::setup_linear_system() : the jump in flux components can't be calculated if the normals, diffusion coefficients or required jumps in solution have not been set beforehand");
      ierr = VecCreateGhostNodesBlock(fine_p4est, fine_nodes, P4EST_DIM, &jump_flux); CHKERRXX(ierr);
      compute_jumps_in_flux_components_at_all_nodes(); // we need the jumps in flux components in order to assemble the (discretized) rhs and we don't know yet exactly where --> do it everywhere
    }

    if(user_rhs_minus == NULL || user_rhs_plus == NULL)
      throw std::runtime_error("my_p4est_xgfm_cells_t::setup_linear_system() : the user must set its cell-sampled rhs's, first!");
    ierr  = VecGetArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
    ierr  = VecGetArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);

    if(rhs == NULL){
      ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr); }
    P4EST_ASSERT(VecIsSetForCells(rhs, p4est, ghost, 1, false));

    ierr  = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr  = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
    ierr  = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  }
  int nullspace_contains_constant_vector = !matrix_is_set; // converted to integer because of required MPI collective determination thereafter, we don't care about that if the matrix is already set

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      build_discretization_for_quad(q + tree->quadrants_offset, tree_idx, user_rhs_minus_p, user_rhs_plus_p,
                                    jump_u_p, jump_flux_p, rhs_p, &nullspace_contains_constant_vector);
  }

  if(!matrix_is_set)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

    // check for null space
    if(A_null_space != NULL){
      ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);
      A_null_space = NULL;
    }
    ierr = MPI_Allreduce(MPI_IN_PLACE, &nullspace_contains_constant_vector, 1, MPI_INT, MPI_LAND, p4est->mpicomm); CHKERRXX(ierr);
    if (nullspace_contains_constant_vector)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, NULL, &A_null_space); CHKERRXX(ierr);
      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
    }
  }
  /* [Raphael (05/17/2020) :
   * removing the null space from the rhs seems redundant with PetSc operations done under the
   * hood in KSPSolve (see the source code of KSPSolve_Private in /src/kscp/ksp/interface/itfunc.c
   * for more details). --> So long as MatSetNullSpace() was called appropriately on the matrix
   * operator, this operation will be executed in the pre-steps of KSPSolve on a COPY of the provided
   * RHS vector]
   * --> this is what we want : we need the _unmodified_ RHS thereafter, i.e. as we have built it, and
   * we let PetSc do its magic under the hood every time we call KSPSolve, otherwise iteratively
   * correcting and updating the RHS would becomes very complex and require extra info coming from those
   * possibly non-empty nullspace contributions...
   * */

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  if(phi_xxyyzz_p != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  if(jump_u_p != NULL){
    ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr); }
  if(jump_flux_p != NULL){
    ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr); }
  if(rhs_p != NULL){
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr); }
  if(user_rhs_minus_p != NULL){
    ierr = VecRestoreArrayRead(user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr); }
  if(user_rhs_plus_p != NULL){
    ierr = VecRestoreArrayRead(user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr); }

  matrix_is_set = rhs_is_set = true;

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  return;
}

PetscErrorCode my_p4est_xgfm_cells_t::setup_linear_solver(const KSPType& ksp_type, const PCType& pc_type, const double &tolerance_on_rel_residual) const
{
  PetscErrorCode ierr;
  ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER); CHKERRQ(ierr);
  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, tolerance_on_rel_residual, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRQ(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
   * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
   * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
   */
  if (!strcmp(pc_type, PCHYPRE)){
    /* 1- Strong threshold:
     * Between 0 to 1
     * "0 "gives better convergence rate (in 3D).
     * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
    */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRQ(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRQ(ierr);

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRQ(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (A_null_space != NULL){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRQ(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  return ierr;
}

KSPConvergedReason my_p4est_xgfm_cells_t::solve_linear_system()
{
  PetscErrorCode ierr;

  if(solution == NULL) {
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); }

  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  // we need update ghost values of the solution for accurate calculation of the extended interface values
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  KSPConvergedReason convergence_reason;
  ierr = KSPGetConvergedReason(ksp, &convergence_reason); CHKERRXX(ierr);

  return convergence_reason; // positive values indicate convergence
}

void my_p4est_xgfm_cells_t::solve(KSPType ksp_type, PCType pc_type, double absolute_accuracy_threshold, double tolerance_on_rel_residual)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2]));           // make sure we have wall boundary conditions if we need them
  P4EST_ASSERT(levelset_has_been_set() && normals_have_been_set() && jumps_have_been_set());  // make sure the problem is fully defined
  P4EST_ASSERT(absolute_accuracy_threshold > EPS && tolerance_on_rel_residual > EPS);         // those need to be (strictly positive)

  PetscBool saved_ksp_original_guess_flag;
  ierr = KSPGetInitialGuessNonzero(ksp, &saved_ksp_original_guess_flag); // we'll change that one to true internally, but we want to set it back to whatever it originally was

  /* clear the solver monitoring */
  solver_monitor.clear();

  /* Set the linear system, the linear solver and solve it (regular GFM, i.e., "Boundary Condition-Capturing scheme...")*/
  setup_linear_system();
  ierr = setup_linear_solver(ksp_type, pc_type, tolerance_on_rel_residual); CHKERRXX(ierr);
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
      interface_manager.update_jumps_in_flux_at_all_relevant_nodes(); // those jumps are function of extension_on_nodes so they need to be updated first if known for whatever reason
    // fix-point update
    extend_interface_values(former_extension_on_cells, former_extension_on_nodes);
    update_rhs_and_residual(former_rhs, former_residual);
    solver_monitor.log_iteration(0.0, this);


    while (!solver_monitor.reached_converged_within_desired_bounds(absolute_accuracy_threshold, tolerance_on_rel_residual)
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
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

  return;
}

// this function assumes that the (relevant) jumps_in_flux_components are up-to-date and will reset the extension of the appropriate interface-defined value
void my_p4est_xgfm_cells_t::extend_interface_values(Vec &former_extension_on_cells, Vec &former_extension_on_nodes, const double& threshold, const uint& niter_max)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_extend_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(levelset_has_been_set() && jumps_have_been_set() && threshold > EPS && niter_max > 0);

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
    ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_nodes); CHKERRXX(ierr); // we need a new vector to store the interpolated result
  }
  interpolate_cell_extension_to_nodes();

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_extend_interface_values, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

void my_p4est_xgfm_cells_t::
interface_manager_t::update_rhs_in_relevant_cells_only() const
{
  PetscErrorCode ierr;
  // update the rhs values associated with cells involving jump conditions (fix-point update)
  double *rhs_p;
  ierr = VecGetArray(solver.rhs, &rhs_p); CHKERRXX(ierr);
  const double *phi_p, *phi_xxyyzz_p, *jump_u_p, *jump_flux_p, *user_rhs_minus_p, *user_rhs_plus_p;
  ierr = VecGetArrayRead(solver.user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(solver.phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(solver.phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }

#ifdef DEBUG
  splitting_criteria_t* data = (splitting_criteria_t*) solver.p4est->user_pointer;
#endif

  solver.rhs_is_set = false;
  p4est_locidx_t previous_quad_idx = -1;
  for (map_of_interface_neighbors_t::const_iterator it = interface_data.begin(); it != interface_data.end(); ++it)
  {
    const p4est_locidx_t quad_idx  = it->first.loc_idx;
    if(quad_idx != previous_quad_idx)
    {
      // find tree_idx
      p4est_topidx_t tree_idx  = solver.p4est->first_local_tree;
      while(tree_idx < solver.p4est->last_local_tree && quad_idx >= p4est_tree_array_index(solver.p4est->trees, tree_idx + 1)->quadrants_offset){ tree_idx++; }
#ifdef P4EST_DEBUG
      // get the quadrant
      const p4est_tree_t* tree  = p4est_tree_array_index(solver.p4est->trees, tree_idx);
      P4EST_ASSERT(quad_idx - tree->quadrants_offset >= 0);
      const p4est_quadrant_t *quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
      P4EST_ASSERT(quad->level == data->max_lvl);
#endif
      solver.build_discretization_for_quad(quad_idx, tree_idx, user_rhs_minus_p, user_rhs_plus_p, jump_u_p, jump_flux_p, rhs_p);
    }
    previous_quad_idx = quad_idx;
  }

  solver.rhs_is_set = true;

  if(solver.phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(solver.phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.user_rhs_minus, &user_rhs_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.user_rhs_plus, &user_rhs_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(solver.rhs, &rhs_p); CHKERRXX(ierr);

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
  interface_manager.update_jumps_in_flux_at_all_relevant_nodes();
  // update the rhs consistently with those new jumps in flux components
  // save the rhs first
  std::swap(former_rhs, rhs);
  // create a new vector if needed
  if(rhs == former_rhs){
    ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr);
    ierr = VecCopy(former_rhs, rhs); CHKERRXX(ierr);
  }

  interface_manager.update_rhs_in_relevant_cells_only();

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
  double max_correction = 0.0;
  for (size_t idx = 0; idx < MAX(p4est->local_num_quadrants + ghost->ghosts.elem_count, fine_nodes->indep_nodes.elem_count); ++idx) {
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
    if(idx < fine_nodes->indep_nodes.elem_count)
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
  interface_manager.update_jumps_in_flux_at_all_relevant_nodes();

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

void my_p4est_xgfm_cells_t::interface_manager_t::set_current_interface_point_for(const p4est_locidx_t& quad_idx, const u_char& face_dir)
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < solver.p4est->local_num_quadrants && face_dir < P4EST_FACES); // must be a local quadrant
  current_interface_data = find_interface_neighbor_in_map(quad_idx, face_dir);
  if(current_interface_data != interface_data.end())
    return;

  interface_neighbor interface_nb;

  p4est_topidx_t tree_idx  = solver.p4est->first_local_tree;
  while (tree_idx < solver.p4est->last_local_tree && quad_idx >= p4est_tree_array_index(solver.p4est->trees, tree_idx + 1)->quadrants_offset){ tree_idx++; }

  set_of_neighboring_quadrants direct_neighbor;
  solver.cell_ngbd->find_neighbor_cells_of_cell(direct_neighbor, quad_idx, tree_idx, face_dir);
  P4EST_ASSERT(direct_neighbor.size() == 1);
  const p4est_tree_t*     tree = p4est_tree_array_index(solver.p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const p4est_quadrant_t& neighbor_quad = *direct_neighbor.begin();
  const p4est_locidx_t fine_node_idx_for_quad           = get_fine_node_idx_of_quad_center(solver.fine_p4est, solver.fine_nodes, *quad, tree_idx);
  const p4est_locidx_t fine_node_idx_for_neighbor_quad  = get_fine_node_idx_of_quad_center(solver.fine_p4est, solver.fine_nodes, neighbor_quad, neighbor_quad.p.piggy3.which_tree);
  if(!(fine_node_idx_for_quad >= 0 && fine_node_idx_for_neighbor_quad >= 0))
    std::cout << "fine_node_idx_for_quad = " << fine_node_idx_for_quad  << ", fine_node_idx_for_neighbor_quad" << fine_node_idx_for_neighbor_quad << std::endl;
  P4EST_ASSERT(0 <= fine_node_idx_for_quad          && fine_node_idx_for_quad < (p4est_locidx_t)(solver.fine_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(0 <= fine_node_idx_for_neighbor_quad && fine_node_idx_for_neighbor_quad < (p4est_locidx_t)(solver.fine_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(quad->level == neighbor_quad.level && quad->level == (int8_t) ((splitting_criteria_t*) solver.p4est->user_pointer)->max_lvl);

  PetscErrorCode ierr;
  const double *phi_p, *phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(solver.get_subrefined_phi(), &phi_p); CHKERRXX(ierr);
  if(solver.phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(solver.phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  P4EST_ASSERT(signs_of_phi_are_different(phi_p[fine_node_idx_for_quad], phi_p[fine_node_idx_for_neighbor_quad]));
  interface_nb.neighbor_quad_idx      = neighbor_quad.p.piggy3.local_num;
  interface_nb.quad_fine_node_idx     = fine_node_idx_for_quad;
  interface_nb.neighbor_fine_node_idx = fine_node_idx_for_neighbor_quad;
  const quad_neighbor_nodes_of_node_t* qnnn; solver.fine_node_ngbd->get_neighbors(fine_node_idx_for_quad, qnnn);
  interface_nb.mid_point_fine_node_idx  = qnnn->neighbor(face_dir);
  P4EST_ASSERT(0 <= interface_nb.mid_point_fine_node_idx && interface_nb.mid_point_fine_node_idx == solver.fine_node_ngbd->get_neighbors(fine_node_idx_for_neighbor_quad).neighbor(face_dir + (face_dir%2 == 0 ? +1 : -1)));

  const double &mid_point_phi   = phi_p[interface_nb.mid_point_fine_node_idx];
  const bool no_past_mid_point  = signs_of_phi_are_different(phi_p[fine_node_idx_for_quad], mid_point_phi);
  const double &phi_this_side   = (no_past_mid_point ? phi_p[fine_node_idx_for_quad]  : mid_point_phi);
  const double &phi_across      = (no_past_mid_point ? mid_point_phi                  : phi_p[fine_node_idx_for_neighbor_quad]);
  const p4est_locidx_t& fine_idx_this_side  = (no_past_mid_point ? interface_nb.quad_fine_node_idx      : interface_nb.mid_point_fine_node_idx);
  const p4est_locidx_t& fine_idx_across     = (no_past_mid_point ? interface_nb.mid_point_fine_node_idx : interface_nb.neighbor_fine_node_idx);

  if(phi_xxyyzz_p != NULL)
    interface_nb.theta  = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, phi_xxyyzz_p[P4EST_DIM*fine_idx_this_side + face_dir/2], phi_xxyyzz_p[P4EST_DIM*fine_idx_across + face_dir/2], 0.5*solver.dxyz_min[face_dir/2]);
  else
    interface_nb.theta  = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, 0.5*solver.dxyz_min[face_dir/2], 0.5*solver.dxyz_min[face_dir/2]);
  interface_nb.theta = (phi_this_side > 0.0 ? 1.0 - interface_nb.theta : interface_nb.theta);
  interface_nb.theta = MAX(0.0, MIN(interface_nb.theta, 1.0));
  interface_nb.theta = 0.5*(interface_nb.theta + (no_past_mid_point ? 0.0 : 1.0));
  P4EST_ASSERT(0.0 <= interface_nb.theta && interface_nb.theta <= 1.0);

  ierr = VecRestoreArrayRead(solver.get_subrefined_phi(), &phi_p); CHKERRXX(ierr);
  if(phi_xxyyzz_p != NULL){
    ierr = VecRestoreArrayRead(solver.phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }

  which_interface_neighbor_t which_one = {quad_idx, face_dir};
  std::pair<map_of_interface_neighbors_t::iterator, bool> ret = interface_data.insert({which_one, interface_nb}); // add it to the map so that future access is read from memory;
  current_interface_data = ret.first;
  P4EST_ASSERT(ret.second && current_interface_point_is_set_for(quad_idx, face_dir));

  return;
}

const my_p4est_xgfm_cells_t::interface_neighbor
my_p4est_xgfm_cells_t::interface_manager_t::get_interface_neighbor(const p4est_locidx_t& quad_idx, const u_char& face_dir)
{
  if(!current_interface_point_is_set_for(quad_idx, face_dir))
    set_current_interface_point_for(quad_idx, face_dir);
  return current_interface_data->second;
}

void my_p4est_xgfm_cells_t::initialize_extension_on_cells()
{
  P4EST_ASSERT(levelset_has_been_set() && jumps_have_been_set());
  PetscErrorCode ierr;
  P4EST_ASSERT(extension_on_cells != NULL && VecIsSetForCells(extension_on_cells, p4est, ghost, 1));

  const double *phi_p, *solution_p, *jump_u_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  double *extension_on_cells_p;
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

#ifdef CASL_THROWS
  P4EST_ASSERT(interface_manager.is_map_consistent());
#endif
  ierr = VecRestoreArray(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  return;
}

void my_p4est_xgfm_cells_t::initialize_extension_on_cells_local(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                const double* const &solution_p, double* const &extension_on_cells_p) const
{
  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_q          = interp_subrefined_phi(xyz_quad);

  // build the educated initial guess : we extend u^{-}_{interface} if mu_m is larger, u^{+}_{interface} otherwise
  extension_on_cells_p[quad_idx] = solution_p[quad_idx];
  if(mu_minus_is_larger() && phi_q > 0.0)
    extension_on_cells_p[quad_idx] -= interp_subrefined_jump_u(xyz_quad);
  else if(!mu_minus_is_larger() && phi_q <= 0.0)
    extension_on_cells_p[quad_idx] += interp_subrefined_jump_u(xyz_quad);

  return;
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_operator_t::local_cell_TVD_extension_operator::add_interface_neighbor(const double* signed_normal, const double* dxyz_min, const interface_neighbor& neighbor, const u_char& face_dir_)
{
  if(too_close) // already found an interface neighbor that was too close --> you're done with this one
    return;
  if(neighbor.theta < EPS)
  {
    too_close = true;
    diag_entry = 0.0;
    forced_interface_value_face_dir = face_dir_;
    clear_extension_entries();
  }
  if(!too_close)
  {
    interface_entry* entry = new interface_entry;
    const double coeff = +signed_normal[face_dir_/2]*(face_dir_%2 == 0 ? +1.0 : -1.0)/(neighbor.theta*dxyz_min[face_dir_/2]);
    entry->face_dir = face_dir_; entry->coeff = coeff;
    extension_entries.push_back(entry);
    diag_entry -= coeff;
    dtau = MIN(dtau, neighbor.theta*dxyz_min[face_dir_/2]/(double) P4EST_DIM);
  }
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_operator_t::local_cell_TVD_extension_operator::
add_one_sided_derivative(const p4est_locidx_t& quad_idx, const double* signed_normal, const u_char& face_dir,
                         const linear_combination_of_dof_t& one_side_derivative)
{
  if(too_close) // already found an interface neighbor that was too close --> you're done with this one
    return;
  double discretization_distance = 0.0;
  for (size_t k = 0; k < one_side_derivative.size(); ++k) {
    const dof_weighted_term& derivative_term = one_side_derivative[k];
    if(derivative_term.dof_idx == quad_idx)
      diag_entry -= derivative_term.weight*signed_normal[face_dir/2];
    else
    {
      regular_quad_entry *off_diag = new regular_quad_entry;
      off_diag->loc_idx = derivative_term.dof_idx;
      off_diag->coeff   = -signed_normal[face_dir/2]*derivative_term.weight;
      extension_entries.push_back(off_diag);
    }
    discretization_distance = MAX(discretization_distance, fabs(1.0/one_side_derivative[k].weight));
  }
  dtau = MIN(dtau, discretization_distance/(double) P4EST_DIM);

  return;
}

double my_p4est_xgfm_cells_t::cell_TVD_extension_operator_t::
advance_one_pseudo_time_step(const p4est_locidx_t& quad_idx, const double* extension_on_cells_p, const double* solution_p, const double* jump_u_p, const double* jump_flux_p,
                             double& max_correction_in_band, const double& band_to_diag_ratio) const
{
  const local_cell_TVD_extension_operator& local_operator = my_local_operators[quad_idx];
  double increment;
  const bool in_positive_domain = (local_operator.phi_q > 0.0);
  const double& mu_this_side    = (in_positive_domain ? solver.mu_plus   : solver.mu_minus);
  const double& mu_across       = (in_positive_domain ? solver.mu_minus  : solver.mu_plus);
  const bool extending_positive_values = !solver.mu_minus_is_larger();
  if(local_operator.too_close)
  {
    P4EST_ASSERT(fabs(local_operator.diag_entry) < EPS && local_operator.forced_interface_value_face_dir < P4EST_FACES);
    increment = solver.interface_manager.interface_value(quad_idx, local_operator.forced_interface_value_face_dir, mu_this_side, mu_across, in_positive_domain, extending_positive_values,
                                                         solution_p, jump_u_p, jump_flux_p) - extension_on_cells_p[quad_idx];
  }
  else
  {
    increment = local_operator.diag_entry*extension_on_cells_p[quad_idx];
    for (size_t i = 0; i < local_operator.extension_entries.size(); ++i)
      increment += local_operator.extension_entries[i]->contribution_to_negative_normal_derivative(extension_on_cells_p, solver.interface_manager, mu_this_side, mu_across, in_positive_domain, extending_positive_values,
                                                                                                   quad_idx, solution_p, jump_u_p, jump_flux_p);
    increment *= local_operator.dtau;
  }
  if(fabs(local_operator.phi_q) < band_to_diag_ratio*solver.diag_min())
    max_correction_in_band = MAX(fabs(increment), max_correction_in_band);

  return extension_on_cells_p[quad_idx] + increment;
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_operator_t::build_local_operator_for(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx)
{
  local_cell_TVD_extension_operator& local_operator = my_local_operators[quad_idx];
  // we'll build the cell-sampled interface-value extension operators, here!
  // the extension procedure, can be formally written as
  // pseudo_time_increment = dtau*(A*cell_values + B*interface_values)
  // where A and B are appropriate (sparse) matrices: here below, we build the entries of A and B.
  local_operator.diag_entry = 0.0;
  const p4est_tree_t* tree = p4est_tree_array_index(solver.p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const double *phi_p, *normals_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(solver.phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.normals, &normals_p); CHKERRXX(ierr);

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, solver.p4est, solver.ghost, xyz_quad);
  local_operator.phi_q = solver.interp_subrefined_phi(xyz_quad);
  double grad_phi[P4EST_DIM]; solver.interp_subrefined_normals(xyz_quad, grad_phi);
  const double mag_grad_phi = sqrt(SUMD(SQR(grad_phi[0]), SQR(grad_phi[1]), SQR(grad_phi[2])));
  const double signed_normal[P4EST_DIM] = {DIM((mag_grad_phi > EPS ? (local_operator.phi_q <= 0.0 ? -1.0 : +1.0)*grad_phi[0]/mag_grad_phi : 0.0),
                                   (mag_grad_phi > EPS ? (local_operator.phi_q <= 0.0 ? -1.0 : +1.0)*grad_phi[1]/mag_grad_phi : 0.0),
                                   (mag_grad_phi > EPS ? (local_operator.phi_q <= 0.0 ? -1.0 : +1.0)*grad_phi[2]/mag_grad_phi : 0.0))};
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    const unsigned char dir = 2*dim + (signed_normal[dim] > 0.0 ? 0 : 1);
    if(is_quad_Wall(solver.p4est, tree_idx, quad, dir))
      continue; // homogeneous Neumann boundary condition

    set_of_neighboring_quadrants neighbor_cells;
    bool derivative_is_one_side;
    linear_combination_of_dof_t stable_projection_derivative_operator = solver.stable_projection_derivative_operator_at_face(quad_idx, tree_idx, dir, neighbor_cells, derivative_is_one_side);

    if(derivative_is_one_side)
      local_operator.add_one_sided_derivative(quad_idx, signed_normal, dir, stable_projection_derivative_operator);
    else
      local_operator.add_interface_neighbor(signed_normal, solver.dxyz_min, solver.interface_manager.get_interface_neighbor(quad_idx, dir), dir);
  }

  ierr = VecRestoreArrayRead(solver.phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.normals, &normals_p); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_of_interface_values(const double& threshold, const uint& niter_max)
{
  P4EST_ASSERT(levelset_has_been_set() && threshold > EPS && niter_max > 0);
  const double *solution_p, *jump_u_p, *jump_flux_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  Vec extension_on_cells_np1; // (next pseudo-time step values)
  ierr = VecCreateGhostCells(p4est, ghost, &extension_on_cells_np1); CHKERRXX(ierr);

  const double band_to_diag_ratio = 3.0;
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
      if(!cell_TVD_extension_operator.is_set)
        cell_TVD_extension_operator.build_local_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_layer_quadrant(k));
      extension_on_cells_np1_p[quad_idx] = cell_TVD_extension_operator.advance_one_pseudo_time_step(quad_idx, extension_on_cells_n_p, solution_p, jump_u_p, jump_flux_p,
                                                                                                    max_increment_in_band, band_to_diag_ratio);
    }
    ierr = VecGhostUpdateBegin(extension_on_cells_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < cell_ngbd->get_hierarchy()->get_inner_size(); ++k) {
      const p4est_locidx_t quad_idx = cell_ngbd->get_hierarchy()->get_local_index_of_inner_quadrant(k);
      if(!cell_TVD_extension_operator.is_set)
        cell_TVD_extension_operator.build_local_operator_for(quad_idx, cell_ngbd->get_hierarchy()->get_tree_index_of_inner_quadrant(k));
      extension_on_cells_np1_p[quad_idx] = cell_TVD_extension_operator.advance_one_pseudo_time_step(quad_idx, extension_on_cells_n_p, solution_p, jump_u_p, jump_flux_p,
                                                                                                    max_increment_in_band, band_to_diag_ratio);
    }
    if(!cell_TVD_extension_operator.is_set)
      cell_TVD_extension_operator.is_set = true;

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

void my_p4est_xgfm_cells_t::interpolate_cell_extension_to_nodes()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_interpolate_cell_extension_to_nodes, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef WITH_SUBREFINEMENT
  const my_p4est_node_neighbors_t *ngbd_n = fine_node_ngbd;
#else
  const my_p4est_node_neighbors_t *ngbd_n = node_ngbd;
#endif

  const double *extension_on_cells_p;
  double *extension_on_nodes_p;
  ierr = VecGetArrayRead(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = VecGetArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);

  for (size_t nnn = 0; nnn < ngbd_n->get_layer_size(); ++nnn)
  {
    const p4est_locidx_t node_idx = ngbd_n->get_layer_node(nnn);
    extension_on_nodes_p[node_idx] = interpolate_cell_field_at_local_node(node_idx, extension_on_cells_p);
  }
  ierr = VecGhostUpdateBegin(extension_on_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t nnn = 0; nnn < ngbd_n->get_local_size(); ++nnn)
  {
    const p4est_locidx_t node_idx = ngbd_n->get_local_node(nnn);
    extension_on_nodes_p[node_idx] = interpolate_cell_field_at_local_node(node_idx, extension_on_cells_p);
  }
  ierr = VecGhostUpdateEnd(extension_on_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(!local_interpolators_are_set) // if they were not already set, now they are!
    local_interpolators_are_set = true;

  ierr = VecRestoreArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(extension_on_cells, &extension_on_cells_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_interpolate_cell_extension_to_nodes, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

double my_p4est_xgfm_cells_t::interpolate_cell_field_at_local_node(const p4est_locidx_t &node_idx, const double *cell_field_p)
{
  linear_combination_of_dof_t& cell_interpolator = local_interpolators[node_idx];

  if(local_interpolators_are_set) // no need for the hardwork if we have already done and memorized it
    return cell_interpolator(cell_field_p);

#ifdef WITH_SUBREFINEMENT
  if(((splitting_criteria_t*) fine_p4est->user_pointer)->max_lvl > ((splitting_criteria_t*) p4est->user_pointer)->max_lvl + 1)
    throw std::runtime_error("my_p4est_xgfm_cells_t::interpolate_cell_field_at_fine_node() is not implemented yet for a level difference larger than 1");
#endif

  cell_interpolator.clear();
  double to_return;
  double xyz_node[P4EST_DIM];
#ifdef WITH_SUBREFINEMENT
  node_xyz_fr_n(node_idx, fine_p4est, fine_nodes, xyz_node);
  const p4est_indep_t* ni = (p4est_indep_t*) sc_const_array_index(&fine_nodes->indep_nodes, node_idx);
  p4est_locidx_t coarse_node_idx = -1;
  if(index_of_node((const p4est_quadrant_t*) ni, nodes, coarse_node_idx))
#else
  node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
  const p4est_indep_t* ni = (p4est_indep_t*) sc_const_array_index(&nodes->indep_nodes, node_idx);
  const p4est_locidx_t coarse_node_idx = node_idx;
#endif
  {
    P4EST_ASSERT(coarse_node_idx >= 0);
    set_of_neighboring_quadrants nearby_cell_neighbors;
    const p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = node_ngbd->gather_neighbor_cells_of_node(nearby_cell_neighbors, cell_ngbd, coarse_node_idx);
    const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
    to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, nearby_cell_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
    // note : if locally uniform the result of the above is the expected arithmetic average of neighbor cells
  }
#ifdef WITH_SUBREFINEMENT
  else
  {
    // find smallest (coarse) quad containing point
    p4est_quadrant_t coarse_quad; std::vector<p4est_quadrant_t> remote_matches; remote_matches.resize(0);
    int rank = cell_ngbd->hierarchy->find_smallest_quadrant_containing_point(xyz_node, coarse_quad, remote_matches, false, true); P4EST_ASSERT(rank != -1); (void) rank;
    double quad_xyz[P4EST_DIM]; quad_xyz_fr_q(coarse_quad.p.piggy3.local_num, coarse_quad.p.piggy3.which_tree, p4est, ghost, quad_xyz);
    if(ORD(periodicity[0], periodicity[1], periodicity[2]))
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
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
      P4EST_ASSERT(SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])) < P4EST_DIM); // because if not, it is a vertex on the coarse grid and it should have been found...
      set_of_neighboring_quadrants quads_sharing_fine_node;
      quads_sharing_fine_node.insert(coarse_quad);
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
                quads_sharing_fine_node.insert(*it);
            }
          }
      P4EST_ASSERT(!fine_uniform_neighborhood || quads_sharing_fine_node.size() == (size_t) 2*SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])));
      fine_uniform_neighborhood = fine_uniform_neighborhood && quads_sharing_fine_node.size() == (size_t) 2*SUMD(abs(local_orientation_in_coarse_cell[0]), abs(local_orientation_in_coarse_cell[1]), abs(local_orientation_in_coarse_cell[2])); // should be useless but let's play safe in release...
      if(fine_uniform_neighborhood) // avoid lsqr in that case, keep it simple
      {
        to_return = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = quads_sharing_fine_node.begin(); it != quads_sharing_fine_node.end(); ++it)
        {
          to_return += cell_field_p[it->p.piggy3.local_num]/(double) quads_sharing_fine_node.size();
          cell_interpolator.add_term(it->p.piggy3.local_num, 1.0/(double) quads_sharing_fine_node.size());
        }
        P4EST_ASSERT(cell_interpolator.size() == quads_sharing_fine_node.size());
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
        for (set_of_neighboring_quadrants::const_iterator it = quads_sharing_fine_node.begin(); it != quads_sharing_fine_node.end(); ++it)
          logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, cell_ngbd->gather_neighbor_cells_of_cell(*it, lsqr_neighbors, false, no_search));
        const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
        to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, lsqr_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
      }
    }
  }
#endif

#ifdef DEBUG
  const double value_check = cell_interpolator(cell_field_p);
  if(ISNAN(value_check))
    std::cout << "my_p4est_xgfm_cells_t::interpolate_cell_field_at_fine_node(): tracking a NAN : the local_interpolator returns a NAN for node_idx = " << node_idx
             #ifdef WITH_SUBREFINEMENT
              << " (on the interface-capturing grid)"
             #endif
              << " on proc " << p4est->mpirank << std::endl;
  P4EST_ASSERT(fabs(value_check - to_return) < MAX(EPS, 0.000001*MAX(fabs(value_check), fabs(to_return))));
#endif

  return to_return;
}

void my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities_local(const p4est_locidx_t& f_idx, const u_char& dim, const my_p4est_faces_t* faces, const double* solution_p,
                                                                                        const double* jump_u_p, const double* jump_flux_p,
                                                                                        double* flux_dir_p, const double* vstar_dir_p, double* vnp1_plus_dir_p, double* vnp1_minus_dir_p)
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(f_idx, dim, quad_idx, tree_idx);
  u_char face_dir = (faces->q2f(quad_idx, 2*dim) == f_idx ? 2*dim : 2*dim + 1);
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_quad_size = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_quad_size, tree_dimensions[1]*logical_quad_size, tree_dimensions[2]*logical_quad_size)};

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])}; xyz_face[dim] += (face_dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[dim];
  const double phi_q        = interp_subrefined_phi(xyz_quad);
  const double phi_face     = interp_subrefined_phi(xyz_face);
  const double mu_face      = (phi_face > 0.0 ? mu_plus : mu_minus);

  if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
  {
    P4EST_ASSERT(f_idx != NO_VELOCITY);
#ifdef CASL_THROWS
    if(signs_of_phi_are_different(phi_q, phi_face))
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities_local(): a wall-cell is crossed by the interface...");
#endif
    switch(bc->wallType(xyz_face))
    {
    case DIRICHLET:
      flux_dir_p[f_idx] = (face_dir%2 == 1 ? +1.0 : -1.0)*(2.0*mu_face*(bc->wallValue(xyz_face) - solution_p[quad_idx])/cell_dxyz[dim]);
      break;
    case NEUMANN:
      flux_dir_p[f_idx] = (face_dir%2 == 1 ? +1.0 : -1.0)*mu_face*bc->wallValue(xyz_face);
      break;
    default:
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities_local(): unknown boundary condition on a wall.");
    }
  }
  else
  {
    set_of_neighboring_quadrants direct_neighbors;
    bool one_sided;
    linear_combination_of_dof_t stable_projection_derivative = stable_projection_derivative_operator_at_face(quad_idx, tree_idx, face_dir, direct_neighbors, one_sided);

    if(one_sided)
    {
      if(signs_of_phi_are_different(phi_q, phi_face)) // can be under-resolved :  +-+ or -+-
      {
        P4EST_ASSERT(signs_of_phi_are_different(phi_q, phi_face));
        const p4est_locidx_t fine_node_idx_for_face = get_fine_node_idx_of_face_in_quad(fine_p4est, fine_nodes, *quad, tree_idx, face_dir);
        flux_dir_p[f_idx] = (phi_face > 0.0 ? mu_minus : mu_plus)*stable_projection_derivative(solution_p) + (phi_face > 0.0 ? +1.0 : -1.0)*jump_flux_p[P4EST_DIM*fine_node_idx_for_face + face_dir];
      }
      else
        flux_dir_p[f_idx]              = mu_face*stable_projection_derivative(solution_p);
    }
    else
    {
      const double &mu_this_side    = (phi_q <= 0.0 ? mu_minus : mu_plus);
      const double &mu_across       = (phi_q <= 0.0 ? mu_plus : mu_minus);
      const bool in_positive_domain = (phi_q > 0.0);
      flux_dir_p[f_idx] = interface_manager.GFM_flux_at_center_face(quad_idx, face_dir, mu_this_side, mu_across, in_positive_domain, !signs_of_phi_are_different(phi_q, phi_face), solution_p[quad_idx], solution_p[direct_neighbors.begin()->p.piggy3.local_num], jump_u_p, jump_flux_p);
    }
  }
  // subtract from the star velocities if they were provided
  if(vstar_dir_p != NULL && vnp1_plus_dir_p != NULL && vnp1_minus_dir_p != NULL)
  {
    double* &vnp1_this_side_p  = (phi_face > 0.0 ? vnp1_plus_dir_p   : vnp1_minus_dir_p);
    double* &vnp1_across_p     = (phi_face > 0.0 ? vnp1_minus_dir_p  : vnp1_plus_dir_p);
    vnp1_this_side_p[f_idx] = vstar_dir_p[f_idx] - flux_dir_p[f_idx];
    vnp1_across_p[f_idx]    = DBL_MAX;
  }

  return;
}

void my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec flux[], my_p4est_faces_t *faces, Vec vstar[], Vec vnp1_minus[], Vec vnp1_plus[])
{
  P4EST_ASSERT(faces->get_p4est() == p4est); // the faces must be built from the same computational grid
  P4EST_ASSERT(jump_flux != NULL); // the flux vectors cannot be calculated if the jumps in flux components have been returned to the user
  P4EST_ASSERT((vstar == NULL && vnp1_minus == NULL && vnp1_plus == NULL) ||
               (VecsAreSetForFaces(vstar, faces, 1) && VecsAreSetForFaces(vnp1_minus, faces, 1) && VecsAreSetForFaces(vnp1_plus, faces, 1))); // the face-sampled velocities vstart and vnp1 vectors vectors must either be all defined or be all NULL.
  P4EST_ASSERT(VecsAreSetForFaces(flux, faces, 1));

  if(solution == NULL)
    solve();
  double *flux_p[P4EST_DIM];
  const double *phi_p, *solution_p, *jump_u_p, *jump_flux_p;
  const double *phi_xxyyzz_p = NULL;
  const bool velocities_provided = (vstar != NULL && vnp1_plus != NULL && vnp1_minus != NULL);
  const double *vstar_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *vnp1_plus_p[P4EST_DIM]    = {DIM(NULL, NULL, NULL)};
  double *vnp1_minus_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    if(velocities_provided)
    {
      P4EST_ASSERT(vstar[dim] != NULL && vnp1_plus[dim] != NULL && vnp1_minus[dim] != NULL);
      ierr = VecGetArrayRead(vstar[dim], &vstar_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
  }

  // layer faces, first
  for (u_char dim = 0; dim < P4EST_DIM; ++dim){
    for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
      get_flux_components_and_subtract_them_from_velocities_local(faces->get_layer_face(dim, k), dim, faces, solution_p,
                                                                  jump_u_p, jump_flux_p,
                                                                  flux_p[dim], vstar_p[dim], vnp1_plus_p[dim], vnp1_minus_p[dim]);
    // start the ghost updates
    ierr = VecGhostUpdateBegin(flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(vstar != NULL && vnp1_plus != NULL && vnp1_minus != NULL){
      ierr = VecGhostUpdateBegin(vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  // inner faces, second
  for (u_char dim = 0; dim < P4EST_DIM; ++dim){
    for (size_t k = 0; k < faces->get_local_size(dim); ++k)
      get_flux_components_and_subtract_them_from_velocities_local(faces->get_local_face(dim, k), dim, faces, solution_p,
                                                                  jump_u_p, jump_flux_p,
                                                                  flux_p[dim], vstar_p[dim], vnp1_plus_p[dim], vnp1_minus_p[dim]);
    // finish the ghost updates
    ierr = VecGhostUpdateEnd(flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(vstar != NULL && vnp1_plus != NULL && vnp1_minus != NULL){
      ierr = VecGhostUpdateEnd(vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz_p != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    if(velocities_provided)
    {
      ierr = VecRestoreArrayRead(vstar[dim], &vstar_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
  }

  return;
}

#ifdef DEBUG
int my_p4est_xgfm_cells_t::interface_manager_t::is_map_consistent()
{
  int mpiret;

  std::vector<int> senders(solver.p4est->mpisize, 0);
  int num_expected_replies = 0;

  const double *solution_p, *jump_u_p, *jump_flux_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(solver.solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);

  int it_is_alright = true;
  std::map<int, std::vector<which_interface_neighbor_t> > map_of_query_interface_neighbors; map_of_query_interface_neighbors.clear();
  std::map<int, std::vector<which_interface_neighbor_t> > map_of_mirrors; map_of_mirrors.clear();
  for (map_of_interface_neighbors_t::const_iterator it = interface_data.begin(); it != interface_data.end(); ++it)
  {
    const which_interface_neighbor_t& this_one        = it->first;
    const interface_neighbor& this_interface_neighbor = it->second;
    // the neighbor is a local quad
    if(this_interface_neighbor.neighbor_quad_idx < solver.p4est->local_num_quadrants)
    {
      which_interface_neighbor_t other_one                = {this_interface_neighbor.neighbor_quad_idx, (u_char)(this_one.face_dir + (this_one.face_dir%2 == 0 ? +1 : -1))};
      const interface_neighbor& other_interface_neighbor  = interface_data.at(other_one);
      it_is_alright = it_is_alright && this_interface_neighbor.is_consistent_with(other_interface_neighbor);

      if(!this_interface_neighbor.is_consistent_with(other_interface_neighbor))
        std::cerr << "Inconsistency found for quad " << this_one.loc_idx << " on proc " << solver.p4est->mpirank << " which has quad " << other_one.loc_idx << " as a neighbor across the interface, on proc " << solver.p4est->mpirank << std::endl;
    }
    else
    {
      const p4est_quadrant_t* ghost_nb_quad = p4est_const_quadrant_array_index(&solver.ghost->ghosts, this_interface_neighbor.neighbor_quad_idx - solver.p4est->local_num_quadrants);
      int rank_owner = p4est_comm_find_owner(solver.p4est, ghost_nb_quad->p.piggy3.which_tree, ghost_nb_quad, solver.p4est->mpirank);
      P4EST_ASSERT(rank_owner != solver.p4est->mpirank);
      num_expected_replies += (senders[rank_owner] == 1 ? 0 : 1);
      senders[rank_owner] = 1;
      which_interface_neighbor_t other_one;

      other_one.loc_idx = ghost_nb_quad->p.piggy3.local_num;
      other_one.face_dir = this_one.face_dir + (this_one.face_dir%2 == 0 ? 1 : -1);
      map_of_query_interface_neighbors[rank_owner].push_back(other_one);
      map_of_mirrors[rank_owner].push_back(this_one);
    }
  }

  std::vector<int> recvcount(solver.p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, solver.p4est->mpicomm); SC_CHECK_MPI(mpiret);

  std::vector<MPI_Request> mpi_query_requests; mpi_query_requests.resize(0);
  std::vector<MPI_Request> mpi_reply_requests; mpi_reply_requests.resize(0);

  // send the requests...
  for (std::map<int, std::vector<which_interface_neighbor_t> >::const_iterator it = map_of_query_interface_neighbors.begin();
       it != map_of_query_interface_neighbors.end(); ++it) {
    if (it->first == solver.p4est->mpirank)
      continue;

    int rank = it->first;
    P4EST_ASSERT(senders[rank] == 1);

    MPI_Request req;
    mpiret = MPI_Isend((void*)&map_of_query_interface_neighbors[rank][0], sizeof(which_interface_neighbor_t)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, solver.p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    mpi_query_requests.push_back(req);
  }

  std::map<int, std::vector<interface_neighbor> > map_of_responses; map_of_responses.clear();

  MPI_Status status;
  bool done = (num_expected_replies == 0 && num_remaining_queries == 0);
  while (!done) {
    if(num_remaining_queries > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 15351, solver.p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(which_interface_neighbor_t) == 0);
        std::vector<which_interface_neighbor_t> queries(byte_count/sizeof(which_interface_neighbor_t));

        mpiret = MPI_Recv((void*) &queries[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 15351, solver.p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

        std::vector<interface_neighbor>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(which_interface_neighbor_t));

        for (size_t kk = 0; kk < response.size(); ++kk)
          response[kk] = interface_data.at(queries[kk]);

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*)&response[0], (response.size())*sizeof(interface_neighbor), MPI_BYTE, status.MPI_SOURCE, 42624, solver.p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
        mpi_reply_requests.push_back(req);
        num_remaining_queries--;
      }
    }

    if(num_expected_replies > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 42624, solver.p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {

        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(interface_neighbor) == 0);
        std::vector<interface_neighbor> reply_buffer (byte_count / sizeof(interface_neighbor));

        mpiret = MPI_Recv((void*)&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 42624, solver.p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        for (size_t kk = 0; kk < reply_buffer.size(); ++kk) {
          which_interface_neighbor_t mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          which_interface_neighbor_t queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];
          it_is_alright = it_is_alright && interface_data.at(mirror).is_consistent_with(reply_buffer[kk]);

          if(!interface_data.at(mirror).is_consistent_with(reply_buffer[kk]))
            std::cerr << "Inconsistency found for quad " << mirror.loc_idx << " on proc " << solver.p4est->mpirank << " which has quad " << queried.loc_idx << " as a neighbor across the interface, on proc " << status.MPI_SOURCE << std::endl;
        }

        num_expected_replies--;
      }
    }
    done = (num_expected_replies == 0 && num_remaining_queries == 0);
  }

  mpiret = MPI_Waitall(mpi_query_requests.size(), &mpi_query_requests[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(mpi_reply_requests.size(), &mpi_reply_requests[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpi_query_requests.clear();
  mpi_reply_requests.clear();

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &it_is_alright, 1, MPI_INT, MPI_LAND, solver.p4est->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = VecRestoreArrayRead(solver.solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver.jump_flux, &jump_flux_p); CHKERRXX(ierr);

  return it_is_alright;
}
#endif

