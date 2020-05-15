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
extern PetscLogEvent log_my_p4est_xgfm_cells_extend_field;
extern PetscLogEvent log_my_p4est_xgfm_cells_interpolate_cell_field_to_nodes;
extern PetscLogEvent log_my_p4est_xgfm_cells_get_corrected_rhs;
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
    #ifdef SUBREFINED
    fine_p4est(fine_ngbd_n->get_p4est()), fine_nodes(fine_ngbd_n->get_nodes()), fine_ghost(fine_ngbd_n->get_ghost()), fine_node_ngbd(fine_ngbd_n),
    #endif
    activate_xGFM(activate_xGFM_)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  mu_m = mu_p = -1.0;
  add_diag_m = add_diag_p = 0.0;
  user_rhs = rhs = solution = extension_on_cells = extension_on_nodes = jump_flux = NULL;
#ifdef SUBREFINED
  phi = normals = phi_xxyyzz = NULL;
  jump_u = jump_normal_flux_u = NULL;
#else
  interp_phi = interp_normals = NULL;
  interp_jump_u = interp_jump_normal_flux_u = NULL;
#endif

  A = NULL;
  A_null_space = NULL;
  bc = NULL;
  matrix_is_preallocated = matrix_is_set = rhs_is_set = false;

  splitting_criteria_t *data      = (splitting_criteria_t*) p4est->user_pointer;
#ifdef SUBREFINED
  splitting_criteria_t *data_fine = (splitting_criteria_t*) fine_p4est->user_pointer;
#endif

  // Domain and grid parameters
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
  {
    dxyz_min[dir]       = tree_dimensions[dir] / (double) (1 << data->max_lvl);
#ifdef SUBREFINED
    dxyz_min_fine[dir]  = tree_dimensions[dir] / (double) (1 << data_fine->max_lvl);
#endif
  }

  map_of_interface_neighbors.clear();
  extension_entries.clear();
  local_interpolator.clear();
  extension_entries.resize(p4est->local_num_quadrants);
#ifdef SUBREFINED
  local_interpolator.resize(fine_nodes->num_owned_indeps);
#else
  local_interpolator.resize(nodes->num_owned_indeps);
#endif
  extension_entries_are_set       = false;
  local_interpolators_are_set     = false;
  interface_values_are_set        = false;
  solution_is_set                 = false;
  use_initial_guess               = false;
}

my_p4est_xgfm_cells_t::~my_p4est_xgfm_cells_t()
{
  if (A               != NULL)    { ierr = MatDestroy(A);                       CHKERRXX(ierr); }
  if (A_null_space    != NULL)    { ierr = MatNullSpaceDestroy (A_null_space);  CHKERRXX(ierr); }
  if (ksp             != NULL)    { ierr = KSPDestroy(ksp);                     CHKERRXX(ierr); }
  if (extension_on_cells != NULL) { ierr = VecDestroy(extension_on_cells);      CHKERRXX(ierr); }
  if (extension_on_nodes != NULL) { ierr = VecDestroy(extension_on_nodes);      CHKERRXX(ierr); }
  if (rhs   != NULL)              { ierr = VecDestroy(rhs);                     CHKERRXX(ierr); }
  if (solution        != NULL)    { ierr = VecDestroy(solution);                CHKERRXX(ierr); }
  if (jump_flux       != NULL)    { ierr = VecDestroy(jump_flux);               CHKERRXX(ierr); }
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
  matrix_is_set = rhs_is_set = false;
}

void my_p4est_xgfm_cells_t::set_normals(Vec node_sampled_normals)
{
  P4EST_ASSERT(node_sampled_normals != NULL && VecIsSetForNodes(node_sampled_normals, fine_nodes, fine_p4est->mpicomm, P4EST_DIM));
  normals = node_sampled_normals;
  rhs_is_set = false;
}

void my_p4est_xgfm_cells_t::set_jumps(Vec node_sampled_jump_u, Vec node_sampled_jump_normal_flux)
{
  P4EST_ASSERT(node_sampled_jump_u != NULL);
  P4EST_ASSERT(node_sampled_jump_normal_flux != NULL);
  P4EST_ASSERT(VecIsSetForNodes(node_sampled_jump_u,            fine_nodes, fine_p4est->mpicomm, 1));
  P4EST_ASSERT(VecIsSetForNodes(node_sampled_jump_normal_flux,  fine_nodes, fine_p4est->mpicomm, 1));
  jump_u              = node_sampled_jump_u;
  jump_normal_flux_u  = node_sampled_jump_normal_flux;
  rhs_is_set = false;
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components()
{
  if(!normals_have_been_set() || !diffusion_coefficients_have_been_set() || !jumps_have_been_set())
    throw std::runtime_error("my_p4est_xgfm_cells_t::compute_jump_mu_grad_u() : the jump in flux components can't be calculated if the normals, diffusion coefficients or required jumps in solution have not been set beforehand");

  if(jump_flux == NULL){
    ierr = VecCreateGhostNodesBlock(fine_p4est, fine_nodes, P4EST_DIM, &jump_flux); CHKERRXX(ierr); }

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
}

void my_p4est_xgfm_cells_t::compute_jumps_in_flux_components_for_node(const p4est_locidx_t& node_idx, double *jump_flux_p,
                                                                      const double *jump_normal_flux_p, const double *normals_p, const double *jump_u_p, const double *extension_on_nodes_p)
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
}

void my_p4est_xgfm_cells_t::preallocate_matrix()
{
  if(matrix_is_preallocated)
    return;

  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  if (A != NULL){
    ierr = MatDestroy(A); CHKERRXX(ierr);}

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
      for(u_char dir = 0; dir < P4EST_FACES; ++dir)
      {
        set_of_neighboring_quadrants direct_neighbors;
        cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, dir);

        for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
          neighbor_quads_involved.insert(*it);

        if(direct_neighbors.size() == 1 && direct_neighbors.begin()->level < quad->level)
          cell_ngbd->find_neighbor_cells_of_cell(neighbor_quads_involved, direct_neighbors.begin()->p.piggy3.local_num, direct_neighbors.begin()->p.piggy3.which_tree, dir%2 == 0 ? dir + 1 : dir - 1);
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
  matrix_is_preallocated = true;
  return;
}

void my_p4est_xgfm_cells_t::build_discretization_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                          const double *phi_p, const double* phi_xxyyzz_p, const my_p4est_interpolation_nodes_t& interp_phi,
                                                          const double *user_rhs_p, const double *jump_u_p, const double *jump_flux_p,
                                                          int &nullspace_contains_constant_vector, double* rhs_p)
{
  const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  p4est_locidx_t fine_node_idx_for_quad = -1, fine_node_idx_for_neighbor_quad = -1, fine_node_idx_for_face = -1;

  const double logical_size_quad = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double cell_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

  const PetscInt quad_gloidx = compute_global_index(quad_idx);

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_quad = (quad_center_is_fine_node(*quad, tree_idx, fine_node_idx_for_quad) ? phi_p[fine_node_idx_for_quad] : interp_phi(xyz_quad));
  const double &mu_q = (phi_quad > 0.0 ? mu_p : mu_m);

  /* First add the diagonal term */
  const double& add_diag = (phi_quad > 0.0 ? add_diag_p : add_diag_m);
  if(!matrix_is_set && fabs(add_diag) > EPS)
  {
    ierr = MatSetValue(A, quad_gloidx, quad_gloidx, cell_volume*add_diag, ADD_VALUES); CHKERRXX(ierr);
    nullspace_contains_constant_vector = 0;
  }
  if(!rhs_is_set)
    rhs_p[quad_idx] = user_rhs_p[quad_idx]*cell_volume;

  for(u_char dir = 0; dir < P4EST_FACES; ++dir)
  {
    const double face_area = cell_volume/cell_dxyz[dir/2];

    /* first check if the cell is a wall
     * We will assume that walls are not crossed by the interface, in a first attempt! */
    if(is_quad_Wall(p4est, tree_idx, quad, dir))
    {
      double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])};
      xyz_face[dir/2] += (dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[dir/2];
#ifdef CASL_THROWS
      const double phi_face = (face_in_quad_is_fine_node(*quad, tree_idx, dir, fine_node_idx_for_face) ? phi_p[fine_node_idx_for_face] : interp_phi(xyz_face));
      if(signs_of_phi_are_different(phi_quad, phi_face))
        throw std::invalid_argument("my_p4est_xgfm_cells_t::build_discretization_for_quad() : a wall-cell is crossed by the interface...");
#endif
      switch(bc->wallType(xyz_face))
      {
      case DIRICHLET:
      {
        if(!matrix_is_set)
        {
          nullspace_contains_constant_vector = 0;
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu_q*face_area/cell_dxyz[dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
          rhs_p[quad_idx]  += 2.0*mu_q*face_area*bc->wallValue(xyz_face)/cell_dxyz[dir/2];
      }
        break;
      case NEUMANN:
      {
        if(!rhs_is_set)
          rhs_p[quad_idx]  += mu_q*face_area*bc->wallValue(xyz_face);
      }
        break;
      default:
        throw std::invalid_argument("my_p4est_xgfm_cells_t::build_discretization_for_quad() : unknown boundary condition on a wall.");
      }
    }

    /* now get the neighbors */
    set_of_neighboring_quadrants direct_neighbors;
    cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, dir);

    if(direct_neighbors.size() == 1)
    {
      const p4est_quadrant_t &neighbor_quad = *direct_neighbors.begin();
      double xyz_neighbor_quad[P4EST_DIM]; quad_xyz_fr_q(neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, p4est, ghost, xyz_neighbor_quad);
      const double phi_neighbor_quad = (quad_center_is_fine_node(neighbor_quad, neighbor_quad.p.piggy3.which_tree, fine_node_idx_for_neighbor_quad) ? phi_p[fine_node_idx_for_neighbor_quad] : interp_phi(xyz_neighbor_quad));

      /* If interface across the two cells.
       * We assume that the interface is tesselated with uniform finest grid level */
      if(signs_of_phi_are_different(phi_quad, phi_neighbor_quad))
      {
        if(!(fine_node_idx_for_quad >= 0 && fine_node_idx_for_neighbor_quad >= 0))
          std::cout << "fine_node_idx_for_quad = " << fine_node_idx_for_quad  << ", fine_node_idx_for_neighbor_quad" << fine_node_idx_for_neighbor_quad << std::endl;
        P4EST_ASSERT(fine_node_idx_for_quad >= 0 && fine_node_idx_for_neighbor_quad >= 0);
        P4EST_ASSERT(quad->level == neighbor_quad.level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
        interface_neighbor jump_data = get_interface_neighbor(quad_idx, dir, neighbor_quad.p.piggy3.local_num, fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad, phi_p, phi_xxyyzz_p);
        const double& mu_across = (phi_neighbor_quad  > 0.0 ? mu_p : mu_m);
        const double mu_jump = mu_m*mu_p/((1.0 - jump_data.theta)*mu_q + jump_data.theta*mu_across);

        if(!matrix_is_set)
        {
          ierr = MatSetValue(A, quad_gloidx, quad_gloidx,                                             mu_jump * face_area/dxyz_min[dir/2], ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(neighbor_quad.p.piggy3.local_num), -mu_jump * face_area/dxyz_min[dir/2], ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set)
        {
          P4EST_ASSERT(0.0 <= jump_data.theta && jump_data.theta <= 1.0);
          const bool past_mid_point = jump_data.theta >= 0.5;
          const double theta_between_fine_nodes     = 2.0*jump_data.theta - (past_mid_point ? 1.0 : 0.0);
          const p4est_locidx_t &fine_node_this_side = (past_mid_point ? jump_data.mid_point_fine_node_idx  : jump_data.quad_fine_node_idx);
          const p4est_locidx_t &fine_node_across    = (past_mid_point ? jump_data.nb_fine_node_idx         : jump_data.mid_point_fine_node_idx);
          const double jump_sol_across        = theta_between_fine_nodes*jump_u_p[fine_node_across]                      + (1.0 - theta_between_fine_nodes)*jump_u_p[fine_node_this_side];
          const double jump_flux_comp_across  = theta_between_fine_nodes*jump_flux_p[P4EST_DIM*fine_node_across + dir/2] + (1.0 - theta_between_fine_nodes)*jump_flux_p[P4EST_DIM*fine_node_this_side + dir/2];
          P4EST_ASSERT(fabs(jump_data.phi_q - phi_quad) < MAX(EPS, EPS*MAX(fabs(jump_data.phi_q), fabs(phi_quad))));
          rhs_p[quad_idx] += mu_jump*(jump_data.phi_q > 0.0 ? +1.0 : -1.0)*face_area*
              ((dir%2 == 0 ? -1.0 : +1.0)*jump_flux_comp_across*(1 - jump_data.theta)/mu_across /*jump_data.mu_other_side*/ + jump_sol_across/dxyz_min[dir/2]);
        }
      }
      /* no interface - regular discretization */
      else
      {
        const double surface_direct_neighbor = pow((double)P4EST_QUADRANT_LEN(neighbor_quad.level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1); // logical surface

        set_of_neighboring_quadrants sharing_quads;
        cell_ngbd->find_neighbor_cells_of_cell(sharing_quads, neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, dir%2 == 0 ? dir + 1 : dir - 1);
        P4EST_ASSERT(sharing_quads.size() >= 1);

        double discretization_distance = 0.0;
#ifdef DEBUG
        double split_face_check = 0.0; bool quad_is_among_sharers = false;
#endif
        for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
        {
          const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_direct_neighbor;
          discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(neighbor_quad.level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
#ifdef DEBUG
          split_face_check += surface_ratio; quad_is_among_sharers = quad_is_among_sharers || (it->p.piggy3.local_num == quad_idx);
#endif
        }
        P4EST_ASSERT(quad_is_among_sharers && fabs(split_face_check - 1.0) < EPS);
        discretization_distance *= tree_dimensions[dir/2];

        for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
        {
          const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_direct_neighbor;
          if(!matrix_is_set){
            ierr = MatSetValue(A, quad_gloidx, compute_global_index(it->p.piggy3.local_num), mu_q*face_area * surface_ratio/discretization_distance, ADD_VALUES); CHKERRXX(ierr); }
        }
        if(!matrix_is_set){
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(neighbor_quad.p.piggy3.local_num), -mu_q*face_area/discretization_distance, ADD_VALUES); CHKERRXX(ierr); }
      }
    }
    /* there is more than one neighbor, regular bulk case. This assumes uniform on interface ! */
    else if(direct_neighbors.size() > 1)
    {
      const double surface_quad = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1); // logical surface

      double discretization_distance = 0.0;
#ifdef DEBUG
      double split_face_check = 0.0;
#endif
      for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
      {
        const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_quad;
        discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
#ifdef DEBUG
          split_face_check += surface_ratio;
#endif
      }
      P4EST_ASSERT(fabs(split_face_check - 1.0) < EPS);
      discretization_distance *= tree_dimensions[dir/2];

      for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
      {
        const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_quad;
        if(!matrix_is_set){
          ierr = MatSetValue(A, quad_gloidx, compute_global_index(it->p.piggy3.local_num), -mu_q*face_area * surface_ratio/discretization_distance, ADD_VALUES); CHKERRXX(ierr); }
      }
      if(!matrix_is_set){
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu_q*face_area/discretization_distance, ADD_VALUES); CHKERRXX(ierr); }
    }
  }
}

void my_p4est_xgfm_cells_t::setup_linear_system()
{
  preallocate_matrix();

  if(matrix_is_set && rhs_is_set)
    return;
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  const double *phi_p;
  const double *jump_u_p     = NULL;
  const double *jump_flux_p  = NULL;
  const double *phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  my_p4est_interpolation_nodes_t interp_phi(fine_node_ngbd); interp_phi.set_input(phi, linear);

  double *rhs_p = NULL;
  const double *user_rhs_p = NULL;
  if(!rhs_is_set)
  {
    compute_jumps_in_flux_components(); // we need those to assemble the (discretized) rhs

    if(user_rhs == NULL)
      throw std::runtime_error("my_p4est_xgfm_cells_t::setup_linear_system() : the user must set its cell-sampled rhs, first!");
    ierr  = VecGetArrayRead(user_rhs, &user_rhs_p); CHKERRXX(ierr);

    if(rhs == NULL){
      ierr = VecCreateNoGhostCells(p4est, &rhs); CHKERRXX(ierr); }
    P4EST_ASSERT(VecIsSetForCells(rhs, p4est, ghost, 1, false));

    ierr  = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr  = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
    ierr  = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  }
  int nullspace_contains_constant_vector = !matrix_is_set; // integer because of required MPI collective determination thereafter, we don't care about that if the matrix is already set

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      build_discretization_for_quad(q + tree->quadrants_offset, tree_idx, phi_p, phi_xxyyzz_p, interp_phi, user_rhs_p, jump_u_p, jump_flux_p, nullspace_contains_constant_vector, rhs_p);
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
  // remove nullspace from discretized rhs
  if(!rhs_is_set && A_null_space != NULL) {
    ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr); }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  if(phi_xxyyzz_p != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  if(jump_u_p != NULL){
    ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr); }
  if(jump_flux_p != NULL){
    ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr); }
  if(rhs_p != NULL){
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr); }
  if(user_rhs_p != NULL){
    ierr = VecRestoreArrayRead(user_rhs, &user_rhs_p); CHKERRXX(ierr); }

  matrix_is_set = rhs_is_set = true;

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::solve(KSPType ksp_type, PCType pc_type,
                                  double absolute_accuracy_threshold, double tolerance_on_rel_residual)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2]));           // make sure we have wall boundary conditions if we need them
  P4EST_ASSERT(levelset_has_been_set() && normals_have_been_set() && jumps_have_been_set());  // make sure the problem is fully defined
  P4EST_ASSERT(absolute_accuracy_threshold > EPS && tolerance_on_rel_residual > EPS);         // those need to be (strictly positive)
  P4EST_ASSERT((solution == NULL && !use_initial_guess) || (solution != NULL && use_initial_guess));
  if(solution == NULL) {
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr); }

  /* reinitialize the counters and monitoring data */
  numbers_of_ksp_iterations.resize(0);
  max_corrections.resize(0);
  relative_residuals.resize(0);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  setup_linear_system();
  ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, tolerance_on_rel_residual, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  if (use_initial_guess) {
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr); }
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

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
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (A_null_space != NULL){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // Solve the system
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  PetscInt nb_iterations;
  ierr = KSPGetIterationNumber(ksp, &nb_iterations); CHKERRXX(ierr);
  numbers_of_ksp_iterations.push_back(nb_iterations);
  max_corrections.push_back(0.0);
  relative_residuals.push_back(0.0);
  ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(activate_xGFM && !mus_are_equal())// no correction if mu_m = mu_p
  {
    // clear the workspace before we start...
    if(extension_on_cells != NULL){
      ierr = VecDestroy(extension_on_cells); CHKERRXX(ierr);}
    if(extension_on_nodes != NULL) {
      ierr = VecDestroy(extension_on_nodes); extension_on_nodes = NULL; CHKERRXX(ierr); }

    // extend interface values
    ierr = VecCreateGhostCells(p4est, ghost, &extension_on_cells); CHKERRXX(ierr);
    extend_interface_values(solution, extension_on_cells);
    // interpolate it on the fine grid
    ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_nodes); CHKERRXX(ierr);
    double *extension_cell_values_p;
    ierr = VecGetArray(extension_on_cells, &extension_cell_values_p); CHKERRXX(ierr);
    interpolate_cell_field_to_nodes(extension_cell_values_p, extension_on_nodes);
    // calculate the correction for the rhs
    Vec corrected_rhs;
    ierr = VecCreateNoGhostCells(p4est, &corrected_rhs); CHKERRXX(ierr);
    compute_jumps_in_flux_components();
    get_corrected_rhs(corrected_rhs);
    // calculate the residual of the current solution
    Vec residual;
    ierr = VecCreateNoGhostCells(p4est, &residual); CHKERRXX(ierr);
    ierr = MatMult(A, solution, residual); CHKERRXX(ierr);
    // finish the operation as residual -= corrected_rhs
    // --> can be done in the same loop as the calculation of relevant two-norms
    // done in the following for optimization
    // (knowingly avoiding separate Petsc operations that would multiply the number of such loops)
    double *residual_p, *corrected_rhs_p, norm_residual_and_corrected_rhs[2];
    norm_residual_and_corrected_rhs[0] = norm_residual_and_corrected_rhs[1] = 0.0;
    ierr = VecGetArray(residual, &residual_p); CHKERRXX(ierr);
    ierr = VecGetArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
    for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
      residual_p[qq] -= corrected_rhs_p[qq];
      norm_residual_and_corrected_rhs[0] += SQR(residual_p[qq]);
      norm_residual_and_corrected_rhs[1] += SQR(corrected_rhs_p[qq]);
    }
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, norm_residual_and_corrected_rhs, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    norm_residual_and_corrected_rhs[0] = sqrt(norm_residual_and_corrected_rhs[0]);
    norm_residual_and_corrected_rhs[1] = sqrt(norm_residual_and_corrected_rhs[1]);

    relative_residuals[0] = norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1];

    double inner_products[2]; // inner_products[0] = <residual, residual - residual_tilde>; inner_products[1] = <residual - residual_tilde, residual - residual_tilde>;
    double step_size;
    Vec solution_tilde = NULL;
    ierr = VecCreateGhostCells(p4est, ghost, &solution_tilde); CHKERRXX(ierr);
    // update solution_tilde <= solution to have a good initial guess for next KSP solve
    // the original vector's ghost values are already up-to-date, no need to GhostUpdate...
    // [knowingly avoiding separate Petsc operations, because of simplicity (no VecGhostGetLocalForm, etc.)]
    ierr = VecCopyGhost(solution, solution_tilde); CHKERRXX(ierr);
    // other tilde vectors
    Vec extension_cell_values_tilde = NULL; double *extension_cell_values_tilde_p = NULL;
    Vec extension_on_nodes_tilde = NULL; double *extension_on_nodes_tilde_p = NULL;
    Vec rhs_tilde = NULL;
    Vec residual_tilde = NULL; double *residual_tilde_p = NULL;
    // convergence flag
    bool solver_has_converged = false;
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);

    while (!solver_has_converged) {

      ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution_tilde, corrected_rhs, ksp, 0); CHKERRXX(ierr);
      if (A_null_space != NULL) {
        ierr = MatNullSpaceRemove(A_null_space, corrected_rhs, NULL); CHKERRXX(ierr);
      }
      ierr = KSPSolve(ksp, corrected_rhs, solution_tilde); CHKERRXX(ierr);
      ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution_tilde, corrected_rhs, ksp, 0); CHKERRXX(ierr);
      ierr = KSPGetIterationNumber(ksp, &nb_iterations); CHKERRXX(ierr);
      if(nb_iterations == 0)
      {
        // the solver converged if the initial guess was so close that ksp did not run a single iteration --> fixed-point reached, no further correction to expect, break...
        numbers_of_ksp_iterations.push_back(nb_iterations);
        max_corrections.push_back(0.0);
        relative_residuals.push_back(relative_residuals[relative_residuals.size()-1]);
        break;
      }
      if(extension_cell_values_tilde == NULL)
      {
        ierr = VecCreateGhostCells(p4est, ghost, &extension_cell_values_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(extension_cell_values_tilde, &extension_cell_values_tilde_p); CHKERRXX(ierr);
        // update extension_cell_values_tilde <= extension_cell_values as well for good initial guess
        // the original vector's ghost values are already up-to-date, no need to GhostUpdate...
        // [knowingly avoiding separate Petsc operations, because of simplicity (no VecGhostGetLocalForm, etc.)]
        for (p4est_locidx_t qq = 0; qq < (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); ++qq)
          extension_cell_values_tilde_p[qq] = extension_cell_values_p[qq];
      }
      if(extension_on_nodes_tilde == NULL)
      {
        ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_nodes_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(extension_on_nodes_tilde, &extension_on_nodes_tilde_p); CHKERRXX(ierr);
      }
      if(rhs_tilde == NULL){
        ierr = VecCreateNoGhostCells(p4est, &rhs_tilde); CHKERRXX(ierr);

      }
      if(residual_tilde == NULL)
      {
        ierr = VecCreateNoGhostCells(p4est, &residual_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(residual_tilde, &residual_tilde_p); CHKERRXX(ierr);
      }

      ierr = VecGhostUpdateBegin(solution_tilde, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(solution_tilde, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      compute_jumps_in_flux_components();
      extend_interface_values(solution_tilde, extension_cell_values_tilde);
      interpolate_cell_field_to_nodes(extension_cell_values_tilde_p, extension_on_nodes_tilde);

      Vec tmp = extension_on_nodes;
      extension_on_nodes = extension_on_nodes_tilde;
      compute_jumps_in_flux_components();
      get_corrected_rhs(rhs_tilde);
      extension_on_nodes_tilde = extension_on_nodes;
      extension_on_nodes = tmp;

      ierr = MatMult(A, solution_tilde, residual_tilde); CHKERRXX(ierr);
      // finish the operation as residual -= rhs_tilde
      // --> can be done in the same loop as the calculation of relevant inner_product for min-res step
      // done in the following for optimization
      // (knowingly avoiding separate Petsc operations that would multiply the number of such loops)
      const double *rhs_tilde_p;
      ierr = VecGetArrayRead(rhs_tilde, &rhs_tilde_p); CHKERRXX(ierr);
      inner_products[0] = inner_products[1] = 0;
      for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
        residual_tilde_p[qq]  -= rhs_tilde_p[qq];
        inner_products[0]     += residual_p[qq]*(residual_p[qq] - residual_tilde_p[qq]);
        inner_products[1]     += (residual_p[qq] - residual_tilde_p[qq])*(residual_p[qq] - residual_tilde_p[qq]);
      }
      mpiret = MPI_Allreduce(MPI_IN_PLACE, inner_products, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      step_size = inner_products[0]/inner_products[1];
      ierr = VecRestoreArrayRead(rhs_tilde, &rhs_tilde_p); CHKERRXX(ierr);


      double *solution_tilde_p, *solution_p, *extension_on_nodes_p;
      ierr = VecGetArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);
      ierr = VecGetArray(solution_tilde, &solution_tilde_p); CHKERRXX(ierr);
      ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(rhs_tilde, &rhs_tilde_p); CHKERRXX(ierr);
      double max_correction = 0.0;
      norm_residual_and_corrected_rhs[0] = norm_residual_and_corrected_rhs[1] = 0.0;
      for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
        max_correction                      = MAX(max_correction, fabs(step_size*(solution_tilde_p[qq] - solution_p[qq])));
        residual_p[qq]                      = (1.0 - step_size)*residual_p[qq] + step_size*residual_tilde_p[qq];
        corrected_rhs_p[qq]                 = (1.0 - step_size)*corrected_rhs_p[qq] + step_size*rhs_tilde_p[qq];
        norm_residual_and_corrected_rhs[0] += SQR(residual_p[qq]);
        norm_residual_and_corrected_rhs[1] += SQR(corrected_rhs_p[qq]);
        solution_p[qq]                      = (1.0 - step_size)*solution_p[qq] + step_size*solution_tilde_p[qq];
        solution_tilde_p[qq]                = solution_p[qq]; // update initial guess for next KSPSolve
        extension_cell_values_p[qq]         = (1.0 - step_size)*extension_cell_values_p[qq] + step_size*extension_cell_values_tilde_p[qq];
        extension_cell_values_tilde_p[qq]   = extension_cell_values_p[qq]; // update next extension_cell_values_tilde for initial guess for extension
        if(qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count)
          extension_on_nodes_p[qq]          = (1.0 - step_size)*extension_on_nodes_p[qq] + step_size*extension_on_nodes_tilde_p[qq];
      }
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_correction, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, norm_residual_and_corrected_rhs, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      norm_residual_and_corrected_rhs[0] = sqrt(norm_residual_and_corrected_rhs[0]);
      norm_residual_and_corrected_rhs[1] = sqrt(norm_residual_and_corrected_rhs[1]);
      // update the ghost elements of ghosted cell-vectors
      for (p4est_locidx_t qq = p4est->local_num_quadrants; qq < (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); ++qq) {
        solution_p[qq]                      = (1.0 - step_size)*solution_p[qq] + step_size*solution_tilde_p[qq];
        solution_tilde_p[qq]                = solution_p[qq]; // update initial guess for next KSPSolve
        extension_cell_values_p[qq]         = (1.0 - step_size)*extension_cell_values_p[qq] + step_size*extension_cell_values_tilde_p[qq];
        extension_cell_values_tilde_p[qq]   = extension_cell_values_p[qq]; // update next extension_cell_values_tilde for initial guess for extension
        if(qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count)
          extension_on_nodes_p[qq]          = (1.0 - step_size)*extension_on_nodes_p[qq] + step_size*extension_on_nodes_tilde_p[qq];
      }
      // update the last element of extension_on_fine_nodes if needed
      if((p4est_locidx_t) fine_nodes->indep_nodes.elem_count > (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count))
        for (p4est_locidx_t qq = (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count; ++qq)
          extension_on_nodes_p[qq]          = (1.0 - step_size)*extension_on_nodes_p[qq] + step_size*extension_on_nodes_tilde_p[qq];
      ierr = VecRestoreArray(solution_tilde, &solution_tilde_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(rhs_tilde, &rhs_tilde_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(extension_on_nodes, &extension_on_nodes_p); CHKERRXX(ierr);

      numbers_of_ksp_iterations.push_back(nb_iterations);
      max_corrections.push_back(max_correction);
      relative_residuals.push_back(norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1]);

      size_t iter = relative_residuals.size() -1;
      solver_has_converged =
          (max_correction < absolute_accuracy_threshold) && // if_nb_iterations != 0, max_correction must be below the accuracy requirement AND
          ((norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1] < tolerance_on_rel_residual) || // either the relative residual is below the desired threshold as well
          (fabs(relative_residuals[iter] - relative_residuals[iter-1]) < 1.0e-6*MAX(relative_residuals[iter], relative_residuals[iter-1]))); // or we have reached a fixed-point for which the relative residual is above the threshold but can't really be made any smaller apparently
    }

    ierr = VecRestoreArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(residual, &residual_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(extension_on_cells, &extension_cell_values_p); CHKERRXX(ierr);

    compute_jumps_in_flux_components(); // now that we now the final (extended) interface solution, we can calculate the correct jumps in flux components (required for accurate evaluation of fluxes/one-sided derivatives thereafter)

    P4EST_ASSERT(solution_tilde != NULL);
    ierr = VecDestroy(solution_tilde); CHKERRXX(ierr);

    if(extension_cell_values_tilde_p != NULL){
      ierr = VecRestoreArray(extension_cell_values_tilde, &extension_cell_values_tilde_p); CHKERRXX(ierr);
      ierr = VecDestroy(extension_cell_values_tilde); CHKERRXX(ierr); }
    if(extension_on_nodes_tilde_p != NULL){
      ierr = VecRestoreArray(extension_on_nodes_tilde, &extension_on_nodes_tilde_p); CHKERRXX(ierr);
      ierr = VecDestroy(extension_on_nodes_tilde); CHKERRXX(ierr);
    }
    if(rhs_tilde != NULL){
      ierr = VecDestroy(rhs_tilde); CHKERRXX(ierr); }
    if(residual_tilde_p != NULL)
    {
      ierr = VecRestoreArray(residual_tilde, &residual_tilde_p); CHKERRXX(ierr);
      ierr = VecDestroy(residual_tilde); CHKERRXX(ierr);
    }
    ierr  = VecDestroy(corrected_rhs); CHKERRXX(ierr);
    ierr  = VecDestroy(residual); CHKERRXX(ierr);
  }
  solution_is_set = true;
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}

bool my_p4est_xgfm_cells_t::interface_neighbor_is_found(const p4est_locidx_t &quad_idx, const u_char& dir, interface_neighbor& int_nb) const
{
  P4EST_ASSERT(quad_idx >= 0 && quad_idx < p4est->local_num_quadrants && dir >= 0 && dir < P4EST_FACES);
  std::map<p4est_locidx_t, std::map<u_char, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.find(quad_idx);
  if(it != map_of_interface_neighbors.end())
  {
    std::map<u_char, interface_neighbor>::const_iterator itt = it->second.find(dir);
    if(itt != it->second.end())
    {
      int_nb = itt->second;
      return true;
    }
  }
  // not found in map!
  return false;
}

interface_neighbor my_p4est_xgfm_cells_t::get_interface_neighbor(const p4est_locidx_t& quad_idx, const u_char& dir, const p4est_locidx_t& nb_quad_idx,
                                                                 const p4est_locidx_t& quad_fine_node_idx, const p4est_locidx_t& nb_fine_node_idx,
                                                                 const double *phi_p, const double *phi_xxyyzz_p)
{
  P4EST_ASSERT(quad_idx >= 0 && quad_idx < p4est->local_num_quadrants && dir >= 0 && dir < P4EST_FACES); // must be a local quadrant

  interface_neighbor interface_nb;
  if(interface_neighbor_is_found(quad_idx, dir, interface_nb))
    return interface_nb;

  P4EST_ASSERT(quad_fine_node_idx >= 0);
  P4EST_ASSERT(nb_fine_node_idx >= 0);
#ifdef SUBREFINED
  P4EST_ASSERT(quad_fine_node_idx < (p4est_locidx_t)(fine_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(nb_fine_node_idx < (p4est_locidx_t)(fine_nodes->indep_nodes.elem_count));
#endif
  interface_nb.phi_q  = phi_p[quad_fine_node_idx];
  interface_nb.phi_nb = phi_p[nb_fine_node_idx];
  P4EST_ASSERT(signs_of_phi_are_different(phi_p[quad_fine_node_idx], phi_p[nb_fine_node_idx]));
  interface_nb.quad_nb_idx = nb_quad_idx;
  interface_nb.quad_fine_node_idx = quad_fine_node_idx;
  interface_nb.nb_fine_node_idx   = nb_fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn; fine_node_ngbd->get_neighbors(quad_fine_node_idx, qnnn);
  interface_nb.mid_point_fine_node_idx  = qnnn->neighbor(dir);
  P4EST_ASSERT(interface_nb.mid_point_fine_node_idx >= 0 && interface_nb.mid_point_fine_node_idx == fine_node_ngbd->get_neighbors(nb_fine_node_idx).neighbor(dir + (dir%2 == 0 ? +1 : -1)));

  const double &mid_point_phi   = phi_p[interface_nb.mid_point_fine_node_idx];
  const bool no_past_mid_point  = signs_of_phi_are_different(interface_nb.phi_q, mid_point_phi);
  const double &phi_this_side   = (no_past_mid_point ? interface_nb.phi_q : mid_point_phi);
  const double &phi_across      = (no_past_mid_point ? mid_point_phi      : interface_nb.phi_nb);
  const p4est_locidx_t& fine_idx_this_side  = (no_past_mid_point ? quad_fine_node_idx                   : interface_nb.mid_point_fine_node_idx);
  const p4est_locidx_t& fine_idx_across     = (no_past_mid_point ? interface_nb.mid_point_fine_node_idx : nb_fine_node_idx);

  if(phi_xxyyzz_p != NULL)
    interface_nb.theta  = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, phi_xxyyzz_p[P4EST_DIM*fine_idx_this_side + dir/2], phi_xxyyzz_p[P4EST_DIM*fine_idx_across + dir/2], 0.5*dxyz_min[dir/2]);
  else
    interface_nb.theta  = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, 0.5*dxyz_min[dir/2], 0.5*dxyz_min[dir/2]);
  interface_nb.theta = (phi_this_side > 0.0 ? 1.0 - interface_nb.theta : interface_nb.theta);
  interface_nb.theta = MAX(0.0, MIN(interface_nb.theta, 1.0));
  interface_nb.theta = 0.5*(interface_nb.theta + (no_past_mid_point ? 0.0 : 1.0));

  map_of_interface_neighbors[quad_idx][dir] = interface_nb; // add it to the map so that future access is read from memory;
  return interface_nb;
}

void my_p4est_xgfm_cells_t::update_interface_values(Vec new_cell_extension, const double *solution_p)
{
  P4EST_ASSERT(VecIsSetForCells(new_cell_extension, p4est, ghost, 1));
  P4EST_ASSERT(levelset_has_been_set() && jumps_have_been_set());
  P4EST_ASSERT(normals_have_been_set());

  const double *phi_p, *normals_p, *phi_xxyyzz_p, *jump_u_p, *jump_flux_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);

  if(interface_values_are_set)
  {
    for (std::map<p4est_locidx_t, std::map<u_char, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
         it != map_of_interface_neighbors.end(); ++it)
    {
      p4est_locidx_t quad_idx = it->first;
      for (std::map<u_char, interface_neighbor>::const_iterator itt = map_of_interface_neighbors[quad_idx].begin();
           itt != map_of_interface_neighbors[quad_idx].end(); ++itt)
      {
        u_char dir           = itt->first;
        interface_neighbor& int_nb  = map_of_interface_neighbors[quad_idx][dir];
        int_nb.int_value = int_nb.interface_value(mu_m, mu_p, quad_idx, dir, dxyz_min, solution_p, jump_u_p, jump_flux_p);
      }
    }
  }
  else
  {
    // The interface values are set but initial guesses for the extended values are built as well
    double *new_cell_extension_p;
    ierr = VecGetArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
    splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
    p4est_locidx_t fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad;
    my_p4est_interpolation_nodes_t interp_fine_nodes(fine_node_ngbd);
    Vec inputs[2] = {phi, jump_u}; interp_fine_nodes.set_input(inputs, linear, 2);
    double interp_output[2];
    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
      const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q)
      {
        const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, q);
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;

        double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
        const bool is_quad_a_fine_node = quad_center_is_fine_node(*quad, tree_idx, fine_node_idx_for_quad);
        if(!is_quad_a_fine_node)
          interp_fine_nodes(xyz_quad, interp_output);
        const double phi_q          = (is_quad_a_fine_node ? phi_p[fine_node_idx_for_quad] : interp_output[0]);
        const double jump_u_at_quad = (is_quad_a_fine_node ? jump_u_p[fine_node_idx_for_quad] : interp_output[1]);

        // build the educated guess
        new_cell_extension_p[quad_idx] = solution_p[quad_idx];
        if(mu_m_is_larger() && phi_q > 0.0)
          new_cell_extension_p[quad_idx] -= jump_u_at_quad;
        else if(!mu_m_is_larger() && phi_q <= 0.0)
          new_cell_extension_p[quad_idx] += jump_u_at_quad;
        // set the interface values
        if(quad->level == data->max_lvl)
        {
          // finest quadrant, check if neighboring the interface and fill relevant data in std::vector structures
          for(u_char dir = 0; dir < P4EST_FACES; ++dir)
          {
            set_of_neighboring_quadrants neighbor_cells;
            cell_ngbd->find_neighbor_cells_of_cell(neighbor_cells, quad_idx, tree_idx, dir);
            if(neighbor_cells.size() == 1 && neighbor_cells.begin()->level == quad->level)
            {
              const p4est_quadrant_t& neighbor_quad = *neighbor_cells.begin();

              double xyz_nb_quad[P4EST_DIM]; quad_xyz_fr_q(neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, p4est, ghost, xyz_nb_quad);
              const bool is_neighbor_quad_a_fine_node = quad_center_is_fine_node(neighbor_quad, neighbor_quad.p.piggy3.which_tree, fine_node_idx_for_neighbor_quad);
              if(!is_neighbor_quad_a_fine_node)
                interp_fine_nodes(xyz_nb_quad, interp_output);
              const double phi_tmp          = (is_neighbor_quad_a_fine_node ? phi_p[fine_node_idx_for_neighbor_quad] : interp_output[0]);
              if(signs_of_phi_are_different(phi_q, phi_tmp))
              {
                P4EST_ASSERT(is_quad_a_fine_node && is_neighbor_quad_a_fine_node);
                P4EST_ASSERT(quad->level == neighbor_quad.level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
                interface_neighbor int_nb = get_interface_neighbor(quad_idx, dir, neighbor_quad.p.piggy3.local_num, fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad, phi_p, phi_xxyyzz_p);
                P4EST_ASSERT(interface_neighbor_is_found(quad_idx, dir, int_nb));
                map_of_interface_neighbors[quad_idx][dir].int_value = int_nb.interface_value(mu_m, mu_p, quad_idx, dir, dxyz_min, solution_p, jump_u_p, jump_flux_p);
              }
            }
          }
        }
      }
    }
    ierr = VecGhostUpdateBegin(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef CASL_THROWS
    P4EST_ASSERT(is_map_consistent());
#endif
    ierr = VecGhostUpdateEnd(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  interface_values_are_set = true;
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_of_interface_values(Vec new_cell_extension, const double& threshold, const uint& niter_max)
{
  P4EST_ASSERT(VecIsSetForCells(new_cell_extension, p4est, ghost, 1));
  P4EST_ASSERT(levelset_has_been_set() && threshold > EPS && niter_max > 0);
  const double *phi_p, *normals_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normals, &normals_p); CHKERRXX(ierr);
  double *new_cell_extension_p;
  ierr = VecGetArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);

  p4est_locidx_t fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad;
  my_p4est_interpolation_nodes_t *interp_phi = NULL, *interp_normal = NULL;
  if(!extension_entries_are_set){
    interp_phi    = new my_p4est_interpolation_nodes_t(fine_node_ngbd); interp_phi->set_input(phi, linear);
    interp_normal = new my_p4est_interpolation_nodes_t(fine_node_ngbd); interp_normal->set_input(normals, linear, P4EST_DIM);
  }

  double max_corr = 10.0*threshold;
  uint iter = 0;

  bool reverse_flag = true; // we reverse the z-ordering between two iterations to alleviate the slow convergence along the characteristic direction perpendicular to the main domain diagonal (z-order)
  while (max_corr > threshold && iter < niter_max)
  {
    max_corr = 0.0; // reset the measure;
    /* Main loop over all local quadrant */
    for (p4est_topidx_t tree_idx = (reverse_flag ? p4est->last_local_tree : p4est->first_local_tree) ;
         (reverse_flag ?  tree_idx >= p4est->first_local_tree : tree_idx <= p4est->last_local_tree) ;
         (reverse_flag ?  --tree_idx: ++tree_idx))
    {
      const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0 ; q < tree->quadrants.elem_count ; q++)
      {
        double phi_q;
        double increment = 0.0;
        const p4est_locidx_t quad_idx = (reverse_flag ? tree->quadrants.elem_count - 1 - q : q) + tree->quadrants_offset;
        if(extension_entries_are_set)
        {
          if(extension_entries[quad_idx].too_close)
          {
            P4EST_ASSERT(fabs(extension_entries[quad_idx].diag_entry) < EPS);
            new_cell_extension_p[quad_idx]  = map_of_interface_neighbors[quad_idx][extension_entries[quad_idx].forced_interface_value_dir].int_value;
          }
          else
          {
            increment += (extension_entries[quad_idx].diag_entry)*new_cell_extension_p[quad_idx];
            for (size_t i = 0; i < extension_entries[quad_idx].interface_entries.size(); ++i)
              increment += (extension_entries[quad_idx].interface_entries[i].coeff)*(map_of_interface_neighbors[quad_idx][extension_entries[quad_idx].interface_entries[i].dir].int_value);
            for (size_t i = 0; i < extension_entries[quad_idx].quad_entries.size(); ++i)
              increment += (extension_entries[quad_idx].quad_entries[i].coeff)*(new_cell_extension_p[extension_entries[quad_idx].quad_entries[i].loc_idx]);
            increment *= extension_entries[quad_idx].dtau;
          }
          phi_q = extension_entries[quad_idx].phi_q;
        }
        else
        {
          // we'll build the extension matrix entries!
          // the extension procedure, can be formally written as
          // increment = A*cell_values + B*interface_values
          // where A and B are appropriate (sparse) matrices
          // here below, we build the entries of A and B.
          // For the local quadrant of index quad_idx,
          // - extension_entries[quad_idx].quad_entries = vector of structures encompassing i) the local number of a relevant neighbor ii) the coefficient by which this neighbor-value must be multiplied to contribute to -sgn_n*grad_u
          // - extension_entries[quad_idx].interface_entries = vector of structures encompassing i) the relevant direction in which the interface neighbor can be found, ii) the coefficient by which this neighbor-value must be multiplied to contribute to -sgn_n*grad_u
          // - extension_entries[quad_idx].diag_entry = value of the coefficient multiplying the local value of the current quadrant when contributing to -sgn_n*grad_u
          // - extension_entries[quad_idx].dtau = is the fictititious time step for the considered cell, satisfying the cfl condition;
          // - extension_entries[quad_idx].too_close is set to true when the cell value is an interface point --> enforcing the relevant interface value there;
          extension_entries[quad_idx].diag_entry = 0.0;
          const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, (reverse_flag ? tree->quadrants.elem_count - 1 - q: q));

          double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
          const bool is_quad_a_fine_node = quad_center_is_fine_node(*quad, tree_idx, fine_node_idx_for_quad);
          phi_q = (is_quad_a_fine_node ? phi_p[fine_node_idx_for_quad] : (*interp_phi)(xyz_quad));
          double sgn_n[P4EST_DIM];
          if(is_quad_a_fine_node){
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] = normals_p[P4EST_DIM*fine_node_idx_for_quad + dim];
          }
          else
            interp_normal->operator()(xyz_quad, sgn_n);
          const double mag = sqrt(SUMD(SQR(sgn_n[0]), SQR(sgn_n[1]), SQR(sgn_n[2])));

          if (mag < EPS)
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] = 0.0;
          else
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] = (phi_q <= 0.0 ? -1.0 : +1.0)*sgn_n[dim]/mag;

          double dtau = DBL_MAX;
          for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          {
            int dir = 2*dim + (sgn_n[dim] > 0.0 ? 0 : 1);
            if(is_quad_Wall(p4est, tree_idx, quad, dir))
              continue; // homogeneous Neumann boundary condition

            set_of_neighboring_quadrants neighbor_cells;
            cell_ngbd->find_neighbor_cells_of_cell(neighbor_cells, quad_idx, tree_idx, dir);

            if(neighbor_cells.size() == 1)
            {
              const p4est_quadrant_t& neighbor_quad = *neighbor_cells.begin();
              double xyz_nb_quad[P4EST_DIM]; quad_xyz_fr_q(neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, p4est, ghost, xyz_nb_quad);
              const double phi_nb = (quad_center_is_fine_node(neighbor_quad, neighbor_quad.p.piggy3.which_tree, fine_node_idx_for_neighbor_quad) ? phi_p[fine_node_idx_for_neighbor_quad] : (*interp_phi)(xyz_nb_quad));

              /* If interface across the two cells.
               * We assume that the interface is tesselated with uniform finest grid level */
              if(signs_of_phi_are_different(phi_q, phi_nb))
              {
                double theta = map_of_interface_neighbors[quad_idx][dir].theta;
                double interface_value = map_of_interface_neighbors[quad_idx][dir].int_value;
                if(theta > EPS)
                {
                  increment      -= sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)*(new_cell_extension_p[quad_idx] - interface_value)/(theta*dxyz_min[dim]);
                  dtau            = MIN(dtau, theta*dxyz_min[dim]/((double) P4EST_DIM));
                  extension_interface_value_entry int_entry; int_entry.dir = dir; int_entry.coeff = +sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)/(theta*dxyz_min[dim]);
                  extension_entries[quad_idx].interface_entries.push_back(int_entry);
                  extension_entries[quad_idx].diag_entry -= sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)/(theta*dxyz_min[dim]);
                }
                else
                {
                  new_cell_extension_p[quad_idx]  = interface_value;
                  dtau                            = 0.0;
                  extension_entries[quad_idx].interface_entries.resize(0);
                  extension_entries[quad_idx].quad_entries.resize(0);
                  extension_entries[quad_idx].too_close = true;
                  extension_entries[quad_idx].diag_entry = 0.0;
                  extension_entries[quad_idx].forced_interface_value_dir = dir;
                }
              }
              /* no interface - regular discretization */
              else
              {
                const double surface_direct_neighbor = pow((double)P4EST_QUADRANT_LEN(neighbor_cells.begin()->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1); // logical surface

                set_of_neighboring_quadrants sharing_quads;
                cell_ngbd->find_neighbor_cells_of_cell(sharing_quads, neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, (dir%2 == 0 ? dir + 1 : dir - 1));

                double discretization_distance = 0.0;
                for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
                {
                  const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_direct_neighbor;
                  discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(neighbor_quad.level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
                }
                discretization_distance *= tree_dimensions[dir/2];
                extension_matrix_entry mat_entry_nb_quad; mat_entry_nb_quad.loc_idx = neighbor_quad.p.piggy3.local_num; mat_entry_nb_quad.coeff = 0.0;
                for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
                {
                  const double  surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_direct_neighbor;
                  increment -= sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)*(new_cell_extension_p[it->p.piggy3.local_num] - new_cell_extension_p[neighbor_quad.p.piggy3.local_num])*surface_ratio/discretization_distance;
                  extension_matrix_entry mat_entry; mat_entry.loc_idx = it->p.piggy3.local_num; mat_entry.coeff = -sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)*surface_ratio/discretization_distance;
                  extension_entries[quad_idx].quad_entries.push_back(mat_entry);
                  mat_entry_nb_quad.coeff += sgn_n[dir/2]*(dir%2 == 0 ? +1.0 : -1.0)*surface_ratio/discretization_distance;
                }
                extension_entries[quad_idx].quad_entries.push_back(mat_entry_nb_quad);
                dtau = MIN(dtau, discretization_distance/((double) P4EST_DIM));
              }
            }
            /* there is more than one neighbor, regular bulk case. This assumes uniform on interface ! */
            else
            {
              const double surface_quad = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1); // logical surface
              double discretization_distance = 0.0;
              for (set_of_neighboring_quadrants::const_iterator it = neighbor_cells.begin(); it != neighbor_cells.end(); ++it)
              {
                const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_quad;
                discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
              }
              discretization_distance *= tree_dimensions[dir/2];
              for (set_of_neighboring_quadrants::const_iterator it = neighbor_cells.begin(); it != neighbor_cells.end(); ++it)
              {
                const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_quad;
                increment -= sgn_n[dir/2]*(dir%2 == 0 ? -1.0 : +1.0)*(new_cell_extension_p[it->p.piggy3.local_num] - new_cell_extension_p[quad_idx])*surface_ratio/discretization_distance;
                extension_matrix_entry mat_entry; mat_entry.loc_idx = it->p.piggy3.local_num; mat_entry.coeff = -sgn_n[dir/2]*(dir%2 == 0 ? -1.0 : +1.0)*surface_ratio/discretization_distance;
                extension_entries[quad_idx].quad_entries.push_back(mat_entry);
                extension_entries[quad_idx].diag_entry += sgn_n[dir/2]*(dir%2 == 0 ? -1.0 : +1.0)*surface_ratio/discretization_distance;
              }
              dtau = MIN(dtau, discretization_distance/((double) P4EST_DIM));
            }
          }
          extension_entries[quad_idx].dtau = dtau;
          extension_entries[quad_idx].phi_q = phi_q;
          increment *= dtau;
        }
        new_cell_extension_p[quad_idx] += increment;

        if(fabs(phi_q) < 3.0*diag_min())
          max_corr = MAX(max_corr, fabs(increment));
      }
    }

    extension_entries_are_set = true;
    ierr = VecGhostUpdateBegin(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_corr, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = VecGhostUpdateEnd(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    reverse_flag = !reverse_flag;
    iter++;
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
  if(interp_phi != NULL)
    delete interp_phi;
  if(interp_normal != NULL)
    delete interp_normal;
}

void my_p4est_xgfm_cells_t::extend_interface_values(Vec cell_solution, Vec new_cell_extension, double threshold, uint niter_max)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_extend_field, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(new_cell_extension != NULL && VecIsSetForCells(new_cell_extension, p4est, ghost, 1));
  P4EST_ASSERT(levelset_has_been_set() && jumps_have_been_set() && threshold > EPS && niter_max > 0);

  // Update (or set) the map of interface information before iterative procedure
  // and build the educated guess for cell_extension_p if extension_not_defined_yet
  const double* solution_p;
  ierr = VecGetArrayRead(cell_solution, &solution_p); CHKERRXX(ierr);
  update_interface_values(new_cell_extension, solution_p);
  ierr = VecRestoreArrayRead(cell_solution, &solution_p); CHKERRXX(ierr);
  // TVD cell-centered extension
  cell_TVD_extension_of_interface_values(new_cell_extension, threshold, niter_max);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_extend_field, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::interpolate_cell_field_to_nodes(const double *cell_field_p, Vec interpolated_node_field)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_interpolate_cell_field_to_nodes, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef SUBREFINED
  P4EST_ASSERT(VecIsSetForNodes(interpolated_node_field, fine_nodes, p4est->mpicomm, 1));
  const my_p4est_node_neighbors_t *ngbd_n = fine_node_ngbd;
#else
  P4EST_ASSERT(VecIsSetForNodes(interpolated_node_field, nodes, p4est->mpicomm, 1));
  const my_p4est_node_neighbors_t *ngbd_n = node_ngbd;
#endif

  double *interpolated_node_field_p;
  ierr = VecGetArray(interpolated_node_field, &interpolated_node_field_p); CHKERRXX(ierr);

  for (size_t nnn = 0; nnn < ngbd_n->get_layer_size(); ++nnn)
  {
    const p4est_locidx_t fine_node_idx = ngbd_n->get_layer_node(nnn);
    interpolated_node_field_p[fine_node_idx] = interpolate_cell_field_at_local_node(fine_node_idx, cell_field_p);
  }
  ierr = VecGhostUpdateBegin(interpolated_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t nnn = 0; nnn < ngbd_n->get_local_size(); ++nnn)
  {
    const p4est_locidx_t fine_node_idx = ngbd_n->get_local_node(nnn);
    interpolated_node_field_p[fine_node_idx] = interpolate_cell_field_at_local_node(fine_node_idx, cell_field_p);
  }
  ierr = VecGhostUpdateEnd(interpolated_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(!local_interpolators_are_set) // if they were not already set, now they are!
    local_interpolators_are_set = true;

  ierr = VecRestoreArray(interpolated_node_field, &interpolated_node_field_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_interpolate_cell_field_to_nodes, 0, 0, 0, 0); CHKERRXX(ierr);
}

double my_p4est_xgfm_cells_t::interpolate_cell_field_at_local_node(const p4est_locidx_t &node_idx, const double *cell_field_p)
{
  cell_field_interpolator_t &cell_interpolator = local_interpolator[node_idx];

  if(local_interpolators_are_set) // no need for the hardwork if we have already done and memorized it
    return cell_interpolator.interpolate(cell_field_p);

#ifdef SUBREFINED
  if(((splitting_criteria_t*) fine_p4est->user_pointer)->max_lvl > ((splitting_criteria_t*) p4est->user_pointer)->max_lvl + 1)
    throw std::runtime_error("my_p4est_xgfm_cells_t::interpolate_cell_field_at_fine_node() is not implemented yet for a level difference larger than 1");
#endif

  cell_interpolator.clear();
  double to_return;
  double xyz_node[P4EST_DIM];
#ifdef SUBREFINED
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
    const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double)logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
    to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, nearby_cell_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
    // note : if locally uniform the result of the above is the expected arithmetic average of neighbor cells
  }
#ifdef SUBREFINED
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
      cell_interpolator.add_interpolation_factor(coarse_quad.p.piggy3.local_num, 1.0);
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
          cell_interpolator.add_interpolation_factor(it->p.piggy3.local_num, 1.0/(double) quads_sharing_fine_node.size());
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
        const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double)logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
        to_return = get_lsqr_interpolation_at_node(ni, xyz_node, cell_ngbd, lsqr_neighbors, scaling, cell_field_p, 2, xgfm_threshold_cond_number_lsqr, &cell_interpolator);
      }
    }
  }
#endif

#ifdef DEBUG
  const double value_check = cell_interpolator.interpolate(cell_field_p);
  if(ISNAN(value_check))
    std::cout << "my_p4est_xgfm_cells_t::interpolate_cell_field_at_fine_node(): tracking a NAN : the local_interpolator returns a NAN for node_idx = " << node_idx
             #ifdef SUBREFINED
              << " (on the interface-capturing grid)"
             #endif
              << " on proc " << p4est->mpirank << std::endl;
  P4EST_ASSERT(fabs(value_check - to_return) < MAX(EPS, 0.000001*MAX(fabs(value_check), fabs(to_return))));
#endif
  return to_return;
}

//void my_p4est_xgfm_cells_t::get_corrected_rhs(Vec corrected_rhs, const double *fine_extension_interface_values_p)
void my_p4est_xgfm_cells_t::get_corrected_rhs(Vec corrected_rhs)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_get_corrected_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(VecIsSetForCells(corrected_rhs, p4est, ghost, 1, false));
  P4EST_ASSERT(interface_values_are_set); // the corrected_rhs cannot be calculated before the first pass in the extension (the map of neighbor interface values must be set first)
  P4EST_ASSERT(normals_have_been_set());
  // reset corrected_rhs!
  ierr = VecCopy(rhs, corrected_rhs); CHKERRXX(ierr);

  // correct it!
  double *corrected_rhs_p;
  ierr = VecGetArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
  const double *phi_p, *normals_p, *phi_xxyyzz_p, *jump_u_p, *jump_flux_p, *user_rhs_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(user_rhs, &user_rhs_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  my_p4est_interpolation_nodes_t interp_phi(fine_node_ngbd); interp_phi.set_input(phi, linear);
  int dont_care;

#ifdef DEBUG
  splitting_criteria_t* data = (splitting_criteria_t*) p4est->user_pointer;
#endif

  rhs_is_set = false;

  for (std::map<p4est_locidx_t, std::map<u_char, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
       it != map_of_interface_neighbors.end(); ++it)
  {
    const p4est_locidx_t quad_idx  = it->first;
    // find tree_idx
    p4est_topidx_t tree_idx  = p4est->first_local_tree;
    while(tree_idx < p4est->last_local_tree && quad_idx >= p4est_tree_array_index(p4est->trees, tree_idx + 1)->quadrants_offset){ tree_idx++; }
#ifdef P4EST_DEBUG
    // get the quadrant
    const p4est_tree_t* tree  = p4est_tree_array_index(p4est->trees, tree_idx);
    P4EST_ASSERT(quad_idx - tree->quadrants_offset >=0);
    const p4est_quadrant_t *quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
    P4EST_ASSERT(quad->level == data->max_lvl);
#endif
    build_discretization_for_quad(quad_idx, tree_idx, phi_p, phi_xxyyzz_p, interp_phi, user_rhs_p, jump_u_p, jump_flux_p, dont_care, corrected_rhs_p);
  }

  rhs_is_set = true;

  ierr = VecRestoreArrayRead(normals, &normals_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(user_rhs, &user_rhs_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_get_corrected_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec flux[], my_p4est_faces_t *faces, Vec vstar[], Vec vnp1_minus[], Vec vnp1_plus[])
{
  P4EST_ASSERT(p4est_is_equal(p4est, faces->get_p4est(), P4EST_FALSE)); // the faces must be built from the same computational grid
  P4EST_ASSERT(jump_flux != NULL); // the flux vectors cannot be calculated if the jumps in flux components have been returned to the user
  P4EST_ASSERT(!solution_is_set || solution != NULL); // the flux vectors cannot be calculated if the solution has been returned to the user
  P4EST_ASSERT((vstar == NULL && vnp1_minus == NULL && vnp1_plus == NULL) || (VecsAreSetForFaces(vstar, faces, 1) && VecsAreSetForFaces(vnp1_minus, faces, 1) && VecsAreSetForFaces(vnp1_plus, faces, 1))); // the face-sampled velocities vstart and vnp1 vectors vectors must either be all defined or be all NULL.
  P4EST_ASSERT(VecsAreSetForFaces(flux, faces, 1));

  if(!solution_is_set)
    solve();
  P4EST_ASSERT(solution_is_set && solution != NULL);
  const bool velocities_are_defined = (vstar != NULL && vnp1_minus != NULL && vnp1_plus != NULL);

  const double *vstar_p[P4EST_DIM], *phi_p, *solution_p, *jump_u_p, *jump_flux_p;
  const double *phi_xxyyzz_p = NULL;
  double *vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM], *flux_p[P4EST_DIM];
  std::vector<bool> visited_faces[P4EST_DIM];
  p4est_locidx_t num_visited_faces[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr); }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    visited_faces[dim].resize(faces->num_local[dim], false);
    num_visited_faces[dim] = 0;
    ierr = VecGetArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    if(velocities_are_defined){
      ierr = VecGetArrayRead(vstar[dim], &vstar_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
  }
  my_p4est_interpolation_nodes_t interp_phi(fine_node_ngbd); interp_phi.set_input(phi, linear);
  p4est_locidx_t fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad, fine_node_idx_for_face;

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      const p4est_quadrant_t* quad  = p4est_const_quadrant_array_index(&tree->quadrants, q);
      const double logical_quad_size = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
      const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_quad_size, tree_dimensions[1]*logical_quad_size, tree_dimensions[2]*logical_quad_size)};
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
      const double phi_q = (quad_center_is_fine_node(*quad, tree_idx, fine_node_idx_for_quad) ? phi_p[fine_node_idx_for_quad] : interp_phi(xyz_quad));

      for (u_char dir = 0; dir < P4EST_FACES; ++dir) {
        p4est_locidx_t face_idx = faces->q2f(quad_idx, dir);
        if (face_idx >= faces->num_local[dir/2] || (face_idx != NO_VELOCITY && visited_faces[dir/2][face_idx]))
          continue;

        double xyz_face[P4EST_DIM] = {DIM(xyz_quad[0], xyz_quad[1], xyz_quad[2])};
        xyz_face[dir/2] += (dir%2 == 1 ? +0.5 : -0.5)*cell_dxyz[dir/2];
        const double phi_face = (face_in_quad_is_fine_node(*quad, tree_idx, dir, fine_node_idx_for_face) ? phi_p[fine_node_idx_for_face] : interp_phi(xyz_face));

        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          P4EST_ASSERT(face_idx != NO_VELOCITY);
#ifdef CASL_THROWS
          if(signs_of_phi_are_different(phi_q, phi_face))
            throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(): a wall-cell is crossed by the interface...");
#endif
          const double mu_face   = (phi_face > 0.0 ? mu_p : mu_m);
          switch(bc->wallType(xyz_face))
          {
          case DIRICHLET:
            flux_p[dir/2][face_idx] = (dir%2 == 1 ? +1.0 : -1.0)*(2.0*mu_face*(bc->wallValue(xyz_face) - solution_p[quad_idx])/cell_dxyz[dir/2]);
            break;
          case NEUMANN:
            flux_p[dir/2][face_idx] = (dir%2 == 1 ? +1.0 : -1.0)*mu_face*bc->wallValue(xyz_face);
            break;
          default:
            throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(): unknown boundary condition on a wall.");
          }
          if(velocities_are_defined)
          {
            if(phi_face > 0.0)
            {
              vnp1_plus_p[dir/2][face_idx]  = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
              vnp1_minus_p[dir/2][face_idx]  = DBL_MAX;
            }
            else
            {
              vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
              vnp1_minus_p[dir/2][face_idx] = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
            }
          }
          visited_faces[dir/2][face_idx] = true;
          num_visited_faces[dir/2]++;
          continue;
        }
        /* now get the neighbors */
        set_of_neighboring_quadrants direct_neighbors;
        cell_ngbd->find_neighbor_cells_of_cell(direct_neighbors, quad_idx, tree_idx, dir);
        if(direct_neighbors.size() == 1)
        {
          P4EST_ASSERT(face_idx != NO_VELOCITY);
          const p4est_quadrant_t &neighbor_quad = *direct_neighbors.begin();
          double xyz_neighbor_quad[P4EST_DIM]; quad_xyz_fr_q(neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, p4est, ghost, xyz_neighbor_quad);
          const double phi_neighbor_quad = (quad_center_is_fine_node(neighbor_quad, neighbor_quad.p.piggy3.which_tree, fine_node_idx_for_neighbor_quad) ? phi_p[fine_node_idx_for_neighbor_quad] : interp_phi(xyz_neighbor_quad));

          /* If interface across the two cells.
           * We assume that the interface is tesselated with uniform finest grid level */
          if(signs_of_phi_are_different(phi_q, phi_face) || signs_of_phi_are_different(phi_neighbor_quad, phi_face))
          {
            P4EST_ASSERT(quad->level == neighbor_quad.level && quad->level == ((splitting_criteria_t*)(p4est->user_pointer))->max_lvl);
            P4EST_ASSERT(!visited_faces[dir/2][face_idx]);
            P4EST_ASSERT(fine_node_idx_for_quad >= 0 && fine_node_idx_for_neighbor_quad >= 0 && fine_node_idx_for_face >= 0);

            if(signs_of_phi_are_different(phi_q, phi_neighbor_quad) || !signs_of_phi_are_different(phi_q, phi_face))// not under-resolved
            {
              p4est_quadrant_t this_quad = *quad;
              this_quad.p.piggy3.local_num = quad_idx; this_quad.p.piggy3.which_tree = tree_idx;
              P4EST_ASSERT(this_quad.level == neighbor_quad.level && this_quad.level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
              const interface_neighbor int_nb = get_interface_neighbor(this_quad.p.piggy3.local_num, dir, neighbor_quad.p.piggy3.local_num, fine_node_idx_for_quad, fine_node_idx_for_neighbor_quad, phi_p, phi_xxyyzz_p);
              P4EST_ASSERT(int_nb.mid_point_fine_node_idx == fine_node_idx_for_face);

              const double &mu_this_side  = (int_nb.phi_q <= 0.0 ? mu_m : mu_p);
              const double &mu_across     = (int_nb.phi_nb <= 0.0 ? mu_m : mu_p);
              const double mu_tilde       = (1.0 - int_nb.theta)*mu_this_side + int_nb.theta*mu_across;
              if(int_nb.theta >= 0.5)
              {
                P4EST_ASSERT(signs_of_phi_are_different(phi_face, int_nb.phi_nb));
                P4EST_ASSERT(!signs_of_phi_are_different(phi_face, int_nb.phi_q));
                P4EST_ASSERT(int_nb.theta <= 1.0);
                flux_p[dir/2][face_idx] =
                    (phi_face <= 0.0? mu_m : mu_p)*
                    ((dir%2 == 1?  1.0 : -1.0)*(mu_across/mu_tilde)*(solution_p[int_nb.quad_nb_idx] + (int_nb.phi_q <= 0.0 ? -1.0 : 1.0)*((2.0*int_nb.theta - 1.0)*jump_u_p[int_nb.nb_fine_node_idx] + (2.0 - 2.0*int_nb.theta)*jump_u_p[int_nb.mid_point_fine_node_idx]) - solution_p[quad_idx])/cell_dxyz[dir/2] +
                    (int_nb.phi_q <= 0.0 ? -1.0 : +1.0)*((2.0*int_nb.theta - 1.0)*jump_flux_p[P4EST_DIM*int_nb.nb_fine_node_idx + (dir/2)] + + (2.0 - 2.0*int_nb.theta)*jump_flux_p[P4EST_DIM*int_nb.mid_point_fine_node_idx + (dir/2)])*(1.0 - int_nb.theta)/mu_tilde);
                visited_faces[dir/2][face_idx] = true;
                num_visited_faces[dir/2]++;
              }
              else
              {
                P4EST_ASSERT(!signs_of_phi_are_different(phi_face, int_nb.phi_nb));
                P4EST_ASSERT(signs_of_phi_are_different(phi_face, int_nb.phi_q));
                P4EST_ASSERT(int_nb.theta >= 0.0);
                flux_p[dir/2][face_idx] =
                    (phi_face <= 0.0 ? mu_m : mu_p)*
                    ((dir%2 == 1 ? 1.0 : -1.0)*(mu_this_side/mu_tilde)*(solution_p[int_nb.quad_nb_idx] + (int_nb.phi_q <= 0.0 ? -1.0 : 1.0)*(2.0*int_nb.theta*jump_u_p[int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_u_p[int_nb.quad_fine_node_idx]) - solution_p[quad_idx])/cell_dxyz[dir/2] +
                    (int_nb.phi_q <= 0.0 ? +1.0 : -1.0)*(2.0*int_nb.theta*jump_flux_p[P4EST_DIM*int_nb.mid_point_fine_node_idx + (dir/2)] + (1.0 - 2.0*int_nb.theta)*jump_flux_p[P4EST_DIM*int_nb.quad_fine_node_idx + (dir/2)])*int_nb.theta/mu_tilde);
                visited_faces[dir/2][face_idx] = true;
                num_visited_faces[dir/2]++;
              }
            }
            else
            {
              P4EST_ASSERT(!signs_of_phi_are_different(phi_q, phi_neighbor_quad) && signs_of_phi_are_different(phi_q, phi_face));
              flux_p[dir/2][face_idx] = (phi_face > 0.0 ? mu_m : mu_p)*(solution_p[neighbor_quad.p.piggy3.local_num] - solution_p[quad_idx])/cell_dxyz[dir/2] + (phi_face > 0.0 ? +1.0 : -1.0)*jump_flux_p[P4EST_DIM*fine_node_idx_for_face + dir/2];
              visited_faces[dir/2][face_idx] = true;
              num_visited_faces[dir/2]++;
            }

            if(velocities_are_defined)
            {
              if(phi_face > 0.0)
              {
                vnp1_plus_p[dir/2][face_idx]  = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
              }
              else
              {
                vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                vnp1_minus_p[dir/2][face_idx] = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
              }
            }
          }
          /* no interface - regular discretization */
          else
          {
            const double surface_direct_neighbor = pow((double)P4EST_QUADRANT_LEN(neighbor_quad.level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1);
            set_of_neighboring_quadrants sharing_quads;
            cell_ngbd->find_neighbor_cells_of_cell(sharing_quads, neighbor_quad.p.piggy3.local_num, neighbor_quad.p.piggy3.which_tree, dir%2 == 0 ? dir + 1 : dir - 1);
            double discretization_distance = 0.0;
#ifdef DEBUG
            double split_face_check = 0.0; bool quad_is_among_sharers = false;
#endif
            double local_flux = 0.0;
            for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
            {
              const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_direct_neighbor;
              discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(neighbor_quad.level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
              local_flux += (phi_face <= 0.0 ? mu_m : mu_p)*(solution_p[neighbor_quad.p.piggy3.local_num] - solution_p[it->p.piggy3.local_num])*surface_ratio;
#ifdef DEBUG
              split_face_check += surface_ratio; quad_is_among_sharers = quad_is_among_sharers || (it->p.piggy3.local_num == quad_idx);
#endif
            }
            P4EST_ASSERT(quad_is_among_sharers && fabs(split_face_check - 1.0) < EPS);
            discretization_distance *= tree_dimensions[dir/2];
            local_flux *= (dir%2 == 1 ? +1.0 : -1.0)/discretization_distance;
            for (set_of_neighboring_quadrants::const_iterator it = sharing_quads.begin(); it != sharing_quads.end(); ++it)
            {
              face_idx = faces->q2f(it->p.piggy3.local_num, dir);
              P4EST_ASSERT(face_idx != NO_VELOCITY);
              if(face_idx < faces->num_local[dir/2] && !visited_faces[dir/2][face_idx])
              {
                flux_p[dir/2][face_idx]         = local_flux;
                visited_faces[dir/2][face_idx]  = true;
                num_visited_faces[dir/2]++;
                if(velocities_are_defined)
                {
                  if(phi_face > 0.0)
                  {
                    vnp1_plus_p[dir/2][face_idx]  = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                    vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
                  }
                  else
                  {
                    vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                    vnp1_minus_p[dir/2][face_idx] = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                  }
                }
              }
            }
          }
        }
        /* There is more than one neighbor, regular bulk case.
         * The current face_idx is (must be) NO_VELOCITY, but all neighbors have a well-defined velocity dof.
         * The flux component is identical for all those neighbor's well-defined face dofs, by construction.
         * --> Update all of them simultaneously
         * We assumes a uniform tesselation on the interface ! */
        else if(direct_neighbors.size() > 1)
        {
          P4EST_ASSERT(face_idx == NO_VELOCITY);
          const double surface_quad = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1); // logical surface

          double discretization_distance = 0.0;
#ifdef DEBUG
          double split_face_check = 0.0;
#endif
          double local_flux = 0.0;
          for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
          {
            const double surface_ratio = pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM - 1)/surface_quad;
            discretization_distance += surface_ratio * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level) + P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;
            local_flux += (phi_face <= 0.0 ? mu_m : mu_p)*(solution_p[it->p.piggy3.local_num] - solution_p[quad_idx])*surface_ratio;
#ifdef DEBUG
            split_face_check += surface_ratio;
#endif
          }
          P4EST_ASSERT(fabs(split_face_check - 1.0) < EPS);
          discretization_distance *= tree_dimensions[dir/2];
          local_flux *= (dir%2 == 1 ? +1.0:-1.0)/discretization_distance;
          for (set_of_neighboring_quadrants::const_iterator it = direct_neighbors.begin(); it != direct_neighbors.end(); ++it)
          {
            face_idx = faces->q2f(it->p.piggy3.local_num, dir + (dir%2 == 1 ? -1: +1));
            P4EST_ASSERT(face_idx != NO_VELOCITY);
            if(face_idx < faces->num_local[dir/2] && !visited_faces[dir/2][face_idx])
            {
              flux_p[dir/2][face_idx]         = local_flux;
              visited_faces[dir/2][face_idx]  = true;
              num_visited_faces[dir/2]++;
              if(velocities_are_defined)
              {
                if(phi_face > 0.0)
                {
                  vnp1_plus_p[dir/2][face_idx]  = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                  vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
                }
                else
                {
                  vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                  vnp1_minus_p[dir/2][face_idx] = vstar_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_flux, &jump_flux_p); CHKERRXX(ierr);
  if(phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(phi_xxyyzz, &phi_xxyyzz_p); CHKERRXX(ierr);}
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    P4EST_ASSERT(num_visited_faces[dim] == faces->num_local[dim]);
    ierr = VecRestoreArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(velocities_are_defined){
      ierr = VecRestoreArrayRead(vstar[dim], &vstar_p[dim]); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
    ierr = VecGhostUpdateEnd  (flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
}

#ifdef DEBUG
int my_p4est_xgfm_cells_t::is_map_consistent() const
{
  int mpiret;

  std::vector<int> senders(p4est->mpisize, 0);
  int num_expected_replies = 0;

  int it_is_alright = true;
  std::map<int, std::vector<which_interface_nb> > map_of_query_interface_neighbors; map_of_query_interface_neighbors.clear();
  std::map<int, std::vector<which_interface_nb> > map_of_mirrors; map_of_mirrors.clear();
  for (std::map<p4est_locidx_t, std::map<u_char, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
       it != map_of_interface_neighbors.end(); ++it)
  {
    p4est_locidx_t quad_idx = it->first;
    for (std::map<u_char, interface_neighbor>::const_iterator itt = (map_of_interface_neighbors.at(quad_idx)).begin();
         itt != (map_of_interface_neighbors.at(quad_idx)).end(); ++itt)
    {
      u_char dir = itt->first;
      // the neighbor is a local quad
      if(((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_nb_idx < p4est->local_num_quadrants)
      {
        p4est_locidx_t loc_quad_idx_tmp = ((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_nb_idx;
        it_is_alright = it_is_alright && ((map_of_interface_neighbors.at(quad_idx)).at(dir)).is_consistent_with_neighbor_across(((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + (dir%2 == 0 ? 1 : -1))));
        if(!it_is_alright && !((map_of_interface_neighbors.at(quad_idx)).at(dir)).is_consistent_with_neighbor_across(((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + (dir%2 == 0 ? 1 : -1)))))
          std::cerr << "quad " << quad_idx << " on proc " << p4est->mpirank << " has quad " << loc_quad_idx_tmp << " as a neighbor on proc " << p4est->mpirank << " across and the interface value is " << ((map_of_interface_neighbors.at(quad_idx)).at(dir)).int_value << " while the value from the other is " << ((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + (dir%2 == 0 ? 1 : -1))).int_value <<  std::endl;
      }
      else
      {
        const p4est_quadrant_t* ghost_nb_quad = p4est_const_quadrant_array_index(&ghost->ghosts, ((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_nb_idx - p4est->local_num_quadrants);
        int rank_owner = p4est_comm_find_owner(p4est, ghost_nb_quad->p.piggy3.which_tree, ghost_nb_quad, p4est->mpirank);
        P4EST_ASSERT(rank_owner != p4est->mpirank);
        num_expected_replies += (senders[rank_owner] == 1 ? 0 : 1);
        senders[rank_owner] = 1;
        which_interface_nb new_query;

        new_query.loc_idx = ghost_nb_quad->p.piggy3.local_num;
        new_query.dir = dir + (dir%2 == 0 ? 1 : -1);
        map_of_query_interface_neighbors[rank_owner].push_back(new_query);
        which_interface_nb this_one;
        this_one.loc_idx  = quad_idx;
        this_one.dir      = dir;
        map_of_mirrors[rank_owner].push_back(this_one);
      }
    }
  }

  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  std::vector<MPI_Request> mpi_query_requests; mpi_query_requests.resize(0);
  std::vector<MPI_Request> mpi_reply_requests; mpi_reply_requests.resize(0);

  // send the requests...
  for (std::map<int, std::vector<which_interface_nb> >::const_iterator it = map_of_query_interface_neighbors.begin();
       it != map_of_query_interface_neighbors.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    int rank = it->first;
    P4EST_ASSERT(senders[rank] == 1);

    MPI_Request req;
    mpiret = MPI_Isend((void*)&map_of_query_interface_neighbors[rank][0], sizeof(which_interface_nb)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    mpi_query_requests.push_back(req);
  }

  std::map<int, std::vector<interface_neighbor> > map_of_responses; map_of_responses.clear();

  MPI_Status status;
  bool done = (num_expected_replies == 0 && num_remaining_queries == 0);
  while (!done) {
    if(num_remaining_queries > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 15351, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(which_interface_nb) == 0);
        std::vector<which_interface_nb> queried_indices(byte_count/sizeof(which_interface_nb));

        mpiret = MPI_Recv((void*) &queried_indices[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 15351, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

        std::vector<interface_neighbor>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(which_interface_nb));

        for (size_t kk = 0; kk < response.size(); ++kk)
          response[kk] = (map_of_interface_neighbors.at(queried_indices[kk].loc_idx)).at(queried_indices[kk].dir);

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*)&response[0], (response.size())*sizeof(interface_neighbor), MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
        mpi_reply_requests.push_back(req);
        num_remaining_queries--;
      }
    }

    if(num_expected_replies > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 42624, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {

        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(interface_neighbor) == 0);
        std::vector<interface_neighbor> reply_buffer (byte_count / sizeof(interface_neighbor));

        mpiret = MPI_Recv((void*)&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        for (size_t kk = 0; kk < reply_buffer.size(); ++kk) {
          which_interface_nb mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          which_interface_nb queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];
          it_is_alright = it_is_alright && ((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).is_consistent_with_neighbor_across(reply_buffer[kk]);
          if(!it_is_alright && !((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).is_consistent_with_neighbor_across(reply_buffer[kk]))
            std::cerr << "quad " << mirror.loc_idx << " on proc " << p4est->mpirank << " has quad " << queried.loc_idx << " as a neighbor on proc " << status.MPI_SOURCE << " across and the interface value is " << ((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).int_value << " while the value from the other is " << reply_buffer[kk].int_value <<  std::endl;
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

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &it_is_alright, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  return it_is_alright;
}
#endif
