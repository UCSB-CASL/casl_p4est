#ifdef P4_TO_P8
#include "my_p8est_poisson_jump_faces_xgfm.h"
#else
#include "my_p4est_poisson_jump_faces_xgfm.h"
#endif

my_p4est_poisson_jump_faces_xgfm_t::my_p4est_poisson_jump_faces_xgfm_t(const my_p4est_faces_t *faces_, const p4est_nodes_t* nodes_)
  : my_p4est_poisson_jump_faces_t(faces_, nodes_), activate_xGFM(true)
{
  xGFM_absolute_accuracy_threshold  = 1e-8;   // default value
  xGFM_tolerance_on_rel_residual    = 1e-12;  // default value

  grad_jump_u_dot_n = NULL;
  interp_grad_jump_u_dot_n = NULL;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    residual[dim] = extension[dim] = NULL;
    xgfm_jump_between_faces[dim].clear();
    pseudo_time_step_increment_operator[dim].resize(faces->num_local[dim]);
    extension_operators_are_stored_and_set[dim] = false;
  }
  print_residuals_and_corrections_with_solve_info = false;
  scale_systems_by_diagonals = false;
  use_face_dofs_only_in_extrapolations = false;

  validation_jump_u = NULL;
  validation_jump_mu_grad_u = NULL;
  interp_validation_jump_u = NULL;
  interp_validation_jump_mu_grad_u = NULL;
  set_for_testing_backbone = false;
}

my_p4est_poisson_jump_faces_xgfm_t::~my_p4est_poisson_jump_faces_xgfm_t()
{
  PetscErrorCode ierr;
  if (grad_jump_u_dot_n != NULL)  { ierr = VecDestroy(grad_jump_u_dot_n); CHKERRXX(ierr); }
  if (interp_grad_jump_u_dot_n != NULL)
    delete interp_grad_jump_u_dot_n;

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if (extension[dim] != NULL)  { ierr = VecDestroy(extension[dim]); CHKERRXX(ierr); }
    if (residual[dim]  != NULL)  { ierr = VecDestroy(residual[dim]);  CHKERRXX(ierr); }
    if (solution[dim]  != NULL)  { ierr = VecDestroy(solution[dim]);  CHKERRXX(ierr); }
  }

  if(interp_validation_jump_u != NULL)
    delete interp_validation_jump_u;
  if(interp_validation_jump_mu_grad_u != NULL)
    delete interp_validation_jump_mu_grad_u;
}

void my_p4est_poisson_jump_faces_xgfm_t::get_numbers_of_faces_involved_in_equation_for_face(const u_char& dir, const p4est_locidx_t& face_idx,
                                                                                            PetscInt& number_of_local_faces_involved, PetscInt& number_of_ghost_faces_involved)
{
  // initialize
  number_of_local_faces_involved = 1; // the local face is always entering the discretization
  number_of_ghost_faces_involved = 0;
  const Voronoi_DIM& voro_cell = get_voronoi_cell(face_idx, dir);
  P4EST_ASSERT(voro_cell.get_type() != unknown);

  if(face_is_dirichlet_wall(face_idx, dir))
    return; // you're done here, value enforced there

  const vector<ngbdDIMseed> *points;
  voro_cell.get_neighbor_seeds(points);

  for(size_t n = 0; n < points->size(); ++n)
    if((*points)[n].n >= 0 && (*points)[n].n != face_idx)
    {
      if((*points)[n].n < faces->num_local[dir])  number_of_local_faces_involved++;
      else                                        number_of_ghost_faces_involved++;
    }

  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::build_discretization_for_face(const u_char& dir, const p4est_locidx_t& face_idx, int *nullspace_contains_constant_vector)
{
  PetscErrorCode ierr;

  const p4est_gloidx_t global_face_idx = faces->global_index(face_idx, dir);
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(face_idx, dir, quad_idx, tree_idx);
#ifdef P4EST_DEBUG
  const u_char face_touch = (faces->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
  P4EST_ASSERT(faces->q2f(quad_idx, face_touch) == face_idx);
#endif

  double *rhs_dir_p = NULL;
  ierr = VecGetArray(rhs[dir], &rhs_dir_p); CHKERRXX(ierr);
  double xyz_face[P4EST_DIM]; faces->xyz_fr_f(face_idx, dir, xyz_face);
  p4est_quadrant_t qm, qp;
  faces->find_quads_touching_face(face_idx, dir, qm, qp);
  const p4est_quadrant_t* quad = (qm.p.piggy3.local_num == quad_idx ? &qm : &qp);
  P4EST_ASSERT(quad->p.piggy3.local_num == quad_idx);
  /* check for walls */
  if((qm.p.piggy3.local_num == -1 || qp.p.piggy3.local_num == -1) && bc[dir].wallType(xyz_face) == DIRICHLET)
  {
    if(nullspace_contains_constant_vector != NULL)
      *nullspace_contains_constant_vector = 0;
    if(!matrix_is_set[dir]){
      ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, 1.0, ADD_VALUES); CHKERRXX(ierr); }
    if(!rhs_is_set[dir])
      rhs_dir_p[face_idx] = bc[dir].wallValue(xyz_face); //  + interp_dxyz_hodge(xyz);
    ierr = VecRestoreArray(rhs[dir], &rhs_dir_p); CHKERRXX(ierr);
    return;
  }

  const double *user_rhs_minus_dir_p    = NULL;
  const double *user_rhs_plus_dir_p     = NULL;
  const double *extension_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  const double *viscous_extrapolation_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  ierr = VecGetArrayRead(user_rhs_minus[dir], &user_rhs_minus_dir_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(user_rhs_plus[dir],  &user_rhs_plus_dir_p); CHKERRXX(ierr);
  Vec* viscous_extrapolation = (extend_negative_interface_values() ? extrapolation_minus : extrapolation_plus);
  for (u_char comp = 0; comp < P4EST_DIM; ++comp)
  {
    if(extension[comp] != NULL){
      ierr = VecGetArrayRead(extension[comp], &extension_p[comp]); CHKERRXX(ierr); }
    if(viscous_extrapolation[comp] != NULL){
      ierr = VecGetArrayRead(viscous_extrapolation[comp], &viscous_extrapolation_p[comp]); CHKERRXX(ierr); }
  }

  bool wall[P4EST_FACES];
  for(u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
  {
    if(oriented_dir/2 == dir)
      wall[oriented_dir]  = (oriented_dir%2 == 0 ? (qm.p.piggy3.local_num == -1) : (qp.p.piggy3.local_num == -1));
    else
      wall[oriented_dir]  = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, oriented_dir)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, oriented_dir));
  }

  const Voronoi_DIM& voro_cell = get_voronoi_cell(face_idx, dir);
  const vector<ngbdDIMseed> *points;
#ifndef P4_TO_P8
  const vector<Point2> *partition;
  voro_cell.get_partition(partition);
#endif
  voro_cell.get_neighbor_seeds(points);

  const double volume = voro_cell.get_volume();
  const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);

  if((sgn_face < 0 ? add_diag_minus : add_diag_plus) > 0.0 && nullspace_contains_constant_vector != NULL)
    *nullspace_contains_constant_vector = 0;
  if(!matrix_is_set[dir]) {
    ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, (sgn_face < 0 ? add_diag_minus : add_diag_plus)*volume, ADD_VALUES); CHKERRXX(ierr);
    // (note: current_diag_m and current_diag_p are both initially set to be 0.0 --> no conflict here)
  }

  if(!rhs_is_set[dir])
    rhs_dir_p[face_idx] = (sgn_face < 0 ? user_rhs_minus_dir_p[face_idx] : user_rhs_plus_dir_p[face_idx])*volume;

  if(voro_cell.get_type() == parallelepiped)
  {
    // some (x)GFM discretization, multiplied by the volume of the parallelipiped cell (*not* cut/clipped by the interface <-> "finite-difference like")
    P4EST_ASSERT((qm.p.piggy3.local_num != -1 && qp.p.piggy3.local_num != -1 && qm.level == qp.level) || ((qm.p.piggy3.local_num == -1) != (qp.p.piggy3.local_num == -1)));
    const double logical_local_size = ((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN);
    const double dxyz_local[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_local_size, tree_dimensions[1]*logical_local_size, tree_dimensions[2]*logical_local_size)};

    // Although intrinsically finite difference in nature for this case, we multiply every "flux contribution" to the local
    // discretized negative laplacian by the area of the face of the associated FV cell, clipped in domain (i.e. regular
    // expected tranverse grid cell area, except in presence of NEUMANN wall faces). The "flux contribution" are evaluated
    // using standard or (x)GFM-like approximations or by wall value in case of NEUMANN wall boundary condition.
    // - We do NOT clip those FV cells by the interface in any way; in other words, the presence of the interface is irrelevant
    // to the definition of those areas.
    // - The right hand side and diagonal contributions are consistently multiplied by the volume of the FV cell (see above).
    // --> This is done in order to ensure symmetry and consistency between off-diagonal weights by comparison with other
    // discretized equations involving nearby faces that do not have any neighbor across the interface
    // (reminder: if no neigbor across, FV on Voronoi cells)
    P4EST_ASSERT(fabs(volume - (wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_local[0], dxyz_local[1], dxyz_local[2])) < 0.001*(wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_local[0], dxyz_local[1], dxyz_local[2])); // half the "regular" volume if the face is a NEUMANN wall face

    for (u_char ff = 0; ff < P4EST_FACES; ++ff)
    {
      // get the face index of the direct face/wall neighbor
#ifndef P4_TO_P8
      const p4est_locidx_t neighbor_face_idx = (*points)[face_order_to_counterclock_cycle_order[ff]].n;
      const size_t k = mod(face_order_to_counterclock_cycle_order[ff] - 1, points->size());
      const double neighbor_area = ((*partition)[face_order_to_counterclock_cycle_order[ff]] - (*partition)[k]).norm_L2();
#else
      const p4est_locidx_t neighbor_face_idx = (*points)[ff].n; // already ordered like this, in 3D
      const double neighbor_area = (*points)[ff].s;
#endif
      P4EST_ASSERT((!wall[ff] && neighbor_face_idx >= 0) || (neighbor_face_idx == WALL_idx(ff)));
      // area between the current face and the direct neighbor
      // this area should be the standard, regular finest uniform-grid area EXCEPT if the considered face is a wall face and we are looking in a tranverse direction
#ifdef P4EST_DEBUG
      const double expected_area = ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir ? 0.5 : 1.0)*dxyz_local[(ff/2 + 1)%P4EST_DIM] ONLY3D(*dxyz_local[(ff/2 + 2)%P4EST_DIM]);
      P4EST_ASSERT(fabs(neighbor_area - expected_area) < EPS*expected_area);
#endif
      double offdiag_coeff;
      // get the contribution of the direct neighbor to the discretization of the negative laplacian and add it to the matrix
      if(neighbor_face_idx >= 0)
      {
        double xyz_neighbor_face[P4EST_DIM]; faces->xyz_fr_f(neighbor_face_idx, dir, xyz_neighbor_face);
        const char sgn_neighbor_face = (interface_manager->phi_at_point(xyz_neighbor_face) <= 0.0 ? -1 : +1);
        if(sgn_neighbor_face != sgn_face)
        {
          P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid());
          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
          const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
          const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
          const bool is_in_positive_domain = (sgn_face > 0);
          offdiag_coeff = -face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)*neighbor_area/dxyz_local[ff/2];
          if(!rhs_is_set[dir])
          {
            if(set_for_testing_backbone)
            {
              double xyz_interface[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
              xyz_interface[ff/2] += (ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.theta*dxyz_local[ff/2];
              if(periodicity[ff/2])
                xyz_interface[ff/2] = xyz_interface[ff/2] - floor((xyz_interface[ff/2] - xyz_min[ff/2])/(xyz_max[ff/2] - xyz_min[ff/2]))*(xyz_max[ff/2] - xyz_min[ff/2]);
              rhs_dir_p[face_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain,
                                                                                                                                       (*interp_validation_jump_u).operator()(xyz_interface, dir),
                                                                                                                                       (*interp_validation_jump_mu_grad_u).operator()(xyz_interface, P4EST_DIM*dir + (ff/2)),
                                                                                                                                       dxyz_local[ff/2]);
            }
            else
            {
              const vector_field_component_xgfm_jump& jump_info = get_xgfm_jump_between_faces(dir, face_idx, neighbor_face_idx, ff);
              rhs_dir_p[face_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain,
                                                                                                                                       jump_info.jump_component,
                                                                                                                                       jump_info.jump_flux_component(extension_p, viscous_extrapolation_p),
                                                                                                                                       dxyz_local[ff/2]);
            }
          }
        }
        else
          offdiag_coeff = -(sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_local[ff/2];

        // check if the neighbor face is a wall (to subtract it from the RHS right away if it is a Dirichlet BC and keep a symmetric matrix for CG)
        // the neigbor face is a wall, if
        // - either we are currently at a non-wall face and we are fetching a neighbor in a face-normal direction that happens to lie on a wall
        // - or we are currently at a wall face (which could happen if the current face is Neumann face) and we are fetching any face neighbor in a tranverse direction
        double xyz_face_neighbor[P4EST_DIM]; faces->xyz_fr_f(neighbor_face_idx, dir, xyz_face_neighbor); // we need it only if it's wall, avoid useless steps...
        const bool neighbor_face_is_dirichlet_wall = ((ff/2 == dir && (ff%2 == 1 ? is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, ff) : is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, ff))) || ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir)) && (bc[dir].wallType(xyz_face_neighbor) == DIRICHLET);
        // this is a regular face so
        if(!matrix_is_set[dir]){
          // we do not add the off-diagonal element to the matrix if the neighbor is a Dirichlet wall face
          // but we modify the rhs correspondingly, right away instead (see if statement here below)
          // --> ensures full symmetry of the matrix, one can use CG solver, safely!
          if(!neighbor_face_is_dirichlet_wall) {
            ierr = MatSetValue(matrix[dir], global_face_idx, faces->global_index(neighbor_face_idx, dir),  offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
          ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr);
        }
        if(!rhs_is_set[dir] && neighbor_face_is_dirichlet_wall)
          rhs_dir_p[face_idx] -= offdiag_coeff*(bc[dir].wallValue(xyz_face_neighbor) /* + interp_dxyz_hodge(xyz_face_neighbor) <- maybe this requires global interpolation /!\ */);
      }
      else
      {
        P4EST_ASSERT(neighbor_face_idx == WALL_idx(ff));
        if(ff/2 == dir) // parallel wall --> the face itself is the wall --> it *cannot* be DIRICHLET
        {
          P4EST_ASSERT(wall[ff] && bc[dir].wallType(xyz_face) == NEUMANN); // the face is a wall face so it MUST be NEUMANN (non-DIRICHLET) boundary condition in that case
          // the contribution of the the local term in the discretization (to the negative laplacian) is
          // -(wall value of NEUMANN boundary condition)*neighbor_area --> goes ot RHS
          if(!rhs_is_set[dir])
            rhs_dir_p[face_idx] += (sgn_face < 0 ? mu_minus  : mu_plus)*neighbor_area*bc[dir].wallValue(xyz_face);
          // no subtraction/addition to diagonal in this, and a nullspace still exists
        }
        else // it is a tranverse wall
        {
          double xyz_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
          xyz_wall[ff/2] = (ff%2 == 1 ? xyz_max[ff/2] : xyz_min[ff/2]);
          const char sgn_wall_neighbor = (interface_manager->phi_at_point(xyz_wall) <= 0 ? -1 : +1);
          const bool across = (sgn_wall_neighbor != sgn_face);
          if(across) // the tranverse wall is across the interface
          {
            P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid());
            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
            // /!\ WARNING /!\ : theta is relative to 0.5*dxyz_min[ff/2] in this case!
            const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
            const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
            const bool is_in_positive_domain = (sgn_face > 0);

            switch (bc[dir].wallType(xyz_wall)) {
            case DIRICHLET:
            {
              offdiag_coeff = -face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)*neighbor_area/(0.5*dxyz_min[ff/2]);
              if(!rhs_is_set[dir])
              {
                if(set_for_testing_backbone)
                {
                  double xyz_interface[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
                  xyz_interface[ff/2] += (ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.theta*0.5*dxyz_local[ff/2]; // can't be out of the domain, by definition in such a case
                  rhs_dir_p[face_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain,
                                                                                                                                           (*interp_validation_jump_u).operator()(xyz_interface, dir),
                                                                                                                                           (*interp_validation_jump_mu_grad_u).operator()(xyz_interface, P4EST_DIM*dir + (ff/2)),
                                                                                                                                           0.5*dxyz_local[ff/2]);
                }
                else
                {
                  const vector_field_component_xgfm_jump& jump_info = get_xgfm_jump_between_faces(dir, face_idx, neighbor_face_idx, ff);
                  rhs_dir_p[face_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain,
                                                                                                                                           jump_info.jump_component,
                                                                                                                                           jump_info.jump_flux_component(extension_p, viscous_extrapolation_p),
                                                                                                                                           0.5*dxyz_local[ff/2]);
                }
              }
              if(!matrix_is_set[dir]) {
                ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
              *nullspace_contains_constant_vector = 0; // no nullspace in this case...
                break;
            }
            case NEUMANN:
            {
              // The tranverse wall has a NEUMANN boundary condition but it is across the interface
              // We will make that an equivalent NEUMANN boundary condition on this_sided field, subtracting/adding the known flux in jump component
              // We take a crude estimation: (wall_value +/- jump_in_flux_component) since jump_in_flux_component is 1st order only (and if we do
              // things right), anyways...
              // Therefore, the contribution of the local term in the discretization (to the negative laplacian) is
              //
              // mu_this_side*(u[f_idx] - "ghost_across_wall")*neighbor_area/dxyz_min[ff/2]
              // = -(wall value of NEUMANN boundary condition + (my_cell.is_in_negative_domain ? -1.0 : +1.0)*(ff%2 == 1 ? +1.0 : -1.0)*jump_flux_component)*neighbor_area
              //
              // therefore we have
              if(!rhs_is_set[dir])
                rhs_dir_p[face_idx] += neighbor_area*((sgn_wall_neighbor < 0 ? mu_minus : mu_plus)*bc[dir].wallValue(xyz_wall) + ((double) sgn_face)*(ff%2 == 1 ? +1.0 : -1.0)*0.0);
              // no subtraction/addition to diagonal in this, and a nullspace still exists
              break;
            }
            default:
              throw std::invalid_argument("my_p4est_poisson_jump_faces_xgfm::build_discretization_for_face: unknown wall type for a tranverse wall neighbor of a face, across the interface --> not handled yet, TO BE DONE if you need it...");
              break;
            }
          }
          else // the tranverse wall is on the same side of the interface
          {
            switch (bc[dir].wallType(xyz_wall)) {
            case DIRICHLET:
              // ghost value = 2*wall_value - center value, so
              //
              // mu_this_side*(u[f_idx] - ghost_value)*neighbor_area/dxyz_min[ff/2]
              // = 2.0*mu_this_side*(u[f_idx] - wall_value)*neighbor_area/dxyz_min[ff/2]
              // --> no nullspace anymore
              offdiag_coeff =  -2.0*(sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_local[ff/2];
              if(!rhs_is_set[dir])
                rhs_dir_p[face_idx] -= offdiag_coeff*bc[dir].wallValue(xyz_wall);
              if(!matrix_is_set[dir]) {
                ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
              if(nullspace_contains_constant_vector != NULL)
                *nullspace_contains_constant_vector = 0; // no nullspace in this case...
              break;
            case NEUMANN:
              // The contribution of the the local term in the discretization (to the negative laplacian) is
              //
              // mu_this_side*(u[f_idx] - "ghost_across_wall")*neighbor_area/dxyz_min[ff/2]
              // = -(wall value of NEUMANN boundary condition)*neighbor_area
              if(!rhs_is_set[dir])
                rhs_dir_p[face_idx] += neighbor_area*(sgn_face < 0 ? mu_minus : mu_plus)*bc[dir].wallValue(xyz_wall);
              // no subtraction/addition to diagonal in this, and a nullspace still exists
              break;
            default:
              throw std::invalid_argument("my_p4est_poisson_jump_faces_xgfm::build_discretization : unknown wall type for a tranverse wall neighbor of a face, on the same side of the interface --> not handled yet, TO BE DONE if you need it...");
              break;
            }
          }
        }
      }
    } /* end of going through uniform neighbors in face order, taking care of jump conditions, where needed */
  }
  else // "regular" stuff --> FV on voronoi cell!
  {
    for(size_t m = 0; m < points->size(); ++m)
    {
#ifdef P4_TO_P8
      const double surface = (*points)[m].s;
#else
      size_t k = mod(m - 1, points->size());
      const double surface = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      const double distance_to_neighbor = ABSD((*points)[m].p.x - xyz_face[0], (*points)[m].p.y - xyz_face[1], (*points)[m].p.z - xyz_face[2]);
      const double mu_this_side = (sgn_face < 0 ? mu_minus : mu_plus);

      switch((*points)[m].n)
      {
      case WALL_m00:
      case WALL_p00:
      case WALL_0m0:
      case WALL_0p0:
#ifdef P4_TO_P8
      case WALL_00m:
      case WALL_00p:
#endif
      {
        char wall_orientation = -1 - (*points)[m].n;
        P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
        double wall_eval[P4EST_DIM];
        const double lambda = ((wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]) - xyz_face[wall_orientation/2])/((*points)[m].p.xyz(wall_orientation/2) - xyz_face[wall_orientation/2]);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
          if(dim == wall_orientation/2)
            wall_eval[dim] = (wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]); // on the wall of interest
          else
            wall_eval[dim] = MIN(MAX(xyz_face[dim] + lambda*((*points)[m].p.xyz(dim) - xyz_face[dim]), xyz_min[dim] + 2.0*EPS*(xyz_max[dim] - xyz_min[dim])), xyz_max[dim] - 2.0*EPS*(xyz_max[dim] - xyz_min[dim])); // make sure it's indeed inside, just to be safe in case the bc object needs that
        }
        const char sgn_wall_neighbor = (interface_manager->phi_at_point(wall_eval) <= 0.0 ? -1 : +1);
        switch(bc[dir].wallType(wall_eval))
        {
        case DIRICHLET:
        {
          if(dir == wall_orientation/2)
            throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm::build_discretization_for_face : Dirichlet boundary conditions on walls parallel to faces should have been done before... You might be using an unconventional aspect ratio for your cells: if yes, it is not taken care of yet, sorry!");
          const bool across = (sgn_wall_neighbor != sgn_face);
          double offdiag_coeff;
          // WARNING distance_to_neighbor is actually *twice* what we would need here, hence the "0.5*" factors here under!
          if(!across)
            offdiag_coeff = -mu_this_side*surface/(0.5*distance_to_neighbor);
          else
          {
            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, wall_orientation);
            const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
            const bool is_in_positive_domain = (sgn_face > 0);
            offdiag_coeff = -surface*face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)/(0.5*distance_to_neighbor);
            if(!rhs_is_set[dir])
            {
              if(set_for_testing_backbone)
              {
                double xyz_interface[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
                xyz_interface[wall_orientation/2] += (wall_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.theta*0.5*distance_to_neighbor; // can't be out of the domain, by definition in such a case
                rhs_dir_p[face_idx] += surface*(wall_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, wall_orientation, is_in_positive_domain,
                                                                                                                                                 (*interp_validation_jump_u).operator()(xyz_interface, dir),
                                                                                                                                                 (*interp_validation_jump_mu_grad_u).operator()(xyz_interface, P4EST_DIM*dir + (wall_orientation/2)),
                                                                                                                                                 0.5*distance_to_neighbor);
              }
              else
              {
                const vector_field_component_xgfm_jump& jump_info = get_xgfm_jump_between_faces(dir, face_idx, (*points)[m].n, wall_orientation);
                rhs_dir_p[face_idx] += surface*(wall_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, wall_orientation, is_in_positive_domain,
                                                                                                                                                 jump_info.jump_component,
                                                                                                                                                 jump_info.jump_flux_component(extension_p, viscous_extrapolation_p),
                                                                                                                                                 0.5*distance_to_neighbor);
              }
            }
          }
          if(nullspace_contains_constant_vector != NULL)
            *nullspace_contains_constant_vector = 0;
          if(!matrix_is_set[dir]) {
            ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); } // needs only to be done if fully reset
          if(!rhs_is_set[dir])
            rhs_dir_p[face_idx] -= offdiag_coeff*(bc[dir].wallValue(wall_eval) /*+ interp_dxyz_hodge(wall_eval)*/);
          break;
        }
        case NEUMANN:
          if(sgn_face != sgn_wall_neighbor)
            throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm::build_discretization_for_face : Neumann boundary condition to be imposed on a tranverse wall that lies across the interface, but the face has non-uniform neighbors : this is not implemented yet, sorry...");
          if(!rhs_is_set[dir])
            rhs_dir_p[face_idx] += mu_this_side*surface*(bc[dir].wallValue(wall_eval) /*+ (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)*/); // apply_hodge_second_derivative_if_neumann: would need to be fixed later --> good luck!
          break;
        default:
          throw std::invalid_argument("my_p4est_poisson_jump_faces_xgfm::build_discretization_for_face : unknown wall type for a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular --> not handled yet, TO BE DONE IF NEEDED.");
        }
        break;
      }
      case INTERFACE:
        throw std::logic_error("my_p4est_poisson_jump_faces_xgfm::build_discretization_for_face : a Voronoi seed neighbor was marked INTERFACE in a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped. This must not happen in this solver, have you constructed your Voronoi cell using clip_interface()?");
        break;
      default:
        // this is a regular face so
        double xyz_face_neighbor[P4EST_DIM]; faces->xyz_fr_f((*points)[m].n, dir, xyz_face_neighbor);
        const char sgn_neighbor = (interface_manager->phi_at_point(xyz_face_neighbor) <= 0.0 ? -1 : +1);
        const bool across = (sgn_face != sgn_neighbor);
        const bool neighbor_face_is_wall = fabs(xyz_face_neighbor[dir] - xyz_max[dir]) < 0.1*dxyz_min[dir] || fabs(xyz_face_neighbor[dir] - xyz_min[dir]) < 0.1*dxyz_min[dir];
        double offdiag_coeff;
        if(!across)
          offdiag_coeff = -mu_this_side*surface/distance_to_neighbor;
        else
        {
          // std::cerr << "This is bad: your grid is messed up here but, hey, I don't want to crash either..." << std::endl;
          char neighbor_orientation = -1;
          for (u_char dim = 0; dim < P4EST_DIM; ++dim)
            if(fabs(xyz_face_neighbor[dim] - xyz_face[dim]) > 0.1*dxyz_min[dim])
              neighbor_orientation = 2*dim + (xyz_face_neighbor[dim] - xyz_face[dim] > 0.0 ? 1 : 0);
          P4EST_ASSERT(fabs(distance_to_neighbor - dxyz_min[neighbor_orientation/2]) < 0.001*dxyz_min[neighbor_orientation/2]);
          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, neighbor_orientation);
          const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
          const bool is_in_positive_domain = (sgn_face > 0);
          offdiag_coeff = surface*face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)/distance_to_neighbor;
          if(!rhs_is_set[dir])
          {
            if(set_for_testing_backbone)
            {
              double xyz_interface[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
              xyz_interface[neighbor_orientation/2] += (neighbor_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.theta*distance_to_neighbor;
              if(periodicity[neighbor_orientation/2])
                xyz_interface[neighbor_orientation/2] = xyz_interface[neighbor_orientation/2] - floor((xyz_interface[neighbor_orientation/2] - xyz_min[neighbor_orientation/2])/(xyz_max[neighbor_orientation/2] - xyz_min[neighbor_orientation/2]))*(xyz_max[neighbor_orientation/2] - xyz_min[neighbor_orientation/2]);
              rhs_dir_p[face_idx] += surface*(neighbor_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, neighbor_orientation, is_in_positive_domain,
                                                                                                                                                   (*interp_validation_jump_u).operator()(xyz_interface, dir),
                                                                                                                                                   (*interp_validation_jump_mu_grad_u).operator()(xyz_interface, P4EST_DIM*dir + (neighbor_orientation/2)),
                                                                                                                                                   0.5*distance_to_neighbor);
            }
            else
            {
              const vector_field_component_xgfm_jump& jump_info = get_xgfm_jump_between_faces(dir, face_idx, (*points)[m].n, neighbor_orientation);
              rhs_dir_p[face_idx] += surface*(neighbor_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, neighbor_orientation, is_in_positive_domain,
                                                                                                                                                   jump_info.jump_component,
                                                                                                                                                   jump_info.jump_flux_component(extension_p, viscous_extrapolation_p),
                                                                                                                                                   distance_to_neighbor);
            }
          }
        }
        if(!matrix_is_set[dir]){
          // we do not add the off-diagonal element to the matrix if the neighbor is a Dirichlet wall face
          // but we modify the rhs correspondingly, right away instead (see if statement here below)
          // --> ensures full symmetry of the matrix, one can use CG solver, safely!
          if(!neighbor_face_is_wall || bc[dir].wallType(xyz_face_neighbor) != DIRICHLET) {
            ierr = MatSetValue(matrix[dir], global_face_idx, faces->global_index((*points)[m].n, dir), offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
          ierr = MatSetValue(matrix[dir], global_face_idx, global_face_idx, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr);
        }
        if(neighbor_face_is_wall && bc[dir].wallType(xyz_face_neighbor) == DIRICHLET)
          rhs_dir_p[face_idx] -= offdiag_coeff*(bc[dir].wallValue(xyz_face_neighbor) /* + interp_dxyz_hodge(xyz_face_neighbor) <- maybe this requires global interpolation /!\ */);
      }
    }
  }

  ierr = VecRestoreArrayRead(user_rhs_minus[dir], &user_rhs_minus_dir_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(user_rhs_plus[dir],  &user_rhs_plus_dir_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs[dir], &rhs_dir_p); CHKERRXX(ierr);
  for (u_char comp = 0; comp < P4EST_DIM; ++comp)
  {
    if(extension[comp] != NULL){
      ierr = VecRestoreArrayRead(extension[comp], &extension_p[comp]); CHKERRXX(ierr); }
    if(viscous_extrapolation[comp] != NULL){
      ierr = VecRestoreArrayRead(viscous_extrapolation[comp], &viscous_extrapolation_p[comp]); CHKERRXX(ierr); }
  }
}

const vector_field_component_xgfm_jump& my_p4est_poisson_jump_faces_xgfm_t::get_xgfm_jump_between_faces(const u_char& dim, const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir)
{
  couple_of_dofs face_couple({face_idx, neighbor_face_idx});
  map_of_vector_field_component_xgfm_jumps_t::const_iterator it = xgfm_jump_between_faces[dim].find(face_couple);
  if(it != xgfm_jump_between_faces[dim].end())
    return it->second;

  // not found in map --> build it and insert in map
  vector_field_component_xgfm_jump to_insert_in_map;
  double xyz_interface_point[P4EST_DIM];
  double normal[P4EST_DIM];
  interface_manager->get_coordinates_of_FD_interface_point_between_faces(dim, face_idx, neighbor_face_idx, oriented_dir, xyz_interface_point);
  interface_manager->normal_vector_at_point(xyz_interface_point, normal);

  const double jump_normal_velocity_at_interface_point = (interp_jump_u_dot_n != NULL ? (*interp_jump_u_dot_n)(xyz_interface_point) : 0.0);

  to_insert_in_map.jump_component   = jump_normal_velocity_at_interface_point*normal[dim];
  to_insert_in_map.known_jump_flux  = 0.0;
  if (interp_jump_tangential_stress != NULL)
  {
    double interface_stress[P4EST_DIM];
    (*interp_jump_tangential_stress)(xyz_interface_point, interface_stress);
    for (u_char dd = 0; dd < P4EST_DIM; ++dd)
      to_insert_in_map.known_jump_flux += ((dim == dd ? 1.0 : 0.0) - normal[dim]*normal[dd])*interface_stress[dd]*normal[oriented_dir/2];
  }
  if(activate_xGFM)
  {
    if(interp_grad_jump_u_dot_n != NULL)
    {
      double local_grad_jump_u_dot_n[P4EST_DIM], local_grad_normal[SQR_P4EST_DIM];
      (*interp_grad_jump_u_dot_n)(xyz_interface_point, local_grad_jump_u_dot_n);
      interface_manager->gradient_of_normal_vector_at_point(xyz_interface_point, local_grad_normal);
      const double local_curvature = interface_manager->curvature_at_point(xyz_interface_point);

      const double& mu_bar = (extend_negative_interface_values() ? mu_plus : mu_minus);
      for (u_char k = 0 ; k < P4EST_DIM; ++k)
      {
        to_insert_in_map.known_jump_flux += mu_bar*((oriented_dir/2 == k ? 1.0 : 0.0) - normal[oriented_dir/2]*normal[k])*(local_grad_jump_u_dot_n[k]*normal[dim] + jump_normal_velocity_at_interface_point*local_grad_normal[P4EST_DIM*dim + k]);
        to_insert_in_map.known_jump_flux -= mu_bar*((dim == k ? 1.0 : 0.0)            - normal[dim]*normal[k]           )*(local_grad_jump_u_dot_n[k]*normal[oriented_dir/2]);
      }
      to_insert_in_map.known_jump_flux -= normal[dim]*normal[oriented_dir/2]*local_curvature*jump_normal_velocity_at_interface_point;
    }

    if(!mus_are_equal())
      build_xgfm_jump_flux_correction_operators_at_point(to_insert_in_map, xyz_interface_point, normal, dim, face_idx, neighbor_face_idx, oriented_dir);
  }

  xgfm_jump_between_faces[dim].insert(std::pair<couple_of_dofs, vector_field_component_xgfm_jump>(face_couple, to_insert_in_map));
  return xgfm_jump_between_faces[dim].at(face_couple);
}

// I hope I did it right here below: if you want more details, check out my very own EquationsAndJumps.pdf document
// long story short: derivations valid only for two-sided solenoidal vector field, mass flux only and no tangential slip
void my_p4est_poisson_jump_faces_xgfm_t::build_xgfm_jump_flux_correction_operators_at_point(vector_field_component_xgfm_jump& xgfm_jump_data,
                                                                                            const double* xyz, const double* normal,
                                                                                            const u_char& dim, const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir) const
{
  p4est_quadrant_t qm, qp;
  p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = P4EST_ROOT_LEN;
  faces->find_quads_touching_face(face_idx, dim, qm, qp);
  set_of_neighboring_quadrants nearby_cell_neighbors;
  if(oriented_dir/2 == dim)
  {
    const p4est_quadrant_t& middle_quad = (oriented_dir%2 == 1 ? qp : qm);
    P4EST_ASSERT(faces->q2f(middle_quad.p.piggy3.local_num, (oriented_dir%2 == 1 ? oriented_dir - 1 : oriented_dir + 1)) == face_idx
                 && faces->q2f(middle_quad.p.piggy3.local_num, oriented_dir) == neighbor_face_idx); (void) neighbor_face_idx; // to avoid compiler's complain
    logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, ngbd_c->gather_neighbor_cells_of_cell(middle_quad, nearby_cell_neighbors));
  }
  else
  {
    set_of_neighboring_quadrants closest_cells;
    if(qm.p.piggy3.local_num != -1)
    {
      closest_cells.insert(qm);
      ngbd_c->find_neighbor_cells_of_cell(closest_cells, qm.p.piggy3.local_num, qm.p.piggy3.which_tree, oriented_dir);
    }
    if(qp.p.piggy3.local_num != -1)
    {
      closest_cells.insert(qp);
      ngbd_c->find_neighbor_cells_of_cell(closest_cells, qp.p.piggy3.local_num, qp.p.piggy3.which_tree, oriented_dir);
    }
    for (set_of_neighboring_quadrants::const_iterator it = closest_cells.begin(); it != closest_cells.end(); ++it)
      logical_size_smallest_first_degree_cell_neighbor = MIN(logical_size_smallest_first_degree_cell_neighbor, ngbd_c->gather_neighbor_cells_of_cell(*it, nearby_cell_neighbors));
  }
  const double scaling_distance = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double) logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;
  std::set<indexed_and_located_face> set_of_neighbor_faces[P4EST_DIM];
  add_all_faces_to_sets_and_clear_set_of_quad(faces, set_of_neighbor_faces, nearby_cell_neighbors);
  const bool grad_is_known[P4EST_DIM] = {DIM(false, false, false)};
  linear_combination_of_dof_t gradient_of_face_sampled_field[P4EST_DIM][P4EST_DIM]; // we neeed the gradient of all vector components <--> stress balance across the interface!
  for (u_char comp = 0; comp < P4EST_DIM; ++comp)
  {
    get_lsqr_face_gradient_at_point(xyz, faces, set_of_neighbor_faces[comp], scaling_distance, gradient_of_face_sampled_field[comp], NULL, grad_is_known);
    xgfm_jump_data.xgfm_jump_flux_tangential_correction[comp].clear();
    xgfm_jump_data.xgfm_jump_flux_normal_correction[comp].clear();
  }

  const double jump_in_mu = get_jump_in_mu();
  const u_char j = oriented_dir/2; // in my EquationsAndJumps.pdf document, i == dim, j == oriented_dir/2
  for (u_char k = 0; k < P4EST_DIM; ++k) {
    xgfm_jump_data.xgfm_jump_flux_tangential_correction[dim].add_operator_on_same_dofs(gradient_of_face_sampled_field[dim][k], jump_in_mu*((j == k ? 1.0 : 0.0) - normal[j]*normal[k]));
    for (u_char r = 0; r < P4EST_DIM; ++r) {
      xgfm_jump_data.xgfm_jump_flux_tangential_correction[k].add_operator_on_same_dofs(gradient_of_face_sampled_field[k][r], -jump_in_mu*normal[j]*((dim == r ? 1.0 : 0.0) - normal[dim]*normal[r])*normal[k]);
      xgfm_jump_data.xgfm_jump_flux_normal_correction[k].add_operator_on_same_dofs(gradient_of_face_sampled_field[k][r], jump_in_mu*normal[r]*normal[k]*normal[dim]*normal[j]);
    }
  }
  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::solve_for_sharp_solution(const KSPType& ksp_type, const PCType& pc_type)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_faces_xgfm_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);
  // make sure the problem is fully defined
  P4EST_ASSERT(bc != NULL || ANDD(periodicity[0], periodicity[1], periodicity[2])); // boundary conditions
  P4EST_ASSERT(diffusion_coefficients_have_been_set() && interface_is_set());       // essential parameters

  // /!\ STEP "-1":
  // clear information that is critial to internal xGFM convergence monitoring (i.e., residuals)
  // and/or to the intrinsical consistency of the coupled system of equations being considered (i.e., extensions and extrapolations)
  // if needed.
  // [NB: even if an initial guess is given, we *cannot* use it as a "zeroth iterate" of the xGFM strategy because
  // it does not necessarily correspond to the solution of the set of discrete equations being considered.
  // --> Indeed, we do not know the hypothetical extensions/extrapolations and former residuals attached with that
  // user-defined guess(es) and we cannot assume that the initial guesses are their own xGFM consistent extrapolations
  // since they would basically be the fixpoint being searched if they were...]
  for(u_char dir = 0; dir < P4EST_DIM; dir++)
  {
    ierr = delete_and_nullify_vector(residual[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extension[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(extrapolation_plus[dir]); CHKERRXX(ierr);
  }

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    /* Set the linear system, the linear solver and solve it */
    setup_linear_system(dir);
    ierr = setup_linear_solver(dir, ksp_type, pc_type); CHKERRXX(ierr);

    if(user_initial_guess_minus != NULL && user_initial_guess_plus != NULL
       && ANDD(user_initial_guess_minus[0] != NULL, user_initial_guess_minus[1] != NULL, user_initial_guess_minus[2] != NULL)
       && ANDD(user_initial_guess_plus[0] != NULL, user_initial_guess_plus[1] != NULL, user_initial_guess_plus[2] != NULL))
      build_solution_with_initial_guess(dir);
  }

  if(!activate_xGFM || mus_are_equal() || set_for_testing_backbone)
  {
    solve_linear_systems();
    const double dummy_input[P4EST_DIM] = {DIM(DBL_MAX, DBL_MAX, DBL_MAX)};
    solver_monitor.log_iteration(dummy_input, this); // we just want to log the number of ksp iterations in this case, DBL_MAX because irrelevant in this case...
  }
  else
  {
    // We will need to memorize former solver's states for the xgfm iterative procedure
    Vec former_rhs[P4EST_DIM], former_solution[P4EST_DIM], former_extension[P4EST_DIM], former_extrapolation_minus[P4EST_DIM], former_extrapolation_plus[P4EST_DIM], former_residual[P4EST_DIM];
    double max_correction[P4EST_DIM];
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      former_rhs[dim] = former_solution[dim] = former_extension[dim] = former_extrapolation_minus[dim] = former_extrapolation_plus[dim] = former_residual[dim] = NULL; // the procedure will adequately determine/create them

    int xGFM_iter = 0;
    while(update_solution(former_solution))
    {
      // fix-point update
      update_extensions_and_extrapolations(former_extension, former_extrapolation_minus, former_extrapolation_plus);
      update_rhs_and_residual(former_rhs, former_residual);
      // linear combination of the last two solver's states to minimize minimize L2 residual:
      set_solver_state_minimizing_L2_norm_of_residual(former_solution, former_extension, former_extrapolation_minus, former_extrapolation_plus,
                                                      former_rhs, former_residual, max_correction);
      solver_monitor.log_iteration(max_correction, this);
      // check if good enough, yet
      if(xGFM_iter >= max_iter ||
         (xGFM_iter > 0 && solver_monitor.reached_convergence_within_desired_bounds(xGFM_absolute_accuracy_threshold, xGFM_tolerance_on_rel_residual)))
        break;
      xGFM_iter++;
    }

    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = delete_and_nullify_vector(former_rhs[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(former_solution[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(former_extension[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(former_extrapolation_minus[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(former_extrapolation_plus[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(former_residual[dim]); CHKERRXX(ierr);
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_xgfm_solve_for_sharp_solution, A, rhs, ksp, 0); CHKERRXX(ierr);

  return;
}

bool my_p4est_poisson_jump_faces_xgfm_t::update_solution(Vec former_solution[P4EST_DIM])
{
  PetscErrorCode ierr;

  // save current solution needed by xgfm iterative procedure
  for (u_char dim = 0; dim < P4EST_DIM; ++dim){
    std::swap(former_solution[dim], solution[dim]); // save solution into "former solution" (for residual minimization thereafter)
    if(former_solution[dim] != NULL) // we had a former solution, we'll use it as an "initial guess" for what comes next
    {
      if(solution[dim] == NULL){
        ierr = VecCreateGhostFaces(p4est, faces, &solution[dim], dim); CHKERRXX(ierr);
      }
      ierr = VecCopyGhost(former_solution[dim], solution[dim]); CHKERRXX(ierr);
    }
  }
  solve_linear_systems();

  PetscInt nksp_iteration[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = KSPGetIterationNumber(ksp[dim], &nksp_iteration[dim]); CHKERRXX(ierr);
  }

  return SUMD(nksp_iteration[0], nksp_iteration[1], nksp_iteration[2]) != 0; // if the ksp solver did at least one iteration, i.e., there was an update
}

void my_p4est_poisson_jump_faces_xgfm_t::update_extensions_and_extrapolations(Vec former_extension[P4EST_DIM], Vec former_extrapolation_minus[P4EST_DIM], Vec former_extrapolation_plus[P4EST_DIM],
                                                                              const double& threshold, const uint& niter_max)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_faces_xgfm_t_update_extensions_and_extrapolations, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(interface_is_set() && threshold > EPS && niter_max > 0);

  const double *solution_p[P4EST_DIM];
  const double *current_extension_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)}; // --> this feeds the jump conditions that were used to define "solution" so it's required to determine the interface-defined values
  const double *current_viscous_extrapolation_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)}; // --> this feeds the jump conditions that were used to define "solution" so it's required to determine the interface-defined values
  Vec extension_n[P4EST_DIM], extension_np1[P4EST_DIM]; // (extensions at pseudo times n and np1)
  Vec extrapolation_minus_n[P4EST_DIM], extrapolation_minus_np1[P4EST_DIM]; // (extrapolations of minus field at pseudo times n and np1)
  Vec extrapolation_plus_n[P4EST_DIM], extrapolation_plus_np1[P4EST_DIM]; // (extrapolations of plus field at pseudo times n and np1)
  Vec normal_derivative_minus_n[P4EST_DIM], normal_derivative_minus_np1[P4EST_DIM]; // (extrapolations of plus field at pseudo times n and np1)
  Vec normal_derivative_plus_n[P4EST_DIM], normal_derivative_plus_np1[P4EST_DIM]; // (extrapolations of plus field at pseudo times n and np1)
  //
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(solution[dim], &solution_p[dim]); CHKERRXX(ierr);
    if(extension[dim] != NULL){
      ierr = VecGetArrayRead(extension[dim], &current_extension_p[dim]); CHKERRXX(ierr);
    }
    if((extend_negative_interface_values() ? extrapolation_minus[dim] : extrapolation_plus[dim]) != NULL){
      ierr = VecGetArrayRead((extend_negative_interface_values() ? extrapolation_minus[dim] : extrapolation_plus[dim]), &current_viscous_extrapolation_p[dim]); CHKERRXX(ierr); }

    ierr = VecCreateGhostFaces(p4est, faces, &extension_n[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &extension_np1[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_minus_n[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_minus_np1[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_plus_n[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_plus_np1[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_minus_n[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_minus_np1[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_plus_n[dim], dim); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &normal_derivative_plus_np1[dim], dim); CHKERRXX(ierr);
  }

  initialize_extensions_and_extrapolations(extension_n, extrapolation_minus_n, extrapolation_plus_n, normal_derivative_minus_n, normal_derivative_plus_n);

  double *normal_derivative_minus_np1_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  double *normal_derivative_plus_np1_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_minus_n_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_plus_n_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  /* EXTRAPOLATE normal derivatives of solution */
  for (uint iter = 0; iter < niter_max; ++iter) {
    // get pointers
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecGetArray(normal_derivative_minus_np1[dim], &normal_derivative_minus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_plus_np1[dim], &normal_derivative_plus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(normal_derivative_minus_n[dim], &normal_derivative_minus_n_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(normal_derivative_plus_n[dim], &normal_derivative_plus_n_p[dim]); CHKERRXX(ierr);
    }

    // local layer faces first
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
        extrapolate_normal_derivatives_local(dim, faces->get_layer_face(dim, k),
                                             normal_derivative_minus_np1_p, normal_derivative_plus_np1_p, normal_derivative_minus_n_p, normal_derivative_plus_n_p);
      // start updates
      ierr = VecGhostUpdateBegin(normal_derivative_minus_np1[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(normal_derivative_plus_np1[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    // local inner faces first
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      for (size_t k = 0; k < faces->get_local_size(dim); ++k)
        extrapolate_normal_derivatives_local(dim, faces->get_local_face(dim, k),
                                             normal_derivative_minus_np1_p, normal_derivative_plus_np1_p, normal_derivative_minus_n_p, normal_derivative_plus_n_p);
      // finish updates
      ierr = VecGhostUpdateEnd(normal_derivative_minus_np1[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(normal_derivative_plus_np1[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    // restore pointers pointers
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecRestoreArray(normal_derivative_minus_np1[dim], &normal_derivative_minus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_derivative_plus_np1[dim], &normal_derivative_plus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(normal_derivative_minus_n[dim], &normal_derivative_minus_n_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(normal_derivative_plus_n[dim], &normal_derivative_plus_n_p[dim]); CHKERRXX(ierr);
    }

    // swap (n) and (n + 1) pseudo time iterates (more efficient to swap pointers than copying large chunks of data)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      std::swap(normal_derivative_minus_n[dim], normal_derivative_minus_np1[dim]);
      std::swap(normal_derivative_plus_n[dim], normal_derivative_plus_np1[dim]);
    }
  }

  // done with the normal derivatives
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    // the "n" ones are the most advanced --> delete the np1 ones
    ierr = VecDestroy(normal_derivative_minus_np1[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(normal_derivative_plus_np1[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normal_derivative_minus_n[dim], &normal_derivative_minus_n_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normal_derivative_plus_n[dim], &normal_derivative_plus_n_p[dim]); CHKERRXX(ierr);
  }

  const double control_band = 3.0*diag_min();
  const bool extend_positive_interface_values = !extend_negative_interface_values();
  double max_increment_in_band[P4EST_DIM] = {DIM(10.0*threshold, 10.0*threshold, 10.0*threshold)};
  uint iter = 0;
  while (ANDD(max_increment_in_band[0] > threshold, max_increment_in_band[1] > threshold, max_increment_in_band[2] > threshold) && iter < niter_max)
  {
    const double *extension_n_p[P4EST_DIM];           double *extension_np1_p[P4EST_DIM];
    const double *extrapolation_minus_n_p[P4EST_DIM]; double *extrapolation_minus_np1_p[P4EST_DIM];
    const double *extrapolation_plus_n_p[P4EST_DIM];  double *extrapolation_plus_np1_p[P4EST_DIM];
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArrayRead(extension_n[dim],            &extension_n_p[dim]);           CHKERRXX(ierr); ierr = VecGetArray(extension_np1[dim],            &extension_np1_p[dim]);           CHKERRXX(ierr);
      ierr = VecGetArrayRead(extrapolation_minus_n[dim],  &extrapolation_minus_n_p[dim]); CHKERRXX(ierr); ierr = VecGetArray(extrapolation_minus_np1[dim],  &extrapolation_minus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(extrapolation_plus_n[dim],   &extrapolation_plus_n_p[dim]);  CHKERRXX(ierr); ierr = VecGetArray(extrapolation_plus_np1[dim],   &extrapolation_plus_np1_p[dim]);  CHKERRXX(ierr);
    }
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      max_increment_in_band[dim] = 0.0; // reset the measure;
      /* Main loop over all local faces */
      for (size_t k = 0; k < faces->get_layer_size(dim); ++k) {
        const p4est_locidx_t face_idx = faces->get_layer_face(dim, k);
        const extension_increment_operator& extension_increment = get_extension_increment_operator_for(dim, face_idx, control_band);
        extension_np1_p[dim][face_idx] = extension_n_p[dim][face_idx]
            + extension_increment(dim, extension_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, extend_positive_interface_values, max_increment_in_band[dim]);
        if(extension_increment.in_positive_domain)
          extrapolation_minus_np1_p[dim][face_idx] = extrapolation_minus_n_p[dim][face_idx] + extension_increment(dim, extrapolation_minus_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, false, max_increment_in_band[dim], normal_derivative_minus_n_p);
        else
          extrapolation_plus_np1_p[dim][face_idx] = extrapolation_plus_n_p[dim][face_idx] + extension_increment(dim, extrapolation_plus_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, true, max_increment_in_band[dim], normal_derivative_plus_n_p);
      }
      ierr = VecGhostUpdateBegin(extension_np1[dim],            INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(extrapolation_minus_np1[dim],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(extrapolation_plus_np1[dim],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for (size_t k = 0; k < faces->get_local_size(dim); ++k) {
        const p4est_locidx_t face_idx = faces->get_local_face(dim, k);
        const extension_increment_operator& extension_increment = get_extension_increment_operator_for(dim, face_idx, control_band);
        extension_np1_p[dim][face_idx] = extension_n_p[dim][face_idx]
            + extension_increment(dim, extension_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, extend_positive_interface_values, max_increment_in_band[dim]);
        if(extension_increment.in_positive_domain)
          extrapolation_minus_np1_p[dim][face_idx] = extrapolation_minus_n_p[dim][face_idx] + extension_increment(dim, extrapolation_minus_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, false, max_increment_in_band[dim], normal_derivative_minus_n_p);
        else
          extrapolation_plus_np1_p[dim][face_idx] = extrapolation_plus_n_p[dim][face_idx] + extension_increment(dim, extrapolation_plus_n_p, solution_p, current_extension_p, current_viscous_extrapolation_p, *this, true, max_increment_in_band[dim], normal_derivative_plus_n_p);
      }
      ierr = VecGhostUpdateEnd(extension_np1[dim],            INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(extrapolation_minus_np1[dim],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(extrapolation_plus_np1[dim],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      if(!extension_operators_are_stored_and_set[dim])
        extension_operators_are_stored_and_set[dim] = true;
    }

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, max_increment_in_band, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArrayRead(extension_n[dim],            &extension_n_p[dim]);           CHKERRXX(ierr); ierr = VecRestoreArray(extension_np1[dim],            &extension_np1_p[dim]);           CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extrapolation_minus_n[dim],  &extrapolation_minus_n_p[dim]); CHKERRXX(ierr); ierr = VecRestoreArray(extrapolation_minus_np1[dim],  &extrapolation_minus_np1_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extrapolation_plus_n[dim],   &extrapolation_plus_n_p[dim]);  CHKERRXX(ierr); ierr = VecRestoreArray(extrapolation_plus_np1[dim],   &extrapolation_plus_np1_p[dim]);  CHKERRXX(ierr);

      // swap the vectors before moving on
      std::swap(extension_n[dim],           extension_np1[dim]);
      std::swap(extrapolation_minus_n[dim], extrapolation_minus_np1[dim]);
      std::swap(extrapolation_plus_n[dim],  extrapolation_plus_np1[dim]);
    }

    iter++;
  }

  // restore pointers and free data
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(solution[dim], &solution_p[dim]); CHKERRXX(ierr);
    if(current_extension_p[dim] != NULL){
      ierr = VecRestoreArrayRead(extension[dim], &current_extension_p[dim]); CHKERRXX(ierr); }
    if(current_viscous_extrapolation_p[dim] != NULL){
      ierr = VecRestoreArrayRead((extend_negative_interface_values() ? extrapolation_minus[dim] : extrapolation_plus[dim]), &current_viscous_extrapolation_p[dim]); CHKERRXX(ierr); }

    ierr = VecRestoreArrayRead(normal_derivative_minus_n[dim], &normal_derivative_minus_n_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normal_derivative_plus_n[dim], &normal_derivative_plus_n_p[dim]); CHKERRXX(ierr);

    // destroy what needs be
    // no longer need the normal derivatives
    ierr = VecDestroy(normal_derivative_minus_n[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(normal_derivative_plus_n[dim]); CHKERRXX(ierr);
    // extension_n is the most advanced in pseudo-time at this point (because of the final "swap" in the loop here above) --> destroy extension_np1
    ierr = VecDestroy(extension_np1[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(extrapolation_minus_np1[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(extrapolation_plus_np1[dim]); CHKERRXX(ierr);

    // avoid memory leak, destroy former_* stuff if needed before making it point to the former extension
    if(former_extension[dim] != NULL){
      ierr = VecDestroy(former_extension[dim]); CHKERRXX(ierr); }
    if(former_extrapolation_minus[dim] != NULL){
      ierr = VecDestroy(former_extrapolation_minus[dim]); CHKERRXX(ierr); }
    if(former_extrapolation_plus[dim] != NULL){
      ierr = VecDestroy(former_extrapolation_plus[dim]); CHKERRXX(ierr); }
    former_extension[dim] = extension[dim];                     extension[dim] = extension_n[dim];
    former_extrapolation_minus[dim] = extrapolation_minus[dim]; extrapolation_minus[dim] = extrapolation_minus_n[dim];
    former_extrapolation_plus[dim] = extrapolation_plus[dim];   extrapolation_plus[dim] = extrapolation_plus_n[dim];
  }

  niter_extrapolations_done = niter_max;
  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_faces_xgfm_t_update_extensions_and_extrapolations, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::update_rhs_and_residual(Vec former_rhs[P4EST_DIM], Vec former_residual[P4EST_DIM])
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_jump_cells_xgfm_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(solution != NULL && rhs != NULL);

  // update the rhs in the faces that are affected by the new jumps in flux components (because of new extension and extrapolations)
  // save the current rhs first
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    std::swap(former_rhs[dim], rhs[dim]);
    // create a new vector if needed
    if(rhs[dim] == NULL){
      ierr = VecCreateNoGhostFaces(p4est, faces, &rhs[dim], dim); CHKERRXX(ierr);
      ierr = VecCopy(former_rhs[dim], rhs[dim]); CHKERRXX(ierr);
    }
    rhs_is_set[dim] = false; // lower this flag in order to update the rhs terms appropriately
  }

  std::set<p4est_locidx_t> already_done[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    already_done[dim].clear();
    for (map_of_vector_field_component_xgfm_jumps_t::const_iterator it = xgfm_jump_between_faces[dim].begin();
         it != xgfm_jump_between_faces[dim].end(); ++it)
    {
      if(it->first.local_dof_idx < faces->num_local[dim] && already_done[dim].find(it->first.local_dof_idx) == already_done[dim].end())
      {
        build_discretization_for_face(dim, it->first.local_dof_idx);
        already_done[dim].insert(it->first.local_dof_idx);
      }
      if(it->first.neighbor_dof_idx < faces->num_local[dim] && already_done[dim].find(it->first.neighbor_dof_idx) == already_done[dim].end())
      {
        build_discretization_for_face(dim, it->first.neighbor_dof_idx);
        already_done[dim].insert(it->first.neighbor_dof_idx);
      }
    }
    rhs_is_set[dim] = true; // rise the flag up again since you're done
  }


  // save the current residuals
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    std::swap(former_residual[dim], residual[dim]);
    get_residual(dim, residual[dim]);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_jump_cells_xgfm_update_rhs_and_residual, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::set_solver_state_minimizing_L2_norm_of_residual(Vec former_solution[P4EST_DIM],
                                                                                         Vec former_extension[P4EST_DIM], Vec former_extrapolation_minus[P4EST_DIM], Vec former_extrapolation_plus[P4EST_DIM],
                                                                                         Vec former_rhs[P4EST_DIM], Vec former_residual[P4EST_DIM],
                                                                                         double max_correction[P4EST_DIM])
{
  P4EST_ASSERT(ANDD(solution[0] != NULL,      solution[1] != NULL,            solution[2] != NULL           )
      && ANDD(extension[0] != NULL,           extension[1] != NULL,           extension[2] != NULL          )
      && ANDD(extrapolation_minus[0] != NULL, extrapolation_minus[1] != NULL, extrapolation_minus[2] != NULL)
      && ANDD(extrapolation_plus[0] != NULL,  extrapolation_plus[1] != NULL,  extrapolation_plus[2] != NULL )
      && ANDD(rhs[0] != NULL,                 rhs[1] != NULL,                 rhs[2] != NULL                )
      && ANDD(residual[0] != NULL,            residual[1] != NULL,            residual[2] != NULL           ));

  if(ANDD(former_residual[0] == NULL, former_residual[1] == NULL, former_residual[2] == NULL))
  {
    P4EST_ASSERT(ANDD(former_extension[0] == NULL,            former_extension[1] == NULL,            former_extension[2] == NULL)
        && ANDD(former_extrapolation_minus[0] == NULL,  former_extrapolation_minus[1] == NULL,  former_extrapolation_minus[2] == NULL)
        && ANDD(former_extrapolation_plus[0] == NULL,   former_extrapolation_plus[1] == NULL,   former_extrapolation_plus[2] == NULL)); // otherwise, something went wrong...
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      max_correction[dim] = DBL_MAX;

    return;
    // if the former residual is not known (0th step of the iterative process), we can't do anything and we need
    // to leave the solver's state as is
    // DBL_MAX returned for max_correction because no actual combination of states happened and we don't want
    // to trigger a false positive convergence termination
  }

  PetscErrorCode ierr;

  PetscReal former_residual_dot_residual, L2_norm_residual;
  former_residual_dot_residual = L2_norm_residual = 0.0;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    PetscReal tmp;
    ierr = VecDot(former_residual[dim], residual[dim], &tmp); CHKERRXX(ierr);
    former_residual_dot_residual += tmp;
    ierr = VecNorm(residual[dim], NORM_2, &tmp); CHKERRXX(ierr);
    L2_norm_residual += SQR(tmp);
  }
  L2_norm_residual = sqrt(L2_norm_residual);
  const double step_size = (SQR(solver_monitor.latest_L2_norm_of_residual()) - former_residual_dot_residual)/(SQR(solver_monitor.latest_L2_norm_of_residual()) - 2.0*former_residual_dot_residual + SQR(L2_norm_residual));

  // doing the state update of relevant internal variable all at once and knowingly avoiding separate Petsc operations that would multiply the number of such loops
  const double *former_rhs_p[P4EST_DIM], *former_extension_p[P4EST_DIM], *former_extrapolation_minus_p[P4EST_DIM], *former_extrapolation_plus_p[P4EST_DIM], *former_solution_p[P4EST_DIM], *former_residual_p[P4EST_DIM];
  double *rhs_p[P4EST_DIM], *extension_p[P4EST_DIM], *extrapolation_minus_p[P4EST_DIM], *extrapolation_plus_p[P4EST_DIM], *solution_p[P4EST_DIM], *residual_p[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(former_rhs[dim], &former_rhs_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(rhs[dim], &rhs_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(former_extension[dim], &former_extension_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(extension[dim], &extension_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(former_extrapolation_minus[dim], &former_extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_minus[dim], &extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(former_extrapolation_plus[dim], &former_extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(extrapolation_plus[dim], &extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(former_solution[dim], &former_solution_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(solution[dim], &solution_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(former_residual[dim], &former_residual_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(residual[dim], &residual_p[dim]); CHKERRXX(ierr);
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    max_correction[dim] = 0.0;
    for (p4est_locidx_t idx = 0; idx < faces->num_local[dim] + faces->num_ghost[dim]; ++idx) {
      if(idx < faces->num_local[dim]) // face-sampled field without ghosts
      {
        max_correction[dim]   = MAX(max_correction[dim], fabs(step_size*(solution_p[dim][idx] - former_solution_p[dim][idx])));
        residual_p[dim][idx]  = (1.0 - step_size)*former_residual_p[dim][idx] + step_size*residual_p[dim][idx];
        rhs_p[dim][idx]       = (1.0 - step_size)*former_rhs_p[dim][idx] + step_size*rhs_p[dim][idx];
      }

      solution_p[dim][idx]    = (1.0 - step_size)*former_solution_p[dim][idx] + step_size*solution_p[dim][idx];
      extension_p[dim][idx]   = (1.0 - step_size)*former_extension_p[dim][idx] + step_size*extension_p[dim][idx];
      extrapolation_minus_p[dim][idx] = (1.0 - step_size)*former_extrapolation_minus_p[dim][idx] + step_size*extrapolation_minus_p[dim][idx];
      extrapolation_plus_p[dim][idx]  = (1.0 - step_size)*former_extrapolation_plus_p[dim][idx] + step_size*extrapolation_plus_p[dim][idx];
    }
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(former_rhs[dim], &former_rhs_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs[dim], &rhs_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(former_extension[dim], &former_extension_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(extension[dim], &extension_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(former_extrapolation_minus[dim], &former_extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(extrapolation_minus[dim], &extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(former_extrapolation_plus[dim], &former_extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(extrapolation_plus[dim], &extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(former_solution[dim], &former_solution_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(solution[dim], &solution_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(former_residual[dim], &former_residual_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(residual[dim], &residual_p[dim]); CHKERRXX(ierr);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, max_correction, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  return;
}

const my_p4est_poisson_jump_faces_xgfm_t::extension_increment_operator&
my_p4est_poisson_jump_faces_xgfm_t::get_extension_increment_operator_for(const u_char& dim, const p4est_locidx_t& face_idx, const double& control_band)
{
  if(extension_operators_are_stored_and_set[dim])
    return pseudo_time_step_increment_operator[dim][face_idx];

  // we'll build the face-sampled interface-value extension operators, here!
  // the extension procedure, can be formally written as
  // pseudo_time_increment = dtau*(A*face_values + B*interface_values)
  // where A and B are appropriate (sparse) matrices: here below, we build the entries of A and B from our knowledge of the regular extrapolation operators.
  P4EST_ASSERT(extrapolation_operators_are_stored_and_set[dim]);
  extension_increment_operator& pseudo_time_step_operator = pseudo_time_step_increment_operator[dim][face_idx];
  pseudo_time_step_operator.clear();
  double xyz_face[P4EST_DIM]; faces->xyz_fr_f(face_idx, dim, xyz_face);
  const double phi_face = interface_manager->phi_at_point(xyz_face);
  const char sgn_face = (phi_face <= 0.0 ? -1 : +1);
  const extrapolation_operator_t& extrapolation_operator = (sgn_face < 0 ? extrapolation_operator_plus[dim][face_idx] : extrapolation_operator_minus[dim][face_idx]);
  pseudo_time_step_operator.face_idx = face_idx;
  pseudo_time_step_operator.in_band = fabs(phi_face) < control_band;
  pseudo_time_step_operator.in_positive_domain = phi_face > 0.0;
  pseudo_time_step_operator.dtau = extrapolation_operator.dtau;
  pseudo_time_step_operator.interface_terms.clear(); pseudo_time_step_operator.regular_terms.clear();
  double diagonal_coefficient = 0.0;

  for (size_t k = 0; k < extrapolation_operator.n_dot_grad.size(); ++k) {
    const p4est_locidx_t neighbor_face_idx = extrapolation_operator.n_dot_grad[k].dof_idx;
    if(neighbor_face_idx == face_idx)
      continue;
    double xyz_neighbor[P4EST_DIM]; faces->xyz_fr_f(neighbor_face_idx, dim, xyz_neighbor);
    const char sgn_neighbor = (interface_manager->phi_at_point(xyz_neighbor) <= 0.0 ? -1 : +1);
    if(sgn_neighbor == sgn_face)
    {
      pseudo_time_step_operator.regular_terms.add_term(extrapolation_operator.n_dot_grad[k].dof_idx, -extrapolation_operator.n_dot_grad[k].weight);
      diagonal_coefficient += extrapolation_operator.n_dot_grad[k].weight;
    }
    else
    {
      // god damn it, this is ugly, but whatever...
      u_char oriented_dir = UCHAR_MAX;
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        const double xyz_diff = (xyz_neighbor[comp] - xyz_face[comp]) - (periodicity[dim] ? floor((xyz_neighbor[comp] - xyz_face[comp])/(xyz_max[comp] - xyz_min[comp]))*(xyz_max[comp] - xyz_min[comp]) : 0.0);
        if(fabs(xyz_diff) > 0.1*dxyz_min[comp])
        {
          P4EST_ASSERT(oriented_dir == UCHAR_MAX); // if not, you're not considering a Cartesian neighbor across and you're F*****
          oriented_dir = 2*comp + (xyz_diff > 0.0 ? 1 : 0);
        }
      }
      const FD_interface_neighbor& interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dim, oriented_dir);
      if(interface_neighbor.theta < EPS) // too close --> force the interface value!
      {
        pseudo_time_step_operator.regular_terms.clear();
        pseudo_time_step_operator.regular_terms.add_term(face_idx, -1.0);
        pseudo_time_step_operator.dtau = 0.0; // we force the interface value if too close --> normal derivatives are irrelevant in that case
        pseudo_time_step_operator.interface_terms.clear();
        pseudo_time_step_operator.interface_terms.push_back({+1.0, neighbor_face_idx, oriented_dir});
        return pseudo_time_step_operator;
      }
      pseudo_time_step_operator.interface_terms.push_back({-extrapolation_operator.n_dot_grad[k].weight/interface_neighbor.theta, neighbor_face_idx, oriented_dir});
      diagonal_coefficient += extrapolation_operator.n_dot_grad[k].weight/interface_neighbor.theta;
      pseudo_time_step_operator.dtau = MIN(pseudo_time_step_operator.dtau, interface_neighbor.theta*dxyz_min[oriented_dir/2]/(double) P4EST_DIM);
    }
  }
  pseudo_time_step_operator.regular_terms.add_term(face_idx, diagonal_coefficient);

  pseudo_time_step_operator.regular_terms *= pseudo_time_step_operator.dtau;
  for (size_t k = 0; k < pseudo_time_step_operator.interface_terms.size(); ++k)
    pseudo_time_step_operator.interface_terms[k].weight *= pseudo_time_step_operator.dtau;

  return pseudo_time_step_operator;
}

void my_p4est_poisson_jump_faces_xgfm_t::initialize_extensions_and_extrapolations(Vec new_extension[P4EST_DIM], Vec new_extrapolation_minus[P4EST_DIM], Vec new_extrapolation_plus[P4EST_DIM],
                                                                                  Vec new_normal_derivative_minus[P4EST_DIM], Vec new_normal_derivative_plus[P4EST_DIM])
{
  P4EST_ASSERT(interface_is_set());
  PetscErrorCode ierr;
  P4EST_ASSERT(ANDD(new_extension[0] != NULL,               new_extension[1] != NULL,               new_extension[2] != NULL)               && VecsAreSetForFaces(new_extension, faces, 1));
  P4EST_ASSERT(ANDD(new_extrapolation_minus[0] != NULL,     new_extrapolation_minus[1] != NULL,     new_extrapolation_minus[2] != NULL)     && VecsAreSetForFaces(new_extrapolation_minus, faces, 1));
  P4EST_ASSERT(ANDD(new_extrapolation_plus[0] != NULL,      new_extrapolation_plus[1] != NULL,      new_extrapolation_plus[2] != NULL)      && VecsAreSetForFaces(new_extrapolation_plus, faces, 1));
  P4EST_ASSERT(ANDD(new_normal_derivative_minus[0] != NULL, new_normal_derivative_minus[1] != NULL, new_normal_derivative_minus[2] != NULL) && VecsAreSetForFaces(new_normal_derivative_minus, faces, 1));
  P4EST_ASSERT(ANDD(new_normal_derivative_plus[0] != NULL,  new_normal_derivative_plus[1] != NULL,  new_normal_derivative_plus[2] != NULL)  && VecsAreSetForFaces(new_normal_derivative_plus, faces, 1));

  double *new_extension_p[P4EST_DIM]            = {DIM(NULL, NULL, NULL)};
  double *new_extrapolation_minus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *new_extrapolation_plus_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  double *new_normal_derivative_minus_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *new_normal_derivative_plus_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  const double* sharp_solution_p[P4EST_DIM]     = {DIM(NULL, NULL, NULL)};

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(new_extension[dim],                &new_extension_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(new_extrapolation_minus[dim],      &new_extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(new_extrapolation_plus[dim],       &new_extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(new_normal_derivative_minus[dim],  &new_normal_derivative_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(new_normal_derivative_plus[dim],   &new_normal_derivative_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    for (size_t k = 0; k < faces->get_layer_size(dim); ++k)
    {
      const p4est_locidx_t f_idx = faces->get_layer_face(dim, k);
      initialize_extrapolation_local(dim, f_idx,
                                     sharp_solution_p, new_extrapolation_minus_p, new_extrapolation_plus_p, new_normal_derivative_minus_p, new_normal_derivative_plus_p, 1, NULL);
      new_extension_p[dim][f_idx] = sharp_solution_p[dim][f_idx];
      if(interp_jump_u_dot_n != NULL) // there is a possibly nonzero jump
      {
        double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dim, xyz_face);
        const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
        if((sgn_face < 0) != extend_negative_interface_values()) // we are actually on the wrong side and there is a nonzero jump
        {
          double local_normal[P4EST_DIM];
          interface_manager->normal_vector_at_point(xyz_face, local_normal);
          new_extension_p[dim][f_idx] += (sgn_face < 0 ? +1.0 : -1.0)*((*interp_jump_u_dot_n)(xyz_face)*local_normal[dim]);
        }
      }
    }
    ierr = VecGhostUpdateBegin(new_extension[dim],                INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(new_extrapolation_minus[dim],      INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(new_extrapolation_plus[dim],       INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(new_normal_derivative_minus[dim],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(new_normal_derivative_plus[dim],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    for (size_t k = 0; k < faces->get_local_size(dim); ++k)
    {
      const p4est_locidx_t f_idx = faces->get_local_face(dim, k);
      initialize_extrapolation_local(dim, f_idx,
                                     sharp_solution_p, new_extrapolation_minus_p, new_extrapolation_plus_p, new_normal_derivative_minus_p, new_normal_derivative_plus_p, 1, NULL);
      new_extension_p[dim][f_idx] = sharp_solution_p[dim][f_idx];
      if(interp_jump_u_dot_n != NULL) // there is a possibly nonzero jump
      {
        double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dim, xyz_face);
        const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
        if((sgn_face < 0) != extend_negative_interface_values()) // we are actually on the wrong side and there is a nonzero jump
        {
          double local_normal[P4EST_DIM];
          interface_manager->normal_vector_at_point(xyz_face, local_normal);
          new_extension_p[dim][f_idx] += (sgn_face < 0 ? +1.0 : -1.0)*((*interp_jump_u_dot_n)(xyz_face)*local_normal[dim]);
        }
      }
    }
    ierr = VecGhostUpdateEnd(new_extension[dim],                INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(new_extrapolation_minus[dim],      INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(new_extrapolation_plus[dim],       INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(new_normal_derivative_minus[dim],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(new_normal_derivative_plus[dim],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    extrapolation_operators_are_stored_and_set[dim] = true; // if they were not known yet, now they are!
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(new_extension[dim],                &new_extension_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_extrapolation_minus[dim],      &new_extrapolation_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_extrapolation_plus[dim],       &new_extrapolation_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_normal_derivative_minus[dim],  &new_normal_derivative_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_normal_derivative_plus[dim],   &new_normal_derivative_plus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
  }

  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::initialize_extrapolation_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                                                        double* extrapolation_minus_p[P4EST_DIM], double* extrapolation_plus_p[P4EST_DIM],
                                                                        double* normal_derivative_of_solution_minus_p[P4EST_DIM], double* normal_derivative_of_solution_plus_p[P4EST_DIM], const u_char& degree,
                                                                        double* sharp_max_component)
{
  double xyz_face[P4EST_DIM];
  faces->xyz_fr_f(face_idx, dim, xyz_face);
  const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
  double oriented_normal[P4EST_DIM]; // to calculate the normal derivative of the solution (in the local subdomain) --> the opposite of that vector is required when extrapolating the solution from across the interface
  interface_manager->normal_vector_at_point(xyz_face, oriented_normal, (sgn_face < 0 ? +1.0 : -1.0));

  if(sgn_face < 0)
  {
    extrapolation_minus_p[dim][face_idx]  = sharp_solution_p[dim][face_idx];
    extrapolation_plus_p[dim][face_idx]   = sharp_solution_p[dim][face_idx] + (interp_jump_u_dot_n == NULL ? 0.0 : oriented_normal[dim]*((*interp_jump_u_dot_n)(xyz_face)));
    if(set_for_testing_backbone)
      extrapolation_plus_p[dim][face_idx] = sharp_solution_p[dim][face_idx] + (*interp_validation_jump_u).operator()(xyz_face, dim);
    if(sharp_max_component != NULL)
      sharp_max_component[0] = MAX(sharp_max_component[0], fabs(sharp_solution_p[dim][face_idx]));
  }
  else
  {
    extrapolation_plus_p[dim][face_idx]   = sharp_solution_p[dim][face_idx];
    extrapolation_minus_p[dim][face_idx]  = sharp_solution_p[dim][face_idx] + (interp_jump_u_dot_n == NULL ? 0.0 : oriented_normal[dim]*((*interp_jump_u_dot_n)(xyz_face))); //  the sign is _in_ the oriented_normal already...
    if(set_for_testing_backbone)
      extrapolation_minus_p[dim][face_idx] = sharp_solution_p[dim][face_idx] - (*interp_validation_jump_u).operator()(xyz_face, dim);
    if(sharp_max_component != NULL)
      sharp_max_component[1] = MAX(sharp_max_component[1], fabs(sharp_solution_p[dim][face_idx]));
  }

  double diagonal_coeff_for_n_dot_grad_this_side = 0.0, diagonal_coeff_for_n_dot_grad_across = 0.0;
  extrapolation_operator_t extrapolation_operator_across, extrapolation_operator_this_side; // ("_this_side" may not be required)

  double n_dot_grad_u = 0.0; // let's evaluate this term on the side of face, set the corresponding value across to 0.0

  PetscErrorCode ierr;
  const double *extension_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *viscous_extrapolation_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  Vec* viscous_exrapolation = (extend_negative_interface_values() ? extrapolation_minus : extrapolation_plus);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(extension[dim] != NULL){
      ierr = VecGetArrayRead(extension[dim], &extension_p[dim]); CHKERRXX(ierr); }
    if(viscous_exrapolation[dim] != NULL){
      ierr = VecGetArrayRead(viscous_exrapolation[dim], &viscous_extrapolation_p[dim]); CHKERRXX(ierr); }
  }

  bool un_is_well_defined = true;
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(face_idx, dim, quad_idx, tree_idx);
  const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est, ghost);
  const u_char touch_dir = (faces->q2f(quad_idx, 2*dim) == face_idx ? 2*dim : 2*dim + 1);
  P4EST_ASSERT(faces->q2f(quad_idx, touch_dir) == face_idx);
  const bool face_is_wall = is_quad_Wall(p4est, tree_idx, quad, touch_dir);

  const double logical_size_of_quad = ((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN);
  const double dxyz_local[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_of_quad, tree_dimensions[1]*logical_size_of_quad, tree_dimensions[2]*logical_size_of_quad)};
  const Voronoi_DIM& voro_cell = get_voronoi_cell(face_idx, dim);
  const vector<ngbdDIMseed> *points;
  voro_cell.get_neighbor_seeds(points);
  if(voro_cell.get_type() == parallelepiped)
  {
    for (u_char der = 0; der < P4EST_DIM; ++der) {
      double sharp_derivative_at_face;
      if(dim == der && face_is_wall)
      {
        const u_char dual_dir = (touch_dir%2 == 1 ? touch_dir - 1 : touch_dir + 1);
        const p4est_locidx_t neighbor_face_in_dual_direction = faces->q2f(quad_idx, dual_dir);
        double xyz_neighbor_face[P4EST_DIM];
        faces->xyz_fr_f(neighbor_face_in_dual_direction, dim, xyz_neighbor_face);
        const char sgn_neighbor_face = (interface_manager->phi_at_point(xyz_neighbor_face) <= 0.0 ? -1 : +1);
        if(sgn_face == sgn_neighbor_face)
          sharp_derivative_at_face = (touch_dir%2 == 1 ? +1.0 : -1.0)*(sharp_solution_p[dim][face_idx] - sharp_solution_p[dim][neighbor_face_in_dual_direction])/dxyz_local[der];
        else
        {
          if(!activate_xGFM || use_face_dofs_only_in_extrapolations)
            un_is_well_defined = false;
          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_in_dual_direction, dim, dual_dir);
          const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
          const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
          const bool in_positive_domain = (sgn_face > 0);

          const couple_of_dofs face_couple({face_idx, neighbor_face_in_dual_direction});
          map_of_vector_field_component_xgfm_jumps_t::const_iterator it = xgfm_jump_between_faces[dim].find(face_couple);
          if(it == xgfm_jump_between_faces[dim].end())
            throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm_t::initialize_extrapolation_local(): found an interface neighbor that was not stored internally by the solver... Have you called solve()?");
          const vector_field_component_xgfm_jump& jump_info = it->second;
          sharp_derivative_at_face = (1.0/mu_this_side)*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, dual_dir, in_positive_domain, sharp_solution_p[dim][face_idx], sharp_solution_p[dim][neighbor_face_in_dual_direction],
                                                                                                   jump_info.jump_component,
                                                                                                   jump_info.jump_flux_component(extension_p, viscous_extrapolation_p), dxyz_local[der]);
        }

        if(!extrapolation_operators_are_stored_and_set[dim])
        {
          const bool derivative_is_relevant_for_extrapolation_of_this_side = (oriented_normal[der] <= 0.0 && dual_dir%2 == 1) || (oriented_normal[der] > 0.0 && dual_dir%2 == 0);
          const double relevant_normal_component = (derivative_is_relevant_for_extrapolation_of_this_side ? +1.0 : -1.0)*oriented_normal[der];
          extrapolation_operator_t& relevant_operator = (derivative_is_relevant_for_extrapolation_of_this_side ? extrapolation_operator_this_side : extrapolation_operator_across);
          double& relevant_diagonal_term = (derivative_is_relevant_for_extrapolation_of_this_side ? diagonal_coeff_for_n_dot_grad_this_side : diagonal_coeff_for_n_dot_grad_across);
          relevant_diagonal_term += relevant_normal_component*(touch_dir%2 == 1 ? +1.0 : -1.0)/dxyz_local[der];
          relevant_operator.n_dot_grad.add_term(neighbor_face_in_dual_direction, -relevant_normal_component*(touch_dir%2 == 1 ? +1.0 : -1.0)/dxyz_local[der]);
          relevant_operator.dtau = MIN(relevant_operator.dtau, dxyz_local[der]/(double) P4EST_DIM);
        }

        n_dot_grad_u += oriented_normal[der]*sharp_derivative_at_face;
        continue;
      }
      else
      {
        double sharp_derivative_m, sharp_derivative_p;
        double dist_m = dxyz_local[der], dist_p = dxyz_local[der]; // relevant only if fetching some interface neighbor/dirichlet wall BC (i.e., to handle subcell resolution)
        for (u_char orientation = 0; orientation < 2; ++orientation) {
          double &oriented_sharp_derivative = (orientation == 1 ? sharp_derivative_p  : sharp_derivative_m);
          double &oriented_dist             = (orientation == 1 ? dist_p              : dist_m);
#ifdef P4_TO_P8
          const p4est_locidx_t neighbor_face_idx = (*points)[2*der + orientation].n; // already ordered like this, in 3D
#else
          const p4est_locidx_t neighbor_face_idx = (*points)[face_order_to_counterclock_cycle_order[2*der + orientation]].n;
#endif
          P4EST_ASSERT(neighbor_face_idx >= 0 || neighbor_face_idx == WALL_idx(2*der + orientation));
          double xyz_neighbor[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])}; xyz_neighbor[der] = (orientation == 1 ? xyz_max[der] : xyz_min[der]); // initialized to wall point
          if(neighbor_face_idx >= 0)
            faces->xyz_fr_f(neighbor_face_idx, dim, xyz_neighbor);
          const char sgn_neighbor = (interface_manager->phi_at_point(xyz_neighbor) <= 0.0 ? -1 : +1);

          if(sgn_neighbor == sgn_face)
          {
            if(neighbor_face_idx >= 0)
              oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*(sharp_solution_p[dim][neighbor_face_idx] - sharp_solution_p[dim][face_idx])/dxyz_local[der];
            else
            {
              if(bc[dim].wallType(xyz_neighbor) == NEUMANN)
                oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*bc[dim].wallValue(xyz_neighbor);
              else
              {
                oriented_dist = 0.5*dxyz_local[der];
                oriented_sharp_derivative = (orientation == 1 ? +1.0 : -1.0)*(bc[dim].wallValue(xyz_neighbor) - sharp_solution_p[dim][face_idx])/oriented_dist;
              }
            }
          }
          else
          {
            if(!activate_xGFM || use_face_dofs_only_in_extrapolations)
              un_is_well_defined = false;
            P4EST_ASSERT(quad->level == interface_manager->get_max_level_computational_grid());
            const double& mu_this_side  = (sgn_face > 0 ? mu_plus   : mu_minus);
            const double& mu_across     = (sgn_face > 0 ? mu_minus  : mu_plus);
            const bool is_in_positive_domain = (sgn_face > 0);
            if(neighbor_face_idx >= 0 || bc[dim].wallType(xyz_neighbor) == DIRICHLET)
            {
              const double solution_across = (neighbor_face_idx >= 0 ? sharp_solution_p[dim][neighbor_face_idx] : bc[dim].wallValue(xyz_neighbor));
              const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dim, 2*der+ orientation);
              // fetch the interface-defined value (on the same side)
              oriented_dist = face_interface_neighbor.theta*(neighbor_face_idx >= 0 ? 1.0 : 0.5)*dxyz_local[dim];

              const couple_of_dofs face_couple({face_idx, neighbor_face_idx});
              map_of_vector_field_component_xgfm_jumps_t::const_iterator it = xgfm_jump_between_faces[dim].find(face_couple);
              if(it == xgfm_jump_between_faces[dim].end())
                throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm_t::initialize_extrapolation_local(): found an interface neighbor that was not stored internally by the solver... Have you called solve()?");
              const vector_field_component_xgfm_jump& jump_info = it->second;
              oriented_sharp_derivative = face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, 2*der + orientation, is_in_positive_domain, sharp_solution_p[dim][face_idx], solution_across,
                                                                                     jump_info.jump_component,
                                                                                     jump_info.jump_flux_component(extension_p, viscous_extrapolation_p),
                                                                                     (neighbor_face_idx >= 0 ? 1.0 : 0.5)*dxyz_local[der])/mu_this_side;
            }
            else // wall neighbor across the interface with Neumann boundary condition --> as for within the solver, take the wall BC and convert it to a Neumann condition on the field on this side
              oriented_sharp_derivative = (mu_across*bc[dim].wallValue(xyz_neighbor) + ((double) sgn_face)*(orientation == 1 ? +1.0 : -1.0)*0.0)/mu_this_side;
          }

          if(neighbor_face_idx >= 0 && !extrapolation_operators_are_stored_and_set[dim]) // if the neighbor is a wall, it is not relevant for any extrapolation operator (homogeneous neumann BC for extrapolation purposes...)
          {
            // add the (regular, i.e. without interface-fetching) derivative term(s) to the relevant extrapolation operator (for extrapolating normal derivatives, for instance)
            const bool derivative_is_relevant_for_extrapolation_of_this_side = (oriented_normal[der] <= 0.0 && orientation == 1) || (oriented_normal[der] > 0.0 && orientation == 0);
            const double relevant_normal_component = (derivative_is_relevant_for_extrapolation_of_this_side ? +1.0 : -1.0)*oriented_normal[der];
            extrapolation_operator_t& relevant_operator = (derivative_is_relevant_for_extrapolation_of_this_side ? extrapolation_operator_this_side : extrapolation_operator_across);
            double& relevant_diagonal_term = (derivative_is_relevant_for_extrapolation_of_this_side ? diagonal_coeff_for_n_dot_grad_this_side : diagonal_coeff_for_n_dot_grad_across);
            relevant_diagonal_term += relevant_normal_component*(orientation == 0 ? +1.0 : -1.0)/dxyz_local[der];
            relevant_operator.n_dot_grad.add_term(neighbor_face_idx, -relevant_normal_component*(orientation == 0 ? +1.0 : -1.0)/dxyz_local[der]);
            relevant_operator.dtau = MIN(relevant_operator.dtau, dxyz_local[der]/(double) P4EST_DIM);
          }
        }

        // the following is equivalent to FD evaluation using the interface-fetched point(s)
        sharp_derivative_at_face = (dist_p*sharp_derivative_m + dist_m*sharp_derivative_p)/(dist_p + dist_m);
        if(dist_m + dist_p < 0.1*pow(2.0, -interface_manager->get_max_level_computational_grid())*dxyz_min[der]) // "0.1" <--> minimum for coarse grid.
          sharp_derivative_at_face = 0.5*(sharp_derivative_m + sharp_derivative_p); // it was an underresolved case, the above operation is too risky...
      }

      n_dot_grad_u += oriented_normal[der]*sharp_derivative_at_face;
    }
    // complete the extrapolation operators
    if(!extrapolation_operators_are_stored_and_set[dim])
    {
      extrapolation_operator_across.n_dot_grad.add_term(face_idx, diagonal_coeff_for_n_dot_grad_across);
      extrapolation_operator_this_side.n_dot_grad.add_term(face_idx, diagonal_coeff_for_n_dot_grad_this_side);
    }
  }
  else
  {
    // must be far in one of the two subdomains, we build only one of the extrapolation operators
    const Voronoi_DIM& voro_cell = get_voronoi_cell(face_idx, dim);
    const vector<ngbdDIMseed>* neighbor_seeds;
    voro_cell.get_neighbor_seeds(neighbor_seeds);

    indexed_and_located_face tmp; tmp.face_idx = face_idx; tmp.field_value = sharp_solution_p[dim][face_idx];
    for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      tmp.xyz_face[comp] = xyz_face[comp];

    std::set<indexed_and_located_face> all_neighbors;     all_neighbors.insert(tmp);
    std::set<indexed_and_located_face> upwind_neighbors;  upwind_neighbors.insert(tmp);
    double min_dist         = DBL_MAX;
    double upwind_min_dist  = DBL_MAX;

    bool grad_component_is_known[P4EST_DIM] = {DIM(false, false, false)}; // to avoid lsqr routines to fail if not enough real neighbors, you need to constrain it with what you may know from wall boundary conditions
    double lsqr_face_gradient[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)}; // that's what we're after for initialization purposes
    bool homogeneous_grad_component_for_extrapolation[P4EST_DIM] = {DIM(false, false, false)}; // for exrapolation purposes, if upwind direction corresponds to a wall for some reason, it will be a homogeneous Neumann boundary condition

    for (size_t k = 0; k < neighbor_seeds->size(); ++k) {
      P4EST_ASSERT((*neighbor_seeds)[k].n != face_idx);
      tmp.face_idx = (*neighbor_seeds)[k].n;
      BoundaryConditionType wall_bc = IGNORE;
      if((*neighbor_seeds)[k].n < 0)
      {
        char wall_orientation = -1 - (*neighbor_seeds)[k].n;
        P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
        const double lambda = ((wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]) - xyz_face[wall_orientation/2])/((*points)[k].p.xyz(wall_orientation/2) - xyz_face[wall_orientation/2]);
        for (u_char comp = 0; comp < P4EST_DIM; ++comp) {
          if(comp == wall_orientation/2)
            tmp.xyz_face[comp] = (wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]); // on the wall of interest
          else
            tmp.xyz_face[comp] = MIN(MAX(xyz_face[comp] + lambda*((*points)[k].p.xyz(comp) - xyz_face[comp]), xyz_min[comp] + 2.0*EPS*(xyz_max[comp] - xyz_min[comp])), xyz_max[comp] - 2.0*EPS*(xyz_max[comp] - xyz_min[comp])); // make sure it's indeed inside, just to be safe in case the bc object needs that
        }
        wall_bc = bc[dim].wallType(tmp.xyz_face);
        if(wall_bc == DIRICHLET)
          tmp.field_value = bc[dim].wallValue(tmp.xyz_face);
        if(wall_bc == NEUMANN)
        {
          if(wall_orientation == touch_dir)
          {
            grad_component_is_known[wall_orientation/2] = true;
            lsqr_face_gradient[wall_orientation/2] = (wall_orientation%2 == 1 ? +1.0 : -1.0)*bc[dim].wallValue(tmp.xyz_face);
          }
          else
            tmp.field_value = sharp_solution_p[dim][face_idx] + bc[dim].wallValue(tmp.xyz_face)*(tmp.xyz_face[wall_orientation/2] - xyz_face[wall_orientation/2]); // build the sample value
        }
      }
      else
      {
        tmp.field_value = sharp_solution_p[dim][tmp.face_idx];
        for (u_char comp = 0; comp < P4EST_DIM; ++comp)
          tmp.xyz_face[comp] = (*neighbor_seeds)[k].p.xyz(comp);
      }

      double inner_product = 0.0;
      double neighbor_distance = 0.0;
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        inner_product += oriented_normal[comp]*((*neighbor_seeds)[k].p.xyz(comp) - xyz_face[comp]);
        neighbor_distance += SQR((*neighbor_seeds)[k].p.xyz(comp) - xyz_face[comp]);
      }
      neighbor_distance = sqrt(neighbor_distance);
      inner_product /= neighbor_distance;
      if(inner_product >= -1.0e-6) // use a small negative threshold instead of strictly 0.0 here
      {
        if(tmp.face_idx >= 0) // regular neighbor, not the same
        {
          upwind_neighbors.insert(tmp);
          upwind_min_dist = MIN(upwind_min_dist, neighbor_distance);
        }
        else
          homogeneous_grad_component_for_extrapolation[(-1 - tmp.face_idx)/2] = true;
      }
      if(tmp.face_idx < 0 && tmp.face_idx == WALL_idx(touch_dir) && wall_bc == DIRICHLET)
        continue; // equivalent to center-seed (although it's a "wall neighbor") --> don't add it to the lsqr stencil, that would lead to singular entries
      all_neighbors.insert(tmp);
      min_dist = MIN(min_dist, neighbor_distance);
    }

    linear_combination_of_dof_t upwind_lsqr_face_grad_operator[P4EST_DIM];
    get_lsqr_face_gradient_at_point(xyz_face, faces, all_neighbors, min_dist, NULL, lsqr_face_gradient, grad_component_is_known, true, face_idx);
    try {
      get_lsqr_face_gradient_at_point(xyz_face, faces, upwind_neighbors, upwind_min_dist, upwind_lsqr_face_grad_operator, NULL, homogeneous_grad_component_for_extrapolation, true, face_idx);
    } catch (std::exception& e) {
      // probably not enough upwind neighbors and/or messed-up normal with respect to grid adaptivity
      // --> consider an extended set of face neighbors and do it again (hopefully, that will work)
      set_of_neighboring_quadrants enlarged_set_of_quad_neighbors;
      p4est_quadrant_t qm, qp;
      faces->find_quads_touching_face(face_idx, dim, qm, qp);
      if(qm.p.piggy3.local_num >= 0)
        ngbd_c->gather_neighbor_cells_of_cell(qm, enlarged_set_of_quad_neighbors, true);
      if(qp.p.piggy3.local_num >= 0)
        ngbd_c->gather_neighbor_cells_of_cell(qp, enlarged_set_of_quad_neighbors, true);

      for (set_of_neighboring_quadrants::const_iterator it = enlarged_set_of_quad_neighbors.begin();
           it != enlarged_set_of_quad_neighbors.end(); ++it) {
        for (u_char touch = 0; touch < 2; ++touch) {
          tmp.face_idx = faces->q2f(it->p.piggy3.local_num, 2*dim + touch);
          if(tmp.face_idx != NO_VELOCITY && (upwind_neighbors.find(tmp) == upwind_neighbors.end()) && (all_neighbors.find(tmp) == all_neighbors.end())) // well-defined face and not considered yet
          {
            faces->xyz_fr_f(tmp.face_idx, dim, tmp.xyz_face);
            if(ORD(periodicity[0], periodicity[1], periodicity[2]))
              for (u_char comp = 0; comp < P4EST_DIM; ++comp)
                if(periodicity[comp])
                {
                  const double pp = (tmp.xyz_face[comp] - xyz_face[comp])/(xyz_max[comp] - xyz_min[comp]);
                  tmp.xyz_face[comp] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[comp] - xyz_min[comp]);
                }

            double inner_product = 0.0;
            double neighbor_distance = 0.0;
            for (u_char comp = 0; comp < P4EST_DIM; ++comp)
            {
              inner_product += oriented_normal[comp]*(tmp.xyz_face[comp] - xyz_face[comp]);
              neighbor_distance += SQR(tmp.xyz_face[comp] - xyz_face[comp]);
            }
            neighbor_distance = sqrt(neighbor_distance);
            inner_product /= neighbor_distance;
            if(inner_product >= -1.0e-6) // use a small negative threshold instead of strictly 0.0 here
            {
              upwind_neighbors.insert(tmp);
              upwind_min_dist = MIN(upwind_min_dist, neighbor_distance);
            }
          }
        }
      }

      try {
        get_lsqr_face_gradient_at_point(xyz_face, faces, upwind_neighbors, upwind_min_dist, upwind_lsqr_face_grad_operator, NULL, homogeneous_grad_component_for_extrapolation, true, face_idx);
      } catch (std::exception& e){
        std::ostringstream os;
        os << "x = " << xyz_face[0] << ", y = " << xyz_face[1] ONLY3D( << ", z = " << xyz_face[2]);
        throw std::runtime_error("my_p4est_poisson_jump_faces_xgfm_t::initialize_extrapolation_local(): \n"
                                 "\tcouldn't construct an upwind face-sampled gradient, even with an extended neighborhood, \n"
                                 "\tfor face located at" + os.str() +". Giving up here!");
        throw e;
      }
    }

    if(!extrapolation_operators_are_stored_and_set[dim])
    {
      extrapolation_operator_across.n_dot_grad.clear();
      extrapolation_operator_across.dtau  = upwind_min_dist/P4EST_DIM;
    }
    for (u_char comp = 0; comp < P4EST_DIM; ++comp)
    {
      n_dot_grad_u += lsqr_face_gradient[comp]*oriented_normal[comp];
      if(!extrapolation_operators_are_stored_and_set[dim])
        extrapolation_operator_across.n_dot_grad.add_operator_on_same_dofs(upwind_lsqr_face_grad_operator[comp], -oriented_normal[comp]);
    }
  }

  if(degree > 0)
  {
    if(sgn_face < 0)
    {
      normal_derivative_of_solution_minus_p[dim][face_idx]  = (un_is_well_defined ? n_dot_grad_u : 0.0); // if not well-defined, will be estimated via extrapolation
      normal_derivative_of_solution_plus_p[dim][face_idx]   = 0.0; // to be calculated later on in actual extrapolation
    }
    else
    {
      normal_derivative_of_solution_minus_p[dim][face_idx]  = 0.0; // to be calculated later on in actual extrapolation
      normal_derivative_of_solution_plus_p[dim][face_idx]   = (un_is_well_defined ? n_dot_grad_u : 0.0); // if not well-defined, will be estimated via extrapolation
    }
  }

  if(!extrapolation_operators_are_stored_and_set[dim])
  {
    if(sgn_face < 0)
      extrapolation_operator_plus[dim][face_idx] = extrapolation_operator_across;
    else
      extrapolation_operator_minus[dim][face_idx] = extrapolation_operator_across;
    if(!un_is_well_defined)
    {
      if(sgn_face < 0)
        extrapolation_operator_minus[dim][face_idx] = extrapolation_operator_this_side;
      else
        extrapolation_operator_plus[dim][face_idx]  = extrapolation_operator_this_side;
    }
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(extension[dim] != NULL){
      ierr = VecRestoreArrayRead(extension[dim], &extension_p[dim]); CHKERRXX(ierr); }
    if(viscous_exrapolation[dim] != NULL){
      ierr = VecRestoreArrayRead(viscous_exrapolation[dim], &viscous_extrapolation_p[dim]); CHKERRXX(ierr); }
  }

  return;
}

void my_p4est_poisson_jump_faces_xgfm_t::extrapolate_solution_local(const u_char& dim, const p4est_locidx_t& face_idx, const double* sharp_solution_p[P4EST_DIM],
                                                                    double* tmp_minus_p[P4EST_DIM], double* tmp_plus_p[P4EST_DIM],
                                                                    const double* extrapolation_minus_p[P4EST_DIM], const double* extrapolation_plus_p[P4EST_DIM],
                                                                    const double* normal_derivative_of_solution_minus_p[P4EST_DIM], const double* normal_derivative_of_solution_plus_p[P4EST_DIM])
{
  tmp_minus_p[dim][face_idx] = extrapolation_minus_p[dim][face_idx];
  tmp_plus_p[dim][face_idx] = extrapolation_plus_p[dim][face_idx];
  double xyz_face[P4EST_DIM];
  faces->xyz_fr_f(face_idx, dim, xyz_face);
  const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);

  double** extrapolation_np1_p        = (sgn_face < 0 ? tmp_plus_p : tmp_minus_p);
  const double** extrapolation_n_p    = (sgn_face < 0 ? extrapolation_plus_p : extrapolation_minus_p);
  const double** normal_derivative_p  = (sgn_face < 0 ? normal_derivative_of_solution_plus_p : normal_derivative_of_solution_minus_p);

  if(activate_xGFM && !use_face_dofs_only_in_extrapolations)
  {
    const extension_increment_operator& xgfm_extension_operator = get_extension_increment_operator_for(dim, face_idx, DBL_MAX);
    const bool fetch_positive_interface_values = (sgn_face < 0);
    const double *extension_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    const double *viscous_extrapolation_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    Vec* viscous_extrapolation = extend_negative_interface_values() ? extrapolation_minus : extrapolation_plus;
    PetscErrorCode ierr;
    for (u_char comp = 0; comp < P4EST_DIM; ++comp) {
      if(extension[comp] != NULL){
        ierr = VecGetArrayRead(extension[comp], &extension_p[comp]); CHKERRXX(ierr); }
      if(viscous_extrapolation[dim] != NULL){
        ierr = VecGetArrayRead(viscous_extrapolation[comp], &viscous_extrapolation_p[comp]); CHKERRXX(ierr); }
    }
    double dummy;
    extrapolation_np1_p[dim][face_idx] = extrapolation_n_p[dim][face_idx] + xgfm_extension_operator(dim, extrapolation_n_p, sharp_solution_p, extension_p, viscous_extrapolation_p, *this, fetch_positive_interface_values, dummy, normal_derivative_p);

    for (u_char comp = 0; comp < P4EST_DIM; ++comp) {
      if(extension[comp] != NULL){
        ierr = VecRestoreArrayRead(extension[comp], &extension_p[comp]); CHKERRXX(ierr); }
      if(viscous_extrapolation[dim] != NULL){
        ierr = VecRestoreArrayRead(viscous_extrapolation[comp], &viscous_extrapolation_p[comp]); CHKERRXX(ierr); }
    }
  }
  else
  {
    const extrapolation_operator_t& extrapolation_operator = (sgn_face < 0 ? extrapolation_operator_plus[dim].at(face_idx) : extrapolation_operator_minus[dim].at(face_idx));
    extrapolation_np1_p[dim][face_idx] -= extrapolation_operator.dtau*(extrapolation_operator.n_dot_grad(extrapolation_n_p[dim]) - (normal_derivative_p[dim] != NULL ? normal_derivative_p[dim][face_idx] : 0.0));
  }

  return;
}

