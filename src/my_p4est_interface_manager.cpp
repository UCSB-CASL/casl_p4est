#include "my_p4est_interface_manager.h"

my_p4est_interface_manager_t::my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const p4est_nodes_t* nodes_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_)
  : faces(faces_), c_ngbd(faces_->get_ngbd_c()), p4est(faces_->get_p4est()), ghost(faces_->get_ghost()),
    nodes(nodes_), dxyz_min(faces_->get_smallest_dxyz()),
    interpolation_node_ngbd(interpolation_node_ngbd_), interp_phi(interpolation_node_ngbd_),
    max_level_p4est(((splitting_criteria_t*) p4est->user_pointer)->max_lvl),
    max_level_interpolation_p4est(((splitting_criteria_t*) interpolation_node_ngbd_->get_p4est()->user_pointer)->max_lvl)
{
#ifdef CASL_THROWS
  if(max_level_interpolation_p4est < max_level_p4est)
    throw std::invalid_argument("my_p4est_interface_manager_t(): you're using UNDER-resolved interpolation tools for capturing the interface. Are you mentally sane? Go see a doctor or check your code...");
#endif
  interp_grad_phi   = NULL;
  interp_phi_xxyyzz = NULL;
  cell_FD_interface_neighbors = new map_of_interface_neighbors_t;
  tmp_FD_interface_neighbor   = new FD_interface_neighbor;
  clear_cell_FD_interface_neighbors();
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    face_FD_interface_neighbors[dim] = new map_of_interface_neighbors_t;
    clear_face_FD_interface_neighbors(dim);
  }
  grad_phi_local              = NULL;
  phi_on_computational_nodes  = NULL;
  use_second_derivative_when_computing_FD_theta = true;
}

my_p4est_interface_manager_t::~my_p4est_interface_manager_t()
{
  if(cell_FD_interface_neighbors != NULL)
    delete cell_FD_interface_neighbors;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(face_FD_interface_neighbors[dim] != NULL)
      delete face_FD_interface_neighbors[dim];

  delete tmp_FD_interface_neighbor;

  if(interp_grad_phi != NULL)
    delete interp_grad_phi;
  if(interp_phi_xxyyzz != NULL)
    delete interp_phi_xxyyzz;
  if(grad_phi_local != NULL){
    PetscErrorCode ierr = VecDestroy(grad_phi_local); CHKERRXX(ierr); }
}

void my_p4est_interface_manager_t::do_not_store_cell_FD_interface_neighbors()
{
  if(cell_FD_interface_neighbors != NULL){
    delete cell_FD_interface_neighbors;
    cell_FD_interface_neighbors = NULL;
  }
  return;
}

void my_p4est_interface_manager_t::do_not_store_face_FD_interface_neighbors()
{
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(face_FD_interface_neighbors[dim] != NULL){
      delete face_FD_interface_neighbors[dim];
      face_FD_interface_neighbors[dim] = NULL;
    }

  return;
}

void my_p4est_interface_manager_t::clear_all_FD_interface_neighbors()
{
  clear_cell_FD_interface_neighbors();
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    clear_face_FD_interface_neighbors(dim);
  return;
}

void my_p4est_interface_manager_t::set_levelset(Vec phi, const interpolation_method& method_interp_phi, Vec phi_xxyyzz, const bool& build_and_set_grad_phi_locally)
{
  P4EST_ASSERT(phi != NULL);
  P4EST_ASSERT(VecIsSetForNodes(phi, interpolation_node_ngbd->get_nodes(), interpolation_node_ngbd->get_p4est()->mpicomm, 1));
  if(phi_xxyyzz != NULL)
  {
    P4EST_ASSERT(VecIsSetForNodes(phi_xxyyzz, interpolation_node_ngbd->get_nodes(), interpolation_node_ngbd->get_p4est()->mpicomm, P4EST_DIM));
    interp_phi.set_input(phi, phi_xxyyzz, method_interp_phi);
    if(interp_phi_xxyyzz == NULL)
      interp_phi_xxyyzz = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
    interp_phi_xxyyzz->set_input(phi_xxyyzz, linear, P4EST_DIM);
  }
  else
    interp_phi.set_input(phi, method_interp_phi);

  if(nodes == interpolation_node_ngbd->get_nodes())
    phi_on_computational_nodes = phi;

  if(build_and_set_grad_phi_locally)
    set_grad_phi();

  return;
}

void my_p4est_interface_manager_t::set_under_resolved_levelset(Vec phi_on_computational_nodes_)
{
  P4EST_ASSERT(phi_on_computational_nodes_ != NULL);
  P4EST_ASSERT(VecIsSetForNodes(phi_on_computational_nodes_, nodes, p4est->mpicomm, 1));

  phi_on_computational_nodes = phi_on_computational_nodes_;
  if(subcell_resolution() == 0 && interp_phi.get_input_fields().size() < 1 &&
     interpolation_node_ngbd->get_p4est() == p4est && interpolation_node_ngbd->get_nodes() == nodes)
    set_levelset(phi_on_computational_nodes_, linear);

  return;
}

void my_p4est_interface_manager_t::build_grad_phi_locally()
{
#ifdef CASL_THROWS
  if(interp_phi.get_input_fields().size() != 1 || interp_phi.get_blocksize_of_input_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::build_grad_phi_locally(): can't determine the gradient of the levelset function if the levelset function wasn't set first...");
#endif

  if(grad_phi_local == NULL){
    PetscErrorCode ierr = VecCreateGhostNodesBlock(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), P4EST_DIM, &grad_phi_local); CHKERRXX(ierr); }
  interpolation_node_ngbd->first_derivatives_central(interp_phi.get_input_fields()[0], grad_phi_local);

  return;
}

void my_p4est_interface_manager_t::set_grad_phi(Vec grad_phi_in)
{
  Vec grad_phi = grad_phi_in;
  if(grad_phi == NULL)
  {
    build_grad_phi_locally();
    grad_phi = grad_phi_local;
  }
  P4EST_ASSERT(grad_phi != NULL && VecIsSetForNodes(grad_phi, interpolation_node_ngbd->get_nodes(), p4est->mpicomm, P4EST_DIM, 1));
  if(interp_grad_phi == NULL)
    interp_grad_phi = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_grad_phi->set_input(grad_phi_local, linear, P4EST_DIM);
  return;
}

const FD_interface_neighbor& my_p4est_interface_manager_t::get_cell_FD_interface_neighbor_for(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants && // must be a local quadrant
               0 <= neighbor_quad_idx && neighbor_quad_idx < p4est->local_num_quadrants + (p4est_locidx_t) ghost->ghosts.elem_count); // must be a known quadrant

  if(cell_FD_interface_neighbors != NULL) // check in map if storing them, first
  {
    map_of_interface_neighbors_t::iterator it = cell_FD_interface_neighbors->find({quad_idx, neighbor_quad_idx});
    if(it != cell_FD_interface_neighbors->end())
    {
      if((it->first.local_dof_idx == neighbor_quad_idx && !it->second.swapped) || (it->first.local_dof_idx == quad_idx && it->second.swapped)) // currently set for the reversed pair --> swap it
      {
        it->second.theta = 1.0 - it->second.theta;
        it->second.swapped = !it->second.swapped;
      }
      return it->second;
    }
  }

  const p4est_topidx_t    tree_idx          = tree_index_of_quad(quad_idx, p4est, ghost);
  const p4est_topidx_t    neighbor_tree_idx = tree_index_of_quad(neighbor_quad_idx, p4est, ghost);
#ifdef P4EST_DEBUG
  // check that they're both as fine as it gets :
  const p4est_tree_t*     tree  = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad  = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const p4est_quadrant_t* neighbor_quad;
  if(neighbor_quad_idx >= p4est->local_num_quadrants)
    neighbor_quad = p4est_const_quadrant_array_index(&ghost->ghosts, neighbor_quad_idx - p4est->local_num_quadrants);
  else
  {
    const p4est_tree_t* tree_neighbor = p4est_tree_array_index(p4est->trees, neighbor_tree_idx);
    neighbor_quad = p4est_const_quadrant_array_index(&tree_neighbor->quadrants, neighbor_quad_idx - tree_neighbor->quadrants_offset);
  }
  P4EST_ASSERT(quad->level == neighbor_quad->level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
#endif

  // compute the finite-difference infamous theta
  double xyz_Q[P4EST_DIM];  quad_xyz_fr_q(quad_idx,           tree_idx,           p4est, ghost, xyz_Q);
  double xyz_N[P4EST_DIM];  quad_xyz_fr_q(neighbor_quad_idx,  neighbor_tree_idx,  p4est, ghost, xyz_N);
  double phi_Q = interp_phi(xyz_Q);
  double phi_N = interp_phi(xyz_N);
  P4EST_ASSERT(signs_of_phi_are_different(phi_Q, phi_N));
  tmp_FD_interface_neighbor->theta  = 0.0;
  double rel_scale = 1.0;
  double xyz_M[P4EST_DIM] = {DIM(xyz_Q[0], xyz_Q[1], xyz_Q[2])};
  for (int k = 0; k < max_level_interpolation_p4est - max_level_p4est; ++k)
  {
    // if using subcell resolution, check intermediate points to have the accurate description
    // The following is equivalent to a dichotomy search, it is reliable so long as we do not several sign changes
    // along the grid line joining the dofs, that is
    //                        this dof                                       neighbor dof
    //    |                                               |                                               |         --> regular (computational grid)
    //    |                                               |           |           |           |           |         --> interface-capturing grid
    // ----------------------------------------------------------++++++++++++++++++++++++++++++++++++++++++++++     --> this is handled correctly
    // -----------------------------------------------+++++++++++++++----------+++++++++++---------------------     --> this is less safe (but your computational grid might be under-resolved as well in such a case)
    rel_scale /= 2.0;
    xyz_M[oriented_dir/2] = xyz_Q[oriented_dir/2] + (oriented_dir%2 == 1 ? +1.0 : -1.0)*rel_scale*dxyz_min[oriented_dir/2]; // no need to worry about periodicity, the interpolation object will
    const double phi_M = interp_phi(xyz_M);
    if(!signs_of_phi_are_different(phi_Q, phi_M))
    {
      tmp_FD_interface_neighbor->theta += rel_scale;
      phi_Q = phi_M;
      xyz_Q[oriented_dir/2] = xyz_M[oriented_dir/2];
    }
    else
    {
      phi_N = phi_M;
      xyz_N[oriented_dir/2] = xyz_M[oriented_dir/2];
    }
  }
  double subscale_theta_negative;
  if(interp_phi_xxyyzz != NULL && use_second_derivative_when_computing_FD_theta)
  {
    const double phi_dd_Q = (*interp_phi_xxyyzz).operator()(xyz_Q, oriented_dir/2);
    const double phi_dd_N = (*interp_phi_xxyyzz)(xyz_N, oriented_dir/2);
    subscale_theta_negative = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_Q, phi_N, phi_dd_Q, phi_dd_N, rel_scale*dxyz_min[oriented_dir/2]);
  }
  else
    subscale_theta_negative = fraction_Interval_Covered_By_Irregular_Domain(phi_Q, phi_N, rel_scale*dxyz_min[oriented_dir/2], rel_scale*dxyz_min[oriented_dir/2]);
  const double to_add = rel_scale*(phi_Q > 0.0 ? 1.0 - subscale_theta_negative : subscale_theta_negative);
  tmp_FD_interface_neighbor->theta += to_add;
  tmp_FD_interface_neighbor->theta = MAX(0.0, MIN(1.0, tmp_FD_interface_neighbor->theta));

  P4EST_ASSERT(0.0 <= tmp_FD_interface_neighbor->theta && tmp_FD_interface_neighbor->theta <= 1.0);
  tmp_FD_interface_neighbor->swapped = false;

  if(cell_FD_interface_neighbors != NULL)
  {
    couple_of_dofs quad_couple = {quad_idx, neighbor_quad_idx};
    std::pair<map_of_interface_neighbors_t::iterator, bool> ret = cell_FD_interface_neighbors->insert({quad_couple, *tmp_FD_interface_neighbor}); // add it to the map so that future access is read from memory;
    P4EST_ASSERT(ret.second);
    return ret.first->second;
  }
  else
    return *tmp_FD_interface_neighbor;
}

void my_p4est_interface_manager_t::get_coordinates_of_FD_interface_point_between_cells(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir, double *xyz) const
{
  const FD_interface_neighbor& interface_point = get_cell_FD_interface_neighbor_for(quad_idx, neighbor_quad_idx, oriented_dir);
  const p4est_topidx_t tree_idx = tree_index_of_quad(quad_idx, p4est, ghost);
  quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz);
  xyz[oriented_dir/2] += (oriented_dir%2 == 1 ? +1.0 : -1.0)*interface_point.theta*dxyz_min[oriented_dir/2];
  if(interpolation_node_ngbd->get_hierarchy()->get_periodicity()[oriented_dir/2]) // do the periodic wrapping if necessary
  {
    const my_p4est_brick_t* brick = c_ngbd->get_brick();
    const double x_min = brick->xyz_min[oriented_dir/2];
    const double x_max = brick->xyz_max[oriented_dir/2];
    xyz[oriented_dir/2] -= floor((xyz[oriented_dir/2] - x_min)/(x_max - x_min))*(x_max - x_min);
  }
  return;
}


void my_p4est_interface_manager_t::compute_subvolumes_in_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const
{
#ifdef CASL_THROWS
  if(quad_idx >= p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_interface_manager_t::compute_subvolumes_in_cell(): cannot be called on ghost cells");
#endif
  const p4est_tree_t     *tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t *quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double* tree_dimensions = c_ngbd->get_tree_dimensions();
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double quad_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

  if(quad->level == (int8_t) max_level_p4est)
  {
    std::vector<p4est_locidx_t> indices_of_subrefining_quads;
    interpolation_node_ngbd->get_hierarchy()->get_all_quadrants_in(quad, tree_idx, indices_of_subrefining_quads);
    if(indices_of_subrefining_quads.size() > 0) // if it is zero, it means that it was not even matched by the hierarchy --> it must be far away from the interface in that case...
    {
      negative_volume = 0.0;
      const p4est_tree_t* interface_capturing_tree = p4est_tree_array_index(interpolation_node_ngbd->get_p4est()->trees, tree_idx);
      for (size_t k = 0; k < indices_of_subrefining_quads.size(); ++k)
      {
        const p4est_quadrant_t *subrefining_quad = p4est_const_quadrant_array_index(&interface_capturing_tree->quadrants, indices_of_subrefining_quads[k] - interface_capturing_tree->quadrants_offset);
        negative_volume += area_in_negative_domain_in_one_quadrant(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), subrefining_quad, indices_of_subrefining_quads[k], interp_phi.get_input_fields()[0]);
      }

      P4EST_ASSERT(0.0 <= negative_volume && negative_volume <= quad_volume);
      positive_volume = MAX(0.0, MIN(quad_volume, quad_volume - negative_volume));
      return;
    }
  }

  // If it reaches this point, it's either a coarser quad or it is not matched by the interface-capturing grid
  // --> must be far away from the interface : just sample the levelset at the quadrant's center

  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const double phi_q = interp_phi(xyz_quad);
#ifdef P4EST_DEBUG
  // we check that the cell is indeed not crossed by the interface in DEBUG
  const double* phi_on_computational_nodes_p = NULL;
  if(phi_on_computational_nodes != NULL){
    PetscErrorCode ierr = VecGetArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }
  for (char kx = 0; kx < 2; ++kx)
    for (char ky = 0; ky < 2; ++ky)
#ifdef P4_TO_P8
      for (char kz = 0; kz < 2; ++kz)
#endif
      {
        if(phi_on_computational_nodes_p == NULL)
          P4EST_ASSERT(!signs_of_phi_are_different(phi_q, phi_on_computational_nodes_p[P4EST_CHILDREN*quad_idx + SUMD(kx, 2*ky, 4*kz)])); // otherwise you are not using finest cells across your interface...
        else
        {
          double xyz_vertex[P4EST_DIM] = {DIM(xyz_quad[0] + (kx - 0.5)*cell_dxyz[0], xyz_quad[1] + (ky - 0.5)*cell_dxyz[1], xyz_quad[2] + (kz - 0.5)*cell_dxyz[2])};
          P4EST_ASSERT(!signs_of_phi_are_different(phi_q, interp_phi(xyz_vertex))); // otherwise you are not using finest cells across your interface...
        }
      }
  if(phi_on_computational_nodes_p != NULL){
    PetscErrorCode ierr = VecRestoreArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }
#endif

  negative_volume = (phi_q <= 0.0 ? quad_volume : 0.0);
  positive_volume = (phi_q <= 0.0 ? 0.0         : quad_volume);

  return;
}

void my_p4est_interface_manager_t::detect_mls_interface_in_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                                                bool &intersection_found, bool which_face_is_intersected[P4EST_FACES]) const
{
  const double *phi_on_computational_nodes_p = NULL;
  if(phi_on_computational_nodes != NULL){
    PetscErrorCode ierr = VecGetArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }

  const double *tree_xyz_min, *tree_xyz_max;
  const p4est_quadrant_t* quad;
  fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
  double xyz_mmm[P4EST_DIM]; xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_mmm);
  const double dxyz_quad[P4EST_DIM] = {DIM((tree_xyz_max[0] - tree_xyz_min[0])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN),
                                       (tree_xyz_max[1] - tree_xyz_min[1])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN),
                                       (tree_xyz_max[2] - tree_xyz_min[2])*((double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN))};
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    xyz_mmm[dim] -= 0.5*dxyz_quad[dim];
  const double xyz_ppp[P4EST_DIM] = {DIM(xyz_mmm[0] + dxyz_quad[0], xyz_mmm[1] + dxyz_quad[1], xyz_mmm[2] + dxyz_quad[2])};

  const double phi_mmm = (phi_on_computational_nodes_p != NULL ? phi_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx                      ]] : interp_phi(xyz_mmm));
  const double phi_ppp = (phi_on_computational_nodes_p != NULL ? phi_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + P4EST_CHILDREN - 1 ]] : interp_phi(xyz_ppp));
  intersection_found        = signs_of_phi_are_different(phi_mmm, phi_ppp);
  if(which_face_is_intersected != NULL)
    for (u_char face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
      which_face_is_intersected[face_dir] = false;

  // check the vertices first
  for (char kx = 0; kx < 2; ++kx)
    for (char ky = 0; ky < 2; ++ky)
#ifdef P4_TO_P8
      for (char kz = 0; kz < 2; ++kz)
#endif
      {
        if(ANDD(kx == 0, ky == 0, kz == 0) || ANDD(kx == 1, ky == 1, kz == 1)) // mmm and ppp vertices
          continue;
        const double phi_vertex = (phi_on_computational_nodes_p != NULL ? phi_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + SUMD(kx, 2*ky, 4*kz)]] :
                                   interp_phi(DIM(xyz_mmm[0] + kx*dxyz_quad[0], xyz_mmm[1] + ky*dxyz_quad[1], xyz_mmm[2] + kz*dxyz_quad[2])));
        intersection_found = intersection_found || signs_of_phi_are_different(phi_mmm, phi_vertex);

        if(which_face_is_intersected != NULL)
        {
          which_face_is_intersected[2*dir::x + kx] = which_face_is_intersected[2*dir::x + kx] || signs_of_phi_are_different((kx == 0 ? phi_mmm : phi_ppp), phi_vertex);
          which_face_is_intersected[2*dir::y + ky] = which_face_is_intersected[2*dir::y + ky] || signs_of_phi_are_different((ky == 0 ? phi_mmm : phi_ppp), phi_vertex);
#ifdef P4_TO_P8
          which_face_is_intersected[2*dir::z + kz] = which_face_is_intersected[2*dir::z + kz] || signs_of_phi_are_different((kz == 0 ? phi_mmm : phi_ppp), phi_vertex);
#endif
        }
      }
  if(phi_on_computational_nodes_p != NULL){
    PetscErrorCode ierr = VecRestoreArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }

  const u_int n_mls_points_per_dimension = (1 << subcell_resolution())*interpolation_degree() + 1;
  if(n_mls_points_per_dimension > 2) // more points to check
  {
    const double dxyz_mls[P4EST_DIM]  = {DIM(dxyz_quad[0]/(double) (n_mls_points_per_dimension - 1), dxyz_quad[1]/(double) (n_mls_points_per_dimension - 1), dxyz_quad[2]/(double) (n_mls_points_per_dimension - 1))};

    for (u_int kx = 0; kx < n_mls_points_per_dimension; ++kx)
      for (u_int ky = 0; ky < n_mls_points_per_dimension; ++ky)
  #ifdef P4_TO_P8
        for (u_int kz = 0; kz < n_mls_points_per_dimension; ++kz)
  #endif
        {
          if(ANDD(kx == 0 || kx == n_mls_points_per_dimension - 1, ky == 0 || ky == n_mls_points_per_dimension - 1, kz == 0 || kz == n_mls_points_per_dimension - 1)) // that's a vertex, already done
            continue;
          const double xyz_mls_point[P4EST_DIM] = {DIM(xyz_mmm[0] + ((double) kx)*dxyz_mls[0], xyz_mmm[1] + ((double) ky)*dxyz_mls[1], xyz_mmm[2] + ((double) kz)*dxyz_mls[2])};
          const double phi_mls_point = interp_phi(xyz_mls_point);

          intersection_found = intersection_found || signs_of_phi_are_different(phi_mmm, phi_mls_point);
          if(which_face_is_intersected != NULL && ORD(kx == 0 || kx == n_mls_points_per_dimension - 1, ky == 0 || ky == n_mls_points_per_dimension - 1, kz == 0 || kz == n_mls_points_per_dimension - 1)) // mls point on face
          {
            if(kx == 0 || kx == n_mls_points_per_dimension - 1)
              which_face_is_intersected[2*dir::x + (kx == 0 ? 0 : 1)] = which_face_is_intersected[2*dir::x + (kx == 0 ? 0 : 1)] || signs_of_phi_are_different((kx == 0 ? phi_mmm : phi_ppp), phi_mls_point);
            if(ky == 0 || ky == n_mls_points_per_dimension - 1)
              which_face_is_intersected[2*dir::y + (ky == 0 ? 0 : 1)] = which_face_is_intersected[2*dir::y + (ky == 0 ? 0 : 1)] || signs_of_phi_are_different((ky == 0 ? phi_mmm : phi_ppp), phi_mls_point);
#ifdef P4_TO_P8
            if(kz == 0 || kz == n_mls_points_per_dimension - 1)
              which_face_is_intersected[2*dir::z + (kz == 0 ? 0 : 1)] = which_face_is_intersected[2*dir::z + (kz == 0 ? 0 : 1)] || signs_of_phi_are_different((kz == 0 ? phi_mmm : phi_ppp), phi_mls_point);
#endif
          }
        }
  }
  return;
}

bool my_p4est_interface_manager_t::is_quad_crossed_by_interface(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, bool which_face_is_intersected[P4EST_FACES]) const
{
  const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est, ghost);

  if(quad->level < (int8_t) max_level_p4est)
  {
#ifdef CASL_THROWS
    bool intersection_found;
    detect_mls_interface_in_quad(quad_idx, tree_idx, intersection_found);
    if(intersection_found)
      throw std::logic_error("my_p4est_interface_manager_t::is_quad_crossed_by_interface() : found a cell bigger than expected but containing an interface intersection.");
#endif
    if(which_face_is_intersected != NULL)
      for (u_char oriented_dir = 0; oriented_dir < P4EST_FACES; ++oriented_dir)
        which_face_is_intersected[oriented_dir] = false;
    return false;
  }

  bool intersection_found;
  detect_mls_interface_in_quad(quad_idx, tree_idx, intersection_found, which_face_is_intersected);

  return intersection_found;
}

my_p4est_finite_volume_t my_p4est_interface_manager_t::get_finite_volume_for_quad(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx) const
{
  my_p4est_finite_volume_t fv_to_build;
  construct_finite_volume(fv_to_build, quad_idx, tree_idx, p4est, &interp_phi, interpolation_degree(), subcell_resolution());
  return fv_to_build;
}

#ifdef DEBUG
int my_p4est_interface_manager_t::cell_FD_map_is_consistent_across_procs()
{
  P4EST_ASSERT(cell_FD_interface_neighbors != NULL);

  int mpiret;
  std::vector<int> senders(p4est->mpisize, 0);
  int num_expected_replies = 0;

  int it_is_alright = true;
  std::map<int, std::vector<couple_of_dofs> > map_of_query_interface_neighbors; map_of_query_interface_neighbors.clear();
  std::map<int, std::vector<couple_of_dofs> > map_of_mirrors; map_of_mirrors.clear();
  int first_rank_ghost_owner = 0;
  while (first_rank_ghost_owner < p4est->mpisize && ghost->proc_offsets[first_rank_ghost_owner + 1] == 0) { first_rank_ghost_owner++; }

  for (map_of_interface_neighbors_t::const_iterator it = cell_FD_interface_neighbors->begin(); it != cell_FD_interface_neighbors->end(); ++it)
  {
    const p4est_locidx_t local_quad_idx = MIN(it->first.local_dof_idx, it->first.neighbor_dof_idx);
    const p4est_locidx_t ghost_quad_idx = MAX(it->first.local_dof_idx, it->first.neighbor_dof_idx);
    P4EST_ASSERT(local_quad_idx != ghost_quad_idx && local_quad_idx < p4est->local_num_quadrants);
    if(ghost_quad_idx >= p4est->local_num_quadrants) // check if it is indeed a ghost, we have inherent consistency for local data, by nature...
    {
      const p4est_quadrant_t* ghost_nb_quad = p4est_const_quadrant_array_index(&ghost->ghosts, ghost_quad_idx - p4est->local_num_quadrants);
      int rank_owner = first_rank_ghost_owner;
      while (ghost_quad_idx >= p4est->local_num_quadrants + ghost->proc_offsets[rank_owner + 1]) { rank_owner++; }
      P4EST_ASSERT(rank_owner != p4est->mpirank);
      if(senders[rank_owner] == 0)
      {
        num_expected_replies += 1;
        senders[rank_owner] = 1;
      }
      couple_of_dofs remote_quad_couple;
      remote_quad_couple.local_dof_idx    = ghost_nb_quad->p.piggy3.local_num;
      remote_quad_couple.neighbor_dof_idx = local_quad_idx; // will be overwritten after communication...
      map_of_query_interface_neighbors[rank_owner].push_back(remote_quad_couple);
      map_of_mirrors[rank_owner].push_back({local_quad_idx, ghost_quad_idx});
    }
  }

  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  std::vector<MPI_Request> mpi_query_requests; mpi_query_requests.resize(0);
  std::vector<MPI_Request> mpi_reply_requests; mpi_reply_requests.resize(0);

  // send the requests...
  for (std::map<int, std::vector<couple_of_dofs> >::const_iterator it = map_of_query_interface_neighbors.begin();
       it != map_of_query_interface_neighbors.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    int rank = it->first;
    P4EST_ASSERT(senders[rank] == 1);

    MPI_Request req;
    mpiret = MPI_Isend((void*) map_of_query_interface_neighbors[rank].data(), sizeof(couple_of_dofs)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    mpi_query_requests.push_back(req);
  }

  std::map<int, std::vector<double> > map_of_responses; map_of_responses.clear();

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
        P4EST_ASSERT(byte_count%sizeof(couple_of_dofs) == 0);
        std::vector<couple_of_dofs> queries(byte_count/sizeof(couple_of_dofs));

        mpiret = MPI_Recv((void*) queries.data(), byte_count, MPI_BYTE, status.MPI_SOURCE, 15351, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

        std::vector<double>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(couple_of_dofs));

        for (size_t kk = 0; kk < response.size(); ++kk)
        {
          for (p4est_locidx_t q = ghost->proc_offsets[status.MPI_SOURCE]; q < ghost->proc_offsets[status.MPI_SOURCE + 1]; ++q)
          {
            const p4est_quadrant_t* ghost_quad = p4est_const_quadrant_array_index(&ghost->ghosts, q);
            if(ghost_quad->p.piggy3.local_num == queries[kk].neighbor_dof_idx)
            {
              queries[kk].neighbor_dof_idx = q + p4est->local_num_quadrants; // overwrite it to match local indexing
              break;
            }
          }
          if(cell_FD_interface_neighbors->find(queries[kk]) == cell_FD_interface_neighbors->end())
          {
            std::cerr << "Queried FD cell interface data not found locally : local index = " << queries[kk].local_dof_idx << ", neighbor index " << queries[kk].neighbor_dof_idx << " on proc " << p4est->mpirank << ", queried from proc " << status.MPI_SOURCE << std::endl;
            it_is_alright = false;
          }
          else
            response[kk] = cell_FD_interface_neighbors->at(queries[kk]).theta;
        }

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*) response.data(), (response.size())*sizeof(double), MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
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
        P4EST_ASSERT(byte_count%sizeof(double) == 0);
        std::vector<double> reply_buffer (byte_count / sizeof(double));

        mpiret = MPI_Recv((void*)&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        for (size_t kk = 0; kk < reply_buffer.size(); ++kk) {
          const couple_of_dofs& mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          const couple_of_dofs& queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];

          const bool consistent_with_other_proc_data = fabs(cell_FD_interface_neighbors->at(mirror).theta + reply_buffer[kk] - 1.0) < EPS;
          it_is_alright = it_is_alright && consistent_with_other_proc_data;
          reply_buffer[kk];

          if(!consistent_with_other_proc_data)
            std::cerr << "Inconsistency found for quad " << mirror.local_dof_idx << " on proc " << p4est->mpirank << " which has quad " << queried.local_dof_idx << " as a neighbor across the interface, on proc " << status.MPI_SOURCE << std::endl;
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
