#ifdef P4_TO_P8
#include "my_p8est_interface_manager.h"
#else
#include "my_p4est_interface_manager.h"
#endif

my_p4est_interface_manager_t::my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const p4est_nodes_t* nodes_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_)
  : faces(faces_), c_ngbd(faces_->get_ngbd_c()), p4est(faces_->get_p4est()), ghost(faces_->get_ghost()),
    nodes(nodes_), dxyz_min(faces_->get_smallest_dxyz()),
    interpolation_node_ngbd(interpolation_node_ngbd_), interp_phi(interpolation_node_ngbd_),
    max_level_p4est(((splitting_criteria_t*) faces_->get_p4est()->user_pointer)->max_lvl),
    max_level_interpolation_p4est(((splitting_criteria_t*) interpolation_node_ngbd_->get_p4est()->user_pointer)->max_lvl)
{
#ifdef CASL_THROWS
  if(max_level_interpolation_p4est < max_level_p4est)
    throw std::invalid_argument("my_p4est_interface_manager_t(): you're using UNDER-resolved interpolation tools for capturing the interface. Are you mentally sane? Go see a doctor or check your code...");
#endif
  interp_grad_phi   = NULL;
  interp_curvature  = NULL;
  interp_phi_xxyyzz = NULL;
  interp_gradient_of_normal = NULL;
  cell_FD_interface_neighbors = new map_of_interface_neighbors_t;
  tmp_FD_interface_neighbor   = new FD_interface_neighbor;
  clear_cell_FD_interface_neighbors();
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    face_FD_interface_neighbors[dim] = new map_of_interface_neighbors_t;
    clear_face_FD_interface_neighbors(dim);
  }
  grad_phi_local              = NULL;
  phi_xxyyzz_local            = NULL;
  curvature_local             = NULL;
  gradient_of_normal_local    = NULL;
  phi_on_computational_nodes  = NULL;
  use_second_derivative_when_computing_FD_theta = false;
  throw_if_ill_defined_grad = false; // <-- set this one to true if you want to know when real interface shit is going on
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
  if(interp_curvature != NULL)
    delete interp_curvature;
  if(interp_phi_xxyyzz != NULL)
    delete interp_phi_xxyyzz;
  if(interp_gradient_of_normal != NULL)
    delete interp_gradient_of_normal;
  if(grad_phi_local != NULL){
    PetscErrorCode ierr = VecDestroy(grad_phi_local); CHKERRXX(ierr); }
  if(phi_xxyyzz_local != NULL){
    PetscErrorCode ierr = VecDestroy(phi_xxyyzz_local); CHKERRXX(ierr); }
  if(curvature_local != NULL){
    PetscErrorCode ierr = VecDestroy(curvature_local); CHKERRXX(ierr); }
  if(gradient_of_normal_local != NULL){
    PetscErrorCode ierr = VecDestroy(gradient_of_normal_local); CHKERRXX(ierr); }
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

void my_p4est_interface_manager_t::set_levelset(Vec phi, const interpolation_method& method_interp_phi, Vec phi_xxyyzz,
                                                const bool& build_and_set_grad_phi_locally,
                                                const bool& build_and_set_curvature_locally)
{
  P4EST_ASSERT(phi != NULL);
  P4EST_ASSERT(VecIsSetForNodes(phi, interpolation_node_ngbd->get_nodes(), interpolation_node_ngbd->get_p4est()->mpicomm, 1));
#ifdef CASL_THROWS
  if(method_interp_phi == quadratic_non_oscillatory)
    std::cerr << "my_p4est_interface_manager_t::set_levelset() : using quadratic_non_oscillatory for the levelset interpolation may lead to erratic behavior for some solvers (especially with finite-volume approaches) because of the inherent discontinuities in the interpolation scheme. Use at your own risk!" << std::endl;
#endif
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

  if((method_interp_phi != linear || use_second_derivative_when_computing_FD_theta) && interp_phi_xxyyzz == NULL)
  {
    set_phi_xxyyzz(); // build the second derivatives locally if they were not given but would be eventually needed
    if(method_interp_phi != linear)
      interp_phi.set_input(phi, phi_xxyyzz_local, method_interp_phi); // faster sampling if the second derivatives are pre-computed
  }

  if(nodes == interpolation_node_ngbd->get_nodes())
    phi_on_computational_nodes = phi;

  if(build_and_set_grad_phi_locally)
    set_grad_phi();

  if(build_and_set_curvature_locally)
    set_curvature();

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

void my_p4est_interface_manager_t::build_phi_xxyyzz_locally()
{
#ifdef CASL_THROWS
  if(interp_phi.get_input_fields().size() != 1 || interp_phi.get_blocksize_of_input_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::build_phi_xxyyzz_locally(): can't determine the gradient of the levelset function if the levelset function wasn't set first...");
#endif

  if(phi_xxyyzz_local == NULL){
    PetscErrorCode ierr = VecCreateGhostNodesBlock(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), P4EST_DIM, &phi_xxyyzz_local); CHKERRXX(ierr); }
  interpolation_node_ngbd->second_derivatives_central(interp_phi.get_input_fields()[0], phi_xxyyzz_local);

  return;
}

void my_p4est_interface_manager_t::build_curvature_locally()
{
#ifdef CASL_THROWS
  if(interp_phi.get_input_fields().size() != 1 || interp_phi.get_blocksize_of_input_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::build_grad_phi_locally(): can't determine the curvature of the levelset function if the levelset function wasn't set first...");
#endif

  PetscErrorCode ierr;

  if(interp_grad_phi == NULL && grad_phi_local == NULL)
    build_grad_phi_locally();

  if(curvature_local == NULL){
    ierr = VecCreateGhostNodes(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), &curvature_local); CHKERRXX(ierr); }

  double *curvature_p;
  const double *phi_p, *grad_phi_p;
  const double *phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(interp_phi.get_input_fields()[0], &phi_p); CHKERRXX(ierr);
  Vec grad_phi_to_read = (interp_grad_phi == NULL ? grad_phi_local : interp_grad_phi->get_input_fields()[0]);
  P4EST_ASSERT(grad_phi_to_read != NULL);
  ierr = VecGetArrayRead(grad_phi_to_read, &grad_phi_p); CHKERRXX(ierr);
  if(interp_phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(interp_phi_xxyyzz->get_input_fields()[0], &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecGetArray(curvature_local, &curvature_p); CHKERRXX(ierr);


  quad_neighbor_nodes_of_node_t qnnn_buffer;
  const quad_neighbor_nodes_of_node_t* qnnn_p = (interpolation_node_ngbd->neighbors_are_initialized() ? NULL : &qnnn_buffer);

  for (size_t k = 0; k < interpolation_node_ngbd->get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = interpolation_node_ngbd->get_layer_node(k);
    if(interpolation_node_ngbd->neighbors_are_initialized())
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_p);
    else
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_buffer);
    curvature_p[node_idx] = qnnn_p->get_curvature(grad_phi_p, phi_p, phi_xxyyzz_p);
  }
  ierr = VecGhostUpdateBegin(curvature_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < interpolation_node_ngbd->get_local_size(); ++k) {
    const p4est_locidx_t node_idx = interpolation_node_ngbd->get_local_node(k);
    if(interpolation_node_ngbd->neighbors_are_initialized())
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_p);
    else
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_buffer);
    curvature_p[node_idx] = qnnn_p->get_curvature(grad_phi_p, phi_p, phi_xxyyzz_p);
  }
  ierr = VecGhostUpdateEnd(curvature_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(curvature_local, &curvature_p); CHKERRXX(ierr);
  if(interp_phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(interp_phi_xxyyzz->get_input_fields()[0], &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(grad_phi_to_read, &grad_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(interp_phi.get_input_fields()[0], &phi_p); CHKERRXX(ierr);
  return;
}

void my_p4est_interface_manager_t::build_grad_normal_locally()
{
#ifdef CASL_THROWS
  if(interp_phi.get_input_fields().size() != 1 || interp_phi.get_blocksize_of_input_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::build_grad_normal_locally(): can't determine the gradient of normal vectors associated with the levelset function if the levelset function wasn't set first...");
#endif

  PetscErrorCode ierr;

  if(interp_grad_phi == NULL && grad_phi_local == NULL)
    build_grad_phi_locally();

  if(gradient_of_normal_local == NULL){
    ierr = VecCreateGhostNodesBlock(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), SQR_P4EST_DIM, &gradient_of_normal_local); CHKERRXX(ierr); }

  double *gradient_of_normal_local_p;
  const double *phi_p, *grad_phi_p;
  const double *phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(interp_phi.get_input_fields()[0], &phi_p); CHKERRXX(ierr);
  Vec grad_phi_to_read = (interp_grad_phi == NULL ? grad_phi_local : interp_grad_phi->get_input_fields()[0]);
  P4EST_ASSERT(grad_phi_to_read != NULL);
  ierr = VecGetArrayRead(grad_phi_to_read, &grad_phi_p); CHKERRXX(ierr);
  if(interp_phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(interp_phi_xxyyzz->get_input_fields()[0], &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecGetArray(gradient_of_normal_local, &gradient_of_normal_local_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn_buffer;
  const quad_neighbor_nodes_of_node_t* qnnn_p = (interpolation_node_ngbd->neighbors_are_initialized() ? NULL : &qnnn_buffer);

  for (size_t k = 0; k < interpolation_node_ngbd->get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = interpolation_node_ngbd->get_layer_node(k);
    if(interpolation_node_ngbd->neighbors_are_initialized())
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_p);
    else
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_buffer);
    qnnn_p->get_gradient_of_normal(gradient_of_normal_local_p, phi_p, grad_phi_p, NULL, phi_xxyyzz_p, NULL);
  }
  ierr = VecGhostUpdateBegin(gradient_of_normal_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < interpolation_node_ngbd->get_local_size(); ++k) {
    const p4est_locidx_t node_idx = interpolation_node_ngbd->get_local_node(k);
    if(interpolation_node_ngbd->neighbors_are_initialized())
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_p);
    else
      interpolation_node_ngbd->get_neighbors(node_idx, qnnn_buffer);
    qnnn_p->get_gradient_of_normal(gradient_of_normal_local_p, phi_p, grad_phi_p, NULL, phi_xxyyzz_p, NULL);
  }
  ierr = VecGhostUpdateEnd(gradient_of_normal_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(gradient_of_normal_local, &gradient_of_normal_local_p); CHKERRXX(ierr);
  if(interp_phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(interp_phi_xxyyzz->get_input_fields()[0], &phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(grad_phi_to_read, &grad_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(interp_phi.get_input_fields()[0], &phi_p); CHKERRXX(ierr);
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
  P4EST_ASSERT(grad_phi != NULL && VecIsSetForNodes(grad_phi, interpolation_node_ngbd->get_nodes(), p4est->mpicomm, P4EST_DIM, true));
  if(interp_grad_phi == NULL)
    interp_grad_phi = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);
  return;
}

void my_p4est_interface_manager_t::set_phi_xxyyzz(Vec phi_xxyyzz_in)
{
  Vec phi_xxyyzz = phi_xxyyzz_in;
  if(phi_xxyyzz == NULL)
  {
    build_phi_xxyyzz_locally();
    phi_xxyyzz = phi_xxyyzz_local;
  }
  P4EST_ASSERT(phi_xxyyzz != NULL && VecIsSetForNodes(phi_xxyyzz, interpolation_node_ngbd->get_nodes(), p4est->mpicomm, P4EST_DIM, true));
  if(interp_phi_xxyyzz == NULL)
    interp_phi_xxyyzz = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_phi_xxyyzz->set_input(phi_xxyyzz, linear, P4EST_DIM);
  return;
}

void my_p4est_interface_manager_t::set_curvature(Vec curvature_in)
{
  Vec curvature = curvature_in;
  if(curvature == NULL)
  {
    build_curvature_locally();
    curvature = curvature_local;
  }
  P4EST_ASSERT(curvature != NULL && VecIsSetForNodes(curvature, interpolation_node_ngbd->get_nodes(), p4est->mpicomm, 1));

  if(interp_curvature == NULL)
    interp_curvature = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_curvature->set_input(curvature, linear);
  return;
}

void my_p4est_interface_manager_t::set_gradient_of_normal(Vec grad_normal_in)
{
  Vec grad_normal = grad_normal_in;
  if(grad_normal == NULL)
  {
    build_grad_normal_locally();
    grad_normal = gradient_of_normal_local;
  }
  P4EST_ASSERT(grad_normal != NULL && VecIsSetForNodes(grad_normal, interpolation_node_ngbd->get_nodes(), p4est->mpicomm, SQR_P4EST_DIM));

  if(interp_gradient_of_normal == NULL)
    interp_gradient_of_normal = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_gradient_of_normal->set_input(grad_normal, linear, SQR_P4EST_DIM);
  return;
}

double my_p4est_interface_manager_t::find_FD_interface_theta_in_cartesian_direction(const double* xyz_dof, const u_char& oriented_dir, const bool& is_neighbor_wall) const
{
  // compute the finite-difference infamous theta
  double xyz_this_side[P4EST_DIM] = {DIM(xyz_dof[0], xyz_dof[1], xyz_dof[2])};
  const double hh = (is_neighbor_wall ? 0.5 : 1.0)*dxyz_min[oriented_dir/2]; // distance between the dof of interest and the neighbor across the interface
  double xyz_across[P4EST_DIM]    = {DIM(xyz_dof[0], xyz_dof[1], xyz_dof[2])}; xyz_across[oriented_dir/2] += (oriented_dir%2 == 1 ? +1.0 : -1.0)*hh; // no need to worry about periodicity, the interpolation object will
  double phi_this_side  = interp_phi(xyz_this_side);
  double phi_across     = interp_phi(xyz_across);
  P4EST_ASSERT(signs_of_phi_are_different(phi_this_side, phi_across));
  double theta = 0.0; // initialize
  double rel_scale = 1.0;
  double xyz_M[P4EST_DIM] = {DIM(xyz_this_side[0], xyz_this_side[1], xyz_this_side[2])};
  for (int k = 0; k < max_level_interpolation_p4est - max_level_p4est - (is_neighbor_wall ? 1 : 0); ++k)
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
    xyz_M[oriented_dir/2] = xyz_this_side[oriented_dir/2] + (oriented_dir%2 == 1 ? +1.0 : -1.0)*rel_scale*hh; // no need to worry about periodicity, the interpolation object will
    const double phi_M = interp_phi(xyz_M);
    if(!signs_of_phi_are_different(phi_this_side, phi_M))
    {
      theta += rel_scale;
      phi_this_side = phi_M;
      xyz_this_side[oriented_dir/2] = xyz_M[oriented_dir/2];
    }
    else
    {
      phi_across = phi_M;
      xyz_across[oriented_dir/2] = xyz_M[oriented_dir/2];
    }
  }
  double subscale_theta_negative;
  if(use_second_derivative_when_computing_FD_theta)
  {
    P4EST_ASSERT(interp_phi_xxyyzz != NULL);
    const double phi_dd_Q = (*interp_phi_xxyyzz)(xyz_this_side, oriented_dir/2);
    const double phi_dd_N = (*interp_phi_xxyyzz)(xyz_across,    oriented_dir/2);
    subscale_theta_negative = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, phi_dd_Q, phi_dd_N, rel_scale*hh);
  }
  else
    subscale_theta_negative = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, rel_scale*hh, rel_scale*hh);
  theta += rel_scale*(phi_this_side > 0.0 ? 1.0 - subscale_theta_negative : subscale_theta_negative);
  return MAX(0.0, MIN(1.0, theta));
}

const FD_interface_neighbor& my_p4est_interface_manager_t::get_cell_FD_interface_neighbor_for(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants && // must be a local quadrant
               -P4EST_FACES <= neighbor_quad_idx && neighbor_quad_idx < p4est->local_num_quadrants + (p4est_locidx_t) ghost->ghosts.elem_count); // must be a known quadrant or a wall

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

  const p4est_topidx_t tree_idx = tree_index_of_quad(quad_idx, p4est, ghost);
#ifdef P4EST_DEBUG
  const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est, ghost);
  if(neighbor_quad_idx >= 0)
  {
    const p4est_topidx_t neighbor_tree_idx = tree_index_of_quad(neighbor_quad_idx, p4est, ghost);
    // check that they're both as fine as it gets :
    const p4est_quadrant_t* neighbor_quad = fetch_quad(neighbor_quad_idx, neighbor_tree_idx, p4est, ghost);
    P4EST_ASSERT(quad->level == neighbor_quad->level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);
  }
  else
    P4EST_ASSERT(is_quad_Wall(p4est, tree_idx, quad, oriented_dir) && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl); // it is indeed a wall cell and it is indeed as fine as it gets
#endif

  // compute the finite-difference infamous theta
  double xyz_Q[P4EST_DIM];  quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_Q);
  tmp_FD_interface_neighbor->theta = find_FD_interface_theta_in_cartesian_direction(xyz_Q, oriented_dir, (neighbor_quad_idx < 0));
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

const FD_interface_neighbor& my_p4est_interface_manager_t::get_face_FD_interface_neighbor_for(const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& dim, const u_char& oriented_dir) const
{
  P4EST_ASSERT(0 <= face_idx && face_idx < faces->num_local[dim] && // must be a local face
               -P4EST_FACES <= neighbor_face_idx && neighbor_face_idx < faces->num_local[dim] + faces->num_ghost[dim]); // must be a known face or a wall
  P4EST_ASSERT(neighbor_face_idx >= 0 || (-1 - neighbor_face_idx)/2 != dim); // if it is a wall, it must be a tranverse wall

  if(face_FD_interface_neighbors[dim] != NULL) // check in map if storing them, first
  {
    map_of_interface_neighbors_t::iterator it = face_FD_interface_neighbors[dim]->find({face_idx, neighbor_face_idx});
    if(it != face_FD_interface_neighbors[dim]->end())
    {
      if((it->first.local_dof_idx == neighbor_face_idx && !it->second.swapped) || (it->first.local_dof_idx == face_idx && it->second.swapped)) // currently set for the reversed pair --> swap it
      {
        it->second.theta = 1.0 - it->second.theta;
        it->second.swapped = !it->second.swapped;
      }
      return it->second;
    }
  }

  // compute the finite-difference infamous theta
  double xyz_face[P4EST_DIM]; faces->xyz_fr_f(face_idx, dim, xyz_face);
#ifdef P4EST_DEBUG
  double xyz_neighbor[P4EST_DIM];
  const double* xyz_max = faces->get_xyz_max();
  const double* xyz_min = faces->get_xyz_min();
  // check that the neighbor is indeed dxyz_min[oriented_dir/2] (or half of that if wall) away in the expected direction:
  if(neighbor_face_idx >= 0)
    faces->xyz_fr_f(neighbor_face_idx, dim, xyz_neighbor);
  else
  {
    const int wall_orientation = (-1 - neighbor_face_idx); P4EST_ASSERT(0 <= wall_orientation && wall_orientation < P4EST_FACES);
    for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      xyz_neighbor[comp] = xyz_face[comp];
    xyz_neighbor[oriented_dir/2] = (wall_orientation%2 == 1 ? xyz_max[oriented_dir/2] : xyz_min[oriented_dir/2]);
  }
  bool check = true;
  for (u_char comp = 0; comp < P4EST_DIM; ++comp)
  {
    double diff = xyz_neighbor[comp] - xyz_face[comp];
    if(comp == oriented_dir/2 && faces->periodicity(oriented_dir/2))
      diff -= round(diff/(xyz_max[oriented_dir/2] - xyz_min[oriented_dir/2]))*(xyz_max[oriented_dir/2] - xyz_min[oriented_dir/2]);

    if(comp != oriented_dir/2)
      check = check && (fabs(diff) < 0.001*dxyz_min[comp]); // should be "0.0" but using floating-point comparison with threshold
    else
      check = check && (fabs(fabs(diff) - (neighbor_face_idx < 0 ? 0.5 : 1.0)*dxyz_min[oriented_dir/2]) < 0.001*dxyz_min[comp]); // should be "0.0" but using floating-point comparison with threshold
  }
  P4EST_ASSERT(check);
#endif

  tmp_FD_interface_neighbor->theta = find_FD_interface_theta_in_cartesian_direction(xyz_face, oriented_dir, (neighbor_face_idx < 0));
  tmp_FD_interface_neighbor->swapped = false;

  if(face_FD_interface_neighbors[dim] != NULL)
  {
    couple_of_dofs couple_of_face_indices = {face_idx, neighbor_face_idx};
    std::pair<map_of_interface_neighbors_t::iterator, bool> ret = face_FD_interface_neighbors[dim]->insert({couple_of_face_indices, *tmp_FD_interface_neighbor}); // add it to the map so that future access is read from memory;
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

void my_p4est_interface_manager_t::get_coordinates_of_FD_interface_point_between_faces(const u_char& dim, const p4est_locidx_t& face_idx, const p4est_locidx_t& neighbor_face_idx, const u_char& oriented_dir, double *xyz) const
{
  const FD_interface_neighbor& interface_point = get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dim, oriented_dir);
  faces->xyz_fr_f(face_idx, dim, xyz);
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
                                                                bool &intersection_found, bool which_face_is_intersected[P4EST_FACES], const u_char& check_only_this_face) const
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
  const bool check_all_points = (check_only_this_face >= P4EST_FACES);
  intersection_found = (check_all_points ? signs_of_phi_are_different(phi_mmm, phi_ppp) : false);
  if(which_face_is_intersected != NULL)
    for (u_char face_dir = 0; face_dir < P4EST_FACES; ++face_dir)
      which_face_is_intersected[face_dir] = false;
  const double& phi_ref = ((check_all_points || check_only_this_face%2 == 0) ? phi_mmm : phi_ppp);

  u_int kxyz_min[P4EST_DIM] = {DIM(0, 0, 0)}; if(!check_all_points) kxyz_min[check_only_this_face/2] = (check_only_this_face%2 == 0 ? 0 : 1);
  u_int kxyz_max[P4EST_DIM] = {DIM(1, 1, 1)}; if(!check_all_points) kxyz_max[check_only_this_face/2] = kxyz_min[check_only_this_face/2];

  // check the vertices first
  for (u_int kx = kxyz_min[0]; kx <= kxyz_max[0]; ++kx)
    for (u_int ky = kxyz_min[1]; ky <= kxyz_max[1]; ++ky)
#ifdef P4_TO_P8
      for (u_int kz = kxyz_min[2]; kz <= kxyz_max[2]; ++kz)
#endif
      {
        if(ANDD(kx == 0, ky == 0, kz == 0) || ANDD(kx == 1, ky == 1, kz == 1)) // mmm and ppp vertices
          continue;
        const double phi_vertex = (phi_on_computational_nodes_p != NULL ? phi_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + SUMD(kx, 2*ky, 4*kz)]] :
                                   interp_phi(DIM(xyz_mmm[0] + kx*dxyz_quad[0], xyz_mmm[1] + ky*dxyz_quad[1], xyz_mmm[2] + kz*dxyz_quad[2])));
        intersection_found = intersection_found || signs_of_phi_are_different(phi_ref, phi_vertex);

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

  if(intersection_found && !check_all_points) // shortcut the rest if you can
    return;

  const u_int n_mls_points_per_dimension = (1 << subcell_resolution())*interpolation_degree() + 1;
  if(n_mls_points_per_dimension > 2) // more points to check
  {
    const double dxyz_mls[P4EST_DIM]  = {DIM(dxyz_quad[0]/(double) (n_mls_points_per_dimension - 1), dxyz_quad[1]/(double) (n_mls_points_per_dimension - 1), dxyz_quad[2]/(double) (n_mls_points_per_dimension - 1))};

    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      kxyz_min[dim] = (check_all_points || dim != check_only_this_face/2 ?                               0  : (check_only_this_face%2 == 0 ? 0 : n_mls_points_per_dimension - 1));
      kxyz_max[dim] = (check_all_points || dim != check_only_this_face/2 ? (n_mls_points_per_dimension - 1) : (check_only_this_face%2 == 0 ? 0 : n_mls_points_per_dimension - 1));
    }

    for (u_int kx = kxyz_min[0]; kx <= kxyz_max[0]; ++kx)
      for (u_int ky = kxyz_min[1]; ky <= kxyz_max[1]; ++ky)
  #ifdef P4_TO_P8
        for (u_int kz = kxyz_min[2]; kz <= kxyz_max[2]; ++kz)
  #endif
        {
          if(ANDD(kx == 0 || kx == n_mls_points_per_dimension - 1, ky == 0 || ky == n_mls_points_per_dimension - 1, kz == 0 || kz == n_mls_points_per_dimension - 1)) // that's a vertex, already done
            continue;
          const double xyz_mls_point[P4EST_DIM] = {DIM(xyz_mmm[0] + ((double) kx)*dxyz_mls[0], xyz_mmm[1] + ((double) ky)*dxyz_mls[1], xyz_mmm[2] + ((double) kz)*dxyz_mls[2])};
          const double phi_mls_point = interp_phi(xyz_mls_point);

          intersection_found = intersection_found || signs_of_phi_are_different(phi_ref, phi_mls_point);
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
      throw std::logic_error("my_p4est_interface_manager_t::is_quad_crossed_by_interface() : found a cell bigger than expected containing an interface intersection.");
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

bool my_p4est_interface_manager_t::is_face_crossed_by_interface(const p4est_locidx_t& face_idx, const u_char dim) const
{
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces->f2q(face_idx, dim, quad_idx, tree_idx);
  const u_char oriented_dir = 2*dim + (faces->q2f(quad_idx, 2*dim) == face_idx ? 0 : + 1);

  const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est, ghost);

  if(quad->level < (int8_t) max_level_p4est)
  {
#ifdef CASL_THROWS
    bool intersection_found;
    detect_mls_interface_in_quad(quad_idx, tree_idx, intersection_found, NULL, oriented_dir);
    if(intersection_found)
      throw std::logic_error("my_p4est_interface_manager_t::is_face_crossed_by_interface() : found a cell bigger than expected containing an interface intersection on one of its faces.");
#endif
    return false;
  }
  bool intersection_found;
  detect_mls_interface_in_quad(quad_idx, tree_idx, intersection_found, NULL, oriented_dir);
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
