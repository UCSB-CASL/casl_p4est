#include "my_p4est_interface_manager.h"

my_p4est_interface_manager_t::my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const my_p4est_cell_neighbors_t* cell_ngbd, const double* dxyz_min_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_)
  : c_ngbd(cell_ngbd), faces(faces_), p4est(cell_ngbd->get_p4est()), ghost(cell_ngbd->get_ghost()), dxyz_min(dxyz_min_),
    interpolation_node_ngbd(interpolation_node_ngbd_), interp_phi(interpolation_node_ngbd_)
{
#ifdef WITH_SUBREFINEMENT
  if(((splitting_criteria_t*) interpolation_node_ngbd_->get_p4est()->user_pointer)->max_lvl <= ((splitting_criteria_t*) p4est->user_pointer)->max_lvl)
    throw std::invalid_argument("my_p4est_interface_manager_t(): this object needs a finer interpolation grid to handle interface subresolved points properly...");
#endif
  interp_grad_phi   = NULL;
  interp_phi_xxyyzz = NULL;
  cell_FD_interface_data  = new map_of_interface_neighbors_t;
  tmp_FD_interface_data   = new FD_interface_data;
  clear_cell_FD_interface_data();
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    face_FD_interface_data[dim] = new map_of_interface_neighbors_t;
    clear_face_FD_interface_data(dim);
  }
  grad_phi_local = NULL;
}

my_p4est_interface_manager_t::~my_p4est_interface_manager_t()
{
  if(cell_FD_interface_data != NULL)
    delete cell_FD_interface_data;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(face_FD_interface_data[dim] != NULL)
      delete face_FD_interface_data[dim];

  delete tmp_FD_interface_data;

  if(interp_grad_phi != NULL)
    delete interp_grad_phi;
  if(interp_phi_xxyyzz != NULL)
    delete interp_phi_xxyyzz;
  if(grad_phi_local != NULL){
    PetscErrorCode ierr = VecDestroy(grad_phi_local); CHKERRXX(ierr); }
}

void my_p4est_interface_manager_t::set_levelset(Vec phi, const interpolation_method& method_interp_phi, Vec phi_xxyyzz)
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
  return;
}

void my_p4est_interface_manager_t::set_grad_phi()
{
  if(interp_phi.get_input_fields().size() != 1 || interp_phi.get_blocksize_of_inupt_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::set_grad_phi(): can't determine the gradient of the levelset function if the levelset function wasn't set first...");

  if(grad_phi_local == NULL){
    PetscErrorCode ierr = VecCreateGhostNodes(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), &grad_phi_local); CHKERRXX(ierr); }

  interpolation_node_ngbd->first_derivatives_central(interp_phi.get_input_fields()[0], grad_phi_local);
  if(interp_grad_phi== NULL)
    interp_grad_phi = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_grad_phi->set_input(grad_phi_local, linear, P4EST_DIM);
  return;
}

const FD_interface_data& my_p4est_interface_manager_t::get_cell_FD_interface_data_for(const p4est_locidx_t& quad_idx, const p4est_locidx_t& neighbor_quad_idx, const u_char& oriented_dir) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants && // must be a local quadrant
               0 <= neighbor_quad_idx && neighbor_quad_idx < p4est->local_num_quadrants + (p4est_locidx_t) ghost->ghosts.elem_count); // must be a known quadrant

  const p4est_topidx_t tree_idx = tree_index_of_quad(quad_idx, p4est, ghost);
  const p4est_tree_t*     tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const p4est_quadrant_t* neighbor_quad;
  p4est_topidx_t tree_idx_neighbor;
  if(neighbor_quad_idx >= p4est->local_num_quadrants)
  {
    neighbor_quad = p4est_const_quadrant_array_index(&ghost->ghosts, neighbor_quad_idx - p4est->local_num_quadrants);
    tree_idx_neighbor = neighbor_quad->p.piggy3.which_tree;
  }
  else
  {
    tree_idx_neighbor = tree_index_of_quad(neighbor_quad_idx, p4est, ghost);
    const p4est_tree_t* tree_neighbor = p4est_tree_array_index(p4est->trees, tree_idx_neighbor);
    neighbor_quad = p4est_const_quadrant_array_index(&tree_neighbor->quadrants, neighbor_quad_idx - tree_neighbor->quadrants_offset);
  }

  if(cell_FD_interface_data != NULL) // check if stored, first
  {
    map_of_interface_neighbors_t::iterator it = cell_FD_interface_data->find({quad_idx, neighbor_quad_idx});
    if(it != cell_FD_interface_data->end())
    {
      if((it->first.local_dof_idx == neighbor_quad_idx && !it->second.swapped) || (it->first.local_dof_idx == quad_idx && it->second.swapped)) // currently set for the reversed pair --> swap it
      {
        it->second.theta = 1.0 - it->second.theta;
        it->second.swapped = !it->second.swapped;
      }
      return it->second;
    }
  }

  const p4est_t* subrefined_p4est = interpolation_node_ngbd->get_p4est();
  const p4est_nodes_t* subrefined_nodes = interpolation_node_ngbd->get_nodes();
  const p4est_locidx_t fine_node_idx_for_quad           = get_fine_node_idx_of_quad_center(subrefined_p4est, subrefined_nodes, *quad, tree_idx);
  const p4est_locidx_t fine_node_idx_for_neighbor_quad  = get_fine_node_idx_of_quad_center(subrefined_p4est, subrefined_nodes, *neighbor_quad, tree_idx_neighbor);
  if(fine_node_idx_for_quad < 0 || fine_node_idx_for_neighbor_quad < 0)
    std::cout << "fine_node_idx_for_quad = " << fine_node_idx_for_quad  << ", fine_node_idx_for_neighbor_quad" << fine_node_idx_for_neighbor_quad << std::endl;
  P4EST_ASSERT(0 <= fine_node_idx_for_quad          && fine_node_idx_for_quad < (p4est_locidx_t)(subrefined_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(0 <= fine_node_idx_for_neighbor_quad && fine_node_idx_for_neighbor_quad < (p4est_locidx_t)(subrefined_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(quad->level == neighbor_quad->level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);


  double xyz_quad[P4EST_DIM];     quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  double xyz_neighbor[P4EST_DIM]; quad_xyz_fr_q(neighbor_quad_idx, tree_idx_neighbor, p4est, ghost, xyz_neighbor);

  const double phi_quad     = interp_phi(xyz_quad);
  const double phi_neighbor = interp_phi(xyz_neighbor);
  P4EST_ASSERT(signs_of_phi_are_different(phi_quad, phi_neighbor));
  const quad_neighbor_nodes_of_node_t* qnnn; interpolation_node_ngbd->get_neighbors(fine_node_idx_for_quad, qnnn);
  p4est_locidx_t mid_point_fine_node_idx = qnnn->neighbor(oriented_dir);
  P4EST_ASSERT(0 <= mid_point_fine_node_idx && mid_point_fine_node_idx == interpolation_node_ngbd->get_neighbors(fine_node_idx_for_neighbor_quad).neighbor(oriented_dir + (oriented_dir%2 == 0 ? +1 : -1)));

  double xyz_midpoint[P4EST_DIM]; node_xyz_fr_n(mid_point_fine_node_idx, subrefined_p4est, subrefined_nodes, xyz_midpoint);
  const double mid_point_phi      = interp_phi(xyz_midpoint);
  const bool no_past_mid_point    = signs_of_phi_are_different(phi_quad, mid_point_phi);
  const double &phi_this_side     = (no_past_mid_point ? phi_quad       : mid_point_phi);
  const double &phi_across        = (no_past_mid_point ? mid_point_phi  : phi_neighbor);

  if(interp_phi_xxyyzz != NULL)
  {
    const double phi_dd_this_side = (no_past_mid_point ? (*interp_phi_xxyyzz).operator()(xyz_quad, oriented_dir/2)  : (*interp_phi_xxyyzz)(xyz_midpoint, oriented_dir/2));
    const double phi_dd_across    = (no_past_mid_point ? (*interp_phi_xxyyzz)(xyz_midpoint, oriented_dir/2)         : (*interp_phi_xxyyzz)(xyz_neighbor, oriented_dir/2));
    tmp_FD_interface_data->theta  = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, phi_dd_this_side, phi_dd_across, 0.5*dxyz_min[oriented_dir/2]);
  }
  else
    tmp_FD_interface_data->theta  = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, 0.5*dxyz_min[oriented_dir/2], 0.5*dxyz_min[oriented_dir/2]);
  tmp_FD_interface_data->theta = (phi_this_side > 0.0 ? 1.0 - tmp_FD_interface_data->theta : tmp_FD_interface_data->theta);
  tmp_FD_interface_data->theta = MAX(0.0, MIN(tmp_FD_interface_data->theta, 1.0));
  tmp_FD_interface_data->node_interpolant.clear();
  tmp_FD_interface_data->node_interpolant.add_term((no_past_mid_point ? fine_node_idx_for_quad  : mid_point_fine_node_idx),         1.0 - tmp_FD_interface_data->theta);
  tmp_FD_interface_data->node_interpolant.add_term((no_past_mid_point ? mid_point_fine_node_idx : fine_node_idx_for_neighbor_quad), tmp_FD_interface_data->theta);

  tmp_FD_interface_data->theta = 0.5*(tmp_FD_interface_data->theta + (no_past_mid_point ? 0.0 : 1.0));
  P4EST_ASSERT(0.0 <= tmp_FD_interface_data->theta && tmp_FD_interface_data->theta <= 1.0);
  tmp_FD_interface_data->swapped = false;

  if(cell_FD_interface_data != NULL)
  {
    which_interface_neighbor_t which_one = {quad_idx, neighbor_quad_idx};
    std::pair<map_of_interface_neighbors_t::iterator, bool> ret = cell_FD_interface_data->insert({which_one, *tmp_FD_interface_data}); // add it to the map so that future access is read from memory;
    P4EST_ASSERT(ret.second);
    return ret.first->second;
  }
  else
    return *tmp_FD_interface_data;
}

void my_p4est_interface_manager_t::compute_subvolumes_in_cell(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, double& negative_volume, double& positive_volume) const
{
  if(quad_idx >= p4est->local_num_quadrants)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::compute_subvolumes_in_computational_cell(): cannot be called on ghost cells");
  const p4est_tree* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  const double logical_size_quad = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
  const double* tree_dimensions = c_ngbd->get_tree_dimensions();
  const double cell_dxyz[P4EST_DIM] = {DIM(tree_dimensions[0]*logical_size_quad, tree_dimensions[1]*logical_size_quad, tree_dimensions[2]*logical_size_quad)};
  const double quad_volume = MULTD(cell_dxyz[0], cell_dxyz[1], cell_dxyz[2]);

#ifdef WITH_SUBREFINEMENT
  double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  const p4est_t* subrefined_p4est       = interpolation_node_ngbd->get_p4est();
  const p4est_nodes_t* subrefined_nodes = interpolation_node_ngbd->get_nodes();
  if(quadrant_is_subrefined(subrefined_p4est, subrefined_nodes, *quad, tree_idx)) // the quadrant is subrefined, find all subrefining quads and do the calculations therein
  {
    std::vector<p4est_locidx_t> indices_of_subrefining_quads;
    interpolation_node_ngbd->get_hierarchy()->get_all_quadrants_in(quad, tree_idx, indices_of_subrefining_quads);
    P4EST_ASSERT(indices_of_subrefining_quads.size() > 0);
    negative_volume = 0.0;
    const p4est_tree* subrefined_tree = p4est_tree_array_index(subrefined_p4est->trees, tree_idx);
    Vec subrefined_phi = interp_phi.get_input_fields()[0];
    for (size_t k = 0; k < indices_of_subrefining_quads.size(); ++k)
    {
      const p4est_locidx_t& subrefined_quad_idx = indices_of_subrefining_quads[k];
      const p4est_quadrant_t* subrefined_quad = p4est_const_quadrant_array_index(&subrefined_tree->quadrants, subrefined_quad_idx - subrefined_tree->quadrants_offset);
      negative_volume += area_in_negative_domain_in_one_quadrant(subrefined_p4est, subrefined_nodes, subrefined_quad, subrefined_quad_idx, subrefined_phi);
    }
  }
  else
  {
    const double phi_q = interp_phi(xyz_quad);
#ifdef P4EST_DEBUG
    for (char kx = -1; kx < 2; kx += 2)
      for (char ky = -1; ky < 2; ky += 2)
        for (char kz = -1; kz < 2; kz += 2)
        {
          double xyz_vertex[P4EST_DIM] = {DIM(xyz_quad[0] + kx*0.5*cell_dxyz[0], xyz_quad[1] + ky*0.5*cell_dxyz[1], xyz_quad[2] + kz*0.5*cell_dxyz[2])};
          P4EST_ASSERT(!signs_of_phi_are_different(phi_q, interp_phi(xyz_vertex)));
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

#ifdef DEBUG
int my_p4est_interface_manager_t::cell_FD_map_is_consistent_across_procs()
{
  P4EST_ASSERT(cell_FD_interface_data != NULL);

  int mpiret;
  std::vector<int> senders(p4est->mpisize, 0);
  int num_expected_replies = 0;

  int it_is_alright = true;
  std::map<int, std::vector<which_interface_neighbor_t> > map_of_query_interface_neighbors; map_of_query_interface_neighbors.clear();
  std::map<int, std::vector<which_interface_neighbor_t> > map_of_mirrors; map_of_mirrors.clear();
  int first_rank_ghost_owner = 0;
  while (first_rank_ghost_owner < p4est->mpisize && ghost->proc_offsets[first_rank_ghost_owner + 1] == 0) { first_rank_ghost_owner++; }

  for (map_of_interface_neighbors_t::const_iterator it = cell_FD_interface_data->begin(); it != cell_FD_interface_data->end(); ++it)
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
      which_interface_neighbor_t other_one;
      other_one.local_dof_idx = ghost_nb_quad->p.piggy3.local_num;
      other_one.neighbor_dof_idx = local_quad_idx;
      map_of_query_interface_neighbors[rank_owner].push_back(other_one);
      map_of_mirrors[rank_owner].push_back({local_quad_idx, ghost_quad_idx});
    }
  }

  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  std::vector<MPI_Request> mpi_query_requests; mpi_query_requests.resize(0);
  std::vector<MPI_Request> mpi_reply_requests; mpi_reply_requests.resize(0);

  // send the requests...
  for (std::map<int, std::vector<which_interface_neighbor_t> >::const_iterator it = map_of_query_interface_neighbors.begin();
       it != map_of_query_interface_neighbors.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    int rank = it->first;
    P4EST_ASSERT(senders[rank] == 1);

    MPI_Request req;
    mpiret = MPI_Isend((void*) &map_of_query_interface_neighbors[rank][0], sizeof(which_interface_neighbor_t)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
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
        P4EST_ASSERT(byte_count%sizeof(which_interface_neighbor_t) == 0);
        std::vector<which_interface_neighbor_t> queries(byte_count/sizeof(which_interface_neighbor_t));

        mpiret = MPI_Recv((void*) &queries[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 15351, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

        std::vector<double>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(which_interface_neighbor_t));

        for (size_t kk = 0; kk < response.size(); ++kk)
        {
          for (p4est_locidx_t q = ghost->proc_offsets[status.MPI_SOURCE]; q < ghost->proc_offsets[status.MPI_SOURCE + 1]; ++q)
          {
            const p4est_quadrant_t* ghost_quad = p4est_const_quadrant_array_index(&ghost->ghosts, q);
            if(ghost_quad->p.piggy3.local_num == queries[kk].neighbor_dof_idx)
              queries[kk].neighbor_dof_idx = q + p4est->local_num_quadrants;
          }
          if(cell_FD_interface_data->find(queries[kk]) == cell_FD_interface_data->end())
            it_is_alright = false;
          else
            response[kk] = cell_FD_interface_data->at(queries[kk]).theta;
        }

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*)&response[0], (response.size())*sizeof(double), MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
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
          which_interface_neighbor_t mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          which_interface_neighbor_t queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];

          const bool consistent_with_other_proc_data = fabs(cell_FD_interface_data->at(mirror).theta + reply_buffer[kk] - 1.0) < EPS;
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
