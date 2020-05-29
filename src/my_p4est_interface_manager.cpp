#include "my_p4est_interface_manager.h"

my_p4est_interface_manager_t::my_p4est_interface_manager_t(const my_p4est_faces_t* faces_, const my_p4est_cell_neighbors_t* cell_ngbd, const double* dxyz_min_, const my_p4est_node_neighbors_t* interpolation_node_ngbd_)
  : c_ngbd(cell_ngbd), faces(faces_), p4est(cell_ngbd->get_p4est()), ghost(cell_ngbd->get_ghost()), dxyz_min(dxyz_min_),
    interpolation_node_ngbd(interpolation_node_ngbd_)
{
#ifdef WITH_SUBREFINEMENT
  if(((splitting_criteria_t*) interpolation_node_ngbd_->get_p4est()->user_pointer)->max_lvl <= ((splitting_criteria_t*) p4est->user_pointer)->max_lvl)
    throw std::invalid_argument("my_p4est_interface_manager_t(): this object needs a finer interpolation grid to handle interface subresolved points properly...");
#endif
  interp_phi        = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
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
  {
    std::cout << "this interface manager contained " << cell_FD_interface_data->size() << " elements" << " (from proc " << p4est->mpirank << " which has " << p4est->local_num_quadrants << " local cells)"<< std::endl;
    delete cell_FD_interface_data;
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(face_FD_interface_data[dim] != NULL)
      delete face_FD_interface_data[dim];

  delete tmp_FD_interface_data;

  delete interp_phi;
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
    interp_phi->set_input(phi, phi_xxyyzz, method_interp_phi);
    if(interp_phi_xxyyzz == NULL)
      interp_phi_xxyyzz = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
    interp_phi_xxyyzz->set_input(phi_xxyyzz, linear, P4EST_DIM);
  }
  else
    interp_phi->set_input(phi, method_interp_phi);
  return;
}

void my_p4est_interface_manager_t::set_grad_phi()
{
  if(interp_phi->get_input_fields().size() != 1 || interp_phi->get_blocksize_of_inupt_fields() != 1)
    throw std::runtime_error("my_p4est_interface_manager_t::set_grad_phi(): can't determine the gradient of the levelset function if the levelset function wasn't set first...");

  if(grad_phi_local == NULL){
    PetscErrorCode ierr = VecCreateGhostNodes(interpolation_node_ngbd->get_p4est(), interpolation_node_ngbd->get_nodes(), &grad_phi_local); CHKERRXX(ierr); }

  interpolation_node_ngbd->first_derivatives_central(interp_phi->get_input_fields()[0], grad_phi_local);
  if(interp_grad_phi== NULL)
    interp_grad_phi = new my_p4est_interpolation_nodes_t(interpolation_node_ngbd);
  interp_grad_phi->set_input(grad_phi_local, linear, P4EST_DIM);
  return;
}

const FD_interface_data& my_p4est_interface_manager_t::get_cell_interface_data_for(const p4est_locidx_t& quad_idx, const u_char& face_dir) const
{
  P4EST_ASSERT(0 <= quad_idx && quad_idx < p4est->local_num_quadrants && face_dir < P4EST_FACES); // must be a local quadrant

  if(cell_FD_interface_data != NULL) // check if stored, first
  {
    map_of_interface_neighbors_t::const_iterator it = cell_FD_interface_data->find({quad_idx, face_dir});
    if(it != cell_FD_interface_data->end())
      return it->second;
  }

  const p4est_topidx_t tree_idx = tree_index_of_quad(quad_idx, p4est, ghost);
  set_of_neighboring_quadrants direct_neighbor;
  c_ngbd->find_neighbor_cells_of_cell(direct_neighbor, quad_idx, tree_idx, face_dir);
  P4EST_ASSERT(direct_neighbor.size() == 1);
  const p4est_tree_t*     tree = p4est_tree_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_const_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const p4est_quadrant_t& neighbor_quad = *direct_neighbor.begin();

  const p4est_t* subrefined_p4est = interpolation_node_ngbd->get_p4est();
  const p4est_nodes_t* subrefined_nodes = interpolation_node_ngbd->get_nodes();
  const p4est_locidx_t fine_node_idx_for_quad           = get_fine_node_idx_of_quad_center(subrefined_p4est, subrefined_nodes, *quad, tree_idx);
  const p4est_locidx_t fine_node_idx_for_neighbor_quad  = get_fine_node_idx_of_quad_center(subrefined_p4est, subrefined_nodes, neighbor_quad, neighbor_quad.p.piggy3.which_tree);
  if(fine_node_idx_for_quad < 0 || fine_node_idx_for_neighbor_quad < 0)
    std::cout << "fine_node_idx_for_quad = " << fine_node_idx_for_quad  << ", fine_node_idx_for_neighbor_quad" << fine_node_idx_for_neighbor_quad << std::endl;
  P4EST_ASSERT(0 <= fine_node_idx_for_quad          && fine_node_idx_for_quad < (p4est_locidx_t)(subrefined_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(0 <= fine_node_idx_for_neighbor_quad && fine_node_idx_for_neighbor_quad < (p4est_locidx_t)(subrefined_nodes->indep_nodes.elem_count));
  P4EST_ASSERT(quad->level == neighbor_quad.level && quad->level == (int8_t) ((splitting_criteria_t*) p4est->user_pointer)->max_lvl);


  double xyz_quad[P4EST_DIM];     quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);
  double xyz_neighbor[P4EST_DIM]; quad_xyz_fr_q(direct_neighbor.begin()->p.piggy3.local_num, direct_neighbor.begin()->p.piggy3.which_tree, p4est, ghost, xyz_neighbor);

  const double phi_quad     = (*interp_phi)(xyz_quad);
  const double phi_neighbor = (*interp_phi)(xyz_neighbor);
  P4EST_ASSERT(signs_of_phi_are_different(phi_quad, phi_neighbor));
  tmp_FD_interface_data->neighbor_quad_idx      = neighbor_quad.p.piggy3.local_num;
  tmp_FD_interface_data->quad_fine_node_idx     = fine_node_idx_for_quad;
  tmp_FD_interface_data->neighbor_fine_node_idx = fine_node_idx_for_neighbor_quad;
  const quad_neighbor_nodes_of_node_t* qnnn; interpolation_node_ngbd->get_neighbors(fine_node_idx_for_quad, qnnn);
  tmp_FD_interface_data->mid_point_fine_node_idx  = qnnn->neighbor(face_dir);
  P4EST_ASSERT(0 <= tmp_FD_interface_data->mid_point_fine_node_idx && tmp_FD_interface_data->mid_point_fine_node_idx == interpolation_node_ngbd->get_neighbors(fine_node_idx_for_neighbor_quad).neighbor(face_dir + (face_dir%2 == 0 ? +1 : -1)));

  double xyz_midpoint[P4EST_DIM]; node_xyz_fr_n(tmp_FD_interface_data->mid_point_fine_node_idx, subrefined_p4est, subrefined_nodes, xyz_midpoint);
  const double mid_point_phi      = (*interp_phi)(xyz_midpoint);
  const bool no_past_mid_point    = signs_of_phi_are_different(phi_quad, mid_point_phi);
  const double &phi_this_side     = (no_past_mid_point ? phi_quad       : mid_point_phi);
  const double &phi_across        = (no_past_mid_point ? mid_point_phi  : phi_neighbor);
//  const p4est_locidx_t& fine_idx_this_side  = (no_past_mid_point ? tmp_FD_interface_data->quad_fine_node_idx      : tmp_FD_interface_data->mid_point_fine_node_idx);
//  const p4est_locidx_t& fine_idx_across     = (no_past_mid_point ? tmp_FD_interface_data->mid_point_fine_node_idx : tmp_FD_interface_data->neighbor_fine_node_idx);

  if(interp_phi_xxyyzz != NULL)
  {
    const double phi_dd_this_side = (no_past_mid_point ? (*interp_phi_xxyyzz).operator()(xyz_quad, face_dir/2)  : (*interp_phi_xxyyzz)(xyz_midpoint, face_dir/2));
    const double phi_dd_across    = (no_past_mid_point ? (*interp_phi_xxyyzz)(xyz_midpoint, face_dir/2)         : (*interp_phi_xxyyzz)(xyz_neighbor, face_dir/2));
    tmp_FD_interface_data->theta  = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, phi_dd_this_side, phi_dd_across, 0.5*dxyz_min[face_dir/2]);
  }
  else
    tmp_FD_interface_data->theta  = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, 0.5*dxyz_min[face_dir/2], 0.5*dxyz_min[face_dir/2]);
  tmp_FD_interface_data->theta = (phi_this_side > 0.0 ? 1.0 - tmp_FD_interface_data->theta : tmp_FD_interface_data->theta);
  tmp_FD_interface_data->theta = MAX(0.0, MIN(tmp_FD_interface_data->theta, 1.0));
  tmp_FD_interface_data->theta = 0.5*(tmp_FD_interface_data->theta + (no_past_mid_point ? 0.0 : 1.0));
  P4EST_ASSERT(0.0 <= tmp_FD_interface_data->theta && tmp_FD_interface_data->theta <= 1.0);

  if(cell_FD_interface_data != NULL)
  {
    which_interface_neighbor_t which_one = {quad_idx, face_dir};
    std::pair<map_of_interface_neighbors_t::iterator, bool> ret = cell_FD_interface_data->insert({which_one, *tmp_FD_interface_data}); // add it to the map so that future access is read from memory;
    P4EST_ASSERT(ret.second);
    return ret.first->second;
  }
  else
    return *tmp_FD_interface_data;
}

FD_interface_data my_p4est_interface_manager_t::get_interface_neighbor(const p4est_locidx_t& quad_idx, const u_char& face_dir) const
{
  return get_cell_interface_data_for(quad_idx, face_dir);
}

#ifdef DEBUG
int my_p4est_interface_manager_t::is_map_consistent()
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
    const which_interface_neighbor_t& this_one        = it->first;
    const FD_interface_data& this_interface_neighbor = it->second;
    // the neighbor is a local quad
    if(this_interface_neighbor.neighbor_quad_idx < p4est->local_num_quadrants)
    {
      which_interface_neighbor_t other_one                = {this_interface_neighbor.neighbor_quad_idx, (u_char)(this_one.face_dir + (this_one.face_dir%2 == 0 ? +1 : -1))};
      const FD_interface_data& other_interface_neighbor  = cell_FD_interface_data->at(other_one);
      it_is_alright = it_is_alright && this_interface_neighbor.is_consistent_with(other_interface_neighbor);

      if(!this_interface_neighbor.is_consistent_with(other_interface_neighbor))
        std::cerr << "Inconsistency found for quad " << this_one.loc_idx << " on proc " << p4est->mpirank << " which has quad " << other_one.loc_idx << " as a neighbor across the interface, on proc " << p4est->mpirank << std::endl;
    }
    else
    {
      const p4est_quadrant_t* ghost_nb_quad = p4est_const_quadrant_array_index(&ghost->ghosts, this_interface_neighbor.neighbor_quad_idx - p4est->local_num_quadrants);
      int rank_owner = first_rank_ghost_owner;
      while (this_interface_neighbor.neighbor_quad_idx >= p4est->local_num_quadrants + ghost->proc_offsets[rank_owner + 1]) { rank_owner++; }
      P4EST_ASSERT(rank_owner != p4est->mpirank);
      num_expected_replies += (senders[rank_owner] == 1 ? 0 : 1);
      senders[rank_owner] = 1;
      which_interface_neighbor_t other_one;

      other_one.loc_idx = ghost_nb_quad->p.piggy3.local_num;
      other_one.face_dir = this_one.face_dir + (this_one.face_dir%2 == 0 ? 1 : -1);
      map_of_query_interface_neighbors[rank_owner].push_back(other_one);
      map_of_mirrors[rank_owner].push_back(this_one);
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
    mpiret = MPI_Isend((void*)&map_of_query_interface_neighbors[rank][0], sizeof(which_interface_neighbor_t)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    mpi_query_requests.push_back(req);
  }

  std::map<int, std::vector<FD_interface_data> > map_of_responses; map_of_responses.clear();

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

        std::vector<FD_interface_data>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(which_interface_neighbor_t));

        for (size_t kk = 0; kk < response.size(); ++kk)
          response[kk] = cell_FD_interface_data->at(queries[kk]);

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*)&response[0], (response.size())*sizeof(FD_interface_data), MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
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
        P4EST_ASSERT(byte_count%sizeof(FD_interface_data) == 0);
        std::vector<FD_interface_data> reply_buffer (byte_count / sizeof(FD_interface_data));

        mpiret = MPI_Recv((void*)&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        for (size_t kk = 0; kk < reply_buffer.size(); ++kk) {
          which_interface_neighbor_t mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          which_interface_neighbor_t queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];
          it_is_alright = it_is_alright && cell_FD_interface_data->at(mirror).is_consistent_with(reply_buffer[kk]);

          if(!cell_FD_interface_data->at(mirror).is_consistent_with(reply_buffer[kk]))
            std::cerr << "Inconsistency found for quad " << mirror.loc_idx << " on proc " << p4est->mpirank << " which has quad " << queried.loc_idx << " as a neighbor across the interface, on proc " << status.MPI_SOURCE << std::endl;
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
