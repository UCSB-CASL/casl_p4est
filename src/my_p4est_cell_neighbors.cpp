#ifdef P4_TO_P8
#include "my_p8est_cell_neighbors.h"
#include "my_p8est_node_neighbors.h"
#include "my_p8est_solve_lsqr.h"
#else
#include "my_p4est_cell_neighbors.h"
#include "my_p4est_node_neighbors.h"
#include "my_p4est_solve_lsqr.h"
#endif

/*
 * + SOURCE FILE REVISED, DEBUGGED (existing bug in 3D) AND CLEANED UP BY on (FEBRUARY 6, 2020) [Raphael Egan]
 * + added optional functionality to calculate smallest logical size of cells on-the-fly, on April 4, 2020 [Raphael Egan]
 * + cleaned up even more to alleviate duplication with find_neighbor_cell_of_node, on June 2, 2020 [Raphael Egan]
 */

void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const char dir_xyz[P4EST_DIM], p4est_qcoord_t *smallest_quad_size) const
{
  const p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

  /* construct the node that would be the center of the (possibly hypothetical) quadrant of the same size to find in the direction queried by the user */
  const p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);
  p4est_indep_t node; node.level = P4EST_MAXLEVEL; node.p.which_tree = tree_idx;
  node.x = quad->x + size/2 + dir_xyz[0]*size;
  node.y = quad->y + size/2 + dir_xyz[1]*size;
#ifdef P4_TO_P8
  node.z = quad->z + size/2 + dir_xyz[2]*size;
#endif
  if(!is_node_in_domain(node, myb, p4est->connectivity))
    return; // no neighbor to find there --> return

  /* find the constructed neighbor HierarchyCell of the same size */
  const p4est_topidx_t owning_tree_idx = node.p.which_tree;
  p4est_quadrant_t neighbor_quad; neighbor_quad.level = quad->level;
  neighbor_quad.x = node.x - size/2;
  neighbor_quad.y = node.y - size/2;
#ifdef P4_TO_P8
  neighbor_quad.z = node.z - size/2;
#endif
  int ind = hierarchy->get_index_of_hierarchy_cell_matching_or_containing_quad(&neighbor_quad, owning_tree_idx);

  /* now find the children of this constructed cell in the desired direction and add them to the list */
  find_neighbor_cells_of_cell_recursive(ngbd, owning_tree_idx, ind, dir_xyz, smallest_quad_size);
}

void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive(set_of_neighboring_quadrants& ngbd, const p4est_topidx_t& tr, const int& ind, const char dir_xyz[P4EST_DIM], p4est_qcoord_t *smallest_quad_size) const
{
  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return; // -> "space-filling" quadrant in the hierarchy, out of the knowledge of local domain

    const p4est_locidx_t locid = hierarchy->trees[tr][ind].quad;

    p4est_quadrant_t quad;
    quad.p.piggy3.local_num   = locid;
    quad.p.piggy3.which_tree  = tr;
    // --> the piggy3 needs to be filled and valid for the following "find" to work as expected!
    if(ngbd.find(quad) == ngbd.end()) // we copy the rest of the quadrant's data only if needed (if not in the set yet), and then we insert it
    {
      if (locid < p4est->local_num_quadrants) {
        p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
        quad = *p4est_quadrant_array_index(&tree->quadrants, locid - tree->quadrants_offset);
      } else
        quad = *p4est_quadrant_array_index(&ghost->ghosts, locid - p4est->local_num_quadrants);

      quad.p.piggy3.local_num   = locid;
      quad.p.piggy3.which_tree  = tr;     // we write in the piggy3 again because the above copy has actually overwritten them...
#ifdef DEBUG
      std::pair<set_of_neighboring_quadrants::iterator, bool> ret = ngbd.insert(quad);
      P4EST_ASSERT(ret.second); // dummy check
#else
      ngbd.insert(quad);
#endif
      if(smallest_quad_size != NULL)
        *smallest_quad_size = MIN(*smallest_quad_size, P4EST_QUADRANT_LEN(quad.level));
    }
    return;
  }

  for (char child_x = (dir_xyz[0] == 0 ? -1 : -dir_xyz[0]); child_x < (dir_xyz[0] == 0 ? 2 : -dir_xyz[0] + 1); child_x += 2)    // child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
    for (char child_y = (dir_xyz[1] == 0 ? -1 : -dir_xyz[1]); child_y < (dir_xyz[1] == 0 ? 2 : -dir_xyz[1] + 1); child_y += 2)  // child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
#ifdef P4_TO_P8
      for (char child_z = (dir_xyz[2] == 0 ? -1 : -dir_xyz[2]); child_z < (dir_xyz[2] == 0 ? 2 : -dir_xyz[2] + 1); child_z += 2)// child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
#endif
        find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + SUMD((child_x == -1 ? 0 : 1), (child_y == -1 ? 0 : 2), (child_z == -1 ? 0 : 4)), dir_xyz, smallest_quad_size);
}

p4est_qcoord_t my_p4est_cell_neighbors_t::gather_neighbor_cells_of_cell(const p4est_quadrant_t& quad_with_correct_local_num_in_piggy3, set_of_neighboring_quadrants& ngbd, const bool& add_second_degree_neighbors, const bool *no_search) const
{
  /* gather the neighborhood */
  ngbd.insert(quad_with_correct_local_num_in_piggy3);
  p4est_qcoord_t smallest_quad_size = P4EST_QUADRANT_LEN(quad_with_correct_local_num_in_piggy3.level);
  const p4est_locidx_t& quad_idx = quad_with_correct_local_num_in_piggy3.p.piggy3.local_num;
  const p4est_topidx_t& tree_idx = quad_with_correct_local_num_in_piggy3.p.piggy3.which_tree;
  set_of_neighboring_quadrants *close_ngbd = &ngbd;
  if(add_second_degree_neighbors)
    close_ngbd = new set_of_neighboring_quadrants; // we need to get the close neighbors separately first in that case --> use a temporary buffer set

  char search_range_low[P4EST_DIM]  = {DIM(-1, -1, -1)};
  char search_range_high[P4EST_DIM] = {DIM( 1,  1,  1)};
  if(no_search != NULL)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      if(no_search[dim])
        search_range_low[dim] = search_range_high[dim] = 0;

  for(char i = search_range_low[0]; i <= search_range_high[0]; ++i)
    for(char j = search_range_low[1]; j <= search_range_high[1]; ++j)
#ifdef P4_TO_P8
      for(char k = search_range_low[2]; k <= search_range_high[2]; ++k)
#endif
      {
        if(ANDD(i == 0, j == 0, k == 0)) // no need to search for that one, of course...
          continue;
        find_neighbor_cells_of_cell(*close_ngbd, quad_idx, tree_idx, DIM(i, j, k), &smallest_quad_size);
        if(add_second_degree_neighbors) // in that case, loop through elements of the close_ngbd, insert them in the final set and repeat the operation for those elements
        {
          for (set_of_neighboring_quadrants::const_iterator it = close_ngbd->begin(); it != close_ngbd->end(); ++it)
          {
            ngbd.insert(*it); // the smallest_quad_size was already done in the operation here above, if needed/desired
            for(char ii = search_range_low[0]; ii <= search_range_high[0]; ++ii)
              for(char jj = search_range_low[1]; jj <= search_range_high[1]; ++jj)
#ifdef P4_TO_P8
                for(char kk = search_range_low[2]; kk <= search_range_high[2]; ++kk)
#endif
                {
                  if(ANDD(ii == 0, jj == 0, kk == 0))
                    continue;
                  find_neighbor_cells_of_cell(ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(ii, jj, kk));
                }
          }
          close_ngbd->clear(); // clear the temporary buffer before going on, to avoid useless steps thereafter
        }
      }

  if(add_second_degree_neighbors)
    delete close_ngbd;

  return smallest_quad_size;
}

p4est_qcoord_t my_p4est_cell_neighbors_t::gather_neighbor_cells_of_node(const p4est_locidx_t& node_idx, const p4est_nodes_t* nodes, set_of_neighboring_quadrants& cell_neighbors, const bool& add_second_degree_neighbors) const
{
#ifdef CASL_THROWS
  bool at_least_one_direct_neighbor_is_local = false;
#endif
  p4est_qcoord_t smallest_quad_size = P4EST_ROOT_LEN;
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  for(char i = -1; i < 2; i += 2)
    for(char j = -1; j < 2; j += 2)
#ifdef P4_TO_P8
      for(char k = -1; k < 2; k += 2)
#endif
      {
        hierarchy->find_neighbor_cell_of_node(node_idx, nodes, DIM(i, j, k), quad_idx, tree_idx);
        if(quad_idx != NOT_A_VALID_QUADRANT)
        {
          p4est_quadrant_t quad;
          if(quad_idx < p4est->local_num_quadrants)
          {
            p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
            quad = *p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
          }
          else
            quad = *p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

          quad.p.piggy3.local_num = quad_idx;
          quad.p.piggy3.which_tree = tree_idx;

#ifdef CASL_THROWS
          at_least_one_direct_neighbor_is_local = at_least_one_direct_neighbor_is_local || quad_idx < p4est->local_num_quadrants;
#endif

          cell_neighbors.insert(quad);
          smallest_quad_size = MIN(smallest_quad_size, P4EST_QUADRANT_LEN(quad.level));
          if(add_second_degree_neighbors)
          {
            // fetch an extra layer in all nonzero directions and their possible combinations
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx, DIM(i, 0, 0));
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx, DIM(0, j, 0));
#ifdef P4_TO_P8
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx,     0, 0, k );
#endif
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx, DIM(i, j, 0));
#ifdef P4_TO_P8
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx,     i, 0, k );
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx,     0, j, k );
            find_neighbor_cells_of_cell(cell_neighbors, quad_idx, tree_idx,     i, j, k );
#endif
          }
        }
      }

#ifdef CASL_THROWS
  if(!at_least_one_direct_neighbor_is_local) {
    PetscErrorCode ierr = PetscPrintf(p4est->mpicomm, "Warning !! my_p4est_cell_neighbors_t::gather_neighbor_cells_of_node(): the node has no direct local neighbor quadrant."); CHKERRXX(ierr); }
#endif

  return smallest_quad_size;
}


double interpolate_cell_field_at_node(const p4est_locidx_t& node_idx, const my_p4est_cell_neighbors_t* c_ngbd, const my_p4est_node_neighbors_t* n_ngbd, const Vec cell_field, const BoundaryConditionsDIM* bc, const Vec phi)
{
  const p4est_t* p4est = c_ngbd->get_p4est();
  const p4est_nodes_t* nodes = n_ngbd->get_nodes();
  const double * tree_dimensions = c_ngbd->get_tree_dimensions();

  double xyz_node[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
  const p4est_indep_t *node = (p4est_indep_t*) sc_const_array_index(&nodes->indep_nodes, node_idx);

  if(bc != NULL && is_node_Wall(p4est, node) && bc->wallType(xyz_node) == DIRICHLET)
    return bc->wallValue(xyz_node);

  /* gather the neighborhood and get the (logical) size of the smallest quadrant in the first-degree neighborhood */
  set_of_neighboring_quadrants cell_ngbd; cell_ngbd.clear();
  const p4est_qcoord_t logical_size_smallest_first_degree_cell_neighbor = c_ngbd->gather_neighbor_cells_of_node(node_idx, nodes, cell_ngbd, true);
  const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double)logical_size_smallest_first_degree_cell_neighbor/(double) P4EST_ROOT_LEN;

  PetscErrorCode ierr;
  const double *cell_field_p;
  ierr = VecGetArrayRead(cell_field, &cell_field_p); CHKERRXX(ierr);
  const double *phi_p = NULL;
  P4EST_ASSERT(bc == NULL || bc->interfaceType() == NOINTERFACE || phi != NULL); // if we have BCs for an interface, we need node_sampled_phi!
  if(phi != NULL){
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr); }

  const double to_return = get_lsqr_interpolation_at_node(xyz_node, c_ngbd, cell_ngbd, scaling, cell_field_p, bc, n_ngbd, phi_p);

  ierr = VecRestoreArrayRead(cell_field, &cell_field_p); CHKERRXX(ierr);
  if(phi != NULL){
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr); }

  return to_return;
}

double get_lsqr_interpolation_at_node(const double xyz_node[P4EST_DIM], const my_p4est_cell_neighbors_t* ngbd_c, const set_of_neighboring_quadrants &ngbd_of_cells, const double &scaling,
                                      const double* cell_sampled_field_p, const BoundaryConditionsDIM* bc, const my_p4est_node_neighbors_t* ngbd_n, const double* node_sampled_phi_p,
                                      const u_char &degree, const double &thresh_condition_number, linear_combination_of_dof_t* interpolator)
{
  matrix_t A;
  A.resize(1, 1 + (degree > 0 ? P4EST_DIM : 0) + (degree  > 1 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0));
  std::vector<double> lsqr_rhs; lsqr_rhs.resize(0);
  if(interpolator != NULL)
    interpolator->clear();
  std::set<int64_t> set_of_qcoord[P4EST_DIM];

  const double min_weight     = 1.0e-6;
  const double inv_max_weight = 1.0e-6;
  P4EST_ASSERT(ngbd_of_cells.size() > 0);

  for(set_of_neighboring_quadrants::const_iterator it = ngbd_of_cells.begin(); it != ngbd_of_cells.end(); ++it)
    if(bc == NULL || quadrant_value_is_well_defined(*bc, ngbd_n->get_p4est(), ngbd_n->get_ghost(), ngbd_n->get_nodes(), it->p.piggy3.local_num, it->p.piggy3.which_tree, node_sampled_phi_p))
    {
      /* the value is well-defined we can use it */
      const p4est_locidx_t &quad_idx = it->p.piggy3.local_num;

      double xyz_t[P4EST_DIM];
      int64_t qcoord_quad[P4EST_DIM];
      rel_xyz_quad_fr_point(ngbd_c->get_p4est(), *it, xyz_node, ngbd_c->get_brick(), xyz_t, qcoord_quad);

      for(u_char i = 0; i < P4EST_DIM; ++i)
      {
        xyz_t[i] /= scaling;
        set_of_qcoord[i].insert(qcoord_quad[i]);
      }

      const double weight = MAX(min_weight, 1./MAX(inv_max_weight, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

      u_char col_idx = 0;
      // constant term
      A.set_value(lsqr_rhs.size(), col_idx++, weight);
      // linear terms
      if(degree > 0)
        for (u_char uu = 0; uu < P4EST_DIM; ++uu)
          A.set_value(lsqr_rhs.size(), col_idx++, xyz_t[uu]*weight);
      // quadratic terms
      if(degree > 1)
        for (u_char uu = 0; uu < P4EST_DIM; ++uu)
          for (u_char vv = uu; vv < P4EST_DIM; ++vv)
            A.set_value(lsqr_rhs.size(), col_idx++, xyz_t[uu]*xyz_t[vv]*weight);

      lsqr_rhs.push_back(cell_sampled_field_p[quad_idx]*weight);

      if(interpolator != NULL)
        interpolator->add_term(quad_idx, weight);
    }

  P4EST_ASSERT((bc == NULL ? 0 < A.num_rows() && (size_t) A.num_rows() == ngbd_of_cells.size() : (size_t)A.num_rows() <= ngbd_of_cells.size()));

  if(lsqr_rhs.size() == 0)
  {
    if(interpolator != NULL)
      interpolator->clear();
    return 0.0; // no valid neighbor (way into the positive domain for instance)
  }

  const double abs_max = A.scale_by_maxabs(lsqr_rhs);
  if(interpolator != NULL)
    *interpolator /= abs_max;
  P4EST_ASSERT(interpolator == NULL || interpolator->size() > 0);
  std::vector<double>* interp_weights = (interpolator == NULL ? NULL : new std::vector<double>(interpolator->size()));

  const double value_to_return = solve_lsqr_system(A, lsqr_rhs, DIM(set_of_qcoord[0].size(), set_of_qcoord[1].size(), set_of_qcoord[2].size()), degree, 0, interp_weights, thresh_condition_number);
  if(interpolator != NULL)
    (*interpolator) *= *interp_weights;

  return value_to_return;
}

void get_lsqr_cell_gradient_operator_at_point(const double xyz_node[P4EST_DIM], const my_p4est_cell_neighbors_t* ngbd_c, const set_of_neighboring_quadrants &ngbd_of_cells, const double &scaling,
                                              linear_combination_of_dof_t grad_operator[], const bool& is_point_quad_center, const p4est_locidx_t& idx_of_quad_center)
{
  P4EST_ASSERT(ngbd_of_cells.size() > 0);
  P4EST_ASSERT(!is_point_quad_center || idx_of_quad_center >= 0);
#ifdef CASL_THROWS
  if(ngbd_of_cells.size() < 1 + P4EST_DIM)
    throw std::invalid_argument("get_lsqr_cell_gradient_operator_at_point() : not enough neighbor cells to build a gradient");
#endif

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    grad_operator[dim].clear();

  matrix_t A(ngbd_of_cells.size() - (is_point_quad_center ? 1 : 0), (is_point_quad_center ? 0 : 1 ) + P4EST_DIM);

  std::set<int64_t> set_of_qcoord[P4EST_DIM];
  int row_idx = 0;
  for(set_of_neighboring_quadrants::const_iterator it = ngbd_of_cells.begin(); it != ngbd_of_cells.end(); ++it)
  {
    if(is_point_quad_center && it->p.piggy3.local_num == idx_of_quad_center)
      continue; // done at the very end for that term
    double xyz_t[P4EST_DIM];
    int64_t qcoord_quad[P4EST_DIM];
    rel_xyz_quad_fr_point(ngbd_c->get_p4est(), *it, xyz_node, ngbd_c->get_brick(), xyz_t, qcoord_quad);

    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      grad_operator[dim].add_term(it->p.piggy3.local_num, 1.0);
      set_of_qcoord[dim].insert(qcoord_quad[dim]);
    }

    // constant term
    u_char col_idx = 0;
    if(!is_point_quad_center)
      A.set_value(row_idx, col_idx++, 1.0);
    // linear terms
    for (u_char uu = 0; uu < P4EST_DIM; ++uu)
      A.set_value(row_idx, col_idx++, xyz_t[uu]/scaling);

    row_idx++;
  }

  matrix_t AtA, inv_AtA;
  A.mtm_product(AtA);
  const bool is_inversion_successful = solve_cholesky(AtA, NULL, NULL, 0, 1e4, NULL, &inv_AtA);

  if(!is_inversion_successful)
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      if(set_of_qcoord[dim].size() < 2)
        throw std::runtime_error(std::string("get_lsqr_cell_gradient_operator_at_point : you have less than two linearly independent neighbors along cartesian direction ")
                                 + std::string(dim == dir::x ? "x" : ONLY3D(OPEN_PARENTHESIS dim == dir::y ? ) "y" ONLY3D( : "z" CLOSE_PARENTHESIS)));
    throw std::runtime_error("get_lsqr_cell_gradient_operator_at_point : the inversion of the lsqr system failed.");
  }

  matrix_t operator_coeffs;
  A.matrix_product(inv_AtA, operator_coeffs);

  double sum_of_coeffs[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};

  for (int term = 0; term < A.num_rows(); ++term)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      grad_operator[dim][term].weight = operator_coeffs.get_value(term, (is_point_quad_center ? 0 : 1) + dim)/scaling;
      sum_of_coeffs[dim] += grad_operator[dim][term].weight;
    }

  if(is_point_quad_center)
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      grad_operator[dim].add_term(idx_of_quad_center, -sum_of_coeffs[dim]);

  return;
}

