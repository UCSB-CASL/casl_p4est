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
 * SOURCE FILE REVISED AND CLEANED UP BY RAPHAEL EGAN (FEBRUARY 6, 2020)
 */

void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell(set_of_neighboring_quadrants& ngbd, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const char dir_xyz[P4EST_DIM]) const
{
  const p4est_quadrant_t *quad;
  if(quad_idx < p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
    quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);

  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  const p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);
  const p4est_qcoord_t quad_xyz[P4EST_DIM] = {DIM(quad->x, quad->y, quad->z)};
  const bool* pxyz = hierarchy->get_periodicity();

  /* (logical) coordinates of the (possibly hypothetical) quadrant of the same size to find: */
  p4est_qcoord_t ijk_nb[P4EST_DIM];
  /* tree index of the (possibly hypothetical) quadrant  to find, except if border case(s) */
  p4est_topidx_t nb_tree_idx = tree_idx;
  /* list of border cases (corner if size of P4EST_DIM, edge (in 3D if size 2), or face is size of 1, inside the same tree if size of 0) */
  std::vector<unsigned char> border_cases; border_cases.resize(0);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    P4EST_ASSERT(dir_xyz[dim] == -1 || dir_xyz[dim] == 0 || dir_xyz[dim] == 1);
    ijk_nb[dim] = quad_xyz[dim] + dir_xyz[dim]*size;
    if((quad_xyz[dim] == 0 && dir_xyz[dim] == -1) || ijk_nb[dim]== P4EST_ROOT_LEN)
      border_cases.push_back(dim);
  }
  P4EST_ASSERT(border_cases.size() <= P4EST_DIM);

  if(border_cases.size() == P4EST_DIM) // across tree corner
  {
    if(ORD(!pxyz[0] && myb->nxyztrees[0] == 1, !pxyz[1] && myb->nxyztrees[1] == 1, !pxyz[2] && myb->nxyztrees[2] == 1)) return; // no neighbor to be found here...
    const unsigned char local_corner_idx = SUMD((dir_xyz[0] == -1 ? 0 : 1), (dir_xyz[1] == -1 ? 0 : 2), (dir_xyz[2] == -1 ? 0 : 4));
    P4EST_ASSERT(local_corner_idx < P4EST_CHILDREN);
    const p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + local_corner_idx];
    if(corner == -1) return; // the corner exists indeed, but it does not connect with any other tree -> no neighbor to be found here

    const p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + local_corner_idx];
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim){
      P4EST_ASSERT(dir_xyz[dim] != 0);
      ijk_nb[dim] = (dir_xyz[dim] == 1 ? 0 : P4EST_ROOT_LEN - size);
    }
  }
#ifdef P4_TO_P8
  else if (border_cases.size() == 2) // across tree edge
  {
    // we go through two different trees in this case
    const unsigned char first_dim = border_cases[0];
    P4EST_ASSERT(first_dim < P4EST_DIM - 1 && dir_xyz[first_dim] != 0);
    const unsigned char first_face_dir = 2*first_dim + (dir_xyz[first_dim] == -1 ? 0 : 1);
    const p4est_topidx_t first_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + first_face_dir];
    if(!pxyz[first_dim] && first_tree_idx == tree_idx) return; // no other tree there, so nothing to find
    ijk_nb[first_dim] = (dir_xyz[first_dim] == -1 ? P4EST_ROOT_LEN - size : 0);

    // find the second direction
    const unsigned char second_dim = border_cases[1];
    P4EST_ASSERT(second_dim < P4EST_DIM && dir_xyz[second_dim] != 0);
    const unsigned char second_face_dir = 2*second_dim + (dir_xyz[second_dim] == -1 ? 0 : 1);
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*first_tree_idx + second_face_dir];
    if(!pxyz[second_dim] && nb_tree_idx == first_tree_idx) return; // no other tree there, so nothing to find
    ijk_nb[second_dim] = (dir_xyz[second_dim] == -1 ? P4EST_ROOT_LEN - size : 0);
  }
#endif
  else if(border_cases.size() == 1)
  {
    const unsigned char dim = border_cases[0];
    P4EST_ASSERT(dim < P4EST_DIM && dir_xyz[dim] != 0);
    const unsigned char face_dir = 2*dim + (dir_xyz[dim] == -1 ? 0 : 1);
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + face_dir];
    if(!pxyz[dim] && nb_tree_idx == tree_idx) return; // no other tree there, so nothing to find
    ijk_nb[dim] = (dir_xyz[dim] == -1 ? P4EST_ROOT_LEN - size : 0);
  }

  /* find the constructed neighbor cell of the same size */
  int ind = 0;
  while( hierarchy->trees[nb_tree_idx][ind].level != quad->level && hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF )
  {
    p4est_qcoord_t half_size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool i_search = ( ijk_nb[0] >= hierarchy->trees[nb_tree_idx][ind].imin + half_size );
    bool j_search = ( ijk_nb[1] >= hierarchy->trees[nb_tree_idx][ind].jmin + half_size );
#ifdef P4_TO_P8
    bool k_search = ( ijk_nb[2] >= hierarchy->trees[nb_tree_idx][ind].kmin + half_size );
#endif
    ind = hierarchy->trees[nb_tree_idx][ind].child + SUMD(i_search, 2*j_search, 4*k_search);
  }

  /* now find the children of this constructed cell in the desired direction and add them to the list */
  find_neighbor_cells_of_cell_recursive(ngbd, nb_tree_idx, ind, dir_xyz);
}

void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive(set_of_neighboring_quadrants& ngbd, const p4est_topidx_t& tr, const int& ind, const char dir_xyz[P4EST_DIM]) const
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
    }
    return;
  }

  for (char child_x = (dir_xyz[0] == 0 ? -1 : -dir_xyz[0]); child_x < (dir_xyz[0] == 0 ? 2 : -dir_xyz[0] + 1); child_x += 2)    // child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
    for (char child_y = (dir_xyz[1] == 0 ? -1 : -dir_xyz[1]); child_y < (dir_xyz[1] == 0 ? 2 : -dir_xyz[1] + 1); child_y += 2)  // child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
#ifdef P4_TO_P8
      for (char child_z = (dir_xyz[2] == 0 ? -1 : -dir_xyz[2]); child_z < (dir_xyz[2] == 0 ? 2 : -dir_xyz[2] + 1); child_z += 2)// child cell in the mirror direction if search is nonzero, sweep {-1, 1} if zero
#endif
        find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + SUMD((child_x == -1 ? 0 : 1), (child_y == -1 ? 0 : 2), (child_z == -1 ? 0 : 4)), dir_xyz);
}

double interpolate_cell_field_at_node(const p4est_locidx_t& node_idx, const my_p4est_cell_neighbors_t* c_ngbd, const my_p4est_node_neighbors_t* n_ngbd, const Vec cell_field, const BoundaryConditionsDIM* bc, const Vec phi)
{
  PetscErrorCode ierr;

  const p4est_t* p4est = c_ngbd->get_p4est();
  const p4est_nodes_t* nodes = n_ngbd->get_nodes();
  const my_p4est_brick_t* brick = c_ngbd->get_brick();
  const double * tree_dimensions = c_ngbd->get_tree_dimensions();


  double xyz_node[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);

  SC_ASSERT ((size_t) node_idx < nodes->indep_nodes.elem_count);
  const p4est_indep_t *node = (p4est_indep_t*) (nodes->indep_nodes.array + ((size_t) node_idx)*nodes->indep_nodes.elem_size);

  if(bc != NULL && is_node_Wall(p4est, node) && bc->wallType(xyz_node) == DIRICHLET)
    return bc->wallValue(xyz_node);

  const double *cell_field_p;
  ierr = VecGetArrayRead(cell_field, &cell_field_p); CHKERRXX(ierr);

  set_of_neighboring_quadrants ngbd_tmp;
  double scaling = n_ngbd->gather_neighbor_cells_of_node(ngbd_tmp, c_ngbd, node_idx, true);
  scaling *= 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]));

  matrix_t A;
  A.resize(1, 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2);
  std::vector<double> rhs; rhs.resize(0);
  std::set<int64_t> nb[P4EST_DIM];

  const double min_weight     = 1e-6;
  const double inv_max_weight = 1e-6;

  const double *phi_p = NULL;
  P4EST_ASSERT(bc == NULL || bc->interfaceType() == NOINTERFACE || phi != NULL); // if we have BCs for an interface, we need phi!
  if(phi != NULL){
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr); }

  for(set_of_neighboring_quadrants::const_iterator it = ngbd_tmp.begin(); it != ngbd_tmp.end(); ++it)
  {
    const p4est_locidx_t qm_idx = it->p.piggy3.local_num;
    /* check if quadrant is well defined */
    bool neumann_is_neg = false;
    double phi_q = -1.0;
    if(phi_p != NULL)
    {
      phi_q = 0.0;
      for(unsigned char i = 0; i < P4EST_CHILDREN; ++i)
      {
        double tmp = phi_p[nodes->local_nodes[P4EST_CHILDREN*qm_idx + i]];
        neumann_is_neg = neumann_is_neg || tmp < 0.0;
        phi_q += tmp;
      }
      phi_q /= (double) P4EST_CHILDREN;
    }

    if(bc == NULL || bc->interfaceType() == NOINTERFACE || (bc->interfaceType(xyz_node) == DIRICHLET && phi_q < 0.0) || (bc->interfaceType(xyz_node) == NEUMANN && neumann_is_neg))
    {
      double xyz_t[P4EST_DIM];
      int64_t logical_qcoord_diff[P4EST_DIM];
      rel_qxyz_quad_fr_node(p4est, *it, xyz_node, node, tree_dimensions, brick, xyz_t, logical_qcoord_diff);

      for(unsigned char i = 0; i < P4EST_DIM; ++i)
        xyz_t[i] /= scaling;

      const double weight = MAX(min_weight, 1./MAX(inv_max_weight, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

      A.set_value(rhs.size(), 0,                1                 * weight);
      A.set_value(rhs.size(), 1,                xyz_t[0]          * weight);
      A.set_value(rhs.size(), 2,                xyz_t[1]          * weight);
#ifdef P4_TO_P8
      A.set_value(rhs.size(), 3,                xyz_t[2]          * weight);
#endif
      A.set_value(rhs.size(), 1 +   P4EST_DIM,  xyz_t[0]*xyz_t[0] * weight);
      A.set_value(rhs.size(), 2 +   P4EST_DIM,  xyz_t[0]*xyz_t[1] * weight);
#ifdef P4_TO_P8
      A.set_value(rhs.size(), 3 +   P4EST_DIM,  xyz_t[0]*xyz_t[2] * weight);
#endif
      A.set_value(rhs.size(), 1 + 2*P4EST_DIM,  xyz_t[1]*xyz_t[1] * weight);
#ifdef P4_TO_P8
      A.set_value(rhs.size(), 2 + 2*P4EST_DIM,  xyz_t[1]*xyz_t[2] * weight);
      A.set_value(rhs.size(),     3*P4EST_DIM,  xyz_t[2]*xyz_t[2] * weight);
#endif
      rhs.push_back(cell_field_p[qm_idx]*weight);

      for(unsigned char d = 0; d < P4EST_DIM; ++d)
        nb[d].insert(logical_qcoord_diff[d]);
    }
  }

  if(phi != NULL){
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(cell_field, &cell_field_p); CHKERRXX(ierr);

  if(rhs.size() == 0)
    return 0.0; // no valid neighbor (way into the positive domain for instance)

  A.scale_by_maxabs(rhs);

  return solve_lsqr_system(A, rhs, DIM(nb[0].size(), nb[1].size(), nb[2].size()));
}
