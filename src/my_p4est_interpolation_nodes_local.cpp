#ifdef P4_TO_P8
#include "my_p8est_interpolation_nodes_local.h"
#else
#include "my_p4est_interpolation_nodes_local.h"
#endif

void my_p4est_interpolation_nodes_local_t::initialize(p4est_locidx_t n)
{
  // clear
  for (short i_quad = 0; i_quad < P4EST_CHILDREN; ++i_quad)
  {
    quad_idx[i_quad] = NOT_A_VALID_QUADRANT;
    tree_idx[i_quad] = -1;
  }

  // find neighboring quadrants
#ifdef P4_TO_P8
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1, -1, quad_idx[dir::v_mmm], tree_idx[dir::v_mmm]);
  node_neighbors->find_neighbor_cell_of_node(n, -1,  1, -1, quad_idx[dir::v_mpm], tree_idx[dir::v_mpm]);
  node_neighbors->find_neighbor_cell_of_node(n,  1, -1, -1, quad_idx[dir::v_pmm], tree_idx[dir::v_pmm]);
  node_neighbors->find_neighbor_cell_of_node(n,  1,  1, -1, quad_idx[dir::v_ppm], tree_idx[dir::v_ppm]);
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1,  1, quad_idx[dir::v_mmp], tree_idx[dir::v_mmp]);
  node_neighbors->find_neighbor_cell_of_node(n, -1,  1,  1, quad_idx[dir::v_mpp], tree_idx[dir::v_mpp]);
  node_neighbors->find_neighbor_cell_of_node(n,  1, -1,  1, quad_idx[dir::v_pmp], tree_idx[dir::v_pmp]);
  node_neighbors->find_neighbor_cell_of_node(n,  1,  1,  1, quad_idx[dir::v_ppp], tree_idx[dir::v_ppp]);
#else
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1, quad_idx[dir::v_mmm], tree_idx[dir::v_mmm]);
  node_neighbors->find_neighbor_cell_of_node(n, -1,  1, quad_idx[dir::v_mpm], tree_idx[dir::v_mpm]);
  node_neighbors->find_neighbor_cell_of_node(n,  1, -1, quad_idx[dir::v_pmm], tree_idx[dir::v_pmm]);
  node_neighbors->find_neighbor_cell_of_node(n,  1,  1, quad_idx[dir::v_ppm], tree_idx[dir::v_ppm]);
#endif

  // get info about quadrants (coordinates and level)
  for (short i_quad = 0; i_quad < P4EST_CHILDREN; ++i_quad)
  {
    if (quad_idx[i_quad] != NOT_A_VALID_QUADRANT)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx[i_quad]);

      p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[tree_idx[i_quad]*P4EST_CHILDREN + 0];
      p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[tree_idx[i_quad]*P4EST_CHILDREN + P4EST_CHILDREN-1];

      double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
      double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
      double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  #ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
      double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  #endif

      p4est_locidx_t quad_idx_tree = quad_idx[i_quad] - tree->quadrants_offset;

      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx_tree);

      level_of_quad[i_quad] = quad->level;
      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (tree_xmax-tree_xmin)*dmin; xyz_quad_min[i_quad*P4EST_DIM + 0] = (tree_xmax-tree_xmin)*(double)quad->x/(double)P4EST_ROOT_LEN + tree_xmin;  xyz_quad_max[i_quad*P4EST_DIM + 0] = xyz_quad_min[i_quad*P4EST_DIM + 0] + dx;
      double dy = (tree_ymax-tree_ymin)*dmin; xyz_quad_min[i_quad*P4EST_DIM + 1] = (tree_ymax-tree_ymin)*(double)quad->y/(double)P4EST_ROOT_LEN + tree_ymin;  xyz_quad_max[i_quad*P4EST_DIM + 1] = xyz_quad_min[i_quad*P4EST_DIM + 1] + dy;
#ifdef P4_TO_P8
      double dz = (tree_zmax-tree_zmin)*dmin; xyz_quad_min[i_quad*P4EST_DIM + 2] = (tree_zmax-tree_zmin)*(double)quad->z/(double)P4EST_ROOT_LEN + tree_zmin;  xyz_quad_max[i_quad*P4EST_DIM + 2] = xyz_quad_min[i_quad*P4EST_DIM + 2] + dz;
#endif
    }
  }
}

#ifdef P4_TO_P8
double my_p4est_interpolation_nodes_local_t::interpolate(double x, double y, double z)
#else
double my_p4est_interpolation_nodes_local_t::interpolate(double x, double y)
#endif
{
  PetscErrorCode ierr;

#ifdef P4_TO_P8
  double xyz [] = { x, y, z };
#else
  double xyz [] = { x, y };
#endif

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz[i] > xyz_max[i]) xyz[i] = is_periodic(p4est,i) ? xyz[i]-(xyz_max[i]-xyz_min[i]) : xyz_max[i];
    if (xyz[i] < xyz_min[i]) xyz[i] = is_periodic(p4est,i) ? xyz[i]+(xyz_max[i]-xyz_min[i]) : xyz_min[i];
  }

  // find to which quadrant the point belongs to
  short which_quadrant = -1;
  short current_level = 0;
  bool is_point_in_quadrant;

  for (short i_quad = 0; i_quad < P4EST_CHILDREN; ++i_quad)
    if (quad_idx[i_quad] != NOT_A_VALID_QUADRANT)
    {
      is_point_in_quadrant = true;

      for (short i=0; i<P4EST_DIM; i++)
      {
        if (xyz[i] < xyz_quad_min[i_quad*P4EST_DIM + i]-0.1*eps) { is_point_in_quadrant = false; break; }
        if (xyz[i] > xyz_quad_max[i_quad*P4EST_DIM + i]+0.1*eps) { is_point_in_quadrant = false; break; }
      }

      if (is_point_in_quadrant && level_of_quad[i_quad] > current_level)
      {
        which_quadrant = i_quad;
        current_level = level_of_quad[i_quad];
      }
    }
  
  if (which_quadrant == -1)
    throw std::invalid_argument("[ERROR]: Point does not belong to any neighbouring quadrant.");

  // get pointers to inputs if necessary
  if (is_input_in_vec)
  {
    ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

    if (method == quadratic || method == quadratic_non_oscillatory)
    {
      ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }
  }

  int node_offset = quad_idx[which_quadrant]*P4EST_CHILDREN;

  for (short i = 0; i<P4EST_CHILDREN; i++)
  {
    p4est_locidx_t node_idx = nodes->local_nodes[node_offset + i];
    f[i] = Fi_p[node_idx];
  }

  if (method == quadratic || method == quadratic_non_oscillatory)
  {
    for (short j = 0; j<P4EST_CHILDREN; j++)
    {
      p4est_locidx_t node_idx = nodes->local_nodes[node_offset + j];

      fxx[j] = Fxx_p[node_idx];
      fyy[j] = Fyy_p[node_idx];
#ifdef P4_TO_P8
      fzz[j] = Fzz_p[node_idx];
#endif
    }
  }

  // restore arrays
  if (is_input_in_vec)
  {
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

    if (method == quadratic || method == quadratic_non_oscillatory) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }
  }

  double value=0;


  interp.initialize(xyz_quad_min[which_quadrant*P4EST_DIM + 0], xyz_quad_max[which_quadrant*P4EST_DIM + 0],
                    xyz_quad_min[which_quadrant*P4EST_DIM + 1], xyz_quad_max[which_quadrant*P4EST_DIM + 1],
                    #ifdef P4_TO_P8
                    xyz_quad_min[which_quadrant*P4EST_DIM + 2], xyz_quad_max[which_quadrant*P4EST_DIM + 2],
                    1,
                    #endif
                    1,1);

  if (method == linear) {
#ifdef P4_TO_P8
    value = interp.linear(f, xyz[0], xyz[1], xyz[2]);
#else
    value = interp.linear(f, xyz[0], xyz[1]);
#endif
  } else if (method == quadratic) {
#ifdef P4_TO_P8
    value = interp.quadratic(f, fxx, fyy, fzz,  xyz[0], xyz[1], xyz[2]);
#else
    value = interp.quadratic(f, fxx, fyy,       xyz[0], xyz[1]);
#endif
  }

  return value;
}
