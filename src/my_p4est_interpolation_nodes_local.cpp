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
    level_of_quad[i_quad] = -2;

    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      xyz_quad_min[i_quad*P4EST_DIM + dim] = 0.0;
      xyz_quad_max[i_quad*P4EST_DIM + dim] = 0.0;
    }

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

      p4est_quadrant_t *quad;

      if(quad_idx[i_quad] < p4est->local_num_quadrants)
      {
//        tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx[i_quad]);
        quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx[i_quad]-tree->quadrants_offset);
      }
      else /* in the ghost layer */
      {
        quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx[i_quad]-p4est->local_num_quadrants);
        /* TODO: make sure that a ghost quadrant and a tree are consistent */
      }

//      p4est_locidx_t quad_idx_tree = quad_idx[i_quad] - tree->quadrants_offset;

//      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx_tree);

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
double my_p4est_interpolation_nodes_local_t::interpolate(double x, double y, double z) const
#else
double my_p4est_interpolation_nodes_local_t::interpolate(double x, double y) const
#endif
{
  PetscErrorCode ierr;

  const double *Fi_p_;
  const double *Fxx_p_, *Fyy_p_;
#ifdef P4_TO_P8
  const double *Fzz_p_;
#endif

  double f  [P4EST_CHILDREN];
  double fxx[P4EST_CHILDREN];
  double fyy[P4EST_CHILDREN];
#ifdef P4_TO_P8
  double fzz[P4EST_CHILDREN];
#endif

#ifdef P4_TO_P8
  double xyz [] = { x, y, z };
#else
  double xyz [] = { x, y };
#endif

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++)
  {
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
  {
    std::cout << xyz[0] << ", " << xyz[1]
                       #ifdef P4_TO_P8
                        << ", " << xyz[2]
                       #endif
              << std::endl;

    for (short i_quad = 0; i_quad < P4EST_CHILDREN; ++i_quad)
    {

      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx[i_quad]);
      std::cout << tree_idx[i_quad] << " : " << quad_idx[i_quad]- tree->quadrants_offset << " : " << level_of_quad[i_quad] << " : " << tree->quadrants.elem_count << " : "
                << xyz_quad_min[i_quad*P4EST_DIM + 0] << ", " << xyz_quad_min[i_quad*P4EST_DIM + 1]
             #ifdef P4_TO_P8
                << ", " << xyz_quad_min[i_quad*P4EST_DIM + 2]
             #endif
                << " : "
                << xyz_quad_max[i_quad*P4EST_DIM + 0] << ", " << xyz_quad_max[i_quad*P4EST_DIM + 1]
             #ifdef P4_TO_P8
                << ", " << xyz_quad_max[i_quad*P4EST_DIM + 2]
             #endif
                << std::endl;
    }
    throw std::invalid_argument("[ERROR]: Point does not belong to any neighbouring quadrant.");
  }

  // get pointers to inputs if necessary
  if (is_input_in_vec)
  {
    ierr = VecGetArrayRead(Fi, &Fi_p_); CHKERRXX(ierr);

    if (method == quadratic ||
        method == quadratic_non_oscillatory ||
        method == quadratic_non_oscillatory_continuous_v1 ||
        method == quadratic_non_oscillatory_continuous_v2)
    {
      ierr = VecGetArrayRead(Fxx, &Fxx_p_); CHKERRXX(ierr);
      ierr = VecGetArrayRead(Fyy, &Fyy_p_); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArrayRead(Fzz, &Fzz_p_); CHKERRXX(ierr);
#endif
    }

  } else {

    Fi_p_ = Fi_p;

    if (method == quadratic ||
        method == quadratic_non_oscillatory ||
        method == quadratic_non_oscillatory_continuous_v1 ||
        method == quadratic_non_oscillatory_continuous_v2)
    {
      Fxx_p_ = Fxx_p;
      Fyy_p_ = Fyy_p;
#ifdef P4_TO_P8
      Fzz_p_ = Fzz_p;
#endif
    }
  }

  int node_offset = quad_idx[which_quadrant]*P4EST_CHILDREN;

  for (short i = 0; i<P4EST_CHILDREN; i++)
  {
    p4est_locidx_t node_idx = nodes->local_nodes[node_offset + i];
    f[i] = Fi_p_[node_idx];
  }

  if (method == quadratic ||
      method == quadratic_non_oscillatory ||
      method == quadratic_non_oscillatory_continuous_v1 ||
      method == quadratic_non_oscillatory_continuous_v2)
  {
    for (short j = 0; j<P4EST_CHILDREN; j++)
    {
      p4est_locidx_t node_idx = nodes->local_nodes[node_offset + j];

      fxx[j] = Fxx_p_[node_idx];
      fyy[j] = Fyy_p_[node_idx];
#ifdef P4_TO_P8
      fzz[j] = Fzz_p_[node_idx];
#endif
    }
  }

  // restore arrays
  if (is_input_in_vec)
  {
    ierr = VecRestoreArrayRead(Fi, &Fi_p_); CHKERRXX(ierr);

    if (method == quadratic ||
        method == quadratic_non_oscillatory ||
        method == quadratic_non_oscillatory_continuous_v1 ||
        method == quadratic_non_oscillatory_continuous_v2) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p_); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p_); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p_); CHKERRXX(ierr);
#endif
    }
  }

  double value=0;


//  interp.initialize(xyz_quad_min[which_quadrant*P4EST_DIM + 0], xyz_quad_max[which_quadrant*P4EST_DIM + 0],
//                    xyz_quad_min[which_quadrant*P4EST_DIM + 1], xyz_quad_max[which_quadrant*P4EST_DIM + 1],
//                    #ifdef P4_TO_P8
//                    xyz_quad_min[which_quadrant*P4EST_DIM + 2], xyz_quad_max[which_quadrant*P4EST_DIM + 2],
//                    1,
//                    #endif
//                    1,1);

  if (method == linear) {
    value = this->linear_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, xyz);
  } else if (method == quadratic) {
#ifdef P4_TO_P8
    value = this->quadratic_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, fzz, xyz);
#else
    value = this->quadratic_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, xyz);
#endif
  } else if (method == quadratic_non_oscillatory_continuous_v1) {
#ifdef P4_TO_P8
    value = this->quadratic_non_oscillatory_continuous_v1_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, fzz, xyz);
#else
    value = this->quadratic_non_oscillatory_continuous_v1_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, xyz);
#endif
  } else if (method == quadratic_non_oscillatory_continuous_v2) {
#ifdef P4_TO_P8
    value = this->quadratic_non_oscillatory_continuous_v2_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, fzz, xyz);
#else
    value = this->quadratic_non_oscillatory_continuous_v2_interpolation(&xyz_quad_min[which_quadrant*P4EST_DIM], &xyz_quad_max[which_quadrant*P4EST_DIM], f, fxx, fyy, xyz);
#endif
  }

  return value;
}

#ifdef P4_TO_P8
double my_p4est_interpolation_nodes_local_t::quadratic_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *Fzz, const double *xyz_global) const
#else
double my_p4est_interpolation_nodes_local_t::quadratic_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *xyz_global) const
#endif
{
  double dx = (xyz_quad_max[0] - xyz_quad_min[0]);
  double dy = (xyz_quad_max[1] - xyz_quad_min[1]);
#ifdef P4_TO_P8
  double dz = (xyz_quad_max[2] - xyz_quad_min[2]);
#endif

  double d_m00 = (xyz_global[0] - xyz_quad_min[0])/dx;
  double d_p00 = 1.-d_m00;
  double d_0m0 = (xyz_global[1] - xyz_quad_min[1])/dy;
  double d_0p0 = 1.-d_0m0;
#ifdef P4_TO_P8
  double d_00m = (xyz_global[2] - xyz_quad_min[2])/dz;
  double d_00p = 1.-d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif


#ifdef P4_TO_P8
  double fdd[P4EST_DIM] = { 0, 0, 0 };
#else
  double fdd[P4EST_DIM] = { 0, 0 };
#endif

  for (short j=0; j<P4EST_CHILDREN; j++)
  {
    fdd[0] += Fxx[j] * w_xyz[j];
    fdd[1] += Fyy[j] * w_xyz[j];
#ifdef P4_TO_P8
    fdd[2] += Fzz[j] * w_xyz[j];
#endif
  }

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1] + dz*dz*d_00p*d_00m*fdd[2]);
#else
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1]);
#endif

  if (value != value)
    throw std::domain_error("[CASL_ERROR]: interpolation result is nan");

  return value;
}

double my_p4est_interpolation_nodes_local_t::linear_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *xyz_global) const
{
  double dx = (xyz_quad_max[0] - xyz_quad_min[0]);
  double dy = (xyz_quad_max[1] - xyz_quad_min[1]);
#ifdef P4_TO_P8
  double dz = (xyz_quad_max[2] - xyz_quad_min[2]);
#endif

  double d_m00 = (xyz_global[0] - xyz_quad_min[0])/dx;
  double d_p00 = 1.-d_m00;
  double d_0m0 = (xyz_global[1] - xyz_quad_min[1])/dy;
  double d_0p0 = 1.-d_0m0;
#ifdef P4_TO_P8
  double d_00m = (xyz_global[2] - xyz_quad_min[2])/dz;
  double d_00p = 1.-d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

  if (value != value)
    throw std::domain_error("[CASL_ERROR]: interpolation result is nan");

  return value;
}

#ifdef P4_TO_P8
double my_p4est_interpolation_nodes_local_t::quadratic_non_oscillatory_continuous_v1_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *Fzz, const double *xyz_global) const
#else
double my_p4est_interpolation_nodes_local_t::quadratic_non_oscillatory_continuous_v1_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *xyz_global) const
#endif
{
  double dx = (xyz_quad_max[0] - xyz_quad_min[0]);
  double dy = (xyz_quad_max[1] - xyz_quad_min[1]);
#ifdef P4_TO_P8
  double dz = (xyz_quad_max[2] - xyz_quad_min[2]);
#endif

  double d_m00 = (xyz_global[0] - xyz_quad_min[0])/dx;
  double d_p00 = 1.-d_m00;
  double d_0m0 = (xyz_global[1] - xyz_quad_min[1])/dy;
  double d_0p0 = 1.-d_0m0;
#ifdef P4_TO_P8
  double d_00m = (xyz_global[2] - xyz_quad_min[2])/dz;
  double d_00p = 1.-d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif

// First alternative scheme: first, minmod on every edge, then weight-average
  double fdd[P4EST_DIM];
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = 0;

  int i, jm, jp;

  i = 0;
  jm = 0; jp = 1; fdd[i] += MINMOD(Fxx[jm], Fxx[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 2; jp = 3; fdd[i] += MINMOD(Fxx[jm], Fxx[jp])*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
  jm = 4; jp = 5; fdd[i] += MINMOD(Fxx[jm], Fxx[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 6; jp = 7; fdd[i] += MINMOD(Fxx[jm], Fxx[jp])*(w_xyz[jm]+w_xyz[jp]);
#endif

  i = 1;
  jm = 0; jp = 2; fdd[i] += MINMOD(Fyy[jm], Fyy[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 1; jp = 3; fdd[i] += MINMOD(Fyy[jm], Fyy[jp])*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
  jm = 4; jp = 6; fdd[i] += MINMOD(Fyy[jm], Fyy[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 5; jp = 7; fdd[i] += MINMOD(Fyy[jm], Fyy[jp])*(w_xyz[jm]+w_xyz[jp]);
#endif

#ifdef P4_TO_P8
  i = 2;
  jm = 0; jp = 4; fdd[i] += MINMOD(Fzz[jm], Fzz[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 1; jp = 5; fdd[i] += MINMOD(Fzz[jm], Fzz[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 2; jp = 6; fdd[i] += MINMOD(Fzz[jm], Fzz[jp])*(w_xyz[jm]+w_xyz[jp]);
  jm = 3; jp = 7; fdd[i] += MINMOD(Fzz[jm], Fzz[jp])*(w_xyz[jm]+w_xyz[jp]);
#endif

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1] + dz*dz*d_00p*d_00m*fdd[2]);
#else
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1]);
#endif

  if (value != value)
    throw std::domain_error("[CASL_ERROR]: interpolation result is nan");

  return value;
}

#ifdef P4_TO_P8
double my_p4est_interpolation_nodes_local_t::quadratic_non_oscillatory_continuous_v2_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *Fzz, const double *xyz_global) const
#else
double my_p4est_interpolation_nodes_local_t::quadratic_non_oscillatory_continuous_v2_interpolation(const double *xyz_quad_min, const double *xyz_quad_max, const double *F, const double *Fxx, const double *Fyy, const double *xyz_global) const
#endif
{
  double dx = (xyz_quad_max[0] - xyz_quad_min[0]);
  double dy = (xyz_quad_max[1] - xyz_quad_min[1]);
#ifdef P4_TO_P8
  double dz = (xyz_quad_max[2] - xyz_quad_min[2]);
#endif

  double d_m00 = (xyz_global[0] - xyz_quad_min[0])/dx;
  double d_p00 = 1.-d_m00;
  double d_0m0 = (xyz_global[1] - xyz_quad_min[1])/dy;
  double d_0p0 = 1.-d_0m0;
#ifdef P4_TO_P8
  double d_00m = (xyz_global[2] - xyz_quad_min[2])/dz;
  double d_00p = 1.-d_00m;
#endif

#ifdef P4_TO_P8
  double w_xyz[] =
  {
    d_p00*d_0p0*d_00p,
    d_m00*d_0p0*d_00p,
    d_p00*d_0m0*d_00p,
    d_m00*d_0m0*d_00p,
    d_p00*d_0p0*d_00m,
    d_m00*d_0p0*d_00m,
    d_p00*d_0m0*d_00m,
    d_m00*d_0m0*d_00m
  };
#else
  double w_xyz[] =
  {
    d_p00*d_0p0,
    d_m00*d_0p0,
    d_p00*d_0m0,
    d_m00*d_0m0
  };
#endif


// Second alternative scheme: first, weight-average in perpendicular plane, then minmod
  double fdd[P4EST_DIM];
  for (short i = 0; i<P4EST_DIM; i++)
    fdd[i] = 0;

  int i, jm, jp;
  double fdd_m, fdd_p;

  i = 0;
  fdd_m = 0;
  fdd_p = 0;
  jm = 0; jp = 1; fdd_m += Fxx[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fxx[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 2; jp = 3; fdd_m += Fxx[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fxx[jp]*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
  jm = 4; jp = 5; fdd_m += Fxx[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fxx[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 6; jp = 7; fdd_m += Fxx[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fxx[jp]*(w_xyz[jm]+w_xyz[jp]);
#endif
  fdd[i] = MINMOD(fdd_m, fdd_p);

  i = 1;
  fdd_m = 0;
  fdd_p = 0;
  jm = 0; jp = 2; fdd_m += Fyy[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fyy[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 1; jp = 3; fdd_m += Fyy[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fyy[jp]*(w_xyz[jm]+w_xyz[jp]);
#ifdef P4_TO_P8
  jm = 4; jp = 6; fdd_m += Fyy[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fyy[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 5; jp = 7; fdd_m += Fyy[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fyy[jp]*(w_xyz[jm]+w_xyz[jp]);
#endif
  fdd[i] = MINMOD(fdd_m, fdd_p);

#ifdef P4_TO_P8
  i = 2;
  fdd_m = 0;
  fdd_p = 0;
  jm = 0; jp = 4; fdd_m += Fzz[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fzz[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 1; jp = 5; fdd_m += Fzz[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fzz[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 2; jp = 6; fdd_m += Fzz[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fzz[jp]*(w_xyz[jm]+w_xyz[jp]);
  jm = 3; jp = 7; fdd_m += Fzz[jm]*(w_xyz[jm]+w_xyz[jp]); fdd_p += Fzz[jp]*(w_xyz[jm]+w_xyz[jp]);
  fdd[i] = MINMOD(fdd_m, fdd_p);
#endif

  double value = 0;
  for (short j = 0; j<P4EST_CHILDREN; j++)
    value += F[j]*w_xyz[j];

#ifdef P4_TO_P8
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1] + dz*dz*d_00p*d_00m*fdd[2]);
#else
  value -= 0.5*(dx*dx*d_p00*d_m00*fdd[0] + dy*dy*d_0p0*d_0m0*fdd[1]);
#endif

  if (value != value)
    throw std::domain_error("[CASL_ERROR]: interpolation result is nan");

  return value;
}


//#ifdef P4_TO_P8
//double my_p4est_interpolation_nodes_local_t::operator() (double x, double y, double z)
//{
//  return interpolate(x,y,z);
//}
//#else
//double my_p4est_interpolation_nodes_local_t::operator() (double x, double y)
//{
//  return interpolate(x,y);
//}
//#endif


//#ifdef P4_TO_P8
//  double my_p4est_interpolation_nodes_local_t::operator () (double x, double y, double z) const
//  {
//    return interpolate(x,y,z);
//  }
//#else
//  double my_p4est_interpolation_nodes_local_t::operator() (double x, double y) const
//  {
//    double r = interpolate(x,y);
////    return interpolate(x,y);
//    return 0;
//  }
//#endif
