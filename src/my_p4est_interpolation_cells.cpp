#ifdef P4_TO_P8
#include "my_p4est_interpolation_cells.h"
#else
#include "my_p4est_interpolation_cells.h"
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>


my_p4est_interpolation_cells_t::my_p4est_interpolation_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n)
  : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes), ngbd_c(ngbd_c), phi(NULL), bc(NULL)

{
}


#ifdef P4_TO_P8
void my_p4est_interpolation_cells_t::set_input(Vec F, const Vec phi, const BoundaryConditions3D *bc)
#else
void my_p4est_interpolation_cells_t::set_input(Vec F, const Vec phi, const BoundaryConditions2D *bc)
#endif
{
  this->Fi = F;
  this->phi = phi;
  this->bc = bc;
}


#ifdef P4_TO_P8
double my_p4est_interpolation_cells_t::operator ()(double x, double y, double z) const
#else
double my_p4est_interpolation_cells_t::operator ()(double x, double y) const
#endif
{
#ifdef P4_TO_P8
  double xyz [] = { x, y, z };
#else
  double xyz [] = { x, y };
#endif

  /* first clip the coordinates */
#ifdef P4_TO_P8
  double xyz_clip [] = { x, y, z };
#else
  double xyz_clip [] = { x, y };
#endif

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = xyz_min[i];
  }

  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

//  if (rank_found == p4est->mpirank)
  if (rank_found != -1)
    return interpolate(best_match, xyz);

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_cells_t->interpolate(): the point does not belong to the local forest.");
}


double my_p4est_interpolation_cells_t::interpolate(const p4est_quadrant_t &quad, const double *xyz) const
{
  PetscErrorCode ierr;

  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  /* check if exactly on a point */
  if(fabs(xyz[0]-quad_x_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS &&
     fabs(xyz[1]-quad_y_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS
   #ifdef P4_TO_P8
     && fabs(xyz[2]-quad_z_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS
   #endif
     )
  {
    double Fi_tmp = Fi_p[quad_idx];
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
    return Fi_tmp;
  }

  /* gather the neighborhood */
  std::vector<p4est_quadrant_t> ngbd;
  double scaling = DBL_MAX;

  for(int i=-1; i<2; ++i)
    for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
      for(int k=-1; k<2; ++k)
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, j, k);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, i, j);
#endif

  int nb_ngbd = ngbd.size();
  for(int m=0; m<nb_ngbd; ++m)
    for(int i=-1; i<2; ++i)
      for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
        for(int k=-1; k<2; ++k)
          ngbd_c->find_neighbor_cells_of_cell(ngbd, ngbd[m].p.piggy3.local_num, ngbd[m].p.piggy3.which_tree, i, j, k);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd, ngbd[m].p.piggy3.local_num, ngbd[m].p.piggy3.which_tree, i, j);
#endif

  for(unsigned int m=0; m<ngbd.size(); ++m)
    scaling = MIN(scaling, (double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];

  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
#ifdef P4_TO_P8
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
#endif

#ifdef P4_TO_P8
  scaling *= MIN(tree_xmax-tree_xmin, tree_ymax-tree_ymin, tree_zmax-tree_zmin);
#else
  scaling *= MIN(tree_xmax-tree_xmin, tree_ymax-tree_ymin);
#endif
  double domain_size[P4EST_DIM];
  if(is_periodic(p4est, dir::x) || is_periodic(p4est, dir::y)
   #ifdef P4_TO_P8
     || is_periodic(p4est, dir::z)
   #endif
     )
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      domain_size[dim] =
          p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1] + dim] -
          p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[0 + 0] + dim];

  std::vector<p4est_locidx_t> interp_points;
  matrix_t A;
#ifdef P4_TO_P8
  A.resize(1,10);
#else
  A.resize(1,6);
#endif
  std::vector<double> p;
  std::vector<double> nb[P4EST_DIM];

  scaling *= .5;
  double min_w = 1e-6;
  double inv_max_w = 1e-6;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for(unsigned int m=0; m<ngbd.size(); m++)
  {
    p4est_locidx_t qm_idx = ngbd[m].p.piggy3.local_num;
    if(std::find(interp_points.begin(), interp_points.end(),qm_idx)==interp_points.end())
    {
      /* check if quadrant is well defined */
      double phi_q = 0;
      bool neumann_is_neg = false;
      for(int i=0; i<P4EST_CHILDREN; ++i)
      {
        double tmp = phi_p[nodes->local_nodes[P4EST_CHILDREN*qm_idx + i]];
        neumann_is_neg = neumann_is_neg || (tmp<0);
        phi_q += tmp;
      }
      phi_q /= (double) P4EST_CHILDREN;

      if( (bc->interfaceType()==NOINTERFACE || (bc->interfaceType(xyz)==DIRICHLET && phi_q<0) || (bc->interfaceType(xyz)==NEUMANN && neumann_is_neg) ) )
      {
        double xyz_t[P4EST_DIM];

        xyz_t[0] = quad_x_fr_q(ngbd[m].p.piggy3.local_num, ngbd[m].p.piggy3.which_tree, p4est, ghost);
        xyz_t[1] = quad_y_fr_q(ngbd[m].p.piggy3.local_num, ngbd[m].p.piggy3.which_tree, p4est, ghost);
#ifdef P4_TO_P8
        xyz_t[2] = quad_z_fr_q(ngbd[m].p.piggy3.local_num, ngbd[m].p.piggy3.which_tree, p4est, ghost);
#endif

        for(int i=0; i<P4EST_DIM; ++i)
        {
          double rel_dist = (xyz[i] - xyz_t[i]);
          if(is_periodic(p4est, i))
            for (short cc = -1; cc < 2; cc+=2)
              if(fabs((xyz[i] - xyz_t[i] + ((double) cc)*domain_size[i])) < fabs(rel_dist))
                rel_dist = (xyz[i] - xyz_t[i] + ((double) cc)*domain_size[i]);
          xyz_t[i] = rel_dist / scaling;
        }
  //        xyz_t[i] = (xyz[i] - xyz_t[i]) / scaling;

#ifdef P4_TO_P8
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
        double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif

#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[2]          * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 5, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 6, xyz_t[0]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 7, xyz_t[1]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 8, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 9, xyz_t[2]*xyz_t[2] * w);
#else
        A.set_value(interp_points.size(), 0, 1                 * w);
        A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
        A.set_value(interp_points.size(), 3, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 5, xyz_t[1]*xyz_t[1] * w);
#endif

        p.push_back(Fi_p[qm_idx] * w);

        for(int d=0; d<P4EST_DIM; ++d)
          if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
            nb[d].push_back(xyz_t[d]);

        interp_points.push_back(qm_idx);
      }
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  if(interp_points.size()==0)
    return 0;

  A.scale_by_maxabs(p);

#ifdef P4_TO_P8
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size(), nb[2].size());
#else
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size());
#endif
}
