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
  p4est_topidx_t vtx_0_max    = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  p4est_topidx_t vtx_0_min    = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + 0];
  p4est_topidx_t vtx_last_max = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    tree_dimension[dim]     = p4est->connectivity->vertices[3*vtx_0_max     + dim] - p4est->connectivity->vertices[3*vtx_0_min + dim];
    domain_dimension[dim]   = p4est->connectivity->vertices[3*vtx_last_max  + dim] - p4est->connectivity->vertices[3*vtx_0_min + dim];
  }
}


#ifdef P4_TO_P8
void my_p4est_interpolation_cells_t::set_input(Vec* F, const Vec phi, const BoundaryConditions3D *bc, unsigned int n_vecs_)
#else
void my_p4est_interpolation_cells_t::set_input(Vec* F, const Vec phi, const BoundaryConditions2D *bc, unsigned int n_vecs_)
#endif
{
  set_input(F, n_vecs_, 1);
  this->phi = phi;
  this->bc = bc;
}


#ifdef P4_TO_P8
void  my_p4est_interpolation_cells_t::operator ()(double x, double y, double z, double* results) const
#else
void my_p4est_interpolation_cells_t::operator ()(double x, double y, double* results) const
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
  {
    interpolate(best_match, xyz, results);
    return;
  }

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_cells_t->interpolate(): the point does not belong to the local forest.");
}


void my_p4est_interpolation_cells_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results) const
{
  PetscErrorCode ierr;

  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f == 1); // not implemented for bs_f > 1 yet
  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  /* check if exactly on a point */
  if(fabs(xyz[0]-quad_x_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS*tree_dimension[0] &&
     fabs(xyz[1]-quad_y_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS*tree_dimension[1]
   #ifdef P4_TO_P8
     && fabs(xyz[2]-quad_z_fr_q(quad_idx, tree_idx, p4est, ghost))<EPS*tree_dimension[2]
   #endif
     )
  {
    for (unsigned int k = 0; k < n_functions; ++k)
    {
      results[k] = Fi_p[k][quad_idx];
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }
    return;
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

#ifdef P4_TO_P8
  scaling *= MIN(tree_dimension[0], tree_dimension[1], tree_dimension[2]);
#else
  scaling *= MIN(tree_dimension[0], tree_dimension[1]);
#endif

  std::vector<p4est_locidx_t> interp_points;
  matrix_t A;
#ifdef P4_TO_P8
  A.resize(1,10);
#else
  A.resize(1,6);
#endif
  std::vector<double> p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k)
    p[k].resize(0);
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
              if(fabs((xyz[i] - xyz_t[i] + ((double) cc)*domain_dimension[i])) < fabs(rel_dist))
                rel_dist = (xyz[i] - xyz_t[i] + ((double) cc)*domain_dimension[i]);
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

        for (unsigned int k = 0; k < n_functions; ++k)
          p[k].push_back(Fi_p[k][qm_idx] * w);

        for(int d=0; d<P4EST_DIM; ++d)
          if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end()) // comparison of doubles... VERY bad :-/
            nb[d].push_back(xyz_t[d]);

        interp_points.push_back(qm_idx);
      }
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    results[k] = 0.0;
  }
  if(interp_points.size()==0)
    return;

  A.scale_by_maxabs(p, n_functions);


#ifdef P4_TO_P8
  solve_lsqr_system(A, p, n_functions, results, nb[0].size(), nb[1].size(), nb[2].size());
#else
  solve_lsqr_system(A, p, n_functions, results, nb[0].size(), nb[1].size());
#endif
  return;
}
