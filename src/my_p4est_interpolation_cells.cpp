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
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    tree_dimension[dim]     = p4est->connectivity->vertices[3*vtx_0_max     + dim] - p4est->connectivity->vertices[3*vtx_0_min + dim];
    domain_dimension[dim]   = xyz_max[dim] - xyz_min[dim];
  }
}


void my_p4est_interpolation_cells_t::set_input(Vec* F, const Vec phi, const BoundaryConditionsDIM *bc, unsigned int n_vecs_)
{
  set_input(F, n_vecs_, 1);
  this->phi = phi;
  this->bc = bc;
}


void  my_p4est_interpolation_cells_t::operator ()(DIM(double x, double y, double z), double* results) const
{
  double xyz [P4EST_DIM] = {DIM(x, y, z)};

  /* first clip the coordinates */
  double xyz_clip[P4EST_DIM] = {DIM(x, y, z)};
  clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);

  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

//  if (rank_found == p4est->mpirank)
  if (rank_found != -1)
  {
    interpolate(best_match, xyz, results, 1); // last argument is dummy
    return;
  }

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_cells_t->interpolate(): the point does not belong to the local forest.");
}


void my_p4est_interpolation_cells_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &) const
{
  PetscErrorCode ierr;

  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f == 1); // not implemented for bs_f > 1 yet
  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  /* check if exactly on a point */
  double quad_xyz[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, quad_xyz);
  bool is_quad_center = true;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    is_quad_center = is_quad_center && fabs(xyz[dim] - quad_xyz[dim]) < EPS*tree_dimension[dim];
  if(is_quad_center)
  {
    for (unsigned int k = 0; k < n_functions; ++k)
    {
      results[k] = Fi_p[k][quad_idx];
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }
    return;
  }

  /* gather the neighborhood */
  set_of_neighboring_quadrants close_ngbd;
  double scaling = DBL_MAX;

  for(char i = -1; i < 2; ++i)
    for(char j = -1; j < 2; ++j)
#ifdef P4_TO_P8
      for(char k = -1; k < 2; ++k)
#endif
        ngbd_c->find_neighbor_cells_of_cell(close_ngbd, quad_idx, tree_idx, DIM(i, j, k));

  set_of_neighboring_quadrants ngbd(close_ngbd);

  for (set_of_neighboring_quadrants::const_iterator it = close_ngbd.begin(); it != close_ngbd.end(); ++it)
    for(char i = -1; i < 2; ++i)
      for(char j = -1; j < 2; ++j)
#ifdef P4_TO_P8
        for(char k = -1; k < 2; ++k)
#endif
          ngbd_c->find_neighbor_cells_of_cell(ngbd, it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(i, j, k));

  for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
    scaling = MIN(scaling, (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN);

  scaling *= MIN(DIM(tree_dimension[0], tree_dimension[1], tree_dimension[2]));

  std::vector<p4est_locidx_t> interp_points;
  matrix_t A;
  A.resize(1, 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM+1)/2);
  std::vector<double> p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k)
    p[k].resize(0);
  std::vector<double> nb[P4EST_DIM];

  scaling *= .5;
  double min_w = 1e-6;
  double inv_max_w = 1e-6;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
  {
    p4est_locidx_t qm_idx = it->p.piggy3.local_num;
    if(std::find(interp_points.begin(), interp_points.end(), qm_idx) == interp_points.end())
    {
      /* check if quadrant is well defined */
      double phi_q = 0.0;
      bool neumann_is_neg = false;
      for(unsigned char i = 0; i < P4EST_CHILDREN; ++i)
      {
        double tmp = phi_p[nodes->local_nodes[P4EST_CHILDREN*qm_idx + i]];
        neumann_is_neg = neumann_is_neg || tmp < 0.0;
        phi_q += tmp;
      }
      phi_q /= (double) P4EST_CHILDREN;

      if(bc->interfaceType() == NOINTERFACE || (bc->interfaceType(xyz) == DIRICHLET && phi_q < 0.0) || (bc->interfaceType(xyz) == NEUMANN && neumann_is_neg))
      {
        double xyz_t[P4EST_DIM]; quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_t);

        for(unsigned char i = 0; i < P4EST_DIM; ++i)
        {
          double rel_dist = (xyz[i] - xyz_t[i]);
          if(periodic[i])
            for (short cc = -1; cc < 2; cc+=2)
              if(fabs((xyz[i] - xyz_t[i] + cc*domain_dimension[i])) < fabs(rel_dist))
                rel_dist = (xyz[i] - xyz_t[i] + cc*domain_dimension[i]);
          xyz_t[i] = rel_dist / scaling;
        }

        double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

        A.set_value(interp_points.size(), 0,                1                 * w);
        A.set_value(interp_points.size(), 1,                xyz_t[0]          * w);
        A.set_value(interp_points.size(), 2,                xyz_t[1]          * w);
#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 3,                xyz_t[2]          * w);
#endif
        A.set_value(interp_points.size(), 1 +   P4EST_DIM,  xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 2 +   P4EST_DIM,  xyz_t[0]*xyz_t[1] * w);
#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 3 +   P4EST_DIM,  xyz_t[0]*xyz_t[2] * w);
#endif
        A.set_value(interp_points.size(), 1 + 2*P4EST_DIM,  xyz_t[1]*xyz_t[1] * w);
#ifdef P4_TO_P8
        A.set_value(interp_points.size(), 2 + 2*P4EST_DIM,  xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(),     3*P4EST_DIM,  xyz_t[2]*xyz_t[2] * w);
#endif

        for (unsigned int k = 0; k < n_functions; ++k)
          p[k].push_back(Fi_p[k][qm_idx] * w);

        for(unsigned char d = 0; d < P4EST_DIM; ++d)
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
  if(interp_points.size() == 0)
    return;

  A.scale_by_maxabs(p, n_functions);

  solve_lsqr_system(A, p, n_functions, results, DIM(nb[0].size(), nb[1].size(), nb[2].size()));

  return;
}
