#ifdef P4_TO_P8
#include "my_p8est_interpolation_cells.h"
#else
#include "my_p4est_interpolation_cells.h"
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>

void my_p4est_interpolation_cells_t::set_input(Vec* F, const Vec phi_, const BoundaryConditionsDIM *bc_, const size_t &n_vecs_)
{
  set_input_fields(F, n_vecs_, 1); // only for block_size of 1 with cell-sampled fields
  this->phi = phi_;
  this->bc = bc_;
  P4EST_ASSERT(bc != NULL);
}

void my_p4est_interpolation_cells_t::operator ()(const double *xyz, double* results) const
{
  int rank_found = -1;
  std::vector<p4est_quadrant_t> remote_matches;
  try
  {
    clip_point_and_interpolate_all_on_the_fly(xyz, results, rank_found, remote_matches, true);
    // we allow the procedure to proceed even in ghost layer because it is conceptually possible
    // however with a possibly restricted cell neighborhood
    // --> this may introduce and trigger process-dependent outcomes which is not ideal
    // Therefore, the interpolation procedure prints a warning if called on a nonlocal quadrant
  }
  catch (std::invalid_argument& e)
  {
    // The point could not be handled locally --> let the user know but let's be more specific about the origin of the issue
    P4EST_ASSERT(rank_found == -1);
    std::ostringstream oss;
    oss << "my_p4est_interpolation_cells_t::operator (): Point (" << xyz[0] << "," << xyz[1] << ONLY3D("," << xyz[2] <<) ") cannot be processed on-the-fly by process " << p4est->mpirank << ".";
    oss << "Process(es) likely to own a quadrant containing the point is (are) = ";
    for (size_t i = 0; i < remote_matches.size() - 1; i++)
      oss << remote_matches[i].p.piggy1.owner_rank << ", ";
    oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    throw std::invalid_argument(oss.str());
  }
  return;
}

void my_p4est_interpolation_cells_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &) const
{
  PetscErrorCode ierr;

  const p4est_topidx_t &tree_idx = quad.p.piggy3.which_tree;
  const p4est_locidx_t &quad_idx = quad.p.piggy3.local_num;
  if(quad_idx > p4est->local_num_quadrants)
    std::cerr << "WARNING : interpolating cell-sampled values at point " << xyz[0] << ", " << xyz[1] ONLY3D(<< ", " << xyz[2]) << " which belongs to the ghost layer of process " << p4est->mpirank << std::endl;

  const size_t n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  /* check if exactly on a point */
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();
  double quad_xyz[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, quad_xyz);
  bool is_quad_center = true;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    is_quad_center = is_quad_center && fabs(xyz[dim] - quad_xyz[dim]) < EPS*tree_dimensions[dim];
  if(is_quad_center)
  {
    for (size_t k = 0; k < n_functions; ++k)
    {
      results[k] = Fi_p[k][quad_idx];
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }
    return;
  }

  /* gather the extended neighborhood and the scaling length based on the nearby quadrants only */
  set_of_neighboring_quadrants ngbd; ngbd.clear();
  const p4est_qcoord_t smallest_logical_size_of_nearby_quad = ngbd_c->gather_neighbor_cells_of_cell(quad, ngbd, true);
  // set the scaling length
  const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double)smallest_logical_size_of_nearby_quad/(double) P4EST_ROOT_LEN;

  matrix_t A;
  A.resize(1, 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM+1)/2);
  std::vector<double> p[n_functions];
  for (size_t k = 0; k < n_functions; ++k)
    p[k].resize(0);
  std::vector<double> nb[P4EST_DIM];

  const double *node_sampled_phi_p;
  ierr = VecGetArrayRead(phi, &node_sampled_phi_p); CHKERRXX(ierr);

  const double *xyz_min   = get_xyz_min();
  const double *xyz_max   = get_xyz_max();
  const bool *periodicity = get_periodicity();

  const double min_w = 1e-6;
  const double inv_max_w = 1e-6;
  unsigned int row_idx = 0;

  for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
    if(quadrant_value_is_well_defined(*bc, p4est, ghost, nodes, it->p.piggy3.local_num, it->p.piggy3.which_tree, node_sampled_phi_p))
    {
      double xyz_t[P4EST_DIM]; quad_xyz_fr_q(it->p.piggy3.local_num, it->p.piggy3.which_tree, p4est, ghost, xyz_t);

      for(unsigned char i = 0; i < P4EST_DIM; ++i)
      {
        xyz_t[i] = (xyz_t[i] - xyz[i]);
        if(periodicity[i])
        {
          const double pp = xyz_t[i]/(xyz_max[i] - xyz_min[i]);
          xyz_t[i] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[i] - xyz_min[i]);
        }
        xyz_t[i] /= scaling;
      }

      const double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

      unsigned char col_idx = 0;
      A.set_value(row_idx, col_idx++, w); // constant term
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        A.set_value(row_idx, col_idx++, xyz_t[dir]*w); // linear terms
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        for (unsigned char dd = dir; dd < P4EST_DIM; ++dd)
          A.set_value(row_idx, col_idx++, xyz_t[dir]*xyz_t[dd]*w); // quadratic terms
      P4EST_ASSERT(col_idx == 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2);

      for (size_t k = 0; k < n_functions; ++k)
        p[k].push_back(Fi_p[k][it->p.piggy3.local_num] * w);

      for(unsigned char d = 0; d < P4EST_DIM; ++d)
        if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end()) // comparison of doubles... --> bad :-/
          nb[d].push_back(xyz_t[d]);

      row_idx++;
    }

  ierr = VecRestoreArrayRead(phi, &node_sampled_phi_p); CHKERRXX(ierr);
  for (size_t k = 0; k < n_functions; ++k) {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    results[k] = 0.0;
  }
  if(row_idx == 0) // that means no valid neighbor was found, we can't compute anything...
    return;

  A.scale_by_maxabs(p, n_functions);

  solve_lsqr_system(A, p, n_functions, results, DIM(nb[0].size(), nb[1].size(), nb[2].size()));

  return;
}
