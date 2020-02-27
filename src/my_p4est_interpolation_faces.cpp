#ifdef P4_TO_P8
#include "my_p4est_interpolation_faces.h"
#else
#include "my_p4est_interpolation_faces.h"
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>


my_p4est_interpolation_faces_t::my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces)
  : my_p4est_interpolation_t(ngbd_n), faces(faces), ngbd_c(faces->ngbd_c), face_is_well_defined(NULL), bc(NULL) { }


void my_p4est_interpolation_faces_t::set_input(Vec *F, unsigned char dir, unsigned int n_vecs_, int order, Vec face_is_well_defined, BoundaryConditionsDIM *bc)
{
  set_input(F, n_vecs_, 1);
  this->face_is_well_defined = face_is_well_defined;
  this->dir = dir;
  this->bc = bc;
  this->order = order;
}


void my_p4est_interpolation_faces_t::operator ()(DIM(double x, double y, double z), double *results) const
{
  double xyz[P4EST_DIM] = {DIM(x, y, z)};

  /* first clip the coordinates */
  double xyz_clip[P4EST_DIM] = {DIM(x, y, z)};
  // clip to bounding box
  clip_in_domain(xyz_clip, xyz_min, xyz_max, periodic);

  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

  if(rank_found == p4est->mpirank)
  {
    interpolate(best_match, xyz, results, 1); // last argument is dummy
    return;
  }

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_faces_t->interpolate(): the point does not belong to the local forest.");
}


void my_p4est_interpolation_faces_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const unsigned int &) const
{
  PetscErrorCode ierr;

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f == 1); // not implemented for bs_f > 1 yet

  /* [Raphael Egan (01/22/2020): results seem more accurate (2nd order vs 1st order) when doing nothing special for Neumann wall boundary condition (from tests in main file for testing poisson_faces)...] */
  bool is_on_wall = false;
  for (unsigned char dim = 0; dim < P4EST_DIM && !is_on_wall; ++dim)
    is_on_wall = is_on_wall || ((fabs(xyz[dim] - xyz_min[dim]) < EPS*faces->tree_dimensions[dim] || fabs(xyz[dim] - xyz_max[dim]) < EPS*faces->tree_dimensions[dim]) && !periodic[dim]);
  if(is_on_wall && bc != NULL && bc->wallType(xyz) == DIRICHLET)
  {
    for (unsigned int k = 0; k < n_functions; ++k)
      results[k] = bc->wallValue(xyz);
    return;
  }

  double qh = MIN(DIM(faces->tree_dimensions[0], faces->tree_dimensions[1], faces->tree_dimensions[2]));
  double scaling = .5 * qh*(double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN;

  p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, quad.p.piggy3.which_tree);
  p4est_quadrant_t quad_with_correct_local_num = quad;
  quad_with_correct_local_num.p.piggy3.local_num = quad.p.piggy3.local_num + tree->quadrants_offset;

  /* gather the neighborhood */
  set_of_neighboring_quadrants close_ngbd_tmp; close_ngbd_tmp.insert(quad_with_correct_local_num);
  std::set<indexed_and_located_face> face_ngbd;
  indexed_and_located_face f_tmp;

  for(char i = -1; i < 2; ++i)
    for(char j = -1; j < 2; ++j)
#ifdef P4_TO_P8
      for(char k = -1; k < 2; ++k)
#endif
      {
        if(ANDD(i == 0, j == 0, k == 0))
          continue;
        ngbd_c->find_neighbor_cells_of_cell(close_ngbd_tmp, quad_with_correct_local_num.p.piggy3.local_num, quad.p.piggy3.which_tree, DIM(i, j, k));
      }

  set_of_neighboring_quadrants ngbd_tmp(close_ngbd_tmp);
  for (set_of_neighboring_quadrants::const_iterator it = close_ngbd_tmp.begin(); it != close_ngbd_tmp.end(); ++it)
    for(char i = -1; i < 2; ++i)
      for(char j = -1; j < 2; ++j)
#ifdef P4_TO_P8
        for(char k = -1; k < 2; ++k)
#endif
        {
          if(ANDD(i == 0, j == 0, k == 0))
            continue;
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp,  it->p.piggy3.local_num, it->p.piggy3.which_tree, DIM(i, j, k));
        }

  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  const PetscScalar *face_is_well_defined_p;
  if(face_is_well_defined!=NULL)
    ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  matrix_t A;
  A.resize(1, 1 + P4EST_DIM + (order >= 2 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0)); // constant term + P4EST_DIM linear terms + (if second order) P4EST_DIM squared terms + 0.5*P4EST_DIM*(P4EST_DIM + 1) squared and crossed terms

  std::vector<double> p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k)
    p[k].resize(0);
  std::vector<double> nb[P4EST_DIM];
  double min_w = 1e-6;
  double inv_max_w = 1e-6;
  int row_idx = 0;

  for (set_of_neighboring_quadrants::const_iterator it = ngbd_tmp.begin(); it != ngbd_tmp.end(); ++it)
  {
    for(char touch = 0; touch < 2; ++touch)
    {
      f_tmp.face_idx = faces->q2f(it->p.piggy3.local_num, 2*dir + touch);

      if(f_tmp.face_idx != NO_VELOCITY && face_ngbd.find(f_tmp) == face_ngbd.end())
      {
        faces->xyz_fr_f(f_tmp.face_idx, dir, f_tmp.xyz_face);
        bool point_is_on_face = true;
        for (int dim = 0; dim < P4EST_DIM && point_is_on_face; ++dim)
          point_is_on_face = point_is_on_face && fabs(xyz[dim] - f_tmp.xyz_face[dim]) < EPS*faces->tree_dimensions[dim];
        if((face_is_well_defined == NULL || face_is_well_defined_p[f_tmp.face_idx]) && point_is_on_face)
        {
          for (unsigned int k = 0; k < n_functions; ++k) {
            results[k] = Fi_p[k][f_tmp.face_idx];
            ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
          }
          if(face_is_well_defined!=NULL)
            ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);
          return;
        }

        face_ngbd.insert(f_tmp);

        if(face_is_well_defined == NULL || face_is_well_defined_p[f_tmp.face_idx])
        {
          double xyz_t[P4EST_DIM] = {DIM(f_tmp.xyz_face[0], f_tmp.xyz_face[1], f_tmp.xyz_face[2])};

          for(unsigned char i = 0; i < P4EST_DIM; ++i)
          {
            double rel_dist = (xyz[i] - xyz_t[i]);
            if(periodic[i])
              for (char cc = -1; cc < 2; cc+=2)
                if(fabs((xyz[i] - xyz_t[i] + cc*(xyz_max[i] - xyz_min[i]))) < fabs(rel_dist))
                  rel_dist = (xyz[i] - xyz_t[i] + cc*(xyz_max[i] - xyz_min[i]));
            xyz_t[i] = rel_dist / scaling;
          }

          double w = MAX(min_w,1./MAX(inv_max_w,sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

          A.set_value(row_idx, 0,                 1.0               * w);
          A.set_value(row_idx, 1,                 xyz_t[0]          * w);
          A.set_value(row_idx, 2,                 xyz_t[1]          * w);
#ifdef P4_TO_P8
          A.set_value(row_idx, 3,                 xyz_t[2]          * w);
#endif
          if(order >= 2)
          {
            A.set_value(row_idx,   1 + P4EST_DIM, xyz_t[0]*xyz_t[0] * w);
            A.set_value(row_idx,   2 + P4EST_DIM, xyz_t[0]*xyz_t[1] * w);
#ifdef P4_TO_P8
            A.set_value(row_idx,   3 + P4EST_DIM, xyz_t[0]*xyz_t[2] * w);
#endif
            A.set_value(row_idx, 1 + 2*P4EST_DIM, xyz_t[1]*xyz_t[1] * w);
#ifdef P4_TO_P8
            A.set_value(row_idx, 2 + 2*P4EST_DIM, xyz_t[1]*xyz_t[2] * w);
            A.set_value(row_idx,     3*P4EST_DIM, xyz_t[2]*xyz_t[2] * w);
#endif
          }

          for (unsigned int k = 0; k < n_functions; ++k)
            p[k].push_back((Fi_p[k][f_tmp.face_idx] /* + neumann_term*/)* w);

          for(unsigned char d = 0; d < P4EST_DIM; ++d)
            if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
              nb[d].push_back(xyz_t[d]);

          row_idx++;
        }
      }
    }
  }

  for (unsigned int k = 0; k < n_functions; ++k)
  {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    results[k] = 0.0;
  }
  if(face_is_well_defined!=NULL)
    ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  if(row_idx == 0)
    return;

  A.scale_by_maxabs(p, n_functions);

  solve_lsqr_system(A, p, n_functions, results, DIM(nb[0].size(), nb[1].size(), nb[2].size()));
  return;
}
