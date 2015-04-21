#ifdef P4_TO_P8
#include "my_p4est_interpolation_faces.h"
#else
#include "my_p4est_interpolation_faces.h"
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>


my_p4est_interpolation_faces_t::my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces)
  : my_p4est_interpolation_t(ngbd_n), faces(faces), ngbd_c(faces->ngbd_c), face_is_well_defined(NULL)

{
}


void my_p4est_interpolation_faces_t::set_input(Vec F, Vec face_is_well_defined, int dir)
{
  this->Fi = F;
  this->face_is_well_defined = face_is_well_defined;
  this->dir = dir;
}


#ifdef P4_TO_P8
double my_p4est_interpolation_faces_t::operator ()(double x, double y, double z) const
#else
double my_p4est_interpolation_faces_t::operator ()(double x, double y) const
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

  if (rank_found == p4est->mpirank)
    return interpolate(best_match, xyz);

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_faces_t->interpolate(): the point does not belong to the local forest.");
}


double my_p4est_interpolation_faces_t::interpolate(const p4est_quadrant_t &quad, const double *xyz) const
{
  PetscErrorCode ierr;

  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  /* gather the neighborhood */
  std::vector<p4est_quadrant_t> ngbd_tmp;
  std::vector<p4est_locidx_t> ngbd;
  p4est_locidx_t f_tmp;
  double scaling = DBL_MAX;

  for(int i=-1; i<2; ++i)
    for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
      for(int k=-1; k<2; ++k)
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j, k);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j);
#endif

  int nb_ngbd = ngbd_tmp.size();
  for(int m=0; m<nb_ngbd; ++m)
  {
    for(int i=-1; i<2; ++i)
      for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
        for(int k=-1; k<2; ++k)
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, ngbd_tmp[m].p.piggy3.local_num, ngbd_tmp[m].p.piggy3.which_tree, i, j, k);
#else
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, ngbd_tmp[m].p.piggy3.local_num, ngbd_tmp[m].p.piggy3.which_tree, i, j);
#endif
  }

  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  for(unsigned int m=0; m<ngbd_tmp.size(); ++m)
  {
    scaling = MIN(scaling, (double)P4EST_QUADRANT_LEN(ngbd_tmp[m].level)/(double)P4EST_ROOT_LEN);

    for(int d=0; d<2; ++d)
    {
      f_tmp = faces->q2f(ngbd_tmp[m].p.piggy3.local_num, 2*dir+d);
      if(f_tmp!=NO_VELOCITY && std::find(ngbd.begin(), ngbd.end(),f_tmp)==ngbd.end())
      {
#ifdef P4_TO_P8
        if(fabs(xyz[0]-faces->x_fr_f(f_tmp,dir))<EPS && fabs(xyz[1]-faces->y_fr_f(f_tmp,dir))<EPS && fabs(xyz[2]-faces->z_fr_f(f_tmp,dir))<EPS)
#else
        if(fabs(xyz[0]-faces->x_fr_f(f_tmp,dir))<EPS && fabs(xyz[1]-faces->y_fr_f(f_tmp,dir))<EPS)
#endif
        {
          double Fi_tmp = Fi_p[f_tmp];
          ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
          return Fi_tmp;
        }

        ngbd.push_back(f_tmp);
      }
    }
  }

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

  const PetscScalar *face_is_well_defined_p;
  ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(unsigned int m=0; m<ngbd.size(); m++)
  {
    p4est_locidx_t fm_idx = ngbd[m];
    if(face_is_well_defined_p[fm_idx] && std::find(interp_points.begin(), interp_points.end(),fm_idx)==interp_points.end() )
    {
      double xyz_t[P4EST_DIM];
      faces->xyz_fr_f(fm_idx, dir, xyz_t);
      for(int i=0; i<P4EST_DIM; ++i)
        xyz_t[i] = (xyz[i] - xyz_t[i]) / scaling;

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

      p.push_back(Fi_p[fm_idx] * w);

      for(int d=0; d<P4EST_DIM; ++d)
        if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
          nb[d].push_back(xyz_t[d]);

      interp_points.push_back(fm_idx);
    }
  }

  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  if(interp_points.size()==0)
    return 0;

  A.scale_by_maxabs(p);

#ifdef P4_TO_P8
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size(), nb[2].size());
#else
  return solve_lsqr_system(A, p, nb[0].size(), nb[1].size());
#endif
}
