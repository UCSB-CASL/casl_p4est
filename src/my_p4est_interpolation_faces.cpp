#ifdef P4_TO_P8
#include "my_p4est_interpolation_faces.h"
#else
#include "my_p4est_interpolation_faces.h"
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>


my_p4est_interpolation_faces_t::my_p4est_interpolation_faces_t(const my_p4est_node_neighbors_t* ngbd_n, const my_p4est_faces_t *faces)
  : my_p4est_interpolation_t(ngbd_n), faces(faces), ngbd_c(faces->ngbd_c), face_is_well_defined(NULL), bc(NULL)
{
  p4est_topidx_t vtx_0_max    = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  p4est_topidx_t vtx_0_min    = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + 0];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    tree_dimension[dim]       = p4est->connectivity->vertices[3*vtx_0_max+dim] - p4est->connectivity->vertices[3*vtx_0_min + dim];
}


#ifdef P4_TO_P8
void my_p4est_interpolation_faces_t::set_input(Vec *F, int dir, unsigned int n_vecs_, int order, Vec face_is_well_defined, BoundaryConditions3D *bc)
#else
void my_p4est_interpolation_faces_t::set_input(Vec *F, int dir, unsigned int n_vecs_, int order, Vec face_is_well_defined, BoundaryConditions2D *bc)
#endif
{
  set_input(F, n_vecs_);
  this->face_is_well_defined = face_is_well_defined;
  this->dir = dir;
  this->bc = bc;
  this->order = order;
}


#ifdef P4_TO_P8
void my_p4est_interpolation_faces_t::operator ()(double x, double y, double z, double *results) const
#else
void my_p4est_interpolation_faces_t::operator ()(double x, double y, double *results) const
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
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = is_periodic(p4est,i) ?  xyz_clip[i]-(xyz_max[i]-xyz_min[i]) : xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = is_periodic(p4est,i) ?  xyz_clip[i]+(xyz_max[i]-xyz_min[i]) : xyz_min[i];
  }

  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);

//  if(rank_found!=-1)
  if(rank_found == p4est->mpirank)
  {
    interpolate(best_match, xyz, results);
    return;
  }

  throw std::invalid_argument("[ERROR]: my_p4est_interpolation_faces_t->interpolate(): the point does not belong to the local forest.");
}


void my_p4est_interpolation_faces_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results) const
{
  PetscErrorCode ierr;

  unsigned int n_functions = n_vecs();

  if(bc!=NULL && bc->wallType(xyz)==DIRICHLET &&
     (((fabs(xyz[0]-xyz_min[0])<EPS*tree_dimension[0] || fabs(xyz[0]-xyz_max[0])<EPS*tree_dimension[0]) && (!is_periodic(p4est, dir::x)))
      || ((fabs(xyz[1]-xyz_min[1])<EPS*tree_dimension[1] || fabs(xyz[1]-xyz_max[1])<EPS*tree_dimension[1]) && (!is_periodic(p4est, dir::y)))
    #ifdef P4_TO_P8
      || ((fabs(xyz[2]-xyz_min[2])<EPS*tree_dimension[2] || fabs(xyz[2]-xyz_max[2])<EPS*tree_dimension[2]) && (!is_periodic(p4est, dir::z)))
    #endif
      ))
  {
    for (unsigned int k = 0; k < n_functions; ++k)
      results[k] = bc->wallValue(xyz);
    return;
  }


#ifdef P4_TO_P8
  double qh = MIN(tree_dimension[0], tree_dimension[1], tree_dimension[2]);
#else
  double qh = MIN(tree_dimension[0], tree_dimension[1]);
#endif
  double scaling = .5 * qh*(double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN;

  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  /* gather the neighborhood */
  std::vector<p4est_quadrant_t> ngbd_tmp;
  std::vector<p4est_locidx_t> ngbd;
  p4est_locidx_t f_tmp;

  for(int i=-1; i<2; ++i)
    for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
      for(int k=-1; k<2; ++k)
//        if((i==0 && j==0) || (i==0 && k==0) || (j==0 && k==0))
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j, k);
#else
//      if(i==0 || j==0)
        ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j);
#endif

  int nb_ngbd = ngbd_tmp.size();
  for(int m=0; m<nb_ngbd; ++m)
  {
    for(int i=-1; i<2; ++i)
      for(int j=-1; j<2; ++j)
#ifdef P4_TO_P8
        for(int k=-1; k<2; ++k)
//          if((i==0 && j==0) || (i==0 && k==0) || (j==0 && k==0))
            ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, ngbd_tmp[m].p.piggy3.local_num, ngbd_tmp[m].p.piggy3.which_tree, i, j, k);
#else
//        if(i==0 || j==0)
          ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, ngbd_tmp[m].p.piggy3.local_num, ngbd_tmp[m].p.piggy3.which_tree, i, j);
#endif
  }

  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  const PetscScalar *face_is_well_defined_p;
  if(face_is_well_defined!=NULL)
    ierr = VecGetArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(unsigned int m=0; m<ngbd_tmp.size(); ++m)
  {
    for(int d=0; d<2; ++d)
    {
      f_tmp = faces->q2f(ngbd_tmp[m].p.piggy3.local_num, 2*dir+d);

      if(f_tmp!=NO_VELOCITY && std::find(ngbd.begin(), ngbd.end(),f_tmp)==ngbd.end())
      {
#ifdef P4_TO_P8
        if((face_is_well_defined==NULL || face_is_well_defined_p[f_tmp]) &&
           fabs(xyz[0]-faces->x_fr_f(f_tmp,dir))<EPS*tree_dimension[0] && fabs(xyz[1]-faces->y_fr_f(f_tmp,dir))<EPS*tree_dimension[1] && fabs(xyz[2]-faces->z_fr_f(f_tmp,dir))<EPS*tree_dimension[2])
#else
        if((face_is_well_defined==NULL || face_is_well_defined_p[f_tmp]) &&
           fabs(xyz[0]-faces->x_fr_f(f_tmp,dir))<EPS*tree_dimension[0] && fabs(xyz[1]-faces->y_fr_f(f_tmp,dir))<EPS*tree_dimension[1])
#endif
        {
          for (unsigned int k = 0; k < n_functions; ++k) {
            results[k] = Fi_p[k][f_tmp];
            ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
          }
          if(face_is_well_defined!=NULL)
            ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);
          return;
        }

        ngbd.push_back(f_tmp);
      }
    }
  }

  std::vector<p4est_locidx_t> interp_points;
  matrix_t A;
  A.resize(1,(1+P4EST_DIM+((order>=2)?(P4EST_DIM+(P4EST_DIM*(P4EST_DIM-1))/2):0))); // constant term + P4EST_DIM linear terms + (if second order) P4EST_DIM squared terms + 0.5*P4EST_DIM*(P4EST_DIM-1) crossed terms

  std::vector<double> p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k)
    p[k].resize(0);
  std::vector<double> nb[P4EST_DIM];

  double min_w = 1e-6;
  double inv_max_w = 1e-6;

  for(unsigned int m=0; m<ngbd.size(); m++)
  {
    p4est_locidx_t fm_idx = ngbd[m];
    if((face_is_well_defined==NULL || face_is_well_defined_p[fm_idx]) && std::find(interp_points.begin(), interp_points.end(),fm_idx)==interp_points.end() )
    {
      double xyz_t[P4EST_DIM];
      faces->xyz_fr_f(fm_idx, dir, xyz_t);

      for(int i=0; i<P4EST_DIM; ++i)
      {
        double rel_dist = (xyz[i] - xyz_t[i]);
        if(is_periodic(p4est, i))
          for (short cc = -1; cc < 2; cc+=2)
            if(fabs((xyz[i] - xyz_t[i] + ((double) cc)*(xyz_max[i] - xyz_min[i]))) < fabs(rel_dist))
              rel_dist = (xyz[i] - xyz_t[i] + ((double) cc)*(xyz_max[i] - xyz_min[i]));
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
      if(order>=2)
      {
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 5, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 6, xyz_t[0]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 7, xyz_t[1]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 8, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interp_points.size(), 9, xyz_t[2]*xyz_t[2] * w);
      }
#else
      A.set_value(interp_points.size(), 0, 1                 * w);
      A.set_value(interp_points.size(), 1, xyz_t[0]          * w);
      A.set_value(interp_points.size(), 2, xyz_t[1]          * w);
      if(order>=2)
      {
        A.set_value(interp_points.size(), 3, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interp_points.size(), 4, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interp_points.size(), 5, xyz_t[1]*xyz_t[1] * w);
      }
#endif

      for (unsigned int k = 0; k < n_functions; ++k)
        p[k].push_back(Fi_p[k][fm_idx] * w);

      for(int d=0; d<P4EST_DIM; ++d)
        if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
          nb[d].push_back(xyz_t[d]);

      interp_points.push_back(fm_idx);
    }
  }

  for (unsigned int k = 0; k < n_functions; ++k)
  {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    results[k] = 0.0;
  }
  if(face_is_well_defined!=NULL)
    ierr = VecRestoreArrayRead(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

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
