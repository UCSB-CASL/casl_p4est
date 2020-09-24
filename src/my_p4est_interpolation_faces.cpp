#ifdef P4_TO_P8
#include <src/my_p8est_solve_lsqr.h>
#include "my_p8est_interpolation_faces.h"
#else
#include <src/my_p4est_solve_lsqr.h>
#include "my_p4est_interpolation_faces.h"
#endif

#include <algorithm>
#include <src/matrix.h>

void my_p4est_interpolation_faces_t::set_input(const Vec *F, const u_char& dir_, const size_t &n_vecs_, const u_char& degree_, Vec face_is_well_defined_dir_, BoundaryConditionsDIM *bc_array_)
{
  set_input_fields(F, n_vecs_, 1); // only for block_size of 1 with face-sampled fields
  this->face_is_well_defined_dir = face_is_well_defined_dir_;
  this->which_face  = dir_;
  this->bc_array    = bc_array_;
  this->degree      = degree_;
}

void my_p4est_interpolation_faces_t::operator ()(const double *xyz, double *results, const u_int&) const
{
  int rank_found = -1;
  std::vector<p4est_quadrant_t> remote_matches;
  try
  {
    clip_point_and_interpolate_all_on_the_fly(xyz, ALL_COMPONENTS, results, rank_found, remote_matches, true); // second input is dummy in this case, interpolate allows only bs_f == 1
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
    oss << "my_p4est_interpolation_faces_t::operator (): Point (" << xyz[0] << "," << xyz[1] << ONLY3D("," << xyz[2] <<) ") cannot be processed on-the-fly by process " << p4est->mpirank << ".";
    oss << "Process(es) likely to own a quadrant containing the point is (are) = ";
    for (size_t i = 0; i < remote_matches.size() - 1; i++)
      oss << remote_matches[i].p.piggy1.owner_rank << ", ";
    oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    throw std::invalid_argument(oss.str());
  }
  return;
}

void my_p4est_interpolation_faces_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const u_int &) const
{
  PetscErrorCode ierr;

  if(quad.p.piggy3.local_num > p4est->local_num_quadrants)
    std::cerr << "WARNING : interpolating face-sampled values at point " << xyz[0] << ", " << xyz[1] ONLY3D(<< ", " << xyz[2]) << " which belongs to the ghost layer of process " << p4est->mpirank << std::endl;

  const size_t n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f == 1); // not implemented for bs_f > 1 yet

  const double *xyz_min   = get_xyz_min();
  const double *xyz_max   = get_xyz_max();
  const bool *periodicity = get_periodicity();
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();

  bool is_on_wall = false;
  char neumann_wall[P4EST_DIM] = {DIM(0, 0, 0)};
  u_char nb_neumann_walls = 0;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    const char is_wall_ = (!periodicity[dim] && fabs(xyz[dim] - xyz_min[dim]) < EPS*tree_dimensions[dim] ? -1 : (!periodicity[dim] && fabs(xyz[dim] - xyz_max[dim]) < EPS*tree_dimensions[dim] ? +1 : 0));
    is_on_wall = is_on_wall || is_wall_ != 0;
    if(is_wall_!= 0 && degree >= 1 && bc_array != NULL && bc_array[which_face].wallType(xyz) == NEUMANN)
    {
      neumann_wall[dim] = is_wall_;
      nb_neumann_walls += abs(neumann_wall[dim]);
    }
  }
  if(is_on_wall && bc_array != NULL && bc_array[which_face].wallType(xyz) == DIRICHLET)
  {
    for (size_t k = 0; k < n_functions; ++k)
      results[k] = bc_array[which_face].wallValue(xyz);
    return;
  }

  /* gather the cell neighborhood and get the (logical) size of the smallest quadrant in that neighborhood */
  set_of_neighboring_quadrants ngbd; ngbd.clear();
  const p4est_qcoord_t smallest_logical_size_of_nearby_quad = ngbd_c->gather_neighbor_cells_of_cell(quad, ngbd, true);
  const double scaling = 0.5*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))*(double)smallest_logical_size_of_nearby_quad/(double)P4EST_ROOT_LEN;

  std::vector<const double *> Fi_p(n_functions);
  for (size_t k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  const PetscScalar *face_is_well_defined_dir_p;
  if(face_is_well_defined_dir != NULL){
    ierr = VecGetArrayRead(face_is_well_defined_dir, &face_is_well_defined_dir_p); CHKERRXX(ierr);
  }

  matrix_t A;
  A.resize(1, 1 + (degree >= 1 ? P4EST_DIM - nb_neumann_walls : 0) + (degree >= 2 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0)); // constant term + P4EST_DIM linear terms + (if second order) P4EST_DIM squared terms + 0.5*P4EST_DIM*(P4EST_DIM + 1) squared and crossed terms

  std::vector<double>* p = new std::vector<double>[n_functions];
  for (size_t k = 0; k < n_functions; ++k)
    p[k].resize(0);
  std::vector<double> nb[P4EST_DIM];
  const double min_w = 1e-6;
  const double inv_max_w = 1e-6;
  int row_idx = 0;

  // keep track of the faces that have already been dealt with using a set
  indexed_and_located_face f_tmp;
  std::set<indexed_and_located_face> face_ngbd; face_ngbd.clear();

  for (set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
  {
    for(u_char touch = 0; touch < 2; ++touch)
    {
      f_tmp.face_idx = faces->q2f(it->p.piggy3.local_num, 2*which_face + touch);

      if(f_tmp.face_idx != NO_VELOCITY && face_ngbd.find(f_tmp) == face_ngbd.end())
      {
        faces->xyz_fr_f(f_tmp.face_idx, which_face, f_tmp.xyz_face);
        bool point_is_on_face = true;
        for (u_char dim = 0; dim < P4EST_DIM && point_is_on_face; ++dim)
          point_is_on_face = point_is_on_face && fabs(xyz[dim] - f_tmp.xyz_face[dim]) < EPS*tree_dimensions[dim];
        if((face_is_well_defined_dir == NULL || face_is_well_defined_dir_p[f_tmp.face_idx]) && point_is_on_face)
        {
          for (size_t k = 0; k < n_functions; ++k) {
            results[k] = Fi_p[k][f_tmp.face_idx];
            ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
          }
          if(face_is_well_defined_dir != NULL){
            ierr = VecRestoreArrayRead(face_is_well_defined_dir, &face_is_well_defined_dir_p); CHKERRXX(ierr);
          }
          delete [] p;
          return;
        }

        face_ngbd.insert(f_tmp);

        if(face_is_well_defined_dir == NULL || face_is_well_defined_dir_p[f_tmp.face_idx])
        {
          double xyz_t[P4EST_DIM] = {DIM(f_tmp.xyz_face[0], f_tmp.xyz_face[1], f_tmp.xyz_face[2])};

          for(u_char dim = 0; dim < P4EST_DIM; ++dim)
          {
            xyz_t[dim] = xyz_t[dim] - xyz[dim];
            if(periodicity[dim])
            {
              const double pp = xyz_t[dim]/(xyz_max[dim] - xyz_min[dim]);
              xyz_t[dim] -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[dim] - xyz_min[dim]);
            }
            xyz_t[dim] /=  scaling;
          }

          const double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

          unsigned char col_idx = 0;
          A.set_value(row_idx, col_idx++, w); // constant term
          if(degree >= 1)
            for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
              if(neumann_wall[dim] == 0)
                A.set_value(row_idx, col_idx++, xyz_t[dim]*w); // linear terms
          P4EST_ASSERT(col_idx == 1 + (degree >= 1 ? P4EST_DIM - nb_neumann_walls : 0));
          if(degree >= 2)
            for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
              for (unsigned char dd = dim; dd < P4EST_DIM; ++dd)
                A.set_value(row_idx, col_idx++, xyz_t[dim]*xyz_t[dd]*w); // quadratic terms
          P4EST_ASSERT(col_idx == 1 + (degree >= 1 ? P4EST_DIM - nb_neumann_walls : 0) + (degree >= 2 ? P4EST_DIM*(P4EST_DIM + 1)/2 : 0));

          for (size_t k = 0; k < n_functions; ++k)
          {
            const double neumann_term = (degree >= 1 && nb_neumann_walls > 0 ? SUMD(neumann_wall[0]*bc_array[which_face].wallValue(xyz)*xyz_t[0]*scaling, neumann_wall[1]*bc_array[which_face].wallValue(xyz)*xyz_t[1]*scaling, neumann_wall[2]*bc_array[which_face].wallValue(xyz)*xyz_t[2]*scaling) : 0.0);
            p[k].push_back((Fi_p[k][f_tmp.face_idx] - neumann_term)* w);
          }

          for(unsigned char d = 0; d < P4EST_DIM; ++d)
            if(std::find(nb[d].begin(), nb[d].end(), xyz_t[d]) == nb[d].end())
              nb[d].push_back(xyz_t[d]);

          row_idx++;
        }
      }
    }
  }

  for (size_t k = 0; k < n_functions; ++k)
  {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    results[k] = 0.0;
  }
  if(face_is_well_defined_dir != NULL){
    ierr = VecRestoreArrayRead(face_is_well_defined_dir, &face_is_well_defined_dir_p); CHKERRXX(ierr);
  }

  if(row_idx == 0)
  {
    delete [] p;
    return;
  }

  A.scale_by_maxabs(p, n_functions);

  solve_lsqr_system(A, p, n_functions, results, DIM(nb[0].size(), nb[1].size(), nb[2].size()), degree, nb_neumann_walls);
  delete [] p;

  return;
}
