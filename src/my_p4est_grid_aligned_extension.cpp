#ifdef P4_TO_P8
#include "my_p8est_grid_aligned_extension.h"
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_grid_aligned_extension.h"
#include <src/my_p4est_macros.h>
#endif

#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <petsclog.h>

#undef MAX
#undef MIN

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
//extern PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_grid_aligned_extension_t::initialize(Vec phi, unsigned int order, bool weighted, unsigned int max_iters, double band_extend, double band_check, Vec normal[], Vec mask)
{
  initialized_ = true;
  phi_         = phi;
  order_       = order;
  weighted_    = weighted;
  max_iters_   = max_iters;
  band_extend_ = band_extend;
  band_check_  = band_check;
  num_points_to_extend_ = 0;

  // set extrapolation weights
  extrapolation_weights_.resize(order_+1);

  switch (order_) {
    case 0:
      extrapolation_weights_[0] =  1;
    break;
    case 1:
      extrapolation_weights_[0] =  2;
      extrapolation_weights_[1] = -1;
    break;
    case 2:
      extrapolation_weights_[0] =  3;
      extrapolation_weights_[1] = -3;
      extrapolation_weights_[2] =  1;
    break;
    case 3:
      extrapolation_weights_[0] =  4;
      extrapolation_weights_[1] = -6;
      extrapolation_weights_[2] =  4;
      extrapolation_weights_[3] = -1;
    break;
    default:
      throw;
  }

  // set number of directions used
  num_dirs_ = weighted_ ? P4EST_DIM : 1;

  double dxyz[P4EST_DIM];
  double diag_min;
  get_dxyz_min(p4est_, dxyz, NULL, &diag_min);

  // compute normal
  bool is_normal_owned = false;
  Vec normal_own[P4EST_DIM];
  if (normal == NULL) {
    is_normal_owned = true;
    normal = normal_own;
    foreach_dimension(dim) {
      ierr = VecCreateGhostNodes(p4est_, nodes_, &normal[dim]); CHKERRXX(ierr);
    }
    compute_normals(*ngbd_, phi_, normal);
  }

  ierr = VecCreateGhostNodes(p4est_, nodes_, &well_defined_); CHKERRXX(ierr);

  double *mask_ptr;
  double *phi_ptr;
  double *well_defined_ptr;
  ierr = VecGetArray(phi_, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(well_defined_, &well_defined_ptr); CHKERRXX(ierr);
  if (mask == NULL) {
    mask_ptr = phi_ptr;
  } else {
    ierr = VecGetArray(mask, &mask_ptr); CHKERRXX(ierr);
  }

  double *normal_ptr[P4EST_DIM];
  foreach_dimension(dim) {
    ierr = VecGetArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
  }

  foreach_node(n, nodes_) {
    if (mask_ptr[n] <= 0) {
      well_defined_ptr[n] = 1;
    } else {
      well_defined_ptr[n] = -1;

      if (phi_ptr[n] < band_extend_*diag_min)
      {
        double xyz[P4EST_DIM];

        if (weighted_) {
          double dir0[P4EST_DIM];
          double dir1[P4EST_DIM];

          dir0[0] = normal_ptr[0][n] > 0 ? -1 : 1;
          dir0[1] = normal_ptr[1][n] > 0 ? -1 : 1;

          if (fabs(normal_ptr[0][n]) > fabs(normal_ptr[1][n])) {
            dir1[0] = normal_ptr[0][n] > 0 ? -1 : 1;
            dir1[1] = 0;
          } else {
            dir1[0] = 0;
            dir1[1] = normal_ptr[1][n] > 0 ? -1 : 1;
          }

          double det = dir0[0]*dir1[1] - dir0[1]*dir1[0];
          double a = (normal_ptr[0][n]*dir1[1] - normal_ptr[1][n] *dir1[0])/det;
          double b = (normal_ptr[1][n]*dir0[0] - normal_ptr[0][n] *dir0[1])/det;

          EXECD(dir0[0] *= dxyz[0], dir0[1] *= dxyz[1], dir0[2] *= dxyz[2]);
          EXECD(dir1[0] *= dxyz[0], dir1[1] *= dxyz[1], dir1[2] *= dxyz[2]);

          node_xyz_fr_n(n, p4est_, nodes_, xyz);
          for (unsigned int j = 0; j < order_+1; ++j) {
            // TODO: check for walls
            foreach_dimension(dim) { xyz[dim] += dir0[dim]; }
            interp_.add_point((num_points_to_extend_*num_dirs_+0)*(order_+1) + j, xyz);
          }

          node_xyz_fr_n(n, p4est_, nodes_, xyz);
          for (unsigned int j = 0; j < order_+1; ++j) {
            // TODO: check for walls
            foreach_dimension(dim) { xyz[dim] += dir1[dim]; }
            interp_.add_point((num_points_to_extend_*num_dirs_+1)*(order_+1) + j, xyz);
          }

          num_points_to_extend_++;
          points_to_extend_.push_back(n);
          mixing_weights_.push_back(a/(a+b));
          mixing_weights_.push_back(b/(a+b));
        } else {
          // determine direction
          double dir[P4EST_DIM];
          double max_neg_projection = 1;

#ifdef P4_TO_P8
          for (int k = -1; k <= 1; k++)
#endif
            for (int j = -1; j <= 1; j++)
              for (int i = -1; i <= 1; i++) {
                if (SUMD(fabs(i), fabs(j), fabs(k)) != 0) {
                  double proj = SUMD(normal_ptr[0][n]*double(i)*dxyz[0],
                      normal_ptr[1][n]*double(j)*dxyz[1],
                      normal_ptr[2][n]*double(k)*dxyz[2])
                      / ABSD(double(i)*dxyz[0], double(j)*dxyz[1], double(k)*dxyz[2]);

                  if (proj < max_neg_projection) {
                    max_neg_projection = proj;
                    EXECD(dir[0] = double(i), dir[1] = double(j), dir[2] = double(k));
                  }
                }
              }

          EXECD(dir[0] *= dxyz[0], dir[1] *= dxyz[1], dir[2] *= dxyz[2]);

          node_xyz_fr_n(n, p4est_, nodes_, xyz);
          for (unsigned int j = 0; j < order_+1; ++j) {
            // TODO: check for walls
            foreach_dimension(dim) { xyz[dim] += dir[dim]; }
            interp_.add_point(num_points_to_extend_*(order_+1) + j, xyz);
          }

          num_points_to_extend_++;
          points_to_extend_.push_back(n);
          mixing_weights_.push_back(1);
        }
      }
    }
  }

  ierr = VecRestoreArray(phi_, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(well_defined_, &well_defined_ptr); CHKERRXX(ierr);
  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_ptr); CHKERRXX(ierr);
  }

  foreach_dimension(dim) {
    ierr = VecRestoreArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
  }

  if (is_normal_owned) {
    foreach_dimension(dim) {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }
  }
}

void my_p4est_grid_aligned_extension_t::extend(unsigned int num_fields, Vec fields[])
{
  if (!initialized_) {
    throw std::domain_error("Grid aligned extrapolator is not initialized");
  }

  double diag_min;
  get_dxyz_min(p4est_, NULL, NULL, &diag_min);

  double *well_defined_tmp_ptr;
  Vec     well_defined_tmp;
  ierr = VecDuplicate(well_defined_, &well_defined_tmp); CHKERRXX(ierr);
  ierr = VecCopyGhost(well_defined_,  well_defined_tmp); CHKERRXX(ierr);

  // allocate arrays to store interpolated points
  vector<double> interpolated_well_defined(num_points_to_extend_*num_dirs_*(order_+1));
  vector<vector<double> > interpolated_fields(num_fields, vector<double> (num_points_to_extend_*num_dirs_*(order_+1)));
  vector<double *> pointers_to_interpolated_fields(num_fields);
  for (unsigned int i = 0; i < num_fields; ++i) {
    pointers_to_interpolated_fields[i] = interpolated_fields[i].data();
  }

  double *phi_ptr;
  ierr = VecGetArray(phi_, &phi_ptr); CHKERRXX(ierr);

  bool not_done = true;
  unsigned int iteration = 0;

  while (not_done && (iteration < max_iters_)) {
    iteration++;

//    PetscPrintf(p4est_->mpicomm, "Iteration: %d\n", iteration);

    // interpolate well_defined flag
    interp_.set_input(well_defined_tmp, linear);
    interp_.interpolate(interpolated_well_defined.data());

    // interpolate data
    interp_.set_input(fields, interp_method_, num_fields);
    interp_.interpolate(pointers_to_interpolated_fields.data());

    ierr = VecGetArray(well_defined_tmp, &well_defined_tmp_ptr); CHKERRXX(ierr);

    vector<double *> fields_ptr(num_fields, NULL);
    for (unsigned int j = 0; j < num_fields; ++j) {
      ierr = VecGetArray(fields[j], &fields_ptr[j]); CHKERRXX(ierr);
    }

    not_done = false;
    // loop through all points that need extension
    for (unsigned int j = 0; j < num_points_to_extend_; ++j) {
      // check whether a node already have a well-defined value
      p4est_locidx_t n = points_to_extend_[j];
      if (well_defined_tmp_ptr[n] != 1) {
        // check whether interpolated values are all well-defined
        bool neighbors_well_defined = true;
        for (unsigned int dir = 0; dir < num_dirs_; ++dir) {
          for (unsigned int k = 0; k < order_+1; ++k) {
            neighbors_well_defined = neighbors_well_defined &&
                                     (interpolated_well_defined[(j*num_dirs_+dir)*(order_+1) + k] == 1);
          }
        }

        if (neighbors_well_defined) {
          well_defined_tmp_ptr[n] = 1;
          for (unsigned int k = 0; k < num_fields; ++k) {
            double result = 0;
            for (unsigned int dir = 0; dir < num_dirs_; ++dir) {
              double result_in_dir = 0;
              for (unsigned int pt = 0; pt < order_ + 1; ++pt) {
                result_in_dir += extrapolation_weights_[pt]*interpolated_fields[k][(j*num_dirs_+dir)*(order_+1) + pt];
              }
              result += mixing_weights_[j*num_dirs_ + dir]*result_in_dir;
            }
            fields_ptr[k][n] = result;
          }
        } else if (phi_ptr[n] < band_check_*diag_min) {
          not_done = true;
        }
      }
    }

    ierr = MPI_Allreduce(MPI_IN_PLACE, &not_done, 1, MPI_C_BOOL, MPI_LOR, p4est_->mpicomm);

    ierr = VecRestoreArray(well_defined_tmp, &well_defined_tmp_ptr); CHKERRXX(ierr);
    for (unsigned int j = 0; j < num_fields; ++j) {
      ierr = VecRestoreArray(fields[j], &fields_ptr[j]); CHKERRXX(ierr);
    }
  }

  ierr = VecRestoreArray(phi_, &phi_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(well_defined_tmp); CHKERRXX(ierr);
}

