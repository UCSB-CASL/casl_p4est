#ifdef P4_TO_P8
#include "my_p8est_level_set_faces.h"
#include <src/point3.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include "my_p4est_level_set_faces.h"
#include <src/point2.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_refine_coarsen.h>
#endif

#include "petsc_compatibility.h"
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
extern PetscLogEvent log_my_p4est_level_set_faces_geometric_extrapolation_over_interface;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_level_set_faces_t::geometric_extrapolation_over_interface(Vec face_field_dir, const my_p4est_interpolation_nodes_t &interp_phi, const my_p4est_interpolation_nodes_t &interp_grad_phi,
                                                                        const BoundaryConditionsDIM &bc_dir, const unsigned char &dir, Vec face_is_well_defined_dir,
                                                                        Vec dxyz_hodge_dir, const unsigned char& degree, const unsigned int& band_to_extend) const
{
#ifdef CASL_THROWS
  if(bc_dir.interfaceType() == NOINTERFACE)
    throw std::invalid_argument("my_p4est_level_set_faces_t::geometric_extrapolation_over_interface(): no interface defined in the boundary condition ... needs to be dirichlet, neumann or mixed.");
  if(degree > 2)
    throw std::invalid_argument("my_p4est_level_set_faces_t::geometric_extrapolation_over_interface(): the degree of the extrapolant polynomial must be less than or equal to 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_faces_geometric_extrapolation_over_interface, phi, field_dir, 0, 0); CHKERRXX(ierr);

  // get the (max) number of points to sample in negative domain for every node where extrapolation is required
  const unsigned char nsamples_across = number_of_samples_across_the_interface_for_geometric_extrapolation(degree, bc_dir.interfaceType());
  P4EST_ASSERT(nsamples_across <= 2);

  const double* dxyz_smallest = faces->get_smallest_dxyz();
  const double smallest_diag = sqrt(SUMD(SQR(dxyz_smallest[0]), SQR(dxyz_smallest[1]), SQR(dxyz_smallest[2])));

  // supposed "levels" of the levelset for sampling the values along the normal direction
  // (--> not the actual values of the levelset function at sampled points but more like
  // distances between the sampling points)
  // "signed distances" from the 0-level in the negative normal direction
  const double phi_sampling_levels[2] = {-2.0*smallest_diag, -3.0*smallest_diag};

  // prepare objects to sample field values and interface boundary conditions at possibly nonlocal points
  my_p4est_interpolation_nodes_t interp_bc(ngbd_n); // could be _nodes, _cells or _faces, it is irrelevant in this case, we do use base method from my_p4est_interpolation_t anyways
  my_p4est_interpolation_faces_t *interp_dxyz_hodge_dir = NULL;
  if(dxyz_hodge_dir != NULL)
  {
    interp_dxyz_hodge_dir = new my_p4est_interpolation_faces_t(ngbd_n, faces); // for correcting DIRICHLET boundary conditions with appropriate partial derivative of hodge variable if required
    interp_dxyz_hodge_dir->set_input(dxyz_hodge_dir, dir, 1, face_is_well_defined_dir); // degree of lsqr interpolation is 1 because input data is 1st order accurate at best anyways
  }

  my_p4est_interpolation_faces_t interp_field(ngbd_n, faces);
  interp_field.set_input(face_field_dir, dir, 2, face_is_well_defined_dir);

  std::map<p4est_locidx_t, data_for_geometric_extapolation> face_data_for_extrapolation; face_data_for_extrapolation.clear();

  const PetscScalar *face_is_well_defined_dir_p;
  ierr = VecGetArrayRead(face_is_well_defined_dir, &face_is_well_defined_dir_p); CHKERRXX(ierr);

  /* now buffer the interpolation points */
  for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx)
    if(!face_is_well_defined_dir_p[f_idx])
    {
      double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz_face);
      const double phi_f = interp_phi(xyz_face);

      if(phi_f < band_to_extend*smallest_diag)
      {
        double grad_phi[P4EST_DIM];
        interp_grad_phi(xyz_face, grad_phi);
        add_dof_to_extrapolation_map(face_data_for_extrapolation, f_idx, xyz_face, phi_f, grad_phi, nsamples_across, phi_sampling_levels,
                                     &interp_field, &interp_bc, interp_dxyz_hodge_dir);
      }
    }

  std::vector<double> field_samples(nsamples_across*face_data_for_extrapolation.size());
  std::vector<bc_sample> interface_bc(face_data_for_extrapolation.size());
  interp_field.interpolate(field_samples.data());
  std::vector<double> *calculated_bc_dxyz_hodge_dir = (interp_dxyz_hodge_dir != NULL ? new std::vector<double>(face_data_for_extrapolation.size(), 0.0) : NULL);
  if(interp_dxyz_hodge_dir != NULL)
    interp_dxyz_hodge_dir->interpolate(calculated_bc_dxyz_hodge_dir->data());
  interp_bc.evaluate_interface_bc(bc_dir, interface_bc.data());

  /* now compute the extrapolated values */
  double *face_field_dir_p;
  ierr = VecGetArray(face_field_dir, &face_field_dir_p); CHKERRXX(ierr);
  std::map<p4est_locidx_t, data_for_geometric_extapolation>::iterator it;
  for (size_t k = 0; k < faces->get_layer_size(dir); ++k) {
    p4est_locidx_t f_idx = faces->get_layer_face(dir, k);
    it = face_data_for_extrapolation.find(f_idx);
    if(it != face_data_for_extrapolation.end())
      face_field_dir_p[f_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples_across, field_samples, &interface_bc, calculated_bc_dxyz_hodge_dir);
  }
  ierr = VecGhostUpdateBegin(face_field_dir, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < faces->get_local_size(dir); ++k) {
    p4est_locidx_t f_idx = faces->get_local_face(dir, k);
    it = face_data_for_extrapolation.find(f_idx);
    if(it != face_data_for_extrapolation.end())
      face_field_dir_p[f_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples_across, field_samples, &interface_bc, calculated_bc_dxyz_hodge_dir);
  }
  ierr = VecGhostUpdateEnd(face_field_dir, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(face_field_dir, &face_field_dir_p); CHKERRXX(ierr);
  if(calculated_bc_dxyz_hodge_dir != NULL)
    delete calculated_bc_dxyz_hodge_dir;
  if(interp_dxyz_hodge_dir != NULL)
    delete interp_dxyz_hodge_dir;

  ierr = PetscLogEventEnd(log_my_p4est_level_set_faces_geometric_extrapolation_over_interface, phi, field_dir, 0, 0); CHKERRXX(ierr);

  return;
}
