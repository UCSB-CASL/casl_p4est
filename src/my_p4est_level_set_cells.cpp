#ifdef P4_TO_P8
#include "my_p8est_level_set_cells.h"
#include <src/point3.h>
#include <src/cube3.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include "my_p4est_level_set_cells.h"
#include <src/point2.h>
#include <src/cube2.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
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
extern PetscLogEvent log_my_p4est_level_set_cells_geometric_extrapolation_over_interface;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double my_p4est_level_set_cells_t::integrate_over_interface(Vec node_sampled_phi, Vec cell_field) const
{
  PetscErrorCode ierr;

#ifdef P4_TO_P8
  Cube3 cube;
  OctValue phi_vals;
#else
  Cube2 cube;
  QuadValue phi_vals;
#endif

  double sum = 0.0;
  const double *node_sampled_phi_read_p, *cell_field_read_p;
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();
  ierr = VecGetArrayRead(node_sampled_phi, &node_sampled_phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cell_field, &cell_field_read_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);

      const double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        cube.xyz_mmm[dir] = xyz_quad[dir] - tree_dimensions[dir]*dmin/2.0;
        cube.xyz_ppp[dir] = xyz_quad[dir] + tree_dimensions[dir]*dmin/2.0;
      }

#ifdef P4_TO_P8
      for (unsigned char vz = 0; vz < 2; ++vz)
#endif
        for (unsigned char vy = 0; vy < 2; ++vy)
          for (unsigned char vx = 0; vx < 2; ++vx)
            phi_vals.val[SUMD((1 << (P4EST_DIM - 1))*vx, (1 << (P4EST_DIM - 2))*vy, vz)] = node_sampled_phi_read_p[nodes->local_nodes[quad_idx*P4EST_CHILDREN + SUMD(vx, 2*vy, 4*vz)]];

#ifdef P4_TO_P8
      sum += cell_field_read_p[quad_idx]*cube.interface_Area_In_Cell(phi_vals);
#else
      sum += cell_field_read_p[quad_idx]*cube.interface_Length_In_Cell(phi_vals);
#endif
    }
  }

  ierr = VecRestoreArrayRead(node_sampled_phi, &node_sampled_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cell_field, &cell_field_read_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  return sum;
}

void my_p4est_level_set_cells_t::normal_vector_weighted_integral_over_interface(Vec node_sampled_phi, Vec cell_weight, double *integral, Vec node_sampled_grad_phi) const
{
  PetscErrorCode ierr;

  const double *node_sampled_phi_read_p, *cell_weight_read_p, *node_sampled_grad_phi_read_p;
  ierr = VecGetArrayRead(node_sampled_phi, &node_sampled_phi_read_p);           CHKERRXX(ierr);
  ierr = VecGetArrayRead(cell_weight, &cell_weight_read_p);                     CHKERRXX(ierr);
  ierr = VecGetArrayRead(node_sampled_grad_phi, &node_sampled_grad_phi_read_p); CHKERRXX(ierr);
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();

#ifdef P4_TO_P8
  Cube3 cube;
  OctValue n_vals[P4EST_DIM];
  OctValue phi_vals;
#else
  Cube2 cube;
  QuadValue n_vals[P4EST_DIM];
  QuadValue phi_vals;
#endif

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      const p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);

      const double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        cube.xyz_mmm[dir] = xyz_quad[dir] - tree_dimensions[dir]*dmin/2.0;
        cube.xyz_ppp[dir] = xyz_quad[dir] + tree_dimensions[dir]*dmin/2.0;
      }

#ifdef P4_TO_P8
      for (unsigned char vz = 0; vz < 2; ++vz)
#endif
        for (unsigned char vy = 0; vy < 2; ++vy)
          for (unsigned char vx = 0; vx < 2; ++vx)
          {
            const unsigned char l_idx = SUMD((1 << (P4EST_DIM - 1))*vx, (1 << (P4EST_DIM - 2))*vy, vz);
            const unsigned char r_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + SUMD(vx, 2*vy, 4*vz)];
            double normal[P4EST_DIM] = {DIM(node_sampled_grad_phi_read_p[P4EST_DIM*r_idx], node_sampled_grad_phi_read_p[P4EST_DIM*r_idx + 1], node_sampled_grad_phi_read_p[P4EST_DIM*r_idx + 2])};
            const double mag_normal = sqrt(SUMD(SQR(normal[0]), SQR(normal[1]), SQR(normal[0])));
            phi_vals.val[l_idx] = node_sampled_phi_read_p[r_idx];
            for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
              n_vals[dir].val[l_idx] = (mag_normal > EPS ? normal[dir]/mag_normal : 0.0);
          }

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        integral[dir] += cell_weight_read_p[quad_idx]*cube.integrate_Over_Interface(n_vals[dir], phi_vals);
    }
  }

  ierr = VecRestoreArrayRead(node_sampled_grad_phi, &node_sampled_grad_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(node_sampled_phi, &node_sampled_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cell_weight, &cell_weight_read_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, integral, P4EST_DIM, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}

double my_p4est_level_set_cells_t::integrate(Vec node_sampled_phi, Vec cell_field) const
{
  PetscErrorCode ierr;

#ifdef P4_TO_P8
  Cube3 cube;
  OctValue phi_vals;
#else
  Cube2 cube;
  QuadValue phi_vals;
#endif

  double sum = 0.0;
  const double *node_sampled_phi_read_p, *cell_field_read_p;
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();
  ierr = VecGetArrayRead(node_sampled_phi, &node_sampled_phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(cell_field, &cell_field_read_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      const double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double xyz_quad[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_quad);

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        cube.xyz_mmm[dir] = xyz_quad[dir] - 0.5*dmin*tree_dimensions[dir];
        cube.xyz_ppp[dir] = xyz_quad[dir] + 0.5*dmin*tree_dimensions[dir];
      }

#ifdef P4_TO_P8
      for (unsigned char vz = 0; vz < 2; ++vz)
#endif
        for (unsigned char vy = 0; vy < 2; ++vy)
          for (unsigned char vx = 0; vx < 2; ++vx)
            phi_vals.val[SUMD((1 << (P4EST_DIM - 1))*vx, (1 << (P4EST_DIM - 2))*vy, vz)] = node_sampled_phi_read_p[nodes->local_nodes[quad_idx*P4EST_CHILDREN + SUMD(vx, 2*vy, 4*vz)]];

#ifdef P4_TO_P8
      sum += cube.volume_In_Negative_Domain(phi_vals)*cell_field_read_p[quad_idx];
#else
      sum += cube.area_In_Negative_Domain(phi_vals)*cell_field_read_p[quad_idx];
#endif
    }
  }

  ierr = VecRestoreArrayRead(node_sampled_phi, &node_sampled_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cell_field, &cell_field_read_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  return sum;
}

void my_p4est_level_set_cells_t::geometric_extrapolation_over_interface(Vec cell_field, Vec node_sampled_phi, const my_p4est_interpolation_nodes_t& interp_grad_phi,
                                                                        const BoundaryConditionsDIM& bc, const unsigned char& degree, const unsigned int& band_to_extend) const
{
#ifdef CASL_THROWS
  if(bc.interfaceType() == NOINTERFACE)
    throw std::invalid_argument("my_p4est_level_set_cells_t::geometric_extrapolation_over_interface(): no interface defined in the boundary condition ... needs to be dirichlet, neumann or mixed.");
  if(degree > 2)
    throw std::invalid_argument("my_p4est_level_set_cells_t::geometric_extrapolation_over_interface(): the degree of the extrapolant polynomial must be less than or equal to 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_cells_geometric_extrapolation_over_interface, cell_field, node_sampled_phi, 0, 0); CHKERRXX(ierr);

  // get the (max) number of points to sample in negative domain for every node where extrapolation is required
  const unsigned char nsamples = number_of_samples_across_the_interface_for_geometric_extrapolation(degree, bc.interfaceType());
  P4EST_ASSERT(nsamples <= 2);

  /* find smallest diag */
  const splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  const double *tree_dimensions = ngbd_c->get_tree_dimensions();
  const double diag_min = sqrt(SUMD(SQR(tree_dimensions[0]), SQR(tree_dimensions[1]), SQR(tree_dimensions[2])))/pow(2., (double) data->max_lvl);
  // supposed "levels" of the levelset for sampling the values along the normal direction
  // (--> not the actual values of the levelset function at sampled points but more like
  // distances between the sampling points)
  // "signed distances" from the 0-level in the negative normal direction
  const double phi_sampling_levels[2] = {-2.0*diag_min, -3.0*diag_min};

  // prepare objects to sample field values and interface boundary conditions at possibly nonlocal points
  my_p4est_interpolation_nodes_t interp_bc(ngbd_n); // could be _nodes, _cells or _faces, it is irrelevant in this case, we do use base method from my_p4est_interpolation_t anyways
  my_p4est_interpolation_cells_t interp_cell_field(ngbd_c, ngbd_n);
  interp_cell_field.set_input(cell_field, node_sampled_phi, &bc);
  // get point to node-sampled values of the levelset
  const double *node_sampled_phi_p;
  ierr = VecGetArrayRead(node_sampled_phi, &node_sampled_phi_p); CHKERRXX(ierr);
  std::map<p4est_locidx_t, data_for_geometric_extapolation> cell_data_for_extrapolation; cell_data_for_extrapolation.clear();
  /* now buffer the interpolation points */
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t qq = 0; qq < tree->quadrants.elem_count; ++qq)
    {
      const p4est_locidx_t quad_idx = qq + tree->quadrants_offset;
      double phi_q;
      /* check if cell is well defined */
      if(!quadrant_value_is_well_defined(phi_q, bc, p4est, ghost, nodes, quad_idx, tree_idx, node_sampled_phi_p) && phi_q < band_to_extend*diag_min)
      {
        P4EST_ASSERT(phi_q > 0.0); // sanity_check, can't be in negative domain, either way
        double xyz_q[P4EST_DIM];
        quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_q);

        double grad_phi[P4EST_DIM];
        interp_grad_phi(xyz_q, grad_phi);
        add_dof_to_extrapolation_map(cell_data_for_extrapolation, quad_idx, xyz_q, phi_q, grad_phi, nsamples, phi_sampling_levels,
                                     &interp_cell_field, &interp_bc, NULL);
      }
    }
  }

  std::vector<double> cell_field_samples;
  cell_field_samples.resize(nsamples*cell_data_for_extrapolation.size());
  std::vector<bc_sample> calculated_bc_samples; calculated_bc_samples.resize(cell_data_for_extrapolation.size());
  interp_cell_field.interpolate(cell_field_samples.data());
  interp_bc.evaluate_interface_bc(bc, calculated_bc_samples.data());
  /* now compute the extrapolated values */
  double *cell_field_p;
  ierr = VecGetArray(cell_field, &cell_field_p); CHKERRXX(ierr);
  const my_p4est_hierarchy_t* hierarchy = ngbd_c->get_hierarchy();
  std::map<p4est_locidx_t, data_for_geometric_extapolation>::iterator it;
  for (size_t k = 0; k < hierarchy->get_layer_size(); ++k) {
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_layer_quadrant(k);
    it = cell_data_for_extrapolation.find(quad_idx);
    if(it != cell_data_for_extrapolation.end())
      cell_field_p[quad_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples, cell_field_samples, &calculated_bc_samples);
  }
  ierr = VecGhostUpdateBegin(cell_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < hierarchy->get_inner_size(); ++k) {
    const p4est_locidx_t quad_idx = hierarchy->get_local_index_of_inner_quadrant(k);
    it = cell_data_for_extrapolation.find(quad_idx);
    if(it != cell_data_for_extrapolation.end())
      cell_field_p[quad_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples, cell_field_samples, &calculated_bc_samples);
  }
  ierr = VecGhostUpdateEnd(cell_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(cell_field, &cell_field_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(node_sampled_phi, &node_sampled_phi_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_level_set_cells_geometric_extrapolation_over_interface, cell_field, node_sampled_phi, 0, 0); CHKERRXX(ierr);
}

