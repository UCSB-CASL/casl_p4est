#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_multialloy.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#else
#include "my_p4est_poisson_nodes_multialloy.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>

my_p4est_poisson_nodes_multialloy_t::my_p4est_poisson_nodes_multialloy_t(my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    p4est_(node_neighbors->p4est), nodes_(node_neighbors->nodes), ghost_(node_neighbors->ghost), myb_(node_neighbors->myb),
    interp_(node_neighbors),
    bc_tolerance_(1.e-12)
  #ifdef P4_TO_P8
    ,
    phi_zz_(NULL)
  #endif
{
}

my_p4est_poisson_nodes_multialloy_t::~my_p4est_poisson_nodes_multialloy_t()
{
  if (is_phi_dd_owned_)
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      if (phi_dd_.vec[dir] != NULL) { ierr = VecDestroy(phi_dd_.vec[dir]); CHKERRXX(ierr); }

//  if (is_normal_owned_)
//    for (short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      if (normal_[dir].vec != NULL) { ierr = VecDestroy(normal_[dir].vec); CHKERRXX(ierr); }
//      for (short dir2 = 0; dir2 < P4EST_DIM; ++dir2)
//        if (normal_dd_[dir].vec[dir2] != NULL) { ierr = VecDestroy(normal_[dir].vec[dir2]); CHKERRXX(ierr); }
//    }

  if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }
}


void my_p4est_poisson_nodes_multialloy_t::set_phi(Vec phi, Vec* phi_dd, Vec* normal, Vec kappa, Vec theta)
{
  phi_.vec = phi;

  if (phi_dd != NULL)
  {
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      phi_dd_.vec[dir] = phi_dd[dir];

    is_phi_dd_owned_ = false;

  } else {

    for (short dir = 0; dir < P4EST_DIM; ++dir)
    {
      if(phi_dd_.vec[dir] != NULL) { ierr = VecDestroy(phi_dd_.vec[dir]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_dd_.vec[dir]); CHKERRXX(ierr);
    }

    node_neighbors_->second_derivatives_central(phi_.vec, phi_dd_.vec);
    is_phi_dd_owned_ = true;
  }

  if (normal != NULL)
  {
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      normal_.vec[dir] = normal[dir];

    is_normal_owned_ = false;
  } else {
  }

  kappa_.vec = kappa;
  theta_.vec = theta;

//  //compute normals
//  for (int dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    if(normal_[dir].vec != NULL) { ierr = VecDestroy(normal_[dir].vec); CHKERRXX(ierr); }
//    ierr = VecCreateGhostNodes(p4est_, nodes_, &normal_[dir]); CHKERRXX(ierr);
//    normal_[dir].get_array();
//  }

//  phi_.get_array();

//  quad_neighbor_nodes_of_node_t qnnn;

//  for(size_t i = 0; i < node_neighbors_->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_layer_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);
//    normal_[0].ptr[n] = qnnn.dx_central(phi_.ptr);
//    normal_[1].ptr[n] = qnnn.dy_central(phi_.ptr);
//#ifdef P4_TO_P8
//    normal_[2].ptr[n] = qnnn.dz_central(phi_.ptr);
//    double norm = sqrt(SQR(normal_[0].ptr[n]) + SQR(normal_[1].ptr[n]) + SQR(normal_[2].ptr[n]));
//#else
//    double norm = sqrt(SQR(normal_[0].ptr[n]) + SQR(normal_[1].ptr[n]));
//#endif

//    normal_[0].ptr[n] = norm<EPS ? 0 : normal_[0].ptr[n]/norm;
//    normal_[1].ptr[n] = norm<EPS ? 0 : normal_[1].ptr[n]/norm;
//#ifdef P4_TO_P8
//    normal_[2].ptr[n] = norm<EPS ? 0 : normal_[2].ptr[n]/norm;
//#endif
//  }

//  for(int dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    ierr = VecGhostUpdateBegin(normal_[dir].vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  for(size_t i = 0; i < node_neighbors_->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_local_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);
//    normal_[0].ptr[n] = qnnn.dx_central(phi_.ptr);
//    normal_[1].ptr[n] = qnnn.dy_central(phi_.ptr);
//#ifdef P4_TO_P8
//    normal_[2].ptr[n] = qnnn.dz_central(phi_.ptr);
//    double norm = sqrt(SQR(normal_[0].ptr[n]) + SQR(normal_[1].ptr[n]) + SQR(normal_[2].ptr[n]));
//#else
//    double norm = sqrt(SQR(normal_[0].ptr[n]) + SQR(normal_[1].ptr[n]));
//#endif

//    normal_[0].ptr[n] = norm<EPS ? 0 : normal_[0].ptr[n]/norm;
//    normal_[1].ptr[n] = norm<EPS ? 0 : normal_[1].ptr[n]/norm;
//#ifdef P4_TO_P8
//    normal_[2].ptr[n] = norm<EPS ? 0 : normal_[2].ptr[n]/norm;
//#endif
//  }

//  phi_.restore_array();

//  for (int dir = 0; dir < P4EST_DIM; ++dir)
//    normal_[dir].restore_array();

//  for(int dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    ierr = VecGhostUpdateEnd(normal_[dir].vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  is_normal_owned_ = true;

//  for (int dim = 0; dim < P4EST_DIM; ++dim)
//    node_neighbors_->second_derivatives_central(normal_[dim].vec, normal_dd_[dim].vec);


}

void my_p4est_poisson_nodes_multialloy_t::solve(Vec t, Vec t_dd[], Vec c0, Vec c0_dd[], Vec c1, Vec c1_dd[], double& bc_error_max, Vec& bc_error)
{
  t_.vec = t;
  c0_.vec = c0;
  c1_.vec = c1;

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    t_dd_.vec[dim] = t_dd[dim];
    c0_dd_.vec[dim] = c0_dd[dim];
    c1_dd_.vec[dim] = c1_dd[dim];
  }

  bc_error_.vec = bc_error;

  // allocate memory for lagrangian multipliers
  ierr = VecDuplicate(t_.vec, &psi_t_.vec); CHKERRXX(ierr);
  ierr = VecDuplicate(c0_.vec, &psi_c0_.vec); CHKERRXX(ierr);
  ierr = VecDuplicate(c1_.vec, &psi_c1_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecDuplicate(t_dd_.vec[dim], &psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(c0_dd_.vec[dim], &psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(c1_dd_.vec[dim], &psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  initialize_solvers();

  solve_psi_t();

  int iteration = 0;
  bc_error_max_ = 1.;
  while(bc_error_max_ > bc_tolerance_)
  {
    solve_c0();
    compute_c0n();

    solve_t();
//    solve_psi_t();
    solve_c1();
    solve_psi_c1();

    solve_psi_c0();
    compute_psi_c0n();

    adjust_c0_gamma(iteration);

    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g\n", iteration, bc_error_max_); CHKERRXX(ierr);

    ++iteration;
  }


  // clean everything
  delete solver_t;
  delete solver_c0;
  delete solver_c1;
  delete solver_psi_c0;

  ierr = VecDestroy(psi_t_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c0_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c1_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecDestroy(psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  bc_error_max = bc_error_max_;
}




void my_p4est_poisson_nodes_multialloy_t::initialize_solvers()
{
  solver_t      = new my_p4est_poisson_nodes_t(node_neighbors_);
  solver_c0     = new my_p4est_poisson_nodes_t(node_neighbors_);
  solver_c1     = new my_p4est_poisson_nodes_t(node_neighbors_);
  solver_psi_c0 = new my_p4est_poisson_nodes_t(node_neighbors_);

  solver_t->      set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_c0->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_c1->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_psi_c0-> set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);

  // t
  solver_t->set_diagonal(1.0);
  solver_t->set_mu(dt_*t_diff_);
  solver_t->set_use_refined_cube(true);
  is_t_matrix_computed_ = false;

  // c0 and psi_c0
#ifdef P4_TO_P8
  BoundaryConditions3D bc_c0_tmp;
#else
  BoundaryConditions2D bc_c0_tmp;
#endif
  bc_c0_tmp.setWallTypes(bc_t_.getWallType());
  bc_c0_tmp.setWallValues(zero_cf_);
  bc_c0_tmp.setInterfaceType(DIRICHLET);
  bc_c0_tmp.setInterfaceValue(zero_cf_);

  solver_c0->set_diagonal(1.);
  solver_c0->set_mu(dt_*Dl0_);
  solver_c0->set_bc(bc_c0_tmp);
  solver_c0->set_use_pointwise_dirichlet(true);
  solver_c0->assemble_matrix(c0_.vec);

  solver_psi_c0->set_diagonal(1.);
  solver_psi_c0->set_mu(dt_*Dl0_);
  solver_psi_c0->set_bc(bc_c0_tmp);
  solver_psi_c0->set_use_pointwise_dirichlet(true);
  solver_psi_c0->assemble_matrix(psi_c0_.vec);

  // c1
  solver_c1->set_diagonal(1.);
  solver_c1->set_mu(dt_*Dl1_);
  solver_c1->set_use_refined_cube(true);
  is_c1_matrix_computed_ = false;
}




void my_p4est_poisson_nodes_multialloy_t::solve_t()
{
  Vec rhs_tmp;
  ierr = VecDuplicate(psi_t_.vec, &rhs_tmp); CHKERRXX(ierr);

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(NOINTERFACE);
  bc_tmp.setWallTypes(bc_t_.getWallType());
  bc_tmp.setWallValues(bc_t_.getWallValue());

  solver_t->set_bc(bc_tmp);
  solver_t->set_is_matrix_computed(is_t_matrix_computed_);
  solver_t->assemble_jump_rhs(rhs_tmp, bc_t_.getInterfaceValue(), tn_jump_, rhs_tl_.vec, rhs_ts_.vec);
  solver_t->set_rhs(rhs_tmp);
  solver_t->solve(t_.vec);

  is_t_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, t_.vec);

  node_neighbors_->second_derivatives_central(t_.vec, t_dd_.vec);

  ierr = VecDestroy(rhs_tmp); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_t()
{
  Vec rhs_tmp;
  ierr = VecDuplicate(psi_t_.vec, &rhs_tmp); CHKERRXX(ierr);

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(NOINTERFACE);
  bc_tmp.setWallTypes(bc_t_.getWallType());
  bc_tmp.setWallValues(zero_cf_);

  solver_t->set_bc(bc_tmp);
  solver_t->set_is_matrix_computed(is_t_matrix_computed_);
  solver_t->assemble_jump_rhs(rhs_tmp, zero_cf_, jump_psi_tn_, zero_cf_, zero_cf_);
  solver_t->set_rhs(rhs_tmp);
  solver_t->solve(psi_t_.vec);

  is_t_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, psi_t_.vec);

  node_neighbors_->second_derivatives_central(psi_t_.vec, psi_t_dd_.vec);

  ierr = VecDestroy(rhs_tmp); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0()
{
  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c0_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_c0_.get_array();
  rhs_tmp.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
    rhs_tmp.ptr[n] = rhs_c0_.ptr[n];

  rhs_c0_.restore_array();
  rhs_tmp.restore_array();

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(DIRICHLET);
  bc_tmp.setInterfaceValue(zero_cf_);
  bc_tmp.setWallTypes(bc_c0_.getWallType());
  bc_tmp.setWallValues(bc_c0_.getWallValue());

  solver_c0->set_bc(bc_tmp);
  solver_c0->set_is_matrix_computed(true);
  solver_c0->set_rhs(rhs_tmp.vec);
  solver_c0->solve(c0_.vec);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec);

  node_neighbors_->second_derivatives_central(c0_.vec, c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c0()
{
  // compute bondary conditions
  t_.get_array();
  t_dd_.get_array();
  c1_.get_array();
  c1_dd_.get_array();
  psi_c1_.get_array();
  psi_c1_dd_.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
  {
    for (short i = 0; i < solver_psi_c0->pointwise_bc[n].size(); ++i)
    {
      double xyz[P4EST_DIM];
      solver_psi_c0->get_xyz_interface_point(n, i, xyz);
      double c0_gamma = solver_c0->get_interface_point_value(n, i);

      double psi_c0_gamma = -(t_diff_*latent_heat_/t_cond_*solver_psi_c0->interpolate_at_interface_point(n, i, t_.ptr, t_dd_.ptr)
                              + (1.-kp1_)*solver_psi_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr)
                              *solver_psi_c0->interpolate_at_interface_point(n, i, psi_c1_.ptr, psi_c1_dd_.ptr))
          /(1.-kp0_)/c0_gamma;

      solver_psi_c0->set_interface_point_value(n, i, psi_c0_gamma);
    }
  }

  t_.restore_array();
  t_dd_.restore_array();
  c1_.restore_array();
  c1_dd_.restore_array();
  psi_c1_.restore_array();
  psi_c1_dd_.restore_array();

  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c0_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_tmp.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
    rhs_tmp.ptr[n] = 0.;

  rhs_tmp.restore_array();

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(DIRICHLET);
  bc_tmp.setInterfaceValue(zero_cf_);
  bc_tmp.setWallTypes(bc_c0_.getWallType());
  bc_tmp.setWallValues(zero_cf_);

  solver_psi_c0->set_bc(bc_tmp);
  solver_psi_c0->set_is_matrix_computed(true);
  solver_psi_c0->set_rhs(rhs_tmp.vec);
  solver_psi_c0->solve(psi_c0_.vec);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, psi_c0_.vec);

  node_neighbors_->second_derivatives_central(psi_c0_.vec, psi_c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c1()
{
  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c1_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_c1_.get_array();
  rhs_tmp.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
    rhs_tmp.ptr[n] = rhs_c1_.ptr[n];

  rhs_c1_.restore_array();
  rhs_tmp.restore_array();

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(ROBIN);
  bc_tmp.setInterfaceValue(zero_cf_);
  bc_tmp.setRobinCoef(c1_robin_coef_);
  bc_tmp.setWallTypes(bc_c1_.getWallType());
  bc_tmp.setWallValues(bc_c1_.getWallValue());

  solver_c1->set_bc(bc_tmp);
  solver_c1->set_is_matrix_computed(is_c1_matrix_computed_);
  solver_c1->set_rhs(rhs_tmp.vec);
  solver_c1->solve(c1_.vec);

  is_c1_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, c1_.vec);

  node_neighbors_->second_derivatives_central(c1_.vec, c1_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c1()
{
  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c1_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_tmp.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
    rhs_tmp.ptr[n] = 0.;

  rhs_tmp.restore_array();

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(ROBIN);
  bc_tmp.setInterfaceValue(psi_c1_interface_value_);
  bc_tmp.setRobinCoef(c1_robin_coef_);
  bc_tmp.setWallTypes(bc_c1_->getWallType());
  bc_tmp.setWallValues(zero_cf_);

  solver_c1->set_bc(bc_tmp);
  solver_c1->set_is_matrix_computed(is_c1_matrix_computed);
  solver_c1->set_rhs(rhs_tmp);
  solver_c1->solve(psi_c1_.vec);

  is_c1_matrix_computed = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_, psi_c1_.vec);

  node_neighbors_->second_derivatives_central(psi_c1_.vec, psi_c1_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::compute_c0n()
{
  if (c0n_.vec != NULL) { ierr = VecDestroy(c0n_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_, &c0n_); CHKERRXX(ierr);

  c0n_.get_array();
  normal_.get_array();

  quad_neighbor_nodes_of_node_t qnnn;

  for(size_t i = 0; i < node_neighbors_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = node_neighbors_->get_layer_node(i);
    qnnn = node_neighbors_->get_neighbors(n);

#ifdef P4_TO_P8
    c0n_.ptr[n] = qnnn.dx_central(c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(c0_.ptr)*normal_.ptr[1][n] + qnnn.dz_central(c0_.ptr)*normal_.ptr[2][n];
#else
    c0n_.ptr[n] = qnnn.dx_central(c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(c0_.ptr)*normal_.ptr[1][n];
#endif
  }

  ierr = VecGhostUpdateBegin(c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i = 0; i < node_neighbors_->get_local_size(); ++i)
  {
    p4est_locidx_t n = node_neighbors_->get_local_node(i);
    qnnn = node_neighbors_->get_neighbors(n);

#ifdef P4_TO_P8
    c0n_.ptr[n] = qnnn.dx_central(c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(c0_.ptr)*normal_.ptr[1][n] + qnnn.dz_central(c0_.ptr)*normal_.ptr[2][n];
#else
    c0n_.ptr[n] = qnnn.dx_central(c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(c0_.ptr)*normal_.ptr[1][n];
#endif
  }

  ierr = VecGhostUpdateEnd(c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  c0n_.restore_array();
  normal_.restore_array();

  // compute second derivatives for interpolation purposes
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_.vec[dim], &c0n_dd_.vec[dim]); CHKERRXX(ierr);
  }

  node_neighbors_->second_derivatives_central(c0n_.vec, c0n_dd_.vec);
}


void my_p4est_poisson_nodes_multialloy_t::compute_psi_c0n()
{
  if (psi_c0n_.vec != NULL) { ierr = VecDestroy(psi_c0n_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_, &psi_c0n_); CHKERRXX(ierr);

  psi_c0n_.get_array();
  normal_.get_array();

  quad_neighbor_nodes_of_node_t qnnn;

  for(size_t i = 0; i < node_neighbors_->get_layer_size(); ++i)
  {
    p4est_locidx_t n = node_neighbors_->get_layer_node(i);
    qnnn = node_neighbors_->get_neighbors(n);

#ifdef P4_TO_P8
    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*normal_.ptr[1][n] + qnnn.dz_central(psi_c0_.ptr)*normal_.ptr[2][n];
#else
    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*normal_.ptr[1][n];
#endif
  }

  ierr = VecGhostUpdateBegin(c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i = 0; i < node_neighbors_->get_local_size(); ++i)
  {
    p4est_locidx_t n = node_neighbors_->get_local_node(i);
    qnnn = node_neighbors_->get_neighbors(n);

#ifdef P4_TO_P8
    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*normal_.ptr[1][n] + qnnn.dz_central(psi_c0_.ptr)*normal_.ptr[2][n];
#else
    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*normal_.ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*normal_.ptr[1][n];
#endif
  }

  ierr = VecGhostUpdateEnd(psi_c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  psi_c0n_.restore_array();
  normal_.restore_array();

  // compute second derivatives for interpolation purposes
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (psi_c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(psi_c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_.vec[dim], &psi_c0n_dd_.vec[dim]); CHKERRXX(ierr);
  }

  node_neighbors_->second_derivatives_central(psi_c0n_.vec, psi_c0n_dd_.vec);
}


//void my_p4est_poisson_nodes_multialloy_t::compute_dc0()
//{
//  Vec dc0_tmp[P4EST_DIM] = {dc0_[0].vec, dc0_[1].vec};

//  // compute first order derivatives
//  node_neighbors_->first_derivatives_central(c0_.vec, dc0_tmp);

//  // compute second order derivatives of first order derivatives
//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//    node_neighbors_->second_derivatives_central(dc0_[dim].vec, dc0_dd_[dim].vec);
//}

//void my_p4est_poisson_nodes_multialloy_t::compute_psi_dc0()
//{
//  Vec psi_dc0_tmp[P4EST_DIM] = {psi_dc0_[0].vec, psi_dc0_[1].vec};

//  // compute first order derivatives
//  node_neighbors_->first_derivatives_central(psi_c0_.vec, psi_dc0_tmp);

//  // compute second order derivatives of first order derivatives
//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//    node_neighbors_->second_derivatives_central(psi_dc0_[dim].vec, psi_dc0_dd_[dim].vec);
//}

void my_p4est_poisson_nodes_multialloy_t::adjust_c0_gamma(int iteration)
{
  /* get pointers */

  t_.get_array();
  t_dd_.get_array();

  c1_.get_array();
  c1_dd_.get_array();

  normal_.get_array();

  c0n_.get_array();
  c0n_dd_.get_array();

  psi_c0n_.get_array();
  psi_c0n_dd_.get_array();

  bc_error_.get_array();

  theta_.get_array();
  kappa_.get_array();

  /* main loop */

  bc_error_ = 0;
  double xyz[P4EST_DIM];

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
  {
    bc_error_.ptr[n] = 0;
    if (solver_c0->pointwise_bc[n].size())
    {
      for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
      {
        solver_c0->get_xyz_interface_point(n, i, xyz);

        double c0_gamma     = solver_c0->get_interface_point_value(n, i);
        double psi_c0_gamma = solver_psi_c0->get_interface_point_value(n, i);

        // interpolate concentration
        double conc_term = ml0_*c0_gamma + ml1_*solver_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr);

        // interpolate temperature
        double ther_term = solver_c0->interpolate_at_interface_point(n, i, t_.ptr, t_dd_.ptr);

        // interpolate velocity
        double c0n      = solver_c0->interpolate_at_interface_point(n, i, c0n_.ptr,     c0n_dd_.ptr);
        double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr, psi_c0n_dd_.ptr);

        double theta = solver_c0->interpolate_at_interface_point(n, i, theta_.ptr);
        double kappa = solver_c0->interpolate_at_interface_point(n, i, kappa_.ptr);

        double error = (conc_term - ther_term - (*eps_v_)(theta)*( Dl0_/(1.-kp0_)*c0n/c0_gamma ) - (*eps_c_)(theta)*kappa + (*GT_)(xyz));

        bc_error_.ptr[n] = MAX(error, bc_error_.ptr[n]);
        bc_error_max_ = MAX(bc_error_, error);

        double change = error/ml0_;

        if (iteration%pin_every_n_steps_ != 0)
          change /= (1. + Dl0_*psi_c0n - Dl0_*c0n*psi_c0_gamma/c0_gamma);

        solver_c0->set_interface_point_value(n, i, c0_gamma-change);
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &bc_error_, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);



  /* restore pointers */

  t_.restore_array();
  t_dd_.restore_array();

  c1_.restore_array();
  c1_dd_.restore_array();

  normal_.get_array();

  c0n_.get_array();
  c0n_dd_.get_array();

  psi_c0n_.get_array();
  psi_c0n_dd_.get_array();

  bc_error_.restore_array();

  theta_.restore_array();
  kappa_.restore_array();
}
