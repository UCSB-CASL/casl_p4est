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
    bc_tolerance_(1.e-12),
    max_iterations_(10),
    is_phi_dd_owned_(false), is_normal_owned_(false)
  #ifdef P4_TO_P8
    ,
    phi_zz_(NULL)
  #endif
{
  jump_psi_tn_.set_ptr(this);
  psi_c1_interface_value_.set_ptr(this);
  vn_from_c0_.set_ptr(this);
  c0_robin_coef_.set_ptr(this);
  c1_robin_coef_.set_ptr(this);
  tn_jump_.set_ptr(this);

  GT_ = &zero_cf_;
  eps_c_ = &zero_cf1_;
  eps_v_ = &zero_cf1_;

  dt_ = 1.;
  t_diff_ = 1.;
  t_cond_ = 1.;
  latent_heat_ = 1.;
  Tm_ = 1.;
  Dl0_ = 1.; kp0_ = 1.; ml0_ = 1.;
  Dl1_ = 1.; kp1_ = 1.; ml1_ = 1.;
  bc_error_max_ = 1.;
  pin_every_n_steps_ = 3;

  solver_t  = NULL;
  solver_c0 = NULL;
  solver_c1 = NULL;
  solver_psi_c0 = NULL;

  is_t_matrix_computed_  = false;
  is_c1_matrix_computed_ = false;

  use_refined_cube_ = true;
  second_derivatives_owned_ = false;
}

my_p4est_poisson_nodes_multialloy_t::~my_p4est_poisson_nodes_multialloy_t()
{
  if (is_phi_dd_owned_)
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      if (phi_dd_.vec[dir] != NULL) { ierr = VecDestroy(phi_dd_.vec[dir]); CHKERRXX(ierr); }


  if (c0n_.vec != NULL) { ierr = VecDestroy(c0n_.vec); CHKERRXX(ierr); }
  if (psi_c0n_.vec != NULL) { ierr = VecDestroy(psi_c0n_.vec); CHKERRXX(ierr); }

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    if (psi_c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(psi_c0n_dd_.vec[dim]); CHKERRXX(ierr); }
  }

  if (solver_t != NULL) delete solver_t;
  if (solver_c0 != NULL) delete solver_c0;
  if (solver_c1 != NULL) delete solver_c1;
  if (solver_psi_c0 != NULL) delete solver_psi_c0;

//  if (is_normal_owned_)
//    for (short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      if (normal_[dir].vec != NULL) { ierr = VecDestroy(normal_[dir].vec); CHKERRXX(ierr); }
//      for (short dir2 = 0; dir2 < P4EST_DIM; ++dir2)
//        if (normal_dd_[dir].vec[dir2] != NULL) { ierr = VecDestroy(normal_[dir].vec[dir2]); CHKERRXX(ierr); }
//    }

  if (second_derivatives_owned_)
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      if (t_dd_.vec[dim]  != NULL) { ierr = VecDestroy(t_dd_.vec[dim]); CHKERRXX(ierr); }
      if (c0_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0_dd_.vec[dim]); CHKERRXX(ierr); }
      if (c1_dd_.vec[dim] != NULL) { ierr = VecDestroy(c1_dd_.vec[dim]); CHKERRXX(ierr); }
    }
  }
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

//void my_p4est_poisson_nodes_multialloy_t::solve(Vec t, Vec t_dd[], Vec c0, Vec c0_dd[], Vec c1, Vec c1_dd[], double& bc_error_max, Vec& bc_error)
//{
//  t_.vec = t;
//  c0_.vec = c0;
//  c1_.vec = c1;

//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    t_dd_.vec[dim] = t_dd[dim];
//    c0_dd_.vec[dim] = c0_dd[dim];
//    c1_dd_.vec[dim] = c1_dd[dim];
//  }

//  bc_error_.vec = bc_error;

//  // allocate memory for lagrangian multipliers
//  ierr = VecDuplicate(t_.vec, &psi_t_.vec); CHKERRXX(ierr);
//  ierr = VecDuplicate(c0_.vec, &psi_c0_.vec); CHKERRXX(ierr);
//  ierr = VecDuplicate(c1_.vec, &psi_c1_.vec); CHKERRXX(ierr);

//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    ierr = VecDuplicate(t_dd_.vec[dim], &psi_t_dd_.vec[dim]); CHKERRXX(ierr);
//    ierr = VecDuplicate(c0_dd_.vec[dim], &psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
//    ierr = VecDuplicate(c1_dd_.vec[dim], &psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
//  }

//  initialize_solvers();

//  solve_psi_t();

//  int iteration = 0;
//  bc_error_max_ = 1.;
//  while(bc_error_max_ > bc_tolerance_ && iteration < max_iterations_)
//  {
//    ++iteration;

//    solve_c0();
//    compute_c0n();
//    is_c1_matrix_computed_ = false;

//    solve_t();
//    solve_c1();

//    if (iteration%pin_every_n_steps_ != 0)
//    {
//      solve_psi_c1();
//      solve_psi_c0();
//      compute_psi_c0n();
//    }

//    adjust_c0_gamma(iteration);
//    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g\n", iteration, bc_error_max_); CHKERRXX(ierr);

////    std::cout << "Iteration " << iteration << " Rank " << p4est_->mpirank << std::endl;
//  }


//  // clean everything
//  ierr = VecDestroy(psi_t_.vec); CHKERRXX(ierr);
//  ierr = VecDestroy(psi_c0_.vec); CHKERRXX(ierr);
//  ierr = VecDestroy(psi_c1_.vec); CHKERRXX(ierr);

//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    ierr = VecDestroy(psi_t_dd_.vec[dim]); CHKERRXX(ierr);
//    ierr = VecDestroy(psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
//    ierr = VecDestroy(psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
//  }

//  bc_error_max = bc_error_max_;

////  solve_c1();
////  solve_c0_robin();
//}


void my_p4est_poisson_nodes_multialloy_t::solve(Vec t, Vec c0, Vec c1, Vec bc_error, double &bc_error_max)
{
  t_.vec = t;
  c0_.vec = c0;
  c1_.vec = c1;

  second_derivatives_owned_ = true;

  if (tm_.vec != NULL) { ierr = VecDestroy(tm_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(t_.vec, &tm_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (t_dd_.vec[dim]  != NULL) { ierr = VecDestroy(t_dd_.vec[dim]); CHKERRXX(ierr); }
    if (tm_dd_.vec[dim] != NULL) { ierr = VecDestroy(tm_dd_.vec[dim]); CHKERRXX(ierr); }
    if (c0_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0_dd_.vec[dim]); CHKERRXX(ierr); }
    if (c1_dd_.vec[dim] != NULL) { ierr = VecDestroy(c1_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &tm_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &c1_dd_.vec[dim]); CHKERRXX(ierr);
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
  while(bc_error_max_ > bc_tolerance_ && iteration < max_iterations_)
  {
    ++iteration;

    solve_c0();
    compute_c0n();
    is_c1_matrix_computed_ = false;

    solve_t();
    solve_c1();

    if (iteration%pin_every_n_steps_ != 0)
    {
      solve_psi_c1();
      solve_psi_c0();
      compute_psi_c0n();
    }

    adjust_c0_gamma(iteration);
    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g\n", iteration, bc_error_max_); CHKERRXX(ierr);

//    std::cout << "Iteration " << iteration << " Rank " << p4est_->mpirank << std::endl;
  }


  // clean everything
  ierr = VecDestroy(tm_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_t_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c0_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c1_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecDestroy(tm_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  bc_error_max = bc_error_max_;

//  solve_c1();
//  solve_c0_robin();
}



void my_p4est_poisson_nodes_multialloy_t::solve(Vec t, Vec c0, Vec c1, Vec bc_error, double &bc_error_max, Vec c0n)
{
  t_.vec = t;
  c0_.vec = c0;
  c1_.vec = c1;

  second_derivatives_owned_ = true;

  if (tm_.vec != NULL) { ierr = VecDestroy(tm_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(t_.vec, &tm_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (t_dd_.vec[dim]  != NULL) { ierr = VecDestroy(t_dd_.vec[dim]); CHKERRXX(ierr); }
    if (tm_dd_.vec[dim] != NULL) { ierr = VecDestroy(tm_dd_.vec[dim]); CHKERRXX(ierr); }
    if (c0_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0_dd_.vec[dim]); CHKERRXX(ierr); }
    if (c1_dd_.vec[dim] != NULL) { ierr = VecDestroy(c1_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_, nodes_, &t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &tm_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(t_dd_.vec[dim], &c1_dd_.vec[dim]); CHKERRXX(ierr);
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

  // initialize c0
  c0n_.vec = c0n;
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_.vec[dim], &c0n_dd_.vec[dim]); CHKERRXX(ierr);
  }
  node_neighbors_->second_derivatives_central(c0n_.vec, c0n_dd_.vec);
  solve_t();
  solve_c1();
  adjust_c0_gamma(0);
  ierr = PetscPrintf(p4est_->mpicomm, "Iteration 0: bc error = %g\n", bc_error_max_); CHKERRXX(ierr);

  int iteration = 0;
  bc_error_max_ = 1.;
  while(bc_error_max_ > bc_tolerance_ && iteration < max_iterations_)
  {
    ++iteration;

    solve_c0();
    compute_c0n();
    is_c1_matrix_computed_ = false;

    solve_t();
    solve_c1();

    if (iteration%pin_every_n_steps_ != 0)
    {
      solve_psi_c1();
      solve_psi_c0();
      compute_psi_c0n();
    }

    adjust_c0_gamma(iteration);
    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g\n", iteration, bc_error_max_); CHKERRXX(ierr);

//    std::cout << "Iteration " << iteration << " Rank " << p4est_->mpirank << std::endl;
  }


  // clean everything
  ierr = VecDestroy(tm_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_t_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c0_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c1_.vec); CHKERRXX(ierr);

  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecDestroy(tm_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  bc_error_max = bc_error_max_;

//  solve_c1();
//  solve_c0_robin();
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
  solver_t->set_use_refined_cube(use_refined_cube_);
  is_t_matrix_computed_ = false;

  // c0 and psi_c0
#ifdef P4_TO_P8
  BoundaryConditions3D bc_c0_tmp;
#else
  BoundaryConditions2D bc_c0_tmp;
#endif
  bc_c0_tmp.setWallTypes(bc_c0_.getWallType());
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
  solver_c1->set_use_refined_cube(use_refined_cube_);
  is_c1_matrix_computed_ = false;


  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
  {
    for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
    {
      double xyz[P4EST_DIM];
      solver_c0->get_xyz_interface_point(n, i, xyz);
      double c0_gamma = (*c0_guess_)(xyz[0],xyz[1]);

      solver_c0->set_interface_point_value(n, i, c0_gamma);
    }
  }
}




void my_p4est_poisson_nodes_multialloy_t::solve_t()
{
  Vec rhs_tmp;
  ierr = VecDuplicate(t_.vec, &rhs_tmp); CHKERRXX(ierr);

  BoundaryConditions2D bc_tmp;

  bc_tmp.setInterfaceType(NOINTERFACE);
  bc_tmp.setWallTypes(bc_t_.getWallType());
  bc_tmp.setWallValues(bc_t_.getWallValue());

  solver_t->set_bc(bc_tmp);
  solver_t->set_is_matrix_computed(is_t_matrix_computed_);
  solver_t->assemble_jump_rhs(rhs_tmp, *jump_t_, tn_jump_, rhs_tl_.vec, rhs_ts_.vec);
  solver_t->set_rhs(rhs_tmp);
  solver_t->solve(t_.vec);

  t_.get_array();
  tm_.get_array();

  for (size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
    tm_.ptr[n] = t_.ptr[n];

  t_.restore_array();
  tm_.restore_array();

  is_t_matrix_computed_ = true;

//  my_p4est_level_set_t ls(node_neighbors_);
//  ls.extend_Over_Interface_TVD(phi_.vec, t_.vec, 20, 2, normal_.vec);

//  node_neighbors_->second_derivatives_central(t_.vec, t_dd_.vec);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, tm_.vec, 20, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(tm_.vec, tm_dd_.vec);

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
  solver_t->assemble_jump_rhs(rhs_tmp, zero_cf_, jump_psi_tn_);
  solver_t->set_rhs(rhs_tmp);
  solver_t->solve(psi_t_.vec);

  is_t_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, psi_t_.vec, 20, 2, normal_.vec);

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

//  double *c0_p;
//  ierr = VecGetArray(c0_.vec, &c0_p);

//  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
//  {
//    double XYZ[P4EST_DIM];
//    node_xyz_fr_n(n, p4est_, nodes_, XYZ);
//    if (fabs(XYZ[1]-1.) < EPS)
//      std::cout << c0_p[n] << " " << bc_c0_.wallType(XYZ) << " | ";
//  }
//  std::cout << std::endl;
//  ierr = VecRestoreArray(c0_.vec, &c0_p);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec, 100, 2, normal_.vec);
//  ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec, solver_c0, 100, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(c0_.vec, c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c0()
{
  // compute bondary conditions
  psi_t_.get_array();
  psi_t_dd_.get_array();
  c1_.get_array();
  c1_dd_.get_array();
  psi_c1_.get_array();
  psi_c1_dd_.get_array();
  theta_.get_array();

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
  {
    for (short i = 0; i < solver_psi_c0->pointwise_bc[n].size(); ++i)
    {
      double xyz[P4EST_DIM];
      solver_psi_c0->get_xyz_interface_point(n, i, xyz);
      double c0_gamma = solver_c0->get_interface_point_value(n, i);

//      double psi_c0_gamma = -( (1.-kp1_)*solver_psi_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr)
//                                       *solver_psi_c0->interpolate_at_interface_point(n, i, psi_c1_.ptr, psi_c1_dd_.ptr)
//                                + t_diff_*latent_heat_/t_cond_*solver_psi_c0->interpolate_at_interface_point(n, i, psi_t_.ptr, psi_t_dd_.ptr)
//                               + (*eps_v_)(solver_psi_c0->interpolate_at_interface_point(n,i,theta_.ptr))/ml0_ )
//          /(1.-kp0_)/c0_gamma;

      double psi_c0_gamma = -( (1.-kp1_)*solver_psi_c0->interpolate_at_interface_point(n, i, c1_.ptr)
                                       *solver_psi_c0->interpolate_at_interface_point(n, i, psi_c1_.ptr)
                                + t_diff_*latent_heat_/t_cond_*solver_psi_c0->interpolate_at_interface_point(n, i, psi_t_.ptr)
                               + (*eps_v_)(solver_psi_c0->interpolate_at_interface_point(n,i,theta_.ptr))/ml0_ )
          /(1.-kp0_)/c0_gamma;

      solver_psi_c0->set_interface_point_value(n, i, psi_c0_gamma);
    }
  }

  psi_t_.restore_array();
  psi_t_dd_.restore_array();
  c1_.restore_array();
  c1_dd_.restore_array();
  psi_c1_.restore_array();
  psi_c1_dd_.restore_array();
  theta_.restore_array();

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
  ls.extend_Over_Interface_TVD(phi_.vec, psi_c0_.vec, 50, 2, normal_.vec);
//  ls.extend_Over_Interface_TVD(phi_.vec, psi_c0_.vec, solver_psi_c0, 100, 2, normal_.vec);

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
  bc_tmp.setInterfaceValue(*c1_flux_);
  bc_tmp.setRobinCoef(c1_robin_coef_);
  bc_tmp.setWallTypes(bc_c1_.getWallType());
  bc_tmp.setWallValues(bc_c1_.getWallValue());

  solver_c1->set_bc(bc_tmp);
  solver_c1->set_is_matrix_computed(is_c1_matrix_computed_);
  solver_c1->set_rhs(rhs_tmp.vec);
  solver_c1->solve(c1_.vec);

  is_c1_matrix_computed_ = true;

  Vec mask = solver_c1->get_mask();

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, mask, c1_.vec, 20, 2);
//  ls.extend_Over_Interface_TVD(phi_.vec, c1_.vec, 20, 2, normal_.vec);

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
  bc_tmp.setWallTypes(bc_c1_.getWallType());
  bc_tmp.setWallValues(zero_cf_);

  solver_c1->set_bc(bc_tmp);
  solver_c1->set_is_matrix_computed(is_c1_matrix_computed_);
  solver_c1->set_rhs(rhs_tmp.vec);
  solver_c1->solve(psi_c1_.vec);

  is_c1_matrix_computed_ = true;

  Vec mask = solver_c1->get_mask();

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, mask, psi_c1_.vec, 20, 2);
//  ls.extend_Over_Interface_TVD(phi_.vec, psi_c1_.vec, 20, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(psi_c1_.vec, psi_c1_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0_robin()
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

  bc_tmp.setInterfaceType(ROBIN);
  bc_tmp.setInterfaceValue(*c0_flux_);
  bc_tmp.setRobinCoef(c0_robin_coef_);
  bc_tmp.setWallTypes(bc_c0_.getWallType());
  bc_tmp.setWallValues(bc_c0_.getWallValue());


  // solve for c0 using Robin BC
  my_p4est_poisson_nodes_t solver_c0_robin(node_neighbors_);
  solver_c0_robin.set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_c0_robin.set_diagonal(1.);
  solver_c0_robin.set_mu(dt_*Dl0_);
  solver_c0_robin.set_use_refined_cube(use_refined_cube_);
  solver_c0_robin.set_bc(bc_tmp);
  solver_c0_robin.set_rhs(rhs_tmp.vec);
  solver_c0_robin.solve(c0_.vec);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec, 100, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(c0_.vec, c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::compute_c0n()
{
  if (c0n_.vec != NULL) { ierr = VecDestroy(c0n_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_.vec, &c0n_.vec); CHKERRXX(ierr);

  c0_.get_array();
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

  c0_.restore_array();
  c0n_.restore_array();
  normal_.restore_array();

//  my_p4est_level_set_t ls(node_neighbors_);
//  ls.extend_Over_Interface_TVD(phi_.vec, c0n_.vec, 20, 2, normal_.vec);

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
  ierr = VecDuplicate(phi_.vec, &psi_c0n_.vec); CHKERRXX(ierr);

  psi_c0_.get_array();
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

  ierr = VecGhostUpdateBegin(psi_c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

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

  psi_c0_.restore_array();
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

  tm_.get_array();
  tm_dd_.get_array();

  c1_.get_array();
  c1_dd_.get_array();

  normal_.get_array();

  c0n_.get_array();
  c0n_dd_.get_array();

  if (iteration%pin_every_n_steps_ != 0)
  {
    psi_c0n_.get_array();
    psi_c0n_dd_.get_array();
  }

  bc_error_.get_array();

  theta_.get_array();
  kappa_.get_array();

  /* main loop */

  bc_error_max_ = 0;
  double xyz[P4EST_DIM];

  for (p4est_locidx_t n = 0; n < nodes_->num_owned_indeps; ++n)
  {
    bc_error_.ptr[n] = 0;
    if (solver_c0->pointwise_bc[n].size())
    {
      for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
      {
        solver_c0->get_xyz_interface_point(n, i, xyz);

        double c0_gamma = solver_c0->get_interface_point_value(n, i);

        // interpolate concentration
        double conc_term = ml0_*c0_gamma + ml1_*solver_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr);
//        double conc_term = ml0_*c0_gamma + ml1_*solver_c0->interpolate_at_interface_point(n, i, c1_.ptr);

        // interpolate temperature
        double ther_term = solver_c0->interpolate_at_interface_point(n, i, tm_.ptr, tm_dd_.ptr);
//        double ther_term = solver_c0->interpolate_at_interface_point(n, i, tm_.ptr);

        // interpolate velocity
        double c0n = solver_c0->interpolate_at_interface_point(n, i, c0n_.ptr, c0n_dd_.ptr);
//        double c0n = solver_c0->interpolate_at_interface_point(n, i, c0n_.ptr);

        double theta = solver_c0->interpolate_at_interface_point(n, i, theta_.ptr);
        double kappa = solver_c0->interpolate_at_interface_point(n, i, kappa_.ptr);

        double error = (conc_term + Tm_ - ther_term - (*eps_v_)(theta)*( Dl0_/(1.-kp0_)*(c0n-c0_flux_->value(xyz))/c0_gamma ) + (*eps_c_)(theta)*kappa + (*GT_)(xyz[0], xyz[1]));

        bc_error_.ptr[n] = MAX(bc_error_.ptr[n], fabs(error));
        bc_error_max_ = MAX(bc_error_max_, fabs(error));

        double change = error/ml0_;

        if (iteration%pin_every_n_steps_ != 0)
        {
          double psi_c0_gamma = solver_psi_c0->get_interface_point_value(n, i);
          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr, psi_c0n_dd_.ptr);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
//          double vn = vn_from_c0_.value(xyz);
//          double dem = (1. + Dl0_*psi_c0n - (1.-kp0_)*vn*psi_c0_gamma);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
//          change /= (1. + Dl0_*psi_c0n - (1.-kp0_)*vn*psi_c0_gamma);
          change /= (1. + Dl0_*psi_c0n - Dl0_*(c0n-c0_flux_->value(xyz))*psi_c0_gamma/c0_gamma);
        }

        solver_c0->set_interface_point_value(n, i, c0_gamma-change);
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &bc_error_max_, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);



  /* restore pointers */

  tm_.restore_array();
  tm_dd_.restore_array();

  c1_.restore_array();
  c1_dd_.restore_array();

  normal_.get_array();

  c0n_.get_array();
  c0n_dd_.get_array();

  if (iteration%pin_every_n_steps_ != 0)
  {
    psi_c0n_.get_array();
    psi_c0n_dd_.get_array();
  }

  bc_error_.restore_array();

  theta_.restore_array();
  kappa_.restore_array();
}
