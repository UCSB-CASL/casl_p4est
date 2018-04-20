#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_multialloy.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_poisson_nodes_multialloy.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_macros.h>
#endif
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_initialize_solvers;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_c0;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_c1;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_t;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_c0;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_c1;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_solve_psi_t;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_compute_c0n;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n;
extern PetscLogEvent log_my_p4est_poisson_nodes_multialloy_adjust_c0;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
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
{
  jump_psi_tn_.set_ptr(this);
  psi_c1_interface_value_.set_ptr(this);
  vn_from_c0_.set_ptr(this);
  c0_robin_coef_.set_ptr(this);
  c1_robin_coef_.set_ptr(this);
  tn_jump_.set_ptr(this);

  GT_ = &zero_cf_;
  eps_c_ = &zero_cf_;
  eps_v_ = &zero_cf_;

  dt_ = 1.;
  t_diff_ = 1.;
  t_cond_ = 1.;
  latent_heat_ = 1.;
  Tm_ = 1.;
  Dl0_ = 1.; kp0_ = 1.; ml0_ = 1.;
  Dl1_ = 1.; kp1_ = 1.; ml1_ = 1.;
  bc_error_max_ = 1.;
  pin_every_n_steps_ = 3;

  solver_t      = NULL;
  solver_c0     = NULL;
  solver_c1     = NULL;
  solver_psi_c0 = NULL;

  solver_c1_sc  = NULL;

  is_t_matrix_computed_  = false;
  is_c1_matrix_computed_ = false;

  use_refined_cube_ = 1;
  second_derivatives_owned_ = false;

  use_continuous_stencil_    = false;
  use_one_sided_derivatives_ = false;
  use_superconvergent_robin_ = false;
  use_superconvergent_jump_  = false;
  update_c0_robin_           = false;
  use_points_on_interface_   = true;
  zero_negative_velocity_    = false;

  volume_thresh_ = 1.e-2;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est_, dxyz);

#ifdef P4_TO_P8
  min_volume_ = dxyz[0]*dxyz[1]*dxyz[2];
#else
  min_volume_ = dxyz[0]*dxyz[1];
#endif
}

my_p4est_poisson_nodes_multialloy_t::~my_p4est_poisson_nodes_multialloy_t()
{
  if (is_phi_dd_owned_)
    foreach_dimension(dim)
      if (phi_dd_.vec[dim] != NULL) { ierr = VecDestroy(phi_dd_.vec[dim]); CHKERRXX(ierr); }


  if (c0n_.vec != NULL) { ierr = VecDestroy(c0n_.vec); CHKERRXX(ierr); }
  if (psi_c0n_.vec != NULL) { ierr = VecDestroy(psi_c0n_.vec); CHKERRXX(ierr); }

  foreach_dimension(dim)
  {
    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    if (psi_c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(psi_c0n_dd_.vec[dim]); CHKERRXX(ierr); }
  }

  if (solver_t      != NULL) delete solver_t;
  if (solver_c0     != NULL) delete solver_c0;
  if (solver_c1     != NULL) delete solver_c1;
  if (solver_psi_c0 != NULL) delete solver_psi_c0;
  if (solver_c1_sc  != NULL) delete solver_c1_sc;

  if (c0_gamma_.vec != NULL) { ierr = VecDestroy(c0_gamma_.vec); CHKERRXX(ierr); }
  if (c0n_gamma_.vec != NULL) { ierr = VecDestroy(c0n_gamma_.vec); CHKERRXX(ierr); }

  if (volumes_.vec != NULL) { ierr = VecDestroy(volumes_.vec); CHKERRXX(ierr); }

//  if (is_normal_owned_)
//    for (short dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      if (normal_[dir].vec != NULL) { ierr = VecDestroy(normal_[dir].vec); CHKERRXX(ierr); }
//      for (short dir2 = 0; dir2 < P4EST_DIM; ++dir2)
//        if (normal_dd_[dir].vec[dir2] != NULL) { ierr = VecDestroy(normal_[dir].vec[dir2]); CHKERRXX(ierr); }
//    }

  if (second_derivatives_owned_)
  {
    foreach_dimension(dim)
    {
      if (t_dd_.vec[dim]  != NULL) { ierr = VecDestroy(t_dd_.vec[dim]); CHKERRXX(ierr); }
      if (c0_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0_dd_.vec[dim]); CHKERRXX(ierr); }
      if (c1_dd_.vec[dim] != NULL) { ierr = VecDestroy(c1_dd_.vec[dim]); CHKERRXX(ierr); }
    }
  }
}

void my_p4est_poisson_nodes_multialloy_t::set_phi(Vec phi, Vec* phi_dd, Vec* normal, Vec kappa)
{
  phi_.vec = phi;

  if (phi_dd != NULL)
  {
    foreach_dimension(dim) phi_dd_.vec[dim] = phi_dd[dim];

    is_phi_dd_owned_ = false;

  } else {

    foreach_dimension(dim)
    {
      if(phi_dd_.vec[dim] != NULL) { ierr = VecDestroy(phi_dd_.vec[dim]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_dd_.vec[dim]); CHKERRXX(ierr);
    }

    node_neighbors_->second_derivatives_central(phi_.vec, phi_dd_.vec);
    is_phi_dd_owned_ = true;
  }

  if (normal != NULL)
  {
    foreach_dimension(dim) normal_.vec[dim] = normal[dim];

    is_normal_owned_ = false;
  } else {
  }

  kappa_.vec = kappa;
//  theta_xz_.vec = theta_xz;
//#ifdef P4_TO_P8
//  theta_yz_.vec = theta_yz;
//#endif
}

int my_p4est_poisson_nodes_multialloy_t::solve(Vec t, Vec c0, Vec c1, Vec bc_error, double &bc_error_max, double &dt, double cfl, bool use_non_zero_guess)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  use_non_zero_guess_ = use_non_zero_guess;

  t_.vec = t;
  c0_.vec = c0;
  c1_.vec = c1;

  second_derivatives_owned_ = true;

  if (tm_.vec != NULL) { ierr = VecDestroy(tm_.vec); CHKERRXX(ierr); }
  ierr = VecDuplicate(t_.vec, &tm_.vec); CHKERRXX(ierr);

  foreach_dimension(dim)
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

  foreach_dimension(dim)
  {
    ierr = VecDuplicate(t_dd_.vec[dim], &psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(c0_dd_.vec[dim], &psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(c1_dd_.vec[dim], &psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  double dxyz[P4EST_DIM];

  dxyz_min(p4est_, dxyz);

#ifdef P4_TO_P8
  double d_min = MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
  double d_min = MIN(dxyz[0], dxyz[1]);
#endif

  dt_ = dt;
  initialize_solvers();

  int iteration = 0;
  bc_error_max_ = 1.;

  solve_psi_t();

  bool need_one = true;
  while(bc_error_max_ > bc_tolerance_ && iteration < max_iterations_ || need_one)
  {
    ++iteration;

    // solve for physical quantities
    solve_c0();
    compute_c0n();
    is_c1_matrix_computed_ = false;

    solve_t();
    solve_c1();

    // solve for lagrangian multipliers
    if (iteration%pin_every_n_steps_ != 0)
    {
      solve_psi_c1();
      solve_psi_c0();
      compute_psi_c0n();
    }

    // adjust boundary conditions
    adjust_c0_gamma(iteration);

    need_one = false;
    // check if max velocity violates CFL condition
    if (iteration > 3 && dt_ > 1.5*cfl*d_min/velo_max_)
    {
      need_one = true;

      // adjust time-step
      dt_ = cfl*d_min/velo_max_;

      ierr = PetscPrintf(p4est_->mpicomm, "Adjusting time-step. New dt = %g\n", dt_); CHKERRXX(ierr);

      // re-initialize solvers
      std::vector< std::vector<my_p4est_poisson_nodes_t::interface_point_t> > pointwise_bc_tmp;
      pointwise_bc_tmp = solver_c0->pointwise_bc;

      initialize_solvers();

      solver_c0->pointwise_bc = pointwise_bc_tmp;

      solve_psi_t();
    }

    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g, time step = %g, max velo = %g\n", iteration, bc_error_max_, dt_, velo_max_); CHKERRXX(ierr);
  }

  solve_c0();
  compute_c0n();
  is_c1_matrix_computed_ = false;

  if (zero_negative_velocity_)
  {
    c0n_gamma_.get_array();
    foreach_node(n, nodes_) { if (c0n_gamma_.ptr[n] < 0) c0n_gamma_.ptr[n] = 0; }
    c0n_gamma_.restore_array();
  }

  solve_t();
  solve_c1();

//  if (update_c0_robin_) solve_c0_robin();

  // return time-step back
  dt = dt_;

  // clean everything
  ierr = VecDestroy(tm_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_t_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c0_.vec); CHKERRXX(ierr);
  ierr = VecDestroy(psi_c1_.vec); CHKERRXX(ierr);

  foreach_dimension(dim)
  {
    ierr = VecDestroy(tm_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_t_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c0_dd_.vec[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(psi_c1_dd_.vec[dim]); CHKERRXX(ierr);
  }

  bc_error_max = bc_error_max_;

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  return iteration;
}



void my_p4est_poisson_nodes_multialloy_t::initialize_solvers()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);

  phi_vector_.clear();    phi_vector_.push_back(phi_.vec);
  phi_xx_vector_.clear(); phi_xx_vector_.push_back(phi_dd_.vec[0]);
  phi_yy_vector_.clear(); phi_yy_vector_.push_back(phi_dd_.vec[1]);
#ifdef P4_TO_P8
  phi_zz_vector_.clear(); phi_zz_vector_.push_back(phi_dd_.vec[2]);
#endif
  action_.clear(); action_.push_back(INTERSECTION);
  color_.clear();  color_ .push_back(0);

  if (solver_t      != NULL) { delete solver_t      ; }
  if (solver_c0     != NULL) { delete solver_c0     ; }
  if (solver_c1     != NULL) { delete solver_c1     ; }
  if (solver_c1_sc  != NULL) { delete solver_c1_sc  ; }
  if (solver_psi_c0 != NULL) { delete solver_psi_c0 ; }

  solver_t      = new my_p4est_poisson_nodes_t(node_neighbors_);
  solver_c0     = new my_p4est_poisson_nodes_t(node_neighbors_);
  solver_psi_c0 = new my_p4est_poisson_nodes_t(node_neighbors_);

  if (use_superconvergent_robin_) solver_c1_sc = new my_p4est_poisson_nodes_mls_sc_t(node_neighbors_);
  else                            solver_c1    = new my_p4est_poisson_nodes_t(node_neighbors_);

#ifdef P4_TO_P8
  solver_t->      set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1], phi_dd_.vec[2]);
  solver_c0->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1], phi_dd_.vec[2]);
  solver_psi_c0-> set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1], phi_dd_.vec[2]);
  if (use_superconvergent_robin_) solver_c1_sc->  set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_, &phi_zz_vector_);
  else                            solver_c1->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1], phi_dd_.vec[2]);
#else
  solver_t->      set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_c0->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  solver_psi_c0-> set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
  if (use_superconvergent_robin_) solver_c1_sc->  set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_);
  else                            solver_c1->     set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
#endif

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
  solver_c0->set_use_continuous_stencil(use_continuous_stencil_);
  solver_c0->set_use_pointwise_dirichlet(true);
  solver_c0->assemble_matrix(c0_.vec);

  solver_psi_c0->set_diagonal(1.);
  solver_psi_c0->set_mu(dt_*Dl0_);
  solver_psi_c0->set_bc(bc_c0_tmp);
  solver_psi_c0->set_use_continuous_stencil(use_continuous_stencil_);
  solver_psi_c0->set_use_pointwise_dirichlet(true);
  solver_psi_c0->assemble_matrix(psi_c0_.vec);

  // c1
  if (use_superconvergent_robin_)
  {
    solver_c1_sc->set_diag_add(1.);
    solver_c1_sc->set_mu(Dl1_*dt_);
    solver_c1_sc->set_use_sc_scheme(false);
    solver_c1_sc->set_integration_order(2);

    solver_c1_sc->set_use_taylor_correction(1);
    solver_c1_sc->set_keep_scalling(true);
    solver_c1_sc->set_kink_treatment(1);
    solver_c1_sc->set_try_remove_hanging_cells(0);
  } else {
    solver_c1->set_diagonal(1.);
    solver_c1->set_mu(dt_*Dl1_);
    solver_c1->set_use_refined_cube(use_refined_cube_);
  }

  is_c1_matrix_computed_ = false;

  foreach_local_node(n, nodes_)
  {
    for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
    {
      double xyz[P4EST_DIM];
      solver_c0->get_xyz_interface_point(n, i, xyz);
      double c0_gamma = c0_guess_->value(xyz);

      solver_c0->set_interface_point_value(n, i, c0_gamma);
    }
  }

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);

  Vec rhs_tmp;
  ierr = VecDuplicate(t_.vec, &rhs_tmp); CHKERRXX(ierr);

#ifdef P4_TO_P8
  BoundaryConditions3D bc_tmp;
#else
  BoundaryConditions2D bc_tmp;
#endif

  bc_tmp.setInterfaceType(NOINTERFACE);
  bc_tmp.setWallTypes(bc_t_.getWallType());
  bc_tmp.setWallValues(bc_t_.getWallValue());

  solver_t->set_bc(bc_tmp);
  solver_t->set_is_matrix_computed(is_t_matrix_computed_);
  solver_t->assemble_jump_rhs(rhs_tmp, *jump_t_, tn_jump_, rhs_tl_.vec, rhs_ts_.vec);
  solver_t->set_rhs(rhs_tmp);
  solver_t->solve(t_.vec, use_non_zero_guess_);

  t_.get_array();
  tm_.get_array();

  foreach_node(n, nodes_) tm_.ptr[n] = t_.ptr[n];

  t_.restore_array();
  tm_.restore_array();

  is_t_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD(phi_.vec, tm_.vec, 20, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(tm_.vec, tm_dd_.vec);

  ierr = VecDestroy(rhs_tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);

  Vec rhs_tmp;
  ierr = VecDuplicate(psi_t_.vec, &rhs_tmp); CHKERRXX(ierr);

#ifdef P4_TO_P8
  BoundaryConditions3D bc_tmp;
#else
  BoundaryConditions2D bc_tmp;
#endif

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
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);

  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c0_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_c0_.get_array();
  rhs_tmp.get_array();

  foreach_local_node(n, nodes_) rhs_tmp.ptr[n] = rhs_c0_.ptr[n];

  rhs_c0_.restore_array();
  rhs_tmp.restore_array();

#ifdef P4_TO_P8
  BoundaryConditions3D bc_tmp;
#else
  BoundaryConditions2D bc_tmp;
#endif

  bc_tmp.setInterfaceType(DIRICHLET);
  bc_tmp.setInterfaceValue(zero_cf_);
  bc_tmp.setWallTypes(bc_c0_.getWallType());
  bc_tmp.setWallValues(bc_c0_.getWallValue());

  solver_c0->set_bc(bc_tmp);
  solver_c0->set_is_matrix_computed(true);
  solver_c0->set_rhs(rhs_tmp.vec);

  solver_c0->solve(c0_.vec, use_non_zero_guess_);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  if (use_points_on_interface_) ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec, solver_c0, 50, 2, normal_.vec);
  else                          ls.extend_Over_Interface_TVD(phi_.vec, c0_.vec, 50, 2, normal_.vec);

  if (c0_gamma_.vec != NULL) { ierr = VecDestroy(c0_gamma_.vec); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_.vec, &c0_gamma_.vec); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi_.vec, c0_.vec, c0_gamma_.vec);

  node_neighbors_->second_derivatives_central(c0_.vec, c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c0()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c0, 0, 0, 0, 0); CHKERRXX(ierr);

  // compute bondary conditions
  psi_t_.get_array();
  psi_t_dd_.get_array();
  c1_.get_array();
  c1_dd_.get_array();
  psi_c1_.get_array();
  psi_c1_dd_.get_array();

  normal_.get_array();

  foreach_local_node(n, nodes_)
  {
    for (short i = 0; i < solver_psi_c0->pointwise_bc[n].size(); ++i)
    {
      double xyz[P4EST_DIM];
      solver_psi_c0->get_xyz_interface_point(n, i, xyz);
      double c0_gamma = solver_c0->get_interface_point_value(n, i);

      double nx = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[0]);
      double ny = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[1]);
#ifdef P4_TO_P8
      double nz = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[2]);
      double eps_v = (*eps_v_)(nx, ny, nz);
#else
      double eps_v = (*eps_v_)(nx, ny);
#endif

//      double theta_xz = solver_psi_c0->interpolate_at_interface_point(n,i,theta_xz_.ptr);
//#ifdef P4_TO_P8
//      double theta_yz = solver_psi_c0->interpolate_at_interface_point(n,i,theta_yz_.ptr);
//      double eps_v = (*eps_v_)(theta_xz, theta_yz);
//#else
//      double eps_v = (*eps_v_)(theta_xz);
//#endif

//      double psi_c0_gamma = -( (1.-kp1_)*solver_psi_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr)
//                                       *solver_psi_c0->interpolate_at_interface_point(n, i, psi_c1_.ptr, psi_c1_dd_.ptr)
//                                + t_diff_*latent_heat_/t_cond_*solver_psi_c0->interpolate_at_interface_point(n, i, psi_t_.ptr, psi_t_dd_.ptr)
//                               + (*eps_v_)(solver_psi_c0->interpolate_at_interface_point(n,i,theta_.ptr))/ml0_ )
//          /(1.-kp0_)/c0_gamma;

      double psi_c0_gamma = -( (1.-kp1_)*solver_psi_c0->interpolate_at_interface_point(n, i, c1_.ptr)
                                       *solver_psi_c0->interpolate_at_interface_point(n, i, psi_c1_.ptr)
                                + t_diff_*latent_heat_/t_cond_*solver_psi_c0->interpolate_at_interface_point(n, i, psi_t_.ptr)
                               + eps_v/ml0_ )
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

  normal_.restore_array();

  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c0_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_tmp.get_array();

  foreach_local_node(n, nodes_) rhs_tmp.ptr[n] = 0.;

  rhs_tmp.restore_array();

#ifdef P4_TO_P8
  BoundaryConditions3D bc_tmp;
#else
  BoundaryConditions2D bc_tmp;
#endif

  bc_tmp.setInterfaceType(DIRICHLET);
  bc_tmp.setInterfaceValue(zero_cf_);
  bc_tmp.setWallTypes(bc_c0_.getWallType());
  bc_tmp.setWallValues(zero_cf_);

  solver_psi_c0->set_bc(bc_tmp);
  solver_psi_c0->set_is_matrix_computed(true);
  solver_psi_c0->set_rhs(rhs_tmp.vec);
  solver_psi_c0->solve(psi_c0_.vec);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  if (use_points_on_interface_) ls.extend_Over_Interface_TVD(phi_.vec, psi_c0_.vec, solver_psi_c0, 50, 2, normal_.vec);
  else                          ls.extend_Over_Interface_TVD(phi_.vec, psi_c0_.vec, 50, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(psi_c0_.vec, psi_c0_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_psi_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c1()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_c1, 0, 0, 0, 0); CHKERRXX(ierr);

  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c1_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_c1_.get_array();
  rhs_tmp.get_array();

  foreach_node(n, nodes_) rhs_tmp.ptr[n] = rhs_c1_.ptr[n];

  rhs_c1_.restore_array();
  rhs_tmp.restore_array();

  vec_and_ptr_t mask;

  if (use_superconvergent_robin_)
  {
    std::vector<BoundaryConditionType> interface_type(1, ROBIN);
#ifdef P4_TO_P8
    std::vector<CF_3 *>                interface_value(1, c1_flux_);
    std::vector<CF_3 *>                interface_coeff(1, &c1_robin_coef_);
#else
    std::vector<CF_2 *>                interface_value(1, c1_flux_);
    std::vector<CF_2 *>                interface_coeff(1, &c1_robin_coef_);
#endif

//#ifdef P4_TO_P8
//    solver_c1_sc->set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_, &phi_zz_vector_, NULL, volumes_.vec);
//#else
//    solver_c1_sc->set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_, NULL, volumes_.vec);
//#endif

    solver_c1_sc->set_bc_wall_type(bc_c1_.getWallType());
    solver_c1_sc->set_bc_wall_value(bc_c1_.getWallValue());
    solver_c1_sc->set_bc_interface_type(interface_type);
    solver_c1_sc->set_bc_interface_value(interface_value);
    solver_c1_sc->set_bc_interface_coeff(interface_coeff);
    solver_c1_sc->set_is_matrix_computed(is_c1_matrix_computed_);
    solver_c1_sc->set_rhs(rhs_tmp.vec);
    solver_c1_sc->solve(c1_.vec, use_non_zero_guess_);

    mask.vec = solver_c1_sc->get_mask();

    vec_and_ptr_t volumes;

    volumes.vec = solver_c1_sc->get_volumes();

    volumes.get_array();
    mask.get_array();

    foreach_node(n, nodes_) if (mask.ptr[n] >= -0.3 || volumes.ptr[n] < volume_thresh_*min_volume_) mask.ptr[n] = 1.;

    mask.restore_array();
    volumes.restore_array();

//    if (volumes_.vec == NULL)
//    {
//      Vec volumes_tmp = solver_c1_sc->get_volumes();
//      ierr = VecDuplicate(phi_.vec, &volumes_.vec); CHKERRXX(ierr);
//      copy_ghosted_vec(volumes_tmp, volumes_.vec);
//    }

  } else {
#ifdef P4_TO_P8
    BoundaryConditions3D bc_tmp;
#else
    BoundaryConditions2D bc_tmp;
#endif

    bc_tmp.setInterfaceType(ROBIN);
    bc_tmp.setInterfaceValue(*c1_flux_);
    bc_tmp.setRobinCoef(c1_robin_coef_);
    bc_tmp.setWallTypes(bc_c1_.getWallType());
    bc_tmp.setWallValues(bc_c1_.getWallValue());

    solver_c1->set_bc(bc_tmp);
    solver_c1->set_is_matrix_computed(is_c1_matrix_computed_);
    solver_c1->set_rhs(rhs_tmp.vec);
    solver_c1->solve(c1_.vec, use_non_zero_guess_);

    mask.vec = solver_c1->get_mask();
  }

  is_c1_matrix_computed_ = true;


  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  ls.extend_Over_Interface_TVD(phi_.vec, mask.vec, c1_.vec, 20, 2);
//  ls.extend_Over_Interface_TVD(phi_.vec, c1_.vec, 20, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(c1_.vec, c1_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c1()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);

  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c1_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_tmp.get_array();

  foreach_node(n, nodes_) rhs_tmp.ptr[n] = 0.;

  rhs_tmp.restore_array();

  vec_and_ptr_t mask;

  if (use_superconvergent_robin_)
  {
    std::vector<BoundaryConditionType> interface_type(1, ROBIN);
#ifdef P4_TO_P8
    std::vector<CF_3 *>                interface_value(1, &psi_c1_interface_value_);
    std::vector<CF_3 *>                interface_coeff(1, &c1_robin_coef_);
#else
    std::vector<CF_2 *>                interface_value(1, &psi_c1_interface_value_);
    std::vector<CF_2 *>                interface_coeff(1, &c1_robin_coef_);
#endif

    solver_c1_sc->set_bc_wall_type(bc_c1_.getWallType());
    solver_c1_sc->set_bc_wall_value(zero_cf_);
    solver_c1_sc->set_bc_interface_type(interface_type);
    solver_c1_sc->set_bc_interface_value(interface_value);
    solver_c1_sc->set_bc_interface_coeff(interface_coeff);
    solver_c1_sc->set_is_matrix_computed(is_c1_matrix_computed_);
    solver_c1_sc->set_rhs(rhs_tmp.vec);
    solver_c1_sc->solve(psi_c1_.vec);

    mask.vec = solver_c1_sc->get_mask();

    vec_and_ptr_t volumes;

    volumes.vec = solver_c1_sc->get_volumes();

    volumes.get_array();
    mask.get_array();

    foreach_node(n, nodes_) if (mask.ptr[n] >= -0.3 || volumes.ptr[n] < volume_thresh_*min_volume_) mask.ptr[n] = 1.;

    mask.restore_array();
    volumes.restore_array();

  } else {
#ifdef P4_TO_P8
    BoundaryConditions3D bc_tmp;
#else
    BoundaryConditions2D bc_tmp;
#endif

    bc_tmp.setInterfaceType(ROBIN);
    bc_tmp.setInterfaceValue(psi_c1_interface_value_);
    bc_tmp.setRobinCoef(c1_robin_coef_);
    bc_tmp.setWallTypes(bc_c1_.getWallType());
    bc_tmp.setWallValues(zero_cf_);

    solver_c1->set_bc(bc_tmp);
    solver_c1->set_is_matrix_computed(is_c1_matrix_computed_);
    solver_c1->set_rhs(rhs_tmp.vec);
    solver_c1->solve(psi_c1_.vec);

    mask.vec = solver_c1->get_mask();
  }

  is_c1_matrix_computed_ = true;

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

//  ls.extend_Over_Interface_TVD(phi_.vec, mask.vec, psi_c1_.vec, 20, 2);
  ls.extend_Over_Interface_TVD(phi_.vec, psi_c1_.vec, 20, 2, normal_.vec);

  node_neighbors_->second_derivatives_central(psi_c1_.vec, psi_c1_dd_.vec);

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0_robin()
{
  vec_and_ptr_t rhs_tmp;
  ierr = VecDuplicate(rhs_c0_.vec, &rhs_tmp.vec); CHKERRXX(ierr);

  rhs_c0_.get_array();
  rhs_tmp.get_array();

  foreach_node(n, nodes_) rhs_tmp.ptr[n] = rhs_c0_.ptr[n];

  rhs_c0_.restore_array();
  rhs_tmp.restore_array();

  if (use_superconvergent_robin_)
  {
    my_p4est_poisson_nodes_mls_sc_t solver_c0_sc(node_neighbors_);

#ifdef P4_TO_P8
    solver_c0_sc.set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_, &phi_zz_vector_, NULL, volumes_.vec);
#else
    solver_c0_sc.set_geometry(1, &action_, &color_, &phi_vector_, &phi_xx_vector_, &phi_yy_vector_, NULL, volumes_.vec);
#endif

    solver_c0_sc.set_diag_add(1.);
    solver_c0_sc.set_mu(Dl0_*dt_);
    solver_c0_sc.set_use_sc_scheme(false);
    solver_c0_sc.set_integration_order(2);
    solver_c0_sc.set_use_taylor_correction(1);
    solver_c0_sc.set_keep_scalling(true);
    solver_c0_sc.set_kink_treatment(1);
    solver_c0_sc.set_try_remove_hanging_cells(0);

    std::vector<BoundaryConditionType> interface_type(1, ROBIN);
#ifdef P4_TO_P8
    std::vector<CF_3 *>                interface_value(1, c0_flux_);
    std::vector<CF_3 *>                interface_coeff(1, &c0_robin_coef_);
#else
    std::vector<CF_2 *>                interface_value(1, c0_flux_);
    std::vector<CF_2 *>                interface_coeff(1, &c0_robin_coef_);
#endif

    solver_c0_sc.set_bc_wall_type(bc_c0_.getWallType());
    solver_c0_sc.set_bc_wall_value(bc_c0_.getWallValue());
    solver_c0_sc.set_bc_interface_type(interface_type);
    solver_c0_sc.set_bc_interface_value(interface_value);
    solver_c0_sc.set_bc_interface_coeff(interface_coeff);
    solver_c0_sc.set_rhs(rhs_tmp.vec);
    solver_c0_sc.solve(c0_.vec);

    vec_and_ptr_t mask;
    mask.vec = solver_c0_sc.get_mask();

    vec_and_ptr_t volumes;

    volumes.vec = solver_c0_sc.get_volumes();

    volumes.get_array();
    mask.get_array();

    foreach_node(n, nodes_) if (mask.ptr[n] >= -0.3 || volumes.ptr[n] < volume_thresh_*min_volume_) mask.ptr[n] = 1.;

    mask.restore_array();
    volumes.restore_array();

    my_p4est_level_set_t ls(node_neighbors_);
    ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    ls.extend_Over_Interface_TVD(phi_.vec, mask.vec, c0_.vec, 100, 2);

  } else {
#ifdef P4_TO_P8
    BoundaryConditions3D bc_tmp;
#else
    BoundaryConditions2D bc_tmp;
#endif

    bc_tmp.setInterfaceType(ROBIN);
    bc_tmp.setInterfaceValue(*c0_flux_);
    bc_tmp.setRobinCoef(c0_robin_coef_);
    bc_tmp.setWallTypes(bc_c0_.getWallType());
    bc_tmp.setWallValues(bc_c0_.getWallValue());


    // solve for c0 using Robin BC
    my_p4est_poisson_nodes_t solver_c0_robin(node_neighbors_);
#ifdef P4_TO_P8
    solver_c0_robin.set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1], phi_dd_.vec[2]);
#else
    solver_c0_robin.set_phi(phi_.vec, phi_dd_.vec[0], phi_dd_.vec[1]);
#endif
    solver_c0_robin.set_diagonal(1.);
    solver_c0_robin.set_mu(dt_*Dl0_);
    solver_c0_robin.set_use_refined_cube(use_refined_cube_);
    solver_c0_robin.set_bc(bc_tmp);
    solver_c0_robin.set_rhs(rhs_tmp.vec);
    solver_c0_robin.solve(c0_.vec);

    Vec mask = solver_c0->get_mask();

    my_p4est_level_set_t ls(node_neighbors_);
    ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    ls.extend_Over_Interface_TVD(phi_.vec, mask, c0_.vec, 100, 2);
  }

  ierr = VecDestroy(rhs_tmp.vec); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::compute_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);

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

//    c0n_.ptr[n] = MAX(c0n_.ptr[n], 0.);
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

//    c0n_.ptr[n] = MAX(c0n_.ptr[n], 0.);
  }

  ierr = VecGhostUpdateEnd(c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  c0_.restore_array();
  c0n_.restore_array();
  normal_.restore_array();

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

//  ls.extend_Over_Interface_TVD(phi_.vec, c0n_.vec, 20, 2, normal_.vec);

  // compute second derivatives for interpolation purposes
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_.vec[dim], &c0n_dd_.vec[dim]); CHKERRXX(ierr);
  }

  if (c0n_gamma_.vec != NULL) { ierr = VecDestroy(c0n_gamma_.vec); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_.vec, &c0n_gamma_.vec); CHKERRXX(ierr);

  ls.extend_from_interface_to_whole_domain_TVD(phi_.vec, c0n_.vec, c0n_gamma_.vec);

  node_neighbors_->second_derivatives_central(c0n_.vec, c0n_dd_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_poisson_nodes_multialloy_t::compute_psi_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);

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
  foreach_dimension(dim)
  {
    if (psi_c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(psi_c0n_dd_.vec[dim]); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_dd_.vec[dim], &psi_c0n_dd_.vec[dim]); CHKERRXX(ierr);
  }

  node_neighbors_->second_derivatives_central(psi_c0n_.vec, psi_c0n_dd_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::adjust_c0_gamma(int iteration)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);

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

  kappa_.get_array();

  /* main loop */

  bc_error_max_ = 0;
  double xyz[P4EST_DIM];

  velo_max_ = 0;

  foreach_local_node(n, nodes_)
  {
    bc_error_.ptr[n] = 0;
    if (solver_c0->pointwise_bc[n].size())
    {
      for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
      {
        solver_c0->get_xyz_interface_point(n, i, xyz);

        double c0_gamma = solver_c0->get_interface_point_value(n, i);

        // interpolate concentration
//        double conc_term = ml0_*c0_gamma + ml1_*solver_c0->interpolate_at_interface_point(n, i, c1_.ptr, c1_dd_.ptr);
        double conc_term = ml0_*c0_gamma + ml1_*solver_c0->interpolate_at_interface_point(n, i, c1_.ptr);

        // interpolate temperature
//        double ther_term = solver_c0->interpolate_at_interface_point(n, i, tm_.ptr, tm_dd_.ptr);
        double ther_term = solver_c0->interpolate_at_interface_point(n, i, tm_.ptr);

        // interpolate velocity
//        double c0n = solver_c0->interpolate_at_interface_point(n, i, c0n_.ptr, c0n_dd_.ptr);
        double c0n = solver_c0->interpolate_at_interface_point(n, i, c0n_.ptr);

//        double theta_xz = solver_c0->interpolate_at_interface_point(n, i, theta_xz_.ptr);
//#ifdef P4_TO_P8
//        double theta_yz = solver_c0->interpolate_at_interface_point(n, i, theta_yz_.ptr);
//#endif
        double kappa = solver_c0->interpolate_at_interface_point(n, i, kappa_.ptr);

//#ifdef P4_TO_P8
//        double eps_v = (*eps_v_)(theta_xz, theta_yz);
//        double eps_c = (*eps_c_)(theta_xz, theta_yz);
//#else
//        double eps_v = (*eps_v_)(theta_xz);
//        double eps_c = (*eps_c_)(theta_xz);
//#endif

        double nx = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[0]);
        double ny = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[1]);
#ifdef P4_TO_P8
        double nz = solver_psi_c0->interpolate_at_interface_point(n,i,normal_.ptr[2]);
        double eps_v = (*eps_v_)(nx, ny, nz);
        double eps_c = (*eps_c_)(nx, ny, nz);
#else
        double eps_v = (*eps_v_)(nx, ny);
        double eps_c = (*eps_c_)(nx, ny);
#endif


//        double velo = vn_from_c0_.value(xyz);
        double velo = ( Dl0_/(1.-kp0_)*(c0n-c0_flux_->value(xyz))/c0_gamma );
//        double error = (conc_term + Tm_ - ther_term - eps_v*( Dl0_/(1.-kp0_)*(c0n-c0_flux_->value(xyz))/c0_gamma ) + eps_c*kappa + GT_->value(xyz));
        double error = (conc_term + Tm_ - ther_term - eps_v*velo + eps_c*kappa + GT_->value(xyz));

        bc_error_.ptr[n] = MAX(bc_error_.ptr[n], fabs(error));
        bc_error_max_ = MAX(bc_error_max_, fabs(error));

        double change = error/ml0_;

        if (iteration%pin_every_n_steps_ != 0)
        {
          double psi_c0_gamma = solver_psi_c0->get_interface_point_value(n, i);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr, psi_c0n_dd_.ptr);
          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
//          double vn = vn_from_c0_.value(xyz);
//          double dem = (1. + Dl0_*psi_c0n - (1.-kp0_)*vn*psi_c0_gamma);
//          double psi_c0n  = solver_c0->interpolate_at_interface_point(n, i, psi_c0n_.ptr);
          change /= (1. + Dl0_*psi_c0n - (1.-kp0_)*velo*psi_c0_gamma);
//          change /= (1. + Dl0_*psi_c0n - Dl0_*(c0n-c0_flux_->value(xyz))*psi_c0_gamma/c0_gamma);
        }

        solver_c0->set_interface_point_value(n, i, c0_gamma-change);

//        velo_max_ = MAX(velo_max_, fabs(Dl0_/(1.-kp0_)*c0n/c0_gamma));
        velo_max_ = MAX(velo_max_, fabs(velo));
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &bc_error_max_, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &velo_max_,     1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);



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
  kappa_.restore_array();

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}
