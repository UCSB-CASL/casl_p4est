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

my_p4est_poisson_nodes_multialloy_t::my_p4est_poisson_nodes_multialloy_t(my_p4est_node_neighbors_t *node_neighbors, int num_comps)
  : node_neighbors_(node_neighbors),
    p4est_(node_neighbors->p4est), nodes_(node_neighbors->nodes), ghost_(node_neighbors->ghost), myb_(node_neighbors->myb),
    interp_(node_neighbors)
{
//  psi_c1_interface_value_.set_ptr(this);
//  jump_psi_tn_  .set_ptr(this);
//  vn_from_c0_   .set_ptr(this);
//  c0_robin_coef_.set_ptr(this);
//  c1_robin_coef_.set_ptr(this);
//  tn_jump_      .set_ptr(this);
//  bc_error_cf_  .set_ptr(this);

  pin_every_n_iterations_ = INT_MAX;
  bc_tolerance_           = 1.e-12;
  max_iterations_         = 10;

  contr_phi_dd_owned_    = false;
  front_phi_dd_owned_    = false;
  front_normal_owned_    = false;
  front_curvature_owned_ = false;

  gibbs_thomson_ = &zero_cf;

  temp_diff_l_ = 1.;
  temp_diag_l_ = 1.;
  temp_diff_s_ = 1.;
  temp_diag_s_ = 1.;
  latent_heat_ = 1.;
  melting_temp_ = 1.;
  bc_error_max_ = 1.;

  solver_temp_ = NULL;
  solver_conc_leading_ = NULL;

  num_comps_ = num_comps;

  rhs_c_      .resize(num_comps_);
  c_          .resize(num_comps_);
  c_d_        .resize(num_comps_);
  c_dd_       .resize(num_comps_);
  psi_c_      .resize(num_comps_);
  psi_c_d_    .resize(num_comps_);
  solver_conc_.resize(num_comps_, NULL);

  conc_diag_          .resize(num_comps_, 1);
  conc_diff_          .resize(num_comps_, 1);
  part_coeff_         .resize(num_comps_, .5);
  front_conc_flux_    .resize(num_comps_, NULL);
  wall_bc_type_conc_  .resize(num_comps_, NULL);
  wall_bc_value_conc_ .resize(num_comps_, NULL);
  contr_bc_type_conc_ .resize(num_comps_, NEUMANN);
  contr_bc_value_conc_.resize(num_comps_, NULL);

  tl_dd_.resize((P4EST_DIM-1)*3);
  ts_dd_.resize((P4EST_DIM-1)*3);

  for (int i = 0; i < num_comps_; ++i)
  {
    c_dd_[i].resize((P4EST_DIM-1)*3);
  }

  cube_refinement_ = 1;
  integration_order_ = 2;

  second_derivatives_owned_  = false;
  use_superconvergent_robin_ = false;
  use_superconvergent_jump_  = false;
  update_c0_robin_           = 0;
  use_points_on_interface_   = true;
  zero_negative_velocity_    = false;
  flatten_front_values_      = true;
  always_use_centroid_       = false;
  verbose_                   = false;

  volume_thresh_ = 1.e-2;
  err_eps_ = 1.e-5;

  num_extend_iterations_ = 50;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est_, dxyz);

  min_volume_ = MULTD(dxyz[0], dxyz[1], dxyz[2]);

  extension_band_use_    = 8.*pow(min_volume_, 1./ double(P4EST_DIM));
  extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM));
  extension_band_check_  = 6.*pow(min_volume_, 1./ double(P4EST_DIM));
  extension_tol_  = 1.e-9;
  extension_use_nonzero_guess_ = false;

  poisson_use_nonzero_guess_ = true;
}

my_p4est_poisson_nodes_multialloy_t::~my_p4est_poisson_nodes_multialloy_t()
{
  clear_contr();
  clear_front();

  c0n_           .destroy();
  c0n_dd_        .destroy();
  psi_c0n_       .destroy();
  psi_c0n_dd_    .destroy();
  c0_gamma_      .destroy();
  c0n_gamma_     .destroy();
  bc_error_gamma_.destroy();
  rhs_zero_      .destroy();
  psi_c0d_       .destroy();

  if (solver_temp_ != NULL) delete solver_temp_;
  if (solver_conc_leading_ != NULL) delete solver_conc_leading_;

  for (int i = 0; i < num_comps_; ++i)
  {
    if (solver_conc_[i] != NULL) delete solver_conc_[i];
  }


  for (int i = 0; i < num_comps_; ++i)
  {
    c_d_[i].destroy();
    c_dd_[i].destroy();
  }

  ts_d_.destroy();
  ts_dd_.destroy();

  tl_d_.destroy();
  tl_dd_.destroy();
//  if (second_derivatives_owned_)
//  if (1)
//  {
//    tl_dd_.destroy();
//    ts_dd_.destroy();

//    for (int i = 0; i < num_comps_; ++i) c_dd_[i].destroy();
//  }
}

void my_p4est_poisson_nodes_multialloy_t::clear_front()
{
  if (front_phi_dd_owned_)    front_phi_dd_   .destroy();
  if (front_normal_owned_)    front_normal_   .destroy();
  if (front_curvature_owned_) front_curvature_.destroy();
}

void my_p4est_poisson_nodes_multialloy_t::set_front(Vec phi, Vec* phi_dd, Vec* normal, Vec curvature)
{
  clear_front();

  // level-set function itself
  front_phi_.vec = phi;

  // second derivatives
  if (phi_dd != NULL)
  {
    front_phi_dd_owned_ = false;
    foreach_dimension(dim) front_phi_dd_.vec[dim] = phi_dd[dim];
  }
  else
  {
    front_phi_dd_owned_ = true;
    front_phi_dd_.create(p4est_, nodes_);
    node_neighbors_->second_derivatives_central(front_phi_.vec, front_phi_dd_.vec);
  }

  // normal vector
  if (normal != NULL)
  {
    front_normal_owned_ = false;
    foreach_dimension(dim) front_normal_.vec[dim] = normal[dim];
  }
  else
  {
    front_normal_owned_ = true;
    front_normal_.create(p4est_, nodes_);
    compute_normals(*node_neighbors_, front_phi_.vec, front_normal_.vec);
  }

  // curvature
  if (curvature != NULL)
  {
    front_curvature_owned_ = false;
    front_curvature_.vec   = curvature;
  }
  else
  {
    front_curvature_owned_ = true;
    front_curvature_.create(phi);

    vec_and_ptr_dim_t tmp; tmp.create(front_normal_.vec);
    compute_normals_and_mean_curvature(*node_neighbors_, front_phi_.vec, tmp.vec, front_curvature_.vec);
    tmp.destroy();
  }
}

void my_p4est_poisson_nodes_multialloy_t::clear_contr()
{
  if (contr_phi_dd_owned_) contr_phi_dd_.destroy();
}

void my_p4est_poisson_nodes_multialloy_t::set_container(Vec phi, Vec* phi_dd)
{
  clear_contr();

  // level-set function itself
  contr_phi_.vec = phi;

  // second derivatives
  if (phi_dd != NULL)
  {
    contr_phi_dd_owned_ = false;
    foreach_dimension(dim) contr_phi_dd_.vec[dim] = phi_dd[dim];
  }
  else
  {
    contr_phi_dd_owned_ = true;
    contr_phi_dd_.create(p4est_, nodes_);
    node_neighbors_->second_derivatives_central(contr_phi_.vec, contr_phi_dd_.vec);
  }
}

int my_p4est_poisson_nodes_multialloy_t::solve(Vec tl, Vec ts, Vec c[], Vec c0d[], Vec bc_error, double &bc_error_max, bool use_non_zero_guess,
                                               std::vector<double> *num_pdes, std::vector<double> *error,
                                               Vec psi_tl, Vec psi_ts, Vec psi_cl[])
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  poisson_use_nonzero_guess_ = use_non_zero_guess;
  second_derivatives_owned_ = true;

  // create level sets for liquid and solid
  liquid_phi_.create(front_phi_.vec);
  solid_phi_.create(front_phi_.vec);

  VecCopyGhost(front_phi_.vec, liquid_phi_.vec);
  VecCopyGhost(front_phi_.vec, solid_phi_.vec);
  VecScaleGhost(solid_phi_.vec, -1.);

  if (contr_phi_.vec != NULL)
  {
    VecPointwiseMaxGhost(liquid_phi_.vec, liquid_phi_.vec, contr_phi_.vec);
    VecPointwiseMaxGhost(solid_phi_.vec, solid_phi_.vec, contr_phi_.vec);
  }

  liquid_normal_.create(front_normal_.vec);
  solid_normal_.create(front_normal_.vec);

  compute_normals(*node_neighbors_, liquid_phi_.vec, liquid_normal_.vec);
  compute_normals(*node_neighbors_, solid_phi_.vec,  solid_normal_.vec);

  // get input Vec's
  tl_ .set(tl);
  ts_ .set(ts);
  c0d_.set(c0d);
  bc_error_.set(bc_error);

  for (int i = 0; i < num_comps_; ++i)
    c_[i].set(c[i]);

  // allocate memory for second order derivatives
  tl_d_.destroy();
  tl_d_.create(p4est_, nodes_);
  tl_dd_.destroy();
  tl_dd_.create(p4est_, nodes_);

  ts_d_.destroy();
  ts_d_.create(tl_d_.vec);
  ts_dd_.destroy();
  ts_dd_.create(tl_dd_.vec.data());

  for (int i = 0; i < num_comps_; ++i)
  {
    c_d_[i].destroy();
    c_d_[i].create(tl_d_.vec);
    c_dd_[i].destroy();
    c_dd_[i].create(tl_dd_.vec.data());
  }

  // allocate memory for lagrangian multipliers
  if (psi_tl == NULL) psi_tl_.create(tl_.vec);
  else                psi_tl_.set(psi_tl);

  if (psi_ts == NULL) psi_ts_.create(ts_.vec);
  else                psi_ts_.set(psi_ts);

  for (int i = 0; i < num_comps_; ++i)
  {
    if (psi_cl == NULL) psi_c_[i].create(tl_.vec);
    else                psi_c_[i].set(psi_cl[i]);
  }

  psi_tl_d_.create(tl_d_.vec);
  psi_ts_d_.create(ts_d_.vec);

  for (int i = 0; i < num_comps_; ++i)
  {
    psi_c_d_[i].create(tl_d_.vec);
  }

  psi_c0d_.create(front_normal_.vec);

  // precompute first and second derivatives for extension purposes
  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, tl_.vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normal_.vec, NULL, NULL, 1, tl_d_.vec, tl_dd_.vec.data());
  ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  ts_.vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normal_.vec,  NULL, NULL, 1, ts_d_.vec, ts_dd_.vec.data());
  for (int i = 0; i < num_comps_; ++i)
  {
    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[i].vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normal_.vec, NULL, NULL, 1, c_d_[i].vec, c_dd_[i].vec.data());
  }

  if (psi_tl != NULL) ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_tl_.vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normal_.vec, NULL, NULL, 1, psi_tl_d_.vec, NULL);
  if (psi_ts != NULL) ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  psi_ts_.vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normal_.vec,  NULL, NULL, 1, psi_ts_d_.vec, NULL);

  if (psi_cl != NULL)
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[i].vec, 0, 1, 0, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normal_.vec, NULL, NULL, 1, psi_c_d_[i].vec);
    }
  }

  // for logging purposes
  if (num_pdes != NULL) num_pdes->clear();
  if (error    != NULL) error   ->clear();
  int num_pdes_solved = 0;

  initialize_solvers();

  int  iteration = 0;
  bc_error_max_  = DBL_MAX;

  int conc_start = update_c0_robin_ == 2 ? 0 : 1;
  int conc_num = num_comps_ - conc_start;

  while (bc_error_max_ > bc_tolerance_ &&
         iteration < max_iterations_)
  {
    ++iteration;

    // solve for physical quantities

    solve_c0(); ++num_pdes_solved;
    compute_c0n();
    compute_pw_bc_values(conc_start, conc_num);
    solve_t();  ++num_pdes_solved;
    solve_c(conc_start, conc_num);  ++num_pdes_solved;

//    if (iteration < max_iterations_)
    bool check = iteration%pin_every_n_iterations_ == 0;

    {
      // solve for lagrangian multipliers
      if (iteration%pin_every_n_iterations_ != 0)
      {
        compute_pw_bc_psi_values(conc_start, conc_num);

//        if (var_scheme_ != VALUE || iteration == 1)
        if (iteration == 1)
        {
          solve_psi_t();   ++num_pdes_solved;
        }
        solve_psi_c(conc_start, conc_num);   ++num_pdes_solved;
        solve_psi_c0();  ++num_pdes_solved;
        compute_psi_c0n();
      }

      // adjust boundary conditions
      adjust_c0_gamma(iteration%pin_every_n_iterations_ == 0);
    }

    // logging for convergence studies
    if (num_pdes != NULL) { num_pdes->push_back(num_pdes_solved); }
    if (error    != NULL) { error->push_back(bc_error_max_); }

    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: bc error = %g, max velo = %g\n", iteration, bc_error_max_, velo_max_); CHKERRXX(ierr);
  }


  if (zero_negative_velocity_)
  {
    c0n_gamma_.get_array();
    foreach_node(n, nodes_) { if (c0n_gamma_.ptr[n] < 0) c0n_gamma_.ptr[n] = 0; }
    c0n_gamma_.restore_array();
  }

  if (update_c0_robin_ == 1)
  {
    compute_pw_bc_values(0, 1);
    solve_c(0, 1);
  }

  if (psi_ts != NULL)
  {
    ls.set_show_convergence(verbose_);
    ls.extend_Over_Interface_TVD_Full(solid_phi_.vec, psi_ts_.vec, num_extend_iterations_, 1,
                                      extension_tol_, extension_band_use_, extension_band_extend_, extension_band_check_,
                                      solid_normal_.vec, NULL, NULL,
                                      false, psi_ts_d_.vec, NULL);
  }


  // clean everything
  if (psi_tl == NULL) psi_tl_.destroy();
  if (psi_ts == NULL) psi_ts_.destroy();
  psi_tl_d_.destroy();
  psi_ts_d_.destroy();

  for (int i = 0; i < num_comps_; ++i)
  {
    if (psi_cl == NULL) psi_c_[i].destroy();
    psi_c_d_[i].destroy();
  }

  psi_c0d_.destroy();

  liquid_phi_.destroy();
  solid_phi_.destroy();

  liquid_normal_.destroy();
  solid_normal_.destroy();

  bc_error_max = bc_error_max_;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  return iteration;
}



void my_p4est_poisson_nodes_multialloy_t::initialize_solvers()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);

  rhs_zero_.destroy();
  rhs_zero_.create(rhs_tl_.vec);
  VecSetGhost(rhs_zero_.vec, 0);

//  while (1)
//  {
//    my_p4est_poisson_nodes_mls_t test(node_neighbors_);

//    test.add_boundary(MLS_INTERSECTION, front_phi_.vec, front_phi_dd_.vec, DIRICHLET, zero_cf, zero_cf);
//    test.set_diag(conc_diag_[0]);
//    test.set_mu(conc_diff_[0]);
//    test.set_wc(*wall_bc_type_conc_[0], *wall_bc_value_conc_[0]);
//    test.set_rhs(rhs_c_[0].vec);

//    if (contr_phi_.vec != NULL)
//    {
//      test.add_boundary(MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_conc_[0], zero_cf, zero_cf);
//    }

//    test.preassemble_linear_system();
//  }

  if (solver_temp_ != NULL) delete solver_temp_ ;
  solver_temp_ = new my_p4est_poisson_nodes_mls_t(node_neighbors_);

  if (solver_conc_leading_ != NULL) delete solver_conc_leading_ ;
  solver_conc_leading_ = new my_p4est_poisson_nodes_mls_t(node_neighbors_);

  for (int i = update_c0_robin_ == 0 ? 1 : 0; i < num_comps_; ++i)
  {
    if (solver_conc_[i] != NULL) delete solver_conc_[i];
    solver_conc_[i] = new my_p4est_poisson_nodes_mls_t(node_neighbors_);
  }

  // c[0]
  solver_conc_leading_->add_boundary(MLS_INTERSECTION, front_phi_.vec, front_phi_dd_.vec, DIRICHLET, zero_cf, zero_cf);
  solver_conc_leading_->set_diag(conc_diag_[0]);
  solver_conc_leading_->set_mu(conc_diff_[0]);
  solver_conc_leading_->set_wc(*wall_bc_type_conc_[0], *wall_bc_value_conc_[0]);
  solver_conc_leading_->set_rhs(rhs_c_[0].vec);
  solver_conc_leading_->set_store_finite_volumes(1);
  solver_conc_leading_->set_cube_refinement(cube_refinement_);
  solver_conc_leading_->set_use_sc_scheme(use_superconvergent_robin_);
  solver_conc_leading_->set_integration_order(integration_order_);

  if (contr_phi_.vec != NULL)
  {
    solver_conc_leading_->add_boundary(MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_conc_[0], zero_cf, zero_cf);
  }

  solver_conc_leading_->preassemble_linear_system();

  // t
  solver_temp_->add_interface(MLS_INTERSECTION, front_phi_.vec, front_phi_dd_.vec, zero_cf, zero_cf);
  solver_temp_->set_diag(temp_diag_l_, temp_diag_s_);
  solver_temp_->set_mu(temp_diff_l_, temp_diff_s_);
  solver_temp_->set_integration_order(integration_order_);
  solver_temp_->set_use_sc_scheme(0);
  solver_temp_->set_cube_refinement(cube_refinement_);
  solver_temp_->set_store_finite_volumes(1);
//  solver_temp_->set_finite_volumes(NULL, NULL, fvs, fvs_map);
  solver_temp_->set_wc(*wall_bc_type_temp_, *wall_bc_value_temp_);
  solver_temp_->set_rhs(rhs_tl_.vec, rhs_ts_.vec);

  if (contr_phi_.vec != NULL)
  {
    solver_temp_->add_boundary (MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_temp_, *contr_bc_value_temp_, zero_cf);
  }

  solver_temp_->preassemble_linear_system();

  // copy finite volumes
  std::vector<my_p4est_finite_volume_t> *fvs;
  std::vector<int> *fvs_map;

  // rest of c[]
  int i_start = update_c0_robin_ == 0 ? 1 : 0;
  for (int i = i_start; i < num_comps_; ++i)
  {
    solver_conc_[i]->add_boundary(MLS_INTERSECTION, front_phi_.vec, front_phi_dd_.vec, ROBIN, zero_cf, zero_cf);
    solver_conc_[i]->set_diag(conc_diag_[i]);
    solver_conc_[i]->set_mu(conc_diff_[i]);
    solver_conc_[i]->set_use_sc_scheme(use_superconvergent_robin_);
    solver_conc_[i]->set_integration_order(integration_order_);
    solver_conc_[i]->set_use_taylor_correction(1);
    solver_conc_[i]->set_kink_treatment(1);
    solver_conc_[i]->set_store_finite_volumes(1);
    solver_conc_[i]->set_cube_refinement(cube_refinement_);
    solver_conc_[i]->set_wc(*wall_bc_type_conc_[i], *wall_bc_value_conc_[i]);
    if (i != i_start)
    {
      solver_conc_[i]->set_finite_volumes(fvs, fvs_map, NULL, NULL);
    }

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->add_boundary(MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_conc_[i], zero_cf, zero_cf);
    }

    solver_conc_[i]->preassemble_linear_system();

    if (i == i_start)
    {
      solver_conc_[i]->get_boundary_finite_volumes(fvs, fvs_map);
    }
  }

  // allocate memory for pointwise values
  pw_c_values_      .resize(num_comps_);
  pw_c_values_robin_.resize(num_comps_);
  pw_c_coeffs_robin_.resize(num_comps_);

  pw_psi_c_values_      .resize(num_comps_);
  pw_psi_c_values_robin_.resize(num_comps_);
  pw_psi_c_coeffs_robin_.resize(num_comps_);

  for (int i = update_c0_robin_ == 0 ? 1 : 0; i < num_comps_; ++i)
  {
    pw_c_values_      [i].resize(solver_conc_[i]->pw_bc_num_value_pts(0), 0);
    pw_c_values_robin_[i].resize(solver_conc_[i]->pw_bc_num_robin_pts(0), 0);
    pw_c_coeffs_robin_[i].resize(solver_conc_[i]->pw_bc_num_robin_pts(0), 0);

    pw_psi_c_values_      [i].resize(solver_conc_[i]->pw_bc_num_value_pts(0), 0);
    pw_psi_c_values_robin_[i].resize(solver_conc_[i]->pw_bc_num_robin_pts(0), 0);
    pw_psi_c_coeffs_robin_[i].resize(solver_conc_[i]->pw_bc_num_robin_pts(0), 0);
  }

  pw_t_sol_jump_taylor_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);
  pw_t_flx_jump_taylor_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);
  pw_t_flx_jump_integr_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);

  pw_psi_t_sol_jump_taylor_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);
  pw_psi_t_flx_jump_taylor_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);
  pw_psi_t_flx_jump_integr_.resize(solver_temp_->pw_jc_num_taylor_pts(0), 0);

  pw_c0_values_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);
  pw_psi_c0_values_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);

  // sample c0 guess at boundary points
  double xyz[P4EST_DIM];
  for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0); ++i)
  {
    solver_conc_leading_->pw_bc_xyz_value_pt(0, i, xyz);
    pw_c0_values_[i] = c0_guess_->value(xyz);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Solving for temperature... \n"); CHKERRXX(ierr);
  }

  vec_and_ptr_t sol;
  sol.create(tl_.vec);

  if (poisson_use_nonzero_guess_)
  {
    front_phi_.get_array();
    tl_       .get_array();
    ts_       .get_array();
    sol       .get_array();

    foreach_node(n, nodes_)
    {
      sol.ptr[n] = front_phi_.ptr[n] < 0 ? tl_.ptr[n] : ts_.ptr[n];
    }

    front_phi_.restore_array();
    tl_       .restore_array();
    ts_       .restore_array();
    sol       .restore_array();
  }

  solver_temp_->set_wc(*wall_bc_type_temp_, *wall_bc_value_temp_, false);
  solver_temp_->set_rhs(rhs_tl_.vec, rhs_ts_.vec);
  solver_temp_->set_jc(0, pw_t_sol_jump_taylor_, pw_t_flx_jump_taylor_, pw_t_flx_jump_integr_);

  if (contr_phi_.vec != NULL)
  {
    solver_temp_->set_bc(0, contr_bc_type_temp_, *contr_bc_value_temp_, zero_cf);
  }

  solver_temp_->solve(sol.vec, poisson_use_nonzero_guess_);

//  VecCopyGhost(sol.vec, tl_.vec);
//  VecCopyGhost(sol.vec, ts_.vec);

  liquid_phi_.get_array();
  solid_phi_ .get_array();
  tl_        .get_array();
  ts_        .get_array();
  sol        .get_array();

  foreach_node(n, nodes_)
  {
    if (liquid_phi_.ptr[n] < 0) tl_.ptr[n] = sol.ptr[n];
    if (solid_phi_ .ptr[n] < 0) ts_.ptr[n] = sol.ptr[n];
  }

  liquid_phi_.restore_array();
  solid_phi_ .restore_array();
  tl_        .restore_array();
  ts_        .restore_array();
  sol        .restore_array();

  sol.destroy();

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, tl_.vec, num_extend_iterations_, 2,
                                    extension_tol_, extension_band_use_, extension_band_extend_, extension_band_check_,
                                    liquid_normal_.vec, NULL, NULL,
                                    false, tl_d_.vec, tl_dd_.vec.data());
  ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  ts_.vec, num_extend_iterations_, 2,
                                    extension_tol_, extension_band_use_, extension_band_extend_, extension_band_check_,
                                    solid_normal_.vec, NULL, NULL,
                                    false, ts_d_.vec, ts_dd_.vec.data());

//  node_neighbors_->second_derivatives_central(tl_.vec, tl_dd_.vec.data());

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_)
  {
  ierr = PetscPrintf(p4est_->mpicomm, "Solving for temperature multiplier... \n"); CHKERRXX(ierr);
  }

  solver_temp_->set_wc(*wall_bc_type_temp_, zero_cf, false);
  solver_temp_->set_rhs(rhs_zero_.vec);
  solver_temp_->set_jc(0, pw_psi_t_sol_jump_taylor_, pw_psi_t_flx_jump_taylor_, pw_psi_t_flx_jump_integr_);

  if (contr_phi_.vec != NULL)
  {
    solver_temp_->set_bc(0, contr_bc_type_temp_, zero_cf, zero_cf);
  }

  vec_and_ptr_t sol(psi_tl_.vec);

  if (poisson_use_nonzero_guess_)
  {
    front_phi_.get_array();
    psi_tl_   .get_array();
    psi_ts_    .get_array();
    sol       .get_array();

    foreach_node(n, nodes_)
    {
      sol.ptr[n] = front_phi_.ptr[n] < 0 ? psi_tl_.ptr[n] : psi_ts_.ptr[n];
    }

    front_phi_.restore_array();
    psi_tl_   .restore_array();
    psi_ts_   .restore_array();
    sol       .restore_array();
  }

  solver_temp_->solve(sol.vec, poisson_use_nonzero_guess_);

  double psi_tl_max = 0;

  liquid_phi_.get_array();
  solid_phi_ .get_array();
  psi_tl_    .get_array();
  psi_ts_    .get_array();
  sol        .get_array();

  foreach_node(n, nodes_)
  {
    if (liquid_phi_.ptr[n] < 0)
    {
      psi_tl_.ptr[n] = sol.ptr[n];
      if (fabs(sol.ptr[n]) > psi_tl_max) psi_tl_max = fabs(sol.ptr[n]);
    }
    if (solid_phi_ .ptr[n] < 0) psi_ts_.ptr[n] = sol.ptr[n];
  }

  liquid_phi_.restore_array();
  solid_phi_ .restore_array();
  psi_tl_    .restore_array();
  psi_ts_    .restore_array();
  sol        .restore_array();

  sol.destroy();

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &psi_tl_max, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_tl_.vec, num_extend_iterations_, 1,
                                    extension_tol_*psi_tl_max, extension_band_use_, extension_band_extend_, extension_band_check_,
                                    liquid_normal_.vec, NULL, NULL,
                                    false, psi_tl_d_.vec);

//  node_neighbors_->second_derivatives_central(psi_t_.vec, psi_t_dd_.vec);

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Max value: %e. \n", psi_tl_max); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Solving for leading concentration... \n"); CHKERRXX(ierr);
  }

  solver_conc_leading_->set_wc(*wall_bc_type_conc_[0], *wall_bc_value_conc_[0], false);
  solver_conc_leading_->set_bc(0, DIRICHLET, pw_c0_values_);
  solver_conc_leading_->set_rhs(rhs_c_[0].vec);

  if (contr_phi_.vec != NULL)
  {
    solver_conc_leading_->set_bc(1, contr_bc_type_conc_[0], *contr_bc_value_conc_[0], zero_cf);
  }

  solver_conc_leading_->solve(c_[0].vec, poisson_use_nonzero_guess_);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[0].vec, num_extend_iterations_, 2,
      extension_tol_, extension_band_use_, extension_band_extend_, extension_band_check_,
      liquid_normal_.vec, NULL, bc,
      false, c_d_[0].vec, c_dd_[0].vec.data());

//  node_neighbors_->second_derivatives_central(c_[0].vec, c_dd_[0].vec.data());

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c0()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c0, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_)
  {
  ierr = PetscPrintf(p4est_->mpicomm, "Solving for leading concentration multiplier... \n"); CHKERRXX(ierr);
  }

  // compute bondary conditions
  seed_map_.get_array();
  psi_tl_  .get_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_       [i].get_array();
    c_dd_    [i].get_array();
    psi_c_   [i].get_array();
  }

  front_normal_   .get_array();
  front_curvature_.get_array();

  int    idx;
  double xyz[P4EST_DIM];
  double normal[P4EST_DIM];
  double conc_term;
  double eps_v;
  double eps_c;
  double kappa;
  interface_point_cartesian_t *pt;

  std::vector<double> c_gamma_all(num_comps_);

  foreach_local_node(n, nodes_)
  {
    for (unsigned short i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0, n); ++i)
    {
      idx = solver_conc_leading_->pw_bc_idx_value_pt(0, n, i);

      solver_conc_leading_->pw_bc_get_boundary_pt(0, idx, pt);
      solver_conc_leading_->pw_bc_xyz_value_pt(0, idx, xyz);

      c_gamma_all[0] = pw_c0_values_[idx];

      XCODE( normal[0] = pt->interpolate(node_neighbors_, front_normal_.ptr[0]) );
      YCODE( normal[1] = pt->interpolate(node_neighbors_, front_normal_.ptr[1]) );
      ZCODE( normal[2] = pt->interpolate(node_neighbors_, front_normal_.ptr[2]) );

      eps_v = eps_v_[seed_map_.ptr[n]]->value(normal);
      eps_c = eps_c_[seed_map_.ptr[n]]->value(normal);

      kappa = pt->interpolate(node_neighbors_, front_curvature_.ptr);

      conc_term = 0;
      for (int i = update_c0_robin_ == 2 ? 0 : 1; i < num_comps_; ++i)
      {
        c_gamma_all[i] = pt->interpolate(node_neighbors_, c_[i].ptr, c_dd_[i].ptr.data());
        conc_term += (1.-part_coeff_[i]) * c_gamma_all[i]
                     * pt->interpolate(node_neighbors_, psi_c_[i].ptr);
      }

      pw_psi_c0_values_[idx] =
          -( conc_term + eps_v
             + latent_heat_*(1.+eps_c*kappa)*pt->interpolate(node_neighbors_, psi_tl_.ptr)
             ) /(1.-part_coeff_[0])/pw_c0_values_[idx];
    }
  }

  seed_map_.restore_array();
  psi_tl_  .restore_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_       [i].restore_array();
    c_dd_    [i].restore_array();
    psi_c_   [i].restore_array();
  }

  front_normal_   .restore_array();
  front_curvature_.restore_array();

  solver_conc_leading_->set_wc(*wall_bc_type_conc_[0], zero_cf, false);
  solver_conc_leading_->set_rhs(rhs_zero_.vec);
  solver_conc_leading_->set_bc(0, DIRICHLET, pw_psi_c0_values_);

  if (contr_phi_.vec != NULL)
  {
    solver_conc_leading_->set_bc(1, contr_bc_type_conc_[0], zero_cf, zero_cf);
  }

  solver_conc_leading_->solve(psi_c_[0].vec, poisson_use_nonzero_guess_);

  double psi_max = 0;
  psi_c_[0].get_array();
  liquid_phi_.get_array();

  foreach_local_node(n, nodes_)
  {
    if (liquid_phi_.ptr[n] < 0)
      if (fabs(psi_c_[0].ptr[n]) > psi_max)
        psi_max = fabs(psi_c_[0].ptr[n]);
  }

  psi_c_[0].restore_array();
  liquid_phi_.restore_array();

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &psi_max, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;

  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[0].vec, num_extend_iterations_, 1,
      extension_tol_*psi_max, extension_band_use_, extension_band_extend_, extension_band_check_,
      liquid_normal_.vec, NULL, bc,
      false, psi_c_d_[0].vec);

//  node_neighbors_->second_derivatives_central(psi_c_[0].vec, psi_c_dd_[0].vec);

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Max value: %e. \n", psi_max); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_psi_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c(int start, int num)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_c1, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_)
  {
  ierr = PetscPrintf(p4est_->mpicomm, "Solve for concentrations... \n"); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  for (int i = start; i < start+num; ++i)
  {
    solver_conc_[i]->set_wc(*wall_bc_type_conc_[i], *wall_bc_value_conc_[i], false);
    solver_conc_[i]->set_bc(0, ROBIN, pw_c_values_[i], pw_c_values_robin_[i], pw_c_coeffs_robin_[i]);
    solver_conc_[i]->set_rhs(rhs_c_[i].vec);

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->set_bc(1, contr_bc_type_conc_[i], *contr_bc_value_conc_[i], zero_cf);
    }

    solver_conc_[i]->solve(c_[i].vec, poisson_use_nonzero_guess_);

    Vec mask = solver_conc_[i]->get_mask();

    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[i].vec, num_extend_iterations_, 2,
                                      extension_tol_, extension_band_use_, extension_band_extend_, extension_band_check_,
                                      liquid_normal_.vec, mask, NULL,
                                      false, c_d_[i].vec, c_dd_[i].vec.data());

//    node_neighbors_->second_derivatives_central(c_[i].vec, c_dd_[i].vec.data());
  }

  if (verbose_)
  {
  ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c(int start, int num)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_)
  {
  ierr = PetscPrintf(p4est_->mpicomm, "Solve for concentrations multipliers... \n"); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_show_convergence(verbose_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  for (int i = start; i < start+num; ++i)
  {
    solver_conc_[i]->set_wc(*wall_bc_type_conc_[i], zero_cf, false);
    solver_conc_[i]->set_bc(0, ROBIN, pw_psi_c_values_[i], pw_psi_c_values_robin_[i], pw_c_coeffs_robin_[i]);
    solver_conc_[i]->set_new_submat_robin(false);
    solver_conc_[i]->set_rhs(rhs_zero_.vec);

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->set_bc(1, contr_bc_type_conc_[i], zero_cf, zero_cf);
    }

    solver_conc_[i]->solve(psi_c_[i].vec, poisson_use_nonzero_guess_);

    Vec mask = solver_conc_[i]->get_mask();

    double psi_max = 0;
    psi_c_[i].get_array();
    liquid_phi_.get_array();

    foreach_local_node(n, nodes_)
    {
      if (liquid_phi_.ptr[n] < 0)
        if (fabs(psi_c_[i].ptr[n]) > psi_max)
          psi_max = fabs(psi_c_[i].ptr[n]);
    }

    psi_c_[i].restore_array();
    liquid_phi_.restore_array();

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &psi_max, 1, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[i].vec, num_extend_iterations_, 1,
                                      extension_tol_*psi_max, extension_band_use_, extension_band_extend_, extension_band_check_,
                                      liquid_normal_.vec, mask, NULL,
                                      false, psi_c_d_[i].vec);

//    node_neighbors_->second_derivatives_central(psi_c_[i].vec, psi_c_dd_[i].vec);

    if (verbose_)
    {
      ierr = PetscPrintf(p4est_->mpicomm, "Max value: %e. \n", psi_max); CHKERRXX(ierr);
    }
  }

  if (verbose_)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::compute_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);

//  if (c0n_.vec != NULL) { ierr = VecDestroy(c0n_.vec); CHKERRXX(ierr); }
//  ierr = VecDuplicate(front_phi_, &c0n_.vec); CHKERRXX(ierr);

//  c0_.get_array();
//  c0d_.get_array();
//  c0n_.get_array();

//  double *front_normal_ptr[P4EST_DIM];
//  foreach_dimension(dim)
//  {
//    ierr = VecGetArray(front_normal_[dim], &front_normal_ptr[dim]); CHKERRXX(ierr);
//  }

//  quad_neighbor_nodes_of_node_t qnnn;

//  double dxyz[P4EST_DIM];

//  dxyz_min(p4est_, dxyz);

//#ifdef P4_TO_P8
//  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
//#else
//  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
//#endif

//  double rel_thresh = 1.e-2;

//  vec_and_ptr_t mask;

//  ierr = VecDuplicate(front_phi_, &mask.vec); CHKERRXX(ierr);

//  copy_ghosted_vec(front_phi_, mask.vec);

//  mask.get_array();

//  for(size_t i = 0; i < node_neighbors_->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_layer_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);

//    c0d_.ptr[0][n] = qnnn.dx_central(c0_.ptr);
//    c0d_.ptr[1][n] = qnnn.dy_central(c0_.ptr);
//#ifdef P4_TO_P8
//    c0d_.ptr[2][n] = qnnn.dz_central(c0_.ptr);
//#endif

////    // correct near the boundary
////    if (solver_c0->pointwise_bc[n].size() > 0 && use_points_on_interface_)
////    {
////      double d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00;
////      double d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0;
////#ifdef P4_TO_P8
////      double d_00m = qnnn.d_00m, d_00p = qnnn.d_00p;
////#endif

////      // assuming grid is uniform near the interface
////      double q_m00 = qnnn.f_m00_linear(c0_.ptr), q_p00 = qnnn.f_p00_linear(c0_.ptr);
////      double q_0m0 = qnnn.f_0m0_linear(c0_.ptr), q_0p0 = qnnn.f_0p0_linear(c0_.ptr);
////#ifdef P4_TO_P8
////      double q_00m = qnnn.f_00m_linear(c0_.ptr), q_00p = qnnn.f_00p_linear(c0_.ptr);
////#endif
////      double q_000 = c0_.ptr[n];

////      double d_min = diag;
////      for (unsigned int i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
////      {
////        switch (solver_c0->pointwise_bc[n][i].dir)
////        {
////          case 0: d_m00 = solver_c0->pointwise_bc[n][i].dist; q_m00 = solver_c0->pointwise_bc[n][i].value; break;
////          case 1: d_p00 = solver_c0->pointwise_bc[n][i].dist; q_p00 = solver_c0->pointwise_bc[n][i].value; break;
////          case 2: d_0m0 = solver_c0->pointwise_bc[n][i].dist; q_0m0 = solver_c0->pointwise_bc[n][i].value; break;
////          case 3: d_0p0 = solver_c0->pointwise_bc[n][i].dist; q_0p0 = solver_c0->pointwise_bc[n][i].value; break;
////#ifdef P4_TO_P8
////          case 4: d_00m = solver_c0->pointwise_bc[n][i].dist; q_00m = solver_c0->pointwise_bc[n][i].value; break;
////          case 5: d_00p = solver_c0->pointwise_bc[n][i].dist; q_00p = solver_c0->pointwise_bc[n][i].value; break;
////#endif
////        }

////        d_min = MIN(d_min, solver_c0->pointwise_bc[n][i].dist);
////      }

////      if (d_min > rel_thresh*diag && solver_c0->pointwise_bc[n].size() < 3)
////      {
////        c0d_.ptr[0][n] = ((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00);
////        c0d_.ptr[1][n] = ((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0);
////#ifdef P4_TO_P8
////        c0d_.ptr[2][n] = ((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p);
////#endif
////      } else {
////        mask.ptr[n] = 1;
////      }
////    }
//  }

//  if (use_points_on_interface_)
//  {
//    ierr = VecGhostUpdateBegin(mask.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  foreach_dimension(dim)
//  {
//    ierr = VecGhostUpdateBegin(c0d_.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  for(size_t i = 0; i < node_neighbors_->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_local_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);

//    c0d_.ptr[0][n] = qnnn.dx_central(c0_.ptr);
//    c0d_.ptr[1][n] = qnnn.dy_central(c0_.ptr);
//#ifdef P4_TO_P8
//    c0d_.ptr[2][n] = qnnn.dz_central(c0_.ptr);
//#endif

////    // correct near the boundary
////    if (solver_c0->pointwise_bc[n].size() > 0 && use_points_on_interface_)
////    {
////      double d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00;
////      double d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0;
////#ifdef P4_TO_P8
////      double d_00m = qnnn.d_00m, d_00p = qnnn.d_00p;
////#endif

////      // assuming grid is uniform near the interface
////      double q_m00 = qnnn.f_m00_linear(c0_.ptr), q_p00 = qnnn.f_p00_linear(c0_.ptr);
////      double q_0m0 = qnnn.f_0m0_linear(c0_.ptr), q_0p0 = qnnn.f_0p0_linear(c0_.ptr);
////#ifdef P4_TO_P8
////      double q_00m = qnnn.f_00m_linear(c0_.ptr), q_00p = qnnn.f_00p_linear(c0_.ptr);
////#endif
////      double q_000 = c0_.ptr[n];

////      double d_min = diag;
////      for (unsigned int i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
////      {
////        switch (solver_c0->pointwise_bc[n][i].dir)
////        {
////          case 0: d_m00 = solver_c0->pointwise_bc[n][i].dist; q_m00 = solver_c0->pointwise_bc[n][i].value; break;
////          case 1: d_p00 = solver_c0->pointwise_bc[n][i].dist; q_p00 = solver_c0->pointwise_bc[n][i].value; break;
////          case 2: d_0m0 = solver_c0->pointwise_bc[n][i].dist; q_0m0 = solver_c0->pointwise_bc[n][i].value; break;
////          case 3: d_0p0 = solver_c0->pointwise_bc[n][i].dist; q_0p0 = solver_c0->pointwise_bc[n][i].value; break;
////#ifdef P4_TO_P8
////          case 4: d_00m = solver_c0->pointwise_bc[n][i].dist; q_00m = solver_c0->pointwise_bc[n][i].value; break;
////          case 5: d_00p = solver_c0->pointwise_bc[n][i].dist; q_00p = solver_c0->pointwise_bc[n][i].value; break;
////#endif
////        }

////        d_min = MIN(d_min, solver_c0->pointwise_bc[n][i].dist);
////      }

////      if (d_min > rel_thresh*diag && solver_c0->pointwise_bc[n].size() < 3)
////      {
////        c0d_.ptr[0][n] = ((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00);
////        c0d_.ptr[1][n] = ((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0);
////#ifdef P4_TO_P8
////        c0d_.ptr[2][n] = ((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p);
////#endif
////      } else {
////        mask.ptr[n] = 1;
////      }
////    }
//  }

//  if (use_points_on_interface_)
//  {
//    ierr = VecGhostUpdateEnd(mask.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  foreach_dimension(dim)
//  {
//    ierr = VecGhostUpdateEnd(c0d_.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  foreach_node(n, nodes_)
//  {
//#ifdef P4_TO_P8
//    c0n_.ptr[n] = c0d_.ptr[0][n]*front_normal_ptr[0][n] + c0d_.ptr[1][n]*front_normal_ptr[1][n] + c0d_.ptr[2][n]*front_normal_ptr[2][n];
//#else
//    c0n_.ptr[n] = c0d_.ptr[0][n]*front_normal_ptr[0][n] + c0d_.ptr[1][n]*front_normal_ptr[1][n];
//#endif
//  }

//  c0_.restore_array();
//  c0n_.restore_array();
//  c0d_.restore_array();

//  foreach_dimension(dim)
//  {
//    ierr = VecRestoreArray(front_normal_[dim], &front_normal_ptr[dim]); CHKERRXX(ierr);
//  }

//  mask.restore_array();

//  my_p4est_level_set_t ls(node_neighbors_);
//  ls.set_use_one_sided_derivaties(use_one_sided_derivatives_);
//  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

//  foreach_dimension(dim)
//  {
//    ls.extend_Over_Interface_TVD_Full(front_phi_, mask.vec, c0d_.vec[dim], num_extend_iterations_, 2);
//  }

//  ierr = VecDestroy(mask.vec); CHKERRXX(ierr);

//  // compute second derivatives for interpolation purposes
//  for (short dim = 0; dim < P4EST_DIM; ++dim)
//  {
//    if (c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(c0n_dd_.vec[dim]); CHKERRXX(ierr); }
//    ierr = VecDuplicate(front_phi_dd_[dim], &c0n_dd_.vec[dim]); CHKERRXX(ierr);
//  }

//  if (c0n_gamma_.vec != NULL) { ierr = VecDestroy(c0n_gamma_.vec); CHKERRXX(ierr); }

//  ierr = VecDuplicate(front_phi_, &c0n_gamma_.vec); CHKERRXX(ierr);

//  ls.extend_from_interface_to_whole_domain_TVD(front_phi_, c0n_.vec, c0n_gamma_.vec);

//  node_neighbors_->second_derivatives_central(c0n_.vec, c0n_dd_.vec);

//  Vec tmp; ierr = VecDuplicate(front_phi_, &tmp); CHKERRXX(ierr);
//  foreach_dimension(dim)
//  {
//    copy_ghosted_vec(c0d_.vec[dim], tmp);
//    ls.extend_from_interface_to_whole_domain_TVD(front_phi_, tmp, c0d_.vec[dim]);
//  }
//  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  node_neighbors_->first_derivatives_central(c_[0].vec, c0d_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_poisson_nodes_multialloy_t::compute_psi_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);

//  if (psi_c0n_.vec != NULL) { ierr = VecDestroy(psi_c0n_.vec); CHKERRXX(ierr); }
//  ierr = VecDuplicate(front_phi_, &psi_c0n_.vec); CHKERRXX(ierr);

//  psi_c0_.get_array();
//  psi_c0n_.get_array();

//  double *front_normal_ptr[P4EST_DIM];
//  foreach_dimension(dim)
//  {
//    ierr = VecGetArray(front_normal_[dim], &front_normal_ptr[dim]); CHKERRXX(ierr);
//  }

//  quad_neighbor_nodes_of_node_t qnnn;

//  for(size_t i = 0; i < node_neighbors_->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_layer_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);

//#ifdef P4_TO_P8
//    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*front_normal_ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*front_normal_ptr[1][n] + qnnn.dz_central(psi_c0_.ptr)*front_normal_ptr[2][n];
//#else
//    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*front_normal_ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*front_normal_ptr[1][n];
//#endif
//  }

//  ierr = VecGhostUpdateBegin(psi_c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  for(size_t i = 0; i < node_neighbors_->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = node_neighbors_->get_local_node(i);
//    qnnn = node_neighbors_->get_neighbors(n);

//#ifdef P4_TO_P8
//    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*front_normal_ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*front_normal_ptr[1][n] + qnnn.dz_central(psi_c0_.ptr)*front_normal_ptr[2][n];
//#else
//    psi_c0n_.ptr[n] = qnnn.dx_central(psi_c0_.ptr)*front_normal_ptr[0][n] + qnnn.dy_central(psi_c0_.ptr)*front_normal_ptr[1][n];
//#endif
//  }

//  ierr = VecGhostUpdateEnd(psi_c0n_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  psi_c0_.restore_array();
//  psi_c0n_.restore_array();

//  foreach_dimension(dim)
//  {
//    ierr = VecRestoreArray(front_normal_[dim], &front_normal_ptr[dim]); CHKERRXX(ierr);
//  }

//  // compute second derivatives for interpolation purposes
//  foreach_dimension(dim)
//  {
//    if (psi_c0n_dd_.vec[dim] != NULL) { ierr = VecDestroy(psi_c0n_dd_.vec[dim]); CHKERRXX(ierr); }
//    ierr = VecDuplicate(front_phi_dd_[dim], &psi_c0n_dd_.vec[dim]); CHKERRXX(ierr);
//  }

//  node_neighbors_->second_derivatives_central(psi_c0n_.vec, psi_c0n_dd_.vec);
  node_neighbors_->first_derivatives_central(psi_c_[0].vec, psi_c0d_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::compute_pw_bc_values(int start, int num)
{
//  ierr = PetscPrintf(p4est_->mpicomm, "Computing bc values... \n"); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  bool interp_initialized;

  int    idx;
  double vn_pr;
  double vn_cd;
  double kappa_pr;
  double kappa_cd;
  double eps_c_pr;
  double eps_c_cd;
  double xyz_pr[P4EST_DIM];
  double xyz_cd[P4EST_DIM];
  double normal[P4EST_DIM];

  front_normal_   .get_array();
  front_curvature_.get_array();
  seed_map_       .get_array();
  c0d_            .get_array();
  c_[0]           .get_array();
  c_dd_[0]        .get_array();

  foreach_local_node(n, nodes_)
  {
    interp_initialized = false;
    if (solver_temp_->pw_jc_num_taylor_pts(0, n) > 0)
    {
      interp_local.initialize(n);
      interp_initialized = true;

      // projection point
      idx = solver_temp_->pw_jc_idx_taylor_pt(0, n, 0);
      solver_temp_->pw_jc_xyz_taylor_pt(0, idx, xyz_pr);

      vn_pr = 0;
      foreach_dimension(dim)
      {
        interp_local.set_input(front_normal_.ptr[dim], linear); normal[dim] = interp_local.value(xyz_pr);
        interp_local.set_input(c0d_.ptr[dim],          linear); vn_pr += normal[dim]*interp_local.value(xyz_pr);
      }

      interp_local.set_input(c_[0].ptr, DIM(c_dd_[0].ptr[0], c_dd_[0].ptr[1], c_dd_[0].ptr[2]), quadratic_non_oscillatory_continuous_v2);
      vn_pr = (conc_diff_[0]*vn_pr - front_conc_flux_[0]->value(xyz_pr))/interp_local.value(xyz_pr)/(1.0-part_coeff_[0]);

      interp_local.set_input(front_curvature_.vec, linear);
      kappa_pr = interp_local.value(xyz_pr);

      eps_c_pr = 0.*eps_c_[seed_map_.ptr[n]]->value(normal);

//      vn_pr = vn_exact_->value(xyz_pr);

      pw_t_sol_jump_taylor_[idx] = front_temp_value_jump_->value(xyz_pr);
      pw_t_flx_jump_taylor_[idx] = front_temp_flux_jump_->value(xyz_pr) - latent_heat_*(1.0+eps_c_pr*kappa_pr)*vn_pr;

      // centroid
      idx = solver_temp_->pw_jc_idx_integr_pt(0, n, 0);
      solver_temp_->pw_jc_xyz_integr_pt(0, idx, xyz_cd);

      vn_cd = 0;
      foreach_dimension(dim)
      {
        interp_local.set_input(front_normal_.ptr[dim], linear);
        normal[dim] = interp_local.value(xyz_cd);

        interp_local.set_input(c0d_.ptr[dim],          linear);
        vn_cd += normal[dim]*interp_local.value(xyz_cd);
      }

      interp_local.set_input(c_[0].ptr, DIM(c_dd_[0].ptr[0], c_dd_[0].ptr[1], c_dd_[0].ptr[2]), quadratic_non_oscillatory_continuous_v2);
      vn_cd = (conc_diff_[0]*vn_cd - front_conc_flux_[0]->value(xyz_cd))/interp_local.value(xyz_cd)/(1.0-part_coeff_[0]);

//      vn_cd = vn_exact_->value(xyz_cd);

      interp_local.set_input(front_curvature_.vec, linear);
      kappa_cd = interp_local.value(xyz_cd);

      eps_c_cd = 0.*eps_c_[seed_map_.ptr[n]]->value(normal);

      pw_t_flx_jump_integr_[idx] = front_temp_flux_jump_->value(xyz_cd) - latent_heat_*(1.0+eps_c_cd*kappa_cd)*vn_cd;
    }

    for (int i = start; i < start+num; ++i)
    {
      if (solver_conc_[i]->pw_bc_num_value_pts(0,n) > 0)
      {
        if (!interp_initialized)
        {
          interp_local.initialize(n);
          interp_initialized = true;
        }

        idx = solver_conc_[i]->pw_bc_idx_value_pt(0, n, 0);
        solver_conc_[i]->pw_bc_xyz_value_pt(0, idx, xyz_cd);
        pw_c_values_[i][idx] = front_conc_flux_[i]->value(xyz_cd);

        idx = solver_conc_[i]->pw_bc_idx_robin_pt(0, n, 0);
        solver_conc_[i]->pw_bc_xyz_robin_pt(0, idx, xyz_cd);

        vn_cd = 0;
        foreach_dimension(dim)
        {
          interp_local.set_input(front_normal_.ptr[dim], linear); normal[dim] = interp_local.value(xyz_cd);
          interp_local.set_input(c0d_.ptr[dim],          linear); vn_cd += normal[dim]*interp_local.value(xyz_cd);
        }


        interp_local.set_input(c_[0].ptr, DIM(c_dd_[0].ptr[0], c_dd_[0].ptr[1], c_dd_[0].ptr[2]), quadratic_non_oscillatory_continuous_v2);
        vn_cd = (conc_diff_[0]*vn_cd - front_conc_flux_[0]->value(xyz_cd))/interp_local.value(xyz_cd)/(1.0-part_coeff_[0]);

//        vn_cd = vn_exact_->value(xyz_cd);
        pw_c_values_robin_[i][idx] = front_conc_flux_[i]->value(xyz_cd);
        pw_c_coeffs_robin_[i][idx] = -(1.0-part_coeff_[i])*vn_cd;
      }
    }
  }

  front_normal_   .restore_array();
  front_curvature_.restore_array();
  seed_map_       .restore_array();
  c0d_            .restore_array();
  c_[0]           .restore_array();
  c_dd_[0]        .restore_array();

//  MPI_Barrier(p4est_->mpicomm);
//  ierr = PetscPrintf(p4est_->mpicomm, "Done.\n"); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::compute_pw_bc_psi_values(int start, int num)
{
//  ierr = PetscPrintf(p4est_->mpicomm, "Computing bc psi values... \n"); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  bool interp_initialized;

  int    idx;
  double vn_pr;
  double vn_cd;
  double kappa_pr;
  double kappa_cd;
  double eps_c_pr;
  double eps_c_cd;
  double xyz_pr[P4EST_DIM];
  double xyz_cd[P4EST_DIM];
  double normal[P4EST_DIM];

  vector<double> c_all(num_comps_);

  front_normal_   .get_array();
  front_curvature_.get_array();
  seed_map_       .get_array();
  c0d_            .get_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].get_array();
    c_dd_[i].get_array();
  }

  foreach_local_node(n, nodes_)
  {
    interp_initialized = false;
    if (solver_temp_->pw_jc_num_taylor_pts(0, n) > 0)
    {
//      interp_local.initialize(n);
//      interp_initialized = true;

      // projection point
      idx = solver_temp_->pw_jc_idx_taylor_pt(0, n, 0);
      solver_temp_->pw_jc_xyz_taylor_pt(0, idx, xyz_pr);

//      vn_pr = 0;
//      foreach_dimension(dim)
//      {
//        interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz_pr);
//        interp_local.set_input(c0d_.vec[dim],          linear); vn_pr += normal[dim]*interp_local.value(xyz_pr);
//      }

//      interp_local.set_input(c_[0].vec, DIM(c_dd_[0].vec[0], c_dd_[0].vec[1], c_dd_[0].vec[2]), quadratic_non_oscillatory_continuous_v2);
//      vn_pr = (front_conc_flux_[0]->value(xyz_pr) - conc_diff_[0]*vn_pr)/interp_local.value(xyz_pr)/(1.0-part_coeff_[0]);

//      interp_local.set_input(front_curvature_.vec, linear);
//      kappa_pr = interp_local.value(xyz_pr);

//      eps_c_pr = eps_c_[seed_map_.ptr[n]]->value(normal);

      pw_psi_t_sol_jump_taylor_[idx] = 0;
      pw_psi_t_flx_jump_taylor_[idx] = 1;

      // centroid
      idx = solver_temp_->pw_jc_idx_integr_pt(0, n, 0);
      solver_temp_->pw_jc_xyz_integr_pt(0, idx, xyz_cd);

//      vn_cd = 0;
//      foreach_dimension(dim)
//      {
//        interp_local.set_input(front_normal_.vec[dim], linear);
//        normal[dim] = interp_local.value(xyz_cd);

//        interp_local.set_input(c0d_.vec[dim],          linear);
//        vn_cd += normal[dim]*interp_local.value(xyz_cd);
//      }

//      interp_local.set_input(c_[0].vec, DIM(c_dd_[0].vec[0], c_dd_[0].vec[1], c_dd_[0].vec[2]), quadratic_non_oscillatory_continuous_v2);
//      vn_cd = (front_conc_flux_[0]->value(xyz_cd) - conc_diff_[0]*vn_cd)/interp_local.value(xyz_cd)/(1.0-part_coeff_[0]);

//      interp_local.set_input(front_curvature_.vec, linear);
//      kappa_cd = interp_local.value(xyz_cd);

//      eps_c_cd = eps_c_[seed_map_.ptr[n]]->value(normal);

      pw_psi_t_flx_jump_integr_[idx] = 1;
    }

    if (num_comps_ > start)
    {
      if (solver_conc_[1]->pw_bc_num_value_pts(0, n) > 0)
      {
        if (!interp_initialized)
        {
          interp_local.initialize(n);
          interp_initialized = true;
        }

        idx = solver_conc_[1]->pw_bc_idx_robin_pt(0, n, 0);
        solver_conc_[1]->pw_bc_xyz_robin_pt(0, idx, xyz_cd);

        for (int i = 0; i < num_comps_; ++i)
        {
          interp_local.set_input(c_[i].vec, DIM(c_dd_[i].vec[0], c_dd_[i].vec[1], c_dd_[i].vec[2]), quadratic_non_oscillatory_continuous_v2);
          c_all[i] = interp_local.value(xyz_cd);
        }
      }
    }

    for (int i = start; i < start+num; ++i)
    {
      if (solver_conc_[i]->pw_bc_num_value_pts(0, n) > 0)
      {
        idx = solver_conc_[i]->pw_bc_idx_value_pt(0, n, 0);
        solver_conc_[i]->pw_bc_xyz_value_pt(0, idx, xyz_cd);
        pw_psi_c_values_[i][idx] = liquidus_slope_(i, c_all.data());

        idx = solver_conc_[i]->pw_bc_idx_robin_pt(0, n, 0);
        solver_conc_[i]->pw_bc_xyz_robin_pt(0, idx, xyz_cd);

//        vn_cd = 0;
//        foreach_dimension(dim)
//        {
//          interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz_cd);
//          interp_local.set_input(c0d_.vec[dim],          linear); vn_cd += normal[dim]*interp_local.value(xyz_cd);
//        }

//        interp_local.set_input(c_[0].vec, DIM(c_dd_[0].vec[0], c_dd_[0].vec[1], c_dd_[0].vec[2]), quadratic_non_oscillatory_continuous_v2);
//        vn_cd = (front_conc_flux_[0]->value(xyz_cd) - conc_diff_[0]*vn_cd)/interp_local.value(xyz_cd)/(1.0-part_coeff_[0]);

        pw_psi_c_values_robin_[i][idx] = liquidus_slope_(i, c_all.data());
//        pw_psi_c_coeffs_robin_[idx] = (1.0-part_coeff_[i])*vn_cd;
      }
    }
  }

  front_normal_   .restore_array();
  front_curvature_.restore_array();
  seed_map_       .restore_array();
  c0d_            .restore_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].restore_array();
    c_dd_[i].restore_array();
  }

//  MPI_Barrier(p4est_->mpicomm);
//  ierr = PetscPrintf(p4est_->mpicomm, "Done.\n"); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::adjust_c0_gamma(bool simple)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);

  /* get pointers */
  tl_      .get_array();
  tl_dd_   .get_array();
  bc_error_.get_array();

  c0d_     .get_array();
  psi_c0d_ .get_array();

  seed_map_       .get_array();
  front_normal_   .get_array();
  front_curvature_.get_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].get_array();
    c_dd_[i].get_array();
  }

  /* main loop */
  bc_error_max_ = 0;
  velo_max_     = 0;

  int    idx;
  double vn;
  double tl_val;
  double eps_c;
  double eps_v;
  double kappa;
  double error;
  double change;
  double psi_c0;
  double psi_c0n;
  double xyz   [P4EST_DIM];
  double normal[P4EST_DIM];
  std::vector<double> c_all(num_comps_);
  interface_point_cartesian_t *pt;

  foreach_local_node(n, nodes_)
  {
    bc_error_.ptr[n] = 0;

    for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0, n); ++i)
    {
      idx = solver_conc_leading_->pw_bc_idx_value_pt(0, n, i);

      // fetch boundary point
      solver_conc_leading_->pw_bc_xyz_value_pt(0, idx, xyz);
      solver_conc_leading_->pw_bc_get_boundary_pt(0, idx, pt);

      // interpolate concentration
      c_all[0] = pw_c0_values_[idx];
      for (int k = update_c0_robin_ == 2 ? 0 : 1; k < num_comps_; ++k)
      {
        c_all[k] = pt->interpolate(node_neighbors_, c_[k].ptr, c_dd_[k].ptr.data());
//        c_all[k] = pt->interpolate(node_neighbors_, c_[k].ptr);
//        c_all[k] = c_[k].ptr[n];
      }

      // interpolate temperature
      tl_val = pt->interpolate(node_neighbors_, tl_.ptr, tl_dd_.ptr.data());
//      tl_val = pt->interpolate(node_neighbors_, tl_.ptr);
//      tl_val = tl_.ptr[n];

      // normal velocity
      vn = 0;
      foreach_dimension(dim)
      {
        normal[dim] = pt->interpolate(node_neighbors_, front_normal_.ptr[dim]);
        vn += pt->interpolate(node_neighbors_, c0d_.ptr[dim])*normal[dim];
      }
      vn = ( (conc_diff_[0]*vn - front_conc_flux_[0]->value(xyz))/(1.-part_coeff_[0])/pw_c0_values_[idx] );

      // curvature
      kappa = pt->interpolate(node_neighbors_, front_curvature_.ptr);

      // undercoolings
      eps_v = eps_v_[seed_map_.ptr[n]]->value(normal);
      eps_c = eps_c_[seed_map_.ptr[n]]->value(normal);

      //if (xyz[1] > 0.12)
      //{
      //  eps_v += 1;
      //  eps_v -= 1;
      //}

      // error
      error = tl_val
              - melting_temp_*(1.0 + eps_c*kappa)
              - liquidus_value_(c_all.data())
              - eps_v*vn
              - gibbs_thomson_->value(xyz);

      bc_error_.ptr[n] = MAX(bc_error_.ptr[n], fabs(error));
      bc_error_max_    = MAX(bc_error_max_,    fabs(error));

      change = error;

      //        if (var_scheme_ == ABS_ALTER)
      //        {
      //          if (use_neg) change = MIN(change, 0.);
      //          else         change = MAX(change, 0.);

      ////          change *= 1.0;
      //        }

      //        double factor = 1;

      //        if (var_scheme_ == ABS_VALUE) factor = change < 0 ? -1 : 1;
      //        if (var_scheme_ == ABS_SMTH1) factor = 2.*exp(change/err_eps_)/(1.+exp(change/err_eps_)) - 1.;
      //        if (var_scheme_ == ABS_SMTH2) factor = change/sqrt(change*change + err_eps_*err_eps_);
      //        if (var_scheme_ == ABS_SMTH3) factor = change/sqrt(change*change + err_eps_*err_eps_) + err_eps_*err_eps_*change/pow(change*change + err_eps_*err_eps_, 1.5);
      //        if (var_scheme_ == QUADRATIC) factor = change;

      //        if (var_scheme_ == ABS_VALUE) change = fabs(change);
      //        if (var_scheme_ == ABS_SMTH1) change = (2.*err_eps_*log(0.5*(1.+exp(change/err_eps_))) - change);
      //        if (var_scheme_ == ABS_SMTH2) change = (sqrt(change*change + err_eps_*err_eps_));
      //        if (var_scheme_ == ABS_SMTH3) change = (sqrt(change*change + err_eps_*err_eps_) - err_eps_*err_eps_/sqrt(change*change + err_eps_*err_eps_));
      ////        if (var_scheme_ == ABS_SMTH3) change = fabs(change);
      //        if (var_scheme_ == QUADRATIC) change = pow(change, 2.);

      if (simple)
      {
        change /= -liquidus_slope_(0, c_all.data());
      }
      else
      {
        psi_c0  = pw_psi_c0_values_[idx];
        psi_c0n = 0;
        foreach_dimension(dim)
        {
          psi_c0n += pt->interpolate(node_neighbors_, psi_c0d_.ptr[dim])*normal[dim];
        }
        change /= - (update_c0_robin_ == 2 ? 0 : liquidus_slope_(0, c_all.data())) + conc_diff_[0]*psi_c0n - (1.-part_coeff_[0])*vn*psi_c0;
      }

      pw_c0_values_[idx] -= change;
//      pw_c0_values_[idx] = c_all[0] - change;

      velo_max_ = MAX(velo_max_, fabs(vn));
    }
  }

  double buffer[2] = { bc_error_max_, velo_max_ };
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, buffer, 2, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  bc_error_max_ = buffer[0];
  velo_max_     = buffer[1];

  /* restore pointers */
  tl_      .restore_array();
  tl_dd_   .restore_array();
  bc_error_.restore_array();

  c0d_     .restore_array();
  psi_c0d_ .restore_array();

  seed_map_       .restore_array();
  front_normal_   .restore_array();
  front_curvature_.restore_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].restore_array();
    c_dd_[i].restore_array();
  }

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}

//void my_p4est_poisson_nodes_multialloy_t::compute_bc_error()
//{
//  // allocate memory
//  if (bc_error_gamma_.vec != NULL) { ierr = VecDestroy(bc_error_gamma_.vec); CHKERRXX(ierr); }
//  ierr = VecDuplicate(front_phi_, &bc_error_gamma_.vec); CHKERRXX(ierr);

//  // extend all quantities from interface in the normal direction
//  vec_and_ptr_t c1_gamma;
//  vec_and_ptr_t tl_gamma;

//  ierr = VecDuplicate(front_phi_, &c1_gamma.vec); CHKERRXX(ierr);
//  ierr = VecDuplicate(front_phi_, &tl_gamma.vec); CHKERRXX(ierr);

//  my_p4est_level_set_t ls(node_neighbors_);

//  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

//  ls.extend_from_interface_to_whole_domain_TVD(front_phi_, c1_.vec, c1_gamma.vec);
//  ls.extend_from_interface_to_whole_domain_TVD(front_phi_, tl_.vec, tl_gamma.vec);

//  bc_error_gamma_.get_array();
//  c0_gamma_.get_array();
//  c1_gamma.get_array();
//  tl_gamma.get_array();

//  double xyz[P4EST_DIM];
//  foreach_node(n, nodes_)
//  {
//    node_xyz_fr_n(n, p4est_, nodes_, xyz);
//    bc_error_gamma_.ptr[n] = c0_gamma_.ptr[n] + (c1_gamma.ptr[n]*ml1_ + tl_ - tl_gamma.ptr[n] + GT_->value(xyz))/ml0_;
//  }

//  bc_error_gamma_.restore_array();
//  c0_gamma_.restore_array();
//  c1_gamma.restore_array();
//  tl_gamma.restore_array();

//  ierr = VecDestroy(c1_gamma.vec); CHKERRXX(ierr);
//  ierr = VecDestroy(tl_gamma.vec); CHKERRXX(ierr);
//}
