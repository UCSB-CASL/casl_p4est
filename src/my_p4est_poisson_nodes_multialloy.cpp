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
    interp_(node_neighbors), interp_bc_points(node_neighbors)
{
//  psi_c1_interface_value_.set_ptr(this);
//  jump_psi_tn_  .set_ptr(this);
//  vn_from_c0_   .set_ptr(this);
//  c0_robin_coef_.set_ptr(this);
//  c1_robin_coef_.set_ptr(this);
//  tn_jump_      .set_ptr(this);
//  bc_error_cf_  .set_ptr(this);

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
  bc_error_max_ = 1.;
  bc_error_avg_ = 1.;

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
  front_conc_flux_    .resize(num_comps_, NULL);
  wall_bc_value_conc_ .resize(num_comps_, NULL);
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
  update_c0_robin_           = 0;
  use_points_on_interface_   = true;
  verbose_                   = false;

  liquidus_value_ = NULL;
  liquidus_slope_ = NULL;
  part_coeff_     = NULL;

  volume_thresh_ = 1.e-2;
  err_eps_ = 1.e-5;

  num_extend_iterations_ = 50;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est_, dxyz);

  min_volume_ = MULTD(dxyz[0], dxyz[1], dxyz[2]);

  extension_band_use_    = 8.*pow(min_volume_, 1./ double(P4EST_DIM));
  extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM));
  extension_use_nonzero_guess_ = true;

  poisson_use_nonzero_guess_ = true;

  iteration_scheme_ = 2;
  dirichlet_scheme_ = 2;
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

int my_p4est_poisson_nodes_multialloy_t::solve(Vec tl, Vec ts, Vec c[], Vec c0d[], bool use_non_zero_guess,
                                               Vec bc_error, double *bc_error_max, double *bc_error_avg,
                                               std::vector<int> *num_pdes, std::vector<double> *bc_error_max_all, std::vector<double> *bc_error_avg_all,
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
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, tl_.vec, 0, 1, extension_band_use_, extension_band_extend_, liquid_normal_.vec, NULL, NULL, 1, tl_d_.vec, tl_dd_.vec.data());
  ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  ts_.vec, 0, 1, extension_band_use_, extension_band_extend_, solid_normal_.vec,  NULL, NULL, 1, ts_d_.vec, ts_dd_.vec.data());
  for (int i = 0; i < num_comps_; ++i)
  {
    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[i].vec, 0, 1, extension_band_use_, extension_band_extend_, liquid_normal_.vec, NULL, NULL, 1, c_d_[i].vec, c_dd_[i].vec.data());
  }

  if (psi_tl != NULL) ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_tl_.vec, 0, 1, extension_band_use_, extension_band_extend_, liquid_normal_.vec, NULL, NULL, 1, psi_tl_d_.vec, NULL);
  if (psi_ts != NULL) ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  psi_ts_.vec, 0, 1, extension_band_use_, extension_band_extend_, solid_normal_.vec,  NULL, NULL, 1, psi_ts_d_.vec, NULL);

  if (psi_cl != NULL)
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[i].vec, 0, 1, extension_band_use_, extension_band_extend_, liquid_normal_.vec, NULL, NULL, 1, psi_c_d_[i].vec);
    }
  }

  // for logging purposes
  if (num_pdes != NULL) num_pdes->clear();
  if (bc_error_max_all != NULL) bc_error_max_all->clear();
  if (bc_error_avg_all != NULL) bc_error_avg_all->clear();
  int num_pdes_solved = 0;

  initialize_solvers();

  int  iteration = 0;
  bc_error_max_  = DBL_MAX;

  int conc_start = update_c0_robin_ == 2 ? 0 : 1;
  int conc_num   = num_comps_ - conc_start;

  while (bc_error_max_ > bc_tolerance_ &&
         iteration < max_iterations_)
  {
    ++iteration;

    // solve for physical quantities
//    PetscPrintf(p4est_->mpicomm, "Solving c0 ... \n");
    solve_c0(); ++num_pdes_solved;

//    PetscPrintf(p4est_->mpicomm, "Computing c0n ... \n");
    compute_c0n();

//    PetscPrintf(p4est_->mpicomm, "Computing pw bc values ... \n");


    compute_pw_bc_values(conc_start, conc_num);
//    PetscPrintf(p4est_->mpicomm, "Solving t ... \n");
    solve_t();  ++num_pdes_solved;
//    PetscPrintf(p4est_->mpicomm, "Solving c ... \n");
    solve_c(conc_start, conc_num);  ++num_pdes_solved;

//    PetscPrintf(p4est_->mpicomm, "Solving lagrange multipliers ... \n");

    // solve for lagrangian multipliers
    if ((iteration-1)%1 == 0)
    {
      switch (iteration_scheme_)
      {
        case 0:
          // do nothing
        break;

        case 1:
          compute_pw_bc_psi_values(conc_start, conc_num);
          if (iteration == 1)
          {
            solve_psi_t(); num_pdes_solved += 1;
          }
          solve_psi_c(conc_start, conc_num); num_pdes_solved += conc_num;
          solve_psi_c0(1);                   num_pdes_solved += 1;
          compute_psi_c0n();
        break;

        case 2:
          if (iteration == 1)
          {
            solve_psi_c0(2);  ++num_pdes_solved;
            compute_psi_c0n();
          }
          compute_pw_bc_psi_values(conc_start, conc_num);
          solve_psi_c(conc_start, conc_num); num_pdes_solved += conc_num;
          solve_psi_t();                     num_pdes_solved += 1;
        break;

        case 3:
        {
          //        if (iteration == 1)
          //        {
          solve_psi_c0(2);  ++num_pdes_solved;
          compute_psi_c0n();
          //        }
          compute_pw_bc_psi_values(conc_start, conc_num);
          solve_psi_c(conc_start, conc_num); num_pdes_solved += conc_num;
          solve_psi_t();                     num_pdes_solved += 1;

          compute_c0_change(2);

          bool poisson_use_nonzero_guess_tmp = poisson_use_nonzero_guess_;
          poisson_use_nonzero_guess_ = 0;

          solve_psi_c0(3);  ++num_pdes_solved;
          compute_psi_c0n();
          compute_pw_bc_psi_values(conc_start, conc_num);
          solve_psi_c(conc_start, conc_num); num_pdes_solved += conc_num;
          solve_psi_t();                     num_pdes_solved += 1;

          poisson_use_nonzero_guess_ = poisson_use_nonzero_guess_tmp;

          break;
        }

        default:
          throw;
      }
    }

//    MPI_Barrier(p4est_->mpicomm);
//    PetscPrintf(p4est_->mpicomm, "Adjusting c0 boundary conditions ... \n");

    // adjust boundary conditions
    compute_c0_change(iteration_scheme_);

//    MPI_Barrier(p4est_->mpicomm);
//    PetscPrintf(p4est_->mpicomm, "Updating pw c0 values ... \n");


    for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0); ++i) {
      pw_c0_values_[i] -= pw_c0_change_[i];
    }
//    MPI_Barrier(p4est_->mpicomm);
//    PetscPrintf(p4est_->mpicomm, "Done. \n");


    // logging for convergence studies
    if (num_pdes != NULL) { num_pdes->push_back(num_pdes_solved); }
    if (bc_error_max_all != NULL) { bc_error_max_all->push_back(bc_error_max_); }
    if (bc_error_avg_all != NULL) { bc_error_avg_all->push_back(bc_error_avg_); }

    ierr = PetscPrintf(p4est_->mpicomm, "Iteration %d: max bc error = %2.3e, avg bc error = %2.3e, max velo = %2.5e\n", iteration, bc_error_max_, bc_error_avg_, velo_max_); CHKERRXX(ierr);
  }

  if (update_c0_robin_ == 1)
  {
    compute_pw_bc_values(0, 1);
    solve_c(0, 1);
  }

  if (psi_ts != NULL)
  {
    ls.extend_Over_Interface_TVD_Full(solid_phi_.vec, psi_ts_.vec, num_extend_iterations_, 1,
                                      extension_band_use_, extension_band_extend_,
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

  if (bc_error_max != NULL) *bc_error_max = bc_error_max_;
  if (bc_error_avg != NULL) *bc_error_avg = bc_error_avg_;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  return iteration;
}



void my_p4est_poisson_nodes_multialloy_t::initialize_solvers()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);

  rhs_zero_.destroy();
  rhs_zero_.create(rhs_tl_.vec);
  VecSetGhost(rhs_zero_.vec, 0);

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
//  solver_conc_leading_->set_wc(wall_bc_type_conc_, *wall_bc_value_conc_[0]);
  solver_conc_leading_->set_wc(*wall_bc_type_conc_, *wall_bc_value_conc_[0]);

  solver_conc_leading_->set_rhs(rhs_c_[0].vec);
  solver_conc_leading_->set_store_finite_volumes(1);
  solver_conc_leading_->set_cube_refinement(cube_refinement_);
  solver_conc_leading_->set_use_sc_scheme(use_superconvergent_robin_);
  solver_conc_leading_->set_integration_order(integration_order_);

  switch (dirichlet_scheme_) {
    case 0:
      solver_conc_leading_->set_dirichlet_scheme(0);
    break;
    case 1:
      solver_conc_leading_->set_dirichlet_scheme(1);
      solver_conc_leading_->set_gf_order(2);
      solver_conc_leading_->set_gf_thresh(-0.1);
      solver_conc_leading_->set_gf_stabilized(2);
    break;
    case 2:
      solver_conc_leading_->set_dirichlet_scheme(2);
    break;
    default:
      throw;
  }

  if (contr_phi_.vec != NULL)
  {
    solver_conc_leading_->add_boundary(MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_conc_, zero_cf, zero_cf);
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
//  solver_temp_->set_wc(wall_bc_type_temp_, *wall_bc_value_temp_);
  solver_temp_->set_wc(*wall_bc_type_temp_, *wall_bc_value_temp_);

  solver_temp_->set_rhs(rhs_tl_.vec, rhs_ts_.vec);

  if (contr_phi_.vec != NULL)
  {
    solver_temp_->add_boundary (MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_temp_, *contr_bc_value_temp_, zero_cf);
  }

  solver_temp_->preassemble_linear_system();

  // copy finite volumes
  std::vector<my_p4est_finite_volume_t> *fvs = NULL;
  std::vector<int> *fvs_map = NULL;

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
//    solver_conc_[i]->set_wc(wall_bc_type_conc_, *wall_bc_value_conc_[i]);
    solver_conc_[i]->set_wc(*wall_bc_type_conc_, *wall_bc_value_conc_[i]);

    if (i != i_start)
    {
      P4EST_ASSERT(fvs != NULL && fvs_map != NULL);
      solver_conc_[i]->set_finite_volumes(fvs, fvs_map, NULL, NULL);
    }

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->add_boundary(MLS_INTERSECTION, contr_phi_.vec, contr_phi_dd_.vec, contr_bc_type_conc_, zero_cf, zero_cf);
    }

    solver_conc_[i]->preassemble_linear_system();

    if (i == i_start)
    {
      solver_conc_[i]->get_boundary_finite_volumes(fvs, fvs_map);
      P4EST_ASSERT(fvs != NULL && fvs_map != NULL);
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

  pw_inverse_gradient_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);
  pw_c0_change_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);
  pw_c0_values_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);
  pw_psi_c0_values_.resize(solver_conc_leading_->pw_bc_num_value_pts(0), 0);

//  // sample c0 guess at boundary points
//  double xyz[P4EST_DIM];
//  for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0); ++i) {
//    solver_conc_leading_->pw_bc_xyz_value_pt(0, i, xyz);
//    pw_c0_values_[i] = c0_guess_->value(xyz);
//  }

  // initialize bc points interpolator
  double xyz[P4EST_DIM];
  interp_bc_points.clear();
  for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0); ++i) {
    solver_conc_leading_->pw_bc_xyz_value_pt(0, i, xyz);
    interp_bc_points.add_point(i, xyz);
  }

  // sample c0 guess at boundary points
  interp_bc_points.set_input(c0_guess_, linear);
  interp_bc_points.interpolate(pw_c0_values_.data());

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_initialize_solvers, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_) {
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

//  solver_temp_->set_wc(wall_bc_type_temp_, *wall_bc_value_temp_, false);
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


  if(0){
    PetscPrintf(p4est_->mpicomm, "\n \n Saving fields after T solution \n");
    // -------------------------------
    // TEMPORARY: save fields before grid update
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};
    point_fields.push_back(Vec_for_vtk_export_t(liquid_phi_.vec, "phi_l"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_.vec, "Tl"));
    point_fields.push_back(Vec_for_vtk_export_t(ts_.vec, "Ts"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_solve_T_%d", out_dir, 0);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, node_neighbors_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    PetscPrintf(p4est_->mpicomm, "Done! \n \n \n");

  }




  my_p4est_level_set_t ls(node_neighbors_);
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, tl_.vec, num_extend_iterations_, 2,
                                    extension_band_use_, extension_band_extend_,
                                    liquid_normal_.vec, NULL, NULL,
                                    extension_use_nonzero_guess_, tl_d_.vec, tl_dd_.vec.data());
  ls.extend_Over_Interface_TVD_Full(solid_phi_.vec,  ts_.vec, num_extend_iterations_, 2,
                                    extension_band_use_, extension_band_extend_,
                                    solid_normal_.vec, NULL, NULL,
                                    extension_use_nonzero_guess_, ts_d_.vec, ts_dd_.vec.data());

//  node_neighbors_->second_derivatives_central(tl_.vec, tl_dd_.vec.data());

  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_solve_t, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_t()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Solving for temperature multiplier... \n"); CHKERRXX(ierr);
  }

//  solver_temp_->set_wc(wall_bc_type_temp_, zero_cf, false);
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

  foreach_node(n, nodes_) {
    if (liquid_phi_.ptr[n] < 0) {
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
  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_tl_.vec, num_extend_iterations_, 1,
                                    extension_band_use_, extension_band_extend_,
                                    liquid_normal_.vec, NULL, NULL,
                                    extension_use_nonzero_guess_, psi_tl_d_.vec);

//  node_neighbors_->second_derivatives_central(psi_t_.vec, psi_t_dd_.vec);

  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_psi_t, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::solve_c0()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Solving for leading concentration... \n"); CHKERRXX(ierr);
  }

//  solver_conc_leading_->set_wc(wall_bc_type_conc_, *wall_bc_value_conc_[0], false);

  solver_conc_leading_->set_wc(*wall_bc_type_conc_, *wall_bc_value_conc_[0], false);
  solver_conc_leading_->set_bc(0, DIRICHLET, pw_c0_values_);
  solver_conc_leading_->set_rhs(rhs_c_[0].vec);


  if (contr_phi_.vec != NULL) {
    solver_conc_leading_->set_bc(1, contr_bc_type_conc_, *contr_bc_value_conc_[0], zero_cf);
  }

  solver_conc_leading_->solve(c_[0].vec, poisson_use_nonzero_guess_);

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  switch (dirichlet_scheme_) {
    case 0:
    {
      boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[0].vec, num_extend_iterations_, 2,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), bc,
          extension_use_nonzero_guess_, c_d_[0].vec, c_dd_[0].vec.data());
      break;
    }
    case 1:
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[0].vec, num_extend_iterations_, 2,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), NULL,
          extension_use_nonzero_guess_, c_d_[0].vec, c_dd_[0].vec.data());
    break;
    case 2:
    {
      boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[0].vec, num_extend_iterations_, 2,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), bc,
          extension_use_nonzero_guess_, c_d_[0].vec, c_dd_[0].vec.data());
      break;
    }
    default:
      throw;
  }

//  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[0].vec, num_extend_iterations_, 2,
//      extension_band_use_, extension_band_extend_,
//      liquid_normal_.vec, NULL, NULL,
//      false, c_d_[0].vec, c_dd_[0].vec.data());

//  node_neighbors_->second_derivatives_central(c_[0].vec, c_dd_[0].vec.data());

  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c0(int scheme)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c0, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Solving for leading concentration multiplier... \n"); CHKERRXX(ierr);
  }

  vector<vector<double> > c_pw;
  vector<vector<double> > psi_c_pw;
  vector<vector<double> > normal_pw;
  vector<double>          tl_pw;
  vector<double>          psi_tl_pw;
  vector<double>          seed_pw;

  if (scheme == 1) {
    int num_bc_points = solver_conc_leading_->pw_bc_num_value_pts(0);

    c_pw.resize     (num_comps_, vector<double> (num_bc_points));
    psi_c_pw.resize (num_comps_, vector<double> (num_bc_points));
    normal_pw.resize(P4EST_DIM,  vector<double> (num_bc_points));
    tl_pw.resize    (num_bc_points);
    psi_tl_pw.resize(num_bc_points);
    seed_pw.resize  (num_bc_points);

    vector<Vec> input_all;
    vector<double *> output_all;

    input_all.push_back(tl_.vec);
    output_all.push_back(tl_pw.data());

    input_all.push_back(psi_tl_.vec);
    output_all.push_back(psi_tl_pw.data());

    for (size_t i = 0; i < num_comps_; ++i) {
      input_all.push_back(c_[i].vec);
      output_all.push_back(c_pw[i].data());
      input_all.push_back(psi_c_[i].vec);
      output_all.push_back(psi_c_pw[i].data());
    }

    foreach_dimension(dim) {
      input_all.push_back(front_normal_.vec[dim]);
      output_all.push_back(normal_pw[dim].data());
    }

    input_all.push_back(seed_map_.vec);
    output_all.push_back(seed_pw.data());

    interp_bc_points.set_input(input_all, quadratic_non_oscillatory_continuous_v2);
    interp_bc_points.interpolate(output_all.data());
  }

  // compute bondary conditions
  foreach_local_node(n, nodes_)
  {
    for (unsigned short i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0, n); ++i)
    {
      int idx = solver_conc_leading_->pw_bc_idx_value_pt(0, n, i);

      switch (scheme)
      {
        case 1:
        {
          interface_point_cartesian_t *pt;
          solver_conc_leading_->pw_bc_get_boundary_pt(0, idx, pt);

          double xyz[P4EST_DIM];
          solver_conc_leading_->pw_bc_xyz_value_pt(0, idx, xyz);

          std::vector<double> c_all(num_comps_);
          c_all[0] = pw_c0_values_[idx];
          for (int i = update_c0_robin_ == 2 ? 0 : 1; i < num_comps_; ++i) {
            c_all[i] = c_pw[i][idx];
          }

          double normal[P4EST_DIM];
          foreach_dimension(dim) {
            normal[dim] = normal_pw[dim][idx];
          }

          double eps_v = eps_v_[round(seed_pw[idx])]->value(normal);

          double conc_term = 0;

          for (int i = update_c0_robin_ == 2 ? 0 : 1; i < num_comps_; ++i) {
            conc_term += (1.-part_coeff_(i, c_all.data())) * c_all[i]
                         * psi_c_pw[i][idx];
          }

          pw_psi_c0_values_[idx] =
              -( conc_term + eps_v
                 + latent_heat_*density_s_*psi_tl_pw[idx]
                 ) /(1.-part_coeff_(0, c_all.data()))/pw_c0_values_[idx];
          // Rochi:: updating this because the stefan condition for temp flux was modified to include density_s_ by Rochi and Elyce
        }
        break;

        case 2:
          pw_psi_c0_values_[idx] = 1;
        break;

        case 3:
          pw_psi_c0_values_[idx] = pw_c0_change_[idx];
        break;

        default:
          throw;
      }
    }
  }

//  solver_conc_leading_->set_wc(wall_bc_type_conc_, zero_cf, false);
  solver_conc_leading_->set_wc(*wall_bc_type_conc_, zero_cf, false);
  solver_conc_leading_->set_rhs(rhs_zero_.vec);
  solver_conc_leading_->set_bc(0, DIRICHLET, pw_psi_c0_values_);

  if (contr_phi_.vec != NULL)
  {
    solver_conc_leading_->set_bc(1, contr_bc_type_conc_, zero_cf, zero_cf);
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
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  switch (dirichlet_scheme_) {
    case 0:
    {
      boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[0].vec, num_extend_iterations_, 1,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), bc,
          extension_use_nonzero_guess_, psi_c_d_[0].vec);
      break;
    }
    case 1:
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[0].vec, num_extend_iterations_, 1,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), NULL,
          extension_use_nonzero_guess_, psi_c_d_[0].vec);
    break;
    case 2:
    {
      boundary_conditions_t *bc = use_points_on_interface_ ? solver_conc_leading_->get_bc(0) : NULL;
      ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[0].vec, num_extend_iterations_, 1,
          extension_band_use_, extension_band_extend_,
          liquid_normal_.vec, solver_conc_leading_->get_mask(), bc,
          extension_use_nonzero_guess_, psi_c_d_[0].vec);
      break;
    }
    default:
      throw;
  }

//  ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[0].vec, num_extend_iterations_, 1,
//      extension_band_use_, extension_band_extend_,
//      liquid_normal_.vec, NULL, NULL,
//      false, psi_c_d_[0].vec);

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
  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Solve for concentrations... \n"); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

  for (int i = start; i < start+num; ++i)
  {
//    solver_conc_[i]->set_wc(wall_bc_type_conc_, *wall_bc_value_conc_[i], false);
    solver_conc_[i]->set_wc(*wall_bc_type_conc_, *wall_bc_value_conc_[i], false);

    solver_conc_[i]->set_bc(0, ROBIN, pw_c_values_[i], pw_c_values_robin_[i], pw_c_coeffs_robin_[i]);
    solver_conc_[i]->set_rhs(rhs_c_[i].vec);

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->set_bc(1, contr_bc_type_conc_, *contr_bc_value_conc_[i], zero_cf);
    }

    solver_conc_[i]->solve(c_[i].vec, poisson_use_nonzero_guess_);

    Vec mask = solver_conc_[i]->get_mask();

    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, c_[i].vec, num_extend_iterations_, 2,
                                      extension_band_use_, extension_band_extend_,
                                      liquid_normal_.vec, mask, NULL,
                                      extension_use_nonzero_guess_, c_d_[i].vec, c_dd_[i].vec.data());

//    node_neighbors_->second_derivatives_central(c_[i].vec, c_dd_[i].vec.data());
  }

  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::solve_psi_c(int start, int num)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);
  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Solve for concentrations multipliers... \n"); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(node_neighbors_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  for (int i = start; i < start+num; ++i)
  {
//    solver_conc_[i]->set_wc(wall_bc_type_conc_, zero_cf, false);
    solver_conc_[i]->set_wc(*wall_bc_type_conc_, zero_cf, false);
    solver_conc_[i]->set_bc(0, ROBIN, pw_psi_c_values_[i], pw_psi_c_values_robin_[i], pw_c_coeffs_robin_[i]);
    solver_conc_[i]->set_new_submat_robin(false);
    solver_conc_[i]->set_rhs(rhs_zero_.vec);

    if (contr_phi_.vec != NULL)
    {
      solver_conc_[i]->set_bc(1, contr_bc_type_conc_, zero_cf, zero_cf);
    }

    solver_conc_[i]->solve(psi_c_[i].vec, poisson_use_nonzero_guess_);

    Vec mask = solver_conc_[i]->get_mask();

    ls.extend_Over_Interface_TVD_Full(liquid_phi_.vec, psi_c_[i].vec, num_extend_iterations_, 1,
                                      extension_band_use_, extension_band_extend_,
                                      liquid_normal_.vec, mask, NULL,
                                      false, psi_c_d_[i].vec);
  }

  if (verbose_) {
    ierr = PetscPrintf(p4est_->mpicomm, "Done. \n"); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_multialloy_solve_psi_c1, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_poisson_nodes_multialloy_t::compute_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
  node_neighbors_->first_derivatives_central(c_[0].vec, c0d_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_poisson_nodes_multialloy_t::compute_psi_c0n()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
  node_neighbors_->first_derivatives_central(psi_c_[0].vec, psi_c0d_.vec);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_compute_psi_c0n, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_multialloy_t::compute_pw_bc_values(int start, int num)
{
  // -----------------------------------------------
  // TEMP: Elyce adding terms for eventual use in multialloy coupled with fluids:
  // -----------------------------------------------
  bool solving_with_nondim_for_fluid = false;

//  double Le_J[num_comps_];
//  double l_char = 1.;
//  double thermal_diff_l = 1.;
//  double delta_C_char[num_comps_];
//  double Cinf_char[num_comps_];

//  // initialize things to 1 for now so I don't break everything:
//  for(size_t k =0; k<num_comps_; k++){
//    Le_J[k] = 1.0;
//    delta_C_char[k] = 1.;
//    Cinf_char[k] = 0.;
//  }

  // -----------------------------------------------
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

  for (int j = 0; j < num_comps_; ++j) {
    c_[j]           .get_array();
    c_dd_[j]        .get_array();
  }
  foreach_local_node(n, nodes_)
  {
    interp_initialized = false;


    // iterate through points that impose jump conditions for temperature
    if (solver_temp_->pw_jc_num_taylor_pts(0, n) > 0)
    {
      interp_local.initialize(n);
      interp_initialized = true;

      // E: Computing at the projection point: the interfacial velocity value from solute rejection, the jump in temperature value, and the jump in temperature flux (governed by the Stefan condition)
      // projection point
      idx = solver_temp_->pw_jc_idx_taylor_pt(0, n, 0);
      solver_temp_->pw_jc_xyz_taylor_pt(0, idx, xyz_pr);

      vn_pr = 0;
      foreach_dimension(dim)
      {
        interp_local.set_input(front_normal_.ptr[dim], linear); normal[dim] = interp_local.value(xyz_pr);

        // Compute dC0/dn and temporarily set it as "vn_pr"
        interp_local.set_input(c0d_.ptr[dim],          linear); vn_pr += normal[dim]*interp_local.value(xyz_pr);
      }

      // interpolate concenrtations
      vector<double> c_all(num_comps_);
      for (int i = 0; i < num_comps_; ++i) {
        interp_local.set_input(c_[i].ptr, DIM(c_dd_[i].ptr[0], c_dd_[i].ptr[1], c_dd_[i].ptr[2]), quadratic_non_oscillatory_continuous_v2);
        c_all[i] = interp_local.value(xyz_pr);
      }

      // Compute actual interfacial velocity now as "vn_pr", as specified by the solute rejection equation
      vn_pr = (conc_diff_[0]*vn_pr - front_conc_flux_[0]->value(xyz_pr))/(c_all[0] + EPS)/(1.0-part_coeff_(0, c_all.data()));
//      printf("vn_pr = %0.3f \n", vn_pr);
//      printf("latent heat * density_s * vn_pr = %0.2e \n", latent_heat_ * density_s_ * vn_pr);
//      // ---------------------
//      // Elyce Modification:
//      // ---------------------
//      if(solving_with_nondim_for_fluid) {
//        vn_pr = (Le_J[0]*vn_pr - (l_char/(thermal_diff_l*delta_C_char[0]))*front_conc_flux_[0]->value(xyz_pr))/
//                (c_all[0] + Cinf_char[0]/delta_C_char[0])/(1.0-part_coeff_(0, c_all.data()));
//      }
//      // ---------------------
      pw_t_sol_jump_taylor_[idx] = front_temp_value_jump_->value(xyz_pr);
      pw_t_flx_jump_taylor_[idx] = front_temp_flux_jump_->value(xyz_pr) - latent_heat_*density_s_*vn_pr;
//      printf("Tl jump = %0.2e, Tl_flux_jump = %0.2e, vn_pr = %0.2e \n", pw_t_sol_jump_taylor_[idx], pw_t_flx_jump_taylor_[idx], vn_pr);

      // TO FIX !! NEED TO MULTIPLY LATENT_HEAT_*VN*DENSITY_S --> NOTE : THIS CODE IS DUPLICATED IN ABOUT 100X PLACES SO MAKE SURE THEYRE ALL CHANGED

      // ALERT ^ line above is not super compatible for modifying for nondimensionalization, I need to look into how front_temp_flux_jump_ is created
      //PetscPrintf(p4est_->mpicomm, "ALERT: multialloy: temp jump is nontrivial to modify for nondimensionalization. Need to address this. \n");
      // Note: Looks like front_temp_flux_jump and front_temp_value_jump are set in "set_front_conditions", and are provided as CF's .
      // I need to look into how multialloy provides these
      // Perhaps these are how the external jumps are provided, i.e the external source terms?

      // If Daniil's condition is [kl dTl/dn - ks dTs/dn] = hS + vn*L, then our analogous one
      // that could match his format would be:
      // [kl dTl/dn - ks dTs/dn] = (hS)/St * (alphal/alphas)*thermal_cond_s + vn/St*(alphal/alphas)*thermal_cond_s

      // BUT, if our system is nondimensionalized, then actually we will want to prescribe
      //[dTl/dn - dTs/dn], which is going to be trickier to do ...

      // OUR actual condition is given by:
      // [(kl/ks)*dTl/dn - dTs/dn] = vn*(1/St)*(alphal/alphas) (neglecting that source term)

      // E: Computing at the centroid point: the interfacial velocity value from solute rejection, the jump in temperature value, and the jump in temperature flux (governed by the Stefan condition)
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

      for (int i = 0; i < num_comps_; ++i) {
        interp_local.set_input(c_[i].ptr, DIM(c_dd_[i].ptr[0], c_dd_[i].ptr[1], c_dd_[i].ptr[2]), quadratic_non_oscillatory_continuous_v2);
        c_all[i] = interp_local.value(xyz_cd);
      }

      vn_cd = (conc_diff_[0]*vn_cd - front_conc_flux_[0]->value(xyz_cd))/(c_all[0] + EPS)/(1.0-part_coeff_(0, c_all.data()));
//      printf("vn_cd = %0.3f \n", vn_cd);

      //      // ---------------------
//      // Elyce Modification:
//      // ---------------------
//      if(solving_with_nondim_for_fluid) {
//        vn_cd = (Le_J[0]*vn_cd - (l_char/(thermal_diff_l*delta_C_char[0]))*front_conc_flux_[0]->value(xyz_cd))/
//                (c_all[0] + Cinf_char[0]/delta_C_char[0])/(1.0-part_coeff_(0, c_all.data()));
//      }
//      // ---------------------


      pw_t_flx_jump_integr_[idx] = front_temp_flux_jump_->value(xyz_cd) - latent_heat_*density_s_*vn_cd;
      // Rochi:: updating this because the stefan condition for temp flux was modified to include density_s_ by Rochi and Elyce
      // ALERT: need to modify the above ^ in the same way I address the other one
      //PetscPrintf(p4est_->mpicomm, "ALERT: multialloy: temp jump is nontrivial to modify for nondimensionalization. Need to address this. \n");
    }


    // iterate through points that impose robin boundary conditions for concentrations
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

        vector<double> c_all(num_comps_);
        for (int j = 0; j < num_comps_; ++j) {
          interp_local.set_input(c_[j].ptr, DIM(c_dd_[j].ptr[0], c_dd_[j].ptr[1], c_dd_[j].ptr[2]), quadratic_non_oscillatory_continuous_v2);
          c_all[j] = interp_local.value(xyz_cd);
        }

        vn_cd = (conc_diff_[0]*vn_cd - front_conc_flux_[0]->value(xyz_cd))/(c_all[0]+EPS)/(1.0-part_coeff_(0, c_all.data()));
//        printf("vn_cd = %0.3f \n", vn_cd);
//        // ---------------------
//        // Elyce Modification:
//        // ---------------------
//        if(solving_with_nondim_for_fluid) {
//          vn_cd = (Le_J[0]*vn_cd - (l_char/(thermal_diff_l*delta_C_char[0]))*front_conc_flux_[0]->value(xyz_cd))/
//                  (c_all[0] + Cinf_char[0]/delta_C_char[0])/(1.0-part_coeff_(0, c_all.data()));
//        }
//        // ---------------------

//        vn_cd = vn_exact_->value(xyz_cd);
        pw_c_values_robin_[i][idx] = front_conc_flux_[i]->value(xyz_cd);
        pw_c_coeffs_robin_[i][idx] = -(1.0-part_coeff_(i, c_all.data()))*vn_cd;

//        // ---------------------
//        // Elyce Modification:
//        // ---------------------
//        if(solving_with_nondim_for_fluid){
//          pw_c_values_robin_[i][idx] = (1./Le_J[i])*(l_char/(thermal_diff_l*delta_C_char[i]))*front_conc_flux_[i]->value(xyz_cd) +
//                                       (1./Le_J[i])*(vn_cd)*(1.0 - part_coeff_(i, c_all.data()))*(Cinf_char[i]/delta_C_char[i]) ;

//          pw_c_coeffs_robin_[i][idx] = -(1./Le_J[i])*(1.0-part_coeff_(i, c_all.data()))*vn_cd;

//        }
//        // ---------------------

      }
    }
  }

  front_normal_   .restore_array();
  front_curvature_.restore_array();
  seed_map_       .restore_array();
  c0d_            .restore_array();

  for (int j = 0; j < num_comps_; ++j) {
    c_[j]   .restore_array();
    c_dd_[j].restore_array();
  }

//  MPI_Barrier(p4est_->mpicomm);
//  ierr = PetscPrintf(p4est_->mpicomm, "Done.\n"); CHKERRXX(ierr);
}


// Elyce note: I believe we will have to modify the below as well, to reflect adjustments we made to the above fxn for BCs
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
  double xyz[P4EST_DIM];
  double normal[P4EST_DIM];

  double del_vn;
  double vn;
  double c0;

  vector<double> c_all(num_comps_);

  front_normal_   .get_array();
  front_curvature_.get_array();
  seed_map_       .get_array();
  c0d_            .get_array();
  psi_c0d_        .get_array();
  psi_c_[0]       .get_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].get_array();
    c_dd_[i].get_array();
  }

  foreach_local_node(n, nodes_)
  {
    interp_initialized = false;
    switch (iteration_scheme_)
    {
      case 1:
        if (solver_temp_->pw_jc_num_taylor_pts(0, n) > 0)
        {
          // projection point
          idx = solver_temp_->pw_jc_idx_taylor_pt(0, n, 0);
          solver_temp_->pw_jc_xyz_taylor_pt(0, idx, xyz_pr);

          pw_psi_t_sol_jump_taylor_[idx] = 0;
          pw_psi_t_flx_jump_taylor_[idx] = 1;

          // centroid
          idx = solver_temp_->pw_jc_idx_integr_pt(0, n, 0);
          solver_temp_->pw_jc_xyz_integr_pt(0, idx, xyz_cd);

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
              interp_local.set_input(c_[i].vec, linear);
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

            pw_psi_c_values_robin_[i][idx] = liquidus_slope_(i, c_all.data());
          }
        }
      break;
      case 2:
      case 3:
        if (solver_temp_->pw_jc_num_taylor_pts(0, n) > 0)
        {
          interp_local.initialize(n);
          interp_initialized = true;

          // projection point
          idx = solver_temp_->pw_jc_idx_taylor_pt(0, n, 0);
          solver_temp_->pw_jc_xyz_taylor_pt(0, idx, xyz);

          del_vn = 0;
          vn     = 0;
          // Rochi comment :: PW BC for psi_t problem is defined here
          foreach_dimension(dim)
          {
            interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz);
            interp_local.set_input(c0d_.vec[dim],          linear); vn     += normal[dim]*interp_local.value(xyz);
            interp_local.set_input(psi_c0d_.vec[dim],      linear); del_vn += normal[dim]*interp_local.value(xyz);
          }

          for (int j = 0; j < num_comps_; ++j) {
            interp_local.set_input(c_[j].vec, linear);
            c_all[j] = interp_local.value(xyz);
          }

          vn = (front_conc_flux_[0]->value(xyz) - conc_diff_[0]*vn)/c_all[0]/(1.0-part_coeff_(0, c_all.data()));

          interp_local.set_input(psi_c_[0].vec, linear);
          del_vn = (vn*(1.0-part_coeff_(0, c_all.data()))*interp_local.value(xyz) - conc_diff_[0]*del_vn)
              /c_all[0]/(1.0-part_coeff_(0, c_all.data()));

          pw_psi_t_sol_jump_taylor_[idx] = 0;
          pw_psi_t_flx_jump_taylor_[idx] = -del_vn*latent_heat_*density_s_;
          // Rochi:: updating this because the stefan condition for temp flux was modified to include density_s_ by Rochi and Elyce

          // centroid
          idx = solver_temp_->pw_jc_idx_integr_pt(0, n, 0);
          solver_temp_->pw_jc_xyz_integr_pt(0, idx, xyz);

          del_vn = 0;
          vn     = 0;
          foreach_dimension(dim)
          {
            interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz);
            interp_local.set_input(c0d_.vec[dim],          linear); vn     += normal[dim]*interp_local.value(xyz);
            interp_local.set_input(psi_c0d_.vec[dim],      linear); del_vn += normal[dim]*interp_local.value(xyz);
          }

          for (int j = 0; j < num_comps_; ++j) {
            interp_local.set_input(c_[j].vec, linear);
            c_all[j] = interp_local.value(xyz);
          }

          vn = (front_conc_flux_[0]->value(xyz) - conc_diff_[0]*vn)/c_all[0]/(1.0-part_coeff_(0, c_all.data()));

          interp_local.set_input(psi_c_[0].vec, linear);
          del_vn = (vn*(1.0-part_coeff_(0, c_all.data()))*interp_local.value(xyz) - conc_diff_[0]*del_vn)
              /c_all[0]/(1.0-part_coeff_(0, c_all.data()));

          pw_psi_t_flx_jump_integr_[idx] = del_vn*latent_heat_*density_s_;
          // Rochi:: updating this because the stefan condition for temp flux was modified to include density_s_ by Rochi and Elyce
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
              interp_local.set_input(c_[i].vec, linear);
              c_all[i] = interp_local.value(xyz_cd);
            }
          }
        }

        for (int i = start; i < start+num; ++i)
        {
          if (solver_conc_[i]->pw_bc_num_value_pts(0, n) > 0)
          {
            idx = solver_conc_[i]->pw_bc_idx_value_pt(0, n, 0);
            solver_conc_[i]->pw_bc_xyz_value_pt(0, idx, xyz);

            del_vn = 0;
            vn     = 0;
            foreach_dimension(dim)
            {
              interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz);
              interp_local.set_input(c0d_.vec[dim],          linear); vn     += normal[dim]*interp_local.value(xyz);
              interp_local.set_input(psi_c0d_.vec[dim],      linear); del_vn += normal[dim]*interp_local.value(xyz);
            }

            interp_local.set_input(c_[0].vec, linear);
            c0 = interp_local.value(xyz);
            vn = (front_conc_flux_[0]->value(xyz) - conc_diff_[0]*vn)/c0/(1.0-part_coeff_(0, c_all.data()));

            interp_local.set_input(psi_c_[0].vec, linear);
            del_vn = (vn*(1.0-part_coeff_(0, c_all.data()))*interp_local.value(xyz) - conc_diff_[0]*del_vn)/c0/(1.0-part_coeff_(0, c_all.data()));

            pw_psi_c_values_[i][idx] = -(1.0-part_coeff_(i, c_all.data()))*del_vn*c_all[i];

            //----
            idx = solver_conc_[i]->pw_bc_idx_robin_pt(0, n, 0);
            solver_conc_[i]->pw_bc_xyz_robin_pt(0, idx, xyz);

            del_vn = 0;
            vn     = 0;
            foreach_dimension(dim)
            {
              interp_local.set_input(front_normal_.vec[dim], linear); normal[dim] = interp_local.value(xyz);
              interp_local.set_input(c0d_.vec[dim],          linear); vn     += normal[dim]*interp_local.value(xyz);
              interp_local.set_input(psi_c0d_.vec[dim],      linear); del_vn += normal[dim]*interp_local.value(xyz);
            }

            interp_local.set_input(c_[0].vec, linear);
            c0 = interp_local.value(xyz);
            vn = (front_conc_flux_[0]->value(xyz) - conc_diff_[0]*vn)/c0/(1.0-part_coeff_(0, c_all.data()));

            interp_local.set_input(psi_c_[0].vec, linear);
            del_vn = (vn*(1.0-part_coeff_(0, c_all.data()))*interp_local.value(xyz) - conc_diff_[0]*del_vn)/c0/(1.0-part_coeff_(0, c_all.data()));

            pw_psi_c_values_robin_[i][idx] = -(1.0-part_coeff_(i, c_all.data()))*del_vn*c_all[i];
          }
        }
      break;
      default:
        throw;
    }
  }

  front_normal_   .restore_array();
  front_curvature_.restore_array();
  seed_map_       .restore_array();
  c0d_            .restore_array();
  psi_c0d_        .restore_array();
  psi_c_[0]       .restore_array();

  for (int i = 0; i < num_comps_; ++i)
  {
    c_   [i].restore_array();
    c_dd_[i].restore_array();
  }

//  MPI_Barrier(p4est_->mpicomm);
//  ierr = PetscPrintf(p4est_->mpicomm, "Done.\n"); CHKERRXX(ierr);
}


// ELYCe NOTE: need to modify the below fxn, I believe that's where the Gibbs error gets calculated?
void my_p4est_poisson_nodes_multialloy_t::compute_c0_change(int scheme)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);

  bc_error_max_ = 0;
  bc_error_avg_ = 0;
  velo_max_     = 0;

  int num_bc_points = solver_conc_leading_->pw_bc_num_value_pts(0);

  vector<vector<double> > c_pw    (num_comps_, vector<double> (num_bc_points));
  vector<vector<double> > psi_c_pw(num_comps_, vector<double> (num_bc_points));
  vector<vector<double> > normal_pw     (P4EST_DIM, vector<double> (num_bc_points));
  vector<vector<double> > grad_c0_pw    (P4EST_DIM, vector<double> (num_bc_points));
  vector<vector<double> > grad_psi_c0_pw(P4EST_DIM, vector<double> (num_bc_points));
  vector<double> tl_pw    (num_bc_points);
  vector<double> psi_tl_pw(num_bc_points);
  vector<double> curv_pw  (num_bc_points);
  vector<double> seed_pw  (num_bc_points);

  vector<Vec> input_all;
  vector<double *> output_all;

  input_all.push_back(tl_.vec);
  output_all.push_back(tl_pw.data());

  for (size_t i = 0; i < num_comps_; ++i) {
    input_all.push_back(c_[i].vec);
    output_all.push_back(c_pw[i].data());
  }

  foreach_dimension(dim) {
    input_all.push_back(front_normal_.vec[dim]);
    output_all.push_back(normal_pw[dim].data());
    input_all.push_back(c0d_.vec[dim]);
    output_all.push_back(grad_c0_pw[dim].data());
  }

  input_all.push_back(front_curvature_.vec);
  output_all.push_back(curv_pw.data());

  input_all.push_back(seed_map_.vec);
  output_all.push_back(seed_pw.data());

  if (scheme > 0) {
    input_all.push_back(psi_tl_.vec);
    output_all.push_back(psi_tl_pw.data());

    for (size_t i = 0; i < num_comps_; ++i) {
      input_all.push_back(psi_c_[i].vec);
      output_all.push_back(psi_c_pw[i].data());
    }

    foreach_dimension(dim) {
      input_all.push_back(psi_c0d_.vec[dim]);
      output_all.push_back(grad_psi_c0_pw[dim].data());
    }
  }

  interp_bc_points.set_input(input_all, quadratic_non_oscillatory_continuous_v2);
  interp_bc_points.interpolate(output_all.data());

  bc_error_.get_array();

  foreach_local_node(n, nodes_)
  {
    bc_error_.ptr[n] = 0;

    for (int i = 0; i < solver_conc_leading_->pw_bc_num_value_pts(0, n); ++i)
    {
      int idx = solver_conc_leading_->pw_bc_idx_value_pt(0, n, i);

      // fetch boundary point
      double xyz[P4EST_DIM];
      solver_conc_leading_->pw_bc_xyz_value_pt(0, idx, xyz);
      interface_point_cartesian_t *pt;
      solver_conc_leading_->pw_bc_get_boundary_pt(0, idx, pt);

      // interpolate concentration
      vector<double> c_all(num_comps_);
      c_all[0] = pw_c0_values_[idx];
      for (int k = update_c0_robin_ == 2 ? 0 : 1; k < num_comps_; ++k) {
        c_all[k] = c_pw[k][idx];
      }
      // interpolate temperature
      double tl_val = tl_pw[idx];

      // normal velocity
      double normal[P4EST_DIM] = { DIM(normal_pw[0][idx],
                                       normal_pw[1][idx],
                                       normal_pw[2][idx]) };
      double vn = SUMD(normal[0]*grad_c0_pw[0][idx],
                       normal[1]*grad_c0_pw[1][idx],
                       normal[2]*grad_c0_pw[2][idx]);

      vn = ( (conc_diff_[0]*vn - front_conc_flux_[0]->value(xyz))/(1.-(*part_coeff_)(0, c_all.data()))/(pw_c0_values_[idx] + EPS) );

      // curvature
      double kappa = curv_pw[idx];

      // undercoolings
      double eps_v = eps_v_[round(seed_pw[idx])]->value(normal);
      double eps_c = eps_c_[round(seed_pw[idx])]->value(normal);


      // error
      double error = tl_val
                     - liquidus_value_(c_all.data())
                     - eps_v*vn
                     - eps_c*kappa
                     - gibbs_thomson_->value(xyz);
      if(0){
        printf("\n--------------------------------------------------------\n");
        printf("Node %d, (%0.2f, %0.2f) -- error = %0.2e \n", n, xyz[0], xyz[1], error);

        printf("tl_val = %0.2e \n", tl_val);
        printf("liquidus = %0.2e \n", liquidus_value_(c_all.data()));
        printf("eps_v = %0.2e \n", eps_v);
        printf("vn = %0.2e \n", vn);
        printf("eps_c = %0.2e \n", eps_c);
        printf("kappa = %0.2e \n", kappa);
        printf("gibbs_thomson_ = %0.2e \n", gibbs_thomson_->value(xyz));
        printf("\n--------------------------------------------------------\n");
      }



      bc_error_.ptr[n] = MAX(bc_error_.ptr[n], fabs(error));
      bc_error_max_    = MAX(bc_error_max_,    fabs(error));
      bc_error_avg_   += fabs(error);


      pw_c0_change_[idx] = error;

      vector<double> part_coeff_all(num_comps_);
      for (int j = 0; j < num_comps_; ++j) {
        part_coeff_all[j] = part_coeff_(j, c_all.data());
      }

//      printf("C0 CHANGE UPDATE SCHEME : %d \n", scheme);
      switch (scheme)
      {
        case 0:
        {
          pw_c0_change_[idx] /= -liquidus_slope_(0, c_all.data());

//          double psi_vn = (-vn*(1.-part_coeff_all[0]) + sqrt(conc_diff_[0]*conc_diag_[0]))/c_all[0]/(1.-part_coeff_all[0]);
//          std::vector<double> psi_c_all(num_comps_);
//          psi_c_all[0] = 1.;
//          for (int k = update_c0_robin_ == 2 ? 0 : 1; k < num_comps_; ++k)
//          {
//            psi_c_all[k] = psi_vn/(-vn*(1.-part_coeff_all[k]) + sqrt(conc_diff_[k]*conc_diag_[k]))*c_all[k]*(1.-part_coeff_all[k]);
//          }

//          double psi_tl_val =psi_vn*latent_heat_/(sqrt(temp_diff_l_*temp_diag_l_)+sqrt(temp_diff_s_*temp_diag_s_));

//          double psi_conc_term = 0;
//          for (int k = 0; k < num_comps_; ++k)
//          {
//            psi_conc_term += liquidus_slope_(k, c_all.data())*psi_c_all[k];
//          }

//          pw_c0_change_[idx] /= psi_tl_val - psi_conc_term - eps_v*psi_vn;
//          pw_inverse_gradient_[idx] = 1./(psi_tl_val - psi_conc_term - eps_v*psi_vn);
        }
        break;
        case 1:
        {
          double psi_c0  = pw_psi_c0_values_[idx];
          double psi_c0n = SUMD(normal[0]*grad_psi_c0_pw[0][idx],
                                normal[1]*grad_psi_c0_pw[1][idx],
                                normal[2]*grad_psi_c0_pw[2][idx]);
          pw_c0_change_[idx] /= - (update_c0_robin_ == 2 ? 0 : liquidus_slope_(0, c_all.data()))
                                + conc_diff_[0]*psi_c0n - (1.-part_coeff_all[0])*vn*psi_c0;
        }
        break;
        case 2:
        {
          double psi_tl_val = psi_tl_pw[idx];
          std::vector<double> psi_c_all(num_comps_);

          psi_c_all[0] = pw_psi_c0_values_[idx];
          for (int k = update_c0_robin_ == 2 ? 0 : 1; k < num_comps_; ++k) {
            psi_c_all[k] = psi_c_pw[k][idx];
          }

          double psi_conc_term = 0;
          for (int k = 0; k < num_comps_; ++k) {
            psi_conc_term += liquidus_slope_(k, c_all.data())*psi_c_all[k];
          }

          double psi_vn = SUMD(normal[0]*grad_psi_c0_pw[0][idx],
                               normal[1]*grad_psi_c0_pw[1][idx],
                               normal[2]*grad_psi_c0_pw[2][idx]);

          psi_vn = (vn*(1.-part_coeff_all[0])*psi_c_all[0] - conc_diff_[0]*psi_vn)/c_all[0]/(1.-part_coeff_all[0]);

          pw_c0_change_[idx] /= psi_tl_val - psi_conc_term - eps_v*psi_vn;
//          printf("Node %d, (%0.2f, %0.2f) : c_all = %0.2e, psi_vn = %0.2e, pw_c0_change = %0.2e \n", n, xyz[0], xyz[1], c_all[0], psi_vn, pw_c0_change_[idx]);

          pw_inverse_gradient_[idx] = 1./(psi_tl_val - psi_conc_term - eps_v*psi_vn);
        }
        break;
        case 3:
        {
          double psi_tl_val = psi_tl_pw[idx];
          std::vector<double> psi_c_all(num_comps_);

          psi_c_all[0] = pw_psi_c0_values_[idx];
          for (int k = update_c0_robin_ == 2 ? 0 : 1; k < num_comps_; ++k) {
            psi_c_all[k] = psi_c_pw[k][idx];
          }

          double psi_conc_term = 0;
          for (int k = 0; k < num_comps_; ++k) {
            psi_conc_term += liquidus_slope_(k, c_all.data())*psi_c_all[k];
          }

          double psi_vn = SUMD(normal[0]*grad_psi_c0_pw[0][idx],
                               normal[1]*grad_psi_c0_pw[1][idx],
                               normal[2]*grad_psi_c0_pw[2][idx]);

          psi_vn = (vn*(1.-part_coeff_all[0])*psi_c_all[0] - conc_diff_[0]*psi_vn)/c_all[0]/(1.-part_coeff_all[0]);
          pw_c0_change_[idx] = (2.*error - (psi_tl_val - psi_conc_term - eps_v*psi_vn))*pw_inverse_gradient_[idx];
        }
        break;
        default:
          throw;
      }

//      pw_c0_values_[idx] -= change;
      velo_max_ = MAX(velo_max_, fabs(vn));
    }
  }
  bc_error_.restore_array();

  double buffer[4] = { bc_error_max_, bc_error_avg_, velo_max_, double(solver_conc_leading_->pw_bc_num_value_pts(0)) };
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, buffer, 4, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  bc_error_max_  = buffer[0];
  bc_error_avg_  = buffer[1];
  velo_max_      = buffer[2];
  double num_points = buffer[3];

  bc_error_avg_ /= num_points;

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_multialloy_adjust_c0, 0, 0, 0, 0); CHKERRXX(ierr);
}
