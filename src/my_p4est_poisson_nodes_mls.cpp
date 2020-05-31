#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_poisson_nodes_mls.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_setup_linear_system;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_rhs;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_preassemble_linear_system;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_solve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_determine_node_types;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submatrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_rhs_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_submat_main_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_matrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_main;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_robin;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_diag;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif



my_p4est_poisson_nodes_mls_t::my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *ngbd)
  : ngbd_(ngbd),
    p4est_(ngbd->p4est),
    nodes_(ngbd->nodes),
    ghost_(ngbd->ghost),
    brick_(ngbd->myb),
    bdry_(ngbd_, p4est_, nodes_),
    infc_(ngbd_, p4est_, nodes_),
    mu_m_interp_(ngbd),
    mu_p_interp_(ngbd)
{
  // grid variables
  lip_ = 2.0;

  ::dxyz_min(p4est_, dxyz_m_);

  XCODE( dx_min_ = dxyz_m_[0]; )
  YCODE( dy_min_ = dxyz_m_[1]; )
  ZCODE( dz_min_ = dxyz_m_[2]; )

  d_min_ = MIN(DIM(dx_min_, dy_min_, dz_min_));
  diag_min_ = sqrt( SUMD(dx_min_*dx_min_, dy_min_*dy_min_, dz_min_*dz_min_) );

  // linear system
  A_            = NULL;
  diag_scaling_ = NULL;
  rhs_          = NULL;
  rhs_ptr       = NULL;
  rhs_jump_     = NULL;
  rhs_jump_ptr  = NULL;
  rhs_gf_       = NULL;
  rhs_gf_ptr    = NULL;

  // rhs_ gonna serve as a template vector for VecDuplicate
  ierr = VecCreateGhostNodes(p4est_, nodes_, &rhs_); CHKERRXX(ierr);

  // subcomponents of linear system
  submat_main_         = NULL;
  submat_diag_         = NULL;
  submat_diag_ptr      = NULL;
  submat_diag_ghost_   = NULL;
  submat_diag_ghost_ptr= NULL;
  submat_jump_         = NULL;
  submat_jump_ghost_   = NULL;
  submat_robin_sc_     = NULL;
  submat_robin_sym_    = NULL;
  submat_robin_sym_ptr = NULL;

  submat_gf_           = NULL;
  submat_gf_ghost_     = NULL;

  new_submat_main_  = true;
  new_submat_diag_  = true;
  new_submat_robin_ = true;

  there_is_diag_      = false;
  there_is_dirichlet_ = false;
  there_is_neumann_   = false;
  there_is_robin_     = false;
  there_is_jump_      = false;
  there_is_jump_mu_   = false;

  A_needs_reassembly_ = true;

  // PETSc solver
  ierr = KSPCreate(p4est_->mpicomm, &ksp_); CHKERRXX(ierr);
  new_pc_ = true;
  atol_   = 1.0e-16;
  rtol_   = 1.0e-16;
  dtol_   = PETSC_DEFAULT;
  itmax_  = 100;

  // Tolerances for solving nonlinear equations
  nonlinear_change_tol_ = 1.0e-12,
  nonlinear_pde_residual_tol_= 0;
  nonlinear_itmax_ = 10;
  nonlinear_method_ = 1;

  // local to global node number mapping
  // compute global numbering of nodes
  global_node_offset_.resize(p4est_->mpisize+1, 0);
  for (int r = 0; r<p4est_->mpisize; ++r)
    global_node_offset_[r+1] = global_node_offset_[r] + (PetscInt)nodes_->global_owned_indeps[r];

  // construct petsc global indices
  petsc_gloidx_.resize(nodes_->indep_nodes.elem_count);

  // local nodes
  for (p4est_locidx_t i = 0; i<nodes_->num_owned_indeps; i++)
    petsc_gloidx_[i] = global_node_offset_[p4est_->mpirank] + i;

  // ghost nodes
  p4est_locidx_t ghost_size = nodes_->indep_nodes.elem_count - nodes_->num_owned_indeps;
  for (p4est_locidx_t i = 0; i<ghost_size; i++){
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, i + nodes_->num_owned_indeps);
    petsc_gloidx_[i+nodes_->num_owned_indeps] = global_node_offset_[nodes_->nonlocal_ranks[i]] + ni->p.piggy3.local_num;
  }

  // pinning point (for ill-defined all-neumann case)
  matrix_has_nullspace_ = false;
  fixed_value_idx_l_    = global_node_offset_[p4est_->mpisize];
  fixed_value_idx_g_    = global_node_offset_[p4est_->mpisize];

  // forces
  rhs_m_ = NULL; rhs_m_ptr = NULL;
  rhs_p_ = NULL; rhs_p_ptr = NULL;

  // linear term
  var_diag_      = false;
  diag_m_scalar_ = 0.;
  diag_p_scalar_ = 0.;
  diag_m_        = NULL;
  diag_p_        = NULL;
  diag_m_ptr     = NULL;
  diag_p_ptr     = NULL;

  // diffusion coefficient
  mu_m_ = 1.;
  mu_p_ = 1.;

  var_mu_            = false;
  is_mue_m_dd_owned_ = false;
  is_mue_p_dd_owned_ = false;
  mue_m_ = NULL; mue_m_xx_ = NULL; mue_m_yy_ = NULL; ONLY3D(mue_m_zz_ = NULL);
  mue_p_ = NULL; mue_p_xx_ = NULL; mue_p_yy_ = NULL; ONLY3D(mue_p_zz_ = NULL);

  mue_m_ptr = NULL; mue_m_xx_ptr = NULL; mue_m_yy_ptr = NULL; ONLY3D(mue_m_zz_ptr = NULL);
  mue_p_ptr = NULL; mue_p_xx_ptr = NULL; mue_p_yy_ptr = NULL; ONLY3D(mue_p_zz_ptr = NULL);

  // wall conditions
  wc_type_  = NULL;
  wc_value_ = NULL;
  wc_coeff_ = NULL;

  // solver options
  integration_order_          = 2;
  cube_refinement_            = 0;
  jump_scheme_                = 0;
  fv_scheme_                  = 1;
  use_taylor_correction_      = 1;
  kink_special_treatment_     = 1;
  neumann_wall_first_order_   = 0;
  enfornce_diag_scaling_      = 1;
  use_centroid_always_        = 0;
  phi_perturbation_           = 1.e-12;
  interp_method_              = quadratic_non_oscillatory_continuous_v2;

  dirichlet_scheme_ = 0;
  gf_order_         = 2;
  gf_stabilized_    = 1;
//  gf_thresh_        = 1.e-6;
  gf_thresh_        = -0.1;

  domain_rel_thresh_    = 1.e-11;
  interface_rel_thresh_ = 1.e-11;

  // auxiliary variables
  mask_m_    = NULL; mask_m_ptr    = NULL;
  mask_p_    = NULL; mask_p_ptr    = NULL;
  areas_m_   = NULL; areas_m_ptr   = NULL;
  areas_p_   = NULL; areas_p_ptr   = NULL;
  volumes_m_ = NULL; volumes_m_ptr = NULL;
  volumes_p_ = NULL; volumes_p_ptr = NULL;

  volumes_owned_    = false;
  volumes_computed_ = false;

  face_area_scalling_ = pow(diag_min_, P4EST_DIM-1);

  // finite volumes
  store_finite_volumes_       = false;
  finite_volumes_initialized_ = false;
  finite_volumes_owned_       = false;
  bdry_fvs_                   = NULL;
  infc_fvs_                   = NULL;
}

my_p4est_poisson_nodes_mls_t::~my_p4est_poisson_nodes_mls_t()
{
  // subcomponents of linear system
  if (submat_main_       != NULL) { ierr = MatDestroy(submat_main_      ); CHKERRXX(ierr); }
  if (submat_diag_       != NULL) { ierr = VecDestroy(submat_diag_      ); CHKERRXX(ierr); }
  if (submat_diag_ghost_ != NULL) { ierr = VecDestroy(submat_diag_ghost_); CHKERRXX(ierr); }
  if (submat_jump_       != NULL) { ierr = MatDestroy(submat_jump_      ); CHKERRXX(ierr); }
  if (submat_jump_ghost_ != NULL) { ierr = MatDestroy(submat_jump_ghost_); CHKERRXX(ierr); }
  if (submat_robin_sc_   != NULL) { ierr = MatDestroy(submat_robin_sc_  ); CHKERRXX(ierr); }
  if (submat_robin_sym_  != NULL) { ierr = VecDestroy(submat_robin_sym_ ); CHKERRXX(ierr); }
  if (submat_gf_         != NULL) { ierr = MatDestroy(submat_gf_        ); CHKERRXX(ierr); }
  if (submat_gf_ghost_   != NULL) { ierr = MatDestroy(submat_gf_ghost_  ); CHKERRXX(ierr); }

  // PETSc solver
  if (ksp_ != NULL) { ierr = KSPDestroy(ksp_); CHKERRXX(ierr); }

  // diffusion coefficient
  if (is_mue_p_dd_owned_)
  {
    XCODE( if (mue_p_xx_ != NULL) { ierr = VecDestroy(mue_p_xx_); CHKERRXX(ierr); } );
    YCODE( if (mue_p_yy_ != NULL) { ierr = VecDestroy(mue_p_yy_); CHKERRXX(ierr); } );
    ZCODE( if (mue_p_zz_ != NULL) { ierr = VecDestroy(mue_p_zz_); CHKERRXX(ierr); } );
  }

  if (is_mue_m_dd_owned_)
  {
    XCODE( if (mue_m_xx_ != NULL) { ierr = VecDestroy(mue_m_xx_); CHKERRXX(ierr); } );
    YCODE( if (mue_m_yy_ != NULL) { ierr = VecDestroy(mue_m_yy_); CHKERRXX(ierr); } );
    ZCODE( if (mue_m_zz_ != NULL) { ierr = VecDestroy(mue_m_zz_); CHKERRXX(ierr); } );
  }

  // auxiliary variables
  if (mask_m_    != NULL) { ierr = VecDestroy(mask_m_); CHKERRXX(ierr); }
  if (mask_p_    != NULL) { ierr = VecDestroy(mask_p_); CHKERRXX(ierr); }
  if (areas_m_   != NULL) { ierr = VecDestroy(areas_m_); CHKERRXX(ierr); }
  if (areas_p_   != NULL) { ierr = VecDestroy(areas_p_); CHKERRXX(ierr); }
  if (volumes_m_ != NULL) { ierr = VecDestroy(volumes_m_); CHKERRXX(ierr); }
  if (volumes_p_ != NULL) { ierr = VecDestroy(volumes_p_); CHKERRXX(ierr); }

  // finite volumes
  if (finite_volumes_owned_)
  {
    if (bdry_fvs_ != NULL) { delete bdry_fvs_; }
    if (infc_fvs_ != NULL) { delete infc_fvs_; }
  }

  // linear system
  if (A_            != NULL) { ierr = MatDestroy(A_           ); CHKERRXX(ierr); }
  if (diag_scaling_ != NULL) { ierr = VecDestroy(diag_scaling_); CHKERRXX(ierr); }
  if (rhs_jump_     != NULL) { ierr = VecDestroy(rhs_jump_    ); CHKERRXX(ierr); }
  if (rhs_gf_       != NULL) { ierr = VecDestroy(rhs_gf_      ); CHKERRXX(ierr); }
  if (rhs_          != NULL) { ierr = VecDestroy(rhs_         ); CHKERRXX(ierr); } // must be deleted last since it's a parent for others
}

my_p4est_poisson_nodes_mls_t::geometry_t::~geometry_t()
{
  PetscErrorCode ierr;
  if (is_phi_eff_owned) { ierr = VecDestroy(this->phi_eff); CHKERRXX(ierr); }

  for (int i=0; i<num_phi; ++i)
  {
    if (is_phi_dd_owned[i])
    {
      XCODE( ierr = VecDestroy(phi_xx[i]); CHKERRXX(ierr); );
      YCODE( ierr = VecDestroy(phi_yy[i]); CHKERRXX(ierr); );
      ZCODE( ierr = VecDestroy(phi_zz[i]); CHKERRXX(ierr); );
    }
  }
}

void my_p4est_poisson_nodes_mls_t::geometry_t::get_arrays()
{
  PetscErrorCode ierr;

  _CODE( phi_ptr.resize   (num_phi, NULL) );
  XCODE( phi_xx_ptr.resize(num_phi, NULL) );
  YCODE( phi_yy_ptr.resize(num_phi, NULL) );
  ZCODE( phi_zz_ptr.resize(num_phi, NULL) );

  if (phi_eff != NULL) { ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr); }

  for (int i=0; i<num_phi; ++i)
  {
    _CODE( ierr = VecGetArray(phi   [i], &phi_ptr   [i]); CHKERRXX(ierr); );
    XCODE( ierr = VecGetArray(phi_xx[i], &phi_xx_ptr[i]); CHKERRXX(ierr); );
    YCODE( ierr = VecGetArray(phi_yy[i], &phi_yy_ptr[i]); CHKERRXX(ierr); );
    ZCODE( ierr = VecGetArray(phi_zz[i], &phi_zz_ptr[i]); CHKERRXX(ierr); );
  }
}

void my_p4est_poisson_nodes_mls_t::geometry_t::restore_arrays()
{
  PetscErrorCode ierr;

  if (phi_eff != NULL) { ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr); }

  for(int i=0; i<num_phi; ++i)
  {
    _CODE( ierr = VecRestoreArray(phi   [i], &phi_ptr   [i]); CHKERRXX(ierr); );
    XCODE( ierr = VecRestoreArray(phi_xx[i], &phi_xx_ptr[i]); CHKERRXX(ierr); );
    YCODE( ierr = VecRestoreArray(phi_yy[i], &phi_yy_ptr[i]); CHKERRXX(ierr); );
    ZCODE( ierr = VecRestoreArray(phi_zz[i], &phi_zz_ptr[i]); CHKERRXX(ierr); );
  }
}

void my_p4est_poisson_nodes_mls_t::geometry_t::calculate_phi_eff()
{
  PetscErrorCode ierr;

  if (phi_eff == NULL)
  {
    if (num_phi == 1)
    {
      is_phi_eff_owned = false;
      phi_eff          = phi[0];
    }
    else
    {
      ierr = VecCreateGhostNodes(this->p4est_, this->nodes_, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(phi_eff, nodes_, phi, opn);
      is_phi_eff_owned = true;
    }
  }
}
Vec my_p4est_poisson_nodes_mls_t::geometry_t::return_phi_eff()
{
  P4EST_ASSERT(phi_eff!=NULL);
  Vec phi_to_return = phi_eff;
  return phi_to_return;
}

void my_p4est_poisson_nodes_mls_t::geometry_t::add_phi(mls_opn_t opn, Vec phi, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz))
{
  PetscErrorCode ierr;

  this->opn.push_back(opn);
  this->phi.push_back(phi);
  this->clr.push_back(num_phi);

  if (ANDD(phi_xx != NULL, phi_yy != NULL, phi_zz != NULL))
  {
    is_phi_dd_owned.push_back(false);

    XCODE( this->phi_xx.push_back(phi_xx) );
    YCODE( this->phi_yy.push_back(phi_yy) );
    ZCODE( this->phi_zz.push_back(phi_zz) );
  }
  else
  {
    is_phi_dd_owned.push_back(true);

    if (this->phi.size() == 1)
    {
      XCODE( this->phi_xx.push_back(Vec()); ierr = VecCreateGhostNodes(this->p4est_, this->nodes_, &this->phi_xx.back()); CHKERRXX(ierr); );
      YCODE( this->phi_yy.push_back(Vec()); ierr = VecCreateGhostNodes(this->p4est_, this->nodes_, &this->phi_yy.back()); CHKERRXX(ierr); );
      ZCODE( this->phi_zz.push_back(Vec()); ierr = VecCreateGhostNodes(this->p4est_, this->nodes_, &this->phi_zz.back()); CHKERRXX(ierr); );
    }
    else
    {
      XCODE( this->phi_xx.push_back(Vec()); ierr = VecDuplicate(this->phi_xx[0], &this->phi_xx.back() ); CHKERRXX(ierr); );
      YCODE( this->phi_yy.push_back(Vec()); ierr = VecDuplicate(this->phi_yy[0], &this->phi_yy.back() ); CHKERRXX(ierr); );
      ZCODE( this->phi_zz.push_back(Vec()); ierr = VecDuplicate(this->phi_zz[0], &this->phi_zz.back() ); CHKERRXX(ierr); );
    }

    ngbd_->second_derivatives_central(this->phi.back(), DIM( this->phi_xx.back(),
                                                             this->phi_yy.back(),
                                                             this->phi_zz.back() ) );
  }

  num_phi++;
}

void my_p4est_poisson_nodes_mls_t::set_mu(Vec mue_m, DIM( Vec mue_m_xx, Vec mue_m_yy, Vec mue_m_zz ),
                                          Vec mue_p, DIM( Vec mue_p_xx, Vec mue_p_yy, Vec mue_p_zz ))
{
  mue_m_ = mue_m;
  mue_p_ = mue_p;

  if (ANDD(mue_m_xx != NULL, mue_m_yy != NULL, mue_m_zz != NULL) &&
      ANDD(mue_p_xx != NULL, mue_p_yy != NULL, mue_p_zz != NULL))
  {
    mue_m_xx_ = mue_m_xx; mue_m_yy_ = mue_m_yy; ONLY3D(mue_m_zz_ = mue_m_zz);
    mue_p_xx_ = mue_p_xx; mue_p_yy_ = mue_p_yy; ONLY3D(mue_p_zz_ = mue_p_zz);
    is_mue_m_dd_owned_ = false;
    is_mue_p_dd_owned_ = false;
  } else {
    compute_mue_dd();
  }

  var_mu_           = true;
  new_submat_main_  = true;
  new_submat_robin_ = true;
  there_is_jump_mu_ = !(mue_m == mue_p);
}

void my_p4est_poisson_nodes_mls_t::compute_mue_dd()
{
  if (is_mue_m_dd_owned_)
  {
    XCODE( if (mue_m_xx_ != NULL) { ierr = VecDestroy(mue_m_xx_); CHKERRXX(ierr); } );
    YCODE( if (mue_m_yy_ != NULL) { ierr = VecDestroy(mue_m_yy_); CHKERRXX(ierr); } );
    ZCODE( if (mue_m_zz_ != NULL) { ierr = VecDestroy(mue_m_zz_); CHKERRXX(ierr); } );
  }

  if (is_mue_p_dd_owned_)
  {
    XCODE( if (mue_p_xx_ != NULL) { ierr = VecDestroy(mue_p_xx_); CHKERRXX(ierr); } );
    YCODE( if (mue_p_yy_ != NULL) { ierr = VecDestroy(mue_p_yy_); CHKERRXX(ierr); } );
    ZCODE( if (mue_p_zz_ != NULL) { ierr = VecDestroy(mue_p_zz_); CHKERRXX(ierr); } );
  }

  XCODE( ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_xx_); CHKERRXX(ierr); );
  YCODE( ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_yy_); CHKERRXX(ierr); );
  ZCODE( ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_zz_); CHKERRXX(ierr); );

  ngbd_->second_derivatives_central(mue_m_, DIM(mue_m_xx_, mue_m_yy_, mue_m_zz_) );
  is_mue_m_dd_owned_ = true;

  if (mue_m_ == mue_p_)
  {
    XCODE( mue_p_xx_ = mue_m_xx_ );
    YCODE( mue_p_yy_ = mue_m_yy_ );
    ZCODE( mue_p_zz_ = mue_m_zz_ );
    is_mue_p_dd_owned_ = false;
  }
  else
  {
    XCODE( ierr = VecDuplicate(mue_m_xx_, &mue_p_xx_); CHKERRXX(ierr); );
    YCODE( ierr = VecDuplicate(mue_m_yy_, &mue_p_yy_); CHKERRXX(ierr); );
    ZCODE( ierr = VecDuplicate(mue_m_zz_, &mue_p_zz_); CHKERRXX(ierr); );
    ngbd_->second_derivatives_central(mue_p_, DIM(mue_p_xx_, mue_p_yy_, mue_p_zz_) );
    is_mue_p_dd_owned_ = true;
  }
}

void my_p4est_poisson_nodes_mls_t::preassemble_linear_system()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_preassemble_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);

  setup_linear_system(false);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_preassemble_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::solve(Vec solution, bool use_nonzero_guess, bool update_ghost, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_solve, 0, 0, 0, 0); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_setup_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);
  setup_linear_system(true);
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_setup_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);

  invert_linear_system(solution, use_nonzero_guess, update_ghost, ksp_type, pc_type);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_solve, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::invert_linear_system(Vec solution, bool use_nonzero_guess, bool update_ghost, KSPType ksp_type, PCType pc_type)
{
  // set ksp type
  ierr = KSPSetType(ksp_, ksp_type); CHKERRXX(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp_, (PetscBool) use_nonzero_guess); CHKERRXX(ierr);

  if (new_pc_ || 1)
  {
    new_pc_ = false;
    ierr = KSPSetOperators(ksp_, A_, A_, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    ierr = KSPSetOperators(ksp_, A_, A_, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp_, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
   * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
   * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
   */
  if (!strcmp(pc_type, PCHYPRE)){
    /* 1- Strong threshold:
     * Between 0 to 1
     * "0 "gives better convergence rate (in 3D).
     * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
    */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.1"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.93"); CHKERRXX(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
//    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "CLJP"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
//    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.5"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0."); CHKERRXX(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (matrix_has_nullspace_){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }

  if (!strcmp(pc_type, PCASM))
  {
    ierr = PetscOptionsSetValue("-sub_pc_type", "ilu"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_pc_factor_levels", "3"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_ksp_type", "preonly"); CHKERRXX(ierr);
  }

  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // Solve the system
  ierr = KSPSetTolerances(ksp_, rtol_, atol_, dtol_, itmax_); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp_); CHKERRXX(ierr);

//  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);
  MatNullSpace A_null;
  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceCreate(p4est_->mpicomm, PETSC_TRUE, 0, NULL, &A_null); CHKERRXX(ierr);
    ierr = MatSetNullSpace(A_, A_null);

    // For purely neumann problems GMRES is more robust
    ierr = KSPSetType(ksp_, KSPGMRES); CHKERRXX(ierr);
  }

  // set
  double *mask_m_ptr; ierr = VecGetArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
  double *mask_p_ptr; ierr = VecGetArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);
  double *sol_ptr;    ierr = VecGetArray(solution, &sol_ptr); CHKERRXX(ierr);
  double *rhs_ptr;    ierr = VecGetArray(rhs_, &rhs_ptr); CHKERRXX(ierr);

  if (use_nonzero_guess) {
    foreach_node(n, nodes_) {
      if (mask_m_ptr[n] > 0 && mask_p_ptr[n] > 0) rhs_ptr[n] = sol_ptr[n];
    }
  } else {
    foreach_node(n, nodes_) {
      if (mask_m_ptr[n] > 0 && mask_p_ptr[n] > 0) sol_ptr[n] = rhs_ptr[n];
    }
  }

  ierr = VecRestoreArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(solution, &sol_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_, &rhs_ptr); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_KSPSolve, 0, 0, 0, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp_, rhs_, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_KSPSolve, 0, 0, 0, 0); CHKERRXX(ierr);
  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceDestroy(A_null);
  }
//  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_mls_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);

  KSPConvergedReason outcome;
  ierr = KSPGetConvergedReason(ksp_, &outcome); CHKERRXX(ierr);

  if (outcome < 0)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "Warning! KSP did not converge (error code %d). Setting solution to 0.\n", outcome); CHKERRXX(ierr);
    VecSetGhost(solution, 0);
  }

  if (there_is_dirichlet_ && dirichlet_scheme_ == 1)
  {
    Vec tmp;
    ierr = VecDuplicate(solution, &tmp); CHKERRXX(ierr);
    ierr = VecCopy(solution, tmp); CHKERRXX(ierr);
    ierr = MatMultAdd(submat_gf_ghost_, tmp, solution, solution); CHKERRXX(ierr);
    ierr = VecAXPY(solution, -1.0, rhs_gf_); CHKERRXX(ierr);
    ierr = VecDestroy(tmp); CHKERRXX(ierr);
  }

  // update ghosts
  if (update_ghost)
  {
    ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
}

void my_p4est_poisson_nodes_mls_t::setup_linear_system(bool setup_rhs)
{
#ifdef CASL_THROWS
  if (wc_type_ == NULL || wc_value_ == NULL)
    throw std::domain_error("[CASL_ERROR]: the boundary conditions on walls have not been set.");
#endif

  // finalize geometry
  set_boundary_phi_eff (bdry_.phi_eff);
  set_interface_phi_eff(infc_.phi_eff);

  bool assembling_main       = new_submat_main_;
  bool assembling_jump       = new_submat_main_  && there_is_jump_;
  bool assembling_jump_ghost = new_submat_main_  && there_is_jump_  && there_is_jump_mu_;
  bool assembling_robin_sc   = new_submat_robin_ && there_is_robin_ && (fv_scheme_ == 1);
  bool assembling_gf         = new_submat_main_  && there_is_dirichlet_ && dirichlet_scheme_ == 1;

  // arrays to store matrices
  int nm = assembling_main       ? nodes_->num_owned_indeps : 0;
  int nj = assembling_jump       ? nodes_->num_owned_indeps : 0;
  int na = assembling_jump_ghost ? nodes_->num_owned_indeps : 0;
  int nr = assembling_robin_sc   ? nodes_->num_owned_indeps : 0;
  int nd = assembling_gf         ? nodes_->num_owned_indeps : 0;

  std::vector< std::vector<mat_entry_t> > entries_main      (nm);
  std::vector< std::vector<mat_entry_t> > entries_jump      (nj);
  std::vector< std::vector<mat_entry_t> > entries_jump_ghost(na);
  std::vector< std::vector<mat_entry_t> > entries_gf        (nd);
  std::vector< std::vector<mat_entry_t> > entries_gf_ghost  (nd);
  if (assembling_robin_sc) entries_robin_sc.assign(nr, std::vector<mat_entry_t>(0));

  std::vector<PetscInt> d_nnz_main      (nm, 1);
  std::vector<PetscInt> o_nnz_main      (nm, 0);
  std::vector<PetscInt> d_nnz_jump      (nj, 1);
  std::vector<PetscInt> o_nnz_jump      (nj, 0);
  std::vector<PetscInt> d_nnz_jump_ghost(na, 1);
  std::vector<PetscInt> o_nnz_jump_ghost(na, 0);
  std::vector<PetscInt> d_nnz_robin_sc  (nr, 1);
  std::vector<PetscInt> o_nnz_robin_sc  (nr, 0);
  std::vector<PetscInt> d_nnz_gf        (nd, 1);
  std::vector<PetscInt> o_nnz_gf        (nd, 0);
  std::vector<PetscInt> d_nnz_gf_ghost  (nd, 1);
  std::vector<PetscInt> o_nnz_gf_ghost  (nd, 0);

  if (assembling_main)
  {
    if (mask_m_ != NULL) { ierr = VecDestroy(mask_m_); CHKERRXX(ierr); }
    if (mask_p_ != NULL) { ierr = VecDestroy(mask_p_); CHKERRXX(ierr); }

    ierr = VecDuplicate(rhs_, &mask_m_); CHKERRXX(ierr);
    ierr = VecDuplicate(rhs_, &mask_p_); CHKERRXX(ierr);

    VecSetGhost(mask_m_, -1);
    VecSetGhost(mask_p_, -1);

    ierr = VecGetArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);
  }

  if (new_submat_diag_ && there_is_diag_)
  {
    if (submat_diag_ != NULL) { ierr = VecDestroy(submat_diag_); CHKERRXX(ierr); }
    ierr = VecDuplicate(rhs_, &submat_diag_); CHKERRXX(ierr);
    ierr = VecSetGhost(submat_diag_, 0.); CHKERRXX(ierr);
    ierr = VecGetArray(submat_diag_, &submat_diag_ptr); CHKERRXX(ierr);

    if (there_is_jump_)
    {
      if (submat_diag_ghost_ != NULL) { ierr = VecDestroy(submat_diag_ghost_); CHKERRXX(ierr); }
      ierr = VecDuplicate(rhs_, &submat_diag_ghost_); CHKERRXX(ierr);
      ierr = VecSetGhost(submat_diag_ghost_, 0.); CHKERRXX(ierr);
      ierr = VecGetArray(submat_diag_ghost_, &submat_diag_ghost_ptr); CHKERRXX(ierr);
    }
  }

  if (var_diag_)
  {
    ierr = VecGetArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);
  }

  if (new_submat_robin_ && (fv_scheme_ == 0) && there_is_robin_)
  {
    if (submat_robin_sym_ != NULL) { ierr = VecDestroy(submat_robin_sym_); CHKERRXX(ierr); }
    ierr = VecDuplicate(rhs_, &submat_robin_sym_); CHKERRXX(ierr);
    ierr = VecSetGhost(submat_robin_sym_, 0.); CHKERRXX(ierr);
    ierr = VecGetArray(submat_robin_sym_, &submat_robin_sym_ptr); CHKERRXX(ierr);
  }

  // structures for quick reassembling
  if (new_submat_main_)
  {
    for (int i = 0; i < bc_.size(); ++i) bc_[i].reset(nodes_->num_owned_indeps);
    for (int i = 0; i < jc_.size(); ++i) jc_[i].reset(nodes_->num_owned_indeps);

    wall_pieces_map.reinitialize(nodes_->num_owned_indeps);
    wall_pieces_id.clear();
    wall_pieces_area.clear();
    wall_pieces_centroid.clear();

    jump_scaling_.resize(nodes_->num_owned_indeps, 1.);
  }

  // get access to vectors

  // level-set functions
  bdry_.get_arrays();
  infc_.get_arrays();

  // diffusion coefficients
  if (var_mu_)
  {
    _CODE( ierr = VecGetArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr); ierr = VecGetArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr); );
    XCODE( ierr = VecGetArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr); ierr = VecGetArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr); );
    YCODE( ierr = VecGetArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr); ierr = VecGetArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr); );
    ZCODE( ierr = VecGetArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr); ierr = VecGetArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr); );
  }

  // rhs
  if (setup_rhs)
  {
    VecSetGhost(rhs_, 0);
    ierr = VecGetArray(rhs_,   &rhs_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
  }

  // max areas (for sc scheme)
  if (there_is_neumann_ || there_is_robin_ || there_is_jump_)
  {
    if (assembling_main)
    {
      if (areas_m_ != NULL) { ierr = VecDestroy(areas_m_); CHKERRXX(ierr); }
      if (areas_p_ != NULL) { ierr = VecDestroy(areas_p_); CHKERRXX(ierr); }

      if (volumes_m_ != NULL) { ierr = VecDestroy(volumes_m_); CHKERRXX(ierr); }
      if (volumes_p_ != NULL) { ierr = VecDestroy(volumes_p_); CHKERRXX(ierr); }

      ierr = VecDuplicate(rhs_, &areas_m_); CHKERRXX(ierr);
      ierr = VecDuplicate(rhs_, &areas_p_); CHKERRXX(ierr);

      ierr = VecDuplicate(rhs_, &volumes_m_); CHKERRXX(ierr);
      ierr = VecDuplicate(rhs_, &volumes_p_); CHKERRXX(ierr);
    }

    ierr = VecGetArray(areas_m_, &areas_m_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(areas_p_, &areas_p_ptr);   CHKERRXX(ierr);

    ierr = VecGetArray(volumes_m_, &volumes_m_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(volumes_p_, &volumes_p_ptr);   CHKERRXX(ierr);
  }

  // auxilary variables

  // extended node neighbourhood (including points on diagonals)
  bool           neighbors_exist[num_neighbors_cube];
  p4est_locidx_t neighbors      [num_neighbors_cube];

  // addition to rhs from jump condition discretization
//  Vec     rhs_jump;
//  double *rhs_jump_ptr;

  if (setup_rhs && there_is_jump_)
  {
    if (rhs_jump_ != NULL) { ierr = VecDestroy(rhs_jump_); CHKERRXX(ierr); }
    ierr = VecDuplicate(rhs_, &rhs_jump_); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_jump_, &rhs_jump_ptr); CHKERRXX(ierr);
  }

  if (setup_rhs && there_is_dirichlet_ && dirichlet_scheme_ == 1)
  {
    if (rhs_gf_ != NULL) { ierr = VecDestroy(rhs_gf_); CHKERRXX(ierr); }
    ierr = VecDuplicate(rhs_, &rhs_gf_); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_gf_, &rhs_gf_ptr); CHKERRXX(ierr);
  }


  // interpolators
  interpolators_initialize();

  if (new_submat_main_)
  {
    //-------------------------------------------------------------------------------------
    // determine discretization type for each node
    //-------------------------------------------------------------------------------------
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_determine_node_types, 0, 0, 0, 0); CHKERRXX(ierr);
    node_scheme_.resize(nodes_->num_owned_indeps, UNDEFINED);
    int num_bdry_fvs = 0;
    int num_infc_fvs = 0;
//    int num_bdry_cart_points = 0;
//    int num_bdry_pieces = 0;
//    int num_infc_pieces = 0;
    int num_wall_pieces = 0;

    foreach_local_node(n, nodes_)
    {
      double bdry_phi_eff_000 = (bdry_.num_phi == 0) ? -1 : bdry_.phi_eff_ptr[n];
      double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];

      p4est_indep_t *ni   = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);

      discretization_scheme_t scheme = UNDEFINED;

      // check if at wall and simple BC
      if (is_node_Wall(p4est_, ni) && bdry_phi_eff_000 < 0.) {
        double xyz_C[P4EST_DIM];
        node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

        if (wc_type_->value(xyz_C) == DIRICHLET) {
          scheme = WALL_DIRICHLET;
        } else if ((wc_type_->value(xyz_C) == NEUMANN) && neumann_wall_first_order_) {
          scheme = WALL_NEUMANN;
        }
      }

      /* TODO:
       * - smarter check for intersecting level-sets
       * - count how many interface intersect a cell
       */
      if (scheme != WALL_DIRICHLET && scheme != WALL_NEUMANN) {
        bool is_ngbd_crossed_dirichlet = false;
        bool is_ngbd_crossed_neumann   = false;
        bool is_ngbd_crossed_immersed  = false;

        // check if neighbourhood is crossed by the domain boundary
        if (fabs(bdry_phi_eff_000) < lip_*diag_min_ && bdry_.num_phi != 0)
        {
          ngbd_->get_all_neighbors(n, neighbors, neighbors_exist);

          // sample level-set function at nodes of the extended cube and check if crossed
          for (unsigned short phi_idx = 0; phi_idx < bdry_.num_phi; ++phi_idx)
          {
            bool is_one_positive = false;
            bool is_one_negative = false;

            double limit = 0;

            if (bc_[phi_idx].type == DIRICHLET && dirichlet_scheme_ == 1) {
              limit = -gf_thresh_*diag_min_;
            }

            for (short i = 0; i < num_neighbors_cube; ++i)
              if (neighbors_exist[i])
              {
                is_one_positive = is_one_positive || bdry_.phi_ptr[phi_idx][neighbors[i]] >  limit;
                is_one_negative = is_one_negative || bdry_.phi_ptr[phi_idx][neighbors[i]] <= limit;
              }

            if (is_one_negative && is_one_positive)
            {
              switch (bc_[phi_idx].type)
              {
                case DIRICHLET: is_ngbd_crossed_dirichlet = true; break;
                case NEUMANN:   is_ngbd_crossed_neumann   = true; break;
                case ROBIN:     is_ngbd_crossed_neumann   = true; break;
                default: throw;
              }
            }
          }
        }

        // check is neighbourhood is crossed by the immersed interface
        if (fabs(infc_phi_eff_000) < lip_*diag_min_ && infc_.num_phi != 0)
        {
          ngbd_->get_all_neighbors(n, neighbors, neighbors_exist);

          // sample level-set function at nodes of the extended cube and check if crossed
          for (unsigned short phi_idx = 0; phi_idx < infc_.num_phi; ++phi_idx)
          {
            bool is_one_positive = false;
            bool is_one_negative = false;

            for (short i = 0; i < num_neighbors_cube; ++i)
              if (neighbors_exist[i])
              {
                is_one_positive = is_one_positive || infc_.phi_ptr[phi_idx][neighbors[i]] >= 0;
                is_one_negative = is_one_negative || infc_.phi_ptr[phi_idx][neighbors[i]] <  0;
              }

            if (is_one_negative && is_one_positive) is_ngbd_crossed_immersed = true;
          }
        }

        if (is_ngbd_crossed_neumann  && is_ngbd_crossed_dirichlet ||
            is_ngbd_crossed_neumann  && is_ngbd_crossed_immersed  ||
            is_ngbd_crossed_immersed && is_ngbd_crossed_dirichlet ) {
          throw std::domain_error("[CASL_ERROR]: No crossing of Dirichlet, Neumann and/or jump at the moment");
        }
        else if (is_ngbd_crossed_dirichlet) { scheme = BOUNDARY_DIRICHLET; }
        else if (is_ngbd_crossed_neumann)   { scheme = BOUNDARY_NEUMANN; }
        else if (is_ngbd_crossed_immersed)  { scheme = IMMERSED_INTERFACE; }
        else                                { scheme = DOMAIN_INSIDE; }
      }

      // points outside the domain
      if (scheme == DOMAIN_INSIDE && bdry_phi_eff_000 > 0) scheme = DOMAIN_OUTSIDE;

      // count how many finite volumes we will need
      if (scheme == BOUNDARY_NEUMANN)   num_bdry_fvs++;
      if (scheme == IMMERSED_INTERFACE) num_infc_fvs++;

      node_scheme_[n] = scheme;
    }
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_determine_node_types, 0, 0, 0, 0); CHKERRXX(ierr);

    wall_pieces_area    .reserve(num_wall_pieces);
    wall_pieces_id      .reserve(num_wall_pieces);
    wall_pieces_centroid.reserve(num_wall_pieces);

    //-------------------------------------------------------------------------------------
    // compute finite volumes
    //-------------------------------------------------------------------------------------
    if (store_finite_volumes_ && !finite_volumes_initialized_)
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_compute_finite_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

      bdry_node_to_fv_.resize(nodes_->num_owned_indeps, -1);
      infc_node_to_fv_.resize(nodes_->num_owned_indeps, -1);

      bdry_fvs_ = new std::vector< my_p4est_finite_volume_t >;
      infc_fvs_ = new std::vector< my_p4est_finite_volume_t >;

      bdry_fvs_->reserve(num_bdry_fvs);
      infc_fvs_->reserve(num_infc_fvs);

      my_p4est_finite_volume_t fv;

      foreach_local_node(n, nodes_)
      {
        switch (node_scheme_[n])
        {
          case DOMAIN_OUTSIDE:
          case DOMAIN_INSIDE:
          case WALL_DIRICHLET:
          case WALL_NEUMANN:
          case BOUNDARY_DIRICHLET: break;
          case BOUNDARY_NEUMANN:
            interpolators_prepare(n);
            construct_finite_volume(fv, n, p4est_, nodes_, bdry_phi_cf_, bdry_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);
            bdry_fvs_->push_back(fv);
            bdry_node_to_fv_[n] = bdry_fvs_->size()-1;
          break;
          case IMMERSED_INTERFACE:
            interpolators_prepare(n);
            construct_finite_volume(fv, n, p4est_, nodes_, infc_phi_cf_, infc_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);
            infc_fvs_->push_back(fv);
            infc_node_to_fv_[n] = infc_fvs_->size()-1;
          break;
        }
      }

      finite_volumes_initialized_ = true;
      finite_volumes_owned_       = true;

      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_compute_finite_volumes, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    //-------------------------------------------------------------------------------------
    // determine which nodes will be part of discretization (needed for superconvergent schemes)
    //-------------------------------------------------------------------------------------
    if ((fv_scheme_ == 1) && !volumes_computed_ && (there_is_neumann_ || there_is_robin_ || there_is_jump_))
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections, 0, 0, 0, 0); CHKERRXX(ierr);

      my_p4est_finite_volume_t fv;

      foreach_local_node(n, nodes_)
      {
        double bdry_phi_eff_000 = (bdry_.num_phi == 0) ? -1 : bdry_.phi_eff_ptr[n];
        double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];

        switch (node_scheme_[n])
        {
          case DOMAIN_OUTSIDE: areas_m_ptr[n] = 0; areas_p_ptr[n] = 0; break;
          case DOMAIN_INSIDE:
          case WALL_DIRICHLET:
          case WALL_NEUMANN:
          case BOUNDARY_DIRICHLET:
            if (infc_phi_eff_000 < 0) { areas_m_ptr[n] = 1; areas_p_ptr[n] = 0; }
            else                      { areas_m_ptr[n] = 0; areas_p_ptr[n] = 1; }
          break;
          case BOUNDARY_NEUMANN:
          {
            if (finite_volumes_initialized_) fv = bdry_fvs_->at(bdry_node_to_fv_[n]);
            else
            {
              interpolators_prepare(n);
              construct_finite_volume(fv, n, p4est_, nodes_, bdry_phi_cf_, bdry_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);
            }

            double face_area_max = 0;

            foreach_direction(i) face_area_max = MAX(fabs(fv.face_area[i]), face_area_max);

            if (infc_phi_eff_000 < 0) { areas_m_ptr[n] = face_area_max/face_area_scalling_; areas_p_ptr[n] = 0; }
            else                      { areas_p_ptr[n] = face_area_max/face_area_scalling_; areas_m_ptr[n] = 0; }
          }
          break;

          case IMMERSED_INTERFACE:
          {
            if (finite_volumes_initialized_) fv = infc_fvs_->at(infc_node_to_fv_[n]);
            else
            {
              interpolators_prepare(n);
              construct_finite_volume(fv, n, p4est_, nodes_, infc_phi_cf_, infc_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);
            }

            double face_area_max_m = 0;
            double face_area_max_p = 0;

            foreach_direction(i)
            {
              face_area_max_m = MAX(fabs(fv.face_area[i]), face_area_max_m);
              face_area_max_p = MAX(fabs(fv.full_face_area[i] - fv.face_area[i]), face_area_max_p);
            }

            areas_m_ptr[n] = face_area_max_m/face_area_scalling_;
            areas_p_ptr[n] = face_area_max_p/face_area_scalling_;
          }
          break;
        }
      }

      ierr = VecGhostUpdateBegin(areas_m_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (areas_m_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(areas_p_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (areas_p_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      volumes_computed_ = true;

      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections, 0, 0, 0, 0); CHKERRXX(ierr);
    }
  }

  // prepare stuff for ghost-fluid method for imposing dirichlet bc
  std::vector<int>    gf_map;
  std::vector<double> gf_nodes;
  std::vector<double> gf_phi;

  if (assembling_main && there_is_dirichlet_ && dirichlet_scheme_ == 1) {

    gf_map.assign(nodes_->num_owned_indeps, -1);

    my_p4est_interpolation_nodes_t interp(ngbd_);

    int ghost_nodes_count = 0;

    foreach_local_node (n, nodes_) {
      if (node_scheme_[n] == BOUNDARY_DIRICHLET) {
        if (bdry_.phi_eff_value(n) > -gf_thresh_*diag_min_) {

          const quad_neighbor_nodes_of_node_t &qnnn = (*ngbd_)[n];

          if (gf_is_ghost(qnnn)) { // check if a node is a ghost node

            // determine good direction for extrapolation
            p4est_locidx_t neighbors[num_neighbors_cube];
            ngbd_->get_all_neighbors(n, neighbors);

            double delta_xyz[P4EST_DIM];
            int    dir;
            gf_direction(qnnn, neighbors, dir, delta_xyz);

            // add points that are potentially nonlocal
            double xyz_c[P4EST_DIM];
            node_xyz_fr_n(n, p4est_, nodes_, xyz_c);

            for (int i = 2; i < gf_stencil_size(); ++i) {
              double xyz[P4EST_DIM] = { DIM( xyz_c[0] + double(i)*delta_xyz[0],
                                             xyz_c[1] + double(i)*delta_xyz[1],
                                             xyz_c[2] + double(i)*delta_xyz[2]) };
              interp.add_point(ghost_nodes_count*(gf_stencil_size()-2) + i-2, xyz);
            }

            gf_map[n] = ghost_nodes_count;
            ghost_nodes_count++;
          }
        }
      }
    }

    gf_nodes.resize(ghost_nodes_count*(gf_stencil_size()-2));
    gf_phi.resize(ghost_nodes_count*(gf_stencil_size()-2));

    // get values of level-set function
    interp.set_input(bdry_.phi_eff, linear);
    interp.interpolate(gf_phi.data());

    // get global node indices
    Vec     node_numbers;
    double *node_numbers_ptr;
    ierr = VecDuplicate(rhs_, &node_numbers); CHKERRXX(ierr);
    ierr = VecGetArray(node_numbers, &node_numbers_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes_) {
      node_numbers_ptr[n] = petsc_gloidx_[n];
    }

    ierr = VecRestoreArray(node_numbers, &node_numbers_ptr); CHKERRXX(ierr);

    interp.set_input(node_numbers, linear);
    interp.interpolate(gf_nodes.data());

    ierr = VecDestroy(node_numbers); CHKERRXX(ierr);
  }


  // pointers to rows for convenience
  std::vector<mat_entry_t> *row_main;
  std::vector<mat_entry_t> *row_robin_sc;
  std::vector<mat_entry_t> *row_jump;
  std::vector<mat_entry_t> *row_jump_ghost;
  std::vector<mat_entry_t> *row_gf;
  std::vector<mat_entry_t> *row_gf_ghost;


  //-------------------------------------------------------------------------------------
  // compute linear system
  //-------------------------------------------------------------------------------------
  if (assembling_main && setup_rhs)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
  } else if (assembling_main) {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_discretize_matrix, 0, 0, 0, 0); CHKERRXX(ierr);
  } else {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_discretize_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  foreach_local_node(n, nodes_)
  {
    row_main       = assembling_main       ? &entries_main      [n] : NULL;
    row_robin_sc   = assembling_robin_sc   ? &entries_robin_sc  [n] : NULL;
    row_jump       = assembling_jump       ? &entries_jump      [n] : NULL;
    row_jump_ghost = assembling_jump_ghost ? &entries_jump_ghost[n] : NULL;
    row_gf         = assembling_gf         ? &entries_gf        [n] : NULL;
    row_gf_ghost   = assembling_gf         ? &entries_gf_ghost  [n] : NULL;

    // Main node characteristics
    p4est_indep_t                       *ni   = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);
    const quad_neighbor_nodes_of_node_t  qnnn = ngbd_->get_neighbors(n);

    bool is_wall[P4EST_FACES];
    is_node_Wall(p4est_, ni, is_wall);

    double xyz_C[P4EST_DIM];
    node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

//    double DIM( x_C = xyz_C[0],
//                y_C = xyz_C[1],
//                z_C = xyz_C[2] );

    double bdry_phi_eff_000 = (bdry_.num_phi == 0) ? -1 : bdry_.phi_eff_ptr[n];
    double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];

    if (assembling_main)
    {
//      row_main->push_back(mat_entry_t(petsc_gloidx_[n], 0));
      mask_m_ptr[n] = MAX(mask_m_ptr[n], infc_phi_eff_000 < 0 ? -1. : 1.);
      mask_p_ptr[n] = MAX(mask_p_ptr[n], infc_phi_eff_000 < 0 ?  1. :-1.);
    }

    // discretize
    switch (node_scheme_[n])
    {
      case DOMAIN_OUTSIDE:
      {
        if (assembling_main) {
          row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
          mask_m_ptr[n] = mask_p_ptr[n] = 1.;
        }

        if (setup_rhs) {
          rhs_ptr[n] = 0;
        }

        break;
      }

      case DOMAIN_INSIDE:
      {
        discretize_inside(setup_rhs, n, qnnn,
                          infc_phi_eff_000, is_wall,
                          row_main, d_nnz_main[n], o_nnz_main[n]);
        break;
      }

      case WALL_DIRICHLET:
      {
        if (assembling_main) {
          row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
          if (bdry_phi_eff_000 < 0. || bdry_.num_phi == 0)
            matrix_has_nullspace_ = false;
        }

        if (setup_rhs) {
          rhs_ptr[n] = wc_value_->value(xyz_C);
        }

        break;
      }

      case WALL_NEUMANN:
      {
        foreach_direction(i)
        {
          if (is_wall[i])
          {
            if (assembling_main)
            {
              row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));

              if (bdry_phi_eff_000 < diag_min_ || bdry_.num_phi == 0)
              {
                p4est_locidx_t n_nei = qnnn.neighbor(i);
                row_main->push_back(mat_entry_t(petsc_gloidx_[n_nei], -1));
                (n_nei < nodes_->num_owned_indeps) ? ++d_nnz_main[n]:
                                                     ++o_nnz_main[n];
              }
            }

            if (setup_rhs) {
              rhs_ptr[n] = wc_value_->value(xyz_C)*qnnn.distance(i);
            }

            continue;
          }
        }
        break;
      }

      case BOUNDARY_DIRICHLET:
      {
        switch (dirichlet_scheme_) {
          case 0:
            discretize_dirichlet_sw(setup_rhs, n, qnnn,
                                    infc_phi_eff_000, is_wall,
                                    row_main, d_nnz_main[n], o_nnz_main[n]);
          break;
          case 1:
            discretize_dirichlet_gf(setup_rhs, n, qnnn,
                                    infc_phi_eff_000, is_wall,
                                    gf_map, gf_nodes, gf_phi,
                                    row_main, d_nnz_main[n], o_nnz_main[n],
                                    row_gf, d_nnz_gf[n], o_nnz_gf[n],
                                    row_gf_ghost, d_nnz_gf_ghost[n], o_nnz_gf_ghost[n]);
          break;
          default:
            throw std::invalid_argument("Unknown dirichlet scheme");
        }
        break;
      }

      case BOUNDARY_NEUMANN:
      {
        discretize_robin(setup_rhs, n, qnnn,
                         infc_phi_eff_000, is_wall,
                         row_main, d_nnz_main[n], o_nnz_main[n],
                         row_robin_sc, d_nnz_robin_sc[n], o_nnz_robin_sc[n]);
        break;
      }

      case IMMERSED_INTERFACE:
      {
        discretize_jump(setup_rhs, n, qnnn,
                        is_wall,
                        row_main, d_nnz_main[n], o_nnz_main[n],
                        row_jump, d_nnz_jump[n], o_nnz_jump[n],
                        row_jump_ghost, d_nnz_jump_ghost[n], o_nnz_jump_ghost[n]);
        break;
      }

      default: throw std::domain_error("Undetermined discretization scheme\n");
    }
  }

  // interpolators
  interpolators_finalize();

  if (new_submat_main_)
  {
    wall_pieces_map.compute_offsets();
  }

  if (assembling_main && setup_rhs)
  {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
  } else if (assembling_main) {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_discretize_matrix, 0, 0, 0, 0); CHKERRXX(ierr);
  } else {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_discretize_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  // restore pointers
  if (assembling_main)
  {
    ierr = VecRestoreArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(mask_m_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_m_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(mask_p_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_p_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (new_submat_diag_ && there_is_diag_)
  {
    ierr = VecRestoreArray(submat_diag_, &submat_diag_ptr); CHKERRXX(ierr);

    if (there_is_jump_)
    {
      ierr = VecRestoreArray(submat_diag_ghost_, &submat_diag_ghost_ptr); CHKERRXX(ierr);
    }
  }

  if (var_diag_)
  {
    ierr = VecRestoreArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);
  }

  if (new_submat_robin_ && (fv_scheme_ == 0) && there_is_robin_)
  {
    ierr = VecRestoreArray(submat_robin_sym_, &submat_robin_sym_ptr); CHKERRXX(ierr);
  }

  bdry_.restore_arrays();
  infc_.restore_arrays();

  if (var_mu_)
  {
    _CODE( ierr = VecRestoreArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr); );
    _CODE( ierr = VecRestoreArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr); );

    XCODE( ierr = VecRestoreArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr); );
    XCODE( ierr = VecRestoreArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr); );

    YCODE( ierr = VecRestoreArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr); );
    YCODE( ierr = VecRestoreArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr); );

    ZCODE( ierr = VecRestoreArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr); );
    ZCODE( ierr = VecRestoreArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr); );
  }

  if (setup_rhs)
  {
    ierr = VecRestoreArray(rhs_, &rhs_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);

//    ierr = VecGhostUpdateBegin(rhs_, ADD_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd  (rhs_, ADD_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (there_is_neumann_ || there_is_robin_ || there_is_jump_)
  {
    ierr = VecRestoreArray(volumes_m_, &volumes_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(volumes_p_, &volumes_p_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(areas_m_, &areas_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(areas_p_, &areas_p_ptr); CHKERRXX(ierr);
  }

  if (setup_rhs && there_is_jump_)
  {
    ierr = VecRestoreArray(rhs_jump_, &rhs_jump_ptr); CHKERRXX(ierr);
  }

  if (setup_rhs && there_is_dirichlet_ && dirichlet_scheme_ == 1)
  {
    ierr = VecRestoreArray(rhs_gf_, &rhs_gf_ptr); CHKERRXX(ierr);
  }

  if (assembling_main)
  {
    if (there_is_dirichlet_ && dirichlet_scheme_ == 1)
    {
//      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_assemble_submat_jump, 0, 0, 0, 0); CHKERRXX(ierr);

      // assemble sub matrices
      assemble_matrix(entries_gf, d_nnz_gf, o_nnz_gf, &submat_gf_);
      assemble_matrix(entries_gf_ghost, d_nnz_gf_ghost, o_nnz_gf_ghost, &submat_gf_ghost_);

//      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_assemble_submat_jump, 0, 0, 0, 0); CHKERRXX(ierr);
//      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_correct_submat_main_jump, 0, 0, 0, 0); CHKERRXX(ierr);

      // compute correction doing matrix product
      Mat mat_product;
      ierr = MatMatMult(submat_gf_, submat_gf_ghost_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mat_product); CHKERRXX(ierr);

      // add jump correction matrix to main matrix
      // variant 1: use PETSc tools (quite slow)
      /*
        ierr = MatAYPX(submat_main_, 1., mat_product, DIFFERENT_NONZERO_PATTERN); CHKERRXX(ierr); //*/

      // variant 2: do it explicitly by hands taking into account that
      // number of nonzero rows in mat_product is much less than the total number of rows
      //*
      PetscInt            offset = global_node_offset_[p4est_->mpirank];
      PetscInt            N;
      PetscBool           done;
      const PetscInt     *ia;
      const PetscInt     *ja;
      Mat                 mat_product_local;
      PetscScalar        *data;
      mat_entry_t         ent;
      std::set<PetscInt>  d_elems;
      std::set<PetscInt>  o_elems;
      PetscInt            cur;

      // get access to matrix structure of matrix product
      if (p4est_->mpisize > 1)
      {
        ierr = MatMPIAIJGetLocalMat(mat_product, MAT_INITIAL_MATRIX, &mat_product_local); CHKERRXX(ierr);
      } else {
        mat_product_local = mat_product;
      }
      ierr = MatGetRowIJ(mat_product_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
      ierr = MatSeqAIJGetArray(mat_product_local, &data); CHKERRXX(ierr);

      // iterate through nonzero rows of mat_product
      foreach_local_node(n, nodes_)
      {
        if (node_scheme_[n] == BOUNDARY_DIRICHLET)
        {
          if (ia[n+1] > ia[n])
          {
            // for better allocations we will recount exact numbers of diagonal and off-diagonal elements
            // using std::set's
            d_elems.clear();
            o_elems.clear();

            //
            for (int i = 0; i < entries_main[n].size(); ++i)
            {
              cur = entries_main[n][i].n;

              if (cur < offset || cur >= offset + nodes_->num_owned_indeps)
                o_elems.insert(cur);
              else
                d_elems.insert(cur);
            }

            for (int i = ia[n]; i < ia[n+1]; ++i)
            {
              // add element from mat_product to submat_main
              if (fabs(data[i]) > EPS)
              {
                ent.n   = ja[i];
                ent.val = data[i];
                entries_main[n].push_back(ent);

                //
                if (ent.n < offset || ent.n >= offset + nodes_->num_owned_indeps)
                  o_elems.insert(ent.n);
                else
                  d_elems.insert(ent.n);
              }
            }

            d_nnz_main[n] = d_elems.size();
            o_nnz_main[n] = o_elems.size();
          }
        }
      }

      // restore matrix structure
      ierr = MatSeqAIJRestoreArray(mat_product_local, &data); CHKERRXX(ierr);
      ierr = MatRestoreRowIJ(mat_product_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
      if (p4est_->mpisize > 1)
      {
        ierr = MatDestroy(mat_product_local); CHKERRXX(ierr);
      }
      //*/

      ierr = MatDestroy(mat_product); CHKERRXX(ierr);

//      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_correct_submat_main_jump, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    if (there_is_jump_)
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_assemble_submat_jump, 0, 0, 0, 0); CHKERRXX(ierr);
      assemble_matrix(entries_jump, d_nnz_jump, o_nnz_jump, &submat_jump_);
      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_assemble_submat_jump, 0, 0, 0, 0); CHKERRXX(ierr);

      if (there_is_jump_mu_)
      {
        ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_correct_submat_main_jump, 0, 0, 0, 0); CHKERRXX(ierr);

        Mat mat_product;

        // assemble matrix that expresses additional degrees of freedom (ghost values) through regular node values
        assemble_matrix(entries_jump_ghost, d_nnz_jump_ghost, o_nnz_jump_ghost, &submat_jump_ghost_);

        // compute correction doing matrix product
        ierr = MatMatMult(submat_jump_, submat_jump_ghost_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mat_product); CHKERRXX(ierr);

        // add jump correction matrix to main matrix
        // variant 1: use PETSc tools (quite slow)
        /*
        ierr = MatAYPX(submat_main_, 1., mat_product, DIFFERENT_NONZERO_PATTERN); CHKERRXX(ierr); //*/

        // variant 2: do it explicitly by hands taking into account that
        // number of nonzero rows in mat_product is much less than the total number of rows
        //*
        PetscInt            offset = global_node_offset_[p4est_->mpirank];
        PetscInt            N;
        PetscBool           done;
        const PetscInt     *ia;
        const PetscInt     *ja;
        Mat                 mat_product_local;
        PetscScalar        *data;
        mat_entry_t         ent;
        std::set<PetscInt>  d_elems;
        std::set<PetscInt>  o_elems;
        PetscInt            cur;

        // get access to matrix structure of matrix product
        if (p4est_->mpisize > 1)
        {
          ierr = MatMPIAIJGetLocalMat(mat_product, MAT_INITIAL_MATRIX, &mat_product_local); CHKERRXX(ierr);
        } else {
          mat_product_local = mat_product;
        }
        ierr = MatGetRowIJ(mat_product_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
        ierr = MatSeqAIJGetArray(mat_product_local, &data); CHKERRXX(ierr);

        // iterate through nonzero rows of mat_product
        foreach_local_node(n, nodes_)
        {
          if (node_scheme_[n] == IMMERSED_INTERFACE)
          {
            if (ia[n+1] > ia[n])
            {
              // for better allocations we will recount exact numbers of diagonal and off-diagonal elements
              // using std::set's
              d_elems.clear();
              o_elems.clear();

              //
              for (int i = 0; i < entries_main[n].size(); ++i)
              {
                cur = entries_main[n][i].n;

                if (cur < offset || cur >= offset + nodes_->num_owned_indeps)
                  o_elems.insert(cur);
                else
                  d_elems.insert(cur);
              }

              for (int i = ia[n]; i < ia[n+1]; ++i)
              {
                // add element from mat_product to submat_main
                if (fabs(data[i]) > EPS)
                {
                  ent.n   = ja[i];
                  ent.val = data[i];
                  entries_main[n].push_back(ent);

                  //
                  if (ent.n < offset || ent.n >= offset + nodes_->num_owned_indeps)
                    o_elems.insert(ent.n);
                  else
                    d_elems.insert(ent.n);
                }
              }

              d_nnz_main[n] = d_elems.size();
              o_nnz_main[n] = o_elems.size();
            }
          }
        }

        // restore matrix structure
        ierr = MatSeqAIJRestoreArray(mat_product_local, &data); CHKERRXX(ierr);
        ierr = MatRestoreRowIJ(mat_product_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
        if (p4est_->mpisize > 1)
        {
          ierr = MatDestroy(mat_product_local); CHKERRXX(ierr);
        }
        //*/

        ierr = MatDestroy(mat_product); CHKERRXX(ierr);

        ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_correct_submat_main_jump, 0, 0, 0, 0); CHKERRXX(ierr);
      }
    }

    // explicitly add zeros in place of elements from robin bc to speed up matrix sum later
    if (there_is_robin_ && (fv_scheme_ == 1))
    {
      PetscInt           offset = global_node_offset_[p4est_->mpirank];
      PetscInt           cur;
      mat_entry_t        ent;
      std::set<PetscInt> d_elems;
      std::set<PetscInt> o_elems;

      ent.val = 0;

      foreach_local_node(n, nodes_)
      {
        if (node_scheme_[n] == BOUNDARY_NEUMANN)
        {
          if (entries_robin_sc[n].size() > 0)
          {
            d_elems.clear();
            o_elems.clear();

            for (int i = 0; i < entries_main[n].size(); ++i)
            {
              cur = entries_main[n][i].n;

              if (cur < offset || cur >= offset + nodes_->num_owned_indeps)
                o_elems.insert(cur);
              else
                d_elems.insert(cur);
            }

            for (int i = 0; i < entries_robin_sc[n].size(); ++i)
            {
              ent.n = entries_robin_sc[n][i].n;
              entries_main[n].push_back(ent);

              if (ent.n < offset || ent.n >= offset + nodes_->num_owned_indeps)
                o_elems.insert(ent.n);
              else
                d_elems.insert(ent.n);
            }

            d_nnz_main[n] = d_elems.size();
            o_nnz_main[n] = o_elems.size();
          }
        }
      }
    }

    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_assemble_submat_main, 0, 0, 0, 0); CHKERRXX(ierr);
    assemble_matrix(entries_main, d_nnz_main, o_nnz_main, &submat_main_);
    ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_mls_assemble_submat_main, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  A_needs_reassembly_ = A_needs_reassembly_ || (new_submat_main_ || (new_submat_diag_ && there_is_diag_) || (new_submat_robin_ && there_is_robin_));

  new_submat_diag_  = false;
  new_submat_main_  = false;
  new_submat_robin_ = false;

  // compute resulting matrix (if needed)
  if (A_needs_reassembly_ && setup_rhs)
  {
    A_needs_reassembly_ = false;
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_assemble_matrix, 0, 0, 0, 0); CHKERRXX(ierr);

    // start by cloning submat_main_ into A_
    if (A_ != NULL) { ierr = MatDestroy(A_); CHKERRXX(ierr); }
    //    ierr = MatConvert(submat_main_, MATSAME, MAT_INITIAL_MATRIX, &A_); CHKERRXX(ierr);
    ierr = MatDuplicate(submat_main_, MAT_COPY_VALUES, &A_); CHKERRXX(ierr);

    // add correction from linear term
    if (there_is_diag_)
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_add_submat_diag, 0, 0, 0, 0); CHKERRXX(ierr);
      ierr = MatDiagonalSet(A_, submat_diag_, ADD_VALUES); CHKERRXX(ierr);

      if (there_is_jump_ && there_is_jump_mu_)
      {
        PetscInt            N;
        PetscBool           done;
        const PetscInt     *ia;
        const PetscInt     *ja;
        PetscScalar        *data;

        ierr = VecGetArray(submat_diag_ghost_, &submat_diag_ghost_ptr); CHKERRXX(ierr);

        // get access to matrix structure of matrix product
        Mat submat_jump_ghost_local = submat_jump_ghost_;
        if (p4est_->mpisize > 1)
        {
          ierr = MatMPIAIJGetLocalMat(submat_jump_ghost_, MAT_INITIAL_MATRIX, &submat_jump_ghost_local); CHKERRXX(ierr);
        }
        ierr = MatGetRowIJ(submat_jump_ghost_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
        ierr = MatSeqAIJGetArray(submat_jump_ghost_local, &data); CHKERRXX(ierr);

        std::vector<PetscInt>    columns;
        std::vector<PetscScalar> values;

        foreach_local_node(n, nodes_)
        {
          if (node_scheme_[n] == IMMERSED_INTERFACE)
          {
            if (ia[n+1] > ia[n])
            {
              PetscInt n_gl = petsc_gloidx_[n];

              columns.clear();
              values.clear();
              for (int i = ia[n]; i < ia[n+1]; ++i)
              {
                double value = submat_diag_ghost_ptr[n]*data[i];
                if (fabs(value) > EPS)
                {
                  columns.push_back(ja[i]);
                  values.push_back(value);
                }
              }
              ierr = MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);  CHKERRXX(ierr);
              ierr = MatSetValues(A_, 1, &n_gl, columns.size(), columns.data(), values.data(), ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }

        ierr = VecRestoreArray(submat_diag_ghost_, &submat_diag_ghost_ptr); CHKERRXX(ierr);

        ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
        ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

        // restore matrix structure
        ierr = MatSeqAIJRestoreArray(submat_jump_ghost_local, &data); CHKERRXX(ierr);
        ierr = MatRestoreRowIJ(submat_jump_ghost_local, 0, PETSC_FALSE, PETSC_FALSE, &N, &ia, &ja, &done); CHKERRXX(ierr);
        if (p4est_->mpisize > 1)
        {
          ierr = MatDestroy(submat_jump_ghost_local); CHKERRXX(ierr);
        }
      }
      ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_mls_add_submat_diag, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // add correction for Robin b.c.
    if (there_is_robin_)
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_add_submat_robin, 0, 0, 0, 0); CHKERRXX(ierr);

      if (fv_scheme_ == 0)
      {
        // if symmetric scheme is used then the only correction is to diagonal elements
        ierr = MatDiagonalSet(A_, submat_robin_sym_, ADD_VALUES); CHKERRXX(ierr);
      }
      else if (fv_scheme_ == 1)
      {
        // if non-symmetric scheme is used then

        // variant 1: create a PETSc Mat that represents the correction and use PETSc tool to add up matrices.
        // It turns out to be rather slow.
        //if (new_submat_robin_)
        //{
        //  assemble_matrix(entries_robin_sc, d_nnz_robin_sc, o_nnz_robin_sc, &submat_robin_sc_);
        //}
//        ierr = MatAXPY(A_, 1., submat_robin_sc_, SUBSET_NONZERO_PATTERN); CHKERRXX(ierr);

        // variant 2: add new elements explicitly by hands taking into account that only small number of rows are affected
        std::vector<PetscInt>    columns;
        std::vector<PetscScalar> values;

        foreach_local_node(n, nodes_)
        {
          if (node_scheme_[n] == BOUNDARY_NEUMANN)
          {
            PetscInt                  n_gl = petsc_gloidx_[n];
            std::vector<mat_entry_t> *row  = &entries_robin_sc[n];

            columns.clear();
            values.clear();
            for (int m=0; m < row->size(); ++m)
            {
              columns.push_back(row->at(m).n);
              values.push_back(row->at(m).val);
            }

            if (row->size() > 0)
            {
              ierr = MatSetValues(A_, 1, &n_gl, row->size(), columns.data(), values.data(), ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }

        ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
        ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
      }
      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_add_submat_robin, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    // get diagonal scaling of the resulting matrix
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling, 0, 0, 0, 0); CHKERRXX(ierr);
    if (diag_scaling_ != NULL) { ierr = VecDestroy(diag_scaling_); CHKERRXX(ierr); }
    ierr = VecDuplicate(rhs_, &diag_scaling_); CHKERRXX(ierr);
    ierr = MatGetDiagonal(A_, diag_scaling_); CHKERRXX(ierr);
    ierr = VecReciprocalGhost(diag_scaling_); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling, 0, 0, 0, 0); CHKERRXX(ierr);

    // scale the matrix by its diagonal (to speed up inversion)
    if (enfornce_diag_scaling_)
    {
      ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
      ierr = MatDiagonalScale(A_, diag_scaling_, NULL); CHKERRXX(ierr);
      ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_assemble_matrix, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs && there_is_jump_)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_correct_rhs_jump, 0, 0, 0, 0); CHKERRXX(ierr);

    // contribution from Laplace term
    ierr = VecScaleGhost(rhs_jump_, -1); CHKERRXX(ierr);
    ierr = MatMultAdd(submat_jump_, rhs_jump_, rhs_, rhs_); CHKERRXX(ierr);

    // contribution from linear term
    Vec rhs_jump_tmp;
    ierr = VecDuplicate(rhs_, &rhs_jump_tmp); CHKERRXX(ierr);
    ierr = VecPointwiseMult(rhs_jump_tmp, rhs_jump_, submat_diag_ghost_); CHKERRXX(ierr);
    ierr = VecAXPY(rhs_, 1., rhs_jump_tmp); CHKERRXX(ierr);

    ierr = VecDestroy(rhs_jump_tmp); CHKERRXX(ierr);

    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_correct_rhs_jump, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs && there_is_dirichlet_ && dirichlet_scheme_ == 1)
  {
//    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_correct_rhs_jump, 0, 0, 0, 0); CHKERRXX(ierr);

    // contribution from Laplace term
//    ierr = VecScaleGhost(rhs_gf_, -1); CHKERRXX(ierr);
    ierr = MatMultAdd(submat_gf_, rhs_gf_, rhs_, rhs_); CHKERRXX(ierr);

    // contribution from linear term

//    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_correct_rhs_jump, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs && enfornce_diag_scaling_)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = VecPointwiseMult(rhs_, rhs_, diag_scaling_); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
  }

  // check for null space
  // FIXME: the return value should be checked for errors ...
  if (new_submat_main_ || new_submat_diag_ || new_submat_robin_) {
    MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace_, 1, MPI_INT, MPI_LAND, p4est_->mpicomm);
  }

//  if (matrix_has_nullspace) {
//    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
//    p4est_gloidx_t fixed_value_idx;
//    MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
//    if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
//      fixed_value_idx_l = -1;
//      fixed_value_idx_g = fixed_value_idx;
//    } else {
//      // reset the value
//      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
//    }
//  }
}

void my_p4est_poisson_nodes_mls_t::assemble_matrix(std::vector< std::vector<mat_entry_t> > &entries, std::vector<int> &d_nnz, std::vector<int> &o_nnz, Mat *matrix)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_assemble_submatrix, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

  if (*matrix != NULL) { ierr = MatDestroy(*matrix); CHKERRXX(ierr); }

  /* set up the matrix */
  ierr = MatCreate(p4est_->mpicomm, matrix); CHKERRXX(ierr);
  ierr = MatSetType(*matrix, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(*matrix, num_owned_local , num_owned_local, num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(*matrix); CHKERRXX(ierr);

  /* allocate the matrix */
  ierr = MatSeqAIJSetPreallocation(*matrix, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(*matrix, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  /* fill the matrix with the values */
  std::vector<PetscInt>    columns;
  std::vector<PetscScalar> values;

  foreach_local_node(n, nodes_)
  {
    std::vector<mat_entry_t> *row  = &entries[n];
    if (row->size() > 0) {
      PetscInt n_gl = petsc_gloidx_[n];

      //    for (int m=0; m < row->size(); ++m)
      //    {
      //      ierr = MatSetValue(*matrix, n_gl, row->at(m).n, row->at(m).val, ADD_VALUES); CHKERRXX(ierr);
      //    }

      columns.clear();
      values.clear();
      for (int m=0; m < row->size(); ++m)
      {
        columns.push_back(row->at(m).n);
        values.push_back(row->at(m).val);
      }

      ierr = MatSetValues(*matrix, 1, &n_gl, row->size(), columns.data(), values.data(), ADD_VALUES); CHKERRXX(ierr);
    }
  }

  /* assemble the matrix */
  ierr = MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (*matrix, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_assemble_submatrix, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::find_projection(const quad_neighbor_nodes_of_node_t& qnnn, const double *phi_ptr, double dxyz_pr[], double &dist_pr, double normal[])
{
  // find projection point
  double phi_d[P4EST_DIM] = { DIM(0,0,0) };

  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, qnnn.node_000);

  // check if the node is a wall node
  bool DIM( xm_wall = is_node_xmWall(p4est_, ni),
            ym_wall = is_node_ymWall(p4est_, ni),
            zm_wall = is_node_zmWall(p4est_, ni) );

  bool DIM( xp_wall = is_node_xpWall(p4est_, ni),
            yp_wall = is_node_ypWall(p4est_, ni),
            zp_wall = is_node_zpWall(p4est_, ni) );

  if (!xm_wall && !xp_wall) phi_d[0] = qnnn.dx_central(phi_ptr);
  else if (!xm_wall)        phi_d[0] = qnnn.dx_backward_linear(phi_ptr);
  else if (!xp_wall)        phi_d[0] = qnnn.dx_forward_linear(phi_ptr);

  if (!ym_wall && !yp_wall) phi_d[1] = qnnn.dy_central(phi_ptr);
  else if (!ym_wall)        phi_d[1] = qnnn.dy_backward_linear(phi_ptr);
  else if (!yp_wall)        phi_d[1] = qnnn.dy_forward_linear(phi_ptr);
#ifdef P4_TO_P8
  if (!zm_wall && !zp_wall) phi_d[2] = qnnn.dz_central(phi_ptr);
  else if (!zm_wall)        phi_d[2] = qnnn.dz_backward_linear(phi_ptr);
  else if (!zp_wall)        phi_d[2] = qnnn.dz_forward_linear(phi_ptr);
#endif

  double phi_d_norm = sqrt( SUMD(SQR(phi_d[0]),SQR(phi_d[1]),SQR(phi_d[2])) );

  dist_pr = phi_ptr[qnnn.node_000]/phi_d_norm;

  foreach_dimension(dim) {
    phi_d[dim]  /=  phi_d_norm;
    dxyz_pr[dim] = -dist_pr*phi_d[dim];
  }

  if (normal != NULL) {
    foreach_dimension(dim) {
      normal[dim] = phi_d[dim];
    }
  }
}

bool my_p4est_poisson_nodes_mls_t::inv_mat2(const double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];

  if (det == 0) return false;

  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;

  return true;
}

bool my_p4est_poisson_nodes_mls_t::inv_mat3(const double *in, double *out)
{
  double det = in[3*0+0]*(in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2]) -
               in[3*0+1]*(in[3*1+0]*in[3*2+2] - in[3*1+2]*in[3*2+0]) +
               in[3*0+2]*(in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1]);

  if (det == 0) return false;

  out[3*0+0] = (in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2])/det;
  out[3*0+1] = (in[3*0+2]*in[3*2+1] - in[3*2+2]*in[3*0+1])/det;
  out[3*0+2] = (in[3*0+1]*in[3*1+2] - in[3*1+1]*in[3*0+2])/det;

  out[3*1+0] = (in[3*1+2]*in[3*2+0] - in[3*2+2]*in[3*1+0])/det;
  out[3*1+1] = (in[3*0+0]*in[3*2+2] - in[3*2+0]*in[3*0+2])/det;
  out[3*1+2] = (in[3*0+2]*in[3*1+0] - in[3*1+2]*in[3*0+0])/det;

  out[3*2+0] = (in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1])/det;
  out[3*2+1] = (in[3*0+1]*in[3*2+0] - in[3*2+1]*in[3*0+0])/det;
  out[3*2+2] = (in[3*0+0]*in[3*1+1] - in[3*1+0]*in[3*0+1])/det;

  return true;
}

bool my_p4est_poisson_nodes_mls_t::inv_mat4(const double *in, double *out)
{
    double inv[16], det;
    int i;

    inv[0] = in[5]  * in[10] * in[15] -
             in[5]  * in[11] * in[14] -
             in[9]  * in[6]  * in[15] +
             in[9]  * in[7]  * in[14] +
             in[13] * in[6]  * in[11] -
             in[13] * in[7]  * in[10];

    inv[4] = -in[4]  * in[10] * in[15] +
              in[4]  * in[11] * in[14] +
              in[8]  * in[6]  * in[15] -
              in[8]  * in[7]  * in[14] -
              in[12] * in[6]  * in[11] +
              in[12] * in[7]  * in[10];

    inv[8] = in[4]  * in[9] * in[15] -
             in[4]  * in[11] * in[13] -
             in[8]  * in[5] * in[15] +
             in[8]  * in[7] * in[13] +
             in[12] * in[5] * in[11] -
             in[12] * in[7] * in[9];

    inv[12] = -in[4]  * in[9] * in[14] +
               in[4]  * in[10] * in[13] +
               in[8]  * in[5] * in[14] -
               in[8]  * in[6] * in[13] -
               in[12] * in[5] * in[10] +
               in[12] * in[6] * in[9];

    inv[1] = -in[1]  * in[10] * in[15] +
              in[1]  * in[11] * in[14] +
              in[9]  * in[2] * in[15] -
              in[9]  * in[3] * in[14] -
              in[13] * in[2] * in[11] +
              in[13] * in[3] * in[10];

    inv[5] = in[0]  * in[10] * in[15] -
             in[0]  * in[11] * in[14] -
             in[8]  * in[2] * in[15] +
             in[8]  * in[3] * in[14] +
             in[12] * in[2] * in[11] -
             in[12] * in[3] * in[10];

    inv[9] = -in[0]  * in[9] * in[15] +
              in[0]  * in[11] * in[13] +
              in[8]  * in[1] * in[15] -
              in[8]  * in[3] * in[13] -
              in[12] * in[1] * in[11] +
              in[12] * in[3] * in[9];

    inv[13] = in[0]  * in[9] * in[14] -
              in[0]  * in[10] * in[13] -
              in[8]  * in[1] * in[14] +
              in[8]  * in[2] * in[13] +
              in[12] * in[1] * in[10] -
              in[12] * in[2] * in[9];

    inv[2] = in[1]  * in[6] * in[15] -
             in[1]  * in[7] * in[14] -
             in[5]  * in[2] * in[15] +
             in[5]  * in[3] * in[14] +
             in[13] * in[2] * in[7] -
             in[13] * in[3] * in[6];

    inv[6] = -in[0]  * in[6] * in[15] +
              in[0]  * in[7] * in[14] +
              in[4]  * in[2] * in[15] -
              in[4]  * in[3] * in[14] -
              in[12] * in[2] * in[7] +
              in[12] * in[3] * in[6];

    inv[10] = in[0]  * in[5] * in[15] -
              in[0]  * in[7] * in[13] -
              in[4]  * in[1] * in[15] +
              in[4]  * in[3] * in[13] +
              in[12] * in[1] * in[7] -
              in[12] * in[3] * in[5];

    inv[14] = -in[0]  * in[5] * in[14] +
               in[0]  * in[6] * in[13] +
               in[4]  * in[1] * in[14] -
               in[4]  * in[2] * in[13] -
               in[12] * in[1] * in[6] +
               in[12] * in[2] * in[5];

    inv[3] = -in[1] * in[6] * in[11] +
              in[1] * in[7] * in[10] +
              in[5] * in[2] * in[11] -
              in[5] * in[3] * in[10] -
              in[9] * in[2] * in[7] +
              in[9] * in[3] * in[6];

    inv[7] = in[0] * in[6] * in[11] -
             in[0] * in[7] * in[10] -
             in[4] * in[2] * in[11] +
             in[4] * in[3] * in[10] +
             in[8] * in[2] * in[7] -
             in[8] * in[3] * in[6];

    inv[11] = -in[0] * in[5] * in[11] +
               in[0] * in[7] * in[9] +
               in[4] * in[1] * in[11] -
               in[4] * in[3] * in[9] -
               in[8] * in[1] * in[7] +
               in[8] * in[3] * in[5];

    inv[15] = in[0] * in[5] * in[10] -
              in[0] * in[6] * in[9] -
              in[4] * in[1] * in[10] +
              in[4] * in[2] * in[9] +
              in[8] * in[1] * in[6] -
              in[8] * in[2] * in[5];

    det = in[0] * inv[0] + in[1] * inv[4] + in[2] * inv[8] + in[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        out[i] = inv[i] * det;

    return true;
}

#ifdef P4_TO_P8
double my_p4est_poisson_nodes_mls_t::compute_weights_through_face(double A, double B, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face)
{

  bool semi_fallback = false;
  bool full_fallback = false;

  map_face[nnf_mm] = false;
  map_face[nnf_0m] = false;
  map_face[nnf_pm] = false;
  map_face[nnf_m0] = false;
  map_face[nnf_00] = true; weights_face[nnf_00] = 1;
  map_face[nnf_p0] = false;
  map_face[nnf_mp] = false;
  map_face[nnf_0p] = false;
  map_face[nnf_pp] = false;

  double a = fabs(A);
  double b = fabs(B);

  if (a > .5 || b > .5) std::cout << "Warning: face's centroid falls outside the face!\n";

  double mask_specific = -1;

  int num_good_neighbors = 0;

  for (int i = 0; i < 9; ++i)
    if (neighbor_exists_face[i]) num_good_neighbors++;

  bool same_line = (num_good_neighbors == 3 &&
                    ( (neighbor_exists_face[nnf_m0] && neighbor_exists_face[nnf_p0]) ||
                      (neighbor_exists_face[nnf_0m] && neighbor_exists_face[nnf_0p]) ||
                      (neighbor_exists_face[nnf_mm] && neighbor_exists_face[nnf_pp]) ||
                      (neighbor_exists_face[nnf_mp] && neighbor_exists_face[nnf_pm]) ) );

  if (a < theta && b < theta)
  {
    map_face[nnf_00] = true;  weights_face[nnf_00] = 1;
    mask_specific = -2;
  }




  else if (A <= 0 && B <= 0 && B <= A &&
           neighbor_exists_face[nnf_0m] &&
           neighbor_exists_face[nnf_mm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-b);
    map_face[nnf_0m] = true; weights_face[nnf_0m] = b-a;
    map_face[nnf_mm] = true; weights_face[nnf_mm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B <= 0 && B >= A &&
           neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_mm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-a);
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a-b;
    map_face[nnf_mm] = true; weights_face[nnf_mm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B <= 0 && B <= -A &&
           neighbor_exists_face[nnf_0m] &&
           neighbor_exists_face[nnf_pm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-b);
    map_face[nnf_0m] = true; weights_face[nnf_0m] = b-a;
    map_face[nnf_pm] = true; weights_face[nnf_pm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B <= 0 && B >= -A &&
           neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_pm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-a);
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a-b;
    map_face[nnf_pm] = true; weights_face[nnf_pm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B >= 0 && B <= -A &&
           neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_mp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-a);
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a-b;
    map_face[nnf_mp] = true; weights_face[nnf_mp] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B >= 0 && B >= -A &&
           neighbor_exists_face[nnf_0p] &&
           neighbor_exists_face[nnf_mp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-b);
    map_face[nnf_0p] = true; weights_face[nnf_0p] = b-a;
    map_face[nnf_mp] = true; weights_face[nnf_mp] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B >= 0 && B <= A &&
           neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_pp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-a);
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a-b;
    map_face[nnf_pp] = true; weights_face[nnf_pp] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B >= 0 && B >= A &&
           neighbor_exists_face[nnf_0p] &&
           neighbor_exists_face[nnf_pp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-b);
    map_face[nnf_0p] = true; weights_face[nnf_0p] = b-a;
    map_face[nnf_pp] = true; weights_face[nnf_pp] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }

  else if ( neighbor_exists_face[nnf_m0] &&
            neighbor_exists_face[nnf_0m] &&
            neighbor_exists_face[nnf_mm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] =(1.+A)*(1.+B);
    map_face[nnf_m0] = true; weights_face[nnf_m0] =(  -A)*(1.+B);
    map_face[nnf_0m] = true; weights_face[nnf_0m] =(1.+A)*(  -B);
    map_face[nnf_mm] = true; weights_face[nnf_mm] =(  -A)*(  -B);
  }
  else if ( neighbor_exists_face[nnf_p0] &&
            neighbor_exists_face[nnf_0m] &&
            neighbor_exists_face[nnf_pm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-A)*(1.+B);
    map_face[nnf_p0] = true; weights_face[nnf_p0] =(   A)*(1.+B);
    map_face[nnf_0m] = true; weights_face[nnf_0m] =(1.-A)*(  -B);
    map_face[nnf_pm] = true; weights_face[nnf_pm] =(   A)*(  -B);
  }
  else if ( neighbor_exists_face[nnf_m0] &&
            neighbor_exists_face[nnf_0p] &&
            neighbor_exists_face[nnf_mp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] =(1.+A)*(1.-B);
    map_face[nnf_m0] = true; weights_face[nnf_m0] =(  -A)*(1.-B);
    map_face[nnf_0p] = true; weights_face[nnf_0p] =(1.+A)*(   B);
    map_face[nnf_mp] = true; weights_face[nnf_mp] =(  -A)*(   B);
  }
  else if ( neighbor_exists_face[nnf_p0] &&
            neighbor_exists_face[nnf_0p] &&
            neighbor_exists_face[nnf_pp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-A)*(1.-B);
    map_face[nnf_p0] = true; weights_face[nnf_p0] =(   A)*(1.-B);
    map_face[nnf_0p] = true; weights_face[nnf_0p] =(1.-A)*(   B);
    map_face[nnf_pp] = true; weights_face[nnf_pp] =(   A)*(   B);
  }



//  else if ( A <= 0 &&
//            B <= 0 &&
//            neighbor_exists_face[nnf_m0] &&
//            neighbor_exists_face[nnf_0m] &&
//            neighbor_exists_face[nnf_mm] )
//  {
//    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-a)*(1.-b);
//    map_face[nnf_m0] = true; weights_face[nnf_m0] =    a *(1.-b);
//    map_face[nnf_0m] = true; weights_face[nnf_0m] =(1.-a)*    b ;
//    map_face[nnf_mm] = true; weights_face[nnf_mm] =    a *    b ;
//  }
//  else if ( A >= 0 &&
//            B <= 0 &&
//            neighbor_exists_face[nnf_p0] &&
//            neighbor_exists_face[nnf_0m] &&
//            neighbor_exists_face[nnf_pm] )
//  {
//    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-a)*(1.-b);
//    map_face[nnf_p0] = true; weights_face[nnf_p0] =    a *(1.-b);
//    map_face[nnf_0m] = true; weights_face[nnf_0m] =(1.-a)*    b ;
//    map_face[nnf_pm] = true; weights_face[nnf_pm] =    a *    b ;
//  }
//  else if ( A <= 0 &&
//            B >= 0 &&
//            neighbor_exists_face[nnf_m0] &&
//            neighbor_exists_face[nnf_0p] &&
//            neighbor_exists_face[nnf_mp] )
//  {
//    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-a)*(1.-b);
//    map_face[nnf_m0] = true; weights_face[nnf_m0] =    a *(1.-b);
//    map_face[nnf_0p] = true; weights_face[nnf_0p] =(1.-a)*    b ;
//    map_face[nnf_mp] = true; weights_face[nnf_mp] =    a *    b ;
//  }
//  else if ( A >= 0 &&
//            B >= 0 &&
//            neighbor_exists_face[nnf_p0] &&
//            neighbor_exists_face[nnf_0p] &&
//            neighbor_exists_face[nnf_pp] )
//  {
//    map_face[nnf_00] = true; weights_face[nnf_00] =(1.-a)*(1.-b);
//    map_face[nnf_p0] = true; weights_face[nnf_p0] =    a *(1.-b);
//    map_face[nnf_0p] = true; weights_face[nnf_0p] =(1.-a)*    b ;
//    map_face[nnf_pp] = true; weights_face[nnf_pp] =    a *    b ;
//  }

  else if (A <= 0 && B <= 0 &&
           neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_0m] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a-b;
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a;
    map_face[nnf_0m] = true; weights_face[nnf_0m] = b;
    mask_specific = -11;
  }
  else if (A >= 0 && B <= 0 &&
           neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_0m] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a-b;
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a;
    map_face[nnf_0m] = true; weights_face[nnf_0m] = b;
    mask_specific = -12;
  }
  else if (A <= 0 && B >= 0 &&
           neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_0p] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a-b;
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a;
    map_face[nnf_0p] = true; weights_face[nnf_0p] = b;
    mask_specific = -13;
  }
  else if (A >= 0 && B >= 0 &&
           neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_0p] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a-b;
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a;
    map_face[nnf_0p] = true; weights_face[nnf_0p] = b;
    mask_specific = -14;
  }





//    else if (num_good_neighbors >= 3 && !same_line)
//    {
//      weights_face[nnf_mm] = 0;
//      weights_face[nnf_0m] = 0;
//      weights_face[nnf_pm] = 0;
//      weights_face[nnf_m0] = 0;
//      weights_face[nnf_00] = 0;
//      weights_face[nnf_p0] = 0;
//      weights_face[nnf_mp] = 0;
//      weights_face[nnf_0p] = 0;
//      weights_face[nnf_pp] = 0;

//      // linear system
//      char num_constraints = 9;

//      std::vector<double> col_c(num_constraints, 0);
//      std::vector<double> col_x(num_constraints, 0);
//      std::vector<double> col_y(num_constraints, 0);

//      for (char j = 0; j < 3; ++j)
//        for (char i = 0; i < 3; ++i)
//        {
//          char idx = 3*j+i;
//          col_c[idx] = 1.;
//          col_x[idx] = ((double) (i-1));
//          col_y[idx] = ((double) (j-1));
//        }


//      double log_weight_min = -10.;
//      //          double gamma = -log_weight_min*2./3./sqrt((double)P4EST_DIM);
//      double gamma = -log_weight_min*2./3./2.;

//      std::vector<double> weight(num_constraints, 0);

//      for (char j = 0; j < 3; ++j)
//        for (char i = 0; i < 3; ++i)
//        {
//          char idx = 3*j+i;

//          double x = ((double) (i-1));
//          double y = ((double) (j-1));

//          //                    if (neighbor_exists_face[idx])
//          //                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//          //                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_)));
//          if (neighbor_exists_face[idx] || idx == nnf_00m)
////            weight[idx] = 1.;
//            weight[idx] = exp(-gamma*(SQR(x-A) +
//                                      SQR(y-B)));
//  //        if (idx == nnf_00m)
//  //          weight[idx] = 1.;
//        }

//      // assemble and invert matrix
//      char A_size = (2+1);
//      double matA[(2+1)*(2+1)];
//      double matA_inv[(2+1)*(2+1)];

//      matA[0*A_size + 0] = 0;
//      matA[0*A_size + 1] = 0;
//      matA[0*A_size + 2] = 0;
//      matA[1*A_size + 1] = 0;
//      matA[1*A_size + 2] = 0;
//      matA[2*A_size + 2] = 0;

//      for (char nei = 0; nei < num_constraints; ++nei)
//      {
//        matA[0*A_size + 0] += col_c[nei]*col_c[nei]*weight[nei];
//        matA[0*A_size + 1] += col_c[nei]*col_x[nei]*weight[nei];
//        matA[0*A_size + 2] += col_c[nei]*col_y[nei]*weight[nei];
//        matA[1*A_size + 1] += col_x[nei]*col_x[nei]*weight[nei];
//        matA[1*A_size + 2] += col_x[nei]*col_y[nei]*weight[nei];
//        matA[2*A_size + 2] += col_y[nei]*col_y[nei]*weight[nei];
//      }

//      matA[1*A_size + 0] = matA[0*A_size + 1];
//      matA[2*A_size + 0] = matA[0*A_size + 2];
//      matA[2*A_size + 1] = matA[1*A_size + 2];

//      inv_mat3(matA, matA_inv);

//      // compute Taylor expansion coefficients
//      std::vector<double> coeff_const_term(num_constraints, 0);
//      std::vector<double> coeff_x_term    (num_constraints, 0);
//      std::vector<double> coeff_y_term    (num_constraints, 0);

//      for (char nei = 0; nei < num_constraints; ++nei)
//      {
//        coeff_const_term[nei] = weight[nei]*
//            ( matA_inv[0*A_size+0]*col_c[nei]
//            + matA_inv[0*A_size+1]*col_x[nei]
//            + matA_inv[0*A_size+2]*col_y[nei] );

//        coeff_x_term[nei] = weight[nei]*
//            ( matA_inv[1*A_size+0]*col_c[nei]
//            + matA_inv[1*A_size+1]*col_x[nei]
//            + matA_inv[1*A_size+2]*col_y[nei] );

//        coeff_y_term[nei] = weight[nei]*
//            ( matA_inv[2*A_size+0]*col_c[nei]
//            + matA_inv[2*A_size+1]*col_x[nei]
//            + matA_inv[2*A_size+2]*col_y[nei] );
//      }

//      // compute integrals
//      double const_term = 1.;
//      double x_term     = 1.*(A);
//      double y_term     = 1.*(B);


//      if (!neighbor_exists_face[nnf_00])
//        std::cout << "weird!\n";

//      // matrix coefficients
//      for (char nei = 0; nei < 9; ++nei)
//      {
//        if (neighbor_exists_face[nei])
//  //        if (neighbor_exists_face[nei] || nei == nnf_00m)
//        {
//          map_face[nei] = true;
//          weights_face[nei] += coeff_const_term[nei]*const_term
//              + coeff_x_term[nei]*x_term
//              + coeff_y_term[nei]*y_term;
//        }
//      }
//      mask_specific = -1;
//      semi_fallback = true;
//    }


  else if (neighbor_exists_face[nnf_0m] &&
           neighbor_exists_face[nnf_mm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.+B);
    map_face[nnf_0m] = true; weights_face[nnf_0m] = -B+A;
    map_face[nnf_mm] = true; weights_face[nnf_mm] = -A;
    mask_specific = -15;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_mm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.+A);
    map_face[nnf_m0] = true; weights_face[nnf_m0] = -A+B;
    map_face[nnf_mm] = true; weights_face[nnf_mm] = -B;
    mask_specific = -16;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_0m] &&
           neighbor_exists_face[nnf_pm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.+B);
    map_face[nnf_0m] = true; weights_face[nnf_0m] = -B-A;
    map_face[nnf_pm] = true; weights_face[nnf_pm] = A;
    mask_specific = -17;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_pm] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-A);
    map_face[nnf_p0] = true; weights_face[nnf_p0] = A+B;
    map_face[nnf_pm] = true; weights_face[nnf_pm] = -B;
    mask_specific = -18;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_mp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.+A);
    map_face[nnf_m0] = true; weights_face[nnf_m0] = -A-B;
    map_face[nnf_mp] = true; weights_face[nnf_mp] = B;
    mask_specific = -19;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_0p] &&
           neighbor_exists_face[nnf_mp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-B);
    map_face[nnf_0p] = true; weights_face[nnf_0p] = B+A;
    map_face[nnf_mp] = true; weights_face[nnf_mp] = -A;
    mask_specific = -20;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_pp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-A);
    map_face[nnf_p0] = true; weights_face[nnf_p0] = A-B;
    map_face[nnf_pp] = true; weights_face[nnf_pp] = B;
    mask_specific = -21;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_0p] &&
           neighbor_exists_face[nnf_pp] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1.-B);
    map_face[nnf_0p] = true; weights_face[nnf_0p] = B-A;
    map_face[nnf_pp] = true; weights_face[nnf_pp] = A;
    mask_specific = -22;
    semi_fallback = true;
  }

  else if (neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_0m] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.+A+B;
    map_face[nnf_m0] = true; weights_face[nnf_m0] = -A;
    map_face[nnf_0m] = true; weights_face[nnf_0m] = -B;
    mask_specific = -23;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_0m] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-A+B;
    map_face[nnf_p0] = true; weights_face[nnf_p0] = A;
    map_face[nnf_0m] = true; weights_face[nnf_0m] = -B;
    mask_specific = -24;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_m0] &&
           neighbor_exists_face[nnf_0p] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.+A-B;
    map_face[nnf_m0] = true; weights_face[nnf_m0] = -A;
    map_face[nnf_0p] = true; weights_face[nnf_0p] = B;
    mask_specific = -25;
    semi_fallback = true;
  }
  else if (neighbor_exists_face[nnf_p0] &&
           neighbor_exists_face[nnf_0p] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-A-B;
    map_face[nnf_p0] = true; weights_face[nnf_p0] = A;
    map_face[nnf_0p] = true; weights_face[nnf_0p] = B;
    mask_specific = -26;
    semi_fallback = true;
  }


  else if (A <= 0 && neighbor_exists_face[nnf_m0] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a;
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a;
    full_fallback = true;
  }
  else if (B <= 0 && neighbor_exists_face[nnf_0m] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-b;
    map_face[nnf_0m] = true; weights_face[nnf_0m] = b;
    full_fallback = true;
  }
  else if (B >= 0 && neighbor_exists_face[nnf_0p] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-b;
    map_face[nnf_0p] = true; weights_face[nnf_0p] = b;
    full_fallback = true;
  }
  else if (A >= 0 && neighbor_exists_face[nnf_p0] )
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = 1.-a;
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a;
    full_fallback = true;
  }

  else
  {
//    std::cout << "!!!!!! Fallback flux between cells!\n";

    int num_good_neighbors = 0;
    mask_specific = 50;

    for (int i = 0; i < 9; ++i)
      if (neighbor_exists_face[i]) num_good_neighbors++;

    if (num_good_neighbors >= 3)
    {
      std::cout << "Possible! " << num_good_neighbors << "\n";
      std::cout << A << ", " << B << "\n";
      for (int i = 0; i < 9; ++i)
        if (neighbor_exists_face[i])  std::cout << i << " ";
      std::cout << "\n";

      if (num_good_neighbors == 3 &&
          ( (neighbor_exists_face[nnf_m0] && neighbor_exists_face[nnf_p0]) ||
            (neighbor_exists_face[nnf_0m] && neighbor_exists_face[nnf_0p]) ||
            (neighbor_exists_face[nnf_mm] && neighbor_exists_face[nnf_pp]) ||
            (neighbor_exists_face[nnf_mp] && neighbor_exists_face[nnf_pm]) ) )
      {
        std::cout << "Same line! " << num_good_neighbors << "\n";
      }
    } else {

      if ( (a > theta || b > theta) )
      {
        std::cout << "Too little good neighbors! " << num_good_neighbors << "\n";
        for (int i = 0; i < 9; ++i)
          if (neighbor_exists_face[i])  std::cout << i << " ";
        std::cout << "\n";
      }
    }

    full_fallback = true;
//    double x[9] = { -1, 0, 1,-1, 0, 1,-1, 0, 1 };
//    double y[9] = { -1,-1,-1, 0, 0, 0, 1, 1, 1 };

//    double x1, y1;
//    double x2, y2;

//    int nei1, nei2;

//    double A[9] = { 1, 0 -A, 0 -B,
//                    1, x1-A, y1-B,
//                    1, x2-A, y2-B };
//    double A_inv[9];

//    inv_mat3(A, A_inv);

//    weights_face[nnf_00] = A_inv[0];
//    weights_face[nei1] = A_inv[1];
//    weights_face[nei2] = A_inv[2];


//    if (num_good_neighbors == 3 &&
//        ( (neighbor_exists_face[nnf_m0] && neighbor_exists_face[nnf_p0]) ||
//          (neighbor_exists_face[nnf_0m] && neighbor_exists_face[nnf_0p]) ||
//          (neighbor_exists_face[nnf_mm] && neighbor_exists_face[nnf_pp]) ||
//          (neighbor_exists_face[nnf_mp] && neighbor_exists_face[nnf_pm]) ) )
//      std::cout << "On the same line! :( \n";

  }

//  return 1;

  if (full_fallback)
  {
//    std::cout << "Full fallback fluxes\n";
    return -0.1;
//    return -1;
  }
  else if (semi_fallback)
  {
//    return -1;
//    std::cout << "Semi fallback fluxes\n";
    return -0.2;
  }
  else
    return -1;
}
#else
double my_p4est_poisson_nodes_mls_t::compute_weights_through_face(double A, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face)
{
  bool full_fallback = false;

  map_face[nnf_m0] = false;
  map_face[nnf_00] = true;  weights_face[nnf_00] = 1;
  map_face[nnf_p0] = false;

  double a = fabs(A);

  if (a > .5) { std::cout << "Warning: face's centroid falls outside the face!\n"; }

  if      (A < -theta && neighbor_exists_face[nnf_m0])
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1. - a);
    map_face[nnf_m0] = true; weights_face[nnf_m0] = a;
  }
  else if (A >  theta && neighbor_exists_face[nnf_p0])
  {
    map_face[nnf_00] = true; weights_face[nnf_00] = (1. - a);
    map_face[nnf_p0] = true; weights_face[nnf_p0] = a;
  }
  else
  {
    if (a >= theta) { full_fallback = true; }
  }

  if (full_fallback) return -0.1;
  else               return -1.0;
}
#endif

void my_p4est_poisson_nodes_mls_t::discretize_inside(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                                                        double infc_phi_eff_000, bool is_wall[],
                                                        std::vector<mat_entry_t> *row_main, int &d_nnz, int &o_nnz)
{
  double  mu;
  double *mue_ptr, *mue_dd_ptr[P4EST_DIM];
  double  diag_add;
  double *mask_ptr;
  double *rhs_loc_ptr;

  // determine on which side from immersed interface
  if (infc_phi_eff_000 < 0) {
    mu          = mu_m_;
    mue_ptr     = mue_m_ptr;
    XCODE( mue_dd_ptr[0] = mue_m_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_m_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_m_zz_ptr );
    diag_add    = var_diag_ ? diag_m_ptr[n] : diag_m_scalar_;
    mask_ptr    = mask_m_ptr;
    rhs_loc_ptr = rhs_m_ptr;
  } else {
    mu          = mu_p_;
    mue_ptr     = mue_p_ptr;
    XCODE( mue_dd_ptr[0] = mue_p_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_p_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_p_zz_ptr );
    diag_add    = var_diag_ ? diag_p_ptr[n] : diag_p_scalar_;
    mask_ptr    = mask_p_ptr;
    rhs_loc_ptr = rhs_p_ptr;
  }


  if (new_submat_main_) {
    row_main->clear();
  }

  // far away from the boundary (not really necessary, already taken care of)
//  double bdry_phi_eff_000 = bdry_.num_phi == 0 ? -1. : bdry_.phi_eff_ptr[n];
//  if (bdry_phi_eff_000 > 0.)
//  {
//    if (new_submat_main_)
//    {
//      row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
//      mask_ptr[n] = 1.;
//    }

//    if (setup_rhs) rhs_ptr[n] = 0;

//    return;
//  }

  double mue_000 = var_mu_ ? mue_ptr[n] : mu;

  //---------------------------------------------------------------------
  // compute submat_diag
  //---------------------------------------------------------------------
  if (new_submat_diag_ && there_is_diag_) {
    submat_diag_ptr[n] = diag_add;

    if (fabs(diag_add) > EPS) {
      matrix_has_nullspace_ = false;
    }
  }

  //---------------------------------------------------------------------
  // compute submat_main
  //---------------------------------------------------------------------
  if (new_submat_main_) {
    p4est_locidx_t node_m00_mm=qnnn.node_m00_mm; p4est_locidx_t node_m00_pm=qnnn.node_m00_pm;
    p4est_locidx_t node_p00_mm=qnnn.node_p00_mm; p4est_locidx_t node_p00_pm=qnnn.node_p00_pm;
    p4est_locidx_t node_0m0_mm=qnnn.node_0m0_mm; p4est_locidx_t node_0m0_pm=qnnn.node_0m0_pm;
    p4est_locidx_t node_0p0_mm=qnnn.node_0p0_mm; p4est_locidx_t node_0p0_pm=qnnn.node_0p0_pm;
#ifdef P4_TO_P8
    p4est_locidx_t node_m00_mp=qnnn.node_m00_mp; p4est_locidx_t node_m00_pp=qnnn.node_m00_pp;
    p4est_locidx_t node_p00_mp=qnnn.node_p00_mp; p4est_locidx_t node_p00_pp=qnnn.node_p00_pp;
    p4est_locidx_t node_0m0_mp=qnnn.node_0m0_mp; p4est_locidx_t node_0m0_pp=qnnn.node_0m0_pp;
    p4est_locidx_t node_0p0_mp=qnnn.node_0p0_mp; p4est_locidx_t node_0p0_pp=qnnn.node_0p0_pp;

    p4est_locidx_t node_00m_mm=qnnn.node_00m_mm; p4est_locidx_t node_00m_mp=qnnn.node_00m_mp;
    p4est_locidx_t node_00m_pm=qnnn.node_00m_pm; p4est_locidx_t node_00m_pp=qnnn.node_00m_pp;
    p4est_locidx_t node_00p_mm=qnnn.node_00p_mm; p4est_locidx_t node_00p_mp=qnnn.node_00p_mp;
    p4est_locidx_t node_00p_pm=qnnn.node_00p_pm; p4est_locidx_t node_00p_pp=qnnn.node_00p_pp;
#endif

    double d_m00 = qnnn.d_m00; double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0; double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m; double d_00p = qnnn.d_00p;
#endif

    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    // interpolate diffusion coefficient if needed
    double DIM( mue_m00=mu, mue_0m0=mu, mue_00m=mu );
    double DIM( mue_p00=mu, mue_0p0=mu, mue_00p=mu );

    if (var_mu_) {
      CODE2D( qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0) );
      CODE3D( qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p) );
    }

    // discretization of Laplace operator
    double w_m00_mm=0, w_m00_pm=0;
    double w_p00_mm=0, w_p00_pm=0;
    double w_0m0_mm=0, w_0m0_pm=0;
    double w_0p0_mm=0, w_0p0_pm=0;
#ifdef P4_TO_P8
    double w_m00_mp=0, w_m00_pp=0;
    double w_p00_mp=0, w_p00_pp=0;
    double w_0m0_mp=0, w_0m0_pp=0;
    double w_0p0_mp=0, w_0p0_pp=0;

    double w_00m_mm=0, w_00m_pm=0;
    double w_00p_mm=0, w_00p_pm=0;
    double w_00m_mp=0, w_00m_pp=0;
    double w_00p_mp=0, w_00p_pp=0;

    //------------------------------------
    // Dfxx =   fxx + a*fyy + b*fzz
    // Dfyy = c*fxx +   fyy + d*fzz
    // Dfzz = e*fxx + f*fyy +   fzz
    //------------------------------------
    double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
    double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

    double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
    double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

    double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
    double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

    //------------------------------------------------------------
    // compensating the error of linear interpolation at T-junction using
    // the derivative in the transversal direction
    //
    // Laplace = wi*Dfxx +
    //           wj*Dfyy +
    //           wk*Dfzz
    //------------------------------------------------------------
    double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
    double wi = (1.-c-e+c*f+e*d-d*f)/det;
    double wj = (1.-a-f+a*e+f*b-b*e)/det;
    double wk = (1.-b-d+b*c+d*a-a*c)/det;

    //---------------------------------------------------------------------
    // Shortley-Weller method, dimension by dimension
    //---------------------------------------------------------------------
    double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

    // if node is at wall, what's below will apply Neumann BC
    if      (is_wall[dir::f_m00]) w_p00 += -1.0/(d_p00*d_p00);
    else if (is_wall[dir::f_p00]) w_m00 += -1.0/(d_m00*d_m00);
    else                          w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

    if      (is_wall[dir::f_p00]) w_m00 += -1.0/(d_m00*d_m00);
    else if (is_wall[dir::f_m00]) w_p00 += -1.0/(d_p00*d_p00);
    else                          w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

    if      (is_wall[dir::f_0m0]) w_0p0 += -1.0/(d_0p0*d_0p0);
    else if (is_wall[dir::f_0p0]) w_0m0 += -1.0/(d_0m0*d_0m0);
    else                          w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

    if      (is_wall[dir::f_0p0]) w_0m0 += -1.0/(d_0m0*d_0m0);
    else if (is_wall[dir::f_0m0]) w_0p0 += -1.0/(d_0p0*d_0p0);
    else                          w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

    if      (is_wall[dir::f_00m]) w_00p += -1.0/(d_00p*d_00p);
    else if (is_wall[dir::f_00p]) w_00m += -1.0/(d_00m*d_00m);
    else                          w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

    if      (is_wall[dir::f_00p]) w_00m += -1.0/(d_00m*d_00m);
    else if (is_wall[dir::f_00m]) w_00p += -1.0/(d_00p*d_00p);
    else                          w_00p += -2.0*wk/d_00p/(d_00m+d_00p);

    if(!is_wall[dir::f_m00]) {
      w_m00_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_mm]) : mu)*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
      w_m00_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_mp]) : mu)*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
      w_m00_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_pm]) : mu)*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
      w_m00_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_pp]) : mu)*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
      w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
    } else {
      w_m00 *= (var_mu_ ? 0.5*(mue_000 + mue_m00) : mu);
    }

    if(!is_wall[dir::f_p00]) {
      w_p00_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_mm]) : mu)*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
      w_p00_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_mp]) : mu)*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
      w_p00_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_pm]) : mu)*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
      w_p00_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_pp]) : mu)*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
      w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
    } else {
      w_p00 *= (var_mu_ ? 0.5*(mue_000 + mue_p00) : mu);
    }

    if(!is_wall[dir::f_0m0]) {
      w_0m0_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_mm]) : mu)*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
      w_0m0_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_mp]) : mu)*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
      w_0m0_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_pm]) : mu)*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
      w_0m0_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_pp]) : mu)*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
      w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
    } else {
      w_0m0 *= (var_mu_ ? 0.5*(mue_000 + mue_0m0) : mu);
    }

    if(!is_wall[dir::f_0p0]) {
      w_0p0_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_mm]) : mu)*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
      w_0p0_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_mp]) : mu)*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
      w_0p0_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_pm]) : mu)*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
      w_0p0_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_pp]) : mu)*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
      w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
    } else {
      w_0p0 *= (var_mu_ ? 0.5*(mue_000 + mue_0p0) : mu);
    }

    if(!is_wall[dir::f_00m]) {
      w_00m_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00m_mm]) : mu)*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
      w_00m_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00m_mp]) : mu)*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
      w_00m_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00m_pm]) : mu)*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
      w_00m_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00m_pp]) : mu)*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
      w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
    } else {
      w_00m *= (var_mu_ ? 0.5*(mue_000 + mue_00m) : mu);
    }

    if(!is_wall[dir::f_00p]) {
      w_00p_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00p_mm]) : mu)*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
      w_00p_mp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00p_mp]) : mu)*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
      w_00p_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00p_pm]) : mu)*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
      w_00p_pp = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_00p_pp]) : mu)*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
      w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
    } else {
      w_00p *= (var_mu_ ? 0.5*(mue_000 + mue_00p) : mu);
    }
#else
    //---------------------------------------------------------------------
    // compensating the error of linear interpolation at T-junction using
    // the derivative in the transversal direction
    //---------------------------------------------------------------------
    double wi = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);
    double wj = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);

    double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

    // note: if node is at wall, what's below will apply Neumann BC (second order)
    if      (is_wall[dir::f_m00]) w_p00 += -1.0/(d_p00*d_p00);
    else if (is_wall[dir::f_p00]) w_m00 += -1.0/(d_m00*d_m00);
    else                          w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

    if      (is_wall[dir::f_p00]) w_m00 += -1.0/(d_m00*d_m00);
    else if (is_wall[dir::f_m00]) w_p00 += -1.0/(d_p00*d_p00);
    else                          w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

    if      (is_wall[dir::f_0m0]) w_0p0 += -1.0/(d_0p0*d_0p0);
    else if (is_wall[dir::f_0p0]) w_0m0 += -1.0/(d_0m0*d_0m0);
    else                          w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

    if      (is_wall[dir::f_0p0]) w_0m0 += -1.0/(d_0m0*d_0m0);
    else if (is_wall[dir::f_0m0]) w_0p0 += -1.0/(d_0p0*d_0p0);
    else                          w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

    //---------------------------------------------------------------------
    // addition to diagonal elements
    //---------------------------------------------------------------------
    if (!is_wall[dir::f_m00]) {
      w_m00_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_mm]) : mu)*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
      w_m00_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_m00_pm]) : mu)*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
      w_m00 = w_m00_mm + w_m00_pm;
    } else {
      w_m00 *= var_mu_ ? 0.5*(mue_000 + mue_m00) : mu;
    }

    if (!is_wall[dir::f_p00]) {
      w_p00_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_mm]) : mu)*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
      w_p00_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_p00_pm]) : mu)*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
      w_p00    = w_p00_mm + w_p00_pm;
    } else {
      w_p00 *= var_mu_ ? 0.5*(mue_000 + mue_p00) : mu;
    }

    if (!is_wall[dir::f_0m0]) {
      w_0m0_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_mm]) : mu)*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
      w_0m0_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0m0_pm]) : mu)*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
      w_0m0 = w_0m0_mm + w_0m0_pm;
    } else {
      w_0m0 *= var_mu_ ? 0.5*(mue_000 + mue_0m0) : mu;
    }

    if (!is_wall[dir::f_0p0]) {
      w_0p0_mm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_mm]) : mu)*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
      w_0p0_pm = (var_mu_ ? 0.5*(mue_000 + mue_ptr[node_0p0_pm]) : mu)*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);
      w_0p0 = w_0p0_mm + w_0p0_pm;
    } else {
      w_0p0 *= var_mu_ ? 0.5*(mue_000 + mue_0p0) : mu;
    }
#endif

    double w_000 = - ( w_m00 + w_p00 + w_0m0 + w_0p0 ONLY3D( + w_00m + w_00p ) );

    //---------------------------------------------------------------------
    // add coefficients in the matrix
    //---------------------------------------------------------------------
    mat_entry_t ent;
    ent.n = petsc_gloidx_[qnnn.node_000]; ent.val = w_000; row_main->push_back(ent);

    if(!is_wall[dir::f_m00]) {
      if (ABS(w_m00_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_m00_mm]; ent.val = w_m00_mm; row_main->push_back(ent); (qnnn.node_m00_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_m00_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_m00_pm]; ent.val = w_m00_pm; row_main->push_back(ent); (qnnn.node_m00_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#ifdef P4_TO_P8
      if (ABS(w_m00_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_m00_mp]; ent.val = w_m00_mp; row_main->push_back(ent); (qnnn.node_m00_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_m00_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_m00_pp]; ent.val = w_m00_pp; row_main->push_back(ent); (qnnn.node_m00_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#endif
    }

    if(!is_wall[dir::f_p00]) {
      if (ABS(w_p00_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_p00_mm]; ent.val = w_p00_mm; row_main->push_back(ent); (qnnn.node_p00_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_p00_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_p00_pm]; ent.val = w_p00_pm; row_main->push_back(ent); (qnnn.node_p00_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#ifdef P4_TO_P8
      if (ABS(w_p00_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_p00_mp]; ent.val = w_p00_mp; row_main->push_back(ent); (qnnn.node_p00_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_p00_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_p00_pp]; ent.val = w_p00_pp; row_main->push_back(ent); (qnnn.node_p00_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#endif
    }

    if(!is_wall[dir::f_0m0]) {
      if (ABS(w_0m0_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0m0_mm]; ent.val = w_0m0_mm; row_main->push_back(ent); (qnnn.node_0m0_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_0m0_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0m0_pm]; ent.val = w_0m0_pm; row_main->push_back(ent); (qnnn.node_0m0_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#ifdef P4_TO_P8
      if (ABS(w_0m0_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0m0_mp]; ent.val = w_0m0_mp; row_main->push_back(ent); (qnnn.node_0m0_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_0m0_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0m0_pp]; ent.val = w_0m0_pp; row_main->push_back(ent); (qnnn.node_0m0_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#endif
    }

    if(!is_wall[dir::f_0p0]) {
      if (ABS(w_0p0_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0p0_mm]; ent.val = w_0p0_mm; row_main->push_back(ent); (qnnn.node_0p0_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_0p0_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0p0_pm]; ent.val = w_0p0_pm; row_main->push_back(ent); (qnnn.node_0p0_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#ifdef P4_TO_P8
      if (ABS(w_0p0_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0p0_mp]; ent.val = w_0p0_mp; row_main->push_back(ent); (qnnn.node_0p0_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_0p0_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_0p0_pp]; ent.val = w_0p0_pp; row_main->push_back(ent); (qnnn.node_0p0_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#endif
    }
#ifdef P4_TO_P8
    if(!is_wall[dir::f_00m]) {
      if (ABS(w_00m_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00m_mm]; ent.val = w_00m_mm; row_main->push_back(ent); (qnnn.node_00m_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00m_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00m_pm]; ent.val = w_00m_pm; row_main->push_back(ent); (qnnn.node_00m_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00m_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00m_mp]; ent.val = w_00m_mp; row_main->push_back(ent); (qnnn.node_00m_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00m_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00m_pp]; ent.val = w_00m_pp; row_main->push_back(ent); (qnnn.node_00m_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
    }

    if(!is_wall[dir::f_00p]) {
      if (ABS(w_00p_mm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00p_mm]; ent.val = w_00p_mm; row_main->push_back(ent); (qnnn.node_00p_mm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00p_pm) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00p_pm]; ent.val = w_00p_pm; row_main->push_back(ent); (qnnn.node_00p_pm < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00p_mp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00p_mp]; ent.val = w_00p_mp; row_main->push_back(ent); (qnnn.node_00p_mp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (ABS(w_00p_pp) > EPS) { ent.n = petsc_gloidx_[qnnn.node_00p_pp]; ent.val = w_00p_pp; row_main->push_back(ent); (qnnn.node_00p_pp < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
    }
#endif

    if (petsc_gloidx_[n] < fixed_value_idx_g_)
    {
      fixed_value_idx_l_ = n;
      fixed_value_idx_g_ = petsc_gloidx_[n];
    }
  }

  //---------------------------------------------------------------------
  // compute rhs
  //---------------------------------------------------------------------
  if (setup_rhs)
  {
    rhs_ptr[n] = rhs_loc_ptr[n];

    // add to rhs wall conditions
    if (ORD(is_wall[dir::f_m00], is_wall[dir::f_0m0], is_wall[dir::f_00m]) ||
        ORD(is_wall[dir::f_p00], is_wall[dir::f_0p0], is_wall[dir::f_00p]) )
    {
      double xyz_C[P4EST_DIM];
      node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

      XCODE( double eps_x = is_wall[dir::f_m00] ? 2.*EPS*diag_min_ : (is_wall[dir::f_p00] ? -2.*EPS*diag_min_ : 0) );
      YCODE( double eps_y = is_wall[dir::f_0m0] ? 2.*EPS*diag_min_ : (is_wall[dir::f_0p0] ? -2.*EPS*diag_min_ : 0) );
      ZCODE( double eps_z = is_wall[dir::f_00m] ? 2.*EPS*diag_min_ : (is_wall[dir::f_00p] ? -2.*EPS*diag_min_ : 0) );

      if (is_wall[dir::f_m00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0], xyz_C[1]+eps_y, xyz_C[2]+eps_z) ) / qnnn.d_p00;
      if (is_wall[dir::f_p00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0], xyz_C[1]+eps_y, xyz_C[2]+eps_z) ) / qnnn.d_m00;

      if (is_wall[dir::f_0m0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0]+eps_x, xyz_C[1], xyz_C[2]+eps_z) ) / qnnn.d_0p0;
      if (is_wall[dir::f_0p0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0]+eps_x, xyz_C[1], xyz_C[2]+eps_z) ) / qnnn.d_0m0;
#ifdef P4_TO_P8
      if (is_wall[dir::f_00m]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0]+eps_x, xyz_C[1]+eps_y, xyz_C[2]) ) / qnnn.d_00p;
      if (is_wall[dir::f_00p]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(xyz_C[0]+eps_x, xyz_C[1]+eps_y, xyz_C[2]) ) / qnnn.d_00m;
#endif
    }
  }
}


void my_p4est_poisson_nodes_mls_t::discretize_dirichlet_sw(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                                                           double infc_phi_eff_000, bool is_wall[],
                                                           std::vector<mat_entry_t> *row_main, int &d_nnz, int &o_nnz)
{
  double  mu;
  double *mue_ptr, *mue_dd_ptr[P4EST_DIM];
  double  diag_add;
  double *mask_ptr;
  double *rhs_loc_ptr;

  if (infc_phi_eff_000 < 0) {
    mu          = mu_m_;
    mue_ptr     = mue_m_ptr;
    XCODE( mue_dd_ptr[0] = mue_m_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_m_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_m_zz_ptr );
    diag_add    = var_diag_ ? diag_m_ptr[n] : diag_m_scalar_;
    mask_ptr    = mask_m_ptr;
    rhs_loc_ptr = rhs_m_ptr;
  } else {
    mu          = mu_p_;
    mue_ptr     = mue_p_ptr;
    XCODE( mue_dd_ptr[0] = mue_p_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_p_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_p_zz_ptr );
    diag_add    = var_diag_ ? diag_p_ptr[n] : diag_p_scalar_;
    mask_ptr    = mask_p_ptr;
    rhs_loc_ptr = rhs_p_ptr;
  }

  double bdry_phi_eff_000 = bdry_.num_phi == 0 ? -1. : bdry_.phi_eff_ptr[n];

  if (new_submat_main_) {
    row_main->clear();
  }

  // far away from the boundary
  if (bdry_phi_eff_000 > 0.) {

    if (new_submat_main_) {
      row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
      mask_ptr[n] = 1.;
    }

    if (setup_rhs) {
      rhs_ptr[n] = 0;
    }

    return;
  }

  double xyz_C[P4EST_DIM];
  node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

  double DIM( x_C = xyz_C[0],
              y_C = xyz_C[1],
              z_C = xyz_C[2] );

  // check if any boundary crosses ngbd
  std::vector<bool>   is_interface     (P4EST_FACES, false);
  std::vector<int>    bdry_point_id    (P4EST_FACES, 0);
  std::vector<double> bdry_point_dist  (P4EST_FACES, 0);
  std::vector<double> bdry_point_weight(P4EST_FACES, 0);

  if (new_submat_main_) {
    find_interface_points(n, ngbd_, bdry_.opn, bdry_.phi_ptr, DIM(bdry_.phi_xx_ptr, bdry_.phi_yy_ptr, bdry_.phi_zz_ptr), bdry_point_id.data(), bdry_point_dist.data());

    int num_interfaces = 0;
    foreach_direction(dim) {
      if (bdry_point_id[dim] >= 0) {
        is_interface[dim] = true;
        ++num_interfaces;
      }
    }
  } else {
    load_cart_points(n, is_interface, bdry_point_id, bdry_point_dist, bdry_point_weight);
  }

  // check whether boundary goes exactly through the node
  int node_on_boundary_phi_id = -1;
  foreach_direction(dim) {
    if (is_interface[dim] && (fabs(bdry_point_dist[dim]) < EPS*diag_min_)) {
      node_on_boundary_phi_id = bdry_point_id[dim];
    }
  }

  // interface boundary
  if (node_on_boundary_phi_id != -1) {

    // compute submat_main
    if (new_submat_main_) {
      // re-asssign boundary points data
      is_interface.assign(P4EST_FACES, false);
      is_interface     [0] = true;
      bdry_point_id    [0] = node_on_boundary_phi_id;
      bdry_point_dist  [0] = 0.5*EPS*diag_min_;
      bdry_point_weight[0] = 1;

      row_main->push_back(mat_entry_t(n, 1));
      matrix_has_nullspace_ = false;
    }

    // compute submat_rhs
    if (setup_rhs) {
      boundary_conditions_t *bc = &bc_[node_on_boundary_phi_id];
      rhs_ptr[n] = bc->pointwise ? bc->get_value_pw(n,0) :
                                   bc->get_value_cf(xyz_C);
    }

  } else {
    // if far away from the interface or close to it but with dirichlet
    // then finite difference method
    double mue_000 = var_mu_ ? mue_ptr[n] : mu;

    //---------------------------------------------------------------------
    // compute submat_diag
    //---------------------------------------------------------------------
    if (new_submat_diag_ && there_is_diag_)
    {
      // assemble submat_diag
      submat_diag_ptr[n] = diag_add;
      if (fabs(diag_add) > EPS) {
        matrix_has_nullspace_ = false;
      }
    }

    //---------------------------------------------------------------------
    // compute submat_main
    //---------------------------------------------------------------------
    if (new_submat_main_)
    {
      p4est_locidx_t node_m00 = qnnn.neighbor_m00(); p4est_locidx_t node_p00 = qnnn.neighbor_p00();
      p4est_locidx_t node_0m0 = qnnn.neighbor_0m0(); p4est_locidx_t node_0p0 = qnnn.neighbor_0p0();
#ifdef P4_TO_P8
      p4est_locidx_t node_0m0 = qnnn.neighbor_0m0(); p4est_locidx_t node_0p0 = qnnn.neighbor_0p0();
#endif

      double d_m00 = qnnn.d_m00; double d_p00 = qnnn.d_p00;
      double d_0m0 = qnnn.d_0m0; double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
      double d_00m = qnnn.d_00m; double d_00p = qnnn.d_00p;
#endif

      // interpolate diffusion coefficient if needed
      double DIM( mue_m00=mu, mue_0m0=mu, mue_00m=mu );
      double DIM( mue_p00=mu, mue_0p0=mu, mue_00p=mu );

      if (var_mu_) {

        CODE2D( qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0) );
        CODE3D( qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p) );

        if (is_interface[dir::f_m00]) mue_m00 = qnnn.interpolate_in_dir(dir::f_m00, bdry_point_dist[dir::f_m00], mue_ptr, mue_dd_ptr);
        if (is_interface[dir::f_p00]) mue_p00 = qnnn.interpolate_in_dir(dir::f_p00, bdry_point_dist[dir::f_p00], mue_ptr, mue_dd_ptr);

        if (is_interface[dir::f_0m0]) mue_0m0 = qnnn.interpolate_in_dir(dir::f_0m0, bdry_point_dist[dir::f_0m0], mue_ptr, mue_dd_ptr);
        if (is_interface[dir::f_0p0]) mue_0p0 = qnnn.interpolate_in_dir(dir::f_0p0, bdry_point_dist[dir::f_0p0], mue_ptr, mue_dd_ptr);
#ifdef P4_TO_P8
        if (is_interface[dir::f_00m]) mue_00m = qnnn.interpolate_in_dir(dir::f_00m, bdry_point_dist[dir::f_00m], mue_ptr, mue_dd_ptr);
        if (is_interface[dir::f_00p]) mue_00p = qnnn.interpolate_in_dir(dir::f_00p, bdry_point_dist[dir::f_00p], mue_ptr, mue_dd_ptr);
#endif
      }

      // adjust stencil's arms
      if (is_interface[dir::f_m00]) { d_m00 = bdry_point_dist[dir::f_m00]; }
      if (is_interface[dir::f_p00]) { d_p00 = bdry_point_dist[dir::f_p00]; }

      if (is_interface[dir::f_0m0]) { d_0m0 = bdry_point_dist[dir::f_0m0]; }
      if (is_interface[dir::f_0p0]) { d_0p0 = bdry_point_dist[dir::f_0p0]; }
#ifdef P4_TO_P8
      if (is_interface[dir::f_00m]) { d_00m = bdry_point_dist[dir::f_00m]; }
      if (is_interface[dir::f_00p]) { d_00p = bdry_point_dist[dir::f_00p]; }
#endif

      // discretization of Laplace operator
      double DIM(w_m00 = 0, w_0m0 = 0, w_00m = 0);
      double DIM(w_p00 = 0, w_0p0 = 0, w_00p = 0);

      // note: if node is at wall, what's below will apply Neumann BC
      if      (is_wall[dir::f_m00]) { w_p00 = -2.0*mue_000/(d_p00*d_p00); }
      else if (is_wall[dir::f_p00]) { w_m00 = -2.0*mue_000/(d_m00*d_m00); }
      else {
        w_m00 = -(mue_000+mue_m00)/d_m00/(d_m00+d_p00);
        w_p00 = -(mue_000+mue_p00)/d_p00/(d_m00+d_p00);
      }

      if      (is_wall[dir::f_0m0]) { w_0p0 = -2.0*mue_000/(d_0p0*d_0p0); }
      else if (is_wall[dir::f_0p0]) { w_0m0 = -2.0*mue_000/(d_0m0*d_0m0); }
      else {
        w_0m0 = -(mue_000+mue_0m0)/d_0m0/(d_0m0+d_0p0);
        w_0p0 = -(mue_000+mue_0p0)/d_0p0/(d_0m0+d_0p0);
      }

#ifdef P4_TO_P8
      if      (is_wall[dir::f_00m]) { w_00p = -2.0*mue_000/(d_00p*d_00p); }
      else if (is_wall[dir::f_00p]) { w_00m = -2.0*mue_000/(d_00m*d_00m); }
      else {
        w_00m = -(mue_000+mue_00m)/d_00m/(d_00m+d_00p);
        w_00p = -(mue_000+mue_00p)/d_00p/(d_00m+d_00p);
      }
#endif

      double w_000 = - SUMD(w_m00, w_0m0, w_00m)
                     - SUMD(w_p00, w_0p0, w_00p);

      //---------------------------------------------------------------------
      // add coefficients in the matrix
      //---------------------------------------------------------------------
      mat_entry_t ent;
      ent.n = petsc_gloidx_[qnnn.node_000]; ent.val = w_000; row_main->push_back(ent);

      if (!is_interface[dir::f_m00] && !is_wall[dir::f_m00] && ABS(w_m00) > EPS) { ent.n = petsc_gloidx_[node_m00]; ent.val = w_m00; row_main->push_back(ent); (node_m00 < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (!is_interface[dir::f_p00] && !is_wall[dir::f_p00] && ABS(w_p00) > EPS) { ent.n = petsc_gloidx_[node_p00]; ent.val = w_p00; row_main->push_back(ent); (node_p00 < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }

      if (!is_interface[dir::f_0m0] && !is_wall[dir::f_0m0] && ABS(w_0m0) > EPS) { ent.n = petsc_gloidx_[node_0m0]; ent.val = w_0m0; row_main->push_back(ent); (node_0m0 < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (!is_interface[dir::f_0p0] && !is_wall[dir::f_0p0] && ABS(w_0p0) > EPS) { ent.n = petsc_gloidx_[node_0p0]; ent.val = w_0p0; row_main->push_back(ent); (node_0p0 < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#ifdef P4_TO_P8
      if (!is_interface[dir::f_00m] && !is_wall[dir::f_00m] && ABS(w_00m) > EPS) { ent.n = petsc_gloidx_[node_00m]; ent.val = w_00m; row_main->push_back(ent); (node_00m < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
      if (!is_interface[dir::f_00p] && !is_wall[dir::f_00p] && ABS(w_00p) > EPS) { ent.n = petsc_gloidx_[node_00p]; ent.val = w_00p; row_main->push_back(ent); (node_00p < nodes_->num_owned_indeps) ? d_nnz++ : o_nnz++; }
#endif

      foreach_direction(dim) {
        if (is_interface[dim]) {
          matrix_has_nullspace_ = false;
        }
      }

      if (petsc_gloidx_[n] < fixed_value_idx_g_) {
        fixed_value_idx_l_ = n;
        fixed_value_idx_g_ = petsc_gloidx_[n];
      }

      bdry_point_weight[dir::f_m00] = w_m00;
      bdry_point_weight[dir::f_p00] = w_p00;

      bdry_point_weight[dir::f_0m0] = w_0m0;
      bdry_point_weight[dir::f_0p0] = w_0p0;
#ifdef P4_TO_P8
      bdry_point_weight[dir::f_00m] = w_00m;
      bdry_point_weight[dir::f_00p] = w_00p;
#endif
    }

    //---------------------------------------------------------------------
    // compute rhs
    //---------------------------------------------------------------------
    if (setup_rhs)
    {
      rhs_ptr[n] = rhs_loc_ptr[n];

      // sample boundary conditions
      std::vector<double> bc_values(P4EST_FACES, 0);

      // first get pointwise given values
      for (int i = 0; i < bc_.size(); ++i) {
        if (bc_[i].type == DIRICHLET && bc_[i].pointwise) {
          for (int j = 0; j < bc_[i].num_value_pts(n); ++j){
            // TODO: this is kind of ugly, fix this
            int idx = bc_[i].idx_value_pt(n, j);
            bc_values[ bc_[i].dirichlet_pts[idx].dir ] = (*bc_[i].value_pw)[idx];
          }
        }
      }

      // second get function-given values
      if (is_interface[dir::f_m00] && (!bc_[bdry_point_id[dir::f_m00]].pointwise)) bc_values[dir::f_m00] = (*bc_[bdry_point_id[dir::f_m00]].value_cf)( DIM(x_C - bdry_point_dist[dir::f_m00], y_C, z_C) );
      if (is_interface[dir::f_p00] && (!bc_[bdry_point_id[dir::f_p00]].pointwise)) bc_values[dir::f_p00] = (*bc_[bdry_point_id[dir::f_p00]].value_cf)( DIM(x_C + bdry_point_dist[dir::f_p00], y_C, z_C) );

      if (is_interface[dir::f_0m0] && (!bc_[bdry_point_id[dir::f_0m0]].pointwise)) bc_values[dir::f_0m0] = (*bc_[bdry_point_id[dir::f_0m0]].value_cf)( DIM(x_C, y_C - bdry_point_dist[dir::f_0m0], z_C) );
      if (is_interface[dir::f_0p0] && (!bc_[bdry_point_id[dir::f_0p0]].pointwise)) bc_values[dir::f_0p0] = (*bc_[bdry_point_id[dir::f_0p0]].value_cf)( DIM(x_C, y_C + bdry_point_dist[dir::f_0p0], z_C) );
#ifdef P4_TO_P8
      if (is_interface[dir::f_00m] && (!bc_[bdry_point_id[dir::f_00m]].pointwise)) bc_values[dir::f_00m] = (*bc_[bdry_point_id[dir::f_00m]].value_cf)( DIM(x_C, y_C, z_C - bdry_point_dist[dir::f_00m]) );
      if (is_interface[dir::f_00p] && (!bc_[bdry_point_id[dir::f_00p]].pointwise)) bc_values[dir::f_00p] = (*bc_[bdry_point_id[dir::f_00p]].value_cf)( DIM(x_C, y_C, z_C + bdry_point_dist[dir::f_00p]) );
#endif

      // add to rhs boundary conditions
      foreach_direction(i) {
        if (is_interface[i]) {
          rhs_ptr[n] -= bdry_point_weight[i] * bc_values[i];
        }
      }

      // add to rhs wall conditions
      XCODE( double eps_x = is_wall[dir::f_m00] ? 2.*EPS*diag_min_ : (is_wall[dir::f_p00] ? -2.*EPS*diag_min_ : 0) );
      YCODE( double eps_y = is_wall[dir::f_0m0] ? 2.*EPS*diag_min_ : (is_wall[dir::f_0p0] ? -2.*EPS*diag_min_ : 0) );
      ZCODE( double eps_z = is_wall[dir::f_00m] ? 2.*EPS*diag_min_ : (is_wall[dir::f_00p] ? -2.*EPS*diag_min_ : 0) );

      if (is_wall[dir::f_m00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C, y_C+eps_y, z_C+eps_z) ) / qnnn.d_p00;
      if (is_wall[dir::f_p00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C, y_C+eps_y, z_C+eps_z) ) / qnnn.d_m00;

      if (is_wall[dir::f_0m0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C, z_C+eps_z) ) / qnnn.d_0p0;
      if (is_wall[dir::f_0p0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C, z_C+eps_z) ) / qnnn.d_0m0;
#ifdef P4_TO_P8
      if (is_wall[dir::f_00m]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C+eps_y, z_C) ) / qnnn.d_00p;
      if (is_wall[dir::f_00p]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C+eps_y, z_C) ) / qnnn.d_00m;
#endif
    }
  }

  // save information about interface points
  if (new_submat_main_ & there_is_dirichlet_)
  {
    save_cart_points(n, is_interface, bdry_point_id, bdry_point_dist, bdry_point_weight);
  }
}

bool my_p4est_poisson_nodes_mls_t::gf_is_ghost(const quad_neighbor_nodes_of_node_t &qnnn)
{
  if (bdry_.num_phi > 0) {
    if (bdry_.phi_eff_ptr[qnnn.node_000] > -gf_thresh_*diag_min_) {
      foreach_direction (nei) {
        if (bdry_.phi_eff_ptr[qnnn.neighbor(nei)] <= -gf_thresh_*diag_min_) {
          return true;
        }
      }
    }
  }
  return false;
}

void my_p4est_poisson_nodes_mls_t::gf_direction(const quad_neighbor_nodes_of_node_t &qnnn, const p4est_locidx_t neighbors[], int &dir, double del_xyz[])
{
  double normal [P4EST_DIM];
  compute_normals(qnnn, bdry_.phi_eff_ptr, normal); // using phi_eff we assume there are no intersecting level set functions

  // we will select the direction with the maximum negative normal projection
  // among all neighboring nodes in negative domain
  double max_projection = 1;
  for (int nei = 0; nei < num_neighbors_cube; ++nei) {
    if (neighbors[nei] != -1) {
      if (bdry_.phi_eff_ptr[neighbors[nei]] <= -gf_thresh_*diag_min_) {

        int nei_dir[P4EST_DIM];
        cube_nei_dir(nei, nei_dir);

//        if (SUMD(fabs(nei_dir[0]), fabs(nei_dir[1]), fabs(nei_dir[2])) > 1.5) {
//          continue;
//        }

        double dxyz_nei[] = { DIM(double(nei_dir[0])*dxyz_m_[0],
                                  double(nei_dir[1])*dxyz_m_[1],
                                  double(nei_dir[2])*dxyz_m_[2]) };

        double projection = SUMD(normal[0]*dxyz_nei[0],
                                 normal[1]*dxyz_nei[1],
                                 normal[2]*dxyz_nei[2])
            / ABSD(dxyz_nei[0], dxyz_nei[1], dxyz_nei[2]);

        if (projection < max_projection) { // because want max negative projection
          max_projection = projection;
          EXECD( del_xyz[0] = dxyz_nei[0],
                 del_xyz[1] = dxyz_nei[1],
                 del_xyz[2] = dxyz_nei[2] );
          dir = nei;
        }
      }
    }
  }

  if (max_projection > 0) {
    std::cout << "[Warning] Ghost-Fluid Dirichlet: selected direction points away from the interface\n";
  }
}

void my_p4est_poisson_nodes_mls_t::discretize_dirichlet_gf(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                                                           double infc_phi_eff_000, bool is_wall[],
                                                           vector<int> &gf_map, vector<double> &gf_nodes, vector<double> &gf_phi,
                                                           std::vector<mat_entry_t> *row_main, int &d_nnz_main, int &o_nnz_main,
                                                           std::vector<mat_entry_t> *row_gf, int &d_nnz_gf, int &o_nnz_gf,
                                                           std::vector<mat_entry_t> *row_gf_ghost, int &d_nnz_gf_ghost, int &o_nnz_gf_ghost)
{
  double  mu;
  double *mue_ptr, *mue_dd_ptr[P4EST_DIM];
  double  diag_add;
  double *mask_ptr;
  double *rhs_loc_ptr;

  if (infc_phi_eff_000 < 0) {
    mu          = mu_m_;
    mue_ptr     = mue_m_ptr;
    XCODE( mue_dd_ptr[0] = mue_m_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_m_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_m_zz_ptr );
    diag_add    = var_diag_ ? diag_m_ptr[n] : diag_m_scalar_;
    mask_ptr    = mask_m_ptr;
    rhs_loc_ptr = rhs_m_ptr;
  } else {
    mu          = mu_p_;
    mue_ptr     = mue_p_ptr;
    XCODE( mue_dd_ptr[0] = mue_p_xx_ptr );
    YCODE( mue_dd_ptr[1] = mue_p_yy_ptr );
    ZCODE( mue_dd_ptr[2] = mue_p_zz_ptr );
    diag_add    = var_diag_ ? diag_p_ptr[n] : diag_p_scalar_;
    mask_ptr    = mask_p_ptr;
    rhs_loc_ptr = rhs_p_ptr;
  }

  double bdry_phi_eff_000 = bdry_.num_phi == 0 ? -1. : bdry_.phi_eff_ptr[n];

//  if (new_submat_main_) {
//    entries_main->at(n).clear();
//  }

  double xyz_C[P4EST_DIM];
  node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

  double DIM( x_C = xyz_C[0],
              y_C = xyz_C[1],
              z_C = xyz_C[2] );

//  // check whether boundary goes exactly through the node
//  int node_on_boundary_phi_id = -1;
//  foreach_direction(dim) {
//    if (is_interface[dim] && (fabs(bdry_point_dist[dim]) < EPS*diag_min_)) {
//      throw;
//      node_on_boundary_phi_id = bdry_point_id[dim];
//    }
//  }

//  // interface boundary
//  if (node_on_boundary_phi_id != -1) {

//    // compute submat_main
//    if (new_submat_main_) {
//      // re-asssign boundary points data
//      is_interface.assign(P4EST_FACES, false);
//      is_interface     [0] = true;
//      bdry_point_id    [0] = node_on_boundary_phi_id;
//      bdry_point_dist  [0] = 0.5*EPS*diag_min_;
//      bdry_point_weight[0] = 1;

//      row_main->push_back(mat_entry_t(n, 1));
//      matrix_has_nullspace_ = false;
//    }

//    // compute submat_rhs
//    if (setup_rhs) {
//      boundary_conditions_t *bc = &bc_[node_on_boundary_phi_id];
//      rhs_ptr[n] = bc->pointwise ? bc->get_value_pw(n,0) :
//                                   bc->get_value_cf(xyz_C);
//    }

//  }
  if ( bdry_phi_eff_000 > -gf_thresh_*diag_min_) {

    double xyz_bc[P4EST_DIM];
    double weight_bc;

    if (new_submat_main_) {
      row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
      mask_ptr[n] = 1.;
    }

    if (setup_rhs) {
      rhs_ptr[n] = 0;
    }

    if (gf_is_ghost(qnnn)) {

      // determine which level-set function intersects
      // (note this is not a bullet-proof algorithm,
      // but should work fine for non-intersecting boundaries)
      int phi_idx = 0;
      double phi_min = fabs(bdry_.phi_ptr[0][n]);

      for (int i = 1; i < bdry_.num_phi; ++i) {
        if (fabs(bdry_.phi_ptr[i][n]) < phi_min) {
          phi_idx = i;
          phi_min = fabs(bdry_.phi_ptr[i][n]) ;
        }
      }

      if (new_submat_main_) {

        // determine good direction for extrapolation
        p4est_locidx_t neighbors[num_neighbors_cube];
        ngbd_->get_all_neighbors(n, neighbors);

        double delta_xyz[P4EST_DIM];
        int    nei;
        gf_direction(qnnn, neighbors, nei, delta_xyz);
        double del = ABSD(delta_xyz[0], delta_xyz[1], delta_xyz[2]);

        // get level-set values an global node indices
        vector<double>   phi_values(gf_stencil_size(), -1);
        vector<PetscInt> global_idx(gf_stencil_size(), -1);

        phi_values[0] = bdry_.phi_eff_ptr[n];
        phi_values[1] = bdry_.phi_eff_ptr[neighbors[nei]];

        global_idx[0] = petsc_gloidx_[n];
        global_idx[1] = petsc_gloidx_[neighbors[nei]];

        int num_good_neis = 1;

        for (int i = 2; i < gf_stencil_size(); ++i) {
          phi_values[i] = gf_phi  [(gf_stencil_size()-2)*gf_map[n] + i-2];
          global_idx[i] = gf_nodes[(gf_stencil_size()-2)*gf_map[n] + i-2];

          if (phi_values[i] <= -gf_thresh_*diag_min_ && fabs(global_idx[i]-round(global_idx[i])) < 1.e-5) {
            num_good_neis++;
          }
        }

        if (num_good_neis < gf_stencil_size()-1) {
          std::cout << "[Warning] Reduced stencil (" << num_good_neis << "/" << gf_stencil_size()-1 << ")\n";
        }

//        // sanity check
//        if (phi_values[1] > 0) {
//          throw std::domain_error("Something went terribly wrong during constructing ghost fluid value");
//        }

        // find interface location
        double phi0_dd = 0;
        double phi1_dd = 0;

        if (neighbors[cube_nei_op(nei)] != -1) {
          phi0_dd = (phi_values[1] - 2.*phi_values[0] + bdry_.phi_eff_ptr[neighbors[cube_nei_op(nei)]])/del/del;
        }

        if (num_good_neis >= 2) {
          phi1_dd = (phi_values[2]-2.*phi_values[1]+phi_values[0])/del/del;
        }

        double dist = 0;
        if (phi_values[0] > 0 && phi_values[1] < 0) {
          dist = interface_Location_With_Second_Order_Derivative(0, del, phi_values[0], phi_values[1], phi0_dd, phi1_dd);
        } else if (phi_values[1] > 0 && phi_values[1] > 0) {
          dist = interface_Location_With_Second_Order_Derivative(del, 2.*del, phi_values[1], phi_values[2], 0, 0);
        } else if (phi_values[0] < 0) {
          // sanity check
          if (bdry_.phi_eff_ptr[neighbors[cube_nei_op(nei)]] < 0) {
            throw std::domain_error("Something went terribly wrong during constructing ghost fluid value");
          }
          dist = interface_Location_With_Second_Order_Derivative(-del, 0, bdry_.phi_eff_ptr[neighbors[cube_nei_op(nei)]], phi_values[0], phi0_dd, phi0_dd);
        }

        // compute extrapolation weights
        double theta = (del-dist)/del;
        vector<double> w(gf_stencil_size(), 0);

        w[0] = 1;

        if (gf_order_ == 3 && num_good_neis >= 4 && gf_stabilized_ != 0) { // cubic continuous

          // double interpolation
          w[1] = 1./36.*(   0. - 108.*theta +  88.*pow(theta, 2) +  39.*pow(theta, 3) - 17.*pow(theta, 4) -  3.*pow(theta, 5) +    pow(theta, 6)),
          w[2] = 1./36.*(-216. + 432.*theta - 108.*pow(theta, 2) - 156.*pow(theta, 3) + 39.*pow(theta, 4) + 12.*pow(theta, 5) - 3.*pow(theta, 6)),
          w[3] = 1./36.*( 144. - 216.*theta -  54.*pow(theta, 2) + 159.*pow(theta, 3) - 21.*pow(theta, 4) - 15.*pow(theta, 5) + 3.*pow(theta, 6)),
          w[4] = 1./36.*(- 36. +  48.*theta +  20.*pow(theta, 2) -  36.*pow(theta, 3) -     pow(theta, 4) +  6.*pow(theta, 5) -    pow(theta, 6)),
          weight_bc = -1./36.*(144. - 156.*theta + 54.*pow(theta, 2) - 6.*pow(theta, 3));

//          // least-squares based
//          double den = 18 + 66*theta+ 301*pow(theta, 2) + 456*pow(theta, 3) + 301*pow(theta, 4) + 90*pow(theta, 5) + 10*pow(theta, 6);
//          w[1] = 3*theta*(-292 - 272*theta+ 153*pow(theta, 2) + 281*pow(theta, 3) + 115*pow(theta, 4) + 15*pow(theta, 5))/2./den;
//          w[2] =    -3*(72 + 120*theta- 56*pow(theta, 2) - 140*pow(theta, 3) - 21*pow(theta, 4) + 20*pow(theta, 5) + 5*pow(theta, 6))/2./den;
//          w[3] =   (144 + 312*theta+ 410*pow(theta, 2) - 189*pow(theta, 3) - 457*pow(theta, 4) - 195*pow(theta, 5) - 25*pow(theta, 6))/2./den;
//          w[4] =   3*(-12 - 28*theta- 50*pow(theta, 2) + 4*pow(theta, 3) + 51*pow(theta, 4) + 30*pow(theta, 5) + 5*pow(theta, 6))/2./den;
//          weight_bc =   -(72 + 570*theta+ 495*pow(theta, 2) + 105*pow(theta, 3))/den;

        } else if (gf_order_ == 3 && num_good_neis >= 3 && gf_stabilized_ != 1) { // cubic native

          w[1] = 3.*(-2. -    theta + 2.*pow(theta,2) +    pow(theta,3))/(theta*(2. + 3.*theta + pow(theta,2)));
          w[2] =    ( 0. + 6.*theta - 3.*pow(theta,2) - 3.*pow(theta,3))/(theta*(2. + 3.*theta + pow(theta,2)));
          w[3] =    ( 0. -    theta + 0.              +    pow(theta,3))/(theta*(2. + 3.*theta + pow(theta,2)));
          weight_bc = -6./(theta*(2. + 3.*theta + pow(theta,2)));

        } else if (gf_order_ >= 2 && num_good_neis >= 3 && gf_stabilized_ != 0) { // quadratic continuous

          // double interpolation
          w[1] =  0. - 2.0*theta + 1.75*pow(theta,2) + 0.5*pow(theta,3) - 0.25*pow(theta,4);
          w[2] = -3. + 6.0*theta - 2.00*pow(theta,2) - 1.5*pow(theta,3) + 0.50*pow(theta,4);
          w[3] =  1. - 1.5*theta - 0.25*pow(theta,2) + 1.0*pow(theta,3) - 0.25*pow(theta,4);
          weight_bc =  -(3. - 2.5*theta + 0.5*pow(theta,2));

//          // least-squares based
//          double den = 2 + 6 *theta+ 15 *pow(theta, 2) + 12 *pow(theta,3) + 3 *pow(theta,4);
//          w[1] = theta*(-13 - theta+ 10 *pow(theta, 2) + 4 *pow(theta,3))/den;
//          w[2] = (-6 - 6 *theta+ 5 *pow(theta, 2) + 6 *pow(theta,3) + pow(theta,4))/den;
//          w[3] = (2 +  3 *theta+ pow(theta, 2) - 4 *pow(theta,3) - 2 *pow(theta,4))/den;
//          weight_bc = -(6 + 22 *theta+ 10 *pow(theta, 2))/den;

        } else if (gf_order_ >= 2 && num_good_neis >= 2 && gf_stabilized_ != 1) { // quadratic native

          w[1] = -2.*(1.-theta)/theta;
          w[2] = (1.-theta)/(1.+theta);
          weight_bc = -2./theta/(1.+theta);

        } else if (gf_order_ >= 1 && num_good_neis >= 2 && gf_stabilized_ != 0) { // linear continuous

          // double interpolation
          w[1] = -(1.-theta)*theta;
          w[2] = -pow(1.-theta, 2.);
          weight_bc = -(2.-theta);

//          // least-squares based
//          double den = 1 + 2*theta + 2*theta*theta;
//          w[1] = ((-1 + theta)*theta)/den;
//          w[2] = (-1 + theta*theta)/den;
//          weight_bc = -(2 + 3*theta)/den;

        } else { // linear native

          w[1] = -(1.-theta)/theta;
          weight_bc = -1./theta;

//          w[1] = 0;
//          weight_bc = -1;

        }

        for (int i = 0; i < num_good_neis+1; ++i) {
          if (w[i] != 0) {
            row_gf_ghost->push_back(mat_entry_t(global_idx[i], w[i]));
          }
        }

        mask_ptr[n] = -.25;
//        mask_ptr[n] =  1.;
        d_nnz_gf_ghost += gf_stencil_size()-1;
        o_nnz_gf_ghost += gf_stencil_size()-1;

        // save information for quick rhs setup
        foreach_dimension(dim) {
          xyz_bc[dim] = xyz_C[dim] + (1.-theta)*delta_xyz[dim];
//          xyz_bc[dim] = xyz_C[dim] ;
        }

        bc_[phi_idx].add_fd_pt(n, nei, dist, xyz_bc, weight_bc);

      } else {

        int idx = bc_[phi_idx].idx_value_pt(n,0);
        interface_point_cartesian_t *pt = &bc_[phi_idx].dirichlet_pts[idx];

        weight_bc = bc_[phi_idx].dirichlet_weights[idx];
        pt->get_xyz(xyz_bc);
      }

      if (setup_rhs) {
        double bc_value;
        if (bc_[phi_idx].pointwise) {
          int idx = bc_[phi_idx].idx_value_pt(n, 0);
          bc_value = (bc_[phi_idx].value_pw)->at(idx);
        } else {
          bc_value = (bc_[phi_idx].value_cf)->value(xyz_bc);
        }
        rhs_gf_ptr[n] = bc_value*weight_bc;
      }
    }
  } else {

    double mue_000 = var_mu_ ? mue_ptr[n] : mu;

    //---------------------------------------------------------------------
    // compute submat_diag
    //---------------------------------------------------------------------
    if (new_submat_diag_ && there_is_diag_) {
      submat_diag_ptr[n] = diag_add;
      if (fabs(diag_add) > EPS) {
        matrix_has_nullspace_ = false;
      }
    }

    //---------------------------------------------------------------------
    // compute submat_main
    //---------------------------------------------------------------------
    if (new_submat_main_)
    {
      // neighboring nodes information
      p4est_locidx_t node [P4EST_FACES]; // node numbers
      double         d    [P4EST_FACES]; // distances
      double         w    [P4EST_FACES]; // discretization weights
      double         mue  [P4EST_FACES]; // diffusion coefficients

      foreach_direction (nei) {
        node [nei] = qnnn.neighbor(nei);
        d    [nei] = qnnn.distance(nei);
        w    [nei] = 0;
        mue  [nei] = mue_000;
      }

      if (var_mu_) { // interpolate diffusion coefficient if needed
        qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue);
      }

      // discretization of Laplace operator
      // note: if node is at wall, what's below will apply Neumann BC
      double w_000 = 0;
      foreach_dimension (dim) {
        int dir_m = dim*2;
        int dir_p = dim*2+1;

        if      (is_wall[dir_m]) { w[dir_p] = -2.0*mue_000/(d[dir_p]*d[dir_p]); }
        else if (is_wall[dir_p]) { w[dir_m] = -2.0*mue_000/(d[dir_m]*d[dir_m]); }
        else {
          w[dir_m] = -(mue_000+mue[dir_m])/d[dir_m]/(d[dir_m]+d[dir_p]);
          w[dir_p] = -(mue_000+mue[dir_p])/d[dir_p]/(d[dir_m]+d[dir_p]);
        }

        w_000 -= w[dir_m] + w[dir_p];
      }

      //---------------------------------------------------------------------
      // add coefficients in the matrix
      //---------------------------------------------------------------------
      mat_entry_t ent; ent.n = petsc_gloidx_[qnnn.node_000]; ent.val = w_000; row_main->push_back(ent);

      foreach_direction (nei) {
        if (!is_wall[nei]) {
          ent.n   = petsc_gloidx_[node[nei]];
          ent.val = w[nei];

          if (bdry_.phi_eff_ptr[node[nei]] <= -gf_thresh_*diag_min_) {
            row_main->push_back(ent);
            node[nei] < nodes_->num_owned_indeps ? d_nnz_main++ : o_nnz_main++;
          } else {
            row_gf->push_back(ent);
            node[nei] < nodes_->num_owned_indeps ? d_nnz_gf++ : o_nnz_gf++;
          }

        }
      }

      if (petsc_gloidx_[n] < fixed_value_idx_g_) {
        fixed_value_idx_l_ = n;
        fixed_value_idx_g_ = petsc_gloidx_[n];
      }

      mask_ptr[n] = -1.;
    }

    //---------------------------------------------------------------------
    // compute rhs
    //---------------------------------------------------------------------
    if (setup_rhs)
    {
      rhs_ptr[n] = rhs_loc_ptr[n];

      // add to rhs wall conditions
      XCODE( double eps_x = is_wall[dir::f_m00] ? 2.*EPS*diag_min_ : (is_wall[dir::f_p00] ? -2.*EPS*diag_min_ : 0) );
      YCODE( double eps_y = is_wall[dir::f_0m0] ? 2.*EPS*diag_min_ : (is_wall[dir::f_0p0] ? -2.*EPS*diag_min_ : 0) );
      ZCODE( double eps_z = is_wall[dir::f_00m] ? 2.*EPS*diag_min_ : (is_wall[dir::f_00p] ? -2.*EPS*diag_min_ : 0) );

      if (is_wall[dir::f_m00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C, y_C+eps_y, z_C+eps_z) ) / qnnn.d_p00;
      if (is_wall[dir::f_p00]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C, y_C+eps_y, z_C+eps_z) ) / qnnn.d_m00;

      if (is_wall[dir::f_0m0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C, z_C+eps_z) ) / qnnn.d_0p0;
      if (is_wall[dir::f_0p0]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C, z_C+eps_z) ) / qnnn.d_0m0;
#ifdef P4_TO_P8
      if (is_wall[dir::f_00m]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C+eps_y, z_C) ) / qnnn.d_00p;
      if (is_wall[dir::f_00p]) rhs_ptr[n] += 2.*mue_000*(*wc_value_)( DIM(x_C+eps_x, y_C+eps_y, z_C) ) / qnnn.d_00m;
#endif
    }
  }
}

void my_p4est_poisson_nodes_mls_t::discretize_robin(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                                                    double infc_phi_eff_000, bool is_wall[],
                                                    std::vector<mat_entry_t> *row_main, int &d_nnz_main, int &o_nnz_main,
                                                    std::vector<mat_entry_t> *row_robin_sc, int &d_nnz_robin_sc, int &o_nnz_robin_sc)
{
  double  mu;
  double *mue_ptr;
  double  diag_add;
  double *mask_ptr;
  double *rhs_loc_ptr;
  double *areas_ptr;
  double *volumes_ptr;
  CF_DIM *mu_cf;

  interpolators_prepare(n);

  if (infc_phi_eff_000 < 0)
  {
    mu          = mu_m_;
    mue_ptr     = mue_m_ptr;
    diag_add    = var_diag_ ? diag_m_ptr[n] : diag_m_scalar_;
    rhs_loc_ptr = rhs_m_ptr;
    mask_ptr    = mask_m_ptr;
    areas_ptr   = areas_m_ptr;
    volumes_ptr = volumes_m_ptr;
    mu_cf       = &mu_m_interp_;
  }
  else
  {
    mu          = mu_p_;
    mue_ptr     = mue_p_ptr;
    diag_add    = var_diag_ ? diag_p_ptr[n] : diag_p_scalar_;
    rhs_loc_ptr = rhs_p_ptr;
    mask_ptr    = mask_p_ptr;
    areas_ptr   = areas_p_ptr;
    volumes_ptr = volumes_p_ptr;
    mu_cf       = &mu_p_interp_;
  }

  double face_area_max   = 0;
  double volume_cut_cell = 0;

  double xyz_C[P4EST_DIM];
  node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

  double DIM( x_C = xyz_C[0],
              y_C = xyz_C[1],
              z_C = xyz_C[2] );

  double interp_min[P4EST_DIM] = { DIM(x_C, y_C, z_C) };
  double interp_max[P4EST_DIM] = { DIM(x_C, y_C, z_C) };

  foreach_dimension(dim)
  {
    if (!is_wall[0+2*dim]) interp_min[dim] -= dxyz_m_[dim];
    if (!is_wall[1+2*dim]) interp_max[dim] += dxyz_m_[dim];
  }

  bool           neighbors_exist[num_neighbors_cube];
  p4est_locidx_t neighbors      [num_neighbors_cube];

  ngbd_->get_all_neighbors(n, neighbors, neighbors_exist);

  // get geometry
  my_p4est_finite_volume_t fv;
  if (new_submat_main_)
  {
    if (finite_volumes_initialized_) fv = bdry_fvs_->at(bdry_node_to_fv_[n]);
    else construct_finite_volume(fv, n, p4est_, nodes_, bdry_phi_cf_, bdry_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);

    foreach_direction(i) face_area_max = MAX(face_area_max, fv.face_area[i]);
    volume_cut_cell = fv.volume;

    face_area_max /= face_area_scalling_;

    areas_ptr  [n] = face_area_max;
    volumes_ptr[n] = volume_cut_cell;
  }
  else
  {
    face_area_max   = areas_ptr  [n];
    volume_cut_cell = volumes_ptr[n];
  }

  if (face_area_max > interface_rel_thresh_) // check if at least one face has a 'good connection' to neighboring cells
  {
    std::vector<int>               bdry_id;
    std::vector<double>            bdry_area;
    std::vector<interface_point_t> bdry_xyz;
    std::vector<interface_point_t> bdry_robin_xyz;

    std::vector<int>               wall_dir;
    std::vector<double>            wall_area;
    std::vector<interface_point_t> wall_xyz;

    if (new_submat_main_)
    {
      // get information about boundaries
      for (int i=0; i<fv.interfaces.size(); ++i)
      {
        interface_point_t pt(DIM(xyz_C[0] + fv.interfaces[i].centroid[0],
                                 xyz_C[1] + fv.interfaces[i].centroid[1],
                                 xyz_C[2] + fv.interfaces[i].centroid[2]));

        bdry_id  .push_back(fv.interfaces[i].id  );
        bdry_area.push_back(fv.interfaces[i].area);
        bdry_xyz .push_back(pt);
      }

      // project interface centroids onto interfaces
      for (int i=0; i<bdry_id.size(); ++i)
      {
        int phi_idx = bdry_id[i];
        interface_point_t *pt = &bdry_xyz[i];

        XCODE( pt->xyz[0] = MIN(pt->xyz[0], interp_max[0]); pt->xyz[0] = MAX(pt->xyz[0], interp_min[0]) );
        YCODE( pt->xyz[1] = MIN(pt->xyz[1], interp_max[1]); pt->xyz[1] = MAX(pt->xyz[1], interp_min[1]) );
        ZCODE( pt->xyz[2] = MIN(pt->xyz[2], interp_max[2]); pt->xyz[2] = MAX(pt->xyz[2], interp_min[2]) );

        // compute signed distance and normal at the centroid
        XCODE( double nx = qnnn.dx_central(bdry_.phi_ptr[phi_idx]) );
        YCODE( double ny = qnnn.dy_central(bdry_.phi_ptr[phi_idx]) );
        ZCODE( double nz = qnnn.dz_central(bdry_.phi_ptr[phi_idx]) );

        double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
        double dist = (*bdry_phi_cf_[phi_idx]).value(pt->xyz)/norm;

        XCODE( pt->xyz[0] -= dist*nx/norm );
        YCODE( pt->xyz[1] -= dist*ny/norm );
        ZCODE( pt->xyz[2] -= dist*nz/norm );

        XCODE( pt->xyz[0] = MIN(pt->xyz[0], interp_max[0]); pt->xyz[0] = MAX(pt->xyz[0], interp_min[0]) );
        YCODE( pt->xyz[1] = MIN(pt->xyz[1], interp_max[1]); pt->xyz[1] = MAX(pt->xyz[1], interp_min[1]) );
        ZCODE( pt->xyz[2] = MIN(pt->xyz[2], interp_max[2]); pt->xyz[2] = MAX(pt->xyz[2], interp_min[2]) );
      }

      // get information about walls
      foreach_direction(i)
      {
        if (is_wall[i] && fv.face_area[i] > 0)
        {
          interface_point_t pt(DIM(xyz_C[0] + fv.face_centroid_x[i],
                                   xyz_C[1] + fv.face_centroid_y[i],
                                   xyz_C[2] + fv.face_centroid_z[i]));

          wall_dir .push_back(i);
          wall_area.push_back(fv.face_area[i]);
          wall_xyz .push_back(pt);
        }
      }

      bdry_robin_xyz = bdry_xyz;
    }
    else
    {
      load_bdry_data(n, bdry_id,  bdry_area, bdry_xyz, bdry_robin_xyz);
      load_wall_data(n, wall_dir, wall_area, wall_xyz);
    }

    // check for Dirichlet BC at walls
    bool dirichlet_wall = false;
    if (wall_area.size() > 0)
      if (wc_type_->value(xyz_C) == DIRICHLET)
        dirichlet_wall = true;

    if (dirichlet_wall)
    {
      if (new_submat_main_) row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
      if (setup_rhs)        rhs_ptr[n] = wc_value_->value(xyz_C);
    }
    else
    {
      if (fv_scheme_ == 1)
      {
        for (unsigned short idx = 0; idx < num_neighbors_cube; ++idx)
        {
          if (neighbors_exist[idx])
          {
            neighbors_exist[idx] = neighbors_exist[idx]
                && (areas_ptr[neighbors[idx]] > interface_rel_thresh_);
          }
        }
      }

      //---------------------------------------------------------------------
      // assemble submat_diag
      //---------------------------------------------------------------------
      if (there_is_diag_ && new_submat_diag_)
      {
        submat_diag_ptr[n] = diag_add*volume_cut_cell;
        if (fabs(submat_diag_ptr[n]) > EPS) matrix_has_nullspace_ = false;
      }

      //---------------------------------------------------------------------
      // assemble submat_main
      //---------------------------------------------------------------------
      if (new_submat_main_)
      {
        std::vector<double> w(num_neighbors_cube, 0);

        bool   neighbors_exist_face[num_neighbors_face];
        bool   map_face            [num_neighbors_face];
        double weights_face        [num_neighbors_face];

        double theta = EPS;

        for (int dim=0; dim<P4EST_DIM; ++dim) // loop over all dimensions
        {
          for (int sign=0; sign<2; ++sign) // negative and positive directions
          {
            unsigned short dir = 2*dim + sign;

            if (fv.face_area[dir]/face_area_scalling_ > interface_rel_thresh_ &&
                !is_wall[dir])
            {
              if (!neighbors_exist[f2c_p[dir][nnf_00]])
              {
                std::cout << "Warning: neighbor doesn't exist in the " << dir << "-th direction."
                          << " Own number: " << n
                          << " Nei number: " << neighbors[f2c_p[dir][nnf_00]]
                          << " Face Area:" << fv.face_area[dir]/face_area_scalling_ << "\n";
              }
              else
              {
                double flux   = fv.face_area[dir] / dxyz_m_[dim] *
                    (var_mu_ ? (*mu_cf)(DIM( x_C + fv.face_centroid_x[dir],
                                             y_C + fv.face_centroid_y[dir],
                                             z_C + fv.face_centroid_z[dir] )) : mu);

                if (fv_scheme_ == 0)
                {
                  w[f2c_m[dir][nnf_00]] += flux;
                  w[f2c_p[dir][nnf_00]] -= flux;
                }
                else if (fv_scheme_ == 1)
                {
                  for (int nn=0; nn<num_neighbors_face; ++nn)
                  {
                    neighbors_exist_face[nn] = neighbors_exist[f2c_m[dir][nn]] && neighbors_exist[f2c_p[dir][nn]];
                  }

                  double centroid_xyz[] = { DIM( fv.face_centroid_x[dir]/dx_min_,
                                                 fv.face_centroid_y[dir]/dy_min_,
                                                 fv.face_centroid_z[dir]/dz_min_ ) };

                  CODE2D( double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]],                           neighbors_exist_face, weights_face, theta, map_face) );
                  CODE3D( double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], centroid_xyz[k_idx[dim]], neighbors_exist_face, weights_face, theta, map_face) );

                  mask_ptr[n] = MAX(mask_ptr[n], mask_result);

                  for (int nn=0; nn<num_neighbors_face_; ++nn)
                  {
                    if (map_face[nn])
                    {
                      w[f2c_m[dir][nn]] += weights_face[nn] * flux;
                      w[f2c_p[dir][nn]] -= weights_face[nn] * flux;
                    }
                  }
                }
                else throw;
              } // if neighbour exists
            } // if good connection
          } // sign
        } // dir

        // put weights into the matrix
        for (int i=0; i<num_neighbors_cube; ++i)
        {
          if (neighbors_exist[i] && fabs(w[i]) > EPS)
          {
            row_main->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w[i]));
            ( neighbors[i] < nodes_->num_owned_indeps ) ? d_nnz_main++ : o_nnz_main++;
          }
        }
      } // if (new_submat_main_)


      // deal with boundary conditions

      //---------------------------------------------------------------------
      // rhs from force term, neumann term and bc's on the walls
      //---------------------------------------------------------------------
      if (setup_rhs)
      {
        // forcing term
        rhs_ptr[n] = rhs_loc_ptr[n]*volume_cut_cell;

        // neumann flux through domain boundary
        for (int i=0; i<bdry_area.size(); ++i)
        {
          rhs_ptr[n] += bdry_area[i] * (bc_[bdry_id[i]].pointwise ? bc_[bdry_id[i]].get_value_pw(n,0) : bc_[bdry_id[i]].get_value_cf(bdry_xyz[i].xyz));
        }

        for (int i=0; i<wall_area.size(); ++i)
        {
          rhs_ptr[n] += wall_area[i] * (*wc_value_).value(wall_xyz[i].xyz);
        }
      }

      //---------------------------------------------------------------------
      // assemble submat_robin_sym, submat_robin_sc and rhs from robin term
      //---------------------------------------------------------------------
      if (there_is_robin_ && bdry_area.size() > 0)
      {
        bool sc_scheme_successful = false;
        if (fv_scheme_ == 1)
        {
          std::vector<double> w_robin(num_neighbors_cube, 0);
          double add_to_rhs = 0;

          sc_scheme_successful = true;

          // assemble linear system
          int bdry_offset     = num_neighbors_cube;
          int wall_offset     = num_neighbors_cube + bdry_area.size();
          int num_constraints = num_neighbors_cube + bdry_area.size() + wall_area.size();

          _CODE( std::vector<double> col_c(num_constraints, 0) );
          XCODE( std::vector<double> col_x(num_constraints, 0) );
          YCODE( std::vector<double> col_y(num_constraints, 0) );
          ZCODE( std::vector<double> col_z(num_constraints, 0) );

          std::vector<double> bc_values(bdry_area.size(), 0);
          std::vector<double> bc_coeffs(bdry_area.size(), 0);

          std::vector<double> wc_values(wall_area.size(), 0);
          std::vector<double> wc_coeffs(wall_area.size(), 0);

          // grid nodes
          ZFOR (char k = 0; k < 3; ++k) {
            YFOR (char j = 0; j < 3; ++j) {
              XFOR (char i = 0; i < 3; ++i)
              {
                char idx = i + 3*j CODE3D( + 9*k );

                _CODE( col_c[idx] = 1. );
                XCODE( col_x[idx] = ((double) (i-1)) * dxyz_m_[0] );
                YCODE( col_y[idx] = ((double) (j-1)) * dxyz_m_[1] );
                ZCODE( col_z[idx] = ((double) (k-1)) * dxyz_m_[2] );
              }
            }
          }

          // bdry pieces
          for (int i=0; i<bdry_area.size(); ++i)
          {
            int id = bdry_id[i];

            double normal[P4EST_DIM]; compute_normals(qnnn, bdry_.phi_ptr[id], normal);
            double xyz_pr[P4EST_DIM]; bdry_xyz[i].get_xyz(xyz_pr);

            bdry_robin_xyz[i].set(xyz_pr);

            double mu_proj       = var_mu_ ? mu_cf->value(xyz_pr) : mu;
            double bc_coeff_proj = bc_[id].pointwise ? bc_[id].get_robin_pw_coeff(n) : bc_[id].get_coeff_cf(xyz_pr);
            double bc_value_proj = bc_[id].pointwise ? bc_[id].get_robin_pw_value(n) : bc_[id].get_value_cf(xyz_pr);

            _CODE( col_c[bdry_offset + i] = bc_coeff_proj );
            XCODE( col_x[bdry_offset + i] = bc_coeff_proj*(xyz_pr[0] - xyz_C[0]) + mu_proj*normal[0] );
            YCODE( col_y[bdry_offset + i] = bc_coeff_proj*(xyz_pr[1] - xyz_C[1]) + mu_proj*normal[1] );
            ZCODE( col_z[bdry_offset + i] = bc_coeff_proj*(xyz_pr[2] - xyz_C[2]) + mu_proj*normal[2] );

            bc_coeffs[i] = bc_coeff_proj;
            bc_values[i] = bc_value_proj;
          }

          // wall pieces
          for (int i=0; i<wall_area.size(); ++i)
          {
            double normal[P4EST_DIM]; compute_wall_normal(wall_dir[i], normal);
            double xyz_pr[P4EST_DIM]; wall_xyz[i].get_xyz(xyz_pr);

            double mu_proj       = var_mu_ ? mu_cf->value(xyz_pr) : mu;
            double bc_coeff_proj = 0;
            double bc_value_proj = wc_value_->value(xyz_pr);

            _CODE( col_c[wall_offset + i] = bc_coeff_proj );
            XCODE( col_x[wall_offset + i] = bc_coeff_proj*(xyz_pr[0] - xyz_C[0]) + mu_proj*normal[0] );
            YCODE( col_y[wall_offset + i] = bc_coeff_proj*(xyz_pr[1] - xyz_C[1]) + mu_proj*normal[1] );
            ZCODE( col_z[wall_offset + i] = bc_coeff_proj*(xyz_pr[2] - xyz_C[2]) + mu_proj*normal[2] );

            wc_coeffs[i] = bc_coeff_proj;
            wc_values[i] = bc_value_proj;
          }

          double gamma = 3.*log(10.);

          // loop through all present interfaces and interpolate separately for each of them
          for (int bdry_it=0; bdry_it<bdry_area.size(); ++bdry_it)
          {
            if (sc_scheme_successful)
            {
              interface_point_t *centroid_pr = &bdry_xyz[bdry_it];

              // assemble matrix of weights
              std::vector<double>  weight(num_constraints, 0);
              unsigned short       num_constraints_present = 0;

              // grid nodes
              ZFOR (unsigned short k = 0; k < 3; ++k) {
                YFOR (unsigned short j = 0; j < 3; ++j) {
                  XFOR (unsigned short i = 0; i < 3; ++i)
                  {
                    unsigned short idx = i + 3*j ONLY3D( + 9*k );

                    XCODE( double dx = ((double) (i-1)) * dx_min_ );
                    YCODE( double dy = ((double) (j-1)) * dy_min_ );
                    ZCODE( double dz = ((double) (k-1)) * dz_min_ );

                    if (neighbors_exist[idx])
                    {
                      num_constraints_present++;
                      weight[idx] = exp(-gamma*SUMD(SQR((x_C+dx - centroid_pr->x())/dx_min_),
                                                    SQR((y_C+dy - centroid_pr->y())/dy_min_),
                                                    SQR((z_C+dz - centroid_pr->z())/dz_min_)));
                    }
                  }
                }
              }

              // bdry pieces
              for (int i=0; i<bdry_area.size(); ++i)
              {
                num_constraints_present++;
                interface_point_t *centroid_pr_other = &bdry_xyz[i];
                weight[bdry_offset + i] = exp(-gamma*SUMD(SQR((centroid_pr_other->x() - centroid_pr->x())/dx_min_),
                                                          SQR((centroid_pr_other->y() - centroid_pr->y())/dy_min_),
                                                          SQR((centroid_pr_other->z() - centroid_pr->z())/dz_min_)));
              }

              // wall pieces
              for (int i=0; i<wall_area.size(); ++i)
              {
                num_constraints_present++;
                interface_point_t *centroid_pr_other = &wall_xyz[i];
                weight[wall_offset + i] = exp(-gamma*SUMD(SQR((centroid_pr_other->x() - centroid_pr->x())/dx_min_),
                                                          SQR((centroid_pr_other->y() - centroid_pr->y())/dy_min_),
                                                          SQR((centroid_pr_other->z() - centroid_pr->z())/dz_min_)));
              }

              if (num_constraints_present <= P4EST_DIM+1)
              {
                sc_scheme_successful = false;
                continue;
              }

              // compute AtWA and invert
              // TODO: replace the code below by a single function
              unsigned short A_size = (P4EST_DIM+1);
              double A[(P4EST_DIM+1)*(P4EST_DIM+1)];
              double A_inv[(P4EST_DIM+1)*(P4EST_DIM+1)];

              A[0*A_size + 0] = 0;
              A[0*A_size + 1] = 0;
              A[0*A_size + 2] = 0;
              A[1*A_size + 1] = 0;
              A[1*A_size + 2] = 0;
              A[2*A_size + 2] = 0;
#ifdef P4_TO_P8
              A[0*A_size + 3] = 0;
              A[1*A_size + 3] = 0;
              A[2*A_size + 3] = 0;
              A[3*A_size + 3] = 0;
#endif

              for (int nei=0; nei<num_constraints; ++nei)
              {
                A[0*A_size + 0] += col_c[nei]*col_c[nei]*weight[nei];
                A[0*A_size + 1] += col_c[nei]*col_x[nei]*weight[nei];
                A[0*A_size + 2] += col_c[nei]*col_y[nei]*weight[nei];
                A[1*A_size + 1] += col_x[nei]*col_x[nei]*weight[nei];
                A[1*A_size + 2] += col_x[nei]*col_y[nei]*weight[nei];
                A[2*A_size + 2] += col_y[nei]*col_y[nei]*weight[nei];
#ifdef P4_TO_P8
                A[0*A_size + 3] += col_c[nei]*col_z[nei]*weight[nei];
                A[1*A_size + 3] += col_x[nei]*col_z[nei]*weight[nei];
                A[2*A_size + 3] += col_y[nei]*col_z[nei]*weight[nei];
                A[3*A_size + 3] += col_z[nei]*col_z[nei]*weight[nei];
#endif
              }

              A[1*A_size + 0] = A[0*A_size + 1];
              A[2*A_size + 0] = A[0*A_size + 2];
              A[2*A_size + 1] = A[1*A_size + 2];
#ifdef P4_TO_P8
              A[3*A_size + 0] = A[0*A_size + 3];
              A[3*A_size + 1] = A[1*A_size + 3];
              A[3*A_size + 2] = A[2*A_size + 3];
#endif

              CODE2D( if (!inv_mat3(A, A_inv)) throw std::domain_error("Error: singular LSQR matrix\n"));
              CODE3D( if (!inv_mat4(A, A_inv)) throw std::domain_error("Error: singular LSQR matrix\n"));

              // compute Taylor expansion coefficients
              _CODE( std::vector<double> coeff_c_term(num_constraints, 0) );
              XCODE( std::vector<double> coeff_x_term(num_constraints, 0) );
              YCODE( std::vector<double> coeff_y_term(num_constraints, 0) );
              ZCODE( std::vector<double> coeff_z_term(num_constraints, 0) );

              for (int nei=0; nei<num_constraints; ++nei)
              {
                _CODE( coeff_c_term[nei] = weight[nei]*( A_inv[0*A_size+0]*col_c[nei] + SUMD(A_inv[0*A_size+1]*col_x[nei], A_inv[0*A_size+2]*col_y[nei], A_inv[0*A_size+3]*col_z[nei]) ) );
                XCODE( coeff_x_term[nei] = weight[nei]*( A_inv[0*A_size+1]*col_c[nei] + SUMD(A_inv[1*A_size+1]*col_x[nei], A_inv[1*A_size+2]*col_y[nei], A_inv[1*A_size+3]*col_z[nei]) ) );
                YCODE( coeff_y_term[nei] = weight[nei]*( A_inv[0*A_size+2]*col_c[nei] + SUMD(A_inv[2*A_size+1]*col_x[nei], A_inv[2*A_size+2]*col_y[nei], A_inv[2*A_size+3]*col_z[nei]) ) );
                ZCODE( coeff_z_term[nei] = weight[nei]*( A_inv[0*A_size+3]*col_c[nei] + SUMD(A_inv[3*A_size+1]*col_x[nei], A_inv[3*A_size+2]*col_y[nei], A_inv[3*A_size+3]*col_z[nei]) ) );
              }

              _CODE( double rhs_c_term = 0 );
              XCODE( double rhs_x_term = 0 );
              YCODE( double rhs_y_term = 0 );
              ZCODE( double rhs_z_term = 0 );

              // bdry pieces
              for (int i=0; i<bdry_area.size(); ++i)
              {
                _CODE( rhs_c_term += coeff_c_term[bdry_offset + i] * bc_values[i] );
                XCODE( rhs_x_term += coeff_x_term[bdry_offset + i] * bc_values[i] );
                YCODE( rhs_y_term += coeff_y_term[bdry_offset + i] * bc_values[i] );
                ZCODE( rhs_z_term += coeff_z_term[bdry_offset + i] * bc_values[i] );
              }

              // wall pieces
              for (int i=0; i<wall_area.size(); ++i)
              {
                _CODE( rhs_c_term += coeff_c_term[wall_offset + i] * wc_values[i] );
                XCODE( rhs_x_term += coeff_x_term[wall_offset + i] * wc_values[i] );
                YCODE( rhs_y_term += coeff_y_term[wall_offset + i] * wc_values[i] );
                ZCODE( rhs_z_term += coeff_z_term[wall_offset + i] * wc_values[i] );
              }

              // compute integrals
              _CODE( double c_term = bdry_area[bdry_it]*bc_coeffs[bdry_it] );
              XCODE( double x_term = bdry_area[bdry_it]*bc_coeffs[bdry_it]*(centroid_pr->x() - xyz_C[0]) );
              YCODE( double y_term = bdry_area[bdry_it]*bc_coeffs[bdry_it]*(centroid_pr->y() - xyz_C[1]) );
              ZCODE( double z_term = bdry_area[bdry_it]*bc_coeffs[bdry_it]*(centroid_pr->z() - xyz_C[2]) );

              // matrix coefficients
              if (new_submat_robin_)
              {
                for (int nei=0; nei<num_neighbors_cube; ++nei)
                {
                  w_robin[nei] += coeff_c_term[nei]*c_term + SUMD( coeff_x_term[nei]*x_term,
                                                                   coeff_y_term[nei]*y_term,
                                                                   coeff_z_term[nei]*z_term );
                }

                if (fabs(c_term) > 0) matrix_has_nullspace_ = false;
              }

              if (setup_rhs) add_to_rhs -= rhs_c_term*c_term + SUMD( rhs_x_term*x_term,
                                                                     rhs_y_term*y_term,
                                                                     rhs_z_term*z_term );

            }
          }

          // put values into matrix
          if (sc_scheme_successful)
          {
            if (new_submat_robin_)
            {
              for (int i=0; i<num_neighbors_cube; ++i)
              {
                if (neighbors_exist[i])
                {
                  row_robin_sc->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_robin[i]));
                  (neighbors[i] < nodes_->num_owned_indeps) ? d_nnz_robin_sc++ : o_nnz_robin_sc++;
                }
              }
            }

            if (setup_rhs) rhs_ptr[n] += add_to_rhs;
          }
        }

        //*/

        if ((fv_scheme_ == 0) || !sc_scheme_successful)
        {
          double add_to_matrix = 0;

          if (new_submat_robin_ && (fv_scheme_ == 1))
          {
            mask_ptr[n] = MAX(mask_ptr[n], -0.1);
            std::cout << "Fallback Robin BC\n";
          }

          // code below asumes that we have only Robin or Neumann BC in a cell
          //---------------------------------------------------------------------
          // contribution through interfaces
          //---------------------------------------------------------------------

          // Special treatment for kinks
          if (kink_special_treatment_ && bdry_area.size() + wall_area.size() > 1)
          {
            double N_mat[P4EST_DIM*P4EST_DIM];
            /* TO FIX (case of > 3 interfaces in 3D):
           * Should be N_mat[P4EST_DIM*num_normals],
           * where num_normals = max(P4EST_DIM, num_ifaces);
           */

            double bc_coeffs[P4EST_DIM];
            double bc_values[P4EST_DIM];
            double mu_values[P4EST_DIM];
            double a_coeff[P4EST_DIM];
            double b_coeff[P4EST_DIM];

            // bdry pieces
            for (int i=0; i<bdry_area.size(); ++i)
            {
              int id = bdry_id[i];

              double normal[P4EST_DIM]; compute_normals(qnnn, bdry_.phi_ptr[id], normal);
              double xyz_pr[P4EST_DIM]; bdry_xyz[i].get_xyz(xyz_pr);

              mu_values[i] = var_mu_ ? mu_cf->value(xyz_pr) : mu;
              bc_coeffs[i] = bc_[id].pointwise ? bc_[id].get_robin_pw_coeff(n) : bc_[id].get_coeff_cf(xyz_pr);
              bc_values[i] = bc_[id].pointwise ? bc_[id].get_robin_pw_value(n) : bc_[id].get_value_cf(xyz_pr);

              EXECD( N_mat[i*P4EST_DIM + 0] = normal[0],
                     N_mat[i*P4EST_DIM + 1] = normal[1],
                     N_mat[i*P4EST_DIM + 2] = normal[2] );
            }

            // wall pieces
            int wall_offset = bdry_area.size();
            for (int i=0; i<wall_area.size(); ++i)
            {
              double normal[P4EST_DIM]; compute_wall_normal(wall_dir[i], normal);
              double xyz_pr[P4EST_DIM]; wall_xyz[i].get_xyz(xyz_pr);

              mu_values[wall_offset + i] = var_mu_ ? mu_cf->value(xyz_pr) : mu;
              bc_coeffs[wall_offset + i] = 0;
              bc_values[wall_offset + i] = wc_value_->value(xyz_pr);

              EXECD( N_mat[(wall_offset+i)*P4EST_DIM + 0] = normal[0],
                     N_mat[(wall_offset+i)*P4EST_DIM + 1] = normal[1],
                     N_mat[(wall_offset+i)*P4EST_DIM + 2] = normal[2] );
            }

#ifdef P4_TO_P8
            /* an ad-hoc: in case of an intersection of 2 interfaces in 3D
         * we choose the third direction to be perpendicular to the first two
         * and set du/dn = 0 (i.e., u = const) along this third direction
         */
            if (bdry_area.size() + wall_area.size() == 2)
            {
              N_mat[2*P4EST_DIM + 0] = N_mat[0*P4EST_DIM + 1]*N_mat[1*P4EST_DIM + 2] - N_mat[0*P4EST_DIM + 2]*N_mat[1*P4EST_DIM + 1];
              N_mat[2*P4EST_DIM + 1] = N_mat[0*P4EST_DIM + 2]*N_mat[1*P4EST_DIM + 0] - N_mat[0*P4EST_DIM + 0]*N_mat[1*P4EST_DIM + 2];
              N_mat[2*P4EST_DIM + 2] = N_mat[0*P4EST_DIM + 0]*N_mat[1*P4EST_DIM + 1] - N_mat[0*P4EST_DIM + 1]*N_mat[1*P4EST_DIM + 0];

              double norm = sqrt(SQR(N_mat[2*P4EST_DIM + 0]) + SQR(N_mat[2*P4EST_DIM + 1]) + SQR(N_mat[2*P4EST_DIM + 2]));
              N_mat[2*P4EST_DIM + 0] /= norm;
              N_mat[2*P4EST_DIM + 1] /= norm;
              N_mat[2*P4EST_DIM + 2] /= norm;

              mu_values[2] = 1;
              bc_coeffs[2] = 0;
              bc_values[2] = 0;
            }
#endif

            for (int i=0; i<bdry_area.size() + wall_area.size(); ++i)
            {
              double xyz_pr[P4EST_DIM];
              if (i < bdry_area.size()) bdry_xyz[i].get_xyz(xyz_pr);
              else                      wall_xyz[i-bdry_area.size()].get_xyz(xyz_pr);

              foreach_dimension(dim)
              {
                N_mat[i*P4EST_DIM+dim] = N_mat[i*P4EST_DIM+dim]*mu_values[i] + bc_coeffs[i]*(xyz_pr[dim] - xyz_C[dim]);
              }
            }

            // Solve matrix
            double N_inv_mat[P4EST_DIM*P4EST_DIM];
            /* TO FIX (case of > 3 interfaces in 3D):
         * one should solve an overdetermined matrix N by the least-squares approach
         */
            CODE2D( inv_mat2(N_mat, N_inv_mat) );
            CODE3D( inv_mat3(N_mat, N_inv_mat) );

            // calculate coefficients in Taylor series of u
            for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
            {
              a_coeff[i_dim] = 0;
              b_coeff[i_dim] = 0;
              for (short j_dim = 0; j_dim < P4EST_DIM; j_dim++)
              {
                a_coeff[i_dim] += N_inv_mat[i_dim*P4EST_DIM + j_dim]*bc_coeffs[j_dim];
                b_coeff[i_dim] += N_inv_mat[i_dim*P4EST_DIM + j_dim]*bc_values[j_dim];
              }
            }

            // compute integrals
            for (int i=0; i<bdry_area.size(); ++i)
            {
              double xyz_pr[P4EST_DIM]; bdry_xyz[i].get_xyz(xyz_pr);

              add_to_matrix += bc_coeffs[i]*bdry_area[i]*
                  (1.-SUMD(a_coeff[0]*(xyz_pr[0]-xyz_C[0]),
                           a_coeff[1]*(xyz_pr[1]-xyz_C[1]),
                           a_coeff[2]*(xyz_pr[2]-xyz_C[2])));

              if (setup_rhs)
                rhs_ptr[n] -= bc_coeffs[i]*bdry_area[i]*
                    SUMD(b_coeff[0]*(xyz_pr[0]-xyz_C[0]),
                         b_coeff[1]*(xyz_pr[1]-xyz_C[1]),
                         b_coeff[2]*(xyz_pr[2]-xyz_C[2]));
            }

          }
          else // Cells without kinks
          {
            for (int i=0; i<bdry_area.size(); i++)
            {
              int id = bdry_id[i];

              if (bc_[id].type == ROBIN)
              {
                // compute projection point
                double dxyz_pr[P4EST_DIM];
                double  xyz_pr[P4EST_DIM];
                double dist = 0;
                if (use_centroid_always_)
                {
                  double normal[P4EST_DIM];
                  compute_normals(qnnn, bdry_.phi_ptr[id], normal);
                  bdry_xyz[i].get_xyz(xyz_pr);
                  foreach_dimension(dim) dist += normal[dim]*(xyz_C[dim] - xyz_pr[dim]);
                }
                else
                {
                  find_projection(qnnn, bdry_.phi_ptr[id], dxyz_pr, dist);
                  foreach_dimension(dim) xyz_pr[dim] = xyz_C[dim] + dxyz_pr[dim];
                }

                XCODE( xyz_pr[0] = MIN(xyz_pr[0], interp_max[0]); xyz_pr[0] = MAX(xyz_pr[0], interp_min[0]) );
                YCODE( xyz_pr[1] = MIN(xyz_pr[1], interp_max[1]); xyz_pr[1] = MAX(xyz_pr[1], interp_min[1]) );
                ZCODE( xyz_pr[2] = MIN(xyz_pr[2], interp_max[2]); xyz_pr[2] = MAX(xyz_pr[2], interp_min[2]) );

                bdry_robin_xyz[i].set(xyz_pr);

                // sample values at the projection point
                double mu_proj       = var_mu_ ? mu_cf->value(xyz_pr) : mu;
                double bc_coeff_proj = bc_[id].pointwise ? bc_[id].get_robin_pw_coeff(n) : bc_[id].get_coeff_cf(xyz_pr);
                double bc_value_proj = bc_[id].pointwise ? bc_[id].get_robin_pw_value(n) : bc_[id].get_value_cf(xyz_pr);

                // addition to diagonal term
                if (use_taylor_correction_) { add_to_matrix += bc_coeff_proj*bdry_area[i]/(1.-bc_coeff_proj*dist/mu_proj); }
                else                        { add_to_matrix += bc_coeff_proj*bdry_area[i]; }

                // addition to right-hand-side
                if (setup_rhs && use_taylor_correction_)
                {
                  rhs_ptr[n] -= bdry_area[i]*bc_coeff_proj*bc_value_proj*dist/(bc_coeff_proj*dist-mu_proj);
                }

                if (fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
              }
            }
          }

          if (new_submat_robin_)
          {
            if      (fv_scheme_ == 1) row_robin_sc->push_back(mat_entry_t(petsc_gloidx_[n], add_to_matrix));
            else if (fv_scheme_ == 0) submat_robin_sym_ptr[n] = add_to_matrix;
            else throw;
          }

        } // end of symmetric scheme
      }
    }

    // save info about interfaces
    if (new_submat_main_)
    {
      save_bdry_data(n, bdry_id,  bdry_area, bdry_xyz, bdry_robin_xyz);
      save_wall_data(n, wall_dir, wall_area, wall_xyz);
    }
  }
  else // if finite volume too small, ignore the node
  {
    if (new_submat_main_)
    {
      mask_ptr[n] = MAX(1., mask_ptr[n]);
      row_main->push_back(mat_entry_t(petsc_gloidx_[n], 1));
      if (volume_cut_cell != 0.) { ierr = PetscPrintf(p4est_->mpicomm, "Ignoring tiny volume %e with max face area %e\n", volume_cut_cell, face_area_max); CHKERRXX(ierr); }
    }

    if (setup_rhs) rhs_ptr[n] = 0;
  }
}

void my_p4est_poisson_nodes_mls_t::discretize_jump(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                                                   bool is_wall[],
                                                   std::vector<mat_entry_t> *row_main, int& d_nnz_main, int& o_nnz_main,
                                                   std::vector<mat_entry_t> *row_jump, int& d_nnz_jump, int& o_nnz_jump,
                                                   std::vector<mat_entry_t> *row_jump_ghost, int& d_nnz_jump_ghost, int& o_nnz_jump_ghost)
{
  interpolators_prepare(n);

  my_p4est_finite_volume_t fv_m;
  my_p4est_finite_volume_t fv_p;

  double xyz_C[P4EST_DIM];
  node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

  double DIM( x_C = xyz_C[0],
              y_C = xyz_C[1],
              z_C = xyz_C[2] );

  double interp_min[P4EST_DIM] = { DIM(x_C, y_C, z_C) };
  double interp_max[P4EST_DIM] = { DIM(x_C, y_C, z_C) };

  foreach_dimension(dim)
  {
    if (!is_wall[0+2*dim]) interp_min[dim] -= dxyz_m_[dim];
    if (!is_wall[1+2*dim]) interp_max[dim] += dxyz_m_[dim];
  }

  double face_m_area_max = 0;
  double face_p_area_max = 0;

  double volume_cut_cell_m = 0;
  double volume_cut_cell_p = 0;

  if (new_submat_main_)
  {
    // compute geometric info about negative region
    if (finite_volumes_initialized_) fv_m = infc_fvs_->at(infc_node_to_fv_[n]);
    else construct_finite_volume(fv_m, n, p4est_, nodes_, infc_phi_cf_, infc_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);

    // compute geometric info about positive region
    fv_p.full_cell_volume = fv_m.full_cell_volume;
    fv_p.volume           = fv_m.full_cell_volume - fv_m.volume;

    foreach_direction(i)
    {
      fv_p.full_face_area[i] = fv_m.full_face_area[i];
      fv_p.face_area     [i] = fv_m.full_face_area[i] - fv_m.face_area[i];

      XCODE( fv_p.face_centroid_x[i] = fv_m.face_centroid_x[i] );
      YCODE( fv_p.face_centroid_y[i] = fv_m.face_centroid_y[i] );
      ZCODE( fv_p.face_centroid_z[i] = fv_m.face_centroid_z[i] );

      if (fv_p.face_area[i]/face_area_scalling_ > interface_rel_thresh_)
      {
        XCODE( if (i != dir::f_m00 && i != dir::f_p00) fv_p.face_centroid_x[i] *= -fv_m.face_area[i]/fv_p.face_area[i] );
        YCODE( if (i != dir::f_0m0 && i != dir::f_0p0) fv_p.face_centroid_y[i] *= -fv_m.face_area[i]/fv_p.face_area[i] );
        ZCODE( if (i != dir::f_00m && i != dir::f_00p) fv_p.face_centroid_z[i] *= -fv_m.face_area[i]/fv_p.face_area[i] );
      }
    }

    //---------------------------------------------------------------------
    // compute connections
    //---------------------------------------------------------------------
    volume_cut_cell_m = fv_m.volume;
    volume_cut_cell_p = fv_p.volume;

    foreach_direction(i)
    {
      face_m_area_max =  MAX(face_m_area_max, fabs(fv_m.face_area[i]));
      face_p_area_max =  MAX(face_p_area_max, fabs(fv_p.face_area[i]));
    }

    face_m_area_max /= face_area_scalling_;
    face_p_area_max /= face_area_scalling_;

    // save
    areas_m_ptr[n] = face_m_area_max;
    areas_p_ptr[n] = face_p_area_max;

    volumes_m_ptr[n] = volume_cut_cell_m;
    volumes_p_ptr[n] = volume_cut_cell_p;
  }
  else
  {
    face_m_area_max = areas_m_ptr[n];
    face_p_area_max = areas_p_ptr[n];

    volume_cut_cell_m = volumes_m_ptr[n];
    volume_cut_cell_p = volumes_p_ptr[n];
  }


  bool           neighbors_exist  [num_neighbors_cube];
  bool           neighbors_exist_m[num_neighbors_cube];
  bool           neighbors_exist_p[num_neighbors_cube];
  p4est_locidx_t neighbors        [num_neighbors_cube];

  ngbd_->get_all_neighbors(n, neighbors, neighbors_exist);

  if (fv_scheme_ == 1)
  {
    for (int idx = 0; idx < num_neighbors_cube; ++idx)
      if (neighbors_exist[idx])
      {
        neighbors_exist_p[idx] = neighbors_exist[idx] && (areas_p_ptr[neighbors[idx]] > interface_rel_thresh_);
        neighbors_exist_m[idx] = neighbors_exist[idx] && (areas_m_ptr[neighbors[idx]] > interface_rel_thresh_);
      }
  }
  else if (fv_scheme_ == 0)
  {
    for (int idx = 0; idx < num_neighbors_cube; ++idx)
      if (neighbors_exist[idx])
      {
        neighbors_exist_p[idx] = neighbors_exist[idx];
        neighbors_exist_m[idx] = neighbors_exist[idx];
      }
  }
  else throw;

  if (face_m_area_max < interface_rel_thresh_) { face_m_area_max = 0; volume_cut_cell_m = 0; }
  if (face_p_area_max < interface_rel_thresh_) { face_p_area_max = 0; volume_cut_cell_p = 0; }

  if (face_m_area_max < interface_rel_thresh_ &&
      face_p_area_max < interface_rel_thresh_)
    throw;

  //---------------------------------------------------------------------
  // compute submat_diag
  //---------------------------------------------------------------------
  if (there_is_diag_ && new_submat_diag_)
  {
    if (var_diag_) submat_diag_ptr[n] = diag_m_ptr[n] *volume_cut_cell_m + diag_p_ptr[n] *volume_cut_cell_p;
    else           submat_diag_ptr[n] = diag_m_scalar_*volume_cut_cell_m + diag_p_scalar_*volume_cut_cell_p;

    if (infc_.phi_eff_ptr[n] < 0)
    {
      if (var_diag_) submat_diag_ghost_ptr[n] = diag_p_ptr[n] *volume_cut_cell_p;
      else           submat_diag_ghost_ptr[n] = diag_p_scalar_*volume_cut_cell_p;
    }
    else
    {
      if (var_diag_) submat_diag_ghost_ptr[n] = diag_m_ptr[n] *volume_cut_cell_m;
      else           submat_diag_ghost_ptr[n] = diag_m_scalar_*volume_cut_cell_m;
    }

    if (fabs(submat_diag_ptr[n]) > EPS) matrix_has_nullspace_ = false;
  }

  //---------------------------------------------------------------------
  // compute submat_main
  //---------------------------------------------------------------------
  if (new_submat_main_)
  {
    std::vector<double> w_m(num_neighbors_cube, 0);
    std::vector<double> w_p(num_neighbors_cube, 0);

    bool   neighbors_exist_face[num_neighbors_face];
    bool   map_face            [num_neighbors_face];
    double weights_face        [num_neighbors_face];

    double theta = EPS;

    for (unsigned short dom = 0; dom < 2; ++dom) // negative and positive domains
    {
      if (dom == 0 && face_m_area_max <= interface_rel_thresh_ ||
          dom == 1 && face_p_area_max <= interface_rel_thresh_)
        continue;

      double *w               = (dom == 0 ? w_m.data()        : w_p.data()        );
      double  mu              = (dom == 0 ? mu_m_             : mu_p_             );
      CF_DIM *mu_interp       = (dom == 0 ? &mu_m_interp_     : &mu_p_interp_     );
      bool   *neighbors_exist_pm = (dom == 0 ? neighbors_exist_m     : neighbors_exist_p     );

      my_p4est_finite_volume_t &fv = (dom == 0 ? fv_m : fv_p);

      for (unsigned short dim = 0; dim < P4EST_DIM; ++dim) // loop over all dimensions
        for (unsigned short sign = 0; sign < 2; ++sign) // negative and positive directions
        {
          unsigned short dir = 2*dim + sign;

          if (fv.face_area[dir]/face_area_scalling_ > interface_rel_thresh_)
          {
            if (!neighbors_exist[f2c_p[dir][nnf_00]])
            {
              std::cout << "Warning: neighbor doesn't exist in the zp-direction."
                        << " Own number: " << n
                        << " Nei number: " << neighbors[f2c_p[dir][nnf_00]]
                           //                          << " Own area: " << areas_ptr[neighbors[f2c_m[dir][nnf_00]]]
                           //                          << " Nei area: " << areas_ptr[neighbors[f2c_p[dir][nnf_00]]]
                        << " Face Area:" << fv.face_area[dir]/face_area_scalling_ << "\n";
            }
            else
            {
              double flux   = fv.face_area[dir] / dxyz_m_[dim] *
                  (var_mu_ ? (*mu_interp)(DIM(x_C + fv.face_centroid_x[dir],
                                              y_C + fv.face_centroid_y[dir],
                                              z_C + fv.face_centroid_z[dir])) : mu);
//              if (!use_sc_scheme_)
//              {
                w[f2c_m[dir][nnf_00]] += flux;
                w[f2c_p[dir][nnf_00]] -= flux;
//              }
//              else
//              {
//                for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
//                  neighbors_exist_face[nn] = neighbors_exist_pm[f2c_m[dir][nn]] && neighbors_exist_pm[f2c_p[dir][nn]];

//                double centroid_xyz[] = { DIM( fv.face_centroid_x[dir]/dx_min_,
//                                          fv.face_centroid_y[dir]/dy_min_,
//                                          fv.face_centroid_z[dir]/dz_min_ ) };

//                CODE2D( double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]],                           neighbors_exist_face, weights_face, theta, map_face) );
//                CODE3D( double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], centroid_xyz[k_idx[dim]], neighbors_exist_face, weights_face, theta, map_face) );

//                for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
//                  if (map_face[nn])
//                  {
//                    w[f2c_m[dir][nn]] += weights_face[nn] * flux;
//                    w[f2c_p[dir][nn]] -= weights_face[nn] * flux;
//                  }
//              }

            }
          }
        }
    }

    // put values into matrices
    for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
    {
      if (neighbors_exist[i])
      {
        if (infc_.phi_eff_ptr[neighbors[i]] < 0)
        {
          double w_sum = ((neighbors_exist_m[i] ? w_m[i] : 0) +
                          (neighbors_exist_p[i] ? w_p[i] : 0));

          if (fabs(w_sum) > EPS)
          {
            row_main->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_sum));
            (neighbors[i] < nodes_->num_owned_indeps) ? d_nnz_main++ : o_nnz_main++;
          }
          if (neighbors_exist_p[i] && fabs(w_p[i]) > EPS)
          {
            row_jump->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_p[i]));
            (neighbors[i] < nodes_->num_owned_indeps) ? d_nnz_jump++ : o_nnz_jump++;
          }
        }
        else
        {
          double w_sum = ((neighbors_exist_m[i] ? w_m[i] : 0) +
                          (neighbors_exist_p[i] ? w_p[i] : 0));

          if (neighbors_exist_m[i] && fabs(w_m[i]) > EPS)
          {
            row_jump->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_m[i]));
            (neighbors[i] < nodes_->num_owned_indeps) ? d_nnz_jump++ : o_nnz_jump++;
          }
          if (fabs(w_sum) > EPS)
          {
            row_main->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_sum));
            (neighbors[i] < nodes_->num_owned_indeps) ? d_nnz_main++ : o_nnz_main++;
          }
        }
      }
    }

//    if (there_is_diag_)
//    {
//      double diag;
//      if (infc_.phi_eff_ptr[n] < 0)
//      {
//        if (var_diag_) diag = diag_p_ptr[n] *volume_cut_cell_p;
//        else           diag = diag_p_scalar_*volume_cut_cell_p;
//      }
//      else
//      {
//        if (var_diag_) diag = diag_m_ptr[n] *volume_cut_cell_m;
//        else           diag = diag_m_scalar_*volume_cut_cell_m;
//      }

//      row_jump->push_back(mat_entry_t(petsc_gloidx_[n], diag));
//    }
  }

  //---------------------------------------------------------------------
  // get information about interfaces
  //---------------------------------------------------------------------
  std::vector<int>               infc_id;
  std::vector<double>            infc_area;
  std::vector<interface_point_t> infc_xyz;
  std::vector<interface_point_t> infc_jump_xyz;

  if (new_submat_main_)
  {
    for (int i=0; i<fv_m.interfaces.size(); ++i)
    {
      interface_point_t pt(DIM(xyz_C[0] + fv_m.interfaces[i].centroid[0],
                               xyz_C[1] + fv_m.interfaces[i].centroid[1],
                               xyz_C[2] + fv_m.interfaces[i].centroid[2]));

      infc_id  .push_back(fv_m.interfaces[i].id  );
      infc_area.push_back(fv_m.interfaces[i].area);
      infc_xyz .push_back(pt);
    }

    // project interface centroids onto interfaces
    for (int i=0; i<infc_id.size(); ++i)
    {
      int phi_idx = infc_id[i];
      interface_point_t *pt = &infc_xyz[i];

      XCODE( pt->xyz[0] = MIN(pt->xyz[0], interp_max[0]); pt->xyz[0] = MAX(pt->xyz[0], interp_min[0]) );
      YCODE( pt->xyz[1] = MIN(pt->xyz[1], interp_max[1]); pt->xyz[1] = MAX(pt->xyz[1], interp_min[1]) );
      ZCODE( pt->xyz[2] = MIN(pt->xyz[2], interp_max[2]); pt->xyz[2] = MAX(pt->xyz[2], interp_min[2]) );

      // compute signed distance and normal at the centroid
      XCODE( double nx = qnnn.dx_central(infc_.phi_ptr[phi_idx]) );
      YCODE( double ny = qnnn.dy_central(infc_.phi_ptr[phi_idx]) );
      ZCODE( double nz = qnnn.dz_central(infc_.phi_ptr[phi_idx]) );

      double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
      double dist = (*infc_phi_cf_[phi_idx]).value(pt->xyz)/norm;

      XCODE( pt->xyz[0] -= dist*nx/norm );
      YCODE( pt->xyz[1] -= dist*ny/norm );
      ZCODE( pt->xyz[2] -= dist*nz/norm );

      XCODE( pt->xyz[0] = MIN(pt->xyz[0], interp_max[0]); pt->xyz[0] = MAX(pt->xyz[0], interp_min[0]) );
      YCODE( pt->xyz[1] = MIN(pt->xyz[1], interp_max[1]); pt->xyz[1] = MAX(pt->xyz[1], interp_min[1]) );
      ZCODE( pt->xyz[2] = MIN(pt->xyz[2], interp_max[2]); pt->xyz[2] = MAX(pt->xyz[2], interp_min[2]) );
    }

    infc_jump_xyz = infc_xyz;
  }
  else
  {
    load_infc_data(n, infc_id, infc_area, infc_xyz, infc_jump_xyz);
  }

  //---------------------------------------------------------------------
  // compute rhs from force term and surf generation term
  //---------------------------------------------------------------------
  if (setup_rhs)
  {
    // forcing term
    rhs_ptr[n] = (rhs_m_ptr[n]*volume_cut_cell_m +
                  rhs_p_ptr[n]*volume_cut_cell_p);

    // neumann flux through domain boundary
    for (int i=0; i<infc_area.size(); ++i)
    {
      interface_conditions_t *jc = &jc_[infc_id[i]];

      rhs_ptr[n] -= infc_area[i] * (jc->pointwise ? jc->get_flx_jump_pw_integr(n) :
                                                    jc->get_flx_jump_cf(infc_xyz[i].xyz));
    }
  }

  //---------------------------------------------------------------------
  // express values at ghost cells using values at real cells
  // i.e. compute submat_jump_ghost and rhs_jump
  //---------------------------------------------------------------------
  bool express_ghost = ( face_p_area_max > interface_rel_thresh_ &&
                         face_m_area_max > interface_rel_thresh_ );
  if (express_ghost)
  {
    double dist;
    double xyz_pr [P4EST_DIM];
    double dxyz_pr[P4EST_DIM];
    double normal [P4EST_DIM];

    double sign    = (infc_.phi_eff_ptr[n] < 0 ? 1 : -1);
    double scaling = 1;

    // compute projection point
    find_projection(qnnn, infc_.phi_ptr[0], dxyz_pr, dist, normal);

    foreach_dimension(i_dim)
      xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

    XCODE( xyz_pr[0] = MIN(xyz_pr[0], interp_max[0]); xyz_pr[0] = MAX(xyz_pr[0], interp_min[0]) );
    YCODE( xyz_pr[1] = MIN(xyz_pr[1], interp_max[1]); xyz_pr[1] = MAX(xyz_pr[1], interp_min[1]) );
    ZCODE( xyz_pr[2] = MIN(xyz_pr[2], interp_max[2]); xyz_pr[2] = MAX(xyz_pr[2], interp_min[2]) );

    infc_jump_xyz[0].set(xyz_pr);

    // sample values at the projection point
    double mu_m_proj  = var_mu_ ? mu_m_interp_.value(xyz_pr) : mu_m_;
    double mu_p_proj  = var_mu_ ? mu_p_interp_.value(xyz_pr) : mu_p_;

    // count numbers of neighbors in negative and positive domains
    unsigned short num_neg = 0;
    unsigned short num_pos = 0;
    for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei) {
      if (neighbors_exist[nei] && nei != nn_000) {
        infc_.phi_eff_ptr[neighbors[nei]] < 0 ? num_neg++ : num_pos++;
      }
    }

    // determine which side to use based on values of diffusion coefficients and neighbors' availability
    double sign_to_use;

    switch (jump_scheme_) {
      case 0: sign_to_use = (mu_m_proj < mu_p_proj) ? -1 : 1; break;
      case 1: sign_to_use = (mu_m_proj > mu_p_proj) ? -1 : 1; break;
      case 2: sign_to_use = (num_neg > num_pos)     ? -1 : 1; break;
      default: throw;
    }

    // check if there are enough nodes in the selected region for a least-squares fit
    // (we will do that by checking whether points form a full basis)
    double basis[P4EST_DIM][P4EST_DIM];
    int    num_basis_vectors_found = 0;

    for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei) {
      if (neighbors_exist[nei] && nei != nn_000) {
        if (infc_.phi_eff_ptr[neighbors[nei]]*sign_to_use > 0) {
          // get vector for a given node
          double current[P4EST_DIM];
          cube_nei_dir(nei, current);

          // try to decompose into already found basis vector
          for (int i = 0; i < num_basis_vectors_found; ++i) {
            double projection = SUMD(basis[i][0]*current[0],
                                     basis[i][1]*current[1],
                                     basis[i][2]*current[2]);

            EXECD(current[0] -= projection*basis[i][0],
                  current[1] -= projection*basis[i][1],
                  current[2] -= projection*basis[i][2]);
          }

          double norm = ABSD(current[0], current[1], current[2]);

          if (norm > 1.e-5) {
            EXECD(basis[num_basis_vectors_found][0] = current[0]/norm,
                  basis[num_basis_vectors_found][1] = current[1]/norm,
                  basis[num_basis_vectors_found][2] = current[2]/norm);

            num_basis_vectors_found++;

            if (num_basis_vectors_found == P4EST_DIM) break;
          }
        }
      }
    }

    if (num_basis_vectors_found < P4EST_DIM) sign_to_use *= -1;

    if (there_is_jump_mu_ && new_submat_main_)
    {
      std::vector<double> w_ghosts(num_neighbors_cube, 0);
      // linear system
      char num_constraints = num_neighbors_cube_;

      std::vector<double> weight(num_constraints, 0);

      //        _CODE( std::vector<double> col_c(num_constraints, 0) );
      XCODE( std::vector<double> col_x(num_constraints, 0) );
      YCODE( std::vector<double> col_y(num_constraints, 0) );
      ZCODE( std::vector<double> col_z(num_constraints, 0) );

#ifdef P4_TO_P8
      for (char k = 0; k < 3; ++k)
#endif
        for (char j = 0; j < 3; ++j)
          for (char i = 0; i < 3; ++i)
          {
            char idx = i + 3*j CODE3D( + 9*k );

            XCODE( col_x[idx] = ((double) (i-1)) * dxyz_m_[0] );
            YCODE( col_y[idx] = ((double) (j-1)) * dxyz_m_[1] );
            ZCODE( col_z[idx] = ((double) (k-1)) * dxyz_m_[2] );

            if (neighbors_exist[idx])
              if (((infc_.phi_eff_ptr[neighbors[idx]] <  0 && sign_to_use < 0) ||
                   (infc_.phi_eff_ptr[neighbors[idx]] >= 0 && sign_to_use > 0)) && idx != nn_000)
                weight[idx] = 1;
          }


      // assemble and invert matrix
      unsigned short A_size = (P4EST_DIM);
      double A[(P4EST_DIM)*(P4EST_DIM)];
      double A_inv[(P4EST_DIM)*(P4EST_DIM)];

      A[0*A_size + 0] = 0;
      A[0*A_size + 1] = 0;
      A[1*A_size + 1] = 0;
#ifdef P4_TO_P8
      A[0*A_size + 2] = 0;
      A[1*A_size + 2] = 0;
      A[2*A_size + 2] = 0;
#endif

      for (int nei=0; nei<num_constraints; ++nei)
      {
        A[0*A_size + 0] += col_x[nei]*col_x[nei]*weight[nei];
        A[0*A_size + 1] += col_x[nei]*col_y[nei]*weight[nei];
        A[1*A_size + 1] += col_y[nei]*col_y[nei]*weight[nei];
#ifdef P4_TO_P8
        A[0*A_size + 2] += col_x[nei]*col_z[nei]*weight[nei];
        A[1*A_size + 2] += col_y[nei]*col_z[nei]*weight[nei];
        A[2*A_size + 2] += col_z[nei]*col_z[nei]*weight[nei];
#endif
      }

      A[1*A_size + 0] = A[0*A_size + 1];
#ifdef P4_TO_P8
      A[2*A_size + 0] = A[0*A_size + 2];
      A[2*A_size + 1] = A[1*A_size + 2];
#endif

      if ( !CODEDIM(inv_mat2(A, A_inv), inv_mat3(A, A_inv)) ) {
          throw std::domain_error("Error: singular LSQR matrix\n");
      }

      // compute Taylor expansion coefficients
      XCODE( std::vector<double> coeff_x_term(num_constraints, 0) );
      YCODE( std::vector<double> coeff_y_term(num_constraints, 0) );
      ZCODE( std::vector<double> coeff_z_term(num_constraints, 0) );

      for (unsigned short nei = 0; nei < num_constraints; ++nei)
        if (nei != nn_000)
        {
          XCODE( coeff_x_term[nei] = weight[nei] * SUMD(A_inv[0*A_size+0]*col_x[nei], A_inv[0*A_size+1]*col_y[nei], A_inv[0*A_size+2]*col_z[nei]) );
          YCODE( coeff_y_term[nei] = weight[nei] * SUMD(A_inv[1*A_size+0]*col_x[nei], A_inv[1*A_size+1]*col_y[nei], A_inv[1*A_size+2]*col_z[nei]) );
          ZCODE( coeff_z_term[nei] = weight[nei] * SUMD(A_inv[2*A_size+0]*col_x[nei], A_inv[2*A_size+1]*col_y[nei], A_inv[2*A_size+2]*col_z[nei]) );
          XCODE( coeff_x_term[nn_000] -= coeff_x_term[nei]);
          YCODE( coeff_y_term[nn_000] -= coeff_y_term[nei]);
          ZCODE( coeff_z_term[nn_000] -= coeff_z_term[nei]);
        }

      w_ghosts[nn_000] = 0;

      if (sign_to_use < 0)
      {
        for (int nei=0; nei<num_neighbors_cube; ++nei)
        {
          if (nei != nn_000)
            w_ghosts[nei] += sign*dist*(mu_m_proj/mu_p_proj - 1.)
                *SUMD(coeff_x_term[nei]*normal[0],
                      coeff_y_term[nei]*normal[1],
                      coeff_z_term[nei]*normal[2]);
        }

        double coeff_000 = sign*dist*(mu_m_proj/mu_p_proj - 1.)
            *SUMD(coeff_x_term[nn_000]*normal[0],
                  coeff_y_term[nn_000]*normal[1],
                  coeff_z_term[nn_000]*normal[2]);

        w_ghosts[nn_000] += coeff_000;

        if (infc_.phi_eff_ptr[n] >= 0) scaling = 1. - coeff_000;
      }
      else
      {
        for (int nei=0; nei<num_neighbors_cube; ++nei)
        {
          if (nei != nn_000)
            w_ghosts[nei] += sign*dist*(1. - mu_p_proj/mu_m_proj)
                * SUMD(coeff_x_term[nei]*normal[0],
                       coeff_y_term[nei]*normal[1],
                       coeff_z_term[nei]*normal[2]);
        }

        double coeff_000 = sign*dist*(1. - mu_p_proj/mu_m_proj)
            *SUMD(coeff_x_term[nn_000]*normal[0],
                  coeff_y_term[nn_000]*normal[1],
                  coeff_z_term[nn_000]*normal[2]);

        w_ghosts[nn_000] += coeff_000;

        if (infc_.phi_eff_ptr[n] < 0) scaling = 1. - coeff_000;
      }

      for (int nei=0; nei<num_neighbors_cube; ++nei)
        w_ghosts[nei] /= scaling;

      // put values into matrix
      for (int i=0; i<num_neighbors_cube; ++i)
      {
        if (neighbors_exist[i] && fabs(w_ghosts[i]) > EPS)
        {
          row_jump_ghost->push_back(mat_entry_t(petsc_gloidx_[neighbors[i]], w_ghosts[i]));
          (neighbors[i] < nodes_->num_owned_indeps) ? ++d_nnz_jump_ghost:
                                                      ++o_nnz_jump_ghost;
        }
      }

      jump_scaling_[n] = scaling;
    }

    if (setup_rhs)
    {
      if (there_is_jump_mu_ && !new_submat_main_) scaling = jump_scaling_[n];
      double flx_jump_proj = jc_[0].pointwise ? jc_[0].get_flx_jump_pw_taylor(n) : jc_[0].get_flx_jump_cf(xyz_pr);
      double sol_jump_proj = jc_[0].pointwise ? jc_[0].get_sol_jump_pw_taylor(n) : jc_[0].get_sol_jump_cf(xyz_pr);
      rhs_jump_ptr[n]      = sign*(sol_jump_proj + dist*flx_jump_proj/( sign_to_use < 0 ? mu_p_proj : mu_m_proj))/scaling;

//      // contribution from ghost diag term
//      if (there_is_diag_)
//      {
//        double diag;
//        if (infc_.phi_eff_ptr[n] < 0)
//        {
//          if (var_diag_) diag = diag_p_ptr[n] *volume_cut_cell_p;
//          else           diag = diag_p_scalar_*volume_cut_cell_p;
//        }
//        else
//        {
//          if (var_diag_) diag = diag_m_ptr[n] *volume_cut_cell_m;
//          else           diag = diag_m_scalar_*volume_cut_cell_m;
//        }

//        rhs_ptr[n] -= rhs_jump_ptr[n]*diag;
//      }
    }
  }

  if (new_submat_main_)
  {
    save_infc_data(n, infc_id,  infc_area, infc_xyz, infc_jump_xyz);
  }
}

void my_p4est_poisson_nodes_mls_t::interpolators_initialize()
{
  mu_m_interp_.set_input(mue_m_, DIM(mue_m_xx_, mue_m_yy_, mue_m_zz_), interp_method_);
  mu_p_interp_.set_input(mue_p_, DIM(mue_p_xx_, mue_p_yy_, mue_p_zz_), interp_method_);

  bdry_phi_interp_.assign(bdry_.num_phi, NULL);
  infc_phi_interp_.assign(infc_.num_phi, NULL);

  bdry_phi_cf_.assign(bdry_.num_phi, NULL);
  infc_phi_cf_.assign(infc_.num_phi, NULL);

  for (int i=0; i<bdry_.num_phi; ++i)
  {
    bdry_phi_interp_[i] = new my_p4est_interpolation_nodes_local_t(ngbd_);
    bdry_phi_cf_    [i] = bdry_phi_interp_[i];
    bdry_phi_interp_[i]->set_input(bdry_.phi_ptr[i], DIM(bdry_.phi_xx_ptr[i], bdry_.phi_yy_ptr[i], bdry_.phi_zz_ptr[i]), interp_method_);
  }

  for (int i=0; i<infc_.num_phi; ++i)
  {
    infc_phi_interp_[i] = new my_p4est_interpolation_nodes_local_t(ngbd_);
    infc_phi_cf_    [i] = infc_phi_interp_[i];
    infc_phi_interp_[i]->set_input(infc_.phi_ptr[i], DIM(infc_.phi_xx_ptr[i], infc_.phi_yy_ptr[i], infc_.phi_zz_ptr[i]), interp_method_);
  }
}

void my_p4est_poisson_nodes_mls_t::interpolators_prepare(p4est_locidx_t n)
{
  mu_m_interp_.initialize(n);
  mu_p_interp_.copy_init(mu_m_interp_);

  for (int i=0; i<bdry_.num_phi; ++i) bdry_phi_interp_[i]->copy_init(mu_m_interp_);
  for (int i=0; i<infc_.num_phi; ++i) infc_phi_interp_[i]->copy_init(mu_m_interp_);
}

void my_p4est_poisson_nodes_mls_t::interpolators_finalize()
{
  for (int i=0; i<bdry_phi_interp_.size(); ++i) delete bdry_phi_interp_[i];
  for (int i=0; i<infc_phi_interp_.size(); ++i) delete infc_phi_interp_[i];
}



void my_p4est_poisson_nodes_mls_t::save_bdry_data(p4est_locidx_t n, vector<int> &bdry_ids, vector<double> &bdry_areas, vector<interface_point_t> &bdry_value_pts, vector<interface_point_t> &bdry_robin_pts)
{
#ifdef CASL_THROWS
  if (bdry_ids.size() != bdry_areas    .size() ||
      bdry_ids.size() != bdry_value_pts.size()  ||
      bdry_ids.size() != bdry_robin_pts.size())
    throw std::invalid_argument("Vectors of different sizes\n");
#endif

  for (int i=0; i<bdry_ids.size(); ++i)
  {
    bc_[bdry_ids[i]].add_fv_pt(n, bdry_areas[i], bdry_value_pts[i], bdry_robin_pts[i]);
  }
}

void my_p4est_poisson_nodes_mls_t::load_bdry_data(p4est_locidx_t n, vector<int> &bdry_ids, vector<double> &bdry_areas, vector<interface_point_t> &bdry_value_pts, vector<interface_point_t> &bdry_robin_pts)
{
  bdry_ids      .clear();
  bdry_areas    .clear();
  bdry_value_pts.clear();
  bdry_robin_pts.clear();

  for (int i=0; i<bc_.size(); ++i)
  {
    boundary_conditions_t *bc = &bc_[i];
    if ((bc->type == ROBIN || bc->type == NEUMANN) && bc->is_boundary_node(n))
    {
      int idx = bc->node_map[n];
      bdry_ids      .push_back(i);
      bdry_areas    .push_back(bc->areas[idx]);
      bdry_value_pts.push_back(bc->neumann_pts[idx]);
      bdry_robin_pts.push_back(bc->robin_pts[idx]);
    }
  }
}

void my_p4est_poisson_nodes_mls_t::save_cart_points(p4est_locidx_t n, vector<bool> &is_interface, vector<int> &id, vector<double> &dist, vector<double> &weights)
{
  if (id.size() != is_interface.size() ||
      id.size() != dist.size()  ||
      id.size() != weights.size())
    throw std::invalid_argument("Vectors of different sizes\n");

  static double xyz[P4EST_DIM];

  for (int i=0; i<P4EST_FACES; ++i)
  {
    if (is_interface[i])
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      switch (i)
      {
        case 0: xyz[0] -= dist[i]; break;
        case 1: xyz[0] += dist[i]; break;

        case 2: xyz[1] -= dist[i]; break;
        case 3: xyz[1] += dist[i]; break;
    #ifdef P4_TO_P8
        case 4: xyz[2] -= dist[i]; break;
        case 5: xyz[2] += dist[i]; break;
    #endif
      }
      bc_[id[i]].add_fd_pt(n, i, dist[i], xyz, weights[i]);
    }
  }
}

void my_p4est_poisson_nodes_mls_t::load_cart_points(p4est_locidx_t n, vector<bool> &is_interface, vector<int> &id, vector<double> &dist, vector<double> &weights)
{
  is_interface.assign(P4EST_FACES, false);
  weights     .assign(P4EST_FACES, 0);

  for (int i = 0; i < bc_.size(); ++i)
  {
    boundary_conditions_t *bc = &bc_[i];

    if (bc->type == DIRICHLET && bc->is_boundary_node(n))
    {
      for (int k = 0; k < bc->num_value_pts(n); ++k)
      {
        int idx = bc->idx_value_pt(n,k);
        interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];

        is_interface[pt->dir] = true;
        id          [pt->dir] = i;
        dist        [pt->dir] = pt->dist;
        weights     [pt->dir] = bc->dirichlet_weights[idx];
      }
    }
  }
}



void my_p4est_poisson_nodes_mls_t::save_infc_data(p4est_locidx_t n, vector<int> &infc_ids, vector<double> &infc_areas, vector<interface_point_t> &infc_integr_pts, vector<interface_point_t> &infc_taylor_pts)
{
  if (infc_ids.size() != infc_areas     .size() ||
      infc_ids.size() != infc_integr_pts.size() ||
      infc_ids.size() != infc_taylor_pts.size())
    throw std::invalid_argument("Vectors of different sizes\n");

  for (int i=0; i<infc_ids.size(); ++i)
  {
    jc_[infc_ids[i]].add_pt(n, infc_areas[i], infc_taylor_pts[i], infc_integr_pts[i]);
  }
}

void my_p4est_poisson_nodes_mls_t::load_infc_data(p4est_locidx_t n, vector<int> &infc_ids, vector<double> &infc_areas, vector<interface_point_t> &infc_integr_pts, vector<interface_point_t> &infc_taylor_pts)
{
  infc_ids       .clear();
  infc_areas     .clear();
  infc_integr_pts.clear();
  infc_taylor_pts.clear();

  for (int i = 0; i < jc_.size(); ++i)
  {
    interface_conditions_t *jc = &jc_[i];

    if (jc->is_interface_node(n))
    {
      int idx = jc->node_map[n];
      infc_ids       .push_back(i);
      infc_areas     .push_back(jc->areas[idx]);
      infc_taylor_pts.push_back(jc->taylor_pts[idx]);
      infc_integr_pts.push_back(jc->integr_pts[idx]);
    }
  }
}



void my_p4est_poisson_nodes_mls_t::save_wall_data(p4est_locidx_t n, vector<int> &wall_id, vector<double> &wall_area, vector<interface_point_t> &wall_xyz)
{
  if (wall_id.size() != wall_area.size() || wall_id.size() != wall_xyz.size())
    throw std::invalid_argument("Vectors of different sizes\n");

  for (int i=0; i<wall_id.size(); ++i)
  {
    wall_pieces_map     .add_point(n);
    wall_pieces_id      .push_back(wall_id  [i]);
    wall_pieces_area    .push_back(wall_area[i]);
    wall_pieces_centroid.push_back(wall_xyz [i]);
  }
}

void my_p4est_poisson_nodes_mls_t::load_wall_data(p4est_locidx_t n, vector<int> &wall_id, vector<double> &wall_area, vector<interface_point_t> &wall_xyz)
{
  wall_id  .clear();
  wall_xyz .clear();
  wall_area.clear();

  for (int i=0; i<wall_pieces_map.size[n]; ++i)
  {
    int idx = wall_pieces_map.get_idx(n,i);
    wall_id  .push_back(wall_pieces_id      [idx]);
    wall_area.push_back(wall_pieces_area    [idx]);
    wall_xyz .push_back(wall_pieces_centroid[idx]);
  }
}



void my_p4est_poisson_nodes_mls_t::find_interface_points(p4est_locidx_t n, const my_p4est_node_neighbors_t *ngbd,
                                                         std::vector<mls_opn_t> opn,
                                                         std::vector<double *> phi_ptr, DIM( std::vector<double *> phi_xx_ptr,
                                                                                             std::vector<double *> phi_yy_ptr,
                                                                                             std::vector<double *> phi_zz_ptr ),
                                                         int phi_idx[], double dist[])
{
  const quad_neighbor_nodes_of_node_t qnnn = ngbd->get_neighbors(n);

  unsigned short size = opn.size();

  std::vector<double> phi_000(size,-1);

  std::vector<double> DIM(phi_m00(size,-1), phi_0m0(size,-1), phi_00m(size,-1));
  std::vector<double> DIM(phi_p00(size,-1), phi_0p0(size,-1), phi_00p(size,-1));

  std::vector<double> DIM(phi_xx_m00(size, 0), phi_yy_0m0(size, 0), phi_zz_00m(size, 0));
  std::vector<double> DIM(phi_xx_p00(size, 0), phi_yy_0p0(size, 0), phi_zz_00p(size, 0));

  for (unsigned short i = 0; i < phi_ptr.size(); ++i)
  {
    CODE2D( qnnn.ngbd_with_quadratic_interpolation(phi_ptr[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]) );
    CODE3D( qnnn.ngbd_with_quadratic_interpolation(phi_ptr[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]) );

    phi_xx_m00[i] = MINMOD(phi_xx_ptr[i][n], qnnn.f_m00_linear(phi_xx_ptr[i]));
    phi_xx_p00[i] = MINMOD(phi_xx_ptr[i][n], qnnn.f_p00_linear(phi_xx_ptr[i]));

    phi_yy_0m0[i] = MINMOD(phi_yy_ptr[i][n], qnnn.f_0m0_linear(phi_yy_ptr[i]));
    phi_yy_0p0[i] = MINMOD(phi_yy_ptr[i][n], qnnn.f_0p0_linear(phi_yy_ptr[i]));
#ifdef P4_TO_P8
    phi_zz_00m[i] = MINMOD(phi_zz_ptr[i][n], qnnn.f_00m_linear(phi_zz_ptr[i]));
    phi_zz_00p[i] = MINMOD(phi_zz_ptr[i][n], qnnn.f_00p_linear(phi_zz_ptr[i]));
#endif
  }

  find_closest_interface_location(phi_idx[dir::f_m00], dist[dir::f_m00], qnnn.d_m00, opn, phi_000, phi_m00, phi_xx_m00, phi_xx_m00);
  find_closest_interface_location(phi_idx[dir::f_p00], dist[dir::f_p00], qnnn.d_p00, opn, phi_000, phi_p00, phi_xx_p00, phi_xx_p00);

  find_closest_interface_location(phi_idx[dir::f_0m0], dist[dir::f_0m0], qnnn.d_0m0, opn, phi_000, phi_0m0, phi_yy_0m0, phi_yy_0m0);
  find_closest_interface_location(phi_idx[dir::f_0p0], dist[dir::f_0p0], qnnn.d_0p0, opn, phi_000, phi_0p0, phi_yy_0p0, phi_yy_0p0);
#ifdef P4_TO_P8
  find_closest_interface_location(phi_idx[dir::f_00m], dist[dir::f_00m], qnnn.d_00m, opn, phi_000, phi_00m, phi_zz_00m, phi_zz_00m);
  find_closest_interface_location(phi_idx[dir::f_00p], dist[dir::f_00p], qnnn.d_00p, opn, phi_000, phi_00p, phi_zz_00p, phi_zz_00p);
#endif
}


int my_p4est_poisson_nodes_mls_t::solve_nonlinear(Vec sol, bool use_nonzero_guess, bool update_ghost, KSPType ksp_type, PCType pc_type)
{
  if (!use_nonzero_guess)
  {
    ierr = VecSetGhost(sol, 0.0); CHKERRXX(ierr);
  }

  Vec del_sol;
  Vec sol_ghost;
  Vec residual;

  Vec rhs_m_current;
  Vec rhs_p_current;

  Vec diag_m_current;
  Vec diag_p_current;

  ierr = VecDuplicate(sol, &del_sol);    CHKERRXX(ierr);
  ierr = VecDuplicate(sol, &sol_ghost);  CHKERRXX(ierr);
  ierr = VecDuplicate(sol, &residual); CHKERRXX(ierr);

  ierr = VecDuplicate(sol, &rhs_m_current); CHKERRXX(ierr);
  ierr = VecDuplicate(sol, &rhs_p_current); CHKERRXX(ierr);

  ierr = VecDuplicate(sol, &diag_m_current); CHKERRXX(ierr);
  ierr = VecDuplicate(sol, &diag_p_current); CHKERRXX(ierr);

  ierr = VecSetGhost(del_sol, 0.0); CHKERRXX(ierr);
  ierr = VecSetGhost(sol_ghost, 0.0); CHKERRXX(ierr);
  ierr = VecSetGhost(residual, 0.0); CHKERRXX(ierr);

  // just in case
  ierr = VecSetGhost(rhs_m_current, 0.0); CHKERRXX(ierr);
  ierr = VecSetGhost(rhs_p_current, 0.0); CHKERRXX(ierr);

  ierr = VecSetGhost(diag_m_current, 0.0); CHKERRXX(ierr);
  ierr = VecSetGhost(diag_p_current, 0.0); CHKERRXX(ierr);

  // get original equation parameters
  Vec diag_m_original = diag_m_;
  Vec diag_p_original = diag_p_;

  Vec rhs_m_original = rhs_m_;
  Vec rhs_p_original = rhs_p_;

  // auxiliary stuff
  double diag_m_original_value = diag_m_scalar_;
  double diag_p_original_value = diag_p_scalar_;

  double nonlinear_term_m_coeff_value = nonlinear_term_m_coeff_scalar_;
  double nonlinear_term_p_coeff_value = nonlinear_term_p_coeff_scalar_;

  // iterations
  int    iter = 0;
  double change_norm = DBL_MAX;
  double pde_residual_norm = DBL_MAX;

  setup_linear_system(true);

  while (iter < nonlinear_itmax_ && change_norm > nonlinear_change_tol_ && pde_residual_norm > nonlinear_pde_residual_tol_)
  {
    // compute ghost values
    if (there_is_jump_ && iter > 0)
    {
      if (there_is_jump_mu_)
      {
        ierr = MatMultAdd(submat_jump_ghost_, sol, sol, sol_ghost); CHKERRXX(ierr);
      }
      else
      {
        ierr = VecCopyGhost(sol, sol_ghost); CHKERRXX(ierr);
      }

      ierr = VecAXPY(sol_ghost, -1.0, rhs_jump_); CHKERRXX(ierr);

//      double norm_m;
//      double norm_p;
//      ierr = VecMax(sol_ghost, NULL, &norm_m); CHKERRXX(ierr);
//      ierr = VecMin(sol_ghost, NULL, &norm_p); CHKERRXX(ierr);

//      std::cout << norm_m << " " << norm_p << "\n";
    }

    // compute current diag and rhs.
    ierr = VecGetArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);

    if (var_nonlinear_term_coeff_)
    {
      ierr = VecGetArray(nonlinear_term_m_coeff_, &nonlinear_term_m_coeff_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(nonlinear_term_p_coeff_, &nonlinear_term_p_coeff_ptr); CHKERRXX(ierr);
    }

    double *diag_m_original_ptr;
    double *diag_p_original_ptr;

    if (var_diag_)
    {
      ierr = VecGetArray(diag_m_original, &diag_m_original_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(diag_p_original, &diag_p_original_ptr); CHKERRXX(ierr);
    }

    double *diag_m_current_ptr;
    double *diag_p_current_ptr;

    ierr = VecGetArray(diag_m_current, &diag_m_current_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(diag_p_current, &diag_p_current_ptr); CHKERRXX(ierr);

    double *rhs_m_original_ptr;
    double *rhs_p_original_ptr;

    ierr = VecGetArray(rhs_m_original, &rhs_m_original_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_original, &rhs_p_original_ptr); CHKERRXX(ierr);

    double *rhs_m_current_ptr;
    double *rhs_p_current_ptr;

    ierr = VecGetArray(rhs_m_current, &rhs_m_current_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_current, &rhs_p_current_ptr); CHKERRXX(ierr);

    double *sol_ptr;
    double *sol_ghost_ptr;

    ierr = VecGetArray(sol, &sol_ptr);CHKERRXX(ierr);
    ierr = VecGetArray(sol_ghost, &sol_ghost_ptr);CHKERRXX(ierr);

    foreach_local_node(n, nodes_)
    {
      if (mask_m_ptr[n] < 0 || mask_p_ptr[n] < 0)
      {
        if (mask_m_ptr[n] < 0 && mask_p_ptr[n] < 0) throw;
        double sol_m = mask_m_ptr[n] < 0 ? sol_ptr[n] : sol_ghost_ptr[n];
        double sol_p = mask_p_ptr[n] < 0 ? sol_ptr[n] : sol_ghost_ptr[n];

        if (var_diag_)
        {
          diag_m_original_value = diag_m_original_ptr[n];
          diag_p_original_value = diag_p_original_ptr[n];
        }

        if (var_nonlinear_term_coeff_)
        {
          nonlinear_term_m_coeff_value = nonlinear_term_m_coeff_ptr[n];
          nonlinear_term_p_coeff_value = nonlinear_term_p_coeff_ptr[n];
        }

        if (mask_m_ptr[n] < 0 || node_scheme_[n] == IMMERSED_INTERFACE)
        {
          diag_m_current_ptr[n] = diag_m_original_value + nonlinear_term_m_coeff_value*(*nonlinear_term_m_prime_)(sol_m);
          rhs_m_current_ptr [n] = rhs_m_original_ptr[n] - nonlinear_term_m_coeff_value*((*nonlinear_term_m_)(sol_m) - (*nonlinear_term_m_prime_)(sol_m)*sol_m);
        }

        if (mask_p_ptr[n] < 0 || node_scheme_[n] == IMMERSED_INTERFACE)
        {
          diag_p_current_ptr[n] = diag_p_original_value + nonlinear_term_p_coeff_value*(*nonlinear_term_p_prime_)(sol_p);
          rhs_p_current_ptr [n] = rhs_p_original_ptr[n] - nonlinear_term_p_coeff_value*((*nonlinear_term_p_)(sol_p) - (*nonlinear_term_p_prime_)(sol_p)*sol_p);
        }
      }
    }

    ierr = VecRestoreArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);

    if (var_nonlinear_term_coeff_)
    {
      ierr = VecRestoreArray(nonlinear_term_m_coeff_, &nonlinear_term_m_coeff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(nonlinear_term_p_coeff_, &nonlinear_term_p_coeff_ptr); CHKERRXX(ierr);
    }

    if (var_diag_)
    {
      ierr = VecRestoreArray(diag_m_original, &diag_m_original_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(diag_p_original, &diag_p_original_ptr); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(diag_m_current, &diag_m_current_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(diag_p_current, &diag_p_current_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(rhs_m_original, &rhs_m_original_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_original, &rhs_p_original_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(rhs_m_current, &rhs_m_current_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_current, &rhs_p_current_ptr); CHKERRXX(ierr);

    ierr = VecRestoreArray(sol, &sol_ptr);CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_ghost, &sol_ghost_ptr);CHKERRXX(ierr);

    set_diag(diag_m_current, diag_p_current);
    set_rhs(rhs_m_current, rhs_p_current);

//    double norm_m;
//    double norm_p;
//    ierr = VecNorm(diag_m_current, NORM_2, &norm_m); CHKERRXX(ierr);
//    ierr = VecNorm(diag_p_current, NORM_2, &norm_p); CHKERRXX(ierr);

//    std::cout << norm_m << " " << norm_p << "\n";

    // assemble current linear system
    setup_linear_system(true);

    // compute residual of linear system
    ierr = MatMult(A_, sol, residual);
    ierr = VecAYPX(residual, -1.0, rhs_);

    double *residual_ptr;

    ierr = VecGetArray(residual, &residual_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mask_m_,  &mask_m_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(mask_p_,  &mask_p_ptr);   CHKERRXX(ierr);

    foreach_local_node(n, nodes_)
    {
      if (mask_m_ptr[n] > 0 && mask_p_ptr[n] > 0) residual_ptr[n] = 0;
    }

    ierr = VecRestoreArray(residual, &residual_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_m_,  &mask_m_ptr);   CHKERRXX(ierr);
    ierr = VecRestoreArray(mask_p_,  &mask_p_ptr);   CHKERRXX(ierr);

    // solve the linear system
    switch (nonlinear_method_)
    {
      case 0:
        VecCopyGhost(sol, del_sol);
        invert_linear_system(sol, true, false, ksp_type, pc_type);
        ierr = VecAXPY(del_sol, -1., sol);CHKERRXX(ierr);
        break;
      case 1:
      {
        Vec tmp = rhs_; rhs_ = residual;
        invert_linear_system(del_sol, false, false, ksp_type, pc_type);
        rhs_ = tmp;
        ierr = VecAXPY(sol,  1., del_sol);CHKERRXX(ierr);
        break;
      }
      default:
        throw;
    }

    // compute norms of change and residual
    ierr = VecNorm(residual, NORM_2, &pde_residual_norm); CHKERRXX(ierr);
    ierr = VecNorm(del_sol,  NORM_2, &change_norm);       CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_->mpicomm, "Iteration no. %d, norm of change: %1.2e, norm of pde residual: %1.2e\n", iter, change_norm, pde_residual_norm); CHKERRXX(ierr);

    iter++;
  }

  // clean up
  ierr = VecDestroy(del_sol);   CHKERRXX(ierr);
  ierr = VecDestroy(sol_ghost); CHKERRXX(ierr);
  ierr = VecDestroy(residual);  CHKERRXX(ierr);

  ierr = VecDestroy(rhs_m_current); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_p_current); CHKERRXX(ierr);

  ierr = VecDestroy(diag_m_current); CHKERRXX(ierr);
  ierr = VecDestroy(diag_p_current); CHKERRXX(ierr);

  diag_m_ = diag_m_original;
  diag_p_ = diag_p_original;

  rhs_m_ = rhs_m_original;
  rhs_p_ = rhs_p_original;

  // update ghosts
  if (update_ghost)
  {
    ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  return iter;
}
