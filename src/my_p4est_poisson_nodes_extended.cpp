#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_extended.h"
#include <src/my_p8est_refine_coarsen.h>
#else
#include "my_p4est_poisson_nodes_extended.h"
#include <src/my_p4est_refine_coarsen.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_solve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_sc_compute_volumes;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0


my_p4est_poisson_nodes_extended_t::my_p4est_poisson_nodes_extended_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    // p4est
    p4est_(node_neighbors->p4est), nodes_(node_neighbors->nodes), ghost_(node_neighbors->ghost), myb_(node_neighbors->myb),
    // linear system
    extrapolation_order_(1),
    A_(NULL),
    new_pc_(true),
    is_matrix_computed_(false), matrix_has_nullspace_(false),
    // geometry
    phi_(NULL), phi_xx_(NULL), phi_yy_(NULL), phi_zz_(NULL), is_phi_dd_owned_(false),
    // equation
    rhs_(NULL),
    diag_add_scalar_(0.), diag_add_(NULL),
    mu_(1.), mue_(NULL), mue_xx_(NULL), mue_yy_(NULL), mue_zz_(NULL), is_mue_dd_owned_(false), variable_mu_(false),
    // bc
    bc_(NULL),
    use_pointwise_dirichlet_(false),
    //other
    keep_scalling_(false),
{
  // compute global numbering of nodes
  global_node_offset_.resize(p4est_->mpisize+1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est_->mpisize; ++r)
    global_node_offset_[r+1] = global_node_offset_[r] + (PetscInt)nodes_->global_owned_indeps[r];

  // set up the KSP solver
  ierr = KSPCreate(p4est_->mpicomm, &ksp_); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp_, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size. Must be generalized if different trees have
  // different sizes
  p4est_topidx_t vm = p4est_->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est_->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est_->connectivity->vertices[3*vm + 0];
  double ymin = p4est_->connectivity->vertices[3*vm + 1];
  double xmax = p4est_->connectivity->vertices[3*vp + 0];
  double ymax = p4est_->connectivity->vertices[3*vp + 1];
  dx_min_ = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  dy_min_ = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est_->connectivity->vertices[3*vm + 2];
  double zmax = p4est_->connectivity->vertices[3*vp + 2];
  dz_min_ = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif
#ifdef P4_TO_P8
  d_min_ = MIN(dx_min_, dy_min_, dz_min_);
  diag_min_ = sqrt(dx_min_*dx_min_ + dy_min_*dy_min_ + dz_min_*dz_min_);
#else
  d_min_ = MIN(dx_min_, dy_min_);
  diag_min_ = sqrt(dx_min_*dx_min_ + dy_min_*dy_min_);
#endif

  dxyz_m_[0] = dx_min_;
  dxyz_m_[1] = dy_min_;
#ifdef P4_TO_P8
  dxyz_m_[2] = dz_min_;
#endif

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

  // for all-neumann case
  fixed_value_idx_g_ = global_node_offset_[p4est_->mpisize];

  scalling_.resize(nodes_->num_owned_indeps, 1);
//  pointwise_bc.resize(nodes->num_owned_indeps);
}

my_p4est_poisson_nodes_extended_t::~my_p4est_poisson_nodes_extended_t()
{
  if (A_             != NULL) {ierr = MatDestroy(A_);                      CHKERRXX(ierr);}
  if (ksp_           != NULL) {ierr = KSPDestroy(ksp_);                    CHKERRXX(ierr);}
  if (is_phi_dd_owned_)
  {
    if (phi_xx_ != NULL) { for (int i = 0; i < phi_xx_->size(); i++) {ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr);} delete phi_xx_; }
    if (phi_yy_ != NULL) { for (int i = 0; i < phi_yy_->size(); i++) {ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr);} delete phi_yy_; }
#ifdef P4_TO_P8
    if (phi_zz_ != NULL) { for (int i = 0; i < phi_zz_->size(); i++) {ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr);} delete phi_zz_; }
#endif
  }

  if (is_mue_dd_owned_)
  {
    if (mue_xx_     != NULL) {ierr = VecDestroy(mue_xx_);                CHKERRXX(ierr);}
    if (mue_yy_     != NULL) {ierr = VecDestroy(mue_yy_);                CHKERRXX(ierr);}
#ifdef P4_TO_P8
    if (mue_zz_     != NULL) {ierr = VecDestroy(mue_zz_);                CHKERRXX(ierr);}
#endif
  }
}

void my_p4est_poisson_nodes_extended_t::compute_phi_dd_()
{
  if (phi_xx_ != NULL && is_phi_dd_owned_) { ierr = VecDestroy(phi_xx_); CHKERRXX(ierr); }
  if (phi_yy_ != NULL && is_phi_dd_owned_) { ierr = VecDestroy(phi_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if (phi_zz_ != NULL && is_phi_dd_owned_) { ierr = VecDestroy(phi_zz_); CHKERRXX(ierr); }
#endif

  // Allocate memory for second derivaties
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_xx_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_, phi_zz_);
#else
  node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_);
#endif

  is_phi_dd_owned_ = true;
}

void my_p4est_poisson_nodes_extended_t::compute_mue_dd_()
{
  if (mue_xx_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_xx_); CHKERRXX(ierr); }
  if (mue_yy_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if (mue_zz_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_zz_); CHKERRXX(ierr); }
#endif

  // Allocate memory for second derivaties
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_xx_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  node_neighbors_->second_derivatives_central(mue_, mue_xx_, mue_yy_, mue_zz_);
#else
  node_neighbors_->second_derivatives_central(mue_, mue_xx_, mue_yy_);
#endif

  is_mue_dd_owned_ = true;
}

void my_p4est_poisson_nodes_extended_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_solve, A_, rhs_, ksp_, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if (bc_wall_type_ == NULL || bc_wall_value_ == NULL)
    throw std::domain_error("[CASL_ERROR]: the boundary conditions on walls have not been set.");

  if (num_interfaces_ > 0)
    if (bc_interface_coeff_->size()!= num_interfaces_ ||
        bc_interface_type_->size() != num_interfaces_ ||
        bc_interface_value_->size()!= num_interfaces_)
      throw std::domain_error("[CASL_ERROR]: the boundary conditions on interfaces have not been set.");

  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
    if (sol_size != nodes_->num_owned_indeps){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes_->num_owned_indeps << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set local add if none was given
  bool local_add = false;
  if(diag_add_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &diag_add_); CHKERRXX(ierr);
    ierr = VecSet(diag_add_, diag_add_scalar_); CHKERRXX(ierr);
  }

  // set a local phi if not was given
  bool local_phi = false;
  if(num_interfaces_ == 0)
  {
    local_phi = true;

    phi_    = new std::vector<Vec> ();
    color_  = new std::vector<int> ();
    action_ = new std::vector<action_t> ();

    phi_->push_back(Vec());
    color_->push_back(0);
    action_->push_back(INTERSECTION);

    ierr = VecDuplicate(solution, &phi_->at(0)); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(phi_->at(0), &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_->at(0), &tmp); CHKERRXX(ierr);
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    set_geometry(1, action_, color_, phi_);
  }

  bool local_rhs = false;
  if (rhs_ == NULL)
  {
    ierr = VecDuplicate(solution, &rhs_); CHKERRXX(ierr);
    Vec rhs_local;
    VecGhostGetLocalForm(rhs_, &rhs_local);
    VecSet(rhs_local, 0);
    VecGhostRestoreLocalForm(rhs_, &rhs_local);
    local_rhs = true;
  }

  // set ksp type
  ierr = KSPSetType(ksp_, ksp_type); CHKERRXX(ierr);
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_computed_)
  {
    matrix_has_nullspace_ = true;
//    setup_negative_variable_coeff_laplace_matrix_();
//    setup_linear_system(true, false);
    setup_linear_system(true, true);
    is_matrix_computed_ = true;
    new_pc_ = true;
  } else {
    setup_linear_system(false, true);
  }
//  setup_linear_system(false, true);

  if (new_pc_)
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
//    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0."); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.91"); CHKERRXX(ierr);

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



  // setup rhs
//  setup_negative_variable_coeff_laplace_rhsvec_();
//  setup_linear_system(false, true);

  // Solve the system
  ierr = KSPSetTolerances(ksp_, 1.e-15, 1.e-15, PETSC_DEFAULT, 100); CHKERRXX(ierr);
//  ierr = KSPSetTolerances(ksp_, 1e-15, PETSC_DEFAULT, PETSC_DEFAULT, 30); CHKERRXX(ierr);
//  ierr = KSPSetTolerances(ksp_, 1e-200, 1e-30, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp_); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);
  MatNullSpace A_null;
  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceCreate(p4est_->mpicomm, PETSC_TRUE, 0, NULL, &A_null); CHKERRXX(ierr);
    ierr = MatSetNullSpace(A_, A_null);

    // For purely neumann problems GMRES is more robust
    ierr = KSPSetType(ksp_, KSPGMRES); CHKERRXX(ierr);
  }

  ierr = KSPSolve(ksp_, rhs_, solution); CHKERRXX(ierr);
  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceDestroy(A_null);
  }
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_mls_sc_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(diag_add_); CHKERRXX(ierr);
    diag_add_ = NULL;
  }
  if(local_rhs)
  {
    ierr = VecDestroy(rhs_); CHKERRXX(ierr);
    rhs_ = NULL;
  }
  if(local_phi)
  {
    ierr = VecDestroy(phi_->at(0)); CHKERRXX(ierr);
    delete phi_;
    phi_ = NULL;

    ierr = VecDestroy(phi_xx_->at(0)); CHKERRXX(ierr);
    delete phi_xx_;
    phi_xx_ = NULL;

    ierr = VecDestroy(phi_yy_->at(0)); CHKERRXX(ierr);
    delete phi_yy_;
    phi_yy_ = NULL;

#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_->at(0)); CHKERRXX(ierr);
    delete phi_zz_;
    phi_zz_ = NULL;
#endif

    delete action_;
    delete color_;
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_solve, A_, rhs_, ksp_, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_extended_t::setup_linear_system(bool setup_matrix, bool setup_rhs)
{
  PetscInt num_owned_global = global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

  std::vector< std::vector<mat_entry_t>* > matrix_entries(nodes_->num_owned_indeps, NULL);
  std::vector<PetscInt> d_nnz(nodes_->num_owned_indeps, 1), o_nnz(nodes_->num_owned_indeps, 0);

  if (!setup_matrix && !setup_rhs)
    throw std::invalid_argument("[CASL_ERROR]: If you aren't assembling either matrix or RHS, what the heck then are you trying to do? lol :)");

  if (setup_matrix)
  {
    if (use_pointwise_dirichlet_)
    {
      pointwise_bc_.clear();
      pointwise_bc_.resize(nodes_->num_owned_indeps);
    }
#ifdef DO_NOT_PREALLOCATE
    for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
    {
      matrix_entries[n] = new std::vector<mat_entry_t>;
    }
#else
    preallocate_matrix();
#endif
  }

  // register for logging purpose
  // not sure if we need to register both in case we're assembling matrix and rhs simultaneously
  if (setup_matrix)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
  }

  double eps = 1E-6*d_min_*d_min_;

  double *exact_ptr;
  if (exact_ != NULL) { ierr = VecGetArray(exact_, &exact_ptr); CHKERRXX(ierr); }

  //---------------------------------------------------------------------
  // get access to LSFs
  //---------------------------------------------------------------------
  std::vector<double *> phi_p (num_interfaces_, NULL);
  std::vector<double *> phi_xx_p (num_interfaces_, NULL);
  std::vector<double *> phi_yy_p (num_interfaces_, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p (num_interfaces_, NULL);
#endif

  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecGetArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *phi_eff_p;

  ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  double *mue_p=NULL;
  double *mue_xx_p=NULL;
  double *mue_yy_p=NULL;
  double *mue_zz_p=NULL;

  if (variable_mu_)
  {
    ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }

  double *diag_add_p;

  ierr = VecGetArray(diag_add_, &diag_add_p); CHKERRXX(ierr);

  double *rhs_p;
  if (setup_rhs)
  {
    ierr = VecGetArray(rhs_, &rhs_p); CHKERRXX(ierr);
  }

  std::vector<double> phi_000(num_interfaces_,-1);
  std::vector<double> phi_p00(num_interfaces_, 0);
  std::vector<double> phi_m00(num_interfaces_, 0);
  std::vector<double> phi_0m0(num_interfaces_, 0);
  std::vector<double> phi_0p0(num_interfaces_, 0);
#ifdef P4_TO_P8
  std::vector<double> phi_00m(num_interfaces_, 0);
  std::vector<double> phi_00p(num_interfaces_, 0);
#endif

  double *mask_p;
  if (setup_matrix)
  {
    if (!volumes_computed_) compute_volumes_();
    if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_->at(0), &mask_); CHKERRXX(ierr);
  }
  ierr = VecGetArray(mask_, &mask_p); CHKERRXX(ierr);

  if (setup_matrix)
    for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
      mask_p[n] = -1;

  double *volumes_p;
  double *node_type_p;

//  if (use_sc_scheme_)
//  {
    ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);
    ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);
//  }

  double mue_000 = mu_;
  double mue_p00 = mu_;
  double mue_m00 = mu_;
  double mue_0m0 = mu_;
  double mue_0p0 = mu_;
#ifdef P4_TO_P8
  double mue_00m = mu_;
  double mue_00p = mu_;
#endif

  /* Daniil: while working on solidification of alloys I came to realize
   * that due to very irregular structure of solidification fronts (where
   * there constantly exist nodes, which belong to one phase but the vertices
   * of their dual cells belong to the other phase. In such cases simple dual
   * cells consisting of 2 (6 in 3D) simplices don't capture such complex
   * topologies. To aleviate this issue I added an option to use a more detailed
   * dual cells consisting of 8 (48 in 3D) simplices. Clearly, it might increase
   * the computational cost quite significantly, but oh well...
   */

  // data for refined cells
  double fv_size_x = 0;
  double fv_size_y = 0;
#ifdef P4_TO_P8
  double fv_size_z = 0;
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  double full_cell_volume;
  double dxyz_pr[P4EST_DIM];
  double xyz_pr[P4EST_DIM];
  double dist;

  bool neighbors_exist[num_neighbors_max_];
  p4est_locidx_t neighbors[num_neighbors_max_];

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);

  if (variable_mu_)
#ifdef P4_TO_P8
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, mue_zz_p, interp_method_);
#else
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, interp_method_);
#endif

  my_p4est_interpolation_nodes_local_t phi_x_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t phi_y_local(node_neighbors_);
#ifdef P4_TO_P8
  my_p4est_interpolation_nodes_local_t phi_z_local(node_neighbors_);
#endif

  std::vector<double *> phi_x_ptr(num_interfaces_, NULL);
  std::vector<double *> phi_y_ptr(num_interfaces_, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_z_ptr(num_interfaces_, NULL);
#endif

  for (int i = 0; i < num_interfaces_; ++i)
  {
    ierr = VecGetArray(phi_x_->at(i), &phi_x_ptr[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y_->at(i), &phi_y_ptr[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z_->at(i), &phi_z_ptr[i]); CHKERRXX(ierr);
#endif
  }

  p4est_locidx_t node_m00_mm; p4est_locidx_t node_m00_pm;
  p4est_locidx_t node_p00_mm; p4est_locidx_t node_p00_pm;
  p4est_locidx_t node_0m0_mm; p4est_locidx_t node_0m0_pm;
  p4est_locidx_t node_0p0_mm; p4est_locidx_t node_0p0_pm;

#ifdef P4_TO_P8
  p4est_locidx_t node_m00_mp; p4est_locidx_t node_m00_pp;
  p4est_locidx_t node_p00_mp; p4est_locidx_t node_p00_pp;
  p4est_locidx_t node_0m0_mp; p4est_locidx_t node_0m0_pp;
  p4est_locidx_t node_0p0_mp; p4est_locidx_t node_0p0_pp;

  p4est_locidx_t node_00m_mm; p4est_locidx_t node_00m_mp;
  p4est_locidx_t node_00m_pm; p4est_locidx_t node_00m_pp;
  p4est_locidx_t node_00p_mm; p4est_locidx_t node_00p_mp;
  p4est_locidx_t node_00p_pm; p4est_locidx_t node_00p_pp;
#endif

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
#endif

  double val_interface_m00 = 0.;
  double val_interface_p00 = 0.;
  double val_interface_0m0 = 0.;
  double val_interface_0p0 = 0.;
#ifdef P4_TO_P8
  double val_interface_00m = 0.;
  double val_interface_00p = 0.;
#endif

#ifdef DO_NOT_PREALLOCATE
  mat_entry_t ent;
#endif

  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
  {
#ifdef DO_NOT_PREALLOCATE
    std::vector<mat_entry_t> * row = matrix_entries[n];
#endif
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    node_xyz_fr_n(n, p4est_, nodes_, xyz_C);
    double x_C  = xyz_C[0];
    double y_C  = xyz_C[1];
#ifdef P4_TO_P8
    double z_C  = xyz_C[2];
#endif

    double phi_eff_000 = phi_eff_p[n];

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
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

    if (setup_matrix)
    {
      /*
       * NOTE: All nodes are in PETSc' local numbering
       */
      node_m00_mm=qnnn.node_m00_mm; node_m00_pm=qnnn.node_m00_pm;
      node_p00_mm=qnnn.node_p00_mm; node_p00_pm=qnnn.node_p00_pm;
      node_0m0_mm=qnnn.node_0m0_mm; node_0m0_pm=qnnn.node_0m0_pm;
      node_0p0_mm=qnnn.node_0p0_mm; node_0p0_pm=qnnn.node_0p0_pm;

#ifdef P4_TO_P8
      node_m00_mp=qnnn.node_m00_mp; node_m00_pp=qnnn.node_m00_pp;
      node_p00_mp=qnnn.node_p00_mp; node_p00_pp=qnnn.node_p00_pp;
      node_0m0_mp=qnnn.node_0m0_mp; node_0m0_pp=qnnn.node_0m0_pp;
      node_0p0_mp=qnnn.node_0p0_mp; node_0p0_pp=qnnn.node_0p0_pp;

      node_00m_mm=qnnn.node_00m_mm; node_00m_mp=qnnn.node_00m_mp;
      node_00m_pm=qnnn.node_00m_pm; node_00m_pp=qnnn.node_00m_pp;
      node_00p_mm=qnnn.node_00p_mm; node_00p_mp=qnnn.node_00p_mp;
      node_00p_pm=qnnn.node_00p_pm; node_00p_pp=qnnn.node_00p_pp;
#endif

      w_m00_mm=0; w_m00_pm=0;
      w_p00_mm=0; w_p00_pm=0;
      w_0m0_mm=0; w_0m0_pm=0;
      w_0p0_mm=0; w_0p0_pm=0;
#ifdef P4_TO_P8
      w_m00_mp=0; w_m00_pp=0;
      w_p00_mp=0; w_p00_pp=0;
      w_0m0_mp=0; w_0m0_pp=0;
      w_0p0_mp=0; w_0p0_pp=0;

      w_00m_mm=0; w_00m_pm=0;
      w_00p_mm=0; w_00p_pm=0;
      w_00m_mp=0; w_00m_pp=0;
      w_00p_mp=0; w_00p_pp=0;
#endif
    }

    /*
     * global indecies: Note that to insert values into the matrix we need to
     * use global index. Note that although PETSc has a MatSetValuesLocal, that
     * function wont work properly with ghost nodes since the matix does not
     * know the partition of the grid and global indecies for ghost nodes.
     *
     * As a result we compute the ghost indecies manually and insert them using
     * the MatSetValue function.
     *
     * NOTE: Ideally we should be using p4est_gloidx_t for global numbers.
     * However, this requires PetscInt to be 64bit as well otherwise we might
     * run into problems since PETSc internally uses PetscInt for all integer
     * values.
     *
     * As a result, and to prevent weird things from happening, we simpy use
     * PetscInt instead of p4est_gloidx_t for global numbers. This should work
     * for problems that are up to about 2B point big (2^31-1). To go to bigger
     * problems, one should compile PETSc with 64bit support using
     * --with-64-bit-indecies. Please consult PETSc manual for more information.
     *
     * TODO: To get better performance we could first buffer the values in a
     * local SparseCRS matrix and insert them all at once at the end instead of
     * calling MatSetValue every single time. I'm not sure if it will result in
     * much better performance ... to be tested!
     */

    PetscInt node_000_g = petsc_gloidx_[qnnn.node_000];

    //* FIX THIS
    if(is_node_Wall(p4est_, ni) && phi_eff_000 < 0.)
    {
#ifdef P4_TO_P8
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
#else
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
#endif
      {
        if (setup_matrix)
        {
#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
#endif
          if (phi_eff_000 < 0. || num_interfaces_ == 0)
          {
            matrix_has_nullspace_ = false;
          }
        }

        if (setup_rhs)
        {
          rhs_p[n] = (*bc_wall_value_).value(xyz_C);
        }

        continue;
      }

      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
      if(neumann_wall_first_order_ &&
   #ifdef P4_TO_P8
         (*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == NEUMANN)
#else
         (*bc_wall_type_)(xyz_C[0], xyz_C[1]) == NEUMANN)
#endif
      {
        if (is_node_xpWall(p4est_, ni)){
          if (setup_matrix)
          {
#ifdef P4_TO_P8
            p4est_locidx_t n_m00 = d_m00_0m == 0 ? ( d_m00_m0==0 ? node_m00_mm : node_m00_pm )
                                                 : ( d_m00_m0==0 ? node_m00_mp : node_m00_pp );
#else
            p4est_locidx_t n_m00 = d_m00_m0 == 0 ? node_m00_mm : node_m00_pm;
#endif
            PetscInt node_m00_g  = petsc_gloidx_[n_m00];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_m00_g; ent.val = -1.; row->push_back(ent); ( n_m00 < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_m00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_m00;
          continue;
        }

        if (is_node_xmWall(p4est_, ni)){
          if (setup_matrix)
          {
#ifdef P4_TO_P8
            p4est_locidx_t n_p00 = d_p00_0m == 0 ? ( d_p00_m0 == 0 ? node_p00_mm : node_p00_pm )
                                                 : ( d_p00_m0 == 0 ? node_p00_mp : node_p00_pp );
#else
            p4est_locidx_t n_p00 = d_p00_m0 == 0 ? node_p00_mm : node_p00_pm;
#endif
            PetscInt node_p00_g  = petsc_gloidx_[n_p00];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_p00_g; ent.val = -1.; row->push_back(ent); ( n_p00 < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_p00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_p00;
          continue;
        }

        if (is_node_ypWall(p4est_, ni)){
          if (setup_matrix)
          {
#ifdef P4_TO_P8
            p4est_locidx_t n_0m0 = d_0m0_0m == 0 ? ( d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm )
                                                 : ( d_0m0_m0 == 0 ? node_0m0_mp : node_0m0_pp );
#else
            p4est_locidx_t n_0m0 = d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm;
#endif
            PetscInt node_0m0_g  = petsc_gloidx_[n_0m0];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_0m0_g; ent.val = -1.; row->push_back(ent); ( n_0m0 < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_0m0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0m0;
          continue;
        }
        if (is_node_ymWall(p4est_, ni)){
          if (setup_matrix)
          {
#ifdef P4_TO_P8
            p4est_locidx_t n_0p0 = d_0p0_0m == 0 ? ( d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm )
                                                 : ( d_0p0_m0 == 0 ? node_0p0_mp : node_0p0_pp );
#else
            p4est_locidx_t n_0p0 = d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm;
#endif
            PetscInt node_0p0_g  = petsc_gloidx_[n_0p0];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            //ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_0p0_g; ent.val = -1.; row->push_back(ent); ( n_0p0 < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_0p0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0p0;
          continue;
        }
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est_, ni)){
          if (setup_matrix)
          {
            p4est_locidx_t n_00m = d_00m_0m == 0 ? ( d_00m_m0 == 0 ? node_00m_mm : node_00m_pm )
                                                 : ( d_00m_m0 == 0 ? node_00m_mp : node_00m_pp );
            PetscInt node_00m_g  = petsc_gloidx_[n_00m];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_00m_g; ent.val = -1.; row->push_back(ent); ( n_00m < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_00m_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00m;
          continue;
        }

        if (is_node_zmWall(p4est_, ni)){
          if (setup_matrix)
          {
            p4est_locidx_t n_00p = d_00p_0m == 0 ? ( d_00p_m0 == 0 ? node_00p_mm : node_00p_pm )
                                                 : ( d_00p_m0 == 0 ? node_00p_mp : node_00p_pp );
            PetscInt node_00p_g  = petsc_gloidx_[n_00p];

#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_00p_g; ent.val = -1.; row->push_back(ent); ( n_00p < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_00p_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
#endif
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00p;
          continue;
        }
#endif
      }

    }
    //*/


#ifdef P4_TO_P8
    for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      qnnn.ngbd_with_quadratic_interpolation(phi_p[phi_idx], phi_000[phi_idx], phi_m00[phi_idx], phi_p00[phi_idx], phi_0m0[phi_idx], phi_0p0[phi_idx], phi_00m[phi_idx], phi_00p[phi_idx]);

    if (variable_mu_)
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
    for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      qnnn.ngbd_with_quadratic_interpolation(phi_p[phi_idx], phi_000[phi_idx], phi_m00[phi_idx], phi_p00[phi_idx], phi_0m0[phi_idx], phi_0p0[phi_idx]);

    if (variable_mu_)
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif

    interp_local.initialize(n);
    phi_interp_local.copy_init(interp_local);
    phi_x_local.copy_init(interp_local);
    phi_y_local.copy_init(interp_local);
#ifdef P4_TO_P8
    phi_z_local.copy_init(interp_local);
#endif

    //---------------------------------------------------------------------
    // check if finite volume is crossed
    //---------------------------------------------------------------------
    bool is_ngbd_crossed_dirichlet = false;
    bool is_ngbd_crossed_neumann   = false;

    if (fabs(phi_eff_000) < lip_*diag_min_)
    {
      get_all_neighbors(n, neighbors, neighbors_exist);

      // sample level-set function at nodes of the extended cube and check if crossed
      for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
        bool is_one_positive = false;
        bool is_one_negative = false;

        for (short i = 0; i < num_neighbors_max_; ++i)
          if (neighbors_exist[i])
          {
            is_one_positive = is_one_positive || phi_p[phi_idx][neighbors[i]] > 0;
            is_one_negative = is_one_negative || phi_p[phi_idx][neighbors[i]] < 0;
          }

        if (is_one_negative && is_one_positive)
        {
          if (bc_interface_type_->at(phi_idx) == DIRICHLET) is_ngbd_crossed_dirichlet = true;
          if (bc_interface_type_->at(phi_idx) == NEUMANN)   is_ngbd_crossed_neumann   = true;
          if (bc_interface_type_->at(phi_idx) == ROBIN)     is_ngbd_crossed_neumann   = true;
        }
      }
    }

    if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet) { throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment"); }
    else if (is_ngbd_crossed_neumann)                         { discretization_scheme_ = FVM; }
    else                                                      { discretization_scheme_ = FDM; }

    if (discretization_scheme_ == FDM)
    {
      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if (ABS(phi_eff_000) < EPS*diag_min_)
      {
        if (setup_matrix)
        {
#ifdef DO_NOT_PREALLOCATE
            ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
#endif

          if (use_pointwise_dirichlet_)
            pointwise_bc_[n].push_back(interface_point_t(0, EPS*diag_min_));

          matrix_has_nullspace_ = false;
        }

        if (setup_rhs)
        {
          if (use_pointwise_dirichlet_)
            rhs_p[n] = pointwise_bc_[n][0].value;
          else
            for (int i = 0; i < num_interfaces_; ++i)
              if (fabs(phi_p[i][n]) < EPS*diag_min_ && bc_interface_type_->at(i) == DIRICHLET)
                rhs_p[n] = bc_interface_value_->at(i)->value(xyz_C);
        }

        continue;
      }

      // far away from the interface
      if (phi_eff_000 > 0.)
      {
        if (setup_matrix)
        {
#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
#endif
          mask_p[n] = MAX(1., mask_p[n]);
        }

        if (setup_rhs)
        {
          rhs_p[n] = 0;
        }
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if (phi_eff_000 < 0.)
      {
        double theta_m00 = d_m00;
        double theta_p00 = d_p00;
        double theta_0m0 = d_0m0;
        double theta_0p0 = d_0p0;
#ifdef P4_TO_P8
        double theta_00m = d_00m;
        double theta_00p = d_00p;
#endif

        bool is_interface_m00 = false;
        bool is_interface_p00 = false;
        bool is_interface_0m0 = false;
        bool is_interface_0p0 = false;
#ifdef P4_TO_P8
        bool is_interface_00m = false;
        bool is_interface_00p = false;
#endif

        int phi_idx_m00 = -1;
        int phi_idx_p00 = -1;
        int phi_idx_0m0 = -1;
        int phi_idx_0p0 = -1;
#ifdef P4_TO_P8
        int phi_idx_00m = -1;
        int phi_idx_00p = -1;
#endif

        if (setup_rhs)
        {
          val_interface_m00 = 0.;
          val_interface_p00 = 0.;
          val_interface_0m0 = 0.;
          val_interface_0p0 = 0.;
#ifdef P4_TO_P8
          val_interface_00m = 0.;
          val_interface_00p = 0.;
#endif
        }

        // check if any Dirichlet interface crosses ngbd
        for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
          if (bc_interface_type_->at(phi_idx) == DIRICHLET)
          {
            if (phi_000[phi_idx]*phi_m00[phi_idx] <= 0.)
            {
              double phixx_m00 = qnnn.f_m00_linear(phi_xx_p[phi_idx]);
              double theta_m00_c = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000[phi_idx], phi_m00[phi_idx], phi_xx_p[phi_idx][n], phixx_m00);

              if (theta_m00_c < eps)   theta_m00_c = eps;
              if (theta_m00_c > d_m00) theta_m00_c = d_m00;

              if (theta_m00_c < theta_m00)
              {
                theta_m00   = theta_m00_c;
                phi_idx_m00 = phi_idx;
              }

              is_interface_m00 = true;
            }

            if (phi_000[phi_idx]*phi_p00[phi_idx] <= 0.)
            {
              double phixx_p00 = qnnn.f_p00_linear(phi_xx_p[phi_idx]);
              double theta_p00_c = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000[phi_idx], phi_p00[phi_idx], phi_xx_p[phi_idx][n], phixx_p00);

              if (theta_p00_c < eps)   theta_p00_c = eps;
              if (theta_p00_c > d_p00) theta_p00_c = d_p00;

              if (theta_p00_c < theta_p00)
              {
                theta_p00   = theta_p00_c;
                phi_idx_p00 = phi_idx;
              }

              is_interface_p00 = true;
            }

            if (phi_000[phi_idx]*phi_0m0[phi_idx] <= 0.)
            {
              double phixx_0m0 = qnnn.f_0m0_linear(phi_xx_p[phi_idx]);
              double theta_0m0_c = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000[phi_idx], phi_0m0[phi_idx], phi_xx_p[phi_idx][n], phixx_0m0);

              if (theta_0m0_c < eps)   theta_0m0_c = eps;
              if (theta_0m0_c > d_0m0) theta_0m0_c = d_0m0;

              if (theta_0m0_c < theta_0m0)
              {
                theta_0m0   = theta_0m0_c;
                phi_idx_0m0 = phi_idx;
              }

              is_interface_0m0 = true;
            }

            if (phi_000[phi_idx]*phi_0p0[phi_idx] <= 0.)
            {
              double phixx_0p0 = qnnn.f_0p0_linear(phi_xx_p[phi_idx]);
              double theta_0p0_c = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000[phi_idx], phi_0p0[phi_idx], phi_xx_p[phi_idx][n], phixx_0p0);

              if (theta_0p0_c < eps)   theta_0p0_c = eps;
              if (theta_0p0_c > d_0p0) theta_0p0_c = d_0p0;

              if (theta_0p0_c < theta_0p0)
              {
                theta_0p0   = theta_0p0_c;
                phi_idx_0p0 = phi_idx;
              }

              is_interface_0p0 = true;
            }
#ifdef P4_TO_P8
            if (phi_000[phi_idx]*phi_00m[phi_idx] <= 0.)
            {
              double phixx_00m = qnnn.f_00m_linear(phi_xx_p[phi_idx]);
              double theta_00m_c = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000[phi_idx], phi_00m[phi_idx], phi_xx_p[phi_idx][n], phixx_00m);

              if (theta_00m_c < eps)   theta_00m_c = eps;
              if (theta_00m_c > d_00m) theta_00m_c = d_00m;

              if (theta_00m_c < theta_00m)
              {
                theta_00m   = theta_00m_c;
                phi_idx_00m = phi_idx;
              }

              is_interface_00m = true;
            }

            if (phi_000[phi_idx]*phi_00p[phi_idx] <= 0.)
            {
              double phixx_00p = qnnn.f_00p_linear(phi_xx_p[phi_idx]);
              double theta_00p_c = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000[phi_idx], phi_00p[phi_idx], phi_xx_p[phi_idx][n], phixx_00p);

              if (theta_00p_c < eps)   theta_00p_c = eps;
              if (theta_00p_c > d_00p) theta_00p_c = d_00p;

              if (theta_00p_c < theta_00p)
              {
                theta_00p   = theta_00p_c;
                phi_idx_00p = phi_idx;
              }

              is_interface_00p = true;
            }
#endif
          }
        }

        if (is_interface_m00)
        {
          d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
          d_m00_0m = d_m00_0p = 0;
#endif
          if (variable_mu_)
          {
            double mxx_000 = mue_xx_p[n];
            double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
            mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);
          }

          d_m00 = theta_m00;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(0, theta_m00));

          if (setup_rhs)
#ifdef P4_TO_P8
            val_interface_m00 = (*bc_interface_value_->at(phi_idx_m00))(x_C - theta_m00, y_C, z_C);
#else
            val_interface_m00 = (*bc_interface_value_->at(phi_idx_m00))(x_C - theta_m00, y_C);
#endif
        }

        if (is_interface_p00)
        {
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif
          if (variable_mu_)
          {
            double mxx_000 = mue_xx_p[n];
            double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
            mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);
          }

          d_p00 = theta_p00;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(1, theta_p00));

          if (setup_rhs)
#ifdef P4_TO_P8
            val_interface_p00 = (*bc_interface_value_->at(phi_idx_p00))(x_C + theta_p00, y_C, z_C);
#else
            val_interface_p00 = (*bc_interface_value_->at(phi_idx_p00))(x_C + theta_p00, y_C);
#endif
        }

        if (is_interface_0m0)
        {
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif
          if (variable_mu_)
          {
            double myy_000 = mue_yy_p[n];
            double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
            mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);
          }

          d_0m0 = theta_0m0;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(2, theta_0m0));

          if (setup_rhs)
#ifdef P4_TO_P8
            val_interface_0m0 = (*bc_interface_value_->at(phi_idx_0m0))(x_C, y_C - theta_0m0, z_C);
#else
            val_interface_0m0 = (*bc_interface_value_->at(phi_idx_0m0))(x_C, y_C - theta_0m0);
#endif
        }

        if (is_interface_0p0)
        {
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
          d_0p0_0m = d_0p0_0p = 0;
#endif
          if (variable_mu_)
          {
            double myy_000 = mue_yy_p[n];
            double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
            mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);
          }

          d_0p0 = theta_0p0;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(3, theta_0p0));

          if (setup_rhs)
#ifdef P4_TO_P8
            val_interface_0p0 = (*bc_interface_value_->at(phi_idx_0p0))(x_C, y_C + theta_0p0, z_C);
#else
            val_interface_0p0 = (*bc_interface_value_->at(phi_idx_0p0))(x_C, y_C + theta_0p0);
#endif
        }
#ifdef P4_TO_P8
        if (is_interface_00m){
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;

          if (variable_mu_)
          {
            double mzz_000 = mue_zz_p[n];
            double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
            mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);
          }

          d_00m = theta_00m;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(4, theta_00m));

          if (setup_rhs)
            val_interface_00m = (*bc_interface_value_->at(phi_idx_00m))(x_C, y_C, z_C - theta_00m);
        }

        if (is_interface_00p){
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

          if (variable_mu_)
          {
            double mzz_000 = mue_zz_p[n];
            double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
            mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);
          }

          d_00p = theta_00p;

          if (setup_matrix)
            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(5, theta_00p));

          if (setup_rhs)
            val_interface_00p = (*bc_interface_value_->at(phi_idx_00p))(x_C, y_C, z_C + theta_00p);
        }
#endif

        if (setup_rhs)
          if (use_pointwise_dirichlet_)
          {
            std::vector<double> val_interface(2*P4EST_DIM, 0);
            for (short i = 0; i < pointwise_bc_[n].size(); ++i)
            {
              interface_point_t *pnt = &pointwise_bc_[n][i];
              val_interface[pnt->dir] = pnt->value;
            }

            val_interface_m00 = val_interface[0];
            val_interface_p00 = val_interface[1];
            val_interface_0m0 = val_interface[2];
            val_interface_0p0 = val_interface[3];
#ifdef P4_TO_P8
            val_interface_00m = val_interface[4];
            val_interface_00p = val_interface[5];
#endif
          }

        if ( is_interface_m00 || is_interface_p00 ||
     #ifdef P4_TO_P8
             is_interface_00m || is_interface_00p ||
     #endif
             is_interface_0m0 || is_interface_0p0 )
          matrix_has_nullspace_ = false;

#ifdef P4_TO_P8
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
        if(is_node_xmWall(p4est_, ni))      w_p00 += -1.0/(d_p00*d_p00);
        else if(is_node_xpWall(p4est_, ni)) w_m00 += -1.0/(d_m00*d_m00);
        else                                w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est_, ni))      w_m00 += -1.0/(d_m00*d_m00);
        else if(is_node_xmWall(p4est_, ni)) w_p00 += -1.0/(d_p00*d_p00);
        else                                w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est_, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est_, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
        else                                w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est_, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est_, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
        else                                w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

        if(is_node_zmWall(p4est_, ni))      w_00p += -1.0/(d_00p*d_00p);
        else if(is_node_zpWall(p4est_, ni)) w_00m += -1.0/(d_00m*d_00m);
        else                                w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

        if(is_node_zpWall(p4est_, ni))      w_00m += -1.0/(d_00m*d_00m);
        else if(is_node_zmWall(p4est_, ni)) w_00p += -1.0/(d_00p*d_00p);
        else                                w_00p += -2.0*wk/d_00p/(d_00m+d_00p);

        if (variable_mu_)
        {
          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && setup_matrix) {
            w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_mp = 0.5*(mue_000 + mue_p[node_m00_mp])*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pp = 0.5*(mue_000 + mue_p[node_m00_pp])*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          } else {
            w_m00 *= 0.5*(mue_000 + mue_m00);
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && setup_matrix) {
            w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_mp = 0.5*(mue_000 + mue_p[node_p00_mp])*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pp = 0.5*(mue_000 + mue_p[node_p00_pp])*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          } else {
            w_p00 *= 0.5*(mue_000 + mue_p00);
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && setup_matrix) {
            w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_mp = 0.5*(mue_000 + mue_p[node_0m0_mp])*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pp = 0.5*(mue_000 + mue_p[node_0m0_pp])*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          } else {
            w_0m0 *= 0.5*(mue_000 + mue_0m0);
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && setup_matrix) {
            w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_mp = 0.5*(mue_000 + mue_p[node_0p0_mp])*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pp = 0.5*(mue_000 + mue_p[node_0p0_pp])*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          } else {
            w_0p0 *= 0.5*(mue_000 + mue_0p0);
          }

          if(!is_interface_00m && !is_node_zmWall(p4est_, ni) && setup_matrix) {
            w_00m_mm = 0.5*(mue_000 + mue_p[node_00m_mm])*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_mp = 0.5*(mue_000 + mue_p[node_00m_mp])*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pm = 0.5*(mue_000 + mue_p[node_00m_pm])*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pp = 0.5*(mue_000 + mue_p[node_00m_pp])*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          } else {
            w_00m *= 0.5*(mue_000 + mue_00m);
          }

          if(!is_interface_00p && !is_node_zpWall(p4est_, ni) && setup_matrix) {
            w_00p_mm = 0.5*(mue_000 + mue_p[node_00p_mm])*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_mp = 0.5*(mue_000 + mue_p[node_00p_mp])*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pm = 0.5*(mue_000 + mue_p[node_00p_pm])*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pp = 0.5*(mue_000 + mue_p[node_00p_pp])*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
          } else {
            w_00p *= 0.5*(mue_000 + mue_00p);
          }

        } else {

          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && setup_matrix) {
            w_m00_mm = mu_*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_mp = mu_*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pm = mu_*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pp = mu_*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          } else {
            w_m00 *= mu_;
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && setup_matrix) {
            w_p00_mm = mu_*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_mp = mu_*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pm = mu_*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pp = mu_*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          } else {
            w_p00 *= mu_;
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && setup_matrix) {
            w_0m0_mm = mu_*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_mp = mu_*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pm = mu_*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pp = mu_*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          } else {
            w_0m0 *= mu_;
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && setup_matrix) {
            w_0p0_mm = mu_*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_mp = mu_*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pm = mu_*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pp = mu_*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          } else {
            w_0p0 *= mu_;
          }

          if(!is_interface_00m && !is_node_zmWall(p4est_, ni) && setup_matrix) {
            w_00m_mm = mu_*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_mp = mu_*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pm = mu_*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pp = mu_*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          } else {
            w_00m *= mu_;
          }

          if(!is_interface_00p && !is_node_zpWall(p4est_, ni) && setup_matrix) {
            w_00p_mm = mu_*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_mp = mu_*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pm = mu_*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pp = mu_*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
          } else {
            w_00p *= mu_;
          }

        }
#else
        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double wi = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);
        double wj = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

        // if node is at wall, what's below will apply Neumann BC
        if(is_node_xmWall(p4est_, ni))      w_p00 += -1.0/(d_p00*d_p00);
        else if(is_node_xpWall(p4est_, ni)) w_m00 += -1.0/(d_m00*d_m00);
        else                                w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est_, ni))      w_m00 += -1.0/(d_m00*d_m00);
        else if(is_node_xmWall(p4est_, ni)) w_p00 += -1.0/(d_p00*d_p00);
        else                                w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est_, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est_, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
        else                                w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est_, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est_, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
        else                                w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        if (variable_mu_) {

          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && setup_matrix) {
            w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
            w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
            w_m00 = w_m00_mm + w_m00_pm;
          } else {
            w_m00 *= 0.5*(mue_000 + mue_m00);
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && setup_matrix) {
            w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
            w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
            w_p00    = w_p00_mm + w_p00_pm;
          } else {
            w_p00 *= 0.5*(mue_000 + mue_p00);
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && setup_matrix) {
            w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
            w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
            w_0m0 = w_0m0_mm + w_0m0_pm;
          } else {
            w_0m0 *= 0.5*(mue_000 + mue_0m0);
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && setup_matrix) {
            w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
            w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);
            w_0p0 = w_0p0_mm + w_0p0_pm;
          } else {
            w_0p0 *= 0.5*(mue_000 + mue_0p0);
          }

        } else {

          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && setup_matrix) {
            w_m00_mm = mu_*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
            w_m00_pm = mu_*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
            w_m00 = w_m00_mm + w_m00_pm;
          } else {
            w_m00 *= mu_;
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && setup_matrix) {
            w_p00_mm = mu_*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
            w_p00_pm = mu_*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
            w_p00    = w_p00_mm + w_p00_pm;
          } else {
            w_p00 *= mu_;
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && setup_matrix) {
            w_0m0_mm = mu_*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
            w_0m0_pm = mu_*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
            w_0m0 = w_0m0_mm + w_0m0_pm;
          } else {
            w_0m0 *= mu_;
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && setup_matrix) {
            w_0p0_mm = mu_*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
            w_0p0_pm = mu_*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);
            w_0p0 = w_0p0_mm + w_0p0_pm;
          } else {
            w_0p0 *= mu_;
          }

        }
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

#ifdef P4_TO_P8
        double w_000  = diag_add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
        double w_000  = diag_add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif
        if (setup_matrix)
        {
//          mask_p[n] = -1;
          //---------------------------------------------------------------------
          // add coefficients in the matrix
          //---------------------------------------------------------------------
          if (node_000_g < fixed_value_idx_g_){
            fixed_value_idx_l_ = n;
            fixed_value_idx_g_ = node_000_g;
          }

#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);

          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni)) {
            if (ABS(w_m00_mm) > EPS) { ent.n = petsc_gloidx_[node_m00_mm]; ent.val = w_m00_mm/w_000; row->push_back(ent); ( node_m00_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_m00_pm) > EPS) { ent.n = petsc_gloidx_[node_m00_pm]; ent.val = w_m00_pm/w_000; row->push_back(ent); ( node_m00_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#ifdef P4_TO_P8
            if (ABS(w_m00_mp) > EPS) { ent.n = petsc_gloidx_[node_m00_mp]; ent.val = w_m00_mp/w_000; row->push_back(ent); ( node_m00_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_m00_pp) > EPS) { ent.n = petsc_gloidx_[node_m00_pp]; ent.val = w_m00_pp/w_000; row->push_back(ent); ( node_m00_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#endif
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni)) {
            if (ABS(w_p00_mm) > EPS) { ent.n = petsc_gloidx_[node_p00_mm]; ent.val = w_p00_mm/w_000; row->push_back(ent); ( node_p00_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_p00_pm) > EPS) { ent.n = petsc_gloidx_[node_p00_pm]; ent.val = w_p00_pm/w_000; row->push_back(ent); ( node_p00_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#ifdef P4_TO_P8
            if (ABS(w_p00_mp) > EPS) { ent.n = petsc_gloidx_[node_p00_mp]; ent.val = w_p00_mp/w_000; row->push_back(ent); ( node_p00_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_p00_pp) > EPS) { ent.n = petsc_gloidx_[node_p00_pp]; ent.val = w_p00_pp/w_000; row->push_back(ent); ( node_p00_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#endif
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni)) {
            if (ABS(w_0m0_mm) > EPS) { ent.n = petsc_gloidx_[node_0m0_mm]; ent.val = w_0m0_mm/w_000; row->push_back(ent); ( node_0m0_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_0m0_pm) > EPS) { ent.n = petsc_gloidx_[node_0m0_pm]; ent.val = w_0m0_pm/w_000; row->push_back(ent); ( node_0m0_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#ifdef P4_TO_P8
            if (ABS(w_0m0_mp) > EPS) { ent.n = petsc_gloidx_[node_0m0_mp]; ent.val = w_0m0_mp/w_000; row->push_back(ent); ( node_0m0_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_0m0_pp) > EPS) { ent.n = petsc_gloidx_[node_0m0_pp]; ent.val = w_0m0_pp/w_000; row->push_back(ent); ( node_0m0_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#endif
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni)) {
            if (ABS(w_0p0_mm) > EPS) { ent.n = petsc_gloidx_[node_0p0_mm]; ent.val = w_0p0_mm/w_000; row->push_back(ent); ( node_0p0_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_0p0_pm) > EPS) { ent.n = petsc_gloidx_[node_0p0_pm]; ent.val = w_0p0_pm/w_000; row->push_back(ent); ( node_0p0_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#ifdef P4_TO_P8
            if (ABS(w_0p0_mp) > EPS) { ent.n = petsc_gloidx_[node_0p0_mp]; ent.val = w_0p0_mp/w_000; row->push_back(ent); ( node_0p0_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_0p0_pp) > EPS) { ent.n = petsc_gloidx_[node_0p0_pp]; ent.val = w_0p0_pp/w_000; row->push_back(ent); ( node_0p0_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
#endif
          }
#ifdef P4_TO_P8
          if(!is_interface_00m && !is_node_zmWall(p4est_, ni)) {
            if (ABS(w_00m_mm) > EPS) { ent.n = petsc_gloidx_[node_00m_mm]; ent.val = w_00m_mm/w_000; row->push_back(ent); ( node_00m_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00m_pm) > EPS) { ent.n = petsc_gloidx_[node_00m_pm]; ent.val = w_00m_pm/w_000; row->push_back(ent); ( node_00m_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00m_mp) > EPS) { ent.n = petsc_gloidx_[node_00m_mp]; ent.val = w_00m_mp/w_000; row->push_back(ent); ( node_00m_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00m_pp) > EPS) { ent.n = petsc_gloidx_[node_00m_pp]; ent.val = w_00m_pp/w_000; row->push_back(ent); ( node_00m_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
          }

          if(!is_interface_00p && !is_node_zpWall(p4est_, ni)) {
            if (ABS(w_00p_mm) > EPS) { ent.n = petsc_gloidx_[node_00p_mm]; ent.val = w_00p_mm/w_000; row->push_back(ent); ( node_00p_mm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00p_pm) > EPS) { ent.n = petsc_gloidx_[node_00p_pm]; ent.val = w_00p_pm/w_000; row->push_back(ent); ( node_00p_pm < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00p_mp) > EPS) { ent.n = petsc_gloidx_[node_00p_mp]; ent.val = w_00p_mp/w_000; row->push_back(ent); ( node_00p_mp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
            if (ABS(w_00p_pp) > EPS) { ent.n = petsc_gloidx_[node_00p_pp]; ent.val = w_00p_pp/w_000; row->push_back(ent); ( node_00p_pp < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++; }
          }
#endif
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni)) {
            if (ABS(w_m00_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_m00_mm], w_m00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_m00_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_m00_pm], w_m00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
            if (ABS(w_m00_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_m00_mp], w_m00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_m00_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_m00_pp], w_m00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni)) {
            if (ABS(w_p00_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_p00_mm], w_p00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_p00_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_p00_pm], w_p00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
            if (ABS(w_p00_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_p00_mp], w_p00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_p00_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_p00_pp], w_p00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni)) {
            if (ABS(w_0m0_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0m0_mm], w_0m0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0m0_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0m0_pm], w_0m0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
            if (ABS(w_0m0_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0m0_mp], w_0m0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0m0_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0m0_pp], w_0m0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni)) {
            if (ABS(w_0p0_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0p0_mm], w_0p0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0p0_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0p0_pm], w_0p0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
            if (ABS(w_0p0_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0p0_mp], w_0p0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0p0_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_0p0_pp], w_0p0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
          }
#ifdef P4_TO_P8
          if(!is_interface_00m && !is_node_zmWall(p4est_, ni)) {
            if (ABS(w_00m_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00m_mm], w_00m_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00m_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00m_pm], w_00m_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00m_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00m_mp], w_00m_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00m_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00m_pp], w_00m_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          }

          if(!is_interface_00p && !is_node_zpWall(p4est_, ni)) {
            if (ABS(w_00p_mm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00p_mm], w_00p_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00p_pm) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00p_pm], w_00p_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00p_mp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00p_mp], w_00p_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00p_pp) > EPS) {ierr = MatSetValue(A_, node_000_g, petsc_gloidx_[node_00p_pp], w_00p_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          }
#endif
#endif

          if (keep_scalling_)
            scalling_[n] = w_000;

          if (diag_add_p[n] > 0) matrix_has_nullspace_ = false;
        }

        if (setup_rhs)
        {
          //---------------------------------------------------------------------
          // add coefficients to the right hand side
          //---------------------------------------------------------------------
          // FIX this for variable mu
//          if (variable_mu_) throw std::domain_error("This part doesn't work for variable mu yet\n");
#ifdef P4_TO_P8
          double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_xpWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);
          double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_ypWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);
          double eps_z = is_node_zmWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_zpWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);

          if(is_node_xmWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_p00;
          else if(is_interface_m00)      rhs_p[n] -= w_m00 * val_interface_m00;

          if(is_node_xpWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_m00;
          else if(is_interface_p00)      rhs_p[n] -= w_p00 * val_interface_p00;

          if(is_node_ymWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0p0;
          else if(is_interface_0m0)      rhs_p[n] -= w_0m0 * val_interface_0m0;

          if(is_node_ypWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0m0;
          else if(is_interface_0p0)      rhs_p[n] -= w_0p0 * val_interface_0p0;

          if(is_node_zmWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00p;
          else if(is_interface_00m)      rhs_p[n] -= w_00m * val_interface_00m;

          if(is_node_zpWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00m;
          else if(is_interface_00p)      rhs_p[n] -= w_00p * val_interface_00p;
#else

          double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_xpWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);
          double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_ypWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);

          if(is_node_xmWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_p00;
          else if(is_interface_m00)      rhs_p[n] -= w_m00*val_interface_m00;

          if(is_node_xpWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_m00;
          else if(is_interface_p00)      rhs_p[n] -= w_p00*val_interface_p00;

          if(is_node_ymWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0p0;
          else if(is_interface_0m0)      rhs_p[n] -= w_0m0*val_interface_0m0;

          if(is_node_ypWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0m0;
          else if(is_interface_0p0)      rhs_p[n] -= w_0p0*val_interface_0p0;
#endif

          rhs_p[n] /= w_000;
        }

        continue;
      }

    }
    else if (discretization_scheme_ == FVM)
    {
      if (use_sc_scheme_)
        for (char idx = 0; idx < num_neighbors_max_; ++idx)
          if (neighbors_exist[idx])
            neighbors_exist[idx] = neighbors_exist[idx] && (volumes_p[neighbors[idx]] > domain_rel_thresh_);

      // check for hanging neighbors
      int network[num_neighbors_max_];
      bool hanging_neighbor[num_neighbors_max_];
      bool expand[2*P4EST_DIM];
      bool attempt_to_expand = false;

      if (try_remove_hanging_cells_ && use_sc_scheme_)
      {
        for (char idx = 0; idx < num_neighbors_max_; ++idx)
          network[idx] = neighbors_exist[idx] ? (int) node_type_p[neighbors[idx]] : 0;

        find_hanging_cells(network, hanging_neighbor);

        for (char dir = 0; dir < P4EST_FACES; ++dir)
          expand[dir] = false;

#ifdef P4_TO_P8
        for (char k = 0; k < 3; ++k)
#endif
          for (char j = 0; j < 3; ++j)
            for (char i = 0; i < 3; ++i)
            {
#ifdef P4_TO_P8
              char idx = 9*k + 3*j +i;
#else
              char idx = 3*j +i;
#endif
              if (neighbors_exist[idx])
                if (hanging_neighbor[idx] && volumes_p[neighbors[idx]] < 10.5)
                {
                  if (i == 0) expand[dir::f_m00] = true;
                  if (i == 2) expand[dir::f_p00] = true;
                  if (j == 0) expand[dir::f_0m0] = true;
                  if (j == 2) expand[dir::f_0p0] = true;
#ifdef P4_TO_P8
                  if (k == 0) expand[dir::f_00m] = true;
                  if (k == 2) expand[dir::f_00p] = true;
#endif
                }
            }


        for (char dir = 0; dir < P4EST_FACES; ++dir)
          attempt_to_expand = attempt_to_expand || expand[dir];
      }
#ifdef P4_TO_P8
      cube3_mls_t cube;
#else
      cube2_mls_t cube;
#endif

      while (1)
      {
        // determine dimensions of cube
        fv_size_x = 0;
        fv_size_y = 0;
  #ifdef P4_TO_P8
        fv_size_z = 0;
  #endif

        if (!is_node_xmWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmin = x_C - .5*dx_min_; } else { fv_xmin = x_C; }
        if (!is_node_xpWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmax = x_C + .5*dx_min_; } else { fv_xmax = x_C; }
        if (!is_node_ymWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymin = y_C - .5*dy_min_; } else { fv_ymin = y_C; }
        if (!is_node_ypWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymax = y_C + .5*dy_min_; } else { fv_ymax = y_C; }
#ifdef P4_TO_P8
        if (!is_node_zmWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmin = z_C - .5*dz_min_; } else { fv_zmin = z_C; }
        if (!is_node_zpWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmax = z_C + .5*dz_min_; } else { fv_zmax = z_C; }
#endif

        if (cube_refinement_ == 0)
        {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }

        if (attempt_to_expand)
        {
          ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment...\n");
          if (expand[dir::f_m00]) { fv_size_x += 1; fv_xmin -= 1.*0.5*dx_min_; }
          if (expand[dir::f_p00]) { fv_size_x += 1; fv_xmax += 1.*0.5*dx_min_; }
          if (expand[dir::f_0m0]) { fv_size_y += 1; fv_ymin -= 1.*0.5*dy_min_; }
          if (expand[dir::f_0p0]) { fv_size_y += 1; fv_ymax += 1.*0.5*dy_min_; }
#ifdef P4_TO_P8
          if (expand[dir::f_00m]) { fv_size_z += 1; fv_zmin -= 1.*0.5*dz_min_; }
          if (expand[dir::f_00p]) { fv_size_z += 1; fv_zmax += 1.*0.5*dz_min_; }
#endif
        }

//        if (volumes_p[n] < 0.001 && use_sc_scheme_)
//        {
//          fv_size_x = 6;
//          fv_size_y = 6;
//#ifdef P4_TO_P8
//          fv_size_z = 6;
//#endif
//        }

        // Reconstruct geometry
#ifdef P4_TO_P8
        double cube_xyz_min[] = { fv_xmin, fv_ymin, fv_zmin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax, fv_zmax };
        int  cube_mnk[] = { fv_size_x, fv_size_y, fv_size_z };
#else
        double cube_xyz_min[] = { fv_xmin, fv_ymin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax };
        int  cube_mnk[] = { fv_size_x, fv_size_y };
#endif

        cube.initialize(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);

        // get points at which values of level-set functions are needed
        std::vector<double> x_grid; cube.get_x_coord(x_grid);
        std::vector<double> y_grid; cube.get_y_coord(y_grid);
#ifdef P4_TO_P8
        std::vector<double> z_grid; cube.get_z_coord(z_grid);
#endif

        int points_total = x_grid.size();

        std::vector<double> phi_cube(num_interfaces_*points_total,-1);

        // compute values of level-set functions at needed points
        for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
          if (phi_cf_ == NULL)
          {
#ifdef P4_TO_P8
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], interp_method_);
#else
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], interp_method_);
#endif
          }
          for (int i = 0; i < points_total; ++i)
          {
            if (phi_cf_ == NULL)
            {
#ifdef P4_TO_P8
              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
#else
              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
#endif
            } else {
#ifdef P4_TO_P8
              phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i], z_grid[i]);
#else
              phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i]);
#endif
            }

            // push interfaces inside the domain
            if (phi_cube[phi_idx*points_total + i] <  phi_perturbation_*diag_min_ &&
                phi_cube[phi_idx*points_total + i] > -phi_perturbation_*diag_min_)
              phi_cube[phi_idx*points_total + i] = phi_perturbation_*diag_min_;
          }
        }

        // reconstruct geometry
        cube.reconstruct(phi_cube, *action_, *color_);

        if (attempt_to_expand)
        {
          std::vector<double> W, X, Y, Z;
          // check if the attempt to expand was successful
          for (char dir = 0; dir < P4EST_FACES; ++dir)
          {
            if (expand[dir])
            {
              W.clear();
              X.clear();
              Y.clear();
              Z.clear();
#ifdef P4_TO_P8
              cube.quadrature_in_dir(dir, W, X, Y, Z);
#else
              cube.quadrature_in_dir(dir, W, X, Y);
#endif
              if (W.size() != 0) attempt_to_expand = false;
            }
          }

          if (attempt_to_expand)
          {
            std::cout << "Attempting hanging neighbors attachment... Success!\n";

            char im = expand[dir::f_m00] ? 0 : 1;
            char ip = expand[dir::f_p00] ? 2 : 1;
            char jm = expand[dir::f_0m0] ? 0 : 1;
            char jp = expand[dir::f_0p0] ? 2 : 1;
#ifdef P4_TO_P8
            char km = expand[dir::f_00m] ? 0 : 1;
            char kp = expand[dir::f_00p] ? 2 : 1;
#endif

#ifdef P4_TO_P8
            for (char k = km; k <= kp; ++k)
#endif
              for (char j = jm; j <= jp; ++j)
                for (char i = im; i <= ip; ++i)
                {
#ifdef P4_TO_P8
                  char idx = 9*k + 3*j + i;
#else
                  char idx = 3*j + i;
#endif
                  if (idx != nn_000) { neighbors_exist[idx] = false; if (setup_matrix) mask_p[neighbors[idx]] = 1; }
                }
            break;
          } else {
            std::cout << "Attempting hanging neighbors attachment... Failure!\n";
          }

        } else {
          break;
        }
      }

#ifdef P4_TO_P8
      full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin)*(fv_zmax-fv_zmin);
#else
      full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin);
#endif

      // get quadrature points
      std::vector<double> cube_dom_w;
      std::vector<double> cube_dom_x;
      std::vector<double> cube_dom_y;
#ifdef P4_TO_P8
      std::vector<double> cube_dom_z;
#endif

#ifdef P4_TO_P8
      cube.quadrature_over_domain(cube_dom_w, cube_dom_x, cube_dom_y, cube_dom_z);
#else
      cube.quadrature_over_domain(cube_dom_w, cube_dom_x, cube_dom_y);
#endif

      std::vector<std::vector<double> > cube_ifc_w(num_interfaces_);
      std::vector<std::vector<double> > cube_ifc_x(num_interfaces_);
      std::vector<std::vector<double> > cube_ifc_y(num_interfaces_);
#ifdef P4_TO_P8
      std::vector<std::vector<double> > cube_ifc_z(num_interfaces_);
#endif

      for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
#ifdef P4_TO_P8
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx], cube_ifc_z[phi_idx]);
#else
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx]);
#endif
      }

      // compute cut-cell volume
      double volume_cut_cell = 0.;
      double x_ctrd_cut_cell = 0.;
      double y_ctrd_cut_cell = 0.;
#ifdef P4_TO_P8
      double z_ctrd_cut_cell = 0.;
#endif


      for (int i = 0; i < cube_dom_w.size(); ++i)
      {
        volume_cut_cell += cube_dom_w[i];
        x_ctrd_cut_cell += cube_dom_w[i]*cube_dom_x[i];
        y_ctrd_cut_cell += cube_dom_w[i]*cube_dom_y[i];
#ifdef P4_TO_P8
        z_ctrd_cut_cell += cube_dom_w[i]*cube_dom_z[i];
#endif
      }

      x_ctrd_cut_cell /= volume_cut_cell;
      y_ctrd_cut_cell /= volume_cut_cell;
#ifdef P4_TO_P8
      z_ctrd_cut_cell /= volume_cut_cell;
      double xyz_c_cut_cell[P4EST_DIM] = { x_ctrd_cut_cell, y_ctrd_cut_cell, z_ctrd_cut_cell };
#else
      double xyz_c_cut_cell[P4EST_DIM] = { x_ctrd_cut_cell, y_ctrd_cut_cell };
#endif

      // compute area of the interface and integral of function from boundary conditions
      double integral_bc = 0.;

      std::vector<double> interface_centroid_x(num_interfaces_, 0);
      std::vector<double> interface_centroid_y(num_interfaces_, 0);
#ifdef P4_TO_P8
      std::vector<double> interface_centroid_z(num_interfaces_, 0);
#endif
      std::vector<double> interface_area(num_interfaces_, 0);

      for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
        if (cube_ifc_w[phi_idx].size() > 0)
        {
          for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
          {
            interface_area[phi_idx] += cube_ifc_w[phi_idx][i];
            #ifdef P4_TO_P8
                        integral_bc    += cube_ifc_w[phi_idx][i] * (*bc_interface_value_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i]);
            #else
                        integral_bc    += cube_ifc_w[phi_idx][i] * (*bc_interface_value_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i]);
            #endif
            interface_centroid_x[phi_idx] += cube_ifc_w[phi_idx][i]*cube_ifc_x[phi_idx][i];
            interface_centroid_y[phi_idx] += cube_ifc_w[phi_idx][i]*cube_ifc_y[phi_idx][i];
#ifdef P4_TO_P8
            interface_centroid_z[phi_idx] += cube_ifc_w[phi_idx][i]*cube_ifc_z[phi_idx][i];
#endif
          }
          interface_centroid_x[phi_idx] /= interface_area[phi_idx];
          interface_centroid_y[phi_idx] /= interface_area[phi_idx];
#ifdef P4_TO_P8
          interface_centroid_z[phi_idx] /= interface_area[phi_idx];
#endif

//#ifdef P4_TO_P8
//          integral_bc += interface_area[phi_idx] * (*bc_interface_value_->at(phi_idx))(interface_centroid_x[phi_idx], interface_centroid_y[phi_idx], interface_centroid_z[phi_idx]);
//#else
//          integral_bc += interface_area[phi_idx] * (*bc_interface_value_->at(phi_idx))(interface_centroid_x[phi_idx], interface_centroid_y[phi_idx]);
//#endif
        }
      }

      double volume_tmp = use_sc_scheme_ ? volumes_p[n] : volume_cut_cell;

//      if (volume_cut_cell/full_cell_volume > domain_rel_thresh_)
      if (volume_tmp > domain_rel_thresh_)
      {
        if (setup_rhs)
        {
          rhs_p[n] = rhs_p[n]*volume_cut_cell + integral_bc;
//          rhs_p[n] = rhs_cf_->value(xyz_c_cut_cell)*volume_cut_cell + integral_bc;
        }

        // get quadrature points
        std::vector<double> cube_dir_w;
        std::vector<double> cube_dir_x;
        std::vector<double> cube_dir_y;
#ifdef P4_TO_P8
        std::vector<double> cube_dir_z;
#endif

#ifdef P4_TO_P8
        double full_sx = (fv_ymax - fv_ymin)*(fv_zmax - fv_zmin);
        double full_sy = (fv_xmax - fv_xmin)*(fv_zmax - fv_zmin);
        double full_sz = (fv_xmax - fv_xmin)*(fv_ymax - fv_ymin);
#else
        double full_sx = fv_ymax - fv_ymin;
        double full_sy = fv_xmax - fv_xmin;
#endif

#ifdef P4_TO_P8
        double full_area_in_dir[] = { full_sx, full_sx, full_sy, full_sy, full_sz, full_sz };
#else
        double full_area_in_dir[] = { full_sx, full_sx, full_sy, full_sy };
#endif

        std::vector<double> area_in_dir(2*P4EST_DIM, 0);
        std::vector<double> centroid_x (2*P4EST_DIM, 0);
        std::vector<double> centroid_y (2*P4EST_DIM, 0);
#ifdef P4_TO_P8
        std::vector<double> centroid_z (2*P4EST_DIM, 0);
#endif

        for (int dir_idx = 0; dir_idx < 2*P4EST_DIM; ++dir_idx)
        {
          cube_dir_w.clear();
          cube_dir_x.clear();
          cube_dir_y.clear();
#ifdef P4_TO_P8
          cube_dir_z.clear();

          cube.quadrature_in_dir(dir_idx, cube_dir_w, cube_dir_x, cube_dir_y, cube_dir_z);
#else
          cube.quadrature_in_dir(dir_idx, cube_dir_w, cube_dir_x, cube_dir_y);
#endif
          if (cube_dir_w.size() > 0)
          {
            for (int i = 0; i < cube_dir_w.size(); ++i)
            {
              area_in_dir[dir_idx] += cube_dir_w[i];
              if (use_sc_scheme_)
              {
                centroid_x[dir_idx] += cube_dir_w[i]*(cube_dir_x[i] - x_C);
                centroid_y[dir_idx] += cube_dir_w[i]*(cube_dir_y[i] - y_C);
#ifdef P4_TO_P8
                centroid_z[dir_idx] += cube_dir_w[i]*(cube_dir_z[i] - z_C);
#endif
              }
            }

            centroid_x[dir_idx] /= area_in_dir[dir_idx];
            centroid_y[dir_idx] /= area_in_dir[dir_idx];
#ifdef P4_TO_P8
            centroid_z[dir_idx] /= area_in_dir[dir_idx];
#endif
          }
        }

        double s_m00 = area_in_dir[dir::f_m00], y_m00 = centroid_y[dir::f_m00];
        double s_p00 = area_in_dir[dir::f_p00], y_p00 = centroid_y[dir::f_p00];
        double s_0m0 = area_in_dir[dir::f_0m0], x_0m0 = centroid_x[dir::f_0m0];
        double s_0p0 = area_in_dir[dir::f_0p0], x_0p0 = centroid_x[dir::f_0p0];
#ifdef P4_TO_P8
        double s_00m = area_in_dir[dir::f_00m], x_00m = centroid_x[dir::f_00m], y_00m = centroid_y[dir::f_00m];
        double s_00p = area_in_dir[dir::f_00p], x_00p = centroid_x[dir::f_00p], y_00p = centroid_y[dir::f_00p];

        double z_m00 = centroid_z[dir::f_m00];
        double z_p00 = centroid_z[dir::f_p00];
        double z_0m0 = centroid_z[dir::f_0m0];
        double z_0p0 = centroid_z[dir::f_0p0];
#endif

        //---------------------------------------------------------------------
        // contributions through cell faces
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        double w[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0,
                                         0,0,0, 0,0,0, 0,0,0,
                                         0,0,0, 0,0,0, 0,0,0 };

        bool neighbors_exist_2d[9];
        double weights_2d[9];
        bool map_2d[9];

        double theta = EPS;

        // face m00
        if (s_m00/full_sx > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_m00])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_m00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_m00]]
                         << " Face Area:" << s_m00/full_sx << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(fv_xmin, y_C + y_m00, z_C + z_m00) : mu_;
            double flux = mu_val * s_m00 / dx_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_m00] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_0mm] && neighbors_exist[nn_mmm];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_00m] && neighbors_exist[nn_m0m];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_0pm] && neighbors_exist[nn_mpm];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_0m0] && neighbors_exist[nn_mm0];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_m00];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_0p0] && neighbors_exist[nn_mp0];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_0mp] && neighbors_exist[nn_mmp];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_00p] && neighbors_exist[nn_m0p];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_0pp] && neighbors_exist[nn_mpp];

              double A = y_m00/dy_min_;
              double B = z_m00/dz_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_0mm] += weights_2d[nn_mmm] * flux;   w[nn_mmm] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_00m] += weights_2d[nn_0mm] * flux;   w[nn_m0m] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_0pm] += weights_2d[nn_pmm] * flux;   w[nn_mpm] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_0m0] += weights_2d[nn_m0m] * flux;   w[nn_mm0] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_m00] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_0p0] += weights_2d[nn_p0m] * flux;   w[nn_mp0] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_0mp] += weights_2d[nn_mpm] * flux;   w[nn_mmp] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_00p] += weights_2d[nn_0pm] * flux;   w[nn_m0p] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_0pp] += weights_2d[nn_ppm] * flux;   w[nn_mpp] -= weights_2d[nn_ppm] * flux; }
            }

          }
        }

        // face p00
        if (s_p00/full_sx > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_p00])
          {
            std::cout << "Warning: neighbor doesn't exist in the xp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_p00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_p00]]
                         << " Face Area:" << s_p00/full_sx << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(fv_xmax, y_C + y_p00, z_C + z_p00) : mu_;
            double flux = mu_val * s_p00 / dx_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_p00] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_0mm] && neighbors_exist[nn_pmm];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_00m] && neighbors_exist[nn_p0m];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_0pm] && neighbors_exist[nn_ppm];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_0m0] && neighbors_exist[nn_pm0];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_p00];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_0p0] && neighbors_exist[nn_pp0];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_0mp] && neighbors_exist[nn_pmp];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_00p] && neighbors_exist[nn_p0p];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_0pp] && neighbors_exist[nn_ppp];

              double A = y_p00/dy_min_;
              double B = z_p00/dz_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_0mm] += weights_2d[nn_mmm] * flux;   w[nn_pmm] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_00m] += weights_2d[nn_0mm] * flux;   w[nn_p0m] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_0pm] += weights_2d[nn_pmm] * flux;   w[nn_ppm] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_0m0] += weights_2d[nn_m0m] * flux;   w[nn_pm0] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_p00] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_0p0] += weights_2d[nn_p0m] * flux;   w[nn_pp0] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_0mp] += weights_2d[nn_mpm] * flux;   w[nn_pmp] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_00p] += weights_2d[nn_0pm] * flux;   w[nn_p0p] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_0pp] += weights_2d[nn_ppm] * flux;   w[nn_ppp] -= weights_2d[nn_ppm] * flux; }
            }

          }
        }


        // face 0m0
        if (s_0m0/full_sy > interface_rel_thresh_)
        {

          if (!neighbors_exist[nn_0m0])
          {
            std::cout << "Warning: neighbor doesn't exist in the ym-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0m0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0m0]]
                         << " Face Area:" << s_0m0/full_sy << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_0m0, fv_ymin, z_C + z_0m0) : mu_;
            double flux = mu_val * s_0m0 / dy_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_0m0] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_m0m] && neighbors_exist[nn_mmm];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_m00] && neighbors_exist[nn_mm0];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_m0p] && neighbors_exist[nn_mmp];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_00m] && neighbors_exist[nn_0mm];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_0m0];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_00p] && neighbors_exist[nn_0mp];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_p0m] && neighbors_exist[nn_pmm];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_p00] && neighbors_exist[nn_pm0];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_p0p] && neighbors_exist[nn_pmp];

              double A = z_0m0/dz_min_;
              double B = x_0m0/dx_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_m0m] += weights_2d[nn_mmm] * flux;   w[nn_mmm] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_m00] += weights_2d[nn_0mm] * flux;   w[nn_mm0] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_m0p] += weights_2d[nn_pmm] * flux;   w[nn_mmp] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_00m] += weights_2d[nn_m0m] * flux;   w[nn_0mm] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_0m0] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_00p] += weights_2d[nn_p0m] * flux;   w[nn_0mp] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_p0m] += weights_2d[nn_mpm] * flux;   w[nn_pmm] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_p00] += weights_2d[nn_0pm] * flux;   w[nn_pm0] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_p0p] += weights_2d[nn_ppm] * flux;   w[nn_pmp] -= weights_2d[nn_ppm] * flux; }
            }
          }
        }

        // face 0p0
        if (s_0p0/full_sy > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_0p0])
          {
            std::cout << "Warning: neighbor doesn't exist in the yp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0p0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0p0]]
                         << " Face Area:" << s_0p0/full_sy << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_0p0, fv_ymax, z_C + z_0p0) : mu_;
            double flux = mu_val * s_0p0 / dy_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_0p0] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_m0m] && neighbors_exist[nn_mpm];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_m00] && neighbors_exist[nn_mp0];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_m0p] && neighbors_exist[nn_mpp];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_00m] && neighbors_exist[nn_0pm];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_0p0];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_00p] && neighbors_exist[nn_0pp];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_p0m] && neighbors_exist[nn_ppm];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_p00] && neighbors_exist[nn_pp0];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_p0p] && neighbors_exist[nn_ppp];

              double A = z_0p0/dz_min_;
              double B = x_0p0/dx_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_m0m] += weights_2d[nn_mmm] * flux;   w[nn_mpm] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_m00] += weights_2d[nn_0mm] * flux;   w[nn_mp0] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_m0p] += weights_2d[nn_pmm] * flux;   w[nn_mpp] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_00m] += weights_2d[nn_m0m] * flux;   w[nn_0pm] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_0p0] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_00p] += weights_2d[nn_p0m] * flux;   w[nn_0pp] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_p0m] += weights_2d[nn_mpm] * flux;   w[nn_ppm] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_p00] += weights_2d[nn_0pm] * flux;   w[nn_pp0] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_p0p] += weights_2d[nn_ppm] * flux;   w[nn_ppp] -= weights_2d[nn_ppm] * flux; }
            }

          }
        }


        // face 00m
        if (s_00m/full_sz > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_00m])
          {
            std::cout << "Warning: neighbor doesn't exist in the zm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00m]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00m]]
                         << " Face Area:" << s_00m/full_sz << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_00m, y_C + y_00m, fv_zmin) : mu_;
            double flux = mu_val * s_00m / dz_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_00m] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_mm0] && neighbors_exist[nn_mmm];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_0m0] && neighbors_exist[nn_0mm];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_pm0] && neighbors_exist[nn_pmm];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_m00] && neighbors_exist[nn_m0m];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_00m];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_p00] && neighbors_exist[nn_p0m];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_mp0] && neighbors_exist[nn_mpm];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_0p0] && neighbors_exist[nn_0pm];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_pp0] && neighbors_exist[nn_ppm];

              double A = x_00m/dx_min_;
              double B = y_00m/dy_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_mm0] += weights_2d[nn_mmm] * flux;   w[nn_mmm] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_0m0] += weights_2d[nn_0mm] * flux;   w[nn_0mm] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_pm0] += weights_2d[nn_pmm] * flux;   w[nn_pmm] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_m00] += weights_2d[nn_m0m] * flux;   w[nn_m0m] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_00m] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_p00] += weights_2d[nn_p0m] * flux;   w[nn_p0m] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_mp0] += weights_2d[nn_mpm] * flux;   w[nn_mpm] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_0p0] += weights_2d[nn_0pm] * flux;   w[nn_0pm] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_pp0] += weights_2d[nn_ppm] * flux;   w[nn_ppm] -= weights_2d[nn_ppm] * flux; }
            }

          }
        }

        // face 00p
        if (s_00p/full_sz > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_00p])
          {
            std::cout << "Warning: neighbor doesn't exist in the zp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00p]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00p]]
                         << " Face Area:" << s_00p/full_sz << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_00p, y_C + y_00p, fv_zmax) : mu_;
            double flux = mu_val * s_00p / dz_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_00p] -= flux;

            } else {

              neighbors_exist_2d[nn_mmm] = neighbors_exist[nn_mm0] && neighbors_exist[nn_mmp];
              neighbors_exist_2d[nn_0mm] = neighbors_exist[nn_0m0] && neighbors_exist[nn_0mp];
              neighbors_exist_2d[nn_pmm] = neighbors_exist[nn_pm0] && neighbors_exist[nn_pmp];
              neighbors_exist_2d[nn_m0m] = neighbors_exist[nn_m00] && neighbors_exist[nn_m0p];
              neighbors_exist_2d[nn_00m] = neighbors_exist[nn_000] && neighbors_exist[nn_00p];
              neighbors_exist_2d[nn_p0m] = neighbors_exist[nn_p00] && neighbors_exist[nn_p0p];
              neighbors_exist_2d[nn_mpm] = neighbors_exist[nn_mp0] && neighbors_exist[nn_mpp];
              neighbors_exist_2d[nn_0pm] = neighbors_exist[nn_0p0] && neighbors_exist[nn_0pp];
              neighbors_exist_2d[nn_ppm] = neighbors_exist[nn_pp0] && neighbors_exist[nn_ppp];

              double A = x_00p/dx_min_;
              double B = y_00p/dy_min_;
              double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

              if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

              if (map_2d[nn_mmm]) { w[nn_mm0] += weights_2d[nn_mmm] * flux;   w[nn_mmp] -= weights_2d[nn_mmm] * flux; }
              if (map_2d[nn_0mm]) { w[nn_0m0] += weights_2d[nn_0mm] * flux;   w[nn_0mp] -= weights_2d[nn_0mm] * flux; }
              if (map_2d[nn_pmm]) { w[nn_pm0] += weights_2d[nn_pmm] * flux;   w[nn_pmp] -= weights_2d[nn_pmm] * flux; }
              if (map_2d[nn_m0m]) { w[nn_m00] += weights_2d[nn_m0m] * flux;   w[nn_m0p] -= weights_2d[nn_m0m] * flux; }
              if (map_2d[nn_00m]) { w[nn_000] += weights_2d[nn_00m] * flux;   w[nn_00p] -= weights_2d[nn_00m] * flux; }
              if (map_2d[nn_p0m]) { w[nn_p00] += weights_2d[nn_p0m] * flux;   w[nn_p0p] -= weights_2d[nn_p0m] * flux; }
              if (map_2d[nn_mpm]) { w[nn_mp0] += weights_2d[nn_mpm] * flux;   w[nn_mpp] -= weights_2d[nn_mpm] * flux; }
              if (map_2d[nn_0pm]) { w[nn_0p0] += weights_2d[nn_0pm] * flux;   w[nn_0pp] -= weights_2d[nn_0pm] * flux; }
              if (map_2d[nn_ppm]) { w[nn_pp0] += weights_2d[nn_ppm] * flux;   w[nn_ppp] -= weights_2d[nn_ppm] * flux; }
            }

          }
        }

#else
        double w[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0 };

        double theta = EPS;

        //*

        // face m00
        if (s_m00/full_sx > interface_rel_thresh_)
        {
          if (!neighbors_exist[nn_m00])
          {
//            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
//                      << " Own number: " << n
//                      << " Nei number: " << neighbors[nn_m00]
//                         << " Own volume: " << volumes_p[neighbors[nn_000]]
//                         << " Nei volume: " << volumes_p[neighbors[nn_m00]]
//                         << " Face Area:" << s_m00/full_sx << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(fv_xmin, y_C + y_m00) : mu_;
            double flux = mu_val * s_m00/dx_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_m00] -= flux;

            } else {

              double ratio = fabs(y_m00)/dy_min_;

              if (y_m00/dy_min_ < -theta && neighbors_exist[nn_mm0] && neighbors_exist[nn_0m0]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_m00] -= (1. - ratio) * flux;
                w[nn_0m0] += ratio * flux;
                w[nn_mm0] -= ratio * flux;

              } else if (y_m00/dy_min_ > theta && neighbors_exist[nn_mp0] && neighbors_exist[nn_0p0]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_m00] -= (1. - ratio) * flux;
                w[nn_0p0] += ratio * flux;
                w[nn_mp0] -= ratio * flux;

              } else {

                w[nn_000] += flux;
                w[nn_m00] -= flux;

                if (setup_matrix && ratio >= theta)
                {
                  mask_p[n] = MAX(mask_p[n], -0.1);
//                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face p00
        if (s_p00/full_sx > interface_rel_thresh_)
        {

          if (!neighbors_exist[nn_p00])
          {
//            std::cout << "Warning: neighbor doesn't exist in the xp-direction."
//                      << " Own number: " << n
//                      << " Nei number: " << neighbors[nn_p00]
//                         << " Own volume: " << volumes_p[neighbors[nn_000]]
//                         << " Nei volume: " << volumes_p[neighbors[nn_p00]]
//                         << " Face Area:" << s_p00/full_sx << "\n";
          } else {

            double mu_val = variable_mu_ ? mu_val = interp_local.interpolate(fv_xmax, y_C + y_p00) : mu_;
            double flux = mu_val * s_p00/dx_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_p00] -= flux;

            } else {

              double ratio = fabs(y_p00)/dy_min_;

              if (y_p00/dy_min_ < -theta && neighbors_exist[nn_pm0] && neighbors_exist[nn_0m0]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_p00] -= (1. - ratio) * flux;
                w[nn_0m0] += ratio * flux;
                w[nn_pm0] -= ratio * flux;

              } else if (y_p00/dy_min_ > theta && neighbors_exist[nn_pp0] && neighbors_exist[nn_0p0]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_p00] -= (1. - ratio) * flux;
                w[nn_0p0] += ratio * flux;
                w[nn_pp0] -= ratio * flux;

              } else {

                w[nn_000] += flux;
                w[nn_p00] -= flux;

                if (ratio > theta && setup_matrix)
                {
                  mask_p[n] = MAX(mask_p[n], -0.1);
//                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face_0m0
        if (s_0m0/full_sy > interface_rel_thresh_)
        {

          if (!neighbors_exist[nn_0m0])
          {
//            std::cout << "Warning: neighbor doesn't exist in the ym-direction."
//                      << " Own number: " << n
//                      << " Nei number: " << neighbors[nn_0m0]
//                         << " Own volume: " << volumes_p[neighbors[nn_000]]
//                         << " Nei volume: " << volumes_p[neighbors[nn_0m0]]
//                         << " Face Area:" << s_0m0/full_sy << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_0m0, fv_ymin) : mu_;
            double flux = mu_val * s_0m0/dy_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_0m0] -= flux;

            } else {

              double ratio = fabs(x_0m0)/dx_min_;

              if (x_0m0/dx_min_ < -theta && neighbors_exist[nn_mm0] && neighbors_exist[nn_m00]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_0m0] -= (1. - ratio) * flux;
                w[nn_m00] += ratio * flux;
                w[nn_mm0] -= ratio * flux;

              } else if (x_0m0/dx_min_ >  theta && neighbors_exist[nn_pm0] && neighbors_exist[nn_p00]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_0m0] -= (1. - ratio) * flux;
                w[nn_p00] += ratio * flux;
                w[nn_pm0] -= ratio * flux;

              } else {

                w[nn_000] += flux;
                w[nn_0m0] -= flux;

                if (ratio > theta && setup_matrix)
                {
                  mask_p[n] = MAX(mask_p[n], -0.1);
//                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face_0p0
        if (s_0p0/full_sy > interface_rel_thresh_)
        {

          if (!neighbors_exist[nn_0p0])
          {
//            std::cout << "Warning: neighbor doesn't exist in the yp-direction."
//                      << " Own number: " << n
//                      << " Nei number: " << neighbors[nn_0p0]
//                         << " Own volume: " << volumes_p[neighbors[nn_000]]
//                         << " Nei volume: " << volumes_p[neighbors[nn_0p0]]
//                         << " Face Area:" << s_0p0/full_sy << "\n";
          } else {

            double mu_val = variable_mu_ ? interp_local.interpolate(x_C + x_0p0, fv_ymax) : mu_;
            double flux = mu_val * s_0p0/dy_min_;

            if (!use_sc_scheme_)
            {
              w[nn_000] += flux;
              w[nn_0p0] -= flux;

            } else {

              double ratio = fabs(x_0p0)/dx_min_;

              if (x_0p0/dx_min_ < -theta && neighbors_exist[nn_mp0] && neighbors_exist[nn_m00]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_0p0] -= (1. - ratio) * flux;
                w[nn_m00] += ratio * flux;
                w[nn_mp0] -= ratio * flux;

              } else if (x_0p0/dx_min_ >  theta && neighbors_exist[nn_pp0] && neighbors_exist[nn_p00]) {

                w[nn_000] += (1. - ratio) * flux;
                w[nn_0p0] -= (1. - ratio) * flux;
                w[nn_p00] += ratio * flux;
                w[nn_pp0] -= ratio * flux;

              } else {

                w[nn_000] += flux;
                w[nn_0p0] -= flux;

                if (ratio > theta && setup_matrix)
                {
                  mask_p[n] = MAX(mask_p[n], -0.1);
//                  std::cout << "Fallback fluxes\n";
                }
              }
            }

          }
        }

        //*/
#endif
        /* some trash
        //          if (mask_p[n] > 0)
        //          {
        //            std::cout << "Relative volume: " << volumes_p[n] << "\n";
        //            std::cout << "          " << neighbors_exist[18] << "-----------" << neighbors_exist[21] << "-----------" << neighbors_exist[24] << " \n";
        //            std::cout << "         /|          /|          /| \n";
        //            std::cout << "        / |         / |         / | \n";
        //            std::cout << "       /  |        /  |        /  | \n";
        //            std::cout << "      " << neighbors_exist[19] << "-----------" << neighbors_exist[22] << "-----------" << neighbors_exist[25] << "   | \n";
        //            std::cout << "     /|   |      /|   |      /|   | \n";
        //            std::cout << "    / |   " << neighbors_exist[9] << "-----/-|---" << neighbors_exist[12] << "-----/-|---" << neighbors_exist[15] << " \n";
        //            std::cout << "   /  |  /|    /  |  /|    /  |  /| \n";
        //            std::cout << "  " << neighbors_exist[20] << "-----------" << neighbors_exist[23] << "-----------" << neighbors_exist[26] << "   | / | \n";
        //            std::cout << "  |   |/  |   |   |/  |   |   |/  | \n";
        //            std::cout << "  |   " << neighbors_exist[10] << "-------|---" << neighbors_exist[13] << "-------|---" << neighbors_exist[16] << "   | \n";
        //            std::cout << "  |  /|   |   |  /|   |   |  /|   | \n";
        //            std::cout << "  | / |   " << neighbors_exist[0] << "---|-/-|---" << neighbors_exist[3] << "---|-/-|---" << neighbors_exist[6] << " \n";
        //            std::cout << "  |/  |  /    |/  |  /    |/  |  /  \n";
        //            std::cout << "  " << neighbors_exist[11] << "-----------" << neighbors_exist[14] << "-----------" << neighbors_exist[17] << "   | /   \n";
        //            std::cout << "  |   |/      |   |/      |   |/    \n";
        //            std::cout << "  |   " << neighbors_exist[1] << "-------|---" << neighbors_exist[4] << "-------|---" << neighbors_exist[7] << "     \n";
        //            std::cout << "  |  /        |  /        |  /      \n";
        //            std::cout << "  | /         | /         | /       \n";
        //            std::cout << "  |/          |/          |/        \n";
        //            std::cout << "  " << neighbors_exist[2] << "-----------" << neighbors_exist[5] << "-----------" << neighbors_exist[8] << "         \n";
        //          }
        // testing hypothesis
        //          if (mask_p[n] > 0)

        //          {
        //            std::vector<int> present_interfaces;

        //            std::vector<double> measure_of_interface(num_interfaces_, 0);

        //            bool is_there_kink = false;

        //            for (int phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
        //            {
        //              for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
        //                measure_of_interface[phi_idx] += cubes[cube_idx]->integrate_over_interface(unity_cf_, color_->at(phi_idx));

        //              if (bc_interface_type_->at(phi_idx) == ROBIN && measure_of_interface[phi_idx] > eps_ifc_)
        //              {
        //                if (present_interfaces.size() > 0 and action_->at(phi_idx) != COLORATION)
        //                  is_there_kink = true;

        //                present_interfaces.push_back(phi_idx);
        //              }
        //            }

        //            short num_interfaces_present = present_interfaces.size();

//                    if (num_interfaces_present > 1)
//                    {
//                      mask_p[n] = 1;
//        //              std::cout << num_interfaces_present << std::endl;
//                      if (setup_matrix)
//                      {
//                        ierr = MatSetValue(A_, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
//                      }

//                      if (setup_rhs)
//                      {
//                        rhs_p[n] = bc_wall_value_->value(xyz_C);
//                      }

//                      continue;
//                    }
//                  }
//*/


        bool sc_scheme_successful = false;
        // a variation of the least-square fitting approach
        if (use_sc_scheme_)
        {
          sc_scheme_successful = true;

          // linear system
          char num_constraints = num_neighbors_max_ + num_interfaces_;

          std::vector<double> col_1st(num_constraints, 0);
          std::vector<double> col_2nd(num_constraints, 0);
          std::vector<double> col_3rd(num_constraints, 0);
#ifdef P4_TO_P8
          std::vector<double> col_4th(num_constraints, 0);
#endif

          std::vector<double> bc_values(num_interfaces_, 0);
          std::vector<double> bc_coeffs(num_interfaces_, 0);

          std::vector<double> S(num_interfaces_, 0);

          std::vector<double> x0(num_interfaces_, 0);
          std::vector<double> y0(num_interfaces_, 0);
#ifdef P4_TO_P8
          std::vector<double> z0(num_interfaces_, 0);
#endif

#ifdef P4_TO_P8
          for (char k = 0; k < 3; ++k)
#endif
            for (char j = 0; j < 3; ++j)
              for (char i = 0; i < 3; ++i)
              {
#ifdef P4_TO_P8
                char idx = 9*k+3*j+i;
#else
                char idx = 3*j+i;
#endif
                col_1st[idx] = 1.;
                col_2nd[idx] = ((double) (i-1)) * dxyz_m_[0];
                col_3rd[idx] = ((double) (j-1)) * dxyz_m_[1];
#ifdef P4_TO_P8
                col_4th[idx] = ((double) (k-1)) * dxyz_m_[2];
#endif
              }

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], interp_method_);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], interp_method_);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif

              // compute centroid of an interface
              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                S [phi_idx] += cube_ifc_w[phi_idx][i];
                x0[phi_idx] += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0[phi_idx] += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0[phi_idx] += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
              }

              x0[phi_idx] /= S[phi_idx];
              y0[phi_idx] /= S[phi_idx];
#ifdef P4_TO_P8
              z0[phi_idx] /= S[phi_idx];
#endif

#ifdef P4_TO_P8
              double xyz0[P4EST_DIM] = { x0[phi_idx], y0[phi_idx], z0[phi_idx] };
#else
              double xyz0[P4EST_DIM] = { x0[phi_idx], y0[phi_idx] };
#endif

              // compute signed distance and normal at the centroid
              double nx = phi_x_local.value(xyz0);
              double ny = phi_y_local.value(xyz0);
#ifdef P4_TO_P8
              double nz = phi_z_local.value(xyz0);
#endif

#ifdef P4_TO_P8
              double norm = sqrt(nx*nx+ny*ny+nz*nz);
#else
              double norm = sqrt(nx*nx+ny*ny);
#endif
              nx /= norm;
              ny /= norm;
#ifdef P4_TO_P8
              nz /= norm;
#endif
              double dist0 = phi_interp_local.value(xyz0)/norm;

              double xyz_pr[P4EST_DIM];

              xyz_pr[0] = xyz0[0] - dist0*nx;
              xyz_pr[1] = xyz0[1] - dist0*ny;
#ifdef P4_TO_P8
              xyz_pr[2] = xyz0[2] - dist0*nz;
#endif
              xyz_pr[0] = MIN(xyz_pr[0], fv_xmax); xyz_pr[0] = MAX(xyz_pr[0], fv_xmin);
              xyz_pr[1] = MIN(xyz_pr[1], fv_ymax); xyz_pr[1] = MAX(xyz_pr[1], fv_ymin);
#ifdef P4_TO_P8
              xyz_pr[2] = MIN(xyz_pr[2], fv_zmax); xyz_pr[2] = MAX(xyz_pr[2], fv_zmin);
#endif

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              col_1st[num_neighbors_max_ + phi_idx] = bc_coeff_proj;
              col_2nd[num_neighbors_max_ + phi_idx] = (mu_proj*nx + bc_coeff_proj*(xyz_pr[0] - xyz_C[0]));
              col_3rd[num_neighbors_max_ + phi_idx] = (mu_proj*ny + bc_coeff_proj*(xyz_pr[1] - xyz_C[1]));
#ifdef P4_TO_P8
              col_4th[num_neighbors_max_ + phi_idx] = (mu_proj*nz + bc_coeff_proj*(xyz_pr[2] - xyz_C[2]));
#endif
              bc_coeffs[phi_idx] = bc_coeff_proj;
              bc_values[phi_idx] = bc_value_proj;
            }
          }

          double gamma = 3.*log(10.);


          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0 && sc_scheme_successful)
            {
              std::vector<double> weight(num_constraints, 0);

              char num_constraints_present = 0;

#ifdef P4_TO_P8
              for (char k = 0; k < 3; ++k)
#endif
                for (char j = 0; j < 3; ++j)
                  for (char i = 0; i < 3; ++i)
                  {
#ifdef P4_TO_P8
                    char idx = 9*k+3*j+i;
#else
                    char idx = 3*j+i;
#endif
                    double dx = ((double) (i-1)) * dxyz_m_[0];
                    double dy = ((double) (j-1)) * dxyz_m_[1];
#ifdef P4_TO_P8
                    double dz = ((double) (k-1)) * dxyz_m_[2];
#endif

                    if (neighbors_exist[idx])
                    {
                      weight[idx] = exp(-gamma*(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
                          #ifdef P4_TO_P8
                                                SQR((z_C+dz-z0[phi_idx])/dz_min_) +
                          #endif
                                                SQR((y_C+dy-y0[phi_idx])/dy_min_)));
                      num_constraints_present++;
                    }
                  }

              for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                if (cube_ifc_w[phi_jdx].size() > 0)
                {
                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
                                                   #ifdef P4_TO_P8
                                                                     SQR((z0[phi_jdx]-z0[phi_idx])/dz_min_) +
                                                   #endif
                                                                     SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_)));
                  num_constraints_present++;
                }
              }

              if (num_constraints_present <= P4EST_DIM+1)
              {
                sc_scheme_successful = false;
                continue;
              }

              // assemble and invert matrix
              char A_size = (P4EST_DIM+1);
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

              for (char nei = 0; nei < num_constraints; ++nei)
              {
                A[0*A_size + 0] += col_1st[nei]*col_1st[nei]*weight[nei];
                A[0*A_size + 1] += col_1st[nei]*col_2nd[nei]*weight[nei];
                A[0*A_size + 2] += col_1st[nei]*col_3rd[nei]*weight[nei];
                A[1*A_size + 1] += col_2nd[nei]*col_2nd[nei]*weight[nei];
                A[1*A_size + 2] += col_2nd[nei]*col_3rd[nei]*weight[nei];
                A[2*A_size + 2] += col_3rd[nei]*col_3rd[nei]*weight[nei];
#ifdef P4_TO_P8
                A[0*A_size + 3] += col_1st[nei]*col_4th[nei]*weight[nei];
                A[1*A_size + 3] += col_2nd[nei]*col_4th[nei]*weight[nei];
                A[2*A_size + 3] += col_3rd[nei]*col_4th[nei]*weight[nei];
                A[3*A_size + 3] += col_4th[nei]*col_4th[nei]*weight[nei];
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

#ifdef P4_TO_P8
              if (!inv_mat4_(A, A_inv)) throw;
#else
              if (!inv_mat3_(A, A_inv)) throw;
#endif

              // compute Taylor expansion coefficients
              std::vector<double> coeff_const_term(num_constraints, 0);
              std::vector<double> coeff_x_term    (num_constraints, 0);
              std::vector<double> coeff_y_term    (num_constraints, 0);
#ifdef P4_TO_P8
              std::vector<double> coeff_z_term    (num_constraints, 0);
#endif

              for (char nei = 0; nei < num_constraints; ++nei)
              {
#ifdef P4_TO_P8
                coeff_const_term[nei] = weight[nei]*
                    ( A_inv[0*A_size+0]*col_1st[nei]
                    + A_inv[0*A_size+1]*col_2nd[nei]
                    + A_inv[0*A_size+2]*col_3rd[nei]
                    + A_inv[0*A_size+3]*col_4th[nei] );

                coeff_x_term[nei] = weight[nei]*
                    ( A_inv[1*A_size+0]*col_1st[nei]
                    + A_inv[1*A_size+1]*col_2nd[nei]
                    + A_inv[1*A_size+2]*col_3rd[nei]
                    + A_inv[1*A_size+3]*col_4th[nei] );

                coeff_y_term[nei] = weight[nei]*
                    ( A_inv[2*A_size+0]*col_1st[nei]
                    + A_inv[2*A_size+1]*col_2nd[nei]
                    + A_inv[2*A_size+2]*col_3rd[nei]
                    + A_inv[2*A_size+3]*col_4th[nei] );

                coeff_z_term[nei] = weight[nei]*
                    ( A_inv[3*A_size+0]*col_1st[nei]
                    + A_inv[3*A_size+1]*col_2nd[nei]
                    + A_inv[3*A_size+2]*col_3rd[nei]
                    + A_inv[3*A_size+3]*col_4th[nei] );
#else
                coeff_const_term[nei] = weight[nei]*
                    ( A_inv[0*A_size+0]*col_1st[nei]
                    + A_inv[0*A_size+1]*col_2nd[nei]
                    + A_inv[0*A_size+2]*col_3rd[nei] );

                coeff_x_term[nei] = weight[nei]*
                    ( A_inv[1*A_size+0]*col_1st[nei]
                    + A_inv[1*A_size+1]*col_2nd[nei]
                    + A_inv[1*A_size+2]*col_3rd[nei] );

                coeff_y_term[nei] = weight[nei]*
                    ( A_inv[2*A_size+0]*col_1st[nei]
                    + A_inv[2*A_size+1]*col_2nd[nei]
                    + A_inv[2*A_size+2]*col_3rd[nei] );
#endif
              }

              double rhs_const_term = 0;
              double rhs_x_term     = 0;
              double rhs_y_term     = 0;
#ifdef P4_TO_P8
              double rhs_z_term     = 0;
#endif
              for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                rhs_const_term += coeff_const_term[num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
                rhs_x_term     += coeff_x_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
                rhs_y_term     += coeff_y_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
#ifdef P4_TO_P8
                rhs_z_term     += coeff_z_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
#endif
              }

              // compute integrals
              double const_term = S[phi_idx]*bc_coeffs[phi_idx];
              double x_term     = S[phi_idx]*bc_coeffs[phi_idx]*(x0[phi_idx] - xyz_C[0]);
              double y_term     = S[phi_idx]*bc_coeffs[phi_idx]*(y0[phi_idx] - xyz_C[1]);
#ifdef P4_TO_P8
              double z_term     = S[phi_idx]*bc_coeffs[phi_idx]*(z0[phi_idx] - xyz_C[2]);
#endif

              // matrix coefficients
              for (char nei = 0; nei < num_neighbors_max_; ++nei)
              {
#ifdef P4_TO_P8
                w[nei] += coeff_const_term[nei]*const_term
                    + coeff_x_term[nei]*x_term
                    + coeff_y_term[nei]*y_term
                    + coeff_z_term[nei]*z_term;
#else
                w[nei] += coeff_const_term[nei]*const_term
                    + coeff_x_term[nei]*x_term
                    + coeff_y_term[nei]*y_term;
#endif
              }

              if (setup_rhs)
#ifdef P4_TO_P8
                rhs_p[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term + rhs_z_term*z_term;
#else
                rhs_p[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term;
#endif

              if (setup_matrix && fabs(const_term) > 0) matrix_has_nullspace_ = false;
            }
          }
        }

        //*/

        if (!use_sc_scheme_ || !sc_scheme_successful)
        {
          if (setup_matrix && use_sc_scheme_)
          {
            mask_p[n] = MAX(mask_p[n], -0.1);
            std::cout << "Fallback Robin BC\n";
          }

          // code below asumes that we have only Robin or Neumann BC in a cell
          //---------------------------------------------------------------------
          // contribution through interfaces
          //---------------------------------------------------------------------

          // count interfaces
          std::vector<int> present_interfaces;

          std::vector<double> measure_of_interface(num_interfaces_, 0);

          bool is_there_kink = false;

          for (int phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
          {
            for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              measure_of_interface[phi_idx] += cube_ifc_w[phi_idx][i];

            if (bc_interface_type_->at(phi_idx) == ROBIN && cube_ifc_w[phi_idx].size() > 0)
            {
              if (present_interfaces.size() > 0 and action_->at(phi_idx) != COLORATION)
                is_there_kink = true;

              present_interfaces.push_back(phi_idx);
            }
          }

          short num_interfaces_present = present_interfaces.size();

          // Special treatment for kinks
          if (is_there_kink && kink_special_treatment_ && num_interfaces_present > 1)
          {
            double N_mat[P4EST_DIM*P4EST_DIM];
#ifdef P4_TO_P8
            /* TO FIX (case of > 3 interfaces in 3D):
             * Should be N_mat[P4EST_DIM*num_normals],
             * where num_normals = max(P4EST_DIM, num_ifaces);
             */
#endif
            double bc_interface_coeff_avg[P4EST_DIM];
            double bc_interface_value_avg[P4EST_DIM];
            double a_coeff[P4EST_DIM];
            double b_coeff[P4EST_DIM];

            // Compute normals to interfaces
            for (short i = 0; i < num_interfaces_present; ++i)
              compute_normal_(phi_p[present_interfaces[i]], qnnn, &N_mat[i*P4EST_DIM]);

#ifdef P4_TO_P8
            /* an ad-hoc: in case of an intersection of 2 interfaces in 3D
             * we choose the third direction to be perpendicular to the first two
             * and set du/dn = 0 (i.e., u = const) along this third direction
             */
            if (num_interfaces_present == 2)
            {
              N_mat[2*P4EST_DIM + 0] = N_mat[0*P4EST_DIM + 1]*N_mat[1*P4EST_DIM + 2] - N_mat[0*P4EST_DIM + 2]*N_mat[1*P4EST_DIM + 1];
              N_mat[2*P4EST_DIM + 1] = N_mat[0*P4EST_DIM + 2]*N_mat[1*P4EST_DIM + 0] - N_mat[0*P4EST_DIM + 0]*N_mat[1*P4EST_DIM + 2];
              N_mat[2*P4EST_DIM + 2] = N_mat[0*P4EST_DIM + 0]*N_mat[1*P4EST_DIM + 1] - N_mat[0*P4EST_DIM + 1]*N_mat[1*P4EST_DIM + 0];

              double norm = sqrt(SQR(N_mat[2*P4EST_DIM + 0]) + SQR(N_mat[2*P4EST_DIM + 1]) + SQR(N_mat[2*P4EST_DIM + 2]));
              N_mat[2*P4EST_DIM + 0] /= norm;
              N_mat[2*P4EST_DIM + 1] /= norm;
              N_mat[2*P4EST_DIM + 2] /= norm;

              bc_interface_coeff_avg[2] = 0;
              bc_interface_value_avg[2] = 0;
            }
#endif


            // compute centroids projections and sample BC
            for (char i = 0; i < num_interfaces_present; ++i)
            {
              char phi_idx = present_interfaces[i];

#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], interp_method_);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], interp_method_);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif
              // compute centroid of an interface
              double S = 0;
              double x0 = 0;
              double y0 = 0;
              double z0 = 0;

              for (int ii = 0; ii < cube_ifc_w[phi_idx].size(); ++ii)
              {
                S  += cube_ifc_w[phi_idx][ii];
                x0 += cube_ifc_w[phi_idx][ii]*(cube_ifc_x[phi_idx][ii]);
                y0 += cube_ifc_w[phi_idx][ii]*(cube_ifc_y[phi_idx][ii]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][ii]*(cube_ifc_z[phi_idx][ii]);
#endif
              }

              x0 /= S;
              y0 /= S;
#ifdef P4_TO_P8
              z0 /= S;
#endif

#ifdef P4_TO_P8
              double xyz0[P4EST_DIM] = { x0, y0, z0 };
#else
              double xyz0[P4EST_DIM] = { x0, y0 };
#endif

              // compute signed distance and normal at the centroid
              double nx = phi_x_local.value(xyz0);
              double ny = phi_y_local.value(xyz0);
#ifdef P4_TO_P8
              double nz = phi_z_local.value(xyz0);
#endif

#ifdef P4_TO_P8
              double norm = sqrt(nx*nx+ny*ny+nz*nz);
#else
              double norm = sqrt(nx*nx+ny*ny);
#endif
              nx /= norm;
              ny /= norm;
#ifdef P4_TO_P8
              nz /= norm;
#endif
              double dist0 = phi_interp_local.value(xyz0)/norm;

              double xyz_pr[P4EST_DIM];

              xyz_pr[0] = xyz0[0] - dist0*nx;
              xyz_pr[1] = xyz0[1] - dist0*ny;
#ifdef P4_TO_P8
              xyz_pr[2] = xyz0[2] - dist0*nz;
#endif

              xyz_pr[0] = MIN(xyz_pr[0], fv_xmax); xyz_pr[0] = MAX(xyz_pr[0], fv_xmin);
              xyz_pr[1] = MIN(xyz_pr[1], fv_ymax); xyz_pr[1] = MAX(xyz_pr[1], fv_ymin);
#ifdef P4_TO_P8
              xyz_pr[2] = MIN(xyz_pr[2], fv_zmax); xyz_pr[2] = MAX(xyz_pr[2], fv_zmin);
#endif

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              for (char dim = 0; dim < P4EST_DIM; ++dim)
              {
                N_mat[i*P4EST_DIM+dim] = N_mat[i*P4EST_DIM+dim]*mu_proj + bc_coeff_proj*(xyz_pr[dim] - xyz_C[dim]);
              }

              bc_interface_coeff_avg[i] = bc_coeff_proj;
              bc_interface_value_avg[i] = bc_value_proj;
            }

            // Solve matrix
            double N_inv_mat[P4EST_DIM*P4EST_DIM];
#ifdef P4_TO_P8
            /* TO FIX (case of > 3 interfaces in 3D):
             * one should solve an overdetermined matrix N by the least-squares approach
             */
            inv_mat3_(N_mat, N_inv_mat);
#else
            inv_mat2_(N_mat, N_inv_mat);
#endif

            // calculate coefficients in Taylor series of u
            for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
            {
              a_coeff[i_dim] = 0;
              b_coeff[i_dim] = 0;
              for (short j_dim = 0; j_dim < P4EST_DIM; j_dim++)
              {
                a_coeff[i_dim] += N_inv_mat[i_dim*P4EST_DIM + j_dim]*bc_interface_coeff_avg[j_dim];
                b_coeff[i_dim] += N_inv_mat[i_dim*P4EST_DIM + j_dim]*bc_interface_value_avg[j_dim];
              }
            }

            // compute integrals
            for (short interface_idx = 0; interface_idx < present_interfaces.size(); ++interface_idx)
            {
              int phi_idx = present_interfaces[interface_idx];

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
#ifdef P4_TO_P8
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i] };
#else
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
#endif
                double bc_coeff = bc_interface_coeff_->at(phi_idx)->value(xyz);

                double taylor_expansion_coeff_term = bc_coeff*(1.-a_coeff[0]*(xyz[0]-xyz_C[0])
    #ifdef P4_TO_P8
                                                                 -a_coeff[2]*(xyz[2]-xyz_C[2])
    #endif
                                                                 -a_coeff[1]*(xyz[1]-xyz_C[1]));

                double taylor_expansion_const_term = bc_coeff*(b_coeff[0]*(xyz[0]-xyz_C[0]) +
    #ifdef P4_TO_P8
                                                               b_coeff[2]*(xyz[2]-xyz_C[2]) +
    #endif
                                                               b_coeff[1]*(xyz[1]-xyz_C[1]));

                w[nn_000] += cube_ifc_w[phi_idx][i] * taylor_expansion_coeff_term;

                if (setup_rhs)
                  rhs_p[n] -= cube_ifc_w[phi_idx][i] * taylor_expansion_const_term;
              }

            }

          }

          // Cells without kinks
          if (!is_there_kink || !kink_special_treatment_)
          {
            for (int interface_idx = 0; interface_idx < present_interfaces.size(); interface_idx++)
            {
              short i_phi = present_interfaces[interface_idx];

              if (bc_interface_type_->at(i_phi) == ROBIN)
              {
                double measure_of_iface = measure_of_interface[i_phi];

                // compute projection point
                find_projection_(phi_p[i_phi], qnnn, dxyz_pr, dist);

                for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                  xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

                xyz_pr[0] = MIN(xyz_pr[0], fv_xmax); xyz_pr[0] = MAX(xyz_pr[0], fv_xmin);
                xyz_pr[1] = MIN(xyz_pr[1], fv_ymax); xyz_pr[1] = MAX(xyz_pr[1], fv_ymin);
#ifdef P4_TO_P8
                xyz_pr[2] = MIN(xyz_pr[2], fv_zmax); xyz_pr[2] = MAX(xyz_pr[2], fv_zmin);
#endif

                // sample values at the projection point
                double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
                double bc_coeff_proj = bc_interface_coeff_->at(i_phi)->value(xyz_pr);

                // addition to diagonal term
                if (use_taylor_correction_) { w[nn_000] += bc_coeff_proj*measure_of_iface/(1.-bc_coeff_proj*dist/mu_proj); }
                else                        { w[nn_000] += bc_coeff_proj*measure_of_iface; }

                // addition to right-hand-side
                if (setup_rhs && use_taylor_correction_)
                {
                  double bc_value_proj = bc_interface_value_->at(i_phi)->value(xyz_pr);
                  rhs_p[n] -= measure_of_iface*bc_coeff_proj*bc_value_proj*dist/(bc_coeff_proj*dist-mu_proj);
                }

                if (fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
              }
            }
          }

        } // end of symmetric scheme

        // add diagonal term
        w[nn_000] += diag_add_p[n]*volume_cut_cell;

        // scale all coefficient by the diagonal one and insert them into the matrix
        if (setup_rhs)
        {
          rhs_p[n] /= w[nn_000];
        }

        if (setup_matrix)
        {
          if (fabs(diag_add_p[n]) > 0) matrix_has_nullspace_ = false;
          if (keep_scalling_) scalling_[n] = w[nn_000]/full_cell_volume; // scale to measure of the full cell dx*dy(*dz) for consistence

          double w_000 = w[nn_000];

          if (w_000 != w_000) throw std::domain_error("Diag is nan\n");

          for (int i = 0; i < num_neighbors_max_; ++i)
            w[i] /= w_000;

          w[nn_000] = 1.;

#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1,  ADD_VALUES); CHKERRXX(ierr);
#endif

          for (int i = 0; i < num_neighbors_max_; ++i)
            if (neighbors_exist[i] && fabs(w[i]) > EPS && i != nn_000)
            {
              if (w[i] != w[i]) throw std::domain_error("Non-diag is nan\n");
              PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
#ifdef DO_NOT_PREALLOCATE
              ent.n = node_nei_g; ent.val = w[i]; row->push_back(ent); ( neighbors[i] < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
#else
              ierr = MatSetValue(A_, node_000_g, node_nei_g, w[i],  ADD_VALUES); CHKERRXX(ierr);
#endif
            }

//#ifdef P4_TO_P8
//          if (volumes_p[n] < 1.e-3 && use_sc_scheme_)
//            mask_p[n] = MAX(-0.3, mask_p[n]);
//#endif
        }

      } else {

        if (setup_matrix && volume_cut_cell != 0.)
          ierr = PetscPrintf(p4est_->mpicomm, "Ignoring tiny volume %e\n", volume_cut_cell);

        // if finite volume too small, ignore the node
        if (setup_matrix)
        {
          mask_p[n] = MAX(1., mask_p[n]);
#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
#endif
        }

        if (setup_rhs)
          rhs_p[n] = 0;
      }

    } // end of if (discretization_scheme_ = discretization_scheme_t::FVM)

  }

  if (setup_matrix)
  {
    ierr = VecGhostUpdateBegin(mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (setup_matrix)
  {
#ifdef DO_NOT_PREALLOCATE
    if (A_ != NULL) { ierr = MatDestroy(A_); CHKERRXX(ierr); }
    /* set up the matrix */
    ierr = MatCreate(p4est_->mpicomm, &A_); CHKERRXX(ierr);
    ierr = MatSetType(A_, MATAIJ); CHKERRXX(ierr);
    ierr = MatSetSizes(A_, num_owned_local , num_owned_local,
                       num_owned_global, num_owned_global); CHKERRXX(ierr);
    ierr = MatSetFromOptions(A_); CHKERRXX(ierr);

    /* allocate the matrix */
    ierr = MatSeqAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
    ierr = MatMPIAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

    /* fill the matrix with the values */
    for(unsigned int n=0; n<num_owned_local; ++n)
    {
      PetscInt global_n_idx = petsc_gloidx_[n];
      std::vector<mat_entry_t> * row = matrix_entries[n];
      for(unsigned int m = 0; m < row->size(); ++m)
      {
        ierr = MatSetValue(A_, global_n_idx, row->at(m).n, row->at(m).val, ADD_VALUES); CHKERRXX(ierr);
      }
      delete row;
    }
#endif

    /* assemble the matrix */
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);
  }


//  if (use_sc_scheme_)
  {
    ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);
  }

  // restore pointers
  ierr = VecRestoreArray(mask_, &mask_p); CHKERRXX(ierr);

  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  for (int i = 0; i < num_interfaces_; ++i)
  {
    ierr = VecRestoreArray(phi_x_->at(i), &phi_x_ptr[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y_->at(i), &phi_y_ptr[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z_->at(i), &phi_z_ptr[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  if (variable_mu_) {
    ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(diag_add_,    &diag_add_p   ); CHKERRXX(ierr);

  if (exact_ != NULL) { ierr = VecRestoreArray(exact_, &exact_ptr); CHKERRXX(ierr); }

  if (setup_rhs)
  {
    ierr = VecRestoreArray(rhs_, &rhs_p); CHKERRXX(ierr);
  }

//  for (int i = 0; i < num_interfaces_; ++i)
//    delete phi_interp_local[i];

  // check for null space
  // FIXME: the return value should be checked for errors ...
  if (setup_matrix)
    MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace_, 1, MPI_INT, MPI_LAND, p4est_->mpicomm);

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

  if (setup_matrix)
  {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
  }


}
