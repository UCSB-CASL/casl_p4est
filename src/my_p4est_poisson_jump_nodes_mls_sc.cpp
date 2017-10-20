#ifdef P4_TO_P8
#include "my_p8est_poisson_jump_nodes_mls_sc.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/simplex3_mls_vtk.h>
#else
#include "my_p4est_poisson_jump_nodes_mls_sc.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#endif

#include <tools/plotting.h>

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


my_p4est_poisson_jump_nodes_mls_sc_t::my_p4est_poisson_jump_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    // p4est
    p4est_(node_neighbors->p4est), nodes_(node_neighbors->nodes), ghost_(node_neighbors->ghost), myb_(node_neighbors->myb),
    // linear system
    A_(NULL),
    new_pc_(true),
    is_matrix_computed_(false), matrix_has_nullspace_(false),
    // geometry
    phi_(NULL), phi_xx_(NULL), phi_yy_(NULL), phi_zz_(NULL), is_phi_dd_owned_(false), phi_eff_(NULL), is_phi_eff_owned_(false),
    action_(NULL), color_(NULL),
    num_interfaces_(0),
    // equation
    rhs_block_(NULL),
    rhs_m_(NULL), diag_add_scalar_m_(0.), diag_add_m_(NULL),
    rhs_p_(NULL), diag_add_scalar_p_(0.), diag_add_p_(NULL),
    mu_m_(1.), mue_m_(NULL), mue_m_xx_(NULL), mue_m_yy_(NULL), mue_m_zz_(NULL),
    mu_p_(1.), mue_p_(NULL), mue_p_xx_(NULL), mue_p_yy_(NULL), mue_p_zz_(NULL),
    is_mue_dd_owned_(false), variable_mu_(false),
    // bc
    neumann_wall_first_order_(false),
    bc_wall_type_(NULL), bc_wall_value_(NULL),
    u_jump_(NULL), mu_un_jump_(NULL),
    //other
    mask_m_(NULL),
    mask_p_(NULL),
    keep_scalling_(false),
    volumes_(NULL)
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

  scalling_.resize(2*nodes_->num_owned_indeps, 1);
//  pointwise_bc.resize(nodes->num_owned_indeps);

//  double eps = 1E-9*d_min_;

//#ifdef P4_TO_P8
//  eps_dom_ = eps*eps*eps;
//  eps_ifc_ = eps*eps;
//#else
//  eps_dom_ = eps*eps;
//  eps_ifc_ = eps;
//#endif

  eps_dom_ = 1.e-13;
  eps_ifc_ = 1.e-13;

  use_sc_scheme_ = false;
}

my_p4est_poisson_jump_nodes_mls_sc_t::~my_p4est_poisson_jump_nodes_mls_sc_t()
{
  if (mask_m_ != NULL) { ierr = VecDestroy(mask_m_); CHKERRXX(ierr); }
  if (mask_p_ != NULL) { ierr = VecDestroy(mask_p_); CHKERRXX(ierr); }

  if (A_             != NULL) {ierr = MatDestroy(A_);                      CHKERRXX(ierr);}
  if (ksp_           != NULL) {ierr = KSPDestroy(ksp_);                    CHKERRXX(ierr);}
  if (is_phi_dd_owned_)
  {
    if (phi_xx_ != NULL)
    {
      for (int i = 0; i < phi_xx_->size(); i++) {ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr);}
      delete phi_xx_;
    }

    if (phi_yy_ != NULL)
    {
      for (int i = 0; i < phi_yy_->size(); i++) {ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr);}
      delete phi_yy_;
    }

#ifdef P4_TO_P8
    if (phi_zz_ != NULL)
    {
      for (int i = 0; i < phi_zz_->size(); i++) {ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr);}
      delete phi_zz_;
    }
#endif
  }

  if (is_mue_dd_owned_)
  {
    if (mue_m_xx_     != NULL) {ierr = VecDestroy(mue_m_xx_);                CHKERRXX(ierr);}
    if (mue_m_yy_     != NULL) {ierr = VecDestroy(mue_m_yy_);                CHKERRXX(ierr);}
#ifdef P4_TO_P8
    if (mue_m_zz_     != NULL) {ierr = VecDestroy(mue_m_zz_);                CHKERRXX(ierr);}
#endif
    if (mue_p_xx_     != NULL) {ierr = VecDestroy(mue_p_xx_);                CHKERRXX(ierr);}
    if (mue_p_yy_     != NULL) {ierr = VecDestroy(mue_p_yy_);                CHKERRXX(ierr);}
#ifdef P4_TO_P8
    if (mue_p_zz_     != NULL) {ierr = VecDestroy(mue_p_zz_);                CHKERRXX(ierr);}
#endif
  }

  if (volumes_ != NULL)  {ierr = VecDestroy(volumes_); CHKERRXX(ierr);}
  if (is_phi_eff_owned_) {ierr = VecDestroy(phi_eff_);  CHKERRXX(ierr);}

  if (rhs_block_ != NULL) { ierr = VecDestroy(rhs_block_); CHKERRXX(ierr); }
}


void my_p4est_poisson_jump_nodes_mls_sc_t::compute_phi_eff_()
{
  if (phi_eff_ != NULL && is_phi_eff_owned_) { ierr = VecDestroy(phi_eff_); CHKERRXX(ierr); }

  is_phi_eff_owned_ = true;

  ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_eff_); CHKERRXX(ierr);

  std::vector<double *>   phi_p(num_interfaces_, NULL);
  double                 *phi_eff_p;

  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecGetArray(phi_->at(i), &phi_p[i]);  CHKERRXX(ierr);
  }

  ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n) // loop over nodes
  {
    phi_eff_p[n] = -10.;

    for (int i = 0; i < num_interfaces_; i++)
    {
      switch (action_->at(i))
      {
      case INTERSECTION:  phi_eff_p[n] = (phi_eff_p[n] > phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case ADDITION:      phi_eff_p[n] = (phi_eff_p[n] < phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case COLORATION:    /* do nothing */ break;
      }
    }
  }

  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]);  CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(phi_eff_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (phi_eff_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_nodes_mls_sc_t::compute_phi_dd_()
{
  // Allocate memory for second derivaties
  if (phi_xx_ != NULL && is_phi_dd_owned_)
  {
    for (int i = 0; i < phi_xx_->size(); i++) {ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr);}
    delete phi_xx_;
  }
  phi_xx_ = new std::vector<Vec> ();

  if (phi_yy_ != NULL && is_phi_dd_owned_)
  {
    for (int i = 0; i < phi_yy_->size(); i++) {ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr);}
    delete phi_yy_;
  }
  phi_yy_ = new std::vector<Vec> ();

#ifdef P4_TO_P8
  if (phi_zz_ != NULL && is_phi_dd_owned_)
  {
    for (int i = 0; i < phi_zz_->size(); i++) {ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr);}
    delete phi_zz_;
  }
  phi_zz_ = new std::vector<Vec> ();
#endif

  for (unsigned int i = 0; i < num_interfaces_; i++)
  {
    phi_xx_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_xx_->at(i)); CHKERRXX(ierr);
    phi_yy_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_yy_->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    phi_zz_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_zz_->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors_->second_derivatives_central(phi_->at(i), phi_xx_->at(i), phi_yy_->at(i), phi_zz_->at(i));
#else
    node_neighbors_->second_derivatives_central(phi_->at(i), phi_xx_->at(i), phi_yy_->at(i));
#endif
  }
  is_phi_dd_owned_ = true;
}

void my_p4est_poisson_jump_nodes_mls_sc_t::compute_mue_dd_()
{
  if (mue_m_xx_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_m_xx_); CHKERRXX(ierr); }
  if (mue_m_yy_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_m_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if (mue_m_zz_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_m_zz_); CHKERRXX(ierr); }
#endif

  // Allocate memory for second derivaties
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_xx_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  node_neighbors_->second_derivatives_central(mue_m_, mue_m_xx_, mue_m_yy_, mue_m_zz_);
#else
  node_neighbors_->second_derivatives_central(mue_m_, mue_m_xx_, mue_m_yy_);
#endif


  if (mue_p_xx_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_p_xx_); CHKERRXX(ierr); }
  if (mue_p_yy_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_p_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
  if (mue_p_zz_ != NULL && is_mue_dd_owned_) { ierr = VecDestroy(mue_p_zz_); CHKERRXX(ierr); }
#endif

  // Allocate memory for second derivaties
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_xx_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  node_neighbors_->second_derivatives_central(mue_p_, mue_p_xx_, mue_p_yy_, mue_p_zz_);
#else
  node_neighbors_->second_derivatives_central(mue_p_, mue_p_xx_, mue_p_yy_);
#endif


  is_mue_dd_owned_ = true;
}

void my_p4est_poisson_jump_nodes_mls_sc_t::preallocate_row(p4est_locidx_t n, const quad_neighbor_nodes_of_node_t& qnnn, std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz)
{
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

#ifdef P4_TO_P8
  if (qnnn.d_m00_p0*qnnn.d_m00_0p != 0) // node_m00_mm will enter discretization
    qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_m00_m0*qnnn.d_m00_0p != 0) // node_m00_pm will enter discretization
    qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_m00_p0*qnnn.d_m00_0m != 0) // node_m00_mp will enter discretization
    qnnn.node_m00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_m00_m0*qnnn.d_m00_0m != 0) // node_m00_pp will enter discretization
    qnnn.node_m00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
  if (qnnn.d_m00_p0 != 0) // node_m00_mm will enter discretization
    qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_m00_m0 != 0) // node_m00_pm will enter discretization
    qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
  if (qnnn.d_p00_p0*qnnn.d_p00_0p != 0) // node_p00_mm will enter discretization
    qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_p00_m0*qnnn.d_p00_0p != 0) // node_p00_pm will enter discretization
    qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_p00_p0*qnnn.d_p00_0m != 0) // node_p00_mp will enter discretization
    qnnn.node_p00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_p00_m0*qnnn.d_p00_0m != 0) // node_p00_pp will enter discretization
    qnnn.node_p00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
  if (qnnn.d_p00_p0 != 0) // node_p0_m will enter discretization
    qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_p00_m0 != 0) // node_p0_p will enter discretization
    qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
  if (qnnn.d_0m0_p0*qnnn.d_0m0_0p != 0) // node_0m0_mm will enter discretization
    qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0m0_m0*qnnn.d_0m0_0p != 0) // node_0m0_pm will enter discretization
    qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0m0_p0*qnnn.d_0m0_0m != 0) // node_0m0_mp will enter discretization
    qnnn.node_0m0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0m0_m0*qnnn.d_0m0_0m != 0) // node_0m0_pp will enter discretization
    qnnn.node_0m0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
  if (qnnn.d_0m0_p0 != 0) // node_0m_m will enter discretization
    qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0m0_m0 != 0) // node_0m_p will enter discretization
    qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
  if (qnnn.d_0p0_p0*qnnn.d_0p0_0p != 0) // node_0p0_mm will enter discretization
    qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0p0_m0*qnnn.d_0p0_0p != 0) // node_0p0_pm will enter discretization
    qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0p0_p0*qnnn.d_0p0_0m != 0) // node_0p0_mp will enter discretization
    qnnn.node_0p0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0p0_m0*qnnn.d_0p0_0m != 0) // node_0p0_pp will enter discretization
    qnnn.node_0p0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
  if (qnnn.d_0p0_p0 != 0) // node_0p_m will enter discretization
    qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_0p0_m0 != 0) // node_0p_p will enter discretization
    qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
  if (qnnn.d_00m_p0*qnnn.d_00m_0p != 0) // node_00m_mm will enter discretization
    qnnn.node_00m_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00m_m0*qnnn.d_00m_0p != 0) // node_00m_pm will enter discretization
    qnnn.node_00m_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00m_p0*qnnn.d_00m_0m != 0) // node_00m_mp will enter discretization
    qnnn.node_00m_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00m_m0*qnnn.d_00m_0m != 0) // node_00m_pp will enter discretization
    qnnn.node_00m_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;

  if (qnnn.d_00p_p0*qnnn.d_00p_0p != 0) // node_00p_mm will enter discretization
    qnnn.node_00p_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00p_m0*qnnn.d_00p_0p != 0) // node_00p_pm will enter discretization
    qnnn.node_00p_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00p_p0*qnnn.d_00p_0m != 0) // node_00p_mp will enter discretization
    qnnn.node_00p_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
  if (qnnn.d_00p_m0*qnnn.d_00p_0m != 0) // node_00p_pp will enter discretization
    qnnn.node_00p_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif
}

void my_p4est_poisson_jump_nodes_mls_sc_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = 2 * global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = 2 * (PetscInt)(nodes_->num_owned_indeps);

  if (A_ != NULL)
    ierr = MatDestroy(A_); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est_->mpicomm, &A_); CHKERRXX(ierr);
  ierr = MatSetType(A_, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A_, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A_); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);
  double *phi_p;
  ierr = VecGetArray(phi_eff_, &phi_p); CHKERRXX(ierr);

  bool neighbors_exist[num_neighbors_max_];
  p4est_locidx_t neighbors[num_neighbors_max_];

  double *volumes_p;
  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  for (p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

    if (volumes_p[n] < eps_dom_ || volumes_p[n] > 1.-eps_dom_)
    {
      if (phi_p[n] <  2*diag_min_) preallocate_row(2*n  , qnnn, d_nnz, o_nnz); // allocate for omega^-
      if (phi_p[n] > -2*diag_min_) preallocate_row(2*n+1, qnnn, d_nnz, o_nnz); // allocate for omega^+
    } else {

      get_all_neighbors_(n, neighbors, neighbors_exist);

//      if (phi_p[n] > 0)
//      {
//        preallocate_row(2*n, qnnn, d_nnz, o_nnz);
//        d_nnz[2*n] += 1;
//        for (int i = 0; i < num_neighbors_max_; i++)
//          if (neighbors_exist[i])
//            neighbors[i] < nodes_->num_owned_indeps ? d_nnz[2*n+1] += 2 : o_nnz[2*n+1] += 2;
//      } else {
//        preallocate_row(2*n+1, qnnn, d_nnz, o_nnz);
//        d_nnz[2*n+1] += 1;
//        for (int i = 0; i < num_neighbors_max_; i++)
//          if (neighbors_exist[i])
//            neighbors[i] < nodes_->num_owned_indeps ? d_nnz[2*n  ] += 2 : o_nnz[2*n  ] += 2;
//      }

      for (int i = 0; i < num_neighbors_max_; i++)
        if (neighbors_exist[i])
        {
          neighbors[i] < nodes_->num_owned_indeps ? d_nnz[2*n+1] += 2 : o_nnz[2*n+1] += 2;
          neighbors[i] < nodes_->num_owned_indeps ? d_nnz[2*n  ] += 2 : o_nnz[2*n  ] += 2;
        }


      if (use_sc_scheme_)
      {
        bool is_interface_m00 = phi_p[n]*phi_p[neighbors[nn_m00]] < 0;
        bool is_interface_p00 = phi_p[n]*phi_p[neighbors[nn_p00]] < 0;
        bool is_interface_0m0 = phi_p[n]*phi_p[neighbors[nn_0m0]] < 0;
        bool is_interface_0p0 = phi_p[n]*phi_p[neighbors[nn_0p0]] < 0;
#ifdef P4_TO_P8
        bool is_interface_00m = phi_p[n]*phi_p[neighbors[nn_00m]] < 0;
        bool is_interface_00p = phi_p[n]*phi_p[neighbors[nn_00p]] < 0;
#endif

        p4est_locidx_t neighbors_of_nei[num_neighbors_max_];
        bool neighbors_of_nei_exist[num_neighbors_max_];

        if (is_interface_m00) { get_all_neighbors_(neighbors[nn_m00], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_m00] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
        if (is_interface_p00) { get_all_neighbors_(neighbors[nn_p00], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_p00] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
        if (is_interface_0m0) { get_all_neighbors_(neighbors[nn_0m0], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_0m0] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
        if (is_interface_0p0) { get_all_neighbors_(neighbors[nn_0p0], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_0p0] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
#ifdef P4_TO_P8
        if (is_interface_00m) { get_all_neighbors_(neighbors[nn_00m], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_00m] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
        if (is_interface_00p) { get_all_neighbors_(neighbors[nn_00p], neighbors_of_nei, neighbors_of_nei_exist); if (neighbors_of_nei[nn_00p] < nodes_->num_owned_indeps) { d_nnz[2*n+1] += 1; d_nnz[2*n] += 1; } else { o_nnz[2*n+1] += 1; o_nnz[2*n] += 1; } }
#endif
      }

    }
  }

  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_eff_, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_nodes_mls_sc_t::solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_solve, A_, rhs_block_, ksp_, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if (bc_wall_type_ == NULL || bc_wall_value_ == NULL)
    throw std::domain_error("[CASL_ERROR]: the boundary conditions on walls have not been set.");

  if (num_interfaces_ > 0)
    if (mu_un_jump_ == NULL || u_jump_ == NULL || mu_un_jump_->size() != num_interfaces_)
      throw std::domain_error("[CASL_ERROR]: the boundary conditions on interfaces have not been set.");

  {
    PetscInt sol_m_size;   ierr = VecGetLocalSize(sol_m, &sol_m_size); CHKERRXX(ierr);
    PetscInt sol_p_size;   ierr = VecGetLocalSize(sol_p, &sol_p_size); CHKERRXX(ierr);

    if (sol_m_size != nodes_->num_owned_indeps || sol_p_size != nodes_->num_owned_indeps)
    {
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "sol_m.local_size = " << sol_m_size
          << "sol_p.local_size = " << sol_p_size
          << " nodes->num_owned_indeps = " << nodes_->num_owned_indeps << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set local add if none was given
  bool local_add_m = false;
  if(diag_add_m_ == NULL)
  {
    local_add_m = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &diag_add_m_); CHKERRXX(ierr);
    ierr = VecSet(diag_add_m_, diag_add_scalar_m_); CHKERRXX(ierr);
  }

  bool local_add_p = false;
  if(diag_add_p_ == NULL)
  {
    local_add_p = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &diag_add_p_); CHKERRXX(ierr);
    ierr = VecSet(diag_add_p_, diag_add_scalar_p_); CHKERRXX(ierr);
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

    ierr = VecDuplicate(sol_m, &phi_->at(0)); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(phi_->at(0), &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_->at(0), &tmp); CHKERRXX(ierr);
    set_geometry(1, action_, color_, phi_);
  }

  bool local_rhs_m = false;
  if (rhs_m_ == NULL)
  {
    ierr = VecDuplicate(sol_m, &rhs_m_); CHKERRXX(ierr);
    Vec rhs_local;
    VecGhostGetLocalForm(rhs_m_, &rhs_local);
    VecSet(rhs_local, 0);
    VecGhostRestoreLocalForm(rhs_m_, &rhs_local);
    local_rhs_m = true;
  }

  bool local_rhs_p = false;
  if (rhs_p_ == NULL)
  {
    ierr = VecDuplicate(sol_p, &rhs_p_); CHKERRXX(ierr);
    Vec rhs_local;
    VecGhostGetLocalForm(rhs_p_, &rhs_local);
    VecSet(rhs_local, 0);
    VecGhostRestoreLocalForm(rhs_p_, &rhs_local);
    local_rhs_p = true;
  }


  if (rhs_block_ != NULL) { ierr = VecDestroy(rhs_block_); CHKERRXX(ierr); }

  ierr = VecCreateGhostNodesBlock(p4est_, nodes_, 2, &rhs_block_);

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
    setup_linear_system_(true, false);
    is_matrix_computed_ = true;
    new_pc_ = true;
  }

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
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "CLJP"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (matrix_has_nullspace_){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }

  if (!strcmp(pc_type, PCGAMG))
  {
//    PCASMSetOverlap(pc, 5);
//    PCASMSetType(pc, PC_ASM_NONE);
    ierr = PetscOptionsSetValue("-pc_gamg_sym_graph", "true"); CHKERRXX(ierr);
  }

  if (!strcmp(pc_type, PCASM))
  {
//    ierr = PetscOptionsSetValue("-pc_asm_blocks", "1000"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_overlap", "0"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_type", "restrict"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_local_type", "additive"); CHKERRXX(ierr);

    ierr = PetscOptionsSetValue("-sub_pc_type", "ilu"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_pc_factor_levels", "3"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_ksp_type", "preonly"); CHKERRXX(ierr);
  }

  if (!strcmp(pc_type, PCGASM))
  {
//    ierr = PetscOptionsSetValue("-pc_asm_blocks", "100"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_overlap", "0"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_type", "restrict"); CHKERRXX(ierr);
//    ierr = PetscOptionsSetValue("-pc_asm_local_type", "additive"); CHKERRXX(ierr);

    ierr = PetscOptionsSetValue("-sub_pc_type", "ilu"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_pc_factor_levels", "5"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_ksp_type", "preonly"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_pc_factor_diagonal_fill", "true"); CHKERRXX(ierr);
    ierr = PetscOptionsSetValue("-sub_pc_factor_nonzeros_along_diagonal", "true"); CHKERRXX(ierr);
  }


//  if (!strcmp(ksp_type, KSPGMRES))
//  {
//    KSPGMRESSetRestart(ksp_, 30);
//    KSPGMRESModifiedGramSchmidtOrthogonalization(ksp_, 29);
//  }

  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
//  setup_negative_variable_coeff_laplace_rhsvec_();
  setup_linear_system_(false, true);

  // Solve the system
  Vec sol_block; double *sol_block_ptr; ierr = VecCreateGhostNodesBlock(p4est_, nodes_, 2, &sol_block);

  ierr = KSPSetTolerances(ksp_, 1e-15, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp_); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_KSPSolve, ksp_, rhs_block_, sol_block, 0); CHKERRXX(ierr);
  MatNullSpace A_null;
  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceCreate(p4est_->mpicomm, PETSC_TRUE, 0, NULL, &A_null); CHKERRXX(ierr);
    ierr = MatSetNullSpace(A_, A_null);

    // For purely neumann problems GMRES is more robust
    ierr = KSPSetType(ksp_, KSPGMRES); CHKERRXX(ierr);
  }


  ierr = KSPSolve(ksp_, rhs_block_, sol_block); CHKERRXX(ierr);

  if (matrix_has_nullspace_) {
    ierr = MatNullSpaceDestroy(A_null);
  }

  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_mls_sc_KSPSolve, ksp_, rhs_block_, sol_block, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(sol_block, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(sol_block, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // split block solution vector into normal vectors
  double *sol_m_ptr;
  double *sol_p_ptr;

  ierr = VecGetArray(sol_block, &sol_block_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(sol_m,     &sol_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(sol_p,     &sol_p_ptr); CHKERRXX(ierr);

  for (size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
  {
    sol_m_ptr[n] = sol_block_ptr[2*n  ];
    sol_p_ptr[n] = sol_block_ptr[2*n+1];
  }

  ierr = VecRestoreArray(sol_block, &sol_block_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_m,     &sol_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_p,     &sol_p_ptr); CHKERRXX(ierr);

  // get rid of local stuff
  if (local_add_m) { ierr = VecDestroy(diag_add_m_); CHKERRXX(ierr); diag_add_m_ = NULL; }
  if (local_add_p) { ierr = VecDestroy(diag_add_p_); CHKERRXX(ierr); diag_add_p_ = NULL; }

  if (local_rhs_m) { ierr = VecDestroy(rhs_m_); CHKERRXX(ierr); rhs_m_ = NULL; }
  if (local_rhs_p) { ierr = VecDestroy(rhs_p_); CHKERRXX(ierr); rhs_p_ = NULL; }

  ierr = VecDestroy(sol_block); CHKERRXX(ierr);

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

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_solve, A_, rhs_block_, ksp_, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_nodes_mls_sc_t::setup_linear_system_(bool setup_matrix, bool setup_rhs)
{
  if (!setup_matrix && !setup_rhs)
    throw std::invalid_argument("[CASL_ERROR]: If you aren't assembling either matrix or RHS, what the heck then are you trying to do? lol :)");

  if (setup_matrix)
  {
    compute_volumes_();
    preallocate_matrix();
  }

  // register for logging purpose
  // not sure if we need to register both in case we're assembling matrix and rhs simultaneously
  if (setup_matrix)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup, rhs_block_, 0, 0, 0); CHKERRXX(ierr);
  }


//#ifdef P4_TO_P8
//  double eps = 1E-6*d_min*d_min*d_min;
//#else
  double eps = 1E-6*d_min_*d_min_;
//#endif

  double domain_rel_thresh = 1.e-13;
  double interface_rel_thresh = 1.e-13;//0*eps;
//  double interface_rel_thresh = eps_ifc_;//0*eps;

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

  double *mue_m_ptr=NULL, *mue_m_xx_ptr=NULL, *mue_m_yy_ptr=NULL, *mue_m_zz_ptr=NULL;
  double *mue_p_ptr=NULL, *mue_p_xx_ptr=NULL, *mue_p_yy_ptr=NULL, *mue_p_zz_ptr=NULL;

  if (variable_mu_)
  {
    ierr = VecGetArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr);
#endif
  }

  double *diag_add_m_ptr;
  double *diag_add_p_ptr;

  ierr = VecGetArray(diag_add_m_, &diag_add_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(diag_add_p_, &diag_add_p_ptr); CHKERRXX(ierr);

  double *rhs_m_ptr;
  double *rhs_p_ptr;
  double *rhs_block_ptr;

  if (setup_rhs)
  {
    ierr = VecGetArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_block_, &rhs_block_ptr); CHKERRXX(ierr);
  }

  std::vector<double> phi_000(num_interfaces_, -1),
      phi_p00(num_interfaces_, 0),
      phi_m00(num_interfaces_, 0),
      phi_0m0(num_interfaces_, 0),
      phi_0p0(num_interfaces_, 0);
#ifdef P4_TO_P8
  std::vector<double> phi_00m(num_interfaces_, 0), phi_00p(num_interfaces_, 0);
#endif

  double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;
#ifdef P4_TO_P8
  double mue_00m, mue_00p;
#endif

//  double mue_m_000, mue_m_p00, mue_m_m00, mue_m_0m0, mue_m_0p0;
//  double mue_p_000, mue_p_p00, mue_p_m00, mue_p_0m0, mue_p_0p0;
//#ifdef P4_TO_P8
//  double mue_m_00m, mue_m_00p;
//  double mue_p_00m, mue_p_00p;
//#endif

  double *mask_m_ptr;
  double *mask_p_ptr;

  if (setup_matrix)
  {
//    compute_volumes_();

    if (mask_m_ != NULL) { ierr = VecDestroy(mask_m_); CHKERRXX(ierr); }
    if (mask_p_ != NULL) { ierr = VecDestroy(mask_p_); CHKERRXX(ierr); }

    ierr = VecCreateGhostNodes(p4est_, nodes_, &mask_m_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_, nodes_, &mask_p_); CHKERRXX(ierr);
  }

  ierr = VecGetArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);

  double *volumes_p;
  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  if (!variable_mu_)
  {
    mue_000 = mu_m_; mue_p00 = mu_m_; mue_m00 = mu_m_; mue_0m0 = mu_m_; mue_0p0 = mu_m_;
#ifdef P4_TO_P8
    mue_00m = mu_m_; mue_00p = mu_m_;
#endif
//    mue_m_000 = mu_m_; mue_m_p00 = mu_m_; mue_m_m00 = mu_m_; mue_m_0m0 = mu_m_; mue_m_0p0 = mu_m_;
//    mue_p_000 = mu_p_; mue_p_p00 = mu_p_; mue_p_m00 = mu_p_; mue_p_0m0 = mu_p_; mue_p_0p0 = mu_p_;
//#ifdef P4_TO_P8
//    mue_m_00m = mu_m_; mue_m_00p = mu_m_;
//    mue_p_00m = mu_p_; mue_p_00p = mu_p_;
//#endif
  }

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
  std::vector<double> phi_fv(pow(2*cube_refinement_+1,P4EST_DIM),-1);
  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement_+1, 0);
  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement_+1, 0);
#ifdef P4_TO_P8
  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement_+1, 0);
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  double full_cell_volume;

  bool neighbors_exist_m[num_neighbors_max_];
  bool neighbors_exist_p[num_neighbors_max_];
  p4est_locidx_t neighbors[num_neighbors_max_];

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local_m(node_neighbors_);
  my_p4est_interpolation_nodes_local_t interp_local_p(node_neighbors_);
  std::vector<my_p4est_interpolation_nodes_local_t *> phi_interp_local(num_interfaces_, NULL);

  if (variable_mu_)
  {
#ifdef P4_TO_P8
    interp_local_m.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, mue_m_zz_ptr, quadratic);
    interp_local_p.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, mue_p_zz_ptr, quadratic);
#else
    interp_local_m.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, quadratic);
    interp_local_p.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, quadratic);
#endif
  }

  for (int i = 0; i < num_interfaces_; ++i)
  {
    phi_interp_local[i] = new my_p4est_interpolation_nodes_local_t (node_neighbors_);
#ifdef P4_TO_P8
    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], phi_zz_p[i], quadratic);
#else
    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], quadratic);
#endif
  }

#ifdef P4_TO_P8
  std::vector<CF_3 *> phi_interp_local_cf(num_interfaces_, NULL);
#else
  std::vector<CF_2 *> phi_interp_local_cf(num_interfaces_, NULL);
#endif
  for (int i = 0; i < num_interfaces_; ++i)
    phi_interp_local_cf[i] = phi_interp_local[i];

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

//  double val_interface_m00 = 0.;
//  double val_interface_p00 = 0.;
//  double val_interface_0m0 = 0.;
//  double val_interface_0p0 = 0.;
//#ifdef P4_TO_P8
//  double val_interface_00m = 0.;
//  double val_interface_00p = 0.;
//#endif

  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
  {
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

      mask_m_ptr[n] = 1;
      mask_p_ptr[n] = 1;
    }

    interp_local_m.initialize(n);
    interp_local_p.copy_init(interp_local_m);

    for (short i = 0; i < num_interfaces_; ++i)
      phi_interp_local[i]->copy_init(interp_local_m);



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

    if(is_node_Wall(p4est_, ni))
    {
#ifdef P4_TO_P8
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
#else
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
#endif
      {
        if (setup_matrix)
        {
          ierr = MatSetValue(A_, 2*node_000_g    , 2*node_000_g    , 1.0, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A_, 2*node_000_g + 1, 2*node_000_g + 1, 1.0, ADD_VALUES); CHKERRXX(ierr);

          if (phi_eff_p[n] < 0.)
            mask_m_ptr[n] = -1;
          else
            mask_p_ptr[n] = -1;

          matrix_has_nullspace_ = false;
        }

        if (setup_rhs)
        {
          rhs_block_ptr[2*n  ] = (phi_eff_000 < 0.) ? bc_wall_value_->value(xyz_C) : 0;
          rhs_block_ptr[2*n+1] = (phi_eff_000 > 0.) ? bc_wall_value_->value(xyz_C) : 0;
        }

        continue;
      }

//      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
//      if(neumann_wall_first_order_ &&
//   #ifdef P4_TO_P8
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == NEUMANN)
//   #else
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1]) == NEUMANN)
//   #endif
//      {
//        if (is_node_xpWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//#ifdef P4_TO_P8
//            p4est_locidx_t n_m00 = d_m00_0m == 0 ? ( d_m00_m0==0 ? node_m00_mm : node_m00_pm )
//                                                 : ( d_m00_m0==0 ? node_m00_mp : node_m00_pp );
//#else
//            p4est_locidx_t n_m00 = d_m00_m0 == 0 ? node_m00_mm : node_m00_pm;
//#endif
//            PetscInt node_m00_g  = petsc_gloidx_[n_m00];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_m00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_m00;
//          continue;
//        }

//        if (is_node_xmWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//#ifdef P4_TO_P8
//            p4est_locidx_t n_p00 = d_p00_0m == 0 ? ( d_p00_m0 == 0 ? node_p00_mm : node_p00_pm )
//                                                 : ( d_p00_m0 == 0 ? node_p00_mp : node_p00_pp );
//#else
//            p4est_locidx_t n_p00 = d_p00_m0 == 0 ? node_p00_mm : node_p00_pm;
//#endif
//            PetscInt node_p00_g  = petsc_gloidx_[n_p00];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_p00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_p00;
//          continue;
//        }

//        if (is_node_ypWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//#ifdef P4_TO_P8
//            p4est_locidx_t n_0m0 = d_0m0_0m == 0 ? ( d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm )
//                                                 : ( d_0m0_m0 == 0 ? node_0m0_mp : node_0m0_pp );
//#else
//            p4est_locidx_t n_0m0 = d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm;
//#endif
//            PetscInt node_0m0_g  = petsc_gloidx_[n_0m0];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_0m0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0m0;
//          continue;
//        }
//        if (is_node_ymWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//#ifdef P4_TO_P8
//            p4est_locidx_t n_0p0 = d_0p0_0m == 0 ? ( d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm )
//                                                 : ( d_0p0_m0 == 0 ? node_0p0_mp : node_0p0_pp );
//#else
//            p4est_locidx_t n_0p0 = d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm;
//#endif
//            PetscInt node_0p0_g  = petsc_gloidx_[n_0p0];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_0p0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0p0;
//          continue;
//        }
//#ifdef P4_TO_P8
//        if (is_node_zpWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//            p4est_locidx_t n_00m = d_00m_0m == 0 ? ( d_00m_m0 == 0 ? node_00m_mm : node_00m_pm )
//                                                 : ( d_00m_m0 == 0 ? node_00m_mp : node_00m_pp );
//            PetscInt node_00m_g  = petsc_gloidx_[n_00m];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_00m_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00m;
//          continue;
//        }

//        if (is_node_zmWall(p4est_, ni)){
//          if (setup_matrix)
//          {
//            p4est_locidx_t n_00p = d_00p_0m == 0 ? ( d_00p_m0 == 0 ? node_00p_mm : node_00p_pm )
//                                                 : ( d_00p_m0 == 0 ? node_00p_mp : node_00p_pp );
//            PetscInt node_00p_g  = petsc_gloidx_[n_00p];

//            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

//            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//            {
//              mask_p[n] = -1;
//              ierr = MatSetValue(A_, node_000_g, node_00p_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
//            }
//          }

//          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00p;
//          continue;
//        }
//#endif
//      }

    }

    if (volumes_p[n] < 1.-eps_dom_ && volumes_p[n] > eps_dom_)
    {
      mask_m_ptr[n] = -1;
      mask_p_ptr[n] = -1;

      double wr_m00 = 0;
      double wr_p00 = 0;
      double wr_0m0 = 0;
      double wr_0p0 = 0;
#ifdef P4_TO_P8
      double wr_00m = 0;
      double wr_00p = 0;
#endif
      double wr_000 = 0;

      double wg_000 = 0;
      double wg_m00 = 0;
      double wg_p00 = 0;
      double wg_0m0 = 0;
      double wg_0p0 = 0;
#ifdef P4_TO_P8
      double wg_00m = 0;
      double wg_00p = 0;
#endif

      // second order neighbors
      double wg_M00 = 0;
      double wg_P00 = 0;
      double wg_0M0 = 0;
      double wg_0P0 = 0;
#ifdef P4_TO_P8
      double wg_00M = 0;
      double wg_00P = 0;
#endif

      double rhs_fd = 0;

      double volume_cut_cell = 0.;
      double interface_area  = 0.;
      double integral_bc = 0.;

      // Reconstruct geometry
      // determine dimensions of cube
      fv_size_x = 0;
      fv_size_y = 0;
#ifdef P4_TO_P8
      fv_size_z = 0;
#endif
      if(!is_node_xmWall(p4est_, ni)) {fv_size_x += cube_refinement_; fv_xmin = x_C-0.5*dx_min_;} else {fv_xmin = x_C;}
      if(!is_node_xpWall(p4est_, ni)) {fv_size_x += cube_refinement_; fv_xmax = x_C+0.5*dx_min_;} else {fv_xmax = x_C;}

      if(!is_node_ymWall(p4est_, ni)) {fv_size_y += cube_refinement_; fv_ymin = y_C-0.5*dy_min_;} else {fv_ymin = y_C;}
      if(!is_node_ypWall(p4est_, ni)) {fv_size_y += cube_refinement_; fv_ymax = y_C+0.5*dy_min_;} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
      if(!is_node_zmWall(p4est_, ni)) {fv_size_z += cube_refinement_; fv_zmin = z_C-0.5*dz_min_;} else {fv_zmin = z_C;}
      if(!is_node_zpWall(p4est_, ni)) {fv_size_z += cube_refinement_; fv_zmax = z_C+0.5*dz_min_;} else {fv_zmax = z_C;}
#endif

#ifdef P4_TO_P8
      full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin)*(fv_zmax-fv_zmin);
#else
      full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin);
#endif
      if (!use_refined_cube_) {
        fv_size_x = 1;
        fv_size_y = 1;
#ifdef P4_TO_P8
        fv_size_z = 1;
#endif
      }

      fv_nx = fv_size_x+1;
      fv_ny = fv_size_y+1;
#ifdef P4_TO_P8
      fv_nz = fv_size_z+1;
#endif

      // get coordinates of cube nodes
      double fv_dx = (fv_xmax-fv_xmin) / (double)(fv_size_x);
      fv_x[0] = fv_xmin;
      for (short i = 1; i < fv_nx; ++i)
        fv_x[i] = fv_x[i-1] + fv_dx;

      double fv_dy = (fv_ymax-fv_ymin) / (double)(fv_size_y);
      fv_y[0] = fv_ymin;
      for (short i = 1; i < fv_ny; ++i)
        fv_y[i] = fv_y[i-1] + fv_dy;
#ifdef P4_TO_P8
      double fv_dz = (fv_zmax-fv_zmin) / (double)(fv_size_z);
      fv_z[0] = fv_zmin;
      for (short i = 1; i < fv_nz; ++i)
        fv_z[i] = fv_z[i-1] + fv_dz;
#endif

#ifdef USE_QUADRATIC_CUBES
#ifdef P4_TO_P8
      std::vector<cube3_mls_quadratic_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
#else
      std::vector<cube2_mls_quadratic_t *> cubes(fv_size_x*fv_size_y, NULL);
#endif
#else
#ifdef P4_TO_P8
      std::vector<cube3_mls_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
#else
      std::vector<cube2_mls_t *> cubes(fv_size_x*fv_size_y, NULL);
#endif
#endif

#ifdef P4_TO_P8
      for (short k = 0; k < fv_size_z; ++k)
#endif
        for (short j = 0; j < fv_size_y; ++j)
          for (short i = 0; i < fv_size_x; ++i)
          {
#ifdef USE_QUADRATIC_CUBES
#ifdef P4_TO_P8
            int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
            cubes[idx] = new cube3_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
#else
            int idx = j*fv_size_x + i;
            cubes[idx] = new cube2_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
#endif
#else
#ifdef P4_TO_P8
            int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
            cubes[idx] = new cube3_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
#else
            int idx = j*fv_size_x + i;
            cubes[idx] = new cube2_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
#endif
#endif

            cubes[idx]->construct_domain(phi_interp_local_cf, *action_, *color_);

            volume_cut_cell += cubes[idx]->integrate_over_domain(unity_cf_);
            //            interface_area  += cubes[idx]->integrate_over_interface(unity_cf_, -1);

            if (setup_rhs)
            {
              for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
                integral_bc += cubes[idx]->integrate_over_interface(*mu_un_jump_->at(phi_idx), color_->at(phi_idx));
            }
          }

      double volume_cut_cell_m = volume_cut_cell;
      double volume_cut_cell_p = full_cell_volume - volume_cut_cell;

      // Compute areas (lengths) of faces of the finite volume
      delta_x_cf_.set(xyz_C);
      delta_y_cf_.set(xyz_C);
#ifdef P4_TO_P8
      delta_z_cf_.set(xyz_C);
#endif

#ifdef P4_TO_P8
      double full_sx = (fv_ymax - fv_ymin)*(fv_zmax - fv_zmin);
      double full_sy = (fv_xmax - fv_xmin)*(fv_zmax - fv_zmin);
      double full_sz = (fv_xmax - fv_xmin)*(fv_ymax - fv_ymin);
#else
      double full_sx = fv_ymax - fv_ymin;
      double full_sy = fv_xmax - fv_xmin;
#endif

      double s_m00_m = 0, s_p00_m = 0;
      double y_m00_m = 0, y_p00_m = 0;
#ifdef P4_TO_P8
      double z_m00_m = 0, z_p00_m = 0;
      for (short k = 0; k < fv_size_z; ++k)
#endif
        for (short j = 0; j < fv_size_y; ++j)
        {
          int i_m = 0;
          int i_p = fv_size_x-1.;
#ifdef P4_TO_P8
          int idx_m = k*fv_size_x*fv_size_y + j*fv_size_x + i_m;
          int idx_p = k*fv_size_x*fv_size_y + j*fv_size_x + i_p;
#else
          int idx_m = j*fv_size_x + i_m;
          int idx_p = j*fv_size_x + i_p;
#endif
          s_m00_m += cubes[idx_m]->integrate_in_dir(unity_cf_, 0);
          s_p00_m += cubes[idx_p]->integrate_in_dir(unity_cf_, 1);

          y_m00_m += cubes[idx_m]->integrate_in_dir(delta_y_cf_, 0);
          y_p00_m += cubes[idx_p]->integrate_in_dir(delta_y_cf_, 1);
#ifdef P4_TO_P8
          z_m00_m += cubes[idx_m]->integrate_in_dir(delta_z_cf_, 0);
          z_p00_m += cubes[idx_p]->integrate_in_dir(delta_z_cf_, 1);
#endif
        }

      if (s_m00_m/full_sx > eps_ifc_) y_m00_m /= s_m00_m; else y_m00_m = 0;
      if (s_p00_m/full_sx > eps_ifc_) y_p00_m /= s_p00_m; else y_p00_m = 0;
#ifdef P4_TO_P8
      if (s_m00_m/full_sx > eps_ifc_) z_m00_m /= s_m00_m; else z_m00_m = 0;
      if (s_p00_m/full_sx > eps_ifc_) z_p00_m /= s_p00_m; else z_p00_m = 0;
#endif

      double s_m00_p = full_sx - s_m00_m;
      double s_p00_p = full_sx - s_p00_m;

      double y_m00_p = (s_m00_p/full_sx > eps_ifc_) ? - s_m00_m/s_m00_p * y_m00_m : 0;
      double y_p00_p = (s_p00_p/full_sx > eps_ifc_) ? - s_p00_m/s_p00_p * y_p00_m : 0;
#ifdef P4_TO_P8
      double z_m00_p = (s_m00_p/full_sx > eps_ifc_) ? - s_m00_m/s_m00_p * z_m00_m : 0;
      double z_p00_p = (s_p00_p/full_sx > eps_ifc_) ? - s_p00_m/s_p00_p * z_p00_m : 0;
#endif

      double s_0m0_m = 0, s_0p0_m = 0;
      double x_0m0_m = 0, x_0p0_m = 0;
#ifdef P4_TO_P8
      double z_0m0_m = 0, z_0p0_m = 0;
      for (short k = 0; k < fv_size_z; ++k)
#endif
        for (short i = 0; i < fv_size_x; ++i)
        {
          int j_m = 0;
          int j_p = fv_size_y-1.;
#ifdef P4_TO_P8
          int idx_m = k*fv_size_x*fv_size_y + j_m*fv_size_x + i;
          int idx_p = k*fv_size_x*fv_size_y + j_p*fv_size_x + i;
#else
          int idx_m = j_m*fv_size_x + i;
          int idx_p = j_p*fv_size_x + i;
#endif
          s_0m0_m += cubes[idx_m]->integrate_in_dir(unity_cf_, 2);
          s_0p0_m += cubes[idx_p]->integrate_in_dir(unity_cf_, 3);

          x_0m0_m += cubes[idx_m]->integrate_in_dir(delta_x_cf_, 2);
          x_0p0_m += cubes[idx_p]->integrate_in_dir(delta_x_cf_, 3);
#ifdef P4_TO_P8
          z_0m0_m += cubes[idx_m]->integrate_in_dir(delta_z_cf_, 2);
          z_0p0_m += cubes[idx_p]->integrate_in_dir(delta_z_cf_, 3);
#endif
        }

      if (s_0m0_m/full_sy > eps_ifc_) x_0m0_m /= s_0m0_m; else x_0m0_m = 0;
      if (s_0p0_m/full_sy > eps_ifc_) x_0p0_m /= s_0p0_m; else x_0p0_m = 0;
#ifdef P4_TO_P8
      if (s_0m0_m/full_sy > eps_ifc_) z_0m0_m /= s_0m0_m; else z_0m0_m = 0;
      if (s_0p0_m/full_sy > eps_ifc_) z_0p0_m /= s_0p0_m; else z_0p0_m = 0;
#endif

      double s_0m0_p = full_sy - s_0m0_m;
      double s_0p0_p = full_sy - s_0p0_m;

      double x_0m0_p = (s_0m0_p/full_sy > eps_ifc_) ? - s_0m0_m/s_0m0_p * x_0m0_m : 0;
      double x_0p0_p = (s_0p0_p/full_sy > eps_ifc_) ? - s_0p0_m/s_0p0_p * x_0p0_m : 0;
#ifdef P4_TO_P8
      double z_0m0_p = (s_0m0_p/full_sy > eps_ifc_) ? - s_0m0_m/s_0m0_p * z_0m0_m : 0;
      double z_0p0_p = (s_0p0_p/full_sy > eps_ifc_) ? - s_0p0_m/s_0p0_p * z_0p0_m : 0;
#endif

#ifdef P4_TO_P8
      double s_00m_m = 0, s_00p_m = 0;
      double x_00m_m = 0, x_00p_m = 0;
      double y_00m_m = 0, y_00p_m = 0;
      for (short j = 0; j < fv_size_y; ++j)
        for (short i = 0; i < fv_size_x; ++i)
        {
          int k_m = 0;
          int k_p = fv_size_z-1.;

          int idx_m = k_m*fv_size_x*fv_size_y + j*fv_size_x + i;
          int idx_p = k_p*fv_size_x*fv_size_y + j*fv_size_x + i;

          s_00m_m += cubes[idx_m]->integrate_in_dir(unity_cf_, 4);
          s_00p_m += cubes[idx_p]->integrate_in_dir(unity_cf_, 5);

          x_00m_m += cubes[idx_m]->integrate_in_dir(delta_x_cf_, 4);
          x_00p_m += cubes[idx_p]->integrate_in_dir(delta_x_cf_, 5);

          y_00m_m += cubes[idx_m]->integrate_in_dir(delta_y_cf_, 4);
          y_00p_m += cubes[idx_p]->integrate_in_dir(delta_y_cf_, 5);
        }

      if (s_00m_m/full_sz > eps_ifc_) x_00m_m /= s_00m_m; else x_00m_m = 0;
      if (s_00p_m/full_sz > eps_ifc_) x_00p_m /= s_00p_m; else x_00p_m = 0;
      if (s_00m_m/full_sz > eps_ifc_) y_00m_m /= s_00m_m; else y_00m_m = 0;
      if (s_00p_m/full_sz > eps_ifc_) y_00p_m /= s_00p_m; else y_00p_m = 0;

      double s_00m_p = full_sz - s_00m_m;
      double s_00p_p = full_sz - s_00p_m;

      double x_00m_p = (s_00m_p/full_sz > eps_ifc_) ? - s_00m_m/s_00m_p * x_00m_m : 0;
      double x_00p_p = (s_00p_p/full_sz > eps_ifc_) ? - s_00p_m/s_00p_p * x_00p_m : 0;
      double y_00m_p = (s_00m_p/full_sz > eps_ifc_) ? - s_00m_m/s_00m_p * y_00m_m : 0;
      double y_00p_p = (s_00p_p/full_sz > eps_ifc_) ? - s_00p_m/s_00p_p * y_00p_m : 0;
#endif

      // second equation is from FD discretization
      if (0)
      {
        int shift   = phi_eff_000 <= 0 ? 1 : 0;
//        int shift   = phi_eff_000 <= 0 ? 0 : 1;

        int shift_r = phi_eff_000 <= 0 ? 1 : 0;
//        int shift_r = phi_eff_000 <= 0 ? 0 : 1;

        if (setup_matrix)
        {
          ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_000_g + shift_r, 1.,  ADD_VALUES); CHKERRXX(ierr);
        }

        if (setup_rhs)
        {
          rhs_block_ptr[2*n+shift] = bc_wall_value_->value(xyz_C);
        }
      }
      else
      {
        double *mue_ptr = phi_eff_000 <= 0 ? mue_m_ptr : mue_p_ptr;
        double mu = phi_eff_000 <= 0 ? mu_m_ : mu_p_;

        get_all_neighbors_(n, neighbors, neighbors_exist_m);

        if (variable_mu_)
        {
  #ifdef P4_TO_P8
            qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
  #else
            qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
  #endif
        } else {
            mue_000 = mu;
            mue_m00 = mu; mue_p00 = mu;
            mue_0m0 = mu; mue_0p0 = mu;
  #ifdef P4_TO_P8
            mue_00m = mu; mue_00p = mu;
  #endif
        }

        // find interface location
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

//        bool more_than_half_face_m00 = phi_eff_000 < 0 ? (s_m00_p/full_sx > .5) : (s_m00_m/full_sx > .5);
//        bool more_than_half_face_p00 = phi_eff_000 < 0 ? (s_p00_p/full_sx > .5) : (s_p00_m/full_sx > .5);

//        bool more_than_half_face_0m0 = phi_eff_000 < 0 ? (s_0m0_p/full_sy > .5) : (s_0m0_m/full_sy > .5);
//        bool more_than_half_face_0p0 = phi_eff_000 < 0 ? (s_0p0_p/full_sy > .5) : (s_0p0_m/full_sy > .5);
//#ifdef P4_TO_P8
//        bool more_than_half_face_00m = phi_eff_000 < 0 ? (s_00m_p/full_sz > .5) : (s_00m_m/full_sz > .5);
//        bool more_than_half_face_00p = phi_eff_000 < 0 ? (s_00p_p/full_sz > .5) : (s_00p_m/full_sz > .5);
//#endif

//        && more_than_half_face_m00
//        && more_than_half_face_p00
//        && more_than_half_face_0m0
//        && more_than_half_face_0p0

//        && more_than_half_face_00m
//        && more_than_half_face_00p

        if (phi_eff_000*phi_eff_p[neighbors[nn_m00]] < 0) { is_interface_m00 = true; theta_m00 = find_interface_location_mls(n, neighbors[nn_m00], d_m00, phi_p, phi_xx_p); if (theta_m00 < eps) theta_m00 = eps; }
        if (phi_eff_000*phi_eff_p[neighbors[nn_p00]] < 0) { is_interface_p00 = true; theta_p00 = find_interface_location_mls(n, neighbors[nn_p00], d_p00, phi_p, phi_xx_p); if (theta_p00 < eps) theta_p00 = eps; }
        if (phi_eff_000*phi_eff_p[neighbors[nn_0m0]] < 0) { is_interface_0m0 = true; theta_0m0 = find_interface_location_mls(n, neighbors[nn_0m0], d_0m0, phi_p, phi_yy_p); if (theta_0m0 < eps) theta_0m0 = eps; }
        if (phi_eff_000*phi_eff_p[neighbors[nn_0p0]] < 0) { is_interface_0p0 = true; theta_0p0 = find_interface_location_mls(n, neighbors[nn_0p0], d_0p0, phi_p, phi_yy_p); if (theta_0p0 < eps) theta_0p0 = eps; }
#ifdef P4_TO_P8
        if (phi_eff_000*phi_eff_p[neighbors[nn_00m]] < 0) { is_interface_00m = true; theta_00m = find_interface_location_mls(n, neighbors[nn_00m], d_00m, phi_p, phi_zz_p); if (theta_00m < eps) theta_00m = eps; }
        if (phi_eff_000*phi_eff_p[neighbors[nn_00p]] < 0) { is_interface_00p = true; theta_00p = find_interface_location_mls(n, neighbors[nn_00p], d_00p, phi_p, phi_zz_p); if (theta_00p < eps) theta_00p = eps; }
#endif

        double diag_add_value = phi_eff_000 < 0 ? diag_add_m_ptr[n] : diag_add_p_ptr[n];

#ifdef P4_TO_P8
        if (is_interface_m00) { phi_eff_000 < 0 ? mue_m00 = interp_local_m.interpolate(x_C - theta_m00, y_C, z_C) : mue_m00 = interp_local_p.interpolate(x_C - theta_m00, y_C, z_C); }
        if (is_interface_p00) { phi_eff_000 < 0 ? mue_p00 = interp_local_m.interpolate(x_C + theta_p00, y_C, z_C) : mue_p00 = interp_local_p.interpolate(x_C + theta_p00, y_C, z_C); }
        if (is_interface_0m0) { phi_eff_000 < 0 ? mue_0m0 = interp_local_m.interpolate(x_C, y_C - theta_0m0, z_C) : mue_0m0 = interp_local_p.interpolate(x_C, y_C - theta_0m0, z_C); }
        if (is_interface_0p0) { phi_eff_000 < 0 ? mue_0p0 = interp_local_m.interpolate(x_C, y_C + theta_0p0, z_C) : mue_0p0 = interp_local_p.interpolate(x_C, y_C + theta_0p0, z_C); }
        if (is_interface_00m) { phi_eff_000 < 0 ? mue_00m = interp_local_m.interpolate(x_C, y_C, z_C - theta_00m) : mue_00m = interp_local_p.interpolate(x_C, y_C, z_C - theta_00m); }
        if (is_interface_00p) { phi_eff_000 < 0 ? mue_00p = interp_local_m.interpolate(x_C, y_C, z_C + theta_00p) : mue_00p = interp_local_p.interpolate(x_C, y_C, z_C + theta_00p); }
#else
        if (is_interface_m00) { phi_eff_000 < 0 ? mue_m00 = interp_local_m.interpolate(x_C - theta_m00, y_C) : mue_m00 = interp_local_p.interpolate(x_C - theta_m00, y_C); }
        if (is_interface_p00) { phi_eff_000 < 0 ? mue_p00 = interp_local_m.interpolate(x_C + theta_p00, y_C) : mue_p00 = interp_local_p.interpolate(x_C + theta_p00, y_C); }
        if (is_interface_0m0) { phi_eff_000 < 0 ? mue_0m0 = interp_local_m.interpolate(x_C, y_C - theta_0m0) : mue_0m0 = interp_local_p.interpolate(x_C, y_C - theta_0m0); }
        if (is_interface_0p0) { phi_eff_000 < 0 ? mue_0p0 = interp_local_m.interpolate(x_C, y_C + theta_0p0) : mue_0p0 = interp_local_p.interpolate(x_C, y_C + theta_0p0); }
#endif

        wr_m00 = -(mue_000+mue_m00)/theta_m00/(theta_m00+theta_p00);
        wr_p00 = -(mue_000+mue_p00)/theta_p00/(theta_m00+theta_p00);
        wr_0m0 = -(mue_000+mue_0m0)/theta_0m0/(theta_0m0+theta_0p0);
        wr_0p0 = -(mue_000+mue_0p0)/theta_0p0/(theta_0m0+theta_0p0);
#ifdef P4_TO_P8
        wr_00m = -(mue_000+mue_00m)/theta_00m/(theta_00m+theta_00p);
        wr_00p = -(mue_000+mue_00p)/theta_00p/(theta_00m+theta_00p);

        wr_000 = diag_add_value -(wr_m00 + wr_p00 + wr_0m0 + wr_0p0 + wr_00m + wr_00p);
#else
        wr_000 = diag_add_value -(wr_m00 + wr_p00 + wr_0m0 + wr_0p0);
#endif

        if (!use_sc_scheme_)
        {
          if (is_interface_m00) { wg_000 += wr_m00*(d_m00-theta_m00)/d_m00; wg_m00 += wr_m00*theta_m00/d_m00; }
          if (is_interface_p00) { wg_000 += wr_p00*(d_p00-theta_p00)/d_p00; wg_p00 += wr_p00*theta_p00/d_p00; }
          if (is_interface_0m0) { wg_000 += wr_0m0*(d_0m0-theta_0m0)/d_0m0; wg_0m0 += wr_0m0*theta_0m0/d_0m0; }
          if (is_interface_0p0) { wg_000 += wr_0p0*(d_0p0-theta_0p0)/d_0p0; wg_0p0 += wr_0p0*theta_0p0/d_0p0; }
#ifdef P4_TO_P8
          if (is_interface_00m) { wg_000 += wr_00m*(d_00m-theta_00m)/d_00m; wg_00m += wr_00m*theta_00m/d_00m; }
          if (is_interface_00p) { wg_000 += wr_00p*(d_00p-theta_00p)/d_00p; wg_00p += wr_00p*theta_00p/d_00p; }
#endif
        }
        else
        {
          if (is_interface_m00) { double a = .5*theta_m00/dx_min_; wg_000 += wr_m00*(1.-3.*a+2.*a*a); wg_m00 += wr_m00*4.*a*(1.-a); wg_M00 += wr_m00*a*(2.*a-1.); }
          if (is_interface_p00) { double a = .5*theta_p00/dx_min_; wg_000 += wr_p00*(1.-3.*a+2.*a*a); wg_p00 += wr_p00*4.*a*(1.-a); wg_P00 += wr_p00*a*(2.*a-1.); }
          if (is_interface_0m0) { double a = .5*theta_0m0/dy_min_; wg_000 += wr_0m0*(1.-3.*a+2.*a*a); wg_0m0 += wr_0m0*4.*a*(1.-a); wg_0M0 += wr_0m0*a*(2.*a-1.); }
          if (is_interface_0p0) { double a = .5*theta_0p0/dy_min_; wg_000 += wr_0p0*(1.-3.*a+2.*a*a); wg_0p0 += wr_0p0*4.*a*(1.-a); wg_0P0 += wr_0p0*a*(2.*a-1.); }
#ifdef P4_TO_P8
          if (is_interface_00m) { double a = .5*theta_00m/dy_min_; wg_000 += wr_00m*(1.-3.*a+2.*a*a); wg_00m += wr_00m*4.*a*(1.-a); wg_00M += wr_00m*a*(2.*a-1.); }
          if (is_interface_00p) { double a = .5*theta_00p/dy_min_; wg_000 += wr_00p*(1.-3.*a+2.*a*a); wg_00p += wr_00p*4.*a*(1.-a); wg_00P += wr_00p*a*(2.*a-1.); }
#endif
        }
//        if (is_interface_m00) { std::cout << (*phi_interp_local[0])(x_C - theta_m00, y_C) << std::endl; }

//        int shift   = phi_eff_000 <= 0 ? 1 : 0;
        int shift   = phi_eff_000 <= 0 ? 0 : 1;


        int shift_r = phi_eff_000 <= 0 ? 0 : 1;
        int shift_g = phi_eff_000 <= 0 ? 1 : 0;

//        int shift_r = phi_eff_000 <= 0 ? 1 : 0;
//        int shift_g = phi_eff_000 <= 0 ? 0 : 1;

        if (setup_matrix)
        {
          if (keep_scalling_) scalling_[2*n+shift_r] = wr_000;
          if (fabs(diag_add_value) > 0) matrix_has_nullspace_ = false;

          ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_000_g + shift_r, 1.,  ADD_VALUES); CHKERRXX(ierr);
//          ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_000_g + shift_g, 1.,  ADD_VALUES); CHKERRXX(ierr);

          // TODO: I think it needs a fix for the case when interface is close to domain boundary

          if (fabs(wr_m00/wr_000) > EPS && !is_interface_m00 && neighbors_exist_m[nn_m00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_m00]] + shift_r, wr_m00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wr_p00/wr_000) > EPS && !is_interface_p00 && neighbors_exist_m[nn_p00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_p00]] + shift_r, wr_p00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wr_0m0/wr_000) > EPS && !is_interface_0m0 && neighbors_exist_m[nn_0m0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0m0]] + shift_r, wr_0m0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wr_0p0/wr_000) > EPS && !is_interface_0p0 && neighbors_exist_m[nn_0p0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0p0]] + shift_r, wr_0p0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#ifdef P4_TO_P8
          if (fabs(wr_00m/wr_000) > EPS && !is_interface_00m && neighbors_exist_m[nn_00m]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00m]] + shift_r, wr_00m/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wr_00p/wr_000) > EPS && !is_interface_00p && neighbors_exist_m[nn_00p]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00p]] + shift_r, wr_00p/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#endif

          if (fabs(wg_000/wr_000) > EPS) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_000]] + shift_g, wg_000/wr_000,  ADD_VALUES); CHKERRXX(ierr); }

          if (fabs(wg_m00/wr_000) > EPS && is_interface_m00 && neighbors_exist_m[nn_m00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_m00]] + shift_g, wg_m00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wg_p00/wr_000) > EPS && is_interface_p00 && neighbors_exist_m[nn_p00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_p00]] + shift_g, wg_p00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wg_0m0/wr_000) > EPS && is_interface_0m0 && neighbors_exist_m[nn_0m0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0m0]] + shift_g, wg_0m0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wg_0p0/wr_000) > EPS && is_interface_0p0 && neighbors_exist_m[nn_0p0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0p0]] + shift_g, wg_0p0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#ifdef P4_TO_P8
          if (fabs(wg_00m/wr_000) > EPS && is_interface_00m && neighbors_exist_m[nn_00m]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00m]] + shift_g, wg_00m/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
          if (fabs(wg_00p/wr_000) > EPS && is_interface_00p && neighbors_exist_m[nn_00p]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00p]] + shift_g, wg_00p/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#endif


          if (use_sc_scheme_)
          {
            p4est_locidx_t neighbors_of_nei[num_neighbors_max_];
            bool neighbors_of_nei_exist[num_neighbors_max_];

            p4est_locidx_t node_M00, node_P00, node_0M0, node_0P0, node_00M, node_00P;

            if (is_interface_m00) { get_all_neighbors_(neighbors[nn_m00], neighbors_of_nei, neighbors_of_nei_exist); node_M00 = neighbors_of_nei[nn_m00]; }
            if (is_interface_p00) { get_all_neighbors_(neighbors[nn_p00], neighbors_of_nei, neighbors_of_nei_exist); node_P00 = neighbors_of_nei[nn_p00]; }
            if (is_interface_0m0) { get_all_neighbors_(neighbors[nn_0m0], neighbors_of_nei, neighbors_of_nei_exist); node_0M0 = neighbors_of_nei[nn_0m0]; }
            if (is_interface_0p0) { get_all_neighbors_(neighbors[nn_0p0], neighbors_of_nei, neighbors_of_nei_exist); node_0P0 = neighbors_of_nei[nn_0p0]; }
#ifdef P4_TO_P8
            if (is_interface_00m) { get_all_neighbors_(neighbors[nn_00m], neighbors_of_nei, neighbors_of_nei_exist); node_00M = neighbors_of_nei[nn_00m]; }
            if (is_interface_00p) { get_all_neighbors_(neighbors[nn_00p], neighbors_of_nei, neighbors_of_nei_exist); node_00P = neighbors_of_nei[nn_00p]; }
#endif


            if (fabs(wg_M00/wr_000) > EPS && is_interface_m00) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_M00] + shift_g, wg_M00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(wg_P00/wr_000) > EPS && is_interface_p00) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_P00] + shift_g, wg_P00/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(wg_0M0/wr_000) > EPS && is_interface_0m0) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0M0] + shift_g, wg_0M0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(wg_0P0/wr_000) > EPS && is_interface_0p0) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0P0] + shift_g, wg_0P0/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#ifdef P4_TO_P8
            if (fabs(wg_00M/wr_000) > EPS && is_interface_00m) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00M] + shift_g, wg_00M/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(wg_00P/wr_000) > EPS && is_interface_00p) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00P] + shift_g, wg_00P/wr_000,  ADD_VALUES); CHKERRXX(ierr); }
#endif
          }

//          if (fabs(wr_m00/wg_000) > EPS && !is_interface_m00 && neighbors_exist_m[nn_m00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_m00]] + shift_r, wr_m00/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wr_p00/wg_000) > EPS && !is_interface_p00 && neighbors_exist_m[nn_p00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_p00]] + shift_r, wr_p00/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wr_0m0/wg_000) > EPS && !is_interface_0m0 && neighbors_exist_m[nn_0m0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0m0]] + shift_r, wr_0m0/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wr_0p0/wg_000) > EPS && !is_interface_0p0 && neighbors_exist_m[nn_0p0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0p0]] + shift_r, wr_0p0/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//#ifdef P4_TO_P8
//          if (fabs(wr_00m/wg_000) > EPS && !is_interface_00m && neighbors_exist_m[nn_00m]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00m]] + shift_r, wr_00m/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wr_00p/wg_000) > EPS && !is_interface_00p && neighbors_exist_m[nn_00p]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00p]] + shift_r, wr_00p/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//#endif

//          if (fabs(wr_000/wg_000) > EPS) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_000]] + shift_r, wr_000/wg_000,  ADD_VALUES); CHKERRXX(ierr); }

//          if (fabs(wg_m00/wg_000) > EPS && is_interface_m00 && neighbors_exist_m[nn_m00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_m00]] + shift_g, wg_m00/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wg_p00/wg_000) > EPS && is_interface_p00 && neighbors_exist_m[nn_p00]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_p00]] + shift_g, wg_p00/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wg_0m0/wg_000) > EPS && is_interface_0m0 && neighbors_exist_m[nn_0m0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0m0]] + shift_g, wg_0m0/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wg_0p0/wg_000) > EPS && is_interface_0p0 && neighbors_exist_m[nn_0p0]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_0p0]] + shift_g, wg_0p0/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//#ifdef P4_TO_P8
//          if (fabs(wg_00m/wg_000) > EPS && is_interface_00m && neighbors_exist_m[nn_00m]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00m]] + shift_g, wg_00m/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//          if (fabs(wg_00p/wg_000) > EPS && is_interface_00p && neighbors_exist_m[nn_00p]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[neighbors[nn_00p]] + shift_g, wg_00p/wg_000,  ADD_VALUES); CHKERRXX(ierr); }
//#endif
        }

//        if (is_interface_m00) { val_interface_m00 = (*u_jump_)(x_C - theta_m00, y_C, z_C); }
//        if (is_interface_p00) { val_interface_p00 = (*u_jump_)(x_C + theta_p00, y_C, z_C); }
//        if (is_interface_0m0) { val_interface_0m0 = (*u_jump_)(x_C, y_C - theta_0m0, z_C); }
//        if (is_interface_0p0) { val_interface_0p0 = (*u_jump_)(x_C, y_C + theta_0p0, z_C); }
//        if (is_interface_00m) { val_interface_00m = (*u_jump_)(x_C, y_C, z_C - theta_00m); }
//        if (is_interface_00p) { val_interface_00p = (*u_jump_)(x_C, y_C, z_C + theta_00p); }

        if (setup_rhs)
        {
          rhs_block_ptr[2*n+shift] = (phi_eff_000 < 0 ? rhs_m_ptr[n] : rhs_p_ptr[n]);

          double u_jump_coeff = phi_eff_000 < 0 ? -1 : 1;

#ifdef P4_TO_P8
          if (is_interface_m00) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_m00 * (*u_jump_)(x_C - theta_m00, y_C, z_C); }
          if (is_interface_p00) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_p00 * (*u_jump_)(x_C + theta_p00, y_C, z_C); }
          if (is_interface_0m0) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_0m0 * (*u_jump_)(x_C, y_C - theta_0m0, z_C); }
          if (is_interface_0p0) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_0p0 * (*u_jump_)(x_C, y_C + theta_0p0, z_C); }
          if (is_interface_00m) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_00m * (*u_jump_)(x_C, y_C, z_C - theta_00m); }
          if (is_interface_00p) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_00p * (*u_jump_)(x_C, y_C, z_C + theta_00p); }
#else
          if (is_interface_m00) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_m00 * (*u_jump_)(x_C - theta_m00, y_C); }
          if (is_interface_p00) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_p00 * (*u_jump_)(x_C + theta_p00, y_C); }
          if (is_interface_0m0) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_0m0 * (*u_jump_)(x_C, y_C - theta_0m0); }
          if (is_interface_0p0) { rhs_block_ptr[2*n+shift] -= u_jump_coeff * wr_0p0 * (*u_jump_)(x_C, y_C + theta_0p0); }
#endif
          rhs_block_ptr[2*n+shift] /= wr_000;
//          rhs_block_ptr[2*n+shift] /= wg_000;

          rhs_fd = rhs_block_ptr[2*n+shift];
        }

        if (is_interface_m00) { wr_m00 = 0; }
        if (is_interface_p00) { wr_p00 = 0; }
        if (is_interface_0m0) { wr_0m0 = 0; }
        if (is_interface_0p0) { wr_0p0 = 0; }
#ifdef P4_TO_P8
        if (is_interface_00m) { wr_00m = 0; }
        if (is_interface_00p) { wr_00p = 0; }
#endif
      }


      // use FV/FD combination for nodes around interface

      if (0)
      {
//        int shift   = phi_eff_000 <= 0 ? 0 : 1;
        int shift   = phi_eff_000 <= 0 ? 1 : 0;

        int shift_r = phi_eff_000 <= 0 ? 1 : 0;
//        int shift_r = phi_eff_000 <= 0 ? 0 : 1;

        if (setup_matrix)
        {
          ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_000_g + shift_r, 1.,  ADD_VALUES); CHKERRXX(ierr);
        }

        if (setup_rhs)
        {
          rhs_block_ptr[2*n+shift] = bc_wall_value_->value(xyz_C);
        }
      }
      else
      {
        // first equation is from FV discretization

        //---------------------------------------------------------------------
        // contributions through cell faces
        //---------------------------------------------------------------------

#ifdef P4_TO_P8
        double w_m[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0,
                                           0,0,0, 0,0,0, 0,0,0,
                                           0,0,0, 0,0,0, 0,0,0 };
        double w_p[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0,
                                           0,0,0, 0,0,0, 0,0,0,
                                           0,0,0, 0,0,0, 0,0,0 };

        bool neighbors_exist_2d[9];
        double weights_2d[9];
        bool map_2d[9];

        get_all_neighbors_(n, neighbors, neighbors_exist_m);

        for (short i = 0; i < num_neighbors_max_; ++i)
          if (neighbors_exist_m[i])
          {
            neighbors_exist_p[i] = neighbors_exist_m[i] && (volumes_p[neighbors[i]] < 1.-eps_dom_);
            neighbors_exist_m[i] = neighbors_exist_m[i] && (volumes_p[neighbors[i]] > eps_dom_);
          }

        double theta = EPS;
        if (!use_sc_scheme_) theta = 10;

        // face m00
        if (s_m00_m/full_sx > interface_rel_thresh)
        {
          double A = y_m00_m/dy_min_;
          double B = z_m00_m/dz_min_;

          if (!neighbors_exist_m[nn_m00])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_m00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_m00]]
                         << " Face Area:" << s_m00_m/full_sx << "\n";
          } else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_0mm] && neighbors_exist_m[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_00m] && neighbors_exist_m[nn_m0m];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_0pm] && neighbors_exist_m[nn_mpm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_0m0] && neighbors_exist_m[nn_mm0];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_m00];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_0p0] && neighbors_exist_m[nn_mp0];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_0mp] && neighbors_exist_m[nn_mmp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_00p] && neighbors_exist_m[nn_m0p];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_0pp] && neighbors_exist_m[nn_mpp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(fv_xmin, y_C + y_m00_m, z_C + z_m00_m);

            double flux = mu_val * s_m00_m / dx_min_;

            if (map_2d[nn_mmm]) { w_m[nn_0mm] += weights_2d[nn_mmm] * flux;   w_m[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_00m] += weights_2d[nn_0mm] * flux;   w_m[nn_m0m] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_0pm] += weights_2d[nn_pmm] * flux;   w_m[nn_mpm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_0m0] += weights_2d[nn_m0m] * flux;   w_m[nn_mm0] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_m00] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_0p0] += weights_2d[nn_p0m] * flux;   w_m[nn_mp0] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_0mp] += weights_2d[nn_mpm] * flux;   w_m[nn_mmp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_00p] += weights_2d[nn_0pm] * flux;   w_m[nn_m0p] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_0pp] += weights_2d[nn_ppm] * flux;   w_m[nn_mpp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_m00_p/full_sx > interface_rel_thresh)
        {
          double A = y_m00_p/dy_min_;
          double B = z_m00_p/dz_min_;

          if (!neighbors_exist_p[nn_m00])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_m00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_m00]]
                         << " Face Area:" << s_m00_p/full_sx << "\n";
          } else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_0mm] && neighbors_exist_p[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_00m] && neighbors_exist_p[nn_m0m];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_0pm] && neighbors_exist_p[nn_mpm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_0m0] && neighbors_exist_p[nn_mm0];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_m00];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_0p0] && neighbors_exist_p[nn_mp0];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_0mp] && neighbors_exist_p[nn_mmp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_00p] && neighbors_exist_p[nn_m0p];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_0pp] && neighbors_exist_p[nn_mpp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(fv_xmin, y_C + y_m00_p, z_C + z_m00_p);

            double flux = mu_val * s_m00_p / dx_min_;

            if (map_2d[nn_mmm]) { w_p[nn_0mm] += weights_2d[nn_mmm] * flux;   w_p[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_00m] += weights_2d[nn_0mm] * flux;   w_p[nn_m0m] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_0pm] += weights_2d[nn_pmm] * flux;   w_p[nn_mpm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_0m0] += weights_2d[nn_m0m] * flux;   w_p[nn_mm0] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_m00] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_0p0] += weights_2d[nn_p0m] * flux;   w_p[nn_mp0] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_0mp] += weights_2d[nn_mpm] * flux;   w_p[nn_mmp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_00p] += weights_2d[nn_0pm] * flux;   w_p[nn_m0p] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_0pp] += weights_2d[nn_ppm] * flux;   w_p[nn_mpp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        // face p00
        if (s_p00_m/full_sx > interface_rel_thresh)
        {
          double A = y_p00_m/dy_min_;
          double B = z_p00_m/dz_min_;

          if (!neighbors_exist_m[nn_p00])
            std::cout << "Warning: neighbor doesn't exist in the xp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_p00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_p00]]
                         << " Face Area:" << s_p00_m/full_sx << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_0mm] && neighbors_exist_m[nn_pmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_00m] && neighbors_exist_m[nn_p0m];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_0pm] && neighbors_exist_m[nn_ppm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_0m0] && neighbors_exist_m[nn_pm0];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_p00];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_0p0] && neighbors_exist_m[nn_pp0];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_0mp] && neighbors_exist_m[nn_pmp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_00p] && neighbors_exist_m[nn_p0p];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_0pp] && neighbors_exist_m[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(fv_xmax, y_C + y_p00_m, z_C + z_p00_m);

            double flux = mu_val * s_p00_m / dx_min_;

            if (map_2d[nn_mmm]) { w_m[nn_0mm] += weights_2d[nn_mmm] * flux;   w_m[nn_pmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_00m] += weights_2d[nn_0mm] * flux;   w_m[nn_p0m] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_0pm] += weights_2d[nn_pmm] * flux;   w_m[nn_ppm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_0m0] += weights_2d[nn_m0m] * flux;   w_m[nn_pm0] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_p00] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_0p0] += weights_2d[nn_p0m] * flux;   w_m[nn_pp0] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_0mp] += weights_2d[nn_mpm] * flux;   w_m[nn_pmp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_00p] += weights_2d[nn_0pm] * flux;   w_m[nn_p0p] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_0pp] += weights_2d[nn_ppm] * flux;   w_m[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_p00_p/full_sx > interface_rel_thresh)
        {
          double A = y_p00_p/dy_min_;
          double B = z_p00_p/dz_min_;

          if (!neighbors_exist_p[nn_p00])
            std::cout << "Warning: neighbor doesn't exist in the xp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_p00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_p00]]
                         << " Face Area:" << s_p00_p/full_sx << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_0mm] && neighbors_exist_p[nn_pmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_00m] && neighbors_exist_p[nn_p0m];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_0pm] && neighbors_exist_p[nn_ppm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_0m0] && neighbors_exist_p[nn_pm0];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_p00];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_0p0] && neighbors_exist_p[nn_pp0];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_0mp] && neighbors_exist_p[nn_pmp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_00p] && neighbors_exist_p[nn_p0p];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_0pp] && neighbors_exist_p[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(fv_xmax, y_C + y_p00_p, z_C + z_p00_p);

            double flux = mu_val * s_p00_p / dx_min_;

            if (map_2d[nn_mmm]) { w_p[nn_0mm] += weights_2d[nn_mmm] * flux;   w_p[nn_pmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_00m] += weights_2d[nn_0mm] * flux;   w_p[nn_p0m] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_0pm] += weights_2d[nn_pmm] * flux;   w_p[nn_ppm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_0m0] += weights_2d[nn_m0m] * flux;   w_p[nn_pm0] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_p00] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_0p0] += weights_2d[nn_p0m] * flux;   w_p[nn_pp0] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_0mp] += weights_2d[nn_mpm] * flux;   w_p[nn_pmp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_00p] += weights_2d[nn_0pm] * flux;   w_p[nn_p0p] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_0pp] += weights_2d[nn_ppm] * flux;   w_p[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }


        // face 0m0
        if (s_0m0_m/full_sy > interface_rel_thresh)
        {
          double A = z_0m0_m/dz_min_;
          double B = x_0m0_m/dx_min_;

          if (!neighbors_exist_m[nn_0m0])
            std::cout << "Warning: neighbor doesn't exist in the ym-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0m0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0m0]]
                         << " Face Area:" << s_0m0_m/full_sy << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_m0m] && neighbors_exist_m[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_m00] && neighbors_exist_m[nn_mm0];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_m0p] && neighbors_exist_m[nn_mmp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_00m] && neighbors_exist_m[nn_0mm];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_0m0];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_00p] && neighbors_exist_m[nn_0mp];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_p0m] && neighbors_exist_m[nn_pmm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_p00] && neighbors_exist_m[nn_pm0];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_p0p] && neighbors_exist_m[nn_pmp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(x_C + x_0m0_m, fv_ymin, z_C + z_0m0_m);

            double flux = mu_val * s_0m0_m / dy_min_;

            if (map_2d[nn_mmm]) { w_m[nn_m0m] += weights_2d[nn_mmm] * flux;   w_m[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_m00] += weights_2d[nn_0mm] * flux;   w_m[nn_mm0] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_m0p] += weights_2d[nn_pmm] * flux;   w_m[nn_mmp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_00m] += weights_2d[nn_m0m] * flux;   w_m[nn_0mm] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_0m0] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_00p] += weights_2d[nn_p0m] * flux;   w_m[nn_0mp] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_p0m] += weights_2d[nn_mpm] * flux;   w_m[nn_pmm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_p00] += weights_2d[nn_0pm] * flux;   w_m[nn_pm0] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_p0p] += weights_2d[nn_ppm] * flux;   w_m[nn_pmp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_0m0_p/full_sy > interface_rel_thresh)
        {
          double A = z_0m0_p/dz_min_;
          double B = x_0m0_p/dx_min_;

          if (!neighbors_exist_p[nn_0m0])
            std::cout << "Warning: neighbor doesn't exist in the ym-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0m0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0m0]]
                         << " Face Area:" << s_0m0_p/full_sy << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_m0m] && neighbors_exist_p[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_m00] && neighbors_exist_p[nn_mm0];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_m0p] && neighbors_exist_p[nn_mmp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_00m] && neighbors_exist_p[nn_0mm];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_0m0];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_00p] && neighbors_exist_p[nn_0mp];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_p0m] && neighbors_exist_p[nn_pmm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_p00] && neighbors_exist_p[nn_pm0];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_p0p] && neighbors_exist_p[nn_pmp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(x_C + x_0m0_p, fv_ymin, z_C + z_0m0_p);

            double flux = mu_val * s_0m0_p / dy_min_;

            if (map_2d[nn_mmm]) { w_p[nn_m0m] += weights_2d[nn_mmm] * flux;   w_p[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_m00] += weights_2d[nn_0mm] * flux;   w_p[nn_mm0] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_m0p] += weights_2d[nn_pmm] * flux;   w_p[nn_mmp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_00m] += weights_2d[nn_m0m] * flux;   w_p[nn_0mm] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_0m0] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_00p] += weights_2d[nn_p0m] * flux;   w_p[nn_0mp] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_p0m] += weights_2d[nn_mpm] * flux;   w_p[nn_pmm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_p00] += weights_2d[nn_0pm] * flux;   w_p[nn_pm0] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_p0p] += weights_2d[nn_ppm] * flux;   w_p[nn_pmp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        // face 0p0
        if (s_0p0_m/full_sy > interface_rel_thresh)
        {
          double A = z_0p0_m/dz_min_;
          double B = x_0p0_m/dx_min_;

          if (!neighbors_exist_m[nn_0p0])
            std::cout << "Warning: neighbor doesn't exist in the yp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0p0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0p0]]
                         << " Face Area:" << s_0p0_m/full_sy << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_m0m] && neighbors_exist_m[nn_mpm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_m00] && neighbors_exist_m[nn_mp0];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_m0p] && neighbors_exist_m[nn_mpp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_00m] && neighbors_exist_m[nn_0pm];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_0p0];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_00p] && neighbors_exist_m[nn_0pp];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_p0m] && neighbors_exist_m[nn_ppm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_p00] && neighbors_exist_m[nn_pp0];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_p0p] && neighbors_exist_m[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(x_C + x_0p0_m, fv_ymax, z_C + z_0p0_m);

            double flux = mu_val * s_0p0_m / dy_min_;

            if (map_2d[nn_mmm]) { w_m[nn_m0m] += weights_2d[nn_mmm] * flux;   w_m[nn_mpm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_m00] += weights_2d[nn_0mm] * flux;   w_m[nn_mp0] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_m0p] += weights_2d[nn_pmm] * flux;   w_m[nn_mpp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_00m] += weights_2d[nn_m0m] * flux;   w_m[nn_0pm] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_0p0] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_00p] += weights_2d[nn_p0m] * flux;   w_m[nn_0pp] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_p0m] += weights_2d[nn_mpm] * flux;   w_m[nn_ppm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_p00] += weights_2d[nn_0pm] * flux;   w_m[nn_pp0] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_p0p] += weights_2d[nn_ppm] * flux;   w_m[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_0p0_p/full_sy > interface_rel_thresh)
        {
          double A = z_0p0_p/dz_min_;
          double B = x_0p0_p/dx_min_;

          if (!neighbors_exist_p[nn_0p0])
            std::cout << "Warning: neighbor doesn't exist in the yp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0p0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0p0]]
                         << " Face Area:" << s_0p0_p/full_sy << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_m0m] && neighbors_exist_p[nn_mpm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_m00] && neighbors_exist_p[nn_mp0];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_m0p] && neighbors_exist_p[nn_mpp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_00m] && neighbors_exist_p[nn_0pm];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_0p0];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_00p] && neighbors_exist_p[nn_0pp];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_p0m] && neighbors_exist_p[nn_ppm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_p00] && neighbors_exist_p[nn_pp0];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_p0p] && neighbors_exist_p[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(x_C + x_0p0_p, fv_ymax, z_C + z_0p0_p);

            double flux = mu_val * s_0p0_p / dy_min_;

            if (map_2d[nn_mmm]) { w_p[nn_m0m] += weights_2d[nn_mmm] * flux;   w_p[nn_mpm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_m00] += weights_2d[nn_0mm] * flux;   w_p[nn_mp0] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_m0p] += weights_2d[nn_pmm] * flux;   w_p[nn_mpp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_00m] += weights_2d[nn_m0m] * flux;   w_p[nn_0pm] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_0p0] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_00p] += weights_2d[nn_p0m] * flux;   w_p[nn_0pp] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_p0m] += weights_2d[nn_mpm] * flux;   w_p[nn_ppm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_p00] += weights_2d[nn_0pm] * flux;   w_p[nn_pp0] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_p0p] += weights_2d[nn_ppm] * flux;   w_p[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }


        // face 00m
        if (s_00m_m/full_sz > interface_rel_thresh)
        {
          double A = x_00m_m/dx_min_;
          double B = y_00m_m/dy_min_;

          if (!neighbors_exist_m[nn_00m])
            std::cout << "Warning: neighbor doesn't exist in the zm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00m]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00m]]
                         << " Face Area:" << s_00m_m/full_sz << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_mm0] && neighbors_exist_m[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_0m0] && neighbors_exist_m[nn_0mm];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_pm0] && neighbors_exist_m[nn_pmm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_m00] && neighbors_exist_m[nn_m0m];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_00m];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_p00] && neighbors_exist_m[nn_p0m];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_mp0] && neighbors_exist_m[nn_mpm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_0p0] && neighbors_exist_m[nn_0pm];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_pp0] && neighbors_exist_m[nn_ppm];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(x_C + x_00m_m, y_C + y_00m_m, fv_zmin);

            double flux = mu_val * s_00m_m / dz_min_;

            if (map_2d[nn_mmm]) { w_m[nn_mm0] += weights_2d[nn_mmm] * flux;   w_m[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_0m0] += weights_2d[nn_0mm] * flux;   w_m[nn_0mm] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_pm0] += weights_2d[nn_pmm] * flux;   w_m[nn_pmm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_m00] += weights_2d[nn_m0m] * flux;   w_m[nn_m0m] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_00m] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_p00] += weights_2d[nn_p0m] * flux;   w_m[nn_p0m] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_mp0] += weights_2d[nn_mpm] * flux;   w_m[nn_mpm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_0p0] += weights_2d[nn_0pm] * flux;   w_m[nn_0pm] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_pp0] += weights_2d[nn_ppm] * flux;   w_m[nn_ppm] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_00m_p/full_sz > interface_rel_thresh)
        {
          double A = x_00m_p/dx_min_;
          double B = y_00m_p/dy_min_;

          if (!neighbors_exist_p[nn_00m])
            std::cout << "Warning: neighbor doesn't exist in the zm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00m]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00m]]
                         << " Face Area:" << s_00m_p/full_sz << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_mm0] && neighbors_exist_p[nn_mmm];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_0m0] && neighbors_exist_p[nn_0mm];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_pm0] && neighbors_exist_p[nn_pmm];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_m00] && neighbors_exist_p[nn_m0m];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_00m];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_p00] && neighbors_exist_p[nn_p0m];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_mp0] && neighbors_exist_p[nn_mpm];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_0p0] && neighbors_exist_p[nn_0pm];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_pp0] && neighbors_exist_p[nn_ppm];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(x_C + x_00m_p, y_C + y_00m_p, fv_zmin);

            double flux = mu_val * s_00m_p / dz_min_;

            if (map_2d[nn_mmm]) { w_p[nn_mm0] += weights_2d[nn_mmm] * flux;   w_p[nn_mmm] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_0m0] += weights_2d[nn_0mm] * flux;   w_p[nn_0mm] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_pm0] += weights_2d[nn_pmm] * flux;   w_p[nn_pmm] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_m00] += weights_2d[nn_m0m] * flux;   w_p[nn_m0m] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_00m] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_p00] += weights_2d[nn_p0m] * flux;   w_p[nn_p0m] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_mp0] += weights_2d[nn_mpm] * flux;   w_p[nn_mpm] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_0p0] += weights_2d[nn_0pm] * flux;   w_p[nn_0pm] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_pp0] += weights_2d[nn_ppm] * flux;   w_p[nn_ppm] -= weights_2d[nn_ppm] * flux; }

          }
        }

        // face 00p
        if (s_00p_m/full_sz > interface_rel_thresh)
        {
          double A = x_00p_m/dx_min_;
          double B = y_00p_m/dy_min_;

          if (!neighbors_exist_m[nn_00p])
            std::cout << "Warning: neighbor doesn't exist in the zp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00p]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00p]]
                         << " Face Area:" << s_00p_m/full_sz << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_m[nn_mm0] && neighbors_exist_m[nn_mmp];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_m[nn_0m0] && neighbors_exist_m[nn_0mp];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_m[nn_pm0] && neighbors_exist_m[nn_pmp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_m[nn_m00] && neighbors_exist_m[nn_m0p];
            neighbors_exist_2d[nn_00m] = neighbors_exist_m[nn_000] && neighbors_exist_m[nn_00p];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_m[nn_p00] && neighbors_exist_m[nn_p0p];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_m[nn_mp0] && neighbors_exist_m[nn_mpp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_m[nn_0p0] && neighbors_exist_m[nn_0pp];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_m[nn_pp0] && neighbors_exist_m[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_m_ptr[n] = MAX(mask_m_ptr[n], mask_result);

            double mu_val = mu_m_;
            if (variable_mu_)
              mu_val = interp_local_m.interpolate(x_C + x_00p_m, y_C + y_00p_m, fv_zmax);

            double flux = mu_val * s_00p_m / dz_min_;

            if (map_2d[nn_mmm]) { w_m[nn_mm0] += weights_2d[nn_mmm] * flux;   w_m[nn_mmp] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_m[nn_0m0] += weights_2d[nn_0mm] * flux;   w_m[nn_0mp] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_m[nn_pm0] += weights_2d[nn_pmm] * flux;   w_m[nn_pmp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_m[nn_m00] += weights_2d[nn_m0m] * flux;   w_m[nn_m0p] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_m[nn_000] += weights_2d[nn_00m] * flux;   w_m[nn_00p] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_m[nn_p00] += weights_2d[nn_p0m] * flux;   w_m[nn_p0p] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_m[nn_mp0] += weights_2d[nn_mpm] * flux;   w_m[nn_mpp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_m[nn_0p0] += weights_2d[nn_0pm] * flux;   w_m[nn_0pp] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_m[nn_pp0] += weights_2d[nn_ppm] * flux;   w_m[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        if (s_00p_p/full_sz > interface_rel_thresh)
        {
          double A = x_00p_p/dx_min_;
          double B = y_00p_p/dy_min_;

          if (!neighbors_exist_p[nn_00p])
            std::cout << "Warning: neighbor doesn't exist in the zp-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_00p]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_00p]]
                         << " Face Area:" << s_00p_p/full_sz << "\n";
          else {

            neighbors_exist_2d[nn_mmm] = neighbors_exist_p[nn_mm0] && neighbors_exist_p[nn_mmp];
            neighbors_exist_2d[nn_0mm] = neighbors_exist_p[nn_0m0] && neighbors_exist_p[nn_0mp];
            neighbors_exist_2d[nn_pmm] = neighbors_exist_p[nn_pm0] && neighbors_exist_p[nn_pmp];
            neighbors_exist_2d[nn_m0m] = neighbors_exist_p[nn_m00] && neighbors_exist_p[nn_m0p];
            neighbors_exist_2d[nn_00m] = neighbors_exist_p[nn_000] && neighbors_exist_p[nn_00p];
            neighbors_exist_2d[nn_p0m] = neighbors_exist_p[nn_p00] && neighbors_exist_p[nn_p0p];
            neighbors_exist_2d[nn_mpm] = neighbors_exist_p[nn_mp0] && neighbors_exist_p[nn_mpp];
            neighbors_exist_2d[nn_0pm] = neighbors_exist_p[nn_0p0] && neighbors_exist_p[nn_0pp];
            neighbors_exist_2d[nn_ppm] = neighbors_exist_p[nn_pp0] && neighbors_exist_p[nn_ppp];

            double mask_result = compute_weights_through_face(A, B, neighbors_exist_2d, weights_2d, theta, map_2d);

            //          if (setup_matrix) mask_p_ptr[n] = MAX(mask_p_ptr[n], mask_result);

            double mu_val = mu_p_;
            if (variable_mu_)
              mu_val = interp_local_p.interpolate(x_C + x_00p_p, y_C + y_00p_p, fv_zmax);

            double flux = mu_val * s_00p_p / dz_min_;

            if (map_2d[nn_mmm]) { w_p[nn_mm0] += weights_2d[nn_mmm] * flux;   w_p[nn_mmp] -= weights_2d[nn_mmm] * flux; }
            if (map_2d[nn_0mm]) { w_p[nn_0m0] += weights_2d[nn_0mm] * flux;   w_p[nn_0mp] -= weights_2d[nn_0mm] * flux; }
            if (map_2d[nn_pmm]) { w_p[nn_pm0] += weights_2d[nn_pmm] * flux;   w_p[nn_pmp] -= weights_2d[nn_pmm] * flux; }
            if (map_2d[nn_m0m]) { w_p[nn_m00] += weights_2d[nn_m0m] * flux;   w_p[nn_m0p] -= weights_2d[nn_m0m] * flux; }
            if (map_2d[nn_00m]) { w_p[nn_000] += weights_2d[nn_00m] * flux;   w_p[nn_00p] -= weights_2d[nn_00m] * flux; }
            if (map_2d[nn_p0m]) { w_p[nn_p00] += weights_2d[nn_p0m] * flux;   w_p[nn_p0p] -= weights_2d[nn_p0m] * flux; }
            if (map_2d[nn_mpm]) { w_p[nn_mp0] += weights_2d[nn_mpm] * flux;   w_p[nn_mpp] -= weights_2d[nn_mpm] * flux; }
            if (map_2d[nn_0pm]) { w_p[nn_0p0] += weights_2d[nn_0pm] * flux;   w_p[nn_0pp] -= weights_2d[nn_0pm] * flux; }
            if (map_2d[nn_ppm]) { w_p[nn_pp0] += weights_2d[nn_ppm] * flux;   w_p[nn_ppp] -= weights_2d[nn_ppm] * flux; }

          }
        }

        //          if (mask_p[n] > 0 && setup_matrix)
        //          {
        //            std::vector<simplex3_mls_t *> simplices;
        //            int n_sps = NTETS;

        //            for (int k = 0; k < cubes.size(); k++)
        //              if (cubes[k]->loc == FCE)
        //                for (int l = 0; l < n_sps; l++)
        //                  simplices.push_back(&cubes[k]->simplex[l]);

        //            simplex3_mls_vtk::write_simplex_geometry(simplices, to_string("/home/dbochkov/Outputs/nodes_mls/nodes/"), to_string(n));
        //          }
        //          // testing hypothesis
        //          if (mask_p[n] > 0)
        //          {
        //            if (setup_matrix)
        //            {
        //              ierr = MatSetValue(A_, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        //            }

        //            if (setup_rhs)
        //            {
        //              rhs_p[n] = bc_wall_value_->value(xyz_C);
        //            }
        //            continue;
        //          }

#else
        double w_m[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0 };
        double w_p[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0 };

        get_all_neighbors_(n, neighbors, neighbors_exist_m);

        for (short i = 0; i < num_neighbors_max_; ++i)
          if (neighbors_exist_m[i])
          {
            neighbors_exist_p[i] = neighbors_exist_m[i] && (volumes_p[neighbors[i]] < 1.-eps_dom_);
            neighbors_exist_m[i] = neighbors_exist_m[i] && (volumes_p[neighbors[i]] > eps_dom_);
          }

        double theta = EPS;
        if (!use_sc_scheme_) theta = 10;

        // face m00
        if (s_m00_m/full_sx > interface_rel_thresh)
        {
          double mu_val = mu_m_;
          if (variable_mu_)
            mu_val = interp_local_m.interpolate(fv_xmin, y_C + y_m00_m);

          double flux = mu_val * s_m00_m/dx_min_;
          double ratio = fabs(y_m00_m)/dy_min_;

          if (y_m00_m/full_sx < -theta && neighbors_exist_m[nn_mm0] && neighbors_exist_m[nn_0m0]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_m00] -= (1. - ratio) * flux;
            w_m[nn_0m0] += ratio * flux;
            w_m[nn_mm0] -= ratio * flux;

          } else if (y_m00_m/full_sx > theta && neighbors_exist_m[nn_mp0] && neighbors_exist_m[nn_0p0]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_m00] -= (1. - ratio) * flux;
            w_m[nn_0p0] += ratio * flux;
            w_m[nn_mp0] -= ratio * flux;

          } else {

            w_m[nn_000] += flux;
            w_m[nn_m00] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_m_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        if (s_m00_p/full_sx > interface_rel_thresh)
        {
          double mu_val = mu_p_;
          if (variable_mu_)
            mu_val = interp_local_p.interpolate(fv_xmin, y_C + y_m00_p);

          double flux = mu_val * s_m00_p/dx_min_;
          double ratio = fabs(y_m00_p)/dy_min_;

          if (y_m00_p/full_sx < -theta && neighbors_exist_p[nn_mm0] && neighbors_exist_p[nn_0m0]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_m00] -= (1. - ratio) * flux;
            w_p[nn_0m0] += ratio * flux;
            w_p[nn_mm0] -= ratio * flux;

          } else if (y_m00_p/full_sx > theta && neighbors_exist_p[nn_mp0] && neighbors_exist_p[nn_0p0]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_m00] -= (1. - ratio) * flux;
            w_p[nn_0p0] += ratio * flux;
            w_p[nn_mp0] -= ratio * flux;

          } else {

            w_p[nn_000] += flux;
            w_p[nn_m00] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_p_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        // face p00
        if (s_p00_m/full_sx > interface_rel_thresh)
        {
          double mu_val = mu_m_;
          if (variable_mu_)
            mu_val = interp_local_m.interpolate(fv_xmax, y_C + y_p00_m);

          double flux = mu_val * s_p00_m/dx_min_;
          double ratio = fabs(y_p00_m)/dy_min_;

          if (y_p00_m/full_sx < -theta && neighbors_exist_m[nn_pm0] && neighbors_exist_m[nn_0m0]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_p00] -= (1. - ratio) * flux;
            w_m[nn_0m0] += ratio * flux;
            w_m[nn_pm0] -= ratio * flux;

          } else if (y_p00_m/full_sx > theta && neighbors_exist_m[nn_pp0] && neighbors_exist_m[nn_0p0]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_p00] -= (1. - ratio) * flux;
            w_m[nn_0p0] += ratio * flux;
            w_m[nn_pp0] -= ratio * flux;

          } else {

            w_m[nn_000] += flux;
            w_m[nn_p00] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_m_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        if (s_p00_p/full_sx > interface_rel_thresh)
        {
          double mu_val = mu_p_;
          if (variable_mu_)
            mu_val = interp_local_p.interpolate(fv_xmax, y_C + y_p00_p);

          double flux = mu_val * s_p00_p/dx_min_;
          double ratio = fabs(y_p00_p)/dy_min_;

          if (y_p00_p/full_sx < -theta && neighbors_exist_p[nn_pm0] && neighbors_exist_p[nn_0m0]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_p00] -= (1. - ratio) * flux;
            w_p[nn_0m0] += ratio * flux;
            w_p[nn_pm0] -= ratio * flux;

          } else if (y_p00_p/full_sx > theta && neighbors_exist_p[nn_pp0] && neighbors_exist_p[nn_0p0]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_p00] -= (1. - ratio) * flux;
            w_p[nn_0p0] += ratio * flux;
            w_p[nn_pp0] -= ratio * flux;

          } else {

            w_p[nn_000] += flux;
            w_p[nn_p00] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_p_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        // face_0m0
        if (s_0m0_m/full_sy > interface_rel_thresh)
        {
          double mu_val = mu_m_;
          if (variable_mu_)
            mu_val = interp_local_m.interpolate(x_C + x_0m0_m, fv_ymin);

          double flux = mu_val * s_0m0_m/dy_min_;
          double ratio = fabs(x_0m0_m)/dx_min_;

          if (x_0m0_m/full_sy < -theta && neighbors_exist_m[nn_mm0] && neighbors_exist_m[nn_m00]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_0m0] -= (1. - ratio) * flux;
            w_m[nn_m00] += ratio * flux;
            w_m[nn_mm0] -= ratio * flux;

          } else if (x_0m0_m/full_sy >  theta && neighbors_exist_m[nn_pm0] && neighbors_exist_m[nn_p00]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_0m0] -= (1. - ratio) * flux;
            w_m[nn_p00] += ratio * flux;
            w_m[nn_pm0] -= ratio * flux;

          } else {

            w_m[nn_000] += flux;
            w_m[nn_0m0] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_m_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        if (s_0m0_p/full_sy > interface_rel_thresh)
        {
          double mu_val = mu_p_;
          if (variable_mu_)
            mu_val = interp_local_p.interpolate(x_C + x_0m0_p, fv_ymin);

          double flux = mu_val * s_0m0_p/dy_min_;
          double ratio = fabs(x_0m0_p)/dx_min_;

          if (x_0m0_p/full_sy < -theta && neighbors_exist_p[nn_mm0] && neighbors_exist_p[nn_m00]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_0m0] -= (1. - ratio) * flux;
            w_p[nn_m00] += ratio * flux;
            w_p[nn_mm0] -= ratio * flux;

          } else if (x_0m0_p/full_sy >  theta && neighbors_exist_p[nn_pm0] && neighbors_exist_p[nn_p00]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_0m0] -= (1. - ratio) * flux;
            w_p[nn_p00] += ratio * flux;
            w_p[nn_pm0] -= ratio * flux;

          } else {

            w_p[nn_000] += flux;
            w_p[nn_0m0] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_p_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }


        // face_0p0
        if (s_0p0_m/full_sy > interface_rel_thresh)
        {
          double mu_val = mu_m_;
          if (variable_mu_)
            mu_val = interp_local_m.interpolate(x_C + x_0p0_m, fv_ymax);

          double flux = mu_val * s_0p0_m/dy_min_;
          double ratio = fabs(x_0p0_m)/dx_min_;

          if (x_0p0_m/full_sy < -theta && neighbors_exist_m[nn_mp0] && neighbors_exist_m[nn_m00]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_0p0] -= (1. - ratio) * flux;
            w_m[nn_m00] += ratio * flux;
            w_m[nn_mp0] -= ratio * flux;

          } else if (x_0p0_m/full_sy >  theta && neighbors_exist_m[nn_pp0] && neighbors_exist_m[nn_p00]) {

            w_m[nn_000] += (1. - ratio) * flux;
            w_m[nn_0p0] -= (1. - ratio) * flux;
            w_m[nn_p00] += ratio * flux;
            w_m[nn_pp0] -= ratio * flux;

          } else {

            w_m[nn_000] += flux;
            w_m[nn_0p0] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_m_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }

        if (s_0p0_p/full_sy > interface_rel_thresh)
        {
          double mu_val = mu_p_;
          if (variable_mu_)
            mu_val = interp_local_p.interpolate(x_C + x_0p0_p, fv_ymax);

          double flux = mu_val * s_0p0_p/dy_min_;
          double ratio = fabs(x_0p0_p)/dx_min_;

          if (x_0p0_p/full_sy < -theta && neighbors_exist_p[nn_mp0] && neighbors_exist_p[nn_m00]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_0p0] -= (1. - ratio) * flux;
            w_p[nn_m00] += ratio * flux;
            w_p[nn_mp0] -= ratio * flux;

          } else if (x_0p0_p/full_sy >  theta && neighbors_exist_p[nn_pp0] && neighbors_exist_p[nn_p00]) {

            w_p[nn_000] += (1. - ratio) * flux;
            w_p[nn_0p0] -= (1. - ratio) * flux;
            w_p[nn_p00] += ratio * flux;
            w_p[nn_pp0] -= ratio * flux;

          } else {

            w_p[nn_000] += flux;
            w_p[nn_0p0] -= flux;

            if (ratio > theta && setup_matrix)
            {
              mask_p_ptr[n] = 1;
              std::cout << "Fallback fluxes\n";
            }
          }
        }
#endif

        w_m[nn_000] += diag_add_m_ptr[n]*volume_cut_cell_m;
        w_p[nn_000] += diag_add_p_ptr[n]*volume_cut_cell_p;

        double add_to_rhs = 0;

//        if (phi_eff_000 < 0)
//        {
//          w_m[nn_m00] -= w_m[nn_000]*wr_m00/wr_000;
//          w_m[nn_p00] -= w_m[nn_000]*wr_p00/wr_000;
//          w_m[nn_0m0] -= w_m[nn_000]*wr_0m0/wr_000;
//          w_m[nn_0p0] -= w_m[nn_000]*wr_0p0/wr_000;

//          w_p[nn_000] -= w_m[nn_000]*wg_000/wr_000;
//          w_p[nn_m00] -= w_m[nn_000]*wg_m00/wr_000;
//          w_p[nn_p00] -= w_m[nn_000]*wg_p00/wr_000;
//          w_p[nn_0m0] -= w_m[nn_000]*wg_0m0/wr_000;
//          w_p[nn_0p0] -= w_m[nn_000]*wg_0p0/wr_000;

//          add_to_rhs = -w_m[nn_000]*rhs_fd;

//          w_m[nn_000] = 0.;
//        }
//        else
//        {
//          w_p[nn_m00] -= w_p[nn_000]*wr_m00/wr_000;
//          w_p[nn_p00] -= w_p[nn_000]*wr_p00/wr_000;
//          w_p[nn_0m0] -= w_p[nn_000]*wr_0m0/wr_000;
//          w_p[nn_0p0] -= w_p[nn_000]*wr_0p0/wr_000;

//          w_m[nn_000] -= w_p[nn_000]*wg_000/wr_000;
//          w_m[nn_m00] -= w_p[nn_000]*wg_m00/wr_000;
//          w_m[nn_p00] -= w_p[nn_000]*wg_p00/wr_000;
//          w_m[nn_0m0] -= w_p[nn_000]*wg_0m0/wr_000;
//          w_m[nn_0p0] -= w_p[nn_000]*wg_0p0/wr_000;

//          add_to_rhs = -w_p[nn_000]*rhs_fd;

//          w_p[nn_000] = 0.;
//        }

//        int shift    = phi_eff_000 <= 0 ? 0 : 1;
        int shift    = phi_eff_000 <= 0 ? 1 : 0;

//        double w_000 = phi_eff_000 <= 0 ? w_m[nn_000] : w_p[nn_000];
        double w_000 = phi_eff_000 <= 0 ? w_p[nn_000] : w_m[nn_000];

        if (setup_matrix)
        {
          if (fabs(diag_add_m_ptr[n]*volume_cut_cell_m) > 0) matrix_has_nullspace_ = false;
          if (fabs(diag_add_p_ptr[n]*volume_cut_cell_p) > 0) matrix_has_nullspace_ = false;

          for (int i = 0; i < num_neighbors_max_; ++i)
          {
//            if (neighbors_exist_m[i]) w_m[i] /= w_000;
//            if (neighbors_exist_p[i]) w_p[i] /= w_000;
            w_m[i] /= w_000;
            w_p[i] /= w_000;

            if (w_m[i] != w_m[i]) { std::cout << "w_m = " << w_m[i] << " w_000 = " << w_000 << " v_m = " << volume_cut_cell_m << " v_p = " << volume_cut_cell_p <<"\n"; throw; }
            if (w_p[i] != w_p[i]) { std::cout << "w_p = " << w_p[i] << " w_000 = " << w_000 << "\n"; throw; }
          }

//          phi_eff_000 <= 0 ? w_m[nn_000] = 1. : w_p[nn_000] = 1.;
          phi_eff_000 <= 0 ? w_p[nn_000] = 1. : w_m[nn_000] = 1.;

          if (keep_scalling_) scalling_[2*n + shift] = w_000/full_cell_volume;

          // TODO: this won't probably work near computational domain boundary
          for (int i = 0; i < num_neighbors_max_; ++i)
            //          if (neighbors_exist[i])
          {
            PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
//            if (fabs(w_m[i]) > EPS && neighbors_exist_m[i]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_nei_g,     w_m[i],  ADD_VALUES); CHKERRXX(ierr); }
//            if (fabs(w_p[i]) > EPS && neighbors_exist_p[i]) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_nei_g + 1, w_p[i],  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(w_m[i]) > EPS) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_nei_g,     w_m[i],  ADD_VALUES); CHKERRXX(ierr); }
            if (fabs(w_p[i]) > EPS) { ierr = MatSetValue(A_, 2*node_000_g + shift, 2*node_nei_g + 1, w_p[i],  ADD_VALUES); CHKERRXX(ierr); }
          }
        }

        if (setup_rhs)
        {
          rhs_block_ptr[2*n+shift] = (rhs_m_ptr[n]*volume_cut_cell_m + rhs_p_ptr[n]*volume_cut_cell_p - integral_bc + add_to_rhs)/w_000;
        }

        // free cubes
        for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
        {
          delete cubes[cube_idx];
        }
      }


    }
    else
    {
      // use pure FD away from interface

      double *mue_ptr = phi_eff_000 <= 0 ? mue_m_ptr : mue_p_ptr;
      double mu = phi_eff_000 <= 0 ? mu_m_ : mu_p_;

      if (variable_mu_)
      {
#ifdef P4_TO_P8
          qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
          qnnn.ngbd_with_quadratic_interpolation(mue_ptr, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif
      } else {
          mue_000 = mu;
          mue_m00 = mu; mue_p00 = mu;
          mue_0m0 = mu; mue_0p0 = mu;
#ifdef P4_TO_P8
          mue_00m = mu; mue_00p = mu;
#endif
      }


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
      if (is_node_xmWall(p4est_, ni))      w_p00 += -1.0/(d_p00*d_p00);
      else if (is_node_xpWall(p4est_, ni)) w_m00 += -1.0/(d_m00*d_m00);
      else                                 w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

      if (is_node_xpWall(p4est_, ni))      w_m00 += -1.0/(d_m00*d_m00);
      else if (is_node_xmWall(p4est_, ni)) w_p00 += -1.0/(d_p00*d_p00);
      else                                 w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

      if (is_node_ymWall(p4est_, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
      else if (is_node_ypWall(p4est_, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
      else                                 w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

      if (is_node_ypWall(p4est_, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
      else if (is_node_ymWall(p4est_, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
      else                                 w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

      if (is_node_zmWall(p4est_, ni))      w_00p += -1.0/(d_00p*d_00p);
      else if (is_node_zpWall(p4est_, ni)) w_00m += -1.0/(d_00m*d_00m);
      else                                 w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

      if (is_node_zpWall(p4est_, ni))      w_00m += -1.0/(d_00m*d_00m);
      else if (is_node_zmWall(p4est_, ni)) w_00p += -1.0/(d_00p*d_00p);
      else                                 w_00p += -2.0*wk/d_00p/(d_00m+d_00p);

      if (variable_mu_)
      {
        if (setup_matrix)
        {
          w_m00_mm = 0.5*(mue_000 + mue_ptr[node_m00_mm])*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_mp = 0.5*(mue_000 + mue_ptr[node_m00_mp])*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pm = 0.5*(mue_000 + mue_ptr[node_m00_pm])*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pp = 0.5*(mue_000 + mue_ptr[node_m00_pp])*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);

          w_p00_mm = 0.5*(mue_000 + mue_ptr[node_p00_mm])*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_mp = 0.5*(mue_000 + mue_ptr[node_p00_mp])*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pm = 0.5*(mue_000 + mue_ptr[node_p00_pm])*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pp = 0.5*(mue_000 + mue_ptr[node_p00_pp])*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);

          w_0m0_mm = 0.5*(mue_000 + mue_ptr[node_0m0_mm])*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_mp = 0.5*(mue_000 + mue_ptr[node_0m0_mp])*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pm = 0.5*(mue_000 + mue_ptr[node_0m0_pm])*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pp = 0.5*(mue_000 + mue_ptr[node_0m0_pp])*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);

          w_0p0_mm = 0.5*(mue_000 + mue_ptr[node_0p0_mm])*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_mp = 0.5*(mue_000 + mue_ptr[node_0p0_mp])*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pm = 0.5*(mue_000 + mue_ptr[node_0p0_pm])*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pp = 0.5*(mue_000 + mue_ptr[node_0p0_pp])*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);

          w_00m_mm = 0.5*(mue_000 + mue_ptr[node_00m_mm])*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_mp = 0.5*(mue_000 + mue_ptr[node_00m_mp])*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pm = 0.5*(mue_000 + mue_ptr[node_00m_pm])*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pp = 0.5*(mue_000 + mue_ptr[node_00m_pp])*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);

          w_00p_mm = 0.5*(mue_000 + mue_ptr[node_00p_mm])*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_mp = 0.5*(mue_000 + mue_ptr[node_00p_mp])*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pm = 0.5*(mue_000 + mue_ptr[node_00p_pm])*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pp = 0.5*(mue_000 + mue_ptr[node_00p_pp])*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);

          w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
        } else {
          w_m00 *= 0.5*(mue_000 + mue_m00);
          w_p00 *= 0.5*(mue_000 + mue_p00);
          w_0m0 *= 0.5*(mue_000 + mue_0m0);
          w_0p0 *= 0.5*(mue_000 + mue_0p0);
          w_00m *= 0.5*(mue_000 + mue_00m);
          w_00p *= 0.5*(mue_000 + mue_00p);
        }
      } else {
        if (setup_matrix)
        {
          w_m00_mm = mu*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_mp = mu*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pm = mu*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pp = mu*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);

          w_p00_mm = mu*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_mp = mu*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pm = mu*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pp = mu*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);

          w_0m0_mm = mu*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_mp = mu*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pm = mu*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pp = mu*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);

          w_0p0_mm = mu*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_mp = mu*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pm = mu*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pp = mu*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);

          w_00m_mm = mu*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_mp = mu*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pm = mu*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pp = mu*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);

          w_00p_mm = mu*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_mp = mu*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pm = mu*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pp = mu*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);

          w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
        } else {
          w_m00 *= mu;
          w_p00 *= mu;
          w_0m0 *= mu;
          w_0p0 *= mu;
          w_00m *= mu;
          w_00p *= mu;
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

      // if node is at wall, what's below will enforce Neumann BC
      if (is_node_xmWall(p4est_, ni))      w_p00 += -1.0/(d_p00*d_p00);
      else if (is_node_xpWall(p4est_, ni)) w_m00 += -1.0/(d_m00*d_m00);
      else                                 w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

      if (is_node_xpWall(p4est_, ni))      w_m00 += -1.0/(d_m00*d_m00);
      else if(is_node_xmWall(p4est_, ni))  w_p00 += -1.0/(d_p00*d_p00);
      else                                 w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

      if (is_node_ymWall(p4est_, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
      else if (is_node_ypWall(p4est_, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
      else                                 w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

      if (is_node_ypWall(p4est_, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
      else if (is_node_ymWall(p4est_, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
      else                                 w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

      //---------------------------------------------------------------------
      // addition to diagonal elements
      //---------------------------------------------------------------------
      if (variable_mu_)
      {
        if (setup_matrix)
        {
          w_m00_mm = 0.5*(mue_000 + mue_ptr[node_m00_mm])*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
          w_m00_pm = 0.5*(mue_000 + mue_ptr[node_m00_pm])*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
          w_p00_mm = 0.5*(mue_000 + mue_ptr[node_p00_mm])*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
          w_p00_pm = 0.5*(mue_000 + mue_ptr[node_p00_pm])*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
          w_0m0_mm = 0.5*(mue_000 + mue_ptr[node_0m0_mm])*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
          w_0m0_pm = 0.5*(mue_000 + mue_ptr[node_0m0_pm])*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
          w_0p0_mm = 0.5*(mue_000 + mue_ptr[node_0p0_mm])*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
          w_0p0_pm = 0.5*(mue_000 + mue_ptr[node_0p0_pm])*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);

          w_m00 = w_m00_mm + w_m00_pm;
          w_p00 = w_p00_mm + w_p00_pm;
          w_0m0 = w_0m0_mm + w_0m0_pm;
          w_0p0 = w_0p0_mm + w_0p0_pm;
        } else {
          w_m00 *= 0.5*(mue_000 + mue_m00);
          w_p00 *= 0.5*(mue_000 + mue_p00);
          w_0m0 *= 0.5*(mue_000 + mue_0m0);
          w_0p0 *= 0.5*(mue_000 + mue_0p0);
        }
      } else {
        if (setup_matrix)
        {
          w_m00_mm = mu*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
          w_m00_pm = mu*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
          w_p00_mm = mu*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
          w_p00_pm = mu*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
          w_0m0_mm = mu*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
          w_0m0_pm = mu*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
          w_0p0_mm = mu*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
          w_0p0_pm = mu*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);

          w_m00 = w_m00_mm + w_m00_pm;
          w_p00 = w_p00_mm + w_p00_pm;
          w_0m0 = w_0m0_mm + w_0m0_pm;
          w_0p0 = w_0p0_mm + w_0p0_pm;
        } else {
          w_m00 *= mu;
          w_p00 *= mu;
          w_0m0 *= mu;
          w_0p0 *= mu;
        }
      }
#endif

      double diag_add_value = phi_eff_000 < 0 ? diag_add_m_ptr[n] : diag_add_p_ptr[n];
#ifdef P4_TO_P8
      double w_000  = diag_add_value - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
      double w_000  = diag_add_value - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif

      if (setup_matrix)
      {
        phi_eff_000 < 0 ? mask_m_ptr[n] = -1 : mask_p_ptr[n] = -1;
        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        if (node_000_g < fixed_value_idx_g_){
          fixed_value_idx_l_ = n;
          fixed_value_idx_g_ = node_000_g;
        }

        ierr = MatSetValue(A_, 2*node_000_g,     2*node_000_g,     1.0, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A_, 2*node_000_g + 1, 2*node_000_g + 1, 1.0, ADD_VALUES); CHKERRXX(ierr);

        int shift = phi_eff_000 > 0 ? 1 : 0;

        if (!is_node_xmWall(p4est_, ni))
        {
          if (ABS(w_m00_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_m00_mm] + shift, w_m00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_m00_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_m00_pm] + shift, w_m00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_m00_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_m00_mp] + shift, w_m00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_m00_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_m00_pp] + shift, w_m00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if (!is_node_xpWall(p4est_, ni))
        {
          if (ABS(w_p00_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_p00_mm] + shift, w_p00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_p00_pm] + shift, w_p00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_p00_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_p00_mp] + shift, w_p00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_p00_pp] + shift, w_p00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if (!is_node_ymWall(p4est_, ni))
        {
          if (ABS(w_0m0_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0m0_mm] + shift, w_0m0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0m0_pm] + shift, w_0m0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_0m0_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0m0_mp] + shift, w_0m0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0m0_pp] + shift, w_0m0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if (!is_node_ypWall(p4est_, ni))
        {
          if (ABS(w_0p0_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0p0_mm] + shift, w_0p0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0p0_pm] + shift, w_0p0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_0p0_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0p0_mp] + shift, w_0p0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_0p0_pp] + shift, w_0p0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }
#ifdef P4_TO_P8
        if (!is_node_zmWall(p4est_, ni))
        {
          if (ABS(w_00m_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00m_mm] + shift, w_00m_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00m_pm] + shift, w_00m_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00m_mp] + shift, w_00m_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00m_pp] + shift, w_00m_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }

        if (!is_node_zpWall(p4est_, ni))
        {
          if (ABS(w_00p_mm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00p_mm] + shift, w_00p_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_pm/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00p_pm] + shift, w_00p_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_mp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00p_mp] + shift, w_00p_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_pp/w_000) > EPS) {ierr = MatSetValue(A_, 2*node_000_g + shift, 2*petsc_gloidx_[node_00p_pp] + shift, w_00p_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }
#endif

        if (keep_scalling_)
        {
          scalling_[2*n  ] = phi_eff_000 <=0 ? w_000 : 1;
          scalling_[2*n+1] = phi_eff_000 > 0 ? w_000 : 1;
        }

        if (diag_add_value > 0) matrix_has_nullspace_ = false;
      }

      if (setup_rhs)
      {
        //---------------------------------------------------------------------
        // add coefficients to the right hand side
        //---------------------------------------------------------------------
        // FIX this for variable mu

        rhs_block_ptr[2*n  ] = phi_eff_000 <=0 ? rhs_m_ptr[n] : 0;
        rhs_block_ptr[2*n+1] = phi_eff_000 > 0 ? rhs_p_ptr[n] : 0;

        int shift = phi_eff_000 > 0 ? 1 : 0;
#ifdef P4_TO_P8
        double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS : (is_node_xpWall(p4est_, ni) ? -2.*EPS : 0);
        double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS : (is_node_ypWall(p4est_, ni) ? -2.*EPS : 0);
        double eps_z = is_node_zmWall(p4est_, ni) ? 2.*EPS : (is_node_zpWall(p4est_, ni) ? -2.*EPS : 0);

        if(is_node_xmWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_p00;
        if(is_node_xpWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_m00;
        if(is_node_ymWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0p0;
        if(is_node_ypWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0m0;
        if(is_node_zmWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00p;
        if(is_node_zpWall(p4est_, ni)) rhs_block_ptr[2*n + shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00m;
#else

        double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS : (is_node_xpWall(p4est_, ni) ? -2.*EPS : 0);
        double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS : (is_node_ypWall(p4est_, ni) ? -2.*EPS : 0);

        if(is_node_xmWall(p4est_, ni)) rhs_block_ptr[2*n+shift] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_p00;
        if(is_node_xpWall(p4est_, ni)) rhs_block_ptr[2*n+shift] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_m00;
        if(is_node_ymWall(p4est_, ni)) rhs_block_ptr[2*n+shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0p0;
        if(is_node_ypWall(p4est_, ni)) rhs_block_ptr[2*n+shift] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0m0;
#endif

        rhs_block_ptr[2*n  ] /= w_000;
        rhs_block_ptr[2*n+1] /= w_000;
      }
      continue;
    }
  }

  if (setup_matrix)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }


  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(mask_m_, &mask_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask_p_, &mask_p_ptr); CHKERRXX(ierr);
  if (setup_matrix)
  {
    ierr = VecGhostUpdateBegin(mask_m_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(mask_p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_m_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }


  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  if (variable_mu_) {
    ierr = VecRestoreArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(diag_add_m_, &diag_add_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(diag_add_p_, &diag_add_p_ptr); CHKERRXX(ierr);

  if (setup_rhs)
  {
    ierr = VecRestoreArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_block_, &rhs_block_ptr); CHKERRXX(ierr);
  }

  for (int i = 0; i < num_interfaces_; ++i)
    delete phi_interp_local[i];

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
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup, rhs_block_, 0, 0, 0); CHKERRXX(ierr);
  }

}

double my_p4est_poisson_jump_nodes_mls_sc_t::find_interface_location_mls(p4est_locidx_t n0, p4est_locidx_t n1, double h,
                                                                         std::vector<double *> &phi_p,
                                                                         std::vector<double *> &phi_xx_p)
{

  std::vector<double> points;
  std::vector<loc_t>  points_loc;

  points.push_back(0); points_loc.push_back(INS);
  points.push_back(h); points_loc.push_back(INS);

  std::vector<int> vtx0, vtx1;
  std::vector<loc_t> edgs_loc;
  std::vector<int> edgs_recycled;

  vtx0.push_back(0); vtx1.push_back(1); edgs_loc.push_back(INS); edgs_recycled.push_back(0);

  for (int i = 0; i < num_interfaces_; ++i)
  {
    // find interface location
    double theta_tmp;
    bool pos_slope = phi_p[i][n1] > 0;

    if (phi_p[i][n0] < 0 && phi_p[i][n1] < 0)
    {
      theta_tmp = -h;
    }
    else if (phi_p[i][n0] > 0 && phi_p[i][n1] > 0)
    {
      theta_tmp = -h;
    }
    else
    {
      theta_tmp = interface_Location_With_Second_Order_Derivative(0., h, phi_p[i][n0], phi_p[i][n1], phi_xx_p[i][n0], phi_xx_p[i][n1]);
    }

    // loop through all points first
    for (int pnt_idx = 0; pnt_idx < points.size(); ++pnt_idx)
    {
      if (theta_tmp > points[pnt_idx])
      {
        if ( pos_slope && action_->at(i) == ADDITION)     points_loc[pnt_idx] = INS;
        if (!pos_slope && action_->at(i) == INTERSECTION) points_loc[pnt_idx] = OUT;
      }
      else
      {
        if (!pos_slope && action_->at(i) == ADDITION)     points_loc[pnt_idx] = INS;
        if ( pos_slope && action_->at(i) == INTERSECTION) points_loc[pnt_idx] = OUT;
      }
    }

    // loop through edges
    int num_edgs = edgs_loc.size();
    for (int edg_idx = 0; edg_idx < num_edgs; ++edg_idx)
      if (edgs_recycled[edg_idx] == 0)
      {
        if (theta_tmp < points[vtx0[edg_idx]])
        {
          if ( pos_slope && action_->at(i) == INTERSECTION) edgs_loc[edg_idx] = OUT;
          if (!pos_slope && action_->at(i) == ADDITION)     edgs_loc[edg_idx] = INS;
        }
        else if (theta_tmp > points[vtx1[edg_idx]])
        {
          if (!pos_slope && action_->at(i) == INTERSECTION) edgs_loc[edg_idx] = OUT;
          if ( pos_slope && action_->at(i) == ADDITION)     edgs_loc[edg_idx] = INS;
        }
        else
        {
          edgs_recycled[edg_idx] = 1;
          points.push_back(theta_tmp);

          vtx0.push_back(vtx0[edg_idx]);   vtx1.push_back(points.size()-1); edgs_recycled.push_back(0);
          vtx0.push_back(points.size()-1); vtx1.push_back(vtx1[edg_idx]);   edgs_recycled.push_back(0);

          if (edgs_loc[edg_idx] == INS)
          {
            switch (action_->at(i))
            {
              case ADDITION:
                points_loc.push_back(INS);
                edgs_loc.push_back(INS);
                edgs_loc.push_back(INS);
                break;
              case INTERSECTION:
                points_loc.push_back(FCE);
                if (pos_slope)
                {
                  edgs_loc.push_back(INS);
                  edgs_loc.push_back(OUT);
                }
                else
                {
                  edgs_loc.push_back(OUT);
                  edgs_loc.push_back(INS);
                }
                break;
            }
          }
          else if (edgs_loc[edg_idx] == OUT)
          {
            switch (action_->at(i))
            {
              case ADDITION:
                points_loc.push_back(FCE);
                if (pos_slope)
                {
                  edgs_loc.push_back(INS);
                  edgs_loc.push_back(OUT);
                }
                else
                {
                  edgs_loc.push_back(OUT);
                  edgs_loc.push_back(INS);
                }
                break;
              case INTERSECTION:
                points_loc.push_back(OUT);
                edgs_loc.push_back(OUT);
                edgs_loc.push_back(OUT);
                break;
            }
          }
          else throw;
        }
      }
  }

  for (int i = 0; i < points.size(); ++i)
    if (points_loc[i] == FCE) return points[i];

  throw;
}


void my_p4est_poisson_jump_nodes_mls_sc_t::compute_volumes_()
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

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

  double *volumes_p;
  if (volumes_ != NULL) { ierr = VecDestroy(volumes_); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_->at(0), &volumes_); CHKERRXX(ierr);
  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  // data for refined cells
  std::vector<double> phi_fv(pow(2*cube_refinement_+1,P4EST_DIM),-1);
  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement_+1, 0);
  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement_+1, 0);
#ifdef P4_TO_P8
  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement_+1, 0);
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  double full_cell_volume;

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  std::vector<my_p4est_interpolation_nodes_local_t *> phi_interp_local(num_interfaces_, NULL);

  for (int i = 0; i < num_interfaces_; ++i)
  {
    phi_interp_local[i] = new my_p4est_interpolation_nodes_local_t (node_neighbors_);
#ifdef P4_TO_P8
    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], phi_zz_p[i], quadratic);
#else
    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], quadratic);
#endif
//#ifdef P4_TO_P8
//    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], phi_zz_p[i], linear);
//#else
//    phi_interp_local[i]->set_input(phi_p[i], phi_xx_p[i], phi_yy_p[i], linear);
//#endif
  }

#ifdef P4_TO_P8
  std::vector<CF_3 *> phi_interp_local_cf(num_interfaces_, NULL);
#else
  std::vector<CF_2 *> phi_interp_local_cf(num_interfaces_, NULL);
#endif
  for (int i = 0; i < num_interfaces_; ++i)
    phi_interp_local_cf[i] = phi_interp_local[i];

  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
  {
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

    volumes_p[n] = 0;

    if(is_node_Wall(p4est_, ni))
    {
#ifdef P4_TO_P8
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
#else
      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
#endif
      {
        if (phi_eff_000 < 0. || num_interfaces_ == 0)
        {
          volumes_p[n] = 1;
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

        if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
        {
          volumes_p[n] = 1;
        }
        continue;
      }

    }

    {
      interp_local.initialize(n);

      for (short i = 0; i < num_interfaces_; ++i)
        phi_interp_local[i]->copy_init(interp_local);

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_ngbd_crossed_neumann   = false;

      if (fabs(phi_eff_000) < 2.*diag_min_)
      {
        // determine dimensions of cube
        fv_size_x = 0;
        fv_size_y = 0;
#ifdef P4_TO_P8
        fv_size_z = 0;
#endif
        if(!is_node_xmWall(p4est_, ni)) {fv_size_x += cube_refinement_; fv_xmin = x_C-0.5*dx_min_;} else {fv_xmin = x_C;}
        if(!is_node_xpWall(p4est_, ni)) {fv_size_x += cube_refinement_; fv_xmax = x_C+0.5*dx_min_;} else {fv_xmax = x_C;}

        if(!is_node_ymWall(p4est_, ni)) {fv_size_y += cube_refinement_; fv_ymin = y_C-0.5*dy_min_;} else {fv_ymin = y_C;}
        if(!is_node_ypWall(p4est_, ni)) {fv_size_y += cube_refinement_; fv_ymax = y_C+0.5*dy_min_;} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
        if(!is_node_zmWall(p4est_, ni)) {fv_size_z += cube_refinement_; fv_zmin = z_C-0.5*dz_min_;} else {fv_zmin = z_C;}
        if(!is_node_zpWall(p4est_, ni)) {fv_size_z += cube_refinement_; fv_zmax = z_C+0.5*dz_min_;} else {fv_zmax = z_C;}
#endif

#ifdef P4_TO_P8
        full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin)*(fv_zmax-fv_zmin);
#else
        full_cell_volume = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin);
#endif
        if (!use_refined_cube_) {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }

        fv_nx = fv_size_x+1;
        fv_ny = fv_size_y+1;
#ifdef P4_TO_P8
        fv_nz = fv_size_z+1;
#endif

        // get coordinates of cube nodes
        double fv_dx = (fv_xmax-fv_xmin) / (double)(fv_size_x);
        fv_x[0] = fv_xmin;
        for (short i = 1; i < fv_nx; ++i)
          fv_x[i] = fv_x[i-1] + fv_dx;

        double fv_dy = (fv_ymax-fv_ymin) / (double)(fv_size_y);
        fv_y[0] = fv_ymin;
        for (short i = 1; i < fv_ny; ++i)
          fv_y[i] = fv_y[i-1] + fv_dy;
#ifdef P4_TO_P8
        double fv_dz = (fv_zmax-fv_zmin) / (double)(fv_size_z);
        fv_z[0] = fv_zmin;
        for (short i = 1; i < fv_nz; ++i)
          fv_z[i] = fv_z[i-1] + fv_dz;
#endif
        // sample level-set function at cube nodes and check if crossed
        for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
          bool is_one_positive = false;
          bool is_one_negative = false;

#ifdef P4_TO_P8
          for (short k = 0; k < fv_nz; ++k)
#endif
            for (short j = 0; j < fv_ny; ++j)
              for (short i = 0; i < fv_nx; ++i)
              {
#ifdef P4_TO_P8
                int idx = k*fv_nx*fv_ny + j*fv_nx + i;
                phi_fv[idx] = phi_interp_local[phi_idx]->interpolate(fv_x[i], fv_y[j], fv_z[k]);
#else
                int idx = j*fv_nx + i;
                phi_fv[idx] = phi_interp_local[phi_idx]->interpolate(fv_x[i], fv_y[j]);
#endif
                is_one_positive = is_one_positive || phi_fv[idx] > 0;
                is_one_negative = is_one_negative || phi_fv[idx] < 0;
              }

          if (is_one_negative && is_one_positive)
          {
            is_ngbd_crossed_neumann   = true;
          }
        }
      }

      if (is_ngbd_crossed_neumann) {

        double volume_cut_cell = 0.;

#ifdef USE_QUADRATIC_CUBES
#ifdef P4_TO_P8
        std::vector<cube3_mls_quadratic_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
#else
        std::vector<cube2_mls_quadratic_t *> cubes(fv_size_x*fv_size_y, NULL);
#endif
#else
#ifdef P4_TO_P8
        std::vector<cube3_mls_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
#else
        std::vector<cube2_mls_t *> cubes(fv_size_x*fv_size_y, NULL);
#endif
#endif

#ifdef P4_TO_P8
        for (short k = 0; k < fv_size_z; ++k)
#endif
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
#ifdef USE_QUADRATIC_CUBES
#ifdef P4_TO_P8
              int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
              cubes[idx] = new cube3_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
#else
              int idx = j*fv_size_x + i;
              cubes[idx] = new cube2_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
#endif
#else
#ifdef P4_TO_P8
              int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
              cubes[idx] = new cube3_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
#else
              int idx = j*fv_size_x + i;
              cubes[idx] = new cube2_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
#endif
#endif

              cubes[idx]->construct_domain(phi_interp_local_cf, *action_, *color_);

              volume_cut_cell += cubes[idx]->integrate_over_domain(unity_cf_);
//              interface_area  += cubes[idx]->integrate_over_interface(unity_cf_, -1);

            }

        volumes_p[n] = volume_cut_cell/full_cell_volume;

        // free cubes
        for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
        {
          delete cubes[cube_idx];
        }
      }
      else if (phi_eff_000 < 0)
      {
        volumes_p[n] = 1;
      }
    }
  }

  // restore pointers
  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

}

void my_p4est_poisson_jump_nodes_mls_sc_t::find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr)
{
  // find projection point
  double phi_x = 0., phi_y = 0.;
#ifdef P4_TO_P8
  double phi_z = 0;
#endif

  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, qnnn.node_000);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est_, ni);
  bool xp_wall = is_node_xpWall(p4est_, ni);
  bool ym_wall = is_node_ymWall(p4est_, ni);
  bool yp_wall = is_node_ypWall(p4est_, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est_, ni);
  bool zp_wall = is_node_zpWall(p4est_, ni);
#endif

  if (!xm_wall && !xp_wall) phi_x = qnnn.dx_central(phi_p);
  else if (!xm_wall)        phi_x = qnnn.dx_backward_linear(phi_p);
  else if (!xp_wall)        phi_x = qnnn.dx_forward_linear(phi_p);

  if (!ym_wall && !yp_wall) phi_y = qnnn.dy_central(phi_p);
  else if (!ym_wall)        phi_y = qnnn.dy_backward_linear(phi_p);
  else if (!yp_wall)        phi_y = qnnn.dy_forward_linear(phi_p);

#ifdef P4_TO_P8
  if (!zm_wall && !zp_wall) phi_z = qnnn.dz_central(phi_p);
  else if (!zm_wall)        phi_z = qnnn.dz_backward_linear(phi_p);
  else if (!zp_wall)        phi_z = qnnn.dz_forward_linear(phi_p);
#endif

#ifdef P4_TO_P8
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y)+SQR(phi_z));
#else
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y));
#endif

  phi_x /= phi_d;
  phi_y /= phi_d;
#ifdef P4_TO_P8
  phi_z /= phi_d;
#endif

  dist_pr = phi_p[qnnn.node_000]/phi_d;

  dxyz_pr[0] = - dist_pr*phi_x;
  dxyz_pr[1] = - dist_pr*phi_y;
#ifdef P4_TO_P8
  dxyz_pr[2] = - dist_pr*phi_z;
#endif
}

void my_p4est_poisson_jump_nodes_mls_sc_t::compute_normal_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[])
{
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, qnnn.node_000);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est_, ni);
  bool xp_wall = is_node_xpWall(p4est_, ni);
  bool ym_wall = is_node_ymWall(p4est_, ni);
  bool yp_wall = is_node_ypWall(p4est_, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est_, ni);
  bool zp_wall = is_node_zpWall(p4est_, ni);
#endif

  if (!xm_wall && !xp_wall) n[0] = qnnn.dx_central        (phi_p);
  else if (!xm_wall)        n[0] = qnnn.dx_backward_linear(phi_p);
  else if (!xp_wall)        n[0] = qnnn.dx_forward_linear (phi_p);

  if (!ym_wall && !yp_wall) n[1] = qnnn.dy_central        (phi_p);
  else if (!ym_wall)        n[1] = qnnn.dy_backward_linear(phi_p);
  else if (!yp_wall)        n[1] = qnnn.dy_forward_linear (phi_p);

#ifdef P4_TO_P8
  if (!zm_wall && !zp_wall) n[2] = qnnn.dz_central        (phi_p);
  else if (!zm_wall)        n[2] = qnnn.dz_backward_linear(phi_p);
  else if (!zp_wall)        n[2] = qnnn.dz_forward_linear (phi_p);
#endif

#ifdef P4_TO_P8
  double phi_d = sqrt(SQR(n[0])+SQR(n[1])+SQR(n[2]));
#else
  double phi_d = sqrt(SQR(n[0])+SQR(n[1]));
#endif

  n[0] /= phi_d;
  n[1] /= phi_d;
#ifdef P4_TO_P8
  n[2] /= phi_d;
#endif
}



void my_p4est_poisson_jump_nodes_mls_sc_t::inv_mat3_(double *in, double *out)
{
  double det = in[3*0+0]*(in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2]) -
               in[3*0+1]*(in[3*1+0]*in[3*2+2] - in[3*1+2]*in[3*2+0]) +
               in[3*0+2]*(in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1]);

  out[3*0+0] = (in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2])/det;
  out[3*0+1] = (in[3*0+2]*in[3*2+1] - in[3*2+2]*in[3*0+1])/det;
  out[3*0+2] = (in[3*0+1]*in[3*1+2] - in[3*1+1]*in[3*0+2])/det;

  out[3*1+0] = (in[3*1+2]*in[3*2+0] - in[3*2+2]*in[3*1+0])/det;
  out[3*1+1] = (in[3*0+0]*in[3*2+2] - in[3*2+0]*in[3*0+2])/det;
  out[3*1+2] = (in[3*0+2]*in[3*1+0] - in[3*1+2]*in[3*0+0])/det;

  out[3*2+0] = (in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1])/det;
  out[3*2+1] = (in[3*0+1]*in[3*2+0] - in[3*2+1]*in[3*0+0])/det;
  out[3*2+2] = (in[3*0+0]*in[3*1+1] - in[3*1+0]*in[3*0+1])/det;
}

void my_p4est_poisson_jump_nodes_mls_sc_t::inv_mat2_(double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];
  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;
}


void my_p4est_poisson_jump_nodes_mls_sc_t::get_all_neighbors_(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists)
{
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est_, ni);
  bool xp_wall = is_node_xpWall(p4est_, ni);
  bool ym_wall = is_node_ymWall(p4est_, ni);
  bool yp_wall = is_node_ypWall(p4est_, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est_, ni);
  bool zp_wall = is_node_zpWall(p4est_, ni);
#endif

  // count neighbors
  for (int i = 0; i < num_neighbors_max_; i++) neighbor_exists[i] = true;

  if (xm_wall)
  {
    int i = 0;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (xp_wall)
  {
    int i = 2;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (ym_wall)
  {
    int j = 0;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (yp_wall)
  {
    int j = 2;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

#ifdef P4_TO_P8
  if (zm_wall)
  {
    int k = 0;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }

  if (zp_wall)
  {
    int k = 2;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }
#endif

  // find neighboring quadrants
  p4est_locidx_t quad_mmm_idx; p4est_topidx_t tree_mmm_idx;
  p4est_locidx_t quad_mpm_idx; p4est_topidx_t tree_mpm_idx;
  p4est_locidx_t quad_pmm_idx; p4est_topidx_t tree_pmm_idx;
  p4est_locidx_t quad_ppm_idx; p4est_topidx_t tree_ppm_idx;
#ifdef P4_TO_P8
  p4est_locidx_t quad_mmp_idx; p4est_topidx_t tree_mmp_idx;
  p4est_locidx_t quad_mpp_idx; p4est_topidx_t tree_mpp_idx;
  p4est_locidx_t quad_pmp_idx; p4est_topidx_t tree_pmp_idx;
  p4est_locidx_t quad_ppp_idx; p4est_topidx_t tree_ppp_idx;
#endif

#ifdef P4_TO_P8
  node_neighbors_->find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx); //nei_quads[dir::v_mmp] = quad_mmp_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx); //nei_quads[dir::v_mpp] = quad_mpp_idx;
  node_neighbors_->find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx); //nei_quads[dir::v_pmp] = quad_pmp_idx;
  node_neighbors_->find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx); //nei_quads[dir::v_ppp] = quad_ppp_idx;
#else
  node_neighbors_->find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, -1, +1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, +1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  node_neighbors_->find_neighbor_cell_of_node(n, +1, +1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
#endif

  // find neighboring nodes
#ifdef P4_TO_P8
  neighbors[nn_000] = n;

  // m00
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmm];

  // p00
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

  // 0m0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmm];

  // 0p0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

  // 00m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_pmm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mmm];

  // 00p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_ppp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_pmp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

  // 0mm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];
  // 0pm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  // 0mp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmp];
  // 0pp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

  // m0m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];
  // p0m
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];
  // m0p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmp];
  // p0p
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

  // mm0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
  // pm0
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
  // mp0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
  // pp0
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

  // mmm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmm] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  // pmm
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmm] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  // mpm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpm] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  // ppm
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppm] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

  // mmp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmp] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
  // pmp
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmp] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
  // mpp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpp] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
  // ppp
  if      (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppp] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];
#else
  neighbors[nn_000] = n;

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];

  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];

  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes_->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes_->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif
}


#ifdef P4_TO_P8
double my_p4est_poisson_jump_nodes_mls_sc_t::compute_weights_through_face(double A, double B, bool *neighbors_exists_2d, double *weights_2d, double theta, bool *map_2d)
{

  bool semi_fallback = false;
  bool full_fallback = false;

  map_2d[nn_mmm] = false;
  map_2d[nn_0mm] = false;
  map_2d[nn_pmm] = false;
  map_2d[nn_m0m] = false;
  map_2d[nn_00m] = true;  weights_2d[nn_00m] = 1;
  map_2d[nn_p0m] = false;
  map_2d[nn_mpm] = false;
  map_2d[nn_0pm] = false;
  map_2d[nn_ppm] = false;

//  weights_2d[nn_mmm] = 0;
//  weights_2d[nn_0mm] = 0;
//  weights_2d[nn_pmm] = 0;
//  weights_2d[nn_m0m] = 0;
//  weights_2d[nn_00m] = 1;
//  weights_2d[nn_p0m] = 0;
//  weights_2d[nn_mpm] = 0;
//  weights_2d[nn_0pm] = 0;
//  weights_2d[nn_ppm] = 0;

  double a = fabs(A);
  double b = fabs(B);

  if (a > .5 || b > .5) std::cout << "Warning!\n";

  if (a < theta && b < theta)
  {
    map_2d[nn_00m] = true;  weights_2d[nn_00m] = 1;
  }
//  else if ( A <= 0 &&
//            B <= 0 &&
//            neighbors_exists_2d[nn_m0m] &&
//            neighbors_exists_2d[nn_0mm] &&
//            neighbors_exists_2d[nn_mmm] )
//  {
//    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-a)*(1.-b);
//    map_2d[nn_m0m] = true; weights_2d[nn_m0m] =    a *(1.-b);
//    map_2d[nn_0mm] = true; weights_2d[nn_0mm] =(1.-a)*    b ;
//    map_2d[nn_mmm] = true; weights_2d[nn_mmm] =    a *    b ;
//  }
//  else if ( A >= 0 &&
//            B <= 0 &&
//            neighbors_exists_2d[nn_p0m] &&
//            neighbors_exists_2d[nn_0mm] &&
//            neighbors_exists_2d[nn_pmm] )
//  {
//    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-a)*(1.-b);
//    map_2d[nn_p0m] = true; weights_2d[nn_p0m] =    a *(1.-b);
//    map_2d[nn_0mm] = true; weights_2d[nn_0mm] =(1.-a)*    b ;
//    map_2d[nn_pmm] = true; weights_2d[nn_pmm] =    a *    b ;
//  }
//  else if ( A <= 0 &&
//            B >= 0 &&
//            neighbors_exists_2d[nn_m0m] &&
//            neighbors_exists_2d[nn_0pm] &&
//            neighbors_exists_2d[nn_mpm] )
//  {
//    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-a)*(1.-b);
//    map_2d[nn_m0m] = true; weights_2d[nn_m0m] =    a *(1.-b);
//    map_2d[nn_0pm] = true; weights_2d[nn_0pm] =(1.-a)*    b ;
//    map_2d[nn_mpm] = true; weights_2d[nn_mpm] =    a *    b ;
//  }
//  else if ( A >= 0 &&
//            B >= 0 &&
//            neighbors_exists_2d[nn_p0m] &&
//            neighbors_exists_2d[nn_0pm] &&
//            neighbors_exists_2d[nn_ppm] )
//  {
//    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-a)*(1.-b);
//    map_2d[nn_p0m] = true; weights_2d[nn_p0m] =    a *(1.-b);
//    map_2d[nn_0pm] = true; weights_2d[nn_0pm] =(1.-a)*    b ;
//    map_2d[nn_ppm] = true; weights_2d[nn_ppm] =    a *    b ;
//  }

  else if (A <= 0 && B <= 0 && B <= A &&
           neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b-a;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = a;
  }
  else if (A <= 0 && B <= 0 && B >= A &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a-b;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = b;
  }
  else if (A >= 0 && B <= 0 && B <= -A &&
           neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b-a;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = a;
  }
  else if (A >= 0 && B <= 0 && B >= -A &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a-b;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = b;
  }
  else if (A <= 0 && B >= 0 && B <= -A &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a-b;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = b;
  }
  else if (A <= 0 && B >= 0 && B >= -A &&
           neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b-a;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = a;
  }
  else if (A >= 0 && B >= 0 && B <= A &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a-b;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = b;
  }
  else if (A >= 0 && B >= 0 && B >= A &&
           neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b-a;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = a;
  }


  else if (A <= 0 && B <= 0 &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-b-a;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b;
  }
  else if (A >= 0 && B <= 0 &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b;
  }
  else if (A <= 0 && B >= 0 &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b;
  }
  else if (A >= 0 && B >= 0 &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b;
  }



  else if (neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B+A;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = -A;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+A);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A+B;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = -B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B-A;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = A;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-A);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A+B;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = -B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+A);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A-B;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B+A;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = -A;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-A);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A-B;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B-A;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = A;
    semi_fallback = true;
  }

  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.+A+B;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-A+B;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.+A-B;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-A-B;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B;
    semi_fallback = true;
  }


  else if (A <= 0 && neighbors_exists_2d[nn_m0m] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a;
    full_fallback = true;
  }
  else if (B <= 0 && neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-b;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b;
    full_fallback = true;
  }
  else if (B >= 0 && neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-b;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b;
    full_fallback = true;
  }
  else if (A >= 0 && neighbors_exists_2d[nn_p0m] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a;
    full_fallback = true;
  }

  else
  {
    int num_good_neighbors = 0;

    for (int i = 0; i < 9; ++i)
      if (neighbors_exists_2d[i]) num_good_neighbors++;

    if (num_good_neighbors >= 3)
    {
      std::cout << "Possible! " << num_good_neighbors << "\n";
      for (int i = 0; i < 9; ++i)
        if (neighbors_exists_2d[i])  std::cout << i << " ";
      std::cout << "\n";

      if (num_good_neighbors == 3 &&
          ( (neighbors_exists_2d[nn_m0m] && neighbors_exists_2d[nn_p0m]) ||
            (neighbors_exists_2d[nn_0mm] && neighbors_exists_2d[nn_0pm]) ||
            (neighbors_exists_2d[nn_mmm] && neighbors_exists_2d[nn_ppm]) ||
            (neighbors_exists_2d[nn_mpm] && neighbors_exists_2d[nn_pmm]) ) )
      {
        std::cout << "Same line! " << num_good_neighbors << "\n";
      }
    } else {

      if ( (a > theta || b > theta) )
      {
        std::cout << "Too little good neighbors! " << num_good_neighbors << "\n";
        for (int i = 0; i < 9; ++i)
          if (neighbors_exists_2d[i])  std::cout << i << " ";
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

//    inv_mat3_(A, A_inv);

//    weights_2d[nn_00m] = A_inv[0];
//    weights_2d[nei1] = A_inv[1];
//    weights_2d[nei2] = A_inv[2];


//    if (num_good_neighbors == 3 &&
//        ( (neighbors_exists_2d[nn_m0m] && neighbors_exists_2d[nn_p0m]) ||
//          (neighbors_exists_2d[nn_0mm] && neighbors_exists_2d[nn_0pm]) ||
//          (neighbors_exists_2d[nn_mmm] && neighbors_exists_2d[nn_ppm]) ||
//          (neighbors_exists_2d[nn_mpm] && neighbors_exists_2d[nn_pmm]) ) )
//      std::cout << "On the same line! :( \n";

  }

  if (full_fallback)
    return 5;
//    return -1;
  else if (semi_fallback)
    return -1;
//    return 2;
  else
    return -1;
}
#endif

bool my_p4est_poisson_jump_nodes_mls_sc_t::find_x_derivative(bool *neighbors_exist, double *weights, bool *map)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_000] && neighbors_exist[nn_m00]) { weights[nn_000] = 1.; weights[nn_m00] =-1.; map[nn_000] = true; map[nn_m00] = true; }
  else if (neighbors_exist[nn_000] && neighbors_exist[nn_p00]) { weights[nn_000] =-1.; weights[nn_p00] = 1.; map[nn_000] = true; map[nn_p00] = true; }

  // check m0
  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_mm0]) { weights[nn_0m0] = 1.; weights[nn_mm0] =-1.; map[nn_0m0] = true; map[nn_mm0] = true; }
  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_pm0]) { weights[nn_0m0] =-1.; weights[nn_pm0] = 1.; map[nn_0m0] = true; map[nn_pm0] = true; }

  // check p0
  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_mp0]) { weights[nn_0p0] = 1.; weights[nn_mp0] =-1.; map[nn_0p0] = true; map[nn_mp0] = true; }
  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_pp0]) { weights[nn_0p0] =-1.; weights[nn_pp0] = 1.; map[nn_0p0] = true; map[nn_pp0] = true; }
#ifdef P4_TO_P8
  // check 0m
  else if (neighbors_exist[nn_00m] && neighbors_exist[nn_m0m]) { weights[nn_00m] = 1.; weights[nn_m0m] =-1.; map[nn_00m] = true; map[nn_m0m] = true; }
  else if (neighbors_exist[nn_00m] && neighbors_exist[nn_p0m]) { weights[nn_00m] =-1.; weights[nn_p0m] = 1.; map[nn_00m] = true; map[nn_p0m] = true; }

  // check 0p
  else if (neighbors_exist[nn_00p] && neighbors_exist[nn_m0p]) { weights[nn_00p] = 1.; weights[nn_m0p] =-1.; map[nn_00p] = true; map[nn_m0p] = true; }
  else if (neighbors_exist[nn_00p] && neighbors_exist[nn_p0p]) { weights[nn_00p] =-1.; weights[nn_p0p] = 1.; map[nn_00p] = true; map[nn_p0p] = true; }

  // check mm
  else if (neighbors_exist[nn_0mm] && neighbors_exist[nn_mmm]) { weights[nn_0mm] = 1.; weights[nn_mmm] =-1.; map[nn_0mm] = true; map[nn_mmm] = true; }
  else if (neighbors_exist[nn_0mm] && neighbors_exist[nn_pmm]) { weights[nn_0mm] =-1.; weights[nn_pmm] = 1.; map[nn_0mm] = true; map[nn_pmm] = true; }

  // check pm
  else if (neighbors_exist[nn_0pm] && neighbors_exist[nn_mpm]) { weights[nn_0pm] = 1.; weights[nn_mpm] =-1.; map[nn_0pm] = true; map[nn_mpm] = true; }
  else if (neighbors_exist[nn_0pm] && neighbors_exist[nn_ppm]) { weights[nn_0pm] =-1.; weights[nn_ppm] = 1.; map[nn_0pm] = true; map[nn_ppm] = true; }

  // check mp
  else if (neighbors_exist[nn_0mp] && neighbors_exist[nn_mmp]) { weights[nn_0mp] = 1.; weights[nn_mmp] =-1.; map[nn_0mp] = true; map[nn_mmp] = true; }
  else if (neighbors_exist[nn_0mp] && neighbors_exist[nn_pmp]) { weights[nn_0mp] =-1.; weights[nn_pmp] = 1.; map[nn_0mp] = true; map[nn_pmp] = true; }

  // check pp
  else if (neighbors_exist[nn_0pp] && neighbors_exist[nn_mpp]) { weights[nn_0pp] = 1.; weights[nn_mpp] =-1.; map[nn_0pp] = true; map[nn_mpp] = true; }
  else if (neighbors_exist[nn_0pp] && neighbors_exist[nn_ppp]) { weights[nn_0pp] =-1.; weights[nn_ppp] = 1.; map[nn_0pp] = true; map[nn_ppp] = true; }
#endif
  else
    derivative_found = false;

  return derivative_found;
}

bool my_p4est_poisson_jump_nodes_mls_sc_t::find_y_derivative(bool *neighbors_exist, double *weights, bool *map)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_000] && neighbors_exist[nn_0m0]) { weights[nn_000] = 1.; weights[nn_0m0] =-1.; map[nn_000] = true; map[nn_0m0] = true; }
  else if (neighbors_exist[nn_000] && neighbors_exist[nn_0p0]) { weights[nn_000] =-1.; weights[nn_0p0] = 1.; map[nn_000] = true; map[nn_0p0] = true; }
  // check 0m
  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_mm0]) { weights[nn_m00] = 1.; weights[nn_mm0] =-1.; map[nn_m00] = true; map[nn_mm0] = true; }
  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_mp0]) { weights[nn_m00] =-1.; weights[nn_mp0] = 1.; map[nn_m00] = true; map[nn_mp0] = true; }

  // check 0p
  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_pm0]) { weights[nn_p00] = 1.; weights[nn_pm0] =-1.; map[nn_p00] = true; map[nn_pm0] = true; }
  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_pp0]) { weights[nn_p00] =-1.; weights[nn_pp0] = 1.; map[nn_p00] = true; map[nn_pp0] = true; }

#ifdef P4_TO_P8
  // check m0
  else if (neighbors_exist[nn_00m] && neighbors_exist[nn_0mm]) { weights[nn_00m] = 1.; weights[nn_0mm] =-1.; map[nn_00m] = true; map[nn_0mm] = true; }
  else if (neighbors_exist[nn_00m] && neighbors_exist[nn_0pm]) { weights[nn_00m] =-1.; weights[nn_0pm] = 1.; map[nn_00m] = true; map[nn_0pm] = true; }

  // check p0
  else if (neighbors_exist[nn_00p] && neighbors_exist[nn_0mp]) { weights[nn_00p] = 1.; weights[nn_0mp] =-1.; map[nn_00p] = true; map[nn_0mp] = true; }
  else if (neighbors_exist[nn_00p] && neighbors_exist[nn_0pp]) { weights[nn_00p] =-1.; weights[nn_0pp] = 1.; map[nn_00p] = true; map[nn_0pp] = true; }

  // check mm
  else if (neighbors_exist[nn_m0m] && neighbors_exist[nn_mmm]) { weights[nn_m0m] = 1.; weights[nn_mmm] =-1.; map[nn_m0m] = true; map[nn_mmm] = true; }
  else if (neighbors_exist[nn_m0m] && neighbors_exist[nn_mpm]) { weights[nn_m0m] =-1.; weights[nn_mpm] = 1.; map[nn_m0m] = true; map[nn_mpm] = true; }

  // check pm
  else if (neighbors_exist[nn_m0p] && neighbors_exist[nn_mmp]) { weights[nn_m0p] = 1.; weights[nn_mmp] =-1.; map[nn_m0p] = true; map[nn_mmp] = true; }
  else if (neighbors_exist[nn_m0p] && neighbors_exist[nn_mpp]) { weights[nn_m0p] =-1.; weights[nn_mpp] = 1.; map[nn_m0p] = true; map[nn_mpp] = true; }

  // check mp
  else if (neighbors_exist[nn_p0m] && neighbors_exist[nn_pmm]) { weights[nn_p0m] = 1.; weights[nn_pmm] =-1.; map[nn_p0m] = true; map[nn_pmm] = true; }
  else if (neighbors_exist[nn_p0m] && neighbors_exist[nn_ppm]) { weights[nn_p0m] =-1.; weights[nn_ppm] = 1.; map[nn_p0m] = true; map[nn_ppm] = true; }

  // check pp
  else if (neighbors_exist[nn_p0p] && neighbors_exist[nn_pmp]) { weights[nn_p0p] = 1.; weights[nn_pmp] =-1.; map[nn_p0p] = true; map[nn_pmp] = true; }
  else if (neighbors_exist[nn_p0p] && neighbors_exist[nn_ppp]) { weights[nn_p0p] =-1.; weights[nn_ppp] = 1.; map[nn_p0p] = true; map[nn_ppp] = true; }
#endif
  else
    derivative_found = false;

  return derivative_found;
}

#ifdef P4_TO_P8
bool my_p4est_poisson_jump_nodes_mls_sc_t::find_z_derivative(bool *neighbors_exist, double *weights, bool *map)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_000] && neighbors_exist[nn_00m]) { weights[nn_000] = 1.; weights[nn_00m] =-1.; map[nn_000] = true; map[nn_00m] = true; }
  else if (neighbors_exist[nn_000] && neighbors_exist[nn_00p]) { weights[nn_000] =-1.; weights[nn_00p] = 1.; map[nn_000] = true; map[nn_00p] = true; }

  // check m0
  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_m0m]) { weights[nn_m00] = 1.; weights[nn_m0m] =-1.; map[nn_m00] = true; map[nn_m0m] = true; }
  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_m0p]) { weights[nn_m00] =-1.; weights[nn_m0p] = 1.; map[nn_m00] = true; map[nn_m0p] = true; }

  // check p0
  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_p0m]) { weights[nn_p00] = 1.; weights[nn_p0m] =-1.; map[nn_p00] = true; map[nn_p0m] = true; }
  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_p0p]) { weights[nn_p00] =-1.; weights[nn_p0p] = 1.; map[nn_p00] = true; map[nn_p0p] = true; }
  // check 0m
  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_0mm]) { weights[nn_0m0] = 1.; weights[nn_0mm] =-1.; map[nn_0m0] = true; map[nn_0mm] = true; }
  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_0mp]) { weights[nn_0m0] =-1.; weights[nn_0mp] = 1.; map[nn_0m0] = true; map[nn_0mp] = true; }

  // check 0p
  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_0pm]) { weights[nn_0p0] = 1.; weights[nn_0pm] =-1.; map[nn_0p0] = true; map[nn_0pm] = true; }
  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_0pp]) { weights[nn_0p0] =-1.; weights[nn_0pp] = 1.; map[nn_0p0] = true; map[nn_0pp] = true; }

  // check mm
  else if (neighbors_exist[nn_mm0] && neighbors_exist[nn_mmm]) { weights[nn_mm0] = 1.; weights[nn_mmm] =-1.; map[nn_mm0] = true; map[nn_mmm] = true; }
  else if (neighbors_exist[nn_mm0] && neighbors_exist[nn_mmp]) { weights[nn_mm0] =-1.; weights[nn_mmp] = 1.; map[nn_mm0] = true; map[nn_mmp] = true; }

  // check pm
  else if (neighbors_exist[nn_pm0] && neighbors_exist[nn_pmm]) { weights[nn_pm0] = 1.; weights[nn_pmm] =-1.; map[nn_pm0] = true; map[nn_pmm] = true; }
  else if (neighbors_exist[nn_pm0] && neighbors_exist[nn_pmp]) { weights[nn_pm0] =-1.; weights[nn_pmp] = 1.; map[nn_pm0] = true; map[nn_pmp] = true; }

  // check mp
  else if (neighbors_exist[nn_mp0] && neighbors_exist[nn_mpm]) { weights[nn_mp0] = 1.; weights[nn_mpm] =-1.; map[nn_mp0] = true; map[nn_mpm] = true; }
  else if (neighbors_exist[nn_mp0] && neighbors_exist[nn_mpp]) { weights[nn_mp0] =-1.; weights[nn_mpp] = 1.; map[nn_mp0] = true; map[nn_mpp] = true; }

  // check pp
  else if (neighbors_exist[nn_pp0] && neighbors_exist[nn_ppm]) { weights[nn_pp0] = 1.; weights[nn_ppm] =-1.; map[nn_pp0] = true; map[nn_ppm] = true; }
  else if (neighbors_exist[nn_pp0] && neighbors_exist[nn_ppp]) { weights[nn_pp0] =-1.; weights[nn_ppp] = 1.; map[nn_pp0] = true; map[nn_ppp] = true; }
  else
    derivative_found = false;

  return derivative_found;
}
#endif

//void my_p4est_poisson_jump_nodes_mls_sc_t::assemble_matrix(Vec solution)
//{
//#ifdef CASL_THROWS
//  if(bc_ == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

//  {
//    PetscInt sol_size;
//    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
//    if (sol_size != nodes->num_owned_indeps){
//      std::ostringstream oss;
//      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
//          << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes->num_owned_indeps << std::endl;
//      throw std::invalid_argument(oss.str());
//    }
//  }
//#endif

//  // set local add if none was given
//  bool local_add = false;
//  if(add_ == NULL)
//  {
//    local_add = true;
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
//    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
//  }

//  // set a local phi if not was given
//  bool local_phi = false;
//  if(phi_ == NULL)
//  {
//    local_phi = true;
//    ierr = VecDuplicate(solution, &phi_); CHKERRXX(ierr);

//    Vec tmp;
//    ierr = VecGhostGetLocalForm(phi_, &tmp); CHKERRXX(ierr);
//    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(phi_, &tmp); CHKERRXX(ierr);
////    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
//    set_phi(phi_);
//  }

//  matrix_has_nullspace = true;
//  setup_negative_variable_coeff_laplace_matrix();
//  is_matrix_computed = true;
//  new_pc = true;

//  // get rid of local stuff
//  if(local_add)
//  {
//    ierr = VecDestroy(add_); CHKERRXX(ierr);
//    add_ = NULL;
//  }
//  if(local_phi)
//  {
//    ierr = VecDestroy(phi_); CHKERRXX(ierr);
//    phi_ = NULL;

//    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
//    phi_xx_ = NULL;

//    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
//    phi_yy_ = NULL;

//#ifdef P4_TO_P8
//    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
//    phi_zz_ = NULL;
//#endif
//  }
//}

////void my_p4est_poisson_jump_nodes_mls_sc_t::assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p)
//void my_p4est_poisson_jump_nodes_mls_sc_t::assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in, Vec rhs_p_in)
//{

//  // set local add if none was given
//  bool local_add = false;
//  if(add_ == NULL)
//  {
//    local_add = true;
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
//    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
//  }

//  bool local_rhs = false;
//  Vec rhs_m, rhs_p;
//  if(rhs_m_in == NULL && rhs_p_in == NULL)
//  {
//    local_rhs = true;
//    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_m); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_p); CHKERRXX(ierr);
////    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &rhs_m); CHKERRXX(ierr);
////    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &rhs_p); CHKERRXX(ierr);
//    ierr = VecSet(rhs_m, 0); CHKERRXX(ierr);
//    ierr = VecSet(rhs_p, 0); CHKERRXX(ierr);
//  } else {
//    rhs_m = rhs_m_in;
//    rhs_p = rhs_p_in;
//  }

//  if(phi_ == NULL)
//    throw std::domain_error("[CASL_ERROR]: no interface to impose jump conditions on.");

//  if (variable_mu)
//    throw std::domain_error("[CASL_ERROR]: simple jump solver works only for const mu at the moment.");

//  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
//  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  double *phi_zz_p;
//  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

//  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);

//  double *rhs_m_p; ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
//  double *rhs_p_p; ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);

//  double *rhs_out_p;
//  ierr = VecGetArray(rhs_out, &rhs_out_p); CHKERRXX(ierr);

//  for (size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
//    rhs_out_p[n] = 0;

//  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);
//  phi_interp_local.set_input(phi_p, phi_xx_p, phi_yy_p, quadratic);

//  /* Daniil: while working on solidification of alloys I came to realize
//   * that due to very irregular structure of solidification fronts (where
//   * there constantly exist nodes, which belong to one phase but the vertices
//   * of their dual cells belong to the other phase. In such cases simple dual
//   * cells consisting of 2 (6 in 3D) simplices don't capture such complex
//   * topologies. To aleviate this issue I added an option to use a more detailed
//   * dual cells consisting of 8 (48 in 3D) simplices. Clearly, it might increase
//   * the computational cost quite significantly, but oh well...
//   */

//  // data for refined cells
//  std::vector<double> phi_fv(pow(2*cube_refinement+1,P4EST_DIM),-1);
//  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement+1, 0);
//  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement+1, 0);
//#ifdef P4_TO_P8
//  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement+1, 0);
//#endif

//  double fv_xmin, fv_xmax;
//  double fv_ymin, fv_ymax;
//#ifdef P4_TO_P8
//  double fv_zmin, fv_zmax;
//#endif

//  double xyz_C[P4EST_DIM];

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

//    node_xyz_fr_n(n, p4est, nodes, xyz_C);

//    double x_C  = xyz_C[0];
//    double y_C  = xyz_C[1];
//#ifdef P4_TO_P8
//    double z_C  = xyz_C[2];
//#endif

//    if (phi_p[n] > 2.*diag_min)

//      rhs_out_p[n] += rhs_p_p[n];
////    rhs_out_p[n] += rhs_p(xyz_C[0], xyz_C[1]);

//    else if (phi_p[n] < -2.*diag_min)

//      rhs_out_p[n] += rhs_m_p[n];
////    rhs_out_p[n] += rhs_p(xyz_C[0], xyz_C[1]);

//    else {

//      //---------------------------------------------------------------------
//      // check if finite volume is crossed
//      //---------------------------------------------------------------------
//      bool is_one_positive = false;
//      bool is_one_negative = false;

//      phi_interp_local.initialize(n);
//      // determine dimensions of cube
//      fv_size_x = 0;
//      fv_size_y = 0;
//#ifdef P4_TO_P8
//      fv_size_z = 0;
//#endif
//      if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dx_min;} else {fv_xmin = x_C;}
//      if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dx_min;} else {fv_xmax = x_C;}

//      if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dy_min;} else {fv_ymin = y_C;}
//      if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dy_min;} else {fv_ymax = y_C;}
//#ifdef P4_TO_P8
//      if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dz_min;} else {fv_zmin = z_C;}
//      if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dz_min;} else {fv_zmax = z_C;}
//#endif

//      if (!use_refined_cube) {
//        fv_size_x = 1;
//        fv_size_y = 1;
//#ifdef P4_TO_P8
//        fv_size_z = 1;
//#endif
//      }

//      fv_nx = fv_size_x+1;
//      fv_ny = fv_size_y+1;
//#ifdef P4_TO_P8
//      fv_nz = fv_size_z+1;
//#endif

//      // get coordinates of cube nodes
//      double fv_dx = (fv_xmax-fv_xmin)/ (double)(fv_size_x);
//      fv_x[0] = fv_xmin;
//      for (short i = 1; i < fv_nx; ++i)
//        fv_x[i] = fv_x[i-1] + fv_dx;

//      double fv_dy = (fv_ymax-fv_ymin)/ (double)(fv_size_y);
//      fv_y[0] = fv_ymin;
//      for (short i = 1; i < fv_ny; ++i)
//        fv_y[i] = fv_y[i-1] + fv_dy;
//#ifdef P4_TO_P8
//      double fv_dx = (fv_zmax-fv_zmin)/ (double)(fv_size_z);
//      fv_z[0] = fv_zmin;
//      for (short i = 1; i < fv_nz; ++i)
//        fv_z[i] = fv_z[i-1] + fv_dz;
//#endif

//      // sample level-set function at cube nodes and check if crossed
//#ifdef P4_TO_P8
//      for (short k = 0; k < fv_nz; ++k)
//#endif
//        for (short j = 0; j < fv_ny; ++j)
//          for (short i = 0; i < fv_nx; ++i)
//          {
//#ifdef P4_TO_P8
//            int idx = k*fv_nx*fv_ny + j*fv_nx + i;
//            phi_fv[idx] = phi_interp(fv_x[i], fv_y[j], fv_z[k]);
//            phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j], fv_z[k]);
//#else
//            int idx = j*fv_nx + i;
//            //              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
//            phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j]);
//#endif
//            is_one_positive = is_one_positive || phi_fv[idx] > 0;
//            is_one_negative = is_one_negative || phi_fv[idx] < 0;
//          }


//      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

//      if (!is_ngbd_crossed_neumann)
//      {
//        if (phi_p[n] > 0.)  rhs_out_p[n] += rhs_p_p[n];
//        else                rhs_out_p[n] += rhs_m_p[n];
////        if (phi_p[n] > 0.)  rhs_out_p[n] += rhs_p(xyz_C[0], xyz_C[1]);
////        else                rhs_out_p[n] += rhs_m(xyz_C[0], xyz_C[1]);
//      } else {

//        const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

//        double volume_cut_cell = 0.;
//        double interface_area  = 0.;
//        double integral_bc     = 0.;

//#ifdef P4_TO_P8
//        p4est_locidx_t n_m00 = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
//                                                : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp) ;
//        p4est_locidx_t n_p00 = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
//                                                : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp) ;
//        p4est_locidx_t n_0m0 = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
//                                                : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp) ;
//        p4est_locidx_t n_0p0 = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
//                                                : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp) ;
//        p4est_locidx_t n_00m = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
//                                                : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp) ;
//        p4est_locidx_t n_00p = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
//                                                : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp) ;

//        double volume_full_cell = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin)*(fv_zmax-fv_zmin);

//        Cube3 cube;
//        OctValue  phi_cube;

//        for (short k = 0; k < fv_size_z; ++k)
//          for (short j = 0; j < fv_size_y; ++j)
//            for (short i = 0; i < fv_size_x; ++i)
//            {
//              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
//              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
//              cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

//              phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
//              phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
//              phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
//              phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
//              phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
//              phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
//              phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
//              phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

//              volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
//              interface_area  += cube.interface_Length_In_Cell(phi_cube);
//              integral_bc += cube.integrate_Over_Interface(jump_un, phi_cube);
//            }
//#else
//        p4est_locidx_t n_m00 = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm;
//        p4est_locidx_t n_p00 = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm;
//        p4est_locidx_t n_0m0 = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
//        p4est_locidx_t n_0p0 = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;

//        double volume_full_cell = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin);
//        Cube2 cube;
//        QuadValue phi_cube;

//        for (short j = 0; j < fv_size_y; ++j)
//          for (short i = 0; i < fv_size_x; ++i)
//          {
//            cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
//            cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

//            phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
//            phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
//            phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
//            phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

//            volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
//            interface_area  += cube.interface_Length_In_Cell(phi_cube);
//            integral_bc += cube.integrate_Over_Interface(jump_un, phi_cube);
//          }

//#endif
//        double volume_cut_cell_m = volume_cut_cell;
//        double volume_cut_cell_p = volume_full_cell - volume_cut_cell;

//#ifdef P4_TO_P8
//        Cube2 c2;
//        QuadValue qv;

//        double s_m00 = 0, s_p00 = 0;
//        for (short k = 0; k < fv_size_z; ++k)
//          for (short j = 0; j < fv_size_y; ++j) {

//            c2.x0 = fv_y[j]; c2.x1 = fv_y[j+1];
//            c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

//            int i = 0;
//            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
//            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

//            s_m00 += c2.area_In_Negative_Domain(qv);

//            i = fv_size_x;
//            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
//            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

//            s_p00 += c2.area_In_Negative_Domain(qv);
//          }

//        double s_0m0 = 0, s_0p0 = 0;
//        for (short k = 0; k < fv_size_z; ++k)
//          for (short i = 0; i < fv_size_x; ++i) {

//            c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
//            c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

//            int j = 0;
//            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
//            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

//            s_0m0 += c2.area_In_Negative_Domain(qv);

//            j = fv_size_y;
//            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
//            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

//            s_0p0 += c2.area_In_Negative_Domain(qv);
//          }

//        double s_00m = 0, s_00p = 0;
//        for (short j = 0; j < fv_size_j; ++j)
//          for (short i = 0; i < fv_size_x; ++i) {

//            c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
//            c2.y0 = fv_y[j]; c2.y1 = fv_y[j+1];

//            int k = 0;
//            qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
//            qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
//            qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
//            qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

//            s_00m += c2.area_In_Negative_Domain(qv);

//            k = fv_size_z;
//            qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
//            qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
//            qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
//            qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

//            s_00p += c2.area_In_Negative_Domain(qv);
//          }
//#else
//        double s_m00 = 0, s_p00 = 0;
//        for (short j = 0; j < fv_size_y; ++j) {
//          int i;
//          i = 0;          s_m00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
//          i = fv_size_x;  s_p00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
//        }

//        double s_0m0 = 0, s_0p0 = 0;
//        for (short i = 0; i < fv_size_x; ++i) {
//          int j;
//          j = 0;          s_0m0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
//          j = fv_size_y;  s_0p0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
//        }
//#endif

//        double s_m00_m = s_m00, s_m00_p = (fv_xmax-fv_xmin) - s_m00;
//        double s_p00_m = s_p00, s_p00_p = (fv_xmax-fv_xmin) - s_p00;
//        double s_0m0_m = s_0m0, s_0m0_p = (fv_ymax-fv_ymin) - s_0m0;
//        double s_0p0_m = s_0p0, s_0p0_p = (fv_ymax-fv_ymin) - s_0p0;
//#ifdef P4_TO_P8
//        double s_00m_m = s_00m, s_00m_p = (fv_zmax-fv_zmin) - s_00m;
//        double s_00p_m = s_00p, s_00p_p = (fv_zmax-fv_zmin) - s_00p;
//#endif

//        double xyz_p[P4EST_DIM];
//        double normal[P4EST_DIM];

//        normal[0] = qnnn.dx_central(phi_p);
//        normal[1] = qnnn.dy_central(phi_p);
//#ifdef P4_TO_P8
//        normal[2] = qnnn.dz_central(phi_p);
//        double norm = sqrt(SQR(normal[0])+SQR(normal[1])+SQR(normal[2]));
//#else
//        double norm = sqrt(SQR(normal[0])+SQR(normal[1]));
//#endif

//        for (short dir = 0; dir < P4EST_DIM; ++dir)
//        {
//          xyz_p[dir] = xyz_C[dir] - phi_p[n]*normal[dir]/norm;

//          if      (xyz_p[dir] < xyz_C[dir]-dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]-dxyz_m[dir]+EPS;
//          else if (xyz_p[dir] > xyz_C[dir]+dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]+dxyz_m[dir]+EPS;
//        }

//        double alpha_proj = jump_u(xyz_p[0],xyz_p[1]);
//        double beta_proj = jump_un(xyz_p[0],xyz_p[1]);

////        double rhs_m_val = rhs_m(xyz_p[0],xyz_p[1]);
////        double rhs_p_val = rhs_p(xyz_p[0],xyz_p[1]);
//        double rhs_m_val = rhs_m_p[n];
//        double rhs_p_val = rhs_p_p[n];

//        rhs_out_p[n] += (rhs_m_val*volume_cut_cell_m + rhs_p_val*volume_cut_cell_p + mu_*integral_bc)/volume_full_cell;

//        if (phi_p[n] < 0.) {
//          double factor = (fabs(phi_p[n])*beta_proj - alpha_proj)/volume_full_cell;

//          rhs_out_p[n] -= factor*(add_p[n]*volume_cut_cell_p + mu_*((s_m00_p+s_p00_p)/dx_min+(s_0m0_p+s_0p0_p)/dy_min));
//          rhs_out_p[n_m00] += mu_*s_m00_p*factor/dx_min;
//          rhs_out_p[n_p00] += mu_*s_p00_p*factor/dx_min;
//          rhs_out_p[n_0m0] += mu_*s_0m0_p*factor/dy_min;
//          rhs_out_p[n_0p0] += mu_*s_0p0_p*factor/dy_min;
//        } else {
//          double factor = (fabs(phi_p[n])*beta_proj + alpha_proj)/volume_full_cell;

//          rhs_out_p[n] -= factor*(add_p[n]*volume_cut_cell_m + mu_*((s_m00_m+s_p00_m)/dx_min+(s_0m0_m+s_0p0_m)/dy_min));
//          rhs_out_p[n_m00] += mu_*s_m00_m*factor/dx_min;
//          rhs_out_p[n_p00] += mu_*s_p00_m*factor/dx_min;
//          rhs_out_p[n_0m0] += mu_*s_0m0_m*factor/dy_min;
//          rhs_out_p[n_0p0] += mu_*s_0p0_m*factor/dy_min;
//        }
//      }
//    }
//  }

//  // restore pointers
//  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

//  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

//  ierr = VecRestoreArray(rhs_out, &rhs_out_p   ); CHKERRXX(ierr);

//  ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);

//  // get rid of local stuff
//  if(local_add)
//  {
//    ierr = VecDestroy(add_); CHKERRXX(ierr);
//    add_ = NULL;
//  }

//  if(local_rhs)
//  {
//    ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
//    ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
//  }


//  // update ghosts
//  ierr = VecGhostUpdateBegin(rhs_out, ADD_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(rhs_out, ADD_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);

//}





//#ifdef P4_TO_P8
//          std::vector< cube2_mls_quadratic_t * > cubes_m00(fv_size_y*fv_size_z, NULL);
//          std::vector< cube2_mls_quadratic_t * > cubes_p00(fv_size_y*fv_size_z, NULL);

//          std::vector< cube2_mls_quadratic_t * > cubes_0m0(fv_size_x*fv_size_z, NULL);
//          std::vector< cube2_mls_quadratic_t * > cubes_0p0(fv_size_x*fv_size_z, NULL);

//          std::vector< cube2_mls_quadratic_t * > cubes_00m(fv_size_x*fv_size_y, NULL);
//          std::vector< cube2_mls_quadratic_t * > cubes_00p(fv_size_x*fv_size_y, NULL);

//          std::vector< restriction_to_yz_t > phi_interp_local_m00;
//          std::vector< restriction_to_yz_t > phi_interp_local_p00;
//          std::vector< restriction_to_zx_t > phi_interp_local_0m0;
//          std::vector< restriction_to_zx_t > phi_interp_local_0p0;
//          std::vector< restriction_to_xy_t > phi_interp_local_00m;
//          std::vector< restriction_to_xy_t > phi_interp_local_00p;

//          for (int i = 0; i < num_interfaces_; ++i)
//          {
//            phi_interp_local_m00.push_back(restriction_to_yz_t(phi_interp_local_cf[i], fv_xmin));
//            phi_interp_local_p00.push_back(restriction_to_yz_t(phi_interp_local_cf[i], fv_xmax));

//            phi_interp_local_0m0.push_back(restriction_to_zx_t(phi_interp_local_cf[i], fv_ymin));
//            phi_interp_local_0p0.push_back(restriction_to_zx_t(phi_interp_local_cf[i], fv_ymax));

//            phi_interp_local_00m.push_back(restriction_to_xy_t(phi_interp_local_cf[i], fv_zmin));
//            phi_interp_local_00p.push_back(restriction_to_xy_t(phi_interp_local_cf[i], fv_zmax));
//          }

//          std::vector< CF_2 *> phi_interp_local_m00_cf(num_interfaces_, NULL);
//          std::vector< CF_2 *> phi_interp_local_p00_cf(num_interfaces_, NULL);
//          std::vector< CF_2 *> phi_interp_local_0m0_cf(num_interfaces_, NULL);
//          std::vector< CF_2 *> phi_interp_local_0p0_cf(num_interfaces_, NULL);
//          std::vector< CF_2 *> phi_interp_local_00m_cf(num_interfaces_, NULL);
//          std::vector< CF_2 *> phi_interp_local_00p_cf(num_interfaces_, NULL);

//          for (int i = 0; i < num_interfaces_; ++i)
//          {
//            phi_interp_local_m00_cf[i] = &phi_interp_local_m00[i];
//            phi_interp_local_p00_cf[i] = &phi_interp_local_p00[i];
//            phi_interp_local_0m0_cf[i] = &phi_interp_local_0m0[i];
//            phi_interp_local_0p0_cf[i] = &phi_interp_local_0p0[i];
//            phi_interp_local_00m_cf[i] = &phi_interp_local_00m[i];
//            phi_interp_local_00p_cf[i] = &phi_interp_local_00p[i];
//          }

//          restriction_to_yz_t delta_y_m00(&delta_y_cf_, fv_xmin);
//          restriction_to_yz_t delta_y_p00(&delta_y_cf_, fv_xmax);
//          restriction_to_yz_t delta_z_m00(&delta_z_cf_, fv_xmin);
//          restriction_to_yz_t delta_z_p00(&delta_z_cf_, fv_xmax);

//          restriction_to_zx_t delta_x_0m0(&delta_x_cf_, fv_ymin);
//          restriction_to_zx_t delta_x_0p0(&delta_x_cf_, fv_ymax);
//          restriction_to_zx_t delta_z_0m0(&delta_z_cf_, fv_ymin);
//          restriction_to_zx_t delta_z_0p0(&delta_z_cf_, fv_ymax);

//          restriction_to_xy_t delta_x_00m(&delta_x_cf_, fv_zmin);
//          restriction_to_xy_t delta_x_00p(&delta_x_cf_, fv_zmax);
//          restriction_to_xy_t delta_y_00m(&delta_y_cf_, fv_zmin);
//          restriction_to_xy_t delta_y_00p(&delta_y_cf_, fv_zmax);

//          restriction_to_xy_t unity_cf_2(&unity_cf_, 0);

//          double s_m00 = 0, s_p00 = 0;
//          double y_m00 = 0, y_p00 = 0;
//          double z_m00 = 0, z_p00 = 0;

//          for (short k = 0; k < fv_size_z; ++k)
//            for (short j = 0; j < fv_size_y; ++j)
//            {
//              int idx = k*fv_size_y + j;

//              cubes_m00[idx] = new cube2_mls_quadratic_t ( fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
//              cubes_p00[idx] = new cube2_mls_quadratic_t ( fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );

//              cubes_m00[idx]->construct_domain(phi_interp_local_m00_cf, *action_, *color_);
//              cubes_p00[idx]->construct_domain(phi_interp_local_p00_cf, *action_, *color_);

//              s_m00 += cubes_m00[idx]->integrate_over_domain(unity_cf_2);
//              s_p00 += cubes_p00[idx]->integrate_over_domain(unity_cf_2);

//              y_m00 += cubes_m00[idx]->integrate_over_domain(delta_y_m00);
//              y_p00 += cubes_p00[idx]->integrate_over_domain(delta_y_p00);

//              z_m00 += cubes_m00[idx]->integrate_over_domain(delta_z_m00);
//              z_p00 += cubes_p00[idx]->integrate_over_domain(delta_z_p00);
//            }

//          if (s_m00/full_sx > 0*EPS) y_m00 /= s_m00; else y_m00 = 0;
//          if (s_p00/full_sx > 0*EPS) y_p00 /= s_p00; else y_p00 = 0;
//          if (s_m00/full_sx > 0*EPS) z_m00 /= s_m00; else z_m00 = 0;
//          if (s_p00/full_sx > 0*EPS) z_p00 /= s_p00; else z_p00 = 0;

//          double s_0m0 = 0, s_0p0 = 0;
//          double x_0m0 = 0, x_0p0 = 0;
//          double z_0m0 = 0, z_0p0 = 0;
//          for (short k = 0; k < fv_size_z; ++k)
//            for (short i = 0; i < fv_size_x; ++i)
//            {
//              int idx = i*fv_size_z + k;

//              cubes_0m0[idx] = new cube2_mls_quadratic_t ( fv_z[k], fv_z[k+1], fv_x[i], fv_x[i+1] );
//              cubes_0p0[idx] = new cube2_mls_quadratic_t ( fv_z[k], fv_z[k+1], fv_x[i], fv_x[i+1] );

//              cubes_0m0[idx]->construct_domain(phi_interp_local_0m0_cf, *action_, *color_);
//              cubes_0p0[idx]->construct_domain(phi_interp_local_0p0_cf, *action_, *color_);

//              s_0m0 += cubes_0m0[idx]->integrate_over_domain(unity_cf_2);
//              s_0p0 += cubes_0p0[idx]->integrate_over_domain(unity_cf_2);

//              x_0m0 += cubes_0m0[idx]->integrate_over_domain(delta_x_0m0);
//              x_0p0 += cubes_0p0[idx]->integrate_over_domain(delta_x_0p0);

//              z_0m0 += cubes_0m0[idx]->integrate_over_domain(delta_z_0m0);
//              z_0p0 += cubes_0p0[idx]->integrate_over_domain(delta_z_0p0);
//            }

//          if (s_0m0/full_sy > 0*EPS) x_0m0 /= s_0m0; else x_0m0 = 0;
//          if (s_0p0/full_sy > 0*EPS) x_0p0 /= s_0p0; else x_0p0 = 0;
//          if (s_0m0/full_sy > 0*EPS) z_0m0 /= s_0m0; else z_0m0 = 0;
//          if (s_0p0/full_sy > 0*EPS) z_0p0 /= s_0p0; else z_0p0 = 0;

//          double s_00m = 0, s_00p = 0;
//          double x_00m = 0, x_00p = 0;
//          double y_00m = 0, y_00p = 0;
//          for (short j = 0; j < fv_size_y; ++j)
//            for (short i = 0; i < fv_size_x; ++i)
//            {
//              int idx = j*fv_size_x + j;

//              cubes_00m[idx] = new cube2_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
//              cubes_00p[idx] = new cube2_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );

//              cubes_00m[idx]->construct_domain(phi_interp_local_00m_cf, *action_, *color_);
//              cubes_00p[idx]->construct_domain(phi_interp_local_00p_cf, *action_, *color_);

//              s_00m += cubes_00m[idx]->integrate_over_domain(unity_cf_2);
//              s_00p += cubes_00p[idx]->integrate_over_domain(unity_cf_2);

//              x_00m += cubes_00m[idx]->integrate_over_domain(delta_x_00m);
//              x_00p += cubes_00p[idx]->integrate_over_domain(delta_x_00p);

//              y_00m += cubes_00m[idx]->integrate_over_domain(delta_y_00m);
//              y_00p += cubes_00p[idx]->integrate_over_domain(delta_y_00p);
//            }

//          if (s_00m/full_sz > 0*EPS) x_00m /= s_00m; else x_00m = 0;
//          if (s_00p/full_sz > 0*EPS) x_00p /= s_00p; else x_00p = 0;
//          if (s_00m/full_sz > 0*EPS) y_00m /= s_00m; else y_00m = 0;
//          if (s_00p/full_sz > 0*EPS) y_00p /= s_00p; else y_00p = 0;

//          for (int i = 0; i < fv_size_y*fv_size_z; ++i)
//          {
//            delete cubes_m00[i];
//            delete cubes_p00[i];
//          }

//          for (int i = 0; i < fv_size_x*fv_size_z; ++i)
//          {
//            delete cubes_0m0[i];
//            delete cubes_0p0[i];
//          }

//          for (int i = 0; i < fv_size_y*fv_size_x; ++i)
//          {
//            delete cubes_00m[i];
//            delete cubes_00p[i];
//          }
//#else
//          double s_m00 = 0, s_p00 = 0;
//          double y_m00 = 0, y_p00 = 0;
//            for (short j = 0; j < fv_size_y; ++j)
//            {
//              int i_m = 0;
//              int i_p = fv_size_x-1.;

//              int idx_m = j*fv_size_x + i_m;
//              int idx_p = j*fv_size_x + i_p;

//              s_m00 += cubes[idx_m]->integrate_in_dir(unity_cf_, 0);
//              s_p00 += cubes[idx_p]->integrate_in_dir(unity_cf_, 1);

//              y_m00 += cubes[idx_m]->integrate_in_dir(delta_y_cf_, 0);
//              y_p00 += cubes[idx_p]->integrate_in_dir(delta_y_cf_, 1);
//            }

//          if (s_m00/full_sx > 0*EPS) y_m00 /= s_m00; else y_m00 = 0;
//          if (s_p00/full_sx > 0*EPS) y_p00 /= s_p00; else y_p00 = 0;

//          double s_0m0 = 0, s_0p0 = 0;
//          double x_0m0 = 0, x_0p0 = 0;
//            for (short i = 0; i < fv_size_x; ++i)
//            {
//              int j_m = 0;
//              int j_p = fv_size_y-1.;

//              int idx_m = j_m*fv_size_x + i;
//              int idx_p = j_p*fv_size_x + i;

//              s_0m0 += cubes[idx_m]->integrate_in_dir(unity_cf_, 2);
//              s_0p0 += cubes[idx_p]->integrate_in_dir(unity_cf_, 3);

//              x_0m0 += cubes[idx_m]->integrate_in_dir(delta_x_cf_, 2);
//              x_0p0 += cubes[idx_p]->integrate_in_dir(delta_x_cf_, 3);
//            }

//          if (s_0m0/full_sy > 0*EPS) x_0m0 /= s_0m0; else x_0m0 = 0;
//          if (s_0p0/full_sy > 0*EPS) x_0p0 /= s_0p0; else x_0p0 = 0;
//#endif
