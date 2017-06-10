#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#else
#include "my_p4est_poisson_nodes_mls.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/my_p4est_interpolation_nodes_local.h>
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
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0


my_p4est_poisson_nodes_mls_t::my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors)
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
    node_vol_(NULL),
    // equation
    rhs_(NULL),
    diag_add_scalar_(0.), diag_add_(NULL),
    mu_(1.), mue_(NULL), mue_xx_(NULL), mue_yy_(NULL), mue_zz_(NULL), is_mue_dd_owned_(false), variable_mu_(false),
    // bc
    neumann_wall_first_order_(false),
    use_pointwise_dirichlet_(false),
    bc_wall_type_(NULL), bc_wall_value_(NULL),
    bc_interface_coeff_(NULL), bc_interface_type_(NULL), bc_interface_value_(NULL),
    //other
    mask_(NULL),
    keep_scalling_(false)
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

  double eps = 1E-9*d_min_;

#ifdef P4_TO_P8
  eps_dom_ = eps*eps*eps;
  eps_ifc_ = eps*eps;
#else
  eps_dom_ = eps*eps;
  eps_ifc_ = eps;
#endif
}

my_p4est_poisson_nodes_mls_t::~my_p4est_poisson_nodes_mls_t()
{
  if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }

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
    if (mue_xx_     != NULL) {ierr = VecDestroy(mue_xx_);                CHKERRXX(ierr);}
    if (mue_yy_     != NULL) {ierr = VecDestroy(mue_yy_);                CHKERRXX(ierr);}
#ifdef P4_TO_P8
    if (mue_zz_     != NULL) {ierr = VecDestroy(mue_zz_);                CHKERRXX(ierr);}
#endif
  }

  if (node_vol_ != NULL) {ierr = VecDestroy(node_vol_); CHKERRXX(ierr);}
  if (is_phi_eff_owned_)    {ierr = VecDestroy(phi_eff_);  CHKERRXX(ierr);}
}


void my_p4est_poisson_nodes_mls_t::compute_phi_eff_()
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

void my_p4est_poisson_nodes_mls_t::compute_phi_dd_()
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

void my_p4est_poisson_nodes_mls_t::compute_mue_dd_()
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

void my_p4est_poisson_nodes_mls_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

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

  for (p4est_locidx_t n=0; n<num_owned_local; n++)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

//    if (!bc_->interfaceType() == NOINTERFACE)
      if (phi_p[n] > 2.*diag_min_)
        continue;

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

  ierr = VecRestoreArray(phi_eff_, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_solve, A_, rhs_, ksp_, 0); CHKERRXX(ierr);

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
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

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
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
//  setup_negative_variable_coeff_laplace_rhsvec_();
  setup_linear_system_(false, true);

  // Solve the system
  ierr = KSPSetTolerances(ksp_, 1e-16, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp_); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);
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
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_KSPSolve, ksp_, rhs_, solution, 0); CHKERRXX(ierr);

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

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_solve, A_, rhs_, ksp_, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::setup_linear_system_(bool setup_matrix, bool setup_rhs)
{
  if (!setup_matrix && !setup_rhs)
    throw std::invalid_argument("[CASL_ERROR]: If you aren't assembling either matrix or RHS, what the heck then are you trying to do? lol :)");

  if (setup_matrix)
  {
    if (use_pointwise_dirichlet_)
    {
      pointwise_bc_.clear();
      pointwise_bc_.resize(nodes_->num_owned_indeps);
    }

    preallocate_matrix();
  }

  // register for logging purpose
  // not sure if we need to register both in case we're assembling matrix and rhs simultaneously
  if (setup_matrix)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
  }


//#ifdef P4_TO_P8
//  double eps = 1E-6*d_min*d_min*d_min;
//#else
  double eps = 1E-6*d_min_*d_min_;
//#endif

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

  double *mue_p=NULL, *mue_xx_p=NULL, *mue_yy_p=NULL, *mue_zz_p=NULL;
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
  ierr = VecGetArray(diag_add_,    &diag_add_p   ); CHKERRXX(ierr);

  double *rhs_p;
  if (setup_rhs)
  {
    ierr = VecGetArray(rhs_, &rhs_p); CHKERRXX(ierr);
  }

  std::vector<double> phi_000(num_interfaces_, -1),
      phi_p00(num_interfaces_, 0),
      phi_m00(num_interfaces_, 0),
      phi_0m0(num_interfaces_, 0),
      phi_0p0(num_interfaces_, 0);
  double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;

#ifdef P4_TO_P8
  std::vector<double> phi_00m(num_interfaces_, 0), phi_00p(num_interfaces_, 0);
  double mue_00m, mue_00p;
#endif

  double *mask_p;
  if (setup_matrix)
  {
    if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_->at(0), &mask_); CHKERRXX(ierr);
    ierr = VecGetArray(mask_, &mask_p); CHKERRXX(ierr);
  }

  if (!variable_mu_) {
    mue_000 = mu_; mue_p00 = mu_; mue_m00 = mu_; mue_0m0 = mu_; mue_0p0 = mu_;
#ifdef P4_TO_P8
    mue_00m = mu_; mue_00p = mu_;
#endif
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
  double dxyz_pr[P4EST_DIM];
  double xyz_pr[P4EST_DIM];
  double xyz_c[P4EST_DIM];
  double dist;
//  double measure_of_iface;
//  double measure_of_cut_cell;
//  double mu_proj, bc_value_proj, bc_coeff_proj;

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  std::vector<my_p4est_interpolation_nodes_local_t *> phi_interp_local(num_interfaces_, NULL);

  if (variable_mu_)
#ifdef P4_TO_P8
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, mue_zz_p, quadratic);
#else
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, quadratic);
#endif

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

  double val_interface_m00 = 0.;
  double val_interface_p00 = 0.;
  double val_interface_0m0 = 0.;
  double val_interface_0p0 = 0.;
#ifdef P4_TO_P8
  double val_interface_00m = 0.;
  double val_interface_00p = 0.;
#endif

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
      mask_p[n] = 1;
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
          ierr = MatSetValue(A_, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_eff_p[n] < 0. || num_interfaces_ == 0)
          {
            matrix_has_nullspace_ = false;
            mask_p[n] = -1;
          }
        }

        if (setup_rhs)
        {
          rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_m00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_p00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_0m0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_0p0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_00m_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

            if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
            {
              mask_p[n] = -1;
              ierr = MatSetValue(A_, node_000_g, node_00p_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
            }
          }

          if (setup_rhs) rhs_p[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00p;
          continue;
        }
#endif
      }

    }

    {
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

      for (short i = 0; i < num_interfaces_; ++i)
//        phi_interp_local[i]->initialize(n);
        phi_interp_local[i]->copy_init(interp_local);

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_ngbd_crossed_dirichlet = false;
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
                //              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
                phi_fv[idx] = phi_interp_local[phi_idx]->interpolate(fv_x[i], fv_y[j]);
#endif
                is_one_positive = is_one_positive || phi_fv[idx] > 0;
                is_one_negative = is_one_negative || phi_fv[idx] < 0;
              }

          if (is_one_negative && is_one_positive)
          {
            if (bc_interface_type_->at(phi_idx) == DIRICHLET) is_ngbd_crossed_dirichlet = true;
            if (bc_interface_type_->at(phi_idx) == NEUMANN)   is_ngbd_crossed_neumann   = true;
            if (bc_interface_type_->at(phi_idx) == ROBIN)     is_ngbd_crossed_neumann   = true;
          }
        }
      }

      if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet)
        throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment");
      else if (is_ngbd_crossed_neumann)
        discretization_scheme_ = discretization_scheme_t::FVM;
      else
        discretization_scheme_ = discretization_scheme_t::FDM;

      if (discretization_scheme_ == discretization_scheme_t::FDM)
      {
        //---------------------------------------------------------------------
        // interface boundary
        //---------------------------------------------------------------------
        if (ABS(phi_eff_000) < EPS)
        {
          if (setup_matrix)
          {
            ierr = MatSetValue(A_, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
            mask_p[n] = -1;

            if (use_pointwise_dirichlet_)
              pointwise_bc_[n].push_back(interface_point_t(0, EPS));

            matrix_has_nullspace_ = false;
          }

          if (setup_rhs)
          {
            if (use_pointwise_dirichlet_)
              rhs_p[n] = pointwise_bc_[n][0].value;
            else
              for (int i = 0; i < num_interfaces_; ++i)
                if (fabs(phi_p[i][n]) < EPS && bc_interface_type_->at(i) == DIRICHLET)
                  rhs_p[n] = bc_strength*bc_interface_value_->at(i)->value(xyz_C);
          }

          continue;
        }

        // far away from the interface
        if (phi_eff_000 > 0.)
        {
          if (setup_matrix)
          {
            ierr = MatSetValue(A_, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
            mask_p[n] = 1;
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
            if(!is_interface_m00 && setup_matrix) {
              w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_mp = 0.5*(mue_000 + mue_p[node_m00_mp])*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_pp = 0.5*(mue_000 + mue_p[node_m00_pp])*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
            } else {
              w_m00 *= 0.5*(mue_000 + mue_m00);
            }

            if(!is_interface_p00 && setup_matrix) {
              w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_mp = 0.5*(mue_000 + mue_p[node_p00_mp])*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_pp = 0.5*(mue_000 + mue_p[node_p00_pp])*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
            } else {
              w_p00 *= 0.5*(mue_000 + mue_p00);
            }

            if(!is_interface_0m0 && setup_matrix) {
              w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_mp = 0.5*(mue_000 + mue_p[node_0m0_mp])*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_pp = 0.5*(mue_000 + mue_p[node_0m0_pp])*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
            } else {
              w_0m0 *= 0.5*(mue_000 + mue_0m0);
            }

            if(!is_interface_0p0 && setup_matrix) {
              w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_mp = 0.5*(mue_000 + mue_p[node_0p0_mp])*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_pp = 0.5*(mue_000 + mue_p[node_0p0_pp])*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
            } else {
              w_0p0 *= 0.5*(mue_000 + mue_0p0);
            }

            if(!is_interface_00m && setup_matrix) {
              w_00m_mm = 0.5*(mue_000 + mue_p[node_00m_mm])*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_mp = 0.5*(mue_000 + mue_p[node_00m_mp])*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_pm = 0.5*(mue_000 + mue_p[node_00m_pm])*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_pp = 0.5*(mue_000 + mue_p[node_00m_pp])*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
            } else {
              w_00m *= 0.5*(mue_000 + mue_00m);
            }

            if(!is_interface_00p && setup_matrix) {
              w_00p_mm = 0.5*(mue_000 + mue_p[node_00p_mm])*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
              w_00p_mp = 0.5*(mue_000 + mue_p[node_00p_mp])*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
              w_00p_pm = 0.5*(mue_000 + mue_p[node_00p_pm])*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
              w_00p_pp = 0.5*(mue_000 + mue_p[node_00p_pp])*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
              w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
            } else {
              w_00p *= 0.5*(mue_000 + mue_00p);
            }

          } else {

            if(!is_interface_m00 && setup_matrix) {
              w_m00_mm = mu_*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_mp = mu_*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_pm = mu_*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00_pp = mu_*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
              w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
            } else {
              w_m00 *= mu_;
            }

            if(!is_interface_p00 && setup_matrix) {
              w_p00_mm = mu_*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_mp = mu_*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_pm = mu_*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00_pp = mu_*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
              w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
            } else {
              w_p00 *= mu_;
            }

            if(!is_interface_0m0 && setup_matrix) {
              w_0m0_mm = mu_*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_mp = mu_*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_pm = mu_*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0_pp = mu_*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
              w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
            } else {
              w_0m0 *= mu_;
            }

            if(!is_interface_0p0 && setup_matrix) {
              w_0p0_mm = mu_*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_mp = mu_*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_pm = mu_*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0_pp = mu_*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
              w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
            } else {
              w_0p0 *= mu_;
            }

            if(!is_interface_00m && setup_matrix) {
              w_00m_mm = mu_*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_mp = mu_*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_pm = mu_*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m_pp = mu_*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
              w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
            } else {
              w_00m *= mu_;
            }

            if(!is_interface_00p && setup_matrix) {
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

            if(!is_interface_m00 && setup_matrix) {
              w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
              w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
              w_m00 = w_m00_mm + w_m00_pm;
            } else {
              w_m00 *= 0.5*(mue_000 + mue_m00);
            }

            if(!is_interface_p00 && setup_matrix) {
              w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
              w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
              w_p00    = w_p00_mm + w_p00_pm;
            } else {
              w_p00 *= 0.5*(mue_000 + mue_p00);
            }

            if(!is_interface_0m0 && setup_matrix) {
              w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
              w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
              w_0m0 = w_0m0_mm + w_0m0_pm;
            } else {
              w_0m0 *= 0.5*(mue_000 + mue_0m0);
            }

            if(!is_interface_0p0 && setup_matrix) {
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
          //        w_m00 /= w_000; w_p00 /= w_000;
          //        w_0m0 /= w_000; w_0p0 /= w_000;
          //#ifdef P4_TO_P8
          //        w_00m /= w_000; w_00p /= w_000;
          //#endif

          if (setup_matrix)
          {
            mask_p[n] = -1;
            //---------------------------------------------------------------------
            // add coefficients in the matrix
            //---------------------------------------------------------------------
            if (node_000_g < fixed_value_idx_g_){
              fixed_value_idx_l_ = n;
              fixed_value_idx_g_ = node_000_g;
            }

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
    #ifdef P4_TO_P8
            double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS : (is_node_xpWall(p4est_, ni) ? -2.*EPS : 0);
            double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS : (is_node_ypWall(p4est_, ni) ? -2.*EPS : 0);
            double eps_z = is_node_zmWall(p4est_, ni) ? 2.*EPS : (is_node_zpWall(p4est_, ni) ? -2.*EPS : 0);

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

            double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS : (is_node_xpWall(p4est_, ni) ? -2.*EPS : 0);
            double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS : (is_node_ypWall(p4est_, ni) ? -2.*EPS : 0);

    //        double eps_x = 0;
    //        double eps_y = 0;

            if(is_node_xmWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_p00;
            else if(is_interface_m00)     rhs_p[n] -= w_m00*val_interface_m00;

            if(is_node_xpWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_m00;
            else if(is_interface_p00)     rhs_p[n] -= w_p00*val_interface_p00;

            if(is_node_ymWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0p0;
            else if(is_interface_0m0)     rhs_p[n] -= w_0m0*val_interface_0m0;

            if(is_node_ypWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0m0;
            else if(is_interface_0p0)     rhs_p[n] -= w_0p0*val_interface_0p0;
    #endif

            rhs_p[n] /= w_000;

          }

          continue;
        }

      } else if (discretization_scheme_ = discretization_scheme_t::FVM) {

        double volume_cut_cell = 0.;
        double interface_area  = 0.;
        double integral_bc = 0.;


#ifdef P4_TO_P8
        std::vector<cube3_mls_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
#else
        std::vector<cube2_mls_t *> cubes(fv_size_x*fv_size_y, NULL);
#endif

//#ifdef P4_TO_P8
//        std::vector<cube3_mls_quadratic_t *> cubes(fv_size_x*fv_size_y*fv_size_z, NULL);
//#else
//        std::vector<cube2_mls_quadratic_t *> cubes(fv_size_x*fv_size_y, NULL);
//#endif

#ifdef P4_TO_P8
        for (short k = 0; k < fv_size_z; ++k)
#endif
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
#ifdef P4_TO_P8
              int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
              cubes[idx] = new cube3_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
#else
              int idx = j*fv_size_x + i;
              cubes[idx] = new cube2_mls_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
#endif

//#ifdef P4_TO_P8
//              int idx = k*fv_size_x*fv_size_y + j*fv_size_x + i;
//              cubes[idx] = new cube3_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1], fv_z[k], fv_z[k+1] );
//#else
//              int idx = j*fv_size_x + i;
//              cubes[idx] = new cube2_mls_quadratic_t ( fv_x[i], fv_x[i+1], fv_y[j], fv_y[j+1] );
//#endif

              cubes[idx]->construct_domain(phi_interp_local_cf, *action_, *color_);

              volume_cut_cell += cubes[idx]->integrate_over_domain(unity_cf_);
              interface_area  += cubes[idx]->integrate_over_interface(unity_cf_, -1);

              if (setup_rhs)
                for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
                  if (bc_interface_type_->at(phi_idx) == ROBIN || bc_interface_type_->at(phi_idx) == NEUMANN)
                    integral_bc += cubes[idx]->integrate_over_interface(*bc_interface_value_->at(phi_idx), color_->at(phi_idx));
            }

        if (setup_matrix)
        {
//          if (volume_cut_cell < 0.2*full_cell_volume) mask_p[n] = 1;
          if (volume_cut_cell/full_cell_volume < EPS) mask_p[n] = 1;
          else mask_p[n] = -1;
        }

        if (volume_cut_cell/full_cell_volume > EPS)
        {
          if (setup_rhs)
          {
            rhs_p[n] = rhs_p[n]*volume_cut_cell + integral_bc;
          }
          // Compute areas (lengths) of faces of the finite volume
          double s_m00 = 0, s_p00 = 0;
#ifdef P4_TO_P8
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
              s_m00 += cubes[idx_m]->integrate_in_dir(unity_cf_, 0);
              s_p00 += cubes[idx_p]->integrate_in_dir(unity_cf_, 1);
            }

          double s_0m0 = 0, s_0p0 = 0;
#ifdef P4_TO_P8
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
              s_0m0 += cubes[idx_m]->integrate_in_dir(unity_cf_, 2);
              s_0p0 += cubes[idx_p]->integrate_in_dir(unity_cf_, 3);
            }

#ifdef P4_TO_P8
          double s_00m = 0, s_00p = 0;
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
              int k_m = 0;
              int k_p = fv_size_z-1.;

              int idx_m = k_m*fv_size_x*fv_size_y + j*fv_size_x + i;
              int idx_p = k_p*fv_size_x*fv_size_y + j*fv_size_x + i;

              s_00m += cubes[idx_m]->integrate_in_dir(unity_cf_, 4);
              s_00p += cubes[idx_p]->integrate_in_dir(unity_cf_, 5);
            }
#endif

          /* the code above should return:
           * volume_cut_cell
           * interface_area
           * s_m00, s_p00, ...
           */

          //---------------------------------------------------------------------
          // contributions through cell faces
          //---------------------------------------------------------------------
          double w_m00 = -.5*(mue_000+mue_m00) * s_m00/dx_min_;
          double w_p00 = -.5*(mue_000+mue_p00) * s_p00/dx_min_;
          double w_0m0 = -.5*(mue_000+mue_0m0) * s_0m0/dy_min_;
          double w_0p0 = -.5*(mue_000+mue_0p0) * s_0p0/dy_min_;
#ifdef P4_TO_P8
          double w_00m = -.5*(mue_000+mue_00m) * s_00m/dz_min_;
          double w_00p = -.5*(mue_000+mue_00p) * s_00p/dz_min_;
#endif

#ifdef P4_TO_P8
          double w_000 = diag_add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
          double w_000 = diag_add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0);
#endif

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
            for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
              measure_of_interface[phi_idx] += cubes[cube_idx]->integrate_over_interface(unity_cf_, color_->at(phi_idx));

            if (bc_interface_type_->at(phi_idx) == ROBIN && measure_of_interface[phi_idx] > eps_ifc_)
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

              bc_interface_coeff_avg[3] = 0;
              bc_interface_value_avg[3] = 0;
            }
#endif

            // Evaluate bc_interface_coefs and bc_interface_value at interfaces
            for (short i = 0; i < num_interfaces_present; ++i)
            {
              short phi_idx = present_interfaces[i];

              bc_interface_coeff_avg[i] = 0.;
              bc_interface_value_avg[i] = 0.;

              for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
              {
                bc_interface_coeff_avg[i] += cubes[cube_idx]->integrate_over_interface(*bc_interface_coeff_->at(phi_idx), color_->at(phi_idx));
                bc_interface_value_avg[i] += cubes[cube_idx]->integrate_over_interface(*bc_interface_value_->at(phi_idx), color_->at(phi_idx));
              }

              bc_interface_coeff_avg[i] /= measure_of_interface[phi_idx];
              bc_interface_value_avg[i] /= measure_of_interface[phi_idx];
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
              a_coeff[i_dim] /= mue_000;
              b_coeff[i_dim] /= mue_000;
            }

            // compute integrals
            for (short interface_idx = 0; interface_idx < present_interfaces.size(); ++interface_idx)
            {
              int phi_idx = present_interfaces[interface_idx];

              taylor_expansion_const_term_.set(*bc_interface_coeff_->at(phi_idx), b_coeff, xyz_C);
              taylor_expansion_coeff_term_.set(*bc_interface_coeff_->at(phi_idx), a_coeff, xyz_C);

              for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
              {
                w_000 += cubes[cube_idx]->integrate_over_interface(taylor_expansion_coeff_term_, color_->at(phi_idx));

                if (setup_rhs)
                  rhs_p[n] -= cubes[cube_idx]->integrate_over_interface(taylor_expansion_const_term_, color_->at(phi_idx));
              }
            }

          } // if (is_there_kink && kink_special_treatment && num_ifaces > 1)

          // cells without kinks
          if (!is_there_kink || !kink_special_treatment_) {

            /* In case of COLORATION we need some correction:
             * A LSF that is used for COLORATION doesn't give any information about geometrical properties of interfaces.
             * To find such quantites as the distance to an interface or the projection point
             * one has to refer to a LSF that WAS colorated (not the colorating LSF).
             * That's why in case of COLORATION we loop through all LSFs,
             * which the colorating LSF could colorate.
             */
            for (int interface_idx = 0; interface_idx < present_interfaces.size(); interface_idx++)
            {
              short i_phi = present_interfaces[interface_idx];

              if (bc_interface_type_->at(i_phi) == ROBIN)
              {
                int num_iterations;

                if (action_->at(i_phi) == COLORATION) num_iterations = i_phi;
                else                                  num_iterations = 1;

                double measure_of_iface;
                for (int j_phi = 0; j_phi < num_iterations; j_phi++)
                {
                  if (action_->at(i_phi) != COLORATION) measure_of_iface = measure_of_interface[i_phi];
                  else
                  {
                    measure_of_iface = 0;
                    for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
                    {
                      measure_of_iface += cubes[cube_idx]->integrate_over_colored_interface(unity_cf_, color_->at(j_phi), color_->at(i_phi));
                    }
                  }

                  if (measure_of_iface > eps_ifc_)
                  {
                    if (action_->at(i_phi) == COLORATION) find_projection_(phi_p[j_phi], qnnn, dxyz_pr, dist);
                    else                                  find_projection_(phi_p[i_phi], qnnn, dxyz_pr, dist);

                    for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                      xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

                    double mu_proj = mu_;

                    if (variable_mu_)
                      mu_proj = interp_local.value(xyz_pr);

                    double bc_coeff_proj = bc_interface_coeff_->at(i_phi)->value(xyz_pr);

                    if (use_taylor_correction_) { w_000 += mu_proj*bc_coeff_proj*measure_of_iface/(mu_proj-bc_coeff_proj*dist); }
                    else                        { w_000 += bc_coeff_proj*measure_of_iface; }

                    if (setup_rhs && use_taylor_correction_)
                    {
                      double bc_value_proj = bc_interface_value_->at(i_phi)->value(xyz_pr);
                      rhs_p[n] -= measure_of_iface*bc_coeff_proj*bc_value_proj*dist/(bc_coeff_proj*dist-mu_proj);
                    }

                    if (fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
                  }
                }
              }
            }
          }


          if (setup_rhs)
          {
            rhs_p[n] /= w_000;
//            rhs_p[n] /= full_cell_volume;
          }


//          if (fabs(w_000) < EPS)
//          {
//            std::cout << w_000 << " " << volume_cut_cell << " " << interface_area << " " << eps_dom_;
//            throw std::domain_error("Diagonal term is zero");
//          }

          if (setup_matrix)
          {
            if (diag_add_p[n] > 0) matrix_has_nullspace_ = false;
            if (keep_scalling_) scalling_[n] = w_000/full_cell_volume; // scale to measure of the full cell dx*dy(*dz) for consistence

            w_m00 /= w_000; w_p00 /= w_000;
            w_0m0 /= w_000; w_0p0 /= w_000;
  #ifdef P4_TO_P8
            w_00m /= w_000; w_00p /= w_000;
  #endif
            //----------------------------------------------------------------------------------------------------------------
            // add coefficients in the matrix
            //----------------------------------------------------------------------------------------------------------------
#ifdef P4_TO_P8
            PetscInt node_m00_g = petsc_gloidx_[qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                                                 : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp) ];
            PetscInt node_p00_g = petsc_gloidx_[qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                                                 : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp) ];
            PetscInt node_0m0_g = petsc_gloidx_[qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                                                 : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp) ];
            PetscInt node_0p0_g = petsc_gloidx_[qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                                                 : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp) ];
            PetscInt node_00m_g = petsc_gloidx_[qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                                                 : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp) ];
            PetscInt node_00p_g = petsc_gloidx_[qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                                                 : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp) ];
#else
            PetscInt node_m00_g = petsc_gloidx_[qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm];
            PetscInt node_p00_g = petsc_gloidx_[qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm];
            PetscInt node_0m0_g = petsc_gloidx_[qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm];
            PetscInt node_0p0_g = petsc_gloidx_[qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm];
#endif

            ierr = MatSetValue(A_, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);

            if (ABS(w_m00) > EPS) {ierr = MatSetValue(A_, node_000_g, node_m00_g, w_m00,  ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_p00) > EPS) {ierr = MatSetValue(A_, node_000_g, node_p00_g, w_p00,  ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0m0) > EPS) {ierr = MatSetValue(A_, node_000_g, node_0m0_g, w_0m0,  ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_0p0) > EPS) {ierr = MatSetValue(A_, node_000_g, node_0p0_g, w_0p0,  ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
            if (ABS(w_00m) > EPS) {ierr = MatSetValue(A_, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
            if (ABS(w_00p) > EPS) {ierr = MatSetValue(A_, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}
#endif
          }

        } else {
          if (setup_matrix)
          {
            mask_p[n] = 1;
            ierr = MatSetValue(A_, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
          }

          if (setup_rhs)
            rhs_p[n] = 0;
        }

        // free cubes
        for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
        {
          delete cubes[cube_idx];
        }
      }
    }
  }

  if (setup_matrix)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }

  // restore pointers
  ierr = VecRestoreArray(mask_, &mask_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(mask_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(mask_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


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
    ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(diag_add_,    &diag_add_p   ); CHKERRXX(ierr);

  if (setup_rhs)
  {
    ierr = VecRestoreArray(rhs_, &rhs_p); CHKERRXX(ierr);
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
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
  }

}

//void my_p4est_poisson_nodes_mls_t::setup_negative_variable_coeff_laplace_rhsvec()
//{
//  // register for logging purpose
//  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

////#ifdef P4_TO_P8
////  double eps = 1E-6*d_min*d_min*d_min;
////#else
//  double eps = 1E-6*d_min*d_min;
////#endif

//  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
//  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  double *phi_zz_p;
//  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

//  double *mue_p=NULL, *mue_xx_p=NULL, *mue_yy_p=NULL, *mue_zz_p=NULL;
//  if (variable_mu)
//  {
//    ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
//    ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
//    ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
//#endif
//  }

//  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
//  double *rhs_p;
//  ierr = VecGetArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);

//  double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
//  double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;

//#ifdef P4_TO_P8
//  double phi_00m, phi_00p;
//  double mue_00m, mue_00p;
//#endif

//  if (!variable_mu) {
//    mue_000 = mu_; mue_p00 = mu_; mue_m00 = mu_; mue_0m0 = mu_; mue_0p0 = mu_;
//#ifdef P4_TO_P8
//    mue_00m = mu_; mue_00p = mu_;
//#endif
//  }

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
//    // tree information
//    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

//    //---------------------------------------------------------------------
//    // Information at neighboring nodes
//    //---------------------------------------------------------------------
//    node_xyz_fr_n(n, p4est, nodes, xyz_C);
//    double x_C  = xyz_C[0];
//    double y_C  = xyz_C[1];
//#ifdef P4_TO_P8
//    double z_C  = xyz_C[2];
//#endif

//    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

//    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
//    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
//#ifdef P4_TO_P8
//    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
//#endif

//    /*
//     * NOTE: All nodes are in PETSc' local numbering
//     */
//    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
//    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
//    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
//    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
//#ifdef P4_TO_P8
//    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
//    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
//    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
//    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

//    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
//    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
//    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
//    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
//#endif

//    if(is_node_Wall(p4est, ni))
//    {
//      if(bc_->wallType(xyz_C) == DIRICHLET) {
////        double val = bc_strength*bc_->wallValue(xyz_C);
//        rhs_p[n] = bc_strength*bc_->wallValue(xyz_C);
//        continue;
//      }

//      // In case if you want first order neumann at walls. Why is it still a thing anyway?
//      if(bc_->wallType(xyz_C) == NEUMANN && neumann_wall_first_order)
//      {
//        if (is_node_xpWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_m00; continue;}
//        if (is_node_xmWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_p00; continue;}
//        if (is_node_ypWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_0m0; continue;}
//        if (is_node_ymWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_0p0; continue;}
//#ifdef P4_TO_P8
//        if (is_node_zpWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_00m; continue;}
//        if (is_node_zmWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_00p; continue;}
//#endif
//      }
//    }

//    {
//#ifdef P4_TO_P8
//      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
//      if (variable_mu)
//        qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
//#else
//      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
//      if (variable_mu)
//        qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
//#endif

//      //---------------------------------------------------------------------
//      // interface boundary
//      //---------------------------------------------------------------------
//      if((ABS(phi_000)<EPS && bc_->interfaceType() == DIRICHLET) ){
//        if (use_pointwise_dirichlet)
//          rhs_p[n] = pointwise_bc[n][0].value;
//        else
//          rhs_p[n] = bc_strength*bc_->interfaceValue(xyz_C);
//        continue;
//      }

//      //---------------------------------------------------------------------
//      // check if finite volume is crossed
//      //---------------------------------------------------------------------
//      bool is_one_positive = false;
//      bool is_one_negative = false;

//      if (fabs(phi_000) < 2.*diag_min && (bc_->interfaceType() == ROBIN || bc_->interfaceType() == NEUMANN))
////      if (fabs(phi_000) < 2.*diag_min)
//      {
//        phi_interp_local.initialize(n);
//        // determine dimensions of cube
//        fv_size_x = 0;
//        fv_size_y = 0;
//#ifdef P4_TO_P8
//        fv_size_z = 0;
//#endif
//        if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dx_min;} else {fv_xmin = x_C;}
//        if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dx_min;} else {fv_xmax = x_C;}

//        if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dy_min;} else {fv_ymin = y_C;}
//        if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dy_min;} else {fv_ymax = y_C;}
//#ifdef P4_TO_P8
//        if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dz_min;} else {fv_zmin = z_C;}
//        if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dz_min;} else {fv_zmax = z_C;}
//#endif

//        if (!use_refined_cube) {
//          fv_size_x = 1;
//          fv_size_y = 1;
//#ifdef P4_TO_P8
//          fv_size_z = 1;
//#endif
//        }

//        fv_nx = fv_size_x+1;
//        fv_ny = fv_size_y+1;
//#ifdef P4_TO_P8
//        fv_nz = fv_size_z+1;
//#endif

//        // get coordinates of cube nodes
//        double fv_dx = (fv_xmax-fv_xmin)/ (double)(fv_size_x);
//        fv_x[0] = fv_xmin;
//        for (short i = 1; i < fv_nx; ++i)
//          fv_x[i] = fv_x[i-1] + fv_dx;

//        double fv_dy = (fv_ymax-fv_ymin)/ (double)(fv_size_y);
//        fv_y[0] = fv_ymin;
//        for (short i = 1; i < fv_ny; ++i)
//          fv_y[i] = fv_y[i-1] + fv_dy;
//#ifdef P4_TO_P8
//        double fv_dx = (fv_zmax-fv_zmin)/ (double)(fv_size_z);
//        fv_z[0] = fv_zmin;
//        for (short i = 1; i < fv_nz; ++i)
//          fv_z[i] = fv_z[i-1] + fv_dz;
//#endif

//        // sample level-set function at cube nodes and check if crossed
//#ifdef P4_TO_P8
//        for (short k = 0; k < fv_nz; ++k)
//#endif
//          for (short j = 0; j < fv_ny; ++j)
//            for (short i = 0; i < fv_nx; ++i)
//            {
//#ifdef P4_TO_P8
//              int idx = k*fv_nx*fv_ny + j*fv_nx + i;
//              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j], fv_z[k]);
//#else
//              int idx = j*fv_nx + i;
////              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
//              phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j]);
//#endif
//              is_one_positive = is_one_positive || phi_fv[idx] > 0;
//              is_one_negative = is_one_negative || phi_fv[idx] < 0;
//            }
//      }

//      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

//      // far away from the interface
//      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET ) && (bc_->interfaceType() != NOINTERFACE)){
//        if(bc_->interfaceType()==DIRICHLET)
//          rhs_p[n] = bc_strength*bc_->interfaceValue(xyz_C);
//        else
//          rhs_p[n] = 0;
//        continue;
//      }

//      // if far away from the interface or close to it but with dirichlet
//      // then finite difference method
//      if ( (bc_->interfaceType() == DIRICHLET && phi_000<0.) ||
//           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
//           (bc_->interfaceType() == ROBIN     && !is_ngbd_crossed_neumann ) ||
//            bc_->interfaceType() == NOINTERFACE)
//      {
//        double phixx_C = phi_xx_p[n];
//        double phiyy_C = phi_yy_p[n];
//#ifdef P4_TO_P8
//        double phizz_C = phi_zz_p[n];
//#endif

//        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
//        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
//        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
//        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
//#ifdef P4_TO_P8
//        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
//        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
//#endif


////#ifdef P4_TO_P8
////        if (!( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0  || is_interface_00m || is_interface_00p) && !is_node_Wall(p4est, ni))
////#else
////        if (!( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0 ) && !is_node_Wall(p4est, ni))
////#endif
////        {
////          rhs_p[n] /= scalling[n];
////          continue;
////        }

//        double val_interface_m00 = 0.;
//        double val_interface_p00 = 0.;
//        double val_interface_0m0 = 0.;
//        double val_interface_0p0 = 0.;
//#ifdef P4_TO_P8
//        double val_interface_00m = 0.;
//        double val_interface_00p = 0.;
//#endif

//        // given boundary condition at interface from quadratic interpolation
//        if( is_interface_m00) {
//          double phixx_m00 = qnnn.f_m00_linear(phi_xx_p);
//          double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phixx_C, phixx_m00);
//          if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
//          d_m00_m0 = d_m00_p0 = 0;
//#ifdef P4_TO_P8
//          d_m00_0m = d_m00_0p = 0;
//#endif

//          if (variable_mu) {
//            double mxx_000 = mue_xx_p[n];
//            double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
//            mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);
//          }

//          d_m00 = theta_m00;

//#ifdef P4_TO_P8
//          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C, z_C);
//#else
//          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C);
//#endif
//        }
//        if( is_interface_p00){
//          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
//          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
//          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
//          d_p00_m0 = d_p00_p0 = 0;
//#ifdef P4_TO_P8
//          d_p00_0m = d_p00_0p = 0;
//#endif

//          if (variable_mu) {
//            double mxx_000 = mue_xx_p[n];
//            double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
//            mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);
//          }

//          d_p00 = theta_p00;

//#ifdef P4_TO_P8
//          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C, z_C);
//#else
//          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C);
//#endif
//        }
//        if( is_interface_0m0){
//          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
//          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
//          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
//          d_0m0_m0 = d_0m0_p0 = 0;
//#ifdef P4_TO_P8
//          d_0m0_0m = d_0m0_0p = 0;
//#endif

//          if (variable_mu) {
//            double myy_000 = mue_yy_p[n];
//            double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
//            mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);
//          }

//          d_0m0 = theta_0m0;

//#ifdef P4_TO_P8
//          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0, z_C);
//#else
//          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0);
//#endif
//        }
//        if( is_interface_0p0){
//          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
//          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
//          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
//          d_0p0_m0 = d_0p0_p0 = 0;
//#ifdef P4_TO_P8
//          d_0p0_0m = d_0p0_0p = 0;
//#endif

//          if (variable_mu) {
//            double myy_000 = mue_yy_p[n];
//            double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
//            mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);
//          }

//          d_0p0 = theta_0p0;
//#ifdef P4_TO_P8
//          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0, z_C);
//#else
//          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0);
//#endif
//        }
//#ifdef P4_TO_P8
//        if( is_interface_00m){
//          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
//          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
//          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
//          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;

//          if (variable_mu) {
//            double mzz_000 = mue_zz_p[n];
//            double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
//            mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);
//          }

//          d_00m = theta_00m;

//          val_interface_00m = bc_->interfaceValue(x_C, y_C , z_C - theta_00m);
//        }
//        if( is_interface_00p){
//          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
//          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
//          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
//          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

//          if (variable_mu) {
//            double mzz_000 = mue_zz_p[n];
//            double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
//            mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);
//          }

//          d_00p = theta_00p;

//          val_interface_00p = bc_->interfaceValue(x_C, y_C , z_C + theta_00p);
//        }
//#endif

//        if (use_pointwise_dirichlet)
//        {
//          std::vector<double> val_interface(2*P4EST_DIM, 0);
//          for (short i = 0; i < pointwise_bc[n].size(); ++i)
//          {
//            interface_point_t *pnt = &pointwise_bc[n][i];
//            val_interface[pnt->dir] = pnt->value;
//          }

//          val_interface_m00 = val_interface[0];
//          val_interface_p00 = val_interface[1];
//          val_interface_0m0 = val_interface[2];
//          val_interface_0p0 = val_interface[3];
//#ifdef P4_TO_P8
//          val_interface_00m = val_interface[4];
//          val_interface_00p = val_interface[5];
//#endif
//        }

//#ifdef P4_TO_P8
//        //------------------------------------
//        // Dfxx =   fxx + a*fyy + b*fzz
//        // Dfyy = c*fxx +   fyy + d*fzz
//        // Dfzz = e*fxx + f*fyy +   fzz
//        //------------------------------------
//        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
//        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

//        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
//        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

//        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
//        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

//        //------------------------------------------------------------
//        // compensating the error of linear interpolation at T-junction using
//        // the derivative in the transversal direction
//        //
//        // Laplace = wi*Dfxx +
//        //           wj*Dfyy +
//        //           wk*Dfzz
//        //------------------------------------------------------------

//        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
//        double wi = (1.-c-e+c*f+e*d-d*f)/det;
//        double wj = (1.-a-f+a*e+f*b-b*e)/det;
//        double wk = (1.-b-d+b*c+d*a-a*c)/det;
//#else
//        //------------------------------------------------------------
//        // compensating the error of linear interpolation at T-junction using
//        // the derivative in the transversal direction
//        //
//        // Laplace = wi*Dfxx +
//        //           wj*Dfyy
//        //------------------------------------------------------------

//        double wi = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);
//        double wj = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
//#endif

//        //---------------------------------------------------------------------
//        // Shortley-Weller method, dimension by dimension
//        //---------------------------------------------------------------------
//        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;
//#ifdef P4_TO_P8
//        double w_00m=0, w_00p=0;
//#endif
//        if(is_node_xmWall(p4est, ni))      w_p00 += -1.0*wi/(d_p00*d_p00);
//        else if(is_node_xpWall(p4est, ni)) w_m00 += -1.0*wi/(d_m00*d_m00);
//        else                               w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

//        if(is_node_xpWall(p4est, ni))      w_m00 += -1.0*wi/(d_m00*d_m00);
//        else if(is_node_xmWall(p4est, ni)) w_p00 += -1.0*wi/(d_p00*d_p00);
//        else                               w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

//        if(is_node_ymWall(p4est, ni))      w_0p0 += -1.0*wj/(d_0p0*d_0p0);
//        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1.0*wj/(d_0m0*d_0m0);
//        else                               w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

//        if(is_node_ypWall(p4est, ni))      w_0m0 += -1.0*wj/(d_0m0*d_0m0);
//        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1.0*wj/(d_0p0*d_0p0);
//        else                               w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

//#ifdef P4_TO_P8
//        if(is_node_zmWall(p4est, ni))      w_00p += -1.0*wk/(d_00p*d_00p);
//        else if(is_node_zpWall(p4est, ni)) w_00m += -1.0*wk/(d_00m*d_00m);
//        else                               w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

//        if(is_node_zpWall(p4est, ni))      w_00m += -1.0*wk/(d_00m*d_00m);
//        else if(is_node_zmWall(p4est, ni)) w_00p += -1.0*wk/(d_00p*d_00p);
//        else                               w_00p += -2.0*wk/d_00p/(d_00m+d_00p);
//#endif
//        w_m00 *= 0.5*(mue_000 + mue_m00);
//        w_p00 *= 0.5*(mue_000 + mue_p00);
//        w_0m0 *= 0.5*(mue_000 + mue_0m0);
//        w_0p0 *= 0.5*(mue_000 + mue_0p0);
//#ifdef P4_TO_P8
//        w_00m *= 0.5*(mue_000 + mue_00m);
//        w_00p *= 0.5*(mue_000 + mue_00p);
//#endif

//        //---------------------------------------------------------------------
//        // diag scaling
//        //---------------------------------------------------------------------
//#ifdef P4_TO_P8
//        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
//#else
//        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
//#endif

////        w_m00 /= w_000; w_p00 /= w_000;
////        w_0m0 /= w_000; w_0p0 /= w_000;
////#ifdef P4_TO_P8
////        w_00m /= w_000; w_00p /= w_000;
////#endif

//        //---------------------------------------------------------------------
//        // add coefficients to the right hand side
//        //---------------------------------------------------------------------
////        if(is_interface_m00) rhs_p[n] -= w_m00 * val_interface_m00;
////        if(is_interface_p00) rhs_p[n] -= w_p00 * val_interface_p00;
////        if(is_interface_0m0) rhs_p[n] -= w_0m0 * val_interface_0m0;
////        if(is_interface_0p0) rhs_p[n] -= w_0p0 * val_interface_0p0;
////#ifdef P4_TO_P8
////        if(is_interface_00m) rhs_p[n] -= w_00m * val_interface_00m;
////        if(is_interface_00p) rhs_p[n] -= w_00p * val_interface_00p;
////#endif
//        // FIX this for variable mu
//#ifdef P4_TO_P8
//        double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
//        double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);
//        double eps_z = is_node_zmWall(p4est, ni) ? 2*EPS : (is_node_zpWall(p4est, ni) ? -2*EPS : 0);

//        if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y, z_C+eps_z) / d_p00;
//        else if(is_interface_m00)     rhs_p[n] -= w_m00 * val_interface_m00;

//        if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y, z_C+eps_z) / d_m00;
//        else if(is_interface_p00)     rhs_p[n] -= w_p00 * val_interface_p00;

//        if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C, z_C+eps_z) / d_0p0;
//        else if(is_interface_0m0)     rhs_p[n] -= w_0m0 * val_interface_0m0;

//        if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C, z_C+eps_z) / d_0m0;
//        else if(is_interface_0p0)     rhs_p[n] -= w_0p0 * val_interface_0p0;

//        if(is_node_zmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C+eps_y, z_C) / d_00p;
//        else if(is_interface_00m)     rhs_p[n] -= w_00m * val_interface_00m;

//        if(is_node_zpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C+eps_y, z_C) / d_00m;
//        else if(is_interface_00p)     rhs_p[n] -= w_00p * val_interface_00p;
//#else

//        double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
//        double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);

////        double eps_x = 0;
////        double eps_y = 0;

//        if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y) / d_p00;
//        else
//          if(is_interface_m00)     rhs_p[n] -= w_m00*val_interface_m00;

//        if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y) / d_m00;
//        else
//          if(is_interface_p00)     rhs_p[n] -= w_p00*val_interface_p00;

//        if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C) / d_0p0;
//        else
//          if(is_interface_0m0)     rhs_p[n] -= w_0m0*val_interface_0m0;

//        if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C) / d_0m0;
//        else
//          if(is_interface_0p0)     rhs_p[n] -= w_0p0*val_interface_0p0;
//#endif

//        rhs_p[n] /= w_000;
//        continue;
//      }

//      // if ngbd is crossed and neumman BC
//      // then use finite volume method
//      // only work if the mesh is uniform close to the interface

//      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
//      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN) )
//      {
//        double volume_cut_cell = 0.;
//        double interface_area  = 0.;

//#ifdef P4_TO_P8
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
//            }
//#else
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
//          }
//#endif

//        if (volume_cut_cell>eps*eps)
//        {
//#ifdef P4_TO_P8
//          Cube2 c2;
//          QuadValue qv;

//          double s_m00 = 0, s_p00 = 0;
//          for (short k = 0; k < fv_size_z; ++k)
//            for (short j = 0; j < fv_size_y; ++j) {

//              c2.x0 = fv_y[j]; c2.x1 = fv_y[j+1];
//              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

//              int i = 0;
//              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
//              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

//              s_m00 += c2.area_In_Negative_Domain(qv);

//              i = fv_size_x;
//              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
//              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
//              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

//              s_p00 += c2.area_In_Negative_Domain(qv);
//            }

//          double s_0m0 = 0, s_0p0 = 0;
//          for (short k = 0; k < fv_size_z; ++k)
//            for (short i = 0; i < fv_size_x; ++i) {

//              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
//              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

//              int j = 0;
//              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
//              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

//              s_0m0 += c2.area_In_Negative_Domain(qv);

//              j = fv_size_y;
//              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
//              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
//              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

//              s_0p0 += c2.area_In_Negative_Domain(qv);
//            }

//          double s_00m = 0, s_00p = 0;
//          for (short j = 0; j < fv_size_j; ++j)
//            for (short i = 0; i < fv_size_x; ++i) {

//              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
//              c2.y0 = fv_y[j]; c2.y1 = fv_y[j+1];

//              int k = 0;
//              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
//              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
//              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
//              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

//              s_00m += c2.area_In_Negative_Domain(qv);

//              k = fv_size_z;
//              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
//              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
//              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
//              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

//              s_00p += c2.area_In_Negative_Domain(qv);
//            }
//#else
//          double s_m00 = 0, s_p00 = 0;
//          for (short j = 0; j < fv_size_y; ++j) {
//            int i;
//            i = 0;          s_m00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
//            i = fv_size_x;  s_p00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
//          }

//          double s_0m0 = 0, s_0p0 = 0;
//          for (short i = 0; i < fv_size_x; ++i) {
//            int j;
//            j = 0;          s_0m0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
//            j = fv_size_y;  s_0p0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
//          }
//#endif


//          double w_m00 = -0.5*(mue_000 + mue_m00)*s_m00/dx_min;
//          double w_p00 = -0.5*(mue_000 + mue_p00)*s_p00/dx_min;
//          double w_0m0 = -0.5*(mue_000 + mue_0m0)*s_0m0/dy_min;
//          double w_0p0 = -0.5*(mue_000 + mue_0p0)*s_0p0/dy_min;
//#ifdef P4_TO_P8
//          double w_00m = -0.5*(mue_000 + mue_00m)*s_00m/dz_min;
//          double w_00p = -0.5*(mue_000 + mue_00p)*s_00p/dz_min;
//          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
//#else
//          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0);
//#endif
//          rhs_p[n] *= volume_cut_cell;

//          if (bc_->interfaceType() == ROBIN)
//          {
//            double xyz_p[P4EST_DIM];
//            double normal[P4EST_DIM];

//            normal[0] = qnnn.dx_central(phi_p);
//            normal[1] = qnnn.dy_central(phi_p);
//#ifdef P4_TO_P8
//            normal[2] = qnnn.dz_central(phi_p);
//            double norm = sqrt(SQR(normal[0])+SQR(normal[1])+SQR(normal[2]));
//#else
//            double norm = sqrt(SQR(normal[0])+SQR(normal[1]));
//#endif

//            for (short dir = 0; dir < P4EST_DIM; ++dir)
//            {
//              xyz_p[dir] = xyz_C[dir] - phi_p[n]*normal[dir]/norm;

//              if      (xyz_p[dir] < xyz_C[dir]-dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]-dxyz_m[dir]+EPS;
//              else if (xyz_p[dir] > xyz_C[dir]+dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]+dxyz_m[dir]+EPS;
//            }

//            double robin_coef_proj = bc_->robinCoef(xyz_p);

//            if (fabs(robin_coef_proj) > 0) matrix_has_nullspace = false;

//            // FIX this for variable mu
//            if (robin_coef_proj*phi_p[n] < 1. || 1)
//            {
//              w_000 += mue_000*(robin_coef_proj/(1.-phi_p[n]*robin_coef_proj))*interface_area;

//              double beta_proj = bc_->interfaceValue(xyz_p);
//              rhs_p[n] += mue_000*robin_coef_proj*phi_p[n]/(1.-robin_coef_proj*phi_p[n]) * interface_area*beta_proj;
//            }
//            else
//            {
//              w_000 += mue_000*robin_coef_proj*interface_area;
//            }
//          }

//          double integral_bc = 0;
//#ifdef P4_TO_P8
//          Cube3 cube;
//          OctValue  phi_cube;

//          for (short k = 0; k < fv_size_z; ++k)
//            for (short j = 0; j < fv_size_y; ++j)
//              for (short i = 0; i < fv_size_x; ++i)
//              {
//                cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
//                cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
//                cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

//                phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
//                phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
//                phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
//                phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
//                phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
//                phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
//                phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
//                phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

//                integral_bc += cube.integrate_Over_Interface(bc_->getInterfaceValue(), phi_cube);
//              }
//#else
//          Cube2 cube;
//          QuadValue phi_cube;

//          for (short j = 0; j < fv_size_y; ++j)
//            for (short i = 0; i < fv_size_x; ++i)
//            {
//              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
//              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

//              phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
//              phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
//              phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
//              phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

////              const CF_2 *fff = &bc_->getRobinCoef();
//              integral_bc += cube.integrate_Over_Interface(bc_->getInterfaceValue(), phi_cube);
//            }
//#endif

//          rhs_p[n] += mue_000*integral_bc;
//          rhs_p[n] /= w_000;

//        } else {
//          rhs_p[n] = 0.;
//        }
//      }
//    }
//  }

////  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
////    rhs_p[fixed_value_idx_l] = 0;
////  }


//  // restore pointers
//  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

//  if (variable_mu) {
//    ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
//    ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
//#endif
//  }
//  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

//  ierr = VecRestoreArray(rhs_,    &phi_p   ); CHKERRXX(ierr);

//  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
//}


void my_p4est_poisson_nodes_mls_t::find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr)
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

void my_p4est_poisson_nodes_mls_t::compute_normal_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[])
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



void my_p4est_poisson_nodes_mls_t::inv_mat3_(double *in, double *out)
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

void my_p4est_poisson_nodes_mls_t::inv_mat2_(double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];
  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;
}


//void my_p4est_poisson_nodes_mls_t::assemble_matrix(Vec solution)
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

////void my_p4est_poisson_nodes_mls_t::assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p)
//void my_p4est_poisson_nodes_mls_t::assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in, Vec rhs_p_in)
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
