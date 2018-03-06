#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls_sc.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#include <src/my_p8est_interpolation_nodes_local.h>
//#include <src/simplex3_mls_vtk.h>
//#include <src/simplex3_mls_quadratic_vtk.h>
#else
#include "my_p4est_poisson_nodes_mls_sc.h"
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


my_p4est_poisson_nodes_mls_sc_t::my_p4est_poisson_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors)
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
    keep_scalling_(false),
    volumes_(NULL),
    node_type_(NULL),
    phi_x_(NULL), phi_y_(NULL), phi_z_(NULL), is_phi_d_owned_(false),
    use_sc_scheme_(true),
    lip_(2.0), integration_order_(2)
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

my_p4est_poisson_nodes_mls_sc_t::~my_p4est_poisson_nodes_mls_sc_t()
{
  if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }

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

  if (is_phi_d_owned_)
  {
    if (phi_x_ != NULL) { for (int i = 0; i < phi_x_->size(); i++) {ierr = VecDestroy(phi_x_->at(i)); CHKERRXX(ierr);} delete phi_x_; }
    if (phi_y_ != NULL) { for (int i = 0; i < phi_y_->size(); i++) {ierr = VecDestroy(phi_y_->at(i)); CHKERRXX(ierr);} delete phi_y_; }
#ifdef P4_TO_P8
    if (phi_z_ != NULL) { for (int i = 0; i < phi_z_->size(); i++) {ierr = VecDestroy(phi_z_->at(i)); CHKERRXX(ierr);} delete phi_z_; }
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

  if (volumes_ != NULL) {ierr = VecDestroy(volumes_); CHKERRXX(ierr);}
  if (node_type_ != NULL) {ierr = VecDestroy(node_type_); CHKERRXX(ierr);}
  if (is_phi_eff_owned_)    {ierr = VecDestroy(phi_eff_);  CHKERRXX(ierr);}
}


void my_p4est_poisson_nodes_mls_sc_t::compute_phi_eff_()
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
}

void my_p4est_poisson_nodes_mls_sc_t::compute_phi_dd_()
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

void my_p4est_poisson_nodes_mls_sc_t::compute_phi_d_()
{
  // Allocate memory for second derivaties
  if (phi_x_ != NULL && is_phi_d_owned_) { for (int i = 0; i < phi_x_->size(); i++) {ierr = VecDestroy(phi_x_->at(i)); CHKERRXX(ierr);} delete phi_x_; } phi_x_ = new std::vector<Vec> ();
  if (phi_y_ != NULL && is_phi_d_owned_) { for (int i = 0; i < phi_y_->size(); i++) {ierr = VecDestroy(phi_y_->at(i)); CHKERRXX(ierr);} delete phi_y_; } phi_y_ = new std::vector<Vec> ();
#ifdef P4_TO_P8
  if (phi_z_ != NULL && is_phi_d_owned_) { for (int i = 0; i < phi_z_->size(); i++) {ierr = VecDestroy(phi_z_->at(i)); CHKERRXX(ierr);} delete phi_z_; } phi_z_ = new std::vector<Vec> ();
#endif

  for (unsigned int i = 0; i < num_interfaces_; i++)
  {
    phi_x_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_x_->at(i)); CHKERRXX(ierr);
    phi_y_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_y_->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    phi_z_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_z_->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    Vec phi_d[P4EST_DIM] = { phi_x_->at(i), phi_y_->at(i), phi_z_->at(i) };
#else
    Vec phi_d[P4EST_DIM] = { phi_x_->at(i), phi_y_->at(i) };
#endif

    node_neighbors_->first_derivatives_central(phi_->at(i), phi_d);
  }
  is_phi_d_owned_ = true;
}

void my_p4est_poisson_nodes_mls_sc_t::compute_mue_dd_()
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

void my_p4est_poisson_nodes_mls_sc_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);

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

    // TODO: fix this bullshit
    if (fabs(phi_p[n]) < 2.*diag_min_)
    {
#ifdef P4_TO_P8
      d_nnz[n] = 27;
      o_nnz[n] = 27;
#else
      d_nnz[n] = 9;
      o_nnz[n] = 9;
#endif
    }

  }

  ierr = VecRestoreArray(phi_eff_, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A_, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_sc_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
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
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.9"); CHKERRXX(ierr);

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

void my_p4est_poisson_nodes_mls_sc_t::setup_linear_system(bool setup_matrix, bool setup_rhs)
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
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_setup, A_, 0, 0, 0); CHKERRXX(ierr);
  }

  if (setup_rhs)
  {
    ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
  }


//#ifdef P4_TO_P8
//  double eps = 1E-6*d_min*d_min*d_min;
//#else
  double eps = 1E-6*d_min_*d_min_;
//#endif

//  double domain_rel_thresh = eps*eps;
  double domain_rel_thresh = 1.e-14;
  double interface_rel_thresh = 1.e-100;//0*eps;
//  double interface_rel_thresh = 0*EPS;//0*eps;

  double *exact_ptr;
  ierr = VecGetArray(exact_, &exact_ptr); CHKERRXX(ierr);

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
    compute_volumes_();
    if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_->at(0), &mask_); CHKERRXX(ierr);
  }
  ierr = VecGetArray(mask_, &mask_p); CHKERRXX(ierr);

  if (setup_matrix)
    for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
      mask_p[n] = -1;

  double *volumes_p;
  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  double *node_type_p;
  ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);

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
  double dist;
//  double measure_of_iface;
//  double measure_of_cut_cell;
//  double mu_proj, bc_value_proj, bc_coeff_proj;

  bool neighbors_exist[num_neighbors_max_];
  p4est_locidx_t neighbors[num_neighbors_max_];

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);

  if (variable_mu_)
#ifdef P4_TO_P8
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, mue_zz_p, quadratic);
#else
    interp_local.set_input(mue_p, mue_xx_p, mue_yy_p, quadratic);
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

//    if (setup_matrix)
    {
//      mask_p[n] = 1;
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

    /* FIX THIS
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

    if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet)
      throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment");
    else if (is_ngbd_crossed_neumann)
      discretization_scheme_ = discretization_scheme_t::FVM;
    else
      discretization_scheme_ = discretization_scheme_t::FDM;


//    if (fabs(phi_eff_000) < lip_*diag_min_)
//    {
//      for (char idx = 0; idx < num_neighbors_max_; ++idx)
//        if (neighbors_exist[idx])
//          neighbors_exist[idx] = neighbors_exist[idx] && (volumes_p[neighbors[idx]] > 0.1);
////          neighbors_exist[idx] = neighbors_exist[idx] && (phi_eff_p[neighbors[idx]] < 0.);

//      if (neighbors_exist[nn_m00] &&
//          neighbors_exist[nn_p00] &&
//          neighbors_exist[nn_0m0] &&
//          neighbors_exist[nn_0p0]
//    #ifdef P4_TO_P8
//          && neighbors_exist[nn_00m]
//          && neighbors_exist[nn_00p]
//    #endif
//          )
//        discretization_scheme_ = discretization_scheme_t::FDM;

//      get_all_neighbors(n, neighbors, neighbors_exist);
//    }

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
//          mask_p[n] = -1;

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
          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_mp = 0.5*(mue_000 + mue_p[node_m00_mp])*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pp = 0.5*(mue_000 + mue_p[node_m00_pp])*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          } else {
            w_m00 *= 0.5*(mue_000 + mue_m00);
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_mp = 0.5*(mue_000 + mue_p[node_p00_mp])*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pp = 0.5*(mue_000 + mue_p[node_p00_pp])*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          } else {
            w_p00 *= 0.5*(mue_000 + mue_p00);
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_mp = 0.5*(mue_000 + mue_p[node_0m0_mp])*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pp = 0.5*(mue_000 + mue_p[node_0m0_pp])*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          } else {
            w_0m0 *= 0.5*(mue_000 + mue_0m0);
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_mp = 0.5*(mue_000 + mue_p[node_0p0_mp])*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pp = 0.5*(mue_000 + mue_p[node_0p0_pp])*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          } else {
            w_0p0 *= 0.5*(mue_000 + mue_0p0);
          }

          if(!is_interface_00m && !is_node_zmWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_00m_mm = 0.5*(mue_000 + mue_p[node_00m_mm])*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_mp = 0.5*(mue_000 + mue_p[node_00m_mp])*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pm = 0.5*(mue_000 + mue_p[node_00m_pm])*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pp = 0.5*(mue_000 + mue_p[node_00m_pp])*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          } else {
            w_00m *= 0.5*(mue_000 + mue_00m);
          }

          if(!is_interface_00p && !is_node_zpWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
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

          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
            w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
            w_m00 = w_m00_mm + w_m00_pm;
          } else {
            w_m00 *= 0.5*(mue_000 + mue_m00);
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
            w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
            w_p00    = w_p00_mm + w_p00_pm;
          } else {
            w_p00 *= 0.5*(mue_000 + mue_p00);
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
            w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
            w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
            w_0m0 = w_0m0_mm + w_0m0_pm;
          } else {
            w_0m0 *= 0.5*(mue_000 + mue_0m0);
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni) && (setup_matrix || setup_rhs)) {
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
//          mask_p[n] = -1;
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
          else if(is_interface_m00)      rhs_p[n] -= w_m00*val_interface_m00;

          if(is_node_xpWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_m00;
          else if(is_interface_p00)      rhs_p[n] -= w_p00*val_interface_p00;

          if(is_node_ymWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0p0;
          else if(is_interface_0m0)      rhs_p[n] -= w_0m0*val_interface_0m0;

          if(is_node_ypWall(p4est_, ni)) rhs_p[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0m0;
          else if(is_interface_0p0)      rhs_p[n] -= w_0p0*val_interface_0p0;
#endif

          rhs_p[n] /= w_000;

//          if(!is_interface_m00 && !is_node_xmWall(p4est_, ni)) {
//            if (ABS(w_m00_mm) > EPS) { rhs_p[n] -= exact_ptr[node_m00_mm]*w_m00_mm/w_000; }
//            if (ABS(w_m00_pm) > EPS) { rhs_p[n] -= exact_ptr[node_m00_pm]*w_m00_pm/w_000; }
//#ifdef P4_TO_P8
//            if (ABS(w_m00_mp) > EPS) { rhs_p[n] -= exact_ptr[node_m00_mp]*w_m00_mp/w_000; }
//            if (ABS(w_m00_pp) > EPS) { rhs_p[n] -= exact_ptr[node_m00_pp]*w_m00_pp/w_000; }
//#endif
//          }

//          if(!is_interface_p00 && !is_node_xpWall(p4est_, ni)) {
//            if (ABS(w_p00_mm) > EPS) { rhs_p[n] -= exact_ptr[node_p00_mm]*w_p00_mm/w_000; }
//            if (ABS(w_p00_pm) > EPS) { rhs_p[n] -= exact_ptr[node_p00_pm]*w_p00_pm/w_000; }
//#ifdef P4_TO_P8
//            if (ABS(w_p00_mp) > EPS) { rhs_p[n] -= exact_ptr[node_p00_mp]*w_p00_mp/w_000; }
//            if (ABS(w_p00_pp) > EPS) { rhs_p[n] -= exact_ptr[node_p00_pp]*w_p00_pp/w_000; }
//#endif
//          }

//          if(!is_interface_0m0 && !is_node_ymWall(p4est_, ni)) {
//            if (ABS(w_0m0_mm) > EPS) { rhs_p[n] -= exact_ptr[node_0m0_mm]*w_0m0_mm/w_000; }
//            if (ABS(w_0m0_pm) > EPS) { rhs_p[n] -= exact_ptr[node_0m0_pm]*w_0m0_pm/w_000; }
//#ifdef P4_TO_P8
//            if (ABS(w_0m0_mp) > EPS) { rhs_p[n] -= exact_ptr[node_0m0_mp]*w_0m0_mp/w_000; }
//            if (ABS(w_0m0_pp) > EPS) { rhs_p[n] -= exact_ptr[node_0m0_pp]*w_0m0_pp/w_000; }
//#endif
//          }

//          if(!is_interface_0p0 && !is_node_ypWall(p4est_, ni)) {
//            if (ABS(w_0p0_mm) > EPS) { rhs_p[n] -= exact_ptr[node_0p0_mm]*w_0p0_mm/w_000; }
//            if (ABS(w_0p0_pm) > EPS) { rhs_p[n] -= exact_ptr[node_0p0_pm]*w_0p0_pm/w_000; }
//#ifdef P4_TO_P8
//            if (ABS(w_0p0_mp) > EPS) { rhs_p[n] -= exact_ptr[node_0p0_mp]*w_0p0_mp/w_000; }
//            if (ABS(w_0p0_pp) > EPS) { rhs_p[n] -= exact_ptr[node_0p0_pp]*w_0p0_pp/w_000; }
//#endif
//          }
//#ifdef P4_TO_P8
//          if(!is_interface_00m && !is_node_zmWall(p4est_, ni)) {
//            if (ABS(w_00m_mm) > EPS) { rhs_p[n] -= exact_ptr[node_00m_mm]*w_00m_mm/w_000; }
//            if (ABS(w_00m_pm) > EPS) { rhs_p[n] -= exact_ptr[node_00m_pm]*w_00m_pm/w_000; }
//            if (ABS(w_00m_mp) > EPS) { rhs_p[n] -= exact_ptr[node_00m_mp]*w_00m_mp/w_000; }
//            if (ABS(w_00m_pp) > EPS) { rhs_p[n] -= exact_ptr[node_00m_pp]*w_00m_pp/w_000; }
//          }

//          if(!is_interface_00p && !is_node_zpWall(p4est_, ni)) {
//            if (ABS(w_00p_mm) > EPS) { rhs_p[n] -= exact_ptr[node_00p_mm]*w_00p_mm/w_000; }
//            if (ABS(w_00p_pm) > EPS) { rhs_p[n] -= exact_ptr[node_00p_pm]*w_00p_pm/w_000; }
//            if (ABS(w_00p_mp) > EPS) { rhs_p[n] -= exact_ptr[node_00p_mp]*w_00p_mp/w_000; }
//            if (ABS(w_00p_pp) > EPS) { rhs_p[n] -= exact_ptr[node_00p_pp]*w_00p_pp/w_000; }
//          }
//#endif

        }

        continue;
      }

    }
    else if (discretization_scheme_ == discretization_scheme_t::FVM)
    {
      for (char idx = 0; idx < num_neighbors_max_; ++idx)
        if (neighbors_exist[idx])
          neighbors_exist[idx] = neighbors_exist[idx] && (volumes_p[neighbors[idx]] > domain_rel_thresh);

      // check for hanging neighbors
      bool hanging_neighbor[num_neighbors_max_];

      int network[num_neighbors_max_];

      for (char idx = 0; idx < num_neighbors_max_; ++idx)
        network[idx] = neighbors_exist[idx] ? (int) node_type_p[neighbors[idx]] : 0;

      find_hanging_cells(network, hanging_neighbor);

      bool expand[2*P4EST_DIM];
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

      bool attempt_to_expand = false;

      for (char dir = 0; dir < P4EST_FACES; ++dir)
        attempt_to_expand = attempt_to_expand || expand[dir];

#ifdef P4_TO_P8
      cube3_mls_t cube;
#else
      cube2_mls_t cube;
#endif
      attempt_to_expand = false;
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

        if (!use_refined_cube_)
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
          if (expand[dir::f_m00]) { fv_size_x += cube_refinement_; fv_xmin -= 2.*0.5*dx_min_; }
          if (expand[dir::f_p00]) { fv_size_x += cube_refinement_; fv_xmax += 2.*0.5*dx_min_; }
          if (expand[dir::f_0m0]) { fv_size_y += cube_refinement_; fv_ymin -= 2.*0.5*dy_min_; }
          if (expand[dir::f_0p0]) { fv_size_y += cube_refinement_; fv_ymax += 2.*0.5*dy_min_; }
#ifdef P4_TO_P8
          if (expand[dir::f_00m]) { fv_size_z += cube_refinement_; fv_zmin -= 0.5*dz_min_; }
          if (expand[dir::f_00p]) { fv_size_z += cube_refinement_; fv_zmax += 0.5*dz_min_; }
#endif
        }

        if (volumes_p[n] < 0.001)
        {
          fv_size_x = 4;
          fv_size_y = 4;
#ifdef P4_TO_P8
          fv_size_z = 4;
#endif
        }

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
#ifdef P4_TO_P8
          phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
          phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
          for (int i = 0; i < points_total; ++i)
          {
//#ifdef P4_TO_P8
//            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
//#else
//            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
//#endif
#ifdef P4_TO_P8
            phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i], z_grid[i]);
#else
            phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i]);
#endif
            if (phi_cube[phi_idx*points_total + i] <  phi_perturbation_*dx_min_ &&
                phi_cube[phi_idx*points_total + i] > -phi_perturbation_*dx_min_)
              phi_cube[phi_idx*points_total + i] = phi_perturbation_*dx_min_;
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
//      double interface_area  = 0.;
      double integral_bc = 0.;

      std::vector<double> interface_centroid_x(num_interfaces_, 0);
      std::vector<double> interface_centroid_y(num_interfaces_, 0);
      std::vector<double> interface_centroid_z(num_interfaces_, 0);
      std::vector<double> interface_area(num_interfaces_, 0);

      for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
        if (cube_ifc_w[phi_idx].size() > 0)
        {
          for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
          {
            interface_area[phi_idx] += cube_ifc_w[phi_idx][i];
//            #ifdef P4_TO_P8
//                        integral_bc    += cube_ifc_w[phi_idx][i] * (*bc_interface_value_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i]);
//            #else
//                        integral_bc    += cube_ifc_w[phi_idx][i] * (*bc_interface_value_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i]);
//            #endif
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

#ifdef P4_TO_P8
          integral_bc += interface_area[phi_idx] * (*bc_interface_value_->at(phi_idx))(interface_centroid_x[phi_idx], interface_centroid_y[phi_idx], interface_centroid_z[phi_idx]);
#else
          integral_bc += interface_area[phi_idx] * (*bc_interface_value_->at(phi_idx))(interface_centroid_x[phi_idx], interface_centroid_y[phi_idx]);
#endif
        }
      }

//      if (setup_matrix)
//      {
//        if (volume_cut_cell/full_cell_volume <= domain_rel_thresh)
//          mask_p[n] = 1;
////        else mask_p[n] = -1;

////        if (volume_cut_cell/full_cell_volume < domain_rel_thresh)
////          mask_p[n] = 1;
//        //          else mask_p[n] = -1;
//      }

      if (volume_cut_cell/full_cell_volume > domain_rel_thresh)
//        if (volumes_p[n] > domain_rel_thresh)
      {
        if (setup_rhs)
        {
//          rhs_p[n] = rhs_p[n]*volume_cut_cell + integral_bc;
          rhs_p[n] = rhs_cf_->value(xyz_c_cut_cell)*volume_cut_cell + integral_bc;
        }

        // Compute areas (lengths) and centroid of cut-faces between finite volumes
        delta_x_cf_.set(xyz_C);
        delta_y_cf_.set(xyz_C);
#ifdef P4_TO_P8
        delta_z_cf_.set(xyz_C);
#endif

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

            //          if (area_in_dir[dir_idx]/full_area_in_dir[dir_idx] > interface_rel_thresh)
            //          {
            centroid_x[dir_idx] /= area_in_dir[dir_idx];
            centroid_y[dir_idx] /= area_in_dir[dir_idx];
#ifdef P4_TO_P8
            centroid_z[dir_idx] /= area_in_dir[dir_idx];
#endif
            //          } else {
            //            centroid_x[dir_idx] = 0;
            //            centroid_y[dir_idx] = 0;
            //#ifdef P4_TO_P8
            //            centroid_z[dir_idx] = 0;
            //#endif
            //          }
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
        if (s_m00/full_sx > interface_rel_thresh)
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
        if (s_p00/full_sx > interface_rel_thresh)
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
        if (s_0m0/full_sy > interface_rel_thresh)
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
        if (s_0p0/full_sy > interface_rel_thresh)
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
        if (s_00m/full_sz > interface_rel_thresh)
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
        if (s_00p/full_sz > interface_rel_thresh)
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

//        theta = 100;

        //*

        // face m00
        if (s_m00/full_sx > interface_rel_thresh)
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
//                  mask_p[n] = 10;
                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face p00
        if (s_p00/full_sx > interface_rel_thresh)
        {

          if (!neighbors_exist[nn_p00])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_p00]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_p00]]
                         << " Face Area:" << s_p00/full_sx << "\n";
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
//                  mask_p[n] = 10;
                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face_0m0
        if (s_0m0/full_sy > interface_rel_thresh)
        {

          if (!neighbors_exist[nn_0m0])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0m0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0m0]]
                         << " Face Area:" << s_0m0/full_sy << "\n";
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
//                  mask_p[n] = 10;
                  std::cout << "Fallback fluxes\n";
                }
              }
            }
          }
        }

        // face_0p0
        if (s_0p0/full_sy > interface_rel_thresh)
        {

          if (!neighbors_exist[nn_0p0])
          {
            std::cout << "Warning: neighbor doesn't exist in the xm-direction."
                      << " Own number: " << n
                      << " Nei number: " << neighbors[nn_0p0]
                         << " Own volume: " << volumes_p[neighbors[nn_000]]
                         << " Nei volume: " << volumes_p[neighbors[nn_0p0]]
                         << " Face Area:" << s_0p0/full_sy << "\n";
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
//                  mask_p[n] = 10;
                  std::cout << "Fallback fluxes\n";
                }
              }
            }

          }
        }

        //*/


        /* fluxes between cells taking into account more points and boundary conditions and using least-squares - didn't work out
        if (use_sc_scheme_)
        {
          // linear system
          char num_constraints = num_neighbors_max_ + num_interfaces_;

          std::vector<double> col_1st(num_constraints, 0);
          std::vector<double> col_2nd(num_constraints, 0);
          std::vector<double> col_3rd(num_constraints, 0);
          std::vector<double> col_4th(num_constraints, 0);

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
                col_4th[idx] = col_2nd[idx]*col_3rd[idx];
              }

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
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

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              col_1st[num_neighbors_max_ + phi_idx] = bc_coeff_proj;
              col_2nd[num_neighbors_max_ + phi_idx] = (mu_proj*nx + bc_coeff_proj*(xyz_pr[0] - xyz_C[0]));
              col_3rd[num_neighbors_max_ + phi_idx] = (mu_proj*ny + bc_coeff_proj*(xyz_pr[1] - xyz_C[1]));
              col_4th[num_neighbors_max_ + phi_idx] = (mu_proj*nx*(xyz_pr[1] - xyz_C[1])
                                                     + mu_proj*ny*(xyz_pr[0] - xyz_C[0])
                                                     + bc_coeff_proj*(xyz_pr[0] - xyz_C[0])*(xyz_pr[1] - xyz_C[1]));

              bc_coeffs[phi_idx] = bc_coeff_proj;
              bc_values[phi_idx] = bc_value_proj;
            }
          }

          double log_weight_min = -6.;
//          double gamma = -log_weight_min*2./3./sqrt((double)P4EST_DIM);
          double gamma = -log_weight_min*2./3./((double)P4EST_DIM);

          double x_dir_c;
          double y_dir_c;
          double z_dir_c;
          double centr;

          char nei_idx;

          for (char dir_idx = 0; dir_idx < P4EST_FACES; ++dir_idx)
          {
            switch (dir_idx)
            {
              case 0: x_dir_c = x_C - .5*dx_min_; y_dir_c = y_C;              centr = centroid_y[dir_idx]; nei_idx = nn_m00; break;
              case 1: x_dir_c = x_C + .5*dx_min_; y_dir_c = y_C;              centr = centroid_y[dir_idx]; nei_idx = nn_p00; break;
              case 2: x_dir_c = x_C;              y_dir_c = y_C - .5*dy_min_; centr = centroid_x[dir_idx]; nei_idx = nn_0m0; break;
              case 3: x_dir_c = x_C;              y_dir_c = y_C + .5*dy_min_; centr = centroid_x[dir_idx]; nei_idx = nn_0p0; break;
              default: throw;
            }

            if (area_in_dir[dir_idx]/full_area_in_dir[dir_idx] > interface_rel_thresh && neighbors_exist[nei_idx])
            {
              std::vector<double> weight(num_constraints, 0);

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

#ifdef P4_TO_P8
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_) +
//                                                    SQR((z_C+dz-z0[phi_idx])/dz_min_)));
                    if (neighbors_exist[idx])
                      weight[idx] = exp(-gamma*(SQR((x_C+dx-x_dir_c)/dx_min_) +
                                                SQR((y_C+dy-y_dir_c)/dy_min_) +
                                                SQR((z_C+dz-z_dir_c)/dz_min_)));
#else
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_)));
//                    if (neighbors_exist[idx])
//                      weight[idx] =
////                          1.;
////                          exp(-gamma*(SQR((x_C+dx-x_dir_c)/dx_min_) +
////                                      SQR((y_C+dy-y_dir_c)/dy_min_)));
//                          exp(-gamma*(SQR((dx-centroid_x[dir_idx])/dx_min_) +
//                                      SQR((dy-centroid_y[dir_idx])/dy_min_)));

//                    if (neighbors_exist[idx] && fabs(dx-centroid_x[dir_idx]) <= dx_min_ && fabs(dy-centroid_y[dir_idx]) <= dy_min_)
                    if (neighbors_exist[idx])
//                      if (fabs(dx-centroid_x[dir_idx]) <= dx_min_ && fabs(dy-centroid_y[dir_idx]) <= dy_min_)
                      weight[idx] =
                          1.e-5*
//                          1.;
//                          exp(-gamma*(SQR((x_C+dx-x_dir_c)/dx_min_) +
//                                      SQR((y_C+dy-y_dir_c)/dy_min_)));
                          exp(-gamma*(SQR((dx-centroid_x[dir_idx])/dx_min_) +
                                      SQR((dy-centroid_y[dir_idx])/dy_min_)));

//                    if (neighbors_exist[idx] && fabs(dx-centroid_x[dir_idx]) <= dx_min_ && fabs(dy-centroid_y[dir_idx]) <= dy_min_)
//                      weight[idx] =
//                          1.;
                    if (idx == nn_000 || idx == nei_idx)
                      weight[idx] =
                          1.;
#endif
                  }

              for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                if (cube_ifc_w[phi_jdx].size() > 0)
                {
#ifdef P4_TO_P8
//                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
//                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_) +
//                                                                         SQR((z0[phi_jdx]-z0[phi_idx])/dz_min_)));

                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x_dir_c)/dx_min_) +
                                                                     SQR((y0[phi_jdx]-y_dir_c)/dy_min_) +
                                                                     SQR((z0[phi_jdx]-z_dir_c)/dz_min_)));
#else
//                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
//                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_)));

                  weight[num_neighbors_max_ + phi_jdx] =
                      1.e-5*
//                      1.;
////                      exp(-gamma*(SQR((x0[phi_jdx]-x_dir_c)/dx_min_) +
////                                  SQR((y0[phi_jdx]-y_dir_c)/dy_min_)));
                      exp(-gamma*(SQR((x0[phi_jdx]-x_C-centroid_x[dir_idx])/dx_min_) +
                                  SQR((y0[phi_jdx]-y_C-centroid_y[dir_idx])/dy_min_)));
#endif
                }
              }

              // assemble and invert matrix
              char A_size = 4;
              double A[(4)*(4)];
              double A_inv[(4)*(4)];

              A[0*A_size + 0] = 0;
              A[0*A_size + 1] = 0;
              A[0*A_size + 2] = 0;
              A[1*A_size + 1] = 0;
              A[1*A_size + 2] = 0;
              A[2*A_size + 2] = 0;
              A[0*A_size + 3] = 0;
              A[1*A_size + 3] = 0;
              A[2*A_size + 3] = 0;
              A[3*A_size + 3] = 0;

              for (char nei = 0; nei < num_constraints; ++nei)
              {
                A[0*A_size + 0] += col_1st[nei]*col_1st[nei]*weight[nei];
                A[0*A_size + 1] += col_1st[nei]*col_2nd[nei]*weight[nei];
                A[0*A_size + 2] += col_1st[nei]*col_3rd[nei]*weight[nei];
                A[1*A_size + 1] += col_2nd[nei]*col_2nd[nei]*weight[nei];
                A[1*A_size + 2] += col_2nd[nei]*col_3rd[nei]*weight[nei];
                A[2*A_size + 2] += col_3rd[nei]*col_3rd[nei]*weight[nei];
                A[0*A_size + 3] += col_1st[nei]*col_4th[nei]*weight[nei];
                A[1*A_size + 3] += col_2nd[nei]*col_4th[nei]*weight[nei];
                A[2*A_size + 3] += col_3rd[nei]*col_4th[nei]*weight[nei];
                A[3*A_size + 3] += col_4th[nei]*col_4th[nei]*weight[nei];
              }

              A[1*A_size + 0] = A[0*A_size + 1];
              A[2*A_size + 0] = A[0*A_size + 2];
              A[2*A_size + 1] = A[1*A_size + 2];
              A[3*A_size + 0] = A[0*A_size + 3];
              A[3*A_size + 1] = A[1*A_size + 3];
              A[3*A_size + 2] = A[2*A_size + 3];

#ifdef P4_TO_P8
#else
              if (!inv_mat4_(A, A_inv)) throw;
#endif

              // compute Taylor expansion coefficients
              std::vector<double> coeff_const_term(num_constraints, 0);
              std::vector<double> coeff_x_term    (num_constraints, 0);
              std::vector<double> coeff_y_term    (num_constraints, 0);
              std::vector<double> coeff_xy_term   (num_constraints, 0);
#ifdef P4_TO_P8
              std::vector<double> coeff_z_term    (num_constraints, 0);
#endif

              for (char nei = 0; nei < num_constraints; ++nei)
              {
#ifdef P4_TO_P8
#else
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

                coeff_xy_term[nei] = weight[nei]*
                    ( A_inv[3*A_size+0]*col_1st[nei]
                    + A_inv[3*A_size+1]*col_2nd[nei]
                    + A_inv[3*A_size+2]*col_3rd[nei]
                    + A_inv[3*A_size+3]*col_4th[nei] );
#endif
              }

              double rhs_const_term = 0;
              double rhs_x_term     = 0;
              double rhs_y_term     = 0;
              double rhs_xy_term    = 0;
#ifdef P4_TO_P8
              double rhs_z_term     = 0;
#endif
              for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                rhs_const_term += coeff_const_term[num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
                rhs_x_term     += coeff_x_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
                rhs_y_term     += coeff_y_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
                rhs_xy_term    += coeff_xy_term   [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
#ifdef P4_TO_P8
                rhs_z_term     += coeff_z_term    [num_neighbors_max_ + phi_jdx] * bc_values[phi_jdx];
#endif

              }

              // compute integrals
              double x_term     = area_in_dir[dir_idx];
              double y_term     = area_in_dir[dir_idx];
              double xy_term    = 1.0*area_in_dir[dir_idx]*centr;

              // matrix coefficients
              for (char nei = 0; nei < num_neighbors_max_; ++nei)
              {
#ifdef P4_TO_P8
#else

                switch (dir_idx)
                {
                  case 0: w[nei] += coeff_x_term[nei]*x_term + coeff_xy_term[nei]*xy_term; break;
                  case 1: w[nei] -= coeff_x_term[nei]*x_term + coeff_xy_term[nei]*xy_term; break;
                  case 2: w[nei] += coeff_y_term[nei]*y_term + coeff_xy_term[nei]*xy_term; break;
                  case 3: w[nei] -= coeff_y_term[nei]*y_term + coeff_xy_term[nei]*xy_term; break;
                  default: throw;
                }

//                w[nei] += coeff_xy_term[nei]*xy_term;
#endif
              }

//              for (char nei = 0; nei < num_neighbors_max_; ++nei)
//              {
//#ifdef P4_TO_P8
//#else
//                rhs_p[n] -= coeff_xy_term[nei]*xy_term*exact_ptr[neighbors[nei]];
//#endif
//              }


              if (setup_rhs)
#ifdef P4_TO_P8
#else
                switch (dir_idx)
                {
                  case 0: rhs_p[n] -= rhs_x_term*x_term + rhs_xy_term*xy_term; break;
                  case 1: rhs_p[n] += rhs_x_term*x_term + rhs_xy_term*xy_term; break;
                  case 2: rhs_p[n] -= rhs_y_term*y_term + rhs_xy_term*xy_term; break;
                  case 3: rhs_p[n] += rhs_y_term*y_term + rhs_xy_term*xy_term; break;
                  default: throw;
                }
//                rhs_p[n] -= rhs_xy_term*xy_term;
#endif

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

        char num_ifc = 0;

        for (int phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
        {
          if (cube_ifc_w[phi_idx].size() > 0)
            ++num_ifc;
        }

        /*
        // Scheme based on bi-linear interpolation
        if (use_sc_scheme_)
//          if (use_sc_scheme_ && num_ifc < 2)
        {
          sc_scheme_successful = true;
          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif
              // compute centroid of an interface
              double x0 = 0;
              double y0 = 0;
#ifdef P4_TO_P8
              double z0 = 0;
#endif
              double S  = 0;

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                x0 += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0 += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
                S  += cube_ifc_w[phi_idx][i];
              }

//              if (S/dx_min_ > interface_rel_thresh)
              {
                x0 /= S;
                y0 /= S;
#ifdef P4_TO_P8
                z0 /= S;
#endif
              }

              // compute weights of multi-linear interpolation
              double d_m00 = fabs(x0-x_C)/dx_min_, d_p00 = 1.-d_m00;
              double d_0m0 = fabs(y0-y_C)/dy_min_, d_0p0 = 1.-d_0m0;
#ifdef P4_TO_P8
              double d_00m = fabs(z0-z_C)/dz_min_, d_00p = 1.-d_00m;
#endif

#ifdef P4_TO_P8
//              double w_xyz[] =
//              {
//                1,
//                0,
//                0,
//                0,
//                0,
//                0,
//                0,
//                0
//              };
              double w_xyz[] =
              {
                d_p00*d_0p0*d_00p,
                d_m00*d_0p0*d_00p,
                d_p00*d_0m0*d_00p,
                d_m00*d_0m0*d_00p,
                d_p00*d_0p0*d_00m,
                d_m00*d_0p0*d_00m,
                d_p00*d_0m0*d_00m,
                d_m00*d_0m0*d_00m
              };
#else
              double w_xyz[] =
              {
                d_p00*d_0p0,
                d_m00*d_0p0,
                d_p00*d_0m0,
                d_m00*d_0m0
              };
//              double w_xyz[] =
//              {
//                1,
//                0,
//                0,
//                0
//              };
#endif

              char nei_mmm = nn_000;
              char nei_pmm = nn_p00;
              char nei_mpm = nn_0p0;
              char nei_ppm = nn_pp0;
#ifdef P4_TO_P8
              char nei_mmp;
              char nei_pmp;
              char nei_mpp;
              char nei_ppp;
#endif

#ifdef P4_TO_P8
              if      (x0 >= x_C && y0 >= y_C && z0 >= z_C) { nei_pmm = nn_p00; nei_mpm = nn_0p0; nei_ppm = nn_pp0; nei_mmp = nn_00p; nei_pmp = nn_p0p; nei_mpp = nn_0pp; nei_ppp = nn_ppp; }
              else if (x0 <  x_C && y0 >= y_C && z0 >= z_C) { nei_pmm = nn_m00; nei_mpm = nn_0p0; nei_ppm = nn_mp0; nei_mmp = nn_00p; nei_pmp = nn_m0p; nei_mpp = nn_0pp; nei_ppp = nn_mpp; }
              else if (x0 >= x_C && y0 <  y_C && z0 >= z_C) { nei_pmm = nn_p00; nei_mpm = nn_0m0; nei_ppm = nn_pm0; nei_mmp = nn_00p; nei_pmp = nn_p0p; nei_mpp = nn_0mp; nei_ppp = nn_pmp; }
              else if (x0 <  x_C && y0 <  y_C && z0 >= z_C) { nei_pmm = nn_m00; nei_mpm = nn_0m0; nei_ppm = nn_mm0; nei_mmp = nn_00p; nei_pmp = nn_m0p; nei_mpp = nn_0mp; nei_ppp = nn_mmp; }
              else if (x0 >= x_C && y0 >= y_C && z0 <  z_C) { nei_pmm = nn_p00; nei_mpm = nn_0p0; nei_ppm = nn_pp0; nei_mmp = nn_00m; nei_pmp = nn_p0m; nei_mpp = nn_0pm; nei_ppp = nn_ppm; }
              else if (x0 <  x_C && y0 >= y_C && z0 <  z_C) { nei_pmm = nn_m00; nei_mpm = nn_0p0; nei_ppm = nn_mp0; nei_mmp = nn_00m; nei_pmp = nn_m0m; nei_mpp = nn_0pm; nei_ppp = nn_mpm; }
              else if (x0 >= x_C && y0 <  y_C && z0 <  z_C) { nei_pmm = nn_p00; nei_mpm = nn_0m0; nei_ppm = nn_pm0; nei_mmp = nn_00m; nei_pmp = nn_p0m; nei_mpp = nn_0mm; nei_ppp = nn_pmm; }
              else if (x0 <  x_C && y0 <  y_C && z0 <  z_C) { nei_pmm = nn_m00; nei_mpm = nn_0m0; nei_ppm = nn_mm0; nei_mmp = nn_00m; nei_pmp = nn_m0m; nei_mpp = nn_0mm; nei_ppp = nn_mmm; }
              else { throw; }
#else
              if      (x0 >= x_C && y0 >= y_C) { nei_pmm = nn_p00; nei_mpm = nn_0p0; nei_ppm = nn_pp0; }
              else if (x0 <  x_C && y0 >= y_C) { nei_pmm = nn_m00; nei_mpm = nn_0p0; nei_ppm = nn_mp0; }
              else if (x0 >= x_C && y0 <  y_C) { nei_pmm = nn_p00; nei_mpm = nn_0m0; nei_ppm = nn_pm0; }
              else if (x0 <  x_C && y0 <  y_C) { nei_pmm = nn_m00; nei_mpm = nn_0m0; nei_ppm = nn_mm0; }
              else { throw; }
#endif

              // compute signed distance and normal at the centroid
#ifdef P4_TO_P8
              double xyz0[P4EST_DIM] = { x0, y0, z0 };
#else
              double xyz0[P4EST_DIM] = { x0, y0 };
#endif

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

//              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz0);
//              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              double robin_term = bc_coeff_proj*S;

              w[nei_mmm] += robin_term*w_xyz[dir::v_mmm];
              w[nei_pmm] += robin_term*w_xyz[dir::v_pmm];
              w[nei_mpm] += robin_term*w_xyz[dir::v_mpm];
              w[nei_ppm] += robin_term*w_xyz[dir::v_ppm];
#ifdef P4_TO_P8
              w[nei_mmp] += robin_term*w_xyz[dir::v_mmp];
              w[nei_pmp] += robin_term*w_xyz[dir::v_pmp];
              w[nei_mpp] += robin_term*w_xyz[dir::v_mpp];
              w[nei_ppp] += robin_term*w_xyz[dir::v_ppp];
#endif

              if (setup_matrix)
              {
                if (!neighbors_exist[nei_pmm]) { mask_p[neighbors[nei_pmm]] = 777; neighbors_exist[nei_pmm] = true; }
                if (!neighbors_exist[nei_mpm]) { mask_p[neighbors[nei_mpm]] = 777; neighbors_exist[nei_mpm] = true; }
                if (!neighbors_exist[nei_ppm]) { mask_p[neighbors[nei_ppm]] = 777; neighbors_exist[nei_ppm] = true; }
#ifdef P4_TO_P8
                if (!neighbors_exist[nei_mmp]) { mask_p[neighbors[nei_mmp]] = 777; neighbors_exist[nei_mmp] = true; }
                if (!neighbors_exist[nei_pmp]) { mask_p[neighbors[nei_pmp]] = 777; neighbors_exist[nei_pmp] = true; }
                if (!neighbors_exist[nei_mpp]) { mask_p[neighbors[nei_mpp]] = 777; neighbors_exist[nei_mpp] = true; }
                if (!neighbors_exist[nei_ppp]) { mask_p[neighbors[nei_ppp]] = 777; neighbors_exist[nei_ppp] = true; }
#endif
              }

              if (setup_matrix && fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
            }
        }
        //*/

        //*
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
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
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

          double log_weight_min = -6.;
//          double gamma = -log_weight_min*2./3./sqrt((double)P4EST_DIM);
          double gamma = -log_weight_min*2./3./((double)P4EST_DIM);

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0)
            {
              std::vector<double> weight(num_constraints, 0);

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

#ifdef P4_TO_P8
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_) +
//                                                    SQR((z_C+dz-z0[phi_idx])/dz_min_)));
                    if (neighbors_exist[idx])
                      weight[idx] = exp(-gamma*(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
                                                SQR((y_C+dy-y0[phi_idx])/dy_min_) +
                                                SQR((z_C+dz-z0[phi_idx])/dz_min_)));
#else
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_)));
                    if (neighbors_exist[idx])
                      weight[idx] = exp(-gamma*(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
                                                SQR((y_C+dy-y0[phi_idx])/dy_min_)));
#endif
                  }

              for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                if (cube_ifc_w[phi_jdx].size() > 0)
                {
#ifdef P4_TO_P8
//                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
//                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_) +
//                                                                         SQR((z0[phi_jdx]-z0[phi_idx])/dz_min_)));

                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
                                                                     SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_) +
                                                                     SQR((z0[phi_jdx]-z0[phi_idx])/dz_min_)));
#else
//                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
//                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_)));

                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
                                                                     SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_)));
#endif
                }
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
              inv_mat3_(A, A_inv);
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
                    + A_inv[1*A_size+3]*col_4th[nei] );//dx_min_;

                coeff_y_term[nei] = weight[nei]*
                    ( A_inv[2*A_size+0]*col_1st[nei]
                    + A_inv[2*A_size+1]*col_2nd[nei]
                    + A_inv[2*A_size+2]*col_3rd[nei]
                    + A_inv[2*A_size+3]*col_4th[nei] );//dy_min_;

                coeff_z_term[nei] = weight[nei]*
                    ( A_inv[3*A_size+0]*col_1st[nei]
                    + A_inv[3*A_size+1]*col_2nd[nei]
                    + A_inv[3*A_size+2]*col_3rd[nei]
                    + A_inv[3*A_size+3]*col_4th[nei] );//dz_min_;
#else
                coeff_const_term[nei] = weight[nei]*
                    ( A_inv[0*A_size+0]*col_1st[nei]
                    + A_inv[0*A_size+1]*col_2nd[nei]
                    + A_inv[0*A_size+2]*col_3rd[nei] );//dx_min_;

                coeff_x_term[nei] = weight[nei]*
                    ( A_inv[1*A_size+0]*col_1st[nei]
                    + A_inv[1*A_size+1]*col_2nd[nei]
                    + A_inv[1*A_size+2]*col_3rd[nei] );//dy_min_;

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

          if (1)
          {
          std::vector<double> weight(num_constraints, 0);

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

#ifdef P4_TO_P8
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_) +
//                                                    SQR((z_C+dz-z0[phi_idx])/dz_min_)));
                if (neighbors_exist[idx])
//                  weight[idx] = 1.;
                  weight[idx] = exp(-gamma*(SQR((x_C+dx-x_ctrd_cut_cell)/dx_min_) +
                                            SQR((y_C+dy-y_ctrd_cut_cell)/dy_min_) +
                                            SQR((z_C+dz-z_ctrd_cut_cell)/dz_min_)));
#else
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_)));
                if (neighbors_exist[idx])
                  weight[idx] = exp(-gamma*(SQR((x_C+dx-x_ctrd_cut_cell)/dx_min_) +
                                            SQR((y_C+dy-y_ctrd_cut_cell)/dy_min_)));
#endif
              }

//          for (char phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
//          {
//            if (cube_ifc_w[phi_jdx].size() > 0)
//            {
//#ifdef P4_TO_P8
////                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
////                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_) +
////                                                                         SQR((z0[phi_jdx]-z0[phi_idx])/dz_min_)));

//              weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x_ctrd_cut_cell)/dx_min_) +
//                                                                 SQR((y0[phi_jdx]-y_ctrd_cut_cell)/dy_min_) +
//                                                                 SQR((z0[phi_jdx]-z_ctrd_cut_cell)/dz_min_)));
//#else
////                  weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*sqrt(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
////                                                                         SQR((y0[phi_jdx]-y0[phi_idx])/dy_min_)));

//              weight[num_neighbors_max_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x_ctrd_cut_cell)/dx_min_) +
//                                                                 SQR((y0[phi_jdx]-y_ctrd_cut_cell)/dy_min_)));
//#endif
//            }
//          }

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
          inv_mat3_(A, A_inv);
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
                + A_inv[1*A_size+3]*col_4th[nei] );//dx_min_;

            coeff_y_term[nei] = weight[nei]*
                ( A_inv[2*A_size+0]*col_1st[nei]
                + A_inv[2*A_size+1]*col_2nd[nei]
                + A_inv[2*A_size+2]*col_3rd[nei]
                + A_inv[2*A_size+3]*col_4th[nei] );//dy_min_;

            coeff_z_term[nei] = weight[nei]*
                ( A_inv[3*A_size+0]*col_1st[nei]
                + A_inv[3*A_size+1]*col_2nd[nei]
                + A_inv[3*A_size+2]*col_3rd[nei]
                + A_inv[3*A_size+3]*col_4th[nei] );//dz_min_;
#else
            coeff_const_term[nei] = weight[nei]*
                ( A_inv[0*A_size+0]*col_1st[nei]
                + A_inv[0*A_size+1]*col_2nd[nei]
                + A_inv[0*A_size+2]*col_3rd[nei] );//dx_min_;

            coeff_x_term[nei] = weight[nei]*
                ( A_inv[1*A_size+0]*col_1st[nei]
                + A_inv[1*A_size+1]*col_2nd[nei]
                + A_inv[1*A_size+2]*col_3rd[nei] );//dy_min_;

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
          double const_term = diag_add_p[n]*volume_cut_cell;
          double x_term     = diag_add_p[n]*volume_cut_cell*(x_ctrd_cut_cell - xyz_C[0]);
          double y_term     = diag_add_p[n]*volume_cut_cell*(y_ctrd_cut_cell - xyz_C[1]);
#ifdef P4_TO_P8
          double z_term     = diag_add_p[n]*volume_cut_cell*(z_ctrd_cut_cell - xyz_C[2]);
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

        //*/

        /*
        // a variation of the least-square fitting approach
        if (use_sc_scheme_)
        {
          sc_scheme_successful = true;

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif
              // compute centroid of an interface
              double x0 = 0;
              double y0 = 0;
#ifdef P4_TO_P8
              double z0 = 0;
#endif
              double S  = 0;

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                x0 += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0 += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
                S  += cube_ifc_w[phi_idx][i];
              }

//              if (S/dx_min_ > interface_rel_thresh)
              {
                x0 /= S;
                y0 /= S;
#ifdef P4_TO_P8
                z0 /= S;
#endif
              }

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

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              char num_constraints = num_neighbors_max_ + 1;

              // linear system
              std::vector<double> weight (num_constraints, 0);
              std::vector<double> col_1st(num_constraints, 0);
              std::vector<double> col_2nd(num_constraints, 0);
              std::vector<double> col_3rd(num_constraints, 0);
#ifdef P4_TO_P8
              std::vector<double> col_4th(num_constraints, 0);
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

                    double dx = ((double) (i-1)) * dxyz_m_[0];
                    double dy = ((double) (j-1)) * dxyz_m_[1];
#ifdef P4_TO_P8
                    double dz = ((double) (k-1)) * dxyz_m_[2];
#endif

                    double log_weight_min = -6.;
//                    double gamma = -log_weight_min*2./3./sqrt((double)P4EST_DIM);
                    double gamma = -log_weight_min*2./3./SQR((double)P4EST_DIM);
//                    if (neighbors_exist[idx] && volumes_p[neighbors[idx]] > 1.e-6) weight[idx] = 1.;
//                    if (neighbors_exist[idx]) weight[idx] = 1./(SQR(SQR((x_C+dx-x0)/dx_min_) +
//                                                                    SQR((y_C+dy-y0)/dy_min_)) + 1.e-5);
#ifdef P4_TO_P8
//                    if (neighbors_exist[idx] && volumes_p[neighbors[idx]] > 1.e-6 || idx == nn_000)
//                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0)/dx_min_) +
//                                                    SQR((y_C+dy-y0)/dy_min_) +
                    //                                                    SQR((z_C+dz-z0)/dz_min_)));
                    if (neighbors_exist[idx]) weight[idx] = exp(-gamma*(SQR((x_C+dx-x0)/dx_min_) +
                                                                        SQR((y_C+dy-y0)/dy_min_) +
                                                                        SQR((z_C+dz-z0)/dz_min_)));
#else
                    if (neighbors_exist[idx])
                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0)/dx_min_) +
                                                    SQR((y_C+dy-y0)/dy_min_)));
//                    if (neighbors_exist[idx])
//                      weight[idx] = exp(-gamma*(SQR((x_C+dx-x0)/dx_min_) +
//                                                SQR((y_C+dy-y0)/dy_min_)));
#endif
//                    if (idx == nn_000) weight[idx] = 1.e5;
                  }

              col_1st[num_neighbors_max_] = bc_coeff_proj;
              col_2nd[num_neighbors_max_] = (mu_proj*nx + bc_coeff_proj*(xyz_pr[0] - xyz_C[0]));//dxyz_m_[0];
              col_3rd[num_neighbors_max_] = (mu_proj*ny + bc_coeff_proj*(xyz_pr[1] - xyz_C[1]));//dxyz_m_[1];
#ifdef P4_TO_P8
              col_4th[num_neighbors_max_] = (mu_proj*nz + bc_coeff_proj*(xyz_pr[2] - xyz_C[2]));//dxyz_m_[2];
#endif
              weight [num_neighbors_max_] = 1.;

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
              inv_mat3_(A, A_inv);
    #endif

              // compute Taylor expansion coefficients
              double coeff_const_term[num_neighbors_max_ + 1];
              double coeff_x_term[num_neighbors_max_ + 1];
              double coeff_y_term[num_neighbors_max_ + 1];
    #ifdef P4_TO_P8
              double coeff_z_term[num_neighbors_max_ + 1];
    #endif

              for (char nei = 0; nei < num_constraints; ++nei)
              {
    #ifdef P4_TO_P8
                coeff_const_term[nei] = weight[nei]*
                    ( A_inv[0*A_size+0]*col_1st[nei] +
                      A_inv[0*A_size+1]*col_2nd[nei] +
                      A_inv[0*A_size+2]*col_3rd[nei] +
                      A_inv[0*A_size+3]*col_4th[nei] );

                coeff_x_term[nei] = weight[nei]*
                    ( A_inv[1*A_size+0]*col_1st[nei] +
                      A_inv[1*A_size+1]*col_2nd[nei] +
                      A_inv[1*A_size+2]*col_3rd[nei] +
                      A_inv[1*A_size+3]*col_4th[nei] );//dx_min_;

                coeff_y_term[nei] = weight[nei]*
                    ( A_inv[2*A_size+0]*col_1st[nei] +
                      A_inv[2*A_size+1]*col_2nd[nei] +
                      A_inv[2*A_size+2]*col_3rd[nei] +
                      A_inv[2*A_size+3]*col_4th[nei] );//dy_min_;

                coeff_z_term[nei] = weight[nei]*
                    ( A_inv[3*A_size+0]*col_1st[nei] +
                      A_inv[3*A_size+1]*col_2nd[nei] +
                      A_inv[3*A_size+2]*col_3rd[nei] +
                      A_inv[3*A_size+3]*col_4th[nei] );//dz_min_;
    #else
                coeff_const_term[nei] = weight[nei]*
                    ( A_inv[0*A_size+0]*col_1st[nei] +
                      A_inv[0*A_size+1]*col_2nd[nei] +
                      A_inv[0*A_size+2]*col_3rd[nei] );//dx_min_;

                coeff_x_term[nei] = weight[nei]*
                    ( A_inv[1*A_size+0]*col_1st[nei] +
                      A_inv[1*A_size+1]*col_2nd[nei] +
                      A_inv[1*A_size+2]*col_3rd[nei] );//dy_min_;

                coeff_y_term[nei] = weight[nei]*
                    ( A_inv[2*A_size+0]*col_1st[nei] +
                      A_inv[2*A_size+1]*col_2nd[nei] +
                      A_inv[2*A_size+2]*col_3rd[nei] );
    #endif
              }

    #ifdef P4_TO_P8
              double rhs_const_term = coeff_const_term[num_neighbors_max_] * bc_value_proj;
              double rhs_x_term     = coeff_x_term    [num_neighbors_max_] * bc_value_proj;
              double rhs_y_term     = coeff_y_term    [num_neighbors_max_] * bc_value_proj;
              double rhs_z_term     = coeff_z_term    [num_neighbors_max_] * bc_value_proj;
    #else
              double rhs_const_term = coeff_const_term[num_neighbors_max_] * bc_value_proj;
              double rhs_x_term     = coeff_x_term    [num_neighbors_max_] * bc_value_proj;
              double rhs_y_term     = coeff_y_term    [num_neighbors_max_] * bc_value_proj;
    #endif

              // compute integrals
              double const_term = S*bc_coeff_proj;
              double x_term     = S*bc_coeff_proj*(xyz0[0] - xyz_C[0]);
              double y_term     = S*bc_coeff_proj*(xyz0[1] - xyz_C[1]);
#ifdef P4_TO_P8
              double z_term     = S*bc_coeff_proj*(xyz0[2] - xyz_C[2]);
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


        /*
        // scheme employing least-square fitting
//        if (use_sc_scheme_ && num_ifc < 2)
        if (use_sc_scheme_)
        {
          sc_scheme_successful = true;
          char num_constraints = num_neighbors_max_ + P4EST_DIM;

          // linear system
          std::vector<double> weight (num_neighbors_max_ + P4EST_DIM, 0);
          std::vector<double> col_1st(num_neighbors_max_ + P4EST_DIM, 0);
          std::vector<double> col_2nd(num_neighbors_max_ + P4EST_DIM, 0);
          std::vector<double> col_3rd(num_neighbors_max_ + P4EST_DIM, 0);
#ifdef P4_TO_P8
          std::vector<double> col_4th(num_neighbors_max_ + P4EST_DIM, 0);
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
//              col_2nd[idx] = ((double) (i-1));
//              col_3rd[idx] = ((double) (j-1));
//#ifdef P4_TO_P8
//              col_4th[idx] = ((double) (k-1));
//#endif
              col_2nd[idx] = ((double) (i-1)) * dxyz_m_[0];
              col_3rd[idx] = ((double) (j-1)) * dxyz_m_[1];
#ifdef P4_TO_P8
              col_4th[idx] = ((double) (k-1)) * dxyz_m_[2];
#endif

              if (neighbors_exist[idx] && volumes_p[neighbors[idx]] > 1.e-6) weight[idx] = 1.;
//              if (neighbors_exist[idx] && volumes_p[neighbors[idx]] == 1.) weight[idx] = 0.1;
//              if (neighbors_exist[idx]) weight[idx] = 1./(double)(pow(10, abs(i)+abs(j)+abs(k)));
//#ifdef P4_TO_P8
//              if (neighbors_exist[idx]) weight[idx] = 1./(double)(pow(10, abs(i-1)+abs(j-1)+abs(k-1)));
//#else
//              if (neighbors_exist[idx]) weight[idx] = 1./(double)(pow(10, abs(i-1)+abs(j-1)));
              if (neighbors_exist[idx]) weight[idx] = 1./(sqrt(SQR(i-1)+SQR(j-1))+0.0001);
//#endif
//              if (neighbors_exist[idx] && volumes_p[neighbors[idx]] > 1.e-6) weight[idx] = volumes_p[neighbors[idx]];
//              if (neighbors_exist[idx]) weight[idx] = dx_min_/MIN(dx_min_*1.e-6, phi_eff_p[neighbors[idx]]);
//              if (idx == nn_000) weight[idx] = 1.e1;
//              if (idx == nn_000) weight[idx] = 1.;
            }

          std::vector<double> bc_values(P4EST_DIM, 0);
          std::vector<double> bc_coeffs(num_interfaces_, 0);

          char num_present_interfaces = 0;

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif
              // compute centroid of an interface
              double x0 = 0;
              double y0 = 0;
#ifdef P4_TO_P8
              double z0 = 0;
#endif
              double S  = 0;

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                x0 += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0 += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
                S  += cube_ifc_w[phi_idx][i];
              }

//              if (S/dx_min_ > interface_rel_thresh)
              {
                x0 /= S;
                y0 /= S;
#ifdef P4_TO_P8
                z0 /= S;
#endif
              }

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

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              col_1st[num_neighbors_max_+num_present_interfaces] = bc_coeff_proj;
              col_2nd[num_neighbors_max_+num_present_interfaces] = (mu_proj*nx + bc_coeff_proj*(xyz_pr[0] - xyz_C[0]));//dxyz_m_[0];
              col_3rd[num_neighbors_max_+num_present_interfaces] = (mu_proj*ny + bc_coeff_proj*(xyz_pr[1] - xyz_C[1]));//dxyz_m_[1];
#ifdef P4_TO_P8
              col_4th[num_neighbors_max_+num_present_interfaces] = (mu_proj*nz + bc_coeff_proj*(xyz_pr[2] - xyz_C[2]));//dxyz_m_[2];
#endif
              weight [num_neighbors_max_+num_present_interfaces] = 1.e-5;

              bc_coeffs[phi_idx] = bc_coeff_proj;
              bc_values[num_present_interfaces] = bc_value_proj;

              ++num_present_interfaces;
            }
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
          inv_mat3_(A, A_inv);
#endif

          // compute Taylor expansion coefficients
          double coeff_const_term[num_neighbors_max_ + P4EST_DIM];
          double coeff_x_term[num_neighbors_max_ + P4EST_DIM];
          double coeff_y_term[num_neighbors_max_ + P4EST_DIM];
#ifdef P4_TO_P8
          double coeff_z_term[num_neighbors_max_ + P4EST_DIM];
#endif

          for (char nei = 0; nei < num_constraints; ++nei)
          {
#ifdef P4_TO_P8
            coeff_const_term[nei] = weight[nei]*
                ( A_inv[0*A_size+0]*col_1st[nei] +
                  A_inv[0*A_size+1]*col_2nd[nei] +
                  A_inv[0*A_size+2]*col_3rd[nei] +
                  A_inv[0*A_size+3]*col_4th[nei] );

            coeff_x_term[nei] = weight[nei]*
                ( A_inv[1*A_size+0]*col_1st[nei] +
                  A_inv[1*A_size+1]*col_2nd[nei] +
                  A_inv[1*A_size+2]*col_3rd[nei] +
                  A_inv[1*A_size+3]*col_4th[nei] );//dx_min_;

            coeff_y_term[nei] = weight[nei]*
                ( A_inv[2*A_size+0]*col_1st[nei] +
                  A_inv[2*A_size+1]*col_2nd[nei] +
                  A_inv[2*A_size+2]*col_3rd[nei] +
                  A_inv[2*A_size+3]*col_4th[nei] );//dy_min_;

            coeff_z_term[nei] = weight[nei]*
                ( A_inv[3*A_size+0]*col_1st[nei] +
                  A_inv[3*A_size+1]*col_2nd[nei] +
                  A_inv[3*A_size+2]*col_3rd[nei] +
                  A_inv[3*A_size+3]*col_4th[nei] );//dz_min_;
#else
            coeff_const_term[nei] = weight[nei]*
                ( A_inv[0*A_size+0]*col_1st[nei] +
                  A_inv[0*A_size+1]*col_2nd[nei] +
                  A_inv[0*A_size+2]*col_3rd[nei] );//dx_min_;

            coeff_x_term[nei] = weight[nei]*
                ( A_inv[1*A_size+0]*col_1st[nei] +
                  A_inv[1*A_size+1]*col_2nd[nei] +
                  A_inv[1*A_size+2]*col_3rd[nei] );//dy_min_;

            coeff_y_term[nei] = weight[nei]*
                ( A_inv[2*A_size+0]*col_1st[nei] +
                  A_inv[2*A_size+1]*col_2nd[nei] +
                  A_inv[2*A_size+2]*col_3rd[nei] );
#endif
          }

#ifdef P4_TO_P8
          double rhs_const_term = coeff_const_term[num_neighbors_max_+0] * bc_values[0]
                                + coeff_const_term[num_neighbors_max_+1] * bc_values[1]
                                + coeff_const_term[num_neighbors_max_+2] * bc_values[2];

          double rhs_x_term = coeff_x_term[num_neighbors_max_+0] * bc_values[0]
                            + coeff_x_term[num_neighbors_max_+1] * bc_values[1]
                            + coeff_x_term[num_neighbors_max_+2] * bc_values[2];

          double rhs_y_term = coeff_y_term[num_neighbors_max_+0] * bc_values[0]
                            + coeff_y_term[num_neighbors_max_+1] * bc_values[1]
                            + coeff_y_term[num_neighbors_max_+2] * bc_values[2];

          double rhs_z_term = coeff_z_term[num_neighbors_max_+0] * bc_values[0]
                            + coeff_z_term[num_neighbors_max_+1] * bc_values[1]
                            + coeff_z_term[num_neighbors_max_+2] * bc_values[2];
#else
          double rhs_const_term = coeff_const_term[num_neighbors_max_+0] * bc_values[0]
                                + coeff_const_term[num_neighbors_max_+1] * bc_values[1];

          double rhs_x_term = coeff_x_term[num_neighbors_max_+0] * bc_values[0]
                            + coeff_x_term[num_neighbors_max_+1] * bc_values[1];

          double rhs_y_term = coeff_y_term[num_neighbors_max_+0] * bc_values[0]
                            + coeff_y_term[num_neighbors_max_+1] * bc_values[1];
#endif

          // compute integrals
          double const_term = 0;
          double rhs_term = 0;
          double x_term = 0;
          double y_term = 0;
          double z_term = 0;

          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
            {
//              double bc_coeff = bc_coeffs[phi_idx];
#ifdef P4_TO_P8
              double xyz_point[P4EST_DIM] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i] };
#else
              double xyz_point[P4EST_DIM] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
#endif
              double bc_coeff = bc_interface_coeff_->at(phi_idx)->value(xyz_point);

              const_term += cube_ifc_w[phi_idx][i]*bc_coeff;

              x_term += cube_ifc_w[phi_idx][i]*bc_coeff*(xyz_point[0] - xyz_C[0]);
              y_term += cube_ifc_w[phi_idx][i]*bc_coeff*(xyz_point[1] - xyz_C[1]);
#ifdef P4_TO_P8
              z_term += cube_ifc_w[phi_idx][i]*bc_coeff*(xyz_point[2] - xyz_C[2]);
#endif
            }
          }

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

//          if (setup_rhs)
//          for (char nei = 0; nei < num_neighbors_max_; ++nei)
//          {
//            rhs_p[n] -= coeff_const_term[nei]*const_term*exact_ptr[neighbors[nei]]
//                + coeff_x_term[nei]*x_term*exact_ptr[neighbors[nei]]
//                + coeff_y_term[nei]*y_term*exact_ptr[neighbors[nei]];
//          }


          if (setup_rhs)
#ifdef P4_TO_P8
            rhs_p[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term + rhs_z_term*z_term;
#else
            rhs_p[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term;
#endif

          if (setup_matrix && fabs(const_term) > 0) matrix_has_nullspace_ = false;


//          for (int phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
//          {
////              double norm[P4EST_DIM];
////              compute_normal_(phi_p[phi_idx], qnnn, norm);
////              cf_const_t phi_x_local(norm[0]);
////              cf_const_t phi_y_local(norm[1]);
////#ifdef P4_TO_P8
////              cf_const_t phi_z_local(norm[2]);
////#endif
//            phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
//            phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
//#ifdef P4_TO_P8
//            phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
//            const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
//            x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
//            y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
//            z_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
//            rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local, *bc_interface_value_->at(phi_idx));
//#else
//            const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
//            x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
//            y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
//            rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, *bc_interface_value_->at(phi_idx));
//#endif

//            for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
//            {
//#ifdef P4_TO_P8
//              double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i] };
//#else
//              double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
//#endif
//              const_term   += cube_ifc_w[phi_idx][i] * const_coeff_integrand_.value(xyz);
//              rhs_term     += cube_ifc_w[phi_idx][i] * rhs_term_integrand_   .value(xyz);
//              x_term       += cube_ifc_w[phi_idx][i] * x_coeff_integrand_    .value(xyz);
//              y_term       += cube_ifc_w[phi_idx][i] * y_coeff_integrand_    .value(xyz);
//#ifdef P4_TO_P8
//              z_term       += cube_ifc_w[phi_idx][i] * z_coeff_integrand_    .value(xyz);
//#endif
//            }
//          }

//          if (setup_rhs) rhs_p[n] -= rhs_term;

//          if (setup_matrix && fabs(const_term) > 0) matrix_has_nullspace_ = false;

//          w[nn_000] += const_term;

//          for (char nei = 0; nei < num_neighbors_max_; ++nei)
//          {
//            w[nei] +=
//                coeff_x_term[nei]*x_term +
//                coeff_y_term[nei]*y_term
//    #ifdef P4_TO_P8
//                + coeff_z_term[nei]*z_term
//    #endif
//                ;
//          }

////          if (setup_rhs)
////          {
////            rhs_p[n] -= const_term*exact_ptr[neighbors[nn_000]];
////            for (char nei = 0; nei < num_neighbors_max_; ++nei)
////            {
////              rhs_p[n] -=
////                  coeff_x_term[nei]*x_term*exact_ptr[neighbors[nei]] +
////                  coeff_y_term[nei]*y_term*exact_ptr[neighbors[nei]];
////            }
////          }

//          if (setup_rhs)
//            rhs_p[n] -=
//                rhs_x_term*x_term +
//                rhs_y_term*y_term
//    #ifdef P4_TO_P8
//                + rhs_z_term*z_term
//    #endif
//                ;

        }

        //*/

        /*
        // an alternative super-convergent scheme
        // so far only in 2D and without kinks
        if (use_sc_scheme_)
//          if (use_sc_scheme_ && num_ifc < 2)
        {
          sc_scheme_successful = true;
          for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
            if (cube_ifc_w[phi_idx].size() > 0)
            {
#ifdef P4_TO_P8
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
              phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
#endif
              // compute centroid of an interface
              double x0 = 0;
              double y0 = 0;
#ifdef P4_TO_P8
              double z0 = 0;
#endif
              double S  = 0;

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                x0 += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0 += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
                S  += cube_ifc_w[phi_idx][i];
              }

//              if (S/dx_min_ > interface_rel_thresh)
              {
                x0 /= S;
                y0 /= S;
#ifdef P4_TO_P8
                z0 /= S;
#endif
              }

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

              {
                double xyz_pr[P4EST_DIM];

                xyz_pr[0] = xyz0[0] - dist0*nx;
                xyz_pr[1] = xyz0[1] - dist0*ny;
#ifdef P4_TO_P8
                xyz_pr[2] = xyz0[2] - dist0*nz;
#endif

                double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
                double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz0);
                double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

                if (setup_rhs)
                {
                  phi_x_local.set_input(exact_ptr, linear);
                  rhs_p[n] -= bc_coeff_proj*phi_x_local.value(xyz0)*S;

//                  for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
//                  {
//                    double xyz[P4EST_DIM] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
//                    rhs_p[n] -= bc_interface_coeff_->at(phi_idx)->value(xyz)*phi_x_local.value(xyz)*cube_ifc_w[phi_idx][i];
//                  }

                }
                continue;
              }

              // compute tangent vectors
              double t1x =  ny;
              double t1y = -nx;

              // loop through all available neighbors and project then onto tangent plane
              double  eta_C = t1x*(xyz_C[0] - xyz0[0]) + t1y*(xyz_C[1] - xyz0[1]);

              char nn_m = -1; double eta_p =  100, dist_p = 10;
              char nn_p = -1; double eta_m = -100, dist_m = 10;

              for (char nei = 0; nei < num_neighbors_max_; ++nei)
                if (neighbors_exist[nei] && nei != nn_000)
                {
                  double xyz_nei[P4EST_DIM];
                  node_xyz_fr_n(neighbors[nei], p4est_, nodes_, xyz_nei);

                  double eta_tmp = t1x*(xyz_nei[0] - xyz0[0]) + t1y*(xyz_nei[1] - xyz0[1]);
//                  double norm_tmp = sqrt(SQR(phi_x_ptr[phi_idx][neighbors[nei]]) + SQR(phi_y_ptr[phi_idx][neighbors[nei]]) );
//                  double dist_tmp = fabs(phi_p[phi_idx][neighbors[nei]])/norm_tmp;

                  double dist_tmp = sqrt( SQR(xyz_nei[0] - xyz0[0]) + SQR(xyz_nei[1] - xyz0[1]) );

                  if (eta_tmp < 0 && dist_tmp < dist_m) { nn_m = nei; eta_m = eta_tmp; dist_m = dist_tmp; }
                  if (eta_tmp > 0 && dist_tmp < dist_p) { nn_p = nei; eta_p = eta_tmp; dist_p = dist_tmp; }

//                  if (eta_tmp < 0 && eta_tmp > eta_m) { nn_m = nei; eta_m = eta_tmp; dist_m = dist_tmp; }
//                  if (eta_tmp > 0 && eta_tmp < eta_p) { nn_p = nei; eta_p = eta_tmp; dist_p = dist_tmp; }

//                  if (eta_tmp < 0 && SQR(eta_tmp) + SQR(dist_tmp) < SQR(eta_m) + SQR(dist_m) ) { nn_m = nei; eta_m = eta_tmp; dist_m = dist_tmp; }
//                  if (eta_tmp > 0 && SQR(eta_tmp) + SQR(dist_tmp) < SQR(eta_p) + SQR(dist_p) ) { nn_p = nei; eta_p = eta_tmp; dist_p = dist_tmp; }
                }

              char nn0 = nn_000; double eta_nn0 = eta_C;
              char nn1;          double eta_nn1 = 100;

              if (eta_C <  0) { nn1 = nn_p; eta_nn1 = eta_p; }
              if (eta_C >= 0) { nn1 = nn_m; eta_nn1 = eta_m; }

//              nn0 = nn_p; eta_nn0 = eta_p;
//              nn1 = nn_m; eta_nn1 = eta_m;

              if (nn1 == -1 && eta_C <  0) { nn1 = nn_m; eta_nn1 = eta_m; }
              if (nn1 == -1 && eta_C >= 0) { nn1 = nn_p; eta_nn1 = eta_p; }

              if (nn1 == -1 || nn0 == -1)
                throw;


              // compute weights between points
//              double weight_on_nn0 = fabs(eta_nn1)/fabs(eta_nn1-eta_nn0);
//              double weight_on_nn1 = fabs(eta_nn0)/fabs(eta_nn1-eta_nn0);

              double weight_on_nn0 =  eta_nn1/(eta_nn1 - eta_nn0);
              double weight_on_nn1 = -eta_nn0/(eta_nn1 - eta_nn0);

              // compute starting point
              double xyz_nn0[P4EST_DIM]; node_xyz_fr_n(neighbors[nn0], p4est_, nodes_, xyz_nn0);
              double xyz_nn1[P4EST_DIM]; node_xyz_fr_n(neighbors[nn1], p4est_, nodes_, xyz_nn1);

              double xyz_start[P4EST_DIM] = { xyz_nn0[0]*weight_on_nn0 + xyz_nn1[0]*weight_on_nn1,
                                              xyz_nn0[1]*weight_on_nn0 + xyz_nn1[1]*weight_on_nn1 };

              double norm_start = sqrt( SQR(phi_x_local.value(xyz_start)) + SQR(phi_y_local.value(xyz_start)) );
              double dist_start = phi_interp_local.value(xyz_start)/norm_start;

              // add coefficients
              double xyz_pr[P4EST_DIM];

              xyz_pr[0] = xyz0[0] - dist0*nx;
              xyz_pr[1] = xyz0[1] - dist0*ny;

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              double robin_term = bc_coeff_proj*(1.-dist0*bc_coeff_proj/mu_proj)/(1.-dist_start*bc_coeff_proj/mu_proj)*S;
              w[nn0] += weight_on_nn0*robin_term;
              w[nn1] += weight_on_nn1*robin_term;

              if (setup_rhs)
              rhs_p[n] -= bc_coeff_proj*bc_value_proj*(dist0-dist_start)/mu_proj/(1.-dist_start*bc_coeff_proj/mu_proj)*S;

//              w[nn_000] += bc_coeff_proj*S;
              if (setup_matrix && fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
            }
        }
        //*/

        /*
        if (use_sc_scheme_)
//          if (use_sc_scheme_ && num_ifc < 2)
        {
          double weights_x_derivative[num_neighbors_max_];
          double weights_y_derivative[num_neighbors_max_];
#ifdef P4_TO_P8
          double weights_z_derivative[num_neighbors_max_];
#endif

          bool map_x_derivative[num_neighbors_max_];
          bool map_y_derivative[num_neighbors_max_];
#ifdef P4_TO_P8
          bool map_z_derivative[num_neighbors_max_];
#endif

          bool x_derivative_found = find_x_derivative(neighbors_exist, weights_x_derivative, map_x_derivative, neighbors, volumes_p);
          bool y_derivative_found = find_y_derivative(neighbors_exist, weights_y_derivative, map_y_derivative, neighbors, volumes_p);
#ifdef P4_TO_P8
          bool z_derivative_found = find_z_derivative(neighbors_exist, weights_z_derivative, map_z_derivative, neighbors, volumes_p);
#endif

#ifdef P4_TO_P8
          if (x_derivative_found && y_derivative_found && z_derivative_found)
#else
          if (x_derivative_found && y_derivative_found)
#endif
          {


            sc_scheme_successful = true;

            double const_coeff = 0;
            double x_coeff = 0;
            double y_coeff = 0;
            double rhs_term = 0;
#ifdef P4_TO_P8
            double z_coeff = 0;
#endif
//            if (cube_ifc_w[phi_idx].size() > 0)

            // compute centroid of an interface
            double x0 = 0;
            double y0 = 0;
#ifdef P4_TO_P8
            double z0 = 0;
#endif
            double S  = 0;

            for (char phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
                x0 += cube_ifc_w[phi_idx][i]*(cube_ifc_x[phi_idx][i]);
                y0 += cube_ifc_w[phi_idx][i]*(cube_ifc_y[phi_idx][i]);
#ifdef P4_TO_P8
                z0 += cube_ifc_w[phi_idx][i]*(cube_ifc_z[phi_idx][i]);
#endif
                S  += cube_ifc_w[phi_idx][i];
              }

//#ifdef P4_TO_P8
//            if (S/dx_min_/dx_min_ > interface_rel_thresh)
//#else
//            if (S/dx_min_         > interface_rel_thresh)
//#endif
            {
              x0 /= S;
              y0 /= S;
#ifdef P4_TO_P8
              z0 /= S;
#endif
            }

#ifdef P4_TO_P8
            double xyz0[P4EST_DIM] = { x0, y0, z0 };
#else
            double xyz0[P4EST_DIM] = { x0, y0 };
#endif

#ifdef P4_TO_P8
            double xyz_robin[P4EST_DIM] = { x_C, y_C, z_C };
#else
            double xyz_robin[P4EST_DIM] = { x_C, y_C };
#endif
            char nn_robin = nn_000;
            double volume_robin = volumes_p[n];
            double dist_robin = 10;

            if (phi_eff_000 > 0 && volume_robin < 0.0)
              for (char nei = 0; nei < num_neighbors_max_; ++nei)
                if (neighbors_exist[nei])
                {
                  double xyz_tmp[P4EST_DIM];
                  node_xyz_fr_n(neighbors[nei], p4est_, nodes_, xyz_tmp);
                  double dist_tmp = sqrt( SQR(xyz_tmp[0] - xyz0[0]) + SQR(xyz_tmp[1] - xyz0[1]) );

                  if (phi_eff_p[neighbors[nei]] > 0 && dist_tmp < dist_robin)
                  {
                    node_xyz_fr_n(neighbors[nei], p4est_, nodes_, xyz_robin);
                    nn_robin = nei;
                    dist_robin = dist_tmp;
                    volume_robin = volumes_p[neighbors[nei]];
                  }
                }

            for (int phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
            {
//              double norm[P4EST_DIM];
//              compute_normal_(phi_p[phi_idx], qnnn, norm);
//              cf_const_t phi_x_local(norm[0]);
//              cf_const_t phi_y_local(norm[1]);
//#ifdef P4_TO_P8
//              cf_const_t phi_z_local(norm[2]);
//#endif
              phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
              phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
#ifdef P4_TO_P8
              phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
              const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, phi_z_local);
              x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, phi_z_local);
              y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, phi_z_local);
              z_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, phi_z_local);
              rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, phi_z_local, *bc_interface_value_->at(phi_idx));
#else
              const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local);
              x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local);
              y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local);
              rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_robin, interp_local, phi_x_local, phi_y_local, *bc_interface_value_->at(phi_idx));
#endif

              //              double tangent_coeff = 0;

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
#ifdef P4_TO_P8
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i] };
#else
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
#endif
                const_coeff   += cube_ifc_w[phi_idx][i] * const_coeff_integrand_.value(xyz);
                rhs_term      += cube_ifc_w[phi_idx][i] * rhs_term_integrand_   .value(xyz);
                x_coeff       += cube_ifc_w[phi_idx][i] * x_coeff_integrand_    .value(xyz);
                y_coeff       += cube_ifc_w[phi_idx][i] * y_coeff_integrand_    .value(xyz);
#ifdef P4_TO_P8
                z_coeff       += cube_ifc_w[phi_idx][i] * z_coeff_integrand_    .value(xyz);
#endif
                //                tangent_coeff += cubes[cube_idx]->integrate_over_interface(tangent_coeff_integrand_, color_->at(phi_idx));
              }

              //              x_coeff += -norm[1]*tangent_coeff;
              //              y_coeff +=  norm[0]*tangent_coeff;
            }

            if (setup_rhs) rhs_p[n] -= rhs_term;

            if (setup_matrix && fabs(const_coeff) > 0) matrix_has_nullspace_ = false;

            w[nn_robin] += const_coeff;

            for (int i = 0; i < num_neighbors_max_; ++i)
            {
              if (map_x_derivative[i]) w[i] += weights_x_derivative[i]*x_coeff/dx_min_;
              if (map_y_derivative[i]) w[i] += weights_y_derivative[i]*y_coeff/dy_min_;
#ifdef P4_TO_P8
              if (map_z_derivative[i]) w[i] += weights_z_derivative[i]*z_coeff/dz_min_;
#endif
            }
          }
        }
        //*/

        if (!use_sc_scheme_ || !sc_scheme_successful)
        {
          if (setup_matrix && use_sc_scheme_)
//            if (setup_matrix && use_sc_scheme_ && num_ifc < 2)
          {
            mask_p[n] = 20;
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

              bc_interface_coeff_avg[2] = 0;
              bc_interface_value_avg[2] = 0;
            }
#endif

            // Evaluate bc_interface_coefs and bc_interface_value at interfaces
            for (short i = 0; i < num_interfaces_present; ++i)
            {
              short phi_idx = present_interfaces[i];

              bc_interface_coeff_avg[i] = 0.;
              bc_interface_value_avg[i] = 0.;

              for (int j = 0; j < cube_ifc_w[phi_idx].size(); ++j)
              {
#ifdef P4_TO_P8
                double xyz[] = { cube_ifc_x[phi_idx][j], cube_ifc_y[phi_idx][j], cube_ifc_z[phi_idx][j] };
#else
                double xyz[] = { cube_ifc_x[phi_idx][j], cube_ifc_y[phi_idx][j] };
#endif
                bc_interface_coeff_avg[i] += cube_ifc_w[phi_idx][j] * (*bc_interface_coeff_->at(phi_idx)).value(xyz);
                bc_interface_value_avg[i] += cube_ifc_w[phi_idx][j] * (*bc_interface_value_->at(phi_idx)).value(xyz);
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

              for (int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
              {
#ifdef P4_TO_P8
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i] };
#else
                double xyz[] = { cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i] };
#endif
                w[nn_000] += cube_ifc_w[phi_idx][i] * taylor_expansion_coeff_term_.value(xyz);

                if (setup_rhs)
                  rhs_p[n] -= cube_ifc_w[phi_idx][i] * taylor_expansion_const_term_.value(xyz);
              }

            }

            //              double const_coeff = 0;
            //              double x_coeff = 0;
            //              double y_coeff = 0;
            //              double rhs_term = 0;
            //#ifdef P4_TO_P8
            //              double z_coeff = 0;
            //#endif

            //              for (short interface_idx = 0; interface_idx < present_interfaces.size(); ++interface_idx)
            //              {
            //                int phi_idx = present_interfaces[interface_idx];

            //                double norm[P4EST_DIM];
            //                compute_normal_(phi_p[phi_idx], qnnn, norm);

            ////                cf_const_t phi_x_local(norm[0]);
            ////                cf_const_t phi_y_local(norm[1]);
            ////#ifdef P4_TO_P8
            ////                cf_const_t phi_z_local(norm[2]);
            ////                const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            ////                x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            ////                y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            ////                z_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            ////                rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local, *bc_interface_value_->at(phi_idx));
            ////#else
            ////                const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            ////                x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            ////                y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            ////                rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, *bc_interface_value_->at(phi_idx));
            ////#endif

            //                phi_x_local.set_input(phi_x_ptr[phi_idx], linear);
            //                phi_y_local.set_input(phi_y_ptr[phi_idx], linear);
            //#ifdef P4_TO_P8
            //                phi_z_local.set_input(phi_z_ptr[phi_idx], linear);
            //                const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            //                x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            //                y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            //                z_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local);
            //                rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, phi_z_local, *bc_interface_value_->at(phi_idx));
            //#else
            //                const_coeff_integrand_.set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            //                x_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            //                y_coeff_integrand_    .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local);
            //                rhs_term_integrand_   .set(*bc_interface_coeff_->at(phi_idx), xyz_C, interp_local, phi_x_local, phi_y_local, *bc_interface_value_->at(phi_idx));
            //#endif

            //                for (int cube_idx = 0; cube_idx < cubes.size(); ++cube_idx)
            //                {
            //                  const_coeff   += cubes[cube_idx]->integrate_over_interface(const_coeff_integrand_,   color_->at(phi_idx));
            //                  rhs_term      += cubes[cube_idx]->integrate_over_interface(rhs_term_integrand_,      color_->at(phi_idx));
            //                  x_coeff       += cubes[cube_idx]->integrate_over_interface(x_coeff_integrand_,       color_->at(phi_idx));
            //                  y_coeff       += cubes[cube_idx]->integrate_over_interface(y_coeff_integrand_,       color_->at(phi_idx));
            //#ifdef P4_TO_P8
            //                  z_coeff       += cubes[cube_idx]->integrate_over_interface(z_coeff_integrand_,       color_->at(phi_idx));
            //#endif
            //                }

            //              }

            //#ifdef P4_TO_P8
            //              if (setup_rhs) rhs_p[n] -= rhs_term + b_coeff[0]*x_coeff + b_coeff[1]*y_coeff + b_coeff[2]*z_coeff;
            //              w[nn_000] += const_coeff - a_coeff[0]*x_coeff - a_coeff[1]*y_coeff - a_coeff[2]*z_coeff;
            //#else
            //              if (setup_rhs) rhs_p[n] -= rhs_term + b_coeff[0]*x_coeff + b_coeff[1]*y_coeff;
            //              w[nn_000] += const_coeff - a_coeff[0]*x_coeff - a_coeff[1]*y_coeff;
            //#endif

            //              if (setup_matrix && fabs(const_coeff) > 0) matrix_has_nullspace_ = false;
          }

          // cells without kinks
          if (!is_there_kink || !kink_special_treatment_)
          {

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
                double measure_of_iface = measure_of_interface[i_phi];

                if (measure_of_iface > eps_ifc_)
                {
                  find_projection_(phi_p[i_phi], qnnn, dxyz_pr, dist);

                  for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                    xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

                  double mu_proj = mu_;

                  if (variable_mu_)
                    mu_proj = interp_local.value(xyz_pr);

                  double bc_coeff_proj = bc_interface_coeff_->at(i_phi)->value(xyz_pr);

                  if (use_taylor_correction_) { w[nn_000] += mu_proj*bc_coeff_proj*measure_of_iface/(mu_proj-bc_coeff_proj*dist); }
                  else                        { w[nn_000] += bc_coeff_proj*measure_of_iface; }

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

        } // end of symmetric scheme

        // add diagonal term
//        w[nn_000] += diag_add_p[n]*volume_cut_cell;

        // scale all coefficient by the diagonal one and insert them into the matrix
        if (setup_rhs)
        {
          rhs_p[n] /= w[nn_000];
//          rhs_p[n] /= volume_cut_cell;
        }

        if (setup_matrix)
        {
          if (mask_p[n] > 0.)
          {
            ierr = PetscPrintf(p4est_->mpicomm, "Node %d is masked out \n", n); CHKERRXX(ierr);
          }

          if (fabs(diag_add_p[n]) > 0) matrix_has_nullspace_ = false;
          if (keep_scalling_) scalling_[n] = w[nn_000]/full_cell_volume; // scale to measure of the full cell dx*dy(*dz) for consistence

          double w_000 = w[nn_000];

          if (w_000 != w_000) throw;

          for (int i = 0; i < num_neighbors_max_; ++i)
//            w[i] /= volume_cut_cell;
            w[i] /= w_000;

          w[nn_000] = 1.;

          ierr = MatSetValue(A_, node_000_g, node_000_g, 1,  ADD_VALUES); CHKERRXX(ierr);

          for (int i = 0; i < num_neighbors_max_; ++i)
//            if (neighbors_exist[i])
            if (neighbors_exist[i] && fabs(w[i]) > EPS && i != nn_000)
            {
              if (w[i] != w[i]) throw;
              PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
              ierr = MatSetValue(A_, node_000_g, node_nei_g, w[i],  ADD_VALUES); CHKERRXX(ierr);
            }


//          if (volumes_p[n] < 1.e-3)
//            mask_p[n] = MAX(1., mask_p[n]);
        }

//        if (setup_rhs)
//        {
//          double w_000 = w[nn_000];

//          if (w_000 != w_000) throw;

//          for (int i = 0; i < num_neighbors_max_; ++i)
//            w[i] /= w_000;

//          w[nn_000] = 1.;

//          for (int i = 0; i < num_neighbors_max_; ++i)
//            if (neighbors_exist[i] && fabs(w[i]) > EPS && i != nn_000)
//            {
//              rhs_p[n] -= w[i]*exact_ptr[neighbors[i]];
//            }
//        }

      } else {

        if (setup_matrix && volume_cut_cell != 0.)
          ierr = PetscPrintf(p4est_->mpicomm, "Ignoring tiny volume %e\n", volume_cut_cell);

        // if finite volume too small, ignore the node
        if (setup_matrix)
        {
          mask_p[n] = MAX(1., mask_p[n]);
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
        }

        if (setup_rhs)
          rhs_p[n] = 0;
      }

    } // end of if (discretization_scheme_ = discretization_scheme_t::FVM)

  }

  if (setup_matrix)
  {
//    ierr = VecGhostUpdateBegin(mask_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(mask_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

//  // Second sweep to fill out outside nodes that are needed for multilinear interpolation
//  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
//  {
//    if (mask_p[n] == 777)
////      if (mask_p[n] > 0)
//    {
//      PetscInt node_000_g = petsc_gloidx_[n];

//      get_all_neighbors(n, neighbors, neighbors_exist);
//#ifdef P4_TO_P8
//        double w[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0,
//                                         0,0,0, 0,0,0, 0,0,0,
//                                         0,0,0, 0,0,0, 0,0,0 };
//#else
//        double w[num_neighbors_max_] = { 0,0,0, 0,0,0, 0,0,0 };
//#endif

////      if (setup_matrix) w[nn_000] = 1.;

//      if (setup_matrix)
//      {
//        for (int i = 0; i < num_neighbors_max_; ++i)
//          if (neighbors_exist[i] && fabs(w[i]) > EPS)
//          {
//            PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
//            ierr = MatSetValue(A_, node_000_g, node_nei_g, w[i],  ADD_VALUES); CHKERRXX(ierr);
//          }
//      }

//      if (setup_rhs) rhs_p[n] = exact_ptr[n];

//    }
//  }

  if (setup_matrix)
  {
    // Assemble the matrix
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }


  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);

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

  ierr = VecRestoreArray(exact_, &exact_ptr); CHKERRXX(ierr);

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


void my_p4est_poisson_nodes_mls_sc_t::compute_volumes_()
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

  double *node_type_p;
  if (node_type_ != NULL) { ierr = VecDestroy(node_type_); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_->at(0), &node_type_); CHKERRXX(ierr);
  ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);

  // data for refined cells
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
  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);

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

//    if(is_node_Wall(p4est_, ni))
//    {
//#ifdef P4_TO_P8
//      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
//#else
//      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
//#endif
//      {
//        volumes_p[n] = 1;
//        if (phi_eff_000 < 0. || num_interfaces_ == 0)
//        {
//        }
//        continue;
//      }

//      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
//      if(neumann_wall_first_order_ &&
//   #ifdef P4_TO_P8
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == NEUMANN)
//   #else
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1]) == NEUMANN)
//   #endif
//      {

//        if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
//        {
//          volumes_p[n] = 1;
//        }
//        continue;
//      }

//    }

    {
      interp_local.initialize(n);
      phi_interp_local.copy_init(interp_local);


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
        bool neighbors_exist[num_neighbors_max_];
        p4est_locidx_t neighbors[num_neighbors_max_];
        get_all_neighbors(n, neighbors, neighbors_exist);
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
          volumes_p[n] = 1;
          node_type_p[n] = 0;
          continue;
        }

        // far away from the interface
        if (phi_eff_000 > 0.)
        {
          volumes_p[n] = 0;
          node_type_p[n] = 0;
          continue;
        }

        // if far away from the interface or close to it but with dirichlet
        // then finite difference method
        if (phi_eff_000 < 0.)
        {
          volumes_p[n] = 1;
          node_type_p[n] = 63.;
          continue;
        }

      } else if (discretization_scheme_ == discretization_scheme_t::FVM) {

        // Reconstruct geometry
#ifdef P4_TO_P8
        double cube_xyz_min[] = { fv_xmin, fv_ymin, fv_zmin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax, fv_zmax };
        int  cube_mnk[] = { fv_size_x, fv_size_y, fv_size_z };
        cube3_mls_t cube(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);
#else
        double cube_xyz_min[] = { fv_xmin, fv_ymin };
        double cube_xyz_max[] = { fv_xmax, fv_ymax };
        int  cube_mnk[] = { fv_size_x, fv_size_y };
        cube2_mls_t cube(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);
#endif

        // get points at which values of level-set functions are needed
        std::vector<double> x_grid; cube.get_x_coord(x_grid);
        std::vector<double> y_grid; cube.get_y_coord(y_grid);
#ifdef P4_TO_P8
        std::vector<double> z_grid; cube.get_z_coord(z_grid);
#endif
        int points_total = x_grid.size();

        std::vector<double> phi_cube(num_interfaces_*points_total, -1);

        // compute values of level-set functions at needed points
        for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
#ifdef P4_TO_P8
          phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
#else
          phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
#endif
          for (int i = 0; i < points_total; ++i)
          {
//#ifdef P4_TO_P8
//            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
//#else
//            phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
//#endif
#ifdef P4_TO_P8
            phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i], z_grid[i]);
#else
            phi_cube[phi_idx*points_total + i] = (*phi_cf_->at(phi_idx))(x_grid[i], y_grid[i]);
#endif
            if (phi_cube[phi_idx*points_total + i] <  phi_perturbation_*dx_min_ &&
                phi_cube[phi_idx*points_total + i] > -phi_perturbation_*dx_min_)
              phi_cube[phi_idx*points_total + i] = phi_perturbation_*dx_min_;
          }
        }

        // reconstruct geometry
        cube.reconstruct(phi_cube, *action_, *color_);

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

        // compute cut-cell volume
        double volume_cut_cell = 0.;

        for (int i = 0; i < cube_dom_w.size(); ++i)
        {
          volume_cut_cell += cube_dom_w[i];
        }

        volumes_p[n] = volume_cut_cell/full_cell_volume;

        // check for a hanging volume
        bool is_one_positive = false;
        bool is_one_negative = false;

        double type = 0;

        for (int dir = 0; dir < 2*P4EST_DIM; ++dir)
        {
          cube_dom_w.clear();
          cube_dom_x.clear();
          cube_dom_y.clear();
#ifdef P4_TO_P8
          cube_dom_z.clear();
          cube.quadrature_in_dir(dir, cube_dom_w, cube_dom_x, cube_dom_y, cube_dom_z);
#else
          cube.quadrature_in_dir(dir, cube_dom_w, cube_dom_x, cube_dom_y);
#endif
          if (cube_dom_w.size() > 0)
            type += pow(2,dir);
        }

        node_type_p[n] = type;
      }
    }
  }

  // restore pointers
  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(node_type_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(node_type_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_sc_t::reconstruct_domain(std::vector<cube3_mls_t> &cubes)
#else
void my_p4est_poisson_nodes_mls_sc_t::reconstruct_domain(std::vector<cube2_mls_t> &cubes)
#endif
{
  double domain_rel_thresh = 1.e-13;
  double interface_rel_thresh = 1.e-13;

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
  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  double *node_type_p;
  ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);

  // data for refined cells
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

  cubes.reserve(nodes_->num_owned_indeps);

  bool neighbors_exist[num_neighbors_max_];
  p4est_locidx_t neighbors[num_neighbors_max_];

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);

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

//    if(is_node_Wall(p4est_, ni))
//    {
//#ifdef P4_TO_P8
//      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
//#else
//      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
//#endif
//      {
//        continue;
//      }

//      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
//      if(neumann_wall_first_order_ &&
//   #ifdef P4_TO_P8
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == NEUMANN)
//   #else
//         (*bc_wall_type_)(xyz_C[0], xyz_C[1]) == NEUMANN)
//   #endif
//      {
//        continue;
//      }

//    }

    {
      interp_local.initialize(n);
      phi_interp_local.copy_init(interp_local);

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_ngbd_crossed_dirichlet = false;
      bool is_ngbd_crossed_neumann   = false;

      if (fabs(phi_eff_000) < 4.*diag_min_)
      {
        get_all_neighbors(n, neighbors, neighbors_exist);

        // sample level-set function at cube nodes and check if crossed
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

      if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet)
        throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment");
      else if (is_ngbd_crossed_neumann)
        discretization_scheme_ = discretization_scheme_t::FVM;
      else
        discretization_scheme_ = discretization_scheme_t::FDM;



      if (discretization_scheme_ == discretization_scheme_t::FVM)
      {
        for (char idx = 0; idx < num_neighbors_max_; ++idx)
          if (neighbors_exist[idx])
            neighbors_exist[idx] = neighbors_exist[idx] && (volumes_p[neighbors[idx]] > domain_rel_thresh);

        // check for hanging neighbors
        bool hanging_neighbor[num_neighbors_max_];

        int network[num_neighbors_max_];

        for (char idx = 0; idx < num_neighbors_max_; ++idx)
          network[idx] = neighbors_exist[idx] ? (int) node_type_p[neighbors[idx]] : 0;

        find_hanging_cells(network, hanging_neighbor);

        bool expand[2*P4EST_DIM];
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
              char idx = 3*j + i;
  #endif
              if (neighbors_exist[idx])
                if (hanging_neighbor[idx] && volumes_p[neighbors[idx]] < 0.001)
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

        bool attempt_to_expand = false;

        for (char dir = 0; dir < P4EST_FACES; ++dir)
          attempt_to_expand = attempt_to_expand || expand[dir];

#ifdef P4_TO_P8
        cubes.push_back(cube3_mls_t());
#else
        cubes.push_back(cube2_mls_t());
#endif
        attempt_to_expand = false;
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

          if (attempt_to_expand)
          {
            ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment...\n");
            if (expand[dir::f_m00]) { fv_size_x += cube_refinement_; fv_xmin -= 0.5*dx_min_; }
            if (expand[dir::f_p00]) { fv_size_x += cube_refinement_; fv_xmax += 0.5*dx_min_; }
            if (expand[dir::f_0m0]) { fv_size_y += cube_refinement_; fv_ymin -= 0.5*dy_min_; }
            if (expand[dir::f_0p0]) { fv_size_y += cube_refinement_; fv_ymax += 0.5*dy_min_; }
  #ifdef P4_TO_P8
            if (expand[dir::f_00m]) { fv_size_z += cube_refinement_; fv_zmin -= 0.5*dz_min_; }
            if (expand[dir::f_00p]) { fv_size_z += cube_refinement_; fv_zmax += 0.5*dz_min_; }
  #endif
          }

          if (!use_refined_cube_)
          {
            fv_size_x = 1;
            fv_size_y = 1;
  #ifdef P4_TO_P8
            fv_size_z = 1;
  #endif
          }

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

          cubes.back().initialize(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);

          // get points at which values of level-set functions are needed
          std::vector<double> x_grid; cubes.back().get_x_coord(x_grid);
          std::vector<double> y_grid; cubes.back().get_y_coord(y_grid);
  #ifdef P4_TO_P8
          std::vector<double> z_grid; cubes.back().get_z_coord(z_grid);
  #endif
          int points_total = x_grid.size();

          std::vector<double> phi_cube(num_interfaces_*points_total,-1);

          // compute values of level-set functions at needed points
          for (int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
  #ifdef P4_TO_P8
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], quadratic);
  #else
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], quadratic);
  #endif
            for (int i = 0; i < points_total; ++i)
            {
  #ifdef P4_TO_P8
              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
  #else
              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
  #endif
            }
          }

          // reconstruct geometry
          cubes.back().reconstruct(phi_cube, *action_, *color_);

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
                cubes.back().quadrature_in_dir(dir, W, X, Y, Z);
  #else
                cubes.back().quadrature_in_dir(dir, W, X, Y);
  #endif
                if (W.size() != 0) attempt_to_expand = false;
              }
            }

            if (attempt_to_expand)
            {
              ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment... Success!\n");

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
                    if (i != 1 && j != 1 && k != 1) neighbors_exist[9*k + 3*j + i] = false;
  #else
                    if (i != 1 && j != 1)           neighbors_exist[3*j + i]       = false;
  #endif
                  }
              break;
            } else {
              ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment... Failure!\n");
            }

          } else {
            break;
          }
        }

      }
    }
  }

  // restore pointers

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);
  for (int i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

}


void my_p4est_poisson_nodes_mls_sc_t::find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr)
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

void my_p4est_poisson_nodes_mls_sc_t::compute_normal_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[])
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



void my_p4est_poisson_nodes_mls_sc_t::inv_mat3_(double *in, double *out)
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

void my_p4est_poisson_nodes_mls_sc_t::inv_mat2_(double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];
  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;
}

bool my_p4est_poisson_nodes_mls_sc_t::inv_mat4_(const double m[16], double invOut[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
              m[4]  * m[11] * m[14] +
              m[8]  * m[6]  * m[15] -
              m[8]  * m[7]  * m[14] -
              m[12] * m[6]  * m[11] +
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] -
               m[8]  * m[6] * m[13] -
               m[12] * m[5] * m[10] +
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
              m[1]  * m[11] * m[14] +
              m[9]  * m[2] * m[15] -
              m[9]  * m[3] * m[14] -
              m[13] * m[2] * m[11] +
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
              m[0]  * m[11] * m[13] +
              m[8]  * m[1] * m[15] -
              m[8]  * m[3] * m[13] -
              m[12] * m[1] * m[11] +
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
              m[0]  * m[7] * m[14] +
              m[4]  * m[2] * m[15] -
              m[4]  * m[3] * m[14] -
              m[12] * m[2] * m[7] +
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
               m[0]  * m[6] * m[13] +
               m[4]  * m[1] * m[14] -
               m[4]  * m[2] * m[13] -
               m[12] * m[1] * m[6] +
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[9] * m[2] * m[7] +
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[9] +
               m[4] * m[1] * m[11] -
               m[4] * m[3] * m[9] -
               m[8] * m[1] * m[7] +
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}


void my_p4est_poisson_nodes_mls_sc_t::get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists)
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
double my_p4est_poisson_nodes_mls_sc_t::compute_weights_through_face(double A, double B, bool *neighbors_exists_2d, double *weights_2d, double theta, bool *map_2d)
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



  double a = fabs(A);
  double b = fabs(B);

  if (a > .5 || b > .5) std::cout << "Warning!\n";

  double mask_specific = -1;

  int num_good_neighbors = 0;

  for (int i = 0; i < 9; ++i)
    if (neighbors_exists_2d[i]) num_good_neighbors++;

  bool same_line = (num_good_neighbors == 3 &&
                    ( (neighbors_exists_2d[nn_m0m] && neighbors_exists_2d[nn_p0m]) ||
                      (neighbors_exists_2d[nn_0mm] && neighbors_exists_2d[nn_0pm]) ||
                      (neighbors_exists_2d[nn_mmm] && neighbors_exists_2d[nn_ppm]) ||
                      (neighbors_exists_2d[nn_mpm] && neighbors_exists_2d[nn_pmm]) ) );

  if (a < theta && b < theta)
  {
    map_2d[nn_00m] = true;  weights_2d[nn_00m] = 1;
    mask_specific = -2;
  }




  else if (A <= 0 && B <= 0 && B <= A &&
           neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b-a;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B <= 0 && B >= A &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a-b;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B <= 0 && B <= -A &&
           neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b-a;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B <= 0 && B >= -A &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a-b;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B >= 0 && B <= -A &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a-b;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A <= 0 && B >= 0 && B >= -A &&
           neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b-a;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B >= 0 && B <= A &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-a);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a-b;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = b;
    mask_specific = -1;
//    semi_fallback = true;
  }
  else if (A >= 0 && B >= 0 && B >= A &&
           neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-b);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b-a;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = a;
    mask_specific = -1;
//    semi_fallback = true;
  }

  else if ( neighbors_exists_2d[nn_m0m] &&
            neighbors_exists_2d[nn_0mm] &&
            neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.+A)*(1.+B);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] =(  -A)*(1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] =(1.+A)*(  -B);
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] =(  -A)*(  -B);
  }
  else if ( neighbors_exists_2d[nn_p0m] &&
            neighbors_exists_2d[nn_0mm] &&
            neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-A)*(1.+B);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] =(   A)*(1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] =(1.-A)*(  -B);
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] =(   A)*(  -B);
  }
  else if ( neighbors_exists_2d[nn_m0m] &&
            neighbors_exists_2d[nn_0pm] &&
            neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.+A)*(1.-B);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] =(  -A)*(1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] =(1.+A)*(   B);
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] =(  -A)*(   B);
  }
  else if ( neighbors_exists_2d[nn_p0m] &&
            neighbors_exists_2d[nn_0pm] &&
            neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] =(1.-A)*(1.-B);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] =(   A)*(1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] =(1.-A)*(   B);
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] =(   A)*(   B);
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

  else if (A <= 0 && B <= 0 &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b;
    mask_specific = -11;
  }
  else if (A >= 0 && B <= 0 &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = b;
    mask_specific = -12;
  }
  else if (A <= 0 && B >= 0 &&
           neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = a;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b;
    mask_specific = -13;
  }
  else if (A >= 0 && B >= 0 &&
           neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-a-b;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = a;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = b;
    mask_specific = -14;
  }





//    else if (num_good_neighbors >= 3 && !same_line)
//    {
//      weights_2d[nn_mmm] = 0;
//      weights_2d[nn_0mm] = 0;
//      weights_2d[nn_pmm] = 0;
//      weights_2d[nn_m0m] = 0;
//      weights_2d[nn_00m] = 0;
//      weights_2d[nn_p0m] = 0;
//      weights_2d[nn_mpm] = 0;
//      weights_2d[nn_0pm] = 0;
//      weights_2d[nn_ppm] = 0;

//      // linear system
//      char num_constraints = 9;

//      std::vector<double> col_1st(num_constraints, 0);
//      std::vector<double> col_2nd(num_constraints, 0);
//      std::vector<double> col_3rd(num_constraints, 0);

//      for (char j = 0; j < 3; ++j)
//        for (char i = 0; i < 3; ++i)
//        {
//          char idx = 3*j+i;
//          col_1st[idx] = 1.;
//          col_2nd[idx] = ((double) (i-1));
//          col_3rd[idx] = ((double) (j-1));
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

//          //                    if (neighbors_exists_2d[idx])
//          //                      weight[idx] = exp(-gamma*sqrt(SQR((x_C+dx-x0[phi_idx])/dx_min_) +
//          //                                                    SQR((y_C+dy-y0[phi_idx])/dy_min_)));
//          if (neighbors_exists_2d[idx] || idx == nn_00m)
////            weight[idx] = 1.;
//            weight[idx] = exp(-gamma*(SQR(x-A) +
//                                      SQR(y-B)));
//  //        if (idx == nn_00m)
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
//        matA[0*A_size + 0] += col_1st[nei]*col_1st[nei]*weight[nei];
//        matA[0*A_size + 1] += col_1st[nei]*col_2nd[nei]*weight[nei];
//        matA[0*A_size + 2] += col_1st[nei]*col_3rd[nei]*weight[nei];
//        matA[1*A_size + 1] += col_2nd[nei]*col_2nd[nei]*weight[nei];
//        matA[1*A_size + 2] += col_2nd[nei]*col_3rd[nei]*weight[nei];
//        matA[2*A_size + 2] += col_3rd[nei]*col_3rd[nei]*weight[nei];
//      }

//      matA[1*A_size + 0] = matA[0*A_size + 1];
//      matA[2*A_size + 0] = matA[0*A_size + 2];
//      matA[2*A_size + 1] = matA[1*A_size + 2];

//      inv_mat3_(matA, matA_inv);

//      // compute Taylor expansion coefficients
//      std::vector<double> coeff_const_term(num_constraints, 0);
//      std::vector<double> coeff_x_term    (num_constraints, 0);
//      std::vector<double> coeff_y_term    (num_constraints, 0);

//      for (char nei = 0; nei < num_constraints; ++nei)
//      {
//        coeff_const_term[nei] = weight[nei]*
//            ( matA_inv[0*A_size+0]*col_1st[nei]
//            + matA_inv[0*A_size+1]*col_2nd[nei]
//            + matA_inv[0*A_size+2]*col_3rd[nei] );

//        coeff_x_term[nei] = weight[nei]*
//            ( matA_inv[1*A_size+0]*col_1st[nei]
//            + matA_inv[1*A_size+1]*col_2nd[nei]
//            + matA_inv[1*A_size+2]*col_3rd[nei] );

//        coeff_y_term[nei] = weight[nei]*
//            ( matA_inv[2*A_size+0]*col_1st[nei]
//            + matA_inv[2*A_size+1]*col_2nd[nei]
//            + matA_inv[2*A_size+2]*col_3rd[nei] );
//      }

//      // compute integrals
//      double const_term = 1.;
//      double x_term     = 1.*(A);
//      double y_term     = 1.*(B);


//      if (!neighbors_exists_2d[nn_00m])
//        std::cout << "weird!\n";

//      // matrix coefficients
//      for (char nei = 0; nei < 9; ++nei)
//      {
//        if (neighbors_exists_2d[nei])
//  //        if (neighbors_exists_2d[nei] || nei == nn_00m)
//        {
//          map_2d[nei] = true;
//          weights_2d[nei] += coeff_const_term[nei]*const_term
//              + coeff_x_term[nei]*x_term
//              + coeff_y_term[nei]*y_term;
//        }
//      }
//      mask_specific = -1;
//      semi_fallback = true;
//    }


  else if (neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B+A;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = -A;
    mask_specific = -15;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+A);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A+B;
    map_2d[nn_mmm] = true; weights_2d[nn_mmm] = -B;
    mask_specific = -16;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0mm] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+B);
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B-A;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = A;
    mask_specific = -17;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_pmm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-A);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A+B;
    map_2d[nn_pmm] = true; weights_2d[nn_pmm] = -B;
    mask_specific = -18;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.+A);
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A-B;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = B;
    mask_specific = -19;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_mpm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B+A;
    map_2d[nn_mpm] = true; weights_2d[nn_mpm] = -A;
    mask_specific = -20;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-A);
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A-B;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = B;
    mask_specific = -21;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_0pm] &&
           neighbors_exists_2d[nn_ppm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = (1.-B);
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B-A;
    map_2d[nn_ppm] = true; weights_2d[nn_ppm] = A;
    mask_specific = -22;
    semi_fallback = true;
  }

  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.+A+B;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B;
    mask_specific = -23;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0mm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-A+B;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A;
    map_2d[nn_0mm] = true; weights_2d[nn_0mm] = -B;
    mask_specific = -24;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_m0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.+A-B;
    map_2d[nn_m0m] = true; weights_2d[nn_m0m] = -A;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B;
    mask_specific = -25;
    semi_fallback = true;
  }
  else if (neighbors_exists_2d[nn_p0m] &&
           neighbors_exists_2d[nn_0pm] )
  {
    map_2d[nn_00m] = true; weights_2d[nn_00m] = 1.-A-B;
    map_2d[nn_p0m] = true; weights_2d[nn_p0m] = A;
    map_2d[nn_0pm] = true; weights_2d[nn_0pm] = B;
    mask_specific = -26;
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
//    std::cout << "!!!!!! Fallback flux between cells!\n";

    int num_good_neighbors = 0;
    mask_specific = 50;

    for (int i = 0; i < 9; ++i)
      if (neighbors_exists_2d[i]) num_good_neighbors++;

    if (num_good_neighbors >= 3)
    {
      std::cout << "Possible! " << num_good_neighbors << "\n";
      std::cout << A << ", " << B << "\n";
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

//  return 1;

  if (full_fallback)
  {
//    std::cout << "Full fallback fluxes\n";
    return 5;
//    return -1;
  }
  else if (semi_fallback)
  {
//    return -1;
//    std::cout << "Semi fallback fluxes\n";
    return 2;
  }
  else
    return -1;
}
#endif

bool my_p4est_poisson_nodes_mls_sc_t::find_x_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_m00] && neighbors_exist[nn_p00]) { weights[nn_m00] =-.5; weights[nn_p00] =.5; map[nn_m00] = true; map[nn_p00] = true; }
//  else if (neighbors_exist[nn_mm0] && neighbors_exist[nn_pm0]) { weights[nn_mm0] =-.5; weights[nn_pm0] =.5; map[nn_mm0] = true; map[nn_pm0] = true; }
//  else if (neighbors_exist[nn_mp0] && neighbors_exist[nn_pp0]) { weights[nn_mp0] =-.5; weights[nn_pp0] =.5; map[nn_mp0] = true; map[nn_pp0] = true; }
  else
//  if      (neighbors_exist[nn_000] && neighbors_exist[nn_p00]) { weights[nn_000] =-1.; weights[nn_p00] = 1.; map[nn_000] = true; map[nn_p00] = true; }
//  else if (neighbors_exist[nn_000] && neighbors_exist[nn_m00]) { weights[nn_000] = 1.; weights[nn_m00] =-1.; map[nn_000] = true; map[nn_m00] = true; }

  if (neighbors_exist[nn_m00] && neighbors_exist[nn_p00]) {
    if (volumes_p[neighbors[nn_m00]] > volumes_p[neighbors[nn_p00]]) { weights[nn_000] = 1.; weights[nn_m00] =-1.; map[nn_000] = true; map[nn_m00] = true; }
    else                                                             { weights[nn_000] =-1.; weights[nn_p00] = 1.; map[nn_000] = true; map[nn_p00] = true; }
  }
  else if (neighbors_exist[nn_p00]) { weights[nn_000] =-1.; weights[nn_p00] = 1.; map[nn_000] = true; map[nn_p00] = true; }
  else if (neighbors_exist[nn_m00]) { weights[nn_000] = 1.; weights[nn_m00] =-1.; map[nn_000] = true; map[nn_m00] = true; }


//  // check m0
//  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_mm0]) { weights[nn_0m0] = 1.; weights[nn_mm0] =-1.; map[nn_0m0] = true; map[nn_mm0] = true; }
//  else if (neighbors_exist[nn_0m0] && neighbors_exist[nn_pm0]) { weights[nn_0m0] =-1.; weights[nn_pm0] = 1.; map[nn_0m0] = true; map[nn_pm0] = true; }

//  // check p0
//  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_mp0]) { weights[nn_0p0] = 1.; weights[nn_mp0] =-1.; map[nn_0p0] = true; map[nn_mp0] = true; }
//  else if (neighbors_exist[nn_0p0] && neighbors_exist[nn_pp0]) { weights[nn_0p0] =-1.; weights[nn_pp0] = 1.; map[nn_0p0] = true; map[nn_pp0] = true; }
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

bool my_p4est_poisson_nodes_mls_sc_t::find_y_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_0m0] && neighbors_exist[nn_0p0]) { weights[nn_0m0] =-.5; weights[nn_0p0] = .5; map[nn_0m0] = true; map[nn_0p0] = true; }
//  else if (neighbors_exist[nn_mm0] && neighbors_exist[nn_mp0]) { weights[nn_mm0] =-.5; weights[nn_mp0] = .5; map[nn_mm0] = true; map[nn_mp0] = true; }
//  else if (neighbors_exist[nn_pm0] && neighbors_exist[nn_pp0]) { weights[nn_pm0] =-.5; weights[nn_pp0] = .5; map[nn_pm0] = true; map[nn_pp0] = true; }
  else
//  if      (neighbors_exist[nn_000] && neighbors_exist[nn_0p0]) { weights[nn_000] =-1.; weights[nn_0p0] = 1.; map[nn_000] = true; map[nn_0p0] = true; }
//  else if (neighbors_exist[nn_000] && neighbors_exist[nn_0m0]) { weights[nn_000] = 1.; weights[nn_0m0] =-1.; map[nn_000] = true; map[nn_0m0] = true; }

    if      (neighbors_exist[nn_0m0] && neighbors_exist[nn_0p0]) {
      if (volumes_p[neighbors[nn_0m0]] > volumes_p[neighbors[nn_0p0]]) { weights[nn_000] = 1.; weights[nn_0m0] =-1.; map[nn_000] = true; map[nn_0m0] = true; }
      else                                                             { weights[nn_000] =-1.; weights[nn_0p0] = 1.; map[nn_000] = true; map[nn_0p0] = true; }
    }
    else if (neighbors_exist[nn_0p0]) { weights[nn_000] =-1.; weights[nn_0p0] = 1.; map[nn_000] = true; map[nn_0p0] = true; }
    else if (neighbors_exist[nn_0m0]) { weights[nn_000] = 1.; weights[nn_0m0] =-1.; map[nn_000] = true; map[nn_0m0] = true; }

//  // check 0m
//  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_mm0]) { weights[nn_m00] = 1.; weights[nn_mm0] =-1.; map[nn_m00] = true; map[nn_mm0] = true; }
//  else if (neighbors_exist[nn_m00] && neighbors_exist[nn_mp0]) { weights[nn_m00] =-1.; weights[nn_mp0] = 1.; map[nn_m00] = true; map[nn_mp0] = true; }

//  // check 0p
//  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_pm0]) { weights[nn_p00] = 1.; weights[nn_pm0] =-1.; map[nn_p00] = true; map[nn_pm0] = true; }
//  else if (neighbors_exist[nn_p00] && neighbors_exist[nn_pp0]) { weights[nn_p00] =-1.; weights[nn_pp0] = 1.; map[nn_p00] = true; map[nn_pp0] = true; }

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
bool my_p4est_poisson_nodes_mls_sc_t::find_z_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p)
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


bool my_p4est_poisson_nodes_mls_sc_t::find_xy_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p)
{
  bool derivative_found = true;

  for (short i = 0; i < num_neighbors_max_; ++i)
    map[i] = false;

  // check 00
  if      (neighbors_exist[nn_mm0] &&
           neighbors_exist[nn_m00] &&
           neighbors_exist[nn_0m0]) { weights[nn_mm0] = .25; map[nn_mm0] = true;
                                      weights[nn_0m0] =-.25; map[nn_0m0] = true;
                                      weights[nn_m00] =-.25; map[nn_m00] = true;
                                      weights[nn_000] = .25; map[nn_000] = true; }
  else if (neighbors_exist[nn_0m0] &&
           neighbors_exist[nn_pm0] &&
           neighbors_exist[nn_p00]) { weights[nn_0m0] = .25; map[nn_0m0] = true;
                                      weights[nn_pm0] =-.25; map[nn_pm0] = true;
                                      weights[nn_000] =-.25; map[nn_000] = true;
                                      weights[nn_p00] = .25; map[nn_p00] = true; }
  else if (neighbors_exist[nn_m00] &&
           neighbors_exist[nn_mp0] &&
           neighbors_exist[nn_0p0]) { weights[nn_m00] = .25; map[nn_m00] = true;
                                      weights[nn_000] =-.25; map[nn_000] = true;
                                      weights[nn_mp0] =-.25; map[nn_mp0] = true;
                                      weights[nn_0p0] = .25; map[nn_0p0] = true; }
  else if (neighbors_exist[nn_0p0] &&
           neighbors_exist[nn_pp0] &&
           neighbors_exist[nn_p00]) { weights[nn_000] = .25; map[nn_000] = true;
                                      weights[nn_p00] =-.25; map[nn_p00] = true;
                                      weights[nn_0p0] =-.25; map[nn_0p0] = true;
                                      weights[nn_pp0] = .25; map[nn_pp0] = true; }
#ifdef P4_TO_P8
#endif
  else
    derivative_found = false;

  return derivative_found;
}

void my_p4est_poisson_nodes_mls_sc_t::find_hanging_cells(int *network, bool *hanging_cells)
{
  bool connection_matrix[num_neighbors_max_][num_neighbors_max_];
  bool visited[num_neighbors_max_];

  // loop through all nodes and construct connection matrix
#ifdef P4_TO_P8
  for (int k = 0; k < 3; ++k)
#endif
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i)
      {
#ifdef P4_TO_P8
        int idx = 9*k + 3*j +i;
#else
        int idx = 3*j +i;
#endif

        for (int nei_idx = 0; nei_idx < num_neighbors_max_; ++nei_idx)
          connection_matrix[idx][nei_idx] = false;

        hanging_cells[idx] = true;
        visited[idx] = false;

        int type = network[idx];

        bool connected;

        int I, J, K;

#ifdef P4_TO_P8
        connected = type % 2 == 0 ? false : true; type /= 2; I = i - 1; J = j    ; K = k    ; if (I < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i + 1; J = j    ; K = k    ; if (I > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j - 1; K = k    ; if (J < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j + 1; K = k    ; if (J > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j    ; K = k - 1; if (K < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j    ; K = k + 1; if (K > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
#else
        connected = type % 2 == 0 ? false : true; type /= 2; I = i - 1; J = j    ; if (I < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i + 1; J = j    ; if (I > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j - 1; if (J < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j + 1; if (J > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
#endif
      }


  // loop through all nodes
  for (int idx = 0; idx < num_neighbors_max_; ++idx)
  {
    // if a node connected to outside
    if (hanging_cells[idx] || visited[idx] || idx == nn_000) continue;

    // then initiate a queue
    std::vector<int> queue;
    queue.push_back(idx);

    // loop through nodes in the queue
    for (int i = 0; i < queue.size(); ++i)
    {
      int current_idx = queue[i];

      // for every node check which other nodes it's connected to
      for (int nei_idx = 0; nei_idx < num_neighbors_max_; ++nei_idx)
      {
        if (current_idx == nei_idx || nei_idx == nn_000 || visited[nei_idx]) continue;
        // put connected nodes into the queue
        if (connection_matrix[current_idx][nei_idx]) { hanging_cells[nei_idx] = false; queue.push_back(nei_idx); }
      }

      visited[current_idx] = true;
    }
  }

//#ifdef P4_TO_P8
//  for (int k = 0; k < 3; ++k)
//#endif
//    for (int j = 0; j < 3; ++j)
//      for (int i = 0; i < 3; ++i)
//      {
//#ifdef P4_TO_P8
//        int idx = 9*k + 3*j +i;
//#else
//        int idx = 3*j +i;
//#endif

//        for (int nei_idx = 0; nei_idx < num_neighbors_max_; ++nei_idx)
//          connection_matrix[idx][nei_idx] = false;

//        hanging_cells[idx] = true;
//        visited[idx] = false;

//        int type = network[idx];

//        bool connected;

//        int I, J, K;

//#ifdef P4_TO_P8
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i - 1; J = j    ; K = k    ; if (I < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i + 1; J = j    ; K = k    ; if (I > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j - 1; K = k    ; if (J < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j + 1; K = k    ; if (J > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j    ; K = k - 1; if (K < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j    ; K = k + 1; if (K > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][9*K + 3*J + I] = connected; }
//#else
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i - 1; J = j    ; if (I < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i + 1; J = j    ; if (I > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j - 1; if (J < 0 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
//        connected = type % 2 == 0 ? false : true; type /= 2; I = i    ; J = j + 1; if (J > 2 && connected) { hanging_cells[idx] = false; } else { connection_matrix[idx][3*J + I] = connected; }
//#endif
//      }
}

//void my_p4est_poisson_nodes_mls_sc_t::assemble_matrix(Vec solution)
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

////void my_p4est_poisson_nodes_mls_sc_t::assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p)
//void my_p4est_poisson_nodes_mls_sc_t::assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in, Vec rhs_p_in)
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
