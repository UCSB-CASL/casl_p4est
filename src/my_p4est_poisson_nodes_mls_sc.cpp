#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls_sc.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#include <src/my_p8est_interpolation_nodes_local.h>
//#include <src/simplex3_mls_vtk.h>
//#include <src/simplex3_mls_quadratic_vtk.h>
#include <src/my_p8est_macros.h>
#else
#include "my_p4est_poisson_nodes_mls_sc.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_macros.h>
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


my_p4est_poisson_nodes_mls_sc_t::my_p4est_poisson_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    p4est_(node_neighbors->p4est),
    nodes_(node_neighbors->nodes),
    ghost_(node_neighbors->ghost),
    myb_  (node_neighbors->myb)
{
  // default parameters
  lip_ = 2.0;

  // linear system
  A_ = NULL;

  // geometry
  num_interfaces_ = 0;

  phi_    = NULL;
  phi_cf_ = NULL;

  phi_xx_ = NULL;
  phi_yy_ = NULL;
#ifdef P4_TO_P8
  phi_zz_ = NULL;
#endif

  phi_eff_ = NULL;

  action_ = NULL;
  color_  = NULL;

  phi_x_ = NULL;
  phi_y_ = NULL;
#ifdef P4_TO_P8
  phi_z_ = NULL;
#endif

  is_phi_d_owned_   = false;
  is_phi_dd_owned_  = false;
  is_phi_eff_owned_ = false;

  // equation
  rhs_    = NULL;
  rhs_m_  = NULL;
  rhs_p_  = NULL;
  rhs_cf_ = NULL;

  diag_add_m_      = NULL;
  diag_add_p_      = NULL;
  diag_add_m_scalar_ = 0.;
  diag_add_p_scalar_ = 0.;

  mu_m_ = 1.;
  mu_p_ = 1.;

  variable_mu_     = false;
  is_mue_m_dd_owned_ = false;
  is_mue_p_dd_owned_ = false;
  mue_m_             = NULL;
  mue_p_             = NULL;
  mue_m_xx_          = NULL;
  mue_p_xx_          = NULL;
  mue_m_yy_          = NULL;
  mue_p_yy_          = NULL;
#ifdef P4_TO_P8
  mue_m_zz_          = NULL;
  mue_p_zz_          = NULL;
#endif

  // solver options
  integration_order_          = 2;
  cube_refinement_            = 0;
  use_sc_scheme_              = 1;
  use_pointwise_dirichlet_    = 0;
  use_taylor_correction_      = 1;
  kink_special_treatment_     = 1;
  update_ghost_after_solving_ = 0;
  try_remove_hanging_cells_   = 0;
  neumann_wall_first_order_   = 0;
  phi_perturbation_           = 1.e-12;
  interp_method_              = quadratic_non_oscillatory_continuous_v2;

  domain_rel_thresh_    = 1.e-11;
  interface_rel_thresh_ = 1.e-11;

  // some flags
  new_pc_               = true;
  is_matrix_computed_   = false;
  matrix_has_nullspace_ = false;

  // bc
  bc_wall_type_  = NULL;
  bc_wall_value_ = NULL;
  bc_interface_coeff_ = NULL;
  bc_interface_type_  = NULL;
  bc_interface_value_ = NULL;

  // auxiliary variables
  mask_      = NULL;
  areas_     = NULL;
  areas_m_   = NULL;
  areas_p_   = NULL;
  node_type_ = NULL;
  exact_ = NULL;
//  volumes_   = NULL;

  keep_scalling_    = false;
  volumes_owned_    = false;
  volumes_computed_ = false;

  // compute global numbering of nodes
  global_node_offset_.resize(p4est_->mpisize+1, 0);
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

#ifdef P4_TO_P8
  face_area_scalling_ = diag_min_*diag_min_;
#else
  face_area_scalling_ = diag_min_;
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
}

my_p4est_poisson_nodes_mls_sc_t::~my_p4est_poisson_nodes_mls_sc_t()
{
  if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }

  if (A_   != NULL) { ierr = MatDestroy(A_);   CHKERRXX(ierr); }
  if (ksp_ != NULL) { ierr = KSPDestroy(ksp_); CHKERRXX(ierr); }

  if (is_phi_dd_owned_)
  {
    if (phi_xx_ != NULL) { for (unsigned int i = 0; i < phi_xx_->size(); i++) { ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr); } delete phi_xx_; }
    if (phi_yy_ != NULL) { for (unsigned int i = 0; i < phi_yy_->size(); i++) { ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr); } delete phi_yy_; }
#ifdef P4_TO_P8
    if (phi_zz_ != NULL) { for (unsigned int i = 0; i < phi_zz_->size(); i++) { ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr); } delete phi_zz_; }
#endif
  }

  if (is_phi_d_owned_)
  {
    if (phi_x_ != NULL) { for (unsigned int i = 0; i < phi_x_->size(); i++) { ierr = VecDestroy(phi_x_->at(i)); CHKERRXX(ierr); } delete phi_x_; }
    if (phi_y_ != NULL) { for (unsigned int i = 0; i < phi_y_->size(); i++) { ierr = VecDestroy(phi_y_->at(i)); CHKERRXX(ierr); } delete phi_y_; }
#ifdef P4_TO_P8
    if (phi_z_ != NULL) { for (unsigned int i = 0; i < phi_z_->size(); i++) { ierr = VecDestroy(phi_z_->at(i)); CHKERRXX(ierr); } delete phi_z_; }
#endif
  }

  if (is_mue_m_dd_owned_)
  {
    if (mue_m_xx_ != NULL) { ierr = VecDestroy(mue_m_xx_); CHKERRXX(ierr); }
    if (mue_m_yy_ != NULL) { ierr = VecDestroy(mue_m_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
    if (mue_m_zz_ != NULL) { ierr = VecDestroy(mue_m_zz_); CHKERRXX(ierr); }
#endif
  }

  if (is_mue_p_dd_owned_)
  {
    if (mue_p_xx_ != NULL) { ierr = VecDestroy(mue_p_xx_); CHKERRXX(ierr); }
    if (mue_p_yy_ != NULL) { ierr = VecDestroy(mue_p_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
    if (mue_p_zz_ != NULL) { ierr = VecDestroy(mue_p_zz_); CHKERRXX(ierr); }
#endif
  }

//  if (volumes_ != NULL && volumes_owned_) {ierr = VecDestroy(volumes_); CHKERRXX(ierr);}
  if (areas_   != NULL && volumes_owned_) {ierr = VecDestroy(areas_);   CHKERRXX(ierr);}
  if (areas_m_ != NULL && volumes_owned_) {ierr = VecDestroy(areas_m_); CHKERRXX(ierr);}
  if (areas_p_ != NULL && volumes_owned_) {ierr = VecDestroy(areas_p_); CHKERRXX(ierr);}

  if (node_type_ != NULL) {ierr = VecDestroy(node_type_); CHKERRXX(ierr);}
  if (is_phi_eff_owned_) {ierr = VecDestroy(phi_eff_);  CHKERRXX(ierr);}

  // immersed interface stuff
  if (ii_.is_phi_dd_owned)
  {
    if (ii_.phi_xx != NULL) { for (unsigned int i = 0; i < ii_.phi_xx->size(); i++) { ierr = VecDestroy(ii_.phi_xx->at(i)); CHKERRXX(ierr); } delete ii_.phi_xx; }
    if (ii_.phi_yy != NULL) { for (unsigned int i = 0; i < ii_.phi_yy->size(); i++) { ierr = VecDestroy(ii_.phi_yy->at(i)); CHKERRXX(ierr); } delete ii_.phi_yy; }
#ifdef P4_TO_P8
    if (ii_.phi_zz != NULL) { for (unsigned int i = 0; i < ii_.phi_zz->size(); i++) { ierr = VecDestroy(ii_.phi_zz->at(i)); CHKERRXX(ierr); } delete ii_.phi_zz; }
#endif
  }

  if (ii_.is_phi_d_owned)
  {
    if (ii_.phi_x != NULL) { for (unsigned int i = 0; i < ii_.phi_x->size(); i++) { ierr = VecDestroy(ii_.phi_x->at(i)); CHKERRXX(ierr); } delete ii_.phi_x; }
    if (ii_.phi_y != NULL) { for (unsigned int i = 0; i < ii_.phi_y->size(); i++) { ierr = VecDestroy(ii_.phi_y->at(i)); CHKERRXX(ierr); } delete ii_.phi_y; }
#ifdef P4_TO_P8
    if (ii_.phi_z != NULL) { for (unsigned int i = 0; i < ii_.phi_z->size(); i++) { ierr = VecDestroy(ii_.phi_z->at(i)); CHKERRXX(ierr); } delete ii_.phi_z; }
#endif
  }
  if (ii_.is_phi_eff_owned) { ierr = VecDestroy(ii_.phi_eff);  CHKERRXX(ierr); }
}


void my_p4est_poisson_nodes_mls_sc_t::compute_phi_eff(Vec &phi_eff, std::vector<Vec> *&phi, std::vector<action_t> *&action, bool &is_phi_eff_owned)
{
  if (phi_eff != NULL && is_phi_eff_owned) { ierr = VecDestroy(phi_eff); CHKERRXX(ierr); }


  int num_interfaces = phi->size();

  if (num_interfaces == 1)
  {
    is_phi_eff_owned = false;
    phi_eff = phi->at(0);
  }
  else
  {
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_eff); CHKERRXX(ierr);
    is_phi_eff_owned = true;
    std::vector<double *>   phi_p(num_interfaces, NULL);
    double                 *phi_eff_p;

    for (unsigned int i = 0; i < num_interfaces; i++)
    {
      ierr = VecGetArray(phi->at(i), &phi_p[i]);  CHKERRXX(ierr);
    }

    ierr = VecGetArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n) // loop over nodes
    {
      phi_eff_p[n] = -10.;

      for (unsigned int i = 0; i < num_interfaces; i++)
      {
        switch (action->at(i))
        {
          case INTERSECTION:  phi_eff_p[n] = (phi_eff_p[n] > phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
          case ADDITION:      phi_eff_p[n] = (phi_eff_p[n] < phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
          case COLORATION:    /* do nothing */ break;
        }
      }
    }

    for (unsigned int i = 0; i < num_interfaces; i++)
    {
      ierr = VecRestoreArray(phi->at(i), &phi_p[i]);  CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);
  }
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_sc_t::compute_phi_dd(std::vector<Vec> *&phi, std::vector<Vec> *&phi_xx, std::vector<Vec> *&phi_yy, std::vector<Vec> *&phi_zz, bool &is_phi_dd_owned)
#else
void my_p4est_poisson_nodes_mls_sc_t::compute_phi_dd(std::vector<Vec> *&phi, std::vector<Vec> *&phi_xx, std::vector<Vec> *&phi_yy, bool &is_phi_dd_owned)
#endif
{
  int num_interfaces = phi->size();

  // Allocate memory for second derivaties
  if (phi_xx != NULL && is_phi_dd_owned)
  {
    for (unsigned int i = 0; i < phi_xx->size(); i++) {ierr = VecDestroy(phi_xx->at(i)); CHKERRXX(ierr);}
    delete phi_xx;
  }
  phi_xx = new std::vector<Vec> ();

  if (phi_yy != NULL && is_phi_dd_owned)
  {
    for (unsigned int i = 0; i < phi_yy->size(); i++) {ierr = VecDestroy(phi_yy->at(i)); CHKERRXX(ierr);}
    delete phi_yy;
  }
  phi_yy = new std::vector<Vec> ();

#ifdef P4_TO_P8
  if (phi_zz != NULL && is_phi_dd_owned)
  {
    for (unsigned int i = 0; i < phi_zz->size(); i++) {ierr = VecDestroy(phi_zz->at(i)); CHKERRXX(ierr);}
    delete phi_zz;
  }
  phi_zz = new std::vector<Vec> ();
#endif

  for (unsigned int i = 0; i < num_interfaces; i++)
  {
    phi_xx->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_xx->at(i)); CHKERRXX(ierr);
    phi_yy->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_yy->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    phi_zz->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_zz->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors_->second_derivatives_central(phi->at(i), phi_xx->at(i), phi_yy->at(i), phi_zz->at(i));
#else
    node_neighbors_->second_derivatives_central(phi->at(i), phi_xx->at(i), phi_yy->at(i));
#endif
  }
  is_phi_dd_owned = true;
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_sc_t::compute_phi_d(std::vector<Vec> *&phi, std::vector<Vec> *&phi_x, std::vector<Vec> *&phi_y, std::vector<Vec> *&phi_z, bool &is_phi_d_owned)
#else
void my_p4est_poisson_nodes_mls_sc_t::compute_phi_d(std::vector<Vec> *&phi, std::vector<Vec> *&phi_x, std::vector<Vec> *&phi_y, bool &is_phi_d_owned)
#endif
{
  int num_interfaces = phi->size();

  // Allocate memory for second derivaties
  if (phi_x != NULL && is_phi_d_owned) { for (unsigned int i = 0; i < phi_x->size(); i++) {ierr = VecDestroy(phi_x->at(i)); CHKERRXX(ierr);} delete phi_x; } phi_x = new std::vector<Vec> ();
  if (phi_y != NULL && is_phi_d_owned) { for (unsigned int i = 0; i < phi_y->size(); i++) {ierr = VecDestroy(phi_y->at(i)); CHKERRXX(ierr);} delete phi_y; } phi_y = new std::vector<Vec> ();
#ifdef P4_TO_P8
  if (phi_z != NULL && is_phi_d_owned) { for (unsigned int i = 0; i < phi_z->size(); i++) {ierr = VecDestroy(phi_z->at(i)); CHKERRXX(ierr);} delete phi_z; } phi_z = new std::vector<Vec> ();
#endif

  for (unsigned int i = 0; i < num_interfaces; i++)
  {
    phi_x->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_x->at(i)); CHKERRXX(ierr);
    phi_y->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_y->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    phi_z->push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_z->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    Vec phi_d[P4EST_DIM] = { phi_x->at(i), phi_y->at(i), phi_z->at(i) };
#else
    Vec phi_d[P4EST_DIM] = { phi_x->at(i), phi_y->at(i) };
#endif

    node_neighbors_->first_derivatives_central(phi->at(i), phi_d);
  }
  is_phi_d_owned = true;
}

void my_p4est_poisson_nodes_mls_sc_t::compute_mue_dd()
{
  if (is_mue_m_dd_owned_)
  {
    if (mue_m_xx_ != NULL) { ierr = VecDestroy(mue_m_xx_); CHKERRXX(ierr); }
    if (mue_m_yy_ != NULL) { ierr = VecDestroy(mue_m_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
    if (mue_m_zz_ != NULL) { ierr = VecDestroy(mue_m_zz_); CHKERRXX(ierr); }
#endif
  }

  if (is_mue_p_dd_owned_)
  {
    if (mue_p_xx_ != NULL) { ierr = VecDestroy(mue_p_xx_); CHKERRXX(ierr); }
    if (mue_p_yy_ != NULL) { ierr = VecDestroy(mue_p_yy_); CHKERRXX(ierr); }
#ifdef P4_TO_P8
    if (mue_p_zz_ != NULL) { ierr = VecDestroy(mue_p_zz_); CHKERRXX(ierr); }
#endif
  }

  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_xx_); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_m_zz_); CHKERRXX(ierr);

  node_neighbors_->second_derivatives_central(mue_m_, mue_m_xx_, mue_m_yy_, mue_m_zz_);
#else
  node_neighbors_->second_derivatives_central(mue_m_, mue_m_xx_, mue_m_yy_);
#endif
  is_mue_m_dd_owned_ = true;

  if (mue_m_ == mue_p_)
  {
    mue_p_xx_ = mue_m_xx_;
    mue_p_yy_ = mue_m_yy_;
#ifdef P4_TO_P8
    mue_p_zz_ = mue_m_zz_;
#endif
    is_mue_p_dd_owned_ = false;
  } else {
    ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_yy_); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est_, nodes_, &mue_p_zz_); CHKERRXX(ierr);

    node_neighbors_->second_derivatives_central(mue_p_, mue_p_xx_, mue_p_yy_, mue_p_zz_);
  #else
    node_neighbors_->second_derivatives_central(mue_p_, mue_p_xx_, mue_p_yy_);
  #endif
    is_mue_p_dd_owned_ = true;
  }
}

void my_p4est_poisson_nodes_mls_sc_t::preallocate_matrix()
{
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_matrix_preallocation, A_, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

  if (A_ != NULL) { ierr = MatDestroy(A_); CHKERRXX(ierr); }

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
  if(diag_add_m_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &diag_add_m_); CHKERRXX(ierr);
    ierr = VecSet(diag_add_m_, diag_add_m_scalar_); CHKERRXX(ierr);
  }

  if(diag_add_p_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &diag_add_p_); CHKERRXX(ierr);
    ierr = VecSet(diag_add_p_, diag_add_p_scalar_); CHKERRXX(ierr);
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

    bc_interface_type_ = new std::vector<BoundaryConditionType> (1, DIRICHLET);
  }

  bool local_immersed_phi = false;
  if(ii_.num_interfaces == 0)
  {
    local_immersed_phi = true;

    ii_.phi    = new std::vector<Vec> ();
    ii_.color  = new std::vector<int> ();
    ii_.action = new std::vector<action_t> ();

    ii_.phi->push_back(Vec());
    ii_.color->push_back(0);
    ii_.action->push_back(INTERSECTION);

    ierr = VecDuplicate(solution, &ii_.phi->at(0)); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(ii_.phi->at(0), &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ii_.phi->at(0), &tmp); CHKERRXX(ierr);
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    set_immersed_interface(1, ii_.action, ii_.color, ii_.phi);
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
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.93"); CHKERRXX(ierr);

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
  ierr = KSPSetTolerances(ksp_, 1.e-15, 1.e-15, PETSC_DEFAULT, 50); CHKERRXX(ierr);
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
    ierr = VecDestroy(diag_add_m_); CHKERRXX(ierr);
    ierr = VecDestroy(diag_add_p_); CHKERRXX(ierr);
    diag_add_m_ = NULL;
    diag_add_p_ = NULL;
  }
  if(local_rhs)
  {
    ierr = VecDestroy(rhs_); CHKERRXX(ierr);
    rhs_ = NULL;
  }
  if(local_phi)
  {
    num_interfaces_ = 0;

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
    delete bc_interface_type_;
  }

  if(local_immersed_phi)
  {
    ii_.num_interfaces = 0;

    ierr = VecDestroy(ii_.phi->at(0)); CHKERRXX(ierr);
    delete ii_.phi;
    ii_.phi = NULL;

    ierr = VecDestroy(ii_.phi_xx->at(0)); CHKERRXX(ierr);
    delete ii_.phi_xx;
    ii_.phi_xx = NULL;

    ierr = VecDestroy(ii_.phi_yy->at(0)); CHKERRXX(ierr);
    delete ii_.phi_yy;
    ii_.phi_yy = NULL;

#ifdef P4_TO_P8
    ierr = VecDestroy(ii_.phi_zz->at(0)); CHKERRXX(ierr);
    delete ii_.phi_zz;
    ii_.phi_zz = NULL;
#endif

    delete ii_.action;
    delete ii_.color;
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_solve, A_, rhs_, ksp_, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_sc_t::setup_linear_system(bool setup_matrix, bool setup_rhs)
{
  PetscInt num_owned_global = global_node_offset_[p4est_->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes_->num_owned_indeps);

  std::vector< std::vector<mat_entry_t>* > matrix_entries(nodes_->num_owned_indeps, NULL);
  std::vector<PetscInt> d_nnz(nodes_->num_owned_indeps, 1), o_nnz(nodes_->num_owned_indeps, 0);

  // auxiliary matrices for immersed interface
  std::vector< std::vector<mat_entry_t>* > B_matrix_entries(nodes_->num_owned_indeps, NULL);
  std::vector< std::vector<mat_entry_t>* > C_matrix_entries(nodes_->num_owned_indeps, NULL);
  std::vector<PetscInt> B_d_nnz(nodes_->num_owned_indeps, 1), B_o_nnz(nodes_->num_owned_indeps, 0);
  std::vector<PetscInt> C_d_nnz(nodes_->num_owned_indeps, 1), C_o_nnz(nodes_->num_owned_indeps, 0);

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
      B_matrix_entries[n] = new std::vector<mat_entry_t>;
      C_matrix_entries[n] = new std::vector<mat_entry_t>;
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

  // domain
  std::vector<double *> phi_p   (num_interfaces_, NULL);
  std::vector<double *> phi_xx_p(num_interfaces_, NULL);
  std::vector<double *> phi_yy_p(num_interfaces_, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p(num_interfaces_, NULL);
#endif

  for (unsigned short i = 0; i < num_interfaces_; i++)
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

  // immersed interface
  std::vector<double *> ii_phi_p   (ii_.num_interfaces, NULL);
  std::vector<double *> ii_phi_xx_p(ii_.num_interfaces, NULL);
  std::vector<double *> ii_phi_yy_p(ii_.num_interfaces, NULL);
#ifdef P4_TO_P8
  std::vector<double *> ii_phi_zz_p(ii_.num_interfaces, NULL);
#endif

  for (unsigned short i = 0; i < ii_.num_interfaces; i++)
  {
    ierr = VecGetArray(ii_.phi->at(i),    &ii_phi_p[i]);    CHKERRXX(ierr);
    ierr = VecGetArray(ii_.phi_xx->at(i), &ii_phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(ii_.phi_yy->at(i), &ii_phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(ii_.phi_zz->at(i), &ii_phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *ii_phi_eff_p;
  ierr = VecGetArray(ii_.phi_eff, &ii_phi_eff_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // diffusion coefficients
  //---------------------------------------------------------------------
  double *mue_m_ptr=NULL;
  double *mue_m_xx_ptr=NULL;
  double *mue_m_yy_ptr=NULL;
#ifdef P4_TO_P8
  double *mue_m_zz_ptr=NULL;
#endif

  if (variable_mu_)
  {
    ierr = VecGetArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr);
#endif
  }

  double *mue_p_ptr=NULL;
  double *mue_p_xx_ptr=NULL;
  double *mue_p_yy_ptr=NULL;
#ifdef P4_TO_P8
  double *mue_p_zz_ptr=NULL;
#endif

  if (variable_mu_)
  {
    ierr = VecGetArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr);
#endif
  }

  double mu_ = mu_m_;
  double *mue_p=NULL;
  double *mue_xx_p=NULL;
  double *mue_yy_p=NULL;
#ifdef P4_TO_P8
  double *mue_zz_p=NULL;
#endif

  double *diag_add_m_ptr;
  double *diag_add_p_ptr;

  ierr = VecGetArray(diag_add_m_, &diag_add_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(diag_add_p_, &diag_add_p_ptr); CHKERRXX(ierr);

  double *rhs_ptr;
  double *rhs_p_ptr;
  double *rhs_m_ptr;
  if (setup_rhs)
  {
    ierr = VecGetArray(rhs_,   &rhs_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
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

  std::vector<double> ii_phi_000(ii_.num_interfaces,-1);
  std::vector<double> ii_phi_p00(ii_.num_interfaces, 0);
  std::vector<double> ii_phi_m00(ii_.num_interfaces, 0);
  std::vector<double> ii_phi_0m0(ii_.num_interfaces, 0);
  std::vector<double> ii_phi_0p0(ii_.num_interfaces, 0);
#ifdef P4_TO_P8
  std::vector<double> ii_phi_00m(ii_.num_interfaces, 0);
  std::vector<double> ii_phi_00p(ii_.num_interfaces, 0);
#endif

  double *mask_p;
  if (setup_matrix)
  {
    if (!volumes_computed_) compute_volumes();
    if (mask_ != NULL) { ierr = VecDestroy(mask_); CHKERRXX(ierr); }
    ierr = VecDuplicate(phi_->at(0), &mask_); CHKERRXX(ierr);
  }
  ierr = VecGetArray(mask_, &mask_p); CHKERRXX(ierr);

  if (setup_matrix)
    for(size_t n=0; n<nodes_->indep_nodes.elem_count; ++n)
      mask_p[n] = -1;

//  double *volumes_p;
  double *areas_ptr;
  double *areas_m_ptr;
  double *areas_p_ptr;
  double *node_type_p;

  if (use_sc_scheme_)
  {
//    ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);
    ierr = VecGetArray(areas_,   &areas_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(areas_m_, &areas_m_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(areas_p_, &areas_p_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);
  }

  double mue_000 = mu_m_;
  double mue_p00 = mu_m_;
  double mue_m00 = mu_m_;
  double mue_0m0 = mu_m_;
  double mue_0p0 = mu_m_;
#ifdef P4_TO_P8
  double mue_00m = mu_m_;
  double mue_00p = mu_m_;
#endif

  // data for refined cells
  unsigned short fv_size_x = 0;
  unsigned short fv_size_y = 0;
#ifdef P4_TO_P8
  unsigned short fv_size_z = 0;
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double interp_xmin, interp_xmax;
  double interp_ymin, interp_ymax;
#ifdef P4_TO_P8
  double interp_zmin, interp_zmax;
#endif

  double xyz_C[P4EST_DIM];

  double full_cell_volume;
  double dxyz_pr[P4EST_DIM];
  double xyz_pr[P4EST_DIM];
  double dist;

  bool neighbors_exist[num_neighbors_cube_];
  bool neighbors_exist_m[num_neighbors_cube_];
  bool neighbors_exist_p[num_neighbors_cube_];
  p4est_locidx_t neighbors[num_neighbors_cube_];

  // interpolations
  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);
  my_p4est_interpolation_nodes_local_t mu_m_interp(node_neighbors_);
  my_p4est_interpolation_nodes_local_t mu_p_interp(node_neighbors_);

  if (variable_mu_)
  {
#ifdef P4_TO_P8
    interp_local.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, mue_m_zz_ptr, interp_method_);
    mu_m_interp.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, mue_m_zz_ptr, interp_method_);
    mu_p_interp.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, mue_p_zz_ptr, interp_method_);
#else
    interp_local.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, interp_method_);
    mu_m_interp.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, interp_method_);
    mu_p_interp.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, interp_method_);
#endif
  }

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

  for (unsigned int i = 0; i < num_interfaces_; ++i)
  {
    ierr = VecGetArray(phi_x_->at(i), &phi_x_ptr[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y_->at(i), &phi_y_ptr[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z_->at(i), &phi_z_ptr[i]); CHKERRXX(ierr);
#endif
  }

  p4est_locidx_t node_m00_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_m00_pm = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_p00_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_p00_pm = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_0m0_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_0m0_pm = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_0p0_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_0p0_pm = NOT_A_VALID_QUADRANT;

#ifdef P4_TO_P8
  p4est_locidx_t node_m00_mp = NOT_A_VALID_QUADRANT; p4est_locidx_t node_m00_pp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_p00_mp = NOT_A_VALID_QUADRANT; p4est_locidx_t node_p00_pp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_0m0_mp = NOT_A_VALID_QUADRANT; p4est_locidx_t node_0m0_pp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_0p0_mp = NOT_A_VALID_QUADRANT; p4est_locidx_t node_0p0_pp = NOT_A_VALID_QUADRANT;

  p4est_locidx_t node_00m_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_00m_mp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_00m_pm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_00m_pp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_00p_mm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_00p_mp = NOT_A_VALID_QUADRANT;
  p4est_locidx_t node_00p_pm = NOT_A_VALID_QUADRANT; p4est_locidx_t node_00p_pp = NOT_A_VALID_QUADRANT;
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

  double cell_expansion = (phi_cf_ == NULL ? .5 : 1.);

#ifdef DO_NOT_PREALLOCATE
  mat_entry_t ent;
#endif

  bool interface_present = false;

  Vec rhs_add;
  double *rhs_add_ptr;

  if (setup_rhs)
  {
    ierr = VecDuplicate(rhs_, &rhs_add); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_add, &rhs_add_ptr); CHKERRXX(ierr);
  }

  double diag_add = 0;

  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
  {
#ifdef DO_NOT_PREALLOCATE
    std::vector<mat_entry_t> * row = matrix_entries[n];
    std::vector<mat_entry_t> * row_B = B_matrix_entries[n];
    std::vector<mat_entry_t> * row_C = C_matrix_entries[n];
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
    double ii_phi_eff_000 = ii_phi_eff_p[n];

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
      if(bc_wall_type_->value(xyz_C) == DIRICHLET)
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
          rhs_ptr[n] = (*bc_wall_value_).value(xyz_C);
        }

        continue;
      }

      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
      if(neumann_wall_first_order_ && bc_wall_type_->value(xyz_C) == NEUMANN)
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_m00;
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_p00;
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0m0;
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
            ierr = MatSetValue(A_, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_0p0;
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00m;
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

          if (setup_rhs) rhs_ptr[n] = bc_strength*bc_wall_value_->value(xyz_C)*d_00p;
          continue;
        }
#endif
      }

    }
    //*/


#ifdef P4_TO_P8
    for (short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      qnnn.ngbd_with_quadratic_interpolation(phi_p[phi_idx], phi_000[phi_idx], phi_m00[phi_idx], phi_p00[phi_idx], phi_0m0[phi_idx], phi_0p0[phi_idx], phi_00m[phi_idx], phi_00p[phi_idx]);
#else
    for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      qnnn.ngbd_with_quadratic_interpolation(phi_p[phi_idx], phi_000[phi_idx], phi_m00[phi_idx], phi_p00[phi_idx], phi_0m0[phi_idx], phi_0p0[phi_idx]);
#endif

    interp_local.initialize(n);
    mu_m_interp.copy_init(interp_local);
    mu_p_interp.copy_init(interp_local);
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
    bool is_ngbd_crossed_immersed  = false;

    if (fabs(phi_eff_000) < lip_*diag_min_)
    {
      get_all_neighbors(n, neighbors, neighbors_exist);

      // sample level-set function at nodes of the extended cube and check if crossed
      for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
        bool is_one_positive = false;
        bool is_one_negative = false;

        for (short i = 0; i < num_neighbors_cube_; ++i)
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

    if (fabs(ii_phi_eff_000) < lip_*diag_min_)
    {
      get_all_neighbors(n, neighbors, neighbors_exist);

      // sample level-set function at nodes of the extended cube and check if crossed
      for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
      {
        bool is_one_positive = false;
        bool is_one_negative = false;

        for (short i = 0; i < num_neighbors_cube_; ++i)
          if (neighbors_exist[i])
          {
            is_one_positive = is_one_positive || ii_phi_p[phi_idx][neighbors[i]] > 0;
            is_one_negative = is_one_negative || ii_phi_p[phi_idx][neighbors[i]] < 0;
          }

        if (is_one_negative && is_one_positive) is_ngbd_crossed_immersed = true;
      }
    }

    if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet) { throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment"); }
    else if (is_ngbd_crossed_neumann)                         { discretization_scheme_ = FVM; }
    else if (is_ngbd_crossed_immersed)                        { discretization_scheme_ = JUMP; }
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
            rhs_ptr[n] = pointwise_bc_[n][0].value;
          else
            for (unsigned int i = 0; i < num_interfaces_; ++i)
              if (fabs(phi_p[i][n]) < EPS*diag_min_ && bc_interface_type_->at(i) == DIRICHLET)
                rhs_ptr[n] = bc_interface_value_->at(i)->value(xyz_C);
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
          rhs_ptr[n] = 0;
        }
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if (phi_eff_000 < 0.)
      {
        if (ii_phi_eff_000 < 0)
        {
          mu_ = mu_m_;
          mue_p    = mue_m_ptr;
          mue_xx_p = mue_m_xx_ptr;
          mue_yy_p = mue_m_yy_ptr;
#ifdef P4_TO_P8
          mue_zz_p = mue_m_zz_ptr;
#endif
          if (setup_rhs) rhs_ptr[n] = rhs_m_ptr[n];
          diag_add = diag_add_m_ptr[n];
        } else {
          mu_ = mu_p_;
          mue_p    = mue_p_ptr;
          mue_xx_p = mue_p_xx_ptr;
          mue_yy_p = mue_p_yy_ptr;
#ifdef P4_TO_P8
          mue_zz_p = mue_p_zz_ptr;
#endif
          if (setup_rhs) rhs_ptr[n] = rhs_p_ptr[n];
          diag_add = diag_add_p_ptr[n];
        }

        if (variable_mu_)
        {
#ifdef P4_TO_P8
          qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
          qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif
        }
        else
        {
          mue_000 = mue_m00 = mue_p00 = mue_0m0 = mue_0p0 = mu_;
#ifdef P4_TO_P8
          mue_00m = mue_00p = mu_;
#endif
        }

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
        for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
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
            for (unsigned short i = 0; i < pointwise_bc_[n].size(); ++i)
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
        double w_000  = diag_add - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
        double w_000  = diag_add - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
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

          if (diag_add > 0) matrix_has_nullspace_ = false;
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

          if(is_node_xmWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_p00;
          else if(is_interface_m00)      rhs_ptr[n] -= w_m00 * val_interface_m00;

          if(is_node_xpWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y, z_C+eps_z) / d_m00;
          else if(is_interface_p00)      rhs_ptr[n] -= w_p00 * val_interface_p00;

          if(is_node_ymWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0p0;
          else if(is_interface_0m0)      rhs_ptr[n] -= w_0m0 * val_interface_0m0;

          if(is_node_ypWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C, z_C+eps_z) / d_0m0;
          else if(is_interface_0p0)      rhs_ptr[n] -= w_0p0 * val_interface_0p0;

          if(is_node_zmWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00p;
          else if(is_interface_00m)      rhs_ptr[n] -= w_00m * val_interface_00m;

          if(is_node_zpWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C+eps_y, z_C) / d_00m;
          else if(is_interface_00p)      rhs_ptr[n] -= w_00p * val_interface_00p;
#else

          double eps_x = is_node_xmWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_xpWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);
          double eps_y = is_node_ymWall(p4est_, ni) ? 2.*EPS*diag_min_ : (is_node_ypWall(p4est_, ni) ? -2.*EPS*diag_min_ : 0);

          if(is_node_xmWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_p00;
          else if(is_interface_m00)      rhs_ptr[n] -= w_m00*val_interface_m00;

          if(is_node_xpWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C, y_C+eps_y) / d_m00;
          else if(is_interface_p00)      rhs_ptr[n] -= w_p00*val_interface_p00;

          if(is_node_ymWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0p0;
          else if(is_interface_0m0)      rhs_ptr[n] -= w_0m0*val_interface_0m0;

          if(is_node_ypWall(p4est_, ni)) rhs_ptr[n] += 2.*mue_000*(*bc_wall_value_)(x_C+eps_x, y_C) / d_0m0;
          else if(is_interface_0p0)      rhs_ptr[n] -= w_0p0*val_interface_0p0;
#endif

          rhs_ptr[n] /= w_000;
        }

        continue;
      }

    }
    else if (discretization_scheme_ == FVM)
    {
      if (ii_phi_eff_000 < 0)
      {
        mu_ = mu_m_;
#ifdef P4_TO_P8
        interp_local.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, mue_m_zz_ptr, interp_method_);
#else
        interp_local.set_input(mue_m_ptr, mue_m_xx_ptr, mue_m_yy_ptr, interp_method_);
#endif
        if (setup_rhs) rhs_ptr[n] = rhs_m_ptr[n];
        diag_add = diag_add_m_ptr[n];
      } else {
        mu_ = mu_p_;
#ifdef P4_TO_P8
        interp_local.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, mue_p_zz_ptr, interp_method_);
#else
        interp_local.set_input(mue_p_ptr, mue_p_xx_ptr, mue_p_yy_ptr, interp_method_);
#endif
        if (setup_rhs) rhs_ptr[n] = rhs_p_ptr[n];
        diag_add = diag_add_p_ptr[n];
      }

      if (use_sc_scheme_)
        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
          if (neighbors_exist[idx])
            neighbors_exist[idx] = neighbors_exist[idx] && (areas_ptr[neighbors[idx]] > interface_rel_thresh_);

      // check for hanging neighbors
      int network[num_neighbors_cube_];
      bool hanging_neighbor[num_neighbors_cube_];
      bool expand[2*P4EST_DIM];
      bool attempt_to_expand = false;

      if (try_remove_hanging_cells_ && use_sc_scheme_)
      {
        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
          network[idx] = neighbors_exist[idx] ? (int) node_type_p[neighbors[idx]] : 0;

        find_hanging_cells(network, hanging_neighbor);

        for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
          expand[dir] = false;

#ifdef P4_TO_P8
        for (unsigned short k = 0; k < 3; ++k)
#endif
          for (unsigned short j = 0; j < 3; ++j)
            for (unsigned short i = 0; i < 3; ++i)
            {
#ifdef P4_TO_P8
              unsigned short idx = 9*k + 3*j +i;
#else
              unsigned short idx = 3*j +i;
#endif
              if (neighbors_exist[idx])
                if (hanging_neighbor[idx])
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


        for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
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

        if (!is_node_xmWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmin = x_C - .5*dx_min_; interp_xmin = x_C - dx_min_; } else { fv_xmin = x_C; interp_xmin = x_C; }
        if (!is_node_xpWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmax = x_C + .5*dx_min_; interp_xmax = x_C + dx_min_; } else { fv_xmax = x_C; interp_xmax = x_C; }
        if (!is_node_ymWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymin = y_C - .5*dy_min_; interp_ymin = y_C - dy_min_; } else { fv_ymin = y_C; interp_ymin = y_C; }
        if (!is_node_ypWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymax = y_C + .5*dy_min_; interp_ymax = y_C + dy_min_; } else { fv_ymax = y_C; interp_ymax = y_C; }
#ifdef P4_TO_P8
        if (!is_node_zmWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmin = z_C - .5*dz_min_; interp_zmin = z_C - dz_min_; } else { fv_zmin = z_C; interp_zmin = z_C; }
        if (!is_node_zpWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmax = z_C + .5*dz_min_; interp_zmax = z_C + dz_min_; } else { fv_zmax = z_C; interp_zmax = z_C; }
#endif

        if (attempt_to_expand)
        {
          ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment...\n");
          if (expand[dir::f_m00]) { fv_size_x += cube_refinement_; fv_xmin -= cell_expansion*dx_min_; }
          if (expand[dir::f_p00]) { fv_size_x += cube_refinement_; fv_xmax += cell_expansion*dx_min_; }
          if (expand[dir::f_0m0]) { fv_size_y += cube_refinement_; fv_ymin -= cell_expansion*dy_min_; }
          if (expand[dir::f_0p0]) { fv_size_y += cube_refinement_; fv_ymax += cell_expansion*dy_min_; }
#ifdef P4_TO_P8
          if (expand[dir::f_00m]) { fv_size_z += cube_refinement_; fv_zmin -= cell_expansion*dz_min_; }
          if (expand[dir::f_00p]) { fv_size_z += cube_refinement_; fv_zmax += cell_expansion*dz_min_; }
#endif
        }

        if (cube_refinement_ == 0)
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
        for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
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
          for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
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
            for (unsigned short k = km; k <= kp; ++k)
#endif
              for (unsigned short j = jm; j <= jp; ++j)
                for (unsigned short i = im; i <= ip; ++i)
                {
#ifdef P4_TO_P8
                  unsigned short idx = 9*k + 3*j + i;
#else
                  unsigned short idx = 3*j + i;
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

      for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
#ifdef P4_TO_P8
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx], cube_ifc_z[phi_idx]);
#else
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx]);
#endif
      }

      // compute cut-cell volume
      double volume_cut_cell = 0.;

      for (unsigned int i = 0; i < cube_dom_w.size(); ++i)
      {
        volume_cut_cell += cube_dom_w[i];
      }

      // compute area of the interface and integral of function from boundary conditions
      double integral_bc = 0.;

      std::vector<double> interface_centroid_x(num_interfaces_, 0);
      std::vector<double> interface_centroid_y(num_interfaces_, 0);
#ifdef P4_TO_P8
      std::vector<double> interface_centroid_z(num_interfaces_, 0);
#endif
      std::vector<double> interface_area(num_interfaces_, 0);

      for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
      {
        if (cube_ifc_w[phi_idx].size() > 0)
        {
          for (unsigned int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
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

      // compute cut-face areas and their centroids
      std::vector<double> cube_dir_w;
      std::vector<double> cube_dir_x;
      std::vector<double> cube_dir_y;
#ifdef P4_TO_P8
      std::vector<double> cube_dir_z;
#endif

      std::vector<double> face_area      (P4EST_FACES, 0);
      std::vector<double> face_centroid_x(P4EST_FACES, 0);
      std::vector<double> face_centroid_y(P4EST_FACES, 0);
#ifdef P4_TO_P8
      std::vector<double> face_centroid_z(P4EST_FACES, 0);
#endif

      double face_area_max = 0;

      for (int dir_idx = 0; dir_idx < P4EST_FACES; ++dir_idx)
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
          for (unsigned int i = 0; i < cube_dir_w.size(); ++i)
          {
            face_area[dir_idx] += cube_dir_w[i];
            if (use_sc_scheme_)
            {
              face_centroid_x[dir_idx] += cube_dir_w[i]*(cube_dir_x[i] - x_C);
              face_centroid_y[dir_idx] += cube_dir_w[i]*(cube_dir_y[i] - y_C);
#ifdef P4_TO_P8
              face_centroid_z[dir_idx] += cube_dir_w[i]*(cube_dir_z[i] - z_C);
#endif
            }
          }

          face_area_max = MAX(face_area_max, face_area[dir_idx]);

          if (use_sc_scheme_)
          {
            face_centroid_x[dir_idx] /= face_area[dir_idx];
            face_centroid_y[dir_idx] /= face_area[dir_idx];
#ifdef P4_TO_P8
            face_centroid_z[dir_idx] /= face_area[dir_idx];
#endif
          }
        }
      }

      face_centroid_x[dir::f_m00] = fv_xmin - x_C; face_centroid_x[dir::f_p00] = fv_xmax - x_C;
      face_centroid_y[dir::f_0m0] = fv_ymin - y_C; face_centroid_y[dir::f_0p0] = fv_ymax - y_C;
#ifdef P4_TO_P8
      face_centroid_z[dir::f_00m] = fv_zmin - z_C; face_centroid_z[dir::f_00p] = fv_zmax - z_C;
#endif

      if (use_sc_scheme_) face_area_max = areas_ptr[n];

      if (face_area_max > interface_rel_thresh_) // check if at least one face has a 'good connection' to neighboring cells
      {
        if (setup_rhs)
        {
          rhs_ptr[n] = rhs_ptr[n]*volume_cut_cell + integral_bc;
        }

        //---------------------------------------------------------------------
        // contributions through cell faces
        //---------------------------------------------------------------------
        double w[num_neighbors_cube_] = { 0,0,0, 0,0,0, 0,0,0,
                                  #ifdef P4_TO_P8
                                          0,0,0, 0,0,0, 0,0,0,
                                          0,0,0, 0,0,0, 0,0,0
                                  #endif
                                        };

        bool   neighbors_exist_face[num_neighbors_face_];
        bool   map_face            [num_neighbors_face_];
        double weights_face        [num_neighbors_face_];

        double theta = EPS;

        for (unsigned short dim = 0; dim < P4EST_DIM; ++dim) // loop over all dimensions
        {
          for (unsigned short sign = 0; sign < 2; ++sign) // negative and positive directions
          {
            unsigned short dir = 2*dim + sign;

            if (face_area[dir]/face_area_scalling_ > interface_rel_thresh_)
            {
              if (!neighbors_exist[f2c_p[dir][nnf_00]])
              {
                std::cout << "Warning: neighbor doesn't exist in the " << dir << "-th direction."
                          << " Own number: " << n
                          << " Nei number: " << neighbors[f2c_p[dir][nnf_00]]
//                          << " Own area: " << areas_ptr[neighbors[f2c_m[dir][nnf_00]]]
//                          << " Nei area: " << areas_ptr[neighbors[f2c_p[dir][nnf_00]]]
                          << " Face Area:" << face_area[dir]/face_area_scalling_ << "\n";
              }
              else
              {
#ifdef P4_TO_P8
                double face_centroid_xyz_gl[P4EST_DIM] = { x_C + face_centroid_x[dir],
                                                           y_C + face_centroid_y[dir],
                                                           z_C + face_centroid_z[dir] };
#else
                double face_centroid_xyz_gl[P4EST_DIM] = { x_C + face_centroid_x[dir],
                                                           y_C + face_centroid_y[dir] };
#endif
                double mu_val = variable_mu_ ? interp_local.value(face_centroid_xyz_gl) : mu_;

                double flux = mu_val * face_area[dir] / dxyz_m_[dim];

                if (!use_sc_scheme_)
                {
                  w[f2c_m[dir][nnf_00]] += flux;
                  w[f2c_p[dir][nnf_00]] -= flux;
                }
                else
                {
                  for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
                    neighbors_exist_face[nn] = neighbors_exist[f2c_m[dir][nn]] && neighbors_exist[f2c_p[dir][nn]];

#ifdef P4_TO_P8
                  double centroid_xyz[] = { face_centroid_x[dir]/dx_min_,
                                            face_centroid_y[dir]/dy_min_,
                                            face_centroid_z[dir]/dz_min_ };
                  double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], centroid_xyz[k_idx[dim]], neighbors_exist_face, weights_face, theta, map_face);
#else
                  double centroid_xyz[] = { face_centroid_x[dir]/dx_min_,
                                            face_centroid_y[dir]/dy_min_};
                  double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], neighbors_exist_face, weights_face, theta, map_face);
#endif

                  if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

                  for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
                    if (map_face[nn])
                    {
                      w[f2c_m[dir][nn]] += weights_face[nn] * flux;
                      w[f2c_p[dir][nn]] -= weights_face[nn] * flux;
                    }
                }

              }
            }
          }
        }

        bool sc_scheme_successful = false;
        // a variation of the least-square fitting approach
        if (use_sc_scheme_)
        {
          sc_scheme_successful = true;

          // linear system
          char num_constraints = num_neighbors_cube_ + num_interfaces_;

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

          for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
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
              for (unsigned int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
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
              xyz_pr[0] = MIN(xyz_pr[0], interp_xmax); xyz_pr[0] = MAX(xyz_pr[0], interp_xmin);
              xyz_pr[1] = MIN(xyz_pr[1], interp_ymax); xyz_pr[1] = MAX(xyz_pr[1], interp_ymin);
#ifdef P4_TO_P8
              xyz_pr[2] = MIN(xyz_pr[2], interp_zmax); xyz_pr[2] = MAX(xyz_pr[2], interp_zmin);
#endif

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              col_1st[num_neighbors_cube_ + phi_idx] = bc_coeff_proj;
              col_2nd[num_neighbors_cube_ + phi_idx] = (mu_proj*nx + bc_coeff_proj*(xyz_pr[0] - xyz_C[0]));
              col_3rd[num_neighbors_cube_ + phi_idx] = (mu_proj*ny + bc_coeff_proj*(xyz_pr[1] - xyz_C[1]));
#ifdef P4_TO_P8
              col_4th[num_neighbors_cube_ + phi_idx] = (mu_proj*nz + bc_coeff_proj*(xyz_pr[2] - xyz_C[2]));
#endif
              bc_coeffs[phi_idx] = bc_coeff_proj;
              bc_values[phi_idx] = bc_value_proj;
            }
          }

          double gamma = 3.*log(10.);

          for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
          {
            if (cube_ifc_w[phi_idx].size() > 0 && sc_scheme_successful)
            {
              std::vector<double> weight(num_constraints, 0);

              unsigned short num_constraints_present = 0;

#ifdef P4_TO_P8
              for (unsigned short k = 0; k < 3; ++k)
#endif
                for (unsigned short j = 0; j < 3; ++j)
                  for (unsigned short i = 0; i < 3; ++i)
                  {
#ifdef P4_TO_P8
                    unsigned short idx = 9*k+3*j+i;
#else
                    unsigned short idx = 3*j+i;
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

              for (unsigned short phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                if (cube_ifc_w[phi_jdx].size() > 0)
                {
                  weight[num_neighbors_cube_ + phi_jdx] = exp(-gamma*(SQR((x0[phi_jdx]-x0[phi_idx])/dx_min_) +
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

              for (unsigned short nei = 0; nei < num_constraints; ++nei)
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
              if (!inv_mat4(A, A_inv)) throw;
#else
              if (!inv_mat3(A, A_inv)) throw;
#endif

              // compute Taylor expansion coefficients
              std::vector<double> coeff_const_term(num_constraints, 0);
              std::vector<double> coeff_x_term    (num_constraints, 0);
              std::vector<double> coeff_y_term    (num_constraints, 0);
#ifdef P4_TO_P8
              std::vector<double> coeff_z_term    (num_constraints, 0);
#endif

              for (unsigned short nei = 0; nei < num_constraints; ++nei)
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
              for (unsigned short phi_jdx = 0; phi_jdx < num_interfaces_; ++phi_jdx)
              {
                rhs_const_term += coeff_const_term[num_neighbors_cube_ + phi_jdx] * bc_values[phi_jdx];
                rhs_x_term     += coeff_x_term    [num_neighbors_cube_ + phi_jdx] * bc_values[phi_jdx];
                rhs_y_term     += coeff_y_term    [num_neighbors_cube_ + phi_jdx] * bc_values[phi_jdx];
#ifdef P4_TO_P8
                rhs_z_term     += coeff_z_term    [num_neighbors_cube_ + phi_jdx] * bc_values[phi_jdx];
#endif
              }

              // compute integrals
              double const_term = S[phi_idx]*bc_coeffs[phi_idx];
              double x_term     = S[phi_idx]*bc_coeffs[phi_idx]*(x0[phi_idx] - xyz_C[0]);
              double y_term     = S[phi_idx]*bc_coeffs[phi_idx]*(y0[phi_idx] - xyz_C[1]);
#ifdef P4_TO_P8
              double z_term     = S[phi_idx]*bc_coeffs[phi_idx]*(z0[phi_idx] - xyz_C[2]);
#endif
//              double const_term = S[phi_idx]*(*bc_interface_coeff_->at(phi_idx))(x0[phi_idx], y0[phi_idx]);
//              double x_term     = S[phi_idx]*(*bc_interface_coeff_->at(phi_idx))(x0[phi_idx], y0[phi_idx])*(x0[phi_idx] - xyz_C[0]);
//              double y_term     = S[phi_idx]*(*bc_interface_coeff_->at(phi_idx))(x0[phi_idx], y0[phi_idx])*(y0[phi_idx] - xyz_C[1]);
//#ifdef P4_TO_P8
//              double z_term     = S[phi_idx]*(*bc_interface_coeff_->at(phi_idx))(x0[phi_idx], y0[phi_idx])*(z0[phi_idx] - xyz_C[2]);
//#endif

              // matrix coefficients
              for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei)
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
                rhs_ptr[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term + rhs_z_term*z_term;
#else
                rhs_ptr[n] -= rhs_const_term*const_term + rhs_x_term*x_term + rhs_y_term*y_term;
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

          for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; phi_idx++)
          {
            for (unsigned int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
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
              compute_normal(phi_p[present_interfaces[i]], qnnn, &N_mat[i*P4EST_DIM]);

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
            for (unsigned short i = 0; i < num_interfaces_present; ++i)
            {
              unsigned short phi_idx = present_interfaces[i];

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
#ifdef P4_TO_P8
              double z0 = 0;
#endif

              for (unsigned int ii = 0; ii < cube_ifc_w[phi_idx].size(); ++ii)
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

              xyz_pr[0] = MIN(xyz_pr[0], interp_xmax); xyz_pr[0] = MAX(xyz_pr[0], interp_xmin);
              xyz_pr[1] = MIN(xyz_pr[1], interp_ymax); xyz_pr[1] = MAX(xyz_pr[1], interp_ymin);
#ifdef P4_TO_P8
              xyz_pr[2] = MIN(xyz_pr[2], interp_zmax); xyz_pr[2] = MAX(xyz_pr[2], interp_zmin);
#endif

              double mu_proj       = variable_mu_ ? interp_local.value(xyz_pr) : mu_;
              double bc_coeff_proj = bc_interface_coeff_->at(phi_idx)->value(xyz_pr);
              double bc_value_proj = bc_interface_value_->at(phi_idx)->value(xyz_pr);

              for (unsigned short dim = 0; dim < P4EST_DIM; ++dim)
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
            inv_mat3(N_mat, N_inv_mat);
#else
            inv_mat2(N_mat, N_inv_mat);
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
            for (unsigned short interface_idx = 0; interface_idx < present_interfaces.size(); ++interface_idx)
            {
              int phi_idx = present_interfaces[interface_idx];

              for (unsigned int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
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
                  rhs_ptr[n] -= cube_ifc_w[phi_idx][i] * taylor_expansion_const_term;
              }

            }

          }

          // Cells without kinks
          if (!is_there_kink || !kink_special_treatment_)
          {
            for (unsigned int interface_idx = 0; interface_idx < present_interfaces.size(); interface_idx++)
            {
              unsigned short i_phi = present_interfaces[interface_idx];

              if (bc_interface_type_->at(i_phi) == ROBIN)
              {
                double measure_of_iface = measure_of_interface[i_phi];

                // compute projection point
                find_projection_(phi_p[i_phi], qnnn, dxyz_pr, dist);

                for (unsigned short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                  xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

                xyz_pr[0] = MIN(xyz_pr[0], interp_xmax); xyz_pr[0] = MAX(xyz_pr[0], interp_xmin);
                xyz_pr[1] = MIN(xyz_pr[1], interp_ymax); xyz_pr[1] = MAX(xyz_pr[1], interp_ymin);
#ifdef P4_TO_P8
                xyz_pr[2] = MIN(xyz_pr[2], interp_zmax); xyz_pr[2] = MAX(xyz_pr[2], interp_zmin);
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
                  rhs_ptr[n] -= measure_of_iface*bc_coeff_proj*bc_value_proj*dist/(bc_coeff_proj*dist-mu_proj);
                }

                if (fabs(bc_coeff_proj) > 0) matrix_has_nullspace_ = false;
              }
            }
          }

        } // end of symmetric scheme

        // add diagonal term
        w[nn_000] += diag_add*volume_cut_cell;

        // scale all coefficient by the diagonal one and insert them into the matrix
        if (setup_rhs)
        {
          rhs_ptr[n] /= w[nn_000];
        }

        if (setup_matrix)
        {
          if (fabs(diag_add) > 0) matrix_has_nullspace_ = false;
          if (keep_scalling_) scalling_[n] = w[nn_000]/full_cell_volume; // scale to measure of the full cell dx*dy(*dz) for consistence

          double w_000 = w[nn_000];

          if (w_000 != w_000) throw std::domain_error("Diag is nan\n");

          for (int i = 0; i < num_neighbors_cube_; ++i)
            w[i] /= w_000;

          w[nn_000] = 1.;

#ifdef DO_NOT_PREALLOCATE
          ent.n = node_000_g; ent.val = 1.; row->push_back(ent);
#else
          ierr = MatSetValue(A_, node_000_g, node_000_g, 1,  ADD_VALUES); CHKERRXX(ierr);
#endif

          for (int i = 0; i < num_neighbors_cube_; ++i)
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

        // if finite volume too small, ignore the node
        if (setup_matrix && volume_cut_cell != 0.)
          ierr = PetscPrintf(p4est_->mpicomm, "Ignoring tiny volume %e with max face area %e\n", volume_cut_cell, face_area_max);

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
          rhs_ptr[n] = 0;
      }

    } // end of if (discretization_scheme_ = discretization_scheme_t::FVM)
    //*
    else if (discretization_scheme_ == JUMP)
    {
      interface_present = true;
      // reconstruct domain and compute geometric quantities
#ifdef P4_TO_P8
      cube3_mls_t cube;
#else
      cube2_mls_t cube;
#endif

      // determine dimensions of cube
      fv_size_x = 0;
      fv_size_y = 0;
#ifdef P4_TO_P8
      fv_size_z = 0;
#endif

      if (!is_node_xmWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmin = x_C - .5*dx_min_; interp_xmin = x_C - dx_min_; } else { fv_xmin = x_C; interp_xmin = x_C; }
      if (!is_node_xpWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmax = x_C + .5*dx_min_; interp_xmax = x_C + dx_min_; } else { fv_xmax = x_C; interp_xmax = x_C; }
      if (!is_node_ymWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymin = y_C - .5*dy_min_; interp_ymin = y_C - dy_min_; } else { fv_ymin = y_C; interp_ymin = y_C; }
      if (!is_node_ypWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymax = y_C + .5*dy_min_; interp_ymax = y_C + dy_min_; } else { fv_ymax = y_C; interp_ymax = y_C; }
#ifdef P4_TO_P8
      if (!is_node_zmWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmin = z_C - .5*dz_min_; interp_zmin = z_C - dz_min_; } else { fv_zmin = z_C; interp_zmin = z_C; }
      if (!is_node_zpWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmax = z_C + .5*dz_min_; interp_zmax = z_C + dz_min_; } else { fv_zmax = z_C; interp_zmax = z_C; }
#endif

      if (cube_refinement_ == 0)
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

      cube.initialize(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);

      // get points at which values of level-set functions are needed
      std::vector<double> x_grid; cube.get_x_coord(x_grid);
      std::vector<double> y_grid; cube.get_y_coord(y_grid);
#ifdef P4_TO_P8
      std::vector<double> z_grid; cube.get_z_coord(z_grid);
#endif
      int points_total = x_grid.size();

      std::vector<double> phi_cube(ii_.num_interfaces*points_total,-1);

      // sample values of level-set functions at those points
      for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
      {
#ifdef P4_TO_P8
        phi_interp_local.set_input(ii_phi_p[phi_idx], ii_phi_xx_p[phi_idx], ii_phi_yy_p[phi_idx], ii_phi_zz_p[phi_idx], interp_method_);
#else
        phi_interp_local.set_input(ii_phi_p[phi_idx], ii_phi_xx_p[phi_idx], ii_phi_yy_p[phi_idx], interp_method_);
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
      cube.reconstruct(phi_cube, *ii_.action, *ii_.color);

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

      std::vector<std::vector<double> > cube_ifc_w(ii_.num_interfaces);
      std::vector<std::vector<double> > cube_ifc_x(ii_.num_interfaces);
      std::vector<std::vector<double> > cube_ifc_y(ii_.num_interfaces);
#ifdef P4_TO_P8
      std::vector<std::vector<double> > cube_ifc_z(ii_.num_interfaces);
#endif

      for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
      {
#ifdef P4_TO_P8
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx], cube_ifc_z[phi_idx]);
#else
        cube.quadrature_over_interface(phi_idx, cube_ifc_w[phi_idx], cube_ifc_x[phi_idx], cube_ifc_y[phi_idx]);
#endif
      }

      // compute cut-cell volume
      double volume_cut_cell_m = 0.;
      double volume_cut_cell_p = 0.;

      for (unsigned int i = 0; i < cube_dom_w.size(); ++i)
      {
        volume_cut_cell_m += cube_dom_w[i];
      }

      volume_cut_cell_p = full_cell_volume - volume_cut_cell_m;

      // compute area of the interface and integral of function from jump conditions
      double integral_bc = 0.;

      std::vector<double> interface_centroid_x(ii_.num_interfaces, 0);
      std::vector<double> interface_centroid_y(ii_.num_interfaces, 0);
#ifdef P4_TO_P8
      std::vector<double> interface_centroid_z(ii_.num_interfaces, 0);
#endif
      std::vector<double> interface_area(ii_.num_interfaces, 0);

      for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
      {
        if (cube_ifc_w[phi_idx].size() > 0)
        {
          for (unsigned int i = 0; i < cube_ifc_w[phi_idx].size(); ++i)
          {
            interface_area[phi_idx] += cube_ifc_w[phi_idx][i];
#ifdef P4_TO_P8
            integral_bc    += cube_ifc_w[phi_idx][i] * (*jump_flux_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i], cube_ifc_z[phi_idx][i]);
#else
            integral_bc    += cube_ifc_w[phi_idx][i] * (*jump_flux_->at(phi_idx))(cube_ifc_x[phi_idx][i], cube_ifc_y[phi_idx][i]);
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
        }
      }

      if (setup_rhs)
      {
        rhs_ptr[n] = rhs_m_ptr[n]*volume_cut_cell_m + rhs_p_ptr[n]*volume_cut_cell_p - integral_bc;
//        rhs_ptr[n] = 0;
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
      double face_area_full[] = { full_sx, full_sx, full_sy, full_sy, full_sz, full_sz };
#else
      double face_area_full[] = { full_sx, full_sx, full_sy, full_sy };
#endif

      double face_center_x[] = { fv_xmin - x_C, fv_xmax - x_C, 0, 0, 0, 0 };
      double face_center_y[] = { 0, 0, fv_ymin - y_C, fv_ymax - y_C, 0, 0 };
#ifdef P4_TO_P8
      double face_center_z[] = { 0, 0, 0, 0, fv_zmin - z_C, fv_zmax - z_C };
#endif

      std::vector<double> face_m_area      (P4EST_FACES, 0);
      std::vector<double> face_m_centroid_x(P4EST_FACES, 0);
      std::vector<double> face_m_centroid_y(P4EST_FACES, 0);
#ifdef P4_TO_P8
      std::vector<double> face_m_centroid_z(P4EST_FACES, 0);
#endif

      std::vector<double> face_p_area      (P4EST_FACES, 0);
      std::vector<double> face_p_centroid_x(P4EST_FACES, 0);
      std::vector<double> face_p_centroid_y(P4EST_FACES, 0);
#ifdef P4_TO_P8
      std::vector<double> face_p_centroid_z(P4EST_FACES, 0);
#endif

      double face_m_area_max = 0;
      double face_p_area_max = 0;

      for (int dir_idx = 0; dir_idx < P4EST_FACES; ++dir_idx)
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
          for (unsigned int i = 0; i < cube_dir_w.size(); ++i)
          {
            face_m_area[dir_idx] += cube_dir_w[i];
            if (use_sc_scheme_)
            {
              face_m_centroid_x[dir_idx] += cube_dir_w[i]*(cube_dir_x[i] - x_C);
              face_m_centroid_y[dir_idx] += cube_dir_w[i]*(cube_dir_y[i] - y_C);
#ifdef P4_TO_P8
              face_m_centroid_z[dir_idx] += cube_dir_w[i]*(cube_dir_z[i] - z_C);
#endif
            }
          }

          if (use_sc_scheme_)
          {
            face_m_centroid_x[dir_idx] /= face_m_area[dir_idx];
            face_m_centroid_y[dir_idx] /= face_m_area[dir_idx];
#ifdef P4_TO_P8
            face_m_centroid_z[dir_idx] /= face_m_area[dir_idx];
#endif
          }
        }

        face_p_area[dir_idx] = face_area_full[dir_idx] - face_m_area[dir_idx];

        if (use_sc_scheme_)
        {
          if (face_p_area[dir_idx]/face_area_scalling_ > interface_rel_thresh_)
          {
            face_p_centroid_x[dir_idx] = face_center_x[dir_idx] - face_m_area[dir_idx]/face_p_area[dir_idx] * (face_m_centroid_x[dir_idx] - face_center_x[dir_idx]);
            face_p_centroid_y[dir_idx] = face_center_y[dir_idx] - face_m_area[dir_idx]/face_p_area[dir_idx] * (face_m_centroid_y[dir_idx] - face_center_y[dir_idx]);
#ifdef P4_TO_P8
            face_p_centroid_z[dir_idx] = face_center_z[dir_idx] - face_m_area[dir_idx]/face_p_area[dir_idx] * (face_m_centroid_z[dir_idx] - face_center_z[dir_idx]);
#endif
          }
          else
          {
            face_p_centroid_x[dir_idx] = face_center_x[dir_idx];
            face_p_centroid_y[dir_idx] = face_center_y[dir_idx];
#ifdef P4_TO_P8
            face_p_centroid_z[dir_idx] = face_center_z[dir_idx];
#endif
          }
        }

        face_m_area_max = MAX(face_m_area_max, face_m_area[dir_idx]);
        face_p_area_max = MAX(face_p_area_max, face_p_area[dir_idx]);
      }


      face_m_centroid_x[dir::f_m00] = fv_xmin - x_C; face_m_centroid_x[dir::f_p00] = fv_xmax - x_C;
      face_p_centroid_x[dir::f_m00] = fv_xmin - x_C; face_p_centroid_x[dir::f_p00] = fv_xmax - x_C;
      face_m_centroid_y[dir::f_0m0] = fv_ymin - y_C; face_m_centroid_y[dir::f_0p0] = fv_ymax - y_C;
      face_p_centroid_y[dir::f_0m0] = fv_ymin - y_C; face_p_centroid_y[dir::f_0p0] = fv_ymax - y_C;
#ifdef P4_TO_P8
      face_m_centroid_z[dir::f_00m] = fv_zmin - z_C; face_m_centroid_z[dir::f_00p] = fv_zmax - z_C;
      face_p_centroid_z[dir::f_00m] = fv_zmin - z_C; face_p_centroid_z[dir::f_00p] = fv_zmax - z_C;
#endif

      face_m_area_max /= face_area_scalling_;
      face_p_area_max /= face_area_scalling_;

      if (use_sc_scheme_)
      {
        face_m_area_max = areas_m_ptr[n];
        face_p_area_max = areas_p_ptr[n];

        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
          if (neighbors_exist[idx])
          {
            neighbors_exist_p[idx] = neighbors_exist[idx] && (areas_p_ptr[neighbors[idx]] > interface_rel_thresh_);
            neighbors_exist_m[idx] = neighbors_exist[idx] && (areas_m_ptr[neighbors[idx]] > interface_rel_thresh_);
          }
      } else {
        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
          if (neighbors_exist[idx])
          {
            neighbors_exist_p[idx] = neighbors_exist[idx];
            neighbors_exist_m[idx] = neighbors_exist[idx];
          }
      }

      //---------------------------------------------------------------------
      // contributions through cell faces
      //---------------------------------------------------------------------
      double w_m[num_neighbors_cube_] = { 0,0,0, 0,0,0, 0,0,0,
                                    #ifdef P4_TO_P8
                                          0,0,0, 0,0,0, 0,0,0,
                                          0,0,0, 0,0,0, 0,0,0
                                    #endif
                                        };
      double w_p[num_neighbors_cube_] = { 0,0,0, 0,0,0, 0,0,0,
                                    #ifdef P4_TO_P8
                                          0,0,0, 0,0,0, 0,0,0,
                                          0,0,0, 0,0,0, 0,0,0
                                    #endif
                                        };

      bool   neighbors_exist_face[num_neighbors_face_];
      bool   map_face            [num_neighbors_face_];
      double weights_face        [num_neighbors_face_];

      double theta = EPS;

      if (face_m_area_max < interface_rel_thresh_) { face_m_area_max = 0; volume_cut_cell_m = 0; }
      if (face_p_area_max < interface_rel_thresh_) { face_p_area_max = 0; volume_cut_cell_p = 0; }

      if (face_m_area_max < interface_rel_thresh_ &&
          face_p_area_max < interface_rel_thresh_)
        throw;

//      w_m[nn_000] = 4;
//      w_m[nn_m00] =-1;
//      w_m[nn_p00] =-1;
//      w_m[nn_0m0] =-1;
//      w_m[nn_0p0] =-1;
//      rhs_ptr[n] = sin(x_C)*cos(y_C);
//      rhs_ptr[n] = dx_min_*dx_min_*rhs_p_ptr[n];
      //*
      for (unsigned short dom = 0; dom < 2; ++dom) // negative and positive domains
      {
        if (dom == 0 && face_m_area_max <= interface_rel_thresh_ ||
            dom == 1 && face_p_area_max <= interface_rel_thresh_)
          continue;

        double *face_area = (dom == 0 ? face_m_area.data() : face_p_area.data());
//        double *areas_ptr = (dom == 0 ? areas_m_ptr : areas_p_ptr);

        double *face_centroid_x = (dom == 0 ? face_m_centroid_x.data() : face_p_centroid_x.data());
        double *face_centroid_y = (dom == 0 ? face_m_centroid_y.data() : face_p_centroid_y.data());
#ifdef P4_TO_P8
        double *face_centroid_z = (dom == 0 ? face_m_centroid_z.data() : face_p_centroid_z.data());
#endif

        double *w = (dom == 0 ? w_m : w_p);

        double mu_ = (dom == 0 ? mu_m_ : mu_p_);

        double volume_cut_cell = (dom == 0 ? volume_cut_cell_m : volume_cut_cell_p);

        double *diag_add_ptr = (dom == 0 ? diag_add_m_ptr : diag_add_p_ptr);

        my_p4est_interpolation_nodes_local_t *mu_interp = (dom == 0 ? &mu_m_interp : &mu_p_interp);

        for (unsigned short dim = 0; dim < P4EST_DIM; ++dim) // loop over all dimensions
        {
          for (unsigned short sign = 0; sign < 2; ++sign) // negative and positive directions
          {
            unsigned short dir = 2*dim + sign;

            if (face_area[dir]/face_area_scalling_ > interface_rel_thresh_)
            {
              if (!neighbors_exist[f2c_p[dir][nnf_00]])
              {
                std::cout << "Warning: neighbor doesn't exist in the zp-direction."
                          << " Own number: " << n
                          << " Nei number: " << neighbors[f2c_p[dir][nnf_00]]
//                          << " Own area: " << areas_ptr[neighbors[f2c_m[dir][nnf_00]]]
//                          << " Nei area: " << areas_ptr[neighbors[f2c_p[dir][nnf_00]]]
                          << " Face Area:" << face_area[dir]/face_area_scalling_ << "\n";
              }
              else
              {
#ifdef P4_TO_P8
                double face_centroid_xyz_gl[P4EST_DIM] = { x_C + face_centroid_x[dir],
                                                           y_C + face_centroid_y[dir],
                                                           z_C + face_centroid_z[dir] };
#else
                double face_centroid_xyz_gl[P4EST_DIM] = { x_C + face_centroid_x[dir],
                                                           y_C + face_centroid_y[dir] };
#endif
                double mu_val = variable_mu_ ? mu_interp->value(face_centroid_xyz_gl) : mu_;

                double flux = mu_val * face_area[dir] / dxyz_m_[dim];

                if (!use_sc_scheme_)
                {
                  w[f2c_m[dir][nnf_00]] += flux;
                  w[f2c_p[dir][nnf_00]] -= flux;
                }
                else
                {
                  for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
                    neighbors_exist_face[nn] = neighbors_exist[f2c_m[dir][nn]] && neighbors_exist[f2c_p[dir][nn]];

#ifdef P4_TO_P8
                  double centroid_xyz[] = { face_centroid_x[dir]/dx_min_,
                                            face_centroid_y[dir]/dy_min_,
                                            face_centroid_z[dir]/dz_min_ };
                  double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], centroid_xyz[k_idx[dim]], neighbors_exist_face, weights_face, theta, map_face);
#else
                  double centroid_xyz[] = { face_centroid_x[dir]/dx_min_,
                                            face_centroid_y[dir]/dy_min_};
                  double mask_result = compute_weights_through_face(centroid_xyz[j_idx[dim]], neighbors_exist_face, weights_face, theta, map_face);
#endif

                  //            if (setup_matrix) mask_p[n] = MAX(mask_p[n], mask_result);

                  for (unsigned short nn = 0; nn < num_neighbors_face_; ++nn)
                  {
                    if (map_face[nn]) { w[f2c_m[dir][nn]] += weights_face[nn] * flux;   w[f2c_p[dir][nn]] -= weights_face[nn] * flux; }
                  }
                }

              }
            }
          }
        }

        w[nn_000] += diag_add_ptr[n]*volume_cut_cell;
      }
      //*/


      // put values into matrices
      for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
      {
        if (neighbors_exist[i])
        {
          PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
          if (ii_phi_eff_p[neighbors[i]] < 0)
          {
            if (neighbors_exist_m[i] && fabs(w_m[i]) > EPS)
            {
              if (w_m[i] != w_m[i]) throw std::domain_error("Matrix element is nan\n");
              ent.n = node_nei_g;
              ent.val = w_m[i];
              row->push_back(ent);
              ( neighbors[i] < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
            }
            if (neighbors_exist_p[i] && fabs(w_p[i]) > EPS)
            {
              if (w_p[i] != w_p[i]) throw std::domain_error("Matrix element is nan\n");
              ent.n = node_nei_g;
              ent.val = w_p[i];
              row_B->push_back(ent);
              ( neighbors[i] < num_owned_local ) ? B_d_nnz[n]++ : B_o_nnz[n]++;
            }
          }
          else
          {
            if (neighbors_exist_m[i] && fabs(w_m[i]) > EPS)
            {
              if (w_m[i] != w_m[i]) throw std::domain_error("Matrix element is nan\n");
              ent.n = node_nei_g;
              ent.val = w_m[i];
              row_B->push_back(ent);
              ( neighbors[i] < num_owned_local ) ? B_d_nnz[n]++ : B_o_nnz[n]++;
            }
            if (neighbors_exist_p[i] && fabs(w_p[i]) > EPS)
            {
              if (w_p[i] != w_p[i]) throw std::domain_error("Matrix element is nan\n");
              ent.n = node_nei_g;
              ent.val = w_p[i];
              row->push_back(ent);
              ( neighbors[i] < num_owned_local ) ? d_nnz[n]++ : o_nnz[n]++;
            }
          }
        }
      }

      // express values at ghost cells using values at real cells
      if (face_p_area_max > interface_rel_thresh_ &&
          face_m_area_max > interface_rel_thresh_)
      {
        std::vector<double> w_ghosts(num_neighbors_cube_, 0);

        double normal[P4EST_DIM];

        // compute projection point
        find_projection_(ii_phi_p[0], qnnn, dxyz_pr, dist, normal);

        for (unsigned short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
          xyz_pr[i_dim] = xyz_C[i_dim] + dxyz_pr[i_dim];

        xyz_pr[0] = MIN(xyz_pr[0], interp_xmax); xyz_pr[0] = MAX(xyz_pr[0], interp_xmin);
        xyz_pr[1] = MIN(xyz_pr[1], interp_ymax); xyz_pr[1] = MAX(xyz_pr[1], interp_ymin);
#ifdef P4_TO_P8
        xyz_pr[2] = MIN(xyz_pr[2], interp_zmax); xyz_pr[2] = MAX(xyz_pr[2], interp_zmin);
#endif

        // sample values at the projection point
        double mu_m_proj = variable_mu_ ? mu_m_interp.value(xyz_pr) : mu_m_;
        double mu_p_proj = variable_mu_ ? mu_p_interp.value(xyz_pr) : mu_p_;
        double flux_proj = jump_flux_->at(0)->value(xyz_pr);

        // count numbers of neighbors in negative and positive domains
        unsigned short num_neg = 0;
        unsigned short num_pos = 0;
        for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei)
          if (neighbors_exist[nei] && nei != nn_000)
          {
            if (ii_phi_eff_p[neighbors[nei]] < 0) num_neg++;
            else                                  num_pos++;
          }

        // determine which side to use based on values of diffucion coefficients and neighbors' availability
        double sign_to_use;

        if (mu_m_proj < mu_p_proj)
        {
          if (num_neg >= P4EST_DIM-1 && face_m_area_max > 0.01) sign_to_use = -1;
          else                        sign_to_use =  1;
        } else {
          if (num_pos >= P4EST_DIM-1 && face_p_area_max > 0.01) sign_to_use =  1;
          else                        sign_to_use = -1;
        }

        // linear system
        char num_constraints = num_neighbors_cube_;

//        double gamma = 3.*log(10.);
        double gamma = 0;

        std::vector<double> weight(num_constraints, 0);
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

              if (neighbors_exist[idx])
              {
                if (ii_phi_eff_p[neighbors[idx]]*sign_to_use > 0 || idx == nn_000)
                {
                  weight[idx] = exp(-gamma*(SQR((x_C+col_2nd[idx]-xyz_pr[0])/dx_min_) +
                  #ifdef P4_TO_P8
                                    SQR((z_C+col_4th[idx]-xyz_pr[2])/dz_min_) +
    #endif
                      SQR((y_C+col_3rd[idx]-xyz_pr[1])/dy_min_)));
                  if (idx == nn_000) weight[idx] = 10;
                }
              }
            }


        // assemble and invert matrix
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

        for (unsigned short nei = 0; nei < num_constraints; ++nei)
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
        if (!inv_mat4(A, A_inv)) throw;
#else
        if (!inv_mat3(A, A_inv)) throw;
#endif

        // compute Taylor expansion coefficients
        std::vector<double> coeff_const_term(num_constraints, 0);
        std::vector<double> coeff_x_term    (num_constraints, 0);
        std::vector<double> coeff_y_term    (num_constraints, 0);
#ifdef P4_TO_P8
        std::vector<double> coeff_z_term    (num_constraints, 0);
#endif

        for (unsigned short nei = 0; nei < num_constraints; ++nei)
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

//        xyz_pr[0] = MIN(xyz_pr[0], fv_xmax); xyz_pr[0] = MAX(xyz_pr[0], fv_xmin);
//        xyz_pr[1] = MIN(xyz_pr[1], fv_ymax); xyz_pr[1] = MAX(xyz_pr[1], fv_ymin);
//  #ifdef P4_TO_P8
//        xyz_pr[2] = MIN(xyz_pr[2], fv_zmax); xyz_pr[2] = MAX(xyz_pr[2], fv_zmin);
//  #endif


        double sign = ii_phi_eff_p[n] < 0 ? 1 : -1;

        rhs_add_ptr[n] = sign*jump_value_->value(xyz_pr);
        w_ghosts[nn_000] = 1;

        double scaling = 1;

        if (sign_to_use < 0)
        {
          for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei)
          {
            if (nei != nn_000)
            w_ghosts[nei] += sign*dist*(mu_m_proj/mu_p_proj - 1.)*(coeff_x_term[nei]*normal[0]
    #ifdef P4_TO_P8
                + coeff_z_term[nei]*normal[2]
    #endif
                + coeff_y_term[nei]*normal[1]);
          }
          rhs_add_ptr[n] += sign*dist*flux_proj/mu_p_proj;

          double coeff_000 = sign*dist*(mu_m_proj/mu_p_proj - 1.)*(coeff_x_term[nn_000]*normal[0]
    #ifdef P4_TO_P8
                + coeff_z_term[nn_000]*normal[2]
    #endif
                + coeff_y_term[nn_000]*normal[1]);

          if (ii_phi_eff_p[n] > 0)
          {
            scaling = 1. - coeff_000;
          } else {
            w_ghosts[nn_000] += coeff_000;
          }
        }
        else
        {
          for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei)
          {
            if (nei != nn_000)
            w_ghosts[nei] += sign*dist*(1. - mu_p_proj/mu_m_proj)*(coeff_x_term[nei]*normal[0]
    #ifdef P4_TO_P8
                + coeff_z_term[nei]*normal[2]
    #endif
                + coeff_y_term[nei]*normal[1]);
          }
          rhs_add_ptr[n] += sign*dist*flux_proj/mu_m_proj;

          double coeff_000 = sign*dist*(1. - mu_p_proj/mu_m_proj)*(coeff_x_term[nn_000]*normal[0]
    #ifdef P4_TO_P8
                + coeff_z_term[nn_000]*normal[2]
    #endif
                + coeff_y_term[nn_000]*normal[1]);

          if (ii_phi_eff_p[n] < 0)
          {
            scaling = 1. - coeff_000;
          } else {
            w_ghosts[nn_000] += coeff_000;
          }
        }
        for (unsigned short nei = 0; nei < num_neighbors_cube_; ++nei)
        {
          w_ghosts[nei] /= scaling;
        }
        rhs_add_ptr[n] /= scaling;

        // put values into matrix
        for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
          if (neighbors_exist[i] && fabs(w_ghosts[i]) > EPS)
          {
            PetscInt node_nei_g = petsc_gloidx_[neighbors[i]];
            if (w_ghosts[i] != w_ghosts[i]) throw std::domain_error("Matrix element is nan\n");
            ent.n = node_nei_g;
            ent.val = w_ghosts[i];
            row_C->push_back(ent);
            ( neighbors[i] < num_owned_local ) ? C_d_nnz[n]++ : C_o_nnz[n]++;
          }
      }

    }//*/

  }

  // restore pointers
  ierr = VecRestoreArray(mask_, &mask_p); CHKERRXX(ierr);

  if (setup_matrix)
  {
    ierr = VecGhostUpdateBegin(mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_, MAX_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(mask_, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (mask_, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
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
    for(int n=0; n<num_owned_local; ++n)
    {
      PetscInt global_n_idx = petsc_gloidx_[n];
      std::vector<mat_entry_t> * row = matrix_entries[n];
      for(unsigned int m = 0; m < row->size(); ++m)
      {
        ierr = MatSetValue(A_, global_n_idx, row->at(m).n, row->at(m).val, ADD_VALUES); CHKERRXX(ierr);
      }
//      delete row;
    }
#endif

    /* assemble the matrix */
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A_, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

    if (interface_present)
    {
#ifdef DO_NOT_PREALLOCATE
      Mat B, C;
      /* set up the matrix */
      ierr = MatCreate(p4est_->mpicomm, &B); CHKERRXX(ierr);
      ierr = MatCreate(p4est_->mpicomm, &C); CHKERRXX(ierr);
      ierr = MatSetType(B, MATAIJ); CHKERRXX(ierr);
      ierr = MatSetType(C, MATAIJ); CHKERRXX(ierr);
      ierr = MatSetSizes(B, num_owned_local , num_owned_local, num_owned_global, num_owned_global); CHKERRXX(ierr);
      ierr = MatSetSizes(C, num_owned_local , num_owned_local, num_owned_global, num_owned_global); CHKERRXX(ierr);
      ierr = MatSetFromOptions(B); CHKERRXX(ierr);
      ierr = MatSetFromOptions(C); CHKERRXX(ierr);

      /* allocate the matrix */
      ierr = MatSeqAIJSetPreallocation(B, 0, (const PetscInt*)&B_d_nnz[0]); CHKERRXX(ierr);
      ierr = MatSeqAIJSetPreallocation(C, 0, (const PetscInt*)&C_d_nnz[0]); CHKERRXX(ierr);
      ierr = MatMPIAIJSetPreallocation(B, 0, (const PetscInt*)&B_d_nnz[0], 0, (const PetscInt*)&B_o_nnz[0]); CHKERRXX(ierr);
      ierr = MatMPIAIJSetPreallocation(C, 0, (const PetscInt*)&C_d_nnz[0], 0, (const PetscInt*)&C_o_nnz[0]); CHKERRXX(ierr);

      /* fill the matrix with the values */
      for(int n=0; n<num_owned_local; ++n)
      {
        PetscInt global_n_idx = petsc_gloidx_[n];
        std::vector<mat_entry_t> * B_row = B_matrix_entries[n];
        std::vector<mat_entry_t> * C_row = C_matrix_entries[n];
        for(unsigned int m = 0; m < B_row->size(); ++m)
        {
          ierr = MatSetValue(B, global_n_idx, B_row->at(m).n, B_row->at(m).val, ADD_VALUES); CHKERRXX(ierr);
        }
        for(unsigned int m = 0; m < C_row->size(); ++m)
        {
          ierr = MatSetValue(C, global_n_idx, C_row->at(m).n, C_row->at(m).val, ADD_VALUES); CHKERRXX(ierr);
        }
        delete B_row;
        delete C_row;
      }
#endif
      ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
      ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
      ierr = MatAssemblyEnd  (B, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
      ierr = MatAssemblyEnd  (C, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

      Mat BC;
//      ierr = MatCreate(p4est_->mpicomm, &BC); CHKERRXX(ierr);

      ierr = MatMatMult(B, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BC); CHKERRXX(ierr);
//      ierr = MatMatMultSymbolic(B, C, PETSC_DEFAULT, &BC); CHKERRXX(ierr);
//      ierr = MatMatMultNumeric(B, C, BC); CHKERRXX(ierr);

      ierr = MatAXPY(A_, 1., BC, DIFFERENT_NONZERO_PATTERN); CHKERRXX(ierr);
//      ierr = MatAXPY(A_, 1., B, DIFFERENT_NONZERO_PATTERN); CHKERRXX(ierr);

      invert_phi(nodes_, rhs_add);
      ierr = MatMultAdd(B, rhs_add, rhs_, rhs_); CHKERRXX(ierr);

      ierr = MatDestroy(B); CHKERRXX(ierr);
      ierr = MatDestroy(C); CHKERRXX(ierr);
      ierr = MatDestroy(BC); CHKERRXX(ierr);
    }
  }

  for(int n=0; n<num_owned_local; ++n)
  {
    delete matrix_entries[n];
    delete B_matrix_entries[n];
    delete C_matrix_entries[n];
  }

//  if (setup_matrix || setup_rhs)
//  {
//    Vec diagonal;

//    ierr = VecDuplicate(rhs_, &diagonal); CHKERRXX(ierr);
//    ierr = MatGetDiagonal(A_, diagonal); CHKERRXX(ierr);

//    double *diagonal_ptr;
//    ierr = VecGetArray(diagonal, &diagonal_ptr); CHKERRXX(ierr);
//    foreach_local_node(n, nodes_)
//    {
//      diagonal_ptr[n] = 1./diagonal_ptr[n];
//    }
//    ierr = VecRestoreArray(diagonal, &diagonal_ptr); CHKERRXX(ierr);

//    if (setup_matrix) { ierr = MatDiagonalScale(A_, diagonal, NULL); CHKERRXX(ierr); }
//    if (setup_rhs)    { ierr = VecPointwiseMult(rhs_, rhs_, diagonal); CHKERRXX(ierr); }
//    ierr = VecDestroy(diagonal); CHKERRXX(ierr);
//  }


//  if (use_sc_scheme_)
  {
//    ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(areas_, &areas_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(areas_m_, &areas_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(areas_p_, &areas_p_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);
  }

  for (unsigned short i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  for (unsigned short i = 0; i < num_interfaces_; ++i)
  {
    ierr = VecRestoreArray(phi_x_->at(i), &phi_x_ptr[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y_->at(i), &phi_y_ptr[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z_->at(i), &phi_z_ptr[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ii_.phi_eff, &ii_phi_eff_p); CHKERRXX(ierr);

  if (variable_mu_) {
    ierr = VecRestoreArray(mue_m_,    &mue_m_ptr   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_,    &mue_p_ptr   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_m_xx_, &mue_m_xx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_xx_, &mue_p_xx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_m_yy_, &mue_m_yy_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_yy_, &mue_p_yy_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_m_zz_, &mue_m_zz_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_p_zz_, &mue_p_zz_ptr); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(diag_add_m_, &diag_add_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(diag_add_p_, &diag_add_p_ptr); CHKERRXX(ierr);

  if (exact_ != NULL) { ierr = VecRestoreArray(exact_, &exact_ptr); CHKERRXX(ierr); }

  if (setup_rhs)
  {
    ierr = VecRestoreArray(rhs_, &rhs_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_add, &rhs_add_ptr); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_add); CHKERRXX(ierr);
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


void my_p4est_poisson_nodes_mls_sc_t::compute_volumes()
{
  volumes_owned_ = true;
  volumes_computed_ = true;

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // get access to LSFs
  //---------------------------------------------------------------------

  // domain
  std::vector<double *> phi_p (num_interfaces_, NULL);
  std::vector<double *> phi_xx_p (num_interfaces_, NULL);
  std::vector<double *> phi_yy_p (num_interfaces_, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p (num_interfaces_, NULL);
#endif

  for (unsigned short i = 0; i < num_interfaces_; i++)
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

  // immersed interface
  std::vector<double *> ii_phi_p   (ii_.num_interfaces, NULL);
  std::vector<double *> ii_phi_xx_p(ii_.num_interfaces, NULL);
  std::vector<double *> ii_phi_yy_p(ii_.num_interfaces, NULL);
#ifdef P4_TO_P8
  std::vector<double *> ii_phi_zz_p(ii_.num_interfaces, NULL);
#endif

  for (unsigned short i = 0; i < ii_.num_interfaces; i++)
  {
    ierr = VecGetArray(ii_.phi->at(i),    &ii_phi_p[i]);    CHKERRXX(ierr);
    ierr = VecGetArray(ii_.phi_xx->at(i), &ii_phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(ii_.phi_yy->at(i), &ii_phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(ii_.phi_zz->at(i), &ii_phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *ii_phi_eff_p;
  ierr = VecGetArray(ii_.phi_eff, &ii_phi_eff_p); CHKERRXX(ierr);

//  double *volumes_p;
//  if (volumes_ != NULL) { ierr = VecDestroy(volumes_); CHKERRXX(ierr); }
//  ierr = VecDuplicate(phi_->at(0), &volumes_); CHKERRXX(ierr);
//  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

  double *areas_ptr;
  double *areas_m_ptr;
  double *areas_p_ptr;

  if (areas_   != NULL) { ierr = VecDestroy(areas_);   CHKERRXX(ierr); }
  if (areas_m_ != NULL) { ierr = VecDestroy(areas_m_); CHKERRXX(ierr); }
  if (areas_p_ != NULL) { ierr = VecDestroy(areas_p_); CHKERRXX(ierr); }

  ierr = VecDuplicate(phi_eff_, &areas_);   CHKERRXX(ierr);
  ierr = VecDuplicate(phi_eff_, &areas_m_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_eff_, &areas_p_); CHKERRXX(ierr);

  ierr = VecGetArray(areas_,   &areas_ptr);   CHKERRXX(ierr);
  ierr = VecGetArray(areas_m_, &areas_m_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(areas_p_, &areas_p_ptr); CHKERRXX(ierr);

  double *node_type_p;
  if (node_type_ != NULL) { ierr = VecDestroy(node_type_); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi_eff_, &node_type_); CHKERRXX(ierr);
  ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);

  // data for refined cells
  unsigned short fv_size_x = 0;
  unsigned short fv_size_y = 0;
#ifdef P4_TO_P8
  unsigned short fv_size_z = 0;
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

  bool neighbors_exist[num_neighbors_cube_];
  p4est_locidx_t neighbors[num_neighbors_cube_];

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
    double ii_phi_eff_000 = ii_phi_eff_p[n];

//    volumes_p[n] = 0;
    areas_ptr[n]   = 0;
    areas_m_ptr[n] = 0;
    areas_p_ptr[n] = 0;

    if(is_node_Wall(p4est_, ni) && phi_eff_000 < 0.)
    {
      if(bc_wall_type_->value(xyz_C) == DIRICHLET)
      {
//        volumes_p[n] = 1;
        areas_ptr[n]   = 1;
        areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
        areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;
        continue;
      }

      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
      if(neumann_wall_first_order_ && bc_wall_type_->value(xyz_C) == NEUMANN)
      {
        if (phi_eff_000 < diag_min_ || num_interfaces_ == 0)
        {
//          volumes_p[n] = 1;
          areas_ptr[n]   = 1;
          areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
          areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;
        }
        continue;
      }
    }

    {
      interp_local.initialize(n);
      phi_interp_local.copy_init(interp_local);


      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_ngbd_crossed_dirichlet = false;
      bool is_ngbd_crossed_neumann   = false;
      bool is_ngbd_crossed_immersed  = false;

      if (fabs(phi_eff_000) < lip_*diag_min_)
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
        if (cube_refinement_ == 0)
        {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }

        // sample level-set function at cube nodes and check if crossed
        get_all_neighbors(n, neighbors, neighbors_exist);
        for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
          bool is_one_positive = false;
          bool is_one_negative = false;

          for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
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

      if (fabs(ii_phi_eff_000) < lip_*diag_min_)
      {
        get_all_neighbors(n, neighbors, neighbors_exist);

        // sample level-set function at nodes of the extended cube and check if crossed
        for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
        {
          bool is_one_positive = false;
          bool is_one_negative = false;

          for (short i = 0; i < num_neighbors_cube_; ++i)
            if (neighbors_exist[i])
            {
              is_one_positive = is_one_positive || ii_phi_p[phi_idx][neighbors[i]] > 0;
              is_one_negative = is_one_negative || ii_phi_p[phi_idx][neighbors[i]] < 0;
            }

          if (is_one_negative && is_one_positive) is_ngbd_crossed_immersed = true;
        }
      }

      if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet) { throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment"); }
      else if (is_ngbd_crossed_neumann)                         { discretization_scheme_ = FVM; }
      else if (is_ngbd_crossed_immersed)                        { discretization_scheme_ = JUMP; }
      else                                                      { discretization_scheme_ = FDM; }

      if (discretization_scheme_ == FDM)
      {
        //---------------------------------------------------------------------
        // interface boundary
        //---------------------------------------------------------------------
        if (ABS(phi_eff_000) < EPS)
        {
//          volumes_p[n] = 1;
          areas_ptr[n]   = 1;
          areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
          areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;
          node_type_p[n] = 0;
          continue;
        }

        // far away from the interface
        if (phi_eff_000 > 0.)
        {
//          volumes_p[n] = 0;
          areas_ptr[n]   = 0;
          areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
          areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;
          node_type_p[n] = 0;
          continue;
        }

        // if far away from the interface or close to it but with dirichlet
        // then finite difference method
        if (phi_eff_000 < 0.)
        {
//          volumes_p[n] = 1;
          areas_ptr[n]   = 1;
          areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
          areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;
          node_type_p[n] = 63.;
          continue;
        }

      }
      else if (discretization_scheme_ == FVM)
      {
        areas_ptr[n] = 0;
        areas_m_ptr[n] = ii_phi_eff_000 < 0 ? 1 : 0;
        areas_p_ptr[n] = ii_phi_eff_000 < 0 ? 0 : 1;

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
        unsigned int points_total = x_grid.size();

        std::vector<double> phi_cube(num_interfaces_*points_total, -1);

        // compute values of level-set functions at needed points
        for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
        {
          if (phi_cf_ == NULL)
          {
#ifdef P4_TO_P8
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], interp_method_);
#else
            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], interp_method_);
#endif
          }
          for (unsigned int i = 0; i < points_total; ++i)
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

//        // compute cut-cell volume
//        double volume_cut_cell = 0.;

//        for (unsigned int i = 0; i < cube_dom_w.size(); ++i)
//        {
//          volume_cut_cell += cube_dom_w[i];
//        }

//        volumes_p[n] = volume_cut_cell/full_cell_volume;

        // check for a hanging volume
//        bool is_one_positive = false;
//        bool is_one_negative = false;

        double type = 0;

        for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
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

          double cut_area = 0;
          for (unsigned int i = 0; i < cube_dom_w.size(); ++i) cut_area += cube_dom_w[i];

          areas_ptr[n] = MAX(areas_ptr[n], cut_area/face_area_scalling_);
        }

        node_type_p[n] = type;
      }
      else if (discretization_scheme_ == JUMP)
      {
        areas_ptr[n] = 0;
        areas_m_ptr[n] = 0;
        areas_p_ptr[n] = 0;

        // reconstruct domain and compute geometric quantities
  #ifdef P4_TO_P8
        cube3_mls_t cube;
  #else
        cube2_mls_t cube;
  #endif

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

        std::vector<double> phi_cube(ii_.num_interfaces*points_total,-1);

        // sample values of level-set functions at those points
        for (unsigned short phi_idx = 0; phi_idx < ii_.num_interfaces; ++phi_idx)
        {
  #ifdef P4_TO_P8
          phi_interp_local.set_input(ii_phi_p[phi_idx], ii_phi_xx_p[phi_idx], ii_phi_yy_p[phi_idx], ii_phi_zz_p[phi_idx], interp_method_);
  #else
          phi_interp_local.set_input(ii_phi_p[phi_idx], ii_phi_xx_p[phi_idx], ii_phi_yy_p[phi_idx], interp_method_);
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
        cube.reconstruct(phi_cube, *ii_.action, *ii_.color);

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
        double face_area_full[] = { full_sx, full_sx, full_sy, full_sy, full_sz, full_sz };
  #else
        double face_area_full[] = { full_sx, full_sx, full_sy, full_sy };
  #endif

        double face_m_area = 0;
        double face_p_area = 0;

        for (int dir_idx = 0; dir_idx < P4EST_FACES; ++dir_idx)
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
            face_m_area = 0;
            for (unsigned int i = 0; i < cube_dir_w.size(); ++i)
            {
              face_m_area += cube_dir_w[i];
            }
          }

          face_p_area = face_area_full[dir_idx] - face_m_area;

          areas_m_ptr[n] = MAX(areas_m_ptr[n], face_m_area/face_area_scalling_);
          areas_p_ptr[n] = MAX(areas_p_ptr[n], face_p_area/face_area_scalling_);
        }
      }
    }
  }

  // restore pointers
//  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(areas_, &areas_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(areas_m_, &areas_m_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(areas_p_, &areas_p_ptr); CHKERRXX(ierr);
//  ierr = VecGhostUpdateBegin(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(volumes_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(areas_,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (areas_,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(areas_m_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (areas_m_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(areas_p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (areas_p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(node_type_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (node_type_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(ii_.phi_eff, &ii_phi_eff_p); CHKERRXX(ierr);

  for (unsigned short i = 0; i < num_interfaces_; i++)
  {
    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  for (unsigned short i = 0; i < ii_.num_interfaces; i++)
  {
    ierr = VecRestoreArray(ii_.phi->at(i),    &ii_phi_p[i]);    CHKERRXX(ierr);
    ierr = VecRestoreArray(ii_.phi_xx->at(i), &ii_phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(ii_.phi_yy->at(i), &ii_phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(ii_.phi_zz->at(i), &ii_phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

}

//#ifdef P4_TO_P8
//void my_p4est_poisson_nodes_mls_sc_t::reconstruct_domain(std::vector<cube3_mls_t> &cubes)
//#else
//void my_p4est_poisson_nodes_mls_sc_t::reconstruct_domain(std::vector<cube2_mls_t> &cubes)
//#endif
//{
//  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

//  //---------------------------------------------------------------------
//  // get access to LSFs
//  //---------------------------------------------------------------------
//  std::vector<double *> phi_p (num_interfaces_, NULL);
//  std::vector<double *> phi_xx_p (num_interfaces_, NULL);
//  std::vector<double *> phi_yy_p (num_interfaces_, NULL);
//#ifdef P4_TO_P8
//  std::vector<double *> phi_zz_p (num_interfaces_, NULL);
//#endif

//  for (unsigned short i = 0; i < num_interfaces_; i++)
//  {
//    ierr = VecGetArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
//    ierr = VecGetArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
//    ierr = VecGetArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecGetArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
//#endif
//  }

//  double *phi_eff_p;
//  ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

////  double *volumes_p;
////  ierr = VecGetArray(volumes_, &volumes_p); CHKERRXX(ierr);

//  double *node_type_p;
//  ierr = VecGetArray(node_type_, &node_type_p); CHKERRXX(ierr);

//  // data for refined cells
//  int fv_size_x = 0;
//  int fv_size_y = 0;
//#ifdef P4_TO_P8
//  int fv_size_z = 0;
//#endif

//  double fv_xmin, fv_xmax;
//  double fv_ymin, fv_ymax;
//#ifdef P4_TO_P8
//  double fv_zmin, fv_zmax;
//#endif

//  double xyz_C[P4EST_DIM];

//  cubes.reserve(nodes_->num_owned_indeps);

//  bool neighbors_exist[num_neighbors_cube_];
//  p4est_locidx_t neighbors[num_neighbors_cube_];

//  // interpolations
//  my_p4est_interpolation_nodes_local_t interp_local(node_neighbors_);
//  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);

//  for(p4est_locidx_t n=0; n<nodes_->num_owned_indeps; n++) // loop over nodes
//  {
//    // tree information
//    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);

//    //---------------------------------------------------------------------
//    // Information at neighboring nodes
//    //---------------------------------------------------------------------
//    node_xyz_fr_n(n, p4est_, nodes_, xyz_C);
//    double x_C  = xyz_C[0];
//    double y_C  = xyz_C[1];
//#ifdef P4_TO_P8
//    double z_C  = xyz_C[2];
//#endif

//    double phi_eff_000 = phi_eff_p[n];

////    if(is_node_Wall(p4est_, ni))
////    {
////#ifdef P4_TO_P8
////      if((*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == DIRICHLET)
////#else
////      if((*bc_wall_type_)(xyz_C[0], xyz_C[1]) == DIRICHLET)
////#endif
////      {
////        continue;
////      }

////      // In case if you want first order neumann at walls. Why is it still a thing anyway? Daniil.
////      if(neumann_wall_first_order_ &&
////   #ifdef P4_TO_P8
////         (*bc_wall_type_)(xyz_C[0], xyz_C[1], xyz_C[2]) == NEUMANN)
////   #else
////         (*bc_wall_type_)(xyz_C[0], xyz_C[1]) == NEUMANN)
////   #endif
////      {
////        continue;
////      }

////    }

//    {
//      interp_local.initialize(n);
//      phi_interp_local.copy_init(interp_local);

//      //---------------------------------------------------------------------
//      // check if finite volume is crossed
//      //---------------------------------------------------------------------
//      bool is_ngbd_crossed_dirichlet = false;
//      bool is_ngbd_crossed_neumann   = false;

//      if (fabs(phi_eff_000) < 4.*diag_min_)
//      {
//        get_all_neighbors(n, neighbors, neighbors_exist);

//        // sample level-set function at cube nodes and check if crossed
//        for (unsigned short phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
//        {
//          bool is_one_positive = false;
//          bool is_one_negative = false;

//          for (unsigned short i = 0; i < num_neighbors_cube_; ++i)
//            if (neighbors_exist[i])
//            {
//              is_one_positive = is_one_positive || phi_p[phi_idx][neighbors[i]] > 0;
//              is_one_negative = is_one_negative || phi_p[phi_idx][neighbors[i]] < 0;
//            }

//          if (is_one_negative && is_one_positive)
//          {
//            if (bc_interface_type_->at(phi_idx) == DIRICHLET) is_ngbd_crossed_dirichlet = true;
//            if (bc_interface_type_->at(phi_idx) == NEUMANN)   is_ngbd_crossed_neumann   = true;
//            if (bc_interface_type_->at(phi_idx) == ROBIN)     is_ngbd_crossed_neumann   = true;
//          }
//        }
//      }

//      if (is_ngbd_crossed_neumann && is_ngbd_crossed_dirichlet)
//        throw std::domain_error("[CASL_ERROR]: No crossing Dirichlet and Neumann at the moment");
//      else if (is_ngbd_crossed_neumann)
//        discretization_scheme_ = FVM;
//      else
//        discretization_scheme_ = FDM;



//      if (discretization_scheme_ == FVM)
//      {
//        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
//          if (neighbors_exist[idx])
//            neighbors_exist[idx] = neighbors_exist[idx] && (volumes_p[neighbors[idx]] > domain_rel_thresh_);

//        // check for hanging neighbors
//        bool hanging_neighbor[num_neighbors_cube_];

//        int network[num_neighbors_cube_];

//        for (unsigned short idx = 0; idx < num_neighbors_cube_; ++idx)
//          network[idx] = neighbors_exist[idx] ? (int) node_type_p[neighbors[idx]] : 0;

//        find_hanging_cells(network, hanging_neighbor);

//        bool expand[2*P4EST_DIM];
//        for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
//          expand[dir] = false;


//  #ifdef P4_TO_P8
//        for (unsigned short k = 0; k < 3; ++k)
//  #endif
//          for (unsigned short j = 0; j < 3; ++j)
//            for (unsigned short i = 0; i < 3; ++i)
//            {
//  #ifdef P4_TO_P8
//              unsigned short idx = 9*k + 3*j +i;
//  #else
//              unsigned short idx = 3*j + i;
//  #endif
//              if (neighbors_exist[idx])
//                if (hanging_neighbor[idx] && volumes_p[neighbors[idx]] < 0.001)
//                {
//                  if (i == 0) expand[dir::f_m00] = true;
//                  if (i == 2) expand[dir::f_p00] = true;
//                  if (j == 0) expand[dir::f_0m0] = true;
//                  if (j == 2) expand[dir::f_0p0] = true;
//  #ifdef P4_TO_P8
//                  if (k == 0) expand[dir::f_00m] = true;
//                  if (k == 2) expand[dir::f_00p] = true;
//  #endif
//                }
//            }

//        bool attempt_to_expand = false;

//        for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
//          attempt_to_expand = attempt_to_expand || expand[dir];

//#ifdef P4_TO_P8
//        cubes.push_back(cube3_mls_t());
//#else
//        cubes.push_back(cube2_mls_t());
//#endif
//        attempt_to_expand = false;
//        while (1)
//        {

//          // determine dimensions of cube
//          fv_size_x = 0;
//          fv_size_y = 0;
//    #ifdef P4_TO_P8
//          fv_size_z = 0;
//    #endif

//          if (!is_node_xmWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmin = x_C - .5*dx_min_; } else { fv_xmin = x_C; }
//          if (!is_node_xpWall(p4est_, ni)) { fv_size_x += cube_refinement_; fv_xmax = x_C + .5*dx_min_; } else { fv_xmax = x_C; }
//          if (!is_node_ymWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymin = y_C - .5*dy_min_; } else { fv_ymin = y_C; }
//          if (!is_node_ypWall(p4est_, ni)) { fv_size_y += cube_refinement_; fv_ymax = y_C + .5*dy_min_; } else { fv_ymax = y_C; }
//  #ifdef P4_TO_P8
//          if (!is_node_zmWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmin = z_C - .5*dz_min_; } else { fv_zmin = z_C; }
//          if (!is_node_zpWall(p4est_, ni)) { fv_size_z += cube_refinement_; fv_zmax = z_C + .5*dz_min_; } else { fv_zmax = z_C; }
//  #endif

//          if (attempt_to_expand)
//          {
//            ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment...\n");
//            if (expand[dir::f_m00]) { fv_size_x += cube_refinement_; fv_xmin -= 0.5*dx_min_; }
//            if (expand[dir::f_p00]) { fv_size_x += cube_refinement_; fv_xmax += 0.5*dx_min_; }
//            if (expand[dir::f_0m0]) { fv_size_y += cube_refinement_; fv_ymin -= 0.5*dy_min_; }
//            if (expand[dir::f_0p0]) { fv_size_y += cube_refinement_; fv_ymax += 0.5*dy_min_; }
//  #ifdef P4_TO_P8
//            if (expand[dir::f_00m]) { fv_size_z += cube_refinement_; fv_zmin -= 0.5*dz_min_; }
//            if (expand[dir::f_00p]) { fv_size_z += cube_refinement_; fv_zmax += 0.5*dz_min_; }
//  #endif
//          }

//          if (cube_refinement_ == 0)
//          {
//            fv_size_x = 1;
//            fv_size_y = 1;
//  #ifdef P4_TO_P8
//            fv_size_z = 1;
//  #endif
//          }

//          // Reconstruct geometry
//  #ifdef P4_TO_P8
//          double cube_xyz_min[] = { fv_xmin, fv_ymin, fv_zmin };
//          double cube_xyz_max[] = { fv_xmax, fv_ymax, fv_zmax };
//          int  cube_mnk[] = { fv_size_x, fv_size_y, fv_size_z };
//  #else
//          double cube_xyz_min[] = { fv_xmin, fv_ymin };
//          double cube_xyz_max[] = { fv_xmax, fv_ymax };
//          int  cube_mnk[] = { fv_size_x, fv_size_y };
//  #endif

//          cubes.back().initialize(cube_xyz_min, cube_xyz_max, cube_mnk, integration_order_);

//          // get points at which values of level-set functions are needed
//          std::vector<double> x_grid; cubes.back().get_x_coord(x_grid);
//          std::vector<double> y_grid; cubes.back().get_y_coord(y_grid);
//  #ifdef P4_TO_P8
//          std::vector<double> z_grid; cubes.back().get_z_coord(z_grid);
//  #endif
//          unsigned int points_total = x_grid.size();

//          std::vector<double> phi_cube(num_interfaces_*points_total,-1);

//          // compute values of level-set functions at needed points
//          for (unsigned int phi_idx = 0; phi_idx < num_interfaces_; ++phi_idx)
//          {
//  #ifdef P4_TO_P8
//            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], phi_zz_p[phi_idx], interp_method_);
//  #else
//            phi_interp_local.set_input(phi_p[phi_idx], phi_xx_p[phi_idx], phi_yy_p[phi_idx], interp_method_);
//  #endif
//            for (unsigned int i = 0; i < points_total; ++i)
//            {
//  #ifdef P4_TO_P8
//              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i], z_grid[i]);
//  #else
//              phi_cube[phi_idx*points_total + i] = phi_interp_local(x_grid[i], y_grid[i]);
//  #endif
//            }
//          }

//          // reconstruct geometry
//          cubes.back().reconstruct(phi_cube, *action_, *color_);

//          if (attempt_to_expand)
//          {
//            std::vector<double> W, X, Y, Z;
//            // check if the attempt to expand was successful
//            for (unsigned short dir = 0; dir < P4EST_FACES; ++dir)
//            {
//              if (expand[dir])
//              {
//                W.clear();
//                X.clear();
//                Y.clear();
//                Z.clear();
//  #ifdef P4_TO_P8
//                cubes.back().quadrature_in_dir(dir, W, X, Y, Z);
//  #else
//                cubes.back().quadrature_in_dir(dir, W, X, Y);
//  #endif
//                if (W.size() != 0) attempt_to_expand = false;
//              }
//            }

//            if (attempt_to_expand)
//            {
//              ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment... Success!\n");

//              char im = expand[dir::f_m00] ? 0 : 1;
//              char ip = expand[dir::f_p00] ? 2 : 1;
//              char jm = expand[dir::f_0m0] ? 0 : 1;
//              char jp = expand[dir::f_0p0] ? 2 : 1;
//  #ifdef P4_TO_P8
//              char km = expand[dir::f_00m] ? 0 : 1;
//              char kp = expand[dir::f_00p] ? 2 : 1;
//  #endif

//  #ifdef P4_TO_P8
//              for (char k = km; k <= kp; ++k)
//  #endif
//                for (char j = jm; j <= jp; ++j)
//                  for (char i = im; i <= ip; ++i)
//                  {
//  #ifdef P4_TO_P8
//                    if (i != 1 && j != 1 && k != 1) neighbors_exist[9*k + 3*j + i] = false;
//  #else
//                    if (i != 1 && j != 1)           neighbors_exist[3*j + i]       = false;
//  #endif
//                  }
//              break;
//            } else {
//              ierr = PetscPrintf(p4est_->mpicomm, "Attempting hanging neighbors attachment... Failure!\n");
//            }

//          } else {
//            break;
//          }
//        }

//      }
//    }
//  }

//  // restore pointers

//  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);
//  for (unsigned int i = 0; i < num_interfaces_; i++)
//  {
//    ierr = VecRestoreArray(phi_->at(i), &phi_p[i]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi_xx_->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi_yy_->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//    ierr = VecRestoreArray(phi_zz_->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
//#endif
//  }

//  ierr = VecRestoreArray(volumes_, &volumes_p); CHKERRXX(ierr);

//  ierr = VecRestoreArray(node_type_, &node_type_p); CHKERRXX(ierr);

//  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_sc_compute_volumes, 0, 0, 0, 0); CHKERRXX(ierr);

//}


void my_p4est_poisson_nodes_mls_sc_t::find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr, double normal[])
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

  if (normal != NULL)
  {
    normal[0] = phi_x;
    normal[1] = phi_y;
#ifdef P4_TO_P8
    normal[2] = phi_z;
#endif
  }
}

void my_p4est_poisson_nodes_mls_sc_t::compute_normal(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[])
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

bool my_p4est_poisson_nodes_mls_sc_t::inv_mat2(double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];

  if (det == 0) return false;

  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;

  return true;
}

bool my_p4est_poisson_nodes_mls_sc_t::inv_mat3(double *in, double *out)
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

bool my_p4est_poisson_nodes_mls_sc_t::inv_mat4(const double m[16], double invOut[16])
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
  for (int i = 0; i < num_neighbors_cube_; i++) neighbor_exists[i] = true;

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
double my_p4est_poisson_nodes_mls_sc_t::compute_weights_through_face(double A, double B, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face)
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

//      inv_mat3(matA, matA_inv);

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
double my_p4est_poisson_nodes_mls_sc_t::compute_weights_through_face(double A, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face)
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

void my_p4est_poisson_nodes_mls_sc_t::find_hanging_cells(int *network, bool *hanging_cells)
{
  bool connection_matrix[num_neighbors_cube_][num_neighbors_cube_];
  bool visited[num_neighbors_cube_];

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

        for (int nei_idx = 0; nei_idx < num_neighbors_cube_; ++nei_idx)
          connection_matrix[idx][nei_idx] = false;

        hanging_cells[idx] = true;
        visited[idx] = false;

        int type = network[idx];

        bool connected;

        short I, J;
#ifdef P4_TO_P8
        short K;
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
  for (int idx = 0; idx < num_neighbors_cube_; ++idx)
  {
    // if a node connected to outside
    if (hanging_cells[idx] || visited[idx] || idx == nn_000) continue;

    // then initiate a queue
    std::vector<int> queue;
    queue.push_back(idx);

    // loop through nodes in the queue
    for (unsigned int i = 0; i < queue.size(); ++i)
    {
      int current_idx = queue[i];

      // for every node check which other nodes it's connected to
      for (int nei_idx = 0; nei_idx < num_neighbors_cube_; ++nei_idx)
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

//        for (int nei_idx = 0; nei_idx < num_neighbors_cube_; ++nei_idx)
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
