#ifndef MY_P4EST_POISSON_NODES_MLS_SC_H
#define MY_P4EST_POISSON_NODES_MLS_SC_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#endif

#include <src/mls_integration/cube3_mls.h>
#include <src/mls_integration/cube2_mls.h>

#define DO_NOT_PREALLOCATE

#ifdef P4_TO_P8
const int num_neighbors_max_ = 27;
#else
const int num_neighbors_max_ = 9;
#endif

class my_p4est_poisson_nodes_mls_sc_t
{
  int cube_refinement_;

  double phi_perturbation_;

  interpolation_method interp_method_;

  typedef struct
  {
    double val;
    PetscInt n;
  } mat_entry_t;

  enum node_neighbor_t
  {
  #ifdef P4_TO_P8
    // zm plane
    nn_mmm = 0, nn_0mm = 1, nn_pmm = 2,
    nn_m0m = 3, nn_00m = 4, nn_p0m = 5,
    nn_mpm = 6, nn_0pm = 7, nn_ppm = 8,

    // z0 plane
    nn_mm0, nn_0m0, nn_pm0,
    nn_m00, nn_000, nn_p00,
    nn_mp0, nn_0p0, nn_pp0,

    // zp plane
    nn_mmp, nn_0mp, nn_pmp,
    nn_m0p, nn_00p, nn_p0p,
    nn_mpp, nn_0pp, nn_ppp

  #else
    nn_mm0 = 0, nn_0m0, nn_pm0,
    nn_m00, nn_000, nn_p00,
    nn_mp0, nn_0p0, nn_pp0
  #endif
  };

  const my_p4est_node_neighbors_t *node_neighbors_;

  Vec exact_;

  // p4est objects
  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  // PETSc objects
  Mat A_;
  p4est_gloidx_t fixed_value_idx_g_;
  p4est_gloidx_t fixed_value_idx_l_;
  KSP ksp_;
  PetscErrorCode ierr;
  std::vector<PetscInt> global_node_offset_;
  std::vector<PetscInt> petsc_gloidx_;

  // Geometry
#ifdef P4_TO_P8
  std::vector< CF_3 *> *phi_cf_;
#else
  std::vector< CF_2 *> *phi_cf_;
#endif

  std::vector<Vec> *phi_, *phi_xx_, *phi_yy_, *phi_zz_;
  std::vector<int>        *color_;
  std::vector<action_t>   *action_;
  Vec phi_eff_;
  int num_interfaces_;
  bool is_phi_eff_owned_, is_phi_dd_owned_;

  int integration_order_;

  double lip_;
  void set_lip(double lip) { lip_ = lip; }

  std::vector<Vec> *phi_x_;
  std::vector<Vec> *phi_y_;
  std::vector<Vec> *phi_z_;
  bool is_phi_d_owned_;

  double dxyz_m_[P4EST_DIM];
  double dx_min_, dy_min_, d_min_, diag_min_;
#ifdef P4_TO_P8
  double dz_min_;
#endif

  // Equation
  double mu_, diag_add_scalar_;
  Vec diag_add_;
  Vec rhs_;
  Vec mue_;
  Vec mue_xx_, mue_yy_, mue_zz_;

#ifdef P4_TO_P8
  CF_3 *rhs_cf_;
#else
  CF_2 *rhs_cf_;
#endif

  bool variable_mu_;
  bool is_mue_dd_owned_;

  bool volumes_computed_;
  bool volumes_owned_;

  // Some flags
  bool is_matrix_computed_;
  int matrix_has_nullspace_;
  bool use_pointwise_dirichlet_;
  bool new_pc_;
  bool update_ghost_after_solving_;
  bool use_taylor_correction_;
  bool kink_special_treatment_;

  bool use_sc_scheme_;
  int fallback_;

  // Bondary conditions
  bool neumann_wall_first_order_;

#ifdef P4_TO_P8
  const WallBC3D *bc_wall_type_;
  const CF_3     *bc_wall_value_;

  std::vector< BoundaryConditionType > *bc_interface_type_;
  std::vector< CF_3 *> *bc_interface_value_;
  std::vector< CF_3 *> *bc_interface_coeff_;
#else
  const WallBC2D *bc_wall_type_;
  const CF_2     *bc_wall_value_;

  std::vector< BoundaryConditionType > *bc_interface_type_;
  std::vector< CF_2 *> *bc_interface_value_;
  std::vector< CF_2 *> *bc_interface_coeff_;
#endif

  Vec mask_;
  std::vector<double> scalling_;
  bool keep_scalling_;

  Vec volumes_;
  Vec node_type_;

  bool try_remove_hanging_cells_;

  double eps_ifc_, eps_dom_;

  double domain_rel_thresh_;
  double interface_rel_thresh_;

  enum discretization_scheme_t { FDM, FVM } discretization_scheme_;

  void compute_volumes_();
  void compute_phi_eff_();
  void compute_phi_dd_();
  void compute_phi_d_();
  void compute_mue_dd_();

#ifdef P4_TO_P8
  double compute_weights_through_face(double A, double B, bool *neighbors_exists_2d, double *weights_2d, double theta, bool *map_2d);
#endif

  void preallocate_matrix();

  void setup_linear_system(bool setup_matrix, bool setup_rhs);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_poisson_nodes_mls_sc_t& other);
  my_p4est_poisson_nodes_mls_sc_t& operator=(const my_p4est_poisson_nodes_mls_sc_t& other);

  bool find_x_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p);
  bool find_y_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p);
#ifdef P4_TO_P8
  bool find_z_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p);
#endif

  bool find_xy_derivative(bool *neighbors_exist, double *weights, bool *map, p4est_locidx_t *neighbors, double *volumes_p);

public:
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_sc_t();

  inline void set_fallback(int value) { fallback_ = value; }
  inline void set_try_remove_hanging_cells(bool value) { try_remove_hanging_cells_ = value; }

#ifdef P4_TO_P8
  inline void set_phi_cf(std::vector< CF_3 *> &phi_cf) { phi_cf_ = &phi_cf; }
#else
  inline void set_phi_cf(std::vector< CF_2 *> &phi_cf) { phi_cf_ = &phi_cf; }
#endif

  inline void set_exact(Vec exact) { exact_ = exact; }

  // set geometry
  inline void set_geometry(int num_interfaces,
                           std::vector<action_t> *action, std::vector<int> *color,
                           std::vector<Vec> *phi,
                           std::vector<Vec> *phi_xx = NULL,
                           std::vector<Vec> *phi_yy = NULL,
                         #ifdef P4_TO_P8
                           std::vector<Vec> *phi_zz = NULL,
                         #endif
                           Vec phi_eff = NULL,
                           Vec volumes = NULL)
  {
    num_interfaces_ = num_interfaces;
    action_  = action;
    color_   = color;
    phi_     = phi;

    if (phi_xx != NULL &&
    #ifdef P4_TO_P8
        phi_zz != NULL &&
    #endif
        phi_yy != NULL)
    {
      phi_xx_  = phi_xx;
      phi_yy_  = phi_yy;
#ifdef P4_TO_P8
      phi_zz_  = phi_zz;
#endif
      is_phi_dd_owned_ = false;
    } else {
      compute_phi_dd_();
      is_phi_dd_owned_ = true;
    }

    compute_phi_d_();
    is_phi_d_owned_ = true;

    if (phi_eff != NULL)
      phi_eff_ = phi_eff;
    else
      compute_phi_eff_();

    if (volumes == NULL)
      volumes_computed_ = false;
    else
    {
      volumes_computed_ = true;
      volumes_ = volumes;
    }
    volumes_owned_ = false;
//    compute_volumes_();

#ifdef CASL_THROWS
    if (num_interfaces_ > 0)
      if (action_->size() != num_interfaces_ ||
          color_->size()  != num_interfaces_ ||
          phi_->size()    != num_interfaces_ ||
          phi_xx_->size() != num_interfaces_ ||
    #ifdef P4_TO_P8
          phi_zz_->size() != num_interfaces_ ||
    #endif
          phi_yy_->size() != num_interfaces_ )
        throw std::invalid_argument("[CASL_ERROR]: invalid geometry");
#endif

  }


//  void set_geometry(std::vector<Vec> &phi,
//                    std::vector<action_t> &action, std::vector<int> &color,
//                    Vec phi_eff = NULL);


  // set BCs
#ifdef P4_TO_P8
  inline void set_bc_wall_type(WallBC3D &wall_type) { bc_wall_type_ = &wall_type;}
  inline void set_bc_wall_value(CF_3 &wall_value)   { bc_wall_value_ = &wall_value; }
  inline void set_bc_interface_type (std::vector<BoundaryConditionType> &bc_interface_type)  { bc_interface_type_  = &bc_interface_type;  is_matrix_computed_ = false; }
  inline void set_bc_interface_value(std::vector< CF_3 *> &bc_interface_value)               { bc_interface_value_ = &bc_interface_value; is_matrix_computed_ = false; }
  inline void set_bc_interface_coeff(std::vector< CF_3 *> &bc_interface_coeff)               { bc_interface_coeff_ = &bc_interface_coeff; is_matrix_computed_ = false; }
#else
  inline void set_bc_wall_type(const WallBC2D &wall_type) { bc_wall_type_ = &wall_type;}
  inline void set_bc_wall_value(const CF_2 &wall_value)   { bc_wall_value_ = &wall_value; }
  inline void set_bc_interface_type (std::vector<BoundaryConditionType> &bc_interface_type)  { bc_interface_type_  = &bc_interface_type;  is_matrix_computed_ = false; }
  inline void set_bc_interface_value(std::vector< CF_2 *> &bc_interface_value)               { bc_interface_value_ = &bc_interface_value; is_matrix_computed_ = false; }
  inline void set_bc_interface_coeff(std::vector< CF_2 *> &bc_interface_coeff)               { bc_interface_coeff_ = &bc_interface_coeff; is_matrix_computed_ = false; }
#endif

  inline void set_diag_add(double diag_add_scalar) { diag_add_scalar_  = diag_add_scalar; is_matrix_computed_ = false; }
  inline void set_diag_add(Vec diag_add)           { diag_add_         = diag_add;        is_matrix_computed_ = false; }

  inline void set_mu(double mu) { mu_ = mu; variable_mu_ = false; }

#ifdef P4_TO_P8
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL, Vec mue_zz = NULL)
#else
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL)
#endif
  {
    mue_ = mue;

    if (mue_xx != NULL &&
    #ifdef P4_TO_P8
        mue_zz != NULL &&
    #endif
        mue_yy != NULL)
    {
      mue_xx_ = mue_xx;
      mue_yy_ = mue_yy;
#ifdef P4_TO_P8
      mue_zz_ = mue_zz;
#endif
      is_mue_dd_owned_ = false;
    } else {
      compute_mue_dd_();
      is_mue_dd_owned_ = true;
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }


  inline void set_rhs(Vec rhs) { rhs_ = rhs; }
#ifdef P4_TO_P8
  inline void set_rhs(CF_3 &rhs_cf)   { rhs_cf_ = &rhs_cf; }
#else
  inline void set_rhs(CF_2 &rhs_cf)   { rhs_cf_ = &rhs_cf; }
#endif

  inline void set_is_matrix_computed(bool is_matrix_computed) { is_matrix_computed_ = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp_, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace_; }

  inline void set_first_order_neumann_wall(bool value) { neumann_wall_first_order_  = value; }
  inline void set_use_pointwise_dirichlet (bool value) { use_pointwise_dirichlet_   = value; }
  inline void set_use_taylor_correction   (bool value) { use_taylor_correction_     = value; }
  inline void set_keep_scalling           (bool value) { keep_scalling_             = value; }
  inline void set_kink_treatment          (bool value) { kink_special_treatment_    = value; }
  inline void set_use_sc_scheme           (bool value) { use_sc_scheme_             = value; }
  inline void set_integration_order       (int  value) { integration_order_         = value; }

  bool inv_mat2_(double *in, double *out);
  bool inv_mat3_(double *in, double *out);
  bool inv_mat4_(const double m[16], double invOut[16]);

//  void find_projection_(double *phi_p, p4est_locidx_t *neighbors, bool *neighbor_exists, double dxyz_pr[], double &dist_pr);
  void find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr);

  void compute_normal_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[]);

  void get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists);

  void find_hanging_cells(int *network, bool *hanging_cells);

//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCASM);
  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCASM);

  void assemble_matrix(Vec solution);

  inline Vec get_mask() { return mask_; }
  inline Vec get_volumes() { return volumes_; }

  inline void get_phi_dd(std::vector<Vec> **phi_dd)
  {
    phi_dd[0] = phi_xx_;
    phi_dd[1] = phi_yy_;
#ifdef P4_TO_P8
    phi_dd[2] = phi_zz_;
#endif
  }

  inline Vec get_phi_eff() { return phi_eff_; }

  inline Mat get_matrix() { return A_; }

  inline std::vector<double>* get_scalling() { return &scalling_; }

  inline void assemble_rhs_only() { setup_linear_system(false, true); }

#ifdef P4_TO_P8
void reconstruct_domain(std::vector<cube3_mls_t> &cubes);
#else
void reconstruct_domain(std::vector<cube2_mls_t> &cubes);
#endif


  //---------------------------------------------------------------------------------
  // some stuff for pointwise dirichlet
  //---------------------------------------------------------------------------------

  struct interface_point_t
  {
    short dir;
    double dist;
    double value;
    interface_point_t(double dir_, double dist_) {dir = dir_; dist = dist_;}
  };

  std::vector< std::vector<interface_point_t> > pointwise_bc_;

  inline void get_xyz_interface_point(p4est_locidx_t n, short i, double *xyz)
  {
    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

    switch (dir) {
      case 0: xyz[0] -= dist; break;
      case 1: xyz[0] += dist; break;
      case 2: xyz[1] -= dist; break;
      case 3: xyz[1] += dist; break;
#ifdef P4_TO_P8
      case 4: xyz[2] -= dist; break;
      case 5: xyz[2] += dist; break;
#endif
    }
  }

  inline void set_interface_point_value(p4est_locidx_t n, short i, double val)
  {
    pointwise_bc_[n][i].value = val;
  }

  inline double get_interface_point_value(p4est_locidx_t n, short i)
  {
    return pointwise_bc_[n][i].value;
  }

  // linear interpolation of a Vec at an interface point
  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min_; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min_; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min_; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min_; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min_; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min_; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min_; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min_; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min_; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min_; break;
#endif
    }

    return (ptr[n]*(h-dist) + ptr[neigh]*dist)/h;
  }

  // quadratic interpolation of a Vec at an interface point
  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr, double *ptr_dd[P4EST_DIM])
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    short dim =0;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min_; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min_; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min_; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min_; dim = 1; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min_; dim = 2; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min_; dim = 2; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min_; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min_; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min_; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min_; dim = 1; break;
#endif
    }

    double p_dd = .5*(ptr_dd[dim][n] + ptr_dd[dim][neigh]);
    double p0 = ptr[n];
    double p1 = ptr[neigh];

    return .5*(p0+p1) + (p1-p0)*(dist/h-.5) + .5*p_dd*(dist*dist-dist*h);
  }

  // assemble RHS for Poisson equation with jump conditions and continuous and constant mu
//  void assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p);
  void assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in = NULL, Vec rhs_p_in = NULL);
};

#endif // MY_P4EST_POISSON_NODES_MLS_H
