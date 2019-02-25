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

class my_p4est_poisson_nodes_mls_sc_t
{
  typedef struct
  {
    double val;
    PetscInt n;
  } mat_entry_t;

  // p4est objects
  const my_p4est_node_neighbors_t *node_neighbors_;

  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  // grid variables
  double lip_;
  double diag_min_;
  double d_min_;
  double dx_min_;
  double dy_min_;
#ifdef P4_TO_P8
  double dz_min_;
#endif
  double dxyz_m_[P4EST_DIM];

  // PETSc objects
  Mat A_;
  KSP ksp_;
  PetscErrorCode ierr;
  p4est_gloidx_t fixed_value_idx_g_;
  p4est_gloidx_t fixed_value_idx_l_;
  std::vector<PetscInt> global_node_offset_;
  std::vector<PetscInt> petsc_gloidx_;

  // Geometry
  unsigned int num_interfaces_;

  std::vector<Vec>     *phi_;
#ifdef P4_TO_P8
  std::vector< CF_3 *> *phi_cf_;
#else
  std::vector< CF_2 *> *phi_cf_;
#endif

  std::vector<Vec> *phi_xx_;
  std::vector<Vec> *phi_yy_;
#ifdef P4_TO_P8
  std::vector<Vec> *phi_zz_;
#endif

  Vec phi_eff_;

  std::vector<action_t> *action_;
  std::vector<int>      *color_;

  std::vector<Vec> *phi_x_;
  std::vector<Vec> *phi_y_;
#ifdef P4_TO_P8
  std::vector<Vec> *phi_z_;
#endif

  bool is_phi_d_owned_;
  bool is_phi_dd_owned_;
  bool is_phi_eff_owned_;

  // Equation
  Vec   rhs_;
  Vec   rhs_m_;
  Vec   rhs_p_;
#ifdef P4_TO_P8
  CF_3 *rhs_cf_;
#else
  CF_2 *rhs_cf_;
#endif

  Vec    diag_add_m_;
  Vec    diag_add_p_;
  double diag_add_m_scalar_;
  double diag_add_p_scalar_;

  double mu_m_;
  double mu_p_;

  bool variable_mu_;

  bool is_mue_m_dd_owned_;
  bool is_mue_p_dd_owned_;

  Vec  mue_m_;
  Vec  mue_p_;
  Vec  mue_m_xx_;
  Vec  mue_p_xx_;
  Vec  mue_m_yy_;
  Vec  mue_p_yy_;
#ifdef P4_TO_P8
  Vec  mue_m_zz_;
  Vec  mue_p_zz_;
#endif

  // solver options
  int integration_order_;
  int cube_refinement_;
  bool use_sc_scheme_;
  bool use_pointwise_dirichlet_;
  bool use_taylor_correction_;
  bool kink_special_treatment_;
  bool update_ghost_after_solving_;
  bool try_remove_hanging_cells_;
  bool neumann_wall_first_order_;
  bool enfornce_diag_scalling_;
  double phi_perturbation_;
  double domain_rel_thresh_;
  double interface_rel_thresh_;
  interpolation_method interp_method_;

  // Some flags
  bool new_pc_;
  bool is_matrix_computed_;
  int matrix_has_nullspace_;

  // Bondary conditions
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

  // auxiliary variables
  Vec mask_;
  Vec areas_;
  Vec areas_m_;
  Vec areas_p_;
//  Vec volumes_;
  Vec node_type_;

  double face_area_scalling_;

  std::vector<double> scalling_;

  bool keep_scalling_;
  bool volumes_computed_;
  bool volumes_owned_;

  Vec exact_;

  enum discretization_scheme_t { FDM, FVM, JUMP } discretization_scheme_;

  int jump_scheme_;

  void compute_volumes();
  void compute_phi_eff(Vec &phi_eff, std::vector<Vec> *&phi, std::vector<action_t> *&action, bool &is_phi_eff_owned);
#ifdef P4_TO_P8
  void compute_phi_dd(std::vector<Vec> *&phi, std::vector<Vec> *&phi_xx, std::vector<Vec> *&phi_yy, std::vector<Vec> *&phi_zz, bool &is_phi_dd_owned);
  void compute_phi_d (std::vector<Vec> *&phi, std::vector<Vec> *&phi_x,  std::vector<Vec> *&phi_y,  std::vector<Vec> *&phi_z,  bool &is_phi_d_owned);
#else
  void compute_phi_dd(std::vector<Vec> *&phi, std::vector<Vec> *&phi_xx, std::vector<Vec> *&phi_yy, bool &is_phi_dd_owned);
  void compute_phi_d (std::vector<Vec> *&phi, std::vector<Vec> *&phi_x,  std::vector<Vec> *&phi_y,  bool &is_phi_d_owned);
#endif
  void compute_mue_dd();

#ifdef P4_TO_P8
  double compute_weights_through_face(double A, double B, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face);
#else
  double compute_weights_through_face(double A, bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face);
#endif

  void preallocate_matrix();

  void setup_linear_system(bool setup_matrix, bool setup_rhs);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_poisson_nodes_mls_sc_t& other);
  my_p4est_poisson_nodes_mls_sc_t& operator=(const my_p4est_poisson_nodes_mls_sc_t& other);

  struct immersed_interface_t
  {
    unsigned int num_interfaces;

    std::vector<Vec> *phi;

    std::vector<action_t> *action;
    std::vector<int>      *color;

    Vec phi_eff;

    std::vector<Vec> *phi_x;
    std::vector<Vec> *phi_y;
#ifdef P4_TO_P8
    std::vector<Vec> *phi_z;
#endif

    std::vector<Vec> *phi_xx;
    std::vector<Vec> *phi_yy;
#ifdef P4_TO_P8
    std::vector<Vec> *phi_zz;
#endif

    bool is_phi_d_owned;
    bool is_phi_dd_owned;
    bool is_phi_eff_owned;

    immersed_interface_t() : num_interfaces(0), phi(NULL),  action(NULL), color(NULL), phi_eff(NULL),
      phi_x(NULL),  phi_y(NULL),
  #ifdef P4_TO_P8
      phi_z(NULL),
  #endif
      phi_xx(NULL), phi_yy(NULL),
  #ifdef P4_TO_P8
      phi_zz(NULL),
  #endif
      is_phi_d_owned(false), is_phi_dd_owned(false), is_phi_eff_owned(false) {}
  } ii_;

#ifdef P4_TO_P8
  std::vector< CF_3 *> *jump_flux_;
  CF_3 *jump_value_;
#else
  std::vector< CF_2 *> *jump_flux_;
  CF_2 *jump_value_;
#endif

public:
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_sc_t();

  inline PetscInt get_global_idx(p4est_locidx_t n) { return petsc_gloidx_[n]; }
  inline void set_try_remove_hanging_cells(bool value) { try_remove_hanging_cells_ = value; }

#ifdef P4_TO_P8
  inline void set_phi_cf(std::vector< CF_3 *> &phi_cf) { phi_cf_ = &phi_cf; }
#else
  inline void set_phi_cf(std::vector< CF_2 *> &phi_cf) { phi_cf_ = &phi_cf; }
#endif

  inline void set_exact(Vec exact) { exact_ = exact; }

  void set_lip(double lip) { lip_ = lip; }

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
                           Vec areas = NULL)
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
#ifdef P4_TO_P8
      compute_phi_dd(phi_, phi_xx_, phi_yy_, phi_zz_, is_phi_dd_owned_);
#else
      compute_phi_dd(phi_, phi_xx_, phi_yy_, is_phi_dd_owned_);
#endif
      is_phi_dd_owned_ = true;
    }

#ifdef P4_TO_P8
    compute_phi_d(phi_, phi_x_, phi_y_, phi_z_, is_phi_d_owned_);
#else
    compute_phi_d(phi_, phi_x_, phi_y_, is_phi_d_owned_);
#endif
    is_phi_d_owned_ = true;

    if (phi_eff != NULL)
      phi_eff_ = phi_eff;
    else
      compute_phi_eff(phi_eff_, phi_, action_, is_phi_eff_owned_);

    if (areas == NULL)
      volumes_computed_ = false;
    else
    {
      volumes_computed_ = true;
      areas_ = areas;
    }
    volumes_owned_ = false;
//    compute_volumes();

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

  inline void set_immersed_interface(int num_interfaces,
                           std::vector<action_t> *action, std::vector<int> *color,
                           std::vector<Vec> *phi,
                           std::vector<Vec> *phi_xx = NULL,
                           std::vector<Vec> *phi_yy = NULL,
                         #ifdef P4_TO_P8
                           std::vector<Vec> *phi_zz = NULL,
                         #endif
                           Vec phi_eff = NULL,
                           Vec areas_m = NULL,
                           Vec areas_p = NULL)
  {
    if (num_interfaces > 1) throw std::invalid_argument("Error: piece-wise smooth immersed interfaces are not supported at the moment.\n");
    ii_.num_interfaces = num_interfaces;
    ii_.action  = action;
    ii_.color   = color;
    ii_.phi     = phi;

    if (phi_xx != NULL &&
    #ifdef P4_TO_P8
        phi_zz != NULL &&
    #endif
        phi_yy != NULL)
    {
      ii_.phi_xx  = phi_xx;
      ii_.phi_yy  = phi_yy;
#ifdef P4_TO_P8
      ii_.phi_zz  = phi_zz;
#endif
      ii_.is_phi_dd_owned = false;
    } else {
#ifdef P4_TO_P8
      compute_phi_dd(ii_.phi, ii_.phi_xx, ii_.phi_yy, ii_.phi_zz, ii_.is_phi_dd_owned);
#else
      compute_phi_dd(ii_.phi, ii_.phi_xx, ii_.phi_yy, ii_.is_phi_dd_owned);
#endif
      ii_.is_phi_dd_owned = true;
    }

#ifdef P4_TO_P8
    compute_phi_d(ii_.phi, ii_.phi_x, ii_.phi_y, ii_.phi_z, ii_.is_phi_d_owned);
#else
    compute_phi_d(ii_.phi, ii_.phi_x, ii_.phi_y, ii_.is_phi_d_owned);
#endif
    ii_.is_phi_d_owned = true;

    if (phi_eff != NULL)
      ii_.phi_eff = phi_eff;
    else
      compute_phi_eff(ii_.phi_eff, ii_.phi, ii_.action, ii_.is_phi_eff_owned);

    if (areas_m == NULL || areas_p == NULL)
      volumes_computed_ = false;
    else
    {
      volumes_computed_ = true;
      areas_m_ = areas_m;
      areas_p_ = areas_p;
    }
    volumes_owned_ = false;
//    compute_volumes();

#ifdef CASL_THROWS
    if (ii_.num_interfaces > 0)
      if (ii_.action->size() != ii_.num_interfaces ||
          ii_.color->size()  != ii_.num_interfaces ||
          ii_.phi->size()    != ii_.num_interfaces ||
          ii_.phi_xx->size() != ii_.num_interfaces ||
    #ifdef P4_TO_P8
          ii_.phi_zz->size() != ii_.num_interfaces ||
    #endif
          ii_.phi_yy->size() != ii_.num_interfaces )
        throw std::invalid_argument("[CASL_ERROR]: invalid geometry for the immersed interface");
#endif

  }

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

#ifdef P4_TO_P8
  inline void set_jump_conditions(CF_3 &jump_value, std::vector< CF_3 *> &jump_flux) { jump_value_ = &jump_value; jump_flux_ = &jump_flux; }
#else
  inline void set_jump_conditions(CF_2 &jump_value, std::vector< CF_2 *> &jump_flux) { jump_value_ = &jump_value; jump_flux_ = &jump_flux; }
#endif

  inline void set_diag_add(double diag_add_scalar)   { diag_add_m_scalar_ = diag_add_scalar;
                                                       diag_add_p_scalar_ = diag_add_scalar; is_matrix_computed_ = false; }

  inline void set_diag_add(double diag_add_m_scalar,
                           double diag_add_p_scalar) { diag_add_m_scalar_ = diag_add_m_scalar;
                                                       diag_add_p_scalar_ = diag_add_p_scalar; is_matrix_computed_ = false; }

  inline void set_diag_add(Vec diag_add)   { diag_add_m_ = diag_add;
                                             diag_add_p_ = diag_add;   is_matrix_computed_ = false; }
  inline void set_diag_add(Vec diag_add_m,
                           Vec diag_add_p) { diag_add_m_ = diag_add_m;
                                             diag_add_p_ = diag_add_p; is_matrix_computed_ = false; }

  inline void set_mu(double mu) { mu_m_ = mu; mu_p_ = mu; variable_mu_ = false; }
  inline void set_mu(double mu_m, double mu_p) { mu_m_ = mu_m; mu_p_ = mu_p;  variable_mu_ = false; }

#ifdef P4_TO_P8
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL, Vec mue_zz = NULL)
#else
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL)
#endif
  {
    mue_m_ = mue;
    mue_p_ = mue;

    if (mue_xx != NULL &&
    #ifdef P4_TO_P8
        mue_zz != NULL &&
    #endif
        mue_yy != NULL)
    {
      mue_m_xx_ = mue_xx;
      mue_p_xx_ = mue_xx;
      mue_m_yy_ = mue_yy;
      mue_p_yy_ = mue_yy;
#ifdef P4_TO_P8
      mue_m_zz_ = mue_zz;
      mue_p_zz_ = mue_zz;
#endif
      is_mue_m_dd_owned_ = false;
      is_mue_p_dd_owned_ = false;
    } else {
      compute_mue_dd();
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }

#ifdef P4_TO_P8
  void set_mu2(Vec mue_m,
              Vec mue_p,
              Vec mue_m_xx = NULL, Vec mue_m_yy = NULL, Vec mue_m_zz = NULL,
              Vec mue_p_xx = NULL, Vec mue_p_yy = NULL, Vec mue_p_zz = NULL)
#else
  void set_mu2(Vec mue_m,
              Vec mue_p,
              Vec mue_m_xx = NULL, Vec mue_m_yy = NULL,
              Vec mue_p_xx = NULL, Vec mue_p_yy = NULL)
#endif
  {
    mue_m_ = mue_m;
    mue_p_ = mue_p;

    if (mue_m_xx != NULL &&
        mue_p_xx != NULL &&
    #ifdef P4_TO_P8
        mue_m_zz != NULL &&
        mue_p_zz != NULL &&
    #endif
        mue_m_yy != NULL &&
        mue_p_yy != NULL)
    {
      mue_m_xx_ = mue_m_xx;
      mue_p_xx_ = mue_p_xx;
      mue_m_yy_ = mue_m_yy;
      mue_p_yy_ = mue_p_yy;
#ifdef P4_TO_P8
      mue_m_zz_ = mue_m_zz;
      mue_p_zz_ = mue_p_zz;
#endif
      is_mue_m_dd_owned_ = false;
      is_mue_p_dd_owned_ = false;
    } else {
      compute_mue_dd();
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }


  inline void set_rhs(Vec rhs)   { rhs_m_ = rhs;   rhs_p_ = rhs;   }
  inline void set_rhs(Vec rhs_m,
                      Vec rhs_p) { rhs_m_ = rhs_m; rhs_p_ = rhs_p; }
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
  inline void set_jump_scheme             (int  value) { jump_scheme_               = value; }
  inline void set_enfornce_diag_scalling  (int  value) { enfornce_diag_scalling_    = value; }
  inline void set_update_ghost_after_solving(bool value) { update_ghost_after_solving_ = value; }

  bool inv_mat2(double *in, double *out);
  bool inv_mat3(double *in, double *out);
  bool inv_mat4(const double m[16], double invOut[16]);

//  void find_projection_(double *phi_p, p4est_locidx_t *neighbors, bool *neighbor_exists, double dxyz_pr[], double &dist_pr);
  void find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr, double normal[] = NULL);

  void compute_normal(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[]);

  void get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists);

  void find_hanging_cells(int *network, bool *hanging_cells);

//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCASM);
  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPGMRES, PCType pc_type = PCSOR);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE);
//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCASM);

  void assemble_matrix(Vec solution);

  inline Vec get_mask() { return mask_; }
  inline Vec get_areas() { return areas_; }

  inline void get_phi_dd(std::vector<Vec> **phi_dd)
  {
    phi_dd[0] = phi_xx_;
    phi_dd[1] = phi_yy_;
#ifdef P4_TO_P8
    phi_dd[2] = phi_zz_;
#endif
  }

  inline Vec get_phi_eff() { return phi_eff_; }
  inline Vec get_immersed_phi_eff() { return ii_.phi_eff; }

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
