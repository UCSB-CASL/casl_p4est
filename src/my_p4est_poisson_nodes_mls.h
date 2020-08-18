#ifndef MY_P4EST_POISSON_NODES_MLS_H
#define MY_P4EST_POISSON_NODES_MLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_utils.h>
#endif

#include <src/mls_integration/cube3_mls.h>
#include <src/mls_integration/cube2_mls.h>

using std::vector;

class my_p4est_poisson_nodes_mls_t
{
protected:
  const int phi_idx_wall_shift_ = 10;
  struct mat_entry_t
  {
    double val;
    PetscInt n;
    mat_entry_t(PetscInt n=0, double val=0) : n(n), val(val) {}
  };
  PetscErrorCode ierr;

  // p4est objects
  p4est_t          *p4est_;
  p4est_nodes_t    *nodes_;
  p4est_ghost_t    *ghost_;
  my_p4est_brick_t *brick_;
  const my_p4est_node_neighbors_t *ngbd_;

  // grid variables
  double lip_;
  double d_min_;
  double diag_min_;
  double dxyz_m_[P4EST_DIM];
  double DIM( dx_min_, dy_min_, dz_min_ );

  // linear system
  Mat     A_;
  Vec     diag_scaling_;
  Vec     rhs_;
  double *rhs_ptr;
  Vec     rhs_jump_;
  double *rhs_jump_ptr;

  Vec     rhs_gf_;
  double *rhs_gf_ptr;

  // subcomponents of linear system
  Mat     submat_main_;
  Vec     submat_diag_;
  double *submat_diag_ptr;
  Vec     submat_diag_ghost_;
  double *submat_diag_ghost_ptr;
  Mat     submat_jump_; // describes contribution of ghost nodes into main matrix
  Mat     submat_jump_ghost_; // expresses ghost values through real values
  Mat     submat_robin_sc_;
  Vec     submat_robin_sym_;
  double *submat_robin_sym_ptr;
  std::vector< std::vector<mat_entry_t> > entries_robin_sc;

  // for imposing dirichlet using ghost fluid method
  Mat     submat_gf_; // describes contribution of ghost nodes into main matrix
  Mat     submat_gf_ghost_; // expresses ghost values through real values

  bool new_submat_main_;
  bool new_submat_diag_;
  bool new_submat_robin_;

  bool there_is_diag_;
  bool there_is_dirichlet_;
  bool there_is_neumann_;
  bool there_is_robin_;
  bool there_is_jump_;
  bool there_is_jump_mu_;

  bool A_needs_reassembly_;

  // PETSc solver
  KSP         ksp_;
  bool        new_pc_;
  PetscInt    itmax_;
  PetscScalar rtol_, atol_, dtol_;

  // Tolerances for solving nonlinear equations
  double nonlinear_change_tol_, nonlinear_pde_residual_tol_;
  double nonlinear_itmax_;
  int    nonlinear_method_;

  // local to global node number mapping
  std::vector<PetscInt> global_node_offset_;
  std::vector<PetscInt> petsc_gloidx_;

  // tracking nullspace
  bool nullspace_main_;
  bool nullspace_diag_;
  bool nullspace_robin_;

  // geometry
  class geometry_t
  {
    const my_p4est_node_neighbors_t *ngbd_;
    p4est_t       *p4est_;
    p4est_nodes_t *nodes_;
  public:
    int                    num_phi;
    Vec                    phi_eff;
    std::vector<mls_opn_t> opn;
    std::vector<int>       clr;
    std::vector<Vec>       phi;
    std::vector<Vec>       DIM( phi_xx,
                                phi_yy,
                                phi_zz );

    // pointers
    double                *phi_eff_ptr;
    std::vector<double *>  phi_ptr;
    std::vector<double *>  DIM( phi_xx_ptr,
                                phi_yy_ptr,
                                phi_zz_ptr );

    // auxilary
    bool              is_phi_eff_owned;
    std::vector<bool> is_phi_dd_owned;

    geometry_t(const my_p4est_node_neighbors_t *ngbd, p4est_t *p4est, p4est_nodes_t *nodes)
      : ngbd_(ngbd), p4est_(p4est), nodes_(nodes), num_phi(0), phi_eff(NULL), is_phi_eff_owned(0) {}
    ~geometry_t();

    void get_arrays();
    void restore_arrays();
    void calculate_phi_eff();
    Vec  return_phi_eff();
    void add_phi(mls_opn_t opn, Vec phi, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz));
    inline double phi_eff_value(p4est_locidx_t n) { return (num_phi == 0) ? -1 : phi_eff_ptr[n]; }
  };

  geometry_t bdry_;
  geometry_t infc_;

  // forces
  Vec     rhs_m_;
  Vec     rhs_p_;
  double *rhs_m_ptr;
  double *rhs_p_ptr;

  // linear term
  bool    var_diag_;
  double  diag_m_scalar_;
  double  diag_p_scalar_;
  Vec     diag_m_;
  Vec     diag_p_;
  double *diag_m_ptr;
  double *diag_p_ptr;

  // diffusion coefficient
  bool   var_mu_;
  double mu_m_;
  double mu_p_;
  Vec    mue_m_, DIM(mue_m_xx_, mue_m_yy_, mue_m_zz_);
  Vec    mue_p_, DIM(mue_p_xx_, mue_p_yy_, mue_p_zz_);
  bool   is_mue_m_dd_owned_;
  bool   is_mue_p_dd_owned_;

  double *mue_m_ptr, DIM(*mue_m_xx_ptr, *mue_m_yy_ptr, *mue_m_zz_ptr);
  double *mue_p_ptr, DIM(*mue_p_xx_ptr, *mue_p_yy_ptr, *mue_p_zz_ptr);

  // nonlinear term
  CF_1   *nonlinear_term_m_;
  CF_1   *nonlinear_term_p_;
  CF_1   *nonlinear_term_m_prime_;
  CF_1   *nonlinear_term_p_prime_;
  Vec     nonlinear_term_m_coeff_;
  Vec     nonlinear_term_p_coeff_;
  double *nonlinear_term_m_coeff_ptr;
  double *nonlinear_term_p_coeff_ptr;
  double  nonlinear_term_m_coeff_scalar_;
  double  nonlinear_term_p_coeff_scalar_;
  bool    var_nonlinear_term_coeff_;

  // wall conditions
  points_around_node_map_t       wall_pieces_map;
  std::vector<int>               wall_pieces_id;
  std::vector<double>            wall_pieces_area;
  std::vector<interface_point_t> wall_pieces_centroid;

  const WallBCDIM *wc_type_;
  const CF_DIM    *wc_value_;
  const CF_DIM    *wc_coeff_;

  void save_wall_data(p4est_locidx_t n, vector<int> &wall_id, vector<double> &wall_area, vector<interface_point_t> &wall_xyz);
  void load_wall_data(p4est_locidx_t n, vector<int> &wall_id, vector<double> &wall_area, vector<interface_point_t> &wall_xyz);

  vector< boundary_conditions_t> bc_;
  vector<interface_conditions_t> jc_;

  void save_cart_points(p4est_locidx_t n, vector<bool> &is_interface, vector<int> &bdry_points_id, vector<double> &bdry_dist, vector<double> &bdry_points_weights);
  void load_cart_points(p4est_locidx_t n, vector<bool> &is_interface, vector<int> &bdry_points_id, vector<double> &bdry_dist, vector<double> &bdry_points_weights);

  void save_bdry_data(p4est_locidx_t n, vector<int> &bdry_ids, vector<double> &bdry_areas, vector<interface_point_t> &bdry_value_pts, vector<interface_point_t> &bdry_robin_pts);
  void load_bdry_data(p4est_locidx_t n, vector<int> &bdry_ids, vector<double> &bdry_areas, vector<interface_point_t> &bdry_value_pts, vector<interface_point_t> &bdry_robin_pts);

  void save_infc_data(p4est_locidx_t n, vector<int> &infc_ids, vector<double> &infc_areas, vector<interface_point_t> &infc_integr_pts, vector<interface_point_t> &infc_taylor_pts);
  void load_infc_data(p4est_locidx_t n, vector<int> &infc_ids, vector<double> &infc_areas, vector<interface_point_t> &infc_integr_pts, vector<interface_point_t> &infc_taylor_pts);

  // solver options
  int    integration_order_;
  int    cube_refinement_;
  int    jump_scheme_;
  int    fv_scheme_;
  int    dirichlet_scheme_; // 0 - Shortley-Weller, 1 - ghost fluid
  int    gf_order_;
  int    gf_stabilized_; // 0 - only non-stab, 1 - only stab, 2 - both (stab prefered over non-stab)


  bool   use_taylor_correction_;
  bool   kink_special_treatment_;
  bool   neumann_wall_first_order_;
  bool   enfornce_diag_scaling_;
  bool   use_centroid_always_;

  double phi_perturbation_;
  double domain_rel_thresh_;
  double interface_rel_thresh_;
  double gf_thresh_;

  interpolation_method interp_method_;

  // auxiliary variables
  Vec volumes_m_; double *volumes_m_ptr;
  Vec volumes_p_; double *volumes_p_ptr;
  Vec areas_m_;   double *areas_m_ptr;
  Vec areas_p_;   double *areas_p_ptr;
  Vec mask_m_;    double *mask_m_ptr;
  Vec mask_p_;    double *mask_p_ptr;

  bool volumes_computed_;
  bool volumes_owned_;

  double face_area_scalling_;
  vector<double> jump_scaling_;

  // finite volumes
  bool store_finite_volumes_;
  bool finite_volumes_initialized_;
  bool finite_volumes_owned_;
  std::vector<int> bdry_node_to_fv_;
  std::vector<int> infc_node_to_fv_;
  std::vector<my_p4est_finite_volume_t> *bdry_fvs_;
  std::vector<my_p4est_finite_volume_t> *infc_fvs_;

  // discretization type
  enum discretization_scheme_t
  {
    UNDEFINED,
    DOMAIN_OUTSIDE,
    DOMAIN_INSIDE,
    WALL_DIRICHLET,
    WALL_NEUMANN,
    BOUNDARY_DIRICHLET,
    BOUNDARY_NEUMANN,
    IMMERSED_INTERFACE,
  };

  std::vector<discretization_scheme_t> node_scheme_;

  // interpolators
  my_p4est_interpolation_nodes_local_t mu_m_interp_;
  my_p4est_interpolation_nodes_local_t mu_p_interp_;

  std::vector<my_p4est_interpolation_nodes_local_t *> bdry_phi_interp_;
  std::vector<my_p4est_interpolation_nodes_local_t *> infc_phi_interp_;

  std::vector<CF_DIM *> bdry_phi_cf_;
  std::vector<CF_DIM *> infc_phi_cf_;

  void interpolators_initialize();
  void interpolators_prepare(p4est_locidx_t n);
  void interpolators_finalize();

  // discretization
  void setup_linear_system (bool setup_rhs);

  void discretize_inside      (bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                               double infc_phi_eff_000, bool is_wall[],
                               std::vector<mat_entry_t> *row_main, PetscInt &d_nnz, PetscInt &o_nnz);

  void discretize_dirichlet_sw(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                               double infc_phi_eff_000, bool is_wall[],
                               std::vector<mat_entry_t> *row_main, PetscInt &d_nnz, PetscInt &o_nnz);

  void discretize_dirichlet_gf(bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                               double infc_phi_eff_000, bool is_wall[],
                               vector<int> &gf_map, vector<double> &gf_nodes, vector<double> &gf_phi,
                               std::vector<mat_entry_t> *row_main, PetscInt &d_nnz_main, PetscInt &o_nnz_main,
                               std::vector<mat_entry_t> *row_gf, PetscInt &d_nnz_gf, PetscInt &o_nnz_gf,
                               std::vector<mat_entry_t> *row_gf_ghost, PetscInt &d_nnz_gf_ghost, PetscInt &o_nnz_gf_ghost);



  void discretize_robin       (bool setup_rhs, p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                               double infc_phi_eff_000, bool is_wall[],
                               std::vector<mat_entry_t> *row_main, PetscInt &d_nnz_main, PetscInt &o_nnz_main,
                               std::vector<mat_entry_t> *row_robin_sc, PetscInt &d_nnz_robin_sc, PetscInt &o_nnz_robin_sc);

  void discretize_jump        (bool setup_rhs,  p4est_locidx_t n, const quad_neighbor_nodes_of_node_t &qnnn,
                               bool is_wall[],
                               std::vector<mat_entry_t> *row_main, PetscInt &d_nnz_main, PetscInt &o_nnz_main,
                               std::vector<mat_entry_t> *row_jump, PetscInt &d_nnz_jump, PetscInt &o_nnz_jump,
                               std::vector<mat_entry_t> *row_jump_ghost, PetscInt &d_nnz_jump_ghost, PetscInt &o_nnz_jump_ghost);

  void find_interface_points(p4est_locidx_t n, const my_p4est_node_neighbors_t *ngbd,
                             std::vector<mls_opn_t> opn,
                             std::vector<double *> phi_ptr, DIM( std::vector<double *> phi_xx_ptr,
                                                                 std::vector<double *> phi_yy_ptr,
                                                                 std::vector<double *> phi_zz_ptr ),
                             int phi_idx[], double dist[]);

  bool inv_mat2(const double *in, double *out);
  bool inv_mat3(const double *in, double *out);
  bool inv_mat4(const double *in, double *out);

  void compute_mue_dd();
  double compute_weights_through_face(double A P8C(double B), bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face);
  void find_projection(const quad_neighbor_nodes_of_node_t& qnnn, const double *phi_p, double dxyz_pr[], double &dist_pr, double normal[] = NULL);
  void invert_linear_system(Vec solution, bool use_nonzero_guess, bool update_ghost, KSPType ksp_type, PCType pc_type);
  void assemble_matrix(std::vector< std::vector<mat_entry_t> > &entries, std::vector<PetscInt> &d_nnz, std::vector<PetscInt> &o_nnz, Mat *matrix);

  inline int gf_stencil_size() {
    return gf_stabilized_ == 0 ? gf_order_ + 1 : gf_order_ + 2;
  }

  bool gf_is_ghost(const quad_neighbor_nodes_of_node_t &qnnn);
  void gf_direction(const quad_neighbor_nodes_of_node_t &qnnn, const p4est_locidx_t neighbors[], int &dir, double del_xyz[]);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_t(const my_p4est_poisson_nodes_mls_t& other);
  my_p4est_poisson_nodes_mls_t& operator=(const my_p4est_poisson_nodes_mls_t& other);
public:

  my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_poisson_nodes_mls_t();

  inline int  pw_bc_num_value_pts(int phi_idx) { return bc_[phi_idx].num_value_pts(); }
  inline int  pw_bc_num_robin_pts(int phi_idx) { return bc_[phi_idx].num_robin_pts(); }

  inline void pw_bc_xyz_value_pt (int phi_idx, int pt_idx, double pt_xyz[]) { bc_[phi_idx].xyz_value_pt(pt_idx, pt_xyz); }
  inline void pw_bc_xyz_robin_pt (int phi_idx, int pt_idx, double pt_xyz[]) { bc_[phi_idx].xyz_robin_pt(pt_idx, pt_xyz); }

  inline int  pw_bc_num_value_pts(int phi_idx, p4est_locidx_t n) { return bc_[phi_idx].num_value_pts(n); }
  inline int  pw_bc_num_robin_pts(int phi_idx, p4est_locidx_t n) { return bc_[phi_idx].num_robin_pts(n); }

  inline int  pw_bc_idx_value_pt (int phi_idx, p4est_locidx_t n, int k) { return bc_[phi_idx].idx_value_pt(n, k); }
  inline int  pw_bc_idx_robin_pt (int phi_idx, p4est_locidx_t n, int k) { return bc_[phi_idx].idx_robin_pt(n, k); }

  inline int  pw_bc_get_boundary_pt(int phi_idx, int pt_idx, interface_point_cartesian_t* &pt) { pt = &bc_[phi_idx].dirichlet_pts[pt_idx]; }

  inline int  pw_jc_num_integr_pts(int phi_idx) { return jc_[phi_idx].num_integr_pts(); }
  inline int  pw_jc_num_taylor_pts(int phi_idx) { return jc_[phi_idx].num_taylor_pts(); }

  inline void pw_jc_xyz_integr_pt (int phi_idx, int pt_idx, double pt_xyz[]) { jc_[phi_idx].xyz_integr_pt(pt_idx, pt_xyz); }
  inline void pw_jc_xyz_taylor_pt (int phi_idx, int pt_idx, double pt_xyz[]) { jc_[phi_idx].xyz_taylor_pt(pt_idx, pt_xyz); }

  inline int  pw_jc_num_integr_pts(int phi_idx, p4est_locidx_t n) { return jc_[phi_idx].num_integr_pts(n); }
  inline int  pw_jc_num_taylor_pts(int phi_idx, p4est_locidx_t n) { return jc_[phi_idx].num_taylor_pts(n); }

  inline int  pw_jc_idx_integr_pt (int phi_idx, p4est_locidx_t n, int k) { return jc_[phi_idx].idx_integr_pt(n, k); }
  inline int  pw_jc_idx_taylor_pt (int phi_idx, p4est_locidx_t n, int k) { return jc_[phi_idx].idx_taylor_pt(n, k); }

  // set geometry
  inline void set_lip(double lip) { lip_ = lip; }

  inline void add_boundary (mls_opn_t opn, Vec phi, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz), BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff)
  {
    boundary_conditions_t bc;
    bc.type     =  bc_type;
    bc.value_cf = &bc_value;
    bc.coeff_cf = &bc_coeff;

    this->bc_.push_back(bc);

    bdry_.add_phi(opn, phi, DIM(phi_xx, phi_yy, phi_zz));

    switch (bc_type)
    {
      case NEUMANN:   there_is_neumann_   = true; break;
      case ROBIN:     there_is_robin_     = true; break;
      case DIRICHLET: there_is_dirichlet_ = true; break;
//      default:
//#ifdef CASL_THROWS
//      throw std::run_time_error("my_p4est_poisson_nodes_mls_t:add_boundary:unknown boundary");
//#endif
//      break;
    }
  }

  inline void add_boundary (mls_opn_t opn, Vec phi, Vec* phi_dd, BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff)
  {
    if (phi_dd != NULL) add_boundary(opn, phi, DIM(phi_dd[0], phi_dd[1], phi_dd[2]), bc_type, bc_value, bc_coeff);
    else                add_boundary(opn, phi, DIM(NULL,      NULL,      NULL),      bc_type, bc_value, bc_coeff);
  }

  inline void add_interface(mls_opn_t opn, Vec phi, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz), CF_DIM &jc_value, CF_DIM &jc_flux)
  {
    interface_conditions_t jc;
    jc.sol_jump_cf = &jc_value;
    jc.flx_jump_cf = &jc_flux;

    this->jc_.push_back(jc);

    infc_.add_phi(opn, phi, DIM(phi_xx, phi_yy, phi_zz));

    there_is_jump_ = true;
  }

  inline void add_interface(mls_opn_t opn, Vec phi, Vec* phi_dd, CF_DIM &jc_value, CF_DIM &jc_flux)
  {
    if (phi_dd != NULL) add_interface(opn, phi, DIM(phi_dd[0], phi_dd[1], phi_dd[2]), jc_value, jc_flux);
    else                add_interface(opn, phi, DIM(NULL,      NULL,      NULL),      jc_value, jc_flux);
  }

  inline void set_boundary_phi_eff (Vec phi_eff) { bdry_.phi_eff = phi_eff; bdry_.calculate_phi_eff(); }
  inline void set_interface_phi_eff(Vec phi_eff) { infc_.phi_eff = phi_eff; infc_.calculate_phi_eff(); }

  // set wall conditions
  //  inline void set_wc(const WallBCDIM &wc_type, const CF_DIM &wc_value, const CF_DIM &wc_coeff)
  inline void set_wc(const WallBCDIM &wc_type, const CF_DIM &wc_value, bool new_submat_main = true)
  {
    this->wc_type_  = &wc_type;
    this->wc_value_ = &wc_value;

    new_submat_main_  = new_submat_main;
  }

  inline void set_wc(BoundaryConditionType wc_type, const CF_DIM &wc_value, bool new_submat_main = true)
  {
    switch (wc_type) {
      case DIRICHLET: this->wc_type_ = &dirichlet_cf; break;
      case NEUMANN:   this->wc_type_ = &neumann_cf;   break;
      default: throw;
    }

    this->wc_value_ = &wc_value;

    new_submat_main_  = new_submat_main;
  }

  // overwrite boundary conditions (optional)
  inline void set_bc(int phi_idx, BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff)
  {
    if (this->bc_[phi_idx].type != bc_type) throw std::invalid_argument("Cannot change BC on fly\n");
    if (bc_type == ROBIN) new_submat_robin_ = true;

    this->bc_[phi_idx].pointwise = false;
    this->bc_[phi_idx].value_cf  = &bc_value;
    this->bc_[phi_idx].coeff_cf  = &bc_coeff;
  }

  inline void set_bc(int phi_idx, BoundaryConditionType bc_type, vector<double> &bc_value_pw, vector<double> &bc_value_pw_robin, vector<double> &bc_coeff_pw_robin)
  {
    if (this->bc_[phi_idx].type != bc_type) throw std::invalid_argument("Cannot change BC on fly\n");
    if (bc_type == ROBIN) new_submat_robin_ = true;

    this->bc_[phi_idx].pointwise      = true;
    this->bc_[phi_idx].value_pw       = &bc_value_pw;
    this->bc_[phi_idx].value_pw_robin = &bc_value_pw_robin;
    this->bc_[phi_idx].coeff_pw_robin = &bc_coeff_pw_robin;
  }

  inline void set_bc(int phi_idx, BoundaryConditionType bc_type, vector<double> &bc_value_pw)
  {
    if (this->bc_[phi_idx].type != bc_type) throw std::invalid_argument("Cannot change BC on fly\n");
    if (bc_type == ROBIN) throw;

    this->bc_[phi_idx].pointwise      = true;
    this->bc_[phi_idx].value_pw       = &bc_value_pw;
    this->bc_[phi_idx].value_pw_robin = NULL;
    this->bc_[phi_idx].coeff_pw_robin = NULL;
  }

  // overwtire jump conditions (optional)
  inline void set_jc(int phi_idx, CF_DIM &jc_value, CF_DIM &jc_flux)
  {
    this->jc_[phi_idx].pointwise   = false;
    this->jc_[phi_idx].sol_jump_cf = &jc_value;
    this->jc_[phi_idx].flx_jump_cf = &jc_flux;
  }

  inline void set_jc(int phi_idx, vector<double> &jc_sol_jump_taylor, vector<double> &jc_flx_jump_taylor, vector<double> &jc_flx_jump_integr)
  {
    this->jc_[phi_idx].pointwise          = true;
    this->jc_[phi_idx].sol_jump_pw_taylor = &jc_sol_jump_taylor;
    this->jc_[phi_idx].flx_jump_pw_taylor = &jc_flx_jump_taylor;
    this->jc_[phi_idx].flx_jump_pw_integr = &jc_flx_jump_integr;
  }

  // set linear term
  inline void set_diag(double diag_m, double diag_p) { diag_m_scalar_ = diag_m; diag_p_scalar_ = diag_p; var_diag_ = false; new_submat_diag_ = true; there_is_diag_ = true; }
  inline void set_diag(double diag)                  { diag_m_scalar_ = diag;   diag_p_scalar_ = diag;   var_diag_ = false; new_submat_diag_ = true; there_is_diag_ = true; }

  inline void set_diag(Vec diag_m, Vec diag_p) { diag_m_ = diag_m; diag_p_ = diag_p; var_diag_ = true; new_submat_diag_ = true; there_is_diag_ = true; }
  inline void set_diag(Vec diag)               { diag_m_ = diag;   diag_p_ = diag;   var_diag_ = true; new_submat_diag_ = true; there_is_diag_ = true; }

  // set diffusion coefficient
  inline void set_mu(double mu_m, double mu_p) { mu_m_ = mu_m; mu_p_ = mu_p;  var_mu_ = false; new_submat_main_ = new_submat_robin_ = true; there_is_jump_mu_ = !(mu_m == mu_p); }
  inline void set_mu(double mu)                { mu_m_ = mu;   mu_p_ = mu;    var_mu_ = false; new_submat_main_ = new_submat_robin_ = true; there_is_jump_mu_ = false; }

  void set_mu(Vec mue_m, DIM(Vec mue_m_xx, Vec mue_m_yy, Vec mue_m_zz),
              Vec mue_p, DIM(Vec mue_p_xx, Vec mue_p_yy, Vec mue_p_zz));

  inline void set_mu(Vec mue, DIM(Vec mue_xx, Vec mue_yy, Vec mue_zz))
  {
    set_mu(mue, DIM(mue_xx, mue_yy, mue_zz),
           mue, DIM(mue_xx, mue_yy, mue_zz));
  }
  inline void set_mu(Vec mue_m, Vec mue_p)
  {
    set_mu(mue_m, DIM(NULL, NULL, NULL),
           mue_p, DIM(NULL, NULL, NULL));
  }
  inline void set_mu(Vec mue)
  {
    set_mu(mue, DIM(NULL, NULL, NULL),
           mue, DIM(NULL, NULL, NULL));
  }

  // set rhs
  inline void set_rhs(Vec rhs)              { rhs_m_ = rhs;   rhs_p_ = rhs;   }
  inline void set_rhs(Vec rhs_m, Vec rhs_p) { rhs_m_ = rhs_m; rhs_p_ = rhs_p; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT)
  {
    this->rtol_  = rtol;
    this->atol_  = atol;
    this->dtol_  = dtol;
    this->itmax_ = itmax;
  }

  // set nonlinear term

  inline void set_nonlinear_term(double nonlinear_term_m_coeff, CF_1 &nonlinear_m_term, CF_1 &nonlinear_term_m_prime,
                                 double nonlinear_term_p_coeff, CF_1 &nonlinear_p_term, CF_1 &nonlinear_term_p_prime)
  {
    var_nonlinear_term_coeff_      = false;
    nonlinear_term_m_              =&nonlinear_m_term;
    nonlinear_term_p_              =&nonlinear_p_term;
    nonlinear_term_m_prime_        =&nonlinear_term_m_prime;
    nonlinear_term_p_prime_        =&nonlinear_term_p_prime;
    nonlinear_term_m_coeff_scalar_ = nonlinear_term_m_coeff;
    nonlinear_term_p_coeff_scalar_ = nonlinear_term_p_coeff;
  }

  inline void set_nonlinear_term(Vec nonlinear_term_m_coeff, CF_1 &nonlinear_m_term, CF_1 &nonlinear_term_m_prime,
                                 Vec nonlinear_term_p_coeff, CF_1 &nonlinear_p_term, CF_1 &nonlinear_term_p_prime)
  {
    var_nonlinear_term_coeff_ = true;
    nonlinear_term_m_         =&nonlinear_m_term;
    nonlinear_term_p_         =&nonlinear_p_term;
    nonlinear_term_m_prime_   =&nonlinear_term_m_prime;
    nonlinear_term_p_prime_   =&nonlinear_term_p_prime;
    nonlinear_term_m_coeff_   = nonlinear_term_m_coeff;
    nonlinear_term_p_coeff_   = nonlinear_term_p_coeff;
  }

  inline void set_nonlinear_term(double nonlinear_term_coeff, CF_1 &nonlinear_term, CF_1 &nonlinear_term_prime)
  {
    set_nonlinear_term(nonlinear_term_coeff, nonlinear_term, nonlinear_term_prime,
                       nonlinear_term_coeff, nonlinear_term, nonlinear_term_prime);
  }

  inline void set_nonlinear_term(Vec nonlinear_term_coeff, CF_1 &nonlinear_term, CF_1 &nonlinear_term_prime)
  {
    set_nonlinear_term(nonlinear_term_coeff, nonlinear_term, nonlinear_term_prime,
                       nonlinear_term_coeff, nonlinear_term, nonlinear_term_prime);
  }


  // solver options
  inline void set_integration_order(int value) { integration_order_ = value; }
  inline void set_cube_refinement  (int value) { cube_refinement_   = value; }
  inline void set_jump_scheme      (int value) { jump_scheme_       = value; }
  inline void set_fv_scheme        (int value) { fv_scheme_         = value; }

  inline void set_use_taylor_correction   (bool value) { use_taylor_correction_    = value; }
  inline void set_kink_treatment          (bool value) { kink_special_treatment_   = value; }
  inline void set_first_order_neumann_wall(bool value) { neumann_wall_first_order_ = value; }
  inline void set_enfornce_diag_scaling   (bool value) { enfornce_diag_scaling_    = value; }
  inline void set_use_centroid_always     (bool value) { use_centroid_always_      = value; }
  inline void set_store_finite_volumes    (bool value) { store_finite_volumes_     = value; }

  inline void set_phi_perturbation    (double value) { phi_perturbation_     = value; }
  inline void set_domain_rel_thresh   (double value) { domain_rel_thresh_    = value; }
  inline void set_interface_rel_thresh(double value) { interface_rel_thresh_ = value; }

  inline void set_interpolation_method(interpolation_method value) { interp_method_ = value; }

  inline void set_use_sc_scheme           (bool value) { if (value) fv_scheme_ = 1; else fv_scheme_ = 0; } // deprecated

  // some override capabilities just in case
  inline void set_new_submat_main (bool value) { new_submat_main_  = value; }
  inline void set_new_submat_diag (bool value) { new_submat_diag_  = value; }
  inline void set_new_submat_robin(bool value) { new_submat_robin_ = value; }

  void preassemble_linear_system();
  void solve          (Vec solution, bool use_nonzero_guess = false, bool update_ghost = true, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
  int  solve_nonlinear(Vec solution, bool use_nonzero_guess = false, bool update_ghost = true, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  inline Vec get_mask()   { return mask_m_; }
  inline Vec get_mask_m() { return mask_m_; }
  inline Vec get_mask_p() { return mask_p_; }

  inline Vec get_areas()   { return areas_m_; }
  inline Vec get_areas_m() { return areas_m_; }
  inline Vec get_areas_p() { return areas_p_; }

  inline Vec get_boundary_phi_eff()  { return bdry_.phi_eff; }
  inline Vec get_interface_phi_eff() { return infc_.phi_eff; }

//  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace_; }

  inline Mat get_matrix() { return A_; }

  inline boundary_conditions_t* get_bc(int phi_idx) { return &bc_[phi_idx]; }

  // finite volumes
  inline void get_boundary_finite_volumes(vector< my_p4est_finite_volume_t > *&bdry_fvs, vector<int> *&bdry_node_to_fv)
  {
    bdry_fvs        =  bdry_fvs_;
    bdry_node_to_fv = &bdry_node_to_fv_;
  }

  inline void get_interface_finite_volumes(vector< my_p4est_finite_volume_t > *&infc_fvs, vector<int> *&infc_node_to_fv)
  {
    infc_fvs        =  infc_fvs_;
    infc_node_to_fv = &infc_node_to_fv_;
  }

  inline void set_finite_volumes(vector< my_p4est_finite_volume_t > *bdry_fvs, vector<int> *bdry_node_to_fv,
                                 vector< my_p4est_finite_volume_t > *infc_fvs, vector<int> *infc_node_to_fv)
  {
    if (bdry_fvs != NULL && bdry_node_to_fv != NULL)
    {
      bdry_fvs_        =  bdry_fvs;
      bdry_node_to_fv_ = *bdry_node_to_fv;
    }
    if (infc_fvs != NULL && infc_node_to_fv != NULL)
    {
      infc_fvs_        =  infc_fvs;
      infc_node_to_fv_ = *infc_node_to_fv;
    }
    finite_volumes_initialized_ = true;
  }

  inline PetscInt get_global_idx(p4est_locidx_t n) { return petsc_gloidx_[n]; }

  inline void set_solve_nonlinear_parameters(int method=1, double itmax=10, double change_tol=1.e-10, double pde_residual_tol=0)
  {
    nonlinear_method_           = method;
    nonlinear_itmax_            = itmax;
    nonlinear_change_tol_       = change_tol;
    nonlinear_pde_residual_tol_ = pde_residual_tol;
  }

  inline void set_dirichlet_scheme (int    val) { dirichlet_scheme_ = val; }
  inline void set_gf_order         (int    val) { gf_order_         = val; }
  inline void set_gf_stabilized    (int    val) { gf_stabilized_    = val; }
  inline void set_gf_thresh        (double val) { gf_thresh_        = val; }
};

#endif // MY_P4EST_POISSON_NODES_MLS_H
