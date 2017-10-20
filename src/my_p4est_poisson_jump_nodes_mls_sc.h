#ifndef MY_P4EST_POISSON_JUMP_NODES_MLS_SC_H
#define MY_P4EST_POISSON_JUMP_NODES_MLS_SC_H

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

#include <src/cube3_mls.h>
#include <src/cube2_mls.h>

#include <src/cube3_mls_quadratic.h>
#include <src/cube2_mls_quadratic.h>

#define USE_QUADRATIC_CUBES

class my_p4est_poisson_jump_nodes_mls_sc_t
{
  static const bool use_refined_cube_ = 1;
  static const int cube_refinement_ = 1;
  static const int num_neighbors_max_ = pow(3, P4EST_DIM);

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

#ifdef P4_TO_P8
  class unity_cf_t: public CF_3
  {
  public:
    double operator()(double x, double y, double z) const
    {
      return 1.;
    }
  } unity_cf_;
#else
  class unity_cf_t: public CF_2
  {
  public:
    double operator()(double x, double y) const
    {
      return 1.;
    }
  } unity_cf_;
#endif

#ifdef P4_TO_P8
  class delta_x_cf_t: public CF_3
  {
    double *xyz_;
  public:
    inline void set(double *xyz) { xyz_ = xyz; }
    double operator()(double x, double y, double z) const
    {
      return x - xyz_[0];
    }
  } delta_x_cf_;
#else
  class delta_x_cf_t: public CF_2
  {
    double *xyz_;
  public:
    inline void set(double *xyz) { xyz_ = xyz; }
    double operator()(double x, double y) const
    {
      return x - xyz_[0];
    }
  } delta_x_cf_;
#endif

#ifdef P4_TO_P8
  class delta_y_cf_t: public CF_3
  {
    double *xyz_;
  public:
    inline void set(double *xyz) { xyz_ = xyz; }
    double operator()(double x, double y, double z) const
    {
      return y - xyz_[1];
    }
  } delta_y_cf_;
#else
  class delta_y_cf_t: public CF_2
  {
    double *xyz_;
  public:
    inline void set(double *xyz) { xyz_ = xyz; }
    double operator()(double x, double y) const
    {
      return y - xyz_[1];
    }
  } delta_y_cf_;
#endif

#ifdef P4_TO_P8
  class delta_z_cf_t: public CF_3
  {
    double *xyz_;
  public:
    inline void set(double *xyz) { xyz_ = xyz; }
    double operator()(double x, double y, double z) const
    {
      return z - xyz_[2];
    }
  } delta_z_cf_;
#endif

#ifdef P4_TO_P8
  class restriction_to_yz_t : public CF_2
  {
    CF_3 *f_;
    double x_;
  public:
    restriction_to_yz_t(CF_3 *f, double x) : x_(x), f_(f) {}

    inline double operator()(double y, double z) const
    {
      return f_->operator ()(x_, y, z);
    }
  };

  class restriction_to_zx_t : public CF_2
  {
    CF_3 *f_;
    double y_;
  public:
    restriction_to_zx_t(CF_3 *f, double y) : y_(y), f_(f) {}

    inline double operator()(double z, double x) const
    {
      return f_->operator ()(x, y_, z);
    }
  };

  class restriction_to_xy_t : public CF_2
  {
    CF_3 *f_;
    double z_;
  public:
    restriction_to_xy_t(CF_3 *f, double z) : z_(z), f_(f) {}

    inline double operator()(double x, double y) const
    {
      return f_->operator ()(x, y, z_);
    }
  };
#endif

  const my_p4est_node_neighbors_t *node_neighbors_;

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
  std::vector<Vec> *phi_, *phi_xx_, *phi_yy_, *phi_zz_;
  std::vector<int>        *color_;
  std::vector<action_t>   *action_;
  Vec phi_eff_;
  int num_interfaces_;
  bool is_phi_eff_owned_, is_phi_dd_owned_;

  double dxyz_m_[P4EST_DIM];
  double dx_min_, dy_min_, d_min_, diag_min_;
#ifdef P4_TO_P8
  double dz_min_;
#endif

  // Equation
  double mu_m_, diag_add_scalar_m_;
  double mu_p_, diag_add_scalar_p_;
  Vec diag_add_m_;
  Vec diag_add_p_;
  Vec rhs_m_;
  Vec rhs_p_;
  Vec rhs_block_;
  Vec mue_m_, mue_m_xx_, mue_m_yy_, mue_m_zz_;
  Vec mue_p_, mue_p_xx_, mue_p_yy_, mue_p_zz_;

  bool variable_mu_;
  bool is_mue_dd_owned_;

  // Some flags
  bool is_matrix_computed_;
  int matrix_has_nullspace_;
  bool new_pc_;
  bool update_ghost_after_solving_;

  // Bondary conditions
  bool neumann_wall_first_order_;

#ifdef P4_TO_P8
  WallBC3D *bc_wall_type_;
  CF_3     *bc_wall_value_;

  CF_3 *u_jump_;
  std::vector< CF_3 *> *mu_un_jump_;
#else
  WallBC2D *bc_wall_type_;
  CF_2     *bc_wall_value_;

  CF_2 *u_jump_;
  std::vector< CF_2 *> *mu_un_jump_;
#endif

  Vec mask_m_;
  Vec mask_p_;
  std::vector<double> scalling_;
  bool keep_scalling_;

  Vec volumes_;

  double eps_ifc_, eps_dom_;

  void compute_volumes_();
  void compute_phi_eff_();
  void compute_phi_dd_();
  void compute_mue_dd_();

#ifdef P4_TO_P8
  double compute_weights_through_face(double A, double B, bool *neighbors_exists_2d, double *weights_2d, double theta, bool *map_2d);
#endif

  double find_interface_location_mls(p4est_locidx_t n0, p4est_locidx_t n1, double h,
                                     std::vector<double *> &phi_p,
                                     std::vector<double *> &phi_xx_p);

  void preallocate_matrix();

  void setup_linear_system_(bool setup_matrix, bool setup_rhs);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_nodes_mls_sc_t(const my_p4est_poisson_jump_nodes_mls_sc_t& other);
  my_p4est_poisson_jump_nodes_mls_sc_t& operator=(const my_p4est_poisson_jump_nodes_mls_sc_t& other);

  bool find_x_derivative(bool *neighbors_exist, double *weights, bool *map);
  bool find_y_derivative(bool *neighbors_exist, double *weights, bool *map);
#ifdef P4_TO_P8
  bool find_z_derivative(bool *neighbors_exist, double *weights, bool *map);
#endif

  void preallocate_row(p4est_locidx_t n, const quad_neighbor_nodes_of_node_t& qnnn, std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz);

public:
  my_p4est_poisson_jump_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_jump_nodes_mls_sc_t();

  // set geometry
  inline void set_geometry(int num_interfaces,
                           std::vector<action_t> *action, std::vector<int> *color,
                           std::vector<Vec> *phi,
                           std::vector<Vec> *phi_xx = NULL,
                           std::vector<Vec> *phi_yy = NULL,
                         #ifdef P4_TO_P8
                           std::vector<Vec> *phi_zz = NULL,
                         #endif
                           Vec phi_eff = NULL)
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

    if (phi_eff != NULL)
      phi_eff_ = phi_eff;
    else
      compute_phi_eff_();

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
  inline void set_jumps(CF_3 &u_jump, std::vector< CF_3 *> &mu_un_jump) { u_jump_ = &u_jump; mu_un_jump_ = &mu_un_jump; is_matrix_computed_ = false; }
#else
  inline void set_bc_wall_type(WallBC2D &wall_type) { bc_wall_type_ = &wall_type;}
  inline void set_bc_wall_value(CF_2 &wall_value)   { bc_wall_value_ = &wall_value; }
  inline void set_jumps(CF_2 &u_jump, std::vector< CF_2 *> &mu_un_jump) { u_jump_ = &u_jump; mu_un_jump_ = &mu_un_jump; is_matrix_computed_ = false; }
#endif

  inline void set_diag_add(double diag_add_scalar_m, double diag_add_scalar_p) { diag_add_scalar_m_ = diag_add_scalar_m; diag_add_scalar_p_ = diag_add_scalar_p; is_matrix_computed_ = false; }
  inline void set_diag_add(Vec diag_add_m, Vec diag_add_p) { diag_add_m_ = diag_add_m; diag_add_p_ = diag_add_p; is_matrix_computed_ = false; }

  inline void set_mu(double mu_m, double mu_p) { mu_m_ = mu_m; mu_p_ = mu_p; variable_mu_ = false; }

  void set_mu(Vec mue_m, Vec mue_p)
  {
    mue_m_ = mue_m;
    mue_p_ = mue_p;

    compute_mue_dd_();

    is_mue_dd_owned_ = true;
    variable_mu_ = true;
    is_matrix_computed_ = false;
  }

  inline void set_rhs(Vec rhs_m, Vec rhs_p) { rhs_m_ = rhs_m; rhs_p_ = rhs_p; }

  inline void set_is_matrix_computed(bool is_matrix_computed) { is_matrix_computed_ = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp_, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace_; }

  inline void set_first_order_neumann_wall(bool value) { neumann_wall_first_order_  = value; }
  inline void set_keep_scalling           (bool value) { keep_scalling_             = value; }

  void inv_mat2_(double *in, double *out);
  void inv_mat3_(double *in, double *out);

//  void find_projection_(double *phi_p, p4est_locidx_t *neighbors, bool *neighbor_exists, double dxyz_pr[], double &dist_pr);
  void find_projection_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr);

  void compute_normal_(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[]);

  void get_all_neighbors_(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists);

//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCASM);
  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCGASM);
//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPGMRES, PCType pc_type = PCASM);
//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPGMRES, PCType pc_type = PCJACOBI);
//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPFGMRES, PCType pc_type = PCEXOTIC);
//  void solve(Vec sol_m, Vec sol_p, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCGAMG);

  void assemble_matrix(Vec solution);

  inline Vec get_mask_m() { return mask_m_; }
  inline Vec get_mask_p() { return mask_p_; }

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

  inline Vec get_rhs() { return rhs_block_; }

  inline std::vector<double>* get_scalling() { return &scalling_; }

  inline void assemble_rhs_only() { setup_linear_system_(false, true); }

  bool use_sc_scheme_;
  inline void set_use_sc_scheme(bool value) { use_sc_scheme_ = value; }

};

#endif // MY_P4EST_POISSON_JUMP_NODES_MLS_SC_H
