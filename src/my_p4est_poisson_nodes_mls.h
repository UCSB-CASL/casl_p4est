#ifndef MY_P4EST_POISSON_NODES_MLS_H
#define MY_P4EST_POISSON_NODES_MLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/my_p8est_utils.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_utils.h>
#endif

#include <vector>

#include <src/cube3_mls.h>
#include <src/cube2_mls.h>

#include <src/cube3_refined_mls.h>
#include <src/cube2_refined_mls.h>

//class pointer_to_vec_t
//{
//  double *p;
//  static PetscError ierr;

//  pointer_to_vec_t(Vec vec) {ierr = VecGetArray(vec, &p);     CHKERRXX(ierr);}
//  ~pointer_to_vec_t()       {ierr = VecRestoreArray(vec, &p); CHKERRXX(ierr);}

//  double operator () (p4est_locidx_t n) {return p[n];}
//};

class my_p4est_poisson_nodes_mls_t
{
public:
  enum node_loc_t {NODE_INS, NODE_NMN, NODE_OUT};

  struct quantity_t
  {
    bool is_constant;
    double constant;

    bool is_vec;
    Vec vec;
    double *vec_p;

    bool is_cf;
#ifdef P4_TO_P8
    CF_3 *cf;
#else
    CF_2 *cf;
#endif

    quantity_t() : constant(0), is_constant(true),
      vec(NULL), vec_p(NULL), is_vec(false),
      cf(NULL), is_cf(false) {}

    void initialize()
    {
      PetscErrorCode ierr;
      if (is_vec)
      {
        ierr = VecGetArray(vec, &vec_p); CHKERRXX(ierr);
      }
    }

    void finalize()
    {
      PetscErrorCode ierr;
      if (is_vec)
      {
        ierr = VecRestoreArray(vec, &vec_p); CHKERRXX(ierr);
        vec_p = NULL;
      }
    }

    void set(double val_) {constant = val_; is_constant = true; is_vec = false; is_cf = false;}
    void set(Vec    vec_) {vec = vec_;      is_constant = false; is_vec = true; is_cf = false;}
#ifdef P4_TO_P8
    void set(CF_3   &cf_) {cf = &cf_;       is_constant = false; is_vec = false; is_cf = true;}
#else
    void set(CF_2   &cf_) {cf = &cf_;       is_constant = false; is_vec = false; is_cf = true;}
#endif

//    double operator() (int n)
//    {
//      if (constant) {return val;}
//      else          {return vec_p[n];}
//    }
  };

  // p4est objects
  const my_p4est_node_neighbors_t *node_neighbors;

  p4est_t           *p4est;
  p4est_nodes_t     *nodes;
  p4est_ghost_t     *ghost;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t phi_interp;
  my_p4est_interpolation_nodes_local_t interp_local;

  bool    is_matrix_computed;
  int     matrix_has_nullspace;
  double  dx_min, dy_min, d_min, diag_min, vol_min;
#ifdef P4_TO_P8
  double  dz_min;
#endif

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // Equation
  quantity_t mu, wall_value, diag_add;
  Vec rhs;

  // Geometry
  std::vector<Vec> *phi, *phi_xx, *phi_yy, *phi_zz;
  std::vector<int>        *color;
  std::vector<action_t>   *action;
  Vec phi_eff;
  int n_phis;
  bool phi_eff_owned, phi_dd_owned;
  Vec node_vol;

  bool use_taylor_correction;

  // Interfaces
  std::vector<BoundaryConditionType> *bc_types;
  std::vector<quantity_t> bc_values, bc_coeffs;

  // PETSc objects
  Mat A;
  Vec RHS;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;

  bool keep_scalling;
  Vec scalling;

  KSP ksp;
  PetscErrorCode ierr;

  // type of nodes
  std::vector<node_loc_t>  node_loc;

  void preallocate_matrix();

  void setup_negative_laplace_matrix_sym();
  void setup_negative_laplace_rhsvec_sym();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_t(const my_p4est_poisson_nodes_mls_t& other);
  my_p4est_poisson_nodes_mls_t& operator=(const my_p4est_poisson_nodes_mls_t& other);

  enum node_neighbor_t
  {
#ifdef P4_TO_P8
    // zm plane
    nn_mmm = 0, nn_0mm, nn_pmm,
    nn_m0m, nn_00m, nn_p0m,
    nn_mpm, nn_0pm, nn_ppm,

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

  double eps_ifc, eps_dom;

  bool kink_special_treatment;


//public:


  my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_t();

  // set geometry
  void set_geometry(std::vector<Vec> &phi_,
                    std::vector<Vec> &phi_xx_,
                    std::vector<Vec> &phi_yy_,
                    #ifdef P4_TO_P8
                    std::vector<Vec> &phi_zz_,
                    #endif
                    std::vector<action_t> &action_, std::vector<int> &color_,
                    Vec phi_eff_ = NULL);

  void set_geometry(std::vector<Vec> &phi_,
                    std::vector<action_t> &action_, std::vector<int> &color_,
                    Vec phi_eff_ = NULL);

  void compute_phi_eff();
  void compute_phi_dd();
  void compute_volumes();

  // set BCs
  void set_bc_type(std::vector<BoundaryConditionType> &bc_types_) {bc_types = &bc_types_;}

  void set_bc_values(std::vector<Vec> &bc_vecs)
  {
    bc_values.resize(bc_vecs.size());
    for (int i = 0; i < bc_vecs.size(); i++)
      bc_values[i].set(bc_vecs[i]);
  }
  void set_bc_values(std::vector<double> &bc_vals)
  {
    bc_values.resize(bc_vals.size());
    for (int i = 0; i < bc_vals.size(); i++)
      bc_values[i].set(bc_vals[i]);
  }

  void set_bc_coeffs(std::vector<Vec> &bc_vecs)
  {
    bc_coeffs.resize(bc_vecs.size());
    for (int i = 0; i < bc_vecs.size(); i++)
      bc_coeffs[i].set(bc_vecs[i]);
  }
  void set_bc_coeffs(std::vector<double> &bc_vals)
  {
    bc_coeffs.resize(bc_vals.size());
    for (int i = 0; i < bc_vals.size(); i++)
      bc_coeffs[i].set(bc_vals[i]);
  }

  void set_diag_add(double val) { diag_add.set(val); }
  void set_diag_add(Vec val)    { diag_add.set(val); }

  void set_mu(double val) { mu.set(val); }
  void set_mu(Vec val)    { mu.set(val); }

  void set_wall_value(double val) { wall_value.set(val); }
  void set_wall_value(Vec val)    { wall_value.set(val); }

  void set_rhs(Vec &rhs_) {rhs = rhs_;}

  void set_use_taylor_correction(bool val) {use_taylor_correction = val;}

  void set_keep_scalling(bool keep_scalling_)    {keep_scalling = keep_scalling_;}

  void set_is_matrix_computed(bool is_matrix_computed)   {is_matrix_computed = is_matrix_computed; }

  void set_kinks_treatment(bool in) {kink_special_treatment = in;}

  void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT)
  {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  bool is_calc(int n) {
    if (node_loc[n] == NODE_INS || node_loc[n] == NODE_NMN) return true;
    else return false;
  }

  bool is_inside(int n) {
    if (node_loc[n] == NODE_INS) return true;
    else return false;
  }

  void inv_mat2(double *in, double *out);
  void inv_mat3(double *in, double *out);

  int cube_refinement;
  void set_cube_refinement(int r) {cube_refinement = r;}

  void find_projection(double *phi_p, p4est_locidx_t *neighbors, bool *neighbor_exists, double dxyz_pr[], double &dist_pr);
  void find_projection(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr);

  void compute_normal(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[]);

  void sample_vec_at_neighbors(double *in_p, int *neighbors, bool *neighbor_exists, double *output);
  void sample_vec_at_neighbors(double *in_p, int *neighbors, bool *neighbor_exists, std::vector<double> &output);
  void sample_qty_at_neighbors(quantity_t &qty, int *neighbors, bool *neighbor_exists, double *output);

  double sample_qty(quantity_t &qty, double *xyz);
  double sample_qty(quantity_t &qty, p4est_locidx_t n);
  double sample_qty(quantity_t &qty, p4est_locidx_t n, double *xyz);

  void get_all_neighbors(p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists);

#ifdef P4_TO_P8
  void compute_error_sl(CF_3 &exact_cf, Vec sol, Vec err);
  void compute_error_tr(CF_3 &exact_cf, Vec error);
  void compute_error_gr(CF_3 &ux_cf, CF_3 &uy_cf, CF_3 &uz_cf, Vec sol, Vec err_ux, Vec err_uy, Vec err_uz);
#else
  void compute_error_sl(CF_2 &exact_cf, Vec sol, Vec err);
  void compute_error_tr(CF_2 &exact_cf, Vec error);
  void compute_error_gr(CF_2 &ux_cf, CF_2 &uy_cf, Vec sol, Vec err_ux, Vec err_uy);
//  void compute_error_xy(CF_2 &uxy_cf, Vec sol, Vec err_uxy);
#endif

//  double interpolate_near_node_linear   (double *in_p, p4est_locidx_t *nei_quads, bool *nei_quad_exists, double x, double y, double z);
//  double interpolate_near_node_quadratic(double *in_p, double *inxx_p, double *inyy_p, double *inzz_p, p4est_locidx_t *nei_quads, bool *nei_quad_exists, double x, double y, double z);

#ifdef P4_TO_P8
  double sample_qty_on_uni_grid(grid_interpolation3_t &grid, double *output, quantity_t &qty);
#else
  double sample_qty_on_uni_grid(grid_interpolation2_t &grid, double *output, quantity_t &qty);
#endif

#ifdef P4_TO_P8
  double compute_qty_avg_over_iface(cube3_mls_t &cube, int color, quantity_t &qty);
#else
  double compute_qty_avg_over_iface(cube2_mls_t &cube, int color, quantity_t &qty);
#endif

};

#endif // MY_P4EST_POISSON_NODES_MLS_H
