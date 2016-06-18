#ifndef MY_P4EST_POISSON_NODES_MLS_H
#define MY_P4EST_POISSON_NODES_MLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#endif

#include <vector>


class my_p4est_poisson_nodes_mls_t
{
//public:

  struct quantity_t
  {
    double val;
    double *vec_p;
    Vec vec;
#ifdef P4_TO_P8
    CF_3 *cf;
#else
    CF_2 *cf;
#endif
    quantity_t() : val(0), vec(NULL), cf(NULL), vec_p(NULL) {}

    void initialize()
    {
      if (cf == NULL && vec != NULL)
      {
        ierr = VecGetArray(vec, &vec_p); CHKERRXX(ierr);
      }
    }

    void finalize()
    {
      if (vec_p != NULL)
      {
        ierr = VecRestoreArray(vec, &vec_p); CHKERRXX(ierr);
        vec_p = NULL;
      }
    }

#ifdef P4_TO_P8
    double operator() (int n, double x, double y, double z)
    {
      if (cf != NULL)         {return (*cf)(x,y);}
#else
    double operator() (int n, double x, double y)
    {
      if (cf != NULL)         {return (*cf)(x,y);}
#endif
      else if (vec_p != NULL) {return vec_p[n];}
      else                    {return val;}
    }
  };


  // p4est objects
  const my_p4est_node_neighbors_t *node_neighbors_;

  p4est_t           *p4est;
  p4est_nodes_t     *nodes;
  p4est_ghost_t     *ghost;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t phi_interp;

  double  mu_, diag_add_;
  bool    is_matrix_computed;
  int     matrix_has_nullspace;
  double  dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double  dz_min;
#endif

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  quantity_t force, mu, wall_value;
  std::vector<quantity_t> phi, phi_xx, phi_yy, phi_zz;
  std::vector<quantity_t> interface_value, robin_coef;

  // Interfaces
  std::vector<int>        *color_;
  std::vector<action_t>   *action_;
  std::vector<BoundaryConditionType> *bc_types;

  // Additional diagonal term

  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  bool is_phi_dd_owned, is_mue_dd_owned;

  bool keep_scalling;
  Vec scalling;

  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();

  void setup_negative_laplace_matrix_non_sym();
  void setup_negative_laplace_rhsvec_non_sym();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_t(const my_p4est_poisson_nodes_mls_t& other);
  my_p4est_poisson_nodes_mls_t& operator=(const my_p4est_poisson_nodes_mls_t& other);

public:

  Vec *phi_eff_;

  enum node_loc_t {NODE_INS,NODE_DIR,NODE_MXI,NODE_MXO,NODE_NMN,NODE_OUT};

  enum node_neighbor_t
  {
    nn_000 = -1,
    nn_m00, nn_p00,nn_0m0,nn_0p0,
#ifdef P4_TO_P8
    nn_00m, nn_00p,
#endif
    nn_mm0, nn_pm0, nn_mp0, nn_pp0
#ifdef P4_TO_P8
    ,
    nn_m0m, nn_p0m, nn_m0p, nn_p0p,
    nn_0mm, nn_0pm, nn_0mp, nn_0pp,
    nn_mmm, nn_pmm, nn_mpm, nn_ppm,
    nn_mmp, nn_pmp, nn_mpp, nn_ppp
#endif
  };

  std::vector<node_loc_t>  node_loc;

  my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_t();

#ifdef P4_TO_P8
  void set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx = NULL, std::vector<Vec> *phi_yy = NULL, std::vector<Vec> *phi_zz = NULL);
#else
  void set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx = NULL, std::vector<Vec> *phi_yy = NULL);
#endif

  inline void set_action(std::vector<action_t>& action) {action_  = &action;}
  inline void set_color(std::vector<int>& color)        {color_   = &color;}

  inline void set_keep_scalling(bool keep_scalling_)    {keep_scalling = keep_scalling_;}

#ifdef P4_TO_P8
  inline void set_force(CF_3 &force) {force_ = &force;}
#else
  inline void set_force(CF_2 &force) {force_ = &force;}
#endif

#ifdef P4_TO_P8
  inline void set_phi_cf(std::vector<CF_3*> &phi_cf) {phi_cf_ = &phi_cf;}
#else
  inline void set_phi_cf(std::vector<CF_2*> &phi_cf) {phi_cf_ = &phi_cf;}
#endif

  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add;  is_matrix_computed = false;}
  inline void set_diagonal(Vec add)            {add_      = add;  is_matrix_computed = false;}

#ifdef P4_TO_P8
  inline void set_bc(std::vector<BoundaryConditions3D>& bc) {bc_ = &bc; is_matrix_computed = false;}
#else
  inline void set_bc(std::vector<BoundaryConditions2D>& bc) {bc_ = &bc; is_matrix_computed = false;}
#endif

  inline void set_robin_coef(std::vector<Vec>& robin_coef)      {robin_coef_ = &robin_coef; is_matrix_computed = false;}
  inline void set_mu(double mu)                                 {mu_       = mu;            is_matrix_computed = false;}
  inline void set_is_matrix_computed(bool is_matrix_computed)   { this->is_matrix_computed = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }


  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

#ifdef P4_TO_P8
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL, Vec mu_zz = NULL);
#else
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL);
#endif

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  bool is_calc(int n) {if (node_loc[n] == NODE_INS || node_loc[n] == NODE_NMN) return true; else return false;}
  bool is_inside(int n) {if (node_loc[n] == NODE_INS) return true; else return false;}
//  bool is_inside(int n) {if (node_loc[n] == NODE_NMN) return true; else return false;}
//  bool is_inside(int n) {if (node_loc[n] == NODE_INS) return true; else return false;}

//  std::vector<double> node_vol;
//  double node_volume(int n) {return node_vol[n];}
  //
//  void construct_domain();

  void inv_mat2(double *in, double *out);
  void inv_mat3(double *in, double *out);

  void find_centroid(bool &node_in, bool &altered, double &x, double &y, p4est_locidx_t n, double *vol = NULL);

  std::vector<double> node_vol;

  double calculate_trunc_error(CF_2 &exact);
  void calculate_gradient_error(Vec sol, Vec err_ux, Vec err_uy, CF_2 &ux, CF_2 &uy);
  void calculate_equation_error(Vec sol, Vec err_eq);

  int cube_refinement;
  void set_cube_refinement(int r) {cube_refinement = r;}

  void set_phidd_cf(std::vector<CF_2 *> &phixx_cf, std::vector<CF_2 *> &phixy_cf, std::vector<CF_2 *> &phiyy_cf)
  {
    phixx_cf_ = &phixx_cf; phixy_cf_ = &phixy_cf; phiyy_cf_ = &phiyy_cf;
  }

  void set_phid_cf(std::vector<CF_2 *> &phix_cf, std::vector<CF_2 *> &phiy_cf)
  {
    phix_cf_ = &phix_cf; phiy_cf_ = &phiy_cf;
  }
};

#endif // MY_P4EST_POISSON_NODES_MLS_H
