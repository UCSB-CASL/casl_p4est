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

#include <src/cube3_mls.h>
#include <src/cube2_mls.h>

#include <src/cube3_refined_mls.h>
#include <src/cube2_refined_mls.h>


class my_p4est_poisson_nodes_mls_t
{
public:
  enum node_loc_t {NODE_INS,NODE_NMN,NODE_OUT};

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
      PetscErrorCode ierr;
      if (cf == NULL && vec != NULL)
      {
        ierr = VecGetArray(vec, &vec_p); CHKERRXX(ierr);
      }
    }

    void finalize()
    {
      PetscErrorCode ierr;
      if (vec_p != NULL)
      {
        ierr = VecRestoreArray(vec, &vec_p); CHKERRXX(ierr);
        vec_p = NULL;
      }
    }

    inline void set(double &val_) {val = val_;}
    inline void set(Vec    &vec_) {vec = vec_;}
#ifdef P4_TO_P8
    inline void set(CF_3   &cf_)  {cf  = &cf_;}
#else
    inline void set(CF_2   &cf_)  {cf  = &cf_;}
#endif

#ifdef P4_TO_P8
    double operator() (int n, double &x, double &y, double &z)
    {
      if (cf != NULL)         {return (*cf)(x,y,z);}
#else
    double operator() (int n, double &x, double &y)
    {
      if (cf != NULL)         {return (*cf)(x,y);}
#endif
      else if (vec_p != NULL) {return vec_p[n];}
      else                    {return val;}
    }
  };

  struct vec_quantity_t
  {
    std::vector<double>   *val;
    std::vector<Vec>      *vec;
    std::vector<double *> vec_p;
#ifdef P4_TO_P8
    std::vector<CF_3 *>   *cf;
#else
    std::vector<CF_2 *>   *cf;
#endif

    vec_quantity_t() : val(NULL), vec(NULL), cf(NULL) {}

    void initialize()
    {
      PetscErrorCode ierr;
      if (cf == NULL || vec != NULL)
      {
        int N = vec->size();
        vec_p.resize(N, NULL);
        for (int i = 0; i < N; i++)
        {
          ierr = VecGetArray(vec->at(i), &vec_p[i]); CHKERRXX(ierr);
        }
      }
    }

    void finalize()
    {
      PetscErrorCode ierr;
      if (cf == NULL || vec != NULL)
      {
        int N = vec->size();
        for (int i = 0; i < N; i++)
        {
          ierr = VecRestoreArray(vec->at(i), &vec_p[i]); CHKERRXX(ierr);
          vec_p[i] = NULL;
        }
      }
    }

    inline void set(std::vector<double> &val_) {val = &val_;}
    inline void set(std::vector<Vec>    &vec_) {vec = &vec_;}
#ifdef P4_TO_P8
    inline void set(std::vector<CF_3 *> &cf_)  {cf  = &cf_;}
#else
    inline void set(std::vector<CF_2 *> &cf_)  {cf  = &cf_;}
#endif


#ifdef P4_TO_P8
    double get_value(int i, int n, double &x, double &y, double &z)
    {
      if (cf != NULL)     {return (*cf->at(i))(x,y,z);}
#else
    double get_value(int i, int n, double &x, double &y)
    {
      if (cf != NULL)     {return (*cf->at(i))(x,y);}
#endif
      if (vec != NULL)  {return vec_p[i][n];}
      else                {return val->at(i);}
    }
  };


  // p4est objects
  const my_p4est_node_neighbors_t *node_neighbors;

  p4est_t           *p4est;
  p4est_nodes_t     *nodes;
  p4est_ghost_t     *ghost;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t phi_interp;

  bool    is_matrix_computed;
  int     matrix_has_nullspace;
  double  dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double  dz_min;
#endif

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // Equation
  quantity_t rhs, mu, wall_value, diag_add;

  // Geometry
  std::vector<Vec> *phi, *phi_xx, *phi_yy, *phi_zz;
  std::vector<int>        *color;
  std::vector<action_t>   *action;
  Vec phi_eff;
  int n_phis;
  bool phi_eff_owned, phi_dd_owned;
  Vec node_vol;


  // Interfaces
  std::vector<BoundaryConditionType> *bc_types;
  vec_quantity_t bc_value, bc_coeff;

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

  void setup_negative_laplace_matrix_non_sym();
  void setup_negative_laplace_rhsvec_non_sym();

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

  inline void set_keep_scalling(bool keep_scalling_)    {keep_scalling = keep_scalling_;}

  inline void set_is_matrix_computed(bool is_matrix_computed)   {is_matrix_computed = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT)
  {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  bool is_calc(int n) {if (node_loc[n] == NODE_INS || node_loc[n] == NODE_NMN) return true; else return false;}
  bool is_inside(int n) {if (node_loc[n] == NODE_INS) return true; else return false;}

  void inv_mat2(double *in, double *out);
  void inv_mat3(double *in, double *out);

  void find_centroid(bool &node_in, bool &altered, double &x, double &y, p4est_locidx_t n, double *vol = NULL);

  double calculate_trunc_error(CF_2 &exact);
  void calculate_gradient_error(Vec sol, Vec err_ux, Vec err_uy, CF_2 &ux, CF_2 &uy);
  void calculate_equation_error(Vec sol, Vec err_eq);

  int cube_refinement;
  void set_cube_refinement(int r) {cube_refinement = r;}

};

#endif // MY_P4EST_POISSON_NODES_MLS_H
