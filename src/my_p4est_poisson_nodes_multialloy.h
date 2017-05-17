#ifndef MY_P4EST_POISSON_NODES_MULTIALLOY_H
#define MY_P4EST_POISSON_NODES_MULTIALLOY_H

#include <petsc.h>
#include <src/types.h>
#include <src/casl_math.h>

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_voronoi.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

class my_p4est_poisson_nodes_multialloy_t
{
  PetscErrorCode ierr;
  struct vec_and_ptr_t
  {
    static PetscErrorCode ierr;
    Vec vec;
    double *ptr;

    vec_and_ptr_t() : vec(NULL), ptr(NULL) {}

    inline void get_array()
    {
      ierr = VecGetArray(vec, &ptr); CHKERRXX(ierr);
    }
    inline void restore_array()
    {
      ierr = VecRestoreArray(vec, &ptr); CHKERRXX(ierr);
    }
  };


  struct vec_and_ptr_dim_t
  {
    static PetscErrorCode ierr;
    Vec vec[P4EST_DIM];
    double *ptr[P4EST_DIM];

    vec_and_ptr_dim_t() {
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        vec[dim] = NULL; ptr[dim] = NULL;
      }
    }

    inline void get_array()
    {
      for (short dim = 0; dim < P4EST_DIM; ++dim)
      {
        ierr = VecGetArray(vec[dim], &ptr[dim]); CHKERRXX(ierr);
      }
    }
    inline void restore_array()
    {
      for (short dim = 0; dim < P4EST_DIM; ++dim)
      {
        ierr = VecRestoreArray(vec[dim], &ptr[dim]); CHKERRXX(ierr);
      }
    }
  };


  class zero_cf_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero_cf_;


  class jump_psi_tn_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return -1./t_diff_/ml0_;
    }
  } jump_psi_tn_;


  class vn_from_c0_t : public CF_2
  {
  public:
    double operator()(double x, double y) const
    {
      // OPTIMIZE THIS!!!
      interp_.set_input(c0_.vec, c0_dd_.vec[0], c0_dd_.vec[1], quadratic);
      double c0 = interp(x, y);

      interp_.set_input(c0n_.vec, c0n_dd_.vec[0], c0n_dd_.vec[1], quadratic);
      double c0n = interp(x, y);

      return Dl0_/(1.-kp0_)*c0n/c0;
    }
  } vn_from_c0_;

  class c1_robin_coef_t : public CF_2
  {
  public:
    double operator()(double x, double y) const
    {
      return -(1.-kp1_)/Dl1_*vn_from_c0_(x,y);
    }
  } c1_robin_coef_;

  class tn_jump_t : public CF_2
  {
  public:
    double operator()(double x, double y) const
    {
      return latent_heat_/t_cond_*vn_from_c0_(x,y);
    }
  } tn_jump_;



  my_p4est_interpolation_nodes_t interp_;

  my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  // level-set function
  vec_and_ptr_t phi_;
  vec_and_ptr_dim_t phi_dd_;

  vec_and_ptr_t theta_;
  vec_and_ptr_t kappa_;

  vec_and_ptr_dim_t normal_;
//  vec_and_ptr_dim_t normal_dd_[P4EST_DIM];

  bool is_phi_dd_owned_;
  bool is_normal_owned_;

  double dx_min_, dy_min_, dz_min_, d_min_, diag_min_;

  // boundary condtions
#ifdef P4_TO_P8
  BoundaryConditions3D bc_t_;
  BoundaryConditions3D bc_c0_;
  BoundaryConditions3D bc_c1_;
#else
  BoundaryConditions2D bc_t_;
  BoundaryConditions2D bc_c0_;
  BoundaryConditions2D bc_c1_;
#endif

  // gibbs-thompson relation
  CF_2 *GT_;
  CF_1 *eps_v_;
  CF_1 *eps_c_;

  // parameters of alloy
  int num_comps_;
  double dt_;
  double t_diff_, t_cond_, latent_heat_, Tm_;
  double Dl0_, kp0_, ml0_;
  double Dl1_, kp1_, ml1_;

  vec_and_ptr_t bc_error_;
  double bc_error_max_;
  double bc_tolerance_;
  int pin_every_n_steps_;
  int max_iterations_;

  // solvers
  my_p4est_poisson_nodes_t *solver_t;
  my_p4est_poisson_nodes_t *solver_c0;
  my_p4est_poisson_nodes_t *solver_c1;
  my_p4est_poisson_nodes_t *solver_psi_c0;

  bool is_t_matrix_computed_;
  bool is_c1_matrix_computed_;

  // solution
  vec_and_ptr_t t_;  vec_and_ptr_dim_t t_dd_;
  vec_and_ptr_t c0_; vec_and_ptr_dim_t c0_dd_;
  vec_and_ptr_t c1_; vec_and_ptr_dim_t c1_dd_;

  // lagrangian multipliers
  vec_and_ptr_t psi_t_;  vec_and_ptr_dim_t psi_t_dd_;
  vec_and_ptr_t psi_c0_; vec_and_ptr_dim_t psi_c0_dd_;
  vec_and_ptr_t psi_c1_; vec_and_ptr_dim_t psi_c1_dd_;

  // velocity related quatities
  vec_and_ptr_t c0n_;     vec_and_ptr_dim_t c0n_dd_;
  vec_and_ptr_t psi_c0n_; vec_and_ptr_dim_t psi_c0n_dd_;
//  vec_and_ptr_t dc0_    [P4EST_DIM]; vec_and_ptr_dim_t dc0_dd_    [P4EST_DIM];
//  vec_and_ptr_t psi_dc0_[P4EST_DIM]; vec_and_ptr_dim_t psi_dc0_dd_[P4EST_DIM];

  // rhs
  vec_and_ptr_t rhs_tl_;
  vec_and_ptr_t rhs_ts_;
  vec_and_ptr_t rhs_c0_;
  vec_and_ptr_t rhs_c1_;


//  // pointers
//  double *t_ptr_, *t_dd_ptr_[P4EST_DIM];
//  double *c0_ptr_, *c0_dd_ptr_[P4EST_DIM];
//  double *c1_ptr_, *c1_dd_ptr_[P4EST_DIM];

//  double *psi_t_ptr_, *psi_t_dd_ptr_[P4EST_DIM];
//  double *psi_c0_ptr_, *psi_c0_dd_ptr_[P4EST_DIM];
//  double *psi_c1_ptr_, *psi_c1_dd_ptr_[P4EST_DIM];

//  double *dc0_ptr_    [P4EST_DIM], *dc0_dd_ptr_    [P4EST_DIM][P4EST_DIM];
//  double *psi_dc0_ptr_[P4EST_DIM], *psi_dc0_dd_ptr_[P4EST_DIM][P4EST_DIM];

//  double *phi_ptr_;
//  double *phi_dd_ptr_[P4EST_DIM];

//  double *normal_ptr_[P4EST_DIM];
//  double *normal_dd_ptr_[P4EST_DIM][P4EST_DIM];

  // mask for extension
  Vec mask_;

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_multialloy_t(const my_p4est_poisson_nodes_t& other);
  my_p4est_poisson_nodes_multialloy_t& operator=(const my_p4est_poisson_nodes_t& other);

public:
  my_p4est_poisson_nodes_multialloy_t(my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_multialloy_t();

  void set_phi(Vec phi, Vec* phi_dd, Vec* normal, Vec kappa, Vec theta);

  inline void set_parameters(int num_comps, double dt,
                             double thermal_diffusivity, double thermal_conductivity, double latent_heat, double Tm,
                             double Dl0, double kp0, double ml0,
                             double Dl1, double kp1, double ml1) {
    num_comps_ = num_comps;
    dt_ = dt;
    t_diff_ = thermal_diffusivity;
    t_cond_ = thermal_conductivity;
    latent_heat_ = latent_heat;
    Tm_ = Tm;
    Dl0_ = Dl0; kp0_ = kp0; ml0_ = ml0;
    Dl1_ = Dl1; kp1_ = kp1; ml1_ = ml1;
  }

  inline void set_undercoolings(CF_1& eps_v, CF_1& eps_c) {eps_v_ = &eps_v; eps_c_ = &eps_c;}
  inline void set_GT(CF_2& GT_cf) {GT_ = &GT_cf;}

  inline void set_rhs(Vec rhs_tl, Vec rhs_ts, Vec rhs_c0, Vec rhs_c1) {rhs_tl_.vec = rhs_tl; rhs_ts_.vec = rhs_ts; rhs_c0_.vec = rhs_c0; rhs_c1_.vec = rhs_c1;}
  inline void set_bc(BoundaryConditions2D bc_t, BoundaryConditions2D bc_c0, BoundaryConditions2D bc_c1) {bc_t_ = bc_t; bc_c0_ = bc_c0; bc_c1_ = bc_c1;}
  inline void set_tolerance(double bc_tolerance, int max_iterations = 10) {bc_tolerance_ = bc_tolerance; max_iterations_ = max_iterations;}
  inline void set_pin_every_n_steps(int pin_every_n_steps) {pin_every_n_steps_ = pin_every_n_steps;}

  void initialize_solvers();

  void solve(Vec t, Vec t_dd[P4EST_DIM], Vec c0, Vec c0_dd[P4EST_DIM], Vec c1, Vec c1_dd[P4EST_DIM], double& bc_error_max, Vec& bc_error);

  void solve_t();
  void solve_psi_t();

  void solve_c0();
  void solve_psi_c0();

  void solve_c1();
  void solve_psi_c1();

  void compute_c0n();
  void compute_psi_c0n();
  void adjust_c0_gamma(int iteration);

};

#endif // MY_P4EST_POISSON_NODES_MULTIALLOY_H
