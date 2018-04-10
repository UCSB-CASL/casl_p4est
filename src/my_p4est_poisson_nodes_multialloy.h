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
#include <src/my_p8est_poisson_nodes_mls_sc.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_mls_sc.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

class my_p4est_poisson_nodes_multialloy_t
{
  PetscErrorCode ierr;

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

  vec_and_ptr_t volumes_;

  std::vector<Vec> phi_vector_;
  std::vector<Vec> phi_xx_vector_;
  std::vector<Vec> phi_yy_vector_;
#ifdef P4_TO_P8
  std::vector<Vec> phi_zz_vector_;
#endif
  std::vector<action_t> action_;
  std::vector<int> color_;

  vec_and_ptr_t theta_xz_;
#ifdef P4_TO_P8
  vec_and_ptr_t theta_yz_;
#endif
  vec_and_ptr_t kappa_;

  vec_and_ptr_dim_t normal_;

  bool is_phi_dd_owned_;
  bool is_normal_owned_;

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

#ifdef P4_TO_P8
  CF_3 *eps_c_;
  CF_3 *eps_v_;
#else
  CF_2 *eps_c_;
  CF_2 *eps_v_;
#endif

#ifdef P4_TO_P8
  CF_3 *GT_;
  CF_3 *jump_t_;
  CF_3 *c0_flux_;
  CF_3 *c1_flux_;
  CF_3 *c0_guess_;
#else
  CF_2 *GT_;
  CF_2 *jump_t_;
  CF_2 *c0_flux_;
  CF_2 *c1_flux_;
  CF_2 *c0_guess_;
#endif

  // DEBUG
#ifdef P4_TO_P8
  CF_3 *vn_exact_;
#else
  CF_2 *vn_exact_;
#endif


  // parameters of alloy
  double dt_;
  double t_diff_, t_cond_, latent_heat_, Tm_;
  double Dl0_, kp0_, ml0_;
  double Dl1_, kp1_, ml1_;

  vec_and_ptr_t bc_error_;
  double bc_error_max_;
  double bc_tolerance_;
  int pin_every_n_steps_;
  int max_iterations_;

  double velo_max_;

  bool use_refined_cube_;

  // solvers
  my_p4est_poisson_nodes_t *solver_t;
  my_p4est_poisson_nodes_t *solver_c0;
  my_p4est_poisson_nodes_t *solver_c1;
  my_p4est_poisson_nodes_t *solver_psi_c0;

  my_p4est_poisson_nodes_mls_sc_t *solver_c1_sc;

  bool is_t_matrix_computed_;
  bool is_c1_matrix_computed_;

  // solution
  bool second_derivatives_owned_;
  vec_and_ptr_t t_;  vec_and_ptr_dim_t t_dd_;
  vec_and_ptr_t c0_; vec_and_ptr_dim_t c0_dd_;
  vec_and_ptr_t c1_; vec_and_ptr_dim_t c1_dd_;

  vec_and_ptr_t c0_gamma_;
  vec_and_ptr_t c0n_gamma_;

  vec_and_ptr_t tm_; vec_and_ptr_dim_t tm_dd_;

  // lagrangian multipliers
  vec_and_ptr_t psi_t_;  vec_and_ptr_dim_t psi_t_dd_;
  vec_and_ptr_t psi_c0_; vec_and_ptr_dim_t psi_c0_dd_;
  vec_and_ptr_t psi_c1_; vec_and_ptr_dim_t psi_c1_dd_;

  // velocity related quatities
  vec_and_ptr_t c0n_;     vec_and_ptr_dim_t c0n_dd_;
  vec_and_ptr_t psi_c0n_; vec_and_ptr_dim_t psi_c0n_dd_;

  // rhs
  vec_and_ptr_t rhs_tl_;
  vec_and_ptr_t rhs_ts_;
  vec_and_ptr_t rhs_c0_;
  vec_and_ptr_t rhs_c1_;

  bool use_continuous_stencil_;
  bool use_one_sided_derivatives_;
  bool use_superconvergent_robin_;
  bool use_superconvergent_jump_;
  bool update_c0_robin_;
  bool use_points_on_interface_;

  bool use_non_zero_guess_;

  double min_volume_;
  double volume_thresh_;

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_multialloy_t(const my_p4est_poisson_nodes_t& other);
  my_p4est_poisson_nodes_multialloy_t& operator=(const my_p4est_poisson_nodes_t& other);

public:
  my_p4est_poisson_nodes_multialloy_t(my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_multialloy_t();

  void set_phi(Vec phi, Vec* phi_dd, Vec* normal, Vec kappa);

  inline void set_parameters(double dt,
                             double thermal_diffusivity, double thermal_conductivity, double latent_heat, double Tm,
                             double Dl0, double kp0, double ml0,
                             double Dl1, double kp1, double ml1) {
    dt_ = dt;
    t_diff_ = thermal_diffusivity;
    t_cond_ = thermal_conductivity;
    latent_heat_ = latent_heat;
    Tm_ = Tm;
    Dl0_ = Dl0; kp0_ = kp0; ml0_ = ml0;
    Dl1_ = Dl1; kp1_ = kp1; ml1_ = ml1;
  }

#ifdef P4_TO_P8
  inline void set_GT(CF_3& GT_cf) {GT_ = &GT_cf;}
#else
  inline void set_GT(CF_2& GT_cf) {GT_ = &GT_cf;}
#endif

#ifdef P4_TO_P8
  inline void set_undercoolings(CF_3& eps_v, CF_3& eps_c) {eps_v_ = &eps_v; eps_c_ = &eps_c;}
#else
  inline void set_undercoolings(CF_2& eps_v, CF_2& eps_c) {eps_v_ = &eps_v; eps_c_ = &eps_c;}
#endif

  inline void set_rhs(Vec rhs_tl, Vec rhs_ts, Vec rhs_c0, Vec rhs_c1)
  {
    rhs_tl_.vec = rhs_tl;
    rhs_ts_.vec = rhs_ts;
    rhs_c0_.vec = rhs_c0;
    rhs_c1_.vec = rhs_c1;
  }

#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D bc_t, BoundaryConditions3D bc_c0, BoundaryConditions3D bc_c1)
#else
  inline void set_bc(BoundaryConditions2D bc_t, BoundaryConditions2D bc_c0, BoundaryConditions2D bc_c1)
#endif
  {
    bc_t_  = bc_t;
    bc_c0_ = bc_c0;
    bc_c1_ = bc_c1;
  }

  inline void set_tolerance(double bc_tolerance, int max_iterations = 10)
  {
    bc_tolerance_ = bc_tolerance;
    max_iterations_ = max_iterations;
  }

  inline void set_pin_every_n_steps(int pin_every_n_steps) { pin_every_n_steps_ = pin_every_n_steps; }

#ifdef P4_TO_P8
  inline void set_jump_t(CF_3& jump_t) { jump_t_ = &jump_t; }
  inline void set_flux_c(CF_3& c0_flux, CF_3& c1_flux)
#else
  inline void set_jump_t(CF_2& jump_t) { jump_t_ = &jump_t; }
  inline void set_flux_c(CF_2& c0_flux, CF_2& c1_flux)
#endif
  {
    c0_flux_ = &c0_flux;
    c1_flux_ = &c1_flux;
  }

#ifdef P4_TO_P8
  inline void set_c0_guess(CF_3& c0_guess) { c0_guess_ = &c0_guess; }
#else
  inline void set_c0_guess(CF_2& c0_guess) { c0_guess_ = &c0_guess; }
#endif

#ifdef P4_TO_P8
  inline void set_vn(CF_3& vn) { vn_exact_ = &vn; }
#else
  inline void set_vn(CF_2& vn) { vn_exact_ = &vn; }
#endif

  inline void set_use_continuous_stencil   (bool value) { use_continuous_stencil_    = value;}
  inline void set_use_one_sided_derivatives(bool value) { use_one_sided_derivatives_ = value;}
  inline void set_use_superconvergent_robin(bool value) { use_superconvergent_robin_ = value;}
  inline void set_use_superconvergent_jump (bool value) { use_superconvergent_jump_  = value;}
  inline void set_use_points_on_interface  (bool value) { use_points_on_interface_   = value;}
  inline void set_update_c0_robin          (bool value) { update_c0_robin_           = value;}

  void initialize_solvers();

  void solve(Vec t, Vec c0, Vec c1, Vec bc_error, double &bc_error_max, double &dt, double cfl, bool use_non_zero_guess = false);

  void solve_t();
  void solve_psi_t();

  void solve_c0();
  void solve_psi_c0();
  void solve_c0_robin();

  void solve_c1();
  void solve_psi_c1();

  void compute_c0n();
  void compute_psi_c0n();
  void adjust_c0_gamma(int iteration);

  my_p4est_poisson_nodes_t* get_solver_c0() { return solver_c0; }
#ifdef P4_TO_P8
  CF_3* get_vn() { return &vn_from_c0_; }
#else
  CF_2* get_vn() { return &vn_from_c0_; }
#endif

private:
#ifdef P4_TO_P8
  class zero_cf_t : public CF_3
  {
  public:
    double operator()(double, double, double) const
    {
      return 0;
    }
  } zero_cf_;

  class zero_cf1_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero_cf1_;
#else
  class zero_cf_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero_cf_;

  class zero_cf1_t : public CF_1
  {
  public:
    double operator()(double) const
    {
      return 0;
    }
  } zero_cf1_;
#endif

  // boundary conditions for Lagrangian multipliers
#ifdef P4_TO_P8
  class jump_psi_tn_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double, double, double) const
    {
      return 1./ptr_->t_diff_/ptr_->ml0_;
    }
  } jump_psi_tn_;
#else
  class jump_psi_tn_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double, double) const
    {
      return 1./ptr_->t_diff_/ptr_->ml0_;
    }
  } jump_psi_tn_;
#endif

#ifdef P4_TO_P8
  class psi_c1_interface_value_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double, double, double) const
    {
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->ml1_/ptr_->ml0_*ptr_->dt_;
      else
        return -ptr_->ml1_/ptr_->ml0_/ptr_->Dl1_;
    }
  } psi_c1_interface_value_;
#else
  class psi_c1_interface_value_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double, double) const
    {
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->ml1_/ptr_->ml0_*ptr_->dt_;
      else
        return -ptr_->ml1_/ptr_->ml0_/ptr_->Dl1_;
    }
  } psi_c1_interface_value_;
#endif

  // velocity
#ifdef P4_TO_P8
  class vn_from_c0_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y, double z) const
    {
      ptr_->interp_.set_input(ptr_->c0_.vec, ptr_->c0_dd_.vec[0], ptr_->c0_dd_.vec[1], ptr_->c0_dd_.vec[2], quadratic_non_oscillatory_continuous_v1);
//      ptr_->interp_.set_input(ptr_->c0_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0_gamma_.vec, linear);
      double c0 = ptr_->interp_(x, y, z);

      ptr_->interp_.set_input(ptr_->c0n_.vec, ptr_->c0n_dd_.vec[0], ptr_->c0n_dd_.vec[1], ptr_->c0n_dd_.vec[2], quadratic_non_oscillatory_continuous_v1);
//      ptr_->interp_.set_input(ptr_->c0n_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0n_gamma_.vec, linear);
      double c0n = ptr_->interp_(x, y, z);

      return ptr_->Dl0_/(1.-ptr_->kp0_)*(c0n - (*ptr_->c0_flux_)(x,y,z))/c0;
    }
  } vn_from_c0_;
#else
  class vn_from_c0_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y) const
    {
      ptr_->interp_.set_input(ptr_->c0_.vec, ptr_->c0_dd_.vec[0], ptr_->c0_dd_.vec[1], quadratic_non_oscillatory_continuous_v1);
//      ptr_->interp_.set_input(ptr_->c0_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0_gamma_.vec, linear);
      double c0 = ptr_->interp_(x, y);

      ptr_->interp_.set_input(ptr_->c0n_.vec, ptr_->c0n_dd_.vec[0], ptr_->c0n_dd_.vec[1], quadratic_non_oscillatory_continuous_v1);
//      ptr_->interp_.set_input(ptr_->c0n_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0n_gamma_.vec, linear);
      double c0n = ptr_->interp_(x, y);

      return ptr_->Dl0_/(1.-ptr_->kp0_)*(c0n - (*ptr_->c0_flux_)(x,y))/c0;
    }
  } vn_from_c0_;
#endif

  // robin coefficients for concentrations
#ifdef P4_TO_P8
  class c1_robin_coef_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y, double z) const
    {
//      return -(1.-ptr_->kp1_)/ptr_->Dl1_*(*ptr_->vn_exact_)(x,y,z);
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->dt_*(1.-ptr_->kp1_)*ptr_->vn_from_c0_(x,y,z);
      else
        return -(1.-ptr_->kp1_)/ptr_->Dl1_*ptr_->vn_from_c0_(x,y,z);
    }
  } c1_robin_coef_;
#else
  class c1_robin_coef_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y) const
    {
//      return -((1.-ptr_->kp1_)/ptr_->Dl1_)*(*ptr_->vn_exact_)(x,y);
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->dt_*(1.-ptr_->kp1_)*ptr_->vn_from_c0_(x,y);
      else
        return -(1.-ptr_->kp1_)/ptr_->Dl1_*ptr_->vn_from_c0_(x,y);
    }
  } c1_robin_coef_;
#endif

#ifdef P4_TO_P8
  class c0_robin_coef_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y, double z) const
    {
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->dt_*(1.-ptr_->kp0_)*ptr_->vn_from_c0_(x,y,z);
      else
        return -(1.-ptr_->kp0_)/ptr_->Dl0_*ptr_->vn_from_c0_(x,y,z);
    }
  } c0_robin_coef_;
#else
  class c0_robin_coef_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y) const
    {
      if (ptr_->use_superconvergent_robin_)
        return -ptr_->dt_*(1.-ptr_->kp0_)*ptr_->vn_from_c0_(x,y);
      else
        return -(1.-ptr_->kp0_)/ptr_->Dl0_*ptr_->vn_from_c0_(x,y);
    }
  } c0_robin_coef_;
#endif

  // jump in normal derivative of temperature
#ifdef P4_TO_P8
  class tn_jump_t : public CF_3
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y, double z) const
    {
      // from now on liquid phase is in \phi < 0 and solid phase is in \phi > 0
//      return ptr_->latent_heat_/ptr_->t_cond_*(*ptr_->vn_exact_)(x,y,z);
      return ptr_->latent_heat_/ptr_->t_cond_*ptr_->vn_from_c0_(x,y,z);
    }
  } tn_jump_;
#else
  class tn_jump_t : public CF_2
  {
    my_p4est_poisson_nodes_multialloy_t *ptr_;
  public:
    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
    double operator()(double x, double y) const
    {
      // from now on liquid phase is in \phi < 0 and solid phase is in \phi > 0
//      return ptr_->latent_heat_/ptr_->t_cond_*(*ptr_->vn_exact_)(x,y);
      return ptr_->latent_heat_/ptr_->t_cond_*ptr_->vn_from_c0_(x,y);
    }
  } tn_jump_;
#endif

};

#endif // MY_P4EST_POISSON_NODES_MULTIALLOY_H
