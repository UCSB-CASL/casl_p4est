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
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_macros.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#endif

class my_p4est_poisson_nodes_multialloy_t
{
  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_multialloy_t(const my_p4est_poisson_nodes_t& other);
  my_p4est_poisson_nodes_multialloy_t& operator=(const my_p4est_poisson_nodes_t& other);

public:
  my_p4est_poisson_nodes_multialloy_t(my_p4est_node_neighbors_t *node_neighbors, int num_comps);
  ~my_p4est_poisson_nodes_multialloy_t();
private:

  PetscErrorCode ierr;

  my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t interp_;

  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  vec_and_ptr_t contr_phi_;
  vec_and_ptr_t front_phi_;
  vec_and_ptr_t front_curvature_;
  vec_and_ptr_t solid_phi_;
  vec_and_ptr_t liquid_phi_;

  vec_and_ptr_dim_t contr_phi_dd_;
  vec_and_ptr_dim_t front_phi_dd_;
  vec_and_ptr_dim_t front_normal_;
  vec_and_ptr_dim_t solid_normal_;
  vec_and_ptr_dim_t liquid_normal_;

  bool contr_phi_dd_owned_;
  bool front_phi_dd_owned_, front_normal_owned_, front_curvature_owned_;

public:
  void clear_contr();
  void clear_front();

  void set_container(Vec phi, Vec* phi_dd);
  void set_front(Vec phi, Vec* phi_dd, Vec *normal, Vec curvature);
private:

  //--------------------------------------------------
  // Equation parameters
  //--------------------------------------------------

  // composition parameters
  int num_comps_;
  vector<double> conc_diag_;
  vector<double> conc_diff_;
  vector<double> part_coeff_;

  // thermal parameters
  double temp_diag_l_, temp_diff_l_;
  double temp_diag_s_, temp_diff_s_;
  double latent_heat_;
  double melting_temp_;

  // front conditions
  CF_DIM           *c0_guess_;
  CF_DIM           *gibbs_thomson_;
  CF_DIM           *front_temp_value_jump_;
  CF_DIM           *front_temp_flux_jump_;
  vector<CF_DIM *>  front_conc_flux_;

  double (*liquidus_value_)(double *);
  double (*liquidus_slope_)(int, double *);

  // right-hand sides
  vec_and_ptr_t         rhs_zero_;
  vec_and_ptr_t         rhs_tl_;
  vec_and_ptr_t         rhs_ts_;
  vector<vec_and_ptr_t> rhs_c_;

  // undercoolings
  int              num_seeds_;
  vec_and_ptr_t    seed_map_;
  vector<CF_DIM *> eps_c_;
  vector<CF_DIM *> eps_v_;

  // boundary conditions at container
  BoundaryConditionType         contr_bc_type_temp_;
  vector<BoundaryConditionType> contr_bc_type_conc_;

  CF_DIM           *contr_bc_value_temp_;
  vector<CF_DIM *>  contr_bc_value_conc_;

  // boundary condtions at walls
  WallBCDIM           *wall_bc_type_temp_;
  vector<WallBCDIM *>  wall_bc_type_conc_;

  CF_DIM           *wall_bc_value_temp_;
  vector<CF_DIM *>  wall_bc_value_conc_;

public:
//  inline void set_number_of_components(int num)
//  {
//  }

  inline void set_composition_parameters(double conc_diag[], double conc_diff[], double part_coeff[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      conc_diag_ [i] = conc_diag [i];
      conc_diff_ [i] = conc_diff [i];
      part_coeff_[i] = part_coeff[i];
    }
  }

  inline void set_thermal_parameters(double latent_heat,
                                     double temp_diag_l, double temp_diff_l,
                                     double temp_diag_s, double temp_diff_s)
  {
    latent_heat_ = latent_heat;
    temp_diag_l_ = temp_diag_l; temp_diff_l_ = temp_diff_l;
    temp_diag_s_ = temp_diag_s; temp_diff_s_ = temp_diff_s;
  }

  inline void set_gibbs_thomson(CF_DIM &gibbs_thomson) { gibbs_thomson_ = &gibbs_thomson; }
  inline void set_liquidus(double melting_temp, double (*liquidus_value)(double *), double (*liquidus_slope)(int, double *))
  {
    melting_temp_   = melting_temp;
    liquidus_value_ = liquidus_value;
    liquidus_slope_ = liquidus_slope;
  }

  inline void set_undercoolings(int num_seeds, Vec seed_map, CF_DIM *eps_v[], CF_DIM *eps_c[])
  {
    num_seeds_    = num_seeds;
    seed_map_.vec = seed_map;

    eps_v_.resize(num_seeds, NULL);
    eps_c_.resize(num_seeds, NULL);

    for (int i = 0; i < num_seeds; ++i)
    {
      eps_v_[i] = eps_v[i];
      eps_c_[i] = eps_c[i];
    }
  }

  inline void set_rhs(Vec rhs_tl, Vec rhs_ts, Vec rhs_c[])
  {
    rhs_tl_.vec = rhs_tl;
    rhs_ts_.vec = rhs_ts;
    for (int i = 0; i < num_comps_; ++i) rhs_c_[i].vec = rhs_c[i];
  }

  inline void set_front_conditions(CF_DIM &front_temp_value_jump,
                                   CF_DIM &front_temp_flux_jump,
                                   CF_DIM *front_conc_flux[])
  {
    front_temp_value_jump_ = &front_temp_value_jump;
    front_temp_flux_jump_  = &front_temp_flux_jump;
    for (int i = 0; i < num_comps_; ++i)
    {
      front_conc_flux_[i] = front_conc_flux[i];
    }
  }

  inline void set_container_conditions_thermal(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    contr_bc_type_temp_  =  bc_type;
    contr_bc_value_temp_ = &bc_value;
  }

  inline void set_container_conditions_composition(BoundaryConditionType bc_type[], CF_DIM *bc_value[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      contr_bc_type_conc_ [i] = bc_type [i];
      contr_bc_value_conc_[i] = bc_value[i];
    }
  }

  inline void set_wall_conditions_thermal(WallBCDIM &bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_temp_  = &bc_type;
    wall_bc_value_temp_ = &bc_value;
  }

  inline void set_wall_conditions_composition(WallBCDIM *bc_type[], CF_DIM *bc_value[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      wall_bc_type_conc_ [i] = bc_type [i];
      wall_bc_value_conc_[i] = bc_value[i];
    }
  }

  inline void set_c0_guess(CF_DIM &c0_guess) { c0_guess_ = &c0_guess; }
private:

  //--------------------------------------------------
  // For debugging
  //--------------------------------------------------
  CF_DIM *vn_exact_;
public:
  inline void set_vn(CF_DIM &vn_cf) { vn_exact_ = &vn_cf; }
private:

  //--------------------------------------------------
  // Solvers and solutions
  //--------------------------------------------------
  my_p4est_poisson_nodes_mls_t          *solver_temp_;
  my_p4est_poisson_nodes_mls_t          *solver_conc_leading_;
  vector<my_p4est_poisson_nodes_mls_t *> solver_conc_;

  // solution
  vector<vec_and_ptr_t>       c_;
  vector<vec_and_ptr_dim_t>   c_d_;
  vector<vec_and_ptr_array_t> c_dd_;

  vec_and_ptr_t tl_;
  vec_and_ptr_t ts_;
  vec_and_ptr_dim_t tl_d_;
  vec_and_ptr_dim_t ts_d_;
  vec_and_ptr_array_t tl_dd_;
  vec_and_ptr_array_t ts_dd_;

  vec_and_ptr_t c0_gamma_;
  vec_and_ptr_t c0n_gamma_;

  // lagrangian multipliers
  vector<vec_and_ptr_t>     psi_c_;
  vector<vec_and_ptr_dim_t> psi_c_d_;

  vec_and_ptr_t     psi_tl_;
  vec_and_ptr_t     psi_ts_;
  vec_and_ptr_dim_t psi_tl_d_;
  vec_and_ptr_dim_t psi_ts_d_;

  // velocity related quatities
  vec_and_ptr_t c0n_;
  vec_and_ptr_t psi_c0n_;
  vec_and_ptr_dim_t c0n_dd_;
  vec_and_ptr_dim_t psi_c0n_dd_;

  vec_and_ptr_dim_t c0d_;
  vec_and_ptr_dim_t psi_c0d_;

  vec_and_ptr_t bc_error_gamma_;
  vec_and_ptr_t bc_error_;

  double bc_error_max_;
  double velo_max_;

  void initialize_solvers();

  void solve_t();
  void solve_psi_t();

  void solve_c0();
  void solve_c0_robin();
  void solve_psi_c0();

  void solve_c(int start, int num);
  void solve_psi_c(int start, int num);

  void compute_c0n();
  void compute_psi_c0n();

  void adjust_c0_gamma(bool simple);
  void compute_pw_bc_values(int start, int num);
  void compute_pw_bc_psi_values(int start, int num);

public:
  my_p4est_poisson_nodes_mls_t* get_solver_temp() { return solver_temp_; }
  int solve(Vec tl, Vec ts, Vec c[], Vec c0d[], Vec bc_error, double &bc_error_max, bool use_non_zero_guess = false,
            std::vector<double> *num_pdes = NULL, std::vector<double> *error = NULL,
            Vec psi_tl=NULL, Vec psi_ts=NULL, Vec psi_c[]=NULL);

//#ifdef P4_TO_P8
//  CF_3* get_vn() { return &vn_from_c0_; }
//#else
//  CF_2* get_vn() { return &vn_from_c0_; }
//#endif
private:

  //--------------------------------------------------
  // Auxiliary arrays that sample boundary and interface conditions
  //--------------------------------------------------
  vector< vector<double> > pw_c_values_;
  vector< vector<double> > pw_c_values_robin_;
  vector< vector<double> > pw_c_coeffs_robin_;

  vector< vector<double> > pw_psi_c_values_;
  vector< vector<double> > pw_psi_c_values_robin_;
  vector< vector<double> > pw_psi_c_coeffs_robin_;

  vector<double> pw_t_sol_jump_taylor_;
  vector<double> pw_t_flx_jump_taylor_;
  vector<double> pw_t_flx_jump_integr_;

  vector<double> pw_psi_t_sol_jump_taylor_;
  vector<double> pw_psi_t_flx_jump_taylor_;
  vector<double> pw_psi_t_flx_jump_integr_;

  vector<double> pw_c0_values_;
  vector<double> pw_psi_c0_values_;

  //--------------------------------------------------
  // Solver parameters
  //--------------------------------------------------
  double bc_tolerance_;
  int    max_iterations_;
  int    pin_every_n_iterations_;
  int    update_c0_robin_;

  bool   second_derivatives_owned_;
  bool   use_superconvergent_robin_;
  bool   use_superconvergent_jump_; // not used atm
  bool   use_points_on_interface_;
  bool   zero_negative_velocity_;
  bool   poisson_use_nonzero_guess_;
  bool   flatten_front_values_;
  bool   always_use_centroid_;
  bool   extension_use_nonzero_guess_;
  bool   verbose_;

  double err_eps_;
  double min_volume_;
  double volume_thresh_;
  double extension_band_use_;
  double extension_band_check_;
  double extension_band_extend_;
  double extension_tol_;

  int    num_extend_iterations_;
  int    cube_refinement_;
  int    integration_order_;

  enum var_scheme_t
  {
    VALUE,
    ABS_VALUE,
    QUADRATIC,
    ABS_ALTER,
    ABS_SMTH1,
    ABS_SMTH2,
    ABS_SMTH3
  } var_scheme_;

public:
  inline void set_pin_every_n_iterations    (int          value) { pin_every_n_iterations_    = value; }
  inline void set_cube_refinement           (int          value) { cube_refinement_           = value; }
  inline void set_integration_order         (int          value) { integration_order_         = value; }
  inline void set_update_c0_robin           (int          value) { update_c0_robin_           = value; }
  inline void set_use_superconvergent_robin (bool         value) { use_superconvergent_robin_ = value; }
  inline void set_use_superconvergent_jump  (bool         value) { use_superconvergent_jump_  = value; }
  inline void set_use_points_on_interface   (bool         value) { use_points_on_interface_   = value; }
  inline void set_zero_negative_velocity    (bool         value) { zero_negative_velocity_    = value; }
  inline void set_flatten_front_values      (bool         value) { flatten_front_values_      = value; }
  inline void set_scheme                    (var_scheme_t value) { var_scheme_                = value; }
  inline void set_verbose_mode              (bool         value) { verbose_                   = value; }

  inline void set_tolerance(double bc_tolerance, int max_iterations = 10)
  {
    bc_tolerance_   = bc_tolerance;
    max_iterations_ = max_iterations;
  }

  inline double compute_vn(double *xyz)
  {
    double nd;
    double c0n = 0;
    foreach_dimension(dim)
    {
      interp_.set_input(front_normal_.vec[dim], linear); nd = interp_.value(xyz);
      interp_.set_input(c0d_.vec[dim],          linear); c0n += nd*interp_.value(xyz);
    }

    interp_.set_input(c_[0].vec, DIM(c_dd_[0].vec[0], c_dd_[0].vec[1], c_dd_[0].vec[2]), quadratic_non_oscillatory_continuous_v2);
    return (conc_diff_[0]*c0n - front_conc_flux_[0]->value(xyz))/interp_.value(xyz)/(1.0-part_coeff_[0]);
  }

//private:
//#ifdef P4_TO_P8
//  class bc_error_cf_t : public CF_3
//  {
//  public:
//    double operator()(double, double, double) const
//    {
//      return 1;
//    }
//  } bc_error_cf_;
//#else
//  class bc_error_cf_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
//      ptr_->interp_.set_input(ptr_->bc_error_gamma_.vec, linear);
//      return ptr_->interp_(x, y);
//    }
//  } bc_error_cf_;
//#endif

//  // boundary conditions for Lagrangian multipliers
//#ifdef P4_TO_P8
//  class jump_psi_tn_t : public CF_3
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double, double, double) const
//    {
//      return ptr_->dt_/ptr_->ml0_;
//    }
//  } jump_psi_tn_;
//#else
//  class jump_psi_tn_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
//      double factor = ptr_->dt_/ptr_->ml0_;
//      switch (ptr_->var_scheme_) {
//        case VALUE:     return factor;
//        case ABS_ALTER: return factor;
//        case ABS_VALUE: return factor*(ptr_->bc_error_cf_(x,y) < 0 ? -1. : 1.);
//        case ABS_SMTH1: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(2.*exp(err/err_eps)/(1.+exp(err/err_eps)) - 1.); }
//        case ABS_SMTH2: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(err/sqrt(err*err + err_eps*err_eps)); }
//        case ABS_SMTH3: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(err/sqrt(err*err + err_eps*err_eps) + err_eps*err_eps*err/pow(err*err + err_eps*err_eps, 1.5)); }
//        case QUADRATIC: return factor*(ptr_->bc_error_cf_(x,y));
//      }
//    }
//  } jump_psi_tn_;
//#endif

//#ifdef P4_TO_P8
//  class psi_c1_interface_value_t : public CF_3
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double, double, double) const
//    {
//      return -ptr_->ml1_/ptr_->ml0_*ptr_->dt_;
//    }
//  } psi_c1_interface_value_;
//#else
//  class psi_c1_interface_value_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
//      double factor = -ptr_->ml1_/ptr_->ml0_*ptr_->dt_;

//      switch (ptr_->var_scheme_) {
//        case VALUE:     return factor;
//        case ABS_ALTER: return factor;
//        case ABS_VALUE: return factor*(ptr_->bc_error_cf_(x,y) < 0 ? -1. : 1.);
//        case ABS_SMTH1: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(2.*exp(err/err_eps)/(1.+exp(err/err_eps)) - 1.); }
//        case ABS_SMTH2: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(err/sqrt(err*err + err_eps*err_eps)); }
//        case ABS_SMTH3: { double err = ptr_->bc_error_cf_(x,y); double err_eps = ptr_->err_eps_; return factor*(err/sqrt(err*err + err_eps*err_eps) + err_eps*err_eps*err/pow(err*err + err_eps*err_eps, 1.5)); }
//        case QUADRATIC: return factor*(ptr_->bc_error_cf_(x,y));
//      }
//    }
//  } psi_c1_interface_value_;
//#endif

//  // velocity
//#ifdef P4_TO_P8
//  class vn_from_c0_t : public CF_3
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y, double z) const
//    {
////      ptr_->interp_.set_input(ptr_->c0_.vec, ptr_->c0_dd_.vec[0], ptr_->c0_dd_.vec[1], ptr_->c0_dd_.vec[2], quadratic_non_oscillatory_continuous_v1);
////      ptr_->interp_.set_input(ptr_->c0_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0_gamma_.vec, linear);
//      double c0 = ptr_->interp_(x, y, z);

////      ptr_->interp_.set_input(ptr_->c0n_.vec, ptr_->c0n_dd_.vec[0], ptr_->c0n_dd_.vec[1], ptr_->c0n_dd_.vec[2], quadratic_non_oscillatory_continuous_v1);
////      ptr_->interp_.set_input(ptr_->c0n_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0n_gamma_.vec, linear);
//      double c0n = ptr_->interp_(x, y, z);

//      return ((*ptr_->c0_flux_)(x,y,z)-ptr_->Dl0_*c0n)/c0/(1.-ptr_->kp0_);
//    }
//  } vn_from_c0_;
//#else
//  class vn_from_c0_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
////      ptr_->interp_.set_input(ptr_->c0_.vec, ptr_->c0_dd_.vec[0], ptr_->c0_dd_.vec[1], quadratic_non_oscillatory_continuous_v1);
////      ptr_->interp_.set_input(ptr_->c0_.vec, linear);
//      ptr_->interp_.set_input(ptr_->c0_gamma_.vec, linear);
//      double c0 = ptr_->interp_(x, y);

//////      ptr_->interp_.set_input(ptr_->c0n_.vec, ptr_->c0n_dd_.vec[0], ptr_->c0n_dd_.vec[1], quadratic_non_oscillatory_continuous_v1);
//////      ptr_->interp_.set_input(ptr_->c0n_.vec, linear);
////      ptr_->interp_.set_input(ptr_->c0n_gamma_.vec, linear);
////      double c0n = ptr_->interp_(x, y);

//      double n = 0;
//      double c0n = 0;

//      foreach_dimension(dim)
//      {
//        ptr_->interp_.set_input(ptr_->front_normal_[dim], linear);
//        n = ptr_->interp_(x, y);
//        ptr_->interp_.set_input(ptr_->c0d_.vec[dim], linear);
//        c0n += n*ptr_->interp_(x, y);
//      }

//      return ((*ptr_->c0_flux_)(x,y) - ptr_->Dl0_*c0n)/c0/(1.-ptr_->kp0_);
//    }
//  } vn_from_c0_;
//#endif

//  // robin coefficients for concentrations
//#ifdef P4_TO_P8
//  class c1_robin_coef_t : public CF_3
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y, double z) const
//    {
//      return (1.-ptr_->kp1_)*ptr_->vn_from_c0_(x,y,z);
//    }
//  } c1_robin_coef_;
//#else
//  class c1_robin_coef_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
////      return (1.-ptr_->kp1_)*(*ptr_->vn_exact_)(x,y);
//      return (1.-ptr_->kp1_)*ptr_->vn_from_c0_(x,y);
//    }
//  } c1_robin_coef_;
//#endif

//#ifdef P4_TO_P8
//  class c0_robin_coef_t : public CF_3
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y, double z) const
//    {
//      return (1.-ptr_->kp0_)*ptr_->vn_from_c0_(x,y,z);
//    }
//  } c0_robin_coef_;
//#else
//  class c0_robin_coef_t : public CF_2
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) {ptr_ = ptr;}
//    double operator()(double x, double y) const
//    {
//      return (1.-ptr_->kp0_)*ptr_->vn_from_c0_(x,y);
//    }
//  } c0_robin_coef_;
//#endif

//  // jump in normal derivative of temperature
//  class tn_jump_t :
//    #ifdef P4_TO_P8
//      public CF_3
//    #else
//      public CF_2
//    #endif
//  {
//    my_p4est_poisson_nodes_multialloy_t *ptr_;
//  public:
//    inline void set_ptr(my_p4est_poisson_nodes_multialloy_t* ptr) { ptr_ = ptr; }
//#ifdef P4_TO_P8
//    double operator()(double x, double y, double z) const
//    {
//      return -ptr_->t_diff_*ptr_->dt_*(ptr_->latent_heat_/ptr_->t_cond_*ptr_->vn_from_c0_(x,y,z) - (*ptr_->jump_tn_)(x,y,z));
//    }
//#else
//    double operator()(double x, double y) const
//    {
//      return -ptr_->t_diff_*ptr_->dt_*(ptr_->latent_heat_/ptr_->t_cond_*ptr_->vn_from_c0_(x,y) - (*ptr_->jump_tn_)(x,y));
//    }
//#endif
//  } tn_jump_;

};

#endif // MY_P4EST_POISSON_NODES_MULTIALLOY_H
