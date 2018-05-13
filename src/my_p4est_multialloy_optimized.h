#ifndef MY_P4EST_MULTIALLOY_H
#define MY_P4EST_MULTIALLOY_H


#include <src/types.h>
#include <src/casl_math.h>


#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes_multialloy.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif


class my_p4est_multialloy_t
{
private:

  PetscErrorCode ierr;

#ifdef P4_TO_P8
  class ZERO : public CF_3
  {
  public:
    double operator()(double, double, double) const
    {
      return 0;
    }
  } zero_;
#else
  class ZERO : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero_;
#endif

#ifdef P4_TO_P8
  class wall_bc_smoothing_t : public WallBC3D
  {
  public:
    BoundaryConditionType operator()( double , double , double ) const
    {
      return NEUMANN;
    }
  } wall_bc_smoothing_;
#else
  class wall_bc_smoothing_t : public WallBC2D
  {
  public:
    BoundaryConditionType operator()( double , double ) const
    {
      return NEUMANN;
    }
  } wall_bc_smoothing_;
#endif

#ifdef P4_TO_P8
  BoundaryConditions3D bc_t_;
  BoundaryConditions3D bc_c0_;
  BoundaryConditions3D bc_c1_;
#else
  BoundaryConditions2D bc_t_;
  BoundaryConditions2D bc_c0_;
  BoundaryConditions2D bc_c1_;
#endif

  /* grid */
  my_p4est_brick_t            *brick_;
  p4est_connectivity_t        *connectivity_;
  p4est_t                     *p4est_;
  p4est_ghost_t               *ghost_;
  p4est_nodes_t               *nodes_;
  my_p4est_hierarchy_t        *hierarchy_;
  my_p4est_node_neighbors_t   *ngbd_;

  double dxyz_[P4EST_DIM];
  double dxyz_max_, dxyz_min_;
  double dxyz_close_interface_;
  double diag_;

  /* temperature */
//  Vec t_n_, t_np1_;
  Vec tl_n_, tl_np1_;
  Vec ts_n_, ts_np1_;

  Vec tf_; // temperature at which alloy solidified

  /* concentration */
  Vec c0_n_, c0_np1_;
  Vec c1_n_, c1_np1_;

  Vec c0s_;
  Vec c1s_;

  Vec c0n_np1_, c0n_n_;

  /* velocity */
  Vec v_interface_n_  [P4EST_DIM];
  Vec v_interface_np1_[P4EST_DIM];
  Vec normal_velocity_n_;
  Vec normal_velocity_np1_;
  /* max interface velocity in normal direction in band 4*MIN(dx,dy,dz) */
  double vgamma_max_;

  /* level-set */
  Vec phi_, phi_dd_[P4EST_DIM];
  Vec normal_[P4EST_DIM];
  Vec kappa_;

  Vec phi_smooth_;

#ifdef P4_TO_P8
  Vec theta_xz_, theta_yz_;
#else
  Vec theta_;
#endif

  Vec bc_error_;
  double bc_error_max_;

  /* physical parameters */
  double time_, time_limit_;
  double dt_nm1_, dt_n_;
  double cooling_velocity_;     /* V */
  double latent_heat_;          /* L */
  double thermal_conductivity_; /* k */
  double thermal_diffusivity_;  /* lambda, dT/dt = lambda Laplace(T) */
  double Tm_;                   /* melting temperature */

  double Dl0_;                  /* Dl, dCl/dt = Dl Laplace(Cl) */
  double kp0_;                  /* partition coefficient */
  double c00_;                  /* initial concentration */
  double ml0_;                  /* liquidus slope */

  double Dl1_;                  /* Dl, dCl/dt = Dl Laplace(Cl) */
  double kp1_;                  /* partition coefficient */
  double c01_;                  /* initial concentration */
  double ml1_;                  /* liquidus slope */

  double scaling_;

  double cfl_number_;
  int pin_every_n_steps_;
  double bc_tolerance_;
  int max_iterations_;
  double phi_thresh_;

  bool use_continuous_stencil_;
  bool use_one_sided_derivatives_;
  bool use_superconvergent_robin_;
  bool use_superconvergent_jump_;
  bool use_points_on_interface_;
  bool update_c0_robin_;
  bool zero_negative_velocity_;

#ifdef P4_TO_P8
  CF_3 *rhs_tl_;
  CF_3 *rhs_ts_;
  CF_3 *rhs_c0_;
  CF_3 *rhs_c1_;
#else
  CF_2 *rhs_tl_;
  CF_2 *rhs_ts_;
  CF_2 *rhs_c0_;
  CF_2 *rhs_c1_;
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
  CF_3 *jump_tn_;
  CF_3 *c0_flux_;
  CF_3 *c1_flux_;
#else
  CF_2 *GT_;
  CF_2 *jump_t_;
  CF_2 *jump_tn_;
  CF_2 *c0_flux_;
  CF_2 *c1_flux_;
#endif

  interpolation_method interpolation_between_grids_;

public:

  my_p4est_multialloy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_multialloy_t();

  inline void set_parameters(double latent_heat,
                      double thermal_conductivity,
                      double thermal_diffusivity,
                      double cooling_velocity,
                      double Tm,
                      double scaling,
                      double Dl0, double kp0, double c00, double ml0,
                      double Dl1, double kp1, double c01, double ml1)
  {
    this->latent_heat_          = latent_heat;
    this->thermal_conductivity_ = thermal_conductivity;
    this->thermal_diffusivity_  = thermal_diffusivity;
    this->cooling_velocity_     = cooling_velocity;
    this->Tm_                   = Tm;
    this->scaling_              = scaling;

    this->Dl0_ = Dl0;
    this->kp0_ = kp0;
    this->c00_ = c00;
    this->ml0_ = ml0;

    this->Dl1_ = Dl1;
    this->kp1_ = kp1;
    this->c01_ = c01;
    this->ml1_ = ml1;
  }

  inline void set_use_continuous_stencil   (bool value) { use_continuous_stencil_    = value;}
  inline void set_use_one_sided_derivatives(bool value) { use_one_sided_derivatives_ = value;}
  inline void set_use_superconvergent_robin(bool value) { use_superconvergent_robin_ = value;}
  inline void set_use_superconvergent_jump (bool value) { use_superconvergent_jump_  = value;}
  inline void set_use_points_on_interface  (bool value) { use_points_on_interface_   = value;}
  inline void set_update_c0_robin          (bool value) { update_c0_robin_           = value;}
  inline void set_zero_negative_velocity   (bool value) { zero_negative_velocity_    = value;}


  inline void set_phi(Vec phi)
  {
    this->phi_ = phi;
//    compute_smoothed_phi();
//    copy_ghosted_vec(phi_smooth_, phi_);
    compute_geometric_properties();
  }

#ifdef P4_TO_P8
  inline void set_bc(WallBC3D& bc_wall_type_t,
              WallBC3D& bc_wall_type_c,
              CF_3& bc_wall_value_t,
              CF_3& bc_wall_value_c0,
              CF_3& bc_wall_value_c1)
#else
  inline void set_bc(WallBC2D& bc_wall_type_t,
              WallBC2D& bc_wall_type_c,
              CF_2& bc_wall_value_t,
              CF_2& bc_wall_value_c0,
              CF_2& bc_wall_value_c1)
#endif
  {
    bc_t_.setWallTypes(bc_wall_type_t);
    bc_t_.setWallValues(bc_wall_value_t);
    bc_t_.setInterfaceType(NOINTERFACE);
    bc_t_.setInterfaceValue(zero_);

    bc_c0_.setWallTypes(bc_wall_type_c);
    bc_c0_.setWallValues(bc_wall_value_c0);
    bc_c0_.setInterfaceType(DIRICHLET);
    bc_c0_.setInterfaceValue(zero_);

    bc_c1_.setWallTypes(bc_wall_type_c);
    bc_c1_.setWallValues(bc_wall_value_c1);
    bc_c1_.setInterfaceType(ROBIN);
    bc_c1_.setInterfaceValue(zero_);

  }

  inline void set_temperature(Vec tl, Vec ts)
  {
    tl_n_ = tl;
    ts_n_ = ts;

    ierr = VecDuplicate(tl_n_, &tl_np1_); CHKERRXX(ierr);
    ierr = VecDuplicate(ts_n_, &ts_np1_); CHKERRXX(ierr);

    copy_ghosted_vec(tl_n_, tl_np1_);
    copy_ghosted_vec(ts_n_, ts_np1_);

    ierr = VecDuplicate(ts_n_, &tf_); CHKERRXX(ierr);

    copy_ghosted_vec(ts_n_, tf_);
  }

  inline void set_concentration(Vec c0, Vec c1, Vec c0s, Vec c1s)
  {
    c0_n_ = c0;
    c1_n_ = c1;

    ierr = VecDuplicate(c0_n_, &c0_np1_); CHKERRXX(ierr);
    ierr = VecDuplicate(c1_n_, &c1_np1_); CHKERRXX(ierr);

    copy_ghosted_vec(c0_n_, c0_np1_);
    copy_ghosted_vec(c1_n_, c1_np1_);

    c0s_ = c0s;
    c1s_ = c1s;

//    ierr = VecDuplicate(c0_n_, &c0n_n_); CHKERRXX(ierr);
//    ierr = VecDuplicate(c0_n_, &c0n_np1_); CHKERRXX(ierr);

//    double *c0n_n_p;
//    ierr = VecGetArray(c0n_n_, &c0n_n_p); CHKERRXX(ierr);
//    for (size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
//      c0n_n_p[n] = 0.;
//    ierr = VecRestoreArray(c0n_n_, &c0n_n_p); CHKERRXX(ierr);
  }

  void set_normal_velocity(Vec v);

  inline p4est_t*       get_p4est() { return p4est_; }
  inline p4est_nodes_t* get_nodes() { return nodes_; }
  inline p4est_ghost_t* get_ghost() { return ghost_; }
  inline my_p4est_node_neighbors_t* get_ngbd()  { return ngbd_; }

  inline Vec get_phi() { return phi_; }
  inline Vec* get_phi_dd() { return phi_dd_; }


  inline Vec get_normal_velocity() { return normal_velocity_np1_; }

  inline double get_dt() { return dt_n_; }

  inline double get_max_interface_velocity() { return vgamma_max_; }

  inline void set_dt( double dt )
  {
    dt_nm1_ = dt;
    dt_n_   = dt;
  }

  inline void set_cfl (double val) {cfl_number_ = val;}

  inline void set_bc_tolerance (double bc_tolerance) { bc_tolerance_ = bc_tolerance; }
  inline void set_pin_every_n_steps (int pin_every_n_steps) { pin_every_n_steps_ = pin_every_n_steps; }
  inline void set_max_iterations (int max_iterations) { max_iterations_ = max_iterations; }
  inline void set_phi_thresh (int phi_thresh) { phi_thresh_ = phi_thresh; }

  void compute_geometric_properties();
  void compute_velocity();

  void compute_dt();
  void update_grid();
  void update_grid_eno();
  int  one_step();
  void save_VTK(int iter);

  void compute_smoothed_phi();

#ifdef P4_TO_P8
  inline void set_rhs(CF_3& rhs_tl, CF_3& rhs_ts, CF_3& rhs_c0, CF_3& rhs_c1)
#else
  inline void set_rhs(CF_2& rhs_tl, CF_2& rhs_ts, CF_2& rhs_c0, CF_2& rhs_c1)
#endif
  {
    rhs_tl_ = &rhs_tl;
    rhs_ts_ = &rhs_ts;
    rhs_c0_ = &rhs_c0;
    rhs_c1_ = &rhs_c1;
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

#ifdef P4_TO_P8
  inline void set_jump_t (CF_3& jump_t)  { jump_t_  = &jump_t;  }
  inline void set_jump_tn(CF_3& jump_tn) { jump_tn_ = &jump_tn; }
  inline void set_flux_c (CF_3& c0_flux,
                          CF_3& c1_flux)
#else
  inline void set_jump_t (CF_2& jump_t)  { jump_t_  = &jump_t;  }
  inline void set_jump_tn(CF_2& jump_tn) { jump_tn_ = &jump_tn; }
  inline void set_flux_c (CF_2& c0_flux,
                          CF_2& c1_flux)
#endif
  {
    c0_flux_ = &c0_flux;
    c1_flux_ = &c1_flux;
  }

  inline void set_time_limit (double time_limit) { time_limit_ = time_limit; }

  inline Vec get_tl() { return tl_n_; }
  inline Vec get_ts() { return ts_n_; }
  inline Vec get_c0() { return c0_np1_; }
  inline Vec get_c1() { return c1_np1_; }
};


#endif /* MY_P4EST_MULTIALLOY_H */
