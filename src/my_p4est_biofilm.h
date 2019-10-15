#ifndef MY_P4EST_BIOFILM_H
#define MY_P4EST_BIOFILM_H


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
#include <src/my_p8est_macros.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#endif


class my_p4est_biofilm_t
{
private:
  PetscErrorCode ierr;

#ifdef P4_TO_P8
  class zero_cf_t : public CF_3
  {
  public:
    double operator()(double, double, double) const
    {
      return 0;
    }
  } zero_cf;
#else
  class zero_cf_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero_cf;
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

  /* problem parameters */
  double Df_;       /* diffusivity of nutrients in air           - m^2/s     */
  double Db_;       /* diffusivity of nutrients in biofilm       - m^2/s     */
  double Da_;       /* diffusivity of nutrients in agar          - m^2/s     */
  double sigma_;    /* surface tension of air/film interface     - N/m       */
  double rho_;      /* density of biofilm                        - kg/m^3    */
  double lambda_;   /* mobility of biofilm                       - m^4/(N*s) */
  double gam_;    /* biofilm yield per nutrient mass           -           */
  double scaling_;

  int velocity_type_;

  bool steady_state_;

  /* boundary conditions */
#ifdef P4_TO_P8
  WallBC3D *bc_wall_type_;
  CF_3     *bc_wall_value_;
#else
  WallBC2D *bc_wall_type_;
  CF_2     *bc_wall_value_;
#endif

  /* kinetics */
  CF_1 *f_cf_, *fc_cf_;

  /* concentration */
  Vec Ca0_, Ca1_;
  Vec Cb0_, Cb1_;
  Vec Cf0_, Cf1_;
  Vec C_;

  /* pressure */
  Vec P_;

  /* velocity */
  Vec vn_;
  Vec v0_[P4EST_DIM];
  Vec v1_[P4EST_DIM];

  /* max interface velocity in normal direction in band dxyz_close_interface_ */
  double vn_max_;

  /* level-set */
  Vec phi_biof_, phi_biof_dd_[P4EST_DIM];
  Vec phi_agar_;
  Vec phi_free_;
  Vec normal_[P4EST_DIM];
  Vec kappa_;
  double kappa_max_;

  /* solving non-linear equation */
  int iteration_scheme_;
  int max_iterations_;
  double tolerance_;

  /* time discretization */
  int time_scheme_;
  int advection_scheme_;
  double time_;
  double dt_max_;
  double dt0_, dt1_;
  double cfl_number_;

  bool use_godunov_scheme_;
  bool first_iteration_;

  /* general poisson solver parameters */
  bool use_sc_scheme_;
  bool use_taylor_correction_;
  int integration_order_;
  int extend_iterations_;

  double curvature_smoothing_;
  double curvature_smoothing_steps_;

  interpolation_method interpolation_between_grids_;

public:
  my_p4est_biofilm_t(my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_biofilm_t();

  // model parameters
  inline void set_parameters(double Da,
                             double Db,
                             double Df,
                             double sigma,
                             double rho,
                             double lambda,
                             double gam,
                             double scaling)
  {
    this->Da_      = Da;
    this->Db_      = Db;
    this->Df_      = Df;
    this->sigma_   = sigma;
    this->rho_     = rho;
    this->lambda_  = lambda;
    this->gam_   = gam;
    this->scaling_ = scaling;
  }

#ifdef P4_TO_P8
  inline void set_bc(WallBC3D& bc_wall_type, CF_3& bc_wall_value)
#else
  inline void set_bc(WallBC2D& bc_wall_type, CF_2& bc_wall_value)
#endif
  {
    this->bc_wall_type_  = &bc_wall_type;
    this->bc_wall_value_ = &bc_wall_value;
  }

  inline void set_kinetics(CF_1& f_cf, CF_1& fc_cf) { this->f_cf_ = &f_cf; this->fc_cf_ = &fc_cf; }
  inline void set_velocity_type(int velocity_type) { this->velocity_type_ = velocity_type; }
  inline void set_steady_state(bool value) { this->steady_state_ = value; }

  // time discretization parameters
  inline void set_advection_scheme(int advection_scheme) { this->advection_scheme_ = advection_scheme; }
  inline void set_time_scheme(int time_scheme) { this->time_scheme_ = time_scheme; }
  inline void set_dt_max(double dt_max )
  {
    dt_max_ = dt_max;
    dt0_ = dt_max_;
    dt1_ = dt_max_;
  }
  inline void set_cfl(double cfl) {cfl_number_ = cfl;}

  // parameters for solving non-linear equation
  inline void set_iteration_scheme(int iteration_scheme) { this->iteration_scheme_ =iteration_scheme; }
  inline void set_max_iterations(int max_iterations) { max_iterations_ = max_iterations; }
  inline void set_tolerance(double tolerance) { this->tolerance_ = tolerance; }

  // general poisson solver parameter
  inline void set_use_sc_scheme(bool use_sc_scheme) { this->use_sc_scheme_ = use_sc_scheme; }
  inline void set_use_taylor_correction(bool use_taylor_correction) { this->use_taylor_correction_ = use_taylor_correction; }
  inline void set_integration_order(int integration_order) { this->integration_order_ = integration_order; }

  // initial geometry and concentrations
  inline void set_phi(Vec phi_free, Vec phi_agar)
  {
    copy_ghosted_vec(phi_free, this->phi_free_);
    copy_ghosted_vec(phi_agar, this->phi_agar_);

    compute_geometric_properties();
  }

  inline void set_concentration(Vec Ca, Vec Cb, Vec Cf)
  {
    copy_ghosted_vec(Ca, Ca0_);
    copy_ghosted_vec(Cb, Cb0_);
    copy_ghosted_vec(Cf, Cf0_);

    if (time_scheme_ == 2)
    {
      copy_ghosted_vec(Ca, Ca1_);
      copy_ghosted_vec(Cb, Cb1_);
      copy_ghosted_vec(Cf, Cf1_);
    }
    compute_concentration_global();
  }

  inline p4est_t*                   get_p4est() { return p4est_; }
  inline p4est_nodes_t*             get_nodes() { return nodes_; }
  inline p4est_ghost_t*             get_ghost() { return ghost_; }
  inline my_p4est_node_neighbors_t* get_ngbd()  { return ngbd_; }

  inline Vec get_phi_agar() { return phi_agar_; }
  inline Vec get_phi_free() { return phi_free_; }
  inline Vec get_phi_biof() { return phi_biof_; }
  inline Vec get_vn() { return vn_; }
  inline double get_vn_max() { return vn_max_; }
  inline Vec get_Ca() { return Ca0_; }
  inline Vec get_Cb() { return Cb0_; }
  inline Vec get_Cf() { return Cf0_; }
  inline Vec get_C()  { return C_; }

  inline double get_dt() { return dt0_; }

  void compute_geometric_properties();
  void compute_velocity_from_concentration();
  void compute_velocity_from_pressure();
  void compute_filtered_curvature();

  void solve_concentration();
  void solve_pressure();

  void compute_dt();
  void update_grid();
  int  one_step();
  void save_VTK(int iter);

  void compute_concentration_global();

  inline void set_curvature_smoothing (double value, int steps) { curvature_smoothing_ = value; curvature_smoothing_steps_ = steps; }

};


#endif /* MY_P4EST_BIOFILM_H */
