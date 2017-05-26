#ifndef MY_P4EST_MULTIALLOY_H
#define MY_P4EST_MULTIALLOY_H


#include <src/types.h>
#include <src/casl_math.h>



#undef P4_TO_P8
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
//#include <src/my_p4est_poisson_nodes_voronoi.h>
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

  class c00_cf_t : public CF_2
  {
    my_p4est_multialloy_t *ptr_;
  public:
    c00_cf_t(my_p4est_multialloy_t *ptr) { ptr_ = ptr; }
    double operator()(double, double) const
    {
      return ptr_->c00_;
    }
  } c00_cf_;

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
  Vec t_n_, t_np1_;

  /* concentration */
  Vec c0_n_, c0_np1_;
  Vec c1_n_, c1_np1_;

  Vec c0n_np1_, c0n_n_;

  /* velocity */
  Vec v_interface_n_  [P4EST_DIM];
  Vec v_interface_np1_[P4EST_DIM];
  Vec normal_velocity_np1_;
  /* max interface velocity in normal direction in band 4*MIN(dx,dy,dz) */
  double vgamma_max_;

  /* level-set */
  Vec phi_, phi_dd_[P4EST_DIM];
  Vec normal_[P4EST_DIM];
  Vec kappa_;

#ifdef P4_TO_P8
  Vec theta_xz_, theta_yz_;
#else
  Vec theta_;
#endif

  Vec bc_error_;
  double bc_error_max_;

  /* physical parameters */
  double dt_nm1_, dt_n_;
  double cooling_velocity_;     /* V */
  double latent_heat_;          /* L */
  double thermal_conductivity_; /* k */
  double thermal_diffusivity_;  /* lambda, dT/dt = lambda Laplace(T) */
  double Tm_;                   /* melting temperature */
  double epsilon_anisotropy_;   /* anisotropy coefficient */
  double epsilon_c_;            /* curvature undercooling coefficient */
  double epsilon_v_;            /* kinetic undercooling coefficient */

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

  interpolation_method interpolation_between_grids_;

  class eps_c_cf_t : public CF_1
  {
    my_p4est_multialloy_t *ptr_;
  public:
    eps_c_cf_t(my_p4est_multialloy_t *ptr) : ptr_(ptr) {}
    double operator()(double theta) const
    {
      return ptr_->epsilon_c_*(1.-15.*ptr_->epsilon_anisotropy_*cos(4.*theta));
    }
  } eps_c_cf_;

  class eps_v_cf_t : public CF_1
  {
    my_p4est_multialloy_t *ptr_;
  public:
    eps_v_cf_t(my_p4est_multialloy_t *ptr) : ptr_(ptr) {}
    double operator()(double theta) const
    {
      return ptr_->epsilon_v_*(1.-15.*ptr_->epsilon_anisotropy_*cos(4.*theta));
    }
  } eps_v_cf_;

public:

  my_p4est_multialloy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_multialloy_t();

  inline void set_parameters(double latent_heat,
                      double thermal_conductivity,
                      double thermal_diffusivity,
                      double cooling_velocity,
                      double Tm,
                      double epsilon_anisotropy,
                      double epsilon_c,
                      double epsilon_v,
                      double scaling,
                      double Dl0, double kp0, double c00, double ml0,
                      double Dl1, double kp1, double c01, double ml1)
  {
    this->latent_heat_          = latent_heat;
    this->thermal_conductivity_ = thermal_conductivity;
    this->thermal_diffusivity_  = thermal_diffusivity;
    this->cooling_velocity_     = cooling_velocity;
    this->Tm_                   = Tm;
    this->epsilon_anisotropy_   = epsilon_anisotropy;
    this->epsilon_c_            = epsilon_c;
    this->epsilon_v_            = epsilon_v;
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

  inline void set_phi(Vec phi)
  {
    this->phi_ = phi;
    compute_normal_and_curvature();
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

  inline void set_temperature(Vec temperature)
  {
    t_n_ = temperature;

    ierr = VecDuplicate(t_n_, &t_np1_); CHKERRXX(ierr);

    copy_ghosted_vec(t_n_, t_np1_);
  }

  inline void set_concentration(Vec c0, Vec c1)
  {
    c0_n_ = c0;
    c1_n_ = c1;

    ierr = VecDuplicate(c0_n_, &c0_np1_); CHKERRXX(ierr);
    ierr = VecDuplicate(c1_n_, &c1_np1_); CHKERRXX(ierr);

    copy_ghosted_vec(c0_n_, c0_np1_);
    copy_ghosted_vec(c1_n_, c1_np1_);

//    ierr = VecDuplicate(c0_n_, &c0n_n_); CHKERRXX(ierr);
//    ierr = VecDuplicate(c0_n_, &c0n_np1_); CHKERRXX(ierr);

//    double *c0n_n_p;
//    ierr = VecGetArray(c0n_n_, &c0n_n_p); CHKERRXX(ierr);
//    for (size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
//      c0n_n_p[n] = 0.;
//    ierr = VecRestoreArray(c0n_n_, &c0n_n_p); CHKERRXX(ierr);
  }

  void set_normal_velocity(Vec v);

  inline p4est_t* get_p4est() { return p4est_; }

  inline p4est_nodes_t* get_nodes() { return nodes_; }

  inline Vec get_phi() { return phi_; }

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

  void compute_normal_and_curvature();
  void compute_velocity();

  void compute_dt();
  void update_grid();
  void one_step();
  void save_VTK(int iter);

  inline void copy_ghosted_vec(Vec input, Vec output)
  {
    Vec src, out;
    ierr = VecGhostGetLocalForm(input, &src); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(output, &out); CHKERRXX(ierr);
    ierr = VecCopy(src, out); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(input, &src); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(output, &out); CHKERRXX(ierr);
  }

  inline void invert_phi()
  {
    double *phi_p;
    ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);
    for (size_t n = 0; n < nodes_->indep_nodes.elem_count; ++n)
      phi_p[n] *= -1.;
    ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);
  }
};


#endif /* MY_P4EST_MULTIALLOY_H */
