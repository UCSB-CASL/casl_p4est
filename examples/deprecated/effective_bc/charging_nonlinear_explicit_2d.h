#ifndef CHARGING_NONLINEAR_EXPLICIT_2D_H
#define CHARGING_NONLINEAR_EXPLICIT_2D_H

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_poisson_node_base.h>
#else
#include <p4est_bits.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_poisson_node_base.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

class ExplicitNonLinearChargingSolver{
  p4est_t* p4est;
  p4est_ghost_t* ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *brick;
  my_p4est_node_neighbors_t *ngbd;

  Vec phi, phi_xx, phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz;
#endif
  bool local_phi_dd;

  double psi_e;
  Vec psi, con, rhs;
  PetscErrorCode ierr;

  InterpolatingFunctionNodeBase psi_interp, con_interp;
  PoissonSolverNodeBase psi_solver;
  PoissonSolverNodeBase con_solver;

#ifdef P4_TO_P8
  BoundaryConditions3D psi_bc, con_bc;
  class: public WallBC3D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */, double /* z */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } psi_wall_bc;

  class: public WallBC3D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */, double /* z */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } con_wall_bc;

  class:public CF_3{
  public:
    double operator()(double /* x */, double /* y */, double /* z */) const {
        return 0.0;
    };
  } psi_wall_value;

  class:public CF_3{
  public:
    double operator()(double /* x */, double /* y */, double /* z */) const {
        return 0.0;
    };
  } con_wall_value;

#else
  BoundaryConditions2D psi_bc, con_bc;

  class: public WallBC2D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } psi_wall_bc;

  class: public WallBC2D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } con_wall_bc;

  class:public CF_2{
  public:
    double operator()(double /* x */, double /* y */) const {
        return 0.0;
    }
  } psi_wall_value;

  class:public CF_2{
  public:
    double operator()(double /* x */, double /* y */) const {
        return 1.0;
    }
  } con_wall_value;
#endif
  double lambda, dt;

  void solve_potential();
  void solve_concentration();
  void nonlinear_solve(int itmax = 5, double tol = 1e-6);
  void nonlinear_solve_decoupled(int itmax = 5, double tol = 1e-6);

  inline double get_q(double c, double psi) {
      return 2*sqrt(c)*sinh(0.5*(psi - psi_e));
  }


  inline double get_w(double c, double psi) {
      return 4*sqrt(c)*SQR(sinh(0.25*(psi-psi_e)));
  }

  inline double get_dq_dpsi(double c, double psi) {
      return sqrt(c) * cosh(0.5*(psi - psi_e));
  }

  inline double get_dw_dc(double c, double psi) {
      return 2.0/sqrt(c) * SQR(sinh(0.25*(psi - psi_e)));
  }

  inline void get_jacobian(double c, double psi, double J[][2]) {
      J[0][0] = 1.0/sqrt(c) * sinh(0.5*(psi - psi_e));       // dq/dc
      J[0][1] = sqrt(c) * cosh(0.5*(psi - psi_e));           // dq/dpsi
      J[1][0] = 2.0/sqrt(c) * SQR(sinh(0.25*(psi - psi_e))); // dw/dc
      J[1][1] = sqrt(c) * sinh(0.5*(psi - psi_e));           // dw/dpsi
  }

  ExplicitNonLinearChargingSolver(const ExplicitNonLinearChargingSolver& other);
  ExplicitNonLinearChargingSolver& operator=(const ExplicitNonLinearChargingSolver& other);

public:
  ExplicitNonLinearChargingSolver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_);
  ~ExplicitNonLinearChargingSolver();

#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif

  inline void set_parameters(double dt, double lambda, double psi_e){
    this->dt     = dt;
    this->lambda = lambda;
    this->psi_e  = psi_e;
  }

  void init();
  void solve();
  void write_vtk(const std::string& filename);
};

#endif // CHARGING_NONLINEAR_EXPLICIT_2D_H
