#ifndef CHARGING_LINEAR_EXPLICIT_2D_H
#define CHARGING_LINEAR_EXPLICIT_2D_H

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

class ExplicitLinearChargingSolver{
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

  Vec psi, rhs, sol;
  PetscErrorCode ierr;

  Vec G;
  InterpolatingFunctionNodeBase G_interp;
  PoissonSolverNodeBase psi_solver;

#ifdef P4_TO_P8
  BoundaryConditions3D psi_bc;
  class: public WallBC3D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */, double /* z */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } wall_bc;

  class:public CF_3{
  public:
    double operator()(double /* x */, double /* y */, double /* z */) const {
        return 0.0;
    };
  } wall_psi_value;
#else
  BoundaryConditions2D psi_bc;

  class: public WallBC2D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } wall_bc;

  class:public CF_2{
  public:
    double operator()(double /* x */, double /* y */) const {
        return 0.0;
    }
  } wall_psi_value;
#endif
  double lambda, dt;

  void solve_potential();
  void solve_concentration();

  ExplicitLinearChargingSolver(const ExplicitLinearChargingSolver& other);
  ExplicitLinearChargingSolver& operator=(const ExplicitLinearChargingSolver& other);

public:
  ExplicitLinearChargingSolver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_);
  ~ExplicitLinearChargingSolver();

#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif

  inline void set_parameters(double dt, double lambda){
    this->dt     = dt;
    this->lambda = lambda;
  }

  void init();
  void solve();
  void write_vtk(const std::string& filename);
};

#endif // CHARGING_LINEAR_EXPLICIT_2D_H
