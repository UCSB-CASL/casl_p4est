#ifndef POISSON_SOLVER_NODE_BASE_H
#define POISSON_SOLVER_NODE_BASE_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#endif

class PoissonSolverNodeBase
{
  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb_;
  InterpolatingFunctionNodeBase phi_interp;
#ifdef P4_TO_P8
  const CF_3* phi_cf;
#else
  const CF_2* phi_cf;
#endif

  double mu_, diag_add_;
  bool is_matrix_ready;
  int matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
#else
  BoundaryConditions2D *bc_;
#endif
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs_, phi_, add_, phi_xx_, phi_yy_;
#ifdef P4_TO_P8
  Vec phi_zz_;
#endif
  bool is_phi_dd_owned;
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();
  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();  

  // disallow copy ctr and copy assignment
  PoissonSolverNodeBase(const PoissonSolverNodeBase& other);
  PoissonSolverNodeBase& operator=(const PoissonSolverNodeBase& other);

public:
  PoissonSolverNodeBase(const my_p4est_node_neighbors_t *node_neighbors);
  ~PoissonSolverNodeBase();

  // inlines setters
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ?
   * Mohammad: Vec is just a typedef to _p_Vec* so its merely a pointer under the hood.
   * If you are only passing the vector to access its data its fine to pass it as 'Vec v'
   * However, if 'v' is supposed to change itself, i.e. the the whole Vec object and not just its data
   * then it should either be passed via reference, Vec& v, or pointer, Vec* v, just like
   * any other object
   */
#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)            {add_      = add; is_matrix_ready = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {bc_       = &bc; is_matrix_ready = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc; is_matrix_ready = false;}
#endif
  inline void set_mu(double mu)                {mu_       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};

#endif // POISSON_SOLVER_NODE_BASE_H
