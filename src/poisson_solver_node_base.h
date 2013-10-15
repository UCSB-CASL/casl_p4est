#ifndef POISSON_SOLVER_NODE_BASE_H
#define POISSON_SOLVER_NODE_BASE_H

#include <petsc.h>

#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/interpolating_function.h>
#include <src/utils.h>

class PoissonSolverNodeBase
{
  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb_;
  InterpolatingFunction phi_interp;

  double mu_, diag_add_;
  bool is_matrix_ready;
  bool matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
  BoundaryConditions2D *bc_;
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs_, phi_, add_, phi_xx_, phi_yy_;
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
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ? */
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)            {add_      = add; is_matrix_ready = false;}
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc; is_matrix_ready = false;}
  inline void set_mu(double mu)                {mu_       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};

#endif // POISSON_SOLVER_NODE_BASE_H
