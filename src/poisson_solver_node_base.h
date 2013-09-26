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

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs_, phi_, add_, phi_xx, phi_yy;
  KSP ksp;
  PetscErrorCode ierr;

  void init();
  void preallocate_matrix();
  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();
  inline PetscInt petsc_node_gloidx(p4est_locidx_t p4est_node_locidx)
  {
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, p4est_node_locidx);
    int r, petsc_node_locidx;

    if (p4est_node_locidx < nodes->offset_owned_indeps){
      petsc_node_locidx = ni->p.piggy3.local_num;
      r = nodes->nonlocal_ranks[p4est_node_locidx];

    } else if (p4est_node_locidx >= nodes->offset_owned_indeps &&
               p4est_node_locidx <  nodes->offset_owned_indeps + nodes->num_owned_indeps) {
      petsc_node_locidx = p4est_node_locidx - nodes->offset_owned_indeps;
      r = p4est->mpirank;

    } else {
      petsc_node_locidx = ni->p.piggy3.local_num;
      r = nodes->nonlocal_ranks[p4est_node_locidx - nodes->num_owned_indeps];
    }

    return (PetscInt)(global_node_offset[r] + petsc_node_locidx);
  }

  // disallow copy ctr and copy assignment
  PoissonSolverNodeBase(const PoissonSolverNodeBase& other);
  PoissonSolverNodeBase& operator=(const PoissonSolverNodeBase& other);

public:
  PoissonSolverNodeBase(const my_p4est_node_neighbors_t *node_neighbors, my_p4est_brick_t *myb);
  ~PoissonSolverNodeBase();

  // inlines setters
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ? */
  void set_phi(Vec phi);
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)            {add_      = add; is_matrix_ready = false;}
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc; is_matrix_ready = false;}
  inline void set_mu(double mu)                {mu_       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};

#endif // POISSON_SOLVER_NODE_BASE_H
