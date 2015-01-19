#ifndef POISSON_SOLVER_NODE_BASE_JUMP_H
#define POISSON_SOLVER_NODE_BASE_JUMP_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function_host.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function_host.h>
#include <src/my_p4est_utils.h>
#endif

#include <src/voronoi2D.h>

class PoissonSolverNodeBaseJump
{
  const my_p4est_node_neighbors_t *ngbd_n;
  const my_p4est_cell_neighbors_t *ngbd_c;

  // p4est objects
  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;

  double xmin, xmax;
  double ymin, ymax;
  double zmin, zmax;

  double dx_min, dy_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
  double d_min;
  double diag_min;

  Vec phi;
  Vec rhs;
  Vec sol_voro;
  std::vector<Voronoi2D> voro;
  std::vector< std::vector<size_t> > grid2voro;

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  BoundaryConditions2D *bc;

  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  KSP ksp;
  PetscErrorCode ierr;

  bool matrix_has_nullspace;
  bool is_matrix_computed;

  void preallocate_matrix();

  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();

  // disallow copy ctr and copy assignment
  PoissonSolverNodeBaseJump(const PoissonSolverNodeBaseJump& other);
  PoissonSolverNodeBaseJump& operator=(const PoissonSolverNodeBaseJump& other);

public:
  void compute_voronoi_mesh();

  PoissonSolverNodeBaseJump(const my_p4est_node_neighbors_t *node_neighbors, const my_p4est_cell_neighbors_t *cell_neighbors);
  ~PoissonSolverNodeBaseJump();

  void set_phi(Vec phi);

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void shift_to_exact_solution(Vec sol, Vec uex);

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
};

#endif // POISSON_SOLVER_NODE_BASE_JUMP_H
