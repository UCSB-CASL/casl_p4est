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
#include <src/my_p4est_interpolating_function_host.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_faces.h>
#include <src/voronoi2D.h>
#endif

class PoissonSolverFaces
{
  const my_p4est_faces_t *faces;
  const p4est_t *p4est;
  const my_p4est_cell_neighbors_t *ngbd_c;
  const my_p4est_node_neighbors_t *ngbd_n;

  vector<p4est_gloidx_t> proc_offset[P4EST_DIM];

  double xmin, xmax;
  double ymin, ymax;
  double dx_min, dy_min;
#ifdef P4_TO_P8
  double dz_min;
  double zmin, zmax;
#endif

  InterpolatingFunctionNodeBaseHost interp_phi;
  Vec phi;

  Vec rhs_u, rhs_v;
#ifdef P4_TO_P8
  Vec rhs_w;
#endif

  double diag_add;
  double mu;

#ifdef P4_TO_P8
  const BoundaryConditions3D *bc_u;
  const BoundaryConditions3D *bc_v;
  const BoundaryConditions3D *bc_w;
#else
  const BoundaryConditions2D *bc_u;
  const BoundaryConditions2D *bc_v;
#endif

  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  PetscErrorCode ierr;

  int matrix_has_nullspace_u;
  int matrix_has_nullspace_v;
#ifdef P4_TO_P8
  int matrix_has_nullspace_w;
#endif

  void compute_voronoi_cell_u(p4est_locidx_t u_idx, Voronoi2D& voro) const;
  void preallocate_matrix_u();
  void setup_linear_system_u();

  // disallow copy ctr and copy assignment
  PoissonSolverFaces(const PoissonSolverFaces& other);
  PoissonSolverFaces& operator=(const PoissonSolverFaces& other);

public:
  PoissonSolverFaces(const my_p4est_faces_t *faces, const my_p4est_node_neighbors_t *ngbd);
  ~PoissonSolverFaces();

  void set_phi(Vec phi);

#ifdef P4_TO_P8
  void set_rhs(Vec rhs_u, Vec rhs_v, Vec rhs_w);
#else
  void set_rhs(Vec rhs_u, Vec rhs_v);
#endif

  void set_diagonal(double add);

  void set_mu(double mu);

#ifdef P4_TO_P8
  void set_bc(const BoundaryConditions3D& bc_u, const BoundaryConditions3D& bc_v, const BoundaryConditions3D& bc_w);
#else
  void set_bc(const BoundaryConditions2D& bc_u, const BoundaryConditions2D& bc_v);
#endif


  void solve(Vec solution_u, Vec solution_v);

  void solve_u(Vec solution_u);

  void print_partition_u_VTK();
};

#endif // POISSON_SOLVER_NODE_BASE_H
