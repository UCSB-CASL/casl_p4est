#ifndef POISSON_SOLVER_NODE_BASE_H
#define POISSON_SOLVER_NODE_BASE_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function_host.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_faces.h>
#include <src/voronoi3D.h>
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

#ifdef P4_TO_P8
  vector<Voronoi3D> voro;
#else
  vector<Voronoi2D> voro;
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

  void preallocate_matrix(int dir);

  void compute_voronoi_cell_u(p4est_locidx_t u_idx);
  void compute_voronoi_cell_v(p4est_locidx_t v_idx);
#ifdef P4_TO_P8
  void compute_voronoi_cell_w(p4est_locidx_t w_idx);
#endif

  void setup_linear_system_u();
  void setup_linear_system_v();
#ifdef P4_TO_P8
  void setup_linear_system_w();
#endif

  // disallow copy ctr and copy assignment
  PoissonSolverFaces(const PoissonSolverFaces& other);
  PoissonSolverFaces& operator=(const PoissonSolverFaces& other);

  inline p4est_topidx_t face_global_number(p4est_locidx_t f_idx, int dir)
  {
    if(f_idx<faces->num_local[dir])
      return f_idx + proc_offset[dir][p4est->mpirank];
    f_idx -= faces->num_local[dir];
    return faces->ghost_local_num[dir][f_idx] + proc_offset[dir][faces->nonlocal_ranks[dir][f_idx]];
  }

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

  inline bool is_nullspace_u() { return matrix_has_nullspace_u; }

  inline bool is_nullspace_v() { return matrix_has_nullspace_v; }

#ifdef P4_TO_P8
  void solve(Vec solution_u, Vec solution_v, Vec solution_w, bool use_nonzero_initial_guess=false, KSPType ksp_type=KSPBCGS, PCType pc_type=PCSOR);
#else
  void solve(Vec solution_u, Vec solution_v, bool use_nonzero_initial_guess=false, KSPType ksp_type=KSPBCGS, PCType pc_type=PCSOR);
#endif

  void solve_u(Vec solution_u);
  void solve_v(Vec solution_v);
#ifdef P4_TO_P8
  void solve_w(Vec solution_w);
#endif

  void print_partition_u_VTK(const char *file);
};

#endif // POISSON_SOLVER_NODE_BASE_H
