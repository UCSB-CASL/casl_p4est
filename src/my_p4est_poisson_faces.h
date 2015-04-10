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

  Vec *rhs;

  Vec rhs_u, rhs_v;
#ifdef P4_TO_P8
  Vec rhs_w;
#endif

  double diag_add;
  double mu;

#ifdef P4_TO_P8
  const BoundaryConditions3D *bc;
#else
  const BoundaryConditions2D *bc;
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

  int matrix_has_nullspace[P4EST_DIM];

  void preallocate_matrix(int dir);

  void compute_voronoi_cell(p4est_locidx_t f_idx, int dir);

  void setup_linear_system(int dir);

  // disallow copy ctr and copy assignment
  PoissonSolverFaces(const PoissonSolverFaces& other);
  PoissonSolverFaces& operator=(const PoissonSolverFaces& other);

  void reset_linear_solver(bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type);

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

  void set_rhs(Vec *rhs);

  void set_diagonal(double add);

  void set_mu(double mu);

#ifdef P4_TO_P8
  void set_bc(const BoundaryConditions3D *bc);
#else
  void set_bc(const BoundaryConditions2D *bc);
#endif

//  inline bool is_nullspace_u() { return matrix_has_nullspace_u; }
//  inline bool is_nullspace_v() { return matrix_has_nullspace_v; }

  void solve(Vec *solution, bool use_nonzero_initial_guess=false, KSPType ksp_type=KSPBCGS, PCType pc_type=PCSOR);

  void print_partition_VTK(const char *file);
};

#endif // POISSON_SOLVER_NODE_BASE_H
