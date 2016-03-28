#ifndef MY_P4EST_POISSON_JUMP_VORONOI_BLOCK_H
#define MY_P4EST_POISSON_JUMP_VORONOI_BLOCK_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/voronoi2D.h>
#endif

class my_p4est_poisson_jump_voronoi_block_t
{
  typedef struct check_comm
  {
    unsigned int n;
    unsigned int k;
  } check_comm_t;

  typedef struct voro_comm
  {
    p4est_locidx_t local_num;
    double x;
    double y;
#ifdef P4_TO_P8
    double z;
#endif
  } voro_comm_t;

  typedef struct added_point
  {
    double x;
    double y;
#ifdef P4_TO_P8
    double z;
#endif
    double dx;
    double dy;
#ifdef P4_TO_P8
    double dz;
#endif
  } added_point_t;

  int block_size;
  const my_p4est_node_neighbors_t *ngbd_n;
  const my_p4est_cell_neighbors_t *ngbd_c;

  // p4est objects
  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;

  double xmin, xmax;
  double ymin, ymax;
#ifdef P4_TO_P8
  double zmin, zmax;
#endif

  double dx_min, dy_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
  double d_min;
  double diag_min;

  Vec phi;
  Vec rhs;
  Vec sol_voro;
  unsigned int num_local_voro;
#ifdef P4_TO_P8
  std::vector<Point3> voro_points;
#else
  std::vector<Point2> voro_points;
#endif
  std::vector< std::vector<size_t> > grid2voro;

  /* each rank's offset to compute global index for voro points */
  std::vector<PetscInt> voro_global_offset;

  /* ranks of the owners of the voro ghost points */
  std::vector<p4est_locidx_t> voro_ghost_rank;

  /*
   * remote local number for ghost points. a point is ghost if n>=num_local_voro
   * - if the voro point is local, then its local number is its location in the voro array
   * - if the voro point is ghost, then its local number is voro_ghost_local_num[n-local_num_voro]
   * the global index is then local_num + global_voro_offset[owner's rank]
   */
  std::vector<p4est_locidx_t> voro_ghost_local_num;

  my_p4est_interpolation_nodes_t interp_phi;
  vector<my_p4est_interpolation_nodes_t*> rhs_m;
  vector<my_p4est_interpolation_nodes_t*> rhs_p;

#ifdef P4_TO_P8
  typedef CF_3 cf_t;
  typedef BoundaryConditions3D bc_t;
#else
  typedef CF_2 cf_t;
  typedef BoundaryConditions2D bc_t;
#endif
  vector<vector<cf_t*>> mu_m, mu_p, add;
  vector<cf_t*> u_jump, mu_grad_u_jump;
  vector<bc_t> bc;

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  PetscErrorCode ierr;

  bool is_voronoi_partition_constructed;
  bool is_matrix_computed;
  int matrix_has_nullspace;

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_voronoi_block_t(const my_p4est_poisson_jump_voronoi_block_t& other);
  my_p4est_poisson_jump_voronoi_block_t& operator=(const my_p4est_poisson_jump_voronoi_block_t& other);

  PetscErrorCode VecCreateGhostVoronoiRhs();

  void inverse(double** mue, double** mue_inv);
  void matmult(double** mue_1, double **mue_2, double **mue);

public:
  void compute_voronoi_points();
#ifdef P4_TO_P8
  void compute_voronoi_cell(unsigned int n, Voronoi3D &voro) const;
#else
  void compute_voronoi_cell(unsigned int n, Voronoi2D &voro) const;
#endif
  void print_voronoi_VTK(const char* path) const;
  void setup_linear_system();
//  void setup_negative_laplace_rhsvec();

  my_p4est_poisson_jump_voronoi_block_t(int block_size, const my_p4est_node_neighbors_t *node_neighbors, const my_p4est_cell_neighbors_t *cell_neighbors);
  ~my_p4est_poisson_jump_voronoi_block_t();

  void set_phi(Vec phi);

  void set_rhs(Vec rhs_m[], Vec rhs_p[]);

  void set_diagonal(vector<vector<cf_t*>> &add);

  void set_bc(vector<bc_t>& bc);

  void set_mu(vector<vector<cf_t*>> &mu_m, vector<vector<cf_t*> >& mu_p);

  void set_u_jump(vector<cf_t*> &u_jump);

  void set_mu_grad_u_jump(vector<cf_t*> &mu_grad_u_jump);

  inline bool get_matrix_has_nullspace(void) const { return matrix_has_nullspace; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void solve(Vec solution[], bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);

  void interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n, vector<double>& vals) const;
  void interpolate_solution_from_voronoi_to_tree(Vec solution[]) const;

  void write_stats(const char *path) const;

  void check_voronoi_partition() const;
};

#endif // MY_P4EST_POISSON_JUMP_VORONOI_BLOCK_H
