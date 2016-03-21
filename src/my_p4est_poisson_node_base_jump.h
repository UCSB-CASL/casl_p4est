#ifndef POISSON_SOLVER_NODE_BASE_JUMP_H
#define POISSON_SOLVER_NODE_BASE_JUMP_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/voronoi2D.h>
#endif

class PoissonSolverNodeBaseJump
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

#ifdef P4_TO_P8
  class ZERO: public CF_3
  {
  public:
    double operator()(double, double, double) const
    {
      return 0;
    }
  } zero;

  class MU_CONSTANT: public CF_3
  {
  private:
    double cst;
  public:
    MU_CONSTANT() { cst = 1; }
    void set(double cst) { this->cst = cst; }
    double operator()(double, double, double) const
    {
      return cst;
    }
  } mu_constant;

  class ADD_CONSTANT: public CF_3
  {
  private:
    double cst;
  public:
    ADD_CONSTANT() { cst = 0; }
    void set(double cst) { this->cst = cst; }
    double operator()(double, double, double) const
    {
      return cst;
    }
  } add_constant;
#else
  class ZERO: public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero;

  class MU_CONSTANT: public CF_2
  {
  private:
    double cst;
  public:
    MU_CONSTANT() { cst = 1; }
    void set(double cst) { this->cst = cst; }
    double operator()(double, double) const
    {
      return cst;
    }
  } mu_constant;

  class ADD_CONSTANT: public CF_2
  {
  private:
    double cst;
  public:
    ADD_CONSTANT() { cst = 0; }
    void set(double cst) { this->cst = cst; }
    double operator()(double, double) const
    {
      return cst;
    }
  } add_constant;
#endif


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

#ifdef P4_TO_P8
  BoundaryConditions3D *bc;
#else
  BoundaryConditions2D *bc;
#endif

  my_p4est_interpolation_nodes_t interp_phi;
  my_p4est_interpolation_nodes_t rhs_m;
  my_p4est_interpolation_nodes_t rhs_p;

  bool local_mu;
  bool local_add;
  bool local_u_jump;
  bool local_mu_grad_u_jump;

#ifdef P4_TO_P8
  CF_3 *mu_m, *mu_p;
  CF_3 *add;
  CF_3 *u_jump;
  CF_3 *mu_grad_u_jump;
#else
  CF_2 *mu_m, *mu_p;
  CF_2 *add;
  CF_2 *u_jump;
  CF_2 *mu_grad_u_jump;
#endif

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  KSP ksp;
  PetscErrorCode ierr;

  bool is_voronoi_partition_constructed;
  bool is_matrix_computed;
  int matrix_has_nullspace;

  // disallow copy ctr and copy assignment
  PoissonSolverNodeBaseJump(const PoissonSolverNodeBaseJump& other);
  PoissonSolverNodeBaseJump& operator=(const PoissonSolverNodeBaseJump& other);

  PetscErrorCode VecCreateGhostVoronoiRhs();

public:
  void compute_voronoi_points();
#ifdef P4_TO_P8
  void compute_voronoi_cell(unsigned int n, Voronoi3D &voro) const;
#else
  void compute_voronoi_cell(unsigned int n, Voronoi2D &voro) const;
#endif
  void print_voronoi_VTK(const char* path) const;
  void setup_linear_system();
  void setup_negative_laplace_rhsvec();

  PoissonSolverNodeBaseJump(const my_p4est_node_neighbors_t *node_neighbors, const my_p4est_cell_neighbors_t *cell_neighbors);
  ~PoissonSolverNodeBaseJump();

  void set_phi(Vec phi);

  void set_rhs(Vec rhs_m, Vec rhs_p);

  void set_diagonal(double add);

  void set_diagonal(Vec add);

#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D& bc);
#else
  void set_bc(BoundaryConditions2D& bc);
#endif

  void set_mu(double mu);

  void set_mu(Vec mu_m, Vec mu_p);

  void set_u_jump(Vec u_jump);

  void set_mu_grad_u_jump(Vec mu_grad_u_jump);

  inline bool get_matrix_has_nullspace(void) const { return matrix_has_nullspace; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPCG, PCType pc_type = PCSOR);

  double interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n) const;
  void interpolate_solution_from_voronoi_to_tree(Vec solution) const;

  void write_stats(const char *path) const;

  void check_voronoi_partition() const;
};

#endif // POISSON_SOLVER_NODE_BASE_JUMP_H
