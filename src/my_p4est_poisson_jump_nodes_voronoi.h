#ifndef MY_P4EST_POISSON_JUMP_NODES_VORONOI_H
#define MY_P4EST_POISSON_JUMP_NODES_VORONOI_H

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

class my_p4est_poisson_jump_nodes_voronoi_t
{
  typedef struct
  {
    double val;
    PetscInt n;
  } mat_entry_t;

  typedef struct
  {
    unsigned int n;
    unsigned int k;
  } check_comm_t;

  typedef struct
  {
    p4est_locidx_t local_num;
    double x;
    double y;
#ifdef P4_TO_P8
    double z;
#endif
  } voro_seed_comm_t;

  typedef struct
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
  } projected_point_t;

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
  };

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
  };
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
  };

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
  };
#endif

  MU_CONSTANT   mu_constant_m, mu_constant_p;
  ADD_CONSTANT  add_constant_m, add_constant_p;

  const my_p4est_node_neighbors_t *ngbd_n;
  const my_p4est_cell_neighbors_t *ngbd_c;

  // p4est objects
  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  double dxyz_min_[P4EST_DIM];
  double d_min;
  double diag_min;
  const double close_distance_factor = 10.0; // seems arbitrary to me, should be greater than 1.0 for sure...

  Vec phi;
  Vec rhs;
  Vec sol_voro;
  unsigned int num_local_voro; // number of locally owned Voronoi seeds
  /*!
   * \brief voro_seeds: vector of Voronoi seeds, contains num_local_voro
   * locally owned Voronoi seeds, first, then all the ghost Voronoi seeds
   */
#ifdef P4_TO_P8
  std::vector<Point3> voro_seeds;
#else
  std::vector<Point2> voro_seeds;
#endif
  /*!
   * \brief grid2voro: map between grid nodes and closest Voronoi seeds
   * grid2voro[k] is a vector of local indices in Voronoi seeds in
   * voro_seeds that are close to grid node of local index k
   */
  std::vector< std::vector<size_t> > grid2voro;

  /*!
   * \brief voro_global_offset: each rank's offset to compute global index
   * for voro points, vector of size p4est->mpisize
   */
  std::vector<PetscInt> voro_global_offset;

  /*!
   * \brief voro_ghost_local_num: remote local number for ghost Voronoi seeds.
   * A voronoi seed is ghost if n>=num_local_voro
   * - if the Voronoi seed is local, then its local number local_num is
   * its location in the voro_seeds array
   * - if the Voronoi seed is ghost, then its local number local_num is
   * local_num = voro_ghost_local_num[n-local_num_voro]
   * the global index is then local_num + global_voro_offset[owner's rank]
   */
  std::vector<p4est_locidx_t> voro_ghost_local_num;

  /*!
   * \brief voro_ghost_rank: rank of the owner process of a ghost Voronoi seed.
   * A voronoi seed is ghost if n>=num_local_voro, the rank of its owner process \
   * is voro_ghost_rank[n-local_num_voro]
   */
  std::vector<p4est_locidx_t> voro_ghost_rank;

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
//  CF_3 *add;
  CF_3 *add_m, *add_p;
  CF_3 *u_jump;
  CF_3 *mu_grad_u_jump;
#else
  CF_2 *mu_m, *mu_p;
//  CF_2 *add;
  CF_2 *add_m, *add_p;
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
  my_p4est_poisson_jump_nodes_voronoi_t(const my_p4est_poisson_jump_nodes_voronoi_t& other);
  my_p4est_poisson_jump_nodes_voronoi_t& operator=(const my_p4est_poisson_jump_nodes_voronoi_t& other);

  PetscErrorCode VecCreateGhostVoronoiRhs();

public:
  /*!
   * \brief compute_voronoi_points constructs the distributed list of voronoi seeds.
   * - every grid node that is far from the interface (i.e. that is not a vertex of a quad
   * crossed by the interface) becomes the seed of a Voronoi cell;
   * - wall grid nodes are forced to be Voronoi seeds as well;
   * - for grid nodes that are close to the interface, their projection on the interface are
   * calculated if it is safe to do so (i.e. if the local normal is not ill-defined). Otherwise,
   * it is an under-resolved situation (e.g. the grid node is the center of an under-resolved
   * sphere) and the grid node is forced to be a Voronoi seed as well.
   * Doing so, the interface is somehow sampled. The sampling points are then treated such that
   * two of them are not closer than diag_min/close_distance_factor from one another (this is
   * ensured globally, by treating the projected points associated with ghost layers' grid nodes
   * first, in a globally consistent way. Note that the result is NOT partition-independent).
   * - Once those sampling nodes are determines, the mirror points across the interface are
   * calculated and attributed to the process owning the quadrant to which they belong. Ghost
   * Voronoi seeds are determined and the corresponding data structures are updated.
   * At termination,
   * - voro_seeds is constructed, contains the Voronoi seeds locally owned first, then ghost
   * Voronoi seeds;
   * - grid2voro, voro_ghost_local_num and voro_ghost_rank are constructed;
   */
  void compute_voronoi_points();

  /*!
   * \brief compute_voronoi_cell
   * \param n
   * \param voro
   */
#ifdef P4_TO_P8
  void compute_voronoi_cell(unsigned int n, Voronoi3D &voro) const;
#else
  void compute_voronoi_cell(unsigned int n, Voronoi2D &voro) const;
#endif
  void print_voronoi_VTK(const char* path) const;
  void setup_linear_system();
  void setup_negative_laplace_rhsvec();

  my_p4est_poisson_jump_nodes_voronoi_t(const my_p4est_node_neighbors_t *node_neighbors, const my_p4est_cell_neighbors_t *cell_neighbors);
  ~my_p4est_poisson_jump_nodes_voronoi_t();

  void set_phi(Vec phi);

  void set_rhs(Vec rhs_m, Vec rhs_p);

  void set_diagonal(double add_) {set_diagonals(add_, add_);}
  void set_diagonals(double add_m_, double add_p_);

  void set_diagonal(Vec add_) {set_diagonals(add_, add_);}
  void set_diagonals(Vec add_m_, Vec add_p_);

#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D& bc);
#else
  void set_bc(BoundaryConditions2D& bc);
#endif

  void set_mu(double mu_) {set_mu(mu_, mu_);}
  void set_mu(double mu_m_, double mu_p_);

  void set_mu(Vec mu_) {set_mu(mu_, mu_);}
  void set_mu(Vec mu_m_, Vec mu_p_);

  void set_u_jump(Vec u_jump);

  void set_mu_grad_u_jump(Vec mu_grad_u_jump);

  inline bool get_matrix_has_nullspace(void) const { return matrix_has_nullspace; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);

  double interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n) const;
  void interpolate_solution_from_voronoi_to_tree(Vec solution) const;

  void write_stats(const char *path) const;

  void check_voronoi_partition() const;
};

#endif // MY_P4EST_POISSON_JUMP_NODES_VORONOI_H
