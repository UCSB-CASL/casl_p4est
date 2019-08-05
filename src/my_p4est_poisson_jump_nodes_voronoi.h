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

struct error_sample
{
  double error_value;
  double error_location_x;
  double error_location_y;
#ifdef P4_TO_P8
  double error_location_z;
#endif
  void operator=(error_sample& rhs)
  {
    this->error_value       = rhs.error_value;
    this->error_location_x  = rhs.error_location_x;
    this->error_location_y  = rhs.error_location_y;
#ifdef P4_TO_P8
    this->error_location_z  = rhs.error_location_z;
#endif
  }
  bool operator>(error_sample& rhs)
  {
    return (this->error_value > rhs.error_value);
  }
  error_sample()
  {
    this->error_value       = 0.0;
    this->error_location_x  = -DBL_MAX;
    this->error_location_y  = -DBL_MAX;
#ifdef P4_TO_P8
    this->error_location_z  = -DBL_MAX;
#endif
  }
  error_sample(double value, double x_loc, double y_loc
             #ifdef P4_TO_P8
               , double z_loc
             #endif
               )
  {
    this->error_value       = value;
    this->error_location_x  = x_loc;
    this->error_location_y  = y_loc;
#ifdef P4_TO_P8
    this->error_location_z  = z_loc;
#endif
  }
};

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

  struct neighbor_seed
  {
    size_t local_seed_idx;
    double distance;
    inline bool operator <(const neighbor_seed& rhs_seed) const
    {
      return (this->distance < rhs_seed.distance);
    }
  };


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
   * - Once those sampling nodes are determined, the mirror points across the interface are
   * calculated and attributed to the process owning the quadrant to which they belong. Ghost
   * Voronoi seeds are determined and the corresponding data structures are updated.
   * At termination,
   * - voro_seeds is constructed, contains the Voronoi seeds locally owned first, then ghost
   * Voronoi seeds;
   * - grid2voro, voro_ghost_local_num and voro_ghost_rank are constructed;
   */
  void compute_voronoi_points();

  /*!
   * \brief compute_voronoi_cell: creates the Voronoi cell associated with the seed voro_seeds[seed_idx]
   * \param seed_idx: index of the Voronoi seed
   * \param voro: Voronoi2D/3D object to be constructed, i.e. the Voronoi cell of seed voro_seeds[n]
   * - If voro_seeds[n] is a grid node, the algorithm finds all direct neighbor quadrants and their
   * face neighbors (+ their edge neighbors in 3D), i.e. for a uniform grid in 2D where '*' is the
   * seed of interest, the following quadrants are found (enumerated in order)
   *
   *                     ---------------------
   *                     |         |         |
   *                     |    3    |    5    |
   *                     |         |         |
   *                     |         |         |
   *           -----------------------------------------
   *           |         |         |         |         |
   *           |    3    |    2    |    4    |    5    |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           --------------------*--------------------
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |    1    |    0    |    6    |    7    |
   *           |         |         |         |         |
   *           -----------------------------------------
   *                     |         |         |
   *                     |         |         |
   *                     |    1    |    7    |
   *                     |         |         |
   *                     ---------------------
   * - If voro_seeds[n] is _not_ a grid node, the algorithm finds the owner quadrant first, then all
   * its face neighbors (note: no edge neighbor in 3D) and their own face neighbors, as well. Corner
   * neighbors of the owner quadrant are added as well ([Raphael]: isn't that irrelevant in 2D?),
   * i.e. for a uniform grid in 2D where '*' is the seed of interest, the following quadrants are found
   *
   *                               -----------
   *                               |         |
   *                               |   2     |
   *                               |         |
   *                               |         |
   *                     -------------------------------
   *                     |         |         |         |
   *                     |   2     |   1     |   2     |
   *                     |         |         |         |
   *                     |         |         |         |
   *           ---------------------------------------------------
   *           |         |         |         |         |         |
   *           |   2     |   1     |   0     |   1     |    2    |
   *           |         |         |         |         |         |
   *           |         |         | *       |         |         |
   *           ---------------------------------------------------
   *                     |         |         |         |
   *                     |   2     |   1     |   2     |
   *                     |         |         |         |
   *                     |         |         |         |
   *                     -------------------------------
   *                               |         |
   *                               |    2    |
   *                               |         |
   *                               |         |
   *                               -----------
   *
   * - Then, for all grid node (local index v) being a vertex of one of those neighbor quadrants
   * of the voronoi seed of interest, the voronoi seeds pointed by the grid2voro[v] array are
   * added as potential candidates for neighbor Voronoi seeds of the Voronoi cell to be constructed.
   * The Voronoi cell is then constructed correspondingly.
   * [Raphael's note: shouldn't the second case be an overlap of the P4EST_CHILDREN first case
   * scenarii where the node of interest is one of the P4EST_CHILDREN vertices in the owner cell?
   * That would ensure geomertical consistency and avoid ill-behaved problems when points are very
   * close to vertices or numerically considered as non-vertices]
   */
#ifdef P4_TO_P8
  void compute_voronoi_cell(unsigned int seed_idx, Voronoi3D &voro) const;
#else
  void compute_voronoi_cell(unsigned int seed_idx, Voronoi2D &voro) const;
#endif
  /*!
   * \brief push_quad_idx_to_list adds a local quadrant index to a list if not already in it
   * \param loc_quad_idx: local index of the quadrant under consideration
   * \param list_of_quad_idx: list of local quadrant indices
   */
  inline void push_quad_idx_to_list(const p4est_locidx_t loc_quad_idx, std::vector<p4est_locidx_t>& list_of_quad_idx) const
  {
    if(loc_quad_idx >= 0) // invalid quad if loc_quad_idx < 0
    {
      bool add_it = true;
      for(unsigned int nn=0; nn<list_of_quad_idx.size(); ++nn)
        if(list_of_quad_idx[nn] == loc_quad_idx)
        {
          add_it = false;
          break;
        }
      if(add_it)
      {
#ifndef P4_TO_P8
        std::cout << std::endl;
        std::cout << "GUESS WHAT; I'm NOT useless" << std::endl; // see comment above in compute_voronoi_cell
        std::cout << std::endl;
#endif
        list_of_quad_idx.push_back(loc_quad_idx);
      }
    }
  }
  /*!
   * \brief setup_linear_system: self-explanatory, core of the solver, see Arthur's paper
   * "Solving elliptic problems with discontinuities on irregular domains – the Voronoi Interface Method"
   */
  void setup_linear_system();
  /*!
   * \brief setup_negative_laplace_rhsvec: self-explanatory, core of the solver, see Arthur's paper
   * "Solving elliptic problems with discontinuities on irregular domains – the Voronoi Interface Method"
   */
  void setup_negative_laplace_rhsvec();

  void print_voronoi_VTK(const char* path) const;
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

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR, const bool destroy_solution_on_voronoi_mesh = true);

  void destroy_solution()
  {
    if(sol_voro != PETSC_NULL)
    {
      ierr = VecDestroy(sol_voro); CHKERRXX(ierr); sol_voro = NULL;
    }
  }

  /*!
   * \brief interpolate_solution_from_voronoi_to_tree_on_node_n self-explanatory
   * \param n: local index of the grid node of interest
   * - If the grid node of interest is a Voronoi seed the solution is simply read from
   * the appropriate Voronoi cell
   * - Else, the algorithm finds the (P4EST_DIM + 1) closest Voronoi seeds that create
   * a non-degenerate simplex around (or close to) the grid node of interest and
   * interpolates the solution from those Voronoi seeds, the interpolation is linear.
   * The procedure first finds the two closest Voronoi seeds p0 and p1. The third point
   * p2 is defined as the closest point such that the angles (p1, p0, p2) and
   * (p0, p1, p2) (modulo PI) are both either greater than PI/5 (or PI/10 in 3D) or
   * smaller than 4*PI/5 (or 9*PI/10 in 3D).
   * In 3D, a fourht point p3 is required. It's chosen as the closest Voronoi seed such
   * that the height of the tetrahedron (p0, p1, p2, p3) from base (p0, p1, p2) is
   * greater than 2.0*diag_min/close_distance_factor.
   * The points p0, p1, p2 and p3 are found among all the Voronoi seeds pointed by all
   * the grid2voro[v] arrays where v is a vertex of one of all the two-layer neighbor
   * quadrants of the grid node of interest, i.e. for a uniform grid in 2D where '*' is
   * the grid node of interest, the vertices of the following quadrants are found
   *
   *           -----------------------------------------
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           -----------------------------------------
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           --------------------*--------------------
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           -----------------------------------------
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           |         |         |         |         |
   *           -----------------------------------------
   * \return
   */
  double interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n) const;
  /*!
   * \brief interpolate_solution_from_voronoi_to_tree self-explanatory
   * \param[inout] solution: node-based vector to be calculated
   * calls interpolate_solution_from_voronoi_to_tree_on_node_n() for all local grid nodes
   */
  void interpolate_solution_from_voronoi_to_tree(Vec solution) const;

  /*!
   * \brief write_stats writes statistics about the Voronoi seeds
   * \param path absolute path of the statistics file to be written
   * The method opens a file of the given absolute path and writes a file
   * of the table form
   * % rank  |  total number of Voronoi seeds locally owned  |  ... that are ...  |  local grid nodes  |  close to a local grid node  |  ghost grid nodes  |  close to a ghost grid node  |  validity check
   * which is self-explanatory.
   * The validity check checks that
   * 1) no proc claims a Voronoi seed that is one of its ghost grid nodes;
   * 2) all Voronoi seeds that are owned by a processor are either local grid nodes, or
   * kept track of within the grid2voro struc as a Voronoi seed nearby a local node or
   * a ghost grid node
   */
  void write_stats(const char *path) const;

  /*!
   * \brief check_voronoi_partition checks that neighbor relationships in Voronoi seeds
   * are consistent: loop over every local Voronoi seed, and checks that all its neighbor
   * seeds consider itself as a neighbor, as well.
   * If an inconsistency is found, an information message is printed on std::cout
   */
  void check_voronoi_partition() const;

  void get_max_error_at_seed_locations(error_sample& max_error_on_seeds, int& rank_max_error,
                                     #ifdef P4_TO_P8
                                       double (*exact_solution) (double, double, double),
                                     #else
                                       double (*exact_solution) (double, double),
                                     #endif
                                       const double& shift_value = 0.0
                                       )  const;
};
/*
POTENTIAL IMPROVEMENTS [Raphael]
--------------------------------
- call compute_voronoi_cell for all cells only once and store it in memory somehow;
- check_voronoi_partition prints warning messages in 3D for a grid 5/8 with small
surface areas--> inconsistent neighbors breaking down the symmetry of the linear
system, undesirable. Potential origins (imho) in the considered neighboring quads in
compute_voronoi_cell and/or in construct_partition (i.e. from voro++)
*/


#endif // MY_P4EST_POISSON_JUMP_NODES_VORONOI_H
