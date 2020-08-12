#ifndef MY_P4EST_TWO_PHASE_FLOWS_H
#define MY_P4EST_TWO_PHASE_FLOWS_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_poisson_jump_cells_fv.h>
#include <src/my_p8est_poisson_jump_cells_xgfm.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_interface_manager.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_jump_cells_fv.h>
#include <src/my_p4est_poisson_jump_cells_xgfm.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/voronoi2D.h>
#endif

static const double value_not_needed = NAN;

#if __cplusplus >= 201103L
#include <unordered_map> // if c++11 is fully supported, use unordered maps (i.e. hash tables) as they are apparently much faster
#else
#include <map>
#endif

using std::set;

class my_p4est_two_phase_flows_t
{
private:

  struct augmented_voronoi_cell
  {
    Voronoi_DIM voro;
    voro_cell_type cell_type;
    bool is_set, has_neighbor_across;
    augmented_voronoi_cell() : is_set(false), has_neighbor_across(false) {}
  };

  class splitting_criteria_computational_grid_two_phase_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est_np1, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const p4est_nodes_t* nodes_np1,
                      const double *phi_np1_on_computational_nodes_p,
                      const double *vorticity_magnitude_np1_on_computational_nodes_minus_p,
                      const double *vorticity_magnitude_np1_on_computational_nodes_plus_p);
    const my_p4est_two_phase_flows_t *owner;
  public:
    splitting_criteria_computational_grid_two_phase_t(my_p4est_two_phase_flows_t* parent_solver) :
      splitting_criteria_tag_t((splitting_criteria_t*)(parent_solver->p4est_n->user_pointer)), owner(parent_solver) {}
    bool refine_and_coarsen(p4est_t* p4est_np1, const p4est_nodes_t* nodes_np1,
                            Vec phi_np1_on_computational_nodes,
                            Vec vorticity_magnitude_np1_on_computational_nodes_minus,
                            Vec vorticity_magnitude_np1_on_computational_nodes_plus);
  };

  my_p4est_brick_t          *brick;
  p4est_connectivity_t      *conn;

  p4est_t                   *p4est_nm1;
  p4est_ghost_t             *ghost_nm1;
  p4est_nodes_t             *nodes_nm1;
  my_p4est_hierarchy_t      *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  p4est_t                   *p4est_n,     *fine_p4est_n;
  p4est_ghost_t             *ghost_n,     *fine_ghost_n;
  p4est_nodes_t             *nodes_n,     *fine_nodes_n;
  my_p4est_hierarchy_t      *hierarchy_n, *fine_hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n,      *fine_ngbd_n;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_faces_t          *faces_n;
  my_p4est_interface_manager_t  *interface_manager;

  my_p4est_poisson_jump_cells_t* pressure_guess_solver;
  bool pressure_guess_is_set;
  my_p4est_poisson_jump_cells_t* divergence_free_projector;
  poisson_jump_cell_solver_tag cell_jump_solver_to_use;
  bool fetch_interface_FD_neighbors_with_second_order_accuracy;

  const double *xyz_min, *xyz_max;
  double tree_dimension[P4EST_DIM];
  double dxyz_smallest_quad[P4EST_DIM];
  bool periodicity[P4EST_DIM];
  double tree_diagonal;

  double surface_tension;
  double mu_plus, mu_minus;
  double rho_plus, rho_minus;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_velocity_minus, max_L2_norm_velocity_plus;
  double uniform_band_minus, uniform_band_plus;
  double threshold_split_cell;
  double cfl_advection, cfl_surface_tension;
  bool   dt_updated;
  interpolation_method levelset_interpolation_method;

  int sl_order;

  const double threshold_dbl_max;

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM *bc_velocity;

  CF_DIM *force_per_unit_mass[P4EST_DIM];

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  Vec phi;
  Vec mass_flux; // mass_flux <-> jump in velocity
  Vec pressure_jump;
  // vector fields and/or other P4EST_DIM-block-structured
  Vec phi_xxyyzz, interface_stress;
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  Vec phi_on_computational_nodes;
  Vec vorticity_magnitude_minus, vorticity_magnitude_plus;
  // vector fields and/or other P4EST_DIM-block-structured
  Vec vnp1_nodes_minus,  vnp1_nodes_plus;
  Vec vn_nodes_minus,    vn_nodes_plus;
  Vec interface_velocity_np1; // yes, np1, yes! (used right after compute_dt in update_from_n_to_np1, so it looks like n but it's actually np1)
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vn_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_n
  Vec vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz, interface_velocity_np1_xxyyzz;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  Vec grad_p_guess_over_rho_minus[P4EST_DIM], grad_p_guess_over_rho_plus[P4EST_DIM];
  Vec vnp1_face_minus[P4EST_DIM], vnp1_face_plus[P4EST_DIM];
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields, P4EST_DIM-block-structured
  Vec vnm1_nodes_minus,  vnm1_nodes_plus;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vnm1_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n-1, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_nm1
  Vec vnm1_nodes_minus_xxyyzz, vnm1_nodes_plus_xxyyzz;

  // The value of the dir velocity component at the points at time n and nm1 backtraced from the face of orientation dir and local index f_idx are
  // backtraced_vn_faces[dir][f_idx] and backtraced_vnm1_faces[dir][f_idx].
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> backtraced_vn_faces_minus[P4EST_DIM], backtraced_vn_faces_plus[P4EST_DIM];
  std::vector<double> backtraced_vnm1_faces_minus[P4EST_DIM], backtraced_vnm1_faces_plus[P4EST_DIM]; // used only if sl_order == 2

  // viscosity solver
  bool voronoi_on_the_fly;
  vector<augmented_voronoi_cell> voro_cell[P4EST_DIM];
  class jump_face_solver
  {
    /* NOTE : inspired from my_p4est_poisson_faces_t with an xgfm twist for interface jump problems,
     * and without any Shortley-Weller kind of treatment (to keep symmetric systems).
     * There is a notable difference though in the treatment of Neumann boundary conditions, since we assume
     * that bc_v[dir].wallValue(...) is the prescribed value for mu*(normal derivative of u[dir]), everywhere,
     * as opposed as the prescribed value for (normal derivative of u[dir]) only in my_p4est_poisson_faces_t.
     */

    // NOTE for two-phase flows :
    // Most (in fact, ideally all) of your faces having a neighbor across the interface *SHOULD* be either of type parallelepiped_no_wall
    // or of type parallelepiped_with_wall. If you lived in an ideal world, you would enforce that and make sure that it's never invalidated...
    // HOWEVER, that would required costly inter-processor communications to ensure that the grid satisfies that requirement at all times
    // So the face_jump_solver owned by the two-phase flow solver allows "nonuniform" types to have one neighbor across the interface!
    // EXAMPLE of such a scenario:
    //
    // ___.2________3_____________4_________
    // |   .    |        |                 |
    // |    .   |        |                 |
    // |     .  |        |                 |
    // |___0_.__|___1____|                 |
    // |     .  |        |                 |
    // |     .  |        |                 |
    // |    .   |        |                 |
    // |   .    |        |                 |
    // ---.6--------7-------------5---------
    //
    // Face 1 here above has neighbor face 0 across the interface, but is not locally uniform. because of
    // neighbors 4 and 5.
    // (we want to ensure proper behavior even in such cases because ensuring that the grid)

  private:
    bool matrix_is_preallocated[P4EST_DIM];
    Mat matrix[P4EST_DIM];
    // if we do have a null space, it will be the constant nullspace
    // (unless some other border is added) -> no need to build it ourselves
    KSP ksp[P4EST_DIM];
    Vec rhs[P4EST_DIM];
    Vec solution[P4EST_DIM];
    int matrix_has_nullspace[P4EST_DIM];
    bool matrix_is_ready[P4EST_DIM], only_diags_are_modified[P4EST_DIM];
    double current_diag_minus[P4EST_DIM], current_diag_plus[P4EST_DIM];
    double desired_diag_minus[P4EST_DIM], desired_diag_plus[P4EST_DIM];
    bool ksp_is_set_from_options[P4EST_DIM], pc_is_set_from_options[P4EST_DIM];
    my_p4est_two_phase_flows_t* env;

    inline void reset_current_diagonals(const u_char &dir)
    {
      current_diag_minus[dir] = current_diag_plus[dir] = 0.0;
    }

    inline bool current_diags_are_as_desired(const u_char &dir) const
    {
      return (fabs(current_diag_minus[dir] - desired_diag_minus[dir]) < EPS*MAX(fabs(current_diag_minus[dir]), fabs(desired_diag_minus[dir])) || (fabs(current_diag_minus[dir]) < EPS && fabs(desired_diag_minus[dir]) < EPS))
          && (fabs(current_diag_plus[dir] - desired_diag_plus[dir]) < EPS*MAX(fabs(current_diag_plus[dir]), fabs(desired_diag_plus[dir])) || (fabs(current_diag_plus[dir]) < EPS && fabs(desired_diag_plus[dir]) < EPS));
    }

    inline void destroy_and_nullify_owned_members()
    {
      PetscErrorCode ierr;
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        if(matrix[dim] != NULL)   { ierr = MatDestroy(matrix[dim]);   CHKERRXX(ierr); matrix[dim] = NULL; }
        if(ksp[dim] != NULL)      { ierr = KSPDestroy(ksp[dim]);      CHKERRXX(ierr); ksp[dim]    = NULL; }
        if(rhs[dim] != NULL)      { ierr = VecDestroy(rhs[dim]);      CHKERRXX(ierr); rhs[dim]    = NULL; }
        if(solution[dim] != NULL) { ierr = VecDestroy(solution[dim]); CHKERRXX(ierr); solution[dim]    = NULL; }
      }
    }

    void preallocate_matrix(const u_char &dir);
    void setup_linear_system(const u_char &dir);
    void setup_linear_solver(const u_char &dir, const PetscBool &use_nonzero_initial_guess, const KSPType &ksp_type, const PCType &pc_type);

  public:
    jump_face_solver(my_p4est_two_phase_flows_t *parent_solver = NULL) : env(parent_solver)
    {
      /* initialize the KSP solvers and other parameters */
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        matrix_is_preallocated[dim]   = false;
        matrix[dim]                   = NULL;
        ksp[dim]                      = NULL;
        rhs[dim]                      = NULL;
        solution[dim]                 = NULL;
        matrix_has_nullspace[dim] = matrix_is_ready[dim] = only_diags_are_modified[dim]  = false;
        current_diag_minus[dim] = current_diag_plus[dim] = 0.0;
        desired_diag_minus[dim] = desired_diag_plus[dim] = 0.0;
        ksp_is_set_from_options[dim] = pc_is_set_from_options[dim] = false;
      }
    }
    ~jump_face_solver() { destroy_and_nullify_owned_members(); }

    inline void set_environment(my_p4est_two_phase_flows_t* env_)
    {
      env = env_;
      PetscErrorCode ierr;
      for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
        if(ksp[dir] == NULL)
        {
          ierr = KSPCreate(env->p4est_n->mpicomm, &ksp[dir]); CHKERRXX(ierr);
          ierr = KSPSetTolerances(ksp[dir], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
        }
      }
    }


    inline void set_diagonals(const double &add_minus, const double &add_plus)
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        desired_diag_minus[dim] = add_minus;
        desired_diag_plus[dim]  = add_plus;
        if(!current_diags_are_as_desired(dim))
        {
          // actual modification of diag, do not change the flag values otherwise, especially not of matrix_is_ready
          only_diags_are_modified[dim]  = matrix_is_ready[dim];
          matrix_is_ready[dim]          = false;
        }
      }
    }

    void solve(const PetscBool &use_nonzero_initial_guess = PETSC_FALSE, const KSPType &ksp_type = KSPCG, const PCType &pc_type = PCHYPRE);

    inline void reset()
    {
      destroy_and_nullify_owned_members();
      PetscErrorCode ierr;
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        matrix_is_preallocated[dim]   = false;
        matrix_has_nullspace[dim] = matrix_is_ready[dim] = only_diags_are_modified[dim]  = false;
        current_diag_minus[dim] = current_diag_plus[dim] = 0.0;
        desired_diag_minus[dim] = desired_diag_plus[dim] = 0.0;
        ksp_is_set_from_options[dim] = pc_is_set_from_options[dim] = false;
        if(ksp[dim] == NULL)
        {
          ierr = KSPCreate(env->p4est_n->mpicomm, &ksp[dim]); CHKERRXX(ierr);
          ierr = KSPSetTolerances(ksp[dim], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
        }
      }
    }

//    void get_vstar_velocities(Vec vnp1_face_minus[P4EST_DIM], Vec vnp1_face_plus[P4EST_DIM]);
    void extrapolate_face_velocities_across_interface(Vec vnp1_face_minus[P4EST_DIM], Vec vnp1_face_plus[P4EST_DIM], const u_int& n_iteration = 10, const u_char& degree = 1);
    void initialize_face_extrapolation(const p4est_locidx_t &f_idx, const u_char &dir, const double* sharp_solution_p[P4EST_DIM],
                                       double *vnp1_face_minus_p[P4EST_DIM], double *vnp1_face_plus_p[P4EST_DIM],
                                       double *normal_derivative_of_vnp1_face_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_face_plus_p[P4EST_DIM], const u_char& degree);
    void extrapolate_normal_derivatives_of_face_velocity_local(const p4est_locidx_t &f_idx, const u_char &dir,
                                                               double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM]);

    void face_velocity_extrapolation_local(const p4est_locidx_t &f_idx, const u_char &dir, const double* sharp_solution_p[P4EST_DIM],
                                           const double *normal_derivative_of_vnp1_face_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_face_plus_p[P4EST_DIM],
                                           double *vnp1_face_minus_p[P4EST_DIM], double *vnp1_face_plus_p[P4EST_DIM]);

  } viscosity_solver;

  inline bool face_is_dirichlet_wall(const p4est_locidx_t& face_idx, const u_char& dir) const
  {
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
    u_char loc_face_idx = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
    P4EST_ASSERT(faces_n->q2f(quad_idx, loc_face_idx) == face_idx);
    const p4est_quadrant_t* quad = fetch_quad(quad_idx, tree_idx, p4est_n, ghost_n);

    if(is_quad_Wall(p4est_n, tree_idx, quad, loc_face_idx))
    {
      double xyz_face[P4EST_DIM];
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      return (bc_velocity[dir].wallType(xyz_face) == DIRICHLET);
    }

    return false;
  }

  inline char sgn_of_wall_neighbor_of_face(const p4est_locidx_t& face_idx, const u_char &dir, const u_char &wall_dir, const double *xyz_wall = NULL)
  {
    if(xyz_wall != NULL)
      return (interface_manager->phi_at_point(xyz_wall) <= 0.0 ? -1 : +1);
    double xyz_w[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_w);
    xyz_w[wall_dir/2] = (wall_dir%2 == 1 ? xyz_max[wall_dir/2] : xyz_min[wall_dir/2]);
    return (interface_manager->phi_at_point(xyz_w) <= 0.0 ? -1 : +1);
  }
  inline char sgn_of_face(const p4est_locidx_t& face_idx, const u_char& dir, const double *xyz_face = NULL)
  {
    if(xyz_face != NULL)
      return (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
    double xyz_f[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_f);
    return (interface_manager->phi_at_point(xyz_f) <= 0.0 ? -1 : +1);
  }

  inline double BDF_alpha() const { return (sl_order == 1 ? 1.0 : (2.0*dt_n + dt_nm1)/(dt_n + dt_nm1)); }
  inline double BDF_beta() const  { return (sl_order == 1 ? 0.0 : -dt_n/(dt_n + dt_nm1));               }

  double div_mu_grad_u_dir(const p4est_locidx_t &face_idx, const u_char &dir, const double *vn_dir_p);


  inline const augmented_voronoi_cell& get_augmented_voronoi_cell(const p4est_locidx_t &face_idx, const u_char &dir)
  {
    augmented_voronoi_cell &my_cell = (voronoi_on_the_fly ? voro_cell[dir][0] : voro_cell[dir][face_idx]);
    if(!voronoi_on_the_fly && my_cell.is_set)
      return my_cell;

    P4EST_ASSERT(!my_cell.is_set);

    my_cell.cell_type = compute_voronoi_cell(my_cell.voro, faces_n, face_idx, dir, bc_velocity, NULL);
    P4EST_ASSERT(my_cell.cell_type != not_well_defined); // prohibited in two-phase flows, all Voronoi cell must be defined...

    my_cell.has_neighbor_across = false;
    const char sgn_face = sgn_of_face(face_idx, dir);

#ifndef P4_TO_P8
    if(my_cell.cell_type != dirichlet_wall_face && my_cell.cell_type != parallelepiped_no_wall)
      clip_voronoi_cell_by_parallel_walls(my_cell.voro, dir);
#endif

    if(my_cell.cell_type != dirichlet_wall_face)
    {
      const vector<ngbdDIMseed> *points;
      my_cell.voro.get_neighbor_seeds(points);
      for (size_t n = 0; n < points->size() && !my_cell.has_neighbor_across; ++n)
      {
        if((*points)[n].n >= 0)
          my_cell.has_neighbor_across = (sgn_face != sgn_of_face((*points)[n].n, dir));
        else // neighbor is a wall, but it could very well be across the interface too...
        {
          const char wall_dir = -1 - (*points)[n].n;
          P4EST_ASSERT(0 <= wall_dir && wall_dir < P4EST_FACES);
          if(wall_dir/2 != dir) // if(wall_dir/2 == dir), it means it's the face itself (or you are using a grid that is not tolerated, so you're fucked) --> it cannot be across...
            my_cell.has_neighbor_across = sgn_face != sgn_of_wall_neighbor_of_face(face_idx, dir, wall_dir);
        }
      }
    }

    my_cell.is_set = !voronoi_on_the_fly; // we NEVER raise the flag if doing it on the fly (safety measure /!\)

    return my_cell;
  }

#ifndef P4_TO_P8
  inline void clip_voronoi_cell_by_parallel_walls(Voronoi2D &voro_cell, const u_char &dir)
  {
    const vector<ngbd2Dseed> *points;
    vector<Point2> *partition;

    voro_cell.get_neighbor_seeds(points);
    voro_cell.get_partition(partition);

    /* clip the voronoi partition at the boundary of the domain */
    const u_char other_dir = (dir + 1)%P4EST_DIM;
    for(size_t m = 0; m < points->size(); m++)
      if((*points)[m].n < 0 &&  (- 1 - (*points)[m].n)/2 == dir) // there is a "wall" point added because of a parallel wall
      {
        const char which_wall = (- 1 - (*points)[m].n)%2;
        for(char i = -1; i < 1; ++i)
        {
          size_t k                        = mod(m + i                 , partition->size());
          size_t tmp                      = mod(k + (i == -1 ? -1 : 1), partition->size());
          Point2 segment                  = (*partition)[tmp] - (*partition)[k];
          double lambda                   = ((which_wall == 0 ? xyz_min[dir] : xyz_max[dir]) - (*partition)[k].xyz(dir))/segment.xyz(dir);
          (*partition)[k].xyz(dir)        = (which_wall == 0 ? xyz_min[dir] : xyz_max[dir]);
          (*partition)[k].xyz(other_dir)  = (*partition)[k].xyz(other_dir) + lambda*segment.xyz(other_dir);
        }
      }
  }
#endif


//  void compute_local_jump_mu_grad_v_elements(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *fine_qnnn,
//                                             const my_p4est_interpolation_nodes_t &interp_grad_underlined_vn_nodes,
//                                             const double* fine_normal_p, const double *fine_mass_flux_p, const double *fine_jump_u_p,
//                                             const double *fine_variable_surface_tension_p, const double *fine_curvature_p,
//                                             double* fine_jump_mu_grad_v_p) const;

  inline double jump_mass_density() const { return (rho_plus - rho_minus); }
  inline double jump_inverse_mass_density() const { return (1.0/rho_plus - 1.0/rho_minus); }
  inline double jump_viscosity() const { return (mu_plus - mu_minus); }

  void interpolate_velocities_at_node(const p4est_locidx_t &node_idx, double *vnp1_nodes_minus_p, double *vnp1_nodes_plus_p,
                                      const double *vnp1_face_minus_p[P4EST_DIM], const double *vnp1_face_plus_p[P4EST_DIM]);

  void TVD_extrapolation_of_np1_node_velocities(const u_int& niterations = 20, const u_char& order = 2);

  void compute_backtraced_velocities();

  void advect_interface(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1,
                        const p4est_nodes_t *known_nodes, Vec known_phi_np1 = NULL);
  void sample_static_levelset_on_nodes(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1);
  void compute_vorticities();

  void set_interface_velocity();

  inline void create_vnp1_face_vectors_if_needed()
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(vnp1_face_minus[dim] == NULL){
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_minus[dim], dim); CHKERRXX(ierr); }
      if(vnp1_face_plus[dim] == NULL){
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_plus[dim], dim); CHKERRXX(ierr); }
    }
    return;
  }

public:
  my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1_, my_p4est_node_neighbors_t *ngbd_n_, my_p4est_faces_t *faces_n_,
                             my_p4est_node_neighbors_t *fine_ngbd_n = NULL);
  ~my_p4est_two_phase_flows_t();

  inline void compute_dt(const double &min_value_for_u_max = 1.0)
  {
    dt_nm1 = dt_n;
    const double max_L2_norm_u_overall = MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus);
    dt_n = MIN(1/min_value_for_u_max, 1/max_L2_norm_u_overall) * cfl_advection * MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]));
    dt_n = MIN(dt_n, cfl_surface_tension*sqrt((rho_minus + rho_plus)*pow(MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])), 3)/(M_PI*surface_tension)));

    dt_updated = true;
  }

  inline void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p)
  {
    bc_velocity = bc_v;
    bc_pressure = bc_p;
  }

  inline void set_external_forces_per_unit_mass(CF_DIM *external_forces_per_unit_mass_[P4EST_DIM])
  {
    for(u_char dir = 0; dir < P4EST_DIM; ++dir)
      this->force_per_unit_mass[dir] = external_forces_per_unit_mass_[dir];
  }

  inline void set_dynamic_viscosities(const double& mu_m_, const double& mu_p_)
  {
    mu_minus  = mu_m_;
    mu_plus   = mu_p_;
  }

  inline void set_surface_tension(const double& surface_tension_)
  {
    surface_tension = surface_tension_;
  }

  inline void set_densities(const double& rho_m_, const double& rho_p_)
  {
    rho_minus = rho_m_;
    rho_plus  = rho_p_;
  }

  void set_phi(Vec phi_on_interface_capturing_nodes, const interpolation_method& method = linear, Vec phi_on_computational_nodes_ = NULL);
  void set_node_velocities(CF_DIM* vnm1_minus_functor[P4EST_DIM], CF_DIM* vn_minus_functor[P4EST_DIM],
                           CF_DIM* vnm1_plus_functor[P4EST_DIM],  CF_DIM* vn_plus_functor[P4EST_DIM]);
  void set_face_velocities_np1(CF_DIM* vnp1_m_[P4EST_DIM], CF_DIM* vnp1_p_[P4EST_DIM]);
//  void set_jump_mu_grad_v(CF_DIM* jump_mu_grad_v_op[P4EST_DIM][P4EST_DIM]);

  void compute_second_derivatives_of_n_velocities();
  void compute_second_derivatives_of_nm1_velocities();

  inline void set_semi_lagrangian_order(const int& sl_)
  {
    sl_order = sl_;
  }

  inline void set_uniform_bands(const double& uniform_band_m_, const double&uniform_band_p_)
  {
    uniform_band_minus  = uniform_band_m_;
    uniform_band_plus   = uniform_band_p_;
  }

  inline void set_uniform_band(const double&  uniform_band_) { set_uniform_bands(uniform_band_, uniform_band_);}

  inline void set_vorticity_split_threshold(double thresh_)
  {
    threshold_split_cell = thresh_;
  }

  inline void set_cfls(const double& cfl_adv, const double& cfl_surf_tens)
  {
    cfl_advection = cfl_adv;
    cfl_surface_tension = cfl_surf_tens;
  }
  inline void set_cfl(const double& cfl) { set_cfls(cfl, cfl); }

  inline void set_dt(double dt_nm1_, double dt_n_)
  {
    dt_nm1  = dt_nm1_;
    dt_n    = dt_n_;
  }

  inline void set_dt(double dt_n_) {dt_n = dt_n_; }

  inline double get_dt() const                                              { return dt_n; }
  inline double get_dtnm1() const                                           { return dt_nm1; }
  inline p4est_t* get_p4est_n() const                                       { return p4est_n; }
  inline p4est_nodes_t* get_nodes_n() const                                 { return nodes_n; }
  inline my_p4est_faces_t* get_faces_n() const                              { return faces_n ; }
  inline p4est_ghost_t* get_ghost_n() const                                 { return ghost_n; }
  inline my_p4est_node_neighbors_t* get_ngbd_n() const                      { return ngbd_n; }

  inline const my_p4est_interface_manager_t* get_interface_manager() const  { return interface_manager; }
  inline Vec get_vnp1_nodes_minus() const                                   { return vnp1_nodes_minus; }
  inline Vec get_vnp1_nodes_plus() const                                    { return vnp1_nodes_plus; }
  inline double get_diag_min() const                                        { return tree_diagonal/((double) (1 << (interface_manager->get_max_level_computational_grid()))); }

  inline void do_voronoi_computations_on_the_fly(const bool& do_it_on_the_fly)
  {
    voronoi_on_the_fly = do_it_on_the_fly;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      voro_cell[dir].clear();
      if(voronoi_on_the_fly)
        voro_cell[dir].resize(1);
      else
        voro_cell[dir].resize(faces_n->num_local[dir]);
    }
  }

  void solve_viscosity();
  void solve_viscosity_explicit();

  void compute_pressure_jump();
  void solve_for_pressure_guess(const KSPType ksp = KSPBCGS, const PCType pc = PCHYPRE);
  void solve_projection(const KSPType ksp = KSPBCGS, const PCType pc = PCHYPRE);

  inline void set_projection_solver(const poisson_jump_cell_solver_tag& solver_to_use) { cell_jump_solver_to_use = solver_to_use; }
  inline void fetch_interface_points_with_second_order_accuracy() {
    fetch_interface_FD_neighbors_with_second_order_accuracy = true;
    interface_manager->evaluate_FD_theta_with_quadratics(fetch_interface_FD_neighbors_with_second_order_accuracy);
  }

  void compute_velocities_at_nodes();
  void save_vtk(const std::string& vtk_directory, const int& index) const;
  void update_from_tn_to_tnp1(const bool& reinitialize_levelset = true, const bool& static_interface = false);

  inline double get_max_velocity() const        { return MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus); }
  inline double get_max_velocity_minus() const  { return max_L2_norm_velocity_minus; }
  inline double get_max_velocity_plus() const   { return max_L2_norm_velocity_plus; }

  inline double volume_in_negative_domain() const { return interface_manager->volume_in_negative_domain(); }

  inline int get_rank() const { return p4est_n->mpirank; }


};

#endif // MY_P4EST_TWO_PHASE_FLOWS_H
