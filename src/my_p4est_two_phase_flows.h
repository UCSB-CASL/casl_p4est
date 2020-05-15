#ifndef MY_P4EST_TWO_PHASE_FLOWS_H
#define MY_P4EST_TWO_PHASE_FLOWS_H

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_xgfm_cells.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_xgfm_cells.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/voronoi2D.h>
#endif

#if __cplusplus >= 201103L
#include <unordered_map> // if c++11 is fully supported, use unordered maps (i.e. hash tables) as they are apparently much faster
#else
#include <map>
#endif

#if __cplusplus >= 201103L
typedef std::unordered_map<p4est_locidx_t, p4est_locidx_t> computational_to_fine_node_t;
#else
typedef std::map<p4est_locidx_t, p4est_locidx_t> computational_to_fine_node_t;
#endif

using std::set;

typedef enum {
  PSEUDO_TIME = 426624,
  EXPLICIT_ITERATIVE
} extrapolation_technique;

typedef enum {
  OMEGA_MINUS = 2789,
  OMEGA_PLUS
} domain_side;

typedef  enum {
  velocity_field = 153759,
  hodge_field
} two_sided_field;

typedef struct
{
  double value;
  double distance;
} neighbor_value;

typedef struct
{
  double derivative;
  double theta;
  bool xgfm;
} sharp_derivative;

typedef struct
{
  double theta;
  double jump_field;
  double jump_flux_component;
  p4est_locidx_t fine_intermediary_idx;
} interface_data;

static const double value_not_needed = NAN;

class my_p4est_two_phase_flows_t
{
private:

  struct augmented_voronoi_cell
  {
    Voronoi_DIM voro;
    p4est_locidx_t fine_idx_of_face;
    voro_cell_type cell_type;
    bool is_set, is_in_negative_domain, has_neighbor_across;
    augmented_voronoi_cell() : fine_idx_of_face(-1), is_set(false), has_neighbor_across(false) {}
  };

  class splitting_criteria_computational_grid_two_phase_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est_np1, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes_np1,
                      const double *phi_np1_p, const double *vorticities_np1_p);
    const my_p4est_two_phase_flows_t *owner;
  public:
    splitting_criteria_computational_grid_two_phase_t(my_p4est_two_phase_flows_t* parent_solver) :
      splitting_criteria_tag_t ((splitting_criteria_t*)(parent_solver->p4est_n->user_pointer)), owner(parent_solver) {}
    bool refine_and_coarsen(p4est_t* p4est_np1, p4est_nodes_t* nodes_np1, Vec phi_coarse_np1, Vec vorticities);
  };


  class wall_bc_value_hodge_t : public CF_DIM {
  private:
    my_p4est_two_phase_flows_t* _parent;
  public:
    wall_bc_value_hodge_t(my_p4est_two_phase_flows_t* obj) : _parent(obj) {}
    inline double operator()(DIM(double x, double y, double z)) const { return _parent->bc_pressure->wallValue(DIM(x,y,z)) * _parent->dt_n / (_parent->BDF_alpha()); }
    double operator()(const double *xyz) const {return this->operator()(DIM(xyz[0], xyz[1], xyz[2]));}
  };

  my_p4est_brick_t *brick;
  p4est_connectivity_t *conn;

  p4est_t *p4est_nm1;
  p4est_ghost_t *ghost_nm1;
  p4est_nodes_t *nodes_nm1;
  my_p4est_hierarchy_t *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  p4est_t *p4est_n, *fine_p4est_n;
  p4est_ghost_t *ghost_n, *fine_ghost_n;
  p4est_nodes_t *nodes_n, *fine_nodes_n;
  my_p4est_hierarchy_t *hierarchy_n, *fine_hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n, *fine_ngbd_n;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_faces_t *faces_n;

  double dxyz_min[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double convert_to_xyz[P4EST_DIM];
  double tree_diag;
  bool periodic[P4EST_DIM];

  double surface_tension;
  double mu_p, mu_m;
  double rho_p, rho_m;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_u[2]; // 0::minus, 1::plus
  double uniform_band_m, uniform_band_p;
  double threshold_split_cell;
  double cfl;
  bool   dt_updated;

  int sl_order;

  double threshold_dbl_max;
  const double threshold_norm_of_n = 1.0e-6;

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM bc_hodge;
  BoundaryConditionsDIM *bc_v;

  wall_bc_value_hodge_t wall_bc_value_hodge;

  CF_DIM *external_forces[P4EST_DIM];
  my_p4est_interpolation_nodes_t *interp_phi;

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  Vec fine_phi, fine_curvature, fine_jump_hodge, fine_jump_normal_flux_hodge, fine_mass_flux, fine_variable_surface_tension;
  // vector fields, P4EST_DIM-block-structured
  Vec fine_normal, fine_jump_u, fine_phi_xxyyzz;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // fine_jump_mu_grad_v_p[SQR_P4EST_DIM*i+P4EST_DIM*dir+der] is the jump in mu \dfrac{\partial u_{dir}}{\partial x_{der}}, evaluated at local node i of fine_p4est_n
  Vec fine_jump_mu_grad_v;
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  Vec vorticities;
  // vector fields, P4EST_DIM-block-structured
  Vec vnp1_nodes_m,  vnp1_nodes_p;
  Vec vn_nodes_m,    vn_nodes_p;
  Vec interface_velocity_np1; // yes, np1, yes! (used right after compute_dt in update_from_n_to_np1, so it looks like n but it's actually np1)
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vn_nodes_m_xxyyzz_p[SQR_P4EST_DIM*i+P4EST_DIM*dir+der] is the second derivative of u^{n, -}_{dir} with respect to cartesian direction x_{der}, evaluated at local node i of p4est_n
  Vec vn_nodes_m_xxyyzz, vn_nodes_p_xxyyzz, interface_velocity_np1_xxyyzz;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT CELL CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // scalar fields
  Vec hodge;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  Vec dxyz_hodge[P4EST_DIM], vstar[P4EST_DIM], vnp1_m[P4EST_DIM], vnp1_p[P4EST_DIM];
  Vec diffusion_vnm1_faces[P4EST_DIM], diffusion_vn_faces[P4EST_DIM];
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields, P4EST_DIM-block-structured
  Vec vnm1_nodes_m,  vnm1_nodes_p;
  Vec interface_velocity_n; // yes, n, yes! (used right after compute_dt in update_from_n_to_np1, so it looks like nm1 but it's actually n)
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vnm1_nodes_m_xxyyzz_p[SQR_P4EST_DIM*i+P4EST_DIM*dir+der] is the second derivative of u^{n-1, -}_{dir} with respect to cartesian direction x_{der}, evaluated at local node i of p4est_nm1
  Vec vnm1_nodes_m_xxyyzz, vnm1_nodes_p_xxyyzz, interface_velocity_n_xxyyzz;

  // semi-lagrangian backtraced points for faces (needed in viscosity step's setup, needs to be done only once)
  // no need to destroy these, not dynamically allocated...
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> backtraced_vn_faces[P4EST_DIM];
  std::vector<double> backtraced_vnm1_faces[P4EST_DIM];
  // The value of the dir velocity component at the points at time n and nm1 backtraced from the face of orientation dir and local index f_idx are
  // backtraced_vn_faces[dir][f_idx], and
  // backtraced_vnm1_faces[dir][f_idx].

  computational_to_fine_node_t face_to_fine_node[P4EST_DIM], cell_to_fine_node, node_to_fine_node;
  bool face_to_fine_node_maps_are_set[P4EST_DIM], cell_to_fine_node_map_is_set, node_to_fine_node_map_is_set;

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
    bool run_for_diffusion;
    bool matrix_is_preallocated[P4EST_DIM];
    Mat matrix[P4EST_DIM];
    // if we do have a null space, it will be the constant nullspace
    // (unless some other border is added) -> no need to build it ourselves
    KSP ksp[P4EST_DIM];
    Vec rhs[P4EST_DIM];
    int matrix_has_nullspace[P4EST_DIM];
    bool matrix_is_ready[P4EST_DIM], only_diags_are_modified[P4EST_DIM];
    double current_diag_m[P4EST_DIM], current_diag_p[P4EST_DIM];
    double desired_diag_m[P4EST_DIM], desired_diag_p[P4EST_DIM];
    bool ksp_is_set_from_options[P4EST_DIM], pc_is_set_from_options[P4EST_DIM];
    my_p4est_two_phase_flows_t* env;

    inline void reset_current_diagonals(const unsigned char &dir)
    {
      current_diag_m[dir] = current_diag_p[dir] = 0.0;
    }

    inline bool current_diags_are_as_desired(const unsigned char &dir) const
    {
      return (fabs(current_diag_m[dir] - desired_diag_m[dir]) < EPS*MAX(fabs(current_diag_m[dir]), fabs(desired_diag_m[dir])) || (fabs(current_diag_m[dir]) < EPS && fabs(desired_diag_m[dir]) < EPS))
          && (fabs(current_diag_p[dir] - desired_diag_p[dir]) < EPS*MAX(fabs(current_diag_p[dir]), fabs(desired_diag_p[dir])) || (fabs(current_diag_p[dir]) < EPS && fabs(desired_diag_p[dir]) < EPS));
    }

    inline void destroy_and_nullify_owned_members()
    {
      PetscErrorCode ierr;
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        if(matrix[dim] != NULL) { ierr = MatDestroy(matrix[dim]); CHKERRXX(ierr); matrix[dim] = NULL; }
        if(ksp[dim] != NULL)    { ierr = KSPDestroy(ksp[dim]);    CHKERRXX(ierr); ksp[dim]    = NULL; }
        if(rhs[dim] != NULL)    { ierr = VecDestroy(rhs[dim]);    CHKERRXX(ierr); rhs[dim]    = NULL; }
      }
    }

    void preallocate_matrix(const unsigned char &dir);
    void setup_linear_system(const unsigned char &dir);
    void setup_linear_solver(const unsigned char &dir, const PetscBool &use_nonzero_initial_guess, const KSPType &ksp_type, const PCType &pc_type);

  public:
    jump_face_solver(my_p4est_two_phase_flows_t *parent_solver = NULL) : run_for_diffusion(false), env(parent_solver)
    {
      /* initialize the KSP solvers and other parameters */
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        matrix_is_preallocated[dim]   = false;
        matrix[dim]                   = NULL;
        ksp[dim]                      = NULL;
        rhs[dim]                      = NULL;
        matrix_has_nullspace[dim] = matrix_is_ready[dim] = only_diags_are_modified[dim]  = false;
        current_diag_m[dim] = current_diag_p[dim] = 0.0;
        desired_diag_m[dim] = desired_diag_p[dim] = 0.0;
        ksp_is_set_from_options[dim] = pc_is_set_from_options[dim] = false;
      }
    }
    ~jump_face_solver() { destroy_and_nullify_owned_members(); }

    inline void set_environment(my_p4est_two_phase_flows_t* env_)
    {
      env = env_;
      PetscErrorCode ierr;
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        if(ksp[dir] == NULL)
        {
          ierr = KSPCreate(env->p4est_n->mpicomm, &ksp[dir]); CHKERRXX(ierr);
          ierr = KSPSetTolerances(ksp[dir], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
        }
      }
    }


    inline void set_diagonals(const double &add_m, const double &add_p)
    {
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        desired_diag_m[dim]           = add_m;
        desired_diag_p[dim]           = add_p;
        if(!current_diags_are_as_desired(dim))
        {
          // actual modification of diag, do not change the flag values otherwise, especially not of matrix_is_ready
          only_diags_are_modified[dim]  = matrix_is_ready[dim];
          matrix_is_ready[dim]          = false;
        }
      }
    }

    inline void set_for_diffusion() { run_for_diffusion = true; }

    void solve(Vec solution[P4EST_DIM], const PetscBool &use_nonzero_initial_guess = PETSC_FALSE, const KSPType &ksp_type = KSPCG, const PCType &pc_type = PCHYPRE);

    void reset()
    {
      destroy_and_nullify_owned_members();
      PetscErrorCode ierr;
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        matrix_is_preallocated[dim]   = false;
        matrix_has_nullspace[dim] = matrix_is_ready[dim] = only_diags_are_modified[dim]  = false;
        current_diag_m[dim] = current_diag_p[dim] = 0.0;
        desired_diag_m[dim] = desired_diag_p[dim] = 0.0;
        ksp_is_set_from_options[dim] = pc_is_set_from_options[dim] = false;
        if(ksp[dim] == NULL)
        {
          ierr = KSPCreate(env->p4est_n->mpicomm, &ksp[dim]); CHKERRXX(ierr);
          ierr = KSPSetTolerances(ksp[dim], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
        }
      }
    }
  } viscosity_solver;

  //  inline bool get_close_coarse_node(const p4est_indep_t* fine_node, p4est_locidx_t& coarse_node_idx, bool subrefined [P4EST_DIM]) const
  //  {
  //    int lmax = ((const splitting_criteria_t*)p4est_n->user_pointer)->max_lvl;
  //    bool is_coarse = true;
  //    subrefined[0] = ((fine_node->x%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[0];
  //    subrefined[1] = ((fine_node->y%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[1];
  //#ifdef P4_TO_P8
  //    subrefined[2] = ((fine_node->z%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[2];
  //#endif

  //    p4est_quadrant_t r;
  //    r.level = P4EST_MAXLEVEL;
  //    r.x = fine_node->x - (subrefined[0] ? P4EST_QUADRANT_LEN(lmax+1) : 0);
  //    r.y = fine_node->y - (subrefined[1] ? P4EST_QUADRANT_LEN(lmax+1) : 0);
  //#ifdef P4_TO_P8
  //    r.z = fine_node->z - (subrefined[2] ? P4EST_QUADRANT_LEN(lmax+1) : 0);
  //#endif
  //    r.p.which_tree = fine_node->p.which_tree;
  //    // theoretically no need to canonicalize here, the quad center will always be INSIDE a tree
  //    // --> check for it in debug!
  //    P4EST_ASSERT((r.x != P4EST_ROOT_LEN) && (r.y != P4EST_ROOT_LEN));
  //#ifdef P4_TO_P8
  //    P4EST_ASSERT(r.z != P4EST_ROOT_LEN);
  //#endif
  //    P4EST_ASSERT (p4est_quadrant_is_node (&r, 1));
  //    bool tmp = index_of_node(&r, nodes_n, coarse_node_idx);
  //    if(!tmp)
  //      throw std::runtime_error("my_p4est_two_phase_flows::get_close_coarse_node() could not find close coarse node.");

  //    return is_coarse;
  //  }

  //#ifdef P4_TO_P8
  //  inline p4est_locidx_t neighbor_coarse_node(const p4est_indep_t* coarse_node, const unsigned char& ii, const unsigned char& jj, const unsigned char& kk) const
  //#else
  //  inline p4est_locidx_t neighbor_coarse_node(const p4est_indep_t* coarse_node, const unsigned char& ii, const unsigned char& jj) const
  //#endif
  //  {
  //    P4EST_ASSERT(((ii == 0) || (ii == 1)) && ((jj == 0) || (jj == 1)));
  //#ifdef P4_TO_P8
  //    P4EST_ASSERT((kk == 0) || (kk == 1));
  //    P4EST_ASSERT((ii != 0) || (jj != 0) || (kk != 0));
  //#else
  //    P4EST_ASSERT((ii != 0) || (jj != 0));
  //#endif
  //    int lmax = ((const splitting_criteria_t*)p4est_n->user_pointer)->max_lvl;

  //    p4est_quadrant_t *tmp;
  //    p4est_quadrant_t r, n;
  //    r.level = P4EST_MAXLEVEL;
  //    r.x = coarse_node->x + ii*P4EST_QUADRANT_LEN(lmax);
  //    r.y = coarse_node->y + jj*P4EST_QUADRANT_LEN(lmax);
  //#ifdef P4_TO_P8
  //    r.z = coarse_node->z + kk*P4EST_QUADRANT_LEN(lmax);
  //#endif
  //    r.p.which_tree = coarse_node->p.which_tree;
  //#ifdef P4_TO_P8
  //    if(r.x==P4EST_ROOT_LEN || r.y==P4EST_ROOT_LEN || r.z==P4EST_ROOT_LEN)
  //#else
  //    if(r.x==P4EST_ROOT_LEN || r.y==P4EST_ROOT_LEN)
  //#endif
  //    {
  //      p4est_node_canonicalize(p4est_n, r.p.which_tree, &r, &n);
  //      tmp = &n;
  //    }
  //    else
  //      tmp = &r;
  //    P4EST_ASSERT (p4est_quadrant_is_node (tmp, 1));
  //    p4est_locidx_t return_idx;
  //    bool check = index_of_node(tmp, nodes_n, return_idx);
  //    if(!check)
  //      throw std::runtime_error("my_p4est_two_phase_flows::neighbor_coarse_node() could not find neighbor coarse node.");
  //    return return_idx;
  //  }

  inline bool is_subresolved(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, p4est_locidx_t& fine_center_idx, const p4est_quadrant_t* coarse_quad = NULL)
  {
    computational_to_fine_node_t::const_iterator got_it = cell_to_fine_node.find(quad_idx);
    if(got_it != cell_to_fine_node.end()) // found in map
    {
      fine_center_idx = got_it->second;
      return true;
    }
    else
    {
      if(cell_to_fine_node_map_is_set)
        return false;
      if(coarse_quad == NULL)
      {
        p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
        coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
      }
      if(coarse_quad->level < (((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl))
        return false;
      p4est_quadrant_t r;
      r.level = P4EST_MAXLEVEL;
      r.x = coarse_quad->x + P4EST_QUADRANT_LEN(coarse_quad->level+1);
      r.y = coarse_quad->y + P4EST_QUADRANT_LEN(coarse_quad->level+1);
      ONLY3D(r.z = coarse_quad->z + P4EST_QUADRANT_LEN(coarse_quad->level+1));
      P4EST_ASSERT (p4est_quadrant_is_node (&r, 1));
      // theoretically no need to canonicalize here, the quad center will always be INSIDE a tree
      // --> check for it in debug!
      P4EST_ASSERT((r.x != 0) && (r.x != P4EST_ROOT_LEN) && (r.y != 0) && (r.y != P4EST_ROOT_LEN));
      ONLY3D(P4EST_ASSERT((r.z != 0) && (r.z != P4EST_ROOT_LEN)));
      r.p.which_tree = tree_idx;
      bool to_return = index_of_node(&r, fine_nodes_n, fine_center_idx);
      if(to_return)
        cell_to_fine_node[quad_idx] = fine_center_idx;
      return to_return;
    }
  };

  inline bool get_fine_node_idx_of_logical_vertex(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, DIM(const char& vx, const char& vy, const char& vz), p4est_locidx_t& fine_vertex_idx,  const p4est_quadrant_t* coarse_quad)
  {
    const unsigned char sum_v = SUMD(abs(vx), abs(vy), abs(vz));
    P4EST_ASSERT(vx == -1 || vx == 0 || vx == 1);
    P4EST_ASSERT(vy == -1 || vy == 0 || vy == 1);
    ONLY3D(P4EST_ASSERT(vz == -1 || vz == 0 || vz == 1));
    P4EST_ASSERT(sum_v <= P4EST_DIM);
    // looking for saved shortcuts first
    const bool is_center  = (sum_v == 0);
    const bool is_face    = (sum_v == 1);
#ifdef P4_TO_P8
    const bool is_edge    = (sum_v == 2);
#endif
    const bool is_corner  = (sum_v == P4EST_DIM);
    p4est_locidx_t face_idx, node_idx;
    char local_face_dir = -1;
    if(is_center)
    {
      computational_to_fine_node_t::const_iterator got_it = cell_to_fine_node.find(quad_idx);
      if(got_it != cell_to_fine_node.end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (cell_to_fine_node_map_is_set)
        return false;
    }
    if(is_face)
    {
#ifdef P4_TO_P8
      local_face_dir = (abs(vx) == 1 ? 1 + vx : (abs(vy) == 1 ? 5 + vy : 9 + vz))/2;
#else
      local_face_dir = (abs(vx) == 1 ? 1 + vx : 5 + vy)/2;
#endif
      P4EST_ASSERT(local_face_dir >= 0 && local_face_dir < P4EST_FACES);
      face_idx = faces_n->q2f(quad_idx, local_face_dir);
      computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[local_face_dir/2].find(face_idx);
      if(got_it != face_to_fine_node[local_face_dir/2].end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (face_to_fine_node_maps_are_set[local_face_dir/2])
        return false;
    }
#ifdef P4_TO_P8
    // TO BE IMPLEMENTED
    if(is_edge)
    {
      std::cerr << "you have more work to do here!" << std::endl;
//      throw std::invalid_argument("you have a bit more work in 3D, here...");
    }
#endif
    if (is_corner)
    {
      char local_node_idx = SUMD((vx+1)/2, (vy+1), 2*(vz+1));
      P4EST_ASSERT((local_node_idx >= 0) && (local_node_idx < P4EST_CHILDREN));
      node_idx = nodes_n->local_nodes[P4EST_CHILDREN*quad_idx+local_node_idx];
      computational_to_fine_node_t::const_iterator got_it = node_to_fine_node.find(node_idx);
      if(got_it != node_to_fine_node.end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (node_to_fine_node_map_is_set)
        return false;
    }
    // core of the routine here below
    if(coarse_quad == NULL)
    {
      if(quad_idx < p4est_n->local_num_quadrants)
      {
        p4est_tree_t* tree  = p4est_tree_array_index(p4est_n->trees, tree_idx);
        coarse_quad         = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
      }
      else
        coarse_quad         = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
    }
    p4est_quadrant_t *tmp_ptr;
    p4est_quadrant_t r;
    r.level = P4EST_MAXLEVEL;
    XCODE(r.x = coarse_quad->x + (vx+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    YCODE(r.y = coarse_quad->y + (vy+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    ZCODE(r.z = coarse_quad->z + (vz+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    r.p.which_tree = tree_idx;
    P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
    if(ORD((r.x == 0) || (r.x == P4EST_ROOT_LEN), (r.y == 0) || (r.y == P4EST_ROOT_LEN), (r.z == 0) || (r.z == P4EST_ROOT_LEN)))
    {
      p4est_quadrant_t n;
      p4est_node_canonicalize(fine_p4est_n, tree_idx, &r, &n);
      tmp_ptr = &n;
    }
    else
      tmp_ptr = &r;
    bool to_return = index_of_node(tmp_ptr, fine_nodes_n, fine_vertex_idx);
    if(to_return && is_center)
      cell_to_fine_node[quad_idx] = fine_vertex_idx;
    if(to_return && is_face)
    {
      P4EST_ASSERT(local_face_dir >= 0 && local_face_dir < P4EST_FACES);
      face_to_fine_node[local_face_dir/2][face_idx] = fine_vertex_idx;
    }
#ifdef P4_TO_P8
    // TO BE IMPLEMENTED
//    if(to_return && is_edge)
//    {
//      throw std::invalid_argument("you have a bit more work in 3D, here...");
//    }
#endif
    if(to_return && is_corner)
      node_to_fine_node[node_idx] = fine_vertex_idx;

    return to_return;
  }

  inline bool get_fine_node_idx_node(const p4est_locidx_t& node_idx, p4est_locidx_t& fine_vertex_idx)
  {
    P4EST_ASSERT(node_idx >= 0 && node_idx < nodes_n->num_owned_indeps);
    computational_to_fine_node_t::const_iterator got_it = node_to_fine_node.find(node_idx);
    if(got_it != node_to_fine_node.end())
    {
      fine_vertex_idx = got_it->second;
      return true;
    }
    else if (node_to_fine_node_map_is_set)
      return false;
    // core of the routine here below
    p4est_quadrant_t r;
    r = *((p4est_quadrant_t*) sc_array_index(&nodes_n->indep_nodes, node_idx));
    P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
    // theoretically no need to canonicalize here, the point should already be INSIDE
    // a tree
    P4EST_ASSERT(ANDD(r.x != P4EST_ROOT_LEN, r.y != P4EST_ROOT_LEN, r.z != P4EST_ROOT_LEN));
    bool to_return = index_of_node(&r, fine_nodes_n, fine_vertex_idx);
    if(to_return)
      node_to_fine_node[node_idx] = fine_vertex_idx;
    return to_return;
  }

  inline bool get_fine_node_idx_of_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& loc_face_dir, p4est_locidx_t& fine_face_idx, const p4est_quadrant_t* coarse_quad = NULL, const p4est_locidx_t* face_idx_ptr = NULL)
  {
    computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[loc_face_dir/2].find(face_idx_ptr != NULL ? *face_idx_ptr : faces_n->q2f(quad_idx, loc_face_dir));
    if(got_it != face_to_fine_node[loc_face_dir/2].end()) // found in map
    {
      fine_face_idx = got_it->second;
      return true;
    }
    else
    {
      if(face_to_fine_node_maps_are_set[loc_face_dir/2])
        return false;
      if(coarse_quad == NULL)
      {
        if(quad_idx < p4est_n->local_num_quadrants)
        {
          p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
          coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
        }
        else
          coarse_quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
      }
      if(coarse_quad->level < (((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl))
        return false;
      char search[P4EST_DIM] = {DIM(0, 0, 0)}; search[loc_face_dir/2] = (loc_face_dir%2 == 1 ? 1 : -1);
      return get_fine_node_idx_of_logical_vertex(quad_idx, tree_idx, DIM(search[0], search[1], search[2]), fine_face_idx, coarse_quad);
    }
  }
  inline bool signs_of_phi_are_different(const double& phi_0, const double& phi_1) const
  {
    return ((phi_0 > 0.0 && phi_1 <= 0.0) || (phi_0 <= 0.0 && phi_1 > 0.0));
  }
  inline bool face_is_across(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& dir, const double* fine_phi_p, p4est_locidx_t& fine_center_idx, p4est_locidx_t& fine_face_idx, const p4est_quadrant_t* coarse_quad = NULL)
  {
    if(!is_subresolved(quad_idx, tree_idx, fine_center_idx, coarse_quad))
      return false;
    get_fine_node_idx_of_face(quad_idx, tree_idx, dir, fine_face_idx, coarse_quad);
    return signs_of_phi_are_different(fine_phi_p[fine_center_idx], fine_phi_p[fine_face_idx]);
  }
  inline bool face_is_dirichlet_wall(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* xyz_face) const
  {
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
    unsigned char loc_face_idx = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
    P4EST_ASSERT(faces_n->q2f(quad_idx, loc_face_idx) == face_idx);
    const p4est_quadrant_t* quad;
    if(quad_idx < p4est_n->local_num_quadrants)
    {
      p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
      quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
    }
    else
      quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
    return (is_quad_Wall(p4est_n, tree_idx, quad, loc_face_idx) && bc_v[dir].wallType(xyz_face) == DIRICHLET);
  }
  inline bool is_wall_neighbor_of_face_in_negative_domain(p4est_locidx_t& fine_wall_node_idx, const p4est_locidx_t& face_idx, const p4est_locidx_t &quad_idx, const p4est_topidx_t &tree_idx,
                                                          const unsigned char &face_touch, const unsigned char &der, const p4est_quadrant_t *quad, const double* fine_phi_p, const double *xyz_wall = NULL)
  {
    P4EST_ASSERT(faces_n->q2f(quad_idx, face_touch) == face_idx);
    P4EST_ASSERT(is_quad_Wall(p4est_n, tree_idx, quad, der));
    char search[P4EST_DIM]  = {DIM(0, 0, 0)};
    search[face_touch/2]    = (face_touch%2 == 1 ? 1 : -1);
    search[der/2]           = (der%2 == 1 ? 1 : -1);
    if(get_fine_node_idx_of_logical_vertex(quad_idx, tree_idx, DIM(search[0], search[1],search[2]), fine_wall_node_idx, quad))
    {
#ifdef P4EST_DEBUG
      const p4est_indep_t *found_node = (const p4est_indep_t *) sc_array_index(&fine_nodes_n->indep_nodes, fine_wall_node_idx);
      P4EST_ASSERT(is_node_Wall(fine_p4est_n, found_node, der));
#endif
      return (fine_phi_p[fine_wall_node_idx] <= 0.0);
    }
    else
    {
      if(xyz_wall != NULL)
        return ((*interp_phi)(xyz_wall) <= 0.0);
      double xyz_w[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, face_touch/2, xyz_w);
      xyz_w[der/2] = (der%2 == 1 ? xyz_max[der/2] : xyz_min[der/2]);
      return ((*interp_phi)(xyz_w) <= 0.0);
    }
  }
  inline bool is_face_in_negative_domain(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* fine_phi_p, p4est_locidx_t& fine_face_idx, const double *xyz_face = NULL)
  {
    computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[dir].find(face_idx);
    if(got_it != face_to_fine_node[dir].end()) // found in map
    {
      fine_face_idx = got_it->second;
      return (fine_phi_p[fine_face_idx] <= 0.0);
    }
    else
    {
      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;
      faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
      unsigned char loc_face_dir = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
      P4EST_ASSERT(face_idx >= 0);
      P4EST_ASSERT(faces_n->q2f(quad_idx, loc_face_dir) == face_idx);
      const p4est_quadrant_t* coarse_quad;
      if(quad_idx<p4est_n->local_num_quadrants)
      {
        p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
        coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
      }
      else
        coarse_quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
      if (!face_to_fine_node_maps_are_set[dir] && coarse_quad->level == ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl && get_fine_node_idx_of_face(quad_idx, tree_idx, loc_face_dir, fine_face_idx, coarse_quad, &face_idx))
        return (fine_phi_p[fine_face_idx] <= 0.0);
      else
      {
        if(xyz_face != NULL)
          return ((*interp_phi)(xyz_face) <= 0.0);
        double xyz_f[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_f);
        return ((*interp_phi)(xyz_f) <= 0.0);
      }
    }
  }
  inline bool is_face_in_negative_domain(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* fine_phi_p)
  {
    p4est_locidx_t dummy;
    return is_face_in_negative_domain(face_idx, dir, fine_phi_p, dummy);
  }

  inline double BDF_alpha() const { return (sl_order == 1 ? 1.0 : (2.0*dt_n + dt_nm1)/(dt_n + dt_nm1)); }
  inline double BDF_beta() const  { return (sl_order == 1 ? 0.0 : -dt_n/(dt_n + dt_nm1));               }

  void get_velocity_seen_from_cell(neighbor_value& neighbor_velocity, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const int& face_dir,
                                   const double *vstar_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const double *vstar_p[], const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p);

  double div_mu_grad_u_dir(bool &face_is_in_negative_domain, const p4est_locidx_t &face_idx, const unsigned char &dir,
                           const double *vn_dir_p, const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                           const double *fine_phi_p, const double *fine_phi_xxyyzz_p);


  inline const augmented_voronoi_cell& get_augmented_voronoi_cell(const p4est_locidx_t &face_idx, const unsigned char &dir, const double *fine_phi_p)
  {
    augmented_voronoi_cell &my_cell = (voronoi_on_the_fly ? voro_cell[dir][0] : voro_cell[dir][face_idx]);
    if(!voronoi_on_the_fly && my_cell.is_set)
      return my_cell;

    P4EST_ASSERT(!my_cell.is_set);

    my_cell.cell_type = compute_voronoi_cell(my_cell.voro, faces_n, face_idx, dir, bc_v, NULL);
    P4EST_ASSERT(my_cell.cell_type != not_well_defined); // prohibited in two-phase flows...

    my_cell.has_neighbor_across = false;
    my_cell.is_in_negative_domain = is_face_in_negative_domain(face_idx, dir, fine_phi_p, my_cell.fine_idx_of_face);

#ifndef P4_TO_P8
    if(my_cell.cell_type != dirichlet_wall_face && my_cell.cell_type != parallelepiped_no_wall)
      clip_voronoi_cell_by_parallel_walls(my_cell.voro, dir);
#endif

    if(my_cell.cell_type != dirichlet_wall_face && my_cell.cell_type != not_well_defined)
    {
      const vector<ngbdDIMseed> *points;
      my_cell.voro.get_neighbor_seeds(points);
      for (size_t n = 0; n < points->size() && !my_cell.has_neighbor_across; ++n)
      {
        if((*points)[n].n >= 0)
          my_cell.has_neighbor_across = (my_cell.is_in_negative_domain != is_face_in_negative_domain((*points)[n].n, dir, fine_phi_p));
        else // neighbor is a wall, but it could very well be across the interface too...
        {
          char wall_dir = -1 - (*points)[n].n;
          P4EST_ASSERT(0 <= wall_dir && wall_dir < P4EST_FACES);
          if(wall_dir/2 != dir) // if(wall_dir/2 == dir), it means it's the face itself (or you are using a grid that is not tolerated, so you're fucked) --> it cannot be across...
          {
            // it's a tranverse wall so
            p4est_locidx_t quad_idx, dummy;
            p4est_topidx_t tree_idx;
            faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
            const unsigned char face_touch = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
            const p4est_quadrant_t *quad;
            if(quad_idx < p4est_n->local_num_quadrants)
            {
              p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
              quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
            }
            else
              quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx - p4est_n->local_num_quadrants);
            my_cell.has_neighbor_across = (my_cell.is_in_negative_domain != is_wall_neighbor_of_face_in_negative_domain(dummy, face_idx, quad_idx, tree_idx, face_touch, wall_dir, quad, fine_phi_p));
          }
        }
      }
    }

    my_cell.is_set = !voronoi_on_the_fly; // we NEVER raise the flag if doing it on the fly (safety measure /!\)

    return my_cell;
  }

#ifndef P4_TO_P8
  inline void clip_voronoi_cell_by_parallel_walls(Voronoi2D &voro_cell, const unsigned char &dir)
  {
    const vector<ngbd2Dseed> *points;
    vector<Point2> *partition;

    voro_cell.get_neighbor_seeds(points);
    voro_cell.get_partition(partition);

    /* clip the voronoi partition at the boundary of the domain */
    const unsigned char other_dir = (dir + 1)%P4EST_DIM;
    for(size_t m = 0; m < points->size(); m++)
      if((*points)[m].n < 0 &&  (-1 - (*points)[m].n)/2 == dir) // there is a "wall" point added because of a parallel wall
      {
        const unsigned char which_wall = (-1 - (*points)[m].n)%2;
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

  void compute_normals_curvature_and_second_derivatives(const bool& set_second_derivatives);
  void compute_curvature();
  void normalize_normals();

  PetscErrorCode inline delete_and_nullify_vector(Vec& vv)
  {
    PetscErrorCode ierr = 0;
    if(vv != NULL){
      ierr = VecDestroy(vv); CHKERRQ(ierr);
      vv = NULL;
    }
    return ierr;
  }

  PetscErrorCode inline create_node_vector_if_needed(Vec& vv, const p4est_t* forest, const p4est_nodes_t* nodes, const unsigned int &block_size = 1)
  {
    PetscErrorCode ierr = 0;
    // destroy and nullify the vector if it is not correctly set
    if(vv != NULL && !VecIsSetForNodes(vv, nodes, forest->mpicomm, block_size)){
      ierr = VecDestroy(vv); CHKERRQ(ierr);
      vv = NULL;
    }
    if(vv == NULL){
      ierr = VecCreateGhostNodesBlock(forest, nodes, block_size, &vv); CHKERRQ(ierr);
    }
    return ierr;
  }

  void inline compute_gradient_and_second_derivatives(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *qnnn,
                                                      const double* fine_phi_p, double *fine_grad_phi_p, double *fine_phi_xxyyzz_p)
  {
    qnnn->gradient(fine_phi_p, (fine_grad_phi_p+P4EST_DIM*fine_node_idx));
    if(fine_phi_xxyyzz_p != NULL)
      qnnn->laplace(fine_phi_p, (fine_phi_xxyyzz_p+P4EST_DIM*fine_node_idx));
  }

  void inline compute_local_curvature(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *qnnn,
                                      const double* fine_phi_p, const double *fine_phi_xxyyzz_p,
                                      const double *fine_grad_phi_p, double *fine_curvature_p)
  {
//    fine_curvature_p[fine_node_idx] = 1.0/0.5;
    // compute first derivatives
    double norm_of_grad = 0.0;
    double dx = fine_grad_phi_p[P4EST_DIM*fine_node_idx+0]; norm_of_grad += SQR(dx);
    double dy = fine_grad_phi_p[P4EST_DIM*fine_node_idx+1]; norm_of_grad += SQR(dy);
#ifdef P4_TO_P8
    double dz = fine_grad_phi_p[P4EST_DIM*fine_node_idx+2]; norm_of_grad += SQR(dz);
#endif
    norm_of_grad = sqrt(norm_of_grad);

    if(norm_of_grad > threshold_norm_of_n)
    {
      // compute second derivatives
      double dxxyyzz[P4EST_DIM];
      if(fine_phi_xxyyzz_p != NULL){
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          dxxyyzz[der] = fine_phi_xxyyzz_p[P4EST_DIM*fine_node_idx+der];
      } else
        qnnn->laplace(fine_phi_p, dxxyyzz);


      double dxy = qnnn->dy_central_component(fine_grad_phi_p, P4EST_DIM, dir::x);
#ifdef P4_TO_P8
      double dxz = qnnn->dz_central_component(fine_grad_phi_p, P4EST_DIM, dir::x); // d/dz{d/dx}
      double dyz = qnnn->dz_central_component(fine_grad_phi_p, P4EST_DIM, dir::y); // d/dz{d/dy}
#endif
#ifdef P4_TO_P8
      fine_curvature_p[fine_node_idx] = ((dxxyyzz[1]+dxxyyzz[2])*SQR(dx) + (dxxyyzz[0]+dxxyyzz[2])*SQR(dy) + (dxxyyzz[0]+dxxyyzz[1])*SQR(dz) -
          2*(dx*dy*dxy + dx*dz*dxz + dy*dz*dyz)) / (norm_of_grad*norm_of_grad*norm_of_grad);
#else
      fine_curvature_p[fine_node_idx] = (dxxyyzz[1]*SQR(dx) + dxxyyzz[0]*SQR(dy) - 2*dx*dy*dxy) / (norm_of_grad*norm_of_grad*norm_of_grad);
#endif
    }
    else
      fine_curvature_p[fine_node_idx] = 0.0; // nothing better to suggest for now, sorry
  }

  void inline compute_local_jump_mu_grad_v_elements(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *fine_qnnn,
                                                    const my_p4est_interpolation_nodes_t &interp_grad_underlined_vn_nodes,
                                                    const double* fine_normal_p, const double *fine_mass_flux_p, const double *fine_jump_u_p,
                                                    const double *fine_variable_surface_tension_p, const double *fine_curvature_p,
                                                    double* fine_jump_mu_grad_v_p) const
  {
    const double overlined_mu = overlined_viscosity();
    double grad_underlined_u[SQR_P4EST_DIM];  // grad_underlined_u[P4EST_DIM*i+der] = partical derivative of component i of underlined u along direction der
    double grad_jump_u[SQR_P4EST_DIM];        // grad_jump_u[P4EST_DIM*i+der] = partical derivative of component i of grad_jump_u along direction der
    double grad_mass_flux[P4EST_DIM];
    double grad_surface_tension[P4EST_DIM];
    double xyz_fine_node[P4EST_DIM]; node_xyz_fr_n(fine_node_idx, fine_p4est_n, fine_nodes_n, xyz_fine_node);
    interp_grad_underlined_vn_nodes(xyz_fine_node, grad_underlined_u);
    if(fine_mass_flux_p != NULL){
      fine_qnnn->gradient(fine_mass_flux_p, grad_mass_flux);
      P4EST_ASSERT(fine_jump_u_p != NULL);
      fine_qnnn->gradient_all_components(fine_jump_u_p, grad_jump_u, P4EST_DIM);
    }
    if(fine_variable_surface_tension_p != NULL)
      fine_qnnn->gradient(fine_variable_surface_tension_p, grad_surface_tension);

    // jump in div(u) is implicitly assumed to be 0.0! (only assumption)
    p4est_locidx_t dim_dim_fine_node_idx  = SQR_P4EST_DIM*fine_node_idx;
    p4est_locidx_t dim_fine_node_idx      = P4EST_DIM*fine_node_idx;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      const unsigned char dim_dir = P4EST_DIM*dir;
      for (unsigned char der = 0; der < P4EST_DIM; ++der) {
        fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] = 0.0;
        if(fine_mass_flux_p != NULL)
          fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] -=
              overlined_mu*fine_normal_p[dim_fine_node_idx+dir]*fine_normal_p[dim_fine_node_idx + der]*fine_curvature_p[fine_node_idx]*fine_mass_flux_p[fine_node_idx]*jump_inverse_mass_density();
        for (unsigned char k = 0; k < P4EST_DIM; ++k) {
          if(fine_mass_flux_p != NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] +=
                overlined_mu*((k == der ? 1.0 : 0.0) - fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k])*grad_jump_u[dim_dir+k];
          fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] +=
              jump_viscosity()*((k == der ? 1.0 : 0.0) - fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k])*grad_underlined_u[dim_dir+k];
          if(fine_variable_surface_tension_p != NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] +=
                ((k == dir ? 1.0 : 0.0) - fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k])*grad_surface_tension[k]*fine_normal_p[dim_fine_node_idx + der];
          if(fine_mass_flux_p != NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] -=
                overlined_mu*fine_normal_p[dim_fine_node_idx + der]*((dir == k ? 1.0 : 0.0) - fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k])*grad_mass_flux[k]*jump_inverse_mass_density();
          for (unsigned char r = 0; r < P4EST_DIM; ++r)
          {
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] -=
                jump_viscosity()*fine_normal_p[dim_fine_node_idx + der]*((dir == k ? 1.0 : 0.0) - fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k])*grad_underlined_u[P4EST_DIM*r + k]*fine_normal_p[dim_fine_node_idx+r];
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx + dim_dir + der] +=
                jump_viscosity()*fine_normal_p[dim_fine_node_idx+dir]*fine_normal_p[dim_fine_node_idx + der]*fine_normal_p[dim_fine_node_idx + k]*fine_normal_p[dim_fine_node_idx+r]*grad_underlined_u[P4EST_DIM*k + r];
          }
        }
      }
    }
  }

  inline double jump_mass_density() const { return (rho_p - rho_m);}
  inline double jump_inverse_mass_density() const { return (1.0/rho_p - 1.0/rho_m);}
  inline double jump_viscosity() const { return (mu_p - mu_m);}
  inline domain_side underlined_side(two_sided_field field_) const {
    return (field_ == velocity_field ? (mu_p >= mu_m ? OMEGA_PLUS : OMEGA_MINUS) : (rho_p >= rho_m ? OMEGA_MINUS : OMEGA_PLUS));
  }
  inline domain_side overlined_side(two_sided_field field_) const {
    return (underlined_side(field_) == OMEGA_PLUS ? OMEGA_MINUS : OMEGA_PLUS);
  }
  inline double overlined_viscosity() const { return (overlined_side(velocity_field) == OMEGA_PLUS ? mu_p : mu_m); }

  void interpolate_velocity_at_node(const p4est_locidx_t &node_idx, double *v_nodes_p_p, double *v_nodes_m_p,
                                    const double *vnp1_p_p[P4EST_DIM], const double *vnp1_m_p[P4EST_DIM]);


  interface_data interface_data_between_faces(const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                              const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                              const p4est_locidx_t &fine_idx_of_face, const p4est_locidx_t &fine_idx_of_other_face,
                                              const p4est_quadrant_t &qm, const p4est_quadrant_t &qp,
                                              const unsigned char &dir, const unsigned char der);

  interface_data interface_data_between_face_and_tranverse_wall(const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                                const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                                                const p4est_locidx_t &fine_idx_of_face, const p4est_locidx_t &fine_idx_of_wall_node,
                                                                const unsigned char &dir, const unsigned char der);

  /*
   * qm and qp must be defined and have their p.piggy3 filled!
   */
  sharp_derivative sharp_derivative_of_face_field(const p4est_locidx_t &face_idx, const bool &face_is_in_negative_domain, const p4est_locidx_t &fine_idx_of_face, const uniform_face_ngbd *face_neighbors,
                                                  const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                  const unsigned char &der, const unsigned char &dir,
                                                  const double *vn_dir_m_p, const double *vn_dir_p_p, const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                                  const p4est_quadrant_t &qm, const p4est_quadrant_t &qp);
  /*
   * qm and qp must be defined and have their p.piggy3 filled!
   */
  void get_velocity_from_other_domain_seen_from_face(neighbor_value &neighbor_velocity, const p4est_locidx_t &face_idx, const p4est_locidx_t &neighbor_face_idx,
                                                     const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                     const double *vn_dir_m_p, const double *vn_dir_p_p, const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                                     const unsigned char &der, const unsigned char &dir);

  void initialize_face_extrapolation(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                     const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const unsigned char &extrapolation_degree,
                                     const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                     double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM], double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM]);

  void extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                                double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM]);

  void solve_velocity_extrapolation_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                             const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const unsigned char &extrapolation_degree,
                                                             const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                                             const double *normal_derivative_of_vnp1_m_p[P4EST_DIM], const double *normal_derivative_of_vnp1_p_p[P4EST_DIM],
                                                             double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM]);

  void extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                         double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM]);

  void solve_velocity_extrapolation_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                      const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const unsigned char &extrapolation_degree,
                                                      const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
                                                      const double *normal_derivative_of_vnp1_m_p[P4EST_DIM], const double *normal_derivative_of_vnp1_p_p[P4EST_DIM],
                                                      double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM]);

public:
  void interpolate_linearly_from_fine_nodes_to_coarse_nodes(const Vec& vv_fine, Vec& vv_coarse);
private:

  void trajectory_from_all_faces_two_phases(my_p4est_faces_t *faces_n, my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n,
                                            const double *fine_phi_p,
                                            Vec vnm1_nodes_m, Vec vnm1_nodes_m_xxyyzz,
                                            Vec vnm1_nodes_p, Vec vnm1_nodes_p_xxyyzz,
                                            Vec vn_nodes_m, Vec vn_nodes_m_xxyyzz,
                                            Vec vn_nodes_p, Vec vn_nodes_p_xxyyzz,
                                            double dt_nm1, double dt_n,
                                            std::vector<double> xyz_n[P4EST_DIM],
                                            std::vector<double> xyz_nm1[P4EST_DIM]);

  void get_interface_velocity(Vec interface_velocity);
  void advect_interface(p4est_t *fine_p4est_np1, p4est_nodes_t *fine_nodes_np1, Vec fine_phi_np1,
                        p4est_nodes_t *known_nodes, Vec known_phi_np1 = NULL);
  void compute_vorticities();

  void set_interface_velocity();

  void extend_vector_field_from_interface_to_whole_domain();

  void do_semi_lagrangian_backtracing_from_faces_if_needed();

public:
  my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, my_p4est_node_neighbors_t *fine_ngbd_n);
  ~my_p4est_two_phase_flows_t();

  inline void compute_dt(const double &min_value_for_u_max=1.0)
  {
    dt_nm1 = dt_n;
    double max_L2_norm_u_overall = MAX(max_L2_norm_u[0], max_L2_norm_u[1]);
    dt_n = MIN(1/min_value_for_u_max, 1/max_L2_norm_u_overall) * cfl * MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));
    dt_n = MIN(dt_n, sqrt((rho_m + rho_p)*pow(MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2])), 3)/(4.0*M_PI*surface_tension)));
//    dt_n = MIN(dt_n, 1.0/(MAX(mu_m/rho_m, mu_p/rho_p)*2.0*(SUMD(1.0/SQR(dxyz_min[0]), 1.0/SQR(dxyz_min[1]), 1.0/SQR(dxyz_min[2])))));

    dt_updated = true;
  }

  inline void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p)
  {
    this->bc_v          = bc_v;
    this->bc_pressure   = bc_p;
    bc_hodge.setWallTypes(bc_pressure->getWallType());
    bc_hodge.setWallValues(wall_bc_value_hodge);
  }

  inline void set_external_forces(CF_DIM *external_forces_[P4EST_DIM])
  {
    for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
      this->external_forces[dir] = external_forces_[dir];
  }

  inline void set_dynamic_viscosities(double mu_m_, double mu_p_)
  {
    mu_m  = mu_m_;
    mu_p  = mu_p_;
  }

  inline void set_surface_tension(double surface_tension_)
  {
    surface_tension = surface_tension_;
  }

  inline void set_densities(double rho_m_, double rho_p_)
  {
    rho_m = rho_m_;
    rho_p = rho_p_;
  }

  void set_phi(Vec fine_phi_, bool set_second_derivatives = false);
  void set_node_velocities(CF_DIM* vnm1_m_[P4EST_DIM], CF_DIM* vn_m_[P4EST_DIM], CF_DIM* vnm1_p_[P4EST_DIM], CF_DIM* vn_p_[P4EST_DIM]);
  void set_face_velocities_np1(CF_DIM* vnp1_m_[P4EST_DIM], CF_DIM* vnp1_p_[P4EST_DIM]);
  void set_jump_mu_grad_v(CF_DIM* jump_mu_grad_v_op[P4EST_DIM][P4EST_DIM]);
//  void set_node_vorticities(CF_DIM* vorticity_m, CF_DIM* vorticity_p);
  void compute_second_derivatives_of_n_velocities();
  void compute_second_derivatives_of_nm1_velocities();
  inline void compute_second_derivatives_of_n_and_nm1_velocities()
  {
    compute_second_derivatives_of_nm1_velocities();
    compute_second_derivatives_of_n_velocities();
  }

  inline void set_semi_lagrangian_order(int sl_)
  {
    sl_order = sl_;
  }

  inline void set_uniform_bands(double uniform_band_m_, double uniform_band_p_)
  {
    uniform_band_m  = uniform_band_m_;
    uniform_band_p  = uniform_band_p_;
  }

  void set_uniform_band(double uniform_band_) {set_uniform_bands(uniform_band_, uniform_band_);}

  inline void set_vorticity_split_threshold(double thresh_)
  {
    threshold_split_cell = thresh_;
  }

  inline void set_cfl(double cfl_)
  {
    cfl = cfl_;
  }

  inline void set_dt(double dt_nm1_, double dt_n_)
  {
    dt_nm1  = dt_nm1_;
    dt_n    = dt_n_;
  }

  inline void set_dt(double dt_n_) {dt_n = dt_n_; }

  inline double get_dt()                                        { return dt_n; }
  inline double get_dtnm1()                                     { return dt_nm1; }
  inline p4est_t* get_p4est()                                   { return p4est_n; }
  inline p4est_nodes_t* get_nodes()                             { return nodes_n; }
  inline my_p4est_faces_t* get_faces()                          { return faces_n ; }

  inline p4est_nodes_t* get_fine_nodes()                        { return fine_nodes_n; }
  inline p4est_ghost_t* get_ghost()                             { return ghost_n; }
  inline Vec get_hodge()                                        { return hodge; }
  inline Vec get_normals()                                      { return fine_normal; }
  inline p4est_t* get_p4est_n() const                           { return p4est_n; }
  inline Vec get_vnp1_nodes_m() const                           { return vnp1_nodes_m; }
  inline Vec get_vnp1_nodes_p() const                           { return vnp1_nodes_p; }
  inline my_p4est_node_neighbors_t* get_ngbd_n() const          { return ngbd_n; }
  inline my_p4est_interpolation_nodes_t* get_interp_phi() const { return interp_phi; }
  inline p4est_nodes_t* get_nodes_n() const                     { return nodes_n; }
  inline p4est_ghost_t* get_ghost_n() const                     { return ghost_n; }
  inline double get_diag_min() const                            { return tree_diag/((double) (1<<(((splitting_criteria_t*)p4est_n->user_pointer)->max_lvl))); }
  inline Vec get_curvature()                                    { return fine_curvature; }
//  inline Vec get_phi()                                          { return fine_phi; }

  inline void do_voronoi_computations_on_the_fly(bool do_it_on_the_fly)
  {
    voronoi_on_the_fly = do_it_on_the_fly;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      voro_cell[dir].clear();
      if(voronoi_on_the_fly)
        voro_cell[dir].resize(1);
      else
        voro_cell[dir].resize(faces_n->num_local[dir]);
    }
  }

  void compute_viscosity_jumps();
  void compute_jumps_hodge();
  void solve_viscosity();
  void solve_diffusion_viscosity();
  void solve_viscosity_explicit();
  void solve_diffusion_viscosity_explicit();

  inline void initialize_diffusion_problem_vectors()
  {
    PetscErrorCode ierr;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      if(diffusion_vnm1_faces[dir] == NULL){
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &diffusion_vnm1_faces[dir], dir);  CHKERRXX(ierr);
      }
      if(diffusion_vn_faces[dir] == NULL){
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &diffusion_vn_faces[dir], dir);    CHKERRXX(ierr);
      }
      if(vstar[dir] == NULL){
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir], dir);                 CHKERRXX(ierr);
      }
    }
  }

  void solve_projection(const bool activate_xgfm)
  {
    my_p4est_xgfm_cells_t* cell_poisson_jump_solver = NULL;
    solve_projection(cell_poisson_jump_solver, activate_xgfm);
    delete cell_poisson_jump_solver;
  }
  void solve_projection(my_p4est_xgfm_cells_t* &cell_poisson_jump_solver, const bool activate_xgfm, const KSPType ksp = KSPCG, const PCType pc = PCSOR /*PCHYPRE*/);

  void extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(const extrapolation_technique& extrapolation_method = PSEUDO_TIME, const unsigned int& n_iteration = 10, const unsigned char& extrapolation_degree = 1);
  void compute_velocity_at_nodes();
  void save_vtk(const char* name, const bool& export_fine_grid = false, const char* name_fine = NULL);
  void update_from_tn_to_tnp1(/*const unsigned int &nnn*/);

  inline double get_max_velocity() const { return MAX(max_L2_norm_u[0], max_L2_norm_u[1]); }
  inline double get_max_velocity_m() const { return max_L2_norm_u[0]; }
  inline double get_max_velocity_p() const { return max_L2_norm_u[1]; }

  inline Vec* get_diffusion_vnm1_faces()                        { return diffusion_vnm1_faces;                        }
  inline Vec* get_diffusion_vn_faces()                          { return diffusion_vn_faces;                          }
  inline Vec* get_diffusion_vnp1_faces()                        { return vstar;                                       }
  inline void set_fine_jump_mu_grad_v(Vec fine_jump_mu_grad_v_) { fine_jump_mu_grad_v = fine_jump_mu_grad_v_; return; }
  inline Vec  get_fine_jump_mu_grad_v()                         { return fine_jump_mu_grad_v;                         }
  inline void set_fine_jump_velocity(Vec fine_jump_u_)          { fine_jump_u = fine_jump_u_; return;                 }
  inline Vec  get_fine_jump_velocity()                          { return fine_jump_u;                                 }
  inline void slide_face_fields(const bool projection_was_done = false)
  {
    PetscErrorCode ierr;
    Vec tmp;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      tmp                       = diffusion_vnm1_faces[dir];
      diffusion_vnm1_faces[dir] = diffusion_vn_faces[dir];
      if(!projection_was_done)
      {
        diffusion_vn_faces[dir] = vstar[dir];
        vstar[dir]  = tmp;
      }
      else
      {
        double *diffusion_vn_faces_dir_p;
        const double *fine_phi_p, *vnp1_m_dir_p, *vnp1_p_dir_p;
        ierr = VecCreateGhostFaces(p4est_n, faces_n, &diffusion_vn_faces[dir], dir); CHKERRXX(ierr);
        ierr = VecGetArray(diffusion_vn_faces[dir], &diffusion_vn_faces_dir_p); CHKERRXX(ierr);
        ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
        ierr = VecGetArrayRead(vnp1_m[dir], &vnp1_m_dir_p); CHKERRXX(ierr);
        ierr = VecGetArrayRead(vnp1_p[dir], &vnp1_p_dir_p); CHKERRXX(ierr);
        for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
          p4est_locidx_t f_idx = faces_n->get_layer_face(dir, k);
          if(is_face_in_negative_domain(f_idx, dir, fine_phi_p))
            diffusion_vn_faces_dir_p[f_idx] = vnp1_m_dir_p[f_idx];
          else
            diffusion_vn_faces_dir_p[f_idx] = vnp1_p_dir_p[f_idx];
        }
        ierr = VecGhostUpdateBegin(diffusion_vn_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
          p4est_locidx_t f_idx = faces_n->get_local_face(dir, k);
          if(is_face_in_negative_domain(f_idx, dir, fine_phi_p))
            diffusion_vn_faces_dir_p[f_idx] = vnp1_m_dir_p[f_idx];
          else
            diffusion_vn_faces_dir_p[f_idx] = vnp1_p_dir_p[f_idx];
        }
        ierr = VecGetArrayRead(vnp1_p[dir], &vnp1_p_dir_p); CHKERRXX(ierr);
        ierr = VecGetArrayRead(vnp1_m[dir], &vnp1_m_dir_p); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(fine_phi,&fine_phi_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(diffusion_vn_faces[dir], &diffusion_vn_faces_dir_p); CHKERRXX(ierr);
        ierr = VecDestroy(tmp); CHKERRXX(ierr);
      }
    }
    if(projection_was_done)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        ierr = VecGhostUpdateEnd(diffusion_vn_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    return;
  }

  inline void slide_node_velocities()
  {
    Vec tmp_m = vnm1_nodes_m;
    Vec tmp_p = vnm1_nodes_p;

    vnm1_nodes_m = vn_nodes_m;
    vnm1_nodes_p = vn_nodes_p;
    Vec tmp_m_xxyyzz = vnm1_nodes_m_xxyyzz; vnm1_nodes_m_xxyyzz = vn_nodes_m_xxyyzz; vn_nodes_m_xxyyzz = tmp_m_xxyyzz;
    Vec tmp_p_xxyyzz = vnm1_nodes_p_xxyyzz; vnm1_nodes_p_xxyyzz = vn_nodes_p_xxyyzz; vn_nodes_p_xxyyzz = tmp_p_xxyyzz;

    vn_nodes_m = vnp1_nodes_m; vn_nodes_p = vnp1_nodes_p;
    Vec inputs[2]   = {vn_nodes_m, vn_nodes_p}; Vec outputs[2]  = {vn_nodes_m_xxyyzz, vn_nodes_p_xxyyzz};
    ngbd_n->second_derivatives_central(inputs, outputs, 2, P4EST_DIM);

    vnp1_nodes_m = tmp_m;
    vnp1_nodes_p = tmp_p;

    return;
  }
  // to be called by the main after the time has been advanced!
  void enforce_dirichlet_bc_on_diffusion_vnp1_faces();

};

#endif // MY_P4EST_TWO_PHASE_FLOWS_H
