#ifndef MY_P4EST_TWO_PHASE_FLOWS_H
#define MY_P4EST_TWO_PHASE_FLOWS_H

#ifdef P4_TO_P8
#include <src/my_p8est_interface_manager.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_poisson_jump_cells_fv.h>
#include <src/my_p8est_poisson_jump_cells_xgfm.h>
#include <src/my_p8est_poisson_jump_faces_xgfm.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_save_load.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_interface_manager.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_jump_cells_fv.h>
#include <src/my_p4est_poisson_jump_cells_xgfm.h>
#include <src/my_p4est_poisson_jump_faces_xgfm.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_save_load.h>
#include <src/voronoi2D.h>
#endif

#if __cplusplus >= 201103L
#include <unordered_map> // if c++11 is fully supported, use unordered maps (i.e. hash tables) as they are apparently much faster
#else
#include <map>
#endif

using std::set;

class my_p4est_two_phase_flows_t
{
private:

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
  jump_solver_tag cell_jump_solver_to_use;
  bool fetch_interface_FD_neighbors_with_second_order_accuracy;

  const double *xyz_min, *xyz_max;
  double tree_dimension[P4EST_DIM];
  double dxyz_smallest_quad[P4EST_DIM];
  bool periodicity[P4EST_DIM];
  double tree_diagonal, smallest_diagonal;

  double surface_tension;
  double mu_minus, mu_plus;
  double rho_minus, rho_plus;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_velocity_minus, max_L2_norm_velocity_plus;
  double uniform_band_minus, uniform_band_plus;
  double threshold_split_cell;
  double cfl_advection, cfl_visco_capillary, cfl_capillary;
  bool   dt_updated;
  interpolation_method levelset_interpolation_method;

  int sl_order, sl_order_interface;

  const double threshold_dbl_max;

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM *bc_velocity;

  CF_DIM *force_per_unit_mass[P4EST_DIM];

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  Vec phi;
  Vec jump_normal_velocity; // jump_in_normal_velocity <-> mass_flux*(jump in inverse mass density)
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
  // the "np1" interface velocitiy is determined and used right after compute_dt in update_from_n_to_np1 but
  // _BEFORE_ final data update, so it is actually a function of np1 velocities _BEFORE_ those are eventually
  // slid in time. (yet used as "n" and "nm1" respectively when advecting the interface)
  Vec interface_velocity_np1;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vn_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_n
  Vec vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz, interface_velocity_np1_xxyyzz;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  Vec grad_p_guess_over_rho_minus[P4EST_DIM], grad_p_guess_over_rho_plus[P4EST_DIM];
  Vec vnp1_face_minus[P4EST_DIM], vnp1_face_plus[P4EST_DIM];
  Vec viscosity_rhs_minus[P4EST_DIM], viscosity_rhs_plus[P4EST_DIM];
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields, P4EST_DIM-block-structured
  Vec vnm1_nodes_minus,  vnm1_nodes_plus;
  Vec interface_velocity_n;
  Vec interface_velocity_n_xxyyzz;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vnm1_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n-1, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_nm1
  Vec vnm1_nodes_minus_xxyyzz, vnm1_nodes_plus_xxyyzz;

  // The value of the dir velocity component at the points at time n and nm1 backtraced from the face of orientation dir and local index f_idx are
  // backtraced_vn_faces[dir][f_idx] and backtraced_vnm1_faces[dir][f_idx].
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> backtraced_vn_faces_minus[P4EST_DIM], backtraced_vn_faces_plus[P4EST_DIM];
  std::vector<double> backtraced_vnm1_faces_minus[P4EST_DIM], backtraced_vnm1_faces_plus[P4EST_DIM]; // used only if sl_order == 2

  bool voronoi_on_the_fly;
  my_p4est_poisson_jump_faces_xgfm_t *viscosity_solver;

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

  inline double BDF_advection_alpha() const { return (sl_order == 1 ? 1.0 : (2.0*dt_n + dt_nm1)/(dt_n + dt_nm1)); }
  inline double BDF_advection_beta() const  { return (sl_order == 1 ? 0.0 : -dt_n/(dt_n + dt_nm1));               }

  inline double jump_mass_density() const { return (rho_plus - rho_minus); }
  inline double jump_inverse_mass_density() const { return (1.0/rho_plus - 1.0/rho_minus); }
  inline double jump_viscosity() const { return (mu_plus - mu_minus); }

  void interpolate_velocities_at_node(const p4est_locidx_t &node_idx, double *vnp1_nodes_minus_p, double *vnp1_nodes_plus_p,
                                      const double *vnp1_face_minus_p[P4EST_DIM], const double *vnp1_face_plus_p[P4EST_DIM]);

//  void TVD_extrapolation_of_np1_node_velocities(const u_int& niterations = 20, const u_char& order = 2);

  void compute_backtraced_velocities();
  void compute_viscosity_rhs();

  void advect_interface(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1,
                        const p4est_nodes_t *known_nodes, Vec known_phi_np1 = NULL);
  void sample_static_levelset_on_nodes(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1);
  void compute_vorticities();

  /*!
   * \brief save_or_load_parameters : save or loads the solver parameters in the two files of paths
   * given by sprintf(path_1, "%s_integers", filename) and sprintf(path_2, "%s_doubles", filename)
   * The integer parameters that are saved/loaded are (in this order):
   * - P4EST_DIM
   * - cell_jump_solver_to_use
   * - fetch_interface_FD_neighbors_with_second_order_accuracy
   * - data->min_lvl
   * - data->max_lvl
   * - fine_data->min_lvl (value exported is the same as data->min_lvl if not using subrefinement)
   * - fine_data->max_lvl (value exported is the same as data->max_lvl if not using subrefinement)
   * - levelset_interpolation_method
   * - sl_order
   * - sl_order_interface
   * - voronoi_on_the_fly
   * The double parameters/variables that are saved/loaded are (in this order):
   * - tree_dimension[0 : P4EST_DIM - 1]
   * - dxyz_smallest_quad[0 : P4EST_DIM - 1]
   * - surface_tension
   * - mu_minus
   * - mu_plus
   * - rho_minus
   * - rho_plus
   * - the simulation time tn
   * - dt_n
   * - dt_nm1
   * - max_L2_norm_velocity_minus
   * - max_L2_norm_velocity_plus
   * - uniform_band_minus
   * - uniform_band_plus
   * - threshold_split_cell
   * - cfl_advection
   * - cfl_surface_tension
   * - splitting_criterion->lip
   * - fine_splitting_criterion->lip (or a duplicate of the above value if not using subrefinement)
   * The integer and double parameters are saved separately in two different files to avoid reading errors due to
   * byte padding (occurs in order to ensure data alignment when written in file)...
   * \param filename [in] : basename of the path to the files to be written or read (absolute path)
   * \param splitting_criterion [inout]       : pointer to the splitting criterion to be exported/loaded (computational grid)
   * \param fine_splitting_criterion [inout]  : pointer to the splitting criterion to be exported/loaded (interface-capturing grid --> set to NULL if not using subrefinement)
   * \param flag[in]      : switch the behavior between write or read
   * \param tn[inout]     : in write mode, simulation time at which the function is called (to be saved, unmodified)
   *                        in read mode, simulation time at which the data were saved (to be read from file and stored in tn)
   * \param mpi[in]       : pointer to the mpi_environment_t (necessary for the load, disregarded for the save)
   * [note: implemented in one given function with switched behavior to avoid ambiguity and confusion due to code duplication
   * in several functions to be modified in the future if the parameter/variable order or the parameter/variable list is changed
   * (the save-state files are binary files, order and number of read/write operations is crucial)]
   * WARNING: this function throws an std::invalid_argument exception if the files can't be found when loading parameters
   * Raphael EGAN
   */
  void save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion,
                               save_or_load flag, double& tn, const mpi_environment_t* mpi = NULL);
  void fill_or_load_double_parameters(save_or_load flag, std::vector<PetscReal>& data, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion, double& tn);
  void fill_or_load_integer_parameters(save_or_load flag, std::vector<PetscInt>& data, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion);

public:
  my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1_, my_p4est_node_neighbors_t *ngbd_n_, my_p4est_faces_t *faces_n_,
                             my_p4est_node_neighbors_t *fine_ngbd_n = NULL);

  my_p4est_two_phase_flows_t(const mpi_environment_t& mpi, const char* path_to_saved_state, double& simulation_time);
  ~my_p4est_two_phase_flows_t();

  inline double get_capillary_dt() const
  {
    return sqrt(cfl_capillary*(rho_minus + rho_plus)*pow(MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])), 3)/(4.0*M_PI*surface_tension));
  }

  inline double get_visco_capillary_dt() const
  {
    return cfl_visco_capillary*MIN(mu_minus, mu_plus)*MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]))/surface_tension;
  }

  inline double get_advection_dt() const
  {
    return cfl_advection * MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]))/MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus);
  }
  inline void compute_dt()
  {
    dt_nm1 = dt_n;
    dt_n = MIN(get_advection_dt(), get_visco_capillary_dt() + sqrt(SQR(get_visco_capillary_dt()) + SQR(get_capillary_dt())));

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

  void compute_second_derivatives_of_n_velocities();
  void compute_second_derivatives_of_nm1_velocities();

  inline void set_semi_lagrangian_order_advection(const int& sl_)
  {
    sl_order = sl_;
  }
  inline void set_semi_lagrangian_order_interface(const int& sl_)
  {
    sl_order_interface = sl_;
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

  inline void set_cfls(const double& cfl_advection_, const double& cfl_visco_capillary_, const double& cfl_capillary_)
  {
    cfl_advection = cfl_advection_;
    cfl_visco_capillary = cfl_visco_capillary_;
    cfl_capillary = cfl_capillary_;
  }
  inline void set_cfl(const double& cfl) { set_cfls(cfl, cfl, cfl); } // all the same values: YOLO!

  inline void set_dt(double dt_nm1_, double dt_n_)
  {
    dt_nm1  = dt_nm1_;
    dt_n    = dt_n_;
  }

  inline bool viscosities_are_equal() const { return fabs(mu_minus - mu_plus) < EPS*MAX(fabs(mu_minus), fabs(mu_plus)); }
  inline bool mass_densities_are_equal() const { return fabs(rho_minus - rho_plus) < EPS*MAX(fabs(rho_minus), fabs(rho_plus)); }

  inline void set_dt(double dt_n_) {dt_n = dt_n_; }

  inline double get_dt() const                                              { return dt_n; }
  inline double get_dtnm1() const                                           { return dt_nm1; }
  inline p4est_t* get_p4est_n() const                                       { return p4est_n; }
  inline p4est_t* get_fine_p4est_n() const                                  { return fine_p4est_n; }
  inline p4est_nodes_t* get_nodes_n() const                                 { return nodes_n; }
  inline my_p4est_faces_t* get_faces_n() const                              { return faces_n ; }
  inline p4est_ghost_t* get_ghost_n() const                                 { return ghost_n; }
  inline my_p4est_node_neighbors_t* get_ngbd_n() const                      { return ngbd_n; }

  inline const my_p4est_interface_manager_t* get_interface_manager() const  { return interface_manager; }
  inline Vec get_vnp1_nodes_minus() const                                   { return vnp1_nodes_minus; }
  inline Vec get_vnp1_nodes_plus() const                                    { return vnp1_nodes_plus; }
  inline double get_diag_min() const                                        { return tree_diagonal/((double) (1 << (interface_manager->get_max_level_computational_grid()))); }
  inline double get_mu_minus() const                                        { return mu_minus; }
  inline double get_mu_plus() const                                         { return mu_plus; }
  inline double get_rho_minus() const                                       { return rho_minus; }
  inline double get_rho_plus() const                                        { return rho_plus; }
  inline double get_surface_tension() const                                 { return surface_tension; }

  void solve_viscosity();

  void compute_pressure_jump();
  void solve_for_pressure_guess(const KSPType ksp = KSPBCGS, const PCType pc = PCHYPRE);
  void solve_projection(const KSPType ksp = KSPBCGS, const PCType pc = PCHYPRE);

  inline void set_projection_solver(const jump_solver_tag& solver_to_use) { cell_jump_solver_to_use = solver_to_use; }
  inline void fetch_interface_points_with_second_order_accuracy() {
    fetch_interface_FD_neighbors_with_second_order_accuracy = true;
    interface_manager->evaluate_FD_theta_with_quadratics(fetch_interface_FD_neighbors_with_second_order_accuracy);
  }

  void compute_velocities_at_nodes();
  void set_interface_velocity_np1();
  void save_vtk(const std::string& vtk_directory, const int& index) const;
  void update_from_tn_to_tnp1(const bool& reinitialize_levelset = true, const bool& static_interface = false);

  inline double get_max_velocity() const        { return MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus); }
  inline double get_max_velocity_minus() const  { return max_L2_norm_velocity_minus; }
  inline double get_max_velocity_plus() const   { return max_L2_norm_velocity_plus; }

  inline double volume_in_negative_domain() const { return interface_manager->volume_in_negative_domain(); }

  inline int get_rank() const { return p4est_n->mpirank; }

  /*!
   * \brief save_state saves the solver states in a subdirectory 'backup_' created under the user-provided root-directory.
   * the n_states (> 0) latest succesive states can be saved, with automatic update of the subdirectory names.
   * If more than n_states subdirectories exist at any time when this function is called, it will automatically delete the
   * extra subdirectories.
   * \param path_to_root_directory: path to the root exportation directory. n_saved subdirectories 'backup_' will be created
   * under the root directory, in which successive solver states will be saved.
   * \param tn: simulation time at which the function is called
   * \param n_saved: number of solver states to keep in memory (default is 1)
   */
  void save_state(const char* path_to_root_directory, double& tn, const int& n_saved = 1);

  /*!
   * \brief load_state loads a solver state that has been previously saved on disk
   * \param mpi             [in]    mpi environment to load the solver state in
   * \param path_to_folder  [in]    path to the folder where the solver state has been stored (absolute path)
   * \param tn              [inout] simulation time at which the data were saved (to be read from saved solver state)
   * [NOTE :] the function will destroy and overwrite any grid-related structure like p4est_n, nodes_n, ghost_n, faces_n, etc.
   * if they have already been constructed beforehand...
   * WARNING: this function throws an std::invalid_argument exception if path_to_folder is invalid
   */
  void load_state(const mpi_environment_t& mpi, const char* path_to_folder, double& tn);


  inline my_p4est_brick_t* get_brick() const { return brick; }
  inline p4est_connectivity_t* get_connetivity() const { return conn; }


  inline void print_velocities_at_nodes() const
  {
    PetscErrorCode ierr;
    if(p4est_n->mpirank == 0)
      std::cout << "vn_nodes_minus = " << std::endl;
    ierr = VecView(vn_nodes_minus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);

    if(p4est_n->mpirank == 0)
      std::cout << "vn_nodes_plus = " << std::endl;
    ierr = VecView(vn_nodes_plus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);


    if(p4est_n->mpirank == 0)
      std::cout << "vnm1_nodes_minus = " << std::endl;
    ierr = VecView(vnm1_nodes_minus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);

    if(p4est_n->mpirank == 0)
      std::cout << "vnm1_nodes_plus = " << std::endl;
    ierr = VecView(vnm1_nodes_plus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
  }

  inline void print_phi() const
  {
    PetscErrorCode ierr;
    if(phi_on_computational_nodes != NULL && phi_on_computational_nodes != phi)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi_on_computational_nodes = " << std::endl;
      ierr = VecView(phi_on_computational_nodes, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
    if(phi != NULL)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi = " << std::endl;
      ierr = VecView(phi, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
    if(phi_xxyyzz != NULL)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi_xxyyzz = " << std::endl;
      ierr = VecView(phi_xxyyzz, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_pressure_guess() const
  {
    if(pressure_guess_solver != NULL && pressure_guess_solver->get_solution() != NULL)
    {
      PetscErrorCode ierr;
      if(p4est_n->mpirank == 0)
        std::cout << "pressure_guess = " << std::endl;
      ierr = VecView(pressure_guess_solver->get_solution(), PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_viscosity_rhs() const
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(p4est_n->mpirank == 0)
        std::cout << "viscosity rhs[" << int(dim) << "] = " << std::endl;
      ierr = VecView(viscosity_solver->rhs[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_vnp1_faces() const
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(p4est_n->mpirank == 0)
        std::cout << "vnp1_faces_minus[" << int(dim) << "] = " << std::endl;
      ierr = VecView(vnp1_face_minus[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
      if(p4est_n->mpirank == 0)
        std::cout << "vnp1_face_plus[" << int(dim) << "] = " << std::endl;
      ierr = VecView(vnp1_face_plus[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_sharp_viscous_solutions() const
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(p4est_n->mpirank == 0)
        std::cout << "sharp_viscous_solution[" << int(dim) << "] = " << std::endl;
      ierr = VecView(viscosity_solver->solution[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_vnp1_nodes() const
  {
    PetscErrorCode ierr;
    if(p4est_n->mpirank == 0)
      std::cout << "vnp1_nodes_minus = " << std::endl;
    ierr = VecView(vnp1_nodes_minus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    if(p4est_n->mpirank == 0)
      std::cout << "vnp1_nodes_plus = " << std::endl;
    ierr = VecView(vnp1_nodes_plus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
  }
  inline int get_sl_order()           const { return sl_order; }
  inline int get_sl_order_interface() const { return sl_order_interface; }
  inline double get_advection_cfl()   const { return cfl_advection; }

};

#endif // MY_P4EST_TWO_PHASE_FLOWS_H
