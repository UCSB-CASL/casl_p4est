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
                      const double *phi_np2_on_computational_nodes_p, bool &coarse_cell_crossed,
                      const double *vorticity_magnitude_np1_on_computational_nodes_minus_p,
                      const double *vorticity_magnitude_np1_on_computational_nodes_plus_p);
    const my_p4est_two_phase_flows_t *owner;
  public:
    splitting_criteria_computational_grid_two_phase_t(my_p4est_two_phase_flows_t* parent_solver) :
      splitting_criteria_tag_t((splitting_criteria_t*)(parent_solver->p4est_n->user_pointer)), owner(parent_solver) {}
    bool refine_and_coarsen(p4est_t* p4est_np1, const p4est_nodes_t* nodes_np1,
                            Vec phi_np2_on_computational_nodes, bool& coarse_cell_crossed,
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

  jump_solver_tag cell_jump_solver_to_use;
  KSPType Krylov_solver_for_cell_problems;
  PCType preconditioner_for_cell_problems;
  bool fetch_interface_FD_neighbors_with_second_order_accuracy; // relevant for FD xgfm approach(es)
  bool pressure_guess_is_set;
  my_p4est_poisson_jump_cells_t* cell_jump_solver;

  jump_solver_tag face_jump_solver_to_use;
  KSPType Krylov_solver_for_face_problems;
  PCType preconditioner_for_face_problems;
  bool voronoi_on_the_fly;
  my_p4est_poisson_jump_faces_t *face_jump_solver;

  void build_face_jump_solver();
  void build_cell_jump_solver();

  const double *xyz_min, *xyz_max;
  double tree_dimension[P4EST_DIM];
  double dxyz_smallest_quad[P4EST_DIM];
  bool periodicity[P4EST_DIM];
  double tree_diagonal, smallest_diagonal;

  double surface_tension;
  double mu_minus, mu_plus;
  double rho_minus, rho_plus;
  //
  // ---------|-------------|------------------|--------------|------------------>
  //          t_nm1         t_n                t_np1          t_np2
  //          <-------------><-----------------><------------->
  //               dt_nm1           dt_n             dt_np1
  double t_n;
  double dt_n;
  double dt_nm1;
  double dt_np1;
  double max_L2_norm_velocity_minus, max_L2_norm_velocity_plus;
  double uniform_band_minus, uniform_band_plus;
  double threshold_split_cell;
  double cfl_advection, cfl_visco_capillary, cfl_capillary;
  interpolation_method levelset_interpolation_method;

  int sl_order, sl_order_interface, degree_guess_v_star_face_k;
  int n_viscous_subiterations;
  bool static_interface;
  double max_velocity_component_before_projection[2]; // 0 <--> minus domain, 1 <--> plus domain
  double max_velocity_correction_in_projection[2];    // 0 <--> minus domain, 1 <--> plus domain
  double max_surface_tension_in_band_of_two_cells;
  double final_time;

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM *bc_velocity;
  FILE* log_file;
  int nsolve_calls;
  double tstart; // tn at the construction (0.0 or value as loaded from file)

  CF_DIM *force_per_unit_mass_minus[P4EST_DIM], *force_per_unit_mass_plus[P4EST_DIM];

  // --------------------------------------------------------------
  // -------------- FIELDS *NOT* OWNED BY THE SOLVER --------------
  // (must be provided by the user at every time step, if relevant)
  // --------------------------------------------------------------
  // - sampled at the nodes of the INTERFACE-CAPTURING grid:
  // -------------------------------------------------------
  Vec user_defined_nonconstant_surface_tension;   // in case of nonconstant surface tension, user-defined
  Vec user_defined_mass_flux;                     // mass flux across the interface, user-defined
  Vec user_defined_interface_force;               // vector field, P4EST_DIM-block-structured, interface-defined force term (ON TOP of surface tension effects!!!)

  // --------------------------------------------------------------
  // ----------------- FIELDS OWNED BY THE SOLVER -----------------
  // (the solver is responsible for related memory management and
  // updates, etc.
  // --------------------------------------------------------------
  // - sampled at the nodes of the INTERFACE-CAPTURING grid:
  // -------------------------------------------------------
  // scalar fields
  Vec phi_np1;
  Vec non_viscous_pressure_jump;    // pressure jump terms due to all but binormal viscous stress term
  Vec jump_normal_velocity;         // scalar fields, jump_in_normal_velocity == mass_flux*(jump in inverse mass density)
  // vector fields and/or other P4EST_DIM-block-structured
  Vec phi_np1_xxyyzz;
  Vec interface_tangential_force;   // vector field, P4EST_DIM-block-structured (tangential components of the interface-defined force and/or gradient of non-constant surface tension, i.e. Marangoni force)
  // -------------------------------------------------------
  // - sampled at the nodes of the COMPUTATIONAL grid n:
  // -------------------------------------------------------
  // scalar fields
  Vec phi_np1_on_computational_nodes;
  Vec vorticity_magnitude_minus, vorticity_magnitude_plus;
  // vector fields and/or other P4EST_DIM-block-structured
  Vec vnp1_nodes_minus,  vnp1_nodes_plus;
  Vec vn_nodes_minus,    vn_nodes_plus;
  Vec interface_velocity_np1;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vn_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_n
  Vec vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz, interface_velocity_np1_xxyyzz;
  // -------------------------------------------------------
  // - sampled at the cells of the COMPUTATIONAL grid n:
  // -------------------------------------------------------
  // scalar fields
  Vec pressure_minus, pressure_plus;
  // -------------------------------------------------------
  // - sampled at the faces of the COMPUTATIONAL grid n:
  // -------------------------------------------------------
  // vector fields
  Vec vnp1_face_star_minus_k[P4EST_DIM],    vnp1_face_star_plus_k[P4EST_DIM];     // face-sampled velocity, before projection, after the second-to-last viscosity step (or as used wihin the pressure guess jumps)
  Vec vnp1_face_star_minus_kp1[P4EST_DIM],  vnp1_face_star_plus_kp1[P4EST_DIM];   // face-sampled velocity, before projection, after the latest viscosity step
  Vec vnp1_face_minus[P4EST_DIM],           vnp1_face_plus[P4EST_DIM];            // divergence free face-sampled velocity, i.e. vnp1_face_*_kp1 made divergence-free
  Vec viscosity_rhs_minus[P4EST_DIM],       viscosity_rhs_plus[P4EST_DIM];
  // -------------------------------------------------------
  // - sampled at the nodes of the COMPUTATIONAL grid nm1:
  // -------------------------------------------------------
  // vector fields, P4EST_DIM-block-structured
  Vec vnm1_nodes_minus,  vnm1_nodes_plus;
  Vec interface_velocity_n;
  // tensor/matrix fields, (SQR_P4EST_DIM)-block-structured
  // vnm1_nodes_minus_xxyyzz_p[SQR_P4EST_DIM*i + P4EST_DIM*dir + der] is the second derivative of u^{n-1, -}_{dir} with respect to cartesian direction {der}, evaluated at local node i of p4est_nm1
  Vec interface_velocity_n_xxyyzz;
  Vec vnm1_nodes_minus_xxyyzz, vnm1_nodes_plus_xxyyzz;

  // The value of the dir velocity component at the points at time n and nm1 backtraced from the face of orientation dir and local index f_idx are
  // backtraced_vn_faces[dir][f_idx] and backtraced_vnm1_faces[dir][f_idx].
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> backtraced_vn_faces_minus[P4EST_DIM], backtraced_vn_faces_plus[P4EST_DIM];
  std::vector<double> backtraced_vnm1_faces_minus[P4EST_DIM], backtraced_vnm1_faces_plus[P4EST_DIM]; // used only if sl_order == 2

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

  void advect_interface(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np2,
                        const p4est_nodes_t *known_nodes_np2 = NULL, Vec known_phi_np2 = NULL);
  void sample_static_levelset_on_nodes(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np2);
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
   * - n_viscous_subiterations
   * - voronoi_on_the_fly
   * The double parameters/variables that are saved/loaded are (in this order):
   * - tree_dimension[0 : P4EST_DIM - 1]
   * - dxyz_smallest_quad[0 : P4EST_DIM - 1]
   * - surface_tension
   * - mu_minus
   * - mu_plus
   * - rho_minus
   * - rho_plus
   * - t_n
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
                               save_or_load flag, const mpi_environment_t* mpi = NULL);
  void fill_or_load_double_parameters(save_or_load flag, std::vector<PetscReal>& data, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion);
  void fill_or_load_integer_parameters(save_or_load flag, std::vector<PetscInt>& data, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion);

  inline u_int npseudo_time_steps() const
  {
    return 10*MAX(3, (int)ceil((sl_order + 1)*cfl_advection)); // in case someone has the brilliant idea of using a stupidly large advection cfl ("+1" for safety)
  }

  void transfer_face_sampled_vnp1_to_cells(Vec vnp1_minus_on_cells, Vec vnp1_plus_on_cells) const; // for exhaustive vtk exportations

  void build_sharp_pressure(Vec sharp_pressure) const;

  inline void compute_dt_np1()
  {
    dt_np1 = MIN(get_advection_dt(), get_visco_capillary_dt() + sqrt(SQR(get_visco_capillary_dt()) + SQR(get_capillary_dt())));
    dt_np1 = MIN(dt_np1, final_time - t_n - dt_n);
  }

  void build_jump_in_normal_velocity(); // possible jump in normal velocity = mass flux*(jump in 1.0/rho)
  void build_total_interface_tangential_force(); // possible Marangoni + possible component of user-defined

public:
  my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1_, my_p4est_node_neighbors_t *ngbd_n_, my_p4est_faces_t *faces_n_,
                             my_p4est_node_neighbors_t *fine_ngbd_n = NULL);

  my_p4est_two_phase_flows_t(const mpi_environment_t& mpi, const char* path_to_saved_state);
  ~my_p4est_two_phase_flows_t();

  inline double get_capillary_dt() const
  {
    return sqrt(cfl_capillary*(rho_minus + rho_plus)*pow(MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])), 3)/(4.0*M_PI*max_surface_tension_in_band_of_two_cells));
  }

  inline double get_visco_capillary_dt() const
  {
    return cfl_visco_capillary*MIN(mu_minus, mu_plus)*MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]))/max_surface_tension_in_band_of_two_cells;
  }

  inline double get_advection_dt() const
  {
    return cfl_advection * MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]))/MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus);
  }

  inline void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p)
  {
    bc_velocity = bc_v;
    bc_pressure = bc_p;
  }

  // use this in real life
  inline void set_external_forces_per_unit_mass(CF_DIM *external_force_per_unit_mass_[P4EST_DIM])
  {
    for(u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      this->force_per_unit_mass_minus[dir] = external_force_per_unit_mass_[dir];
      this->force_per_unit_mass_plus[dir] = external_force_per_unit_mass_[dir];
    }
  }
  // use this for your mind experiments/self-convincing endeavors/earn your ticket out of a dysfunctional work environment...
  inline void set_external_forces_per_unit_mass(CF_DIM *external_force_per_unit_mass_minus_[P4EST_DIM], CF_DIM *external_force_per_unit_mass_plus_[P4EST_DIM])
  {
    for(u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      this->force_per_unit_mass_minus[dir] = external_force_per_unit_mass_minus_[dir];
      this->force_per_unit_mass_plus[dir] = external_force_per_unit_mass_plus_[dir];
    }
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

  inline void set_surface_tension(Vec surface_tension_)
  {
    P4EST_ASSERT(VecIsSetForNodes(surface_tension_, interface_manager->get_interface_capturing_ngbd_n().get_nodes(), p4est_n->mpicomm, 1));
    user_defined_nonconstant_surface_tension = surface_tension_;
  }

  inline void set_mass_flux(Vec mass_flux_)
  {
    P4EST_ASSERT(VecIsSetForNodes(mass_flux_, interface_manager->get_interface_capturing_ngbd_n().get_nodes(), p4est_n->mpicomm, 1));
    user_defined_mass_flux = mass_flux_;
  }

  inline void set_densities(const double& rho_m_, const double& rho_p_)
  {
    rho_minus = rho_m_;
    rho_plus  = rho_p_;
  }

  void set_phi_np1(Vec phi_np1_on_interface_capturing_nodes, const interpolation_method& method = linear, Vec phi_np1_on_computational_nodes_ = NULL);
  void set_interface_velocity_n(CF_DIM* interface_velocity_n_functor[P4EST_DIM]);
  void set_node_velocities_nm1(const CF_DIM* vnm1_minus_functor[P4EST_DIM], const CF_DIM* vnm1_plus_functor[P4EST_DIM]);
  void set_node_velocities_n(const CF_DIM* vn_minus_functor[P4EST_DIM], const CF_DIM* vn_plus_functor[P4EST_DIM]);
//  void set_face_velocities_np1(CF_DIM* vnp1_m_[P4EST_DIM], CF_DIM* vnp1_p_[P4EST_DIM]);

  void compute_second_derivatives_of_n_velocities();
  void compute_second_derivatives_of_nm1_velocities();
  void compute_second_derivatives_of_interface_velocity_n();

  /*!
   * \brief set_degree_guess_vstar_k sets the degree of the extrapolation for guessing vstar_k in the grid update process.
   * \param degree_ [in] : desired "degree", accepted values are:
   * -1 : no guess is built;
   * 0  : the latest velocity field is considered, i.e., vnp2_star <-- vnp1;
   * 1  : a linear extrapolation in time is considered, i.e., vnp2_star <-- vnp1 + vnp1_prime*dt_np1
   * 2  : a quadratic extrapolation in time is considered, i.e., vnp2_star <-- vnp1 + vnp1_prime*dt_np1 + vnp1_prime_prime*0.5*dt_np1*dt_np1
   * (the time derivative vnp1_prime and/or vnp1_prime_prime are evaluated using backward difference formulae with vnp1, vn (degree 1 and 2) and vnm1 (degree 2))
   */
  inline void set_degree_guess_vstar_k(const int& degree_)
  {
    if(degree_ < -1 || degree_ > 2)
      throw  std::invalid_argument("my_p4est_two_phase_flows_t::set_degree_guess_vstar_k(): choose -1 (deactivate guess), 0 (take latest velocity field), 1 (linear extrapolation in time) or 2 (quadratic extrapolation in time)");
    degree_guess_v_star_face_k = degree_;
  }

  inline void set_semi_lagrangian_order_advection(const int& sl_)
  {
    sl_order = sl_;
  }
  inline void set_semi_lagrangian_order_interface(const int& sl_)
  {
    sl_order_interface = sl_;
  }
  inline void set_n_viscous_subiterations(const int& nn) { n_viscous_subiterations = nn; }

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
    dt_np1  = -DBL_MAX; // absurd value
  }

  inline void set_interface_force(Vec interface_force_) { user_defined_interface_force = interface_force_; };

  inline bool viscosities_are_equal() const { return fabs(mu_minus - mu_plus) < EPS*MAX(fabs(mu_minus), fabs(mu_plus)); }
  inline bool mass_densities_are_equal() const { return fabs(rho_minus - rho_plus) < EPS*MAX(fabs(rho_minus), fabs(rho_plus)); }

  inline void set_dt(double dt_n_) {dt_n = dt_n_; }

  inline double get_dt_n() const                                            { return dt_n; }
  inline double get_dt_nm1() const                                          { return dt_nm1; }
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


  void compute_non_viscous_pressure_jump();
  void solve_for_pressure_guess();
  void solve_viscosity();
  void solve_projection();

  void solve_time_step(const double& velocity_relative_threshold, const int& max_niter);

  void set_cell_jump_solver(const jump_solver_tag& solver_to_use,   const KSPType& KSP_ = "default", const PCType& PC_ = "default");
  void set_face_jump_solvers(const jump_solver_tag& solver_to_use,  const KSPType& KSP_ = "default", const PCType& PC_ = "default");

  inline void fetch_interface_points_with_second_order_accuracy() {
    fetch_interface_FD_neighbors_with_second_order_accuracy = true;
    interface_manager->evaluate_FD_theta_with_quadratics(fetch_interface_FD_neighbors_with_second_order_accuracy);
  }

  void compute_velocities_at_nodes();
  void set_interface_velocity_np1();
  void save_vtk(const std::string& vtk_directory, const int& index, const bool& exhaustive = false) const;
  void update_from_tn_to_tnp1(const int& n_reinit_iter = 1);

  inline double get_max_velocity() const        { return MAX(max_L2_norm_velocity_minus, max_L2_norm_velocity_plus); }
  inline double get_max_velocity_minus() const  { return max_L2_norm_velocity_minus; }
  inline double get_max_velocity_plus() const   { return max_L2_norm_velocity_plus; }

  inline double volume_in_negative_domain() const { return interface_manager->volume_in_negative_domain(); }

  void get_average_interface_velocity(double avg_itfc_velocity[P4EST_DIM]);

  inline int get_rank() const { return p4est_n->mpirank; }

  /*!
   * \brief save_state saves the solver states in a subdirectory 'backup_' created under the user-provided root-directory.
   * the n_states (> 0) latest succesive states can be saved, with automatic update of the subdirectory names.
   * If more than n_states subdirectories exist at any time when this function is called, it will automatically delete the
   * extra subdirectories.
   * \param path_to_root_directory: path to the root exportation directory. n_saved subdirectories 'backup_' will be created
   * under the root directory, in which successive solver states will be saved.
   * \param n_saved: number of solver states to keep in memory (default is 1)
   */
  void save_state(const char* path_to_root_directory, const int& n_saved = 1);

  /*!
   * \brief load_state loads a solver state that has been previously saved on disk
   * \param mpi             [in]    mpi environment to load the solver state in
   * \param path_to_folder  [in]    path to the folder where the solver state has been stored (absolute path)
   * [NOTE :] the function will destroy and overwrite any grid-related structure like p4est_n, nodes_n, ghost_n, faces_n, etc.
   * if they have already been constructed beforehand...
   * WARNING: this function throws an std::invalid_argument exception if path_to_folder is invalid
   */
  void load_state(const mpi_environment_t& mpi, const char* path_to_folder);


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

  inline void print_phi_np1() const
  {
    PetscErrorCode ierr;
    if(phi_np1_on_computational_nodes != NULL && phi_np1_on_computational_nodes != phi_np1)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi_np1_on_computational_nodes = " << std::endl;
      ierr = VecView(phi_np1_on_computational_nodes, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
    if(phi_np1 != NULL)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi_np1 = " << std::endl;
      ierr = VecView(phi_np1, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
    if(phi_np1_xxyyzz != NULL)
    {
      if(p4est_n->mpirank == 0)
        std::cout << "phi_np1_xxyyzz = " << std::endl;
      ierr = VecView(phi_np1_xxyyzz, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_pressures() const
  {
    if(pressure_minus != NULL)
    {
      PetscErrorCode ierr;
      if(p4est_n->mpirank == 0)
        std::cout << "pressure_minus = " << std::endl;
      ierr = VecView(pressure_minus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
    if(pressure_plus != NULL)
    {
      PetscErrorCode ierr;
      if(p4est_n->mpirank == 0)
        std::cout << "pressure_plus = " << std::endl;
      ierr = VecView(pressure_plus, PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
    }
  }

  inline void print_viscosity_rhs() const
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(p4est_n->mpirank == 0)
        std::cout << "viscosity rhs[" << int(dim) << "] = " << std::endl;
      ierr = VecView(face_jump_solver->rhs[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
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
      ierr = VecView(face_jump_solver->solution[dim], PETSC_VIEWER_STDOUT_WORLD); CHKERRXX(ierr);
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
  inline int get_n_viscous_subiterations() const { return n_viscous_subiterations; }
  inline double get_advection_cfl()   const { return cfl_advection; }

  inline void set_static_interface(bool interface_is_static)
  {
    static_interface = interface_is_static;
  }

  inline void set_log_file(FILE* log_)
  {
    log_file = log_;
  }

  inline void set_final_time(const double& final_time_) { final_time = final_time_; }

  /*!
   * \brief initialize_time_steps sets dt_nm1 = dt_n = time_step;
   * --> to be called after full initialization (i.e., after mus,
   * rhos, surface_tension and node_velocity at tn have been set up)
   */
  void initialize_time_steps();

  inline double get_tn()            const { return t_n;                 }
  inline double get_tnp1()          const { return t_n + dt_n;          }
  inline double get_progress_n()    const { return t_n - tstart;        }
  inline double get_progress_np1()  const { return t_n + dt_n - tstart; }


  // GENERAL functions of interests for initializing runs, monitoring exportations, etc...
  // made static since we are not even working on an object (don't even need one for pre-construction steps for instance)
  static void build_initial_computational_grids(const mpi_environment_t &mpi, my_p4est_brick_t *brick, p4est_connectivity_t* connectivity,
                                                const splitting_criteria_cf_and_uniform_band_t* data_with_with_phi_n, const splitting_criteria_cf_and_uniform_band_t* data_with_with_phi_np1,
                                                p4est_t* &p4est_nm1, p4est_ghost_t* &ghost_nm1, p4est_nodes_t* &nodes_nm1, my_p4est_hierarchy_t* &hierarchy_nm1, my_p4est_node_neighbors_t* &ngbd_nm1,
                                                p4est_t* &p4est_n, p4est_ghost_t* &ghost_n, p4est_nodes_t* &nodes_n, my_p4est_hierarchy_t* &hierarchy_n, my_p4est_node_neighbors_t* &ngbd_n,
                                                my_p4est_cell_neighbors_t* &ngbd_c, my_p4est_faces_t* &faces, Vec &phi_np1_computational_nodes);

  static void build_initial_interface_capturing_grid(p4est_t* p4est_n, my_p4est_brick_t* brick, const splitting_criteria_cf_t* subrefined_data_with_phi_np1,
                                                     p4est_t* &subrefined_p4est, p4est_ghost_t* &subrefined_ghost, p4est_nodes_t* &subrefined_nodes,
                                                     my_p4est_hierarchy_t* &subrefined_hierarchy, my_p4est_node_neighbors_t* &subrefined_ngbd_n, Vec &subrefined_phi);



};

#endif // MY_P4EST_TWO_PHASE_FLOWS_H
