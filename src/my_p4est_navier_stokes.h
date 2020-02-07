#ifndef MY_P4EST_NAVIER_STOKES_H
#define MY_P4EST_NAVIER_STOKES_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_save_load.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_save_load.h>
#endif

typedef enum
{
  SAVE=3541,
  LOAD
} save_or_load;

class my_p4est_navier_stokes_t
{
protected:

  class splitting_criteria_vorticity_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
                      const double* tree_dimensions,
                      const double *phi_p, const double *vorticity_p, const double *smoke_p = NULL);
  public:
    double max_L2_norm_u;
    double threshold;
    double uniform_band;
    double smoke_thresh;
    splitting_criteria_vorticity_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold, double max_L2_norm_u, double smoke_thresh);
    bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, Vec vorticity, Vec smoke);
  };

  class wall_bc_value_hodge_t : public CF_DIM
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    wall_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
    double operator()(DIM(double x, double y, double z)) const;
  };

  class interface_bc_value_hodge_t : public CF_DIM
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    interface_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
    double operator()(DIM(double x, double y, double z)) const;
  };

  my_p4est_brick_t *brick;
  p4est_connectivity_t *conn;

  p4est_t *p4est_nm1;
  p4est_ghost_t *ghost_nm1;
  p4est_nodes_t *nodes_nm1;
  my_p4est_hierarchy_t *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  p4est_t *p4est_n;
  p4est_ghost_t *ghost_n;
  p4est_nodes_t *nodes_n;
  my_p4est_hierarchy_t *hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_faces_t *faces_n;

  double dxyz_min[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double convert_to_xyz[P4EST_DIM];

  double mu;
  double rho;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_u;
  double uniform_band;
  double threshold_split_cell;
  double n_times_dt;
  bool   dt_updated;

  Vec phi;
  Vec hodge;
  Vec dxyz_hodge[P4EST_DIM];

  Vec vstar[P4EST_DIM];
  Vec vnp1 [P4EST_DIM];

  Vec vnm1_nodes[P4EST_DIM];
  Vec vn_nodes  [P4EST_DIM];
  Vec vnp1_nodes[P4EST_DIM];

  // semi-lagrangian backtraced points for faces (needed in viscosity step's setup, needs to be done only once)
  // no need to destroy these, not dynamically allocated...
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM];
  std::vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM]; // used only if sl_order == 2

  // second_derivatives...[i][j] = second derivatives of velocity component j along Cartesian direction i
  Vec second_derivatives_vnm1_nodes[P4EST_DIM][P4EST_DIM];
  Vec second_derivatives_vn_nodes[P4EST_DIM][P4EST_DIM];

  Vec vorticity;

  Vec pressure;

  Vec smoke;
  CF_DIM *bc_smoke;
  bool refine_with_smoke;
  double smoke_thresh;

  int sl_order;

  Vec face_is_well_defined[P4EST_DIM];

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM bc_hodge;
  BoundaryConditionsDIM *bc_v;

  wall_bc_value_hodge_t wall_bc_value_hodge;
  interface_bc_value_hodge_t interface_bc_value_hodge;

  CF_DIM *external_forces[P4EST_DIM];

  my_p4est_interpolation_nodes_t *interp_phi;

  double compute_dxyz_hodge( p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int dir);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx);

  void compute_max_L2_norm_u();

  void compute_vorticity();

  void compute_norm_grad_v();

  bool is_in_domain(const double xyz_[]) const {
    bool to_return = true;
    for (unsigned char dd = 0; dd < P4EST_DIM && to_return; ++dd)
      to_return = (xyz_[dd] - xyz_min[dd] > -0.1-dxyz_min[dd] && xyz_[dd] - xyz_max[dd] < 0.1*dxyz_min[dd]) || is_periodic(p4est_n, dd);
    return to_return;
  };

  bool is_no_slip(const double xyz_[]) const {
    return (ANDD(bc_v[0].wallType(xyz_) == DIRICHLET, bc_v[1].wallType(xyz_) == DIRICHLET, bc_v[2].wallType(xyz_) == DIRICHLET) && bc_pressure->wallType(xyz_) == NEUMANN);
  }

  /*!
   * \brief save_or_load_parameters : save or loads the solver parameters in the two files of paths
   * given by sprintf(path_1, "%s_integers", filename) and sprintf(path_2, "%s_doubles", filename)
   * The integer parameters that are saved/loaded are (in this order):
   * - P4EST_DIM
   * - refine_with_smoke
   * - data->min_lvl
   * - data->max_lvl
   * - sl_order
   * The double parameters/variables that are saved/loaded are (in this order):
   * - dxyz_min[0:P4EST_DIM-1]
   * - xyz_min[0:P4EST_DIM-1]
   * - xyz_max[0:P4EST_DIM-1]
   * - convert_to_xyz[0:P4EST_DIM-1]
   * - mu
   * - rho
   * - the simulation time tn
   * - dt_n
   * - dt_nm1
   * - max_L2_norm_u
   * - uniform_band
   * - threshold_split_cell
   * - n_times_dt
   * - smoke_threshold
   * - data->lip
   * The integer and double parameters are saved separately in two different files to avoid reading errors due to
   * byte padding (occurs in order to ensure data alignment when written in file)...
   * \param filename[in]: basename of the path to the files to be written or read (absolute path)
   * \param data[inout] : splitting criterion to be exported/loaded
   * \param flag[in]    : switch the behavior between write or read
   * \param tn[inout]   : in write mode, simulation time at which the function is called (to be saved, unmodified)
   *                      in read mode, simulation time at which the data were saved (to be read from file and stored in tn)
   * \param mpi[in]     : pointer to the mpi_environment_t (necessary for the load, disregarded for the save)
   * [note: implemented in one given function with switched behavior to avoid ambiguity and confusion due to code duplication
   * in several functions to be modified in the future if the parameter/variable order or the parameter/variable list is changed
   * (the save-state files are binary files, order and number of read/write operations is crucial)]
   * WARNING: this function throws an std::invalid_argument exception if the files can't be found when loading parameters
   * Raphael EGAN
   */
  void save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_criterion, save_or_load flag, double& tn, const mpi_environment_t* mpi = NULL);
  void fill_or_load_double_parameters(save_or_load flag, PetscReal* data, splitting_criteria_t* splitting_criterion, double& tn);
  void fill_or_load_integer_parameters(save_or_load flag, PetscInt* data, splitting_criteria_t* splitting_criterion);

  /*!
   * \brief load_state loads a solver state that has been previously saved on disk
   * \param mpi             [in]    mpi environment to load the solver state in
   * \param path_to_folder  [in]    path to the folder where the solver state has been stored (absolute path)
   * \param tn              [inout] simulation time at which the data were saved (to be read from saved solver state)
   * [NOTE :] the function will destroy and overwrite any grid-related structure like p4est_n, nodes_n, ghost_n, faces_n, etc.
   * if they have already been constructed beforehand...
   * WARNING: this function throws an std::invalid_argument exception if path_to_folder is invalid
   * Raphael EGAN
   */
  void load_state(const mpi_environment_t& mpi, const char* path_to_folder, double& tn);
public:
  my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n);
  my_p4est_navier_stokes_t(const mpi_environment_t& mpi, const char* path_to_saved_state, double &simulation_time);
  ~my_p4est_navier_stokes_t();

  void set_parameters(double mu, double rho, int sl_order, double uniform_band, double threshold_split_cell, double n_times_dt);

  void set_smoke(Vec smoke, CF_DIM *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);

  void set_phi(Vec phi);

  void set_external_forces(CF_DIM **external_forces);

  void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p);

  void set_velocities(Vec *vnm1, Vec *vn);

  void set_velocities(CF_DIM **vnm1, CF_DIM **vn);

  void set_vstar(Vec *vstar);

  void set_hodge(Vec hodge);

  inline double get_dt() { return dt_n; }

  inline my_p4est_node_neighbors_t* get_ngbd_n() { return ngbd_n; }

  inline my_p4est_cell_neighbors_t* get_ngbd_c() { return ngbd_c; }

  inline p4est_t *get_p4est() { return p4est_n; }

  inline p4est_t *get_p4est_nm1() { return p4est_nm1; }

  inline p4est_ghost_t *get_ghost() { return ghost_n; }

  inline p4est_nodes_t *get_nodes() { return nodes_n; }

  inline my_p4est_faces_t* get_faces() { return faces_n; }

  inline my_p4est_hierarchy_t* get_hierarchy() { return hierarchy_n; }

  inline Vec get_phi() { return phi; }

  inline Vec* get_velocity() { return vn_nodes; }

  inline Vec* get_velocity_np1() { return vnp1_nodes; }

  inline Vec* get_vstar() { return vstar; }

  inline Vec* get_vnp1() { return vnp1; }

  inline Vec get_hodge() { return hodge; }

  inline Vec get_smoke() { return smoke; }
  inline bool get_refine_with_smoke() { return refine_with_smoke; }
  inline double get_smoke_threshold() { return smoke_thresh; }

  inline Vec get_pressure() { return pressure; }

  inline my_p4est_interpolation_nodes_t* get_interp_phi() { return interp_phi; }

  inline double get_max_L2_norm_u() { return max_L2_norm_u; }

  inline double get_mu() const {return mu;}
  inline double get_split_threshold() const {return threshold_split_cell;}
  inline double get_rho() const {return rho;}
  inline double get_uniform_band() const {return uniform_band;}
  inline double get_cfl() const {return n_times_dt;}
  inline int get_sl_order() const {return sl_order;}
  inline double get_length_of_domain() const {return (xyz_max[0]-xyz_min[0]);}
  inline double get_height_of_domain() const {return (xyz_max[1]-xyz_min[1]);}
#ifdef P4_TO_P8
  inline double get_width_of_domain() const {return (xyz_max[2]-xyz_min[2]);}
#endif
  inline my_p4est_brick_t* get_brick() const {return brick;}

  void solve_viscosity()
  {
    my_p4est_poisson_faces_t* face_solver = NULL;
    solve_viscosity(face_solver);
    delete face_solver;
  }
  void solve_viscosity(my_p4est_poisson_faces_t* &face_poisson_solver, const bool use_initial_guess = false, const KSPType ksp = KSPBCGS, const PCType pc = PCSOR);

  void solve_projection()
  {
    my_p4est_poisson_cells_t* cell_solver = NULL;
    solve_projection(cell_solver);
    delete cell_solver;
  }
  void solve_projection(my_p4est_poisson_cells_t* &cell_poisson_solver, const bool use_initial_guess = false, const KSPType ksp = KSPBCGS, const PCType pc = PCSOR);


  void compute_velocity_at_nodes();

  void set_dt(double dt_nm1, double dt_n);

  void set_dt(double dt_n);

  /*!
   * \brief computes the next time step based on the desired cfl condition, but _locally_, i.e.,
   * for each quadrant in the domain, the local velocity magnitude is calculated and a local maximum
   * time step is estimated based on that local velocity magnitude and the quadrant size. Then, the
   * minimum of all such dt is enforced. This should avoid very small time steps due to large velocities in
   * coarse areas when using a very fine grid to capture zero no-slip conditions elsewhere.
   * \param min_value_for_umax: minimum value to be considered for the local velocities (to avoid crazy large
   * time steps because a local velocity is close to 0)...
   * Raphael EGAN
   */
  void compute_adapted_dt(double min_value_for_umax = 1.0);
  void compute_dt(double min_value_for_umax = 1.0);

  void advect_smoke(my_p4est_node_neighbors_t* ngbd_n_np1, Vec* vnp1, Vec smoke_np1);

  void extrapolate_bc_v(my_p4est_node_neighbors_t *ngbd, Vec *v, Vec phi);

  bool update_from_tn_to_tnp1(const CF_DIM *level_set=NULL, bool keep_grid_as_such=false, bool do_reinitialization=true);

  void compute_pressure();

  void compute_forces(double *f);

  void save_vtk(const char* name);

  /*!
   * \brief calculates the mass flow through slices in Cartesian direction in the computational domain. The slices must coincide with
   * cell faces, they mustn't cross any quadrant in the forest. Therefore, their location must coincide with a logical coordinate
   * for faces of the coarsest computational cells.
   * In debug mode, the function throws std::invalid_argument if this is not satisfied. In release, the section's location is changed to
   * the closest consistent location.
   * \param dir         [in]: Cartesian direction of the normal to the slice of interest (dir::x, dir::y or dir::z).
   * \param section     [in]: vector of coordinates along the direction of ineterest for the slices. section[ii] must be such that
   * section[ii] = xyz_min[dir] + nn*(xyz_max[dir]-xyz_min[dir])/(ntrees[dir]*(1<<min_lvl)) where nn must be a positive integer.
   * \param mass_flows  [out]: vector of computed mass flows across the sections of interest.
   * Raphael Egan
   */
  void global_mass_flow_through_slice(const unsigned int& dir, std::vector<double>& section, std::vector<double>& mass_flows) const;

  /*!
   * \brief calculates the friction force applied onto the fluid from the no-slip walls. This function requires a uniform tesselation of all
   * no-slip sections of the wall of the computational domain. A wall point in a wall quadrant is considered a no-slip wall point if the boundary
   * condition type is DIRICHLET for all velocity components and the boundary condition type is NEUMANN for pressure.
   * NOTE 1: this function uses the velocity field at FACES (no use of the point-interpolated values) to exploit the consistency with regard to the
   * cell-centered pressure values!
   * NOTE 2: the force component in Cartesian direction dir (dir = 0,1,2) is obtained by surface integration of the dir component of the surface
   * stress vector on all no-slip wall surfaces. Given a wall face f of normal aligned with direction dir, the corresponding wall surface element is
   * - the face itself if the wall normal is aligned with direction dir as well;
   * - the wall element of dimensions dxyz_min[(dir+1)%P4EST_DIM]*dxyz_min[(dir+2)%P4EST_DIM] (dxyz_min[(dir+1)%P4EST_DIM] in 2D) centered at
   * the wall projection of the considered face center;
   * Only the 'logical' no-slip fraction of that element contributes to the global integral: if (the wall projection of) the considered face center
   * is no-slip, two neighbors that are rr*0.5*dxyz away in a transverse directions are found. For each such wall neighbor, if it is a no-slip point,
   * i) 0.5 is added to the fraction of the wall area  element that is considered no-slip in 2D
   * ii) two further neighbors of that point are found in the other transverse direction and each no-slip of them contributes with 0.25 to the fraction
   * of the wall area element that is considered no-slip in 3D
   * --> done as such to deal with confusing transitions from slip to no-slip in SHS channels...
   * \param wall_forces [out]: wall force components (P4EST_DIM array of doubles)
   * \param with_pressure [in]: flag including the pressure terms in the calculations (i.e. for wall-aligned faces)
   * Raphael EGAN
   */
  void get_noslip_wall_forces(double wall_forces[], const bool with_pressure = false) const;

  /*!
   * \brief save_state saves the solver states in a subdirectory 'backup_' created under the user-provided root-directory.
   * the n_states (>0) latest succesive states can be saved, with automatic update of the subdirectory names.
   * If more than n_states subdirectories exist at any time when this function is called, it will automatically delete the extra
   * subdirectories.
   * \param path_to_root_directory: path to the root exportation directory. n_saved subdirectories 'backup_' will be created
   * under the root directory, in which successive solver states will be saved.
   * \param tn: simulation time at which the function is called
   * \param n_saved: number of solver states to keep in memory (default is 1)
   * Raphael EGAN
   */
  void save_state(const char* path_to_root_directory, double tn, unsigned int n_saved=1);

  void refine_coarsen_grid_after_restart(const CF_DIM *level_set, bool do_reinitialization = true);
  size_t memory_estimate() const;

  void get_slice_averaged_vnp1_profile(unsigned short vel_component, unsigned short axis, std::vector<double>& avg_velocity_profile);

};



#endif /* MY_P4EST_NAVIER_STOKES_H */
