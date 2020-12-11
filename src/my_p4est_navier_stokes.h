#ifndef MY_P4EST_NAVIER_STOKES_H
#define MY_P4EST_NAVIER_STOKES_H

#include <petsc.h>
#include <algorithm>

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

typedef enum {
  grid_update,
  viscous_step,
  projection_step,
  velocity_interpolation
} ns_task;

class execution_time_accumulator
{
private:
  unsigned int counter;
  double total_time;
  double fixed_point_extra_time;
public:
  execution_time_accumulator()
  {
    reset();
  }
  void reset()
  {
    counter = 0;
    total_time = 0.0;
    fixed_point_extra_time = 0.0;
  }
  void add(const double &execution_time)
  {
    total_time += execution_time;
    if(counter > 0)
      fixed_point_extra_time += execution_time;
    counter++;
  }

  double read_total_time() const { return total_time; }
  double read_fixed_point_extra_time() const { return fixed_point_extra_time; }
  unsigned int read_counter() const { return counter; }
};

class my_p4est_navier_stokes_t
{
  friend class my_p4est_shs_channel_t;
protected:

  class splitting_criteria_vorticity_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
                      const double* tree_dimensions,
                      const double *phi_p, const double *vorticity_p, const double *smoke_p, const double* norm_grad_u_p);
  public:
    double max_L2_norm_u;
    double threshold_vorticity;
    double threshold_norm_grad_u;
    double uniform_band;
    double smoke_thresh;
    splitting_criteria_vorticity_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold_vorticity, double max_L2_norm_u, double smoke_thresh, double threshold_norm_grad_u);
    bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, Vec vorticity, Vec smoke, Vec norm_grad_u);
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
  double vorticity_threshold_split_cell;
  double norm_grad_u_threshold_split_cell;
  double n_times_dt;
  bool   dt_updated;

  interpolation_method interp_v_viscosity;
  interpolation_method interp_v_update;

  Vec phi, grad_phi;
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
  std::vector<double> backtraced_v_n[P4EST_DIM];
  std::vector<double> backtraced_v_nm1[P4EST_DIM]; // used only if sl_order == 2

  // face interpolator to nodes: store them in memory to accelerate execution if static grid
  bool interpolators_from_face_to_nodes_are_set;
  std::vector<face_interpolator> interpolator_from_face_to_nodes[P4EST_DIM];

  // second_derivatives...[i][j] = second derivatives of velocity component j along Cartesian direction i
  Vec second_derivatives_vnm1_nodes[P4EST_DIM][P4EST_DIM];
  Vec second_derivatives_vn_nodes[P4EST_DIM][P4EST_DIM];

  Vec vorticity;
  Vec norm_grad_u;

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

  CF_DIM *external_forces_per_unit_volume[P4EST_DIM];
  CF_DIM *external_forces_per_unit_mass[P4EST_DIM];

  my_p4est_interpolation_nodes_t *interp_phi, *interp_grad_phi;

  double compute_dxyz_hodge(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& dir);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx);

  void compute_max_L2_norm_u();

  void compute_vorticity();
  void compute_Q_and_lambda_2_value(Vec& Q_value_nodes, Vec& lambda_2_nodes, const double U_scaling, const double x_scaling) const;

  inline void get_Q_and_lambda_2_values(const quad_neighbor_nodes_of_node_t& qnnn, const double *vnp1_p[P4EST_DIM], const double& x_scaling, const double& U_scaling, double& Qvalue, double& lambda_2_value) const
  {
    struct low_tri_idx {
      unsigned char operator()(const unsigned char& i, const unsigned char& j) const
      {
        P4EST_ASSERT(j <= i && i < P4EST_DIM);
        return (i*(i + 1)/2 + j);
      }

      double get_S_squared_plus_omega_squared(unsigned char i, unsigned char j, const double* S, const double* omega) const
      {
        double rr = 0.0;
        for (unsigned char k = 0; k < P4EST_DIM; ++k)
          rr += S[operator()(MAX(i, k), MIN(i, k))]*S[operator()(MAX(k, j), MIN(k, j))]
              + (k < i && j < k ? omega[operator()(i, k)]*omega[operator()(k, j)] : 0.0)
            - (k < i && k < j ? omega[operator()(i, k)]*omega[operator()(j, k)] : 0.0)
          - (i < k && j < k ? omega[operator()(k, i)]*omega[operator()(k, j)] : 0.0)
          + (i < k && k < j ? omega[operator()(k, i)]*omega[operator()(j, k)] : 0.0);
        return rr;
      }
    } tri_idx;

    // we will calculate only the lower triangular parts of the matrices
    double S[P4EST_DIM*(P4EST_DIM + 1)/2];                            // strain-rate tensor, symmetric
    double omega[P4EST_DIM*(P4EST_DIM + 1)/2];                        // vorticity tensor, anti-symmetric
    double S_squared_plus_omega_squared[P4EST_DIM*(P4EST_DIM + 1)/2]; // symmetric
    double lambda_coeffs[P4EST_DIM + 1];

    S[tri_idx(0, 0)] = qnnn.dx_central(vnp1_p[0])*x_scaling/U_scaling;                                    omega[tri_idx(0, 0)] = 0.0;
    S[tri_idx(1, 0)] = 0.5*(qnnn.dy_central(vnp1_p[0]) + qnnn.dx_central(vnp1_p[1]))*x_scaling/U_scaling; omega[tri_idx(1, 0)] = 0.5*(qnnn.dy_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[1]))*x_scaling/U_scaling;
    S[tri_idx(1, 1)] = qnnn.dy_central(vnp1_p[1])*x_scaling/U_scaling;                                    omega[tri_idx(1, 1)] = 0.0;
#ifdef P4_TO_P8
    S[tri_idx(2, 0)] = 0.5*(qnnn.dz_central(vnp1_p[0]) + qnnn.dx_central(vnp1_p[2]))*x_scaling/U_scaling; omega[tri_idx(2, 0)] = 0.5*(qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]))*x_scaling/U_scaling;
    S[tri_idx(2, 1)] = 0.5*(qnnn.dz_central(vnp1_p[1]) + qnnn.dy_central(vnp1_p[2]))*x_scaling/U_scaling; omega[tri_idx(2, 1)] = 0.5*(qnnn.dz_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[2]))*x_scaling/U_scaling;
    S[tri_idx(2, 2)] = qnnn.dz_central(vnp1_p[2])*x_scaling/U_scaling;                                    omega[tri_idx(2, 2)] = 0.0;
#endif

    Qvalue = 0; // Q = 0.5*(sum of all squared terms in omega - sum of all squared terms in S)
    for (unsigned char ii = 0; ii < P4EST_DIM; ++ii) {
      Qvalue -= 0.5*SQR(S[tri_idx(ii, ii)]); // only diagonal of S is nonzero
      S_squared_plus_omega_squared[tri_idx(ii, ii)] = tri_idx.get_S_squared_plus_omega_squared(ii, ii, S, omega);
      for (unsigned char jj = 0; jj < ii; ++jj){
        Qvalue += SQR(omega[tri_idx(ii, jj)]) - SQR(S[tri_idx(ii, jj)]); // no 0.5 factor by symmetry
        S_squared_plus_omega_squared[tri_idx(ii, jj)] = tri_idx.get_S_squared_plus_omega_squared(ii, jj, S, omega);
      }
    }

    lambda_coeffs[0] = 1.0;
    lambda_coeffs[1] = -SUMD(S_squared_plus_omega_squared[tri_idx(0, 0)], S_squared_plus_omega_squared[tri_idx(1, 1)], S_squared_plus_omega_squared[tri_idx(2, 2)]);
    lambda_coeffs[2] = S_squared_plus_omega_squared[tri_idx(0, 0)]*S_squared_plus_omega_squared[tri_idx(1, 1)] ONLY3D(+ S_squared_plus_omega_squared[tri_idx(0, 0)]*S_squared_plus_omega_squared[tri_idx(2, 2)] + S_squared_plus_omega_squared[tri_idx(1, 1)]*S_squared_plus_omega_squared[tri_idx(2, 2)])
        - SQR(S_squared_plus_omega_squared[tri_idx(1, 0)]) ONLY3D(- SQR(S_squared_plus_omega_squared[tri_idx(2, 0)]) - SQR(S_squared_plus_omega_squared[tri_idx(2, 1)]));
#ifdef P4_TO_P8
    lambda_coeffs[3] = SQR(S_squared_plus_omega_squared[tri_idx(2, 0)])*S_squared_plus_omega_squared[tri_idx(1, 1)] + SQR(S_squared_plus_omega_squared[tri_idx(2, 1)])*S_squared_plus_omega_squared[tri_idx(0, 0)] + SQR(S_squared_plus_omega_squared[tri_idx(1, 0)])*S_squared_plus_omega_squared[tri_idx(2, 2)]
        - 2.0*S_squared_plus_omega_squared[tri_idx(1, 0)]*S_squared_plus_omega_squared[tri_idx(2, 1)]*S_squared_plus_omega_squared[tri_idx(2, 0)] - S_squared_plus_omega_squared[tri_idx(0, 0)]*S_squared_plus_omega_squared[tri_idx(1, 1)]*S_squared_plus_omega_squared[tri_idx(2, 2)];
#endif

#ifndef P4_TO_P8
    const double discriminant = SQR(S_squared_plus_omega_squared[tri_idx(0, 0)] - S_squared_plus_omega_squared[tri_idx(1, 1)]) + 4.0*SQR(S_squared_plus_omega_squared[tri_idx(1, 0)]);
    P4EST_ASSERT(discriminant >= 0.0);
#endif

#ifndef P4_TO_P8
    lambda_2_value = 0.5*(-lambda_coeffs[1] - sqrt(discriminant))/lambda_coeffs[0];
#else
    double pp = (3.0*lambda_coeffs[0]*lambda_coeffs[2] - SQR(lambda_coeffs[1]))/(3.0*SQR(lambda_coeffs[0])); // must be strictly negative
#ifdef CASL_THROWS
    if(pp > 0.0)
      throw std::runtime_error("my_p4est_navier_stokes_t::get_Q_and_lambda_2_values(): obtained a positive pp");
#endif
    pp = MIN(pp, -EPS);
    double qq = (2.0*pow(lambda_coeffs[1], 3) - 9.0*lambda_coeffs[0]*lambda_coeffs[1]*lambda_coeffs[2] + 27.0*SQR(lambda_coeffs[0])*lambda_coeffs[3])/(27.0*pow(lambda_coeffs[0], 3));
    std::vector<double> lambda_values(P4EST_DIM);
    for (unsigned char kk = 0; kk < P4EST_DIM; ++kk)
    {
#ifdef CASL_THROWS
      if(fabs(3.0*qq*sqrt(-3.0/pp)/(2.0*pp)) > 1.0 + EPS)
        throw std::runtime_error("my_p4est_navier_stokes_t::get_Q_and_lambda_2_values(): argument of acos > 1 in absolute value");
#endif
      lambda_values[kk] = (lambda_coeffs[1]/(3.0*lambda_coeffs[0])) + 2.0*sqrt(-pp/3.0)*cos(acos(MAX(-1.0, MIN(1.0, 3.0*qq*sqrt(-3.0/pp)/(2.0*pp))))/3.0 - 2.0*M_PI*((double) kk)/3.0);
    }
    std::sort(lambda_values.begin(), lambda_values.end());
    lambda_2_value = lambda_values[1];
#endif
  }

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

  /*!
   * \brief compute_velocity_at_local_node : function for interpolating face-sampled velocity components at a local grid node. Made private
   * and such to avoid code duplication between loops over layer and inner nodes when interpolating velocities at all grid nodes
   * \param dir                 [in] : Cartesian component of the velocity component to interpolate
   * \param node_idx            [in] : local node index of to interpolate the dir^th velocity component at
   * \param vnp1_dir_read_p     [in] : pointer to (read-only) values of dir^th component of face-sampled velocity, at time (n + 1)
   * \param store_interpolators [in] : flag activating the storage of local face-interpolators (much faster to re-use when playing with static
   * grids) --> only valid with homogeneous Neumann boundary conditions if Neumann BC on the walls
   * \return value of the wegihted-lsqr-interpolated velocity component at the desired node
   */
  double compute_velocity_at_local_node(const p4est_locidx_t& node_idx, const unsigned char& dir, const double* vnp1_dir_read_p, const bool& store_interpolators);

  void calculate_viscous_stress_at_local_nodes(const p4est_locidx_t& node_idx, const double* phi_read_p, const double *grad_phi_read_p,
                                               const double* vnodes_read_p[P4EST_DIM], double* viscous_stress_p[P4EST_DIM]) const;

  class ns_time_step_analyzer_t
  {
    bool is_active;
    bool measuring;
    ns_task task;
    double start_time;
    std::map<ns_task, execution_time_accumulator> timings;

  public:
    ns_time_step_analyzer_t() : is_active(false), measuring(false) {}

    void reset()
    {
      for (std::map<ns_task, execution_time_accumulator>::iterator it = timings.begin(); it != timings.end(); ++it)
        it->second.reset();
    }
    const std::map<ns_task, execution_time_accumulator> & get_execution_times() const { return timings; }
    void activate()     { is_active = true; }
    bool is_on() const  { return is_active; }
    void start(const ns_task &task_)
    {
#ifdef CASL_THROWS
      if(measuring)
        throw std::runtime_error("my_p4est_navier_stokes_t::ns_timer_t::start_watch(): the watch needs to be stopped before being restarted.");
#endif
      measuring  = true;
      task       = task_;
      start_time = MPI_Wtime();
    }
    void stop()
    {
#ifdef CASL_THROWS
      if(!measuring)
        throw std::runtime_error("my_p4est_navier_stokes_t::ns_timer_t::stop_watch(): the watch can't be stopped if it wasn't started first.");
#endif
      timings[task].add(MPI_Wtime() - start_time);
      measuring = false;
    }
  } ns_time_step_analyzer;

public:
  my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n);
  my_p4est_navier_stokes_t(const mpi_environment_t& mpi, const char* path_to_saved_state, double &simulation_time);
  ~my_p4est_navier_stokes_t();

  void set_parameters(double mu, double rho, int sl_order, double uniform_band, double vorticity_threshold_split_cell, double n_times_dt, double norm_grad_u_threshold_split_cell = DBL_MAX);

  inline void set_interpolation_method_for_velocity_in_viscosity_step(const interpolation_method& desired_method) { interp_v_viscosity = desired_method; }
  inline void set_interpolation_method_for_velocity_in_update_step(const interpolation_method& desired_method) { interp_v_update = desired_method; }

  void set_smoke(Vec smoke, CF_DIM *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);

  void set_phi(Vec phi);

  // [Raphael:] original behavior was for forcing term defined as a force per unit volume
  inline void set_external_forces(CF_DIM **external_forces) { set_external_forces_per_unit_volume(external_forces); }

  /*!
   * \brief set_external_forces_per_unit_volume sets the external forcing term as a force *PER UNIT VOLUME*, that is,
   * if you consider the generic momentum
   *                      rho*D(u)/Dt = -grad(P) + rho*f + mu\nabla^2 u,
   * this function sets (rho*f) and not f only!
   *
   * WARNING: if this function is called *after* set_external_forces_per_unit_mass was called, the forcing term will be
   * reset to the new input, defined as a force per unit volume (the force per unit mass that was previously defined
   * will be discarded)
   * \param external_forces_per_unit_volume_ array of P4EST_DIM pointers to external forcing functions (in physical dimensions M/(L^2 T^2))
   */
  void set_external_forces_per_unit_volume(CF_DIM **external_forces_per_unit_volume_);

  /*!
   * \brief set_external_forces_per_unit_mass sets the external forcing term as a force *PER UNIT MASS*, that is,
   * if you consider the generic momentum
   *                      rho*D(u)/Dt = -grad(P) + rho*f + mu\nabla^2 u,
   * this function sets f and not (rho*f)!
   *
   * WARNING: if this function is called *after* set_external_forces_per_unit_volume was called, the forcing term will be
   * reset to the new input, defined as a force per unit mass (the force per unit volume that was previously defined
   * will be discarded)
   * \param external_forces_per_unit_mass_ array of P4EST_DIM pointers to external forcing functions (in physical dimensions L/T^2)
   */
  void set_external_forces_per_unit_mass  (CF_DIM **external_forces_per_unit_mass_);

  void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p);

  void set_velocities(Vec *vnm1, Vec *vn, const double *max_L2_norm_u = NULL);

  void set_velocities(CF_DIM **vnm1, CF_DIM **vn, const bool set_max_L2_norm_u = false);

  inline void set_vnp1_nodes(CF_DIM **vnp1)
  {
    PetscErrorCode ierr;
    double *vnp1_nodes_p[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      if(vnp1_nodes[dir] != NULL)
      {
        ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_nodes[dir], &vnp1_nodes_p[dir]); CHKERRXX(ierr);
    }

    for (size_t k = 0; k < nodes_n->indep_nodes.elem_count; ++k) {
      double node_xyz[P4EST_DIM];
      node_xyz_fr_n(k, p4est_n, nodes_n, node_xyz);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        vnp1_nodes_p[dir][k] = (*vnp1[dir])(node_xyz);
    }
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(vnp1_nodes[dir], &vnp1_nodes_p[dir]); CHKERRXX(ierr);
    }
  }

  inline void set_pressure(const CF_DIM &pressure_field)
  {
    PetscErrorCode ierr;
    double *pressure_p;
    if(pressure != NULL)
    {
      ierr = VecDestroy(pressure); CHKERRXX(ierr);
    }
    ierr = VecCreateGhostCells(p4est_n, ghost_n, &pressure); CHKERRXX(ierr);
    ierr = VecGetArray(pressure, &pressure_p); CHKERRXX(ierr);

    for (p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx) {
      p4est_tree_t* tree  = p4est_tree_array_index(p4est_n->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        double xyz_quad[P4EST_DIM];
        quad_xyz_fr_q(quad_idx, tree_idx, p4est_n, ghost_n, xyz_quad);
        pressure_p[quad_idx] = pressure_field(xyz_quad);
      }
    }

    for (size_t k = 0; k < ghost_n->ghosts.elem_count; ++k) {
      const p4est_quadrant_t* quad = p4est_quadrant_array_index(&ghost_n->ghosts, k);
      p4est_locidx_t quad_idx = p4est_n->local_num_quadrants + k;
      double xyz_quad[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, quad->p.piggy3.which_tree, p4est_n, ghost_n, xyz_quad);
      pressure_p[quad_idx] = pressure_field(xyz_quad);
    }

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(pressure, &pressure_p); CHKERRXX(ierr);
    }
  }

  void set_vstar(Vec *vstar);

  void set_hodge(Vec hodge);

  inline double get_dt() { return dt_n; }

  inline my_p4est_node_neighbors_t* get_ngbd_n() { return ngbd_n; }
  inline my_p4est_node_neighbors_t* get_ngbd_nm1() { return ngbd_nm1; }

  inline my_p4est_cell_neighbors_t* get_ngbd_c() { return ngbd_c; }

  inline p4est_t *get_p4est() { return p4est_n; }

  // ONLY FOR PEOPLE WHO KNOW WHAT THEY ARE DOING!!!
  inline void nullify_p4est_nm1() { p4est_nm1 = NULL; }


  inline const BoundaryConditionsDIM &get_bc_hodge() const { return bc_hodge; }
  inline p4est_t *get_p4est_nm1() { return p4est_nm1; }

  inline p4est_ghost_t *get_ghost() { return ghost_n; }

  inline p4est_nodes_t *get_nodes() { return nodes_n; }
  inline p4est_nodes_t *get_nodes_nm1() { return nodes_nm1; }

  inline my_p4est_faces_t* get_faces() { return faces_n; }

  inline my_p4est_hierarchy_t* get_hierarchy() { return hierarchy_n; }

  inline Vec get_phi() { return phi; }

  inline Vec* get_velocity() { return vn_nodes; }

  void copy_velocity_n(Vec* v_n_external){
    // Allows an external user to copy the object without interfering with NS solver's internal handling and objects
    PetscErrorCode ierr;
    for(unsigned char dim = 0; dim < P4EST_DIM;dim++){
        ierr = VecCopyGhost(vn_nodes[dim],v_n_external[dim]); CHKERRXX(ierr);
      }
  }

  inline Vec* get_velocity_np1() { return vnp1_nodes; }

  void copy_velocity_np1(Vec* v_np1_external){
    // Allows an external user to copy the object without interfering with NS solver's internal handling and objects
    PetscErrorCode ierr;
    for(unsigned char dim = 0; dim < P4EST_DIM;dim++){
        ierr = VecCopyGhost(vnp1_nodes[dim],v_np1_external[dim]); CHKERRXX(ierr);
      }
  }

  inline Vec* get_vstar() { return vstar; }

  inline Vec* get_vnp1() { return vnp1; }

  inline Vec get_hodge() { return hodge; }

  inline void copy_dxyz_hodge(Vec local_face_vectors[P4EST_DIM]) const
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      PetscErrorCode ierr = VecCopy(dxyz_hodge[dir], local_face_vectors[dir]); CHKERRXX(ierr);
    }
  }

  void copy_hodge(Vec hodge_external, const bool with_ghost = true){
    PetscErrorCode ierr;
    if(with_ghost){
      ierr = VecCopyGhost(hodge,hodge_external); CHKERRXX(ierr);
    }
    else{
      ierr = VecCopy(hodge,hodge_external); CHKERRXX(ierr);
    }
  }

  inline Vec get_smoke() { return smoke; }

  inline Vec get_vorticity(){return vorticity;}

  void copy_vorticity(Vec vort){
    PetscErrorCode ierr;
    ierr = VecCopyGhost(vorticity, vort); CHKERRXX(ierr);
  }
  inline bool get_refine_with_smoke() { return refine_with_smoke; }
  inline double get_smoke_threshold() { return smoke_thresh; }

  inline Vec get_pressure() { return pressure; }

  void copy_pressure(Vec press){
    PetscErrorCode ierr;
    ierr = VecCopyGhost(pressure,press); CHKERRXX(ierr);
  }

  inline my_p4est_interpolation_nodes_t* get_interp_phi() { return interp_phi; }
  inline my_p4est_interpolation_nodes_t* get_interp_grad_phi() { return interp_grad_phi; }

  inline double get_max_L2_norm_u() { return max_L2_norm_u; }

  inline double get_mu()                const { return mu; }
  inline double get_vorticity_split_threshold() const { return vorticity_threshold_split_cell; }
  inline double get_norm_grad_u_split_threshold() const { return norm_grad_u_threshold_split_cell; }
  inline double get_split_threshold()   const { return get_vorticity_split_threshold(); }
  inline double get_rho()               const { return rho; }
  inline double get_nu()                const { return mu/rho; }
  inline double get_uniform_band()      const { return uniform_band; }
  inline double get_cfl()               const { return n_times_dt; }
  inline int get_sl_order()             const { return sl_order; }
  inline int    get_lmax()              const { return ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl; }
  inline int    get_lmin()              const { return ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl; }
  inline MPI_Comm get_mpicomm()         const { return p4est_n->mpicomm; }
  inline int      get_mpirank()         const { return p4est_n->mpirank; }
  inline int      get_mpisize()         const { return p4est_n->mpisize; }
  inline double get_length_of_domain()  const { return (xyz_max[0] - xyz_min[0]); }
  inline double get_height_of_domain()  const { return (xyz_max[1] - xyz_min[1]); }
#ifdef P4_TO_P8
  inline double get_width_of_domain()   const { return (xyz_max[2] - xyz_min[2]); }
#endif
  inline my_p4est_brick_t* get_brick()  const { return brick; }

  void solve_viscosity()
  {
    my_p4est_poisson_faces_t* face_solver = NULL;
    solve_viscosity(face_solver);
    delete face_solver;
  }
  void solve_viscosity(my_p4est_poisson_faces_t* &face_poisson_solver, const bool& use_initial_guess = false, const KSPType& ksp = KSPBCGS, const PCType& pc = PCSOR);

  void solve_projection()
  {
    my_p4est_poisson_cells_t* cell_solver = NULL;
    solve_projection(cell_solver);
    delete cell_solver;
  }
  double solve_projection(my_p4est_poisson_cells_t* &cell_poisson_solver, const bool& use_initial_guess = false, const KSPType& ksp = KSPBCGS, const PCType& pc = PCSOR,
                          const bool& shift_to_zero_mean_if_floating = true, Vec hodge_old = NULL, Vec former_dxyz_hodge[P4EST_DIM] = NULL, const hodge_control& dxyz_hodge_chek = hodge_value);

  /*!
   * \brief get_correction_in_hodge_derivative_for_enforcing_mass_flow
   * This function calculates the (constant-in-space) correction to the gradient of the Hodge variable to enforce a desired mass flow.
   * The mass-flow forcing direction MUST be periodic (otherwise the approach is simply incoherent).
   * This function is ideally called after the projection step and its output should be used to dynamically adapt the driving body force
   * in order to enforce the desired mass flow rate in an inner loop, for every time step.
   * \param force_direction               [in]    Cartesian direction in which the forcing is applied (the problem must be periodic along that direction)
   * \param desired_mean_velocity         [in]    double value, specifying the desired bulk velocity in the forcing direction
   * \param mass_flow                     [in]    (optional) pointer to a constant double, *mass_flow is the mass flow along the Cartesian direction force_direction
   *                                              before forcing. If NULL, the function calculates the relevant value internally.
   * \return the (constant-in-space) correction required in the partial derivative of the Hodge variable along the Cartesian direction 'force_direction' that would be
   * required to have the desired mass flow --> can be used to correct the driving force term afterwards
   * Raphael EGAN
   */
  double get_correction_in_hodge_derivative_for_enforcing_mass_flow(const unsigned char& force_direction, const double& desired_mean_velocity, const double* mass_flow = NULL);

  void compute_velocity_at_nodes(const bool store_interpolators = false);

  double interpolate_pressure_at_node(const p4est_locidx_t& node_idx) const;

  void set_dt(double dt_nm1, double dt_n);

  void set_dt(double dt_n);

  /*!
   * \brief Allows you to set the grid for the solver yourself in each timestep -- in the case that the grid is handled externally from the navier stokes class, rather than internally
   * \param ngbd_nm1 [in]
   * \param ngbd [in]
   * \param faces [in]
   */
  void set_grids(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd, my_p4est_faces_t *faces);

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

// [Raphael:] remnant method that wasn't used anywhere
//  void extrapolate_bc_v(my_p4est_node_neighbors_t *ngbd, Vec *v, Vec phi);

  bool update_from_tn_to_tnp1(const CF_DIM *level_set=NULL, bool keep_grid_as_such=false, bool do_reinitialization=true);


  void update_from_tn_to_tnp1_grid_external(Vec phi_np1, p4est_t* p4est_np1, p4est_nodes_t* nodes_np1, p4est_ghost_t* ghost_np1, my_p4est_node_neighbors_t* ngbd_np1, my_p4est_faces_t* faces_np1, my_p4est_cell_neighbors_t* ngbd_c_np1, my_p4est_hierarchy_t* hierarchy_np1);
  void compute_pressure();

  void compute_forces(double *f);

  void save_vtk(const char* name, bool with_Q_and_lambda_2_value = false, const double U_scaling_for_Q_and_lambda_2 = 1.0, const double x_scaling_for_Q_and_lambda_2 = 1.0);

  /*!
   * \brief calculates the mass flow through a slice in Cartesian direction in the computational domain. The slice must coincide with
   * cell faces, it mustn't cross any quadrant in the forest. Therefore, their location must coincide with a logical coordinate
   * for faces of the coarsest computational cells.
   * In debug mode, the function throws std::invalid_argument if this is not satisfied. In release, the section's location is changed to
   * the closest consistent location.
   * \param dir         [in]: Cartesian direction of the normal to the slice of interest (dir::x, dir::y or dir::z).
   * \param section     [in]: coordinate along the direction of interest for the slice. section must be such that
   *                          section = xyz_min[dir] + nn*(xyz_max[dir]-xyz_min[dir])/(ntrees[dir]*(1<<min_lvl)) where nn must be a positive integer.
   * \param mass_flow  [out]: computed mass flows across the section of interest.
   * Raphael Egan
   */
  void global_mass_flow_through_slice(const unsigned int& dir, double& section, double& mass_flow) const;

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

  /*!
   * \brief refine_coarsen_grid_after_restart: this function refines and/or coarsens the grid to satisfy the (new) grid requirements after being
   * loaded from disk.
   * \param level_set           [in] : levelset function
   * \param do_reinitialization [in] : requires reinitialization for the node-sampled levelset function values, if true (default is true)
   * Raphael EGAN
   */
  void refine_coarsen_grid_after_restart(const CF_DIM *level_set, bool do_reinitialization = true);

  /*!
   * \brief memory_estimate: self-explanatory
   * \return memory estimates in number of bytes
   * Raphael EGAN
   */
  unsigned long int memory_estimate() const;

  /*!
   * \brief get_slice_averaged_vnp1_profile: calculates a slice-averaged profile for a velocity component in the domain. The direction along which
   * the profile is calculated (i.e. axis) cannot be equal to the velocity component (i.e. vel_component). The computational domain must be periodic
   * in all direction(s) perpendicular to axis.
   * The profile is calculated as mapped to an equivalent uniform grid of the finest refinement level. The velocity component value associated with
   * a face that is bigger than the finest possible is considered constant on the entire face and its associated weighting area is defined as
   * in 3D:
   *   (half the sum of the lengths of neighboring quads in one transverse direction) x (length of the face in the other transverse direction)
   * in 2D:
   *   (half the sum of the lengths of neighboring quads in the transverse direction).
   * Note 1:            only proc 0 has the correct result after completion.
   * Note 2:            assumes no interface in the domain, i.e. levelset < 0 everywhere.
   * local complexity:  every processor loops through their local faces only once.
   * communication:     MPI_reduce to proc 0 who is the only holding the correct results after completion.
   * \param vel_component         [in]    : velocity component of interest in the profile, 0 <= vel_component < P4EST_DIM
   * \param axis                  [in]    : axis along which the profile is calculated, 0 <= axis < P4EST_DIM
   * \param avg_velocity_profile  [inout] : vector containing the values of the desired velocity components (slice-averaged) along the profile axis
   *                                        this vector is resized (if needed) to contain brick->nxyztrees[axis]*(1<<data->max_lvl) elements as if
   *                                        the grid was uniform.
   * Raphael EGAN
   */
  void get_slice_averaged_vnp1_profile(const unsigned char& vel_component, const unsigned char& axis, std::vector<double>& avg_velocity_profile, const double u_scaling = 1.0);

  /*!
   * \brief get_line_averaged_vnp1_profiles: calculates line-averaged profiles for a velocity component in the domain. The direction along which
   * the profile is calculated (i.e. axis) cannot be equal to the velocity component (i.e. vel_component). The computational domain must be periodic
   * in the velocity component direction and/or in the averaging direction. The "transverse" direction is defined as the direction
   * - perpendicular to axis in 2D;
   * - perpendicular to axis and to averaging_direction in 3D.
   * Every line along the averaging_direction in the computational is mapped to a coordinate in a profile of appropriate index via the bin_idx
   * vector. The total number of lines, as on an equivalent uniform grid (i.e. brick->nxyztrees[transverse_direction]*(1<<data->max_lvl)), must be
   * dividible by the number of elements in bin_idx. The total number of velocity profiles may be smaller than the size of bin_idx (but not larger).
   * Illustration for a 2D SHS simulation:
   *   Say that you are interested in z-averaged y-profiles of the x-component of velocity in a SHS simulation with spanwise (i.e. z-aligned) ridges.
   *   Say that the boundary conditions on the bottom and top walls alternate as (for a z-perpendicular cross section)
   *
   *  __                                                               ____________________________________________________________
   *    |                      slip                                    |                      no slip                              |        ... etc.
   *    |______________________________________________________________|                                                           |___________________
   * |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   <-- limits of the finest computational cells
   * 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35   <-- logical coordinates of the x-faces
   * 8   7   6   5   4   3   2   1   0   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  14  13  12  11  10   9   8   7   6   5   4   <-- appropriate profiles indices by mapping periodicity and symmetry equivalences
   * 8   7   6   5   4   3   2   1   0   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  14  13  12  11  10   9                       <-- corresponding (elementary) bin_idx vector
   *                                                                                                                                           \
   *                                                                                                                                            \
   * ____________________________________________________________________________________________________________________________________________\  x axis
   *                                                                                                                                             /
   *                                                                                                                                            /
   *                                                                                                                                           /
   *
   * The profile is calculated as mapped to an equivalent uniform grid of the finest refinement level. The velocity component value associated with
   * a face that is bigger than the finest possible is considered constant on the entire face.
   * If the averaging direction matches the velocity component of interest (possible only in 3D), the associated weighting length is defined as half
   * the sum of the lengths of neighboring quads in the transverse direction.
   * If the averaging direction is perpendicular to velocity component of interest (only possible case, i.e. default, in 2D), the associated weighting
   * length is defined as the length of the considered face in the averaging direction. The velocity component is considered constant in the tranverse
   * direction between the centers of the neighboring quadrants. (A weighing factor 0.5 applies for extremity-points if not equivalent to the considered
   * face)
   * Note 1:            calculations done at the faces (--> check if boundary conditions are correctly enforced even before interpolation).
   * Note 2:            proc r has the correct result after completion for all profile indices p_idx such that (p_idx%mpi_size == r).
   * Note 3:            assumes no interface in the domain, i.e. levelset < 0 everywhere.
   * local complexity:  every processor loops through their local faces only once.
   * communication:     for every profile index p_idx, a non-blocking MPI_Ireduce to proc r=p_dx%mpi_size is performed, so that only proc r holds the
   *                    correct results for that profile after completion.
   * \param vel_component         [in]    : velocity component of interest in the profile, 0 <= vel_component < P4EST_DIM
   * \param axis                  [in]    : axis along which the profile is calculated, 0 <= axis < P4EST_DIM
   * \param averaging_direction   [in]    : direction along which averaging is desired (only in 3D, default to unrepresented perpendicular-to-plane direction in 2D)
   * \param bin_idx               [in]    : elementary vector of mapping to profile indices by periodic repetition, as illustrated here above
   * \param avg_velocity_profile  [out]   : set of vectors containing the values of the desired velocity components (line-averaged) along the profile axis.
   *                                        These vectors are resized (if needed) to contain brick->nxyztrees[tranverse_direction]*(1<<data->max_lvl) elements as if
   *                                        the grid was uniform.
   */
  void get_line_averaged_vnp1_profiles(DIM(const unsigned char& vel_component, const unsigned char& axis, const unsigned char& averaging_direction),
                                       const std::vector<unsigned int>& bin_idx, std::vector<std::vector<double> >& avg_velocity_profile, const double u_scaling = 1.0);

  inline double alpha() const { return (sl_order == 1 ? 1.0 : (2.0*dt_n + dt_nm1)/(dt_n + dt_nm1)); }
  inline double beta()  const { return (sl_order == 1 ? 0.0 : -dt_n/(dt_n + dt_nm1)); }

  inline void activate_timer() { ns_time_step_analyzer.activate(); }
  const std::map<ns_task, execution_time_accumulator>& get_timings() const { return  ns_time_step_analyzer.get_execution_times(); }

  void coupled_problem_partial_destructor();

  const my_p4est_cell_neighbors_t* get_ngbd_c() const { return ngbd_c; }
  const my_p4est_node_neighbors_t* get_ngbd_n() const { return ngbd_n; }
  Vec get_pressure() const { return pressure; }
  Vec const* get_node_velocities_nm1() const  { return vnm1_nodes;  }
  Vec const* get_node_velocities_n() const    { return vn_nodes;    }
  Vec const* get_node_velocities_np1() const  { return vnp1_nodes;  }
  Vec get_phi() const { return phi; }
};



#endif /* MY_P4EST_NAVIER_STOKES_H */
