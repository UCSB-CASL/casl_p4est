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

#ifdef P4_TO_P8
  class wall_bc_value_hodge_t : public CF_3
    #else
  class wall_bc_value_hodge_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    wall_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
  };

#ifdef P4_TO_P8
  class interface_bc_value_hodge_t : public CF_3
    #else
  class interface_bc_value_hodge_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    interface_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
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

  // face interpolator to nodes: store them in memory to accelerate execution if static grid
  bool interpolators_from_face_to_nodes_are_set;
  std::vector<face_interpolator> interpolator_from_face_to_nodes[P4EST_DIM];

  // second_derivatives...[i][j] = second derivatives of velocity component j along Cartesian direction i
  Vec second_derivatives_vnm1_nodes[P4EST_DIM][P4EST_DIM];
  Vec second_derivatives_vn_nodes[P4EST_DIM][P4EST_DIM];

  Vec vorticity;

  Vec pressure;

  Vec smoke;
#ifdef P4_TO_P8
  CF_3 *bc_smoke;
#else
  CF_2 *bc_smoke;
#endif
  bool refine_with_smoke;
  double smoke_thresh;

  int sl_order;

  Vec face_is_well_defined[P4EST_DIM];

#ifdef P4_TO_P8
  BoundaryConditions3D *bc_pressure;
  BoundaryConditions3D bc_hodge;
  BoundaryConditions3D *bc_v;
#else
  BoundaryConditions2D *bc_pressure;
  BoundaryConditions2D bc_hodge;
  BoundaryConditions2D *bc_v;
#endif

  wall_bc_value_hodge_t wall_bc_value_hodge;
  interface_bc_value_hodge_t interface_bc_value_hodge;

#ifdef P4_TO_P8
  CF_3 *external_forces[P4EST_DIM];
#else
  CF_2 *external_forces[P4EST_DIM];
#endif

  my_p4est_interpolation_nodes_t *interp_phi;

  double compute_dxyz_hodge( p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int dir);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx);

  void compute_max_L2_norm_u();

  void compute_vorticity();
  void compute_Q_and_lambda_2_value(Vec& Q_value_nodes, Vec& lambda_2_nodes, const double U_scaling, const double x_scaling) const;

  inline void get_Q_and_lambda_2_values(const quad_neighbor_nodes_of_node_t& qnnn, const double *vnp1_p[P4EST_DIM], const double& x_scaling, const double& U_scaling, double& Qvalue, double& lambda_2_value) const
  {
    double S[P4EST_DIM][P4EST_DIM];
    double omega[P4EST_DIM][P4EST_DIM];
    double S_squared_plus_omega_squared[P4EST_DIM][P4EST_DIM];
    double lambda_coeffs[P4EST_DIM+1];

    S[0][0] = qnnn.dx_central(vnp1_p[0])*x_scaling/U_scaling;                                     omega[0][0] = 0.0;
    S[0][1] = 0.5*(qnnn.dy_central(vnp1_p[0]) + qnnn.dx_central(vnp1_p[1]))*x_scaling/U_scaling;  omega[0][1] = 0.5*(qnnn.dy_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[1]))*x_scaling/U_scaling;
#ifdef P4_TO_P8
    S[0][2] = 0.5*(qnnn.dz_central(vnp1_p[0]) + qnnn.dx_central(vnp1_p[2]))*x_scaling/U_scaling;  omega[0][2] = 0.5*(qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]))*x_scaling/U_scaling;
#endif
    S[1][0] = 0.5*(qnnn.dx_central(vnp1_p[1]) + qnnn.dy_central(vnp1_p[0]))*x_scaling/U_scaling;  omega[1][0] = 0.5*(qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]))*x_scaling/U_scaling;
    S[1][1] = qnnn.dy_central(vnp1_p[1])*x_scaling/U_scaling;                                     omega[1][1] = 0.0;
#ifdef P4_TO_P8
    S[1][2] = 0.5*(qnnn.dz_central(vnp1_p[1]) + qnnn.dy_central(vnp1_p[2]))*x_scaling/U_scaling;  omega[1][2] = 0.5*(qnnn.dz_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[2]))*x_scaling/U_scaling;
    S[2][0] = 0.5*(qnnn.dx_central(vnp1_p[2]) + qnnn.dz_central(vnp1_p[0]))*x_scaling/U_scaling;  omega[2][0] = 0.5*(qnnn.dx_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[0]))*x_scaling/U_scaling;
    S[2][1] = 0.5*(qnnn.dy_central(vnp1_p[2]) + qnnn.dz_central(vnp1_p[1]))*x_scaling/U_scaling;  omega[2][1] = 0.5*(qnnn.dy_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[1]))*x_scaling/U_scaling;
    S[2][2] = qnnn.dz_central(vnp1_p[2])*x_scaling/U_scaling;                                     omega[2][2] = 0.0;
#endif

    Qvalue = 0;
    for (unsigned short ii = 0; ii < P4EST_DIM; ++ii) {
      for (unsigned short jj = 0; jj < P4EST_DIM; ++jj){
        Qvalue += 0.5*(SQR(omega[ii][jj])-SQR(S[ii][jj]));
        S_squared_plus_omega_squared[ii][jj] = 0.0;
        for (unsigned short kk = 0; kk < P4EST_DIM; ++kk){
          S_squared_plus_omega_squared[ii][jj] += S[ii][kk]*S[kk][jj] + omega[ii][kk]*omega[kk][jj];
        }
      }
    }
    lambda_coeffs[0] = 1.0;
#ifdef P4_TO_P8
    lambda_coeffs[1] = -S_squared_plus_omega_squared[0][0]-S_squared_plus_omega_squared[1][1]-S_squared_plus_omega_squared[2][2];
    lambda_coeffs[2] = S_squared_plus_omega_squared[0][0]*S_squared_plus_omega_squared[1][1] + S_squared_plus_omega_squared[0][0]*S_squared_plus_omega_squared[2][2] + S_squared_plus_omega_squared[1][1]*S_squared_plus_omega_squared[2][2] - S_squared_plus_omega_squared[0][1]*S_squared_plus_omega_squared[1][0] - S_squared_plus_omega_squared[0][2]*S_squared_plus_omega_squared[2][0] - S_squared_plus_omega_squared[1][2]*S_squared_plus_omega_squared[2][1];
    lambda_coeffs[3] = SQR(S_squared_plus_omega_squared[0][2])*S_squared_plus_omega_squared[1][1] + SQR(S_squared_plus_omega_squared[1][2])*S_squared_plus_omega_squared[0][0] + SQR(S_squared_plus_omega_squared[0][1])*S_squared_plus_omega_squared[2][2] - 2.0*S_squared_plus_omega_squared[0][1]*S_squared_plus_omega_squared[1][2]*S_squared_plus_omega_squared[0][2] - S_squared_plus_omega_squared[0][0]*S_squared_plus_omega_squared[1][1]*S_squared_plus_omega_squared[2][2];
#else
    lambda_coeffs[1] = -S_squared_plus_omega_squared[0][0]-S_squared_plus_omega_squared[1][1];
    lambda_coeffs[2] = S_squared_plus_omega_squared[0][0]*S_squared_plus_omega_squared[1][1]-S_squared_plus_omega_squared[0][1]*S_squared_plus_omega_squared[1][0];
#endif

#ifdef DEBUG
#ifdef P4_TO_P8
    double discriminant = 18.0*lambda_coeffs[0]*lambda_coeffs[1]*lambda_coeffs[2]*lambda_coeffs[3] - 4.0*pow(lambda_coeffs[1], 3)*lambda_coeffs[3] + SQR(lambda_coeffs[1])*SQR(lambda_coeffs[2]) - 4.0*lambda_coeffs[0]*pow(lambda_coeffs[2], 3) - 27.0*SQR(lambda_coeffs[0])*SQR(lambda_coeffs[3]);
#else
    double discriminant = SQR(lambda_coeffs[1])-4.0*lambda_coeffs[0]*lambda_coeffs[2];
#endif
    P4EST_ASSERT(discriminant >=0.0);
#endif

#ifndef P4_TO_P8
    lambda_2_value = 0.5*(-lambda_coeffs[1] - sqrt(SQR(lambda_coeffs[1]) - 4.0*lambda_coeffs[0]*lambda_coeffs[2]))/lambda_coeffs[0];
#else
    double pp = (3.0*lambda_coeffs[0]*lambda_coeffs[2]-SQR(lambda_coeffs[1]))/(3.0*SQR(lambda_coeffs[0]));
    double qq = (2.0*pow(lambda_coeffs[1], 3) - 9.0*lambda_coeffs[0]*lambda_coeffs[1]*lambda_coeffs[2] + 27.0*SQR(lambda_coeffs[0])*lambda_coeffs[3])/(27.0*pow(lambda_coeffs[0], 3));
    std::vector<double> lambda_values(P4EST_DIM);
    if(pp > 0.0)
      throw std::runtime_error("negative pp");
    for (unsigned short kk = 0; kk < P4EST_DIM; ++kk)
    {
      if(fabs(3.0*qq*sqrt(-3.0/pp)/(2.0*pp)) > 1.0)
        throw std::runtime_error("argument of acos > 1 in absolute value");
      lambda_values[kk] = (lambda_coeffs[1]/(3.0*lambda_coeffs[0])) + 2.0*sqrt(-pp/3.0)*cos(acos(3.0*qq*sqrt(-3.0/pp)/(2.0*pp))/3.0 - 2.0*M_PI*((double) kk)/3.0);
    }
    std::sort(lambda_values.begin(), lambda_values.end());
    lambda_2_value = lambda_values[1];
#endif
  }

  void compute_norm_grad_v();

  bool is_in_domain(const double xyz_[]) const {
    double threshold[P4EST_DIM];
    for (short dd = 0; dd < P4EST_DIM; ++dd)
      threshold[dd] = 0.1*dxyz_min[dd];
    return ((((xyz_[0] - xyz_min[0] > -threshold[0]) && (xyz_[0] - xyz_max[0] < threshold[0])) || is_periodic(p4est_n, dir::x))
        && (((xyz_[1] - xyz_min[1] > -threshold[1]) && (xyz_[1] - xyz_max[1] < threshold[1])) || is_periodic(p4est_n, dir::y))
    #ifdef P4_TO_P8
        && (((xyz_[2] - xyz_min[2] > -threshold[2]) && (xyz_[2] - xyz_max[2] < threshold[2])) || is_periodic(p4est_n, dir::z))
    #endif
        );
  };

  bool is_no_slip(const double xyz_[]) const {
    return ((bc_v[0].wallType(xyz_) == DIRICHLET) && (bc_v[1].wallType(xyz_) == DIRICHLET) &&
    #ifdef P4_TO_P8
        (bc_v[2].wallType(xyz_) == DIRICHLET) &&
    #endif
        bc_pressure->wallType(xyz_) == NEUMANN);
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

#ifdef P4_TO_P8
  void set_smoke(Vec smoke, CF_3 *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);
#else
  void set_smoke(Vec smoke, CF_2 *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);
#endif

  void set_phi(Vec phi);

#ifdef P4_TO_P8
  void set_external_forces(CF_3 **external_forces);
#else
  void set_external_forces(CF_2 **external_forces);
#endif

#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p);
#else
  void set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p);
#endif

  void set_velocities(Vec *vnm1, Vec *vn);

#ifdef P4_TO_P8
  void set_velocities(CF_3 **vnm1, CF_3 **vn);
#else
  void set_velocities(CF_2 **vnm1, CF_2 **vn);
#endif

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

  inline Vec get_vorticity(){return vorticity;}
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

  /*!
   * \brief enforce_mass_flow enforces the mass flow in a desired direction to be equivalent to one of a given mean (bulk) velocity.
   * This function calculates the (constant-in-space) correction to the gradient of the Hodge variable to enforce a desired mass flow.
   * The mass-flow forcing direction MUST be periodic (otherwise the approach is simply inconsistent).
   * This function is ideally called after the projection step and its output should be used to dynamically adapt the driving body
   * force that enforces the constant mass flow rate (possibly with an added convergence criterion within the inner loop).
   * \param force_in_direction          [in]    array of P4EST_DIM flags, forcing is applied in the direction dir if force_in_direction[dir] is true
   * \param desired_mean_velocity       [in]    array of P4EST_DIM doubles, specifying the desired bulk velocity in the forcing direction (the value
   *                                            desired_mean_velocity[dd] is disregarded if force_in_direction[dd] is false)
   * \param forcing_mean_hodge_gradient [out]   array of P4EST_DIM doubles, returning the correction to gradient component of the the
   *                                            Hodge variable to enforce the desired mass flow --> can be used to correct the driving force term
   *                                            afterwards
   * \param mass_flow                   [inout] (optional) array of P4EST_DIM doubles, mass_flow[d] is the mass flow along cartesian direction d
   *                                            before forcing on input, after forcing on input. Only the values for which force_in_direction[d] is
   *                                            true are relevant on input. If NULL, the function calculates the relevant value(s) internally.
   * Raphael EGAN
   */
  void enforce_mass_flow(const bool* force_in_direction, const double* desired_mean_velocity, double* forcing_mean_hodge_gradient, double* mass_flow = NULL);

  void compute_velocity_at_nodes(const bool store_interpolators = false);

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

  void extrapolate_bc_v(my_p4est_node_neighbors_t *ngbd, Vec *v, Vec phi);

#ifdef P4_TO_P8
  bool update_from_tn_to_tnp1(const CF_3 *level_set=NULL, bool keep_grid_as_such=false, bool do_reinitialization=true);
#else
  bool update_from_tn_to_tnp1(const CF_2 *level_set=NULL, bool keep_grid_as_such=false, bool do_reinitialization=true);
#endif

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
#ifdef P4_TO_P8
  void refine_coarsen_grid_after_restart(const CF_3 *level_set, bool do_reinitialization = true);
#else
  void refine_coarsen_grid_after_restart(const CF_2 *level_set, bool do_reinitialization = true);
#endif

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
  void get_slice_averaged_vnp1_profile(const unsigned short& vel_component, const unsigned short& axis, std::vector<double>& avg_velocity_profile, const double u_scaling = 1.0);

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
   * Note 2:            proc r has the correct result after completion for all profile indices p_idx such that (p_idx%mpi_size==r).
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
  void get_line_averaged_vnp1_profiles(const unsigned short& vel_component, const unsigned short& axis,
                                     #ifdef P4_TO_P8
                                       const unsigned short& averaging_direction,
                                     #endif
                                       const std::vector<unsigned int>& bin_idx, std::vector<std::vector<double> >& avg_velocity_profile, const double u_scaling = 1.0);

  inline double alpha() const { return ((sl_order == 1)? (1.0): ((2.0*dt_n+dt_nm1)/(dt_n+dt_nm1)));}

  void coupled_problem_partial_destructor();

};



#endif /* MY_P4EST_NAVIER_STOKES_H */
