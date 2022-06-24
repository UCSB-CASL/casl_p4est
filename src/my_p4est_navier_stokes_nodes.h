#ifndef MY_P4EST_NAVIER_STOKES_NODES_H
#define MY_P4EST_NAVIER_STOKES_NODES_H

#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_macros.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_macros.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif


enum problem_dimensionalization_type_t{
  NONDIM_BY_FLUID_VELOCITY, // nondim by the characteristic fluid velocity
  NONDIM_BY_SCALAR_DIFFUSIVITY, // nondimensionalized by the temperature or concentration fluid diffusivity
  DIMENSIONAL // dimensional problem (highly not recommended)
};

typedef enum {
  grid_update,
  viscous_step,
  projection_step,
  velocity_interpolation
}ns_task;

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


class my_p4est_navier_stokes_nodes_t{

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

  mpi_environment_t* mpi;
  PetscErrorCode ierr;

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

  my_p4est_level_set_t *ls;

  // -----------------------------------------------
  // Level set function(s):
  // -----------------------------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_nm1; // LSF for previous timestep... we must keep this so that hodge fields can be updated correctly in NS process

  // Second derivatives of LSFs
  vec_and_ptr_dim_t phi_dd;

  // -----------------------------------------------
  // Interface geometry:
  // -----------------------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  // Solvers and relevant parameters
  int cube_refinement;
  my_p4est_poisson_nodes_mls_t *solver_v_star;

  // Fields related to the velocity problem:
  vec_and_ptr_t v_n;
  vec_and_ptr_t v_nm1;
  vec_and_ptr_t v_backtrace_n;
  vec_and_ptr_t v_backtrace_nm1;
  vec_and_ptr_t rhs_v;


  // First Derivatives of V
  vec_and_ptr_dim_t v_d;

  // Second Derivatives of V
  vec_and_ptr_dim_t v_dd;

  // Boundary conditions:
  BoundaryConditionsDIM bc_velocity[P4EST_DIM];
  BoundaryConditionsDIM bc_pressure;

  //interfacial_bc_fluid_velocity_t* bc_interface_value_velocity[P4EST_DIM];// <-- this gets declared later bc the type is a nested class within stefan w fluids, but I have included it here for readability
  BoundaryConditionType* bc_interface_type_velocity[P4EST_DIM];

  CF_DIM* bc_wall_value_velocity[P4EST_DIM];
  WallBCDIM* bc_wall_type_velocity[P4EST_DIM];

  CF_DIM* bc_interface_value_pressure;
  BoundaryConditionType bc_interface_type_pressure;

  CF_DIM* bc_wall_value_pressure;
  WallBCDIM* bc_wall_type_pressure;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t pressure_nodes;

  Vec dxyz_hodge_old[P4EST_DIM];

  // User provided forcing terms: (And the means to set it)
  CF_DIM* user_provided_external_forces_NS[P4EST_DIM];
  bool there_is_user_provided_external_force_NS;

  // Other parameters:
  double NS_norm; // for keeping track of NS norm
  double NS_max_allowed;

  double hodge_tolerance;
  double hodge_percentage_of_max_u;

  int hodge_max_it;

  // whether or not to compute pressure for a given tstep
  // (this saves computational time bc we don't need to compute pressure
  // unless we are going to visualize and/or compute forces )
  bool compute_pressure_;

  // ----------------------------------------------
  // Related to domain:
  // ----------------------------------------------
  double xyz_min[P4EST_DIM]; double xyz_max[P4EST_DIM];
  int ntrees[P4EST_DIM];
  int periodicity[P4EST_DIM];

  // Variables for refining the fields
  int lmin, lint, lmax;
  double uniform_band;

  bool use_uniform_band;
  bool refine_by_vorticity;
  double vorticity_threshold;


  // For initializationg from load state:
  bool loading_from_previous_state;

  bool check_if_domain_info_is_set(){
    bool check = true;

    foreach_dimension(d){
      check = check && (xyz_min[d]<DBL_MAX);
      check = check && (xyz_max[d]<DBL_MAX);
      check = check && (periodicity[d]<INT_MAX);
      check = check && (ntrees[d]<INT_MAX);
    }
    return check;
  }
  // ----------------------------------------------
  // Related to interpolation bw grids:
  // ----------------------------------------------
  int num_fields_interp;
  interpolation_method interp_bw_grids;


  // ----------------------------------------------
  // Related to current grid size:
  // ----------------------------------------------
  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;
  double dxyz_close_to_interface_mult; // multiplier set by user on dxyz_close_to_interface

  // ----------------------------------------------
  // Related to time/timestepping:
  // ----------------------------------------------
  double tn;
  double dt;
  double dt_nm1;
  double dt_max_allowed;
  double dt_min_allowed;

  double tstart; // for tracking percentage done // TO-DO: revisit if this is needed
  double tfinal; // used only to clip timestep when we are nearing the end
  double v_interface_max_norm; // for keeping track of max norm of vinterface
  double v_interface_max_allowed; // max allowable value before we trigger a crash state

  int advection_sl_order; // advec order for Navier Stokes problem
  double advection_alpha_coeff;
  double advection_beta_coeff;

  int tstep;
  int load_tstep;
  int last_tstep;


  // ----------------------------------------------
  // Related to dimensionalization type:
  // ----------------------------------------------
  problem_dimensionalization_type_t problem_dimensionalization_type;


  // Converting nondim to dim:
  double time_nondim_to_dim;
  double vel_nondim_to_dim;



  // ----------------------------------------------
  // Nondimensional groups
  // ----------------------------------------------
  double Re; // Reynolds number (rho Uinf l_char)/mu_l

  // ----------------------------------------------
  // Physical parameters:
  // Note: these must be provided for the solver to run !
  // ----------------------------------------------
  double l_char; // Characteristic length scale (assumed in meters)
  double u_inf; // Characteristic velocity scale (assumed in m/s)

  double mu_l; // Fluid viscosity [Pa s]
  double grav; // Gravity

  bool use_boussinesq;

public:
  // -------------------------------------------------------
  // Constructor/Destructor:
  // -------------------------------------------------------

  my_p4est_navier_stokes_nodes_t(mpi_environment_t* mpi_);
  ~my_p4est_navier_stokes_nodes_t();


  // -------------------------------------------------------
  // Auxiliary initialization fxns:
  // -------------------------------------------------------

  /*!
   * \brief initialize_grids:This function initializes the grids p4est_n and p4est_nm1 depending on the
   * domain, periodicity, and grid min level/max level information provided by the user
   *
   * Note: this function is intended for internal use by the fxn perform_initializations
  */
  void initialize_grids();


  /*!
   * \brief initialize_fields:This function initializes the fields phi v_n (navier stokes), and v_nm1 (navier stokes)
   * It does so either by CF_DIM's provided by the user for each of these fields, or by vectors provided by the user (WIP)
   *
   * It then computes an initial interfacial velocity, which is used only to compute an initial timestep for the problem
   *
   * Note: this function is intended for internal use by the fxn perform_initializations
  */
  void initialize_fields();


  // -------------------------------------------------------
  // Functions related to Navier-Stokes problem:
  // -------------------------------------------------------
  void set_ns_parameters();
  void initialize_ns_solver();
  bool navier_stokes_step(); // output is whether or not it crashed, if it crashes we save a vtk crash file
  void setup_and_solve_navier_stokes_problem();



};



#endif
