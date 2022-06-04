#ifndef MY_P4EST_STEFAN_WITH_FLUIDS_H
#define MY_P4EST_STEFAN_WITH_FLUIDS_H


#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>

#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_multialloy.h>
#include <src/my_p4est_macros.h>

#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>

#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif


enum problem_dimensionalization_type_t{
  NONDIM_BY_FLUID_VELOCITY, // nondim by the characteristic fluid velocity
  NONDIM_BY_SCALAR_DIFFUSIVITY, // nondimensionalized by the temperature or concentration fluid diffusivity
  DIMENSIONAL // dimensional problem (highly not recommended)
};

enum domain_phase_t{
  LIQUID_DOMAIN=0, SOLID_DOMAIN=1
};

class my_p4est_stefan_with_fluids_t
{

private:

  // TO-DO: documentation:
  // Write a list of all the parameters/inputs that the user *must* provide for this to work
  // WRite a list of option parameters and how those cases will be handled

  // Misc useful things:
  // ---------------------
  mpi_environment_t* mpi;
  PetscErrorCode ierr;


  // -----------------------------------------------
  // Grid variables
  // -----------------------------------------------
  /* A note on grid variable notation:
   * - The notation for grids is a bit confusing, but this is the way it's been done:
   *
   * - We refer to the grid p4est_np1 as the grid we are
   *   currently solving for the fields at time np1,
   *   a.k.a. the grid which has the *interface location* at time np1
   *
   * - This means that throughout the process, we will have
   * the fields at time n *sampled* on the grid p4est_np1, and
   * the fields at time nm1 *sampled* on the p4est_n
   *
   * - At each timestep we solve for the fields on the grid np1,
   *   then using the interfacial velocity soln v_gamma^(np1),
   *   we advect the LSF Gamma^np1 to find Gamma^np2
   *
   * - Then, we slide the grids so we delete p4est_n, slide p4est_np1 --> p4est_n, and
   *   compute a new p4est_np1 corresponding to Gamma^np2 which will become Gamma^np1
   *   for the next timestep
   *
   */

  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // Grid at time n (usually housing nm1 variables, except for interface location @ n)
  p4est_t*              p4est_n;
  p4est_nodes_t*        nodes_n;
  p4est_ghost_t*        ghost_n;

  my_p4est_hierarchy_t* hierarchy_n;
  my_p4est_node_neighbors_t* ngbd_n;

  // Grid at time np1 (usually housing n variables, except for interface location @ np1)
  p4est_t*               p4est_np1;
  p4est_nodes_t*         nodes_np1;
  p4est_ghost_t*         ghost_np1;
  my_p4est_hierarchy_t* hierarchy_np1;
  my_p4est_node_neighbors_t* ngbd_np1;

  // Splitting criteria
  splitting_criteria_cf_and_uniform_band_t* sp;

  // Level set object
  my_p4est_level_set_t *ls;

  // -----------------------------------------------
  // Level set function(s):
  // -----------------------------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_nm1; // LSF for previous timestep... we must keep this so that hodge fields can be updated correctly in NS process

  // LSF for solid domain (internal use only) This will be assigned as needed as the negative of phi
  vec_and_ptr_t phi_solid;

  // LSF for the inner substrate (if applicable)
  vec_and_ptr_t phi_substrate;

  // Effective LSF used when we have a substrate -- this will be used by the extension, interfacial velocity computation, and navier stokes steps
  vec_and_ptr_t phi_eff;

  // Second derivatives of LSFs
  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_substrate_dd;

  // -----------------------------------------------
  // Interface geometry:
  // -----------------------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;
  vec_and_ptr_dim_t substrate_normals;

  // Island numbers -- for cases where we are tracking evolving grain geometries
  vec_and_ptr_t island_numbers;

  // -----------------------------------------------
  // Temperature/concentration problem:
  // -----------------------------------------------
  // Solvers and relevant parameters
  int cube_refinement;
  my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

  // Fields related to the liquid temperature/concentration problem:
  vec_and_ptr_t T_l_n;
  vec_and_ptr_t T_l_nm1;
  vec_and_ptr_t T_l_backtrace_n;
  vec_and_ptr_t T_l_backtrace_nm1;
  vec_and_ptr_t rhs_Tl;

  // Fields related to the solid temperature/concentration problem:
  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  // First derivatives of T_l_n and T_s_n
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;

  // Second derivatives of T_l
  vec_and_ptr_dim_t T_l_dd;

  // Boundary conditions: // will figure out how to address this later
//interfacial_bc_temp_t* bc_interface_val_temp[2]; // <-- this gets declared later bc the type is a nested class within stefan w fluids, but I have included it here for readability
  BoundaryConditionType* bc_interface_type_temp[2];
  CF_DIM* bc_interface_robin_coeff_temp[2];

  CF_DIM* bc_wall_value_temp[2];
  WallBCDIM* bc_wall_type_temp[2];

  CF_DIM* bc_interface_val_temp_substrate[2];
  BoundaryConditionType* bc_interface_type_temp_substrate[2];
  CF_DIM* bc_interface_robin_coeff_temp_substrate[2];

  // User provided heat source term: (And the means to set it)
  CF_DIM* user_provided_external_heat_source[2];
  bool there_is_user_provided_heat_source;

  // ----------------------------------------------
  // Stefan problem:
  // ----------------------------------------------
  vec_and_ptr_dim_t v_interface;
  vec_and_ptr_dim_t jump;

  // ----------------------------------------------
  // Navier-Stokes problem:
  // ----------------------------------------------

  my_p4est_navier_stokes_t* ns;

  my_p4est_poisson_faces_t* face_solver;
  my_p4est_poisson_cells_t* cell_solver;
  PCType pc_face;
  KSPType face_solver_type;
  PCType pc_cell;
  KSPType cell_solver_type;

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  Vec dxyz_hodge_old[P4EST_DIM];

  my_p4est_cell_neighbors_t *ngbd_c_np1;
  my_p4est_faces_t *faces_np1;

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


  // -----------------------------------------------
  // Fields for assisting in the multicomponent problem 
  // (whose main operations are handled by my_p4est_multialloy):
  // -----------------------------------------------

  // Concentration fields & backtraces
  vec_and_ptr_array_t Cl_n;
  vec_and_ptr_array_t Cl_nm1;

  vec_and_ptr_array_t Cl_backtrace_n;
  vec_and_ptr_array_t Cl_backtrace_nm1;

  //TO-DO MULTICOMP: add a check to make sure new fields have been received from multialloy before doing the backtrace


  // Fields called in multialloy initialize fxn:
  // ----------------------
  // Need to determine which of these we need our own version of, and what these all 
  // correspond to, and when they need to be set for multialloy functionalities
  // to work correctly  
  
  // Geom: 
  // -
  // front_phi_ --> assume this corresponds to the interface, should double check
  // front_phi_dd_
  // front_curvature_ --> "" kappa, ""
  // front_normal_ --> "" normal, "" 
  // contr_phi_ 
  // contr_phi_dd_ 
  
  // Physical fields
  // -
  // tl_ --> our T_l_n and T_l_nm1
  // ts_ --> our T_s_n
  // cl_ --> we have defined above
  // cl0_grad_ --> we have defined above
  // seed_map_
  // dendrite_number_
  // dendrite_tip_
  // bc_error_
  // smoothed_nodes_
  // front_phi_unsmooth_ 

  // Lagrange multipliers
  // - 
  // psi_tl_
  // psi_ts_
  // psi_cl_

  // solid_front_phi
  // solid_front_phi_nm1
  // solid_front_curvature_
  // solid_front_velo_norm_
  // solid_time_
  // solid_tf_ 
  // solid_cl_
  // solid_part_coeff_
  // solid_seed_
  // solid_Smoothed_nodes_



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
  bool refine_by_d2T;
  double vorticity_threshold;
  double d2T_threshold;

  // For initializationg from load state:
  bool loading_from_previous_state;

  // Functions to check initialization status:
  // Used to make sure user has set the domain info before attempting initializations
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
  // TO-DO: need to have a place where interp_bw_grids is defined??

  // ----------------------------------------------
  // Related to current grid size:
  // ----------------------------------------------
  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;
  double dxyz_close_to_interface_mult; // multiplier set by user on dxyz_close_to_interface


  // ----------------------------------------------
  // Variables used for extension of fields:
  // ----------------------------------------------
  double min_volume_;
  double extension_band_use_;
  double extension_band_extend_;

  void compute_extension_bands_and_dxyz_close_to_interface(){
        dxyz_min(p4est_np1, dxyz_smallest);
    min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
    extension_band_use_    = (8.)*pow(min_volume_, 1./ double(P4EST_DIM)); //8
    extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
    dxyz_close_to_interface = dxyz_close_to_interface_mult*MAX(dxyz_smallest[0],dxyz_smallest[1]);
  }

  // ----------------------------------------------
  // Related to time/timestepping:
  // ----------------------------------------------
  double tn;
  double dt;
  double dt_nm1;
  double dt_Stefan;
  double dt_NS;
  double dt_max_allowed;
  double dt_min_allowed;



  double tstart; // for tracking percentage done // TO-DO: revisit if this is needed
  double tfinal; // used only to clip timestep when we are nearing the end
  double v_interface_max_norm; // for keeping track of max norm of vinterface
  double v_interface_max_allowed; // max allowable value before we trigger a crash state



  int advection_sl_order; // advec order for scalar temp/conc problem
  int NS_advection_sl_order; // advec order for Navier Stokes problem
  double advection_alpha_coeff;
  double advection_beta_coeff;

  int tstep;
  int load_tstep;
  int last_tstep;

  double cfl_Stefan;
  double cfl_NS;


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
  double Pr; // Prandtl number - (mu_l/(rho_l * alpha_l)) = (nu_l/alpha_l)
  double Sc; // Schmidt number - (mu_l/(rho_l * D)) = (nu_l/D)
  double Pe; // Peclet number - (Uint lchar)/alpha_l , or Re*Pr
  double St; // Stefan number (cp_s deltaT/L)
  double Da; // Damkohler number (k_diss*l_char/D_diss)
  double RaT; // Rayleigh number by temperature TO-DO: add definition
  double RaC; // Rayleigh number by concentration TO-DO: add definition

  // Note: we will give the user the option to set the above nondimensional groups.
  // If the user specifies the group, the group will be accepted and used to compute other
  // quantities of interest relevant to the nondim problem (i.e. uinf to convert time_nondim to time_dim)

  // Otherwise, these groups will be computed by the solver using the provided physical parameters



  // ----------------------------------------------
  // Physical parameters:
  // Note: these must be provided for the solver to run !
  // ----------------------------------------------
  double l_char; // Characteristic length scale (assumed in meters)
  double u_inf; // Characteristic velocity scale (assumed in m/s)



  double T0; // characteristic solid temperature of the problem
  double Tinterface; // Interface temperature or concentration
  double Tinfty; // Freestream fluid temperature or concentration
  double Tflush; // Flush temperature (K) or concentration that inlet BC is changed to if flush_dim_time is activated




  double alpha_l, alpha_s; // Liquid and solid thermal diffusivities, [m^2/s]
  double k_l, k_s; // Liquid and solid thermal conductivities, [W/(mK)]
  double rho_l, rho_s; // Liquid and solid densities, [kg/m^3]
  double cp_s; // Solid heat capacity [J/(kg K)]
  double L; // Latent heat of fusion [J/kg]
  double mu_l; // Fluid viscosity [Pa s]

  double sigma; // Interfacial tension [m] between the solid and liquid phase, used in solidification contexts

  double grav; // Gravity
  double beta_T; // Thermal expansion coefficient for the boussinesq approx
  double beta_C; // Concentration expansion coefficient for the boussinesq approx

  double gamma_diss; // TO-DO: add description
  double stoich_coeff_diss; // The stoichiometric coefficient of the dissolution reaction
  double molar_volume_diss; // The molar volume of the dissolving solid

  double Dl, Ds; //Concentration diffusion coefficient m^2/s,
  double k_diss; // Dissolution rate constant per unit area of reactive surface (m/s)



  // ----------------------------------------------
  // Booleans related to what kind of physics we are solving
  // ----------------------------------------------
  bool solve_stefan;
  bool solve_navier_stokes;
  bool there_is_a_substrate/*example_uses_inner_LSF*/;
  bool start_w_merged_grains; // performs a front regularization during initialization to merge any geom in initial LSF

  bool do_we_solve_for_Ts;

  bool use_boussinesq;

  bool is_dissolution_case;

  bool force_interfacial_velocity_to_zero;

  bool solve_multicomponent; // This is for the multialloy problem, 
  //                          or any other problem where there is temperature and at least
  //                          one concentration field

  // ----------------------------------------------
  // Other misc parameters
  // ----------------------------------------------
  bool print_checkpoints; // can set this to true to debug where code might be crashing

  double scale_vgamma_by; // Used in coupled convergence test to switch the sign of the interface velocity
  // ----------------------------------------------
  // Specific to diff cases --> may change these now that they are within a class structure
  // ----------------------------------------------
//  bool analytical_IC_BC_forcing_term;
//  bool example_is_a_test_case;

  bool interfacial_temp_bc_requires_curvature;
  bool interfacial_temp_bc_requires_normal;

  bool interfacial_vel_bc_requires_vint;

//  bool example_has_known_max_vint;

  bool track_evolving_geometries;

  // ----------------------------------------------
  // Related to any front regularization:
  // ----------------------------------------------
  bool use_regularize_front;
  bool use_collapse_onto_substrate;

  double proximity_smoothing;
  double proximity_collapse;


  // ----------------------------------------------
  // Related to LSF reinitialization
  // ----------------------------------------------
  int reinit_every_iter;

  // ----------------------------------------------
  // Level set functions
  // ----------------------------------------------
  CF_DIM* level_set;
  CF_DIM* substrate_level_set;

  // ----------------------------------------------
  // Temperature problem variables -- nondim:
  // ----------------------------------------------
  // These things get computed depending on the provided dimensional quantities and case
  double deltaT;
  double theta_infty, theta_interface, theta0;


  // -------------------------------------------------------
  // Auxiliary initializations:
  // -------------------------------------------------------
  bool grids_are_initialized;
  bool fields_are_initialized;

  CF_DIM* initial_refinement_CF; // fxn for initial refinement criteria, can be LSF or something else

  CF_DIM* initial_temp_n[2];
  CF_DIM* initial_temp_nm1[2];
  CF_DIM* initial_NS_velocity_n[P4EST_DIM];
  CF_DIM* initial_NS_velocity_nm1[P4EST_DIM];

  std::vector <CF_DIM*> initial_conc_n;
  std::vector <CF_DIM*> initial_conc_nm1;



public:
  // -------------------------------------------------------
  // Constructor/Destructor:
  // -------------------------------------------------------

  my_p4est_stefan_with_fluids_t(mpi_environment_t* mpi_);
  ~my_p4est_stefan_with_fluids_t();


  // -------------------------------------------------------
  // Auxiliary initialization fxns:
  // -------------------------------------------------------

  /*!
   * \brief initialize_grids:This function initializes the grids p4est_n and p4est_np1 depending on the
   * domain, periodicity, and grid min level/max level information provided by the user
   *
   * Note: this function is intended for internal use by the fxn perform_initializations
  */
  void initialize_grids();
  /*!
   * \brief initialize_fields:This function initializes the fields phi, T_l_n, T_l_nm1, T_s_n, v_n (navier stokes), and v_nm1 (navier stokes)
   * It does so either by CF_DIM's provided by the user for each of these fields, or by vectors provided by the user (WIP)
   *
   * It then computes an initial interfacial velocity, which is used only to compute an initial timestep for the problem
   *
   * Note: this function is intended for internal use by the fxn perform_initializations
  */
  void initialize_fields();

  /*!
  * \brief initialize_fields_multicomponent: This function initializes the fields Cl for i number concentration components
  * It does so either by CF_DIM's provided by the user for each of these fields, or by vectors provided by the user (WIP)
  * This is an internal fxn and is called by initialize_fields in the case when the solve_multicomponenet flag is turned on, indicating that
  * the user wishes to solve the multicomponent problem (alloy or other, anything w temp and at least one concentration field)
  */
  void initialize_fields_multicomponent();

  void initialize_grids_and_fields_from_load_state();
  void perform_initializations();


  // -------------------------------------------------------
  // Functions related to scalar temp/conc problem:
  // -------------------------------------------------------

  void do_backtrace_for_scalar_temp_conc_problem(bool do_multicomponent_fields, int num_conc_fields);
  void setup_rhs_for_scalar_temp_conc_problem();
  void poisson_nodes_step_for_scalar_temp_conc_problem();
  void setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc();

  // -------------------------------------------------------
  // Functions related to the multicomponent problem:
  // -------------------------------------------------------

  void setup_and_solve_multicomponent_problem();


  // -------------------------------------------------------
  // Functions related to computation of the interfacial velocity and timestep:
  // -------------------------------------------------------
  void extend_relevant_fields();
  double interfacial_velocity_expression(double Tl_d, double Ts_d);
  bool compute_interfacial_velocity();
  bool compute_interfacial_velocity_stefan();
  bool compute_interfacial_velocity_multicomponent();
  void compute_timestep();

  // -------------------------------------------------------
  // Functions related to Navier-Stokes problem:
  // -------------------------------------------------------
  void set_ns_parameters();
  void initialize_ns_solver();
  bool navier_stokes_step(); // output is whether or not it crashed, if it crashes we save a vtk crash file
  void setup_and_solve_navier_stokes_problem();

  // -------------------------------------------------------
  // Functions related to LSF advection/grid update:
  // -------------------------------------------------------
  void prepare_refinement_fields();
  void perform_reinitialization();
  void refine_and_coarsen_grid_and_advect_lsf_if_applicable();
  void update_the_grid();
  void interpolate_fields_onto_new_grid();
  void perform_lsf_advection_grid_update_and_interp_of_fields();


  // -------------------------------------------------------
  // Functions related to compound LSF, LSF regularization and tracking geometries:
  // -------------------------------------------------------
  void create_and_compute_phi_sub_and_phi_eff();
  void track_evolving_geometry();

  void regularize_front();
  void check_collapse_on_substrate();

  // -------------------------------------------------------
  // Function(s) that solve one timestep of the desired coupled problem:
  // -------------------------------------------------------
  void solve_all_fields_for_one_timestep();

  // -------------------------------------------------------
  // Functions related to VTK saving:
  // -------------------------------------------------------
  void save_fields_to_vtk(int out_idx, bool is_crash=false, char crash_type[]=NULL);

  // -------------------------------------------------------
  // Functions and variables related to save state/load state:
  // -------------------------------------------------------

  void fill_or_load_double_parameters(save_or_load flag, PetscInt num, PetscReal *data);
  void fill_or_load_integer_parameters(save_or_load flag, PetscInt num, PetscInt *data);
  void save_or_load_parameters(const char* filename, save_or_load flag);
  void prepare_fields_for_save_or_load(vector<save_or_load_element_t> &fields_to_save_np1,
                                       vector<save_or_load_element_t> &fields_to_save_n);
  void save_state(const char* path_to_directory, unsigned int n_saved);
  void load_state(const char* path_to_folder);

  // -------------------------------------------------------
  // Classes and/or options for handling coupled boundary conditions:
  // -------------------------------------------------------
  // Interfacial bc value for temp/concentration:
  // -------------------------
  /* This allows the user to inherit this class in the main and set the
   * BC values they want by overloading the operator() function
   *
   * The user may either have the operator() return the Gibbs Thomson function
   * as already developed here in the class .cpp file, or they can
   * define an operator themselves, making use of curvature and normals as desired
   *
  */

  class interfacial_bc_temp_t: public CF_DIM{
    private:
      my_p4est_stefan_with_fluids_t* owner;

      my_p4est_node_neighbors_t* ngbd_bc_temp;

      bool do_we_use_curvature;
      bool do_we_use_normals;
    protected:
      // Curvature interp:
      my_p4est_interpolation_nodes_t* kappa_interp;

      // Normals interp:
      my_p4est_interpolation_nodes_t* nx_interp;
      my_p4est_interpolation_nodes_t* ny_interp;
      // TO-DO: add 3d case

    public:
      interfacial_bc_temp_t(my_p4est_stefan_with_fluids_t* parent_solver, bool do_we_use_curvature_, bool do_we_use_normals_) :
          owner(parent_solver), do_we_use_curvature(do_we_use_curvature_), do_we_use_normals(do_we_use_normals_){

          // Set the appropriate flags in the owning class to apply the BC's we want:
          owner->interfacial_temp_bc_requires_curvature = do_we_use_curvature;
          owner->interfacial_temp_bc_requires_normal = do_we_use_normals;

      }

      void set_kappa_interp(my_p4est_node_neighbors_t* ngbd_, Vec &kappa){
        ngbd_bc_temp = ngbd_;
        kappa_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
        kappa_interp->set_input(kappa, linear);
      }
      void clear_kappa_interp(){
        kappa_interp->clear();
        delete kappa_interp;
      }
      void set_normals_interp(my_p4est_node_neighbors_t* ngbd_, Vec &nx, Vec &ny){
        ngbd_bc_temp = ngbd_;
        nx_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
        nx_interp->set_input(nx, linear);

        ny_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
        ny_interp->set_input(ny, linear);
      }
      void clear_normals_interp(){
        nx_interp->clear();
        delete nx_interp;

        ny_interp->clear();
        delete ny_interp;
      }
      double Gibbs_Thomson(DIM(double x, double y, double z)) const;
      virtual double operator()(DIM(double x, double y, double z)) const {
        throw std::runtime_error("my_p4est_stefan_with_fluids_t::interfacial_bc_temp_t::operator() -- to properly use this BC, the user needs to define an overloaded definition of the operator. \n You may either return Gibbs_Thomsom(DIM(x,y,z)) in which the solver will use the standard Gibbs Thomson condition, or you need to define a different user defined function which may or may not make use of curvature and normals. \n");

      }
  }; // end of nested class interfacial_bc_temp_t

  // Declaration of the bc associated with this:
  interfacial_bc_temp_t* bc_interface_val_temp[2];

  // -------------------------
  // Interfacial bc value for fluid velocity
  // -------------------------

  class interfacial_bc_fluid_velocity_t: public CF_DIM{
private:
    my_p4est_stefan_with_fluids_t* owner;

    my_p4est_node_neighbors_t* ngbd_bc_vNS;
    my_p4est_interpolation_nodes_t* v_interface_interp;
    bool do_we_use_v_interface; // Note that here v_interface refers to the velocity of the moving interface

public:
    // Constructor:
    interfacial_bc_fluid_velocity_t(my_p4est_stefan_with_fluids_t* parent_solver, bool do_we_use_vgamma_for_bc):
        owner(parent_solver), do_we_use_v_interface(do_we_use_vgamma_for_bc) {
      // Set the appropriate flags in the owning stefan_w_fluids class to apply the BCs we want:
      owner->interfacial_vel_bc_requires_vint = do_we_use_v_interface;
    }
    // Functions to set/clear:
    void set(my_p4est_node_neighbors_t* ngbd_, Vec v_interface){
      v_interface_interp = new my_p4est_interpolation_nodes_t(ngbd_);
      v_interface_interp->set_input(v_interface, linear);
    }
    void clear(){
      v_interface_interp->clear();
      delete v_interface_interp;
    }

    // Functions for diff options: (in cpp)
    double Conservation_of_Mass(DIM(double x, double y, double z)) const;
    double Strict_No_Slip(DIM(double x, double y, double z)) const;

    // Operator:
    virtual double operator()(DIM(double x, double y, double z)) const{
      throw std::runtime_error("my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t::operator() -- to properly use this BC, the user needs to define an overloaded definition of the operator. \n "
                               "You may either return Conservation_of_Mass(DIM(x,y,z)) which will enforce the cons of mass condition, Strict_No_Slip(DIM(x, y, z)) which will enforce a component-wise equality of vgamma and vNS, or you need to define a different user-defined function which may or may not make use of the interfacial velocity (vel of moving interface). i.e. you could just return 0. for a homogeneous no slip sort of deal. Or if you're using a Neumann condition for some reason, then you will definitely want to provide your own fxn. \n");
    }
  }; // end of nested class interfacial_bc_temp_t

  // Declaration of the bc associated with this:
  interfacial_bc_fluid_velocity_t* bc_interface_value_velocity[P4EST_DIM];





  // -----------------------------------------------------------------------
  // Functions for setting / getting!
  // -----------------------------------------------------------------------
  // -----------------------------------------------
  // Grid variables
  // -----------------------------------------------

  p4est_nodes* get_nodes_np1(){return nodes_np1;}
  my_p4est_node_neighbors_t* get_ngbd_np1(){return ngbd_np1;}
  // (WIP)

  my_p4est_cell_neigbors_t* get_ngbd_c_np1(){
    return ngbd_c_np1;
  }

  my_p4est_faces_t* get_faces_np1(){
    return faces_np1;
  }

  void set_p4est_np1(p4est_t* p4est_np1_){
    p4est_np1 = p4est_np1_;
  }
  void set_nodes_np1(p4est_nodes_t* nodes_np1_){
    nodes_np1 = nodes_np1_;
  }
  void set_ngbd_np1(my_p4est_node_neigbors_t* ngbd_np1_){
    ngbd_np1 = ngbd_np1_;
  }

  void set_p4est_n(p4est_t* p4est_n_){
      p4est_n = p4est_n_;
  }
  void set_nodes_n(p4est_nodes_t* nodes_n_){
      nodes_n = nodes_n_;
  }
  void set_ngbd_n(my_p4est_node_neighbors_t* ngbd_n_){
      ngbd_n = ngbd_n_;
  }

  // -----------------------------------------------
  // Level set function(s):
  // -----------------------------------------------

  // (WIP)
  vec_and_ptr_t get_phi(){return phi;}

  // -----------------------------------------------
  // Interface geometry:
  // -----------------------------------------------

  // (WIP)

  // -----------------------------------------------
  // Temperature/concentration problem:
  // -----------------------------------------------
  vec_and_ptr_t get_T_l_n(){return T_l_n;}
  vec_and_ptr_t get_T_s_n(){return T_s_n;}

  // TO-DO MULTICOMP: verify that the below does indeed work the way I think it works?
  void set_T_l_n(vec_and_ptr_t &T_l_n_){
    T_l_n = T_l_n_;
  }
  void set_T_l_nm1(vec_and_ptr_t &T_l_nm1_){
    T_l_nm1 = T_l_nm1_;
  }

  void set_T_l_backtrace_n(vec_and_ptr_t &T_l_backtrace_n_){
    T_l_backtrace_n = T_l_backtrace_n_;
  }
  void set_T_l_backtrace_nm1(vec_and_ptr_t &T_l_backtrace_nm1_){
    T_l_backtrace_nm1 = T_l_backtrace_nm1_;
  }

  // ------
  // Interface:
  // ------
  void set_bc_interface_value_temp(interfacial_bc_temp_t* bc_interface_value_temp_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_val_temp[i] = bc_interface_value_temp_[i];
    }
  }

  void set_bc_interface_type_temp(BoundaryConditionType* bc_interface_type_temp_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_type_temp[i] = bc_interface_type_temp_[i];
    }
  }
  void set_bc_interface_robin_coeff_temp(CF_DIM* bc_interface_robin_coeff_temp_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_robin_coeff_temp[i] = bc_interface_robin_coeff_temp_[i];
    }
  }
  // ------
  // Wall:
  // ------
  void set_bc_wall_value_temp(CF_DIM* bc_wall_value_temp_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_wall_value_temp[i] = bc_wall_value_temp_[i];
    }
  }

  void set_bc_wall_type_temp(WallBCDIM* bc_wall_type_temp_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_wall_type_temp[i] = bc_wall_type_temp_[i];
    }
  }
  // ------
  // Substrate interface:
  // ------
  void set_bc_interface_value_temp_substrate(CF_DIM* bc_interface_value_temp_substrate_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_val_temp_substrate[i] = bc_interface_value_temp_substrate_[i];
    }
  }

  void set_bc_interface_type_temp_substrate(BoundaryConditionType* bc_interface_type_temp_substrate_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_type_temp_substrate[i] = bc_interface_type_temp_substrate_[i];
    }
  }
  void set_bc_interface_robin_coeff_temp_substrate(CF_DIM* bc_interface_robin_coeff_temp_substrate_[2]){
    for(unsigned char i=0; i<2; i++){
      bc_interface_robin_coeff_temp_substrate[i] = bc_interface_robin_coeff_temp_substrate_[i];
    }
  }

  // ------
  // External heat sources:
  // ------
  void set_user_provided_external_heat_source(CF_DIM* user_heat_source_[2]){
    for (unsigned int i=0; i<2; i++){
      user_provided_external_heat_source[i] = user_heat_source_[i];
    }
    there_is_user_provided_heat_source = true;
  }


  // ------
  // Setting concentration-related fields used by multialloy:
  // (You should not need fxns for getting these since they won't be destroyed/recreated, 
  // so the address of any modified provided vec_and_ptr will be the same as original)
  // ------
  void set_Cl_n(vec_and_ptr_array_t &Cl_n_){
    Cl_n = Cl_n_;
  }
  void set_Cl_nm1(vec_and_ptr_array_t &Cl_nm1_){
    Cl_nm1 = Cl_nm1_;
  }
  void set_Cl_backtrace_n(vec_and_ptr_array_t &Cl_backtrace_n_){
    Cl_backtrace_n = Cl_backtrace_n_;
  }
  void set_Cl_backtrace_nm1(vec_and_ptr_array_t &Cl_backtrace_nm1_){
    Cl_backtrace_nm1 = Cl_backtrace_nm1_;
  }

  // ----------------------------------------------
  // Stefan problem:
  // ----------------------------------------------
  vec_and_ptr_dim_t get_v_interface(){return v_interface;}

  void set_v_interface(vec_and_ptr_dim_t v_interface_){
    v_interface = v_interface_;
  }
  // ----------------------------------------------
  // Navier-Stokes problem:
  // ----------------------------------------------
  my_p4est_navier_stokes_t* get_ns_solver(){return ns;}
  vec_and_ptr_dim_t get_v_n(){return v_n;}
  vec_and_ptr_t get_vorticity(){return vorticity;}
  vec_and_ptr_t get_press_nodes(){return press_nodes;}

  // ------
  // Interface velocity:
  // ------
  void set_bc_interface_value_velocity(interfacial_bc_fluid_velocity_t* bc_interface_value_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_interface_value_velocity[d] = bc_interface_value_velocity_[d];
    }
  }

  void set_bc_interface_type_velocity(BoundaryConditionType* bc_interface_type_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_interface_type_velocity[d] = bc_interface_type_velocity_[d];
    }
  }

  // ------
  // Wall velocity:
  // ------
  void set_bc_wall_value_velocity(CF_DIM* bc_wall_value_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_wall_value_velocity[d] = bc_wall_value_velocity_[d];
    }
  }

  void set_bc_wall_type_velocity(WallBCDIM* bc_wall_type_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_wall_type_velocity[d] = bc_wall_type_velocity_[d];
    }
  }

  // ------
  // Interface pressure:
  // ------
  void set_bc_interface_value_pressure(CF_DIM* bc_interface_value_pressure_){
    bc_interface_value_pressure = bc_interface_value_pressure_;
  }
  void set_bc_interface_type_pressure(BoundaryConditionType bc_interface_type_pressure_){
    bc_interface_type_pressure = bc_interface_type_pressure_;
  }

  // ------
  // Wall pressure:
  // ------
  void set_bc_wall_value_pressure(CF_DIM* bc_wall_value_pressure_){
    bc_wall_value_pressure = bc_wall_value_pressure_;
  }
  void set_bc_wall_type_pressure(WallBCDIM* bc_wall_type_pressure_){
    bc_wall_type_pressure = bc_wall_type_pressure_;
  }

  // ------
  // User provided force terms
  // ------
  void set_user_provided_external_force_NS(CF_DIM* user_external_forces_NS_[2]){
    foreach_dimension(d){
      user_provided_external_forces_NS[d] = user_external_forces_NS_[d];
    }
    there_is_user_provided_external_force_NS = true;
  }
  // ------
  // Other parameters
  // ------
  void set_NS_max_allowed(double NS_norm_max_allowed_){NS_max_allowed = NS_norm_max_allowed_;}
  void set_hodge_percentage_of_max_u(double max_perc){
    hodge_percentage_of_max_u = max_perc;
  }
  void set_hodge_max_iteration(int max_it){hodge_max_it = max_it;}

  void set_compute_pressure(bool compute_press){compute_pressure_ = compute_press;}
  // ----------------------------------------------
  // Related to domain:
  // ----------------------------------------------
  void set_xyz_min(double xyz_min_[P4EST_DIM]){
    foreach_dimension(d){
      xyz_min[d] = xyz_min_[d];
    }
  }
  void set_xyz_max(double xyz_max_[P4EST_DIM]){
    foreach_dimension(d){
      xyz_max[d] = xyz_max_[d];
    }
  }

  void set_ntrees(int ntrees_[P4EST_DIM]){
    foreach_dimension(d){
      ntrees[d] = ntrees_[d];
    }
  }

  void set_periodicity(int periodicity_[P4EST_DIM]){
    foreach_dimension(d){
      periodicity[d] = periodicity_[d];
    }
  }

  void set_lmin_lint_lmax(int lmin_, int lint_, int lmax_){
    lmin = lmin_; lint = lint_; lmax= lmax_;
  }
  int get_lmin(){return lmin;}
  int get_lint(){return lint;}
  int get_lmax(){return lmax;}

  // If you set a uniform band, we assume you use it
  void set_uniform_band(double uniform_band_){
    use_uniform_band = true;
    uniform_band = uniform_band_;}

  // NOTE: ref by vorticity is assumed true whenever navier stokes is solved
  void set_refine_by_d2T(bool ref_by_d2T_){refine_by_d2T = ref_by_d2T_;}

  void set_vorticity_ref_threshold(double vort_thresh){
    vorticity_threshold = vort_thresh;
  }
  void set_d2T_ref_threshold(double d2T_thresh){d2T_threshold = d2T_thresh;}
  void set_loading_from_previous_state(bool loading_from_prev_state_){
    loading_from_previous_state = loading_from_prev_state_;
  }

  // ----------------------------------------------
  // Related to current grid size:
  // ----------------------------------------------
  void set_dxyz_close_to_interface_mult(double dxyz_close_to_interface_mult_){
    dxyz_close_to_interface_mult = dxyz_close_to_interface_mult_;
  }

  double get_dxyz_close_to_interface(){return dxyz_close_to_interface;}
  double get_dxyz_smallest(){
      return MIN(DIM(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]));
  }

  // ----------------------------------------------
  // Related to time/timestepping:
  // ----------------------------------------------
  void set_tn(double tn_){tn = tn_;}
  double get_tn(){return tn;}

  void set_dt(double dt_){dt = dt_;}
  double get_dt(){return dt;}

  void set_dt_nm1(double dt_nm1_){dt_nm1 = dt_nm1_;}
  void set_dt_max_allowed(double dt_max_allowed_){dt_max_allowed = dt_max_allowed_;}
  void set_dt_min_allowed(double dt_min_allowed_){dt_min_allowed = dt_min_allowed_;}

  void set_tfinal(double tf_){tfinal = tf_;}

  void set_tstep(int tstep_){tstep = tstep_;}
  int get_tstep(){return tstep;}

  int get_last_tstep(){return last_tstep;}

  void set_v_interface_max_allowed(double vint_max_all_){v_interface_max_allowed = vint_max_all_;}

  void set_cfl_Stefan(double cfl_stefan_){cfl_Stefan = cfl_stefan_;}
  void set_cfl_NS(double cfl_ns_){cfl_NS = cfl_ns_;}

  // ----------------------------------------------
  // Related to dimensionalization type:
  // ----------------------------------------------
  void set_problem_dimensionalization_type(problem_dimensionalization_type_t prob_dim_type_){
    problem_dimensionalization_type = prob_dim_type_;
  }
  double get_time_nondim_to_dim(){
    return time_nondim_to_dim;
  }
  double get_vel_nondim_to_dim(){
    return vel_nondim_to_dim;
  }

  void set_Re(double Re_){Re = Re_;}
  void set_Pr(double Pr_){Pr = Pr_;}
  void set_Sc(double Sc_){Sc = Sc_;}
  void set_Pe(double Pe_){Pe = Pe_;}
  void set_St(double St_){St = St_;}
  void set_Da(double Da_){Da = Da_;}
  void set_RaT(double RaT_){RaT = RaT_;}
  void set_RaC(double RaC_){RaC = RaC_;}

  void set_nondimensional_groups(){
    // Setup the temperature stuff properly first:
    //    set_temp_conc_nondim_defns(); -- IN CLASS SETTING WE ASSUME THE USER SETS THESE

    // Compute the stuff that doesn't depend on velocity nondim:
    if(Pr<0.) Pr = mu_l/(alpha_l * rho_l);

    if(Sc<0.) Sc = mu_l/(Dl*rho_l);

    if(St<0.) St = cp_s * deltaT/L;


    if(is_dissolution_case){
      // if deltaT is set to zero, we are using the C/Cinf nondim and set gamma_diss = Vm * Cinf/stoic_coeff. otherwise, we are using (C-Cinf)/(C0 - Cinf) = (C-Cinf)/(deltaC) nondim and set gamma_diss = Vm * deltaC/stoic_coeff
      // -- use deltaT/Tinfty to make it of order 1 since concentrations can be quite small depending on units
      if(fabs(deltaT/Tinfty) < EPS){
        gamma_diss = molar_volume_diss*Tinfty/stoich_coeff_diss;
      }
      else{
        gamma_diss = molar_volume_diss*deltaT/stoich_coeff_diss;
      }
      if(Da<0.) Da = k_diss*l_char/Dl;
    }

    // Elyce To-do 12/14/21: add Rayleigh number computations if you are solving boussinesq
    switch(problem_dimensionalization_type){
    case NONDIM_BY_FLUID_VELOCITY:{
      // In this case, we assume a prescribed:
      // (1) free-stream Reynolds number, (2) characteristic length scale, (3) characteristic temperature/concentrations
      // From these, we compute a characteristic velocity, Peclet number, Stefan number, etc.
      // This is also then used to specify the time_nondim_to_dim and vel_nondim_to_dim conversions
      u_inf = (Re*mu_l)/(rho_l * l_char);

      if(!is_dissolution_case){
        Pe = l_char * u_inf/alpha_l;
      }
      else{
        Pe = l_char * u_inf/Dl;
      }

      vel_nondim_to_dim = u_inf;
      time_nondim_to_dim = l_char/u_inf;

      break;
    }
    case NONDIM_BY_SCALAR_DIFFUSIVITY:{
      double u_char = (is_dissolution_case? (Dl/l_char):(alpha_l/l_char));
      vel_nondim_to_dim = u_char;
      time_nondim_to_dim = l_char/u_char;

      if(!is_dissolution_case){
        // Elyce to-do: this is a work in progress
        if(RaC<0.) RaC = beta_C * grav * deltaT * pow(l_char, 3.)/(Dl * (mu_l/rho_l)) ; // note that here deltaT actually corresponds to a change in concentration
        // T variable in this code refers to either temp or conc
      }
      else{
        if(RaT<0.) RaT = beta_T * grav * deltaT * pow(l_char, 3.)/(alpha_l * (mu_l/rho_l)) ;
      }
      break;
    }
    case DIMENSIONAL:{
      vel_nondim_to_dim = 1.0;
      time_nondim_to_dim = 1.0;
    }
    break;
    default:{
      throw std::runtime_error("set_nondimensional_groups: unrecognized nondim formulation in switch case \n");
      break;
    }
    } // end of switch case
  } // end of function

  double get_Re(){return Re;}
  double get_Pr(){return Pr;}
  double get_Sc(){return Sc;}
  double get_Pe(){return Pe;}
  double get_St(){return St;}
  double get_Da(){return Da;}
  double get_RaT(){return RaT;}
  double get_RaC(){return RaC;}


  // ----------------------------------------------
  // Physical parameters:
  // Note: these must be provided for the solver to run !
  // ----------------------------------------------
  void set_l_char(double l_char_){l_char = l_char_;}
  void set_u_inf(double u_inf_){u_inf = u_inf_;}

  void set_dim_temp_conc_variables(double Tinfty_, double Tinterface_, double T0_){
    Tinfty = Tinfty_;
    Tinterface = Tinterface_;
    T0 = T0_;
  };

  void set_Tflush(double Tflush_){Tflush = Tflush_;}

  void set_alpha_l(double alpha_l_){alpha_l = alpha_l_;}
  void set_alpha_s(double alpha_s_){alpha_s = alpha_s_;}
  void set_k_l(double k_l_){k_l = k_l_;}
  void set_k_s(double k_s_){k_s = k_s_;}
  void set_rho_l(double rho_l_){rho_l = rho_l_;}
  void set_rho_s(double rho_s_){rho_s = rho_s_;}
  void set_cp_s(double cp_s_){cp_s = cp_s_;}
  void set_L(double L_){L = L_;}
  void set_mu_l(double mu_l_){mu_l = mu_l_;}
  void set_sigma(double sigma_){sigma = sigma_;}
  void set_grav(double grav_){grav = grav_;}
  void set_beta_T(double beta_T_){beta_T = beta_T_;}
  void set_beta_C(double beta_C_){beta_C = beta_C_;}
  void set_gamma_diss(double gamma_diss_){gamma_diss = gamma_diss_;}
  void set_stoich_coeff_diss(double stoich_coeff_diss_){stoich_coeff_diss = stoich_coeff_diss_;}
  void set_molar_volume_diss(double molar_volume_diss_){molar_volume_diss = molar_volume_diss_;}
  void set_Dl(double Dl_){Dl = Dl_;}
  void set_Ds(double Ds_){Ds = Ds_;}
  void set_k_diss(double k_diss_){k_diss = k_diss_;}

  // Let the user check that they set what they think they set:
  void print_physical_parameters(){
    PetscPrintf(mpi->comm(), "alpha_l = %e, alpha_s= %e \n"
                             "k_l = %e, k_s = %e \n"
                             "rho_l = %e, rho_s = %e \n"
                             "cp_s = %e \n"
                             "L = %e \n"
                             "mu_l = %e \n"
                             "sigma = %e \n"
                             "grav = %e \n"
                             "betaT = %e, betaC = %e \n"
                             "gamma_diss = %e, stoich_coeff_diss = %e, \n "
                             "molar_volume_diss = %e, k_diss = %e \n"
                             "Dl = %e, Ds = %e \n",
                alpha_l, alpha_s,
                k_l, k_s,
                rho_l, rho_s,
                cp_s,
                L,
                mu_l,
                sigma,
                grav,
                beta_T, beta_C,
                gamma_diss, stoich_coeff_diss,
                molar_volume_diss, k_diss,
                Dl, Ds);
  }

  //TO-DO: look into if some of these fxns should be declared as inline
  // ----------------------------------------------
  // Booleans related to what kind of physics we are solving
  // ----------------------------------------------
  void set_solve_stefan(bool solve_){solve_stefan = solve_;}
  void set_solve_navier_stokes(bool solve_){solve_navier_stokes = solve_;}
  void set_do_we_solve_for_Ts(bool solve_){do_we_solve_for_Ts = solve_;}
  void set_use_boussinesq(bool do_we_use){use_boussinesq = do_we_use;}
  void set_is_dissolution_case(bool is_disso){is_dissolution_case =  is_disso;}
  void set_force_interfacial_velocity_to_zero(bool force_){force_interfacial_velocity_to_zero = force_;}

  void set_there_is_substrate(bool there_is_sub ){there_is_a_substrate = there_is_sub;}

  void set_start_w_merged_grains(bool start_w_merged){start_w_merged_grains  = start_w_merged;}


  void set_solve_multicomponent(bool solve_multi_){solve_multicomponent = solve_multi_;}
  void set_number_components(int num_comp){num_conc_fields = num_comp;}

  // ----------------------------------------------
  // Other misc parameters
  // ----------------------------------------------
  void set_print_checkpoints(bool print_){print_checkpoints = print_;}
  void set_scale_vgamma_by(double scale_vgamma_by_){scale_vgamma_by = scale_vgamma_by_;}

  // ----------------------------------------------
  // Specific to diff cases
  // ----------------------------------------------
  void set_track_evolving_geometries(bool track_evo_geom_){track_evolving_geometries = track_evo_geom_;}

  // ----------------------------------------------
  // Related to any front regularization:
  // ----------------------------------------------
  void set_use_regularize_front(bool use_reg_front_ ){ use_regularize_front = use_reg_front_;}
  void set_use_collapse_onto_substrate(bool use_collapse_onto_sub_){ use_collapse_onto_substrate = use_collapse_onto_sub_;}
  void set_proximity_smoothing(double prox_smoothing_){
    proximity_smoothing = prox_smoothing_;
  }
  void set_proximity_collapse(double prox_collapse_){
    proximity_collapse = prox_collapse_;
  }

  // ----------------------------------------------
  // Related to LSF reinitialization
  // ----------------------------------------------
  void set_reinit_every_iter(int reinit_every_iter_){reinit_every_iter = reinit_every_iter_;}

  // ----------------------------------------------
  // Level set functions
  // ----------------------------------------------
  void set_LSF_CF(CF_DIM* lsf_CF){level_set = lsf_CF;}
  void set_substrate_LSF_CF(CF_DIM* sub_lsf_CF){substrate_level_set = sub_lsf_CF;}

  // ----------------------------------------------
  // Temperature problem variables -- nondim:
  // ----------------------------------------------
  void set_nondim_temp_conc_variables(double theta_infty_, double theta_interface_,
                                      double theta0_, double deltaT_){
    theta_infty = theta_infty_;
    theta_interface = theta_interface_;
    theta0 = theta0_;
    deltaT = deltaT_;
  };

  // -------------------------------------------------------
  // Auxiliary initializations:
  // -------------------------------------------------------
  void set_initial_refinement_CF(CF_DIM* initial_refinement_CF_){
    initial_refinement_CF = initial_refinement_CF_;
  }

  void set_initial_temp_n(CF_DIM* init_temp_n_[2]){
    for(unsigned char i=0; i<2; i++){
      initial_temp_n[i] = init_temp_n_[i];
    }
  }

  void set_initial_temp_nm1(CF_DIM* init_temp_nm1_[2]){
    for(unsigned char i=0; i<2; i++){
      initial_temp_nm1[i] = init_temp_nm1_[i];
    }
  }

  void set_initial_NS_velocity_n_(CF_DIM* init_vel_n_[P4EST_DIM]){
    foreach_dimension(d){
      initial_NS_velocity_n[d] = init_vel_n_[d];
    }
  }

  void set_initial_NS_velocity_nm1_(CF_DIM* init_vel_nm1_[P4EST_DIM]){
    foreach_dimension(d){
      initial_NS_velocity_nm1[d] = init_vel_nm1_[d];
    }
  }

  void set_initial_conc_n(std::vector<CF_DIM*> init_conc_n_){
    int N = init_conc_n_.size();
    initial_conc_n.resize(N);

    for(int i = 0; i<N; i++){
      initial_conc_n[i] = init_conc_n_[i];
    }    
  }

  void set_initial_conc_nm1(std::vector<CF_DIM*> init_conc_nm1_){
    int N = init_conc_nm1_.size();
    initial_conc_nm1.resize(N);

    for(int i = 0; i<N; i++){
      initial_conc_nm1[i] = init_conc_nm1_[i];
    }    
  }

};

#endif // STEFAN_WITH_FLUIDS_H
