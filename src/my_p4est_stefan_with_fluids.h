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
  mpi_environment_t mpi;
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

  // Grid at time n (usually housing nm1 variables, except for interface location @ n)
  p4est_t*              p4est_n;
  p4est_nodes_t*        nodes_n;
  p4est_ghost_t*        ghost_n;
  p4est_connectivity_t* conn_n;
  my_p4est_brick_t      brick_n;
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
  int cube_refinement = 1;
  my_p4est_poisson_nodes_mls_t *solver_Tl = NULL;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts = NULL;  // will solve poisson problem for Temperature in solid domain

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
  // perhaps have a fxn--> set bc_interface_val to Gibbs Thomson, or to dendrite, or to user defined?
  CF_DIM* bc_interface_val_temp[2];
  BoundaryConditionType bc_interface_type_temp;

  CF_DIM* bc_wall_value_temp[2];
  BoundaryConditionType bc_wall_type_temp;

  CF_DIM* bc_interface_val_temp_substrate;
  BoundaryConditionType bc_interface_type_temp_substrate;

  // User provided heat source term: (And the means to set it)
  CF_DIM* user_provided_external_heat_source[2];
  bool there_is_user_provided_heat_source = false;

  void set_user_provided_external_heat_source(CF_DIM* user_heat_source_[2]){
    for (unsigned int i=0; i<2; i++){
      user_provided_external_heat_source[i] = user_heat_source_[i];
    }
    there_is_user_provided_heat_source = true;
  }
  // ----------------------------------------------
  // Stefan problem:
  // ----------------------------------------------
  vec_and_ptr_dim_t v_interface;;
  vec_and_ptr_dim_t jump;

  // ----------------------------------------------
  // Navier-Stokes problem:
  // ----------------------------------------------

  my_p4est_navier_stokes_t* ns = NULL;

  PCType pc_face = PCSOR;
  KSPType face_solver_type = KSPBCGS;
  PCType pc_cell = PCSOR;
  KSPType cell_solver_type = KSPBCGS;

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  my_p4est_cell_neighbors_t *ngbd_c_np1 = NULL;
  my_p4est_faces_t *faces_np1 = NULL;

  // Boundary conditions:
  BoundaryConditionsDIM bc_velocity[P4EST_DIM];
  BoundaryConditionsDIM bc_pressure;

  CF_DIM* bc_interface_value_velocity[P4EST_DIM];
  BoundaryConditionType bc_interface_type_velocity[P4EST_DIM];

  CF_DIM* bc_wall_value_velocity[P4EST_DIM];
  BoundaryConditionType bc_wall_type_velocity[P4EST_DIM];

  CF_DIM* bc_interface_value_pressure;
  BoundaryConditionType bc_interface_type_pressure;

  CF_DIM* bc_wall_value_pressure;
  BoundaryConditionType bc_wall_type_pressure;

  // User provided forcing terms: (And the means to set it)
  CF_DIM* user_provided_external_forces_NS[P4EST_DIM];
  bool there_is_user_provided_external_force_NS = false;

  void set_user_provided_external_force_NS(CF_DIM* user_external_forces_NS_[2]){
    foreach_dimension(d){
      user_provided_external_forces_NS[d] = user_external_forces_NS_[d];
    }
    there_is_user_provided_external_force_NS = true;
  }
  // ----------------------------------------------
  // Related to domain:
  // ----------------------------------------------
  double xyz_min[P4EST_DIM]; double xyz_max[P4EST_DIM];
  int ntrees[P4EST_DIM];
  bool periodicity[P4EST_DIM];

  // Variables for refining the fields
  int lmin, lint, lmax;
  double uniform_band;

  bool use_uniform_band;
  bool refine_by_d2T;
  double vorticity_threshold;
  double gradT_threshold;

  // ----------------------------------------------
  // Related to interpolation bw grids:
  // ----------------------------------------------
  int num_fields_interp;
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // ----------------------------------------------
  // Related to current grid size:
  // ----------------------------------------------
  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;

  // ----------------------------------------------
  // Variables used for extension of fields:
  // ----------------------------------------------
  double min_volume_;
  double extension_band_use_;
  double extension_band_extend_;

  // ----------------------------------------------
  // Related to time/timestepping:
  // ----------------------------------------------
  double tn;
  double dt;
  double dt_nm1;
  double dt_Stefan;
  double dt_NS;

  int advection_sl_order;
  double advection_alpha_coeff;
  double advection_beta_coeff;

  int tstep;
  int load_tstep;
  int last_tstep;

  double cfl_Stefan;
  double cfl_NS;
  void set_cfl_Stefan(double cfl_stefan_){cfl_Stefan = cfl_stefan_;}
  void set_cfl_NS(double cfl_ns_){cfl_NS = cfl_ns_;}

  // ----------------------------------------------
  // Related to dimensionalization type:
  // ----------------------------------------------
  problem_dimensionalization_type_t problem_dimensionalization_type;
  void set_problem_dimensionalization_type(problem_dimensionalization_type_t prob_dim_type_){
    problem_dimensionalization_type = prob_dim_type_;
  }

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
      // Rochi temp change

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



  // ----------------------------------------------
  // Physical parameters:
  // Note: these must be provided for the solver to run !
  // ----------------------------------------------
  double l_char; // Characteristic length scale (assumed in meters)
  double u_inf; // Characteristic velocity scale (assumed in m/s)

  void set_l_char(double l_char_){l_char = l_char_;}

  double T0; // characteristic solid temperature of the problem
  double Tinterface; // Interface temperature or concentration
  double Tinfty; // Freestream fluid temperature or concentration
  double Tflush; // Flush temperature (K) or concentration that inlet BC is changed to if flush_dim_time is activated
  void set_dim_temp_conc_variables(double Tinfty_, double Tinterface_, double T0_){
    Tinfty = Tinfty_;
    Tinterface = Tinterface_;
    T0 = T0_;
  };

  void set_Tflush(double Tflush_){Tflush = Tflush_;}

  double alpha_l, alpha_s; // Liquid and solid thermal diffusivities, [m^2/s]
  double k_l, k_s; // Liquid and solid thermal conductivities, [W/(mK)]
  double rho_l, rho_s; // Liquid and solid densities, [kg/m^3]
  double cp_s; // Solid heat capacity [J/(kg K)]
  double L; // Latent heat of fusion [J/kg]
  double mu_l; // Fluid viscosity [Pa s]

  double grav; // Gravity
  double beta_T; // Thermal expansion coefficient for the boussinesq approx
  double beta_C; // Concentration expansion coefficient for the boussinesq approx

  double gamma_diss; // TO-DO: add description
  double stoich_coeff_diss; // The stoichiometric coefficient of the dissolution reaction
  double molar_volume_diss; // The molar volume of the dissolving solid

  double Dl, Ds; //Concentration diffusion coefficient m^2/s,
  double k_diss; // Dissolution rate constant per unit area of reactive surface (m/s)

  void set_alpha_l(double alpha_l_){alpha_l = alpha_l_;}
  void set_alpha_s(double alpha_s_){alpha_s = alpha_s_;}
  void set_k_l(double k_l_){k_l = k_l_;}
  void set_k_s(double k_s_){k_s = k_s_;}
  void set_rho_l(double rho_l_){rho_l = rho_l_;}
  void set_rho_s(double rho_s_){rho_s = rho_s_;}
  void set_cp_s(double cp_s_){cp_s = cp_s_;}
  void set_L(double L_){L = L_;}
  void set_mu_l(double mu_l_){mu_l = mu_l_;}
  void set_grav(double grav_){grav = grav_;}
  void set_beta_T(double beta_T_){beta_T = beta_T_;}
  void set_beta_C(double beta_C_){beta_C = beta_C_;}
  void set_gamma_diss(double gamma_diss_){gamma_diss = gamma_diss_;}
  void set_stoich_coeff_diss(double stoich_coeff_diss_){stoich_coeff_diss = stoich_coeff_diss_;}
  void set_molar_volume_diss(double molar_volume_diss_){molar_volume_diss = molar_volume_diss_;}
  void set_Dl(double Dl_){Dl = Dl_;}
  void set_Ds(double Ds_){Ds = Ds_;}
  void set_k_diss(double k_diss_){k_diss = k_diss_;}


  // ----------------------------------------------
  // Booleans related to what kind of physics we are solving
  // ----------------------------------------------
  bool solve_stefan;
  bool solve_navier_stokes;
  bool there_is_a_substrate/*example_uses_inner_LSF*/;

  bool do_we_solve_for_Ts;

  bool use_boussinesq;

  bool is_dissolution_case;

  bool force_interfacial_velocity_to_zero;

  // ----------------------------------------------
  // Booleans (misc)
  // ----------------------------------------------
  bool print_checkpoints; // can set this to true to debug where code might be crashing

  // ----------------------------------------------
  // Specific to diff cases --> may change these now that they are within a class structure
  // ----------------------------------------------
  bool analytical_IC_BC_forcing_term;
  bool example_is_a_test_case;

  bool interfacial_temp_bc_requires_curvature;
  bool interfacial_temp_bc_requires_normal;

  bool interfacial_vel_bc_requires_vint;

  bool example_has_known_max_vint;

  // ----------------------------------------------
  // Related to any front regularization:
  // ----------------------------------------------
  bool use_regularize_front;
  bool use_collapse_onto_substrate;

  double proximity_smoothing;
  double proxmity_collapse;

  // ----------------------------------------------
  // Related to LSF reinitialization
  // ----------------------------------------------
  int reinit_every_iter = 1;

  // ----------------------------------------------
  // Temperature problem variables -- nondim:
  // ----------------------------------------------
  // These things get computed depending on the provided dimensional quantities and case
  double deltaT;
  double theta_infty, theta_interface, theta0;
  void set_nondim_temp_conc_variables(double theta_infty_, double theta_interface_,
                                      double theta0_, double deltaT_){
    theta_infty = theta_infty_;
    theta_interface = theta_interface_;
    theta0 = theta0_;
    deltaT = deltaT_;
  };

  // ----------------------------------------------
  /* Classes related to temperature and velocity boundary conditions
   * (which depend on fields owned by the class that need to be updated in time)
   *  i.e. ) bc temp interface condition may depend on kappa or normals
   *  i.e.) bc velocity interface condition may depend on vinterface
   *  i.e.) both of these values may depend on some analytical form
   */
  // Classes related to temperature and velocity boundary conditions
  // (which depend on fields owned by the class that need to be updated in time, i.e.
  // ----------------------------------------------





  // -------------------------------------------------------
  // Functions related to scalar temp/conc problem:
  // -------------------------------------------------------
  void setup_rhs_for_scalar_temp_conc_problem();
  void do_backtrace_for_scalar_temp_conc_problem();


  // --------------------------------------------------------
  public:

    my_p4est_stefan_with_fluids_t();

};

#endif // STEFAN_WITH_FLUIDS_H
