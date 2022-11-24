#include "my_p4est_stefan_with_fluids.h"

#include <linux/limits.h>

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


// -------------------------------------------------------
// Constructor and other auxiliary initializations:
// -------------------------------------------------------

my_p4est_stefan_with_fluids_t::my_p4est_stefan_with_fluids_t(mpi_environment_t* mpi_)
{
  mpi = mpi_;
  // -----------------------------------------------
  // Grid variables
  // -----------------------------------------------
  conn = NULL;
  p4est_n = NULL; nodes_n = NULL; ghost_n = NULL;
  hierarchy_n = NULL; ngbd_n = NULL;

  p4est_np1 = NULL; nodes_np1 = NULL; ghost_np1 = NULL;
  hierarchy_np1 = NULL; ngbd_np1 = NULL;

  sp = NULL;
  ls = NULL;

  // -----------------------------------------------
  // Temperature/concentration problem:
  // -----------------------------------------------
  cube_refinement = 1;
  solver_Tl = NULL;
  solver_Ts = NULL;
  // Temp/conc problem BCs
  for(unsigned char i=0; i<2; i++){
    // Interface:
    bc_interface_val_temp[i] = NULL;
    bc_interface_type_temp[i] = NULL;
    bc_interface_robin_coeff_temp[i] = NULL;
    // Wall:
    bc_wall_value_temp[i] = NULL;
    bc_wall_type_temp[i] = NULL;
    // Substrate:
    bc_interface_val_temp_substrate[i] = NULL;
    bc_interface_type_temp_substrate[i] = NULL;
    bc_interface_robin_coeff_temp_substrate[i] = NULL;

    // Possible heat source:
    user_provided_external_heat_source[i] = NULL;
  }
  there_is_user_provided_heat_source = false;


  // ----------------------------------------------
  // Navier-Stokes problem:
  // ----------------------------------------------
  ns = NULL;
  face_solver = NULL;
  cell_solver = NULL;

  pc_face = PCSOR;
  face_solver_type = KSPBCGS;

  pc_cell=PCSOR;
  cell_solver_type = KSPBCGS;

  ngbd_c_np1 = NULL;
  faces_np1 = NULL;

  // Velocity Boundary conditions:
  foreach_dimension(d){
    // Interface:
    bc_interface_value_velocity[d] = NULL;
    bc_interface_type_velocity[d] = NULL;

    // Wall:
    bc_wall_value_velocity[d] = NULL;
    bc_wall_type_velocity[d] = NULL;
  }

  // Pressure boundary conditions:
  bc_interface_value_pressure = NULL;
  bc_interface_type_pressure=NOINTERFACE;

  bc_wall_value_pressure = NULL;
  bc_wall_type_pressure = NULL;

  // Forcing terms:
  foreach_dimension(d){
    user_provided_external_forces_NS[d] = NULL;
  }
  there_is_user_provided_external_force_NS = false;

  // Other parameters
  NS_norm = 1.;
  NS_max_allowed = DBL_MAX;

  hodge_max_it = 100;
  hodge_tolerance = 0.; // this gets overwritten in ns step
  hodge_percentage_of_max_u = 0.1;

  compute_pressure_ = false;// this will need to be provided by user at every timestep relevant
  // TO-DO: make sure compute_pressure_ handled correctly in main

  // ----------------------------------------------
  // Related to domain:
  // ----------------------------------------------
  // Set initial values (purposely unreasonable) for the domain info:
  foreach_dimension(d){
    xyz_min[d] = DBL_MAX;
    xyz_max[d] = DBL_MAX;
    periodicity[d] = 0;
    ntrees[d] = 0;
  }

  lmin = 0; lmax = 0; lint = 0;

  // Uniform band:
  uniform_band = 4.;
  use_uniform_band = true; // just default true for now

  refine_by_vorticity = false; // this gets set to true in ns step if ns being solved
  refine_by_d2T = false; // this needs to be set by user

  vorticity_threshold = 0.25;
  d2T_threshold = 1.e-2;

  loading_from_previous_state = false;


  // ----------------------------------------------
  // Related to interpolation bw grids:
  // ----------------------------------------------
  num_fields_interp = 0;
  // Interpolation bw grids:
  interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // ----------------------------------------------
  // Related to current grid size:
  // ----------------------------------------------
  foreach_dimension(d){
    dxyz_smallest[d] = DBL_MAX; // this gets overwritten in soln process
  }
  dxyz_close_to_interface = DBL_MAX; // this gets overwritten in soln process
  dxyz_close_to_interface_mult=1.2;

  // ----------------------------------------------
  // Variables used for extension of fields:
  // ----------------------------------------------
  // These all get overwritten in soln process
  min_volume_ = DBL_MAX;
  extension_band_use_ = DBL_MAX;
  extension_band_extend_= DBL_MAX;


  // ----------------------------------------------
  // Related to time/timestepping:
  // ----------------------------------------------
  // Init time variables to 0
  tn = 0.; dt = 0.; dt_nm1 = 0.;
  dt_Stefan = 0.; dt_NS = 0.; dt_max_allowed = DBL_MAX; dt_min_allowed = 0.;

  tstart = 0.; // TO-DO: revisit need for tstart
  // perhaps if user loads from prev state, they can set_tn accordingly?

  v_interface_max_allowed = DBL_MAX;

  advection_sl_order = 2;
  NS_advection_sl_order = 2;

  advection_alpha_coeff = advection_beta_coeff = 0.; // these get computed in soln process

  // Initialize tstep and loadtstep as diff numbers to ensure there is no equality unless they truly are the same
  tstep = -1;
  load_tstep = -2;

  cfl_Stefan = 0.5;
  cfl_NS = 2.;

  // ----------------------------------------------
  // Related to dimensionalization type:
  // ----------------------------------------------

  // Choose an initial dim type, but user should overwrite this
  problem_dimensionalization_type = DIMENSIONAL;

  time_nondim_to_dim = 1.;
  vel_nondim_to_dim = 1.;

  // ----------------------------------------------
  // Nondimensional groups
  // ----------------------------------------------
  Re = Pr = Sc = Pe = St = Da = RaT = RaC = -1.;

  // -------------------------------
  // Physical parameters:
  // -------------------------------
  l_char = 1.; u_inf = 1.;

  T0 = Tinterface = Tinfty = Tflush = 0.;

  alpha_l = alpha_s = k_l = k_s = rho_l = rho_s =
      cp_s = L = mu_l = sigma = 1;

  grav = beta_T = beta_C = 1;

  gamma_diss = stoich_coeff_diss = molar_volume_diss = k_diss = 1;
  Dl = Ds = 1;

  // ----------------------------------------------
  // Booleans related to what kind of physics we are solving
  // ---------------------------------------------
  solve_stefan = false;
  solve_navier_stokes = false;
  there_is_a_substrate = false;
  start_w_merged_grains = false;
  do_we_solve_for_Ts = false;
  use_boussinesq = false;
  is_dissolution_case = false;
  force_interfacial_velocity_to_zero = false;

  // ----------------------------------------------
  // Other misc parameters
  // ---------------------------------------------
  print_checkpoints = false;
  scale_vgamma_by = 1.;

  // ----------------------------------------------
  // Specific to diff cases
  // ----------------------------------------------
  interfacial_temp_bc_requires_curvature = false;
  interfacial_temp_bc_requires_normal = false;
  interfacial_vel_bc_requires_vint = false;

  track_evolving_geometries = false;

  // ----------------------------------------------
  // Related to any front regularization:
  // ----------------------------------------------
  use_regularize_front = false;
  use_collapse_onto_substrate = false;

  proximity_smoothing = 0.;
  proximity_collapse = 0.;

  // ----------------------------------------------
  // Related to LSF reinitialization
  // ----------------------------------------------
  reinit_every_iter = 1;

  // ----------------------------------------------
  // Level set functions
  // ----------------------------------------------
  level_set = NULL;
  substrate_level_set = NULL;

  // ----------------------------------------------
  // Temperature problem variables -- nondim:
  // ----------------------------------------------
  deltaT = theta_infty = theta_interface = theta0 = 0.;

  // -------------------------------------------------------
  // Auxiliary initializations:
  // -------------------------------------------------------
  grids_are_initialized = fields_are_initialized = false;
  initial_refinement_CF = NULL;

  for (unsigned char i=0; i<2; i++){
    initial_temp_n[i] = NULL;
    initial_temp_nm1[i] = NULL;
  }

  foreach_dimension(d){
    initial_NS_velocity_n[d] = NULL;
    initial_NS_velocity_nm1[d] = NULL;
  }


  // TO-DO: Initialize vectors to NULL in the constructor


  // That's all she wrote!

} // end of constructor

void my_p4est_stefan_with_fluids_t::initialize_grids(){

  // Create the p4est at time n:
  p4est_n = my_p4est_new(mpi->comm(), conn, 0, NULL, NULL);
  p4est_n->user_pointer = sp;


  for(int l=0; l<sp->max_lvl; l++){
    my_p4est_refine(p4est_n,P4EST_FALSE,refine_levelset_cf,NULL);
    my_p4est_partition(p4est_n,P4EST_FALSE,NULL);
  }
  p4est_balance(p4est_n,P4EST_CONNECT_FULL,NULL);
  my_p4est_partition(p4est_n,P4EST_FALSE,NULL);

  ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n,ghost_n);
  nodes_n = my_p4est_nodes_new(p4est_n, ghost_n); //same

  hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();

  // Create the p4est at time np1:(this will be modified but is useful for initializing solvers):
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE); // copy the grid but not the data
  p4est_np1->user_pointer = sp;
  my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);

  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1,ghost_np1);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  // Get the new neighbors:
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

  // Initialize the neigbors:
  ngbd_np1->init_neighbors();

  // Create the level set object:
  ls = new my_p4est_level_set_t(ngbd_np1);

  // Flag the grid initialization as complete:
  grids_are_initialized=true;

} // end of "initialize_grids()"

void my_p4est_stefan_with_fluids_t::initialize_fields(){

  // ---------------------------------
  // Level-set function(s):
  // ---------------------------------

  if(print_checkpoints) PetscPrintf(mpi->comm(),"Initializing the level set function (s) ... \n");
  phi.create(p4est_np1, nodes_np1);
  sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi.vec);
  ls->perturb_level_set_function(phi.vec, EPS);
  if(solve_stefan)ls->reinitialize_2nd_order(phi.vec,30); // reinitialize initial LSF to get good signed distance property

  if(start_w_merged_grains) {regularize_front();}

  if(there_is_a_substrate){
    create_and_compute_phi_sub_and_phi_eff();
  }

  if(solve_navier_stokes){
    // NS solver requires us to keep phi_nm1 for interpolating the hodge variable to the new grid.
    // Since p4est_np1 = p4est at initialization, it's safe to just copy phi and have them both on p4est_np1.
    // We will handle sliding these correctly later on
    // Initialize phi_nm1:
    phi_nm1.create(p4est_np1, nodes_np1);
    VecCopyGhost(phi.vec, phi_nm1.vec); // should be an option for phi_eff -- TO DO  , doesnt really matter i think, but in principle !
  }

  // ---------------------------------
  // Temperature/conc fields:
  // ---------------------------------

  if(solve_stefan){
    if(print_checkpoints) PetscPrintf(mpi->comm(),"Initializing the temperature fields (s) ... \n");

    T_l_n.create(p4est_np1, nodes_np1);
    sample_cf_on_nodes(p4est_np1, nodes_np1, *initial_temp_n[LIQUID_DOMAIN], T_l_n.vec);

    if(do_we_solve_for_Ts){
      T_s_n.create(p4est_np1, nodes_np1);
      sample_cf_on_nodes(p4est_np1, nodes_np1,*initial_temp_n[SOLID_DOMAIN],T_s_n.vec);
    }

    if(solve_navier_stokes && advection_sl_order ==2){
      T_l_nm1.create(p4est_np1, nodes_np1);
      sample_cf_on_nodes(p4est_np1, nodes_np1,*initial_temp_nm1[LIQUID_DOMAIN],T_l_nm1.vec);
    }

    // Extend fields temperature fields :
    // TO-DO: if we change defn's/declarations of extension bands elsewhere, make sure they're changed here as well
    if(print_checkpoints) PetscPrintf(mpi->comm(),"Doing initial field extension  ... \n");
    extend_relevant_fields();

    // Compute vinterface:
    if(print_checkpoints) PetscPrintf(mpi->comm(),"Computing initial velocity ... \n");
    v_interface.create(p4est_np1, nodes_np1);
    // TO-DO MULTICOMP: adjust this initial vel computation for the concentration case
    compute_interfacial_velocity();
  }

  // ---------------------------------
  // Navier-Stokes fields:
  // ---------------------------------

  if(solve_navier_stokes){
    if(print_checkpoints) PetscPrintf(mpi->comm(),"Initializing the Navier-Stokes fields (s) ... \n");

    v_n.create(p4est_np1, nodes_np1);
    v_nm1.create(p4est_np1, nodes_np1);

    foreach_dimension(d){
      sample_cf_on_nodes(p4est_np1,nodes_np1,*initial_NS_velocity_n[d],v_n.vec[d]);
      sample_cf_on_nodes(p4est_np1,nodes_np1,*initial_NS_velocity_nm1[d],v_nm1.vec[d]);
    }
  }

  // Flag the field initializations as complete:
  fields_are_initialized=true;

  // TO-DO: commented out below, make sure it's handled in main and then remove here
//  for(unsigned char d=0;d<P4EST_DIM;d++){
//    if(analytical_IC_BC_forcing_term){
//      delete analytical_soln[d];
//    }
//    if(solve_navier_stokes) delete v_init_cf[d];

//  }

  // Initialize the NS norm
  // TO-DO: make sure the NS norm is initialized elsewhere (or actually, see if this is even needed, I suspect it no longer is)

} // end of "initialize_fields()"


void my_p4est_stefan_with_fluids_t::initialize_grids_and_fields_from_load_state(){

  // Set everything to NULL at first:
  p4est_n=NULL;

  if(conn!=NULL){ // Destroy conn and set to NULL to prep for load (since we still created a conn when we made the brick)
    p4est_connectivity_destroy(conn);
    conn=NULL;
  }

  p4est_n=NULL; ghost_n=NULL;nodes_n=NULL;
  hierarchy_n=NULL;ngbd_n=NULL;

  phi.vec=NULL;
  T_l_n.vec=NULL; T_l_nm1.vec=NULL;
  T_s_n.vec=NULL;
  foreach_dimension(d){
    v_n.vec[d]=NULL;
    v_nm1.vec[d]=NULL;

  }
  vorticity.vec=NULL;

  // Get the load directory:
  const char* load_path = getenv("LOAD_STATE_PATH");
  if(!load_path){
    throw std::invalid_argument("You need to set the  directory for the desired load state");
  }
  PetscPrintf(mpi->comm(),"Load dir is:  %s \n",load_path);

  load_state(load_path);


  PetscPrintf(mpi->comm(),"State was loaded successfully from %s \n",load_path);

  // Update the neigborhood and hierarchy:
  if(hierarchy_n!=NULL) {
    delete hierarchy_n;
  }

  if(ngbd_n!=NULL) {delete ngbd_n;}


  hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();


  // Update the neigborhood and hierarchy:
  if(hierarchy_np1!=NULL) {
    delete hierarchy_np1;
  }
  if(ngbd_np1!=NULL) {delete ngbd_np1;}


  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
  ngbd_np1->init_neighbors();

  // Initialize the level set object
  ls = new my_p4est_level_set_t(ngbd_np1);

  // Create the additional phi_sub and phi_eff if required for this example
  if(there_is_a_substrate){
    create_and_compute_phi_sub_and_phi_eff();
  }

  // Extend fields:
  // TO-DO : if defn of extension bands is generalized, make sure that is applied here (or maybe we should just make a function called "compute extension bands" and use that everywhere
  if(solve_stefan)
  {
      dxyz_min(p4est_np1, dxyz_smallest);
      min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
      extension_band_use_    = (8.)*pow(min_volume_, 1./ double(P4EST_DIM)); //8
      extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
      dxyz_close_to_interface = dxyz_close_to_interface_mult*MAX(dxyz_smallest[0],dxyz_smallest[1]);

      extend_relevant_fields();

      // Compute vinterface:
      v_interface.create(p4est_np1, nodes_np1);
      compute_interfacial_velocity();
  }


  load_tstep =tstep;
  tstart=tn;
  dt_NS = dt_nm1;

  // Flag the initializations as completed:
  grids_are_initialized=true;
  fields_are_initialized=true;

} // end of "initialize_grids_from_load_state()"


void my_p4est_stefan_with_fluids_t::perform_initializations(){

  // Check and make sure the user has provided all the necessary information:


  // TO-DO: keeping this function in main, developing other checks for the class , make sure this works:
  // User is responsible for setting: dt_max_allowed situation, temp/conc defns, physical properties, etc.
  // User will need to convert their save_every_dt to nondim time to be compatible
//  setup_initial_parameters_and_report(mpi);

  // Replacement for "set_geometry()" in main
  if(!check_if_domain_info_is_set()){

    throw std::runtime_error("Not all of the necessary domain information is set to properly initialize the problem. Please set the xyz min and max, periodicity, and ntrees and then try again \n");
  }

  // Define number of fields for interp bw grids:
  num_fields_interp=0;
  if(solve_stefan){
    num_fields_interp+=1;
    if(do_we_solve_for_Ts) num_fields_interp+=1;
  }
  if(solve_navier_stokes){
    num_fields_interp+=2; // vNS_x, vNS_y
  }

  // User will need to set temp_and_conc defns first: // TO-DO make sure handled properly in main
  set_nondimensional_groups();


  // -----------------------------------------------
  // Report relevant information:
  // -----------------------------------------------
  PetscPrintf(mpi->comm(),"------------------------------------"
                          "\n \n"
                          "INITIAL PROBLEM INFORMATION\n"
                          "------------------------------------\n\n");

  PetscPrintf(mpi->comm(), "The nondimensionalizaton formulation being used is %s \n \n",
              (problem_dimensionalization_type == 0)?
                                                     ("NONDIM BY FLUID VELOCITY"):
                                                     ((problem_dimensionalization_type == 1) ?
                                                                                             ("NONDIM BY DIFFUSIVITY") : ("DIMENSIONAL")));

  PetscPrintf(mpi->comm(), "Nondim = %d \n"
                          "lmin = %d, lmax = %d \n"
                          "Number of mpi tasks: %d \n"
                          "Stefan = %d, NS = %d \n \n ", problem_dimensionalization_type,
              lmin, lmax,
              mpi->size(),
              solve_stefan, solve_navier_stokes);
  PetscPrintf(mpi->comm(),"The nondimensional groups are: \n"
                          "Re = %f \n"
                          "Pr = %f \n"
                          "Sc = %f \n"
                          "Pe = %f \n"
                          "St = %f \n"
                          "Da = %f \n"
                          "gamma_diss = %f \n"
                          "With: \n"
                          "u_inf = %0.3e [m/s]\n"
                          "delta T = %0.2f [K]\n"
                          "sigma = %0.3e, l_char = %0.3e, sigma/l_char = %0.3e \n \n",
              Re, Pr, Sc, Pe, St, Da, gamma_diss,
              u_inf,deltaT,sigma,l_char,sigma/l_char);


// TO-DO: commented out below, make sure it's handled in main
// We will do time handling externally:
//  PetscPrintf(mpi->comm(),"Simulation time: %0.4f [min] = %0.4f [sec] = %0.4f [nondim]\n\n",
//              tfinal*time_nondim_to_dim/60.,
//              tfinal*time_nondim_to_dim,
//              tfinal);


  // TO-DO: not printing any of this time info here, it should be handled in the main
//  bool using_startup = (startup_dim_time>0.) || (startup_nondim_time >0.);
//  bool using_dim_startup = using_startup && (startup_dim_time>0.);

//  PetscPrintf(mpi->comm(),"Are we using startup time? %s \n \n",using_startup? "Yes": "No");
//  if(using_startup){
//    PetscPrintf(mpi->comm(),"Startup time: %s = %0.2f %s \n", using_dim_startup? "Dimensional" : "Nondimensional", using_dim_startup? startup_dim_time:startup_nondim_time,using_dim_startup? "[seconds]": "[nondim]");
//  }


  PetscPrintf(mpi->comm(),"Uniform band is %0.1f \n \n ",uniform_band);

//  PetscPrintf(mpi->comm(),"Are we ramping bcs? %s \n t_ramp = %0.2f [nondim] = %0.2f [seconds] \n \n",ramp_bcs?"Yes":"No",t_ramp,t_ramp*time_nondim_to_dim);

//  PetscPrintf(mpi->comm(),"Are we loading from previous state? %s \n"
//                          "Starting timestep = %d \n"
//                          "Save state every iter = %d \n"
//                          "Save to vtk? %s \n"
//                          "Save using %s \n"
//                          "Save every dt = %0.5e [nondim] = %0.2f [seconds]\n"
//                          "Save every iter = %d \n \n",loading_from_previous_state?"Yes":"No",
//              tstep,
//              save_state_every_iter,
//              save_to_vtk?"Yes":"No",
//              save_using_dt? "dt" :"iter",
//              save_every_dt, save_every_dt*time_nondim_to_dim,
//              save_every_iter);
  PetscPrintf(mpi->comm(),"------------------------------------\n\n");



  // -----------------------------------------------
  // Perform grid and field initializations
  // -----------------------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning grid and field initializations ... \n");

  sp = new splitting_criteria_cf_and_uniform_band_t(lmin,
                                                    lmax,
                                                    initial_refinement_CF,
                                                    uniform_band, 2.0);

  conn = my_p4est_brick_new(ntrees, xyz_min, xyz_max, &brick, periodicity);

  // Call the relevant initialization fxns
  if(loading_from_previous_state){
    initialize_grids_and_fields_from_load_state();
  }
  else{
    initialize_grids();
    initialize_fields();

    tstep = 0;
    tn = tstart;
  }

  // ------------------------------------------------------------
  // Initialize Navier Stokes solver:
  // ------------------------------------------------------------
  if(solve_navier_stokes){
    initialize_ns_solver();
  }


  // ------------------------------------------------------------
  // Compute the initial timestep:
  // ------------------------------------------------------------

  dxyz_min(p4est_np1, dxyz_smallest);
  dxyz_close_to_interface = dxyz_close_to_interface_mult*MAX(dxyz_smallest[0],dxyz_smallest[1]);
  compute_timestep();

  if(!loading_from_previous_state) dt_nm1 = dt; // since we are just starting up
} // end of "perform_initializations()"



// -------------------------------------------------------
// Destructor:
// -------------------------------------------------------
my_p4est_stefan_with_fluids_t::~my_p4est_stefan_with_fluids_t()
{

  // Final destructions
  phi.destroy();

  if(there_is_a_substrate){
    phi_substrate.destroy();
    phi_eff.destroy();
  }

  if(ls!=NULL) delete ls;
  if(sp!=NULL) delete sp;

  if(solve_stefan){
    T_l_n.destroy();
    T_s_n.destroy();
    v_interface.destroy();

    if(advection_sl_order==2) T_l_nm1.destroy();


    if(!solve_navier_stokes){
      // destroy the structures leftover (in non NS case)
      if(nodes_n !=NULL) p4est_nodes_destroy(nodes_n);
      if(ghost_n !=NULL) p4est_ghost_destroy(ghost_n);
      if(p4est_n !=NULL) p4est_destroy      (p4est_n);

      if(nodes_np1 !=NULL) p4est_nodes_destroy(nodes_np1);
      if(ghost_np1 !=NULL) p4est_ghost_destroy(ghost_np1);
      if(p4est_np1 !=NULL) p4est_destroy(p4est_np1);

      if(conn !=NULL) my_p4est_brick_destroy(conn, &brick);
      if(hierarchy_n !=NULL) delete hierarchy_n;
      if(ngbd_n !=NULL) delete ngbd_n;

      if(hierarchy_np1 !=NULL) delete hierarchy_np1;
      if(ngbd_np1 !=NULL) delete ngbd_np1;
    }
  }

  if(solve_navier_stokes){
    v_n.destroy();
    v_nm1.destroy();
    phi_nm1.destroy();

    // NS takes care of destroying v_NS_n and v_NS_nm1
    vorticity.destroy();
    press_nodes.destroy();

    // NUllify some NS stuff that we have already deleted so she doesn't get angry:

    ns->nullify_phi();
    ns->nullify_velocities_at_nodes();
    ns->nullify_vorticity();
    if(ns!=NULL) delete ns;
    MPI_Barrier(mpi->comm());
  }

} // end of destructor

// -------------------------------------------------------
// Functions related to scalar temp/conc problem: ( in order of their usage in the main step)
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::do_backtrace_for_scalar_temp_conc_problem(bool do_multicomponent_fields=false, int num_conc_fields=0){
  // -------------------
  // A note on notation:
  // -------------------
  // Recall that at this stage, we are computing backtrace points for
  // -- T_n (sampled on the grid np1) and T_nm1 (sampled on the grid n)
  // using the fluid velocities
  // -- v_n_NS (sampled on grid np1) and v_nm1_NS (sampled on the grid n)

  // This notation can be a bit confusing, but stems from the fact that the grid np1 has been chosen around the interface location at time np1,
  // and all the fields at n have been interpolated to this new grid to solve for fields at np1.
  // Thus, while T_n is sampled on the grid np1, it is indeed still the field at time n, simply transferred to the grid used to solve for the np1 fields.


  // NOTE FOR THE MULTICOMPONENT CASE:
  // This function assumes that the user has already provided 
  // --> The fields:
  //         -Cl_n, Cl_nm1, Cl_n_backtrace, Cl_nm1_backtrace to the solver as vec_and_ptr_array_t objects sampled on the 
  // grids p4est_np1 and p4est_n respectively. 
  // --> The grids/grid objects:
  //         - p4est_np1, nodes_np1, ngbd_np1, ngbd_n  
  // This function will not create nor destroy those objects.

  // Therefore, before the multicomponent user calls this fxn, they should call the following: 
  // set_Cl_n
  // set_Cl_nm1
  // set_Cl_n_backtrace (this can just be a vec which will hold the values computed by this fxn)
  // set_Cl_nm1_backtrace (" ")
  // set_T_l

  // Commenting out below because we will assume that multialloy has provided *dimensional* velocities to begin with
//  if(do_multicomponent_fields){
//    // In this case, we need dimensional fluid velocities (bc the multialloy problem is solved dimensionally)

//    printf("vel_nondim_to_dim in stefan_w_fluids= %0.2e \n", vel_nondim_to_dim);
//    foreach_dimension(d){
//      ierr = VecScaleGhost(v_n.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
//      ierr = VecScaleGhost(v_nm1.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
//    }
//  }




  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning to do backtrace \n");
  PetscPrintf(mpi->comm(), "[SWF] Addresses for ngbd objects: \n"
                           "ngbd_np1 = %p \n"
                           "ngbd_n = %p \n", ngbd_np1, ngbd_n);

  // If we are doing the multialloy case, verify that all the necessary things are defined:

  if(0){
    // -------------------------------
    // TEMPORARY: save fields before backtrace
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};


    point_fields.push_back(Vec_for_vtk_export_t(T_l_n.vec, "Tl_tn"));
    point_fields.push_back(Vec_for_vtk_export_t(Cl_n.vec[0], "Cl0_tn"));
    point_fields.push_back(Vec_for_vtk_export_t(Cl_n.vec[1], "Cl1_tn"));


    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_inside_backtrace_np1_fields", out_dir);
    my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ngbd_np1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();
  }

  if(do_multicomponent_fields){

    bool conc_check = true;

    for(int j=0; j<num_conc_fields; j++){

      conc_check = (Cl_n.vec[j]!=NULL) && (Cl_nm1.vec[j]!=NULL) &&
                            (Cl_backtrace_nm1.vec[j]!=NULL);
    }

    if(!conc_check){
      throw std::runtime_error("my_p4est_stefan_with_fluids:do_backtrace_for_scalar_temp_conc_problem():" 
                               "The concentration check has failed for multicomponent usage." 
                               "Please provide the necessary concentration vectors to the stefan_with_fluids solver before calling this fxn.");
    }

    bool temp_check = (T_l_n.vec!=NULL) && (T_l_nm1.vec!=NULL) && 
                      (T_l_backtrace_n.vec!=NULL) && (T_l_backtrace_nm1.vec!=NULL);
    if(!temp_check){
      throw std::runtime_error("my_p4est_stefan_with_fluids:do_backtrace_for_scalar_temp_conc_problem():" 
                               "The temperature check has failed for multicomponent usage." 
                               "Please provide the necessary concentration vectors to the stefan_with_fluids solver before calling this fxn.");
    }

    bool grid_check = (p4est_np1!=NULL) && (nodes_np1!=NULL) && (ngbd_np1!=NULL) && (ngbd_n!=NULL);


    if(!grid_check){
      throw std::runtime_error("my_p4est_stefan_with_fluids:do_backtrace_for_scalar_temp_conc_problem():" 
                               "The grid check has failed for multicomponent usage." 
                               "Please provide the necessary concentration vectors to the stefan_with_fluids solver before calling this fxn.");
    }

    // Initialize the neighbors (just in case)
    ngbd_np1->init_neighbors();
    ngbd_n->init_neighbors();
    
  }

  // Initialize objects we will use in this function:
  // PETSC Vectors for second derivatives
  vec_and_ptr_dim_t T_l_dd, T_l_dd_nm1;
  Vec v_dd[P4EST_DIM][P4EST_DIM];
  Vec v_dd_nm1[P4EST_DIM][P4EST_DIM];

  // Initialize for potential concentration fields from multialloy
  std::vector<vec_and_ptr_dim_t> Cl_dd;
  std::vector<vec_and_ptr_dim_t> Cl_dd_nm1;

  if(do_multicomponent_fields){
    Cl_dd.resize(num_conc_fields);
    if(advection_sl_order == 2) Cl_dd_nm1.resize(num_conc_fields);

    // ^ The above will be a [num_conc_fields]x[P4EST_DIM]x[num_nodes] object. 
    // i.e.) Cl_dd[j].vec[d][n] will hold the derivative of the jth concentration field in the d Cartesian direction
    // at node index n 
  }

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Create the necessary interpolators
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd_np1); /*= NULL;*/
  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd_n);/* = NULL;*/

  // Get the relevant second derivatives
  T_l_dd.create(p4est_np1, nodes_np1);
  ngbd_np1->second_derivatives_central(T_l_n.vec, T_l_dd.vec);

  if(advection_sl_order==2) {
    T_l_dd_nm1.create(p4est_n, nodes_n);
    ngbd_n->second_derivatives_central(T_l_nm1.vec,T_l_dd_nm1.vec);
  }

  if(do_multicomponent_fields){
    for(int j=0; j<num_conc_fields; j++){
      Cl_dd[j].create(p4est_np1, nodes_np1);
      ngbd_np1->second_derivatives_central(Cl_n.vec[j], Cl_dd[j].vec);

      if(advection_sl_order==2){
        Cl_dd_nm1[j].create(p4est_np1, nodes_np1);
        ngbd_n->second_derivatives_central(Cl_nm1.vec[j], Cl_dd_nm1[j].vec);
      }
    }
  }

  foreach_dimension(d){
    foreach_dimension(dd){
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2){
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &v_dd_nm1[d][dd]); CHKERRXX(ierr);
      }
    }
  }

  // v_dd[k] is the second derivative of the velocity components n along cartesian direction k
  // v_dd_nm1[k] is the second derivative of the velocity components nm1 along cartesian direction k

  ngbd_np1->second_derivatives_central(v_n.vec,v_dd[0],v_dd[1],P4EST_DIM);
  if(advection_sl_order ==2){
    ngbd_n->second_derivatives_central(v_nm1.vec, DIM(v_dd_nm1[0], v_dd_nm1[1], v_dd_nm1[2]), P4EST_DIM);
  }

  // Do the Semi-Lagrangian backtrace:
  if(advection_sl_order ==2){
    trajectory_from_np1_to_nm1(p4est_np1, nodes_np1, ngbd_n, ngbd_np1, v_nm1.vec, v_dd_nm1, v_n.vec, v_dd, dt_nm1, dt, xyz_d_nm1, xyz_d);

    if(print_checkpoints) PetscPrintf(p4est_np1->mpicomm,"Completes backtrace trajectory \n");
  }
  else{
    trajectory_from_np1_to_n(p4est_np1, nodes_np1, ngbd_np1, dt, v_n.vec, v_dd, xyz_d);
  }

  // Add backtrace points to the interpolator(s):
//  foreach_local_node(n, nodes_np1){
//  VecView(v_n.vec[1], PETSC_VIEWER_STDOUT_WORLD);
  foreach_node(n, nodes_np1){
    double xyz_temp[P4EST_DIM];
    double xyz_temp_nm1[P4EST_DIM];

    double xyz_[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz_);
//    printf("\n node %d, (x,y) = (%0.2f, %0.2f) has (xdn, ydn) = (%0.2f, %0.2f) and (xdnm1, ydnm1) = (%0.2f, %0.2f) \n",
//           n, xyz_[0], xyz_[1], xyz_d[0][n], xyz_d[1][n], xyz_d_nm1[0][n], xyz_d_nm1[1][n]);


    foreach_dimension(d){
      xyz_temp[d] = xyz_d[d][n];

      if(advection_sl_order ==2){
        xyz_temp_nm1[d] = xyz_d_nm1[d][n];
      }
    } // end of "for each dimension"

    SL_backtrace_interp.add_point(n,xyz_temp);
    if(advection_sl_order ==2 ) SL_backtrace_interp_nm1.add_point(n,xyz_temp_nm1);
  } // end of loop over local nodes

  // Interpolate the Temperature data to back-traced points:
  SL_backtrace_interp.set_input(T_l_n.vec, T_l_dd.vec[0], T_l_dd.vec[1],quadratic_non_oscillatory_continuous_v2);
  SL_backtrace_interp.interpolate(T_l_backtrace_n.vec);

  if(advection_sl_order ==2){
    SL_backtrace_interp_nm1.set_input(T_l_nm1.vec, T_l_dd_nm1.vec[0], T_l_dd_nm1.vec[1], quadratic_non_oscillatory_continuous_v2);
    SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
  }

  // Interpolate the concentration data to back-traced points:
  if(do_multicomponent_fields){
    for(int j=0; j<num_conc_fields; j++){

      SL_backtrace_interp.set_input(Cl_n.vec[j], Cl_dd[j].vec[0], Cl_dd[j].vec[1], quadratic_non_oscillatory_continuous_v2);
      SL_backtrace_interp.interpolate(Cl_backtrace_n.vec[j]);

    if(advection_sl_order==2){
      SL_backtrace_interp_nm1.set_input(Cl_nm1.vec[j], Cl_dd_nm1[j].vec[0], Cl_dd_nm1[j].vec[1], quadratic_non_oscillatory_continuous_v2);
      SL_backtrace_interp_nm1.interpolate(Cl_backtrace_nm1.vec[j]);
    }

    }
  }

  // Destroy velocity derivatives now that not needed:
  foreach_dimension(d){
    foreach_dimension(dd)
    {
      ierr = VecDestroy(v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2) ierr = VecDestroy(v_dd_nm1[d][dd]); CHKERRXX(ierr);
    }
  }

  // Destroy temperature derivatives
  T_l_dd.destroy();
  if(advection_sl_order==2) {
    T_l_dd_nm1.destroy();
  }

  // Destroy multicomp derivatives:
  if(do_multicomponent_fields){
    for(int j=0; j<num_conc_fields; j++){
      Cl_dd[j].destroy();
      if(advection_sl_order==2){
        Cl_dd_nm1[j].destroy();
      }
    }

    Cl_dd.clear();
    if(advection_sl_order == 2) Cl_dd_nm1.clear();
    // ^ The above will be a [num_conc_fields]x[P4EST_DIM]x[num_nodes] object. 
    // i.e.) Cl_dd[j].vec[d][n] will hold the derivative of the jth concentration field in the d Cartesian direction
    // at node index n 
  }

  // Clear interp points:
  xyz_d->clear();xyz_d->shrink_to_fit();
  xyz_d_nm1->clear();xyz_d_nm1->shrink_to_fit();

  // Clear and delete interpolators:
  SL_backtrace_interp.clear();
  SL_backtrace_interp_nm1.clear();

  // Commenting out below because we will assume that multialloy has provided *dimensional* velocities to begin with
//  if(do_multicomponent_fields){
//    // In this case, we converted to dimensional, so we need to convert back
//    foreach_dimension(d){
//      ierr = VecScaleGhost(v_n.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
//      ierr = VecScaleGhost(v_nm1.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
//    }
//  }


  if(print_checkpoints) PetscPrintf(p4est_np1->mpicomm,"Completes backtrace \n");
} // end of "do_backtrace_for_scalar_temp_conc_problem"


void my_p4est_stefan_with_fluids_t::setup_rhs_for_scalar_temp_conc_problem(){

  // In building RHS, if we are doing advection, we have two options:
  // (1) 1st order -- approx is (dT/dt + u dot grad(T)) ~ (T(n+1) - Td(n))/dt --> so we add Td/dt to the RHS
  // (2) 2nd order -- approx is (dT/dt + u dot grad(T)) ~ alpha*(T(n+1) - Td(n))/dt + beta*(Td(n) - Td(n-1))/dt_nm1
  //                       --> so we add Td(n)*(alpha/dt - beta/dt_nm1) + Td(n-1)*(beta/dt_nm1) to the RHS
  //               -- where alpha and beta are weights of the two timesteps
  // See Semi-Lagrangian backtrace advection schemes for more details

  // If we are not doing advection, then we have:
  // (1) dT/dt = (T(n+1) - T(n)/dt) --> which is a backward euler 1st order approximation (since the RHS is discretized spatially at T(n+1))
  // (2) dT/dt = alpha*laplace(T) ~ (T(n+1) - T(n)/dt) = (1/2)*(laplace(T(n)) + laplace(T(n+1)) )  ,
  //                              in which case we need the second derivatives of the temperature field at time n


  // Establish forcing terms if applicable:
  vec_and_ptr_t forcing_term_liquid;
  vec_and_ptr_t forcing_term_solid;

  if(there_is_user_provided_heat_source){
    forcing_term_liquid.create(p4est_np1, nodes_np1);
    sample_cf_on_nodes(p4est_np1, nodes_np1, *user_provided_external_heat_source[LIQUID_DOMAIN], forcing_term_liquid.vec);

    if(do_we_solve_for_Ts) {
      forcing_term_solid.create(p4est_np1, nodes_np1);
      sample_cf_on_nodes(p4est_np1, nodes_np1, *user_provided_external_heat_source[SOLID_DOMAIN], forcing_term_solid.vec);
    }
  }

  // Prep coefficients if we are doing 2nd order advection:
  // TO-DO: probably should move calculation of these coefficients elsewhere
  if(solve_navier_stokes && advection_sl_order==2){
    advection_alpha_coeff = (2.*dt + dt_nm1)/(dt + dt_nm1);
    advection_beta_coeff = (-1.*dt)/(dt + dt_nm1);
  }
  // Get Ts arrays:
  if(do_we_solve_for_Ts){
    T_s_n.get_array();
    rhs_Ts.get_array();
  }

  // Get Tl arrays:
  rhs_Tl.get_array();
  if(solve_navier_stokes){
    T_l_backtrace_n.get_array();
    if(advection_sl_order ==2) T_l_backtrace_nm1.get_array();
  }
  else{
    T_l_n.get_array();
  }

  if(there_is_user_provided_heat_source){
    forcing_term_liquid.get_array();
    if(do_we_solve_for_Ts) forcing_term_solid.get_array();
  }

  phi.get_array();
  // 3-7-22 : Elyce changed from foreach_local_node to foreach_node --> when I visualized rhs it was patchy ...
  foreach_node(n, nodes_np1){
    if(do_we_solve_for_Ts){
      // Backward Euler
      rhs_Ts.ptr[n] = T_s_n.ptr[n]/dt;
    }

    // Now for Tl depending on case:
    if(solve_navier_stokes){
      if(advection_sl_order ==2){
        rhs_Tl.ptr[n] = T_l_backtrace_n.ptr[n]*((advection_alpha_coeff/dt) - (advection_beta_coeff/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1);
      }
      else{
        rhs_Tl.ptr[n] = T_l_backtrace_n.ptr[n]/dt;
      }
    }
    else{
      // Backward Euler
      rhs_Tl.ptr[n] = T_l_n.ptr[n]/dt;
    }
    if(there_is_user_provided_heat_source){
      // Add forcing terms:
      rhs_Tl.ptr[n]+=forcing_term_liquid.ptr[n];
      if(do_we_solve_for_Ts) rhs_Ts.ptr[n]+=forcing_term_solid.ptr[n];
    }
//    std::cout<<rhs_Tl.ptr[n]<<"\n";
//    std::cout<< "Time step dt0" << dt << "\n";
//    std::cout<< "Time step dtnm1" << dt_nm1 << "\n";
//    std::cout<< "T_l_backtrace_n : " << T_l_backtrace_n.ptr[n] <<"\n";
//    std::cout<< "T_l_backtrace_nm1 : " << T_l_backtrace_nm1.ptr[n] <<"\n";
//    std::cout<<"First term :: "<< T_l_backtrace_n.ptr[n]*((advection_alpha_coeff/dt))<<"\n";
//    std::cout<<"Second term :: "<< -T_l_backtrace_n.ptr[n]*((advection_beta_coeff/dt_nm1))<<"\n";
//    std::cout<<"Third term :: "<< T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1)<<"\n";
  }// end of loop over nodes

  // Restore arrays:
  phi.restore_array();

  if(do_we_solve_for_Ts){
    T_s_n.restore_array();
    rhs_Ts.restore_array();
  }

  rhs_Tl.restore_array();
  if(solve_navier_stokes){
    T_l_backtrace_n.restore_array();
    if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
  }
  else{
    T_l_n.restore_array();
  }

  if(there_is_user_provided_heat_source){
    forcing_term_liquid.restore_array();

    if(do_we_solve_for_Ts) {
      forcing_term_solid.restore_array(); forcing_term_solid.destroy();
    }

    // Destroy these if they were created
    forcing_term_liquid.destroy();
  }
}


void my_p4est_stefan_with_fluids_t::poisson_nodes_step_for_scalar_temp_conc_problem(){
  // Create solvers:
  solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd_np1);
  if(do_we_solve_for_Ts) solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

  // Add the appropriate interfaces and interfacial boundary conditions:
  solver_Tl->add_boundary(MLS_INTERSECTION, phi.vec,
                          phi_dd.vec[0], phi_dd.vec[1],
                          *bc_interface_type_temp[LIQUID_DOMAIN],
                          *bc_interface_val_temp[LIQUID_DOMAIN], *bc_interface_robin_coeff_temp[LIQUID_DOMAIN]);
//  PetscPrintf(mpi->comm(), "bc interface type = %d, bc interface val = %f, robin coeff = %f \n",
//              *bc_interface_type_temp[LIQUID_DOMAIN], (*bc_interface_val_temp[LIQUID_DOMAIN])(1.,1.), *bc_interface_robin_coeff_temp[LIQUID_DOMAIN]);

  if(do_we_solve_for_Ts){
    solver_Ts->add_boundary(MLS_INTERSECTION, phi_solid.vec,
                            phi_solid_dd.vec[0], phi_solid_dd.vec[1],
                            *bc_interface_type_temp[SOLID_DOMAIN],
                            *bc_interface_val_temp[SOLID_DOMAIN],
                            *bc_interface_robin_coeff_temp[SOLID_DOMAIN]);
  }

  if(there_is_a_substrate){
    // Need to add this is the event that phi collapses onto the substrate and we need the phi_substrate BC to take over in that region
    solver_Tl->add_boundary(MLS_INTERSECTION, phi_substrate.vec,
                            phi_substrate_dd.vec[0], phi_substrate_dd.vec[1],
                            *bc_interface_type_temp_substrate[LIQUID_DOMAIN],
                            *bc_interface_val_temp_substrate[LIQUID_DOMAIN],
                            *bc_interface_robin_coeff_temp_substrate[LIQUID_DOMAIN]);
    if(do_we_solve_for_Ts){
      // Need to add this to fully define the solid domain (assuming solid is sitting on substrate, thus bounded by liquid and substrate)
      solver_Ts->add_boundary(MLS_INTERSECTION, phi_substrate.vec,
                              phi_substrate_dd.vec[0], phi_substrate_dd.vec[1],
                              *bc_interface_type_temp_substrate[SOLID_DOMAIN],
                              *bc_interface_val_temp_substrate[SOLID_DOMAIN],
                              *bc_interface_robin_coeff_temp_substrate[SOLID_DOMAIN]);
    }
  }

  // Set diagonal for Tl:
  if(solve_navier_stokes){ // Cases with advection use semi lagrangian advection discretization in time
    if(advection_sl_order ==2){ // 2nd order semi lagrangian (BDF2 coefficients)
      solver_Tl->set_diag(advection_alpha_coeff/dt);
    }
    else{ // 1st order semi lagrangian (Backward Euler but with backtrace)
      solver_Tl->set_diag(1./dt);
    }
  }
  else{ // Cases with no temperature advection
    // Backward Euler
    solver_Tl->set_diag(1./dt);
  }

  if(do_we_solve_for_Ts){
    // Set diagonal for Ts:
    // Backward Euler
    solver_Ts->set_diag(1./dt);
  }
  switch(problem_dimensionalization_type){
  case NONDIM_BY_FLUID_VELOCITY:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(1./Pe);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((1./Pe)*(alpha_s/alpha_l));
    }
    else{
      solver_Tl->set_mu(1./Pe);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((1./Pe)*(Ds/Dl));
    }
    break;
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(1.);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((alpha_s/alpha_l));
    }
    else{
      solver_Tl->set_mu(1.);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((Ds/Dl));
    }
    break;
  }
  case DIMENSIONAL:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(alpha_l);
      if(do_we_solve_for_Ts) solver_Ts->set_mu(alpha_s);
    }
    else{
      solver_Tl->set_mu(Dl);
      if(do_we_solve_for_Ts) solver_Ts->set_mu(Ds);
    }
    break;
  }
  default:{
    throw std::runtime_error("main_2d:poisson_step: unrecognized problem dimensionalization type when setting diffusion coefficients for poisson solver \n");
    break;
  }
  }


  // Set RHS:
  solver_Tl->set_rhs(rhs_Tl.vec);
  if(do_we_solve_for_Ts) solver_Ts->set_rhs(rhs_Ts.vec);

  // Set some other solver properties:
  solver_Tl->set_integration_order(1);
//  solver_Tl->set_use_sc_scheme(0);
  solver_Tl->set_fv_scheme(0);
  solver_Tl->set_cube_refinement(cube_refinement);
  solver_Tl->set_store_finite_volumes(0);
  if(do_we_solve_for_Ts){
    solver_Ts->set_integration_order(1);
//    solver_Ts->set_use_sc_scheme(0);
    solver_Ts->set_fv_scheme(0);
    solver_Ts->set_cube_refinement(cube_refinement);
    solver_Ts->set_store_finite_volumes(0);
  }


  // Set the wall BC and RHS:
  solver_Tl->set_wc(*bc_wall_type_temp[LIQUID_DOMAIN], *bc_wall_value_temp[LIQUID_DOMAIN]);
  if(do_we_solve_for_Ts) solver_Ts->set_wc(*bc_wall_type_temp[SOLID_DOMAIN], *bc_wall_value_temp[SOLID_DOMAIN]);

  // Preassemble the linear system
  solver_Tl->preassemble_linear_system();

  if(do_we_solve_for_Ts) solver_Ts->preassemble_linear_system();

  // Solve the system:
  solver_Tl->solve(T_l_n.vec, false, true, KSPBCGS, PCHYPRE);
  if(do_we_solve_for_Ts) solver_Ts->solve(T_s_n.vec, false, true, KSPBCGS, PCHYPRE);

  // Delete solvers:
  delete solver_Tl;
  if(do_we_solve_for_Ts) delete solver_Ts;
} // end of "poisson_nodes_step_for_scalar_temp_conc_problem()"


void my_p4est_stefan_with_fluids_t::setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc()
{
  PetscErrorCode ierr;

  // -------------------------------
  // Create all vectors that will be used
  // strictly for the stefan step
  // (aka created and destroyed in stefan step)
  // -------------------------------

  // Solid LSF:
  phi_solid.create(p4est_np1,nodes_np1);

  //Curvature and normal for BC's and setting up solver:
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature) normal.create(p4est_np1,nodes_np1);
  if(interfacial_temp_bc_requires_curvature)curvature.create(p4est_np1,nodes_np1);

  // Second derivatives of LSF's (for solver):
  phi_solid_dd.create(p4est_np1,nodes_np1);
  phi_dd.create(p4est_np1,nodes_np1);

  if(there_is_a_substrate){
    phi_substrate_dd.create(p4est_np1,nodes_np1);
  }
  if(solve_navier_stokes){
    T_l_backtrace_n.create(p4est_np1,nodes_np1);
    if(advection_sl_order ==2){
      T_l_backtrace_nm1.create(p4est_np1,nodes_np1);
    }
  }
  // Create arrays to hold the RHS:
  rhs_Tl.create(p4est_np1,nodes_np1);
  if(do_we_solve_for_Ts) rhs_Ts.create(p4est_np1,nodes_np1);

  // -------------------------------
  // Compute the normal and curvature of the interface
  //-- curvature is used in some of the interfacial boundary condition(s) on temperature
  // -------------------------------

  if(print_checkpoints) PetscPrintf(mpi->comm(),"Computing normal and curvature ... \n");
  // Get the new solid LSF:
  VecCopyGhost(phi.vec, phi_solid.vec);
  VecScaleGhost(phi_solid.vec, -1.0);

  // Compute normals on the interface (if needed) :
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature){
    compute_normals(*ngbd_np1, phi_solid.vec, normal.vec); // normal here is outward normal of solid domain

    // Feed the normals if relevant
    if(interfacial_temp_bc_requires_normal){
      for(unsigned char d=0; d<2; d++){
        bc_interface_val_temp[d]->set_normals_interp(ngbd_np1, normal.vec[0], normal.vec[1]);
      }
    }
  }

  // Compute curvature if needed and feed to bc object:
  if(interfacial_temp_bc_requires_curvature){
    // We need curvature of the solid domain, so we use phi_solid and negative of normals
    compute_mean_curvature(*ngbd_np1, normal.vec, curvature.vec);

    for(unsigned char d=0;d<2;d++){
      bc_interface_val_temp[d]->set_kappa_interp(ngbd_np1, curvature.vec);
    }
  }


  // -------------------------------
  // Get most updated derivatives of the LSF's (on current grid)
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning Poisson problem ... \n");

  // Get derivatives of liquid and solid LSF's
  if (print_checkpoints) PetscPrintf(mpi->comm(),"New solid LSF acquired \n");
  ngbd_np1->second_derivatives_central(phi.vec, phi_dd.vec);
  ngbd_np1->second_derivatives_central(phi_solid.vec, phi_solid_dd.vec);

  // Get inner LSF and derivatives if required:
  if(there_is_a_substrate){
    ngbd_np1->second_derivatives_central(phi_substrate.vec, phi_substrate_dd.vec);
  }

  // -------------------------------
  // Compute advection terms (if applicable):
  // -------------------------------
  if (solve_navier_stokes){
    if(print_checkpoints) PetscPrintf(mpi->comm(),"Computing advection terms ... \n");
    do_backtrace_for_scalar_temp_conc_problem();
  } // end of solve_navier_stokes if statement

  // -------------------------------
  // Set up the RHS for Poisson step:
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Setting up RHS for Poisson problem ... \n");

  setup_rhs_for_scalar_temp_conc_problem();


  // -------------------------------
  // Execute the Poisson step:
  // -------------------------------
  // Slide Temp fields:
  if(solve_navier_stokes && advection_sl_order==2){
    T_l_nm1.destroy();
    T_l_nm1.create(p4est_np1, nodes_np1);
    ierr = VecCopyGhost(T_l_n.vec, T_l_nm1.vec);CHKERRXX(ierr);
  }
  // Solve Poisson problem:
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning Poisson problem solution step... \n");

  poisson_nodes_step_for_scalar_temp_conc_problem();

  if(print_checkpoints) PetscPrintf(mpi->comm(),"Poisson step completed ... \n");

  // -------------------------------
  // Clear interfacial BC if needed (curvature, normals, or both depending on example)
  // -------------------------------
  if(interfacial_temp_bc_requires_curvature){
    for(unsigned char d=0;d<2;++d){
      if (bc_interface_val_temp[d]!=NULL){
        bc_interface_val_temp[d]->clear_kappa_interp();
      }
    }
  }
  if(interfacial_temp_bc_requires_normal){
    for(unsigned char d=0;d<2;++d){
      if (bc_interface_val_temp[d]!=NULL){
        bc_interface_val_temp[d]->clear_normals_interp();
      }
    }
  }

  // -------------------------------
  // Destroy all vectors
  // that were used strictly for the
  // stefan step (aka created and destroyed in stefan step)
  // -------------------------------
  // Solid LSF:
  phi_solid.destroy();

  // Curvature and normal for BC's and setting up solver:
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature) normal.destroy();
  if(interfacial_temp_bc_requires_curvature) curvature.destroy();

  // Second derivatives of LSF's (for solver):
  phi_solid_dd.destroy();
  phi_dd.destroy();

  if(there_is_a_substrate){
    phi_substrate_dd.destroy();
  }

  if(solve_navier_stokes){
    T_l_backtrace_n.destroy();
    if(advection_sl_order ==2){
      T_l_backtrace_nm1.destroy();
    }
  }

  // Destroy arrays to hold the RHS:
  rhs_Tl.destroy();
  if(do_we_solve_for_Ts) rhs_Ts.destroy();

} // end of "setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc()"


// -------------------------------------------------------
// Functions related to the multicomponent problem: ( in order of their usage in the main step)
// -------------------------------------------------------
// TO-DO MULTICOMP: flesh this out
void my_p4est_stefan_with_fluids_t::setup_and_solve_multicomponent_problem(){}



// -------------------------------------------------------
// Functions related to computation of the interfacial velocity and timestep:
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::extend_relevant_fields(){

  compute_extension_bands_and_dxyz_close_to_interface();

  vec_and_ptr_t phi_solid, phi_solid_eff;
  vec_and_ptr_dim_t liquid_normals, solid_normals;

  // -------------------------------
  // Create all fields for this procedure:
  // -------------------------------
  phi_solid.create(phi.vec);
  if(there_is_a_substrate){
    //    phi_eff.create(phi.vec);
    phi_solid_eff.create(phi.vec);
  }
  liquid_normals.create(p4est_np1, nodes_np1);
  solid_normals.create(p4est_np1, nodes_np1);

  // -------------------------------
  // Get the solid LSF:
  // -------------------------------
  VecCopyGhost(phi.vec,phi_solid.vec);
  VecScaleGhost(phi_solid.vec,-1.0);

  // -------------------------------
  // Get the effective LSFs (if we are using a substrate):
  // -------------------------------
  if(there_is_a_substrate){
    // Only compute phi_solid_eff bc we create phi_eff in the main time loop!
    //    // For computing phi_effective:
    //    std::vector<Vec> phi_eff_list;
    //    std::vector<mls_opn_t> phi_eff_opn_list;

    //    phi_eff_list.push_back(phi.vec); phi_eff_list.push_back(phi_sub.vec);
    //    phi_eff_opn_list.push_back(MLS_INTERSECTION); phi_eff_opn_list.push_back(MLS_INTERSECTION);

    //    compute_phi_eff(phi_eff.vec, nodes_np1, phi_eff_list, phi_eff_opn_list);

    //    phi_eff_list.clear(); phi_eff_opn_list.clear();

    // For computing phi_solid_effective:
    std::vector<Vec> phi_solid_eff_list;
    std::vector<mls_opn_t> phi_solid_eff_opn_list;

    phi_solid_eff_list.push_back(phi_solid.vec); phi_solid_eff_list.push_back(phi_substrate.vec);
    phi_solid_eff_opn_list.push_back(MLS_INTERSECTION); phi_solid_eff_opn_list.push_back(MLS_INTERSECTION);

    compute_phi_eff(phi_solid_eff.vec, nodes_np1, phi_solid_eff_list, phi_solid_eff_opn_list);

    phi_solid_eff_list.clear(); phi_solid_eff_opn_list.clear();

  }


  // -------------------------------
  // Compute normals for each domain:
  // -------------------------------

  compute_normals(*ngbd_np1, (there_is_a_substrate? phi_eff.vec : phi.vec), liquid_normals.vec);
  compute_normals(*ngbd_np1, (there_is_a_substrate? phi_solid_eff.vec : phi_solid.vec), solid_normals.vec);

  if(0){
    // -------------------------------
    // TEMPORARY: save phi_eff fields to see what we are working with
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(phi_eff.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(phi_eff.vec, "phi_solid"));
    point_fields.push_back(Vec_for_vtk_export_t(phi_eff.vec, "phi_eff"));
    point_fields.push_back(Vec_for_vtk_export_t(phi_solid_eff.vec, "phi_solid_eff"));


    const char* out_dir = getenv("OUT_DIR_VTK");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }
    //          char output[] = "/home/elyce/workspace/projects/multialloy_with_fluids/output_two_grain_clogging/gradP_0pt01_St_0pt07/grid57_flush_no_collapse_after_extension_bc_added";
    char filename[1000];
    sprintf(filename, "%s/snapshot_extend_phi_effs_%d", out_dir, tstep);
    my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ngbd_np1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();

  }

  // -------------------------------
  // Extend Temperature Fields across the interface:
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Calling extension over phi \n");

  // Extend liquid temperature:
  ls->extend_Over_Interface_TVD_Full((there_is_a_substrate? phi_eff.vec : phi.vec), T_l_n.vec,
                                    50, 2,
                                    extension_band_use_, extension_band_extend_,
                                    liquid_normals.vec, NULL,NULL, false, NULL,NULL);

  // Extend solid temperature:
  if(do_we_solve_for_Ts){
      ls->extend_Over_Interface_TVD_Full((there_is_a_substrate? phi_solid_eff.vec : phi_solid.vec), T_s_n.vec,
                                         50, 2,
                                         extension_band_use_, extension_band_extend_,
                                         solid_normals.vec, NULL, NULL, false, NULL, NULL);
  }

  // -------------------------------
  // Destroy all fields that were created for the procedure:
  // -------------------------------
  phi_solid.destroy();
  liquid_normals.destroy();
  solid_normals.destroy();
  if(there_is_a_substrate){
    //    phi_eff.destroy();
    phi_solid_eff.destroy();
  }
} // end of "extend_relevant_fields()"

double my_p4est_stefan_with_fluids_t::interfacial_velocity_expression(double Tl_d, double Ts_d){
  switch(problem_dimensionalization_type){
  // Note: removed curvature from Stefan condition after discussing w frederic and looking at Daniil's thesis 11/24/2020
  case NONDIM_BY_FLUID_VELOCITY:{
    if(!is_dissolution_case){
      return ( -1.*(St/Pe)*(alpha_s/alpha_l) * ((k_l/k_s)*Tl_d - Ts_d) );
    }
    else{
      return -1.*(gamma_diss/Pe)*Tl_d;
    }
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      return ( -1.*(St)*(alpha_s/alpha_l)*( (k_l/k_s)*Tl_d - Ts_d ) );
    }
    else{
      return -1.*gamma_diss*Tl_d;
    }
  }
  case DIMENSIONAL:{
    if(!is_dissolution_case){
      return (k_s*Ts_d -k_l*Tl_d)/(L*rho_s);
    }
    else{
      return -1.*molar_volume_diss*(Dl/stoich_coeff_diss)*Tl_d;
    }
  }

  default:{
    throw std::invalid_argument("interfacial_velocity_expression: Unrecognized stefan condition type case \n");
  }
  }
} // end of "interfacial_velocity_expression()"

bool my_p4est_stefan_with_fluids_t::compute_interfacial_velocity(){
  // Some vec_and_ptrs owned by this fxn:
  vec_and_ptr_t vgamma_n;

  // Begin calculation:
  if(!force_interfacial_velocity_to_zero){
    // Cut the extension band in half for region to actually compute vgamma:
    extension_band_extend_/=2;


    // Get the first derivatives to compute the jump
    T_l_d.create(p4est_np1, nodes_np1);
    ngbd_np1->first_derivatives_central(T_l_n.vec, T_l_d.vec);

    if(do_we_solve_for_Ts){
      T_s_d.create(T_l_d.vec);
      ngbd_np1->first_derivatives_central(T_s_n.vec, T_s_d.vec);
    }

    // Create vgamma and normals, and compute normals:
    vgamma_n.create(p4est_np1, nodes_np1);
    normal.create(p4est_np1, nodes_np1);
    // TO-DO: not sure how important this is, but it's possible we compute normals 2x per timestep in some cases which is not very efficient. Consider changing this later. Would need to be handled in main loop.
    // Could add a boolean flag for (are_normals_computed) or something
    compute_normals(*ngbd_np1, phi.vec, normal.vec);

    // Create vector to hold the jump values:
    jump.create(p4est_np1, nodes_np1);

    // Get arrays:
    normal.get_array();
    vgamma_n.get_array();
    jump.get_array();
    T_l_d.get_array();
    if(do_we_solve_for_Ts) T_s_d.get_array();
    phi.get_array();

    // First, compute jump in the layer nodes:
    for(size_t i=0; i<ngbd_np1->get_layer_size();i++){
      p4est_locidx_t n = ngbd_np1->get_layer_node(i);

      if(fabs(phi.ptr[n])<extension_band_extend_){ // TO-DO: should be nondim for ALL cases

        vgamma_n.ptr[n] = 0.; // Initialize
        foreach_dimension(d){
          jump.ptr[d][n] = interfacial_velocity_expression(T_l_d.ptr[d][n], do_we_solve_for_Ts?T_s_d.ptr[d][n]:0.);

          // Calculate V_gamma,n using dot product:
          vgamma_n.ptr[n] += jump.ptr[d][n] * normal.ptr[d][n];
        } // end of loop over dimensions

        // Now, go back and set jump equal to the enforced normal velocity (a scalar) multiplied by the normal --> to get a velocity vector:
        foreach_dimension(d){
          jump.ptr[d][n] = vgamma_n.ptr[n] * normal.ptr[d][n];
        }
      }
    }

    // Begin updating the ghost values of the layer nodes:
    foreach_dimension(d){
      VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }

    // Compute the jump in the local nodes:
    for(size_t i = 0; i<ngbd_np1->get_local_size();i++){
      p4est_locidx_t n = ngbd_np1->get_local_node(i);
      if(fabs(phi.ptr[n])<extension_band_extend_){
        vgamma_n.ptr[n] = 0.; // initialize
        foreach_dimension(d){
          jump.ptr[d][n] = interfacial_velocity_expression(T_l_d.ptr[d][n], do_we_solve_for_Ts?T_s_d.ptr[d][n]:0.);

          // calculate the dot product to find V_gamma,n
          vgamma_n.ptr[n] += jump.ptr[d][n] * normal.ptr[d][n];

        } // end over loop on dimensions

        // Now, go back and set jump equal to the enforced normal velocity (a scalar) multiplied by the normal --> to get a velocity vector:
        foreach_dimension(d){
          jump.ptr[d][n] = vgamma_n.ptr[n] * normal.ptr[d][n];
        }
      }
    }

    // Finish updating the ghost values of the layer nodes:
    foreach_dimension(d){
      VecGhostUpdateEnd(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }

    // Restore arrays:
    jump.restore_array();
    T_l_d.restore_array();
    if(do_we_solve_for_Ts) T_s_d.restore_array();

    // Elyce trying something:
    normal.restore_array();
    normal.destroy();
    vgamma_n.restore_array();
    vgamma_n.destroy();


    // Extend the interfacial velocity to the whole domain for advection of the LSF:
    foreach_dimension(d){
      ls->extend_from_interface_to_whole_domain_TVD((there_is_a_substrate? phi_eff.vec : phi.vec),
                                                   jump.vec[d], v_interface.vec[d]); // , 20/*, NULL, 2., 4.*/);
    }


    // Set to zero if we are inside the substrate:
    if(there_is_a_substrate){
      phi_substrate.get_array();
      v_interface.get_array();

      // Layer nodes:
      for(size_t i=0; i<ngbd_np1->get_layer_size();i++){
        p4est_locidx_t n = ngbd_np1->get_layer_node(i);

        foreach_dimension(d){
          if(phi_substrate.ptr[n]>0.){
            v_interface.ptr[d][n] = 0.;
          }
        }
      }

      // Begin communication:
      // Finish updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateBegin(v_interface.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
      // Local nodes:
      for(size_t i = 0; i<ngbd_np1->get_local_size();i++){
        p4est_locidx_t n = ngbd_np1->get_local_node(i);
        foreach_dimension(d){
          if(phi_substrate.ptr[n]>0.){
            v_interface.ptr[d][n] = 0.;
          }
        }
      }
      // End communication:
      // Finish updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateEnd(v_interface.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
      v_interface.get_array();
      phi_substrate.restore_array();
    }

    // Scale v_interface computed by appropriate sign if we are doing the coupled test case:
    if(fabs(scale_vgamma_by - 1.)>EPS){
      foreach_dimension(d){
        VecScaleGhost(v_interface.vec[d], scale_vgamma_by);
      }
    }

    // Destroy values once no longer needed:
    T_l_d.destroy();
    if(do_we_solve_for_Ts) T_s_d.destroy();
    jump.destroy();

  }
  else{ // Case where we are forcing interfacial velocity to zero
    foreach_dimension(d){
      VecScaleGhost(v_interface.vec[d], 0.0);
    }
  }


  bool did_crash = false;
  if(v_interface_max_norm>v_interface_max_allowed){
    did_crash=true;
  }
  return did_crash;


} // end of "compute_interfacial_velocity()"

void my_p4est_stefan_with_fluids_t::compute_timestep(){

  // Initialize variables and set max vint if known:
  double max_v_norm = 0.0;
  double global_max_vnorm = 0.0;

  // Compute dt_Stefan (and interfacial velocity if needed)
  if(solve_stefan){
    // Check the values of v_interface locally:
    v_interface.get_array();
    phi.get_array();
    foreach_local_node(n, nodes_np1){
      if (fabs(phi.ptr[n]) < uniform_band*dxyz_close_to_interface){
        max_v_norm = MAX(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
      }
    }
    v_interface.restore_array();
    phi.restore_array();

    // Get the maximum v norm across all the processors:
    int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est_np1->mpicomm);
    SC_CHECK_MPI(mpi_ret);


    // Compute new Stefan timestep:
    dt_Stefan = cfl_Stefan*MIN(dxyz_smallest[0], dxyz_smallest[1])/global_max_vnorm;
//    std::cout<< "Plotting contents of dt_Stefan\n";
//    std::cout<< "cfl stefan :: " << cfl_Stefan <<"\n";
//    std::cout<< "dxy_min :: " << MIN(dxyz_smallest[0], dxyz_smallest[1]) <<"\n";;
//    std::cout<< "velocity max norm " << global_max_vnorm <<"\n";

  } // end of if solve stefan

  // Compute dt_NS if necessary
  if(solve_navier_stokes){
    ns->compute_dt();
    dt_NS = ns->get_dt();
  }


  // Compute the timestep that will be used depending on what physics we have:
  if(solve_stefan && solve_navier_stokes){
    // Take the minimum timestep of the NS and Stefan (dt_Stefan computed previously):
    dt = MIN(dt_Stefan, dt_NS);
    dt = MIN(dt, dt_max_allowed);
  }
  else if(solve_stefan && !solve_navier_stokes){
    dt = MIN(dt_Stefan, dt_max_allowed);
  }
  else if(!solve_stefan && solve_navier_stokes){
    dt = MIN(dt_NS, dt_max_allowed);
  }
  else{
    throw std::runtime_error("setting the timestep : you are not solving any of the possible physics ... \n");
  }
//  std::cout<< "dt_stefan " <<dt_Stefan<<"\n";
//  std::cout<< "dt_NS" << dt_NS<<"\n";
  v_interface_max_norm = global_max_vnorm;
  // std::exit(EXIT_FAILURE);
  // TO-DO: want to move last tstep and clipping business to outside of the class,
  // don't forget to make that consistent in the main

//  // Clip the timestep if we are near the end of our simulation, to get the proper end time:
//  PetscPrintf(mpi->comm(), "tn = %f, tfinal = %f \n", tn, tfinal);
//  if((tn + dt > tfinal) && (last_tstep<0)){

//    dt = MAX(tfinal - tn,dt_min_allowed);

//    // if time remaining is too small for one more step, end here. otherwise, do one more step and clip timestep to end on exact ending time
//    if(fabs(tfinal-tn)>dt_min_allowed){
//      last_tstep = tstep+1;
//    }
//    else{
//      last_tstep = tstep;
//    }

//    PetscPrintf(mpi->comm(),"Final tstep will be %d \n",last_tstep);
//  }

  // TO-DO: remove temp failsafe below w something cleverer?
  if(dt<dt_min_allowed){
    dt = dt_min_allowed;
  }
//  std::cout<< "dt" << dt<<"\n";
//  std::cout<< "dt_min_allowed" << dt_min_allowed<<"\n";
  // Print the interface velocity info:
  PetscPrintf(mpi->comm(),"\n"
                       "Computed interfacial velocity: \n"
                       " - %0.3e [nondim] "
                       " - %0.3e [m/s] "
                       " - %0.3e [mm/s] \n",
              v_interface_max_norm,
              v_interface_max_norm*vel_nondim_to_dim,
              v_interface_max_norm*vel_nondim_to_dim*1000.);

  // Print the timestep info:
  PetscPrintf(mpi->comm(),"\n"
                       "Computed timestep: \n"
                       " - dt used: %0.3e "
                       " - dt_Stefan: %0.3e "
                       " - dt_NS : %0.3e  "
                       " - dt_max_allowed : %0.3e \n"
                       " - dxyz close to interface : %0.3e "
                       "\n",
              dt, dt_Stefan, dt_NS, dt_max_allowed,
              dxyz_close_to_interface);
} // end of "compute_timestep()"


// -------------------------------------------------------
// Functions related to Navier-Stokes problem:
// -------------------------------------------------------

void my_p4est_stefan_with_fluids_t::set_ns_parameters(){
  switch(problem_dimensionalization_type){
  case NONDIM_BY_FLUID_VELOCITY:{
    ns->set_parameters((1./Re), 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    break;
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      ns->set_parameters(Pr, 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    }
    else{
      ns->set_parameters(Sc, 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    }
    break;
  }
  case DIMENSIONAL:{
    ns->set_parameters(mu_l, rho_l, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
  }
  }// end switch case

} // end of "set_ns_parameters()"

void my_p4est_stefan_with_fluids_t::initialize_ns_solver(bool convert_to_nondim_for_multialloy){

  // Create the initial neigbhors and faces (after first step, NS grid update will handle this internally
  ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);


  faces_np1 = new my_p4est_faces_t(p4est_np1, ghost_np1, &brick, ngbd_c_np1);

  // Create the solver
  ns = new my_p4est_navier_stokes_t(ngbd_n,ngbd_np1,faces_np1);

  // Set the LSF:
  ns->set_phi((there_is_a_substrate ? phi_eff.vec:phi.vec));

  ns->set_dt(dt_nm1,dt);

  if(convert_to_nondim_for_multialloy){
    // Convert dimensional to nondimensional
    foreach_dimension(d){
      ierr = VecScaleGhost(v_n.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
      ierr = VecScaleGhost(v_nm1.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
    }
  }

  ns->set_velocities(v_nm1.vec, v_n.vec);

  if(convert_to_nondim_for_multialloy){
    // Convert dimensional to nondimensional
    foreach_dimension(d){
      ierr = VecScaleGhost(v_n.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
      ierr = VecScaleGhost(v_nm1.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
    }
  }

  /*
  // To-do: move this switch case to the set parameters fxn since it makes more sense to have it there
  switch(problem_dimensionalization_type){
  case NONDIM_BY_FLUID_VELOCITY:
    PetscPrintf(mpi->comm(),"NS solver initialization: CFL_NS: %0.2f, rho position : %0.2f, mu position : %0.3e \n",
                cfl_NS, 1. , 1./Re);
    break;
  case NONDIM_BY_SCALAR_DIFFUSIVITY:
    PetscPrintf(mpi->comm(),"NS solver initialization: CFL_NS: %0.2f, rho position : %0.2f, mu position : %0.3e \n", cfl_NS, 1., is_dissolution_case? Sc:Pr);
    break;
  case DIMENSIONAL:
    PetscPrintf(mpi->comm(),"NS solver initialization: CFL_NS: %0.2f, rho position : %0.2f, mu position : %0.3e \n", cfl_NS, rho_l,mu_l);
    break;
  default:
    break;
  }

  // Use a function to set ns parameters to avoid code duplication
  set_ns_parameters();

  // Set an initial norm for the first hodge iteration
  NS_norm = NS_max_allowed;

  PetscPrintf(mpi->comm(), "NS norm max allowed = %f, NS norm = %f \n", NS_max_allowed, NS_norm);
  */
} // end of "initialize_ns_solver()"

bool my_p4est_stefan_with_fluids_t::navier_stokes_step(){

//  std:: cout<< "Hello world ns 1 \n";
  // Destroy old pressure at nodes (if it exists) and create vector to hold new solns:
  press_nodes.destroy(); press_nodes.create(p4est_np1, nodes_np1);

//  std:: cout<< "Hello world ns 2 \n";

  // Create vector to store old dxyz hodge:
  for (unsigned char d=0; d<P4EST_DIM; d++){
    ierr = VecCreateNoGhostFaces(p4est_np1, faces_np1, &dxyz_hodge_old[d], d); CHKERRXX(ierr);
  }
  //std:: cout<< "Hello world ns 3 \n";
  //std:: cout<< "Hello world ns 4 NS norm: " << NS_norm<<"\n";
  //std:: cout<< "Hello world ns 4 NS hodge% of max: " << hodge_percentage_of_max_u<<"\n";
//  std:: cout<< "Hello world ns 4 NS hodge tolerance: " << hodge_tolerance<<"\n";
  hodge_tolerance = NS_norm*hodge_percentage_of_max_u;
  //std:: cout<< "Hello world ns 4 NS hodge tolerance: " << hodge_tolerance<<"\n";
  //qPetscPrintf(mpi->comm(),"Hodge tolerance is %e \n",hodge_tolerance);
  //std:: cout<< "Hello world ns 5 \n";
  int hodge_iteration = 0;
  double convergence_check_on_dxyz_hodge = DBL_MAX;
  //std:: cout<< "Hello world ns 6 \n";
  face_solver = NULL;
  cell_solver = NULL;
  //std:: cout<< "Hello world ns 7 \n";
  // Update the parameters: (this is only done to update the cfl potentially)
  // Use a function to set ns parameters to avoid code duplication
  set_ns_parameters();
  //std:: cout<< "Hello world ns 8 \n";
  //std:: cout<< "Hello world ns 8" << hodge_max_it<<" \n";

  // Enter the loop on the hodge variable and solve the NS equations
  while((hodge_iteration<hodge_max_it) && (convergence_check_on_dxyz_hodge>hodge_tolerance)){
//    std:: cout<< "Hello world ns 8 HODGE iteration " << hodge_iteration<<" \n";
//    std:: cout<< "Hello world ns 8 CONVERGENCE CHECK ON DXYZ HODGE" << hodge_tolerance<<" \n";
    ns->copy_dxyz_hodge(dxyz_hodge_old);
//    std:: cout<< "Hello world ns 8_1 \n";
    ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);
//    std:: cout<< "Hello world ns 8_2 \n";
    //std:: cout<< "Hello world ns 9 \n";
    convergence_check_on_dxyz_hodge=
        ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell,
                             false,NULL,dxyz_hodge_old,uvw_components);
//    std:: cout<< "Hello world ns 10 \n";
    //std:: cout << "Hodge iteration :: " << hodge_iteration <<"\n";
    //std:: cout << "convergence check :: " << convergence_check_on_dxyz_hodge <<"\n";
    //std:: cout << " NS_norm :: "<< NS_norm <<"\n";
    //std:: cout << " mpi  :: "<< mpi <<"\n";
    ierr= PetscPrintf(mpi->comm(),"Hodge iteration : %d, (hodge error)/(NS_max): %0.3e \n",hodge_iteration,convergence_check_on_dxyz_hodge/NS_norm);CHKERRXX(ierr);
    hodge_iteration++;
  }
//  std:: cout<< "Hello world ns 11 \n";
  //ierr = PetscPrintf(mpi->comm(), "Hodge loop exited \n");
  //std:: cout<< "Hello world ns 12 \n";
  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecDestroy(dxyz_hodge_old[d]); CHKERRXX(ierr);
  }
//  std:: cout<< "Hello world ns 13 \n";
  // Delete solvers:
  delete face_solver;
  delete cell_solver;

//  std:: cout<< "Hello world ns 14\n";
  // Compute velocity at the nodes
  ns->compute_velocity_at_nodes();
  // ------------------------
  // Slide velocity fields (for our use):
  // ------------------------
  // (a) get rid of old vnm1, now vn becomes the new vnm1
  // (b) no need to destroy vn, bc now we put vnp1 into vn's slot
  v_nm1.destroy();

  foreach_dimension(d){
    ns->get_node_velocities_n(v_nm1.vec[d], d);
    ns->get_node_velocities_np1(v_n.vec[d], d);
  }

  // ------------------------
  // Compute the pressure
  if(compute_pressure_){
    ns->compute_pressure(); // note: only compute pressure at nodes when we are saving to VTK (or evaluating some errors)
    ns->compute_pressure_at_nodes(&press_nodes.vec);
  }
  // Get the computed values of vorticity
  vorticity.vec = ns->get_vorticity();

  // Check the L2 norm of u to make sure nothing is blowing up
  NS_norm = ns->get_max_L2_norm_u();
  PetscPrintf(mpi->comm(),"\n Max NS velocity norm: \n"
                        " - Computational value: %0.4f  "
                        " - Physical value: %0.3e [m/s]  "
                        " - Physical value: %0.3f [mm/s] \n",NS_norm,NS_norm*vel_nondim_to_dim,NS_norm*vel_nondim_to_dim*1000.);

  // Stop simulation if things are blowing up
  bool did_crash;
  if(NS_norm>NS_max_allowed){
    MPI_Barrier(mpi->comm());
    PetscPrintf(mpi->comm(),"The simulation blew up ! ");
    did_crash = true;
  }
  else{
    did_crash = false;
  }
  return did_crash;


} // end of "navier_stokes_step()"

void my_p4est_stefan_with_fluids_t::setup_and_solve_navier_stokes_problem(bool use_external_boussinesq_vec, Vec externally_defined_boussinesq_vec, bool convert_to_nondim_for_multialloy){
  //std:: cout<< "Hello world 1 \n";

//  int vnsize;
//  VecGetSize(v_n.vec[0], &vnsize);
//  PetscPrintf(p4est_np1->mpicomm, "vn size before setup and solve navier stokes problem = %d \n", vnsize);

//  int num_nodesn = nodes_n->num_owned_indeps;
//  MPI_Allreduce(MPI_IN_PLACE,&num_nodesn,1,MPI_INT,MPI_SUM, p4est_n->mpicomm);
//  PetscPrintf(p4est_n->mpicomm, "!!! stefan w fluids nodes_n: %d \n", num_nodesn);

//  int num_nodesnp1 = nodes_np1->num_owned_indeps;
//  MPI_Allreduce(MPI_IN_PLACE,&num_nodesnp1,1,MPI_INT,MPI_SUM, p4est_np1->mpicomm);
//  PetscPrintf(p4est_np1->mpicomm, "!!! stefan w fluids nodes_np1: %d \n", num_nodesnp1);

  if(convert_to_nondim_for_multialloy){
    // Convert dimensional to nondimensional
    foreach_dimension(d){
      ierr = VecScaleGhost(v_n.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
      ierr = VecScaleGhost(v_nm1.vec[d], 1./vel_nondim_to_dim);CHKERRXX(ierr);
    }
  }

  // -------------------------------
  // Set the NS timestep:
  // -------------------------------
  if(advection_sl_order ==2){
    ns->set_dt(dt_nm1,dt);
  }
  else{
    ns->set_dt(dt);
  }
//  std:: cout<< "Hello world 2 \n";
  // -------------------------------
  // Update BC and RHS objects for navier-stokes problem:
  // -------------------------------
  // NOTE: we update NS grid first, THEN set new BCs and forces. This is because the update grid interpolation of the hodge variable
  // requires knowledge of the boundary conditions from that same timestep (the previous one, in our case)
  // -------------------------------
  // Setup velocity conditions
  for(unsigned char d=0;d<P4EST_DIM;d++){
    if(interfacial_vel_bc_requires_vint){
      bc_interface_value_velocity[d]->set(ngbd_np1, v_interface.vec[d]);
    }
    bc_velocity[d].setInterfaceType(*bc_interface_type_velocity[d]);
    bc_velocity[d].setInterfaceValue(*bc_interface_value_velocity[d]);
    bc_velocity[d].setWallValues(*bc_wall_value_velocity[d]);
    bc_velocity[d].setWallTypes(*bc_wall_type_velocity[d]);
  }
//  std:: cout<< "Hello world 3 \n";
  // Setup pressure conditions:
  bc_pressure.setInterfaceType(bc_interface_type_pressure);
  bc_pressure.setInterfaceValue(*bc_interface_value_pressure);
  bc_pressure.setWallTypes(*bc_wall_type_pressure);
  bc_pressure.setWallValues(*bc_wall_value_pressure);
//  std:: cout<< "Hello world 4 \n";
  // -------------------------------
  // Set BC's and external forces if relevant
  // (note: these are actually updated in the fxn dedicated to it, aka setup_analytical_ics_and_bcs_for_this_tstep() )
  // -------------------------------
  // Set the boundary conditions:
  ns->set_bc(bc_velocity,&bc_pressure);
//  std:: cout<< "Hello world 5 \n";
  // Set the RHS:
  if(there_is_user_provided_external_force_NS){
    ns->set_external_forces(user_provided_external_forces_NS);
  }
//  std:: cout<< "Hello world 6 \n";
  // -------------------------------
  // Handle the Boussinesq case setup for the RHS, if relevant:
  // ---------------------------
  // ALERT: at this time, we assume that if the boussinesq approx is activated, there cannot also be user defined external forces provided by a CF.
  // To-do: fix this at some point
  if(use_boussinesq && (!there_is_user_provided_external_force_NS)){
    switch(problem_dimensionalization_type){
    case NONDIM_BY_FLUID_VELOCITY:{
      ns->boussinesq_approx=true;
      ierr = VecScaleGhost(T_l_n.vec, -1.);
      ns->set_external_forces_using_vector(T_l_n.vec);
      ierr = VecScaleGhost(T_l_n.vec, -1.);
      break;
    }
    case NONDIM_BY_SCALAR_DIFFUSIVITY:{
      ns->boussinesq_approx=true;
      if(!is_dissolution_case){
        ierr = VecScaleGhost(T_l_n.vec, -1.*RaT*Pr);
        ns->set_external_forces_using_vector(T_l_n.vec);
        ierr = VecScaleGhost(T_l_n.vec, -1./(RaT*Pr));
      }
      else{
        // Elyce to-do: 12/15/21 - this is a work in process, havent nailed down nondim def yet
        ierr = VecScaleGhost(T_l_n.vec, -1.*RaC*Sc);
        ns->set_external_forces_using_vector(T_l_n.vec);
        ierr = VecScaleGhost(T_l_n.vec, -1./(RaC*Sc));
      }

      break;
    }
    case DIMENSIONAL:{
      throw std::invalid_argument("AHHHHH!!! this is not fully developed yet. don't use this setup with natural convection \n");

      break;
    }
    default:{
      throw std::runtime_error("setting natural convection -- unrecognized problem dimensionalization formulation \n");
    }
    }
  }

  // This case right now is only being used by the multialloy w/ fluids project
  if(use_external_boussinesq_vec){
    ns->boussinesq_approx=true;
    ns->set_external_forces_using_vector(externally_defined_boussinesq_vec);

  }

//  std:: cout<< "Hello world 7 \n";
  // -------------------------------
  // Solve the Navier-Stokes problem:
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning Navier-Stokes solution step... \n");

//  printf("\n Addresses of vns vectors (inside SWF): \n "
//         "v_n.vec = %p, v_nm1.vec = %p \n", v_n.vec[0], v_nm1.vec[0]);

  // Check if we are going to be saving to vtk for the next timestep... if so, we will compute pressure at nodes for saving

  bool did_crash = navier_stokes_step();
  // -------------------------------
  // Clear out the interfacial BC for the next timestep, if needed
  // -------------------------------
  if(interfacial_vel_bc_requires_vint){
    for(unsigned char d=0;d<P4EST_DIM;d++){
      bc_interface_value_velocity[d]->clear();
    }
  }
//  std:: cout<< "Hello world 10 \n";
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Completed Navier-Stokes step \n");

  // Convert back to dimensional units for multialloy if relevant
  if(convert_to_nondim_for_multialloy){
    // Convert nondimensional result back to dimensional
    foreach_dimension(d){
      ierr = VecScaleGhost(v_n.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
      ierr = VecScaleGhost(v_nm1.vec[d], vel_nondim_to_dim);CHKERRXX(ierr);
    }
  }

  if(did_crash){
    char crash_tag[10];
    sprintf(crash_tag, "NS");
    save_fields_to_vtk(0, did_crash, crash_tag);
    throw std::runtime_error("The Navier Stokes step crashed \n");
  }
  //std:: cout<< "Hello world 11 \n";
//  int vnsize2;
//  VecGetSize(v_n.vec[0], &vnsize2);
//  PetscPrintf(p4est_np1->mpicomm, "vn size after setup and solve navier stokes problem = %d \n", vnsize2);

} // end of "setup_and_solve_navier_stokes_problem()"

// -------------------------------------------------------
// Functions related to LSF advection/grid update:
// -------------------------------------------------------

void my_p4est_stefan_with_fluids_t::prepare_refinement_fields(){
  PetscErrorCode ierr;

  // Get relevant arrays:
  if(refine_by_vorticity){
    vorticity.get_array();
    vorticity_refine.get_array();
  }
  if(refine_by_d2T) {T_l_dd.get_array();}
  phi.get_array();

  // Compute proper refinement fields on layer nodes:
  for(size_t i = 0; i<ngbd_n->get_layer_size(); i++){
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    if(phi.ptr[n] < 0.){
      if(refine_by_vorticity)vorticity_refine.ptr[n] = vorticity.ptr[n];
    }
    else{
      if(refine_by_vorticity) vorticity_refine.ptr[n] = 0.0;
      if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
        foreach_dimension(d){
          T_l_dd.ptr[d][n]=0.;
        }
      }
    }
  } // end of loop over layer nodes

  // Begin updating the ghost values:
  if(refine_by_vorticity)ierr = VecGhostUpdateBegin(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateBegin(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  //Compute proper refinement fields on local nodes:
  for(size_t i = 0; i<ngbd_n->get_local_size(); i++){
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    if(phi.ptr[n] < 0.){
      if(refine_by_vorticity)vorticity_refine.ptr[n] = vorticity.ptr[n];
    }
    else{
      if(refine_by_vorticity)vorticity_refine.ptr[n] = 0.0;
      if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
        foreach_dimension(d){
          T_l_dd.ptr[d][n]=0.;
        }
      }
    }
  } // end of loop over local nodes

  // Finish updating the ghost values:
  if(refine_by_vorticity)ierr = VecGhostUpdateEnd(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateEnd(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  // Restore appropriate arrays:
  if(refine_by_d2T) {T_l_dd.restore_array();}
  if(refine_by_vorticity){
    vorticity.restore_array();
    vorticity_refine.restore_array();
  }
  phi.restore_array();
} // end of "prepare_refinement_fields()"


void my_p4est_stefan_with_fluids_t::perform_reinitialization(){
  // Time to reinitialize in a clever way depending on the scenario:

  if(solve_stefan && !force_interfacial_velocity_to_zero){
    // If interface velocity *is* forced to zero, we do not reinitialize -- that way we don't degrade the LSF through unnecessary reinitializations



    // There are some cases where we may not want to reinitialize after every time iteration
    // i.e.) In the coupled case, if fluid velocity is much larger than interfacial velocity, may not need to reinitialize as much
    // (bc the timestepping is much smaller than necessary for the interface growth, and we don't want the reinitialization to govern more of the interface change than the actual physical interface change)
    // For this reason, we have the user option of reinit_every_iter (which is default set to 1)

    if((tstep % reinit_every_iter) == 0){
      ls->reinitialize_2nd_order(phi.vec);
      PetscPrintf(mpi->comm(), "reinit every iter =%d, LSF was reinitialized \n", reinit_every_iter);
    }
  }
  else{
    // If only solving Navier-Stokes, or just no interface motion, only need to do this once, not every single timestep
    if(tstep==0) ls->reinitialize_2nd_order(phi.vec);
  }
} // end of "perform_reinitialization()"


void my_p4est_stefan_with_fluids_t::refine_and_coarsen_grid_and_advect_lsf_if_applicable(){
  // ------------------------------------------------------------
  // Define the things needed for the refinement/coarsening tool:
  // ------------------------------------------------------------
  if(!solve_stefan) refine_by_d2T=false; // override settings if there *is* no temperature field

  bool use_block = false;
  bool expand_ghost_layer = true;

  std::vector<compare_option_t> compare_opn;
  std::vector<compare_diagonal_option_t> diag_opn;
  std::vector<double> criteria;
  std::vector<int> custom_lmax;

  PetscInt num_fields = 0;
  refine_by_vorticity = solve_navier_stokes && vorticity.vec!=NULL;
  // -----------------------
  // Count number of refinement fields and create vectors for necessary fields:
  // ------------------------
  if(refine_by_vorticity) {
    num_fields+=1;
    vorticity_refine.create(p4est_n, nodes_n);
  }// for vorticity
  if(refine_by_d2T){
    num_fields+=2;
    T_l_dd.create(p4est_n, nodes_n);
    ngbd_n->second_derivatives_central(T_l_n.vec,T_l_dd.vec);
  } // for second derivatives of temperature

  // Create array of fields we wish to refine by, to pass to the refinement tools
  Vec fields_[num_fields];

  // ------------------------------------------------------------
  // Begin preparing the refine/coarsen criteria:
  // ------------------------------------------------------------
  if(num_fields>0){
    // ------------------------------------------------------------
    // Prepare refinement fields:
    // ------------------------------------------------------------
    prepare_refinement_fields();

    // ------------------------------------------------------------
    // Add our refinement fields to the array:
    // ------------------------------------------------------------
    PetscInt fields_idx = 0;
    if(refine_by_vorticity)fields_[fields_idx++] = vorticity_refine.vec;
    if(refine_by_d2T){
      fields_[fields_idx++] = T_l_dd.vec[0];
      fields_[fields_idx++] = T_l_dd.vec[1];
    }

    P4EST_ASSERT(fields_idx ==num_fields);

    // ------------------------------------------------------------
    // Add our instructions:
    // ------------------------------------------------------------
    // Coarsening instructions: (for vorticity)
    if(refine_by_vorticity){
      compare_opn.push_back(LESS_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm/2.);

      // Refining instructions: (for vorticity)
      compare_opn.push_back(GREATER_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm);

      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}
    }
    if(refine_by_d2T){
      double dxyz_smallest[P4EST_DIM];
      dxyz_min(p4est_n,dxyz_smallest);

      double dTheta= fabs(theta_infty - theta_interface)>0 ? fabs(theta_infty - theta_interface): 1.0;
      dTheta/=SQR(MIN(dxyz_smallest[0],dxyz_smallest[1])); // max d2Theta in liquid subdomain

      // Define variables for the refine/coarsen instructions for d2T fields:
      compare_diagonal_option_t diag_opn_d2T = DIVIDE_BY;
      compare_option_t compare_opn_d2T = SIGN_CHANGE;
      double refine_criteria_d2T = dTheta*d2T_threshold;
      double coarsen_criteria_d2T = dTheta*d2T_threshold*0.1;

      // Coarsening instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T); // did 0.1* () for the coarsen if no sign change OR below threshold case

      // Refining instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}

      // Coarsening instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T);

      // Refining instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}
    }
  } // end of "if num_fields!=0"

  // -------------------------------
  // Call grid advection and update:
  // -------------------------------

  if(solve_stefan){
    // Create second derivatives for phi in the case that we are using update_p4est:
    phi_dd.create(p4est_n, nodes_n);
    ngbd_n->second_derivatives_central(phi.vec, phi_dd.vec);

//    if(there_is_a_substrate){
//            if(start_w_merged_grains){regularize_front(p4est, nodes, ngbd, phi_substrate.vec);}
//    }

    // Call advection and refinement
    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n);
    sl.update_p4est(v_interface.vec, dt,
                    phi.vec, phi_dd.vec, there_is_a_substrate ? phi_substrate.vec: NULL,
                    num_fields, use_block, true,
                    uniform_band, uniform_band*(1.5),
                    fields_, NULL,
                    criteria, compare_opn, diag_opn, custom_lmax,
                    expand_ghost_layer);

    if(print_checkpoints) PetscPrintf(mpi->comm(),"Grid update completed \n");

    // Destroy 2nd derivatives of LSF now that not needed
    phi_dd.destroy();

  } // case for stefan or coupled
  else {
    // NS only case --> no advection --> do grid update iteration manually:
    splitting_criteria_tag_t sp_NS(lmin, lmax, sp->lip);

    // Create a new vector which will hold the updated values of the fields -- since we will interpolate with each grid iteration
    Vec fields_new_[num_fields];
    if(num_fields!=0)
    {
      for(int k = 0; k<num_fields; k++){
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &fields_new_[k]);
        ierr = VecCopyGhost(fields_[k],fields_new_[k]);
      }
    }

    // Create a vector which will hold the updated values of the LSF:
    vec_and_ptr_t phi_new;
    phi_new.create(p4est_n, nodes_n);
    ierr = VecCopyGhost(phi.vec, phi_new.vec);

    bool is_grid_changing = true;
    int no_grid_changes = 0;
    bool last_grid_balance = false;
    while(is_grid_changing){
      if(!last_grid_balance){
        is_grid_changing = sp_NS.refine_and_coarsen(p4est_np1, nodes_np1, phi_new.vec,
                                                    num_fields, use_block, true,
                                                    uniform_band, uniform_band*1.5,
                                                    fields_new_, NULL, criteria,
                                                    compare_opn, diag_opn, custom_lmax);

        if(no_grid_changes>0 && !is_grid_changing){
          last_grid_balance = true; // if the grid isn't changing anymore but it has changed, we need to do one more special interp of fields and balancing of the grid
        }
      }

      if(is_grid_changing || last_grid_balance){
        no_grid_changes++;
        PetscPrintf(mpi->comm(),"NS grid changed %d times \n",no_grid_changes);
        if(last_grid_balance){
          p4est_balance(p4est_np1,P4EST_CONNECT_FULL,NULL);
          PetscPrintf(mpi->comm(),"Does last grid balance \n");
        }

        my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est_np1,ghost_np1);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

        // Destroy fields_new and create it on the new grid:
        if(num_fields!=0){
          for(unsigned int k = 0; k<num_fields; k++){
            ierr = VecDestroy(fields_new_[k]);
            ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
          }
        }
        phi_new.destroy();
        phi_new.create(p4est_np1,nodes_np1);

        // Interpolate fields onto new grid:
        my_p4est_interpolation_nodes_t interp_refine_and_coarsen(ngbd_n);
        double xyz_interp[P4EST_DIM];
        foreach_node(n,nodes_np1){
          node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_interp);
          interp_refine_and_coarsen.add_point(n,xyz_interp);
        }
        if(num_fields!=0){
          interp_refine_and_coarsen.set_input(fields_,quadratic_non_oscillatory_continuous_v2,num_fields);
          // Interpolate fields
          interp_refine_and_coarsen.interpolate(fields_new_);
        }
        interp_refine_and_coarsen.set_input(phi.vec,quadratic_non_oscillatory_continuous_v2);
        interp_refine_and_coarsen.interpolate(phi_new.vec);

        if(last_grid_balance){
          last_grid_balance = false;
        }
      } // End of if grid is changing

      // Do last balancing of the grid, and final interp of phi:
      if(no_grid_changes>10) {PetscPrintf(mpi->comm(),"NS grid did not converge!\n"); break;}
    } // end of while grid is changing

    // Update the LSF accordingly:
    phi.destroy();
    phi.create(p4est_np1,nodes_np1);
    ierr = VecCopyGhost(phi_new.vec,phi.vec);

    // Destroy the vectors we created for refine and coarsen:
    for(int k = 0;k<num_fields; k++){
      ierr = VecDestroy(fields_new_[k]);
    }
    phi_new.destroy();
  } // end of if only navier stokes

  // -------------------------------
  // Destroy refinement fields now that they're not in use:
  // -------------------------------
  if(refine_by_vorticity){
    vorticity_refine.destroy();
  }
  if(refine_by_d2T){
    T_l_dd.destroy();
  }

  // -------------------------------
  // Clear up the memory from the std vectors holding refinement info:
  // -------------------------------
  compare_opn.clear(); diag_opn.clear(); criteria.clear();
  compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();
  custom_lmax.clear(); custom_lmax.shrink_to_fit();

} // end of "refine_and_coarsen_grid_and_advect_lsf_if_applicable()"

void my_p4est_stefan_with_fluids_t::update_the_grid(){
  // --------------------------------
  // Destroy p4est at n and slide grids:
  // -----------------------------------
  p4est_destroy(p4est_n);
  p4est_ghost_destroy(ghost_n);
  p4est_nodes_destroy(nodes_n);
  delete ngbd_n;
  delete hierarchy_n;

  p4est_n = p4est_np1;
  ghost_n = ghost_np1;
  nodes_n = nodes_np1;

  hierarchy_n = hierarchy_np1;
  ngbd_n = ngbd_np1;

  // -------------------------------
  // Create the new p4est at time np1:
  // -------------------------------
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE); // copy the grid but not the data
  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1,ghost_np1);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  // Get the new neighbors: // TO-DO : no need to do this here, is there ?
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

  // Initialize the neigbors:
  ngbd_np1->init_neighbors();

  // ------------------------------------------------------
  // Nullify the nm1 grid inside the NS solver if relevant:
  // ------------------------------------------------------
  if(solve_navier_stokes /*&& (tstep>1)*/){
    // TO-DO: i commented out the tstep>1 thing, so it could be a source of error, we will see
    ns->nullify_p4est_nm1(); // the nm1 grid has just been destroyed, but pointer within NS has not been updated, so it needs to be nullified (p4est_nm1 in NS == p4est in main)
  }

  // -------------------------------
  // Perform the advection/grid update:
  // -------------------------------
  // If solving NS, save the previous LSF to provide to NS solver, to correctly
  // interpolate hodge variable to new grid

  if(solve_navier_stokes){
    // Rochi addition 10/1/22 solved memory leak
    if (phi_nm1.vec!= NULL){
      phi_nm1.destroy();
    }
    phi_nm1.create(p4est_n, nodes_n);
    ierr = VecCopyGhost((there_is_a_substrate? phi_eff.vec : phi.vec), phi_nm1.vec); CHKERRXX(ierr); //--> this will need to be provided to NS update_from_tn_to_tnp1_grid_external
    // copy over phi eff if we are using a substrate
    // Note: this is done because the update_p4est destroys the old LSF, but we need to keep it
    // for NS update procedure
    if(print_checkpoints) ierr= PetscPrintf(mpi->comm(),"Phi nm1 copy is created ... \n");
  }


  refine_and_coarsen_grid_and_advect_lsf_if_applicable();

  // -------------------------------
  // Update hierarchy and neighbors to match new updated grid:
  // -------------------------------
  hierarchy_np1->update(p4est_np1,ghost_np1);
  ngbd_np1->update(hierarchy_np1,nodes_np1);

  // Initialize the neigbors:
  ngbd_np1->init_neighbors();

} // end of "update_the_grid()"

void my_p4est_stefan_with_fluids_t::interpolate_fields_onto_new_grid(){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  Vec all_fields_old[num_fields_interp];
  Vec all_fields_new[num_fields_interp];

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  //  my_p4est_interpolation_nodes_t* interp_nodes = NULL;
  //  interp_nodes = new my_p4est_interpolation_nodes_t(ngbd_old_grid);



  // Set existing vectors as elements of the array of vectors: --------------------------
  unsigned int i = 0;
  if(solve_stefan){
    all_fields_old[i++] = T_l_n.vec; // Now, all_fields_old[0] and T_l both point to same object (where old T_l vec sits)
    if(do_we_solve_for_Ts) all_fields_old[i++] = T_s_n.vec;

  }
  if(solve_navier_stokes){
    foreach_dimension(d){
      all_fields_old[i++] = v_n.vec[d];
    }
  }
  P4EST_ASSERT(i == num_fields_interp);

  // Create the array of vectors to hold the new values: ------------------------------
  PetscErrorCode ierr;
  for(unsigned int j = 0;j<num_fields_interp;j++){
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &all_fields_new[j]);CHKERRXX(ierr);
  }

  // Do interpolation:--------------------------------------------
  interp_nodes.set_input(all_fields_old, interp_bw_grids, num_fields_interp);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n, nodes_np1){
    node_xyz_fr_n(n, p4est_np1, nodes_np1,  xyz);
    interp_nodes.add_point(n,xyz);
  }

  interp_nodes.interpolate(all_fields_new);
  interp_nodes.clear();
  // Destroy the old fields no longer in use:------------------------
  for(unsigned int k=0;k<num_fields_interp;k++){
    ierr = VecDestroy(all_fields_old[k]); CHKERRXX(ierr); // Destroy objects where the old vectors were
  }
  // Slide the newly interpolated fields to back to their passed objects
  i = 0;
  if(solve_stefan){
    T_l_n.vec = all_fields_new[i++]; // Now, T_l points to (new T_l vec)
    if(do_we_solve_for_Ts) T_s_n.vec = all_fields_new[i++];

  }
  if(solve_navier_stokes){
    foreach_dimension(d){
      v_n.vec[d] = all_fields_new[i++];
    }
  }

  P4EST_ASSERT(i==num_fields_interp);
} // end of "interpolate_fields_onto_new_grid()"


void my_p4est_stefan_with_fluids_t::perform_lsf_advection_grid_update_and_interp_of_fields(){
  // ---------------------------------------------------
  // (4) Advance the LSF and (5) Update the grid:
  // ---------------------------------------------------
  /* In Coupled case: advect the LSF and update the grid according to vorticity, d2T/dd2, and phi
       * In Stefan case:  advect the LSF and update the grid according to phi
       * In NS case:      update the grid according to phi (no advection)
      */

  //-------------------------------------------------------------
  // (4/5b) Update the grids so long as this is not the last timestep:
  //-------------------------------------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Beginning grid update process ... \n"
                            "Refine by d2T = %s \n",refine_by_d2T? "true": "false");
  update_the_grid();

  // -------------------------------
  // (4/5c) Reinitialize the LSF on the new grid (if it has been advected):
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Reinitializing LSF... \n");
  ls->update(ngbd_np1);
  perform_reinitialization();

  // Regularize the front (if we are doing that)
  if(use_regularize_front){
    PetscPrintf(mpi->comm(), "Calling regularlize front: \n");
    regularize_front();
  }

  // Check collapse on the substrate (if we are doing that)
  if(there_is_a_substrate && use_collapse_onto_substrate){
    PetscPrintf(mpi->comm(), "Checking collapse \n ");
    //          if(start_w_merged_grains){regularize_front(p4est_np1, nodes_np1, ngbd_np1, phi_substrate.vec);}
    check_collapse_on_substrate();
  }

  //------------------------------------------------------
  // (4/5d) Destroy substrate LSF and phi_eff (if used) and re-create for upcoming timestep:
  //------------------------------------------------------
  if(there_is_a_substrate){
    phi_substrate.destroy();
    phi_eff.destroy();
    create_and_compute_phi_sub_and_phi_eff();
  }

  // ---------------------------------------------------
  // (4/5e) Interpolate Values onto New Grid:
  // ---------------------------------------------------

  if(print_checkpoints) PetscPrintf(mpi->comm(),"Interpolating fields to new grid ... \n");

  interpolate_fields_onto_new_grid();
  if(solve_navier_stokes){
    ns->update_from_tn_to_tnp1_grid_external((there_is_a_substrate? phi_eff.vec : phi.vec), phi_nm1.vec,
                                             v_n.vec, v_nm1.vec,
                                             p4est_np1, nodes_np1, ghost_np1,
                                             ngbd_np1,
                                             faces_np1,ngbd_c_np1,
                                             hierarchy_np1);
  }
  if(print_checkpoints) PetscPrintf(mpi->comm(),"Done. \n");


} // end of "perform_lsf_advection_grid_update_and_interp_of_fields()"


// -------------------------------------------------------
// Functions related to compound LSF, LSF regularization and tracking geometries:
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::create_and_compute_phi_sub_and_phi_eff(){

  phi_substrate.create(p4est_np1, nodes_np1);
  // TO-DO: should probably allow a case where the substrate is set by the user at the beginning of each timestep and handled in main, but leaving it for now
  sample_cf_on_nodes(p4est_np1, nodes_np1, *substrate_level_set, phi_substrate.vec);

  // For computing phi_effective for this step:
  phi_eff.create(p4est_np1, nodes_np1);

  std::vector<Vec> phi_eff_list;
  std::vector<mls_opn_t> phi_eff_opn_list;

  phi_eff_list.push_back(phi.vec); phi_eff_list.push_back(phi_substrate.vec);
  phi_eff_opn_list.push_back(MLS_INTERSECTION); phi_eff_opn_list.push_back(MLS_INTERSECTION);

  compute_phi_eff(phi_eff.vec, nodes_np1, phi_eff_list, phi_eff_opn_list);

  phi_eff_list.clear(); phi_eff_opn_list.clear();

  // Reinitialize:
  ls->reinitialize_2nd_order(phi_eff.vec);

} // end of "create_and_compute_phi_sub_and_phi_eff()"

void my_p4est_stefan_with_fluids_t::track_evolving_geometry(){

  // Count the islands:
  int num_islands = 0;

  compute_islands_numbers(*ngbd_np1, phi.vec, num_islands, island_numbers.vec);

  // Scale phi since we want to compute area of grains (which are in solid domain)
  VecScaleGhost(phi.vec, -1.);

  // Extend the island numbers across the interface a little bit (for paraview analysis purposes):
  my_p4est_level_set_t ls(ngbd_np1);

  double dxyz_[P4EST_DIM];
  dxyz_min(p4est_np1, dxyz_);
  double dxyz_min_ = MAX(DIM(dxyz_[0], dxyz_[1], dxyz_[2]));

  ls.extend_Over_Interface_TVD_Full(phi.vec, island_numbers.vec, 20, 2,
                                    floor(uniform_band/2)*dxyz_min_, uniform_band*dxyz_min_);


  // Scale phi back once done
  VecScaleGhost(phi.vec, -1.);



} // end of "track_evolving_geometry()"


void my_p4est_stefan_with_fluids_t::regularize_front()
{
  // WIP
  // FUNCTION FOR REGULARIZING THE SOLIDIFICATION FRONT:
  // adapted from function in my_p4est_multialloy_t originally developed by Daniil Bochkov, adapted by Elyce Bayat 08/24/2020

  double proximity_smoothing_ = proximity_smoothing;

  double dxyz_[P4EST_DIM];
  dxyz_min(p4est_np1, dxyz_);

  double dxyz_min_ = MIN(DIM(dxyz_[0],dxyz_[1],dxyz_[2]));
  double new_phi_val = .5*dxyz_min_;


  PetscErrorCode ierr;
  int mpi_comm = p4est_np1->mpicomm;

  ierr = PetscLogEventBegin(log_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "Removing problem geometries... "); CHKERRXX(ierr);

  int num_nodes_smoothed = 0;
  if (fabs(proximity_smoothing_)>EPS) {


    // First pass -- shift LSF up, see if there are any islands, remove subpools if there. Then reinitialize, shift back. "Solidify" any nodes that changed sign.
    // shift level-set upwards and reinitialize

    // Updated Daniil comment:
    // first pass: bridge too narrow regions by lifting level-set function, reinitializing,
    // brigning it down and checking which nodes flipped sign
    // (note that it also smooths out too sharp corners which are usually formed by
    // solidifying front ``getting stuck'' on grid nodes)
    my_p4est_level_set_t ls(ngbd_np1);
    vec_and_ptr_t front_phi_tmp(phi.vec);

    // Shift the LSF:
    double shift = dxyz_min_*proximity_smoothing_;
    ierr = VecCopyGhost(phi.vec, front_phi_tmp.vec);CHKERRXX(ierr);
    ierr = VecShiftGhost(front_phi_tmp.vec, shift);CHKERRXX(ierr);

    // eliminate small pools created by lifting
    int num_islands = 0;
    vec_and_ptr_t island_number(phi.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd_np1, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1)
    {
      ierr = PetscPrintf(mpi_comm, "%d subpools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_np1) {
        if (island_number.ptr[n] >= 0) {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i) {
        if (island_area[i] > island_area_max) {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // remove all but the biggest pool
      foreach_node(n, nodes_np1) {
        if (front_phi_tmp.ptr[n] < 0 && island_number.ptr[n] != main_island) {
          front_phi_tmp.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      front_phi_tmp.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
    // reinitilize and bring it down

    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);
    ierr = VecShiftGhost(front_phi_tmp.vec, -shift); CHKERRXX(ierr);

    // "solidify" nodes that changed sign
    phi.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes_np1) {
      if (phi.ptr[n] < 0 && front_phi_tmp.ptr[n] > 0) {
        phi.ptr[n] = front_phi_tmp.ptr[n];
        num_nodes_smoothed++;
      }
    }

    phi.restore_array();
    front_phi_tmp.restore_array();
    front_phi_tmp.destroy();
  }

  if (fabs(proximity_smoothing_)>0.) {
    // (Optional, not used anymore)
    // Second pass --  we shift LSF down, reinitialize, shift back, and see if some of those nodes are still "stuck"
    // shift level-set downwards and reinitialize
    my_p4est_level_set_t ls(ngbd_np1);
    vec_and_ptr_t front_phi_tmp(phi.vec);

    // shift up
    double shift = -0.1*dxyz_min_*proximity_smoothing_;
    VecCopyGhost(phi.vec, front_phi_tmp.vec);
    VecShiftGhost(front_phi_tmp.vec, shift);

    // reinitialize
    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);

    // shift back
    VecShiftGhost(front_phi_tmp.vec, -shift);

    // "solidify" nodes that changed sign
    phi.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes_np1) {
      if (phi.ptr[n] > 0 && front_phi_tmp.ptr[n] < 0) {
        phi.ptr[n] = front_phi_tmp.ptr[n];
        num_nodes_smoothed++;
      }
    }

    phi.restore_array();
    front_phi_tmp.restore_array();
    front_phi_tmp.destroy();
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE, &num_nodes_smoothed, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(ierr);

  // ELYCE TO DO-- THIS THIRD PART DOES NOT GET USED, SHOULD PROBABLY BYPASS THIS --> jk, I think it does get used , i think set() makes the two things the same so it's actually updating phi
  // third pass: look for isolated pools of liquid and remove them
  if (num_nodes_smoothed > 0) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
  {
    ierr = PetscPrintf(mpi_comm, "%d nodes smoothed... ", num_nodes_smoothed); CHKERRXX(ierr);
    int num_islands = 0;
    vec_and_ptr_t island_number(phi.vec);

    VecScaleGhost(phi.vec, -1.);
    compute_islands_numbers(*ngbd_np1, phi.vec, num_islands, island_number.vec);
    VecScaleGhost(phi.vec, -1.);

    if (num_islands > 1)
    {
      ierr = PetscPrintf(mpi_comm, "%d pools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      phi.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_np1)
      {
        if (island_number.ptr[n] >= 0)
        {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i)
      {
        if (island_area[i] > island_area_max)
        {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // solidify all but the biggest pool
      foreach_node(n, nodes_np1)
      {
        if (phi.ptr[n] < 0 && island_number.ptr[n] != main_island)
        {
          phi.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      phi.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
  }

  ierr = PetscPrintf(mpi_comm, "done!\n"); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
} // end of "regularize_front()"


void my_p4est_stefan_with_fluids_t::check_collapse_on_substrate(){
//  // Some things to set if you want to save collapse results and see what's going on:
//  bool save_collapse_vtk = false; // set this to true if you want to save collapse files
//  char collapse_folder[] = "/home/elyce/workspace/projects/multialloy_with_fluids/output_two_grain_clogging/gradP_0pt01_St_0pt07/grid57_flushing_growth_off/collapse_vtks";



  // Define values of interest:
  double proximity_smoothing_ = proximity_collapse;

  double dxyz_[P4EST_DIM];
  dxyz_min(p4est_np1,dxyz_);

  double dxyz_min_ = MIN(DIM(dxyz_[0],dxyz_[1],dxyz_[2]));
  double new_phi_val = .5*dxyz_min_;


  PetscErrorCode ierr;
  int mpi_comm = p4est_np1->mpicomm;

  ierr = PetscPrintf(mpi_comm, "Checking collapse onto substrate... "); CHKERRXX(ierr);

  int num_nodes_smoothed = 0;

  // Check the distance bw the LSF and substrate at all points along the interface first, only trigger techqniue if we are within proximity:

  bool substrate_is_within_proximity = false;
  phi.get_array(); phi_substrate.get_array();
  foreach_node(n, nodes_np1){
    if(fabs(phi.ptr[n]) < dxyz_min_*1.2){
      double substrate_dist = fabs(phi_substrate.ptr[n]);
      bool is_collapsed_already = fabs(phi.ptr[n] - phi_substrate.ptr[n]) < EPS;

      if((substrate_dist < proximity_smoothing_*dxyz_min_) && !is_collapsed_already){
        substrate_is_within_proximity = true;
      }
    }
  }
  phi.restore_array();
  phi_substrate.restore_array();

  //  printf("Rank %d has substrate_within_proximity = %d \n", p4est->mpirank, substrate_is_within_proximity);

  ierr = MPI_Allreduce(MPI_IN_PLACE, &substrate_is_within_proximity, 1, MPI_INT, MPI_LOR, mpi_comm); SC_CHECK_MPI(ierr);
  if(substrate_is_within_proximity){
    PetscPrintf(mpi_comm, "Interface is within proximity of substrate, proceeding with collapse ... \n");
  }



  //        my_p4est_interpolation_nodes_t interp(ngbd_);


  // Begin procedure of checking distance bw LSF and substrate, and collapsing if necessary
  if((fabs(proximity_smoothing_) > EPS) && substrate_is_within_proximity){
    my_p4est_level_set_t ls(ngbd_np1);
    vec_and_ptr_t phi_solid; phi_solid.create(phi.vec);
    vec_and_ptr_t front_phi_tmp; front_phi_tmp.create(phi.vec);

    // Compute phi_solid:
    VecCopyGhost(phi.vec, phi_solid.vec);
    VecScaleGhost(phi_solid.vec, -1.0);

    // Compute the effective LSF for this situation:
    std::vector<Vec> phi_list;
    std::vector<mls_opn_t> phi_opn;

    phi_list.push_back(phi_solid.vec); phi_list.push_back(phi_substrate.vec);
    phi_opn.push_back(MLS_INTERSECTION); phi_opn.push_back(MLS_INTERSECTION);

    compute_phi_eff(front_phi_tmp.vec, nodes_np1, phi_list, phi_opn);

    // Reinitialize the effective LSF
    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);

//    if(save_collapse_vtk){
//      // Save to vtk before: ---------------------------------------------------
//      std::vector<Vec_for_vtk_export_t> point_fields;
//      point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
//      point_fields.push_back(Vec_for_vtk_export_t(phi_substrate.vec, "phi_sub"));
//      point_fields.push_back(Vec_for_vtk_export_t(front_phi_tmp.vec, "phi_tmp"));
//      std::vector<Vec_for_vtk_export_t> cell_fields = {};
//      char filename[PATH_MAX];
//      sprintf(filename, "%s/before_collapse_tstep_%d", collapse_folder, tstep);
//      my_p4est_vtk_write_all_lists(p4est_np1,nodes_np1,ngbd_np1->get_ghost(),
//                                   P4EST_TRUE,P4EST_TRUE,filename,
//                                   point_fields, cell_fields);


//      point_fields.clear();
//      cell_fields.clear();
//      // -------------------------------------------------------------------------
//    }


//    // Create some vectors for vtk saving purposes (to visualize this process)
//    vec_and_ptr_t phi_tmp_up;
//    vec_and_ptr_t phi_tmp_up_reinit;
//    if(save_collapse_vtk) {
//      phi_tmp_up.create(front_phi_tmp.vec);
//      phi_tmp_up_reinit.create(front_phi_tmp.vec);
//    }

    // Now shift the effective LSF up:
    double shift = dxyz_min_*proximity_smoothing_;
    ierr = VecShiftGhost(front_phi_tmp.vec, shift);CHKERRXX(ierr);

//    if(save_collapse_vtk) VecCopyGhost(front_phi_tmp.vec, phi_tmp_up.vec);

    // Reinitialize:
    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);

//    if(save_collapse_vtk) VecCopyGhost(front_phi_tmp.vec, phi_tmp_up_reinit.vec);

    // Shift back down:
    ierr = VecShiftGhost(front_phi_tmp.vec, -shift);CHKERRXX(ierr);


//    if(save_collapse_vtk){
//      // Save to vtk intermediate: ---------------------------------------------------
//      std::vector<Vec_for_vtk_export_t> point_fields;
//      std::vector<Vec_for_vtk_export_t> cell_fields = {};
//      char filename[PATH_MAX];


//      point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
//      point_fields.push_back(Vec_for_vtk_export_t(phi_substrate.vec, "phi_sub"));
//      point_fields.push_back(Vec_for_vtk_export_t(phi_tmp_up.vec, "phi_tmp_up"));
//      point_fields.push_back(Vec_for_vtk_export_t(phi_tmp_up_reinit.vec, "phi_tmp_reinit"));
//      point_fields.push_back(Vec_for_vtk_export_t(front_phi_tmp.vec, "phi_tmp"));


//      sprintf(filename, "%s/collapse_vtks/intermediate_collapse_tstep_%d", collapse_folder, tstep);
//      my_p4est_vtk_write_all_lists(p4est_np1,nodes_np1,ngbd_np1->get_ghost(),
//                                   P4EST_TRUE,P4EST_TRUE,filename,
//                                   point_fields, cell_fields);


//      point_fields.clear();
//      cell_fields.clear();
//    }

//    // Destroy the vectors if we saved the collapse
//    if(save_collapse_vtk){
//      phi_tmp_up.destroy();
//      phi_tmp_up_reinit.destroy();
//    }


    // Collapse/Solidify nodes that changed sign:
    front_phi_tmp.get_array(); phi_solid.get_array();
    phi.get_array(); phi_substrate.get_array();
    foreach_node(n, nodes_np1){
      if((phi_solid.ptr[n] < 0) && (front_phi_tmp.ptr[n] >0)){
        phi.ptr[n] = MAX(-front_phi_tmp.ptr[n], phi_substrate.ptr[n]);
        num_nodes_smoothed++;
      }
    }
    front_phi_tmp.restore_array(); phi_solid.restore_array();
    phi.restore_array(); phi_substrate.restore_array();


    // Look for isolated pools of liquid and remove them
    if (num_nodes_smoothed > 0) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
    {
      ierr = PetscPrintf(mpi_comm, "%d nodes smoothed... ", num_nodes_smoothed); CHKERRXX(ierr);
      int num_islands = 0;
      vec_and_ptr_t island_number; island_number.create(phi.vec);

      VecScaleGhost(phi.vec, -1.);
      compute_islands_numbers(*ngbd_np1, phi.vec, num_islands, island_number.vec);
      VecScaleGhost(phi.vec, -1.);

      if (num_islands > 1)
      {
        ierr = PetscPrintf(mpi_comm, "%d pools removed... ", num_islands-1); CHKERRXX(ierr);
        island_number.get_array();
        phi.get_array();

        // compute liquid pools areas
        // TODO: make it real area instead of number of points
        std::vector<double> island_area(num_islands, 0);

        foreach_local_node(n, nodes_np1)
        {
          if (island_number.ptr[n] >= 0)
          {
            ++island_area[ (int) island_number.ptr[n] ];
          }
        }

        int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);

        // find the biggest liquid pool
        int main_island = 0;
        int island_area_max = island_area[0];

        for (int i = 1; i < num_islands; ++i)
        {
          if (island_area[i] > island_area_max)
          {
            main_island     = i;
            island_area_max = island_area[i];
          }
        }

        if (main_island < 0) throw;

        // solidify all but the biggest pool
        foreach_node(n, nodes_np1)
        {
          if (phi.ptr[n] < 0 && island_number.ptr[n] != main_island)
          {
            phi.ptr[n] = new_phi_val;
          }
        }

        island_number.restore_array();
        phi.restore_array();

      }

      island_number.destroy();
    }

    // Reinitialize the final result
    ls.reinitialize_2nd_order(phi.vec, 50);

//    if(save_collapse_vtk){
//      // Save to vtk after: ---------------------------------------------------
//      std::vector<Vec_for_vtk_export_t> point_fields;
//      std::vector<Vec_for_vtk_export_t> cell_fields = {};
//      char filename[PATH_MAX];


//      point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
//      point_fields.push_back(Vec_for_vtk_export_t(phi_substrate.vec, "phi_sub"));

//      sprintf(filename, "/home/elyce/workspace/projects/multialloy_with_fluids/output_two_grain_clogging/gradP_0pt01_St_0pt07/grid57_flushing_growth_off/collapse_vtks/after_collapse_tstep_%d", tstep);
//      my_p4est_vtk_write_all_lists(p4est_np1,nodes_np1,ngbd_np1->get_ghost(),
//                                   P4EST_TRUE,P4EST_TRUE,filename,
//                                   point_fields, cell_fields);


//      point_fields.clear();
//      cell_fields.clear();
//      // -------------------------------------------------------------------------
//    }

    // Destroy necessary things:
    phi_solid.destroy();
    front_phi_tmp.destroy();

    // Report:
    ierr = MPI_Allreduce(MPI_IN_PLACE, &num_nodes_smoothed, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(ierr);
    ierr = PetscPrintf(mpi_comm, "%d nodes smoothed... ", num_nodes_smoothed); CHKERRXX(ierr);

  }
} // end of "check_collapse_on_substrate()"

// -------------------------------------------------------
// Function(s) that solve one timestep of the desired coupled problem:
// Note: you could theoretically do this in a main.cpp as well, using this kind of step as a template
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::solve_all_fields_for_one_timestep(){

  // Make sure things are initialized:
  if(!(fields_are_initialized && grids_are_initialized)){
    throw std::runtime_error("Fields and grids are not marked as initialized. You must initialize these before solving a timestep. You can do this using the function perform_initializations() and by providing the required initial fields and parameters \n");
  }
  // ------------------------------------------------------------
  // (1) Poisson Problem at Nodes (for temp and/or conc scalar fields):
  // Setup and solve a Poisson problem on both the liquid and solidified subdomains
  // ------------------------------------------------------------
  if(0){
    // -------------------------------
    // TEMPORARY: save phi_eff fields to see what we are working with
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(T_l_n.vec, "T_l"));


    const char* out_dir = getenv("OUT_DIR_VTK");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }
    //          char output[] = "/home/elyce/workspace/projects/multialloy_with_fluids/output_two_grain_clogging/gradP_0pt01_St_0pt07/grid57_flush_no_collapse_after_extension_bc_added";
    char filename[1000];
    sprintf(filename, "%s/snapshot_before_poisson_%d", out_dir, tstep);
    my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ngbd_np1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();

  }

  if(solve_stefan){
    setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc();
  } // end of "if solve stefan"
  if(0){
    // -------------------------------
    // TEMPORARY: save phi_eff fields to see what we are working with
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(T_l_n.vec, "T_l"));


    const char* out_dir = getenv("OUT_DIR_VTK");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }
    //          char output[] = "/home/elyce/workspace/projects/multialloy_with_fluids/output_two_grain_clogging/gradP_0pt01_St_0pt07/grid57_flush_no_collapse_after_extension_bc_added";
    char filename[1000];
    sprintf(filename, "%s/snapshot_after_poisson_%d", out_dir, tstep);
    my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ngbd_np1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();

  }
  // ------------------------------------------------------------
  // (2) Computation of the interfacial velocity
  // ------------------------------------------------------------
  // (2a) Extend Fields Across Interface (if solving Stefan):
  // -- Note: we do not extend NS velocity fields bc NS solver handles that internally
  // ------------------------------------------------------------
  // Get smallest grid size: (this gets used in all examples at some point)
  dxyz_min(p4est_np1, dxyz_smallest);

  dxyz_close_to_interface = dxyz_close_to_interface_mult*MAX(dxyz_smallest[0],dxyz_smallest[1]);
  ls->update(ngbd_np1);
  if(solve_stefan){
    // Extend fields across the interface:

    extend_relevant_fields();

    // -------------------------------------------------------------------------
    // (2b) Actually compute the interfacial velocity (Stefan): -- do now so it can be used for
    // NS boundary condition
    // -------------------------------------------------------------------------

    if(print_checkpoints) PetscPrintf(mpi->comm(),"Computing interfacial velocity ... \n");
    v_interface.destroy();
    v_interface.create(p4est_np1,nodes_np1);


    bool did_crash = compute_interfacial_velocity();
    // Save crash file to vtk if vint is above max allowed (aka crashed)
    if(did_crash){
      char crash_tag[10];
      sprintf(crash_tag, "VINT");
      save_fields_to_vtk(0, true, crash_tag);
    }
  } // end of "if solve stefan"

  // ---------------------------------------------------------------------------
  // (3) Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
  // ---------------------------------------------------------------------------
  if (solve_navier_stokes){
    // Note: saving of crash file for NS gets handled internally
    setup_and_solve_navier_stokes_problem();
  } // End of "if solve navier stokes"

  // --------------------------------
  // (4/5a) Compute the timestep
  // (needed for the grid advection, and will be used as timestep for np1 step)
  // --------------------------------
  dt_nm1 = dt; // Slide the timestep

  // Compute stefan timestep:
  char stefan_timestep[1000];
  sprintf(stefan_timestep,"Computed interfacial velocity: \n"
                           " - Computational : %0.3e  "
                           "- Physical : %0.3e [m/s]  "
                           "- Physical : %0.3f  [mm/s] \n",
          v_interface_max_norm,
          v_interface_max_norm*vel_nondim_to_dim,
          v_interface_max_norm*vel_nondim_to_dim*1000.);

  compute_timestep(); // this function modifies the variable dt

} // end of "solve_all_fields_for_one_timestep()"


// -------------------------------------------------------
// Functions related to VTK saving:
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::save_fields_to_vtk(int out_idx, bool is_crash, char crash_type[]){

  char output[1000];

  const char* out_dir = getenv("OUT_DIR_VTK");
  if(!out_dir){
    throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
  }
  sprintf(output, "%s/grid_lmin%d_lint%d_lmax%d", out_dir, lmin, lint, lmax);
  // Create outdir if it does not exist:
  if(!file_exists(output)){
    create_directory(output, mpi->rank(), mpi->comm());
  }
  if(!is_folder(output)){
    if(!create_directory(output, mpi->rank(), mpi->comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "saving geometry information: the path %s is invalid and the directory could not be created", output);
      throw std::invalid_argument(error_msg);
    }
  }
  // Now save to vtk:


  PetscPrintf(mpi->comm(),"Saving to vtk, outidx = %d ...\n",out_idx);

  //    if(example_uses_inner_LSF){
  //      if(start_w_merged_grains){regularize_front(p4est, nodes, ngbd, phi_substrate.vec);}
  //    }

  if(is_crash){
    sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_%s_CRASH", output, lmin, lmax, crash_type);

  }
  else{
    sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_outidx_%d", output, lmin, lmax, out_idx);
  }

  // Calculate curvature:
  vec_and_ptr_t kappa;
  vec_and_ptr_dim_t normal;

  kappa.create(p4est_np1, nodes_np1);
  normal.create(p4est_np1, nodes_np1);

  VecScaleGhost(phi.vec,-1.0);
  compute_normals(*ngbd_np1, phi.vec,normal.vec);
  compute_mean_curvature(*ngbd_np1, normal.vec,kappa.vec);

  VecScaleGhost(phi.vec,-1.0);

  // Save data:
  std::vector<Vec_for_vtk_export_t> point_fields;
  point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
  point_fields.push_back(Vec_for_vtk_export_t(kappa.vec, "kappa"));


  //phi substrate and phi eff
  if(there_is_a_substrate){
    point_fields.push_back(Vec_for_vtk_export_t(phi_eff.vec, "phi_eff"));
    point_fields.push_back(Vec_for_vtk_export_t(phi_substrate.vec, "phi_sub"));
  }

  // stefan related fields
  if(solve_stefan){
    point_fields.push_back(Vec_for_vtk_export_t(T_l_n.vec, "Tl"));
    if(do_we_solve_for_Ts){
      point_fields.push_back(Vec_for_vtk_export_t(T_s_n.vec, "Ts"));
    }
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[0], "v_interface_x"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[1], "v_interface_y"));
  }

  if(solve_navier_stokes){
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "u"));

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "v"));

    if(tstep!=0){
        point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vorticity"));
        point_fields.push_back(Vec_for_vtk_export_t(press_nodes.vec, "pressure"));

    }
  }

  if(track_evolving_geometries && !is_crash){
    island_numbers.create(p4est_np1, nodes_np1);
    track_evolving_geometry();
    point_fields.push_back(Vec_for_vtk_export_t(island_numbers.vec, "island_no"));
  }

  std::vector<Vec_for_vtk_export_t> cell_fields = {};


  my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ghost_np1,
                               P4EST_TRUE,P4EST_TRUE, output,
                               point_fields, cell_fields);


  point_fields.clear();
  cell_fields.clear();

  kappa.destroy();
  normal.destroy();

  if(track_evolving_geometries && !is_crash){
    island_numbers.destroy();
  }

  if(print_checkpoints) PetscPrintf(mpi->comm(),"Finishes saving to VTK \n");

} // end of "save_fields_to_vtk()"


// -------------------------------------------------------
// Functions related to save state/load state:
// -------------------------------------------------------

void my_p4est_stefan_with_fluids_t::fill_or_load_double_parameters(save_or_load flag, PetscInt num, PetscReal *data){
  size_t idx=0;
  switch(flag){
  case SAVE:{
    data[idx++] = tn;
    data[idx++] = dt;
    data[idx++] = dt_nm1;
    data[idx++] = cfl_Stefan;
    data[idx++] = cfl_NS;
    data[idx++] = uniform_band;
    data[idx++] = sp->lip;
    data[idx++] = NS_norm;
    data[idx++] = v_interface_max_norm;
    break;
  }
  case LOAD:{
    tn = data[idx++];
    dt = data[idx++];
    dt_nm1 = data[idx++];
    cfl_Stefan = data[idx++];
    cfl_NS = data[idx++];
    uniform_band= data[idx++];
    sp->lip = data[idx++];
    NS_norm = data[idx++];
    v_interface_max_norm = data[idx++];
  }

  }
  P4EST_ASSERT(idx == num);
} // end of "fill_or_load_double_parameters()"

void my_p4est_stefan_with_fluids_t::fill_or_load_integer_parameters(save_or_load flag, PetscInt num, PetscInt *data){
  size_t idx=0;
  switch(flag){
  case SAVE:{
    data[idx++] = advection_sl_order;
    data[idx++] = tstep;
    data[idx++] = sp->min_lvl;
    data[idx++] = sp->max_lvl;
    break;
  }
  case LOAD:{
    advection_sl_order = data[idx++];
    tstep = data[idx++];
    sp->min_lvl=data[idx++];
    sp->max_lvl=data[idx++];
  }

  }
  P4EST_ASSERT(idx == num);
} // end of "fill_or_load_integer_parameters()"

void my_p4est_stefan_with_fluids_t::save_or_load_parameters(const char* filename, save_or_load flag){
  PetscErrorCode ierr;

  // Double parameters we need to save:
  PetscInt num_doubles = 9;
  PetscReal double_parameters[num_doubles];

  // Integer parameters we need to save:
  PetscInt num_integers = 4;
  PetscInt integer_parameters[num_integers];

  int fd;
  char diskfilename[PATH_MAX];

  switch(flag){
  case SAVE:{
    if(mpi->rank() ==0){

      // Save the integer parameters to a file
      sprintf(diskfilename,"%s_integers",filename);

      fill_or_load_integer_parameters(flag, num_integers, integer_parameters);
      ierr = PetscBinaryOpen(diskfilename,FILE_MODE_WRITE,&fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, integer_parameters, num_integers, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

      // Save the double parameters to a file:

      sprintf(diskfilename, "%s_doubles", filename);
      fill_or_load_double_parameters(flag,num_doubles, double_parameters);

      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, double_parameters, num_doubles, PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    break;
  }
  case LOAD: {
    // First, load the integer parameters:
    sprintf(diskfilename, "%s_integers", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("The file storing the solver's integer parameters could not be found");
    if(mpi->rank()==0){
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, integer_parameters, num_integers, NULL, PETSC_INT); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    int mpiret = MPI_Bcast(integer_parameters, num_integers, MPI_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
    fill_or_load_integer_parameters(flag,num_integers, integer_parameters);

    // Now, load the double parameters:
    sprintf(diskfilename, "%s_doubles", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("The file storing the solver's double parameters could not be found");
    if(mpi->rank() == 0)
    {
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, double_parameters, num_doubles, NULL, PETSC_DOUBLE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

    }
    mpiret = MPI_Bcast(double_parameters, num_doubles, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
    fill_or_load_double_parameters(flag,num_doubles, double_parameters);
    break;
  }
  default:
    throw std::runtime_error("Unkown flag values were used when load/saving parameters \n");
  }
} // end of "save_or_load_parameters()"

void my_p4est_stefan_with_fluids_t::prepare_fields_for_save_or_load(vector<save_or_load_element_t> &fields_to_save_np1,
                                     vector<save_or_load_element_t> &fields_to_save_n){

  save_or_load_element_t to_add;
  // ----------------------
  // Add relevant fields to the vector of fields to save:
  // ----------------------
  // Level set function
  to_add.name = "phi";
  to_add.DATA_SAMPLING = NODE_DATA;
  to_add.nvecs = 1;
  to_add.pointer_to_vecs = &phi.vec;
  fields_to_save_np1.push_back(to_add);

  // Temperature fields
  if(solve_stefan){
    // Fluid temp
    to_add.name = "T_l_n";
    to_add.DATA_SAMPLING = NODE_DATA;
    to_add.nvecs = 1;
    to_add.pointer_to_vecs = &T_l_n.vec;
    fields_to_save_np1.push_back(to_add);

    if(advection_sl_order==2 && solve_navier_stokes){
      // Fluid temp at nm1
      to_add.name = "T_l_nm1";
      to_add.DATA_SAMPLING = NODE_DATA;
      to_add.nvecs = 1;
      to_add.pointer_to_vecs = &T_l_nm1.vec;
      fields_to_save_n.push_back(to_add);
    }

    if(do_we_solve_for_Ts){
      // Solid temp if relevant
      to_add.name = "T_s_n";
      to_add.DATA_SAMPLING = NODE_DATA;
      to_add.nvecs = 1;
      to_add.pointer_to_vecs = &T_s_n.vec;
      fields_to_save_np1.push_back(to_add);
    }
  }

  // Navier Stokes fields:
  if(solve_navier_stokes){
    to_add.name = "v_NS_n";
    to_add.DATA_SAMPLING = NODE_DATA;
    to_add.nvecs = P4EST_DIM;
    to_add.pointer_to_vecs = v_n.vec;
    fields_to_save_np1.push_back(to_add);

    to_add.name = "v_NS_nm1";
    to_add.DATA_SAMPLING = NODE_DATA;
    to_add.nvecs = P4EST_DIM;
    to_add.pointer_to_vecs = v_nm1.vec;
    fields_to_save_n.push_back(to_add);
  }

} // end of "prepare_fields_for_save_or_load"


void my_p4est_stefan_with_fluids_t::save_state(const char* path_to_directory,unsigned int n_saved){
  PetscErrorCode ierr;

  if(!file_exists(path_to_directory)){
    create_directory(path_to_directory,p4est_np1->mpirank, p4est_np1->mpicomm);
  }
  if(!is_folder(path_to_directory)){
    if(!create_directory(path_to_directory, p4est_np1->mpirank, p4est_np1->mpicomm))
    {
      char error_msg[1024];
      sprintf(error_msg, "save_state: the path %s is invalid and the directory could not be created", path_to_directory);
      throw std::invalid_argument(error_msg);
    }
  }

  unsigned int backup_idx = 0;

  if(mpi->rank() ==0){
    unsigned int n_backup_subfolders = 0;

    // Get the current number of backups already present:
    // (Delete extra ones that exist for whatever reason)
    std::vector<std::string> subfolders; subfolders.resize(0);
    get_subdirectories_in(path_to_directory,subfolders);

    for(size_t idx =0; idx<subfolders.size(); ++idx){
      if(!subfolders[idx].compare(0,7,"backup_")){
        unsigned int backup_idx;
        sscanf(subfolders[idx].c_str(), "backup_%d", &backup_idx);

        if(backup_idx >= n_saved)
        {
          char full_path[PATH_MAX];
          sprintf(full_path, "%s/%s", path_to_directory, subfolders[idx].c_str());
          delete_directory(full_path, p4est_np1->mpirank, p4est_np1->mpicomm, true);
        }
        else
          n_backup_subfolders++;
      }
    }

    // check that they are successively indexed if less than the max number
    if(n_backup_subfolders < n_saved)
    {
      backup_idx = 0;
      for (unsigned int idx = 0; idx < n_backup_subfolders; ++idx) {
        char expected_dir[PATH_MAX];
        sprintf(expected_dir, "%s/backup_%d", path_to_directory, (int) idx);

        if(!is_folder(expected_dir))
          break; // well, it's a mess in there, but I can't really do any better...
        backup_idx++;
      }
    }
    // Slide the names of the backup folders in time:
    if ((n_saved > 1) && (n_backup_subfolders == n_saved))
    {
      char full_path_zeroth_index[PATH_MAX];
      sprintf(full_path_zeroth_index, "%s/backup_0", path_to_directory);
      // delete the 0th
      delete_directory(full_path_zeroth_index, p4est_np1->mpirank, p4est_np1->mpicomm, true);
      // shift the others
      for (size_t idx = 1; idx < n_saved; ++idx) {
        char old_name[PATH_MAX], new_name[PATH_MAX];
        sprintf(old_name, "%s/backup_%d", path_to_directory, (int) idx);
        sprintf(new_name, "%s/backup_%d", path_to_directory, (int) (idx-1));
        rename(old_name, new_name);
      }
      backup_idx = n_saved-1;
    }

    subfolders.clear();

  } // end of operations only on rank 0

  int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est_np1->mpicomm); SC_CHECK_MPI(mpiret);// acts as a MPI_Barrier, too

  char path_to_folder[PATH_MAX];
  sprintf(path_to_folder, "%s/backup_%d", path_to_directory, (int) backup_idx);
  create_directory(path_to_folder, p4est_np1->mpirank, p4est_np1->mpicomm);

  char filename[PATH_MAX];

  // save the solver parameters
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, SAVE);

  // Save the p4est and corresponding data:

  vector<save_or_load_element_t> fields_to_save_n, fields_to_save_np1;


  prepare_fields_for_save_or_load(fields_to_save_np1, fields_to_save_n);

  // Save the state:
  // choosing not to save the faces because we don't need them saved ? Elyce to-do double check this, 1/6/21
  my_p4est_save_forest_and_data(path_to_folder, p4est_np1, nodes_np1, NULL,
                                "p4est_np1", fields_to_save_np1);

  my_p4est_save_forest_and_data(path_to_folder, p4est_n, nodes_n, NULL,
                                "p4est_n", fields_to_save_n);

  ierr = PetscPrintf(p4est_np1->mpicomm,"Saved solver state in ... %s \n",path_to_folder);CHKERRXX(ierr);

} // end of "save_state()"

void my_p4est_stefan_with_fluids_t::load_state(const char* path_to_folder){

  char filename[PATH_MAX];
  if(!is_folder(path_to_folder)) throw std::invalid_argument("Load state: path to directory is invalid \n");

  // First load the general solver parameters -- integers and doubles
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, LOAD);

  // Load p4est_n and corresponding objections
  vector<save_or_load_element_t> fields_to_load_np1, fields_to_load_n;


  prepare_fields_for_save_or_load(fields_to_load_np1,
                                  fields_to_load_n);

  // Load the time n grid:
  my_p4est_load_forest_and_data(mpi->comm(), path_to_folder,
                                p4est_n, conn, P4EST_TRUE, ghost_n, nodes_n,
                                "p4est_n", fields_to_load_n);

  // Load the np1 grid:
  my_p4est_load_forest_and_data(mpi->comm(), path_to_folder,
                                p4est_np1, conn, P4EST_TRUE, ghost_np1, nodes_np1,
                                "p4est_np1", fields_to_load_np1);


  P4EST_ASSERT(find_max_level(p4est_n) == sp->max_lvl);
  P4EST_ASSERT(find_max_level(p4est_np1) == sp->max_lvl);

  // Update the user pointer:
  splitting_criteria_cf_and_uniform_band_t* sp_new = new splitting_criteria_cf_and_uniform_band_t(*sp);
  p4est_n->user_pointer = (void*) sp_new;
  p4est_np1->user_pointer = (void*) sp_new;

  PetscPrintf(mpi->comm(),"Loads forest and data \n");
} // end of "load_state()"

// -------------------------------------------------------
// Interfacial boundary condition for temperature:
// -------------------------------------------------------
double my_p4est_stefan_with_fluids_t::interfacial_bc_temp_t::Gibbs_Thomson(DIM(double x, double y, double z))const {
  switch(owner->problem_dimensionalization_type){
    // Note slight difference in condition bw diff nondim types -- T0 vs Tinf
    case NONDIM_BY_FLUID_VELOCITY:{
      return (owner->theta_interface - (owner->sigma/owner->l_char)*((*kappa_interp)(x,y))*(owner->theta_interface + owner->T0/owner->deltaT));
    }
    case NONDIM_BY_SCALAR_DIFFUSIVITY:{
      return (owner->theta_interface - (owner->sigma/owner->l_char)*((*kappa_interp)(x,y))*(owner->theta_interface + owner->Tinfty/owner->deltaT));
    }
    case DIMENSIONAL:{
      return (owner->Tinterface*(1 - owner->sigma*((*kappa_interp)(x,y))));
    }
    default:{
      throw std::runtime_error("Gibbs_thomson: unrecognized problem dimensionalization type \n");
    }
  }
}

// -------------------------------------------------------
// Interfacial boundary condition(s) for fluid velocity:
// -------------------------------------------------------
double my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t::Conservation_of_Mass(DIM(double x, double y, double z)) const{
  return (*v_interface_interp)(x,y)*(1. - (owner->rho_s/owner->rho_l));
}

double my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t::Strict_No_Slip(DIM(double x, double y, double z)) const{
  return (*v_interface_interp)(x,y);
}





