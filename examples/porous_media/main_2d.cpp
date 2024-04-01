/*
 * Title: multialloy_with_fluids
 * Description:
 * Author: Elyce
 * Date Created: 08-06-2019
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
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
#include <src/my_p4est_stefan_with_fluids.h>
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

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>


#undef MIN
#undef MAX

using namespace std;

parameter_list_t pl;

// ---------------------------------------
// Examples to run:
// ---------------------------------------
// Define the numeric label for each type of example to make implementation a bit more clear
enum:int {
  FRANK_SPHERE = 0,
  NS_GIBOU_EXAMPLE = 1,
  COUPLED_TEST_2 = 2,
  COUPLED_PROBLEM_EXAMPLE = 3,
  ICE_AROUND_CYLINDER = 4,
  FLOW_PAST_CYLINDER = 5,
  DENDRITE_TEST = 6,
  MELTING_ICE_SPHERE = 7,
  EVOLVING_POROUS_MEDIA = 8,
  PLANE_POIS_FLOW=9,
  DISSOLVING_DISK_BENCHMARK=10,
  MELTING_ICE_SPHERE_NAT_CONV=11,
  COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP=12,
};
// Elyce to-do: add dissolving porous media example

//enum{LIQUID_DOMAIN=0, SOLID_DOMAIN=1};

// ---------------------------------------
// Example/application options:
// ---------------------------------------
DEFINE_PARAMETER(pl, int, example_, 0,"example number: \n"
                                   "0 - Frank Sphere (Stefan only) \n"
                                   "1 - NS Gibou example (Navier Stokes only) \n"
                                   "2 - Additional coupled verification test (not fully verified) \n"
                                   "3 - Coupled problem example for verification \n"
                                   "4 - Ice solidifying around a cooled cylinder \n"
                                   "5 - Flow past a cylinder (Navier Stokes only)\n"
                                   "6 - dendrite solidification test (WIP) \n"
                                   "7 - melting of an ice sphere \n"
                                   "8 - melting of a porous media (with fluid flow) \n"
                                   "9 - plane Poiseuille flow \n "
                                   "10 - dissolving disk benchmark for dissolution problem \n"
                                   "11 - Melting of an ice sphere in natural convection \n"
                                   "12 - Coupled problem example for verification with boussinesq approximation"
                                   "default: 4 \n");

// ---------------------------------------
// Save options:
// ---------------------------------------
// Options for saving to vtk:
DEFINE_PARAMETER(pl, bool, save_to_vtk, true, "We save vtk files using a "
                                              "given dt increment if this is set to true \n");

DEFINE_PARAMETER(pl, bool, save_using_dt, false, "We save vtk files using a "
                                                 "given dt increment if this is set to true \n");

DEFINE_PARAMETER(pl, bool, save_using_iter, true, "We save every prescribed number "
                                                   "of iterations if this is set to true \n");

DEFINE_PARAMETER(pl, int, save_every_iter, 1, "Saves vtk every n number "
                                              "of iterations (default is 1) \n");

DEFINE_PARAMETER(pl, double, save_every_dt, 1, "Saves vtk every dt amount of "
                                               "time in seconds of dimensional time (default is 1) \n");

// Options to compute and save fluid forces to a file:
DEFINE_PARAMETER(pl, bool, save_fluid_forces, false, "Saves fluid forces if true (default: false) \n");

DEFINE_PARAMETER(pl, bool, save_area_data, false, "Save area data if true (default: false, but some examples will set this to true automatically)\n");

DEFINE_PARAMETER(pl, double, save_data_every_dt, 0.01, "Saves fluid forces and/or area data every dt amount of time in seconds of dimensional time (default is 1.0) \n");

// Options to track and output island numbers for an evolving geometry:
DEFINE_PARAMETER(pl, bool, track_evolving_geometries, false, "Flag to track island numbers for the evolving geometry(ies) and output the island numbers to vtk and other information to a file in the same folder as the vtk info. Default: false. For use with the evolving porous media problem. \n");

DEFINE_PARAMETER(pl, bool, wrap_up_simulation_if_solid_has_vanished, false, "If set to true, the simulation will check if the solid region has vanished and pause the simulation at that point \n");

// Save state options
DEFINE_PARAMETER(pl, bool, save_state, false, "Saves the simulation state (as per other specified options). Default: false \n");
DEFINE_PARAMETER(pl, bool, save_state_using_iter, false, "Saves the simulation state every specified number of iterations \n");
DEFINE_PARAMETER(pl, bool, save_state_using_dt, false, "Saves the simulation state using every specified dt \n");

DEFINE_PARAMETER(pl, int, save_state_every_iter, -1., "Saves simulation state every n number of iterations (default is -1) \n");
DEFINE_PARAMETER(pl, double, save_state_every_dt, -1., "Save state every dt -- tells us how often (in seconds) to save the simulation state. Default is -1. \n");
DEFINE_PARAMETER(pl, int, num_save_states, 20, "Number of save states we keep on file (default is 20) \n");

// Load state options
DEFINE_PARAMETER(pl ,bool, loading_from_previous_state, false,"Loads simulation from previous state if marked true \n");


// ---------------------------------------
// Debugging options:
// ---------------------------------------
// Options for checking memory usage: -- this was more heavily used when I was investigating a memory leak . TO-DO: clean this stuff up ?
DEFINE_PARAMETER(pl, int, check_mem_every_iter, -1, "Checks memory usage every n number of iterations (default is -1 aka, don't check. To check, set to a positive integer value) \n");
DEFINE_PARAMETER(pl, double, mem_safety_limit, 60.e9, "Memory upper limit before closing the program -- in bytes \n");

// Options for checking timing:
DEFINE_PARAMETER(pl, int, timing_every_n, -1, "Print timing info every n iterations (default -1 aka no use, to use this feature, set to a positive integer value) \n");


DEFINE_PARAMETER(pl, bool, print_checkpoints, false, "Print checkpoints throughout script for debugging? \n");

// ---------------------------------------
// Solution options:
// ---------------------------------------
// Related to which physics we solve:
DEFINE_PARAMETER(pl, bool, solve_stefan, false, "Solve stefan ? \n");
DEFINE_PARAMETER(pl, bool, solve_navier_stokes, false, "Solve navier stokes? \n");
DEFINE_PARAMETER(pl, bool, solve_coupled, true, "Solve the coupled problem? \n"); // <-- get rid of

DEFINE_PARAMETER(pl, bool, do_we_solve_for_Ts, false, "True/false to describe whether or not we solve for the solid temperature (or concentration). Default: false. This is set to true for select examples in select_solvers()\n");
DEFINE_PARAMETER(pl, bool, use_boussinesq, false, "True/false to describe whether or not we are solving the problem considering natural convection effects using the boussinesq approx. Default: false. This is set true for the dissolving disk benchmark case. This is used to distinguish the dissolution-specific stefan condition, as contrasted with other concentration driven problems in solidification. \n");

DEFINE_PARAMETER(pl, bool, use_regularize_front, false, "True/false to describe whether or not we use Daniil's algorithm for smoothing problem geometries and bridging gaps of a certain proximity. Default:false \n");

DEFINE_PARAMETER(pl, bool, use_collapse_onto_substrate, false, "True/false to describe whether or not we use algorithm for collapsing interface onto a substrate within a certain proximity. Default:false \n");

DEFINE_PARAMETER(pl, double, proximity_smoothing, 2.5, "Parameter for front regularization. default: 2.5 \n");
DEFINE_PARAMETER(pl, double, proximity_collapse, 3.0, "Parameter for collapse onto front. default: 3.0 \n");


DEFINE_PARAMETER(pl, bool, is_dissolution_case, false, "True/false to describe whether or not we are solving dissolution. Default: false. This is set true for the dissolving disk benchmark case. This is used to distinguish the dissolution-specific stefan condition, as contrasted with other concentration driven problems in solidification. \n");
DEFINE_PARAMETER(pl, int, nondim_type_used, -1., "Integer value to overwrite the nondimensionalization type used for a given problem. The default is -1. If this is specified to a nonnegative number, it will overwrite the particular example's default. 0 - nondim by fluid velocity, 1 - nondim by diffusivity (thermal or conc), 2 - dimensional.  \n");


// Set the method for calculationg the dissolution/precipitation interfacial velocity:
DEFINE_PARAMETER(pl, int, precip_disso_vgamma_calc, 1, "The type of calculation used for the interface velocity in precipitation/dissolution problems. 0 - compute by concentration value. 1- compute by concentration flux. \n");
//precipitation_dissolution_interface_velocity_calc_type_t precip_disso_vgamma_calc_type = (precipitation_dissolution_interface_velocity_calc_type_t) precip_disso_vgamma_calc;

// Related to LSF reinitialization:
DEFINE_PARAMETER(pl, int, reinit_every_iter, 1, "An integer option for how many iterations we wait "
                                                "before reinitializing the LSF in the case of a coupled problem"
                                                " (only implemented for coupled problem!). "
                                                "Default : 1. \n This can be helpful when the "
                                                "timescales governing interface evolution are much "
                                                "larger than those governing the flow. "
                                                "For example, if vint is 1000x smaller than vNS, "
                                                "we may want to reinitialize only every 1000 timesteps, or etc. "
                                                "This can prevent the interface from degrading via frequent "
                                                "reinitializations relative to the amount of movement it experiences. ");

// Related to the Stefan and temperature/concentration problem:
DEFINE_PARAMETER(pl, double, cfl, 0.25, "CFL number for Stefan problem (default:0.5) \n");
DEFINE_PARAMETER(pl, int, advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid temperature field(default:2) \n");
DEFINE_PARAMETER(pl, bool, force_interfacial_velocity_to_zero, false, "Force the interfacial velocity to zero? \n");

// Related to the Navier-Stokes problem:
//DEFINE_PARAMETER(pl, double, Re_overwrite, -100.0, "Overwrite the examples set Reynolds number (works if set to a positive number, default:-100.00");
DEFINE_PARAMETER(pl, int, NS_advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid velocity fields (default:2) \n");
DEFINE_PARAMETER(pl, double, cfl_NS, 1.0, "CFL number for Navier-Stokes problem (default:1.0) \n");
DEFINE_PARAMETER(pl, double, hodge_tolerance, 1.e-3, "Tolerance on hodge for error convergence (default:1.e-3) \n");

// Specifying flow or no flow: Elyce to-do 12/14/21: remove this no_flow option, we should be able to just turn on and off the solve_navier_stokes and have the same effect!
DEFINE_PARAMETER(pl, bool, no_flow, false, "An override switch for the ice cylinder example to run a case with no flow (default: false) \n");

// Related to simulation duration settings:
DEFINE_PARAMETER(pl, double, duration_overwrite, -100.0, "Overwrite the duration in minutes (works if set to a positive number, default:-100.0 \n");
DEFINE_PARAMETER(pl, double, duration_overwrite_nondim, -10.,"Overwrite the duration in nondimensional time (in nondimensional time) -- not fully implemented \n");



// whether or not to use inner cylinders for the porous media example:
DEFINE_PARAMETER(pl, bool, use_inner_surface_porous_media, false, "If true, will use inner cylinders in the porous media problem with an initial solid layer on top. This might represent a media with some solidified material or deposit already present on a fixed structure. a.k.a. Okada style (ice on cooled cyl) Default: false. \n");

DEFINE_PARAMETER(pl, bool, start_w_merged_grains, false, "If true, we assume the LSF provided contains geometry for grains that are already merged, and the regularize front procedure will be called after initializing the level set to remove problem geometry. Default: false. For use in porous media project. \n");

DEFINE_PARAMETER(pl, double, porous_media_initial_thickness, 4.0 , "The initial thickness outer interface in relation to the inner interface, in number of smallest grid cells. Default value: 4.0 \n");


// Potential for phi advection subiterations (for slow evolving interfaces and quasi-steady approach)
DEFINE_PARAMETER(pl, bool, do_phi_advection_substeps, false, "do phi advection substeps? \n");
//DEFINE_PARAMETER(pl, int, num_phi_advection_substeps, 0, "Number of phi advection substeps per timestep \n");
DEFINE_PARAMETER(pl, double, phi_advection_substeps_coeff, 0., "Number of phi advection substeps per timestep \n");
DEFINE_PARAMETER(pl, double, phi_advection_substep_startup_time, 0., "dimensional startup time in seconds before activating LSF substeps \n");
DEFINE_PARAMETER(pl, double, cfl_phi_advection_substep, 0.8, "CFL for choosing the timestep of the phi advection substep \n");


// ---------------------------------------
// Booleans that we select to simplify logic in the main program for different processes that are required for different examples:
// ---------------------------------------

bool analytical_IC_BC_forcing_term;
bool example_is_a_test_case;

bool interfacial_temp_bc_requires_curvature;
bool interfacial_temp_bc_requires_normal;

bool interfacial_vel_bc_requires_vint;

bool example_uses_inner_LSF;
//bool example_requires_area_computation;


bool example_has_known_max_vint;

double max_vint_known_for_ex = 1.0;
unsigned int num_fields_interp = 0;

void select_solvers(){
  switch(example_){
    case FRANK_SPHERE:
      solve_stefan = true;
      solve_navier_stokes = false;
      break;

    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
      if(!no_flow){
        solve_stefan = true;
        solve_navier_stokes = true;
      }
      else{
        solve_stefan=true;
        solve_navier_stokes=false;
      }
      break;
    case MELTING_ICE_SPHERE_NAT_CONV:
      solve_stefan=true;
      solve_navier_stokes=true;
    break;
    case NS_GIBOU_EXAMPLE:
      solve_stefan = false;
      solve_navier_stokes = true;
      break;

    case FLOW_PAST_CYLINDER:
      solve_stefan = false;
      solve_navier_stokes = true;
      break;

    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
      solve_stefan = true;
      solve_navier_stokes = true;
      break;
    case DENDRITE_TEST: // will need to select solvers manually
      break;
    case EVOLVING_POROUS_MEDIA:
//      solve_stefan=true;
//      solve_navier_stokes=true;
        // 1/31/22 -- changed to select solvers manually - in the bash, those are set to true when running a EVOLVING_POROUS_MEDIA.
      break;
    case PLANE_POIS_FLOW:
      solve_stefan=false;
      solve_navier_stokes=true;
      break;
    case DISSOLVING_DISK_BENCHMARK:
      solve_stefan=true;
      solve_navier_stokes=true;
      break;
    }
  if(save_using_dt && save_using_iter){
      throw std::invalid_argument("You have selected to save using dt and using iteration, you need to select only one \n");
    }

    // Set number of interpolation fields:
    // Number of fields interpolated from one grid to the next depends on which equations
    // we are solving, therefore we select appropriately
    num_fields_interp = 0;
    if(solve_stefan){
      num_fields_interp+=1; // Tl, // DONT ACTUALLY NEED TO INTERP: vint_x, vint_y
      if(do_we_solve_for_Ts) num_fields_interp+=1; // Ts
    }
    if(solve_navier_stokes){
      num_fields_interp+=2; // vNS_x, vNS_y
    }

    // Define other settings to be used depending on the example:
    analytical_IC_BC_forcing_term = (example_ == COUPLED_PROBLEM_EXAMPLE) ||
                                    (example_ == COUPLED_TEST_2) ||
                                    (example_ == NS_GIBOU_EXAMPLE) ||
                                    (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP); // whether or not we need to create analytical bc terms

    example_is_a_test_case = (example_ == COUPLED_PROBLEM_EXAMPLE) ||
                             (example_ == COUPLED_TEST_2) ||
                             (example_ == FRANK_SPHERE) ||
                             (example_ == NS_GIBOU_EXAMPLE)||
                             (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP);

    interfacial_temp_bc_requires_curvature = (example_ == ICE_AROUND_CYLINDER) ||
                                             (example_ == MELTING_ICE_SPHERE) ||
                                             (example_ == MELTING_ICE_SPHERE_NAT_CONV) ||
                                             (example_ == DENDRITE_TEST) ||
                                             (example_ == EVOLVING_POROUS_MEDIA);
    interfacial_temp_bc_requires_normal = (example_ == DENDRITE_TEST);

    interfacial_vel_bc_requires_vint = (example_ == ICE_AROUND_CYLINDER) ||
                                       (example_ == MELTING_ICE_SPHERE) ||
                                       (example_ == MELTING_ICE_SPHERE_NAT_CONV) ||
                                       (example_ == DENDRITE_TEST)||
                                       (example_ == EVOLVING_POROUS_MEDIA) ||
                                       (example_ == DISSOLVING_DISK_BENCHMARK);

    example_uses_inner_LSF = (example_ == ICE_AROUND_CYLINDER) || ((example_ == EVOLVING_POROUS_MEDIA) && use_inner_surface_porous_media);

    save_area_data = (example_ == ICE_AROUND_CYLINDER) ||
                                        (example_ == MELTING_ICE_SPHERE) ||
                                        (example_ == MELTING_ICE_SPHERE_NAT_CONV) ||
                                        (example_  ==DISSOLVING_DISK_BENCHMARK) || save_area_data; // change to just option to compute and save area?

    is_dissolution_case = (example_ == DISSOLVING_DISK_BENCHMARK) || is_dissolution_case; // allows for user to externally set it

    do_we_solve_for_Ts = (!is_dissolution_case); /*(example_ != DISSOLVING_DISK_BENCHMARK)*/;

    example_has_known_max_vint = ((example_ == COUPLED_PROBLEM_EXAMPLE) || (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP)
                                  || (example_ == COUPLED_TEST_2) || (example_ == FRANK_SPHERE));

    // Set the known maximum vint for relevant examples:
    switch(example_){
    case COUPLED_PROBLEM_EXAMPLE:
        max_vint_known_for_ex = PI;
        break;
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
        max_vint_known_for_ex = PI;
        break;
    case COUPLED_TEST_2:
        max_vint_known_for_ex = 1.5;
        break;
    case FRANK_SPHERE:
        max_vint_known_for_ex = 0.5;
        break;
    default:
        break;
    }
}
// ---------------------------------------
// Refinement options:
// ---------------------------------------

DEFINE_PARAMETER(pl, double, vorticity_threshold, 0.1,"Threshold to refine vorticity by, default is 0.1 \n");
//DEFINE_PARAMETER(pl, double, gradT_threshold, 1.e-4,"Threshold to refine the nondimensionalized temperature gradient by \n (default: 0.99)");

DEFINE_PARAMETER(pl, double, d2T_refine_threshold, 10.0, "Threshold by which to multiply the owning quadrant's diagonal when considering whether to refine around a sign change. Default: 3.0, i.e. we refine if a sign change is present AND the magnitude of the field value is above 3.0*quad_diag \n");

DEFINE_PARAMETER(pl, double, d2T_coarsen_threshold, 0.5, "Threshold by which to multiply the owning quadrant's diagonal when considering whether to allow coarsening around a sign change. Default: 0.5, i.e. we allow coarsening if a sign change is present AND the magnitude of the field value is below 0.5*quad_diag \n");

DEFINE_PARAMETER(pl, bool, use_uniform_band, true, "Boolean whether or not to refine using a uniform band \n");
DEFINE_PARAMETER(pl, double, uniform_band, 8., "Uniform band (default:8.) \n");
DEFINE_PARAMETER(pl, double, dxyz_close_to_interface_mult, 1.2, "Multiplier that defines dxyz_close_to_interface = mult* max(dxyz_smallest) \n");
// ---------------------------------------
// Geometry and grid refinement options:
// ---------------------------------------
// General options: // TO-DO: maybe all these geometry options should be OVERWRITE options --aka, defaults are specified per example unless user states otherwise
DEFINE_PARAMETER(pl, double, xmin, 0., "Minimum dimension in x (default: 0) \n");
DEFINE_PARAMETER(pl, double, xmax, 1., "Maximum dimension in x (default: 0) \n");

DEFINE_PARAMETER(pl, double, ymin, 0., "Minimum dimension in y (default: 0) \n");
DEFINE_PARAMETER(pl, double, ymax, 1., "Maximum dimension in y (default: 1) \n");

DEFINE_PARAMETER(pl, int, nx, 1, "Number of trees in x (default:1) \n");
DEFINE_PARAMETER(pl, int, ny, 1, "Number of trees in y (default:1) \n");

DEFINE_PARAMETER(pl, int, px, 0, "Periodicity in x (default false) \n");
DEFINE_PARAMETER(pl, int, py, 0, "Periodicity in y (default false) \n");

DEFINE_PARAMETER(pl, int, lmin, 3, "Minimum level of refinement \n");
DEFINE_PARAMETER(pl, int, lint, 0, "Intermediate level of refinement (default: 0, won't be used unless set) \n");
DEFINE_PARAMETER(pl, int, lmax, 5, "Maximum level of refinement \n");
DEFINE_PARAMETER(pl, double, lip, 1.75, "Lipschitz coefficient \n");

DEFINE_PARAMETER(pl, int, num_splits, 0, "Number of splits -- used for convergence tests \n");
DEFINE_PARAMETER(pl, bool, refine_by_ucomponent, false, "Flag for whether or not to refine by a backflow condition for the fluid velocity \n");
DEFINE_PARAMETER(pl, bool, refine_by_d2T, false, "Flag for whether or not to refine by the nondimensionalized temperature gradient \n");


// For level set:
double r0=0.;

// For frank sphere:
double s0=0.;

// For ice cylinder case:
double r_cyl; // non dim variable used to set up LSF: set in set_geometry()

std::vector<double> xshifts;
std::vector<double> yshifts;
std::vector<double> rvals;

void set_geometry(){
  switch(example_){
    case FRANK_SPHERE: {
      // Corresponds to the Frank Sphere 2d analytical solution to the Stefan problem
      // Was added to verify that the Stefan problem was being solved correctly independent of flow
      // Grid size
      xmin = -5.0; xmax = 5.0; //5.0;
      ymin = -5.0; ymax = 5.0;

      // Number of trees
      nx = 2; ny = 2;

      // Periodicity
      px = 0; py = 0;

      // Uniform band
      uniform_band=4.;

      // Problem geometry:
      s0 = 0.628649269043202;
      r0 = s0; // for consistency, and for setting up NS problem (if wanted)
      break;
      }

    case FLOW_PAST_CYLINDER: // intentionally waterfalls into same settings as ice around cylinder

    case ICE_AROUND_CYLINDER:{ // Ice layer growing around a constant temperature cooled cylinder
      // Corresponds with Section 5 of Bayat et. al -- A Sharp numerical method for the solution of Stefan problems with convective effects

      // Domain size:
      xmin = 0.0; xmax = 30.0;//20.0;/*32.0;*/
      ymin = 0.0; ymax = 15.0;//10.0;/*16.0;*/

      // Number of trees:
      nx =10.0;
      ny =5.0;

      // Periodicity:
      px = 0;
      py = 1;

      // Problem geometry:
      r_cyl = 0.5;     // Computational radius of the cylinder (mini level set)
      r0 = r_cyl*1.10; // Computational radius of ice (level set) -- TO-DO: maybe initial ice thickness should be a user parameter you can change
      break;
    }
    case MELTING_ICE_SPHERE_NAT_CONV:{
      // Domain size:
      xmin = 0.0; xmax = 3.0;
      ymin = 0.0; ymax = 3.0;

      // Number of trees:
      nx =1.0;
      ny =1.0;

      // Periodicity:
      px = 1;
      py = 0;

      // Problem geometry: hi there
      r0 = 0.5;     // Computational radius of the sphere
      break;
    }
    case MELTING_ICE_SPHERE:{
      // (WIP)-- was originally set up to try and validate Hao et al melting ice sphere experiments. TO-DO: can revisit this! now that BC's are corrected and etc.
      // Domain size:
      xmin = 0.0; xmax = 30.0;
      ymin = 0.0; ymax = 15.0;

      // Number of trees:
      nx =10.0;
      ny =5.0;

      // Periodicity:
      px = 0;
      py = 1;

      // Problem geometry:
      r0 = 0.5;     // Computational radius of the sphere
      break;
    }
    case EVOLVING_POROUS_MEDIA:{
      // EASIER TO DO LOCALLY:
//      xmin = 0.0; xmax = 2.0;
//      ymin = 0.0; ymax = 2.0;
      // ALLOW THE USER TO DECIDE VIA INPUT

//      // Number of trees:
//      nx = 1.0;
//      ny = 1.0;

      // Periodicity:
      px = 0;
      py = 1;

      // Problem geometry:
      r0 = 0.1;     // Computational radius of the sphere // to-do: check this, p sure it gets ignored
      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      // Domain size:
      xmin = 0.0; xmax = 10.0;
      ymin = 0.0; ymax = 5.0;


      // Number of trees:
      nx = 2.0;
      ny = 1.0;

      // Periodicity:
      px = 0;
      py = 0;

      // Problem geometry:
      r0 = 1.0; // radius of the disk ( in comp domain) -- r = 0.01 cm, so 1.0 unit in comp domain = 0.1 mm in physical
      break;

    }


    case PLANE_POIS_FLOW:{
      xmin = 0.0; xmax = 20.0;
      ymin = 0.0; ymax = 10.0;

      // Number of trees:
      nx = 4.0;
      ny = 2.0;

      // periodicity:
      px = 0; py = 0;

      // problem geometry:
      r0 = 1.0; // this is used as height above ymin of interface (interface is flat plate in this case)
      break;
    }
    case NS_GIBOU_EXAMPLE: {
      // Corresponds with Section 4.1.1 from Guittet et al. - A stable projection method for the incompressible Navier-Stokes equations on arbitrary geometries and adaptive Quad/Octrees
      // Was added here to verify that the NS was working correctly independently
      // Domain size:
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // Radius of the level set function:
      r0 = 0.20;
      break;
    }

    case COUPLED_PROBLEM_EXAMPLE:{
      // Corresponds with Section 6 of Bayat et. al -- A Sharp numerical method for the solution of Stefan problems with convective effects
      // Domain size:
      xmin = -PI; xmax = PI;
      ymin = -PI; ymax = PI;

      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // Radius of the level set function:
      r0 = PI/2.;

      break;
    }
    case COUPLED_TEST_2:{
      // An additional coupled test that was not used in the paper.
      // To-do: revisit this, or remove it
      // Domain size:
      xmin = -1.0; xmax = 1.0;
      ymin = -1.0; ymax = 1.0;
      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      uniform_band = 4.;
      break;
    }
    case DENDRITE_TEST:{
      // TO-DO: clean this out
      // (WIP) : was added to further verify coupled solver by demonstrating dendritic solidification of a pure substance, but never fully fledged this out.
      // Domain size:
      xmin = 0.; xmax = 3200.;
      ymin = 0.; ymax = 3200.;

      // Number of trees and periodicity:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // level set size (initial seed size)
      r0 = 30.; // to match the paper we are comparing with

      // capillary length scale and etc is set in set_physical_properties() 5/3/21
      break;
    }
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
      // Corresponds with Section 6 of Bayat et. al -- A Sharp numerical method for the solution of Stefan problems with convective effects
      // Domain size:
      xmin = -PI; xmax = PI;
      ymin = -PI; ymax = PI;

      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // Radius of the level set function:
      r0 = PI/2.;

      break;
     }
  }
}

// Define a few parameters for the porous media case to create random grains:
DEFINE_PARAMETER(pl, int, num_grains, 10., "Number of grains in porous media (default: 10) \n");
void make_LSF_for_porous_media(mpi_environment_t &mpi){
    // initialize random number generator:
    srand(1);
    // Resize the vectors on all processes appropriately so we all have consistent sizing (for broadcast later)
    xshifts.resize(num_grains);
    yshifts.resize(num_grains);
    rvals.resize(num_grains);

    if(mpi.rank() == 0){

      const char* geom_dir = getenv("GEOMETRY_DIR");

      if(geom_dir == nullptr){
        throw std::invalid_argument("You need to set the environment variable for the geometry: GEOMETRY_DIR \n");
      }
//      char geom_x[]= ;
//      geom_x=
      char geom_x[PATH_MAX]; char geom_y[PATH_MAX]; char geom_r[PATH_MAX];
      sprintf(geom_x, "%s/xshifts.txt", geom_dir);
      sprintf(geom_y, "%s/yshifts.txt", geom_dir);
      sprintf(geom_r, "%s/rvals.txt", geom_dir);
      std::ifstream infile_x(geom_x);
      // Read x data:
      if(infile_x){
        double curr_val;

        int count = 0;
        while(infile_x >> curr_val){
          xshifts[count] = curr_val;
          count+=1;
        }

        if(count!=num_grains){
          throw std::invalid_argument("The number of grains inputted does not match the number provided in the input file \n");
        }
      }
      std::ifstream infile_y(geom_y);

      // Read y data:
      if(infile_y){
        double curr_val;
        int count = 0;
        while(infile_y >> curr_val){
          yshifts[count] = curr_val;
          count+=1;
        }
        if(count!=num_grains){
          throw std::invalid_argument("The number of grains inputted does not match the number provided in the input file \n");
        }
      }
      std::ifstream infile_r(geom_r);

      // Read r data:
      if(infile_r){
        double curr_val;

        int count=0;
        while(infile_r >> curr_val){
         rvals[count]=curr_val;
         count+=1;

        }
        if(count!=num_grains){
          throw std::invalid_argument("The number of grains inputted does not match the number provided in the input file \n");
        }
      }
      printf("end of operation on rank 0 \n");
    } // end of defining the grains on rank 0, now we need to broadcast the result to everyone

    // Tell everyone else what we came up with!
    int mpi_err;
    mpi_err = MPI_Bcast(xshifts.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);
    mpi_err = MPI_Bcast(yshifts.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);
    mpi_err = MPI_Bcast(rvals.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);

}

double return_LSF_porous_media(DIM(double x, double y, double z), bool is_inner_){
  // bool is_inner_ corresponds to whether or not the LSF we are returning is for the inner or the outer LSF, in the case that we are using an inner and outer LSF (aka initial deposit layer, etc)

  //  double radius_multiplier=1.0;
//  if(!is_inner_) radius_multiplier = porous_media_initial_thickness_multiplier;


  double tree_size = (xmax-xmin)/nx; // assuming only square trees
  double dxyz_min = tree_size/(pow(2.0, lmax));

  double radius_addition = 0.0;
  if(!is_inner_) radius_addition = porous_media_initial_thickness*dxyz_min;

  double lsf_vals[num_grains];
  // First, grab all the relevant LSF values for each grain:
  for(int n=0; n<num_grains; n++){
    double r = sqrt(SQR(x - xshifts[n]) + SQR(y - yshifts[n]));
    //lsf_vals[n] = radius_multiplier*rvals[n] - r;
    // above wasn't working, I want the initial thickness to be the same (regardless of radius), so I'm gonna just try and add the distance of approx 4 grid cells and see where that lands us

    lsf_vals[n] = (rvals[n] + radius_addition) - r;

  }

  // Now, loop back over and return the value which has the min magnitude:
  double current_min = 1.e9;
  for(int n=0; n<num_grains; n++){
    if(fabs(lsf_vals[n]) < fabs(current_min)){
      current_min = lsf_vals[n];
    }
  }
  return current_min;

}



double v_interface_max_norm=0.; // For keeping track of the interfacial velocity maximum norm


// ---------------------------------------
// Physical properties/General problem parameters:
// ---------------------------------------
// For solidification problem:
//double alpha_s; // Thermal diffusivity of solid [m^2/s]
//double alpha_l; // Thermal diffusivity of liquid [m^2/s]
//double k_s;     // Thermal conductivity of solid [W/(mK)]
//double k_l;     // Thermal conductivity of liquid [W/(mK)]
//double L;       // Latent heat of fusion [J/kg]
//double rho_l;   // Density of fluid [kg/m^3]
//double rho_s;   // Density of solid [kg/m^3]
//double cp_s;    // Specific heat of solid [J/(kg K)]
//double mu_l;    // Dynamic viscosity of fluid [Pa s]

DEFINE_PARAMETER(pl, double, l_char, 0., "Characteristic length scale for the problem (in meters). i.e. For okada flow past cylinder, this should be set to the cylinder diameter \n. ");

DEFINE_PARAMETER(pl, double, alpha_l, 1.0, "Thermal diffusivity of liquid [m^2/s]. Default: 1. \n"
                                           "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, alpha_s, 1.0, "Thermal diffusivity of solid [m^2/s]. Default: 1. \n"
                                           "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, k_l, 1.0, "Thermal conductivity of liquid [W/(mK)]. Default: 1. \n"
                                           "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, k_s, 1.0, "Thermal conductivity of solid [W/(mK)]. Default: 1. \n"
                                           "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, rho_l, 1.0, "Density of fluid [kg/m^3]. Default: 1. \n"
                                       "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, rho_s, 1.0, "Density of solid [kg/m^3]. Default: 1. \n"
                                       "This property is set inside specific examples. \n");

DEFINE_PARAMETER(pl, double, cp_s, 1.0, "Specific heat of solid [J/(kg K)]. Default: 1. \n"
                                       "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, L, 1.0, "Latent heat of fusion [J/kg]. Default: 1. \n"
                                       "This property is set inside specific examples. \n");
DEFINE_PARAMETER(pl, double, mu_l, 1.0, "Dynamic viscosity of fluid [Pa s]. Default: 1. \n"
                                       "This property is set inside specific examples. \n");

DEFINE_PARAMETER(pl,double,sigma,4.20e-10,"Interfacial tension [m] between ice and water, default: 2*2.10e-10 \n");


DEFINE_PARAMETER(pl, double, grav, 9.81, "Gravity (m/s^2). Default: 9.81. \n");
DEFINE_PARAMETER(pl, double, beta_T, 1.0, "Thermal expansion coefficient for the boussinesq approx. default: 1 . This gets set inside specific examples. \n ");
DEFINE_PARAMETER(pl, double, beta_C, 1.0, "Concentration expansion coefficient for the boussinesq approx. default: 1 . This gets set inside specific examples. \n ");

// For dissolution problem:
DEFINE_PARAMETER(pl, double, gamma_diss, 1.0, "The parameter dictates some dissolution behavior, default value is 1. This gets calculated internally for the dissolution benchmark problem, otherwise it's up to the user to set. \n");

DEFINE_PARAMETER(pl, double, stoich_coeff_diss, 1.0, "The stoichiometric coefficient of the dissolution reaction. Default is 1. Used to compute gamma_diss for dissolution benchmark problem. \n");

DEFINE_PARAMETER(pl, double, molar_volume_diss, 1.0, "The molar volume of the dissolving solid. Default is 1. Used to compute gamma_diss for dissolution benchmark problem. \n");

DEFINE_PARAMETER(pl, double , Dl, -1., "Liquid phase diffusion coefficient m^2/s, default is : 9e-4 mm2/s = 9e-10 m2/s \n");
DEFINE_PARAMETER(pl, double , Ds, 0., "Solid phase diffusion coefficient m^2/s, default is : 0 \n");
// Elyce commented out 12/14/21: the below parameter is now obselete
//DEFINE_PARAMETER(pl, double, l_diss, 2.0e-4, "Dissolution length scale. The physical length (in m) that corresponds to a length of 1 in the computational domain. Default: 20e-3 m (20 mm), since the initial diameter of the disk is 20 mm \n");

DEFINE_PARAMETER(pl, double, k_diss, 1.0, "Dissolution rate constant per unit area of reactive surface (m/s). Default 4.5e-3 mm/s aka 4.5e-6 m/s \n");

// Scalar temp/conc problem parameters:
DEFINE_PARAMETER(pl, double, T0, 0., "Characteristic solid temperature of the problem. Usually corresponds to the solid phase. i.e.) For ice growth over cooled cylinder example, this refers to temperature of cooled cylinder in K. For the melting ice sphere example, this refers to the initial temperature of the ice in K (default: 0). This must be specified by the user. \n");

DEFINE_PARAMETER(pl, double, Tinterface, 0.5, "The interface temperature (or concentration) in K (or INSERT HERE), i.e. the melt temperature. (default: 0.5 This needs to be set by the user to run a meaningful example). \n");

DEFINE_PARAMETER(pl, double, Tinfty, 1., "The freestream fluid temperature T_infty in K. (default: 1. This needs to be set by the user to run a meaningful example). \n");

DEFINE_PARAMETER(pl, double, theta_infty, 1., "The freestream temp or concentration, nondimensional. Default:1 . This may not be used, but in the dissolution porous media case allows the user to control this as an input variable. \n");

DEFINE_PARAMETER(pl, double, theta_initial, 0., "This allows you to set the initial nondim C value for the evolving porous media disso/precip problem in the fluid. Default: 0. \n ");

DEFINE_PARAMETER(pl, double, theta0, 0., "This allows you to set the initial nondim C value for the evolving porous media disso/precip problem in the fluid. Default: 0. \n ");

DEFINE_PARAMETER(pl, double, Tflush, -1.0, "The flush temperature (K) or concentration that the inlet BC is changed to if flush_dim_time is activated. Default: -1.0. \n");

DEFINE_PARAMETER(pl, double, theta_flush, -1.0, "The nondim flush temperature or concentration that the inlet BC is changed to if flush_dim_time is activated. Default: -1.0. \n");

DEFINE_PARAMETER(pl, double, flush_every_dt, -1.0, "Flush domain with theta_flush every X amount of dimensional time \n ");

DEFINE_PARAMETER(pl, double, flush_duration, -1.0, "Duration of each flush \n");

// Function to setup the physical properties (for diff examples):
void set_physical_properties(){
  switch(example_){
    case FRANK_SPHERE:{
      alpha_s = 1.0;
      alpha_l = 1.0;
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      rho_s = 1.0;

      // Necessary boundary condition info:
      Tinfty = -0.2;

      Tinterface = 0.0;

      break;
    }
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:{
      alpha_s = (1.18e-6);    //ice
      alpha_l = (0.13275e-6); //water
      k_s = 2.22;       // W/[m*K]
      k_l = 558.61e-3;  // W/[m*K]

      rho_l = 1000.0;   // kg/m^3
      rho_s = 920.;     //[kg/m^3]


      mu_l = 0.001730725; // [Pa * s]
      cp_s = k_s/(alpha_s*rho_s); // [J/(kg K)]

      L = 334.e3;  // J/kg
      sigma = (4.20e-10); // [m] // changed from original 2.10e-10 by alban
      break;
      }
    case EVOLVING_POROUS_MEDIA:
      // TO-DO: intentionally waterfalling for now, will change once i fine tune the example more
      if(is_dissolution_case){
        rho_l = 1000; // kg/m^3
        mu_l = 1.00160e-3; // Pa s
        rho_s = 2700; // kg/m^3
      }
      //otherwise waterfall
    case MELTING_ICE_SPHERE:{

      // Using properties of water at 20 C: (engineering toolbox)
      alpha_l = 0.143e-6; // m2/s
      k_l = 598.03e-3; // W/mK
      rho_l = 1000; // kg/m^3
      mu_l = 1.00160e-3; // Pa s

      // Using properties of ice at -10 C:
      k_s = 2.30; // W/mK
      cp_s = 2.00e3; // J/kgK
      rho_s = 918.9; // kg/m^3

      alpha_s = k_s/cp_s/rho_s; // m^2/s

      L = 334.e3;  // J/kg
      sigma = (4.20e-10); // [m] // changed from original 2.10e-10 by alban

      break;
    }
    case MELTING_ICE_SPHERE_NAT_CONV:{
      // Using properties of water at 20 C: (engineering toolbox)
      alpha_l = 1.00160e-6; // m2/s
      k_l = 598.03e-3; // W/mK
      rho_l = 1000; // kg/m^3
      mu_l = 1.00160e-3; // Pa s

      // Using properties of ice at -10 C:
      k_s = 2.30; // W/mK
      cp_s = 2.00e3; // J/kgK
      rho_s = 918.9; // kg/m^3

      alpha_s = k_s/cp_s/rho_s; // m^2/s

      L = 334.e3;  // J/kg
      sigma = (4.20e-10); // [m] // changed from original 2.10e-10 by alban

      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      mu_l = 1.0e-3; // Pa s // back calculated using experimental Re and u0 reported
      rho_l = 1000.0;
      rho_s = 2710.0;

//      gamma_diss = molar_volume_diss*Tinfty/stoich_coeff_diss;
      break;
    }
    case PLANE_POIS_FLOW:{
      break;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      break;
    }
    case NS_GIBOU_EXAMPLE:{
      break;
    }
    case DENDRITE_TEST:{
      mu_l=1.0;//660000;
      rho_l=1.0;//1000;
      alpha_l=1.0;//28.5714;
      // can be specified by the user
      break;
    }

    } // end of switch cases
}


// ---------------------------------------
// Related to the nondimensional problem / nondimensionalization:
// ---------------------------------------
DEFINE_PARAMETER(pl, double, Re, -1., "Reynolds number (rho Uinf d)/mu, where d is the characteristic length scale - default is 0. \n");
DEFINE_PARAMETER(pl, double, Pr, -1., "Prandtl number - computed from mu_l, alpha_l, rho_l \n");
DEFINE_PARAMETER(pl, double, Sc, -1., "Schmidt number - computed from mu_l, D, rho_l \n");

DEFINE_PARAMETER(pl, double, Pe, -1., "Peclet number - computed from Re and Pr \n");
DEFINE_PARAMETER(pl, double, St, -1., "Stefan number (cp_s deltaT/L)- computed from cp_s, deltaT, L \n");

DEFINE_PARAMETER(pl, double, Da, -1., "Damkohler number (k_diss*l_char/D_diss) \n");

DEFINE_PARAMETER(pl, double, RaT, -1., "Rayleigh number by temperature (default:0) \n");
DEFINE_PARAMETER(pl, double, RaC, -1., "Rayleigh number by concentration (default:0) \n");


DEFINE_PARAMETER(pl,double,Gibbs_eps4,0.005,"Gibbs Thomson anisotropy coefficient (default: 0.005), applicable in dendrite test cases \n");

problem_dimensionalization_type_t problem_dimensionalization_type;
// NONDIM_BY_FLUID_VELOCITY -- corresponds to nondimensionalization where the velocities in the problem are nondimensionalized by
//                             a characteristic fluid velocity, and Reynolds number is used to setup the Navier-Stokes equations

// NONDIM_BY_SCALAR_DIFFUSIVITY -- corresponds to nondimensionalization where the velocities in the problem are nondimensionalized by
//                          a characteristic velocity defined by the fluid's thermal or concentration diffisuvity and char. length scale,
//                          and Prandtl/Schmidt number is used to setup the Navier-Stokes equations

// DIMENSIONAL -- corresponds to solving the dimensional problem

void select_problem_nondim_or_dim_formulation(){
  switch(example_){
  case FRANK_SPHERE:{
    problem_dimensionalization_type = DIMENSIONAL;
    break;
  }
  case NS_GIBOU_EXAMPLE:{
    problem_dimensionalization_type = DIMENSIONAL;
    break;
  }
  case COUPLED_TEST_2:{
    problem_dimensionalization_type = DIMENSIONAL;
    break;
  }
  case COUPLED_PROBLEM_EXAMPLE:{
    problem_dimensionalization_type = DIMENSIONAL;
    break;
  }
  case ICE_AROUND_CYLINDER:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
  case FLOW_PAST_CYLINDER:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
  case DENDRITE_TEST:{
    problem_dimensionalization_type = NONDIM_BY_SCALAR_DIFFUSIVITY;
    break;
  }
  case MELTING_ICE_SPHERE:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
  case EVOLVING_POROUS_MEDIA:{
    break;
  }
  case PLANE_POIS_FLOW:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
  case DISSOLVING_DISK_BENCHMARK:{
    problem_dimensionalization_type = NONDIM_BY_SCALAR_DIFFUSIVITY; // elyce to-do : will want to run this benchmark using the other formulation as well
    break;
  }
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
  case MELTING_ICE_SPHERE_NAT_CONV:{
    problem_dimensionalization_type = NONDIM_BY_FLUID_VELOCITY;
    break;
  }
    // Elyce to-do: add a case for Rochi's example
  default:{
    throw std::runtime_error("main_2d.cpp:select_problem_nondim_or_dim_formulation: example is unrecognized or has not been set up \n");
  }
  } // end of switch case
  if(nondim_type_used>=0){
    problem_dimensionalization_type = (problem_dimensionalization_type_t)nondim_type_used;
  }
};

// For defining appropriate nondimensional groups:
double time_nondim_to_dim = 1.;
double vel_nondim_to_dim = 1.;

// Nondimensional temperature values (computed in set_physical_properties)
//double theta_infty=0.; // I've allowed the user to set this
double theta_interface=0.;
//double theta0=0.;
double deltaT=0.;

void set_temp_conc_nondim_defns(){
  switch(example_){
  case FRANK_SPHERE:{
    break;
  }
  case NS_GIBOU_EXAMPLE:{
    break;
  }
  case COUPLED_TEST_2:{
    break;
  }
  case COUPLED_PROBLEM_EXAMPLE:{
    break;
  }
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
    break;
  }
  case ICE_AROUND_CYLINDER:{
    deltaT = fabs(Tinfty - T0);

    theta0 = 0.0;
    theta_infty = 1.0;

    theta_interface = (Tinterface - T0)/deltaT;
    break;
  }
  case FLOW_PAST_CYLINDER:{
    break;
  }
  case DENDRITE_TEST:{
    deltaT = T0 - Tinfty;

    theta0 = 1.0;
    theta_infty = 0.0;

    theta_interface = (Tinterface - Tinfty)/deltaT;

    break;
  }
  case MELTING_ICE_SPHERE:{
    deltaT = fabs(Tinfty - T0);

    theta0 = 0.0;
    theta_infty = 1.0;

    theta_interface = (Tinterface - T0)/deltaT;

    break;
  }
  case MELTING_ICE_SPHERE_NAT_CONV:{
    deltaT = fabs(Tinfty - T0);

    theta0 = 0.0;
    theta_infty = 1.0;

    theta_interface = (Tinterface - T0)/deltaT;
    break;
  }

  case EVOLVING_POROUS_MEDIA:{

    if(is_dissolution_case){
//      // Using the nondim setup theta = C/Cinf
//      theta_infty=1.0; // wall undersaturation
//      theta0 = 0.0; // used for IC -- initial concentration of ions in solid is zero -- verified by Molins benchmark paper

      // Using nondim theta = C/Csat
//      theta_infty = theta_inf_prescribed;
      // We allow this to be set by the user

      // We don't need deltaT in the actual equations, but we still set it for refinement purposes
      deltaT = fabs(theta_infty - theta_initial);


    }
    else{
      deltaT = fabs(Tinfty - T0);

      theta0 = 0.0;
      theta_infty = 1.0;

      theta_interface = (Tinterface - T0)/deltaT;
    }

    break;
  }
  case PLANE_POIS_FLOW:{
    break;
  }
  case DISSOLVING_DISK_BENCHMARK:{
    deltaT = 0.; // setting deltaT to zero will default to the C/Cinf formulation of nondim conc.
    // there is an if statement in the robin B.C. that will set it accordingly, if you use deltaT nonzero it will automatically change the B.C. to be compatible w that formulation
    // same goes for gamma_diss --> we will define it accordingly in set_nondim_groups

    // Using the nondim setup theta = C/Cinf
    theta_infty=1.0; // wall undersaturation
    theta0 = 0.0; // used for IC -- initial concentration of ions in solid is zero -- verified by Molins benchmark paper

    // no need to set theta_interface --> we have a robin BC there
    break;
  }
  // Elyce to-do: add a case for Rochi's example
  default:{
    throw std::runtime_error("main_2d.cpp:select_problem_nondim_or_dim_formulation: example is unrecognized or has not been set up \n");
  }
  }
}

// For solution of temperature fields:
DEFINE_PARAMETER(pl, double, back_wall_temp_flux, 0.0, "Temperature flux at back wall. Default: 0.0 \n");

// Maximum allowed v_interface value:
DEFINE_PARAMETER(pl, double, v_int_max_allowed, 50.0, "Max allowed v_interface value. Default: 50 \n");

//-----------------------------------------
// Properties to set if you are solving NS
// ----------------------------------------

// For velocity BCs/ICs
double u0=0.;
double v0=0.;
double outflow_u=0.;
double outflow_v=0.;

DEFINE_PARAMETER(pl, double, u_inf, 0., "Freestream velocity value (in m/s). Default is 0. This is usually overwritten bc it is computed by a provided Reynolds number. However, in dissolving disk benchmark example with diffusivity nondimensionalization, this can be used to pass in the freestream fluid boundary condition at the wall. \n"); // physical value of freestream velocity


//to-do: (not sure if this comment is still relevant--> )decide whether to keep this pressure drop option. will need to add logic to either select pressure drop OR overwrite Reynolds, but not both
DEFINE_PARAMETER(pl, double, pressure_drop, 1.0, "The dimensional pressure drop value you are using, in Pa. This value will be used in conjunction with the nondim length scale to compute wall Reynolds number for relevant examples using a channel flow-type setup(i.e. melting porous media). Default: 1.0.");

// For setting hodge criteria:
DEFINE_PARAMETER(pl, double, hodge_percentage_of_max_u, 1.e-3, "Percentage of the max NS norm that hodge variable has to converge within. Default: 1.e-2.");

// For keeping track of hodge error:
double hodge_global_error;

DEFINE_PARAMETER(pl, double, NS_max_allowed, 100., "Max allowed NS norm before throwing a blow up error. Default: 100. \n");

// For okada example, amount to perturb initial flow (to induce vortex shedding)
double perturb_flow_noise =0.25;


// TO-DO: clean this out so we have only what is actually used
void set_NS_info(){

  // Note: fluid velocity is set via Re and u0,v0 --> v0 = 0 is equivalent to single direction flow, u0=1, v0=1 means both directions will flow at Re (TO-DO:make this more clear)
  switch(example_){
    case FRANK_SPHERE:throw std::invalid_argument("NS isnt setup for this example");
    case EVOLVING_POROUS_MEDIA:{
      u0 = 0.0;
      v0 = 0.;

//      G_press = 1.0; // Pressure drop of order 1

      hodge_percentage_of_max_u = 1.e-3;

      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      switch(problem_dimensionalization_type){
        case NONDIM_BY_FLUID_VELOCITY:{
          u0 = 1.0;
          v0 = 0.;
          break;
        }
        case NONDIM_BY_SCALAR_DIFFUSIVITY:{
          u0 = (u_inf)/(Dl/l_char);

          v0 = 0.;
          break;
        }
        default:{
          throw std::runtime_error("unrecognized problem dimensionalization type for set_NS_info() \n");
        }
      }



      hodge_percentage_of_max_u=1.e-3;
      break;
    }

    case FLOW_PAST_CYLINDER:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:{

      u0 = 1.0; // computational freestream velocity
      v0 = 0.0;
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case MELTING_ICE_SPHERE_NAT_CONV:{
      Re = 316.;
      u0 = 0.0; // computational freestream velocity
      v0 = 0.0;
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case PLANE_POIS_FLOW:{
      Re = 1.0; // this will get overwritten
      u0 = 1.0;//5.0625;
      throw std::invalid_argument("The pressure drop condition has not been correctly set for this plane pois flow example. Please update this before using. \n");
      //G_press = 16.*Re/SQR(ymax-(r0 + ymin)); // selecting to yield uavg = 1, r0 = height of interface (y-wise)
      // to-do : fix this pois example now that we are using wall reynolds for channel flow type problems ...
      v0 = 0.;
      hodge_percentage_of_max_u=1.0e-3;
      break;
    }
    case NS_GIBOU_EXAMPLE:{
      Re = 1.0;

      u0 = 1.0;
      v0 = 1.0;

      u_inf = 1.0; // u_inf usually corresponds to a physical value for velocity, but this example doesnt have that //to-do: is this actually used in this example?
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      Re = 1.0;
      u0 = 1.0;
      v0 = 1.0;
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case DENDRITE_TEST:{
      u0 = 0.;
      v0 = -0.035;
      hodge_percentage_of_max_u = 1.e-2;
      break;
    }
  }
  outflow_u = 0.0;
  outflow_v = 0.0;
}

// ---------------------------------------
// For setting up simulation timing information:
// ---------------------------------------
double tfinal;
double dt_max_allowed;

double tn;
double tstart;
double dt = 1.e-5;
int tstep;
double dt_min_allowed = 1.e-5; // TO-DO: make this compatible w stefan_w_fluids solver?

DEFINE_PARAMETER(pl,double,t_ramp,0.1,"Time at which boundary conditions are ramped up to their desired value [input should be dimensional time, in seconds] (default: 3 seconds) \n");
DEFINE_PARAMETER(pl,bool,ramp_bcs,false,"Boolean option to ramp the BCs over a specified ramp time (default: false) \n");
DEFINE_PARAMETER(pl,int,startup_iterations,-1,"Number of startup iterations to do before entering real time loop, used for verification tests to allow v_interface and NS fields to stabilize. Default:-1, to use this, set number to positive integer value.");
DEFINE_PARAMETER(pl,double,startup_nondim_time,-10.0,"Startup time in nondimesional time, before the simulation allows interfacial growth to occur (Default : 0)");
DEFINE_PARAMETER(pl,double,startup_dim_time,-10.0,"Startup time in dimesional time (seconds), before the simulation allows interfacial growth to occur (Default : 0)");

DEFINE_PARAMETER(pl, double, flush_dim_time, -10.0, "Time (in seconds) at which the domain is flushed with a new wall boundary condition for temperature (or concentration) at the inlet. This condition is specified by Tflush. If this is greater than zero, it will be activated. Default:-10(unused) \n");

DEFINE_PARAMETER(pl,bool,perturb_initial_flow,false,"Perturb initial flow? For melting refinement case. Default: true. Applies to initial condition for velocity field. ");

void simulation_time_info(){
  // TO-DO: handle ramp case now that we are using class
//  t_ramp /= time_nondim_to_dim; // divide input in seconds by time_nondim_to_dim because we are going from dim--> nondim
//  save_every_dt/=time_nondim_to_dim; // convert save_every_dt input (in seconds) to nondimensional time


  // dt_max_allowed will be set according to the grid size. If a different value is desired, the user may overwrite it below by simply defining it as something else in the relevant example block
  dt_max_allowed = ((xmax-xmin)/nx)/(pow(2., lmax))*10.;

  switch(example_){
    case FRANK_SPHERE:{
      tfinal = 1.1;
      dt_max_allowed = 0.05;
      tstart = 1.0;
      break;
    }
    case EVOLVING_POROUS_MEDIA:
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER: {
      // ice solidifying around isothermally cooled cylinder
      tfinal = (40.*60.)/(time_nondim_to_dim); // 40 minutes

      tstart = 0.0;
      break;
    }
    case MELTING_ICE_SPHERE_NAT_CONV:
    case MELTING_ICE_SPHERE:{
      //tfinal = (2.*60)/(time_nondim_to_dim); // 2 minutes
      tfinal = (1000.0*60)/(time_nondim_to_dim);; // 1000 in nondim time for refinement test
      tstart = 0.0;

      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      tstart = 0.0;
      dt_max_allowed = 10.0;
//      dt_max_allowed = 0.01;

      dt = 1.0e-3; // initial timestep
      break;
    }

    case NS_GIBOU_EXAMPLE:{
      //tfinal = 0.025;
      tfinal = PI/3.;

      tstart = 0.0;
      break;
    }
    case PLANE_POIS_FLOW:{
      tfinal = 100.;
      tstart=0.0;
      break;
    }

    case COUPLED_PROBLEM_EXAMPLE:{
      tfinal = PI/3.;//PI/2.;
      tstart = 0.0;
      break;
    }
    case COUPLED_TEST_2:{
      tfinal = 0.75;//1.;
      tstart=0.0;
      break;
    }
    case DENDRITE_TEST:{
      tfinal = 10000000./time_nondim_to_dim;
      tstart=0.;
      dt_max_allowed = 1000.; // unrestricted for now

      break;
    }
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
      tfinal = PI/3.;//PI/2.;
      tstart = 0.0;
      break;
    }
  }

  // Overwrite dt max allowed if we are saving a certain dt:
  if(save_using_dt){
    dt_max_allowed = min(dt_max_allowed, save_every_dt - EPS);
  }

  if((duration_overwrite>0.) || (duration_overwrite_nondim>0.)){
    if((duration_overwrite>0.) && (duration_overwrite_nondim>0.)){
      throw std::invalid_argument("You have selected BOTH a dimensional and nondimensional duration overwrite. Please only select one \n");
    }

    if(duration_overwrite>0.) {
      tfinal = (duration_overwrite*60.)/(time_nondim_to_dim); // convert input in minutes to nondimensional time
    }
    else{
      tfinal = duration_overwrite_nondim;
    }
  }
}

// ---------------------------------------
// Other parameters:
// ---------------------------------------

bool is_ice_melted = false; // Boolean for checking if the ice is melted for melting ice sphere example

// ----------------
// For the coupled test case where we have to swtich sign:
// ----------------
double coupled_test_sign=1; // in header -- to define only for that example
bool vel_has_switched=false;
void coupled_test_switch_sign(){coupled_test_sign*=-1.;}


// --------------------------------------------------------------------------------------------------------------
// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
/* Frank sphere functions --
 * Functions necessary for evaluating the analytical
 * solution of the Frank sphere problem
*/
// --------------------------------------------------------------------------------------------------------------
double s(double r, double t){
  //std::cout<<"Time being used to compute s is: " << t << "\n"<< std::endl;
  return r/sqrt(t);
}

// Error function : taken from existing code in examples/stefan/main_2d.cpp
double E1(double x)
{
  const double EULER=0.5772156649;
  const int    MAXIT=100;
  const double FPMIN=1.0e-20;

  int i,ii;
  double a,b,c,d,del,fact,h,psi,ans=0;

  int n   =1;
  int nm1 =0;

  if (x > 1.0)
  {        /* Lentz's algorithm */
    b=x+n;
    c=1.0/FPMIN;
    d=1.0/b;
    h=d;
    for (i=1;i<=MAXIT;i++)
    {
      a = -i*(nm1+i);
      b += 2.0;
      d=1.0/(a*d+b);    /* Denominators cannot be zero */
      c=b+a/c;
      del=c*d;
      h *= del;
      if (fabs(del-1.0) < EPS)
      {
        ans=h*exp(-x);
        return ans;
      }
    }
    //printf("Continued fraction failed in expint\n");
  }
  else
  {
    ans = (nm1!=0 ? 1.0/nm1 : -log(x)-EULER);    /* Set first term */
    fact=1.0;
    for (i=1;i<=MAXIT;i++)
    {
      fact *= -x/i;
      if (i != nm1) del = -fact/(i-nm1);
      else
      {
        psi = -EULER;  /* Compute psi(n) */
        for (ii=1;ii<=nm1;ii++) psi += 1.0/ii;
        del=fact*(-log(x)+psi);
      }
      ans += del;
      if (fabs(del) < fabs(ans)*EPS) return ans;
    }
    printf("series failed in expint\n");
    printf("x value used was : %0.5f",x);

  }

  return ans;
}


double F(double s){
  double z = SQR(s)/4.0;
  return E1(z);
}

double dF(double s){
  double z = SQR(s)/4.0;
  return -0.5*s*exp(z)/z;
}

double frank_sphere_solution_t(double s){

  if (s<s0) return 0;
  else      return Tinfty*(1.0 - F(s)/F(s0));


}

//------------------------------------------------------------------------
// Functions/Structures for validating the analytical test cases:
// -----------------------------------------------------------------------
struct temperature_field: CF_DIM
{
  const double factor = 3.;
  const double N = 1.;
  const unsigned char dom; //dom signifies which domain--> domain liq = 0, domain solid =1
  temperature_field(const unsigned char& dom_) : dom(dom_){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }

  double T(DIM(double x, double y, double z)) const{
    switch(example_){
      case COUPLED_PROBLEM_EXAMPLE:{
        switch(dom){
        case LIQUID_DOMAIN:
          return sin(x)*sin(y)*(x + cos(t)*cos(x)*cos(y));
        case SOLID_DOMAIN:
          return cos(x)*cos(y)*(cos(t)*sin(x)*sin(y) - 1.);
        default:
          throw std::runtime_error("analytical solution temperature: unknown domain \n");
        }
      }
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
        switch(dom){
        case LIQUID_DOMAIN:
          return sin(x)*sin(y)*(x + cos(t)*cos(x)*cos(y));
        case SOLID_DOMAIN:
          return cos(x)*cos(y)*(cos(t)*sin(x)*sin(y) - 1.);
        default:
          throw std::runtime_error("analytical solution temperature: unknown domain \n");
        }
      }
      case COUPLED_TEST_2:{
        switch(dom){
        case LIQUID_DOMAIN:
          return ((x-1.)*(y+1.)/factor + cos(N*PI*x*y*t));//sin(2.*SQR(t)*PI)*SQR(x*y) + ((x-1.)*(y+1.))/factor;// cos(PI*x*y*t);
        case SOLID_DOMAIN:
          return (-1.*(x-1.)*(y-1.)/factor + cos(N*PI*x*y*t));//sin(2.*SQR(t)*PI)*SQR(x*y) - ((x-1.)*(y-1.))/factor; // sin(PI*x*y*t);//
        default:
          throw std::runtime_error("analytical solution temperature: unknown domain \n");
        }
      }
      default:{
        throw std::runtime_error("analytical solution temperature: unkown example \n");
      }
    }
  }
  double operator()(DIM(double x, double y, double z)) const{ // Returns the velocity field
    return T(DIM(x,y,z));
    }
  double dT_d(const unsigned char& dir,DIM(double x, double y, double z)){
    switch(example_){
      case COUPLED_PROBLEM_EXAMPLE:{
        switch(dom){
        case LIQUID_DOMAIN:
          switch(dir){
          case dir::x:
            return cos(x)*sin(y)*(x + cos(t)*cos(x)*cos(y)) - sin(x)*sin(y)*(cos(t)*cos(y)*sin(x) - 1.);
          case dir::y:
            return cos(y)*sin(x)*(x + cos(t)*cos(x)*cos(y)) - cos(t)*cos(x)*sin(x)*SQR(sin(y));

          default:
            throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
          }
        case SOLID_DOMAIN:
          switch(dir){
          case dir::x:
            return cos(t)*SQR(cos(x))*cos(y)*sin(y) - cos(y)*sin(x)*(cos(t)*sin(x)*sin(y) - 1.);
          case dir::y:
            return cos(t)*cos(x)*SQR(cos(y))*sin(x) - cos(x)*sin(y)*(cos(t)*sin(x)*sin(y) - 1.);

          default:
            throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
          }
        default:
          throw std::runtime_error("dT_dd of analytical temperature field: unrecognized domain \n");
        } // end of switch domain
      } //end of coupled problem example
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
        switch(dom){
        case LIQUID_DOMAIN:
          switch(dir){
          case dir::x:
            return cos(x)*sin(y)*(x + cos(t)*cos(x)*cos(y)) - sin(x)*sin(y)*(cos(t)*cos(y)*sin(x) - 1.);
          case dir::y:
            return cos(y)*sin(x)*(x + cos(t)*cos(x)*cos(y)) - cos(t)*cos(x)*sin(x)*SQR(sin(y));

          default:
            throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
          }
        case SOLID_DOMAIN:
          switch(dir){
          case dir::x:
            return cos(t)*SQR(cos(x))*cos(y)*sin(y) - cos(y)*sin(x)*(cos(t)*sin(x)*sin(y) - 1.);
          case dir::y:
            return cos(t)*cos(x)*SQR(cos(y))*sin(x) - cos(x)*sin(y)*(cos(t)*sin(x)*sin(y) - 1.);

          default:
            throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
          }
        default:
          throw std::runtime_error("dT_dd of analytical temperature field: unrecognized domain \n");
        } // end of switch domain
      } //end of coupled problem example
      case COUPLED_TEST_2:{
        switch(dom){
          case LIQUID_DOMAIN:
            switch(dir){
            case dir::x:
              return (y + 1.)/factor - N*t*y*PI*sin(N*PI*t*x*y); //2.*x*sin(2.*SQR(t)*PI)*SQR(y) + (y + 1.)/factor;
            case dir::y:
              return (x - 1.)/factor - N*t*x*PI*sin(N*PI*t*x*y); //2.*y*sin(2.*SQR(t)*PI)*SQR(x) + (x - 1.)/factor;
            default:
              throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
            }
          case SOLID_DOMAIN:
            switch(dir){
            case dir::x:
              return -1.*(y - 1.)/factor - N*t*y*PI*sin(N*PI*t*x*y);
            case dir::y:
              return -1.*(x - 1.)/factor - N*t*x*PI*sin(N*PI*t*x*y);

            default:
              throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
            }
        default:
          throw std::runtime_error("dT_dd of analytical temperature field: unrecognized domain \n");
        } // end of switch domain
      } // end of coupled test 2
      default:{
        throw std::runtime_error("dT_dd of analytical temp field: unrecognized example \n");
      }
    }
  }

  double dT_dt(DIM(double x, double y, double z)){
    switch(example_){
      case COUPLED_PROBLEM_EXAMPLE:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);
        case SOLID_DOMAIN:
          return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);
        default:
          throw std::runtime_error("dT_dt in analytical temperature: unrecognized domain \n");
        }
      } // end coupled problem example
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);
        case SOLID_DOMAIN:
          return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);
        default:
          throw std::runtime_error("dT_dt in analytical temperature: unrecognized domain \n");
        }
      } // end coupled problem example
      case COUPLED_TEST_2:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -N*x*y*PI*sin(N*PI*t*x*y);//4.*t*SQR(x)*SQR(y)*PI*cos(2.*PI*SQR(t));
        case SOLID_DOMAIN:
          return -N*x*y*PI*sin(N*PI*t*x*y);//4.*t*SQR(x)*SQR(y)*PI*cos(2.*PI*SQR(t)); // x*y*PI*cos(PI*t*x*y);//
        default:
          throw std::runtime_error("dT_dt in analytical temperature: unrecognized domain \n");
        }
      } // end coupled test 2
      default:{
        throw std::runtime_error("dT_dt in analytical temperature: unrecognized example \n");
      }
    } // end switch example
  }

  double laplace(DIM(double x, double y, double z)){
    switch(example_){
      case COUPLED_PROBLEM_EXAMPLE:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -2.*sin(y)*(x*sin(x) - cos(x) + 4.*cos(t)*cos(x)*cos(y)*sin(x));
        case SOLID_DOMAIN:
          return -2.*cos(x)*cos(y)*(4.*cos(t)*sin(x)*sin(y) - 1.);
        default:
          throw std::runtime_error("laplace for analytical temperature field: unrecognized domain \n");
        }
      }
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -2.*sin(y)*(x*sin(x) - cos(x) + 4.*cos(t)*cos(x)*cos(y)*sin(x));
        case SOLID_DOMAIN:
          return -2.*cos(x)*cos(y)*(4.*cos(t)*sin(x)*sin(y) - 1.);
        default:
          throw std::runtime_error("laplace for analytical temperature field: unrecognized domain \n");
        }
      }
      case COUPLED_TEST_2:{
        switch(dom){
        case LIQUID_DOMAIN:
          return -1.*SQR(t*PI*N)*cos(N*PI*t*x*y)*(SQR(x) + SQR(y));
        case SOLID_DOMAIN:
          return -1.*SQR(t*PI*N)*cos(N*PI*t*x*y)*(SQR(x) + SQR(y));

        default:
          throw std::runtime_error("laplace for analytical temperature field: unrecognized domain \n");
        }
      }
    }
  }
};

struct interfacial_velocity : CF_DIM{ // will yield analytical solution to interfacial velocity in a given cartesian direction (not including the multiplication by the normal, which will have to be done outside of this struct)

public:
  const unsigned char dir;
  temperature_field** temperature_;
  interfacial_velocity(const unsigned char &dir_,temperature_field** analytical_soln):dir(dir_),temperature_(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const{
    return (temperature_[SOLID_DOMAIN]->dT_d(dir,x,y) - temperature_[LIQUID_DOMAIN]->dT_d(dir,x,y))*coupled_test_sign;
  }
};

struct velocity_component: CF_DIM
{
  const unsigned char dir;
  const double k_NS=1.0; //1.0;
  velocity_component(const unsigned char& dir_) : dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double v(DIM(double x, double y, double z)) const{ // gives vel components without the time component
    switch(example_){
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      switch(dir){
      case dir::x:
        return sin(x)*cos(y);
      case dir::y:
        return -1.0*cos(x)*sin(y);
      default:
        throw std::runtime_error("analytical solution velocity: unknown cartesian direction \n");
      }
    }
    case COUPLED_TEST_2:{
      switch(dir){
      case dir::x:
        return (pow(x,3.)*(y - 1))/16. - (x*sin(2.*PI*(t - y)))/2.;//-cos(PI*(t - x))*(- 3.*SQR(y) + 2.*y);
      case dir::y:
        return (3.*SQR(x)*(- SQR(y)/2. + y))/16. + cos(2.*PI*(t - y))/(4.*PI); //PI*sin(PI*(t - x))*(-1.*pow(y,3.) + SQR(y));
      default:
        throw std::runtime_error("analytical solution velocity: unknown cartesian direction \n");
      }
    }
    default:{
      throw std::runtime_error("analytical solution velocity: unknown example \n");
    }
    }
  }
  double dv_d(const unsigned char& dirr,DIM(double x, double y, double z)) const{
    switch(example_){
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      switch(dir){
      case dir::x:
        switch(dirr){
        case dir::x: //du_dx (without time component)
          return cos(x)*cos(y);
        case dir::y: // du_dy (without time component)
          return -sin(x)*sin(y);
        }
      case dir::y:
        switch(dirr){
        case dir::x: // dv_dx ("")
          return sin(x)*sin(y);
        case dir::y: // dv_dy ("")
          return -cos(x)*cos(y);
        }
      }
    }
    case COUPLED_TEST_2:{
      switch(dir){
      case dir::x:
        switch(dirr){
        case dir::x: //du_dx (with time component)
          return (3.*SQR(x)*(y - 1.))/16. - sin(2.*PI*(t - y))/2.; // -1.*PI*sin(PI*(t - x))*(-3.*SQR(y) + 2.*y);
        case dir::y: // du_dy (with time component)
          return pow(x,3.)/16. + PI*cos(2.*PI*(t - y))*x; // cos(PI*(t - x))*(6.*y - 2.);
        }
      case dir::y:
        switch(dirr){
        case dir::x: // dv_dx ("")
          return (3.*x*(-1.*SQR(y)/2. + y))/8.; //-SQR(PI)*cos(PI*(t - x))*(-1.*pow(y,3.) + SQR(y));
        case dir::y: // dv_dy ("")
          return sin(2.*PI*(t - y))/2. - (3.*SQR(x)*(y - 1.))/16.; //PI*sin(PI*(t - x))*(-3.*SQR(y) + 2.*y);
        }
      }
    }
    }
  }

  double operator()(DIM(double x, double y, double z)) const{ // Returns the velocity field
    switch (example_) {
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      return cos(t*k_NS)*v(DIM(x,y,z));
    }
    case COUPLED_TEST_2:{
      return v(DIM(x,y,z)); // time component included in vel expression for this example
    }
    default:{
      throw std::runtime_error("analytical velocity : unknown example \n");
    }
    }

  }
  double _d(const unsigned char& dirr, DIM(double x, double y, double z)){ // Returns spatial derivatives of velocity field in given cartesian direction
    switch (example_) {
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      return cos(t*k_NS)*dv_d(dirr,DIM(x,y,z));
    }
    case COUPLED_TEST_2:{
      return dv_d(dirr,DIM(x,y,z)); // put whole derivative in the expression for this particular example
    }
    default:{
      throw std::runtime_error("analytical velocity : unknown example \n");
    }
    }
  }
  double laplace(DIM(double x, double y, double z)){
    switch (example_) {
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      return -P4EST_DIM*cos(t*k_NS)*v(DIM(x,y,z));
    }
    case COUPLED_TEST_2:{
      switch(dir){
      case dir::x:{
        return (3.*x*(y - 1.))/8. + 2.*x*SQR(PI)*sin(2.*PI*(t - y));//cos(PI*(t - x))*(-3.*SQR(PI)*SQR(y) + 2.*SQR(PI)*y + 6.);
      }
      case dir::y:{
        return (3.*y)/8. - PI*cos(2.*PI*(t - y)) - (3.*SQR(x))/16. - (3.*SQR(y))/16.; //-1.*PI*sin(PI*(t - x))*(-1.*SQR(PI)*pow(y,3.) + SQR(PI)*SQR(y) + 6.*y - 2.);
      }
      default:{
        throw std::runtime_error("analytical velocity: laplace: unknown cartesian direction \n");
      }
      }
    }
    default:{
      throw std::runtime_error("analytical velocity : laplace: unknown example \n");
    }
    }
  }
  double dv_dt(DIM(double x, double y, double z)){
    switch (example_) {
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      return -sin(k_NS*t)*v(DIM(x,y,z));
    }
    case COUPLED_TEST_2:{
      switch(dir){
      case dir::x:{
        return -1.*x*PI*cos(2.*PI*(t - y)); // PI*sin(PI*(t - x))*(-3.*SQR(y) + 2.*y);
      }
      case dir::y:{
        return -1.*sin(2.*PI*(t - y))/2.; //SQR(PI)*cos(PI*(t - x))*(-1.*pow(y,3.) + SQR(y));
      }
      default:{
        throw std::runtime_error("analytical velocity: dv_dt: unknown cartesian direction \n");
      }
      }
    }
    default:{
      throw std::runtime_error("analytical velocity:dv_dt : unknown example \n");
    }
    }
  }
};

struct external_heat_source: CF_DIM{
  const unsigned char dom;
  temperature_field** temperature_;
  velocity_component** velocity_;

  external_heat_source(const unsigned char &dom_,temperature_field** analytical_T,velocity_component** analytical_v):dom(dom_),temperature_(analytical_T),velocity_(analytical_v){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }

  double operator()(DIM(double x, double y, double z)) const {
    double advective_term;
    switch(dom){
    case LIQUID_DOMAIN:
      advective_term= (*velocity_[dir::x])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN]->dT_d(dir::x,x,y) + (*velocity_[dir::y])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN]->dT_d(dir::y,x,y);
      break;
    case SOLID_DOMAIN:
      advective_term= 0.;
      break;
    default:
      throw std::runtime_error("external heat source : advective term : unrecognized domain \n");
    }
    return temperature_[dom]->dT_dt(DIM(x,y,z)) + advective_term - temperature_[dom]->laplace(DIM(x,y,z));
  }
};

struct pressure_field: CF_DIM{
  public:
  double operator()(DIM(double x,double y, double z)) const {
    switch(example_){
    case COUPLED_PROBLEM_EXAMPLE:{
      return 0.0;
    }
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
      return 0.0;
    }
    case COUPLED_TEST_2:{
      return 0.0;//return sin(2.*PI*x)*(3.*y + PI*cos(PI*t));
    }
    case NS_GIBOU_EXAMPLE:{
      return 0.0;
    }
    default:{
      throw std::invalid_argument("pressure_field: Unrecognized example \n");
    }
    }
  }

  double gradP(const unsigned char& dir,DIM(double x, double y, double z)){
    switch(example_){
    case COUPLED_PROBLEM_EXAMPLE:{
      return 0.0;
    }
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
      return 0.0;
    }
    case COUPLED_TEST_2:{
      switch(dir){
      case dir::x:{
        return 0.0; //return 2.*PI*cos(2.*PI*x)*(3.*y + PI*cos(PI*t));
      }
      case dir::y:{
        return 0.0;//return 3.*sin(2.*PI*x);
      }
      default:{
        throw std::invalid_argument("gradP: unrecognized direction \n");
      }
      }
    }
    case NS_GIBOU_EXAMPLE:{
      return 0.0;
    }
    default:{
      throw std::invalid_argument("pressure_field: Unrecognized example \n");
    }
    } // end of switch example
  }
}pressure_field_analytical;

struct external_force_per_unit_volume_component : CF_DIM{
  const unsigned char dir;
  velocity_component** velocity_field;
  external_force_per_unit_volume_component(const unsigned char& dir_, velocity_component** analytical_soln):dir(dir_),velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const{ // returns the forcing term in a given direction
    pressure_field_analytical.t = t;
    return velocity_field[dir]->dv_dt(DIM(x,y,z)) +
           SUMD((*velocity_field[0])(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z)),
                (*velocity_field[1])(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z)),
                (*velocity_field[2])(DIM(x,y,z))*velocity_field[dir]->_d(dir::z,DIM(x,y,z))) -
           velocity_field[dir]->laplace(DIM(x,y,z)) + pressure_field_analytical.gradP(dir,DIM(x,y,z));
  }
};

// with Boussinesq Approximation:(TO-DO:: combine this function with existing external_force_per_unit_volume())
struct external_force_per_unit_volume_component_with_boussinesq_approx : CF_DIM{
  const unsigned char dom;
  const unsigned char dir;
  temperature_field** temperature_;
  velocity_component** velocity_field;

  external_force_per_unit_volume_component_with_boussinesq_approx(const unsigned char &dom_,const unsigned char &dir_, temperature_field** analytical_T,velocity_component** analytical_v):dom(dom_),dir(dir_),temperature_(analytical_T),velocity_field(analytical_v){
    P4EST_ASSERT(dir<P4EST_DIM);
    P4EST_ASSERT(dom>=0 && dom <=1);
  }
  double operator()(DIM(double x, double y, double z)) const{ // returns the forcing term in a given direction
    pressure_field_analytical.t = t;
    if (dir==1){
        switch(dom){
            case LIQUID_DOMAIN:
                return velocity_field[dir]->dv_dt(DIM(x,y,z)) +
                    SUMD((*velocity_field[0])(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z)),
                    (*velocity_field[1])(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z)),
                    (*velocity_field[2])(DIM(x,y,z))*velocity_field[dir]->_d(dir::z,DIM(x,y,z))) -
                    velocity_field[dir]->laplace(DIM(x,y,z)) + pressure_field_analytical.gradP(dir,DIM(x,y,z)) - temperature_[LIQUID_DOMAIN]->T(DIM(x,y,z));
                break;
            case SOLID_DOMAIN:
                return velocity_field[dir]->dv_dt(DIM(x,y,z)) +
                    SUMD((*velocity_field[0])(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z)),
                    (*velocity_field[1])(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z)),
                    (*velocity_field[2])(DIM(x,y,z))*velocity_field[dir]->_d(dir::z,DIM(x,y,z))) -
                    velocity_field[dir]->laplace(DIM(x,y,z)) + pressure_field_analytical.gradP(dir,DIM(x,y,z));
                break;
            default:
                break;
        }
    }else{
        return velocity_field[dir]->dv_dt(DIM(x,y,z)) +
            SUMD((*velocity_field[0])(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z)),
            (*velocity_field[1])(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z)),
            (*velocity_field[2])(DIM(x,y,z))*velocity_field[dir]->_d(dir::z,DIM(x,y,z))) -
            velocity_field[dir]->laplace(DIM(x,y,z)) + pressure_field_analytical.gradP(dir,DIM(x,y,z));
    }
  }
};
// --------------------------------------------------------------------------------------------------------------
// Level set functions:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (example_){
      case FRANK_SPHERE:{
        return s0 - sqrt(SQR(x) + SQR(y));
      }
      case EVOLVING_POROUS_MEDIA:{
        return return_LSF_porous_media(DIM(x, y, z), false);
      }
      case DISSOLVING_DISK_BENCHMARK: {
        return r0 - sqrt(SQR(x-(xmax/2.)) + SQR(y - (ymax/2.)));
      }
      case MELTING_ICE_SPHERE_NAT_CONV:{
        return r0 - sqrt(SQR(x - (xmax/2.0)) + SQR(y - (ymax/2.0)));
      }
      case MELTING_ICE_SPHERE:{
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      }
      case ICE_AROUND_CYLINDER:{
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      }
      case NS_GIBOU_EXAMPLE:{
        return r0 - sin(x)*sin(y);
      }
      case COUPLED_PROBLEM_EXAMPLE:{
        return r0 - sqrt(SQR(x) + SQR(y));
      }
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:{
        return r0 - sqrt(SQR(x) + SQR(y));
      }
      case COUPLED_TEST_2:{
        double x0 = 0.;//1./6.;
        double y0 = -1./4.;
        double R = 1./3.;//1./4.;
        double rval = sqrt(SQR(x-x0) + SQR(y-y0));
        return -1.*(rval - (R)  - (pow(y - y0,5.) + 5.*(pow(x-x0,4.))*(y-y0) - 10.*SQR(x-x0)*pow(y-y0,3.))/(3.*pow(rval,5.) + 1e-5));
      }
      case DENDRITE_TEST:{
        // check this later to-do
        double noise = 0.001;
        double xc =xmax/2.0;
        double yc =ymax/2.0;
        double theta = atan2(y-yc,x-xc);
        //return r0*(1.0 - noise*fabs(sin(theta)) - noise*fabs(cos(theta))) - sqrt(SQR(x - xc) + SQR(y - yc));
        return r0*(1.0 - noise*fabs(pow(sin(2*theta),2)))- sqrt(SQR(x - xc) + SQR(y - yc));
      }
      case PLANE_POIS_FLOW:{
        if(y > r0){
          return -1.;
        }
        else{
          return 1.0;
        }
      }
      default: throw std::invalid_argument("You must choose an example type\n");
    } // end of switch case
  } // end of operator
} level_set; // end of function


// Inner level set function for relevant cases:
struct SUBSTRATE_LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER: {
        return r_cyl - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      }

      case EVOLVING_POROUS_MEDIA:{
        return return_LSF_porous_media(DIM(x, y, z), true);
      }
      case FRANK_SPHERE:
      case MELTING_ICE_SPHERE_NAT_CONV:
      case MELTING_ICE_SPHERE:
      case DISSOLVING_DISK_BENCHMARK:
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
      case DENDRITE_TEST:
      case NS_GIBOU_EXAMPLE:
        throw std::invalid_argument("This option may not be used for the particular example being called");
      }
  }
} substrate_level_set;


struct INITIAL_REFINEMENT_CF : CF_DIM {
  public:
      double operator() (DIM(double x, double y, double z)) const{
        if(example_uses_inner_LSF){
          bool main_phi_has_smaller_abs_val = fabs(level_set(x,y)) < fabs(substrate_level_set(x,y));
          if(main_phi_has_smaller_abs_val){
            return level_set(x,y);
          }
          else{
            return substrate_level_set(x,y);
          }
        }
        else{
          return level_set(x,y);
        }

      }
}initial_refinement_cf;


// Function for ramping the boundary conditions:
double ramp_BC(double initial,double goal_value){
  if(tn<t_ramp){
    return initial + ((goal_value - initial)/(t_ramp - tstart))*(tn - tstart);
    }
  else {
      return goal_value;
    }
}



// --------------------------------------------------------------------------------------------------------------
// Initial temperature condition objects/functions:
// --------------------------------------------------------------------------------------------------------------
class INITIAL_TEMP: public CF_DIM
{
  public:
  const unsigned char dom;
  temperature_field** temperature_;
  INITIAL_TEMP(const unsigned char &dom_,temperature_field** analytical_T=NULL):dom(dom_),temperature_(analytical_T){}
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
    case FRANK_SPHERE:{
      double r = sqrt(SQR(x) + SQR(y));
      double sval = s(r,t);
      return frank_sphere_solution_t(sval); // Initial distribution is the analytical solution of Frank Sphere problem at t = tstart
    }
    case DENDRITE_TEST:{
      if(level_set(DIM(x,y,z))<0){
        return theta_infty;
      }
      else{
        return theta_interface;
      }
    }
    case EVOLVING_POROUS_MEDIA:{
      //      return theta_interface;
      switch(dom){
      case LIQUID_DOMAIN:
        if(is_dissolution_case){
          return theta_initial;
        }
        else{
          return theta_infty;
        }
      case SOLID_DOMAIN:
        return theta_interface;
      default:
        throw std::invalid_argument("Initial temp: domain unrecognized (needs to be solid or liquid)");
      }
    }
    case MELTING_ICE_SPHERE_NAT_CONV:
    case MELTING_ICE_SPHERE:{
      switch(dom){
      case LIQUID_DOMAIN:{
        return theta_infty;
      }
      case SOLID_DOMAIN:{
        return theta0; // coolest temp
      }
      default:{
        throw std::runtime_error("Initial condition for temperature: unrecognized domain \n");
      }
      }
    }
    case ICE_AROUND_CYLINDER:{ // TO-DO: is this best initial condition? might be missing on serious initial interface growth...
      switch(dom){
      case LIQUID_DOMAIN:{
        return theta_infty;
      }
      case SOLID_DOMAIN:{
        if(ramp_bcs){
          return theta_infty;
        }
        else{
          return theta_interface;
        }
      }
      default:{
        throw std::runtime_error("Initial condition for temperature: unrecognized domain \n");
      }
      }
    }
    case DISSOLVING_DISK_BENCHMARK:{
      return theta_infty;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:{
      return temperature_[dom]->T(DIM(x,y,z));
    }
    }
  }
};

// --------------------------------------------------------------------------------------------------------------
// Initial fluid velocity condition objects/functions: for fluid velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
struct INITIAL_VELOCITY : CF_DIM
{
  const unsigned char dir;
  velocity_component** velocity_field;

  INITIAL_VELOCITY(const unsigned char& dir_,velocity_component** analytical_soln=NULL):dir(dir_), velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double operator() (DIM(double x, double y,double z)) const{
    switch(example_){
    case DENDRITE_TEST:
    case PLANE_POIS_FLOW:{
      switch(dir){
      case dir::x:
        return u0;
      case dir::y:
        return v0;
      default:
        throw std::runtime_error("initial velocity direction unrecognized \n");
      }
    }

    case EVOLVING_POROUS_MEDIA: {
      //        return 0.0;
      //        double h_ = ymax - ymin;
      //        double U = G_press/(2.* Re);
      //        double Uy = (Da_init/porosity_init) *U*(y)*(h_ - y);

      //        double lsf_dist = 0.1;

      //        if(fabs(level_set.operator()(DIM(x,y,z)))<lsf_dist){
      //          return (Uy/lsf_dist)*(-1.*level_set.operator()(DIM(x,y,z)));
      //        }
      //        else{
      //          switch(dir){
      //          case dir::x:{
      //            return Uy;}
      //          case dir::y:
      //            return 0.;
      //          default:
      //              throw std::runtime_error("Unrecognized direction for velocity initial condition \n");
      //          }
      //        }

      switch(dir){
      case dir::x:
        return u0;
      case dir::y:
        return v0;
      default:
        throw std::runtime_error("initial velocity direction unrecognized \n");
      }

    }
    case MELTING_ICE_SPHERE_NAT_CONV:{
      if(ramp_bcs) return 0.;
      else{
        switch(dir){
        case dir::x:
          if(perturb_initial_flow){
            return u0*(1 + perturb_flow_noise*sin(2.*PI*x/xmax));
          }
          else{
            return u0;
          }
        case dir::y:
          if(perturb_initial_flow){
            return v0*(1 + perturb_flow_noise*sin(2.*PI*x/xmax));
          }
          else{
            return v0;
          }
        default:
          throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
        }
      }
    }
    case ICE_AROUND_CYLINDER:
    case FLOW_PAST_CYLINDER:
    case MELTING_ICE_SPHERE:{
      if(ramp_bcs) return 0.;
      else{
        switch(dir){
        case dir::x:
          if(perturb_initial_flow){
            return u0*(1 + perturb_flow_noise*sin(2.*PI*y/ymax));
          }
          else{
            return u0;
          }
        case dir::y:
          if(perturb_initial_flow){
            return v0*(1 + perturb_flow_noise*sin(2.*PI*y/ymax));
          }
          else{
            return v0;
          }
        default:
          throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
        }
      }
    }
    case DISSOLVING_DISK_BENCHMARK:{
      switch(dir){
      case dir::x:
        return u0;
      case dir::y:
        return v0;
      default:
        throw std::runtime_error("initial velocity direction unrecognized \n");
      }
    }
    case NS_GIBOU_EXAMPLE:
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
      switch(dir){
      case dir::x:
        return (*velocity_field[0])(x,y);
      case dir::y:
        return (*velocity_field[1])(x,y);
      default:
        throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
      }
    default:
      throw std::runtime_error("vel initial: unrecognized example_ being run \n");
    }
  }
} ;


// --------------------------------------------------------------------------------------------------------------
// Auxiliary fxns for evaluating BCs
// --------------------------------------------------------------------------------------------------------------
// Wall functions -- these evaluate to true or false depending on
// if the location is on the wall --  they just add coding simplicity for wall boundary conditions
// --------------------------
bool xlower_wall(DIM(double x, double y, doublze z)){
  // Front x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmin) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS));
};
bool xupper_wall(DIM(double x, double y, double z)){
  // back x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmax) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS));
};
bool ylower_wall(DIM(double x, double y, double z)){
  return (fabs(y - ymin) <= EPS);
}
bool yupper_wall(DIM(double x, double y, double z)){
  return (fabs(y - ymax) <= EPS);
};

bool is_x_wall(DIM(double x, double y, double z)){
  return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
};
bool is_y_wall(DIM(double x, double y, double z)){
  return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
};
bool test_2_two_walls_temp(DIM(double x, double y, double z)){
  return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
};

double sign_neumann_wall(DIM(double x,double y,double z)){
  if (xlower_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
    return -1.0;
  }
  else{
    return 1.0;
  }
};

// --------------------------
// Auxiliary fxn for BC Type settings for temperature at walls :
// --------------------------
bool dirichlet_temperature_walls(DIM(double x, double y, double z)){
  switch(example_){
  case FRANK_SPHERE:{
    return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
  }
  case DENDRITE_TEST:{
    // Dirichlet on y upper wall (where bulk flow is incoming)
    return (yupper_wall(DIM(x,y,z)));
  }
  case EVOLVING_POROUS_MEDIA:
  {
    return xlower_wall(DIM(x,y,z));
  }
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:{
    return (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)));
  }
  case MELTING_ICE_SPHERE_NAT_CONV:{
    return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
  }
  case DISSOLVING_DISK_BENCHMARK:{
    return xlower_wall(DIM(x,y,z));
  }
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:{
    return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
  }
  case COUPLED_TEST_2:{
    return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || xlower_wall(DIM(x,y,z)));
  }
  }
};

// --------------------------
// Auxiliary fxn forBC type settings for velocity at walls :
// --------------------------
bool dirichlet_velocity_walls(DIM(double x, double y, double z)){
  switch(example_){
  case DENDRITE_TEST:{
    // Dirichlet on y upper wall (where bulk flow is incoming
    return ( yupper_wall(DIM(x,y,z)) || xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
  }
  case FLOW_PAST_CYLINDER:
  case EVOLVING_POROUS_MEDIA:{
    // no dirichlet wall velocity conditions
//    return 0.;/*(ylower_wall(DIM(x,y,z)) || (yupper_wall(DIM(x,y,z))))*/;
    return (ylower_wall(DIM(x,y,z)) || (yupper_wall(DIM(x,y,z))));
  }
  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:{
    return (xlower_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
  }
  case DISSOLVING_DISK_BENCHMARK:{
    return (xlower_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
  }
  case NS_GIBOU_EXAMPLE:{
    return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) ||ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
  }
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:{
    return (xlower_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)));
  }
  case COUPLED_TEST_2:{
    return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || xlower_wall(DIM(x,y,z)));

    //      return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)));
  }
  case PLANE_POIS_FLOW:{
    return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,y)));
  }
  case FRANK_SPHERE:{
    throw std::runtime_error("dirichlet velocity walls: invalid example: frank sphere");
  }
  }
};

// --------------------------------------------------------------------------------------------------------------
// Boundary conditions for scalar temp/conc problem:
// --------------------------------------------------------------------------------------------------------------
//----------------
// Interface:
//----------------
// Value:
class BC_INTERFACE_VALUE_TEMP: public my_p4est_stefan_with_fluids_t::interfacial_bc_temp_t{ // TO CHECK -- changed how interp is initialized
  private:

    temperature_field** temperature_;
    unsigned char dom;

  public:
      BC_INTERFACE_VALUE_TEMP(my_p4est_stefan_with_fluids_t* parent_solver, bool do_we_use_curvature_, bool do_we_use_normals_) : interfacial_bc_temp_t(parent_solver, do_we_use_curvature_, do_we_use_normals_){}

    void setup_ana_case(temperature_field** analytical_T, unsigned char dom_){
      temperature_= analytical_T;
      dom = dom_;
    }

    double dissolution_bc_expression() const {
      if(example_ == DISSOLVING_DISK_BENCHMARK){
//        // if deltaT is set to zero, we are using the C/Cinf nondim and set RHS=0. otherwise, we are using (C-Cinf)/(C0 - Cinf) = (C-Cinf)/(deltaC) nondim and set RHS to appropriate expression (see my derivation notes)
//        // -- use deltaT/Tinfty to make it of order 1 since concentrations can be quite small depending on units
//        if(fabs(deltaT/Tinfty) < EPS){
//          return 0.0;
//        }
//        else{
//          return -1.*(k_diss*l_char/Dl)*(Tinfty/deltaT);
//        }

        // This corresponds to the Robin BC: dC/dn + Da(C) = 0,
        // which arises from the surface reaction model R = k*C, where C_nondim = C/Cin, and
        // D dC/dn = - R = - k * C, which when nondimensionalized reduces to:
        // dC/dn = -(k * l_char/D) * C
        // dC/dn + (Da * C) = 0

        return 0.;
      }
      else{
        // This corresponds to the Robin BC condition (dC/dn + Da(C-1) = 0) --> (dC/dn + Da(C) = Da),
        // which arises from the surface reaction model R = k(C - Csat) , when C_nondim = C/Csat
        return Da;
      }
    }
    double operator()(DIM(double x, double y, double z)) const
    {
      switch(example_){
      case FRANK_SPHERE:{ // Frank sphere case, no surface tension
        return Tinterface; // TO-DO : CHANGE THIS TO ANALYTICAL SOLN
      }
      case DENDRITE_TEST:{
        double eps = 0.05;

        double cos_theta_ = (*nx_interp)(x,y);
        double cos_4_theta = 8.0*pow(cos_theta_,4.)-8.0*pow(cos_theta_,2.)+1;
        double int_val =1. + (-l_char)*(1. - 15.*eps*cos_4_theta)*((*kappa_interp)(x,y))/St;

        return int_val;
      }
      case EVOLVING_POROUS_MEDIA:{
        if(is_dissolution_case){
          return dissolution_bc_expression();
        }
        else{
          // temperature case, go with usual Gibbs-Thomson
          double interface_val = Gibbs_Thomson(DIM(x,y,z));

          // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
          if(ramp_bcs){
            return ramp_BC(theta_infty,interface_val);
          }
          else {
            return interface_val;
          }
        }
      }
      case MELTING_ICE_SPHERE_NAT_CONV:
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER: {
        double interface_val = Gibbs_Thomson(DIM(x,y,z));

        // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
        if(ramp_bcs){
          return ramp_BC(theta_infty,interface_val);
        }
        else {
          return interface_val;
        }
      }
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
      case COUPLED_PROBLEM_EXAMPLE:{
        return temperature_[dom]->T(DIM(x,y,z));
      }
      case DISSOLVING_DISK_BENCHMARK:{
        return dissolution_bc_expression();
      }
      default:
        throw std::runtime_error("BC INTERFACE VALUE TEMP: unrecognized example \n");
      }
    }
};

// Type:
BoundaryConditionType interface_bc_type_temp;
void interface_bc_temp(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
  case FRANK_SPHERE:
  case DENDRITE_TEST:
  case FLOW_PAST_CYLINDER:

  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:
  case COUPLED_TEST_2:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:
    interface_bc_type_temp = DIRICHLET;
    break;
  case DISSOLVING_DISK_BENCHMARK:
    interface_bc_type_temp = ROBIN;
    break;
  case EVOLVING_POROUS_MEDIA:
    if(is_dissolution_case){
      interface_bc_type_temp = ROBIN;
    }
    else{
      interface_bc_type_temp = DIRICHLET;
    }
    break;
  }
}

// Robin coeff:
class BC_interface_coeff: public CF_DIM{
  public:
  double operator()(double x, double y) const
  {

    switch(example_){
    case FRANK_SPHERE:
    case DENDRITE_TEST:

    case MELTING_ICE_SPHERE_NAT_CONV:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
      return 1.0;
    case EVOLVING_POROUS_MEDIA:{
      if(is_dissolution_case){
        return Da;
      }
      else{
        return 1.0;
      }
    }
    case DISSOLVING_DISK_BENCHMARK:
      // Elyce to-do 12/14/21: update this appropriately (may have to adjust depending on nondim type)
      //        double Da = k_diss*l_char/D;
      //return Da/Pe;//(k_diss*l_diss/D_diss);//(k_diss/u_inf); // Coefficient in front of C
      // ^^^ 12/17/21 why on earth did i have an effing peclet number there ???? aahhhhhhh
//      return 1.0;

      return Da;
    }
  }
}bc_interface_coeff;

//----------------
// Wall:
//----------------

// Function for handling the flushing scenarios:
// For this function, we assume that the flush times are given as dimensional time in seconds
int flush_idx = 0;
double theta_wall_flushing_scenario(){
  flush_idx = (int) floor(tn*time_nondim_to_dim / flush_every_dt);

  // We flush every given dt for a certain duration, but we ignore the first flush cycle to give some startup time (since by this formula, the first flush cycle would be at t=0)
  bool do_flush = ((tn*time_nondim_to_dim - flush_idx * flush_every_dt) < flush_duration) && (flush_idx > 0);

  if(do_flush){
    return theta_flush;
  }
  else {
    return theta_infty;
  }

}

// Value:
class BC_WALL_VALUE_TEMP: public CF_DIM
{
  public:
  const unsigned char dom;
  temperature_field** temperature_;
  BC_WALL_VALUE_TEMP(const unsigned char& dom_, temperature_field** analytical_soln=NULL): dom(dom_),temperature_(analytical_soln){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
    case FRANK_SPHERE:{
      if (dirichlet_temperature_walls(DIM(x,y,z))){
        double r= sqrt(SQR(x) + SQR(y));;
        double sval = r/sqrt(tn + dt);
        return frank_sphere_solution_t(sval);
      }
      break;
    }
    case DENDRITE_TEST:
    case EVOLVING_POROUS_MEDIA:
    {
      if(dirichlet_temperature_walls(DIM(x,y,z))){
        if(flush_every_dt>0. && flush_duration>0.){
          return theta_wall_flushing_scenario();
        }
        else {
          return theta_infty;
        }
      }
      else{
        return back_wall_temp_flux;
      }
    }
    case MELTING_ICE_SPHERE:
    case MELTING_ICE_SPHERE_NAT_CONV:
    case ICE_AROUND_CYLINDER:{
      if(dirichlet_temperature_walls(DIM(x,y,z))){
        return theta_infty;
      }
      else{
        return back_wall_temp_flux;
      }
    }
    case DISSOLVING_DISK_BENCHMARK:{
      if(dirichlet_temperature_walls(DIM(x,y,z))){
        return theta_infty;
      }
      else{
        return back_wall_temp_flux;
      }
      break;
    }
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
    case COUPLED_TEST_2:{
      if (dirichlet_temperature_walls(DIM(x,y,z))){ // dirichlet case
        return temperature_[dom]->T(DIM(x,y,z));
      }
      else{ // neumann case
        if(is_x_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*temperature_[dom]->dT_d(dir::x,DIM(x,y,z));
        }
        if(is_y_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*temperature_[dom]->dT_d(dir::y,DIM(x,y,z));
        }
        break;
      }
    }
    default:
      throw std::runtime_error("WALL BC TYPE TEMP: unrecognized example \n");
    }
  }
};

// Type:
class BC_WALL_TYPE_TEMP: public WallBCDIM
{
  public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    if(dirichlet_temperature_walls(DIM(x,y,z))){
      return DIRICHLET;
    }
    else{
      return NEUMANN;
    }
  }
}bc_wall_type_temp;

//----------------
// Substrate interface:
//----------------
// Value:
class BC_interface_value_inner: public CF_DIM{
  public:
  double operator()(double x, double y) const
  {
    switch(example_){
    case EVOLVING_POROUS_MEDIA:
      if(is_dissolution_case){
        return 0.; // if we are at the substrate, prescribe homogeneous neumann (no penetration)
      }
      // if not dissolution case, intentionally waterfall the case into the same settings as ice around cylinder
    case ICE_AROUND_CYLINDER:
      if(ramp_bcs){
        return ramp_BC(theta_infty,theta0);
      }
      else return theta0;
    }
  }
}bc_interface_val_inner;

// Type:
BoundaryConditionType inner_interface_bc_type_temp;
void inner_interface_bc_temp(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
  case EVOLVING_POROUS_MEDIA:
    if(is_dissolution_case){
      inner_interface_bc_type_temp = NEUMANN;
      // if we are at the substrate, prescribe homogeneous neumann (no penetration)
    }
    else{
      inner_interface_bc_type_temp = DIRICHLET;
    }
    break;
  case FLOW_PAST_CYLINDER:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:
  case COUPLED_TEST_2:
  case DENDRITE_TEST:
  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case FRANK_SPHERE:{
    throw std::invalid_argument("This option may not be used for the particular example being called");
  }
  case ICE_AROUND_CYLINDER:
    inner_interface_bc_type_temp = DIRICHLET;
    break;
  }
}

// Robin coeff:
class BC_interface_coeff_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    switch(example_){
      case EVOLVING_POROUS_MEDIA:
        if(is_dissolution_case){// if we are at the substrate, prescribe homogeneous neumann (no penetration)
          return 0.;
        }
        // if not dissolution, intentionally waterfall into the ice around cylinder case

      case ICE_AROUND_CYLINDER:
        return 1.0;
      }
  }
}bc_interface_coeff_inner;


// --------------------------------------------------------------------------------------------------------------
// Boundary conditions for Navier-Stokes problem:
// --------------------------------------------------------------------------------------------------------------
// Velocity interface:
//----------------
// Value:
// Interfacial condition:
class BC_interface_value_velocity: public my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t{

  public:
    unsigned char dir;
    velocity_component** velocity_field;
    BC_interface_value_velocity(my_p4est_stefan_with_fluids_t* parent_solver, bool do_we_use_vgamma_for_bc) : interfacial_bc_fluid_velocity_t(parent_solver, do_we_use_vgamma_for_bc){}

    void setup_ana_case(velocity_component** analyical_soln, unsigned char dir_){
      velocity_field = analyical_soln;
      dir = dir_;
    }
    double operator()(double x, double y) const
    {
      switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case PLANE_POIS_FLOW:
        return 0.; // homogeneous dirichlet no slip
      case FLOW_PAST_CYLINDER:
        return 0.;
      case EVOLVING_POROUS_MEDIA:{
//        if(is_dissolution_case) {
//          return 0.; // not strict no slip
//        }
//        else{
//          return Conservation_of_Mass(DIM(x,y,z));
//        }
        return Conservation_of_Mass(DIM(x,y,z));
      }

      case MELTING_ICE_SPHERE_NAT_CONV:
      case MELTING_ICE_SPHERE:
      case DISSOLVING_DISK_BENCHMARK:
//        return 0.;
        if(!solve_stefan) return 0.;
        else{
          return Conservation_of_Mass(DIM(x,y,z));
        }
      case ICE_AROUND_CYLINDER:{ // Ice solidifying around a cylinder
        if(!solve_stefan) return 0.;
        else{
          return Conservation_of_Mass(DIM(x,y,z)); // Condition derived from mass balance across interface
        }
      }

      case NS_GIBOU_EXAMPLE:
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
      case COUPLED_PROBLEM_EXAMPLE:
        return (*velocity_field[dir])(x,y);
      case DENDRITE_TEST:{
        return 0.0;
      }
      default:
        throw std::runtime_error("BC INTERFACE VALUE VELOCITY: unrecognized example ");
      }
    }


};

// Type:
BoundaryConditionType interface_bc_type_velocity[P4EST_DIM];
void BC_INTERFACE_TYPE_VELOCITY(const unsigned char& dir){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
  case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
  case PLANE_POIS_FLOW:
  case FLOW_PAST_CYLINDER:
  case DENDRITE_TEST:
  case DISSOLVING_DISK_BENCHMARK:
  case EVOLVING_POROUS_MEDIA:
  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:
    interface_bc_type_velocity[dir] = DIRICHLET;
    break;
  case NS_GIBOU_EXAMPLE:
    interface_bc_type_velocity[dir] = DIRICHLET;
    break;
  case COUPLED_TEST_2:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:{
    interface_bc_type_velocity[dir] = DIRICHLET;
    break;
  }
  }
}

//----------------
// Velocity wall:
//----------------
// Value:
class BC_WALL_VALUE_VELOCITY: public CF_DIM
{
  public:
  const unsigned char dir;
  velocity_component** velocity_field;

  BC_WALL_VALUE_VELOCITY(const unsigned char& dir_, velocity_component** analytical_soln=NULL):dir(dir_),velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
    //------------------------------------------------------------------
    case FRANK_SPHERE:{
      throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
    }
    case DENDRITE_TEST:{
      if(dirichlet_velocity_walls(DIM(x,y,z))){
        switch(dir){
        case dir::x:{
          return u0;
        }
        case dir::y:{
          return v0;
        }
        default:
          throw std::runtime_error("unrecognized cartesian direction for bc wall value velocity \n");
        }
      }
      else{
        return 0.0; // homogeneous neumann
      }
    }
    case FLOW_PAST_CYLINDER:
    case EVOLVING_POROUS_MEDIA:{
      if(dirichlet_velocity_walls(DIM(x,y,z))){
        return 0.0; // no slip at walls
      }
      else{
        return 0.0; // homogeneous neumann
      }
    }
    case MELTING_ICE_SPHERE_NAT_CONV:{
      if (dirichlet_velocity_walls(DIM(x,y,z))){
        if(ylower_wall(DIM(x,y,z))){
          return 0.0;
        }
        if(yupper_wall(DIM(x,y,z))){
          return 0.0;
        }
      }
    }
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:{
      if (dirichlet_velocity_walls(DIM(x,y,z))){ // dirichlet case
        if(ramp_bcs && tn<=t_ramp){

          switch(dir){
          case dir::x:
            return ramp_BC(0.,u0);
          case dir::y:
            return ramp_BC(0.,v0);
          default:
            throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
          }
        } // end of ramp BC case
        else{

          switch(dir){
          case dir::x:
            return u0;
          case dir::y:
            return v0;
          default:
            throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
          }
        }
      }
      else { // Neumann case
        switch(dir){
        case dir::x:
          return outflow_u;
        case dir::y:
          return outflow_v;
        default:
          throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
        }
      }
    }
    case DISSOLVING_DISK_BENCHMARK:{
      if(dirichlet_velocity_walls(DIM(x,y,z))){
        if(xlower_wall(DIM(x,y,z))){
          switch(dir){
          case dir::x:
            return u0;
          case dir::y:
            return v0;
          default:
            throw std::runtime_error("Velocity boundary condition at wall - unrecognized direction \n");
          }
        }
        else{
          return 0.0; // no slip on y walls
        }
      }
      else{
        return 0.0;// homogeneous neumann
      }
      break;
    }
    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
    case COUPLED_TEST_2:{
      if(dirichlet_velocity_walls(DIM(x,y,z))){ // dirichlet case
        return (*velocity_field[dir])(DIM(x,y,z));
      }
      else{ // neumann case
        if(is_x_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z));
        }
        if(is_y_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z));
        }
      }
      break;
    }
    case PLANE_POIS_FLOW:{
      return 0.; // homogeneous dirichlet on y walls (no slip), homogeneous neumann on x walls (prescribed pressure)
    }
    default:
      throw std::runtime_error("WALL BC VALUE VELOCITY: unrecognized example \n");
    }
  }
};


// Type:
class BC_WALL_TYPE_VELOCITY: public WallBCDIM
{
public:
  const unsigned char dir;
  BC_WALL_TYPE_VELOCITY(const unsigned char& dir_):dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    if(dirichlet_velocity_walls(DIM(x,y,z))){
       return DIRICHLET;
    }
    else{
     return NEUMANN;
    }
  }
};

//----------------
// Pressure interface:
//----------------
// Value:
class BC_INTERFACE_VALUE_PRESSURE: public CF_DIM{
  public:
  double operator()(DIM(double x, double y,double z)) const
  {
    switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case DENDRITE_TEST:
    case FLOW_PAST_CYLINDER:
    case EVOLVING_POROUS_MEDIA:
    case DISSOLVING_DISK_BENCHMARK:
    case MELTING_ICE_SPHERE_NAT_CONV:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder
      return 0.0;

    case PLANE_POIS_FLOW:
    case NS_GIBOU_EXAMPLE: // Benchmark NS
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
      return 0.0;
    default:
      throw std::runtime_error("INTERFACE BC VAL PRESSURE: unrecognized example \n");
    }
  }
};

// Type:
static BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
  case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
  case NS_GIBOU_EXAMPLE:
  case COUPLED_TEST_2:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:
  case DENDRITE_TEST:
  case PLANE_POIS_FLOW:
  case FLOW_PAST_CYLINDER:
  case EVOLVING_POROUS_MEDIA:
  case DISSOLVING_DISK_BENCHMARK:
  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:
    interface_bc_type_pressure = NEUMANN;
    break;
  }
}

//----------------
// Pressure wall:
//----------------
// Value:
class BC_WALL_VALUE_PRESSURE: public CF_DIM
{
  public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
    case FRANK_SPHERE:
      throw std::invalid_argument("Navier Stokes solution is not "
                                  "compatible with this example, please choose another \n");
    case DENDRITE_TEST:{
      return 0.0; // homogeneous dirichlet or neumann
    }
    case FLOW_PAST_CYLINDER:
    case EVOLVING_POROUS_MEDIA:{
      // Dirichlet prescribed pressure on xlower wall, all other walls are either dirichlet zero or neumann zero
      if(xlower_wall(DIM(x,y,z))){
        double pressure_drop_nondim;
        switch(problem_dimensionalization_type){
        case NONDIM_BY_FLUID_VELOCITY:{
          pressure_drop_nondim = pressure_drop/(rho_l * u_inf * u_inf);
          break;
        }
        case NONDIM_BY_SCALAR_DIFFUSIVITY:{

          pressure_drop_nondim = is_dissolution_case?
                                                     (pressure_drop*l_char*l_char/rho_l/Dl/Dl):
                                                     (pressure_drop*l_char*l_char/rho_l/alpha_l/alpha_l);
//          printf("Prescribed pressure drop is %0.3e using Dl = %0.3e, l_char = %0.3e, rho_l = %0.3e, pressure_drop_presc = %0.3e \n", pressure_drop_nondim, Dl, l_char, rho_l, pressure_drop);

          break;
        }
        case DIMENSIONAL:{
          pressure_drop_nondim = pressure_drop;
          break;
        }
        default:{
          throw std::runtime_error("Unrecognized dimensionalization type \n");
        }
        }
        return pressure_drop_nondim;
      }
      else{
        return 0.0;
      }
    }
    case DISSOLVING_DISK_BENCHMARK:
      return 0.0; // returns homogeneous condition either way
    case MELTING_ICE_SPHERE_NAT_CONV:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:{ // coupled problem
      return 0.0;
    }

    case NS_GIBOU_EXAMPLE:
    case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
    case COUPLED_PROBLEM_EXAMPLE:
    case COUPLED_TEST_2:{
      pressure_field_analytical.t= this->t;
      if(!dirichlet_velocity_walls(DIM(x,y,z))){
        return pressure_field_analytical(DIM(x,y,z));
      }
      else {
        if(is_x_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*pressure_field_analytical.gradP(dir::x,DIM(x,y,z));
        }
        if(is_y_wall(DIM(x,y,z))){
          return sign_neumann_wall(DIM(x,y,z))*pressure_field_analytical.gradP(dir::y,DIM(x,y,z));
        }
      }
    }
    case PLANE_POIS_FLOW:{
      if(!dirichlet_velocity_walls(DIM(x,y,z))){
        if(xlower_wall(DIM(x,y,z))){
          return pressure_drop;
        }
        else{
          return 0.0;
        }
      }
      else{
        return 0.; // homogeneous neumann
      }
    }
    default:
      throw std::runtime_error("WALL BC VALUE PRESSURE: unrecognized example \n");
    }
  }
};

// Type:
class BC_WALL_TYPE_PRESSURE: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    if(example_ == EVOLVING_POROUS_MEDIA){ // in this case we have overlap of neumann pressure and velocity ... will clean up this implementation later, isn't as clean as I'd like
      if(xlower_wall(DIM(x,y,z)) || (xupper_wall(DIM(x,y,z)))){
        return DIRICHLET;
      }
      else{
        return NEUMANN;
      }
    }
    else{
      if(!dirichlet_velocity_walls(DIM(x,y,z))){ // pressure is dirichlet where vel is neumann
        return DIRICHLET;
      }
      else{
        return NEUMANN;
      }
    }
  }
};

// --------------------------------------------------------------------------------------------------------------
// Auxiliary functions for solving the problem!:
// --------------------------------------------------------------------------------------------------------------

// (V) want to handle this in main
void handle_any_startup_t_dt_and_bc_cases(mpi_environment_t& mpi, my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver,
                                          double cfl_NS_steady, double hodge_percentage_steady){

  // -----------------------------
  // Handle any startup time/startup iterations/ flush time/ etc. // TO-DO: will move this stuff to its own fxn
  // ------------------------------
  // TO-DO: remove below, I believe it is obselete
  //  // Enforce startup iterations for verification tests if needed:
//  if((startup_iterations>0)){
//    if(tstep<startup_iterations){
//      force_interfacial_velocity_to_zero=true;
//      tn=tstart;
//    }
//    else if(tstep==startup_iterations){
//      force_interfacial_velocity_to_zero=false;
//      tn = tstart;
//    }
//  }

  if(solve_navier_stokes){
    // Adjust the cfl_NS depending on the timestep:
    if(tstep<=5 ){
      cfl_NS=0.5;

      // loosen hodge criteria for initialization for porous media case:
      if(example_ == EVOLVING_POROUS_MEDIA){
        hodge_percentage_of_max_u = hodge_percentage_steady/**10*/;
        stefan_w_fluids_solver->set_hodge_max_iteration(500);

      }

    }
    else if((loading_from_previous_state && (tstep <= stefan_w_fluids_solver->get_load_tstep() + 5))){
      cfl_NS = 2.0;
      hodge_percentage_of_max_u = hodge_percentage_steady;
    }
    else{
      cfl_NS = cfl_NS_steady;

      if(example_ == EVOLVING_POROUS_MEDIA){
        hodge_percentage_of_max_u = hodge_percentage_steady; // to-do : clean up, num startup iterations should be a user intput, instead of just being set to 10
        stefan_w_fluids_solver->set_hodge_max_iteration(50);

      }
    }
  }

  // Check if some startup time (before allowing interfacial growth) has been requested
  if((startup_dim_time>0.) || (startup_nondim_time>0.)){
    if((startup_dim_time>0.) && (startup_nondim_time>0.)){
      throw std::invalid_argument("Must choose startup dim time, OR startup nondim time, but not both \n");
    }

    if(startup_dim_time>0.){ // Dimensional case
      if(tn*time_nondim_to_dim < startup_dim_time){
        force_interfacial_velocity_to_zero=true;
      }
      else{
        force_interfacial_velocity_to_zero = false;
      }
    } // end of dimensional case
    else{ // nondimensional case
      if(tn<startup_nondim_time){
        force_interfacial_velocity_to_zero=true;
      }
      else{
        force_interfacial_velocity_to_zero=false;
      }
    } // end of nondimensional case
  } // end of considering startup times

  // Check if some flush time (at which to change the wall temp or conc BC) has been requested:
  bool flush_time_initiated = false;
  if((flush_dim_time>0.) && (tn*time_nondim_to_dim >= flush_dim_time) && (!flush_time_initiated)){
    if(!is_dissolution_case){
      PetscPrintf(mpi.comm(), "Flush time is reached, activating new temperature BC value (s) \n");
      theta_infty = (Tflush - T0)/deltaT;
    }
    else{
      PetscPrintf(mpi.comm(), "Flush time is reached, activating new concentration BC value (s) \n");
      theta_infty = (Tflush - T0)/deltaT;
    }
    if(example_uses_inner_LSF){
      theta0 = theta_interface;
    }
    flush_time_initiated = true;

    stefan_w_fluids_solver->set_nondim_temp_conc_variables(theta_infty, theta_interface, theta0, deltaT);
    // TO-DO: check if flush case still works, havent sorted this out super rigorously
  }

  // Set the relevant settings in the solver:
  stefan_w_fluids_solver->set_cfl_NS(cfl_NS);
  stefan_w_fluids_solver->set_hodge_percentage_of_max_u(hodge_percentage_of_max_u);
  stefan_w_fluids_solver->set_force_interfacial_velocity_to_zero(force_interfacial_velocity_to_zero);

  if(do_phi_advection_substeps){
    // Only allow if we have passed the startup time
    stefan_w_fluids_solver->set_do_phi_advection_substeps((tn*time_nondim_to_dim) > phi_advection_substep_startup_time);
  }


} // handle_any_startup_t_dt_and_bc_cases()


// want to handle this in main (V)
void setup_analytical_ics_and_bcs_for_this_tstep(BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2],
                                             BC_WALL_VALUE_TEMP* bc_wall_value_temp[2],
                                             temperature_field* analytical_T[2],
                                             external_heat_source* external_heat_source_T[2],
                                             velocity_component* analytical_soln_v[P4EST_DIM],
                                             BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM],
                                             BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                                 BC_WALL_VALUE_PRESSURE& bc_wall_value_pressure,
                                             external_force_per_unit_volume_component* external_force_components[P4EST_DIM],
                                             external_force_per_unit_volume_component_with_boussinesq_approx* external_force_components_with_BA[P4EST_DIM],
                                                 CF_DIM* external_forces_NS[P4EST_DIM]){
  // Necessary for coupled case:

  // Note, this object does not actually get used directly, but
  // forcing terms for some examples depend upon it existing and being updated

  // IN PARTICULAR-- forcing term for liquid temperature problem also depends on this,
  // which is why it currently gets updated before the poisson problem
  for(unsigned char d=0;d<P4EST_DIM;d++){
    analytical_soln_v[d]->t = tn+dt;
  }
  // -------------------------------
  // Update BC objects for stefan problem:
  // -------------------------------
  if(solve_stefan){
    for(unsigned char d=0;d<2;d++){
      analytical_T[d]->t = tn+dt;
      bc_interface_val_temp[d]->t = tn+dt;
      bc_wall_value_temp[d]->t = tn+dt;
      external_heat_source_T[d]->t = tn+dt;
    }
  }

  // If not, we use the curvature and neighbors, but we have to wait till curvature is computed in Poisson step to apply this, so it is applied later

  // -------------------------------
  // Update analytical objects for the NS problem:
  // -------------------------------
  if(solve_navier_stokes){
    foreach_dimension(d){
      bc_interface_value_velocity[d]->t = tn+dt;
      bc_wall_value_velocity[d]->t = tn+dt;

      if (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP){
        external_force_components_with_BA[d]->t = tn+dt;
        external_forces_NS[d] = external_force_components_with_BA[d];
      }
      else{
        external_force_components[d]->t = tn+dt;
        external_forces_NS[d] = external_force_components[d];
      }

    }
    bc_wall_value_pressure.t=tn+dt;
  }

} // end of "setup_analytical_ics_and_bcs_for_this_tstep()"



// Want to handle this in main (V)
bool are_we_saving_vtk(int tstep_, double tn_, bool is_load_step, int& out_idx, bool get_new_outidx){
  bool out = false;
  if(save_to_vtk){
    if(save_using_dt){
        out= ((int (floor(tn_/save_every_dt) )) !=out_idx) && (!is_load_step);
        if(get_new_outidx){
          out_idx = int (floor(tn_/save_every_dt) );
        }
      }
    else if (save_using_iter){
        out = (( (int) floor(tstep_/save_every_iter) ) !=out_idx) && (!is_load_step);
        if(get_new_outidx) {
          out_idx = ((int) floor(tstep_/save_every_iter) );
        }
      }
  }
  return out;
}

bool are_we_saving_simulation_state(int tstep_, double tn_, bool is_load_step, int& out_state_idx){
  bool out = false;

  if(save_state_using_dt){
    out = (((int) (floor(tn_/save_state_every_dt) )) !=out_state_idx) && (!is_load_step);
    out_state_idx = int (floor(tn_/save_state_every_dt) );
  }
  else if(save_state_using_iter){
    out = (( (int) floor(tstep_/save_state_every_iter) ) != out_state_idx) && (!is_load_step);
    out_state_idx = ((int) floor(tstep_/save_state_every_iter) );
  }

  return out;
};

// Want to handle this in main (V)
bool are_we_saving_data(double& tn_,bool is_load_step, int& out_idx, bool get_new_outidx){
  bool out = false;
  if(save_fluid_forces || save_area_data){
      out = ((int (floor(tn_/save_data_every_dt) )) !=out_idx) && (!is_load_step);
      if(get_new_outidx){
        out_idx = int (floor(tn_/save_data_every_dt) );
      }
  }
  return out;
}

// --------------------------------------------------------------------------------------------------------------
// Functions for checking the error in test cases (and then saving to vtk): // NOTE: eventually, we will move these to be owned by the examples class
// --------------------------------------------------------------------------------------------------------------
void save_stefan_test_case(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_ghost_t *ghost, vec_and_ptr_t& T_l, vec_and_ptr_t& T_s, vec_and_ptr_t& phi, vec_and_ptr_dim_t& v_interface,  double dxyz_close_to_interface, bool are_we_saving_vtk, char* filename_vtk,char *name, FILE *fich){
  PetscErrorCode ierr;

  vec_and_ptr_t T_ana,phi_ana, v_interface_ana;
  T_ana.create(p4est,nodes); phi_ana.create(p4est,nodes);v_interface_ana.create(p4est,nodes);

  vec_and_ptr_t T_l_err,T_s_err,phi_err,v_interface_err;
  T_l_err.create(p4est,nodes); T_s_err.create(p4est,nodes); phi_err.create(p4est,nodes); v_interface_err.create(p4est,nodes);

  T_l.get_array(); T_s.get_array();
  phi.get_array(); v_interface.get_array();

  T_ana.get_array(); v_interface_ana.get_array();phi_ana.get_array();
  T_l_err.get_array();T_s_err.get_array();v_interface_err.get_array();phi_err.get_array();

  double Linf_Tl = 0.0;
  double Linf_Ts = 0.0;
  double Linf_phi = 0.0;
  double Linf_v_int = 0.0;

  double r;
  double sval;
  double vel;


  // Now loop through nodes to compare errors between LSF and Temperature profiles:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes){
    node_xyz_fr_n(n,p4est,nodes,xyz);

    r = sqrt(SQR(xyz[0]) + SQR(xyz[1]));
    sval = r/sqrt(tn+dt);

    phi_ana.ptr[n] = s0*sqrt(tn+dt) - r;

    T_ana.ptr[n] = frank_sphere_solution_t(sval);

    v_interface_ana.ptr[n] = s0/(2.0*sqrt(tn+dt));

    // Error on phi and v_int:
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){

      // Errors on phi:
      phi_err.ptr[n] = fabs(phi.ptr[n] - phi_ana.ptr[n]);

      Linf_phi = max(Linf_phi, phi_err.ptr[n]);

      // Errors on v_int:
      vel = sqrt(SQR(v_interface.ptr[0][n])+ SQR(v_interface.ptr[1][n]));
      v_interface_err.ptr[n] = fabs(vel - v_interface_ana.ptr[n]);
      Linf_v_int = max(Linf_v_int,v_interface_err.ptr[n]);
      }

    // Check error in the negative subdomain (T_liquid) (Domain = Omega_minus)
    if(phi.ptr[n]<0.){
        T_l_err.ptr[n]  = fabs(T_l.ptr[n] - T_ana.ptr[n]);
        Linf_Tl = max(Linf_Tl,T_l_err.ptr[n]);
      }
    if (phi.ptr[n]>0.){
        T_s_err.ptr[n]  = fabs(T_s.ptr[n] - T_ana.ptr[n]);
        Linf_Ts = max(Linf_Ts,T_s_err.ptr[n]);
      }
  }

  double global_Linf_errors[] = {0.0, 0.0, 0.0,0.0,0.0};
  double local_Linf_errors[] = {Linf_phi,Linf_Tl,Linf_Ts,Linf_v_int};


  // Now get the global maximum errors:
  int mpiret = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm); SC_CHECK_MPI(mpiret);


  // Print Errors to application output:
  PetscPrintf(p4est->mpicomm,"\n----------------\n Errors on frank sphere: \n --------------- \n");
  PetscPrintf(p4est->mpicomm,"dxyz close to interface: %0.2e \n", dxyz_close_to_interface);

  int num_nodes = nodes->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM, p4est->mpicomm);

  PetscPrintf(p4est->mpicomm,"Number of grid points used: %d \n \n", num_nodes);


  PetscPrintf(p4est->mpicomm," Linf on phi: %0.4e \n Linf on T_l: %0.4e \n Linf on T_s: %0.4e \n Linf on v_int: %0.4e \n", global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3]);

  // Print errors to file:
  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  PetscFPrintf(p4est->mpicomm,fich,"%e %e %d %e %e %e %e %d %e \n", tn+dt, dt, tstep,
                                                                    global_Linf_errors[0], global_Linf_errors[1],
                                                                    global_Linf_errors[2], global_Linf_errors[3],
                                                                    num_nodes, dxyz_close_to_interface);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);

  // If we are saving this timestep, output the results to vtk:
  if(are_we_saving_vtk){
//      std::vector<std::string> point_names;
//      std::vector<double*> point_data;

//      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","phi_ana","T_ana","v_interface_vec_ana","phi_err","T_l_err","T_s_err","v_interface_vec_err"};
//      point_data = {phi.ptr,T_l.ptr,T_s.ptr,v_interface.ptr[0],v_interface.ptr[1],phi_ana.ptr,T_ana.ptr,v_interface_ana.ptr,phi_err.ptr,T_l_err.ptr,T_s_err.ptr,v_interface_err.ptr};

      std::vector<Vec_for_vtk_export_t> point_fields;
      point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
      point_fields.push_back(Vec_for_vtk_export_t(T_l.vec, "T_l"));
      point_fields.push_back(Vec_for_vtk_export_t(T_s.vec, "T_s"));
      point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[0], "v_interface_x"));
      point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[1], "v_interface_y"));
      point_fields.push_back(Vec_for_vtk_export_t(phi_ana.vec, "phi_ana"));
      point_fields.push_back(Vec_for_vtk_export_t(T_ana.vec, "T_ana"));
      point_fields.push_back(Vec_for_vtk_export_t(v_interface_ana.vec, "v_interface_vec_ana"));
      point_fields.push_back(Vec_for_vtk_export_t(phi_err.vec, "phi_err"));
      point_fields.push_back(Vec_for_vtk_export_t(T_l_err.vec, "T_l_err"));
      point_fields.push_back(Vec_for_vtk_export_t(T_s_err.vec, "T_s_err"));
      point_fields.push_back(Vec_for_vtk_export_t(v_interface_err.vec, "v_interface_vec_err"));


//      std::vector<std::string> cell_names = {};
//      std::vector<double*> cell_data = {};

      std::vector<Vec_for_vtk_export_t> cell_fields = {};
      my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_fields, cell_fields);

      point_fields.clear();
      cell_fields.clear();

    }

  T_l.restore_array();
  T_s.restore_array();
  phi.restore_array();
  v_interface.restore_array();

  T_ana.restore_array();v_interface_ana.restore_array();phi_ana.restore_array();
  T_l_err.restore_array();T_s_err.restore_array();v_interface_err.restore_array();phi_err.restore_array();
  T_ana.destroy();v_interface_ana.destroy();phi_ana.destroy();
  T_l_err.destroy();T_s_err.destroy();v_interface_err.destroy();phi_err.destroy();
} // end of "save_stefan_test_case()"

void save_navier_stokes_test_case(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_ghost_t *ghost, vec_and_ptr_t& phi, vec_and_ptr_dim_t& v_NS, vec_and_ptr_t& press, vec_and_ptr_t& vorticity, double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk, char* filename_err_output, FILE* fich){

  // Save NS analytical to compare:
  vec_and_ptr_dim_t vn_analytical;
  vec_and_ptr_t pn_analytical;

  vn_analytical.create(p4est,nodes);
  pn_analytical.create(p4est,nodes);

  velocity_component* analytical_soln_comp[P4EST_DIM];
  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_comp[d] = new velocity_component(d);
    analytical_soln_comp[d]->t = tn+dt;
  }
  CF_DIM *analytical_soln[P4EST_DIM] = {DIM(analytical_soln_comp[0],analytical_soln_comp[1],analytical_soln_comp[2])};

  foreach_dimension(d){
    sample_cf_on_nodes(p4est,nodes,*analytical_soln[d],vn_analytical.vec[d]);
  }
  sample_cf_on_nodes(p4est,nodes,zero_cf,pn_analytical.vec);

  // Get errors:
  vec_and_ptr_dim_t vn_error;
  vec_and_ptr_t press_error;

  vn_error.create(p4est,nodes);
  press_error.create(p4est,nodes);

  double L_inf_u = 0., L_inf_v = 0.,L_inf_P = 0.;

  vn_analytical.get_array(); vn_error.get_array(); v_NS.get_array();
  pn_analytical.get_array(); press_error.get_array(); press.get_array();

  phi.get_array();
  foreach_node(n,nodes){
    if(phi.ptr[n]<0.){
      press_error.ptr[n] = fabs(press.ptr[n] - pn_analytical.ptr[n]);
      vn_error.ptr[0][n] = fabs(v_NS.ptr[0][n] - vn_analytical.ptr[0][n]);
      vn_error.ptr[1][n] = fabs(v_NS.ptr[1][n] - vn_analytical.ptr[1][n]);

      L_inf_u = max(L_inf_u,vn_error.ptr[0][n]);
      L_inf_v = max(L_inf_v,vn_error.ptr[1][n]);
      L_inf_P = max(L_inf_P,press_error.ptr[n]);
    }

  }

  // Get the global errors:
  double local_Linf_errors[3] = {L_inf_u,L_inf_v,L_inf_P};
  double global_Linf_errors[3] = {0.0,0.0,0.0};

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,3,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n"
                             "Errors on NS Validation "
                             "\n -------------------------------------\n"
                             "Linf on u: %0.3e \n"
                             "Linf on v: %0.3e \n"
                             "Linf on P: %0.3e \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.3e \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              num_nodes, dxyz_close_to_interface);



  // Print errors to file:
  PetscErrorCode ierr;
  ierr = PetscFOpen(p4est->mpicomm,filename_err_output,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],hodge_global_error,num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  if(are_we_saving_vtk){
//    vorticity.get_array();

    // Save data:
//    std::vector<std::string> point_names;
//    std::vector<double*> point_data;

//    point_names = {"phi","u","v","vorticity","pressure","u_ana","v_ana","P_ana","u_err","v_err","P_err"};
//    point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],
//                  pn_analytical.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr};


//    std::vector<std::string> cell_names = {};
//    std::vector<double*> cell_data = {};
    std::vector<Vec_for_vtk_export_t> point_fields = {};
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(v_NS.vec[0], "u"));
    point_fields.push_back(Vec_for_vtk_export_t(v_NS.vec[1], "v"));
    point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vorticity"));
    point_fields.push_back(Vec_for_vtk_export_t(press.vec, "pressure"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_analytical.vec[0], "u_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_analytical.vec[1], "v_ana"));

    point_fields.push_back(Vec_for_vtk_export_t(pn_analytical.vec, "P_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_error.vec[0], "u_err"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_error.vec[1], "v_err"));
    point_fields.push_back(Vec_for_vtk_export_t(press_error.vec, "P_err"));



    // ELYCE TO-DO -- update these with the vector export wrapper
//    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);
    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_fields, cell_fields);
    point_fields.clear();
    cell_fields.clear();

//    point_names.clear(); point_data.clear();
//    cell_names.clear(); cell_data.clear();

//    vorticity.restore_array();
  }


  // Delete the analytical solution now that it is done being used:
  for(unsigned char d=0;d<P4EST_DIM;++d){
    delete analytical_soln_comp[d];
  }
  v_NS.restore_array();press.restore_array();
  vn_analytical.restore_array(); pn_analytical.restore_array();
  vn_error.restore_array(); press_error.restore_array(); phi.restore_array();

  vn_analytical.destroy();
  pn_analytical.destroy();

  vn_error.destroy();
  press_error.destroy();
} // end of "save_navier_stokes_test_case()"

void save_coupled_test_case(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_ghost_t *ghost, my_p4est_node_neighbors_t* ngbd,
                            vec_and_ptr_t& phi, vec_and_ptr_t& Tl, vec_and_ptr_t& Ts, vec_and_ptr_dim_t& v_interface,
                            vec_and_ptr_dim_t& v_NS, vec_and_ptr_t& press, vec_and_ptr_t& vorticity,
                            double dxyz_close_to_interface, bool are_we_saving_vtk,
                            char* filename_vtk, char* filename_err_output, FILE* fich){

  // Save analytical fields to compare:
  vec_and_ptr_dim_t vn_analytical;
  vec_and_ptr_t pn_analytical;
  vec_and_ptr_t Tl_analytical;
  vec_and_ptr_t Ts_analytical;
  vec_and_ptr_dim_t v_interface_analytical;
  vec_and_ptr_t phi_analytical; // Only for last timestep, we see how well phi is recovered after deforming

  vn_analytical.create(p4est,nodes);
  pn_analytical.create(p4est,nodes);
  Tl_analytical.create(p4est,nodes);
  Ts_analytical.create(p4est,nodes);
  v_interface_analytical.create(p4est,nodes);

  velocity_component* analytical_soln_velNS[P4EST_DIM];
  temperature_field* analytical_soln_temp[2];
  interfacial_velocity* analytical_soln_velINT[P4EST_DIM];

  for(unsigned char d=0;d<2;++d){
    analytical_soln_temp[d] = new temperature_field(d);
    analytical_soln_temp[d]->t = tn+dt;
  }

  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_velNS[d] = new velocity_component(d);
    analytical_soln_velNS[d]->t = tn+dt;

    analytical_soln_velINT[d] = new interfacial_velocity(d,analytical_soln_temp);
    analytical_soln_velINT[d]->t = tn+dt;
  }

  pressure_field_analytical.t=tn+dt;
  CF_DIM *analytical_soln_velNS_cf[P4EST_DIM] = {DIM(analytical_soln_velNS[0],analytical_soln_velNS[1],analytical_soln_velNS[2])};
  CF_DIM *analytical_soln_velINT_cf[P4EST_DIM] = {DIM(analytical_soln_velINT[0],analytical_soln_velINT[1],analytical_soln_velINT[2])};
  CF_DIM *analytical_soln_temp_cf[2] = {analytical_soln_temp[LIQUID_DOMAIN],analytical_soln_temp[SOLID_DOMAIN]};

  foreach_dimension(d){
    sample_cf_on_nodes(p4est,nodes,*analytical_soln_velNS_cf[d],vn_analytical.vec[d]);
    sample_cf_on_nodes(p4est,nodes,*analytical_soln_velINT_cf[d],v_interface_analytical.vec[d]);
  }
  sample_cf_on_nodes(p4est,nodes,*analytical_soln_temp_cf[LIQUID_DOMAIN],Tl_analytical.vec);
  sample_cf_on_nodes(p4est,nodes,*analytical_soln_temp_cf[SOLID_DOMAIN],Ts_analytical.vec);
  sample_cf_on_nodes(p4est,nodes,pressure_field_analytical,pn_analytical.vec);

  // Compute normal: (for evaluating error in vgamma)
  vec_and_ptr_dim_t normal;
  normal.create(p4est,nodes);
  compute_normals(*ngbd, phi.vec, normal.vec);
  normal.get_array();

  // Get errors:
  vec_and_ptr_dim_t vn_error;
  vec_and_ptr_t press_error;
  vec_and_ptr_t Tl_error;
  vec_and_ptr_t Ts_error;
  vec_and_ptr_t v_int_error; // Only use magnitude for v_int error, not component by component
  vec_and_ptr_t phi_error; // only for last timestep

  vn_error.create(p4est,nodes);
  press_error.create(p4est,nodes);
  Tl_error.create(p4est,nodes);
  Ts_error.create(p4est,nodes);
  v_int_error.create(p4est,nodes);

  double L_inf_u = 0., L_inf_v = 0.,L_inf_P = 0.,L_inf_Tl = 0., L_inf_Ts = 0., L_inf_vint = 0.,L_inf_phi=0.;

  vn_analytical.get_array(); vn_error.get_array(); v_NS.get_array();
  pn_analytical.get_array(); press_error.get_array(); press.get_array();

  Tl_analytical.get_array(); Tl_error.get_array(); Tl.get_array();
  Ts_analytical.get_array(); Ts_error.get_array(); Ts.get_array();
  v_interface_analytical.get_array();v_int_error.get_array(); v_interface.get_array();

  if((tn+dt)>=tfinal){
    phi_analytical.create(p4est,nodes); phi_error.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,level_set,phi_analytical.vec);

    phi_analytical.get_array();
    phi_error.get_array();

  }

  phi.get_array();
  foreach_node(n,nodes){
    if(phi.ptr[n]<0.){
      press_error.ptr[n] = fabs(press.ptr[n] - pn_analytical.ptr[n]);
      vn_error.ptr[0][n] = fabs(v_NS.ptr[0][n] - vn_analytical.ptr[0][n]);
      vn_error.ptr[1][n] = fabs(v_NS.ptr[1][n] - vn_analytical.ptr[1][n]);

      Tl_error.ptr[n] = fabs(Tl.ptr[n] - Tl_analytical.ptr[n]);

      L_inf_u = max(L_inf_u,vn_error.ptr[0][n]);
      L_inf_v = max(L_inf_v,vn_error.ptr[1][n]);
      L_inf_P = max(L_inf_P,press_error.ptr[n]);
      L_inf_Tl = max(L_inf_Tl,Tl_error.ptr[n]);
    }
    else{
      Ts_error.ptr[n] = fabs(Ts.ptr[n] - Ts_analytical.ptr[n]);

      L_inf_Ts = max(L_inf_Ts,Ts_error.ptr[n]);
    }

    // Check error in v_int and phi only in a uniform band around the interface
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){

      // Calculate the normal direction "strength" of the analytical velocity and calculated velocity:
      double v_int_ana_strength_normal = 0.0;
      double v_int_strength_normal = 0.0;
      foreach_dimension(d){
          v_int_ana_strength_normal += (v_interface_analytical.ptr[d][n]) *normal.ptr[d][n];
          v_int_strength_normal += (v_interface.ptr[d][n]) * normal.ptr[d][n];
      }

      // Now, calculate the error as the difference between the normal direction "strengths"
      v_int_error.ptr[n] = fabs(v_int_strength_normal - v_int_ana_strength_normal);

      L_inf_vint = max(L_inf_vint,v_int_error.ptr[n]);

      if((tn+dt)>=tfinal){ // Check phi error only at the final time
        phi_error.ptr[n] = fabs(phi.ptr[n] - phi_analytical.ptr[n]);
        L_inf_phi = max(L_inf_phi,phi_error.ptr[n]);
      }
    }


  }

  // Get the global errors:
  double local_Linf_errors[7] = {L_inf_u,L_inf_v,L_inf_P,L_inf_Tl,L_inf_Ts,L_inf_vint,L_inf_phi};
  double global_Linf_errors[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,7,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM,p4est->mpicomm);

  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n"
                             "Errors on Coupled Validation "
                             "\n -------------------------------------\n"
                             "Linf on u: %0.3e \n"
                             "Linf on v: %0.3e \n"
                             "Linf on P: %0.3e \n"
                             "Linf on Tl: %0.3e \n"
                             "Linf on Ts: %0.3e \n"
                             "Linf on v_int: %0.3e \n"
                             "Linf on phi: %0.3e (only relevant for last timestep)\n \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.3e \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],global_Linf_errors[6],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:
  PetscErrorCode ierr;
  ierr = PetscFOpen(p4est->mpicomm,filename_err_output,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d "
                                          "%g %g %g "
                                          "%g %g %g "
                                          "%g "
                                          "%d %g \n", tn+dt, dt, tstep,
                                                     global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                                                     global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],
                                                     global_Linf_errors[6],
                                                     num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  if(are_we_saving_vtk){
//    vorticity.get_array();

    // Save data:
//    std::vector<std::string> point_names;
//    std::vector<double*> point_data;
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(v_NS.vec[0], "u"));
    point_fields.push_back(Vec_for_vtk_export_t(v_NS.vec[1], "v"));
    point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vorticity"));
    point_fields.push_back(Vec_for_vtk_export_t(press.vec, "pressure"));
    point_fields.push_back(Vec_for_vtk_export_t(Tl.vec, "Tl"));
    point_fields.push_back(Vec_for_vtk_export_t(Ts.vec, "Ts"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[0], "v_int_x"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[1], "v_int_y"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_analytical.vec[0], "u_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_analytical.vec[1], "v_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(pn_analytical.vec, "P_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(Tl_analytical.vec, "Tl_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(Ts_analytical.vec, "Ts_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface_analytical.vec[0], "v_int_x_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface_analytical.vec[1], "v_int_y_ana"));
//    point_fields.push_back(Vec_for_vtk_export_t(phi_error.vec, "phi_err"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_error.vec[0], "u_err"));
    point_fields.push_back(Vec_for_vtk_export_t(vn_error.vec[1], "v_err"));
    point_fields.push_back(Vec_for_vtk_export_t(press_error.vec, "P_err"));
    point_fields.push_back(Vec_for_vtk_export_t(Tl_error.vec, "Tl_err"));
    point_fields.push_back(Vec_for_vtk_export_t(Ts_error.vec, "Ts_err"));
    point_fields.push_back(Vec_for_vtk_export_t(v_int_error.vec, "v_int_err"));


    if((tn+dt)>=tfinal){
        point_fields.push_back(Vec_for_vtk_export_t(phi_analytical.vec, "phi_ana"));
        point_fields.push_back(Vec_for_vtk_export_t(phi_error.vec, "phi_err"));


//      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
//                     "phi_ana","u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
//                     "phi_err","u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
//      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
//                    phi_analytical.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,
        //Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
//                    phi_error.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};

    }
//    else{
//      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
//                     "u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
//                     "u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
//      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
//                    vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
//                    vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};
//    }


//    std::vector<std::string> cell_names = {};
//    std::vector<double*> cell_data = {};
    // ELYCE TO-DO -- update these with the vector export wrapper

    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_fields, cell_fields);

    point_fields.clear(); cell_fields.clear();

//    point_names.clear(); point_data.clear();
//    cell_names.clear(); cell_data.clear();

//    vorticity.restore_array();
  }

  // Restore arrays:
  vn_analytical.restore_array(); vn_error.restore_array(); v_NS.restore_array();
  pn_analytical.restore_array(); press_error.restore_array(); press.restore_array();

  Tl_analytical.restore_array(); Tl_error.restore_array(); Tl.restore_array();
  Ts_analytical.restore_array(); Ts_error.restore_array(); Ts.restore_array();
  v_interface_analytical.restore_array();v_int_error.restore_array(); v_interface.restore_array();

  normal.restore_array();

  phi.restore_array();

  // Destroy arrays:
  vn_analytical.destroy(); vn_error.destroy();
  pn_analytical.destroy(); press_error.destroy();

  Tl_analytical.destroy(); Tl_error.destroy();
  Ts_analytical.destroy(); Ts_error.destroy();
  v_interface_analytical.destroy();v_int_error.destroy();

  normal.destroy();

  // Handle phi error checking last if it was done:
  if((tn+dt)>=tfinal){
    phi_error.restore_array();
    phi_error.destroy();
    phi_analytical.restore_array();
    phi_analytical.destroy();
  }

  // Delete the objects created:
  for(unsigned char d=0;d<2;++d){
    delete analytical_soln_temp[d];
  }
  for(unsigned char d=0;d<P4EST_DIM;++d){
    delete analytical_soln_velNS[d];
    delete analytical_soln_velINT[d];
  }
} // end of "save_coupled_test_case()"


void save_test_case_errors_and_vtk(mpi_environment_t& mpi,
                                   my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver,
                                   const p4est_t* p4est_np1, const p4est_nodes_t* nodes_np1,
                                   const p4est_ghost_t* ghost_np1, my_p4est_node_neighbors_t* ngbd_np1,
                                   vec_and_ptr_t phi,
                                   vec_and_ptr_t T_l_n, vec_and_ptr_t T_s_n,
                                   vec_and_ptr_dim_t v_interface,
                                   vec_and_ptr_dim_t v_n,
                                   vec_and_ptr_t press_nodes, vec_and_ptr_t vorticity,
                                   FILE* fich_errors, char name_errors[],
                                   double dxyz_close_to_interface,
                                   bool are_we_saving, int out_idx){

  // Get the vtk output directory:
  const char* out_dir_test_vtk = getenv("OUT_DIR_VTK");
  char output[1000];
  sprintf(output, "%s/snapshot_test_%d_lmin_%d_lmax_%d_outidx_%d", out_dir_test_vtk, example_, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax(), out_idx);

  switch(example_){
  case NS_GIBOU_EXAMPLE:{
    // In typical saving, only compute pressure nodes when we save to vtk. For this example, save pressure nodes every time so we can check the error

    // Save the test case info:
    save_navier_stokes_test_case(p4est_np1, nodes_np1, ghost_np1,
                                 phi, v_n, press_nodes, vorticity,
                                 dxyz_close_to_interface, are_we_saving, output,
                                 name_errors, fich_errors);

    break;
  }
  case COUPLED_TEST_2:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:{
    save_coupled_test_case(p4est_np1, nodes_np1, ghost_np1, ngbd_np1,
                           phi, T_l_n, T_s_n, v_interface, v_n, press_nodes, vorticity,
                           dxyz_close_to_interface, are_we_saving, output,
                           name_errors, fich_errors);
    break;
  }

  case FRANK_SPHERE:{
    save_stefan_test_case(p4est_np1, nodes_np1, ghost_np1,
                          T_l_n, T_s_n,
                          phi, v_interface,
                          dxyz_close_to_interface, are_we_saving,
                          output, name_errors, fich_errors);
    break;
  }
  }
} // end of "save_test_case_errors_and_vtk()"



void save_fluid_forces_and_or_area_data(mpi_environment_t& mpi,
                                        const p4est_t* p4est_np1, const p4est_nodes_t* nodes_np1,
                                        my_p4est_navier_stokes_t* ns,
                                        vec_and_ptr_t phi,
                                        bool compute_pressure,
                                        FILE* fich_data, char name_data[], int& last_tstep){

  PetscErrorCode ierr;

  double forces[P4EST_DIM];
  double solid_area;
  if(save_fluid_forces || save_area_data){
    // Open file
    ierr = PetscFOpen(mpi.comm(), name_data, "a", &fich_data);
    // Print time to console
    ierr = PetscPrintf(mpi.comm(), "tn = %g \n", tn);
    // Print time to file
    ierr = PetscFPrintf(mpi.comm(), fich_data, "%g ", tn);
  }

  if(save_area_data || wrap_up_simulation_if_solid_has_vanished){
    // Compute the solid area:
    VecScaleGhost(phi.vec, -1.);
    solid_area = area_in_negative_domain(p4est_np1, nodes_np1, phi.vec);
    VecScaleGhost(phi.vec, -1.);

    if(save_area_data){
      // Print area to console
      ierr = PetscPrintf(mpi.comm(), "A = %g \n", solid_area);

      // Print area to file
      ierr = PetscFPrintf(mpi.comm(), fich_data, "%g ", solid_area);
    }

    if(wrap_up_simulation_if_solid_has_vanished){
      // Check if the solid has vanished:
      if(solid_area<EPS){
        PetscPrintf(mpi.comm(), "The solid region has vanished, wrapping up now ... \n");
        last_tstep = tstep;
      }
    }
  }


  if(save_fluid_forces){
    // Compute the fluid forces
    ns->compute_forces(forces);

    // Print force data to console
    ierr = PetscPrintf(mpi.comm(), "fx = %g, fy = %g \n", forces[0], forces[1]);

    // Print area to file
    ierr = PetscFPrintf(mpi.comm(), fich_data, "%g %g ", forces[0], forces[1]);

  }


  if(save_fluid_forces || save_area_data){
    // End the line and Close the file
    ierr = PetscFPrintf(mpi.comm(), fich_data, "\n");
    ierr = PetscFClose(mpi.comm(), fich_data); CHKERRXX(ierr);
    PetscPrintf(mpi.comm(), "The data has been saved ... \n");
  }
} // end of "save_fluid_forces_and_or_area_data()"

void do_mem_safety_check(mpi_environment_t& mpi,
                         p4est_nodes_t* nodes_np1, bool are_we_saving,
                         FILE* fich_mem, char name_mem[]){

  PetscLogDouble mem_safety_check;
  PetscErrorCode ierr;

  MPI_Barrier(mpi.comm());
  PetscMemoryGetCurrentUsage(&mem_safety_check);


  int no = nodes_np1->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&no,1,MPI_INT,MPI_SUM,mpi.comm());


  MPI_Allreduce(MPI_IN_PLACE,&mem_safety_check,1,MPI_DOUBLE,MPI_SUM,mpi.comm());

  PetscPrintf(mpi.comm(),"\n"
                          "Memory safety check:\n"
                          " - Current memory usage is : %0.9e GB \n"
                          " - Number of grid nodes is: %d \n"
                          " - Percent of safety limit: %0.2f % \n \n \n",
              mem_safety_check*1.e-9,
              no,
              (mem_safety_check)/(mem_safety_limit)*100.0);

    ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_mem,"%d %g %d %d \n",tstep, mem_safety_check, no, are_we_saving);CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);
} // end of "do_mem_safety_check()"




void perform_saving_tasks_of_interest(mpi_environment_t &mpi, my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver, int& out_idx, int& data_save_out_idx, bool& compute_pressure_,
                                      int load_tstep, int last_tstep,
                                      FILE* fich_errors, char name_errors[],
                                      FILE* fich_data, char name_data[],
                                      FILE* fich_mem, char name_mem[]){
  // --------------------------------------------------------------------------------------------------------------
  // Save the fluid forces and/or area
  // --------------------------------------------------------------------------------------------------------------
  // TO-DO: Update this using the solver:

  if(are_we_saving_data(tn, tstep==load_tstep, data_save_out_idx, false) || wrap_up_simulation_if_solid_has_vanished){
    my_p4est_node_neighbors_t* ngbd_np1 = stefan_w_fluids_solver->get_ngbd_np1();

    save_fluid_forces_and_or_area_data(mpi,
                                       ngbd_np1->get_p4est(), ngbd_np1->get_nodes(),
                                       stefan_w_fluids_solver->get_ns_solver(),
                                       stefan_w_fluids_solver->get_phi(), compute_pressure_,
                                       fich_data, name_data, last_tstep);
  }


  // --------------------------------------------------------------------------------------------------------------
  // Saving to VTK: either every specified number of iterations, or every specified dt:
  // Note: we do this after extension of fields to make visualization nicer
  // --------------------------------------------------------------------------------------------------------------
  // (Saving-a) Determine if we are saving this timestep:
  // ---------------------------
  bool are_we_saving = false;

  are_we_saving = are_we_saving_vtk(tstep, tn, tstep==load_tstep, out_idx, true) /*&& (tstep>0)*/;
  // ---------------------------
  // (Saving-b) Save to VTK if applicable:
  // ---------------------------
  if(are_we_saving){
//    PetscPrintf(mpi.comm(), "outidx = %d \n", out_idx);
    stefan_w_fluids_solver->save_fields_to_vtk(out_idx);
  } // end of if "are we saving"

  // ---------------------------
  // (Saving-c) Check errors on validation cases if relevant,
  // save errors to vtk if we are saving this timestep
  // ---------------------------

  if(example_is_a_test_case){
    my_p4est_node_neighbors_t* ngbd_np1 = stefan_w_fluids_solver->get_ngbd_np1();
    PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n", stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
    save_test_case_errors_and_vtk(mpi, stefan_w_fluids_solver,
                                  ngbd_np1->get_p4est(), ngbd_np1->get_nodes(), ngbd_np1->get_ghost(), ngbd_np1,
                                  stefan_w_fluids_solver->get_phi(),
                                  stefan_w_fluids_solver->get_T_l_n(), stefan_w_fluids_solver->get_T_s_n(),
                                  stefan_w_fluids_solver->get_v_interface(),
                                  stefan_w_fluids_solver->get_v_n(),
                                  stefan_w_fluids_solver->get_press_nodes(), stefan_w_fluids_solver->get_vorticity(),
                                  fich_errors, name_errors,
                                  stefan_w_fluids_solver->get_dxyz_close_to_interface(),
                                  are_we_saving, out_idx);
  }

  // -------------------------------
  // (Saving-d) Do a memory safety check as user specified
  // -------------------------------
  if((check_mem_every_iter>0) && ((tstep%check_mem_every_iter)==0)){
    do_mem_safety_check(mpi, stefan_w_fluids_solver->get_nodes_np1(), are_we_saving, fich_mem, name_mem);
  }
} // end of "perform_saving_tasks_of_interest()"

// --------------------------------------------------------------------------------------------------------------
// Initializations and destructions:
// --------------------------------------------------------------------------------------------------------------


void setup_initial_parameters_and_report(mpi_environment_t& mpi, my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver, int grid_res_iter){
  select_solvers(); // Note, this function must be called "BEFORE" set_geometry()

  stefan_w_fluids_solver->set_solve_stefan(solve_stefan);
  stefan_w_fluids_solver->set_solve_navier_stokes(solve_navier_stokes);
  stefan_w_fluids_solver->set_there_is_substrate(example_uses_inner_LSF);
  stefan_w_fluids_solver->set_do_we_solve_for_Ts(do_we_solve_for_Ts);
  stefan_w_fluids_solver->set_is_dissolution_case(is_dissolution_case);
  stefan_w_fluids_solver->set_start_w_merged_grains(start_w_merged_grains);
  stefan_w_fluids_solver->set_use_regularize_front(use_regularize_front);
  stefan_w_fluids_solver->set_use_collapse_onto_substrate(use_collapse_onto_substrate);
  stefan_w_fluids_solver->set_track_evolving_geometries(track_evolving_geometries);
  stefan_w_fluids_solver->set_use_boussinesq(use_boussinesq);
  stefan_w_fluids_solver->set_force_interfacial_velocity_to_zero(force_interfacial_velocity_to_zero);

  stefan_w_fluids_solver->set_loading_from_previous_state(loading_from_previous_state);

  stefan_w_fluids_solver->set_proximity_smoothing(proximity_smoothing);
  stefan_w_fluids_solver->set_proximity_collapse(proximity_collapse);

  stefan_w_fluids_solver->set_v_interface_max_allowed(v_int_max_allowed);

  stefan_w_fluids_solver->set_NS_adv_order(NS_advection_sl_order);
  stefan_w_fluids_solver->set_advection_sl_order(advection_sl_order);
  // ------------------------------
  // Make sure your flags are set to solve at least one of the problems:
  // ------------------------------
  if(!solve_stefan && !solve_navier_stokes){
    throw std::invalid_argument("Woops, you haven't set options to solve either type of physical problem. \n"
                                "You must at least set solve_stefan OR solve_navier_stokes to true. ");
  }


  // -----------------------------------------------
  // Set up domain :
  // -----------------------------------------------
  // domain size information
  set_geometry();


  // INSERT HERE: Set values in solver
  double xyz_min[P4EST_DIM] = {DIM(xmin, ymin, 0)};
  double xyz_max[P4EST_DIM] = {DIM(xmax, ymax, 0)};
  int periodicity[P4EST_DIM] = {DIM(px, py, 0)};
  int ntrees[P4EST_DIM] = {DIM(nx, ny, 0)};

  stefan_w_fluids_solver->set_xyz_min(xyz_min);
  stefan_w_fluids_solver->set_xyz_max(xyz_max);
  stefan_w_fluids_solver->set_periodicity(periodicity);
  stefan_w_fluids_solver->set_ntrees(ntrees);

  // TO-DO: update this to include grid res iter
  stefan_w_fluids_solver->set_lmin_lint_lmax(lmin+grid_res_iter, (lint>0) ? lint+grid_res_iter: lint, lmax+grid_res_iter);
  stefan_w_fluids_solver->set_uniform_band(uniform_band);
  // Note: we only pass lint if it was already specified as nonzero, otherwise just pass whatever the user passed

  //
  // if porous media example, create the porous media geometry:
  // do this operation on one process so they don't have different grain defn's.
  /*if(mpi.rank() == 0) */
  if(example_ == EVOLVING_POROUS_MEDIA){
    make_LSF_for_porous_media(mpi);
  }


  // Set the LSF(s):
  stefan_w_fluids_solver->set_LSF_CF(&level_set);
  if(example_uses_inner_LSF) stefan_w_fluids_solver->set_substrate_LSF_CF(&substrate_level_set);

  // -----------------------------------------------
  // Set applicable parameters and nondim groups:
  // -----------------------------------------------
  select_problem_nondim_or_dim_formulation();
  stefan_w_fluids_solver->set_problem_dimensionalization_type(problem_dimensionalization_type);

  // Set the defn's for the nondim temp/conc problem, and pass to solver:
  set_temp_conc_nondim_defns();
  stefan_w_fluids_solver->set_dim_temp_conc_variables(Tinfty, Tinterface, T0);
  PetscPrintf(mpi.comm(), "we are setting theta_inf = %0.2f, theta_int = %0.2f, theta0 = %0.2f, delta T as %0.4e \n ", theta_infty, theta_interface, theta0, deltaT);
  stefan_w_fluids_solver->set_nondim_temp_conc_variables(theta_infty, theta_interface, theta0, deltaT);

  // Set the physical properties and pass to solver:
  set_physical_properties();
  stefan_w_fluids_solver->set_l_char(l_char);

  stefan_w_fluids_solver->set_alpha_l(alpha_l);
  stefan_w_fluids_solver->set_alpha_s(alpha_s);

  stefan_w_fluids_solver->set_k_l(k_l);
  stefan_w_fluids_solver->set_k_s(k_s);

  stefan_w_fluids_solver->set_rho_l(rho_l);
  stefan_w_fluids_solver->set_rho_s(rho_s);

  stefan_w_fluids_solver->set_cp_s(cp_s);
  stefan_w_fluids_solver->set_L(L);
  stefan_w_fluids_solver->set_sigma(sigma);

  stefan_w_fluids_solver->set_mu_l(mu_l);

  stefan_w_fluids_solver->set_grav(grav);
  stefan_w_fluids_solver->set_beta_T(beta_T);
  stefan_w_fluids_solver->set_beta_C(beta_C);

  stefan_w_fluids_solver->set_gamma_diss(gamma_diss); // this one gets computed in the solver
  stefan_w_fluids_solver->set_stoich_coeff_diss(stoich_coeff_diss);
  stefan_w_fluids_solver->set_molar_volume_diss(molar_volume_diss);
  stefan_w_fluids_solver->set_k_diss(k_diss);
  stefan_w_fluids_solver->set_precip_disso_vgamma_calc_type(
      (precipitation_dissolution_interface_velocity_calc_type_t)precip_disso_vgamma_calc);
  if(is_dissolution_case){
    stefan_w_fluids_solver->set_gamma_diss(gamma_diss); // this one gets computed in the solver
    stefan_w_fluids_solver->set_stoich_coeff_diss(stoich_coeff_diss);
    stefan_w_fluids_solver->set_molar_volume_diss(molar_volume_diss);
    stefan_w_fluids_solver->set_k_diss(k_diss);
    stefan_w_fluids_solver->set_precip_disso_vgamma_calc_type(
        (precipitation_dissolution_interface_velocity_calc_type_t)precip_disso_vgamma_calc);

    // This governs the expression of the RObin BC (and possibly interface velocity expression) in the
    // precipitation/dissolution problem. For the dissolving disk benchmark, we want to match their
    // specific rate law so the added term is 0. However for more general cases, we use the k(C-Csat) formulation, so we
    // will need an added -1 in the nondimensional expression
    // TO-DO: probably need to add an argument to throw if we do this dimensionally, since we don't have that implemented at the moment
    if(example_ == DISSOLVING_DISK_BENCHMARK) {
      stefan_w_fluids_solver->set_disso_interface_condition_added_term(0.);
    }
    else {
      stefan_w_fluids_solver->set_disso_interface_condition_added_term(-1.);
    }

    // If we only prescribe Sc, we gotta compute some effective "Dl" for us to use for boundary conditions, and for the stefan solver to use for vel_nondim_to_dim
    if(Dl<0.){
      if(Sc>0.){
        Dl = mu_l / (rho_l * Sc);
        PetscPrintf(mpi.comm(), "\n \n ---------- \n Warning!! \n ---------- \n : No physical Dl was provided, we computed Dl = %0.3e using rho_l = %0.3e, mu_l = %0.3e, and Sc = %0.2e. Please modify this if this is not the desired outcome. \n \n", Dl, rho_l, mu_l, Sc);
      }
      else{
        throw std::invalid_argument("main_2d.cpp: You must provide either a valid Dl or Sc in order to run this problem with nondim_by_scalar_diffusivity \n");

      }
    }

    stefan_w_fluids_solver->set_Dl(Dl);
    stefan_w_fluids_solver->set_Ds(Ds);
  }

  // Check that the parameters were set:
  stefan_w_fluids_solver->print_physical_parameters();

  // Set nondim groups (if they've been prescribed, a.k.a not the default of -1):
  if(Re>=0.) stefan_w_fluids_solver->set_Re(Re);
  if(Pr>=0.) stefan_w_fluids_solver->set_Pr(Pr);
  if(Sc>=0.) stefan_w_fluids_solver->set_Sc(Sc);
  if(Pe>=0.) stefan_w_fluids_solver->set_Pe(Pe);
  if(St>=0.) stefan_w_fluids_solver->set_St(St);
  if(Da>=0.) stefan_w_fluids_solver->set_Da(Da);
  if(RaT>=0.) stefan_w_fluids_solver->set_RaT(RaT);
  if(RaC>=0.) stefan_w_fluids_solver->set_RaC(RaC);

  // The solver will automatically set nondim groups during the initializations, but
  // we do it here so we can get the time_nondim_to_dim info we need to set a proper duration for sim time
  stefan_w_fluids_solver->set_nondimensional_groups();

  // Get nondim groups as theyve been computed
  Re = stefan_w_fluids_solver->get_Re();
  Pr = stefan_w_fluids_solver->get_Pr();
  Sc = stefan_w_fluids_solver->get_Sc();
  Pe = stefan_w_fluids_solver->get_Pe();
  St = stefan_w_fluids_solver->get_St();
  Da = stefan_w_fluids_solver->get_Da();
  RaT = stefan_w_fluids_solver->get_RaT();
  RaC = stefan_w_fluids_solver->get_RaC();

  if(solve_navier_stokes){
    set_NS_info();
  }
  stefan_w_fluids_solver->set_hodge_percentage_of_max_u(hodge_percentage_of_max_u);
  stefan_w_fluids_solver->set_hodge_max_iteration(50);

  stefan_w_fluids_solver->set_NS_max_allowed(max(NS_max_allowed, u0*10.));
  PetscPrintf(mpi.comm()," we are setting %f \n",max(NS_max_allowed, u0*10.) );
  // INSERT HERE: Set values in solver?
  // INSERT HERE: set_nondimensional_groups() (from solver);

  // Refinement:
  stefan_w_fluids_solver->set_refine_by_d2T(refine_by_d2T);
//  stefan_w_fluids_solver->set_d2T_ref_threshold(gradT_threshold);

  stefan_w_fluids_solver->set_d2T_refinement_thresholds(d2T_refine_threshold, d2T_coarsen_threshold);
  stefan_w_fluids_solver->set_vorticity_ref_threshold(vorticity_threshold);

  // Reinitialization:
  stefan_w_fluids_solver->set_reinit_every_iter(reinit_every_iter);

  // Phi advection substeps:
  stefan_w_fluids_solver->set_do_phi_advection_substeps(do_phi_advection_substeps);
//  stefan_w_fluids_solver->set_num_phi_advection_substeps(num_phi_advection_substeps);
//  stefan_w_fluids_solver->set_phi_advection_substeps_coeff(phi_advection_substeps_coeff);
  stefan_w_fluids_solver->set_CFL_phi_advection_substep(cfl_phi_advection_substep);


  // -----------------------------------------------
  // Get the simulation time info (it is example dependent): -- Must be set after non dim groups
  // -----------------------------------------------

  time_nondim_to_dim = stefan_w_fluids_solver->get_time_nondim_to_dim();
  vel_nondim_to_dim = stefan_w_fluids_solver->get_vel_nondim_to_dim();

  PetscPrintf(mpi.comm(), "Values in main are: \n "
                          "time_nondim_to_dim = %0.3e \n"
                          "vel_nondim_to_dim = %0.3e \n "
                          "pressure_drop_nondim = %0.3e \n", time_nondim_to_dim, vel_nondim_to_dim, (pressure_drop*l_char*l_char/rho_l/Dl/Dl) );

  simulation_time_info(); // INSERT HERE: update in solver as necessary
  stefan_w_fluids_solver->set_tn(tstart);
  stefan_w_fluids_solver->set_tfinal(tfinal);
  PetscPrintf(mpi.comm(), "Set tfinal as %f \n", tfinal);
  stefan_w_fluids_solver->set_dt_max_allowed(dt_max_allowed);

  stefan_w_fluids_solver->set_cfl_Stefan(cfl);
  stefan_w_fluids_solver->set_cfl_NS(cfl_NS);

  // Other:
  stefan_w_fluids_solver->set_print_checkpoints(print_checkpoints);

//  // Tell the user if they've chosen some incompatible save options:
  if(save_state_using_dt>0 && save_state_using_iter){
    throw std::invalid_argument("main_2d.cpp: you have chosen save_state_using_dt and save_state_using_iter. You can only choose one of these \n");
  }


  // -----------------------------------------------
  // Report relevant information:
  // -----------------------------------------------

    PetscPrintf(mpi.comm(),"------------------------------------"
                            "\n \n"
                            "--> We are running EXAMPLE %d \n"
                            "------------------------------------\n\n",example_);
} // end of "setup_initial_parameters_and_report()"


void initialize_error_files_for_test_cases(mpi_environment_t& mpi,
                                           my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver,
                                           FILE* fich_errors, char name_errors[],
                                           FILE* fich_data, char name_data[],
                                           FILE* fich_mem, char name_mem[]){
  PetscErrorCode ierr;

  // Get the output directory to put the error files:
  // Output file for Frank Sphere errors:
  const char* out_dir_files = getenv("OUT_DIR_FILES");
  if(!out_dir_files){
    throw std::invalid_argument("You need to set the environment variable OUT_DIR_FILES to save stefan errors");
  }

  switch(example_){
  case FRANK_SPHERE:{

    sprintf(name_errors,"%s/frank_sphere_error_lmin_%d_lmax_%d.dat",
            out_dir_files, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());

    ierr = PetscFOpen(mpi.comm(),name_errors,"w",&fich_errors); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_errors,"time " "timestep " "iteration "
                                                        "phi_error " "T_l_error " "T_s_error "
                                                        "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_errors); CHKERRXX(ierr);
    break;
  }
  case NS_GIBOU_EXAMPLE:{
    // Output file for NS test case errors:

    sprintf(name_errors,"%s/navier_stokes_error_lmin_%d_lmax_%d.dat",
            out_dir_files, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());

    ierr = PetscFOpen(mpi.comm(),name_errors, "w", &fich_errors); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_errors,"time " "timestep " "iteration " "u_error "
                                                    "v_error " "P_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_errors); CHKERRXX(ierr);

    break;
  }
  case COUPLED_TEST_2:
  case COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP:
  case COUPLED_PROBLEM_EXAMPLE:{
    // Output file for coupled problem test case:
    sprintf(name_errors,"%s/coupled_error_lmin_%d_lmax_%d.dat",
            out_dir_files,  stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());

    ierr = PetscFOpen(mpi.comm(),name_errors,"w",&fich_errors); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_errors,"time " "timestep " "iteration "
                                                         "u_error " "v_error " "P_error "
                                                         "Tl_error " "Ts_error " "vint_error" "phi_error "
                                                         "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_errors); CHKERRXX(ierr);
    break;
  }
  case FLOW_PAST_CYLINDER:
  case DISSOLVING_DISK_BENCHMARK:
  case EVOLVING_POROUS_MEDIA:
  case MELTING_ICE_SPHERE_NAT_CONV:
  case MELTING_ICE_SPHERE:
  case ICE_AROUND_CYLINDER:{
    if(save_fluid_forces || save_area_data){
      switch(problem_dimensionalization_type){
        case NONDIM_BY_FLUID_VELOCITY:{
          if(is_dissolution_case){
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_Re_%0.2f_gamma_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Re, gamma_diss, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }
          else{
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_Re_%0.2f_St_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Re, St, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }


          break;
        }
        case NONDIM_BY_SCALAR_DIFFUSIVITY:{
          if(is_dissolution_case){
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_Sc_%0.2f_gamma_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Sc, gamma_diss, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }
          else{
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_Pr_%0.2f_St_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Pr, St, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }
          break;
        }
        case DIMENSIONAL:{
          if(is_dissolution_case){
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_dimensional_Re_%0.2f_gamma_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Re, gamma_diss, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }
          else{
            // Output file for fluid forces or area data
            sprintf(name_data,"%s/area_and_or_force_data_dimensional_Re_%0.2f_St_%0.2f_lmin_%d_lmax_%d.dat",
                    out_dir_files, Re, St, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());
          }
          break;
        }
        default:{
          throw std::invalid_argument("Output file initialization: unknown problem dimensionalization type \n");
        }
      }
      // Test if the file exists, change it accordingly:
      // Note: we copy the file name over first because on Pod for whatever reason, it doesn't like when you sprintf a string into itself and it fails, so that's why I have to do this here even though it's weird

      bool file_exists = true;

      char name_data_test[1000];
      int i = 1;
      while(file_exists){
        sprintf(name_data_test, "%s", name_data);


        if(mpi.rank() == 0){
          std::ifstream file2check(name_data_test);
          file_exists = file2check.good();
        }
        int mpi_err = MPI_Bcast(&file_exists, 1, MPI_INTEGER, 0, mpi.comm());

        if(file_exists){
          sprintf(name_data, "%s-%d", name_data_test, i);
          i++;
        }
      }


      // TO-DO: adjust this logic appropriately
      ierr = PetscFOpen(mpi.comm(),name_data,"w",&fich_data); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(),fich_data,"tn ");CHKERRXX(ierr);

      if(save_area_data){
        ierr = PetscFPrintf(mpi.comm(),fich_data,"A ");CHKERRXX(ierr);
      }
      if(save_fluid_forces){
        ierr = PetscFPrintf(mpi.comm(),fich_data,"fx fy ");CHKERRXX(ierr);

      }
      ierr = PetscFPrintf(mpi.comm(),fich_data,"\n");CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(),fich_data); CHKERRXX(ierr);

    break;
  }
  }

  default:{
    break;
  }
  }


  // Initialize memory file if we are doing a memory safety check:
  if(check_mem_every_iter>0){
    sprintf(name_mem,"%s/memory_check_lmin_%d_lmax_%d.dat",
            out_dir_files, stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax());

    ierr = PetscFOpen(mpi.comm(), name_mem, "w", &fich_mem); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_mem, "tstep mem num_nodes vtk_bool \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich_mem); CHKERRXX(ierr);
  }
} // end of "initialize_error_files_for_test_cases()"



//(V) want to keep this in main
void initialize_all_relevant_bcs_ics_forcing_terms(my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver,
                                                   temperature_field* analytical_T[2],
                                                   BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2],
                                                   BC_WALL_VALUE_TEMP* bc_wall_value_temp[2],
                                                   external_heat_source* external_heat_source_T[2],
                                                   velocity_component* analytical_soln_v[P4EST_DIM],
                                                   BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM],
                                                   BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                                   BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM],
                                                   BC_INTERFACE_VALUE_PRESSURE& bc_interface_value_pressure,
                                                   BC_WALL_VALUE_PRESSURE& bc_wall_value_pressure,
                                                   BC_WALL_TYPE_PRESSURE& bc_wall_type_pressure,
                                                   external_force_per_unit_volume_component* external_force_components[P4EST_DIM],
                                                   external_force_per_unit_volume_component_with_boussinesq_approx* external_force_components_with_BA[P4EST_DIM],
                                                   INITIAL_TEMP* T_init_cf[2], temperature_field* analytical_temp_init[2],
                                                   INITIAL_VELOCITY* v_init_cf[P4EST_DIM], velocity_component* analytical_soln_v_init[P4EST_DIM]){

  // ------------------------------
  // Analytical v (used for some coupled cases so has to be
  // initialized first bc forcing terms in heat and momentum both rely on this)
  // --> Create analytical velocity field for each Cartesian direction if needed:
  // ------------------------------
  if(analytical_IC_BC_forcing_term){
    for(unsigned char d=0;d<P4EST_DIM;d++){
      analytical_soln_v[d] = new velocity_component(d);
      analytical_soln_v[d]->t = tn+dt;
    }
  }
  // ------------------------------
  // For temperature problem:
  // ------------------------------
  if(solve_stefan){
    // Create analytical temperature field for each domain if needed:
    for(unsigned char d=0;d<2;++d){
      if(analytical_IC_BC_forcing_term){ // TO-DO: make all incrementing consistent
        analytical_T[d] = new temperature_field(d);
        analytical_T[d]->t = tn+dt;
      }
    }
    // Set the bc interface type for temperature:
    interface_bc_temp();
    if(example_uses_inner_LSF){inner_interface_bc_temp();} // inner boundary bc

    // Create necessary RHS forcing terms and BC's
    for(unsigned char d=0;d<2;++d){
      bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(stefan_w_fluids_solver,
                                                             interfacial_temp_bc_requires_curvature,
                                                             interfacial_temp_bc_requires_normal);

      if(analytical_IC_BC_forcing_term){
        external_heat_source_T[d] = new external_heat_source(d,analytical_T,analytical_soln_v);
        external_heat_source_T[d]->t = tn+dt;
//        bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(NULL,NULL,analytical_T,d);
        bc_interface_val_temp[d]->setup_ana_case(analytical_T, d);
        bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d,analytical_T);
      }
      else{
        bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d);
      }
    }
  }


  if(solve_navier_stokes){
    for(unsigned char d=0;d<P4EST_DIM;d++){
      bc_interface_value_velocity[d] = new BC_interface_value_velocity(stefan_w_fluids_solver, interfacial_vel_bc_requires_vint);

      // Set the BC types:
      BC_INTERFACE_TYPE_VELOCITY(d);
      bc_wall_type_velocity[d] = new BC_WALL_TYPE_VELOCITY(d);

      // Set the BC values (and potential forcing terms) depending on what we are running:
      if(analytical_IC_BC_forcing_term){
        // Interface conditions values:
        bc_interface_value_velocity[d]->setup_ana_case(analytical_soln_v, d);
//        bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL,analytical_soln_v);
        bc_interface_value_velocity[d]->t = tn+dt;

        // Wall conditions values:
        bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d,analytical_soln_v);
        bc_wall_value_velocity[d]->t = tn+dt;

        if (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP){
          for(unsigned char domain=0;domain<2;++domain){
            // External forcing terms:
            external_force_components_with_BA[d] = new external_force_per_unit_volume_component_with_boussinesq_approx(domain,d,analytical_T,analytical_soln_v);
            external_force_components_with_BA[d]->t = tn+dt;
          }
        }else{
          // External forcing terms:
          external_force_components[d] = new external_force_per_unit_volume_component(d,analytical_soln_v);
          external_force_components[d]->t = tn+dt;
        }
      }
      else{
//        // Interface condition values:
//        bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL); // initialize null for now, will add relevant neighbors and vector as required later on

        // Wall condition values:
        bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d);
      }
    }
    interface_bc_pressure(); // sets the interfacial bc type for pressure
    bc_wall_value_pressure.t = tn+dt;
  }



  // ------------------------------
  // Initial condition fields:
  // ------------------------------
//  INITIAL_TEMP *T_init_cf[2];
//  temperature_field* analytical_temp[2];
  if(solve_stefan){
    if(analytical_IC_BC_forcing_term){
      coupled_test_sign = 1.;
      vel_has_switched=false;

      for(unsigned char d=0;d<2;++d){
        analytical_temp_init[d]= new temperature_field(d);
        analytical_temp_init[d]->t = tstart;
      }
      for(unsigned char d=0;d<2;++d){
        T_init_cf[d]= new INITIAL_TEMP(d,analytical_temp_init);
      }

    }
    else{
      for(unsigned char d=0;d<2;++d){
        T_init_cf[d] = new INITIAL_TEMP(d);
        T_init_cf[d]->t = tstart;
      }
    }
  }

//  INITIAL_VELOCITY *v_init_cf[P4EST_DIM];
//  velocity_component* analytical_soln[P4EST_DIM];
  if(solve_navier_stokes){
    if(analytical_IC_BC_forcing_term)
    {
      for(unsigned char d=0;d<P4EST_DIM;++d){
        analytical_soln_v_init[d] = new velocity_component(d);
        analytical_soln_v_init[d]->t = tstart;
      }
    }
    for(unsigned char d=0;d<P4EST_DIM;++d){
      if(analytical_IC_BC_forcing_term){
        v_init_cf[d] = new INITIAL_VELOCITY(d,analytical_soln_v_init);
        v_init_cf[d]->t = tstart;
      }
      else {
        v_init_cf[d] = new INITIAL_VELOCITY(d);
      }
    }
  }

  // TO-DO: check that these new guys get deleted, I think it might be okay bc they're defined within the fxn?

  // ------------------------------
  // Set initial fields for the solver!
  // ------------------------------
  // Initial fields:
  CF_DIM* initial_temp[2] = {T_init_cf[0], T_init_cf[1]};
  stefan_w_fluids_solver->set_initial_temp_n(initial_temp);
  stefan_w_fluids_solver->set_initial_temp_nm1(initial_temp);

  CF_DIM* initial_velocity[P4EST_DIM] = {DIM(v_init_cf[0], v_init_cf[1], v_init_cf[2])};
  stefan_w_fluids_solver->set_initial_NS_velocity_n_(initial_velocity);
  stefan_w_fluids_solver->set_initial_NS_velocity_nm1_(initial_velocity);

  stefan_w_fluids_solver->set_initial_refinement_CF(&initial_refinement_cf);


  // TO-DO: do these BC's need to be set at every timestep, or only for the first one??
  // ------------------------------
  // Set temp/conc BC's and forcing terms
  // ------------------------------
  // Interface:
  // ------------
  // Value:
  my_p4est_stefan_with_fluids_t::interfacial_bc_temp_t* bc_interface_val_temp_[2] =
      {bc_interface_val_temp[LIQUID_DOMAIN],
       bc_interface_val_temp[SOLID_DOMAIN]};
  stefan_w_fluids_solver->set_bc_interface_value_temp(bc_interface_val_temp_);

  // Type:
  BoundaryConditionType* bc_interface_type_temp_[2] = {&interface_bc_type_temp, &interface_bc_type_temp};
  stefan_w_fluids_solver->set_bc_interface_type_temp(bc_interface_type_temp_);

  // Robin coeff:
  CF_DIM* bc_interface_coeff_[2] = {&bc_interface_coeff, &bc_interface_coeff};
  stefan_w_fluids_solver->set_bc_interface_robin_coeff_temp(bc_interface_coeff_);

  // ------------
  // Wall:
  // ------------
  // Value:
  CF_DIM* bc_wall_value_temp_[2] = {bc_wall_value_temp[LIQUID_DOMAIN], bc_wall_value_temp[SOLID_DOMAIN]};
  stefan_w_fluids_solver->set_bc_wall_value_temp(bc_wall_value_temp_);

  // Type:
  WallBCDIM* bc_wall_type_temp_[2] = {&bc_wall_type_temp, &bc_wall_type_temp};
  stefan_w_fluids_solver->set_bc_wall_type_temp(bc_wall_type_temp_);


  // ------------
  // Substrate interface:
  // ------------
  // Value:
  CF_DIM* bc_interface_val_substrate_[2] = {&bc_interface_val_inner, &bc_interface_val_inner};
  stefan_w_fluids_solver->set_bc_interface_value_temp_substrate(bc_interface_val_substrate_);

  // Type:
  BoundaryConditionType* bc_interface_type_substrate_[2] = {&inner_interface_bc_type_temp, &inner_interface_bc_type_temp};
  stefan_w_fluids_solver->set_bc_interface_type_temp_substrate(bc_interface_type_substrate_);

  // Robin coeff:
  CF_DIM* bc_interface_coeff_sub_[2] = {&bc_interface_coeff_inner, &bc_interface_coeff_inner};
  stefan_w_fluids_solver->set_bc_interface_robin_coeff_temp_substrate(bc_interface_coeff_sub_);

  // ------------
  // External heat source:
  // ------------
  CF_DIM* external_heat_source_[2] = {external_heat_source_T[LIQUID_DOMAIN], external_heat_source_T[SOLID_DOMAIN]};
  if(analytical_IC_BC_forcing_term) stefan_w_fluids_solver->set_user_provided_external_heat_source(external_heat_source_);


  // ------------------------------
  // Set NS BC's and forcing terms
  // ------------------------------
  // Velocity interface:
  // ------------
  // Velocity interface
  my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t* bc_interface_value_velocity_[P4EST_DIM];
  BoundaryConditionType* bc_interface_type_velocity_[P4EST_DIM];

  // Velocity wall
  CF_DIM* bc_wall_value_velocity_[P4EST_DIM];
  WallBCDIM* bc_wall_type_velocity_[P4EST_DIM];

  // External forces:
  CF_DIM* external_forces_NS[P4EST_DIM];

  foreach_dimension(d){
    // Vel interface
    bc_interface_value_velocity_[d] = bc_interface_value_velocity[d];
    bc_interface_type_velocity_[d] = &interface_bc_type_velocity[d];

    // Vel wall
    bc_wall_value_velocity_[d] = bc_wall_value_velocity[d];
    bc_wall_type_velocity_[d] = bc_wall_type_velocity[d];

    // Forcing terms:
    if(analytical_IC_BC_forcing_term){
      if(example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP){
        external_forces_NS[d] = external_force_components_with_BA[d];
      }
      else{
        external_forces_NS[d] = external_force_components[d];
      }
    }

  }
  // Velocity interface:
  stefan_w_fluids_solver->set_bc_interface_value_velocity(bc_interface_value_velocity_);
  stefan_w_fluids_solver->set_bc_interface_type_velocity(bc_interface_type_velocity_);

  // Velocity wall:
  stefan_w_fluids_solver->set_bc_wall_value_velocity(bc_wall_value_velocity_);
  stefan_w_fluids_solver->set_bc_wall_type_velocity(bc_wall_type_velocity_);

  // Pressure interface:
  stefan_w_fluids_solver->set_bc_interface_value_pressure(&bc_interface_value_pressure);
  stefan_w_fluids_solver->set_bc_interface_type_pressure(interface_bc_type_pressure);

  // Pressure wall:
  stefan_w_fluids_solver->set_bc_wall_value_pressure(&bc_wall_value_pressure);
  stefan_w_fluids_solver->set_bc_wall_type_pressure(&bc_wall_type_pressure);

  // External forces:
  if(analytical_IC_BC_forcing_term){
    stefan_w_fluids_solver->set_user_provided_external_force_NS(external_forces_NS);
  }
} // end of "initialize_all_relevant_bcs_ics_forcing_terms()"



void destroy_all_relevant_bcs_ics_forcing_terms(mpi_environment_t &mpi,
                                temperature_field* analytical_T[2], external_heat_source* external_heat_source_T[2],
                                BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2], BC_WALL_VALUE_TEMP* bc_wall_value_temp[2],
                                velocity_component* analytical_soln_v[P4EST_DIM],
                                external_force_per_unit_volume_component* external_force_components[P4EST_DIM],
                                external_force_per_unit_volume_component_with_boussinesq_approx* external_force_components_with_BA[P4EST_DIM],
                                BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM],
                                BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM],
                                                INITIAL_TEMP* T_init_cf[2], temperature_field* analytical_temp_init[2],
                                                INITIAL_VELOCITY* v_init_cf[P4EST_DIM], velocity_component* analytical_soln_v_init[P4EST_DIM])
{
  // Destroy all the bcs/ics/forcing terms we created
  if(solve_stefan){
    // Destroy relevant BC and RHS info:
    for(unsigned char d=0;d<2;++d){
      if(analytical_IC_BC_forcing_term){
        delete analytical_T[d];
        delete external_heat_source_T[d];
      }
      delete bc_interface_val_temp[d];
      delete bc_wall_value_temp[d];
    }
  }

  if(solve_navier_stokes){
    for(unsigned char d=0;d<P4EST_DIM;d++){
      if(analytical_IC_BC_forcing_term){
        delete analytical_soln_v[d];
        if (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP){
          delete external_force_components_with_BA[d];
        }
        else{
          delete external_force_components[d];
        }
      }

      delete bc_interface_value_velocity[d];
      delete bc_wall_value_velocity[d];
      delete bc_wall_type_velocity[d];
    }
  }

  // Destroy IC objects:
  // TO-DO: clean thisup

  if(solve_stefan){
    for(unsigned char d=0;d<2;++d){
      if(analytical_IC_BC_forcing_term){
        delete analytical_temp_init[d];
      }
      delete T_init_cf[d];
    }
  }

  if(solve_navier_stokes){
    for(unsigned char d=0;d<P4EST_DIM;++d){
      if(analytical_IC_BC_forcing_term){
        delete analytical_soln_v_init[d];
      }
      delete v_init_cf[d];
    }
  }
} // end of "destroy_all_relevant_bcs_ics_forcing_terms()"


// ---------------------------------
// End of auxiliary functions
// --------------------------------

// --------------------------------------------------------------------------------------------------------------
// Begin main operation:
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi_environment_t* mpi_p = &mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;

  // -----------------------------------------------
  // Parse the user inputs:
  // -----------------------------------------------
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc,argv);
  pl.get_all(cmd);
  pl.print_all();


  // -----------------------------------------------
  // Declare all needed variables:
  // -----------------------------------------------
  my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver;


  // p4est variables
  p4est_t*              p4est_n;
  p4est_nodes_t*        nodes_n;
  p4est_ghost_t*        ghost_n;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;
  my_p4est_hierarchy_t* hierarchy_n;
  my_p4est_node_neighbors_t* ngbd_n;

  p4est_t               *p4est_np1;
  p4est_nodes_t         *nodes_np1;
  p4est_ghost_t         *ghost_np1;
  my_p4est_hierarchy_t* hierarchy_np1;
  my_p4est_node_neighbors_t* ngbd_np1;

  splitting_criteria_cf_and_uniform_band_t* sp;

  // Level set function(s):---------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_nm1; // LSF for previous timestep... we must keep this so that hodge fields can be updated correctly in NS process

  vec_and_ptr_t phi_solid; // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_substrate;   // LSF for the inner substrate, if applicable (example ICE_OVER_CYLINDER)

  vec_and_ptr_t phi_eff; // Effective LSF used when we have a substrate -- this will be used by the extension, interfacial velocity computation, and navier stokes steps

  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_substrate_dd;

  my_p4est_level_set_t *ls;

  // Interface geometry:------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;
  vec_and_ptr_dim_t substrate_normals;
  vec_and_ptr_t island_numbers;

  // Temperature/concentration problem:---------------------------------
  vec_and_ptr_t T_l_n;
  vec_and_ptr_t T_l_nm1;
  vec_and_ptr_t T_l_backtrace;
  vec_and_ptr_t T_l_backtrace_nm1;
  vec_and_ptr_t rhs_Tl;

  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  // First derivatives of T_l_n and T_s_n
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;

  // Second derivatives of T_l
  vec_and_ptr_dim_t T_l_dd;

  // Stefan problem:------------------------------------
  vec_and_ptr_dim_t v_interface;;
  vec_and_ptr_dim_t jump;

  // Navier-Stokes problem:-----------------------------

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  // Poisson boundary conditions:
  //TO-DO: update the bcs with the one from the class
  temperature_field* analytical_T[2];
  external_heat_source* external_heat_source_T[2];

  BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2];
  BC_WALL_VALUE_TEMP* bc_wall_value_temp[2];

  // Initial conditions:
  INITIAL_TEMP* T_init_cf[2];
  temperature_field* analytical_temp_init[2];

  // Navier-Stokes boundary conditions: -----------------
  BoundaryConditions2D bc_velocity[P4EST_DIM];
  BoundaryConditions2D bc_pressure;

  // TO-DO: update this bc with the one from the class
  BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM];
  BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM];
  BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM];

  BC_INTERFACE_VALUE_PRESSURE bc_interface_value_pressure;
  BC_WALL_VALUE_PRESSURE bc_wall_value_pressure;
  BC_WALL_TYPE_PRESSURE bc_wall_type_pressure;

  external_force_per_unit_volume_component* external_force_components[P4EST_DIM];
  external_force_per_unit_volume_component_with_boussinesq_approx* external_force_components_with_BA[P4EST_DIM];

  CF_DIM* external_forces_NS[P4EST_DIM];

  // Coupled/NS boundary conditions:
  velocity_component* analytical_soln_v[P4EST_DIM];

  // External forcing terms:
  vec_and_ptr_t external_forces_Tl;
  vec_and_ptr_t external_forces_Ts;
  vec_and_ptr_dim_t external_forces_ns;

  // Initial conditions:
  INITIAL_VELOCITY* v_init_cf[P4EST_DIM];
  velocity_component* analytical_soln_v_init[P4EST_DIM];

  // Files for outputting relevant information : ---------
  FILE *fich_errors = NULL;
  char name_errors[1000];

  FILE *fich_data = NULL;
  char name_data[1000];

  FILE *fich_mem = NULL;
  char name_mem[1000];

  // stopwatch
  parStopWatch w;
  w.start("Running example: stefan_with_fluids");
  // -----------------------------------------------
  // Begin loop through number of grid splits:
  // -----------------------------------------------
  for(int grid_res_iter=0;grid_res_iter<=num_splits;grid_res_iter++){
    // Create the solver:
    stefan_w_fluids_solver = new my_p4est_stefan_with_fluids_t(mpi_p);

    // Set all the things needed:
    setup_initial_parameters_and_report(mpi, stefan_w_fluids_solver, grid_res_iter);

    // ------------------------------------------------------------
    // Initialize relevant boundary condition objects:
    // ------------------------------------------------------------
    if(print_checkpoints)PetscPrintf(mpi.comm(),"Initializing all BCs/ICs/forcing terms ... \n");

    initialize_all_relevant_bcs_ics_forcing_terms(stefan_w_fluids_solver,
                                                  analytical_T,
                                                  bc_interface_val_temp,
                                                  bc_wall_value_temp,
                                                  external_heat_source_T,
                                                  analytical_soln_v,
                                                  bc_interface_value_velocity,
                                                  bc_wall_value_velocity,
                                                  bc_wall_type_velocity,
                                                  bc_interface_value_pressure,
                                                  bc_wall_value_pressure,
                                                  bc_wall_type_pressure,
                                                  external_force_components,
                                                  external_force_components_with_BA,
                                                  T_init_cf, analytical_temp_init,
                                                  v_init_cf, analytical_soln_v_init);

    // -----------------------------------------------
    // Perform grid and field initializations
    // -----------------------------------------------
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning grid and field initializations ... \n");

    // Initialize output file numbering:
    int out_idx = -1;
    int state_idx = -1;

    int data_save_out_idx = -1;
    bool compute_pressure_ = false; // will be updated by the NS

    // Initialize the load step (in event we use load)
    int load_tstep=-1;
    int last_tstep=-1;

    double t_original_start = tstart;

    // Store desired CFL and hodge criteria -- we will relax these for several startup iterations, then use them
    double cfl_NS_steady = cfl_NS;
    double hodge_percentage_steady = hodge_percentage_of_max_u;

    handle_any_startup_t_dt_and_bc_cases(mpi, stefan_w_fluids_solver,
                                         cfl_NS_steady, hodge_percentage_steady);
    stefan_w_fluids_solver->perform_initializations();

    if(save_to_vtk){
      if(!loading_from_previous_state){
        out_idx=-1;
        stefan_w_fluids_solver->set_tstep(0);
        stefan_w_fluids_solver->save_fields_to_vtk(out_idx);
      }
      // if we are loading from a previous state, we want the tstep and outidx the be loaded/computed from the load state
    }

    // Initialize tn and tstep
    if(loading_from_previous_state){
      tn = stefan_w_fluids_solver->get_tn();
      tstep = stefan_w_fluids_solver->get_tstep();
      stefan_w_fluids_solver->save_fields_to_vtk(-1);

    }
    else{
      tn = tstart;
      tstep = 0;
    }

    // Restrict the timestep if we are saving every dt
    // TO-DO: saving every dt could probably be done more cleanly
    if(save_using_dt){
      PetscPrintf(mpi.comm(), "save_every_dt = %f  sec, time nondim 2 dim = %e  ", save_every_dt, time_nondim_to_dim);
      save_every_dt/=time_nondim_to_dim;
      PetscPrintf(mpi.comm(), "save_every_dt = %f  nondim ", save_every_dt);

      stefan_w_fluids_solver->set_dt_max_allowed(save_every_dt - EPS);
    }
    if(save_state_using_dt){
      save_state_every_dt/=time_nondim_to_dim;
    }

    // -----------------------------------------------
    // Initialize files to output various data of interest:
    // -----------------------------------------------
    if(print_checkpoints)PetscPrintf(mpi.comm(),"Initializing output files ... \n");

    initialize_error_files_for_test_cases(mpi, stefan_w_fluids_solver,
                                          fich_errors, name_errors,
                                          fich_data, name_data,
                                          fich_mem, name_mem);

    // to-do later: fix this if I want it
//    pl.generate_bash_file("$PWD", "multialloy_with_fluids", "generic_bash_file.sh");

    // ------------------------------------------------------------

    // ---------------------------------------
    // Begin time loop
    // ---------------------------------------
    bool keep_going = true;
    while(keep_going /*(tfinal - tn)>-EPS*/){
      if((timing_every_n>0) && (tstep%timing_every_n == 0)) {
        PetscPrintf(mpi.comm(),"Current time info : \n");
        w.read_duration_current();
      }

      // ---------------------------------------
      // Handle any modifications to cfl, dt, vint, or bcs related with "startup" conditions
      // ---------------------------------------
      handle_any_startup_t_dt_and_bc_cases(mpi, stefan_w_fluids_solver,
                                           cfl_NS_steady, hodge_percentage_steady);


      // ---------------------------------------
      // Print iteration information:
      // ---------------------------------------
      int num_nodes = stefan_w_fluids_solver->get_nodes_np1()->num_owned_indeps;
      MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM,mpi.comm());
      dt = stefan_w_fluids_solver->get_dt();
      ierr = PetscPrintf(mpi.comm(),"\n -------------------------------------------\n"
                                     "Iteration %d , Time: %0.3f [nondim] "
                                     "= Time: %0.3f [nondim] "
                                     "= %0.3f [sec] "
                                     "= %0.3f [min],"
                                     " Timestep: %0.3e [nondim] = %0.3e [sec],"
                                     " Percent Done : %0.2f %"
                                     " \n ------------------------------------------- \n"
                                     "Number of nodes : %d \n \n",
                         tstep,tn,tn,tn*time_nondim_to_dim,tn*(time_nondim_to_dim)/60.,
                         dt, dt*(time_nondim_to_dim),
                         ((tn-t_original_start)/(tfinal-t_original_start))*100.0,num_nodes);

      // check out the flushing situation:
      flush_idx = (int) floor(tn*time_nondim_to_dim / flush_every_dt);

      // We flush every given dt for a certain duration, but we ignore the first flush cycle to give some startup time (since by this formula, the first flush cycle would be at t=0)
      bool do_flush = ((tn*time_nondim_to_dim - flush_idx * flush_every_dt) < flush_duration) && (flush_idx > 0);

//      PetscPrintf(mpi.comm(), "Flushing info: \n"
//                              "tn = %0.3e, tn_dim = %0.3e, \n flush_every_dt = %0.3e, flush_dur = %0.3e, \n "
//                              "tnew = %0.3e, flush_idx = %d, do_flush = %d, theta_wall = %0.2f \n \n",
//                  tn, tn*time_nondim_to_dim,
//                  flush_every_dt, flush_duration,
//                  (tn*time_nondim_to_dim - flush_idx * flush_every_dt),
//                  flush_idx, do_flush, theta_wall_flushing_scenario() );


      // -------------------------------
      // Set up analytical ICs/BCs/forcing terms if needed
      // -------------------------------
      if(analytical_IC_BC_forcing_term) {
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up analytical ICs/BCs/forcing terms... \n");
        setup_analytical_ics_and_bcs_for_this_tstep(bc_interface_val_temp, bc_wall_value_temp,
                                                    analytical_T, external_heat_source_T,
                                                    analytical_soln_v, bc_interface_value_velocity, bc_wall_value_velocity,
                                                    bc_wall_value_pressure,
                                                    external_force_components, external_force_components_with_BA,
                                                    external_forces_NS);
      }

      // -------------------------------
      // Solve all the fields for one timestep
      // -------------------------------

      compute_pressure_ = are_we_saving_vtk(tstep, tn, tstep == load_tstep, out_idx, false ) || example_is_a_test_case;
      stefan_w_fluids_solver->set_compute_pressure(compute_pressure_);

      stefan_w_fluids_solver->solve_all_fields_for_one_timestep();

      dt = stefan_w_fluids_solver->get_dt();

      /*
       * The code below corresponds to Elyce trying to intentionally reproduce the nefarious "process_incoming_query" bug using the Two_Grains_Clogging setup. I'm leaving it here for now because I'm not entirely sure this is fixed for good.
       *
      // ELYCE TRYING TO TRIGGER A BUG:
      MPI_Barrier(mpi.comm());
      PetscPrintf(mpi.comm(), "\nBEGIN:: ELYCE TRYING TO TRIGGER THE BUG \n");


      ngbd_np1 = stefan_w_fluids_solver->get_ngbd_np1();
      nodes_np1 = stefan_w_fluids_solver->get_nodes_np1();
      p4est_np1 = stefan_w_fluids_solver->get_p4est_np1();

      // TEMPORARY: output the grid that we are seeing
      // -------------------------------------------------
      if(1){
        std::vector<Vec_for_vtk_export_t> point_fields;
        std::vector<Vec_for_vtk_export_t> cell_fields = {};
        phi = stefan_w_fluids_solver->get_phi();
        point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));


        const char* out_dir = getenv("OUT_DIR_VTK");
        if(!out_dir){
          throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
        }

        char filename[1000];
        sprintf(filename, "%s/snapshot_before_interp_bug_%d", out_dir, tstep);
        my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ngbd_np1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
        point_fields.clear();
      }

      my_p4est_interpolation_nodes_t interp_test(ngbd_np1);

      const static double  threshold_  = 0.01*(double)P4EST_QUADRANT_LEN(P4EST_MAXLEVEL) / (double) P4EST_ROOT_LEN;
      PetscPrintf(mpi.comm(), "Threshold is %0.12e \n", threshold_);

      double threshold = 0.000000000027939429;//2.794003e-11;//03125e-11;//3.5e-11;
      PetscPrintf(mpi.comm(), "Our defined threshold is %0.12e \n", threshold);
      interp_test.set_debugging_error_report(true);

      foreach_node(n, nodes_np1){
        double xyz_n[P4EST_DIM];
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz_n);

        double xloc = 3.0;
        double yloc = 1.171875;//1.96875;//1.99219;
        // working counterexample -- y loc = 1.03125;//


        bool is_xloc = fabs(xloc - xyz_n[0])<EPS;
        bool is_yloc = fabs(yloc - xyz_n[1])<EPS;

//        if(is_xloc){printf("We have xloc on rank %d \n", mpi.rank());}
//        if(is_yloc){printf("We have yloc on rank %d \n", mpi.rank());}
        if(is_xloc && is_yloc){
          printf("We've got our point registered by rank %d \n", mpi.rank());
          xyz_n[0]+=threshold;
          // triggers problem:
//          xyz_n[0]=3.000000000027939429;
//          xyz_n[0]=3.000000000027950;

          // try this instead:
//          xyz_n[0]=3.000000000028;
          interp_test.add_point(n, xyz_n);
        }
//        else if(is_xloc && !is_yloc){
//          interp_test.add_point(n, xyz_n);

//        }


      }
      T_l_n = stefan_w_fluids_solver->get_T_l_n();
      interp_test.set_input(T_l_n.vec, quadratic_non_oscillatory_continuous_v2);

      vec_and_ptr_t Tl_out;
      Tl_out.create(p4est_np1, nodes_np1);
      PetscPrintf(mpi.comm(), "Trying to interpolate ... \n");
      interp_test.set_debugging_error_report(false);

      MPI_Barrier(mpi.comm());

      interp_test.interpolate(Tl_out.vec);
      Tl_out.destroy();

      MPI_Barrier(mpi.comm());

      PetscPrintf(mpi.comm(), "END:: ELYCE TRYING TO TRIGGER THE BUG \n \n");
      if(tstep==0){
        std::exit(0);
      }

      */
      // -------------------------------
      // Save as relevant
      // -------------------------------

      perform_saving_tasks_of_interest(mpi, stefan_w_fluids_solver,
                                       out_idx, data_save_out_idx, compute_pressure_,
                                       load_tstep, last_tstep,
                                       fich_errors, name_errors,
                                       fich_data, name_data,
                                       fich_mem, name_mem);


      // -------------------------------
      // FOR examples with a known max vinterface analytical
      // --------------------------------
      if(example_has_known_max_vint){
          dt = stefan_w_fluids_solver->get_dt();
          double dxyz_smallest = stefan_w_fluids_solver->get_dxyz_smallest();
          double dt_ana = cfl*dxyz_smallest/max_vint_known_for_ex;
          stefan_w_fluids_solver->set_dt(MIN(dt, dt_ana));
      }


      // -------------------------------
      // FOR COUPLED CONVERGENCE TEST: Clip time and switch vel direction
      // for coupled problem examples:
      // --------------------------------
      bool examples_w_switch_sign = (example_ == COUPLED_PROBLEM_EXAMPLE)||
                                    (example_ == COUPLED_TEST_2) ||
                                    (example_ == COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP);
      if(examples_w_switch_sign){
        dt = stefan_w_fluids_solver->get_dt();
        if(((tn+dt) >= tfinal/2.0) && !vel_has_switched){
          if((tfinal/2. - tn)>dt_min_allowed){ // if we have some uneven situation
            PetscPrintf(mpi.comm(),"uneven situation \n");
            dt = (tfinal/2.) - tn;

            stefan_w_fluids_solver->set_dt(dt);
          }
          PetscPrintf(mpi.comm(),"SWITCH SIGN : %0.1f \n", coupled_test_sign);
          coupled_test_switch_sign();
          stefan_w_fluids_solver->set_scale_vgamma_by(coupled_test_sign);
          vel_has_switched=true;
          PetscPrintf(mpi.comm(),"SWITCH SIGN : %0.1f \n dt : %e \n", coupled_test_sign,dt);
        }
      }

      // -------------------------------
      // Clip the timestep if we are near the end of our simulation to get the proper end time
      // If the resulting timestep will be too small, we just allow to end naturally
      // --------------------------------
      dt = stefan_w_fluids_solver->get_dt();



      // Normal end:
      if((tn + dt > tfinal) && (last_tstep<0)){
        // This computed value below is what will get set in SWF
        dt = max(tfinal - tn, dt_min_allowed);

        // if time remaining is too small for one more step, end here. otherwise, do one more step and clip timestep to end on exact ending time
        if(fabs(dt)>dt_min_allowed){
          last_tstep = tstep+1;
        }
        else{
          last_tstep = tstep;
        }
        PetscPrintf(mpi.comm(),"Final tstep will be %d \n",last_tstep);
      }

      // account for phi subiter dt if we are doing that *and* it is currently activated in SWF
      // This gets checked after the "normal end" bc it's a bit less restrictive
      double dt_phi_subiter=0.;
      if(do_phi_advection_substeps && stefan_w_fluids_solver->get_do_phi_advection_substeps()){
        dt_phi_subiter = stefan_w_fluids_solver->get_dt_phi_advection_substep();
      }

      // if phi subiter dt is going to exceed final time, let's now deactivate for end of run
      if((tn+ dt + dt_phi_subiter) > tfinal && (last_tstep<0)){
        PetscPrintf(mpi.comm(), "Exceeds: dt_phi_subiter = %0.3e \n", dt_phi_subiter);

        stefan_w_fluids_solver->set_dt_phi_advection_substep(tfinal - dt);
        last_tstep = tstep+1;
        PetscPrintf(mpi.comm(),"Final tstep will be %d \n",last_tstep);

      }

      PetscPrintf(mpi.comm(), "\n \n tn = %0.3e, dt = %0.3e, dt_phi = %0.3e \n tfinal = %0.3e,  tn + dt + dt_phi = %0.3e \n Last tstep = %d \n",
                  tn, dt, dt_phi_subiter, tfinal, tn + dt+ dt_phi_subiter, last_tstep);

//      if(do_phi_advection_substeps && (tn + dt + dt_phi_subiter > tfinal) && (last_tstep<0)){
//        // if time remaining is too small for one more step, end here. otherwise, do one more step and clip timestep to end on exact ending time
//        if(fabs(tfinal-(tn + dt + dt_phi_subiter))>dt_min_allowed){
//          last_tstep = tstep+1;
//        }
//        else{
//          last_tstep = tstep;
//        }
//        PetscPrintf(mpi.comm(),"Final tstep will be %d \n",last_tstep);
//      }



      // -------------------------------
      // Update the grid if this is not the last timestep
      // -------------------------------
      // INSERT HERE: perform_lsf_advection_grid_update_and_interp_of_fields
      if(tstep!=last_tstep){
        // If we are on the last timestep and tn==tfinal (more or less), we will skip this last step
        stefan_w_fluids_solver->perform_lsf_advection_grid_update_and_interp_of_fields();

        // Get the current time (in case it was updated by doing phi subiterations)
        tn = stefan_w_fluids_solver->get_tn();

        // Get the number of substeps:
        if(do_phi_advection_substeps){
//          int substeps = stefan_w_fluids_solver->get_num_phi_advection_substeps();
//          PetscPrintf(mpi.comm(), "Number of LSF advection substeps: %d \n", phi_advection_substeps_coeff);
          double dt_phi_adv = stefan_w_fluids_solver->get_dt_phi_advection_substep();

          if(save_using_dt && (dt_phi_adv > save_every_dt)){
            PetscPrintf(mpi.comm(), "WARNING: tstep taken by phi advection substeps is larger than save_every_dt. VTK output may be not as predicted. \n");

          }
        }
      }

      // -------------------------------
      // Update time:
      // -------------------------------
//      keep_going = ((tfinal - tn + dt) > -EPS) && (tstep!=last_tstep);
      keep_going = (tfinal - tn > -EPS) &&
                   (last_tstep > 0 ? (tstep<last_tstep):true);

      tn+=dt;
      // Increment if we are doing the phi advection substep, and if it is actually activated inside SWF
      if(do_phi_advection_substeps && stefan_w_fluids_solver->get_do_phi_advection_substeps()){
        tn+=dt_phi_subiter;
      }
      tstep++;
      stefan_w_fluids_solver->set_tstep(tstep); // Required for reinit_every_iter to be processed properly
      stefan_w_fluids_solver->set_tn(tn); // Update tn in case of save state (so save state has accurate time)


      // --------------------------------------------------------------------------------------------------------------
      // Save simulation state every specified number of iterations
      // We save the state here to be consistent with how states are loaded -- a state that is loaded will be loaded as if its an initial condition, and then the temperature problem will be solved, and so on and so forth.
      // Thus we save the state here, since the natural next step would be as described above
      // --------------------------------------------------------------------------------------------------------------

//      bool do_we_save_state = tstep>0 &&
//                              ((tstep%save_state_every_iter)==0) &&
//                              tstep!=load_tstep &&
//                              (last_tstep>0 ? (tstep<last_tstep): true);

      bool do_we_save_state = are_we_saving_simulation_state(tstep, tn, ((tstep==0) || (tstep==load_tstep)), state_idx);
      PetscPrintf(mpi.comm(), "Do we save: %d , state_idx = %d \n", do_we_save_state, state_idx);
      // the last condition above is bc last_tstep is initialized to be negative, except when we are getting close to it, so we can
      // only compare tstep to last_tstep if last_tstep has been updated to some non-negative value

      if(do_we_save_state){
        char output[1000];
        const char* out_dir_save_state = getenv("OUT_DIR_SAVE_STATE");
        if(!out_dir_save_state){
          throw std::invalid_argument("You need to set the output directory for save states: OUT_DIR_SAVE_STATE");
        }
        PetscPrintf(mpi.comm(),"Beginning save state process... \n");
        sprintf(output,
                "%s/save_states_output_lmin_%d_lmax_%d_advection_order_%d_example_%d",
                out_dir_save_state,
                stefan_w_fluids_solver->get_lmin(), stefan_w_fluids_solver->get_lmax(),
                advection_sl_order,example_);
        PetscPrintf(mpi.comm(), "Output directory is %s \n", output);

        stefan_w_fluids_solver->save_state(output, num_save_states);

        PetscPrintf(mpi.comm(),"Simulation state was saved . \n");
      }


    } // <-- End of for loop through time

  PetscPrintf(mpi.comm(),"Time loop exited \n");

  // Do the final destructions!
  destroy_all_relevant_bcs_ics_forcing_terms(mpi,
                                             analytical_T, external_heat_source_T,
                                             bc_interface_val_temp, bc_wall_value_temp,
                                             analytical_soln_v,
                                             external_force_components, external_force_components_with_BA,
                                             bc_interface_value_velocity, bc_wall_value_velocity,
                                             bc_wall_type_velocity,
                                             T_init_cf, analytical_temp_init,
                                             v_init_cf, analytical_soln_v_init);

  delete stefan_w_fluids_solver;

  }// end of loop through number of splits

  w.stop(); w.read_duration();
  return 0;
}
