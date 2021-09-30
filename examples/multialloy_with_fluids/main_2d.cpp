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
  MELTING_POROUS_MEDIA = 8,
  PLANE_POIS_FLOW=9,
  DISSOLVING_DISK_BENCHMARK=10
};

enum{LIQUID_DOMAIN=0, SOLID_DOMAIN=1};

// ---------------------------------------
// Example/application options:
// ---------------------------------------
DEFINE_PARAMETER(pl, int, example_, 4,"example number: \n"
                                   "0 - Frank Sphere (Stefan only) \n"
                                   "1 - NS Gibou example (Navier Stokes only) \n"
                                   "2 - Additional coupled verification test (not fully verified) \n"
                                   "3 - Coupled problem example for verification \n"
                                   "4 - Ice solidifying around a cooled cylinder \n"
                                   "5 - Flow past a cylinder (Navier Stokes only)\n"
                                   "6 - dendrite solidification test (WIP) \n"
                                   "7 - melting of an ice sphere \n"
                                   "8 - melting of a porous media (with fluid flow) \n"
                                   "9 - plane poiseuille flow \n "
                                   "10 - dissolving disk benchmark for dissolution problem \n"
                                   "default: 4");

// ---------------------------------------
// Save options:
// ---------------------------------------
// Options for saving to vtk:
DEFINE_PARAMETER(pl, bool, save_to_vtk, true, "We save vtk files using a given dt increment if this is set to true \n");
DEFINE_PARAMETER(pl, bool, save_using_dt, false, "We save vtk files using a given dt increment if this is set to true \n");
DEFINE_PARAMETER(pl, bool, save_using_iter, false, "We save every prescribed number of iterations if this is set to true \n");

DEFINE_PARAMETER(pl, int, save_every_iter, 1, "Saves vtk every n number of iterations (default is 1)");
DEFINE_PARAMETER(pl, double, save_every_dt, 1, "Saves vtk every dt amount of time in seconds of dimensional time (default is 1)");

// Options to compute and save fluid forces to a file:
DEFINE_PARAMETER(pl, bool, save_fluid_forces, false, "Saves fluid forces if true (default: false) \n");
DEFINE_PARAMETER(pl, double, save_fluid_forces_every_dt, 0.01, "Saves fluid forces every dt amount of time in seconds of dimensional time (default is 1.0) \n");

// Save state options
DEFINE_PARAMETER(pl, int, save_state_every_iter, 10000, "Saves simulation state every n number of iterations (default is 500)");
DEFINE_PARAMETER(pl, int, num_save_states, 20, "Number of save states we keep on file (default is 20)");

// Load state options
DEFINE_PARAMETER(pl ,bool, loading_from_previous_state, false,"Loads simulation from previous state if marked true");

// ---------------------------------------
// Debugging options:
// ---------------------------------------
// Options for checking memory usage: -- this was more heavily used when I was investigating a memory leak . TO-DO: clean this stuff up ?
DEFINE_PARAMETER(pl, int, check_mem_every_iter, -1, "Checks memory usage every n number of iterations (default is -1 aka, don't check. To check, set to a positive integer value)");
DEFINE_PARAMETER(pl, double, mem_safety_limit, 60.e9, "Memory upper limit before closing the program -- in bytes");

// Options for checking timing:
DEFINE_PARAMETER(pl, int, timing_every_n, -1, "Print timing info every n iterations (default -1 aka no use, to use this feature, set to a positive integer value)");

// Options for debugging: -- TO-DO: can remove all these now?
DEFINE_PARAMETER(pl, bool, print_checkpoints, false, "Print checkpoints throughout script for debugging? ");

// ---------------------------------------
// Solution options:
// ---------------------------------------
// Related to which physics we solve:
DEFINE_PARAMETER(pl, bool, solve_stefan, false, "Solve stefan ?");
DEFINE_PARAMETER(pl, bool, solve_navier_stokes, false, "Solve navier stokes?");
DEFINE_PARAMETER(pl, bool, solve_coupled, true, "Solve the coupled problem?");
DEFINE_PARAMETER(pl, bool, do_advection, true, "Boolean flag whether or not to do advection (default : 1)");

// Related to LSF reinitialization:
DEFINE_PARAMETER(pl, int, reinit_every_iter, 1, "An integer option for how many iterations we wait before reinitializing the LSF in the case of a coupled problem (only implemented for coupled problem!). Default : 1. This can be helpful when the timescales governing interface evolution are much larger than those governing the flow. For example, if vint is 1000x smaller than vNS, we may want to reinitialize only every 1000 timesteps, or etc. This can prevent the interface from degrading via frequent reinitializations relative to the amount of movement it experiences. ");
// Related to the Stefan and temperature/concentration problem:
DEFINE_PARAMETER(pl, double, cfl, 0.5, "CFL number for Stefan problem (default:0.5)");
DEFINE_PARAMETER(pl, int, advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid temperature field(default:2)");
DEFINE_PARAMETER(pl, bool, force_interfacial_velocity_to_zero, false, "Force the interfacial velocity to zero? ");

// Related to the Navier-Stokes problem:
DEFINE_PARAMETER(pl, double, Re_overwrite, -100.0, "Overwrite the examples set Reynolds number (works if set to a positive number, default:-100.00");
DEFINE_PARAMETER(pl, int, NS_advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid velocity fields (default:1)");
DEFINE_PARAMETER(pl, double, cfl_NS, 1.0, "CFL number for Navier-Stokes problem (default:1.0)");
DEFINE_PARAMETER(pl, double, hodge_tolerance, 1.e-3, "Tolerance on hodge for error convergence (default:1.e-3)");

// Specifying flow or no flow: TO-DO: clean this up, make it usable for dendrite example too
DEFINE_PARAMETER(pl, bool, no_flow, false, "An override switch for the ice cylinder example to run a case with no flow (default: false)");

// Related to simulation duration settings:
DEFINE_PARAMETER(pl, double, duration_overwrite, -100.0, "Overwrite the duration in minutes (works if set to a positive number, default:-100.0");
DEFINE_PARAMETER(pl, double, duration_overwrite_nondim, -10.,"Overwrite the duration in nondimensional time (in nondimensional time) -- not fully implemented");
// ---------------------------------------

bool analytical_IC_BC_forcing_term;
bool example_is_a_test_case;

bool interfacial_temp_bc_requires_curvature;
bool interfacial_temp_bc_requires_normal;

bool interfacial_vel_bc_requires_vint;

bool example_uses_inner_LSF;
bool example_requires_area_computation;

bool do_we_solve_for_Ts;

void select_solvers(){
  switch(example_){
    case FRANK_SPHERE:
      solve_stefan = true;
      solve_navier_stokes = false;
      do_advection = false;
      break;

    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
      if(!no_flow){
        solve_stefan = true;
        solve_navier_stokes = true;
        do_advection=true;
      }
      else{
        solve_stefan=true;
        solve_navier_stokes=false;
        do_advection=false;
      }
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
      solve_stefan = true;
      solve_navier_stokes = true;
      break;
    case DENDRITE_TEST: // will need to select solvers manually // TO-DO: just use the "no-flow" option for this!
      break;
    case MELTING_POROUS_MEDIA:
      solve_stefan=true;
      solve_navier_stokes=true;
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


    // Define other settings to be used depending on the example:
    analytical_IC_BC_forcing_term = (example_ == COUPLED_PROBLEM_EXAMPLE) ||
                                    (example_ == COUPLED_TEST_2) ||
                                    (example_ == NS_GIBOU_EXAMPLE); // whether or not we need to create analytical bc terms

    example_is_a_test_case = (example_ == COUPLED_PROBLEM_EXAMPLE) ||
                             (example_ == COUPLED_TEST_2) ||
                             (example_ == FRANK_SPHERE) ||
                             (example_ == NS_GIBOU_EXAMPLE);

    interfacial_temp_bc_requires_curvature = (example_ == ICE_AROUND_CYLINDER) ||
                                             (example_ == MELTING_ICE_SPHERE) ||
                                             (example_ == DENDRITE_TEST) ||
                                             (example_ == MELTING_POROUS_MEDIA);
    interfacial_temp_bc_requires_normal = (example_ == DENDRITE_TEST);

    interfacial_vel_bc_requires_vint = (example_ == ICE_AROUND_CYLINDER) ||
                                       (example_ == MELTING_ICE_SPHERE) ||
                                       (example_ == DENDRITE_TEST)||
                                       (example_ == MELTING_POROUS_MEDIA) ||
                                       (example_ == DISSOLVING_DISK_BENCHMARK);

    example_uses_inner_LSF = (example_ == ICE_AROUND_CYLINDER);

    example_requires_area_computation = (example_ == ICE_AROUND_CYLINDER) ||
                                        (example_ == MELTING_ICE_SPHERE) || (example_ == DISSOLVING_DISK_BENCHMARK);

    do_we_solve_for_Ts = (example_ != DISSOLVING_DISK_BENCHMARK);

}
// ---------------------------------------
// Refinement options:
// ---------------------------------------

DEFINE_PARAMETER(pl, double, vorticity_threshold, 0.1,"Threshold to refine vorticity by, default is 0.1 \n");
DEFINE_PARAMETER(pl, double, gradT_threshold, 1.e-4,"Threshold to refine the nondimensionalized temperature gradient by \n (default: 0.99)");
DEFINE_PARAMETER(pl, bool, use_uniform_band, true, "Boolean whether or not to refine using a uniform band");
DEFINE_PARAMETER(pl, double, uniform_band, 8., "Uniform band (default:8.)");

// ---------------------------------------
// Geometry and grid refinement options:
// ---------------------------------------
// General options: // TO-DO: maybe all these geometry options should be OVERWRITE options --aka, defaults are specified per example unless user states otherwise
DEFINE_PARAMETER(pl, double, xmin, 0., "Minimum dimension in x (default: 0)");
DEFINE_PARAMETER(pl, double, xmax, 1., "Maximum dimension in x (default: 0)");

DEFINE_PARAMETER(pl, double, ymin, 0., "Minimum dimension in y (default: 0)");
DEFINE_PARAMETER(pl, double, ymax, 1., "Maximum dimension in y (default: 1)");

DEFINE_PARAMETER(pl, int, nx, 1, "Number of trees in x (default:1)");
DEFINE_PARAMETER(pl, int, ny, 1, "Number of trees in y (default:1)");

DEFINE_PARAMETER(pl, int, px, 0, "Periodicity in x (default false)");
DEFINE_PARAMETER(pl, int, py, 0, "Periodicity in y (default false)");

DEFINE_PARAMETER(pl, int, lmin, 3, "Minimum level of refinement");
DEFINE_PARAMETER(pl, int, lint, 0, "Intermediate level of refinement (default: 0, won't be used unless set)");
DEFINE_PARAMETER(pl, int, lmax, 8, "Maximum level of refinement");
DEFINE_PARAMETER(pl, double, lip, 1.75, "Lipschitz coefficient");
DEFINE_PARAMETER(pl, int, method_, 1, "Solver in time for solid domain, and for fluid if no advection. 1 - Backward Euler, 2 - Crank Nicholson");
DEFINE_PARAMETER(pl, int, num_splits, 0, "Number of splits -- used for convergence tests");
DEFINE_PARAMETER(pl, bool, refine_by_ucomponent, false, "Flag for whether or not to refine by a backflow condition for the fluid velocity");
DEFINE_PARAMETER(pl, bool, refine_by_d2T, true, "Flag for whether or not to refine by the nondimensionalized temperature gradient");

// For level set:
double r0;

// For frank sphere:
double s0;
double T_inf;

// For ice growth on cylinder and melting ice sphere problems: // TO-DO: double check that this is correct
DEFINE_PARAMETER(pl, double, d_cyl, 35.e-3, "Cylinder diamter in meters for ice cylinder problem, (default: 35.e-3) ");
DEFINE_PARAMETER(pl, double, T_cyl, 263., "For ice growth over cooled cylinder example, this refers to Temperature of cooled cylinder in K (default : 263, aka -10 C). For the melting ice sphere example, this refers to the initial temperature of the ice in K (default: 263). ");

DEFINE_PARAMETER(pl, double, Twall, 275.5, "The freestream fluid temperature T_infty. (default: 275.5 K, or 2.5 C)");

double r_cyl; // non dim variable used to set up LSF: set in set_geometry()
double d_seed; // for dendrite test : set in TO-DO:update this

// For solution of temperature fields: set in set_physical_properties() TO-DO: maybe make these actual things the user can specify
// TO-DO: ideal scenario -- the user *could* change all these things via inputs. Might need to make an overwrite boolean. aka bool_overwrite_default_temp_settings
//double Twall;
double Tinterface;
double back_wall_temp_flux;
double deltaT;

// Nondimensional temperature values (computed in set_physical_properties)
double theta_wall;
double theta_interface;
double theta_cyl;

// For surface tension: (used to apply some interfacial BC's in temperature) // TO-DO revisit this?
double sigma; // set in set_physical_properties()
//DEFINE_PARAMETER(pl,double,sigma,4.20e-10,"Interfacial tension [m] between ice and water, default: 2*2.10e-10");

// For the coupled test case where we have to swtich sign:
double coupled_test_sign;
bool vel_has_switched;
void coupled_test_switch_sign(){coupled_test_sign*=-1.;}

// for defining LSF for coupled test case
double x0_lsf;
double y0_lsf;

unsigned int num_fields_interp = 0;
double d0; // for dendrite w convection case


// Define a few parameters for the porous media case to create random grains:
DEFINE_PARAMETER(pl, int, num_grains, 10., "Number of grains in porous media (default: 10)");

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

      // Necessary boundary condition info:
      Twall = -0.2;    T_inf = Twall;

      Tinterface = 0.0;

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
    case MELTING_POROUS_MEDIA:{
      // (WIP)
      // MORE COMPLEX:
//      // Domain size:
//      xmin = 0.0; xmax = 20.0;
//      ymin = 0.0; ymax = 10.0;

//      // Number of trees:
//      nx =10.0;
//      ny =5.0;

      // EASIER TO DO LOCALLY:
      xmin = 0.0; xmax = 10.0;
      ymin = 0.0; ymax = 10.0;

      // Number of trees:
      nx = 5.0;
      ny = 5.0;

      // Periodicity:
      px = 0;
      py = 0;

      // Problem geometry:
      r0 = 0.5;     // Computational radius of the sphere
      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      // Domain size:
      xmin = 0.0; xmax = 5.0; // should technically be 33.0, but that would yield weird cell aspect ratios
      ymin = 0.0; ymax = 2.5;

      // It is 1.9 because the char length scale is 20 mm, and the physical length is 38 mm, so
      // it takes 1.9 nondim length units to get the domain size

      // Number of trees:
      nx = 4.0;
      ny = 2.0;

      // Periodicity:
      px = 0;
      py = 0;

      // Problem geometry:
      r0 = 0.5; // radius of the disk ( in comp domain)
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

      x0_lsf = 0.; y0_lsf = 0.; // TO-DO: can remove the x0_lsf and y0_lsf since they are not being used

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
      xmin = 0.; xmax = 10.;
      ymin = 0.; ymax = 10.;

      // Number of trees and periodicity:
      nx = 2; ny = 2;
      px = 1; py = 0;

      // level set size (initial seed size)
      r0 = 0.5; // This needs to be set to 0.5 in order for it to properly correspond to our dimensional diameter. (aka nondim diameter should always equal 1)


      // capillary length scale and etc is set in set_physical_properties() 5/3/21
      break;
    }
  }

  // Set number of interpolation fields:
  // Number of fields interpolated from one grid to the next depends on which equations
  // we are solving, therefore we select appropriately
  num_fields_interp = 0;
  if(solve_stefan){
    num_fields_interp+=3; // Tl, vint_x, vint_y
    if(do_we_solve_for_Ts) num_fields_interp+=1; // Ts
  }
  if(solve_navier_stokes){
    num_fields_interp+=2; // vNS_x, vNS_y
  }

  // If you're only solving NS, switch off refinement around temp fields:
  if(!solve_stefan && solve_navier_stokes){
    refine_by_d2T = false;
  }
}


  void make_LSF_for_porous_media(mpi_environment_t &mpi){
    // initialize random number generator:
    srand(1);
    // Resize the vectors on all processes appropriately so we all have consistent sizing (for broadcast later)
    xshifts.resize(num_grains);
    yshifts.resize(num_grains);
    rvals.resize(num_grains);

    if(mpi.rank() == 0){

      std::ifstream infile_x("geometry_files/xshifts.txt");
      // Read x data:
      if(infile_x){
        double curr_val;

        int count = 0;
        while(infile_x >> curr_val){
          xshifts[count] = curr_val;
          count+=1;
        }
      }
      std::ifstream infile_y("geometry_files/yshifts.txt");

      // Read y data:
      if(infile_y){
        double curr_val;
        int count = 0;
        while(infile_y >> curr_val){
          yshifts[count] = curr_val;
          count+=1;
        }
      }
      std::ifstream infile_r("geometry_files/rvals.txt");

      // Read r data:
      if(infile_r){
        double curr_val;

        int count=0;
        while(infile_r >> curr_val){
         rvals[count]=curr_val;
         count+=1;

        }
      }

      /*

      // Initialize the arrays:
      xshifts[0] = 0.; yshifts[0] = 0.; rvals[0] = 0.;

      // First, generate the desired number of grains, with size and location:
      for(int n=0; n<num_grains; n++){
        // Initialize the values:
        xshifts[n] = 0.; yshifts[n] = 0.; rvals[n] = 0.;


        // We will generate random grains and check if they are eligible, aka:
        // (a) is it far enough away from other existing grains?
        // (b) is it far enough away from the wall?
        bool is_this_eligible = false;
        double x_; double y_; double r_;
        while(!is_this_eligible){
          x_ = ((double)rand()/RAND_MAX)*xmax + xmin;
          y_ = ((double)rand()/RAND_MAX)*ymax + ymin;
          r_ = (((double)rand()/RAND_MAX)*2 + 0.5)*r0; // r0*(1-3)

          is_this_eligible=true;
          for(int i=0; i<=n; i++){
            // Check current distance from walls and existing grains:
            double dist_c2c = sqrt(SQR(xshifts[i] - x_) + SQR(yshifts[i] - y_));

            // Check distance to the wall:
            double dist_c2wall[4] = {fabs(x_ - xmax),
                                     fabs(x_ - xmin),
                                     fabs(y_ - ymax),
                                     fabs(y_ - ymin)};
            // grab whatever the closest distance to a wall is:
            double min_dwall=100.;
            for(int k=0; k<4; k++){
              min_dwall = min(min_dwall, dist_c2wall[k]);
            }

            // we are far enough from the wall if dist from grain edge to wall is larger than minimum pore size
            bool is_far_enough_from_wall = min_dwall - r_ > min_pore_size;

            // Update the is_this_eligible bool:
            is_this_eligible = (dist_c2c >= min_pore_size*r0 + r_ + rvals[i]) &&
                               is_far_enough_from_wall &&
                               is_this_eligible;

            // If current config not eligible, break the checking for loop and try again:
            if(!is_this_eligible) break;

//            printf("n = %d, i = %d, dist_c2c = %0.2f, dist_e2e = %0.2f, min_dwall = %0.2f, is_far_from_wall = %s, is_this_eligible = %s \n", n, i, dist_c2c, dist_c2c - r_ - rvals[i],min_dwall, is_far_enough_from_wall? "True": "False", is_this_eligible? "True": "False");

          }
        }
        xshifts[n] = x_;
        yshifts[n] = y_;
        rvals[n] = r_;
      }
*/
      printf("end of operation on rank 0 \n");
    } // end of defining the grains on rank 0, now we need to broadcast the result to everyone

    // Tell everyone else what we came up with!
    int mpi_err;
    mpi_err = MPI_Bcast(xshifts.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);
    mpi_err = MPI_Bcast(yshifts.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);
    mpi_err = MPI_Bcast(rvals.data(), num_grains, MPI_DOUBLE, 0, mpi.comm()); SC_CHECK_MPI(mpi_err);



}
double v_interface_max_norm; // For keeping track of the interfacial velocity maximum norm


// ---------------------------------------
// Non dimensional groups:
// ---------------------------------------
DEFINE_PARAMETER(pl, double, Re, 300., "Reynolds number (rho Uinf d)/mu, where d is the characteristic length scale - default is 300 \n");
DEFINE_PARAMETER(pl, double, Pr, 0., "Prandtl number - computed from mu_l, alpha_l, rho_l \n");
DEFINE_PARAMETER(pl, double, Pe, 0., "Peclet number - computed from Re and Pr \n");
DEFINE_PARAMETER(pl, double, St, 0., "Stefan number (cp_s deltaT/L)- computed from cp_s, deltaT, L \n");

DEFINE_PARAMETER(pl, double, Da_init, 0.001, "Initial Darcy number -- just used to impose initial flow condition as nicely as possible for the solver at startup. Used for the porous media example \n");
DEFINE_PARAMETER(pl, double, porosity_init, 1.0, "Initial porosity -- used to impose the initial flow condition as nicely as possible for the solver at startup. Used for the porous media example \n");


DEFINE_PARAMETER(pl,double,Gibbs_eps4,0.005,"Gibbs Thomson anisotropy coefficient (default: 0.005), applicable in dendrite test cases \n");
// ---------------------------------------
// Physical properties:
// ---------------------------------------
// For solidification problem:
double alpha_s; // Thermal diffusivity of solid [m^2/s]
double alpha_l; // Thermal diffusivity of liquid [m^2/s]
double k_s;     // Thermal conductivity of solid [W/(mK)]
double k_l;     // Thermal conductivity of liquid [W/(mK)]
double L;       // Latent heat of fusion [J/kg]
double rho_l;   // Density of fluid [kg/m^3]
double rho_s;   // Density of solid [kg/m^3]
double cp_s;    // Specific heat of solid [J/(kg K)]
double mu_l;    // Dynamic viscosity of fluid [Pa s]


// For dissolution problem:
DEFINE_PARAMETER(pl, double, gamma_diss, 3.69e-4, "The parameter dictates some dissolution behavior, default value is 0.00288 (corresponds to pure gypsum)");

DEFINE_PARAMETER(pl, double , D_diss, 1.0e-9, "Dissolution diffusion coefficient m^2/s, default is : 9e-4 mm2/s = 9e-10 m2/s ");
DEFINE_PARAMETER(pl, double, l_diss, 2.0e-4, "Dissolution length scale. The physical length (in m) that corresponds to a length of 1 in the computational domain. Default: 20e-3 m (20 mm), since the initial diameter of the disk is 20 mm \n");
DEFINE_PARAMETER(pl, double, k_diss, 8.9125e-4/*4.5e-6*/, "Dissolution rate constant per unit area of reactive surface (m/s). Default 4.5e-3 mm/s aka 4.5e-6 m/s \n");


double n_times_d0; // multiplier on d0 we use to get dseed //(WIP) // TO-DO:clean up dendrite stuff!
void set_physical_properties(){
  double nu;
  switch(example_){
    case FRANK_SPHERE:{
      alpha_s = 1.0;
      alpha_l = 1.0;
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      rho_s = 1.0;
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

      // Boundary condition info:
      //Twall = 273. + 2.5;    // Physical wall temp [K] (aka T_infty)
      Tinterface = 273.15; // Physical interface temp [K]

      back_wall_temp_flux = 0.0; // Flux in temp on back wall (non dim) (?) TO-DO: check this

      deltaT = Twall - T_cyl; // Characteristic Delta T [K] -- used for some non dimensionalization

      theta_wall = 1.0; // Non dim temp at wall
      theta_cyl = 0.0; // Non dim temp at cylinder

      theta_interface = (Tinterface - T_cyl)/(deltaT); // Non dim temp at interface

      break;
      }
    case MELTING_POROUS_MEDIA: // TO-DO: intentionally waterfalling for now, will change once i fine tune the example more
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

      // Temperature settings:
      Tinterface = 273.15;

      back_wall_temp_flux = 0.0; // Flux in temp on back wall

      deltaT = Twall - T_cyl; // Characteristic Delta T [K] -- used for some non dimensionalization

      if(deltaT>0.){
        theta_wall = 1.0;
      }
      else{
        theta_wall = -1.0;
      }
      //ttheta_wall = 1.0; // Non dim temp at wall
      theta_cyl = 0.0; // Non dim temp at cylinder

      theta_interface = (Tinterface - T_cyl)/(deltaT); // Non dim temp at interface

      // In this example, T_cyl corresponds to the initial temperature of the ice
      // T_cyl is used to define the nondimensionalization, and applied as an initial condition.
      // However, this is the only time it is used for this example.
      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      mu_l = 1.0e-3; // Pa s // back calculated using experimental Re and u0 reported
      rho_l = 1000.0;
      rho_s = 2710.0;

      theta_wall = 1.0; // wall undersaturation
      theta_cyl = 0.0; // aka fully saturated at the disk

      back_wall_temp_flux = 0.0;
      // No need to set theta_interface --> we have a robin BC there, not Dirichlet
      break;
    }
    case PLANE_POIS_FLOW:{
      rho_l = 1.0;
      mu_l = 1.0;
      Re = 1.0;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:{
      alpha_s = 1.0;
      alpha_l = 1.0;
      k_s = 1.;
      k_l = 1.;
      L = 1.;
      rho_l = 1.;
      rho_s = 1.0;
      mu_l = 1.0;
      break;
    }
    case NS_GIBOU_EXAMPLE:{
      rho_l = 1.0;
      mu_l = 1.0;
      Re = 1.0;
      break;
    }
    case DENDRITE_TEST:{
      cp_s = 1913.; // Specific heat of solid, [J/kgK]

      //alpha_l = 1.12e-7;
      alpha_s = 1.16e-7;

      alpha_l = alpha_s;

      //k_l = 0.223; // [W /(m K)]
      k_s = 0.225;

      k_l = k_s;
      L = 4.7e4; // [J/kg]

      //rho_l = 988.; // [kg/m^3]
      rho_s = k_s/alpha_s/cp_s; // [kg/m^3]

      rho_l = rho_s;
      nu = 2.6e-6;
      mu_l = nu*rho_l; // [Pa s]

      sigma = 1.92e-10;

      printf("rho_l = %0.3e, rho_s = %0.3e, sigma = %0.3e \n",rho_l,rho_s,sigma);
      // BC info:
      // Tinterface = 331.23;
      // Twall = deltaT*(L/cp_s) + Tinterface;
      // deltaT = Twall-Tinterface;

      deltaT = -0.55;
      theta_wall=0.;
      theta_interface = 1.0;

      Tinterface = 331.23;
      Twall = Tinterface - deltaT;

      // Set length scales:
      // Updated 5/3/21:
      double Tmelt = 331.23;
      double gamma = 8.9e-3;
      d0 = (Tmelt * gamma * cp_s)/(rho_s * SQR(L));
      n_times_d0 = 500.0;
      d_seed = n_times_d0 * d0; // like Al-Rawahi paper 2002
      // physical property paper ""

      break;
    }

    } // end of switch cases
}

//-----------------------------------------
// Properties to set if you are solving NS
// ----------------------------------------
double pressure_prescribed_flux;
double pressure_prescribed_value;
double u0;
double v0;

double Re_u; // reynolds number in x direction
double Re_v; // reynolds number in y direction

double outflow_u;
double outflow_v;
double hodge_percentage_of_max_u;
int hodge_max_it = 100;
double T_l_IC_band = 2.0;
bool ramp_T_l_IC_space = false;
double dt_NS;

double hodge_global_error;

double NS_norm = 0.0; // To keep track of the NS norm
double perturb_flow_noise =0.25;

double u_inf; // physical value of freestream velocity

double G_press; // corresponds to porous media example, it is the prescribed pressure gradient across the channel, applied as a pressure drop, aka (P1 - P2)/L = G --> specified
void set_NS_info(){
  pressure_prescribed_flux = 0.0; // For the Neumann condition on the two x walls and lower y wall
  pressure_prescribed_value = 0.0; // For the Dirichlet condition on the back y wall

  dt_NS = 1.e-3; // initial dt for NS

  // Note: fluid velocity is set via Re and u0,v0 --> v0 = 0 is equivalent to single direction flow, u0=1, v0=1 means both directions will flow at Re (TO-DO:make this more clear)
  switch(example_){
    case FRANK_SPHERE:throw std::invalid_argument("NS isnt setup for this example");
    case MELTING_POROUS_MEDIA:{
      u0 = 1.0;
      v0 = 0.;

      G_press = Re*8./SQR(ymax - ymin);

      hodge_percentage_of_max_u = 1.e-3;

      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      u0 = 1.0;
      v0 = 0.;

      hodge_percentage_of_max_u=1.e-3;
      break;
    }

    case FLOW_PAST_CYLINDER:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:{
      Re = 201.;
      u0 = 1.0; // computational freestream velocity
      v0 = 0.0;
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case PLANE_POIS_FLOW:{
      Re = 1.0;
      u0 = 5.0625;
      v0 = 0.;
      hodge_percentage_of_max_u=1.0e-3;
      break;
    }
    case NS_GIBOU_EXAMPLE:{
      Re = 1.0;

      u0 = 1.0;
      v0 = 1.0;

      u_inf=1.0; // u_inf usually corresponds to a physical value for velocity, but this example doesnt have that //to-do: is this actually used in this example?
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:{
      Re = 1.0;
      u0 = 1.0;
      v0 = 1.0;
      hodge_percentage_of_max_u = 1.e-3;
      break;
    }
    case DENDRITE_TEST:{
      Re = 0.15;//.01507; // 10
      u0 = 0.;
      v0 = -1.;
      hodge_percentage_of_max_u = 1.e-2;
      break;
    }
  }
  outflow_u = 0.0;
  outflow_v = 0.0;
}


// For selecting the appropriate nondimensionalized formulation of the problem:
enum:int{NONDIM_NO_FLUID,NONDIM_YES_FLUID,DIMENSIONAL, NONDIM_DISSOLUTION};
// TO-DO: change this to nondim_no_fluid_freezemelt, nondim_yes_fluid_freezemelt, nondim_no_fluid_dissodepo, nondim_yes_fluid_dissodepo, etc. erosion tbd
int stefan_condition_type;
int select_stefan_formulation(){
  if(solve_navier_stokes){
    if(example_ == DISSOLVING_DISK_BENCHMARK){
      return NONDIM_DISSOLUTION;
    }
    else{
      return NONDIM_YES_FLUID;
    }

  }
  else{
    return NONDIM_NO_FLUID;
  }
};

// For defining appropriate nondimensional groups:
double time_nondim_to_dim;
double vel_nondim_to_dim;

// TO-DO: clean up how nondim groups are set
// TO-DO: make checklist of things to change in main when adding a new example
void set_nondimensional_groups(){
   if(stefan_condition_type==NONDIM_YES_FLUID){
     double d_length_scale = 1.; // set it as 1 if not one of the following examples:
     if(example_ == ICE_AROUND_CYLINDER ||
         example_ == FLOW_PAST_CYLINDER ||
         example_ == MELTING_ICE_SPHERE ||
         example_ == MELTING_POROUS_MEDIA){
       d_length_scale=d_cyl;
     }
     else if (example_ == DENDRITE_TEST){
       d_length_scale = d_seed;
       printf("sigma/d_seed = %0.4e \n",sigma/d_seed);
     }
     if(Re_overwrite>0.) Re = Re_overwrite;

     Pr = mu_l/(alpha_l*rho_l);
     Pe = Re*Pr;

     St = cp_s*fabs(deltaT)/L;

     u_inf= Re*mu_l/rho_l/d_length_scale;
     vel_nondim_to_dim = u_inf;
     time_nondim_to_dim = d_length_scale/u_inf;

   }
   else if(stefan_condition_type==NONDIM_NO_FLUID){
     double d_length_scale = 0.;
     if(example_ == ICE_AROUND_CYLINDER){
       d_length_scale=d_cyl;
     }
     else if (example_ == DENDRITE_TEST){
       d_length_scale = d_seed;
     }
     else if(example_ == FRANK_SPHERE){
       d_length_scale = 1.0;
     }

     Pr = mu_l/(alpha_l*rho_l);
     St = cp_s*fabs(deltaT)/L;
     Re = 0.; Pe = 0.;

     time_nondim_to_dim = SQR(d_length_scale)/alpha_s;
     vel_nondim_to_dim = (alpha_s)/(d_length_scale);
   }
   else if(stefan_condition_type==NONDIM_DISSOLUTION){
     // Assuming Re is the input parameter
     // Calculate u_inf:
     // For this case, have to do it a bit differently, since diameter is not the length scale we use

     u_inf = (Re*mu_l)/(rho_l * (l_diss)); // freestream velocity is based on reynolds defined around sample diameter, even tho sample diameter is not the char length scale here
     Pe = u_inf*l_diss/D_diss;
     double Da = k_diss*l_diss/D_diss;
     printf("Pe = %0.4f, Da = %0.4f, 1/Pe = %0.4f, Da/Pe = %0.4f, D = %0.4e , k = %0.4e\n", Pe, Da, 1./Pe, Da/Pe, D_diss, k_diss);

     vel_nondim_to_dim = u_inf;
     time_nondim_to_dim= l_diss/(u_inf/**gamma_diss*/);


   }
   else{
     time_nondim_to_dim = 1.;
   };
   if((example_ == COUPLED_TEST_2) || (example_ == COUPLED_PROBLEM_EXAMPLE) || (example_ == NS_GIBOU_EXAMPLE) || (example_ == FRANK_SPHERE) ){
     St = 1.0;
   }
}

// ---------------------------------------
// For setting up simulation timing information:
// ---------------------------------------
double tfinal;
double dt_max_allowed;
bool keep_going = true;

double tn;
double tstart;
double dt = 1.e-5;
double dt_nm1 = 1.e-5;
int tstep;
double dt_min_allowed = 1.e-5;

double dt_Stefan;

DEFINE_PARAMETER(pl,double,t_ramp,0.1,"Time at which boundary conditions are ramped up to their desired value [input should be dimensional time, in seconds] (default: 3 seconds) \n");
DEFINE_PARAMETER(pl,bool,ramp_bcs,false,"Boolean option to ramp the BCs over a specified ramp time (default: false) \n");
DEFINE_PARAMETER(pl,int,startup_iterations,-1,"Number of startup iterations to do before entering real time loop, used for verification tests to allow v_interface and NS fields to stabilize. Default:-1, to use this, set number to positive integer value.");
DEFINE_PARAMETER(pl,double,startup_nondim_time,-10.0,"Startup time in nondimesional time, before the simulation allows interfacial growth to occur (Default : 0)");
DEFINE_PARAMETER(pl,double,startup_dim_time,-10.0,"Startup time in dimesional time (seconds), before the simulation allows interfacial growth to occur (Default : 0)");
DEFINE_PARAMETER(pl,bool,perturb_initial_flow,false,"Perturb initial flow? For melting refinement case. Default: true. Applies to initial condition for velocity field. ");

void simulation_time_info(){
  t_ramp /= time_nondim_to_dim; // divide input in seconds by time_nondim_to_dim because we are going from dim--> nondim
  save_every_dt/=time_nondim_to_dim; // convert save_every_dt input (in seconds) to nondimensional time
  switch(example_){
    case FRANK_SPHERE:{
      tfinal = 1.10;
      dt_max_allowed = 0.05;
      tstart = 1.0;
      break;
    }
    case MELTING_POROUS_MEDIA:
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER: {
      // ice solidifying around isothermally cooled cylinder
      tfinal = (40.*60.)/(time_nondim_to_dim); // 40 minutes
      if(save_every_dt>0.){
        dt_max_allowed = save_every_dt - EPS;
      }
      else{
        dt_max_allowed = 0.1;
      }

      tstart = 0.0;
      break;
    }
    case MELTING_ICE_SPHERE:{
      //tfinal = (2.*60)/(time_nondim_to_dim); // 2 minutes
      tfinal = 35.0; // 1000 in nondim time for refinement test
      //dt_max_allowed = 0.9*save_every_dt;
      dt_max_allowed = save_every_dt - EPS;
      tstart = 0.0;

      break;
    }
    case DISSOLVING_DISK_BENCHMARK:{
      tstart = 0.0;
      if(save_every_dt>0.){
        dt_max_allowed = save_every_dt - EPS;
      }
      else{
        dt_max_allowed = 10.0;
      }
      dt = 1.0e-3; // initial timestep
      break;
    }

    case NS_GIBOU_EXAMPLE:{
      tfinal = PI/3.;
      dt_max_allowed = 1.e-2;
      tstart = 0.0;
      break;
    }
    case PLANE_POIS_FLOW:{
      tfinal = 100.;
      dt_max_allowed = 1.e-2;
      tstart=0.0;
      break;
    }

    case COUPLED_PROBLEM_EXAMPLE:{
      tfinal = PI/3.;//PI/2.;
      dt_max_allowed = 1.0e-1;
      tstart = 0.0;
      break;
    }
    case COUPLED_TEST_2:{
      tfinal = 0.75;//1.;
      dt_max_allowed=1.0e-1;
      tstart=0.0;
      break;
    }
    case DENDRITE_TEST:{
      //double tau =5.46e-11;// 3.2e-7;
      //double tf_tau = 2.e4;
      //tfinal =(tau*tf_tau)/(time_nondim_to_dim);//(1./10.)*(50.*tau)/(time_nondim_to_dim); // 1.2 microseconds


      // Modifications (11-24-20):
      //double tau = 3.12e-7;
      //tfinal = (30.*tau)/(time_nondim_to_dim);

      tfinal = (1.e6*SQR(d0)/alpha_s)/time_nondim_to_dim;
      tstart=0.;
      dt_max_allowed = tfinal/(100);
      save_using_iter = 1;
      save_every_iter = 1;
      //save_every_dt = tfinal/100.;
      break;
    }

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
double v_int_max_allowed = 50.0;
// Variables used for advection:
double advection_alpha_coeff= 0.0;
double advection_beta_coeff =0.0;

bool is_ice_melted = false; // Boolean for checking if the ice is melted for melting ice sphere example

// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// Frank sphere functions -- Functions necessary for evaluating the analytical solution of the Frank sphere problem, to validate results for example 1
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
  else      return T_inf*(1.0 - F(s)/F(s0));


}
// --------------------------------------------------------------------------------------------------------------
// Functions/Structures for validating the Navier-Stokes problem:
// --------------------------------------------------------------------------------------------------------------
// Re-doing the NS validation case:
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

struct pressure_field: CF_DIM{
public:
  double operator()(DIM(double x,double y, double z)) const {
    switch(example_){
      case COUPLED_PROBLEM_EXAMPLE:{
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
//------------------------------------------------------------------------
// Functions/Structures for validating the Stefan problem:
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

// --------------------------------------------------------------------------------------------------------------
// Level set functions:
// --------------------------------------------------------------------------------------------------------------

struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (example_){
      case FRANK_SPHERE:
        return s0 - sqrt(SQR(x) + SQR(y));
      case FLOW_PAST_CYLINDER:
      case MELTING_POROUS_MEDIA:{
        double lsf_vals[num_grains];
        // First, grab all the relevant LSF values for each grain:
        for(int n=0; n<num_grains; n++){
          double r = sqrt(SQR(x - xshifts[n]) + SQR(y - yshifts[n]));
          lsf_vals[n] = rvals[n] - r;
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
      case DISSOLVING_DISK_BENCHMARK: {
        return r0 - sqrt(SQR(x-(xmax/2.)) + SQR(y - (ymax/2.)));
      }
      case MELTING_ICE_SPHERE:{
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      }
      case ICE_AROUND_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case NS_GIBOU_EXAMPLE:
        return r0 - sin(x)*sin(y);
      case COUPLED_PROBLEM_EXAMPLE:
        return r0 - sqrt(SQR(x - x0_lsf) + SQR(y - y0_lsf));
      case COUPLED_TEST_2:{
        double x0 = 0.;//1./6.;
        double y0 = -1./4.;
        double R = 1./3.;//1./4.;
        double rval = sqrt(SQR(x-x0) + SQR(y-y0));
        return -1.*(rval - (R)  - (pow(y - y0,5.) + 5.*(pow(x-x0,4.))*(y-y0) - 10.*SQR(x-x0)*pow(y-y0,3.))/(3.*pow(rval,5.) + 1e-5));
      }
      case DENDRITE_TEST:{
        double noise = 0.3;
        double xc =xmax/2.0;
        double yc =ymax/2.0;
        double theta = atan2(y-yc,x-xc);
        return r0*(1.0 - noise*fabs(sin(theta)) - noise*fabs(cos(theta))) - sqrt(SQR(x - xc) + SQR(y - yc));
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
    }
  }
} level_set;


// This one is for the inner cylinder in example 2
struct MINI_LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER: return r_cyl - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case FRANK_SPHERE:
      case MELTING_ICE_SPHERE:
      case MELTING_POROUS_MEDIA:
      case DISSOLVING_DISK_BENCHMARK:
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:
      case DENDRITE_TEST:
      case NS_GIBOU_EXAMPLE: throw std::invalid_argument("This option may not be used for the particular example being called");
      }
  }
} mini_level_set;
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
// Interfacial temperature boundary condition objects/functions:
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_temp;
void interface_bc_temp(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE:
      interface_bc_type_temp = DIRICHLET;
      break;
    case DENDRITE_TEST:
    case FLOW_PAST_CYLINDER:
    case MELTING_POROUS_MEDIA:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_temp = DIRICHLET; 
      break;
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:
      interface_bc_type_temp = DIRICHLET;
      break;
    case DISSOLVING_DISK_BENCHMARK:
      interface_bc_type_temp = ROBIN;
      break;
    }
}

BoundaryConditionType inner_interface_bc_type_temp;
void inner_interface_bc_temp(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case MELTING_POROUS_MEDIA:
    case FLOW_PAST_CYLINDER:
    case COUPLED_PROBLEM_EXAMPLE:
    case COUPLED_TEST_2:
    case DENDRITE_TEST:
    case MELTING_ICE_SPHERE:
    case FRANK_SPHERE:{
        throw std::invalid_argument("This option may not be used for the particular example being called");
    }
    case ICE_AROUND_CYLINDER:
      inner_interface_bc_type_temp = DIRICHLET;
      break;
    }
}

bool print_stuff; // TO-DO: remove this?
class BC_INTERFACE_VALUE_TEMP: public CF_DIM{ // TO CHECK -- changed how interp is initialized
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_node_neighbors_t* ngbd;
  my_p4est_interpolation_nodes_t* kappa_interp;
  temperature_field** temperature_;
  unsigned const char dom;

  // For normals (in dendritic case)
  my_p4est_interpolation_nodes_t* nx_interp;
  my_p4est_interpolation_nodes_t* ny_interp;



  std::vector<double> theta0 = {0., PI/2.,-PI,-PI/2.};

public:

  BC_INTERFACE_VALUE_TEMP(my_p4est_node_neighbors_t *ngbd_=NULL,Vec kappa = NULL, temperature_field** analytical_T=NULL, unsigned const char& dom_=NULL): ngbd(ngbd_),temperature_(analytical_T),dom(dom_)
  {
    if(ngbd!=NULL){
      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd);
      kappa_interp->set_input(kappa,linear);
    }
  }
  double Gibbs_Thomson(double sigma_, double T0, double dval, DIM(double x, double y, double z)) const {
        return (theta_interface - (sigma_/dval)*((*kappa_interp)(x,y))*(theta_interface + T0/deltaT)); // corrected on 9/5/2020 Saturday, double checked 10/26/20 Monday
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
    case FRANK_SPHERE:{ // Frank sphere case, no surface tension
        return Tinterface; // TO-DO : CHANGE THIS TO ANALYTICAL SOLN
      }
    case DENDRITE_TEST:{

        // OLD: TO FIX: first of all, i'm using atan2 incorrectly... so there's that
        //double theta = atan2((*nx_interp)(x,y),(*ny_interp)(x,y));

        // Trying something:
        //double theta = atan2(y-yc,x-xc); // not sure why exactly i was using normals instead of just angle based on location, may need to revisit
        double theta = atan2((*ny_interp)(x,y),(*nx_interp)(x,y));


//        double As = 0.75;
//        double m = 4.0;
//        double theta0 = 0.;
//        double sigma_ = sigma*(1 - As*cos(m*(theta - theta0)));


        double min_theta = 2*PI;
        unsigned long min_idx = 10;
        for(unsigned long i=0;i<4;i++){
          if(fabs(theta - theta0[i])<min_theta){
            min_theta = fabs(theta - theta0[i]);
            min_idx = i; // grab index corresponding to the closest axis of preferred crystal growth direction
          }
        }

        P4EST_ASSERT(min_idx<4);
        double sigma_ =
            //sigma*(1 + Gibbs_eps4*cos(4.*(theta - theta0[min_idx])));
            //sigma*(1. - 15.*Gibbs_eps4*cos(4.*(theta - theta0[min_idx])));
            sigma*(1. + Gibbs_eps4*cos(4.*theta));


        print_stuff = false;
//        if(fabs(fabs(theta)-PI/4)<0.1 || fabs(fabs(theta)-PI/2)<0.1){
//          print_stuff=true;
//          printf("sigma : %0.4e, sigma new: %0.4e, x = %0.2f ,y = %0.2f ,theta = %0.2f, theta_min = %0.2f, theta_diff = %0.2f, kappa = %0.2f ---> ",sigma,sigma_,x,y,theta*180./PI,theta0[min_idx]*180./PI,(theta - theta0[min_idx])*180./PI,(*kappa_interp)(x,y));}

        //double sigma_ = sigma;
        return Gibbs_Thomson(sigma_, 0. , d_seed, DIM(x,y,z));
//        return theta_interface;
      }
    case MELTING_POROUS_MEDIA:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER: {
        double interface_val = Gibbs_Thomson(sigma,T_cyl,d_cyl,DIM(x,y,z));

        // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
        if(ramp_bcs){
          return ramp_BC(theta_wall,interface_val);
        }
        else return interface_val;
    }
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:{
      return temperature_[dom]->T(DIM(x,y,z));
    }
    case DISSOLVING_DISK_BENCHMARK:{
//      printf("INTERFACE VAL CALLED \n");
      //return 0.0; // RHS is zero for the robin condition
      return 0.0;
    }
    default:
      throw std::runtime_error("BC INTERFACE VALUE TEMP: unrecognized example \n");
      }
  }
  void clear(){
    kappa_interp->clear();
  };
  void set(my_p4est_node_neighbors_t *ngbd_,Vec kappa){
    if(ngbd_!=NULL){
      ngbd = ngbd_;
      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd);
      kappa_interp->set_input(kappa,linear);
    }
  }
  void set_normals(my_p4est_node_neighbors_t *ngbd_,Vec nx, Vec ny){
    if(ngbd_!=NULL){
      ngbd = ngbd_;
      nx_interp = new my_p4est_interpolation_nodes_t(ngbd);
      nx_interp->set_input(nx,linear);

      ny_interp = new my_p4est_interpolation_nodes_t(ngbd);
      ny_interp->set_input(ny,linear);
    }
  }
  void clear_normals(){
    nx_interp->clear();
    ny_interp->clear();
  }
};

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const
  {

    switch(example_){
      case FRANK_SPHERE:
      case DENDRITE_TEST:
      case MELTING_POROUS_MEDIA:
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER:
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:
        return 1.0;
      case DISSOLVING_DISK_BENCHMARK:
        double Da = k_diss*l_diss/D_diss;
        return Da/Pe;//(k_diss*l_diss/D_diss);//(k_diss/u_inf); // Coefficient in front of C
      }
  }
}bc_interface_coeff;

class BC_interface_value_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER:
        if(ramp_bcs){
            return ramp_BC(theta_wall,theta_cyl);
          }
        else return theta_cyl;
      }
  }
}bc_interface_val_inner;

class BC_interface_coeff_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER:
        return 1.0;
      }
  }
}bc_interface_coeff_inner;

// --------------------------------------------------------------------------------------------------------------
// Wall functions -- these evaluate to true or false depending on if the location is on the wall --  they just add coding simplicity for wall boundary conditions
// --------------------------------------------------------------------------------------------------------------
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

bool dirichlet_temperature_walls(DIM(double x, double y, double z)){
  switch(example_){
    case FRANK_SPHERE:{
      return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
     }
    case DENDRITE_TEST:{
      // Dirichlet on both x walls, and y upper wall (where bulk flow is incoming)
      return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
    }
    case MELTING_POROUS_MEDIA:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:{
      return (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)));
    }
    case DISSOLVING_DISK_BENCHMARK:{
      return xlower_wall(DIM(x,y,z));
    }
    case COUPLED_PROBLEM_EXAMPLE:{
      return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
    }
    case COUPLED_TEST_2:{
      return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || xlower_wall(DIM(x,y,z)));
    }
  }
};

bool dirichlet_velocity_walls(DIM(double x, double y, double z)){
  switch(example_){
    case DENDRITE_TEST:{
      return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
    }
    case FLOW_PAST_CYLINDER:
    case MELTING_POROUS_MEDIA:{
      return (ylower_wall(DIM(x,y,z)) || (yupper_wall(DIM(x,y,z))));
    }
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
// Wall temperature boundary condition objects/functions:
// --------------------------------------------------------------------------------------------------------------
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
      case MELTING_POROUS_MEDIA:
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER:{
        if(dirichlet_temperature_walls(DIM(x,y,z))){
          return theta_wall;
        }
        else{
          return back_wall_temp_flux;
        }
      }
      case DISSOLVING_DISK_BENCHMARK:{
        if(dirichlet_temperature_walls(DIM(x,y,z))){
          return theta_wall;
        }
        else{
          return back_wall_temp_flux;
        }
        break;
      }
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
          return theta_wall;
        }
        else{
          return theta_interface;
        }
      }
      case MELTING_POROUS_MEDIA:
      case MELTING_ICE_SPHERE:{
        switch(dom){
          case LIQUID_DOMAIN:{
            return theta_wall;
          }
          case SOLID_DOMAIN:{
            return theta_cyl; // coolest temp
          }
          default:{
            throw std::runtime_error("Initial condition for temperature: unrecognized domain \n");
          }
        }
      }
      case ICE_AROUND_CYLINDER:{ // TO-DO: is this best initial condition? might be missing on serious initial interface growth...
        switch(dom){
          case LIQUID_DOMAIN:{
            return theta_wall;
          }
          case SOLID_DOMAIN:{
            if(ramp_bcs){
              return theta_wall;
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
        return theta_wall;
      }
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:{
          return temperature_[dom]->T(DIM(x,y,z));
      }
    }
  }
};

// --------------------------------------------------------------------------------------------------------------
// Wall fluid velocity boundary condition objects/functions: for fluid velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
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
      case DENDRITE_TEST:
      case FLOW_PAST_CYLINDER:
      case MELTING_POROUS_MEDIA:{
        if(dirichlet_velocity_walls(DIM(x,y,z))){
          return 0.0; // no slip at walls
        }
        else{
          return 0.0; // homogeneous neumann
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

// --------------------------------------------------------------------------------------------------------------
// Interfacial fluid velocity condition objects/functions: for fluid velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------

BoundaryConditionType interface_bc_type_velocity[P4EST_DIM];
void BC_INTERFACE_TYPE_VELOCITY(const unsigned char& dir){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case PLANE_POIS_FLOW:
    case FLOW_PAST_CYLINDER:
    case DENDRITE_TEST:
    case DISSOLVING_DISK_BENCHMARK:
    case MELTING_POROUS_MEDIA:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:{
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
      }
    }
}

// Interfacial condition:
class BC_interface_value_velocity: public CF_DIM{
private:
  my_p4est_node_neighbors_t* ngbd;
  my_p4est_interpolation_nodes_t* v_interface_interp;

public:
  const unsigned char dir;
  velocity_component** velocity_field;
  BC_interface_value_velocity(const unsigned char& dir_, my_p4est_node_neighbors_t* ngbd_=NULL, Vec v_interface=NULL,velocity_component** analyical_soln=NULL): ngbd(ngbd_),dir(dir_),velocity_field(analyical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
    if((ngbd_!=NULL) && (v_interface!=NULL)){
      v_interface_interp = new my_p4est_interpolation_nodes_t(ngbd);
      v_interface_interp->set_input(v_interface,linear);
    }
  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case PLANE_POIS_FLOW:
        return 0.; // homogeneous dirichlet no slip
      case FLOW_PAST_CYLINDER:
      case DENDRITE_TEST:
      case MELTING_POROUS_MEDIA:
      case MELTING_ICE_SPHERE:
      case DISSOLVING_DISK_BENCHMARK:
        if(!solve_stefan) return 0.;
        else{
          // 9/23/21 -- currently evalulating effect of diff BC's on result
          return (*v_interface_interp)(x,y); // no slip condition
          //return (*v_interface_interp)(x,y)*(1. - (rho_s/rho_l)); // cons of mass condition

        }
      case ICE_AROUND_CYLINDER:{ // Ice solidifying around a cylinder
        if(!solve_stefan) return 0.;
        else{
          return (*v_interface_interp)(x,y)*(1. - (rho_s/rho_l)); // Condition derived from mass balance across interface

        }
      }
//      case DISSOLVING_DISK_BENCHMARK:{
//        return 0.0;
//      }
      case NS_GIBOU_EXAMPLE:
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:
        return (*velocity_field[dir])(x,y);
    default:
      throw std::runtime_error("BC INTERFACE VALUE VELOCITY: unrecognized example ");
      }
  }
  void clear(){
    v_interface_interp->clear();
  }
  void set(my_p4est_node_neighbors_t* ngbd_, Vec v_interface){
    if((ngbd_!=NULL) && (v_interface!=NULL)){
      ngbd = ngbd_;
      v_interface_interp = new my_p4est_interpolation_nodes_t(ngbd);
      v_interface_interp->set_input(v_interface,linear);
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
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:
      case MELTING_POROUS_MEDIA: {
//        return 0.0;
        double h_ = ymax - ymin;
        double U = G_press/(2.* Re);
        double Uy = (Da_init/porosity_init) *U*(y)*(h_ - y);

        double lsf_dist = 0.1;

        if(fabs(level_set.operator()(DIM(x,y,z)))<lsf_dist){
          return (Uy/lsf_dist)*(-1.*level_set.operator()(DIM(x,y,z)));
        }
        else{
          switch(dir){
          case dir::x:{
            return Uy;}
          case dir::y:
            return 0.;
          default:
              throw std::runtime_error("Unrecognized direction for velocity initial condition \n");
          }
        }

      }
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
// Wall fluid pressure boundary condition objects/functions:
// --------------------------------------------------------------------------------------------------------------
class BC_WALL_TYPE_PRESSURE: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    if(!dirichlet_velocity_walls(DIM(x,y,z))){ // pressure is dirichlet where vel is neumann
        return DIRICHLET;
      }
    else{
        return NEUMANN;
      }
  }
};

class BC_WALL_VALUE_PRESSURE: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not "
                                    "compatible with this example, please choose another \n");
      case DENDRITE_TEST:
      case FLOW_PAST_CYLINDER:
      case MELTING_POROUS_MEDIA:{
        if(!dirichlet_velocity_walls(DIM(x,y,z))){
          // Specifying a pressure drop!
          if(xupper_wall(DIM(x,y,z))){
            return 0.0;
          }
          else if (xlower_wall(DIM(x,y,z))) {
            double L_ = xmax - xmin; // domain length

            double P1 = G_press * L_;
            return P1;

          }
          else{
            throw std::runtime_error("You are trying to specify a dirichlet pressure BC on a wall not set up properly, example: porous media melting \n");
          }

        } // end of Dirichlet pressure case
        else{
          return 0.0; // homogeneous Neumann for other walls
        }
      }
      case DISSOLVING_DISK_BENCHMARK:
        return 0.0; // returns homogeneous condition either way
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER:{ // coupled problem
        return 0.0;
      }

      case NS_GIBOU_EXAMPLE:
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
            return (xmax - xmin); // start with delta P/L = 1 -- where L is length of domain
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


// --------------------------------------------------------------------------------------------------------------
// Interfacial fluid pressure boundary condition objects/functions:
// --------------------------------------------------------------------------------------------------------------
static BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case NS_GIBOU_EXAMPLE:
    case COUPLED_TEST_2:
    case COUPLED_PROBLEM_EXAMPLE:
    case DENDRITE_TEST:
    case PLANE_POIS_FLOW:
    case FLOW_PAST_CYLINDER:
    case MELTING_POROUS_MEDIA:
    case DISSOLVING_DISK_BENCHMARK:
    case MELTING_ICE_SPHERE:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_pressure = NEUMANN;
      break;
    }
}

class BC_INTERFACE_VALUE_PRESSURE: public CF_DIM{
public:
  double operator()(DIM(double x, double y,double z)) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case DENDRITE_TEST:
      case FLOW_PAST_CYLINDER:
      case MELTING_POROUS_MEDIA:
      case DISSOLVING_DISK_BENCHMARK:
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder
        return 0.0;

      case PLANE_POIS_FLOW:
      case NS_GIBOU_EXAMPLE: // Benchmark NS
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      default:
        throw std::runtime_error("INTERFACE BC VAL PRESSURE: unrecognized example \n");
      }
  }
};

// --------------------------------------------------------------------------------------------------------------
// Auxiliary functions for solving the problem!:
// --------------------------------------------------------------------------------------------------------------
void setup_rhs(vec_and_ptr_t phi,vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts,vec_and_ptr_t T_l_backtrace, vec_and_ptr_t T_l_backtrace_nm1, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd, external_heat_source** external_heat_source_term=NULL){

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

  if(analytical_IC_BC_forcing_term){
    forcing_term_liquid.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,*external_heat_source_term[LIQUID_DOMAIN],forcing_term_liquid.vec);

    if(do_we_solve_for_Ts) {
      forcing_term_solid.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,*external_heat_source_term[SOLID_DOMAIN],forcing_term_solid.vec);
    }
  }

  // Get derivatives of temperature fields if we are using Crank Nicholson:
  vec_and_ptr_dim_t T_l_dd;
  vec_and_ptr_dim_t T_s_dd;
  if(method_ ==2){
    if(do_we_solve_for_Ts){
      T_s_dd.create(p4est,nodes);
      ngbd->second_derivatives_central(T_s.vec,T_s_dd.vec[0],T_s_dd.vec[1]);
      T_s_dd.get_array();
    }
    if(!do_advection) {
        T_l_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1]);
        T_l_dd.get_array();
      }
    }

  // Prep coefficients if we are doing 2nd order advection:
  if(do_advection && advection_sl_order==2){
      advection_alpha_coeff = (2.*dt + dt_nm1)/(dt + dt_nm1);
      advection_beta_coeff = (-1.*dt)/(dt + dt_nm1);
    }
  // Get Ts arrays:
  if(do_we_solve_for_Ts){
    T_s.get_array();
    rhs_Ts.get_array();
  }

  // Get Tl arrays:
  rhs_Tl.get_array();
  if(do_advection){
      T_l_backtrace.get_array();
      if(advection_sl_order ==2) T_l_backtrace_nm1.get_array();
    }
  else{
      T_l.get_array();
    }

  if(analytical_IC_BC_forcing_term){
    forcing_term_liquid.get_array();
    if(do_we_solve_for_Ts) forcing_term_solid.get_array();
  }

  phi.get_array();
  foreach_local_node(n,nodes){
    if(do_we_solve_for_Ts){
      // First, assemble system for Ts depending on case:
      if(method_ == 2){ // Crank Nicholson
        rhs_Ts.ptr[n] = 2.*T_s.ptr[n]/dt + alpha_s*(T_s_dd.ptr[0][n] + T_s_dd.ptr[1][n]);
      }
      else{ // Backward Euler
        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;
      }
    }

    // Now for Tl depending on case:
    if(do_advection){
      if(advection_sl_order ==2){
        rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]*((advection_alpha_coeff/dt) - (advection_beta_coeff/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1);
        }
      else{
        rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]/dt;
        }
     }
    else{
      if(method_ ==2){//Crank Nicholson
        rhs_Tl.ptr[n] = 2.*T_l.ptr[n]/dt + alpha_l*(T_l_dd.ptr[0][n] + T_l_dd.ptr[1][n]);
        }
      else{ // Backward Euler
        rhs_Tl.ptr[n] = T_l.ptr[n]/dt;
        }
      }
    if(analytical_IC_BC_forcing_term){
      // Add forcing terms:
      rhs_Tl.ptr[n]+=forcing_term_liquid.ptr[n];
      if(do_we_solve_for_Ts) rhs_Ts.ptr[n]+=forcing_term_solid.ptr[n];
    }

  }// end of loop over nodes

  // Restore arrays:
  phi.restore_array();

  if(do_we_solve_for_Ts){
    T_s.restore_array();
    rhs_Ts.restore_array();
  }

  rhs_Tl.restore_array();
  if(do_advection){
      T_l_backtrace.restore_array();
      if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
    }
  else{
      T_l.restore_array();
    }
  if(method_ ==2){
    if(do_we_solve_for_Ts){
      T_s_dd.restore_array();
      T_s_dd.destroy();
    }
    if(!do_advection){
        T_l_dd.restore_array();
        T_l_dd.destroy();
      }
  }

  if(analytical_IC_BC_forcing_term){
    forcing_term_liquid.restore_array();

    if(do_we_solve_for_Ts) {
      forcing_term_solid.restore_array(); forcing_term_solid.destroy();
    }

    // Destroy these if they were created
    forcing_term_liquid.destroy();

  }
}

void do_backtrace(vec_and_ptr_t T_l,vec_and_ptr_t T_l_nm1,
                  vec_and_ptr_t T_l_backtrace,vec_and_ptr_t T_l_backtrace_nm1,
                  vec_and_ptr_dim_t v, vec_and_ptr_dim_t v_nm1,
                  p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd,
                  p4est_t* p4est_nm1, p4est_nodes_t* nodes_nm1, my_p4est_node_neighbors_t* ngbd_nm1){
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Beginning to do backtrace \n");
  PetscErrorCode ierr;
  // Initialize objects we will use in this function:
  // PETSC Vectors for second derivatives
  vec_and_ptr_dim_t T_l_dd, T_l_dd_nm1;
  Vec v_dd[P4EST_DIM][P4EST_DIM];
  Vec v_dd_nm1[P4EST_DIM][P4EST_DIM];

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Interpolators
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd); /*= NULL;*/
//  SL_backtrace_interp = new my_p4est_interpolation_nodes_t(ngbd);

  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd_nm1);/* = NULL;*/
//  SL_backtrace_interp_nm1 = new my_p4est_interpolation_nodes_t(ngbd_nm1);
  // Get the relevant second derivatives
  T_l_dd.create(p4est,nodes);
  ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec);

  if(advection_sl_order==2) {
      T_l_dd_nm1.create(p4est_nm1,nodes_nm1);
      ngbd_nm1->second_derivatives_central(T_l_nm1.vec,T_l_dd_nm1.vec);
    }

  foreach_dimension(d){
    foreach_dimension(dd){
      ierr = VecCreateGhostNodes(p4est, nodes, &v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2){
          ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &v_dd_nm1[d][dd]); CHKERRXX(ierr);
        }
    }
  }

  // v_dd[k] is the second derivative of the velocity components n along cartesian direction k
  // v_dd_nm1[k] is the second derivative of the velocity components nm1 along cartesian direction k

  ngbd->second_derivatives_central(v.vec,v_dd[0],v_dd[1],P4EST_DIM);
  if(advection_sl_order ==2){
      ngbd_nm1->second_derivatives_central(v_nm1.vec, DIM(v_dd_nm1[0], v_dd_nm1[1], v_dd_nm1[2]), P4EST_DIM);
    }

  // Do the Semi-Lagrangian backtrace:
  if(advection_sl_order ==2){
      trajectory_from_np1_to_nm1(p4est,nodes,ngbd_nm1,ngbd,v_nm1.vec,v_dd_nm1,v.vec,v_dd,dt_nm1,dt,xyz_d_nm1,xyz_d);
      if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Completes backtrace trajectory \n");
    }
  else{
      trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,v.vec,v_dd,xyz_d);
    }

  // Add backtrace points to the interpolator(s):
  foreach_local_node(n,nodes){
    double xyz_temp[P4EST_DIM];
    double xyz_temp_nm1[P4EST_DIM];

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
  SL_backtrace_interp.set_input(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1],quadratic_non_oscillatory_continuous_v2);
  SL_backtrace_interp.interpolate(T_l_backtrace.vec);

  if(advection_sl_order ==2){
      SL_backtrace_interp_nm1.set_input(T_l_nm1.vec,T_l_dd_nm1.vec[0],T_l_dd_nm1.vec[1], quadratic_non_oscillatory_continuous_v2);
      SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
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

  // Clear interp points:
  xyz_d->clear();xyz_d->shrink_to_fit();
  xyz_d_nm1->clear();xyz_d_nm1->shrink_to_fit();

  // Clear and delete interpolators:
  SL_backtrace_interp.clear();
  SL_backtrace_interp_nm1.clear();

  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Completes backtrace \n");
}

void interpolate_values_onto_new_grid(Vec *T_l, Vec *T_s,
                                      Vec v_interface[P4EST_DIM],
                                      Vec v_external[P4EST_DIM],
                                      p4est_nodes_t *nodes_new_grid, p4est_t *p4est_new,
                                      my_p4est_node_neighbors_t *ngbd_old_grid,interpolation_method interp_method/*,
                                      Vec *all_fields_old=NULL, Vec *all_fields_new=NULL*/){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  Vec all_fields_old[num_fields_interp];
  Vec all_fields_new[num_fields_interp];

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_old_grid);
//  my_p4est_interpolation_nodes_t* interp_nodes = NULL;
//  interp_nodes = new my_p4est_interpolation_nodes_t(ngbd_old_grid);

  // Set existing vectors as elements of the array of vectors: --------------------------
  unsigned int i = 0;
  if(solve_stefan){

      all_fields_old[i++] = *T_l; // Now, all_fields_old[0] and T_l both point to same object (where old T_l vec sits)
      if(do_we_solve_for_Ts) all_fields_old[i++] = *T_s;

      foreach_dimension(d){
        all_fields_old[i++] = v_interface[d];
      }
    }
  if(solve_navier_stokes){
      foreach_dimension(d){
        all_fields_old[i++] = v_external[d];
      }
    }
  P4EST_ASSERT(i == num_fields_interp);

  // Create the array of vectors to hold the new values: ------------------------------
  PetscErrorCode ierr;
  for(unsigned int j = 0;j<num_fields_interp;j++){
    ierr = VecCreateGhostNodes(p4est_new,nodes_new_grid,&all_fields_new[j]);CHKERRXX(ierr);
    }

  // Do interpolation:--------------------------------------------
  interp_nodes.set_input(all_fields_old,interp_method,num_fields_interp);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_new_grid){
    node_xyz_fr_n(n,p4est_new,nodes_new_grid,xyz);
    interp_nodes.add_point(n,xyz);
  }

  interp_nodes.interpolate(all_fields_new);
  interp_nodes.clear();

  // Destroy the old fields no longer in use:------------------------
  for(unsigned int k=0;k<num_fields_interp;k++){
    ierr = VecDestroy(all_fields_old[k]);CHKERRXX(ierr); // Destroy objects where the old vectors were
  }
  // Slide the newly interpolated fields to back to their passed objects
  i = 0;
  if(solve_stefan){
      *T_l = all_fields_new[i++]; // Now, T_l points to (new T_l vec)
      if(do_we_solve_for_Ts) *T_s = all_fields_new[i++];

      foreach_dimension(d){
        v_interface[d] = all_fields_new[i++];
      }
    }
  if(solve_navier_stokes){
      foreach_dimension(d){
        v_external[d] = all_fields_new[i++];
      }
    }
  P4EST_ASSERT(i==num_fields_interp);
} // end of interpolate_values_onto_new_grid

// TO-DO : is this used? if not, remove
void compute_curvature(vec_and_ptr_t phi,vec_and_ptr_dim_t normal,vec_and_ptr_t curvature, my_p4est_node_neighbors_t *ngbd,my_p4est_level_set_t LS){

  compute_mean_curvature(*ngbd,normal.vec,curvature.vec);
  /*
  vec_and_ptr_t curvature_tmp;
  curvature_tmp.create(curvature.vec);

  // Get arrays needed:
  curvature_tmp.get_array();
  normal.get_array();
  // Define the qnnn object to help compute the derivatives of the normal:
  quad_neighbor_nodes_of_node_t qnnn;

  // Compute curvature on layer nodes:
  for(size_t i = 0; i<ngbd->get_layer_size(); i++){
      p4est_locidx_t n = ngbd->get_layer_node(i);
      ngbd->get_neighbors(n,qnnn);
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]) CODE3D(+ qnnn.dz_central(normal.ptr[2]));
    }

  // Begin ghost update:
  VecGhostUpdateBegin(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

  // Compute curvature on local nodes:
  for(size_t i = 0; i<ngbd->get_local_size(); i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      ngbd->get_neighbors(n,qnnn);
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]) CODE3D(+ qnnn.dz_central(normal.ptr[2]));
    }

  // End ghost update:
  VecGhostUpdateEnd(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

  // Restore arrays needed:
  curvature_tmp.restore_array();
  normal.restore_array();

  // Now go ahead and extend the curvature values to the whole domain -- Will be used to apply the pointwise Dirichlet condition, dependent on curvature
  LS.extend_from_interface_to_whole_domain_TVD(phi.vec,curvature_tmp.vec,curvature.vec,20);

  // Destroy temp now:
  curvature_tmp.destroy();
*/

}


double interfacial_velocity_expression(double Tl_d, double Ts_d){

  switch(stefan_condition_type){
    // Note: removed curvature from Stefan condition after discussing w frederic and looking at Daniil's thesis 11/24/2020
    case NONDIM_YES_FLUID:{
      return ((St/Pe)*(alpha_s/alpha_l)*(Ts_d - (k_l/k_s)*Tl_d));
    }
    case NONDIM_NO_FLUID:{
      return ((St)*(Ts_d - (k_l/k_s)*Tl_d));
    }
    case DIMENSIONAL:{
      return (k_s*Ts_d -k_l*Tl_d)/(L*rho_s);
    }
    case NONDIM_DISSOLUTION:{
      return -1.*(gamma_diss/Pe)*Tl_d;
    }

    default:{
      throw std::invalid_argument("interfacial_velocity_expression: Unrecognized stefan condition type case \n");
    }
  }
}

void compute_interfacial_velocity(vec_and_ptr_t T_l_n, vec_and_ptr_t T_s_n,
                                  vec_and_ptr_dim_t T_l_d, vec_and_ptr_dim_t T_s_d,
                                  vec_and_ptr_dim_t jump, vec_and_ptr_dim_t &v_interface,
                                  vec_and_ptr_t phi,
                                  p4est_t* p4est, p4est_nodes_t* nodes, my_p4est_node_neighbors_t *ngbd,
                                  double extension_band){

  if(!force_interfacial_velocity_to_zero){

      // Get the first derivatives to compute the jump
      T_l_d.create(p4est,nodes);
      ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
      if(do_we_solve_for_Ts){
        T_s_d.create(T_l_d.vec);
        ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);
       }


      // Initialize level set object -- used in curvature computation, and in extending v interface computed values to entire domain
      my_p4est_level_set_t ls(ngbd);

      vec_and_ptr_t vgamma_n;
      vgamma_n.create(p4est, nodes);
      vgamma_n.get_array();

      vec_and_ptr_dim_t normal;
      normal.create(p4est, nodes);

      compute_normals(*ngbd, phi.vec, normal.vec);
      normal.get_array();

      // Create vector to hold the jump values:
      jump.create(p4est,nodes);

      // Get arrays:
      jump.get_array();
      T_l_d.get_array();
      if(do_we_solve_for_Ts) T_s_d.get_array();
      phi.get_array();

      // First, compute jump in the layer nodes:
      for(size_t i=0; i<ngbd->get_layer_size();i++){
        p4est_locidx_t n = ngbd->get_layer_node(i);

        if(fabs(phi.ptr[n])<extension_band){ // TO-DO: should be nondim for ALL cases

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
      for(size_t i = 0; i<ngbd->get_local_size();i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          if(fabs(phi.ptr[n])<extension_band){
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
         ls.extend_from_interface_to_whole_domain_TVD(phi.vec,jump.vec[d],v_interface.vec[d],20);
      }

      // Scale v_interface computed by appropriate sign if we are doing the coupled test case:
      if((example_ == COUPLED_PROBLEM_EXAMPLE) || (example_ == COUPLED_TEST_2)){
        foreach_dimension(d){
         VecScaleGhost(v_interface.vec[d],coupled_test_sign);
        }
      }

      // Destroy values once no longer needed:
      T_l_d.destroy();
      if(do_we_solve_for_Ts) T_s_d.destroy();
      jump.destroy();

  }
  else{ // Case where we are forcing interfacial velocity to zero
      foreach_dimension(d){
          VecScaleGhost(v_interface.vec[d],0.0);
      }
  }
}

void compute_timestep(vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, double dxyz_close_to_interface, double dxyz_smallest[P4EST_DIM],p4est_nodes_t *nodes, p4est_t *p4est){

  double max_v_norm = 0.0;
  double global_max_vnorm = 0.0;

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
    global_max_vnorm = PI; // known analytically
  }
  else if (example_ == COUPLED_TEST_2){
    global_max_vnorm = 1.5;
  }
  else {

    // Check the values of v_interface locally:
    v_interface.get_array();
    phi.get_array();
    foreach_local_node(n,nodes){
      if (fabs(phi.ptr[n]) < uniform_band*dxyz_close_to_interface){
        max_v_norm = max(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
// For checking dendrite tip velocity case:
//        double xyz_[P4EST_DIM];
//        node_xyz_fr_n(n,p4est,nodes,xyz_);
//        bool is_xaxis = (fabs(xyz_[0] - (xmax - xmin)/2.)<dxyz_close_to_interface);
//        bool is_yaxis = (fabs(xyz_[1] - (ymax - ymin)/2.)<dxyz_close_to_interface);
//        bool is_tip = (is_xaxis && !is_yaxis) || (!is_xaxis && is_yaxis);
//        if(is_tip){

//          double vnorm = sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n]));
//          double Pe_tip = vnorm*vel_nondim_to_dim*d0/alpha_s;
//          printf("Pe_tip = %0.4f \n",Pe_tip);
//        }

      }
    }
    v_interface.restore_array();
    phi.restore_array();

    // Get the maximum v norm across all the processors:
    int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
    SC_CHECK_MPI(mpi_ret);
  }

  // Compute new timestep:
  dt_Stefan = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/global_max_vnorm;//min(global_max_vnorm,1.0);
  //dt = min(dt_computed,dt_max_allowed);

  if((example_ == COUPLED_PROBLEM_EXAMPLE) || (example_ == COUPLED_TEST_2)){
    double N = tfinal*global_max_vnorm/cfl/min(dxyz_smallest[0],dxyz_smallest[1]);
    dt_Stefan = tfinal/N;
  }

  v_interface_max_norm = global_max_vnorm;
}


void prepare_refinement_fields(vec_and_ptr_t phi, vec_and_ptr_t vorticity, vec_and_ptr_t vorticity_refine, vec_and_ptr_dim_t T_l_dd, my_p4est_node_neighbors_t* ngbd){
  PetscErrorCode ierr;

  // Get relevant arrays:
  if(solve_navier_stokes){
    vorticity.get_array();
    vorticity_refine.get_array();
  }
  if(refine_by_d2T) {T_l_dd.get_array();}
  phi.get_array();

  // Compute proper refinement fields on layer nodes:
  for(size_t i = 0; i<ngbd->get_layer_size(); i++){
      p4est_locidx_t n = ngbd->get_layer_node(i);
      if(phi.ptr[n] < 0.){
          if(solve_navier_stokes)vorticity_refine.ptr[n] = vorticity.ptr[n];
        }
      else{
          if(solve_navier_stokes) vorticity_refine.ptr[n] = 0.0;
          if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
              foreach_dimension(d){
                T_l_dd.ptr[d][n]=0.;
              }
            }
        }
    } // end of loop over layer nodes

  // Begin updating the ghost values:
  if(solve_navier_stokes)ierr = VecGhostUpdateBegin(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateBegin(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  //Compute proper refinement fields on local nodes:
  for(size_t i = 0; i<ngbd->get_local_size(); i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      if(phi.ptr[n] < 0.){
          if(solve_navier_stokes)vorticity_refine.ptr[n] = vorticity.ptr[n];
        }
      else{
          if(solve_navier_stokes)vorticity_refine.ptr[n] = 0.0;
          if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
              foreach_dimension(d){
                T_l_dd.ptr[d][n]=0.;
              }
            }
        }
    } // end of loop over local nodes

  // Finish updating the ghost values:
  if(solve_navier_stokes)ierr = VecGhostUpdateEnd(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateEnd(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  // Restore appropriate arrays:
  if(refine_by_d2T) {T_l_dd.restore_array();}
  if(solve_navier_stokes){
    vorticity.restore_array();
    vorticity_refine.restore_array();
  }
  phi.restore_array();
}


void update_the_grid(my_p4est_semi_lagrangian_t sl, splitting_criteria_cf_and_uniform_band_t sp,
                     p4est_t* &p4est_np1, p4est_nodes_t* &nodes_np1, p4est_ghost_t* &ghost_np1,
                     p4est_t* &p4est, p4est_nodes_t* &nodes,
                     vec_and_ptr_t &phi, vec_and_ptr_dim_t v_interface,
                     vec_and_ptr_t phi_cylinder,
                     vec_and_ptr_dim_t phi_dd,
                     vec_and_ptr_t vorticity, vec_and_ptr_t vorticity_refine,
                     vec_and_ptr_t T_l_n,vec_and_ptr_dim_t T_l_dd,
                     my_p4est_node_neighbors_t* ngbd){
  PetscErrorCode ierr;
  int mpi_comm = p4est_np1->mpicomm;

  if(!solve_stefan) refine_by_d2T=false; // override settings if there *is* no temperature field

  bool use_block = false;
  bool expand_ghost_layer = true;

  std::vector<compare_option_t> compare_opn;
  std::vector<compare_diagonal_option_t> diag_opn;
  std::vector<double> criteria;
  std::vector<int> custom_lmax;

  PetscInt num_fields = 0;
  if(solve_navier_stokes) num_fields+=1;// for vorticity
  if(refine_by_d2T)num_fields+=2; // for second derivatives of temperature

  // Create array of fields we wish to refine by, to pass to the refinement tools
  Vec fields_[num_fields];

  // Create vectors for our refinement fields:
  if(num_fields>0){
    // Only use values of vorticity and d2T in the positive subdomain for refinement:
    if(solve_navier_stokes)vorticity_refine.create(p4est,nodes);

    if(refine_by_d2T){
        T_l_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(T_l_n.vec,T_l_dd.vec);
      }

    // Prepare refinement fields:
    prepare_refinement_fields(phi,vorticity,vorticity_refine,T_l_dd,ngbd);

    // Add our refinement fields to the array:
    PetscInt fields_idx = 0;
    if(solve_navier_stokes)fields_[fields_idx++] = vorticity_refine.vec;
    if(refine_by_d2T){
        fields_[fields_idx++] = T_l_dd.vec[0];
        fields_[fields_idx++] = T_l_dd.vec[1];
      }

    P4EST_ASSERT(fields_idx ==num_fields);

    // Add our instructions:
    // Coarsening instructions: (for vorticity)
    if(solve_navier_stokes){
      compare_opn.push_back(LESS_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm/2.);

      // Refining instructions: (for vorticity)
      compare_opn.push_back(GREATER_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm);

      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}
    }
    if(refine_by_d2T){
      double dxyz_smallest[P4EST_DIM];
      dxyz_min(p4est,dxyz_smallest);
      double dTheta = fabs(theta_wall - theta_interface)/(min(dxyz_smallest[0],dxyz_smallest[1])); // max dTheta in liquid subdomain

      // Coarsening instructions: (for dT/dx)
      compare_opn.push_back(SIGN_CHANGE);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(dTheta*gradT_threshold*0.1); // did 0.1* () for the coarsen if no sign change OR below threshold case

      // Refining instructions: (for dT/dx)
      compare_opn.push_back(SIGN_CHANGE);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(dTheta*gradT_threshold);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}

      // Coarsening instructions: (for dT/dy)
      compare_opn.push_back(SIGN_CHANGE);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(dTheta*gradT_threshold*0.1);

      // Refining instructions: (for dT/dy)
      compare_opn.push_back(SIGN_CHANGE);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(dTheta*gradT_threshold);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}
      }
    } // end of "if num_fields!=0"

  // -------------------------------
  // Call grid advection and update:
  // -------------------------------

  if(solve_stefan){
    // Create second derivatives for phi in the case that we are using update_p4est:
    phi_dd.create(p4est,nodes);
    ngbd->second_derivatives_central(phi.vec,phi_dd.vec);

    // Get inner cylinder LSF if needed
    if(example_uses_inner_LSF){
      phi_cylinder.create(p4est,nodes); // create to refine around, then will destroy
      sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
      }

    // Call advection and refinement
    sl.update_p4est(v_interface.vec, dt,
                  phi.vec, phi_dd.vec, example_uses_inner_LSF ? phi_cylinder.vec: NULL,
                  num_fields ,use_block ,true,
                  uniform_band,uniform_band*(1.5),
                  fields_ ,NULL,
                  criteria,compare_opn,diag_opn,custom_lmax,
                  expand_ghost_layer);

  if(print_checkpoints) PetscPrintf(mpi_comm,"Grid update completed \n");

  // Destroy 2nd derivatives of LSF now that not needed
  phi_dd.destroy();

  // Destroy cylinder LSF if it was created, now that it is not needed:
  if(example_uses_inner_LSF){ phi_cylinder.destroy();}
  } // case for stefan or coupled
  else {
      // NS only case --> no advection --> do grid update iteration manually:
      splitting_criteria_tag_t sp_NS(sp.min_lvl,sp.max_lvl,sp.lip);

      // Create a new vector which will hold the updated values of the fields -- since we will interpolate with each grid iteration
      Vec fields_new_[num_fields];
      if(num_fields!=0)
        {
          for(unsigned int k = 0;k<num_fields; k++){
              ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
              ierr = VecCopyGhost(fields_[k],fields_new_[k]);
            }
        }

      // Create a vector which will hold the updated values of the LSF:
      vec_and_ptr_t phi_new;
      phi_new.create(p4est,nodes);
      ierr = VecCopyGhost(phi.vec,phi_new.vec);

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
              PetscPrintf(mpi_comm,"NS grid changed %d times \n",no_grid_changes);
              if(last_grid_balance){
                  p4est_balance(p4est_np1,P4EST_CONNECT_FULL,NULL);
                  PetscPrintf(mpi_comm,"Does last grid balance \n");
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
              my_p4est_interpolation_nodes_t interp_refine_and_coarsen(ngbd);
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
          if(no_grid_changes>10) {PetscPrintf(mpi_comm,"NS grid did not converge!\n"); break;}
        } // end of while grid is changing

      // Update the LSF accordingly:
      phi.destroy();
      phi.create(p4est_np1,nodes_np1);
      ierr = VecCopyGhost(phi_new.vec,phi.vec);

      // Destroy the vectors we created for refine and coarsen:
      for(unsigned int k = 0;k<num_fields; k++){
          ierr = VecDestroy(fields_new_[k]);
        }
      phi_new.destroy();
    } // end of if only navier stokes

  // -------------------------------
  // Destroy refinement fields now that they're not in use:
  // -------------------------------
  if(solve_navier_stokes){
      vorticity_refine.destroy();
      if(refine_by_d2T){T_l_dd.destroy();}
    }
  // -------------------------------
  // Clear up the memory from the std vectors holding refinement info:
  // -------------------------------
  compare_opn.clear(); diag_opn.clear(); criteria.clear();
  compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();
  custom_lmax.clear(); custom_lmax.shrink_to_fit();
};

void poisson_step(Vec phi, Vec phi_solid,
                  Vec phi_dd[P4EST_DIM], Vec phi_solid_dd[P4EST_DIM],
                  Vec* T_l, Vec* T_s,
                  Vec rhs_Tl, Vec rhs_Ts,
                  BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2],
                  BC_WALL_VALUE_TEMP* bc_wall_value_temp[2],
                  my_p4est_node_neighbors_t* ngbd,
                  my_p4est_poisson_nodes_mls_t* &solver_Tl,
                  my_p4est_poisson_nodes_mls_t* &solver_Ts,

                  int cube_refinement,
                  Vec phi_cylinder=NULL, Vec phi_cylinder_dd[P4EST_DIM]=NULL ){

  // Create solvers:
  solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd);
  if(do_we_solve_for_Ts) solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd);

  // Add the appropriate interfaces and interfacial boundary conditions:
  solver_Tl->add_boundary(MLS_INTERSECTION,phi,phi_dd[0],phi_dd[1],
      interface_bc_type_temp,*bc_interface_val_temp[LIQUID_DOMAIN], bc_interface_coeff);

  if(do_we_solve_for_Ts){
    solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid,phi_solid_dd[0],phi_solid_dd[1],
                            interface_bc_type_temp,*bc_interface_val_temp[SOLID_DOMAIN],bc_interface_coeff);
  }

  if(example_uses_inner_LSF && do_we_solve_for_Ts){
    solver_Ts->add_boundary(MLS_INTERSECTION,phi_cylinder,phi_cylinder_dd[0],phi_cylinder_dd[1],
        inner_interface_bc_type_temp,bc_interface_val_inner,bc_interface_coeff_inner);
    }

  // Set diagonal for Tl:
  if(do_advection){ // Cases with advection use semi lagrangian advection discretization in time
      if(advection_sl_order ==2){ // 2nd order semi lagrangian (BDF2 coefficients)
          solver_Tl->set_diag(advection_alpha_coeff/dt);
        }
      else{ // 1st order semi lagrangian (Backward Euler but with backtrace)
          solver_Tl->set_diag(1./dt);
        }
    }
  else{ // Cases with no temperature advection
      if(method_ ==2){ // Crank Nicholson
          solver_Tl->set_diag(2./dt);
        }
      else{ // Backward Euler
         solver_Tl->set_diag(1./dt);
        }
    }

    if(do_we_solve_for_Ts){
      // Set diagonal for Ts:
      if(method_ == 2){ // Crank Nicholson
        solver_Ts->set_diag(2./dt);
      }
      else{ // Backward Euler
        solver_Ts->set_diag(1./dt);
      }
    }

  if(stefan_condition_type == NONDIM_YES_FLUID){
    solver_Tl->set_mu(1./Pe);
    if(do_we_solve_for_Ts) solver_Ts->set_mu((1./Pe)*(alpha_s/alpha_l));
  }
  else if(stefan_condition_type== NONDIM_NO_FLUID){
    solver_Tl->set_mu(alpha_l/alpha_s);
    if(do_we_solve_for_Ts) solver_Ts->set_mu(1.);
  }
  else if(stefan_condition_type == NONDIM_DISSOLUTION){
    solver_Tl->set_mu(1./Pe);
    if(do_we_solve_for_Ts){
      solver_Ts->set_mu(1./Pe); // TO FIX
    }
  }
  else{
    solver_Tl->set_mu(alpha_l);
    if(do_we_solve_for_Ts) solver_Ts->set_mu(alpha_s);
  }


  solver_Tl->set_rhs(rhs_Tl);
  if(do_we_solve_for_Ts) solver_Ts->set_rhs(rhs_Ts);

  // Set some other solver properties:
  solver_Tl->set_integration_order(1);
  solver_Tl->set_use_sc_scheme(0);
  solver_Tl->set_cube_refinement(cube_refinement);
  solver_Tl->set_store_finite_volumes(0);

  if(do_we_solve_for_Ts){
    solver_Ts->set_integration_order(1);
    solver_Ts->set_use_sc_scheme(0);
    solver_Ts->set_cube_refinement(cube_refinement);
    solver_Ts->set_store_finite_volumes(0);
  }

  // Set the wall BC and RHS:
  solver_Tl->set_wc(bc_wall_type_temp,*bc_wall_value_temp[LIQUID_DOMAIN]);
  if(do_we_solve_for_Ts) solver_Ts->set_wc(bc_wall_type_temp,*bc_wall_value_temp[SOLID_DOMAIN]);


  // Preassemble the linear system
  solver_Tl->preassemble_linear_system();
  if(do_we_solve_for_Ts) solver_Ts->preassemble_linear_system();

  // Solve the system:
  solver_Tl->solve(*T_l,false,true,KSPBCGS,PCHYPRE);
  if(do_we_solve_for_Ts) solver_Ts->solve(*T_s,false,true,KSPBCGS,PCHYPRE);


  // Delete solvers:
  delete solver_Tl;
  if(do_we_solve_for_Ts) delete solver_Ts;

}


void navier_stokes_step(my_p4est_navier_stokes_t* ns,
                        p4est_t* p4est_np1,p4est_nodes_t* nodes_np1,
                        Vec v_n[P4EST_DIM], Vec v_nm1[P4EST_DIM],
                        Vec vorticity,Vec press_nodes, Vec phi, double dxyz_close_to_interface,
                        KSPType face_solver_type, PCType pc_face,
                        KSPType cell_solver_type, PCType pc_cell,
                        my_p4est_faces_t* faces_np1, bool compute_pressure_,
                        bool &did_crash,
                        char* name_fluid_forces=NULL, FILE* fich_fluid_forces=NULL){
  PetscErrorCode ierr;

  my_p4est_poisson_faces_t* face_solver;
  my_p4est_poisson_cells_t* cell_solver;
  Vec dxyz_hodge_old[P4EST_DIM];

  int mpi_comm = p4est_np1->mpicomm;

  // Create vector to store old dxyz hodge:
  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecCreateNoGhostFaces(p4est_np1,faces_np1,&dxyz_hodge_old[d],d); CHKERRXX(ierr);
  }

  //if ((ramp_bcs && (tn<t_ramp))) hodge_tolerance = u0*hodge_percentage_of_max_u;
  //else
  hodge_tolerance = NS_norm*hodge_percentage_of_max_u;
  PetscPrintf(mpi_comm,"Hodge tolerance is %e \n",hodge_tolerance);

  int hodge_iteration = 0;
  double convergence_check_on_dxyz_hodge = DBL_MAX;

  face_solver = NULL;
  cell_solver = NULL;

  // Update the parameters: (this is only done to update the cfl potentially)
  ns->set_parameters((1./Re),1.0,NS_advection_sl_order,uniform_band,vorticity_threshold,cfl_NS);

  while((hodge_iteration<hodge_max_it) && (convergence_check_on_dxyz_hodge>hodge_tolerance)){
    ns->copy_dxyz_hodge(dxyz_hodge_old);

    ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);

    convergence_check_on_dxyz_hodge=
        ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell,
                             false,NULL,dxyz_hodge_old,uvw_components);

    ierr= PetscPrintf(mpi_comm,"Hodge iteration : %d, hodge error: %0.3e \n",hodge_iteration,convergence_check_on_dxyz_hodge);CHKERRXX(ierr);
    hodge_iteration++;
  }

  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecDestroy(dxyz_hodge_old[d]); CHKERRXX(ierr);
  }

  // Delete solvers:
  delete face_solver;
  delete cell_solver;

  // Compute velocity at the nodes
  ns->compute_velocity_at_nodes();

  // Set this timestep's "v_n" to be the "v_nm1" for the next timestep
  ns->copy_velocity_n(v_nm1);


  // Now set this step's "v_np1" to be "v_n" for the next timestep -- v_n for next step will be sampled at this grid for now, but will be interpolated onto new grid for next step in beginning of next step
  ns->copy_velocity_np1(v_n);

  // Compute the pressure
  if(compute_pressure_){
    ns->compute_pressure(); // note: only compute pressure at nodes when we are saving to VTK (or evaluating some errors)
    ns->compute_pressure_at_nodes(&press_nodes);
  }


  // Get the computed values of vorticity
  ns->copy_vorticity(vorticity);


  // Compute forces (if we are doing that)
  if((save_fluid_forces && compute_pressure_) || example_requires_area_computation){


    double forces[P4EST_DIM];

    if(save_fluid_forces && compute_pressure_){
      ns->compute_forces(forces);
    }

    // If ice on cylinder case, let's compute the area of the ice, and store that as well:
    double ice_area = 0.0;
    if(example_requires_area_computation){
      // ELYCE DEBUGGING HERE
      // Get total solid domain
      // (including the cylinder bulk -- Note: this will need to be subtracted later, but cyl area is a constant value so no need to compute it over and over):

      // --> First, need to scale phi
      VecScaleGhost(phi,-1.0);

      // --> Compute area of negative domain (aka ice bulk)
      ice_area = area_in_negative_domain(p4est_np1,nodes_np1,phi);

      //--> Scale phi back to normal:
      VecScaleGhost(phi,-1.0);
      // For melting ice sphere example, check if the ice is melted -- if so, halt the simulation:
      if(example_ == MELTING_ICE_SPHERE){
        if((fabs(ice_area) < 0.1*dxyz_close_to_interface) || (ice_area<0.)){
          tfinal = tn;
        }
      }
    }

    // To-do : this may be the source of erroneous first times being reported in fluid force files -- at the first step, dt might be initialized to something much larger than what the eventual dt will use. Might need to do this differently


    ierr = PetscFOpen(mpi_comm, name_fluid_forces,"a",&fich_fluid_forces); CHKERRXX(ierr);
    if(save_fluid_forces && example_requires_area_computation){
      PetscPrintf(mpi_comm,"tn = %g, fx = %g, fy = %g , A = %0.6f \n",tn,forces[0],forces[1],ice_area);
      ierr = PetscFPrintf(mpi_comm, fich_fluid_forces,"%g %g %g %g\n",tn,forces[0],forces[1],ice_area);CHKERRXX(ierr);
    }
    else if(save_fluid_forces && !example_requires_area_computation){
      PetscPrintf(mpi_comm,"tn = %g, fx = %g, fy = %g \n",tn,forces[0],forces[1]);
      ierr = PetscFPrintf(mpi_comm, fich_fluid_forces,"%g %g %g \n",tn,forces[0],forces[1]);CHKERRXX(ierr);
    }
    else if(!save_fluid_forces && example_requires_area_computation){
      PetscPrintf(mpi_comm,"tn = %g, A = %0.6f \n",tn, ice_area);
      ierr = PetscFPrintf(mpi_comm, fich_fluid_forces,"%g %g\n",tn,ice_area);CHKERRXX(ierr);
    }
    ierr = PetscFClose(mpi_comm, fich_fluid_forces); CHKERRXX(ierr);
    PetscPrintf(mpi_comm,"forces saved \n");

  }

  // Check the L2 norm of u to make sure nothing is blowing up
  NS_norm = ns->get_max_L2_norm_u();

  PetscPrintf(mpi_comm,"\n Max NS velocity norm: \n"
                         " - Computational value: %0.4f  "
                         " - Physical value: %0.3e [m/s]  "
                         " - Physical value: %0.3f [mm/s] \n",NS_norm,NS_norm*vel_nondim_to_dim,NS_norm*vel_nondim_to_dim*1000.);

  // Stop simulation if things are blowing up
  if(NS_norm>100.0){
      MPI_Barrier(mpi_comm);
      PetscPrintf(mpi_comm,"The simulation blew up ! ");
      did_crash=true;
    }
  else{
    did_crash=false;
  }

//  // Compute the corresponding timestep:
//  ns->compute_dt();
//  dt_NS = ns->get_dt();

}

void initialize_ns_solver(my_p4est_navier_stokes_t* &ns,
                          p4est_t* p4est_np1,p4est_ghost_t* ghost_np1,
                          my_p4est_node_neighbors_t* ngbd_np1, my_p4est_node_neighbors_t* ngbd_n,
                          my_p4est_hierarchy_t* hierarchy_np1, my_p4est_brick_t* brick,
                          Vec phi, Vec v_n_NS[P4EST_DIM],Vec v_nm1_NS[P4EST_DIM],
                          my_p4est_faces_t* &faces_np1, my_p4est_cell_neighbors_t* &ngbd_c_np1){

  // Create the initial neigbhors and faces (after first step, NS grid update will handle this internally)
  ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  faces_np1 = new my_p4est_faces_t(p4est_np1,ghost_np1,brick,ngbd_c_np1);

  // Create the solver
  ns = new my_p4est_navier_stokes_t(ngbd_n,ngbd_np1,faces_np1);

  // Set the LSF:
  ns->set_phi(phi);
  ns->set_dt(dt_nm1,dt);
  ns->set_velocities(v_nm1_NS,v_n_NS);

  PetscPrintf(p4est_np1->mpicomm,"CFL_NS: %0.2f, rho : %0.2f, mu : %0.3e \n",cfl_NS,rho_l,mu_l);
  ns->set_parameters((1./Re),1.0,NS_advection_sl_order,uniform_band,vorticity_threshold,cfl_NS);


}

bool are_we_saving_vtk(double tstep_, double tn_,bool is_load_step, int& out_idx, bool get_new_outidx){
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

bool are_we_saving_fluid_forces(double tn_,bool is_load_step, int& out_idx, bool get_new_outidx){
  bool out = false;
  if(save_fluid_forces){
      out= ((int (floor(tn_/save_fluid_forces_every_dt) )) !=out_idx) && (!is_load_step);
      if(get_new_outidx){
        out_idx = int (floor(tn_/save_fluid_forces_every_dt) );
      }
  }
  return out;
}


// (WIP -- currently unused:)
void regularize_front(p4est_t* p4est,p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd,vec_and_ptr_t phi)
{
  // FUNCTION FOR REGULARIZING THE SOLIDIFICATION FRONT:
  // adapted from function in my_p4est_multialloy_t originally developed by Daniil Bochkov, adapted by Elyce Bayat 08/24/2020

  // TO-DO: can make these settings variable
  int front_smoothing_ = 0;
  double proximity_smoothing_ = 1.1;

  double dxyz_[P4EST_DIM];
  dxyz_min(p4est,dxyz_);

//  double diag = sqrt(SUMD(SQR(dxyz_[0]),SQR(dxyz_[1]),SQR(dxyz_[2])));
  double dxyz_min_ = MIN(DIM(dxyz_[0],dxyz_[1],dxyz_[2]));
  double new_phi_val = .5*dxyz_min_*pow(2., front_smoothing_);


  PetscErrorCode ierr;
  int mpi_comm = p4est->mpicomm;

  ierr = PetscLogEventBegin(log_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi_comm, "Removing problem geometries... "); CHKERRXX(ierr);

  // auxiliary pointers (in case a corser grid is used further)
//  vec_and_ptr_t front_phi_cur;
//  front_phi_cur.set(front_phi_.vec);
//  p4est_t       *p4est_cur = p4est_;
//  p4est_nodes_t *nodes_cur = nodes_;
//  p4est_ghost_t *ghost_cur = ghost_;
//  my_p4est_node_neighbors_t *ngbd_cur = ngbd_;
//  my_p4est_hierarchy_t *hierarchy_cur = hierarchy_;


  // interpolate level-set function onto a coarses grid (not really used)
//  if (front_smoothing_ != 0)
//  {
//    p4est_cur = p4est_copy(p4est_, P4EST_FALSE);
//    ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
//    nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

//    front_phi_cur.create(front_phi_.vec);
//    VecCopyGhost(front_phi_.vec, front_phi_cur.vec);

//    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est_->user_pointer;
//    bool is_grid_changing = true;
//    while (is_grid_changing)
//    {
//      front_phi_cur.get_array();
//      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl-front_smoothing_, sp_old->lip, sp_old->band);
//      is_grid_changing = sp.refine_and_coarsen(p4est_cur, nodes_cur, front_phi_cur.ptr);
//      front_phi_cur.restore_array();

//      if (is_grid_changing)
//      {
//        my_p4est_partition(p4est_cur, P4EST_TRUE, NULL);

//        // reset nodes, ghost, and phi
//        p4est_ghost_destroy(ghost_cur); ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
//        p4est_nodes_destroy(nodes_cur); nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

//        front_phi_cur.destroy();
//        front_phi_cur.create(p4est_cur, nodes_cur);

//        my_p4est_interpolation_nodes_t interp(ngbd_);

//        double xyz[P4EST_DIM];
//        foreach_node(n, nodes_cur)
//        {
//          node_xyz_fr_n(n, p4est_cur, nodes_cur, xyz);
//          interp.add_point(n, xyz);
//        }

//        interp.set_input(front_phi_.vec, linear); // we know that it is not really an interpolation, rather just a transfer, so therefore linear
//        interp.interpolate(front_phi_cur.vec);
//      }
//    }

//    hierarchy_cur = new my_p4est_hierarchy_t(p4est_cur, ghost_cur, &brick_);
//    ngbd_cur = new my_p4est_node_neighbors_t(hierarchy_cur, nodes_cur);
//    ngbd_cur->init_neighbors();
//  }

  // old "by-hand" procedure to remove sharp corner and narrow necks
  if (0) {
//    vec_and_ptr_t front_phi_tmp(front_phi_cur.vec);

//    p4est_locidx_t nei_n[num_neighbors_cube];
//    bool           nei_e[num_neighbors_cube];

//    double band = 2.*diag_;

//    front_phi_tmp.get_array();
//    front_phi_cur.get_array();

//    // first pass: smooth out extremely curved regions
//    // TODO: make it iterative
//    bool is_changed = false;
//    foreach_local_node(n, nodes_cur)
//    {
//      if (fabs(front_phi_cur.ptr[n]) < band)
//      {
//        ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

//        unsigned short num_neg = 0;
//        unsigned short num_pos = 0;

//        for (unsigned short nn = 0; nn < num_neighbors_cube; ++nn)
//        {
//          if (front_phi_cur.ptr[nei_n[nn]] <= 0) num_neg++;
//          if (front_phi_cur.ptr[nei_n[nn]] >= 0) num_pos++;
//        }

//        if ( (front_phi_cur.ptr[n] <= 0 && num_neg < 3) ||
//             (front_phi_cur.ptr[n] >= 0 && num_pos < 3) )
//        {
//          //        front_phi_cur.ptr[n] = front_phi_cur.ptr[n] < 0 ? 10.*EPS : -10.*EPS;
//          //        front_phi_cur.ptr[n] = front_phi_cur.ptr[n] < 0 ? 10.*EPS : -10.*EPS;
//          if (num_neg < 3) front_phi_cur.ptr[n] =  0.01*diag_;
//          if (num_pos < 3) front_phi_cur.ptr[n] = -0.01*diag_;

//          // check if node is a layer node (= a ghost node for another process)
//          p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
//          if (ni->pad8 != 0) is_changed = true;
//          //        throw;
//        }
//      }
//    }

//    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_C_BOOL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

//    if (is_changed)
//    {
//      ierr = VecGhostUpdateBegin(front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd  (front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }

//    VecCopyGhost(front_phi_cur.vec, front_phi_tmp.vec);

//    // second pass: bridge narrow gaps
//    // TODO: develop a more general approach that works in 3D as well

//    is_changed = false;
//    bool is_ghost_changed = false;
//    // this needs fixing for 3D
//    foreach_local_node(n, nodes_cur)
//    {
//      if (front_phi_cur.ptr[n] < 0 && front_phi_cur.ptr[n] > -band)
//      {
//        ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

//        bool merge = (front_phi_cur.ptr[nei_n[nn_m00]] > 0 &&
//                     front_phi_cur.ptr[nei_n[nn_p00]] > 0 &&
//            front_phi_cur.ptr[nei_n[nn_0m0]] > 0 &&
//            front_phi_cur.ptr[nei_n[nn_0p0]] > 0)
//            || ((front_phi_cur.ptr[nei_n[nn_m00]] > 0 && front_phi_cur.ptr[nei_n[nn_p00]] > 0) &&
//            (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_0m0]] < 0 || front_phi_cur.ptr[nei_n[nn_pm0]] < 0) &&
//            (front_phi_cur.ptr[nei_n[nn_mp0]] < 0 || front_phi_cur.ptr[nei_n[nn_0p0]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0))
//            || ((front_phi_cur.ptr[nei_n[nn_0m0]] > 0 && front_phi_cur.ptr[nei_n[nn_0p0]] > 0) &&
//            (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_m00]] < 0 || front_phi_cur.ptr[nei_n[nn_mp0]] < 0) &&
//            (front_phi_cur.ptr[nei_n[nn_pm0]] < 0 || front_phi_cur.ptr[nei_n[nn_p00]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0));

//        if (merge)
//        {
//          front_phi_tmp.ptr[n] = new_phi_val;

//          // check if node is a layer node (= a ghost node for another process)
//          p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
//          if (ni->pad8 != 0) is_ghost_changed = true;

//          is_changed = true;
//        }

//      }
//    }

//    front_phi_tmp.restore_array();
//    front_phi_cur.restore_array();

//    mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed,       1, MPI_C_BOOL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
//    mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_ghost_changed, 1, MPI_C_BOOL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

//    if (is_ghost_changed)
//    {
//      ierr = VecGhostUpdateBegin(front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd  (front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }
  }

//  bool is_changed = false;
  int num_nodes_smoothed = 0;
  if (proximity_smoothing_ > 0) { // First pass -- shift LSF up, see if there are any islands, remove subpools if there. Then reinitialize, shift back. "Solidify" any nodes that changed sign.
    // shift level-set upwards and reinitialize
    my_p4est_level_set_t ls(ngbd);
    vec_and_ptr_t front_phi_tmp(phi.vec);
    double shift = dxyz_min_*proximity_smoothing_;
    VecCopyGhost(phi.vec, front_phi_tmp.vec);
    VecShiftGhost(front_phi_tmp.vec, shift);

    int num_islands = 0;
    vec_and_ptr_t island_number(phi.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1)
    {
      ierr = PetscPrintf(mpi_comm, "%d subpools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes) {
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

      // solidify all but the biggest pool
      foreach_node(n, nodes) {
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

    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);
    VecShiftGhost(front_phi_tmp.vec, -shift);

    // "solidify" nodes that changed sign
    phi.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes) {
      if (phi.ptr[n] < 0 && front_phi_tmp.ptr[n] > 0) {
//        double blending = smoothstep(2, front_phi_tmp.ptr[n]/shift);
//        front_phi_cur.ptr[n] = blending*front_phi_cur.ptr[n] + (1.-blending)*front_phi_tmp.ptr[n];
        phi.ptr[n] = front_phi_tmp.ptr[n];
        num_nodes_smoothed++;
      }
    }

    phi.restore_array();
    front_phi_tmp.restore_array();
    front_phi_tmp.destroy();
  }

  if (proximity_smoothing_ > 0) { // Second pass --  we shift LSF down, reinitialize, shift back, and see if some of those nodes are still "stuck"
    // shift level-set downwards and reinitialize
    my_p4est_level_set_t ls(ngbd);
    vec_and_ptr_t front_phi_tmp(phi.vec);
    double shift = -0.1*dxyz_min_*proximity_smoothing_;
    VecCopyGhost(phi.vec, front_phi_tmp.vec);
    VecShiftGhost(front_phi_tmp.vec, shift);

    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);
    VecShiftGhost(front_phi_tmp.vec, -shift);

    // "solidify" nodes that changed sign
    phi.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes) {
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

  vec_and_ptr_t front_phi_tmp;
  front_phi_tmp.set(phi.vec);

  // third pass: look for isolated pools of liquid and remove them
  if (num_nodes_smoothed > 0) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
  {
    ierr = PetscPrintf(mpi_comm, "%d nodes smoothed... ", num_nodes_smoothed); CHKERRXX(ierr);
    int num_islands = 0;
    vec_and_ptr_t island_number(phi.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1)
    {
      ierr = PetscPrintf(mpi_comm, "%d pools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes)
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
      foreach_node(n, nodes)
      {
        if (front_phi_tmp.ptr[n] < 0 && island_number.ptr[n] != main_island)
        {
          front_phi_tmp.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      front_phi_tmp.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
  }

//  front_phi_cur.destroy();
//  front_phi_cur.set(front_phi_tmp.vec);
//  VecCopyGhost(front_phi_tmp.vec, front_phi_cur.vec);
//  front_phi_tmp.destroy();

  // iterpolate back onto fine grid
//  if (front_smoothing_ != 0)
//  {
//    my_p4est_level_set_t ls(ngbd);
//    ls.reinitialize_2nd_order(phi.vec, 20);

//    my_p4est_interpolation_nodes_t interp(ngbd_cur);

//    double xyz[P4EST_DIM];
//    foreach_node(n, nodes_)
//    {
//      node_xyz_fr_n(n, p4est_, nodes_, xyz);
//      interp.add_point(n, xyz);
//    }

//    interp.set_input(front_phi_cur.vec, quadratic);
//    interp.interpolate(front_phi_.vec);

//    front_phi_cur.destroy();
//    delete ngbd_cur;
//    delete hierarchy_cur;
//    p4est_nodes_destroy(nodes_cur);
//    p4est_ghost_destroy(ghost_cur);
//    p4est_destroy(p4est_cur);
//  } else {
//    phi.set(front_phi_cur.vec);
//  }

  ierr = PetscPrintf(mpi_comm, "done!\n"); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
}



// --------------------------------------------------------------------------------------------------------------
// Functions for saving to VTK:
// --------------------------------------------------------------------------------------------------------------
void save_everything(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t* ngbd,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, char* filename){
// Things we want to save:
/*
 * LSF
 * LSF2 for ex 2
 * Tl
 * Ts
 * v_interface
 * v NS
 * pressure
 * vorticity
 * smoke -TAKEN OUT
 * */


  // Calculate curvature:

  vec_and_ptr_t kappa;
  vec_and_ptr_dim_t normal;

  kappa.create(p4est,nodes);
  normal.create(p4est,nodes);

  VecScaleGhost(phi.vec,-1.0);
  compute_normals(*ngbd,phi.vec,normal.vec);
  compute_mean_curvature(*ngbd,normal.vec,kappa.vec);

  VecScaleGhost(phi.vec,-1.0);

  // Get arrays:
  phi.get_array();
  if(example_uses_inner_LSF) phi_2.get_array();
  if(solve_stefan){
    Tl.get_array();
    if(do_we_solve_for_Ts) Ts.get_array();
    v_int.get_array();
  }
  if (solve_navier_stokes && !no_flow){
    v_NS.get_array();
    press.get_array();
    vorticity.get_array();
  }
  kappa.get_array();


  // Save data:
  std::vector<std::string> point_names;
  std::vector<double*> point_data;
  // phi
  point_names.push_back("phi");
  point_data.push_back(phi.ptr);

  point_names.push_back("kappa");
  point_data.push_back(kappa.ptr);

  //phi cylinder
  if(example_uses_inner_LSF){
    point_names.push_back("phi_cyl");
    point_data.push_back(phi_2.ptr);
  }

  // stefan related fields
  if(solve_stefan){
    point_names.push_back("T_l");
    point_data.push_back(Tl.ptr);

    if(do_we_solve_for_Ts){
      point_names.push_back("T_s");
      point_data.push_back(Ts.ptr);
    }

    point_names.push_back("v_interface_x");
    point_data.push_back(v_int.ptr[0]);

    point_names.push_back("v_interface_y");
    point_data.push_back(v_int.ptr[1]);
  }

  if(solve_navier_stokes && !no_flow){
    point_names.push_back("u");
    point_data.push_back(v_NS.ptr[0]);

    point_names.push_back("v");
    point_data.push_back(v_NS.ptr[1]);

    point_names.push_back("vorticity");
    point_data.push_back(vorticity.ptr);

    point_names.push_back("pressure");
    point_data.push_back(press.ptr);

  }

  std::vector<std::string> cell_names = {};
  std::vector<double*> cell_data = {};

  my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);

  point_names.clear();point_data.clear();
  cell_names.clear(); cell_data.clear();

  // Restore arrays:
  phi.restore_array();
  if(example_uses_inner_LSF) phi_2.restore_array();

  if(solve_stefan){
    Tl.restore_array();
    if(do_we_solve_for_Ts) Ts.restore_array();
    v_int.restore_array();
  }
  if(solve_navier_stokes && !no_flow){
    v_NS.restore_array();
    press.restore_array();
    vorticity.restore_array();
  }

  kappa.restore_array();
  kappa.destroy();
  normal.destroy();

}

void save_fields_to_vtk(p4est_t* p4est, p4est_nodes_t* nodes,
                       p4est_ghost_t* ghost, my_p4est_node_neighbors_t* ngbd,
                       int out_idx, int grid_res_iter,
                       vec_and_ptr_t phi, vec_and_ptr_t phi_cylinder,
                       vec_and_ptr_t T_l_n,vec_and_ptr_t T_s_n,
                       vec_and_ptr_dim_t v_interface,
                       vec_and_ptr_dim_t v_n, vec_and_ptr_t press_nodes, vec_and_ptr_t vorticity,
                        bool is_crash=false){
  int mpi_comm = p4est->mpicomm;

  // If it's a test case, we ignore, we have our own special save functions for those cases that have error checking as well
  bool test_cases = example_is_a_test_case;

  char output[1000];
  if(!test_cases){
    const char* out_dir = getenv("OUT_DIR_VTK");
    if(!out_dir){
        throw std::invalid_argument("You need to set the output directory for coupled VTK: OUT_DIR_VTK");
      }

    PetscPrintf(mpi_comm,"Saving to vtk, outidx = %d ...\n",out_idx);

    if(example_uses_inner_LSF){
      // Create the cylinder just for visualization purposes, then destroy after saving
      phi_cylinder.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
    }

    if(is_crash){
      sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_CRASH",out_dir,lmin+grid_res_iter,lmax+grid_res_iter);
      save_everything(p4est,nodes,ghost,ngbd,phi,phi_cylinder,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,output);
    }
    else{
      sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_outidx_%d",out_dir,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
      save_everything(p4est,nodes,ghost,ngbd,phi,phi_cylinder,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,output);
    }

    if(example_uses_inner_LSF)phi_cylinder.destroy();
    if(print_checkpoints) PetscPrintf(mpi_comm,"Finishes saving to VTK \n");
  }
};

// --------------------------------------------------------------------------------------------------------------
// Functions for checking the error in test cases (and then saving to vtk):
// --------------------------------------------------------------------------------------------------------------
void save_stefan_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t phi, vec_and_ptr_dim_t v_interface,  double dxyz_close_to_interface, bool are_we_saving_vtk, char* filename_vtk,char *name, FILE *fich){
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
    sval = r/sqrt(tn);

    phi_ana.ptr[n] = s0*sqrt(tn) - r;

    T_ana.ptr[n] = frank_sphere_solution_t(sval);

    v_interface_ana.ptr[n] = s0/(2.0*sqrt(tn));

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
  PetscPrintf(p4est->mpicomm,"dxyz close to interface: %0.2f \n", dxyz_close_to_interface);

  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"Number of grid points used: %d \n \n", num_nodes);


  PetscPrintf(p4est->mpicomm," Linf on phi: %0.4f \n Linf on T_l: %0.4f \n Linf on T_s: %0.4f \n Linf on v_int: %0.4f \n", global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3]);

  // Print errors to file:
  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  PetscFPrintf(p4est->mpicomm,fich,"%e %e %d %e %e %e %e %d %e \n", tn, dt, tstep,
                                                                    global_Linf_errors[0], global_Linf_errors[1],
                                                                    global_Linf_errors[2], global_Linf_errors[3],
                                                                    num_nodes, dxyz_close_to_interface);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);

  // If we are saving this timestep, output the results to vtk:
  if(are_we_saving_vtk){
      std::vector<std::string> point_names;
      std::vector<double*> point_data;

      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","phi_ana","T_ana","v_interface_vec_ana","phi_err","T_l_err","T_s_err","v_interface_vec_err"};
      point_data = {phi.ptr,T_l.ptr,T_s.ptr,v_interface.ptr[0],v_interface.ptr[1],phi_ana.ptr,T_ana.ptr,v_interface_ana.ptr,phi_err.ptr,T_l_err.ptr,T_s_err.ptr,v_interface_err.ptr};


      std::vector<std::string> cell_names = {};
      std::vector<double*> cell_data = {};

      my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    }

  T_l.restore_array();
  T_s.restore_array();
  phi.restore_array();
  v_interface.restore_array();

  T_ana.restore_array();v_interface_ana.restore_array();phi_ana.restore_array();
  T_l_err.restore_array();T_s_err.restore_array();v_interface_err.restore_array();phi_err.restore_array();
  T_ana.destroy();v_interface_ana.destroy();phi_ana.destroy();
  T_l_err.destroy();T_s_err.destroy();v_interface_err.destroy();phi_err.destroy();
}

void save_navier_stokes_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk, char* filename_err_output, FILE* fich){

  // Save NS analytical to compare:
  vec_and_ptr_dim_t vn_analytical;
  vec_and_ptr_t pn_analytical;

  vn_analytical.create(p4est,nodes);
  pn_analytical.create(p4est,nodes);

  velocity_component* analytical_soln_comp[P4EST_DIM];
  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_comp[d] = new velocity_component(d);
    analytical_soln_comp[d]->t = tn;
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
    vorticity.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    point_names = {"phi","u","v","vorticity","pressure","u_ana","v_ana","P_ana","u_err","v_err","P_err"};
    point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],
                  pn_analytical.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr};


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

//    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);
    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();

    vorticity.restore_array();
  }


  v_NS.restore_array();press.restore_array();
  vn_analytical.restore_array(); pn_analytical.restore_array();
  vn_error.restore_array(); press_error.restore_array(); phi.restore_array();

  vn_analytical.destroy();
  pn_analytical.destroy();

  vn_error.destroy();
  press_error.destroy();
}

void save_coupled_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t* ngbd,vec_and_ptr_t phi, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_interface ,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk, char* filename_err_output, FILE* fich){

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
    analytical_soln_temp[d]->t = tn;
  }

  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_velNS[d] = new velocity_component(d);
    analytical_soln_velNS[d]->t = tn;

    analytical_soln_velINT[d] = new interfacial_velocity(d,analytical_soln_temp);
    analytical_soln_velINT[d]->t = tn;
  }

  pressure_field_analytical.t=tn;
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
                                          "%d %g \n", tn, dt, tstep,
                                                     global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                                                     global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],
                                                     global_Linf_errors[6],
                                                     num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  if(are_we_saving_vtk){
    vorticity.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if((tn+dt)>=tfinal){
      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
                     "phi_ana","u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
                     "phi_err","u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
                    phi_analytical.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
                    phi_error.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};

    }
    else{
      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
                     "u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
                     "u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
                    vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
                    vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};
    }


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();

    vorticity.restore_array();
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


}

// --------------------------------------------------------------------------------------------------------------
// Functions for saving or loading the simulation state:
// --------------------------------------------------------------------------------------------------------------

void fill_or_load_double_parameters(save_or_load flag,PetscInt num,splitting_criteria_t* sp, PetscReal *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = tn;
        data[idx++] = dt;
        data[idx++] = dt_nm1;
        data[idx++] = k_l;
        data[idx++] = k_s;
        data[idx++] = alpha_l;
        data[idx++] = alpha_s;
        data[idx++] = rho_l;
        data[idx++] = rho_s;
        data[idx++] = mu_l;
        data[idx++] = L;
        data[idx++] = cfl;
        data[idx++] = uniform_band;
        data[idx++] = sp->lip;
        data[idx++] = NS_norm;
        break;
      }
    case LOAD:{
        tn = data[idx++];
        dt = data[idx++];
        // Note: since these parameters depend on advection sl order, need to load integers first before doubles
        dt_nm1 = data[idx++];
        k_l = data[idx++];
        k_s = data[idx++];
        alpha_l = data[idx++];
        alpha_s = data[idx++];
        rho_l = data[idx++];
        rho_s = data[idx++];
        mu_l = data[idx++];
        L = data[idx++];
        cfl = data[idx++];
        uniform_band= data[idx++];
        sp->lip = data[idx++];
        NS_norm = data[idx++];
      }

    }
  P4EST_ASSERT(idx == num);
};

void fill_or_load_integer_parameters(save_or_load flag, PetscInt num, splitting_criteria_t* sp,PetscInt *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = advection_sl_order;
        data[idx++] = save_every_iter;
        data[idx++] = tstep;
        data[idx++] = sp->min_lvl;
        data[idx++] = sp->max_lvl;
        break;
      }
    case LOAD:{
        advection_sl_order = data[idx++];
        save_every_iter = data[idx++];
        tstep = data[idx++];
        sp->min_lvl=data[idx++];
        sp->max_lvl=data[idx++];
      }

    }
  P4EST_ASSERT(idx == num);
};
void save_or_load_parameters(const char* filename, splitting_criteria_t* sp,save_or_load flag, const mpi_environment_t* mpi=NULL){
  PetscErrorCode ierr;

  // Double parameters we need to save:
  // - tn, dt, dt_nm1 (if 2nd order), k_l, k_s, alpha_l, alpha_s, rho_l, rho_s, mu_l, L, cfl, uniform_band, scaling, data->lip
  PetscInt num_doubles = 15;
  PetscReal double_parameters[num_doubles];

  // Integer parameters we need to save:
  // - current lmin, current lmax, advection_sl_order, save_every_iter, tstep, data->min_lvl, data->max_lvl
  PetscInt num_integers = 5;
  PetscInt integer_parameters[num_integers];

  int fd;
  char diskfilename[PATH_MAX];

  switch(flag){
    case SAVE:{
        if(mpi->rank() ==0){

            // Save the integer parameters to a file
            sprintf(diskfilename,"%s_integers",filename);
            fill_or_load_integer_parameters(flag,num_integers,sp,integer_parameters);
            ierr = PetscBinaryOpen(diskfilename,FILE_MODE_WRITE,&fd); CHKERRXX(ierr);
            ierr = PetscBinaryWrite(fd, integer_parameters, num_integers, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

            // Save the double parameters to a file:

            sprintf(diskfilename, "%s_doubles", filename);
            fill_or_load_double_parameters(flag,num_doubles,sp, double_parameters);

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
            ierr = PetscBinaryRead(fd, integer_parameters, num_integers, PETSC_INT); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
          }
        int mpiret = MPI_Bcast(integer_parameters, num_integers, MPI_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_integer_parameters(flag,num_integers,sp, integer_parameters);

        // Now, load the double parameters:
        sprintf(diskfilename, "%s_doubles", filename);
        if(!file_exists(diskfilename))
          throw std::invalid_argument("The file storing the solver's double parameters could not be found");
        if(mpi->rank() == 0)
        {
          ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
          ierr = PetscBinaryRead(fd, double_parameters, num_doubles, PETSC_DOUBLE); CHKERRXX(ierr);
          ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

        }
        mpiret = MPI_Bcast(double_parameters, num_doubles, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_double_parameters(flag,num_doubles,sp, double_parameters);
        break;
      }
    default:
      throw std::runtime_error("Unkown flag values were used when load/saving parameters \n");


    }

}

void save_state(mpi_environment_t &mpi,const char* path_to_directory,unsigned int n_saved,
                splitting_criteria_cf_and_uniform_band_t* sp, p4est_t* p4est, p4est_nodes_t* nodes,
                Vec phi, Vec T_l_n,Vec T_l_nm1, Vec T_s_n,
                Vec v_NS[P4EST_DIM],Vec v_NS_nm1[P4EST_DIM],Vec vorticity){
  PetscErrorCode ierr;

  if(!file_exists(path_to_directory)){
    create_directory(path_to_directory,p4est->mpirank,p4est->mpicomm);
  }
  if(!is_folder(path_to_directory)){
      if(!create_directory(path_to_directory, p4est->mpirank, p4est->mpicomm))
      {
        char error_msg[1024];
        sprintf(error_msg, "save_state: the path %s is invalid and the directory could not be created", path_to_directory);
        throw std::invalid_argument(error_msg);
      }
    }

  unsigned int backup_idx = 0;

  if(mpi.rank() ==0){
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
                delete_directory(full_path, p4est->mpirank, p4est->mpicomm, true);
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
        delete_directory(full_path_zeroth_index, p4est->mpirank, p4est->mpicomm, true);
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

    int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est->mpicomm); SC_CHECK_MPI(mpiret);// acts as a MPI_Barrier, too

    char path_to_folder[PATH_MAX];
    sprintf(path_to_folder, "%s/backup_%d", path_to_directory, (int) backup_idx);
    create_directory(path_to_folder, p4est->mpirank, p4est->mpicomm);

    char filename[PATH_MAX];

    // save the solver parameters
    sprintf(filename, "%s/solver_parameters", path_to_folder);
    save_or_load_parameters(filename,sp, SAVE,&mpi);

    // Save the p4est and corresponding data:
    if(solve_coupled){
        if(advection_sl_order==2){
            my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                          "p4est",7,
                                          "phi",1,&phi,
                                          "T_l_n",1, &T_l_n,
                                          "T_l_nm1",1, &T_l_nm1,
                                          "T_s_n",1,&T_s_n,
                                          "v_NS_n",P4EST_DIM,v_NS,
                                          "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                          "vorticity",1,&vorticity);
        }
        else{
          my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                        "p4est",6,
                                        "phi",1,&phi,
                                        "T_l_n",1, &T_l_n,
                                        "T_s_n",1,&T_s_n,
                                        "v_NS_n",P4EST_DIM,v_NS,
                                        "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                        "vorticity",1,&vorticity);
        }
    }
    else if (solve_navier_stokes && !solve_stefan){
            my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                          "p4est",4,
                                          "phi",1,&phi,
                                          "v_NS_n",P4EST_DIM,v_NS,
                                          "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                          "vorticity",1,&vorticity);
    }
    ierr = PetscPrintf(p4est->mpicomm,"Saved solver state in ... %s \n",path_to_folder);CHKERRXX(ierr);
}

void load_state(const mpi_environment_t& mpi, const char* path_to_folder,
                splitting_criteria_cf_and_uniform_band_t* sp, p4est_t* &p4est, p4est_nodes_t* &nodes,
                p4est_ghost_t* &ghost,p4est_connectivity* &conn,
                Vec *phi,Vec *T_l_n, Vec *T_l_nm1, Vec *T_s_n,
                Vec v_NS[P4EST_DIM],Vec v_NS_nm1[P4EST_DIM],Vec *vorticity){

  char filename[PATH_MAX];
  if(!is_folder(path_to_folder)) throw std::invalid_argument("Load state: path to directory is invalid \n");

  // First load the general solver parameters -- integers and doubles
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename,sp,LOAD,&mpi);

  // Load p4est_n and corresponding objections
  PetscPrintf(mpi.comm(),"About to try and load forest and data , adv order = %d \n",advection_sl_order);
  if(solve_coupled){
      if(advection_sl_order==2){
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",7,
                                        "phi",NODE_DATA,1,phi,
                                        "T_l_n",NODE_DATA,1,T_l_n,
                                        "T_l_nm1",NODE_DATA,1,T_l_nm1,
                                        "T_s_n",NODE_DATA,1,T_s_n,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,1,vorticity);
      }
      else{
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",6,
                                        "phi",NODE_DATA,1,phi,
                                        "T_l_n",NODE_DATA,1,T_l_n,
                                        "T_s_n",NODE_DATA,1,T_s_n,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,1,vorticity);
      }
  }
  else if (solve_navier_stokes && !solve_stefan){
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",4,
                                        "phi",NODE_DATA,1,phi,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,P4EST_DIM,vorticity);
  }

  P4EST_ASSERT(find_max_level(p4est) == sp->max_lvl);

  // Update the user pointer:
  splitting_criteria_cf_and_uniform_band_t* sp_new = new splitting_criteria_cf_and_uniform_band_t(*sp);
  p4est->user_pointer = (void*) sp_new;

  PetscPrintf(mpi.comm(),"Loads forest and data \n");
}


// --------------------------------------------------------------------------------------------------------------
// Begin main operation:
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;
  PetscViewer viewer;
  int mpi_ret; // Check mpi issues

//  PetscMemorySetGetMaximumUsage();
  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc,argv);

  pl.get_all(cmd);
  select_solvers();

  solve_coupled = solve_navier_stokes && solve_stefan;

  stefan_condition_type = select_stefan_formulation(); // select the form of the stefan condition we will use
  PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d, method = %d \n",lmin,lmax,method_);
  PetscPrintf(mpi.comm(),"Number of mpi tasks: %d \n",mpi.size());
  PetscPrintf(mpi.comm(),"Stefan = %d, NS = %d, Coupled = %d \n",solve_stefan,solve_navier_stokes,solve_coupled);

  // -----------------------------------------------
  // Declare all needed variables:
  // -----------------------------------------------
  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;
  my_p4est_hierarchy_t* hierarchy;
  my_p4est_node_neighbors_t* ngbd;

  p4est_t               *p4est_np1;
  p4est_nodes_t         *nodes_np1;
  p4est_ghost_t         *ghost_np1;
  my_p4est_hierarchy_t* hierarchy_np1;
  my_p4est_node_neighbors_t* ngbd_np1;

  // Level set function(s):---------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_nm1; // LSF for previous timestep... we must keep this so that hodge fields can be updated correctly in NS process

  vec_and_ptr_t phi_solid; // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_cylinder;   // LSF for the inner cylinder, if applicable (example ICE_OVER_CYLINDER)

  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_cylinder_dd;

  // Interface geometry:------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;
  vec_and_ptr_dim_t cyl_normals;

  // Poisson problem:---------------------------------
  int cube_refinement = 1;
  my_p4est_poisson_nodes_mls_t *solver_Tl = NULL;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts = NULL;  // will solve poisson problem for Temperature in solid domain

  vec_and_ptr_t T_l_n;
  vec_and_ptr_t T_l_nm1;
  vec_and_ptr_t T_l_backtrace;
  vec_and_ptr_t T_l_backtrace_nm1;
  vec_and_ptr_t rhs_Tl;

  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  // Vectors to hold first derivatives of T
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;
  vec_and_ptr_dim_t T_l_dd;

  // Stefan problem:------------------------------------
  vec_and_ptr_dim_t v_interface;;
  vec_and_ptr_dim_t jump;

  // Navier-Stokes problem:-----------------------------
  my_p4est_navier_stokes_t* ns = NULL;
  my_p4est_poisson_cells_t* cell_solver; // TO-DO: These may be unnecessary now -- TO-DO: check these unused things, remove if we can
  my_p4est_poisson_faces_t* face_solver;

  PCType pc_face = PCSOR;
  KSPType face_solver_type = KSPBCGS;
  PCType pc_cell = PCSOR;
  KSPType cell_solver_type = KSPBCGS;

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  Vec dxyz_hodge_old[P4EST_DIM];

  my_p4est_cell_neighbors_t *ngbd_c_np1 = NULL;
  my_p4est_faces_t *faces_np1 = NULL;

  // Poisson boundary conditions:
  temperature_field* analytical_T[2];
  external_heat_source* external_heat_source_T[2];

  BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2];
  BC_WALL_VALUE_TEMP* bc_wall_value_temp[2];

  // Navier-Stokes boundary conditions: -----------------
  BoundaryConditions2D bc_velocity[P4EST_DIM];
  BoundaryConditions2D bc_pressure;

  BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM];
  BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM];
  BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM];

  // Note: Pressure BC objects take no arguments, don't need to be initialized

  external_force_per_unit_volume_component* external_force_components[P4EST_DIM];

  // Coupled/NS boundary conditions:
  velocity_component* analytical_soln_v[P4EST_DIM];

  // Interp method: -------------------------------------
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // Variables for extension band and grid size: ---------

  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;

  double min_volume_;
  double extension_band_use_;
  double extension_band_extend_;
  double extension_band_check_;

  // stopwatch
  parStopWatch w;
  w.start("Running example: stefan_with_fluids");
  // -----------------------------------------------
  // Begin loop through number of grid splits:
  // -----------------------------------------------
  for(int grid_res_iter=0;grid_res_iter<=num_splits;grid_res_iter++){
    // Make sure your flags are set to solve at least one of the problems:
    if(!solve_stefan && !solve_navier_stokes){
        throw std::invalid_argument("Woops, you haven't set options to solve either type of physical problem. \n"
                                    "You must at least set solve_stefan OR solve_navier_stokes to true. ");
      }

    // -----------------------------------------------
    // Set up initial grid:
    // -----------------------------------------------
    // domain size information
    set_geometry();

    // if porous media example, create the porous media geometry:
    // do this operation on one process so they don't have different grain defn's.
    /*if(mpi.rank() == 0) */
    if(example_ == MELTING_POROUS_MEDIA){
      make_LSF_for_porous_media(mpi);
    }


    const int n_xyz[]      = { nx,  ny,  0};
    const double xyz_min[] = {xmin, ymin, 0};
    const double xyz_max[] = {xmax,  ymax,  0};
    const int periodic[]   = { px,  py,  0};

    // Set physical properties:
    set_physical_properties();

    // -----------------------------------------------
    // Set properties for the Navier - Stokes problem (if applicable):
    // -----------------------------------------------
    if(solve_navier_stokes){
        set_NS_info();
      }
    set_nondimensional_groups();

    PetscPrintf(mpi.comm(),"\n\nNONDIM GROUPS ARE: \n"
                           "Re = %f \n"
                           "Pr = %f \n"
                           "Pe = %f \n"
                           "St = %f \n"
                           "With: \n"
                           "u_inf = %0.3e [m/s]\n"
                           "delta T = %0.2f [K]\n"
                           "sigma = %0.3e, d = %0.3e, sigma/d = %0.3e \n", Re, Pr, Pe, St,u_inf,deltaT,sigma,d_cyl,sigma/d_cyl);


    // Get the simulation time info (it is example dependent): -- Must be set after non dim groups
    simulation_time_info();
    PetscPrintf(mpi.comm(),"Example number %d \n",example_);
    if(example_==DENDRITE_TEST){
          PetscPrintf(mpi.comm(),"Sim time: %0.3e [sec] = %0.3e [nondim]\n",tfinal*time_nondim_to_dim,tfinal);
    }
    else{
      PetscPrintf(mpi.comm(),"Sim time: %0.2f [min] = %0.2f [nondim]\n",tfinal*time_nondim_to_dim/60.,tfinal);
      bool using_startup = (startup_dim_time>0.) || (startup_nondim_time >0.);
      bool using_dim_startup = using_startup && (startup_dim_time>0.);
      PetscPrintf(mpi.comm(),"Using startup time? %s \n",using_startup? "Yes": "No");
      if(using_startup){
        PetscPrintf(mpi.comm(),"Startup time: %s = %0.2f %s \n",using_dim_startup? "Dimensional" : "Nondimensional", using_dim_startup? startup_dim_time:startup_nondim_time,using_dim_startup? "[seconds]": "[nondim]");
      }
    }

    PetscPrintf(mpi.comm(),"Uniform band is %0.1f \n \n ",uniform_band);
    PetscPrintf(mpi.comm(),"Ramping bcs? %s \n t_ramp = %0.2f [nondim] = %0.2f [seconds]",ramp_bcs?"Yes":"No",t_ramp,t_ramp*time_nondim_to_dim);
    // -----------------------------------------------
    // Create the grid:
    // -----------------------------------------------
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Creating the grid ... \n");

    int load_tstep=-1;

    splitting_criteria_cf_and_uniform_band_t sp(lmin+grid_res_iter,lmax+grid_res_iter,&level_set,uniform_band);
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
    double t_original_start = tstart;

    if(!loading_from_previous_state){
      // Create the p4est at time n:
      p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
      p4est->user_pointer = &sp;

      for(unsigned int l=0;l<lmax+grid_res_iter;l++){
        my_p4est_refine(p4est,P4EST_FALSE,refine_levelset_cf,NULL);
        my_p4est_partition(p4est,P4EST_FALSE,NULL);
      }
      p4est_balance(p4est,P4EST_CONNECT_FULL,NULL);
      my_p4est_partition(p4est,P4EST_FALSE,NULL);

      ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est,ghost);
      nodes = my_p4est_nodes_new(p4est, ghost); //same

      hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
      ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
      ngbd->init_neighbors();

      // Create the p4est at time np1:(this will be modified but is useful for initializing solvers):
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      p4est_np1->user_pointer = &sp;
      my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);

      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est_np1,ghost_np1);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Get the new neighbors:
      hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
      ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

      // Initialize the neigbors:
      ngbd_np1->init_neighbors();
    }
    else{
      p4est=NULL;
      conn=NULL;
      p4est=NULL; ghost=NULL;nodes=NULL;
      hierarchy=NULL;ngbd=NULL;

      phi.vec=NULL;
      T_l_n.vec=NULL; T_l_nm1.vec=NULL;
      T_s_n.vec=NULL;
      foreach_dimension(d){
        v_n.vec[d]=NULL;
        v_nm1.vec[d]=NULL;

      }
      vorticity.vec=NULL;

      const char* load_path = getenv("LOAD_STATE_PATH");
      if(!load_path){
          throw std::invalid_argument("You need to set the  directory for the desired load state");
        }
      PetscPrintf(mpi.comm(),"Load dir is:  %s \n",load_path);

      load_state(mpi,load_path,&sp,p4est,nodes,ghost,conn,
                 &phi.vec,&T_l_n.vec,&T_l_nm1.vec,&T_s_n.vec,
                 v_n.vec,v_nm1.vec,&vorticity.vec);

      PetscPrintf(mpi.comm(),"State was loaded successfully from %s \n",load_path);

      // Update the neigborhood and hierarchy:
      if(hierarchy!=NULL) {
        delete hierarchy;
        }
      if(ngbd!=NULL) {delete ngbd;}

      hierarchy = new my_p4est_hierarchy_t(p4est,ghost,&brick);
      ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
      ngbd->init_neighbors();

      // Create the p4est at time np1 (this will be modified but is useful for initializing solvers):
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est_np1,ghost_np1);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Get the new neighbors:
      hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
      ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

      // Initialize the neigbors:
      ngbd_np1->init_neighbors();

      // Initialize pressure vector (if navier stokes)
      if(solve_navier_stokes) press_nodes.create(p4est,nodes);

      load_tstep =tstep;
      tstart=tn;
      }

    // Initialize level set objet for field extension:
    my_p4est_level_set_t ls(ngbd_np1);

    if(!loading_from_previous_state)tstep = 0;

    PetscPrintf(mpi.comm(),"\nLoading from previous state? %s \n"
                           "Starting timestep = %d \n"
                           "Save state every iter = %d \n"
                           "Save to vtk? %s \n"
                           "Save using %s \n"
                           "Save every dt = %0.5e [nondim] = %0.2f [seconds]\n"
                           "Save every iter = %d \n",loading_from_previous_state?"Yes":"No",
                          tstep,
                          save_state_every_iter,
                          save_to_vtk?"Yes":"No",
                          save_using_dt? "dt" :"iter",
                          save_every_dt, save_every_dt*time_nondim_to_dim,
                          save_every_iter);


    // Initialize output file numbering:
    int out_idx = -1;
    int pressure_save_out_idx = -1;

    // ------------------------------------------------------------
    // Initialize relevant fields:
    // ------------------------------------------------------------
    // Only initialize if we are NOT loading from a previous state
    if(!loading_from_previous_state){
      // LSF:
      if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the level set function (s) ... \n");
      phi.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);
      if(solve_stefan)ls.reinitialize_2nd_order(phi.vec,30); // reinitialize initial LSF to get good signed distance property

      // Temperature fields:
      INITIAL_TEMP *T_init_cf[2];
      temperature_field* analytical_temp[2];
      if(solve_stefan){
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the temperature fields (s) ... \n");

        if(analytical_IC_BC_forcing_term){
          coupled_test_sign = 1.;
          vel_has_switched=false;

          for(unsigned char d=0;d<2;++d){
            analytical_temp[d]= new temperature_field(d);
            analytical_temp[d]->t = tstart;
          }
          for(unsigned char d=0;d<2;++d){
            T_init_cf[d]= new INITIAL_TEMP(d,analytical_temp);
          }

        }
        else{
          for(unsigned char d=0;d<2;++d){
            T_init_cf[d] = new INITIAL_TEMP(d);
            T_init_cf[d]->t = tstart;
          }
        }

        T_l_n.create(p4est,nodes);
        sample_cf_on_nodes(p4est,nodes,*T_init_cf[LIQUID_DOMAIN],T_l_n.vec);

        if(do_we_solve_for_Ts){
          T_s_n.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,*T_init_cf[SOLID_DOMAIN],T_s_n.vec);
        }

        if(do_advection && advection_sl_order ==2){
          T_l_nm1.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,*T_init_cf[LIQUID_DOMAIN],T_l_nm1.vec);
        }

        v_interface.create(p4est,nodes);
        foreach_dimension(d){
          sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[d]);
        }

        if(analytical_IC_BC_forcing_term){
          for(unsigned char d=0;d<2;++d){
            delete analytical_temp[d];
            delete T_init_cf[d];
          }
        }
      }

      // Navier-Stokes fields:

      INITIAL_VELOCITY *v_init_cf[P4EST_DIM];
      velocity_component* analytical_soln[P4EST_DIM];

      if(solve_navier_stokes){
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the Navier-Stokes fields (s) ... \n");

        if(analytical_IC_BC_forcing_term)
        {
          for(unsigned char d=0;d<P4EST_DIM;++d){
            analytical_soln[d] = new velocity_component(d);
            analytical_soln[d]->t = tstart;
          }
        }
        for(unsigned char d=0;d<P4EST_DIM;++d){
          if(analytical_IC_BC_forcing_term){
            v_init_cf[d] = new INITIAL_VELOCITY(d,analytical_soln);
            v_init_cf[d]->t = tstart;
          }
          else {
            v_init_cf[d] = new INITIAL_VELOCITY(d);
          }
        }

        v_n.create(p4est,nodes);
        v_nm1.create(p4est,nodes);
        vorticity.create(p4est,nodes);
        press_nodes.create(p4est,nodes);

        foreach_dimension(d){
          sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_n.vec[d]);
          sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_nm1.vec[d]);
        }
        sample_cf_on_nodes(p4est,nodes,zero_cf,vorticity.vec);
        sample_cf_on_nodes(p4est,nodes,zero_cf,press_nodes.vec);
      }

      for(unsigned char d=0;d<P4EST_DIM;d++){
        if(analytical_IC_BC_forcing_term){
          delete analytical_soln[d];
        }
        if(solve_navier_stokes) delete v_init_cf[d];

      }

      if(solve_navier_stokes)NS_norm = max(fabs(u0),fabs(v0)); // Initialize the NS norm
    } // end of (if not loading from previous state)

    // ------------------------------------------------------------
    // Initialize relevant boundary condition objects:
    // ------------------------------------------------------------
    // For NS or coupled case:
    // Create analytical velocity field for each Cartesian direction if needed:
    if(analytical_IC_BC_forcing_term){
      for(unsigned char d=0;d<P4EST_DIM;d++){
        analytical_soln_v[d] = new velocity_component(d);
        analytical_soln_v[d]->t = tn;
      }
    }

    // For temperature problem:
    if(solve_stefan){
      // Create analytical temperature field for each domain if needed:
      for(unsigned char d=0;d<2;++d){
        if(analytical_IC_BC_forcing_term){ // TO-DO: make all incrementing consistent
          analytical_T[d] = new temperature_field(d);
          analytical_T[d]->t = tn;
        }
      }
      // Set the bc interface type for temperature:
      interface_bc_temp();
      if(example_uses_inner_LSF){inner_interface_bc_temp();} // inner boundary bc

      // Create necessary RHS forcing terms and BC's
      for(unsigned char d=0;d<2;++d){
        if(analytical_IC_BC_forcing_term){
          external_heat_source_T[d] = new external_heat_source(d,analytical_T,analytical_soln_v);
          external_heat_source_T[d]->t = tn;
          bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(NULL,NULL,analytical_T,d);
          bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d,analytical_T);
        }
        else{
          bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(); // will set proper objects later, can be null on initialization
          bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d);
        }
      }
    }

    // For NS problem:
    BC_INTERFACE_VALUE_PRESSURE bc_interface_value_pressure;
    BC_WALL_VALUE_PRESSURE bc_wall_value_pressure;
    BC_WALL_TYPE_PRESSURE bc_wall_type_pressure;

    if(solve_navier_stokes){
      for(unsigned char d=0;d<P4EST_DIM;d++){
        // Set the BC types:
        BC_INTERFACE_TYPE_VELOCITY(d);
        bc_wall_type_velocity[d] = new BC_WALL_TYPE_VELOCITY(d);

        // Set the BC values (and potential forcing terms) depending on what we are running:
        if(analytical_IC_BC_forcing_term){
          // Interface conditions values:
          bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL,analytical_soln_v);
          bc_interface_value_velocity[d]->t = tn;

          // Wall conditions values:
          bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d,analytical_soln_v);
          bc_wall_value_velocity[d]->t = tn;

          // External forcing terms:
          external_force_components[d] = new external_force_per_unit_volume_component(d,analytical_soln_v);
          external_force_components[d]->t = tn;
        }
        else{
          // Interface condition values:
          bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL); // initialize null for now, will add relevant neighbors and vector as required later on

          // Wall condition values:
          bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d);
        }
      }
      interface_bc_pressure(); // sets the interfacial bc type for pressure
      bc_wall_value_pressure.t = tn;
    }


    // ------------------------------------------------------------
    // Initialize relevant solvers:
    // ------------------------------------------------------------
    // First, initialize the Navier-Stokes solver with the grid:
    vec_and_ptr_dim_t v_n_NS, v_nm1_NS;

    // -----------------------------------------------
    // Initialize files to output various data of interest:
    // -----------------------------------------------
    if(print_checkpoints)PetscPrintf(mpi.comm(),"Initializing output files ... \n");
    FILE *fich_stefan_errors;
    char name_stefan_errors[1000];

    FILE *fich_NS_errors;
    char name_NS_errors[1000];

    FILE *fich_coupled_errors;
    char name_coupled_errors[1000];

    FILE *fich_fluid_forces;
    char name_fluid_forces[1000];

    switch(example_){
      case FRANK_SPHERE:{
        // Output file for Frank Sphere errors:
        const char* out_dir_err_stefan = getenv("OUT_DIR_ERR_stefan");
        if(!out_dir_err_stefan){
            throw std::invalid_argument("You need to set the environment variable OUT_DIR_ERR_stefan to save stefan errors");
          }
        sprintf(name_stefan_errors,"%s/frank_sphere_error_lmin_%d_lmax_%d_method_%d.dat",
                out_dir_err_stefan,lmin+grid_res_iter,lmax+grid_res_iter,method_);

        ierr = PetscFOpen(mpi.comm(),name_stefan_errors,"w",&fich_stefan_errors); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(),fich_stefan_errors,"time " "timestep " "iteration "
                                                          "phi_error " "T_l_error " "T_s_error "
                                                          "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
        ierr = PetscFClose(mpi.comm(),fich_stefan_errors); CHKERRXX(ierr);
        break;
        }
      case NS_GIBOU_EXAMPLE:{
          // Output file for NS test case errors:
          const char* out_dir_err_NS = getenv("OUT_DIR_ERR_NS");
          if(!out_dir_err_NS){
              throw std::invalid_argument("You need to set the environment variable OUT_DIR_ERR_NS to save Navier Stokes errors");
            }
          sprintf(name_NS_errors,"%s/navier_stokes_error_lmin_%d_lmax_%d_advection_order_%d.dat",
                  out_dir_err_NS,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

          ierr = PetscFOpen(mpi.comm(),name_NS_errors,"w",&fich_NS_errors); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_NS_errors,"time " "timestep " "iteration " "u_error "
                                                        "v_error " "P_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_NS_errors); CHKERRXX(ierr);

          break;
        }
      case COUPLED_TEST_2:
      case COUPLED_PROBLEM_EXAMPLE:{
          // Output file for coupled problem test case:
          const char* out_dir_err_coupled = getenv("OUT_DIR_ERR_coupled");
          sprintf(name_coupled_errors,"%s/coupled_error_ex_%d_lmin_%d_lmax_%d_method_%d_advection_order_%d.dat",
                  out_dir_err_coupled,example_,lmin+grid_res_iter,lmax + grid_res_iter,method_,advection_sl_order);

          ierr = PetscFOpen(mpi.comm(),name_coupled_errors,"w",&fich_coupled_errors); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_coupled_errors,"time " "timestep " "iteration "
                                                             "u_error " "v_error " "P_error "
                                                             "Tl_error " "Ts_error " "vint_error" "phi_error "
                                                             "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_coupled_errors); CHKERRXX(ierr);
          break;
        }
      case FLOW_PAST_CYLINDER:
      case DISSOLVING_DISK_BENCHMARK:
      case MELTING_POROUS_MEDIA:
      case MELTING_ICE_SPHERE:
      case ICE_AROUND_CYLINDER:{
        if(save_fluid_forces || example_requires_area_computation){
          // Output file for NS test case errors:
          const char* out_dir_fluid_forces = getenv("OUT_DIR_FILES");
          if(!out_dir_fluid_forces){
              throw std::invalid_argument("You need to set the environment variable OUT_DIR_FILES to save fluid forces");
            }
          sprintf(name_fluid_forces,"%s/area_and_or_force_data_Re_%0.2f_lmin_%d_lmax_%d_advection_order_%d.dat",
                  out_dir_fluid_forces,Re,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

          if(no_flow || !save_fluid_forces){
            ierr = PetscFOpen(mpi.comm(),name_fluid_forces,"w",&fich_fluid_forces); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(),fich_fluid_forces,"time A \n");CHKERRXX(ierr);
            ierr = PetscFClose(mpi.comm(),fich_fluid_forces); CHKERRXX(ierr);
          }
          else{
            ierr = PetscFOpen(mpi.comm(),name_fluid_forces,"w",&fich_fluid_forces); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(),fich_fluid_forces,"time fx fy A \n");CHKERRXX(ierr);
            ierr = PetscFClose(mpi.comm(),fich_fluid_forces); CHKERRXX(ierr);}
        }
          break;
      }

      default:{
        break;
        }
      }

    // ------------------------------------------------------------
    // Begin stepping through time
    // ------------------------------------------------------------
//    for (tn=tstart;tn<tfinal; tn+=dt, tstep++){

    if(!loading_from_previous_state){tstep=0;}
    tn = tstart;
    int last_tstep=-1;

    double cfl_NS_steady = cfl_NS; // store desired CFL, will use it eventually, but always use 0.5 for the first 10 iterations just to make sure NS solver stabilizes nicely
    double dt_max_allowed_steady = dt_max_allowed;
    double hodge_percentage_steady = hodge_percentage_of_max_u;

    while(tn<=tfinal){ // trying something
      // Enforce startup iterations for verification tests if needed:
      if((startup_iterations>0)){
        if(tstep<startup_iterations){
          force_interfacial_velocity_to_zero=true;
          tn=tstart;
        }
        else if(tstep==startup_iterations){
          force_interfacial_velocity_to_zero=false;
          tn = tstart;
        }
      }
      if(solve_navier_stokes){
        // Adjust the cfl_NS depending on the timestep:
        if(tstep<=10){
          cfl_NS=0.5;
          double dxyz_s[P4EST_DIM];
          dxyz_min(p4est, dxyz_s);
          dt_max_allowed = cfl_NS*min(dxyz_s[0], dxyz_s[1])/max(u0, v0);

          // loosen hodge criteria for initialization for porous media case:
          if(example_ == MELTING_POROUS_MEDIA){
            hodge_percentage_of_max_u = 0.1;
          }

        }
        else{
          cfl_NS = cfl_NS_steady;
          dt_max_allowed = dt_max_allowed_steady;

          if(example_ == MELTING_POROUS_MEDIA){
            hodge_percentage_of_max_u = hodge_percentage_steady; // to-do : clean up, num startup iterations should be a user intput, instead of just being set to 10
          }

        }
      }

      if(tstep==0){
        dxyz_min(p4est,dxyz_smallest);

        // Initialize timesteps to use:
        if(solve_navier_stokes){
          dt_nm1 = cfl_NS*min(dxyz_smallest[0],dxyz_smallest[1])/max(u0,v0);
          dt = dt_nm1;
        }
        else{
          dt_nm1 = cfl_NS*min(dxyz_smallest[0],dxyz_smallest[1])/1.;
          dt = dt_nm1;}
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


      // ------------------------------------------------------------
      // Print iteration information:
      // ------------------------------------------------------------
      if(example_ == DENDRITE_TEST){
        ierr = PetscPrintf(mpi.comm(),"\n -------------------------------------------\n"
                                      "Iteration %d , Time: %0.4e [nondim] "
                                      "= Time: %0.4e [nondim] "
                                      "= %0.4e [sec], "
                                      " Timestep: %0.4e [nondim] = %0.4e [sec],"
                                      " Percent Done : %0.2f %"
                                      " \n ------------------------------------------- \n",
                                      tstep,tn,tn,tn*time_nondim_to_dim,
                                      dt, dt*(time_nondim_to_dim),
                                      ((tn-t_original_start)/(tfinal-t_original_start))*100.0);
      }
      else{
        int num_nodes = nodes->num_owned_indeps;
        MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM,mpi.comm());
        ierr = PetscPrintf(mpi.comm(),"\n -------------------------------------------\n"
                                      "Iteration %d , Time: %0.3f [nondim] "
                                      "= Time: %0.3f [nondim] "
                                      "= %0.3f [sec] "
                                      "= %0.2f [min],"
                                      " Timestep: %0.3e [nondim] = %0.3e [sec],"
                                      " Percent Done : %0.2f %"
                                      " \n ------------------------------------------- \n"
                                      "Number of nodes : %d \n \n",
                           tstep,tn,tn,tn*time_nondim_to_dim,tn*(time_nondim_to_dim)/60.,
                           dt, dt*(time_nondim_to_dim),
                           ((tn-t_original_start)/(tfinal-t_original_start))*100.0,num_nodes);
      }

      if((timing_every_n>0) && (tstep%timing_every_n == 0)) {
        PetscPrintf(mpi.comm(),"Current time info : \n");
        w.read_duration_current();
      }
      if(solve_stefan){
          if(v_interface_max_norm>v_int_max_allowed){
              PetscPrintf(mpi.comm(),"Interfacial velocity has exceeded its max allowable value \n"
                                     "Max allowed is : %g \n",v_int_max_allowed);
              MPI_Abort(mpi.comm(),1);
            }
        }
      // -------------------------------
      // Update analytical velocity for coupled problem example:
      // -------------------------------
      if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up appropriate boundary conditions... \n");

      if((tstep>0) && analytical_IC_BC_forcing_term){
        for(unsigned char d=0;d<P4EST_DIM;d++){
          analytical_soln_v[d]->t = tn;
        }
      }

      // ------------------------------------------------------------
      // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
      // ------------------------------------------------------------
      if((tstep>0) && solve_stefan){ // mostly memory safe (may have tiniest leak TO-DO)
        // -------------------------------
        // Update BC objects for stefan problem:
        // -------------------------------
        if((tstep>0) &&solve_stefan){
          if(analytical_IC_BC_forcing_term){
            for(unsigned char d=0;d<2;d++){
              analytical_T[d]->t = tn;
              bc_interface_val_temp[d]->t = tn;
              bc_wall_value_temp[d]->t = tn;
              external_heat_source_T[d]->t = tn;
            }
          }
        } // If not, we use the curvature and neighbors, but we have to wait till curvature is computed in Poisson step to apply this, so it is applied later

        // -------------------------------
        // Create all vectors that will be used
        // strictly for the stefan step
        // (aka created and destroyed in stefan step)
        // -------------------------------

        // Solid LSF:
        phi_solid.create(p4est_np1,nodes_np1);

        //Curvature and normal for BC's and setting up solver:
        normal.create(p4est_np1,nodes_np1);
        curvature.create(p4est_np1,nodes_np1);

        // Second derivatives of LSF's (for solver):
        phi_solid_dd.create(p4est_np1,nodes_np1);
        phi_dd.create(p4est_np1,nodes_np1);

        if(example_uses_inner_LSF){
          phi_cylinder.create(p4est_np1,nodes_np1);
          phi_cylinder_dd.create(p4est_np1,nodes_np1);
        }
        if(do_advection){
          T_l_backtrace.create(p4est_np1,nodes_np1);
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

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing normal and curvature ... \n");
        // Get the new solid LSF:
        VecCopyGhost(phi.vec,phi_solid.vec);
        VecScaleGhost(phi_solid.vec,-1.0);

        // Compute normals on the interface:
        compute_normals(*ngbd_np1,phi_solid.vec,normal.vec); // normal here is outward normal of solid domain

        // Compute curvature on the interface:
        my_p4est_level_set_t ls_new_new(ngbd_np1);

        // Feed the curvature computed to the interfacial boundary condition:
        if(interfacial_temp_bc_requires_curvature){
          ls_new_new.reinitialize_2nd_order(phi_solid.vec,30);
          // We need curvature of the solid domain, so we use phi_solid and negative of normals
          compute_curvature(phi_solid,normal,curvature,ngbd_np1,ls_new_new);

          for(unsigned char d=0;d<2;d++){
            bc_interface_val_temp[d]->set(ngbd_np1,curvature.vec);

          }
        }
        // Feed the normals to the interfacial boundary condition if needed:
        if(interfacial_temp_bc_requires_normal){
          for(unsigned char d=0; d<2; d++){
            bc_interface_val_temp[d]->set_normals(ngbd_np1,normal.vec[0],normal.vec[1]);
          }
        }
        // -------------------------------
        // Get most updated derivatives of the LSF's (on current grid)
        // -------------------------------
        if(print_checkpoints)PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");

        // Get derivatives of liquid and solid LSF's
        if (print_checkpoints) PetscPrintf(mpi.comm(),"New solid LSF acquired \n");
        ngbd_np1->second_derivatives_central(phi.vec,phi_dd.vec);
        ngbd_np1->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

        // Get inner LSF and derivatives if required:
        if(example_uses_inner_LSF){
            sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);
            ngbd_np1->second_derivatives_central(phi_cylinder.vec,phi_cylinder_dd.vec);
          }

        // -------------------------------
        // Compute advection terms (if applicable):
        // -------------------------------
        if (do_advection){
            if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing advection terms ... \n");
            do_backtrace(T_l_n,T_l_nm1,
                         T_l_backtrace,T_l_backtrace_nm1,
                         v_n,v_nm1,
                         p4est_np1,nodes_np1,ngbd_np1,
                         p4est,nodes,ngbd);
            // Do backtrace with v_n --> navier-stokes fluid velocity
        } // end of do_advection if statement

        // -------------------------------
        // Set up the RHS for Poisson step:
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up RHS for Poisson problem ... \n");

        setup_rhs(phi,T_l_n,T_s_n,
                  rhs_Tl,rhs_Ts,
                  T_l_backtrace,T_l_backtrace_nm1,
                  p4est_np1,nodes_np1,ngbd_np1,external_heat_source_T);

        // -------------------------------
        // Execute the Poisson step:
        // -------------------------------
        // Slide Temp fields:
        if(do_advection && advection_sl_order==2){
          T_l_nm1.destroy();
          T_l_nm1.create(p4est_np1,nodes_np1);
          ierr = VecCopyGhost(T_l_n.vec,T_l_nm1.vec);CHKERRXX(ierr);
        }
        // Solve Poisson problem:
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Poisson problem solution step... \n");

        poisson_step(phi.vec,phi_solid.vec,
                     phi_dd.vec,phi_solid_dd.vec,
                     &T_l_n.vec,&T_s_n.vec,
                     rhs_Tl.vec,rhs_Ts.vec,
                     bc_interface_val_temp,bc_wall_value_temp,
                     ngbd_np1, solver_Tl, solver_Ts,
                     cube_refinement,
                     example_uses_inner_LSF? phi_cylinder.vec:NULL,
                     example_uses_inner_LSF? phi_cylinder_dd.vec:NULL);
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Poisson step completed ... \n");


        // -------------------------------
        // Destroy all vectors
        // that were used strictly for the
        // stefan step (aka created and destroyed in stefan step)
        // -------------------------------
        // Solid LSF:
        phi_solid.destroy();

        // Curvature and normal for BC's and setting up solver:
        normal.destroy();
        curvature.destroy();

        // Second derivatives of LSF's (for solver):
        phi_solid_dd.destroy();
        phi_dd.destroy();

        if(example_uses_inner_LSF){
          phi_cylinder.destroy();
          phi_cylinder_dd.destroy();
        }
        if(do_advection){
          T_l_backtrace.destroy();
          if(advection_sl_order ==2){
              T_l_backtrace_nm1.destroy();
            }
        }
        // Destroy arrays to hold the RHS:
        rhs_Tl.destroy();
        if(do_we_solve_for_Ts) rhs_Ts.destroy();

        // -------------------------------
        // Clear interfacial BC if needed (curvature, normals, or both depending on example)
        // -------------------------------
        if(interfacial_temp_bc_requires_curvature){
          for(unsigned char d=0;d<2;++d){
            bc_interface_val_temp[d]->clear();
          }
        }
        if(interfacial_temp_bc_requires_normal){
          for(unsigned char d=0;d<2;++d){
            bc_interface_val_temp[d]->clear_normals();
          }
        }
      } // end of "if solve stefan"

      // -------------------------------
      // If first iteration, perturb the LSF(s):
      // -------------------------------
      ls.update(ngbd_np1);
      if(tstep<1){
          // Perturb the LSF on the first iteration
          ls.perturb_level_set_function(phi.vec,EPS);
        }

      // ------------------------------------------------------------
      // Extend Fields Across Interface (if solving Stefan):
      // -- Note: we do not extend NS velocity fields bc NS solver handles that internally
      // ------------------------------------------------------------
      // Get smallest grid size: (this gets used in all examples at some point)
      dxyz_min(p4est,dxyz_smallest);
      dxyz_close_to_interface = 1.2*max(dxyz_smallest[0],dxyz_smallest[1]);
      if(solve_stefan){
        // ------------------------------------------------------------
        // Define some variables needed to specify how to extend across the interface:
        // ------------------------------------------------------------

        min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
        extension_band_use_    = (8.)*pow(min_volume_, 1./ double(P4EST_DIM)); //8
        extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
        extension_band_check_  = (6.)*pow(min_volume_, 1./ double(P4EST_DIM)); // 6

//        if((tstep ==0) && example_ == ICE_AROUND_CYLINDER && solve_coupled){
//            double delta_r = r0 - r_cyl;
//            PetscPrintf(mpi.comm(),"The uniform band is %0.2f\n",uniform_band);
//              if(delta_r<4.*dxyz_close_to_interface ){
//                  PetscPrintf(mpi.comm()," Your initial delta_r is %0.3e, and it must be at least %0.3e \n"
//                                         "Conversely, dxyz_min is %0.3e, it must be less than %e \n",delta_r,6.*dxyz_close_to_interface,dxyz_close_to_interface,delta_r/6.0);
//                  SC_ABORT("Your initial delta_r is too small \n");
//                }
//          }
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning field extension \n");
        // -------------------------------
        // Create all fields for this procedure:
        // -------------------------------
        phi_solid.create(p4est_np1,nodes_np1);
        liquid_normals.create(p4est_np1,nodes_np1);
        solid_normals.create(p4est,nodes_np1);

        // -------------------------------
        // Get the solid LSF:
        // -------------------------------
        VecCopyGhost(phi.vec,phi_solid.vec);
        VecScaleGhost(phi_solid.vec,-1.0);

        // -------------------------------
        // Compute normals for each domain:
        // -------------------------------

        compute_normals(*ngbd_np1,phi.vec,liquid_normals.vec);

        foreach_dimension(d){
          VecCopyGhost(liquid_normals.vec[d],solid_normals.vec[d]);
          VecScaleGhost(solid_normals.vec[d],-1.0);
        }

        // -------------------------------
        // Extend Temperature Fields across the interface:
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling extension over phi \n");
        ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec,
                                          50, 2, 1.e-15,
                                          extension_band_use_, extension_band_extend_,
                                          extension_band_check_,
                                          liquid_normals.vec, NULL,
                                          NULL, false, NULL,NULL);

        if(do_we_solve_for_Ts){
          ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec,
                                            50, 2, 1.e-15,
                                            extension_band_use_, extension_band_extend_,
                                            extension_band_check_,
                                            solid_normals.vec, NULL,
                                            NULL, false, NULL, NULL);
        }

        if(example_uses_inner_LSF){
          phi_cylinder.create(p4est_np1,nodes_np1);
          cyl_normals.create(p4est_np1,nodes_np1);

          sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);
          compute_normals(*ngbd_np1,phi_cylinder.vec,cyl_normals.vec);

          if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling extension over phi_cylinder \n");
          ls.extend_Over_Interface_TVD_Full(phi_cylinder.vec, T_s_n.vec,
                                            50, 2, 1.e-15,
                                            0.5*extension_band_use_, 0.5*extension_band_extend_, 0.5*extension_band_check_,
                                            cyl_normals.vec, NULL, NULL,
                                            false, NULL, NULL);

          // ELYCE DEBUGGING HERE
          /*
          double phi_cyl_area;
          VecScaleGhost(phi_cylinder.vec,-1.0);
          phi_cyl_area = area_in_negative_domain(p4est_np1,nodes_np1,phi_cylinder.vec);
          PetscPrintf(mpi.comm(),"phi cyl area is %0.6f \n",phi_cyl_area);
          */

          cyl_normals.destroy();
          phi_cylinder.destroy();

        }

        // -------------------------------
        // Delete fields now:
        // -------------------------------
        liquid_normals.destroy();
        solid_normals.destroy();
        phi_solid.destroy();


        // --------------------------------------------------------------------------------------------------------------
        // Compute the interfacial velocity (Stefan): -- do now so it can be used for NS boundary condition
        // --------------------------------------------------------------------------------------------------------------

          if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing interfacial velocity ... \n");
          v_interface.destroy();
          v_interface.create(p4est_np1,nodes_np1);


          compute_interfacial_velocity(T_l_n,T_s_n,
                                       T_l_d,T_s_d,
                                       jump,v_interface,
                                       phi,
                                       p4est_np1,nodes_np1,ngbd_np1,extension_band_extend_);



      } // end of "if solve stefan"


      // --------------------------------------------------------------------------------------------------------------
      // Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
      // --------------------------------------------------------------------------------------------------------------
      if ((tstep>0) && solve_navier_stokes){
        // -------------------------------
        // Update BC and RHS objects for navier-stokes problem:
        // -------------------------------
        // Setup velocity conditions
        for(unsigned char d=0;d<P4EST_DIM;d++){
          if(interfacial_vel_bc_requires_vint){
            bc_interface_value_velocity[d]->set(ngbd_np1,v_interface.vec[d]);
          }
          bc_interface_value_velocity[d]->t = tn;
          bc_wall_value_velocity[d]->t = tn;

          bc_velocity[d].setInterfaceType(interface_bc_type_velocity[d]);
          bc_velocity[d].setInterfaceValue(*bc_interface_value_velocity[d]);
          bc_velocity[d].setWallValues(*bc_wall_value_velocity[d]);
          bc_velocity[d].setWallTypes(*bc_wall_type_velocity[d]);
        }
        // Setup pressure conditions:
        bc_pressure.setInterfaceType(interface_bc_type_pressure);
        bc_pressure.setInterfaceValue(bc_interface_value_pressure);
        bc_pressure.setWallTypes(bc_wall_type_pressure);
        bc_pressure.setWallValues(bc_wall_value_pressure);

        // Set external_forces if applicable
        if(analytical_IC_BC_forcing_term){
          foreach_dimension(d){
            external_force_components[d]->t = tn;
          }
          bc_wall_value_pressure.t=tn;
        }

        // -------------------------------
        // Update the NS grid (or initialize the solver)
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling the Navier-Stokes grid update... \n");
        if((tstep==1) || (tstep==load_tstep)){
          PetscPrintf(mpi.comm(),"Initializing Navier-Stokes solver \n");
          v_n_NS.create(p4est_np1,nodes_np1);
          v_nm1_NS.create(p4est,nodes);

          foreach_dimension(d){
            ierr = VecCopyGhost(v_nm1.vec[d],v_nm1_NS.vec[d]); CHKERRXX(ierr);
            ierr = VecCopyGhost(v_n.vec[d],v_n_NS.vec[d]); CHKERRXX(ierr);
          }

          initialize_ns_solver(ns,p4est_np1,ghost_np1,ngbd_np1,ngbd,
                               hierarchy_np1,&brick,
                               phi.vec,v_n_NS.vec,v_nm1_NS.vec,
                               faces_np1,ngbd_c_np1);
        }
        else{
          ns->update_from_tn_to_tnp1_grid_external(phi.vec, phi_nm1.vec,
                                                   p4est_np1,nodes_np1,ghost_np1,
                                                   ngbd_np1,
                                                   faces_np1,ngbd_c_np1,
                                                   hierarchy_np1);
          phi_nm1.destroy(); // Now that hodge has been correctly interpolated, we can destroy this
        }

        // NOTE: we update NS grid first, THEN set new BCs and forces. This is because the update grid interpolation of the hodge variable
        // requires knowledge of the boundary conditions from that same timestep (the previous one, in our case)
        // -------------------------------
        // Set the NS timestep: // change to include both timesteps (dtnm1,dtn)
        // -------------------------------
        if(advection_sl_order ==2){
            ns->set_dt(dt_nm1,dt);
          }
        else{
            ns->set_dt(dt);
          }
        // -------------------------------
        // Set BC's and external forces if relevant
        // -------------------------------
        // Set the boundary conditions:
        ns->set_bc(bc_velocity,&bc_pressure);

        // Set the RHS:
        if(analytical_IC_BC_forcing_term){
          CF_DIM *external_forces[P4EST_DIM]=
          {DIM(external_force_components[0],external_force_components[1],external_force_components[2])};
          ns->set_external_forces(external_forces);
        }

        // -------------------------------
        // Prepare vectors to receive solution for np1 timestep:
        // -------------------------------

        v_n.destroy();v_n.create(p4est_np1,nodes_np1);
        v_nm1.destroy();v_nm1.create(p4est_np1,nodes_np1);
        vorticity.destroy();vorticity.create(p4est_np1,nodes_np1);
        press_nodes.destroy();press_nodes.create(p4est_np1,nodes_np1);

        // -------------------------------
        // Solve the Navier-Stokes problem:
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Navier-Stokes solution step... \n");

        bool compute_pressure_to_save = false;
        compute_pressure_to_save =
            are_we_saving_vtk(tstep,tn, false,out_idx,false) ||
            are_we_saving_fluid_forces(tn,false,pressure_save_out_idx,true);

        compute_pressure_to_save = compute_pressure_to_save || example_is_a_test_case;
        // Check if we are going to be saving to vtk for the next timestep... if so, we will compute pressure at nodes for saving

        bool did_crash = false;

        navier_stokes_step(ns,p4est_np1,nodes_np1,
                           v_n.vec,v_nm1.vec,vorticity.vec,press_nodes.vec,
                           phi.vec, dxyz_close_to_interface,
                           face_solver_type,pc_face,cell_solver_type,pc_cell,
                           faces_np1, compute_pressure_to_save, did_crash,
                           (save_fluid_forces || example_requires_area_computation)? name_fluid_forces:NULL,
                           (save_fluid_forces || example_requires_area_computation)? fich_fluid_forces:NULL);
        if(did_crash){
          PetscPrintf(mpi.comm(),"Outputting crash files ... \n");
          save_fields_to_vtk(p4est_np1,nodes_np1,ghost_np1,ngbd_np1,
                             0,0,phi,phi_cylinder,T_l_n,T_s_n,v_interface,
                             v_n,press_nodes,vorticity,true);
          MPI_Barrier(mpi.comm());
          MPI_Abort(mpi.comm(),0);
        }

        // -------------------------------
        // Update timestep info as needed
        // -------------------------------
        if(dt_NS>dt_max_allowed) dt_NS = dt_max_allowed;

        // -------------------------------
        // Clear out the interfacial BC for the next timestep, if needed
        // -------------------------------
        if(interfacial_vel_bc_requires_vint){
          for(unsigned char d=0;d<P4EST_DIM;d++){
            bc_interface_value_velocity[d]->clear();
            }
        }

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Completed Navier-Stokes step \n");
      } // End of "if solve navier stokes"

      // If not solving NS but you still want area data:

      if((example_requires_area_computation) && (save_fluid_forces) && (no_flow)){
        // ELYCE DEBUGGING HERE
        // Get total solid domain
        // (including the cylinder bulk -- Note: this will need to be subtracted later, but cyl area is a constant value so no need to compute it over and over):

        //PetscPrintf(mpi.comm(),"We are in here ! ");
        // --> First, need to scale phi
        VecScaleGhost(phi.vec,-1.0);

        // --> Compute area of negative domain (aka ice bulk)
        double ice_area;
        ice_area = area_in_negative_domain(p4est_np1,nodes_np1,phi.vec);

        //--> Scale phi back to normal:
        VecScaleGhost(phi.vec,-1.0);


        PetscPrintf(mpi.comm(),"tn = %g, A = %0.6f \n",tn+dt,ice_area);
        ierr = PetscFOpen(mpi.comm(),name_fluid_forces,"a",&fich_fluid_forces); CHKERRXX(ierr);

        ierr = PetscFPrintf(mpi.comm(),fich_fluid_forces,"%g %g\n",tn+dt,ice_area);CHKERRXX(ierr);
        ierr = PetscFClose(mpi.comm(),fich_fluid_forces); CHKERRXX(ierr);
        PetscPrintf(mpi.comm(),"forces saved \n");

      }




      // --------------------------------------------------------------------------------------------------------------
      // Save simulation state every specified number of iterations
      // --------------------------------------------------------------------------------------------------------------
      if(tstep>0 && ((tstep%save_state_every_iter)==0) && tstep!=load_tstep){
          char output[1000];
          const char* out_dir_coupled = getenv("OUT_DIR_SAVE_STATE");
          if(!out_dir_coupled){
              throw std::invalid_argument("You need to set the output directory for save states: OUT_DIR_SAVE_STATE");
            }
          sprintf(output,
                  "%s/save_states_output_lmin_%d_lmax_%d_advection_order_%d_example_%d",
                  out_dir_coupled,
                  lmin+grid_res_iter,lmax+grid_res_iter,
                  advection_sl_order,example_);

          save_state(mpi,output,num_save_states,
                     &sp,p4est_np1,nodes_np1,
                     phi.vec,T_l_n.vec,T_l_nm1.vec,T_s_n.vec,
                     v_n.vec,v_nm1.vec,vorticity.vec);

          PetscPrintf(mpi.comm(),"Simulation state was saved . \n");
        }

      // --------------------------------------------------------------------------------------------------------------
      // Saving to VTK: either every specified number of iterations, or every specified dt:
      // Note: we do this after extension of fields to make visualization nicer
      // --------------------------------------------------------------------------------------------------------------
      bool are_we_saving = false;

      are_we_saving = are_we_saving_vtk( tstep, tn, tstep==load_tstep, out_idx, true) /*&& (tstep>0)*/;

      // Save to VTK if we are saving this timestep:
      if(are_we_saving){
        save_fields_to_vtk(p4est_np1,nodes_np1,ghost_np1,ngbd_np1,out_idx,grid_res_iter,
                           phi,phi_cylinder,T_l_n,T_s_n,
                           v_interface,
                           v_n,press_nodes,vorticity);
      } // end of if "are we saving"

      // Check errors on validation cases if relevant, save errors to vtk if we are saving this timestep
      if(example_ == NS_GIBOU_EXAMPLE){
          const char* out_dir_ns = getenv("OUT_DIR_VTK");
          char output[1000];
          PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

          sprintf(output,"%s/snapshot_NS_Gibou_test_lmin_%d_lmax_%d_outidx_%d",out_dir_ns,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

          if(tstep>0){
            // In typical saving, only compute pressure nodes when we save to vtk. For this example, save pressure nodes every time so we can check the error
            press_nodes.destroy();press_nodes.create(p4est_np1,nodes_np1);
            ns->compute_pressure_at_nodes(&press_nodes.vec);
          }

          save_navier_stokes_test_case(p4est_np1, nodes_np1, ghost_np1,
                                       phi, v_n, press_nodes, vorticity,
                                       dxyz_close_to_interface, are_we_saving, output,
                                       name_NS_errors, fich_NS_errors);
        }
      if((example_ == COUPLED_PROBLEM_EXAMPLE)|| (example_ == COUPLED_TEST_2)){
        const char* out_dir_coupled = getenv("OUT_DIR_VTK");

        char output[1000];
        PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

        sprintf(output,"%s/snapshot_coupled_test_lmin_%d_lmax_%d_outidx_%d",out_dir_coupled,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

        PetscPrintf(mpi.comm(),"Saving coupled problem example \n");
        save_coupled_test_case(p4est_np1, nodes_np1, ghost_np1, ngbd_np1,
                               phi, T_l_n, T_s_n, v_interface, v_n, press_nodes, vorticity,
                               dxyz_close_to_interface, are_we_saving, output,
                               name_coupled_errors, fich_coupled_errors);
        // Don't check first timestep bc have not computed velocity yet
        PetscPrintf(mpi.comm(),"Coupled test case saved \n");

      }
      if(example_ == FRANK_SPHERE){
          const char* out_dir_stefan = getenv("OUT_DIR_VTK");

          char output[1000];

          sprintf(output,"%s/snapshot_Frank_Sphere_test_lmin_%d_lmax_%d_outidx_%d",out_dir_stefan,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
          PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

          save_stefan_test_case(p4est_np1,nodes_np1,ghost_np1,T_l_n, T_s_n, phi, v_interface, dxyz_close_to_interface,are_we_saving,output,name_stefan_errors,fich_stefan_errors);
        }

      // --------------------------------------------------------------------------------------------------------------
      // Advance the LSF/Update the grid :
      // --------------------------------------------------------------------------------------------------------------
      /* In Coupled case: advect the LSF and update the grid according to vorticity, d2T/dd2, and phi
       * In Stefan case:  advect the LSF and update the grid according to phi
       * In NS case:      update the grid according to phi (no advection)
      */
      // --------------------------------
      // Compute the timestep
      // (needed for the grid advection, and will be used as timestep for np1 step)
      // --------------------------------
      dt_nm1 = dt; // Slide the timestep

      // Compute stefan timestep:
      char stefan_timestep[1000];
      if(solve_stefan){
        compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes_np1,p4est_np1); // this function modifies the variable dt

        sprintf(stefan_timestep,"Computed interfacial velocity: \n"
                                " - Computational : %0.3e  "
                                "- Physical : %0.3e [m/s]  "
                                "- Physical : %0.3f  [mm/s] \n",
                v_interface_max_norm,
                v_interface_max_norm*vel_nondim_to_dim,
                v_interface_max_norm*vel_nondim_to_dim*1000.);
        if(example_ == DENDRITE_TEST){
           PetscPrintf(mpi.comm()," ! vint*d0/alpha_s = %0.4f \n", v_interface_max_norm*vel_nondim_to_dim*d0/alpha_s);
        }
      }

      // Take NS timestep into account if relevant:
      if(solve_navier_stokes && tstep>0){
        // Compute the corresponding timestep:
        ns->compute_dt(/*NS_norm*/);
        dt_NS = ns->get_dt();


        // Determine the timestep depending on timestep restrictions from both NS solver and from the Stefan problem
        if(solve_stefan){
            if(tstep==load_tstep){dt_NS=dt_nm1;} // TO-DO: not sure this logic is 100% correct, what about NS only case?
            dt = min(dt_Stefan,dt_NS);
            dt = min(dt, dt_max_allowed);
          }
        else{
            // If we are only solving Navier Stokes
            dt = min(dt_NS, dt_max_allowed);
          }
      }
      // If only solving Stefan problem:
      if(solve_stefan && !solve_navier_stokes){
        dt = dt_Stefan;
      }

      PetscPrintf(mpi.comm(),"\n"
                             "%s \n"
                             "Computed timestep: \n"
                             " - dt used: %0.3e "
                             " - dt_Stefan: %0.3e "
                             " - dt_NS : %0.3e  "
                             " - dxyz close to interface : %0.3e "
                             "\n \n",solve_stefan?stefan_timestep:"",
                            dt, dt_Stefan, dt_NS,
                            dxyz_close_to_interface);

      // Clip the timestep if we are near the end of our simulation, to get the proper end time:
      if((tn + dt > tfinal) && (last_tstep<0)){

          dt = max(tfinal - tn,dt_min_allowed);

          // if time remaining is too small for one more step, end here. otherwise, do one more step and clip timestep to end on exact ending time
          if(fabs(tfinal-tn)>dt_min_allowed){
            last_tstep = tstep+1;
          }
          else{
            last_tstep = tstep;
          }

          PetscPrintf(mpi.comm(),"Final tstep will be %d \n",last_tstep);
        }

      // Clip time and switch vel direction for coupled problem example:
      if((example_ == COUPLED_PROBLEM_EXAMPLE)|| (example_ == COUPLED_TEST_2)){
        if(((tn+dt) >= tfinal/2.0) && !vel_has_switched){
          if((tfinal/2. - tn)>dt_min_allowed){ // if we have some uneven situation
            PetscPrintf(mpi.comm(),"uneven situation \n");
            dt = (tfinal/2.) - tn;
          }
          PetscPrintf(mpi.comm(),"SWITCH SIGN : %0.1f \n",coupled_test_sign);
          coupled_test_switch_sign();
          vel_has_switched=true;
          PetscPrintf(mpi.comm(),"SWITCH SIGN : %0.1f \n dt : %e \n",coupled_test_sign,dt);

        }
      }

      if(tstep!=last_tstep){
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning grid update process ... \n"
                                                     "Refine by d2T = %s \n",refine_by_d2T? "true": "false");

        // --------------------------------
        // Destroy p4est at n and slide grids:
        // -----------------------------------
        p4est_destroy(p4est);
        p4est_ghost_destroy(ghost);
        p4est_nodes_destroy(nodes);
        delete ngbd;
        delete hierarchy;

        p4est = p4est_np1;
        ghost = ghost_np1;
        nodes = nodes_np1;

        hierarchy = hierarchy_np1;
        ngbd = ngbd_np1;

        // -------------------------------
        // Create the new p4est at time np1:
        // -------------------------------
        p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
        ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est_np1,ghost_np1);
        nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // Get the new neighbors: // TO-DO : no need to do this here, is there ?
        hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
        ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

        // Initialize the neigbors:
        ngbd_np1->init_neighbors();


        if(solve_navier_stokes && (tstep>1)){
          ns->nullify_p4est_nm1(); // the nm1 grid has just been destroyed, but pointer within NS has not been updated, so it needs to be nullified (p4est_nm1 in NS == p4est in main)
        }
        // -------------------------------
        // Create the semi-lagrangian object and do the advection/grid update:
        // -------------------------------
        // If solving NS, save the previous LSF to provide to NS solver, to correctly
        // interpolate hodge variable to new grid

        if(solve_navier_stokes){
          phi_nm1.create(p4est,nodes);
          ierr = VecCopyGhost(phi.vec,phi_nm1.vec); CHKERRXX(ierr); //--> this will need to be provided to NS update_from_tn_to_tnp1_grid_external
          // Note: this is done because the update_p4est destroys the old LSF, but we need to keep it
          // for NS update procedure
          if(print_checkpoints) ierr= PetscPrintf(mpi.comm(),"Phi nm1 copy is created ... \n");
        }

        my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

        update_the_grid(sl,sp,
                        p4est_np1,nodes_np1,ghost_np1,
                        p4est,nodes,
                        phi,v_interface,
                        phi_cylinder,phi_dd,
                        vorticity,vorticity_refine,
                        T_l_n,T_l_dd,
                        ngbd);

        // -------------------------------
        // Update hierarchy and neighbors to match new updated grid:
        // -------------------------------

        hierarchy_np1->update(p4est_np1,ghost_np1);
        ngbd_np1->update(hierarchy_np1,nodes_np1);

        // Initialize the neigbors:
        ngbd_np1->init_neighbors();

        // -------------------------------
        // Reinitialize the LSF on the new grid (if it has been advected):
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Reinitializing LSF... \n");

        //      my_p4est_level_set_t ls_new(ngbd_np1);
        ls.update(ngbd_np1);

        // Okay, time to reinitialize in a clever way depending on the scenario:

        if(solve_stefan){
          // If interface velocity *is* forced to zero, we do not reinitialize -- that way we don't degrade the LSF through unnecessary reinitializations
          if(!force_interfacial_velocity_to_zero){
            // If we do need to reinitialize, let's think about the time scales
            if(solve_navier_stokes && tstep>0){
              // If we are solving the coupled problem -- let's check the time scales of the interfacial velocity versus the fluid one.
              // If fluid velocity is much larger than interfacial velocity, may not need to reinitialize as much
              // (bc the timestepping is much smaller than necessary for the interface growth, and we don't want the reinitialization to govern more of the interface change than the actual physical interface change)

              if((tstep % reinit_every_iter) == 0){
                ls.reinitialize_2nd_order(phi.vec,30);
                PetscPrintf(mpi.comm(), "reinit every iter =%d, LSF was reinitialized \n", reinit_every_iter);
              }
            }
            else{
              // if just stefan, go ahead and reinitialize
              ls.reinitialize_2nd_order(phi.vec,30);
            }
          }
        }

        if(solve_navier_stokes && !solve_stefan){
          // If only solving Navier-Stokes, only need to do this once, not every single timestep
          if(tstep==0)ls.reinitialize_2nd_order(phi.vec,30);
        }

        if(example_ == DENDRITE_TEST){
          regularize_front(p4est_np1,nodes_np1,ngbd_np1,phi); // ELYCE DEBUGGING: commented this out
        }

        // --------------------------------------------------------------------------------------------------------------
        // Interpolate Values onto New Grid:
        // -------------------------------------------------------------------------------------------------------------

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Interpolating fields to new grid ... \n");
        if(tstep==0){
          // Slide fields
          if(solve_navier_stokes){
            v_nm1.destroy();v_nm1.create(p4est,nodes);
            foreach_dimension(d){
              ierr = VecCopyGhost(v_n.vec[d],v_nm1.vec[d]); CHKERRXX(ierr);
            }
          }
          if(solve_stefan){
            T_l_nm1.destroy(); T_l_nm1.create(p4est,nodes);
            ierr = VecCopyGhost(T_l_n.vec,T_l_nm1.vec);CHKERRXX(ierr);
          }

        } // end of "if tstep ==0"

        interpolate_values_onto_new_grid(&T_l_n.vec,&T_s_n.vec,
                                         v_interface.vec,v_n.vec,
                                         nodes_np1,p4est_np1,ngbd,interp_bw_grids);
      } // end of "if tstep !=last tstep"

      // -------------------------------
      // Do a memory safety check as user specified
      // -------------------------------
      PetscLogDouble mem_safety_check;
      if((check_mem_every_iter>0) && (tstep%check_mem_every_iter)==0){
        MPI_Barrier(mpi.comm());
        PetscMemoryGetCurrentUsage(&mem_safety_check);


        int no = nodes->num_owned_indeps;
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

        // Output file for NS test case errors:
        const char* out_dir_mem = getenv("OUT_DIR_FILES");
        if(!out_dir_mem){
            throw std::invalid_argument("You need to set the environment variable OUT_DIR_FILES to save memory usage info");
          }
        FILE* fich_mem;
        char name_mem[1000];
        sprintf(name_mem,"%s/memory_check_Re_%0.2f_lmin_%d_lmax_%d_advection_order_%d.dat",
                out_dir_mem,Re,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

        if(tstep==0){
          ierr = PetscFOpen(mpi.comm(),name_mem,"w",&fich_mem); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_mem,"tstep mem vtk_bool\n"
                                                  "%d %g %d \n",tstep,mem_safety_check,are_we_saving);CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);
        }
        else{
          ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_mem,"%d %g %d \n",tstep,mem_safety_check,are_we_saving);CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);

        }
      }

      // -------------------------------
      // Update time: // trying something
      // -------------------------------
      if(tstep==0){dt_nm1 = dt;}
      tn+=dt;
      tstep++;
    } // <-- End of for loop through time

  PetscPrintf(mpi.comm(),"Time loop exited \n");

  // Final destructions: TO-DO: need to revisit these, make sure they're done correctly
  if(solve_stefan){
    T_l_n.destroy();
    T_s_n.destroy();
    v_interface.destroy();


    if(advection_sl_order==2) T_l_nm1.destroy();

    // Destroy relevant BC and RHS info:
    for(unsigned char d=0;d<2;++d){
      if(analytical_IC_BC_forcing_term){
        delete analytical_T[d];
        delete external_heat_source_T[d];
      }
      if(interfacial_temp_bc_requires_curvature){
        // cases where we used curvature, want to clear interpolator before destroying
        bc_interface_val_temp[d]->clear();
      }
      delete bc_interface_val_temp[d];
      delete bc_wall_value_temp[d];
    }

    if(!solve_navier_stokes){
      phi.destroy();

      // destroy the structures leftover (in non NS case)
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_destroy      (p4est);

      p4est_nodes_destroy(nodes_np1);
      p4est_ghost_destroy(ghost_np1);
      p4est_destroy(p4est_np1);

      my_p4est_brick_destroy(conn, &brick);
      delete hierarchy;
      delete ngbd;

      delete hierarchy_np1;
      delete ngbd_np1;
    }
  }

  if(solve_navier_stokes){
    v_n.destroy();
    v_nm1.destroy();

    phi_nm1.destroy();

    // NS takes care of destroying v_NS_n and v_NS_nm1
    vorticity.destroy();
    press_nodes.destroy();

    for(unsigned char d=0;d<P4EST_DIM;d++){
      if(analytical_IC_BC_forcing_term){
        delete analytical_soln_v[d];
        delete external_force_components[d];
      }

      delete bc_interface_value_velocity[d];
      delete bc_wall_value_velocity[d];
      delete bc_wall_type_velocity[d];
    }

    delete ns;
  }
  }// end of loop through number of splits

  MPI_Barrier(mpi.comm());
  w.stop(); w.read_duration();
  return 0;
}

