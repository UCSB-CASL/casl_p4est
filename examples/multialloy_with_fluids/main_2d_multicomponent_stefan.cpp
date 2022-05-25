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
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/erf.hpp>


// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_multialloy.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_stefan_with_fluids.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_macros.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_multialloy.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_stefan_with_fluids.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/point3.h>

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

enum: int {
    BASIC_MULTI_ALLOY =0
};
DEFINE_PARAMETER(pl, int, example_, 0,"example number: \n"
                                   "0 - Basic Multi Alloy w Fluids \n");
                                
// ---------------------------------------
// Save options:
// ---------------------------------------
// Options for saving to vtk:
DEFINE_PARAMETER(pl, bool, save_to_vtk, true, "We save vtk files using a "
                                              "given dt increment if this is set to true \n");

DEFINE_PARAMETER(pl, bool, save_vtk_solid , true, "Save the vtk for solid \n");

DEFINE_PARAMETER(pl, bool, save_vtk_analytical , false, "Save vtk analytical \n");

DEFINE_PARAMETER(pl, bool, save_using_dt, false, "We save vtk files using a "
                                                 "given dt increment if this is set to true \n");

DEFINE_PARAMETER(pl, bool, save_using_iter, false, "We save every prescribed number "
                                                   "of iterations if this is set to true \n");

DEFINE_PARAMETER(pl, int, save_every_iter, 1, "Saves vtk every n number "
                                              "of iterations (default is 1)");

DEFINE_PARAMETER(pl, double, save_every_dt, 1, "Saves vtk every dt amount of "
                                               "time in seconds of dimensional time (default is 1)");

// Options to compute and save fluid forces to a file:
DEFINE_PARAMETER(pl, bool, save_fluid_forces, false, "Saves fluid forces if true (default: false) \n");

DEFINE_PARAMETER(pl, bool, save_area_data, false, "Save area data if true (default: false, but some examples will set this to true automatically)");

DEFINE_PARAMETER(pl, double, save_data_every_dt, 0.01, "Saves fluid forces and/or area data every dt amount of time in seconds of dimensional time (default is 1.0) \n");

// Options to track and output island numbers for an evolving geometry:
DEFINE_PARAMETER(pl, bool, track_evolving_geometries, false, "Flag to track island numbers for the evolving geometry(ies) and output the island numbers to vtk and other information to a file in the same folder as the vtk info. Default: false. For use with the evolving porous media problem. \n");

DEFINE_PARAMETER(pl, bool, wrap_up_simulation_if_solid_has_vanished, false, "If set to true, the simulation will check if the solid region has vanished and pause the simulation at that point \n");

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

DEFINE_PARAMETER(pl, bool, print_checkpoints, false, "Print checkpoints throughout script for debugging? ");


// ---------------------------------------
// Solution options:
// ---------------------------------------
// Related to which physics we solve:
DEFINE_PARAMETER(pl, bool, solve_stefan, false, "Solve stefan ?");
DEFINE_PARAMETER(pl, bool, solve_navier_stokes, false, "Solve navier stokes?");
DEFINE_PARAMETER(pl, bool, solve_coupled, true, "Solve the coupled problem?"); // <-- get rid of

DEFINE_PARAMETER(pl, bool, do_we_solve_for_Ts, false, "True/false to describe whether or not we solve for the solid temperature (or concentration). Default: false. This is set to true for select examples in select_solvers()\n");
DEFINE_PARAMETER(pl, bool, use_boussinesq, false, "True/false to describe whether or not we are solving the problem considering natural convection effects using the boussinesq approx. Default: false. This is set true for the dissolving disk benchmark case. This is used to distinguish the dissolution-specific stefan condition, as contrasted with other concentration driven problems in solidification. \n");

DEFINE_PARAMETER(pl, bool, use_regularize_front, false, "True/false to describe whether or not we use Daniil's algorithm for smoothing problem geometries and bridging gaps of a certain proximity. Default:false \n");

DEFINE_PARAMETER(pl, bool, use_collapse_onto_substrate, false, "True/false to describe whether or not we use algorithm for collapsing interface onto a substrate within a certain proximity. Default:false \n");

DEFINE_PARAMETER(pl, double, proximity_smoothing, 2.5, "Parameter for front regularization. default: 2.5 \n");
DEFINE_PARAMETER(pl, double, proximity_collapse, 3.0, "Parameter for collapse onto front. default: 3.0 \n");

DEFINE_PARAMETER(pl, int, nondim_type_used, -1., "Integer value to overwrite the nondimensionalization type used for a given problem. The default is -1. If this is specified to a nonnegative number, it will overwrite the particular example's default. 0 - nondim by fluid velocity, 1 - nondim by diffusivity (thermal or conc), 2 - dimensional.  \n");

// Related to the Stefan and temperature/concentration problem:
DEFINE_PARAMETER(pl, double, cfl, 0.5, "CFL number for Stefan problem (default:0.5)");
DEFINE_PARAMETER(pl, int, advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid temperature field(default:2)");
DEFINE_PARAMETER(pl, bool, force_interfacial_velocity_to_zero, false, "Force the interfacial velocity to zero? ");

// Related to the Navier-Stokes problem:
//DEFINE_PARAMETER(pl, double, Re_overwrite, -100.0, "Overwrite the examples set Reynolds number (works if set to a positive number, default:-100.00");
DEFINE_PARAMETER(pl, int, NS_advection_sl_order, 2, "Integer for advection solution order (can choose 1 or 2) for the fluid velocity fields (default:1)");
DEFINE_PARAMETER(pl, double, cfl_NS, 1.0, "CFL number for Navier-Stokes problem (default:1.0)");
DEFINE_PARAMETER(pl, double, hodge_tolerance, 1.e-3, "Tolerance on hodge for error convergence (default:1.e-3)");

// Related to simulation duration settings:
DEFINE_PARAMETER(pl, double, duration_overwrite, -100.0, "Overwrite the duration in minutes (works if set to a positive number, default:-100.0");
DEFINE_PARAMETER(pl, double, duration_overwrite_nondim, -10.,"Overwrite the duration in nondimensional time (in nondimensional time) -- not fully implemented");

// ---------------------------------------
// Booleans that we select to simplify logic in the main program for different processes that are required for different examples:
// ---------------------------------------

bool analytical_IC_BC_forcing_term;
bool example_is_a_test_case;

bool interfacial_temp_bc_requires_curvature;
bool interfacial_temp_bc_requires_normal;

bool interfacial_vel_bc_requires_vint;

bool example_uses_inner_LSF;


bool example_has_known_max_vint;

double max_vint_known_for_ex = 1.0;
unsigned int num_fields_interp = 0;

void select_solvers(){
    switch(example_){
        case BASIC_MULTI_ALLOY:
            solve_stefan = true;
            solve_navier_stokes = false;
        break;
    }
    if(save_using_dt && save_using_iter){
      throw std::invalid_argument("You have selected to save using dt and using iteration, you need to select only one \n");
    }
    interfacial_temp_bc_requires_curvature = (example_==BASIC_MULTI_ALLOY);

    interfacial_vel_bc_requires_vint = (example_==BASIC_MULTI_ALLOY);
}

// ---------------------------------------
// Refinement options:
// ---------------------------------------

DEFINE_PARAMETER(pl, double, vorticity_threshold, 0.1,"Threshold to refine vorticity by, default is 0.1 \n");
DEFINE_PARAMETER(pl, double, gradT_threshold, 1.e-4,"Threshold to refine the nondimensionalized temperature gradient by \n (default: 0.99)");
DEFINE_PARAMETER(pl, bool, use_uniform_band, true, "Boolean whether or not to refine using a uniform band");
DEFINE_PARAMETER(pl, double, uniform_band, 8., "Uniform band (default:8.)");
DEFINE_PARAMETER(pl, double, dxyz_close_to_interface_mult, 1.2, "Multiplier that defines dxyz_close_to_interface = mult* max(dxyz_smallest)");

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

DEFINE_PARAMETER(pl, int, num_splits, 0, "Number of splits -- used for convergence tests");
DEFINE_PARAMETER(pl, bool, refine_by_ucomponent, false, "Flag for whether or not to refine by a backflow condition for the fluid velocity");
DEFINE_PARAMETER(pl, bool, refine_by_d2T, true, "Flag for whether or not to refine by the nondimensionalized temperature gradient");

DEFINE_PARAMETER(pl, double, dendrite_cut_off_fraction, 1.05, "Dendrite cut off fraction");
DEFINE_PARAMETER(pl, double, dendrite_min_length, 0.05, "Dendrite minimum length");

// ---------------------------------------
// Alloy Parameters:
// ---------------------------------------
DEFINE_PARAMETER(pl, double, density_l, 8.88e-3, "Density of liq. alloy (kg.cm-3)" );
DEFINE_PARAMETER(pl, double, density_s, 8.88e-3, "Density of sol. alloy (kg.cm-3)" );

DEFINE_PARAMETER(pl, double, heat_capacity_l, 0.46e3, "Heat capacity of liq. alloy (J.kg-1.K-1)" );
DEFINE_PARAMETER(pl, double, heat_capacity_s, 0.46e3, "Heat capacity of sol. alloy (J.kg-1.K-1)" );

DEFINE_PARAMETER(pl, double, thermal_cond_l, 6.07e-1, "Thermal conductivity of liq. alloy (W.cm-1.K-1)" );
DEFINE_PARAMETER(pl, double, thermal_cond_s, 6.07e-1, "Thermal conductivity of sol. alloy (W.cm-1.K-1)" );

DEFINE_PARAMETER(pl, double, latent_heat, 2350, "Latent heat of fusion (J.cm-3)");

DEFINE_PARAMETER(pl, bool, linearized_liquidus, 0, "Use linearized liquidus surface or true one");

DEFINE_PARAMETER(pl, bool, const_part_coeff, 0, "Use averaged partition coefficients or true ones");

DEFINE_PARAMETER(pl, int , num_comps, 1, "Number of solutes" );

DEFINE_PARAMETER(pl, double, initial_conc_0, 0.4, "Initial concentration of component no. 0");
DEFINE_PARAMETER(pl, double, initial_conc_1, 0.4, "Initial concentration of component no. 1");
DEFINE_PARAMETER(pl, double, initial_conc_2, 0.4, "Initial concentration of component no. 2");
DEFINE_PARAMETER(pl, double, initial_conc_3, 0.4, "Initial concentration of component no. 3");

double* initial_conc_all[] = {&initial_conc_0  ,
                              &initial_conc_1  ,
                              &initial_conc_2  ,
                              &initial_conc_3  };

DEFINE_PARAMETER(pl, double, solute_diff_0, 1.0e-5, "Initial concentration of component no. 0");
DEFINE_PARAMETER(pl, double, solute_diff_1, 1.0e-5, "Initial concentration of component no. 1");
DEFINE_PARAMETER(pl, double, solute_diff_2, 1.0e-5, "Initial concentration of component no. 2");
DEFINE_PARAMETER(pl, double, solute_diff_3, 1.0e-5, "Initial concentration of component no. 3");


double* solute_diff_all[] = {&solute_diff_0  ,
                             &solute_diff_1  ,
                             &solute_diff_2  ,
                             &solute_diff_3  };

DEFINE_PARAMETER(pl, double, eps_c, 0, "Curvature undercooling coefficient cm.K");
DEFINE_PARAMETER(pl, double, eps_v, 0, "Kinetic undercooling coefficient s.K.cm-1" );
DEFINE_PARAMETER(pl, double, eps_a, 0, "Anisotropy coefficient" );
DEFINE_PARAMETER(pl, double, symmetry, 4 , "Symmetric of crystals");

DEFINE_PARAMETER(pl, double, liquidus_slope_0, -357, "Slope of linearized liqiudus w.r.t component no. 0 (pl, K^-1)");
DEFINE_PARAMETER(pl, double, liquidus_slope_1, -357, "Slope of linearized liqiudus w.r.t component no. 1 (pl, K^-1)");
DEFINE_PARAMETER(pl, double, liquidus_slope_2, -357, "Slope of linearized liqiudus w.r.t component no. 2 (pl, K^-1)");
DEFINE_PARAMETER(pl, double, liquidus_slope_3, -357, "Slope of linearized liqiudus w.r.t component no. 3 (pl, K^-1)");

double* liquidus_slope_all[] = { &liquidus_slope_0  ,
                                 &liquidus_slope_1  ,
                                 &liquidus_slope_2  ,
                                 &liquidus_slope_3   };

DEFINE_PARAMETER(pl, double, part_coeff_0, 0.86, "Partition coefficient for component no. 0");
DEFINE_PARAMETER(pl, double, part_coeff_1, 0.86, "Partition coefficient for component no. 1");
DEFINE_PARAMETER(pl, double, part_coeff_2, 0.86, "Partition coefficient for component no. 2");
DEFINE_PARAMETER(pl, double, part_coeff_3, 0.86, "Partition coefficient for component no. 3");

double* part_coeff_all[]={&part_coeff_0  ,
                          &part_coeff_1  ,
                          &part_coeff_2  ,
                          &part_coeff_3  };

DEFINE_PARAMETER(pl, double, melting_temp, 1728, "Pure-substance melting point for linearized slope (K)");

DEFINE_PARAMETER(pl, int , alloy, 2, "0: Ni -  0.4at%Cu bi-alloy, "
                                    "1: Ni -  0.2at%Cu -  0.2at%Cu tri-alloy, "
                                    "2: Co - 10.7at%W  -  9.4at%Al tri-alloy, "
                                    "3: Co -  9.4at%Al - 10.7at%W  tri-alloy, "
                                    "4: Ni - 15.2wt%Al -  5.8wt%Ta tri-alloy, "
                                    "5: Ni -  5.8wt%Ta - 15.2wt%Al tri-alloy, "
                                    "6: a made-up tetra-alloy, "", "
                                    "7: a made-up penta-alloy");

double scale = 30;
// Rochi need to deal with this parameter scale
DEFINE_PARAMETER(pl, double, cooling_velocity, 0.001*scale, "Cooling velocty (cm/s)");
DEFINE_PARAMETER(pl, double, gradient_ratio, 0.75, "Ratio of compositional and thermal gradients at the front");
DEFINE_PARAMETER(pl, double, temp_gradient, 500, "Temperature gradient (K/cm)");
DEFINE_PARAMETER(pl, double, start_from_moving_front, 0, "Relevant only for geometry ==0");

DEFINE_PARAMETER(pl, int, smoothstep_order, 5, "Smoothness of cooling/heating");
DEFINE_PARAMETER(pl, double, starting_time, 0.e-3, "Time for cooling or heating to fully switch on (s)" );

DEFINE_PARAMETER(pl, BoundaryConditionType, bc_type_conc, NEUMANN, "DIRICHLET/NEUMANN" );
DEFINE_PARAMETER(pl, BoundaryConditionType, bc_type_temp, NEUMANN, "DIRICHLET/NEUMANN" );

DEFINE_PARAMETER(pl, int, step_limit, INT_MAX, "Step limit");
DEFINE_PARAMETER(pl, double, step_limit, DBL_MAX, "Timelimit");
DEFINE_PARAMETER(pl, double, growth_limit, 15, "Growth limit");
DEFINE_PARAMETER(pl, double, init_perturb, 1.e-10, "Init_perturb");
DEFINE_PARAMETER(pl, bool, enforce_planar_front, 0, "Enforce planar front");
DEFINE_PARAMETER(pl, double, front_location, 0.1, "Front location");
DEFINE_PARAMETER(pl, double, front_location_final, 0.25, "Front location final");
DEFINE_PARAMETER(pl, double, container_radius_inner, 0.1, "Container radius inner");
DEFINE_PARAMETER(pl, double, container_radius_outer, 0.45, "Container radius outer");
DEFINE_PARAMETER(pl, double, seed_raidus, 0.005, "Seed radius");
DEFINE_PARAMETER(pl, double, seed_dist, 0.1, "Seed dist");
DEFINE_PARAMETER(pl, double, seed_rot, PI/12., "Seed rot");
DEFINE_PARAMETER(pl, double, crystal_orientation, 0.*PI/6., "Crystal orientation");
DEFINE_PARAMETER(pl, double, seed_type, 0, "Seed type : 0- Aligned; 1- Misaligned");
DEFINE_PARAMETER(pl, double, box_size, 0.0075, "Physical width (in x) of the box in cm");

DEFINE_PARAMETER(pl, int, 0, "-3 - analytical spherical solidification,"
                             "-2 - analytical cylindrical solidification,"
                             "-1 - analytical planar solidification,"
                             " 0 - direcitonal solidification,"
                             " 1 - growth of a spherical seed in a spherical container,"
                             " 2 - growth of a spherical film in a spherical container," 
                             " 3 - radical directional solidification in,"
                             " 4 - radical directional solidification out,"
                             " 5 - three spherical seeds,"
                             " 6 - planar front and three spherical seeds");
                        
void set_alloy_parameters()
{
  switch (alloy())
  {
    case 0: // Ni - 0.4at%Cu
      density_l         = 8.88e-3; // kg.cm-3
      density_s         = 8.88e-3; // kg.cm-3
      heat_capacity_l   = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s   = 0.46e3;  // J.kg-1.K-1
      thermal_cond_l    = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s    = 6.07e-1; // W.cm-1.K-1
      latent_heat       = 2350;    // J.cm-3

      num_comps   = 1;

      solute_diff_0      = 1.e-3;  // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.4;    // at frac.

      eps_c   = 0;
      eps_v   = 0;
      eps_a   = 0.05;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1728;   // K
      liquidus_slope_0   = -357;   // K / at frac. - liquidous slope
      part_coeff_0       = 0.86;   // partition coefficient

      break;

    case 1: // Ni - 0.2at%Cu - 0.2at%Cu
      density_l         = 8.88e-3; // kg.cm-3
      density_s         = 8.88e-3; // kg.cm-3
      heat_capacity_l   = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s   = 0.46e3;  // J.kg-1.K-1
      thermal_cond_l    = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s    = 6.07e-1; // W.cm-1.K-1
      latent_heat       = 2350;    // J.cm-3

      num_comps   = 2;

      solute_diff_0      = 1.e-5;  // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1      = 5.e-5;  // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.2;    // at frac.
      initial_conc_1     = 0.2;    // at frac.

      eps_c   = 4.e-5;
      eps_v   = 0.0;
      eps_a   = 0.00;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1728;   // K
      liquidus_slope_0   = -357;   // K / at frac. - liquidous slope
      liquidus_slope_1   = -357;   // K / at frac. - liquidous slope
      part_coeff_0       = 0.86;   // partition coefficient
      part_coeff_1       = 0.86;   // partition coefficient
      break;

    case 2: // Co - 10.7at%W - 9.4at%Al (more realistic since D_W < D_Al)
      density_l         = 9.24e-3; // kg.cm-3
      density_s         = 9.24e-3; // kg.cm-3
      heat_capacity_l   = 356;       // J.kg-1.K-1
      heat_capacity_s   = 356;       // J.kg-1.K-1
      thermal_cond_l    = 1.3;       // W.cm-1.K-1
      thermal_cond_s    = 1.3;       // W.cm-1.K-1
      latent_heat       = 2600;    // J.cm-3

      num_comps   = 2;

      solute_diff_0      = 1e-5;     // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1      = 2e-5;     // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.107;    // at frac.
      initial_conc_1     = 0.094;    // at frac.

      eps_c   = 1.0e-5/1.;
      eps_v   = 0.0e-2;
      eps_a   = 0.05*1.;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1910;      // K
      liquidus_slope_0   =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1   =-1036;     // K / at frac. - liquidous slope
      part_coeff_0       = 0.83;    // partition coefficient
      part_coeff_1       = 0.83;    // partition coefficient

      melting_temp      = 1910;      // K
      liquidus_slope_0   =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1   =-1036;     // K / at frac. - liquidous slope
      part_coeff_0       = 0.94;    // partition coefficient
      part_coeff_1       = 0.83;    // partition coefficient

      break;

    case 3: // Co - 9.4at%Al - 10.7at%W
      density_l         = 9.24e-3; // kg.cm-3
      density_s         = 9.24e-3; // kg.cm-3
      heat_capacity_l   = 356;       // J.kg-1.K-1
      heat_capacity_s   = 356;       // J.kg-1.K-1
      thermal_cond_l    = 1.3;       // W.cm-1.K-1
      thermal_cond_s    = 1.3;       // W.cm-1.K-1
      latent_heat       = 2600;    // J.cm-3

      num_comps   = 2;

      solute_diff_0      = 2e-5;     // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1      = 1e-5;     // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.094;    // at frac.
      initial_conc_1     = 0.107;    // at frac.

      eps_c   = 1.0e-5;
      eps_v   = 1.0e-2;
      eps_a   = 0.05;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1910;      // K
      liquidus_slope_0   =-1036;     // K / at frac. - liquidous slope
      liquidus_slope_1   =-543;      // K / at frac. - liquidous slope
      part_coeff_0       = 0.83;    // partition coefficient
      part_coeff_1       = 0.94;    // partition coefficient
      break;

    case 4: // Ni - 15.2wt%Al - 5.8wt%Ta
      density_l         = 7.365e-3; // kg.cm-3
      density_s         = 7.365e-3; // kg.cm-3
      heat_capacity_l   = 660;      // J.kg-1.K-1
      heat_capacity_s   = 660;      // J.kg-1.K-1
      thermal_cond_l    = 0.8;      // W.cm-1.K-1
      thermal_cond_s    = 0.8;      // W.cm-1.K-1
      latent_heat       = 2136;     // J.cm-3

      num_comps   = 2;

      solute_diff_0      = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1      = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.152;   // wt frac.
      initial_conc_1     = 0.058;   // wt frac.

      eps_c   = 0*2.7207e-5;
      eps_v   = 0*2.27e-2;
      eps_a   = 0.05;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1754;     // K
      liquidus_slope_0   =-255;     // K / wt frac. - liquidous slope
      liquidus_slope_1   =-517;     // K / wt frac. - liquidous slope
      part_coeff_0       = 0.48;    // partition coefficient
      part_coeff_1       = 0.54;    // partition coefficient
      break;

    case 5: // Ni - 5.8wt%Ta - 15.2wt%Al
      density_l         = 7.365e-3; // kg.cm-3
      density_s         = 7.365e-3; // kg.cm-3
      heat_capacity_l   = 660;      // J.kg-1.K-1
      heat_capacity_s   = 660;      // J.kg-1.K-1
      thermal_cond_l    = 0.8;      // W.cm-1.K-1
      thermal_cond_s    = 0.8;      // W.cm-1.K-1
      latent_heat       = 2136;     // J.cm-3

      num_comps   = 2;

      solute_diff_0      = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1      = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0     = 0.058;   // wt frac.
      initial_conc_1     = 0.152;   // wt frac.

      eps_c   = 0*2.7207e-5;
      eps_v   = 0*2.27e-2;
      eps_a   = 0.05;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1754;     // K
      liquidus_slope_0   =-517;     // K / wt frac. - liquidous slope
      liquidus_slope_1   =-255;     // K / wt frac. - liquidous slope
      part_coeff_0       = 0.54;    // partition coefficient
      part_coeff_1       = 0.48;    // partition coefficient
      break;

    case 6: // A made-up tetra-alloy based on Ni - 0.4at%Cu
      density_l         = 8.88e-3; // kg.cm-3
      density_s         = 8.88e-3; // kg.cm-3
      heat_capacity_l   = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s   = 0.46e3;  // J.kg-1.K-1
      latent_heat       = 2350;    // J.cm-3
      thermal_cond_l    = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s    = 6.07e-1; // W.cm-1.K-1

      num_comps   = 3;
      solute_diff_0      = 1.e-5;
      solute_diff_1      = 2.e-5;
      solute_diff_2      = 4.e-5;
      initial_conc_0     = 0.1;
      initial_conc_1     = 0.1;
      initial_conc_2     = 0.1;

      eps_c   = 0.e-6;
      eps_v   = 0;
      eps_a   = 0.0;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp       = 1728;    // K
      liquidus_slope_0   = -300;
      liquidus_slope_1   = -500;
      liquidus_slope_2   = -400;
      part_coeff_0       = 0.85;
      part_coeff_1       = 0.75;
      part_coeff_2       = 0.90;
      break;

    case 7: // A made-up penta-alloy based on Ni - 0.4at%Cu
      density_l         = 8.88e-3; // kg.cm-3
      density_s         = 8.88e-3; // kg.cm-3
      heat_capacity_l   = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s   = 0.46e3;  // J.kg-1.K-1
      latent_heat       = 2350;    // J.cm-3
      thermal_cond_l    = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s    = 6.07e-1; // W.cm-1.K-1

      num_comps   = 4;
      solute_diff_0      = 1.e-5;
      solute_diff_1      = 2.e-5;
      solute_diff_2      = 4.e-5;
      solute_diff_3      = 6.e-5;
      initial_conc_0     = 0.15;
      initial_conc_1     = 0.1;
      initial_conc_2     = 0.05;
      initial_conc_3     = 0.05;

      eps_c   = 1.e-5;
      eps_v   = 0;
      eps_a   = 0.05;
      symmetry   = 4;

      // linearized phase diagram
      melting_temp      = 1728;    // K

      liquidus_slope_0   = -300;
      liquidus_slope_1   = -500;
      liquidus_slope_2   = -400;
      liquidus_slope_3   = -600;

      part_coeff_0       = 0.85;
      part_coeff_1       = 0.75;
      part_coeff_2       = 0.90;
      part_coeff_3       = 0.80;

      break;
    default:
      throw std::invalid_argument("Undefined alloy\n");
  }
}

double liquidus_value(double *c)
{
  static double conc_term;
  if (linearized_liquidus   )
  {
    conc_term = melting_temp   ;

    for (int i = 0; i < num_comps   ; ++i) conc_term += (*liquidus_slope_all[i])*c[i];

    return conc_term;
  }
  else
  {
    switch (alloy   )
    {
      case 0: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 1: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 2: // Co-W-Al
      {
        double c0 = 100.*c[0];
        double c1 = 100.*c[1];
        static double p00 =  1.767e+03;
        static double p10 =  2.369e+00;
        static double p01 =  1.771e+00;
        static double p20 = -2.238e-01;
        static double p11 =  1.041e-01;
        static double p02 = -3.046e-01;
        static double p30 =  1.358e-03;
        static double p21 = -1.568e-02;
        static double p12 = -1.457e-02;
        static double p03 =  3.317e-03;
        static double p40 = -3.283e-06;
        static double p31 =  3.559e-04;
        static double p22 =  1.228e-04;
        static double p13 =  1.531e-04;
        static double p04 = -1.866e-04;
        return p00
            + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
            + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
            + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
            + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
      }
      case 3: // Co-Al-W
      {
        double c0 = 100.*c[1];
        double c1 = 100.*c[0];
        static double p00 =  1.767e+03;
        static double p01 =  2.369e+00;
        static double p10 =  1.771e+00;
        static double p02 = -2.238e-01;
        static double p11 =  1.041e-01;
        static double p20 = -3.046e-01;
        static double p03 =  1.358e-03;
        static double p12 = -1.568e-02;
        static double p21 = -1.457e-02;
        static double p30 =  3.317e-03;
        static double p04 = -3.283e-06;
        static double p13 =  3.559e-04;
        static double p22 =  1.228e-04;
        static double p31 =  1.531e-04;
        static double p40 = -1.866e-04;
        return p00
            + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
            + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
            + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
            + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
      }
      case 4: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 5: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      default:
        throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

double liquidus_slope(int which_comp, double *c)
{
  if (linearized_liquidus   )
  {
    switch (which_comp)
    {
      case 0: return liquidus_slope_0   ;
      case 1: return liquidus_slope_1   ;
      case 2: return liquidus_slope_2   ;
      case 3: return liquidus_slope_3   ;
    }
  }
  else
  {
    switch (alloy   )
    {
      case 0: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 1: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 2: // Co-W-Al
      {
        double c0 = 100.*c[0];
        double c1 = 100.*c[1];
        static double p10 =  2.369e+00;
        static double p01 =  1.771e+00;
        static double p20 = -2.238e-01;
        static double p11 =  1.041e-01;
        static double p02 = -3.046e-01;
        static double p30 =  1.358e-03;
        static double p21 = -1.568e-02;
        static double p12 = -1.457e-02;
        static double p03 =  3.317e-03;
        static double p40 = -3.283e-06;
        static double p31 =  3.559e-04;
        static double p22 =  1.228e-04;
        static double p13 =  1.531e-04;
        static double p04 = -1.866e-04;
        switch (which_comp)
        {
          case 0:
          return 100.*(0.
                       + p10*1.*pow(c0,0)*pow(c1,0) + p01*0.*pow(c0,0)*pow(c1,1)
                       + p20*2.*pow(c0,1)*pow(c1,0) + p11*1.*pow(c0,0)*pow(c1,1) + p02*0.*pow(c0,0)*pow(c1,2)
                       + p30*3.*pow(c0,2)*pow(c1,0) + p21*2.*pow(c0,1)*pow(c1,1) + p12*1.*pow(c0,0)*pow(c1,2) + p03*0.*pow(c0,0)*pow(c1,3)
                       + p40*4.*pow(c0,3)*pow(c1,0) + p31*3.*pow(c0,2)*pow(c1,1) + p22*2.*pow(c0,1)*pow(c1,2) + p13*1.*pow(c0,0)*pow(c1,3) + p04*0.*pow(c0,0)*pow(c1,4));
          case 1:
          return 100.*(0.
                       + p10*0.*pow(c0,1)*pow(c1,0) + p01*1.*pow(c0,0)*pow(c1,0)
                       + p20*0.*pow(c0,2)*pow(c1,0) + p11*1.*pow(c0,1)*pow(c1,0) + p02*2.*pow(c0,0)*pow(c1,1)
                       + p30*0.*pow(c0,3)*pow(c1,0) + p21*1.*pow(c0,2)*pow(c1,0) + p12*2.*pow(c0,1)*pow(c1,1) + p03*3.*pow(c0,0)*pow(c1,2)
                       + p40*0.*pow(c0,4)*pow(c1,0) + p31*1.*pow(c0,3)*pow(c1,0) + p22*2.*pow(c0,2)*pow(c1,1) + p13*3.*pow(c0,1)*pow(c1,2) + p04*4.*pow(c0,0)*pow(c1,3));
          default: throw std::invalid_argument("\n");
        }
      }
      case 3: // Co-Al-W
      {
        double c0 = 100.*c[1];
        double c1 = 100.*c[0];
        static double p01 =  2.369e+00;
        static double p10 =  1.771e+00;
        static double p02 = -2.238e-01;
        static double p11 =  1.041e-01;
        static double p20 = -3.046e-01;
        static double p03 =  1.358e-03;
        static double p12 = -1.568e-02;
        static double p21 = -1.457e-02;
        static double p30 =  3.317e-03;
        static double p04 = -3.283e-06;
        static double p13 =  3.559e-04;
        static double p22 =  1.228e-04;
        static double p31 =  1.531e-04;
        static double p40 = -1.866e-04;
        switch (which_comp)
        {
          case 1:
          return 100.*(0.
                       + p10*1.*pow(c0,0)*pow(c1,0) + p01*0.*pow(c0,0)*pow(c1,1)
                       + p20*2.*pow(c0,1)*pow(c1,0) + p11*1.*pow(c0,0)*pow(c1,1) + p02*0.*pow(c0,0)*pow(c1,2)
                       + p30*3.*pow(c0,2)*pow(c1,0) + p21*2.*pow(c0,1)*pow(c1,1) + p12*1.*pow(c0,0)*pow(c1,2) + p03*0.*pow(c0,0)*pow(c1,3)
                       + p40*4.*pow(c0,3)*pow(c1,0) + p31*3.*pow(c0,2)*pow(c1,1) + p22*2.*pow(c0,1)*pow(c1,2) + p13*1.*pow(c0,0)*pow(c1,3) + p04*0.*pow(c0,0)*pow(c1,4));
          case 0:
          return 100.*(0.
                       + p10*0.*pow(c0,1)*pow(c1,0) + p01*1.*pow(c0,0)*pow(c1,0)
                       + p20*0.*pow(c0,2)*pow(c1,0) + p11*1.*pow(c0,1)*pow(c1,0) + p02*2.*pow(c0,0)*pow(c1,1)
                       + p30*0.*pow(c0,3)*pow(c1,0) + p21*1.*pow(c0,2)*pow(c1,0) + p12*2.*pow(c0,1)*pow(c1,1) + p03*3.*pow(c0,0)*pow(c1,2)
                       + p40*0.*pow(c0,4)*pow(c1,0) + p31*1.*pow(c0,3)*pow(c1,0) + p22*2.*pow(c0,2)*pow(c1,1) + p13*3.*pow(c0,1)*pow(c1,2) + p04*4.*pow(c0,0)*pow(c1,3));
          default: throw std::invalid_argument("\n");
        }
      }
      case 4: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 5: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      default: throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

double part_coeff(int which_comp, double *c)
{
  if (const_part_coeff   )
  {
    switch (which_comp)
    {
      case 0: return part_coeff_0   ;
      case 1: return part_coeff_1   ;
      case 2: return part_coeff_2   ;
      case 3: return part_coeff_3   ;
    }
  }
  else
  {
    switch (alloy   )
    {
      case 0: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 1: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 2: // Co-W-Al
        switch (which_comp)
        {
          case 0:
          {
            double c0 = 100.*c[0];
            double c1 = 100.*c[1];
            static double p00 =  1.135e+00;
            static double p10 = -3.118e-02;
            static double p01 =  2.239e-03;
            static double p20 =  1.463e-03;
            static double p11 = -1.917e-03;
            static double p02 =  1.135e-03;
            static double p30 = -4.768e-05;
            static double p21 =  9.953e-05;
            static double p12 =  4.394e-05;
            static double p03 = -6.706e-05;
            static double p40 =  8.710e-07;
            static double p31 = -2.235e-06;
            static double p22 = -5.952e-07;
            static double p13 = -3.904e-07;
            static double p04 =  1.545e-06;
            return p00
                + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
                + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
                + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
                + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
          }
          case 1:
          {
            double c0 = 100.*c[0];
            double c1 = 100.*c[1];
            static double p00 =  1.114e+00;
            static double p10 = -9.187e-03;
            static double p01 = -4.804e-02;
            static double p20 =  1.733e-03;
            static double p11 = -1.406e-04;
            static double p02 =  4.313e-03;
            static double p30 = -5.248e-05;
            static double p21 = -8.236e-05;
            static double p12 =  2.716e-05;
            static double p03 = -2.159e-04;
            static double p40 =  5.841e-07;
            static double p31 =  9.961e-07;
            static double p22 =  1.741e-06;
            static double p13 = -9.305e-07;
            static double p04 =  4.122e-06;
            return p00
                + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
                + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
                + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
                + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
          }
          default: throw std::invalid_argument("\n");
        }
      case 3: // Co-Al-W
        switch (which_comp)
        {
          case 1:
          {
            double c0 = 100.*c[1];
            double c1 = 100.*c[0];
            static double p00 =  1.135e+00;
            static double p10 = -3.118e-02;
            static double p01 =  2.239e-03;
            static double p20 =  1.463e-03;
            static double p11 = -1.917e-03;
            static double p02 =  1.135e-03;
            static double p30 = -4.768e-05;
            static double p21 =  9.953e-05;
            static double p12 =  4.394e-05;
            static double p03 = -6.706e-05;
            static double p40 =  8.710e-07;
            static double p31 = -2.235e-06;
            static double p22 = -5.952e-07;
            static double p13 = -3.904e-07;
            static double p04 =  1.545e-06;
            return p00
                + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
                + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
                + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
                + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
          }
          case 0:
          {
            double c0 = 100.*c[1];
            double c1 = 100.*c[0];
            static double p00 =  1.114e+00;
            static double p10 = -9.187e-03;
            static double p01 = -4.804e-02;
            static double p20 =  1.733e-03;
            static double p11 = -1.406e-04;
            static double p02 =  4.313e-03;
            static double p30 = -5.248e-05;
            static double p21 = -8.236e-05;
            static double p12 =  2.716e-05;
            static double p03 = -2.159e-04;
            static double p40 =  5.841e-07;
            static double p31 =  9.961e-07;
            static double p22 =  1.741e-06;
            static double p13 = -9.305e-07;
            static double p04 =  4.122e-06;
            return p00
                + p10*pow(c0,1)*pow(c1,0) + p01*pow(c0,0)*pow(c1,1)
                + p20*pow(c0,2)*pow(c1,0) + p11*pow(c0,1)*pow(c1,1) + p02*pow(c0,0)*pow(c1,2)
                + p30*pow(c0,3)*pow(c1,0) + p21*pow(c0,2)*pow(c1,1) + p12*pow(c0,1)*pow(c1,2) + p03*pow(c0,0)*pow(c1,3)
                + p40*pow(c0,4)*pow(c1,0) + p31*pow(c0,3)*pow(c1,1) + p22*pow(c0,2)*pow(c1,2) + p13*pow(c0,1)*pow(c1,3) + p04*pow(c0,0)*pow(c1,4);
          }
          default: throw std::invalid_argument("\n");
        }
      case 4: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 5: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      default: throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

// ----------------------------------------
// analytical solution
// ----------------------------------------
namespace analytic {

double Al, Bl;
double As, Bs;
double as, al;
double Ai[4], Bi[4];
double lam;
double Tstar, Cstar[4];
double t_start;
double kp[4];

double F(double x)
{
  switch (geometry   ) {
    case -3: return .5*(exp(-x*x)/x - sqrt(PI)*boost::math::erfc(x));
    case -2: return boost::math::expint(1, x*x);
    case -1: return boost::math::erfc(x);
    default: return 0;
  }
}

double Fp(double x)
{
  switch (geometry   ) {
    case -3: return -.5*exp(-x*x)/(x*x);
    case -2: return -2.*exp(-x*x)/x;
    case -1: return -2.*exp(-x*x)/sqrt(PI);
    default: return 0;
  }
}

double xF_Fp(double x)
{
  return x*F(x)/Fp(x);
}

double xFp(double x)
{
  return x*Fp(x);
}

void set_analytical_solution(double M, double v, double R)
{
  t_start = .5*R/v;
  lam = .5*v*R;

  as = thermal_cond_s   /density_s   /heat_capacity_s   ;
  al = thermal_cond_l   /density_l   /heat_capacity_l   ;

  Bl = 0;

  for (int j = 0; j < num_comps   ; ++j) {
    Cstar[j] = (*initial_conc_all[j]);
  }

  for (int jj = 0; jj < 20; ++jj) {
    for (int j = 0; j < num_comps   ; ++j) {
      kp[j] = part_coeff(j, Cstar);
    }
    for (int j = 0; j < num_comps   ; ++j) {
      Cstar[j] = (*initial_conc_all[j])/(1. + 2.*(1.-kp[j])*xF_Fp(sqrt(lam/(*solute_diff_all[j]))));
    }
  }

  for (int j = 0; j < num_comps   ; ++j)
  {
    Ai[j] = (*initial_conc_all[j]);
    Bi[j] = (Cstar[j]-(*initial_conc_all[j]))/F(sqrt(lam/(*solute_diff_all[j])));

    Bl = Bl + liquidus_slope(j, Cstar)*Bi[j]*xFp(sqrt(lam/(*solute_diff_all[j])));
  }

  Tstar = liquidus_value(Cstar);

  Bl = Bl/xFp(sqrt(lam/al))/M;
  Bs = Bl * thermal_cond_l   /thermal_cond_s    * sqrt(as/al) * Fp(sqrt(lam/al))/Fp(sqrt(lam/as)) + latent_heat   /(density_s   *heat_capacity_s   ) * 2*sqrt(lam/as)/Fp(sqrt(lam/as));

  Al = Tstar - Bl*F(sqrt(lam/al));
  As = Tstar - Bs*F(sqrt(lam/as));

  time_limit    = .25*SQR(container_radius_inner    + front_location_final   )/lam-t_start;
}

inline double nu(double a, double t, double r) {return r/sqrt(4.*a*(t+t_start)); }

inline double tl_exact(double t, double r) {return Al + Bl* F(nu(al,t,r)); }

inline double ts_exact(double t, double r) {return As + Bs* F(nu(al,t,r)); }


inline double cl_exact(int i, double t, double r) { return Ai[i] + Bi[i] * F(nu(*solute_diff_all[i],t,r)); }

inline double rf_exact(double t) { return 2.*sqrt(lam*(t+t_start)); }
inline double vn_exact(double t) { return sqrt(lam/(t+t_start)); }
inline double vf_exact(double r) { return -2.*lam/(EPS+fabs(r)); }
inline double ft_exact(double r) { return .25*SQR(r)/lam-t_start; }
//inline double vf_exact(double r) { return 0; }

inline double dtl_exact(double t, double r) { return Bl * Fp(nu(al,t,r))*nu(al,t,1.); }
inline double dts_exact(double t, double r) { return Bs * Fp(nu(as,t,r))*nu(as,t,1.); }

inline double dcl_exact(int i, double t, double r) { return Bi[i] * Fp(nu(*solute_diff_all[i],t,r))*nu(*solute_diff_all[i],t,1.); }
}
bool periodicity(int dir)
{
  switch (geometry) {
#ifdef P4_TO_P8
    case -3: return false;
#endif
    case -2: return false;
    case -1: return (dir == 0 ? 1 : 0);
    case  0: return (dir == 0 ? 1 : 0);
    case  1: return false;
    case  2: return false;
    case  3: return false;
    case  4: return false;
    case  5: return true;
    case  6: return (dir == 0 ? 1 : 0);
    case  7: return (dir == 0 ? 1 : 0);
    default: throw;
  }
}
class front_phi_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
#ifdef P4_TO_P8
      case -3: return analytic::rf_exact(t) - ABS3(x-xc(),
                                                   y-yc(),
                                                   z-zc());
#endif
      case -2: return analytic::rf_exact(t) - ABS2(x-xc(),
                                                   y-yc());
      case -1: return analytic::rf_exact(t) - ABS1(y-ymin());
      case 0: return -(y - front_location()) + 0.000/(1.+100.*fabs(x/(xmin+xmax)-.5))*double(rand())/double(RAND_MAX)  + 0.000/(1.+1000.*fabs(x/(xmin+xmax)-.75));
      case 1: return -(ABS2(x-xc(), y-yc())-seed_radius());
      case 2: return  (ABS2(x-xc(), y-yc())-(container_radius_outer()-front_location()));
      case 3: return  (ABS2(x-xc(), y-yc())-(container_radius_outer()-front_location()));
      case 4: return -(ABS2(x-xc(), y-yc())-(container_radius_inner()+front_location()));
      case 5:
      {
        double dist0 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*0. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*0. + seed_rot())));
        double dist1 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*1. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*1. + seed_rot())));
        double dist2 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*2. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*2. + seed_rot())));

        return seed_radius() - MIN(dist0, dist1, dist2);
      }
      case 6:
      {
        double front = -(y - front_location());
        double dist0 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*0. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*0. + seed_rot())));
        double dist1 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*1. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*1. + seed_rot())));
        double dist2 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*2. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*2. + seed_rot())));

        return MAX(front, seed_radius() - MIN(dist0, dist1, dist2));
      }
      case 7:
      {
        double front = -(y - front_location());
        double dist0 = ABS2(x-(xc()+.25*(xmax()-xmin())), y-yc());
        double dist1 = ABS2(x-(xc()-.25*(xmax()-xmin())), y-yc());

        return MAX(front, seed_radius() - MIN(dist0, dist1));
      }
      default: throw;
    }
  }
} front_phi_cf;

class contr_phi_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
#ifdef P4_TO_P8
      case -3: return MAX( -container_radius_outer() + ABS3(x-xc(), y-yc(), z-zc()),
                           +container_radius_inner() - ABS3(x-xc(), y-yc(), z-zc()) );
#endif
      case -2: return MAX( -container_radius_outer() + ABS2(x-xc(), y-yc()),
                           +container_radius_inner() - ABS2(x-xc(), y-yc()) );
      case -1: return -1;
      case  0: return -1;
      case  1: return -container_radius_outer() + ABS2(x-xc(), y-yc());
      case  2: return -container_radius_outer() + ABS2(x-xc(), y-yc());
      case  3: return MAX( -container_radius_outer() + ABS2(x-xc(), y-yc()),
                           +container_radius_inner() - ABS2(x-xc(), y-yc()) );
      case  4: return MAX( -container_radius_outer() + ABS2(x-xc(), y-yc()),
                           +container_radius_inner() - ABS2(x-xc(), y-yc()) );
      case  5: return -1;
      case  6: return -1;
      case  7: return -1;
      default: throw;
    }
  }
} contr_phi_cf;

class phi_eff_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return MAX(contr_phi_cf(DIM(x,y,z)), -fabs(front_phi_cf(DIM(x,y,z))));
  }
} phi_eff_cf;

class seed_number_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
#ifdef P4_TO_P8
      case -3: return 0;
#endif
      case -2: return 0;
      case -1: return 0;
      case 0: return x < xc() ? 0 : 1;
      case 1: return 0;
      case 2: return 0;
      case 3: return 0;
      case 4: return 0;
      case 5:
      {
        double dist0 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*0. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*0. + seed_rot())));
        double dist1 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*1. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*1. + seed_rot())));
        double dist2 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*2. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*2. + seed_rot())));

        if (dist0 <= MIN(dist1, dist2)) return 0;
        if (dist1 <= MIN(dist0, dist2)) return 1;
        return 2;
      }
      case 6:
      {
        double front = fabs(y - front_location());
        double dist0 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*0. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*0. + seed_rot())));
        double dist1 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*1. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*1. + seed_rot())));
        double dist2 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*2. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*2. + seed_rot())));

        if (dist0 <= MIN(dist1, dist2, front)) return 0;
        if (dist1 <= MIN(dist2, dist0, front)) return 1;
        if (dist2 <= MIN(dist0, dist1, front)) return 2;
        return 3;
      }
      case 7:
      {
        double front = fabs(y - front_location());
        double dist0 = ABS2(x-(xc()+.25*(xmax()-xmin())), y-yc());
        double dist1 = ABS2(x-(xc()-.25*(xmax()-xmin())), y-yc());

        if (dist0 <= MIN(dist1, front)) return 0;
        if (dist1 <= MIN(dist0, front)) return 1;
        return 2;
      }
      default: throw;
    }
  }
} seed_number_cf;

int num_seeds()
{
  switch (geometry)
  {
#ifdef P4_TO_P8
    case -3: return 1;
#endif
    case -2: return 1;
    case -1: return 1;
    case 0: return 2;
    case 1: return 1;
    case 2: return 1;
    case 3: return 1;
    case 4: return 1;
    case 5: return 3;
    case 6: return 4;
    case 7: return 3;
    default: throw;
  }
}

double theta0(int seed)
{
  if (seed_type == 0) return crystal_orientation;
  else
  {
    switch (geometry)
    {
#ifdef P4_TO_P8
      case -3: return crystal_orientation;
#endif
      case -2: return crystal_orientation;
      case -1: return crystal_orientation;
      case 0:
        switch (seed)
        {
          case 0: return crystal_orientation -PI/6.;
          case 1: return crystal_orientation +PI/6.;
          default: throw;
        }
      case 1: return crystal_orientation;
      case 2: return crystal_orientation;
      case 3: return crystal_orientation;
      case 4: return crystal_orientation;
      case 5:
        switch (seed)
        {
          case 0: return crystal_orientation -PI/7.;
          case 1: return crystal_orientation +PI/6.;
          case 2: return crystal_orientation -PI/5.;
          default: throw;
        }
      case 6:
        switch (seed)
        {
          case 0: return crystal_orientation -PI/7.;
          case 1: return crystal_orientation +PI/6.;
          case 2: return crystal_orientation -PI/5.;
          case 3: return 0.;
          default: throw;
        }
      case 7:
        switch (seed)
        {
          case 0: return crystal_orientation -PI/7.;
          case 1: return crystal_orientation +PI/6.;
          case 2: return 0.;
          default: throw;
        }
      default: throw;
    }
  }
}

class eps_c_cf_t : public CF_DIM
{
  double theta0;
public:
  eps_c_cf_t(double theta0 = 0) : theta0(theta0) {}
  double operator()(DIM(double nx, double ny, double nz)) const
  {
#ifdef P4_TO_P8
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_c*(1.0-4.0*eps_a*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_c*(1.-15.*eps_a*cos(symmetry*(theta-theta0)))/(1.+15.*eps_a);
#endif
  }
};

class eps_v_cf_t : public CF_DIM
{
  double theta0;
public:
  eps_v_cf_t(double theta0 = 0) : theta0(theta0) {}
  double operator()(DIM(double nx, double ny, double nz)) const
  {
#ifdef P4_TO_P8
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_v*(1.0-4.0*eps_a*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_v*(1.-15.*eps_a*cos(symmetry*(theta-theta0)))/(1.+15.*eps_a);
#endif
  }
};
// ----------------------------------------
// define initial fields and boundary conditions
// ----------------------------------------
class cl_cf_t : public CF_DIM
{
  int idx;
public:
  cl_cf_t(int idx) : idx(idx) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry()) {
#ifdef P4_TO_P8
      case -3: return analytic::cl_exact(idx, t, ABS3(x-xc(), y-yc(), z-zc()));
#endif
      case -2: return analytic::cl_exact(idx, t, ABS2(x-xc(), y-yc()));
      case -1: return analytic::cl_exact(idx, t, ABS1(y-ymin()));
      case  0:
        if (start_from_moving_front) {
          return (*initial_conc_all[idx])*(1. + (1.-analytic::kp[idx])/analytic::kp[idx]*
                                           exp(-cooling_velocity/(*solute_diff_all[idx])*(-front_phi_cf(DIM(x,y,z)-0*cooling_velocity*t))));
        }
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7:
        if (start_from_moving_front) {
          return (*initial_conc_all[idx])*(1. + (1.-analytic::kp[idx])/analytic::kp[idx]*
                                           exp(-cooling_velocity/(*solute_diff_all[idx])*(-front_phi_cf(DIM(x,y,z)-0*cooling_velocity*t))));
        }
      return *initial_conc_all[idx];
      default: throw;
    }
  }
};

cl_cf_t cl_cf_0(0);
cl_cf_t cl_cf_1(1);
cl_cf_t cl_cf_2(2);
cl_cf_t cl_cf_3(3);

CF_DIM* cl_cf_all[] = { &cl_cf_0,
                        &cl_cf_1,
                        &cl_cf_2,
                        &cl_cf_3 };

class cs_cf_t : public CF_DIM
{
  int idx;
public:
  cs_cf_t(int idx) : idx(idx) {}
  double operator()(DIM(double, double, double)) const
  {
    switch (geometry()) {
      case -3:
      case -2:
      case -1:
      case  0:
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7: return analytic::Cstar[idx];
      default: throw;
    }
  }
};

cs_cf_t cs_cf_0(0);
cs_cf_t cs_cf_1(1);
cs_cf_t cs_cf_2(2);
cs_cf_t cs_cf_3(3);

CF_DIM* cs_cf_all[] = { &cs_cf_0,
                        &cs_cf_1,
                        &cs_cf_2,
                        &cs_cf_3 };

class tl_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry())
    {
#ifdef P4_TO_P8
      case -3: return analytic::tl_exact(t, ABS3(x-xc, y-yc, z-zc));
#endif
      case -2: return analytic::tl_exact(t, ABS2(x-xc, y-yc));
      case -1: return analytic::tl_exact(t, ABS1(y-ymin));
      case  0: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient();
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())));
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())));
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient();
      case  7: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient();
      default: throw;
    }
  }
} tl_cf;

class ts_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry())
    {
#ifdef P4_TO_P8
      case -3: return analytic::ts_exact(t, ABS3(x-xc(), y-yc(), z-zc()));
#endif
      case -2: return analytic::ts_exact(t, ABS2(x-xc(), y-yc()));
      case -1: return analytic::ts_exact(t, ABS1(y-ymin()));
      case  0: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient()*thermal_cond_l/thermal_cond_s;
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())))*thermal_cond_l()/thermal_cond_s();
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())))*thermal_cond_l()/thermal_cond_s();
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient()*thermal_cond_l/thermal_cond_s;
      case  7: return analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient()*thermal_cond_l/thermal_cond_s;
      default: throw;
    }
  }
} ts_cf;

class tf_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry()) {
#ifdef P4_TO_P8
      case -3:
#endif
      case -2:
      case -1:
      case  0:
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7: return analytic::Tstar;
      default: throw;
    }
  }
} tf_cf;

class v_cf_t : public CF_DIM
{
  cf_value_type_t what;
public:
  v_cf_t(cf_value_type_t what) : what(what) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry()) {
#ifdef P4_TO_P8
      case -3:
        switch (what) {
          case VAL: return analytic::vn_exact(t);
          case DDX: return analytic::vn_exact(t)*(x-xc)/(EPS+ABS3(x-xc, y-yc, z-zc));
          case DDY: return analytic::vn_exact(t)*(y-yc)/(EPS+ABS3(x-xc, y-yc, z-zc));
          case DDZ: return analytic::vn_exact(t)*(z-zc)/(EPS+ABS3(x-xc, y-yc, z-zc));
          default:  return 0;
        }
#endif
      case -2:
        switch (what) {
          case VAL: return analytic::vn_exact(t);
          case DDX: return analytic::vn_exact(t)*(x-xc)/(EPS+ABS2(x-xc, y-yc));
          case DDY: return analytic::vn_exact(t)*(y-yc)/(EPS+ABS2(x-xc, y-yc));
          default:  return 0;
        }
      case -1:
        switch (what) {
          case VAL:
          case DDY: return analytic::vn_exact(t);
          default:  return 0;
        }
      case 0:
        if (start_from_moving_front) {
          switch (what) {
            case VAL:
            case DDY: return cooling_velocity;
            default:  return 0;
          }
        }
      default: return 0;
    }
  }
};

v_cf_t vn_cf(VAL);
v_cf_t vx_cf(DDX);
v_cf_t vy_cf(DDY);
v_cf_t vz_cf(DDZ);

class vf_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry) {
#ifdef P4_TO_P8
      case -3: return -analytic::vf_exact(ABS3(x-xc, y-yc, z-zc));
#endif
      case -2: return -analytic::vf_exact(ABS2(x-xc, y-yc));
      case -1: return -analytic::vf_exact(ABS1(y-ymin));
      case  0:
        if (start_from_moving_front) {
          return -cooling_velocity;
        }
      default: return 0;
    }
  }
} vf_cf;

class ft_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry) {
#ifdef P4_TO_P8
      case -3: return analytic::ft_exact(ABS3(x-xc, y-yc, z-zc));
#endif
      case -2: return analytic::ft_exact(ABS2(x-xc, y-yc));
      case -1: return analytic::ft_exact(ABS1(y-ymin));
      case  0:
        if (start_from_moving_front) {
          return (y-front_location) / cooling_velocity;
        }
      default: return 0;
    }
  }
} ft_cf;

double smooth_start(double t) { return smoothstep(smoothstep_order, (t+EPS)/(starting_time+EPS)); }

class bc_value_temp_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (bc_type_temp) {
      case DIRICHLET:
        switch (geometry) {
#ifdef P4_TO_P8
          case -3: return ABS3(x-xc,y-yc,z-zc) < .5*(container_radius_inner+container_radius_outer) ?
                  analytic::ts_exact(t, container_radius_inner) :
                  analytic::tl_exact(t, container_radius_outer);
#endif
          case -2: return ABS2(x-xc,y-yc) < .5*(container_radius_inner+container_radius_outer) ?
                  analytic::ts_exact(t, container_radius_inner) :
                  analytic::tl_exact(t, container_radius_outer);
          case -1: return y < .5*(ymin+ymax) ? analytic::ts_exact(t, 0) : analytic::tl_exact(t, ymax-ymin);
          case  0: return y < .5*(ymin+ymax) ? analytic::Tstar + (y - (front_location + cooling_velocity*t))*temp_gradient()
                                                     : analytic::Tstar + (y - (front_location + cooling_velocity*t))*(temp_gradient()*thermal_cond_l + cooling_velocity*latent_heat)/thermal_cond_s;
          case  1:
          case  2:
          case  3:
          case  4:
          case  5:
          case  6:
          case  7: return 0;
          default: throw;
        }
      case NEUMANN:
        switch (geometry) {
#ifdef P4_TO_P8
          case -3: return ABS3(x-xc,y-yc,z-zc) < .5*(container_radius_inner+container_radius_outer) ?
                  -thermal_cond_s*analytic::dts_exact(t, container_radius_inner) :
                  +thermal_cond_l*analytic::dtl_exact(t, container_radius_outer);
#endif
          case -2: return ABS2(x-xc,y-yc) < .5*(container_radius_inner+container_radius_outer) ?
                  -thermal_cond_s*analytic::dts_exact(t, container_radius_inner) :
                  +thermal_cond_l*analytic::dtl_exact(t, container_radius_outer);
          case -1: return y < .5*(ymin+ymax) ? -analytic::dts_exact(t, 0) : analytic::dtl_exact(t, ymax-ymin);
          case  0: return y > .5*(ymin+ymax) ? temp_gradient : -(temp_gradient*thermal_cond_l/thermal_cond_s + cooling_velocity*(latent_heat+density_l*temp_gradient*heat_capacity_l*(ymax-ymin))/thermal_cond_s * smooth_start(t));
          case  1: return -seed_radius/container_radius_outer*cooling_velocity*latent_heat*smooth_start(t);
          case  2: return -(container_radius_outer-front_location)/container_radius_outer*cooling_velocity*latent_heat*smooth_start(t);
          case  3:
            if (ABS2(x-xc,y-yc) > .5*(container_radius_inner+container_radius_outer)) {
              return -thermal_cond_s*temp_gradient
                  -(container_radius_outer-front_location)/container_radius_outer
                  *cooling_velocity*latent_heat*smooth_start(t);
            } else {
              return  thermal_cond_s*temp_gradient*container_radius_outer/container_radius_inner;
            }
          case  4:
            if (ABS2(x-xc,y-yc) > .5*(container_radius_inner+container_radius_outer)) {
              return  thermal_cond_l*temp_gradient
                  -(container_radius_inner+front_location)/container_radius_outer
                  *cooling_velocity*latent_heat*smooth_start(t);
            } else {
              return -thermal_cond_l*temp_gradient*container_radius_outer/container_radius_inner;
            }
          case  5: return 0;
          case  6: return y > .5*(ymin+ymax) ? temp_gradient : -(temp_gradient*thermal_cond_l/thermal_cond_s + cooling_velocity*(latent_heat+density_l*temp_gradient*heat_capacity_l*(ymax-ymin))/thermal_cond_s * smooth_start(t));
          case  7: return y > .5*(ymin+ymax) ? temp_gradient : -(temp_gradient*thermal_cond_l/thermal_cond_s + cooling_velocity*(latent_heat+density_l*temp_gradient*heat_capacity_l*(ymax-ymin))/thermal_cond_s * smooth_start(t));
          default: throw;
        }
      default: throw;
    }
  }
} bc_value_temp;

class bc_value_conc_t : public CF_DIM
{
  int idx;
public:
  bc_value_conc_t(int idx) : idx(idx) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (bc_type_conc) {
      case DIRICHLET:
        switch (geometry) {
#ifdef P4_TO_P8
          case -3: return analytic::cl_exact(idx, t, container_radius_outer);
#endif
          case -2: return analytic::cl_exact(idx, t, container_radius_outer);
          case -1: return analytic::cl_exact(idx, t, ABS1(ymax-ymin));
          case  0:
          case  1:
          case  2:
          case  3:
          case  4:
          case  5:
          case  6:
          case  7: return *initial_conc_all[idx];
          default: throw;
        }
      case NEUMANN:
        switch (geometry) {
#ifdef P4_TO_P8
          case -3: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, container_radius_outer);
#endif
          case -2: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, container_radius_outer);
          case -1: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, ymax-ymin);
          case  0:
          case  1:
          case  2:
          case  3:
          case  4:
          case  5:
          case  6:
          case  7: return 0;
          default: throw;
        }
      default: throw;
    }
  }
};

bc_value_conc_t bc_value_conc_0(0);
bc_value_conc_t bc_value_conc_1(1);
bc_value_conc_t bc_value_conc_2(2);
bc_value_conc_t bc_value_conc_3(3);

CF_DIM* bc_value_conc_all[] = { &bc_value_conc_0,
                                &bc_value_conc_1,
                                &bc_value_conc_2,
                                &bc_value_conc_3 };

class volumetric_heat_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    if (geometry == 5) return -cooling_velocity*latent_heat*2.*PI*seed_radius/(xmax-xmin)/(ymax-ymin)/box_size*smooth_start(t);
    else return 0;
  }
} volumetric_heat_cf;
// ---------------------------------------
// Related to the nondimensional problem / nondimensionalization:
// ---------------------------------------
DEFINE_PARAMETER(pl, double, Re, -1., "Reynolds number (rho Uinf d)/mu, where d is the characteristic length scale - default is 0. \n");
DEFINE_PARAMETER(pl, double, Pr, -1., "Prandtl number - computed from mu_l, alpha_l, rho_l \n");
DEFINE_PARAMETER(pl, double, St, -1., "Stefan number (cp_s deltaT/L)- computed from cp_s, deltaT, L \n");
DEFINE_PARAMETER(pl, double, RaT, -1., "Rayleigh number by temperature (default:0) \n");
DEFINE_PARAMETER(pl, double, RaC_0, -1., "Rayleigh number by concentration for component no. 0 (default:0) \n");
DEFINE_PARAMETER(pl, double, RaC_1, -1., "Rayleigh number by concentration for component no. 1 (default:0) \n");
DEFINE_PARAMETER(pl, double, RaC_2, -1., "Rayleigh number by concentration for component no. 2 (default:0) \n");
DEFINE_PARAMETER(pl, double, RaC_3, -1., "Rayleigh number by concentration for component no. 3 (default:0) \n");
DEFINE_PARAMETER(pl, double, Le_0, -1., "Lewis number for component no. 0 (default:0) \n");
DEFINE_PARAMETER(pl, double, Le_1, -1., "Lewis number for component no. 1 (default:0) \n");
DEFINE_PARAMETER(pl, double, Le_2, -1., "Lewis number for component no. 2 (default:0) \n");
DEFINE_PARAMETER(pl, double, Le_3, -1., "Lewis number for component no. 3 (default:0) \n");

problem_dimensionalization_type_t problem_dimensionalization_type;
void select_problem_nondim_or_dim_formulation(){
  switch(example_){
  case BASIC_MULTI_ALLOY:{
    problem_dimensionalization_type = NONDIM_BY_SCALAR_DIFFUSIVITY;
    break;
  }
}
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
  
  cmdParser cmd;
  cmd.parse(argc, argv);
  alloy.set_from_cmd(cmd);
  set_alloy_parameters();
  pl.set_from_cmd_all(cmd);
}









