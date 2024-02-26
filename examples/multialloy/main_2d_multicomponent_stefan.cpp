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
#include <linux/limits.h>

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
#include <src/my_p4est_stefan_with_fluids.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;

param_list_t pl;

param_t<bool> track_memory_usage (pl, 1, "track_memory_usage", "");

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<int>    nx   (pl, 1,  "nx", "Number of trees in the x-direction");
param_t<int>    ny   (pl, 16, "ny", "Number of trees in the y-direction");
param_t<int>    nz   (pl, 1,  "nz", "Number of trees in the z-direction");

param_t<double> xmin (pl, 0, "xmin", "Box xmin");
param_t<double> ymin (pl, 0, "ymin", "Box ymin");
param_t<double> zmin (pl, 0, "zmin", "Box zmin");

param_t<double> xmax (pl, 1,  "xmax", "Box xmax");
param_t<double> ymax (pl, 16, "ymax", "Box ymax");
param_t<double> zmax (pl, 1,  "zmax", "Box zmax");

param_t<double> xc   (pl, .51, "xc", "Centering point x");
param_t<double> yc   (pl, .32, "yc", "Centering point y");
param_t<double> zc   (pl, .53, "zc", "Centering point z");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
param_t<int> lmin (pl, 5, "lmin", "Min level of the tree");
param_t<int> lmax (pl, 5, "lmax", "Max level of the tree");
#else
param_t<int> lmin (pl, 3, "lmin", "Min level of the tree");
param_t<int> lmax (pl, 10, "lmax", "Max level of the tree");
#endif

param_t<int> sub_split_lvl (pl, 0, "sub_split_lvl", "");
param_t<int> sub_split_num (pl, 0, "sub_split_num", "");

param_t<double> lip  (pl, 1.5, "lip",  "Fine-to-coarse grid transition width");
param_t<double> band (pl, 1.5, "band", "Uniform band width around interfaces");

//-------------------------------------
// solver parameters
//-------------------------------------
param_t<bool> use_points_on_interface   (pl, 1, "use_points_on_interface", "");
param_t<bool> use_superconvergent_robin (pl, 0, "use_superconvergent_robin", "");

param_t<int>    update_c0_robin (pl, 1,     "update_c0_robin", "Solve for c0 using Robin BC: 0 - never (pl, 1 - once (pl, 2 - always");
param_t<int>    order_in_time   (pl, 2,     "order_in_time",   "");
param_t<int>    max_iterations  (pl, 7,     "max_iterations",  "");
param_t<double> bc_tolerance    (pl, 1.e-5, "bc_tolerance",    "");
param_t<double> cfl_number      (pl, 0.6,   "cfl_number",      "");
//param_t<double> base_cfl        (pl, pow(1.0,1.5)*0.0811,   "base_cfl",      "");
param_t<double> base_cfl        (pl, 0.111/pow(4.,1.5),   "base_cfl",      "");

param_t<double> proximity_smoothing       (pl, 1.01, "proximity_smoothing",       "");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool>   save_characteristics (pl, 0, "save_characteristics", "");
param_t<bool>   save_dendrites       (pl, 0, "save_dendrites", "");
param_t<bool>   save_accuracy        (pl, 0, "save_accuracy", "");
param_t<bool>   save_timings         (pl, 0, "save_timings", "");
param_t<bool>   save_params          (pl, 1, "save_params", "");
param_t<bool>   save_vtk             (pl, 1, "save_vtk", "");
param_t<bool>   save_vtk_solid       (pl, 0, "save_vtk_solid", "");
param_t<bool>   save_vtk_analytical  (pl, 0, "save_vtk_analytical", "");
param_t<bool>   save_p4est           (pl, 1, "save_p4est", "");
param_t<bool>   save_p4est_solid     (pl, 0, "save_p4est_solid", "");
param_t<bool>   save_step_convergence(pl, 0, "save_step_convergence", "");

param_t<int>    save_every_dn (pl, 1000, "save_every_dn", "");
param_t<double> save_every_dl (pl, 0.025, "save_every_dl", "");
param_t<double> save_every_dt (pl, 0.1,  "save_every_dt",  "");

param_t<int>    save_type (pl, 0, "save_type", "0 - every n iterations (pl, 1 - every dl of growth (pl, 2 - every dt of time");

param_t<double> dendrite_cut_off_fraction (pl, 1.05, "dendrite_cut_off_fraction", "");
param_t<double> dendrite_min_length       (pl, 0.05, "dendrite_min_length", "");

// Parameters for saving/load of simulation state:
param_t<bool>   save_state            (pl, 0, "save_state", "are we saving the simulation state?");
param_t<int>    save_state_every_dn (pl, 1000, "save_state_every_dn", "save the simulation state every n iterations");
param_t<int>    num_state_backups (pl, 20, "num_state_backups", "number of simulation state backup files to keep");


param_t<bool>   loading_from_previous_state(pl, 0, "loading_from_previous_state", "are we loading from a previously saved simulation state?");


//-------------------------------------
// alloy parameters
//-------------------------------------
param_t<double> density_l (pl, 8.88e-3, "density_l", "Density of liq. alloy (pl, kg.cm-3");
param_t<double> density_s (pl, 8.88e-3, "density_s", "Density of sol. alloy (pl, kg.cm-3");

param_t<double> heat_capacity_l (pl, 0.46e3, "heat_capacity_l", "Heat capacity of liq. alloy (pl, J.kg-1.K-1 ");
param_t<double> heat_capacity_s (pl, 0.46e3, "heat_capacity_s", "Heat capacity of sol. alloy (pl, J.kg-1.K-1 ");

param_t<double> thermal_cond_l (pl, 6.07e-1, "thermal_cond_l", "Thermal conductivity of liq. alloy (pl, W.cm-1.K-1 ");
param_t<double> thermal_cond_s (pl, 6.07e-1, "thermal_cond_s", "Thermal conductivity of sol. alloy (pl, W.cm-1.K-1 ");

param_t<double> latent_heat  (pl, 2350, "latent_heat",  "Latent heat of fusion (pl, J.cm-3");

param_t<bool>   linearized_liquidus (pl, 0, "linearized_liquidus", "Use linearized liquidus surface or true one");
param_t<bool>   const_part_coeff    (pl, 0, "const_part_coeff",    "Use averaged partition coefficients or true ones");

param_t<int>    num_comps (pl, 1, "num_comps", "Number of solutes");

param_t<double> initial_conc_0 (pl, 0.4, "initial_conc_0", "Initial concentration of component no. 0");
param_t<double> initial_conc_1 (pl, 0.4, "initial_conc_1", "Initial concentration of component no. 1");
param_t<double> initial_conc_2 (pl, 0.4, "initial_conc_2", "Initial concentration of component no. 2");
param_t<double> initial_conc_3 (pl, 0.4, "initial_conc_3", "Initial concentration of component no. 3");

double* initial_conc_all[] = { &initial_conc_0.val,
                               &initial_conc_1.val,
                               &initial_conc_2.val,
                               &initial_conc_3.val };
param_t<double> eutectic_conc_0 (pl, 0.4, "eutectic_conc_0", "Initial concentration of component no. 0");
param_t<double> eutectic_conc_1 (pl, 0.4, "eutectic_conc_1", "Initial concentration of component no. 1");
param_t<double> eutectic_conc_2 (pl, 0.4, "eutectic_conc_2", "Initial concentration of component no. 2");
param_t<double> eutectic_conc_3 (pl, 0.4, "eutectic_conc_3", "Initial concentration of component no. 3");


param_t<double> solute_diff_0 (pl, 1.0e-5, "solute_diff_0", "Diffusivity of component no. 0 in liquid phase - cm2.s-1");
param_t<double> solute_diff_1 (pl, 1.0e-5, "solute_diff_1", "Diffusivity of component no. 1 in liquid phase - cm2.s-1");
param_t<double> solute_diff_2 (pl, 1.0e-5, "solute_diff_2", "Diffusivity of component no. 2 in liquid phase - cm2.s-1");
param_t<double> solute_diff_3 (pl, 1.0e-5, "solute_diff_3", "Diffusivity of component no. 3 in liquid phase - cm2.s-1");

double* solute_diff_all[] = { &solute_diff_0.val,
                              &solute_diff_1.val,
                              &solute_diff_2.val,
                              &solute_diff_3.val };
double tn = 0.;
param_t<double> eps_c    (pl, 0, "eps_c",    "Curvature undercooling coefficient - cm.K");
param_t<double> eps_v    (pl, 0, "eps_v",    "Kinetic undercooling coefficient - s.K.cm-1");
param_t<double> eps_a    (pl, 0, "eps_a",    "Anisotropy coefficient");
param_t<double> symmetry (pl, 4, "symmetry", "Symmetric of crystals");

// parameters for the fluid flow problem:
// Note: these are better specified in the example defn, and the rayleigh numbers need to be calculated from other parameters
param_t<double> mu_l (pl, 1.0e-3, "mu_l", "viscosity of the fluid ");

param_t<double> Tinf (pl, -1., "Tinf", "Free-stream temperature for the problem (used for nondim w fluids) ");
param_t<double> Cinf0 (pl, -1., "Cinf0", "Free-stream conc 0 for the problem (used for nondim w fluids) ");
param_t<double> Cinf1 (pl, -1., "Cinf1", "Free-stream conc 1 for the problem (used for nondim w fluids)");
param_t<double> Cinf2 (pl, -1., "Cinf2", "Free-stream conc 2 for the problem (used for nondim w fluids)");
param_t<double> Cinf3 (pl, -1., "Cinf3", "Free-stream conc 3 for the problem (used for nondim w fluids)");

param_t<double> Pr (pl, -1., "Pr", "Prandtl number for the problem (with fluids) ");
param_t<double> Sc0 (pl, -1., "Sc0", "Schmidt number (comp 0) for the problem (with fluids) ");
param_t<double> Sc1 (pl, -1., "Sc1", "Schmidt number (comp 1) for the problem (with fluids) ");
param_t<double> Sc2 (pl, -1., "Sc2", "Schmidt number (comp 2) for the problem (with fluids) ");
param_t<double> Sc3 (pl, -1., "Sc3", "Schmidt number (comp 3) for the problem (with fluids) ");


param_t<double> beta_T(pl, 1.e8, "beta_T", "coefficient of thermal expansion - boussinesq");
param_t<double> beta_C_0(pl, 1.e8, "beta_C_0", "coefficient of concentration expansion for comp 0 -- boussinesq");
param_t<double> beta_C_1(pl, 1.e8, "beta_C_1", "coefficient of concentration expansion for comp 1 -- boussinesq");
param_t<double> beta_C_2(pl, 1.e8, "beta_C_2", "coefficient of concentration expansion for comp 2 -- boussinesq");
param_t<double> beta_C_3(pl, 1.e8, "beta_C_3", "coefficient of concentration expansion for comp 3 -- boussinesq");

// Formulae for Rayleigh numbers:
// RaT = (rho_l Beta_T |gravity_vec| DeltaT l_char^3)/(mu_l * alpha_l)
//    where Delta T is the characteristic temperature difference of the problem


// RaCj = (rho_l Beta_Cj |gravity_vec| DeltaCj l_char^3)/(mu_l * D_l,j)
//    where Delta Cj is the characteristic concentration difference of the problem for component j

param_t<double> Ra_T(pl, -1.0, "Ra_T", "thermal Rayleigh number - boussinesq");
param_t<double> Ra_C_0(pl, -1.0, "Ra_C_0", "species Rayleigh number for comp 0 -- boussinesq");
param_t<double> Ra_C_1(pl, -1.0, "Ra_C_1", "species Rayleigh number for comp 1 -- boussinesq");
param_t<double> Ra_C_2(pl, -1.0, "Ra_C_2", "species Rayleigh number for comp 2 -- boussinesq");
param_t<double> Ra_C_3(pl, -1.0, "Ra_C_3", "species Rayleigh number for comp 3 -- boussinesq");

param_t<double> gravity_(pl, 9.81, "gravity_", "gravity value used for boussinesq");


param_t<bool> do_boussinesq(pl, false, "do_boussinesq", "whether or not to use the boussinesq approx when solving the problem with fluid flow ");
param_t<bool> convert_dim_to_nondim_for_fluids(pl, false, "convert_dim_to_nondim_for_fluids", "whether or not to solve the fluid flow part nondimensionally \n ");

param_t<double> l_char(pl, 1.0, "l_char", "characteristic length scale of the problem - used to set nondimensional groups");

// auxiliary variable for linearized phase diagram
param_t<double> melting_temp (pl, 1728, "melting_temp", "Pure-substance melting point for linearized slope (pl, K");

param_t<double> liquidus_slope_0 (pl, -357, "liquidus_slope_0", "Slope of linearized liqiudus w.r.t component no. 0 (pl, K^-1");
param_t<double> liquidus_slope_1 (pl, -357, "liquidus_slope_1", "Slope of linearized liqiudus w.r.t component no. 1 (pl, K^-1");
param_t<double> liquidus_slope_2 (pl, -357, "liquidus_slope_2", "Slope of linearized liqiudus w.r.t component no. 2 (pl, K^-1");
param_t<double> liquidus_slope_3 (pl, -357, "liquidus_slope_3", "Slope of linearized liqiudus w.r.t component no. 3 (pl, K^-1");

double* liquidus_slope_all[] = { &liquidus_slope_0.val,
                                 &liquidus_slope_1.val,
                                 &liquidus_slope_2.val,
                                 &liquidus_slope_3.val };

param_t<double> part_coeff_0 (pl, 0.86, "part_coeff_0", "Partition coefficient for component no. 0");
param_t<double> part_coeff_1 (pl, 0.86, "part_coeff_1", "Partition coefficient for component no. 1");
param_t<double> part_coeff_2 (pl, 0.86, "part_coeff_2", "Partition coefficient for component no. 2");
param_t<double> part_coeff_3 (pl, 0.86, "part_coeff_3", "Partition coefficient for component no. 3");

double* part_coeff_all[] = { &part_coeff_0.val,
                             &part_coeff_1.val,
                             &part_coeff_2.val,
                             &part_coeff_3.val };

param_t<int> alloy (pl, 2, "alloy", "0: Ni -  0.4at%Cu bi-alloy, "
                                    "1: Ni -  0.2at%Cu -  0.2at%Cu tri-alloy, "
                                    "2: Co - 10.7at%W  -  9.4at%Al tri-alloy, "
                                    "3: Co -  9.4at%Al - 10.7at%W  tri-alloy, "
                                    "4: Ni - 15.2wt%Al -  5.8wt%Ta tri-alloy, "
                                    "5: Ni -  5.8wt%Ta - 15.2wt%Al tri-alloy, "
                                    "6: a made-up tetra-alloy, "", "
                                    "7: a made-up penta-alloy"
                                   "8: an alloy with values all = 1 for convergence test with fluids");

double scale = 30.0;//1./*30*/; // [Elyce  2/16/23] changing this to 1 bc it seems dangerous lol

//-------------------------------------
// problem parameters
//-------------------------------------
//param_t<double> volumetric_heat (pl,  0, "", "Volumetric heat generation (pl, J/cm^3");
param_t<double> cooling_velocity        (pl, 0.001,  "cooling_velocity", "Cooling velocity (pl, cm/s");
param_t<double> gradient_ratio          (pl, 0.75,  "gradient_ratio",   "Ratio of compositional and thermal gradients at the front");
param_t<double> temp_gradient           (pl, 500, "temp_gradient",    "Temperature gradient (pl, K/cm");
param_t<bool>   start_from_moving_front (pl, 0, "start_from_moving_front", "Relevant only for geometry==0");

param_t<int>    smoothstep_order (pl, 5,     "smoothstep_order", "Smoothness of cooling/heating ");
param_t<double> starting_time    (pl, 0.e-3, "starting_time",    "Time for cooling/heating to fully switch on (pl, s");

param_t<BoundaryConditionType> bc_type_conc (pl, NEUMANN, "bc_type_conc", "DIRICHLET/NEUMANN");
param_t<BoundaryConditionType> bc_type_temp (pl, NEUMANN, "bc_type_temp", "DIRICHLET/NEUMANN");
//param_t<BoundaryConditionType> bc_wall_type_vel  (pl, NEUMANN, "bc_wall_type_vel", "DIRICHLET/NEUMANN");
// all the above are usually neumann

//param_t<BoundaryConditionType> bc_type_temp (pl, DIRICHLET, "bc_type_temp", "DIRICHLET/NEUMANN");

param_t<int>    step_limit           (pl, INT_MAX, "step_limit",   "");
//param_t<int>    step_limit           (pl, 200, "step_limit",   "");
param_t<double> time_limit           (pl, DBL_MAX, "time_limit",   "");
param_t<double> growth_limit         (pl, 15, "growth_limit", "");
param_t<double> init_perturb         (pl, 1.0e-10,  "init_perturb",         "");
param_t<bool>   enforce_planar_front (pl, 0,       "enforce_planar_front", "");

param_t<double> front_location         (pl, 0.100,     "front_location",         "");
param_t<double> front_location_final   (pl, 0.25,     "front_location_final",   "");
param_t<double> container_radius_inner (pl, 0.1,     "container_radius_inner", "");
param_t<double> container_radius_outer (pl, 0.45,     "container_radius_outer", "");
param_t<double> seed_radius            (pl, 0.005,    "seed_radius",            "");
param_t<double> seed_dist              (pl, 0.1,      "seed_dist",              "");
param_t<double> seed_rot               (pl, PI/12.,   "seed_rot",               "");
param_t<double> crystal_orientation    (pl, 0.*PI/6., "crystal_orientation",    "");
param_t<int>    seed_type              (pl, 0, "seed_type", "0 - aligned,"
                                                            "1 - misaligned");
//param_t<double> box_size (pl, 0.08/sqrt(scale)/3./2., "box_size", "Physical width (in x) of the box in cm");
param_t<double> box_size (pl, 0.0075, "box_size", "Physical width (in x) of the box in cm");

param_t<int>    geometry (pl, 0, "geometry", "-3 - analytical spherical solidification,"
                                              "-2 - analytical cylindrical solidification,"
                                              "-1 - analytical planar solidification,"
                                              " 0 - directional solidification,"
                                              " 1 - growth of a spherical seed in a spherical container,"
                                              " 2 - growth of a spherical film in a spherical container,"
                                              " 3 - radial directional solidification in,"
                                              " 4 - radial directional solidification out,"
                                              " 5 - three spherical seeds -- Elyce and Rochi changed this to be one seed in the center of the domain -- (WIP),"
                                              " 6 - planar front and three spherical seeds,"
                                              " 7 - daniil has not defined this case - no comments - need to see what it is,"
                                              " 8 - convergence test for multicomp solidification with fluids ");

param_t<bool> solve_w_fluids (pl, 0, "solve_w_fluids", "");

param_t<double> convergence_dt (pl, 1.0e-3, "convergence_dt", "timestep used for all grid sizes in the convergence test");

param_t<bool> use_convergence_dt (pl, 0, "use_convergence_dt", "do we use a set dt for convergence test? ");

// ----------------------------------------
// alloy parameters
// ----------------------------------------

void set_alloy_parameters()
{
  switch (alloy())
  {
    case 0: // Ni - 0.4at%Cu
      density_l.val       = 8.88e-3; // kg.cm-3
      density_s.val       = 8.88e-3; // kg.cm-3
      heat_capacity_l.val = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s.val = 0.46e3;  // J.kg-1.K-1
      thermal_cond_l.val  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s.val  = 6.07e-1; // W.cm-1.K-1
      latent_heat.val     = 2350;    // J.cm-3

      num_comps.val = 1;

      solute_diff_0.val    = 1.e-3;  // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.4;    // at frac.

      eps_c.val = 0;
      eps_v.val = 0;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1728;   // K
      liquidus_slope_0.val = -357;   // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.86;   // partition coefficient

      break;

    case 1: // Ni - 0.2at%Cu - 0.2at%Cu
      density_l.val       = 8.88e-3; // kg.cm-3
      density_s.val       = 8.88e-3; // kg.cm-3
      heat_capacity_l.val = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s.val = 0.46e3;  // J.kg-1.K-1
      thermal_cond_l.val  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s.val  = 6.07e-1; // W.cm-1.K-1
      latent_heat.val     = 2350;    // J.cm-3

      num_comps.val = 2;

      solute_diff_0.val    = 1.e-5;  // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 5.e-5;  // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.2;    // at frac.
      initial_conc_1.val   = 0.2;    // at frac.

      eps_c.val = 4.e-5;
      eps_v.val = 0.0;
      eps_a.val = 0.00;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1728;   // K
      liquidus_slope_0.val = -357;   // K / at frac. - liquidous slope
      liquidus_slope_1.val = -357;   // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.86;   // partition coefficient
      part_coeff_1.val     = 0.86;   // partition coefficient
      break;

    case 2: // Co - 10.7at%W - 9.4at%Al (more realistic since D_W < D_Al)
      density_l.val       = 9.24e-3; // kg.cm-3
      density_s.val       = 9.24e-3; // kg.cm-3
      heat_capacity_l.val = 356;       // J.kg-1.K-1
      heat_capacity_s.val = 356;       // J.kg-1.K-1
      thermal_cond_l.val  = 1.3;       // W.cm-1.K-1
      thermal_cond_s.val  = 1.3;       // W.cm-1.K-1
      latent_heat.val     = 2600;    // J.cm-3 (this is latent heat multiplied by density! )

      num_comps.val = 2;

      solute_diff_0.val    = 1e-5;     // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 2e-5;     // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.107;    // at frac.
      initial_conc_1.val   = 0.094;    // at frac.



      /*  // linearized phase diagram
      melting_temp.val     = 1910;      // K
      liquidus_slope_0.val =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1.val =-1036;     // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.83;    // partition coefficient
      part_coeff_1.val     = 0.83;    // partition coefficient */

      melting_temp.val    =  1910;      // K
      liquidus_slope_0.val =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1.val =-1036;     // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.94;    // partition coefficient
      part_coeff_1.val     = 0.83;    // partition coefficient

      eps_c.val = 1.0e-5;
      eps_v.val = 0.0e-2;
      eps_a.val = 0.05;
      symmetry.val = 4;

      //("Warning! this example has a hard-coded PRandtl number. We will want to revisit this \n");

      // Calculate Rayleigh number


      break;

    case 3: // Co - 9.4at%Al - 10.7at%W
      density_l.val       = 9.24e-3; // kg.cm-3
      density_s.val       = 9.24e-3; // kg.cm-3
      heat_capacity_l.val = 356;       // J.kg-1.K-1
      heat_capacity_s.val = 356;       // J.kg-1.K-1
      thermal_cond_l.val  = 1.3;       // W.cm-1.K-1
      thermal_cond_s.val  = 1.3;       // W.cm-1.K-1
      latent_heat.val     = 2600;    // J.cm-3

      num_comps.val = 2;

      solute_diff_0.val    = 2e-5;     // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 1e-5;     // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.094;    // at frac.
      initial_conc_1.val   = 0.107;    // at frac.

      eps_c.val = 1.0e-5;
      eps_v.val = 1.0e-2;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1910;      // K
      liquidus_slope_0.val =-1036;     // K / at frac. - liquidous slope
      liquidus_slope_1.val =-543;      // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.83;    // partition coefficient
      part_coeff_1.val     = 0.94;    // partition coefficient
      break;

    case 4: // Ni - 15.2wt%Al - 5.8wt%Ta
      density_l.val       = 7.365e-3; // kg.cm-3
      density_s.val       = 7.365e-3; // kg.cm-3
      heat_capacity_l.val = 660;      // J.kg-1.K-1
      heat_capacity_s.val = 660;      // J.kg-1.K-1
      thermal_cond_l.val  = 0.8;      // W.cm-1.K-1
      thermal_cond_s.val  = 0.8;      // W.cm-1.K-1
      latent_heat.val     = 2136;     // J.cm-3

      num_comps.val = 2;

      solute_diff_0.val    = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.152;   // wt frac.
      initial_conc_1.val   = 0.058;   // wt frac.

      eps_c.val = 0*2.7207e-5;
      eps_v.val = 0*2.27e-2;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1754;     // K
      liquidus_slope_0.val =-255;     // K / wt frac. - liquidous slope
      liquidus_slope_1.val =-517;     // K / wt frac. - liquidous slope
      part_coeff_0.val     = 0.48;    // partition coefficient
      part_coeff_1.val     = 0.54;    // partition coefficient
      break;

    case 5: // Ni - 5.8wt%Ta - 15.2wt%Al : Rochi- I think there is a typo here alloy is Ni - 5.8 wt% Al - 15.2% wt Ta alloy ref paper: Numerical Investigation of Segregation Evolution during the Vacuum Arc Remelting Process of Ni-Based Superalloy Ingots - Jiang Cui, Baokuan Li, Zhongqiu Liu, Fengsheng Qi, Beijiang Zhang, and Ji Zhang Metals, 2021, 11, 2046
      density_l.val       = 7.365e-3; // kg.cm-3
      density_s.val       = 7.365e-3; // kg.cm-3
      heat_capacity_l.val = 660;      // J.kg-1.K-1
      heat_capacity_s.val = 660;      // J.kg-1.K-1
      thermal_cond_l.val  = 0.40;     //0.8;      // W.cm-1.K-1
      thermal_cond_s.val  = 0.40;     //0.8;      // W.cm-1.K-1
      latent_heat.val     = 2135.8;     // J.cm-3

      num_comps.val = 2;

      solute_diff_0.val    = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 5e-5;    // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.058;   // wt frac.
      initial_conc_1.val   = 0.152;   // wt frac.

      eps_c.val = 2.7207e-5;
      eps_v.val = 2.27e-2;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1754;     // K
      linearized_liquidus.val  = 1;
      liquidus_slope_0.val =-517;     // K / wt frac. - liquidous slope
      liquidus_slope_1.val =-255;     // K / wt frac. - liquidous slope
      const_part_coeff.val = 1;
      part_coeff_0.val     = 0.54;    // partition coefficient
      part_coeff_1.val     = 0.48;    // partition coefficient

      mu_l.val             = 4.9e-5;  // kg cm-1 s-1
      beta_C_0.val         = 2.26;
      beta_C_1.val         = -0.382;
      beta_T.val           = 1.2e-4;


      break;

    case 6: // A made-up tetra-alloy based on Ni - 0.4at%Cu
      density_l.val       = 8.88e-3; // kg.cm-3
      density_s.val       = 8.88e-3; // kg.cm-3
      heat_capacity_l.val = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s.val = 0.46e3;  // J.kg-1.K-1
      latent_heat.val     = 2350;    // J.cm-3
      thermal_cond_l.val  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s.val  = 6.07e-1; // W.cm-1.K-1

      num_comps.val = 3;
      solute_diff_0.val    = 1.e-5;
      solute_diff_1.val    = 2.e-5;
      solute_diff_2.val    = 4.e-5;
      initial_conc_0.val   = 0.1;
      initial_conc_1.val   = 0.1;
      initial_conc_2.val   = 0.1;

      eps_c.val = 0.e-6;
      eps_v.val = 0;
      eps_a.val = 0.0;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1728;    // K
      liquidus_slope_0.val = -300;
      liquidus_slope_1.val = -500;
      liquidus_slope_2.val = -400;
      part_coeff_0.val     = 0.85;
      part_coeff_1.val     = 0.75;
      part_coeff_2.val     = 0.90;
      break;

    case 7: // A made-up penta-alloy based on Ni - 0.4at%Cu
      density_l.val       = 8.88e-3; // kg.cm-3
      density_s.val       = 8.88e-3; // kg.cm-3
      heat_capacity_l.val = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s.val = 0.46e3;  // J.kg-1.K-1
      latent_heat.val     = 2350;    // J.cm-3
      thermal_cond_l.val  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s.val  = 6.07e-1; // W.cm-1.K-1

      num_comps.val = 4;
      solute_diff_0.val    = 1.e-5;
      solute_diff_1.val    = 2.e-5;
      solute_diff_2.val    = 4.e-5;
      solute_diff_3.val    = 6.e-5;
      initial_conc_0.val   = 0.15;
      initial_conc_1.val   = 0.1;
      initial_conc_2.val   = 0.05;
      initial_conc_3.val   = 0.05;

      eps_c.val = 1.e-5;
      eps_v.val = 0;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val    = 1728;    // K

      liquidus_slope_0.val = -300;
      liquidus_slope_1.val = -500;
      liquidus_slope_2.val = -400;
      liquidus_slope_3.val = -600;

      part_coeff_0.val     = 0.85;
      part_coeff_1.val     = 0.75;
      part_coeff_2.val     = 0.90;
      part_coeff_3.val     = 0.80;

      break;
  case 8: // made up alloy -- by Elyce and Rochi for convergence test with fluids
    // Convergence test alloy
    density_l.val       = 1.; // kg.cm-3
    density_s.val       = 1.; // kg.cm-3

    heat_capacity_l.val = 1.;  // J.kg-1.K-1
    heat_capacity_s.val = 1.;  // J.kg-1.K-1

    thermal_cond_l.val  = 1.; // W.cm-1.K-1
    thermal_cond_s.val  = 1.; // W.cm-1.K-1

    latent_heat.val     = 1.;    // J.cm-3

    mu_l.val            = 1.;
//    num_comps.val = 3;

    solute_diff_0.val    = 0.5; // 0.1;  // cm2.s-1 - concentration diffusion coefficient
    solute_diff_1.val    = 0.7;
    solute_diff_2.val    = 1.;

    initial_conc_0.val   = 1.;    // at frac.
    initial_conc_1.val = 1.;
    initial_conc_2.val = 1.;

    eps_c.val = 0.;
    eps_v.val = 0.;
    eps_a.val = 0;
    symmetry.val = 0;

    // linearized phase diagram
    melting_temp.val     = 0;   // K
    linearized_liquidus.val  = 1;
    const_part_coeff.val = 1;

    liquidus_slope_0.val = 0.33;   // K / at frac. - liquidous slope
    liquidus_slope_1.val = 0.33;
    liquidus_slope_2.val = 0.33;

    part_coeff_0.val     = 0.0;   // partition coefficient
    part_coeff_1.val     = 0.0;
    part_coeff_2.val     = 0.;

    // Other parameters needed for the fluids problem:
    Pr.val = 1.0;
    Sc0.val = 1.0;
    Sc1.val = 1.0;
    Sc2.val = 1.0;
    Sc3.val = 1.0;

    Ra_T.val = 100.; // 0.1;
    Ra_C_0.val = 100.; //  0.1;
    Ra_C_1.val = 100.; // 0.1;
    Ra_C_2.val = 100.; // 0.1;

    beta_T.val=1.0;
    beta_C_0.val=1.0;
    beta_C_1.val=1.0;
    beta_C_2.val=1.0;
    beta_C_3.val=1.0;

    // will allow the user to decide from the terminal whether to use boussinesq or not

    break;
    default:
      throw std::invalid_argument("Undefined alloy\n");
  }


  double thermal_diff_l = (thermal_cond_l.val)/(density_l.val * heat_capacity_l.val);

  if(geometry.val!=8){
    Pr.val = (mu_l.val)/(density_l.val * thermal_diff_l);

    Sc0.val = (mu_l.val)/(density_l.val * solute_diff_0.val);
    Sc1.val = (mu_l.val)/(density_l.val * solute_diff_1.val);
    Sc2.val = (mu_l.val)/(density_l.val * solute_diff_2.val);
    Sc3.val = (mu_l.val)/(density_l.val * solute_diff_3.val);
  }

}

double liquidus_value(double *c)
{
  static double conc_term;

  if (linearized_liquidus.val)
  {
    conc_term = melting_temp.val;

    for (int i = 0; i < num_comps.val; ++i) conc_term += (*liquidus_slope_all[i])*c[i];

    return conc_term;
  }
  else
  {
    switch (alloy.val)
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
      case 8: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      default:
        throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

double liquidus_slope(int which_comp, double *c)
{

  if (linearized_liquidus.val)
  {
    switch (which_comp)
    {
      case 0: return liquidus_slope_0.val;
      case 1: return liquidus_slope_1.val;
      case 2: return liquidus_slope_2.val;
      case 3: return liquidus_slope_3.val;
    }
  }
  else
  {
    switch (alloy.val)
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
      case 5: {

        throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      }
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 8: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      default:{

        throw std::invalid_argument("Invalid liquidus surface\n");
      }
    }
  }
}

double part_coeff(int which_comp, double *c)
{
  if (const_part_coeff.val)
  {
    switch (which_comp)
    {
      case 0: return part_coeff_0.val;
      case 1: return part_coeff_1.val;
      case 2: return part_coeff_2.val;
      case 3: return part_coeff_3.val;
    }
  }
  else
  {
    switch (alloy.val)
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
      case 8: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");

      default: throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

/*
void compute_nondimensional_groups(int mpicomm, my_p4est_multialloy_t* multialloy_solver){
// Pr, Le, St, RaT, RaC0, RaC1, RaC2, RaC3
  double thermal_diff_l = thermal_cond_l.val/density_l.val/heat_capacity_l.val;
  double thermal_diff_s = thermal_cond_s.val/density_s.val/heat_capacity_s.val;
  double deltaT = temp_gradient.val * l_char.val;
  double gravity = 9.81;
  PetscPrintf(mpicomm, "RED ALERT: move the defn of gravity to somewhere more reasonable please \n");

  // Prandtl
  double Pr = mu_l.val/(density_l.val * thermal_diff_l);

  // Lewis numbers
  double Le_0 = thermal_diff_l/solute_diff_0.val;
  double Le_1 = thermal_diff_l/solute_diff_1.val;
  double Le_2 = thermal_diff_l/solute_diff_2.val;
  double Le_3 = thermal_diff_l/solute_diff_3.val;

  // Stefan
  //RED ALERT: this definition of Stefan number assumes that the characteristic length scale *is* the
  // length of the computational domain
  PetscPrintf(mpicomm, "RED ALERT: Definition of Stefan number needs to be better refined \n");
  double St = heat_capacity_l.val * (deltaT)/latent_heat.val;

  // Rayleigh numbers
  double RaT = density_l.val * beta_T.val * gravity * deltaT * pow(l_cha
  double deltaC_0 = initial_conc_0.val - eutectic_conc_0.val;
  double deltaC_1 = initial_conc_1.val - eutectic_conc_1.val;
  double deltaC_2 = initial_conc_2.val - eutectic_conc_2.val;r.val, 3.0) / (mu_l.val * thermal_diff_l);
  PetscPrintf(mpicomm, "RED ALERT: delta T definition of Rayleigh number has same assumption as Stefan number \n");

  double deltaC_3 = initial_conc_3.val - eutectic_conc_3.val;


  double RaC_0 = density_l.val * beta_C_0.val * gravity * (initial_conc_0.val - eutectic_conc_0.val) * pow(l_char.val, 3.0)/(mu_l.val * solute_diff_0.val);
  double RaC_1 = density_l.val * beta_C_1.val * gravity * (initial_conc_1.val - eutectic_conc_1.val) * pow(l_char.val, 3.0)/(mu_l.val * solute_diff_1.val);
  double RaC_2 = density_l.val * beta_C_2.val * gravity * (initial_conc_2.val - eutectic_conc_2.val) * pow(l_char.val, 3.0)/(mu_l.val * solute_diff_2.val);
  double RaC_3 = density_l.val * beta_C_3.val * gravity * (initial_conc_3.val - eutectic_conc_3.val) * pow(l_char.val, 3.0)/(mu_l.val * solute_diff_3.val);
  PetscPrintf(mpicomm, "RED ALERT: we gave absolutely no thought to how we defined the eutectic concentrations at this point in time \n");


  PetscPrintf(mpicomm, "Nondimensional groups for fluids problem are: \n"
                       "Pr = %g \n St = %g \n "
                       "RaT = %g \n RaC_0 =%g, RaC_1 = %g, RaC_2 = %g, RaC_3 = %g"
                       "Le_0 = %g, Le_1 = %g, Le_2 = %g, Le_3 = %g \n \n", Pr, St,
              RaT, RaC_0, RaC_1, RaC_2, RaC_3,
              Le_0, Le_1, Le_2, Le_3);

  multialloy_solver->set_nondimensional_groups(Pr, St,
                                               Le_0, Le_1, Le_2, Le_3,
                                               RaT, RaC_0, RaC_1, RaC_2, RaC_3,
                                               deltaT, deltaC_0, deltaC_1, deltaC_2, deltaC_3);
}
*/
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
  switch (geometry.val) {
    case -3: return .5*(exp(-x*x)/x - sqrt(PI)*boost::math::erfc(x));
    case -2: return boost::math::expint(1, x*x);
    case -1: return boost::math::erfc(x);
    default: return 0;
  }
}

double Fp(double x)
{
  switch (geometry.val) {
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

  as = thermal_cond_s.val/density_s.val/heat_capacity_s.val;
  al = thermal_cond_l.val/density_l.val/heat_capacity_l.val;

//  Tstar = melting_temp.val;
  Bl = 0;

  for (int j = 0; j < num_comps.val; ++j) {
    Cstar[j] = (*initial_conc_all[j]);
  }

  for (int jj = 0; jj < 20; ++jj) {
    for (int j = 0; j < num_comps.val; ++j) {
      kp[j] = part_coeff(j, Cstar);
    }
    for (int j = 0; j < num_comps.val; ++j) {
      Cstar[j] = (*initial_conc_all[j])/(1. + 2.*(1.-kp[j])*xF_Fp(sqrt(lam/(*solute_diff_all[j]))));
    }
  }

  for (int j = 0; j < num_comps.val; ++j)
  {
//    Cstar[j] = (*initial_conc_all[j])/(1. + 2.*(1.-(*part_coeff_all[j]))*xF_Fp(sqrt(lam/(*solute_diff_all[j]))));
//    Tstar = Tstar + (*liquidus_slope_all[j])*Cstar[j];
    Ai[j] = (*initial_conc_all[j]);
    Bi[j] = (Cstar[j]-(*initial_conc_all[j]))/F(sqrt(lam/(*solute_diff_all[j])));

//    Bl = Bl + (*liquidus_slope_all[j])*Bi[j]*xFp(sqrt(lam/(*solute_diff_all[j])));
    Bl = Bl + liquidus_slope(j, Cstar)*Bi[j]*xFp(sqrt(lam/(*solute_diff_all[j])));
  }

  Tstar = liquidus_value(Cstar);

  Bl = Bl/xFp(sqrt(lam/al))/M;
  Bs = Bl * thermal_cond_l.val/thermal_cond_s.val * sqrt(as/al) * Fp(sqrt(lam/al))/Fp(sqrt(lam/as)) + latent_heat.val/(density_s.val*heat_capacity_s.val) * 2*sqrt(lam/as)/Fp(sqrt(lam/as));

  Al = Tstar - Bl*F(sqrt(lam/al));
  As = Tstar - Bs*F(sqrt(lam/as));

  time_limit.val = .25*SQR(container_radius_inner.val + front_location_final.val)/lam-t_start;
}

inline double nu(double a, double t, double r) { return r/sqrt(4.*a*(t+t_start)); }

inline double tl_exact(double t, double r) { return Al + Bl * F(nu(al,t,r)); }
inline double ts_exact(double t, double r) { return As + Bs * F(nu(as,t,r)); }

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

// ----------------------------------------
// auxiliary fxn for calculating the Rayleigh numbers
// ----------------------------------------
void calculate_Ra_numbers(){

  // Set the Cinf and Tinf values (in case of doing boussinesq)
  if(geometry.val == 8){
    Tinf.val = 1;
    Cinf0.val = 1;
    Cinf1.val = 1;
    Cinf3.val = 1;
  }
  else{
    Tinf.val = analytic::Tstar;
    Cinf0.val = initial_conc_0.val;
    Cinf1.val = initial_conc_1.val;
    Cinf2.val = initial_conc_2.val;
    Cinf3.val = initial_conc_3.val;
  }



  if(geometry.val!=8){
    // We don't do this for the convergence test
    if(l_char.val<0) throw std::invalid_argument("you need to set l_char to run this case \n");
    if(beta_T.val<0) throw std::invalid_argument("you need to set betaT to run this case \n");
    if(beta_C_0.val>9.e7) throw std::invalid_argument("you need to set beta_C_0 to run this case \n");
    if(num_comps.val>1 && beta_C_1.val>9.e7) throw std::invalid_argument("you need to set beta_C_1 to run this case \n");
    if(num_comps.val>2 && beta_C_2.val>9.e7) throw std::invalid_argument("you need to set beta_C_2 to run this case \n");
    if(num_comps.val>3 && beta_C_3.val>9.e7) throw std::invalid_argument("you need to set beta_C_3 to run this case \n");

    if(Tinf.val<0) throw std::invalid_argument("you need to set Tinf to run this case \n");
    if(Cinf0.val<0) throw std::invalid_argument("you need to set Cinf0 to run this case \n");
    if(num_comps.val>1 && Cinf1.val<0) throw std::invalid_argument("you need to set Cinf1 to run this case \n");
    if(num_comps.val>2 && Cinf2.val<0) throw std::invalid_argument("you need to set Cinf2 to run this case \n");
    if(num_comps.val>3 && Cinf3.val<0) throw std::invalid_argument("you need to set Cinf3 to run this case \n");

    double thermal_diff_l = thermal_cond_l.val / (density_l.val * heat_capacity_l.val);
    Ra_T.val = (density_l.val * beta_T.val * gravity_.val * Tinf.val * pow(l_char.val, 3.))/(mu_l.val * thermal_diff_l);

    Ra_C_0.val = (density_l.val * beta_C_0.val * gravity_.val * Cinf0.val * pow(l_char.val, 3.))/(mu_l.val * solute_diff_0.val);
    Ra_C_1.val = (density_l.val * beta_C_1.val * gravity_.val * Cinf1.val * pow(l_char.val, 3.))/(mu_l.val * solute_diff_1.val);
    Ra_C_2.val = (density_l.val * beta_C_2.val * gravity_.val * Cinf2.val * pow(l_char.val, 3.))/(mu_l.val * solute_diff_2.val);
    Ra_C_3.val = (density_l.val * beta_C_3.val * gravity_.val * Cinf3.val * pow(l_char.val, 3.))/(mu_l.val * solute_diff_3.val);
  }

}

// ----------------------------------------
// convergence test class
// ----------------------------------------
class Convergence_soln{
  public:
    struct temperature: CF_DIM{
        const double factor=1.;
        const double N=1.;
        const unsigned char dom;
        bool make_Ts_zero = false;
        temperature(const unsigned char& dom_) : dom(dom_){
            P4EST_ASSERT(dom >= 0 && dom <=1);
        }
        double T(DIM(double x, double y,double z))const{
            switch(dom){
                case LIQUID_DOMAIN:
//                  return 0.;
                  return sin(x)*sin(y)*(x + (cos(N*y)*cos(t)*cos(x - PI/2.))/factor);

//                  return cos(N*PI*t*x*y)*sin(N*PI*t*x*y)+((x-1)*(y+1))/factor;
                case SOLID_DOMAIN:
                  if(make_Ts_zero) return 0.;
                  return cos(x)*cos(y)*((sin(N*y)*cos(t)*sin(x - PI/2))/factor - 1);

//                  return sin(N*PI*t*x*y)+((x-1)*(y+1))/factor;
                default:
                  throw std::runtime_error("analytical solution temperature unknown domain \n");
            }
        }
        double operator()(DIM(double x, double y, double z))const{
            return T(DIM(x,y,z));
        }
        double dT_d(const unsigned char& dir, DIM(double x,double y, double z)) const{
            switch(dom){
                case LIQUID_DOMAIN:
                    switch(dir){
                        case dir::x:
//                          return 0.;
                          return (sin(y)*(factor*sin(x) + factor*x*cos(x) + 2*cos(N*y)*cos(t)*cos(x)*sin(x)))/factor;

//                          return (y+N*factor*PI*t*y*cos(2*N*PI*t*x*y)+1)/factor;
                        case dir::y:
//                          return 0.;
                          return (sin(x)*(factor*x*cos(y) + cos(N*y)*cos(t)*cos(y)*sin(x) - N*sin(N*y)*cos(t)*sin(x)*sin(y)))/factor;
//                          return (x+N*factor*PI*t*x*cos(2*N*PI*t*x*y)-1)/factor;
                        default:
                           throw std::runtime_error("dT_d of analytical temperature field: unrecognized Cartesian direction \n");
                    }
                case SOLID_DOMAIN:
                    switch(dir){
                        case dir::x:
                          if(make_Ts_zero) return 0.;
                          return (cos(y)*sin(x)*(factor + 2*sin(N*y)*cos(t)*cos(x)))/factor;

//                          return (y+1)/factor + N*PI*t*y*cos(N*PI*t*x*y);
                        case dir::y:
                          if(make_Ts_zero) return 0.;
                          return (cos(x)*(factor*sin(y) + sin(N*y)*cos(t)*cos(x)*sin(y) - N*cos(N*y)*cos(t)*cos(x)*cos(y)))/factor;

//                          return (x-1)/factor + N*PI*t*x*cos(N*PI*t*x*y);
                        default:
                           throw std::runtime_error("dT_d of analytical temperature field: unrecognized Cartesian direction \n");
                    }
            }
        }
        double dT_dt(DIM(double x, double y, double z)) const{
            switch(dom){
                case LIQUID_DOMAIN:
//                  return 0.;
                  return -(cos(N*y)*sin(t)*cos(x - PI/2)*sin(x)*sin(y))/factor;
//                  return N*PI*x*y*pow(cos(N*PI*t*x*y),2) - N*PI*x*y*pow(sin(N*PI*t*x*y),2);
                case SOLID_DOMAIN:
                  if(make_Ts_zero) return 0.;
                  return -(sin(N*y)*cos(x)*cos(y)*sin(t)*sin(x - PI/2))/factor;
//                  return N*PI*x*y*cos(N*PI*t*x*y);
                default:
                    throw std::runtime_error("dT_dt in analytical temperature: unrecognized domain \n");
            }
        }
        double laplace(DIM(double x, double y, double z)) const {
            switch(dom){
                case LIQUID_DOMAIN:
//                  return 0.;
                  return -( 2*factor*x*sin(x)*sin(y) -
                           2*factor*cos(x)*sin(y) -
                           2*cos(N*y)*cos(t)*pow(cos(x),2)*sin(y) +
                           3*cos(N*y)*cos(t)*pow(sin(x),2)*sin(y) +
                           2*N*sin(N*y)*cos(t)*cos(y)*pow(sin(x),2) +
                           pow(N,2)*cos(N*y)*cos(t)*pow(sin(x),2)*sin(y))
                         /factor;
//                  return -2*pow(N,2)*pow(PI,2)*pow(t,2)*sin(2*N*PI*t*x*y)*(pow(x,2)+pow(y,2));
                case SOLID_DOMAIN:
                  if(make_Ts_zero) return 0.;
                  return (2*factor*cos(x)*cos(y) -
                          2*sin(N*y)*cos(t)*cos(y) +
                          5*sin(N*y)*cos(t)*pow(cos(x),2)*cos(y) +
                          2*N*cos(N*y)*cos(t)*pow(cos(x),2)*sin(y) +
                          pow(N,2)*sin(N*y)*cos(t)*pow(cos(x),2)*cos(y))
                         /factor;

//                  return -pow(N,2)*pow(PI,2)*pow(t,2)*sin(N*PI*t*x*y)*(pow(x,2)+pow(y,2));
                default:
                    throw std::runtime_error("laplace for analytical temperature field: unrecognized domain \n");
            }
        }
    };
    struct concentration: CF_DIM{
        const double factor=1.;
        const double N=1.;
        const unsigned char comp;
        concentration(const unsigned char& comp_) : comp(comp_){
            P4EST_ASSERT(comp == 0 || comp == 1 || comp == 2);
            // A WARNING TO NEW IMPLEMENTERS!!! If you want to add a convergence study for more than 3 concentration fields, you will need to update the CF_DIM* array sizes in multialloy corresponding to the convergence_external_source_conc_robin and convergence_external_source_conc
            // As of now, those arrays are hard coded to be size 3 (sorry about that), and the functions in multialloy
            // for setting those fields also may assume only a certain size.
            // so make sure to change that before proceeding!! -- Elyce
        }

        double C0(DIM(double x, double y, double z))const{
          //return cos(y)*sin(x)*cos(t) + 5; // works but not using this because this field is same as velocity
          return cos(t)*( x*cos(y)*sin(x) + y*sin(y)*cos(x) )+ 5; // works
        }

        double C1(DIM(double x, double y, double z))const{
          //return cos(x)*cos(t)*sin(y) + 5; // works but not using this because this field is same as velocity
          //return cos(t)*( y*cos(y)*sin(x) + x*sin(y)*cos(x) ) + 5; // works
          return cos(t)*(x*cos(y)*sin(x) - y*sin(y)*cos(x))+ 5; // works
        }

        double C2(DIM(double x, double y, double z))const{
//          return cos(y)*sin(x)*cos(t) + 5; // works but not using this because this field is same as velocity
          return cos(t) * (y + cos(5*x)) + 5;

        }

        double dC0_d(const unsigned char& dir, DIM(double x,double y,double z)) const{
          switch(dir){
              case dir::x:
                //return cos(x)*cos(y)*cos(t); // works but not using this because this field is same as velocity
                return cos(t)*(x*cos(x)*cos(y) + sin(x)*(cos(y)-y*sin(y))); // works
              case dir::y:
                //return -sin(x)*sin(y)*cos(t); // works but not using this because this field is same as velocity
                return cos(t)*(cos(x)*(sin(y)+y*cos(y))- x*sin(x)*sin(y)); // works
              default:
                throw std::runtime_error("dC0_d of analytical concentration1 field: unrecognized Cartesian");
              }
        }
        double dC1_d(const unsigned char& dir, DIM(double x,double y,double z)) const{
            switch(dir){
              case dir::x:
                //return -cos(t)*sin(x)*sin(y); // works but not using this because this field is same as velocity
                //return cos(t)*(cos(x)*(sin(y)+y*cos(y))-x*sin(x)*sin(y)); // works
                return cos(t)*(x*cos(x)*cos(y) + sin(x)*(y*sin(y)+cos(y)));// works
              case dir::y:
                //return cos(x)*cos(y)*cos(t); // works but not using this because this field is same as velocity
                //return cos(t) *(x*cos(x)*cos(y)+sin(x)*(cos(y)-y*sin(y))); // works
                return -cos(t)*(x*sin(x)*sin(y) + cos(x)*(sin(y)+cos(y)*y));// works
              default:
                throw std::runtime_error("dC1_d of analytical concentration2 field: unrecognized Cartesian direction \n");
              }
        }
        double dC2_d(const unsigned char& dir, DIM(double x,double y,double z)) const{
          switch(dir){
          case dir::x:
            return -5 * sin(5*x) * cos(t);
//            return cos(x)*cos(y)*cos(t); // works but not using this because this field is same as velocity
          case dir::y:
            return cos(t);
            //return -sin(x)*sin(y)*cos(t); // works but not using this because this field is same as velocity
//            printf("dC2_dy: Insert code for c2 case! \n");
//            return 0. * x * y;

          default:
            throw std::runtime_error("dC2_d of analytical concentration2 field: unrecognized Cartesian direction \n");
          }
        }

        double dC_d(const unsigned char& dir, DIM(double x,double y,double z)) const{
          switch(comp){
          case 0:
            return dC0_d(dir, DIM(x,y,z));
          case 1:
            return dC1_d(dir, DIM(x,y,z));
          case 2:
            return dC2_d(dir, DIM(x,y,z));
          default:
            throw std::invalid_argument("Convergence_soln:concentration dC_d() -- unrecognized component number \n");

          }
//          return (comp==0? dC0_d(dir, DIM(x,y,z)) : dC1_d(dir, DIM(x,y,z)) );
        }

        double dC0_dt(DIM(double x, double y, double z)) const{
          //return cos(y)*(-sin(t))*sin(x); // works but not using this because this field is same as velocity
          return -sin(t)*( x*cos(y)*sin(x) + y*sin(y)*cos(x) ); // works
        }
        double dC1_dt(DIM(double x, double y, double z)) const{
          //return -cos(x)*sin(y)*sin(t); // works but not using this because this field is same as velocity
          //return -sin(t)*( y*cos(y)*sin(x) + x*sin(y)*cos(x) ) ; // works
          return sin(t)*(y*cos(x)*sin(y) - x*sin(x)*cos(y)); // works
        }
        double dC2_dt(DIM(double x, double y, double z)) const{
          return -sin(t)*(y + cos(5*x));
          //return cos(y)*(-sin(t))*sin(x); // works but not using this because this field is same as velocity
//          printf("dC2_dt: Insert code for c2 case! \n");
//          return 0. * x * y;
        }
        double dC_dt(DIM(double x, double y, double z)) const{
          switch(comp){
            case 0:
              return dC0_dt(DIM(x,y,z));
            case 1:
              return dC1_dt(DIM(x,y,z));
            case 2:
              return dC2_dt(DIM(x,y,z));
            default:
              throw std::invalid_argument("Convergence_soln:concentration dC_d() -- unrecognized component number \n");

          }
//          return (comp == 0? dC0_dt(DIM(x,y,z)) : dC1_dt(DIM(x,y,z)));
        }

        double laplace_C0(DIM(double x, double y, double z))const{
          //return -2*cos(y)*sin(x)*cos(t); // works but not using this because this field is same as velocity
          return -2* cos(t) *(x *cos(y)* sin(x) + cos(x)* (-2 *cos(y) + y *sin(y))); // works
        }
        double laplace_C1(DIM(double x, double y, double z))const{
          //return -2*cos(x)*cos(t)*sin(y); // works but not using this because this field is same as velocity
          //return -2 *cos(t)*(y*sin(x)*cos(y)+sin(y)*(2*sin(x)+x*cos(x))); // works
          return 2*cos(t)*(y*cos(x)*sin(y) - x*sin(x)*cos(y));// works
        }
        double laplace_C2(DIM(double x, double y, double z))const{
          return -25*cos(5*x)*cos(t);
          //return -2*cos(y)*sin(x)*cos(t); // works but not using this because this field is same as velocity
//          printf("laplace_C2: Insert code for c2 case! \n");
//          return 0. * x * y;
        }
        double laplace(DIM(double x, double y, double z))const{
          switch(comp){
            case 0:
              return laplace_C0(DIM(x,y,z));
            case 1:
              return laplace_C1(DIM(x,y,z));
            case 2:
              return laplace_C2(DIM(x,y,z));
            default:
              throw std::invalid_argument("Convergence_soln:concentration() -- unrecognized component number \n");
          }
//          return (comp == 0? laplace_C0(DIM(x,y,z)) : laplace_C1(DIM(x,y,z)));
        }

        double operator()(DIM(double x,double y, double z))const{
          switch(comp){
            case 0:
              return C0(DIM(x,y,z));
            case 1:
              return C1(DIM(x,y,z));
            case 2:
              return C2(DIM(x,y,z));
            default:
              throw std::invalid_argument("Convergence_soln:concentration() -- unrecognized component number \n");
          }
//          return (comp == 0? C0(DIM(x,y,z)) : C1(DIM(x,y,z)));
        }
    };
    struct interface_velocity: CF_DIM{
//      public:
          double operator()(DIM(double x,double y,double z)) const{
            return 1.;
          }
    } /*convergence_vgamma*/;

    struct velocity_component: CF_DIM{
        const unsigned char dir;
        const double factor=1.0;
        velocity_component(const unsigned char& dir_): dir(dir_){
            P4EST_ASSERT(dir<P4EST_DIM);
        }
        double v(DIM(double x,double y,double z))const{
            switch(dir){
                case dir::x:
                  return cos(t) * cos(y) * sin(x);
                case dir::y:
                  return - cos(t) * cos(x) * sin(y); // works
                default:
                    throw std::runtime_error("analytical solution of velocity: unknown cartesian direction \n");
            }
        }
        double operator()(DIM(double x, double y, double z))const{
            return v(DIM(x,y,z));
        }
        double dv_d(const unsigned char& dirr, DIM(double x,double y, double z)){
            switch(dir){
                case dir::x: {
                    switch(dirr){
                        case dir::x: // du_dx
                          return cos(t) * cos(x) * cos(y); // works
                        case dir::y: // du_dy
                          return -cos(t)*sin(x)*sin(y); // works
                        default:
                            throw std::runtime_error("dvx_d not defined in this direction \n)");
                    }
                }
                case dir::y: {
                    switch(dirr){
                        case dir::x: // dv_dx
                          return cos(t)*sin(x)*sin(y); // works
                        case dir::y: // dv_dy
                          return -cos(t)*cos(x)*cos(y); // works
                        default:
                            throw std::runtime_error("dvx_d not defined in this direction \n)");
                    }
                }
                default:
                    throw std::runtime_error("analytical solution of dv_d :unknown cartesian direction \n");
            }
        }
       double dv_dt(DIM(double x, double y, double z)){
           switch(dir){
                case dir::x: // du_dt
                  return -sin(t)*cos(y)*sin(x); // works
                case dir::y:
                  return sin(t)*cos(x)*sin(y); // works
                default:
                    throw std::runtime_error("analytical solution of velocity: unknown cartesian direction \n");
            }
       }
       double laplace(DIM(double x, double y, double z)){
           switch(dir){
               case dir::x:{
                 return -2.*cos(t)*cos(y)*sin(x); // works
               }
               case dir::y:{
                 return 2*cos(t)*cos(x)*sin(y); // works
               }
               default:{
                   throw std::runtime_error("laplace of velocity not defined in this direction \n");
               }
           }
       }
    };
    struct external_force_temperature: CF_DIM{
        const unsigned char dom;
        temperature* temperature_;
        velocity_component* velocity_component_;
        external_force_temperature(const unsigned char &dom_, temperature* analytical_T, velocity_component* analytical_v):
            dom(dom_),temperature_(analytical_T),velocity_component_(analytical_v){
            P4EST_ASSERT(dom>=0 && dom<=1);
        }
        double operator()(DIM(double x, double y, double z)) const {
            double advective_term;
            switch(dom){
            case LIQUID_DOMAIN:
              advective_term = (velocity_component_[dir::x])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN].dT_d(dir::x, x, y) +
                              (velocity_component_[dir::y])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN].dT_d(dir::y, x, y);

//                advective_term = 0.;
                break;
            case SOLID_DOMAIN:
                advective_term= 0.;
                break;
            default:
                throw std::runtime_error("external heat source : advective term : unrecognized domain \n");
            }

            return (temperature_[dom].dT_dt(DIM(x,y,z)) + advective_term - temperature_[dom].laplace(DIM(x,y,z)));
        }
    };
    struct external_force_concentration: CF_DIM{
        concentration concentration_;
        velocity_component* velocity_component_;
        external_force_concentration(concentration concentration=NULL,
                                     velocity_component* velocity_component=NULL ):
            concentration_(concentration), velocity_component_(velocity_component){
          P4EST_ASSERT(concentration_.comp <3);
        }
        double operator()(DIM(double x, double y, double z)) const {
            double advective_term;

            int comp = concentration_.comp;
            double D = 0./*(comp == 0? solute_diff_0.val : solute_diff_1.val)*/;
//            printf("external source comp = %d \n", comp);
            switch(comp){
              case 0:
                D = solute_diff_0.val; break;
              case 1:
                D = solute_diff_1.val; break;
              case 2:
                D = solute_diff_2.val; break;
              default:
                throw std::invalid_argument("external_force_concentration: unrecognized component number for deterimining solute diffusivity \n");
            }

            advective_term = (velocity_component_[dir::x])(DIM(x,y,z)) * concentration_.dC_d(dir::x, x, y) +
                             (velocity_component_[dir::y])(DIM(x,y,z)) * concentration_.dC_d(dir::y, x, y);

//            advective_term = 0.;

            return concentration_.dC_dt(DIM(x,y,z)) + advective_term - D*concentration_.laplace(DIM(x,y,z));
        }
    };
    struct pressure_field: CF_DIM{
      public:
      double operator()(DIM(double x,double y, double z)) const {
        return 0.0;
      }
      double gradP(const unsigned char& dir,DIM(double x, double y, double z)){
        return 0.0;
      }
    }pressure_field_analytical;

    struct external_force_NS: CF_DIM{
        const unsigned dir;
        velocity_component* velocity_field;
        temperature temperature_;
        concentration* concentrations_;

        external_force_NS(const unsigned char& dir_,
                          velocity_component* analytical_v,
                          temperature temperature = NULL,
                          concentration* concentrations = NULL ):dir(dir_),
                                                         velocity_field(analytical_v),
                                                         temperature_(temperature),
                                                         concentrations_(concentrations)
        {
            P4EST_ASSERT(dir<P4EST_DIM);

        }
        double operator()(DIM(double x, double y, double z)) const{
          double main_term = velocity_field[dir].dv_dt(DIM(x,y,z)) +
                             SUMD((velocity_field[0])(DIM(x,y,z))*velocity_field[dir].dv_d(dir::x,DIM(x,y,z)),
                                  (velocity_field[1])(DIM(x,y,z))*velocity_field[dir].dv_d(dir::y,DIM(x,y,z)),
                                  (velocity_field[2])(DIM(x,y,z))*velocity_field[dir].dv_d(dir::z,DIM(x,y,z))) -
                             velocity_field[dir].laplace(DIM(x,y,z));
          double boussinesq_term = 0.;

          const double RaC_vals[4] = {Ra_C_0.val, Ra_C_1.val, Ra_C_2.val, Ra_C_3.val};
          const double Cinf_vals[4] = {Cinf0.val, Cinf1.val, Cinf2.val, Cinf3.val};
          const double Sc_vals[4] = {Sc0.val, Sc1.val, Sc2.val, Sc3.val};
          double thermal_diff_l = thermal_cond_l.val / (density_l.val * heat_capacity_l.val);
//          const double solute_diff_vals[4] = {solute_diff_0.val, solute_diff_0.val, Cinf2.val, Cinf3.val};

          if(do_boussinesq.val && (dir == dir::y)){

//            printf("Inside source: Ra_T = %0.2e \n ", Ra_T.val);

            boussinesq_term += (((temperature_)(DIM(x,y,z)))/Tinf.val - 1.) * Ra_T.val * Pr.val;

            for (int j = 0; j<num_comps.val; j++){
//              printf("RaC %d = %0.2e \n", j, RaC_vals[j]);
              boussinesq_term+= (((concentrations_[j])(DIM(x,y,z)))/Cinf_vals[j] - 1.) * RaC_vals[j] * Sc_vals[j] * pow((thermal_diff_l/ *solute_diff_all[j]), 2.);
            }
          }
//          printf("Inside source: main term = %0.2e, bouss term = %0.2e \n", main_term, boussinesq_term);
          return main_term + boussinesq_term;
        }
    };

    // Below are the external source terms associated with BC's (i.e. Robin BC source terms, jump condition source terms, etc. etc.)

    // NOte: from poisson nodes mls main file, we see the convention of jump = [Plus - minus], which we interpret to mean [solid domain - liquid domain]
    struct external_source_temperature_jump : CF_DIM{
      temperature* temperature_;

      external_source_temperature_jump(temperature* temperature=NULL): temperature_(temperature){}

      double operator()(DIM(double x, double y, double z)) const{
//        printf("(%0.2f, %0.2f) jump in temp = %0.2e \n", x, y, (temperature_[SOLID_DOMAIN](DIM(x,y,z)) - temperature_[LIQUID_DOMAIN](DIM(x,y,z))));
          return (temperature_[SOLID_DOMAIN](DIM(x,y,z)) - temperature_[LIQUID_DOMAIN](DIM(x,y,z)));

      }
    };

    struct external_source_temperature_flux_jump : CF_DIM{
      temperature* temperature_;
      interface_velocity* vgamma_;

      my_p4est_interpolation_nodes_t* nx_interp=NULL;
      my_p4est_interpolation_nodes_t* ny_interp=NULL;

      external_source_temperature_flux_jump(temperature* temperature=NULL, interface_velocity* vgamma=NULL) : temperature_(temperature), vgamma_(vgamma){}

      void set_inputs(my_p4est_node_neighbors_t* ngbd, Vec& nx, Vec& ny){
        nx_interp = new my_p4est_interpolation_nodes_t(ngbd);
        ny_interp = new my_p4est_interpolation_nodes_t(ngbd);

        nx_interp->set_input(nx, linear);
        ny_interp->set_input(ny, linear);
      }
      void clear_inputs(){
        nx_interp->clear();
        delete nx_interp;
        nx_interp=NULL;

        ny_interp->clear();
        delete ny_interp;
        ny_interp=NULL;
      }
      double operator() (DIM(double x, double y, double z)) const{
        if(nx_interp == NULL || ny_interp == NULL){
          throw std::runtime_error("external source temperature flux jump: you cannot call this operator because the normal interpolators have not been set \n");
        }
//        printf("accesses temp flux term \n");

        double source_x, source_y;

        // x-dir jump in flux
        source_x = (temperature_[SOLID_DOMAIN].dT_d(dir::x, DIM(x,y,z)) - temperature_[LIQUID_DOMAIN].dT_d(dir::x, DIM(x,y,z)));

        // y-dir jump in flux
        source_y = (temperature_[SOLID_DOMAIN].dT_d(dir::y, DIM(x,y,z)) - temperature_[LIQUID_DOMAIN].dT_d(dir::y, DIM(x,y,z)));

//        printf("source_x = %0.2f, source_y = %0.2f \n", source_x, source_y);
        // actual normal direction jump minux the interface velocity --> to get our source term
//        printf("latent heat * rho * vn = %0.2e \n", latent_heat.val * density_s.val * (*vgamma_)(DIM(x,y,z)));

        double dT_dn_jump = (source_x  * (*nx_interp)(DIM(x,y,z))) + (source_y * (*ny_interp)(DIM(x,y,z)));

        double source_term = dT_dn_jump + latent_heat.val * (*vgamma_)(DIM(x,y,z)); // density_s.val * (*vgamma_)(DIM(x,y,z));

//        printf("(%0.4f, %0.4f) nx = %0.2f, ny = %0.2f, dT_dn_jump = %0.4e, extra stuff = %0.4e , vn = %0.4e \n", x, y, (*nx_interp)(DIM(x,y,z)), (*ny_interp)(DIM(x,y,z)), dT_dn_jump,- latent_heat.val * density_s.val * (*vgamma_)(DIM(x,y,z)),(*vgamma_)(DIM(x,y,z))  );

        return source_term;

      }

    }; // end of external source temperature flux jump

    struct external_source_concentration_robin : CF_DIM{
      concentration* concentration_;
      interface_velocity* vgamma_;

      my_p4est_interpolation_nodes_t* nx_interp=NULL;
      my_p4est_interpolation_nodes_t* ny_interp=NULL;


      external_source_concentration_robin(concentration* concentration=NULL, interface_velocity* vgamma=NULL): concentration_(concentration), vgamma_(vgamma){}

      void set_inputs(my_p4est_node_neighbors_t* ngbd, Vec& nx, Vec& ny){
        nx_interp = new my_p4est_interpolation_nodes_t(ngbd);
        ny_interp = new my_p4est_interpolation_nodes_t(ngbd);

        nx_interp->set_input(nx, linear);
        ny_interp->set_input(ny, linear);
      }
      void clear_inputs(){
        nx_interp->clear();
        delete nx_interp;
        nx_interp=NULL;

        ny_interp->clear();
        delete ny_interp;
        ny_interp=NULL;
      }

      double operator()(DIM(double x, double y, double z)) const {
        if(nx_interp==NULL || ny_interp==NULL){
//          printf("operator -- comp = %d \n", concentration_->comp);
          throw std::runtime_error("external source concentration robin: you cannot call this operator because the normal interpolators have not been set \n");
        }
//        printf("accesses conc robin source term \n");

        int comp = concentration_->comp;
//        double D = (comp == 0? solute_diff_0.val : solute_diff_1.val);
//        double part_coeff = (comp == 0? part_coeff_0.val : part_coeff_1.val);

        double D = 0; double part_coeff = 0.;
//        printf("hello hello comp = %d \n", comp);
//        if(comp == 2){
//          printf("\n \n HERE HERE COMP = 2 \n \n");
//        }
        switch(comp){
          case 0:
            D = solute_diff_0.val;
            part_coeff = part_coeff_0.val;
            break;
          case 1:
            D = solute_diff_1.val;
            part_coeff = part_coeff_1.val;
            break;
          case 2:
//            printf("gets into case 2! \n");
            D = solute_diff_2.val;
            part_coeff = part_coeff_2.val;
            break;
          default:
            throw std::invalid_argument("external_force_concentration: unrecognized component number for deterimining solute diffusivity and partition coefficient \n");
        }
//          printf("comp = %d, D = %0.2e \n", comp, D);


//        printf("nx interp = %p \n", nx_interp);
//        printf("ny interp = %p \n", ny_interp);
//        printf("nx interp = %0.2f \n", (*nx_interp)(DIM(x,y,z)));
//        printf("ny interp = %0.2f \n", (*ny_interp)(DIM(x,y,z)));
//        printf("concentration = %0.2f \n", (*concentration_)(DIM(x,y,z)));
//        printf("vgamma = %0.2f \n", (*vgamma_)(DIM(x,y,z)));

//        printf("concentration's t = %0.2f \n", concentration_->t);

        double DdC_dn = D*(concentration_->dC_d(dir::x, DIM(x,y,z))  * (*nx_interp)(DIM(x,y,z))  +
                             concentration_->dC_d(dir::y, DIM(x,y,z))  * (*ny_interp)(DIM(x,y,z)));

        double other_term = (1. - part_coeff)*(*vgamma_)(DIM(x,y,z))*(*concentration_)(DIM(x,y,z));

        double source_term = DdC_dn - other_term;

//        printf("component = %d, source term = %0.2f, vgamma = %0.2f, concentration = %0.2f, DdC_dn = %0.2f, other_term = %0.2f -- at (%0.4f, %0.4f), t = %0.4f \n",
//               concentration_->comp,
//               source_term, (*vgamma_)(DIM(x,y,z)), (*concentration_)(DIM(x,y,z)), DdC_dn, other_term, x, y, concentration_->t);
        return source_term;
      }
    };

    struct external_source_Gibbs_Thomson : CF_DIM{
      temperature* temperature_l_;
      concentration* concentration_0_;
      concentration* concentration_1_;
      concentration* concentration_2_;

      interface_velocity* vgamma_;

      external_source_Gibbs_Thomson(temperature* temperature_l = NULL,
                                    interface_velocity* vgamma=NULL,
                                    concentration* concentration_0 = NULL,
                                    concentration* concentration_1 = NULL,
                                    concentration* concentration_2 = NULL):
          temperature_l_(temperature_l),
          concentration_0_(concentration_0),
          concentration_1_(concentration_1),
          concentration_2_(concentration_2),
          vgamma_(vgamma){}
      // NOTE: this function assumes there is no curvature affect on the Gibbs

      double operator()(DIM(double x, double y, double z)) const {
        if(temperature_l_== NULL ||
            concentration_0_ == NULL ||
            concentration_1_ == NULL ||
            vgamma_ == NULL){
          throw std::invalid_argument("external source Gibbs Thomson : you cannot return the operator because some of the dependent fiels are null \n");
        }
        if(num_comps.val > 2 && concentration_2_== NULL){
          throw std::invalid_argument("external source Gibbs Thomson : There is no field provided for concentration 2 but the number of components mismatches this \n");
        }

        if(eps_c.val>0){
          throw std::invalid_argument("external source gibbs thomson: you are trying to run with an eps_c value greater than 0, but the curvature affect is not implemented in this convergence test");
        }

        double Tm_expression = melting_temp.val;
        if(num_comps.val>0){Tm_expression+= liquidus_slope_0.val * (*concentration_0_)(DIM(x,y,z));}
        if(num_comps.val>1){Tm_expression+= liquidus_slope_1.val * (*concentration_1_)(DIM(x,y,z));}
        if(num_comps.val>2){Tm_expression+= liquidus_slope_2.val * (*concentration_2_)(DIM(x,y,z));}

        double Gibbs = (*temperature_l_)(DIM(x,y,z)) - Tm_expression - eps_v.val * (*vgamma_)(DIM(x,y,z));

//        printf("From BC:\n temperature = %0.4e \n", (*temperature_l_)(DIM(x,y,z)));
//        printf("liquidus = %0.4e \n", Tm_expression);
//        printf("vgamma = %0.4e \n", (*vgamma_)(DIM(x,y,z)));
//        printf("Gibbs = %0.4e \n", Gibbs);

        return Gibbs;
      }

    };

};

// Concentrations:
Convergence_soln::concentration convergence_conc0(0);
Convergence_soln::concentration convergence_conc1(1);
Convergence_soln::concentration convergence_conc2(2);
Convergence_soln::concentration convergence_concentrations[3] = {convergence_conc0,
                                                                 convergence_conc1,
                                                                 convergence_conc2};

// Temperatures:
Convergence_soln::temperature convergence_tl(LIQUID_DOMAIN);
Convergence_soln::temperature convergence_ts(SOLID_DOMAIN);
//Convergence_soln::temperature convergence_ts(LIQUID_DOMAIN);

Convergence_soln::temperature convergence_temp[2] = {convergence_tl, convergence_ts};

// Interfacevelocity:
Convergence_soln::interface_velocity convergence_vgamma;

// Fluid velocity:
Convergence_soln::velocity_component convergence_vx(dir::x);
Convergence_soln::velocity_component convergence_vy(dir::y);
Convergence_soln::velocity_component convergence_vel[2] = {convergence_vx, convergence_vy};

// External source terms:
Convergence_soln::external_force_concentration convergence_force_c0(convergence_conc0, convergence_vel);
Convergence_soln::external_force_concentration convergence_force_c1(convergence_conc1, convergence_vel);
Convergence_soln::external_force_concentration convergence_force_c2(convergence_conc2, convergence_vel);

CF_DIM* convergence_forces_conc[3] = {&convergence_force_c0, &convergence_force_c1, &convergence_force_c2};
//std::vector<CF_DIM*> convergence_forces_conc = {&convergence_force_c0, &convergence_force_c1, &convergence_force_c2};

Convergence_soln::external_force_temperature convergence_force_tl(LIQUID_DOMAIN, convergence_temp, convergence_vel);
Convergence_soln::external_force_temperature convergence_force_ts(SOLID_DOMAIN, convergence_temp, convergence_vel);
CF_DIM* convergence_forces_temp[2] = {&convergence_force_tl, &convergence_force_ts};

Convergence_soln::external_force_NS external_force_NS_x(dir::x, convergence_vel, convergence_tl, convergence_concentrations);
Convergence_soln::external_force_NS external_force_NS_y(dir::y, convergence_vel, convergence_tl, convergence_concentrations);
CF_DIM* convergence_force_NS[2]= {&external_force_NS_x, &external_force_NS_y};

//Convergence_soln::external_force_NS* convergence_force_NS[2];

// External interface bc terms:
Convergence_soln::external_source_temperature_jump external_source_temperature_jump(convergence_temp);
Convergence_soln::external_source_temperature_flux_jump external_source_temperature_flux_jump(convergence_temp, &convergence_vgamma); // note: this one requires us to "set_inputs" and "clear_inputs"

Convergence_soln::external_source_concentration_robin external_source_c0_robin(&convergence_conc0, &convergence_vgamma);
Convergence_soln::external_source_concentration_robin external_source_c1_robin(&convergence_conc1, &convergence_vgamma);
Convergence_soln::external_source_concentration_robin external_source_c2_robin(&convergence_conc2, &convergence_vgamma);

CF_DIM* external_source_conc_robin[3] = {&external_source_c0_robin, &external_source_c1_robin, &external_source_c2_robin};
//std::vector<CF_DIM*> external_source_conc_robin = {&external_source_c0_robin, &external_source_c1_robin, &external_source_c2_robin};

Convergence_soln::external_source_Gibbs_Thomson external_source_Gibbs_Thomson(&convergence_tl,  &convergence_vgamma, &convergence_conc0, &convergence_conc1, &convergence_conc2);


void check_convergence_errors_and_save_to_vtk(my_p4est_multialloy_t* mas, mpi_environment_t& mpi, int iter,
                                              FILE* fich_errors=NULL, char name_errors[]=NULL){

  PetscPrintf(mpi.comm(), "Checking convergence fields errors ... \n");
  vec_and_ptr_t phi;

  vec_and_ptr_t tl, ts, c0, c1, c2, vgamma;
  vec_and_ptr_dim_t u;


  vec_and_ptr_t tl_ana, ts_ana, c0_ana, c1_ana, c2_ana, vgamma_ana;
  vec_and_ptr_dim_t u_ana;

  vec_and_ptr_t tl_err, ts_err, c0_err, c1_err, c2_err, vgamma_err;
  vec_and_ptr_dim_t u_err;

  p4est_t* p4est = mas->get_p4est();
  p4est_nodes_t* nodes = mas->get_nodes();
  my_p4est_node_neighbors_t* ngbd = mas->get_ngbd();

  bool do_c1 = num_comps.val > 1;
  bool do_c2 = num_comps.val > 2;

  tl_ana.create(p4est,nodes); ts_ana.create(p4est, nodes);
  c0_ana.create(p4est, nodes);
  if(do_c1) c1_ana.create(p4est, nodes);
  if(do_c2) c2_ana.create(p4est, nodes);

  vgamma_ana.create(p4est, nodes);
  u_ana.create(p4est, nodes);

  double dt = mas->get_dt();

  // update the time held by all the fields we want to use:
  convergence_temp[LIQUID_DOMAIN].t = tn + dt;
  convergence_temp[SOLID_DOMAIN].t = tn + dt;

  convergence_conc0.t = tn + dt;
  if(do_c1) convergence_conc1.t = tn + dt;
  if(do_c2) convergence_conc2.t = tn + dt;

  // interface vel
  convergence_vgamma.t = tn + dt;

  // ns vels
  foreach_dimension(d){
    convergence_vel[d].t = tn + dt;
  }

  sample_cf_on_nodes(p4est, nodes, convergence_temp[LIQUID_DOMAIN], tl_ana.vec);
  sample_cf_on_nodes(p4est, nodes, convergence_temp[SOLID_DOMAIN], ts_ana.vec);
  sample_cf_on_nodes(p4est, nodes, convergence_conc0, c0_ana.vec);
  if(do_c1) sample_cf_on_nodes(p4est, nodes, convergence_conc1, c1_ana.vec);
  if(do_c2) sample_cf_on_nodes(p4est, nodes, convergence_conc2, c2_ana.vec);

  sample_cf_on_nodes(p4est, nodes, convergence_vgamma, vgamma_ana.vec);

  foreach_dimension(d){
    sample_cf_on_nodes(p4est, nodes, convergence_vel[d], u_ana.vec[d]);
  }

  tl_err.create(p4est, nodes); ts_err.create(p4est, nodes);
  c0_err.create(p4est, nodes);
  if(do_c1) c1_err.create(p4est, nodes);
  if(do_c2) c2_err.create(p4est, nodes);

  vgamma_err.create(p4est, nodes);
  u_err.create(p4est, nodes);

  // Get the fields from multialloy:
  phi.vec = mas->get_front_phi();
  tl.vec = mas->get_tl();
  ts.vec = mas->get_ts();

  c0.vec = mas->get_cl(0);
  if(do_c1) c1.vec = mas->get_cl(1);
  if(do_c2) c2.vec = mas->get_cl(2);

  vgamma.vec = mas->get_normal_velocity();


  // We scale the front velo by -1 bc mas multiplies it by the normal outward from the fluid domain, and we want to check the opposite
  VecScaleGhost(vgamma.vec, -1.);

  u = mas->get_v_n_NS();

  // Initialize the error values to zero:
  VecSet(tl_err.vec, 0.);
  VecSet(ts_err.vec, 0.);

  VecSet(c0_err.vec, 0.);
  if(do_c1) VecSet(c1_err.vec, 0.);
  if(do_c2) VecSet(c2_err.vec, 0.);

  VecSet(vgamma_err.vec, 0.);
  foreach_dimension(d){
    VecSet(u_err.vec[d], 0.);
  }

  phi.get_array();

  tl.get_array(); ts.get_array();
  c0.get_array();
  if(do_c1) c1.get_array();
  if(do_c2) c2.get_array();

  vgamma.get_array(); u.get_array();

  tl_ana.get_array(); ts_ana.get_array();
  c0_ana.get_array();
  if(do_c1) c1_ana.get_array();
  if(do_c2) c2_ana.get_array();

  vgamma_ana.get_array(); u_ana.get_array();

  tl_err.get_array(); ts_err.get_array();
  c0_err.get_array();
  if(do_c1) c1_err.get_array();
  if(do_c2) c2_err.get_array();
  vgamma_err.get_array(); u_err.get_array();

  double xyz_min[P4EST_DIM];
  dxyz_min(p4est, xyz_min);

  double dxyz_close_to_interface = 4.0 * MIN(xyz_min[0], xyz_min[1]);


  double tl_Linf = 0.; double ts_Linf = 0.;
  double c0_Linf = 0.;
  double c1_Linf = 0.;
  double c2_Linf = 0.;
  double vgamma_Linf = 0.;
  double vx_Linf = 0.;  double vy_Linf = 0.;
  double vgamma_max = 0.;
  foreach_node(n, nodes){
    if(phi.ptr[n] < 0.){
      tl_err.ptr[n] = fabs(tl.ptr[n] - tl_ana.ptr[n]);
      c0_err.ptr[n] = fabs(c0.ptr[n] - c0_ana.ptr[n]);
      if(do_c1) c1_err.ptr[n] = fabs(c1.ptr[n] - c1_ana.ptr[n]);
      if(do_c2) c2_err.ptr[n] = fabs(c2.ptr[n] - c2_ana.ptr[n]);

      foreach_dimension(d){
        u_err.ptr[d][n] = fabs(u.ptr[d][n] - u_ana.ptr[d][n]);
      }

      tl_Linf = max(tl_Linf, tl_err.ptr[n]);
      c0_Linf = max(c0_Linf, c0_err.ptr[n]);
      if(do_c1) c1_Linf = max(c1_Linf, c1_err.ptr[n]);
      if(do_c2) c2_Linf = max(c2_Linf, c2_err.ptr[n]);

      vx_Linf = max(vx_Linf, u_err.ptr[0][n]);
      vy_Linf = max(vy_Linf, u_err.ptr[1][n]);
    }
    else{
      ts_err.ptr[n] = (ts.ptr[n] - ts_ana.ptr[n]);
      ts_Linf = max(ts_Linf, ts_err.ptr[n]);
    }
//    printf("Rank %d has vgamma = %0.2e \n", p4est->mpirank, vgamma.ptr[n]);
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){
      vgamma_err.ptr[n] = fabs(vgamma.ptr[n] - vgamma_ana.ptr[n]);

//      if(vgamma.ptr[n]!=0) vgamma_Linf = max(vgamma_Linf, vgamma_err.ptr[n]);
      vgamma_max = max(vgamma_max, vgamma.ptr[n]);
    }
  }

//  VecView(vgamma.vec, PETSC_VIEWER_STDOUT_WORLD);



  int ierr;
  ierr = VecGhostUpdateBegin(tl_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(ts_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(c0_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(do_c1) ierr = VecGhostUpdateBegin(c1_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(do_c2) ierr = VecGhostUpdateBegin(c2_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(vgamma_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  foreach_dimension(d){
    ierr = VecGhostUpdateBegin(u_err.vec[d], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecGhostUpdateEnd(tl_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(ts_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateEnd(c0_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(do_c1) ierr = VecGhostUpdateEnd(c1_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(do_c2) ierr = VecGhostUpdateEnd(c2_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateEnd(vgamma_err.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  foreach_dimension(d){
    ierr = VecGhostUpdateEnd(u_err.vec[d], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // Calculate vgamma error once all the ghost values have been updated:
  foreach_local_node(n, nodes){
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface ){
//      double xyz_n[P4EST_DIM];
//      node_xyz_fr_n(n, p4est, nodes, xyz_n);
//      bool print_stuff = fabs(vgamma_err.ptr[n]) > 0.1;
//      if(print_stuff) printf("node %d (%0.3f, %0.3f) rank %d : vgamma error = %0.3e \n",
//             n, xyz_n[0], xyz_n[1], mpi.rank(),  vgamma_err.ptr[n]);
      vgamma_Linf = max(vgamma_Linf, vgamma_err.ptr[n]);
    }
  }
//  printf("rank %d, Linf_vgamma = %0.3e \n", mpi.rank(), vgamma_Linf);

  phi.restore_array();

  tl.restore_array(); ts.restore_array();
  c0.restore_array();
  if(do_c1) c1.restore_array();
  if(do_c2) c2.restore_array();
  vgamma.restore_array(); u.restore_array();

  tl_ana.restore_array(); ts_ana.restore_array();
  c0_ana.restore_array();
  if(do_c1) c1_ana.restore_array();
  if(do_c2) c2_ana.restore_array();
  vgamma_ana.restore_array(); u_ana.restore_array();

  tl_err.restore_array(); ts_err.restore_array();
  c0_err.restore_array();
  if(do_c1) c1_err.restore_array();
  if(do_c2) c2_err.restore_array();
  vgamma_err.restore_array(); u_err.restore_array();

  // Scale back vgamma now that we are done with it
  // (Not sure this is entirely necessary, i think we might just have a copy of it and not the actual vec, I need to double check)
  VecScaleGhost(vgamma.vec, -1.);

//  vgamma_Linf = fabs(fabs(vgamma_max)-1);
//  printf("Now rank %d has vgamma_Linf = %0.3e \n", p4est->mpirank, vgamma_Linf);


  // Get the global Linf errors:
  double local_Linf_errors[8] = {tl_Linf, ts_Linf, c0_Linf, c1_Linf, c2_Linf, vgamma_Linf, vx_Linf, vy_Linf};
  double global_Linf_errors[8] = {0., 0., 0., 0., 0., 0., 0., 0.};

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors, global_Linf_errors, 8, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE, &num_nodes, 1, MPI_INT, MPI_SUM, p4est->mpicomm);

//  PetscPrintf(p4est->mpicomm, "vgamma error = %0.3e = %0.3e \n", vgamma_Linf, global_Linf_errors[4]);
//  printf("vgamma Linf on rank %d is %0.3e \n", p4est->mpirank, vgamma_Linf);

  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n"
                              "Errors on Coupled Validation "
                              "\n -------------------------------------\n"
                              "Linf on Tl: %0.3e \n"
                              "Linf on Ts: %0.3e \n"
                              "Linf on C0: %0.3e \n"
                              "Linf on C1: %0.3e \n"
                              "Linf on C2: %0.3e \n"
                              "Linf on vgamma: %0.3e \n"
                              "Linf on vx: %0.3e \n"
                              "Linf on vy: %0.3e \n \n"
                              "Number grid points used: %d \n"
                              "dxyz_min : %0.3e \n",
              global_Linf_errors[0], global_Linf_errors[1], global_Linf_errors[2],
              global_Linf_errors[3], global_Linf_errors[4], global_Linf_errors[5],
              global_Linf_errors[6], global_Linf_errors[7],
              num_nodes, min(xyz_min[0], xyz_min[1]));

  // Print errors to file:
  if(fich_errors!=NULL && name_errors!=NULL){
    ierr = PetscFOpen(p4est->mpicomm, name_errors,"a",&fich_errors);CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, fich_errors, "%g %g %d "
                                                     "%g %g %g "
                                                     "%g %g %g "
                                                     "%g %g %d %g \n",
                        tn, mas->get_dt(), iter,
                        global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                        global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],
                        global_Linf_errors[6], global_Linf_errors[7],
                        num_nodes, min(xyz_min[0], xyz_min[1]));CHKERRXX(ierr);
    ierr = PetscFClose(p4est->mpicomm,fich_errors); CHKERRXX(ierr);
  }



  if(save_vtk.val){
    // Save to vtk:
    const char* out_dir = getenv("OUT_DIR");
    if (!out_dir)
    {
      ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
      return;
    }
    char name[1000];
    sprintf(name, "%s/vtu/multialloy_convergence_test_lvl_%d_%d.%05d", out_dir, lmin.val , lmax.val, iter);

    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields;

    point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_ana.vec, "tl_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(tl.vec, "tl"));
    point_fields.push_back(Vec_for_vtk_export_t(tl_err.vec, "tl_err"));

    point_fields.push_back(Vec_for_vtk_export_t(ts_ana.vec, "ts_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(ts.vec, "ts"));
    point_fields.push_back(Vec_for_vtk_export_t(ts_err.vec, "ts_err"));

    point_fields.push_back(Vec_for_vtk_export_t(c0_ana.vec, "c0_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(c0.vec, "c0"));
    point_fields.push_back(Vec_for_vtk_export_t(c0_err.vec, "c0_err"));

    if(do_c1){
      point_fields.push_back(Vec_for_vtk_export_t(c1_ana.vec, "c1_ana"));
      point_fields.push_back(Vec_for_vtk_export_t(c1.vec, "c1"));
      point_fields.push_back(Vec_for_vtk_export_t(c1_err.vec, "c1_err"));
    }

    if(do_c2){
      point_fields.push_back(Vec_for_vtk_export_t(c2_ana.vec, "c2_ana"));
      point_fields.push_back(Vec_for_vtk_export_t(c2.vec, "c2"));
      point_fields.push_back(Vec_for_vtk_export_t(c2_err.vec, "c2_err"));
    }

    point_fields.push_back(Vec_for_vtk_export_t(vgamma_ana.vec, "vgamma_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(vgamma.vec, "vgamma"));
    point_fields.push_back(Vec_for_vtk_export_t(vgamma_err.vec, "vgamma_err"));

    point_fields.push_back(Vec_for_vtk_export_t(u_ana.vec[0], "vx_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(u.vec[0], "vx"));
    point_fields.push_back(Vec_for_vtk_export_t(u_err.vec[0], "vx_err"));

    point_fields.push_back(Vec_for_vtk_export_t(u_ana.vec[1], "vy_ana"));
    point_fields.push_back(Vec_for_vtk_export_t(u.vec[1], "vy"));
    point_fields.push_back(Vec_for_vtk_export_t(u_err.vec[1], "vy_err"));


    my_p4est_vtk_write_all_lists(p4est, nodes, ngbd->get_ghost(),
                                 P4EST_TRUE, P4EST_TRUE,
                                 name, point_fields, cell_fields);
  }


  // Destroy the vectors that we created to evaluate the convergence errors:
  tl_ana.destroy(); ts_ana.destroy();
  c0_ana.destroy();
  if(do_c1) c1_ana.destroy();
  if(do_c2) c2_ana.destroy();
  vgamma_ana.destroy(); u_ana.destroy();

  tl_err.destroy(); ts_err.destroy();
  c0_err.destroy();
  if(do_c1) c1_err.destroy();
  if(do_c2) c2_err.destroy();
  vgamma_err.destroy(); u_err.destroy();

}




// ----------------------------------------
// define problem geometry
// ----------------------------------------
double scaling() { return 1./box_size.val; }
bool periodicity(int dir)
{
  switch (geometry.val) {
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
    case  5: return (dir == 0? 1: 0); /*true*/;
    case  6: return (dir == 0 ? 1 : 0);
    case  7: return (dir == 0 ? 1 : 0);
    case  8: return false;
    default: throw;
  }
}

param_t<double> num_seeds_directional  (pl, 10., "num_seeds_directional", "Number of seeds for directional cases run by Elyce \n");


double phi_directional_seeds(double x, double y){
  const int num_seeds=20;
  double yshift_val = 0.9*front_location(); //front_location.val;
//  double xshifts[5] = {0.1*xmax.val, 0.3*xmax.val, 0.5*xmax.val, 0.7*xmax.val, 0.9*xmax.val};
//  double yshifts[5] = {yshift_val, yshift_val, yshift_val, yshift_val, yshift_val};
//  seed_radius.val = 0.001;


  // many seeds case:
  double xshifts[num_seeds];
  double yshifts[num_seeds];

  double dx = (1.0/(num_seeds+1))*xmax.val;

  // adjust radius if need be:
  seed_radius.val = dx * 0.0001;
  if(dx < seed_radius.val){
    printf("seed_radius = %0.3e, dx = %0.3e \n", seed_radius.val, dx);
    seed_radius.val = 0.25 *dx;

//    seed_radius.val = 0.25*dx;
//    printf("ALERT!!!!! SEED RADIUS HAS BEEN CHANGED TO %0.5e \n", seed_radius.val);
  }

  for(int i=0; i<num_seeds; i++){
    xshifts[i] = (i+1)*dx ;/* * xmax.val*/;
//    printf("xshifts %d = %0.4f \n", i, xshifts[i]);
    yshifts[i] = yshift_val;
  }

//  // First, get all the relevant LSF values for each seed:
  double LSF_vals[num_seeds];
  double current_min =1.0e9;/* -(y - front_location())*/; //1.0e9;
  bool point_inside_seed=false;
  for(int n=0; n < num_seeds; n++){
    double r = sqrt(SQR(x - xshifts[n]) + SQR(y - yshifts[n]));
    LSF_vals[n] = seed_radius.val - r;

//    if((y < front_location())){ // (r > seed_radius.val) &&
//      LSF_vals[n] = -(y - front_location()) /*+ fabs(LSF_vals[n])*/;
//    }

    if(r < seed_radius.val){
      point_inside_seed=true;
    }

    if(fabs(LSF_vals[n]) < fabs(current_min)){
      current_min = LSF_vals[n];
    }
  }

  if(y < front_location() && !point_inside_seed){
    return (front_location() - current_min);
  }
  else{
    if(fabs(y - front_location()) < 0.1){
      return 0.1*(front_location() + current_min);
    }
    else{
      return front_location() + current_min;
    }
  }
//    double noise = 0.001;

//    return -(y - front_location()) + noise*front_location()*sin(5.*PI*(x + 0.1));
//  return current_min;

//  return -(y-front_location());



//  return -(y - front_location());

}


class front_phi_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry.val)
    {
#ifdef P4_TO_P8
      case -3: return analytic::rf_exact(t) - ABS3(x-xc(),
                                                   y-yc(),
                                                   z-zc());
#endif
      case -2: return analytic::rf_exact(t) - ABS2(x-xc(),
                                                   y-yc());
      case -1: return analytic::rf_exact(t) - ABS1(y-ymin());
      case 0: {

        //
//        return phi_directional_seeds(x,y);
        double frequency = floor(num_seeds_directional.val/2.);

        // Normal directional:
        if(x < 5./frequency || x>xmax.val - 5./frequency){
          return -(y - front_location());
        }
        else{
          return -(y - front_location()*(1. +  0.5* abs(sin(frequency * PI * x))));

        }

        //-(y - front_location()) + 0.001/(1.+100.*fabs(x/(xmin.val+xmax.val)-.5))*double(rand())/double(RAND_MAX)  + 0.001/(1.+1000.*fabs(x/(xmin.val+xmax.val)-.75));
      }
      case 1: return -(ABS2(x-xc(), y-yc())-seed_radius());
      case 2: return  (ABS2(x-xc(), y-yc())-(container_radius_outer()-front_location()));
      case 3: return  (ABS2(x-xc(), y-yc())-(container_radius_outer()-front_location()));
      case 4: return -(ABS2(x-xc(), y-yc())-(container_radius_inner()+front_location()));
      case 5:
      {
        double dist0 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*0. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*0. + seed_rot())));
        double dist1 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*1. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*1. + seed_rot())));
        double dist2 = ABS2(x-(xc()+seed_dist()*cos(2.*PI/3.*2. + seed_rot())), y-(yc()+seed_dist()*sin(2.*PI/3.*2. + seed_rot())));

        //return seed_radius() - MIN(dist0, dist1, dist2);
        //return -(ABS2(x-xc(), y-yc())-seed_radius());
        double noise = 0.001;
        double theta = atan2(y-yc(),x-xc());
        //return r0*(1.0 - noise*fabs(sin(theta)) - noise*fabs(cos(theta))) - sqrt(SQR(x - xc) + SQR(y - yc));
        return seed_radius()*(1.0 - noise*fabs(pow(sin(2*theta),2)))- sqrt(SQR(x - xc()) + SQR(y - yc()));
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
      case 8:{
//        if(y < yc()){
//          return 1.;
//        }
//        else{
//          return 0.;
//        }
        return 1.*(seed_radius() - sqrt(SQR(x - xc()) + SQR(y - yc())))/*-(ABS2(x-xc(), y-yc())-seed_radius())*/;
//          return y - yc();


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
    switch (geometry.val)
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
      case  8: return -100*(xmax()); // We multiply by xmax bc the -1 by itself was causing some issues .. (altho we don't remember ... )(to revisit)
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
    switch (geometry.val)
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
      case 8 : return 0;
      default: throw;
    }
  }
} seed_number_cf;

int num_seeds()
{
  switch (geometry.val)
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
    case 8: return 1;
    default: throw;
  }
}

double theta0(int seed)
{
  if (seed_type.val == 0) return crystal_orientation.val;
  else
  {
    switch (geometry.val)
    {
#ifdef P4_TO_P8
      case -3: return crystal_orientation.val;
#endif
      case -2: return crystal_orientation.val;
      case -1: return crystal_orientation.val;
      case 0:
        switch (seed)
        {
          case 0: return crystal_orientation.val -PI/6.;
          case 1: return crystal_orientation.val +PI/6.;
          default: throw;
        }
      case 1: return crystal_orientation.val;
      case 2: return crystal_orientation.val;
      case 3: return crystal_orientation.val;
      case 4: return crystal_orientation.val;
      case 5:
        switch (seed)
        {
          case 0: return crystal_orientation.val -PI/7.;
          case 1: return crystal_orientation.val +PI/6.;
          case 2: return crystal_orientation.val -PI/5.;
          default: throw;
        }
      case 6:
        switch (seed)
        {
          case 0: return crystal_orientation.val -PI/7.;
          case 1: return crystal_orientation.val +PI/6.;
          case 2: return crystal_orientation.val -PI/5.;
          case 3: return 0.;
          default: throw;
        }
      case 7:
        switch (seed)
        {
          case 0: return crystal_orientation.val -PI/7.;
          case 1: return crystal_orientation.val +PI/6.;
          case 2: return 0.;
          default: throw;
        }
      case 8: return 0.; // we just have uniform growth for this case
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
    return eps_c.val*(1.0-4.0*eps_a.val*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_c.val*(1.-15.*eps_a.val*cos(symmetry.val*(theta-theta0)))/(1.+15.*eps_a.val);
    //return eps_c.val;
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
    return eps_v.val*(1.0-4.0*eps_a.val*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_v.val*(1.-15.*eps_a.val*cos(symmetry.val*(theta-theta0)))/(1.+15.*eps_a.val);
#endif
  }
};

// ----------------------------------------
// define initial fields and boundary conditions
// ----------------------------------------
// FLuid concentration
class cl_cf_t : public CF_DIM
{
  int idx;
  Convergence_soln::concentration* concentration;
public:
    cl_cf_t(int idx, Convergence_soln::concentration* concentration_=NULL ) :
        idx(idx), concentration(concentration_){
//      concentration = new Convergence_soln::concentration((unsigned char)idx);
    }/*, concentration(concentration_*/
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry()) {
#ifdef P4_TO_P8
      case -3: return analytic::cl_exact(idx, t, ABS3(x-xc(), y-yc(), z-zc()));
#endif
      case -2: return analytic::cl_exact(idx, t, ABS2(x-xc(), y-yc()));
      case -1: return analytic::cl_exact(idx, t, ABS1(y-ymin()));
      case  0: //return 0.05 + (1 - 0.05)*0.5 * (1 - tanh(10.*(y - 0.5*front_location.val)));
        if (start_from_moving_front.val) {
          return (*initial_conc_all[idx])*(1. + (1.-analytic::kp[idx])/analytic::kp[idx]*
                                           exp(-cooling_velocity.val/(*solute_diff_all[idx])*(-front_phi_cf(DIM(x,y,z)-0*cooling_velocity.val*t))));
        }else{
          return *initial_conc_all[idx];
        }
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7:
          return *initial_conc_all[idx];
      case  8:{
//        Convergence_soln::concentration concentration((const unsigned char) idx);
//        concentration.

        return (*concentration)(DIM(x,y,z));
      }

      default: throw;
    }
  }
};
//Convergence_soln::concentration convergence_conc0(0);
cl_cf_t cl_cf_0(0, &convergence_conc0);
cl_cf_t cl_cf_1(1, &convergence_conc1);
cl_cf_t cl_cf_2(2, &convergence_conc2);
cl_cf_t cl_cf_3(3);

CF_DIM* cl_cf_all[] = { &cl_cf_0,
                        &cl_cf_1,
                        &cl_cf_2,
                        &cl_cf_3 };

// Solid concentration
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
      case  7:return analytic::Cstar[idx];
      case  8: return 0. ; // E: not sure what this is ...
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
// Liquid temp:
class tl_cf_t : public CF_DIM
{
  Convergence_soln::temperature* temperature_;
public:
  tl_cf_t(Convergence_soln::temperature* temperature=NULL): temperature_(temperature){}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry())
    {
#ifdef P4_TO_P8
      case -3: return analytic::tl_exact(t, ABS3(x-xc.val, y-yc.val, z-zc.val));
#endif
      case -2: return analytic::tl_exact(t, ABS2(x-xc.val, y-yc.val));
      case -1: return analytic::tl_exact(t, ABS1(y-ymin.val));
      case  0: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();//*analytic::Tstar+*/ front_location.val - y +exp(-25*(pow(x-2,2)+pow(y-0.5*front_location.val,2)));//front_location.val - y; +exp(-25*(pow(x-2,2)+pow(y-0.5*front_location.val,2)));//analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())));
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())));
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  7: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  8: return (*temperature_)(DIM(x,y,z));
      default: throw;
    }
  }
} /*tl_cf*/;
tl_cf_t tl_cf(&convergence_tl);

// Solid temp:
class ts_cf_t : public CF_DIM
{
  Convergence_soln::temperature* temperature_;
public:
  ts_cf_t(Convergence_soln::temperature* temperature=NULL):temperature_(temperature){}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry())
    {
#ifdef P4_TO_P8
      case -3: return analytic::ts_exact(t, ABS3(x-xc(), y-yc(), z-zc()));
#endif
      case -2: return analytic::ts_exact(t, ABS2(x-xc(), y-yc()));
      case -1: return analytic::ts_exact(t, ABS1(y-ymin()));
      case  0: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())))*thermal_cond_l()/thermal_cond_s();
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())))*thermal_cond_l()/thermal_cond_s();
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  7: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  8: return (*temperature_)(DIM(x,y,z));
      default: throw;
    }
  }
} /*ts_cf*/;
ts_cf_t ts_cf(&convergence_ts);


// Solid tf: (temperature at which alloy solidified)
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
      case -1: return analytic::Tstar;
      case  0: return analytic::Tstar;//return 0.;
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7: return analytic::Tstar;
      case  8: return 0;
      default: throw;
    }
  }
} tf_cf;

// Front velo norm:
class v_cf_t : public CF_DIM
{
  cf_value_type_t what;
  Convergence_soln::interface_velocity* vgamma_;
public:
    v_cf_t(cf_value_type_t what, Convergence_soln::interface_velocity* vgamma) :
        what(what), vgamma_(vgamma) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry()) {
#ifdef P4_TO_P8
      case -3:
        switch (what) {
          case VAL: return analytic::vn_exact(t);
          case DDX: return analytic::vn_exact(t)*(x-xc.val)/(EPS+ABS3(x-xc.val, y-yc.val, z-zc.val));
          case DDY: return analytic::vn_exact(t)*(y-yc.val)/(EPS+ABS3(x-xc.val, y-yc.val, z-zc.val));
          case DDZ: return analytic::vn_exact(t)*(z-zc.val)/(EPS+ABS3(x-xc.val, y-yc.val, z-zc.val));
          default:  return 0;
        }
#endif
      case -2:
        switch (what) {
          case VAL: return analytic::vn_exact(t);
          case DDX: return analytic::vn_exact(t)*(x-xc.val)/(EPS+ABS2(x-xc.val, y-yc.val));
          case DDY: return analytic::vn_exact(t)*(y-yc.val)/(EPS+ABS2(x-xc.val, y-yc.val));
          default:  return 0;
        }
      case -1:
        switch (what) {
          case VAL:
          case DDY: return analytic::vn_exact(t);
          default:  return 0;
        }
      case 0:
        if (start_from_moving_front.val) {
          switch (what) {
            case VAL:
            case DDY: return cooling_velocity.val;
            default:  return 0;
          }
        }
        else{
          return 0.;
        }

      break;
      case 8:{
        switch(what){
        case VAL: return (*vgamma_)(DIM(x,y,z));
          case DDX: return 0.;
          case DDY: return 0.;
          default: return 0.;

        }

      }
      default: return 0;
    }
  }
};

v_cf_t vn_cf(VAL, &convergence_vgamma );
v_cf_t vx_cf(DDX, &convergence_vgamma);
v_cf_t vy_cf(DDY, &convergence_vgamma);
v_cf_t vz_cf(DDZ, &convergence_vgamma);

// Solid front velo norm:
class vf_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry.val) {
#ifdef P4_TO_P8
      case -3: return -analytic::vf_exact(ABS3(x-xc.val, y-yc.val, z-zc.val));
#endif
      case -2: return -analytic::vf_exact(ABS2(x-xc.val, y-yc.val));
      case -1: return -analytic::vf_exact(ABS1(y-ymin.val));
      case  0:
        if (start_from_moving_front.val) {
          return -cooling_velocity.val;
        } else {return 0;}
      default: return 0;
    }
  }
} vf_cf;

// Time at which solid solidified?
class ft_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry.val) {
#ifdef P4_TO_P8
      case -3: return analytic::ft_exact(ABS3(x-xc.val, y-yc.val, z-zc.val));
#endif
      case -2: return analytic::ft_exact(ABS2(x-xc.val, y-yc.val));
      case -1: return analytic::ft_exact(ABS1(y-ymin.val));
      case  0:
        if (start_from_moving_front.val) {
          return (y-front_location.val) / cooling_velocity.val;
        }else {return 0;}
      default: return 0;
    }
  }
} ft_cf;

double smooth_start(double t) { return smoothstep(smoothstep_order.val, (t+EPS)/(starting_time.val+EPS)); }

// --------------------------
// Type:
class BC_WALL_TYPE_TEMP: public WallBCDIM
{
  public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(geometry.val){
      case 5:{
        if((fabs(y-ymax.val) < EPS)){
          return DIRICHLET;
        }
        else{
          return NEUMANN;
        }
      }
      case 8:{
        bc_type_temp.val = DIRICHLET;
        return DIRICHLET;
      }
      default:{
        return bc_type_temp.val;
      }
    }
    //return bc_type_temp.val;
  }
}bc_wall_type_temp_;


class BC_WALL_TYPE_CONC: public WallBCDIM
{
  public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
//    printf("bc wall type conc accessed \n");
    switch(geometry.val){
      case 5:{
        if((fabs(y-ymax.val) < EPS)){
          return DIRICHLET;
        }
        else{
          return NEUMANN;
        }
      }
      case 8:{
        bc_type_conc.val = DIRICHLET;
        return DIRICHLET;
      }
      default:{
        return bc_type_conc.val;
      }
    } // end of switch case
    //return bc_type_conc.val;
  }
}bc_wall_type_conc_;



// -------------------
class bc_value_temp_t : public CF_DIM
{
public:
//    const unsigned char dom_;
//    Convergence_soln** convergence_soln_;

    // NOTE: [Elyce] you will notice here that the boundary conditions are not trivially added for cases where you may not know if you are in the solid or liquid domain at a given location. The reason this has been avoided up till this point is because all of Daniil's examples have periodic x walls and he knows that the bottom boundary will be solid and the top boundary will be liquid.
    // However, this will not hold in a general setting. If you wanted to run a general simulation where the BC's at any given wall depend on the phase, and the location of the interface is not analytically known, the approach would be to take the Vec phi as an input to this boundary condition class, and have this class own a my_p4est_interpolation_nodes_t object which could interpolate the phi value to your given xy point of interest and then apply a BC value depending on whether phi is less than or greater than zero.
    // If you find yourself needing to implement such a thing, please refer to the class my_p4est_stefan_with_fluids_t and particularly, the
    // fluid velocity interface BC which uses such a strategy to interpolate the interfacial velocity to use in the navier stokes velocity boundary condition at the interface.


  Convergence_soln::temperature* temperature_;

  my_p4est_interpolation_nodes_t* phi_interp;
  bc_value_temp_t(Convergence_soln::temperature* temperature=NULL) : temperature_(temperature){}
  void set_inputs(my_p4est_node_neighbors_t* ngbd, Vec& phi){
    phi_interp = new my_p4est_interpolation_nodes_t(ngbd);
    phi_interp->set_input(phi, linear);
  }
  void clear_inputs(){
    phi_interp->clear();
    delete phi_interp;
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    // case 5 is treated seperately because we need different BC on different walls
    if (geometry.val==5){
      if((fabs(y-ymax.val) < EPS)){
        return analytic::Tstar; // for DIRICHLET case
      }
      else{
        return 0.0; // for NEUMANN case
      }
    }
    switch (bc_type_temp.val) {
      case DIRICHLET:
        switch (geometry.val) {
#ifdef P4_TO_P8
          case -3: return ABS3(x-xc.val,y-yc.val,z-zc.val) < .5*(container_radius_inner.val+container_radius_outer.val) ?
                  analytic::ts_exact(t, container_radius_inner.val) :
                  analytic::tl_exact(t, container_radius_outer.val);
#endif
          case -2: return ABS2(x-xc.val,y-yc.val) < .5*(container_radius_inner.val+container_radius_outer.val) ?
                  analytic::ts_exact(t, container_radius_inner.val) :
                  analytic::tl_exact(t, container_radius_outer.val);
          case -1: return y < .5*(ymin.val+ymax.val) ? analytic::ts_exact(t, 0) : analytic::tl_exact(t, ymax.val-ymin.val);
          case  0: return y < .5*(ymin.val+ymax.val) ? analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()
                                                     : analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*(temp_gradient()*thermal_cond_l.val + cooling_velocity.val*latent_heat.val)/thermal_cond_s.val;
          case  1:
          case  2:
          case  3:
          case  4: return 0;
          case  5: return 0;
          case  6:
          case  7:
          case  8: {
            if(temperature_ == NULL){
              throw std::invalid_argument("bc_value_temp_t for geom 8: you must provide the analytical temperature fields \n");
            }
            if(phi_interp == NULL){
              throw std::invalid_argument("bc_value_temp_t for geom 8: you must provide the level set function for determining the domain \n");
            }
            double phi_val = (*phi_interp)(DIM(x,y,z));
//            printf("(%0.2f, %0.2f) -- phi val = %0.2e \n", x,y, phi_val);
            if(phi_val<0.){
              return temperature_[LIQUID_DOMAIN](DIM(x,y,z));
            }
            else{
              return temperature_[SOLID_DOMAIN](DIM(x,y,z));
            }
            // return (*temperature_)(DIM(x,y,z));
          }
          default: throw;
        }
      case NEUMANN:
        switch (geometry.val) {
#ifdef P4_TO_P8
          case -3: return ABS3(x-xc.val,y-yc.val,z-zc.val) < .5*(container_radius_inner.val+container_radius_outer.val) ?
                  -thermal_cond_s.val*analytic::dts_exact(t, container_radius_inner.val) :
                  +thermal_cond_l.val*analytic::dtl_exact(t, container_radius_outer.val);
#endif
          case -2: return ABS2(x-xc.val,y-yc.val) < .5*(container_radius_inner.val+container_radius_outer.val) ?
                  -thermal_cond_s.val*analytic::dts_exact(t, container_radius_inner.val) :
                  +thermal_cond_l.val*analytic::dtl_exact(t, container_radius_outer.val);
          case -1: return y < .5*(ymin.val+ymax.val) ? -analytic::dts_exact(t, 0) : analytic::dtl_exact(t, ymax.val-ymin.val);
          case  0: return y > .5*(ymin.val+ymax.val) ? temp_gradient.val : -(temp_gradient.val*thermal_cond_l.val/thermal_cond_s.val + cooling_velocity.val*(latent_heat.val+density_l.val*temp_gradient.val*heat_capacity_l.val*(ymax.val-ymin.val))/thermal_cond_s.val * smooth_start(t));
          case  1: return -seed_radius.val/container_radius_outer.val*cooling_velocity.val*latent_heat.val*smooth_start(t);
          case  2: return -(container_radius_outer.val-front_location.val)/container_radius_outer.val*cooling_velocity.val*latent_heat.val*smooth_start(t);
          case  3:
            if (ABS2(x-xc.val,y-yc.val) > .5*(container_radius_inner.val+container_radius_outer.val)) {
              return -thermal_cond_s.val*temp_gradient.val
                  -(container_radius_outer.val-front_location.val)/container_radius_outer.val
                  *cooling_velocity.val*latent_heat.val*smooth_start(t);
            } else {
              return  thermal_cond_s.val*temp_gradient.val*container_radius_outer.val/container_radius_inner.val;
            }
          case  4:
            if (ABS2(x-xc.val,y-yc.val) > .5*(container_radius_inner.val+container_radius_outer.val)) {
              return  thermal_cond_l.val*temp_gradient.val
                  -(container_radius_inner.val+front_location.val)/container_radius_outer.val
                  *cooling_velocity.val*latent_heat.val*smooth_start(t);
            } else {
              return -thermal_cond_l.val*temp_gradient.val*container_radius_outer.val/container_radius_inner.val;
            }
          case  5: return 0;
          case  6: return y > .5*(ymin.val+ymax.val) ? temp_gradient.val : -(temp_gradient.val*thermal_cond_l.val/thermal_cond_s.val + cooling_velocity.val*(latent_heat.val+density_l.val*temp_gradient.val*heat_capacity_l.val*(ymax.val-ymin.val))/thermal_cond_s.val * smooth_start(t));
          case  7: return y > .5*(ymin.val+ymax.val) ? temp_gradient.val : -(temp_gradient.val*thermal_cond_l.val/thermal_cond_s.val + cooling_velocity.val*(latent_heat.val+density_l.val*temp_gradient.val*heat_capacity_l.val*(ymax.val-ymin.val))/thermal_cond_s.val * smooth_start(t));
          default: throw;
        }
      default: throw;
    }
  }
} /*bc_value_temp*/;

// we only need to provide tl for convergence bc we only specify tl at the walls, never ts
bc_value_temp_t bc_value_temp(convergence_temp);

class bc_value_conc_t : public CF_DIM
{
  int idx;
  Convergence_soln::concentration* concentration_;
public:
    bc_value_conc_t(int idx, Convergence_soln::concentration* concentration=NULL) :
        idx(idx), concentration_(concentration) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    if (geometry.val==5){
      if((fabs(y-ymax.val) < EPS)){
        return *initial_conc_all[idx]; // for DIRICHLET case
      }
      else{
        return 0.0; // for NEUMANN case
      }
    }
    switch (bc_type_conc.val) {
      case DIRICHLET:
        switch (geometry.val) {
#ifdef P4_TO_P8
          case -3: return analytic::cl_exact(idx, t, container_radius_outer.val);
#endif
          case -2: return analytic::cl_exact(idx, t, container_radius_outer.val);
          case -1: return analytic::cl_exact(idx, t, ABS1(ymax.val-ymin.val));
          case  0:
          case  1:
          case  2:
          case  3:
          case  4:
          case  5:
          case  6:
          case  7: return *initial_conc_all[idx];
          case 8: {
//            printf("accesses wall value concentration \n");
            return (*concentration_)(DIM(x,y,z));
          }
          default: throw std::invalid_argument("bc value conc (neumann): geometry case not recognized \n");
        }
      case NEUMANN:
        switch (geometry.val) {
#ifdef P4_TO_P8
          case -3: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, container_radius_outer.val);
#endif
          case -2: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, container_radius_outer.val);
          case -1: return (*solute_diff_all[idx])*analytic::dcl_exact(idx, t, ymax.val-ymin.val);
          case  0:
          case  1:
          case  2:
          case  3:
          case  4:
          case  5:
          case  6:
          case  7:
          case  8:
            return 0;
          default: throw std::invalid_argument("bc value conc (neumann): geometry case not recognized \n");
        }
      default: throw;
    }
  }
};

bc_value_conc_t bc_value_conc_0(0, &convergence_conc0);
bc_value_conc_t bc_value_conc_1(1, &convergence_conc1);
bc_value_conc_t bc_value_conc_2(2, &convergence_conc2);
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
    if (geometry.val == 5) return -cooling_velocity.val*latent_heat.val*2.*PI*seed_radius.val/(xmax.val-xmin.val)/(ymax.val-ymin.val)/box_size.val*smooth_start(t);
    else return 0;
  }
} volumetric_heat_cf;


// ----------------------------------------
// define initial fields and boundary conditions for NS
// ----------------------------------------

// Velocity
// Elyce TO-DO: couple the fluid velocity interface condition with vgamma and make sure that's working w stefan correctly
class BC_INTERFACE_VALUE_VELOCITY: public my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t{

  Convergence_soln::velocity_component* velocity_field_;
  public:
    unsigned char dir;
    BC_INTERFACE_VALUE_VELOCITY(my_p4est_stefan_with_fluids_t* parent_solver,
                                bool do_we_use_vgamma_for_bc=false,
                                Convergence_soln::velocity_component* velocity_field=NULL) :
        interfacial_bc_fluid_velocity_t(parent_solver, do_we_use_vgamma_for_bc), velocity_field_(velocity_field){}

    double operator()(double x, double y) const
    {
      if(geometry.val == 8){
        return (*velocity_field_)(DIM(x,y,z));

      }
      else{
         return Conservation_of_Mass(x, y);
//          return 0.0;
//        return 0.0;
      }

    }
};

// Type:
BoundaryConditionType interface_bc_type_velocity[P4EST_DIM];
void BC_INTERFACE_TYPE_VELOCITY(const unsigned char& dir){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
    interface_bc_type_velocity[dir] = DIRICHLET;
}

// --------------------------------------------------------------------------------------------------------------
// Auxiliary fxns for evaluating BCs
// --------------------------------------------------------------------------------------------------------------
// Wall functions -- these evaluate to true or false depending on
// if the location is on the wall --  they just add coding simplicity for wall boundary conditions
// --------------------------
bool xlower_wall(DIM(double x, double y, double z)){
  // Front x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmin.val) <= EPS) && (fabs(y - ymin.val)>EPS) && (fabs(y - ymax.val)>EPS));
};
bool xupper_wall(DIM(double x, double y, double z)){
  // back x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmax.val) <= EPS) && (fabs(y - ymin.val)>EPS) && (fabs(y - ymax.val)>EPS));
};
bool ylower_wall(DIM(double x, double y, double z)){
  return (fabs(y - ymin.val) <= EPS);
}
bool yupper_wall(DIM(double x, double y, double z)){
//  printf("yupper? ymax is %0.2f, y is %0.2f \n", ymax.val, y);
  return (fabs(y - ymax.val) <= EPS);
};

bool is_x_wall(DIM(double x, double y, double z)){
  return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
};
bool is_y_wall(DIM(double x, double y, double z)){
  return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
};
// For velocity BCs/ICs
//double u0=0.;

//double v0=1; // cm/s // -1.0e-4;

param_t<double> u0   (pl, 0., "u0", "Fluid velocity value in x direction for dirichlet boundary conditions (and initial conditions) TO-DO: ADD UNITS");
param_t<double> v0   (pl, 0., "v0", "Fluid velocity value in y direction for dirichlet boundary conditions (and initial conditions) TO-DO: ADD UNITS" );




double outflow_u=0.;
double outflow_v=0.;
// --------------------------------------------------------------------------------------------------------------
// Initial fluid velocity condition objects/functions: for fluid velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
struct INITIAL_VELOCITY : CF_DIM
{
  const unsigned char dir;
  Convergence_soln::velocity_component* velocity_;

  INITIAL_VELOCITY(const unsigned char& dir_, Convergence_soln::velocity_component* velocity=NULL): dir(dir_), velocity_(velocity)  {
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double operator() (DIM(double x, double y,double z)) const{
    if(geometry.val == 8){
      return (*velocity_)(DIM(x,y,z));
    }
    else{
      switch(dir){
      case dir::x:
        return u0.val;
      case dir::y:
        return v0.val;
      default:
        throw std::runtime_error("initial velocity direction unrecognized \n");
      }
    }
  }
};
//----------------
// Velocity wall:
//----------------
// Value:

bool dirichlet_velocity_walls(DIM(double x, double y, double z)){
    // Dirichlet on y upper wall (where bulk flow is incoming
    if(geometry.val == 8){
      return true;
    }
    else if (geometry.val == 0){
      // Directional solidification
      return yupper_wall(DIM(x,y,z));
    }
    else{
      return yupper_wall(DIM(x,y,z));//( ylower_wall(DIM(x,y,z)) /*|| xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z))*/);
    }
};

class BC_WALL_VALUE_VELOCITY: public CF_DIM
{
  public:
  const unsigned char dir;
  Convergence_soln::velocity_component* velocity_;

  BC_WALL_VALUE_VELOCITY(const unsigned char& dir_, Convergence_soln::velocity_component* velocity=NULL): dir(dir_), velocity_(velocity){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    if(geometry.val == 8){
      return (*velocity_)(DIM(x,y,z));
    } // end of geometry 8 case
    else{
      if(dirichlet_velocity_walls(DIM(x,y,z))){
        switch(dir){
        case dir::x:{
          return u0.val;
        }
        case dir::y:{
          return v0.val;
        }
        default:
          throw std::runtime_error("unrecognized cartesian direction for bc wall value velocity \n");
        }
      }
      else{
        return 0.0; // homogeneous neumann
      }
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
//    printf("accesses bc pressure value interface \n ");
    return 0.0;
  }
};

// Type:
static BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
    interface_bc_type_pressure = NEUMANN;
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
      return 0.0; // homogeneous dirichlet or neumann
  }
};

// Type:
class BC_WALL_TYPE_PRESSURE: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
//    printf("accesses wall pressure type at (%0.2f, %0.2f) \n");
    if(!dirichlet_velocity_walls(DIM(x,y,z))){ // pressure is dirichlet where vel is neumann
      return DIRICHLET;
    }
    else{
      return NEUMANN;
    }
  }
};


void initialize_all_relevant_ics_and_bcs_for_fluids(my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver,BC_INTERFACE_VALUE_VELOCITY* bc_interface_value_velocity[P4EST_DIM],
                                                    BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                                    BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM],
                                                    BC_INTERFACE_VALUE_PRESSURE& bc_interface_value_pressure,
                                                    BC_WALL_VALUE_PRESSURE& bc_wall_value_pressure,
                                                    BC_WALL_TYPE_PRESSURE& bc_wall_type_pressure,
                                                    my_p4est_multialloy_t* mas,
                                                    INITIAL_VELOCITY* v_init_cf[P4EST_DIM])
{

  for(unsigned char d=0;d<P4EST_DIM;d++){
        // Set the BC types:
        bool use_vgamma_for_vns_interface_bc = (geometry.val != 8); // Only geometry 8 doesn't use this, bc it's the convergence test
        bc_interface_value_velocity[d] = new BC_INTERFACE_VALUE_VELOCITY(stefan_w_fluids_solver, use_vgamma_for_vns_interface_bc, &convergence_vel[d]);
        BC_INTERFACE_TYPE_VELOCITY(d);

        bc_wall_type_velocity[d] = new BC_WALL_TYPE_VELOCITY(d);
        bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d, &convergence_vel[d]);

        v_init_cf[d]= new INITIAL_VELOCITY(d, &convergence_vel[d]);
    }

    interface_bc_pressure();
    CF_DIM* initial_velocity[P4EST_DIM] = {DIM(v_init_cf[0], v_init_cf[1], v_init_cf[2])};
    mas->set_initial_NS_velocity_n_(initial_velocity);
    mas->set_initial_NS_velocity_nm1_(initial_velocity);

    mas->set_bc_interface_value_pressure(&bc_interface_value_pressure);
    mas->set_bc_interface_type_pressure(interface_bc_type_pressure);
    my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t* bc_interface_value_velocity_[P4EST_DIM];
    BoundaryConditionType* bc_interface_type_velocity_[P4EST_DIM];

    // Velocity wall
    CF_DIM* bc_wall_value_velocity_[P4EST_DIM];
    WallBCDIM* bc_wall_type_velocity_[P4EST_DIM];
    foreach_dimension(d){
        // Vel interface
        bc_interface_type_velocity_[d] = &interface_bc_type_velocity[d];
        bc_interface_value_velocity_[d] = bc_interface_value_velocity[d];

        // Vel wall
        bc_wall_value_velocity_[d] = bc_wall_value_velocity[d];
        bc_wall_type_velocity_[d] = bc_wall_type_velocity[d];

    }
    mas->set_bc_wall_type_velocity(bc_wall_type_velocity_);
    mas->set_bc_wall_value_velocity(bc_wall_value_velocity_);

    mas->set_bc_interface_type_velocity(bc_interface_type_velocity_);
    mas->set_bc_interface_value_velocity(bc_interface_value_velocity_);
    mas->set_bc_wall_value_pressure(&bc_wall_value_pressure);
    mas->set_bc_wall_type_pressure(&bc_wall_type_pressure);
}
void remove_all_relevant_ics_and_bcs_for_fluids(BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                                    BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM],
                                                    BC_INTERFACE_VALUE_VELOCITY* bc_interface_value_velocity[P4EST_DIM],
                                                    BC_INTERFACE_VALUE_PRESSURE& bc_interface_value_pressure,
                                                    BC_WALL_VALUE_PRESSURE& bc_wall_value_pressure,
                                                    BC_WALL_TYPE_PRESSURE& bc_wall_type_pressure,
                                                    INITIAL_VELOCITY* v_init_cf[P4EST_DIM])
{
  for(unsigned char d=0;d<P4EST_DIM;d++){
        delete bc_wall_value_velocity[d];
        //delete bc_wall_type_velocity[d];
        delete v_init_cf[d];
        delete bc_interface_value_velocity[d];
    }
}
int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  mpi_environment_t mpi;
  mpi.init(argc, argv);

//  printf("convergence conc 2 --> comp = %d \n", convergence_conc2.comp);
//  printf("ext source c2 conc comp = %d \n", external_source_c2_robin.concentration_->comp);
//  external_source_c2_robin(5, 5);
//  // checking whether mpi version causes random memory leaks
//  int nnn = 0;
//  PetscLogDouble mem_petsc_old = 0;

//  while (1) {
//    Vec test;
//    int nghosts = 10000;
//    int nloc = 1024*1024;
//    vector<PetscInt> ghost_nodes(nghosts, 123);
//    ierr = VecCreateGhost(mpi.comm(), nloc, PETSC_DECIDE,
//                          ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], &test); CHKERRXX(ierr);

//    ierr = VecSetFromOptions(test); CHKERRXX(ierr);
//    VecDestroy(test);

//    if (mpi.rank() == 0) {
//      PetscLogDouble mem_petsc = 0;
//      PetscMemoryGetCurrentUsage(&mem_petsc);
//      if (mem_petsc != mem_petsc_old) {
//        std::cout << nnn << " " << mem_petsc/1024./1024. << "\n";
//      }
//      mem_petsc_old = mem_petsc;
//      nnn++;
//    }
//  }

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  alloy.set_from_cmd(cmd);
  set_alloy_parameters();
  pl.set_from_cmd_all(cmd);

  if (mpi.rank() == 0) pl.print_all();

  // prepare stuff for output
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
  else
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir;
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR directory");
  }

  FILE *fich;
  char filename_characteristics[1024];
  char filename_timings[1024];
  char filename_analytic[1024];
  char filename_error_max[1024];
  char filename_error_avg[1024];
  sprintf(filename_characteristics, "%s/characteristics.dat", out_dir);
  sprintf(filename_analytic, "%s/analytic.dat", out_dir);
  sprintf(filename_timings, "%s/timings.dat", out_dir);
  sprintf(filename_error_max, "%s/error_max.dat", out_dir);
  sprintf(filename_error_avg, "%s/error_avg.dat", out_dir);

  if (save_step_convergence.val) {
    ierr = PetscFOpen(mpi.comm(), filename_error_max, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
    ierr = PetscFOpen(mpi.comm(), filename_error_avg, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  if (save_characteristics.val) {
    ierr = PetscFOpen(mpi.comm(), filename_characteristics, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "step "
                                          "time "
                                          "growth "
                                          "average_interface_velocity "
                                          "max_interface_velocity "
                                          "front_area "
                                          "solid_volume "
                                          "time_elapsed "
                                          "sub_iterations "
                                          "bc_error_max "
                                          "bc_error_avg "
                                          "memory_usage\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }


  if (save_accuracy.val) {
    ierr = PetscFOpen(mpi.comm(), filename_analytic, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "step "
                                          "time "
                                          "phi "
                                          "ts "
                                          "tl "
                                          "vn "
                                          "ft "
                                          "tf "
                                          "vf "); CHKERRXX(ierr);

    for (int i = 0; i < num_comps.val; ++i) {
      ierr = PetscFPrintf(mpi.comm(), fich, "cl%d ", i); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "cs%d ", i); CHKERRXX(ierr);
    }
    ierr = PetscFPrintf(mpi.comm(), fich, "\n"); CHKERRXX(ierr);

    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  // Initialize the separate file to save if we are doing a coupled convergence test:
  FILE *fich_errors = NULL;
  char name_errors[1000];
  if(geometry.val == 8){
    sprintf(name_errors, "%s/coupled_convergence_test_lmin%d_lmax%d.dat", out_dir, lmin.val, lmax.val);
    ierr = PetscFOpen(mpi.comm(), name_errors, "w", &fich_errors); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_errors,"tn " "dt " "iteration "
                                                 "Tl_err " "Ts_err " "C0_err " "C1_err " " C2_err "
                                                 "vgamma_err " "vx_err " "vy_err " "num_nodes " "dxyz_min \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_errors); CHKERRXX(ierr);

  }

  if (mpi.rank() == 0 && save_params.val) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  if (save_vtk.val || save_vtk_solid.val || save_vtk_analytical.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  if (save_p4est.val || save_p4est_solid.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/p4est";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/p4est directory");
  }

  if (save_dendrites.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/dendrites";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/dendrites directory");
  }

  // rescalling to 1x1x1 box
  density_l.val      /= (scaling()*scaling()*scaling());
  density_s.val      /= (scaling()*scaling()*scaling());
  thermal_cond_l.val /= scaling();
  thermal_cond_s.val /= scaling();
  latent_heat.val    /= (scaling()*scaling()*scaling());
  eps_c.val          *= scaling();
  eps_v.val          /= scaling();

  solute_diff_0.val *= (scaling()*scaling());
  solute_diff_1.val *= (scaling()*scaling());
  solute_diff_2.val *= (scaling()*scaling());
  solute_diff_3.val *= (scaling()*scaling());

  //  volumetric_heat.val  /= (scaling()*scaling()*scaling());
  temp_gradient.val    /= scaling();
  cooling_velocity.val *= scaling();

  if(solve_w_fluids.val){
      mu_l.val /=scaling();
      gravity_.val *= scaling() * scaling();
  }


  // initialize constants in initial and boundary conditions
  for (int i = 0; i < num_comps.val; ++i) {
    analytic::Cstar[i] = (*initial_conc_all[i]);
  }

  for (int i = 0; i < num_comps.val; ++i) {
    analytic::kp[i] = part_coeff(i, analytic::Cstar);
  }

  if (start_from_moving_front.val && geometry.val == 0) {
    if (const_part_coeff.val) {
      for (int i = 0; i < num_comps.val; ++i) { analytic::Cstar[i] = (*initial_conc_all[i])/analytic::kp[i]; }
    } else {
      for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < num_comps.val; ++i) { analytic::Cstar[i] = (*initial_conc_all[i])/analytic::kp[i]; }
        for (int i = 0; i < num_comps.val; ++i) { analytic::kp[i]    = part_coeff(i, analytic::Cstar);  }
      }
    }
  }

  analytic::Tstar = liquidus_value(analytic::Cstar);

  switch (geometry.val) {
    case -2: analytic::set_analytical_solution(gradient_ratio.val, cooling_velocity.val, container_radius_inner.val + front_location.val); break;
    case -1: analytic::set_analytical_solution(gradient_ratio.val, cooling_velocity.val, front_location.val); break;
    default: break;
  }

  if (mpi.rank() == 0 && 1)
  {
    ierr = PetscPrintf(mpi.comm(), "density_l: %g\n", density_l.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "density_s: %g\n", density_s.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "thermal_cond_l: %g\n", thermal_cond_l.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "thermal_cond_s: %g\n", thermal_cond_s.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "thermal_diff_l: %g\n", thermal_cond_l.val/density_l.val/heat_capacity_l.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "thermal_diff_s: %g\n", thermal_cond_s.val/density_s.val/heat_capacity_s.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "latent_heat: %g\n", latent_heat.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "solute_diff_0: %g\n", solute_diff_0.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "solute_diff_1: %g\n", solute_diff_1.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "solute_diff_2: %g\n", solute_diff_2.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "solute_diff_3: %g\n", solute_diff_3.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "temp_gradient: %g\n", temp_gradient.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "cooling_velocity: %g\n", cooling_velocity.val); CHKERRXX(ierr);

    ierr = PetscPrintf(mpi.comm(), "related to fluid flow problem : \n \n", cooling_velocity.val); CHKERRXX(ierr);

    ierr = PetscPrintf(mpi.comm(), "viscosity_l: %g\n", mu_l.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "beta_T (bouss.): %g\n", beta_T.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "beta_C_0: %g\n", beta_C_0.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "beta_C_1: %g\n", beta_C_1.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "beta_C_2: %g\n\n", beta_C_2.val); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "beta_C_3: %g\n\n", beta_C_3.val); CHKERRXX(ierr);

    ierr = PetscPrintf(mpi.comm(), "characteristic length scale: %g\n", l_char.val); CHKERRXX(ierr);


    // Physical parameters to add:
    // mu_l
    // beta_T, beta_C (boussinesq)
  }

  parStopWatch w1;
  w1.start("total time");

  int initial_division = pow(2, sub_split_lvl.val) + sub_split_num.val;
  int lvl_decrease = floor(log(double(initial_division))/log(2.));

  while (true) {
    if (initial_division % 2 == 0) {
      initial_division = initial_division / 2;
      --lvl_decrease;
    } else {
      break;
    }
  }
  //std::cout<<"Hello World 1 main \n";
  // domain size information
  int n_xyz[]      = { DIM(nx.val*initial_division, ny.val*initial_division, nz.val*initial_division) };
  double xyz_min[] = { DIM(xmin.val, ymin.val, zmin.val) };
  double xyz_max[] = { DIM(xmax.val, ymax.val, zmax.val)};
  int periodic[]   = { DIM(periodicity(0), periodicity(1), periodicity(2))};

  int lmin_new = lmin.val-lvl_decrease;
  int lmax_new = lmax.val-lvl_decrease;

  // Boundary condition for the fluid problem :: Rochi
  PetscPrintf(mpi.comm(), "\nWarning: The bc interface velocity for Navier-Stokes is not currently coupled with vgamma. We will want to fix this before running real physical applications \n");
  BC_INTERFACE_VALUE_VELOCITY* bc_interface_value_velocity[P4EST_DIM];
  BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM];
  BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM];
  BC_INTERFACE_VALUE_PRESSURE bc_interface_value_pressure;
  BC_WALL_VALUE_PRESSURE bc_wall_value_pressure;
  BC_WALL_TYPE_PRESSURE bc_wall_type_pressure;
  //std::cout<<"Hello World 2 main \n";
  INITIAL_VELOCITY* v_init_cf[P4EST_DIM];
//  double dx_eff = (xmax.val-xmin.val)/double(n_xyz[0])/pow(2., lmax_new);
//  double lmax_eff = lmax_new + log(initial_division)/log(2.);
//  double lmin_eff = lmin_new + log(initial_division)/log(2.);FF


//  std::vector<CF_DIM*> convergence_forces_conc;
//  std::vector<CF_DIM*> external_source_conc_robin;

//  for (int i=0; i< num_comps.val; i++){
//    convergence_forces_conc[i] = convergence_forces_conc_[i];
//    external_source_conc_robin[i] = external_source_conc_robin_[i];

//  }

  /* initialize the solver */
  my_p4est_multialloy_t mas(num_comps.val, order_in_time.val);

  // Set the mpi environment: (elyce addition, needed for fluids, but also needed for save/load)
  mas.set_mpi_env(&mpi);

  // Set if we are loading from a previous state
  mas.set_load_from_previous_state(loading_from_previous_state.val);

  mas.initialize(mpi.comm(), xyz_min, xyz_max, n_xyz, periodic, phi_eff_cf, lmin_new, lmax_new, lip.val, band.val, solve_w_fluids.val);

  // Convergence test (for geometry
  if(geometry.val == 8){
    mas.set_there_is_convergence_test(true);

    // Set PDE sources:
    mas.set_convergence_source_NS(convergence_force_NS);
    mas.set_convergence_source_conc(convergence_forces_conc);
    mas.set_convergence_source_temp(convergence_forces_temp);

    // Set interface BC sources:
    mas.set_convergence_source_temp_jump(&external_source_temperature_jump);
    mas.set_convergence_source_temp_flux_jump(&external_source_temperature_flux_jump);
    mas.set_convergence_source_conc_robin(external_source_conc_robin);
    mas.set_convergence_source_Gibbs_Thomson(&external_source_Gibbs_Thomson);

  }
  ierr = PetscPrintf(mpi.comm(), "initialize complete \n"); CHKERRXX(ierr);

  my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver;
  vector<double> RaC_vals(num_comps.val, 0);
  vector<double> Sc_vals(num_comps.val, 0);
  vector<double> Cinf_vals(num_comps.val, 0);
  vector<double> betaC_vals(num_comps.val, 0);
  if(solve_w_fluids.val){
    // Calculate/set the Rayleigh numbers and relevant related groups
    calculate_Ra_numbers();

    stefan_w_fluids_solver = new my_p4est_stefan_with_fluids_t(&mpi);
//    mas.set_mpi_env(&mpi);
    mas.set_solve_with_fluids();

    // box size is the physical length scale that corresponds to a size of 1 in the computational domain,
    // so we choose it for l_characteristic for the dimensionless fluid problem

    if(convert_dim_to_nondim_for_fluids.val){
      mas.set_lchar(box_size.val);
    }
    else{
      mas.set_lchar(1.);
    }

    mas.set_gravity(gravity_.val);
    mas.set_mu_l(mu_l.val);
    mas.set_Pr(Pr.val);
    mas.set_RaT(Ra_T.val);
    mas.set_Tinf(Tinf.val);
    mas.set_betaT(beta_T.val);

    RaC_vals[0] = Ra_C_0.val;
    Sc_vals[0] = Sc0.val;
    Cinf_vals[0] = Cinf0.val;
    betaC_vals[0] = beta_C_0.val;


    if(num_comps.val>1) {
      RaC_vals[1] = Ra_C_1.val;
      Sc_vals[1] = Sc1.val;
      Cinf_vals[1] = Cinf1.val;
      betaC_vals[1] = beta_C_1.val;
    }

    if(num_comps.val>2) {
      RaC_vals[2] = Ra_C_2.val;
      Sc_vals[2] = Sc2.val;
      Cinf_vals[2] = Cinf2.val;
      betaC_vals[2] = beta_C_2.val;

    }
    if(num_comps.val>3) {
      RaC_vals[3] = Ra_C_3.val;
      Sc_vals[3] = Sc3.val;
      Cinf_vals[3] = Cinf3.val;
      betaC_vals[3] = beta_C_3.val;
    }

    mas.set_RaC(RaC_vals);
    mas.set_Sc(Sc_vals);
    mas.set_Cinf(Cinf_vals);
    mas.set_betaC(betaC_vals);

//    mas.set_RaC0(Ra_C_0.val);
//    mas.set_RaC0(Ra_C_1.val);
//    mas.set_RaC0(Ra_C_2.val);
    mas.set_do_boussinesq(do_boussinesq.val);
    mas.set_convert_dim_to_nondim_for_fluids_step(convert_dim_to_nondim_for_fluids.val);

    // Calculate nondimensional groups:
//    compute_nondimensional_groups(mpi.comm(), &mas);
  }

  p4est_t                   *p4est = mas.get_p4est();
  p4est_nodes_t             *nodes = mas.get_nodes();
  my_p4est_node_neighbors_t *ngbd  = mas.get_ngbd();

  /* initialize the variables */
//  vec_and_ptr_t front_phi(p4est, nodes);
//  vec_and_ptr_t contr_phi(front_phi.vec);
//  vec_and_ptr_t seed_map (front_phi.vec);
  vec_and_ptr_t front_phi, contr_phi, seed_map;
  if(!loading_from_previous_state.val){
    front_phi.create(p4est, nodes);
    contr_phi.create(p4est, nodes);
    seed_map.create(p4est, nodes);

    sample_cf_on_nodes(p4est, nodes, front_phi_cf, front_phi.vec);
    sample_cf_on_nodes(p4est, nodes, contr_phi_cf, contr_phi.vec);
    sample_cf_on_nodes(p4est, nodes, seed_number_cf, seed_map.vec);
  }
  PetscPrintf(mpi.comm(), "ELYCE, WARNING: do you need to save contr phi and seed number to the state so it can be loaded? \n");

  /* set initial time step */
  double dxyz[P4EST_DIM];
  dxyz_min(p4est, dxyz);

  double DIM(dx = dxyz[0],
             dy = dxyz[1],
             dz = dxyz[2]);

  PetscPrintf(mpi.comm(), "Conc dt: %1.3e, Temp dt: %1.3e, Velo dt: %1.3e\n",
              .5*dx*dx/solute_diff_0.val,
              .5*dx*dx*density_l.val*heat_capacity_l.val/thermal_cond_l.val,
              cfl_number.val*dx/cooling_velocity.val);

  // perturb level set
  if (enforce_planar_front.val) init_perturb.val = 0;

  if(!loading_from_previous_state.val){
    front_phi.get_array();

    srand(mpi.rank());

    foreach_node(n, nodes)
    {
      front_phi.ptr[n] += init_perturb.val*dx*double(rand())/double(RAND_MAX);
    }

    front_phi.restore_array();
    ierr = VecGhostUpdateBegin(front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  my_p4est_level_set_t ls(ngbd);
//    ls.reinitialize_2nd_order(front_phi.vec);
//    ls.perturb_level_set_function(front_phi.vec, EPS);

  // set alloy parameters
  double solute_diff_all[] = { solute_diff_0.val, solute_diff_1.val, solute_diff_2.val, solute_diff_3.val };

  mas.set_liquidus(liquidus_value, liquidus_slope, part_coeff);
  mas.set_composition_parameters(solute_diff_all);
  mas.set_thermal_parameters(latent_heat.val,
                             density_l.val, heat_capacity_l.val, thermal_cond_l.val,
                             density_s.val, heat_capacity_s.val, thermal_cond_s.val);


  std::vector<CF_DIM *> eps_c_all(num_seeds(), NULL);
  std::vector<CF_DIM *> eps_v_all(num_seeds(), NULL);

  for (int i = 0; i < num_seeds(); ++i)
  {
    eps_c_all[i] = new eps_c_cf_t(theta0(i));
    eps_v_all[i] = new eps_v_cf_t(theta0(i));
  }
  mas.set_undercoolings(num_seeds(), loading_from_previous_state.val? NULL:seed_map.vec, eps_v_all.data(), eps_c_all.data());

  // set geometry
  if(!loading_from_previous_state.val){
    mas.set_front(front_phi.vec);
    mas.set_container(contr_phi.vec);
  }
  mas.set_scaling(scaling());

  // set boundary conditions
  mas.set_container_conditions_thermal(bc_type_temp.val, bc_value_temp);
  mas.set_container_conditions_composition(bc_type_conc.val, bc_value_conc_all);

//  CF_DIM* bc_wall_type_temp = bc_wall_type_temp_;
  WallBCDIM* bc_wall_type_temp = &bc_wall_type_temp_;
  WallBCDIM* bc_wall_type_conc = &bc_wall_type_conc_;

  mas.set_wall_conditions_thermal(bc_wall_type_temp/*bc_type_temp.val*/, bc_value_temp);
  mas.set_wall_conditions_composition(bc_wall_type_conc, bc_value_conc_all);
//  printf("MAIN: bc_value_conc_all = %p \n", bc_value_conc_all[0]);
  mas.set_volumetric_heat(volumetric_heat_cf);

  // Initialize everything for the fluids
  // NOTE: the below needs to be called in such a way that the correct front phi is provided to SWF, because initializing the NS solver will require that. Therefore, it cannot be called until *after* mas.set_front
  if(solve_w_fluids.val){
    initialize_all_relevant_ics_and_bcs_for_fluids(stefan_w_fluids_solver,
                                                   bc_interface_value_velocity,
                                                   bc_wall_value_velocity,
                                                   bc_wall_type_velocity,
                                                   bc_interface_value_pressure,
                                                   bc_wall_value_pressure,
                                                   bc_wall_type_pressure,
                                                   &mas,v_init_cf);
    mas.initialize_for_fluids(stefan_w_fluids_solver);
    ierr = PetscPrintf(mpi.comm(), "\nInitialize for fluids complete \n"); CHKERRXX(ierr);

  }



  // set time steps
  double dt = cfl_number.val*MIN(DIM(dx,dy,dz))/cooling_velocity.val;

  double dt_max = base_cfl.val*MIN(DIM(dx,dy,dz))/cooling_velocity.val;

//  dt = 1.0e-3;

  double dt_curv = 0.000005*sqrt(dx*dx*dx)/MAX(eps_c.val, 1.e-20);

  //mas.set_dt(MIN(dt, dt_curv));
  //mas.set_dt_limits(0, MIN(dt_curv, 100.*dt));
  mas.set_dt_limits(0,dt_max);


  // TO-DO: revisit if we really want these dt limits or not

  // set initial conditions
  if(!loading_from_previous_state.val){
    mas.set_velocity(vn_cf, DIM(vx_cf, vy_cf, vz_cf), vf_cf);
    mas.set_temperature(tl_cf, ts_cf, tf_cf);
    //  mas.set_temperature_solve_w_fluids(tl_cf, ts_cf, tf_cf);

    mas.set_concentration(cl_cf_all, cs_cf_all);
    mas.set_ft(ft_cf);
  }
  PetscPrintf(mpi.comm(), "Elyce to-do: I think I might need to include the freeze time as a save/load field \n");

  // set solver parameters
  if(geometry.val == 8){
    double dxyz_min_[P4EST_DIM];
    dxyz_min(p4est, dxyz_min_);
    bc_tolerance.val = dxyz_min_[0] * bc_tolerance.val; // make the bc tolerance a percentage of the current grid level
    PetscPrintf(mpi.comm(), "\n Convergence test dxyz_min is set to %0.2e with dxyz_min_ = %0.2e \n", bc_tolerance.val, dxyz_min_[0]);
  }
  mas.set_bc_tolerance             (bc_tolerance.val);
  mas.set_max_iterations           (max_iterations.val);
  mas.set_cfl                      (cfl_number.val);
  mas.set_proximity_smoothing      (proximity_smoothing.val);

  mas.set_use_superconvergent_robin(use_superconvergent_robin.val);
  mas.set_use_points_on_interface  (use_points_on_interface.val);
  mas.set_update_c0_robin          (update_c0_robin.val);
  mas.set_enforce_planar_front     (enforce_planar_front.val);

  mas.set_dendrite_cut_off_fraction(dendrite_cut_off_fraction.val);
  mas.set_dendrite_min_length      (dendrite_min_length.val);

  // compute container volume
  PetscPrintf(mpi.comm(), "Elyce to-do: need to get the front phi and contr phi from mas in the laod state case. also need to actuall save contr phi to begin with which I have not done ... \n");
  if(loading_from_previous_state.val){
    front_phi.vec = mas.get_front_phi();
    contr_phi.vec = mas.get_contr_phi();
  }
  vec_and_ptr_t ones(front_phi.vec);
  VecSetGhost(ones.vec, 1.);
  double container_volume = integrate_over_negative_domain(p4est, nodes, contr_phi.vec, ones.vec);
  ones.destroy();

  // clear up memory
  if(!loading_from_previous_state.val){
    contr_phi.destroy();
    front_phi.destroy();
    seed_map.destroy();
  }

  // loop over time
  bool   keep_going     = true;
  /*double */tn             = (loading_from_previous_state.val ? mas.get_tn() : 0.);
  double total_growth   = 0;
  double base           = front_location.val;
  double bc_error_max   = 0;
  double bc_error_avg   = 0;
  int    iteration      = (loading_from_previous_state.val? mas.get_iteration_one_step(): 0);
  int    sub_iterations = 0;
  int    vtk_idx        = (loading_from_previous_state.val? mas.get_vtk_idx(): 0);
  PetscPrintf(mpi.comm(), "Starting: vtk_idx = %d \n", vtk_idx);
  int    save_state_idx = (loading_from_previous_state.val? floor(mas.get_iteration_one_step()/save_state_every_dn.val): 1); // start at 1 so we don't save the state of the initial condition
  int    mpiret;

  vector<double> bc_error_max_all;
  vector<double> bc_error_avg_all;
  vector<int> num_pdes;

  std::vector<bool>   logevent;
  std::vector<double> old_time;
  std::vector<int>    old_count;

  std::vector<double> errors_max_over_time(2*num_comps.val+7, 0);
  int itn =1;

  // save initial conditon:
//  PetscPrintf(mpi.comm(), "Saving the initial condition to vtk ... \n");
//  mas.save_VTK(-1);
  if(1){
    PetscPrintf(mpi.comm(), "Trying to save initial phi ... \n");
    // TEMPORARY -- troubleshooting
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};
    point_fields.push_back(Vec_for_vtk_export_t(/*front_phi.vec*/ mas.get_front_phi(), "phi"));
    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_initial_phi_%d", out_dir, -1);
    my_p4est_vtk_write_all_lists(p4est, nodes, ngbd->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();
    PetscPrintf(mpi.comm(), "Done! \n \n \n");

  }
  PetscPrintf(mpi.comm(), "Entering time loop ! \n");
  if(geometry.val == 8 && use_convergence_dt.val){
    mas.set_dt(convergence_dt.val);
    mas.set_dt_all(convergence_dt.val);
  }

  bool last_iter=false;
  if (geometry.val==8){
    check_convergence_errors_and_save_to_vtk(&mas, mpi, 10000);
  }

  // Do an initial update grid to kick things off in a nice direction:
  // Basically, we do a one step of update grid first in the no flow case to be compatible with
  // how Daniil originally organized the solver to work.
  // In the flow case, we handle things a little bit differently to be compatible
  // with all the flow stuff and how the time disc. is handled differently.
  if(!solve_w_fluids.val){
    mas.compute_dt();
    mas.save_VTK(-2);
    mas.update_grid();
    mas.update_grid_solid();
    mas.save_VTK(-1);
  }

  while (1)
  {
//    // CHeck to make sure the convergence fields are at least initialized correctly

//    PetscPrintf(mpi.comm(), "\n ------- \n Iteration: %d \n --------------- \n", iteration);
    // determine to save or not
    bool save_now =
        (save_type.val == 0 && iteration    >= vtk_idx*save_every_dn.val) ||
        (save_type.val == 1 && total_growth >= vtk_idx*save_every_dl.val) ||
        (save_type.val == 2 && tn           >= vtk_idx*save_every_dt.val);

    PetscPrintf(mpi.comm(), "save_now ? %d , vtk_idx = %d, tn = %0.2f, vtk_idx * save_every_dt = %0.2f \n", save_now, vtk_idx, tn, save_every_dt.val * vtk_idx);
    bool save_state_now = (save_state.val) && (iteration >= save_state_idx * save_state_every_dn.val);
    if(!last_iter){if (!keep_going) break;}

    // compute time step
    //mas.compute_dt();

    PetscPrintf(mpi.comm(), "\n---------------------------\n Value of tn: %0.5f \n---------------------------\n", tn);
    // for convergence study, update the time variable for each of the fields, and the normals for fields that require it:
    vec_and_ptr_dim_t front_normals_;
    my_p4est_node_neighbors_t* ngbd_;
    vec_and_ptr_t phi_;
    if(geometry.val == 8){
      PetscPrintf(mpi.comm(), "\n---------------------------\nConvergence test: %0.2f Percent Done \n---------------------------\n", (tn/time_limit.val) * 100.);

      double dt = mas.get_dt();

      // temps
      convergence_temp[LIQUID_DOMAIN].t = tn + dt;
      convergence_temp[SOLID_DOMAIN].t = tn + dt;

      // concs:
      convergence_conc0.t = tn + dt;
      if(num_comps.val>1) convergence_conc1.t = tn + dt;
      if(num_comps.val>2) convergence_conc2.t = tn + dt;

      // interface vel
      convergence_vgamma.t = tn + dt;

      // ns vels
      foreach_dimension(d){
        convergence_vel[d].t = tn + dt;
      }

      // Update the front normals as needed for the external bc source terms:
      front_normals_ = mas.get_front_normals();
      ngbd_ = mas.get_ngbd();

      external_source_c0_robin.set_inputs(ngbd_, front_normals_.vec[0], front_normals_.vec[1]);
      if(num_comps.val>1) external_source_c1_robin.set_inputs(ngbd_, front_normals_.vec[0], front_normals_.vec[1]);
      if(num_comps.val>2) external_source_c2_robin.set_inputs(ngbd_, front_normals_.vec[0], front_normals_.vec[1]);
      external_source_temperature_flux_jump.set_inputs(ngbd_, front_normals_.vec[0], front_normals_.vec[1]);

//      PetscPrintf(mpi.comm(), "CALLING EXT SOURCE C2 ROBIN \n");
//      external_source_c2_robin(5,5);

      // ANd update the temperature wall bc:
      phi_.vec = mas.get_front_phi();
      bc_value_temp.set_inputs(ngbd_, phi_.vec );

      mas.set_c0_guess(&convergence_conc0);
    }

    // solve nonlinear system for temperature, concentration and velocity at t_n
    bc_error_max = 0;
    bc_error_avg = 0;

    if (!solve_w_fluids.val){
    sub_iterations += mas.one_step(2, &bc_error_max, &bc_error_avg, &num_pdes, &bc_error_max_all, &bc_error_avg_all);
    }else{
    sub_iterations += mas.one_step_w_fluids(2, &bc_error_max, &bc_error_avg, &num_pdes, &bc_error_max_all, &bc_error_avg_all);
    }
    //tn             += mas.get_dt();
//    exit(0);
    if(geometry.val == 8 && keep_going){
      check_convergence_errors_and_save_to_vtk(&mas, mpi, iteration,
                                               fich_errors, name_errors);
    }

    if (save_step_convergence() && keep_going) {
      // max bc error
      ierr = PetscFOpen(mpi.comm(), filename_error_max, "a", &fich); CHKERRXX(ierr);
      for (size_t i = 0; i < bc_error_max_all.size(); ++i) {
        ierr = PetscFPrintf(mpi.comm(), fich, "%e ", bc_error_max_all[i]); CHKERRXX(ierr);
      }
      for (int i = bc_error_max_all.size(); i < max_iterations(); ++i) {
        ierr = PetscFPrintf(mpi.comm(), fich, "%e ", 0.0); CHKERRXX(ierr);
      }
      ierr = PetscFPrintf(mpi.comm(), fich, "\n"); CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);

      // avg bc error
      ierr = PetscFOpen(mpi.comm(), filename_error_avg, "a", &fich); CHKERRXX(ierr);
      for (size_t i = 0; i < bc_error_avg_all.size(); ++i) {
        ierr = PetscFPrintf(mpi.comm(), fich, "%e ", bc_error_avg_all[i]); CHKERRXX(ierr);
      }
      for (int i = bc_error_avg_all.size(); i < max_iterations(); ++i) {
        ierr = PetscFPrintf(mpi.comm(), fich, "%e ", 0.0); CHKERRXX(ierr);
      }
      ierr = PetscFPrintf(mpi.comm(), fich, "\n"); CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Step convergence saved in %s and %s\n", filename_error_max, filename_error_avg); CHKERRXX(ierr);
    }

    /*
    // compute total growth
    total_growth = base;

    p4est         = mas.get_p4est();
    nodes         = mas.get_nodes();
    front_phi.vec = mas.get_front_phi();

    front_phi.get_array();
    foreach_node(n, nodes) {
      if (front_phi.ptr[n] > 0) {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        total_growth = MAX(total_growth, xyz[1]);
      }
    }
    front_phi.restore_array();


    mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    total_growth -= base;
    */
//    ierr = PetscPrintf(mpi.comm(), "------------------------------------------------------------------------------------------------------\n"
//                                   "Time step %d: growth %e, simulation time %e, compute time %e\n"
//                                   "------------------------------------------------------------------------------------------------------\n",
//                       iteration, total_growth, tn, w1.get_duration_current()); CHKERRXX(ierr);

    if (save_timings.val && save_now)
    {
      PetscStageLog stageLog;
      PetscLogGetStageLog(&stageLog);

      PetscEventPerfInfo *eventInfo      = stageLog->stageInfo[0].eventLog->eventInfo;
      PetscInt            localNumEvents = stageLog->stageInfo[0].eventLog->numEvents;

      if (vtk_idx == 0)
      {
        ierr = PetscFOpen(mpi.comm(), filename_timings, "w", &fich); CHKERRXX(ierr);

        logevent.resize(localNumEvents, false);
        old_time.resize(localNumEvents, 0);
        old_count.resize(localNumEvents, 0);

        PetscFPrintf(mpi.comm(), fich, "iteration ");
        for (PetscInt event = 0; event < localNumEvents; ++event)
        {
          if (eventInfo[event].count > 0)
          {
            PetscFPrintf(mpi.comm(), fich, "%s count ", stageLog->eventLog->eventInfo[event].name);
            logevent[event] = true;
          }
        }
        PetscFPrintf(mpi.comm(), fich, "\n");

        ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      }

      ierr = PetscFOpen(mpi.comm(), filename_timings, "a", &fich); CHKERRXX(ierr);

      PetscFPrintf(mpi.comm(), fich, "%d ", iteration);
      for (PetscInt event = 0; event < localNumEvents; ++event)
      {
        if (logevent[event])
        {
          PetscFPrintf(mpi.comm(), fich, "%e %d ", eventInfo[event].time-old_time[event], eventInfo[event].count-old_count[event] );
          old_time[event]  = eventInfo[event].time;
          old_count[event] = eventInfo[event].count;
        }
      }
      PetscFPrintf(mpi.comm(), fich, "\n");

      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved timings in %s\n", filename_timings); CHKERRXX(ierr);
    }

    // Rochi adding save functionalities here
    // save velocity, area of interface, volume of solid phase, etc
    if (save_characteristics.val)
    {
      vec_and_ptr_t vn;

      // get current fields
      p4est         = mas.get_p4est();
      nodes         = mas.get_nodes();
      front_phi.vec = mas.get_front_phi();
      contr_phi.vec = mas.get_contr_phi();
      vn.vec        = mas.get_normal_velocity();

      // compute level-set of liquid region
      vec_and_ptr_t phi_liquid(front_phi.vec);
      VecPointwiseMaxGhost(phi_liquid.vec, contr_phi.vec, front_phi.vec);

      // compute characteristics (front area, phase volumes, average velocity, number of nodes, etc)
      ones.create(front_phi.vec);
      VecSetGhost(ones.vec, 1.);

      double liquid_volume   = integrate_over_negative_domain(p4est, nodes, phi_liquid.vec, ones.vec);
      double solid_volume    = container_volume - liquid_volume;
      double front_area      = integrate_over_interface(p4est, nodes, front_phi.vec, ones.vec);
      double velocity_avg    = integrate_over_interface(p4est, nodes, front_phi.vec, vn.vec) / front_area;
      double time_elapsed    = w1.get_duration_current();
      int    num_local_nodes = nodes->num_owned_indeps;
      int    num_ghost_nodes = nodes->indep_nodes.elem_count - num_local_nodes;

      int buffer[] = { num_local_nodes, num_ghost_nodes };
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &buffer, 2, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      num_local_nodes = buffer[0];
      num_ghost_nodes = buffer[1];

      PetscLogDouble mem_petsc = 0;
      if (track_memory_usage.val) {
        PetscMemoryGetCurrentUsage(&mem_petsc);
      }

      // write into file
      ierr = PetscFOpen(mpi.comm(), filename_characteristics, "a", &fich); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%d %e %e %e %e %e %e %e %d %e %e %e\n",
                          iteration,
                          tn,
                          total_growth,
                          velocity_avg/scaling(),
                          mas.get_front_velocity_max()/scaling(),
                          front_area,
                          solid_volume,
                          time_elapsed,
                          sub_iterations,
                          bc_error_max,
                          bc_error_avg,
                          mem_petsc/1024./1024.); CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Saved characteristics in %s\n", filename_characteristics); CHKERRXX(ierr);

      sub_iterations = 0;
      ones.destroy();
      phi_liquid.destroy();
    }
    // save field data
    if (save_now)
    {
      if (save_dendrites.val)   mas.count_dendrites(vtk_idx);
      if (save_vtk.val)         mas.save_VTK(vtk_idx);
      if (save_vtk_solid.val)   mas.save_VTK_solid(vtk_idx);
      ierr = PetscPrintf(mpi.comm(), "Saved characteristics in %s\n", filename_characteristics); CHKERRXX(ierr);
      if (save_p4est.val)       mas.save_p4est(vtk_idx);
    }
    if (save_now) vtk_idx++;

    // compute total growth - new position - Rochi - checking if growth is occuring now or not
    total_growth = base;

    p4est         = mas.get_p4est();
    nodes         = mas.get_nodes();
    front_phi.vec = mas.get_front_phi();

    front_phi.get_array();
    foreach_node(n, nodes) {
      if (front_phi.ptr[n] > 0) {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        total_growth = MAX(total_growth, xyz[1]);
      }
    }
    front_phi.restore_array();

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    total_growth -= base;

    //std::cout << "Total growth :: "<< total_growth << " ; Growth Limit :: " << growth_limit.val << "\n";
    // Rochi :: moving compute_dt to a different location
    // compute time step
    /*if(last_iter) {
        keep_going=false;
        last_iter=false;
    }
    mas.compute_dt();
    if (tn + mas.get_dt() > time_limit.val) {
      mas.set_dt(time_limit.val-tn);
      last_iter =true;
    }
    keep_going = keep_going && (iteration < step_limit.val) && (total_growth < growth_limit.val);

    if(keep_going){
      if(solve_w_fluids.val){
        mas.update_grid_w_fluids();
      }
      else{
        mas.update_grid();
      }

      mas.update_grid_solid();
    }*/

    // Clear out the boundary condition info now that we are done w this timestep
    if(geometry.val == 8){
      external_source_c0_robin.clear_inputs();
      if(num_comps.val>1) external_source_c1_robin.clear_inputs();
      if(num_comps.val>2) external_source_c2_robin.clear_inputs();

      external_source_temperature_flux_jump.clear_inputs();
      bc_value_temp.clear_inputs();
    }
//    if(last_iter) {
//        keep_going=false;
//        last_iter=false;
//        continue;
//    }

    mas.compute_dt();

    if(geometry.val == 8 && use_convergence_dt.val){
      mas.set_dt(convergence_dt.val);
    }
//    if (tn + mas.get_dt() > time_limit.val) {
//      mas.set_dt(time_limit.val-tn);
//      last_iter =true;
//    }

    if (tn + mas.get_dt() > time_limit.val) {
      mas.set_dt(time_limit.val-tn);
      keep_going = false;
    }

    if(geometry.val == 8 && use_convergence_dt.val){
      tn+= convergence_dt.val;
    }
    else {
      tn += mas.get_dt();
    }
    keep_going = keep_going && (iteration < step_limit.val) && (total_growth < growth_limit.val);

    if(keep_going){
      if(solve_w_fluids.val){
        mas.update_grid_w_fluids();
      }
      else{
        mas.update_grid();
      }

      mas.update_grid_solid();
    }



    if(save_state_now){
      PetscPrintf(mpi.comm(), "Beginning save state process ... \n");
      char output[1000];
      const char* out_dir_save_state = getenv("OUT_DIR_SAVE_STATE");
      if(!out_dir_save_state){
        throw std::invalid_argument("You need to set the output directory for save states: OUT_DIR_SAVE_STATE");
      }
      sprintf(output,
              "%s/save_states_multialloy_lmin_%d_lmax_%d_geom_%d",
              out_dir_save_state,
              lmin.val, lmax.val, geometry.val);

      mas.save_state(output, num_state_backups.val);

      save_state_idx++;
    }




    iteration++;

  }
  for (int i = 0; i < num_seeds(); ++i)
  {
      if(eps_c_all[i]) delete eps_c_all[i];
      if(eps_v_all[i]) delete eps_v_all[i];
  }
  eps_c_all.clear();
  eps_v_all.clear();
  w1.stop(); w1.read_duration();
  return 0;
}

