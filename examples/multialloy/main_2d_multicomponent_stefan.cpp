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
param_t<int>    max_iterations  (pl, 50,     "max_iterations",  "");
param_t<double> bc_tolerance    (pl, 1.e-5, "bc_tolerance",    "");
param_t<double> cfl_number      (pl, 0.1,   "cfl_number",      "");
//param_t<double> base_cfl        (pl, pow(1.0,1.5)*0.0811,   "base_cfl",      "");
param_t<double> base_cfl        (pl, 0.111/pow(4.,1.5),   "base_cfl",      "");

param_t<double> proximity_smoothing       (pl, 1.01, "proximity_smoothing",       "");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool>   save_characteristics (pl, 1, "save_characteristics", "");
param_t<bool>   save_dendrites       (pl, 0, "save_dendrites", "");
param_t<bool>   save_accuracy        (pl, 0, "save_accuracy", "");
param_t<bool>   save_timings         (pl, 0, "save_timings", "");
param_t<bool>   save_params          (pl, 1, "save_params", "");
param_t<bool>   save_vtk             (pl, 1, "save_vtk", "");
param_t<bool>   save_vtk_solid       (pl, 1, "save_vtk_solid", "");
param_t<bool>   save_vtk_analytical  (pl, 0, "save_vtk_analytical", "");
param_t<bool>   save_p4est           (pl, 1, "save_p4est", "");
param_t<bool>   save_p4est_solid     (pl, 1, "save_p4est_solid", "");
param_t<bool>   save_step_convergence(pl, 0, "save_step_convergence", "");

param_t<int>    save_every_dn (pl, 1000, "save_every_dn", "");
param_t<double> save_every_dl (pl, 0.025, "save_every_dl", "");
param_t<double> save_every_dt (pl, 0.1,  "save_every_dt",  "");

param_t<int>    save_type (pl, 0, "save_type", "0 - every n iterations (pl, 1 - every dl of growth (pl, 2 - every dt of time");

param_t<double> dendrite_cut_off_fraction (pl, 1.05, "dendrite_cut_off_fraction", "");
param_t<double> dendrite_min_length       (pl, 0.05, "dendrite_min_length", "");

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

param_t<double> eps_c    (pl, 0, "eps_c",    "Curvature undercooling coefficient - cm.K");
param_t<double> eps_v    (pl, 0, "eps_v",    "Kinetic undercooling coefficient - s.K.cm-1");
param_t<double> eps_a    (pl, 0, "eps_a",    "Anisotropy coefficient");
param_t<double> symmetry (pl, 4, "symmetry", "Symmetric of crystals");

// parameters for the fluid flow problem:
param_t<double> mu_l (pl, 1.0e-3, "mu_l", "viscosity of the fluid ");
param_t<double> beta_T(pl, 1.0, "beta_T", "coefficient of thermal expansion - boussinesq");
param_t<double> beta_C_0(pl, 1.0, "beta_C_0", "coefficient of concentration expansion for comp 0 -- boussinesq");
param_t<double> beta_C_1(pl, 1.0, "beta_C_1", "coefficient of concentration expansion for comp 1 -- boussinesq");
param_t<double> beta_C_2(pl, 1.0, "beta_C_2", "coefficient of concentration expansion for comp 2 -- boussinesq");
param_t<double> beta_C_3(pl, 1.0, "beta_C_3", "coefficient of concentration expansion for comp 3 -- boussinesq");

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
                                    "7: a made-up penta-alloy");

double scale = 30;

//-------------------------------------
// problem parameters
//-------------------------------------
//param_t<double> volumetric_heat (pl,  0, "", "Volumetric heat generation (pl, J/cm^3");
param_t<double> cooling_velocity        (pl, 0.001*scale,  "cooling_velocity", "Cooling velocity (pl, cm/s");
param_t<double> gradient_ratio          (pl, 0.75,  "gradient_ratio",   "Ratio of compositional and thermal gradients at the front");
param_t<double> temp_gradient           (pl, 500, "temp_gradient",    "Temperature gradient (pl, K/cm");
param_t<bool>   start_from_moving_front (pl, 0, "start_from_moving_front", "Relevant only for geometry==0");

param_t<int>    smoothstep_order (pl, 5,     "smoothstep_order", "Smoothness of cooling/heating ");
param_t<double> starting_time    (pl, 0.e-3, "starting_time",    "Time for cooling/heating to fully switch on (pl, s");

param_t<BoundaryConditionType> bc_type_conc (pl, NEUMANN, "bc_type_conc", "DIRICHLET/NEUMANN");
param_t<BoundaryConditionType> bc_type_temp (pl, NEUMANN, "bc_type_temp", "DIRICHLET/NEUMANN");
param_t<BoundaryConditionType> bc_wall_type_vel  (pl, NEUMANN, "bc_wall_type_vel", "DIRICHLET/NEUMANN");
//param_t<BoundaryConditionType> bc_type_temp (pl, DIRICHLET, "bc_type_temp", "DIRICHLET/NEUMANN");

param_t<int>    step_limit           (pl, INT_MAX, "step_limit",   "");
//param_t<int>    step_limit           (pl, 200, "step_limit",   "");
param_t<double> time_limit           (pl, DBL_MAX, "time_limit",   "");
param_t<double> growth_limit         (pl, 15, "growth_limit", "");
param_t<double> init_perturb         (pl, 1.e-10,  "init_perturb",         "");
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
                                              " 5 - three spherical seeds,"
                                              " 6 - planar front and three spherical seeds,"
                                              " 7 - daniil has not defined this case - no comments - need to see what it is,"
                                              " 8 - single dendrite growth <new addition>");



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
      latent_heat.val     = 2600;    // J.cm-3

      num_comps.val = 2;

      solute_diff_0.val    = 1e-5;     // cm2.s-1 - concentration diffusion coefficient
      solute_diff_1.val    = 2e-5;     // cm2.s-1 - concentration diffusion coefficient
      initial_conc_0.val   = 0.107;    // at frac.
      initial_conc_1.val   = 0.094;    // at frac.

      eps_c.val = 1.0e-5/1.;
      eps_v.val = 0.0e-2;
      eps_a.val = 0.05*1.;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1910;      // K
      liquidus_slope_0.val =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1.val =-1036;     // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.83;    // partition coefficient
      part_coeff_1.val     = 0.83;    // partition coefficient

      melting_temp.val    = 1910;      // K
      liquidus_slope_0.val =-543;      // K / at frac. - liquidous slope
      liquidus_slope_1.val =-1036;     // K / at frac. - liquidous slope
      part_coeff_0.val     = 0.94;    // partition coefficient
      part_coeff_1.val     = 0.83;    // partition coefficient

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

    case 5: // Ni - 5.8wt%Ta - 15.2wt%Al
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
      initial_conc_0.val   = 0.058;   // wt frac.
      initial_conc_1.val   = 0.152;   // wt frac.

      eps_c.val = 0*2.7207e-5;
      eps_v.val = 0*2.27e-2;
      eps_a.val = 0.05;
      symmetry.val = 4;

      // linearized phase diagram
      melting_temp.val     = 1754;     // K
      liquidus_slope_0.val =-517;     // K / wt frac. - liquidous slope
      liquidus_slope_1.val =-255;     // K / wt frac. - liquidous slope
      part_coeff_0.val     = 0.54;    // partition coefficient
      part_coeff_1.val     = 0.48;    // partition coefficient
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
    default:
      throw std::invalid_argument("Undefined alloy\n");
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
      case 5: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy, use linearized instead\n");
      default: throw std::invalid_argument("Invalid liquidus surface\n");
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
      default: throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}


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
  double RaT = density_l.val * beta_T.val * gravity * deltaT * pow(l_char.val, 3.0) / (mu_l.val * thermal_diff_l);
  PetscPrintf(mpicomm, "RED ALERT: delta T definition of Rayleigh number has same assumption as Stefan number \n");

  double deltaC_0 = initial_conc_0.val - eutectic_conc_0.val;
  double deltaC_1 = initial_conc_1.val - eutectic_conc_1.val;
  double deltaC_2 = initial_conc_2.val - eutectic_conc_2.val;
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
    case  5: return true;
    case  6: return (dir == 0 ? 1 : 0);
    case  7: return (dir == 0 ? 1 : 0);
    case  8: return (dir == 0 ? 1 : 0);
    default: throw;
  }
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
      case 0: return -(y - front_location()) + 0.000/(1.+100.*fabs(x/(xmin.val+xmax.val)-.5))*double(rand())/double(RAND_MAX)  + 0.000/(1.+1000.*fabs(x/(xmin.val+xmax.val)-.75));
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
      case 8:
      {
        // <Note make sure xmax = 3200 and ymax= 3200> // RochiNewExample
        double noise = 0.001;
        double x_center = xmax()/2.0;
        double y_center = ymax()/2.0;
        double theta = atan2(y-y_center, x- x_center);
        double r0 = 30.0;
        return r0*(1.0 - noise*fabs(pow(sin(2.*theta),2))) - sqrt(SQR(x- x_center) + SQR(y - y_center));
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
      case  8: return -1; // RochiNewExample
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
      case 8: return crystal_orientation.val;
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
        if (start_from_moving_front.val) {
          return (*initial_conc_all[idx])*(1. + (1.-analytic::kp[idx])/analytic::kp[idx]*
                                           exp(-cooling_velocity.val/(*solute_diff_all[idx])*(-front_phi_cf(DIM(x,y,z)-0*cooling_velocity.val*t))));
        }
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7:
      case  8:
        if (start_from_moving_front.val) {
          return (*initial_conc_all[idx])*(1. + (1.-analytic::kp[idx])/analytic::kp[idx]*
                                           exp(-cooling_velocity.val/(*solute_diff_all[idx])*(-front_phi_cf(DIM(x,y,z)-0*cooling_velocity.val*t))));
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
      case  7:
      case  8: return analytic::Cstar[idx];
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
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry())
    {
#ifdef P4_TO_P8
      case -3: return analytic::tl_exact(t, ABS3(x-xc.val, y-yc.val, z-zc.val));
#endif
      case -2: return analytic::tl_exact(t, ABS2(x-xc.val, y-yc.val));
      case -1: return analytic::tl_exact(t, ABS1(y-ymin.val));
      case  0: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())));
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())));
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  7: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient();
      case  8: return analytic::Tstar;
      default: throw;
    }
  }
} tl_cf;

// Solid temp:
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
      case  0: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  1: return analytic::Tstar;
      case  2: return analytic::Tstar;
      case  3: return analytic::Tstar - container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.+front_phi_cf(DIM(x,y,z))/(container_radius_outer()-front_location())))*thermal_cond_l()/thermal_cond_s();
      case  4: return analytic::Tstar + container_radius_outer()*temp_gradient()*log(MAX(0.001, 1.-front_phi_cf(DIM(x,y,z))/(container_radius_inner()+front_location())))*thermal_cond_l()/thermal_cond_s();
      case  5: return analytic::Tstar;
      case  6: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  7: return analytic::Tstar + (y - (front_location.val + cooling_velocity.val*t))*temp_gradient()*thermal_cond_l.val/thermal_cond_s.val;
      case  8: return analytic::Tstar;
      default: throw;
    }
  }
} ts_cf;

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
      case -1:
      case  0:
      case  1:
      case  2:
      case  3:
      case  4:
      case  5:
      case  6:
      case  7:
      case  8: return analytic::Tstar;
      default: throw;
    }
  }
} tf_cf;

// Front velo norm:
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
      default: return 0;
    }
  }
};

v_cf_t vn_cf(VAL);
v_cf_t vx_cf(DDX);
v_cf_t vy_cf(DDY);
v_cf_t vz_cf(DDZ);

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
        }
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
        }
      default: return 0;
    }
  }
} ft_cf;

double smooth_start(double t) { return smoothstep(smoothstep_order.val, (t+EPS)/(starting_time.val+EPS)); }

class bc_value_temp_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
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
          case  4:
          case  5:
          case  6:
          case  7: return 0;
          case 8:
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
} bc_value_temp;

class bc_value_conc_t : public CF_DIM
{
  int idx;
public:
  bc_value_conc_t(int idx) : idx(idx) {}
  double operator()(DIM(double x, double y, double z)) const
  {
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
          default: throw;
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
    if (geometry.val == 5) return -cooling_velocity.val*latent_heat.val*2.*PI*seed_radius.val/(xmax.val-xmin.val)/(ymax.val-ymin.val)/box_size.val*smooth_start(t);
    else return 0;
  }
} volumetric_heat_cf;


// ----------------------------------------
// define initial fields and boundary conditions for NS
// ----------------------------------------

// Velocity
class BC_INTERFACE_VALUE_VELOCITY: public my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t{

  public:
    unsigned char dir;
    BC_INTERFACE_VALUE_VELOCITY(my_p4est_stefan_with_fluids_t* parent_solver,bool do_we_use_vgamma_for_bc) : interfacial_bc_fluid_velocity_t(parent_solver, do_we_use_vgamma_for_bc=true){}
    double operator()(double x, double y) const
    {
      return 0.0;
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
  return (fabs(y - ymax.val) <= EPS);
};

bool is_x_wall(DIM(double x, double y, double z)){
  return (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));
};
bool is_y_wall(DIM(double x, double y, double z)){
  return (ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)));
};
// For velocity BCs/ICs
double u0=0.;
double v0=-1.;
double outflow_u=0.;
double outflow_v=0.;
// --------------------------------------------------------------------------------------------------------------
// Initial fluid velocity condition objects/functions: for fluid velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
struct INITIAL_VELOCITY : CF_DIM
{
  const unsigned char dir;

  INITIAL_VELOCITY(const unsigned char& dir_):dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double operator() (DIM(double x, double y,double z)) const{
      switch(dir){
      case dir::x:
        return u0;
      case dir::y:
        return v0;
      default:
        throw std::runtime_error("initial velocity direction unrecognized \n");
      }
  }
};
//----------------
// Velocity wall:
//----------------
// Value:

bool dirichlet_velocity_walls(DIM(double x, double y, double z)){
    // Dirichlet on y upper wall (where bulk flow is incoming
    return ( yupper_wall(DIM(x,y,z)) || xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)));

};
class BC_WALL_VALUE_VELOCITY: public CF_DIM
{
  public:
  const unsigned char dir;

  BC_WALL_VALUE_VELOCITY(const unsigned char& dir_):dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
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
    if(!dirichlet_velocity_walls(DIM(x,y,z))){ // pressure is dirichlet where vel is neumann
      return DIRICHLET;
    }
    else{
      return NEUMANN;
    }
  }
};
void initialize_all_relevant_ics_and_bcs_for_fluids(BC_INTERFACE_VALUE_VELOCITY* bc_interface_value_velocity[P4EST_DIM],
                                                    BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM],
                                                    BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM],
                                                    BC_INTERFACE_VALUE_PRESSURE& bc_interface_value_pressure,
                                                    BC_WALL_VALUE_PRESSURE& bc_wall_value_pressure,
                                                    BC_WALL_TYPE_PRESSURE& bc_wall_type_pressure,
                                                    my_p4est_multialloy_t* mas,
                                                    INITIAL_VELOCITY* v_init_cf[P4EST_DIM])
{
  my_p4est_stefan_with_fluids_t* stefan_w_fluid_solver;
  stefan_w_fluid_solver=mas->return_stefan_solver();
  for(unsigned char d=0;d<P4EST_DIM;d++){
        // Set the BC types:
        //std::cout<<"fn 1 \n";
        bc_interface_value_velocity[d] = new BC_INTERFACE_VALUE_VELOCITY(stefan_w_fluid_solver,true);
        //std::cout<<"fn 2 \n";
        BC_INTERFACE_TYPE_VELOCITY(d);
        //std::cout<<"fn 3 \n";
        bc_wall_type_velocity[d] = new BC_WALL_TYPE_VELOCITY(d);
        bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d);
        v_init_cf[d]= new INITIAL_VELOCITY(d);
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
                                                    my_p4est_multialloy_t* mas,
                                                    INITIAL_VELOCITY* v_init_cf[P4EST_DIM])
{
  for(unsigned char d=0;d<P4EST_DIM;d++){
        delete bc_wall_value_velocity[d];
        delete bc_wall_type_velocity[d];
        delete v_init_cf[d];
        delete bc_interface_value_velocity[d];
    }
    delete &bc_interface_value_pressure;
    delete &bc_wall_type_pressure;
}
int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  mpi_environment_t mpi;
  mpi.init(argc, argv);

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

  if (mpi.rank() == 0)
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
//  double lmin_eff = lmin_new + log(initial_division)/log(2.);

  /* initialize the solver */
  my_p4est_multialloy_t mas(num_comps.val, order_in_time.val);


  mas.initialize(mpi.comm(), xyz_min, xyz_max, n_xyz, periodic, phi_eff_cf, lmin_new, lmax_new, lip.val, band.val);
  ierr = PetscPrintf(mpi.comm(), "initialize complete \n"); CHKERRXX(ierr);

  bool solve_w_fluids=false;
  if(solve_w_fluids){
    mas.set_mpi_env(&mpi);
    mas.set_solve_with_fluids();
    //std::cout<<"Hello world 2_1 main \n";

    // Calculate nondimensional groups:
    compute_nondimensional_groups(mpi.comm(), &mas);
  }

  //std::cout<<"Hello world 3 main \n";
  if(solve_w_fluids){
    //std::cout<<"Hello world 3_1 main \n";
    mas.initialize_for_fluids();
    //std::cout<<"Hello world 3_2 main \n";
    initialize_all_relevant_ics_and_bcs_for_fluids(bc_interface_value_velocity,
                                                 bc_wall_value_velocity,
                                                 bc_wall_type_velocity,
                                                 bc_interface_value_pressure,
                                                 bc_wall_value_pressure,
                                                 bc_wall_type_pressure,
                                                 &mas,v_init_cf);
    //std::cout<<"Hello world 3_3 main \n";
  }
  ierr = PetscPrintf(mpi.comm(), "initialize for fluids complete \n"); CHKERRXX(ierr);
  p4est_t                   *p4est = mas.get_p4est();
  p4est_nodes_t             *nodes = mas.get_nodes();
  my_p4est_node_neighbors_t *ngbd  = mas.get_ngbd();
  //std::cout<<"Hello world 4 main \n";
  /* initialize the variables */
  vec_and_ptr_t front_phi(p4est, nodes);
  vec_and_ptr_t contr_phi(front_phi.vec);
  vec_and_ptr_t seed_map (front_phi.vec);

  sample_cf_on_nodes(p4est, nodes, front_phi_cf, front_phi.vec);
  sample_cf_on_nodes(p4est, nodes, contr_phi_cf, contr_phi.vec);
  sample_cf_on_nodes(p4est, nodes, seed_number_cf, seed_map.vec);

  /* set initial time step */
  double dxyz[P4EST_DIM];
  dxyz_min(p4est, dxyz);

  double DIM(dx = dxyz[0],
             dy = dxyz[1],
             dz = dxyz[2]);

  PetscPrintf(mpi.comm(), "Conc dt: %1.3e, Temp dt: %1.3e, Velo dt: %1.3e\n", .5*dx*dx/solute_diff_0.val, .5*dx*dx*density_l.val*heat_capacity_l.val/thermal_cond_l.val, cfl_number.val*dx/cooling_velocity.val);

  // perturb level set
  if (enforce_planar_front.val) init_perturb.val = 0;

  front_phi.get_array();

  srand(mpi.rank());

  foreach_node(n, nodes)
  {
    front_phi.ptr[n] += init_perturb.val*dx*double(rand())/double(RAND_MAX);
  }

  front_phi.restore_array();

  ierr = VecGhostUpdateBegin(front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  //  ls.reinitialize_2nd_order(front_phi.vec);
  //  ls.perturb_level_set_function(front_phi.vec, EPS);

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
  //std::cout<<"Hello world 1 main \n";
  mas.set_undercoolings(num_seeds(), seed_map.vec, eps_v_all.data(), eps_c_all.data());

  // set geometry
  mas.set_front(front_phi.vec);
  mas.set_container(contr_phi.vec);
  mas.set_scaling(scaling());

  // set boundary conditions
  mas.set_container_conditions_thermal(bc_type_temp.val, bc_value_temp);
  mas.set_container_conditions_composition(bc_type_conc.val, bc_value_conc_all);

  mas.set_wall_conditions_thermal(bc_type_temp.val, bc_value_temp);
  mas.set_wall_conditions_composition(bc_type_conc.val, bc_value_conc_all);
  mas.set_volumetric_heat(volumetric_heat_cf);

  // set time steps
  double dt_max = base_cfl.val*MIN(DIM(dx,dy,dz))/cooling_velocity.val;
  mas.set_dt_limits(0, dt_max);

  // set initial conditions
  mas.set_velocity(vn_cf, DIM(vx_cf, vy_cf, vz_cf), vf_cf);
  mas.set_temperature(tl_cf, ts_cf, tf_cf);
  mas.set_concentration(cl_cf_all, cs_cf_all);
  mas.set_ft(ft_cf);

  // Give the bc object back to the solver:


  // initialize for fluids if required
  if(solve_w_fluids){
    mas.initialize_for_fluids();
  }

  // set solver parameters
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
  vec_and_ptr_t ones(front_phi.vec);
  VecSetGhost(ones.vec, 1.);
  double container_volume = integrate_over_negative_domain(p4est, nodes, contr_phi.vec, ones.vec);
  ones.destroy();

  // clear up memory
  contr_phi.destroy();
  front_phi.destroy();
  seed_map.destroy();

  // loop over time
  bool   keep_going     = true;
  double tn             = 0;
  double total_growth   = 0;
  double base           = front_location.val;
  double bc_error_max   = 0;
  double bc_error_avg   = 0;
  int    iteration      = 0;
  int    sub_iterations = 0;
  int    vtk_idx        = 0;
  int    mpiret;

  vector<double> bc_error_max_all;
  vector<double> bc_error_avg_all;
  vector<int> num_pdes;

  std::vector<bool>   logevent;
  std::vector<double> old_time;
  std::vector<int>    old_count;

  std::vector<double> errors_max_over_time(2*num_comps.val+7, 0);
  int itn =1;

//  while(1){
//    PetscPrintf(mpi.comm(), "\n ------------------------- \n"
//                                 "Multialloy Main Iteration %d \n "
//                                 "------------------------- \n", iteration);
//    if (!keep_going) break;
////    std::cout<<"line 2262\n";
//    // compute time step
//    mas.compute_dt();
////    std::cout<<"line 2263\n";
//    if (tn + mas.get_dt() > time_limit.val) {
//      mas.set_dt(time_limit.val-tn);
//      keep_going = false;
//    }
////    std::cout<<"line 2270\n";
//    //if (iteration==1){
////    std::cout<<"line 2275\n";
//    // solve nonlinear system for temperature, concentration and velocity at t_n
//    bc_error_max = 0;
//    bc_error_avg = 0;
////    std::cout<<"line 2279\n";
//    sub_iterations += mas.one_step_w_fluids(2, &bc_error_max, &bc_error_avg, &num_pdes, &bc_error_max_all, &bc_error_avg_all);
//    PetscPrintf(mpi.comm(), "Onestep w fluids is complete \n");
//    mas.update_grid_w_fluids();
////    std::cout<<"line 2280\n";
//    mas.update_grid_solid();
////    std::cout<<"line 2282\n";
//    tn             += mas.get_dt();
////    std::cout<<"line 2284\n";
//    keep_going = keep_going && (iteration < step_limit.val) && (total_growth < growth_limit.val);
//    iteration++;
//  }

  while (1)
  {
    // determine to save or not
    bool save_now =
        (save_type.val == 0 && iteration    >= vtk_idx*save_every_dn.val) ||
        (save_type.val == 1 && total_growth >= vtk_idx*save_every_dl.val) ||
        (save_type.val == 2 && tn           >= vtk_idx*save_every_dt.val);



    if (!keep_going) break;

    // compute time step
    mas.compute_dt();

    if (tn + mas.get_dt() > time_limit.val) {
      mas.set_dt(time_limit.val-tn);
      keep_going = false;
    }

    // solve nonlinear system for temperature, concentration and velocity at t_n
    bc_error_max = 0;
    bc_error_avg = 0;

    sub_iterations += mas.one_step(2, &bc_error_max, &bc_error_avg, &num_pdes, &bc_error_max_all, &bc_error_avg_all);
    PetscPrintf(mpi.comm(), "Onestep is complete \n");
    tn             += mas.get_dt();

    if (save_step_convergence()) {
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

    ierr = PetscPrintf(mpi.comm(), "------------------------------------------------------------------------------------------------------\n"
                                   "Time step %d: growth %e, simulation time %e, compute time %e\n"
                                   "------------------------------------------------------------------------------------------------------\n",
                       iteration, total_growth, tn, w1.get_duration_current()); CHKERRXX(ierr);

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
    std::cout<<"main line 2327 ok \n";
    // save field data
    if (save_now)
    { //std::cout<<"main line 2330 ok \n";
      if (save_dendrites.val)   mas.count_dendrites(vtk_idx);
      //std::cout<<"main line 2332 ok \n";
      if (save_vtk.val)         mas.save_VTK(vtk_idx);
      //std::cout<<"main line 2334 ok \n";
      ierr = PetscPrintf(mpi.comm(), " Saving VTK is complete line 2059 \n"); CHKERRXX(ierr);
      if (save_vtk_solid.val)   mas.save_VTK_solid(vtk_idx);
      PetscPrintf(mpi.comm(), "VTK solid saving complete \n");
      ierr = PetscPrintf(mpi.comm(), "Saved characteristics in %s\n", filename_characteristics); CHKERRXX(ierr);
      if (save_p4est.val)       mas.save_p4est(vtk_idx);
      PetscPrintf(mpi.comm(), "P4est saving complete \n");
    }
    if (save_now) vtk_idx++;
    //std::cout<<"main line 2462 ok \n";
    PetscPrintf(mpi.comm(), "Solve_with_fluids paramter has been set \n");
    //std::cout<<"main line 2464 ok \n";
    if(solve_w_fluids){
      mas.update_grid_w_fluids();
    }else{
      mas.update_grid();
    }


    //mas.update_grid_w_fluids_v2();
    PetscPrintf(mpi.comm(), "Update grid with fluids is complete \n");
   // std::cout<<"main line 2468 ok \n";
    mas.update_grid_solid();
    keep_going = keep_going && (iteration < step_limit.val) && (total_growth < growth_limit.val);
    iteration++;

  }

  w1.stop(); w1.read_duration();

  return 0;
}

