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
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;

parameter_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int, px, 1, "Periodicity in the x-direction (0/1)");
DEFINE_PARAMETER(pl, int, py, 0, "Periodicity in the y-direction (0/1)");
DEFINE_PARAMETER(pl, int, pz, 0, "Periodicity in the z-direction (0/1)");

DEFINE_PARAMETER(pl, int, nx, 1, "Number of trees in the x-direction");
DEFINE_PARAMETER(pl, int, ny, 1, "Number of trees in the y-direction");
DEFINE_PARAMETER(pl, int, nz, 1, "Number of trees in the z-direction");

DEFINE_PARAMETER(pl, double, xmin, 0, "Box xmin");
DEFINE_PARAMETER(pl, double, ymin, 0, "Box ymin");
DEFINE_PARAMETER(pl, double, zmin, 0, "Box zmin");

DEFINE_PARAMETER(pl, double, xmax, 1, "Box xmax");
DEFINE_PARAMETER(pl, double, ymax, 1, "Box ymax");
DEFINE_PARAMETER(pl, double, zmax, 1, "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
DEFINE_PARAMETER(pl, int, lmin, 5, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 5, "Max level of the tree");
#else
DEFINE_PARAMETER(pl, int, lmin, 7, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 10, "Max level of the tree");
#endif

DEFINE_PARAMETER(pl, double, lip, 1.5, "");

//-------------------------------------
// solver parameters
//-------------------------------------
DEFINE_PARAMETER(pl, bool, use_points_on_interface,   false, "");
DEFINE_PARAMETER(pl, bool, use_superconvergent_robin, false, "");

DEFINE_PARAMETER(pl, int,    update_c0_robin, 1, "Solve for c0 using Robin BC: 0 - never, 1 - once, 2 - always");
DEFINE_PARAMETER(pl, int,    num_time_layers, 3, "");
DEFINE_PARAMETER(pl, int,    pin_every_n_iterations, 20, "");
DEFINE_PARAMETER(pl, int,    max_iterations,    10, "");
DEFINE_PARAMETER(pl, double, bc_tolerance,      1.e-5, "");
DEFINE_PARAMETER(pl, double, cfl_number, 0.25, "");
DEFINE_PARAMETER(pl, double, phi_thresh, 0.1, "");

//-------------------------------------
// output parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int,  save_every_n_iteration, 10, "");
DEFINE_PARAMETER(pl, bool, save_characteristics,    0, "");
DEFINE_PARAMETER(pl, bool, save_dendrites,          0, "");
DEFINE_PARAMETER(pl, bool, save_history,            0, "");
DEFINE_PARAMETER(pl, bool, save_params,             0, "");
DEFINE_PARAMETER(pl, bool, save_vtk,                0, "");

DEFINE_PARAMETER(pl, double, save_every_dl, 0.01, "");
DEFINE_PARAMETER(pl, double, save_every_dt, 0.1,  "");

DEFINE_PARAMETER(pl, int, save_type, 0, "0 - every n iterations, 1 - every dl of growth, 2 - every dt of time");

DEFINE_PARAMETER(pl, double, dendrite_cut_off_fraction, 1.05, "");
DEFINE_PARAMETER(pl, double, dendrite_min_length,       0.05, "");

// problem parameters
DEFINE_PARAMETER(pl, bool,   concentration_neumann, 1, "");
DEFINE_PARAMETER(pl, int,    max_total_iterations,  10, "");
DEFINE_PARAMETER(pl, double, time_limit,            DBL_MAX, "");
DEFINE_PARAMETER(pl, double, termination_length,    1.8, "");
DEFINE_PARAMETER(pl, double, init_perturb,          0.000, "");
DEFINE_PARAMETER(pl, bool,   enforce_planar_front,  0,"");

DEFINE_PARAMETER(pl, double, box_size, 1.e-1, "equivalent width (in x) of the box in cm");

double scaling = 1./box_size;

DEFINE_PARAMETER(pl, int, geometry, 0, "0 - directional solidification,"
                                       "1 - growth of a circular seed in a circular container,"
                                       "2 - radial directional solidification in,"
                                       "3 - radial directional solidification out,"
                                       "4 - 10 seeds in a periodic domain");

//-------------------------------------
// alloy parameters
//-------------------------------------
const int num_comps_max = 4; // Number of maximum components allowed
int num_comps = 1; // Number of components used

DEFINE_PARAMETER(pl, double, volumetric_heat,  0, "Volumetric heat generation, J/cm^3");
DEFINE_PARAMETER(pl, double, cooling_velocity, 0.01, "Cooling velocity, cm/s");
DEFINE_PARAMETER(pl, double, temp_gradient,    400, "Temperature gradient, K/cm");

DEFINE_PARAMETER(pl, int, smoothstep_order, 5, "Time for volumetric heat to fully switch on, s");
DEFINE_PARAMETER(pl, double, volumetric_heat_tau, 0, "Time for volumetric heat to fully switch on, s");
DEFINE_PARAMETER(pl, double, cooling_velocity_tau, 5.e-2, "Time for cooling velocity to fully switch on, s");

DEFINE_PARAMETER(pl, double, density_l, 8.88e-3, "Density of liq. alloy, kg.cm-3");
DEFINE_PARAMETER(pl, double, density_s, 8.88e-3, "Density of sol. alloy, kg.cm-3");

DEFINE_PARAMETER(pl, double, heat_capacity_l, 0.46e3, "Heat capacity of liq. alloy, J.kg-1.K-1 ");
DEFINE_PARAMETER(pl, double, heat_capacity_s, 0.46e3, "Heat capacity of sol. alloy, J.kg-1.K-1 ");

DEFINE_PARAMETER(pl, double, thermal_cond_l, 6.07e-1, "Thermal conductivity of liq. alloy, W.cm-1.K-1 ");
DEFINE_PARAMETER(pl, double, thermal_cond_s, 6.07e-1, "Thermal conductivity of sol. alloy, W.cm-1.K-1 ");

DEFINE_PARAMETER(pl, double, latent_heat, 2350, "Latent heat of fusion, J.cm-3");
DEFINE_PARAMETER(pl, double, melting_temp, 1728, "Pure-substance melting point for linearized slope, K");

DEFINE_PARAMETER(pl, bool, linearized_liquidus, 1, "Use linearized liquidus surface or true one");

DEFINE_PARAMETER(pl, double, liquidus_slope_0, -357, "Slope of linearized liqiudus w.r.t component no. 0, K^-1");
DEFINE_PARAMETER(pl, double, liquidus_slope_1, -357, "Slope of linearized liqiudus w.r.t component no. 1, K^-1");
DEFINE_PARAMETER(pl, double, liquidus_slope_2, -357, "Slope of linearized liqiudus w.r.t component no. 2, K^-1");
DEFINE_PARAMETER(pl, double, liquidus_slope_3, -357, "Slope of linearized liqiudus w.r.t component no. 3, K^-1");

DEFINE_PARAMETER(pl, double, part_coeff_0, 0.86, ""); /* partition coefficient */
DEFINE_PARAMETER(pl, double, part_coeff_1, 0.86, ""); /* partition coefficient */
DEFINE_PARAMETER(pl, double, part_coeff_2, 0.86, ""); /* partition coefficient */
DEFINE_PARAMETER(pl, double, part_coeff_3, 0.86, ""); /* partition coefficient */

DEFINE_PARAMETER(pl, double, initial_conc_0, 0.4, "");   // initial concentration                      - at frac.     */
DEFINE_PARAMETER(pl, double, initial_conc_1, 0.4, "");   // initial concentration                      - at frac.     */
DEFINE_PARAMETER(pl, double, initial_conc_2, 0.4, "");   // initial concentration                      - at frac.     */
DEFINE_PARAMETER(pl, double, initial_conc_3, 0.4, "");   // initial concentration                      - at frac.     */

DEFINE_PARAMETER(pl, double, solute_diff_0, 1.0e-5, "");   /* liquid concentration diffusion coefficient - cm2.s-1      */
DEFINE_PARAMETER(pl, double, solute_diff_1, 1.0e-5, "");   /* liquid concentration diffusion coefficient - cm2.s-1      */
DEFINE_PARAMETER(pl, double, solute_diff_2, 1.0e-5, "");   /* liquid concentration diffusion coefficient - cm2.s-1      */
DEFINE_PARAMETER(pl, double, solute_diff_3, 1.0e-5, "");   /* liquid concentration diffusion coefficient - cm2.s-1      */

DEFINE_PARAMETER(pl, double, eps_c, 0, ""); /* curvature undercooling coefficient         - cm.K         */
DEFINE_PARAMETER(pl, double, eps_v, 0, ""); /* kinetic undercooling coefficient           - s.K.cm-1     */
DEFINE_PARAMETER(pl, double, eps_a, 0, ""); /* anisotropy coefficient                                    */

DEFINE_PARAMETER(pl, int, alloy, 7, "0: Ni -  0.4at%Cu bi-alloy, "
                                    "1: Ni -  0.3at%Cu -  0.1at%Cu tri-alloy, "
                                    "2: Co - 10.7at%W  -  9.4at%Al tri-alloy, "
                                    "3: Co -  9.4at%Al - 10.7at%W  tri-alloy, "
                                    "4: Ni - 15.2wt%Al -  5.8wt%Ta tri-alloy, "
                                    "5: Ni -  5.8wt%Ta - 15.2wt%Al tri-alloy, "
                                    "6: a made-up tetra-alloy, "
                                    "7: a made-up penta-alloy");

double* liquidus_slope_all[] = { &liquidus_slope_0,
                                 &liquidus_slope_1,
                                 &liquidus_slope_2,
                                 &liquidus_slope_3 };

void set_alloy_parameters()
{
  switch (alloy)
  {
    case 0: // Ni - 0.4at%Cu
      density_l       = 8.88e-3; // kg.cm-3
      density_s       = 8.88e-3; // kg.cm-3
      heat_capacity_l = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s = 0.46e3;  // J.kg-1.K-1
      melting_temp    = 1728;    // K
      latent_heat     = 2350;    // J.cm-3
      thermal_cond_l  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s  = 6.07e-1; // W.cm-1.K-1

      num_comps = 1;
      solute_diff_0    = 1.e-5;
      liquidus_slope_0 = -357;
      initial_conc_0   = 0.4;
      part_coeff_0     = 0.86;

      eps_c = 0;
      eps_v = 0;
      eps_a = 0.05;

      break;
    case 1:// Ni - 0.3at%Cu - 0.1at%Cu
      density_l       = 8.88e-3; // kg.cm-3
      density_s       = 8.88e-3; // kg.cm-3
      heat_capacity_l = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s = 0.46e3;  // J.kg-1.K-1
      melting_temp    = 1728;    // K
      latent_heat     = 2350;    // J.cm-3
      thermal_cond_l  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s  = 6.07e-1; // W.cm-1.K-1

      num_comps = 2;
      solute_diff_0    = 1.e-4;
      solute_diff_1    = 1.e-4;
      liquidus_slope_0 = -357;
      liquidus_slope_1 = -357;
      initial_conc_0   = 0.3;
      initial_conc_1   = 0.1;
      part_coeff_0     = 0.86;
      part_coeff_1     = 0.86;

      eps_c = 0;
      eps_v = 0;
      eps_a = 0.05;
      break;
    case 7:// Ni - 0.3at%Cu - 0.1at%Cu
      density_l       = 8.88e-3; // kg.cm-3
      density_s       = 8.88e-3; // kg.cm-3
      heat_capacity_l = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s = 0.46e3;  // J.kg-1.K-1
      melting_temp    = 1728;    // K
      latent_heat     = 2350;    // J.cm-3
      thermal_cond_l  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s  = 6.07e-1; // W.cm-1.K-1

      num_comps = 4;
      solute_diff_0    = 1.e-5;
      solute_diff_1    = 2.e-5;
      solute_diff_2    = 4.e-5;
      solute_diff_3    = 8.e-5;
      liquidus_slope_0 = -357;
      liquidus_slope_1 = -357;
      liquidus_slope_2 = -357;
      liquidus_slope_3 = -357;
      initial_conc_0   = 0.1;
      initial_conc_1   = 0.1;
      initial_conc_2   = 0.1;
      initial_conc_3   = 0.1;
      part_coeff_0     = 0.86;
      part_coeff_1     = 0.86;
      part_coeff_2     = 0.86;
      part_coeff_3     = 0.86;

      eps_c = 0.1;
      eps_v = 0;
      eps_a = 0.0;
      break;
    default:
      throw std::invalid_argument("Undefined alloy\n");
  }
}

double liquidus_value(double *c)
{
  static double conc_term;
  if (linearized_liquidus)
  {
    conc_term = (*liquidus_slope_all[0])*c[0];

    for (int i = 1; i < num_comps; ++i) conc_term += (*liquidus_slope_all[i])*c[i];

    return conc_term;
  }
  else
  {
    switch (alloy)
    {
      case 0: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 1: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 2: // Co-W-Al
        return 1495
            +1.344000 * pow(c[1],1) * pow(c[0],0)
            +2.303000 * pow(c[1],0) * pow(c[0],1)
            -0.213300 * pow(c[1],2) * pow(c[0],0)
            +0.070170 * pow(c[1],1) * pow(c[0],1)
            -0.224600 * pow(c[1],0) * pow(c[0],2)
            -0.003831 * pow(c[1],3) * pow(c[0],0)
            -0.011570 * pow(c[1],2) * pow(c[0],1)
            -0.008250 * pow(c[1],1) * pow(c[0],2)
            +0.001531 * pow(c[1],0) * pow(c[0],3);
      case 3: // Co-Al-W
        return 1495
            +1.344000 * pow(c[0],1) * pow(c[1],0)
            +2.303000 * pow(c[0],0) * pow(c[1],1)
            -0.213300 * pow(c[0],2) * pow(c[1],0)
            +0.070170 * pow(c[0],1) * pow(c[1],1)
            -0.224600 * pow(c[0],0) * pow(c[1],2)
            -0.003831 * pow(c[0],3) * pow(c[1],0)
            -0.011570 * pow(c[0],2) * pow(c[1],1)
            -0.008250 * pow(c[0],1) * pow(c[1],2)
            +0.001531 * pow(c[0],0) * pow(c[1],3);
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
  if (linearized_liquidus)
  {
    switch (which_comp)
    {
      case 0: return liquidus_slope_0;
      case 1: return liquidus_slope_1;
      case 2: return liquidus_slope_2;
      case 3: return liquidus_slope_3;
    }
  }
  else
  {
    switch (alloy)
    {
      case 0: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 1: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 2: // Co-W-Al
        switch (which_comp)
        {
          case 0:
            return 0.
                +1.344000 * 1. *pow(c[1],0) * pow(c[0],0)
                +2.303000 * 0. *pow(c[1],0) * pow(c[0],1)
                -0.213300 * 2. *pow(c[1],1) * pow(c[0],0)
                +0.070170 * 1. *pow(c[1],0) * pow(c[0],1)
                -0.224600 * 0. *pow(c[1],0) * pow(c[0],2)
                -0.003831 * 3. *pow(c[1],2) * pow(c[0],0)
                -0.011570 * 2. *pow(c[1],1) * pow(c[0],1)
                -0.008250 * 1. *pow(c[1],0) * pow(c[0],2)
                +0.001531 * 0. *pow(c[1],0) * pow(c[0],3);
          case 1:
            return 0.
                +1.344000 * pow(c[1],1) * 0. * pow(c[0],0)
                +2.303000 * pow(c[1],0) * 1. * pow(c[0],0)
                -0.213300 * pow(c[1],2) * 0. * pow(c[0],0)
                +0.070170 * pow(c[1],1) * 1. * pow(c[0],0)
                -0.224600 * pow(c[1],0) * 2. * pow(c[0],1)
                -0.003831 * pow(c[1],3) * 0. * pow(c[0],0)
                -0.011570 * pow(c[1],2) * 1. * pow(c[0],0)
                -0.008250 * pow(c[1],1) * 2. * pow(c[0],1)
                +0.001531 * pow(c[1],0) * 3. * pow(c[0],2);
          default: throw std::invalid_argument("\n");
        }
      case 3: // Co-Al-W
        switch (which_comp)
        {
          case 0:
            return 0.
                +1.344000 * 1. *pow(c[0],0) * pow(c[1],0)
                +2.303000 * 0. *pow(c[0],0) * pow(c[1],1)
                -0.213300 * 2. *pow(c[0],1) * pow(c[1],0)
                +0.070170 * 1. *pow(c[0],0) * pow(c[1],1)
                -0.224600 * 0. *pow(c[0],0) * pow(c[1],2)
                -0.003831 * 3. *pow(c[0],2) * pow(c[1],0)
                -0.011570 * 2. *pow(c[0],1) * pow(c[1],1)
                -0.008250 * 1. *pow(c[0],0) * pow(c[1],2)
                +0.001531 * 0. *pow(c[0],0) * pow(c[1],3);
          case 1:
            return 0.
                +1.344000 * pow(c[0],1) * 0. * pow(c[1],0)
                +2.303000 * pow(c[0],0) * 1. * pow(c[1],0)
                -0.213300 * pow(c[0],2) * 0. * pow(c[1],0)
                +0.070170 * pow(c[0],1) * 1. * pow(c[1],0)
                -0.224600 * pow(c[0],0) * 2. * pow(c[1],1)
                -0.003831 * pow(c[0],3) * 0. * pow(c[1],0)
                -0.011570 * pow(c[0],2) * 1. * pow(c[1],0)
                -0.008250 * pow(c[0],1) * 2. * pow(c[1],1)
                +0.001531 * pow(c[0],0) * 3. * pow(c[1],2);
          default: throw std::invalid_argument("\n");
        }
      case 4: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 5: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 6: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      case 7: throw std::invalid_argument("Real liquidus surfaces is not available for this alloy\n");
      default: throw std::invalid_argument("Invalid liquidus surface\n");
    }
  }
}

class front_phi_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
//      case 0: return MIN(-(y - 0.1), sqrt(SQR(x-0.5) + SQR(y-0.05))-0.025);
      case 0: return -(y - 0.1);
      default: throw;
    }
  }
} front_phi_cf;

class seed_number_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
      case 0: return 0;
      default: throw;
    }
  }
} seed_number_cf;

int num_seeds()
{
  switch (geometry)
  {
    case 0: return 1;
    default: throw;
  }
}

class contr_phi_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
      case 0: return -1;
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

class wall_bc_type_temp_t : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} wall_bc_type_temp;

class wall_bc_value_temp_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry)
    {
      case 0:
        if (ABS(y-ymax)<EPS) return +(temp_gradient);
        if (ABS(y-ymin)<EPS) return -(temp_gradient + cooling_velocity*latent_heat/thermal_cond_s * smoothstep(smoothstep_order, (t+EPS)/(cooling_velocity_tau+EPS)));
        return 0;
      default: throw;
    }
  }
} wall_bc_value_temp;

class wall_bc_type_conc_t : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z)) const
  {
    return concentration_neumann ? NEUMANN : DIRICHLET;
  }
} wall_bc_type_conc;

WallBCDIM* wall_bc_type_conc_all[] = { &wall_bc_type_conc,
                                       &wall_bc_type_conc,
                                       &wall_bc_type_conc,
                                       &wall_bc_type_conc };

class wall_bc_value_conc_t : public CF_DIM
{
  double *c;
public:
  wall_bc_value_conc_t(double &c) : c(&c) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return concentration_neumann ? 0 : *c;
  }
};

wall_bc_value_conc_t wall_bc_value_conc_0(initial_conc_0);
wall_bc_value_conc_t wall_bc_value_conc_1(initial_conc_1);
wall_bc_value_conc_t wall_bc_value_conc_2(initial_conc_2);
wall_bc_value_conc_t wall_bc_value_conc_3(initial_conc_3);

CF_DIM* wall_bc_value_conc_all[] = { &wall_bc_value_conc_0,
                                     &wall_bc_value_conc_1,
                                     &wall_bc_value_conc_2,
                                     &wall_bc_value_conc_3 };

class cl_cf_t : public CF_DIM
{
  double *c0;
public:
  cl_cf_t(double &c0) : c0(&c0) {}
  double operator()(DIM(double, double, double)) const
  {
    return *c0;
  }
};

cl_cf_t cl_cf_0(initial_conc_0);
cl_cf_t cl_cf_1(initial_conc_1);
cl_cf_t cl_cf_2(initial_conc_2);
cl_cf_t cl_cf_3(initial_conc_3);

CF_DIM* cl_cf_all[] = { &cl_cf_0,
                        &cl_cf_1,
                        &cl_cf_2,
                        &cl_cf_3 };

class cs_cf_t : public CF_DIM
{
  double *c0;
  double *kp;
public:
  cs_cf_t(double &c0, double &kp) : c0(&c0), kp(&kp) {}
  double operator()(DIM(double, double, double)) const
  {
    return (*kp)*(*c0);
  }
};

cs_cf_t cs_cf_0(initial_conc_0, part_coeff_0);
cs_cf_t cs_cf_1(initial_conc_1, part_coeff_1);
cs_cf_t cs_cf_2(initial_conc_2, part_coeff_2);
cs_cf_t cs_cf_3(initial_conc_3, part_coeff_3);

CF_DIM* cs_cf_all[] = { &cs_cf_0,
                        &cs_cf_1,
                        &cs_cf_2,
                        &cs_cf_3 };

class tl_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    std::vector<double> c(num_comps);

    for (int i = 0; i < num_comps; ++i) c[i] = (*cl_cf_all[i])(DIM(x,y,z));
    switch (geometry)
    {
      case 0:
      return -front_phi_cf(DIM(x,y,z))*temp_gradient + liquidus_value(c.data()) + melting_temp;
      default: throw;
    }
  }
} tl_cf;

class ts_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    std::vector<double> c(num_comps);

    for (int i = 0; i < num_comps; ++i) c[i] = (*cl_cf_all[i])(DIM(x,y,z));
    switch (geometry)
    {
      case 0:
      return -front_phi_cf(DIM(x,y,z))*temp_gradient + liquidus_value(c.data()) + melting_temp;
      default: throw;
    }
  }
} ts_cf;

class vn_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return 0;
  }
} vn_cf;

class eps_c_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double nx, double ny, double nz)) const
  {
#ifdef P4_TO_P8
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_c*(1.0-4.0*eps_a*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_c*(1.-15.*eps_a*cos(4.*(theta)));
#endif
  }
};

class eps_v_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double nx, double ny, double nz)) const
  {
#ifdef P4_TO_P8
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_v*(1.0-4.0*eps_a*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
#else
    double theta = atan2(ny, nx);
    return eps_v*(1.-15.*eps_a*cos(4.*(theta)));
#endif
  }
};

class volumetric_heat_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return volumetric_heat * smoothstep(smoothstep_order, (t+EPS)/(volumetric_heat_tau+EPS));
  }
} volumetric_heat_cf;


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);

  alloy = cmd.get("alloy", alloy);
  set_alloy_parameters();

  pl.get_all(cmd);

  if (mpi.rank() == 0) pl.print_all();

  // prepare stuff for output
  FILE *fich;
  char name[10000];

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

  if (mpi.rank() == 0 && save_characteristics)
  {
    sprintf(name, "%s/characteristics.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "time "
                                          "average_interface_velocity "
                                          "max_interface_velocity "
                                          "front_area "
                                          "solid_volume "
                                          "time_elapsed iteration "
                                          "local_nodes "
                                          "ghost_nodes "
                                          "sub_iterations\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  if (mpi.rank() == 0 && save_params) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  if (save_vtk)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  if (save_dendrites)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/dendrites";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/dendrites directory");
  }

  // rescalling to 1x1x1 box
  scaling = 1./box_size;
  density_l      /= (scaling*scaling*scaling);
  density_s      /= (scaling*scaling*scaling);
  thermal_cond_l /= scaling;
  thermal_cond_s /= scaling;
  latent_heat    /= (scaling*scaling*scaling);
  eps_c          *= scaling;
  eps_v          /= scaling;

  solute_diff_0 *= (scaling*scaling);
  solute_diff_1 *= (scaling*scaling);
  solute_diff_2 *= (scaling*scaling);
  solute_diff_3 *= (scaling*scaling);

  volumetric_heat  /= (scaling*scaling*scaling);
  temp_gradient    /= scaling;
  cooling_velocity *= scaling;


  parStopWatch w1;
  w1.start("total time");

  int    n_xyz   [] = { DIM(nx, ny, nz) };
  int    periodic[] = { DIM(px, py, pz) };
  double xyz_min [] = { DIM(xmin, ymin, zmin) };
  double xyz_max [] = { DIM(xmax, ymax, zmax) };

  /* initialize the solver */
  my_p4est_multialloy_t mas(num_comps, num_time_layers);

  mas.initialize(mpi.comm(), xyz_min, xyz_max, n_xyz, periodic, phi_eff_cf, lmin, lmax, lip);

  p4est_t                   *p4est = mas.get_p4est();
  p4est_nodes_t             *nodes = mas.get_nodes();
  my_p4est_node_neighbors_t *ngbd  = mas.get_ngbd();

  /* initialize the variables */
  vec_and_ptr_t front_phi(p4est, nodes);
  vec_and_ptr_t contr_phi(front_phi.vec);
  vec_and_ptr_t seed_map (front_phi.vec);
  vec_and_ptr_t tl(front_phi.vec);
  vec_and_ptr_t ts(front_phi.vec);
  vec_and_ptr_t vn(front_phi.vec);
  vec_and_ptr_array_t cl(num_comps, front_phi.vec);
  vec_and_ptr_array_t cs(num_comps, front_phi.vec);

  sample_cf_on_nodes(p4est, nodes, front_phi_cf, front_phi.vec);
  sample_cf_on_nodes(p4est, nodes, contr_phi_cf, contr_phi.vec);
  sample_cf_on_nodes(p4est, nodes, seed_number_cf, seed_map.vec);
  sample_cf_on_nodes(p4est, nodes, tl_cf, tl.vec);
  sample_cf_on_nodes(p4est, nodes, ts_cf, ts.vec);
  sample_cf_on_nodes(p4est, nodes, vn_cf, vn.vec);

  for (int i = 0; i < num_comps; ++i)
  {
    sample_cf_on_nodes(p4est, nodes, (*cl_cf_all[i]), cl.vec[i]);
    sample_cf_on_nodes(p4est, nodes, (*cs_cf_all[i]), cs.vec[i]);
  }

  /* set initial time step */
  double dxyz[P4EST_DIM];
  dxyz_min(p4est, dxyz);

  double DIM(dx = dxyz[0],
             dy = dxyz[1],
             dz = dxyz[2]);

  // perturb level set
  if (enforce_planar_front) init_perturb = 0;

  front_phi.get_array();

  srand(mpi.rank());

  foreach_node(n, nodes)
  {
    front_phi.ptr[n] += init_perturb*dx*(double)(rand()%1000)/1000.;
  }

  front_phi.restore_array();

  ierr = VecGhostUpdateBegin(front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(front_phi.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(front_phi.vec);
  ls.perturb_level_set_function(front_phi.vec, EPS);

  // set alloy parameters
  double solute_diff_all[] = { solute_diff_0, solute_diff_1, solute_diff_2, solute_diff_3 };
  double part_coeff_all [] = { part_coeff_0, part_coeff_1, part_coeff_2, part_coeff_3 };

  mas.set_liquidus(melting_temp, liquidus_value, liquidus_slope);
  mas.set_composition_parameters(solute_diff_all, part_coeff_all);
  mas.set_thermal_parameters(latent_heat,
                             density_l, heat_capacity_l, thermal_cond_l,
                             density_s, heat_capacity_s, thermal_cond_s);

  eps_c_cf_t eps_c_cf; std::vector<CF_DIM *> eps_v_all(num_seeds(), &eps_c_cf);
  eps_v_cf_t eps_v_cf; std::vector<CF_DIM *> eps_c_all(num_seeds(), &eps_v_cf);

  mas.set_undercoolings(num_seeds(), seed_map.vec, eps_v_all.data(), eps_c_all.data());

  // set geometry
  mas.set_front(front_phi.vec);
  mas.set_container(contr_phi.vec);
  mas.set_scaling(scaling);

  // set boundary conditions
  std::vector<BoundaryConditionType> bc_conc_type(num_comps, concentration_neumann ? NEUMANN : DIRICHLET);
  mas.set_container_conditions_thermal(NEUMANN, wall_bc_value_temp);
  mas.set_container_conditions_composition(bc_conc_type.data(), wall_bc_value_conc_all);

  mas.set_wall_conditions_thermal(wall_bc_type_temp, wall_bc_value_temp);
  mas.set_wall_conditions_composition(wall_bc_type_conc_all, wall_bc_value_conc_all);

  // set time steps
  double dt = cfl_number*MIN(DIM(dx,dy,dz))/cooling_velocity;

  mas.set_dt(dt);
  mas.set_dt_limits(0.001*dt, 10.*dt);

  // set initial conditions
  mas.set_temperature(tl.vec, ts.vec);
  mas.set_concentration(cl.vec.data(), cs.vec.data());
  mas.set_normal_velocity(vn.vec);

  // set solver parameters
  mas.set_pin_every_n_iterations   (pin_every_n_iterations);
  mas.set_bc_tolerance             (bc_tolerance);
  mas.set_max_iterations           (max_iterations);
  mas.set_cfl                      (cfl_number);
  mas.set_phi_thresh               (phi_thresh);

  mas.set_use_superconvergent_robin(use_superconvergent_robin);
  mas.set_use_points_on_interface  (use_points_on_interface);
  mas.set_update_c0_robin          (update_c0_robin);
  mas.set_enforce_planar_front     (enforce_planar_front);

  mas.set_dendrite_cut_off_fraction(dendrite_cut_off_fraction);
  mas.set_dendrite_min_length      (dendrite_min_length);

  vec_and_ptr_t ones(front_phi.vec);
  VecSetGhost(ones.vec, 1.);
  double container_volume = integrate_over_negative_domain(p4est, nodes, contr_phi.vec, ones.vec);
  ones.destroy();

  // clear up memory
  tl.destroy();
  ts.destroy();
  vn.destroy();
  cl.destroy();
  cs.destroy();
  contr_phi.destroy();
  front_phi.destroy();


  // loop over time
  bool   keep_going     = true;
  double tn             = 0;
  double total_growth   = 0;
  double base           = 0.1;
  int    iteration      = 0;
  int    sub_iterations = 0;
  int    vtk_idx        = 0;
  int    mpiret;


  while (keep_going)
  {
    // check for time limit
    if (tn + mas.get_dt() > time_limit)
    {
      mas.set_dt(time_limit-tn);
      keep_going = false;
    }

    // solve nonlinear system for temperature, concentration and velocity at t_n
    sub_iterations += mas.one_step();
    tn             += mas.get_dt();


    // compute total growth
    total_growth = base;

    p4est         = mas.get_p4est();
    nodes         = mas.get_nodes();
    front_phi.vec = mas.get_front_phi();

    front_phi.get_array();
    foreach_node(n, nodes)
    {
      if (front_phi.ptr[n] > 0)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        total_growth = MAX(total_growth, xyz[1]);
      }
    }
    front_phi.restore_array();

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    total_growth -= base;

    ierr = PetscPrintf(mpi.comm(), "Iteration %d, growth %e, time %e\n", iteration, total_growth, tn); CHKERRXX(ierr);

    // determine to save or not
    bool save_now =
        (save_type == 0 && iteration    >= vtk_idx*save_every_n_iteration) ||
        (save_type == 1 && total_growth >= vtk_idx*save_every_dl) ||
        (save_type == 2 && tn           >= vtk_idx*save_every_dt);

    // save velocity, lenght of interface and area of solid phase in time
    if (save_characteristics && save_now)
    {
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
      double time_elapsed    = w1.read_duration_current();
      int    num_local_nodes = nodes->num_owned_indeps;
      int    num_ghost_nodes = nodes->indep_nodes.elem_count - num_local_nodes;

      int buffer[] = { num_local_nodes, num_ghost_nodes };
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &buffer, 2, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      num_local_nodes = buffer[0];
      num_ghost_nodes = buffer[1];

      ones.destroy();

      // write into file
      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e %e %d %d %d %d\n", tn, velocity_avg/scaling, mas.get_front_velocity_max()/scaling, front_area, solid_volume, time_elapsed, iteration, num_local_nodes, num_ghost_nodes, sub_iterations);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved velocity in %s\n", name); CHKERRXX(ierr);

      sub_iterations = 0;
    }

    // save field data
    if (save_vtk && save_now)
    {
      mas.count_dendrites(vtk_idx);
      mas.save_VTK(vtk_idx);
      mas.save_VTK_solid(vtk_idx);
    }

    keep_going = keep_going && (iteration < max_total_iterations) && (total_growth < termination_length);

    // advance front to t_{n+1}
    mas.compute_dt();
    mas.update_grid();

    iteration++;

    if (save_now) vtk_idx++;
  }

  w1.stop(); w1.read_duration();

  return 0;
}
