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
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_nodes_multialloy.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_shapes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_shapes.h>
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
DEFINE_PARAMETER(pl, int, px, 0, "Periodicity in the x-direction (0/1)");
DEFINE_PARAMETER(pl, int, py, 0, "Periodicity in the y-direction (0/1)");
DEFINE_PARAMETER(pl, int, pz, 0, "Periodicity in the z-direction (0/1)");

DEFINE_PARAMETER(pl, int, nx, 1, "Number of trees in the x-direction");
DEFINE_PARAMETER(pl, int, ny, 1, "Number of trees in the y-direction");
DEFINE_PARAMETER(pl, int, nz, 1, "Number of trees in the z-direction");

DEFINE_PARAMETER(pl, double, xmin, -1, "Box xmin");
DEFINE_PARAMETER(pl, double, ymin, -1, "Box ymin");
DEFINE_PARAMETER(pl, double, zmin, -1, "Box zmin");

DEFINE_PARAMETER(pl, double, xmax,  1, "Box xmax");
DEFINE_PARAMETER(pl, double, ymax,  1, "Box ymax");
DEFINE_PARAMETER(pl, double, zmax,  1, "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
DEFINE_PARAMETER(pl, int, lmin, 5, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 5, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           3, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#else
DEFINE_PARAMETER(pl, int, lmin, 7, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 7, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           4, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z, 1, "Number of grid shifts in the z-direction");
#endif

DEFINE_PARAMETER(pl, double, lip, 1.5, "");

//-------------------------------------
// solver parameters
//-------------------------------------
DEFINE_PARAMETER(pl, bool, use_points_on_interface,   false, "");
DEFINE_PARAMETER(pl, bool, update_c0_robin,           false, "");
DEFINE_PARAMETER(pl, bool, use_superconvergent_robin, false, "");

DEFINE_PARAMETER(pl, int,    pin_every_n_iterations, 1000, "");
DEFINE_PARAMETER(pl, int,    max_iterations,    100, "");
DEFINE_PARAMETER(pl, double, bc_tolerance,      1.e-11, "");

//-------------------------------------
// problem geometry
//-------------------------------------
DEFINE_PARAMETER(pl, double, box_size, 0.1, "equivalent width (in x) of the box in cm");

DEFINE_PARAMETER(pl, double, cfl_number, 0.1, "");
DEFINE_PARAMETER(pl, double, dt, 0.333, "");

DEFINE_PARAMETER(pl, int, front_geometry, 0, "0 - circular, 1 - flower-shaped, 2 - assymetrically flower-shaped, ");
DEFINE_PARAMETER(pl, int, container_geometry, 0, "0 - no container, 1 - circular");

//-------------------------------------
// test solutions
//-------------------------------------
DEFINE_PARAMETER(pl, int, n_tl, 0, "Test solution for Tl");
DEFINE_PARAMETER(pl, int, n_ts, 0, "Test solution for Ts");
DEFINE_PARAMETER(pl, int, n_c0, 2, "Test solution for C0");
DEFINE_PARAMETER(pl, int, n_c1, 0, "Test solution for C1");
DEFINE_PARAMETER(pl, int, n_c2, 0, "Test solution for C2");
DEFINE_PARAMETER(pl, int, n_c3, 0, "Test solution for C3");
DEFINE_PARAMETER(pl, int, n_vn, 1, "Test solution for vn");
DEFINE_PARAMETER(pl, int, n_guess, 3, "Guess for C0");

DEFINE_PARAMETER(pl, BoundaryConditionType, wc_type_thermal,     DIRICHLET, "Wall conditions type for temperature");
DEFINE_PARAMETER(pl, BoundaryConditionType, wc_type_composition, DIRICHLET, "Wall conditions type for concentrations");

DEFINE_PARAMETER(pl, BoundaryConditionType, bc_type_thermal,     DIRICHLET, "Boundary conditions type for temperature");
DEFINE_PARAMETER(pl, BoundaryConditionType, bc_type_composition, DIRICHLET, "Boundary conditions type for concentrations");

//-------------------------------------
// alloy parameters
//-------------------------------------
const int num_comps_max = 4; // Number of maximum components allowed
int num_comps = 1; // Number of components used

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

//-------------------------------------
// undercoolings
//-------------------------------------
DEFINE_PARAMETER(pl, double, seed_0_phase_xy, 0, "");
DEFINE_PARAMETER(pl, double, seed_1_phase_xy, 0, "");
DEFINE_PARAMETER(pl, double, seed_2_phase_xy, 0, "");
#ifdef P4_TO_P8
DEFINE_PARAMETER(pl, double, seed_0_phase_yz, 0, "");
DEFINE_PARAMETER(pl, double, seed_1_phase_yz, 0, "");
DEFINE_PARAMETER(pl, double, seed_2_phase_yz, 0, "");
DEFINE_PARAMETER(pl, double, seed_0_phase_zx, 0, "");
DEFINE_PARAMETER(pl, double, seed_1_phase_zx, 0, "");
DEFINE_PARAMETER(pl, double, seed_2_phase_zx, 0, "");
#endif

DEFINE_PARAMETER(pl, int, alloy, 1, "0: Ni -  0.4at%Cu bi-alloy, "
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
      solute_diff_0    = 1.e-3;
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
      solute_diff_0    = 1.e-1;
      solute_diff_1    = 1.e-1;
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
      density_l       = 1.88e-3; // kg.cm-3
      density_s       = 1.88e-3; // kg.cm-3
      heat_capacity_l = 0.46e3;  // J.kg-1.K-1
      heat_capacity_s = 0.46e3;  // J.kg-1.K-1
      melting_temp    = 1728;    // K
      latent_heat     = 2350;    // J.cm-3
      thermal_cond_l  = 6.07e-1; // W.cm-1.K-1
      thermal_cond_s  = 6.07e-1; // W.cm-1.K-1

      num_comps = 4;
      solute_diff_0    = 1.e-1;
      solute_diff_1    = 1.e-1;
      solute_diff_2    = 1.e-1;
      solute_diff_3    = 1.e-1;
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

      eps_c = 0;
      eps_v = 0;
      eps_a = 0.05;
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

//-------------------------------------
// output parameters
//-------------------------------------
bool save_vtk = true;

//-------------------------------------
// problem geometry
//-------------------------------------
double scaling = 1/box_size;

int num_seeds = 3;

int seed_geom_0 = 1;
int seed_geom_1 = 0;
int seed_geom_2 = 0;

void set_geometry()
{
  switch (front_geometry)
  {
    case 0: num_seeds = 1; seed_geom_0 = 1; seed_geom_1 = 0; seed_geom_2 = 0; break;
    case 1: num_seeds = 1; seed_geom_0 = 2; seed_geom_1 = 0; seed_geom_2 = 0; break;
    case 2: num_seeds = 1; seed_geom_0 = 3; seed_geom_1 = 0; seed_geom_2 = 0; break;
    case 3: num_seeds = 3; seed_geom_0 = 4; seed_geom_1 = 5; seed_geom_2 = 6; break;
    default: throw;
  }
}

class seed_phi_cf_t: public CF_DIM
{
  int *n;
  cf_value_type_t what;
public:
  seed_phi_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(*n)
    {
      case 0:
        switch (what) {
          case VAL: return 10;
          default: return 0;
        }
      case 1:
      {
        static double r0 = 0.49, DIM(x0 = 0.03, y0 = 0.02, z0 = 0);
        static radial_shaped_domain_t shape(r0, DIM(x0, y0, z0), -1);
        switch (what) {
          case VAL: return shape.phi(DIM(x,y,z));
          case CUR: return shape.phi_c(DIM(x,y,z));
          case DDX: return shape.phi_x(DIM(x,y,z));
          case DDY: return shape.phi_y(DIM(x,y,z));
#ifdef P4_TO_P8
          case DDZ: return shape.phi_z(DIM(x,y,z));
#endif
          default: return 0;
        }
      }

      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
};

seed_phi_cf_t seed_all_phi  [] = { seed_phi_cf_t(VAL,seed_geom_0), seed_phi_cf_t(VAL,seed_geom_1), seed_phi_cf_t(VAL,seed_geom_2) };
seed_phi_cf_t seed_all_phi_c[] = { seed_phi_cf_t(CUR,seed_geom_0), seed_phi_cf_t(CUR,seed_geom_1), seed_phi_cf_t(CUR,seed_geom_2) };
seed_phi_cf_t seed_all_phi_x[] = { seed_phi_cf_t(DDX,seed_geom_0), seed_phi_cf_t(DDX,seed_geom_1), seed_phi_cf_t(DDX,seed_geom_2) };
seed_phi_cf_t seed_all_phi_y[] = { seed_phi_cf_t(DDY,seed_geom_0), seed_phi_cf_t(DDY,seed_geom_1), seed_phi_cf_t(DDY,seed_geom_2) };
#ifdef P4_TO_P8
seed_phi_cf_t seed_all_phi_z[] = { seed_phi_cf_t(DDZ,seed_geom_0), seed_phi_cf_t(DDZ,seed_geom_1), seed_phi_cf_t(DDZ,seed_geom_2) };
#endif

int closest_seed(DIM(double x, double y, double z))
{
  static int closest;
  static double phi_closest;
  static double phi_current;

  closest = 0;
  phi_closest = seed_all_phi[0](DIM(x,y,z));

  for (int i = 1; i < num_seeds; ++i)
  {
    phi_current = seed_all_phi[i](DIM(x,y,z));
    if (fabs(phi_current) < fabs(phi_closest))
    {
      closest = i;
      phi_closest = phi_current;
    }
  }

  return closest;
}

class seed_number_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
//    int num = -1;

//    for (int i = 0; i < num_seeds; ++i) if (seed_all_phi[i](DIM(x,y,z)) < 0) num = i;

//    return num;

    return seed_all_phi[(int) closest_seed(DIM(x,y,z))](DIM(x,y,z));
  }
} seed_number_cf;


class front_phi_cf_t: public CF_DIM
{
  cf_value_type_t what;
public:
  front_phi_cf_t(cf_value_type_t what) : what(what) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (what) {
      case VAL: return seed_all_phi  [closest_seed(DIM(x,y,z))](DIM(x,y,z));
      case CUR: return seed_all_phi_c[closest_seed(DIM(x,y,z))](DIM(x,y,z));
      case DDX: return seed_all_phi_x[closest_seed(DIM(x,y,z))](DIM(x,y,z));
      case DDY: return seed_all_phi_y[closest_seed(DIM(x,y,z))](DIM(x,y,z));
#ifdef P4_TO_P8
      case DDZ: return seed_all_phi_z[closest_seed(DIM(x,y,z))](DIM(x,y,z));
#endif
      default: return 0;
    }
  }
};

front_phi_cf_t front_phi_cf  (VAL);
front_phi_cf_t front_phi_c_cf(CUR);
front_phi_cf_t front_phi_x_cf(DDX);
front_phi_cf_t front_phi_y_cf(DDY);
#ifdef P4_TO_P8
front_phi_cf_t front_phi_z_cf(DDZ);
#endif

class container_phi_cf_t: public CF_DIM
{
  int *n;
  cf_value_type_t what;
public:
  container_phi_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(*n)
    {
      case 0:
        switch (what) {
          case VAL: return -1;
          default: return 0;
        }
      case 1:
      {
        static double r0 = 0.84, DIM(x0 = 0, y0 = 0, z0 = 0);
        static radial_shaped_domain_t shape(r0, DIM(x0, y0, z0), 1);
        switch (what) {
          case VAL: return shape.phi(DIM(x,y,z));
          case CUR: return shape.phi_c(DIM(x,y,z));
          case DDX: return shape.phi_x(DIM(x,y,z));
          case DDY: return shape.phi_y(DIM(x,y,z));
#ifdef P4_TO_P8
          case DDZ: return shape.phi_z(DIM(x,y,z));
#endif
          default: return 0;
        }
      }

      default:
        throw std::invalid_argument("Choose a valid level set.");
    }
  }
};

container_phi_cf_t contr_phi_cf  (VAL, container_geometry);
container_phi_cf_t contr_phi_x_cf(DDX, container_geometry);
container_phi_cf_t contr_phi_y_cf(DDY, container_geometry);
#ifdef P4_TO_P8
container_phi_cf_t contr_phi_z_cf(DDZ, container_geometry);
#endif

//-------------------------------------
// undercoolings
//-------------------------------------
class undercooling_cf_t : public CF_DIM
{
  double *eps;
  double *phase_xy; // rotation around z axis
#ifdef P4_TO_P8
  double *phase_yz; // rotation around x axis
  double *phase_zx; // rotation around y axis
#endif
  double R[(int)pow(P4EST_DIM,2)]; // rotation matrix;
  bool initialized;
public:
#ifdef P4_TO_P8
  undercooling_cf_t(double &eps, double &phase_xy, double &phase_yz, double &phase_zx)
    : eps(&eps), phase_xy(&phase_xy), phase_yz(&phase_yz), phase_zx(&phase_zx), initialized(false) {}
#else
  undercooling_cf_t(double &eps, double &phase_xy)
    : eps(&eps), phase_xy(&phase_xy), initialized(false) {}
#endif

  inline void initialize()
  {
    if (!initialized)
    {
#ifdef P4_TO_P8
      R[0] =  cos(*phase_zx)*cos(*phase_xy);
      R[1] = -cos(*phase_yz)*sin(*phase_xy) + sin(*phase_yz)*sin(*phase_zx)*cos(*phase_xy);
      R[2] =  sin(*phase_yz)*sin(*phase_xy) + cos(*phase_yz)*sin(*phase_zx)*cos(*phase_xy);
      R[3] =  cos(*phase_zx)*sin(*phase_xy);
      R[4] =  cos(*phase_yz)*cos(*phase_xy) + sin(*phase_yz)*sin(*phase_zx)*sin(*phase_xy);
      R[5] = -sin(*phase_yz)*cos(*phase_xy) + cos(*phase_yz)*sin(*phase_zx)*sin(*phase_xy);
      R[6] = -sin(*phase_zx);
      R[7] =  sin(*phase_yz)*cos(*phase_zx);
      R[8] =  cos(*phase_yz)*cos(*phase_zx);
#else
      R[0] = cos(*phase_xy); R[1] = -sin(*phase_xy);
      R[2] = -R[1];          R[3] = R[0];
#endif
      initialized = true;
    }
  }

  double operator()(DIM(double nx, double ny, double nz)) const
  {
#ifdef P4_TO_P8
    double NX = R[0]*nx + R[1]*ny + R[2]*nz;
    double NY = R[3]*nx + R[4]*ny + R[5]*nz;
    double NZ = R[6]*nx + R[7]*ny + R[8]*nz;
    double norm = sqrt(NX*NX+NY*NY+NZ*NZ) + EPS;
    return (*eps)*(1.0-4.0*eps_a*((pow(NX, 4.) + pow(NY, 4.) + pow(NZ, 4.))/pow(norm, 4.) - 0.75));
#else
    double NX = R[0]*nx + R[1]*ny;
    double NY = R[2]*nx + R[3]*ny;
    double theta = atan2(ny, nx);
    return (*eps)*(1.0-15.0*eps_a*cos(4.0*(theta)));
#endif
  }
};

undercooling_cf_t eps_c_all[] = { undercooling_cf_t(eps_c, seed_0_phase_xy), undercooling_cf_t(eps_c, seed_1_phase_xy), undercooling_cf_t(eps_c, seed_2_phase_xy) };
undercooling_cf_t eps_v_all[] = { undercooling_cf_t(eps_v, seed_0_phase_xy), undercooling_cf_t(eps_v, seed_1_phase_xy), undercooling_cf_t(eps_v, seed_2_phase_xy) };

CF_DIM* eps_c_cf_all[] = { &eps_c_all[0], &eps_c_all[1], &eps_c_all[2] };
CF_DIM* eps_v_cf_all[] = { &eps_v_all[0], &eps_v_all[1], &eps_v_all[2] };

//-------------------------------------
// test solutions
//-------------------------------------
class test_cf_t : public CF_DIM
{
public:
  int    *n;
  double *mag;
  cf_value_type_t what;
  test_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(x)*cos(y)*exp(z);
          case DDX: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDY: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDZ: return  (*mag)*sin(x)*cos(y)*exp(z);
          case LAP: return -(*mag)*sin(x)*cos(y)*exp(z);
#else
          case VAL: return  (*mag)*sin(x)*cos(y);
          case DDX: return  (*mag)*cos(x)*cos(y);
          case DDY: return -(*mag)*sin(x)*sin(y);
          case LAP: return -(*mag)*2.*sin(x)*cos(y);
#endif
        }
      case 1: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*cos(x)*sin(y)*exp(z);
          case DDX: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDY: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDZ: return  (*mag)*cos(x)*sin(y)*exp(z);
          case LAP: return -(*mag)*cos(x)*sin(y)*exp(z);
#else
          case VAL: return  (*mag)*cos(x)*sin(y);
          case DDX: return -(*mag)*sin(x)*sin(y);
          case DDY: return  (*mag)*cos(x)*cos(y);
          case LAP: return -(*mag)*2.*cos(x)*sin(y);
#endif
        }
      case 2: switch (what) {
          case VAL: return (*mag)*exp(x);
          case DDX: return (*mag)*exp(x);
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return (*mag)*exp(x);
        }
      case 3: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*0.5*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDX: return (*mag)*0.5*( 1.0+2.0*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDY: return (*mag)*0.5*( 0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDZ: return (*mag)*0.5*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case LAP: return (*mag)*( 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
                -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.) );
#else
          case VAL: return (*mag)*0.5*log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDX: return (*mag)*(1.0*x+0.80*y+0.50)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDY: return (*mag)*(0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case LAP: {
              double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
              return (*mag)*( 1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C );
            };
#endif
        }
      case 4: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z) );
          case DDX: return (*mag)*( 1.0*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.) );
          case DDY: return (*mag)*( 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.)) );
          case DDZ: return (*mag)*( 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.)) );
          case LAP: return (*mag)*( ( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
                + ( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z) );
#else
          case VAL: return (*mag)*( log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y) );
          case DDX: return (*mag)*( ( 0.7/(0.7*x+3.) )*sin(x+0.5*y) + ( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case DDY: return (*mag)*( ( - 1./(y+3.) )*sin(x+0.5*y) + 0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case LAP: return (*mag)*( ( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
                + ( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y) );
#endif
        }
      case 5: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*exp(x+z-y*y)*(y+cos(x-z));
          case DDX: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
          case DDY: return (*mag)*exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
          case DDZ: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
          case LAP: return (*mag)*exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
#else
          case VAL: return (*mag)*exp(x-y*y)*(y+cos(x));
          case DDX: return (*mag)*exp(x-y*y)*(y+cos(x)-sin(x));
          case DDY: return (*mag)*exp(x-y*y)*(1.-2.*y*(y+cos(x)));
          case LAP: return (*mag)*exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
#endif
        }
      case 6: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5)) );
          case DDX: return (*mag)*( ( 1.0*cos(x+0.3*y)*cos(x-0.7*y) - 1.0*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5) );
          case DDY: return (*mag)*( ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5) );
          case DDZ: return (*mag)*( ( 1.0*cos(x-0.7*y)*sin(x+0.3*y)                                 )*exp(z) + 3.*z/(x*x+y*y+z*z+0.5) );
          case LAP: return (*mag)*( -1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.) );
#else
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5)) );
          case DDX: return (*mag)*(  1.00*cos(x+0.3*y)*cos(x-0.7*y) - 1.00*sin(x+0.3*y)*sin(x-0.7*y) + 3.*x/(x*x+y*y+0.5) );
          case DDY: return (*mag)*(  0.30*cos(x+0.3*y)*cos(x-0.7*y) + 0.70*sin(x+0.3*y)*sin(x-0.7*y) + 3.*y/(x*x+y*y+0.5) );
          case LAP: return (*mag)*( -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y) + 3./pow(x*x+y*y+0.5, 2.) );
#endif
        }
//      case 7: {
//        double p = mu_p_cf(DIM(x,y,z))/mu_m_cf(DIM(x,y,z));
//        double r = sqrt(x*x+y*y);
//        double r0 = 0.5+EPS;
//        switch (what) {
//#ifdef P4_TO_P8
//          case VAL: return (*mag);
//          case DDX: return 0;
//          case DDY: return 0;
//          case DDZ: return 0;
//          case LAP: return 0;
//#else
//          case VAL: return (*mag)*( x*(p+1.) - x*(p-1.)*r0*r0/r/r )        /( p+1.+r0*r0*(p-1.) );
//          case DDX: return (*mag)*( p+1.-r0*r0*(p-1.)*(y*y-x*x)/pow(r,4.) )/( p+1.+r0*r0*(p-1.) );
//          case DDY: return (*mag)*( (p-1.)*r0*r0*2.*x*y/pow(r,4.) )        /( p+1.+r0*r0*(p-1.) );
//          case LAP: return 0;
//#endif
//        }}
//      case 8: {
//        double p = mu_p_cf(DIM(x,y,z))/mu_m_cf(DIM(x,y,z));
//        double r = sqrt(x*x+y*y);
//        if (r < EPS) r = EPS;
//        double r0 = 0.5+EPS;
//        switch (what) {
//#ifdef P4_TO_P8
//          case VAL: return (*mag);
//          case DDX: return 0;
//          case DDY: return 0;
//          case DDZ: return 0;
//          case LAP: return 0;
//#else
//          case VAL: return (*mag)*2.*x/( p+1.+r0*r0*(p-1.) );
//          case DDX: return (*mag)*2.  /( p+1.+r0*r0*(p-1.) );
//          case DDY: return 0;
//          case LAP: return 0;
//#endif
//        }}
      case 9: {
#ifdef P4_TO_P8
        double r2 = x*x+y*y+z*z;
#else
        double r2 = x*x+y*y;
#endif
        if (r2 < EPS) r2 = EPS;
        switch (what) {
          case VAL: return (*mag)*(1+log(2.*sqrt(r2)));
          case DDX: return (*mag)*x/r2;
          case DDY: return (*mag)*y/r2;
#ifdef P4_TO_P8
          case DDZ: return (*mag)*z/r2;
#endif
          case LAP: return 0;
        }}
      case 10: {
        double phase_x =  0.13;
        double phase_y =  1.55;
        double phase_z =  0.7;
        double kx = 1;
        double ky = 1;
        double kz = 1;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDZ: return  (*mag)*PI*kz*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*cos(PI*kz*z+phase_z);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky+kz*kz)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
#else
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
#endif
        }}
      case 11: {
          double X    = (-x+y)/3.;
          double T5   = 16.*pow(X,5.)  - 20.*pow(X,3.) + 5.*X;
          double T5d  = 5.*(16.*pow(X,4.) - 12.*pow(X,2.) + 1.);
          double T5dd = 40.*X*(8.*X*X-3.);
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  T5*log(x+y+3.)*cos(z);
          case DDX: return  (T5/(x+y+3.) - T5d*log(x+y+3.)/3.)*cos(z);
          case DDY: return  (T5/(x+y+3.) + T5d*log(x+y+3.)/3.)*cos(z);
          case DDZ: return -T5*log(x+y+3.)*sin(z);
          case LAP: return  (2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.))*cos(z)
                - T5*log(x+y+3.)*cos(z);
#else
          case VAL: return T5*log(x+y+3.);
          case DDX: return T5/(x+y+3.) - T5d*log(x+y+3.)/3.;
          case DDY: return T5/(x+y+3.) + T5d*log(x+y+3.)/3.;
          case LAP: return 2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.);
#endif
        }}
      case 12: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(2.*x)*cos(2.*y)*exp(z);
          case DDX: return  (*mag)*2.*cos(2.*x)*cos(2.*y)*exp(z);
          case DDY: return -(*mag)*2.*sin(2.*x)*sin(2.*y)*exp(z);
          case DDZ: return  (*mag)*sin(2.*x)*cos(2.*y)*exp(z);
          case LAP: return -(*mag)*(2.*2.*2. - 1.)*sin(2.*x)*cos(2.*y)*exp(z);
#else
          case VAL: return  (*mag)*sin(2.*x)*cos(2.*y);
          case DDX: return  (*mag)*2.*cos(2.*x)*cos(2.*y);
          case DDY: return -(*mag)*2.*sin(2.*x)*sin(2.*y);
          case LAP: return -(*mag)*2.*2.*2.*sin(2.*x)*cos(2.*y);
#endif
        }
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};

double mag = 1;

test_cf_t tl_cf(VAL, n_tl, mag), tl_lap_cf(LAP, n_tl, mag), DIM(tl_x_cf(DDX, n_tl, mag), tl_y_cf(DDY, n_tl, mag), tl_z_cf(DDZ, n_tl, mag));
test_cf_t ts_cf(VAL, n_ts, mag), ts_lap_cf(LAP, n_ts, mag), DIM(ts_x_cf(DDX, n_ts, mag), ts_y_cf(DDY, n_ts, mag), ts_z_cf(DDZ, n_ts, mag));
test_cf_t c0_cf(VAL, n_c0, mag), c0_lap_cf(LAP, n_c0, mag), DIM(c0_x_cf(DDX, n_c0, mag), c0_y_cf(DDY, n_c0, mag), c0_z_cf(DDZ, n_c0, mag));
test_cf_t c1_cf(VAL, n_c1, mag), c1_lap_cf(LAP, n_c1, mag), DIM(c1_x_cf(DDX, n_c1, mag), c1_y_cf(DDY, n_c1, mag), c1_z_cf(DDZ, n_c1, mag));
test_cf_t c2_cf(VAL, n_c2, mag), c2_lap_cf(LAP, n_c2, mag), DIM(c2_x_cf(DDX, n_c2, mag), c2_y_cf(DDY, n_c2, mag), c2_z_cf(DDZ, n_c2, mag));
test_cf_t c3_cf(VAL, n_c3, mag), c3_lap_cf(LAP, n_c3, mag), DIM(c3_x_cf(DDX, n_c3, mag), c3_y_cf(DDY, n_c3, mag), c3_z_cf(DDZ, n_c3, mag));

CF_DIM* c_cf_all[] = { &c0_cf, &c1_cf, &c2_cf, &c3_cf };

class t_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return front_phi_cf(DIM(x,y,z)) < 0 ? tl_cf(DIM(x,y,z)) : ts_cf(DIM(x,y,z));
  }
} t_cf;

class vn_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (n_vn)
    {
      case 0: return 0;
      case 1: return 0.1;
      case 2: return 0.1*sin(x)*cos(y);
    }
  }
} vn_cf;

//------------------------------------------------------------
// right-hand-sides
//------------------------------------------------------------
class rhs_cf_t : public CF_DIM
{
  double  diag;
  double  diff;
  CF_DIM *sol;
  CF_DIM *lap;
public:
  rhs_cf_t(double diag, double diff, CF_DIM &sol, CF_DIM &lap)
    : diag(diag), diff(diff), sol(&sol), lap(&lap) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    return (diag)*(*sol)(DIM(x,y,z)) - (diff)*(*lap)(DIM(x,y,z));
  }
};

//------------------------------------------------------------
// boundary conditions at the front
//------------------------------------------------------------
class front_thermal_jump_value_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return ts_cf(DIM(x,y,z)) - tl_cf(DIM(x,y,z));
  }
} front_thermal_jump_value;

class front_thermal_jump_flux_t : public CF_DIM
{
  int j;
  double n[P4EST_DIM], norm;
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    static int j;
    static double nl[P4EST_DIM], norm;

    j = closest_seed(DIM(x,y,z));

    XCODE(nl[0] = seed_all_phi_x[j](DIM(x,y,z)));
    YCODE(nl[1] = seed_all_phi_y[j](DIM(x,y,z)));
    ZCODE(nl[2] = seed_all_phi_z[j](DIM(x,y,z)));

    norm = sqrt(SUMD(n[0]*n[0], n[1]*n[1], n[2]*n[2]))+EPS;

    return SUMD( (thermal_cond_s*ts_x_cf(DIM(x,y,z)) - thermal_cond_l*tl_x_cf(DIM(x,y,z)))*nl[0],
                 (thermal_cond_s*ts_y_cf(DIM(x,y,z)) - thermal_cond_l*tl_y_cf(DIM(x,y,z)))*nl[1],
                 (thermal_cond_s*ts_z_cf(DIM(x,y,z)) - thermal_cond_l*tl_z_cf(DIM(x,y,z)))*nl[2] ) / norm
        + latent_heat*vn_cf(DIM(x,y,z))*(1. + eps_c_all[j](DIM(n[0],n[1],n[2]))*seed_all_phi_c[j](DIM(x,y,z)));
  }
} front_thermal_jump_flux;

class front_conc_flux_t : public CF_DIM
{
  double *D, *kp;
  CF_DIM *c_cf, DIM(*cx_cf, *cy_cf, *cz_cf);

public:
  front_conc_flux_t(double &D, double &kp, CF_DIM &c_cf, DIM(CF_DIM &cx_cf, CF_DIM &cy_cf, CF_DIM &cz_cf))
    : D(&D), kp(&kp), c_cf(&c_cf), DIM(cx_cf(&cx_cf), cy_cf(&cy_cf), cz_cf(&cz_cf)) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    static int j;
    static double n[P4EST_DIM], norm;

    j = closest_seed(DIM(x,y,z));

    XCODE(n[0] = seed_all_phi_x[j](DIM(x,y,z)));
    YCODE(n[1] = seed_all_phi_y[j](DIM(x,y,z)));
    ZCODE(n[2] = seed_all_phi_z[j](DIM(x,y,z)));

    norm = sqrt(SUMD(n[0]*n[0], n[1]*n[1], n[2]*n[2]))+EPS;

    return (*D)*SUMD((*cx_cf)(DIM(x,y,z))*n[0],
                     (*cy_cf)(DIM(x,y,z))*n[1],
                     (*cz_cf)(DIM(x,y,z))*n[2])/norm - (1.-(*kp))*vn_cf(DIM(x,y,z))*(*c_cf)(DIM(x,y,z));
  }
};

front_conc_flux_t front_conc_flux_0(solute_diff_0, part_coeff_0, c0_cf, DIM(c0_x_cf, c0_y_cf, c0_z_cf));
front_conc_flux_t front_conc_flux_1(solute_diff_1, part_coeff_1, c1_cf, DIM(c1_x_cf, c1_y_cf, c1_z_cf));
front_conc_flux_t front_conc_flux_2(solute_diff_2, part_coeff_2, c2_cf, DIM(c2_x_cf, c2_y_cf, c2_z_cf));
front_conc_flux_t front_conc_flux_3(solute_diff_3, part_coeff_3, c3_cf, DIM(c3_x_cf, c3_y_cf, c3_z_cf));

CF_DIM* front_conc_flux_all[] = { &front_conc_flux_0,
                                  &front_conc_flux_1,
                                  &front_conc_flux_2,
                                  &front_conc_flux_3 };

class container_bc_value_t : public CF_DIM
{

  BoundaryConditionType *bc_type;
  double *D;
  CF_DIM *c_cf, DIM(*cx_cf, *cy_cf, *cz_cf);

public:
  container_bc_value_t(BoundaryConditionType &bc_type, double &D, CF_DIM &c_cf, DIM(CF_DIM &cx_cf, CF_DIM &cy_cf, CF_DIM &cz_cf))
    : bc_type(&bc_type), D(&D), c_cf(&c_cf), DIM(cx_cf(&cx_cf), cy_cf(&cy_cf), cz_cf(&cz_cf)) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    static int j;
    static double n[P4EST_DIM], norm;
    switch (*bc_type)
    {
      case DIRICHLET: return (*c_cf)(DIM(x,y,z));
      case NEUMANN:
        j = closest_seed(DIM(x,y,z));

        XCODE(n[0] = seed_all_phi_x[j](DIM(x,y,z)));
        YCODE(n[1] = seed_all_phi_y[j](DIM(x,y,z)));
        ZCODE(n[2] = seed_all_phi_z[j](DIM(x,y,z)));

        norm = sqrt(SUMD(n[0]*n[0], n[1]*n[1], n[2]*n[2]))+EPS;

        return (*D)*SUMD((*cx_cf)(DIM(x,y,z))*n[0],
                         (*cy_cf)(DIM(x,y,z))*n[1],
                         (*cz_cf)(DIM(x,y,z))*n[2])/norm;
      default: throw;
    }
  }
};

container_bc_value_t contr_bc_value_ts(bc_type_thermal, thermal_cond_s, ts_cf, DIM(ts_x_cf, ts_y_cf, ts_z_cf));
container_bc_value_t contr_bc_value_tl(bc_type_thermal, thermal_cond_l, tl_cf, DIM(tl_x_cf, tl_y_cf, tl_z_cf));

class contr_bc_value_temp_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return front_phi_cf(DIM(x,y,z)) < 0 ? contr_bc_value_tl(DIM(x,y,z)) : contr_bc_value_ts(DIM(x,y,z));
  }
} container_bc_value_thermal;

container_bc_value_t container_bc_value_composition_all[] = { container_bc_value_t(bc_type_composition, solute_diff_0, c0_cf, DIM(c0_x_cf, c0_y_cf, c0_z_cf)),
                                                              container_bc_value_t(bc_type_composition, solute_diff_1, c1_cf, DIM(c1_x_cf, c1_y_cf, c1_z_cf)),
                                                              container_bc_value_t(bc_type_composition, solute_diff_2, c2_cf, DIM(c2_x_cf, c2_y_cf, c2_z_cf)),
                                                              container_bc_value_t(bc_type_composition, solute_diff_3, c3_cf, DIM(c3_x_cf, c3_y_cf, c3_z_cf)) };

CF_DIM* container_bc_value_composition_cf_all[] = { &container_bc_value_composition_all[0],
                                                    &container_bc_value_composition_all[1],
                                                    &container_bc_value_composition_all[2],
                                                    &container_bc_value_composition_all[3] };

class wc_type_t : public WallBCDIM
{
  BoundaryConditionType *type;
public:
  wc_type_t(BoundaryConditionType &type) : type(&type) {}
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return *type;
  }
};

wc_type_t wc_type_thermal_cf    (wc_type_thermal);
wc_type_t wc_type_composition_cf(wc_type_composition);

WallBCDIM* wc_type_conc_all[] = { &wc_type_composition_cf,
                                  &wc_type_composition_cf,
                                  &wc_type_composition_cf,
                                  &wc_type_composition_cf };

class c0_guess_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (n_guess)
    {
      case 0: return c0_cf(DIM(x,y,z));
      case 1: return c0_cf(DIM(x,y,z)) + 0.1;
      case 2: return c0_cf(DIM(x,y,z)) + sin(x)*cos(y);
      case 3: return 1.;
    }
  }
} c0_guess;

class gibbs_thomson_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    static double concentrations[num_comps_max];
    static double n[P4EST_DIM];
    for (int i = 0; i < num_comps; ++i) concentrations[i] = (*c_cf_all[i])(DIM(x,y,z));

    int j = closest_seed(DIM(x,y,z));

    XCODE(n[0] = seed_all_phi_x[j](DIM(x,y,z)));
    YCODE(n[1] = seed_all_phi_y[j](DIM(x,y,z)));
    ZCODE(n[2] = seed_all_phi_z[j](DIM(x,y,z)));

    return tl_cf(DIM(x,y,z)) - liquidus_value(concentrations) -
        melting_temp*(1.0 + eps_c_all[j](DIM(n[0],n[1],n[2]))*seed_all_phi_c[j](DIM(x,y,z)))
        - eps_v_all[j](DIM(n[0],n[1],n[2]))*vn_cf(DIM(x,y,z));
  }
} gibbs_thomson;

class phi_refinement_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return MAX(contr_phi_cf(DIM(x,y,z)), -ABS(front_phi_cf(DIM(x,y,z))));
  }
} phi_refinement_cf;


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);

  alloy = cmd.get("alloy", alloy);
  set_alloy_parameters();

  pl.get_all(cmd);

//  if (mpi.rank() == 0) pl.print_all();
//  if (mpi.rank() == 0 && save_params) {
//    std::ostringstream file;
//    file << out_dir << "/parameters.dat";
//    pl.save_all(file.str().c_str());
//  }


  for (int i = 0; i < num_seeds; ++i)
  {
    eps_c_all[i].initialize();
    eps_v_all[i].initialize();
  }

  scaling = 1/box_size;
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

  parStopWatch w;
  w.start("total time");

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const int    n_xyz   [] = { nx, ny, nz };
  const int    periodic[] = { px, py, pz };
  const double xyz_min [] = { xmin, ymin, zmin };
  const double xyz_max [] = { xmax, ymax, zmax };
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_tl_n, err_tl_nm1;
  double err_ts_n, err_ts_nm1;
  double err_c0_n, err_c0_nm1;
  double err_c1_n, err_c1_nm1;
  double err_c2_n, err_c2_nm1;
  double err_c3_n, err_c3_nm1;

  double err_velo_n, err_velo_nm1;
  double err_curv_n, err_curv_nm1;
  double err_norm_n, err_norm_nm1;

  vector<double> h, e_v, e_tm, e_tp, e_c0, e_c1, e_c2, e_c3, e_g, error_it, pdes_it;

  for(int iter=0; iter<num_splits; ++iter)
  {
    e_g.clear();

    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data_tmp(lmin, lmax, &phi_refinement_cf, lip);
    p4est->user_pointer = (void*)(&data_tmp);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    for (int i = 0; i < iter; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
//      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &phi_refinement_cf, lip);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);
    ngbd_n.init_neighbors();

    my_p4est_level_set_t ls(&ngbd_n);

    /* find dx and dy smallest */
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);
#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
    double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    /* Initialize LSF */
    vec_and_ptr_t front_phi(p4est, nodes);
    vec_and_ptr_t contr_phi(p4est, nodes);

    sample_cf_on_nodes(p4est, nodes, front_phi_cf, front_phi.vec);
    sample_cf_on_nodes(p4est, nodes, contr_phi_cf, contr_phi.vec);

//    if (0) {
//      double xyz[P4EST_DIM];
//      double *phi_ptr;
//      srand(mpi.rank());

//      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
//      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
//      {
//        node_xyz_fr_n(n, p4est, nodes, xyz);

//        double d_phi = 0.000*dx*dx*dx*(double)(rand()%1000)/1000.;
////        double d_phi = 0.01*dx*dx*cos(2.*PI*data.max_lvl*xyz[0])*sin(2.*PI*data.max_lvl*xyz[1]);
//        phi_ptr[n] += d_phi;
//      }
//      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
//      ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
//      ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);
//    }

//    ls.reinitialize_1st_order_time_2nd_order_space(phi, 100);
//    ls.reinitialize_1st_order(phi, 100);
//    ls.reinitialize_2nd_order(phi, 100);
//    ls.perturb_level_set_function(phi, EPS);

    vec_and_ptr_t     front_curvature(front_phi.vec);
    vec_and_ptr_dim_t front_normal(p4est, nodes);

    compute_normals_and_mean_curvature(ngbd_n, front_phi.vec, front_normal.vec, front_curvature.vec);

    vec_and_ptr_dim_t front_phi_dd(front_normal.vec);
    vec_and_ptr_dim_t contr_phi_dd(front_normal.vec);

    ngbd_n.second_derivatives_central(front_phi.vec, front_phi_dd.vec);
    ngbd_n.second_derivatives_central(contr_phi.vec, contr_phi_dd.vec);

    /* Sample RHS */
    rhs_cf_t rhs_tl_cf(density_l*heat_capacity_l/dt, thermal_cond_l, tl_cf, tl_lap_cf);
    rhs_cf_t rhs_ts_cf(density_s*heat_capacity_s/dt, thermal_cond_s, ts_cf, ts_lap_cf);
    rhs_cf_t rhs_c0_cf(1./dt, solute_diff_0, c0_cf, c0_lap_cf);
    rhs_cf_t rhs_c1_cf(1./dt, solute_diff_1, c1_cf, c1_lap_cf);
    rhs_cf_t rhs_c2_cf(1./dt, solute_diff_2, c2_cf, c2_lap_cf);
    rhs_cf_t rhs_c3_cf(1./dt, solute_diff_3, c3_cf, c3_lap_cf);

    vec_and_ptr_t rhs_tl(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_tl_cf, rhs_tl.vec);
    vec_and_ptr_t rhs_ts(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_ts_cf, rhs_ts.vec);
    vec_and_ptr_t rhs_c0(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_c0_cf, rhs_c0.vec);
    vec_and_ptr_t rhs_c1(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_c1_cf, rhs_c1.vec);
    vec_and_ptr_t rhs_c2(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_c2_cf, rhs_c2.vec);
    vec_and_ptr_t rhs_c3(front_phi.vec); sample_cf_on_nodes(p4est, nodes, rhs_c3_cf, rhs_c3.vec);

    Vec rhs_c_all[] = { rhs_c0.vec, rhs_c1.vec, rhs_c2.vec, rhs_c3.vec };

    /* Sample seed map */
    vec_and_ptr_t seed_map(front_phi.vec); sample_cf_on_nodes(p4est, nodes, seed_number_cf, seed_map.vec);

    double solute_diff_all[] = { solute_diff_0,
                                 solute_diff_1,
                                 solute_diff_2,
                                 solute_diff_3 };
    double solute_diag_all[] = { 1./dt,
                                 1./dt,
                                 1./dt,
                                 1./dt };
    double part_coeff_all[] = { part_coeff_0,
                                part_coeff_1,
                                part_coeff_2,
                                part_coeff_3 };

    BoundaryConditionType bc_type_conc_all[] = { bc_type_composition,
                                                 bc_type_composition,
                                                 bc_type_composition,
                                                 bc_type_composition };

    /* set boundary conditions */
    my_p4est_poisson_nodes_multialloy_t solver_all_in_one(&ngbd_n);

    solver_all_in_one.set_front(front_phi.vec, front_phi_dd.vec, front_normal.vec, front_curvature.vec);

    solver_all_in_one.set_number_of_components(num_comps);
    solver_all_in_one.set_composition_parameters(solute_diag_all, solute_diff_all, part_coeff_all);
    solver_all_in_one.set_thermal_parameters(latent_heat,
                                             density_l*heat_capacity_l/dt, thermal_cond_l,
                                             density_s*heat_capacity_s/dt, thermal_cond_s);
    solver_all_in_one.set_gibbs_thomson(gibbs_thomson);
    solver_all_in_one.set_liquidus(melting_temp, liquidus_value, liquidus_slope);
    solver_all_in_one.set_undercoolings(num_seeds, seed_map.vec, eps_v_cf_all, eps_c_cf_all);

    solver_all_in_one.set_rhs(rhs_tl.vec, rhs_ts.vec, rhs_c_all);

    if (container_geometry > 0)
    {
      solver_all_in_one.set_container(contr_phi.vec, contr_phi_dd.vec);
      solver_all_in_one.set_container_conditions_thermal(bc_type_thermal, container_bc_value_thermal);
      solver_all_in_one.set_container_conditions_composition(bc_type_conc_all, container_bc_value_composition_cf_all);
    }

    solver_all_in_one.set_front_conditions(front_thermal_jump_value, front_thermal_jump_flux, front_conc_flux_all);

    solver_all_in_one.set_wall_conditions_thermal(wc_type_thermal_cf, t_cf);
    solver_all_in_one.set_wall_conditions_composition(wc_type_conc_all, c_cf_all);

    solver_all_in_one.set_pin_every_n_iterations(pin_every_n_iterations);
    solver_all_in_one.set_tolerance(bc_tolerance, max_iterations);
    solver_all_in_one.set_use_points_on_interface(use_points_on_interface);
    solver_all_in_one.set_update_c0_robin(update_c0_robin);
    solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin);

    solver_all_in_one.set_c0_guess(c0_guess);
    solver_all_in_one.set_vn(vn_cf);

    vec_and_ptr_t sol_tl(p4est, nodes);
    vec_and_ptr_t sol_ts(p4est, nodes);
    vec_and_ptr_t sol_c0(p4est, nodes);
    vec_and_ptr_t sol_c1(p4est, nodes);
    vec_and_ptr_t sol_c2(p4est, nodes);
    vec_and_ptr_t sol_c3(p4est, nodes);
    vec_and_ptr_t bc_error(p4est, nodes);
    vec_and_ptr_dim_t sol_c0d(p4est, nodes);

    Vec sol_c_all[] = { sol_c0.vec, sol_c1.vec, sol_c2.vec, sol_c3.vec };

//    Vec sol_tm_dd[P4EST_DIM];
//    Vec sol_tp_dd[P4EST_DIM];
//    Vec sol_c0_dd[P4EST_DIM];
//    Vec sol_c1_dd[P4EST_DIM];
//    Vec sol_c0n[P4EST_DIM];
//    Vec bc_error;

//    for (short dim = 0; dim < P4EST_DIM; ++dim)
//    {
//      ierr = VecDuplicate(phi_dd[dim], &sol_tm_dd[dim]); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi_dd[dim], &sol_tp_dd[dim]); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi_dd[dim], &sol_c0_dd[dim]); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi_dd[dim], &sol_c1_dd[dim]); CHKERRXX(ierr);
//      ierr = VecDuplicate(phi_dd[dim], &sol_c0n[dim]); CHKERRXX(ierr);
//    }

//    ierr = VecDuplicate(phi, &bc_error); CHKERRXX(ierr);


    double bc_error_max = 0;
    solver_all_in_one.solve(sol_tl.vec, sol_ts.vec, sol_c_all, sol_c0d.vec, bc_error.vec, bc_error_max, false, &pdes_it, &error_it);


    /* check the error */

    err_velo_nm1 = err_velo_n; err_velo_n = 0;
    err_curv_nm1 = err_curv_n; err_curv_n = 0;
    err_norm_nm1 = err_norm_n; err_norm_n = 0;

    err_tl_nm1 = err_tl_n; err_tl_n = 0;
    err_ts_nm1 = err_ts_n; err_ts_n = 0;
    err_c0_nm1 = err_c0_n; err_c0_n = 0;
    err_c1_nm1 = err_c1_n; err_c1_n = 0;
    err_c2_nm1 = err_c2_n; err_c2_n = 0;
    err_c3_nm1 = err_c3_n; err_c3_n = 0;

    my_p4est_poisson_nodes_mls_t *solver_temp = solver_all_in_one.get_solver_temp();

    int    idx;
    double xyz[P4EST_DIM];

    double velo_error = 0;
    double curv_error = 0;
    double norm_error = 0;

    vec_and_ptr_t err_velo(front_phi.vec);
    vec_and_ptr_t err_curv(front_phi.vec);
    vec_and_ptr_t err_norm(front_phi.vec);

    err_velo.get_array();
    err_curv.get_array();
    err_norm.get_array();

    my_p4est_interpolation_nodes_t interp(&ngbd_n);

    foreach_local_node(n, nodes)
    {
      err_velo.ptr[n] = 0;
      err_curv.ptr[n] = 0;
      err_norm.ptr[n] = 0;

      if (solver_temp->pw_jc_num_taylor_pts(0, n) > 0)
      {
        idx = solver_temp->pw_jc_idx_taylor_pt(0, n, 0);
        solver_temp->pw_jc_xyz_integr_pt(0, idx, xyz);

        err_velo.ptr[n] = fabs(vn_cf.value(xyz) - solver_all_in_one.compute_vn(xyz));

        interp.set_input(front_curvature.vec, linear);
        err_curv.ptr[n] = fabs(front_phi_c_cf.value(xyz) - interp.value(xyz));
      }
      err_velo_n = MAX(err_velo_n, err_velo.ptr[n]);
      err_curv_n = MAX(err_curv_n, err_curv.ptr[n]);
      err_norm_n = MAX(err_norm_n, err_norm.ptr[n]);
    }

    err_velo.restore_array();
    err_curv.restore_array();
    err_norm.restore_array();

    front_phi.get_array();
    contr_phi.get_array();

    vec_and_ptr_t err_tl(sol_tl.vec); err_tl.get_array();
    vec_and_ptr_t err_ts(sol_ts.vec); err_ts.get_array();
    vec_and_ptr_t err_c0(sol_c0.vec); err_c0.get_array();
    vec_and_ptr_t err_c1(sol_c1.vec); err_c1.get_array();
    vec_and_ptr_t err_c2(sol_c2.vec); err_c2.get_array();
    vec_and_ptr_t err_c3(sol_c3.vec); err_c3.get_array();

    sol_tl.get_array();
    sol_ts.get_array();
    sol_c0.get_array();
    sol_c1.get_array();
    sol_c2.get_array();
    sol_c3.get_array();

    foreach_local_node(n, nodes)
    {
      if (contr_phi.ptr[n] < 0)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);

        if (front_phi.ptr[n]<0)
        {
          err_tl.ptr[n] = fabs(sol_tl.ptr[n] - tl_cf.value(xyz));
          err_ts.ptr[n] = 0;
          err_c0.ptr[n] = fabs(sol_c0.ptr[n] - c0_cf.value(xyz));
          err_c1.ptr[n] = fabs(sol_c1.ptr[n] - c1_cf.value(xyz));
          err_c2.ptr[n] = fabs(sol_c2.ptr[n] - c2_cf.value(xyz));
          err_c3.ptr[n] = fabs(sol_c3.ptr[n] - c3_cf.value(xyz));
        } else {
          err_tl.ptr[n] = 0;
          err_ts.ptr[n] = fabs(sol_ts.ptr[n] - ts_cf.value(xyz));
          err_c0.ptr[n] = 0.;
          err_c1.ptr[n] = 0.;
          err_c2.ptr[n] = 0.;
          err_c3.ptr[n] = 0.;
        }

        err_tl_n = MAX(err_tl_n, err_tl.ptr[n]);
        err_ts_n = MAX(err_ts_n, err_ts.ptr[n]);
        err_c0_n = MAX(err_c0_n, err_c0.ptr[n]);
        err_c1_n = MAX(err_c1_n, err_c1.ptr[n]);
        err_c2_n = MAX(err_c2_n, err_c2.ptr[n]);
        err_c3_n = MAX(err_c3_n, err_c3.ptr[n]);
      }
    }

    front_phi.restore_array();
    contr_phi.restore_array();

    sol_tl.restore_array();
    sol_ts.restore_array();
    sol_c0.restore_array();
    sol_c1.restore_array();
    sol_c2.restore_array();
    sol_c3.restore_array();

    err_tl.restore_array();
    err_ts.restore_array();
    err_c0.restore_array();
    err_c1.restore_array();
    err_c2.restore_array();
    err_c3.restore_array();

    ierr = VecGhostUpdateBegin(err_tl.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_tl.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(err_ts.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_ts.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(err_c0.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c0.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(err_c1.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c1.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(err_c2.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c2.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(err_c3.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c3.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_velo_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_norm_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_curv_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tl_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ts_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c0_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c1_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c2_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c3_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    ierr = PetscPrintf(p4est->mpicomm, "Error in curv on nodes : %g, order = %g\n", err_curv_n, log(err_curv_nm1/err_curv_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in norm on nodes : %g, order = %g\n", err_norm_n, log(err_norm_nm1/err_norm_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in velo on nodes : %g, order = %g\n", err_velo_n, log(err_velo_nm1/err_velo_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in Tm on nodes : %g, order = %g\n", err_tl_n, log(err_tl_nm1 /err_tl_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in Tp on nodes : %g, order = %g\n", err_ts_n, log(err_ts_nm1 /err_ts_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C0 on nodes : %g, order = %g\n", err_c0_n, log(err_c0_nm1/err_c0_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C1 on nodes : %g, order = %g\n", err_c1_n, log(err_c1_nm1/err_c1_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C2 on nodes : %g, order = %g\n", err_c2_n, log(err_c2_nm1/err_c2_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C3 on nodes : %g, order = %g\n", err_c3_n, log(err_c3_nm1/err_c3_n)/log(2)); CHKERRXX(ierr);

    h.push_back(dx);
    e_v.push_back(err_velo_n);
    e_tm.push_back(err_tl_n);
    e_tp.push_back(err_ts_n);
    e_c0.push_back(err_c0_n);
    e_c1.push_back(err_c1_n);
    e_c2.push_back(err_c2_n);
    e_c3.push_back(err_c3_n);

    //-------------------------------------------------------------------------------------------
    // Save output
    //-------------------------------------------------------------------------------------------
    if(save_vtk)
    {
      PetscErrorCode ierr;
      const char *out_dir = getenv("OUT_DIR");
      if (!out_dir) {
        out_dir = "out_dir";
        system((string("mkdir -p ")+out_dir).c_str());
      }

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/multiallo_poisson_solver_"
          << "proc"
          << p4est->mpisize << "_"
             << "brick"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
//             "_levels=(" <<lmin << "," << lmax << ")" <<
             ".split" << iter;


      front_phi.get_array();
      contr_phi.get_array();
      sol_tl.get_array();
      sol_ts.get_array();
      sol_c0.get_array();
      sol_c1.get_array();
      sol_c2.get_array();
      sol_c3.get_array();
      err_tl.get_array();
      err_ts.get_array();
      err_c0.get_array();
      err_c1.get_array();
      err_c2.get_array();
      err_c3.get_array();
      bc_error.get_array();

      /* save the size of the leaves */
      Vec leaf_level;
      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      double *l_p;
      ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

      for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          l_p[tree->quadrants_offset+q] = quad->level;
        }
      }

      for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
      {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
      }

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             15, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", front_phi.ptr,
                             VTK_POINT_DATA, "phi_contr", contr_phi.ptr,
                             VTK_POINT_DATA, "sol_tm", sol_tl.ptr,
                             VTK_POINT_DATA, "sol_tp", sol_ts.ptr,
                             VTK_POINT_DATA, "sol_c0", sol_c0.ptr,
                             VTK_POINT_DATA, "sol_c1", sol_c1.ptr,
                             VTK_POINT_DATA, "sol_c2", sol_c2.ptr,
                             VTK_POINT_DATA, "sol_c3", sol_c3.ptr,
                             VTK_POINT_DATA, "err_tl", err_tl.ptr,
                             VTK_POINT_DATA, "err_ts", err_ts.ptr,
                             VTK_POINT_DATA, "err_c0", err_c0.ptr,
                             VTK_POINT_DATA, "err_c1", err_c1.ptr,
                             VTK_POINT_DATA, "err_c2", err_c2.ptr,
                             VTK_POINT_DATA, "err_c3", err_c3.ptr,
                             VTK_POINT_DATA, "bc_error", bc_error.ptr,
//                             VTK_POINT_DATA, "kappa_error", err_kappa_p,
//                             VTK_POINT_DATA, "vn_error", err_vn_p,
//                             VTK_POINT_DATA, "err_ex", err_ex_p,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      front_phi.restore_array();
      contr_phi.restore_array();
      sol_tl.restore_array();
      sol_ts.restore_array();
      sol_c0.restore_array();
      sol_c1.restore_array();
      sol_c2.restore_array();
      sol_c3.restore_array();
      err_tl.restore_array();
      err_ts.restore_array();
      err_c0.restore_array();
      err_c1.restore_array();
      err_c2.restore_array();
      err_c3.restore_array();
      bc_error.restore_array();

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }



    rhs_tl.destroy();
    rhs_ts.destroy();
    rhs_c0.destroy();
    rhs_c1.destroy();
    rhs_c2.destroy();
    rhs_c3.destroy();

    seed_map.destroy();
    sol_tl.destroy();
    sol_ts.destroy();
    sol_c0.destroy();
    sol_c1.destroy();
    sol_c2.destroy();
    sol_c3.destroy();
    bc_error.destroy();
    sol_c0d.destroy();
    err_tl.destroy();
    err_ts.destroy();
    err_c0.destroy();
    err_c1.destroy();
    err_c2.destroy();
    err_c3.destroy();

    front_curvature.destroy();
    front_phi_dd.destroy();
    contr_phi_dd.destroy();
    front_normal.destroy();

    front_phi.destroy();
    contr_phi.destroy();


    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  if (mpi.rank() == 0)
  {
    const char* out_dir = getenv("OUT_DIR");
    if (!out_dir)
    {
      ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save convergence results\n");
      return -1;
    }
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    std::string filename;

    // save level and resolution
    filename = out_dir; filename += "/convergence/h.txt";        save_vector(filename.c_str(), h);
    filename = out_dir; filename += "/convergence/error_v.txt";  save_vector(filename.c_str(), e_v);
    filename = out_dir; filename += "/convergence/error_tm.txt";  save_vector(filename.c_str(), e_tm);
    filename = out_dir; filename += "/convergence/error_tp.txt";  save_vector(filename.c_str(), e_tp);
    filename = out_dir; filename += "/convergence/error_c0.txt";  save_vector(filename.c_str(), e_c0);
    filename = out_dir; filename += "/convergence/error_c1.txt";  save_vector(filename.c_str(), e_c1);
    filename = out_dir; filename += "/convergence/error_pdes.txt";  save_vector(filename.c_str(), pdes_it);
    filename = out_dir; filename += "/convergence/error_error_it.txt";  save_vector(filename.c_str(), error_it);

    for (int i = 0; i < h.size(); ++i)
      std::cout << h[i] << " " << e_v[i] << " " << e_tm[i] << " " << e_tp[i] << " " << e_c0[i] << " " << e_c1[i] << "\n";
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
