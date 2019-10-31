
/*
 * Test the cell based multi level-set p4est.
 * Intersection of two circles
 *
 * run the program with the -help flag to see the available options
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
#include <src/my_p8est_macros.h>
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
#include <src/my_p4est_macros.h>
#include <src/my_p4est_shapes.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;

const int bdry_phi_max_num = 4;
const int infc_phi_max_num = 4;

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
DEFINE_PARAMETER(pl, int, lmin, 7, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 9, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           1, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#else
DEFINE_PARAMETER(pl, int, lmin, 9, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 9, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           1, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#endif

DEFINE_PARAMETER(pl, int, iter_start, 0, "Skip n first iterations");
DEFINE_PARAMETER(pl, double, lip, 10, "Lipschitz constant");

DEFINE_PARAMETER(pl, bool, refine_strict,  0, "Refines every cell starting from the coarsest case if yes");
DEFINE_PARAMETER(pl, bool, refine_rand,    0, "Add randomness into adaptive grid");
DEFINE_PARAMETER(pl, bool, balance_grid,   0, "Enforce 1:2 ratio for adaptive grid");
DEFINE_PARAMETER(pl, bool, coarse_outside, 0, "Use the coarsest possible grid outside the domain (0/1)");
DEFINE_PARAMETER(pl, int,  expand_ghost,   0, "Number of ghost layer expansions");

//-------------------------------------
// test solutions
//-------------------------------------

DEFINE_PARAMETER(pl, int, n_test, 0, "");

// boundary geometry
DEFINE_PARAMETER(pl, int, bdry_phi_num, 0, "Domain geometry");

DEFINE_PARAMETER(pl, bool, bdry_present_00, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, bdry_present_01, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, bdry_present_02, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, bdry_present_03, 0, "Domain geometry");

DEFINE_PARAMETER(pl, int, bdry_geom_00, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_geom_01, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_geom_02, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_geom_03, 0, "Domain geometry");

DEFINE_PARAMETER(pl, int, bdry_opn_00, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_opn_01, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_opn_02, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, bdry_opn_03, MLS_INTERSECTION, "Domain geometry");

//-------------------------------------
// level-set representation parameters
//-------------------------------------
DEFINE_PARAMETER(pl, bool, reinit_level_set, 1, "Reinitialize level-set function");

// artificial perturbation of level-set values
DEFINE_PARAMETER(pl, int,    dom_perturb,     0,   "Artificially pertub level-set functions (0 - no perturbation, 1 - smooth, 2 - noisy)");
DEFINE_PARAMETER(pl, double, dom_perturb_mag, 0.1, "Magnitude of level-set perturbations");
DEFINE_PARAMETER(pl, double, dom_perturb_pow, 2,   "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

//-------------------------------------
// convergence study parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int,    extend_solution,        1, "Extend solution after solving: 0 - no extension, 1 - extend using normal derivatives, 2 - extend using all derivatives");
DEFINE_PARAMETER(pl, bool,   use_nonzero_guess,      0, "");
DEFINE_PARAMETER(pl, double, extension_band_extend,  6000, "");
DEFINE_PARAMETER(pl, double, extension_band_compute, 80, "");
DEFINE_PARAMETER(pl, double, extension_band_check,   5, "");
DEFINE_PARAMETER(pl, double, extension_tol,          -1.e-16, "");
DEFINE_PARAMETER(pl, int,    extension_iterations,   200, "");


//-------------------------------------
// output
//-------------------------------------
DEFINE_PARAMETER(pl, bool, save_vtk,           1, "Save the p4est in vtk format");
DEFINE_PARAMETER(pl, bool, save_params,        0, "Save list of entered parameters");
DEFINE_PARAMETER(pl, bool, save_convergence,   0, "Save convergence results");

DEFINE_PARAMETER(pl, int, n_example, 3, "Predefined example");

void set_example(int n_example)
{
  switch (n_example)
  {
    case 0: // no boundaries, no interfaces

      n_test = 0;

      bdry_phi_num = 0;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

      break;
    case 1: // sphere interior

      n_test = 0;
      bdry_phi_num = 1;

      bdry_present_00 = 1; bdry_geom_00 = 1; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;
    case 2: // sphere exterior

      n_test = 0;
      bdry_phi_num = 1;

      bdry_present_00 = 1; bdry_geom_00 = 2; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;
    case 3: // moderately star-shaped boundary

      n_test = 0;
      bdry_phi_num = 1;

      bdry_present_00 = 1; bdry_geom_00 = 4; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;
    case 4: // highly star-shaped boundary

      n_test = 0;
      bdry_phi_num = 1;

      bdry_present_00 = 1; bdry_geom_00 = 5; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;
    case 5: // triangle/tetrahedron example from (Bochkov&Gibou, JCP, 2019)

      n_test = 0;
      bdry_phi_num = P4EST_DIM+1;

      bdry_present_00 = 1; bdry_geom_00 = 13; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 1; bdry_geom_01 = 14; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 1; bdry_geom_02 = 15; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 1; bdry_geom_03 = 16; bdry_opn_03 = MLS_INT;

      break;

    case 6: // union of spheres example from (Bochkov&Gibou, JCP, 2019)

      n_test = 3;
      bdry_phi_num = 2;

      bdry_present_00 = 1; bdry_geom_00 = 6; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 1; bdry_geom_01 = 7; bdry_opn_01 = MLS_ADD;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;

    case 7: // difference of spheres example from (Bochkov&Gibou, JCP, 2019)

      n_test = 4;
      bdry_phi_num = 2;

      bdry_present_00 = 1; bdry_geom_00 = 8; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 1; bdry_geom_01 = 9; bdry_opn_01 = MLS_INT;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT;

      break;

    case 8: // three stars example from (Bochkov&Gibou, JCP, 2019)

      n_test = 6;
      bdry_phi_num = 3;

      bdry_present_00 = 1; bdry_geom_00 = 10; bdry_opn_00 = MLS_INT;
      bdry_present_01 = 1; bdry_geom_01 = 11; bdry_opn_01 = MLS_ADD;
      bdry_present_02 = 1; bdry_geom_02 = 12; bdry_opn_02 = MLS_INT;
      bdry_present_03 = 0; bdry_geom_03 =  0; bdry_opn_03 = MLS_INT;

      break;
  }
}


bool *bdry_present_all[] = { &bdry_present_00,
                             &bdry_present_01,
                             &bdry_present_02,
                             &bdry_present_03 };

int *bdry_geom_all[] = { &bdry_geom_00,
                         &bdry_geom_01,
                         &bdry_geom_02,
                         &bdry_geom_03 };

int *bdry_opn_all[] = { &bdry_opn_00,
                        &bdry_opn_01,
                        &bdry_opn_02,
                        &bdry_opn_03 };

// TEST SOLUTIONS
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
          case LAP: return (*mag)/r2;
#else
          case LAP: return 0;
#endif
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
      case 13: {
          double X    = (-x+y)/3.;
          double T5   = 16.*pow(X,5.)  - 20.*pow(X,3.) + 5.*X;
          double T5d  = 5.*(16.*pow(X,4.) - 12.*pow(X,2.) + 1.);
          double T5dd = 40.*X*(8.*X*X-3.);
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  1.+T5*log(x+y+3.)*cos(z);
          case DDX: return  (T5/(x+y+3.) - T5d*log(x+y+3.)/3.)*cos(z);
          case DDY: return  (T5/(x+y+3.) + T5d*log(x+y+3.)/3.)*cos(z);
          case DDZ: return -T5*log(x+y+3.)*sin(z);
          case LAP: return  (2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.))*cos(z)
                - T5*log(x+y+3.)*cos(z);
#else
          case VAL: return 1.+T5*log(x+y+3.);
          case DDX: return T5/(x+y+3.) - T5d*log(x+y+3.)/3.;
          case DDY: return T5/(x+y+3.) + T5d*log(x+y+3.)/3.;
          case LAP: return 2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.);
#endif
        }}
      case 14: switch (what) {
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return 0;
        }
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};
double test_mag = 1;
test_cf_t test_cf(VAL, n_test, test_mag);

// DOMAIN GEOMETRY
class bdry_phi_cf_t: public CF_DIM {
public:
  int *n; // geometry number
  cf_value_type_t what;
  bdry_phi_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: // no boundaries
        break;
      case 1: // circle/sphere interior
      {
        static const double r0 = 0.911, DIM(xc = 0, yc = 0, zc = 0);
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc));
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 2: // circle/sphere exterior
      {
        static double r0 = 0.311, DIM(xc = 0, yc = 0, zc = 0);
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0, -1);
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 3: // annular/shell region
      {
        static double r0_in = 0.151, DIM(xc_in = 0, yc_in = 0, zc_in = 0);
        static double r0_ex = 0.911, DIM(xc_ex = 0, yc_ex = 0, zc_ex = 0);
        static flower_shaped_domain_t circle_in(r0_in, DIM(xc_in, yc_in, zc_in), 0, -1);
        static flower_shaped_domain_t circle_ex(r0_ex, DIM(xc_ex, yc_ex, zc_ex), 0,  1);
        switch (what) {
          OCOMP( case VAL: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi  (DIM(x,y,z)) : circle_ex.phi  (DIM(x,y,z)); );
          XCOMP( case DDX: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_x(DIM(x,y,z)) : circle_ex.phi_x(DIM(x,y,z)); );
          YCOMP( case DDY: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_y(DIM(x,y,z)) : circle_ex.phi_y(DIM(x,y,z)); );
          ZCOMP( case DDZ: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_z(DIM(x,y,z)) : circle_ex.phi_z(DIM(x,y,z)); );
        }
      } break;
      case 4: // moderately star-shaped domain
      {
        static double r0 = 0.611, DIM(xc = 0, yc = 0, zc = 0), deform = 0.15;
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform, -1);
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 5: // highly start-shaped domain
      {
        static double r0 = 0.611, DIM(xc = 0, yc = 0, zc = 0), deform = 0.3;
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform);
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 6: // unioun of two spheres: 1st sphere
      case 7: // unioun of two spheres: 2nd sphere
      {
#ifdef P4_TO_P8
        static double r0 = 0.71, xc0 = 0.22, yc0 = 0.17, zc0 = 0.21;
        static double r1 = 0.63, xc1 =-0.19, yc1 =-0.19, zc1 =-0.23;
#else
        static double r0 = 0.77, xc0 = 0.13, yc0 = 0.21;
        static double r1 = 0.49, xc1 =-0.33, yc1 =-0.37;
#endif
        static flower_shaped_domain_t circle0(r0, DIM(xc0, yc0, zc0));
        static flower_shaped_domain_t circle1(r1, DIM(xc1, yc1, zc1));

        flower_shaped_domain_t *shape_ptr = (*n) == 6 ? &circle0 : &circle1;

        switch (what) {
          OCOMP( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
        }
      } break;
      case 8: // difference of two spheres: 1st shpere
      case 9: // difference of two spheres: 2nd shpere
      {
#ifdef P4_TO_P8
        static double r0 = 0.86, xc0 = 0.08, yc0 = 0.11, zc0 = 0.03;
        static double r1 = 0.83, xc1 =-0.51, yc1 =-0.46, zc1 =-0.63;
#else
        static double r0 = 0.84, xc0 = 0.03, yc0 = 0.04;
        static double r1 = 0.63, xc1 =-0.42, yc1 =-0.37;
#endif
        static flower_shaped_domain_t circle0(r0, DIM(xc0, yc0, zc0), 0,  1);
        static flower_shaped_domain_t circle1(r1, DIM(xc1, yc1, zc1), 0, -1);

        flower_shaped_domain_t *shape_ptr = (*n) == 8 ? &circle0 : &circle1;

        switch (what) {
          OCOMP( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
        }

      } break;
      case 10: // three star-shaped domains: 1st domain
      case 11: // three star-shaped domains: 2nd domain
      case 12: // three star-shaped domains: 3rd domain
      {
#ifdef P4_TO_P8
        static double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, zc0 = 0.19, nx0 = 1.0, ny0 = 1.0, nz0 = 1.0, theta0 = 0.3*PI, beta0 = 0.08, inside0 = 1;
        static double r1 = 0.66, xc1 =-0.21, yc1 =-0.23, zc1 =-0.17, nx1 = 1.0, ny1 = 1.0, nz1 = 1.0, theta1 =-0.3*PI, beta1 =-0.08, inside1 = 1;
        static double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, zc2 = 0.03, nx2 =-1.0, ny2 = 1.0, nz2 = 0.0, theta2 =-0.2*PI, beta2 =-0.08, inside2 =-1;

        static flower_shaped_domain_t shape0(r0, xc0, yc0, zc0, beta0, inside0, nx0, ny0, nz0, theta0);
        static flower_shaped_domain_t shape1(r1, xc1, yc1, zc1, beta1, inside1, nx1, ny1, nz1, theta1);
        static flower_shaped_domain_t shape2(r2, xc2, yc2, zc2, beta2, inside2, nx2, ny2, nz2, theta2);
#else
        static double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, theta0 = 0.1*PI, beta0 = 0.08, inside0 = 1;
        static double r1 = 0.66, xc1 =-0.14, yc1 =-0.21, theta1 =-0.2*PI, beta1 =-0.08, inside1 = 1;
        static double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, theta2 = 0.2*PI, beta2 =-0.08, inside2 =-1;

        static flower_shaped_domain_t shape0(r0, xc0, yc0, beta0, inside0, theta0);
        static flower_shaped_domain_t shape1(r1, xc1, yc1, beta1, inside1, theta1);
        static flower_shaped_domain_t shape2(r2, xc2, yc2, beta2, inside2, theta2);
#endif

        flower_shaped_domain_t *shape_ptr;
        switch (*n){
          case 10: shape_ptr = &shape0; break;
          case 11: shape_ptr = &shape1; break;
          case 12: shape_ptr = &shape2; break;
        }

        switch (what) {
          OCOMP( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
        }
      } break;
      case 13: // triangle/tetrahedron: 1st plane
      case 14: // triangle/tetrahedron: 2nd plane
      case 15: // triangle/tetrahedron: 3rd plane
#ifdef P4_TO_P8
      case 16: // triangle/tetrahedron: 4th plane
#endif
      {
#ifdef P4_TO_P8
        static double x0 =-0.86, y0 =-0.87, z0 =-0.83;
        static double x1 = 0.88, y1 =-0.52, z1 = 0.63;
        static double x2 = 0.67, y2 = 0.82, z2 =-0.87;
        static double x3 =-0.78, y3 = 0.73, z3 = 0.85;

        static half_space_t plane0; plane0.set_params_points(x0, y0, z0, x2, y2, z2, x1, y1, z1);
        static half_space_t plane1; plane1.set_params_points(x1, y1, z1, x2, y2, z2, x3, y3, z3);
        static half_space_t plane2; plane2.set_params_points(x0, y0, z0, x3, y3, z3, x2, y2, z2);
        static half_space_t plane3; plane3.set_params_points(x0, y0, z0, x1, y1, z1, x3, y3, z3);
#else
        static double x2 = 0.74, y2 =-0.86;
        static double x1 =-0.83, y1 =-0.11;
        static double x0 = 0.37, y0 = 0.87;

        static half_space_t plane0; plane0.set_params_points(x0, y0, x2, y2);
        static half_space_t plane1; plane1.set_params_points(x2, y2, x1, y1);
        static half_space_t plane2; plane2.set_params_points(x1, y1, x0, y0);
#endif

        half_space_t *shape_ptr;
        switch (*n) {
          case 13: shape_ptr = &plane0; break;
          case 14: shape_ptr = &plane1; break;
          case 15: shape_ptr = &plane2; break;
#ifdef P4_TO_P8
          case 16: shape_ptr = &plane3; break;
#endif
        }

        switch (what) {
          OCOMP( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
        }
      } break;
    }

    // default value
    switch (what) {
      OCOMP( case VAL: return -1 );
      XCOMP( case DDX: return  0 );
      YCOMP( case DDY: return  0 );
      ZCOMP( case DDZ: return  0 );
    }
  }
};

bdry_phi_cf_t bdry_phi_cf_all  [] = { bdry_phi_cf_t(VAL, bdry_geom_00),
                                      bdry_phi_cf_t(VAL, bdry_geom_01),
                                      bdry_phi_cf_t(VAL, bdry_geom_02),
                                      bdry_phi_cf_t(VAL, bdry_geom_03) };

bdry_phi_cf_t bdry_phi_x_cf_all[] = { bdry_phi_cf_t(DDX, bdry_geom_00),
                                      bdry_phi_cf_t(DDX, bdry_geom_01),
                                      bdry_phi_cf_t(DDX, bdry_geom_02),
                                      bdry_phi_cf_t(DDX, bdry_geom_03) };

bdry_phi_cf_t bdry_phi_y_cf_all[] = { bdry_phi_cf_t(DDY, bdry_geom_00),
                                      bdry_phi_cf_t(DDY, bdry_geom_01),
                                      bdry_phi_cf_t(DDY, bdry_geom_02),
                                      bdry_phi_cf_t(DDY, bdry_geom_03) };
#ifdef P4_TO_P8
bdry_phi_cf_t bdry_phi_z_cf_all[] = { bdry_phi_cf_t(DDZ, bdry_geom_00),
                                      bdry_phi_cf_t(DDZ, bdry_geom_01),
                                      bdry_phi_cf_t(DDZ, bdry_geom_02),
                                      bdry_phi_cf_t(DDZ, bdry_geom_03) };
#endif

// the effective LSF (initialized in main!)
mls_eff_cf_t bdry_phi_eff_cf;

class perturb_cf_t: public CF_DIM
{
  int *n;
public:
  perturb_cf_t(int &n) : n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: return 0;
      case 1:
#ifdef P4_TO_P8
        return sin(2.*x-z)*cos(2.*y+z);
#else
        return sin(10.*x)*cos(10.*y);
#endif
      case 2: return 2.*((double) rand() / (double) RAND_MAX - 0.5);
      default: throw std::invalid_argument("Invalid test number\n");
    }

  }
} dom_perturb_cf(dom_perturb);

// additional output functions
double compute_convergence_order(std::vector<double> &x, std::vector<double> &y);

int main (int argc, char* argv[])
{
  // error variables
  PetscErrorCode ierr;
  int mpiret;

  // mpi
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // prepare output directories
  const char* out_dir = getenv("OUT_DIR");

  if (!out_dir &&
      (save_vtk  ||
       save_convergence ))
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save results\n");
    return -1;
  }

  if (save_vtk)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  if (save_convergence)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/convergence directory");
  }

  // parse command line arguments
  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);

  n_example = cmd.get("n_example", n_example);
  set_example(n_example);

  pl.get_all(cmd);

  if (mpi.rank() == 0) pl.print_all();
  if (mpi.rank() == 0 && save_params) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  // initialize effective level-sets
  for (int i = 0; i < bdry_phi_max_num; ++i)
  {
    if (*bdry_present_all[i] == true) bdry_phi_eff_cf.add_domain(bdry_phi_cf_all[i], (mls_opn_t) *bdry_opn_all[i]);
  }

  int num_shifts_total = MULTD(num_shifts_x_dir, num_shifts_y_dir, num_shifts_z_dir);

  int num_resolutions = ((num_splits-1)*num_splits_per_split + 1);
  int num_iter_total = num_resolutions*num_shifts_total;

  const int periodicity[] = { DIM(px, py, pz) };
  const int num_trees[]   = { DIM(nx, ny, nz) };
  const double grid_xyz_min[] = { DIM(xmin, ymin, zmin) };
  const double grid_xyz_max[] = { DIM(xmax, ymax, zmax) };

  // vectors to store convergence results
  vector<double> lvl_arr, h_arr;
  vector<double> error_ex_arr;

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  int iteration = -1;
  int file_idx  = -1;

    for(int iter=0; iter<num_splits; ++iter)
    {
      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin+iter, lmax+iter); CHKERRXX(ierr);

      int num_sub_iter = (iter == 0 ? 1 : num_splits_per_split);

      for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
      {

        double grid_xyz_min_alt[3];
        double grid_xyz_max_alt[3];

        double scale = (double) (num_sub_iter-1-sub_iter) / (double) num_sub_iter;
        grid_xyz_min_alt[0] = grid_xyz_min[0] - .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]); grid_xyz_max_alt[0] = grid_xyz_max[0] + .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]);
        grid_xyz_min_alt[1] = grid_xyz_min[1] - .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]); grid_xyz_max_alt[1] = grid_xyz_max[1] + .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]);
        grid_xyz_min_alt[2] = grid_xyz_min[2] - .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]); grid_xyz_max_alt[2] = grid_xyz_max[2] + .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]);


        double dxyz[3] = { (grid_xyz_max_alt[0]-grid_xyz_min_alt[0])/pow(2., (double) lmax+iter),
                           (grid_xyz_max_alt[1]-grid_xyz_min_alt[1])/pow(2., (double) lmax+iter),
                           (grid_xyz_max_alt[2]-grid_xyz_min_alt[2])/pow(2., (double) lmax+iter) };

        double grid_xyz_min_shift[3];
        double grid_xyz_max_shift[3];

        double dxyz_m = MIN(DIM(dxyz[0],dxyz[1],dxyz[2]));

        h_arr.push_back(dxyz_m);
        lvl_arr.push_back(lmax+iter-scale);

        ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f).\n", lmin+iter, lmax+iter, sub_iter, lmin+iter-scale, lmax+iter-scale); CHKERRXX(ierr);

#ifdef P4_TO_P8
        for (int k_shift = 0; k_shift < num_shifts_z_dir; ++k_shift)
        {
          grid_xyz_min_shift[2] = grid_xyz_min_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir) * dxyz[2];
          grid_xyz_max_shift[2] = grid_xyz_max_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir) * dxyz[2];
#endif
          for (int j_shift = 0; j_shift < num_shifts_y_dir; ++j_shift)
          {
            grid_xyz_min_shift[1] = grid_xyz_min_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir) * dxyz[1];
            grid_xyz_max_shift[1] = grid_xyz_max_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir) * dxyz[1];

            for (int i_shift = 0; i_shift < num_shifts_x_dir; ++i_shift)
            {
              grid_xyz_min_shift[0] = grid_xyz_min_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir) * dxyz[0];
              grid_xyz_max_shift[0] = grid_xyz_max_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir) * dxyz[0];

              iteration++;

              if (iteration < iter_start) continue;

              file_idx++;

              connectivity = my_p4est_brick_new(num_trees, grid_xyz_min_shift, grid_xyz_max_shift, &brick, periodicity);
              p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

              if (refine_strict)
              {
                splitting_criteria_cf_t data_tmp(lmin, lmax, &bdry_phi_eff_cf, lip);
                p4est->user_pointer = (void*)(&data_tmp);

                my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
                for (int i = 0; i < iter; ++i)
                {
                  my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
                  my_p4est_partition(p4est, P4EST_FALSE, NULL);
                }
              } else {
                splitting_criteria_cf_t data_tmp(lmin+iter, lmax+iter, &bdry_phi_eff_cf, lip);
                p4est->user_pointer = (void*)(&data_tmp);
                my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
              }

              splitting_criteria_cf_t data(lmin+iter, lmax+iter, &bdry_phi_eff_cf, lip);
              p4est->user_pointer = (void*)(&data);

              if (refine_rand)
                my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);

              if (balance_grid)
              {
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
                p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
              }

              ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
              if (expand_ghost)
                my_p4est_ghost_expand(p4est, ghost);
              nodes = my_p4est_nodes_new(p4est, ghost);

              my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
              my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);
              ngbd_n.init_neighbors();

              my_p4est_level_set_t ls(&ngbd_n);
              ls.set_show_convergence(1);

              double dxyz[P4EST_DIM];
              dxyz_min(p4est, dxyz);

              double dxyz_max = MAX(DIM(dxyz[0], dxyz[1], dxyz[2]));
              double diag = sqrt(SUMD(dxyz[0]*dxyz[0], dxyz[1]*dxyz[1], dxyz[2]*dxyz[2]));

              // sample level-set functions
              Vec bdry_phi_vec_all[bdry_phi_max_num];

              for (int i = 0; i < bdry_phi_max_num; ++i)
              {
                if (*bdry_present_all[i] == true)
                {
                  ierr = VecCreateGhostNodes(p4est, nodes, &bdry_phi_vec_all[i]); CHKERRXX(ierr);
                  sample_cf_on_nodes(p4est, nodes, bdry_phi_cf_all[i], bdry_phi_vec_all[i]);

                  if (dom_perturb)
                  {
                    double *phi_ptr;
                    ierr = VecGetArray(bdry_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                    {
                      double xyz[P4EST_DIM];
                      node_xyz_fr_n(n, p4est, nodes, xyz);
                      phi_ptr[n] += dom_perturb_mag*dom_perturb_cf.value(xyz)*pow(dxyz_m, dom_perturb_pow);
                    }

                    ierr = VecRestoreArray(bdry_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    ierr = VecGhostUpdateBegin(bdry_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                    ierr = VecGhostUpdateEnd  (bdry_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                  }

//                  if (reinit_level_set)
//                  {
//                    ls.reinitialize_1st_order_time_2nd_order_space(bdry_phi_vec_all[i], 20);
//                  }
                }
              }

              vec_and_ptr_t phi_eff(p4est, nodes);
              sample_cf_on_nodes(p4est, nodes, bdry_phi_eff_cf, phi_eff.vec);

              if (reinit_level_set)
              {
//                ls.reinitialize_1st_order_time_2nd_order_space(phi_eff.vec, 100);
//                ls.reinitialize_1st_order(phi_eff.vec, 50);
                ls.reinitialize_2nd_order(phi_eff.vec, 100);
              }

              vec_and_ptr_t sol(p4est, nodes);
              sample_cf_on_nodes(p4est, nodes, test_cf, sol.vec);

              vec_and_ptr_t sol_ex(p4est, nodes);
              sample_cf_on_nodes(p4est, nodes, test_cf, sol_ex.vec);

              vec_and_ptr_t error_ex(p4est, nodes);

              double band = extension_band_check;

              ls.set_use_two_step_extrapolation(0);
              // extend
              switch (extend_solution)
              {
                case 1:
                  ls.extend_Over_Interface_TVD(phi_eff.vec, sol_ex.vec, extension_iterations, 2, extension_tol, -extension_band_compute*dxyz_max, extension_band_extend*dxyz_max, extension_band_check*dxyz_max, NULL, NULL, NULL, use_nonzero_guess); CHKERRXX(ierr);
                  break;
                case 2:
                  ls.extend_Over_Interface_TVD_Full(phi_eff.vec, sol_ex.vec, extension_iterations, 2, extension_tol, -extension_band_compute*dxyz_max, extension_band_extend*dxyz_max, extension_band_check*dxyz_max, NULL, NULL, NULL, use_nonzero_guess); CHKERRXX(ierr);
                  break;
              }

//              ls.extend_from_interface_to_whole_domain_TVD_1st_order_time(phi_eff.vec, sol.vec, sol_ex.vec, 200);
//              ls.extend_from_interface_to_whole_domain_TVD(phi_eff.vec, sol.vec, sol_ex.vec, 200);

              vec_and_ptr_dim_t sol_ex_d(p4est, nodes);
              vec_and_ptr_array_t sol_ex_dd(3*(P4EST_DIM-1), p4est, nodes);

//              ls.extend_Over_Interface_TVD_Full(phi_eff.vec, sol_ex.vec, extension_iterations, 2,
//                                                extension_tol, -extension_band_compute*dxyz_max, (extension_band_extend+4)*dxyz_max, (extension_band_check+4)*dxyz_max,
//                                                NULL, NULL, NULL,
//                                                0, sol_ex_d.vec, sol_ex_dd.vec.data()); CHKERRXX(ierr);

//              ls.extend_Over_Interface_TVD_Full(phi_eff.vec, sol_ex.vec, 0, 2,
//                                                extension_tol, -extension_band_compute*dxyz_max, extension_band_extend*dxyz_max, extension_band_check*dxyz_max,
//                                                NULL, NULL, NULL,
//                                                1, sol_ex_d.vec, sol_ex_dd.vec.data()); CHKERRXX(ierr);

//              ls.extend_Over_Interface_TVD_Full(phi_eff.vec, sol_ex.vec, extension_iterations, 2,
//                                                extension_tol, -extension_band_compute*dxyz_max, extension_band_extend*dxyz_max, extension_band_check*dxyz_max,
//                                                NULL, NULL, NULL,
//                                                0, sol_ex_d.vec, sol_ex_dd.vec.data()); CHKERRXX(ierr);

              sol_ex_d.destroy();
              sol_ex_dd.destroy();

              // calculate error
              error_ex.get_array();
              phi_eff .get_array();
              sol_ex  .get_array();

              foreach_local_node(n, nodes)
              {
                double xyz[P4EST_DIM];
                node_xyz_fr_n(n, p4est, nodes, xyz);

                error_ex.ptr[n] = (phi_eff.ptr[n] > 0. && phi_eff.ptr[n] < band*dxyz_max) ? ABS(sol_ex.ptr[n] - test_cf.value(xyz)) : 0;
              }
              error_ex.restore_array();
              phi_eff .restore_array();
              sol_ex  .restore_array();

              // compute L-inf norm of errors
              double err_ex_max = 0.;
              ierr = VecMax(error_ex.vec, NULL, &err_ex_max); CHKERRXX(ierr);
              mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
              error_ex_arr.push_back(err_ex_max);

              // Print current errors
              if (iter == 0 )
              {
                ierr = PetscPrintf(p4est->mpicomm, "Error = %3.2e (na)\n", err_ex_max); CHKERRXX(ierr);
              } else {
                ierr = PetscPrintf(p4est->mpicomm, "Error = %3.2e (%+3.2f)\n", err_ex_max, log(error_ex_arr[iter-1]/error_ex_arr[iter])/log(2)); CHKERRXX(ierr);
              }

              if(save_vtk)
              {
                char *out_dir;
                out_dir = getenv("OUT_DIR");

                std::ostringstream oss;

                oss << out_dir
                    << "/vtu/nodes_"
                    << p4est->mpisize << "_"
                    << brick.nxyztrees[0] << "x"
                    << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
                       "x" << brick.nxyztrees[2] <<
       #endif
                       "." << iter;

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

                sol     .get_array();
                sol_ex  .get_array();
                phi_eff .get_array();
                error_ex.get_array();


                my_p4est_vtk_write_all(p4est, nodes, ghost,
                                       P4EST_TRUE, P4EST_TRUE,
                                       4, 1, oss.str().c_str(),
                                       VTK_POINT_DATA, "phi", phi_eff.ptr,
                                       VTK_POINT_DATA, "sol", sol.ptr,
                                       VTK_POINT_DATA, "sol_ex", sol_ex.ptr,
                                       VTK_POINT_DATA, "error_ex", error_ex.ptr,
                                       VTK_CELL_DATA , "leaf_level", l_p);

                sol     .restore_array();
                sol_ex  .restore_array();
                phi_eff .restore_array();
                error_ex.restore_array();


                ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
                ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

                PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
              }

              sol.destroy();
              sol_ex.destroy();
              phi_eff.destroy();
              error_ex.destroy();


              for (unsigned int i = 0; i < bdry_phi_max_num; i++) { if (*bdry_present_all[i] == true) { ierr = VecDestroy(bdry_phi_vec_all[i]); CHKERRXX(ierr); } }

              p4est_nodes_destroy(nodes);
              p4est_ghost_destroy(ghost);
              p4est_destroy      (p4est);
              my_p4est_brick_destroy(connectivity, &brick);

            }
          }
#ifdef P4_TO_P8
        }
#endif
      }
    }


  MPI_Barrier(mpi.comm());

  std::vector<double> error_ex_one(num_resolutions, 0), error_ex_avg(num_resolutions, 0), error_ex_max(num_resolutions, 0);

  // for each resolution compute max, mean and deviation
  for (int p = 0; p < num_resolutions; ++p)
  {
    // one
    error_ex_one[p] = error_ex_arr[p*num_shifts_total];

    // max
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_ex_max[p] = MAX(error_ex_max[p], error_ex_arr[p*num_shifts_total + s]);
    }

    // avg
    for (int s = 0; s < num_shifts_total; ++s)
    {
      error_ex_avg[p] += error_ex_arr[p*num_shifts_total + s];
    }
    error_ex_avg[p] /= num_shifts_total;
  }

  if (mpi.rank() == 0)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    std::string filename;

    // save level and resolution
    filename = out_dir; filename += "/convergence/lvl.txt";   save_vector(filename.c_str(), lvl_arr);
    filename = out_dir; filename += "/convergence/h_arr.txt"; save_vector(filename.c_str(), h_arr);

    filename = out_dir; filename += "/convergence/error_ex_all.txt"; save_vector(filename.c_str(), error_ex_arr);
    filename = out_dir; filename += "/convergence/error_ex_one.txt"; save_vector(filename.c_str(), error_ex_one);
    filename = out_dir; filename += "/convergence/error_ex_avg.txt"; save_vector(filename.c_str(), error_ex_avg);
    filename = out_dir; filename += "/convergence/error_ex_max.txt"; save_vector(filename.c_str(), error_ex_max);
  }

  w.stop(); w.read_duration();

  return 0;
}

double compute_convergence_order(std::vector<double> &x, std::vector<double> &y)
{
  if (x.size() != y.size())
  {
    std::cout << "[ERROR]: sizes of arrays do not coincide\n";
    return 0;
  }

  int n = x.size();

  double sumX  = 0;
  double sumY  = 0;
  double sumXY = 0;
  double sumXX = 0;

  for (int i = 0; i < n; ++i)
  {
    double logX = log(x[i]);
    double logY = log(y[i]);

    sumX  += logX;
    sumY  += logY;
    sumXY += logX*logY;
    sumXX += logX*logX;
  }

  return (sumXY - sumX*sumY/n)/(sumXX - sumX*sumX/n);
}
