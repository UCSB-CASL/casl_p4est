#include "my_p4est_biomolecules.h"

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
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_save_load.h>
#include <src/my_p8est_general_poisson_nodes_mls_solver.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
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
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_save_load.h>
#include <src/my_p4est_general_poisson_nodes_mls_solver.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#endif

#include <fstream>
#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <src/matrix.h>
#include <algorithm>
#include <sys/stat.h>
#include <src/Parser.h>
#include <src/parameter_list.h>
//#include <boost/algorithm/string.hpp>

using namespace std;

const int bdry_phi_max_num = 1;
const int infc_phi_max_num = 1;

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
DEFINE_PARAMETER(pl, int, lmax, 7, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           4, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#else
DEFINE_PARAMETER(pl, int, lmin, 4, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 5, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           5, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#endif

DEFINE_PARAMETER(pl, int, iter_start, 0, "Skip n first iterations");
DEFINE_PARAMETER(pl, double, lip, 2, "Lipschitz constant");

DEFINE_PARAMETER(pl, bool, refine_strict,  1, "Refines every cell starting from the coarsest case if yes");
DEFINE_PARAMETER(pl, bool, refine_rand,    0, "Add randomness into adaptive grid");
DEFINE_PARAMETER(pl, bool, balance_grid,   1, "Enforce 1:2 ratio for adaptive grid");
DEFINE_PARAMETER(pl, bool, coarse_outside, 0, "Use the coarsest possible grid outside the domain (0/1)");
DEFINE_PARAMETER(pl, int,  expand_ghost,   0, "Number of ghost layer expansions");

//-------------------------------------
// test solutions
//-------------------------------------

DEFINE_PARAMETER(pl, int, n_um, 0, "");
DEFINE_PARAMETER(pl, int, n_up, 0, "");

DEFINE_PARAMETER(pl, double, mag_um, 1, "");
DEFINE_PARAMETER(pl, double, mag_up, 1, "");

DEFINE_PARAMETER(pl, int, n_mu_m, 0, "");
DEFINE_PARAMETER(pl, int, n_mu_p, 0, "");

DEFINE_PARAMETER(pl, double, mag_mu_m, 1, "");
DEFINE_PARAMETER(pl, double, mag_mu_p, 1, "");

DEFINE_PARAMETER(pl, double, mu_iter_num, 1, "");
DEFINE_PARAMETER(pl, double, mag_mu_m_min, 1, "");
DEFINE_PARAMETER(pl, double, mag_mu_m_max, 1, "");

DEFINE_PARAMETER(pl, int, n_diag_m, 0, "");
DEFINE_PARAMETER(pl, int, n_diag_p, 0, "");

DEFINE_PARAMETER(pl, double, mag_diag_m, 1, "");
DEFINE_PARAMETER(pl, double, mag_diag_p, 1, "");

DEFINE_PARAMETER(pl, int, bc_wtype, DIRICHLET, "Type of boundary conditions on the walls");

// boundary geometry
DEFINE_PARAMETER(pl, int, bdry_phi_num, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_phi_num, 0, "Domain geometry");

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

DEFINE_PARAMETER(pl, int, bc_coeff_00, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bc_coeff_01, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bc_coeff_02, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, bc_coeff_03, 0, "Domain geometry");

DEFINE_PARAMETER(pl, double, bc_coeff_00_mag, 1, "Domain geometry");
DEFINE_PARAMETER(pl, double, bc_coeff_01_mag, 1, "Domain geometry");
DEFINE_PARAMETER(pl, double, bc_coeff_02_mag, 1, "Domain geometry");
DEFINE_PARAMETER(pl, double, bc_coeff_03_mag, 1, "Domain geometry");

DEFINE_PARAMETER(pl, int, bc_type_00, DIRICHLET, "Type of boundary conditions on the domain boundary");
DEFINE_PARAMETER(pl, int, bc_type_01, DIRICHLET, "Type of boundary conditions on the domain boundary");
DEFINE_PARAMETER(pl, int, bc_type_02, DIRICHLET, "Type of boundary conditions on the domain boundary");
DEFINE_PARAMETER(pl, int, bc_type_03, DIRICHLET, "Type of boundary conditions on the domain boundary");

// interface geometry
DEFINE_PARAMETER(pl, bool, infc_present_00, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, infc_present_01, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, infc_present_02, 0, "Domain geometry");
DEFINE_PARAMETER(pl, bool, infc_present_03, 0, "Domain geometry");

DEFINE_PARAMETER(pl, int, infc_geom_00, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_geom_01, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_geom_02, 0, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_geom_03, 0, "Domain geometry");

DEFINE_PARAMETER(pl, int, infc_opn_00, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_opn_01, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_opn_02, MLS_INTERSECTION, "Domain geometry");
DEFINE_PARAMETER(pl, int, infc_opn_03, MLS_INTERSECTION, "Domain geometry");

DEFINE_PARAMETER(pl, int, jc_value_00, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_value_01, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_value_02, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_value_03, 0, "0 - automatic, others - hardcoded");

DEFINE_PARAMETER(pl, int, jc_flux_00, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_flux_01, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_flux_02, 0, "0 - automatic, others - hardcoded");
DEFINE_PARAMETER(pl, int, jc_flux_03, 0, "0 - automatic, others - hardcoded");

//DEFINE_PARAMETER(pl, int, bc_itype, ROBIN, "");

//-------------------------------------
// solver parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int,  jc_scheme,         0, "Discretization scheme for interface conditions (0 - FVM, 1 - FDM)");
DEFINE_PARAMETER(pl, int,  jc_sub_scheme,     0, "Interpolation subscheme for interface conditions (0 - from slow region, 1 - from fast region, 2 - based on nodes availability)");
DEFINE_PARAMETER(pl, int,  integration_order, 2, "Select integration order (1 - linear, 2 - quadratic)");
DEFINE_PARAMETER(pl, bool, sc_scheme,         1, "Use super-convergent scheme");

// for symmetric scheme:
DEFINE_PARAMETER(pl, bool, taylor_correction,      1, "Use Taylor correction to approximate Robin term (symmetric scheme)");
DEFINE_PARAMETER(pl, bool, kink_special_treatment, 1, "Use the special treatment for kinks (symmetric scheme)");

// for superconvergent scheme:
DEFINE_PARAMETER(pl, bool, try_remove_hanging_cells, 0, "Ask solver to eliminate hanging cells");

DEFINE_PARAMETER(pl, bool, store_finite_volumes,   1, "");
DEFINE_PARAMETER(pl, bool, apply_bc_pointwise,     1, "");
DEFINE_PARAMETER(pl, bool, use_centroid_always,    1, "");
DEFINE_PARAMETER(pl, bool, sample_bc_node_by_node, 0, "");

//-------------------------------------
// level-set representation parameters
//-------------------------------------
DEFINE_PARAMETER(pl, bool, use_phi_cf,       0, "Use analytical level-set functions");
DEFINE_PARAMETER(pl, bool, reinit_level_set, 1, "Reinitialize level-set function");

// artificial perturbation of level-set values
DEFINE_PARAMETER(pl, int,    dom_perturb,     0,   "Artificially pertub level-set functions (0 - no perturbation, 1 - smooth, 2 - noisy)");
DEFINE_PARAMETER(pl, double, dom_perturb_mag, 0.1, "Magnitude of level-set perturbations");
DEFINE_PARAMETER(pl, double, dom_perturb_pow, 2,   "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

DEFINE_PARAMETER(pl, int,    ifc_perturb,     0,   "Artificially pertub level-set functions (0 - no perturbation, 1 - smooth, 2 - noisy)");
DEFINE_PARAMETER(pl, double, ifc_perturb_mag, 0.1e-6, "Magnitude of level-set perturbations");
DEFINE_PARAMETER(pl, double, ifc_perturb_pow, 2,   "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

//-------------------------------------
// convergence study parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int,    compute_cond_num,       0, "Estimate L1-norm condition number");
DEFINE_PARAMETER(pl, int,    extend_solution,        2, "Extend solution after solving: 0 - no extension, 1 - extend using normal derivatives, 2 - extend using all derivatives");
DEFINE_PARAMETER(pl, double, mask_thresh,            0, "Mask threshold for excluding points in convergence study");
DEFINE_PARAMETER(pl, bool,   compute_grad_between,   0, "Computes gradient between points if yes");
DEFINE_PARAMETER(pl, bool,   scale_errors,           0, "Scale errors by max solution/gradient value");
DEFINE_PARAMETER(pl, bool,   use_nonzero_guess,      0, "");
DEFINE_PARAMETER(pl, double, extension_band_extend,  6, "");
DEFINE_PARAMETER(pl, double, extension_band_compute, 6, "");
DEFINE_PARAMETER(pl, double, extension_band_check,   6, "");
DEFINE_PARAMETER(pl, double, extension_tol,          -1.e-10, "");
DEFINE_PARAMETER(pl, int,    extension_iterations,   100, "");


//-------------------------------------
// output
//-------------------------------------
DEFINE_PARAMETER(pl, bool, save_vtk,           1, "Save the p4est in vtk format");
DEFINE_PARAMETER(pl, bool, save_params,        0, "Save list of entered parameters");
DEFINE_PARAMETER(pl, bool, save_domain,        0, "Save the reconstruction of an irregular domain (works only in serial!)");
DEFINE_PARAMETER(pl, bool, save_matrix_ascii,  0, "Save the matrix in ASCII MATLAB format");
DEFINE_PARAMETER(pl, bool, save_matrix_binary, 0, "Save the matrix in BINARY MATLAB format");
DEFINE_PARAMETER(pl, bool, save_convergence,   0, "Save convergence results");

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
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform);
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

bdry_phi_cf_t bdry_phi_cf_all  [] = { bdry_phi_cf_t(VAL, bdry_geom_00) };

bdry_phi_cf_t bdry_phi_x_cf_all[] = { bdry_phi_cf_t(DDX, bdry_geom_00) };

bdry_phi_cf_t bdry_phi_y_cf_all[] = { bdry_phi_cf_t(DDY, bdry_geom_00) };
#ifdef P4_TO_P8
bdry_phi_cf_t bdry_phi_z_cf_all[] = { bdry_phi_cf_t(DDZ, bdry_geom_00) };
#endif
// initialize all static variables
const string Atom::ATOM = "ATOM  ";
p4est_connectivity_t* my_p4est_biomolecules_t::molecule::domain_connectivity = NULL;
mpi_environment_t* my_p4est_biomolecules_t::molecule::mpi = NULL;
const int my_p4est_biomolecules_t::nangle_per_mol =
    #ifdef P4_TO_P8
    3
    #else
    1
    #endif
    ;
FILE* my_p4est_biomolecules_t::log_file     = NULL;
FILE* my_p4est_biomolecules_t::timing_file  = NULL;
FILE* my_p4est_biomolecules_t::error_file   = NULL;
int reduced_list::nb_reduced_lists          = 0;

double norm(const vector<double>& v)
{
  double value = 0.0;
  for (size_t k = 0; k < v.size(); ++k)
    value += SQR(v.at(k));
  return sqrt(value);
}

void my_p4est_biomolecules_t::molecule::read(const string& pqr, const int overlap)
{
#ifdef CASL_THROWS
  int                 my_local_error;
  string              err_msg;
#endif
  int mpiret;
  if (atoms.size() != 0)
  {
    PetscPrintf(mpi->comm(), "------------------------------------------------ \n");
    PetscPrintf(mpi->comm(), "---  Reinitialization of the list of atoms  ---- \n");
    PetscPrintf(mpi->comm(), "------------------------------------------------ \n");
    // reinitialize the list of atoms and related parameters
    atoms.clear();
  }
#ifdef CASL_THROWS
  my_local_error = overlap < 1;
  err_msg = "bio_molecule::read(const string*, const int): require an integer >= 1 for the overlap argument. overlap <= 0 does not make sense. \n";
  mol_err_manager.check_my_local_error(my_local_error, err_msg);
#endif

  string bundle = "bundle";
  string extension = ".pqr";
  int pqr_length = pqr.size();
  bool is_a_bundle = (pqr_length >= 6 && !bundle.compare(pqr.substr(pqr_length-6, string::npos)));
  bool add_extension = !(pqr_length>4 && !extension.compare(pqr.substr(pqr_length-4, string::npos)));
  int file_idx = 1;
  string filename = pqr + ((is_a_bundle)? to_string(file_idx):"") + ((add_extension)? extension:"");
  MPI_File file_handle; // do not use a pointer, allocation needed...

  // initialize the centroid
  for (int k = 0; k < P4EST_DIM; ++k)
    molecule_centroid[k] = 0.0;
  // initialize the largest radius
  largest_radius = -DBL_MAX;
  // vector of atoms read by this proc
  vector<Atom> atoms_in_chunk; atoms_in_chunk.clear();
  // number of charged atoms read by this proc
  n_charged_atoms = 0;

  while ((is_a_bundle || file_idx == 1) && check_if_file(filename)){
    // copy the path to the file (MPI_File_open requires a non-constant char*... :-B)
    char file_name[filename.size()+1];
    filename.copy(file_name, filename.size(), 0);
    file_name[filename.size()] = '\0';

    mpiret = MPI_File_open(mpi->comm(), &file_name[0], MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
#ifdef CASL_THROWS
    my_local_error = mpiret != sc_MPI_SUCCESS;
    err_msg = "bio_molecule::read(const string*, const int): could not open file" + filename + "\n";
    mol_err_manager.check_my_local_error(my_local_error, err_msg);
#endif

    /* read relevant chunk of file, which starts at location
     * globalstart in the file and has size mysize
     */

    /* figure out who reads what */
    MPI_Offset filesize;
    MPI_File_get_size(file_handle, &filesize);
    filesize--;  /* get rid of eof character */
    string chunk;
    int mysize = filesize/mpi->size();
    int rank_max = mpi->size()-1;
    if (mysize < overlap)
    {
      /* too many procs for the number of atoms in the file if this occurs.
     * It's possible that no relevant line is fully considered by any process.
     * This would result in garbage information for the atoms
     */
      if (error_file != NULL)
      {
        string warning_message = "bio_molecule::read(const string*, const int): !WARNING! the file is too small for the number of processes \nbio_molecule::read(const string*, const int): !WARNING! fewer procs will read chunks \n";
        PetscFPrintf(mpi->comm(), error_file, warning_message.c_str());
      }
      /* We'll use less procs, the last ones will do nothing but wait for the final Allgather*/
#ifdef CASL_THROWS
      my_local_error = filesize/overlap>mpi->size();
      err_msg = "bio_molecule::read(const string*, const int): something went wrong, a logic error occurred, inconsistent integer divisions...\n";
      mol_err_manager.check_my_local_error(my_local_error, err_msg);
      my_local_error = filesize/overlap == 0;
      err_msg = "bio_molecule::read(const string*, const int): the file is way too small, doesn't even contain one relevant line...\n";
      mol_err_manager.check_my_local_error(my_local_error, err_msg);
#endif
      rank_max = filesize/overlap;
      mysize = overlap;
      MPI_Offset globalstart;
      if(mpi->rank() <= rank_max)
        globalstart = mysize*mpi->rank();
      else
        globalstart = filesize-1;
      MPI_Offset globalend    = (mpi->rank() < rank_max)?globalstart + mysize:filesize;
      /* add overlap to the end of everyone's chunk except last proc... */
      if (mpi->rank() < rank_max)
        globalend += overlap;
      mysize =  globalend - globalstart;
      /* allocate memory */
      chunk.resize(mysize + 1);
      /* everyone reads in their part */
      MPI_File_read_at_all(file_handle, globalstart, &chunk[0], mysize, MPI_CHAR, MPI_STATUS_IGNORE);
      chunk[mysize] = '\0';
    }
    else
    {
      MPI_Offset globalstart  = mysize*mpi->rank();
      MPI_Offset globalend    = (mpi->rank() != rank_max)?globalstart + mysize:filesize;
      /* add overlap to the end of everyone's chunk except last proc... */
      if (mpi->rank() != rank_max)
        globalend += overlap;
      mysize =  globalend - globalstart;
      /* allocate memory */
      chunk.resize(mysize + 1);
      /* everyone reads in their part */
      MPI_File_read_at_all(file_handle, globalstart, &chunk[0], mysize, MPI_CHAR, MPI_STATUS_IGNORE);
      chunk[mysize] = '\0';
    }

    /*
     * everyone calculates what their start and end *really* are by going
     * from the first newline after start to the end of the overlapped line
     * (after end - overlap)
     */
    int locstart=0, locend=mysize;
    if (mpi->rank() != 0 && mpi->rank() <= rank_max) { /* second condition needed if using fewer procs than available, useless otherwise.*/
      while(chunk[locstart] != '\n' && locstart < locend)
        locstart++;
      locstart++; // skip the found '\n' character
    }
    if (mpi->rank() < rank_max) {
      locend-=overlap;
      while(chunk[locend] != '\n' && locend < mysize) locend++;
    }

    /* Process our chunk line by line */
    string line;
    int i = locstart;
    int line_size;
    while (i <= locend)
    {
      line_size = 0;
      while(chunk[i+line_size] != '\n' && (i+line_size) < locend){line_size++;}
      line = chunk.substr(i, line_size);

      Atom atom;
      if(line >> atom)
      {
        atoms_in_chunk.push_back(atom);
        molecule_centroid[0] += atom.x;
        molecule_centroid[1] += atom.y;
#ifdef P4_TO_P8
        molecule_centroid[2] += atom.z;
#endif
        largest_radius = MAX(largest_radius, atom.r);
        n_charged_atoms += (fabs(atom.q) > 0.00005)? 1:0; // the charge resolution is 0.0001 in regular pqr files
      }
      i += line_size+1; // +1 to skip the '\n' character and/or avoid unterminated loop because of trailing '\n'
    }
    MPI_File_close(&file_handle);

    file_idx++;
    filename = pqr + ((is_a_bundle)? to_string(file_idx):"") + ((add_extension)? extension:"");
  }

  vector<int> byte_offset_in_proc(mpi->size());
  vector<int> nb_of_bytes_in_proc(mpi->size());
  nb_of_bytes_in_proc[mpi->rank()] = atoms_in_chunk.size()*sizeof(Atom);
  mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &nb_of_bytes_in_proc[0], 1, MPI_INT, mpi->comm()); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &molecule_centroid[0], P4EST_DIM, MPI_DOUBLE, MPI_SUM, mpi->comm()); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &n_charged_atoms, 1, MPI_INT, MPI_SUM, mpi->comm()); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &largest_radius, 1, MPI_DOUBLE, MPI_MAX, mpi->comm()); SC_CHECK_MPI(mpiret);
  int total_nb_of_atoms = nb_of_bytes_in_proc[0]/sizeof(Atom);
  byte_offset_in_proc[0] = 0;
  for (int k = 1; k < mpi->size(); ++k) {
    total_nb_of_atoms       += nb_of_bytes_in_proc[k]/sizeof(Atom);
    byte_offset_in_proc[k]  = byte_offset_in_proc[k-1] + nb_of_bytes_in_proc[k-1];
  }
  for (int k = 0; k < P4EST_DIM; ++k) {
    molecule_centroid[k] /= total_nb_of_atoms;
  }

  atoms.resize(total_nb_of_atoms);
  mpiret = MPI_Allgatherv(&atoms_in_chunk[0],
      atoms_in_chunk.size()*sizeof(Atom),
      MPI_BYTE,
      &atoms[0],
      &nb_of_bytes_in_proc[0],
      &byte_offset_in_proc[0],
      MPI_BYTE,
      mpi->comm()); SC_CHECK_MPI(mpiret);

  return;
}
/*
 * if someone ever plans on using this one again: it needs additional
 * features (see the parallel version) to be compatible with other class
 * methods: read bundles, count the number of charged atoms, etc.
 *
void my_p4est_biomolecules_t::molecule::read_serial(const string& pqr)
{
  if (atoms.size() != 0)
  {
    PetscPrintf(mpi->comm(), "------------------------------------------------ \n");
    PetscPrintf(mpi->comm(), "---  Reinitialization of the list of atoms  ---- \n");
    PetscPrintf(mpi->comm(), "------------------------------------------------ \n");
    // reinitialize the list of atoms and related parameters
    atoms.clear();
  }

  // only read on rank 0 and then broadcast the result to others
  if (mpi->rank()== 0) {

    ifstream reader(pqr.c_str());
#ifdef CASL_THROWS // throwing is ok, root proc only...
    if (!reader)
    {
      string err_msg = "bio_molecule::read_serial(const string*): could not open file" + pqr + "\n";
      mol_err_manager.print_message_and_abort(err_msg, 147741);
    }
#else
    MPI_Abort(mpi->comm(), 147741);
#endif

    for (int k = 0; k < P4EST_DIM; ++k) {
      molecule_centroid[k] = 0.0;
    }
    largest_radius = -DBL_MAX;
    // parse line by line
    string line;
    while(getline(reader, line)) {
      istringstream iss(line);
      string keyword; iss >> keyword;
      Atom atom;
      if (keyword == "ATOM") {
        iss >> atom;
        atom.x *= angstrom_to_domain;
        molecule_centroid[0] += atom.x;
        atom.y *= angstrom_to_domain;
        molecule_centroid[1] += atom.y;
  #ifdef P4_TO_P8
        atom.z *= angstrom_to_domain;
        molecule_centroid[2] += atom.z;
  #endif
        atom.r *= angstrom_to_domain;
        largest_radius = MAX(largest_radius, atom.r);
        atoms.push_back(atom);
      }
    }
  }
  for (int k = 0; k < P4EST_DIM; ++k) {
    molecule_centroid[k] /= atoms.size();
  }
  MPI_Bcast(&molecule_centroid[0], 3, MPI_DOUBLE, 0, mpi->comm());
  MPI_Bcast(&largest_radius, 1, MPI_DOUBLE, 0, mpi->comm());

  size_t msg_size = atoms.size()*sizeof(Atom);
  MPI_Bcast(&msg_size, 1, MPI_UNSIGNED_LONG, 0, mpi->comm());
  if (mpi->rank() != 0)
    atoms.resize(msg_size/sizeof(Atom));
  MPI_Bcast(&atoms[0], msg_size, MPI_BYTE, 0, mpi->comm());
}
*/
void my_p4est_biomolecules_t::molecule::calculate_center_of_domain(double* domain_center) const
{
  double *vertices_to_coordinates = domain_connectivity->vertices;
  p4est_topidx_t *tree_to_vertex  = domain_connectivity->tree_to_vertex;

  double xmin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 0];
  double xmax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 0];
  domain_center[0] = 0.5*(xmin + xmax);
  double ymin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 1];
  double ymax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 1];
  domain_center[1] = 0.5*(ymin + ymax);
#ifdef P4_TO_P8
  double zmin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 2];
  double zmax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 2];
  domain_center[2] = 0.5*(zmin + zmax);
#endif
}
int my_p4est_biomolecules_t::molecule::update_connectivity(p4est_connectivity_t* conn_)
{
  if(domain_connectivity == NULL)
    domain_connectivity = conn_;
  return domain_connectivity != conn_;
}
int my_p4est_biomolecules_t::molecule::update_mpi(mpi_environment_t* mpi_)
{
  if(mpi == NULL)
    mpi = mpi_;
  return mpi != mpi_;
}
my_p4est_biomolecules_t::molecule::molecule(const string& pqr_, const double* xyz_c, double* angles, const double angstrom_to_domain_, const int overlap)
  :
    #ifdef CASL_THROWS
    mol_err_manager(mpi->rank(), mpi->size(), mpi->comm(), my_p4est_biomolecules_t::error_file),
    #endif
    angstrom_to_domain(angstrom_to_domain_)
{
  scale_is_set = (fabs(angstrom_to_domain_-1.0)>EPS);
  read(pqr_, overlap);
#ifndef P4_TO_P8
  // remove possible duplicates (one of the coordinates has been disregarded for each atom when reading the list, there might be duplicates)
  sort(atoms.begin(), atoms.end());
  atoms.erase(unique(atoms.begin(), atoms.end()), atoms.end());
#endif
  index_of_charged_atom.clear();
  scale_rotate_and_translate(NULL /* conversion to domain dim has been done at reading stage */
                             , xyz_c
                             , angles);
}
double my_p4est_biomolecules_t::molecule::calculate_scaling_factor(const double cube_side_length_to_min_domain_size) const
{
#ifdef CASL_THROWS
  int local_error = (cube_side_length_to_min_domain_size < EPS || cube_side_length_to_min_domain_size >= 1.);
  string err_msg = "my_p4est_biomolecules_t::molecule::calculate_scaling_factor(const double) requires a value in )0, 1( as an input.";
  mol_err_manager.check_my_local_error(local_error, err_msg);
#endif
  /* calculate the minimal domain dimension */
  double *vertices_to_coordinates = domain_connectivity->vertices;
  p4est_topidx_t *tree_to_vertex  = domain_connectivity->tree_to_vertex;
  double xmin           = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 0];
  double xmax           = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 0];
  double domain_dim_min = xmax - xmin;
  double ymin           = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 1];
  double ymax           = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 1];
  domain_dim_min        = MIN(domain_dim_min, ymax-ymin);
#ifdef P4_TO_P8
  double zmin           = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 2];
  double zmax           = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 2];
  domain_dim_min        = MIN(domain_dim_min, zmax - zmin);
#endif
  // calculate the conversion factor
  double domain_to_angstrom = 1.0/angstrom_to_domain;
  double side_length_of_bounding_cube_angstrom = side_length_of_bounding_cube*domain_to_angstrom;
  return cube_side_length_to_min_domain_size*domain_dim_min/side_length_of_bounding_cube_angstrom; // new angstrom_to_domain factor
  // In real physical space, this value is in [1/ansgtrom] and is equal
  // to the inverse of the distance that becomes "1.0" in the domain
}
void my_p4est_biomolecules_t::molecule::scale_rotate_and_translate(const double* angstrom_to_domain_, const double* xyz_c, double* angles)
{
#ifdef CASL_THROWS
  int local_error = (angstrom_to_domain_!= NULL && *angstrom_to_domain_ <= 0);
  string err_msg = "my_p4est_biomolecules_t::molecule::scale_rotate_and_translate(const double*, ...): the first argument must point to a strictly positive value, if non NULL.";
  mol_err_manager.check_my_local_error(local_error, err_msg);
#endif
  if(!scale_is_set && angstrom_to_domain_!= NULL)
    scale_is_set = true;
  const double domain_to_angstrom = 1.0/angstrom_to_domain; // based on the former value, --> back to angstrom by multiplying by domain_to_angstrom
  angstrom_to_domain = (angstrom_to_domain_!= NULL)? *angstrom_to_domain_: angstrom_to_domain; // new value
  const double multiplicator = domain_to_angstrom*angstrom_to_domain; // appropriate scaling factor from previous domain scale to new domain scale

  vector<double>xyz_tmp; xyz_tmp.resize(P4EST_DIM);
  vector<double> rotated_xyz_tmp;
  matrix_t rotation_matrix;
  if(angles != NULL)
  {
    rotated_xyz_tmp.resize(P4EST_DIM);
    rotation_matrix.resize(P4EST_DIM, P4EST_DIM);
    angles[0] = fmod(angles[0], 2*PI); if(angles[0] < 0) angles[0] += 2*PI;
#ifdef P4_TO_P8
    angles[2] = fmod(angles[2], 2*PI); if(angles[2] < 0) angles[2] += 2*PI;
    angles[1] = fmod(angles[1], PI);
    if(angles[1] < 0)
    {
      angles[2] = fmod(angles[2]+PI, 2*PI); // reverse the azimuthal angle
      angles[1] *= -1;
    }
    double rotation_axis[P4EST_DIM];
    rotation_axis[0] = sin(angles[1])*cos(angles[2]);
    rotation_axis[1] = sin(angles[1])*sin(angles[2]);
    rotation_axis[2] = cos(angles[1]);
    // Rodrigues' rotation formula
    rotation_matrix.set_value(0, 0, (1-cos(angles[0]))*rotation_axis[0]*rotation_axis[0] + cos(angles[0]));
    rotation_matrix.set_value(0, 1, (1-cos(angles[0]))*rotation_axis[0]*rotation_axis[1] - sin(angles[0])*rotation_axis[2]);
    rotation_matrix.set_value(0, 2, (1-cos(angles[0]))*rotation_axis[0]*rotation_axis[2] + sin(angles[0])*rotation_axis[1]);
    rotation_matrix.set_value(1, 0, (1-cos(angles[0]))*rotation_axis[0]*rotation_axis[1] + sin(angles[0])*rotation_axis[2]);
    rotation_matrix.set_value(1, 1, (1-cos(angles[0]))*rotation_axis[1]*rotation_axis[1] + cos(angles[0]));
    rotation_matrix.set_value(1, 2, (1-cos(angles[0]))*rotation_axis[1]*rotation_axis[2] - sin(angles[0])*rotation_axis[0]);
    rotation_matrix.set_value(2, 0, (1-cos(angles[0]))*rotation_axis[0]*rotation_axis[2] - sin(angles[0])*rotation_axis[1]);
    rotation_matrix.set_value(2, 1, (1-cos(angles[0]))*rotation_axis[1]*rotation_axis[2] + sin(angles[0])*rotation_axis[0]);
    rotation_matrix.set_value(2, 2, (1-cos(angles[0]))*rotation_axis[2]*rotation_axis[2] + cos(angles[0]));
#else
    rotation_matrix.set_value(0, 0,  cos(angles[0]));
    rotation_matrix.set_value(0, 1, -sin(angles[0]));
    rotation_matrix.set_value(1, 0,  sin(angles[0]));
    rotation_matrix.set_value(1, 1,  cos(angles[0]));
#endif
  }

  side_length_of_bounding_cube = -DBL_MAX;
  double new_centroid[P4EST_DIM];
  for (int k = 0; k < P4EST_DIM; ++k) {
    new_centroid[k] = (xyz_c == NULL)? multiplicator*molecule_centroid[k]:xyz_c[k];
  }
  const bool create_list_of_charged_atoms = index_of_charged_atom.size() == 0;
  int charged_atoms_found = 0;
  if(create_list_of_charged_atoms)
    index_of_charged_atom.resize(n_charged_atoms);
  for (size_t k = 0; k < atoms.size(); ++k) {
    if(create_list_of_charged_atoms && fabs(atoms[k].q) > 0.00005)
    {
      index_of_charged_atom.at(charged_atoms_found++) = k;
    }
    xyz_tmp[0] = multiplicator*(atoms[k].x - molecule_centroid[0]);
    xyz_tmp[1] = multiplicator*(atoms[k].y - molecule_centroid[1]);
#ifdef P4_TO_P8
    xyz_tmp[2] = multiplicator*(atoms[k].z - molecule_centroid[2]);
#endif
    if(angles != NULL)
      rotation_matrix.matvec(xyz_tmp, rotated_xyz_tmp);
    else
      rotated_xyz_tmp = xyz_tmp;
    atoms[k].r                   *= multiplicator; // don't forget to scale the radius
    atoms[k].x                    = new_centroid[0] + rotated_xyz_tmp[0];
    side_length_of_bounding_cube  = MAX(side_length_of_bounding_cube, fabs(atoms[k].x - new_centroid[0])+atoms[k].r);
    atoms[k].y                    = new_centroid[1] + rotated_xyz_tmp[1];
    side_length_of_bounding_cube  = MAX(side_length_of_bounding_cube, fabs(atoms[k].y - new_centroid[1])+atoms[k].r);
#ifdef P4_TO_P8
    atoms[k].z                    = new_centroid[2] + rotated_xyz_tmp[2];
    side_length_of_bounding_cube  = MAX(side_length_of_bounding_cube, fabs(atoms[k].z - new_centroid[2])+atoms[k].r);
#endif
  }
  if(create_list_of_charged_atoms)
    P4EST_ASSERT(charged_atoms_found == n_charged_atoms);
  // it's the max distance from the centroid of the molecule to the cube faces so far, hence half the side length
  // --> multiply by 2
  side_length_of_bounding_cube   *= 2.;
  for (int k = 0; k < P4EST_DIM; ++k) {
    molecule_centroid[k] = new_centroid[k];
  }
  largest_radius *= multiplicator;
#ifdef CASL_THROWS
  local_error = (scale_is_set && !is_bounding_box_in_domain());
  err_msg = "my_p4est_biomolecules_t::molecule::scale_rotate_and_translate(...): the bounding box of the molecule ends up out of the computational domain.";
  mol_err_manager.check_my_local_error(local_error, err_msg);
#endif
}
void my_p4est_biomolecules_t::molecule::translate()
{
  double domain_center[P4EST_DIM];
  calculate_center_of_domain(&domain_center[0]);
  translate(domain_center);
}
void my_p4est_biomolecules_t::molecule::translate(const double *xyz_c)
{
  scale_rotate_and_translate(NULL, xyz_c, NULL);
}
void my_p4est_biomolecules_t::molecule::rotate(double* angles, const double *xyz_c)
{
  scale_rotate_and_translate(NULL, xyz_c, angles);
}
void my_p4est_biomolecules_t::molecule::scale_and_translate(const double* angstrom_to_domain_, const double* xyz_c)
{
  scale_rotate_and_translate(angstrom_to_domain_, xyz_c, NULL);
}
void my_p4est_biomolecules_t::molecule::reduce_to_single_atom()
{
  std::vector<Atom> atoms_new;
  atoms_new.push_back(atoms[0]);
  atoms = atoms_new;
  n_charged_atoms = (fabs(atoms.at(0).q) > 0.00005)? 1 : 0;
  index_of_charged_atom.resize(n_charged_atoms);
  if(n_charged_atoms)
    index_of_charged_atom.at(0) = 0;

  molecule_centroid[0] = atoms.at(0).x;
  molecule_centroid[1] = atoms.at(0).y;
#ifdef P4_TO_P8
  molecule_centroid[2] = atoms.at(0).z;
#endif
  side_length_of_bounding_cube = 2*atoms.at(0).r;
  largest_radius = atoms.at(0).r;
}
bool my_p4est_biomolecules_t::molecule::is_bounding_box_in_domain(const double* box_c) const
{
  double *vertices_to_coordinates = domain_connectivity->vertices;
  p4est_topidx_t *tree_to_vertex  = domain_connectivity->tree_to_vertex;
  if(!is_periodic(domain_connectivity, 0))
  {
    double xmin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 0];
    double xmax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 0];
    if(((box_c != NULL)? box_c[0]:molecule_centroid[0])+0.5*side_length_of_bounding_cube > xmax || ((box_c != NULL)? box_c[0]:molecule_centroid[0])-0.5*side_length_of_bounding_cube < xmin)
      return false;
  }
  if(!is_periodic(domain_connectivity, 1))
  {
    double ymin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 1];
    double ymax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 1];
    if(((box_c != NULL)? box_c[1]:molecule_centroid[1])+0.5*side_length_of_bounding_cube > ymax || ((box_c != NULL)? box_c[1]:molecule_centroid[1])-0.5*side_length_of_bounding_cube < ymin)
      return false;
  }
#ifdef P4_TO_P8
  if(!is_periodic(domain_connectivity, 2))
  {
    double zmin = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 2];
    double zmax = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(domain_connectivity->num_trees-1) + P4EST_CHILDREN-1] + 2];
    if(((box_c != NULL)? box_c[2]:molecule_centroid[2])+0.5*side_length_of_bounding_cube > zmax || ((box_c != NULL)? box_c[2]:molecule_centroid[2])-0.5*side_length_of_bounding_cube < zmin)
      return false;
  }
#endif
  return true;
}
double my_p4est_biomolecules_t::molecule::operator ()(const double x, const double y
                                                      #ifdef P4_TO_P8
                                                      , const double z
                                                      #endif
                                                      )const {
  double phi = -DBL_MAX;
  for (size_t m = 0; m < atoms.size(); m++) {
    const Atom& a = atoms[m];
    phi = MAX(phi, a.dist_to_vdW_surface(x, y
                                     #ifdef P4_TO_P8
                                         , z
                                     #endif
                                         ));
  }
  return phi;
}

p4est_bool_t my_p4est_biomolecules_t::SAS_creator::refine_for_exact_calculation_fn(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  // Let's not enforce the min level when we build the SAS grid because that might lead to irrelevant
  // handling and/or communications of kinda big reduced lists of atoms for the sole purpose of refining
  // the grid up to a desired minimum level --> does not scale well! (does not scale AT ALL, actually)
  //  int min_lvl = biomol->parameters.min_level();
  int max_lvl = biomol->parameters.max_level();
  (void) which_tree;
  // see the comment above
  /*if (quad->level < min_lvl)
    return P4EST_TRUE;
  else */
  if (quad->level >= max_lvl)
    return P4EST_FALSE;
  else
  {
    const double cell_diag        = biomol->parameters.root_diag()/(1<<quad->level);
    const double rp               = biomol->parameters.probe_radius();
    const double L                = biomol->parameters.lip_cst();
    const p4est_nodes_t* nodes    = biomol->nodes;
    // one MUST use the 'long' user-defined integer since p4est_locidx_t is an alias for int32_t
    // only long format ensures 32 bits, int ensures 16 bits only
    p4est_locidx_t former_quad_idx = (park->mpisize>1)?(quad->p.user_long & (biomol->max_quad_loc_idx - 1)) : quad->p.user_long; // bitwise filtering

    double f[P4EST_CHILDREN];
    for (unsigned short k = 0; k < P4EST_CHILDREN; ++k) {
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx] - rp;
      if(fabs(f[k]) < 0.5*L*cell_diag || f[k] < -biomol->parameters.root_diag())
        return P4EST_TRUE;
    }
#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_TRUE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_TRUE;
#endif
    return P4EST_FALSE;
  }
}

p4est_bool_t my_p4est_biomolecules_t::SAS_creator::refine_for_reinitialization_fn(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  // Let's not enforce the min level when we build the SAS grid because that might lead to irrelevant
  // handling and/or communications of kinda big reduced lists of atoms for the sole purpose of refining
  // the grid up to a desired minimum level --> does not scale well! (does not scale AT ALL, actually)
  //  int min_lvl = biomol->parameters.min_level();
  int max_lvl = biomol->parameters.max_level();
  (void) which_tree;
  // see the comment above
  /*if (quad->level < min_lvl)
    return P4EST_TRUE;
  else */
  if (quad->level >= max_lvl)
    return P4EST_FALSE;
  else
  {
    const double cell_diag        = biomol->parameters.root_diag()/(1<<quad->level);
    const double layer_thickness  = biomol->parameters.layer_thickness();
    const double rp               = biomol->parameters.probe_radius();
    const double L                = biomol->parameters.lip_cst();
    const p4est_nodes_t* nodes    = biomol->nodes;
    // one MUST use the 'long' user-defined integer since p4est_locidx_t is an alias for int32_t
    // only long format ensures 32 bits, int ensures 16 bits only
    p4est_locidx_t former_quad_idx = (park->mpisize>1)?(quad->p.user_long & (biomol->max_quad_loc_idx - 1)) : quad->p.user_long; // bitwise filtering

    double f[P4EST_CHILDREN];
    for (unsigned short k = 0; k < P4EST_CHILDREN; ++k) {
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx];
      if((-layer_thickness <= f[k]) && (f[k] <= rp))
        return P4EST_TRUE;
      if((0 <= f[k] - rp) && (f[k] - rp <= 0.5*L*cell_diag))
        return P4EST_TRUE;
      if(f[k] < -layer_thickness && ((rp-f[k]) <= 0.5*L*cell_diag))
        return P4EST_TRUE;
    }
#ifdef P4_TO_P8
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
        f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0)
      return P4EST_TRUE;
#else
    if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
      return P4EST_TRUE;
#endif
    return P4EST_FALSE;
  }
}

void my_p4est_biomolecules_t::SAS_creator::scatter_locally(p4est_t*& park)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) park->user_pointer;
  Vec& phi_sas                              = biomol->phi;
  double* & phi_sas_p                       = biomol->phi_p;
  p4est_nodes_t* & nodes                    = biomol->nodes;
#ifdef CASL_THROWS
  const par_error_manager* sas_err_manager  = &biomol->err_manager;
#endif
  // needed for point equality check comparison: boundary points are clamped inside
  // the domain and the Morton codes are compared, two points are equivalent if Morton
  // codes are identical after being clamped
  int clamped = 1;
  if((phi_sas == NULL) != (nodes == NULL))
  {
    // should NEVER happen, no way to handle the problem in such a case, kill the program :-(
#ifdef CASL_THROWS
    string sas_err_msg = "my_p4est_biomolecules_t::SAS_creator_brute_force::scatter_locally(const my_p4est_biomolecules_t*&): something weird occured, you might have memory leak here...";
    sas_err_manager->print_message_and_abort(sas_err_msg, 123321);
#else
    MPI_Abort(mpi_comm, 123321);
#endif
  }
  // ok let's work
  // initialization might be needed first
  if(phi_sas == NULL && phi_sas_p == NULL && nodes == NULL)
  {
    parStopWatch* subsubtimer = NULL;
    if(sub_timer != NULL)
    {
      subsubtimer = new parStopWatch(parStopWatch::all_timings, biomol->timing_file, mpi_comm);
      subsubtimer->start("             SAS_creator::initialization routine");
    }
    initialization_routine(park);
    PetscInt my_index_offset = 0;
    for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
      my_index_offset += nodes->global_owned_indeps[proc_rank];
    global_indices_of_known_values.resize(nodes->num_owned_indeps);
    // scatter locally the values that are already known, count the number of points that need calculations
    // and keep track of the global indices of values to be scattered to the new layout afterwards
    for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k) {
      global_indices_of_known_values.at(k) = my_index_offset + k;
    }

    // we don't need the coarse nodes and corresponding data any more
    p4est_nodes_destroy(nodes); nodes = NULL; // we no longer need the nodes
    if(subsubtimer != NULL)
    {
      subsubtimer->stop();subsubtimer->read_duration();
      delete subsubtimer; subsubtimer = NULL;
    }
    return;
  }

  ierr  = VecGetArray(phi_sas, &phi_sas_p); CHKERRXX(ierr);
  p4est_nodes_t* local_refined_nodes = my_p4est_nodes_new(park, NULL);
  Vec     refined_phi_sas;
  double* refined_phi_sas_p;
  ierr  = VecCreateGhostNodes(park, local_refined_nodes, &refined_phi_sas); CHKERRXX(ierr);
  ierr  = VecGetArray(refined_phi_sas, &refined_phi_sas_p); CHKERRXX(ierr);
  p4est_locidx_t coarse_idx   = 0;
  p4est_indep_t *coarse_node = NULL;
  if(coarse_idx < nodes->num_owned_indeps)
    coarse_node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes, coarse_idx);
  PetscInt my_index_offset = 0;
  for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
    my_index_offset += local_refined_nodes->global_owned_indeps[proc_rank];
  global_indices_of_known_values.resize(nodes->num_owned_indeps);
  // scatter locally the values that are already known, count the number of points that need calculations
  // and keep track of the global indices of values to be scattered to the new layout afterwards
  for (p4est_locidx_t k = 0; k < local_refined_nodes->num_owned_indeps; ++k) {
    p4est_indep_t *fine_node  = (p4est_indep_t*) sc_array_index(&local_refined_nodes->indep_nodes,k);
    if(coarse_node!= NULL && p4est_node_equal_piggy_fn (fine_node, coarse_node, &clamped))
    {
      global_indices_of_known_values.at(coarse_idx) = my_index_offset + k;
      refined_phi_sas_p[k] = phi_sas_p[coarse_idx++];
      if(coarse_idx < nodes->num_owned_indeps)
        coarse_node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes,coarse_idx);
    }
  }
  // sanity check
#ifdef CASL_THROWS
  int sas_local_error = (coarse_idx != nodes->num_owned_indeps);
  string sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::scatter_locally(...) killed because of logic error \n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif
  // we don't need the coarse nodes and corresponding data any more
  ierr = VecRestoreArray(phi_sas, &phi_sas_p); CHKERRXX(ierr);
  ierr = VecDestroy(phi_sas); CHKERRXX(ierr); // we no longer need this, all relevant data have been scattered
  p4est_nodes_destroy(nodes); nodes = NULL; // we no longer need the local (unpartitioned) nodes
  p4est_nodes_destroy(local_refined_nodes); // we no longer need the local_refined_nodes neither
  // restore the (unpartitioned) vector of refined_phi_sas
  phi_sas_p = refined_phi_sas_p; // store it in the class pointer
  phi_sas = refined_phi_sas; // store it in the class pointer
  ierr = VecRestoreArray(phi_sas, &phi_sas_p); phi_sas_p = NULL; CHKERRXX(ierr); // we no longer need to access local data
}

void my_p4est_biomolecules_t::SAS_creator::scatter_to_new_layout(p4est_t*& park, const bool ghost_flag)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) park->user_pointer;
  Vec& phi_sas                              = biomol->phi;
  p4est_nodes_t* & nodes                    = biomol->nodes;
  p4est_ghost_t* & ghost                    = biomol->ghost;
#ifdef CASL_THROWS
  const par_error_manager* sas_err_manager  = &biomol->err_manager;
#endif
  // scatter releveant value before calculating the new phi_sas values
  if(ghost_flag)
  {
    P4EST_ASSERT(ghost == NULL);
    ghost = p4est_ghost_new(park, P4EST_CONNECT_FULL);
  }
#ifdef CASL_THROWS
  int sas_local_error = nodes!= NULL;
  string sas_err_msg = "my_p4est_biomolecules_t::SAS_creator::scatter_to_new_layout(), nodes are not NULL, you have memory leak...";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif
  nodes = my_p4est_nodes_new(park, ghost);
  // scatter the vector to the new layout
  // create the new (partioned) vector of phi_sas
  Vec partitioned_phi_sas;
  ierr = VecCreateGhostNodes(park, nodes, &partitioned_phi_sas); CHKERRXX(ierr);
  ierr = VecSet(partitioned_phi_sas, (PetscScalar) -1.5*fabs(phi_sas_lower_bound));
  // initialize it to values that are below the theoretical lower bound so that apporiate
  // values will be computed...

#ifdef CASL_THROWS
  PetscInt size1, size2;
  ierr = VecGetSize(phi_sas, &size1); CHKERRXX(ierr);
  ierr = VecGetSize(partitioned_phi_sas, &size2); CHKERRXX(ierr);
  sas_local_error = size1 != size2;
  sas_err_msg = "my_p4est_biomolecules_t::SAS_creator::scatter_to_new_layout(), vectors do not have the same size...";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif
  // scatter the values that we already know
  VecScatter ctx;
  IS is;

  int nb_indices = (int) global_indices_of_known_values.size();
  const PetscInt* set_of_global_indices = (nb_indices > 0)? (const PetscInt*) &global_indices_of_known_values.at(0) : PETSC_NULL;
  ierr    = ISCreateGeneral(mpi_comm, nb_indices, set_of_global_indices, PETSC_USE_POINTER, &is); CHKERRXX(ierr);
  ierr    = VecScatterCreate(phi_sas, is, partitioned_phi_sas, is, &ctx); CHKERRXX(ierr);
  ierr    = VecScatterBegin(ctx, phi_sas, partitioned_phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr    = VecScatterEnd(ctx, phi_sas, partitioned_phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr    = VecDestroy(phi_sas); CHKERRXX(ierr);
  ierr    = VecScatterDestroy(ctx); CHKERRXX(ierr);
  ierr    = ISDestroy(is); CHKERRXX(ierr);
  phi_sas = partitioned_phi_sas;
}

void my_p4est_biomolecules_t::SAS_creator::partition_forest_and_update_sas(p4est_t*& park)
{
  if(sub_timer != NULL)
    sub_timer->start("        SAS_creator::scatter_locally");
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  biomol->update_max_level();
  scatter_locally(park);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration();
    sub_timer->start("        SAS_creator::weighted_partition");
  }
  weighted_partition(park);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration();
    sub_timer->start("        SAS_creator::scatter_to_new_layout");
  }
  scatter_to_new_layout(park);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration();
    sub_timer->start("        SAS_creator::update_phi_sas_and_quadrant_data");
  }
  update_phi_sas_and_quadrant_data(park);
  if(sub_timer != NULL)
  {
    sub_timer->stop();sub_timer->read_duration();
  }
  if(biomol->global_max_level == biomol->parameters.max_level()) // final call
  {
    if(sub_timer != NULL)
      sub_timer->start("        SAS_creator:ghost creation");
    ghost_creation_and_final_partitioning(park);
    if(sub_timer != NULL)
    {
      sub_timer->stop();sub_timer->read_duration();
      delete sub_timer; sub_timer = NULL;
    }
  }
}

int  my_p4est_biomolecules_t::SAS_creator::reinitialization_weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  p4est_locidx_t former_quad_idx = (park->mpisize>1)?(quadrant->p.user_long & (biomol->max_quad_loc_idx - 1)) : quadrant->p.user_long;
  for (short k = 0; k < P4EST_CHILDREN; ++k) {
    p4est_locidx_t node_idx = biomol->nodes->local_nodes[P4EST_CHILDREN*former_quad_idx+k];
    if (biomol->phi_read_only_p[node_idx] > -EPS)
      return 1;
  }
  return 0;
}

void my_p4est_biomolecules_t::SAS_creator::ghost_creation_and_final_partitioning(p4est_t *&park)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  PetscInt my_index_offset = 0;
  for (int proc_rank = 0; proc_rank < mpi_rank; ++proc_rank)
    my_index_offset += biomol->nodes->global_owned_indeps[proc_rank];
  global_indices_of_known_values.resize(biomol->nodes->num_owned_indeps);
  for (p4est_locidx_t k = 0; k < biomol->nodes->num_owned_indeps; ++k)
    global_indices_of_known_values.at(k) = my_index_offset + k;
  P4EST_ASSERT(biomol->phi_read_only_p == NULL);
  ierr = VecGetArrayRead(biomol->phi, &biomol->phi_read_only_p); CHKERRXX(ierr);
  my_p4est_partition(park, P4EST_TRUE, reinitialization_weight_fn); // balance calculations for next reinitialization, and allow for coarsening afterwards
  ierr = VecRestoreArrayRead(biomol->phi, &biomol->phi_read_only_p); CHKERRXX(ierr);
  biomol->phi_read_only_p = NULL;
  p4est_nodes_destroy(biomol->nodes); biomol->nodes = NULL;
  scatter_to_new_layout(park, true);
  ierr = VecGhostUpdateBegin(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(biomol->phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void my_p4est_biomolecules_t::SAS_creator::refine_and_partition(p4est_t* & park, const int& step_idx)
{
  if(sas_timer != NULL)
    sas_timer->start("    step " + to_string(step_idx) + ": refining the grid");
  refine_the_p4est(park);
  if(sas_timer != NULL)
  {
    sas_timer->stop(); sas_timer->read_duration();
    sas_timer->start("    step " + to_string(step_idx) + ": updating phi_sas, and partitioning the grid");
  }
  partition_forest_and_update_sas(park);
  if (sas_timer != NULL)
  {
    sas_timer->stop(); sas_timer->read_duration();
  }
}

void my_p4est_biomolecules_t::SAS_creator::refine_the_p4est(p4est_t* & park)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  P4EST_ASSERT(biomol->phi_read_only_p == NULL);
  ierr = VecGetArrayRead(biomol->phi, &biomol->phi_read_only_p); CHKERRXX(ierr);
  specific_refinement(park);
  ierr = VecRestoreArrayRead(biomol->phi, &biomol->phi_read_only_p); CHKERRXX(ierr);
  biomol->phi_read_only_p = NULL;
}

my_p4est_biomolecules_t::SAS_creator::SAS_creator(p4est_t*& park, const bool timing_flag, const bool subtiming_flag)
  : mpi_rank(park->mpirank),
    mpi_size(park->mpisize),
    mpi_comm(park->mpicomm),
    phi_sas_lower_bound(-1.5*((my_p4est_biomolecules_t*) (park->user_pointer))->parameters.root_diag()),
    sas_timer((my_p4est_biomolecules_t::timing_file != NULL && timing_flag)?new parStopWatch(parStopWatch::all_timings, my_p4est_biomolecules_t::timing_file, park->mpicomm):NULL),
    sub_timer((my_p4est_biomolecules_t::timing_file != NULL && timing_flag && subtiming_flag)?new parStopWatch(parStopWatch::all_timings, my_p4est_biomolecules_t::timing_file, park->mpicomm):NULL)
{}

void my_p4est_biomolecules_t::SAS_creator::construct_SAS(p4est_t *&park)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  int step_idx=1;
  for (; biomol->global_max_level < biomol->parameters.max_level(); ++step_idx)
    refine_and_partition(park, step_idx);
  if (sas_timer != NULL)
  {
    delete sas_timer; sas_timer = NULL;
  }
}

my_p4est_biomolecules_t::SAS_creator::~SAS_creator()
{
  if(sas_timer != NULL)
  {
    sas_timer->stop();sas_timer->read_duration();
    delete sas_timer; sas_timer = NULL;
  }
  if(sub_timer != NULL)
  {
    sub_timer->stop(); sub_timer->read_duration();
    delete sub_timer; sub_timer = NULL;
  }
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::initialization_routine(p4est_t *&park)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  biomol->nodes = my_p4est_nodes_new(park, NULL); // we don't need ghost cells here...
  ierr  = VecCreateGhostNodes(park, biomol->nodes, &biomol->phi); CHKERRXX(ierr);
  sample_cf_on_nodes(park, biomol->nodes, *biomol, biomol->phi);
}

int  my_p4est_biomolecules_t::SAS_creator_brute_force::weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  if(quadrant->level == biomol->global_max_level)
    return 1; // we'll have work to do only for newly added points, which are associated with newly created quadrants... workload is constant (loop through ALL atoms)
  else
    return 0;
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::weighted_partition(p4est_t*& park)
{
  my_p4est_partition(park, P4EST_FALSE, weight_fn);
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::specific_refinement(p4est_t* & park)
{
  my_p4est_refine(park, P4EST_FALSE, refine_for_reinitialization_fn, NULL);
}

void my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(p4est_t*& park)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) park->user_pointer;
  Vec& phi_sas                              = biomol->phi;
  double* & phi_sas_p                       = biomol->phi_p;
  p4est_nodes_t* & nodes                    = biomol->nodes;
#ifdef CASL_THROWS
  const par_error_manager* sas_err_manager  = &biomol->err_manager;
#endif

  ierr = VecGetArray(phi_sas, &phi_sas_p); CHKERRXX(ierr);
  vector<int> nb_to_send_recv; nb_to_send_recv.resize(mpi_size, 0);
  // get the number of values to be calculated locally
  for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k) {
    if(phi_sas_p[k] < phi_sas_lower_bound)
      nb_to_send_recv.at(mpi_rank)++;
  }

  // determine first the BALANCED number of points that each proc has to calculate
  // so share yours with all procs
  mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &nb_to_send_recv.at(0), 1, MPI_INT, mpi_comm); SC_CHECK_MPI(mpiret);
  // split the number of calculations evenly through the processes (+-1 calculation difference betweeb procs)
  vector<int> nb_to_be_calculated_balanced; nb_to_be_calculated_balanced.resize(mpi_size, 0);
  int total_nb_to_be_calculated = 0;
  for (int k = 0; k < mpi_size; ++k)
    total_nb_to_be_calculated += nb_to_send_recv.at(k);
  for (int k = 0; k < mpi_size; ++k)
  {
    nb_to_be_calculated_balanced.at(k) = ((k < total_nb_to_be_calculated % mpi_size)?1:0) + total_nb_to_be_calculated/mpi_size;
    nb_to_send_recv.at(k) -= nb_to_be_calculated_balanced.at(k);
  }
  // --> if kth value n_k is
  //     - positive: proc k has  n_k points to send for calculation by (an)other proc(s)
  //     - negative: proc k has -n_k points to recv from (an)other proc(s) for calculation
  //     - zero: proc k can be left alone, it has right enough values to calculate;
#ifdef CASL_THROWS
  int sum_check = 0;
  for (int k = 0; k < mpi_size; ++k)
    sum_check += nb_to_send_recv.at(k);
  int sas_local_error = sum_check != 0;
  string sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): something went wrong when balancing the calculations.\n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif
  // coordinates buffer
  double xyz[P4EST_DIM];

  vector<MPI_Request> query_req; query_req.clear();
  vector<MPI_Request> reply_req; reply_req.clear();

  vector<receiver_data> receivers; receivers.clear();
  const int my_nb_to_send_recv = nb_to_send_recv.at(mpi_rank);
  if(my_nb_to_send_recv > 0)
  {
    int how_many_left_to_assign = my_nb_to_send_recv;
    // calculate the number of previous off-proc requests filling existing gaps before yourself
    int nb_sent_by_previous_procs = 0;
    for (int k = 0; k < mpi_rank; ++k)
      nb_sent_by_previous_procs += MAX(nb_to_send_recv.at(k), 0);
    // find the first proc to which you can send request(s)
    int recv_rank = 0;
    int nb_recv_by_proc_before_recv_rank = 0;
    while (nb_recv_by_proc_before_recv_rank + MAX(-nb_to_send_recv.at(recv_rank), 0) <= nb_sent_by_previous_procs)
      nb_recv_by_proc_before_recv_rank += MAX(-nb_to_send_recv.at(recv_rank++), 0);
    P4EST_ASSERT(nb_to_send_recv.at(recv_rank) < 0);
    P4EST_ASSERT(recv_rank != mpi_rank);
    int that_proc_can_recv = nb_recv_by_proc_before_recv_rank + MAX(-nb_to_send_recv.at(recv_rank), 0) - nb_sent_by_previous_procs;
    int nb_to_send_to_that_rank = MIN(that_proc_can_recv, how_many_left_to_assign);
    receivers.push_back({recv_rank, nb_to_send_to_that_rank});
    how_many_left_to_assign -= nb_to_send_to_that_rank;
    while (how_many_left_to_assign > 0) {
      recv_rank++;
      if(nb_to_send_recv.at(recv_rank) < 0)
      {
        P4EST_ASSERT(recv_rank != mpi_rank);
        nb_to_send_to_that_rank = MIN(-nb_to_send_recv.at(recv_rank), how_many_left_to_assign);
        receivers.push_back({recv_rank, nb_to_send_to_that_rank});
        how_many_left_to_assign -= nb_to_send_to_that_rank;
      }
    }
    P4EST_ASSERT(how_many_left_to_assign == 0);
  }
  int num_remaining_replies = receivers.size();
  int num_remaining_queries = 0;
  vector<int> is_a_receiver; is_a_receiver.resize(mpi_size, 0);
  for (int j = 0; j < num_remaining_replies; ++j) {
    receiver_data& r_data = receivers.at(j);
    is_a_receiver.at(r_data.recv_rank) = 1;
  }
  vector<int> nb_results_per_proc; nb_results_per_proc.resize(mpi_size, 1);
  mpiret = MPI_Reduce_scatter(&is_a_receiver[0], &num_remaining_queries, &nb_results_per_proc[0], MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
#ifdef CASL_THROWS
  // sanity checks
  sas_local_error = (my_nb_to_send_recv > 0) && (num_remaining_queries != 0);
  sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): a proc wants to send messages but expects queries too... \n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);

  sas_local_error = (num_remaining_queries > 0) && (my_nb_to_send_recv >= 0);
  sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): a proc expects queries but has to send messages too \n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);

  sas_local_error = (my_nb_to_send_recv == 0) && (num_remaining_queries != 0);
  sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): a proc expects queries but has just the right number of values to compute... \n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);

  sas_local_error = (num_remaining_queries == 0) && (my_nb_to_send_recv < 0);
  sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): a proc does not expect queries but has too few values to compute...\n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif

  // pack the coordinates to send and send them; declare reply buffer(s)
  map<int, query_buffer> query_buffers; query_buffers.clear();
  map<int, vector<double> > reply_buffers; reply_buffers.clear();
  if(my_nb_to_send_recv > 0) // this proc has too many values to calculate
  {
    int value_to_be_calculated_counter = 0;
    p4est_locidx_t k = 0;
    // skip values to be calculated locally
    while(value_to_be_calculated_counter < nb_to_be_calculated_balanced.at(mpi_rank) && k < nodes->num_owned_indeps)
      if(phi_sas_p[k++] <= phi_sas_lower_bound)
        value_to_be_calculated_counter++;
    for (int query_idx = 0; query_idx < num_remaining_replies; ++query_idx) {
      receiver_data& r_data = receivers.at(query_idx);
      vector<double>& coordinates = query_buffers[r_data.recv_rank].node_coordinates;
      coordinates.resize(P4EST_DIM*r_data.recv_count);
      vector<p4est_locidx_t>& node_indices = query_buffers[r_data.recv_rank].node_local_indices;
      node_indices.resize(r_data.recv_count);
      int off_proc_value_index = 0;
      while(off_proc_value_index < r_data.recv_count && k < nodes->num_owned_indeps)
      {
        if(phi_sas_p[k] <= phi_sas_lower_bound)
        {
          node_xyz_fr_n(k, park, nodes, xyz);
          for (int j = 0; j < P4EST_DIM; ++j)
            coordinates[P4EST_DIM*off_proc_value_index+j] = xyz[j];
          node_indices[off_proc_value_index] = k;
          off_proc_value_index++;
        }
        k++;
      }
    }
    for (int query_idx = 0; query_idx < num_remaining_replies; ++query_idx) {
      receiver_data& r_data = receivers.at(query_idx);
      const vector<double>& to_send = query_buffers.at(r_data.recv_rank).node_coordinates;
      MPI_Request req;
      mpiret = MPI_Isend((void*) &to_send.at(0),
                         P4EST_DIM*r_data.recv_count,
                         MPI_DOUBLE,
                         r_data.recv_rank, query_tag, mpi_comm, &req);
      SC_CHECK_MPI(mpiret);
      query_req.push_back(req);
    }
  }
  size_t nb_local_calculated = 0, nb_local_calculated_end = nb_to_be_calculated_balanced.at(mpi_rank) + ((my_nb_to_send_recv>=0)?0:my_nb_to_send_recv);
  MPI_Status status;
  bool done = false;
  int k = -1;
  while (!done) {
    // calculate local values
    if (nb_local_calculated < nb_local_calculated_end)
    {
      while(phi_sas_p[++k] > phi_sas_lower_bound && k < nodes->num_owned_indeps){}
      node_xyz_fr_n(k, park, nodes, xyz);
      phi_sas_p[k] = (*biomol)(xyz[0], xyz[1]
    #ifdef P4_TO_P8
          , xyz[2]
    #endif
          );
      nb_local_calculated++;
    }

    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int nb_coordinates;
        mpiret = MPI_Get_count(&status, MPI_DOUBLE, &nb_coordinates); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(nb_coordinates % P4EST_DIM == 0);
        int nb_nodes = nb_coordinates/P4EST_DIM;
        vector<double> node_coordinates; node_coordinates.resize(nb_coordinates);
        mpiret = MPI_Recv(&node_coordinates.at(0), nb_coordinates, MPI_DOUBLE, status.MPI_SOURCE, query_tag, mpi_comm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);
        vector<double>& reply = reply_buffers[status.MPI_SOURCE];
        reply.resize(nb_nodes);
        for (int j = 0; j < nb_nodes; ++j)
          reply.at(j) = (*biomol)(node_coordinates[P4EST_DIM*j+0], node_coordinates[P4EST_DIM*j+1]
    #ifdef P4_TO_P8
              , node_coordinates[P4EST_DIM*j+2]
    #endif
              );
        const vector<double>& to_reply = reply;
        MPI_Request req;
        mpiret = MPI_Isend((void*) &to_reply.at(0), nb_nodes, MPI_DOUBLE, status.MPI_SOURCE, reply_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
        reply_req.push_back(req);
        num_remaining_queries--;
      }
    }
    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
        int nb_nodes;
        mpiret = MPI_Get_count(&status, MPI_DOUBLE, &nb_nodes); SC_CHECK_MPI(mpiret);
        vector<double> reply; reply.resize(nb_nodes);
        mpiret = MPI_Recv(&reply.at(0), nb_nodes, MPI_DOUBLE, status.MPI_SOURCE, reply_tag, mpi_comm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(nb_nodes == (int) query_buffers.at(status.MPI_SOURCE).node_local_indices.size()); // check that we received all that was expected
        for (int j = 0; j < nb_nodes; ++j) {
          p4est_locidx_t node_index = query_buffers.at(status.MPI_SOURCE).node_local_indices.at(j);
          phi_sas_p[node_index] = reply.at(j);
        }
        num_remaining_replies--;
      }
    }
    done = num_remaining_queries == 0 && num_remaining_replies == 0 && nb_local_calculated == nb_local_calculated_end;
  }
  ierr = VecRestoreArray(phi_sas, & phi_sas_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);


  // update_local_quadrant_indices in the p.user_long quadrant data, as needed in refine_fn
  for (p4est_topidx_t tree_id = park->first_local_tree; tree_id <= park->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(park->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }
  if(mpi_size>1 && (ulong) park->local_num_quadrants > biomol->max_quad_loc_idx - 1)
  {
#ifdef CASL_THROWS
    sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_brute_force::update_phi_sas_and_quadrant_data(...): the maximum number of local quadrants has been reached, the method needs to be redesigned with real quadrant data...\n \n";
    sas_err_manager->print_message_and_abort(sas_err_msg, 11337799);
#else
    MPI_Abort(mpi_comm, 11337799);
#endif
  }
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::initialization_routine(p4est_t *&park)
{
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == 0);
  my_p4est_biomolecules_t* biomol  = (my_p4est_biomolecules_t*) park->user_pointer;
  p4est_nodes_t* & nodes          = biomol->nodes;
  Vec & phi_sas                   = biomol->phi;
  P4EST_ASSERT(nodes == NULL);
  P4EST_ASSERT(phi_sas == NULL);

  int min_nb_of_atom_in_mol = INT_MAX;
  for (size_t k = 0; k < biomol->bio_molecules.size(); ++k) {
    const molecule& mol = biomol->bio_molecules.at(k);
    min_nb_of_atom_in_mol = MIN(min_nb_of_atom_in_mol, mol.get_number_of_atoms());
  }
  /* I wanted to make the initialization truly scalable as well,
   * but it misbehaves sometimes and can't really figure out why...
   * Raphael Egan */
//  if((biomol->global_max_level == 0) && (min_nb_of_atom_in_mol > mpi_size)) // to make true initialization scalable as well even with huge number of atoms and a few (or only 1) trees
//  {
//    vector<int> position_offset_for_proc(mpi_size);
//    vector<int> nb_atom_idx_in_proc(mpi_size);

//    int rank_owner = 0;
//    while (park->global_first_quadrant[rank_owner + 1] == park->global_first_quadrant[rank_owner])
//      rank_owner++;

//    for (p4est_topidx_t tt = 0; tt < park->connectivity->num_trees; ++tt) {
//      if(tt >= park->global_first_quadrant[rank_owner + 1])
//      {
//        rank_owner++;
//        while (park->global_first_quadrant[rank_owner + 1] == park->global_first_quadrant[rank_owner])
//          rank_owner++;
//      }
//      P4EST_ASSERT(tt >= park->global_first_quadrant[rank_owner] && tt < park->global_first_quadrant[rank_owner+1]);
//      if(mpi_rank == rank_owner)
//      {
//        p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
//        P4EST_ASSERT(tree->quadrants.elem_count == 1);
//        p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, 0);
//        quad->p.user_long = (mpi_size>1)? ((((long) mpi_rank) << (8*sizeof(long) - biomol->rank_encoding)) + biomol->reduced_lists.size()) : biomol->reduced_lists.size();
//      }

//      double  xmin_tree = park->connectivity->vertices[3*park->connectivity->tree_to_vertex[P4EST_CHILDREN*tt + 0] + 0];
//      double  ymin_tree = park->connectivity->vertices[3*park->connectivity->tree_to_vertex[P4EST_CHILDREN*tt + 0] + 1];
//#ifdef P4_TO_P8
//      double  zmin_tree = park->connectivity->vertices[3*park->connectivity->tree_to_vertex[P4EST_CHILDREN*tt + 0] + 2];
//#endif
//      double  dxdydz[P4EST_DIM] = {biomol->root_cells_dim.at(0)
//                                   , biomol->root_cells_dim.at(1)
//                             #ifdef P4_TO_P8
//                                   , biomol->root_cells_dim.at(2)
//                             #endif
//                                  };
//      double  xyz_c[P4EST_DIM] = {xmin_tree + 0.5*dxdydz[0]
//                                  , ymin_tree + 0.5*dxdydz[1]
//                            #ifdef P4_TO_P8
//                                  , zmin_tree + 0.5*dxdydz[2]
//                            #endif
//                                 };

//      double  threshold_dist = MAX(biomol->parameters.layer_thickness(), 0.25*biomol->parameters.lip_cst()*biomol->parameters.root_diag() - biomol->parameters.probe_radius());
//      int     global_idx_maximizing_atom = -1; // absurd initialization
//      double  max_phi_sas_to_vertices = -DBL_MAX;

//      vector<int> atoms_to_consider; atoms_to_consider.clear();
//      for (int mol_idx = 0; mol_idx < biomol->nmol(); ++mol_idx) {
//        const molecule& mol = biomol->bio_molecules.at(mol_idx);
//        const double* mol_centroid = mol.get_centroid();
//        const double mol_bounding_box = mol.get_side_length_of_bounding_cube();
//        const double xyz_box[P4EST_DIM]  = {(xyz_c[0] > mol_centroid[0]+0.5*mol_bounding_box)?mol_centroid[0]+0.5*mol_bounding_box:(xyz_c[0] < mol_centroid[0]-0.5*mol_bounding_box)?mol_centroid[0]-0.5*mol_bounding_box:xyz_c[0]
//                                            , (xyz_c[1] > mol_centroid[1]+0.5*mol_bounding_box)?mol_centroid[1]+0.5*mol_bounding_box:(xyz_c[1] < mol_centroid[1]-0.5*mol_bounding_box)?mol_centroid[1]-0.5*mol_bounding_box:xyz_c[1]
//                                    #ifdef P4_TO_P8
//                                            , (xyz_c[2] > mol_centroid[2]+0.5*mol_bounding_box)?mol_centroid[2]+0.5*mol_bounding_box:(xyz_c[2] < mol_centroid[2]-0.5*mol_bounding_box)?mol_centroid[2]-0.5*mol_bounding_box:xyz_c[2]
//                                    #endif
//                                           };
//        const double xyz_quad[P4EST_DIM] = {(xyz_box[0] > xyz_c[0]+0.5*dxdydz[0])?xyz_c[0]+0.5*dxdydz[0]:(xyz_box[0] < xyz_c[0]-0.5*dxdydz[0])?xyz_c[0]-0.5*dxdydz[0]:xyz_box[0]
//                                            , (xyz_box[1] > xyz_c[1]+0.5*dxdydz[1])?xyz_c[1]+0.5*dxdydz[1]:(xyz_box[1] < xyz_c[1]-0.5*dxdydz[1])?xyz_c[1]-0.5*dxdydz[1]:xyz_box[1]
//                                    #ifdef P4_TO_P8
//                                            , (xyz_box[2] > xyz_c[2]+0.5*dxdydz[2])?xyz_c[2]+0.5*dxdydz[2]:(xyz_box[2] < xyz_c[2]-0.5*dxdydz[2])?xyz_c[2]-0.5*dxdydz[2]:xyz_box[2]
//                                    #endif
//                                           };
//        if(sqrt(SQR(xyz_box[0]-xyz_quad[0]) + SQR(xyz_box[1]-xyz_quad[1])
//        #ifdef P4_TO_P8
//                + SQR(xyz_box[2]-xyz_quad[2])
//        #endif
//                ) <= MAX(biomol->parameters.layer_thickness(), 0.5*biomol->parameters.lip_cst()*biomol->parameters.root_diag() - biomol->parameters.probe_radius())) // there might be relevant atoms in the molecule to consider
//        {
//          int     nb_added_from_this_mol = 0;
//          P4EST_ASSERT(mol.get_number_of_atoms() >= mpi_size);
//          for (int at_idx = mol.get_number_of_atoms()*mpi_rank/mpi_size; at_idx < mol.get_number_of_atoms()*(mpi_rank+1)/mpi_size; ++at_idx) {
//            const Atom* a = mol.get_atom(at_idx);
//            double d = biomol->parameters.probe_radius() + a->max_phi_vdW_in_quad(xyz_c, dxdydz);
//            if(d >= -threshold_dist - EPS*biomol->parameters.probe_radius())
//            {
//              atoms_to_consider.push_back(biomol->atom_index_offset.at(mol_idx) + at_idx);
//              nb_added_from_this_mol++;
//            }
//            if(nb_added_from_this_mol == 0 && d >= MAX(biomol->parameters.probe_radius()-0.5*biomol->parameters.lip_cst()*biomol->parameters.root_diag(), max_phi_sas_to_vertices))
//            {
//              double phi_sas_vertex;
//              for (int ii = 0; ii < 2; ++ii) {
//                for (int jj = 0; jj < 2; ++jj) {
//#ifdef P4_TO_P8
//                  for (int kk = 0; kk < 2; ++kk) {
//#endif
//                    phi_sas_vertex = biomol->parameters.probe_radius() + a->dist_to_vdW_surface(xyz_c[0] + ((double) ii - 0.5)*dxdydz[0], xyz_c[1] + ((double) jj - 0.5)*dxdydz[1]
//    #ifdef P4_TO_P8
//                        , xyz_c[2] + ((double) kk - 0.5)*dxdydz[2]
//    #endif
//                        );
//                    if(phi_sas_vertex > max_phi_sas_to_vertices)
//                    {
//                      max_phi_sas_to_vertices = phi_sas_vertex;
//                      global_idx_maximizing_atom = biomol->atom_index_offset.at(mol_idx) + at_idx;
//                    }
//#ifdef P4_TO_P8
//                  }
//#endif
//                }
//              }
//            }
//          }
//          if(nb_added_from_this_mol == 0 && global_idx_maximizing_atom>=0)
//            atoms_to_consider.push_back(global_idx_maximizing_atom);
//        }
//      }

//      nb_atom_idx_in_proc[mpi_rank] = atoms_to_consider.size();
//      mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &nb_atom_idx_in_proc[0], 1, MPI_INT, mpi_comm); SC_CHECK_MPI(mpiret);

//      int root_reduced_list_size = 0;
//      for (int rank = 0; rank < mpi_size; ++rank)
//        root_reduced_list_size += nb_atom_idx_in_proc.at(rank);
//      if(mpi_rank == rank_owner)
//      {
//        reduced_list_ptr root_reduced_list_ptr(new reduced_list);
//        reduced_list& root_reduced_list = (*root_reduced_list_ptr);
//        root_reduced_list.atom_global_idx.resize(root_reduced_list_size);
//        biomol->reduced_lists.push_back(root_reduced_list_ptr);
//      }
//      if(root_reduced_list_size > 0)
//      {
//        int* pointer_to_root_reduced_list = (mpi_rank == rank_owner)?&((*biomol->reduced_lists.back()).atom_global_idx.at(0)):NULL;
//        position_offset_for_proc[0] = 0;
//        for (int rank = 1; rank < mpi_size; ++rank)
//          position_offset_for_proc[rank]  = position_offset_for_proc[rank-1] + nb_atom_idx_in_proc[rank-1];
//        mpiret = MPI_Gatherv(&atoms_to_consider[0],
//            atoms_to_consider.size(),
//            MPI_INT,
//            pointer_to_root_reduced_list,
//            &nb_atom_idx_in_proc[0],
//            &position_offset_for_proc[0],
//            MPI_INT,
//            rank_owner,
//            mpi_comm); SC_CHECK_MPI(mpiret);
//      }
//      else if(mpi_rank == rank_owner)
//        (*biomol->reduced_lists.back()).atom_global_idx.push_back(0); // no refinement of the root cell is required, any atom is good enough in that case
//    }
//  }
//  else
//  {
//    if(error_file != NULL)
//    {
//        // print a warning
//      string warning_message = "my_p4est_biomolecules_t::SAS_creator_list_reduction::initialization_routine(): THIS PART DOES NOT SCALE AND SHOULD NOT BE USED IF POSSIBLE...\n";
//      PetscFPrintf(mpi_comm, error_file, warning_message.c_str());
//    }
    for (p4est_topidx_t tt = park->first_local_tree; tt <= park->last_local_tree; ++tt) {
      p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
      for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq) {
        p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qq);
        reduced_list_ptr parent_list(new reduced_list(biomol->total_nb_atoms));
        biomol->add_reduced_list(tt, quad, parent_list, get_exact_phi);
      }
    }
//  }

  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() == park->local_num_quadrants);
  nodes = my_p4est_nodes_new(park, NULL); // we don't need ghost cells here...
  ierr  = VecCreateGhostNodes(park, nodes, &phi_sas); CHKERRXX(ierr);
  ierr  = VecSet(phi_sas, -1.5*fabs(phi_sas_lower_bound)); CHKERRXX(ierr);
}

int my_p4est_biomolecules_t::SAS_creator_list_reduction::weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  int min_lvl_to_consider = ((int) biomol->update_last_current_level_only)*biomol->global_max_level;
  if(quadrant->level >= min_lvl_to_consider)
  {
    const reduced_list& r_list = *(biomol->reduced_lists.at((quadrant->p.user_long & (biomol->max_quad_loc_idx -1))));
    return r_list.size(); // we'll have work to do only for newly added points, which are associated with newly created quadrants... workload is constant (loop through ALL atoms)
  }
  else
    return 0;
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::weighted_partition(p4est_t *&park)
{
  my_p4est_partition(park, P4EST_FALSE, weight_fn);
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::update_phi_sas_and_quadrant_data(p4est_t*& park)
{
  my_p4est_biomolecules_t* biomol            = (my_p4est_biomolecules_t*) park->user_pointer;
  Vec& phi_sas                              = biomol->phi;
  double* & phi_sas_p                       = biomol->phi_p;
  p4est_nodes_t* & nodes                    = biomol->nodes;
#ifdef CASL_THROWS
  const par_error_manager* sas_err_manager  = &biomol->err_manager;
#endif

  bool last_stage = (biomol->global_max_level == biomol->parameters.max_level());
  int min_lvl_to_consider = ((int) biomol->update_last_current_level_only)*biomol->global_max_level;
  vector<const p4est_quadrant_t*> locally_known_quadrants; locally_known_quadrants.clear();
  vector<p4est_locidx_t> local_idx_of_locally_known_quadrants; local_idx_of_locally_known_quadrants.clear();
  map<int, query_buffer> query_buffers; query_buffers.clear();
  map<int, vector<int> > reply_buffers; reply_buffers.clear();

  for (p4est_topidx_t tt = park->first_local_tree; tt <= park->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
      if(quad->level < min_lvl_to_consider)
        continue;
      int quad_rank_owner = (mpi_size > 1)? (((ulong) quad->p.user_long)) >> (8*sizeof(long) - biomol->rank_encoding) : 0;
      if(quad_rank_owner != mpi_rank)
      {
        P4EST_ASSERT(quad_rank_owner < mpi_size);
        query_buffer& q_buf = query_buffers[quad_rank_owner];
        p4est_locidx_t new_idx = biomol->reduced_lists.size();
        reduced_list_ptr temporary_list(new reduced_list);
        biomol->reduced_lists.push_back(temporary_list);
        q_buf.new_list_idx.push_back(new_idx);
        q_buf.off_proc_list_idx.push_back((quad->p.user_long & (biomol->max_quad_loc_idx -1)));
        q_buf.local_quad_idx.push_back(tree->quadrants_offset + q);
        quad->p.user_long = ((long) mpi_rank << (8*sizeof(long) - biomol->rank_encoding)) + new_idx;
      }
      else
      {
        locally_known_quadrants.push_back(quad);
        local_idx_of_locally_known_quadrants.push_back(tree->quadrants_offset + q);
      }
    }
  }

#ifdef CASL_THROWS
  int sas_local_error = 0;
  for (map<int, query_buffer>::const_iterator it = query_buffers.begin(); it != query_buffers.end(); ++it)
  {
    sas_local_error |= (it->first == mpi_rank);
    if(sas_local_error)
      break;
    const query_buffer& q_buf = it->second;
    sas_local_error |= (q_buf.new_list_idx.size() != q_buf.off_proc_list_idx.size());
    sas_local_error |= (q_buf.new_list_idx.size() != q_buf.local_quad_idx.size());
    if(sas_local_error)
      break;
  }
  string sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_list_reduction::update_phi_sas_and_quadrant_data(...): something went wrong when figuring out the off-proc queries.\n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif

  vector<int> is_a_receiver;  is_a_receiver.resize(mpi_size, 0);
  vector<int> is_a_sender;    is_a_sender.resize(mpi_size, 0);
  for (map<int, query_buffer>::const_iterator it = query_buffers.begin(); it != query_buffers.end(); ++it)
    is_a_receiver.at(it->first) = 1;
  mpiret = MPI_Alltoall(&is_a_receiver.at(0), 1, MPI_INT, &is_a_sender.at(0), 1, MPI_INT, mpi_comm); SC_CHECK_MPI(mpiret);
  int num_remaining_queries = 0;
  int num_remaining_replies = 0;
  for (int k = 0; k < mpi_size; ++k)
  {
    num_remaining_replies += is_a_receiver.at(k);
    num_remaining_queries += is_a_sender.at(k);
  }

#ifdef CASL_THROWS
  int total_num_queries = 0, total_num_replies = 0;
  mpiret = MPI_Allreduce(&num_remaining_queries, &total_num_queries, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(&num_remaining_replies, &total_num_replies, 1, MPI_INT, MPI_SUM, mpi_comm); SC_CHECK_MPI(mpiret);
  sas_local_error = (total_num_queries != total_num_replies);
  sas_err_msg = "\n my_p4est_biomolecules_t::SAS_creator_list_reduction::update_phi_sas_and_quadrant_data(...): the total numbers of expected queries and replies do not match. \n \n";
  sas_err_manager->check_my_local_error(sas_local_error, sas_err_msg);
#endif

  vector<MPI_Request> query_req; query_req.clear();
  vector<MPI_Request> reply_req; reply_req.clear();

  for (map<int, query_buffer>::iterator it = query_buffers.begin(); it != query_buffers.end(); ++it)
  {
    query_buffer& q_buf = it->second;
    MPI_Request req;
    mpiret = MPI_Isend(&q_buf.off_proc_list_idx.at(0), q_buf.off_proc_list_idx.size(), MPI_INT, it->first, query_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
    query_req.push_back(req);
  }

  MPI_Status status;
  set<p4est_locidx_t> known_fine_indices; known_fine_indices.clear();
  int locally_known_quadrants_treated = 0;
  int nb_locally_known_quadrants = locally_known_quadrants.size();
  ierr  = VecGhostUpdateBegin(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr  = VecGhostUpdateEnd(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGetArray(phi_sas, &phi_sas_p); CHKERRXX(ierr);
  p4est_locidx_t quad_idx;
  double xyz[P4EST_DIM];
  bool done = false;

  while (!done)
  {
    if (locally_known_quadrants_treated < nb_locally_known_quadrants)
    {
      const p4est_quadrant_t* quad = locally_known_quadrants.at(locally_known_quadrants_treated);
      quad_idx = local_idx_of_locally_known_quadrants.at(locally_known_quadrants_treated);
      int reduced_list_idx = (mpi_size > 1)?(quad->p.user_long & (biomol->max_quad_loc_idx -1)):quad->p.user_long;
      for (short i = 0; i < P4EST_CHILDREN; i++) {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
        node_xyz_fr_n(node_idx, park, nodes, xyz);
        phi_sas_p[node_idx] = MAX(phi_sas_p[node_idx], biomol->reduced_operator(xyz, reduced_list_idx, get_exact_phi, last_stage));
        // NOTE THE 'MAX': it looks like we might compute several time the value of the function at the grid nodes,
        // but the max makes it a little more complicated...
      }
      locally_known_quadrants_treated++;
    }
    // probe for incoming queries
    if (num_remaining_queries > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, query_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int nb_queried_reduced_lists;
        mpiret = MPI_Get_count(&status, MPI_INT, &nb_queried_reduced_lists); SC_CHECK_MPI(mpiret);
        vector<int> indices_of_queried_lists; indices_of_queried_lists.resize(nb_queried_reduced_lists);
        mpiret = MPI_Recv(&indices_of_queried_lists.at(0), nb_queried_reduced_lists, MPI_INT, status.MPI_SOURCE, query_tag, mpi_comm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);
        vector<int>& serialized_reply = reply_buffers[status.MPI_SOURCE];
        int size_serialized_reply = 0;
        for (int kk = 0; kk < nb_queried_reduced_lists; ++kk)
        {
          const reduced_list& queried_reduced_list = *(biomol->reduced_lists.at(indices_of_queried_lists.at(kk)));
          size_serialized_reply += (1 + queried_reduced_list.size());
        }
        serialized_reply.resize(size_serialized_reply);
        int idx = 0;
        for (int kk = 0; kk < nb_queried_reduced_lists; ++kk)
        {
          const reduced_list& queried_reduced_list = *(biomol->reduced_lists.at(indices_of_queried_lists.at(kk)));
          serialized_reply.at(idx++) = queried_reduced_list.size();
          for (int jj = 0; jj < queried_reduced_list.size(); ++jj)
            serialized_reply.at(idx++) = queried_reduced_list.atom_global_idx.at(jj);
        }
        P4EST_ASSERT(idx == size_serialized_reply);
        MPI_Request req;
        mpiret = MPI_Isend(&serialized_reply.at(0), size_serialized_reply, MPI_INT, status.MPI_SOURCE, reply_tag, mpi_comm, &req); SC_CHECK_MPI(mpiret);
        reply_req.push_back(req);
        num_remaining_queries--;
      }
    }

    // probe for incoming replies
    if (num_remaining_replies > 0) {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, reply_tag, mpi_comm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending) {
        int size_serialized_reply;
        mpiret = MPI_Get_count(&status, MPI_INT, &size_serialized_reply); SC_CHECK_MPI(mpiret);
        const query_buffer& q_buf = query_buffers.at(status.MPI_SOURCE);
        const int nb_queried_lists = (int) q_buf.off_proc_list_idx.size();
        P4EST_ASSERT(size_serialized_reply >= 2*nb_queried_lists); // check that the reply is at least of the minimal size
        vector<int> serialized_reply; serialized_reply.resize(size_serialized_reply);
        mpiret = MPI_Recv(&serialized_reply.at(0), size_serialized_reply, MPI_INT, status.MPI_SOURCE, reply_tag, mpi_comm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        int idx = 0;
        for (int kk = 0; kk < nb_queried_lists; ++kk) {
          reduced_list_ptr new_list(new reduced_list(serialized_reply.at(idx++), -1));
          reduced_list& list_to_add = *new_list;
          for (int jj = 0; jj < list_to_add.size() ; ++jj)
            list_to_add.atom_global_idx.at(jj) = serialized_reply.at(idx++);
          biomol->reduced_lists.at(q_buf.new_list_idx.at(kk)) = new_list;

          quad_idx = q_buf.local_quad_idx.at(kk);
          for (short i = 0; i < P4EST_CHILDREN; i++) {
            p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
            node_xyz_fr_n(node_idx, park, nodes, xyz);
            phi_sas_p[node_idx] = MAX(phi_sas_p[node_idx], biomol->reduced_operator(xyz, q_buf.new_list_idx.at(kk), get_exact_phi, last_stage));
            known_fine_indices.insert(node_idx);
          }
        }
        P4EST_ASSERT(idx == size_serialized_reply);
        num_remaining_replies--;
      }
    }
    done = num_remaining_queries == 0 && num_remaining_replies == 0 && locally_known_quadrants_treated == nb_locally_known_quadrants;
  }
  ierr = VecRestoreArray(phi_sas, & phi_sas_p); CHKERRXX(ierr);

  mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  // this is the reason we have to stick to our silly sign convention: no MIN_VALUES in PETSC...
  ierr = VecGhostUpdateBegin(phi_sas, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_sas, MAX_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);

  P4EST_ASSERT(biomol->old_reduced_lists.size() == 0);

  if (last_stage)
  {
    if(get_exact_phi)
    {
      ierr = VecGetArray(phi_sas, & phi_sas_p); CHKERRXX(ierr);
      double kink_point[P4EST_DIM];
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        kink_point[dim] = 2.0*biomol->parameters.root_diag();
      set<p4est_locidx_t> done_nodes; done_nodes.clear();

      for (p4est_topidx_t tt = park->first_local_tree; tt <= park->last_local_tree; ++tt) {
        p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
        for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
          p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
          if(quad->level < biomol->global_max_level)
            continue;
          p4est_locidx_t quad_idx = tree->quadrants_offset + q;
          int reduced_list_idx = (mpi_size > 1)?(quad->p.user_long & (biomol->max_quad_loc_idx -1)):quad->p.user_long;
          for (short i = 0; i < P4EST_CHILDREN; i++) {
            p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
            if(done_nodes.find(node_idx) == done_nodes.end())
            {
              if(0 < phi_sas_p[node_idx] && phi_sas_p[node_idx] < biomol->parameters.probe_radius() + biomol->parameters.layer_thickness())
              {
                done_nodes.insert(node_idx);
                node_xyz_fr_n(node_idx, park, nodes, xyz);
                double distance_to_sas = biomol->better_distance(xyz, reduced_list_idx, &kink_point[0]);
                phi_sas_p[node_idx] = MAX(phi_sas_p[node_idx], distance_to_sas);
              }
            }
          }
        }
      }
      ierr = VecRestoreArray(phi_sas, & phi_sas_p); CHKERRXX(ierr);
    }

    ierr  = VecGhostUpdateBegin(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  // update_local_quadrant_indices in the p.user_long quadrant data, as needed in refine_fn
  // --> store the shared pointers to reduced lists in the biomol->old_reduced_lists map
  for (p4est_topidx_t tt = park->first_local_tree; tt <= park->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_loc_idx = tree->quadrants_offset + q;
      if(quad->level >= min_lvl_to_consider && !last_stage)
      {
        int reduced_list_idx = ((mpi_size > 1)? quad->p.user_long & (biomol->max_quad_loc_idx - 1) : quad->p.user_long);
        biomol->old_reduced_lists[quad_loc_idx] = biomol->reduced_lists.at(reduced_list_idx);
      }
      quad->p.user_long = (mpi_size> 1)? ((long) mpi_rank << (8*sizeof(long) - biomol->rank_encoding)) + quad_loc_idx : quad_loc_idx;
    }
  }
  if(last_stage)
  {
    ierr  = VecGhostUpdateEnd(phi_sas, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  biomol->reduced_lists.clear();
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::replace_fn(p4est_t *park, p4est_topidx_t which_tree, int num_outgoing, p4est_quadrant_t *outgoing[], int num_incoming, p4est_quadrant_t *incoming[])
{
  (void) num_incoming;
  if (num_outgoing > 1) {
    // this is coarsening, it should NEVER happend
#ifdef CASL_THROWS
    my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
    string err_msg = "my_p4est_biomolecules_t::SAS_creator_list_reduction::replace_fn(...): invoked coarsening...";
    biomol->err_manager.print_message_and_abort(err_msg, 22446688);
#else
    MPI_Abort(park->mpicomm, 22446688);
#endif
  }
  else {
    /* this is refinement */
    my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
    SAS_creator_list_reduction* this_creator = dynamic_cast<SAS_creator_list_reduction*>(biomol->sas_creator);
    int parent_list_idx = (park->mpisize > 1)?(outgoing[0]->p.user_long & (biomol->max_quad_loc_idx -1 )) : outgoing[0]->p.user_long;
    for (short i = 0; i < P4EST_CHILDREN; i++)
      biomol->add_reduced_list(which_tree, incoming[i], biomol->old_reduced_lists.at(parent_list_idx), this_creator->get_exact_phi);
  }
}

void my_p4est_biomolecules_t::SAS_creator_list_reduction::specific_refinement(p4est_t*& park)
{
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  P4EST_ASSERT(biomol->reduced_lists.size() == 0);
  if(!get_exact_phi)
    p4est_refine_ext(park, P4EST_FALSE, -1, refine_for_reinitialization_fn, NULL, replace_fn);
  else
    p4est_refine_ext(park, P4EST_FALSE, -1, refine_for_exact_calculation_fn, NULL, replace_fn);
  biomol->old_reduced_lists.clear();
  P4EST_ASSERT(reduced_list::get_nb_reduced_lists() <= park->local_num_quadrants);
#ifdef CASL_THROWS
  p4est_gloidx_t nb_new_fine_quad = 0;
  for (p4est_topidx_t tt = park->first_local_tree; tt <= park->last_local_tree; ++tt) {
    p4est_tree_t* tree = p4est_tree_array_index(park->trees, tt);
    nb_new_fine_quad += tree->quadrants_per_level[biomol->global_max_level+1];
  }
  int sas_local_error = (nb_new_fine_quad != (p4est_gloidx_t) biomol->reduced_lists.size());
  string sas_error_msg = "my_p4est_biomolecules_t::SAS_creator_list_reduction::specific_refinement(): something went wrong when associating reduced lists to new quadrants...";
  biomol->err_manager.check_my_local_error(sas_local_error, sas_error_msg);
#endif
}

void my_p4est_biomolecules_t::check_if_directory(const string& folder_path) const
{
  par_error_manager no_debug_error_manager(p4est->mpirank, p4est->mpisize, p4est->mpicomm, error_file); // we want this check to be done even in non-debug case...
  struct stat st;
  int err_st = stat(folder_path.c_str(), &st);
  int local_error = (-1 == err_st || !S_ISDIR(st.st_mode));
  string err_msg = "my_p4est_biomolecules_t::check_if_directory(const string*): invalid directory path: " + folder_path + " does not exist or is not a directory.";
  no_debug_error_manager.check_my_local_error(local_error, err_msg);
}
bool my_p4est_biomolecules_t::molecule::check_if_file(const string& file_path) const
{
  struct stat st;
  int err_st = stat(file_path.c_str(), &st);
  return (0 == err_st && S_ISREG(st.st_mode));
}
void my_p4est_biomolecules_t::check_validity_of_vector_of_mol() const
{
#ifdef CASL_THROWS
  int local_error;
  string err_msg;
  // check if there is one molecule at least
  local_error = nmol() <= 0;
  err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): empty list of molecules.";
  err_manager.check_my_local_error(local_error, err_msg);

  // are offset consistent?
  local_error = (atom_index_offset.size() != (size_t) nmol());
  err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): the size of the vector of offset(s) of atom indices is wrong.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = (atom_index_offset.at(0) != 0);
  err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): the vector of offset(s) of atom indices is invalid.";
  err_manager.check_my_local_error(local_error, err_msg);

  for (int k = 1; k < nmol(); ++k) {
    local_error = (atom_index_offset.at(k) != atom_index_offset.at(k-1)+bio_molecules.at(k-1).get_number_of_atoms());
    err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): the vector of offset(s) of atom indices is invalid.";
    err_manager.check_my_local_error(local_error, err_msg);
  }

  // check box_size_of_biggest_mol and index_of_biggest_mol
  const molecule& biggest_mol = bio_molecules.at(index_of_biggest_mol);
  local_error = (fabs(box_size_of_biggest_mol - biggest_mol.get_side_length_of_bounding_cube()) > EPS*box_size_of_biggest_mol);
  err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): the biggest molecule is misidentified.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = !are_all_molecules_scaled_consistently();
  err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): scaling is not consistent.";
  err_manager.check_my_local_error(local_error, err_msg);

  // check the molecules
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol_k = bio_molecules.at(k);
    local_error = !mol_k.is_bounding_box_in_domain();
    err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): a bounding box is not entirely in the domain";
    err_manager.check_my_local_error(local_error, err_msg);
    local_error = (mol_k.get_side_length_of_bounding_cube() > box_size_of_biggest_mol);
    err_msg = "my_p4est_biomolecules_t:: check_validity_of_vector_of_mol(): the biggest molecule is misidentified";
    err_manager.check_my_local_error(local_error, err_msg);
  }
#endif
  return;
}
bool my_p4est_biomolecules_t::are_all_molecules_scaled_consistently() const
{
  bool result = nmol() > 0;
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol_k = bio_molecules.at(k);
    result &= (mol_k.is_scaled() && fabs(angstrom_to_domain - mol_k.get_angstrom_to_domain_factor()) < EPS*angstrom_to_domain);
    if(!result)
      return result;
  }
  return result;
}
bool my_p4est_biomolecules_t::is_no_molecule_scaled() const
{
  bool result = true;
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol_k = bio_molecules.at(k);
    result &= !mol_k.is_scaled();
    if(!result)
      return result;
  }
  return result;
}
void my_p4est_biomolecules_t::get_vector_of_current_centroids(vector<double>& current_centroids)
{
  current_centroids.resize(P4EST_DIM*nmol());
  for (int k = 0; k < nmol(); ++k) {
    molecule& mol_k = bio_molecules.at(k);
    for (int i = 0; i < P4EST_DIM; ++i) {
      current_centroids.at(k*P4EST_DIM+i) = *(mol_k.get_centroid()+i);
    }
  }
}
void my_p4est_biomolecules_t::rescale_all_molecules()
{
  vector<double> current_centroids;
  get_vector_of_current_centroids(current_centroids);
  rescale_all_molecules(&current_centroids.at(0));
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double* new_centroids)
{
  box_size_of_biggest_mol = -DBL_MAX;
  const double new_scale = angstrom_to_domain;
  for (int k = 0; k < nmol(); ++k) {
    molecule& mol_k = bio_molecules.at(k);
    mol_k.scale_and_translate(&new_scale, (new_centroids != NULL)?(new_centroids+P4EST_DIM*k):NULL);
    if(mol_k.get_side_length_of_bounding_cube() > box_size_of_biggest_mol)
    {
      box_size_of_biggest_mol = mol_k.get_side_length_of_bounding_cube();
      index_of_biggest_mol    = k;
    }
  }
}

void my_p4est_biomolecules_t::add_single_molecule(const string& file_path, const double* centroid, double* angles, const double* angstrom_to_domain_)
{
  double new_scaling_factor; // angstrom_to_domain scaling factor to be applied when reading the pqr file, on-the-fly
  if(angstrom_to_domain_ == NULL) // no specific scaling factor given for the new molecule
  {
    if(is_no_molecule_scaled())
      new_scaling_factor = 1.0; // i.e. no scaling given, no scaling used so far, so do not scale when reading
    else if(are_all_molecules_scaled_consistently())
      new_scaling_factor = angstrom_to_domain; // i.e. no scaling given, but consistent scaling for other molecules, so impose the same scaling as for others
    else // should never happen, this is a logic error...
    {
      string err_message = "my_p4est_biomolecules_t::add_single_molecule(...): the current vector of molecules has inconsistent scaling factors, this shouldn't happen...";
#ifdef CASL_THROWS
      int local_error = 1;
      err_manager.check_my_local_error(local_error, err_message);
#endif
      cerr <<  err_message << endl;
      cerr << "my_p4est_biomolecules_t::add_single_molecule(...): ... rescaling the molecules (fixing a logic error)..." << endl;
      rescale_all_molecules();
      new_scaling_factor = angstrom_to_domain;
    }
  }
  else // a specific scaling factor is given for the new molecule, scale all others consistently if needed before reading the new one
  {
    if(fabs(angstrom_to_domain - *angstrom_to_domain_) > EPS*MAX(EPS, MAX(angstrom_to_domain, *angstrom_to_domain_)) || is_no_molecule_scaled())
      rescale_all_molecules(*angstrom_to_domain_);
    else if(!are_all_molecules_scaled_consistently())// should never happen, this is a logic error...
    {
      string err_message = "my_p4est_biomolecules_t::add_single_molecule(...): the current vector of molecules has inconsistent scaling factors, this shouldn't happen...";
#ifdef CASL_THROWS
      int local_error = 1;
      err_manager.check_my_local_error(local_error, err_message);
#endif
      cerr <<  err_message << endl;
      cerr << "my_p4est_biomolecules_t::add_single_molecule(...): ... rescaling the molecules (fixing a logic error)..." << endl;
      rescale_all_molecules(*angstrom_to_domain_);
    }
    new_scaling_factor = *angstrom_to_domain_;
  }
  const double new_angstrom_to_domain = new_scaling_factor; // need a const qualifier for the following, so just copy it...
  molecule mol(file_path, centroid, angles, new_angstrom_to_domain);

  if(nmol() == 0) // first molecule to be added in the vector of molecules
    atom_index_offset.resize(1, 0);
  else
  {
    atom_index_offset.push_back(total_nb_atoms);
  }
  bio_molecules.push_back(mol);
  total_nb_atoms += mol.get_number_of_atoms();

  if(mol.get_side_length_of_bounding_cube() > box_size_of_biggest_mol)
  {
    box_size_of_biggest_mol = mol.get_side_length_of_bounding_cube();
    index_of_biggest_mol = nmol()-1;
  }
}

int my_p4est_biomolecules_t::find_mol_index(const int& global_atom_index, const int& guess) const
{
#ifdef CASL_THROWS
  int local_error = (global_atom_index < 0 || global_atom_index >= total_nb_atoms);
  if(local_error) // local abort since it is not a collective call
  {
    string err_msg = "my_p4est_biomolecules_t::find_mol_index(const int&): invalid global atom index, out of range: global_atom_index = " + to_string(global_atom_index) + " and total_nb_atoms = " + to_string(total_nb_atoms) +", called from proc " + to_string(p4est->mpirank) + " ...";
    err_manager.print_message_and_abort(err_msg, 456654);
  }
#endif
  // check the guess first
  if(0 <= guess && guess < nmol() && atom_index_offset.at(guess) <= global_atom_index && global_atom_index < atom_index_offset.at(guess) + bio_molecules.at(guess).get_number_of_atoms())
    return guess;
  int L = 0;
  if(0 < guess && guess < nmol() && atom_index_offset.at(guess) <= global_atom_index)
    L = guess;
  int R = nmol();
  if(0 <= guess && guess < nmol()-1 && global_atom_index < atom_index_offset.at(guess) + bio_molecules.at(guess).get_number_of_atoms())
    R = guess +1;
  int two = 2;
  int mol_index;
  while (R-L > 1)
  {
    mol_index = (L+R)/two;
    if(atom_index_offset.at(mol_index) <= global_atom_index)
      L = mol_index;
    else
      R = mol_index;
    P4EST_ASSERT(R>L);
  }
  return L;
}

const Atom* my_p4est_biomolecules_t::get_atom(const int& global_atom_index, int& guess) const
{
  guess = find_mol_index(global_atom_index, guess);
  const molecule& mol = bio_molecules.at(guess);
  int atom_index = global_atom_index - atom_index_offset.at(guess);
  P4EST_ASSERT(0<= atom_index && atom_index < mol.get_number_of_atoms());
  return mol.get_atom(atom_index);
}
my_p4est_biomolecules_t::my_p4est_biomolecules_t(p4est_t* p4est_, mpi_environment_t* mpi_,
                                                 const vector<string>* pqr_names, const string* input_folder,
                                                 vector<double>* angles,
                                                 const vector<double>* centroids,
                                                 const double* rel_side_length_biggest_box):
  #ifdef CASL_THROWS
  err_manager(p4est_->mpirank, p4est_->mpisize, p4est_->mpicomm, error_file),
  #endif
  p4est(p4est_),
  domain_dim(calculate_domain_dimensions(p4est_)),
  root_cells_dim(calculate_dimensions_of_root_cells(p4est_)),
  rank_encoding((int) (ceil(log2(p4est_->mpisize)))),
  max_quad_loc_idx(((ulong) 1)<<(8*sizeof(long) - (int) (ceil(log2(p4est_->mpisize))))),
  parameters(norm(calculate_dimensions_of_root_cells(p4est_)))
{
#ifdef CASL_THROWS
  // sanity checks
  int local_error = mpi_== NULL;
  string err_msg = "my_p4est_biomolecules_t::my_p4est_biomolecules_t(p4est_t*, mpi_environment_t*, ...): a valid pointer to an mpi environment is required for the second argument (none provided).";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = p4est_->mpicomm != mpi_->comm();
  err_msg = "my_p4est_biomolecules_t::my_p4est_biomolecules_t(p4est_t*, mpi_environment_t*, ...): the mpi communicators of the p4est and the mpi environment do not match.";
  err_manager.check_my_local_error(local_error, err_msg);

  if(pqr_names == NULL && (angles != NULL || centroids != NULL || rel_side_length_biggest_box != NULL))
    cerr << "my_p4est_biomolecules_t::my_p4est_biomolecules_t(...): angle(s), centroid(s) and/or desired scaling set, but pqr file(s) undefined "
         << ": no molecule will be read so no rotation/translation/scaling will be applied." << endl;
  int nmolecules = (pqr_names != NULL)? pqr_names->size() : 0;

  local_error = ((angles != NULL) && (angles->size() != (size_t) nangle_per_mol*nmolecules) && (angles->size() != (size_t) nangle_per_mol));
  if(local_error)
    err_msg = "my_p4est_biomolecules_t::my_p4est_biomolecules_t(...): invalid number of rotation angles (fifth pointer argument). It should contain "+ to_string(nangle_per_mol*nmolecules) + " or " + to_string(nangle_per_mol) + " values (the same rotation for all molecules), but is of size " + to_string(angles->size()) + ".";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = (centroids != NULL && centroids->size() != (size_t) P4EST_DIM*nmolecules);
  err_msg = "my_p4est_biomolecules_t::my_p4est_biomolecules_t(...): the number of centroid coordinates is invalid (last pointer argument). It should contain " + to_string(P4EST_DIM*nmolecules) + " values, but is of size " + to_string(centroids->size()) + ".";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  if(timing_file != NULL)
    timer = new parStopWatch(parStopWatch::all_timings, timing_file, p4est->mpicomm);

  int update_failure = molecule::update_connectivity(p4est->connectivity);
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::constructor(...), the molecule pointer to the domain connectivity (static variable) cannot be changed once it is set.";
  err_manager.check_my_local_error(update_failure, err_msg);
#else
  (void) update_failure; // avoid compilation warning
#endif

  update_failure = molecule::update_mpi(mpi_);
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::constructor(...), the pointer to the mpi environment (static variable) cannot be changed once set.";
  err_manager.check_my_local_error(update_failure, err_msg);
#endif

  if(timer != NULL)
    timer->start("Construct the my_p4est_biomolecules_t object (reading the molecules, scaling them, ...)");

  // basic initialization of relevant parameters
  bio_molecules.clear();
  atom_index_offset.clear();
  total_nb_atoms          = 0;
  index_of_biggest_mol    = -1;       // absurd value, there is no molecule yet
  box_size_of_biggest_mol = -DBL_MAX; // absurd value, there is no molecule yet
  angstrom_to_domain      = 1.0;      // no molecule yet, so no scaling yet

  reduced_lists.clear(); // make sure no reduced_list exists beforehand
#ifdef CASL_THROWS
  local_error = (reduced_list::get_nb_reduced_lists() > 0);
  err_msg = "my_p4est_biomolecules_t::constructor(...), couldn't delete all reduced lists of global atom indices, there might be memory leak(s)...";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  nodes                     = NULL;     // we will build the nodes internally, NULL initialization
  ghost                     = NULL;     // we will build the ghost internally, NULL initialization
  phi                       = NULL;     // we will build that one too, NULL initialization
  phi_p                     = NULL;
  phi_read_only_p           = NULL;
  inner_domain              = NULL;
  sas_creator               = NULL;     // NULL initialization
  brick                     = NULL; build_brick(); // brick is initialized from the p4est
  hierarchy                 = NULL;     // we will build that one too, NULL initialization
  neighbors                 = NULL;     // we will build that one too, NULL initialization
  ls                        = NULL;     // we will build that one too, NULL initialization

  if(pqr_names != NULL)
  {
    // read and rotate them if desired
    string slash = "/";
    bool add_slash = false;
    if(input_folder != NULL)
    {
      check_if_directory(*input_folder);
      add_slash = input_folder->at(input_folder->size()-1) != '/';
    }
    for (size_t k = 0; k < pqr_names->size(); ++k) {
      string file_path= ((input_folder != NULL)?(*input_folder + ((add_slash)?slash:"")):"") + pqr_names->at(k);
      add_single_molecule(file_path,
                          (centroids != NULL)? &centroids->at(P4EST_DIM*k):NULL,
                          (angles != NULL)? &angles->at((nangle_per_mol*k < angles->size())?nangle_per_mol*k:0):NULL); // no scaling yet, needs to know about the biggest molecule first, scale afterwards
    }
    // scale the molecules and locate them
    if(rel_side_length_biggest_box != NULL)
    {
      set_biggest_bounding_box(*rel_side_length_biggest_box);
      check_validity_of_vector_of_mol();
    }
    print_summary();
  }

  if(timer != NULL)
  {
    timer->stop(); timer->read_duration();
  }
}
Vec my_p4est_biomolecules_t::return_phi_vector()
{
  if(phi_read_only_p != NULL)
  {
#ifdef CASL_THROWS
    string message = "my_p4est_biomolecules_t::return_phi_vector(...): the pointer to the read-only local array of phi was not NULL...\n";
    PetscFPrintf(p4est->mpicomm, error_file, message.c_str());
#endif
    PetscErrorCode ierr = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);
    phi_read_only_p = NULL;
  }
  if(phi_p != NULL)
  {
#ifdef CASL_THROWS
    string message = "my_p4est_biomolecules_t::return_phi_vector(...): the pointer to the local array of phi was not NULL...\n";
    PetscFPrintf(p4est->mpicomm, error_file, message.c_str());
#endif
    PetscErrorCode ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    phi_p = NULL;
  }
#ifdef CASL_THROWS
  int local_error = phi == NULL;
  string message = "my_p4est_biomolecules_t::return_phi_vector(): the phi vector is NULL, it can't be returned...";
  err_manager.check_my_local_error(local_error, message);
#endif
  Vec phi_to_return = phi; phi = NULL;
  return phi_to_return;
}
p4est_nodes_t* my_p4est_biomolecules_t::return_nodes()
{
#ifdef CASL_THROWS
  int local_error = nodes == NULL;
  string message = "my_p4est_biomolecules_t::return_nodes(): the nodes are not valid they can't be returned...";
  err_manager.check_my_local_error(local_error, message);
#endif
  p4est_nodes_t* nodes_to_return = nodes; nodes = NULL;
  return nodes_to_return;
}
p4est_ghost_t* my_p4est_biomolecules_t::return_ghost()
{
#ifdef CASL_THROWS
  int local_error = ghost == NULL;
  string message = "my_p4est_biomolecules_t::return_ghost(): the ghosts are not valid, they can't be returned...";
  err_manager.check_my_local_error(local_error, message);
#endif
  p4est_ghost_t* ghost_to_return = ghost; ghost = NULL;
  return ghost_to_return;
}
void my_p4est_biomolecules_t::return_phi_vector_nodes_and_ghost(Vec& phi_out, p4est_nodes_t*& nodes_out, p4est_ghost_t*& ghost_out)
{
  phi_out           = return_phi_vector();
  nodes_out         = return_nodes();
  ghost_out         = return_ghost();
}
my_p4est_biomolecules_t::~my_p4est_biomolecules_t()
{
  if(nodes != NULL)
  {
    p4est_nodes_destroy(nodes);
    nodes = NULL;
  }
  if(ghost != NULL)
  {
    p4est_ghost_destroy(ghost);
    ghost = NULL;
  }
  if(phi_read_only_p != NULL)
  {
    PetscErrorCode ierr = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);
    phi_read_only_p = NULL;
  }
  if(phi_p != NULL)
  {
    PetscErrorCode ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    phi_read_only_p = NULL;
  }
  if(phi != NULL)
  {
    PetscErrorCode ierr = VecDestroy(phi); CHKERRXX(ierr);
    phi = NULL;
  }
  if(inner_domain != NULL)
  {
    PetscErrorCode ierr = VecDestroy(inner_domain); CHKERRXX(ierr);
    inner_domain = NULL;
  }

  if(brick != NULL)
  {
    P4EST_ASSERT (brick->nxyz_to_treeid != NULL);
    P4EST_FREE (brick->nxyz_to_treeid);
    brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  if(hierarchy != NULL)
  {
    delete hierarchy; hierarchy = NULL;
  }
  if(neighbors != NULL)
  {
    delete neighbors; neighbors = NULL;
  }
  if(ls != NULL)
  {
    delete ls; ls = NULL;
  }
  if(timer != NULL)
  {
    timer->stop();timer->read_duration();
    delete timer;
    timer = NULL;
  }
}
void my_p4est_biomolecules_t::add_single_molecule(const string& file_path, const vector<double> *centroid, vector<double> *angles, const double *angstrom_to_domain)
{
#ifdef CASL_THROWS
  int local_error = (centroid != NULL && centroid->size() != P4EST_DIM);
  string err_msg = "my_p4est_biomolecules_t::add_single_molecule(const string *, const vector<double> *, vector<double> *, const double *): vector of centroid components has invalid size.";
  err_manager.check_my_local_error(local_error, err_msg);
  local_error = (angles != NULL && angles->size() != nangle_per_mol);
  err_msg = "my_p4est_biomolecules_t::add_single_molecule(const string *, const vector<double> *, vector<double> *, const double *): vector of angles has invalid size.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  add_single_molecule(file_path,
                      (centroid != NULL)? &centroid->at(0): NULL,
                      (angles != NULL)? &angles->at(0): NULL,
                      angstrom_to_domain);
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double& new_scaling_factor)
{
  vector<double> current_centroids;
  get_vector_of_current_centroids(current_centroids);
  rescale_all_molecules(new_scaling_factor, &current_centroids);
}
void my_p4est_biomolecules_t::rescale_all_molecules(const double& new_scaling_factor, const vector<double>* centroids)
{
#ifdef CASL_THROWS
  int local_error = (new_scaling_factor < 0);
  string err_msg = "my_p4est_biomolecules_t::rescale_all_molecules(const double&, const vector<double>*): the scaling factor must be strictly positive.";
  err_manager.check_my_local_error(local_error, err_msg);
  local_error = (centroids != NULL && centroids->size() != (size_t) P4EST_DIM*nmol());
  err_msg = "my_p4est_biomolecules_t::rescale_all_molecules(const double&, const vector<double>*): the vector of centroid coordinates has invalid size.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  angstrom_to_domain = new_scaling_factor;
  rescale_all_molecules((centroids != NULL)?(&centroids->at(0)):NULL);
}
void my_p4est_biomolecules_t::set_biggest_bounding_box(const double& biggest_cube_side_length_to_min_domain_size)
{
#ifdef CASL_THROWS
  int local_error = (nmol()==0);
  string err_msg = "my_p4est_biomolecules_t::set_biggest_bounding_box(const double&): empty vector of molecule, impossible action.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  molecule& biggest_mol = bio_molecules.at(index_of_biggest_mol);
  rescale_all_molecules(biggest_mol.calculate_scaling_factor(biggest_cube_side_length_to_min_domain_size));
}
void my_p4est_biomolecules_t::print_summary() const
{
  if(log_file == NULL)
    return;
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol = bio_molecules.at(k);
    string message = "Molecule %d is located at x = "+ to_string(*mol.get_centroid()) + ", y = " + to_string(*(mol.get_centroid()+1)) +
    #ifdef P4_TO_P8
        ", z = " + to_string(*(mol.get_centroid()+2)) +
    #endif
        ", is bounded by a cube of side length %.5f, and contains %d atoms, %d of which being charged. \n" ;
    PetscErrorCode ierr = PetscFPrintf(p4est->mpicomm, log_file, message.c_str(),k,mol.get_side_length_of_bounding_cube(), mol.get_number_of_atoms(), mol.get_number_of_charged_atoms()); CHKERRXX(ierr);
  }
  PetscErrorCode ierr = PetscFPrintf(p4est->mpicomm, log_file, " \n \n "); CHKERRXX(ierr);
}

void my_p4est_biomolecules_t::set_grid_and_surface_parameters(const int &lmin, const int &lmax, const double &lip_, const double &rp_, const int ooa_)
{
  // splitting criterion first
  int need_to_reset_p4est = (int) parameters.set_splitting_criterion(lmin, lmax, lip_);
#ifdef CASL_THROWS
  string err_msg = "my_p4est_biomolecules_t::set_grid_and_surface_parameters(...): the splitting criterion cannot be reset.";
  err_manager.check_my_local_error(need_to_reset_p4est, err_msg);
#endif
  //probe_radius
#ifdef CASL_THROWS
  int local_error = (!are_all_molecules_scaled_consistently());
  err_msg = "my_p4est_biomolecules_t::set_grid_and_surface_parameters(...): the probe radius cannot be (re)set if molecules are not scaled consistently.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  need_to_reset_p4est |= parameters.set_probe_radius(angstrom_to_domain*rp_);
  // order of accuracy (for thickess of accuracy layer)
  need_to_reset_p4est |= parameters.set_OOA(ooa_);
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::set_grid_and_surface_parameters(...): the order of accuracy cannot be (re)set.";
  err_manager.check_my_local_error(need_to_reset_p4est, err_msg);
#endif
  if (need_to_reset_p4est)
    p4est = reset_p4est();
}

void my_p4est_biomolecules_t::set_splitting_criterion(const int& lmin, const int& lmax, const double& lip_)
{
  set_grid_and_surface_parameters(lmin, lmax, lip_, parameters.probe_radius(), parameters.order_of_accuracy());
}
void my_p4est_biomolecules_t::set_probe_radius(const double& rp)
{
  set_grid_and_surface_parameters(parameters.min_level(), parameters.max_level(), parameters.lip_cst(), rp, parameters.order_of_accuracy());
}
void my_p4est_biomolecules_t::set_order_of_accuracy(const int& ooa)
{
  set_grid_and_surface_parameters(parameters.min_level(), parameters.max_level(), parameters.lip_cst(), parameters.probe_radius(), ooa);
}
vector<double> my_p4est_biomolecules_t::calculate_dimensions_of_root_cells(p4est_t* p4est_)
{
  vector<double> dimensions_of_root_cells(P4EST_DIM);
  p4est_connectivity_t* conn      = p4est_->connectivity;
  double *vertices_to_coordinates = conn->vertices;
  p4est_topidx_t *tree_to_vertex  = conn->tree_to_vertex;
  // consider the first (and possibly only) tree only
  double xmin                     = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 0];
  double xmax                     = vertices_to_coordinates[3*tree_to_vertex[0 + P4EST_CHILDREN-1] + 0];
  dimensions_of_root_cells.at(0)  = xmax-xmin;
  double ymin                     = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 1];
  double ymax                     = vertices_to_coordinates[3*tree_to_vertex[0 + P4EST_CHILDREN-1] + 1];
  dimensions_of_root_cells.at(1)  = ymax-ymin;
#ifdef P4_TO_P8
  double zmin                     = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 2];
  double zmax                     = vertices_to_coordinates[3*tree_to_vertex[0 + P4EST_CHILDREN-1] + 2];
  dimensions_of_root_cells.at(2)  = zmax-zmin;
#endif
  return dimensions_of_root_cells;
}
vector<double> my_p4est_biomolecules_t::calculate_domain_dimensions(p4est_t* p4est_)
{
  vector<double> domain_dimensions(P4EST_DIM);
  p4est_connectivity_t* conn      = p4est_->connectivity;
  double *vertices_to_coordinates = conn->vertices;
  p4est_topidx_t *tree_to_vertex  = conn->tree_to_vertex;
  double xmin             = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 0];
  double xmax             = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(conn->num_trees-1) + P4EST_CHILDREN-1] + 0];
  domain_dimensions.at(0) = xmax-xmin;
  double ymin             = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 1];
  double ymax             = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(conn->num_trees-1) + P4EST_CHILDREN-1] + 1];
  domain_dimensions.at(1) = ymax-ymin;
#ifdef P4_TO_P8
  double zmin             = vertices_to_coordinates[3*tree_to_vertex[0 + 0] + 2];
  double zmax             = vertices_to_coordinates[3*tree_to_vertex[P4EST_CHILDREN*(conn->num_trees-1) + P4EST_CHILDREN-1] + 2];
  domain_dimensions.at(2) = zmax-zmin;
#endif
  return domain_dimensions;
}
double my_p4est_biomolecules_t::get_largest_radius_of_all() const
{
  check_validity_of_vector_of_mol();
  double largest_largest_radius = -DBL_MAX;
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol_k = bio_molecules.at(k);
    largest_largest_radius = MAX(largest_largest_radius, mol_k.get_largest_radius());
  }
  return largest_largest_radius;
}
p4est_t* my_p4est_biomolecules_t::reset_p4est()
{
  p4est_t* new_p4est = my_p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_destroy(p4est);
  return new_p4est;
}
void my_p4est_biomolecules_t::update_max_level()
{
  global_max_level = 0;
  for (p4est_topidx_t k = p4est->first_local_tree; k <= p4est->last_local_tree; ++k) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, k);
    global_max_level = MAX(global_max_level, (int) tree_k->maxlevel);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &global_max_level, 1, MPI_INT, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_biomolecules_t::add_reduced_list(p4est_topidx_t which_tree, p4est_quadrant_t *quad, reduced_list_ptr parent_list_ptr, const bool& need_exact_phi)
{
  quad->p.user_long = (p4est->mpisize>1)? ((((long) p4est->mpirank) << (8*sizeof(long) - rank_encoding)) + reduced_lists.size()) : reduced_lists.size();

  if(parent_list_ptr != NULL && (*parent_list_ptr).size() ==1)
  {
    reduced_lists.push_back(parent_list_ptr);
    return;
  }

  double  xmin_tree = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0] + 0];
  double  ymin_tree = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0] + 1];
#ifdef P4_TO_P8
  double  zmin_tree = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*which_tree + 0] + 2];
#endif
  double  quad_rel_size = ((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN);
  double  dxdydz[P4EST_DIM] = {root_cells_dim.at(0)*quad_rel_size
                               , root_cells_dim.at(1)*quad_rel_size
                             #ifdef P4_TO_P8
                               , root_cells_dim.at(2)*quad_rel_size
                             #endif
                              };
  double  xyz_c[P4EST_DIM] = {xmin_tree + root_cells_dim.at(0)*(((double) (quad->x))/((double)P4EST_ROOT_LEN)) + 0.5*dxdydz[0]
                              , ymin_tree + root_cells_dim.at(1)*(((double) (quad->y))/((double)P4EST_ROOT_LEN)) + 0.5*dxdydz[1]
                            #ifdef P4_TO_P8
                              , zmin_tree + root_cells_dim.at(2)*(((double) (quad->z))/((double)P4EST_ROOT_LEN)) + 0.5*dxdydz[2]
                            #endif
                             };
  if(parent_list_ptr == NULL)
  {
    parent_list_ptr = reduced_list_ptr(new reduced_list);
    reduced_list& par_list = *parent_list_ptr;
    for (int mol_idx = 0; mol_idx < nmol(); ++mol_idx) {
      const molecule& mol = bio_molecules.at(mol_idx);
      const double* mol_centroid = mol.get_centroid();
      const double mol_bounding_box = mol.get_side_length_of_bounding_cube();
      const double xyz_box[P4EST_DIM]  = {(xyz_c[0] > mol_centroid[0]+0.5*mol_bounding_box)?mol_centroid[0]+0.5*mol_bounding_box:(xyz_c[0] < mol_centroid[0]-0.5*mol_bounding_box)?mol_centroid[0]-0.5*mol_bounding_box:xyz_c[0]
                                          , (xyz_c[1] > mol_centroid[1]+0.5*mol_bounding_box)?mol_centroid[1]+0.5*mol_bounding_box:(xyz_c[1] < mol_centroid[1]-0.5*mol_bounding_box)?mol_centroid[1]-0.5*mol_bounding_box:xyz_c[1]
                                    #ifdef P4_TO_P8
                                          , (xyz_c[2] > mol_centroid[2]+0.5*mol_bounding_box)?mol_centroid[2]+0.5*mol_bounding_box:(xyz_c[2] < mol_centroid[2]-0.5*mol_bounding_box)?mol_centroid[2]-0.5*mol_bounding_box:xyz_c[2]
                                    #endif
                                         };
      const double xyz_quad[P4EST_DIM] = {(xyz_box[0] > xyz_c[0]+0.5*dxdydz[0])?xyz_c[0]+0.5*dxdydz[0]:(xyz_box[0] < xyz_c[0]-0.5*dxdydz[0])?xyz_c[0]-0.5*dxdydz[0]:xyz_box[0]
                                          , (xyz_box[1] > xyz_c[1]+0.5*dxdydz[1])?xyz_c[1]+0.5*dxdydz[1]:(xyz_box[1] < xyz_c[1]-0.5*dxdydz[1])?xyz_c[1]-0.5*dxdydz[1]:xyz_box[1]
                                    #ifdef P4_TO_P8
                                          , (xyz_box[2] > xyz_c[2]+0.5*dxdydz[2])?xyz_c[2]+0.5*dxdydz[2]:(xyz_box[2] < xyz_c[2]-0.5*dxdydz[2])?xyz_c[2]-0.5*dxdydz[2]:xyz_box[2]
                                    #endif
                                         };
      if(sqrt(SQR(xyz_box[0]-xyz_quad[0]) + SQR(xyz_box[1]-xyz_quad[1])
        #ifdef P4_TO_P8
              + SQR(xyz_box[2]-xyz_quad[2])
        #endif
              ) <= MAX(parameters.layer_thickness() + (need_exact_phi? parameters.probe_radius(): 0.0), 0.5*parameters.lip_cst()*parameters.root_diag()*quad_rel_size - parameters.probe_radius())) // there might be relevant atoms in the molecule to consider
      {
        size_t former_nb_atoms = par_list.size();
        par_list.atom_global_idx.resize(former_nb_atoms+mol.get_number_of_atoms());
        for (int jj = 0; jj < mol.get_number_of_atoms(); ++jj)
          par_list.atom_global_idx.at(former_nb_atoms+jj) = atom_index_offset.at(mol_idx) + jj;
      }
    }
    if(par_list.size() == 0)
    {
      par_list.atom_global_idx.push_back(0);
      reduced_lists.push_back(parent_list_ptr);
      return;
    }
  }

  double  threshold_dist = MAX(parameters.layer_thickness() + (need_exact_phi? parameters.probe_radius(): 0.0), 0.25*parameters.lip_cst()*parameters.root_diag()*quad_rel_size - parameters.probe_radius());
  const reduced_list& parent_list = (*parent_list_ptr);

  int     nb_added_global_idx = 0;
  int     global_idx_maximizing_atom = -1; // absurd initialization
  double  max_phi_sas_to_vertices = -DBL_MAX;
  int     mol_index = 0;

  reduced_list_ptr child_list_ptr(new reduced_list);
  reduced_list& child_list = *child_list_ptr;

  for (int k = 0; k < parent_list.size(); ++k) {
    int global_atom_idx = parent_list.atom_global_idx.at(k);
    const Atom* a = get_atom(global_atom_idx, mol_index);
    double d = parameters.probe_radius() + a->max_phi_vdW_in_quad(xyz_c, dxdydz);
    if(d >= -threshold_dist - EPS*parameters.probe_radius()) // -0.5*parameters.root_diag()/(1<<parameters.max_level()))
    {
      child_list.atom_global_idx.push_back(global_atom_idx);
      nb_added_global_idx++;
    }
    if(nb_added_global_idx == 0 && d >= MAX(parameters.probe_radius()-0.5*parameters.lip_cst()*parameters.root_diag()*quad_rel_size, max_phi_sas_to_vertices))
    {
      double phi_sas_vertex;
      for (int ii = 0; ii < 2; ++ii) {
        for (int jj = 0; jj < 2; ++jj) {
#ifdef P4_TO_P8
          for (int kk = 0; kk < 2; ++kk) {
#endif
            phi_sas_vertex = parameters.probe_radius() + a->dist_to_vdW_surface(xyz_c[0] + ((double) ii - 0.5)*dxdydz[0], xyz_c[1] + ((double) jj - 0.5)*dxdydz[1]
    #ifdef P4_TO_P8
                , xyz_c[2] + ((double) kk - 0.5)*dxdydz[2]
    #endif
                );
            if(phi_sas_vertex > max_phi_sas_to_vertices)
            {
              max_phi_sas_to_vertices = phi_sas_vertex;
              global_idx_maximizing_atom = global_atom_idx;
            }
#ifdef P4_TO_P8
          }
#endif
        }
      }
    }
  }

  if(nb_added_global_idx == 0)
    child_list.atom_global_idx.push_back((global_idx_maximizing_atom>=0)?global_idx_maximizing_atom:parent_list.atom_global_idx.at(0));

  reduced_lists.push_back(child_list_ptr);

  return;
}

double my_p4est_biomolecules_t::operator ()(const double x, const double y
                                            #ifdef P4_TO_P8
                                            , const double z
                                            #endif
                                            ) const
{
  double phi = -DBL_MAX;
  for (int k = 0; k < nmol(); ++k) {
    const molecule& mol_k = bio_molecules.at(k);
    phi = MAX(phi, mol_k(x, y
                     #ifdef P4_TO_P8
                         , z
                     #endif
                         ));
  }
  phi += parameters.probe_radius();
  return phi;
}


double my_p4est_biomolecules_t::reduced_operator(const double *xyz, const int& reduced_list_idx, const bool need_exact_value, const bool last_stage) const
{
  bool get_better_phi = false;
  double phi = -DBL_MAX, tmp;
  const double zero_threshold = parameters.layer_thickness()*((parameters.order_of_accuracy()==1)? 0.01:(1.0/(1<<parameters.max_level())));
  int mol_idx = 0;
  const Atom *atom_i = NULL;
  int reduced_index_of_atom_i = -1;

  const reduced_list& r_list = *(reduced_lists.at(reduced_list_idx));
  for (int k = 0; k < r_list.size(); ++k) {
    const Atom* a = get_atom(r_list.atom_global_idx.at(k), mol_idx);
    tmp = a->dist_to_vdW_surface(xyz);
    if(tmp > phi)
    {
      phi     = tmp;
      if(need_exact_value && (global_max_level >= parameters.threshold_level()))
      {
        atom_i  = a;
        reduced_index_of_atom_i = k;
      }
    }
  }
  phi += parameters.probe_radius();
  if(need_exact_value && global_max_level >= parameters.threshold_level() && 0.0 <= phi && phi <= parameters.probe_radius() + parameters.layer_thickness() + zero_threshold)
  {
    double dist_xyz_to_c_i = sqrt(SQR(xyz[0] - atom_i->x) + SQR(xyz[1] - atom_i->y)
    #ifdef P4_TO_P8
        + SQR(xyz[2] - atom_i->z)
    #endif
        );
    if(dist_xyz_to_c_i > zero_threshold)
    {
      double xyz_proj_i[P4EST_DIM]; // projection on \delta B(c_i, r_p + ri)
      xyz_proj_i[0] = xyz[0] + (xyz[0] - atom_i->x)*phi/dist_xyz_to_c_i;
      xyz_proj_i[1] = xyz[1] + (xyz[1] - atom_i->y)*phi/dist_xyz_to_c_i;
#ifdef P4_TO_P8
      xyz_proj_i[2] = xyz[2] + (xyz[2] - atom_i->z)*phi/dist_xyz_to_c_i;
#endif
      tmp = -DBL_MAX;
      mol_idx = 0;
      get_better_phi = false;
      for (int k = 0; k < r_list.size(); ++k) {
        if(k == reduced_index_of_atom_i)
          continue;
        const Atom* a = get_atom(r_list.atom_global_idx.at(k), mol_idx);
        tmp = MAX(tmp, a->dist_to_vdW_surface(xyz_proj_i) + parameters.probe_radius());
        if(tmp > zero_threshold)
        {
          get_better_phi = true;
          break;
        }
      }
    }
  }
  if(get_better_phi && !last_stage)
    phi = -1.5*parameters.root_diag();

  return phi;
}

double my_p4est_biomolecules_t::better_distance(const double *xyz, const int& reduced_list_idx, double* kink_point) const
{
  double phi  = -DBL_MAX, tmp;
  int mol_idx = 0;

  const double zero_threshold = parameters.layer_thickness()*((parameters.order_of_accuracy()==1)? 0.01:(1.0/(1<<parameters.max_level())));
  vector<sorted_atom> sorted_atoms;
  const reduced_list& r_list = *(reduced_lists.at(reduced_list_idx));
  sorted_atoms.resize(r_list.size());

  for (int k = 0; k < r_list.size(); ++k) {
    const Atom* a = get_atom(r_list.atom_global_idx.at(k), mol_idx);
    tmp = a->dist_to_vdW_surface(xyz) + parameters.probe_radius();
    phi = MAX(tmp, phi);
    sorted_atoms[k].global_atom_idx       = r_list.atom_global_idx.at(k);
    sorted_atoms[k].mol_idx               = mol_idx;
    sorted_atoms[k].distance_from_xyz     = tmp;
    sorted_atoms[k].distance_from_xyz_i   = 0.0;
    sorted_atoms[k].distance_from_graal   = 0.0;
  }

  if(r_list.size() == 1)
    return phi;

  const Atom *atom_i = NULL;
  const Atom* atom_j = NULL;
#ifdef P4_TO_P8
  const Atom* atom_k = NULL;
#endif
  double distance_to_kink = 0.0;
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    distance_to_kink += SQR(xyz[dim] - kink_point[dim]);
  distance_to_kink = sqrt(distance_to_kink);
  double graal_point[P4EST_DIM];
  double closest_graal[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    closest_graal[dim] = DBL_MAX;
  double distance_to_closest_graal = parameters.root_diag(); //1.5*(get_largest_radius_of_all()+parameters.probe_radius());
  bool graal_point_is_valid = false;

  for (size_t ii = 0; ii < sorted_atoms.size(); ++ii) {
    sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
      return (atom_a.distance_from_xyz > atom_b.distance_from_xyz);
    });
    atom_i = get_atom(sorted_atoms[ii].global_atom_idx, sorted_atoms[ii].mol_idx);
    if(sorted_atoms[ii].distance_from_xyz < ((parameters.probe_radius() + atom_i->r)*(1.0-sqrt(1.0 + SQR((parameters.probe_radius() + parameters.layer_thickness())/(parameters.probe_radius() + atom_i->r))))))
      continue; // might be the right atom but its irrelevant since the point is farther than parameters.layer_thickness() away from \Gamma_{\SES} in this case
    if(fabs(atom_i->dist_to_vdW_surface(xyz) + parameters.probe_radius()) > MIN(distance_to_kink, distance_to_closest_graal))
      continue;
    double dist_xyz_to_c_i = sqrt(SQR(xyz[0] - atom_i->x) + SQR(xyz[1] - atom_i->y)
    #ifdef P4_TO_P8
        + SQR(xyz[2] - atom_i->z)
    #endif
        );
    double xyz_proj_i[P4EST_DIM]; // projection on \delta B(c_i, r_p + ri)
    if(dist_xyz_to_c_i > zero_threshold)
    {
      xyz_proj_i[0] = xyz[0] + (xyz[0] - atom_i->x)*sorted_atoms[ii].distance_from_xyz/dist_xyz_to_c_i;
      xyz_proj_i[1] = xyz[1] + (xyz[1] - atom_i->y)*sorted_atoms[ii].distance_from_xyz/dist_xyz_to_c_i;
#ifdef P4_TO_P8
      xyz_proj_i[2] = xyz[2] + (xyz[2] - atom_i->z)*sorted_atoms[ii].distance_from_xyz/dist_xyz_to_c_i;
#endif
    }
    else
    {
      double xyz_to_kink[P4EST_DIM];
      double xyz_to_kink_norm = 0.0;
      for (short dim = 0; dim < P4EST_DIM; ++dim)
      {
        xyz_to_kink[dim] = kink_point[dim] - xyz[dim];
        xyz_to_kink_norm += SQR(xyz_to_kink[dim]);
      }
      xyz_to_kink_norm = sqrt(xyz_to_kink_norm);
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        xyz_proj_i[dim] = xyz[dim] + (atom_i->r + parameters.probe_radius())*xyz_to_kink[dim]/xyz_to_kink_norm;
    }
    double distance_to_projected_point_i = sorted_atoms[ii].distance_from_xyz;

    graal_point_is_valid = true;
    double phi_sas_analytical_at_xyz_i = -DBL_MAX;
    for (size_t jj = 0; jj < sorted_atoms.size(); ++jj){
      const Atom* a = get_atom(sorted_atoms[jj].global_atom_idx, sorted_atoms[jj].mol_idx);
      sorted_atoms[jj].distance_from_xyz_i   = a->dist_to_vdW_surface(xyz_proj_i) + parameters.probe_radius();
      phi_sas_analytical_at_xyz_i = MAX(phi_sas_analytical_at_xyz_i, sorted_atoms[jj].distance_from_xyz_i);
      graal_point_is_valid = graal_point_is_valid && (sorted_atoms[jj].distance_from_xyz_i < zero_threshold);
      sorted_atoms[jj].distance_from_graal   = 0.0;
    }
    graal_point_is_valid = graal_point_is_valid && (fabs(phi_sas_analytical_at_xyz_i) < zero_threshold);
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      graal_point[dim] = xyz_proj_i[dim];
    double distance_xyz_to_graal = distance_to_projected_point_i;

    if(!graal_point_is_valid)
    {
      for (size_t jj = 0; jj < sorted_atoms.size(); ++jj){
        sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
          return (atom_a.distance_from_xyz_i > atom_b.distance_from_xyz_i);
        });

        atom_j = get_atom(sorted_atoms[jj].global_atom_idx, sorted_atoms[jj].mol_idx);
        if(fabs(atom_j->dist_to_vdW_surface(xyz) + parameters.probe_radius()) > MIN(distance_to_kink, distance_to_closest_graal))
          continue;
        const double alpha = sqrt(SQR(atom_i->x - atom_j->x) + SQR(atom_i->y - atom_j->y)
                            #ifdef P4_TO_P8
                                  + SQR(atom_i->z - atom_j->z)
                            #endif
                                  );
        const double lambda = 0.5 + 0.5*(SQR(parameters.probe_radius() + atom_j->r) - SQR(parameters.probe_radius() + atom_i->r))/SQR(alpha);
        double circle_center[P4EST_DIM];
        double normal_vector[P4EST_DIM];
        normal_vector[0]    = (atom_j->x - atom_i->x)/alpha;
        circle_center[0]    = lambda*atom_i->x + (1.0-lambda)*atom_j->x;

        normal_vector[1]    = (atom_j->y - atom_i->y)/alpha;
        circle_center[1]    = lambda*atom_i->y + (1.0-lambda)*atom_j->y;

      #ifdef P4_TO_P8
        normal_vector[2]    = (atom_j->z - atom_i->z)/alpha;
        circle_center[2]    = lambda*atom_i->z + (1.0-lambda)*atom_j->z;
      #endif
        const double circle_radius  = sqrt(0.5*(SQR(parameters.probe_radius() + atom_i->r) + SQR(parameters.probe_radius() + atom_j->r))
                                           - 0.25*SQR(alpha)
                                           - 0.25*(SQR(SQR(parameters.probe_radius() + atom_j->r) - SQR(parameters.probe_radius() + atom_i->r)))/SQR(alpha));

        double mu_vector[P4EST_DIM];
        double projection_along_normal = 0.0;
        for (short k = 0; k < P4EST_DIM; ++k) {
          mu_vector[k] = xyz[k] - circle_center[k];
          projection_along_normal += normal_vector[k]*mu_vector[k];
        }
        double norm_of_mu = 0.0;
        for (short k = 0; k < P4EST_DIM; ++k)
        {
          mu_vector[k] = mu_vector[k] - projection_along_normal*normal_vector[k];
          norm_of_mu += SQR(mu_vector[k]);
        }
        norm_of_mu = sqrt(norm_of_mu);
        if(norm_of_mu > zero_threshold)
        {
          for (short k = 0; k < P4EST_DIM; ++k)
            mu_vector[k] /= norm_of_mu;
        }
        else
        {
          // choose arbitrarily
      #ifndef P4_TO_P8
          mu_vector[0] =  normal_vector[1];
          mu_vector[1] = -normal_vector[0];
      #else
          double nnn;
          if(fabs(normal_vector[1]) >= fabs(normal_vector[0]))
          {
            nnn = sqrt(SQR(normal_vector[1]) + SQR(normal_vector[2]));
            mu_vector[0] = 0.0;
            mu_vector[1] = -normal_vector[2]/nnn;
            mu_vector[2] = normal_vector[1]/nnn;
          }
          else
          {
            nnn = sqrt(SQR(normal_vector[0]) + SQR(normal_vector[2]));
            mu_vector[0] = normal_vector[2]/nnn;
            mu_vector[1] = 0.0;
            mu_vector[2] = -normal_vector[0]/nnn;
          }
      #endif
        }
        norm_of_mu = 1.0;
  #ifdef P4_TO_P8
        double nu_vector[P4EST_DIM];
        nu_vector[0] = normal_vector[1]*mu_vector[2] - normal_vector[2]*mu_vector[1];
        nu_vector[1] = normal_vector[2]*mu_vector[0] - normal_vector[0]*mu_vector[2];
        nu_vector[2] = normal_vector[0]*mu_vector[1] - normal_vector[1]*mu_vector[0];
  #endif
  //      distance_xyz_to_graal = 0.0;
        double theta_angles[P4EST_DIM-1];
        double graal_theta;
        theta_angles[0] = 0.0; // positive
  #ifdef P4_TO_P8
        bool take_first_angle = true;
        theta_angles[1] = 0.0; // negative
        double cc[P4EST_DIM]; // vector pointing from circle_center to the center of the intersection
        // between the other ball and the plane normal to the circle under investigation
        double cc_dot_n = 0.0, norm_of_cc = 0.0;
        double cc_dot_mu = 0.0, cc_dot_nu = 0.0;
        double radius_of_other_circle = 0.0;
        double angle_beta, angle_alpha;
  #endif
        bool set_of_candidates_is_empty = false;

        while(!graal_point_is_valid && !set_of_candidates_is_empty)
        {
  #ifdef P4_TO_P8
          set_of_candidates_is_empty = (fabs(theta_angles[0]-theta_angles[1] - 2.0*M_PI) <= EPS*M_PI);
          take_first_angle = (fabs(theta_angles[0]) <= fabs(theta_angles[1]));
          graal_theta = theta_angles[(take_first_angle?0:1)];
  #else
          set_of_candidates_is_empty = (fabs(theta_angles[0] - M_PI) <= EPS*M_PI);
          graal_theta = theta_angles[0];
  #endif
          distance_xyz_to_graal = 0.0;
          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            graal_point[dim] = circle_center[dim] + cos(graal_theta)*circle_radius*mu_vector[dim]
          #ifdef P4_TO_P8
                + sin(graal_theta)*circle_radius*nu_vector[dim]
          #endif
                ;
            distance_xyz_to_graal += SQR(xyz[dim] - graal_point[dim]);
          }
          distance_xyz_to_graal = sqrt(distance_xyz_to_graal);
          if( distance_xyz_to_graal > MIN(distance_to_closest_graal, distance_to_kink))
            break;
          for (size_t kk = 0; kk < sorted_atoms.size(); ++kk){
            const Atom* a = get_atom(sorted_atoms[kk].global_atom_idx, sorted_atoms[kk].mol_idx);
            sorted_atoms[kk].distance_from_graal  = a->dist_to_vdW_surface(graal_point) + parameters.probe_radius();
          }
          sort(sorted_atoms.begin(), sorted_atoms.end(), [](sorted_atom atom_a, sorted_atom atom_b){
            return (atom_a.distance_from_graal > atom_b.distance_from_graal);
          });
          graal_point_is_valid = (fabs(sorted_atoms[0].distance_from_graal) < zero_threshold);
          if(!graal_point_is_valid)
          {
  #ifndef P4_TO_P8
            theta_angles[0] = M_PI;
  #else
            atom_k = get_atom(sorted_atoms[0].global_atom_idx, sorted_atoms[0].mol_idx);
            cc_dot_n = 0.0;
            cc[0] = atom_k->x - circle_center[0];
            cc_dot_n += cc[0]*normal_vector[0];
            cc[1] = atom_k->y - circle_center[1];
            cc_dot_n += cc[1]*normal_vector[1];
            cc[2] = atom_k->z - circle_center[2];
            cc_dot_n += cc[2]*normal_vector[2];
            radius_of_other_circle = sqrt(SQR(parameters.probe_radius() + atom_k->r) - SQR(cc_dot_n));
            norm_of_cc = 0.0;
            cc_dot_mu = 0.0;
            cc_dot_nu = 0.0;
            for (short dim = 0; dim < P4EST_DIM; ++dim)
            {
              cc[dim] = cc[dim] - cc_dot_n*normal_vector[dim];
              norm_of_cc += SQR(cc[dim]);
              cc_dot_mu += cc[dim]*mu_vector[dim];
              cc_dot_nu += cc[dim]*nu_vector[dim];
            }
            norm_of_cc = sqrt(norm_of_cc);
            if(radius_of_other_circle >= norm_of_cc + circle_radius) // no intersection, investigated circle entirely contained
            {
              theta_angles[0] = M_PI;
              theta_angles[1] = -M_PI;
              set_of_candidates_is_empty = true;
            }
            else
            {
              angle_beta  = acos(cc_dot_mu/norm_of_cc)*((cc_dot_nu >= 0)? +1.0:-1.0);
              angle_alpha = acos((SQR(circle_radius) + SQR(norm_of_cc) - SQR(radius_of_other_circle))/(2.0*circle_radius*norm_of_cc));
              theta_angles[0] = MAX(theta_angles[0], MIN(angle_beta + angle_alpha, 2.0*M_PI + theta_angles[1]));
              theta_angles[1] = MIN(theta_angles[1], MAX(angle_beta - angle_alpha, - (2.0*M_PI - theta_angles[0])));
            }
    #endif
          }
        }
        if(graal_point_is_valid && distance_xyz_to_graal < distance_to_closest_graal)
        {
          distance_to_closest_graal = distance_xyz_to_graal;
          for (short dim = 0; dim < P4EST_DIM; ++dim)
            closest_graal[dim] = graal_point[dim];
        }
        for (short dim = 0; dim < P4EST_DIM; ++dim)
          graal_point[dim] = xyz_proj_i[dim];
        graal_point_is_valid = false;
        distance_xyz_to_graal = distance_to_projected_point_i;
      }
    }
    else
    {
      if(distance_xyz_to_graal < distance_to_closest_graal)
      {
        distance_to_closest_graal = distance_xyz_to_graal;
        for (short dim = 0; dim < P4EST_DIM; ++dim)
          closest_graal[dim] = graal_point[dim];
      }
    }
  }
  if(distance_to_closest_graal < distance_to_kink && distance_to_closest_graal < parameters.probe_radius())
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      kink_point[dim] = closest_graal[dim];
  }
  return MIN(distance_to_closest_graal, distance_to_kink);
}

void my_p4est_biomolecules_t::build_brick()
{
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  brick->xyz_min[0] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 0];
  brick->xyz_min[1] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 1];
#ifdef P4_TO_P8
  brick->xyz_min[2] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 2];
#else
  brick->xyz_min[2] = 0.0;
#endif

  brick->xyz_max[0] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->connectivity->num_trees - 1) + P4EST_CHILDREN-1] + 0];
  brick->xyz_max[1] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->connectivity->num_trees - 1) + P4EST_CHILDREN-1] + 1];
#ifdef P4_TO_P8
  brick->xyz_max[2] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->connectivity->num_trees - 1) + P4EST_CHILDREN-1] + 2];
#else
  brick->xyz_max[2] = 0.0;
#endif

  set<int> x_tree; x_tree.clear();
  set<int> y_tree; y_tree.clear();
#ifdef P4_TO_P8
  set<int> z_tree; z_tree.clear();
#endif
  for (p4est_topidx_t tt = 0; tt < p4est->connectivity->num_trees; ++tt) {
    x_tree.insert((int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 0] - brick->xyz_min[0])/root_cells_dim.at(0)));
    y_tree.insert((int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 1] - brick->xyz_min[1])/root_cells_dim.at(1)));
#ifdef P4_TO_P8
    z_tree.insert((int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 2] - brick->xyz_min[2])/root_cells_dim.at(2)));
#endif
  }
  brick->nxyztrees[0] = x_tree.size();
  brick->nxyztrees[1] = y_tree.size();
#ifdef P4_TP_P8
  brick->nxyztrees[2] = z_tree.size();
#else
  brick->nxyztrees[2] = 1;
#endif

  brick->nxyz_to_treeid = P4EST_ALLOC (p4est_topidx_t, p4est->connectivity->num_trees);
  for (p4est_topidx_t tt = 0; tt < p4est->connectivity->num_trees; ++tt) {
    int i = (int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 0] - brick->xyz_min[0])/root_cells_dim.at(0));
    int j = (int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 1] - brick->xyz_min[1])/root_cells_dim.at(1));
#ifdef P4_TO_P8
    int k = (int)((p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + 0] + 2] - brick->xyz_min[2])/root_cells_dim.at(2));
    brick->nxyz_to_treeid[brick->nxyztrees[0]*brick->nxyztrees[1]*k + brick->nxyztrees[0]*j + i] = tt;
#else
    brick->nxyz_to_treeid[brick->nxyztrees[0]*j + i] = tt;
#endif
  }
}

void my_p4est_biomolecules_t::partition_uniformly(const bool export_cavities, const bool build_ghost)
{
#ifdef CASL_THROWS
  int local_error = (p4est == NULL);
  string err_msg = "my_p4est_biomolecules_t::partition_uniformly(): this function requires a valid p4est object.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = (phi != NULL && nodes == NULL);
  err_msg = "my_p4est_biomolecules_t::partition_uniformly(): this function requires valid nodes if the levelset function is defined.";
  err_manager.check_my_local_error(local_error, err_msg);

  if(phi != NULL)
  {
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    PetscInt size;
    PetscErrorCode ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
    local_error = size != total_nb_nodes;
    err_msg = "my_p4est_biomolecules_t::partition_uniformly(): the size of the levelset vector is not equal to the total number of nodes.";
    err_manager.check_my_local_error(local_error, err_msg);
  }
#endif
  my_p4est_partition(p4est, P4EST_FALSE, NULL); // do not allow for coarsening, it's the last stage...

  if(ghost != NULL)
  {
    p4est_ghost_destroy(ghost); ghost = NULL;
  }
  if(build_ghost)
    ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  if(nodes != NULL)
  {
    if(phi == NULL)
    {
      p4est_nodes_destroy(nodes);
      nodes = my_p4est_nodes_new(p4est, ghost);
    }
    else
    {
      PetscInt global_idx_offset = 0;
      for (int rank = 0; rank < p4est->mpirank; ++rank)
        global_idx_offset += nodes->global_owned_indeps[rank];
      vector<PetscInt> global_indices_of_known_values; global_indices_of_known_values.resize(nodes->num_owned_indeps);
      for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k)
        global_indices_of_known_values.at(k) = global_idx_offset + k;

      int nb_indices = (int) nodes->num_owned_indeps;
      const PetscInt* set_of_global_indices = (nb_indices > 0)? (const PetscInt*) &global_indices_of_known_values.at(0) : PETSC_NULL;

      p4est_nodes_destroy(nodes);
      nodes = my_p4est_nodes_new(p4est, ghost);

      Vec old_phi = phi; // no cost, those are pointers...
      PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
      Vec old_inner_domain = NULL;
      if(export_cavities && inner_domain != NULL)
      {
        old_inner_domain = inner_domain;
        ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);
      }

      IS is;
      ierr    = ISCreateGeneral(p4est->mpicomm, nb_indices, set_of_global_indices, PETSC_USE_POINTER, &is); CHKERRXX(ierr);
      VecScatter ctx, ctx_inner_domain;
      ierr    = VecScatterCreate(old_phi, is, phi, is, &ctx); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecScatterCreate(old_inner_domain, is, inner_domain, is, &ctx_inner_domain); CHKERRXX(ierr);}
      ierr    = VecScatterBegin(ctx, old_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecScatterBegin(ctx_inner_domain, old_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}
      ierr    = VecScatterEnd(ctx, old_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecScatterEnd(ctx_inner_domain, old_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}
      ierr    = VecDestroy(old_phi); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecDestroy(old_inner_domain); CHKERRXX(ierr);}
      ierr    = VecScatterDestroy(ctx); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecScatterDestroy(ctx_inner_domain); CHKERRXX(ierr);}
      ierr    = ISDestroy(is); CHKERRXX(ierr);

      ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}
      ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(export_cavities && inner_domain != NULL){
        ierr  = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}
    }
    if(build_ghost && neighbors != NULL)
    {
      neighbors->update(hierarchy, nodes);
      ls->update(neighbors);
    }
  }
}

int my_p4est_biomolecules_t::partition_weight_for_enforcing_min_level(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  (void) which_tree;
  my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  if(quadrant->level < biomol->parameters.min_level())
    return (1<<(P4EST_DIM*(biomol->parameters.min_level()-quadrant->level)));
  else
    return 1;
}

void my_p4est_biomolecules_t::enforce_min_level(bool export_cavities)
{
#ifdef CASL_THROWS
  int local_error = (p4est == NULL);
  string err_msg = "my_p4est_biomolecules_t::enforce_min_level_and_partition_uniformly(): this function requires a valid p4est object.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = (phi == NULL || nodes == NULL);
  err_msg = "my_p4est_biomolecules_t::enforce_min_level_and_partition_uniformly(): this function requires valid nodes and a valid node-sampled levelset function.";
  err_manager.check_my_local_error(local_error, err_msg);

  if(phi != NULL)
  {
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    PetscInt size;
    PetscErrorCode ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
    local_error = size != total_nb_nodes;
    err_msg = "my_p4est_biomolecules_t::enforce_min_level_and_partition_uniformly(): the size of the node-sampled levelset function is not equal to the total number of nodes.";
    err_manager.check_my_local_error(local_error, err_msg);
  }
#endif
  P4EST_ASSERT((export_cavities && inner_domain != NULL) || (!export_cavities && inner_domain == NULL));
  if(ghost != NULL){
    p4est_ghost_destroy(ghost); ghost = NULL;}
  vector<PetscInt> global_indices;
  global_indices.resize(nodes->num_owned_indeps);
  PetscInt global_idx_offset = 0;
  for (int rank = 0; rank < p4est->mpirank; ++rank)
    global_idx_offset += nodes->global_owned_indeps[rank];
  for (int k = 0; k < nodes->num_owned_indeps; ++k)
    global_indices.at(k) = global_idx_offset + k;
  p4est_nodes_destroy(nodes); nodes = NULL;
  my_p4est_partition(p4est, P4EST_FALSE, my_p4est_biomolecules_t::partition_weight_for_enforcing_min_level);
  // no need to rebuild the ghost right NOW, this is a very local process,
  // build them afterwards if needed
  nodes = my_p4est_nodes_new(p4est, ghost);

  // scatter the vector(s) to the new layout
  // store old vectors of phi (and cavities)
  Vec former_phi, former_inner_domain;
  P4EST_ASSERT(phi_read_only_p == NULL && phi_p == NULL);
  former_phi = phi; phi = NULL;
  if(export_cavities){
    former_inner_domain = inner_domain; inner_domain = NULL;}

  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  if(export_cavities){
    ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);}
#ifdef CASL_THROWS
  PetscInt size1, size2;
  ierr = VecGetSize(former_phi, &size1); CHKERRXX(ierr);
  ierr = VecGetSize(phi, &size2); CHKERRXX(ierr);
  local_error = size1 != size2;
  err_msg = "my_p4est_biomolecules_t::enforce_min_level(): logic error, inconsistent number of points after partitioning...";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  VecScatter ctx, ctx_inner_domain;
  IS is;

  int nb_indices = (int) global_indices.size();
  const PetscInt* index_set = (nb_indices > 0)? (const PetscInt*) &global_indices.at(0) : PETSC_NULL;
  ierr    = ISCreateGeneral(p4est->mpicomm, nb_indices, index_set, PETSC_USE_POINTER, &is); CHKERRXX(ierr);
  ierr    = VecScatterCreate(former_phi, is, phi, is, &ctx); CHKERRXX(ierr);
  if(export_cavities){
    ierr  = VecScatterCreate(former_inner_domain, is, inner_domain, is, &ctx_inner_domain); CHKERRXX(ierr);}
  ierr    = VecScatterBegin(ctx, former_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(export_cavities){
    ierr  = VecScatterBegin(ctx_inner_domain, former_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }
  ierr    = VecScatterEnd(ctx, former_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr    = VecDestroy(former_phi); CHKERRXX(ierr);
  ierr    = VecScatterDestroy(ctx); CHKERRXX(ierr);
  if(export_cavities)
  {
    ierr  = VecScatterEnd(ctx_inner_domain, former_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr  = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr  = VecDestroy(former_inner_domain); former_inner_domain = NULL; CHKERRXX(ierr);
    ierr  = VecScatterDestroy(ctx_inner_domain); CHKERRXX(ierr);
    ierr  = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr    = ISDestroy(is); CHKERRXX(ierr);

  // impose the minimum level
  p4est_nodes_t* coarse_nodes = nodes;
  Vec coarse_phi              = phi;
  const double* coarse_phi_read_only_p = NULL, * coarse_inner_domain_read_only_p = NULL;
  ierr    = VecGetArrayRead(coarse_phi, &coarse_phi_read_only_p); CHKERRXX(ierr);
  Vec coarse_inner_domain     = NULL;
  if(export_cavities)
  {
    coarse_inner_domain       = inner_domain;
    ierr  = VecGetArrayRead(coarse_inner_domain, &coarse_inner_domain_read_only_p); CHKERRXX(ierr);
  }
  // enforce the minimum level
  p4est_refine_ext(p4est, P4EST_TRUE, -1, my_p4est_biomolecules_t::refine_fn_min_level, NULL, my_p4est_biomolecules_t::replace_fn_min_level);
  // Fill missing data by linear interpolation:
  // create the nodes
  nodes   = my_p4est_nodes_new(p4est, ghost);
  // create the vector(s)
  ierr    = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr    = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  double * inner_domain_p;
  if(export_cavities)
  {
    ierr  = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);
    ierr  = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
  }

  // set the new layout of phi
  set<p4est_locidx_t> known_fine_indices; known_fine_indices.clear();
  // firt known nodes
  int clamped = 1;
  p4est_locidx_t  fine_idx;
  p4est_locidx_t  coarse_idx = 0;
  p4est_indep_t*  coarse_node = NULL;
  if(coarse_idx < coarse_nodes->num_owned_indeps)
    coarse_node = (p4est_indep_t*) sc_array_index(&coarse_nodes->indep_nodes, coarse_idx);
  for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k)
  {
    p4est_indep_t *node  = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes,k);
    if(coarse_node!= NULL && p4est_node_equal_piggy_fn (node, coarse_node, &clamped))
    {
      if(export_cavities)
        inner_domain_p[k] = coarse_inner_domain_read_only_p[coarse_idx];
      phi_p[k] = coarse_phi_read_only_p[coarse_idx++];
      known_fine_indices.insert(k);
      if(coarse_idx < coarse_nodes->num_owned_indeps)
        coarse_node = (p4est_indep_t*) sc_array_index(&coarse_nodes->indep_nodes, coarse_idx);
      else
        break;
    }
  }

  p4est_locidx_t  coarse_quad_idx = -1; // absurd initialization
  double          coarse_quad_corners[2][P4EST_DIM], fine_xyz[P4EST_DIM]; // back-lower-left, front-upper-right corners of the interpolation cube, point of interest, respectively
  double          coarse_quad_volume = 1.0; // initialization is irrelevant but the compiler complains otherwise

  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      if(quad->level == parameters.min_level() && coarse_quad_idx != (p4est_locidx_t) quad->p.user_long)
      {
        coarse_quad_idx = (p4est_locidx_t) quad->p.user_long;
        node_xyz_fr_n(coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx + 0], p4est, coarse_nodes, &coarse_quad_corners[0][0]);
        node_xyz_fr_n(coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx + P4EST_CHILDREN-1], p4est, coarse_nodes, &coarse_quad_corners[1][0]);
        coarse_quad_volume = (coarse_quad_corners[1][0]-coarse_quad_corners[0][0])*(coarse_quad_corners[1][1]-coarse_quad_corners[0][1])
    #ifdef P4_TO_P8
            *(coarse_quad_corners[1][2]-coarse_quad_corners[0][2])
    #endif
            ;
      }
      for (short nn = 0; nn < P4EST_CHILDREN; ++nn) {
        fine_idx = nodes->local_nodes[P4EST_CHILDREN*(tree_k->quadrants_offset+q) + nn];
        if(known_fine_indices.find(fine_idx) == known_fine_indices.end())
        {
          if((int) quad->level > parameters.min_level())
          {
            if(export_cavities)
              inner_domain_p[fine_idx] = coarse_inner_domain_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*quad->p.user_long + nn]];
            phi_p[fine_idx] = coarse_phi_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*quad->p.user_long + nn]];
          }
          else
          {
            phi_p[fine_idx] = 0.0;
            if(export_cavities)
              inner_domain_p[fine_idx] = 0.0;
            node_xyz_fr_n(fine_idx, p4est, nodes, fine_xyz);
            for (short ii = 0; ii < 2; ++ii)
            {
              for (short jj = 0; jj < 2; ++jj) {
#ifdef P4_TO_P8
                for (short kk = 0; kk < 2; ++kk) {
#endif
                  phi_p[fine_idx] += coarse_phi_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx+
    #ifdef P4_TO_P8
                      4*kk +
    #endif
                      2*jj + ii]]*fabs((fine_xyz[0]-coarse_quad_corners[1-ii][0])*(fine_xyz[1]-coarse_quad_corners[1-jj][1])
    #ifdef P4_TO_P8
                      *(fine_xyz[2]-coarse_quad_corners[1-kk][2])
    #endif
                      );
                  if(export_cavities)
                    inner_domain_p[fine_idx] += coarse_inner_domain_read_only_p[coarse_nodes->local_nodes[P4EST_CHILDREN*coarse_quad_idx+
    #ifdef P4_TO_P8
                        4*kk +
    #endif
                        2*jj + ii]]*fabs((fine_xyz[0]-coarse_quad_corners[1-ii][0])*(fine_xyz[1]-coarse_quad_corners[1-jj][1])
    #ifdef P4_TO_P8
                        *(fine_xyz[2]-coarse_quad_corners[1-kk][2])
    #endif
                        );
#ifdef P4_TO_P8
                }
#endif
              }
            }
            phi_p[fine_idx] /= coarse_quad_volume;
            if(export_cavities)
              inner_domain_p[fine_idx] /= coarse_quad_volume;
          }
          known_fine_indices.insert(fine_idx);
        }
      }
    }
  }

#ifdef CASL_THROWS
  local_error = (known_fine_indices.size()  != nodes->indep_nodes.elem_count);
  err_msg = "my_p4est_biomolecules_t::enforce_min_level_and_partition_uniformly(): some nodes were not updated when enforcing the min level.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif


  ierr    = VecRestoreArrayRead(coarse_phi, &coarse_phi_read_only_p); coarse_phi_read_only_p = NULL; CHKERRXX(ierr);
  ierr    = VecDestroy(coarse_phi); coarse_phi = NULL; CHKERRXX(ierr);
  ierr    = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  // at this stage, all points have been updated, EXCEPT the points that are located exactly at a
  // processor boundary and are T-junctions. Those points have been treated (i.e. points that belong to the side where the biggest quadrants)
  ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  p4est_nodes_destroy(coarse_nodes);
  if(export_cavities)
  {
    ierr  = VecRestoreArrayRead(coarse_inner_domain, &coarse_inner_domain_read_only_p); coarse_inner_domain_read_only_p = NULL; CHKERRXX(ierr);
    ierr  = VecDestroy(coarse_inner_domain); coarse_inner_domain = NULL; CHKERRXX(ierr);
    ierr  = VecRestoreArray(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierr);
  }
  ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void my_p4est_biomolecules_t::replace_fn_min_level(p4est_t *park, p4est_topidx_t which_tree, int num_outgoing, p4est_quadrant_t *outgoing[], int num_incoming, p4est_quadrant_t *incoming[])
{
  (void) num_incoming;
  (void) which_tree;
  if (num_outgoing > 1) {
    // this is coarsening, it should NEVER happend
#ifdef CASL_THROWS
    my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
    string err_msg = "my_p4est_biomolecules_t::replace_fn_min_level(...): invoked coarsening...";
    biomol->err_manager.print_message_and_abort(err_msg, 22446688);
#else
    MPI_Abort(park->mpicomm, 22446688);
#endif
  }
  else {
    /* this is refinement */
    for (short i = 0; i < P4EST_CHILDREN; ++i)
      incoming[i]->p.user_long = outgoing[0]->p.user_long;
    // copy the local quadrant index of the original parent cell for further linear interpolation
  }
}

p4est_bool_t my_p4est_biomolecules_t::refine_fn_min_level(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  (void) which_tree;
  return ((p4est_bool_t) (quad->level < biomol->parameters.min_level()));
}

void my_p4est_biomolecules_t::remove_internal_cavities_poisson(const bool export_cavities)
{
  PetscErrorCode ierr;
#ifdef CASL_THROWS
  int local_error = phi == NULL;
  string err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_poisson(): this method requires a valid levelset.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = nodes == NULL;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_poisson(): this method requires valid nodes.";
  err_manager.check_my_local_error(local_error, err_msg);

  PetscInt size;
  p4est_gloidx_t nb_total_nodes = 0;
  for (int proc_rank = 0; proc_rank < p4est->mpisize; ++proc_rank)
    nb_total_nodes += nodes->global_owned_indeps[proc_rank];
  ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
  local_error = size != nb_total_nodes;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_poisson(): the size of the level set vector is not equal to the total number of nodes.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = !neighbors->is_initialized;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_poisson(): the neighbors mus be initialized.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif

  if(inner_domain != NULL)
  {
    ierr = VecDestroy(inner_domain); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodes(neighbors->p4est, neighbors->nodes, &inner_domain); CHKERRXX(ierr);
  Vec inner_domain_ghost = NULL;
  ierr = VecGhostGetLocalForm(inner_domain, &inner_domain_ghost); CHKERRXX(ierr);
  ierr = VecSet(inner_domain_ghost, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(inner_domain, &inner_domain_ghost); inner_domain_ghost = NULL; CHKERRXX(ierr);

  struct:
    #ifdef P4_TO_P8
      WallBC3D
    #else
      WallBC2D
    #endif
  {
    BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                     , double
                                 #endif
                                     ) const {return DIRICHLET;}
  } bc_wall_type;

  struct:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    double operator()(double, double
                  #ifdef P4_TO_P8
                      , double
                  #endif
                      ) const { return 10.; }
  } bc_wall_value;

  struct:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    double operator()(double, double
                  #ifdef P4_TO_P8
                      , double
                  #endif
                      ) const { return 0.; }
  } bc_interface_value;

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  splitting_criteria_t sp(parameters.min_level(), parameters.max_level(), parameters.lip_cst());
  void* user_pointer_saved = p4est->user_pointer;
  p4est->user_pointer = (void*) &sp;

  my_p4est_poisson_nodes_t solver(neighbors);

  solver.set_bc(bc);
  solver.set_phi(phi);
  solver.set_rhs(inner_domain);
  solver.set_tolerances(1e-6);
  solver.solve(inner_domain);

  // remove the cavities
  double *inner_domain_p;
  double *phi_p;
  ierr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
  {
    inner_domain_p[i] = (phi_p[i] < 0 && fabs(inner_domain_p[i]) < EPS)? 1.0: 0.0;
    if (inner_domain_p[i])// internal cavity
      phi_p[i] = -phi_p[i];
  }

  ierr = VecRestoreArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
  if(!export_cavities)
  {
    ierr = VecDestroy(inner_domain); inner_domain = NULL; CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  p4est->user_pointer = user_pointer_saved;
}

double my_p4est_biomolecules_t::inner_box_identifier::operator ()(double x, double y
                                                                  #ifdef P4_TO_P8
                                                                  , double z
                                                                  #endif
                                                                  )const
{
  bool is_in_a_box = false;
  for (int mol_idx = 0; mol_idx < biomol_pointer->nmol(); ++mol_idx)
  {
    const molecule& mol           = biomol_pointer->bio_molecules.at(mol_idx);
    const double* mol_centroid    = mol.get_centroid();
    const double box_side_length  = mol.get_side_length_of_bounding_cube();
    is_in_a_box = (mol_centroid[0] - 0.5*box_side_length <= x && x <= mol_centroid[0] + 0.5*box_side_length)
        && (mol_centroid[1] - 0.5*box_side_length <= y && y <= mol_centroid[1] + 0.5*box_side_length)
    #ifdef P4_TO_P8
        && (mol_centroid[2] - 0.5*box_side_length <= z && z <= mol_centroid[2] + 0.5*box_side_length)
    #endif
        ;
    if(is_in_a_box)
      break;
  }
  return is_in_a_box? 1.0: 0.0;
}

bool my_p4est_biomolecules_t::is_point_in_outer_domain_and_updated(p4est_locidx_t k, quad_neighbor_nodes_of_node_t& qnnn, const my_p4est_node_neighbors_t* ngbd, double*& inner_domain_p) const
{
  // (inner_domain_p[k] < 0.5) is equivalent to "grid node k was already tagged tagged as member of outer domain"
  if(inner_domain_p[k] < 0.5 || phi_read_only_p[k] >= 0.0)
    return false; // nothing to be done
  ngbd->get_neighbors(k, qnnn);
  inner_domain_p[k] =(
        (inner_domain_p[qnnn.node_m00_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_m00_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_pm] < 0.5)
    #ifdef P4_TO_P8
      ||(inner_domain_p[qnnn.node_m00_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_mm] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_mp] < 0.5)
      ||(inner_domain_p[qnnn.node_m00_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_p00_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_0m0_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_0p0_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_00m_pp] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_pm] < 0.5)
      ||(inner_domain_p[qnnn.node_00p_pp] < 0.5)
    #endif
      )? 0.0 : 1.0;
  return inner_domain_p[k] < 0.5;
}

void my_p4est_biomolecules_t::remove_internal_cavities_region_growing(const bool export_cavities)
{
  PetscErrorCode ierr;
#ifdef CASL_THROWS
  int local_error = phi == NULL;
  string err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_region_growing(): this method requires a valid levelset.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = nodes == NULL;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_region_growing(): this method requires valid nodes.";
  err_manager.check_my_local_error(local_error, err_msg);

  PetscInt size;
  p4est_gloidx_t nb_total_nodes = 0;
  for (int proc_rank = 0; proc_rank < p4est->mpisize; ++proc_rank)
    nb_total_nodes += nodes->global_owned_indeps[proc_rank];
  ierr = VecGetSize(phi, &size); CHKERRXX(ierr);
  local_error = size != nb_total_nodes;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_region_growing(): the size of the level set vector is not equal to the total number of nodes.";
  err_manager.check_my_local_error(local_error, err_msg);

  local_error = !neighbors->is_initialized;
  err_msg = "my_p4est_biomolecules_t::remove_internal_cavities_region_growing(): the neighbors must be initialized.";
  err_manager.check_my_local_error(local_error, err_msg);
#endif

  if(inner_domain != NULL)
  {
    ierr = VecDestroy(inner_domain); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodes(neighbors->p4est, neighbors->nodes, &inner_domain); CHKERRXX(ierr);

  is_point_in_a_bounding_box.biomol_pointer = this;
  sample_cf_on_nodes(p4est, nodes, is_point_in_a_bounding_box, inner_domain);

  double* inner_domain_p;
  ierr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);

  size_t layer_size = neighbors->layer_nodes.size();
  size_t local_size = neighbors->local_nodes.size();
  quad_neighbor_nodes_of_node_t qnnn;
  int not_converged = 1;
  while (not_converged)
  {
    not_converged = 0;

    // forward
    for (size_t layer_node_idx = 0; layer_node_idx < layer_size; ++layer_node_idx)
    {
      p4est_locidx_t k = neighbors->layer_nodes.at(layer_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p) || not_converged;
    }
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t local_node_idx = 0; local_node_idx < local_size; ++local_node_idx)
    {
      p4est_locidx_t k = neighbors->local_nodes.at(local_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p) || not_converged;
    }
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    // backward
    for (size_t layer_node_idx = 0; layer_node_idx < layer_size; ++layer_node_idx)
    {
      p4est_locidx_t k = neighbors->layer_nodes.at(layer_size-1-layer_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p) || not_converged;
    }
    ierr = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t local_node_idx = 0; local_node_idx < local_size; ++local_node_idx)
    {
      p4est_locidx_t k = neighbors->local_nodes.at(local_size-1-local_node_idx);
      not_converged = is_point_in_outer_domain_and_updated(k, qnnn, neighbors, inner_domain_p) || not_converged;
    }
    ierr = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &not_converged, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  }
  ierr = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);

  // remove the cavities
  P4EST_ASSERT(inner_domain_p != NULL);
  P4EST_ASSERT(phi_p == NULL);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
  {
    inner_domain_p[i] = (phi_p[i] <= 0 && inner_domain_p[i] > 0.5)? 1.0: 0.0;
    if (inner_domain_p[i] > 0.5)// internal cavity
      phi_p[i] = -phi_p[i];
  }

  ierr = VecRestoreArray(inner_domain, &inner_domain_p); CHKERRXX(ierr);
  if(!export_cavities)
  {
    ierr = VecDestroy(inner_domain); inner_domain = NULL; CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr); phi_p = NULL;
}

p4est_t* my_p4est_biomolecules_t::construct_SES(const sas_generation_method& method_to_use, const bool SAS_timing_flag, const bool SAS_subtiming_flag, string vtk_folder)
{
  // sanity checks
  check_validity_of_vector_of_mol();
  int local_error = (nodes != NULL);
#ifdef CASL_THROWS
  string err_msg = "my_p4est_biomolecules_t::construct_SES(...): nodes should be NULL when invoked.";
  err_manager.check_my_local_error(local_error, err_msg);
#else
  if(local_error)
  {
    p4est_nodes_destroy(nodes); nodes = NULL;
  }
#endif
  local_error = (ghost != NULL);
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::construct_SES(...): ghost should be NULL when invoked.";
  err_manager.check_my_local_error(local_error, err_msg);
#else
  if(local_error)
  {
    p4est_ghost_destroy(ghost); ghost = NULL;
  }
#endif
  local_error = phi != NULL;
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::construct_SES(...): phi should be NULL when invoked.";
  err_manager.check_my_local_error(local_error, err_msg);
#else
  if(local_error)
  {
    PetscErrorCode ierr = VecDestroy(phi); CHKERRXX(ierr);
    phi_read_only_p = NULL; phi_p = NULL; phi = NULL;
  }
#endif
  local_error = inner_domain != NULL;
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::construct_SES(...): innder_domain should be NULL when invoked.";
  err_manager.check_my_local_error(local_error, err_msg);
#else
  if(local_error)
  {
    PetscErrorCode ierr = VecDestroy(inner_domain); CHKERRXX(ierr);
    inner_domain = NULL;
  }
#endif
  update_max_level();
  local_error = global_max_level > parameters.min_level();
#ifdef CASL_THROWS
  err_msg = "my_p4est_biomolecules_t::construct_SES(...): the p4est is already refined, the method assumes a coarse p4est when invoked.";
  err_manager.check_my_local_error(local_error, err_msg);
#else
  if(local_error)
    p4est = reset_p4est();
#endif
  local_error = p4est->data_size != 0;
  if(local_error)
  {
#ifdef CASL_THROWS
    err_msg = "my_p4est_biomolecules_t::construct_SES(...): we assume no user-defined data, internal management of user-defined data is not handled (yet)...";
    err_manager.check_my_local_error(local_error, err_msg);
#else
    p4est_reset_data(p4est, 0, NULL, p4est->user_pointer);
#endif
  }

  parStopWatch* log_timer = NULL;
  if(log_file != NULL)
  {
    PetscErrorCode ierr;
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Construction of the grid with %d proc(s) \n", p4est->mpisize); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "------------------------------------------- \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of trees: %d \n", p4est->connectivity->num_trees); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Min level: %d \n", parameters.min_level()); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Max level: %d \n", parameters.max_level()); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Proportionality constant L: %lf \n", parameters.lip_cst()); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Probe radius (in A): %lf \n", parameters.probe_radius()/angstrom_to_domain); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Probe radius (in domain): %lf \n", parameters.probe_radius()); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Finest cell diagonal (in domain): %lf \n", parameters.root_diag()/(1<<parameters.max_level())); CHKERRXX(ierr);
    if(timer != NULL && log_timer == NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, log_file, p4est->mpicomm);
      log_timer->start(" CONSTRUCTION OF THE SES grid ");
    }
  }

  if(timer != NULL)
  {
    if(method_to_use != list_reduction_with_exact_phi)
      timer->start("Constructing the SAS and the initial grid");
    else
      timer->start("Constructing the exact distance to the SAS and the initial grid");
  }

  if(sas_creator != NULL)
  {
    bool need_deletion = false;
    switch (method_to_use) {
    case brute_force:
      need_deletion = (dynamic_cast<SAS_creator_brute_force*>(sas_creator) == nullptr);
      break;
    case list_reduction:
      need_deletion = (dynamic_cast<SAS_creator_list_reduction*>(sas_creator) == nullptr);
      break;
    case list_reduction_with_exact_phi:
      need_deletion = (dynamic_cast<SAS_creator_list_reduction*>(sas_creator) == nullptr);
      break;
    default:
#ifdef CASL_THROWS
      err_msg = "my_p4est_biomolecules_t::construct_SES(const sas_generation_method& ): unknown sas generation method...";
      err_manager.print_message_and_abort(err_msg, 265562);
#else
      MPI_Abort(p4est->mpicomm, 265562);
#endif
      break;
    }
    if(need_deletion)
    {
      delete sas_creator; sas_creator = NULL;
    }
  }
  // the p4est must point to a my_p4est_biomolecules_t object for the construction of the SAS
  void* user_pointer_saved = p4est->user_pointer;
  p4est->user_pointer = (void*) this;

  if(sas_creator == NULL)
  {
    switch (method_to_use) {
    case brute_force:
      sas_creator = new SAS_creator_brute_force(p4est, SAS_timing_flag, SAS_subtiming_flag);
      break;
    case list_reduction:
      sas_creator = new SAS_creator_list_reduction(p4est, false, SAS_timing_flag, SAS_subtiming_flag);
      break;
    case list_reduction_with_exact_phi:
      sas_creator = new SAS_creator_list_reduction(p4est, true, SAS_timing_flag, SAS_subtiming_flag);
      break;
    default:
#ifdef CASL_THROWS
      err_msg = "my_p4est_biomolecules_t::construct_SES(const sas_generation_method& ): unknown SAS generation method...";
      err_manager.print_message_and_abort(err_msg, 265562);
#else
      MPI_Abort(p4est->mpicomm, 265562);
#endif
      break;
    }
  }

  // construct the sas grid and surface
  sas_creator->construct_SAS(p4est);
#ifdef CASL_THROWS
  local_error = reduced_list::get_nb_reduced_lists() != 0;
  err_msg = "my_p4est_biomolecules_t::construct_SES(): some reduced lists have not been deleted after the creation of the SAS surface, memory leak has happened...";
  err_manager.check_my_local_error(local_error, err_msg);
#endif
  delete sas_creator; sas_creator = NULL;

  if(vtk_folder[vtk_folder.size()-1] == '/')
    vtk_folder = vtk_folder.substr(0, vtk_folder.size()-1);
  bool export_intermediary_results = false; // !boost::iequals(vtk_folder, no_vtk);

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration();
      timer->start("Exporting the SAS results");
    }
    PetscErrorCode ierrr = VecGetArray(phi, &phi_p); CHKERRXX(ierrr);
    string vtk_file = vtk_folder + "/SAS_grid";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SAS", phi_p);
    ierrr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierrr);
  }

  if(timer != NULL && method_to_use != list_reduction_with_exact_phi)
  {
    timer->stop();timer->read_duration();
    timer->start("Reinitializing the levelset");
  }

  // make sure the ghost have been constructed in the last step of the SAS construction
  P4EST_ASSERT(ghost != NULL);
  // create hirerachy, nodes neighbors and a levelset
  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  else
    hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);
  if(neighbors != NULL)
    neighbors->update(hierarchy, nodes);
  else
    neighbors = new my_p4est_node_neighbors_t(hierarchy, nodes);
  neighbors->init_neighbors();
  if(ls != NULL)
    ls->update(neighbors);
  else
    ls = new my_p4est_level_set_t(neighbors);

  if(method_to_use != list_reduction_with_exact_phi)
  {
    double smallest_grid_size = root_cells_dim.at(0)/(1<<parameters.max_level());
    for (int dim = 1; dim < P4EST_DIM; ++dim)
      smallest_grid_size = MIN(smallest_grid_size, root_cells_dim.at(dim)/(1<<parameters.max_level()));
    double pseudo_time_step;
    int nb_it;

    // Reinitialization ONLY IN THE POSITIVE DOMAIN for computational efficiency
    // (load is balanced in the last partitioning step of the SAS construction)
    switch (parameters.order_of_accuracy()) {
    case 1:
      pseudo_time_step = 0.5*smallest_grid_size; // as it is in the first order reinitialization method
      nb_it = ceil(3.0*(parameters.probe_radius() + parameters.layer_thickness())/pseudo_time_step); // '3.0' == to ensure convergence
      ls->reinitialize_1st_order_above_threshold(phi, 0.0, MAX(nb_it, 10));
      break;
    case 2:
      pseudo_time_step = smallest_grid_size/((double) P4EST_DIM); // as it is in the second order reinitialization method
      nb_it = ceil(3.0*(parameters.probe_radius() + parameters.layer_thickness())/pseudo_time_step); // '3.0' == to ensure convergence
      ls->reinitialize_2nd_order_above_threshold(phi, 0.0, MAX(nb_it, 10));
      // VERY IMPORTANT NOTE: NEVER EVER use the following for this application, this scheme is not TVD,
      // it will result in an oscillatory behavior and no convergence can be expected...
      //    ls->reinitialize_1st_order_time_2nd_order_space_in_positive_domain(phi, diag_finest_cell, MAX(nb_it, 10));
      // END OF VERY IMPORTANT NOTE
      break;
    default:
  #ifdef CASL_THROWS
      err_msg = "my_p4est_biomolecules_t::construct_SES(): the order of accuracy should be either 1 or 2!!!";
      err_manager.print_message_and_abort(err_msg, 64799746);
  #else
      MPI_Abort(p4est->mpicomm, 64799746);
  #endif
      break;
    }
  }

  // subtract probe radius
  Vec phi_l;
  PetscErrorCode ierr = VecGhostGetLocalForm(phi, &phi_l); CHKERRXX(ierr);
  ierr = VecShift(phi_l, -parameters.probe_radius()); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_l); CHKERRXX(ierr);

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration();
      timer->start("Exporting the calculated SES");
    }
    PetscErrorCode ierrr = VecGetArray(phi, &phi_p); CHKERRXX(ierrr);
    string vtk_file = vtk_folder + "/SES_not_cavity_free";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p);
    ierrr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierrr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration();
    timer->start("Removing cavities");
  }
  remove_internal_cavities(region_growing, export_intermediary_results);
  //  remove_internal_cavities(poisson, export_intermediary_results); // this is NOT ROBUST ENOUGH, SHOULD NEVER BE USED

  if(export_intermediary_results)
  {
    if(timer != NULL)
    {
      timer->stop();timer->read_duration();
      timer->start("Exporting the SES and the cavities identification");
    }
    PetscErrorCode ierrr = VecGetArray(phi, &phi_p); CHKERRXX(ierrr);
    double* inner_domain_p = NULL;
    P4EST_ASSERT(inner_domain != NULL);
    ierrr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierrr);
    string vtk_file = vtk_folder + "/SES_and_cavities";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p,
                           VTK_POINT_DATA, "cavities", inner_domain_p);
    ierrr = VecRestoreArray(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierrr);
    ierrr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierrr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration();
    timer->start("Coarsening steps");
  }

  int nb_coarsening_steps = 0;
  while (coarsening_step(nb_coarsening_steps, export_intermediary_results))
  {
    if(export_intermediary_results)
    {
      PetscErrorCode ierrr = VecGetArray(phi, &phi_p); CHKERRXX(ierrr);
      double* inner_domain_p = NULL;
      P4EST_ASSERT(inner_domain != NULL);
      ierrr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierrr);
      string vtk_file = vtk_folder + "/coarsening_step_" + to_string(nb_coarsening_steps);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, vtk_file.c_str(),
                             VTK_POINT_DATA, "phi_SES", phi_p,
                             VTK_POINT_DATA, "acceleration", inner_domain_p);
      ierrr = VecRestoreArray(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierrr);
      ierrr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierrr);
    }
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration();
    timer->start("Enforcing the min level");
  }

  enforce_min_level(export_intermediary_results);
  if(export_intermediary_results)
  {
    PetscErrorCode ierrr;
    ierrr = VecGetArray(phi, &phi_p); CHKERRXX(ierrr);
    P4EST_ASSERT(inner_domain != NULL);
    double* inner_domain_p = NULL;
    ierrr = VecGetArray(inner_domain, &inner_domain_p); CHKERRXX(ierrr);
    string vtk_file = vtk_folder + "/enforcing_min_level";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, vtk_file.c_str(),
                           VTK_POINT_DATA, "phi_SES", phi_p,
                           VTK_POINT_DATA, "acceleration", inner_domain_p);
    ierrr = VecRestoreArray(inner_domain, &inner_domain_p); inner_domain_p = NULL; CHKERRXX(ierrr);
    ierrr = VecRestoreArray(phi, &phi_p); phi_p = NULL; CHKERRXX(ierrr);
  }

  if(timer != NULL)
  {
    timer->stop();timer->read_duration();
    timer->start("Final uniform partitioning");
  }

  partition_uniformly(export_intermediary_results);

  Vec phi_local;
  ierr = VecGhostGetLocalForm(phi, &phi_local); CHKERRXX(ierr);
  ierr = VecScale(phi_local, -1.0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_local); CHKERRXX(ierr);

  if(timer != NULL)
  {
    timer->stop(); timer->read_duration();
    delete timer; timer = NULL;
  }

  if(log_file != NULL)
  {
    if(log_timer != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer; log_timer = NULL;
    }
    p4est_gloidx_t total_nb_nodes = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      total_nb_nodes += nodes->global_owned_indeps[rank];
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of coarsening steps: %d \n", nb_coarsening_steps); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of nodes: %ld \n", total_nb_nodes); CHKERRXX(ierr);
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Number of quadrants: %ld \n", p4est->global_num_quadrants); CHKERRXX(ierr);
    Vec ones = NULL, ones_ghost_local = NULL;
    ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(ones, &ones_ghost_local); CHKERRXX(ierr);
    ierr = VecSet(ones_ghost_local, 1.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &ones_ghost_local); ones_ghost_local = NULL; CHKERRXX(ierr);
    double molecule_area = integrate_over_interface(p4est, nodes, phi, ones);
    ierr = VecDestroy(ones); ones = NULL; CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Surface of the molecule: %g (in domain dimensions), %g A^2\n", molecule_area, molecule_area*pow(angstrom_to_domain, -(P4EST_DIM-1)), P4EST_DIM-1); CHKERRXX(ierr);
#else
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "Surface of the molecule: %g (in domain dimensions), %g A\n", molecule_area, molecule_area*pow(angstrom_to_domain, -(P4EST_DIM-1))); CHKERRXX(ierr);
#endif
  }
  // return the user pointer of the p4est as it was and return the p4est
  p4est->user_pointer = user_pointer_saved;
  return p4est;
}

void my_p4est_biomolecules_t::expand_ghost()
{
  P4EST_ASSERT(p4est != NULL && nodes != NULL && ghost != NULL &&
      hierarchy != NULL && neighbors != NULL && ls != NULL &&
      phi != NULL);
  Vec old_phi = phi, old_inner_domain = inner_domain;
  inner_domain = NULL;
  phi = NULL;
  PetscInt nb_loc_values = nodes->num_owned_indeps;
  vector<PetscInt> global_indices(nb_loc_values, 0);
  p4est_gloidx_t node_offset = 0;
  for (int r = 0; r < p4est->mpirank; ++r) {
    node_offset += nodes->global_owned_indeps[r];
  }
  for (p4est_locidx_t k = 0; k < nb_loc_values; ++k) {
    global_indices.at(k) = k + node_offset;
  }
  p4est_ghost_expand(p4est, ghost);
  p4est_nodes_destroy(nodes);
  nodes   = my_p4est_nodes_new(p4est, ghost);
  PetscErrorCode ierr;
  ierr    = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);
  }
  P4EST_ASSERT(phi != NULL && ((old_inner_domain == NULL) || (old_inner_domain != NULL && inner_domain != NULL)));
  // so now, we have to rescatter the vector(s), update the hierarchy, the node neighbors, the levelset object
  VecScatter ctx, ctx_inner_domain;
  IS is;
  const PetscInt* set_of_global_indices = (nb_loc_values > 0)? (const PetscInt*) &global_indices.at(0) : PETSC_NULL;
  ierr    = ISCreateGeneral(p4est->mpicomm, nb_loc_values, set_of_global_indices, PETSC_USE_POINTER, &is); CHKERRXX(ierr);
  ierr    = VecScatterCreate(old_phi, is, phi, is, &ctx); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecScatterCreate(old_inner_domain, is, inner_domain, is, &ctx_inner_domain); CHKERRXX(ierr);
  }
  ierr    = VecScatterBegin(ctx, old_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecScatterBegin(ctx_inner_domain, old_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  // place those operations here for optimizing the execution
  hierarchy->update(p4est, ghost);
  neighbors->update(p4est, ghost, nodes);
  ls->update(neighbors);
  // back to scattering vector values
  ierr    = VecScatterEnd(ctx, old_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecScatterEnd(ctx_inner_domain, old_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  ierr    = VecDestroy(old_phi); CHKERRXX(ierr);
  if(old_inner_domain != NULL){
    ierr  = VecDestroy(old_inner_domain); CHKERRXX(ierr);
  }
  ierr    = VecScatterDestroy(ctx); CHKERRXX(ierr);
  if(inner_domain != NULL){
    ierr  = VecScatterDestroy(ctx_inner_domain); CHKERRXX(ierr);
  }
  ierr    = ISDestroy(is); CHKERRXX(ierr);
  ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(inner_domain != NULL){
    ierr  = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

}

void my_p4est_biomolecules_t::set_quad_weight(p4est_quadrant_t* &quad, const p4est_nodes_t* & nodes, const double* const& phi_fct, const double& lower_bound)
{
  p4est_locidx_t quad_idx = quad->p.user_long;
  quad->p.user_long       = 0;
  for (unsigned short n = 0; n < P4EST_CHILDREN; ++n)
  {
    p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+n];
    P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
    if(phi_fct[node_idx] > lower_bound)
    {
      quad->p.user_long = 1;
      return;
    }
  }
}

p4est_bool_t my_p4est_biomolecules_t::coarsen_fn(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad[])
{
  const my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
  const int min_lvl                     = biomol->parameters.min_level();
  const int max_lvl                     = biomol->parameters.max_level();
  const p4est_nodes_t* nodes            = biomol->nodes;
  (void) which_tree;

  p4est_bool_t result;
  if (quad[0]->level <= min_lvl)
    result = P4EST_FALSE;
  else if (quad[0]->level > max_lvl)
    result = P4EST_TRUE;
  else
  {
    const double parent_cell_diag = 2.0*biomol->parameters.root_diag()/(1<<quad[0]->level);
    const double lip              = biomol->parameters.lip_cst();

    double f[P4EST_CHILDREN];
    for (unsigned short k = 0; k < P4EST_CHILDREN; ++k) { // not exactly the same as in the paper, but equivalent!
      p4est_locidx_t quad_idx = quad[k]->p.user_long;
      p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+k];
      P4EST_ASSERT(((size_t) node_idx < nodes->indep_nodes.elem_count));
      f[k] = biomol->phi_read_only_p[node_idx];
      if(fabs(f[k]) <= 0.5*lip*parent_cell_diag)
      {
        result = P4EST_FALSE;
        goto function_end;
      }
    }
    // no need to check for interface crossing, this is prevented by
    // (phi = signed distance) + (lip >= 1)...
    result = P4EST_TRUE;
  }

function_end:
  for (unsigned short q = 0; q < P4EST_CHILDREN; ++q)
    set_quad_weight(quad[q], nodes, biomol->phi_read_only_p, 0.0);
  // if I'm not mistaken when reading p4est_coarsen source file, the weight of the first child is important in case of coarsening...
  // --> should work as well

  return result;
}

int my_p4est_biomolecules_t::weight_for_coarsening(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant)
{
  (void) which_tree;
  (void) park;
  return quadrant->p.user_long;
}

bool my_p4est_biomolecules_t::coarsening_step(int& step_idx, bool export_acceleration)
{
  P4EST_ASSERT(nodes != NULL);
  P4EST_ASSERT(ghost != NULL);
  P4EST_ASSERT(phi_p == NULL);
  P4EST_ASSERT(phi_read_only_p == NULL);
  PetscErrorCode ierr;
  int mpiret;

  if(log_file != NULL && step_idx > log2(box_size_of_biggest_mol*sqrt(P4EST_DIM)*(1<<(parameters.max_level()-1))/(parameters.lip_cst()*parameters.root_diag())))
  {
    ierr = PetscFPrintf(p4est->mpicomm, log_file, "More coarsening steps than expected... This is weird! \n"); CHKERRXX(ierr);
  }
  // Explanation:
  // At each iteration, we reinitialize to capture a new layer of inner cells (away from the interface) that are potential candidates for coarsening.
  // When step_idx = k, the width of that candidate layer is
  //   dist_k = parameters.lip_cst()*(2*diag of cell of level k) = parameters.lip_cst()*(diag of root cell)*(2^(k+1-parameters.max_lvl()))
  // When dist_k > diagonal of the biggest bounding box, it makes no sense to keep doing it (and it should NEVER happen actually)...

  Vec phi_ghost_loc;
  // Given the number of iterations used for capturing the SES accurately,
  // WE assumed that the first layer is captured too, without reinitialization...
  // [Wrong only if L is absurdly large...]
  double already_captured_layer = parameters.lip_cst()*parameters.root_diag()/(1<<(parameters.max_level()-step_idx++));
  // "remove internal cavities" == trick to accelerate the reinitialization steps
  // and avoid more than one local coarsening operation
  ierr = VecGhostGetLocalForm(phi, &phi_ghost_loc); CHKERRXX(ierr);
  ierr = VecShift(phi_ghost_loc, -already_captured_layer); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_ghost_loc); CHKERRXX(ierr);
  remove_internal_cavities(region_growing, export_acceleration);
  ierr = VecGhostGetLocalForm(phi, &phi_ghost_loc); CHKERRXX(ierr);
  ierr = VecShift(phi_ghost_loc, +already_captured_layer); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_ghost_loc); CHKERRXX(ierr);
  P4EST_ASSERT((export_acceleration && inner_domain != NULL) || (!export_acceleration && inner_domain == NULL));

  // needed for coarsening
  for (p4est_topidx_t tree_id = p4est->first_local_tree; tree_id <= p4est->last_local_tree; ++tree_id) {
    p4est_tree_t* tree_k = (p4est_tree_t*) sc_array_index(p4est->trees, tree_id);
    for (size_t q = 0; q < tree_k->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree_k->quadrants, q);
      quad->p.user_long = q + tree_k->quadrants_offset;
    }
  }

  p4est_locidx_t former_quad_count = p4est->local_num_quadrants;
  ierr  = VecGetArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);
  my_p4est_coarsen(p4est, P4EST_FALSE, my_p4est_biomolecules_t::coarsen_fn, NULL);
  ierr = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr); phi_read_only_p = NULL;
  P4EST_ASSERT(former_quad_count >= p4est->local_num_quadrants);
  int grid_has_changed = (former_quad_count != p4est->local_num_quadrants);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &grid_has_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  if(!grid_has_changed)
    return grid_has_changed;

  // else, repartition the tree and the levelset function for further coarsening
  // - repartition for balanced further reinitialization (see set_quad_weight in coarsen_fn)
  // - allowing new families of quadrants to be grouped together for further coarsening steps

  // store the current (fine) nodes, and get the new ones
  p4est_nodes_t* fine_nodes = nodes;
  nodes = my_p4est_nodes_new(p4est, NULL);

  int clamped = 1;
  PetscInt nb_loc_idx = nodes->num_owned_indeps;
  vector<PetscInt> global_idx_to_scatter_from; global_idx_to_scatter_from.resize(nb_loc_idx);
  vector<PetscInt> global_idx_to_scatter_to; global_idx_to_scatter_to.resize(nb_loc_idx);

  p4est_locidx_t idx   = 0;
  p4est_indep_t *node = NULL;
  if(idx < nodes->num_owned_indeps)
    node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes, idx);
  PetscInt my_index_offset_from = 0;
  PetscInt my_index_offset_to = 0;
  for (int proc_rank = 0; proc_rank < p4est->mpirank; ++proc_rank)
  {
    my_index_offset_from += fine_nodes->global_owned_indeps[proc_rank];
    my_index_offset_to   += nodes->global_owned_indeps[proc_rank];
  }
  for (p4est_locidx_t k = 0; k < fine_nodes->num_owned_indeps; ++k)
  {
    p4est_indep_t *fine_node  = (p4est_indep_t*) sc_array_index(&fine_nodes->indep_nodes,k);
    if(node!= NULL && p4est_node_equal_piggy_fn (fine_node, node, &clamped))
    {
      global_idx_to_scatter_to.at(idx)     = my_index_offset_to + idx;
      global_idx_to_scatter_from.at(idx++) = my_index_offset_from + k;
      if(idx < nodes->num_owned_indeps)
        node = (p4est_indep_t*) sc_array_index(&nodes->indep_nodes,idx);
      else
        break;
    }
  }
  // sanity check
#ifdef CASL_THROWS
  int local_error = (idx != nodes->num_owned_indeps);
  string err_msg = "\n my_p4est_biomolecules_t::coarsening_step(...) killed because of logic error \n \n";
  err_manager.check_my_local_error(local_error, err_msg);
#endif

  ierr  = VecGetArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);
  my_p4est_partition(p4est, P4EST_TRUE, my_p4est_biomolecules_t::weight_for_coarsening);
  ierr  = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr); phi_read_only_p = NULL;
  // we don't need the fine ghosts, nor the fine and locally coarse nodes (before the partition)
  p4est_ghost_destroy(ghost); ghost = NULL;
  p4est_nodes_destroy(fine_nodes); fine_nodes = NULL;
  p4est_nodes_destroy(nodes); nodes = NULL;
  // create the new ones
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);


  Vec fine_phi = phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  Vec fine_inner_domain = NULL;
  if(export_acceleration)
  {
    fine_inner_domain = inner_domain;
    ierr = VecCreateGhostNodes(p4est, nodes, &inner_domain); CHKERRXX(ierr);
  }

  IS is_from;
  IS is_to;
  PetscInt* fine_global_indices   = (global_idx_to_scatter_from.size() != 0)? &global_idx_to_scatter_from.at(0) : NULL;
  PetscInt* coarse_global_indices = (global_idx_to_scatter_to.size() != 0)? &global_idx_to_scatter_to.at(0) : NULL;
  ierr    = ISCreateGeneral(p4est->mpicomm, nb_loc_idx, fine_global_indices, PETSC_USE_POINTER, &is_from); CHKERRXX(ierr);
  ierr    = ISCreateGeneral(p4est->mpicomm, nb_loc_idx, coarse_global_indices, PETSC_USE_POINTER, &is_to); CHKERRXX(ierr);
  VecScatter ctx, ctx_inner_domain;
  ierr    = VecScatterCreate(fine_phi, is_from, phi, is_to, &ctx); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecScatterCreate(fine_inner_domain, is_from, inner_domain, is_to, &ctx_inner_domain); CHKERRXX(ierr);
  }
  ierr    = VecScatterBegin(ctx, fine_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecScatterBegin(ctx_inner_domain, fine_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  hierarchy->update(p4est, ghost);
  neighbors->update(hierarchy, nodes);
  ls->update(neighbors);
  ierr    = VecScatterEnd(ctx, fine_phi, phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecScatterEnd(ctx_inner_domain, fine_inner_domain, inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr  = VecDestroy(fine_inner_domain); CHKERRXX(ierr); fine_inner_domain = NULL;
  }
  ierr    = VecDestroy(fine_phi); CHKERRXX(ierr);
  ierr    = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecGhostUpdateBegin(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr    = VecScatterDestroy(ctx); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecScatterDestroy(ctx_inner_domain); CHKERRXX(ierr);
  }
  ierr    = ISDestroy(is_from); CHKERRXX(ierr);
  ierr    = ISDestroy(is_to); CHKERRXX(ierr);
  ierr    = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(export_acceleration)
  {
    ierr  = VecGhostUpdateEnd(inner_domain, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  double min_root_cell_dim = root_cells_dim.at(0);
  for (int k = 1; k < P4EST_DIM; ++k)
    min_root_cell_dim = MIN(min_root_cell_dim, root_cells_dim.at(k));
  int n_iter = ceil(1.5*parameters.lip_cst()*parameters.root_diag()/(0.5*min_root_cell_dim));
  // Explanation:
  // 1) parameters.lip_cst()*diag of current cell level = theoretical step of pseudo-time until which
  // the reinitialization equation must be further solved to capture the next layer of coarsened cells
  // (assuming the current level set function is accurate enough for all other points that are closer to the surface)
  // --> call that pseudo-time tau_end
  // 2) tau_end/(0.5*min dim of current cell level) = corresponding number of iterations in the following
  // first-order reinitialization algorithm
  // 3) (diag of current cell level)/(min dim of current cell level) is level-invariant, so evaluate that ratio for root cell...
  // 4) 1.5 == 'safety factor' for better convergence
  // (might not be 100% exact at the end of the day, but what else can we do?)

  // note the "_above_threshold" and the value of the threshold --> the actual SES does NOT MOVE
  // the order of accuracy is irrelevant for this application, really --> choose the fastest (1st order) method
  ls->reinitialize_1st_order_above_threshold(phi, parameters.layer_thickness(), n_iter);

  return grid_has_changed;
}

my_p4est_biomolecules_solver_t::my_p4est_biomolecules_solver_t(const my_p4est_biomolecules_t *biomolecules_)
  : biomolecules(biomolecules_)
{
  temperature           = -1.0; // absurd initialization
  far_field_ion_density = -1.0; // absurd initialization
  ion_charge            = 0;    // absurd initialization
  mol_rel_permittivity  = 0.0;  // absurd initialization
  elec_rel_permittivity = 0.0;  // absurd initialization

  // initialize psi_star, psi_naught, psi_bar, and psi, will be created when needed
  psi_star              = NULL;
  psi_naught            = NULL;
  psi_bar               = NULL;
  psi_hat               = NULL;
  validation_error      = NULL;
  psi_star_psi_naught_and_psi_bar_are_set = psi_hat_is_set = false;
  // initialize the dirichlet_boundary_condition
  dirichlet_bc.setWallTypes(dirichlet_bc_wall_type);

  // create the cell neighbors
  cell_neighbors  = new my_p4est_cell_neighbors_t(biomolecules->hierarchy);
  // create the solvers
  //jump_solver     = new my_p4est_poisson_jump_nodes_voronoi_t(biomolecules->neighbors, cell_neighbors);
  jump_solver     = new my_p4est_general_poisson_nodes_mls_solver_t(biomolecules->neighbors);
  node_solver     = new my_p4est_poisson_nodes_t(biomolecules->neighbors);
}
void    my_p4est_biomolecules_solver_t::set_molecular_relative_permittivity(double epsilon_molecule)
{
#ifdef CASL_THROWS
  int local_error = (epsilon_molecule < 1.0);
  string error_message = "my_p4est_biomolecules_solver_t::set_molecular_relative_permittivity(...): the value of the molecular relative permittivity must be greater than or equal to 1.0.";
  biomolecules->err_manager.check_my_local_error(local_error, error_message);
  if(mol_rel_permittivity > 1.0-EPS && fabs(mol_rel_permittivity - epsilon_molecule) > EPS*mol_rel_permittivity)
  {
    // print a warning
    string message = "my_p4est_biomolecules_solver_t::set_molecular_relative_permittivity(...): the molecular permittivity was already set, it will be reset...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
  }
#endif
  // if the value is changed, psi_star, psi_bar and psi are no longer valid
  psi_star_psi_naught_and_psi_bar_are_set = psi_hat_is_set = !(fabs(mol_rel_permittivity - epsilon_molecule) > EPS*fabs(mol_rel_permittivity));
  mol_rel_permittivity  = epsilon_molecule;
}
void    my_p4est_biomolecules_solver_t::set_electrolyte_relative_permittivity(double epsilon_electrolyte)
{
#ifdef CASL_THROWS
  int local_error = (epsilon_electrolyte < 1.0);
  string error_message = "my_p4est_biomolecules_solver_t::set_electrolyte_relative_permittivity(...): the value of the electrolyte relative permittivity must be greater than or equal to 1.0.";
  biomolecules->err_manager.check_my_local_error(local_error, error_message);
  if(elec_rel_permittivity > 1.0-EPS && fabs(elec_rel_permittivity - epsilon_electrolyte) > EPS*elec_rel_permittivity)
  {
    // print a warning
    string message = "my_p4est_biomolecules_solver_t::set_electrolyte_relative_permittivity(...): the electrolyte permittivity was already set, it will be reset...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
  }
#endif
  // if the value is changed, psi is no longer valid
  psi_hat_is_set = !(fabs(elec_rel_permittivity - epsilon_electrolyte) > EPS*fabs(elec_rel_permittivity));
  elec_rel_permittivity   = epsilon_electrolyte;
}
void    my_p4est_biomolecules_solver_t::set_relative_permittivities(double epsilon_molecule, double epsilon_electrolyte)
{
  set_molecular_relative_permittivity(epsilon_molecule);
  set_electrolyte_relative_permittivity(epsilon_electrolyte);
}
void    my_p4est_biomolecules_solver_t::set_temperature_in_kelvin(double temperature_in_K)
{
#ifdef CASL_THROWS
  int local_error = (temperature_in_K < EPS);
  string error_message = "my_p4est_biomolecules_solver_t::set_temperature_in_K(...): the value of the temperature must be strictly positive.";
  biomolecules->err_manager.check_my_local_error(local_error, error_message);
  if(is_temperature_set() && temperature_in_K > EPS && fabs(temperature - temperature_in_K) > EPS*temperature)
  {
    // print a warning
    string message = "my_p4est_biomolecules_solver_t::set_temperature_in_kelvin(...): the temperature was already set, it will be reset...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
  }
#endif
  // if the value is changed, psi_star, psi_bar and psi are no longer valid
  psi_star_psi_naught_and_psi_bar_are_set = psi_hat_is_set = !(fabs(temperature - temperature_in_K) > EPS*fabs(temperature));
  temperature = temperature_in_K;
}
void    my_p4est_biomolecules_solver_t::set_far_field_ion_density(double n_0)
{
#ifdef CASL_THROWS
  //int local_error = (n_0 < EPS);
  //string error_message = "my_p4est_biomolecules_solver_t::set_far_field_ion_density(...): the value of the far-field ion density must be strictly positive.";
  //biomolecules->err_manager.check_my_local_error(local_error, error_message);
//  if(is_far_field_ion_density_set() && n_0 > EPS && fabs(far_field_ion_density - n_0) > EPS*far_field_ion_density)
  if(is_far_field_ion_density_set())
  {
    // print a warning
    string message = "my_p4est_biomolecules_solver_t::set_far_field_ion_density(...): the far-field ion density was already set, it will be reset...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
  }
#endif
  // if the value is changed, psi is no longer valid
  psi_hat_is_set = !(fabs(far_field_ion_density - n_0) > EPS*fabs(far_field_ion_density));
  far_field_ion_density = n_0;
}
void    my_p4est_biomolecules_solver_t::set_ion_charge(int z)
{
#ifdef CASL_THROWS
  int local_error = (z <= 0);
  string error_message = "my_p4est_biomolecules_solver_t::set_ion_charge(...): the ion charge must be a strictly positive integer.";
  biomolecules->err_manager.check_my_local_error(local_error, error_message);
  if(is_ion_charge_set() && z > 0 && ion_charge != z)
  {
    // print a warning
    string message = "my_p4est_biomolecules_solver_t::set_ion_charge(...): the ion charge was already set, it will be reset...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
  }
#endif
  // if the value is changed, psi_star, psi_bar and psi are no longer valid
  psi_star_psi_naught_and_psi_bar_are_set = psi_hat_is_set = !(ion_charge != z);
  ion_charge = z;
}
void    my_p4est_biomolecules_solver_t::set_inverse_debye_length_in_meters_inverse(double inverse_debye_length_in_m_inverse)
{
//#ifdef CASL_THROWS
//  int local_error = (debye_length_in_m < EPS);
//  string error_message = "my_p4est_biomolecules_solver_t::set_debye_length_in_meters(...): the value of the debye length must be strictly positive.";
//  biomolecules->err_manager.check_my_local_error(local_error, error_message);
//#endif
  // if all three parameters are set, it's either consistent or not. If not, abort
  if(are_all_debye_parameters_set())
  {
    // Either it's the correct value,
    if(fabs(get_inverse_debye_length_in_meters_inverse() - inverse_debye_length_in_m_inverse) < EPS*get_inverse_debye_length_in_meters_inverse())
      return; // nothing to be done
    // or the user is dumb and we should teach him
#ifdef CASL_THROWS
    string err_msg = "my_p4est_biomolecules_solver_t::set_debye_length_in_meters(): the debye length is already set by the relevant parameters, it can't be set to the new value: abort...";
    biomolecules->err_manager.print_message_and_abort(err_msg, 957759);
#else
    MPI_Abort(biomolecules->p4est->mpicomm, 957759);
#endif
  }
  // if only 2 of the three parameters are set, it's the "easy" case, the third one might be calculated
  if(is_ion_charge_set() && is_temperature_set() && !is_far_field_ion_density_set())
  {
    // temperature and ion charge are set, the far-field ion density is not, let's calculate it
    set_far_field_ion_density(eps_0*kB*temperature*inverse_debye_length_in_m_inverse/(2.0*SQR(((double) ion_charge)*electron)));
    std::cout << "setting far field ion density based on other values \n";
    return;
  }
  if(is_ion_charge_set() && !is_temperature_set() && is_far_field_ion_density_set())
  {
    // far-field ion density and ion charge are set, temperature is not, let's calculate it
    set_temperature_in_kelvin(2.0*far_field_ion_density*SQR((1/inverse_debye_length_in_m_inverse)*((double) ion_charge)*electron)/(eps_0*kB));
    std::cout << "setting temperature based on other values \n";
    return;
  }
  if(!is_ion_charge_set() && is_temperature_set() && is_far_field_ion_density_set())
  {
    // ion charge might be freely set
    double my_z = sqrt(eps_0*kB*temperature/(2.0*far_field_ion_density*SQR((1/inverse_debye_length_in_m_inverse)*electron)));
    // but it's supposed to be an integer, so let's check that
    if(fabs(my_z - round(my_z)) < EPS*my_z)
      set_ion_charge((int) round(my_z));
    else
    {
#ifdef CASL_THROWS
      string err_msg = "my_p4est_biomolecules_solver_t::set_debye_length_in_meters(): the calculated ion charge is not an integer, it can't be fixed: abort...";
      biomolecules->err_manager.print_message_and_abort(err_msg, 95788759);
#else
      MPI_Abort(biomolecules->p4est->mpicomm, 95788759);
#endif
    }
    std::cout << "setting ion charge based on other values \n";
    return;
  }
  // if none or only one of the three parameters is set, we have one more degree of freedom, and we can set arbitrary values
  if(!is_ion_charge_set())
  {
    // set to default (1:1 electrolyte)
    set_ion_charge();
    // one or two of the parameters are set, now
    // if only one (the ion charge), the next pass will set the temperature and the far-field ion density will be calculated
    // if two (the ion charge and either the temperature or the far-field density), the next pass will set the remaining one
    set_inverse_debye_length_in_meters_inverse(inverse_debye_length_in_m_inverse);
    std::cout << "setting ion charge and then inverse_debye_length \n";
  }
  else
  {
    // the ion charge only is set, set the temperature to 300 K (default value) and that's it
    set_temperature_in_kelvin();
    set_inverse_debye_length_in_meters_inverse(inverse_debye_length_in_m_inverse);
   std::cout << "setting temperature and then inverse_debye_length \n";
  }
  return;
}
double  my_p4est_biomolecules_solver_t::get_inverse_debye_length_in_meters_inverse() const
{
  if(are_all_debye_parameters_set()){
    //return sqrt(eps_0*kB*temperature/(2.0*far_field_ion_density*SQR(((double) ion_charge)*electron)));
    return sqrt((2.0*far_field_ion_density*SQR(((double) ion_charge)*electron))/(eps_0*kB*temperature));
  }
  else
  {
#ifdef CASL_THROWS
    string err_msg = "my_p4est_biomolecules_solver_t::get_inverse_debye_length_in_meters_inverse(): not all debye parameters are set, the debye length cannot be calculated: abort...";
    biomolecules->err_manager.print_message_and_abort(err_msg, 95755759);
#else
    MPI_Abort(biomolecules->p4est->mpicomm, 95755759);
#endif
    return DBL_MAX;
  }
}

void    my_p4est_biomolecules_solver_t::return_psi_star_psi_naught_and_psi_bar(Vec &psi_star_out, Vec &psi_naught_out, Vec &psi_bar_out)
{
#ifdef CASL_THROWS
  int local_error = ((psi_bar == NULL) || (psi_star == NULL));
  string message = "my_p4est_biomolecules_solver_t::return_psi_bar_and_psi_star(): (one of) the psi_bar and psi_star vectors is (are) NULL, (it) they can't be returned...";
  biomolecules->err_manager.check_my_local_error(local_error, message);
#endif
  psi_star_out    = psi_star; psi_star = NULL;
  psi_naught_out  = psi_naught; psi_naught = NULL;
  psi_bar_out     = psi_bar; psi_bar = NULL;
  psi_star_psi_naught_and_psi_bar_are_set = false;
}
void     my_p4est_biomolecules_solver_t::return_psi_hat(Vec &psi_hat_out)
{
#ifdef CASL_THROWS
  int local_error = psi_hat == NULL;
  string message = "my_p4est_biomolecules_solver_t::return_psi_hat(): the psi_hat vector is NULL, it can't be returned...";
  biomolecules->err_manager.check_my_local_error(local_error, message);
#endif
  psi_hat_out=psi_hat; psi_hat=NULL;
//  Vec psi_hat_to_return = psi_hat;
//  //psi_hat = NULL;
  psi_hat_is_set = false;
//  return psi_hat_to_return;
}
Vec my_p4est_biomolecules_solver_t::return_validation_error()
{
#ifdef CASL_THROWS
  int local_error = validation_error == NULL;
  string message = "my_p4est_biomolecules_solver_t::return_validation_error(): the validation_error vector is NULL \n(constructed and defined only for validation purposes, i.e. if the validation flag is set to true when solver functions are called), it can't be returned...";
  biomolecules->err_manager.check_my_local_error(local_error, message);
#endif
  Vec validation_error_to_return = validation_error;
  validation_error = NULL; // the new owner takes the responsibility of destroying the vector
  return validation_error_to_return;
}
//Vec my_p4est_biomolecules_solver_t::return_residual()
//{
//#ifdef CASL_THROWS
//  int local_error = jump_solver->rhs == NULL;
//  string message = "my_p4est_biomolecules_solver_t::return_residual(): the rhs of the jump solver (aka as the residual) vector is NULL \nIt can't be interpolated and returned...";
//  biomolecules->err_manager.check_my_local_error(local_error, message);
//#endif
//  Vec residual_on_grid = NULL;
//  make_sure_is_node_sampled(residual_on_grid);
//  Vec sol_voro_saved    = jump_solver->sol_voro;
//  jump_solver->sol_voro = jump_solver->rhs;
//  jump_solver->interpolate_solution_from_voronoi_to_tree(residual_on_grid);
//  jump_solver->rhs      = jump_solver->sol_voro;
//  jump_solver->sol_voro = sol_voro_saved;
//  return residual_on_grid;
//}
void    my_p4est_biomolecules_solver_t::return_all_psi_vectors(Vec &psi_star_out, Vec &psi_naught_out, Vec &psi_bar_out, Vec &psi_hat_out, bool validation_flag)
{
  if(!validation_flag)
    return_psi_star_psi_naught_and_psi_bar(psi_star_out, psi_naught_out, psi_bar_out);
  return_psi_hat(psi_hat_out);
}

void    my_p4est_biomolecules_solver_t::make_sure_is_node_sampled(Vec &vector)
{
  if(vector != NULL)
  {
    PetscInt size_local, size_local_ghost, size_global;
    Vec vector_local_ghost = NULL;
    ierr = VecGetSize(vector, &size_global); CHKERRXX(ierr);
    ierr = VecGetLocalSize(vector, &size_local); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vector, &vector_local_ghost); CHKERRXX(ierr);
    ierr = VecGetSize(vector_local_ghost, &size_local_ghost); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(vector, &vector_local_ghost); vector_local_ghost = NULL; CHKERRXX(ierr);
    p4est_gloidx_t nb_nodes_total = 0;
    for (int rr = 0; rr < biomolecules->p4est->mpisize; ++rr)
      nb_nodes_total += biomolecules->nodes->global_owned_indeps[rr];
    int logic_error = ((size_global != nb_nodes_total) ||
                       (size_local_ghost != (PetscInt) biomolecules->nodes->indep_nodes.elem_count) ||
                       (size_local != biomolecules->nodes->num_owned_indeps));
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &logic_error, 1, MPI_INT, MPI_LOR, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(logic_error)
    {
      ierr = VecDestroy(vector); CHKERRXX(ierr); vector = NULL;
      ierr = VecCreateGhostNodes(biomolecules->p4est, biomolecules->nodes, &vector); CHKERRXX(ierr);
    }
  }
  else
  {
    ierr = VecCreateGhostNodes(biomolecules->p4est, biomolecules->nodes, &vector); CHKERRXX(ierr);
  }
}
void    my_p4est_biomolecules_solver_t::solve_singular_part()
{
  if(psi_star_psi_naught_and_psi_bar_are_set)
    return;
#ifdef CASL_THROWS
  int local_error = !(is_temperature_set() && is_ion_charge_set() && is_molecular_permittivity_set());
  string err_msg  = "my_p4est_biomolecules_solver_t::solve_singular_part(): some parameters are not set yet, the singular problem can't be solved...";
  biomolecules->err_manager.check_my_local_error(local_error, err_msg);
#endif

  // This step is somehow critical, because its solution will determine the jump conditions at the
  // interface for the (non)linear Poisson-Boltzmann solver. Those jump conditions depend on the
  // NORMAL DERIVATIVE of the singular solution psi_bar calculated herein.
  // Therefore, psi_bar must be calculated with one order of accuracy higher than the desired final
  // order of accuracy for psi, that means
  // - order 2 for psi_bar if the desired order of psi is 1;
  // - order 3 for psi_bar if the desired order of psi is 2.
  // Therefore, the Dirichlet boundary condition for psi_naught should be evaluated with 2nd (resp. 3rd)
  // order accurate interpolation, i.e. linear (resp. quadratic non-oscillatory) interpolation
  // if the desired order of accuracy is 1 (resp. 2).

  make_sure_is_node_sampled(psi_star);
  make_sure_is_node_sampled(psi_naught);
  make_sure_is_node_sampled(psi_bar);
  // initialize the three vectors to 0.0;
  double *psi_star_p = NULL, *psi_naught_p = NULL, *psi_bar_p = NULL;
  ierr = VecGetArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_naught, &psi_naught_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  // initialize both to 0.0
  for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
    psi_bar_p[i]  = 0.0;
    psi_star_p[i] = 0.0;
    psi_naught_p[i] = 0.0;
  }
  ierr = VecRestoreArray(psi_bar, &psi_bar_p); psi_bar_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_naught, &psi_naught_p); psi_naught_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_star, &psi_star_p); psi_star_p = NULL; CHKERRXX(ierr);

  // sample the contribution of singular charges at grid nodes, but only in the inner domain(s)
  // Note that the NEGATIVE value is sampled to make the imposition of the Dirichlet boundary condition for psi_0 easier
  double xyz[P4EST_DIM];
  const double* phi_read_only_p = NULL;
  ierr = VecGetArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p); CHKERRXX(ierr);
  // if OOA == 1 --> linear interpolation of psi_star --> need only the closest neighbors --> finest diag
  // if OOA == 2 --> quadratic non-oscillatory interpolation of psi_star --> need also the outer neighbors of the closest neighbors
  //             --> 2.0*finest diag (we have enforced lip >= 1.0)
  for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
    // --> calculate the (negative) contribution of singular charges if the levelset function is smaller than or equal to the layer thickness
    if(phi_read_only_p[i] <= 1.5*biomolecules->parameters.layer_thickness()) //1.5 == safety factor
    {
      node_xyz_fr_n(i, biomolecules->p4est, biomolecules->nodes, xyz);
#ifdef P4_TO_P8
      psi_star_p[i] = -non_dimensional_coulomb_in_mol(xyz[0], xyz[1], xyz[2]);
#else
      psi_star_p[i] = -non_dimensional_coulomb_in_mol(xyz[0], xyz[1]);
#endif
    }
  }
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_star, &psi_star_p); psi_star_p = NULL; CHKERRXX(ierr);

  // solve a poisson equation to obtain the smooth part of the soluton in the inner domain, i.e. psi_0
  // the following are needed in the (OOA == 2) case if you want results by the end of the day
  Vec psi_star_xx = NULL, psi_star_yy = NULL;
#ifdef P4_TO_P8
  Vec psi_star_zz = NULL;
#endif
  my_p4est_interpolation_nodes_t bc_interface_value(biomolecules->neighbors);
  switch (biomolecules->parameters.order_of_accuracy()) {
  case 1:
    bc_interface_value.set_input(psi_star, linear);
    break;
  case 2:
    make_sure_is_node_sampled(psi_star_xx);
    biomolecules->neighbors->dxx_central(psi_star, psi_star_xx);
    make_sure_is_node_sampled(psi_star_yy);
    biomolecules->neighbors->dyy_central(psi_star, psi_star_yy);
#ifdef P4_TO_P8
    make_sure_is_node_sampled(psi_star_zz);
    biomolecules->neighbors->dzz_central(psi_star, psi_star_zz);
#endif
    bc_interface_value.set_input(psi_star, psi_star_xx, psi_star_yy,
                             #ifdef P4_TO_P8
                                 psi_star_zz,
                             #endif
                                 quadratic_non_oscillatory);
    break;
  default:
#ifdef CASL_THROWS
    err_msg = "my_p4est_biomolecules_solver_t::solve_singular_part(), the order of accuracy should be either 1 or 2!!!";
    biomolecules->err_manager.print_message_and_abort(err_msg, 19791);
#else
    MPI_Abort(biomolecules->p4est->mpicomm, 19791);
#endif
    break;
  }

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  struct:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    double operator()(double, double
                  #ifdef P4_TO_P8
                      , double
                  #endif
                      ) const { return 0;}
  } bc_wall_value;

  struct:
    #ifdef P4_TO_P8
      WallBC3D
    #else
      WallBC2D
    #endif
  {
    BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                     , double
                                 #endif
                                     ) const { return DIRICHLET; }
  } bc_wall_type;

  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);
  P4EST_ASSERT(node_solver != NULL);
  node_solver->set_bc(bc);
  node_solver->set_phi(biomolecules->phi);
  node_solver->set_mu(1.0); // the rhs is zero, so any (constant) value is equally good
  node_solver->set_rhs(psi_naught);
  node_solver->solve(psi_naught);
  // we have psi_0

  // add the contribution of singular charges to obtain psi_bar = psi_0 + psi_star (contribution of singular charges)
  // note that the value of psi_star was sampled with a negative sign above (for the calculation of the interface
  // Dirichlet boundary condition) --> scale by "-1.0" to get back to the correct value
  const double *psi_naught_read_only_p = NULL;
  ierr = VecGetArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_naught, &psi_naught_read_only_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p); CHKERRXX(ierr);
  for (size_t i = 0; i<biomolecules->nodes->indep_nodes.elem_count; ++i) {
    psi_star_p[i] *= -1.0;
    if(phi_read_only_p[i] <= 1.5*biomolecules->parameters.layer_thickness() || (fabs(psi_naught_read_only_p[i]) > EPS)) // 1.5 == safety factor
      psi_bar_p[i]  = psi_star_p[i] + psi_naught_read_only_p[i];
    else
      psi_bar_p[i]  = 0.0;
  }
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_bar, &psi_bar_p); psi_bar_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_naught, &psi_naught_read_only_p); psi_naught_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_star, &psi_star_p); psi_star_p = NULL; CHKERRXX(ierr);

  // Once psi_bar is calculated, it needs to be extended over the interface so that its normal gradient can
  // be correctly calculated at the interface for imposing the jump condition on the normal gradient of
  // psi_hat afterwards.
  // If the normal gradient of psi_bar needs to be evaluated with order of accuracy 1 (resp. 2),
  // psi bar must be extended over the interface with a 2nd (resp. 3rd) order accurate method.
  // Therefore, the first (resp. and second) derivative(s) of psi_bar needs to be extended as well
  // NOTE: it is known that the Shortley-Weller method actually leads to 3rd order accurate results close
  // to the interface (leading in turn the superconvergence for the gradient). If psi_star is evaluated at
  // the interface with 3rd order accuracy (second order non-oscillatory interpolation method), the gradient
  // will be second-order accurate close to the interface and, hopefully, still second order accurate after
  // extension of psi_bar over the interface.
  biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, psi_bar, 20, biomolecules->parameters.order_of_accuracy());
  // 20 iterations for the extension over the interface (dafault parameter)
  // 2 in the last argument above no matter what is the desired order of accuracy (see comment above)

  psi_star_psi_naught_and_psi_bar_are_set = true;
  if(psi_star_xx != NULL){
    ierr = VecDestroy(psi_star_xx); psi_star_xx = NULL;}
  if(psi_star_yy != NULL){
    ierr = VecDestroy(psi_star_yy); psi_star_yy = NULL;}
#ifdef P4_TO_P8
  if(psi_star_zz != NULL){
    ierr = VecDestroy(psi_star_zz); psi_star_zz = NULL;}
#endif
}

void my_p4est_biomolecules_solver_t::calculate_jumps_in_normal_gradient(Vec &eps_grad_n_psi_hat_jump, bool validation_flag)
{
  P4EST_ASSERT(eps_grad_n_psi_hat_jump != NULL && biomolecules->phi != NULL && (psi_star_psi_naught_and_psi_bar_are_set || validation_flag));
  const double *phi_read_only_p = NULL , *psi_bar_read_only_p = NULL;
  double *eps_grad_n_psi_hat_jump_p = NULL;

  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p); CHKERRXX(ierr);
  if(!validation_flag)
  {
    ierr = VecGetArrayRead(psi_bar, &psi_bar_read_only_p); CHKERRXX(ierr);
  }
  ierr = VecGetArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  p4est_locidx_t node_idx;
  double n_x, n_y;
#ifdef P4_TO_P8
  double n_z;
#endif
  double xyz[P4EST_DIM];
  double norm_of_gradient;
  for (size_t k = 0; k < biomolecules->neighbors->layer_nodes.size(); ++k)
  {
    node_idx = biomolecules->neighbors->layer_nodes.at(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      norm_of_gradient  = 0.0;
      n_x               = qnnn.dx_central(phi_read_only_p); norm_of_gradient += SQR(n_x);
      n_y               = qnnn.dy_central(phi_read_only_p); norm_of_gradient += SQR(n_y);
#ifdef P4_TO_P8
      n_z               = qnnn.dz_central(phi_read_only_p); norm_of_gradient += SQR(n_z);
#endif
      norm_of_gradient  = MAX(sqrt(norm_of_gradient), EPS);
#ifdef P4_TO_P8
      n_z               /= norm_of_gradient;
#endif
      n_y               /= norm_of_gradient;
      n_x               /= norm_of_gradient;

      if(validation_flag)
      {
        node_xyz_fr_n(node_idx, biomolecules->p4est, biomolecules->nodes, xyz);
        eps_grad_n_psi_hat_jump_p[node_idx]  = (elec_rel_permittivity-mol_rel_permittivity)*
            (n_x*validation_function.x_derivative(xyz[0], xyz[1]
    #ifdef P4_TO_P8
            , xyz[2]
    #endif
            ) +n_y*validation_function.y_derivative(xyz[0], xyz[1]
    #ifdef P4_TO_P8
            , xyz[2]
    #endif
            )
    #ifdef P4_TO_P8
            +n_z*validation_function.z_derivative(xyz[0], xyz[1], xyz[2])
    #endif
            );
      }
      else
        eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*
            (n_x*qnnn.dx_central(psi_bar_read_only_p)
             +n_y*qnnn.dy_central(psi_bar_read_only_p)
     #ifdef P4_TO_P8
             +n_z*qnnn.dz_central(psi_bar_read_only_p)
     #endif
             );
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateBegin(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->neighbors->local_nodes.size(); ++k)
  {
    node_idx = biomolecules->neighbors->local_nodes.at(k);
    if(fabs(phi_read_only_p[node_idx]) <= (1.5*biomolecules->parameters.layer_thickness())) // 1.5 == safety factor
    {
      biomolecules->neighbors->get_neighbors(node_idx, qnnn);
      norm_of_gradient  = 0.0;
      n_x               = qnnn.dx_central(phi_read_only_p); norm_of_gradient += SQR(n_x);
      n_y               = qnnn.dy_central(phi_read_only_p); norm_of_gradient += SQR(n_y);
#ifdef P4_TO_P8
      n_z               = qnnn.dz_central(phi_read_only_p); norm_of_gradient += SQR(n_z);
#endif
      norm_of_gradient  = MAX(sqrt(norm_of_gradient), EPS);
#ifdef P4_TO_P8
      n_z               /= norm_of_gradient;
#endif
      n_y               /= norm_of_gradient;
      n_x               /= norm_of_gradient;

      if(validation_flag)
      {
        node_xyz_fr_n(node_idx, biomolecules->p4est, biomolecules->nodes, xyz);
        eps_grad_n_psi_hat_jump_p[node_idx]  = (elec_rel_permittivity-mol_rel_permittivity)*
            (n_x*validation_function.x_derivative(xyz[0], xyz[1]
    #ifdef P4_TO_P8
            , xyz[2]
    #endif
            ) +n_y*validation_function.y_derivative(xyz[0], xyz[1]
    #ifdef P4_TO_P8
            , xyz[2]
    #endif
            )
    #ifdef P4_TO_P8
            +n_z*validation_function.z_derivative(xyz[0], xyz[1], xyz[2])
    #endif
            );
      }
      else
        eps_grad_n_psi_hat_jump_p[node_idx]  = mol_rel_permittivity*
            (n_x*qnnn.dx_central(psi_bar_read_only_p)
             +n_y*qnnn.dy_central(psi_bar_read_only_p)
     #ifdef P4_TO_P8
             +n_z*qnnn.dz_central(psi_bar_read_only_p)
     #endif
             );
    }
    else
      eps_grad_n_psi_hat_jump_p[node_idx]  = 0.0; // irrelevant far away from the interface but let's set it to 0.0
  }
  ierr = VecGhostUpdateEnd(eps_grad_n_psi_hat_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(eps_grad_n_psi_hat_jump, &eps_grad_n_psi_hat_jump_p); eps_grad_n_psi_hat_jump_p = NULL; CHKERRXX(ierr);
  if(!validation_flag)
  {
    ierr = VecRestoreArrayRead(psi_bar, &psi_bar_read_only_p); psi_bar_read_only_p = NULL; CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
}

int     my_p4est_biomolecules_solver_t::solve_nonlinear(double upper_bound_residual, int it_max, bool validation_flag)
{
  int iter = 0;
  if (psi_hat_is_set && !validation_flag)
    return iter;
#ifdef CASL_THROWS
  int local_error = !are_all_parameters_set();
  string err_msg  = "my_p4est_biomolecules_solver_t::solve_nonlinear(...): some parameters are not set yet, the nonlinear problem can't be solved...";
  biomolecules->err_manager.check_my_local_error(local_error, err_msg);
  local_error     = it_max < 1;
  err_msg         = "my_p4est_biomolecules_solver_t::solve_nonlinear(...): the maximum number of iterations must be one at least!...";
  biomolecules->err_manager.check_my_local_error(local_error, err_msg);
  local_error     = ( (it_max > 1) && (upper_bound_residual <= 0.0));
  err_msg         = "my_p4est_biomolecules_solver_t::solve_nonlinear(...): the upper bound for the residual must be strictly positive!...";
  biomolecules->err_manager.check_my_local_error(local_error, err_msg);
#endif
  parStopWatch *log_timer = NULL, *solve_subtimer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Solving the Poisson-Boltzmann equation on a %d/%d grid with %d proc(s) \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), biomolecules->p4est->mpisize); CHKERRXX(ierr);
    if(validation_flag)
    {
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Solving for validation!!! \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), biomolecules->p4est->mpisize); CHKERRXX(ierr);
    }
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The ionic charge is %d \n", ion_charge); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The far-field electrolyte density is %g m^{-3} \n", far_field_ion_density); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The temperature is %g K \n", temperature); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The iverse debye length %g A^(-1), %g m^(-1), or %g in domain units\n", (get_inverse_debye_length_in_angstrom_inverse()), (get_inverse_debye_length_in_meters_inverse()), get_inverse_debye_length_in_domain()); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n"); CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Resolution of the nonlinear Poisson-Boltzmann Equation");
    }
  }
  if(biomolecules->timing_file != NULL)
  {
    P4EST_ASSERT(solve_subtimer == NULL);
    solve_subtimer = new parStopWatch(parStopWatch::root_timings, biomolecules->timing_file, biomolecules->p4est->mpicomm);
  }
  if(!psi_star_psi_naught_and_psi_bar_are_set && !validation_flag)
  {
    if(solve_subtimer != NULL)
      solve_subtimer->start("Solving for singular parts");
    solve_singular_part();
    if(solve_subtimer != NULL)
    {
      solve_subtimer->stop(); solve_subtimer->read_duration();
    }
  }
  P4EST_ASSERT(psi_bar != NULL || validation_flag);

  // we'll solve the nonlinear system using Newton iterations...
  // A first approximation is calculated by solving the linearized system
  //
  //  -div(eps*grad(psi_hat)) + (kappa^2)*psi_hat = 0.0, kappa = 0 in Omega^{-}, kappa > 0 in Omega^{+}
  //
  // with the jump conditions
  // ** [psi_hat] = 0.0;
  // ** [eps_r*normal_gradient_of_psi_hat] = mol_rel_permittivity*normal_gradient_of_psi_bar;
  //
  // and the boundary conditions
  // ** psi_hat = 0.0 on the walls.
  // That solution is stored in, say, psi_hat_n... Then, non-linear increments delta_psi_hat are successively
  // calculated based on the best current approximation psi_hat_n. Those increments are found by solving
  //
  //  -div(eps*grad(delta_psi_hat)) + (kappa^2)*cosh(psi_hat_n)*delta_psi_hat = div(eps*grad(psi_hat_n)) - (kappa^2)*sinh(psi_hat_n), kappa = 0 in Omega^{-}, kappa > 0 in Omega^{+}
  //
  // with the jump conditions
  // ** [delta_psi_hat] = 0.0;
  // ** [eps_r*normal_gradient_of_delta_psi_hat] = 0.0;
  //
  // and the boundary conditions
  // ** delta_psi_hat = 0.0 on the walls.
  // then the best current approximation is updated by psi_hat_n += delta_psi_hat.

  if(solve_subtimer != NULL)
    solve_subtimer->start("Initializing the solver");

  // Create a node-sampled zero vector (will be useful)
  Vec node_sampled_zero = NULL;
  make_sure_is_node_sampled(node_sampled_zero);
  // Create vector for the jump condition in normal gradient
  // Vec eps_grad_n_psi_hat_jump = NULL;

  Vec eps_grad_n_psi_hat_jump= NULL;
  Vec eps_grad_n_psi_hat_jump_xx_= NULL;
  Vec eps_grad_n_psi_hat_jump_yy_= NULL;
#ifdef P4_TO_P8
  Vec eps_grad_n_psi_hat_jump_zz_= NULL;
#endif

  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump);
  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump_xx_);
  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump_yy_);
#ifdef P4_TO_P8
  make_sure_is_node_sampled(eps_grad_n_psi_hat_jump_zz_);
#endif

  // Create vectors for the diagonal term in the outer domain
  Vec add_plus = NULL;
  make_sure_is_node_sampled(add_plus);

  // define rhs's (nonzero only for validation purposes)
  Vec rhs_minus = NULL, rhs_plus = NULL;
  if(validation_flag)
  {
    make_sure_is_node_sampled(rhs_minus);
    make_sure_is_node_sampled(rhs_plus);
  }

  // Initialization:
  calculate_jumps_in_normal_gradient(eps_grad_n_psi_hat_jump, validation_flag);
  biomolecules->neighbors->second_derivatives_central(eps_grad_n_psi_hat_jump, DIM(eps_grad_n_psi_hat_jump_xx_,eps_grad_n_psi_hat_jump_yy_,eps_grad_n_psi_hat_jump_zz_));
  my_p4est_interpolation_nodes_t eps_grad_n_psi_hat_jump_interp_(biomolecules->neighbors);
  eps_grad_n_psi_hat_jump_interp_.set_input(eps_grad_n_psi_hat_jump, DIM(eps_grad_n_psi_hat_jump_xx_,eps_grad_n_psi_hat_jump_yy_,eps_grad_n_psi_hat_jump_zz_),quadratic_non_oscillatory_continuous_v2);

  double *node_sampled_zero_p = NULL, *add_plus_p = NULL, *rhs_plus_p = NULL, *rhs_minus_p = NULL;
  ierr = VecGetArray(node_sampled_zero, &node_sampled_zero_p); CHKERRXX(ierr);
  ierr = VecGetArray(add_plus, &add_plus_p); CHKERRXX(ierr);
  if(validation_flag)
  {
    ierr = VecGetArray(rhs_minus, &rhs_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_plus, &rhs_plus_p); CHKERRXX(ierr);
  }
  const double inverse_square_debye_length_in_domain = SQR(get_inverse_debye_length_in_domain());
  double xyz[P4EST_DIM], lap_of_val_sol, val_sol;
  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    node_sampled_zero_p[k]  = 0.0;
    add_plus_p[k]           = inverse_square_debye_length_in_domain;
    if(validation_flag)
    {
      node_xyz_fr_n(k, biomolecules->p4est, biomolecules->nodes, xyz);
      val_sol               = validation_function(xyz[0], xyz[1]
    #ifdef P4_TO_P8
          , xyz[2]
    #endif
          );
      lap_of_val_sol        = validation_function.laplacian(xyz[0], xyz[1]
    #ifdef P4_TO_P8
          , xyz[2]
    #endif
          );
      rhs_minus_p[k]        = -mol_rel_permittivity*lap_of_val_sol;
      if(it_max == 1)
        rhs_plus_p[k]       = -elec_rel_permittivity*lap_of_val_sol + inverse_square_debye_length_in_domain*val_sol;
      else
        rhs_plus_p[k]       = -elec_rel_permittivity*lap_of_val_sol + inverse_square_debye_length_in_domain*sinh(val_sol);
    }
  }
  ierr = VecRestoreArray(add_plus, &add_plus_p); add_plus_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(node_sampled_zero, &node_sampled_zero_p); node_sampled_zero_p = NULL; CHKERRXX(ierr);
  if(validation_flag)
  {
    ierr = VecRestoreArray(rhs_plus, &rhs_plus_p); rhs_plus_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_minus, &rhs_minus_p); rhs_minus_p = NULL; CHKERRXX(ierr);
  }
  else
  {
    rhs_minus = node_sampled_zero;
    rhs_plus  = node_sampled_zero;
  }
  jump_solver->set_use_centroid_always(use_centroid_always);
  jump_solver->set_store_finite_volumes(store_finite_volumes);
  jump_solver->set_jump_scheme(jc_scheme);
  jump_solver->set_jump_sub_scheme(jc_sub_scheme);
  jump_solver->set_use_sc_scheme(sc_scheme);
  jump_solver->set_integration_order(integration_order);

  //solver.set_lip(10.5);

  jump_solver->set_lip(lip);

  Vec bdry_phi_vec_all[bdry_phi_max_num];
  Vec infc_phi_vec_all[infc_phi_max_num];

  //parameters for solver
  //n_um = 10; mag_um = 1; n_mu_m = 0; mag_mu_m = 80; n_diag_m = 1; mag_diag_m = 0;
  //n_up = 12; mag_up = 1; n_mu_p = 0; mag_mu_p = 2; n_diag_p = 1; mag_diag_p = 1;

  infc_phi_num = 1;
  bdry_phi_num = 0;

  infc_present_00 = 1; infc_geom_00 = 1; infc_opn_00 = MLS_INT;

  bdry_present_00 = 0;

  bool *bdry_present_all[] = { &bdry_present_00};

  int *bdry_geom_all[] = { &bdry_geom_00};

  int *bdry_opn_all[] = { &bdry_opn_00 };

  int *bc_coeff_all[] = { &bc_coeff_00 };

  double *bc_coeff_all_mag[] = { &bc_coeff_00_mag };

  int *bc_type_all[] = { &bc_type_00 };

  bool *infc_present_all[] = { &infc_present_00};

  int *infc_geom_all[] = { &infc_geom_00};

  int *infc_opn_all[] = { &infc_opn_00};

  int *jc_value_all[] = { &jc_value_00};

  int *jc_flux_all[] = { &jc_flux_00};


  for (int i = 0; i < infc_phi_max_num; ++i)
    if (*infc_present_all[i] == true)
    {
        jump_solver->add_interface((mls_opn_t) *infc_opn_all[i], biomolecules->phi, DIM(NULL, NULL, NULL), zero_cf, eps_grad_n_psi_hat_jump_interp_);
    }

//  dirichlet_bc.setWallTypes(dirichlet_bc_wall_type);
//  if(validation_flag)
//    dirichlet_bc.setWallValues(validation_function);
//  else
//    dirichlet_bc.setWallValues(homogeneous_dirichlet_bc_wall_value);
//  jump_solver->set_bc(dirichlet_bc);
  jump_solver->set_mu(mol_rel_permittivity, elec_rel_permittivity);
  class bc_wall_type_t : public WallBCDIM
  {
  public:
    BoundaryConditionType operator()(DIM(double, double, double)) const
    {
      return (BoundaryConditionType) bc_wtype;
    }
  } bc_wall_type;
  jump_solver->set_wc(bc_wall_type, validation_function);
  jump_solver->set_rhs(rhs_minus,rhs_plus);
  jump_solver->set_diag(node_sampled_zero,add_plus);
  jump_solver->set_use_taylor_correction(taylor_correction);
  jump_solver->set_kink_treatment(kink_special_treatment);
  jump_solver->set_rhs(rhs_minus, rhs_plus);
  make_sure_is_node_sampled(psi_hat);
  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "Solving nonlinear iterations ";
    solve_subtimer->start(timer_msg);
  }
  jump_solver->solve_nonlinear(psi_hat,upper_bound_residual,it_max,true);
  string timer_msg = "End of nonlinear iterations ";

  if(solve_subtimer != NULL)
  {
    solve_subtimer->stop(); solve_subtimer->read_duration();
    string timer_msg = "End of nonlinear iterations ";
    solve_subtimer->stop();
  }
//  if(solve_subtimer != NULL)
//  {
//    solve_subtimer->stop(); solve_subtimer->read_duration();
//    string timer_msg = "Solving iteration " + to_string(iter+1) + " (== linear equation)";
//    solve_subtimer->start(timer_msg);
//  }
//  jump_solver->solve(NULL, false, KSPBCGS, PCHYPRE, false, false);
//  Vec psi_hat_on_voronoi = NULL;
//  Vec pristine_diagonal_terms = NULL;
//  double two_norm_of_residual_voro = DBL_MAX;
//  iter++;
//  if(it_max > 1)
//  {
//    ierr = VecDuplicate(jump_solver->sol_voro, &psi_hat_on_voronoi); CHKERRXX(ierr);
//    ierr = VecCopy(jump_solver->sol_voro, psi_hat_on_voronoi); CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(psi_hat_on_voronoi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd  (psi_hat_on_voronoi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  else
//    psi_hat_on_voronoi = jump_solver->sol_voro;
//  get_linear_diagonal_terms(pristine_diagonal_terms);
//  clean_matrix_diagonal(pristine_diagonal_terms);
//  get_residual_at_voronoi_points_and_set_as_rhs(psi_hat_on_voronoi);
//  ierr = VecNorm(jump_solver->rhs, NORM_2, &two_norm_of_residual_voro); CHKERRXX(ierr);
//  if(biomolecules->log_file != NULL)
//  {
//    if(solve_subtimer != NULL)
//    {
//      solve_subtimer->stop(); solve_subtimer->read_duration();
//    }
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                        "2-norm of the residual after iteration %d: %g \n", iter, two_norm_of_residual_voro); CHKERRXX(ierr);
//  }

//  double *psi_hat_on_voronoi_p = NULL;
//  const double *psi_hat_on_voronoi_read_only_p = NULL, *delta_psi_hat_on_voronoi_read_only_p = NULL;
//  if(iter < it_max && solve_subtimer != NULL)
//  {
//    string timer_msg = "Solving nonlinear iterations";
//    solve_subtimer->start(timer_msg);
//  }
//  while (iter < it_max && two_norm_of_residual_voro > upper_bound_residual)
//  {
//    // add the appropriate diagonal terms to the pristine linear version
//    ierr = VecGetArrayRead(psi_hat_on_voronoi, &psi_hat_on_voronoi_read_only_p); CHKERRXX(ierr);
//    for(unsigned int n=0; n<jump_solver->num_local_voro; ++n)
//    {
//      PetscInt global_n_idx = n+jump_solver->voro_global_offset[biomolecules->p4est->mpirank];
//#ifdef P4_TO_P8
//      Point3 pc = jump_solver->voro_seeds[n];
//#else
//      Point2 pc = jump_solver->voro_seeds[n];
//#endif
//      if( (ABS(pc.x-jump_solver->xyz_min[0])<EPS || ABS(pc.x-jump_solver->xyz_max[0])<EPS ||
//           ABS(pc.y-jump_solver->xyz_min[1])<EPS || ABS(pc.y-jump_solver->xyz_max[1])<EPS
//     #ifdef P4_TO_P8
//           || ABS(pc.z-jump_solver->xyz_min[2])<EPS || ABS(pc.z-jump_solver->xyz_max[2])<EPS
//           ) && jump_solver->bc->wallType(pc.x,pc.y, pc.z)==DIRICHLET)
//#else
//           ) && jump_solver->bc->wallType(pc.x,pc.y)==DIRICHLET)
//#endif
//        continue;

//#ifdef P4_TO_P8
//      Voronoi3D voro;
//#else
//      Voronoi2D voro;
//#endif
//      jump_solver->compute_voronoi_cell(n, voro);

//#ifdef P4_TO_P8
//      const vector<ngbd3Dseed> *points;
//#else
//      const vector<Point2> *partition;
//      const vector<ngbd2Dseed> *points;
//      voro.get_partition(partition);
//#endif
//      voro.get_neighbor_seeds(points);

//#ifdef P4_TO_P8
//      double phi_n = jump_solver->interp_phi(pc.x, pc.y, pc.z);
//#else
//      double phi_n = jump_solver->interp_phi(pc.x, pc.y);
//#endif
//      double add_n = 0.0;

//      if(phi_n>0)
//        add_n = inverse_square_debye_length_in_domain*cosh(psi_hat_on_voronoi_read_only_p[n]);

//#ifndef P4_TO_P8
//      voro.compute_volume();
//#endif
//      double volume = voro.get_volume();
//      ierr = MatSetValue(jump_solver->A, global_n_idx, global_n_idx, volume*add_n, ADD_VALUES); CHKERRXX(ierr);

//    }
//    ierr = VecRestoreArrayRead(psi_hat_on_voronoi, &psi_hat_on_voronoi_read_only_p); psi_hat_on_voronoi_read_only_p = NULL; CHKERRXX(ierr);

//    /* assemble the matrix */
//    ierr = MatAssemblyBegin(jump_solver->A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
//    ierr = MatAssemblyEnd  (jump_solver->A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

//    ierr = KSPSolve(jump_solver->ksp, jump_solver->rhs, jump_solver->sol_voro); CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(jump_solver->sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd  (jump_solver->sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    ierr = VecGetArray(psi_hat_on_voronoi, &psi_hat_on_voronoi_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(jump_solver->sol_voro, &delta_psi_hat_on_voronoi_read_only_p); CHKERRXX(ierr);
//    for(unsigned int n=0; n<jump_solver->num_local_voro; ++n)
//      psi_hat_on_voronoi_p[n] += delta_psi_hat_on_voronoi_read_only_p[n];
//    ierr = VecRestoreArrayRead(jump_solver->sol_voro, &delta_psi_hat_on_voronoi_read_only_p); delta_psi_hat_on_voronoi_read_only_p = NULL; CHKERRXX(ierr);
//    ierr = VecRestoreArray(psi_hat_on_voronoi, &psi_hat_on_voronoi_p); psi_hat_on_voronoi_p = NULL; CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(psi_hat_on_voronoi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    clean_matrix_diagonal(pristine_diagonal_terms);
//    ierr = VecGhostUpdateEnd  (psi_hat_on_voronoi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    get_residual_at_voronoi_points_and_set_as_rhs(psi_hat_on_voronoi);
//    ierr = VecNorm(jump_solver->rhs, NORM_2, &two_norm_of_residual_voro); CHKERRXX(ierr);
//    iter++;
//    if(biomolecules->log_file != NULL){
//      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                          "2-norm of the residual after iteration %d: %g \n", iter, two_norm_of_residual_voro); CHKERRXX(ierr);}
//  }
//  if(biomolecules->log_file != NULL && it_max > 1){
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                        "Non-linear Poisson-Boltzmann equation solved after %d iterations of Newton's method\n", iter); CHKERRXX(ierr);}

//  // free allocated memory
//  if(solve_subtimer != NULL){
//    solve_subtimer->stop(); solve_subtimer->read_duration();
//    solve_subtimer->start("Cleaning memory and interpolating the solution back to the quad/oc-tree grid");}
  if(validation_flag)
  {
    ierr = VecDestroy(rhs_plus); rhs_plus = NULL; CHKERRXX(ierr);
    ierr = VecDestroy(rhs_minus); rhs_minus = NULL; CHKERRXX(ierr);
  }
  ierr = VecDestroy(add_plus); add_plus = NULL; CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump); eps_grad_n_psi_hat_jump = NULL; CHKERRXX(ierr);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump_xx_);
  ierr = VecDestroy(eps_grad_n_psi_hat_jump_yy_);
  ierr = VecDestroy(node_sampled_zero); node_sampled_zero = NULL; CHKERRXX(ierr);

//  // Create the vector for the general solution psi_hat (if needed)
//  make_sure_is_node_sampled(psi_hat);
//  // We deactivated the destruction of the solution for the purpose of the nonlinear solver
//  // so destroy it before setting the actual solution on sol_voro for interpolation purposes
//  if((jump_solver->sol_voro != NULL) && (it_max > 1)){
//    ierr = VecDestroy(jump_solver->sol_voro); jump_solver->sol_voro = NULL; CHKERRXX(ierr);}
//  jump_solver->sol_voro = psi_hat_on_voronoi; // that one will be destroyed at the solver destruction
//  jump_solver->interpolate_solution_from_voronoi_to_tree(psi_hat);

//  if(pristine_diagonal_terms != NULL){
//    ierr = VecDestroy(pristine_diagonal_terms); pristine_diagonal_terms = NULL; CHKERRXX(ierr);}
//  if(solve_subtimer != NULL){
//    solve_subtimer->stop(); solve_subtimer->read_duration();}

  if(validation_flag)
  {
    if(solve_subtimer != NULL){
      solve_subtimer->start("VALIDATION: evaluating the norms");}
    double max_error = -DBL_MAX, loc_error, error_2_norm, error_1_norm;
    const double *psi_hat_read_only_p = NULL;
    Vec negative_ones = NULL, absolute_error_1_norm = NULL, absolute_error_2_norm = NULL;
    make_sure_is_node_sampled(negative_ones);
    make_sure_is_node_sampled(absolute_error_1_norm);
    make_sure_is_node_sampled(absolute_error_2_norm);

    double *absolute_error_1_norm_p = NULL, *absolute_error_2_norm_p = NULL, *negative_ones_p = NULL;
    ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p); CHKERRXX(ierr);
    ierr = VecGetArray(absolute_error_1_norm, &absolute_error_1_norm_p); CHKERRXX(ierr);
    ierr = VecGetArray(absolute_error_2_norm, &absolute_error_2_norm_p); CHKERRXX(ierr);

    ierr = VecGetArray(negative_ones, &negative_ones_p); CHKERRXX(ierr);

    for (p4est_locidx_t k = 0; k < biomolecules->nodes->num_owned_indeps; ++k) {
      node_xyz_fr_n(k, biomolecules->p4est, biomolecules->nodes, xyz);
      loc_error = fabs(psi_hat_read_only_p[k] - validation_function(xyz[0], xyz[1]
    #ifdef P4_TO_P8
          , xyz[2]
    #endif
          ));
      absolute_error_1_norm_p[k]  = loc_error;
      absolute_error_2_norm_p[k]  = SQR(loc_error);
      max_error                   = MAX(max_error, loc_error);
      negative_ones_p[k]          = -1.0;
    }
    ierr = VecRestoreArray(negative_ones, &negative_ones_p); negative_ones_p = NULL; CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(negative_ones, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(absolute_error_2_norm, &absolute_error_2_norm_p); absolute_error_2_norm_p = NULL; CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(absolute_error_2_norm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(absolute_error_1_norm, &absolute_error_1_norm_p); absolute_error_1_norm_p = NULL; CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(absolute_error_1_norm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(psi_hat_on_voronoi, &psi_hat_on_voronoi_read_only_p); CHKERRXX(ierr);
//    double error_1_norm_voro = 0.0, error_2_norm_voro = 0.0;
//    for(unsigned int n=0; n<jump_solver->num_local_voro; ++n)
//    {
//#ifdef P4_TO_P8
//      Point3 pc = jump_solver->voro_seeds[n];
//      Voronoi3D voro;
//#else
//      Point2 pc = jump_solver->voro_seeds[n];
//      Voronoi2D voro;
//#endif
//      jump_solver->compute_voronoi_cell(n, voro);
//#ifndef P4_TO_P8
//      voro.compute_volume();
//#endif
//      double voro_volume = voro.get_volume();
//      loc_error           = fabs(psi_hat_on_voronoi_read_only_p[n] - validation_function(pc.x,pc.y
//                                                                                   #ifdef P4_TO_P8
//                                                                                         , pc.z
//                                                                                   #endif
//                                                                                         ));
//      error_1_norm_voro   += loc_error*voro_volume;
//      error_2_norm_voro   += SQR(loc_error)*voro_volume;
//      max_error_voro      = MAX(max_error_voro, loc_error);
//    }
//    ierr = VecRestoreArrayRead(psi_hat_on_voronoi, &psi_hat_on_voronoi_read_only_p); psi_hat_on_voronoi_read_only_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p); psi_hat_read_only_p = NULL; CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_error, 1, MPI_DOUBLE, MPI_MAX, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
    //mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_error_voro, 1, MPI_DOUBLE, MPI_MAX, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
    double domain_volume = biomolecules->domain_dim.at(0)*biomolecules->domain_dim.at(1)
    #ifdef P4_TO_P8
        *biomolecules->domain_dim.at(2)
    #endif
        ;

//    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_1_norm_voro, 1, MPI_DOUBLE, MPI_SUM, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
//    error_1_norm_voro /= domain_volume;
//    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_2_norm_voro, 1, MPI_DOUBLE, MPI_SUM, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
//    error_2_norm_voro = sqrt(error_2_norm_voro/domain_volume);
    ierr = VecGhostUpdateEnd(negative_ones, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(absolute_error_1_norm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    error_1_norm = integrate_over_negative_domain(biomolecules->p4est, biomolecules->nodes, negative_ones, absolute_error_1_norm)/domain_volume;
    ierr = VecGhostUpdateEnd(absolute_error_2_norm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    error_2_norm = sqrt(integrate_over_negative_domain(biomolecules->p4est, biomolecules->nodes, negative_ones, absolute_error_2_norm)/domain_volume);
    ierr = VecDestroy(negative_ones); negative_ones = NULL; CHKERRXX(ierr);
    validation_error = absolute_error_1_norm; // will either be returned or destructed at the solver destruction
    ierr = VecDestroy(absolute_error_2_norm); absolute_error_2_norm = NULL; CHKERRXX(ierr);
    if(biomolecules->log_file != NULL)
    {
      if(solve_subtimer != NULL){
        solve_subtimer->stop(); solve_subtimer->read_duration();}
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
                          "Error in 1-norm for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), error_1_norm); CHKERRXX(ierr);
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
                          "Error in 2-norm for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), error_2_norm); CHKERRXX(ierr);
      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
                          "Error in infinity norm for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), max_error); CHKERRXX(ierr);
//      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                          "Error in 1-norm, on the voronoi mesh, for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), error_1_norm_voro); CHKERRXX(ierr);
//      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                          "Error in 2-norm, on the voronoi mesh, for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), error_2_norm_voro); CHKERRXX(ierr);
//      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,
//                          "Error in infinity norm, at voronoi points, for a %d/%d grid = %g \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), max_error_voro); CHKERRXX(ierr);
//      ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file,"Saving voronoi mesh...\n"); CHKERRXX(ierr);
//      jump_solver->print_voronoi_VTK("/home/egan/workspace/projects/biomol/output/validation/voronoi");
   }
  }

  if(solve_subtimer != NULL){
    delete solve_subtimer; solve_subtimer = NULL;}

  if(log_timer != NULL)
  {
    log_timer->stop(); log_timer->read_duration();
    delete log_timer; log_timer = NULL;
  }

  psi_hat_is_set = true;
  return iter;
}

void my_p4est_biomolecules_solver_t::get_solvation_free_energy(bool validation_flag)
{
  if(validation_flag)
    return; // irrelevant for validation purposes
#ifndef P4_TO_P8
  // this makes sense only in 3D
#ifdef CASL_THROWS
  string my_msg = "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solvation free energy is not properly defined in 2D, forget it! \n    Returning... \n";
  ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, my_msg.c_str()); CHKERRXX(ierr);
#endif
  return;
#else
#ifdef CASL_THROWS
  int local_error = !are_all_parameters_set();
  string err_msg  = "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): some parameters are not set yet, the nonlinear problem can't be solved, the solvation free energy can't be calculated...";
  biomolecules->err_manager.check_my_local_error(local_error, err_msg);
#endif
  if(!psi_star_psi_naught_and_psi_bar_are_set)
  {
    psi_hat_is_set = false;
#ifdef CASL_THROWS
    err_msg  = "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solution psi_bar for the singular parts is not known yet, it will be calculated...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, err_msg.c_str()); CHKERRXX(ierr);
#endif
    solve_singular_part();
  }
  P4EST_ASSERT(psi_star_psi_naught_and_psi_bar_are_set);
  if(!psi_hat_is_set)
  {
#ifdef CASL_THROWS
    err_msg  = "my_p4est_biomolecules_solver_t::get_solvation_free_energy(): the solution of the general nonlinear Poisson-Boltzmann equation is not known, it will be calculated...\n";
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, err_msg.c_str()); CHKERRXX(ierr);
#endif
    solve_nonlinear(1.0e-8, 10000);
  }
  P4EST_ASSERT(psi_star_psi_naught_and_psi_bar_are_set);
  P4EST_ASSERT((psi_star != NULL) && (psi_naught != NULL) && (psi_bar != NULL) && (psi_hat != NULL));

  parStopWatch* log_timer = NULL;
  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, " \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "Calculating the solvation free energy a %d/%d grid with %d proc(s) \n", biomolecules->parameters.min_level(), biomolecules->parameters.max_level(), biomolecules->p4est->mpisize); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "------------------------------------------------------------------------------- \n"); CHKERRXX(ierr);
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "\n"); CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer = new parStopWatch(parStopWatch::all_timings, biomolecules->log_file, biomolecules->p4est->mpicomm);
      log_timer->start("Calculating the solvation free energy");
    }
  }


  // contribution from the electrolyte
  //std::cout << "Starting calculation of contribution of electrolyte \n";
  Vec integrand = NULL, psi_hat_plus_psi_naught = NULL;
  make_sure_is_node_sampled(integrand);
  make_sure_is_node_sampled(psi_hat_plus_psi_naught);
  double *integrand_p = NULL, *psi_hat_plus_psi_naught_p = NULL, *phi_p = NULL;
  const double *psi_hat_read_only_p = NULL, *psi_naught_read_only_p = NULL;
  ierr = VecGetArray(integrand, &integrand_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_hat_plus_psi_naught, &psi_hat_plus_psi_naught_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(psi_naught, &psi_naught_read_only_p); CHKERRXX(ierr);
  ierr = VecGetArray(biomolecules->phi, &phi_p); CHKERRXX(ierr);
  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
    if(phi_p[k] > 0.0)
    {
      //integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*sinh(psi_hat_read_only_p[k])-2.0*(cosh(psi_hat_read_only_p[k]) - 1.0)); // relevant value
      integrand_p[k] = kB*temperature*far_field_ion_density*(psi_hat_read_only_p[k]*psi_hat_read_only_p[k]*2); // relevant value
      psi_hat_plus_psi_naught_p[k] = 0.0; // irrelevant value
    }
    else
    {
      integrand_p[k] = 0.0;               // needs to be extrapolated (bc of jump on the normal derivative)
      psi_hat_plus_psi_naught_p[k] = psi_hat_read_only_p[k] + psi_naught_read_only_p[k]; // relevant value
    }
    phi_p[k] *= -1.0;                     // we need to integrate over the exterior domain --> reverse the levelset
  }
  ierr = VecRestoreArray(biomolecules->phi, &phi_p); phi_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_naught, &psi_naught_read_only_p); psi_naught_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p); psi_hat_read_only_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_hat_plus_psi_naught, &psi_hat_plus_psi_naught_p); psi_hat_plus_psi_naught_p = NULL; CHKERRXX(ierr);
  ierr = VecRestoreArray(integrand, &integrand_p); integrand_p = NULL; CHKERRXX(ierr);
  biomolecules->ls->extend_Over_Interface_TVD(biomolecules->phi, integrand, 20, 2); // 20 for the number of iterations, default parameter
  solvation_free_energy = integrate_over_negative_domain(biomolecules->p4est, biomolecules->nodes, biomolecules->phi, integrand)*(pow(length_scale_in_meter(), 3.0));
  //std::cout << "contribution of electrolyte ::  " << solvation_free_energy<< "\n";
  Vec phi_ghost_loc = NULL;
  ierr = VecGhostGetLocalForm(biomolecules->phi, &phi_ghost_loc); CHKERRXX(ierr);
  ierr = VecScale(phi_ghost_loc, -1.0); CHKERRXX(ierr); // reverse the levelset function to get back to original state
  ierr = VecGhostRestoreLocalForm(biomolecules->phi, &phi_ghost_loc); phi_ghost_loc = NULL; CHKERRXX(ierr);

  // contributions from singular point charges
  //std::cout << "line 5939 ok \n";
  double integral_contribution_from_singular_charges = 0.0;

  my_p4est_interpolation_nodes_t interpolate_psi_hat_plus_psi_naught(biomolecules->neighbors);
  interpolate_psi_hat_plus_psi_naught.set_input(psi_hat_plus_psi_naught, linear);

  int total_nb_charged_atoms = 0.0;
  for (int mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
  {
    const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
    total_nb_charged_atoms += mol.get_number_of_charged_atoms();
  }
  //std::cout << "total number of charged atoms ::  " << total_nb_charged_atoms<< "\n";
  int proc_has_atom_if_rank_below = MIN(total_nb_charged_atoms, biomolecules->p4est->mpisize);

  int first_charged_atom_idx          = MIN(biomolecules->p4est->mpirank*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int idx_of_charged_atom_after_last  = MIN((biomolecules->p4est->mpirank+1)*total_nb_charged_atoms/proc_has_atom_if_rank_below, total_nb_charged_atoms);
  int nb_atoms_for_me                 = idx_of_charged_atom_after_last - first_charged_atom_idx;

  vector<double> point_values_of_psi_hat_plus_psi_naught(nb_atoms_for_me, 0.0);
  int charged_atom_idx_offset         = 0;
  int global_charged_atom_idx;
  p4est_locidx_t local_idx = 0;
  double xyz_atom[3];
  //std::cout << "first charged atom idx  ::  " << first_charged_atom_idx << "\n";
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (int mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
      {
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;
          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            xyz_atom[0] = a->x;
            xyz_atom[1] = a->y;
            xyz_atom[2] = a->z;
            interpolate_psi_hat_plus_psi_naught.add_point(local_idx++, xyz_atom);
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }
  interpolate_psi_hat_plus_psi_naught.interpolate(point_values_of_psi_hat_plus_psi_naught.data());
  local_idx = 0;
  //std::cout << "first charged atom idx  ::  " << first_charged_atom_idx << "\n";
  //resetting charged_atom_idx_offset
  charged_atom_idx_offset         = 0;
  if(first_charged_atom_idx < total_nb_charged_atoms)
  {
    for (int mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      //std::cout << "line 5994 ok \n";
      //std::cout << "mol_idx ::  " << mol_idx << "\n";
      //std::cout << "biomolecules->nmol() ::  " << biomolecules->nmol() << "\n";
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      //std::cout << "charged_atom_idx offset ::  " << charged_atom_idx_offset<< "\n";
      //std::cout << "number of charged atoms of mol ::  " << mol.get_number_of_charged_atoms() << "\n";
      //std::cout << "first_charged_atom_idx ::  " << first_charged_atom_idx << "\n";
      //std::cout << "idx_of_charged_atom_after_last ::  " << idx_of_charged_atom_after_last<< "\n";
      if((charged_atom_idx_offset + mol.get_number_of_charged_atoms() >= first_charged_atom_idx) && (charged_atom_idx_offset < idx_of_charged_atom_after_last))
      {
          //std::cout << "number of charged atoms ::  " << mol.get_number_of_charged_atoms() << "\n";
        for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
        {
            //std::cout << "line 6002 ok \n";
            //std::cout << "charged_atom_idx ::  " << charged_atom_idx << "\n";
            //std::cout << "number of charged atoms ::  " << mol.get_number_of_charged_atoms() << "\n";
          global_charged_atom_idx = charged_atom_idx_offset + charged_atom_idx;
          //std::cout << "global_charged_atom_idx ::  " << global_charged_atom_idx<< "\n";
          if((first_charged_atom_idx <= global_charged_atom_idx) && (global_charged_atom_idx < idx_of_charged_atom_after_last))
          {
            const Atom* a = mol.get_charged_atom(charged_atom_idx);
            //std::cout << "line 5999 is ok\n";
            integral_contribution_from_singular_charges += (0.5*a->q*kB*temperature/((double) ion_charge))*(point_values_of_psi_hat_plus_psi_naught.at(local_idx++));
            P4EST_ASSERT(local_idx <= nb_atoms_for_me);
          }
        }
      }
      charged_atom_idx_offset += mol.get_number_of_charged_atoms();
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &integral_contribution_from_singular_charges, 1, MPI_DOUBLE, MPI_SUM, biomolecules->p4est->mpicomm); SC_CHECK_MPI(mpiret);
  //std::cout << "integral_contribution_from_singular_charges ::  " << integral_contribution_from_singular_charges<< "\n";
  solvation_free_energy += integral_contribution_from_singular_charges;

  ierr = VecDestroy(integrand); integrand = NULL; CHKERRXX(ierr);
  ierr = VecDestroy(psi_hat_plus_psi_naught); psi_hat_plus_psi_naught = NULL; CHKERRXX(ierr);

  if(biomolecules->log_file != NULL)
  {
    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->log_file, "The value of the solvation free energy is %g J, that is %g kcal/mol \n", solvation_free_energy, solvation_free_energy*avogadro_number*0.000239006); CHKERRXX(ierr);
    if(biomolecules->timing_file != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer; log_timer = NULL;
    }
  }
#endif
}

//void my_p4est_biomolecules_solver_t::get_residual_at_voronoi_points_and_set_as_rhs(const Vec& psi_hat_on_voronoi)
//{
//  P4EST_ASSERT(psi_hat_on_voronoi != NULL && jump_solver->rhs != NULL);
//  ierr = MatMult(jump_solver->A, psi_hat_on_voronoi, jump_solver->rhs); CHKERRXX(ierr);
//  double *residual_voro_p = NULL;
//  const double *psi_hat_on_voro_read_only_p = NULL;
//  ierr = VecGetArray(jump_solver->rhs, &residual_voro_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(psi_hat_on_voronoi, &psi_hat_on_voro_read_only_p); CHKERRXX(ierr);
//  const double inverse_square_debye_length_in_domain = SQR(1.0/get_debye_length_in_domain());
//  for (unsigned int n = 0; n < jump_solver->num_local_voro; ++n)
//  {
//#ifdef P4_TO_P8
//    Point3 pc = jump_solver->voro_seeds[n];
//#else
//    Point2 pc = jump_solver->voro_seeds[n];
//#endif
//    if( (ABS(pc.x-jump_solver->xyz_min[0])<EPS || ABS(pc.x-jump_solver->xyz_max[0])<EPS ||
//         ABS(pc.y-jump_solver->xyz_min[1])<EPS || ABS(pc.y-jump_solver->xyz_max[1])<EPS
//     #ifdef P4_TO_P8
//         || ABS(pc.z-jump_solver->xyz_min[2])<EPS || ABS(pc.z-jump_solver->xyz_max[2])<EPS
//         ) && jump_solver->bc->wallType(pc.x,pc.y, pc.z)==DIRICHLET)
//#else
//         ) && jump_solver->bc->wallType(pc.x,pc.y)==DIRICHLET)
//#endif
//    {
//#ifdef P4_TO_P8
//      residual_voro_p[n] = -residual_voro_p[n] + jump_solver->bc->wallValue(pc.x, pc.y, pc.z);
//#else
//      residual_voro_p[n] = -residual_voro_p[n] + jump_solver->bc->wallValue(pc.x, pc.y);
//#endif
//      continue;
//    }
//#ifdef P4_TO_P8
//    Voronoi3D voro;
//#else
//    Voronoi2D voro;
//#endif
//    jump_solver->compute_voronoi_cell(n, voro);

//#ifdef P4_TO_P8
//    const vector<ngbd3Dseed> *points;
//#else
//    const vector<Point2> *partition;
//    const vector<ngbd2Dseed> *points;
//    voro.get_partition(partition);
//#endif
//    voro.get_neighbor_seeds(points);

//    double mu_n, add_n, rhs_n;
//#ifdef P4_TO_P8
//    double phi_n = jump_solver->interp_phi(pc.x, pc.y, pc.z);
//#else
//    double phi_n = jump_solver->interp_phi(pc.x, pc.y);
//#endif
//    if(phi_n<0)
//    {
//#ifdef P4_TO_P8
//      rhs_n = jump_solver->rhs_m(pc.x, pc.y, pc.z);
//#else
//      rhs_n = jump_solver->rhs_m(pc.x, pc.y);
//#endif
//      mu_n  = mol_rel_permittivity;
//      add_n = 0.0;
//    }
//    else
//    {
//#ifdef P4_TO_P8
//      rhs_n = jump_solver->rhs_p(pc.x, pc.y, pc.z);
//#else
//      rhs_n = jump_solver->rhs_p(pc.x, pc.y);
//#endif
//      mu_n  = elec_rel_permittivity;
//      add_n = inverse_square_debye_length_in_domain*sinh(psi_hat_on_voro_read_only_p[n]);
//    }

//#ifndef P4_TO_P8
//    voro.compute_volume();
//#endif
//    double volume = voro.get_volume();

//    residual_voro_p[n] = rhs_n*volume-residual_voro_p[n] -add_n*volume;

//    for(unsigned int l=0; l<points->size(); ++l)
//    {
//#ifdef P4_TO_P8
//      double s = (*points)[l].s;
//#else
//      int k = (l+partition->size()-1) % partition->size();
//      double s = ((*partition)[k]-(*partition)[l]).norm_L2();
//#endif

//      if((*points)[l].n>=0)
//      {
//        /* regular point */
//#ifdef P4_TO_P8
//        Point3 pl = (*points)[l].p;
//        double phi_l = jump_solver->interp_phi(pl.x, pl.y, pl.z);
//#else
//        Point2 pl = (*points)[l].p;
//        double phi_l = jump_solver->interp_phi(pl.x, pl.y);
//#endif
//        double mu_l;

//        if(phi_l<0) mu_l = mol_rel_permittivity;
//        else        mu_l = elec_rel_permittivity;

//        double mu_harmonic = 2*mu_n*mu_l/(mu_n + mu_l);

//        if(phi_n*phi_l<0)
//        {
//#ifdef P4_TO_P8
//          Point3 p_ln = (pc+pl)/2;
//#else
//          Point2 p_ln = (pc+pl)/2;
//#endif

//#ifdef P4_TO_P8
//          residual_voro_p[n] -= mu_harmonic/mu_l * s/2 * (*jump_solver->mu_grad_u_jump)(p_ln.x, p_ln.y, p_ln.z);
//#else
//          residual_voro_p[n] -= mu_harmonic/mu_l * s/2 * (*jump_solver->mu_grad_u_jump)(p_ln.x, p_ln.y);
//#endif
//        }
//      }
//    }
//  }
//  ierr = VecRestoreArrayRead(psi_hat_on_voronoi, &psi_hat_on_voro_read_only_p); psi_hat_on_voro_read_only_p = NULL; CHKERRXX(ierr);
//  ierr = VecRestoreArray(jump_solver->rhs, &residual_voro_p); residual_voro_p = NULL; CHKERRXX(ierr);
//  ierr = VecGhostUpdateBegin(jump_solver->rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(jump_solver->rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//}
//void my_p4est_biomolecules_solver_t::get_linear_diagonal_terms(Vec& pristine_diagonal_terms)
//{
//  if(pristine_diagonal_terms == NULL)
//  {
//    ierr = VecDuplicate(jump_solver->sol_voro, &pristine_diagonal_terms); CHKERRXX(ierr);
//  }
//  ierr = MatGetDiagonal(jump_solver->A, pristine_diagonal_terms); CHKERRXX(ierr);
//  double *pristine_diagonal_terms_p = NULL;
//  ierr = VecGetArray(pristine_diagonal_terms, &pristine_diagonal_terms_p); CHKERRXX(ierr);
//  const double inverse_square_debye_length_in_domain = SQR(1.0/get_debye_length_in_domain());
//  for (unsigned int n = 0; n < jump_solver->num_local_voro; ++n)
//  {
//#ifdef P4_TO_P8
//    Point3 pc = jump_solver->voro_seeds[n];
//#else
//    Point2 pc = jump_solver->voro_seeds[n];
//#endif
//    if( (ABS(pc.x-jump_solver->xyz_min[0])<EPS || ABS(pc.x-jump_solver->xyz_max[0])<EPS ||
//         ABS(pc.y-jump_solver->xyz_min[1])<EPS || ABS(pc.y-jump_solver->xyz_max[1])<EPS
//     #ifdef P4_TO_P8
//         || ABS(pc.z-jump_solver->xyz_min[2])<EPS || ABS(pc.z-jump_solver->xyz_max[2])<EPS
//         ) && jump_solver->bc->wallType(pc.x,pc.y, pc.z)==DIRICHLET)
//#else
//         ) && jump_solver->bc->wallType(pc.x,pc.y)==DIRICHLET)
//#endif
//      continue;

//#ifdef P4_TO_P8
//    Voronoi3D voro;
//#else
//    Voronoi2D voro;
//#endif
//    jump_solver->compute_voronoi_cell(n, voro);

//#ifdef P4_TO_P8
//    const vector<ngbd3Dseed> *points;
//#else
//    const vector<Point2> *partition;
//    const vector<ngbd2Dseed> *points;
//    voro.get_partition(partition);
//#endif
//    voro.get_neighbor_seeds(points);

//    double add_n;
//#ifdef P4_TO_P8
//    double phi_n = jump_solver->interp_phi(pc.x, pc.y, pc.z);
//#else
//    double phi_n = jump_solver->interp_phi(pc.x, pc.y);
//#endif
//    if(phi_n<0)
//      add_n = 0.0;
//    else
//      add_n = inverse_square_debye_length_in_domain;

//#ifndef P4_TO_P8
//    voro.compute_volume();
//#endif
//    double volume = voro.get_volume();

//    pristine_diagonal_terms_p[n] -= add_n*volume;
//  }
//  ierr = VecRestoreArray(pristine_diagonal_terms, &pristine_diagonal_terms_p); pristine_diagonal_terms_p = NULL; CHKERRXX(ierr);
//  ierr = VecGhostUpdateBegin(pristine_diagonal_terms, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(pristine_diagonal_terms, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//}

//void my_p4est_biomolecules_solver_t::clean_matrix_diagonal(const Vec& pristine_diagonal)
//{
//  P4EST_ASSERT(pristine_diagonal != NULL);
//  const double *pristine_diagonal_read_only_p = NULL;
//  ierr = VecGetArrayRead(pristine_diagonal, &pristine_diagonal_read_only_p); CHKERRXX(ierr);
//  /* fill the matrix with the values */
//  for(unsigned int n=0; n<jump_solver->num_local_voro; ++n)
//  {
//    PetscInt global_n_idx = n+jump_solver->voro_global_offset[biomolecules->p4est->mpirank];
//    ierr = MatSetValue(jump_solver->A, global_n_idx, global_n_idx, pristine_diagonal_read_only_p[n], INSERT_VALUES); CHKERRXX(ierr);
//  }

//  /* assemble the matrix */
//  ierr = MatAssemblyBegin(jump_solver->A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
//  ierr = MatAssemblyEnd  (jump_solver->A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(pristine_diagonal, &pristine_diagonal_read_only_p); pristine_diagonal_read_only_p = NULL; CHKERRXX(ierr);
//}

//Vec my_p4est_biomolecules_solver_t::get_psi(double max_absolute_psi, bool validation_flag)
//{
//  if(validation_flag)
//    return NULL; // irrelevant
//  if(biomolecules->phi == NULL)
//  {
//#ifdef CASL_THROWS
//    string err_msg = "my_p4est_biomolecules_solver_t::get_psi(), the phi vector is not set, psi cannot be constructed... \n";
//    biomolecules->err_manager.print_message_and_abort(err_msg, 297792);
//#else
//    MPI_Abort(biomolecules->p4est->mpicomm, 297792);
//#endif
//  }
//  if(!psi_star_psi_naught_and_psi_bar_are_set)
//  {
//#ifdef CASL_THROWS
//    // print a warning
//    string message  = "my_p4est_biomolecules_solver_t::get_psi(), psi_bar and/or psi_star not set yet, they will be computed and created... \n";
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
//#endif
//    solve_singular_part(); // default number of iterations
//  }
//  if(!psi_hat_is_set)
//  {
//#ifdef CASL_THROWS
//    // print a warning
//    string message  = "my_p4est_biomolecules_solver_t::get_psi(), psi_hat is not set yet, it will be computed using the linear equation ... \n";
//    ierr = PetscFPrintf(biomolecules->p4est->mpicomm, biomolecules->error_file, message.c_str()); CHKERRXX(ierr);
//#endif
//    solve_linear();
//  }
//  P4EST_ASSERT(max_absolute_psi > 0.0);
//  Vec psi = NULL;
//  make_sure_is_node_sampled(psi);
//  const double *psi_bar_read_only_p = NULL, *psi_hat_read_only_p = NULL, *phi_read_only_p = NULL;
//  double* psi_p = NULL;
//  ierr = VecGetArrayRead(psi_bar, &psi_bar_read_only_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(psi_hat, &psi_hat_read_only_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(biomolecules->phi, &phi_read_only_p); CHKERRXX(ierr);
//  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
//  for (size_t k = 0; k < biomolecules->nodes->indep_nodes.elem_count; ++k) {
//    if(phi_read_only_p[k] < EPS && !ISINF(psi_bar_read_only_p[k]))
//      psi_p[k]  = psi_hat_read_only_p[k] + ((fabs(psi_bar_read_only_p[k]) < max_absolute_psi)? psi_bar_read_only_p[k]: (SIGN(psi_bar_read_only_p[k])*max_absolute_psi));
//    else
//      psi_p[k]  = psi_hat_read_only_p[k];
//  }
//  ierr = VecRestoreArray(psi, &psi_p); psi_p = NULL; CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(biomolecules->phi, &phi_read_only_p); phi_read_only_p = NULL; CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(psi_hat, &psi_hat_read_only_p);  psi_hat_read_only_p = NULL; CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(psi_bar, &psi_bar_read_only_p); psi_bar_read_only_p = NULL; CHKERRXX(ierr);
//  return psi;
//}

my_p4est_biomolecules_solver_t::~my_p4est_biomolecules_solver_t()
{
  if(node_solver != NULL){
    delete node_solver; node_solver = NULL;}
  if(jump_solver != NULL){
    delete jump_solver; jump_solver = NULL;}
  if(validation_error != NULL){
    ierr = VecDestroy(validation_error); validation_error = NULL; CHKERRXX(ierr);}
  if(psi_star != NULL){
    ierr = VecDestroy(psi_star); psi_star = NULL; CHKERRXX(ierr);}
  if(psi_naught != NULL){
    ierr = VecDestroy(psi_naught); psi_naught = NULL; CHKERRXX(ierr);}
  if(psi_bar != NULL){
    ierr = VecDestroy(psi_bar); psi_bar = NULL; CHKERRXX(ierr);}
  if(cell_neighbors != NULL){
    delete cell_neighbors; cell_neighbors = NULL;}
}

