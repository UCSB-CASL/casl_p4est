
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
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_shapes.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>
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
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>
#endif

#ifdef MATLAB_PROVIDED
#include <engine.h>
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
DEFINE_PARAMETER(pl, int, lmin, 5, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 5, "Max level of the tree");

DEFINE_PARAMETER(pl, int, num_splits,           5, "Number of recursive splits");
DEFINE_PARAMETER(pl, int, num_splits_per_split, 1, "Number of additional resolutions");

DEFINE_PARAMETER(pl, int, num_shifts_x_dir, 1, "Number of grid shifts in the x-direction");
DEFINE_PARAMETER(pl, int, num_shifts_y_dir, 1, "Number of grid shifts in the y-direction");
DEFINE_PARAMETER(pl, int, num_shifts_z_dir, 1, "Number of grid shifts in the z-direction");
#endif

DEFINE_PARAMETER(pl, int, iter_start, 0, "Skip n first iterations");
DEFINE_PARAMETER(pl, double, lip, 2, "Lipschitz constant");

DEFINE_PARAMETER(pl, bool, refine_strict,  0, "Refines every cell starting from the coarsest case if yes");
DEFINE_PARAMETER(pl, bool, refine_rand,    0, "Add randomness into adaptive grid");
DEFINE_PARAMETER(pl, bool, balance_grid,   0, "Enforce 1:2 ratio for adaptive grid");
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

DEFINE_PARAMETER(pl, int, n_example, 10, "Predefined example");

void set_example(int n_example)
{
  switch (n_example)
  {
    case 0: // no boundaries, no interfaces

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 0;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

      break;
    case 1: // sphere interior

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 1; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 1; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 1;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 1; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;
    case 2: // sphere exterior

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 1;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 2; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;
    case 3: // moderately star-shaped boundary

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 1;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 4; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;
    case 4: // highly star-shaped boundary

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 1;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 5; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;
    case 5: // triangle/tetrahedron example from (Bochkov&Gibou, JCP, 2019)

      n_um = 0; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = P4EST_DIM+1;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 13; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 1; bdry_geom_01 = 14; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 0; bc_type_01 = ROBIN;
      bdry_present_02 = 1; bdry_geom_02 = 15; bdry_opn_02 = MLS_INT; bc_coeff_02 = 2; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 1; bdry_geom_03 = 16; bdry_opn_03 = MLS_INT; bc_coeff_03 = 1; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;

    case 6: // union of spheres example from (Bochkov&Gibou, JCP, 2019)

      n_um = 3; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 2;

      infc_present_00 = 0; infc_geom_00 = 0;
      infc_present_01 = 0; infc_geom_01 = 0;
      infc_present_02 = 0; infc_geom_02 = 0;
      infc_present_03 = 0; infc_geom_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 6; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 1; bdry_geom_01 = 7; bdry_opn_01 = MLS_ADD; bc_coeff_01 = 3; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;

    case 7: // difference of spheres example from (Bochkov&Gibou, JCP, 2019)

      n_um = 4; mag_um = 1; n_mu_m = 1; mag_mu_m = 1; n_diag_m = 1; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 2;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 8; bdry_opn_00 = MLS_INT; bc_coeff_00 = 4; bc_coeff_00_mag = 1; bc_type_00 = ROBIN;
      bdry_present_01 = 1; bdry_geom_01 = 9; bdry_opn_01 = MLS_INT; bc_coeff_01 = 3; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;

    case 8: // three stars example from (Bochkov&Gibou, JCP, 2019)

      n_um = 6; mag_um = 1; n_mu_m = 1; mag_mu_m = 1; n_diag_m = 2; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 0;
      bdry_phi_num = 3;

      infc_present_00 = 0;
      infc_present_01 = 0;
      infc_present_02 = 0;
      infc_present_03 = 0;

      bdry_present_00 = 1; bdry_geom_00 = 10; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = DIRICHLET;
      bdry_present_01 = 1; bdry_geom_01 = 11; bdry_opn_01 = MLS_ADD; bc_coeff_01 = 5; bc_coeff_01_mag = 1; bc_type_01 = DIRICHLET;
      bdry_present_02 = 1; bdry_geom_02 = 12; bdry_opn_02 = MLS_INT; bc_coeff_02 = 6; bc_coeff_02_mag = 1; bc_type_02 = DIRICHLET;
      bdry_present_03 = 0; bdry_geom_03 =  0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = DIRICHLET;

      break;

    case 9: // shperical interface

      n_um = 11; mag_um = 1; n_mu_m = 0; mag_mu_m = 5; n_diag_m = 1; mag_diag_m = 1;
      n_up = 12; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 1; mag_diag_p = 1;

      infc_phi_num = 1;
      bdry_phi_num = 0;

      infc_present_00 = 1; infc_geom_00 = 1; infc_opn_00 = MLS_INT;
      infc_present_01 = 0; infc_geom_01 = 0; infc_opn_01 = MLS_INT;
      infc_present_02 = 0; infc_geom_02 = 0; infc_opn_02 = MLS_INT;
      infc_present_03 = 0; infc_geom_03 = 0; infc_opn_03 = MLS_INT;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

      break;

    case 10: // moderately star-shaped interface

      n_um = 11; mag_um = 1; n_mu_m = 1; mag_mu_m = 5; n_diag_m = 1; mag_diag_m = 1;
      n_up = 12; mag_up = 1; n_mu_p = 0; mag_mu_p = 1; n_diag_p = 1; mag_diag_p = 1;

      infc_phi_num = 1;
      bdry_phi_num = 0;

      infc_present_00 = 1; infc_geom_00 = 2; infc_opn_00 = MLS_INT;
      infc_present_01 = 0; infc_geom_01 = 0; infc_opn_01 = MLS_INT;
      infc_present_02 = 0; infc_geom_02 = 0; infc_opn_02 = MLS_INT;
      infc_present_03 = 0; infc_geom_03 = 0; infc_opn_03 = MLS_INT;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

      break;

    case 11: // highly star-shaped interface

      n_um = 11; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 12; mag_up = 1; n_mu_p = 0; mag_mu_p =  1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 1;
      bdry_phi_num = 0;

      infc_present_00 = 1; infc_geom_00 = 3; infc_opn_00 = MLS_INT;
      infc_present_01 = 0; infc_geom_01 = 0; infc_opn_01 = MLS_INT;
      infc_present_02 = 0; infc_geom_02 = 0; infc_opn_02 = MLS_INT;
      infc_present_03 = 0; infc_geom_03 = 0; infc_opn_03 = MLS_INT;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

      break;

    case 12: // curvy interface in an annular region from (Bochkov&Gibou, JCP, 2019)

      n_um = 11; mag_um = 1; n_mu_m = 1; mag_mu_m = 10; n_diag_m = 0; mag_diag_m = 1;
      n_up = 12; mag_up = 1; n_mu_p = 0; mag_mu_p =  1; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 1;
      bdry_phi_num = 1;

      infc_present_00 = 1; infc_geom_00 = 4; infc_opn_00 = MLS_INT;
      infc_present_01 = 0; infc_geom_01 = 0; infc_opn_01 = MLS_INT;
      infc_present_02 = 0; infc_geom_02 = 0; infc_opn_02 = MLS_INT;
      infc_present_03 = 0; infc_geom_03 = 0; infc_opn_03 = MLS_INT;

      bdry_present_00 = 1; bdry_geom_00 = 3; bdry_opn_00 = MLS_INT; bc_coeff_00 = 0; bc_coeff_00_mag = 1; bc_type_00 = DIRICHLET;
      bdry_present_01 = 0; bdry_geom_01 = 0; bdry_opn_01 = MLS_INT; bc_coeff_01 = 0; bc_coeff_01_mag = 1; bc_type_01 = ROBIN;
      bdry_present_02 = 0; bdry_geom_02 = 0; bdry_opn_02 = MLS_INT; bc_coeff_02 = 0; bc_coeff_02_mag = 1; bc_type_02 = ROBIN;
      bdry_present_03 = 0; bdry_geom_03 = 0; bdry_opn_03 = MLS_INT; bc_coeff_03 = 0; bc_coeff_03_mag = 1; bc_type_03 = ROBIN;

      break;

    case 13: // example form Maxime's multiphase paper

      XCODE( xmin = -2; xmax = 2 );
      YCODE( ymin = -2; ymax = 2 );
      ZCODE( zmin = -2; zmax = 2 );

      n_um = 1; mag_um = 1; n_mu_m = 0; mag_mu_m = 1; n_diag_m = 0; mag_diag_m = 1;
      n_up = 0; mag_up = 1; n_mu_p = 0; mag_mu_p = 1.e8; n_diag_p = 0; mag_diag_p = 1;

      infc_phi_num = 1;
      bdry_phi_num = 0;

      infc_present_00 = 1; infc_geom_00 = 1; infc_opn_00 = MLS_INT;
      infc_present_01 = 0; infc_geom_01 = 0; infc_opn_01 = MLS_INT;
      infc_present_02 = 0; infc_geom_02 = 0; infc_opn_02 = MLS_INT;
      infc_present_03 = 0; infc_geom_03 = 0; infc_opn_03 = MLS_INT;

      bdry_present_00 = 0;
      bdry_present_01 = 0;
      bdry_present_02 = 0;
      bdry_present_03 = 0;

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

int *bc_coeff_all[] = { &bc_coeff_00,
                        &bc_coeff_01,
                        &bc_coeff_02,
                        &bc_coeff_03 };

double *bc_coeff_all_mag[] = { &bc_coeff_00_mag,
                               &bc_coeff_01_mag,
                               &bc_coeff_02_mag,
                               &bc_coeff_03_mag };

int *bc_type_all[] = { &bc_type_00,
                       &bc_type_01,
                       &bc_type_02,
                       &bc_type_03 };

bool *infc_present_all[] = { &infc_present_00,
                             &infc_present_01,
                             &infc_present_02,
                             &infc_present_03 };

int *infc_geom_all[] = { &infc_geom_00,
                         &infc_geom_01,
                         &infc_geom_02,
                         &infc_geom_03 };

int *infc_opn_all[] = { &infc_opn_00,
                        &infc_opn_01,
                        &infc_opn_02,
                        &infc_opn_03 };

int *jc_value_all[] = { &jc_value_00,
                        &jc_value_01,
                        &jc_value_02,
                        &jc_value_03 };

int *jc_flux_all[] = { &jc_flux_00,
                       &jc_flux_01,
                       &jc_flux_02,
                       &jc_flux_03 };

// DIFFUSION COEFFICIENTS
class mu_all_cf_t : public CF_DIM
{
  int    *n;
  double *mag;
  cf_value_type_t what;
public:
  mu_all_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: switch (what) {
          case VAL: return (*mag);
          case DDX: return 0.;
          case DDY: return 0.;
#ifdef P4_TO_P8
          case DDZ: return 0.;
#endif
        }
      case 1: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*(1.+(0.2*cos(x)+0.3*sin(y))*sin(z));
          case DDX: return -0.2*(*mag)*sin(x)*sin(z);
          case DDY: return (*mag)*0.3*cos(y)*sin(z);
          case DDZ: return (*mag)*(0.2*cos(x)+0.3*sin(y))*cos(z);
#else
          case VAL: return (*mag)*(1. + (0.2*cos(x)+0.3*sin(y)));
          case DDX: return (*mag)*(-0.2)*sin(x);
          case DDY: return (*mag)*( 0.3)*cos(y);
#endif
        }
      case 2: switch (what) {
          case VAL: return (*mag)*(y*y*log(x+2.) + 4.);
          case DDX: return (*mag)*y*y/(x+2.);
          case DDY: return (*mag)*2.*y*log(x+2.);
#ifdef P4_TO_P8
          case DDZ: return 0.;
#endif
        }
      case 3: switch (what) {
          case VAL: return  (*mag)*(exp(-x));
          case DDX: return -(*mag)*(exp(-x));
          case DDY: return  0;
#ifdef P4_TO_P8
          case DDZ: return  0;
#endif
        }
      case 4: {
          XCODE( double X = (x-xmin)/(xmax-xmin) );
          YCODE( double Y = (y-ymin)/(ymax-ymin) );
          ZCODE( double Z = (z-zmin)/(zmax-zmin) );
          double v = 0.2;
          double w = 2.*PI;
          switch (what) {
#ifdef P4_TO_P8
            case VAL: return (*mag)*(1. + v*cos(w*(X+Y))*sin(w*(X-Y))*sin(w*Z));
            case DDX: return (*mag)*v*w/(xmax-xmin)*( cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)))*sin(w*Z);
            case DDY: return (*mag)*v*w/(ymax-ymin)*(-cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)))*sin(w*Z);
            case DDZ: return (*mag)*v*w/(zmax-zmin)*cos(w*(X+Y))*sin(w*(X-Y))*cos(w*Z);
#else
            case VAL: return (*mag)*(1. + v*cos(w*(X+Y))*sin(w*(X-Y)));
            case DDX: return (*mag)*v*w/(xmax-xmin)*( cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)));
            case DDY: return (*mag)*v*w/(ymax-ymin)*(-cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)));
#endif
          }
        }
    }
  }
};

mu_all_cf_t mu_m_cf(VAL, n_mu_m, mag_mu_m);
mu_all_cf_t mu_p_cf(VAL, n_mu_p, mag_mu_p);
mu_all_cf_t DIM(mux_m_cf(DDX, n_mu_m, mag_mu_m),
                muy_m_cf(DDY, n_mu_m, mag_mu_m),
                muz_m_cf(DDZ, n_mu_m, mag_mu_m));
mu_all_cf_t DIM(mux_p_cf(DDX, n_mu_p, mag_mu_p),
                muy_p_cf(DDY, n_mu_p, mag_mu_p),
                muz_p_cf(DDZ, n_mu_p, mag_mu_p));


// TEST SOLUTIONS
class u_pm_cf_t : public CF_DIM
{
public:
  int    *n;
  double *mag;
  cf_value_type_t what;
  u_pm_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
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
      case 7: {
        double p = mu_p_cf(DIM(x,y,z))/mu_m_cf(DIM(x,y,z));
        double r = sqrt(x*x+y*y);
        double r0 = 0.5+EPS;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
          case DDZ: return 0;
          case LAP: return 0;
#else
          case VAL: return (*mag)*( x*(p+1.) - x*(p-1.)*r0*r0/r/r )        /( p+1.+r0*r0*(p-1.) );
          case DDX: return (*mag)*( p+1.-r0*r0*(p-1.)*(y*y-x*x)/pow(r,4.) )/( p+1.+r0*r0*(p-1.) );
          case DDY: return (*mag)*( (p-1.)*r0*r0*2.*x*y/pow(r,4.) )        /( p+1.+r0*r0*(p-1.) );
          case LAP: return 0;
#endif
        }}
      case 8: {
        double p = mu_p_cf(DIM(x,y,z))/mu_m_cf(DIM(x,y,z));
        double r = sqrt(x*x+y*y);
        if (r < EPS) r = EPS;
        double r0 = 0.5+EPS;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
          case DDZ: return 0;
          case LAP: return 0;
#else
          case VAL: return (*mag)*2.*x/( p+1.+r0*r0*(p-1.) );
          case DDX: return (*mag)*2.  /( p+1.+r0*r0*(p-1.) );
          case DDY: return 0;
          case LAP: return 0;
#endif
        }}
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

u_pm_cf_t u_m_cf(VAL, n_um, mag_um),  u_p_cf(VAL, n_up, mag_up);
u_pm_cf_t ul_m_cf(LAP, n_um, mag_um), ul_p_cf(LAP, n_up, mag_up);
u_pm_cf_t DIM(ux_m_cf(DDX, n_um, mag_um),
              uy_m_cf(DDY, n_um, mag_um),
              uz_m_cf(DDZ, n_um, mag_um));
u_pm_cf_t DIM(ux_p_cf(DDX, n_up, mag_up),
              uy_p_cf(DDY, n_up, mag_up),
              uz_p_cf(DDZ, n_up, mag_up));

// DIAGONAL TERMS
class diag_cf_t: public CF_DIM
{
  int *n;
  double *mag;
public:
  diag_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: return 0.;
      case 1: return (*mag)*1.;
      case 2:
#ifdef P4_TO_P8
        return (*mag)*cos(x+z)*exp(y);
#else
        return (*mag)*cos(x)*exp(y);
#endif
    }
  }
};

diag_cf_t diag_m_cf(n_diag_m, mag_diag_m);
diag_cf_t diag_p_cf(n_diag_p, mag_diag_p);

// RHS
class rhs_m_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return diag_m_cf(DIM(x,y,z))*(u_m_cf(DIM(x,y,z)))
        - mu_m_cf(DIM(x,y,z))*ul_m_cf(DIM(x,y,z))
        - SUMD(mux_m_cf(DIM(x,y,z))*ux_m_cf(DIM(x,y,z)),
               muy_m_cf(DIM(x,y,z))*uy_m_cf(DIM(x,y,z)),
               muz_m_cf(DIM(x,y,z))*uz_m_cf(DIM(x,y,z)));
  }
} rhs_m_cf;

class rhs_p_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return diag_p_cf(DIM(x,y,z))*(u_p_cf(DIM(x,y,z)))
        - mu_p_cf(DIM(x,y,z))*ul_p_cf(DIM(x,y,z))
        - SUMD(mux_p_cf(DIM(x,y,z))*ux_p_cf(DIM(x,y,z)),
               muy_p_cf(DIM(x,y,z))*uy_p_cf(DIM(x,y,z)),
               muz_p_cf(DIM(x,y,z))*uz_p_cf(DIM(x,y,z)));
  }
} rhs_p_cf;

// ROBIN COEFFICIENTS
class bc_coeff_cf_t: public CF_DIM
{
  int    *n;
  double *mag;
public:
  bc_coeff_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: return (*mag);
      case 1: return (*mag)*(x+y);
      case 2: return (*mag)*(x-y + (x+y)*(x+y));
      case 3:
#ifdef P4_TO_P8
        return (*mag)*(sin(x+y)*cos(x-y)*log(z+4.));
#else
        return (*mag)*(sin(x+y)*cos(x-y));
#endif
      case 4:
#ifdef P4_TO_P8
        return (*mag)*(cos(x+y)*sin(x-y)*exp(z));
#else
        return (*mag)*(cos(x+y)*sin(x-y));
#endif
      case 5:
#ifdef P4_TO_P8
        return (*mag)*( 1.0 + sin(x+z)*cos(y+z) );
#else
        return (*mag)*( 1.0 + sin(x)*cos(y) );
#endif
      case 6:
#ifdef P4_TO_P8
        return (*mag)*(exp(x+y+z));
#else
        return (*mag)*(exp(x+y));
#endif
      case 7:
#ifdef P4_TO_P8
        return (*mag)*(1.0 + sin(x)*cos(y)*exp(z) );
#else
        return (*mag)*(1.0 + sin(x)*cos(y) );
#endif
    }
  }
};

bc_coeff_cf_t bc_coeff_cf_all[] = { bc_coeff_cf_t(bc_coeff_00, bc_coeff_00_mag),
                                    bc_coeff_cf_t(bc_coeff_01, bc_coeff_01_mag),
                                    bc_coeff_cf_t(bc_coeff_02, bc_coeff_02_mag),
                                    bc_coeff_cf_t(bc_coeff_03, bc_coeff_03_mag) };

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

// INTERFACE GEOMETRY
class infc_phi_cf_t: public CF_DIM {
public:
  int    *n;
  cf_value_type_t what;
  infc_phi_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: // no interface
        return -1;
      case 1: // sphere
      {
        static double r0 = 0.5;
        static double DIM( xc = 0, yc = 0, zc = 0 );
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0, -1);
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 2: // moderately star-shaped domain
      {
        static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0.15, -1);
        switch (what) {
          OCOMP( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 3: // highly star-shaped domain
      {
        static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
        static double N = 1;
        static double n[] = { 5.0};
        static double b[] = { .3/r0 };
        static double t[] = { 0.0};
        static radial_shaped_domain_t shape(r0, DIM(xc, yc, zc), -1, N, n, b, t);
        switch (what) {
          OCOMP( case VAL: return MIN(0.2, shape.phi  (DIM(x,y,z))) );
          XCOMP( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape.phi_z(DIM(x,y,z)) );
        }
      }
      case 4: // assymetric curvy domain
      {
        static double r0 = 0.483, DIM( xc = 0, yc = 0, zc = 0 );
        static double N = 3;
        static double n[] = { 7.0, 3.0, 4.0 };
        static double b[] = { .15, .10, -.10 };
        static double t[] = { 0.0, 0.5, 1.8 };
        static radial_shaped_domain_t shape(r0, DIM(xc, yc, zc), 1, N, n, b, t);
        switch (what) {
          OCOMP( case VAL: return shape.phi  (DIM(x,y,z)) );
          XCOMP( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCOMP( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCOMP( case DDZ: return shape.phi_z(DIM(x,y,z)) );
        }
      }
    }
  }
};


infc_phi_cf_t infc_phi_cf_all  [] = { infc_phi_cf_t(VAL, infc_geom_00),
                                      infc_phi_cf_t(VAL, infc_geom_01),
                                      infc_phi_cf_t(VAL, infc_geom_02),
                                      infc_phi_cf_t(VAL, infc_geom_03) };

infc_phi_cf_t infc_phi_x_cf_all[] = { infc_phi_cf_t(DDX, infc_geom_00),
                                      infc_phi_cf_t(DDX, infc_geom_01),
                                      infc_phi_cf_t(DDX, infc_geom_02),
                                      infc_phi_cf_t(DDX, infc_geom_03) };

infc_phi_cf_t infc_phi_y_cf_all[] = { infc_phi_cf_t(DDY, infc_geom_00),
                                      infc_phi_cf_t(DDY, infc_geom_01),
                                      infc_phi_cf_t(DDY, infc_geom_02),
                                      infc_phi_cf_t(DDY, infc_geom_03) };
#ifdef P4_TO_P8
infc_phi_cf_t infc_phi_z_cf_all[] = { infc_phi_cf_t(DDZ, infc_geom_00),
                                      infc_phi_cf_t(DDZ, infc_geom_01),
                                      infc_phi_cf_t(DDZ, infc_geom_02),
                                      infc_phi_cf_t(DDZ, infc_geom_03) };
#endif

// the effective LSF (initialized in main!)
mls_eff_cf_t bdry_phi_eff_cf;
mls_eff_cf_t infc_phi_eff_cf;

class phi_eff_cf_t : public CF_DIM
{
  CF_DIM *bdry_phi_cf_;
  CF_DIM *infc_phi_cf_;
public:
  phi_eff_cf_t(CF_DIM &bdry_phi_cf, CF_DIM &infc_phi_cf) : bdry_phi_cf_(&bdry_phi_cf), infc_phi_cf_(&infc_phi_cf) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return MAX( (*bdry_phi_cf_)(DIM(x,y,z)), -fabs((*infc_phi_cf_)(DIM(x,y,z))) );
  }
} phi_eff_cf(bdry_phi_eff_cf, infc_phi_eff_cf);

class mu_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? mu_p_cf(DIM(x,y,z)) : mu_m_cf(DIM(x,y,z));
  }
} mu_cf;
class u_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? u_p_cf(DIM(x,y,z)) : u_m_cf(DIM(x,y,z));
  }
} u_cf;
class ux_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const  {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? ux_p_cf(DIM(x,y,z)) : ux_m_cf(DIM(x,y,z));
  }
} ux_cf;
class uy_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? uy_p_cf(DIM(x,y,z)) : uy_m_cf(DIM(x,y,z));
  }
} uy_cf;
#ifdef P4_TO_P8
class uz_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? uz_p_cf(DIM(x,y,z)) : uz_m_cf(DIM(x,y,z));
  }
} uz_cf;
#endif

// BC VALUES
class bc_value_robin_t : public CF_DIM
{
  BoundaryConditionType *bc_type_;
  CF_DIM DIM(*phi_x_cf_,
             *phi_y_cf_,
             *phi_z_cf_);
  CF_DIM *bc_coeff_cf_;
public:
  bc_value_robin_t(BoundaryConditionType *bc_type,
                   CF_DIM *bc_coeff_cf,
                   DIM(CF_DIM *phi_x_cf,
                       CF_DIM *phi_y_cf,
                       CF_DIM *phi_z_cf))
    : bc_type_(bc_type),
      bc_coeff_cf_(bc_coeff_cf),
      DIM(phi_x_cf_(phi_x_cf),
          phi_y_cf_(phi_y_cf),
          phi_z_cf_(phi_z_cf)) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*bc_type_) {
      case DIRICHLET: return u_cf(DIM(x,y,z));
      case NEUMANN: {
          double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                      ny = (*phi_y_cf_)(DIM(x,y,z)),
                      nz = (*phi_z_cf_)(DIM(x,y,z)) );

          double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
          nx /= norm; ny /= norm; P8( nz /= norm );

          return mu_cf(DIM(x,y,z))*SUMD(nx*ux_cf(DIM(x,y,z)),
                                        ny*uy_cf(DIM(x,y,z)),
                                        nz*uz_cf(DIM(x,y,z)));
        }
      case ROBIN: {
          double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                      ny = (*phi_y_cf_)(DIM(x,y,z)),
                      nz = (*phi_z_cf_)(DIM(x,y,z)) );

          double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
          nx /= norm; ny /= norm; CODE3D( nz /= norm );

          return mu_cf(DIM(x,y,z))*SUMD(nx*ux_cf(DIM(x,y,z)),
                                        ny*uy_cf(DIM(x,y,z)),
                                        nz*uz_cf(DIM(x,y,z))) + (*bc_coeff_cf_)(DIM(x,y,z))*u_cf(DIM(x,y,z));
        }
    }
  }
};

bc_value_robin_t bc_value_cf_all[] = { bc_value_robin_t((BoundaryConditionType *)&bc_type_00, &bc_coeff_cf_all[0], DIM(&bdry_phi_x_cf_all[0], &bdry_phi_y_cf_all[0], &bdry_phi_z_cf_all[0])),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_01, &bc_coeff_cf_all[1], DIM(&bdry_phi_x_cf_all[1], &bdry_phi_y_cf_all[1], &bdry_phi_z_cf_all[1])),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_02, &bc_coeff_cf_all[2], DIM(&bdry_phi_x_cf_all[2], &bdry_phi_y_cf_all[2], &bdry_phi_z_cf_all[2])),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_03, &bc_coeff_cf_all[3], DIM(&bdry_phi_x_cf_all[3], &bdry_phi_y_cf_all[3], &bdry_phi_z_cf_all[3])) };

// JUMP CONDITIONS
class jc_value_cf_t : public CF_DIM
{
  int    *n;
  double *mag;
public:
  jc_value_cf_t(int &n) : n(&n) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(*n) {
      case 0: return u_p_cf(DIM(x,y,z)) - u_m_cf(DIM(x,y,z));
      case 1: return 0;
    }
  }
};

jc_value_cf_t jc_value_cf_all[] = { jc_value_cf_t(jc_value_00),
                                    jc_value_cf_t(jc_value_01),
                                    jc_value_cf_t(jc_value_02),
                                    jc_value_cf_t(jc_value_03) };

jc_value_cf_t jc_value_cf(jc_value_00);

class jc_flux_t : public CF_DIM
{
  int    *n;
  CF_DIM *phi_x_;
  CF_DIM *phi_y_;
  CF_DIM *phi_z_;
public:
  jc_flux_t(int &n, DIM(CF_DIM *phi_x, CF_DIM *phi_y, CF_DIM *phi_z)) :
    n(&n), DIM(phi_x_(phi_x), phi_y_(phi_y), phi_z_(phi_z)) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    switch(*n) {
      case 0:
      {
        double DIM( nx = (*phi_x_)(DIM(x,y,z)),
                    ny = (*phi_y_)(DIM(x,y,z)),
                    nz = (*phi_z_)(DIM(x,y,z)) );
        double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
        nx /= norm; ny /= norm; CODE3D( nz /= norm; )

            return mu_p_cf(DIM(x,y,z)) * SUMD(ux_p_cf(DIM(x,y,z))*nx, uy_p_cf(DIM(x,y,z))*ny, uz_p_cf(DIM(x,y,z))*nz)
            -  mu_m_cf(DIM(x,y,z)) * SUMD(ux_m_cf(DIM(x,y,z))*nx, uy_m_cf(DIM(x,y,z))*ny, uz_m_cf(DIM(x,y,z))*nz);
      }
      case 1: return 0;
    }
  }
};

jc_flux_t jc_flux_cf_all[] = { jc_flux_t(jc_flux_00, DIM(&infc_phi_x_cf_all[0], &infc_phi_y_cf_all[0], &infc_phi_z_cf_all[0])),
                               jc_flux_t(jc_flux_01, DIM(&infc_phi_x_cf_all[1], &infc_phi_y_cf_all[1], &infc_phi_z_cf_all[1])),
                               jc_flux_t(jc_flux_02, DIM(&infc_phi_x_cf_all[2], &infc_phi_y_cf_all[2], &infc_phi_z_cf_all[2])),
                               jc_flux_t(jc_flux_03, DIM(&infc_phi_x_cf_all[3], &infc_phi_y_cf_all[3], &infc_phi_z_cf_all[3])) };

//jc_flux_t jc_flux_cf_00( DIM(&infc_phi_x_cf_00, &infc_phi_y_cf_00, &infc_phi_z_cf_00) );

class bc_wall_type_t : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return (BoundaryConditionType) bc_wtype;
  }
} bc_wall_type;

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
} dom_perturb_cf(dom_perturb), ifc_perturb_cf(ifc_perturb);

// additional output functions
double compute_convergence_order(std::vector<double> &x, std::vector<double> &y);

int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: get_started_with_poisson_nodes_mls");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz[]      = { 1,  1,  1};
  const double xyz_min[] = {-1, -1, -1};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 0,  0,  0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } circle;
#endif

  splitting_criteria_cf_t sp(3, 8, &circle);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // save the grid into vtk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, "get_started_with_poisson_nodes_mls");

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

