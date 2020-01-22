/*
 * Title: stefan_mls
 * Description:
 * Author: Elyce
 * Date Created: 11-27-2019
 */

//#define P4_TO_P8

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
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_shapes.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex2_mls_q_vtk.h>


#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>

#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>

#include <src/my_p4est_macros.h>

#else
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_shapes.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/mls_integration/vtk/simplex3_mls_q_vtk.h>

#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>

#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>

#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>


#undef MIN
#undef MAX


using namespace std;

const int bdry_phi_max_num = 4;
const int bdry1_phi_max_num = 4;
const int infc_phi_max_num = 4;
const int infc1_phi_max_num = 4;

param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<int> px (pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<int> py (pl, 0, "py", "Periodicity in the y-direction (0/1)");
param_t<int> pz (pl, 0, "pz", "Periodicity in the z-direction (0/1)");

param_t<int> nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<double> xmin (pl, 0, "xmin", "Box xmin");
param_t<double> ymin (pl, 0, "ymin", "Box ymin");
param_t<double> zmin (pl, 0, "zmin", "Box zmin");

param_t<double> xmax (pl,  1, "xmax", "Box xmax");
param_t<double> ymax (pl,  1, "ymax", "Box ymax");
param_t<double> zmax (pl,  1, "zmax", "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
param_t<int> lmin (pl, 7, "lmin", "Min level of the tree");
param_t<int> lmax (pl, 9, "lmax", "Max level of the tree");

param_t<int> num_splits           (pl, 1, "num_splits", "Number of recursive splits");
param_t<int> num_splits_per_split (pl, 1, "num_splits_per_split", "Number of additional resolutions");

param_t<int> num_shifts_x_dir (pl, 1, "num_shifts_x_dir", "Number of grid shifts in the x-direction");
param_t<int> num_shifts_y_dir (pl, 1, "num_shifts_y_dir", "Number of grid shifts in the y-direction");
param_t<int> num_shifts_z_dir (pl, 1, "num_shifts_z_dir", "Number of grid shifts in the z-direction");
#else
param_t<int> lmin (pl, 6, "lmin", "Min level of the tree");
param_t<int> lmax (pl, 8, "lmax", "Max level of the tree");

param_t<int> num_splits           (pl, 1, "num_splits", "Number of recursive splits");
param_t<int> num_splits_per_split (pl, 1, "num_splits_per_split", "Number of additional resolutions");

param_t<int> num_shifts_x_dir (pl, 1, "num_shifts_x_dir", "Number of grid shifts in the x-direction");
param_t<int> num_shifts_y_dir (pl, 1, "num_shifts_y_dir", "Number of grid shifts in the y-direction");
param_t<int> num_shifts_z_dir (pl, 1, "num_shifts_z_dir", "Number of grid shifts in the z-direction");
#endif

param_t<int> iter_start (pl, 0, "iter_start", "Skip n first iterations");
param_t<double> lip (pl, 2.0, "lip", "Lipschitz constant");

param_t<bool> refine_strict  (pl, 0, "refine_strict", "Refines every cell starting from the coarsest case if yes");
param_t<bool> refine_rand    (pl, 0, "refine_rand", "Add randomness into adaptive grid");
param_t<bool> balance_grid   (pl, 0, "balance_grid", "Enforce 1:2 ratio for adaptive grid");
param_t<bool> coarse_outside (pl, 0, "coarse_outside", "Use the coarsest possible grid outside the domain (0/1)");
param_t<int>  expand_ghost   (pl, 0, "expand_ghost", "Number of ghost layer expansions");

// For debugging:
param_t<bool> check_memory_usage (pl, 0, "coarse_outside", "Whether to check memory usage or not(0/1)");

//-------------------------------------
//parameters for time
//-------------------------------------
param_t<double>  tstart   (pl, 0.0, "tstart", "Simulation start time (default: 0.0)");
param_t<double>  tfinal   (pl, 1.0, "tfinal", "Simulation end time (default: 1.4)");
param_t<int> method_      (pl, 1, "method_", "Timestepping method. : 1 = Backward Euler, 2 = Crank Nicholson (default 1)");
param_t<double>      dt   (pl, 0.01, "dt", "time step (default: 0.01)");
double tn; // the current simulation time
bool keep_going = true;
//-------------------------------------

//-------------------------------------
// test solutions
//-------------------------------------
// parameters for domain 1
//-------------------------------------
param_t<int> n_um (pl, 0, "n_um", "");
param_t<int> n_up (pl, 0, "n_up", "");

param_t<double> mag_um (pl, 1, "mag_um", "");
param_t<double> mag_up (pl, 1, "mag_up", "");

param_t<int> n_mu_m (pl, 0, "n_mu_m", "");
param_t<int> n_mu_p (pl, 0, "n_mu_p", "");

param_t<double> mag_mu_m (pl, 1, "mag_mu_m", "");
param_t<double> mag_mu_p (pl, 1, "mag_mu_p", "");

param_t<double> mu_iter_num (pl, 1, "mu_iter_num", "");
param_t<double> mag_mu_m_min (pl, 1, "mag_mu_m_min", "");
param_t<double> mag_mu_m_max (pl, 1, "mag_mu_m_max", "");

param_t<int> n_diag_m (pl, 0, "n_diag_m", "");
param_t<int> n_diag_p (pl, 0, "n_diag_p", "");

param_t<double> mag_diag_m (pl, 1, "mag_diag_m", "");
param_t<double> mag_diag_p (pl, 1, "mag_diag_p", "");

param_t<int> bc_wtype (pl, DIRICHLET, "bc_wtype", "Type of boundary conditions on the walls");

param_t<int>    nonlinear_term_m       (pl, 0, "nonlinear_term_m",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
param_t<int>    nonlinear_term_m_coeff (pl, 0, "nonlinear_term_m_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
param_t<double> nonlinear_term_m_mag   (pl, 1, "nonlinear_term_m_mag",   "Scaling of nonlinear term in negative domain");

param_t<int>    nonlinear_term_p       (pl, 0, "nonlinear_term_p",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
param_t<int>    nonlinear_term_p_coeff (pl, 0, "nonlinear_term_p_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
param_t<double> nonlinear_term_p_mag   (pl, 1, "nonlinear_term_p_mag",   "Scaling of nonlinear term in negative domain");

// boundary geometry
param_t<int> bdry_phi_num (pl, 0, "bdry_phi_num", "Domain geometry");
param_t<int> infc_phi_num (pl, 0, "infc_phi_num", "Domain geometry");

param_t<bool> bdry_present_00 (pl, 0, "bdry_present_00", "Domain geometry");
param_t<bool> bdry_present_01 (pl, 0, "bdry_present_01", "Domain geometry");
param_t<bool> bdry_present_02 (pl, 0, "bdry_present_02", "Domain geometry");
param_t<bool> bdry_present_03 (pl, 0, "bdry_present_03", "Domain geometry");

param_t<int> bdry_geom_00 (pl, 0, "bdry_geom_00", "Domain geometry");
param_t<int> bdry_geom_01 (pl, 0, "bdry_geom_01", "Domain geometry");
param_t<int> bdry_geom_02 (pl, 0, "bdry_geom_02", "Domain geometry");
param_t<int> bdry_geom_03 (pl, 0, "bdry_geom_03", "Domain geometry");

param_t<int> bdry_opn_00 (pl, MLS_INTERSECTION, "bdry_opn_00", "Domain geometry");
param_t<int> bdry_opn_01 (pl, MLS_INTERSECTION, "bdry_opn_01", "Domain geometry");
param_t<int> bdry_opn_02 (pl, MLS_INTERSECTION, "bdry_opn_02", "Domain geometry");
param_t<int> bdry_opn_03 (pl, MLS_INTERSECTION, "bdry_opn_03", "Domain geometry");

param_t<int> bc_coeff_00 (pl, 0, "bc_coeff_00", "Domain geometry");
param_t<int> bc_coeff_01 (pl, 0, "bc_coeff_01", "Domain geometry");
param_t<int> bc_coeff_02 (pl, 0, "bc_coeff_02", "Domain geometry");
param_t<int> bc_coeff_03 (pl, 0, "bc_coeff_03", "Domain geometry");

param_t<double> bc_coeff_00_mag (pl, 1, "bc_coeff_00_mag", "Domain geometry");
param_t<double> bc_coeff_01_mag (pl, 1, "bc_coeff_01_mag", "Domain geometry");
param_t<double> bc_coeff_02_mag (pl, 1, "bc_coeff_02_mag", "Domain geometry");
param_t<double> bc_coeff_03_mag (pl, 1, "bc_coeff_03_mag", "Domain geometry");

param_t<int> bc_type_00 (pl, DIRICHLET, "bc_type_00", "Type of boundary conditions on the domain boundary");
param_t<int> bc_type_01 (pl, DIRICHLET, "bc_type_01", "Type of boundary conditions on the domain boundary");
param_t<int> bc_type_02 (pl, DIRICHLET, "bc_type_02", "Type of boundary conditions on the domain boundary");
param_t<int> bc_type_03 (pl, DIRICHLET, "bc_type_03", "Type of boundary conditions on the domain boundary");

// interface geometry
param_t<bool> infc_present_00 (pl, 0, "infc_present_00", "Domain geometry");
param_t<bool> infc_present_01 (pl, 0, "infc_present_01", "Domain geometry");
param_t<bool> infc_present_02 (pl, 0, "infc_present_02", "Domain geometry");
param_t<bool> infc_present_03 (pl, 0, "infc_present_03", "Domain geometry");

param_t<int> infc_geom_00 (pl, 0, "infc_geom_00", "Domain geometry");
param_t<int> infc_geom_01 (pl, 0, "infc_geom_01", "Domain geometry");
param_t<int> infc_geom_02 (pl, 0, "infc_geom_02", "Domain geometry");
param_t<int> infc_geom_03 (pl, 0, "infc_geom_03", "Domain geometry");

param_t<int> infc_opn_00 (pl, MLS_INTERSECTION, "infc_opn_00", "Domain geometry");
param_t<int> infc_opn_01 (pl, MLS_INTERSECTION, "infc_opn_01", "Domain geometry");
param_t<int> infc_opn_02 (pl, MLS_INTERSECTION, "infc_opn_02", "Domain geometry");
param_t<int> infc_opn_03 (pl, MLS_INTERSECTION, "infc_opn_03", "Domain geometry");

param_t<int> jc_value_00 (pl, 0, "jc_value_00", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_value_01 (pl, 0, "jc_value_01", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_value_02 (pl, 0, "jc_value_02", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_value_03 (pl, 0, "jc_value_03", "0 - automatic (pl, others - hardcoded");

param_t<int> jc_flux_00 (pl, 0, "jc_flux_00", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_flux_01 (pl, 0, "jc_flux_01", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_flux_02 (pl, 0, "jc_flux_02", "0 - automatic (pl, others - hardcoded");
param_t<int> jc_flux_03 (pl, 0, "jc_flux_03", "0 - automatic (pl, others - hardcoded");

param_t<int> rhs_m_value (pl, 0, "rhs_m_value", "0 - automatic (pl, others - hardcoded");
param_t<int> rhs_p_value (pl, 0, "rhs_p_value", "0 - automatic (pl, others - hardcoded");

//param_t<int> bc_itype (pl, ROBIN, "bc_itype", "");
//-------------------------------------
//parameters for domain 2
//-------------------------------------
param_t<int> n_um1 (pl, 0, "n_um", "");
param_t<int> n_up1 (pl, 0, "n_up", "");

param_t<double> mag_um1 (pl, 1, "mag_um", "");
param_t<double> mag_up1 (pl, 1, "mag_up", "");

param_t<int> n_mu_m1 (pl, 0, "n_mu_m", "");
param_t<int> n_mu_p1 (pl, 0, "n_mu_p", "");

param_t<double> mag_mu_m1 (pl, 1, "mag_mu_m", "");
param_t<double> mag_mu_p1 (pl, 1, "mag_mu_p", "");

param_t<double> mu_iter_num1 (pl, 1, "mu_iter_num", "");
param_t<double> mag_mu_m_min1 (pl, 1, "mag_mu_m_min", "");
param_t<double> mag_mu_m_max1 (pl, 1, "mag_mu_m_max", "");

param_t<int> n_diag_m1 (pl, 0, "n_diag_m", "");
param_t<int> n_diag_p1 (pl, 0, "n_diag_p", "");

param_t<double> mag_diag_m1 (pl, 1, "mag_diag_m", "");
param_t<double> mag_diag_p1 (pl, 1, "mag_diag_p", "");

param_t<int> bc_wtype1 (pl, DIRICHLET, "bc_wtype", "Type of boundary conditions on the walls");

param_t<int>    nonlinear_term_m1      (pl, 0, "nonlinear_term_m",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
param_t<int>    nonlinear_term_m_coeff1 (pl, 0, "nonlinear_term_m_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
param_t<double> nonlinear_term_m_mag1   (pl, 1, "nonlinear_term_m_mag",   "Scaling of nonlinear term in negative domain");

param_t<int>    nonlinear_term_p1       (pl, 0, "nonlinear_term_p",       "Nonlinear term in negative domain: 0 - zero, 1 - linear, 2 - sinh, 3 - u/(1+u)");
param_t<int>    nonlinear_term_p_coeff1 (pl, 0, "nonlinear_term_p_coeff", "Coefficient form for nonlinear term in negative domain: 0 - constant, 1 - ... ");
param_t<double> nonlinear_term_p_mag1   (pl, 1, "nonlinear_term_p_mag",   "Scaling of nonlinear term in negative domain");

// boundary geometry
param_t<int> bdry1_phi_num (pl, 0, "bdry_phi_num", "Domain geometry");
param_t<int> infc1_phi_num (pl, 0, "infc_phi_num", "Domain geometry");

param_t<bool> bdry1_present_00 (pl, 0, "bdry_present_00", "Domain geometry");
param_t<bool> bdry1_present_01 (pl, 0, "bdry_present_01", "Domain geometry");
param_t<bool> bdry1_present_02 (pl, 0, "bdry_present_02", "Domain geometry");
param_t<bool> bdry1_present_03 (pl, 0, "bdry_present_03", "Domain geometry");

param_t<int> bdry1_geom_00 (pl, 0, "bdry_geom_00", "Domain geometry");
param_t<int> bdry1_geom_01 (pl, 0, "bdry_geom_01", "Domain geometry");
param_t<int> bdry1_geom_02 (pl, 0, "bdry_geom_02", "Domain geometry");
param_t<int> bdry1_geom_03 (pl, 0, "bdry_geom_03", "Domain geometry");

param_t<int> bdry1_opn_00 (pl, MLS_INTERSECTION, "bdry_opn_00", "Domain geometry");
param_t<int> bdry1_opn_01 (pl, MLS_INTERSECTION, "bdry_opn_01", "Domain geometry");
param_t<int> bdry1_opn_02 (pl, MLS_INTERSECTION, "bdry_opn_02", "Domain geometry");
param_t<int> bdry1_opn_03 (pl, MLS_INTERSECTION, "bdry_opn_03", "Domain geometry");

param_t<int> bc1_coeff_00 (pl, 0, "bc_coeff_00", "Domain geometry");
param_t<int> bc1_coeff_01 (pl, 0, "bc_coeff_01", "Domain geometry");
param_t<int> bc1_coeff_02 (pl, 0, "bc_coeff_02", "Domain geometry");
param_t<int> bc1_coeff_03 (pl, 0, "bc_coeff_03", "Domain geometry");

param_t<double> bc1_coeff_00_mag (pl, 1, "bc_coeff_00_mag", "Domain geometry");
param_t<double> bc1_coeff_01_mag (pl, 1, "bc_coeff_01_mag", "Domain geometry");
param_t<double> bc1_coeff_02_mag (pl, 1, "bc_coeff_02_mag", "Domain geometry");
param_t<double> bc1_coeff_03_mag (pl, 1, "bc_coeff_03_mag", "Domain geometry");

param_t<int> bc1_type_00 (pl, DIRICHLET, "bc_type_00", "Type of boundary conditions on the domain boundary");
param_t<int> bc1_type_01 (pl, DIRICHLET, "bc_type_01", "Type of boundary conditions on the domain boundary");
param_t<int> bc1_type_02 (pl, DIRICHLET, "bc_type_02", "Type of boundary conditions on the domain boundary");
param_t<int> bc1_type_03 (pl, DIRICHLET, "bc_type_03", "Type of boundary conditions on the domain boundary");

// interface geometry
param_t<bool> infc1_present_00 (pl, 0, "infc_present_00", "Domain geometry");
param_t<bool> infc1_present_01 (pl, 0, "infc_present_01", "Domain geometry");
param_t<bool> infc1_present_02 (pl, 0, "infc_present_02", "Domain geometry");
param_t<bool> infc1_present_03 (pl, 0, "infc_present_03", "Domain geometry");

param_t<int> infc1_geom_00 (pl, 0, "infc_geom_00", "Domain geometry");
param_t<int> infc1_geom_01 (pl, 0, "infc_geom_01", "Domain geometry");
param_t<int> infc1_geom_02 (pl, 0, "infc_geom_02", "Domain geometry");
param_t<int> infc1_geom_03 (pl, 0, "infc_geom_03", "Domain geometry");

param_t<int> infc1_opn_00 (pl, MLS_INTERSECTION, "infc_opn_00", "Domain geometry");
param_t<int> infc1_opn_01 (pl, MLS_INTERSECTION, "infc_opn_01", "Domain geometry");
param_t<int> infc1_opn_02 (pl, MLS_INTERSECTION, "infc_opn_02", "Domain geometry");
param_t<int> infc1_opn_03 (pl, MLS_INTERSECTION, "infc_opn_03", "Domain geometry");

param_t<int> jc1_value_00 (pl, 0, "jc_value_00", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_value_01 (pl, 0, "jc_value_01", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_value_02 (pl, 0, "jc_value_02", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_value_03 (pl, 0, "jc_value_03", "0 - automatic (pl, others - hardcoded");

param_t<int> jc1_flux_00 (pl, 0, "jc_flux_00", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_flux_01 (pl, 0, "jc_flux_01", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_flux_02 (pl, 0, "jc_flux_02", "0 - automatic (pl, others - hardcoded");
param_t<int> jc1_flux_03 (pl, 0, "jc_flux_03", "0 - automatic (pl, others - hardcoded");

param_t<int> rhs1_m_value (pl, 0, "rhs_m_value", "0 - automatic (pl, others - hardcoded");
param_t<int> rhs1_p_value (pl, 0, "rhs_p_value", "0 - automatic (pl, others - hardcoded");

//param_t<int> bc_itype (pl, ROBIN, "bc_itype", "");

//-------------------------------------
// solver parameters
//-------------------------------------
param_t<int>  jc_scheme         (pl, 0, "jc_scheme", "Discretization scheme for interface conditions (0 - FVM (pl, 1 - FDM)");
param_t<int>  jc_sub_scheme     (pl, 0, "jc_sub_scheme", "Interpolation subscheme for interface conditions (0 - from slow region (pl, 1 - from fast region (pl, 2 - based on nodes availability)");
param_t<int>  integration_order (pl, 2, "integration_order", "Select integration order (1 - linear (pl, 2 - quadratic)");
param_t<bool> sc_scheme         (pl, 0, "sc_scheme", "Use super-convergent scheme");

// for symmetric scheme:
param_t<bool> taylor_correction      (pl, 1, "taylor_correction", "Use Taylor correction to approximate Robin term (symmetric scheme)");
param_t<bool> kink_special_treatment (pl, 1, "kink_special_treatment", "Use the special treatment for kinks (symmetric scheme)");

// for superconvergent scheme:
param_t<bool> try_remove_hanging_cells (pl, 0, "try_remove_hanging_cells", "Ask solver to eliminate hanging cells");

param_t<bool> store_finite_volumes   (pl, 0, "store_finite_volumes", "");
param_t<bool> apply_bc_pointwise     (pl, 0, "apply_bc_pointwise", "");
param_t<bool> use_centroid_always    (pl, 0, "use_centroid_always", "");
param_t<bool> sample_bc_node_by_node (pl, 0, "sample_bc_node_by_node", "");

// for solving nonlinear equations
param_t<int>    nonlinear_method (pl, 1, "nonlinear_method", "Method to solve nonlinear eqautions: 0 - solving for solution itself, 1 - solving for change in the solution");
param_t<int>    nonlinear_itmax  (pl, 10, "nonlinear_itmax", "Maximum iteration for solving nonlinear equations");
param_t<double> nonlinear_tol    (pl, 1.e-10, "nonlinear_tol", "Tolerance for solving nonlinear equations");

//-------------------------------------
// level-set representation parameters
//-------------------------------------
param_t<bool> use_phi_cf       (pl, 0, "use_phi_cf", "Use analytical level-set functions");
param_t<bool> reinit_level_set (pl, 0, "reinit_level_set", "Reinitialize level-set function");

// artificial perturbation of level-set values
param_t<int>    dom_perturb     (pl, 0, "dom_perturb", "Artificially pertub level-set functions (0 - no perturbation (pl, 1 - smooth (pl, 2 - noisy)");
param_t<double> dom_perturb_mag (pl, 0.1, "dom_perturb_mag", "Magnitude of level-set perturbations");
param_t<double> dom_perturb_pow (pl, 2, "dom_perturb_pow", "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

param_t<int>    ifc_perturb     (pl, 0, "ifc_perturb", "Artificially pertub level-set functions (0 - no perturbation (pl, 1 - smooth (pl, 2 - noisy)");
param_t<double> ifc_perturb_mag (pl, 0.1e-6, "ifc_perturb_mag", "Magnitude of level-set perturbations");
param_t<double> ifc_perturb_pow (pl, 2, "ifc_perturb_pow", "Order of level-set perturbation (e.g. 2 for h^2 perturbations)");

//-------------------------------------
// convergence study parameters
//-------------------------------------
param_t<bool>   compute_cond_num       (pl, 0, "compute_cond_num", "Estimate L1-norm condition number");
param_t<int>    extend_solution        (pl, 0, "extend_solution", "Extend solution after solving: 0 - no extension (pl, 1 - extend using normal derivatives (pl, 2 - extend using all derivatives");
param_t<double> mask_thresh            (pl, 0, "mask_thresh", "Mask threshold for excluding points in convergence study");
param_t<bool>   compute_grad_between   (pl, 0, "compute_grad_between", "Computes gradient between points if yes");
param_t<bool>   scale_errors           (pl, 0, "scale_errors", "Scale errors by max solution/gradient value");
param_t<bool>   use_nonzero_guess      (pl, 0, "use_nonzero_guess", "");
param_t<double> extension_band_extend  (pl, 60, "extension_band_extend", "");
param_t<double> extension_band_compute (pl, 6, "extension_band_compute", "");
param_t<double> extension_band_check   (pl, 6, "extension_band_check", "");
param_t<double> extension_tol          (pl, -1.e-10, "extension_tol", "");
param_t<int>    extension_iterations   (pl, 100, "extension_iterations", "");


//-------------------------------------
// output
//-------------------------------------
param_t<bool> save_vtk           (pl, 1, "save_vtk", "Save the p4est in vtk format");
param_t<int>  save_every_iter    (pl, 1, "save_every_iter","Save every n (provided value) number of iterations (default:1)");
param_t<bool> print_checkpoints  (pl, 1, "print_checkpoints", "Print checkpoints");
param_t<bool> save_params        (pl, 0, "save_params", "Save list of entered parameters");
param_t<bool> save_domain        (pl, 0, "save_domain", "Save the reconstruction of an irregular domain (works only in serial!)");
param_t<bool> save_matrix_ascii  (pl, 0, "save_matrix_ascii", "Save the matrix in ASCII MATLAB format");
param_t<bool> save_matrix_binary (pl, 0, "save_matrix_binary", "Save the matrix in BINARY MATLAB format");
param_t<bool> save_convergence   (pl, 0, "save_convergence", "Save convergence results");

param_t<int> n_example (pl, 1, "n_example", "Predefined example:\n"
                                             "0 - no interfaces, no boudaries\n"
                                             "1 - sphere interior\n"
                                             "2 - sphere exterior\n"
                                             "3 - moderately flower-shaped domain\n"
                                             "4 - highly flower-shaped domain\n"
                                             "5 - triangle/tetrahedron example from (Bochkov&Gibou, JCP, 2019)\n"
                                             "6 - union of spheres example from (Bochkov&Gibou, JCP, 2019)\n"
                                             "7 - difference of spheres example from (Bochkov&Gibou, JCP, 2019)\n"
                                             "8 - three stars example from (Bochkov&Gibou, JCP, 2019)\n"
                                             "9 - shperical interface\n"
                                             "10 - moderately flower-shaped interface\n"
                                             "11 - highly flower-shaped interface\n"
                                             "12 - curvy interface in an annular region from (Bochkov&Gibou, JCP, 2019)\n"
                                             "13 - example form Maxime's multiphase paper\n"
                                             "14 - shperical interface - case 4 from voronoi jump solver3D\n"
                                             "15 - shperical interface - case 1 from voronoi jump solver3D\n"
                                             "16 - shperical interface - case 5 from voronoi jump solver3D\n"
                                             "17 - shperical interface - case 3 from voronoi jump solver2D\n"
                                             "18 - shperical interface - case 0 from voronoi jump solver3D\n"
                                             "19 - not defined\n"
                                             "20 - clusters of particles\n"
                                             "21 - same as no. 0 + nonlinear sinh term\n"
                                             "22 - same as no. 1 + nonlinear sinh term\n"
                                             "23 - same as no. 5 + nonlinear sinh term\n"
                                             "24 - same as no. 9 + nonlinear sinh term\n"
                                             "25 - same as no. 12 + nonlinear sinh term\n");

// define the poisson problem to solve


// specify exact solution of the problem in the in the positive and negative domain defined  in class u_pm_cf_t -> n_up and n_um
// specify  the scalar that is to be multiplied to the exact solution defined using the previous variables -> mag_up and mag_um
// specify mu of the problem in the in the positive and negative domain defined  in class u_pm_cf_t -> n_mu_p and n_mu_m
// specify  the scalar that is to be multiplied to mu defined using the previous variables -> mag_mu_p and mag_mu_m
// specify type of  diagonal term in the positive and negative domain by setting n_diag_m and n_diag_p to 0 or 1 or 2
// specify the scalar to be multiplied to the diagonal terms -> mag_diag_p and mag_diag_m



// define the domain geometry namely boundaries and interfaces
// specify number of boundaries and interfaces -> bdry_phi_num and infc_phi_num
// code is currently setup in a way that it can handle at max 4 boundaries and 4 interfaces and this can easily be extended further

// adding boundaries

// the value of "xx" in the following lines can be either 01, 02, 03 or 04 specifying the boundary  or interface number
// bdry_present_xx -> specifies if the bdry xx exists or not depending on whether the value is 1 or 0
// bdry_geom_xx -> defines the geometry of bdry xx defined in class bdry_phi_cf_t
// bdry_opn_xx -> can be MLS_INT or MLS_ADD
// As evident from the name choosing MLS_INTERSECTION will create a compound domain that is the intersection of two or more given domains
// and MLS_ADDITION will create a compound domain that is the union of two or more given domains
// bc_coeff_xx -> defines the type of robin boundary coefficient defined in bc_c0eff_cf_t
// bc_coeff_xx_mag -> defines the scalar to be multiplied with the Robin boundary coefficient specified using the above mentioned parameter
// bc_type_xx -> type of boundary coefficient DIRICHLET, NEUMANN or ROBIN defined in bc_wall_type_t


// adding interfaces is exactly similar to adding boundaries except in all the names above "bry" is replaced with "infc"
void set_example(int n_example)
{
  switch (n_example)
  {

    case 1: // sphere exterior

      n_um.val = 20; mag_um.val = 0; n_mu_m.val = 0; mag_mu_m.val = 0.1; n_diag_m.val = 0; mag_diag_m.val = 1;
      n_up.val = 20; mag_up.val = 0; n_mu_p.val = 0; mag_mu_p.val = 0.1; n_diag_p.val = 0; mag_diag_p.val = 1;

      infc_phi_num.val = 0;
      bdry_phi_num.val = 0;

      infc_present_00.val = 0; infc_geom_00.val = 0; infc_opn_00.val = MLS_INT;
      infc_present_01.val = 0; infc_geom_01.val = 0; infc_opn_01.val = MLS_INT;
      infc_present_02.val = 0; infc_geom_02.val = 0; infc_opn_02.val = MLS_INT;
      infc_present_03.val = 0; infc_geom_03.val = 0; infc_opn_03.val = MLS_INT;

      bdry_present_00.val = 0; bdry_geom_00.val = 0; bdry_opn_00.val = MLS_INT; bc_coeff_00.val = 0; bc_coeff_00_mag.val = 0; bc_type_00.val = DIRICHLET;
      bdry_present_01.val = 0; bdry_geom_01.val = 0; bdry_opn_01.val = MLS_INT; bc_coeff_01.val = 0; bc_coeff_01_mag.val = 1; bc_type_01.val = DIRICHLET;
      bdry_present_02.val = 0; bdry_geom_02.val = 0; bdry_opn_02.val = MLS_INT; bc_coeff_02.val = 0; bc_coeff_02_mag.val = 1; bc_type_02.val = DIRICHLET;
      bdry_present_03.val = 0; bdry_geom_03.val = 0; bdry_opn_03.val = MLS_INT; bc_coeff_03.val = 0; bc_coeff_03_mag.val = 1; bc_type_03.val = DIRICHLET;


      n_um1.val = 0; mag_um1.val =-5; n_mu_m1.val = 0; mag_mu_m1.val = 1; n_diag_m1.val = 0; mag_diag_m1.val = 1;
      n_up1.val = 0; mag_up1.val = 5; n_mu_p1.val = 0; mag_mu_p1.val = 1; n_diag_p1.val = 0; mag_diag_p1.val = 1;

      infc1_phi_num.val = 0;
      bdry1_phi_num.val = 0;

      infc1_present_00.val = 0;
      infc1_present_01.val = 0;
      infc1_present_02.val = 0;
      infc1_present_03.val = 0;

      bdry1_present_00.val = 1; bdry1_geom_00.val = 2; bdry1_opn_00.val = MLS_INT; bc1_coeff_00.val = 1; bc1_coeff_00_mag.val = 1; bc1_type_00.val = NEUMANN;
      bdry1_present_01.val = 0; bdry1_geom_01.val = 0; bdry1_opn_01.val = MLS_INT; bc1_coeff_01.val = 0; bc1_coeff_01_mag.val = 1; bc1_type_01.val = ROBIN;
      bdry1_present_02.val = 0; bdry1_geom_02.val = 0; bdry1_opn_02.val = MLS_INT; bc1_coeff_02.val = 0; bc1_coeff_02_mag.val = 1; bc1_type_02.val = ROBIN;
      bdry1_present_03.val = 0; bdry1_geom_03.val = 0; bdry1_opn_03.val = MLS_INT; bc1_coeff_03.val = 0; bc1_coeff_03_mag.val = 1; bc1_type_03.val = ROBIN;

      break;
  }
}


bool *bdry_present_all[] = { &bdry_present_00.val,
                             &bdry_present_01.val,
                             &bdry_present_02.val,
                             &bdry_present_03.val };

int *bdry_geom_all[] = { &bdry_geom_00.val,
                         &bdry_geom_01.val,
                         &bdry_geom_02.val,
                         &bdry_geom_03.val };

int *bdry_opn_all[] = { &bdry_opn_00.val,
                        &bdry_opn_01.val,
                        &bdry_opn_02.val,
                        &bdry_opn_03.val };

int *bc_coeff_all[] = { &bc_coeff_00.val,
                        &bc_coeff_01.val,
                        &bc_coeff_02.val,
                        &bc_coeff_03.val };

double *bc_coeff_all_mag[] = { &bc_coeff_00_mag.val,
                               &bc_coeff_01_mag.val,
                               &bc_coeff_02_mag.val,
                               &bc_coeff_03_mag.val };

int *bc_type_all[] = { &bc_type_00.val,
                       &bc_type_01.val,
                       &bc_type_02.val,
                       &bc_type_03.val };

bool *infc_present_all[] = { &infc_present_00.val,
                             &infc_present_01.val,
                             &infc_present_02.val,
                             &infc_present_03.val };

int *infc_geom_all[] = { &infc_geom_00.val,
                         &infc_geom_01.val,
                         &infc_geom_02.val,
                         &infc_geom_03.val };

int *infc_opn_all[] = { &infc_opn_00.val,
                        &infc_opn_01.val,
                        &infc_opn_02.val,
                        &infc_opn_03.val };

int *jc_value_all[] = { &jc_value_00.val,
                        &jc_value_01.val,
                        &jc_value_02.val,
                        &jc_value_03.val };

int *jc_flux_all[] = { &jc_flux_00.val,
                       &jc_flux_01.val,
                       &jc_flux_02.val,
                       &jc_flux_03.val };

bool *bdry1_present_all[] = { &bdry1_present_00.val,
                             &bdry1_present_01.val,
                             &bdry1_present_02.val,
                             &bdry1_present_03.val };

int *bdry1_geom_all[] = { &bdry1_geom_00.val,
                         &bdry1_geom_01.val,
                         &bdry1_geom_02.val,
                         &bdry1_geom_03.val };

int *bdry1_opn_all[] = { &bdry1_opn_00.val,
                        &bdry1_opn_01.val,
                        &bdry1_opn_02.val,
                        &bdry1_opn_03.val };

int *bc1_coeff_all[] = {&bc1_coeff_00.val,
                        &bc1_coeff_01.val,
                        &bc1_coeff_02.val,
                        &bc1_coeff_03.val };

double *bc1_coeff_all_mag[] = { &bc1_coeff_00_mag.val,
                               &bc1_coeff_01_mag.val,
                               &bc1_coeff_02_mag.val,
                               &bc1_coeff_03_mag.val };

int *bc1_type_all[] = {&bc1_type_00.val,
                       &bc1_type_01.val,
                       &bc1_type_02.val,
                       &bc1_type_03.val };

bool *infc1_present_all[] = {&infc1_present_00.val,
                             &infc1_present_01.val,
                             &infc1_present_02.val,
                             &infc1_present_03.val };

int *infc1_geom_all[] = {&infc1_geom_00.val,
                         &infc1_geom_01.val,
                         &infc1_geom_02.val,
                         &infc1_geom_03.val };

int *infc1_opn_all[] = {&infc1_opn_00.val,
                        &infc1_opn_01.val,
                        &infc1_opn_02.val,
                        &infc1_opn_03.val };

int *jc1_value_all[] = {&jc1_value_00.val,
                        &jc1_value_01.val,
                        &jc1_value_02.val,
                        &jc1_value_03.val };

int *jc1_flux_all[] = {&jc1_flux_00.val,
                       &jc1_flux_01.val,
                       &jc1_flux_02.val,
                       &jc1_flux_03.val };

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
          XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
          YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
          ZCODE( double Z = (z-zmin.val)/(zmax.val-zmin.val) );
          double v = 0.2;
          double w = 2.*PI;
          switch (what) {
#ifdef P4_TO_P8
            case VAL: return (*mag)*(1. + v*cos(w*(X+Y))*sin(w*(X-Y))*sin(w*Z));
            case DDX: return (*mag)*v*w/(xmax.val-xmin.val)*( cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)))*sin(w*Z);
            case DDY: return (*mag)*v*w/(ymax.val-ymin.val)*(-cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)))*sin(w*Z);
            case DDZ: return (*mag)*v*w/(zmax.val-zmin.val)*cos(w*(X+Y))*sin(w*(X-Y))*cos(w*Z);
#else
            case VAL: return (*mag)*(1. + v*cos(w*(X+Y))*sin(w*(X-Y)));
            case DDX: return (*mag)*v*w/(xmax.val-xmin.val)*( cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)));
            case DDY: return (*mag)*v*w/(ymax.val-ymin.val)*(-cos(w*(X+Y))*cos(w*(X-Y)) - sin(w*(X+Y))*sin(w*(X-Y)));
#endif
          }
      }
      case 5: {
        XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
        YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
        switch (what) {
          case VAL: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*log((x-0.5*(xmax.val+xmin.val))/(xmax.val- xmin.val)+2) +4;
          case DDX: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*((+1)/(x-0.5*(xmax.val+xmin.val)+2*(xmax.val-xmin.val)));
          case DDY: return (*mag)*(2.0*(y-ymin.val)/SQR(ymax.val-ymin.val))*log((x-0.5*(xmax.val+xmin.val))/(xmax.val- xmin.val)+2);
        }
      }//works
      case 6: {
        XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
        YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
        switch (what) {
          case VAL: return (*mag)*exp((-1)*(y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val));
          case DDX: return 0.0;
          case DDY: return (*mag)*exp((-1)*(y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((-1.0)/(ymax.val-ymin.val));
        }
      }//works
#ifdef P4_TO_P8
      case 7: {
        XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
        YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
        switch (what) {
          case VAL: return (*mag)*exp((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val)+(z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val));
          case DDX: return (*mag)*exp((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val)+(z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*(1/(xmax.val-xmin.val));
          case DDY: return 0.0;
          case DDZ: return (*mag)*exp((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val)+(z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*(1/(zmax.val-zmin.val));
        }
      }
#endif
      case 8: {
        XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
        YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
        switch (what) {
          case VAL: return (*mag)*SQR((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))+5;
          case DDX: return 0.0;
          case DDY: return (*mag)*(2.*(y-0.5*(ymin.val+ymax.val))/SQR(ymax.val-ymin.val));
#ifdef P4_TO_P8
          case DDZ: return 0.;
#endif
        }
      }

      case 9: {
        XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
        YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
        switch (what) {
          case VAL: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*log((x-xmin.val)/(xmax.val- xmin.val)+2) +4;
          case DDX: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*((+1.)/(x-xmin.val+2.*(xmax.val-xmin.val)));
          case DDY: return (*mag)*((2*(y-ymin.val))/SQR(ymax.val-ymin.val))*log((x-xmin.val)/(xmax.val- xmin.val)+2);
#ifdef P4_TO_P8
          case DDZ: return 0.0;
#endif
        }
      }
#ifdef P4_TO_P8
      case 10: {
        switch (what) {
          case VAL: return (*mag)*exp(-(z-zmin.val)/(zmax.val-zmin.val));
          case DDX: return 0.0;
          case DDY: return 0.0;
          case DDZ: return (*mag)*exp(-(z-zmin.val)/(zmax.val-zmin.val))*(-1./(zmax.val-zmin.val));
        }
      }
#endif
      default:
        throw std::invalid_argument("Invalid diffusion coefficient");
    }
  }
};

mu_all_cf_t mu_m_cf(VAL, n_mu_m.val, mag_mu_m.val);
mu_all_cf_t mu_p_cf(VAL, n_mu_p.val, mag_mu_p.val);
mu_all_cf_t DIM(mux_m_cf(DDX, n_mu_m.val, mag_mu_m.val),
                muy_m_cf(DDY, n_mu_m.val, mag_mu_m.val),
                muz_m_cf(DDZ, n_mu_m.val, mag_mu_m.val));
mu_all_cf_t DIM(mux_p_cf(DDX, n_mu_p.val, mag_mu_p.val),
                muy_p_cf(DDY, n_mu_p.val, mag_mu_p.val),
                muz_p_cf(DDZ, n_mu_p.val, mag_mu_p.val));

mu_all_cf_t mu_m_cf1(VAL, n_mu_m1.val, mag_mu_m1.val);
mu_all_cf_t mu_p_cf1(VAL, n_mu_p1.val, mag_mu_p1.val);
mu_all_cf_t DIM(mux_m_cf1(DDX, n_mu_m1.val, mag_mu_m1.val),
                muy_m_cf1(DDY, n_mu_m1.val, mag_mu_m1.val),
                muz_m_cf1(DDZ, n_mu_m1.val, mag_mu_m1.val));
mu_all_cf_t DIM(mux_p_cf1(DDX, n_mu_p1.val, mag_mu_p1.val),
                muy_p_cf1(DDY, n_mu_p1.val, mag_mu_p1.val),
                muz_p_cf1(DDZ, n_mu_p1.val, mag_mu_p1.val));


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
      case 13: switch (what){
          case VAL: return (*mag)*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)));
          case DDX: return (*mag)*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)))*(1/((xmax.val-xmin.val)));
          case DDY: return 0.0;
          case LAP: return (*mag)*(1/(SQR(xmax.val-xmin.val)))*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)));
        }
      case 14: switch (what){
#ifdef P4_TO_P8
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDZ: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(exp((z-zmin.val)/(zmax.val-zmin.val))*(1/(zmax.val-zmin.val)));
          case LAP: return (*mag)*((-1/(SQR(xmax.val-xmin.val)))+(-1/(SQR(ymax.val-ymin.val)))+1/SQR(zmax.val-zmin.val))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val))*exp((z-zmin.val)/(zmax.val-zmin.val)));
#else
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val));
          case LAP: return -(*mag)*((1/(SQR(xmax.val-xmin.val)))+(1/(SQR(ymax.val-ymin.val))))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
#endif
        }
#ifdef P4_TO_P8
      case 15: switch (what){
          case VAL: return (*mag)*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDX: return 0.0;
          case DDY: return 0.0;
          case DDZ: return (*mag)*(exp((z-zmin.val)/(zmax.val-zmin.val))*(1/(zmax.val-zmin.val)));
          case LAP: return (*mag)*(exp((z-zmin.val)/(zmax.val-zmin.val))*SQR(1/(zmax.val-zmin.val)));
        }
#endif
      case 16: switch (what){
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val));
#ifdef P4_TO_P8
          case DDZ: return 0.0;
#endif
          case LAP: return -(*mag)*((1/(SQR(xmax.val-xmin.val)))+(1/(SQR(ymax.val-ymin.val))))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
        }
#ifdef P4_TO_P8
      case 17: switch (what){
          case VAL: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val));
          case DDX: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*cos(x/(xmax.val-xmin.val))*(1/(xmax.val-xmin.val));
          case DDY: return ((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val))*(1/(ymax.val-ymin.val));
          case DDZ: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*sin(x/(xmax.val-xmin.val))*(1/(zmax.val-zmin.val));
          case LAP: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val))*(-SQR(1/(xmax.val-xmin.val)));
        }
#endif
#ifdef P4_TO_P8
      case 18: switch (what){
          case VAL: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*SQR((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))+pow((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val), 3.0);
          case DDX: return SQR((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*(1/(xmax.val-xmin.val));
          case DDY: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*(2*(y-0.5*(ymin.val+ymax.val))/SQR(ymax.val-ymin.val));
          case DDZ: return 3.*SQR(z-0.5*(zmin.val+zmax.val))/(pow((zmax.val-zmin.val), 3.0));
          case LAP: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*(2./SQR(ymax.val-ymin.val))+6.*(z-0.5*(zmin.val+zmax.val))/(pow((zmax.val-zmin.val), 3.0)) ;
        }
#endif
      case 19: {
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
      case 20: switch (what) {
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return 0;
        }
    case 21: switch (what) {
#ifdef P4_TO_P8
      case VAL: return  (*mag)*sin(PI*x/0.911)*sin(PI*y/0.911)*exp(PI*z/0.911);
      case DDX: return  (*mag)*(PI/0.911)*cos(PI*x/0.911)*sin(PI*y/0.911)*exp(PI*z/0.911);
      case DDY: return  (*mag)*(PI/0.911)*sin(PI*x/0.911)*cos(PI*y/0.911)*exp(PI*z/0.911);
      case DDZ: return  (*mag)*(PI/0.911)*sin(PI*x/0.911)*sin(PI*y/0.911)*exp(PI*z/0.911);
      case LAP: return -(*mag)*(PI/0.911)*(PI/0.911)*sin(PI*x/0.911)*sin(PI*y/0.911)*exp(PI*z/0.911);
#else
        case VAL: return  (*mag)*sin(PI*x/0.911)*sin(PI*y/0.911);
        case DDX: return  (*mag)*(PI/0.911)*cos(PI*x/0.911)*sin(PI*y/0.911);
        case DDY: return  (*mag)*(PI/0.911)*sin(PI*x/0.911)*cos(PI*y/0.911);
        case LAP: return -(*mag)*2.*(PI/0.911)*(PI/0.911)*sin(PI*x/0.911)*sin(PI*y/0.911);
#endif
      }
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};

u_pm_cf_t u_m_cf(VAL, n_um.val, mag_um.val),  u_p_cf(VAL, n_up.val, mag_up.val);
u_pm_cf_t ul_m_cf(LAP, n_um.val, mag_um.val), ul_p_cf(LAP, n_up.val, mag_up.val);
u_pm_cf_t DIM(ux_m_cf(DDX, n_um.val, mag_um.val),
              uy_m_cf(DDY, n_um.val, mag_um.val),
              uz_m_cf(DDZ, n_um.val, mag_um.val));
u_pm_cf_t DIM(ux_p_cf(DDX, n_up.val, mag_up.val),
              uy_p_cf(DDY, n_up.val, mag_up.val),
              uz_p_cf(DDZ, n_up.val, mag_up.val));

u_pm_cf_t u_m_cf1(VAL, n_um1.val, mag_um1.val),  u_p_cf1(VAL, n_up1.val, mag_up1.val);
u_pm_cf_t ul_m_cf1(LAP, n_um1.val, mag_um1.val), ul_p_cf1(LAP, n_up1.val, mag_up1.val);
u_pm_cf_t DIM(ux_m_cf1(DDX, n_um1.val, mag_um1.val),
              uy_m_cf1(DDY, n_um1.val, mag_um1.val),
              uz_m_cf1(DDZ, n_um1.val, mag_um1.val));
u_pm_cf_t DIM(ux_p_cf1(DDX, n_up1.val, mag_up1.val),
              uy_p_cf1(DDY, n_up1.val, mag_up1.val),
              uz_p_cf1(DDZ, n_up1.val, mag_up1.val));

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

diag_cf_t diag_m_cf(n_diag_m.val, mag_diag_m.val);
diag_cf_t diag_p_cf(n_diag_p.val, mag_diag_p.val);

diag_cf_t diag_m_cf1(n_diag_m1.val, mag_diag_m1.val);
diag_cf_t diag_p_cf1(n_diag_p1.val, mag_diag_p1.val);
// NONLINEAR TERMS
class nonlinear_term_cf_t: public CF_1
{
  int *n;
  cf_value_type_t what;
public:
  nonlinear_term_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
  double operator()(double u) const {
    switch (*n) {
      case 0:
        switch (what) {
          case VAL: return 0.;
          case DDX: return 0.;
          default: throw;
        }
      case 1:
        switch (what) {
          case VAL: return u;
          case DDX: return 1.;
          default: throw;
        }
      case 2:
        switch (what) {
          case VAL: return sinh(u);
          case DDX: return cosh(u);
          default: throw;
        }
      case 3:
        switch (what) {
          case VAL: return u/(1.+u);
          case DDX: return 1./SQR(1.+u);
          default: throw;
        }
      default:
        throw;
    }
  }
};

nonlinear_term_cf_t nonlinear_term_m_cf(VAL, nonlinear_term_m.val), nonlinear_term_m_prime_cf(DDX, nonlinear_term_m.val);
nonlinear_term_cf_t nonlinear_term_p_cf(VAL, nonlinear_term_p.val), nonlinear_term_p_prime_cf(DDX, nonlinear_term_p.val);

nonlinear_term_cf_t nonlinear_term_m_cf1(VAL, nonlinear_term_m1.val), nonlinear_term_m_prime_cf1(DDX, nonlinear_term_m1.val);
nonlinear_term_cf_t nonlinear_term_p_cf1(VAL, nonlinear_term_p1.val), nonlinear_term_p_prime_cf1(DDX, nonlinear_term_p1.val);

class nonlinear_term_coeff_cf_t: public CF_DIM
{
  int *n;
  double *mag;
public:
  nonlinear_term_coeff_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const {
    switch (*n) {
      case 0: return (*mag)*1.;
      case 1:
#ifdef P4_TO_P8
        return (*mag)*cos(x+z)*exp(y);
#else
        return (*mag)*cos(x)*exp(y);
#endif
    }
  }
};

nonlinear_term_coeff_cf_t nonlinear_term_m_coeff_cf(nonlinear_term_m_coeff.val, nonlinear_term_m_mag.val);
nonlinear_term_coeff_cf_t nonlinear_term_p_coeff_cf(nonlinear_term_p_coeff.val, nonlinear_term_p_mag.val);

nonlinear_term_coeff_cf_t nonlinear_term_m_coeff_cf1(nonlinear_term_m_coeff1.val, nonlinear_term_m_mag1.val);
nonlinear_term_coeff_cf_t nonlinear_term_p_coeff_cf1(nonlinear_term_p_coeff1.val, nonlinear_term_p_mag1.val);

// RHS
class rhs_m_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    switch (rhs_m_value.val)
    {
      case 0:
        return diag_m_cf(DIM(x,y,z))*u_m_cf(DIM(x,y,z))
            + nonlinear_term_m_coeff_cf(DIM(x,y,z))*nonlinear_term_m_cf(u_m_cf(DIM(x,y,z)))
            - mu_m_cf(DIM(x,y,z))*ul_m_cf(DIM(x,y,z))
            - SUMD(mux_m_cf(DIM(x,y,z))*ux_m_cf(DIM(x,y,z)),
                   muy_m_cf(DIM(x,y,z))*uy_m_cf(DIM(x,y,z)),
                   muz_m_cf(DIM(x,y,z))*uz_m_cf(DIM(x,y,z)));
      case 1:
      {
        static int num_particles = 68;
        static double X[] = { 0.294320, 0.292603, 0.296621, 0.301392, 0.293268, 0.289541, 0.290527, 0.295213, 0.300884, 0.935243, 0.936367, 0.937713, 0.927721, 0.926407, 0.939303, 0.926755, 0.936908, 0.934454, 0.160228, 0.164293, 0.173756, 0.168733, 0.170695, 0.160737, 0.725447, 0.740474, 0.736547, 0.723810, 0.742820, 0.728138, 0.728254, 0.052907, 0.034801, 0.048034, 0.038434, 0.037001, 0.048676, -0.182697, -0.180690, -0.171622, -0.177232, -0.182553, -0.169332, 0.082096, 0.070903, 0.086667, 0.089332, -0.416502, -0.435866, -0.428412, 0.367410, 0.364188, 0.357943, 0.354170, 0.355478, 0.356304, 0.368282, 0.370185, 0.352045, -0.187080, -0.177875, -0.177947, -0.193546, -0.190933, -0.184008, -0.192313, -0.190761 };
        static double Y[] = { -0.731980, -0.722570, -0.730048, -0.723932, -0.730192, -0.732616, -0.736414, -0.726113, -0.733236, 0.156877, 0.165333, 0.161907, 0.160662, 0.164860, 0.170991, 0.162948, 0.160965, 0.162162, -0.743358, -0.747939, -0.744021, -0.761421, -0.747981, -0.760843, 0.889409, 0.893693, 0.884784, 0.875899, 0.874970, 0.878376, 0.880038, -0.344586, -0.356601, -0.350902, -0.350375, -0.361473, -0.344655, 0.433937, 0.449481, 0.442325, 0.435290, 0.446933, 0.440621, 0.665721, 0.669204, 0.652859, 0.668903, -0.873554, -0.873537, -0.875426, -0.528637, -0.533808, -0.522965, -0.530541, -0.527060, -0.525395, -0.530008, -0.524812, -0.519527, 0.207027, 0.213721, 0.216082, 0.211842, 0.205428, 0.210547, 0.214133, 0.223399 };
        static double Z[] = { 0.659896, 0.661595, 0.663882, 0.657828, 0.649222, 0.652738, 0.653999, 0.659133, 0.653780, 0.372778, 0.384505, 0.385340, 0.380446, 0.368198, 0.382526, 0.371762, 0.380080, 0.384611, -0.914150, -0.903361, -0.909046, -0.906778, -0.895898, -0.911101, 0.374524, 0.375398, 0.361166, 0.362541, 0.358814, 0.365019, 0.371807, 0.791016, 0.796822, 0.802796, 0.799813, 0.796510, 0.794363, -0.891243, -0.892697, -0.877118, -0.878855, -0.889057, -0.888704, -0.798652, -0.802171, -0.795067, -0.810748, -0.205157, -0.209997, -0.210105, 0.781060, 0.793624, 0.788213, 0.798592, 0.787955, 0.789437, 0.780380, 0.798013, 0.791062, 0.333318, 0.343816, 0.343286, 0.346698, 0.340582, 0.334128, 0.335550, 0.348634 };
        static double R[] = { 0.000948, 0.000196, 0.000553, 0.000374, 0.000798, 0.000755, 0.000341, 0.000792, 0.000857, 0.000462, 0.000946, 0.000114, 0.000653, 0.000243, 0.000443, 0.000482, 0.000713, 0.000448, 0.000329, 0.000792, 0.000264, 0.000999, 0.000389, 0.000209, 0.000966, 0.000453, 0.000708, 0.000367, 0.000306, 0.000857, 0.000451, 0.000660, 0.000834, 0.000961, 0.000546, 0.000859, 0.000590, 0.000642, 0.000499, 0.000714, 0.000706, 0.000927, 0.000652, 0.000309, 0.000229, 0.000256, 0.000778, 0.000168, 0.000167, 0.000329, 0.000239, 0.000271, 0.000682, 0.000809, 0.000107, 0.000985, 0.000107, 0.000830, 0.000198, 0.000928, 0.000126, 0.000647, 0.000749, 0.000170, 0.000752, 0.000925, 0.000728 };
        static double n[] = { 5, 2, 3, 3, 3, 4, 3, 5, 4, 4, 3, 3, 4, 2, 3, 5, 3, 6, 6, 3, 4, 5, 4, 6, 3, 3, 4, 3, 4, 4, 6, 4, 4, 4, 5, 6, 3, 5, 4, 3, 6, 5, 6, 6, 3, 4, 4, 2, 6, 4, 6, 3, 3, 5, 4, 2, 5, 5, 4, 2, 3, 4, 6, 3, 3, 4, 4 };
        static double b[] = { 0.177428, 0.120466, 0.195375, 0.161709, 0.193548, 0.088156, 0.140005, 0.121836, 0.078646, 0.026599, 0.129853, 0.111576, 0.059185, 0.009826, 0.141560, 0.025398, 0.179751, 0.011111, 0.052804, 0.015578, 0.186352, 0.001466, 0.025428, 0.199829, 0.064583, 0.004282, 0.036598, 0.044844, 0.191154, 0.026387, 0.069948, 0.184134, 0.079140, 0.175580, 0.119999, 0.104759, 0.018488, 0.191264, 0.107538, 0.162207, 0.108419, 0.139933, 0.197255, 0.018026, 0.189876, 0.009732, 0.189929, 0.038622, 0.040327, 0.067637, 0.004636, 0.072883, 0.097692, 0.035313, 0.166186, 0.004988, 0.133467, 0.145549, 0.011507, 0.121431, 0.001767, 0.130215, 0.088286, 0.122096, 0.030592, 0.029170, 0.195173 };
        static double t[] = { 3.779031, 0.575194, 5.983689, 1.178485, 6.112317, 2.266495, 6.140219, 0.884964, 1.588758, 1.152778, 2.585702, 3.069164, 4.986538, 3.613796, 2.988958, 3.380732, 0.252419, 2.899990, 4.507911, 3.753404, 6.101247, 4.162976, 0.952773, 0.814897, 5.719127, 5.779404, 0.871440, 0.309913, 1.616747, 5.913726, 2.499319, 2.412480, 4.794653, 1.000997, 5.330543, 2.104655, 3.979605, 0.196232, 3.958288, 4.889946, 2.765306, 0.682710, 0.524252, 3.925875, 1.973639, 5.337312, 0.506358, 4.772787, 4.357852, 4.422504, 1.018504, 2.058938, 4.725618, 0.726723, 5.450653, 3.471888, 5.475823, 6.126808, 3.886958, 1.730282, 0.076936, 6.002824, 4.366932, 0.951230, 3.986605, 1.284015, 5.881409 };
        static double q[] = { 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
        static radial_shaped_domain_t shape;
        double min_dist = DBL_MAX;
        int idx = -1;
        for (int i = 0; i < num_particles; ++i)
        {
          shape.set_params(R[i], DIM(X[i], Y[i], Z[i]), 1, 1, &n[i], &b[i], &t[i]);
          double current = shape.phi(DIM(x,y,z));
          if ((current) < (min_dist))
          {
            idx = i;
            min_dist = current;
          }

//          PetscPrintf(MPI_COMM_WORLD, "%d & %f & %f & %f & %f & %d & %f & %+1.0e \\\\ \n", i+1, X[i], Y[i], R[i], b[i], (int)n[i], t[i], q[i]);
        }
//        throw;
        return 1.e5*q[idx];
      }
      default:
        throw;
    }
  }
} rhs_m_cf;

class rhs_p_cf_t: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const {
    switch (rhs_p_value.val)
    {
      case 0:
        return diag_p_cf(DIM(x,y,z))*u_p_cf(DIM(x,y,z))
            + nonlinear_term_p_coeff_cf(DIM(x,y,z))*nonlinear_term_p_cf(u_p_cf(DIM(x,y,z)))
            - mu_p_cf(DIM(x,y,z))*ul_p_cf(DIM(x,y,z))
            - SUMD(mux_p_cf(DIM(x,y,z))*ux_p_cf(DIM(x,y,z)),
                   muy_p_cf(DIM(x,y,z))*uy_p_cf(DIM(x,y,z)),
                   muz_p_cf(DIM(x,y,z))*uz_p_cf(DIM(x,y,z)));
      case 1:
        return 0;
      default:
        throw;
    }
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

bc_coeff_cf_t bc_coeff_cf_all[] = { bc_coeff_cf_t(bc_coeff_00.val, bc_coeff_00_mag.val),
                                    bc_coeff_cf_t(bc_coeff_01.val, bc_coeff_01_mag.val),
                                    bc_coeff_cf_t(bc_coeff_02.val, bc_coeff_02_mag.val),
                                    bc_coeff_cf_t(bc_coeff_03.val, bc_coeff_03_mag.val) };


bc_coeff_cf_t bc1_coeff_cf_all[] = {bc_coeff_cf_t(bc1_coeff_00.val, bc1_coeff_00_mag.val),
                                    bc_coeff_cf_t(bc1_coeff_01.val, bc1_coeff_01_mag.val),
                                    bc_coeff_cf_t(bc1_coeff_02.val, bc1_coeff_02_mag.val),
                                    bc_coeff_cf_t(bc1_coeff_03.val, bc1_coeff_03_mag.val) };
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
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 2: // circle/sphere exterior
      {
        static double r0 = 0.911, DIM(xc = 0, yc = 0, zc = 0);
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0, -1);
        switch (what) {
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 3: // annular/shell region
      {
        static double r0_in = 0.151, DIM(xc_in = 0, yc_in = 0, zc_in = 0);
        static double r0_ex = 0.911, DIM(xc_ex = 0, yc_ex = 0, zc_ex = 0);
        static flower_shaped_domain_t circle_in(r0_in, DIM(xc_in, yc_in, zc_in), 0, -1);
        static flower_shaped_domain_t circle_ex(r0_ex, DIM(xc_ex, yc_ex, zc_ex), 0,  1);
        switch (what) {
          _CODE( case VAL: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi  (DIM(x,y,z)) : circle_ex.phi  (DIM(x,y,z)); );
          XCODE( case DDX: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_x(DIM(x,y,z)) : circle_ex.phi_x(DIM(x,y,z)); );
          YCODE( case DDY: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_y(DIM(x,y,z)) : circle_ex.phi_y(DIM(x,y,z)); );
          ZCODE( case DDZ: return (circle_in.phi(DIM(x,y,z)) > circle_ex.phi(DIM(x,y,z))) ? circle_in.phi_z(DIM(x,y,z)) : circle_ex.phi_z(DIM(x,y,z)); );
        }
      } break;
      case 4: // moderately star-shaped domain
      {
        static double r0 = 0.611, DIM(xc = 0, yc = 0, zc = 0), deform = 0.15;
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform);
        switch (what) {
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      } break;
      case 5: // highly start-shaped domain
      {
        static double r0 = 0.611, DIM(xc = 0, yc = 0, zc = 0), deform = 0.3;
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform);
        switch (what) {
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 6: // union of two spheres: 1st sphere
      case 7: // union of two spheres: 2nd sphere
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
          _CODE( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
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
          _CODE( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
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
          _CODE( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
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
          _CODE( case VAL: return shape_ptr->phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape_ptr->phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape_ptr->phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape_ptr->phi_z(DIM(x,y,z)) );
        }
      } break;
    }

    // default value
    switch (what) {
      _CODE( case VAL: return -1 );
      XCODE( case DDX: return  0 );
      YCODE( case DDY: return  0 );
      ZCODE( case DDZ: return  0 );
    }
  }
};

bdry_phi_cf_t bdry_phi_cf_all  [] = { bdry_phi_cf_t(VAL, bdry_geom_00.val),
                                      bdry_phi_cf_t(VAL, bdry_geom_01.val),
                                      bdry_phi_cf_t(VAL, bdry_geom_02.val),
                                      bdry_phi_cf_t(VAL, bdry_geom_03.val) };

bdry_phi_cf_t bdry_phi_x_cf_all[] = { bdry_phi_cf_t(DDX, bdry_geom_00.val),
                                      bdry_phi_cf_t(DDX, bdry_geom_01.val),
                                      bdry_phi_cf_t(DDX, bdry_geom_02.val),
                                      bdry_phi_cf_t(DDX, bdry_geom_03.val) };

bdry_phi_cf_t bdry_phi_y_cf_all[] = { bdry_phi_cf_t(DDY, bdry_geom_00.val),
                                      bdry_phi_cf_t(DDY, bdry_geom_01.val),
                                      bdry_phi_cf_t(DDY, bdry_geom_02.val),
                                      bdry_phi_cf_t(DDY, bdry_geom_03.val) };
#ifdef P4_TO_P8
bdry_phi_cf_t bdry_phi_z_cf_all[] = { bdry_phi_cf_t(DDZ, bdry_geom_00.val),
                                      bdry_phi_cf_t(DDZ, bdry_geom_01.val),
                                      bdry_phi_cf_t(DDZ, bdry_geom_02.val),
                                      bdry_phi_cf_t(DDZ, bdry_geom_03.val) };
#endif

bdry_phi_cf_t bdry1_phi_cf_all  [] = { bdry_phi_cf_t(VAL, bdry1_geom_00.val),
                                      bdry_phi_cf_t(VAL, bdry1_geom_01.val),
                                      bdry_phi_cf_t(VAL, bdry1_geom_02.val),
                                      bdry_phi_cf_t(VAL, bdry1_geom_03.val) };

bdry_phi_cf_t bdry1_phi_x_cf_all[] = { bdry_phi_cf_t(DDX, bdry1_geom_00.val),
                                      bdry_phi_cf_t(DDX, bdry1_geom_01.val),
                                      bdry_phi_cf_t(DDX, bdry1_geom_02.val),
                                      bdry_phi_cf_t(DDX, bdry1_geom_03.val) };

bdry_phi_cf_t bdry1_phi_y_cf_all[] = { bdry_phi_cf_t(DDY, bdry1_geom_00.val),
                                      bdry_phi_cf_t(DDY, bdry1_geom_01.val),
                                      bdry_phi_cf_t(DDY, bdry1_geom_02.val),
                                      bdry_phi_cf_t(DDY, bdry1_geom_03.val) };
#ifdef P4_TO_P8
bdry_phi_cf_t bdry1_phi_z_cf_all[] = {bdry_phi_cf_t(DDZ, bdry1_geom_00.val),
                                      bdry_phi_cf_t(DDZ, bdry1_geom_01.val),
                                      bdry_phi_cf_t(DDZ, bdry1_geom_02.val),
                                      bdry_phi_cf_t(DDZ, bdry1_geom_03.val) };
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
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 2: // moderately star-shaped domain
      {
        static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0.15, -1);
        switch (what) {
          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
        }
      }
      case 3: // highly star-shaped domain
      {
//        static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
//        static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0.3, -1);
//        switch (what) {
//          _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
//          XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
//          YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
//          ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
//        }
        static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
        static double N = 1;
        static double n[] = { 5.0};
        static double b[] = { .3/r0 };
        static double t[] = { 0.0};
        static radial_shaped_domain_t shape(r0, DIM(xc, yc, zc), -1, N, n, b, t);
        switch (what) {
          _CODE( case VAL: return MIN(0.2, shape.phi  (DIM(x,y,z))) );
          XCODE( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape.phi_z(DIM(x,y,z)) );
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
          _CODE( case VAL: return shape.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape.phi_z(DIM(x,y,z)) );
        }
      }
      case 5: // clusters of stars
      {
        int num_particles = 68;
        static double X[] = { 0.294320, 0.292603, 0.296621, 0.301392, 0.293268, 0.289541, 0.290527, 0.295213, 0.300884, 0.935243, 0.936367, 0.937713, 0.927721, 0.926407, 0.939303, 0.926755, 0.936908, 0.934454, 0.160228, 0.164293, 0.173756, 0.168733, 0.170695, 0.160737, 0.725447, 0.740474, 0.736547, 0.723810, 0.742820, 0.728138, 0.728254, 0.052907, 0.034801, 0.048034, 0.038434, 0.037001, 0.048676, -0.182697, -0.180690, -0.171622, -0.177232, -0.182553, -0.169332, 0.082096, 0.070903, 0.086667, 0.089332, -0.416502, -0.435866, -0.428412, 0.367410, 0.364188, 0.357943, 0.354170, 0.355478, 0.356304, 0.368282, 0.370185, 0.352045, -0.187080, -0.177875, -0.177947, -0.193546, -0.190933, -0.184008, -0.192313, -0.190761 };
        static double Y[] = { -0.731980, -0.722570, -0.730048, -0.723932, -0.730192, -0.732616, -0.736414, -0.726113, -0.733236, 0.156877, 0.165333, 0.161907, 0.160662, 0.164860, 0.170991, 0.162948, 0.160965, 0.162162, -0.743358, -0.747939, -0.744021, -0.761421, -0.747981, -0.760843, 0.889409, 0.893693, 0.884784, 0.875899, 0.874970, 0.878376, 0.880038, -0.344586, -0.356601, -0.350902, -0.350375, -0.361473, -0.344655, 0.433937, 0.449481, 0.442325, 0.435290, 0.446933, 0.440621, 0.665721, 0.669204, 0.652859, 0.668903, -0.873554, -0.873537, -0.875426, -0.528637, -0.533808, -0.522965, -0.530541, -0.527060, -0.525395, -0.530008, -0.524812, -0.519527, 0.207027, 0.213721, 0.216082, 0.211842, 0.205428, 0.210547, 0.214133, 0.223399 };
        static double Z[] = { 0.659896, 0.661595, 0.663882, 0.657828, 0.649222, 0.652738, 0.653999, 0.659133, 0.653780, 0.372778, 0.384505, 0.385340, 0.380446, 0.368198, 0.382526, 0.371762, 0.380080, 0.384611, -0.914150, -0.903361, -0.909046, -0.906778, -0.895898, -0.911101, 0.374524, 0.375398, 0.361166, 0.362541, 0.358814, 0.365019, 0.371807, 0.791016, 0.796822, 0.802796, 0.799813, 0.796510, 0.794363, -0.891243, -0.892697, -0.877118, -0.878855, -0.889057, -0.888704, -0.798652, -0.802171, -0.795067, -0.810748, -0.205157, -0.209997, -0.210105, 0.781060, 0.793624, 0.788213, 0.798592, 0.787955, 0.789437, 0.780380, 0.798013, 0.791062, 0.333318, 0.343816, 0.343286, 0.346698, 0.340582, 0.334128, 0.335550, 0.348634 };
        static double R[] = { 0.000948, 0.000196, 0.000553, 0.000374, 0.000798, 0.000755, 0.000341, 0.000792, 0.000857, 0.000462, 0.000946, 0.000114, 0.000653, 0.000243, 0.000443, 0.000482, 0.000713, 0.000448, 0.000329, 0.000792, 0.000264, 0.000999, 0.000389, 0.000209, 0.000966, 0.000453, 0.000708, 0.000367, 0.000306, 0.000857, 0.000451, 0.000660, 0.000834, 0.000961, 0.000546, 0.000859, 0.000590, 0.000642, 0.000499, 0.000714, 0.000706, 0.000927, 0.000652, 0.000309, 0.000229, 0.000256, 0.000778, 0.000168, 0.000167, 0.000329, 0.000239, 0.000271, 0.000682, 0.000809, 0.000107, 0.000985, 0.000107, 0.000830, 0.000198, 0.000928, 0.000126, 0.000647, 0.000749, 0.000170, 0.000752, 0.000925, 0.000728 };
        static double n[] = { 5, 2, 3, 3, 3, 4, 3, 5, 4, 4, 3, 3, 4, 2, 3, 5, 3, 6, 6, 3, 4, 5, 4, 6, 3, 3, 4, 3, 4, 4, 6, 4, 4, 4, 5, 6, 3, 5, 4, 3, 6, 5, 6, 6, 3, 4, 4, 2, 6, 4, 6, 3, 3, 5, 4, 2, 5, 5, 4, 2, 3, 4, 6, 3, 3, 4, 4 };
        static double b[] = { 0.177428, 0.120466, 0.195375, 0.161709, 0.193548, 0.088156, 0.140005, 0.121836, 0.078646, 0.026599, 0.129853, 0.111576, 0.059185, 0.009826, 0.141560, 0.025398, 0.179751, 0.011111, 0.052804, 0.015578, 0.186352, 0.001466, 0.025428, 0.199829, 0.064583, 0.004282, 0.036598, 0.044844, 0.191154, 0.026387, 0.069948, 0.184134, 0.079140, 0.175580, 0.119999, 0.104759, 0.018488, 0.191264, 0.107538, 0.162207, 0.108419, 0.139933, 0.197255, 0.018026, 0.189876, 0.009732, 0.189929, 0.038622, 0.040327, 0.067637, 0.004636, 0.072883, 0.097692, 0.035313, 0.166186, 0.004988, 0.133467, 0.145549, 0.011507, 0.121431, 0.001767, 0.130215, 0.088286, 0.122096, 0.030592, 0.029170, 0.195173 };
        static double t[] = { 3.779031, 0.575194, 5.983689, 1.178485, 6.112317, 2.266495, 6.140219, 0.884964, 1.588758, 1.152778, 2.585702, 3.069164, 4.986538, 3.613796, 2.988958, 3.380732, 0.252419, 2.899990, 4.507911, 3.753404, 6.101247, 4.162976, 0.952773, 0.814897, 5.719127, 5.779404, 0.871440, 0.309913, 1.616747, 5.913726, 2.499319, 2.412480, 4.794653, 1.000997, 5.330543, 2.104655, 3.979605, 0.196232, 3.958288, 4.889946, 2.765306, 0.682710, 0.524252, 3.925875, 1.973639, 5.337312, 0.506358, 4.772787, 4.357852, 4.422504, 1.018504, 2.058938, 4.725618, 0.726723, 5.450653, 3.471888, 5.475823, 6.126808, 3.886958, 1.730282, 0.076936, 6.002824, 4.366932, 0.951230, 3.986605, 1.284015, 5.881409 };
        static radial_shaped_domain_t shape;
        double min_dist = DBL_MAX;
        int idx = -1;
        for (int i = 0; i < num_particles; ++i)
        {
          shape.set_params(R[i], DIM(X[i], Y[i], Z[i]), 1, 1, &n[i], &b[i], &t[i]);
          double current = shape.phi(DIM(x,y,z));
          if ((current) < (min_dist))
          {
            idx = i;
            min_dist = current;
          }
        }

        shape.set_params(R[idx], DIM(X[idx], Y[idx], Z[idx]), 1, 1, &n[idx], &b[idx], &t[idx]);
        switch (what) {
          _CODE( case VAL: return shape.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape.phi_z(DIM(x,y,z)) );
        }
      }
      case 6: // highly wavy circle
      {
        static double r0 = 0.483, DIM( xc = 0, yc = 0, zc = 0 );
        static double N = 3;
        static double n[] = { 6., 100.0, 10000.0 };
        static double b[] = { 0.2, .015, .00015, };
        static double t[] = { 0.0, 0.0, 0.0 };
        static radial_shaped_domain_t shape(r0, DIM(xc, yc, zc), 1, N, n, b, t);
        switch (what) {
          _CODE( case VAL: return shape.phi  (DIM(x,y,z)) );
          XCODE( case DDX: return shape.phi_x(DIM(x,y,z)) );
          YCODE( case DDY: return shape.phi_y(DIM(x,y,z)) );
          ZCODE( case DDZ: return shape.phi_z(DIM(x,y,z)) );
        }
      }
    }
  }
};


infc_phi_cf_t infc_phi_cf_all  [] = { infc_phi_cf_t(VAL, infc_geom_00.val),
                                      infc_phi_cf_t(VAL, infc_geom_01.val),
                                      infc_phi_cf_t(VAL, infc_geom_02.val),
                                      infc_phi_cf_t(VAL, infc_geom_03.val) };

infc_phi_cf_t infc_phi_x_cf_all[] = { infc_phi_cf_t(DDX, infc_geom_00.val),
                                      infc_phi_cf_t(DDX, infc_geom_01.val),
                                      infc_phi_cf_t(DDX, infc_geom_02.val),
                                      infc_phi_cf_t(DDX, infc_geom_03.val) };

infc_phi_cf_t infc_phi_y_cf_all[] = { infc_phi_cf_t(DDY, infc_geom_00.val),
                                      infc_phi_cf_t(DDY, infc_geom_01.val),
                                      infc_phi_cf_t(DDY, infc_geom_02.val),
                                      infc_phi_cf_t(DDY, infc_geom_03.val) };
#ifdef P4_TO_P8
infc_phi_cf_t infc_phi_z_cf_all[] = { infc_phi_cf_t(DDZ, infc_geom_00.val),
                                      infc_phi_cf_t(DDZ, infc_geom_01.val),
                                      infc_phi_cf_t(DDZ, infc_geom_02.val),
                                      infc_phi_cf_t(DDZ, infc_geom_03.val) };
#endif

infc_phi_cf_t infc1_phi_cf_all  [] = {infc_phi_cf_t(VAL, infc1_geom_00.val),
                                      infc_phi_cf_t(VAL, infc1_geom_01.val),
                                      infc_phi_cf_t(VAL, infc1_geom_02.val),
                                      infc_phi_cf_t(VAL, infc1_geom_03.val) };

infc_phi_cf_t infc1_phi_x_cf_all[] = {infc_phi_cf_t(DDX, infc1_geom_00.val),
                                      infc_phi_cf_t(DDX, infc1_geom_01.val),
                                      infc_phi_cf_t(DDX, infc1_geom_02.val),
                                      infc_phi_cf_t(DDX, infc1_geom_03.val) };

infc_phi_cf_t infc1_phi_y_cf_all[] = {infc_phi_cf_t(DDY, infc1_geom_00.val),
                                      infc_phi_cf_t(DDY, infc1_geom_01.val),
                                      infc_phi_cf_t(DDY, infc1_geom_02.val),
                                      infc_phi_cf_t(DDY, infc1_geom_03.val) };
#ifdef P4_TO_P8
infc_phi_cf_t infc1_phi_z_cf_all[] = {infc_phi_cf_t(DDZ, infc1_geom_00.val),
                                      infc_phi_cf_t(DDZ, infc1_geom_01.val),
                                      infc_phi_cf_t(DDZ, infc1_geom_02.val),
                                      infc_phi_cf_t(DDZ, infc1_geom_03.val) };
#endif
// the effective LSF (initialized in main!)
mls_eff_cf_t bdry_phi_eff_cf;
mls_eff_cf_t infc_phi_eff_cf;

mls_eff_cf_t bdry1_phi_eff_cf;
mls_eff_cf_t infc1_phi_eff_cf;

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
};
phi_eff_cf_t phi_eff_cf(bdry_phi_eff_cf,infc_phi_eff_cf);
phi_eff_cf_t phi_eff_cf1(bdry1_phi_eff_cf,infc1_phi_eff_cf);
class mu_cf_t : public CF_DIM
{
  CF_DIM *infc_phi_cf_;
  CF_DIM *mu_m_cf_;
  CF_DIM *mu_p_cf_;
public:
  mu_cf_t(CF_DIM &infc_phi_cf, CF_DIM &mu_m_cf, CF_DIM &mu_p_cf) : infc_phi_cf_(&infc_phi_cf), mu_m_cf_(&mu_m_cf), mu_p_cf_(&mu_p_cf) {}
  double operator()(DIM(double x, double y, double z)) const {
    return (*infc_phi_cf_)(DIM(x,y,z)) >= 0 ? (*mu_p_cf_)(DIM(x,y,z)) : (*mu_m_cf_)(DIM(x,y,z));
  }
};
mu_cf_t mu_cf(infc_phi_eff_cf, mu_p_cf, mu_m_cf);
mu_cf_t mu_cf1(infc_phi_eff_cf, mu_p_cf1, mu_m_cf1);


class u_cf_t : public CF_DIM
{
  CF_DIM *infc_phi_cf_;
  CF_DIM *u_m_cf_;
  CF_DIM *u_p_cf_;
public:
    u_cf_t(CF_DIM &infc_phi_cf, CF_DIM &u_m_cf, CF_DIM &u_p_cf) : infc_phi_cf_(&infc_phi_cf), u_m_cf_(&u_m_cf), u_p_cf_(&u_p_cf) {}
    double operator()(DIM(double x, double y, double z)) const {
    return (*infc_phi_cf_)(DIM(x,y,z)) >= 0 ? (*u_p_cf_)(DIM(x,y,z)) : (*u_m_cf_)(DIM(x,y,z));
  }
};
u_cf_t u_cf(infc_phi_eff_cf, u_p_cf, u_m_cf);
u_cf_t u_cf1(infc_phi_eff_cf, u_p_cf1, u_m_cf1);


class ux_cf_t : public CF_DIM
{
  CF_DIM *infc_phi_cf_;
  CF_DIM *ux_m_cf_;
  CF_DIM *ux_p_cf_;
public:
    ux_cf_t(CF_DIM &infc_phi_cf, CF_DIM &ux_m_cf, CF_DIM &ux_p_cf) : infc_phi_cf_(&infc_phi_cf), ux_m_cf_(&ux_m_cf), ux_p_cf_(&ux_p_cf) {}
    double operator()(DIM(double x, double y, double z)) const {
    return (*infc_phi_cf_)(DIM(x,y,z)) >= 0 ? (*ux_p_cf_)(DIM(x,y,z)) : (*ux_m_cf_)(DIM(x,y,z));
  }
};
ux_cf_t ux_cf(infc_phi_eff_cf, ux_p_cf, ux_m_cf);
ux_cf_t ux_cf1(infc_phi_eff_cf, ux_p_cf1, ux_m_cf1);

class uy_cf_t : public CF_DIM
{
  CF_DIM *infc_phi_cf_;
  CF_DIM *uy_m_cf_;
  CF_DIM *uy_p_cf_;
public:
    uy_cf_t(CF_DIM &infc_phi_cf, CF_DIM &uy_m_cf, CF_DIM &uy_p_cf) : infc_phi_cf_(&infc_phi_cf), uy_m_cf_(&uy_m_cf), uy_p_cf_(&uy_p_cf) {}
    double operator()(DIM(double x, double y, double z)) const {
    return (*infc_phi_cf_)(DIM(x,y,z)) >= 0 ? (*uy_p_cf_)(DIM(x,y,z)) : (*uy_m_cf_)(DIM(x,y,z));
  }
};
uy_cf_t uy_cf(infc_phi_eff_cf, uy_p_cf, uy_m_cf);
uy_cf_t uy_cf1(infc_phi_eff_cf, uy_p_cf1, uy_m_cf1);


#ifdef P4_TO_P8
class uz_cf_t : public CF_DIM
{
  CF_DIM *infc_phi_cf_;
  CF_DIM *uz_m_cf_;
  CF_DIM *uz_p_cf_;
public:
    uz_cf_t(CF_DIM &infc_phi_cf, CF_DIM &uz_m_cf, CF_DIM &uz_p_cf) : infc_phi_cf_(&infc_phi_cf), uz_m_cf_(&uz_m_cf), uz_p_cf_(&uz_p_cf) {}
    double operator()(DIM(double x, double y, double z)) const {
    return (*infc_phi_cf_)(DIM(x,y,z)) >= 0 ? (*uz_p_cf_)(DIM(x,y,z)) : (*uz_m_cf_)(DIM(x,y,z));
  }
};
uz_cf_t uz_cf(infc_phi_eff_cf, uz_p_cf, uz_m_cf);
uz_cf_t uz_cf1(infc_phi_eff_cf, uz_p_cf1, uz_m_cf1);
#endif

// BC VALUES
class bc_value_robin_t : public CF_DIM
{
  BoundaryConditionType *bc_type_;
  CF_DIM DIM(*phi_x_cf_,
             *phi_y_cf_,
             *phi_z_cf_);
  CF_DIM *bc_coeff_cf_;
  CF_DIM *u_cf_;
  CF_DIM *mu_cf_;
  CF_DIM DIM(*ux_cf_,
             *uy_cf_,
             *uz_cf_);
public:
  bc_value_robin_t(BoundaryConditionType *bc_type,
                   CF_DIM *bc_coeff_cf,
                   CF_DIM *u_cf,
                   CF_DIM *mu_cf,
                   DIM(CF_DIM *phi_x_cf,
                       CF_DIM *phi_y_cf,
                       CF_DIM *phi_z_cf),
                   DIM(CF_DIM *ux_cf,
                       CF_DIM *uy_cf,
                       CF_DIM *uz_cf))
    : bc_type_(bc_type),
      bc_coeff_cf_(bc_coeff_cf),
      u_cf_(u_cf),
      mu_cf_(mu_cf),
      DIM(phi_x_cf_(phi_x_cf),
          phi_y_cf_(phi_y_cf),
          phi_z_cf_(phi_z_cf)),
      DIM(ux_cf_(ux_cf),
          uy_cf_(uy_cf),
          uz_cf_(uz_cf)){}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*bc_type_) {
      case DIRICHLET: return (*u_cf_)(DIM(x,y,z))*0.0;
      case NEUMANN: {
          double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                      ny = (*phi_y_cf_)(DIM(x,y,z)),
                      nz = (*phi_z_cf_)(DIM(x,y,z)) );

          double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
          nx /= norm; ny /= norm; P8( nz /= norm );

          return (*bc_coeff_cf_)(DIM(x,y,z))*(*mu_cf_)(DIM(x,y,z))*SUMD(nx*(*ux_cf_)(DIM(x,y,z)),
                                                                        ny*(*uy_cf_)(DIM(x,y,z)),
                                                                        nz*(*uz_cf_)(DIM(x,y,z)));
        }
      case ROBIN: {
          double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                      ny = (*phi_y_cf_)(DIM(x,y,z)),
                      nz = (*phi_z_cf_)(DIM(x,y,z)) );

          double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
          nx /= norm; ny /= norm; CODE3D( nz /= norm );

          return (*mu_cf_)(DIM(x,y,z))*SUMD(nx*(*ux_cf_)(DIM(x,y,z)),
                                        ny*(*uy_cf_)(DIM(x,y,z)),
                                        nz*(*uz_cf_)(DIM(x,y,z))) + (*bc_coeff_cf_)(DIM(x,y,z))*(*u_cf_)(DIM(x,y,z));
        }
    }
  }
};

bc_value_robin_t bc_value_cf_all[] = { bc_value_robin_t((BoundaryConditionType *)&bc_type_00.val, &bc_coeff_cf_all[0], &u_cf, &mu_cf, DIM(&bdry_phi_x_cf_all[0], &bdry_phi_y_cf_all[0], &bdry_phi_z_cf_all[0]),DIM(&ux_cf, &uy_cf, &uz_cf)),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_01.val, &bc_coeff_cf_all[1], &u_cf, &mu_cf, DIM(&bdry_phi_x_cf_all[1], &bdry_phi_y_cf_all[1], &bdry_phi_z_cf_all[1]),DIM(&ux_cf, &uy_cf, &uz_cf)),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_02.val, &bc_coeff_cf_all[2], &u_cf, &mu_cf, DIM(&bdry_phi_x_cf_all[2], &bdry_phi_y_cf_all[2], &bdry_phi_z_cf_all[2]),DIM(&ux_cf, &uy_cf, &uz_cf)),
                                       bc_value_robin_t((BoundaryConditionType *)&bc_type_03.val, &bc_coeff_cf_all[3], &u_cf, &mu_cf, DIM(&bdry_phi_x_cf_all[3], &bdry_phi_y_cf_all[3], &bdry_phi_z_cf_all[3]),DIM(&ux_cf, &uy_cf, &uz_cf))};

bc_value_robin_t bc1_value_cf_all[] = { bc_value_robin_t((BoundaryConditionType *)&bc1_type_00.val, &bc1_coeff_cf_all[0], &u_cf1, &mu_cf1, DIM(&bdry1_phi_x_cf_all[0], &bdry1_phi_y_cf_all[0], &bdry1_phi_z_cf_all[0]),DIM(&ux_cf1, &uy_cf1, &uz_cf1)),
                                        bc_value_robin_t((BoundaryConditionType *)&bc1_type_01.val, &bc1_coeff_cf_all[1], &u_cf1, &mu_cf1, DIM(&bdry1_phi_x_cf_all[1], &bdry1_phi_y_cf_all[1], &bdry1_phi_z_cf_all[1]),DIM(&ux_cf1, &uy_cf1, &uz_cf1)),
                                        bc_value_robin_t((BoundaryConditionType *)&bc1_type_02.val, &bc1_coeff_cf_all[2], &u_cf1, &mu_cf1, DIM(&bdry1_phi_x_cf_all[2], &bdry1_phi_y_cf_all[2], &bdry1_phi_z_cf_all[2]),DIM(&ux_cf1, &uy_cf1, &uz_cf1)),
                                        bc_value_robin_t((BoundaryConditionType *)&bc1_type_03.val, &bc1_coeff_cf_all[3], &u_cf1, &mu_cf1, DIM(&bdry1_phi_x_cf_all[3], &bdry1_phi_y_cf_all[3], &bdry1_phi_z_cf_all[3]),DIM(&ux_cf1, &uy_cf1, &uz_cf1))};

// JUMP CONDITIONS
class jc_value_cf_t : public CF_DIM
{
  int    *n;
  double *mag;
  CF_DIM *u_m_cf_;
  CF_DIM *u_p_cf_;
public:
  jc_value_cf_t(int &n, CF_DIM *u_m_cf, CF_DIM *u_p_cf) : n(&n),
                                                          u_m_cf_(u_m_cf),
                                                          u_p_cf_(u_p_cf) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(*n) {
      case 0: return (*u_p_cf_)(DIM(x,y,z)) - (*u_p_cf_)(DIM(x,y,z));
      case 1: return 0;
    }
  }
};

jc_value_cf_t jc_value_cf_all[] = { jc_value_cf_t(jc_value_00.val,&u_m_cf,&u_p_cf),
                                    jc_value_cf_t(jc_value_01.val,&u_m_cf,&u_p_cf),
                                    jc_value_cf_t(jc_value_02.val,&u_m_cf,&u_p_cf),
                                    jc_value_cf_t(jc_value_03.val,&u_m_cf,&u_p_cf) };
jc_value_cf_t jc1_value_cf_all[] = { jc_value_cf_t(jc1_value_00.val,&u_m_cf1,&u_p_cf1),
                                     jc_value_cf_t(jc1_value_01.val,&u_m_cf1,&u_p_cf1),
                                     jc_value_cf_t(jc1_value_02.val,&u_m_cf1,&u_p_cf1),
                                     jc_value_cf_t(jc1_value_03.val,&u_m_cf1,&u_p_cf1) };

jc_value_cf_t jc_value_cf(jc_value_00.val,&u_m_cf,&u_p_cf);
jc_value_cf_t jc_value_cf1(jc1_value_00.val,&u_m_cf1,&u_p_cf1);
class jc_flux_t : public CF_DIM
{
  int    *n;
  CF_DIM *mu_p_cf_;
  CF_DIM *mu_m_cf_;
  CF_DIM *phi_x_;
  CF_DIM *phi_y_;
  CF_DIM *phi_z_;
  CF_DIM *ux_p_cf_;
  CF_DIM *ux_m_cf_;
  CF_DIM *uy_p_cf_;
  CF_DIM *uy_m_cf_;
  CF_DIM *uz_p_cf_;
  CF_DIM *uz_m_cf_;

public:
  jc_flux_t(int &n, CF_DIM *mu_p_cf, CF_DIM *mu_m_cf, DIM(CF_DIM *phi_x, CF_DIM *phi_y, CF_DIM *phi_z), DIM(CF_DIM *ux_p_cf, CF_DIM *uy_p_cf, CF_DIM *uz_p_cf), DIM(CF_DIM *ux_m_cf, CF_DIM *uy_m_cf, CF_DIM *uz_m_cf) ) :
    n(&n), mu_p_cf_(mu_p_cf), mu_m_cf_(mu_m_cf), DIM(phi_x_(phi_x), phi_y_(phi_y), phi_z_(phi_z)), DIM(ux_p_cf_(ux_p_cf),uy_p_cf_(uy_p_cf), uz_p_cf_(uz_p_cf)),  DIM(ux_m_cf_(ux_m_cf),uy_m_cf_(uy_m_cf), uz_m_cf_(uz_m_cf)){}

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

            return (*mu_p_cf_)(DIM(x,y,z)) * SUMD((*ux_p_cf_)(DIM(x,y,z))*nx, (*uy_p_cf_)(DIM(x,y,z))*ny, (*uz_p_cf_)(DIM(x,y,z))*nz)
            -  (*mu_m_cf_)(DIM(x,y,z)) * SUMD((*ux_m_cf_)(DIM(x,y,z))*nx, (*uy_m_cf_)(DIM(x,y,z))*ny, (*uz_m_cf_)(DIM(x,y,z))*nz);
      }
      case 1: return 0;
    }
  }
};

jc_flux_t jc_flux_cf_all[] = { jc_flux_t(jc_flux_00.val, &mu_m_cf, &mu_p_cf, DIM(&infc_phi_x_cf_all[0], &infc_phi_y_cf_all[0], &infc_phi_z_cf_all[0]), DIM(&ux_p_cf, &uy_p_cf, &uz_p_cf), DIM(&ux_m_cf, &uy_m_cf, &uz_m_cf)),
                               jc_flux_t(jc_flux_01.val, &mu_m_cf, &mu_p_cf, DIM(&infc_phi_x_cf_all[1], &infc_phi_y_cf_all[1], &infc_phi_z_cf_all[1]), DIM(&ux_p_cf, &uy_p_cf, &uz_p_cf), DIM(&ux_m_cf, &uy_m_cf, &uz_m_cf)),
                               jc_flux_t(jc_flux_02.val, &mu_m_cf, &mu_p_cf, DIM(&infc_phi_x_cf_all[2], &infc_phi_y_cf_all[2], &infc_phi_z_cf_all[2]), DIM(&ux_p_cf, &uy_p_cf, &uz_p_cf), DIM(&ux_m_cf, &uy_m_cf, &uz_m_cf)),
                               jc_flux_t(jc_flux_03.val, &mu_m_cf, &mu_p_cf, DIM(&infc_phi_x_cf_all[3], &infc_phi_y_cf_all[3], &infc_phi_z_cf_all[3]), DIM(&ux_p_cf, &uy_p_cf, &uz_p_cf), DIM(&ux_m_cf, &uy_m_cf, &uz_m_cf)) };
jc_flux_t jc1_flux_cf_all[] = { jc_flux_t(jc1_flux_00.val, &mu_m_cf1, &mu_p_cf1, DIM(&infc1_phi_x_cf_all[0], &infc1_phi_y_cf_all[0], &infc1_phi_z_cf_all[0]), DIM(&ux_p_cf1, &uy_p_cf1, &uz_p_cf1), DIM(&ux_m_cf1, &uy_m_cf1, &uz_m_cf1)),
                               jc_flux_t(jc1_flux_01.val, &mu_m_cf1, &mu_p_cf1, DIM(&infc1_phi_x_cf_all[1], &infc1_phi_y_cf_all[1], &infc1_phi_z_cf_all[1]), DIM(&ux_p_cf1, &uy_p_cf1, &uz_p_cf1), DIM(&ux_m_cf1, &uy_m_cf1, &uz_m_cf1)),
                               jc_flux_t(jc1_flux_02.val, &mu_m_cf1, &mu_p_cf1, DIM(&infc1_phi_x_cf_all[2], &infc1_phi_y_cf_all[2], &infc1_phi_z_cf_all[2]), DIM(&ux_p_cf1, &uy_p_cf1, &uz_p_cf1), DIM(&ux_m_cf1, &uy_m_cf1, &uz_m_cf1)),
                               jc_flux_t(jc1_flux_03.val, &mu_m_cf1, &mu_p_cf1, DIM(&infc1_phi_x_cf_all[3], &infc1_phi_y_cf_all[3], &infc1_phi_z_cf_all[3]), DIM(&ux_p_cf1, &uy_p_cf1, &uz_p_cf1), DIM(&ux_m_cf1, &uy_m_cf1, &uz_m_cf1)) };


class bc_wall_type_t : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return (BoundaryConditionType) bc_wtype.val;
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
} dom_perturb_cf(dom_perturb.val), ifc_perturb_cf(ifc_perturb.val);
void setup_u_exact(vec_and_ptr_t u_ex, double time, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd){
  u_ex.get_array();

  foreach_node(n,nodes){
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);
    u_ex.ptr[n] = 5*tn*xyz[0]*(xmax.val- xyz[0])*xyz[1]*(xyz[1]-ymax.val);
  }// end of loop over nodes

  u_ex.restore_array();
}

void setup_rhs(vec_and_ptr_t u, vec_and_ptr_t u1,vec_and_ptr_t rhs_u, vec_and_ptr_t rhs_u1, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd, double time){

  // In building RHS, we have two options: (1) Backward Euler 1st order approx, (2) Crank Nicholson 2nd order approx
  // (1) du/dt = (u(n+1) - u(n)/dt) --> which is a backward euler 1st order approximation (since the RHS is discretized spatially at T(n+1))
  // (2) du/dt = alpha*laplace(u) ~ (u(n+1) - u(n)/dt) = (1/2)*(laplace(u(n)) + laplace(u(n+1)) )  ,
  //                              in which case we need the second derivatives of the temperature field at time n
  // Where alpha is the thermal diffusivity of the phase



    // Get derivatives of temperature fields if we are using Crank Nicholson:
    vec_and_ptr_dim_t u_dd;
    vec_and_ptr_dim_t u1_dd;
    if(method_.val ==2){
        u_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(u.vec,u_dd.vec);
        u_dd.get_array();

        u1_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(u1.vec,u1_dd.vec);
        u1_dd.get_array();
      }

  // Get u arrays:
  u.get_array();
  rhs_u.get_array();

  // Get u1 arrays:
  u1.get_array();
  rhs_u1.get_array();


  foreach_node(n,nodes){
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);
    // First, assemble system for Ts depending on case:
    if(method_.val == 2){ // Crank Nicholson
        rhs_u.ptr[n] = 2.*u.ptr[n]/dt.val + mu_cf.value(xyz)*(u_dd.ptr[0][n] + u_dd.ptr[1][n] CODE3D(+ u_dd.ptr[2][n]));
        rhs_u1.ptr[n] = 2.*u1.ptr[n]/dt.val + mu_cf1.value(xyz)*(u1_dd.ptr[0][n] + u1_dd.ptr[1][n] CODE3D(+ u1_dd.ptr[2][n]));
      }
    else{ // Backward Euler
        rhs_u.ptr[n] = u.ptr[n]/dt.val + 5*xyz[0]*(xmax.val- xyz[0])*xyz[1]*(xyz[1]-ymax.val) + 10*0.1*tn*(-xyz[0]*(xmax.val- xyz[0])+xyz[1]*(xyz[1]-ymax.val));
        rhs_u1.ptr[n] = u1.ptr[n]/dt.val + 5*xyz[0]*(xmax.val- xyz[0])*xyz[1]*(xyz[1]-ymax.val) + 10*0.1*tn*(-xyz[0]*(xmax.val- xyz[0])+xyz[1]*(xyz[1]-ymax.val));
      }
  }// end of loop over nodes

  // Restore u arrays:
  u.restore_array();
  rhs_u.restore_array();

  // Restore u1 arrays:
  rhs_u1.restore_array();
  u1.restore_array();

  if(method_.val ==2){
      u_dd.restore_array();
      u_dd.destroy();

      u1_dd.restore_array();
      u1_dd.destroy();
    }
}

void save_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t phi1, vec_and_ptr_t u,vec_and_ptr_t u1, char* filename ){
  // Things we want to save:
  /*
   * LSF
   * LSF2 for ex 2
   * Tl
   * Ts
   * v_interface
   * */

    // First, need to scale the fields appropriately:


    // Get arrays:
    phi.get_array();
    phi1.get_array();
    u.get_array(); u1.get_array();


    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;


    point_names = {"phi","phi1","u","u1"};
    point_data = {phi.ptr,phi1.ptr,u.ptr,u1.ptr};


    std::vector<std::string> cell_names;
    std::vector<double*> cell_data;

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    // Restore arrays:

    phi.restore_array();
    u.restore_array(); u1.restore_array();
    phi1.restore_array();

}

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
      (save_vtk.val ||
       save_domain.val ||
       save_convergence.val ||
       save_matrix_ascii.val ||
       save_matrix_binary.val))
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save results\n");
    return -1;
  }

  if (save_vtk.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  if (save_domain.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/geometry";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/geometry directory");
  }

  if (save_convergence.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/convergence";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/convergence directory");
  }

  if (save_matrix_ascii.val || save_matrix_binary.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/matrix";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/matrix directory");
  }

  // parse command line arguments
  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);

  n_example.set_from_cmd(cmd);
  set_example(n_example.val);

  pl.set_from_cmd_all(cmd);

  if (mpi.rank() == 0) pl.print_all();
  if (mpi.rank() == 0 && save_params.val) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }
  // bdry_phi_max_num -> defines the maximum number of boundaries that exist in this example. infc_phi_max_num -> defines the maximum number of interfaces that exist in this example.
  // The way it is implemented for now, there can be atmost 4 interfaces and 4 boundaries.
  // initialize effective level-sets
  // To add boundaries or interfaces - objects of mls_eff_cf_t -> bdry_phi_eff_cf and infc_phi_eff_cf are created. mls_eff_cf_t is a class that inherits CF_DIM ( CF_2 or CF_3 depending on whether the problem is 2D or 3D )
  // The above mentioned class has an add_domain() which has two arguments namely a level set function representing the boundary (that is stored in bdry_phi_cf_all[i]) and the action (which can either  be MLS_INTERSECTION OR MLS_ADDITION)
  // As evident from the name choosing MLS_INTERSECTION will create a compound domain that is the intersection of two or more given domains and MLS_ADDITION will create a compound domain that is the union of two or more given domains

  for (int i = 0; i < bdry_phi_max_num; ++i)
  {
    if (*bdry_present_all[i] == true) bdry_phi_eff_cf.add_domain(bdry_phi_cf_all[i], (mls_opn_t) *bdry_opn_all[i]);
  }

  for (int i = 0; i < infc_phi_max_num; ++i)
  {
    if (*infc_present_all[i] == true) infc_phi_eff_cf.add_domain(infc_phi_cf_all[i], (mls_opn_t) *infc_opn_all[i]);
  }

  for (int i = 0; i < bdry1_phi_max_num; ++i)
  {
    if (*bdry1_present_all[i] == true) bdry1_phi_eff_cf.add_domain(bdry1_phi_cf_all[i], (mls_opn_t) *bdry1_opn_all[i]);
  }

  for (int i = 0; i < infc1_phi_max_num; ++i)
  {
    if (*infc1_present_all[i] == true) infc1_phi_eff_cf.add_domain(infc1_phi_cf_all[i], (mls_opn_t) *infc1_opn_all[i]);
  }

  int num_shifts_total = MULTD(num_shifts_x_dir.val, num_shifts_y_dir.val, num_shifts_z_dir.val);

  int num_resolutions = ((num_splits.val-1)*num_splits_per_split.val + 1)*mu_iter_num.val;
  int num_iter_total = num_resolutions*num_shifts_total;

  const int periodicity[] = { DIM(px(), py(), pz()) };
  const int num_trees[]   = { DIM(nx(), ny(), nz()) };
  const double grid_xyz_min[] = { DIM(xmin(), ymin(), zmin()) };
  const double grid_xyz_max[] = { DIM(xmax(), ymax(), zmax()) };

  // vectors to store convergence results
  vector<double> lvl_arr, h_arr, mu_arr;

  vector<double> error_sl_m_arr;
  vector<double> error_ex_m_arr;
  vector<double> error_dd_m_arr;
  vector<double> error_gr_m_arr;

  vector<double> error_sl_p_arr;
  vector<double> error_ex_p_arr;
  vector<double> error_dd_p_arr;
  vector<double> error_gr_p_arr;

  vector<double> cond_num_arr;

  // Start up a MATLAB Engine to calculate condidition number
#ifdef MATLAB_PROVIDED
  Engine *mengine = NULL;
  if (mpi.rank() == 0 && compute_cond_num())
  {
    mengine = engOpen("matlab -nodisplay -nojvm");
    if (mengine == NULL) throw std::runtime_error("Cannot start a MATLAB Engine session.\n");
  }
#else
  if (compute_cond_num())
  {
    ierr = PetscPrintf(mpi.comm(), "[Warning]: MATLAB is either not provided or found. Condition numbers will not be computed. \n");
  }
#endif


  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  int iteration = -1;
  int file_idx  = -1;

//  for(int mu_iter = 0; mu_iter < mu_iter_num(); ++mu_iter)
//  {
//    if (mu_iter_num() > 1)
//      mag_mu_m.val = mag_mu_m_min() * pow(mag_mu_m_max()/mag_mu_m_min(), ((double) mu_iter / (double) (mu_iter_num()-1)));

    for(int iter=0; iter<num_splits(); ++iter)
    {
      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin()+iter, lmax()+iter); CHKERRXX(ierr);

      int num_sub_iter = (iter == 0 ? 1 : num_splits_per_split());

      for (int sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter)
      {

        double grid_xyz_min_alt[3];
        double grid_xyz_max_alt[3];

        double scale = (double) (num_sub_iter-1-sub_iter) / (double) num_sub_iter;
        grid_xyz_min_alt[0] = grid_xyz_min[0] - .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]); grid_xyz_max_alt[0] = grid_xyz_max[0] + .5*(pow(2.,scale)-1)*(grid_xyz_max[0]-grid_xyz_min[0]);
        grid_xyz_min_alt[1] = grid_xyz_min[1] - .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]); grid_xyz_max_alt[1] = grid_xyz_max[1] + .5*(pow(2.,scale)-1)*(grid_xyz_max[1]-grid_xyz_min[1]);
        grid_xyz_min_alt[2] = grid_xyz_min[2] - .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]); grid_xyz_max_alt[2] = grid_xyz_max[2] + .5*(pow(2.,scale)-1)*(grid_xyz_max[2]-grid_xyz_min[2]);


        double dxyz[3] = { (grid_xyz_max_alt[0]-grid_xyz_min_alt[0])/pow(2., (double) lmax()+iter),
                           (grid_xyz_max_alt[1]-grid_xyz_min_alt[1])/pow(2., (double) lmax()+iter),
                           (grid_xyz_max_alt[2]-grid_xyz_min_alt[2])/pow(2., (double) lmax()+iter) };

        double grid_xyz_min_shift[3];
        double grid_xyz_max_shift[3];

        double dxyz_m = MIN(DIM(dxyz[0],dxyz[1],dxyz[2]));

        mu_arr.push_back(mag_mu_m());
        h_arr.push_back(dxyz_m);
        lvl_arr.push_back(lmax()+iter-scale);

        ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f).\n", lmin()+iter, lmax()+iter, sub_iter, lmin()+iter-scale, lmax()+iter-scale); CHKERRXX(ierr);

#ifdef P4_TO_P8
        for (int k_shift = 0; k_shift < num_shifts_z_dir(); ++k_shift)
        {
          grid_xyz_min_shift[2] = grid_xyz_min_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir()) * dxyz[2];
          grid_xyz_max_shift[2] = grid_xyz_max_alt[2] + (double) (k_shift) / (double) (num_shifts_z_dir()) * dxyz[2];
#endif
          for (int j_shift = 0; j_shift < num_shifts_y_dir(); ++j_shift)
          {
            grid_xyz_min_shift[1] = grid_xyz_min_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir()) * dxyz[1];
            grid_xyz_max_shift[1] = grid_xyz_max_alt[1] + (double) (j_shift) / (double) (num_shifts_y_dir()) * dxyz[1];

            for (int i_shift = 0; i_shift < num_shifts_x_dir(); ++i_shift)
            {
              grid_xyz_min_shift[0] = grid_xyz_min_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir()) * dxyz[0];
              grid_xyz_max_shift[0] = grid_xyz_max_alt[0] + (double) (i_shift) / (double) (num_shifts_x_dir()) * dxyz[0];

              iteration++;

              if (iteration < iter_start()) continue;

              file_idx++;

              connectivity = my_p4est_brick_new(num_trees, grid_xyz_min_shift, grid_xyz_max_shift, &brick, periodicity);
              p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

              if (refine_strict())
              {
                splitting_criteria_cf_t data_tmp(lmin(), lmax(), &phi_eff_cf, lip());
                p4est->user_pointer = (void*)(&data_tmp);

                my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
                for (int i = 0; i < iter; ++i)
                {
                  my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
                  my_p4est_partition(p4est, P4EST_FALSE, NULL);
                }
              } else {
                splitting_criteria_cf_t data_tmp(lmin()+iter, lmax()+iter, &phi_eff_cf, lip());
                p4est->user_pointer = (void*)(&data_tmp);

                for (int i = 0; i < lmax()+iter; ++i)
                {
                  my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
                  my_p4est_partition(p4est, P4EST_FALSE, NULL);
                }
              }
              // macromesh has been generated at this point.

              splitting_criteria_cf_t data(lmin()+iter, lmax()+iter, &phi_eff_cf, lip());
              p4est->user_pointer = (void*)(&data);

              if (refine_rand())
                my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);

              if (balance_grid())
              {
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
                // Balance type (face or corner/full).
                // Corner balance is almost never required when discretizing a PDE; just causes smoother mesh grading.
                p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
                my_p4est_partition(p4est, P4EST_FALSE, NULL);
              }

              ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
              if (expand_ghost())
                my_p4est_ghost_expand(p4est, ghost);
              nodes = my_p4est_nodes_new(p4est, ghost);

              my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
              my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);
              ngbd_n.init_neighbors();

              my_p4est_level_set_t ls(&ngbd_n);

              double dxyz[P4EST_DIM];
              dxyz_min(p4est, dxyz);
              double dxyz_max = MAX(DIM(dxyz[0], dxyz[1], dxyz[2]));
              double diag = sqrt(SUMD(dxyz[0]*dxyz[0], dxyz[1]*dxyz[1], dxyz[2]*dxyz[2]));

              // sample level-set functions
              Vec bdry_phi_vec_all[bdry_phi_max_num];
              Vec infc_phi_vec_all[infc_phi_max_num];

              Vec bdry1_phi_vec_all[bdry1_phi_max_num];
              Vec infc1_phi_vec_all[infc1_phi_max_num];

              //Perturbing domain and interface boundaries and reinitializing
              for (int i = 0; i < bdry_phi_max_num; ++i)
              {
                if (*bdry_present_all[i] == true)
                {
                  ierr = VecCreateGhostNodes(p4est, nodes, &bdry_phi_vec_all[i]); CHKERRXX(ierr);
                  sample_cf_on_nodes(p4est, nodes, bdry_phi_cf_all[i], bdry_phi_vec_all[i]);

                  if (dom_perturb())
                  {
                    double *phi_ptr;
                    ierr = VecGetArray(bdry_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                    {
                      double xyz[P4EST_DIM];
                      node_xyz_fr_n(n, p4est, nodes, xyz);
                      phi_ptr[n] += dom_perturb_mag()*dom_perturb_cf.value(xyz)*pow(dxyz_m, dom_perturb_pow());
                    }

                    ierr = VecRestoreArray(bdry_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    ierr = VecGhostUpdateBegin(bdry_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                    ierr = VecGhostUpdateEnd  (bdry_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                  }

                  if (reinit_level_set())
                  {
                    ls.reinitialize_1st_order_time_2nd_order_space(bdry_phi_vec_all[i], 20);
                  }
                }
              }
              for (int i = 0; i < bdry1_phi_max_num; ++i)
              {
                if (*bdry1_present_all[i] == true)
                {
                  ierr = VecCreateGhostNodes(p4est, nodes, &bdry1_phi_vec_all[i]); CHKERRXX(ierr);
                  sample_cf_on_nodes(p4est, nodes, bdry1_phi_cf_all[i], bdry1_phi_vec_all[i]);

                  if (dom_perturb())
                  {
                    double *phi_ptr;
                    ierr = VecGetArray(bdry1_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                    {
                      double xyz[P4EST_DIM];
                      node_xyz_fr_n(n, p4est, nodes, xyz);
                      phi_ptr[n] += dom_perturb_mag()*dom_perturb_cf.value(xyz)*pow(dxyz_m, dom_perturb_pow());
                    }

                    ierr = VecRestoreArray(bdry1_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    ierr = VecGhostUpdateBegin(bdry1_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                    ierr = VecGhostUpdateEnd  (bdry1_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                  }

                  if (reinit_level_set())
                  {
                    ls.reinitialize_1st_order_time_2nd_order_space(bdry1_phi_vec_all[i], 20);
                  }
                }
              }
              for (int i = 0; i < infc_phi_max_num; ++i)
              {
                if (*infc_present_all[i] == true)
                {
                  ierr = VecCreateGhostNodes(p4est, nodes, &infc_phi_vec_all[i]); CHKERRXX(ierr);
                  sample_cf_on_nodes(p4est, nodes, infc_phi_cf_all[i], infc_phi_vec_all[i]);

                  if (ifc_perturb())
                  {
                    double *phi_ptr;
                    ierr = VecGetArray(infc_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                    {
                      double xyz[P4EST_DIM];
                      node_xyz_fr_n(n, p4est, nodes, xyz);
                      phi_ptr[n] += ifc_perturb_mag()*ifc_perturb_cf.value(xyz)*pow(dxyz_m, ifc_perturb_pow());
                    }

                    ierr = VecRestoreArray(infc_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    ierr = VecGhostUpdateBegin(infc_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                    ierr = VecGhostUpdateEnd  (infc_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                  }

                  if (reinit_level_set())
                  {
                    ls.reinitialize_1st_order_time_2nd_order_space(infc_phi_vec_all[i], 20);
                  }
                }
              }

              for (int i = 0; i < infc1_phi_max_num; ++i)
              {
                if (*infc1_present_all[i] == true)
                {
                  ierr = VecCreateGhostNodes(p4est, nodes, &infc1_phi_vec_all[i]); CHKERRXX(ierr);
                  sample_cf_on_nodes(p4est, nodes, infc1_phi_cf_all[i], infc1_phi_vec_all[i]);

                  if (ifc_perturb())
                  {
                    double *phi_ptr;
                    ierr = VecGetArray(infc1_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                    {
                      double xyz[P4EST_DIM];
                      node_xyz_fr_n(n, p4est, nodes, xyz);
                      phi_ptr[n] += ifc_perturb_mag()*ifc_perturb_cf.value(xyz)*pow(dxyz_m, ifc_perturb_pow());
                    }

                    ierr = VecRestoreArray(infc1_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                    ierr = VecGhostUpdateBegin(infc1_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                    ierr = VecGhostUpdateEnd  (infc1_phi_vec_all[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                  }

                  if (reinit_level_set())
                  {
                    ls.reinitialize_1st_order_time_2nd_order_space(infc1_phi_vec_all[i], 20);
                  }
                }
              }
              //Initialize vectors relevant for the poisson problem
              vec_and_ptr_t u;

              vec_and_ptr_t u1;


              u.create(p4est,nodes);
              sample_cf_on_nodes(p4est,nodes,u_cf,u.vec);

              u1.create(p4est,nodes);
              sample_cf_on_nodes(p4est,nodes,u_cf1,u1.vec);

              // vectors to hold first derivatives
              vec_and_ptr_t u_d;
              vec_and_ptr_t u1_d;

              // vectors to hold second derivatives
              vec_and_ptr_t u_dd;
              vec_and_ptr_t u1_dd;

              //vectors to hold normals
              vec_and_ptr_dim_t d_normals;
              vec_and_ptr_dim_t d1_normals;
              ierr= PetscPrintf(mpi.comm(),"line 2532 ok \n");CHKERRXX(ierr);

              my_p4est_poisson_nodes_mls_t solver_u(&ngbd_n);  // will solve poisson problem for u in domain 1
              my_p4est_poisson_nodes_mls_t solver_u1(&ngbd_n);  // will solve poisson problem for u in domain 2

              //setting up solver for domains
              ierr= PetscPrintf(mpi.comm(),"line 2538 ok \n");CHKERRXX(ierr);
              solver_u.set_use_centroid_always(use_centroid_always());
              solver_u1.set_use_centroid_always(use_centroid_always());


              solver_u.set_store_finite_volumes(store_finite_volumes());
              solver_u1.set_store_finite_volumes(store_finite_volumes());

              solver_u.set_jump_scheme(jc_scheme());
              solver_u1.set_jump_scheme(jc_scheme());

              solver_u.set_jump_sub_scheme(jc_sub_scheme());
              solver_u1.set_jump_sub_scheme(jc_sub_scheme());

              solver_u.set_use_sc_scheme(sc_scheme());
              solver_u1.set_use_sc_scheme(sc_scheme());

              solver_u.set_integration_order(integration_order());
              solver_u1.set_integration_order(integration_order());

              solver_u.set_lip(lip());
              solver_u1.set_lip(lip());

              ierr= PetscPrintf(mpi.comm(),"line 2561 ok \n");CHKERRXX(ierr);

              // HOW TO ADD BOUNDARY

              //Adding Boundary and Interface for dommain 1
              // In the add_boundary(), creates a structure boundary_conditions_t
              for (int i = 0; i < bdry_phi_max_num; ++i)
                if (*bdry_present_all[i] == true)
                {
                  if (apply_bc_pointwise())
                    solver_u.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i], DIM(NULL, NULL, NULL), (BoundaryConditionType) *bc_type_all[i], zero_cf, zero_cf);
                  else
                    solver_u.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i], DIM(NULL, NULL, NULL), (BoundaryConditionType) *bc_type_all[i], bc_value_cf_all[i], bc_coeff_cf_all[i]);
                }
              // adding interface is exactly similar
              for (int i = 0; i < infc_phi_max_num; ++i)
                if (*infc_present_all[i] == true)
                {
                  if (apply_bc_pointwise())
                    solver_u.add_interface((mls_opn_t) *infc_opn_all[i], infc_phi_vec_all[i], DIM(NULL, NULL, NULL), zero_cf, zero_cf);
                  else
                    solver_u.add_interface((mls_opn_t) *infc_opn_all[i], infc_phi_vec_all[i], DIM(NULL, NULL, NULL), jc_value_cf_all[i], jc_flux_cf_all[i]);
                }
              ierr= PetscPrintf(mpi.comm(),"line 2584 ok \n");CHKERRXX(ierr);
              //Adding Boundary and Interface for dommain 2
              // In the add_boundary(), creates a structure boundary_conditions_t
              for (int i = 0; i < bdry1_phi_max_num; ++i)
                if (*bdry1_present_all[i] == true)
                {
                  if (apply_bc_pointwise())
                    solver_u1.add_boundary((mls_opn_t) *bdry1_opn_all[i], bdry1_phi_vec_all[i], DIM(NULL, NULL, NULL), (BoundaryConditionType) *bc1_type_all[i], zero_cf, zero_cf);
                  else
                    solver_u1.add_boundary((mls_opn_t) *bdry1_opn_all[i], bdry1_phi_vec_all[i], DIM(NULL, NULL, NULL), (BoundaryConditionType) *bc1_type_all[i], bc1_value_cf_all[i], bc1_coeff_cf_all[i]);
                }
              // adding interface is exactly similar
              for (int i = 0; i < infc1_phi_max_num; ++i)
                if (*infc1_present_all[i] == true)
                {
                  if (apply_bc_pointwise())
                    solver_u1.add_interface((mls_opn_t) *infc1_opn_all[i], infc1_phi_vec_all[i], DIM(NULL, NULL, NULL), zero_cf, zero_cf);
                  else
                    solver_u1.add_interface((mls_opn_t) *infc1_opn_all[i], infc1_phi_vec_all[i], DIM(NULL, NULL, NULL), jc1_value_cf_all[i], jc1_flux_cf_all[i]);
                }
              ierr= PetscPrintf(mpi.comm(),"line 2604 ok \n");CHKERRXX(ierr);
//              Vec bdry_phi_eff;
//              solver_u.set_boundary_phi_eff(bdry_phi_eff);
//              ierr= PetscPrintf(mpi.comm(),"line 2607 ok \n");CHKERRXX(ierr);
//              Vec infc_phi_eff;
//              solver_u.set_interface_phi_eff(infc_phi_eff);
//              ierr= PetscPrintf(mpi.comm(),"line 2610 ok \n");CHKERRXX(ierr);

//              Vec bdry1_phi_eff;
//              solver_u1.set_boundary_phi_eff(bdry1_phi_eff);
//              ierr= PetscPrintf(mpi.comm(),"line 2614 ok \n");CHKERRXX(ierr);
//              Vec infc1_phi_eff;
//              solver_u1.set_interface_phi_eff(infc1_phi_eff);
//              ierr= PetscPrintf(mpi.comm(),"line 2617 ok \n");CHKERRXX(ierr);

//              Vec phi_m;
//              ierr= PetscPrintf(mpi.comm(),"line 2619 ok \n");CHKERRXX(ierr);
//              if (bdry_phi_eff == NULL) { ierr= PetscPrintf(mpi.comm(),"bdry_phi_eff is NULL \n");CHKERRXX(ierr); }
//              ierr = VecDuplicate(bdry_phi_eff, &phi_m); CHKERRXX(ierr);
//              ierr= PetscPrintf(mpi.comm(),"line 2620 ok \n");CHKERRXX(ierr);
//              VecCopyGhost(bdry_phi_eff, phi_m);
//              ierr= PetscPrintf(mpi.comm(),"line 2622 ok \n");CHKERRXX(ierr);
//              VecPointwiseMaxGhost(phi_m, phi_m, infc_phi_eff);
//              ierr= PetscPrintf(mpi.comm(),"line 2624 ok \n");CHKERRXX(ierr);
//              VecScaleGhost(infc_phi_eff, -1);
//              ierr= PetscPrintf(mpi.comm(),"line 2626 ok \n");CHKERRXX(ierr);

//              Vec phi1_m; ierr = VecDuplicate(bdry1_phi_eff, &phi1_m); CHKERRXX(ierr); VecCopyGhost(bdry1_phi_eff, phi1_m);
//              VecPointwiseMaxGhost(phi1_m, phi1_m, infc1_phi_eff);
//              VecScaleGhost(infc1_phi_eff, -1);
//              ierr= PetscPrintf(mpi.comm(),"line 2627 ok \n");CHKERRXX(ierr);


              ierr= PetscPrintf(mpi.comm(),"line 2630 ok \n");CHKERRXX(ierr);



//              switch (extend_solution())
//              {
//                case 1:
//                  ls.extend_Over_Interface_TVD(phi_m, u.vec, extension_iterations(), 2, extension_tol(), -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max,  extension_band_check()*dxyz_max, NULL, NULL, NULL, use_nonzero_guess()); CHKERRXX(ierr);
//                  ls.extend_Over_Interface_TVD(phi1_m, u1.vec, extension_iterations(), 2, extension_tol(), -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max,  extension_band_check()*dxyz_max, NULL, NULL, NULL, use_nonzero_guess()); CHKERRXX(ierr);
//                  break;
//                case 2:
//                  ls.extend_Over_Interface_TVD_Full(phi_m, u.vec, extension_iterations(), 2, extension_tol(), -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max,  extension_band_check()*dxyz_max, NULL, NULL, NULL, use_nonzero_guess()); CHKERRXX(ierr);
//                  ls.extend_Over_Interface_TVD_Full(phi1_m, u1.vec, extension_iterations(), 2, extension_tol(), -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max,  extension_band_check()*dxyz_max, NULL, NULL, NULL, use_nonzero_guess()); CHKERRXX(ierr);
//                  break;
//              }
              ierr= PetscPrintf(mpi.comm(),"line 2635 ok \n");CHKERRXX(ierr);
              Vec mu_m;
              ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);

              Vec mu_p;
              ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);


              Vec mu_m1;
              ierr = VecCreateGhostNodes(p4est, nodes, &mu_m1); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, mu_m_cf1, mu_m1);

              Vec mu_p1;
              ierr = VecCreateGhostNodes(p4est, nodes, &mu_p1); CHKERRXX(ierr);
              sample_cf_on_nodes(p4est, nodes, mu_p_cf1, mu_p1);

              ierr= PetscPrintf(mpi.comm(),"line 2653 ok \n");CHKERRXX(ierr);



              solver_u.set_mu(mu_m, DIM(NULL, NULL, NULL),
                            mu_p, DIM(NULL, NULL, NULL));

              solver_u1.set_mu(mu_m1, DIM(NULL, NULL, NULL),
                            mu_p1, DIM(NULL, NULL, NULL));

              ierr= PetscPrintf(mpi.comm(),"line 2663 ok \n");CHKERRXX(ierr);

              if(method_.val == 2){
                  solver_u.set_diag(2./dt.val);
                  solver_u1.set_diag(2./dt.val);
              }
              else if(method_.val ==1){
                  solver_u.set_diag(1./dt.val);
                  solver_u1.set_diag(1./dt.val);
              }
              else{
                  throw std::invalid_argument("Error setting the diagonal. You must select a time-stepping method!\n");
              }


              solver_u.set_wc(bc_wall_type, u_cf);
              solver_u1.set_wc(bc_wall_type, u_cf1);

              solver_u.set_use_taylor_correction(taylor_correction());
              solver_u.set_kink_treatment(kink_special_treatment());

              solver_u1.set_use_taylor_correction(taylor_correction());
              solver_u1.set_kink_treatment(kink_special_treatment());


              int t_iterations= (int)tfinal.val/dt.val+1;
              vector<double> error_in_u(t_iterations,0.0);
              int tstep =0;
              for (tn=0; tn<tfinal.val; tn+=dt.val){
                tstep++;
                if (!keep_going) break;
                // Get current memory usage:
                PetscLogDouble mem1;
                if(check_memory_usage.val) PetscMemoryGetCurrentUsage(&mem1);

                PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
                ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3g , Timestep: %0.3e, Percent Done : %0.2f % \n ------------------------------------------- \n",tstep,tn,dt.val,((tn-tstart.val)/(tfinal.val - tstart.val))*100.0);

                vec_and_ptr_t rhs_u;

                vec_and_ptr_t rhs_u1;

                vec_and_ptr_t u_ex;

                rhs_u.create(p4est,nodes);
                rhs_u1.create(p4est,nodes);
                u_ex.create(p4est,nodes);

                setup_rhs(u,u1,
                          rhs_u,rhs_u1,
                          p4est,nodes,&ngbd_n,tn);

                setup_u_exact(u_ex,tn,p4est,nodes,&ngbd_n);

                solver_u.set_rhs(rhs_u.vec);
                solver_u1.set_rhs(rhs_u1.vec);

                solver_u.preassemble_linear_system();
                solver_u1.preassemble_linear_system();


                vec_and_ptr_t u_new;

                vec_and_ptr_t u1_new;

                u_new.create(p4est,nodes);
                u1_new.create(p4est,nodes);

                solver_u.solve(u_new.vec,use_nonzero_guess());
                solver_u1.solve(u1_new.vec,use_nonzero_guess());

                u.destroy();
                u1.destroy();

                u.create(p4est,nodes);
                u1.create(p4est,nodes);

                VecCopyGhost(u_new.vec,u.vec);
                VecCopyGhost(u1_new.vec,u1.vec);

                vec_and_ptr_t u_error;
                u_error.create(p4est,nodes);
                u.get_array();
                u_ex.get_array();
                u_error.get_array();
                foreach_node(n,nodes){
                  u_error.ptr[n] = abs(u.ptr[n]-u_ex.ptr[n]);
                }
                u_error.restore_array();
                u.restore_array();
                u_ex.restore_array();
                ierr = VecGhostUpdateBegin(u_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                ierr = VecGhostUpdateEnd(u_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


                Vec bdry_phi_eff = solver_u.get_boundary_phi_eff();
                Vec infc_phi_eff = solver_u.get_interface_phi_eff();

                Vec bdry1_phi_eff = solver_u1.get_boundary_phi_eff();
                Vec infc1_phi_eff = solver_u1.get_interface_phi_eff();

                if(save_vtk())
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
                         "." << iter <<
                         "." << tstep;

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


                  double *bdry_phi_eff_ptr, *infc_phi_eff_ptr;
                  double *bdry1_phi_eff_ptr, *infc1_phi_eff_ptr;

                  ierr = VecGetArray(bdry_phi_eff, &bdry_phi_eff_ptr); CHKERRXX(ierr);
                  ierr = VecGetArray(infc_phi_eff, &infc_phi_eff_ptr); CHKERRXX(ierr);

                  ierr = VecGetArray(bdry1_phi_eff, &bdry1_phi_eff_ptr); CHKERRXX(ierr);
                  ierr = VecGetArray(infc1_phi_eff, &infc1_phi_eff_ptr); CHKERRXX(ierr);


                  u.get_array();
                  u1.get_array();
                  u_ex.get_array();

                  double *mu_m_ptr;
                  double *mu_p_ptr;

                  double *mu_m_ptr1;
                  double *mu_p_ptr1;

                  ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
                  ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);

                  ierr = VecGetArray(mu_m1, &mu_m_ptr1); CHKERRXX(ierr);
                  ierr = VecGetArray(mu_p1, &mu_p_ptr1); CHKERRXX(ierr);

                  if (u.vec == NULL){ierr = PetscPrintf(mpi.comm(), "u is NULL\n"); CHKERRXX(ierr);}
                  if (u1.vec == NULL){ierr = PetscPrintf(mpi.comm(), "u1 is NULL\n"); CHKERRXX(ierr);}
                  u_error.get_array();

                  my_p4est_vtk_write_all(p4est, nodes, ghost,
                                         P4EST_TRUE, P4EST_TRUE,
                                         6, 1, oss.str().c_str(),

                                         VTK_POINT_DATA, "phi", bdry_phi_eff_ptr,
                                         VTK_POINT_DATA, "phi1", bdry1_phi_eff_ptr,
                                         VTK_POINT_DATA, "u", u.ptr,
                                         VTK_POINT_DATA, "u1", u1.ptr,
                                         VTK_POINT_DATA, "u_exact", u_ex.ptr,
                                         VTK_POINT_DATA, "u_error", u_error.ptr,
                                         VTK_CELL_DATA , "leaf_level", l_p);

                  u_error.restore_array();
                  ierr = VecRestoreArray(bdry_phi_eff, &bdry_phi_eff_ptr); CHKERRXX(ierr);
                  ierr = VecRestoreArray(infc_phi_eff, &infc_phi_eff_ptr); CHKERRXX(ierr);

                  ierr = VecRestoreArray(bdry1_phi_eff, &bdry1_phi_eff_ptr); CHKERRXX(ierr);
                  ierr = VecRestoreArray(infc1_phi_eff, &infc1_phi_eff_ptr); CHKERRXX(ierr);

                  ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
                  ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);

                  ierr = VecRestoreArray(mu_m1, &mu_m_ptr1); CHKERRXX(ierr);
                  ierr = VecRestoreArray(mu_p1, &mu_p_ptr1); CHKERRXX(ierr);

                  u_ex.restore_array();
                  u1.restore_array();
                  u.restore_array();
                  double u_max =0.; ierr = VecMax(u.vec, NULL, &u_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                  double u_err_max =0.; ierr = VecMax(u_error.vec, NULL, &u_err_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &u_err_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                  //scaling error
                  u_err_max=u_err_max/u_max;
                  error_in_u.push_back(u_max);

                  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
                  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

                  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
                }

                u_new.destroy();
                u1_new.destroy();

                rhs_u.destroy();
                rhs_u1.destroy();

                u_ex.destroy();

            }

              ierr = VecDestroy(mu_m);          CHKERRXX(ierr);
              ierr = VecDestroy(mu_p);          CHKERRXX(ierr);

              ierr = VecDestroy(mu_m1);          CHKERRXX(ierr);
              ierr = VecDestroy(mu_p1);          CHKERRXX(ierr);

              std::string filename_dat_file;
              filename_dat_file="/home/rochi/LabCode/executables/diffusion_debug_2d/output/error_in_u.dat";
              ofstream outputFile;
              outputFile.open (filename_dat_file.c_str());
              outputFile << 'error'<<'\n';
              for(int t=0; t<t_iterations; t++)
              {
                outputFile<<error_in_u[t] <<'\n';
              }
              outputFile<<'\n';
              outputFile.close();
              error_in_u.clear();



              for (unsigned int i = 0; i < bdry_phi_max_num; i++) { if (*bdry_present_all[i] == true) { ierr = VecDestroy(bdry_phi_vec_all[i]); CHKERRXX(ierr); } }
              for (unsigned int i = 0; i < infc_phi_max_num; i++) { if (*infc_present_all[i] == true) { ierr = VecDestroy(infc_phi_vec_all[i]); CHKERRXX(ierr); } }

              for (unsigned int i = 0; i < bdry1_phi_max_num; i++) { if (*bdry1_present_all[i] == true) { ierr = VecDestroy(bdry1_phi_vec_all[i]); CHKERRXX(ierr); } }
              for (unsigned int i = 0; i < infc1_phi_max_num; i++) { if (*infc1_present_all[i] == true) { ierr = VecDestroy(infc1_phi_vec_all[i]); CHKERRXX(ierr); } }

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
