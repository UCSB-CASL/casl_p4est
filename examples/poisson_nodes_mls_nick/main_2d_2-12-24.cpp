/*
 *                              Solves the elliptic system of the form:
 *
 *      -  div ( mu(x,y,z) * grad(h) ) + lambda(x,y,z) * h + nu(x,y,z) * f(h) = rhs
 *
 * nu    (x,y,z) is the variable coefficient in front of the nonlinear term f(h).
 * lambda(x,y,z) is the variable coefficient in front of the    linear term   h .
 * mu    (x,y,z) is the variable coefficient of the Poisson's operator, which corresponds to:
 *                  . the viscosity coefficient in the Navier-Stokes implicit viscosity step,
 *                  . the inverse of the density coefficient in the Navier-Stokes projection step,
 *                  . the diffusion coefficient in diffusion problems,
 *                  . the thermal diffusivity coefficient in heat problems,
 *                  . etc.
 * f = f(h)      is a nonlinear function we can choose.
 * rhs           is the right-hand side that is computed automatically when considering an exact solution.
 *
 * The following describes how to set up a problem in the function set_example():
 *
 * The exact solution h(x,y,z) we consider is defined by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_h_m" ("which_h_p"), which refers to one of the available functional forms
 *         defined in the class "h_choices_cf_t".
 *      b- Specifying the scalar "h_multiplier_m" multiplying the functional form.
 *      Note: the class h_choices_cf_t also defines the solution's gradient (hx, hy, hz) and its
 *      Laplacian div( grad(h) ).
 *      ex:
 *          . h in the negative domain is defined as:
 *            h_choices_cf_t h_m_cf(VAL, which_h_m.val, h_multiplier_m.val);
 *          . hx in the negative domain is defined as:
 *            h_choices_cf_t hx_m_cf(DDX, which_h_m.val, h_multiplier_m.val);
 *          . hy in the negative domain is defined as:
 *            h_choices_cf_t hy_m_cf(DDY, which_h_m.val, h_multiplier_m.val);
 *          . Laplace(h) in the negative domain is defined as:
 *            h_choices_cf_t Laplace_h_m_cf(LAP, which_h_m.val, h_multiplier_m.val);
 *
 * The coefficient mu(x,y,z) > 0 is set by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_mu_m" ("which_mu_p"), which refers to one of the available functional forms
 *         defined in the class "mu_choices_cf_t".
 *      b- Specifying the scalar "mu_multiplier_m" multiplying the functional form.
 *
 *
 * The coefficient lambda(x,y,z) is defined by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_lambda_m" ("which_lambda_p"), which refers to one of the available functional forms
 *         defined in the class "lambda_choices_cf_t".
 *      b- Specifying the scalar "lambda_multiplier_m" multiplying the functional form.
 *         Setting it to 0 means we do not consider the linear term.
 *
 *
 * The coefficient nu(x,y,z) is defined by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_nu_m" ("which_nu_p"), which refers to one of the available functional forms
 *         defined in the class "nu_choices_cf_t".
 *      b- Specifying the scalar "nu_multiplier_m" multiplying the functional form.
 *         Setting it to 0 means we do not consider the nonlinear term.
 *
 *
 * The nonlinear term f(h) is set by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_f_m" ("which_f_p"), which refers to one of the available functional forms
 *         defined in the class "f_choices_cf_t".
 *      Note that the nonlinear term f(h) can be sinh(h) or 1/(1+h), etc., but also h. Although f(h) = h
 *      is not nonlinear, this case is to show how it is handled numerically when treated as a nonlinear
 *      term.
 *
 *
 * The geometry of the boundaries (i.e., where we can impose Dirichlet, Neumann and Robin bc) is defined by:
 *      a- Specifying the number of boundaries with the parameter "num_bdry".
 *         Note that the code is currently setup to handle at most 4 boundaries, but this can easily be
 *         extended further. Therefore, currently the value of "xx" in the following lines can be either
 *         01, 02, 03 or 04, identifying the boundary number.
 *      b- Specifying the parameter "is_bdry_xx_present", a boolean indicating whether bdry_xx exists or not.
 *      c- Specifying the parameter "which_geometry_for_bdry_xx", which refers to one of the available functional forms
 *         defined in the class "bdry_phi_choices_cf_t".
 *      d- Specifying the parameter "bdry_xx_opn", which indicates which operation (MLS_INT for intersection
 *         or MLS_ADD for union) to perform when adding bdry_xx.
 *      e- Specifying the type of boundary condition ("NEUMANN" or "DIRICHLET" or "ROBIN") with the parameter
 *         "bdry_xx_bc_type.val".
 *
 * In the case of a ROBIN boundary condition (\partial h/ \partial n + coefficient(x,y,z) * h = given),
 * the coefficient(x,y,z) is set by:
 *      a- Specifying its functional form in the with the parameter "which_robin_coeff_for_bdry_xx", which
 *         refers to one of the available functional forms defined in the class "robin_bc_coeff_choices_cf_t".
 *      b- Specifying the scalar "bdry_xx_robin_coeff_multiplier" multiplying the functional form.
 *
 *
 * The geometry of the interfaces (i.e., where we impose a jump condition) is defined similarly, by replacing
 * bdry with infc in all the names above.
 *
 *
 * NOTE: For a given exact solution, the rhs of the elliptic equation, of the Neumann, of the Dirichlet, of
 * the Robin boundary, and of the jump conditions are automatically computed.
 *
 * Rhn the program with the -help flag to see the available options.
 *
*/

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <ctime>
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

const int num_bdry_max = 4;
const int num_infc_max = 4;

param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<int>    px (pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<int>    py (pl, 0, "py", "Periodicity in the y-direction (0/1)");
param_t<int>    pz (pl, 0, "pz", "Periodicity in the z-direction (0/1)");

param_t<int>    nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int>    ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int>    nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<double> xmin (pl, -1, "xmin", "Box xmin");
param_t<double> ymin (pl, -1, "ymin", "Box ymin");
param_t<double> zmin (pl, -1, "zmin", "Box zmin");

param_t<double> xmax (pl,  1, "xmax", "Box xmax");
param_t<double> ymax (pl,  1, "ymax", "Box ymax");
param_t<double> zmax (pl,  1, "zmax", "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
param_t<int>    lmin (pl, 1, "lmin", "Min level of the tree");
param_t<int>    lmax (pl, 2, "lmax", "Max level of the tree");

param_t<int>    num_splits (pl, 5, "num_splits", "Number of recursive splits");
param_t<int>    add_splits (pl, 1, "add_splits", "Number of additional resolutions");

param_t<int>    num_shifts_x_dir (pl, 1, "num_shifts_x_dir", "Number of grid shifts in the x-direction");
param_t<int>    num_shifts_y_dir (pl, 1, "num_shifts_y_dir", "Number of grid shifts in the y-direction");
param_t<int>    num_shifts_z_dir (pl, 1, "num_shifts_z_dir", "Number of grid shifts in the z-direction");
#else
param_t<int>    lmin (pl, 3, "lmin", "Min level of the tree");
param_t<int>    lmax (pl, 6, "lmax", "Max level of the tree");

param_t<int>    num_splits (pl, 5, "num_splits", "Number of recursive splits");
param_t<int>    add_splits (pl, 1, "add_splits", "Number of additional resolutions");

param_t<int>    num_shifts_x_dir (pl, 1, "num_shifts_x_dir", "Number of grid shifts in the x-direction");
param_t<int>    num_shifts_y_dir (pl, 1, "num_shifts_y_dir", "Number of grid shifts in the y-direction");
param_t<int>    num_shifts_z_dir (pl, 1, "num_shifts_z_dir", "Number of grid shifts in the z-direction");
#endif

param_t<double> lip  (pl, 1.2, "lip",  "Transition width from coarse grid to fine grid (a.k.a. Lipschitz constant)");
param_t<double> band (pl, 4.0, "band", "Width of the uniform band around boundaries and interfaces (in lengths of smallest quadrants)");

param_t<bool>   refine_strict  (pl, false, "refine_strict",  "Refines every cell starting from the coarsest case if yes");
param_t<bool>   refine_rand    (pl, false, "refine_rand",    "Add randomness into adaptive grid");
param_t<bool>   balance_grid   (pl, true , "balance_grid",   "Enforce 1:2 ratio for adaptive grid");
param_t<bool>   coarse_outside (pl, false, "coarse_outside", "Use the coarsest possible grid outside the domain (0/1)");
param_t<int>    expand_ghost   (pl, false, "expand_ghost",   "Number of ghost layer expansions");
param_t<int>    iter_start     (pl, false, "iter_start",     "Skip n first iterations (for debugging)");

//-------------------------------------
// equation parameters
//-------------------------------------
param_t<int>    which_h_m           (pl, 0, "which_h_m", "Case number (id) of the exact solution we consider in the negative domain - this specifies which of the cases defined in the class h_choices_cf_t we use.");
param_t<int>    which_h_p           (pl, 0, "which_h_p", "Case number (id) of the exact solution we consider in the positive domain - this specifies which of the cases defined in the class h_choices_cf_t we use.");
param_t<double> h_multiplier_m      (pl, 1, "h_multiplier_m", "Multiplier value of the exact solution we consider in the negative domain");
param_t<double> h_multiplier_p      (pl, 1, "h_multiplier_p", "Multiplier value of the exact solution we consider in the positive domain");

//nick add begin
param_t<int>    which_ustar_m           (pl, 0, "which_h_m", "Case number (id) of the exact solution we consider in the negative domain - this specifies which of the cases defined in the class h_choices_cf_t we use.");
param_t<int>    which_ustar_p           (pl, 0, "which_h_p", "Case number (id) of the exact solution we consider in the positive domain - this specifies which of the cases defined in the class h_choices_cf_t we use.");
param_t<double> ustar_multiplier_m      (pl, 1, "h_multiplier_m", "Multiplier value of the exact solution we consider in the negative domain");
param_t<double> ustar_multiplier_p      (pl, 1, "h_multiplier_p", "Multiplier value of the exact solution we consider in the positive domain");
//nick add end

param_t<int>    which_f_m           (pl, 0, "which_f_m", "Case number (id) of the functional form of fh in the negative domain: 0 - no nonlinear term, 1 - fh = u, 2 - fh = sinhh, 3 - fh = u/(1+u)");
param_t<int>    which_f_p           (pl, 0, "which_f_p", "Case number (id) of the functional form of fh in the positive domain: 0 - no nonlinear term, 1 - fh = u, 2 - fh = sinhh, 3 - fh = u/(1+u)");

param_t<int>    which_lambda_m      (pl, 0, "which_lambda_m", "Case number (id) of the coefficient in front of lambda(x,y,z) in the negative domain - this specifies which of the cases defined in the class lambda_choices_cf_t we use.");
param_t<int>    which_lambda_p      (pl, 0, "which_lambda_p", "Case number (id) of the coefficient in front of lambda(x,y,z) in the positive domain - this specifies which of the cases defined in the class lambda_choices_cf_t we use.");
param_t<double> lambda_multiplier_m (pl, 1, "lambda_multiplier_m", "Multiplier value in front of lambda(x,y,z) in the negative domain");
param_t<double> lambda_multiplier_p (pl, 1, "lambda_multiplier_p", "Multiplier value in front of lambda(x,y,z) in the positive domain");

param_t<int>    which_mu_m          (pl, 0, "which_mu_m", "Case number (id) of mu(x,y,z) in the negative domain");
param_t<int>    which_mu_p          (pl, 0, "which_mu_p", "Case number (id) of mu(x,y,z) in the positive domain");
param_t<double> mu_multiplier_m     (pl, 1, "mu_multiplier_m", "Multiplier in front of mu(x,y,z) in the negative domain");
param_t<double> mu_multiplier_p     (pl, 1, "mu_multiplier_p", "Multiplier in front of mu(x,y,z) in the positive domain");

param_t<int>    which_nu_m          (pl, 0, "which_nu_m", "Case number (id) of the coefficient in front nu(x,y,z) in the negative domain: 0 - constant, 1 - ... ");
param_t<int>    which_nu_p          (pl, 0, "which_nu_p", "Case number (id) of the coefficient in front nu(x,y,z) in the positive domain: 0 - constant, 1 - ... ");
param_t<double> nu_multiplier_m     (pl, 1, "nu_multiplier_m", "Multiplier in front of nu(x,y,z) in the negative domain - 0 means no nonlinear term");
param_t<double> nu_multiplier_p     (pl, 1, "nu_multiplier_p", "Multiplier in front of nu(x,y,z) in the positive domain - 0 means no nonlinear term");

param_t<int>    mu_iter_num         (pl, 1, "mu_iter_num", "Number of mu(x,y,z) coefficients to test");
param_t<double> mu_multiplier_min_m (pl, 1, "mu_multiplier_min_m", "Minimum value of the mu(x,y,z) coefficient multiplier in the negative domain");
param_t<double> mu_multiplier_max_m (pl, 1, "mu_multiplier_max_m", "Maximum value of the mu(x,y,z) coefficient multiplier in the negative domain");

param_t<int>    rhs_m_value         (pl, 0, "rhs_m_value", "Source term in negative domain: 0 - automatic (method of manufactured solutions), 1 - zero");
param_t<int>    rhs_p_value         (pl, 0, "rhs_p_value", "Source term in positive domain: 0 - automatic (method of manufactured solutions), 1 - zero");

param_t<int>    num_bdry            (pl, 0, "num_bdry", "Number of domain boundaries");
param_t<int>    num_infc            (pl, 0, "num_infc", "Number of immersed interfaces");

// boundary geometry parameters
param_t<int>    wc_type (pl, DIRICHLET, "wc_type", "Type of boundary conditions on the walls");

param_t<bool>   is_bdry_00_present (pl, false, "is_bdry_00_present", "Turn on/off boundary no. 0");
param_t<bool>   is_bdry_01_present (pl, false, "is_bdry_01_present", "Turn on/off boundary no. 1");
param_t<bool>   is_bdry_02_present (pl, false, "is_bdry_02_present", "Turn on/off boundary no. 2");
param_t<bool>   is_bdry_03_present (pl, false, "is_bdry_03_present", "Turn on/off boundary no. 3");

param_t<int>    which_geometry_for_bdry_00 (pl, 0, "which_geometry_for_bdry_00", "Geometry of boundary no. 0");
param_t<int>    which_geometry_for_bdry_01 (pl, 0, "which_geometry_for_bdry_01", "Geometry of boundary no. 1");
param_t<int>    which_geometry_for_bdry_02 (pl, 0, "which_geometry_for_bdry_02", "Geometry of boundary no. 2");
param_t<int>    which_geometry_for_bdry_03 (pl, 0, "which_geometry_for_bdry_03", "Geometry of boundary no. 3");

param_t<int>    bdry_00_opn     (pl, MLS_INTERSECTION, "bdry_00_opn", "Operation used to add boundary no. 0: 0 - intersection, 1 - union");
param_t<int>    bdry_01_opn     (pl, MLS_INTERSECTION, "bdry_01_opn", "Operation used to add boundary no. 1: 0 - intersection, 1 - union");
param_t<int>    bdry_02_opn     (pl, MLS_INTERSECTION, "bdry_02_opn", "Operation used to add boundary no. 2: 0 - intersection, 1 - union");
param_t<int>    bdry_03_opn     (pl, MLS_INTERSECTION, "bdry_03_opn", "Operation used to add boundary no. 3: 0 - intersection, 1 - union");

param_t<int>    bdry_00_bc_type (pl, DIRICHLET, "bdry_00_bc_type", "Type of boundary conditions on boundary no. 0");
param_t<int>    bdry_01_bc_type (pl, DIRICHLET, "bdry_01_bc_type", "Type of boundary conditions on boundary no. 1");
param_t<int>    bdry_02_bc_type (pl, DIRICHLET, "bdry_02_bc_type", "Type of boundary conditions on boundary no. 2");
param_t<int>    bdry_03_bc_type (pl, DIRICHLET, "bdry_03_bc_type", "Type of boundary conditions on boundary no. 3");

param_t<int>    which_robin_coeff_for_bdry_00 (pl, 0, "which_robin_coeff_for_bdry_00", "Case number (id) of the Robin coefficient on boundary no. 0");
param_t<int>    which_robin_coeff_for_bdry_01 (pl, 0, "which_robin_coeff_for_bdry_01", "Case number (id) of the Robin coefficient on boundary no. 1");
param_t<int>    which_robin_coeff_for_bdry_02 (pl, 0, "which_robin_coeff_for_bdry_02", "Case number (id) of the Robin coefficient on boundary no. 2");
param_t<int>    which_robin_coeff_for_bdry_03 (pl, 0, "which_robin_coeff_for_bdry_03", "Case number (id) of the Robin coefficient on boundary no. 3");

param_t<double> bdry_00_robin_coeff_multiplier (pl, 1, "bdry_00_robin_coeff_multiplier", "Multiplier of robin coefficient on boundary no. 0");
param_t<double> bdry_01_robin_coeff_multiplier (pl, 1, "bdry_01_robin_coeff_multiplier", "Multiplier of robin coefficient on boundary no. 1");
param_t<double> bdry_02_robin_coeff_multiplier (pl, 1, "bdry_02_robin_coeff_multiplier", "Multiplier of robin coefficient on boundary no. 2");
param_t<double> bdry_03_robin_coeff_multiplier (pl, 1, "bdry_03_robin_coeff_multiplier", "Multiplier of robin coefficient on boundary no. 3");

// interface geometry parameters
param_t<bool>   is_infc_00_present (pl, false, "infc_00_present", "Turn on/off interface no. 0");
param_t<bool>   is_infc_01_present (pl, false, "is_infc_01_present", "Turn on/off interface no. 1");
param_t<bool>   is_infc_02_present (pl, false, "is_infc_02_present", "Turn on/off interface no. 2");
param_t<bool>   is_infc_03_present (pl, false, "is_infc_03_present", "Turn on/off interface no. 3");

param_t<int>    which_geometry_for_infc_00 (pl, 0, "which_geometry_for_infc_00", "Geometry of interface no. 0");
param_t<int>    which_geometry_for_infc_01 (pl, 0, "which_geometry_for_infc_01", "Geometry of interface no. 1");
param_t<int>    which_geometry_for_infc_02 (pl, 0, "which_geometry_for_infc_02", "Geometry of interface no. 2");
param_t<int>    which_geometry_for_infc_03 (pl, 0, "which_geometry_for_infc_03", "Geometry of interface no. 3");

param_t<int>    infc_00_opn (pl, MLS_INTERSECTION, "infc_00_opn", "Operation to add interface no. 0: 0 - itersection, 1 - union");
param_t<int>    infc_01_opn (pl, MLS_INTERSECTION, "infc_01_opn", "Operation to add interface no. 1: 0 - itersection, 1 - union");
param_t<int>    infc_02_opn (pl, MLS_INTERSECTION, "infc_02_opn", "Operation to add interface no. 2: 0 - itersection, 1 - union");
param_t<int>    infc_03_opn (pl, MLS_INTERSECTION, "infc_03_opn", "Operation to add interface no. 3: 0 - itersection, 1 - union");

param_t<int>    infc_00_value_jump (pl, 0, "infc_00_value_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_01_value_jump (pl, 0, "infc_01_value_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_02_value_jump (pl, 0, "infc_02_value_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_03_value_jump (pl, 0, "infc_03_value_jump", "0 - automatic, others - hardcoded");

param_t<int>    infc_00_flux_jump (pl, 0, "infc_00_flux_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_01_flux_jump (pl, 0, "infc_01_flux_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_02_flux_jump (pl, 0, "infc_02_flux_jump", "0 - automatic, others - hardcoded");
param_t<int>    infc_03_flux_jump (pl, 0, "infc_03_flux_jump", "0 - automatic, others - hardcoded");

param_t<int>    example (pl, 34, "example", "Predefined example:\n"
                                           "0 - no interfaces, no boundaries\n"
                                           "1 - sphere interior\n"
                                           "2 - sphere exterior\n"
                                           "3 - moderately flower-shaped domain\n"
                                           "4 - highly flower-shaped domain\n"
                                           "5 - triangle/tetrahedron example from (Bochkov&Gibou, JCP, 2019)\n"
                                           "6 - union of spheres example from (Bochkov&Gibou, JCP, 2019)\n"
                                           "7 - difference of spheres example from (Bochkov&Gibou, JCP, 2019)\n"
                                           "8 - three stars example from (Bochkov&Gibou, JCP, 2019)\n"
                                           "9 - spherical interface\n"
                                           "10 - moderately flower-shaped interface\n"
                                           "11 - highly flower-shaped interface\n"
                                           "12 - curvy interface in an annular region from (Bochkov&Gibou, JCP, 2019)\n"
                                           "13 - example fromm Maxime's multiphase paper\n"
                                           "14 - spherical interface - case 4 from voronoi jump solver3D\n"
                                           "15 - spherical interface - case 1 from voronoi jump solver3D\n"
                                           "16 - spherical interface - case 5 from voronoi jump solver3D\n"
                                           "17 - spherical interface - case 3 from voronoi jump solver2D\n"
                                           "18 - spherical interface - case 0 from voronoi jump solver3D\n"
                                           "19 - not defined\n"
                                           "20 - clusters of particles\n"
                                           "21 - same as no. 0 + nonlinear sinh term\n"
                                           "22 - same as no. 1 + nonlinear sinh term\n"
                                           "23 - same as no. 5 + nonlinear sinh term\n"
                                           "24 - same as no. 9 + nonlinear sinh term\n"
                                           "25 - same as no. 12 + nonlinear sinh term\n"
                                           "30 - intersection of two circles\n"
                                           "31 - union of two circles\n"
                                           "32 - sphere interior to test Poisson with Neumann boundary conditions\n"
                                           "33 - no interfaces and no boundaries to test Poisson with Neumann boundary conditions (BUGS)\n"
                                           "34 - sphere interior with u to test projection step\n");

//-------------------------------------
// solver parameters
//-------------------------------------
param_t<int>    jump_scheme            (pl, 0   , "jump_scheme",       "Interpolation scheme for interface conditions (0 - from slow region, 1 - from fast region, 2 - based on nodes availability)");
param_t<int>    integration_order      (pl, 2   , "integration_order", "Integration order for finite volume discretization (1 - linear, 2 - quadratic)");
param_t<bool>   fv_scheme              (pl, true, "fv_scheme",         "Scheme for finite volume discretization on boundaries: 0 - symmetric, 1 - superconvergent");

param_t<bool>   store_finite_volumes   (pl, false, "store_finite_volumes", "");
param_t<bool>   apply_bc_pointwise     (pl, true , "apply_bc_pointwise", "");
param_t<bool>   use_centroid_always    (pl, false, "use_centroid_always", "");
param_t<bool>   sample_bc_node_by_node (pl, false, "sample_bc_node_by_node", "");

// for symmetric finite volume scheme:
param_t<bool>   taylor_correction      (pl, true, "taylor_correction",      "Use Taylor correction to approximate Robin term (symmetric scheme)");
param_t<bool>   kink_special_treatment (pl, true, "kink_special_treatment", "Use the special treatment for kinks (symmetric scheme)");


// for solving nonlinear equations
param_t<int>    nonlinear_method (pl, 1, "nonlinear_method", "Method to solve nonlinear eqautions: 0 - solving for solution itself, 1 - solving for change in the solution");
param_t<int>    nonlinear_itmax  (pl, 10, "nonlinear_itmax", "Maximum iteration for solving nonlinear equations");
param_t<double> nonlinear_tol    (pl, 1.e-10, "nonlinear_tol", "Tolerance for solving nonlinear equations");

//-------------------------------------
// level-set representation parameters
//-------------------------------------
param_t<bool>   use_phi_cf       (pl, false, "use_phi_cf", "Use analytical level-set functions");
param_t<bool>   reinit_level_set (pl, true , "reinit_level_set", "Reinitialize level-set function");

// artificial perturbation of level-set values
param_t<int>    bdry_perturb     (pl, 0,     "bdry_perturb",     "Artificially pertub level-set functions of boundaries (0 - no perturbation, 1 - smooth, 2 - noisy)");
param_t<double> bdry_perturb_mag (pl, 1.e-1, "bdry_perturb_mag", "Magnitude of level-set perturbations for boundaries");
param_t<double> bdry_perturb_pow (pl, 2,     "bdry_perturb_pow", "Order of level-set perturbation for boundaries (e.g. 2 for h^2 perturbations)");

param_t<int>    infc_perturb     (pl, 0,     "infc_perturb",     "Artificially pertub level-set functions of interfaces (0 - no perturbation, 1 - smooth, 2 - noisy)");
param_t<double> infc_perturb_mag (pl, 1.e-6, "infc_perturb_mag", "Magnitude of level-set perturbations for interfaces");
param_t<double> infc_perturb_pow (pl, 2,     "infc_perturb_pow", "Order of level-set perturbation for interfaces (e.g. 2 for h^2 perturbations)");

//-------------------------------------
// convergence study parameters
//-------------------------------------
param_t<bool>   compute_cond_num       (pl, false  , "compute_cond_num", "Estimate L1-norm condition number (requires MATLAB)");
param_t<int>    extend_solution        (pl, 2      , "extend_solution", "Extend solution after solving: 0 - no extension, 1 - extend using normal derivatives, 2 - extend using all derivatives");
param_t<double> mask_thresh            (pl, 0      , "mask_thresh", "Mask threshold for excluding points in convergence study");
param_t<bool>   compute_grad_between   (pl, false  , "compute_grad_between", "Computes gradient between points if yes");
param_t<bool>   scale_errors           (pl, false  , "scale_errors", "Scale errors by max solution/gradient value");
param_t<bool>   use_nonzero_guess      (pl, false  , "use_nonzero_guess", "");
param_t<double> extension_band_extend  (pl, 60     , "extension_band_extend", "");
param_t<double> extension_band_compute (pl, 6      , "extension_band_compute", "");
param_t<double> extension_band_check   (pl, 6      , "extension_band_check", "");
param_t<double> extension_tol          (pl, -1.e-10, "extension_tol", "");
param_t<int>    extension_iterations   (pl, 100    , "extension_iterations", "");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool>   save_vtk           (pl, true , "save_vtk", "Save the p4est in vtk format");
param_t<bool>   save_params        (pl, false, "save_params", "Save list of entered parameters");
param_t<bool>   save_domain        (pl, true , "save_domain", "Save the reconstruction of an irregular domain (works only in serial!)");
param_t<bool>   save_matrix_ascii  (pl, false, "save_matrix_ascii", "Save the matrix in ASCII MATLAB format");
param_t<bool>   save_matrix_binary (pl, false, "save_matrix_binary", "Save the matrix in BINARY MATLAB format");
param_t<bool>   save_convergence   (pl, false, "save_convergence", "Save convergence results");


void set_example(int example_)   // NEED TO BE CLEANED
{
    switch (example_)
    {
        case 34: // To test projections
            lambda_multiplier_m.val = 0;
            lambda_multiplier_p.val = 0;

            // No nonlinear term:
            nu_multiplier_m.val = 0;
            nu_multiplier_p.val = 0;

            // mu = constant (i.e., id = 0) = 1:
            which_mu_m.val = 0; mu_multiplier_m.val = 1;
            which_mu_p.val = 0; mu_multiplier_p.val = 1;

// NEED TO CHANGE WHICH_U TO CONSIDER THE SOLUTION OF THE REPEATED PROJECTION STEP.
            // Exact solution is case 0 in h_choices_cf_t: h_multiplier * sin(x) * cos(y);
            which_h_m.val = 0; h_multiplier_m.val = 1;
            which_h_p.val = 0; h_multiplier_p.val = 1;

            // Define the geometry (here we solve the solution inside a disk/sphere, i.e. which_geometry_for_bdry_00.val = 0):
            num_bdry.val = 1; num_infc.val = 0;
            is_bdry_00_present.val = true; which_geometry_for_bdry_00.val = 0; bdry_00_opn.val = MLS_INT;
            bdry_00_bc_type.val = NEUMANN;

            break;

        default:
            throw std::invalid_argument("Invalid case in function set_example(...)");
    }
}


bool *bdry_present_all[] = { &is_bdry_00_present.val,
                             &is_bdry_01_present.val,
                             &is_bdry_02_present.val,
                             &is_bdry_03_present.val };

int *bdry_geometry_all[] = { &which_geometry_for_bdry_00.val,
                             &which_geometry_for_bdry_01.val,
                             &which_geometry_for_bdry_02.val,
                             &which_geometry_for_bdry_03.val };

int *bdry_opn_all[] = { &bdry_00_opn.val,
                        &bdry_01_opn.val,
                        &bdry_02_opn.val,
                        &bdry_03_opn.val };

int *bc_coeff_all[] = { &which_robin_coeff_for_bdry_00.val,
                        &which_robin_coeff_for_bdry_01.val,
                        &which_robin_coeff_for_bdry_02.val,
                        &which_robin_coeff_for_bdry_03.val };

double *bc_coeff_all_mult[] = { &bdry_00_robin_coeff_multiplier.val,
                                &bdry_01_robin_coeff_multiplier.val,
                                &bdry_02_robin_coeff_multiplier.val,
                                &bdry_03_robin_coeff_multiplier.val };

int *bc_type_all[] = { &bdry_00_bc_type.val,
                       &bdry_01_bc_type.val,
                       &bdry_02_bc_type.val,
                       &bdry_03_bc_type.val };

bool *infc_present_all[] = { &is_infc_00_present.val,
                             &is_infc_01_present.val,
                             &is_infc_02_present.val,
                             &is_infc_03_present.val };

int *infc_geometry_all[] = { &which_geometry_for_infc_00.val,
                             &which_geometry_for_infc_01.val,
                             &which_geometry_for_infc_02.val,
                             &which_geometry_for_infc_03.val };

int *infc_opn_all[] = { &infc_00_opn.val,
                        &infc_01_opn.val,
                        &infc_02_opn.val,
                        &infc_03_opn.val };

int *jc_value_all[] = { &infc_00_value_jump.val,
                        &infc_01_value_jump.val,
                        &infc_02_value_jump.val,
                        &infc_03_value_jump.val };

int *jc_flux_all[] = { &infc_00_flux_jump.val,
                       &infc_01_flux_jump.val,
                       &infc_02_flux_jump.val,
                       &infc_03_flux_jump.val };

// BETA COEFFICIENTS
class mu_choices_cf_t : public CF_DIM
{
    int    *n;
    double *mag;
    cf_value_type_t what;
public:
    mu_choices_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override {
        switch (*n) {
            case 0: switch (what) {
                    case VAL: return (*mag);
                    case DDX:
                    case DDY: return 0.;
#ifdef P4_TO_P8
                        case DDZ: return 0.;
#endif
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 0 of class mu_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 1 of class mu_choices_cf_t");
                }
            case 2: switch (what) {
                    case VAL: return (*mag)*(y*y*log(x+2.) + 4.);
                    case DDX: return (*mag)*y*y/(x+2.);
                    case DDY: return (*mag)*2.*y*log(x+2.);
#ifdef P4_TO_P8
                        case DDZ: return 0.;
#endif
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 2 of class mu_choices_cf_t");
                }
            case 3: switch (what) {
                    case VAL: return  (*mag)*(exp(-x));
                    case DDX: return -(*mag)*(exp(-x));
                    case DDY: return  0;
#ifdef P4_TO_P8
                        case DDZ: return  0;
#endif
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 3 of class mu_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 4 of class mu_choices_cf_t");
                }
            }
            case 5: {
                XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
                YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
                switch (what) {
                    case VAL: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*log((x-0.5*(xmax.val+xmin.val))/(xmax.val- xmin.val)+2) +4;
                    case DDX: return (*mag)*SQR((y-ymin.val)/(ymax.val-ymin.val))*((+1)/(x-0.5*(xmax.val+xmin.val)+2*(xmax.val-xmin.val)));
                    case DDY: return (*mag)*(2.0*(y-ymin.val)/SQR(ymax.val-ymin.val))*log((x-0.5*(xmax.val+xmin.val))/(xmax.val- xmin.val)+2);
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 5 of class mu_choices_cf_t");
                }
            }//works
            case 6: {
                XCODE( double X = (x-xmin.val)/(xmax.val-xmin.val) );
                YCODE( double Y = (y-ymin.val)/(ymax.val-ymin.val) );
                switch (what) {
                    case VAL: return (*mag)*exp((-1)*(y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val));
                    case DDX: return 0.0;
                    case DDY: return (*mag)*exp((-1)*(y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((-1.0)/(ymax.val-ymin.val));
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 6 of class mu_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 7 of class mu_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 8 of class mu_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 9 of class mu_choices_cf_t");
                }
            }
#ifdef P4_TO_P8
            case 10: {
               switch (what) {
                    case VAL: return (*mag)*exp(-(z-zmin.val)/(zmax.val-zmin.val));
                    case DDX: return 0.0;
                    case DDY: return 0.0;
                    case DDZ: return (*mag)*exp(-(z-zmin.val)/(zmax.val-zmin.val))*(-1./(zmax.val-zmin.val));
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 10 of class mu_choices_cf_t");
               }
            }
#endif
            default:
                throw std::invalid_argument("Invalid case of class mu_choices_cf_t");
        }
    }
};

mu_choices_cf_t mu_m_cf(VAL, which_mu_m.val, mu_multiplier_m.val);
mu_choices_cf_t mu_p_cf(VAL, which_mu_p.val, mu_multiplier_p.val);
mu_choices_cf_t DIM(mux_m_cf(DDX, which_mu_m.val, mu_multiplier_m.val),
              muy_m_cf(DDY, which_mu_m.val, mu_multiplier_m.val),
              muz_m_cf(DDZ, diff_coeff_m.val, diff_coeff_m_mult.val));
mu_choices_cf_t DIM(mux_p_cf(DDX, which_mu_p.val, mu_multiplier_p.val),
              muy_p_cf(DDY, which_mu_p.val, mu_multiplier_p.val),
              muz_p_cf(DDZ, diff_coeff_p.val, diff_coeff_p_mult.val));


// EXACT SOLUTIONS
class h_choices_cf_t : public CF_DIM
{
public:
    int    *n;
    double *mag;
    cf_value_type_t what;
    h_choices_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 0 of class h_choices_cf_t");
                }

            default:
                throw std::invalid_argument("Invalid case of class h_choices_cf_t");
        }
    }
};


//nick add begin
//INITIAL (not divergence-free) FIELD :

class ustar_choices_cf_t : public CF_DIM
{
public:
    int    *n;
    double *mag;
    cf_value_type_t what;
    ustar_choices_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override
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
                    case VAL: return  (*mag)*sin(x)*cos(y) + x*(PI-x)* pow(y,2)*((y/3) - (PI/2));
                    case DDX: return  (*mag)*cos(x)*cos(y); //define later, lazy right now
                    case DDY: return -(*mag)*sin(x)*sin(y); //dllrn
                    case LAP: return -(*mag)*2.*sin(x)*cos(y); //dllrn
#endif
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 0 of class ustar_choices_cf_t");
                }

            default:
                throw std::invalid_argument("Invalid case of class ustar_choices_cf_t");
        }
    }
};

ustar_choices_cf_t ustar_m_cf(VAL, which_ustar_m.val, ustar_multiplier_m.val),  ustar_p_cf(VAL, which_ustar_p.val, ustar_multiplier_p.val);
ustar_choices_cf_t laplace_ustar_m_cf(LAP, which_ustar_m.val, ustar_multiplier_m.val), laplace_ustar_p_cf(LAP, which_ustar_p.val, ustar_multiplier_p.val);
ustar_choices_cf_t DIM(ustarx_m_cf(DDX, which_ustar_m.val, ustar_multiplier_m.val),
                   ustary_m_cf(DDY, which_ustar_m.val, ustar_multiplier_m.val),
                   ustarz_m_cf(DDZ, sol_id_m.val, sol_m_mult.val));
ustar_choices_cf_t DIM(ustarx_p_cf(DDX, which_ustar_p.val, ustar_multiplier_p.val),
                   ustary_p_cf(DDY, which_ustar_p.val, ustar_multiplier_p.val),
                   ustarz_p_cf(DDZ, sol_p.val, sol_p_mult.val)); //need to change? what does this one do
//nick add end

h_choices_cf_t h_m_cf(VAL, which_h_m.val, h_multiplier_m.val),  h_p_cf(VAL, which_h_p.val, h_multiplier_p.val);
h_choices_cf_t laplace_h_m_cf(LAP, which_h_m.val, h_multiplier_m.val), laplace_h_p_cf(LAP, which_h_p.val, h_multiplier_p.val);
h_choices_cf_t DIM(hx_m_cf(DDX, which_h_m.val, h_multiplier_m.val),
             hy_m_cf(DDY, which_h_m.val, h_multiplier_m.val),
             hz_m_cf(DDZ, sol_id_m.val, sol_m_mult.val));
h_choices_cf_t DIM(hx_p_cf(DDX, which_h_p.val, h_multiplier_p.val),
             hy_p_cf(DDY, which_h_p.val, h_multiplier_p.val),
             hz_p_cf(DDZ, sol_p.val, sol_p_mult.val));





// ALPHA COEFFICIENT
class lambda_choices_cf_t: public CF_DIM
{
    int *n;
    double *mag;
public:
    lambda_choices_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override {
        switch (*n) {
            case 0: return 0.;
            case 1: return (*mag)*1.;
            case 2:
#ifdef P4_TO_P8
                return (*mag)*cos(x+z)*exp(y);
#else
                return (*mag)*cos(x)*exp(y);
#endif
            default:
                throw std::invalid_argument("Invalid case of class lambda_choices_cf_t");
        }
    }
};

lambda_choices_cf_t lambda_cf_m(which_lambda_m.val, lambda_multiplier_m.val);
lambda_choices_cf_t lambda_cf_p(which_lambda_p.val, lambda_multiplier_p.val);

// NONLINEAR TERM FUNCTIONAL FORM, I.E., Fh
class f_choices_cf_t: public CF_1
{
    int *n;
    cf_value_type_t what;
public:
    f_choices_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
    double operator()(double h) const override {
        switch (*n) {
            case 0:
                switch (what) {
                    case VAL:
                    case DDX: return 0.;
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 0 of class f_choices_cf_t");
                }
            case 1:
                switch (what) {
                    case VAL: return h;
                    case DDX: return 1.;
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 1 of class f_choices_cf_t");
                }
            case 2:
                switch (what) {
                    case VAL: return sinh(h);
                    case DDX: return cosh(h);
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 2 of class f_choices_cf_t");                }
            case 3:
                switch (what) {
                    case VAL: return h/(1.+h);
                    case DDX: return 1./SQR(1.+h);
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 3 of class f_choices_cf_t");                }
            default:
                throw std::invalid_argument("Invalid case of class f_choices_cf_t");
        }
    }
};

f_choices_cf_t f_cf_m(VAL, which_f_m.val), f_prime_cf_m(DDX, which_f_m.val);
f_choices_cf_t f_cf_p(VAL, which_f_p.val), f_prime_cf_p(DDX, which_f_p.val);

// GAMMA COEFFICIENT
class nu_choices_cf_t: public CF_DIM
{
    int *n;
    double *mag;
public:
    nu_choices_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override {
        switch (*n) {
            case 0: return (*mag)*1.;
            case 1:
#ifdef P4_TO_P8
                return (*mag)*cos(x+z)*exp(y);
#else
                return (*mag)*cos(x)*exp(y);
#endif
            default:
                throw std::invalid_argument("Invalid case of class nu_choices_cf_t");
        }
    }
};

nu_choices_cf_t nu_cf_m(which_nu_m.val, nu_multiplier_m.val);
nu_choices_cf_t nu_cf_p(which_nu_p.val, nu_multiplier_p.val);


// RHS
class rhs_m_cf_t: public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override {
        switch (rhs_m_value.val)
        {
            case 0:
                return lambda_cf_m(DIM(x, y, z)) * h_m_cf(DIM(x, y, z))
                       + nu_cf_m(DIM(x, y, z)) * f_cf_m(h_m_cf(DIM(x, y, z)))
                       - mu_m_cf(DIM(x, y, z)) * laplace_h_m_cf(DIM(x, y, z))
                       - SUMD(mux_m_cf(DIM(x,y,z))*hx_m_cf(DIM(x,y,z)),
                              muy_m_cf(DIM(x,y,z))*hy_m_cf(DIM(x,y,z)),
                              muz_m_cf(DIM(x,y,z))*hz_m_cf(DIM(x,y,z)));
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
                auto min_dist = DBL_MAX;
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
                throw std::invalid_argument("Invalid case of class rhs_m_cf_t");
        }
    }
} rhs_m_cf;

class rhs_p_cf_t: public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override {
        switch (rhs_p_value.val)
        {
            case 0:
                return 0 ;
            case 1:
                return 0;
            default:
                throw std::invalid_argument("Invalid case of class rhs_p_cf_t");
        }
    }
} rhs_p_cf;

// ROBIN BOUNDARY CONDITION COEFFICIENT
class robin_bc_coeff_choices_cf_t: public CF_DIM
{
    int    *n;
    double *mag;
public:
    robin_bc_coeff_choices_cf_t(int &n, double &mag) : n(&n), mag(&mag) {}
    double operator()(DIM(double x, double y, double z)) const override {
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
            default:
                throw std::invalid_argument("Invalid case of class robin_bc_coeff_choices_cf_t");
        }
    }
};

robin_bc_coeff_choices_cf_t bc_coeff_cf_all[] = {robin_bc_coeff_choices_cf_t(which_robin_coeff_for_bdry_00.val, bdry_00_robin_coeff_multiplier.val),
                                                 robin_bc_coeff_choices_cf_t(which_robin_coeff_for_bdry_01.val, bdry_01_robin_coeff_multiplier.val),
                                                 robin_bc_coeff_choices_cf_t(which_robin_coeff_for_bdry_02.val, bdry_02_robin_coeff_multiplier.val),
                                                 robin_bc_coeff_choices_cf_t(which_robin_coeff_for_bdry_03.val, bdry_03_robin_coeff_multiplier.val) };

// DOMAIN GEOMETRY
class bdry_phi_choices_cf_t: public CF_DIM {
public:
    int *n; // geometry number
    cf_value_type_t what;
    bdry_phi_choices_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
    double operator()(DIM(double x, double y, double z)) const override {
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 1 of class bdry_phi_choices_cf_t");
                }
            }
            break;

            case 2: // circle/sphere exterior
            {
                static double r0 = 0.311, DIM(xc = 0, yc = 0, zc = 0);
                static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0, -1);
                switch (what) {
                    _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
                    XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
                    YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
                    ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 2 of class bdry_phi_choices_cf_t");
                }
            }
            break;

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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 3 of class bdry_phi_choices_cf_t");
                }
            }
            break;

            case 4: // moderately star-shaped domain
            {
                static double r0 = 0.611, DIM(xc = 0, yc = 0, zc = 0), deform = 0.15;
                static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform);
                switch (what) {
                    _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
                    XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
                    YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
                    ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 4 of class bdry_phi_choices_cf_t");
                }
            }
            break;

            case 5: // highly start-shaped domain
            {
                static double r0 = 0.6037, DIM(xc = 0, yc = 0, zc = 0), deform = 0.3;
                static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), deform, 1, 0.5);
                switch (what) {
                    _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
                    XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
                    YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
                    ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 5 of class bdry_phi_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 6/7 of class bdry_phi_choices_cf_t");
                }
            }
            break;

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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 8/9 of class bdry_phi_choices_cf_t");
                }
            }
            break;

            case 10: // three star-shaped domains: 1st domain
            case 11: // three star-shaped domains: 2nd domain
            case 12: // three star-shaped domains: 3rd domain
            {
#ifdef P4_TO_P8
                static double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, zc0 = 0.19, nx0 = 1.0, ny0 = 1.0, nz0 = 1.0, theta0 = 0.3*PI, mu0 = 0.08, inside0 = 1;
                static double r1 = 0.66, xc1 =-0.21, yc1 =-0.23, zc1 =-0.17, nx1 = 1.0, ny1 = 1.0, nz1 = 1.0, theta1 =-0.3*PI, mu1 =-0.08, inside1 = 1;
                static double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, zc2 = 0.03, nx2 =-1.0, ny2 = 1.0, nz2 = 0.0, theta2 =-0.2*PI, mu2 =-0.08, inside2 =-1;

                static flower_shaped_domain_t shape0(r0, xc0, yc0, zc0, mu0, inside0, nx0, ny0, nz0, theta0);
                static flower_shaped_domain_t shape1(r1, xc1, yc1, zc1, mu1, inside1, nx1, ny1, nz1, theta1);
                static flower_shaped_domain_t shape2(r2, xc2, yc2, zc2, mu2, inside2, nx2, ny2, nz2, theta2);
#else
                static double r0 = 0.73, xc0 = 0.13, yc0 = 0.16, theta0 = 0.1*PI, mu0 = 0.08, inside0 = 1;
                static double r1 = 0.66, xc1 =-0.14, yc1 =-0.21, theta1 =-0.2*PI, mu1 =-0.08, inside1 = 1;
                static double r2 = 0.59, xc2 = 0.45, yc2 =-0.53, theta2 = 0.2*PI, mu2 =-0.08, inside2 =-1;

                static flower_shaped_domain_t shape0(r0, xc0, yc0, mu0, inside0, theta0);
                static flower_shaped_domain_t shape1(r1, xc1, yc1, mu1, inside1, theta1);
                static flower_shaped_domain_t shape2(r2, xc2, yc2, mu2, inside2, theta2);
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 10/11/12 of class bdry_phi_choices_cf_t");
                }
            }
            break;

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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 13/14/15/16 of class bdry_phi_choices_cf_t");
                }
            }
            break;
        }

        // default values
        switch (what) {
            _CODE( case VAL: return -1 );
            XCODE( case DDX: return  0 );
            YCODE( case DDY: return  0 );
            ZCODE( case DDZ: return  0 );
            default:
                throw std::invalid_argument("Invalid \"what\" in default values of class bdry_phi_choices_cf_t");
        }
    }
};

bdry_phi_choices_cf_t bdry_phi_cf_all  [] = { bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_00.val),
                                      bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_01.val),
                                      bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_02.val),
                                      bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_03.val) };

bdry_phi_choices_cf_t bdry_phi_x_cf_all[] = { bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_00.val),
                                      bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_01.val),
                                      bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_02.val),
                                      bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_03.val) };

bdry_phi_choices_cf_t bdry_phi_y_cf_all[] = { bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_00.val),
                                      bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_01.val),
                                      bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_02.val),
                                      bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_03.val) };
#ifdef P4_TO_P8
bdry_phi_choices_cf_t bdry_phi_z_cf_all[] = { bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_00.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_01.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_02.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_03.val) };
#endif

// INTERFACE GEOMETRY
class infc_phi_choices_cf_t: public CF_DIM {
public:
    int    *n;
    cf_value_type_t what;
    infc_phi_choices_cf_t(cf_value_type_t what, int &n) : what(what), n(&n) {}
    double operator()(DIM(double x, double y, double z)) const override {
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 1 of class infc_phi_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 2 of class infc_phi_choices_cf_t");
                }
            }
            case 3: // highly star-shaped domain
            {
//                static double r0 = 0.533, DIM( xc = 0, yc = 0, zc = 0 );
//                static flower_shaped_domain_t circle(r0, DIM(xc, yc, zc), 0.3, -1);
//                switch (what) {
//                  _CODE( case VAL: return circle.phi  (DIM(x,y,z)) );
//                  XCODE( case DDX: return circle.phi_x(DIM(x,y,z)) );
//                  YCODE( case DDY: return circle.phi_y(DIM(x,y,z)) );
//                  ZCODE( case DDZ: return circle.phi_z(DIM(x,y,z)) );
//                }
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 3 of class infc_phi_choices_cf_t");
                }
            }
            case 4: // asymmetric curvy domain
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 4 of class infc_phi_choices_cf_t");
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
                auto min_dist = DBL_MAX;
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 5 of class infc_phi_choices_cf_t");
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
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 6 of class infc_phi_choices_cf_t");
                }
            }
            case 10: // intersection of two circles
            {
                static double r0 = 0.4;
                static double r1 = 0.5;
                static double DIM( xc0 = 0.25, yc0 = 0.30, zc0 = 0.35 );
                static double DIM( xc1 =-0.15, yc1 =-0.10, zc1 =-0.05 );
                static flower_shaped_domain_t circle0(r0, DIM(xc0, yc0, zc0), 0, -1);
                static flower_shaped_domain_t circle1(r1, DIM(xc1, yc1, zc1), 0,  1);

                double val0 = circle0.phi(DIM(x,y,z));
                double val1 = circle1.phi(DIM(x,y,z));

                switch (what) {
                    _CODE( case VAL: return MAX(val0, val1) );
                    XCODE( case DDX: return (fabs(val0) < fabs(val1) ? circle0.phi_x(DIM(x,y,z)) : circle1.phi_x(DIM(x,y,z))) );
                    YCODE( case DDY: return (fabs(val0) < fabs(val1) ? circle0.phi_y(DIM(x,y,z)) : circle1.phi_y(DIM(x,y,z)))  );
                    ZCODE( case DDZ: return (fabs(val0) < fabs(val1) ? circle0.phi_z(DIM(x,y,z)) : circle1.phi_z(DIM(x,y,z)))  );
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 10 of class infc_phi_choices_cf_t");
                }
            }
            case 11: // union of two circles
            {
                static double r0 = 0.4;
                static double r1 = 0.5;
                static double DIM( xc0 = 0.25, yc0 = 0.30, zc0 = 0.35 );
                static double DIM( xc1 =-0.15, yc1 =-0.10, zc1 =-0.05 );
                static flower_shaped_domain_t circle0(r0, DIM(xc0, yc0, zc0), 0, 1);
                static flower_shaped_domain_t circle1(r1, DIM(xc1, yc1, zc1), 0, 1);

                double val0 = circle0.phi(DIM(x,y,z));
                double val1 = circle1.phi(DIM(x,y,z));

                switch (what) {
                    _CODE( case VAL: return MIN(val0, val1) );
                    XCODE( case DDX: return (fabs(val0) < fabs(val1) ? circle0.phi_x(DIM(x,y,z)) : circle1.phi_x(DIM(x,y,z))) );
                    YCODE( case DDY: return (fabs(val0) < fabs(val1) ? circle0.phi_y(DIM(x,y,z)) : circle1.phi_y(DIM(x,y,z)))  );
                    ZCODE( case DDZ: return (fabs(val0) < fabs(val1) ? circle0.phi_z(DIM(x,y,z)) : circle1.phi_z(DIM(x,y,z)))  );
                    default:
                        throw std::invalid_argument("Invalid \"what\" in case 11 of class infc_phi_choices_cf_t");
                }
            }
            default:
                throw std::invalid_argument("Invalid case of class infc_phi_choices_cf_t");
        }
    }
};


infc_phi_choices_cf_t infc_phi_cf_all  [] = { infc_phi_choices_cf_t(VAL, which_geometry_for_infc_00.val),
                                      infc_phi_choices_cf_t(VAL, which_geometry_for_infc_01.val),
                                      infc_phi_choices_cf_t(VAL, which_geometry_for_infc_02.val),
                                      infc_phi_choices_cf_t(VAL, which_geometry_for_infc_03.val) };

infc_phi_choices_cf_t infc_phi_x_cf_all[] = { infc_phi_choices_cf_t(DDX, which_geometry_for_infc_00.val),
                                      infc_phi_choices_cf_t(DDX, which_geometry_for_infc_01.val),
                                      infc_phi_choices_cf_t(DDX, which_geometry_for_infc_02.val),
                                      infc_phi_choices_cf_t(DDX, which_geometry_for_infc_03.val) };

infc_phi_choices_cf_t infc_phi_y_cf_all[] = { infc_phi_choices_cf_t(DDY, which_geometry_for_infc_00.val),
                                      infc_phi_choices_cf_t(DDY, which_geometry_for_infc_01.val),
                                      infc_phi_choices_cf_t(DDY, which_geometry_for_infc_02.val),
                                      infc_phi_choices_cf_t(DDY, which_geometry_for_infc_03.val) };
#ifdef P4_TO_P8
infc_phi_choices_cf_t infc_phi_z_cf_all[] = { infc_phi_choices_cf_t(DDZ, which_geometry_for_infc_00.val),
                                      infc_phi_choices_cf_t(DDZ, which_geometry_for_infc_01.val),
                                      infc_phi_choices_cf_t(DDZ, which_geometry_for_infc_02.val),
                                      infc_phi_choices_cf_t(DDZ, which_geometry_for_infc_03.val) };
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
    double operator()(DIM(double x, double y, double z)) const override
    {
        return MAX( (*bdry_phi_cf_)(DIM(x,y,z)), -fabs((*infc_phi_cf_)(DIM(x,y,z))) );
    }
} phi_eff_cf(bdry_phi_eff_cf, infc_phi_eff_cf);

/*
 * Naming:
 * h_choices_cf_t defines cases for u to choose from. This will be used to define a solution inside and
 * a solution outside
 * h_cf_t gives u everywhere by checking the sign of the level-set.
 * Should we have h_cf_ and h_effective_cf_t or h_global_cf_t?
 * Maybe that's why we had linear coeff, nonlinear coeff, but too confusing.
 * Perhaps h_setup_cf_t and h_cf_t, mu_setup_cf_t and mu_choices_cf_t, etc.
 *         h_setup_cf_t, h_choice_cf_t, h_case_cf_t, choice_of_h_cf_t, which_h_cf_t
 *
 *         which_h_cf_t, which_lambda_choices_cf_t, which_mu_choices_cf_t, which_nu_choices_cf_t, which_f_cf_t
 *
 */


class mu_cf_t : public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override {
        return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? mu_p_cf(DIM(x, y, z)) : mu_m_cf(DIM(x, y, z));
    }
} mu_cf;
class h_cf_t : public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override {
        return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? h_p_cf(DIM(x,y,z)) : h_m_cf(DIM(x,y,z));
    }
} h_cf;
class ux_cf_t : public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override  {
        return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? hx_p_cf(DIM(x,y,z)) : hx_m_cf(DIM(x,y,z));
    }
} ux_cf;
class uy_cf_t : public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const override {
        return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? hy_p_cf(DIM(x,y,z)) : hy_m_cf(DIM(x,y,z));
    }
} uy_cf;
#ifdef P4_TO_P8
class uz_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const override {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? uz_p_cf(DIM(x,y,z)) : uz_m_cf(DIM(x,y,z));
  }
} uz_cf;
#endif

// BC VALUES
class bc_value_cf_t : public CF_DIM
{
    BoundaryConditionType *bc_type_;
    CF_DIM DIM(*phi_x_cf_,
               *phi_y_cf_,
               *phi_z_cf_);
    CF_DIM *bc_coeff_cf_;
public:
    bc_value_cf_t(BoundaryConditionType *bc_type,
                  CF_DIM *bc_coeff_cf,
                  DIM(CF_DIM *phi_x_cf,
                         CF_DIM *phi_y_cf,
                         CF_DIM *phi_z_cf))
            : bc_type_(bc_type),
              bc_coeff_cf_(bc_coeff_cf),
              DIM(phi_x_cf_(phi_x_cf),
                  phi_y_cf_(phi_y_cf),
                  phi_z_cf_(phi_z_cf)) {}
    double operator()(DIM(double x, double y, double z)) const override
    {
        switch (*bc_type_) {
            case DIRICHLET: return h_cf(DIM(x,y,z));
            case NEUMANN: {
                double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                            ny = (*phi_y_cf_)(DIM(x,y,z)),
                            nz = (*phi_z_cf_)(DIM(x,y,z)) );

                double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
                nx /= norm; ny /= norm; P8( nz /= norm );

                return mu_cf(DIM(x, y, z)) * SUMD(nx * ux_cf(DIM(x, y, z)),
                                              ny*uy_cf(DIM(x,y,z)),
                                              nz*uz_cf(DIM(x,y,z)));
            }
            case ROBIN: {
                double DIM( nx = (*phi_x_cf_)(DIM(x,y,z)),
                            ny = (*phi_y_cf_)(DIM(x,y,z)),
                            nz = (*phi_z_cf_)(DIM(x,y,z)) );

                double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
                nx /= norm; ny /= norm; CODE3D( nz /= norm );

                return mu_cf(DIM(x, y, z)) * SUMD(nx * ux_cf(DIM(x, y, z)),
                                              ny*uy_cf(DIM(x,y,z)),
                                              nz*uz_cf(DIM(x,y,z))) + (*bc_coeff_cf_)(DIM(x,y,z))*h_cf(DIM(x,y,z));
            }
            default:
                throw std::invalid_argument("Invalid case of class bc_value_cf_t");
        }
    }
};

bc_value_cf_t bc_value_cf_all[] = {bc_value_cf_t((BoundaryConditionType *)&bdry_00_bc_type.val, &bc_coeff_cf_all[0], DIM(&bdry_phi_x_cf_all[0], &bdry_phi_y_cf_all[0], &bdry_phi_z_cf_all[0])),
                                   bc_value_cf_t((BoundaryConditionType *)&bdry_01_bc_type.val, &bc_coeff_cf_all[1], DIM(&bdry_phi_x_cf_all[1], &bdry_phi_y_cf_all[1], &bdry_phi_z_cf_all[1])),
                                   bc_value_cf_t((BoundaryConditionType *)&bdry_02_bc_type.val, &bc_coeff_cf_all[2], DIM(&bdry_phi_x_cf_all[2], &bdry_phi_y_cf_all[2], &bdry_phi_z_cf_all[2])),
                                   bc_value_cf_t((BoundaryConditionType *)&bdry_03_bc_type.val, &bc_coeff_cf_all[3], DIM(&bdry_phi_x_cf_all[3], &bdry_phi_y_cf_all[3], &bdry_phi_z_cf_all[3])) };

// JUMP CONDITIONS
class jc_value_cf_t : public CF_DIM
{
    int    *n;
public:
    explicit jc_value_cf_t(int &n) : n(&n) {}
    double operator()(DIM(double x, double y, double z)) const override
    {
        switch(*n) {
            case 0: return h_p_cf(DIM(x,y,z)) - h_m_cf(DIM(x,y,z));
            case 1: return 0;
            default:
                throw std::invalid_argument("Invalid case of class jc_value_cf_t");

        }
    }
};

jc_value_cf_t jc_value_cf_all[] = { jc_value_cf_t(infc_00_value_jump.val),
                                    jc_value_cf_t(infc_01_value_jump.val),
                                    jc_value_cf_t(infc_02_value_jump.val),
                                    jc_value_cf_t(infc_03_value_jump.val) };

jc_value_cf_t jc_value_cf(infc_00_value_jump.val);

class jc_flux_cf_t : public CF_DIM
{
    int    *n;
    CF_DIM *phi_x_;
    CF_DIM *phi_y_;
    CF_DIM *phi_z_;
public:
    jc_flux_cf_t(int &n, DIM(CF_DIM * phi_x, CF_DIM * phi_y, CF_DIM * phi_z)) :
            n(&n), DIM(phi_x_(phi_x), phi_y_(phi_y), phi_z_(phi_z)) {}

    double operator()(DIM(double x, double y, double z)) const override
    {
        switch(*n) {
            case 0:
            {
                double DIM( nx = (*phi_x_)(DIM(x,y,z)),
                            ny = (*phi_y_)(DIM(x,y,z)),
                            nz = (*phi_z_)(DIM(x,y,z)) );
                double norm = sqrt(SUMD(nx*nx, ny*ny, nz*nz));
                nx /= norm; ny /= norm; CODE3D( nz /= norm; )

                return mu_p_cf(DIM(x, y, z)) * SUMD(hx_p_cf(DIM(x, y, z)) * nx, hy_p_cf(DIM(x, y, z)) * ny, hz_p_cf(DIM(x, y, z)) * nz)
                     - mu_m_cf(DIM(x, y, z)) * SUMD(hx_m_cf(DIM(x, y, z)) * nx, hy_m_cf(DIM(x, y, z)) * ny, hz_m_cf(DIM(x, y, z)) * nz);
            }
            default: return 0;
        }
    }
};

jc_flux_cf_t jc_flux_cf_all[] = {jc_flux_cf_t(infc_00_flux_jump.val, DIM(&infc_phi_x_cf_all[0], &infc_phi_y_cf_all[0], &infc_phi_z_cf_all[0])),
                                 jc_flux_cf_t(infc_01_flux_jump.val, DIM(&infc_phi_x_cf_all[1], &infc_phi_y_cf_all[1], &infc_phi_z_cf_all[1])),
                                 jc_flux_cf_t(infc_02_flux_jump.val, DIM(&infc_phi_x_cf_all[2], &infc_phi_y_cf_all[2], &infc_phi_z_cf_all[2])),
                                 jc_flux_cf_t(infc_03_flux_jump.val, DIM(&infc_phi_x_cf_all[3], &infc_phi_y_cf_all[3], &infc_phi_z_cf_all[3])) };


class bc_wall_type_t : public WallBCDIM
{
public:
    BoundaryConditionType operator()(DIM(double, double, double)) const override
    {
        return (BoundaryConditionType) wc_type.val;
    }
} bc_wall_type;

class perturb_cf_t: public CF_DIM
{
    int *n;
public:
    explicit perturb_cf_t(int &n) : n(&n) {}
    double operator()(DIM(double x, double y, double z)) const override
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
            default:
                throw std::invalid_argument("Invalid case of class perturb_cf_t");
        }
    }
} bdry_perturb_cf(bdry_perturb.val), infc_perturb_cf(infc_perturb.val);

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

    example.set_from_cmd(cmd);
    set_example(example.val);

    pl.set_from_cmd_all(cmd);

    if (mpi.rank() == 0) pl.print_all();
    if (mpi.rank() == 0 && save_params.val) {
        std::ostringstream file;
        file << out_dir << "/parameters.dat";
        pl.save_all(file.str().c_str());
    }
    // num_bdry_max -> defines the maximum number of boundaries that exist in this example. num_infc_max -> defines the maximum number of interfaces that exist in this example.
    // The way it is implemented for now, there can be at most 4 interfaces and 4 boundaries.
    // initialize effective level-sets
    // To add boundaries or interfaces - objects of mls_eff_cf_t -> bdry_phi_eff_cf and infc_phi_eff_cf are created. mls_eff_cf_t is a class that inherits CF_DIM ( CF_2 or CF_3 depending on whether the problem is 2D or 3D )
    // The above-mentioned class has an add_domain() which has two arguments namely a level set function representing the boundary (that is stored in bdry_phi_cf_all[i]) and the action (which can either  be MLS_INTERSECTION OR MLS_ADDITION)
    // As evident from the name choosing MLS_INTERSECTION will create a compound domain that is the intersection of two or more given domains and MLS_ADDITION will create a compound domain that is the union of two or more given domains


    //add boundaries
    for (int i = 0; i < num_bdry_max; ++i)
    {
        if (*bdry_present_all[i] == true) bdry_phi_eff_cf.add_domain(bdry_phi_cf_all[i], (mls_opn_t) *bdry_opn_all[i]);
    }

    //add interfaces
    for (int i = 0; i < num_infc_max; ++i)
    {
        if (*infc_present_all[i]) infc_phi_eff_cf.add_domain(infc_phi_cf_all[i], (mls_opn_t) *infc_opn_all[i]);
    }


    int num_shifts_total = MULTD(num_shifts_x_dir.val, num_shifts_y_dir.val, num_shifts_z_dir.val);

    int num_resolutions = ((num_splits.val-1)*add_splits.val + 1) * mu_iter_num.val;
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
    Engine *mengine = nullptr;
    if (mpi.rank() == 0 && compute_cond_num())
    {
      mengine = engOpen("matlab -nodisplay -nojvm");
      if (mengine == nullptr) throw std::runtime_error("Cannot start a MATLAB Engine session.\n");
    }
#else
    if (compute_cond_num())
    {
        ierr = PetscPrintf(mpi.comm(), "[Warning]: MATLAB is either not provided or not found. Condition numbers will not be computed. \n");
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



    for(int mu_iter = 0; mu_iter < mu_iter_num(); ++mu_iter)
    {
        if (mu_iter_num() > 1)
            mu_multiplier_m.val = mu_multiplier_min_m() * pow(mu_multiplier_max_m() / mu_multiplier_min_m(), ((double) mu_iter / (double) (mu_iter_num() - 1)));

        for(int iter=0; iter<num_splits(); ++iter)
        {
            ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin()+iter, lmax()+iter); CHKERRXX(ierr);

            int num_sub_iter = (iter == 0 ? 1 : add_splits());

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

                mu_arr.push_back(mu_multiplier_m());
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
                        p4est = my_p4est_new(mpi.comm(), connectivity, 0, nullptr, nullptr);

                        if (refine_strict())
                        {
                            splitting_criteria_cf_t data_tmp(lmin(), lmax(), &phi_eff_cf, lip(), band());
                            p4est->user_pointer = (void*)(&data_tmp);

                            my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, nullptr);
                            my_p4est_partition(p4est, P4EST_FALSE, nullptr);
                            for (int i = 0; i < iter; ++i)
                            {
                                my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, nullptr);
                                my_p4est_partition(p4est, P4EST_FALSE, nullptr);
                            }
                        } else {
                            splitting_criteria_cf_t data_tmp(lmin()+iter, lmax()+iter, &phi_eff_cf, lip(), band());
                            p4est->user_pointer = (void*)(&data_tmp);

                            for (int i = 0; i < lmax()+iter; ++i)
                            {
                                my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, nullptr);
                                my_p4est_partition(p4est, P4EST_FALSE, nullptr);
                            }
                        }
                        // macromesh has been generated at this point.

                        splitting_criteria_cf_t data(lmin()+iter, lmax()+iter, &phi_eff_cf, lip(), band());
                        p4est->user_pointer = (void*)(&data);

                        if (refine_rand())
                            my_p4est_refine(p4est, P4EST_TRUE, refine_random, nullptr);

                        if (balance_grid())
                        {
                            my_p4est_partition(p4est, P4EST_FALSE, nullptr);
                            // Balance type (face or corner/full).
                            // Corner balance is almost never required when discretizing a PDE; just causes smoother mesh grading.
                            p4est_balance(p4est, P4EST_CONNECT_FULL, nullptr);
                            my_p4est_partition(p4est, P4EST_FALSE, nullptr);
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
                        Vec bdry_phi_vec_all[num_bdry_max];
                        Vec infc_phi_vec_all[num_infc_max];


                        //Perturbing domain and interface boundaries and reinitializing
                        for (int i = 0; i < num_bdry_max; ++i)
                        {
                            if (*bdry_present_all[i])
                            {
                                ierr = VecCreateGhostNodes(p4est, nodes, &bdry_phi_vec_all[i]); CHKERRXX(ierr);
                                sample_cf_on_nodes(p4est, nodes, bdry_phi_cf_all[i], bdry_phi_vec_all[i]);

                                if (bdry_perturb())
                                {
                                    double *phi_ptr;
                                    ierr = VecGetArray(bdry_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                                    {
                                        double xyz[P4EST_DIM];
                                        node_xyz_fr_n(n, p4est, nodes, xyz);
                                        phi_ptr[n] += bdry_perturb_mag()*bdry_perturb_cf.value(xyz)*pow(dxyz_m, bdry_perturb_pow());
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

                        for (int i = 0; i < num_infc_max; ++i)
                        {
                            if (*infc_present_all[i])
                            {
                                ierr = VecCreateGhostNodes(p4est, nodes, &infc_phi_vec_all[i]); CHKERRXX(ierr);
                                sample_cf_on_nodes(p4est, nodes, infc_phi_cf_all[i], infc_phi_vec_all[i]);

                                if (infc_perturb())
                                {
                                    double *phi_ptr;
                                    ierr = VecGetArray(infc_phi_vec_all[i], &phi_ptr); CHKERRXX(ierr);

                                    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                                    {
                                        double xyz[P4EST_DIM];
                                        node_xyz_fr_n(n, p4est, nodes, xyz);
                                        phi_ptr[n] += infc_perturb_mag()*infc_perturb_cf.value(xyz)*pow(dxyz_m, infc_perturb_pow());
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
                        // mesh has been set up

                        //initializing vectors needed to setup the problem
                        Vec rhs_m;
                        ierr = VecCreateGhostNodes(p4est, nodes, &rhs_m); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, rhs_m_cf, rhs_m);

                        Vec rhs_p;
                        ierr = VecCreateGhostNodes(p4est, nodes, &rhs_p); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, rhs_p_cf, rhs_p);

                        //nick add: initializing the vector for our unprojected velocty field u*
                        Vec ustar_p;
                        ierr = VecCreateGhostNodes(p4est, nodes, &ustar_p); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, ustar_p_cf, ustar_p);

                        Vec ustar_m;
                        ierr = VecCreateGhostNodes(p4est, nodes, &ustar_m); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, ustar_p_cf, ustar_m);
                        //end nick add

                        Vec mu_m;
                        ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);

                        Vec mu_p;
                        ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

                        Vec linear_term_m_coeff;
                        ierr = VecCreateGhostNodes(p4est, nodes, &linear_term_m_coeff); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, lambda_cf_m, linear_term_m_coeff);

                        Vec linear_term_p_coeff;
                        ierr = VecCreateGhostNodes(p4est, nodes, &linear_term_p_coeff); CHKERRXX(ierr);
                        sample_cf_on_nodes(p4est, nodes, lambda_cf_p, linear_term_p_coeff);

                        Vec sol; double *sol_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);

                        //creating an object of the poisson solver which has to be setup
                        my_p4est_poisson_nodes_mls_t solver(&ngbd_n);

                        solver.set_use_centroid_always(use_centroid_always());
                        solver.set_store_finite_volumes(store_finite_volumes());
                        solver.set_jump_scheme(jump_scheme());
                        solver.set_use_sc_scheme(fv_scheme());
                        solver.set_integration_order(integration_order());
                        solver.set_lip(lip());

                        // HOW TO ADD BOUNDARY
                        // In the add_boundary(), creates a structure boundary_conditions_t
                        for (int i = 0; i < num_bdry_max; ++i)
                            if (*bdry_present_all[i])
                            {
                                if (apply_bc_pointwise())
                                    solver.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i], DIM(nullptr, nullptr, nullptr), (BoundaryConditionType) *bc_type_all[i], zero_cf, zero_cf);
                                else
                                    solver.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i], DIM(nullptr, nullptr, nullptr), (BoundaryConditionType) *bc_type_all[i], bc_value_cf_all[i], bc_coeff_cf_all[i]);
                                std::cout << (BoundaryConditionType) *bc_type_all[i] << std::endl;
                            }

                        // adding interface is exactly similar
                        for (int i = 0; i < num_infc_max; ++i)
                            if (*infc_present_all[i])
                            {
                                if (apply_bc_pointwise())
                                    solver.add_interface((mls_opn_t) *infc_opn_all[i], infc_phi_vec_all[i], DIM(nullptr, nullptr, nullptr), zero_cf, zero_cf);
                                else
                                    solver.add_interface((mls_opn_t) *infc_opn_all[i], infc_phi_vec_all[i], DIM(nullptr, nullptr, nullptr), jc_value_cf_all[i], jc_flux_cf_all[i]);
                            }

                        solver.set_mu(mu_m, DIM(nullptr, nullptr, nullptr),
                                      mu_p, DIM(nullptr, nullptr, nullptr));

                        solver.set_wc(bc_wall_type, h_cf);

                        //NICK ADD BEGIN

                        for (int i = 0; i < p4est.; ++i)





                        //NICK ADD END

                        solver.set_rhs(rhs_m, rhs_p);
                        solver.set_diag(linear_term_m_coeff, linear_term_p_coeff);

                        solver.set_use_taylor_correction(taylor_correction());
                        solver.set_kink_treatment(kink_special_treatment());
//              solver.set_dirichlet_scheme(1);
//              solver.set_gf_order(3);
//              solver.set_gf_thresh(-0.1);
//              solver.set_gf_stabilized(0);


                        vector< vector<double> > pw_bc_values(num_bdry());
                        vector< vector<double> > pw_bc_values_robin(num_bdry());
                        vector< vector<double> > pw_bc_coeffs_robin(num_bdry());

                        vector< vector<double> > pw_jc_sol_jump_taylor(num_infc());
                        vector< vector<double> > pw_jc_flx_jump_taylor(num_infc());
                        vector< vector<double> > pw_jc_flx_jump_integr(num_infc());

                        if (apply_bc_pointwise())
                        {
                            solver.preassemble_linear_system();

                            // allocate memory for bc values
                            for (int i = 0; i < num_bdry(); ++i)
                            {
                                pw_bc_values      [i].assign(solver.pw_bc_num_value_pts(i), 0);
                                pw_bc_values_robin[i].assign(solver.pw_bc_num_robin_pts(i), 0);
                                pw_bc_coeffs_robin[i].assign(solver.pw_bc_num_robin_pts(i), 0);
                            }

                            for (int i = 0; i < num_infc(); ++i)
                            {
                                pw_jc_sol_jump_taylor[i].assign(solver.pw_jc_num_taylor_pts(i), 0);
                                pw_jc_flx_jump_taylor[i].assign(solver.pw_jc_num_taylor_pts(i), 0);
                                pw_jc_flx_jump_integr[i].assign(solver.pw_jc_num_integr_pts(i), 0);
                            }

                            double xyz[P4EST_DIM];
                            // sample bc and jc at requested points
                            if (sample_bc_node_by_node())
                            {
                                foreach_local_node(n, nodes)
                                {
                                    for (int i = 0; i < num_bdry(); ++i)
                                    {
                                        //DIRICHLET & NEUMANN BOUNDARY CONDITIONS
                                        for (int k = 0; k < solver.pw_bc_num_value_pts(i,n); ++k)
                                        {
                                            int j = solver.pw_bc_idx_value_pt(i,n,k);
                                            solver.pw_bc_xyz_value_pt(i, j, xyz);
                                            pw_bc_values[i][j] = bc_value_cf_all[i].value(xyz);
                                        }
                                        //ROBIN CONDITIONS
                                        for (int k = 0; k < solver.pw_bc_num_robin_pts(i,n); ++k)
                                        {
                                            int j = solver.pw_bc_idx_robin_pt(i,n,k);
                                            solver.pw_bc_xyz_robin_pt(i, j, xyz);
                                            pw_bc_values_robin[i][j] = bc_value_cf_all[i].value(xyz);
                                            pw_bc_coeffs_robin[i][j] = bc_coeff_cf_all[i].value(xyz);
                                        }
                                    }
                                    //JUMP CONDITIONS
                                    for (int i = 0; i < num_infc(); ++i)
                                    {
                                        for (int k = 0; k < solver.pw_jc_num_taylor_pts(i,n); ++k)
                                        {
                                            int j = solver.pw_jc_idx_taylor_pt(i,n,k);
                                            solver.pw_jc_xyz_taylor_pt(i, j, xyz);
                                            pw_jc_sol_jump_taylor[i][j] = jc_value_cf_all[i].value(xyz);
                                            pw_jc_flx_jump_taylor[i][j] = jc_flux_cf_all[i].value(xyz);
                                        }

                                        for (int k = 0; k < solver.pw_jc_num_integr_pts(i,n); ++k)
                                        {
                                            int j = solver.pw_jc_idx_integr_pt(i,n,k);
                                            solver.pw_jc_xyz_integr_pt(i, j, xyz);
                                            pw_jc_flx_jump_integr[i][j] = jc_flux_cf_all[i].value(xyz);
                                        }
                                    }
                                }
                            }


                            else
                            {
                                for (int i = 0; i < num_bdry(); ++i)
                                {
                                    for (int j = 0; j < solver.pw_bc_num_value_pts(i); ++j)
                                    {
                                        solver.pw_bc_xyz_value_pt(i, j, xyz);
                                        pw_bc_values[i][j] = bc_value_cf_all[i].value(xyz);
                                    }

                                    for (int j = 0; j < solver.pw_bc_num_robin_pts(i); ++j)
                                    {
                                        solver.pw_bc_xyz_robin_pt(i, j, xyz);
                                        pw_bc_values_robin[i][j] = bc_value_cf_all[i].value(xyz);
                                        pw_bc_coeffs_robin[i][j] = bc_coeff_cf_all[i].value(xyz);
                                    }
                                }

                                for (int i = 0; i < num_infc(); ++i)
                                {
                                    for (int j = 0; j < solver.pw_jc_num_taylor_pts(i); ++j)
                                    {
                                        solver.pw_jc_xyz_taylor_pt(i, j, xyz);
                                        pw_jc_sol_jump_taylor[i][j] = jc_value_cf_all[i].value(xyz);
                                        pw_jc_flx_jump_taylor[i][j] = jc_flux_cf_all [i].value(xyz);
                                    }

                                    for (int j = 0; j < solver.pw_jc_num_integr_pts(i); ++j)
                                    {
                                        solver.pw_jc_xyz_integr_pt(i, j, xyz);
                                        pw_jc_flx_jump_integr[i][j] = jc_flux_cf_all[i].value(xyz);
                                    }
                                }
                            }//end bc, jump conditions formulation

                            // pass the sampled BC values to solver
                            for (int i = 0; i < num_bdry(); ++i)
                            {
                                solver.set_bc(i, (BoundaryConditionType) *bc_type_all[i], pw_bc_values[i], pw_bc_values_robin[i], pw_bc_coeffs_robin[i]);
                            }

                            // pass the sampled JC values to solver
                            for (int i = 0; i < num_infc(); ++i)
                            {
                                solver.set_jc(i, pw_jc_sol_jump_taylor[i], pw_jc_flx_jump_taylor[i], pw_jc_flx_jump_integr[i]);
                            }
                        }


                        //NMB: "What is this nonzero guess parameter??"
                        if (which_f_m() == 0 && which_f_p() == 0)
                        {
                            if (use_nonzero_guess()) sample_cf_on_nodes(p4est, nodes, h_cf, sol);
                            solver.solve(sol, use_nonzero_guess());
                        }
                        else
                        {
                            Vec nonlinear_term_m_coeff_sampled;
                            Vec nonlinear_term_p_coeff_sampled;

                            ierr = VecDuplicate(linear_term_m_coeff, &nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
                            ierr = VecDuplicate(linear_term_p_coeff, &nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);

                            sample_cf_on_nodes(p4est, nodes, nu_cf_m, nonlinear_term_m_coeff_sampled);
                            sample_cf_on_nodes(p4est, nodes, nu_cf_p, nonlinear_term_p_coeff_sampled);

                            solver.set_nonlinear_term(nonlinear_term_m_coeff_sampled, f_cf_m, f_prime_cf_m,
                                                      nonlinear_term_p_coeff_sampled, f_cf_p, f_prime_cf_p);

                            solver.set_solve_nonlinear_parameters(nonlinear_method.val, nonlinear_itmax.val, nonlinear_tol.val, 0);

                            if (use_nonzero_guess()) sample_cf_on_nodes(p4est, nodes, h_cf, sol);
                            solver.solve_nonlinear(sol, use_nonzero_guess());

                            ierr = VecDestroy(nonlinear_term_m_coeff_sampled); CHKERRXX(ierr);
                            ierr = VecDestroy(nonlinear_term_p_coeff_sampled); CHKERRXX(ierr);
                        }

                        //levelset stuff
                        Vec bdry_phi_eff = solver.get_boundary_phi_eff();
                        Vec infc_phi_eff = solver.get_interface_phi_eff();









                        // NMB: mask???
                        Vec mask_m  = solver.get_mask_m();
                        Vec mask_p  = solver.get_mask_p();
                        Mat A       = solver.get_matrix();

                        double *bdry_phi_eff_ptr;
                        double *infc_phi_eff_ptr;
                        double *mask_m_ptr;
                        double *mask_p_ptr;

                        if (save_matrix_ascii())
                        {
                            std::ostringstream oss; oss << out_dir << "/matrix/mat_" << file_idx << ".m";

                            PetscViewer viewer;
                            ierr = PetscViewerASCIIOpen(mpi.comm(), oss.str().c_str(), &viewer); CHKERRXX(ierr);
                            ierr = PetscViewerPushFormat(viewer, 	PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);

                            ierr = PetscObjectSetName((PetscObject)A, "mat");
                            ierr = MatView(A, viewer); CHKERRXX(ierr);

                            Vec lex_order;
                            ierr = VecCreateGhostNodes(p4est, nodes, &lex_order); CHKERRXX(ierr);

                            double *vec_ptr; ierr = VecGetArray(lex_order, &vec_ptr); CHKERRXX(ierr);

                            int nx = round((grid_xyz_max_shift[0]-grid_xyz_min_shift[0])/dxyz[0] + 1);
#ifdef P4_TO_P8
                            int ny = round((grid_xyz_max_shift[1]-grid_xyz_min_shift[1])/dxyz[1] + 1);
                int nz = round((grid_xyz_max_shift[2]-grid_xyz_min_shift[2])/dxyz[2] + 1);
#endif
                            for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
                            {
                                double xyz[P4EST_DIM];
                                node_xyz_fr_n(n, p4est, nodes, xyz);

                                int ix = round((xyz[0]-grid_xyz_min_shift[0])/dxyz[0]);
                                int iy = round((xyz[1]-grid_xyz_min_shift[1])/dxyz[1]);
#ifdef P4_TO_P8
                                int iz = round((xyz[2]-grid_xyz_min_shift[2])/dxyz[2]);
                  vec_ptr[n] = iz*nx*ny + iy*(nx) + ix + 1;
#else
                                vec_ptr[n] = iy*(nx) + ix + 1;
#endif
                            }

                            ierr = VecRestoreArray(lex_order, &vec_ptr); CHKERRXX(ierr);

                            ierr = PetscObjectSetName((PetscObject)lex_order, "vec");
                            ierr = VecView(lex_order, viewer); CHKERRXX(ierr);

                            ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
                        }

                        if (save_matrix_binary())
                        {
                            std::ostringstream oss; oss << out_dir << "/matrix/mat_" << file_idx << ".dat";

                            PetscViewer viewer;
                            ierr = PetscViewerBinaryOpen(mpi.comm(), oss.str().c_str(), FILE_MODE_WRITE, &viewer); CHKERRXX(ierr);
                            ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_BINARY_MATLAB); CHKERRXX(ierr);
                            ierr = MatView(A, viewer); CHKERRXX(ierr);
                            ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
                        }

#ifdef MATLAB_PROVIDED
                        if (iter < compute_cond_num())
              {
                // Get the local AIJ representation of the matrix
                std::vector<double> aij;

                int M,N;

                ierr = MatGetLocalSize(A, &M, &N);

                for (int n = 0; n < M; ++n)
                {
                  int num_elem;
                  const int *icol;
                  const double *vals;

                  PetscInt N = solver.get_global_idx(n);
                  MatGetRow(A, N, &num_elem, &icol, &vals);
                  for (int i = 0; i < num_elem; ++i)
                  {
                    aij.push_back((double) (N+1));
                    aij.push_back((double) (icol[i]+1));
                    aij.push_back(vals[i]);
                  }
                  MatRestoreRow(A, N, &num_elem, &icol, &vals);
                }

                int num_local_entries = aij.size();

                // Collect all chucks of the matrix into global_aij on the 0-rank process
                std::vector<int> local_sizes(mpi.size(), 0);
                std::vector<int> displs(mpi.size(), 0);

                MPI_Gather(&num_local_entries, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, 0, mpi.comm());

                int num_total_entries = local_sizes[0];

                for (int i = 1; i < mpi.size(); ++i)
                {
                  displs[i] = displs[i-1] + local_sizes[i-1];
                  num_total_entries += local_sizes[i];
                }

                mxArray *mat = nullptr;
                mxDouble *mat_data = nullptr;

                if (mpi.rank() == 0)
                {
                  mat = mxCreateDoubleMatrix(3, num_total_entries/3, mxREAL);
                  mat_data = mxGetDoubles(mat);
                }

                MPI_Gatherv(aij.data(), aij.size(), MPI_DOUBLE, mat_data, local_sizes.data(), displs.data(), MPI_DOUBLE, 0, mpi.comm());

                aij.clear();

                // pass the matrix to MATLAB and ask to compute condition number
                if (mpi.rank() == 0)
                {
                  // send the matrix to MATLAB
                  engPutVariable(mengine, "AIJ", mat);
                  mxDestroyArray(mat);

                  // ask to compute condition number
                  engEvalString(mengine, "cn = condest(spconvert(AIJ'));");

                  // get the result
                  mxArray *value = engGetVariable(mengine, "cn");
                  double cn = *mxGetDoubles(value);
                  mxDestroyArray(value);

                  // store
                  cond_num_arr.push_back(cn);
                } else {
                  cond_num_arr.push_back(NAN);
                }
              } else {
                cond_num_arr.push_back(NAN);
              }
#else
                        cond_num_arr.push_back(NAN);
#endif

//              my_p4est_integration_mls_t integrator(p4est, nodes);
//// integrator.set_phi(bdry_phi, bdry_acn, bdry_clr);

//            s/*
//            if (save_domain)
//            {
//              std::ostringstream oss; oss << out_dir << "/geometry";

//#ifdef P4_TO_P8
//              vector<cube3_mls_t> cubes;
//              unsigned int n_sps = 6;
//#else
//              vector<cube2_mls_t> cubes;
//              unsigned int n_sps = 2;
//#endif
//              solver.reconstruct_domain(cubes);

//              if (integration_order == 1)
//              {
//#ifdef P4_TO_P8
//                vector<simplex3_mls_l_t *> simplices;
//#else
//                vector<simplex2_mls_l_t *> simplices;
//#endif
//                for (unsigned int k = 0; k < cubes.size(); k++)
//                  for (unsigned int kk = 0; kk < cubes[k].cubes_l_.size(); kk++)
//                    if (cubes[k].cubes_l_[kk]->loc == FCE)
//                      for (unsigned int l = 0; l < n_sps; l++)
//                        simplices.push_back(&cubes[k].cubes_l_[kk]->simplex[l]);

//#ifdef P4_TO_P8
//                simplex3_mls_l_vtk::write_simplex_geometryetry(simplices, oss.str(), to_string(file_idx));
//#else
//                simplex2_mls_l_vtk::write_simplex_geometryetry(simplices, oss.str(), to_string(file_idx));
//#endif
//              } else if (integration_order == 2) {

//#ifdef P4_TO_P8
//                vector<simplex3_mls_q_t *> simplices;
//#else
//                vector<simplex2_mls_q_t *> simplices;
//#endif
//                for (unsigned int k = 0; k < cubes.size(); k++)
//                  for (unsigned int kk = 0; kk < cubes[k].cubes_q_.size(); kk++)
//                    if (cubes[k].cubes_q_[kk]->loc == FCE)
//                      for (unsigned int l = 0; l < n_sps; l++)
//                        simplices.push_back(&cubes[k].cubes_q_[kk]->simplex[l]);

//#ifdef P4_TO_P8
//                simplex3_mls_q_vtk::write_simplex_geometryetry(simplices, oss.str(), to_string(file_idx));
//#else
//                simplex2_mls_q_vtk::write_simplex_geometryetry(simplices, oss.str(), to_string(file_idx));
//#endif
//              }

//            }
                        //*/

                        // Frederic add begin
                        if (save_domain.val) {
                            my_p4est_integration_mls_t integrator(p4est, nodes);
                            std::vector<CF_DIM *> bdry_phi(num_bdry(), nullptr);
                            std::vector<mls_opn_t> bdry_acn(num_bdry(), MLS_INTERSECTION);
                            std::vector<int> bdry_clr(num_bdry(), 0);

                            for (int i = 0; i < num_bdry(); ++i) {
                                bdry_phi[i] = &bdry_phi_cf_all[i];
                                bdry_acn[i] = mls_opn_t(*bdry_opn_all[i]);
                                bdry_clr[i] = i;
                            }

                            integrator.set_phi(bdry_phi, bdry_acn, bdry_clr, integration_order.val == 1);
                            integrator.initialize();

                            std::ostringstream oss;
                            oss << out_dir << "/geometry";

#ifdef P4_TO_P8
                            vector<cube3_mls_t> &cubes = integrator.cubes;
              unsigned int n_sps = 6;
#else
                            vector<cube2_mls_t> &cubes = integrator.cubes;
                            unsigned int n_sps = 2;
#endif

                            if (integration_order.val == 1) {
#ifdef P4_TO_P8
                                vector<simplex3_mls_l_t *> simplices;
#else
                                vector<simplex2_mls_l_t *> simplices;
#endif
                                for (unsigned int k = 0; k < cubes.size(); k++)
                                    for (unsigned int kk = 0; kk < cubes[k].cubes_l_.size(); kk++)
                                        if (cubes[k].cubes_l_[kk]->loc == FCE)
                                            for (unsigned int l = 0; l < n_sps; l++)
                                                simplices.push_back(&cubes[k].cubes_l_[kk]->simplex[l]);

#ifdef P4_TO_P8
                                simplex3_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#else
                                simplex2_mls_l_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#endif
                            } else if (integration_order.val == 2) {

#ifdef P4_TO_P8
                                vector<simplex3_mls_q_t *> simplices;
#else
                                vector<simplex2_mls_q_t *> simplices;
#endif
                                for (unsigned int k = 0; k < cubes.size(); k++)
                                    for (unsigned int kk = 0; kk < cubes[k].cubes_q_.size(); kk++)
                                        if (cubes[k].cubes_q_[kk]->loc == FCE)
                                            for (unsigned int l = 0; l < n_sps; l++)
                                                simplices.push_back(&cubes[k].cubes_q_[kk]->simplex[l]);

#ifdef P4_TO_P8
                                simplex3_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#else
                                simplex2_mls_q_vtk::write_simplex_geometry(simplices, oss.str(), to_string(file_idx));
#endif
                            }

                        }
                        // Frederic add end
                        Vec sol_m = sol; double *sol_m_ptr;
                        Vec sol_p = sol; double *sol_p_ptr;

                        /* calculate errors */
                        Vec vec_error_sl_m; double *vec_error_sl_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl_m); CHKERRXX(ierr);
                        Vec vec_error_gr_m; double *vec_error_gr_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr_m); CHKERRXX(ierr);
                        Vec vec_error_ex_m; double *vec_error_ex_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex_m); CHKERRXX(ierr);
                        Vec vec_error_dd_m; double *vec_error_dd_m_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd_m); CHKERRXX(ierr);

                        Vec vec_error_sl_p; double *vec_error_sl_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_sl_p); CHKERRXX(ierr);
                        Vec vec_error_gr_p; double *vec_error_gr_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_gr_p); CHKERRXX(ierr);
                        Vec vec_error_ex_p; double *vec_error_ex_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_ex_p); CHKERRXX(ierr);
                        Vec vec_error_dd_p; double *vec_error_dd_p_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_dd_p); CHKERRXX(ierr);

                        //----------------------------------------------------------------------------------------------
                        // calculate error of solution
                        //----------------------------------------------------------------------------------------------
                        ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        double h_max = 0;

                        foreach_local_node(n, nodes)
                        {
                            double xyz[P4EST_DIM];
                            node_xyz_fr_n(n, p4est, nodes, xyz);

                            vec_error_sl_m_ptr[n] = mask_m_ptr[n] < 0 ? ABS(sol_m_ptr[n] - h_m_cf.value(xyz)) : 0;
                            vec_error_sl_p_ptr[n] = mask_p_ptr[n] < 0 ? ABS(sol_p_ptr[n] - h_p_cf.value(xyz)) : 0;

                            h_max = MAX(h_max, fabs(h_m_cf.value(xyz)), fabs(h_p_cf.value(xyz)));
                        }

                        ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);

                        ierr = VecGhostUpdateBegin(vec_error_sl_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateBegin(vec_error_sl_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_sl_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_sl_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

                        //----------------------------------------------------------------------------------------------
                        // calculate error of gradients
                        //----------------------------------------------------------------------------------------------
                        ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);

                        quad_neighbor_nodes_of_node_t qnnn{};

                        double gr_max = 0;

                        foreach_local_node(n, nodes)
                        {
                            double xyz[P4EST_DIM];
                            node_xyz_fr_n(n, p4est, nodes, xyz);

                            auto *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

                            if (!compute_grad_between())
                            {
                                ngbd_n.get_neighbors(n, qnnn);

                                if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_m_ptr))
                                {
                                    double DIM( ux_m_exact = hx_m_cf(DIM(xyz[0], xyz[1], xyz[2])),
                                                uy_m_exact = hy_m_cf(DIM(xyz[0], xyz[1], xyz[2])),
                                                uz_m_exact = hz_m_cf(DIM(xyz[0], xyz[1], xyz[2])) );

                                    gr_max = MAX(gr_max, sqrt(SUMD(SQR(ux_m_exact), SQR(uy_m_exact), SQR(uz_m_exact))));

                                    double DIM( ux_m_error = fabs(qnnn.dx_central(sol_m_ptr) - ux_m_exact),
                                                uy_m_error = fabs(qnnn.dy_central(sol_m_ptr) - uy_m_exact),
                                                uz_m_error = fabs(qnnn.dz_central(sol_m_ptr) - uz_m_exact) );

                                    vec_error_gr_m_ptr[n] = sqrt(SUMD(SQR(ux_m_error), SQR(uy_m_error), SQR(uz_m_error)));
                                } else {
                                    vec_error_gr_m_ptr[n] = 0;
                                }

                                if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_p_ptr))
                                {
                                    double DIM( ux_p_exact = hx_p_cf(DIM(xyz[0], xyz[1], xyz[2])),
                                                uy_p_exact = hy_p_cf(DIM(xyz[0], xyz[1], xyz[2])),
                                                uz_p_exact = hz_p_cf(DIM(xyz[0], xyz[1], xyz[2])) );

                                    gr_max = MAX(gr_max, sqrt(SUMD(SQR(ux_p_exact), SQR(uy_p_exact), SQR(uz_p_exact))));

                                    double DIM( ux_p_error = fabs(qnnn.dx_central(sol_p_ptr) - ux_p_exact),
                                                uy_p_error = fabs(qnnn.dy_central(sol_p_ptr) - uy_p_exact),
                                                uz_p_error = fabs(qnnn.dz_central(sol_p_ptr) - uz_p_exact) );

                                    vec_error_gr_p_ptr[n] = sqrt(SUMD(SQR(ux_p_error), SQR(uy_p_error), SQR(uz_p_error)));
                                } else {
                                    vec_error_gr_p_ptr[n] = 0;
                                }
                            } else {
                                p4est_locidx_t neighbors      [num_neighbors_cube];
                                bool           neighbors_exist[num_neighbors_cube];

                                double xyz_nei[P4EST_DIM];
                                double xyz_mid[P4EST_DIM];
                                double normal[P4EST_DIM];

                                vec_error_gr_m_ptr[n] = 0;
                                vec_error_gr_p_ptr[n] = 0;

                                if (!is_node_Wall(p4est, ni))
                                {
                                    ngbd_n.get_all_neighbors(n, neighbors, neighbors_exist);
                                    for (int j = 1; j < (int)pow(3, P4EST_DIM); j+=2)
                                    {
                                        p4est_locidx_t n_nei = neighbors[j];
                                        node_xyz_fr_n(n_nei, p4est, nodes, xyz_nei);

                                        double delta = 0;

                                        foreach_dimension(i)
                                        {
                                            xyz_mid[i] = .5*(xyz[i]+xyz_nei[i]);
                                            delta += SQR(xyz[i]-xyz_nei[i]);
                                            normal[i] = xyz_nei[i]-xyz[i];
                                        }

                                        delta = sqrt(delta);

                                        foreach_dimension(i)
                                            normal[i] /= delta;

                                        if (mask_m_ptr[n] < 0)
                                            if (mask_m_ptr[n_nei] < 0)
                                            {
                                                double grad_exact = SUMD(hx_m_cf.value(xyz_mid)*normal[0], hy_m_cf.value(xyz_mid)*normal[1], hz_m_cf.value(xyz_mid)*normal[2]);
                                                vec_error_gr_m_ptr[n] = MAX(vec_error_gr_m_ptr[n], fabs((sol_m_ptr[n_nei]-sol_m_ptr[n])/delta - grad_exact));
                                                gr_max = MAX(gr_max, fabs(grad_exact));
                                            }

                                        if (mask_p_ptr[n] < 0)
                                            if (mask_p_ptr[n_nei] < 0)
                                            {
                                                double grad_exact = SUMD(hx_p_cf.value(xyz_mid)*normal[0], hy_p_cf.value(xyz_mid)*normal[1], hz_p_cf.value(xyz_mid)*normal[2]);
                                                vec_error_gr_p_ptr[n] = MAX(vec_error_gr_p_ptr[n], fabs((sol_p_ptr[n_nei]-sol_p_ptr[n])/delta - grad_exact));
                                                gr_max = MAX(gr_max, fabs(grad_exact));
                                            }
                                    }
                                }
                            }

                        }

                        ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);

                        ierr = VecGhostUpdateBegin(vec_error_gr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateBegin(vec_error_gr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_gr_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_gr_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

                        //---------------------------------------------------------------------------------------------
                        // calculate error of Laplacian
                        //----------------------------------------------------------------------------------------------
                        ierr = VecGetArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

                        foreach_local_node(n, nodes)
                        {
                            double xyz[P4EST_DIM];
                            node_xyz_fr_n(n, p4est, nodes, xyz);

                            p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
                            ngbd_n.get_neighbors(n, qnnn);

                            if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_m_ptr))
                            {
                                double hdd_exact = laplace_h_m_cf.value(xyz);
                                double DIM( hxx = qnnn.dxx_central(sol_m_ptr),
                                            hyy = qnnn.dyy_central(sol_m_ptr),
                                            hzz = qnnn.dzz_central(sol_m_ptr) );
                                vec_error_dd_m_ptr[n] = fabs(hdd_exact - SUMD(hxx,hyy,hzz));
                            } else {
                                vec_error_dd_m_ptr[n] = 0;
                            }

                            if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_p_ptr))
                            {
                                double hdd_exact = laplace_h_p_cf.value(xyz);
                                double DIM( hxx = qnnn.dxx_central(sol_p_ptr),
                                            hyy = qnnn.dyy_central(sol_p_ptr),
                                            hzz = qnnn.dzz_central(sol_p_ptr) );
                                vec_error_dd_p_ptr[n] = fabs(hdd_exact - SUMD(hxx,hyy,hzz));
                            } else {
                                vec_error_dd_p_ptr[n] = 0;
                            }
                        }

                        ierr = VecRestoreArray(sol_m, &sol_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(sol_p, &sol_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

                        ierr = VecGhostUpdateBegin(vec_error_dd_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateBegin(vec_error_dd_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_dd_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_dd_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

                        //----------------------------------------------------------------------------------------------
                        // calculate extrapolation error
                        //----------------------------------------------------------------------------------------------
                        double band = extension_band_check();

                        // copy solution into a new Vec
                        Vec sol_m_ex; double *sol_m_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_m_ex); CHKERRXX(ierr);
                        Vec sol_p_ex; double *sol_p_ex_ptr; ierr = VecCreateGhostNodes(p4est, nodes, &sol_p_ex); CHKERRXX(ierr);

                        VecCopyGhost(sol_m, sol_m_ex);
                        VecCopyGhost(sol_p, sol_p_ex);

                        Vec phi_m; ierr = VecDuplicate(bdry_phi_eff, &phi_m); CHKERRXX(ierr); VecCopyGhost(bdry_phi_eff, phi_m);
                        Vec phi_p; ierr = VecDuplicate(bdry_phi_eff, &phi_p); CHKERRXX(ierr); VecCopyGhost(bdry_phi_eff, phi_p);

                        double *phi_m_ptr;
                        double *phi_p_ptr;

                        VecPointwiseMaxGhost(phi_m, phi_m, infc_phi_eff);
                        VecScaleGhost(infc_phi_eff, -1);
                        VecPointwiseMaxGhost(phi_p, phi_p, infc_phi_eff);
                        VecScaleGhost(infc_phi_eff, -1);

                        // extend
                        boundary_conditions_t *bc = nullptr;
                        if (apply_bc_pointwise())
                        {
                            bc = solver.get_bc(0);
                        }

//              ls.set_show_convergence(0);
                        switch (extend_solution())
                        {
                            case 1:
                                ls.extend_Over_Interface_TVD(phi_m, sol_m_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max, extension_band_extend()*dxyz_max, nullptr, mask_m, bc, use_nonzero_guess()); CHKERRXX(ierr);
                                ls.extend_Over_Interface_TVD(phi_p, sol_p_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max, extension_band_extend()*dxyz_max, nullptr, mask_p, bc, use_nonzero_guess()); CHKERRXX(ierr);
                                break;
                            case 2:
//                  ls.extend_Over_Interface_TVD_Full(phi_m, sol_m_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max, nullptr, mask_m, nullptr, use_nonzero_guess()); CHKERRXX(ierr);
//                  ls.extend_Over_Interface_TVD_Full(phi_p, sol_p_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max, nullptr, mask_p, nullptr, use_nonzero_guess()); CHKERRXX(ierr);
                                ls.extend_Over_Interface_TVD_Full(phi_m, sol_m_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max, nullptr, mask_m, bc, use_nonzero_guess()); CHKERRXX(ierr);
                                ls.extend_Over_Interface_TVD_Full(phi_p, sol_p_ex, extension_iterations(), 2, -extension_band_compute()*dxyz_max,  extension_band_extend()*dxyz_max, nullptr, mask_p, bc, use_nonzero_guess()); CHKERRXX(ierr);
                                break;
                        }

                        // calculate error
                        ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecGetArray(phi_m, &phi_m_ptr); CHKERRXX(ierr);
                        ierr = VecGetArray(phi_p, &phi_p_ptr); CHKERRXX(ierr);

                        foreach_local_node(n, nodes)
                        {
                            double xyz[P4EST_DIM];
                            node_xyz_fr_n(n, p4est, nodes, xyz);

                            vec_error_ex_m_ptr[n] = (mask_m_ptr[n] > 0. && phi_m_ptr[n] < band*dxyz_max) ? ABS(sol_m_ex_ptr[n] - h_m_cf.value(xyz)) : 0;
                            vec_error_ex_p_ptr[n] = (mask_p_ptr[n] > 0. && phi_p_ptr[n] < band*dxyz_max) ? ABS(sol_p_ex_ptr[n] - h_p_cf.value(xyz)) : 0;
                        }

                        ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                        ierr = VecRestoreArray(phi_m, &phi_m_ptr); CHKERRXX(ierr);
                        ierr = VecRestoreArray(phi_p, &phi_p_ptr); CHKERRXX(ierr);

                        ierr = VecGhostUpdateBegin(vec_error_ex_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateBegin(vec_error_ex_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

                        ierr = VecGhostUpdateEnd  (vec_error_ex_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                        ierr = VecGhostUpdateEnd  (vec_error_ex_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

                        ierr = VecDestroy(phi_m); CHKERRXX(ierr);
                        ierr = VecDestroy(phi_p); CHKERRXX(ierr);

                        // compute L-inf norm of errors
                        double err_sl_m_max = 0.;   ierr = VecMax(vec_error_sl_m, nullptr, &err_sl_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_gr_m_max = 0.;   ierr = VecMax(vec_error_gr_m, nullptr, &err_gr_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_ex_m_max = 0.;   ierr = VecMax(vec_error_ex_m, nullptr, &err_ex_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_dd_m_max = 0.;   ierr = VecMax(vec_error_dd_m, nullptr, &err_dd_m_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

                        double err_sl_p_max = 0.;   ierr = VecMax(vec_error_sl_p, nullptr, &err_sl_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_gr_p_max = 0.;   ierr = VecMax(vec_error_gr_p, nullptr, &err_gr_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_gr_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_ex_p_max = 0.;   ierr = VecMax(vec_error_ex_p, nullptr, &err_ex_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                        double err_dd_p_max = 0.;   ierr = VecMax(vec_error_dd_p, nullptr, &err_dd_p_max); CHKERRXX(ierr);   mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_dd_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

                        if (scale_errors())
                        {
                            mpiret = MPI_Allreduce(MPI_IN_PLACE, &h_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                            err_sl_m_max /= h_max;
                            err_sl_p_max /= h_max;

                            mpiret = MPI_Allreduce(MPI_IN_PLACE, &gr_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
                            err_gr_m_max /= gr_max;
                            err_gr_p_max /= gr_max;
                        }

                        error_sl_m_arr.push_back(err_sl_m_max);
                        error_gr_m_arr.push_back(err_gr_m_max);
                        error_ex_m_arr.push_back(err_ex_m_max);
                        error_dd_m_arr.push_back(err_dd_m_max);

                        error_sl_p_arr.push_back(err_sl_p_max);
                        error_gr_p_arr.push_back(err_gr_p_max);
                        error_ex_p_arr.push_back(err_ex_p_max);
                        error_dd_p_arr.push_back(err_dd_p_max);

                        // Print current errors
                        if (iter > -1)
                        {
                            ierr = PetscPrintf(p4est->mpicomm, "Errors Neg: "); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "sol = %3.2e (%+3.2f), ", err_sl_m_max, log(error_sl_m_arr[iter-1]/error_sl_m_arr[iter])/log(2)); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "gra = %3.2e (%+3.2f), ", err_gr_m_max, log(error_gr_m_arr[iter-1]/error_gr_m_arr[iter])/log(2)); CHKERRXX(ierr);
                            //ierr = PetscPrintf(p4est->mpicomm, "ext = %3.2e (%+3.2f), ", err_ex_m_max, log(error_ex_m_arr[iter-1]/error_ex_m_arr[iter])/log(2)); CHKERRXX(ierr);
                            //ierr = PetscPrintf(p4est->mpicomm, "lap = %3.2e (%+3.2f). ", err_dd_m_max, log(error_dd_m_arr[iter-1]/error_dd_m_arr[iter])/log(2)); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "\n"); CHKERRXX(ierr);

                            ierr = PetscPrintf(p4est->mpicomm, "Errors Pos: "); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "sol = %3.2e (%+3.2f), ", err_sl_p_max, log(error_sl_p_arr[iter-1]/error_sl_p_arr[iter])/log(2)); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "gra = %3.2e (%+3.2f), ", err_gr_p_max, log(error_gr_p_arr[iter-1]/error_gr_p_arr[iter])/log(2)); CHKERRXX(ierr);
                            //ierr = PetscPrintf(p4est->mpicomm, "ext = %3.2e (%+3.2f), ", err_ex_p_max, log(error_ex_p_arr[iter-1]/error_ex_p_arr[iter])/log(2)); CHKERRXX(ierr);
                            //ierr = PetscPrintf(p4est->mpicomm, "lap = %3.2e (%+3.2f). ", err_dd_p_max, log(error_dd_p_arr[iter-1]/error_dd_p_arr[iter])/log(2)); CHKERRXX(ierr);
                            ierr = PetscPrintf(p4est->mpicomm, "\n"); CHKERRXX(ierr);

                            ierr = PetscPrintf(p4est->mpicomm, "Cond num: %e\n", cond_num_arr[iter]); CHKERRXX(ierr);
                        }

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

                            Vec     exact;
                            double *exact_ptr;

                            ierr = VecDuplicate(sol, &exact); CHKERRXX(ierr);
                            sample_cf_on_nodes(p4est, nodes, h_cf, exact);

                            ierr = VecGetArray(bdry_phi_eff, &bdry_phi_eff_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(infc_phi_eff, &infc_phi_eff_ptr); CHKERRXX(ierr);

                            ierr = VecGetArray(sol,   &sol_ptr);   CHKERRXX(ierr);
                            ierr = VecGetArray(exact, &exact_ptr); CHKERRXX(ierr);

                            ierr = VecGetArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

                            ierr = VecGetArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);

                            ierr = VecGetArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

                            ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                            double *mu_m_ptr;
                            double *mu_p_ptr;

                            ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
                            ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);

                            my_p4est_vtk_write_all(p4est, nodes, ghost,
                                                   P4EST_TRUE, P4EST_TRUE,
                                                   18, 1, oss.str().c_str(),
                                                   VTK_POINT_DATA, "phi", bdry_phi_eff_ptr,
                                                   VTK_POINT_DATA, "infc_phi", infc_phi_eff_ptr,
                                                   VTK_POINT_DATA, "sol", sol_ptr,
                                                   VTK_POINT_DATA, "exact", exact_ptr,
                                                   VTK_POINT_DATA, "sol_m_ex", sol_m_ex_ptr,
                                                   VTK_POINT_DATA, "sol_p_ex", sol_p_ex_ptr,
                                                   VTK_POINT_DATA, "mu_m", mu_m_ptr,
                                                   VTK_POINT_DATA, "mu_p", mu_p_ptr,
                                                   VTK_POINT_DATA, "mask_m", mask_m_ptr,
                                                   VTK_POINT_DATA, "mask_p", mask_p_ptr,
                                                   VTK_POINT_DATA, "error_sl_m", vec_error_sl_m_ptr,
                                                   VTK_POINT_DATA, "error_gr_m", vec_error_gr_m_ptr,
                                                   VTK_POINT_DATA, "error_ex_m", vec_error_ex_m_ptr,
                                                   VTK_POINT_DATA, "error_dd_m", vec_error_dd_m_ptr,
                                                   VTK_POINT_DATA, "error_sl_p", vec_error_sl_p_ptr,
                                                   VTK_POINT_DATA, "error_gr_p", vec_error_gr_p_ptr,
                                                   VTK_POINT_DATA, "error_ex_p", vec_error_ex_p_ptr,
                                                   VTK_POINT_DATA, "error_dd_p", vec_error_dd_p_ptr,
                                                   VTK_CELL_DATA , "leaf_level", l_p);

                            ierr = VecRestoreArray(bdry_phi_eff, &bdry_phi_eff_ptr);    CHKERRXX(ierr);
                            ierr = VecRestoreArray(infc_phi_eff, &infc_phi_eff_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(sol,   &sol_ptr);   CHKERRXX(ierr);
                            ierr = VecRestoreArray(exact, &exact_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(sol_m_ex, &sol_m_ex_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(sol_p_ex, &sol_p_ex_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(vec_error_sl_m, &vec_error_sl_m_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_gr_m, &vec_error_gr_m_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_ex_m, &vec_error_ex_m_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_dd_m, &vec_error_dd_m_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(vec_error_sl_p, &vec_error_sl_p_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_gr_p, &vec_error_gr_p_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_ex_p, &vec_error_ex_p_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(vec_error_dd_p, &vec_error_dd_p_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
                            ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);

                            ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
                            ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
                            ierr = VecDestroy(exact); CHKERRXX(ierr);

                            PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
                        }

                        // destroy Vec's with errors
                        ierr = VecDestroy(vec_error_sl_m); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_gr_m); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_ex_m); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_dd_m); CHKERRXX(ierr);

                        ierr = VecDestroy(vec_error_sl_p); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_gr_p); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_ex_p); CHKERRXX(ierr);
                        ierr = VecDestroy(vec_error_dd_p); CHKERRXX(ierr);

                        ierr = VecDestroy(sol_m_ex); CHKERRXX(ierr);
                        ierr = VecDestroy(sol_p_ex); CHKERRXX(ierr);

                        ierr = VecDestroy(sol);           CHKERRXX(ierr);

                        ierr = VecDestroy(mu_m);          CHKERRXX(ierr);
                        ierr = VecDestroy(mu_p);          CHKERRXX(ierr);

                        ierr = VecDestroy(rhs_m);         CHKERRXX(ierr);
                        ierr = VecDestroy(rhs_p);         CHKERRXX(ierr);

                        ierr = VecDestroy(linear_term_m_coeff);   CHKERRXX(ierr);
                        ierr = VecDestroy(linear_term_p_coeff);   CHKERRXX(ierr);

                        for (unsigned int i = 0; i < num_bdry_max; i++) { if (*bdry_present_all[i] == true) { ierr = VecDestroy(bdry_phi_vec_all[i]); CHKERRXX(ierr); } }
                        for (unsigned int i = 0; i < num_infc_max; i++) { if (*infc_present_all[i] == true) { ierr = VecDestroy(infc_phi_vec_all[i]); CHKERRXX(ierr); } }

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
    }

#ifdef MATLAB_PROVIDED
    if (mpi.rank() == 0 && compute_cond_num())
  {
    engClose(mengine);
  }
#endif


    MPI_Barrier(mpi.comm());

    std::vector<double> error_m_sl_one(num_resolutions, 0), error_m_sl_avg(num_resolutions, 0), error_m_sl_max(num_resolutions, 0);
    std::vector<double> error_m_gr_one(num_resolutions, 0), error_m_gr_avg(num_resolutions, 0), error_m_gr_max(num_resolutions, 0);
    std::vector<double> error_m_dd_one(num_resolutions, 0), error_m_dd_avg(num_resolutions, 0), error_m_dd_max(num_resolutions, 0);
    std::vector<double> error_m_ex_one(num_resolutions, 0), error_m_ex_avg(num_resolutions, 0), error_m_ex_max(num_resolutions, 0);

    std::vector<double> error_p_sl_one(num_resolutions, 0), error_p_sl_avg(num_resolutions, 0), error_p_sl_max(num_resolutions, 0);
    std::vector<double> error_p_gr_one(num_resolutions, 0), error_p_gr_avg(num_resolutions, 0), error_p_gr_max(num_resolutions, 0);
    std::vector<double> error_p_dd_one(num_resolutions, 0), error_p_dd_avg(num_resolutions, 0), error_p_dd_max(num_resolutions, 0);
    std::vector<double> error_p_ex_one(num_resolutions, 0), error_p_ex_avg(num_resolutions, 0), error_p_ex_max(num_resolutions, 0);

    std::vector<double> cond_num_one(num_resolutions, 0), cond_num_avg(num_resolutions, 0), cond_num_max(num_resolutions, 0);

////  error_dd_m_arr = error_sl_m_arr;
//  error_ex_m_arr = error_sl_m_arr;
////  error_dd_p_arr = error_sl_p_arr;
//  error_ex_p_arr = error_sl_p_arr;

    // for each resolution compute max, mean and deviation
    for (int p = 0; p < num_resolutions; ++p)
    {
        // one
        error_m_sl_one[p] = error_sl_m_arr[p*num_shifts_total];
        error_m_gr_one[p] = error_gr_m_arr[p*num_shifts_total];
        error_m_dd_one[p] = error_dd_m_arr[p*num_shifts_total];
        error_m_ex_one[p] = error_ex_m_arr[p*num_shifts_total];

        error_p_sl_one[p] = error_sl_p_arr[p*num_shifts_total];
        error_p_gr_one[p] = error_gr_p_arr[p*num_shifts_total];
        error_p_dd_one[p] = error_dd_p_arr[p*num_shifts_total];
        error_p_ex_one[p] = error_ex_p_arr[p*num_shifts_total];

        cond_num_one[p] = cond_num_arr[p*num_shifts_total];

        // max
        for (int s = 0; s < num_shifts_total; ++s)
        {
            error_m_sl_max[p] = MAX(error_m_sl_max[p], error_sl_m_arr[p*num_shifts_total + s]);
            error_m_gr_max[p] = MAX(error_m_gr_max[p], error_gr_m_arr[p*num_shifts_total + s]);
            error_m_dd_max[p] = MAX(error_m_dd_max[p], error_dd_m_arr[p*num_shifts_total + s]);
            error_m_ex_max[p] = MAX(error_m_ex_max[p], error_ex_m_arr[p*num_shifts_total + s]);

            error_p_sl_max[p] = MAX(error_p_sl_max[p], error_sl_p_arr[p*num_shifts_total + s]);
            error_p_gr_max[p] = MAX(error_p_gr_max[p], error_gr_p_arr[p*num_shifts_total + s]);
            error_p_dd_max[p] = MAX(error_p_dd_max[p], error_dd_p_arr[p*num_shifts_total + s]);
            error_p_ex_max[p] = MAX(error_p_ex_max[p], error_ex_p_arr[p*num_shifts_total + s]);

            cond_num_max[p] = MAX(cond_num_max[p], cond_num_arr[p*num_shifts_total + s]);
        }

        // avg
        for (int s = 0; s < num_shifts_total; ++s)
        {
            error_m_sl_avg[p] += error_sl_m_arr[p*num_shifts_total + s];
            error_m_gr_avg[p] += error_gr_m_arr[p*num_shifts_total + s];
            error_m_dd_avg[p] += error_dd_m_arr[p*num_shifts_total + s];
            error_m_ex_avg[p] += error_ex_m_arr[p*num_shifts_total + s];

            error_p_sl_avg[p] += error_sl_p_arr[p*num_shifts_total + s];
            error_p_gr_avg[p] += error_gr_p_arr[p*num_shifts_total + s];
            error_p_dd_avg[p] += error_dd_p_arr[p*num_shifts_total + s];
            error_p_ex_avg[p] += error_ex_p_arr[p*num_shifts_total + s];

            cond_num_avg[p] += cond_num_arr[p*num_shifts_total + s];
        }

        error_m_sl_avg[p] /= num_shifts_total;
        error_m_gr_avg[p] /= num_shifts_total;
        error_m_dd_avg[p] /= num_shifts_total;
        error_m_ex_avg[p] /= num_shifts_total;

        error_p_sl_avg[p] /= num_shifts_total;
        error_p_gr_avg[p] /= num_shifts_total;
        error_p_dd_avg[p] /= num_shifts_total;
        error_p_ex_avg[p] /= num_shifts_total;

        cond_num_avg[p] /= num_shifts_total;
    }

    //output stuff
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
        filename = out_dir; filename += "/convergence/mu_arr.txt"; save_vector(filename.c_str(), mu_arr);

        filename = out_dir; filename += "/convergence/error_m_sl_all.txt"; save_vector(filename.c_str(), error_sl_m_arr);
        filename = out_dir; filename += "/convergence/error_m_gr_all.txt"; save_vector(filename.c_str(), error_gr_m_arr);
        filename = out_dir; filename += "/convergence/error_m_dd_all.txt"; save_vector(filename.c_str(), error_dd_m_arr);
        filename = out_dir; filename += "/convergence/error_m_ex_all.txt"; save_vector(filename.c_str(), error_ex_m_arr);

        filename = out_dir; filename += "/convergence/error_m_sl_one.txt"; save_vector(filename.c_str(), error_m_sl_one);
        filename = out_dir; filename += "/convergence/error_m_gr_one.txt"; save_vector(filename.c_str(), error_m_gr_one);
        filename = out_dir; filename += "/convergence/error_m_dd_one.txt"; save_vector(filename.c_str(), error_m_dd_one);
        filename = out_dir; filename += "/convergence/error_m_ex_one.txt"; save_vector(filename.c_str(), error_m_ex_one);

        filename = out_dir; filename += "/convergence/error_m_sl_avg.txt"; save_vector(filename.c_str(), error_m_sl_avg);
        filename = out_dir; filename += "/convergence/error_m_gr_avg.txt"; save_vector(filename.c_str(), error_m_gr_avg);
        filename = out_dir; filename += "/convergence/error_m_dd_avg.txt"; save_vector(filename.c_str(), error_m_dd_avg);
        filename = out_dir; filename += "/convergence/error_m_ex_avg.txt"; save_vector(filename.c_str(), error_m_ex_avg);

        filename = out_dir; filename += "/convergence/error_m_sl_max.txt"; save_vector(filename.c_str(), error_m_sl_max);
        filename = out_dir; filename += "/convergence/error_m_gr_max.txt"; save_vector(filename.c_str(), error_m_gr_max);
        filename = out_dir; filename += "/convergence/error_m_dd_max.txt"; save_vector(filename.c_str(), error_m_dd_max);
        filename = out_dir; filename += "/convergence/error_m_ex_max.txt"; save_vector(filename.c_str(), error_m_ex_max);

        filename = out_dir; filename += "/convergence/error_p_sl_all.txt"; save_vector(filename.c_str(), error_sl_p_arr);
        filename = out_dir; filename += "/convergence/error_p_gr_all.txt"; save_vector(filename.c_str(), error_gr_p_arr);
        filename = out_dir; filename += "/convergence/error_p_dd_all.txt"; save_vector(filename.c_str(), error_dd_p_arr);
        filename = out_dir; filename += "/convergence/error_p_ex_all.txt"; save_vector(filename.c_str(), error_ex_p_arr);

        filename = out_dir; filename += "/convergence/error_p_sl_one.txt"; save_vector(filename.c_str(), error_p_sl_one);
        filename = out_dir; filename += "/convergence/error_p_gr_one.txt"; save_vector(filename.c_str(), error_p_gr_one);
        filename = out_dir; filename += "/convergence/error_p_dd_one.txt"; save_vector(filename.c_str(), error_p_dd_one);
        filename = out_dir; filename += "/convergence/error_p_ex_one.txt"; save_vector(filename.c_str(), error_p_ex_one);

        filename = out_dir; filename += "/convergence/error_p_sl_avg.txt"; save_vector(filename.c_str(), error_p_sl_avg);
        filename = out_dir; filename += "/convergence/error_p_gr_avg.txt"; save_vector(filename.c_str(), error_p_gr_avg);
        filename = out_dir; filename += "/convergence/error_p_dd_avg.txt"; save_vector(filename.c_str(), error_p_dd_avg);
        filename = out_dir; filename += "/convergence/error_p_ex_avg.txt"; save_vector(filename.c_str(), error_p_ex_avg);

        filename = out_dir; filename += "/convergence/error_p_sl_max.txt"; save_vector(filename.c_str(), error_p_sl_max);
        filename = out_dir; filename += "/convergence/error_p_gr_max.txt"; save_vector(filename.c_str(), error_p_gr_max);
        filename = out_dir; filename += "/convergence/error_p_dd_max.txt"; save_vector(filename.c_str(), error_p_dd_max);
        filename = out_dir; filename += "/convergence/error_p_ex_max.txt"; save_vector(filename.c_str(), error_p_ex_max);

        filename = out_dir; filename += "/convergence/cond_num_all.txt"; save_vector(filename.c_str(), cond_num_arr);
        filename = out_dir; filename += "/convergence/cond_num_one.txt"; save_vector(filename.c_str(), cond_num_one);
        filename = out_dir; filename += "/convergence/cond_num_avg.txt"; save_vector(filename.c_str(), cond_num_avg);
        filename = out_dir; filename += "/convergence/cond_num_max.txt"; save_vector(filename.c_str(), cond_num_max);

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