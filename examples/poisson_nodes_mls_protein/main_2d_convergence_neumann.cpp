/*
 *
 *                                  Solves the Poisson equation:
 *                            -  div ( mu(x,y,z) * grad(Hodge) ) = rhs with Neumann boundary conditions.
 *
 * mu    (x,y,z) is the variable coefficient of the Poisson's operator, which corresponds to:
 *                  . the inverse of the density coefficient in the Navier-Stokes projection step,
 * rhs           is the right-hand side that is computed automatically when considering an exact solution.
 *
 * The following describes how to set up a problem in the function set_example():
 *
 * The exact solution hodge(x,y,z) we consider is defined by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_hodge_m" ("which_hodge_p"), which refers to one of the available functional forms
 *         defined in the class "hodge_choices_cf_t".
 *      b- Specifying the scalar "hodge_multiplier_m" multiplying the functional form.
 *      Note: the class hodge_choices_cf_t also defines the solution's gradient (ddx_hodge, ddy_hodge, ddz_hodge) and its
 *      Laplacian div( grad(hodge) ).
 *      ex:
 *          . hodge in the negative domain is defined as:
 *            hodge_choices_cf_t hodge_m_cf(VAL, which_hodge_m.val, hodge_multiplier_m.val);
 *          . ddx_hodge in the negative domain is defined as:
 *            hodge_choices_cf_t ddx_hodge_m_cf(DDX, which_hodge_m.val, hodge_multiplier_m.val);
 *          . ddy_hodge in the negative domain is defined as:
 *            hodge_choices_cf_t ddy_hodge_m_cf(DDY, which_hodge_m.val, hodge_multiplier_m.val);
 *          . Laplace(hodge) in the negative domain is defined as:
 *            hodge_choices_cf_t Laplace_hodge_m_cf(LAP, which_hodge_m.val, hodge_multiplier_m.val);
 *
 * The coefficient mu(x,y,z) > 0 is set by:
 *      a- Specifying its functional form in the negative (positive) domain with the parameter
 *         "which_mu_m" ("which_mu_p"), which refers to one of the available functional forms
 *         defined in the class "mu_choices_cf_t".
 *      b- Specifying the scalar "mu_multiplier_m" multiplying the functional form.
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
 *
 * NOTE: For a given exact solution, the rhs of the elliptic equation, of the Neumann, of the Dirichlet, of
 * the Robin boundary, and of the jump conditions are automatically computed.
 *
 * Run the program with the -help flag to see the available options.
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
#include <cstdio>

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


#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX


using namespace std;

const int num_bdry_max = 4;

param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<int> px(pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<int> py(pl, 0, "py", "Periodicity in the y-direction (0/1)");
param_t<int> pz(pl, 0, "pz", "Periodicity in the z-direction (0/1)");

param_t<int> nx(pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny(pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz(pl, 1, "nz", "Number of trees in the z-direction");

param_t<double> xmin(pl, -1, "xmin", "Box xmin");
param_t<double> ymin(pl, -1, "ymin", "Box ymin");
param_t<double> zmin(pl, -1, "zmin", "Box zmin");

param_t<double> xmax(pl, 1, "xmax", "Box xmax");
param_t<double> ymax(pl, 1, "ymax", "Box ymax");
param_t<double> zmax(pl, 1, "zmax", "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
param_t<int>    lmin (pl, 1, "lmin", "Min level of the tree");
param_t<int>    lmax (pl, 2, "lmax", "Max level of the tree");

param_t<int>    num_splits (pl, 5, "num_splits", "Number of recursive splits");
#else
param_t<int> lmin(pl, 3, "lmin", "Min level of the tree");
param_t<int> lmax(pl, 3, "lmax", "Max level of the tree");

param_t<int> num_splits(pl, 7, "num_splits", "Number of recursive splits");
#endif

param_t<double> lip(pl, 1.2, "lip", "Transition width from coarse grid to fine grid (a.k.a. Lipschitz constant)");
param_t<double> band(pl, 4.0, "band","Width of the uniform band around boundaries and interfaces (in lengths of smallest quadrants)");

param_t<bool> refine_strict(pl, false, "refine_strict", "Refines every cell starting from the coarsest case if yes");
// Refine according to level-set (uniform if regular rectangular domain):
param_t<bool> refine_rand(pl, false, "refine_rand", "Add randomness into adaptive grid");
param_t<bool> enforce_graded_grid(pl, true, "enforce_graded_grid", "Enforce 1:2 ratio for adaptive grid");
// Refine random:
//param_t<bool>   refine_rand    (pl, true, "refine_rand",    "Add randomness into adaptive grid");
//param_t<bool>   enforce_graded_grid   (pl, false , "enforce_graded_grid",   "Enforce 1:2 ratio for adaptive grid");
param_t<bool> coarse_outside(pl, false, "coarse_outside", "Use the coarsest possible grid outside the domain (0/1)");
param_t<int> expand_ghost(pl, false, "expand_ghost", "Number of ghost layer expansions");
param_t<int> iter_start(pl, false, "iter_start", "Skip n first iterations (for debugging)");

//-------------------------------------
// equation parameters
//-------------------------------------
param_t<int> which_hodge_m(pl, 0, "which_hodge_m","Case number (id) of the exact hodge solution we consider in the negative domain - this specifies which of the cases defined in the class hodge_choices_cf_t we use.");
param_t<int> which_hodge_p(pl, 0, "which_hodge_p","Case number (id) of the exact hodge solution we consider in the positive domain - this specifies which of the cases defined in the class hodge_choices_cf_t we use.");
param_t<double> hodge_multiplier_m(pl, 1, "hodge_multiplier_m","Multiplier value of the exact hodge solution we consider in the negative domain");
param_t<double> hodge_multiplier_p(pl, 1, "hodge_multiplier_p","Multiplier value of the exact hodge solution we consider in the positive domain");

param_t<int> which_mu_m(pl, 0, "which_mu_m", "Case number (id) of mu(x,y,z) in the negative domain");
param_t<int> which_mu_p(pl, 0, "which_mu_p", "Case number (id) of mu(x,y,z) in the positive domain");
param_t<double> mu_multiplier_m(pl, 1, "mu_multiplier_m", "Multiplier in front of mu(x,y,z) in the negative domain");
param_t<double> mu_multiplier_p(pl, 1, "mu_multiplier_p", "Multiplier in front of mu(x,y,z) in the positive domain");

param_t<int> rhs_vec_value_m(pl, 0, "rhs_vec_value_m", "Source term in negative domain: 0 - automatic (method of manufactured solutions), 1 - zero");
param_t<int> rhs_vec_value_p(pl, 0, "rhs_vec_value_p", "Source term in positive domain: 0 - automatic (method of manufactured solutions), 1 - zero");

param_t<int> num_bdry(pl, 0, "num_bdry", "Number of domain boundaries");

// boundary geometry parameters
param_t<int> wc_type(pl, NEUMANN, "wc_type", "Type of boundary conditions on the walls");

param_t<bool> is_bdry_00_present(pl, false, "is_bdry_00_present", "Turn on/off boundary no. 0");
param_t<bool> is_bdry_01_present(pl, false, "is_bdry_01_present", "Turn on/off boundary no. 1");
param_t<bool> is_bdry_02_present(pl, false, "is_bdry_02_present", "Turn on/off boundary no. 2");
param_t<bool> is_bdry_03_present(pl, false, "is_bdry_03_present", "Turn on/off boundary no. 3");

param_t<int> which_geometry_for_bdry_00(pl, 0, "which_geometry_for_bdry_00", "Geometry of boundary no. 0");
param_t<int> which_geometry_for_bdry_01(pl, 0, "which_geometry_for_bdry_01", "Geometry of boundary no. 1");
param_t<int> which_geometry_for_bdry_02(pl, 0, "which_geometry_for_bdry_02", "Geometry of boundary no. 2");
param_t<int> which_geometry_for_bdry_03(pl, 0, "which_geometry_for_bdry_03", "Geometry of boundary no. 3");

param_t<int> bdry_00_opn(pl, MLS_INTERSECTION, "bdry_00_opn", "Operation used to add boundary no. 0: 0 - intersection, 1 - union");
param_t<int> bdry_01_opn(pl, MLS_INTERSECTION, "bdry_01_opn", "Operation used to add boundary no. 1: 0 - intersection, 1 - union");
param_t<int> bdry_02_opn(pl, MLS_INTERSECTION, "bdry_02_opn", "Operation used to add boundary no. 2: 0 - intersection, 1 - union");
param_t<int> bdry_03_opn(pl, MLS_INTERSECTION, "bdry_03_opn", "Operation used to add boundary no. 3: 0 - intersection, 1 - union");

param_t<int> bdry_00_bc_type(pl, DIRICHLET, "bdry_00_bc_type", "Type of boundary conditions on boundary no. 0");
param_t<int> bdry_01_bc_type(pl, DIRICHLET, "bdry_01_bc_type", "Type of boundary conditions on boundary no. 1");
param_t<int> bdry_02_bc_type(pl, DIRICHLET, "bdry_02_bc_type", "Type of boundary conditions on boundary no. 2");
param_t<int> bdry_03_bc_type(pl, DIRICHLET, "bdry_03_bc_type", "Type of boundary conditions on boundary no. 3");

param_t<int> example(pl, 34, "example", "Predefined example:\n"
                                        "34 - poisson with Neumann on regular rectangular domain\n");

//-------------------------------------
// solver parameters
//-------------------------------------
param_t<int> integration_order(pl, 2, "integration_order","Integration order for finite volume discretization (1 - linear, 2 - quadratic)");
param_t<bool> fv_scheme(pl, true, "fv_scheme","Scheme for finite volume discretization on boundaries: 0 - symmetric, 1 - superconvergent");

param_t<bool> store_finite_volumes(pl, false, "store_finite_volumes", "");
param_t<bool> apply_bc_pointwise(pl, true, "apply_bc_pointwise", "");
param_t<bool> use_centroid_always(pl, false, "use_centroid_always", "");
param_t<bool> sample_bc_node_by_node(pl, false, "sample_bc_node_by_node", "");

// for symmetric finite volume scheme:
param_t<bool> taylor_correction(pl, true, "taylor_correction","Use Taylor correction to approximate Robin term (symmetric scheme)");
param_t<bool> kink_special_treatment(pl, true, "kink_special_treatment","Use the special treatment for kinks (symmetric scheme)");

//-------------------------------------
// level-set representation parameters
//-------------------------------------
param_t<bool> reinit_level_set(pl, true, "reinit_level_set", "Reinitialize level-set function");

//-------------------------------------
// convergence study parameters
//-------------------------------------
param_t<int> extend_solution(pl, 2, "extend_solution","Extend solution after solving: 0 - no extension, 1 - extend using normal derivatives, 2 - extend using all derivatives");
param_t<bool> compute_grad_between(pl, false, "compute_grad_between", "Computes gradient between points if yes");
param_t<bool> scale_errors(pl, false, "scale_errors", "Scale errors by max solution/gradient value");
param_t<bool> use_nonzero_guess(pl, false, "use_nonzero_guess", "");
param_t<double> extension_band_extend(pl, 60, "extension_band_extend", "");
param_t<double> extension_band_compute(pl, 6, "extension_band_compute", "");
param_t<double> extension_band_check(pl, 6, "extension_band_check", "");
param_t<int> extension_iterations(pl, 100, "extension_iterations", "");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool> save_vtk(pl, true, "save_vtk", "Save the p4est in vtk format");
param_t<bool> save_params(pl, false, "save_params", "Save list of entered parameters");
param_t<bool> save_domain(pl, true, "save_domain","Save the reconstruction of an irregular domain (works only in serial!)");
param_t<bool> save_convergence(pl, false, "save_convergence", "Save convergence results");


void set_example(int example_)
{
   switch (example_) {

      case 34: // To test projections step - div ( grad(hodge) ) = rhs in standard rectangular domain with Neumann bc

         // mu = constant (i.e., id = 0) = 1:
         which_mu_m.val = 0; mu_multiplier_m.val = 1;
         which_mu_p.val = 0; mu_multiplier_p.val = 1;

         // Exact hodge is case 0 in hodge_choices_cf_t: hodge_multiplier * sin(x) * cos(y);
         which_hodge_m.val = 0; hodge_multiplier_m.val = 1;
         which_hodge_p.val = 0; hodge_multiplier_p.val = 1;

         // Define the geometry (here we solve for hodge in standard rectangular domain, i.e. which_geometry_for_bdry_00.val = 0):
         num_bdry.val = 0;
         is_bdry_00_present.val = false;
//         which_geometry_for_bdry_00.val = 0;    // 0: entire domain.    1: circle.
//         bdry_00_opn.val = MLS_INT;
//         bdry_00_bc_type.val = NEUMANN;

         break;

      default:
         throw std::invalid_argument("Invalid case in function set_example(...)");
   }
}


bool *bdry_present_all[] = {&is_bdry_00_present.val,
                            &is_bdry_01_present.val,
                            &is_bdry_02_present.val,
                            &is_bdry_03_present.val};

int *bdry_opn_all[] = {&bdry_00_opn.val,
                       &bdry_01_opn.val,
                       &bdry_02_opn.val,
                       &bdry_03_opn.val};


int *bc_type_all[] = {&bdry_00_bc_type.val,
                      &bdry_01_bc_type.val,
                      &bdry_02_bc_type.val,
                      &bdry_03_bc_type.val};

// MU COEFFICIENT CHOICES:
class mu_choices_cf_t : public CF_DIM {
   int *n;
   double *mag;
   cf_value_type_t what;
public:
   mu_choices_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}

   double operator()(DIM(double x, double y, double z)) const override {
      switch (*n) {
         case 0:
            switch (what) {
               case VAL:
                  return (*mag);
               case DDX:
               case DDY:
                  return 0.;
#ifdef P4_TO_P8
                  case DDZ: return 0.;
#endif
               default:
                  throw std::invalid_argument("Invalid \"what\" in case 0 of class mu_choices_cf_t");
            }
         default:
            throw std::invalid_argument("Invalid case of class mu_choices_cf_t");
      }
   }
};

mu_choices_cf_t mu_m_cf(VAL, which_mu_m.val, mu_multiplier_m.val);
mu_choices_cf_t mu_p_cf(VAL, which_mu_p.val, mu_multiplier_p.val);
mu_choices_cf_t DIM(ddx_mu_m_cf(DDX, which_mu_m.val, mu_multiplier_m.val),
                    ddy_mu_m_cf(DDY, which_mu_m.val, mu_multiplier_m.val),
                    ddz_mu_m_cf(DDZ, which_mu_m.val, mu_multiplier_m.val));
mu_choices_cf_t DIM(ddx_mu_p_cf(DDX, which_mu_p.val, mu_multiplier_p.val),
                    ddy_mu_p_cf(DDY, which_mu_p.val, mu_multiplier_p.val),
                    ddz_mu_p_cf(DDZ, which_mu_p.val, mu_multiplier_p.val));


// EXACT HODGE CHOICES:
class hodge_choices_cf_t : public CF_DIM {
public:
   int *n;
   double *mag;
   cf_value_type_t what;

   hodge_choices_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}

   double operator()(DIM(double x, double y, double z)) const override {
      switch (*n) {
         case 0:
            switch (what) {
#ifdef P4_TO_P8
               throw std::invalid_argument("Invalid case of class hodge_choices_cf_t in 3D");
#else
               case VAL:
                  return -(*mag) * (x * x * x / 3. - PI * x * x / 2.) * (y * y * y / 3. - PI * y * y / 2.);
                  //return (*mag)*sin(x)*cos(y);
               case DDX:
                  return -(*mag) * (x * x - PI * x) * (y * y * y / 3. - PI * y * y / 2.);
                  // return (*mag)*cos(x)*cos(y);
               case DDY:
                  return -(*mag) * (y * y - PI * y) * (x * x * x / 3. - PI * x * x / 2.);
                  // return -(*mag)*sin(x)*sin(y);
               case LAP:
                  return -(*mag) * (2 * x - PI) * (y * y * y / 3. - PI * y * y / 2.)
                         - (*mag) * (2 * y - PI) * (x * x * x / 3. - PI * x * x / 2.);
                   //return -2*(*mag)*sin(x)*cos(y);


#endif  //  (PI - 2.*x)*(y * y * y / 3. - PI * y * y / 2.) + (PI - 2.*x)*(x * x * x / 3. - PI * x * x / 2.)
               default:
                  throw std::invalid_argument("Invalid \"what\" in case 0 of class hodge_choices_cf_t");
            }
         default:
            throw std::invalid_argument("Invalid case of class hodge_choices_cf_t");
      }
   }
};

hodge_choices_cf_t hodge_m_cf(VAL, which_hodge_m.val, hodge_multiplier_m.val), hodge_p_cf(VAL, which_hodge_p.val,
                                                                                          hodge_multiplier_p.val);
hodge_choices_cf_t laplace_hodge_m_cf(LAP, which_hodge_m.val, hodge_multiplier_m.val), laplace_hodge_p_cf(LAP,
                                                                                                          which_hodge_p.val,
                                                                                                          hodge_multiplier_p.val);
hodge_choices_cf_t DIM(ddx_hodge_m_cf(DDX, which_hodge_m.val, hodge_multiplier_m.val),
                       ddy_hodge_m_cf(DDY, which_hodge_m.val, hodge_multiplier_m.val),
                       ddz_hodge_m_cf(DDZ, which_hodge_m.val, hodge_multiplier_m.val));
hodge_choices_cf_t DIM(ddx_hodge_p_cf(DDX, which_hodge_p.val, hodge_multiplier_p.val),
                       ddy_hodge_p_cf(DDY, which_hodge_p.val, hodge_multiplier_p.val),
                       ddz_hodge_p_cf(DDZ, which_hodge_p.val, hodge_multiplier_p.val));

// RHS (AUTOMATICALLY COMPUTED FOR EXACT HODGE SOLUTIONS):
class rhs_m_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      switch (rhs_vec_value_m.val) {
         case 0:
            return - mu_m_cf(DIM(x, y, z)) * laplace_hodge_m_cf(DIM(x, y, z))
                   - SUMD(ddx_mu_m_cf(DIM(x, y, z)) * ddx_hodge_m_cf(DIM(x, y, z)),
                          ddy_mu_m_cf(DIM(x, y, z)) * ddy_hodge_m_cf(DIM(x, y, z)),
                          ddz_mu_m_cf(DIM(x, y, z)) * ddz_hodge_m_cf(DIM(x, y, z)));
         default:
            throw std::invalid_argument("Invalid case of class rhs_m_cf_t");
      }
   }
} rhs_m_cf;

class rhs_p_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      switch (rhs_vec_value_p.val) {
         case 0:
            return - mu_p_cf(DIM(x, y, z)) * laplace_hodge_p_cf(DIM(x, y, z))
                   - SUMD(ddx_mu_p_cf(DIM(x, y, z)) * ddx_hodge_p_cf(DIM(x, y, z)),
                          ddy_mu_p_cf(DIM(x, y, z)) * ddy_hodge_p_cf(DIM(x, y, z)),
                          ddz_mu_p_cf(DIM(x, y, z)) * ddz_hodge_p_cf(DIM(x, y, z)));
         default:
            throw std::invalid_argument("Invalid case of class rhs_p_cf_t");
      }
   }
} rhs_p_cf;

// DOMAIN GEOMETRY CHOICES:
class bdry_phi_choices_cf_t : public CF_DIM {
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
               _CODE(case VAL:
                        return circle.phi(DIM(x, y, z)));
               XCODE(case DDX:
                        return circle.phi_x(DIM(x, y, z)));
               YCODE(case DDY:
                        return circle.phi_y(DIM(x, y, z)));
                  ZCODE(case DDZ:
                           return circle.phi_z(DIM(x, y, z)));
               default:
                  throw std::invalid_argument("Invalid \"what\" in case 1 of class bdry_phi_choices_cf_t");
            }
         }
            break;
      }

      // default values
      switch (what) {
         _CODE(case VAL:
                  return 1);
         XCODE(case DDX:
                  return 0);
         YCODE(case DDY:
                  return 0);
            ZCODE(case DDZ:
                     return 0);
         default:
            throw std::invalid_argument("Invalid \"what\" in default values of class bdry_phi_choices_cf_t");
      }
   }
};

bdry_phi_choices_cf_t bdry_phi_cf_all[] = {bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_00.val),
                                           bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_01.val),
                                           bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_02.val),
                                           bdry_phi_choices_cf_t(VAL, which_geometry_for_bdry_03.val)};

bdry_phi_choices_cf_t bdry_phi_x_cf_all[] = {bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_00.val),
                                             bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_01.val),
                                             bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_02.val),
                                             bdry_phi_choices_cf_t(DDX, which_geometry_for_bdry_03.val)};

bdry_phi_choices_cf_t bdry_phi_y_cf_all[] = {bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_00.val),
                                             bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_01.val),
                                             bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_02.val),
                                             bdry_phi_choices_cf_t(DDY, which_geometry_for_bdry_03.val)};
#ifdef P4_TO_P8
bdry_phi_choices_cf_t bdry_phi_z_cf_all[] = { bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_00.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_01.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_02.val),
                                      bdry_phi_choices_cf_t(DDZ, which_geometry_for_bdry_03.val) };
#endif

// the effective LSF (initialized in main!)
mls_eff_cf_t bdry_phi_eff_cf;
mls_eff_cf_t infc_phi_eff_cf;

class phi_eff_cf_t : public CF_DIM {
   CF_DIM *bdry_phi_cf_;
public:
   explicit phi_eff_cf_t(CF_DIM &bdry_phi_cf) : bdry_phi_cf_(&bdry_phi_cf) {}

   double operator()(DIM(double x, double y, double z)) const override {
      return (*bdry_phi_cf_)(DIM(x, y, z));
   }
} phi_eff_cf(bdry_phi_eff_cf);

class mu_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      return infc_phi_eff_cf(DIM(x, y, z)) >= 0 ? mu_p_cf(DIM(x, y, z)) : mu_m_cf(DIM(x, y, z));
   }
} mu_cf;

class hodge_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      return infc_phi_eff_cf(DIM(x, y, z)) >= 0 ? hodge_p_cf(DIM(x, y, z)) : hodge_m_cf(DIM(x, y, z));
   }
} hodge_cf;

class ddx_hodge_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      return infc_phi_eff_cf(DIM(x, y, z)) >= 0 ? ddx_hodge_p_cf(DIM(x, y, z)) : ddx_hodge_m_cf(DIM(x, y, z));
   }
} ddx_hodge_cf;

class ddy_hodge_cf_t : public CF_DIM {
public:
   double operator()(DIM(double x, double y, double z)) const override {
      return infc_phi_eff_cf(DIM(x, y, z)) >= 0 ? ddy_hodge_p_cf(DIM(x, y, z)) : ddy_hodge_m_cf(DIM(x, y, z));
   }
} ddy_hodge_cf;

#ifdef P4_TO_P8
class ddz_hodge_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const override {
    return infc_phi_eff_cf(DIM(x,y,z)) >= 0 ? ddz_hodge_p_cf(DIM(x,y,z)) : ddz_hodge_m_cf(DIM(x,y,z));
  }
} ddz_hodge_cf;
#endif

// BC VALUES
class bc_value_cf_t : public CF_DIM {
   BoundaryConditionType *bc_type_;
   CF_DIM DIM(*phi_x_cf_,
              *phi_y_cf_,
              *phi_z_cf_);
public:
   bc_value_cf_t(BoundaryConditionType *bc_type,
                 CF_DIM *bc_coeff_cf,
                 DIM(CF_DIM * phi_x_cf,
                     CF_DIM * phi_y_cf,
                     CF_DIM * phi_z_cf))
        : bc_type_(bc_type),
          DIM(phi_x_cf_(phi_x_cf),
              phi_y_cf_(phi_y_cf),
              phi_z_cf_(phi_z_cf)) {}

   double operator()(DIM(double x, double y, double z)) const override {
      switch (*bc_type_) {
         case DIRICHLET:
            return hodge_cf(DIM(x, y, z));
         case NEUMANN: {
            double DIM(nx = (*phi_x_cf_)(DIM(x, y, z)),
                       ny = (*phi_y_cf_)(DIM(x, y, z)),
                       nz = (*phi_z_cf_)(DIM(x, y, z)));

            double norm = sqrt(SUMD(nx * nx, ny * ny, nz * nz));
            nx /= norm;
            ny /= norm; P8(nz /= norm);

            return mu_cf(DIM(x, y, z)) * SUMD(nx * ddx_hodge_cf(DIM(x, y, z)),
                                              ny * ddy_hodge_cf(DIM(x, y, z)),
                                              nz * ddz_hodge_cf(DIM(x, y, z)));
         }
         default:
            throw std::invalid_argument("Invalid case of class bc_value_cf_t");
      }
   }
};

bc_value_cf_t bc_value_cf_all[] = {bc_value_cf_t((BoundaryConditionType *) &bdry_00_bc_type.val, &zero_cf,
                                                 DIM(&bdry_phi_x_cf_all[0], &bdry_phi_y_cf_all[0],
                                                     &bdry_phi_z_cf_all[0])),
                                   bc_value_cf_t((BoundaryConditionType *) &bdry_01_bc_type.val, &zero_cf,
                                                 DIM(&bdry_phi_x_cf_all[1], &bdry_phi_y_cf_all[1],
                                                     &bdry_phi_z_cf_all[1])),
                                   bc_value_cf_t((BoundaryConditionType *) &bdry_02_bc_type.val, &zero_cf,
                                                 DIM(&bdry_phi_x_cf_all[2], &bdry_phi_y_cf_all[2],
                                                     &bdry_phi_z_cf_all[2])),
                                   bc_value_cf_t((BoundaryConditionType *) &bdry_03_bc_type.val, &zero_cf,
                                                 DIM(&bdry_phi_x_cf_all[3], &bdry_phi_y_cf_all[3],
                                                     &bdry_phi_z_cf_all[3]))};



class bc_wall_type_t : public WallBCDIM {
public:
   BoundaryConditionType operator()(DIM(double, double, double)) const override {
      return (BoundaryConditionType) wc_type.val;
   }
} bc_wall_type;

class bc_wall_value_t : public CF_DIM
{
public:
    double operator()(DIM(double x, double y, double z)) const
    {
        if (wc_type.val == DIRICHLET) {
            return hodge_cf(DIM(x, y, z));
        } else {
            double dists[P4EST_FACES] = {
                    DIMPM(
                            ABS(x - xmin.val), ABS(x - xmax.val),
                            ABS(y - ymin.val), ABS(y - ymax.val),
                            ABS(z - zmin.val), ABS(z - zmax.val)
                    )
            };

            double closest_dist = dists[0];
            uint8_t closest_wall = 0;
            for (uint8_t wall_ind = 1; wall_ind < P4EST_FACES; ++wall_ind) {
                if (dists[wall_ind] < closest_dist) {
                    closest_dist = dists[wall_ind];
                    closest_wall = wall_ind;
                }
            }

            switch (closest_wall) {
                case 0: return -mu_cf(DIM(x, y, z)) * ddx_hodge_cf(DIM(x, y, z));
                case 1: return mu_cf(DIM(x, y, z)) * ddx_hodge_cf(DIM(x, y, z));
                case 2: return -mu_cf(DIM(x, y, z)) * ddy_hodge_cf(DIM(x, y, z));
                case 3: return mu_cf(DIM(x, y, z)) * ddy_hodge_cf(DIM(x, y, z));
#ifdef P4_TO_P8
                    case 4: return -mu_cf(DIM(x, y, z)) * uz_cf(DIM(x, y, z));
      case 5: return mu_cf(DIM(x, y, z)) * uz_cf(DIM(x, y, z));
#endif
                default: throw;
            }

        }
    }
} bc_wall_value;

int main(int argc, char *argv[]) {

   PetscErrorCode ierr;
   int mpiret;

   // mpi
   mpi_environment_t mpi{};
   mpi.init(argc, argv);

   // prepare output directories
   const char *out_dir = getenv("OUT_DIR");

   if (!out_dir &&
       (save_vtk.val ||
        save_domain.val ||
        save_convergence.val )) {
      ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save results\n");
      return -1;
   }

   if (save_vtk.val) {
      std::ostringstream command;
      command << "mkdir -p " << out_dir << "/vtu";
      if (mpi.rank() == 0) cout << "executing: " << command.str().c_str() << endl;
      int ret_sys = system(command.str().c_str());
      if (ret_sys < 0)
         throw std::invalid_argument("could not create OUT_DIR/vtu directory");
   }

   if (save_domain.val) {
      std::ostringstream command;
      command << "mkdir -p " << out_dir << "/geometry";
      int ret_sys = system(command.str().c_str());
      if (ret_sys < 0)
         throw std::invalid_argument("could not create OUT_DIR/geometry directory");
   }

   if (save_convergence.val) {
      std::ostringstream command;
      int ret_sys = system(command.str().c_str());
      if (ret_sys < 0)
         throw std::invalid_argument("could not create OUT_DIR/convergence directory");
   }

   // parse command line arguments
   cmdParser cmd; pl.initialize_parser(cmd); cmd.parse(argc, argv);

   example.set_from_cmd(cmd); set_example(example.val);

   pl.set_from_cmd_all(cmd);       // todo: what is this after set_example?

   if (mpi.rank() == 0) pl.print_all();
   if (mpi.rank() == 0 && save_params.val) {
      std::ostringstream file;
      file << out_dir << "/parameters.dat";
      pl.save_all(file.str().c_str());
   }


   // Build the effective level-set function for the boundaries:
   for (int i = 0; i < num_bdry_max; ++i) if (*bdry_present_all[i])
         bdry_phi_eff_cf.add_domain(bdry_phi_cf_all[i], (mls_opn_t) *bdry_opn_all[i]);

   const int periodicity[] = {DIM(px(), py(), pz())};
   const int num_trees[] = {DIM(nx(), ny(), nz())};
   const double grid_xyz_min[] = {DIM(xmin(), ymin(), zmin())};
   const double grid_xyz_max[] = {DIM(xmax(), ymax(), zmax())};

   // vectors to store convergence results
   vector<double> lvl_arr, h_arr, mu_arr;

   vector<double> error_hodge_m_arr;
   vector<double> error_hodge_extrapolated_m_arr;
   vector<double> error_gradient_hodge_m_arr;

   vector<double> error_hodge_p_arr;
   vector<double> error_hodge_extrapolated_p_arr;
   vector<double> error_gradient_hodge_p_arr;

   vector<double> error_unp1_m_arr, error_vnp1_m_arr;
   vector<double> error_unp1_p_arr, error_vnp1_p_arr;

   parStopWatch w;
   w.start("total time");

   p4est_connectivity_t *connectivity;
   my_p4est_brick_t brick;

   p4est_t *p4est;
   p4est_nodes_t *nodes;
   p4est_ghost_t *ghost;

   int iteration = -1;
   int file_idx = -1;

   for (int iter = 0; iter < num_splits(); ++iter) {
      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d.\n", lmin() + iter, lmax() + iter); CHKERRXX(ierr);

      double dxyz[3] = {(grid_xyz_max[0] - grid_xyz_min[0]) / pow(2., (double) lmax() + iter),
                        (grid_xyz_max[1] - grid_xyz_min[1]) / pow(2., (double) lmax() + iter),
                        (grid_xyz_max[2] - grid_xyz_min[2]) / pow(2., (double) lmax() + iter)};


      double dxyz_m = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));

      mu_arr.push_back(mu_multiplier_m());
      h_arr.push_back(dxyz_m);
      lvl_arr.push_back(lmax() + iter);

      ierr = PetscPrintf(mpi.comm(), "Level %2d / %2d. Sub split %2d (lvl %5.2f / %5.2f).\n", lmin() + iter,
                         lmax() + iter, 0, lmin() + iter, lmax() + iter);
      CHKERRXX(ierr);


      iteration++;

      if (iteration < iter_start()) continue;

      file_idx++;

      connectivity = my_p4est_brick_new(num_trees, grid_xyz_min, grid_xyz_max, &brick,
                                        periodicity);
      p4est = my_p4est_new(mpi.comm(), connectivity, 0, nullptr, nullptr);

      if (refine_strict()) {
         splitting_criteria_cf_t data_tmp(lmin(), lmax(), &phi_eff_cf, lip(), band());
         p4est->user_pointer = (void *) (&data_tmp);

         my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, nullptr);
         my_p4est_partition(p4est, P4EST_FALSE, nullptr);
         for (int i = 0; i < iter; ++i) {
            my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, nullptr);
            my_p4est_partition(p4est, P4EST_FALSE, nullptr);
         }
      } else {
         splitting_criteria_cf_t data_tmp(lmin() + iter, lmax() + iter, &phi_eff_cf, lip(), band());
         p4est->user_pointer = (void *) (&data_tmp);

         for (int i = 0; i < lmax() + iter; ++i) {
            my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, nullptr);
            my_p4est_partition(p4est, P4EST_FALSE, nullptr);
         }
      }
      // macromesh has been generated at this point.

      splitting_criteria_cf_t data(lmin() + iter, lmax() + iter, &phi_eff_cf, lip(), band());
      p4est->user_pointer = (void *) (&data);

      if (refine_rand()) {
         if (mpi.rank() == 0) cout << "Refining the grid randomly\n";
         my_p4est_refine(p4est, P4EST_TRUE, refine_random, nullptr);
      }
      if (enforce_graded_grid()) {
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

      my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
      my_p4est_node_neighbors_t ngbd_n(&hierarchy, nodes);
      ngbd_n.init_neighbors();

      my_p4est_level_set_t ls(&ngbd_n);

      dxyz_min(p4est, dxyz);
      double dxyz_max = MAX(DIM(dxyz[0], dxyz[1], dxyz[2]));
      double diag = sqrt(SUMD(dxyz[0] * dxyz[0], dxyz[1] * dxyz[1], dxyz[2] * dxyz[2]));

      // sample boundary level-set functions
      Vec bdry_phi_vec_all[num_bdry_max];

      // Perturbing domain boundaries and reinitializing
      for (int i = 0; i < num_bdry_max; ++i)
         if (*bdry_present_all[i]) {
            ierr = VecCreateGhostNodes(p4est, nodes, &bdry_phi_vec_all[i]); CHKERRXX(ierr);
            sample_cf_on_nodes(p4est, nodes, bdry_phi_cf_all[i], bdry_phi_vec_all[i]);

            if (reinit_level_set()) {
               ls.reinitialize_1st_order_time_2nd_order_space(bdry_phi_vec_all[i], 20);
            }
         }

      /***********************************************************************************
       *
       * SOLVE FOR THE HODGE VARIABLE:
       *
       ***********************************************************************************/

      //initializing vectors needed to set up the problem
      Vec vec_mu_m, vec_mu_p, vec_rhs_m, vec_rhs_p;
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_mu_m ); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_mu_p ); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_rhs_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_rhs_p); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, rhs_m_cf, vec_rhs_m);
      sample_cf_on_nodes(p4est, nodes, rhs_p_cf, vec_rhs_p);
      sample_cf_on_nodes(p4est, nodes, mu_m_cf, vec_mu_m);
      sample_cf_on_nodes(p4est, nodes, mu_p_cf, vec_mu_p);

      Vec vec_hodge; double *ptr_hodge;    // hodge is the solution in the minus and plus domains.
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_hodge); CHKERRXX(ierr);

      // creating an object of the poisson solver which has to be setup:
      my_p4est_poisson_nodes_mls_t solver(&ngbd_n);

      // setting the Poisson solver:
      solver.set_use_centroid_always(use_centroid_always());
      solver.set_store_finite_volumes(store_finite_volumes());
      solver.set_use_sc_scheme(fv_scheme());
      solver.set_integration_order(integration_order());
      solver.set_lip(lip());

      // ADD BOUNDARIES, where we impose Neumann boundary condition for hodge:
      // In the add_boundary(), creates a structure boundary_conditions_t
      for (int i = 0; i < num_bdry_max; ++i) {
         if (*bdry_present_all[i]) {
            if (apply_bc_pointwise())
               solver.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i],
                                   DIM(nullptr, nullptr, nullptr), (BoundaryConditionType) *bc_type_all[i],
                                   zero_cf, zero_cf);
            else
               solver.add_boundary((mls_opn_t) *bdry_opn_all[i], bdry_phi_vec_all[i],
                                   DIM(nullptr, nullptr, nullptr), (BoundaryConditionType) *bc_type_all[i],
                                   bc_value_cf_all[i], zero_cf);
            if (mpi.rank() == 0)
               std::cout << "On bdry0" << i << ", we impose a " << (BoundaryConditionType) *bc_type_all[i]
                         << " boundary condition." << std::endl;
         }
      }

      solver.set_mu(vec_mu_m, DIM(nullptr, nullptr, nullptr),
                    vec_mu_p, DIM(nullptr, nullptr, nullptr));

      solver.set_wc(bc_wall_type, bc_wall_value);

      /***********************************************************************************
       *
       * PROJECTION - PART 1: set the right-hand side of the Hodge solver to - div(Ustar)
       *
       ***********************************************************************************/
      Vec vec_ustar_m; double *ptr_ustar_m;
      Vec vec_ustar_p; double *ptr_ustar_p;
      Vec vec_vstar_m; double *ptr_vstar_m;
      Vec vec_vstar_p; double *ptr_vstar_p;
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_ustar_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_ustar_p); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_vstar_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_vstar_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_ustar_m, &ptr_ustar_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_ustar_p, &ptr_ustar_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vstar_m, &ptr_vstar_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vstar_p, &ptr_vstar_p); CHKERRXX(ierr);

      Vec vec_unp1_m; double *ptr_unp1_m;
      Vec vec_unp1_p; double *ptr_unp1_p;
      Vec vec_vnp1_m; double *ptr_vnp1_m;
      Vec vec_vnp1_p; double *ptr_vnp1_p;
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_unp1_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_unp1_p); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_vnp1_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_vnp1_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_unp1_m, &ptr_unp1_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_unp1_p, &ptr_unp1_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vnp1_m, &ptr_vnp1_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vnp1_p, &ptr_vnp1_p); CHKERRXX(ierr);

      Vec vec_minus_div_Ustar_m; double *ptr_minus_div_Ustar_m;
      Vec vec_minus_div_Ustar_p; double *ptr_minus_div_Ustar_p;
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_minus_div_Ustar_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_minus_div_Ustar_p); CHKERRXX(ierr);

      double xyz[P4EST_DIM];
      foreach_local_node(n, nodes) {
         // TODO: We need to separate minus and plus solutions using a global level-set vector.
         // TODO: For now, we use the continuous function phi_eff_cf:
         node_xyz_fr_n(n, p4est, nodes, xyz);
         double x = xyz[0], y = xyz[1], z = xyz[2];
         if (phi_eff_cf(x, y) < 0) {
            ptr_ustar_m[n] = sin(x) * cos(y) + x * (PI - x) * y * y * (y / 3. - PI / 2.);
            ptr_vstar_m[n] = -sin(y) * cos(x) + y * (PI - y) * x * x * (x / 3. - PI / 2.);
            ptr_ustar_p[n] = 0;
            ptr_vstar_p[n] = 0;
         } else {
            ptr_ustar_m[n] = 0;
            ptr_vstar_m[n] = 0;
            ptr_ustar_p[n] = sin(x) * cos(y) + x * (PI - x) * y * y * (y / 3. - PI / 2.);
            ptr_vstar_p[n] = -sin(y) * cos(x) + y * (PI - y) * x * x * (x / 3. - PI / 2.);
         }
      }

      ierr = VecGhostUpdateBegin(vec_ustar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_ustar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_vstar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_vstar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_ustar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_ustar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_vstar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_vstar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


       ierr = VecGetArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
       ierr = VecGetArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);
       quad_neighbor_nodes_of_node_t qnnn{};
      // Compute -div(U*) = - ( ddx(u*) + ddy(v*) ):
      foreach_local_node(n, nodes) {
         auto *ni = (p4est_indep_t *) sc_array_index(&nodes->indep_nodes, n);
         ngbd_n.get_neighbors(n, qnnn);
         // TODO: We need to separate minus and plus solutions using a global level-set vector.
         // TODO: For now, we use the continuous function phi_eff_cf:
         node_xyz_fr_n(n, p4est, nodes, xyz);
         double x = xyz[0], y = xyz[1], z = xyz[2];
         if (phi_eff_cf(x, y) < 0) {
            ptr_minus_div_Ustar_m[n] = -(qnnn.dx_central(ptr_ustar_m) + qnnn.dy_central(ptr_vstar_m));
            if (ABS(ptr_minus_div_Ustar_m[n]+((PI - 2.*x)*(y * y * y / 3. - PI * y * y / 2.) + (PI - 2.*y)*(x * x * x / 3. - PI * x * x / 2.))) > 4e-4)
             {
                 cout << "calculated: " << ptr_minus_div_Ustar_m[n] << "\t exact: "
                      << -((PI - 2. * x) * (y * y * y / 3. - PI * y * y / 2.) +
                           (PI - 2. * y) * (x * x * x / 3. - PI * x * x / 2.)) << "\t error = "
                      << ABS(ptr_minus_div_Ustar_m[n] + ((PI - 2. * x) * (y * y * y / 3. - PI * y * y / 2.) +
                                                         (PI - 2. * y) * (x * x * x / 3. - PI * x * x / 2.))) << endl;
             }
                ptr_minus_div_Ustar_p[n] = 20000000000000;
         } else {
            ptr_minus_div_Ustar_m[n] = 200000000000000;
            ptr_minus_div_Ustar_p[n] = -(qnnn.dx_central(ptr_ustar_p) + qnnn.dy_central(ptr_vstar_p));
             if (ABS(ptr_minus_div_Ustar_p[n]+((PI - 2.*x)*(y * y * y / 3. - PI * y * y / 2.) + (PI - 2.*y)*(x * x * x / 3. - PI * x * x / 2.))) > 4e-4)
             {
                 cout << "calculated: " << ptr_minus_div_Ustar_p[n] << "\t exact: "
                      << -((PI - 2. * x) * (y * y * y / 3. - PI * y * y / 2.) +
                           (PI - 2. * y) * (x * x * x / 3. - PI * x * x / 2.)) << "\t error = "
                      << ABS(ptr_minus_div_Ustar_p[n] + ((PI - 2. * x) * (y * y * y / 3. - PI * y * y / 2.) +
                                                         (PI - 2. * y) * (x * x * x / 3. - PI * x * x / 2.))) << endl;
             }
         }
      }

      ierr = VecRestoreArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);


      // solve for the hodge variable with the right-hand side set to vec_minus_div_Ustar_m
      //solver.set_rhs(vec_rhs_m, vec_rhs_p);
      solver.set_rhs(vec_minus_div_Ustar_m, vec_minus_div_Ustar_p);


      /***********************************************************************************
       *
       * PROJECTION - PART 2: solve for the Hodge variable
       *
       ***********************************************************************************/

      solver.set_use_taylor_correction(taylor_correction());
      solver.set_kink_treatment(kink_special_treatment());

      vector<vector<double> > pw_bc_values(num_bdry());

      if (apply_bc_pointwise()) {
         solver.preassemble_linear_system();

         // allocate memory for bc values
         for (int i = 0; i < num_bdry(); ++i) {
            pw_bc_values[i].assign(solver.pw_bc_num_value_pts(i), 0);
         }


         // sample bc at requested points
         if (sample_bc_node_by_node()) {
            foreach_local_node(n, nodes) {
               for (int i = 0; i < num_bdry(); ++i) {
                  for (int k = 0; k < solver.pw_bc_num_value_pts(i, n); ++k) {
                     int j = solver.pw_bc_idx_value_pt(i, n, k);
                     solver.pw_bc_xyz_value_pt(i, j, xyz);
                     pw_bc_values[i][j] = bc_value_cf_all[i].value(xyz);
                  }
               }
            }
         } else {
            for (int i = 0; i < num_bdry(); ++i) {
               for (int j = 0; j < solver.pw_bc_num_value_pts(i); ++j) {
                  solver.pw_bc_xyz_value_pt(i, j, xyz);
                  pw_bc_values[i][j] = bc_value_cf_all[i].value(xyz);
               }
            }
         }
      }

      if (use_nonzero_guess()) sample_cf_on_nodes(p4est, nodes, hodge_cf, vec_hodge);

      // actual solve for Hodge:
      solver.solve(vec_hodge, use_nonzero_guess());

      /***********************************************************************************
       *
       * PROJECTION - PART 3: project onto the divergence free field
       *
       ***********************************************************************************/
      // Reset the pointer ptr_hodge to the beginning of vec_hodge
      ierr = VecGetArray(vec_hodge, &ptr_hodge); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      Vec mask_m = solver.get_mask_m();
      Vec mask_p = solver.get_mask_p();
      double *mask_m_ptr;
      double *mask_p_ptr;
      ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      foreach_local_node(n, nodes) {
         auto *ni = (p4est_indep_t *) sc_array_index(&nodes->indep_nodes, n);
         node_xyz_fr_n(n, p4est, nodes, xyz);
         double x = xyz[0], y = xyz[1], z = xyz[2];
//         if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_m_ptr)) {
         if (!is_node_Wall(p4est, ni)) {
            ngbd_n.get_neighbors(n, qnnn);
            if (phi_eff_cf(x, y) < 0) {
               // todo: should be a ptr_hodge_m and ptr_hodge_p
               ptr_unp1_m[n] = ptr_ustar_m[n] - qnnn.dx_central(ptr_hodge);
               ptr_vnp1_m[n] = ptr_vstar_m[n] - qnnn.dy_central(ptr_hodge);
               ptr_unp1_p[n] = 0;
               ptr_vnp1_p[n] = 0;
            } else {
               ptr_unp1_m[n] = 0;
               ptr_vnp1_m[n] = 0;
               ptr_unp1_p[n] = ptr_ustar_p[n] - qnnn.dx_central(ptr_hodge);
               ptr_vnp1_p[n] = ptr_vstar_p[n] - qnnn.dy_central(ptr_hodge);
            }
         } else { // set Unp1 to be the exact solution, since in Navier-Stokes we will impose a Dirichlet boundary condition (no-slip):
            ptr_unp1_m[n] =  sin(x)*cos(y);
            ptr_vnp1_m[n] = -sin(y)*cos(x);
            ptr_unp1_p[n] =  sin(x)*cos(y);
            ptr_vnp1_p[n] = -sin(y)*cos(x);
         }
      }

      //----------------------------------------------------------------------------------------------
      // calculate the max error of unp1 and vnp1
      //----------------------------------------------------------------------------------------------
      ierr = VecGetArray(vec_unp1_m, &ptr_unp1_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vnp1_m, &ptr_vnp1_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_unp1_p, &ptr_unp1_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_vnp1_p, &ptr_vnp1_p); CHKERRXX(ierr);
      ierr = VecGetArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
      ierr = VecGetArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);

      double err_unp1_m_max = 0, err_vnp1_m_max = 0;
      double err_unp1_p_max = 0, err_vnp1_p_max = 0;
      double err_dustar_m_max = 0, err_dustar_p_max =0;

      foreach_local_node(n, nodes) {
         node_xyz_fr_n(n, p4est, nodes, xyz);
         double x = xyz[0], y = xyz[1], z = xyz[2];

         if (phi_eff_cf(x, y) < 0) {
            err_unp1_m_max = MAX(err_unp1_m_max, ABS(ptr_unp1_m[n] - sin(x) * cos(y)));
            err_vnp1_m_max = MAX(err_vnp1_m_max, ABS(ptr_vnp1_m[n] + sin(y) * cos(x)));
            err_dustar_m_max = MAX(err_dustar_m_max, ABS(ptr_minus_div_Ustar_m[n] + ((PI - 2.*x)*(y * y * y / 3. - PI * y * y / 2.) + (PI - 2.*y)*(x * x * x / 3. - PI * x * x / 2.)) ));
         } else {

            err_unp1_p_max = MAX(err_unp1_p_max, ABS(ptr_unp1_p[n] - sin(x) * cos(y)));
            err_vnp1_p_max = MAX(err_vnp1_p_max, ABS(ptr_vnp1_p[n] + sin(y) * cos(x)));
            err_dustar_p_max = MAX(err_dustar_p_max, ABS(ptr_minus_div_Ustar_p[n] + ((PI - 2.*x)*(y * y * y / 3. - PI * y * y / 2.) + (PI - 2.*y)*(x * x * x / 3. - PI * x * x / 2.)) ));
         }
         
         
         

        //err_minus_div_ustar_m_max = MAX(err_minus_div_ustar_m_max, ABS(ptr_unp1_p[n] - sin(x) * cos(y)));


      }

      // Take the max of the local unp1 we just computed:
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_unp1_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_vnp1_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_unp1_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_vnp1_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      // Restoring vectors:
      ierr = VecRestoreArray(vec_ustar_m, &ptr_ustar_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_vstar_m, &ptr_vstar_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_ustar_p, &ptr_ustar_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_vstar_p, &ptr_vstar_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_ustar_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_vstar_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_ustar_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_vstar_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(vec_minus_div_Ustar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_minus_div_Ustar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_minus_div_Ustar_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_minus_div_Ustar_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);



      ierr = VecGhostUpdateBegin(vec_unp1_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_vnp1_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_unp1_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_vnp1_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_unp1_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_vnp1_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_unp1_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_vnp1_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_unp1_m, &ptr_unp1_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_vnp1_m, &ptr_vnp1_m); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_unp1_p, &ptr_unp1_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_vnp1_p, &ptr_vnp1_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_unp1_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_vnp1_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_unp1_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_vnp1_p); CHKERRXX(ierr);

      /***********************************************************************************
       *
       * COMPUTE ERRORS AND SAVE TO FILES:
       *
       ***********************************************************************************/

      Vec bdry_phi_eff = solver.get_boundary_phi_eff();

      if (reinit_level_set()) {
         if (bdry_phi_eff != nullptr) ls.reinitialize_1st_order_time_2nd_order_space(bdry_phi_eff, 20);
      }

      Mat A = solver.get_matrix();

      double *bdry_phi_eff_ptr;

      Vec hodge_m = vec_hodge; double *hodge_m_ptr;
      Vec hodge_p = vec_hodge; double *hodge_p_ptr;

      //----------------------------------------------------------------------------------------------
      // calculate error of hodge
      //----------------------------------------------------------------------------------------------
      Vec vec_error_hodge_m; double *vec_error_hodge_m_ptr;
      Vec vec_error_hodge_p; double *vec_error_hodge_p_ptr;

      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_hodge_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_hodge_p); CHKERRXX(ierr);

      ierr = VecGetArray(hodge_m, &hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(hodge_p, &hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(vec_error_hodge_m, &vec_error_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_hodge_p, &vec_error_hodge_p_ptr); CHKERRXX(ierr);

      double hodge_max = 0;

      foreach_local_node(n, nodes) {
         node_xyz_fr_n(n, p4est, nodes, xyz);

         vec_error_hodge_m_ptr[n] = mask_m_ptr[n] < 0 ? ABS(hodge_m_ptr[n] - hodge_m_cf.value(xyz)) : 0;
         vec_error_hodge_p_ptr[n] = mask_p_ptr[n] < 0 ? ABS(hodge_p_ptr[n] - hodge_p_cf.value(xyz)) : 0;

         hodge_max = MAX(hodge_max, fabs(hodge_m_cf.value(xyz)), fabs(hodge_p_cf.value(xyz)));
      }

      ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(hodge_m, &hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(hodge_p, &hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(vec_error_hodge_m, &vec_error_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_hodge_p, &vec_error_hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(vec_error_hodge_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_error_hodge_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_error_hodge_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_error_hodge_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      //----------------------------------------------------------------------------------------------
      // calculate error of |gradient(hodge)|:
      //----------------------------------------------------------------------------------------------
      Vec vec_error_grad_hodge_m; double *vec_error_gradient_hodge_m_ptr;
      Vec vec_error_grad_hodge_p; double *vec_error_gradient_hodge_p_ptr;

      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_grad_hodge_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_grad_hodge_p); CHKERRXX(ierr);

      ierr = VecGetArray(hodge_m, &hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(hodge_p, &hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(vec_error_grad_hodge_m, &vec_error_gradient_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_grad_hodge_p, &vec_error_gradient_hodge_p_ptr); CHKERRXX(ierr);

      double gradient_hodge_max = 0;

      foreach_local_node(n, nodes) {
         double xyz[P4EST_DIM];
         node_xyz_fr_n(n, p4est, nodes, xyz);

         auto *ni = (p4est_indep_t *) sc_array_index(&nodes->indep_nodes, n);

         if (!compute_grad_between()) {
            ngbd_n.get_neighbors(n, qnnn);

            if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_m_ptr)) {
               double DIM(ddx_hodge_m_exact = ddx_hodge_m_cf(DIM(xyz[0], xyz[1], xyz[2])),
                          ddy_hodge_m_exact = ddy_hodge_m_cf(DIM(xyz[0], xyz[1], xyz[2])),
                          ddz_hodge_m_exact = ddz_hodge_m_cf(DIM(xyz[0], xyz[1], xyz[2])));

               gradient_hodge_max = MAX(gradient_hodge_max, sqrt(SUMD(SQR(ddx_hodge_m_exact), SQR(ddy_hodge_m_exact), SQR(ddz_hodge_m_exact))));

               double DIM(ddx_hodge_m_error = fabs(qnnn.dx_central(hodge_m_ptr) - ddx_hodge_m_exact),
                          ddy_hodge_m_error = fabs(qnnn.dy_central(hodge_m_ptr) - ddy_hodge_m_exact),
                          ddz_hodge_m_error = fabs(qnnn.dz_central(hodge_m_ptr) - ddz_hodge_m_exact));

               vec_error_gradient_hodge_m_ptr[n] = sqrt(SUMD(SQR(ddx_hodge_m_error), SQR(ddy_hodge_m_error), SQR(ddz_hodge_m_error)));
            } else {
               vec_error_gradient_hodge_m_ptr[n] = 0;
            }

            if (!is_node_Wall(p4est, ni) && qnnn.is_stencil_in_negative_domain(mask_p_ptr)) {
               double DIM(ddx_hodge_p_exact = ddx_hodge_p_cf(DIM(xyz[0], xyz[1], xyz[2])),
                          ddy_hodge_p_exact = ddy_hodge_p_cf(DIM(xyz[0], xyz[1], xyz[2])),
                          ddz_hodge_p_exact = ddz_hodge_p_cf(DIM(xyz[0], xyz[1], xyz[2])));

               gradient_hodge_max = MAX(gradient_hodge_max, sqrt(SUMD(SQR(ddx_hodge_p_exact), SQR(ddy_hodge_p_exact), SQR(ddz_hodge_p_exact))));

               double DIM(ddx_hodge_p_error = fabs(qnnn.dx_central(hodge_p_ptr) - ddx_hodge_p_exact),
                          ddy_hodge_p_error = fabs(qnnn.dy_central(hodge_p_ptr) - ddy_hodge_p_exact),
                          ddz_hodge_p_error = fabs(qnnn.dz_central(hodge_p_ptr) - ddz_hodge_p_exact));

               vec_error_gradient_hodge_p_ptr[n] = sqrt(SUMD(SQR(ddx_hodge_p_error), SQR(ddy_hodge_p_error), SQR(ddz_hodge_p_error)));
            } else {
               vec_error_gradient_hodge_p_ptr[n] = 0;
            }
         } else {
            p4est_locidx_t neighbors[num_neighbors_cube];
            bool neighbors_exist[num_neighbors_cube];

            double xyz_nei[P4EST_DIM];
            double xyz_mid[P4EST_DIM];
            double normal[P4EST_DIM];

            vec_error_gradient_hodge_m_ptr[n] = 0;
            vec_error_gradient_hodge_p_ptr[n] = 0;

            if (!is_node_Wall(p4est, ni)) {
               ngbd_n.get_all_neighbors(n, neighbors, neighbors_exist);
               for (int j = 1; j < (int) pow(3, P4EST_DIM); j += 2) {
                  p4est_locidx_t n_nei = neighbors[j];
                  node_xyz_fr_n(n_nei, p4est, nodes, xyz_nei);

                  double delta = 0;

                  foreach_dimension(i) {
                     xyz_mid[i] = .5 * (xyz[i] + xyz_nei[i]);
                     delta += SQR(xyz[i] - xyz_nei[i]);
                     normal[i] = xyz_nei[i] - xyz[i];
                  }

                  delta = sqrt(delta);

                  foreach_dimension(i) normal[i] /= delta;

                  if (mask_m_ptr[n] < 0) {
                     if (mask_m_ptr[n_nei] < 0) {
                        double grad_hodge_exact = SUMD(ddx_hodge_m_cf.value(xyz_mid) * normal[0],
                                                       ddy_hodge_m_cf.value(xyz_mid) * normal[1],
                                                       ddz_hodge_m_cf.value(xyz_mid) * normal[2]);
                        vec_error_gradient_hodge_m_ptr[n] = MAX(vec_error_gradient_hodge_m_ptr[n],
                                                                fabs((hodge_m_ptr[n_nei] - hodge_m_ptr[n]) / delta -
                                                                     grad_hodge_exact));
                        gradient_hodge_max = MAX(gradient_hodge_max, fabs(grad_hodge_exact));
                     }
                  }

                  if (mask_p_ptr[n] < 0) {
                     if (mask_p_ptr[n_nei] < 0) {
                        double grad_hodge_exact = SUMD(ddx_hodge_p_cf.value(xyz_mid) * normal[0],
                                                       ddy_hodge_p_cf.value(xyz_mid) * normal[1],
                                                       ddz_hodge_p_cf.value(xyz_mid) * normal[2]);
                        vec_error_gradient_hodge_p_ptr[n] = MAX(vec_error_gradient_hodge_p_ptr[n],
                                                                fabs((hodge_p_ptr[n_nei] - hodge_p_ptr[n]) / delta -
                                                                     grad_hodge_exact));
                        gradient_hodge_max = MAX(gradient_hodge_max, fabs(grad_hodge_exact));
                     }
                  }
               }
            }
         }
      }

      ierr = VecRestoreArray(hodge_m, &hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(hodge_p, &hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(vec_error_grad_hodge_m, &vec_error_gradient_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_grad_hodge_p, &vec_error_gradient_hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(vec_error_grad_hodge_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_error_grad_hodge_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_error_grad_hodge_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_error_grad_hodge_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      //----------------------------------------------------------------------------------------------
      // calculate extrapolation error
      //----------------------------------------------------------------------------------------------
      Vec vec_error_hodge_extrapolated_m, vec_error_hodge_extrapolated_p;
      double *vec_error_extrapolated_hodge_m_ptr, *vec_error_extrapolated_hodge_p_ptr;
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_hodge_extrapolated_m); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &vec_error_hodge_extrapolated_p); CHKERRXX(ierr);

      double band = extension_band_check();

      // copy hodge into a new Vec
      Vec hodge_m_extrapolated;
      double *hodge_m_extrapolated_ptr;
      ierr = VecCreateGhostNodes(p4est, nodes, &hodge_m_extrapolated); CHKERRXX(ierr);
      Vec hodge_p_extrapolated;
      double *hodge_p_extrapolated_ptr;
      ierr = VecCreateGhostNodes(p4est, nodes, &hodge_p_extrapolated); CHKERRXX(ierr);

      VecCopyGhost(hodge_m, hodge_m_extrapolated);
      VecCopyGhost(hodge_p, hodge_p_extrapolated);

      Vec phi_m;
      ierr = VecDuplicate(bdry_phi_eff, &phi_m); CHKERRXX(ierr);
      VecCopyGhost(bdry_phi_eff, phi_m);
      Vec phi_p;
      ierr = VecDuplicate(bdry_phi_eff, &phi_p); CHKERRXX(ierr);
      VecCopyGhost(bdry_phi_eff, phi_p);

      double *phi_m_ptr;
      double *phi_p_ptr;

      // extend
      boundary_conditions_t *bc = nullptr;
      if (apply_bc_pointwise()) {
         bc = solver.get_bc(0);
      }

      switch (extend_solution()) {
         case 1:
            ls.extend_Over_Interface_TVD(phi_m, hodge_m_extrapolated, extension_iterations(), 2,
                                         -extension_band_compute() * dxyz_max,
                                         extension_band_extend() * dxyz_max, nullptr, mask_m, bc,
                                         use_nonzero_guess());

            ls.extend_Over_Interface_TVD(phi_p, hodge_p_extrapolated, extension_iterations(), 2,
                                         -extension_band_compute() * dxyz_max,
                                         extension_band_extend() * dxyz_max, nullptr, mask_p, bc,
                                         use_nonzero_guess());
            break;
         case 2:
            ls.extend_Over_Interface_TVD_Full(phi_m, hodge_m_extrapolated, extension_iterations(), 2,
                                              -extension_band_compute() * dxyz_max,
                                              extension_band_extend() * dxyz_max, nullptr, mask_m, bc,
                                              use_nonzero_guess());

            ls.extend_Over_Interface_TVD_Full(phi_p, hodge_p_extrapolated, extension_iterations(), 2,
                                              -extension_band_compute() * dxyz_max,
                                              extension_band_extend() * dxyz_max, nullptr, mask_p, bc,
                                              use_nonzero_guess());

            break;
      }

      // calculate error of extrapolation:
      ierr = VecGetArray(hodge_m_extrapolated, &hodge_m_extrapolated_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(hodge_p_extrapolated, &hodge_p_extrapolated_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(vec_error_hodge_extrapolated_m, &vec_error_extrapolated_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(vec_error_hodge_extrapolated_p, &vec_error_extrapolated_hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecGetArray(phi_m, &phi_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_p, &phi_p_ptr); CHKERRXX(ierr);

      foreach_local_node(n, nodes) {
         double xyz[P4EST_DIM];
         node_xyz_fr_n(n, p4est, nodes, xyz);

         vec_error_extrapolated_hodge_m_ptr[n] = (mask_m_ptr[n] > 0. && phi_m_ptr[n] < band * dxyz_max) ? ABS(hodge_m_extrapolated_ptr[n] - hodge_m_cf.value(xyz)) : 0;
         vec_error_extrapolated_hodge_p_ptr[n] = (mask_p_ptr[n] > 0. && phi_p_ptr[n] < band * dxyz_max) ? ABS(hodge_p_extrapolated_ptr[n] - hodge_p_cf.value(xyz)) : 0;
      }

      ierr = VecRestoreArray(hodge_m_extrapolated, &hodge_m_extrapolated_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(hodge_p_extrapolated, &hodge_p_extrapolated_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(vec_error_hodge_extrapolated_m, &vec_error_extrapolated_hodge_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(vec_error_hodge_extrapolated_p, &vec_error_extrapolated_hodge_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(phi_m, &phi_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_p, &phi_p_ptr); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(vec_error_hodge_extrapolated_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vec_error_hodge_extrapolated_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecGhostUpdateEnd(vec_error_hodge_extrapolated_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vec_error_hodge_extrapolated_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecDestroy(phi_m); CHKERRXX(ierr);
      ierr = VecDestroy(phi_p); CHKERRXX(ierr);

      // compute L-inf norm of errors
      double err_hodge_m_max = 0.;
      ierr = VecMax(vec_error_hodge_m, nullptr, &err_hodge_m_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      double err_grad_hodge_m_max = 0.;
      ierr = VecMax(vec_error_grad_hodge_m, nullptr, &err_grad_hodge_m_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_grad_hodge_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      double err_hodge_extrapolated_m_max = 0.;
      ierr = VecMax(vec_error_hodge_extrapolated_m, nullptr, &err_hodge_extrapolated_m_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge_extrapolated_m_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      double err_hodge_p_max = 0.;
      ierr = VecMax(vec_error_hodge_p, nullptr, &err_hodge_p_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      double err_grad_hodge_p_max = 0.;
      ierr = VecMax(vec_error_grad_hodge_p, nullptr, &err_grad_hodge_p_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_grad_hodge_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      double err_hodge_extrapolated_p_max = 0.;
      ierr = VecMax(vec_error_hodge_extrapolated_p, nullptr, &err_hodge_extrapolated_p_max); CHKERRXX(ierr);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge_extrapolated_p_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      if (scale_errors()) {
         mpiret = MPI_Allreduce(MPI_IN_PLACE, &hodge_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
         err_hodge_m_max /= hodge_max;
         err_hodge_p_max /= hodge_max;

         mpiret = MPI_Allreduce(MPI_IN_PLACE, &gradient_hodge_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
         err_grad_hodge_m_max /= gradient_hodge_max;
         err_grad_hodge_p_max /= gradient_hodge_max;
      }

      error_hodge_m_arr.push_back(err_hodge_m_max);
      error_gradient_hodge_m_arr.push_back(err_grad_hodge_m_max);
      error_hodge_extrapolated_m_arr.push_back(err_hodge_extrapolated_m_max);

      error_hodge_p_arr.push_back(err_hodge_p_max);
      error_gradient_hodge_p_arr.push_back(err_grad_hodge_p_max);
      error_hodge_extrapolated_p_arr.push_back(err_hodge_extrapolated_p_max);

      error_unp1_m_arr.push_back(err_unp1_m_max);
      error_vnp1_m_arr.push_back(err_vnp1_m_max);
      error_unp1_p_arr.push_back(err_unp1_p_max);
      error_vnp1_p_arr.push_back(err_vnp1_p_max);

      // Print current errors
      if (iter > -1) {
         ierr = PetscPrintf(p4est->mpicomm, "Errors Neg: "); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "hodge = %3.2e (%+3.2f), ", err_hodge_m_max, log(error_hodge_m_arr[iter - 1] / error_hodge_m_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "|grad(hodge)| = %3.2e (%+3.2f), ", err_grad_hodge_m_max, log(error_gradient_hodge_m_arr[iter - 1] / error_gradient_hodge_m_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "unp1 = %3.2e (%+3.2f), ", err_unp1_m_max, log(error_unp1_m_arr[iter - 1] / error_unp1_m_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "vnp1 = %3.2e (%+3.2f), ", err_vnp1_m_max, log(error_vnp1_m_arr[iter - 1] / error_vnp1_m_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "minus divergence ustar = %3.2e .", err_dustar_m_max); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "\n"); CHKERRXX(ierr);

         ierr = PetscPrintf(p4est->mpicomm, "Errors Pos: "); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "hodge = %3.2e (%+3.2f), ", err_hodge_p_max, log(error_hodge_p_arr[iter - 1] / error_hodge_p_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "|grad(hodge)| = %3.2e (%+3.2f), ", err_grad_hodge_p_max, log(error_gradient_hodge_p_arr[iter - 1] / error_gradient_hodge_p_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "unp1 = %3.2e (%+3.2f), ", err_unp1_p_max, log(error_unp1_p_arr[iter - 1] / error_unp1_p_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "vnp1 = %3.2e (%+3.2f), ", err_vnp1_p_max, log(error_vnp1_p_arr[iter - 1] / error_vnp1_p_arr[iter]) / log(2)); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "minus divergence ustar = %3.2e .", err_dustar_p_max); CHKERRXX(ierr);
         ierr = PetscPrintf(p4est->mpicomm, "\n"); CHKERRXX(ierr);
      }

      if (save_vtk()) {
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
         Vec leaf_level; ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
         double *l_p;    ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

         for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
            auto *tree = (p4est_tree_t *) sc_array_index(p4est->trees, tree_idx);
            for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
               const p4est_quadrant_t *quad = (p4est_quadrant_t *) sc_array_index(&tree->quadrants, q);
               l_p[tree->quadrants_offset + q] = quad->level;
            }
         }

         for (size_t q = 0; q < ghost->ghosts.elem_count; ++q) {
            const p4est_quadrant_t *quad = (p4est_quadrant_t *) sc_array_index(&ghost->ghosts, q);
            l_p[p4est->local_num_quadrants + q] = quad->level;
         }

         Vec vec_hodge_exact;
         double *hodge_exact_ptr;

         ierr = VecDuplicate(vec_hodge, &vec_hodge_exact); CHKERRXX(ierr);
         sample_cf_on_nodes(p4est, nodes, hodge_cf, vec_hodge_exact);

         ierr = VecGetArray(bdry_phi_eff, &bdry_phi_eff_ptr); CHKERRXX(ierr);

         ierr = VecGetArray(vec_hodge, &ptr_hodge); CHKERRXX(ierr);
         ierr = VecGetArray(vec_hodge_exact, &hodge_exact_ptr); CHKERRXX(ierr);

         ierr = VecGetArray(hodge_m_extrapolated, &hodge_m_extrapolated_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(hodge_p_extrapolated, &hodge_p_extrapolated_ptr); CHKERRXX(ierr);

         ierr = VecGetArray(vec_error_hodge_m, &vec_error_hodge_m_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(vec_error_grad_hodge_m, &vec_error_gradient_hodge_m_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(vec_error_hodge_extrapolated_m, &vec_error_extrapolated_hodge_m_ptr); CHKERRXX(ierr);

         ierr = VecGetArray(vec_error_hodge_p, &vec_error_hodge_p_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(vec_error_grad_hodge_p, &vec_error_gradient_hodge_p_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(vec_error_hodge_extrapolated_p, &vec_error_extrapolated_hodge_p_ptr); CHKERRXX(ierr);

         ierr = VecGetArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

          ierr = VecGetArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
          ierr = VecGetArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);

         double *vec_mu_m_ptr;
         double *vec_mu_p_ptr;

         ierr = VecGetArray(vec_mu_m, &vec_mu_m_ptr); CHKERRXX(ierr);
         ierr = VecGetArray(vec_mu_p, &vec_mu_p_ptr); CHKERRXX(ierr);

         my_p4est_vtk_write_all(p4est, nodes, ghost,
                                P4EST_TRUE, P4EST_TRUE,
                                17, 1, oss.str().c_str(),
                                VTK_POINT_DATA, "phi", bdry_phi_eff_ptr,
                                VTK_POINT_DATA, "hodge", ptr_hodge,
                                VTK_POINT_DATA, "hodge_exact", hodge_exact_ptr,
                                VTK_POINT_DATA, "hodge_m_extrapolated", hodge_m_extrapolated_ptr,
                                VTK_POINT_DATA, "hodge_p_extrapolated", hodge_p_extrapolated_ptr,
                                VTK_POINT_DATA, "vec_mu_m", vec_mu_m_ptr,
                                VTK_POINT_DATA, "vec_mu_p", vec_mu_p_ptr,
                                VTK_POINT_DATA, "mask_m", mask_m_ptr,
                                VTK_POINT_DATA, "mask_p", mask_p_ptr,
                                VTK_POINT_DATA, "error_hodge_m", vec_error_hodge_m_ptr,
                                VTK_POINT_DATA, "error_gradient_hodge_m", vec_error_gradient_hodge_m_ptr,
                                VTK_POINT_DATA, "error_hodge_extrapolated_m", vec_error_extrapolated_hodge_m_ptr,
                                VTK_POINT_DATA, "error_hodge_p", vec_error_hodge_p_ptr,
                                VTK_POINT_DATA, "error_gradient_hodge_p", vec_error_gradient_hodge_p_ptr,
                                VTK_POINT_DATA, "error_hodge_extrapolated_p", vec_error_extrapolated_hodge_p_ptr,
                                VTK_CELL_DATA, "leaf_level", l_p,
                                VTK_POINT_DATA, "minus_div_u_star_m",vec_minus_div_Ustar_m,
                                VTK_CELL_DATA, "minus_div_u_star_p",vec_minus_div_Ustar_p);

         ierr = VecRestoreArray(bdry_phi_eff, &bdry_phi_eff_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(vec_hodge, &ptr_hodge); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_hodge_exact, &hodge_exact_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(hodge_m_extrapolated, &hodge_m_extrapolated_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(hodge_p_extrapolated, &hodge_p_extrapolated_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(vec_error_hodge_m, &vec_error_hodge_m_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_error_grad_hodge_m, &vec_error_gradient_hodge_m_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_error_hodge_extrapolated_m, &vec_error_extrapolated_hodge_m_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(vec_error_hodge_p, &vec_error_hodge_p_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_error_grad_hodge_p, &vec_error_gradient_hodge_p_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_error_hodge_extrapolated_p, &vec_error_extrapolated_hodge_p_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(mask_m, &mask_m_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(mask_p, &mask_p_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(vec_mu_m, &vec_mu_m_ptr); CHKERRXX(ierr);
         ierr = VecRestoreArray(vec_mu_p, &vec_mu_p_ptr); CHKERRXX(ierr);

         ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
         ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
         ierr = VecDestroy(vec_hodge_exact); CHKERRXX(ierr);


          ierr = VecRestoreArray(vec_minus_div_Ustar_m, &ptr_minus_div_Ustar_m); CHKERRXX(ierr);
          ierr = VecRestoreArray(vec_minus_div_Ustar_p, &ptr_minus_div_Ustar_p); CHKERRXX(ierr);
          ierr = VecDestroy(vec_minus_div_Ustar_m); CHKERRXX(ierr);
          ierr = VecDestroy(vec_minus_div_Ustar_p); CHKERRXX(ierr);

         PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
      }

      // destroy Vec's with errors
      ierr = VecDestroy(vec_error_hodge_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_error_grad_hodge_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_error_hodge_extrapolated_m); CHKERRXX(ierr);

      ierr = VecDestroy(vec_error_hodge_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_error_grad_hodge_p); CHKERRXX(ierr);
      ierr = VecDestroy(vec_error_hodge_extrapolated_p); CHKERRXX(ierr);

      ierr = VecDestroy(hodge_m_extrapolated); CHKERRXX(ierr);
      ierr = VecDestroy(hodge_p_extrapolated); CHKERRXX(ierr);

      ierr = VecDestroy(vec_hodge); CHKERRXX(ierr);

      ierr = VecDestroy(vec_mu_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_mu_p); CHKERRXX(ierr);

      ierr = VecDestroy(vec_rhs_m); CHKERRXX(ierr);
      ierr = VecDestroy(vec_rhs_p); CHKERRXX(ierr);


      for (unsigned int i = 0; i < num_bdry_max; i++) {
         if (*bdry_present_all[i]) {
            ierr = VecDestroy(bdry_phi_vec_all[i]); CHKERRXX(ierr);
         }
      }

      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_destroy(p4est);
      my_p4est_brick_destroy(connectivity, &brick);
   }

   MPI_Barrier(mpi.comm());

   vector<double> error_hodge_m_L1(num_splits.val, 0), error_hodge_m_avg(num_splits.val, 0), error_hodge_m_max(num_splits.val, 0);
   vector<double> error_gradient_hodge_m_L1(num_splits.val, 0), error_gradient_hodge_m_avg(num_splits.val, 0), error_gradient_hodge_m_max(num_splits.val, 0);
   vector<double> error_hodge_extrapolated_m_L1(num_splits.val, 0), error_hodge_extrapolated_m_avg(num_splits.val, 0), error_hodge_extrapolated_m_max(num_splits.val, 0);

   vector<double> error_hodge_p_L1(num_splits.val, 0), error_hodge_p_avg(num_splits.val, 0), error_hodge_p_max(num_splits.val, 0);
   vector<double> error_gradient_hodge_p_L1(num_splits.val, 0), error_gradient_hodge_p_avg(num_splits.val, 0), error_gradient_hodge_p_max(num_splits.val, 0);
   vector<double> error_hodge_extrapolated_p_L1(num_splits.val, 0), error_hodge_extrapolated_p_avg(num_splits.val, 0), error_hodge_extrapolated_p_max(num_splits.val, 0);

   // for each resolution compute the max, mean and deviation of the errors computed above:
   for (int p = 0; p < num_splits.val; ++p) {
      // one
      error_hodge_m_L1[p] = error_hodge_m_arr[p];
      error_gradient_hodge_m_L1[p] = error_gradient_hodge_m_arr[p];
      error_hodge_extrapolated_m_L1[p] = error_hodge_extrapolated_m_arr[p];

      error_hodge_p_L1[p] = error_hodge_p_arr[p];
      error_gradient_hodge_p_L1[p] = error_gradient_hodge_p_arr[p];
      error_hodge_extrapolated_p_L1[p] = error_hodge_extrapolated_p_arr[p];

      // max
      error_hodge_m_max[p] = MAX(error_hodge_m_max[p], error_hodge_m_arr[p]);
      error_gradient_hodge_m_max[p] = MAX(error_gradient_hodge_m_max[p], error_gradient_hodge_m_arr[p]);
      error_hodge_extrapolated_m_max[p] = MAX(error_hodge_extrapolated_m_max[p], error_hodge_extrapolated_m_arr[p]);

      error_hodge_p_max[p] = MAX(error_hodge_p_max[p], error_hodge_p_arr[p]);
      error_gradient_hodge_p_max[p] = MAX(error_gradient_hodge_p_max[p], error_gradient_hodge_p_arr[p]);
      error_hodge_extrapolated_p_max[p] = MAX(error_hodge_extrapolated_p_max[p], error_hodge_extrapolated_p_arr[p]);

      // avg
      error_hodge_m_avg[p] += error_hodge_m_arr[p];
      error_gradient_hodge_m_avg[p] += error_gradient_hodge_m_arr[p];
      error_hodge_extrapolated_m_avg[p] += error_hodge_extrapolated_m_arr[p];

      error_hodge_p_avg[p] += error_hodge_p_arr[p];
      error_gradient_hodge_p_avg[p] += error_gradient_hodge_p_arr[p];
      error_hodge_extrapolated_p_avg[p] += error_hodge_extrapolated_p_arr[p];
   }

   if (mpi.rank() == 0) {
      std::ostringstream command;
      command << "mkdir -p " << out_dir << "/convergence";
      int ret_sys = system(command.str().c_str());
      if (ret_sys < 0)
         throw std::invalid_argument("could not create directory");

      std::string filename;

      // save level and resolution
      filename = out_dir; filename += "/convergence/lvl.txt"; save_vector(filename.c_str(), lvl_arr);
      filename = out_dir; filename += "/convergence/h_arr.txt"; save_vector(filename.c_str(), h_arr);
      filename = out_dir; filename += "/convergence/mu_arr.txt"; save_vector(filename.c_str(), mu_arr);

      filename = out_dir; filename += "/convergence/error_hodge_m_arr.txt"; save_vector(filename.c_str(), error_hodge_m_arr);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_m_arr.txt"; save_vector(filename.c_str(), error_gradient_hodge_m_arr);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_m_arr.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_m_arr);

      filename = out_dir; filename += "/convergence/error_hodge_m_L1.txt"; save_vector(filename.c_str(), error_hodge_m_L1);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_m_L1.txt"; save_vector(filename.c_str(), error_gradient_hodge_m_L1);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_m_L1.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_m_L1);

      filename = out_dir; filename += "/convergence/error_hodge_m_avg.txt"; save_vector(filename.c_str(), error_hodge_m_avg);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_m_avg.txt"; save_vector(filename.c_str(), error_gradient_hodge_m_avg);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_m_avg.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_m_avg);

      filename = out_dir; filename += "/convergence/error_hodge_m_max.txt"; save_vector(filename.c_str(), error_hodge_m_max);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_m_max.txt"; save_vector(filename.c_str(), error_gradient_hodge_m_max);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_m_max.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_m_max);

      filename = out_dir; filename += "/convergence/error_hodge_p_arr.txt"; save_vector(filename.c_str(), error_hodge_p_arr);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_p_arr.txt"; save_vector(filename.c_str(), error_gradient_hodge_p_arr);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_p_arr.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_p_arr);

      filename = out_dir; filename += "/convergence/error_hodge_p_L1.txt"; save_vector(filename.c_str(), error_hodge_p_L1);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_p_L1.txt"; save_vector(filename.c_str(), error_gradient_hodge_p_L1);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_p_L1.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_p_L1);

      filename = out_dir; filename += "/convergence/error_hodge_p_avg.txt"; save_vector(filename.c_str(), error_hodge_p_avg);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_p_avg.txt"; save_vector(filename.c_str(), error_gradient_hodge_p_avg);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_p_avg.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_p_avg);

      filename = out_dir; filename += "/convergence/error_hodge_p_max.txt"; save_vector(filename.c_str(), error_hodge_p_max);
      filename = out_dir; filename += "/convergence/error_gradient_hodge_p_max.txt"; save_vector(filename.c_str(), error_gradient_hodge_p_max);
      filename = out_dir; filename += "/convergence/error_hodge_extrapolated_p_max.txt"; save_vector(filename.c_str(), error_hodge_extrapolated_p_max);
   }

   w.stop();
   w.read_duration();

   return EXIT_SUCCESS;
}