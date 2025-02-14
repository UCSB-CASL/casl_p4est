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
#include <src/my_p8est_biofilm.h>
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
#include <src/my_p4est_biofilm.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;

param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<int> px (pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<int> py (pl, 1, "py", "Periodicity in the y-direction (0/1)");
param_t<int> pz (pl, 1, "pz", "Periodicity in the z-direction (0/1)");

param_t<int> nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<double> xmin (pl, 0, "xmin", "Box xmin");
param_t<double> ymin (pl, 0, "ymin", "Box ymin");
param_t<double> zmin (pl, 0, "zmin", "Box zmin");

param_t<double> xmax (pl, 1, "xmax", "Box xmax");
param_t<double> ymax (pl, 1, "ymax", "Box ymax");
param_t<double> zmax (pl, 1, "zmax", "Box zmax");

//-------------------------------------
// grid resolution parameters
//-------------------------------------
#ifdef P4_TO_P8
param_t<int>    lmin (pl, 4,   "lmin", "Minimum level of refinement");
param_t<int>    lmax (pl, 6,   "lmax", "Maximum level of refinement");
param_t<double> lip  (pl, 1.6, "lip",  "Refinement transition width");
#else
param_t<int>    lmin (pl, 6,   "lmin", "Minimum level of refinement");
param_t<int>    lmax (pl, 10,  "lmax", "Maximum level of refinement");
param_t<double> lip  (pl, 1.6, "lip",  "Refinement transition width");
#endif

//-------------------------------------
// model options
//-------------------------------------
param_t<int>  velocity_type (pl, 1, "velocity_type", "Calculation of velocity: 0 - using concentration (not implemented), 1 - using pressure (Darcy)");
param_t<bool> steady_state  (pl, 0, "steady_state",  "Assume steady state profile for concentration or not");

param_t<double> box_size (pl, 1.0,   "box_size", "Lateral dimensions of simulation box      - m        ");
param_t<double> Df       (pl, 1.e-3, "Df",       "Diffusivity of nutrients in air           - m^2/s    ");
param_t<double> Db       (pl, 1.e-2, "Db",       "Diffusivity of nutrients in biofilm       - m^2/s    ");
param_t<double> Da       (pl, 1.e-1, "Da",       "Diffusivity of nutrients in agar          - m^2/s    ");
param_t<double> sigma    (pl, 0.0,   "sigma",    "Surface tension of air/film interface     - N/m      ");
param_t<double> rho      (pl, 1.0,   "rho",      "Density of biofilm                        - kg/m^3   ");
param_t<double> lambda   (pl, 1.0,   "lambda",   "Mobility of biofilm                       - m^4/(N*s)");
param_t<double> A        (pl, 1.0,   "A",        "Maximum uptake rate                       - kg/m^3   ");
param_t<double> Kc       (pl, 1.0,   "Kc",       "Half-saturation constant                  - kg/m^3   ");
param_t<double> gam      (pl, 0.5,   "gam",      "This represent B/A (see the paper)");
param_t<double> C0f      (pl, 1.0,   "C0f",      "Initial nutrient concentration in air     - kg/m^3");
param_t<double> C0b      (pl, 1.0,   "C0b",      "Initial nutrient concentration in biofilm - kg/m^3");
param_t<double> C0a      (pl, 1.0,   "C0a",      "Initial nutrient concentration in agar    - kg/m^3");

param_t<BoundaryConditionType> bc_agar (pl, NEUMANN, "bc_agar", "BC type (on computatoinal domain boundary) for nutrients in agar   ");
param_t<BoundaryConditionType> bc_free (pl, NEUMANN, "bc_free", "BC type (on computatoinal domain boundary) for nutrients in biofilm");
param_t<BoundaryConditionType> bc_biof (pl, NEUMANN, "bc_biof", "BC type (on computatoinal domain boundary) for nutrients in air    ");

param_t<int> nb_geometry (pl, 1, "nb_geometry", "Initial geometry:\n"
                                                "0 - planar\n"
                                                "1 - sphere\n"
                                                "2 - three spheres\n"
                                                "3 - pipe\n"
                                                "4 - planar + bump\n"
                                                "5 - corrugated agar\n"
                                                "6 - porous media (grains)\n"
                                                "7 - porous media (cavities)");

param_t<double> h_agar (pl, 1, "h_agar", "characteristic size of agar    - m");
param_t<double> h_biof (pl, 1, "h_biof", "characteristic size of biofilm - m");

//-------------------------------------
// specifically for porous media examples
//-------------------------------------
param_t<int>    grain_num        (pl, 1, "grain_num",        "Number of grains or cavities");
param_t<double> grain_dispersity (pl, 1, "grain_dispersity", "Grains/cavities size dispersion");
param_t<double> grain_smoothing  (pl, 1, "grain_smoothing",  "Smoothing of initial geometry");

//-------------------------------------
// time discretization
//-------------------------------------
param_t<int>    advection_scheme (pl, 2,    "advection_scheme", "Advection scheme: 1 - 1st order, 2 - 2nd order");
param_t<int>    time_scheme      (pl, 2,    "time_scheme",      "Time discretization: 1 - Euler (1st order), 2 - BDF2 (2nd order)");
param_t<double> cfl_number       (pl, 0.25, "cfl_number",       "CFL number for computing time step");

//-------------------------------------
// solving non-linear diffusion equation
//-------------------------------------
param_t<int>    iteration_scheme (pl, 1,      "iteration_scheme", "Iterative scheme for solving nonlinear equation: 0 - simple fixed-point, 1 - linearized fixed-point (a.k.a. Newton)");
param_t<int>    max_iterations   (pl, 10,     "max_iterations",   "Maximum iterations for solving nonlinear equation");
param_t<double> tolerance        (pl, 1.e-12, "tolerance",        "Tolerance for solving nonlinear equation");

//-------------------------------------
// general poisson solver parameters
//-------------------------------------
param_t<bool> use_sc_scheme         (pl, 1, "use_sc_scheme",         "For finite volume schemes (jump, neumann, robin): 0 - simple scheme, 1 - superconvergent (second-order accurate gradients)");
param_t<bool> use_taylor_correction (pl, 1, "use_taylor_correction", "For symmetric scheme for robin bc, irrelevant here actually");
param_t<int>  integration_order     (pl, 1, "integration_order",     "Order of geometric reconstruction for computing domain and boundary integrals: 1 - linear, 2 - quadratic");

//-------------------------------------
// smoothing of curvature
//-------------------------------------
param_t<double> curvature_smoothing       (pl, 0, "curvature_smoothing",       "Characteristic distance over which curvature is smoothed (in diagonal lengths)");
param_t<int>    curvature_smoothing_steps (pl, 5,  "curvature_smoothing_steps", "Number of smoothing steps");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool>   save_data     (pl, 1,    "save_data",     "Save scalar characteristics (time elapsed, biofilm volume, surface area, etc)");
param_t<bool>   save_vtk      (pl, 1,    "save_vtk",      "Save spatial data");
param_t<int>    save_type     (pl, 0,    "save_type",     "When save 0 - every dn iterations, 1 - every dl of growth, 2 - every dt of time");
param_t<int>    save_every_dn (pl, 1,    "save_every_dn", "");
param_t<double> save_every_dl (pl, 0.01, "save_every_dl", "");
param_t<double> save_every_dt (pl, 0.5,  "save_every_dt", "");

//-------------------------------------
// simulation run parameters
//-------------------------------------
param_t<int>    limit_iter     (pl, 10000,   "limit_iter",     "Max total time steps");
param_t<double> limit_time     (pl, DBL_MAX, "limit_time",     "Max total run time");
param_t<double> limit_length   (pl, 1.8,     "limit_length",   "Max total biofilm growth (not implemented yet)");
param_t<double> limit_nutrient (pl, 0.02,    "limit_nutrient", "Terminate simulation when this fraction of initial nutrient is left in the system");
param_t<double> limit_wall     (pl, 0.05,    "limit_wall",     "Terminate simulation when biofilm gets that close to walls of computational domain");
param_t<double> init_perturb   (pl, 0.00001, "init_perturb",   "Initial random perturbation of biofilm surface");

//-------------------------------------
// pre-defined examples
//-------------------------------------
param_t<int> nb_experiment (pl, 6, "nb_experiment", "");
/* 0 - (biofilm + agar + water), planar, transient
 * 1 - (biofilm + agar + water), planar, steady-state
 * 2 - (biofilm + agar), planar with a bump, transient
 * 3 - (biofilm + water), spherical, steady-state
 * 4 - (biofilm + water), pipe, transient
 * 5 - (biofilm + water), porous (cavities), transient
 * 6 - (biofilm + water + agar), porous (grains), transient
 */

void setup_experiment()
{
  // common parameters
  velocity_type.val = 1;
  box_size.val = 1;
  sigma.val = 1.e-20;
  rho.val = 1;
  lambda.val = 1.0;
  A.val = 1;
  Kc.val = 1;
  gam.val = 0.5;
  C0a.val = 1;
  C0b.val = 1.0;
  C0f.val = 1;

  switch(nb_experiment.val)
  {
    case 0:
      {
        steady_state.val = 0;

        Da.val = 0.001;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = NEUMANN;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 0;
        h_agar.val      = 0.2;
        h_biof.val      = 0.015;
        break;
      }
    case 1:
      {
        steady_state.val = 1;

        Da.val = 0.001;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = DIRICHLET;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 0;
        h_agar.val      = 0.2;
        h_biof.val      = 0.015;
        break;
      }
    case 2:
      {
        steady_state.val = 0;

        Da.val = 0.001;
        Db.val = 0.01;
        Df.val = 0;

        bc_agar.val = NEUMANN;
        bc_free.val = NEUMANN;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 4;
        h_agar.val      = 0.4;
#ifdef P4_TO_P8
        h_biof.val      = 0.115;
#else
        h_biof.val      = 0.015;
#endif

        C0f.val = 0;
        break;
      }
    case 3:
      {
        steady_state.val = 0;

        Da.val = 0;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = DIRICHLET;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 1;
        h_agar.val      = -0.125;
        h_biof.val      = 0.015;
        break;
      }
    case 4:
      {
        steady_state.val = 0;

        Da.val = 0;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = DIRICHLET;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 3;
        h_agar.val      = 0.4;
        h_biof.val      = 0.015;

        C0a.val = 0;
        break;
      }
    case 5:
      {
        steady_state.val = 0;

        Da.val = 0;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = NEUMANN;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 7;
        h_agar.val      = 0.03;
        h_biof.val      = 0.015;

        grain_num.val        = 100;
        grain_dispersity.val = 2;
        grain_smoothing.val  = 0.01;

        C0a.val = 0;
        break;
      }
    case 6:
      {
        steady_state.val = 0;

        Da.val = 0.001;
        Db.val = 0.01;
        Df.val = 0.1;

        bc_agar.val = NEUMANN;
        bc_free.val = NEUMANN;
        bc_biof.val = NEUMANN;

        nb_geometry.val = 6;
        h_agar.val      = 0.011;
        h_biof.val      = 0.015;

        grain_num.val        = 50;
        grain_dispersity.val = 1.5;
        grain_smoothing.val  = 0.01;
        break;
      }
  }
}

void set_periodicity()
{
  switch (nb_geometry.val)
  {
    case 0: px.val = 0; py.val = 1; pz.val = 1; break;
    case 1: px.val = 0; py.val = 0; pz.val = 0; break;
    case 2: px.val = 0; py.val = 0; pz.val = 0; break;
    case 3: px.val = 0; py.val = 0; pz.val = 1; break;
    case 4: px.val = 0; py.val = 1; pz.val = 1; break;
    case 5: px.val = 0; py.val = 1; pz.val = 1; break;
    case 6: px.val = 1; py.val = 1; pz.val = 1; break;
    case 7: px.val = 1; py.val = 1; pz.val = 1; break;
    default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
  }
}

class phi_agar_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (nb_geometry.val)
    {
      case 0:
        return -(x-h_agar.val);
      case 1:
        {
          double xc = xmin.val +.5*(xmax.val-xmin.val);
          double yc = ymin.val +.5*(ymax.val-ymin.val);
          double zc = zmin.val +.5*(zmax.val-zmin.val);
          return h_agar.val - sqrt( SUMD(SQR(x-xc), SQR(y-yc), SQR(z-zc)) );
        }
      case 2:
        {
          double xc0 = xmin.val +.3*(xmax.val-xmin.val), xc1 = xmin.val +.5*(xmax.val-xmin.val), xc2 = xmin.val +.6*(xmax.val-xmin.val);
          double yc0 = ymin.val +.5*(ymax.val-ymin.val), yc1 = ymin.val +.6*(ymax.val-ymin.val), yc2 = ymin.val +.4*(ymax.val-ymin.val);
          double zc0 = zmin.val +.6*(zmax.val-zmin.val), zc1 = zmin.val +.4*(zmax.val-zmin.val), zc2 = zmin.val +.3*(zmax.val-zmin.val);

          return MAX(h_agar.val - sqrt( SUMD(SQR(x-xc0), SQR(y-yc0), SQR(z-zc0)) ),
                     h_agar.val - sqrt( SUMD(SQR(x-xc1), SQR(y-yc1), SQR(z-zc1)) ),
                     h_agar.val - sqrt( SUMD(SQR(x-xc2), SQR(y-yc2), SQR(z-zc2)) ));
        }
      case 3:
        {
          double xc = xmin.val +.5*(xmax.val-xmin.val);
          double yc = ymin.val +.5*(ymax.val-ymin.val);
          double zc = zmin.val +.5*(zmax.val-zmin.val);
          return -h_agar.val + sqrt( SUMD(SQR(x-xc), SQR(y-yc), SQR(z-zc)) );
        }
      case 4:
        return -(x-h_agar.val);
      case 5:
        return -(x-h_agar.val)
            + 0.02*cos(2.*PI*5.*(y-ymin.val)/(ymax.val-ymin.val))
            ONLY3D(+ 0.02*cos(2.*PI*5.*(z-zmin.val)/(zmax.val-zmin.val)));
      case 6:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num.val; ++i)
          {
            double R = h_agar.val * (1. + grain_dispersity.val*(((double) rand() / (double) RAND_MAX)));
            double X = xmin.val + ((double) rand() / (double) RAND_MAX) *(xmax.val-xmin.val);
            double Y = ymin.val + ((double) rand() / (double) RAND_MAX) *(ymax.val-ymin.val);
            double Z = zmin.val + ((double) rand() / (double) RAND_MAX) *(zmax.val-zmin.val);

            XCODE( int nx = round((X-x)/(xmax.val-xmin.val)) );
            YCODE( int ny = round((Y-y)/(ymax.val-ymin.val)) );
            ZCODE( int nz = round((Z-z)/(zmax.val-zmin.val)) );

            double dist = R - sqrt( SUMD(SQR(x-X + nx*(xmax.val-xmin.val)),
                                         SQR(y-Y + ny*(ymax.val-ymin.val)),
                                         SQR(z-Z + nz*(zmax.val-zmin.val))) );

            sum = MIN(sum, -dist);
          }

          return -sum;
        }
      case 7:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num.val; ++i)
          {
            double R = h_agar.val * (1. + grain_dispersity.val*(((double) rand() / (double) RAND_MAX)));
            double X = xmin.val + ((double) rand() / (double) RAND_MAX) *(xmax.val-xmin.val);
            double Y = ymin.val + ((double) rand() / (double) RAND_MAX) *(ymax.val-ymin.val);
            double Z = zmin.val + ((double) rand() / (double) RAND_MAX) *(zmax.val-zmin.val);

            XCODE( int nx = round((X-x)/(xmax.val-xmin.val)) );
            YCODE( int ny = round((Y-y)/(ymax.val-ymin.val)) );
            ZCODE( int nz = round((Z-z)/(zmax.val-zmin.val)) );

            double dist = R - sqrt( SUMD(SQR(x-X + nx*(xmax.val-xmin.val)),
                                         SQR(y-Y + ny*(ymax.val-ymin.val)),
                                         SQR(z-Z + nz*(zmax.val-zmin.val))) );

            sum = MIN(sum, -dist);
          }

          return sum;
        }
      default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
    }
  }
} phi_agar_cf;

class phi_free_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (nb_geometry.val)
    {
      case 0:
        return (x-(MAX(0.,h_agar.val)+h_biof.val));
      case 1:
        {
          double xc = xmin.val +.5*(xmax.val-xmin.val);
          double yc = ymin.val +.5*(ymax.val-ymin.val);
          double zc = zmin.val +.5*(zmax.val-zmin.val);
          return -(MAX(0.,h_agar.val)+h_biof.val) + sqrt( SUMD(SQR(x-xc), SQR(y-yc), SQR(z-zc)) );
        }
      case 2:
        {
          double xc0 = xmin.val +.3*(xmax.val-xmin.val), xc1 = xmin.val +.5*(xmax.val-xmin.val), xc2 = xmin.val +.6*(xmax.val-xmin.val);
          double yc0 = ymin.val +.5*(ymax.val-ymin.val), yc1 = ymin.val +.6*(ymax.val-ymin.val), yc2 = ymin.val +.4*(ymax.val-ymin.val);
          double zc0 = zmin.val +.6*(zmax.val-zmin.val), zc1 = zmin.val +.4*(zmax.val-zmin.val), zc2 = zmin.val +.3*(zmax.val-zmin.val);

          return -MAX((MAX(0.,h_agar.val)+h_biof.val) - sqrt( SUMD(SQR(x-xc0), SQR(y-yc0), SQR(z-zc0)) ),
                      (MAX(0.,h_agar.val)+h_biof.val) - sqrt( SUMD(SQR(x-xc1), SQR(y-yc1), SQR(z-zc1)) ),
                      (MAX(0.,h_agar.val)+h_biof.val) - sqrt( SUMD(SQR(x-xc2), SQR(y-yc2), SQR(z-zc2)) ));
        }
      case 3:
        {
          double xc = xmin.val +.5*(xmax.val-xmin.val);
          double yc = ymin.val +.5*(ymax.val-ymin.val);
          double zc = zmin.val +.5*(zmax.val-zmin.val);
          return (MAX(0.,h_agar.val)-h_biof.val) - sqrt( SUMD(SQR(x-xc), SQR(y-yc), SQR(z-zc)) );
        }
      case 4:
      {
        double plane = (x-(MAX(0., h_agar.val) + h_biof.val));
        double bump = -(.1*(xmax.val-xmin.val) - sqrt( SQR(x-(MAX(0., h_agar.val) + h_biof.val)) + SQR(y-0.5*(ymax.val+ymin.val))));
        return .5*(plane+bump - sqrt(SQR(plane-bump) + .001*(xmax.val-xmin.val)));
      }
//        {
//          double plane = (x-(MAX(0., h_agar.val) + h_biof.val));

//          double xc = (MAX(0., h_agar.val) + h_biof.val);
//          double yc = ymin.val +.5*(ymax.val-ymin.val);
//          double zc = zmin.val +.5*(zmax.val-zmin.val);

//          double bump = -.1*(xmax.val-xmin.val) + sqrt( SUMD(SQR(x-xc), SQR(y-yc), SQR(z-zc)) );

//          return .5*( plane + bump - sqrt( SQR(plane-bump) + SQR(.01*(xmax.val-xmin.val)) ) );
//        }
      case 5:
        return (x-(MAX(0.,h_agar.val)+h_biof.val));
      case 6:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num.val; ++i)
          {
            double R = (MAX(0.,h_agar.val)+h_biof.val) * (1. + grain_dispersity.val*(((double) rand() / (double) RAND_MAX)));
            double X = xmin.val + ((double) rand() / (double) RAND_MAX) *(xmax.val-xmin.val);
            double Y = ymin.val + ((double) rand() / (double) RAND_MAX) *(ymax.val-ymin.val);
            double Z = zmin.val + ((double) rand() / (double) RAND_MAX) *(zmax.val-zmin.val);

            XCODE( int nx = round((X-x)/(xmax.val-xmin.val)) );
            YCODE( int ny = round((Y-y)/(ymax.val-ymin.val)) );
            ZCODE( int nz = round((Z-z)/(zmax.val-zmin.val)) );

            double dist = R - sqrt( SUMD(SQR(x-X + nx*(xmax.val-xmin.val)),
                                         SQR(y-Y + ny*(ymax.val-ymin.val)),
                                         SQR(z-Z + nz*(zmax.val-zmin.val))) );

            sum = MIN(sum, -dist);
          }

          return sum;
        }
      case 7:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num.val; ++i)
          {
            double R = (MAX(0.,h_agar.val)+h_biof.val) * (1. + grain_dispersity.val*(((double) rand() / (double) RAND_MAX)));
            double X = xmin.val + ((double) rand() / (double) RAND_MAX) *(xmax.val-xmin.val);
            double Y = ymin.val + ((double) rand() / (double) RAND_MAX) *(ymax.val-ymin.val);
            double Z = zmin.val + ((double) rand() / (double) RAND_MAX) *(zmax.val-zmin.val);

            XCODE( int nx = round((X-x)/(xmax.val-xmin.val)) );
            YCODE( int ny = round((Y-y)/(ymax.val-ymin.val)) );
            ZCODE( int nz = round((Z-z)/(zmax.val-zmin.val)) );

            double dist = R - sqrt( SUMD(SQR(x-X + nx*(xmax.val-xmin.val)),
                                         SQR(y-Y + ny*(ymax.val-ymin.val)),
                                         SQR(z-Z + nz*(zmax.val-zmin.val))) );

            sum = MIN(sum, -dist);
          }

          return -sum;
        }
      default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
    }
  }
} phi_free_cf;

class phi_biof_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const { return MAX(phi_agar_cf(DIM(x,y,z)), phi_free_cf(DIM(x,y,z))); }
} phi_biof_cf;

class bc_wall_type_t : public WallBCDIM {
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z)) const
  {
    double pa = phi_agar_cf(DIM(x,y,z));
    double pf = phi_free_cf(DIM(x,y,z));
    if (pa < 0 && pf < 0) return bc_biof.val;
    else if (pa > 0 && pf > 0) throw;
    else if (pa > 0)      return bc_agar.val;
    else if (pf > 0)      return bc_free.val;
  }
} bc_wall_type;

class bc_wall_value_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    double pa = phi_agar_cf(DIM(x,y,z));
    double pf = phi_free_cf(DIM(x,y,z));
    if (pa < 0 && pf < 0) return bc_biof.val == NEUMANN ? 0 : C0b.val;
    else if (pa > 0 && pf > 0) throw;
    else if (pa > 0)      return bc_agar.val == NEUMANN ? 0 : C0a.val;
    else if (pf > 0)      return bc_free.val == NEUMANN ? 0 : C0f.val;
  }
} bc_wall_value;

class initial_concentration_free_t : public CF_DIM {
public:
  double operator()(DIM(double , double, double )) const {
    return C0f.val;
  }
} initial_concentration_free;

class initial_concentration_agar_t : public CF_DIM {
public:
  double operator()(DIM(double , double, double )) const {
    return C0a.val;
  }
} initial_concentration_agar;

class initial_concentration_biof_t : public CF_DIM {
public:
  double operator()(DIM(double , double, double )) const {
    return C0b.val;
  }
} initial_concentration_biof;

class f_cf_t : public CF_1 {
public:
  double operator()(double c) const { return A.val*c/(Kc.val+c); }
} f_cf;

class fc_cf_t : public CF_1 {
public:
  double operator()(double c) const { return A.val*Kc.val/pow(Kc.val+c, 2.); }
} fc_cf;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);

  nb_experiment.set_from_cmd(cmd);
  setup_experiment();
  set_periodicity();

  pl.set_from_cmd_all(cmd);

  // scale computational box
  double scaling = 1/box_size.val;

  rho.val /= (scaling*scaling*scaling);

  Da.val *= (scaling*scaling);
  Db.val *= (scaling*scaling);
  Df.val *= (scaling*scaling);

  sigma.val /= (scaling);

  A.val /= (scaling*scaling*scaling);
  Kc.val /= (scaling*scaling*scaling);

  C0a.val /= (scaling*scaling*scaling);
  C0b.val /= (scaling*scaling*scaling);
  C0f.val /= (scaling*scaling*scaling);

  lambda.val *= (scaling*scaling*scaling*scaling);

  parStopWatch w1;
  w1.start("total time");

  /* create the p4est */
  my_p4est_brick_t brick;
  double xyz_min [] = { DIM(xmin.val, ymin.val, zmin.val) };
  double xyz_max [] = { DIM(xmax.val, ymax.val, zmax.val) };
  int periodic   [] = { DIM(px.val,   py.val,   pz.val) };
  int nxyz       [] = { DIM(nx.val,   ny.val,   nz.val) };

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nxyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin.val, lmax.val, &phi_biof_cf, lip.val);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_tree = p4est->connectivity->vertices[3*vm + 0];
  double ymin_tree = p4est->connectivity->vertices[3*vm + 1];
  double xmax_tree = p4est->connectivity->vertices[3*vp + 0];
  double ymax_tree = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax_tree-xmin_tree) / pow(2., (double) data.max_lvl);
  double dy = (ymax_tree-ymin_tree) / pow(2., (double) data.max_lvl);
#ifdef P4_TO_P8
  double zmin_tree = p4est->connectivity->vertices[3*vm + 2];
  double zmax_tree = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax_tree-zmin_tree) / pow(2.,(double) data.max_lvl);
#endif

  double dt_max =  0.05*(Kc.val+C0b.val)/A.val;

  // experimentally determined time-step restriction due to surface tension
//  double dt_max = 0.5e-10*pow(2., 3.*(11.-lmax))/MAX(sigma, 1.e-30);

  /* initial geometry */
  Vec phi_free; ierr = VecCreateGhostNodes(p4est, nodes, &phi_free); CHKERRXX(ierr);
  Vec phi_agar; ierr = VecCreateGhostNodes(p4est, nodes, &phi_agar); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_free_cf, phi_free);
  sample_cf_on_nodes(p4est, nodes, phi_agar_cf, phi_agar);

  // initial air-biofilm interface perturbation
  {
    double *phi_free_ptr;
    ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

    srand(mpi.rank());

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      phi_free_ptr[n] += init_perturb.val*dx*(double)(rand()%1000)/1000.;
    }

    ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(phi_free, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (phi_free, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (nb_geometry.val == 6)
  {
    shift_ghosted_vec(phi_free, -grain_smoothing.val);
    shift_ghosted_vec(phi_agar, +grain_smoothing.val);
  }

  if (nb_geometry.val == 7)
  {
    shift_ghosted_vec(phi_free, +grain_smoothing.val);
    shift_ghosted_vec(phi_agar, -grain_smoothing.val);
  }

  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi_free);
  ls.reinitialize_1st_order_time_2nd_order_space(phi_agar);

  if (nb_geometry.val == 6)
  {
    shift_ghosted_vec(phi_free, +grain_smoothing.val);
    shift_ghosted_vec(phi_agar, -grain_smoothing.val);
    copy_ghosted_vec(phi_free, phi_agar);
    invert_phi(nodes, phi_agar);
    shift_ghosted_vec(phi_agar, -h_biof.val);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_agar);
  }

  if (nb_geometry.val == 7)
  {
    shift_ghosted_vec(phi_free, -grain_smoothing.val);
    shift_ghosted_vec(phi_agar, +grain_smoothing.val);
    if (h_agar.val > 0)
    {
      copy_ghosted_vec(phi_agar, phi_free);
      invert_phi(nodes, phi_free);
      shift_ghosted_vec(phi_free, -h_biof.val);
      ls.reinitialize_1st_order_time_2nd_order_space(phi_free);
    } else {
      set_ghosted_vec(phi_agar, -1);
    }
  }

  /* initial concentration */
  Vec Ca; ierr = VecDuplicate(phi_free, &Ca); CHKERRXX(ierr);
  Vec Cb; ierr = VecDuplicate(phi_free, &Cb); CHKERRXX(ierr);
  Vec Cf; ierr = VecDuplicate(phi_free, &Cf); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, initial_concentration_agar, Ca);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_biof, Cb);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_free, Cf);

  /* initialize the solver */
  my_p4est_biofilm_t biofilm_solver(ngbd);

  // model parameters
  biofilm_solver.set_velocity_type(velocity_type.val);
  biofilm_solver.set_parameters   (Da.val, Db.val, Df.val, sigma.val, rho.val, lambda.val, gam.val, scaling);
  biofilm_solver.set_kinetics     (f_cf, fc_cf);
  biofilm_solver.set_bc           (bc_wall_type, bc_wall_value);
  biofilm_solver.set_steady_state (steady_state.val);

  // time discretization parameters
  biofilm_solver.set_advection_scheme(advection_scheme.val);
  biofilm_solver.set_time_scheme     (time_scheme.val);
  biofilm_solver.set_dt_max          (dt_max);
  biofilm_solver.set_cfl             (cfl_number.val);

  // parameters for solving non-linear equation
  biofilm_solver.set_iteration_scheme(iteration_scheme.val);
  biofilm_solver.set_max_iterations  (max_iterations.val);
  biofilm_solver.set_tolerance       (tolerance.val);

  // general poisson solver parameter
  biofilm_solver.set_use_sc_scheme(use_sc_scheme.val);
  biofilm_solver.set_use_taylor_correction(use_taylor_correction.val);
  biofilm_solver.set_integration_order(integration_order.val);
  biofilm_solver.set_curvature_smoothing(curvature_smoothing.val, curvature_smoothing_steps.val);

  // initial geometry and concentrations
  biofilm_solver.set_phi(phi_free, phi_agar);
  biofilm_solver.set_concentration(Ca, Cb, Cf);

  ierr = VecDestroy(phi_free); CHKERRXX(ierr);
  ierr = VecDestroy(phi_agar); CHKERRXX(ierr);

  ierr = VecDestroy(Ca); CHKERRXX(ierr);
  ierr = VecDestroy(Cb); CHKERRXX(ierr);
  ierr = VecDestroy(Cf); CHKERRXX(ierr);

  // loop over time
  double tn = 0;
  int iteration = 0;

  FILE *fich;
  char name[10000];

  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
#ifdef P4_TO_P8
  sprintf(name, "%s/data_%dx%dx%d_box_%g_level_%d-%d.dat", out_dir, nxyz[0], nxyz[1], nxyz[2], box_size.val, lmin.val, lmax.val);
#else
  sprintf(name, "%s/data_%dx%d_box_%g_level_%d-%d.dat", out_dir, nxyz[0], nxyz[1], box_size.val, lmin.val, lmax.val);
#endif

  if(save_data.val)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "time average_interface_velocity max_interface_velocity interface_length biofilm_area time_elapsed iteration\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  bool keep_going = true;

  int vtk_idx = 0;

  double total_growth = 0;
  double base = 0.1;
  double nutrient_left = 1;
  double nutrient_init = 1;
  double wall_proximity = MIN(xmax.val-xmin.val, ymax.val-ymin.val);

  biofilm_solver.update_grid();

  while(keep_going)
  {
    if (steady_state.val && iteration == 0)
    {
      biofilm_solver.solve_concentration();
    }

    ierr = PetscPrintf(mpi.comm(), "Solving for pressure\n"); CHKERRXX(ierr);
    biofilm_solver.solve_pressure();
    ierr = PetscPrintf(mpi.comm(), "Calculating velocity\n"); CHKERRXX(ierr);
    biofilm_solver.compute_velocity_from_pressure();

    // compute how far the air-biofilm interface has advanced
    {
      total_growth = base;

      p4est = biofilm_solver.get_p4est();
      nodes = biofilm_solver.get_nodes();
      phi_free = biofilm_solver.get_phi_free();

      Vec C = biofilm_solver.get_C();
      Vec ones;
      ierr = VecDuplicate(phi_free, &ones); CHKERRXX(ierr);
      set_ghosted_vec(ones, -1);

      nutrient_left = integrate_over_negative_domain(p4est, nodes, ones, C);

      ierr = VecDestroy(ones); CHKERRXX(ierr);

      if (iteration == 1) nutrient_init = nutrient_left;

      wall_proximity = MIN(xmax.val-xmin.val, ymax.val-ymin.val);

      const double *phi_free_ptr;
      ierr = VecGetArrayRead(phi_free, &phi_free_ptr); CHKERRXX(ierr);
      foreach_local_node(n, nodes)
      {
        if (phi_free_ptr[n] < 0)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);
          total_growth = MAX(total_growth, xyz[0]);
        }

        if (phi_free_ptr[n] > 0)
        {
          p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
          if (is_node_Wall(p4est, ni))
          {
            wall_proximity = MIN(wall_proximity, phi_free_ptr[n]);
          }
        }
      }
      ierr = VecRestoreArrayRead(phi_free, &phi_free_ptr); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
          mpiret = MPI_Allreduce(MPI_IN_PLACE, &wall_proximity, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      total_growth -= base;
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration %d, growth %e, nutrient left %e, wall proximity %e, time %e\n", iteration, total_growth, nutrient_left/nutrient_init, wall_proximity, tn); CHKERRXX(ierr);

    // determine to save or not
    bool save_now =
        (save_type.val == 0 && iteration    >= vtk_idx*save_every_dn.val) ||
        (save_type.val == 1 && total_growth >= vtk_idx*save_every_dl.val) ||
        (save_type.val == 2 && tn           >= vtk_idx*save_every_dt.val);

    // save velocity, area of interface and volume of biofilm
    if(save_data.val && save_now)
    {
      p4est = biofilm_solver.get_p4est();
      nodes = biofilm_solver.get_nodes();
      phi_free = biofilm_solver.get_phi_free();
      Vec vn = biofilm_solver.get_vn();
      Vec phi_biof = biofilm_solver.get_phi_biof();

      Vec ones;
      ierr = VecDuplicate(phi_free, &ones); CHKERRXX(ierr);
      set_ghosted_vec(ones, 1);

      // calculate the length of the interface and solid phase area
      double interface_area = integrate_over_interface(p4est, nodes, phi_free, ones);
      double biofilm_volume = integrate_over_negative_domain(p4est, nodes, phi_biof, ones);

      ierr = VecDestroy(ones); CHKERRXX(ierr);

#ifdef P4_TO_P8
      interface_area /= (scaling*scaling);
      biofilm_volume /= (scaling*scaling*scaling);
#else
      interface_area /= (scaling);
      biofilm_volume /= (scaling*scaling);
#endif

      double avg_velo = integrate_over_interface(p4est, nodes, phi_free, vn) / interface_area;


      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      double time_elapsed = w1.read_duration_current();

      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e %e %d\n", tn, avg_velo/scaling, biofilm_solver.get_vn_max()/scaling, interface_area, biofilm_volume, time_elapsed, iteration);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved data in %s\n", name); CHKERRXX(ierr);
    }

    keep_going = keep_going
                 && (iteration < limit_iter.val)
                 && (total_growth < limit_length.val)
                 && (nutrient_left > limit_nutrient.val*nutrient_init)
                 && (wall_proximity > limit_wall.val);

    // save field data
    if(save_vtk.val && save_now)
    {
      biofilm_solver.save_VTK(vtk_idx);
    }

    if (save_now) vtk_idx++;

    biofilm_solver.compute_dt();

    if (tn + biofilm_solver.get_dt() > limit_time.val)
    {
      biofilm_solver.set_dt_max(limit_time.val-tn);
      keep_going = false;
    }

    tn += biofilm_solver.get_dt();
    biofilm_solver.update_grid();
    biofilm_solver.solve_concentration();

    iteration++;
  }

  w1.stop(); w1.read_duration();

  return 0;
}
