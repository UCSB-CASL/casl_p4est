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

#undef MIN
#undef MAX

#define ADD_OPTION(i, var, description) \
  i == 0 ? cmd.add_option(#var, description) : (void) (var = cmd.get(#var, var));

#define ADD_OPTION2(i, var, name, description) \
  i == 0 ? cmd.add_option(name, description) : (void) (var = cmd.get(name, var));

// grid parameters
int lmin = 5;
int lmax = 9;
double lip = 2;

#ifdef P4_TO_P8
char direction = 'z';
#else
char direction = 'y';
#endif

double xmin = 0, xmax = 1; int nx = 1; bool px = 0;
double ymin = 0, ymax = 2; int ny = 2; bool py = 1;
#ifdef P4_TO_P8
double zmin = 0, zmax = 1; int ny = 1; bool pz = 0;
#endif

// solver options
double cfl_number              = 0.1;
double bc_tolerance            = 1.e-5;
double phi_thresh              = 1e-3;

int max_iterations             = 6;
int pin_every_n_steps          = 1000;

bool use_continuous_stencil    = 0;
bool use_one_sided_derivatives = 0;
bool use_points_on_interface   = 1;
bool update_c0_robin           = 1;
bool use_superconvergent_robin = 1;
bool zero_negative_velocity    = 0;

bool shift_grids = 0;
int  phi_grid_refinement = 0;

// not implemented yet
bool use_superconvergent_jump  = false;

// output parameters
int save_every_n_iteration = 10;
bool save_velocity         = 1;
bool save_vtk              = 1;
bool save_history          = 1;
bool save_dendrites        = 0;

double save_every_dl = 0.01;
double save_every_dt = 0.1;

int save_type = 0; // 0 - every n iterations, 1 - every dl of growth, 2 - every dt of time

double dendrite_cut_off_fraction = 1.05;
double dendrite_min_length       = 0.05;

using namespace std;

// problem parameters
bool concentration_neumann = 1;
int max_total_iterations   = 10;
double time_limit          = DBL_MAX;
double termination_length  = 1.8;
double init_perturb        = 0.001;

bool enforce_planar_front   = 0;

int alloy_type = 0;

double box_size = 4e-2;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
//double box_size = 1e-1;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1./box_size;

double rho;                  /* density                                    - kg.cm-3      */
double heat_capacity;        /* c, heat capacity                           - J.kg-1.K-1   */
double Tm;                   /* melting temperature                        - K            */
double G;                    /* thermal gradient                           - k.cm-1       */
double V;                    /* cooling velocity                           - cm.s-1       */
double latent_heat;          /* L, latent heat                             - J.cm-3       */
double thermal_conductivity; /* k, thermal conductivity                    - W.cm-1.K-1   */
double lambda;               /* thermal diffusivity                        - cm2.s-1      */

double eps_c;                /* curvature undercooling coefficient         - cm.K         */
double eps_v;                /* kinetic undercooling coefficient           - s.K.cm-1     */
double eps_anisotropy;       /* anisotropy coefficient                                    */

double ml0;                   /* liquidus slope                             - K / at frac. */
double kp0;                   /* partition coefficient                                     */
double c00;                   /* initial concentration                      - at frac.     */
double Dl0;                   /* liquid concentration diffusion coefficient - cm2.s-1      */

double ml1;                   /* liquidus slope                             - K / at frac. */
double kp1;                   /* partition coefficient                                     */
double c01;                   /* initial concentration                      - at frac.     */
double Dl1;                   /* liquid concentration diffusion coefficient - cm2.s-1      */


void set_alloy_parameters()
{
  switch(alloy_type)
  {
    case 0:
      /* Ni - 0.2at%Cu - 0.2at%Cu */
      rho                  = 8.88e-3;        /* kg.cm-3    */
      heat_capacity        = 0.46e3;         /* J.kg-1.K-1 */
      Tm                   = 1728;           /* K           */
      G                    = 4e2;            /* k.cm-1      */
      V                    = 0.01;           /* cm.s-1      */
      latent_heat          = 2350;           /* J.cm-3      */
      thermal_conductivity = 6.07e-1;        /* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */

      eps_c                = 2.7207e-5;
      eps_v                = 2.27e-2;
      eps_anisotropy       = 0.05;

      ml0                   =-357;            /* K / at frac. - liquidous slope */
      kp0                   = 0.86;           /* partition coefficient */
      c00                   = 0.2;            /* at frac.    */
      Dl0                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      ml1                   =-357;            /* K / at frac. - liquidous slope */
      kp1                   = 0.86;           /* partition coefficient */
      c01                   = 0.2;            /* at frac.    */
      Dl1                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

//      box_size = 4e-2;

      break;
    case 1:
      /* Ni - 15.2wt%Al - 5.8wt%Ta  */
      rho            = 7.365e-3;  /* kg.cm-3    */
      heat_capacity  = 660;       /* J.kg-1.K-1 */
      Tm             = 1754;      /* K           */
      G              = 200;       /* K.cm-1      */
      V              = 0.01;      /* cm.s-1      */
      latent_heat    = 2136;      /* J.cm-3      */
      thermal_conductivity =  0.8;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 2.7207e-4;
      eps_v          = 2.27e-2;
      eps_anisotropy = 0.05;

      Dl0 = 5e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml0 =-255;       /* K / wt frac. - liquidous slope */
      c00 = 0.152;     /* wt frac.    */
      kp0 = 0.48;      /* partition coefficient */

      Dl1 = 5e-5;
      ml1 =-517;
      c01 = 0.058;
      kp1 = 0.54;

//      box_size = 2e-1;
      break;

    case 2:
      /* Co - 10.7at%W - 9.4at%Al  */
      rho            = 9.2392e-3;   /* kg.cm-3    */
      heat_capacity  = 356;         /* J.kg-1.K-1 */
      Tm             = 1996;        /* K           */
      G              = 1000;         /* K.cm-1      */
      V              = 0.005;        /* cm.s-1      */
      latent_heat    = 2588.7;      /* J.cm-3      */
      thermal_conductivity =  1.3;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 0*2.7207e-5;
      eps_v          = 0*2.27e-2;
      eps_anisotropy = 0.025;

      Dl0 = 1e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml0 =-874;       /* K / wt frac. - liquidous slope */
      c00 = 0.107;     /* at frac.    */
      kp0 = 0.848;     /* partition coefficient */

      Dl1 = 10e-5;
      ml1 =-1378;
      c01 = 0.094;
      kp1 = 0.848;

      box_size = 1.0e-1;

      break;

    case 3:
      /* Co - 9.4at%Al - 10.7at%W  */
      rho            = 9.2392e-3;   /* kg.cm-3    */
      heat_capacity  = 356;         /* J.kg-1.K-1 */
      Tm             = 1996;        /* K           */
      G              = 5;         /* K.cm-1      */
      V              = 0.01;        /* cm.s-1      */
      latent_heat    = 2588.7;      /* J.cm-3      */
      thermal_conductivity =  1.3;/* W.cm-1.K-1  */
      lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
      eps_c          = 2.7207e-5;
      eps_v          = 2.27e-2;
      eps_anisotropy = 0.05;

      Dl0 = 1e-5;
      ml0 =-1378;
      c00 = 0.094;
      kp0 = 0.848;

      Dl1 = 1e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
      ml1 =-874;       /* K / wt frac. - liquidous slope */
      c01 = 0.107;     /* at frac.    */
      kp1 = 0.848;     /* partition coefficient */

      box_size = 0.5e-1;
      break;
  }
}


#ifdef P4_TO_P8
struct plan_t : CF_3{
  double operator()(double x, double y, double z) const {
    if     (direction=='x') return -(x - 0.1);
    else if(direction=='y') return -(y - 0.1);
    else                    return -(z - 0.1);
  }
} LS;

class WallBCTypeTemperature : public WallBC3D
{
public:
  BoundaryConditionType operator()( double , double , double ) const
  {
    return NEUMANN;
  }
} wall_bc_type_temperature;

class WallBCValueTemperature : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(direction=='x')
    {
      if (ABS(x-xmax)<EPS)
        return G;

      if (ABS(x-xmin)<EPS)
        return -G - V*latent_heat/thermal_conductivity;
    }
    else if(direction=='y')
    {
      if (ABS(y-ymax)<EPS)
        return G;

      if (ABS(y-ymin)<EPS)
        return -G - V*latent_heat/thermal_conductivity;
    }
    else
    {
      if (ABS(z-zmax)<EPS)
        return G;

      if (ABS(z-zmin)<EPS)
        return -G - V*latent_heat/thermal_conductivity;
    }

    return 0;
  }
} wall_bc_value_temperature;

class WallBCTypeConcentration : public WallBC3D
{
public:
  BoundaryConditionType operator()( double x, double y, double z ) const
  {
    if (concentration_neumann)
      return NEUMANN;

    if(direction=='x')
    {
      if (ABS(x-xmin)<EPS || ABS(x-xmax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
    else if(direction=='y')
    {
      if (ABS(y-ymin)<EPS || ABS(y-ymax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
    else
    {
      if (ABS(z-zmin)<EPS || ABS(z-zmax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  }
} wall_bc_type_concentration;

class WallBCValueConcentration0 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (wall_bc_type_concentration(x,y,z)==NEUMANN)
      return 0;
    else
      return c00;
  }
} wall_bc_value_concentration_0;

class InitialConcentration0 : public CF_3
{
public:
  double operator()(double , double , double ) const
  {
    return c00;
  }
} initial_concentration_0;

class WallBCValueConcentration1 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (wall_bc_type_concentration(x,y,z)==NEUMANN)
      return 0;
    else
      return c01;
  }
} wall_bc_value_concentration_1;

class InitialConcentration1 : public CF_3
{
public:
  double operator()(double , double , double ) const
  {
    return c01;
  }
} initial_concentration_1;

class InitialTemperature : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
//    if(LS(x,y,z)>0) return -LS(x,y,z)*(G+latent_heat*V/thermal_conductivity) + initial_concentration_0(x,y,z)*ml0 + initial_concentration_1(x,y,z)*ml1 + Tm;
//    else
      return -LS(x,y,z)*G + initial_concentration_0(x,y,z)*ml0 + initial_concentration_1(x,y,z)*ml1 + Tm;
  }
} initial_temperature;

#else

struct plan_t : CF_2{
  double operator()(double x, double y) const {
    if(direction=='x') return -(x - 0.1);
    else               return -(y - 0.1) + 0.0*sin(10*PI*x);
  }
} LS;


class WallBCTypeTemperature : public WallBC2D
{
public:
  BoundaryConditionType operator()( double, double ) const
  {
    return NEUMANN;
  }
} wall_bc_type_temperature;

class WallBCValueTemperature : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(direction=='x')
    {
      if (ABS(x-xmax)<EPS)
        return G;

      if (ABS(x-xmin)<EPS)
        return -G - V*latent_heat/thermal_conductivity;
    }
    else
    {
      if (ABS(y-ymax)<EPS)
        return G;

      if (ABS(y-ymin)<EPS)
        return -G - V*latent_heat/thermal_conductivity;
    }

    return 0;
  }
} wall_bc_value_temperature;

class WallBCTypeConcentration : public WallBC2D
{
public:
  BoundaryConditionType operator()( double x, double y ) const
  {
    if (concentration_neumann)
      return NEUMANN;

    if(direction=='x')
    {
      if (ABS(x-xmin)<EPS || ABS(x-xmax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
    else
    {
      if (ABS(y-ymax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  }
} wall_bc_type_concentration;

class WallBCValueConcentration0 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (wall_bc_type_concentration(x,y)==NEUMANN)
      return 0;
    else
      return c00;
  }
} wall_bc_value_concentration_0;

class InitialConcentration0 : public CF_2
{
public:
  double operator()(double , double ) const
  {
    return c00;
  }
} initial_concentration_0;

class InitialConcentration0_solid : public CF_2
{
public:
  double operator()(double , double ) const
  {
    return kp0*c00;
  }
} initial_concentration_0_solid;

class WallBCValueConcentration1 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (wall_bc_type_concentration(x,y)==NEUMANN)
      return 0;
    else
      return c01;
  }
} wall_bc_value_concentration_1;

class InitialConcentration1 : public CF_2
{
public:
  double operator()(double , double ) const
  {
    return c01;
  }
} initial_concentration_1;

class InitialConcentration1_solid : public CF_2
{
public:
  double operator()(double , double ) const
  {
    return kp1*c01;
  }
} initial_concentration_1_solid;

class InitialTemperature : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    if(LS(x,y)>0) return -LS(x,y)*(G+latent_heat*V/thermal_conductivity) + initial_concentration_0(x,y)*ml0 + initial_concentration_1(x,y)*ml1 + Tm;
//    else
      return -LS(x,y)*G + initial_concentration_0(x,y)*ml0 + initial_concentration_1(x,y)*ml1 + Tm;
  }
} initial_temperature;

#endif

#ifdef P4_TO_P8
class eps_c_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_c*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_c_cf;

class eps_v_cf_t : public CF_3
{
public:
  double operator()(double nx, double ny, double nz) const
  {
    double norm = sqrt(nx*nx+ny*ny+nz*nz) + EPS;
    return eps_v*(1.0-4.0*eps_anisotropy*((pow(nx, 4.) + pow(ny, 4.) + pow(nz, 4.))/pow(norm, 4.) - 0.75));
  }
} eps_v_cf;
#else
class eps_c_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_c*(1.-15.*eps_anisotropy*cos(4.*(theta)));
  }
} eps_c_cf;

class eps_v_cf_t : public CF_2
{
public:
  double operator()(double nx, double ny) const
  {
    double theta = atan2(ny, nx);
    return eps_v*(1.-15.*eps_anisotropy*cos(4.*(theta)));
  }
} eps_v_cf;
#endif


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  for (short i = 0; i < 2; ++i)
  {
    // grid parameters
    ADD_OPTION(i, lmin, "min level of the tree");
    ADD_OPTION(i, lmax, "max level of the tree");
    ADD_OPTION(i, lip,  "Lipschitz constant");

    ADD_OPTION(i, direction, "direction of the crystal growth x/y");

    ADD_OPTION(i, nx, "number of blox in x-dimension");
    ADD_OPTION(i, ny, "number of blox in y-dimension");
#ifdef P4_TO_P8
    ADD_OPTION(i, nz, "number of blox in z-dimension");
#endif

    ADD_OPTION(i, px, "periodicity in x-dimension 0/1");
    ADD_OPTION(i, py, "periodicity in y-dimension 0/1");
#ifdef P4_TO_P8
    ADD_OPTION(i, pz, "periodicity in z-dimension 0/1");
#endif

    ADD_OPTION(i, xmin, "xmin"); ADD_OPTION(i, xmax, "xmax");
    ADD_OPTION(i, ymin, "ymin"); ADD_OPTION(i, ymax, "ymax");
#ifdef P4_TO_P8
    ADD_OPTION(i, zmin, "zmin"); ADD_OPTION(i, zmax, "zmax");
#endif

    // solver parameters
    ADD_OPTION(i, cfl_number,   "cfl_number");
    ADD_OPTION(i, bc_tolerance, "error tolerance for internal iterations");
    ADD_OPTION(i, phi_thresh,   "phi_thresh");

    ADD_OPTION(i, max_iterations,    "max_iterations");
    ADD_OPTION(i, pin_every_n_steps, "pin_every_n_steps");

    ADD_OPTION(i, use_continuous_stencil,    "use_continuous_stencil");
    ADD_OPTION(i, use_one_sided_derivatives, "use_one_sided_derivatives");
    ADD_OPTION(i, use_points_on_interface,   "use_points_on_interface");
    ADD_OPTION(i, update_c0_robin,           "update_c0_robin");
    ADD_OPTION(i, use_superconvergent_robin, "use_superconvergent_robin");
    ADD_OPTION(i, use_superconvergent_jump,  "use_superconvergent_jump");
    ADD_OPTION(i, zero_negative_velocity,    "zero_negative_velocity");

    ADD_OPTION(i, shift_grids,         "shift_grids");
    ADD_OPTION(i, phi_grid_refinement, "phi_grid_refinement");

    // output parameters
    ADD_OPTION(i, save_every_n_iteration, "save vtk every n iteration");
    ADD_OPTION(i, save_velocity,          "1 to save velocity of the interface, 0 otherwise");
    ADD_OPTION(i, save_vtk,               "1 to save vtu files, 0 otherwise");
    ADD_OPTION(i, save_history,           "save_history");
    ADD_OPTION(i, save_dendrites,         "save_dendrites");
    ADD_OPTION(i, save_every_dl,          "save vtk every dl of growth");
    ADD_OPTION(i, save_every_dt,          "save vtk every dt of time");
    ADD_OPTION(i, save_type,              "save criterion: 0 - every n iterations, 1 - every dl of growth, 2 - every dt of time");

    ADD_OPTION(i, dendrite_cut_off_fraction, "dendrite_cut_off_fraction");
    ADD_OPTION(i, dendrite_min_length,       "dendrite_min_length");

    // problem parameters
    ADD_OPTION(i, concentration_neumann, "concentration_neumann");
    ADD_OPTION(i, max_total_iterations,  "max_total_iterations");
    ADD_OPTION(i, time_limit,            "final time");
    ADD_OPTION(i, termination_length,    "defines when a run will be stopped (fraction of box length, from 0 to 1)");
    ADD_OPTION(i, init_perturb,          "init_perturb");
    ADD_OPTION(i, enforce_planar_front,   "enforce_planar_front");

    ADD_OPTION(i, alloy_type,  "choose the type of alloy. Default is 0.\n  0 - NiCuCu\n  1 - NiAlTa");
    if (i == 1) set_alloy_parameters();

    ADD_OPTION(i, box_size,    "set box_size");
    ADD_OPTION(i, latent_heat, "latent heat");
    ADD_OPTION(i, G,           "heat gradient");
    ADD_OPTION(i, V,           "cooling velocity");
    ADD_OPTION(i, Dl0,         "solute no. 1 diffusivity in liquid");
    ADD_OPTION(i, Dl1,         "solute no. 2 diffusivity in liquid");

    ADD_OPTION(i, eps_c, "curvature undercooling coefficient");
    ADD_OPTION(i, eps_v, "kinetic undercooling coefficient");

    if (i == 0) cmd.parse(argc, argv);
  }

  int periodic[P4EST_DIM];
  periodic[0] = cmd.get("px", (direction=='y' || direction=='z') ? 1 : 0);
  periodic[1] = cmd.get("py", (direction=='x' || direction=='z') ? 1 : 0);
#ifdef P4_TO_P8
  periodic[2] = cmd.get("pz", (direction=='x' || direction=='y') ? 1 : 0);
#endif

  int n_xyz[P4EST_DIM];
  n_xyz[0] = nx;
  n_xyz[1] = ny;
#ifdef P4_TO_P8
  n_xyz[2] = nz;
#endif

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

  PetscErrorCode ierr;

  double latent_heat_orig = latent_heat;
  double G_orig = G;
  double V_orig = V;

  scaling = 1/box_size;
  rho                  /= (scaling*scaling*scaling);
  thermal_conductivity /= scaling;
  G                    /= scaling;
  V                    *= scaling;
  latent_heat          /= (scaling*scaling*scaling);
  eps_c                *= scaling;
  eps_v                /= scaling;
  lambda                = thermal_conductivity/(rho*heat_capacity);

  Dl0                  *= (scaling*scaling);
  Dl1                  *= (scaling*scaling);

  parStopWatch w1;
  w1.start("total time");

#ifdef P4_TO_P8
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
#else
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
#endif

  /* initialize the solver */
  my_p4est_multialloy_t mas;

  mas.initialize_grid(mpi.comm(), xyz_min, xyz_max, n_xyz, periodic, LS, lmin, lmax, lip);

  p4est_t                   *p4est = mas.get_p4est();
  p4est_nodes_t             *nodes = mas.get_nodes();
  my_p4est_node_neighbors_t *ngbd  = mas.get_ngbd();

  /* initialize the variables */
  Vec phi, tl, ts, c0, c1, normal_velocity, c0s, c1s;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &tl             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &ts             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c0             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c1             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c0s            ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c1s            ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &normal_velocity); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, LS, phi);
  sample_cf_on_nodes(p4est, nodes, initial_temperature, tl);
  sample_cf_on_nodes(p4est, nodes, initial_temperature, ts);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_0, c0);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_1, c1);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_0_solid, c0s);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_1_solid, c1s);

  Vec tmp;
  ierr = VecGhostGetLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);
  ierr = VecSet(tmp, V); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);

  /* set initial time step */
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_tree = p4est->connectivity->vertices[3*vm + 0];
  double ymin_tree = p4est->connectivity->vertices[3*vm + 1];
  double xmax_tree = p4est->connectivity->vertices[3*vp + 0];
  double ymax_tree = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax_tree-xmin_tree) / pow(2., (double) lmax);
  double dy = (ymax_tree-ymin_tree) / pow(2., (double) lmax);
#ifdef P4_TO_P8
  double zmin_tree = p4est->connectivity->vertices[3*vm + 2];
  double zmax_tree = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax_tree-zmin_tree) / pow(2.,(double) lmax);
#endif

#ifdef P4_TO_P8
  double dt = 0.45*MIN(dx,dy,dz)/V;
#else
  double dt = 0.45*MIN(dx,dy)/V;
#endif

  if (enforce_planar_front) init_perturb = 0;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  srand(mpi.rank());

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    phi_p[n] += init_perturb*dx*(double)(rand()%1000)/1000.;
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  /* perturb level set */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  ls.perturb_level_set_function(phi, EPS);

  mas.set_parameters(latent_heat, thermal_conductivity, lambda,
                     V, Tm, scaling,
                     Dl0, kp0, c00, ml0,
                     Dl1, kp1, c01, ml1);
  mas.set_phi(phi);
  mas.set_bc(wall_bc_type_temperature,
             wall_bc_type_concentration,
             wall_bc_value_temperature,
             wall_bc_value_concentration_0,
             wall_bc_value_concentration_1);
  mas.set_temperature(tl, ts);
  mas.set_concentration(c0, c1, c0s, c1s);
  mas.set_normal_velocity(normal_velocity);
  mas.set_dt(dt);

  mas.set_GT(zero_cf);
  mas.set_jump_t(zero_cf);
  mas.set_jump_tn(zero_cf);
  mas.set_flux_c(zero_cf, zero_cf);
  mas.set_undercoolings(eps_v_cf, eps_c_cf);

  mas.set_rhs(zero_cf, zero_cf, zero_cf, zero_cf);

  mas.set_bc_tolerance     (bc_tolerance);
  mas.set_max_iterations   (max_iterations);
  mas.set_pin_every_n_steps(pin_every_n_steps);
  mas.set_cfl              (cfl_number);
  mas.set_phi_thresh       (phi_thresh);

  mas.set_use_continuous_stencil   (use_continuous_stencil   );
  mas.set_use_one_sided_derivatives(use_one_sided_derivatives);
  mas.set_use_superconvergent_robin(use_superconvergent_robin);
  mas.set_use_superconvergent_jump (use_superconvergent_jump );
  mas.set_use_points_on_interface  (use_points_on_interface  );
  mas.set_update_c0_robin          (update_c0_robin          );
  mas.set_zero_negative_velocity   (zero_negative_velocity   );

  mas.set_dendrite_cut_off_fraction(dendrite_cut_off_fraction);
  mas.set_dendrite_min_length(dendrite_min_length);

  mas.set_enforce_planar_front(enforce_planar_front);


//  mas.set_zero_negative_velocity(zero_negative_velocity);
//  mas.set_num_of_iterations_per_step(num_of_iters_per_step);

//  mas.compute_velocity();
//  mas.compute_dt();

  // loop over time
  double tn = 0;
  int iteration = 0;

  FILE *fich;
  char name[10000];

  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
#ifdef P4_TO_P8
  sprintf(name, "%s/velo_%dx%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], n_xyz[2], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#else
  sprintf(name, "%s/velo_%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#endif

  if(save_velocity)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "time average_interface_velocity max_interface_velocity interface_length solid_phase_area time_elapsed iteration local_nodes ghost_nodes sub_iterations\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  bool keep_going = true;
  int sub_iterations = 0;

  int vtk_idx = 0;

  double total_growth = 0;
  double base = 0.1;

  while(keep_going)
//  while (iteration < 20)
  {
    if (tn + mas.get_dt() > time_limit) { mas.set_dt(time_limit-tn); keep_going = false; }

    tn += mas.get_dt();

    sub_iterations += mas.one_step();

    {
      total_growth = base;

      p4est = mas.get_p4est();
      nodes = mas.get_nodes();
      phi = mas.get_phi();

      const double *phi_p;
      ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if (phi_p[n] > 0)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          if (direction=='x')
          {
            total_growth = MAX(total_growth, xyz[0]);
          }
          else if (direction=='y')
          {
            total_growth = MAX(total_growth, xyz[1]);
          }
#ifdef P4_TO_P8
          else if (direction=='z')
          {
            total_growth = MAX(total_growth, xyz[2]);
          }
#endif
        }
      }
      ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      total_growth -= base;
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration %d, growth %e, time %e\n", iteration, total_growth, tn); CHKERRXX(ierr);

    // determine to save or not
    bool save_now =
        (save_type == 0 && iteration    >= vtk_idx*save_every_n_iteration) ||
        (save_type == 1 && total_growth >= vtk_idx*save_every_dl) ||
        (save_type == 2 && tn           >= vtk_idx*save_every_dt);

    // save velocity, lenght of interface and area of solid phase in time
    if(save_velocity && save_now)
    {
      p4est = mas.get_p4est();
      nodes = mas.get_nodes();
      phi = mas.get_phi();
      normal_velocity = mas.get_normal_velocity();

      Vec ones;
      ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(ones, &tmp); CHKERRXX(ierr);
      ierr = VecSet(tmp, 1.); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(ones, &tmp); CHKERRXX(ierr);

      // calculate the length of the interface and solid phase area
      double interface_length = integrate_over_interface(p4est, nodes, phi, ones);
      double solid_phase_area = 1.-integrate_over_negative_domain(p4est, nodes, phi, ones);

      double avg_velo = integrate_over_interface(p4est, nodes, phi, normal_velocity) / interface_length;

      ierr = VecDestroy(ones); CHKERRXX(ierr);

      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      double time_elapsed = w1.read_duration_current();

      int num_local_nodes = nodes->num_owned_indeps;
      int num_ghost_nodes = nodes->indep_nodes.elem_count - num_local_nodes;

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &num_local_nodes, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
          mpiret = MPI_Allreduce(MPI_IN_PLACE, &num_ghost_nodes, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e %e %d %d %d %d\n", tn, avg_velo/scaling, mas.get_max_interface_velocity()/scaling, interface_length, solid_phase_area, time_elapsed, iteration, num_local_nodes, num_ghost_nodes, sub_iterations);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved velocity in %s\n", name); CHKERRXX(ierr);
      sub_iterations = 0;
    }

    // save field data
    if(save_vtk && save_now)
    {
      mas.count_dendrites(vtk_idx);
      mas.save_VTK(vtk_idx);
      mas.save_VTK_solid(vtk_idx);
    }

    keep_going = keep_going && (iteration < max_total_iterations) && (total_growth < termination_length);

    mas.update_grid();
    mas.compute_dt();

    iteration++;

    if (save_now) vtk_idx++;

    p4est = mas.get_p4est();
    nodes = mas.get_nodes();
    phi = mas.get_phi();
  }

  w1.stop(); w1.read_duration();

  return 0;
}
