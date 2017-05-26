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
#include <src/my_p8est_bialloy.h>
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
#include <src/my_p4est_multialloy_optimized.h>
//#include <src/my_p4est_multialloy_var2.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

int lmin = 6;
int lmax = 11;
int save_every_n_iteration = 50;

double bc_tolerance = 1.e-8;

double cfl_number = 0.1;
double phi_thresh = 0.01;
double zero_negative_velocity = true;
int max_iterations = 50;
int pin_every_n_steps = 3;


double lip = 1.5;

using namespace std;

bool save_velocity = true;
bool save_vtk = true;

#ifdef P4_TO_P8
char direction = 'z';
#else
char direction = 'y';
#endif

double termination_length = 0.7;

/* 0 - NiCu
 * 1 - AlCu
 */
int alloy_type = 0;

double box_size = 4e-2;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
//double box_size = 5e-1;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1/box_size;

double xmin = 0;
double ymin = 0;
double xmax = 1;
double ymax = 1;
#ifdef P4_TO_P8
double zmin = 0;
double zmax = 1;
int n_xyz[] = {1, 1, 1};
#else
int n_xyz[] = {1, 1};
#endif

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
    /* those are the default parameters for Ni-0.25831at%Cu-0.15at%Cu = Ni-0.40831at%Cu */
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
//    eps_c                = 0.0;
//    eps_v                = 0.0;
//    eps_anisotropy       = 0.01;

//    box_size = 4e-2;

    ml0                   =-357;            /* K / at frac. - liquidous slope */
    kp0                   = 0.86;           /* partition coefficient */
    c00                   = 0.2;            /* at frac.    */
    Dl0                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

    ml1                   =-357;            /* K / at frac. - liquidous slope */
    kp1                   = 0.86;           /* partition coefficient */
    c01                   = 0.2;            /* at frac.    */
    Dl1                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

    break;
    case 1:
      /* those are the default parameters for Ni-0.25831at%Cu-0.15at%Cu = Ni-0.40831at%Cu */
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
  //    eps_c                = 0.0;
  //    eps_v                = 0.0;
      eps_anisotropy       = 0.05;

  //    box_size = 4e-2;

      ml0                   =-357;            /* K / at frac. - liquidous slope */
      kp0                   = 0.86;           /* partition coefficient */
      c00                   = 0.3;            /* at frac.    */
      Dl0                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      ml1                   =-357;            /* K / at frac. - liquidous slope */
      kp1                   = 0.86;           /* partition coefficient */
      c01                   = 0.1;            /* at frac.    */
      Dl1                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      break;
    case 2:
      /* those are the default parameters for Ni-0.25831at%Cu-0.15at%Cu = Ni-0.40831at%Cu */
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
  //    eps_c                = 0.0;
  //    eps_v                = 0.0;
      eps_anisotropy       = 0.05;

  //    box_size = 4e-2;

      ml0                   =-357;            /* K / at frac. - liquidous slope */
      kp0                   = 0.86;           /* partition coefficient */
      c00                   = 0.1;            /* at frac.    */
      Dl0                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      ml1                   =-357;            /* K / at frac. - liquidous slope */
      kp1                   = 0.86;           /* partition coefficient */
      c01                   = 0.3;            /* at frac.    */
      Dl1                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */

      break;
//  case 1:
//    /* experimental Ni-Al-Ta parameters */
//    rho            = 7.365e-3;  /* kg.cm-3    */
//    heat_capacity  = 660;       /* J.kg-1.K-1 */
//    ml             =-255;       /* K / wt frac. - liquidous slope */
//    kp             = 0.48;      /* partition coefficient */
//    c0             = 0.152;     /* wt frac.    */
//    Tm             = 1754;      /* K           */
//    Dl             = 5e-5;      /* cm2.s-1 - concentration diffusion coefficient       */
//    Ds             = 1e-13;     /* cm2.s-1 - solid concentration diffusion coefficient */
//    G              = 20;        /* K.cm-1      */
//    V              = 0.01;      /* cm.s-1      */
//    latent_heat    = 2136;      /* J.cm-3      */
//    thermal_conductivity =  0.8; /* W.cm-1.K-1  */
//    lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
//    eps_c          = 2.7207e-5;
//    eps_v          = 2.27e-2;
//    eps_anisotropy = 0.05;

////    box_size = 5e-1;

//    Dl_sec = 5e-5;
//    Ds_sec = 1e-13;
//    ml_sec =-517;
//    c0_sec = 0.058;
//    kp_sec = 0.54;
//    break;
  }
}


#ifdef P4_TO_P8

struct plan_t : CF_3{
  double operator()(double x, double y, double z) const {
    if     (direction=='x') return x - 0.1;
    else if(direction=='y') return y - 0.1;
    else                    return z - 0.1;
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

class WallBCValueConcentrationS : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (wall_bc_type_concentration(x,y,z)==NEUMANN)
      return 0;
    else
      return kp * c0;
  }
} wall_bc_value_concentration_s;

class WallBCValueConcentrationL : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if (wall_bc_type_concentration(x,y,z)==NEUMANN)
      return 0;
    else
      return c0;
  }
} wall_bc_value_concentration_l;

class InitialTemperature : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(LS(x,y,z)<0) return LS(x,y,z)*(G+latent_heat*V/thermal_conductivity) + c0*ml + Tm;
    else            return LS(x,y,z)*G + c0*ml + Tm;
  }
} initial_temperature;

class InitialConcentrationS : public CF_3
{
public:
  double operator()(double , double, double ) const
  {
    return kp * c0;
  }
} initial_concentration_s;

class InitialConcentrationL : public CF_3
{
public:
  double operator()(double , double , double ) const
  {
    return c0;
  }
} initial_concentration_l;

#else

struct plan_t : CF_2{
  double operator()(double x, double y) const {
    if(direction=='x') return -(x - 0.1);
    else               return -(y - 0.1 + 0.000*cos(pow(2,lmax)*PI*(x-0.50007)));
  }
} LS;


class WallBCTypeTemperature : public WallBC2D
{
public:
  BoundaryConditionType operator()( double x, double y ) const
  {
    (void) x; (void) y;
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
    if(direction=='x')
    {
      if (ABS(x-xmin)<EPS || ABS(x-xmax)<EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
    else
    {
      if (ABS(y-ymin)<EPS || ABS(y-ymax)<EPS)
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

class InitialConcentrationL : public CF_2
{
public:
  double operator()(double , double ) const
  {
    return c01;
  }
} initial_concentration_1;

class InitialTemperature : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    if(LS(x,y)<0) return LS(x,y)*(G+latent_heat*V/thermal_conductivity) + c0*ml + Tm;
//    else          return LS(x,y)*G + c0*ml + Tm;
    if(LS(x,y)>0) return -LS(x,y)*(G+latent_heat*V/thermal_conductivity) + initial_concentration_0(x,y)*ml0 + initial_concentration_1(x,y)*ml1 + Tm;
    else          return -LS(x,y)*G                                      + initial_concentration_0(x,y)*ml0 + initial_concentration_1(x,y)*ml1 + Tm;
  }
} initial_temperature;

#endif



int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "number of blox in x-dimension");
  cmd.add_option("ny", "number of blox in y-dimension");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of blox in z-dimension");
#endif
  cmd.add_option("px", "periodicity in x-dimension 0/1");
  cmd.add_option("py", "periodicity in y-dimension 0/1");
#ifdef P4_TO_P8
  cmd.add_option("pz", "periodicity in z-dimension 0/1");
#endif
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_velo", "1 to save velocity of the interface, 0 otherwise");
  cmd.add_option("save_every_n", "save vtk every n iteration");
  cmd.add_option("write_stats", "write the statistics about the p4est");
  cmd.add_option("tf", "final time");
  cmd.add_option("L", "set latent heat");
  cmd.add_option("G", "set heat gradient");
  cmd.add_option("V", "set velocity");
  cmd.add_option("box_size", "set box_size");
  cmd.add_option("alloy", "choose the type of alloy. Default is 0.\n  0 - NiCuCu\n  1 - NiAlTa");
  cmd.add_option("direction", "direction of the crystal growth x/y");
  cmd.add_option("Dl0", "set the concentration diffusion coefficient in the liquid phase");
  cmd.add_option("Dl1", "set the concentration diffusion coefficient in the liquid phase");
  cmd.add_option("eps_c", "set the curvature undercooling coefficient");
  cmd.add_option("eps_v", "set the kinetic undercooling coefficient");

  cmd.add_option("bc_tolerance", "error tolerance for internal iterations");
  cmd.add_option("termination_length", "defines when a run will be stopped (fraction of box length, from 0 to 1)");
  cmd.add_option("lip", "set the lipschitz constant");
  cmd.add_option("cfl_number", "cfl_number");
  cmd.add_option("phi_thresh", "phi_thresh");
  cmd.add_option("zero_negative_velocity", "zero_negative_velocity");
  cmd.add_option("max_iterations", "max_iterations");
  cmd.add_option("pin_every_n_steps", "pin_every_n_steps");


  cmd.parse(argc, argv);

  alloy_type = cmd.get("alloy", alloy_type);
  set_alloy_parameters();

  save_vtk = cmd.get("save_vtk", save_vtk);
  save_velocity = cmd.get("save_velo", save_velocity);

  int periodic[P4EST_DIM];
  periodic[0] = cmd.get("px", (direction=='y' || direction=='z') ? 1 : 0);
  periodic[1] = cmd.get("py", (direction=='x' || direction=='z') ? 1 : 0);
#ifdef P4_TO_P8
  periodic[2] = cmd.get("pz", (direction=='x' || direction=='y') ? 1 : 0);
#endif

  n_xyz[0] = cmd.get("nx", n_xyz[0]);
  n_xyz[1] = cmd.get("ny", n_xyz[1]);
#ifdef P4_TO_P8
  n_xyz[2] = cmd.get("nz", n_xyz[2]);
#endif

#ifdef P4_TO_P8
  direction = cmd.get("direction", 'z');
#else
  direction = cmd.get("direction", 'y');
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

  save_every_n_iteration = cmd.get("save_every_n", save_every_n_iteration);
  latent_heat = cmd.get("L", latent_heat);
  G = cmd.get("G", G);
  V = cmd.get("V", V);
  box_size = cmd.get("box_size", box_size);
  Dl0 = cmd.get("Dl0", Dl0);
  Dl1 = cmd.get("Dl1", Dl1);
  eps_c = cmd.get("eps_c", eps_c);
  eps_v = cmd.get("eps_v", eps_v);

  termination_length = cmd.get("termination_length", termination_length);
  cfl_number = cmd.get("cfl_number", cfl_number);
  phi_thresh = cmd.get("phi_thresh", phi_thresh);
  zero_negative_velocity = cmd.get("zero_negative_velocity", zero_negative_velocity);
  max_iterations = cmd.get("max_iterations", max_iterations);
  pin_every_n_steps = cmd.get("pin_every_n_steps", pin_every_n_steps);
  bc_tolerance = cmd.get("bc_tolerance", bc_tolerance);


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

  /* create the p4est */
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
#else
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
#endif
  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  lip = cmd.get("lip", lip);

  splitting_criteria_cf_t data(lmin, lmax, &LS, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  /* initialize the variables */
  Vec phi, temperature, c0, c1, normal_velocity;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature    ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c0             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &c1             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &normal_velocity); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, LS, phi);
  sample_cf_on_nodes(p4est, nodes, initial_temperature, temperature);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_0, c0);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_1, c1);

  Vec tmp;
  ierr = VecGhostGetLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);
  ierr = VecSet(tmp, V); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);

  /* set initial time step */
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2., (double) data.max_lvl);
  double dy = (ymax-ymin) / pow(2., (double) data.max_lvl);
#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
#endif

#ifdef P4_TO_P8
  double dt = 0.45*MIN(dx,dy,dz)/V;
#else
  double dt = 0.45*MIN(dx,dy)/V;
#endif

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  srand(mpi.rank());

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    phi_p[n] += 0.1*dx*(double)(rand()%1000)/1000.;
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  /* perturb level set */
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  ls.perturb_level_set_function(phi, EPS);

  /* initialize the solver */
  my_p4est_multialloy_t bas(ngbd);

  bas.set_parameters(latent_heat, thermal_conductivity, lambda,
                     V, Tm, eps_anisotropy, eps_c, eps_v, scaling,
                     Dl0, kp0, c00, ml0,
                     Dl1, kp1, c01, ml1);
  bas.set_phi(phi);
  bas.set_bc(wall_bc_type_temperature,
             wall_bc_type_concentration,
             wall_bc_value_temperature,
             wall_bc_value_concentration_0,
             wall_bc_value_concentration_1);
  bas.set_temperature(temperature);
  bas.set_concentration(c0, c1);
  bas.set_normal_velocity(normal_velocity);
  bas.set_dt(dt);

//  bas.set_dt_method(dt_method);
  bas.set_bc_tolerance(bc_tolerance);
  bas.set_max_iterations(max_iterations);
  bas.set_pin_every_n_steps(pin_every_n_steps);
  bas.set_cfl(cfl_number);
  bas.set_phi_thresh(phi_thresh);
//  bas.set_zero_negative_velocity(zero_negative_velocity);
//  bas.set_num_of_iterations_per_step(num_of_iters_per_step);

//  bas.compute_velocity();
//  bas.compute_dt();

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
    ierr = PetscFPrintf(mpi.comm(), fich, "time average_interface_velocity max_interface_velocity interface_length solid_phase_area\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  bool keep_going = true;
  while(keep_going)
//  while (iteration < 20)
  {


    bas.one_step();

    tn += bas.get_dt();

    // save velocity, lenght of interface and area of solid phase in time
    if(save_velocity && iteration%save_every_n_iteration == 0)
    {
      p4est = bas.get_p4est();
      nodes = bas.get_nodes();
      phi = bas.get_phi();
      normal_velocity = bas.get_normal_velocity();

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
      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e\n", tn, avg_velo/scaling, bas.get_max_interface_velocity()/scaling, interface_length, solid_phase_area);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved velocity in %s\n", name); CHKERRXX(ierr);
    }

    // save field data
    if(save_vtk && iteration%save_every_n_iteration == 0)
    {
      bas.save_VTK(iteration/save_every_n_iteration);
    }
    ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);


    // check if the solid phase has reached the termination length
    if(save_velocity && iteration%save_every_n_iteration == 0)
    {
      int end_of_run = 0;
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
            if ((xyz[0] - xmin)/(xmax-xmin) > termination_length)
              end_of_run = 1;
          }
          else if (direction=='y')
          {
            if ((xyz[1] - ymin)/(ymax-ymin) > termination_length)
              end_of_run = 1;
          }
#ifdef P4_TO_P8
          else if (direction=='z')
          {
            if ((xyz[2] - zmin)/(zmax-zmin) > termination_length)
              end_of_run = 1;
          }
#endif
        }
      }
      ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &end_of_run, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      keep_going = (end_of_run == 0);
    }

    bas.update_grid();
    iteration++;
  }

  w1.stop(); w1.read_duration();

  return 0;
}
