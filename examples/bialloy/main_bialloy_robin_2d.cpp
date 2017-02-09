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
#include <src/my_p8est_bialloy_robin.h>
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
#include <src/my_p4est_bialloy_robin.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

int lmin = 6;
int lmax = 8;
int save_every_n_iteration = 1;

using namespace std;

bool save_velocity = true;
bool save_vtk = true;

#ifdef P4_TO_P8
char direction = 'z';
#else
char direction = 'y';
#endif

/* 0 - NiCu
 * 1 - AlCu
 */
int alloy_type = 0;

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

double box_size = 4e-2;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1/box_size;

double rho;                  /* density                                    - kg.cm-3      */
double heat_capacity;        /* c, heat capacity                           - J.kg-1.K-1   */
double ml;                   /* liquidus slope                             - K / at frac. */
double kp;                   /* partition coefficient                                     */
double c0;                   /* initial concentration                      - at frac.     */
double Tm;                   /* melting temperature                        - K            */
double Dl;                   /* liquid concentration diffusion coefficient - cm2.s-1      */
double G;                    /* thermal gradient                           - k.cm-1       */
double V;                    /* cooling velocity                           - cm.s-1       */
double latent_heat;          /* L, latent heat                             - J.cm-3       */
double thermal_conductivity; /* k, thermal conductivity                    - W.cm-1.K-1   */
double lambda;               /* thermal diffusivity                        - cm2.s-1      */

double eps_c;                /* curvature undercooling coefficient         - cm.K         */
double eps_v;                /* kinetic undercooling coefficient           - s.K.cm-1     */
double eps_anisotropy;       /* anisotropy coefficient                                    */

//double t_final = 1000*ny/V;
double t_final = 10000;

void set_alloy_parameters()
{
  switch(alloy_type)
  {
  case 0:
    /* those are the default parameters for NiCu */
    rho                  = 8.88e-3;        /* kg.cm-3    */
    heat_capacity        = 0.46e3;         /* J.kg-1.K-1 */
    ml                   =-357;            /* K / at frac. - liquidous slope */
//    ml                   =-1.;            /* K / at frac. - liquidous slope */
//    kp                   = 0.86;           /* partition coefficient */
    kp                   = 0.86;           /* partition coefficient */
    c0                   = 0.40831;        /* at frac.    */
    Tm                   = 1728;           /* K           */
    Dl                   = 1e-5;           /* cm2.s-1 - concentration diffusion coefficient       */
//    Ds                   = 1e-13;          /* cm2.s-1 - solid concentration diffusion coefficient */
    G                    = 4e2;            /* k.cm-1      */
    V                    = 0.01;           /* cm.s-1      */
    latent_heat          = 2350;           /* J.cm-3      */
    thermal_conductivity = 6.07e-1;        /* W.cm-1.K-1  */
    lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
//    lambda               = 1e-3; /* cm2.s-1  thermal diffusivity */
    eps_c                = 2.7207e-5;
    eps_v                = 2.27e-2;
    eps_anisotropy       = 0.05;
    break;
  case 1:
    /* experimental AlCu parameters */
    rho            = 2.8e-3;
    heat_capacity  = 1221.6;
    ml             = -652.9;
    kp             = 0.15;
    c0             = 0.6;
    Tm             = 933;
    Dl             = 1e-4;
//    Ds             = 1e-13;
    G              = 50;
    V              = 0.01; /* 1um = 10-4cm */
    latent_heat    = 898.8;
    lambda         = 0.84;
    thermal_conductivity =  lambda*rho*heat_capacity;
    eps_c          = 2.7207e-5;
    eps_v          = 2.27e-2;
    eps_anisotropy = 0.05;
    break;
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
    if(direction=='x') return x - 0.095703125001;
    else               return y - 0.095703125001;
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

class WallBCValueConcentrationL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (wall_bc_type_concentration(x,y)==NEUMANN)
      return 0;
    else
      return c0;
  }
} wall_bc_value_concentration_l;

class InitialTemperatureL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return c0*ml+Tm;
//    return LS(x,y)*(G+latent_heat*V/thermal_conductivity) + c0*ml + Tm;
    return LS(x,y)*G + c0*ml + Tm;
  }
} initial_temperature_l;

class InitialTemperatureS : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return c0*ml+Tm;
//    return LS(x,y)*G + c0*ml + Tm;
    return LS(x,y)*(G+latent_heat*V/thermal_conductivity) + c0*ml + Tm;
  }
} initial_temperature_s;

class InitialConcentrationL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return c0;
  }
} initial_concentration_l;

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
  cmd.add_option("alloy", "choose the type of alloy. Default is 0.\n  0 - NiCu\n  1 - AlCu");
  cmd.add_option("direction", "direction of the crystal growth x/y");
  cmd.add_option("Dl", "set the concentration diffusion coefficient in the liquid phase");
  cmd.add_option("eps_c", "set the curvature undercooling coefficient");
  cmd.add_option("eps_v", "set the kinetic undercooling coefficient");
  cmd.parse(argc, argv);

  alloy_type = cmd.get("alloy", alloy_type);
  set_alloy_parameters();

  save_vtk = cmd.get("save_vtk", save_vtk);
  save_velocity = cmd.get("save_velo", save_velocity);

  n_xyz[0] = cmd.get("nx", n_xyz[0]);
  n_xyz[1] = cmd.get("ny", n_xyz[1]);
#ifdef P4_TO_P8
  n_xyz[2] = cmd.get("nz", n_xyz[2]);
#endif

  int periodic[P4EST_DIM];
  periodic[0] = cmd.get("px", (direction=='y' || direction=='z') ? 1 : 0);
  periodic[1] = cmd.get("py", (direction=='x' || direction=='z') ? 1 : 0);
#ifdef P4_TO_P8
  periodic[2] = cmd.get("pz", (direction=='x' || direction=='y') ? 1 : 0);
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
  t_final = cmd.get("tf", t_final);
  latent_heat = cmd.get("L", latent_heat);
  G = cmd.get("G", G);
  V = cmd.get("V", V);
  box_size = cmd.get("box_size", box_size);
  Dl = cmd.get("Dl", Dl);
  eps_c = cmd.get("eps_c", eps_c);
  eps_v = cmd.get("eps_v", eps_v);

  double latent_heat_orig = latent_heat;
  double G_orig = G;
  double V_orig = V;

//  box_size = 1;
//  Dl = 1;
  scaling = 1/box_size;
  rho                  /= (scaling*scaling*scaling);
  thermal_conductivity /= scaling;
  Dl                   *= (scaling*scaling);
  G                    /= scaling;
  V                    *= scaling;
  latent_heat          /= (scaling*scaling*scaling);
  eps_c                *= scaling;
  eps_v                /= scaling;
  lambda                = thermal_conductivity/(rho*heat_capacity);

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

  splitting_criteria_cf_t data(lmin, lmax, &LS, 5.2);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  /* initialize the variables */
  Vec phi, temperature_l, temperature_s, cl, normal_velocity;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_l  ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &temperature_s  ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &cl             ); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &normal_velocity); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, LS, phi);
  sample_cf_on_nodes(p4est, nodes, initial_temperature_l, temperature_l);
  sample_cf_on_nodes(p4est, nodes, initial_temperature_s, temperature_s);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_l, cl);

  Vec tmp;
  ierr = VecGhostGetLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);
//  ierr = VecSet(tmp, V); CHKERRXX(ierr);
  ierr = VecSet(tmp, 1.412111e-02); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(normal_velocity, &tmp); CHKERRXX(ierr);

  /* perturb level set */
  my_p4est_level_set_t ls(ngbd);
  ls.perturb_level_set_function(phi, EPS);

  /* set initial time step */
  double dxyz[P4EST_DIM];
  dxyz_min(p4est, dxyz);

  /* initialize the solver */
  my_p4est_bialloy_t bas(ngbd);
  bas.set_parameters(latent_heat, thermal_conductivity, lambda,
                     Dl, V, kp, c0, ml, Tm, eps_anisotropy, eps_c, eps_v, scaling);
  bas.set_phi(phi);
  bas.set_bc(wall_bc_type_temperature,
             wall_bc_type_concentration,
             wall_bc_value_temperature,
             wall_bc_value_concentration_l);
  bas.set_temperature(temperature_l, temperature_s);
  bas.set_concentration(cl);
  bas.set_normal_velocity(normal_velocity);

//  bas.compute_normal_velocity_from_temperature();
//  bas.compute_velocity_from_temperature();
  bas.compute_dt();

//#ifdef P4_TO_P8
//  double dt = 0.45*MIN(dxyz[0],dxyz[1],dxyz[2])/V;
//#else
//  double dt = 0.0045*MIN(dxyz[0],dxyz[1])/V;
//#endif
//  bas.set_dt(dt);

  // loop over time
  double tn = 0;
  int iteration = 0;


  FILE *fich;
  char name[10000];

  char *out_dir;
  out_dir = getenv("OUT_DIR");
//  char out_dir[] = "/home/dbochkov/Outputs/bialloy_robi";
#ifdef P4_TO_P8
  sprintf(name, "%s/velo_%dx%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], n_xyz[2], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#else
  sprintf(name, "%s/velo_%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#endif

  if(save_velocity)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "%% time(s)      average interface velocity     max interface velocity\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  if(save_vtk && iteration%save_every_n_iteration == 0)
  {
    bas.save_VTK(iteration/save_every_n_iteration);
  }

  iteration++;

  while(tn<t_final)
  {
    ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);

    bas.one_step();

    tn += bas.get_dt();

    if(save_velocity)
    {
      p4est = bas.get_p4est();
      nodes = bas.get_nodes();
      phi = bas.get_phi();
      normal_velocity = bas.get_normal_velocity();

      if(p4est->mpirank==0)
      {
        p4est_locidx_t nb_nodes_global = 0;
        for(int r=0; r<p4est->mpisize; ++r)
          nb_nodes_global += nodes->global_owned_indeps[r];

        std::cout << "The p4est has " << nb_nodes_global << " nodes." << std::endl;
      }

      Vec ones;
      ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(ones, &tmp); CHKERRXX(ierr);
      ierr = VecSet(tmp, 1.); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(ones, &tmp); CHKERRXX(ierr);

      double avg_velo = integrate_over_interface(p4est, nodes, phi, normal_velocity) / integrate_over_interface(p4est, nodes, phi, ones);

      ierr = VecDestroy(ones); CHKERRXX(ierr);

      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), fich, "%e %e %e\n", tn, avg_velo/scaling, bas.get_max_interface_velocity()/scaling);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved velocity in %s\n", name); CHKERRXX(ierr);
    }


    if(save_vtk && iteration%save_every_n_iteration == 0)
    {
      bas.save_VTK(iteration/save_every_n_iteration);
    }

    iteration++;
//    if(iteration==264) break;
  }

  w1.stop(); w1.read_duration();

  return 0;
}
