
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
#include <src/my_p8est_poisson_nodes_mls_sc.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/my_p8est_scft.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_tools_mls.h>
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
#include <src/my_p4est_poisson_nodes_mls_sc.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_tools_mls.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

#define CMD_OPTIONS for (short option_action = 0; option_action < 2; ++option_action)
#define ADD_OPTION(cmd, var, description) option_action == 0 ? cmd.add_option(#var, description) : (void) (var = cmd.get(#var, var));
#define PARSE_OPTIONS(cmd, argc, argv) if (option_action == 0) cmd.parse(argc, argv);
using namespace std;

// comptational domain

double xmin = -4;
double ymin = -4;
double zmin = -4;

double xmax = 4;
double ymax = 4;
double zmax = 4;

bool px = 0;
bool py = 0;
bool pz = 0;

int nx = 1;
int ny = 1;
int nz = 1;

// grid parameters
#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
#else
int lmin = 5;
int lmax = 7;
#endif
double lip = 1.5;

int steps_back = 3;

// output parameters
bool save_vtk = 0;
int save_every_dn = 10;

// polymer parameters
double XN = 20.;
double f = 0.3;
int ns = 50+1;

int max_iterations = 20;
double tol = 1.0e-5;
bool smooth_pressure = true;

int num_geometry = 0;
int num_surface_tension = 1;
int num_seed = 0;

int num_surfaces = 4;
std::vector<action_t> action(num_surfaces, INTERSECTION);

/* geometry of interfaces */

// interface no. 0
class phi_00_cf_t : public CF_2
{
  struct
  {
//    double rad = 0.25;
//    double xc = 0.5;
//    double yc = 0.5;
//    double zc = 0.5;
    double rad = 3;
    double xc = 0;
    double yc = 0;
    double zc = 0;
  } data_00_;
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return sqrt( SQR(x-data_00_.xc) + SQR(y-data_00_.yc) ) - data_00_.rad;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_00_cf;

// interface no. 1
class phi_01_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return -1;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_01_cf;

// interface no. 2
class phi_02_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return -1;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_02_cf;

// interface no. 3
class phi_03_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return -1;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_03_cf;

class gamma_air_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_air_cf;

/* surface tensions for polymer A across each interface */

class gamma_a_00_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return .5;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_a_00;

class gamma_a_01_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_a_01;

class gamma_a_02_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_a_02;

class gamma_a_03_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_a_03;

/* surface tensions for polymer B across each interface */

class gamma_b_00_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return -.3;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_b_00;

class gamma_b_01_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_b_01;

class gamma_b_02_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_b_02;

class gamma_b_03_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0:
        switch (num_surface_tension)
        {
          case 0: return 0;
          case 1: return 0;
          default: throw std::invalid_argument("Error: Invalid surface tension number\n");
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_b_03;

/* seeds */

class mu_m_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_seed)
    {
      case 0: return 0.5*XN*sin(x+y);
      default: throw std::invalid_argument("Error: Invalid seed number\n");
    }
  }
} mu_m_cf;

class mu_p_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_seed)
    {
      case 0: return -0.2*XN*sin(x+y);
      default: throw std::invalid_argument("Error: Invalid seed number\n");
    }
  }
} mu_p_cf;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // create an output directory
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
    return -1;
  }

  std::ostringstream command;

  command << "mkdir -p " << out_dir << "/vtu";
  int ret_sys = system(command.str().c_str());
  if(ret_sys < 0)
    throw std::invalid_argument("my_p4est_bialloy_t::save_vtk could not create directory");


  cmdParser cmd;
  CMD_OPTIONS
  {
    ADD_OPTION(cmd, nx, "number of trees in x-dimension");
    ADD_OPTION(cmd, ny, "number of trees in y-dimension");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, nz, "number of trees in z-dimension");
#endif

    ADD_OPTION(cmd, px, "periodicity in x-dimension 0/1");
    ADD_OPTION(cmd, py, "periodicity in y-dimension 0/1");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, pz, "periodicity in z-dimension 0/1");
#endif

    ADD_OPTION(cmd, xmin, "xmin"); ADD_OPTION(cmd, xmax, "xmax");
    ADD_OPTION(cmd, ymin, "ymin"); ADD_OPTION(cmd, ymax, "ymax");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, zmin, "zmin"); ADD_OPTION(cmd, zmax, "zmax");
#endif

    ADD_OPTION(cmd, lmin, "min level of trees");
    ADD_OPTION(cmd, lmax, "max level of trees");
    ADD_OPTION(cmd, lip,  "Lipschitz constant");

    ADD_OPTION(cmd, steps_back,  "steps_back");

    ADD_OPTION(cmd, save_vtk,  "save_vtk");

    ADD_OPTION(cmd, XN, "Florry-Higgins parameter");
    ADD_OPTION(cmd, f,  "Fraction of polymer A");
    ADD_OPTION(cmd, ns, "Number of steps along the polymer chain");

    PARSE_OPTIONS(cmd, argc, argv);
  }

#ifdef P4_TO_P8
  double xyz_min[] = { xmin, ymin, zmin };
  double xyz_max[] = { xmax, ymax, zmax };
  int nb_trees[] = { nx, ny, nz };
  int periodic[] = { px, py, pz };
#else
  double xyz_min[] = { xmin, ymin };
  double xyz_max[] = { xmax, ymax };
  int nb_trees[] = { nx, ny };
  int periodic[] = { px, py };
#endif


  std::vector<CF_2 *> phi_all_cf(4);
  std::vector<CF_2 *> gamma_a_cf(4);
  std::vector<CF_2 *> gamma_b_cf(4);

  phi_all_cf[0] = &phi_00_cf;
  phi_all_cf[1] = &phi_01_cf;
  phi_all_cf[2] = &phi_02_cf;
  phi_all_cf[3] = &phi_03_cf;

  gamma_a_cf[0] = &gamma_a_00;
  gamma_a_cf[1] = &gamma_a_01;
  gamma_a_cf[2] = &gamma_a_02;
  gamma_a_cf[3] = &gamma_a_03;

  gamma_b_cf[0] = &gamma_b_00;
  gamma_b_cf[1] = &gamma_b_01;
  gamma_b_cf[2] = &gamma_b_02;
  gamma_b_cf[3] = &gamma_b_03;

  parStopWatch w;
  w.start("total time");

  level_set_tot_t phi_eff_cf(&phi_all_cf, &action, NULL);

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  my_p4est_scft_t scft(ngbd);

  /* create and initialize geometry */
  std::vector<Vec> phi(num_surfaces);

  my_p4est_level_set_t ls(ngbd);

  for (short i = 0; i < num_surfaces; i++)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi[i]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *phi_all_cf[i], phi[i]);
    ls.reinitialize_1st_order_time_2nd_order_space(phi[i]);
  }

  scft.set_geometry(phi, action);

//  for (short i = 0; i < num_surfaces; i++)
//  {
//    ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
//  }

  /* initialize potentials */
  Vec mu_m = scft.get_mu_m();
  Vec mu_p = scft.get_mu_p();

  sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
  sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

  /* initialize density fields */
  Vec rho_a = scft.get_rho_a();
  Vec rho_b = scft.get_rho_b();

  set_ghosted_vec(rho_a, f);
  set_ghosted_vec(rho_b, 1.-f);

  /* create and initialize surface tension fields */
  scft.set_geometry(phi, action);
  scft.set_polymer(f, XN, ns);
  scft.set_surface_tensions(gamma_a_cf, gamma_b_cf, gamma_air_cf);

  scft.initialize_linear_system();
  scft.initialize_bc_simple();

  int iteration = 0;
  while (iteration < max_iterations)
  {
    scft.solve_for_propogators();
    scft.calculate_densities();
    scft.update_potentials();

    if (scft.get_exchange_force() < tol)
    {
      scft.initialize_bc_smart();
      if (smooth_pressure)
      {
        scft.smooth_singularity_in_pressure_field();
        smooth_pressure = false;
      }
      scft.recompute_matrices();
//      scft.initialize_linear_system();
      ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e\n", scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);
    if (save_vtk && iteration%save_every_dn == 0)
    {
      scft.save_VTK(iteration/save_every_dn);
    }

    iteration++;
  }

  w.stop(); w.read_duration();

  for (short i = 0; i < num_surfaces; i++)
  {
    ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
  }

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
