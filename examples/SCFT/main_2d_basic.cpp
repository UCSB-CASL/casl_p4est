
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
#include <src/mls_integration/vtk/simplex3_mls_l_vtk.h>
#include <src/my_p8est_scft.h>
#include <src/my_p8est_shapes.h>
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
#include <src/mls_integration/vtk/simplex2_mls_l_vtk.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_shapes.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX
using namespace std;

// comptational domain
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

DEFINE_PARAMETER(pl, double, xmin, -4, "Box xmin");
DEFINE_PARAMETER(pl, double, ymin, -4, "Box ymin");
DEFINE_PARAMETER(pl, double, zmin, -4, "Box zmin");

DEFINE_PARAMETER(pl, double, xmax, 4, "Box xmax");
DEFINE_PARAMETER(pl, double, ymax, 4, "Box ymax");
DEFINE_PARAMETER(pl, double, zmax, 4, "Box zmax");

//-------------------------------------
// refinement parameters
//-------------------------------------
#ifdef P4_TO_P8
DEFINE_PARAMETER(pl, int, lmin, 5, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 5, "Max level of the tree");
#else
DEFINE_PARAMETER(pl, int, lmin, 7, "Min level of the tree");
DEFINE_PARAMETER(pl, int, lmax, 7, "Max level of the tree");
#endif

DEFINE_PARAMETER(pl, double, lip, 1.75, "");

DEFINE_PARAMETER(pl, double, steps_back, 3, "");

//-------------------------------------
// output parameters
//-------------------------------------
DEFINE_PARAMETER(pl, bool, save_vtk, 1, "");
DEFINE_PARAMETER(pl, int,  save_every_dn, 10, "");

//-------------------------------------
// polymer parameters
//-------------------------------------
DEFINE_PARAMETER(pl, double, XN, 20, "Interaction strength between A and B blocks");
DEFINE_PARAMETER(pl, double, f, .5,  "Fraction of A block in the polymer chain");
DEFINE_PARAMETER(pl, double, ns, 51, "Discretization of the polymer chain");

//-------------------------------------
// solver parameters
//-------------------------------------
DEFINE_PARAMETER(pl, int,    max_iterations, 1001, "");
DEFINE_PARAMETER(pl, double, tol, 1.0e-2, "");
DEFINE_PARAMETER(pl, bool,   smooth_pressure, true, "");

//-------------------------------------
// problem setup
//-------------------------------------
DEFINE_PARAMETER(pl, int, num_geometry, 0, "");
DEFINE_PARAMETER(pl, int, num_surface_tension, 1, "");
DEFINE_PARAMETER(pl, int, num_seed, 0, "");

int num_surfaces = 4;
std::vector<mls_opn_t> action(num_surfaces, MLS_INTERSECTION);

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

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.get_all(cmd);

  if (mpi.rank() == 0) pl.print_all();

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


  std::vector<CF_DIM *> phi_all_cf(4);
  std::vector<CF_DIM *> gamma_a_cf(4);
  std::vector<CF_DIM *> gamma_b_cf(4);

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

  mls_eff_cf_t phi_eff_cf(phi_all_cf, action);

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

  my_p4est_scft_t scft(ngbd, ns);

  /* create and initialize geometry */
  std::vector<Vec> phi(num_surfaces);

  my_p4est_level_set_t ls(ngbd);

  for (short i = 0; i < num_surfaces; i++)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi[i]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *phi_all_cf[i], phi[i]);
    ls.reinitialize_2nd_order(phi[i]);
  }

  for (int i = 0; i < num_surfaces; ++i)
  {
    scft.add_boundary(phi[i], action[i], *gamma_a_cf[i], *gamma_b_cf[i]);
  }

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
//  scft.set_geometry(phi, action);
  scft.set_polymer(f, XN);

  scft.initialize_solvers();
  scft.initialize_bc_simple();

  int iteration = 0;
  while (iteration < max_iterations)
  {
    parStopWatch time_iter;
    time_iter.start();

    scft.solve_for_propogators();
    scft.calculate_densities();
    scft.update_potentials();

    time_iter.stop();

    if (scft.get_exchange_force() < tol)
    {
      scft.initialize_bc_smart();
      if (smooth_pressure)
      {
        scft.smooth_singularity_in_pressure_field();
        smooth_pressure = false;
      }
      ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration no. %d; Energy: %e; Pressure: %e; Exchange: %e; Time: %e\n", iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), time_iter.get_duration()); CHKERRXX(ierr);
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
