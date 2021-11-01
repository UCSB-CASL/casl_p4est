
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
#include <random>

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
#include <src/my_p8est_scft.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_macros.h>
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
#include <src/my_p4est_scft.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX
using namespace std;
static PetscErrorCode ierr;

static parameter_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
static double xmin = -3; ADD_TO_LIST(pl, xmin, "Box xmin");
static double ymin = -3; ADD_TO_LIST(pl, ymin, "Box ymin");
static double zmin = -3; ADD_TO_LIST(pl, zmin, "Box zmin");

static double xmax = 3; ADD_TO_LIST(pl, xmax, "Box xmax");
static double ymax = 3; ADD_TO_LIST(pl, ymax, "Box ymax");
static double zmax = 3; ADD_TO_LIST(pl, zmax, "Box zmax");

static int nx = 1; ADD_TO_LIST(pl, nx, "Number of trees in the x-direction");
static int ny = 1; ADD_TO_LIST(pl, ny, "Number of trees in the y-direction");
static int nz = 1; ADD_TO_LIST(pl, nz, "Number of trees in the z-direction");

static bool px = 1; ADD_TO_LIST(pl, px, "Periodicity in the x-direction (0/1)");
static bool py = 1; ADD_TO_LIST(pl, py, "Periodicity in the y-direction (0/1)");
static bool pz = 1; ADD_TO_LIST(pl, pz, "Periodicity in the z-direction (0/1)");

//-------------------------------------
// refinement parameters
//-------------------------------------
static int lmin = 5; ADD_TO_LIST(pl, lmin, "Min level of the tree");
static int lmax = 7; ADD_TO_LIST(pl, lmax, "Max level of the tree");
static int lip  = 2; ADD_TO_LIST(pl, lip, "Refinement transition");

//-------------------------------------
// output parameters
//-------------------------------------
static bool save_vtk              = 1;  ADD_TO_LIST(pl, save_vtk, "Save vtk files (0/1)");
static int  save_vtk_freq         = 10; ADD_TO_LIST(pl, save_vtk_freq, "Frequency of saving vtk files");
static bool save_convergence      = 1;  ADD_TO_LIST(pl, save_convergence, "Save SCFT convergence into a file (0/1)");
static int  save_convergence_freq = 1;  ADD_TO_LIST(pl, save_convergence_freq, "Frequency of saving convergence");
static bool save_parameters       = 1;  ADD_TO_LIST(pl, save_parameters, "Save parameters into a file (0/1)");

//-------------------------------------
// polymer parameters
//-------------------------------------
static double XN =  20; ADD_TO_LIST(pl, XN, "Interaction strength between A and B blocks");
static double f  = .5;  ADD_TO_LIST(pl, f, "Fraction of A block in the polymer chain");
static double ns =  51; ADD_TO_LIST(pl, ns, "Discretization of the polymer chain");

//-------------------------------------
// solver parameters
//-------------------------------------
static double scft_tol           = 1.0e-10; ADD_TO_LIST(pl, scft_tol, "SCFT convergence criterion");
static int    max_iterations     = 1001;    ADD_TO_LIST(pl, max_iterations, "Total SCFT steps");
static bool   smooth_pressure    = 1;       ADD_TO_LIST(pl, smooth_pressure, "Restart pressure field after the first adjustment of BCs");
static bool   rerefine_at_start  = 1;       ADD_TO_LIST(pl, rerefine_at_start, "Re-refine gridin the beginning using a signed distance function");
static bool   refine_only_inside = 1;       ADD_TO_LIST(pl, refine_only_inside, "Do not refine outside (0/1)");
static double bc_tol             = 1.0e-2;  ADD_TO_LIST(pl, bc_tol, "Tolerance for adjusting boundary conditions");
static int    bc_iters_min       = 5;       ADD_TO_LIST(pl, bc_iters_min, "Minimum iterations before adjusting BCs");

//-------------------------------------
// problem setup
//-------------------------------------
static double box_size            = 1; ADD_TO_LIST(pl, box_size, "Scalling of computational box");
static int    geometry            = 2; ADD_TO_LIST(pl, geometry, "Problem geometry: \n"
                                                          "    0 - rectangular periodic box \n"
                                                          "    1 - rectangular box with neutral walls\n"
                                                          "    2 - circle/sphere\n"
                                                          "    3 - star-shaped domain\n"
                                                          "    4 - periodic film\n"
                                                          "    5 - droplet on a substrate\n"
                                                          "    6 - union of two spheres\n"
                                                          "    7 - difference of two spheres \n"
                                                          "    8 - rectangular periodic box with particles \n"
                                                          "    9 - same as case 8, but different method used");
static int    seed                = 1; ADD_TO_LIST(pl, seed, "Seed type: \n"
                                                      "    0 - random\n"
                                                      "    1 - horizontal stripes\n"
                                                      "    2 - vertical stripes");
static double surface_energy_avg  = 0; ADD_TO_LIST(pl, surface_energy_avg, "Average strength of surface energy");
static double surface_energy_diff = 4; ADD_TO_LIST(pl, surface_energy_diff, "Difference in surface energy strengths of A and B blocks");

//-------------------------------------
// seed
//-------------------------------------
class mu_m_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (seed)
    {
      case 0: return 0.1*XN*2.0*(double(rand())/double(RAND_MAX) - 0.5);
      case 1: return 0.5*XN*sin(y);
      case 2: return 0.5*XN*sin(x);
      default: throw std::invalid_argument("Error: Invalid seed number\n");
    }
  }
} static mu_m_cf;

//-------------------------------------
// geometry
//-------------------------------------
static int num_surfaces = 0;
static std::vector<mls_opn_t> action;
static std::vector<CF_DIM *>  phi_all_cf;
static std::vector<CF_DIM *>  gamma_a_cf;
static std::vector<CF_DIM *>  gamma_b_cf;

void set_geometry();

int main (int argc, char* argv[])
{
  /* initialize MPI */
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  srand(mpi.rank());

  /* parse command line arguments for parameters */
  cmdParser cmd;
  pl.initialize_parser(cmd);
//  cmd.show_help();
  cmd.parse(argc, argv);
  geometry = cmd.get("geometry", geometry);
  set_geometry();
  pl.get_all(cmd);
//  if (mpi.rank() == 0) pl.print_all();

  /* prepare output directories */
  const char* out_dir = getenv("OUT_DIR");
  if (mpi.rank() == 0 && (save_vtk || save_parameters || save_convergence))
  {
    if (!out_dir)
    {
      ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
      return -1;
    }
    std::ostringstream command;
    command << "mkdir -p " << out_dir;
    int ret_sys = system(command.str().c_str());
    if (ret_sys < 0) throw std::invalid_argument("Could not create a directory");
  }

  if (mpi.rank() == 0 && save_vtk)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys < 0) throw std::invalid_argument("Could not create a directory");
  }

  if (mpi.rank() == 0 && save_parameters) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  FILE *file_conv;
  char file_conv_name[10000];
  if (save_convergence)
  {
    sprintf(file_conv_name, "%s/convergence.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_conv_name, "w", &file_conv); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_conv, "iteration "
                                               "total_time "
                                               "iteration_time "
                                               "energy "
                                               "pressure_force "
                                               "exchange_force "
                                               "bc_adjusted\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
  }

  /* start a timer */
  parStopWatch w;
  w.start("total time");

  /* create the p4est */
  double xyz_min[]  = { DIM(xmin, ymin, zmin) };
  double xyz_max[]  = { DIM(xmax, ymax, zmax) };
  int    nb_trees[] = { DIM(nx, ny, nz) };
  int    periodic[] = { DIM(px, py, pz) };

  mls_eff_cf_t phi_eff_cf(phi_all_cf, action);

  my_p4est_brick_t      brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t              *p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);
  data.set_refine_only_inside(true);

  p4est->user_pointer = (void*)(&data);
  for (int i = 0; i < lmax; ++i)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  /* re-refine using a signed distance function */
  if (rerefine_at_start && num_surfaces > 0)
  {
    my_p4est_level_set_t ls(ngbd);

    Vec phi_eff;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_eff_cf, phi_eff);
    ls.reinitialize_1st_order(phi_eff,100);

    p4est_t       *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    Vec phi_eff_np1;
    ierr = VecDuplicate(phi_eff, &phi_eff_np1); CHKERRXX(ierr);
    VecCopyGhost(phi_eff, phi_eff_np1);

    bool is_grid_changing = true;
    while (is_grid_changing)
    {
      double *phi_eff_ptr;
      ierr = VecGetArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin, lmax, lip);
      sp.set_refine_only_inside(refine_only_inside);
      is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

      ierr = VecRestoreArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

      if (is_grid_changing)
      {
        // repartition p4est
        my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // interpolate data between grids
        my_p4est_interpolation_nodes_t interp(ngbd);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_np1)
        {
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          interp.add_point(n, xyz);
        }

        ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_eff_np1); CHKERRXX(ierr);

        interp.set_input(phi_eff, linear);
        interp.interpolate(phi_eff_np1);

      }
    }

    ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
    ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);

    // delete old p4est
    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);
  }

  /* create SCFT solver */
  my_p4est_scft_t scft(ngbd, ns); // scft is an OBJECT of class my_p4est_scft_t

  /* initialize geometry */
  std::vector<Vec> phi(num_surfaces);
  my_p4est_level_set_t ls(ngbd);
  for (short i = 0; i < num_surfaces; i++)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi[i]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *phi_all_cf[i], phi[i]);
    ls.reinitialize_2nd_order(phi[i]);
    scft.add_boundary(phi[i], action[i], *gamma_a_cf[i], *gamma_b_cf[i]);
  }

  /* set polymer parameters */
  scft.set_scalling(1./box_size);
  scft.set_polymer(f, XN);

  /* initialize chemical potentials */
  Vec mu_m = scft.get_mu_m();
  Vec mu_p = scft.get_mu_p();

  sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
  sample_cf_on_nodes(p4est, nodes, zero_cf, mu_p);

  /* initialize density fields */
  Vec rho_a = scft.get_rho_a();
  Vec rho_b = scft.get_rho_b();

  VecSetGhost(rho_a, f);
  VecSetGhost(rho_b, 1.-f);

  /* initialize diffusion solvers for propagators */
  scft.initialize_solvers();
  scft.initialize_bc_simple();

  /* main loop for solving SCFT equations */
  int iteration = 0;
  int bc_iters = 0;
  while (iteration < max_iterations && scft.get_exchange_force() > scft_tol || iteration == 0)
  {
    parStopWatch time_iter;
    time_iter.start();

    // do an SCFT step
    scft.solve_for_propogators();
    scft.calculate_densities();
    scft.update_potentials();

    // adjust boundary conditions to get rid of pressure singularity
    int bc_adjusted = 0;
    if (scft.get_exchange_force() < bc_tol && num_surfaces > 0 && bc_iters >= bc_iters_min)
    {
      scft.initialize_bc_smart();
      if (smooth_pressure)
      {
        scft.smooth_singularity_in_pressure_field();
        smooth_pressure = false;
      }
      ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
      bc_iters = 0;
      bc_adjusted = 1;
    }

    time_iter.stop();

    ierr = PetscPrintf(mpi.comm(), "Iteration no. %d; Energy: %e; Pressure: %e; Exchange: %e; Time: %e\n", iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), time_iter.get_duration()); CHKERRXX(ierr);

    // write data to disk
    if (iteration%save_vtk_freq == 0 && save_vtk) scft.save_VTK(iteration/save_vtk_freq);

    if (iteration%save_convergence_freq == 0 && save_convergence)
    {
      ierr = PetscFOpen(mpi.comm(), file_conv_name, "a", &file_conv); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e %e %e %d\n", iteration, w.get_duration_current(), time_iter.get_duration(), scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), bc_adjusted);
      ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
//      ierr = PetscPrintf(mpi.comm(), "Saved convergence in %s\n", file_conv_name); CHKERRXX(ierr);
    }


    iteration++;
    bc_iters++;
  }

  /* clean-up memory */
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

  /* show total running time */
  w.stop(); w.read_duration();

  return 0;
}

void set_geometry()
{
  switch (geometry)
  {
    case 0: // periodic rectangular box
      box_size = 2;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 1;
      num_surfaces = 0;
      break;
    case 1: // rectangular box with neutral walls
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;
      num_surfaces = 0;
      break;
    case 2: // sphere
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 1;

      static flower_phi_t circle(0.777, DIM(0,0,0), 0);
      static cf_const_t gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t gamma_b(surface_energy_avg - .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&circle);
      gamma_a_cf.push_back(&gamma_a);
      gamma_b_cf.push_back(&gamma_b);

      break;
    }
    case 3: // star-shaped
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 1;

      static flower_phi_t star(0.677, DIM(0,0,0), 0.15);
      static cf_const_t gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t gamma_b(surface_energy_avg - .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&star);
      gamma_a_cf.push_back(&gamma_a);
      gamma_b_cf.push_back(&gamma_b);

      break;
    }
    case 4: // periodic film
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = 1; py = pz = 0;

      num_surfaces = 2;

      class subsrate_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return -(y + 0.777);
        }
      } static subsrate;

      class air_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (y - 0.577) + 0.1*cos(3.*PI*x);
        }
      } static air;

      class substrate_gamma_a_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg + .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_a;

      class substrate_gamma_b_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg - .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_b;

      static cf_const_t air_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t air_gamma_b(surface_energy_avg - .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&subsrate);
      gamma_a_cf.push_back(&substrate_gamma_a);
      gamma_b_cf.push_back(&substrate_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&air);
      gamma_a_cf.push_back(&air_gamma_a);
      gamma_b_cf.push_back(&air_gamma_b);

      break;
    }
    case 5: // droplet on a substrate
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 2;

      class subsrate_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return -(y + 0.377);
        }
      } static subsrate;

      static flower_phi_t air(0.677, DIM(0,0,0), 0);

      static cf_const_t substrate_gamma_a(surface_energy_avg - .5*surface_energy_diff);
      static cf_const_t substrate_gamma_b(surface_energy_avg + .5*surface_energy_diff);

      static cf_const_t air_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t air_gamma_b(surface_energy_avg - .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&subsrate);
      gamma_a_cf.push_back(&substrate_gamma_a);
      gamma_b_cf.push_back(&substrate_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&air);
      gamma_a_cf.push_back(&air_gamma_a);
      gamma_b_cf.push_back(&air_gamma_b);

      break;
    }
    case 6: // two sphere union
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 2;

      static flower_phi_t circle_one(0.41, DIM(0.3,0.3,0.3), 0);
      static flower_phi_t circle_two(0.61, DIM(-.1,-.2,0.0), 0);
      static cf_const_t one_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t one_gamma_b(surface_energy_avg - .5*surface_energy_diff);
      static cf_const_t two_gamma_a(surface_energy_avg - .5*surface_energy_diff);
      static cf_const_t two_gamma_b(surface_energy_avg + .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&circle_one);
      gamma_a_cf.push_back(&one_gamma_a);
      gamma_b_cf.push_back(&one_gamma_b);

      action.push_back(MLS_ADDITION);
      phi_all_cf.push_back(&circle_two);
      gamma_a_cf.push_back(&two_gamma_a);
      gamma_b_cf.push_back(&two_gamma_b);

      break;
    }
    case 7: // two sphere difference
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 2;

      static flower_phi_t circle_one(0.71, DIM(0.1,0.1,0.1), 0);
      static flower_phi_t circle_two(0.51, DIM(-.4,-.2,0.0), 0, -1);
      static cf_const_t one_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t one_gamma_b(surface_energy_avg - .5*surface_energy_diff);
      static cf_const_t two_gamma_a(surface_energy_avg - .5*surface_energy_diff);
      static cf_const_t two_gamma_b(surface_energy_avg + .5*surface_energy_diff);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&circle_one);
      gamma_a_cf.push_back(&one_gamma_a);
      gamma_b_cf.push_back(&one_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&circle_two);
      gamma_a_cf.push_back(&two_gamma_a);
      gamma_b_cf.push_back(&two_gamma_b);

      break;
    }

    case 8: // periodic rectangular box
    {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 9;

      // substrate = block copoymer melt
      class subsrate_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return -(y + 0.777);
        }
      } static subsrate;

      // add nanoparticles to BCP melt
      class particle_t : public CF_DIM {
      public:
        double x_rand;
        double y_rand;
        particle_t(double x, double y) {
        x_rand = x;
        y_rand = y;
        }
        double operator()(DIM(double x, double y, double z)) const
        {
          // need minus here -> see Notizblatt why
          return -(sqrt(SQR(x-(x_rand)) + SQR(y-(y_rand))) - .03);
        }
      };
      static particle_t particle(0,0);
      static particle_t particle2(0,0.3);
      static particle_t particle3(-0.5,-0.1);
      static particle_t particle4(0.8,0.1);
      static particle_t particle5(-0.75,0.3);
      static particle_t particle6(0.89,-0.4);
      static particle_t particle7(0.9,0.9);
      static particle_t particle8(-0.2,-0.85);
      static particle_t particle9(-0.89,0.9);


      class substrate_gamma_a_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg + .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_a;

      class substrate_gamma_b_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg - .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_b;

      static cf_const_t particle_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t particle_gamma_b(surface_energy_avg - .5*surface_energy_diff);

//      action.push_back(MLS_INTERSECTION);
//      phi_all_cf.push_back(&subsrate);
//      gamma_a_cf.push_back(&substrate_gamma_a);
//      gamma_b_cf.push_back(&substrate_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle2);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle3);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle4);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle5);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle6);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle7);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle8);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle9);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

      break;
    }


  case 9: // periodic rectangular box
  {
      box_size = 4;
      xmin = ymin = zmin = -1;
      xmax = ymax = zmax =  1;
      nx = ny = nz = 1;
      px = py = pz = 0;

      num_surfaces = 1;

      // substrate = block copoymer melt
      class subsrate_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return -(y + 0.777);
        }
      } static subsrate;

      // add nanoparticles to BCP melt
      class particle_t : public CF_DIM {
      public:

        double operator()(DIM(double x, double y, double z)) const
        {
         double phi_p1 = -(sqrt(SQR(x-(0.0)) + SQR(y-(0.0))) - .03);
         double phi_p2 = -(sqrt(SQR(x-(0.5)) + SQR(y-(0.5))) - .03);
         double phi_p3 = -(sqrt(SQR(x-(-0.5)) + SQR(y-(-0.5))) - .03);
         double phi_p4 = -(sqrt(SQR(x-(-0.75)) + SQR(y-(0.3))) - .03);
         double phi_p5 = -(sqrt(SQR(x-(-0.2)) + SQR(y-(-0.85))) - .03);
         double phi_p6 = -(sqrt(SQR(x-(-0.89)) + SQR(y-(0.9))) - .03);

         double current_max;
         std::vector<double> all_phi;

         all_phi.push_back(phi_p1);
         all_phi.push_back(phi_p2);
         all_phi.push_back(phi_p3);
         all_phi.push_back(phi_p4);
         all_phi.push_back(phi_p5);
         all_phi.push_back(phi_p6);

         for(int i = 0; i<all_phi.size(); i++){
           if (i==0){
             current_max = all_phi[0];
           }
           else current_max = max(current_max, all_phi[i]);
         }

         return current_max;
        }
      } static particle;

      class substrate_gamma_a_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg + .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_a;

      class substrate_gamma_b_t : public CF_DIM {
      public:
        double operator()(DIM(double x, double y, double z)) const {
          return (surface_energy_avg - .5*surface_energy_diff*cos(3.*PI*x));
        }
      } static substrate_gamma_b;

      static cf_const_t particle_gamma_a(surface_energy_avg + .5*surface_energy_diff);
      static cf_const_t particle_gamma_b(surface_energy_avg - .5*surface_energy_diff);

//      action.push_back(MLS_INTERSECTION);
//      phi_all_cf.push_back(&subsrate);
//      gamma_a_cf.push_back(&substrate_gamma_a);
//      gamma_b_cf.push_back(&substrate_gamma_b);

      action.push_back(MLS_INTERSECTION);
      phi_all_cf.push_back(&particle);
      gamma_a_cf.push_back(&particle_gamma_a);
      gamma_b_cf.push_back(&particle_gamma_b);

    break;
  }
  }
}
