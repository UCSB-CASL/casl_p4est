/*
 * Title: scft_particles
 * Description:
 * Author: ivanabagaric
 * Date Created: 10-24-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif
#include <src/my_p4est_level_set.h>
#include <src/Parser.h>
#include <src/casl_math.h>

#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>

#include <vector>

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
static int lmin = 7;  ADD_TO_LIST(pl, lmin, "Min level of the tree");
static int lmax = 10; ADD_TO_LIST(pl, lmax, "Max level of the tree");
static int lip  = 2;  ADD_TO_LIST(pl, lip, "Refinement transition");

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
static int    geometry            = 0; ADD_TO_LIST(pl, geometry, "Problem geometry: \n"
                                                          "    0 - rectangular periodic box with particles \n"
                                                          "    1 - same as case 8, but different method used");

// static double surface_energy_avg  = 0; ADD_TO_LIST(pl, surface_energy_avg, "Average strength of surface energy");
// static double surface_energy_diff = 0; ADD_TO_LIST(pl, surface_energy_diff, "Difference in surface energy strengths of A and B blocks");

//-------------------------------------
// geometry
//-------------------------------------
static int num_surfaces = 1;
static std::vector<mls_opn_t> action;
static std::vector<CF_DIM *>  phi_all_cf; //'numerical field' (= level set function at every grid point)


// number of particles
int np = 3;


void set_geometry();


class construct_level_set : public CF_DIM
{
private:
  vector<double> xc;
  vector<double> yc;
  vector<double> R;


public:
  construct_level_set(vector<double> xc_, vector<double> yc_, vector<double> R_){
    xc.resize(np); // to make sure that the vectors xc, yc and R are of size np
    yc.resize(np);
    R.resize(np);
    for(int i = 0; i < np; ++i){
      xc[i] = xc_[i];
      yc[i] = yc_[i];
      R[i] = R_[i];
    }
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    vector<double> phi_particles(np);
    double current_max;

    for(int j = 0; j < np; ++j){
      phi_particles[j] =  -(sqrt(SQR(x-(xc[j])) + SQR(y-(yc[j]))) - R[j]);
    }

    for(int k = 0; k < phi_particles.size(); k++){
      if (k==0){
        current_max = phi_particles[0];
      }
      else current_max = max(current_max, phi_particles[k]);
    }

    return current_max;
  }
};


int main(int argc, char** argv)
{
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  /* parse command line arguments for parameters */
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);


 // geometry = cmd.get("geometry", geometry);
 // set_geometry();
  pl.get_all(cmd);


  // stopwatch
  parStopWatch w;
  w.start("Running example: scft_particles");


  // ---------------------------------------------------------
  // create the p4est
  // ---------------------------------------------------------
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



  // ---------------------------------------------------------
  // add particles
  // ---------------------------------------------------------
  vector<double> xc_init(np);
  vector<double> yc_init(np);
  vector<double> xc(np);
  vector<double> yc(np);
  vector<double> radius(np, 0.3); // radius of particles

  Vec particles_level_set; //'analytical field' (= level set function of every particle at time steps t_0 to t_max)


  xc_init = {0, 1, -1.5};
  yc_init = {0, -1, -0.5};


//  for (int i=0; i<np; ++i)
//  {
//    srand (time(NULL));

//    xc_init[i] = ((double)rand()/RAND_MAX)*(xmax - xmin) + xmin;
//    yc_init[i] = ((double)rand()/RAND_MAX)*(ymax - ymin) + ymin;
//  }


  double t = 0;
  double t_max = 1;
  double delta_t = 0.1;
  int step = 0;
  while (t < t_max)
  {
    // initial position of particles
    if(t == 0)
    {
      for(int k = 0; k < np; ++k)
      {
        xc[k] = xc_init[k];
        yc[k] = yc_init[k];
      }
    }
    else
    {
      // calculate velocity
      vector<double> vx (np, 0.5);
      vector<double> vy (np, 0.5);

      // move particles and create new levelset
      for (int j = 0; j < np; ++j)
      {
        xc[j] = xc[j] + vx[j]*delta_t;
        yc[j] = yc[j] + vy[j]*delta_t;
      }
    }


    // create particle_level_set and sample
    construct_level_set particles(xc, yc, radius); //returns a double (=the max of n level-set functions for n particles)
//    phi_all_cf.push_back(&particles);

    ierr = VecCreateGhostNodes(p4est, nodes, &particles_level_set); CHKERRXX(ierr);
//    for (short i = 0; i < num_surfaces; i++)
//    {
    sample_cf_on_nodes(p4est, nodes, particles, particles_level_set);
//    }


    // write particle_level_set into file
    const char *out_dir = getenv("OUT_DIR");
    if (!out_dir) out_dir = ".";
    else
    {
      std::ostringstream command;
      command << "mkdir -p " << out_dir;
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
    }

    std::ostringstream oss;
    oss << out_dir
        << "/vtu/scft_"
        << p4est->mpisize
        << "_" << step;

    double *particles_level_set_p;

    ierr = VecGetArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", particles_level_set_p);

    ierr = VecRestoreArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);


    // destroy particle_level_set
    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);


    t+=delta_t;
    step++;
  }



  // ---------------------------------------------------------
  // create SCFT solver
  // ---------------------------------------------------------
//  my_p4est_scft_t scft(ngbd, ns); // scft is an OBJECT of class my_p4est_scft_t

  // ---------------------------------------------------------
  // add particles
  // ---------------------------------------------------------
  /* initialize geometry */
//  std::vector<Vec> phi(num_surfaces);
//  my_p4est_level_set_t ls(ngbd);
//  for (short i = 0; i < num_surfaces; i++)
//  {
//    ierr = VecCreateGhostNodes(p4est, nodes, &phi[i]); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est, nodes, *phi_all[i], phi[i]);
//    ls.reinitialize_2nd_order(phi[i]);
//    // scft.add_boundary(phi[i], action[i], *gamma_a_cf[i], *gamma_b_cf[i]);
//  }

  /* set polymer parameters */
  // scft.set_scalling(1./box_size);
  // scft.set_polymer(f, XN);

  /* initialize chemical potentials */
  //Vec mu_m = scft.get_mu_m();
  //Vec mu_p = scft.get_mu_p();

  //sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
  //sample_cf_on_nodes(p4est, nodes, zero_cf, mu_p);

  /* initialize density fields */
  //Vec rho_a = scft.get_rho_a();
  //Vec rho_b = scft.get_rho_b();

  //VecSetGhost(rho_a, f);
  //VecSetGhost(rho_b, 1.-f);

  /* initialize diffusion solvers for propagators */
//  scft.initialize_solvers();
//  scft.initialize_bc_simple();



  // ---------------------------------------------------------
  // main loop for solving SCFT equations
  // ---------------------------------------------------------
//  int iteration = 0;
//  int bc_iters = 0;
//  while (iteration < max_iterations && scft.get_exchange_force() > scft_tol || iteration == 0)
//  {
//    my_p4est_integration_mls_t integration(p4est, nodes);

//    parStopWatch time_iter;
//    time_iter.start();


//    // ---------------------------------------------------------
//    // velocity of particles
//    // ---------------------------------------------------------

//    /* velocity_G (describes how energy changes if interface is moved) */
//    Vec velocity_G; ierr = VecCreateGhostNodes(p4est, nodes, &velocity_G); CHKERRXX(ierr);
//    scft.compute_energy_shape_derivative(0, velocity_G);
//    VecScaleGhost(velocity_G, -1);

//    /* compute normals to the particle surface */
//    Vec normal[P4EST_DIM];
//    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
//    for (short i = 0; i < num_surfaces; i++)
//    {
//    compute_normals(*ngbd, phi[i], normal);
//    }

//    /* integrand_g (= velocity_G*normals) */
//    Vec integrand_g;
//    ierr = VecDuplicate(velocity_G, &integrand_g); CHKERRXX(ierr);
//    foreach_dimension(dim)
//    {
//    VecPointwiseMultGhost(integrand_g, velocity_G, normal[dim]);
//    }

//    /* g (g = integral(velocity_G*normal)dr ) */
//    double g = integration.integrate_over_interface(0, integrand_g);

//    /* directions tau */



//    // do an SCFT step
//    scft.solve_for_propogators();
//    scft.calculate_densities();
//    scft.update_potentials();

//    // adjust boundary conditions to get rid of pressure singularity
//    int bc_adjusted = 0;
//    if (scft.get_exchange_force() < bc_tol && num_surfaces > 0 && bc_iters >= bc_iters_min)
//    {
//      scft.initialize_bc_smart();
//      if (smooth_pressure)
//      {
//        scft.smooth_singularity_in_pressure_field();
//        smooth_pressure = false;
//      }
//      ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
//      bc_iters = 0;
//      bc_adjusted = 1;
//    }

//    time_iter.stop();

//    ierr = PetscPrintf(mpi.comm(), "Iteration no. %d; Energy: %e; Pressure: %e; Exchange: %e; Time: %e\n", iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), time_iter.get_duration()); CHKERRXX(ierr);

//    // write data to disk
//    if (iteration%save_vtk_freq == 0 && save_vtk) scft.save_VTK(iteration/save_vtk_freq);

//    if (iteration%save_convergence_freq == 0 && save_convergence)
//    {
//      ierr = PetscFOpen(mpi.comm(), file_conv_name, "a", &file_conv); CHKERRXX(ierr);
//      PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e %e %e %d\n", iteration, w.get_duration_current(), time_iter.get_duration(), scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), bc_adjusted);
//      ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
// //      ierr = PetscPrintf(mpi.comm(), "Saved convergence in %s\n", file_conv_name); CHKERRXX(ierr);
//    }


//    iteration++;
//    bc_iters++;


//    foreach_dimension(dim)
//    {
//      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
//    }

//    ierr = VecDestroy(velocity_G); CHKERRXX(ierr);
//    ierr = VecDestroy(integrand_g); CHKERRXX(ierr);

//  }



  // ---------------------------------------------------------
  // clean-up memory
  // ---------------------------------------------------------



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


// ---------------------------------------------------------
// geometry of particles and BCP (Ivana, Oct 2019)
// ---------------------------------------------------------


