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

static bool px = 0; ADD_TO_LIST(pl, px, "Periodicity in the x-direction (0/1)");
static bool py = 0; ADD_TO_LIST(pl, py, "Periodicity in the y-direction (0/1)");
static bool pz = 0; ADD_TO_LIST(pl, pz, "Periodicity in the z-direction (0/1)");

//-------------------------------------
// refinement parameters
//-------------------------------------
static int lmin = 6; ADD_TO_LIST(pl, lmin, "Min level of the tree");
static int lmax = 8; ADD_TO_LIST(pl, lmax, "Max level of the tree");
static int lip  = 2; ADD_TO_LIST(pl, lip, "Refinement transition");

//-------------------------------------
// output parameters
//-------------------------------------
static bool save_energy      = 1;  ADD_TO_LIST(pl, save_energy, "Save effective energy into a file");

//-------------------------------------
// polymer parameters
//-------------------------------------
static double XN      = 20;  ADD_TO_LIST(pl, XN, "Interaction strength between A and B blocks");
static double gamma_a = 1.5; ADD_TO_LIST(pl, gamma_a, "Surface tension of A block");
static double gamma_b = 0.5; ADD_TO_LIST(pl, gamma_b, "Surface tension of B block");

//-------------------------------------
// number of particles
//-------------------------------------
int np = 5;


// constructs the 'analytical field'
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
    double current_max = 0;

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


class get_particle_number : public CF_DIM
{
private:
  vector<double> xc;
  vector<double> yc;
  vector<double> R;

public:
  get_particle_number(vector<double> xc_, vector<double> yc_, vector<double> R_){
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
    int particle_num;

    for(int j = 0; j < np; ++j){
      phi_particles[j] =  -(sqrt(SQR(x-(xc[j])) + SQR(y-(yc[j]))) - R[j]);
    }

    for(int k = 0; k < np; k++){
      if (k == 0){
        particle_num = 0;
      }
      else {
        if(phi_particles[particle_num] < phi_particles[k]){
          particle_num = k;
        }
      }
    }

    return particle_num;
  }
};


class get_mu_minus : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return 0.5*XN*sin(x+y);
  }
};


class get_gamma_effective : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return (0.5*(gamma_a+gamma_b) + (gamma_b-gamma_a)*(0.5*XN*sin(x+y))/XN);
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
  pl.get_all(cmd);


  // stopwatch
  parStopWatch w;
  w.start("Running example: scft_particles");


  // ---------------------------------------------------------
  // create the p4est for the initial position of the particles
  // ---------------------------------------------------------
  double xyz_min[]  = { DIM(xmin, ymin, zmin) };
  double xyz_max[]  = { DIM(xmax, ymax, zmax) };
  int    nb_trees[] = { DIM(nx, ny, nz) };
  int    periodic[] = { DIM(px, py, pz) };

  vector<double> xc(np);
  vector<double> yc(np);
  vector<double> radius(np, 0.15); // radius of particles

  // create the initial position of particles
  for (int i=0; i<np; ++i)
  {
    xc[i] = ((double)rand()/RAND_MAX)*(xmax - xmin) + xmin;
    yc[i] = ((double)rand()/RAND_MAX)*(ymax - ymin) + ymin;
  }

  // create particle_level_set for initial position
  construct_level_set particles(xc, yc, radius); //returns a double (=the max of n level-set functions for n particles)

  // create get_particle_num for or initial position (creates a field, that shows number of closet particle)
  get_particle_number particlenumber(xc, yc, radius);

  // create the mu_minus field for initial position
  get_mu_minus mu_minus;

  // create the gamma_effective field for initial position
  get_gamma_effective gamma_effective;

  my_p4est_brick_t      brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t              *p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &particles, lip);
  data.set_refine_only_inside(true);
  p4est->user_pointer = (void*)(&data);
  for (int i = 0; i < lmax; ++i)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  Vec particles_level_set; //'analytical field' (= level set function of every particle at time steps t_0 to t_max)
  Vec particle_number;
  Vec mu_minus_field;
  Vec gamma_effective_field;

  ierr = VecCreateGhostNodes(p4est, nodes, &particles_level_set); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, particles, particles_level_set);

  ierr = VecCreateGhostNodes(p4est, nodes, &particle_number); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, particlenumber, particle_number);

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_field); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, mu_minus, mu_minus_field);

  ierr = VecCreateGhostNodes(p4est, nodes, &gamma_effective_field); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, gamma_effective, gamma_effective_field);

  // calculate the energy (only for effective surface tension)
  double energy_eff = integrate_over_interface(p4est, nodes, particles_level_set, gamma_effective_field);


//  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
//  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
//  ngbd->init_neighbors();


  // ---------------------------------------------------------
  // write particle_level_set into file for initial position
  // ---------------------------------------------------------
  int step = 0;
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

  FILE *file_energy;
  char file_energy_name[1000];
  if (save_energy)
  {
    sprintf(file_energy_name, "%s/convergence.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_energy_name, "w", &file_energy); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_energy, "effective energy\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
  }

  double *particles_level_set_p;
  double *particle_number_p;
  double *mu_minus_p;
  double *gamma_effective_p;

  ierr = VecGetArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);
  ierr = VecGetArray(particle_number, &particle_number_p); CHKERRXX(ierr);
  ierr = VecGetArray(mu_minus_field, &mu_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(gamma_effective_field, &gamma_effective_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         4, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", particles_level_set_p,
                         VTK_POINT_DATA, "particle_number", particle_number_p,
                         VTK_POINT_DATA, "mu_minus", mu_minus_p,
                         VTK_POINT_DATA, "gamma_effective", gamma_effective_p);

  ierr = VecRestoreArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(particle_number, &particle_number_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mu_minus_field, &mu_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(gamma_effective_field, &gamma_effective_p); CHKERRXX(ierr);

  // destroy particle_level_set
  ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);
  ierr = VecDestroy(particle_number); CHKERRXX(ierr);
  ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
  ierr = VecDestroy(gamma_effective_field); CHKERRXX(ierr);


  // ---------------------------------------------------------
  // add velocity of particles
  // ---------------------------------------------------------

  double t = 0;
  double t_max = 1;
  double delta_t = 0.0;
  double energy_eff_t;

  while (t < t_max)
  {
    // ---------------------------------------------------------
    // calculate velocity
    // ---------------------------------------------------------

    // diagonal particle movement
//    vector<double> vx (np, 0.5);
//    vector<double> vy (np, 0.5);

    // random particle velocity
    vector<double> vx(np);
    vector<double> vy(np);
    for (int j = 0; j < np; ++j)
    {
      vx[j] = 2.*((double)rand()/RAND_MAX)-1.;
      vy[j] = 2.*((double)rand()/RAND_MAX)-1.;
    }

    // find the maximum velocity
    double vx_max;
    double vy_max;
    double v_max;

    for(int i = 0; i < np; ++i){
      if(i == 0){
        vx_max = vx[0];
        vy_max = vy[0];
      }
      else{
        if(vx_max < vx[i]){
          vx_max = vx[i];
        }
        if(vy_max < vy[i]){
          vy_max = vy[i];
        }
      }
    }
    v_max = max(vx_max, vy_max);

    // move particles and create new levelset
    for (int j = 0; j < np; ++j)
    {
      xc[j] = xc[j] - vx[j]*delta_t;
      yc[j] = yc[j] - vy[j]*delta_t;
    }

    // create particle_level_set and sample
    construct_level_set particles_t(xc, yc, radius);

    // create get_particle_num (creates a field, that shows number of closet particle)
    get_particle_number particlenumber_t(xc, yc, radius);

    // create the mu_minus field
    get_mu_minus mu_minus_t;

    // create the gamma_effective field
    get_gamma_effective gamma_effective_t;

    my_p4est_brick_t      brick;
    p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
    p4est_t              *p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin, lmax, &particles_t, lip);
    data.set_refine_only_inside(true);
    p4est->user_pointer = (void*)(&data);

    // set P4EST_TRUE, because interface is moving (need to refine and partition the grid)
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

    ierr = VecCreateGhostNodes(p4est, nodes, &particles_level_set); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, particles_t, particles_level_set);

    ierr = VecCreateGhostNodes(p4est, nodes, &particle_number); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, particlenumber_t, particle_number);

    ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_field); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_minus_t, mu_minus_field);

    ierr = VecCreateGhostNodes(p4est, nodes, &gamma_effective_field); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, gamma_effective_t, gamma_effective_field);

    // calculate the energy (only for effective surface tension)
    energy_eff_t = integrate_over_interface(p4est, nodes, particles_level_set, gamma_effective_field);


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
        << "_" << step+1;

    double *particles_level_set_p;
    double *particle_number_p;
    double *mu_minus_p;
    double *gamma_effective_p;

    ierr = VecGetArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);
    ierr = VecGetArray(particle_number, &particle_number_p); CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus_field, &mu_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(gamma_effective_field, &gamma_effective_p); CHKERRXX(ierr);


    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           4, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", particles_level_set_p,
                           VTK_POINT_DATA, "particle_number", particle_number_p,
                           VTK_POINT_DATA, "mu_minus", mu_minus_p,
                           VTK_POINT_DATA, "gamma_effective", gamma_effective_p);

    ierr = VecRestoreArray(particles_level_set, &particles_level_set_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(particle_number, &particle_number_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus_field, &mu_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_effective_field, &gamma_effective_p); CHKERRXX(ierr);

    // destroy particle_level_set
    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);
    ierr = VecDestroy(particle_number); CHKERRXX(ierr);
    ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
    ierr = VecDestroy(gamma_effective_field); CHKERRXX(ierr);

    // write data to disk
    if (save_energy)
    {
      ierr = PetscFOpen(mpi.comm(), file_energy_name, "a", &file_energy); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_energy, "%d %f\n", step, energy_eff_t);
      ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
    }

    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);


    ierr = PetscPrintf(mpi.comm(), "The maximum velocity is: v_max=%g\n", v_max); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "The smallest grid diagonal is: diag_min=%g\n", diag_min); CHKERRXX(ierr);


    double CFL = 0.5;
    delta_t = diag_min*CFL/(v_max);
    t+=delta_t;

    step++;
    ierr = PetscPrintf(mpi.comm(), "iteration number=%d\n", step); CHKERRXX(ierr);

    // CRITERION FOR SAFETY
    if(step == 100)
    {
      t = 1.1;
    }

    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);
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
//  delete ngbd;
//  delete hierarchy;
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


