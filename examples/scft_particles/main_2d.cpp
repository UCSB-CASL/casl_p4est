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

//----------------------------------------------------------------------------
// number of particles
//----------------------------------------------------------------------------
int np = 1;


// this class constructs the 'analytical field' (level-set function for every (x,y) coordinate)
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


// this class constructs the 'particle_number' field (indicates for every (x,y) coordinate, which particle is the closest)
// it is needed for the energy minimization of multiple particles
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


// this class constructs the exchange potential (mu_minus) for every (x,y), that describes the interaction between A&B (A, B are the monomer species)
class get_mu_minus : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return 0.5*XN*sin(x+y);
  }
};


// this class constructs the effective surface tension for every (x,y)
// it is needed to calculate the energy which is to be minimized
class get_gamma_effective : public CF_DIM {
private:
    get_mu_minus* mu_minus;

public:
    get_gamma_effective(get_mu_minus* mu_minus_){
      mu_minus = mu_minus_;
    }

  double operator()(DIM(double x, double y, double z)) const
  {
    return (0.5*(gamma_a+gamma_b) + (gamma_b-gamma_a)*(*mu_minus)(DIM(x, y, z))/XN);
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
    // random initial positions of particles
   // xc[i] = ((double)rand()/RAND_MAX)*(xmax - xmin) + xmin;
   // yc[i] = ((double)rand()/RAND_MAX)*(ymax - ymin) + ymin;

    // chosen initial positions of particles
    xc[i] = 2.0;
    yc[i] = 2.0;
  }

  // vectors used to later sample on nodes
  Vec particles_level_set; //'analytical field' (= level set function of every particle at time steps t_0 to t_max)
  Vec particle_number;
  Vec mu_minus_field;
  Vec gamma_effective_field;


  // define the output directory
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
  else
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir;
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
  }

  // make a separate file with the energies in every iteration (for Matlab plot)
  FILE *file_energy;
  char file_energy_name[1000];
  if (save_energy)
  {
    sprintf(file_energy_name, "%s/convergence.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_energy_name, "w", &file_energy); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_energy, "effective energy\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
  }


  // ---------------------------------------------------------
  // add velocity of particles
  // ---------------------------------------------------------
  int step = 0;
  double t = 0;
  double t_max = 200.0;
  double delta_t = 0.0;
  double energy_eff_t;

  while (t < t_max)
  {
    // create particle_level_set and sample
    construct_level_set particles_t(xc, yc, radius);

    // create get_particle_num (creates a field, that shows number of closet particle)
    get_particle_number particlenumber_t(xc, yc, radius);

    // create the mu_minus field
    get_mu_minus mu_minus_t;

    // create the gamma_effective field
    get_gamma_effective gamma_effective_t(&mu_minus_t);

    // create the p4est and splitting criteria
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

    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
    my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
    ngbd->init_neighbors();

    // sample the continuous functions (defined for every (x,y)) onto the nodes
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


    // ---------------------------------------------------------
    // calculate velocity using energy minimization
    // --------------------------------------------------------
    // compute normals and curvature (needed for dE/dt formula)
    Vec normal[P4EST_DIM];
    Vec kappa;
    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);
    compute_normals_and_mean_curvature(*ngbd, particles_level_set, normal, kappa);

    // compute first_derivaties of gamma (del_gamma/del_x and del_gamma/del_y)
    Vec gamma_d[P4EST_DIM];
    Vec G_2;
    ierr = VecDuplicate(kappa, &G_2); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &gamma_d[dim]); CHKERRXX(ierr); }
    ngbd->first_derivatives_central(gamma_effective_field, gamma_d);

    // compute del_gamma/del_n ((del_gamma/del_x)*n_x + (del_gamma/del_y)*n_y)
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(G_2, gamma_d[dim], normal[dim]);
    }

    // calculate G
    Vec G; // G = kappa*gamma_effecitve
    ierr = VecDuplicate(kappa, &G); CHKERRXX(ierr);
    VecPointwiseMultGhost(G, gamma_effective_field, kappa);
    VecAXPY(G,1,G_2); // G = G + 1*G_2

    // calculate g_x and g_y to get velocity
    Vec integrand_g[P4EST_DIM];
    double g;
   // vector<double> vel_x(np);
    double vel_x;
    double vel_y;
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &integrand_g[dim]); CHKERRXX(ierr); }
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(integrand_g[dim], G, normal[dim]);
      g = integrate_over_interface(p4est, nodes, particles_level_set, integrand_g[dim]);
      if(dim == 0){
        vel_x = -g;
      }
      else vel_y = -g;
      ierr = PetscPrintf(mpi.comm(), "g=%f\n", g); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "vel_x=%f\n", vel_x); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "vel_y=%f\n", vel_y); CHKERRXX(ierr);
    }

    // save as vtk files
    std::ostringstream oss;
    oss << out_dir
        << "/vtu/scft_"
        << p4est->mpisize
        << "_" << step;

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


    // write the energy in every iteration into the separate file
    if (save_energy)
    {
      ierr = PetscFOpen(mpi.comm(), file_energy_name, "a", &file_energy); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_energy, "%d %f\n", step, energy_eff_t);
      ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
    }

    // find the maximum velocity (needed to calculate the timestep delta_t)
    double vx_max;
    double vy_max;
    double v_max;

    for(int i = 0; i < np; ++i){
      if(i == 0){
        vx_max = vel_x;
        vy_max = vel_y;
      }
      else{
        if(vx_max < vel_x){
          vx_max = vel_x;
        }
        if(vy_max < vel_y){
          vy_max = vel_y;
        }
      }
    }
    v_max = max(vx_max, vy_max);


    // find the diagonal length of the smallest quadrant (needed to calculate the timestep delta_t)
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    // CFL comes from a stability criterion (needed to calculate the timestep delta_t)
    double CFL = 0.5;

    // calculate timestep delta_t, such that it's small enough to capture 'everything'
    delta_t = diag_min*CFL/abs(v_max);


    // move particles and create new levelset
    for (int j = 0; j < np; ++j)
    {
      xc[j] = xc[j] + vel_x*delta_t;
      yc[j] = yc[j] + vel_y*delta_t;
    }


    ierr = PetscPrintf(mpi.comm(), "The maximum velocity is: v_max=%g\n", v_max); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "The smallest grid diagonal is: diag_min=%g\n", diag_min); CHKERRXX(ierr);

    t+=delta_t;
    ierr = PetscPrintf(mpi.comm(), "t=%f\n", t); CHKERRXX(ierr);

    step++;
    ierr = PetscPrintf(mpi.comm(), "iteration number=%d\n", step); CHKERRXX(ierr);



    // CRITERION FOR SAFETY
    if(step == 250)
    {
      t = 1000;
    }


    // ---------------------------------------------------------
    // clean-up memory
    // ---------------------------------------------------------
    delete ngbd;
    delete hierarchy;
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_d[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(integrand_g[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    ierr = VecDestroy(G_2); CHKERRXX(ierr);
    ierr = VecDestroy(G); CHKERRXX(ierr);
  }

  ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);
  ierr = VecDestroy(particle_number); CHKERRXX(ierr);
  ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
  ierr = VecDestroy(gamma_effective_field); CHKERRXX(ierr);


  /* show total running time */
  w.stop(); w.read_duration();


  return 0;
  }
