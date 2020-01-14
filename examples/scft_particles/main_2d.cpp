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
static int lmin = 6; ADD_TO_LIST(pl, lmin , "Min level of the tree");
static int lmax = 8; ADD_TO_LIST(pl, lmax , "Max level of the tree");
static int lip  = 2; ADD_TO_LIST(pl, lip  , "Refinement transition");

//-------------------------------------
// output parameters
//-------------------------------------
static bool save_energy      = 1;  ADD_TO_LIST(pl, save_energy, "Save effective energy into a file");

//-------------------------------------
// scft parameters
//-------------------------------------
static bool   use_scft            = 1;     ADD_TO_LIST(pl, use_scft           , "Turn on/off SCFT 0/1");
static double scft_tol            = 4.e-3; ADD_TO_LIST(pl, scft_tol           , "Tolerance for SCFT");
static int    max_scft_iterations = 200;   ADD_TO_LIST(pl, max_scft_iterations, "Maximum SCFT iterations");
static int    bc_adjust_min       = 5;     ADD_TO_LIST(pl, bc_adjust_min      , "Minimun SCFT steps between adjusting BC");
static bool   smooth_pressure     = 1;     ADD_TO_LIST(pl, smooth_pressure    , "Smooth pressure after first BC adjustment 0/1");
static double scft_bc_tol         = 4.e-2; ADD_TO_LIST(pl, scft_bc_tol        , "Tolerance for adjusting BC");

//-------------------------------------
// polymer parameters
//-------------------------------------
static double XN       = 20;  ADD_TO_LIST(pl, XN        , "Interaction strength between A and B blocks");
static double gamma_a  = 0.5; ADD_TO_LIST(pl, gamma_a   , "Surface tension of A block");
static double gamma_b  = 0.0; ADD_TO_LIST(pl, gamma_b   , "Surface tension of B block");
static int    ns       = 40;  ADD_TO_LIST(pl, ns        , "Discretization of polymer chain");
static double box_size = 2;  ADD_TO_LIST(pl, box_size  , "Box size in units of Rg");
static double f        = .5;  ADD_TO_LIST(pl, f         , "Fraction of polymer A");

//-------------------------------------
// number of particles
//-------------------------------------
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


class gamma_Aa_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return gamma_a;
  }
} gamma_Aa;


class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return gamma_b;
  }
} gamma_Bb;


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
  vector<double> radius(np, 0.3); // radius of particles

  // create the initial position of particles
//  for (int i=0; i<np; ++i)
//  {
//    // random initial positions of particles
//   // xc[i] = ((double)rand()/RAND_MAX)*(xmax - xmin) + xmin;
//   // yc[i] = ((double)rand()/RAND_MAX)*(ymax - ymin) + ymin;

//    // chosen initial positions of particles
//    xc[i] = 2.0;
//    yc[i] = 2.0;
//  }

  // define particle position by hand
//  xc[0] = 0.75; //Test NP1
//  yc[0] = 0.5;

  xc[0] = -1.5;
  yc[0] = -1;


//  xc[1] = 1.5;
//  yc[1] = 1.5;

//  xc[2] = -2.0;
//  yc[2] = -1.25;

//  xc[3] = 0.5;
//  yc[3] = -1.25;

//  xc[4] = 2.25;
//  yc[4] = -2.25;

//  xc[5] = -2.25;
//  yc[5] = 2.25;

//  xc[6] = 2.25;
//  yc[6] = 2.25;

//  xc[7] = -1.5;
//  yc[7] = 0;

//  xc[8] = -2.25;
//  yc[8] = 0;

//  xc[9] = 0;
//  yc[9] = 2.0;

//  xc[10] = 0;
//  yc[10] = -2.25;


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
    ierr = PetscFPrintf(mpi.comm(), file_energy, "effective energy "
                                                 "expected dE/dt"
                                                 "expected dE"
                                                 "effective dE\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
  }


  // ---------------------------------------------------------
  // add velocity of particles
  // ---------------------------------------------------------
  int step = 0;
  double t = 0;
  double t_max = 200.0;
  double delta_t = 0.0;
  double energy_eff_t = 0;
  double energy_eff_previous = 0; //needed to calculate the energy difference between two iterations
  double energy = 0;
  double dE_dt_expected = 0;

  Vec mu_minus_field;

  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  p4est_connectivity_t *connectivity;
  p4est_t              *p4est;

  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;

  p4est_connectivity_t *connectivity_tmp;
  p4est_t              *p4est_tmp;

  p4est_ghost_t *ghost_tmp;
  p4est_nodes_t *nodes_tmp;

  my_p4est_hierarchy_t *hierarchy_tmp;
  my_p4est_node_neighbors_t *ngbd_tmp;

  while (t < t_max)
  {
    // create particle_level_set and sample
    construct_level_set particles_t(xc, yc, radius);

    // create get_particle_num (creates a field, that shows number of closet particle)
    get_particle_number particlenumber_t(xc, yc, radius);

    // create the mu_minus field
    get_mu_minus mu_minus_t;

//     create the gamma_effective field
//    get_gamma_effective gamma_effective_t(&mu_minus_t);

    if (step != 0)
    {
      connectivity_tmp = connectivity;
      p4est_tmp = p4est;

      ghost_tmp = ghost;
      nodes_tmp = nodes;

      hierarchy_tmp = hierarchy;
      ngbd_tmp = ngbd;
    }

    // create the p4est and splitting criteria
    my_p4est_brick_t      brick;
    connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
    p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin, lmax, &particles_t, lip);
    data.set_refine_only_inside(true);
    p4est->user_pointer = (void*)(&data);

    // set P4EST_TRUE, because interface is moving (need to refine and partition the grid)
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    nodes = my_p4est_nodes_new(p4est, ghost);

    hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
    ngbd      = new my_p4est_node_neighbors_t(hierarchy,nodes);
    ngbd->init_neighbors();

    if (step == 0 || !use_scft)
    {
      // initialize mu_minus field
      ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_field); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, mu_minus_t, mu_minus_field);
    }
    else
    {
      // by interpolating the mu_minus field, the calculation is accelerated AND it 'stays on the same track' (since there are multiple solutions in scft)
      my_p4est_interpolation_nodes_t interp(ngbd_tmp);

      interp.set_input(mu_minus_field, linear);

      Vec mu_minus_field_tmp; ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_field_tmp); CHKERRXX(ierr);

      double xyz[P4EST_DIM];

      foreach_node(n, nodes)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }

      interp.interpolate(mu_minus_field_tmp);

      ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
      mu_minus_field = mu_minus_field_tmp;
    }

    if (step != 0)
    {
      delete hierarchy_tmp;
      delete ngbd_tmp;
      p4est_nodes_destroy (nodes_tmp);
      p4est_ghost_destroy(ghost_tmp);
      p4est_destroy (p4est_tmp);
      p4est_connectivity_destroy(connectivity_tmp);
    }

    // vectors used to sample on nodes
    Vec particles_level_set; //'analytical field' (= level set function of every particle at time steps t_0 to t_max)
    Vec particle_number;
    Vec gamma_effective_field; ierr = VecCreateGhostNodes(p4est, nodes, &gamma_effective_field); CHKERRXX(ierr);

    // sample the continuous functions (defined for every (x,y)) onto the nodes
    ierr = VecCreateGhostNodes(p4est, nodes, &particles_level_set); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, particles_t, particles_level_set);

    ierr = VecCreateGhostNodes(p4est, nodes, &particle_number); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, particlenumber_t, particle_number);

    Vec velo;
    ierr = VecDuplicate(particles_level_set, &velo); CHKERRXX(ierr);

    energy_eff_previous = energy_eff_t;

    if(use_scft)
    {
      my_p4est_scft_t scft(ngbd, ns);

      // set geometry
      scft.add_boundary(particles_level_set, MLS_INTERSECTION, gamma_Aa, gamma_Bb);

      scft.set_scalling(1./box_size);
      scft.set_polymer(f, XN);

      // initialize potentials
      Vec mu_m = scft.get_mu_m();
      VecCopyGhost(mu_minus_field, mu_m);

      // initialize diffusion solvers for propagators
      scft.initialize_solvers();
      scft.initialize_bc_smart(step != 0);

      // main loop for solving SCFT equations
      int    scft_iteration = 0;
      int    bc_iters       = 0;
      double scft_error     = 2.*scft_tol+1.;

      while (scft_iteration < max_scft_iterations && scft_error > scft_tol || scft_iteration < bc_adjust_min+1)
      {
        // do an SCFT step
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();

        if (scft.get_exchange_force() < scft_bc_tol && bc_iters >= bc_adjust_min)
        {
          scft.initialize_bc_smart();
          if (smooth_pressure)
          {
            scft.smooth_singularity_in_pressure_field();
            smooth_pressure = false;
          }
          ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
          bc_iters = 0;
        }

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
      }
      // get energy
      energy = scft.get_energy();
      energy_eff_t = energy;

      scft.sync_and_extend();
      scft.compute_energy_shape_derivative(0, velo); //second part of dE/dt, which is derivative of integral of f dx
      VecScaleGhost(velo, -1.);

      VecCopyGhost(mu_m, mu_minus_field);
    }

    else
    {
//      ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_field); CHKERRXX(ierr);
//      sample_cf_on_nodes(p4est, nodes, mu_minus_t, mu_minus_field);

//      ierr = VecCreateGhostNodes(p4est, nodes, &gamma_effective_field); CHKERRXX(ierr);
//      sample_cf_on_nodes(p4est, nodes, gamma_effective_t, gamma_effective_field);

      VecSetGhost(velo, 0);
    }

    // calculate 'gamma_effective_field' by hand and not using 'sample_cf_on_nodes' (because mu_m is a vetor and not continuous function)
    double *gamma_effective_field_ptr;
    double *mu_m_ptr;
    ierr = VecGetArray(gamma_effective_field, &gamma_effective_field_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus_field, &mu_m_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      gamma_effective_field_ptr[n] = (0.5*(gamma_a+gamma_b) + (gamma_a-gamma_b)*(mu_m_ptr[n])/XN);;
    }

    ierr = VecRestoreArray(gamma_effective_field, &gamma_effective_field_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus_field, &mu_m_ptr); CHKERRXX(ierr);

    if (!use_scft)
    {
      energy_eff_t = integrate_over_interface(p4est, nodes, particles_level_set, gamma_effective_field);
    }


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
    Vec G_2_y;
    ierr = VecDuplicate(kappa, &G_2); CHKERRXX(ierr);
    ierr = VecDuplicate(kappa, &G_2_y); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &gamma_d[dim]); CHKERRXX(ierr); }
    ngbd->first_derivatives_central(gamma_effective_field, gamma_d);

    // compute del_gamma/del_n ((del_gamma/del_x)*n_x + (del_gamma/del_y)*n_y)
    foreach_dimension(dim)
    {
      if(dim == 0)
      {
      VecPointwiseMultGhost(G_2, gamma_d[dim], normal[dim]);
      }
      else{
      VecPointwiseMultGhost(G_2_y, gamma_d[dim], normal[dim]);
      }
    }
    VecAXPBYGhost(G_2, 1., 1., G_2_y); //G_2 = G_2 + 1*G_2_y

    // calculate G
    Vec G; // G = kappa*gamma_effecitve
    ierr = VecDuplicate(kappa, &G); CHKERRXX(ierr);
    VecPointwiseMultGhost(G, gamma_effective_field, kappa);
    VecAXPBYGhost(G, 1., 1., G_2); // G = G + 1*G_2
    VecAXPBYGhost(G, 1., 1., velo); //G = G + 1*velo

    // calculate g_x and g_y to get velocity
    Vec integrand_g[P4EST_DIM];
    std::vector<double> integrands_np; //integrands of every particle will be saved in this vector FOR EVERY NODE

    vector<double> vx(np);
    vector<double> vy(np);


    // calculate q_x and q_y (additional velocity terms; take into account what happens when particles come close to each other)
    vector<double> q_x(np, 0);
    vector<double> q_y(np, 0);
    double difference_x; //xc_i minus xc_j
    double difference_y; //yc_i minus yc_j
    double distance; // r_ij = sqrt((xc_i minus xc_j)^2 + (yc_i minus yc_j)^2)
    double energy_term = 0;
    double a = 0.3;
    double c = 0.1;

    for(int i = 0; i < np; i++){
      for(int j = 0; j < np; j++){
        if(i != j){
          difference_x = xc[i]-xc[j];
          difference_y = yc[i]-yc[j];
          distance     = sqrt(SQR(xc[i]-(xc[j])) + SQR(yc[i]-(yc[j])));

          q_x[i] = q_x[i] + (difference_x/distance)*(-a/(c*c)*distance)*exp(-SQR(distance)/(2*c*c)); // u = a*exp(-r^2/(2*c^2)) => use derivative of u for multiplication
          q_y[i] = q_y[i] + (difference_y/distance)*(-a/(c*c)*distance)*exp(-SQR(distance)/(2*c*c));

          energy_term = energy_term + a*exp(-SQR(distance)/(2*c*c));
        }
      }
    }

    for(int k = 0; k<q_x.size(); k++){
      std::cout << "q_x" << k << " is: " << q_x[k] << '\n';
    }

    for(int k = 0; k<q_y.size(); k++){
      std::cout << "q_y" << k << " is: " << q_y[k] << '\n';
    }

    // calculate the velocity from g_x and q_x as well as g_y and q_y
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &integrand_g[dim]); CHKERRXX(ierr); }
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(integrand_g[dim], G, normal[dim]);
      integrate_over_interface(np, integrands_np, p4est, nodes, particles_level_set, particle_number, integrand_g[dim]);
      for(int k = 0; k<integrands_np.size(); k++){
        std::cout << "the integrand is: " << integrands_np[k] << '\n';
      }
      if(dim == 0)
      {
        for(int i = 0; i < np; i++){
          vx[i] = -integrands_np[i] - q_x[i];
        }
      }
      else
      {
        for(int i = 0; i < np; i++){
          vy[i] = -integrands_np[i] - q_y[i];
        }
      }
    }

    double v_x_squared = 0;
    double v_y_squared = 0;
    for(int i = 0; i<np; i++){
      v_x_squared = v_x_squared + fabs(vx[i])*fabs(vx[i]);
      v_y_squared = v_y_squared + fabs(vy[i])*fabs(vy[i]);
    }

    dE_dt_expected = -v_x_squared-v_y_squared;

    // add correction term to the energy (when particles come too close to each other, 'push' them away)
    energy_eff_t = energy_eff_t + energy_term;

    for(int k = 0; k<vx.size(); k++){
      std::cout << "vx" << k << " is: " << vx[k] << '\n';
    }

    for(int k = 0; k<vy.size(); k++){
      std::cout << "vy" << k << " is: " << vy[k] << '\n';
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
//    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);
//    ierr = VecDestroy(particle_number); CHKERRXX(ierr);
//    ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
//    ierr = VecDestroy(gamma_effective_field); CHKERRXX(ierr);



    // find the maximum velocity (needed to calculate the timestep delta_t)
    double vx_max;
    double vy_max;
    double v_max;

    for(int i = 0; i < np; ++i){
      if(i == 0){
        vx_max = vx[0];
        vy_max = vy[0];
      }
      else{
        if(fabs(vx_max) < fabs(vx[i])){
          vx_max = vx[i];
        }
        if(fabs(vy_max) < fabs(vy[i])){
          vy_max = vy[i];
        }
      }
    }
    v_max = MAX(fabs(vx_max), fabs(vy_max));


    // find the diagonal length of the smallest quadrant (needed to calculate the timestep delta_t)
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    // CFL comes from a stability criterion (needed to calculate the timestep delta_t)
    double CFL = 0.5;

    // calculate timestep delta_t, such that it's small enough to capture 'everything'
    delta_t = diag_min*CFL/fabs(v_max);

    // calculate the expected and effective changes in energy (compare step i with step i+1)
    double expected_change_in_energy;
    expected_change_in_energy = dE_dt_expected*delta_t;

    double effective_change_in_energy;
    effective_change_in_energy = energy_eff_t - energy_eff_previous;

    // write the energy in every iteration into the separate file
    if (save_energy)
    {
      ierr = PetscFOpen(mpi.comm(), file_energy_name, "a", &file_energy); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_energy, "%d %f %f %f %f\n", step, energy_eff_t, dE_dt_expected, expected_change_in_energy, effective_change_in_energy);
      ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
    }




    // move particles and create new levelset
    for (int j = 0; j < np; ++j)
    {
      xc[j] = xc[j] + vx[j]*delta_t;
      yc[j] = yc[j] + vy[j]*delta_t;
    }


  //  ierr = PetscPrintf(mpi.comm(), "integrand: v_max=%g\n", integrands_np[1]); CHKERRXX(ierr);
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
    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_d[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(integrand_g[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    ierr = VecDestroy(G_2); CHKERRXX(ierr);
    ierr = VecDestroy(G); CHKERRXX(ierr);
   // ierr = VecDestroy(integrands_np); CHKERRXX(ierr);

    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);
    ierr = VecDestroy(particle_number); CHKERRXX(ierr);
//    ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);
    ierr = VecDestroy(gamma_effective_field); CHKERRXX(ierr);
  }

  ierr = VecDestroy(mu_minus_field); CHKERRXX(ierr);

  delete hierarchy;
  delete ngbd;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  p4est_connectivity_destroy(connectivity);


  /* show total running time */
  w.stop(); w.read_duration();

  return 0;
  }
