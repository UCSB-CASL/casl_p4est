
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
#include <src/simplex3_mls_vtk.h>
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
#include <src/simplex2_mls_vtk.h>
#include <src/my_p4est_scft.h>
#endif

#include <src/point3.h>
//#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>


#undef MIN
#undef MAX

using namespace std;
#include "shapes.h"

bool save_vtk = true;

double xmin = -3;
double xmax =  3;
double ymin = -3;
double ymax =  3;
double zmin = -2;
double zmax =  2;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
#else
int lmin = 6;
int lmax = 8;
#endif
double lip = 1.5;

int steps_back = 3;
//int steps_back = lmax-lmin;

int nx = 1;
int ny = 1;
int nz = 1;

const int periodic[] = {0, 0, 0};

//

double XN = 20.;
double f = 0.3;
int ns = 100+1;

int ns_coarsed = 50+1;

double tol = 1.0e-4;
int save_every_n_iteration = 10;

int num_of_moves = 1000;
int num_of_bc_adjustments = 10;
double interface_speed = 0.1;

bool save_change = true;


int num_surfaces = 2;
std::vector<action_t> action(num_surfaces, INTERSECTION);

flower_phi_t phi_cf_0(2.5, 0.2, 0.2, 0.0, 1);
flower_phi_t phi_cf_1(2.5, -2.5, -2.5, 0.0, -1);

CF_2* phi_cfs[] = {&phi_cf_0, &phi_cf_1};

#ifdef P4_TO_P8
class double_to_cf_t : public CF_3
{
public:
  double (*func)(double, double, double);
  double_to_cf_t(double (*func)(double, double, double)) { this->func = func; }
  double operator()(double x, double y, double z) const { return (*func)(x,y,z); }
};
#else
class double_to_cf_t : public CF_2
{
public:
  double (*func)(double, double);
  double_to_cf_t(double (*func)(double, double)) { this->func = func; }
  double operator()(double x, double y) const { return (*func)(x,y); }
};
#endif

inline double gamma_air(double x, double y) { return 0.0;} double_to_cf_t gamma_air_cf(&gamma_air);

inline double gamma_a_0(double x, double y) { return 1.0-0.3*cos(5.*atan2(x-0.2,y-0.2));} double_to_cf_t gamma_a_0_cf(&gamma_a_0);
inline double gamma_b_0(double x, double y) { return 1.0+0.3*cos(5.*atan2(x-0.2,y-0.2));} double_to_cf_t gamma_b_0_cf(&gamma_b_0);

//inline double gamma_a_0(double x, double y) { return 0;} double_to_cf_t gamma_a_0_cf(&gamma_a_0);
//inline double gamma_b_0(double x, double y) { return 0;} double_to_cf_t gamma_b_0_cf(&gamma_b_0);

inline double gamma_a_1(double x, double y) { return 1.0+0.5;} double_to_cf_t gamma_a_1_cf(&gamma_a_1);
inline double gamma_b_1(double x, double y) { return 1.0;} double_to_cf_t gamma_b_1_cf(&gamma_b_1);

//inline double mu_m_test(double x, double y) { return XN*(0.2+0.9*sin(x-y)); } double_to_cf_t mu_m_test_cf(&mu_m_test);
//inline double mu_p_test(double x, double y) { return XN*(0.2+0.9*sin(x+y)); } double_to_cf_t mu_p_test_cf(&mu_p_test);

inline double mu_m_test(double x, double y) { return 0.5*XN*sin(x+y); } double_to_cf_t mu_m_test_cf(&mu_m_test);
inline double mu_p_test(double x, double y) { return -0.2*XN*sin(x+y); } double_to_cf_t mu_p_test_cf(&mu_p_test);
//inline double mu_p_test(double x, double y) { return XN*(0.2+0.9*sin(x+y)); } double_to_cf_t mu_p_test_cf(&mu_p_test);


//inline double mu_m_test(double x, double y) { return -0.5*XN*log(2.0+sin(x+y)); } double_to_cf_t mu_m_test_cf(&mu_m_test);
//inline double mu_p_test(double x, double y) { return 0.5*XN*log(2.0+sin(x+y)); } double_to_cf_t mu_p_test_cf(&mu_p_test);


CF_2* gamma_a_cf[] = {&gamma_a_0_cf, &gamma_a_1_cf};
CF_2* gamma_b_cf[] = {&gamma_b_0_cf, &gamma_b_1_cf};

class phi_ref_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double phi_total = (*phi_cfs[0])(x,y);
    for (short i = 1; i < num_surfaces; i++)
    {
      double phi_current = (*phi_cfs[i])(x,y);
      if      (action[i] == INTERSECTION) phi_total = MAX(phi_total, phi_current);
      else if (action[i] == ADDITION)     phi_total = MIN(phi_total, phi_current);
    }
    return phi_total;
  }
} phi_ref_cf;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("XN", "Florry-Higgins parameter");
  cmd.add_option("f", "fraction of polymer A");
  cmd.add_option("ns", "number of steps along the polymer shain");
  cmd.add_option("tol", "tolerance for SCFT equations");

  cmd.add_option("interface_speed", "interface_speed");
  cmd.add_option("num_of_moves", "num_of_moves");

  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("lip", "Lipschitz constant");

  cmd.add_option("steps_back", "number of recursive splits");
  cmd.add_option("ns_coarsed", "chain discretization for coarse grids");

  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("save_change", "track change in the cost functional");
  cmd.add_option("save_every_n_iteration", "save_every_n_iteration");

  cmd.parse(argc, argv);

  cmd.print();

  XN = cmd.get("XN", XN);
  f = cmd.get("f", f);
  ns = cmd.get("ns", ns);

  interface_speed = cmd.get("interface_speed", interface_speed);
  num_of_moves = cmd.get("num_of_moves", num_of_moves);

  tol = cmd.get("tol", tol);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  lip = cmd.get("lip", lip);

  steps_back = cmd.get("steps_back", steps_back);
  ns_coarsed = cmd.get("ns_coarsed", ns_coarsed);

  save_vtk = cmd.get("save_vtk", save_vtk);
  save_change = cmd.get("save_change", save_change);

  save_every_n_iteration = cmd.get("save_every_n_iteration", save_every_n_iteration);

  parStopWatch w;
  w.start("total time");

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

  /* create the p4est */
  my_p4est_brick_t brick;

  const int n_xyz[] = {nx, ny, nz};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};

  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_ref_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_inside_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
//  p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//  my_p4est_ghost_expand(p4est, ghost);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);

  ngbd->init_neighbors();

  my_p4est_level_set_t ls(ngbd);

  /* create and initialize geometry */
  std::vector<Vec> phi;

  for (short i = 0; i < num_surfaces; i++)
  {
    phi.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *phi_cfs[i], phi.back());
    ls.reinitialize_1st_order_time_2nd_order_space(phi.back());
  }

  /* create and initialize potentials */
  Vec mu_m;
  Vec mu_p;

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);

//  sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
//  sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

  ierr = VecSet(mu_p, 0); CHKERRXX(ierr);
  ierr = VecSetRandom(mu_m, NULL); CHKERRXX(ierr);

  /* create and initialize density fields */
  Vec rho_a;
  Vec rho_b;

  ierr = VecCreateGhostNodes(p4est, nodes, &rho_a); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_b); CHKERRXX(ierr);

  Vec vec_local;

  ierr = VecGhostGetLocalForm(rho_a, &vec_local); CHKERRXX(ierr);
  ierr = VecSet(vec_local, f); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho_a, &vec_local); CHKERRXX(ierr);

  ierr = VecGhostGetLocalForm(rho_b, &vec_local); CHKERRXX(ierr);
  ierr = VecSet(vec_local, 1.-f); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(rho_b, &vec_local); CHKERRXX(ierr);

  /* create and initialize surface tension fields */
  std::vector<CF_2 *> gamma_a;
  std::vector<CF_2 *> gamma_b;

  for (short i = 0; i < num_surfaces; i++)
  {
    gamma_a.push_back(gamma_a_cf[i]);
    gamma_b.push_back(gamma_b_cf[i]);
  }

  // create an output directory
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return -1;
  }
  std::ostringstream command;
  command << "mkdir -p " << out_dir << "/vtu";
  int ret_sys = system(command.str().c_str());
  if(ret_sys<0)
    throw std::invalid_argument("my_p4est_bialloy_t::save_vtk could not create directory");

  /* initialize solver */
  my_p4est_scft_t scft(ngbd);

  scft.set_geometry(phi, action);
  scft.set_parameters(f, XN, ns);
  scft.set_surface_tensions(gamma_a, gamma_b, gamma_air_cf);
  scft.set_potentials(mu_m, mu_p);
  scft.set_densities(rho_a, rho_b);

  scft.initialize_bc_simple();
//  scft.initialize_bc_smart();

  scft.initialize_linear_system();
  scft.compute_normal_and_curvature();

  // ----------------------------------------------------------------------------------------------------------------
  // pre-calculations on coarser grids to speed up the process
  // ----------------------------------------------------------------------------------------------------------------
  // initialize auxiliary solver on a coarser grid
  for (short iter = -steps_back; iter < 0; iter++)
  {
    /* create the p4est */
    my_p4est_brick_t brick_aux;

    p4est_connectivity_t *connectivity_aux = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick_aux, periodic);
    p4est_t *p4est_aux = my_p4est_new(mpi.comm(), connectivity_aux, 0, NULL, NULL);

    splitting_criteria_cf_t data_aux(lmin+iter, lmax+iter, &phi_ref_cf, lip);

    p4est_aux->user_pointer = (void*)(&data_aux);
    my_p4est_refine(p4est_aux, P4EST_TRUE, refine_inside_levelset_cf, NULL);
    my_p4est_partition(p4est_aux, P4EST_FALSE, NULL);

    p4est_ghost_t *ghost_aux = my_p4est_ghost_new(p4est_aux, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_aux = my_p4est_nodes_new(p4est_aux, ghost_aux);

    my_p4est_hierarchy_t *hierarchy_aux = new my_p4est_hierarchy_t(p4est_aux,ghost_aux, &brick_aux);
    my_p4est_node_neighbors_t *ngbd_aux = new my_p4est_node_neighbors_t(hierarchy_aux,nodes_aux);

    ngbd_aux->init_neighbors();

    my_p4est_level_set_t ls_aux(ngbd_aux);

    /* create and initialize geometry */
    std::vector<Vec> phi_aux;

    for (short i = 0; i < num_surfaces; i++)
    {
      phi_aux.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est_aux, nodes_aux, &phi_aux.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est_aux, nodes_aux, *phi_cfs[i], phi_aux.back());
      ls_aux.reinitialize_1st_order_time_2nd_order_space(phi_aux.back());
    }

    /* create and initialize potentials */
    Vec mu_m_aux;
    Vec mu_p_aux;

    ierr = VecCreateGhostNodes(p4est_aux, nodes_aux, &mu_m_aux); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_aux, nodes_aux, &mu_p_aux); CHKERRXX(ierr);

    VecSet(mu_p_aux, 0);

    // TODO: interpolate from mu_m and mu_p to mu_m_aux and mu_p_aux
    my_p4est_interpolation_nodes_t interp_to_aux(scft.ngbd);

    double xyz[P4EST_DIM];
    for(size_t n=0; n<nodes_aux->indep_nodes.elem_count; ++n)
    {
      node_xyz_fr_n(n, p4est_aux, nodes_aux, xyz);
      interp_to_aux.add_point(n, xyz);
    }

    interp_to_aux.set_input(scft.mu_m, quadratic_non_oscillatory);
    interp_to_aux.interpolate(mu_m_aux);

    interp_to_aux.set_input(scft.mu_p, quadratic_non_oscillatory);
    interp_to_aux.interpolate(mu_p_aux);

    interp_to_aux.clear();

    /* create and initialize density fields */
    Vec rho_a_aux;
    Vec rho_b_aux;

    ierr = VecCreateGhostNodes(p4est_aux, nodes_aux, &rho_a_aux); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_aux, nodes_aux, &rho_b_aux); CHKERRXX(ierr);

    ierr = VecSet(rho_a_aux, f); CHKERRXX(ierr);
    ierr = VecSet(rho_b_aux, 1.-f); CHKERRXX(ierr);

    /* initialize solver */
    my_p4est_scft_t scft_aux(ngbd_aux);

    scft_aux.set_geometry(phi_aux, action);
    scft_aux.set_parameters(f, XN, ns_coarsed);
    scft_aux.set_surface_tensions(gamma_a, gamma_b, gamma_air_cf);
    scft_aux.set_potentials(mu_m_aux, mu_p_aux);
    scft_aux.set_densities(rho_a_aux, rho_b_aux);

    scft_aux.initialize_bc_simple();
    //  scft.initialize_bc_smart();

    scft_aux.initialize_linear_system();
    scft_aux.compute_normal_and_curvature();

    scft_aux.force_m_avg = 1.0;

    int local_iteration = 0;
    while (scft_aux.get_exchange_force() > 3.0e-3)
    {
      scft_aux.solve_for_propogators();
      scft_aux.calculate_densities();
      scft_aux.update_potentials();
      if (local_iteration%save_every_n_iteration == 0)
      {
        ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e\n", scft_aux.get_energy(), scft_aux.get_pressure_force(), scft_aux.get_exchange_force()); CHKERRXX(ierr);
      }
      local_iteration++;
    }

    // sync and extend potentials
    ierr = VecGhostUpdateBegin(scft_aux.mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(scft_aux.mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (scft_aux.mu_m, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (scft_aux.mu_p, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ls_aux.extend_Over_Interface_TVD(scft_aux.phi_smooth, scft_aux.mu_m);
    ls_aux.extend_Over_Interface_TVD(scft_aux.phi_smooth, scft_aux.mu_p);

    // TODO interpolate back from mu_m_aux to mu_p_aux
    my_p4est_interpolation_nodes_t interp_from_aux(scft_aux.ngbd);

    for(size_t n=0; n<scft.nodes->indep_nodes.elem_count; ++n)
    {
      node_xyz_fr_n(n, scft.p4est, scft.nodes, xyz);
      interp_from_aux.add_point(n, xyz);
    }

    interp_from_aux.set_input(scft_aux.mu_m, quadratic_non_oscillatory);
    interp_from_aux.interpolate(scft.mu_m);

    interp_from_aux.set_input(scft_aux.mu_p, quadratic_non_oscillatory);
    interp_from_aux.interpolate(scft.mu_p);

    interp_from_aux.clear();
  }
  // ----------------------------------------------------------------------------------------------------------------


  int iteration = 0;

  double energy_n = 0;
  double energy_nm1 = 0;

  double immed_change_thr = 0;
  double immed_change_exp = 0;

  double total_change_thr = 0;
  double total_change_exp = 0;

  double initial_energy = 0;

  double contact_term = 0;

  bool first_time = true;

  FILE *fich;
  char name[10000];

  if (!out_dir) out_dir = ".";
#ifdef P4_TO_P8
  sprintf(name, "%s/velo_%dx%dx%d_L_%g_G_%g_V_%g_box_%g_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], n_xyz[2], latent_heat_orig, G_orig, V_orig, box_size, lmin, lmax);
#else
  sprintf(name, "%s/change_%dx%d_level_%d-%d.dat", out_dir, n_xyz[0], n_xyz[1], lmin, lmax);
#endif

  if(save_change)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "%% iteration cost_funct immed_change_exp immed_change_thr total_change_exp total_change_thr\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  std::vector<double> energies(num_of_bc_adjustments, 0);

  double Hnm1 = 0;
  double delH = 0;

  for (int move_iter = 0; move_iter < num_of_moves; move_iter++)
  {
    // solve scft equations
    for (int i = 0; i < num_of_bc_adjustments; i++)
    {
      if (i == 0) scft.initialize_bc_simple();
      else        scft.initialize_bc_smart();
      scft.initialize_linear_system();
      do {
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();

        if (iteration%save_every_n_iteration == 0)
        {
          scft.save_VTK(iteration/save_every_n_iteration);
          ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e; Change in H: %e\n", scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), delH); CHKERRXX(ierr);
        }
        iteration++;

        delH = abs(Hnm1-scft.get_energy());

        Hnm1 = scft.get_energy();
      } while (scft.get_pressure_force() > tol || scft.get_exchange_force() > tol);
      energies[i] = scft.get_energy();

      if (i == 0) scft.smooth_singularity_in_pressure_field();
    }

    ierr = PetscPrintf(mpi.comm(), "Energies with various BCs:\n"); CHKERRXX(ierr);
    for (int i = 0; i < num_of_bc_adjustments; i++)
    {
      if (i == 0)
      { ierr = PetscPrintf(mpi.comm(), "Energy n %d = %e\n", i, energies[i]); CHKERRXX(ierr); }
      else
      { ierr = PetscPrintf(mpi.comm(), "Energy n %d = %e; Change = %e\n", i, energies[i], fabs(energies[i]-energies[i-1])); CHKERRXX(ierr); }
    }

    // advance interface

    // imposed velocity
    Vec velocity;
    ierr = VecCreateGhostNodes(scft.p4est, scft.nodes, &velocity); CHKERRXX(ierr);

    double *velo_ptr;
    ierr = VecGetArray(velocity, &velo_ptr); CHKERRXX(ierr);

    for(size_t n=0; n<scft.nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, scft.p4est, scft.nodes);
      double y = node_y_fr_n(n, scft.p4est, scft.nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, scft.p4est, scft.nodes);
#endif
//          velo_ptr[n] = sin(PI*x)*cos(PI*y);
//      velo_ptr[n] = -sin(x)*cos(y);
      velo_ptr[n] = -cos(5.*atan2(x,y));
//          velo_ptr[n] = -1.0;
    }
    ierr = VecRestoreArray(velocity, &velo_ptr); CHKERRXX(ierr);

    my_p4est_level_set_t ls(scft.ngbd);

    Vec velocity_int;
    ierr = VecCreateGhostNodes(scft.p4est, scft.nodes, &velocity_int); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(scft.phi->at(0), velocity, velocity_int);

    if (first_time)
    {
      initial_energy = scft.get_energy();
      energy_nm1 = scft.get_energy();
      first_time = false;
    }

    total_change_exp = scft.get_energy()-initial_energy;
    immed_change_exp = scft.get_energy()-energy_nm1;

    ierr = PetscPrintf(mpi.comm(), "Func: %e;\n", scft.get_energy()); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Cumulative change in cost func: exp %e, theory %e, diff %e\n", total_change_exp, total_change_thr, fabs(total_change_thr-total_change_exp)/fabs(total_change_exp)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Immediate change in cost func: exp %e, theory %e, diff %e\n", immed_change_exp, immed_change_thr, fabs(immed_change_thr-immed_change_exp)/fabs(immed_change_exp)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Contact term: %e\n", contact_term); CHKERRXX(ierr);

    if(save_change)
    {
      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), fich, "%d %e %e %e %e %e\n", move_iter, scft.get_energy(), immed_change_exp, immed_change_thr, total_change_exp, total_change_thr);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "saved change in %s\n", name); CHKERRXX(ierr);
    }

    energy_nm1 = scft.get_energy();

    scft.compute_energy_shape_derivative(0);
//    scft.compute_energy_shape_derivative_contact_term(1,0);

    double *shape_grad_ptr;
    ierr = VecGetArray(velocity_int, &velo_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(scft.energy_shape_deriv, &shape_grad_ptr); CHKERRXX(ierr);

    for(size_t n=0; n<scft.nodes->indep_nodes.elem_count; ++n)
    {
      velo_ptr[n] = -shape_grad_ptr[n];
    }
    ierr = VecRestoreArray(velocity_int, &velo_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(scft.energy_shape_deriv, &shape_grad_ptr); CHKERRXX(ierr);

    my_p4est_integration_mls_t integration;
    integration.set_p4est(scft.p4est, scft.nodes);
    integration.set_phi(*scft.phi, *scft.action, scft.color);

    double velo_avg = integration.integrate_over_interface(velocity_int, 0)/integration.measure_of_interface(0);

    ierr = VecGetArray(velocity_int, &velo_ptr); CHKERRXX(ierr);

    for(size_t n=0; n<scft.nodes->indep_nodes.elem_count; ++n)
    {
      velo_ptr[n] -= velo_avg;
    }
    ierr = VecRestoreArray(velocity_int, &velo_ptr); CHKERRXX(ierr);

//    immed_change_thr = scft.compute_change_in_energy(0, velocity_int, interface_speed*scft.dxyz_min);
    immed_change_thr = scft.compute_change_in_energy(0, velocity_int, interface_speed*scft.dt_energy);
//    contact_term = scft.compute_change_in_energy_contact_term(1, 0, velocity_int, interface_speed*scft.dxyz_min);
//    contact_term = scft.compute_change_in_energy_contact_term(1, 0, velocity_int, interface_speed*scft.dt_energy);
//    immed_change_thr += contact_term;

    total_change_thr += immed_change_thr;

//    scft.update_grid(velocity_int, 0, small_interval);


    Vec mu_m_tmp;
    mu_m_tmp = scft.mu_m;
    scft.mu_m = velocity_int;

    Vec mu_p_tmp;
    mu_p_tmp = scft.mu_p;
    scft.mu_p = scft.energy_shape_deriv;

    scft.save_VTK(move_iter);

    scft.mu_m = mu_m_tmp;
    scft.mu_p = mu_p_tmp;

//    ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e\n", scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);
//    scft.update_grid(velocity_int, 0, interface_speed*scft.dxyz_min);
    scft.update_grid(velocity_int, 0, interface_speed*scft.dt_energy);
    ierr = PetscPrintf(mpi.comm(), "Interface has been moved\n"); CHKERRXX(ierr);

    ierr = VecDestroy(velocity); CHKERRXX(ierr);
    ierr = VecDestroy(velocity_int); CHKERRXX(ierr);

  }

  w.stop(); w.read_duration();

  return 0;
}
