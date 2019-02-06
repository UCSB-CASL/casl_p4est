
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
#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>


#undef MIN
#undef MAX

using namespace std;
#include "shapes.h"

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

bool save_vtk = true;

double xmin = -2;
double xmax =  2;
double ymin = -2;
double ymax =  2;
double zmin = -2;
double zmax =  2;

#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
#else
int lmin = 6;
int lmax = 9;
#endif
double lip = 2.0;

int nx = 1;
int ny = 1;
int nz = 1;

const int periodic[] = {0, 0, 0};

double XN = 20.;
double f = 0.3;
int ns = 50+1;

int num_surfaces = 1;
std::vector<action_t> action(num_surfaces, INTERSECTION);
#endif

flower_shaped_domain_t domain_0_cf(1.4, 0.21, 0.13, 0.2, 1);

//inline double gamma_air(double x, double y) { return 0.0;} double_to_cf_t gamma_air_cf(&gamma_air);

class u_exact_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.0+sin(x)*cos(y)*(exp(5*t)-1.0);
  }
} u_exact_cf;

class us_exact_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 5*sin(x)*cos(y)*exp(5*t);
  }
} us_exact_cf;

class ux_exact_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return cos(x)*cos(y)*(exp(5*t)-1.0);
  }
} ux_exact_cf;

class uy_exact_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -sin(x)*sin(y)*(exp(5*t)-1.0);
  }
} uy_exact_cf;

class udd_exact_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -2.0*sin(x)*cos(y)*(exp(5*t)-1.0);
  }
} udd_exact_cf;

class mu_m_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -0.5*20.0*sin(x+y);
  }
} mu_m_cf;

class mu_p_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 0.2*20.0*sin(x+y);
//    return 1.0;
  }
} mu_p_cf;

class f_a_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    us_exact_cf.t = t;
    udd_exact_cf.t = t;
    u_exact_cf.t = t;
    return 0.0;
    return us_exact_cf(x,y) + (mu_p_cf(x,y)-mu_m_cf(x,y))*u_exact_cf(x,y) - udd_exact_cf(x,y);
  }
} f_a_cf;

class f_b_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    us_exact_cf.t = t;
    udd_exact_cf.t = t;
    u_exact_cf.t = t;
    return 0.0;
    return us_exact_cf(x,y) + (mu_p_cf(x,y)+mu_m_cf(x,y))*u_exact_cf(x,y) - udd_exact_cf(x,y);
  }
} f_b_cf;

class kappa_a_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 0.7;
  }
} kappa_a_cf;

class kappa_b_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 0.2;
  }
} kappa_b_cf;


class kappa_air_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 0.0;
  }
} kappa_air_cf;

class g_a_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    u_exact_cf.t = t;
    ux_exact_cf.t = t;
    uy_exact_cf.t = t;

    double nx = domain_0_cf.phi_x(x,y);
    double ny = domain_0_cf.phi_y(x,y);
    double norm = sqrt(nx*nx + ny*ny);
    nx /= norm;
    ny /= norm;
    return 0.0;
    return nx * ux_exact_cf(x,y) + ny * uy_exact_cf(x,y) + kappa_a_cf(x,y)*u_exact_cf(x,y);
  }
} g_a_cf;

class g_b_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    u_exact_cf.t = t;
    ux_exact_cf.t = t;
    uy_exact_cf.t = t;

    double nx = domain_0_cf.phi_x(x,y);
    double ny = domain_0_cf.phi_y(x,y);
    double norm = sqrt(nx*nx + ny*ny);
    nx /= norm;
    ny /= norm;
    return 0.0;
    return nx * ux_exact_cf(x,y) + ny * uy_exact_cf(x,y) + kappa_b_cf(x,y)*u_exact_cf(x,y);
  }
} g_b_cf;

class qb_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.0;
  }
} qb_cf;

//class phi_ref_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    double phi_total = (*phi_cfs[0])(x,y);
//    for (short i = 1; i < num_surfaces; i++)
//    {
//      double phi_current = (*phi_cfs[i])(x,y);
//      if      (action[i] == INTERSECTION) phi_total = MAX(phi_total, phi_current);
//      else if (action[i] == ADDITION)     phi_total = MIN(phi_total, phi_current);
//    }
//    return phi_total;
//  }
//} phi_ref_cf;

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

  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("lip", "Lipschitz constant");

  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.parse(argc, argv);

  cmd.print();

  XN = cmd.get("XN", XN);
  f = cmd.get("f", f);
  ns = cmd.get("ns", ns);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  lip = cmd.get("lip", lip);

  save_vtk = cmd.get("save_vtk", save_vtk);

  parStopWatch w;
  w.start("total time");

  /* create the p4est */
  my_p4est_brick_t brick;

  const int n_xyz[] = {nx, ny, nz};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};

  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &domain_0_cf.phi, lip);

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

//  for (short i = 0; i < num_surfaces; i++)
//  {
//    phi.push_back(Vec());
//    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est, nodes, *phi_cfs[i], phi.back());
//    ls.reinitialize_1st_order_time_2nd_order_space(phi.back());
//  }

  phi.push_back(Vec());
  ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, domain_0_cf.phi, phi.back());
  ls.reinitialize_1st_order_time_2nd_order_space(phi.back());

  /* create and initialize potentials */
  Vec mu_m;
  Vec mu_p;

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);

//  sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
//  sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

//  ierr = VecSet(mu_p, 0); CHKERRXX(ierr);
//  ierr = VecSetRandom(mu_m, NULL); CHKERRXX(ierr);

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

//  for (short i = 0; i < num_surfaces; i++)
//  {
//    gamma_a.push_back(kappa_a_cf[i]);
//    gamma_b.push_back(kappa_b_cf[i]);
//  }

  gamma_a.push_back(&kappa_a_cf);
  gamma_b.push_back(&kappa_b_cf);

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
  scft.set_parameters(f, XN, ns, 1, 1);
  scft.set_surface_tensions(gamma_a, gamma_b, kappa_air_cf);
  scft.set_potentials(mu_m, mu_p);
  scft.set_densities(rho_a, rho_b);
  scft.initialize_bc_simple();
//  scft.initialize_bc_smart();

  scft.initialize_linear_system();

//  scft.set_ic(u_exact);
//  scft.set_force_and_bc(f_a_cf, f_b_cf, g_a_cf, g_b_cf);

  scft.compute_normal_and_curvature();

  int iteration = 0;
  int save_every_n_iteration = 10;

  double tol = 1.0e-4;

  bool smooth_pressure = true;
  int bc_adjustments = 0;

  double energy_n = 0;
  double energy_nm1 = 0;

  bool initialized = false;

  int splits = 200;
  double full_interval = pow(2.0,lmax-lmin)*10.0*scft.dxyz_min;
  double small_interval = full_interval / (double) splits;

  double energy_change_theory = 0;
  double energy_change_exp = 0;

  while (iteration < splits && 1)
  {
    scft.set_ic(u_exact_cf);
    scft.set_force_and_bc(f_a_cf, f_b_cf, g_a_cf, g_b_cf);

    sample_cf_on_nodes(scft.p4est, scft.nodes, mu_m_cf, scft.mu_m);
    sample_cf_on_nodes(scft.p4est, scft.nodes, mu_p_cf, scft.mu_p);
    scft.solve_for_propogators();

//    u_exact_cf.t = 1;
//    sample_cf_on_nodes(scft.p4est, scft.nodes, u_exact_cf, scft.qf[scft.ns_total-1]);

//    u_exact_cf.t = 0;
//    sample_cf_on_nodes(scft.p4est, scft.nodes, u_exact_cf, scft.qf[0]);

//    scft.set_exact_solutions(u_exact_cf, qb_cf);

    scft.calculate_densities();
//    scft.compute_energy_shape_derivative(0);

//    for (int is = 0; is < ns; is++)
//      scft.save_VTK_q(is);

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
          velo_ptr[n] = cos(1.0*PI*x)*sin(1.0*PI*y);
//          velo_ptr[n] = sin(1.0*PI*x);
//      velo_ptr[n] = x;
    }
    ierr = VecRestoreArray(velocity, &velo_ptr); CHKERRXX(ierr);

    my_p4est_level_set_t ls(scft.ngbd);

    Vec velocity_int;
    ierr = VecCreateGhostNodes(scft.p4est, scft.nodes, &velocity_int); CHKERRXX(ierr);
    ls.extend_from_interface_to_whole_domain_TVD(scft.phi->at(0), velocity, velocity_int);

//        my_p4est_integration_mls_t integration;
//        integration.set_p4est(scft.p4est, scft.nodes);
//        integration.set_phi(*scft.phi, *scft.action, scft.color);

//        double velo_avg = integration.integrate_over_interface(velocity, 0)/integration.measure_of_interface(0);

//        ierr = VecGetArray(velocity, &velo_ptr); CHKERRXX(ierr);

//        for(size_t n=0; n<scft.nodes->indep_nodes.elem_count; ++n)
//        {
//          velo_ptr[n] -= velo_avg;
//        }
//        ierr = VecRestoreArray(velocity, &velo_ptr); CHKERRXX(ierr);


    if (iteration == 0) energy_change_exp = scft.get_energy();

    ierr = PetscPrintf(mpi.comm(), "Energy: %e;\n", scft.get_energy()); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Cumulative change in energy: exp %e, theory %e, diff %e\n", scft.get_energy()-energy_change_exp, energy_change_theory, fabs(scft.get_energy()-energy_change_exp-energy_change_theory)/fabs(scft.get_energy()-energy_change_exp)); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Immediate change in energy: exp -, theory %e, diff -\n", energy_nm1); CHKERRXX(ierr);

    scft.compute_energy_shape_derivative(0);
    scft.compute_energy_shape_derivative_alt(0);
//    energy_nm1 = scft.compute_change_in_energy(0, scft.energy_shape_deriv, 0.01*small_interval);
//    energy_nm1 = scft.compute_change_in_energy_alt(0, scft.energy_shape_deriv, 0.01*small_interval);
//    energy_nm1 = scft.compute_change_in_energy_alt(0, scft.energy_shape_deriv_alt, 0.01*small_interval);
    energy_nm1 = scft.compute_change_in_energy(0, velocity_int, small_interval);
//    energy_nm1 = scft.compute_change_in_energy_alt(0, velocity_int, small_interval);
    if (iteration < splits-1) energy_change_theory += energy_nm1;

    u_exact_cf.t = 1;
    sample_cf_on_nodes(scft.p4est, scft.nodes, u_exact_cf, scft.mu_m);
    Vec tmp;
    tmp = scft.mu_m;
    scft.mu_m = velocity_int;
    scft.save_VTK(iteration);
    scft.mu_m = tmp;

//    scft.update_grid(scft.energy_shape_deriv, 0, 0.01*small_interval);
//    scft.update_grid(scft.energy_shape_deriv_alt, 0, 0.01*small_interval);
    scft.update_grid(velocity_int, 0, small_interval);
    ierr = VecDestroy(velocity); CHKERRXX(ierr);
    ierr = VecDestroy(velocity_int); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Interface has been moved\n"); CHKERRXX(ierr);
    iteration++;
  }

  w.stop(); w.read_duration();

  return 0;
}
