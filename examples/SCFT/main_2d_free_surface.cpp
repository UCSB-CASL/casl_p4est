
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
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_tools_mls.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_scft.h>
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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_tools_mls.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_scft.h>
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

double xmin = 0;
double ymin = 0;
double zmin = 0;

double xmax = 1;
double ymax = 1;
double zmax = 1;

bool px = 0;
bool py = 0;
bool pz = 0;

int nx = 1;
int ny = 1;
int nz = 1;

// grid parameters
#ifdef P4_TO_P8
int lmin = 1;
int lmax = 6;
#else
int lmin = 8;
int lmax = 8;
#endif
double lip = 1.5;

bool use_neumann = true;
int contact_angle_extension = 2;

int volume_corrections = 2;

bool use_scft = 1;
int  max_scft_iterations = 500;
int  bc_adjust_min = 5;
double scft_tol = 1.e-3;
double scft_bc_tol = 1.e-2;

bool smooth_pressure = 0;

double box_size = 20;

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

// polymer
double f = 0.5;
double XN = 20;
int ns = 40;

// output parameters
bool save_vtk = 1;
int save_every_dn = 10;

int num_drop = 0;
int num_geometry = 0;
int num_density = 1;
int num_surf_energy = 1;

int max_iterations = 1000;

double cfl = 0.5;

struct {
  double r = 0.3;
  double x = .5;
  double y = .3;
  double z = .5;

  double r0 = 0.4;

  double k = 5;

  double deform = 0.0;

} drop;

struct {
  double eps = -0.0; // curvature
  double nx =-0;
  double ny =-1;
  double nz =-0;
  double x  = .5;
  double y  = .35+0.01;
  double z  = -1.35;
} wall;

double gamma_Aw = 1;
double gamma_Bw = 2;
double gamma_Aa = 4;
double gamma_Ba = 1;
double gamma_aw = 1.5;

double G = 0.25;

class gamma_Aa_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_surf_energy)
    {
      case 0: return G*0.e-10;
      case 1: return G*1.;
      case 2: return G*4.;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_surf_energy)
    {
      case 0: return G*0.e-10;
      case 1: return G*1.;
      case 2: return G*1.;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_surf_energy)
    {
      case 0: return G*0.;
      case 1: return G*1.;
      case 2: return G*1.;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_Aw_cf;

class gamma_Bw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_surf_energy)
    {
      case 0: return G*0.;
      case 1: return G*4.;
      case 2: return G*2.;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_Bw_cf;

class gamma_aw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_surf_energy)
    {
      case 0: return G*0.;
      case 1: return G*1.;
      case 2: return G*1.5;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} gamma_aw_cf;


/* geometry of interfaces */

class phi_wall_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_geometry)
    {
      case 0:
        {
          double norm = sqrt(SQR(wall.nx) + SQR(wall.ny) P8( + SQR(wall.nz) ) );
          return - ( (x-wall.x)*((x-wall.x)*wall.eps - 2.*wall.nx / norm) +
                     (y-wall.y)*((y-wall.y)*wall.eps - 2.*wall.ny / norm)
           #ifdef P4_TO_P8
                   + (z-wall.z)*((z-wall.z)*wall.eps - 2.*wall.nz / norm)
           #endif
                         )
              / ( sqrt( SQR((x-wall.x)*wall.eps - wall.nx / norm) +
                        SQR((y-wall.y)*wall.eps - wall.ny / norm)
              #ifdef P4_TO_P8
                      + SQR((z-wall.z)*wall.eps - wall.nz / norm)
              #endif
                        ) + 1. );
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_wall_cf;

class phi_intf_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_drop)
    {
      case 0: return sqrt( SQR(x-drop.x) + SQR(y-drop.y) P8(+ SQR(z-drop.z)) )
            - drop.r
            *(1.+drop.deform*cos(drop.k*atan2(y-drop.y, x-drop.x))
      #ifdef P4_TO_P8
              *(1.-cos(2.*acos((z-drop.z)/sqrt( SQR(x-drop.x) + SQR(y-drop.y) + SQR(z-drop.z) + 1.e-12))))
      #endif
              );
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_intf_cf;

class mu_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z) ) const
  {
    switch (num_density)
    {
      case 0: return 0;
      case 1: return sin(6.*PI*x)*.5*XN;
      case 2: return sin(6.*PI*y)*.5*XN;
      case 3: return sin(6.*PI*x)*cos(6.*PI*y)*.5*XN;
#ifdef P4_TO_P8
      case 4: return sin(6.*PI*x)*cos(6.*PI*y)*cos(6.*PI*z)*.5*XN;
#endif
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} mu_cf;

class phi_eff_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z) ) const
  {
    return MIN( MAX(phi_intf_cf(x, y P8C(z)), phi_wall_cf(x, y P8C(z))), fabs(phi_wall_cf(x, y P8C(z))) );
  }
} phi_eff_cf;

// surface tension
class surf_tns_cf_t : public CF_1
{
public:
  double operator()(double x) const
  {
    return .5*(gamma_Aa + gamma_Ba) + .5*(gamma_Aa - gamma_Ba)*x;
  }
} surf_tns_cf;

class wall_tns_cf_t : public CF_1
{
public:
  double operator()(double x) const
  {
    return .5*(gamma_Aw + gamma_Bw) + .5*(gamma_Aw - gamma_Bw)*x;
  }
} wall_tns_cf;

// contact angle
class cos_angle_cf_t : public CF_1
{
public:
  double operator()(double x) const
  {
    return (.5*(gamma_Aw + gamma_Bw) + .5*(gamma_Aw - gamma_Bw)*x - gamma_aw)
        / (.5*(gamma_Aa + gamma_Ba) + .5*(gamma_Aa - gamma_Ba)*x);
  }
} cos_angle_cf;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  /* initialize MPI */
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  /* create an output directory */
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
    throw std::invalid_argument("Could not create directory");

  /* parse command line arguments */
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

    ADD_OPTION(cmd, save_vtk,  "save_vtk");

    PARSE_OPTIONS(cmd, argc, argv);
  }

  double scalling = 1./box_size;

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

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_inside_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, h, diag);

  /* initialize geometry */
  Vec phi_intf; ierr = VecCreateGhostNodes(p4est, nodes, &phi_intf); CHKERRXX(ierr);
  Vec phi_wall; ierr = VecCreateGhostNodes(p4est, nodes, &phi_wall); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_intf_cf, phi_intf);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

  /* initialize potentials */
  Vec mu_m; ierr = VecDuplicate(phi_intf, &mu_m); CHKERRXX(ierr);
  Vec mu_p; ierr = VecDuplicate(phi_intf, &mu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
  VecSetGhost(mu_p, 0);

  /* compute initial volume for volume-loss corrections */
  double volume = 0;

  {
    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> clr(2);

    phi[0] = phi_intf; clr[0] = 0;
    phi[1] = phi_wall; clr[1] = 1;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    volume = integration.measure_of_domain();
  }

  // in case of constant contact angle and simple geometry we can compute analytically position and volume of the steady-state shape
  double elev = 0;
  if (num_density == 0 && num_geometry == 0 && !use_scft)
  {
    double theta = acos(cos_angle_cf(0));

    elev = (wall.eps*drop.r0 - 2.*cos(PI-theta))*drop.r0
        /(sqrt(SQR(wall.eps*drop.r0) + 1. - 2.*wall.eps*drop.r0*cos(PI-theta)) + 1.);

    double alpha = acos((elev + .5*(SQR(drop.r0) + SQR(elev))*wall.eps)/drop.r0/(1.+wall.eps*elev));
#ifdef P4_TO_P8
    volume = 1./3.*PI*pow(drop.r0, 3.)*(2.*(1.-cos(PI-alpha)) + cos(alpha)*SQR(sin(alpha)));
#else
    volume = SQR(drop.r0)*(PI - alpha + cos(alpha)*sin(alpha));
#endif

    if (wall.eps != 0.)
    {
      double beta = acos(1.+.5*SQR(wall.eps)*(SQR(elev) - SQR(drop.r0))/(1.+wall.eps*elev));
#ifdef P4_TO_P8
      volume -= 1./3.*PI*(2.*(1.-cos(beta)) - cos(beta)*SQR(sin(beta)))/SQR(wall.eps)/wall.eps;
#else
      volume -= (beta - cos(beta)*sin(beta))/fabs(wall.eps)/wall.eps;
#endif
    }
  }

  double cos_angle_A = (gamma_Aw - gamma_aw)/gamma_Aa;
  double cos_angle_B = (gamma_Bw - gamma_aw)/gamma_Ba;

  ierr = PetscPrintf(mpi.comm(), "Contact angle limits: %e, %e\n", cos_angle_A, cos_angle_B);

  double energy = 0;
  double energy_old = 0;

  /* main loop */
  int iteration = 0;
  while (iteration < max_iterations)
  {
    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> acn_int(2, 0);
    std::vector<int> clr(2);
    std::vector<bool> refine_always(2);

    phi[0] = phi_intf; clr[0] = 0; refine_always[0] = false;
    phi[1] = phi_wall; clr[1] = 1; refine_always[1] = true;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi_intf, normal, kappa);

    Vec kappa_tmp;
    ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);

    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, kappa, kappa_tmp);

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    kappa = kappa_tmp;

    /* get density field and non-curvature velocity */
    Vec velo;
    ierr = VecDuplicate(phi_intf, &velo); CHKERRXX(ierr);

    if (use_scft)
    {
      my_p4est_scft_t scft(ngbd);

      scft.set_scalling(scalling);
      scft.set_polymer(f, XN, ns);
      scft.set_geometry(phi, acn);

      /* initialize potentials */
      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      std::vector<CF_DIM *> gamma_a_cf(2, NULL);
      std::vector<CF_DIM *> gamma_b_cf(2, NULL);

      gamma_a_cf[0] = &gamma_Aa_cf; gamma_a_cf[1] = &gamma_Aw_cf;
      gamma_b_cf[0] = &gamma_Ba_cf; gamma_b_cf[1] = &gamma_Bw_cf;

      scft.set_surface_tensions(gamma_a_cf, gamma_b_cf, gamma_aw_cf);

      scft.initialize_linear_system();

      scft.initialize_bc_smart(iteration != 0);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol;
      int bc_iters = 0;
      while (scft_iteration < max_scft_iterations && scft_error > scft_tol || scft_iteration < bc_adjust_min+1)
      {
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
          scft.recompute_matrices();
//          scft.initialize_linear_system();
          ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
          bc_iters = 0;
        }

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
      }

      energy = scft.get_energy();

      scft.compute_energy_shape_derivative(0, velo);

      VecScaleGhost(velo, -1);

      VecCopyGhost(mu_m_tmp, mu_m);
      VecCopyGhost(mu_p_tmp, mu_p);
    }
    else
    {
      sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
      VecSetGhost(mu_p, 0);
      VecSetGhost(velo, 0);

      // compute energy
      Vec mu_wall;
      Vec mu_intf;

      ierr = VecDuplicate(phi_intf, &mu_wall); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_intf, &mu_intf); CHKERRXX(ierr);

      ls.extend_from_interface_to_whole_domain_TVD(phi_wall, mu_m, mu_wall, 20);
      ls.extend_from_interface_to_whole_domain_TVD(phi_intf, mu_m, mu_intf, 20);

      energy  = .5*(gamma_Aa+gamma_Ba)*integration.measure_of_interface(0) + .5*(gamma_Aa-gamma_Ba)*integration.integrate_over_interface(0, mu_intf);
      energy += .5*(gamma_Aw+gamma_Bw)*integration.measure_of_interface(1) + .5*(gamma_Aw-gamma_Bw)*integration.integrate_over_interface(1, mu_wall);

      ierr = VecDestroy(mu_wall); CHKERRXX(ierr);
      ierr = VecDestroy(mu_intf); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e\n", energy, energy-energy_old);
    energy_old = energy;

    /* compute velocity and contact angle */
    Vec surf_tns;
    Vec cos_angle;

    ierr = VecDuplicate(phi_intf, &surf_tns); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_intf, &cos_angle); CHKERRXX(ierr);

    double *mu_ptr;
    double *surf_tns_ptr;
    double *cos_angle_ptr;

    ierr = VecGetArray(mu_m, &mu_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);
      double g_Aw = gamma_Aw_cf.value(xyz)*scalling;
      double g_Bw = gamma_Bw_cf.value(xyz)*scalling;
      double g_Aa = gamma_Aa_cf.value(xyz)*scalling;
      double g_Ba = gamma_Ba_cf.value(xyz)*scalling;
      double g_aw = gamma_aw_cf.value(xyz)*scalling;

      surf_tns_ptr[n] = (.5*(g_Aa+g_Ba) + (g_Aa-g_Ba)*mu_ptr[n]/XN)/volume;
      cos_angle_ptr[n] = (.5*(g_Aw+g_Bw) + (g_Aw-g_Bw)*mu_ptr[n]/XN - g_aw)
          / (.5*(g_Aa+g_Ba) + (g_Aa-g_Ba)*mu_ptr[n]/XN);
    }

    ierr = VecRestoreArray(mu_m, &mu_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

    // average velocity
    Vec integrand;
    Vec surf_tns_d[P4EST_DIM];
    Vec tmp;

    ierr = VecDuplicate(phi_intf, &integrand); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_intf, &tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &surf_tns_d[dim]); CHKERRXX(ierr); }

    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns, tmp);
    VecPointwiseMultGhost(integrand, kappa, tmp);

    ngbd->first_derivatives_central(surf_tns, surf_tns_d);
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(surf_tns_d[dim], surf_tns_d[dim], normal[dim]);
      ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns_d[dim], tmp);
      VecAXPBYGhost(integrand, 1, 1, tmp);
    }

    VecScaleGhost(integrand, -1);
    VecAXPBYGhost(integrand, 1, 1, velo);

    double vn_avg = integration.integrate_over_interface(0, integrand)/integration.measure_of_interface(0);

    ierr = VecDestroy(integrand); CHKERRXX(ierr);
    ierr = VecDestroy(tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDestroy(surf_tns_d[dim]); CHKERRXX(ierr); }

    VecShiftGhost(velo, -vn_avg);
//    VecSetGhost(vn, 0);

    Vec velo_tmp;
    ierr = VecDuplicate(phi_intf, &velo_tmp); CHKERRXX(ierr);

    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, velo, velo_tmp);

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    velo = velo_tmp;

    /* compute time step dt */
    double dt_local = DBL_MAX;
    double dt;

    double *velo_ptr;

    ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;
    foreach_local_node(n, nodes)
    {
      ngbd->get_neighbors(n, qnnn);

      double xyzn[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyzn);

      double s_p00 = fabs(qnnn.d_p00); double s_m00 = fabs(qnnn.d_m00);
      double s_0p0 = fabs(qnnn.d_0p0); double s_0m0 = fabs(qnnn.d_0m0);
  #ifdef P4_TO_P8
      double s_00p = fabs(qnnn.d_00p); double s_00m = fabs(qnnn.d_00m);
  #endif
  #ifdef P4_TO_P8
      double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m));
  #else
      double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0));
  #endif

      /* choose CFL = 0.8 ... just for fun! */
      dt_local = MIN(dt_local, 0.8*fabs(s_min/velo_ptr[n]));
    }

    ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);

    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

    ierr = PetscPrintf(mpi.comm(), "Avg velo: %e, Time step: %e\n", vn_avg, dt);

//    Vec surf_tns_tmp;
//    ierr = VecDuplicate(phi_intf, &surf_tns_tmp); CHKERRXX(ierr);
//    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns, surf_tns_tmp, 20);
//    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
//    surf_tns = surf_tns_tmp;

    /* save data */
    if (save_vtk && iteration%save_every_dn == 0)
    {
      // Effective phi
      Vec phi_eff;
      ierr = VecDuplicate(phi_intf, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(nodes, &phi, &acn_int, NULL, phi_eff);

      // compute reference solution
      Vec phi_exact;
      ierr = VecDuplicate(phi_intf, &phi_exact); CHKERRXX(ierr);
      VecSetGhost(phi_exact, -1);

      if (num_geometry == 0 && num_density == 0 && !use_scft)
      {
        Vec XYZ[P4EST_DIM];
        double *xyz_ptr[P4EST_DIM];

        foreach_dimension(dim)
        {
          ierr = VecCreateGhostNodes(p4est, nodes, &XYZ[dim]); CHKERRXX(ierr);
          ierr = VecGetArray(XYZ[dim], &xyz_ptr[dim]); CHKERRXX(ierr);
        }

        double xyz[P4EST_DIM];
        foreach_node(n, nodes)
        {
          node_xyz_fr_n(n, p4est, nodes, xyz);
          foreach_dimension(dim) xyz_ptr[dim][n] = xyz[dim];
        }

        foreach_dimension(dim) {
          ierr = VecRestoreArray(XYZ[dim], &xyz_ptr[dim]); CHKERRXX(ierr);
        }

        double com[P4EST_DIM];
        double vol = integration.measure_of_domain();
        foreach_dimension(dim)
            com[dim] = integration.integrate_over_domain(XYZ[dim])/vol;

#ifdef P4_TO_P8
        double vec_d[P4EST_DIM] = { wall.x - com[0], wall.y - com[1], wall.z - com[2] };
        double nw_norm = sqrt(SQR(wall.nx) + SQR(wall.ny) + SQR(wall.nz));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]) + SQR(vec_d[2]);
        double dn0 = (vec_d[0]*wall.nx + vec_d[1]*wall.ny + vec_d[2]*wall.nz)/nw_norm;
#else
        double vec_d[P4EST_DIM] = { wall.x - com[0], wall.y - com[1] };
        double nw_norm = sqrt(SQR(wall.nx) + SQR(wall.ny));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]);
        double dn0 = (vec_d[0]*wall.nx + vec_d[1]*wall.ny)/nw_norm;
#endif

        double del = (d_norm2*wall.eps + 2.*dn0)
            / ( sqrt(1. + (d_norm2*wall.eps + 2.*dn0)*wall.eps) + 1. );

        double vec_n[P4EST_DIM];

        vec_n[0] = (wall.nx/nw_norm + vec_d[0]*wall.eps)/(1.+wall.eps*del);
        vec_n[1] = (wall.ny/nw_norm + vec_d[1]*wall.eps)/(1.+wall.eps*del);
#ifdef P4_TO_P8
        vec_n[2] = (wall.nz/nw_norm + vec_d[2]*wall.eps)/(1.+wall.eps*del);
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]) + SQR(vec_n[2]));
#else
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]));
#endif

        double xyz_c[P4EST_DIM];
        foreach_dimension(dim)
            xyz_c[dim] = com[dim] + (del-elev)*vec_n[dim]/norm;

#ifdef P4_TO_P8
        flower_shaped_domain_t exact(drop.r0, xyz_c[0], xyz_c[1], xyz_c[2]);
#else
        flower_shaped_domain_t exact(drop.r0, xyz_c[0], xyz_c[1]);
#endif

        sample_cf_on_nodes(p4est, nodes, exact.phi, phi_exact);

        foreach_dimension(dim) {
          ierr = VecDestroy(XYZ[dim]); CHKERRXX(ierr);
        }
      }

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
       #endif
             "." << (int) round(iteration/save_every_dn);

      PetscPrintf(mpi.comm(), "VTK is being saved in %s\n", oss.str().c_str());

      /* save the size of the leaves */
      Vec leaf_level;
      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      double *l_p;
      ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

      for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          l_p[tree->quadrants_offset+q] = quad->level;
        }
      }

      for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
      {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
      }

      double *phi_ptr;
      double *phi_wall_ptr;
      double *phi_intf_ptr;
      double *phi_exact_ptr;
      double *kappa_ptr;
      double *surf_tns_ptr;
      double *mu_m_ptr;
      double *mu_p_ptr;
      double *velo_ptr;

      ierr = VecGetArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             9, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "phi_wall", phi_wall_ptr,
                             VTK_POINT_DATA, "phi_intf", phi_intf_ptr,
                             VTK_POINT_DATA, "phi_exact", phi_exact_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "velo", velo_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      ierr = VecDestroy(phi_exact); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    /* advect interface and impose contact angle */
    ls.set_use_neumann_for_contact_angle(use_neumann);
    ls.set_contact_angle_extension(contact_angle_extension);
    ls.advect_in_normal_direction_with_contact_angle(velo, surf_tns, cos_angle, phi_wall, phi_intf, dt);

    ls.reinitialize_1st_order_time_2nd_order_space(phi_intf, 50);

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
    ierr = VecDestroy(cos_angle); CHKERRXX(ierr);

    /* correct for volume loss */
    for (short i = 0; i < volume_corrections; ++i)
    {
      double volume_cur = integration.measure_of_domain();
      double intf_len   = integration.measure_of_interface(0);
      double correction = (volume_cur-volume)/intf_len;

      VecShiftGhost(phi_intf, correction);

      double volume_cur2 = integration.measure_of_domain();

      PetscPrintf(mpi.comm(), "Volume loss: %e, after correction: %e\n", (volume_cur-volume)/volume, (volume_cur2-volume)/volume);
    }

    /* refine and coarsen grid */
    {
      Vec phi_eff;
      double *phi_eff_ptr;

      ierr = VecDuplicate(phi_intf, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(nodes, &phi, &acn_int, &refine_always, phi_eff);
      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin, lmax, lip);
      sp.set_refine_only_inside(1);

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      bool is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

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

        Vec phi_wall_tmp;
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_wall_tmp); CHKERRXX(ierr);
        interp.set_input(phi_wall, interpolation_between_grids);
        interp.interpolate(phi_wall_tmp);
        ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
        phi_wall = phi_wall_tmp;

        Vec phi_intf_tmp;
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_intf_tmp); CHKERRXX(ierr);
        interp.set_input(phi_intf, interpolation_between_grids);
        interp.interpolate(phi_intf_tmp);
        ierr = VecDestroy(phi_intf); CHKERRXX(ierr);
        phi_intf = phi_intf_tmp;

        Vec mu_m_tmp;
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &mu_m_tmp); CHKERRXX(ierr);
        interp.set_input(mu_m, interpolation_between_grids);
        interp.interpolate(mu_m_tmp);
        ierr = VecDestroy(mu_m); CHKERRXX(ierr);
        mu_m = mu_m_tmp;

        Vec mu_p_tmp;
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &mu_p_tmp); CHKERRXX(ierr);
        interp.set_input(mu_p, interpolation_between_grids);
        interp.interpolate(mu_p_tmp);
        ierr = VecDestroy(mu_p); CHKERRXX(ierr);
        mu_p = mu_p_tmp;

        // delete old p4est
        p4est_destroy(p4est);       p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;
        hierarchy->update(p4est, ghost);
        ngbd->update(hierarchy, nodes);
      }
    }
    iteration++;

    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi_intf); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
  ierr = VecDestroy(phi_intf); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
