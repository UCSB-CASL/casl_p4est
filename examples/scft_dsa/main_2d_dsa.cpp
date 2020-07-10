
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
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_scft.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

param_list_t pl;

using namespace std;

// comptational domain
param_t<double> xmin (pl, -1, "xmin", "xmin");
param_t<double> ymin (pl, -1, "ymin", "ymin");
param_t<double> zmin (pl, -1, "zmin", "zmin");

param_t<double> xmax (pl,  1, "xmax", "xmax");
param_t<double> ymax (pl,  1, "ymax", "ymax");
param_t<double> zmax (pl,  1, "zmax", "zmax");

param_t<int> nx (pl, 1, "nx", "number of trees in x-dimension");
param_t<int> ny (pl, 1, "ny", "number of trees in y-dimension");
param_t<int> nz (pl, 1, "nz", "number of trees in z-dimension");

// grid parameters
#ifdef P4_TO_P8
param_t<int> lmin (pl, 7, "lmin", "min level of trees");
param_t<int> lmax (pl, 7, "lmax", "max level of trees");
#else
param_t<int> lmin (pl, 7, "lmin", "min level of trees");
param_t<int> lmax (pl, 7, "lmax", "max level of trees");
#endif

param_t<double> lip (pl, 1.5, "lip", "Lipschitz constant");

// advection parameters
param_t<double> cfl               (pl, 0.5, "cfl", "");
param_t<int>    max_iterations    (pl, 1000, "max_iterations", "");
param_t<int>    scheme            (pl, 2, "scheme", "0 - pure curvature, 1 - Gaddiel's method, 2 - gradient-based method");
param_t<double> curvature_penalty (pl, 0.01, "curvature_penalty", "");

// scft parameters
param_t<int>    max_scft_iterations (pl, 500, "max_scft_iterations",   "Maximum SCFT iterations");
param_t<int>    bc_adjust_min       (pl, 5, "bc_adjust_min",     "Minimun SCFT steps between adjusting BC");
param_t<bool>   smooth_pressure     (pl, 1, "smooth_pressure",     "Smooth pressure after first BC adjustment 0/1");
param_t<double> scft_tol            (pl, 1.e-3, "scft_tol", "Tolerance for SCFT");
param_t<double> scft_bc_tol         (pl, 1.e-2, "scft_bc_tol", "Tolerance for adjusting BC");

// polymer
param_t<double> box_size (pl, 7, "box_size.val", "Box size in units of Rg");
param_t<double> f        (pl, 0.3, "f", "Fraction of polymer A");
param_t<double> XN       (pl, 25, "XN.val",  "Flory-Higgins interaction parameter");
param_t<int>    ns       (pl, 40, "ns",  "Discretization of polymer chain");

// output parameters
param_t<bool> save_vtk        (pl, 1, "save_vtk", "");
param_t<int>  save_data       (pl, 1, "save_data", "");
param_t<int>  save_parameters (pl, 1, "save_parameters", "");
param_t<int>  save_every_dn   (pl, 1, "save_every_dn", "");

// problem setting
param_t<int> num_target  (pl, 1, "num_target.val", "Target design: "
                                                   "0 - one circle, "
                                                   "1 - two circles, "
                                                   "2 - three circles, "
                                                   "3 - four circles");
param_t<int> num_guess   (pl, 0, "num_guess", "Type of initial guess: 0 - target with margins, 1 - enclosing box");
param_t<int> num_seed    (pl, 0, "num_seed", "Seed for SCFT: 0 - target field");
param_t<int> num_example (pl, 0, "num_example", "");

// geometry parameters
param_t<double> r0               (pl, 1, "r0", "Radius of target wells");
param_t<double> guess_margin     (pl, 1.5, "guess_margin", "");
param_t<double> target_smoothing (pl, sqrt(XN.val), "target_smoothing", "");

// surface tension
param_t<double> XN_wall_avg (pl, 0, "XN.val_wall_avg", "Polymer-air surface energy strength: average");
param_t<double> XN_wall_del (pl, 0, "XN.val_wall_del", "Polymer-air surface energy strength: difference");

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

/* target morphology */
class phi_target_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_target.val)
    {
      case 0: // 1 cylinder
        {
          double xc0 = 0.0, yc0 = 0.0;

          return sqrt(SQR(x-xc0/box_size.val)+SQR(y-yc0/box_size.val)) - r0.val/box_size.val;
        }
      case 1: // 2 cylinders
        {
          double xc0 = -1.6, yc0 = 0;
          double xc1 =  1.6, yc1 = 0;

          double phi0 = sqrt(SQR(x-xc0/box_size.val)+SQR(y-yc0/box_size.val)) - r0.val/box_size.val;
          double phi1 = sqrt(SQR(x-xc1/box_size.val)+SQR(y-yc1/box_size.val)) - r0.val/box_size.val;

          return MIN(phi0, phi1);
        }
      case 2: // 3 cylinders
        {
          double dist = 1.7;
          double h = dist*cos(PI/6.);
          double xc0 = -dist, yc0 = -h;
          double xc1 =  dist, yc1 = -h;
          double xc2 =  0.0,  yc2 =  h;

          double phi0 = sqrt(SQR(x-xc0/box_size.val)+SQR(y-yc0/box_size.val)) - r0.val/box_size.val;
          double phi1 = sqrt(SQR(x-xc1/box_size.val)+SQR(y-yc1/box_size.val)) - r0.val/box_size.val;
          double phi2 = sqrt(SQR(x-xc2/box_size.val)+SQR(y-yc2/box_size.val)) - r0.val/box_size.val;

          return MIN(phi0, phi1, phi2);
        }
      case 3: // 4 cylinders
        {
          double dist = 1.7;
          double xc0 = -dist, yc0 = -dist;
          double xc1 =  dist, yc1 = -dist;
          double xc2 = -dist, yc2 =  dist;
          double xc3 =  dist, yc3 =  dist;

          double phi0 = sqrt(SQR(x-xc0/box_size.val)+SQR(y-yc0/box_size.val)) - r0.val/box_size.val;
          double phi1 = sqrt(SQR(x-xc1/box_size.val)+SQR(y-yc1/box_size.val)) - r0.val/box_size.val;
          double phi2 = sqrt(SQR(x-xc2/box_size.val)+SQR(y-yc2/box_size.val)) - r0.val/box_size.val;
          double phi3 = sqrt(SQR(x-xc3/box_size.val)+SQR(y-yc3/box_size.val)) - r0.val/box_size.val;

          return MIN(MIN(phi0, phi1), MIN(phi2, phi3));
        }
      case 4: // rectangular
      {
        double H = 1/box_size.val;
        double W = 7/box_size.val;
        double phi_v = MAX(x-.5*W, -x-.5*W);
        double phi_h = MAX(y-.5*H, -y-.5*H);

        return smooth_max(phi_h, phi_v, 0.01);
      }
      default: throw std::invalid_argument("Choose a valid test number");
    }
  }
} phi_target_cf;

class target_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    double phi = phi_target_cf(DIM(x, y, z));
    double sign = -1;

    if      (phi > 0) sign = 1;
    else if (phi < 0) sign =-1;
    else              sign = 0;

    return 0.5*XN.val*(sign*(exp(-fabs(phi)*target_smoothing.val*box_size.val)-1.0));
  }
} target_cf;

/* guess */
class guess_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_guess.val)
    {
      case 0: return phi_target_cf(DIM(x, y, z)) - guess_margin.val/box_size.val;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} guess_cf;

/* seed */
class seed_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z) ) const
  {
    switch (num_seed.val)
    {
      case 0: return 0;
      case 1: return sin(6.*PI*x)*.5*XN.val;
      case 2: return sin(6.*PI*y)*.5*XN.val;
      case 3: return sin(6.*PI*x)*cos(6.*PI*y)*.5*XN.val;
#ifdef P4_TO_P8
      case 4: return sin(6.*PI*x)*cos(6.*PI*y)*cos(6.*PI*z)*.5*XN.val;
#endif
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} seed_cf;

/* surface energies */
class gamma_A_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    EXECD((void) x, (void) y, (void) z);
    return sqrt(XN_wall_avg.val + XN_wall_del.val);
  }
} gamma_A_cf;

class gamma_B_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    EXECD((void) x, (void) y, (void) z);
    return sqrt(XN_wall_avg.val - XN_wall_del.val);
  }
} gamma_B_cf;

inline void interpolate_between_grids(my_p4est_interpolation_nodes_t &interp, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, Vec *vec, Vec parent=NULL, interpolation_method method=quadratic_non_oscillatory_continuous_v2)
{
  PetscErrorCode ierr;
  Vec tmp;

  if (parent == NULL) {
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &tmp); CHKERRXX(ierr);
  } else {
    ierr = VecDuplicate(parent, &tmp); CHKERRXX(ierr);
  }

  interp.set_input(*vec, method);
  interp.interpolate(tmp);

  ierr = VecDestroy(*vec); CHKERRXX(ierr);
  *vec = tmp;
}

PetscErrorCode ierr;

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
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  double scaling = 1./box_size.val;

  double xyz_min[] = { DIM(xmin.val, ymin.val, zmin.val) };
  double xyz_max[] = { DIM(xmax.val, ymax.val, zmax.val) };
  int nb_trees[] = { DIM(nx.val, ny.val, nz.val) };
  int periodic[] = { DIM(0, 0, 0) };

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin.val, lmax.val, &guess_cf, lip.val);
  data.set_refine_only_inside(1);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, &h, &diag);

  /* initialize geometry */
  Vec phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, guess_cf, phi);

  /* initialize potentials */
  Vec mu_m; ierr = VecDuplicate(phi, &mu_m); CHKERRXX(ierr);
  Vec mu_p; ierr = VecDuplicate(phi, &mu_p); CHKERRXX(ierr);
  Vec nu_m; ierr = VecDuplicate(phi, &nu_m); CHKERRXX(ierr);
  Vec nu_p; ierr = VecDuplicate(phi, &nu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, target_cf, mu_m);
  VecSetGhost(mu_p, 0);
  VecSetGhost(nu_m, 0);
  VecSetGhost(nu_p, 0);

  double cost_function_nnn = 0;
  double cost_function_nm1 = 0;

  double rho_avg_old = 1;

  /* main loop */
  int iteration = 0;
  while (iteration < max_iterations.val)
  {
    std::vector<Vec> phi_all(1);
    std::vector<mls_opn_t> acn(1, MLS_INTERSECTION);
    std::vector<int> clr(1);

    phi_all[0] = phi; clr[0] = 0;

    Vec mu_t;
    ierr = VecDuplicate(phi, &mu_t); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, target_cf, mu_t);

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi_all, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi, normal, kappa);
    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, kappa, phi);

    /* get density field and non-curvature velocity */
    Vec velo;
    ierr = VecDuplicate(phi, &velo); CHKERRXX(ierr);

    if (scheme.val == 2)
    {
      my_p4est_scft_t scft(ngbd, ns.val);

      scft.set_scaling(scaling);
      scft.set_polymer(f.val, XN.val);
      scft.add_boundary(phi, MLS_INT, gamma_A_cf, gamma_B_cf);
      scft.set_rho_avg(rho_avg_old);

      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      scft.initialize_solvers();
      scft.initialize_bc_smart(iteration != 0);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol.val;
      int bc_iters = 0;
      while ((scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val) ||
             (scft_iteration < bc_adjust_min.val+1) ||
             bc_iters == 0)
      {

        for (int i=3;i--;) {
          scft.initialize_bc_smart(adaptive); adaptive = true;
          scft.solve_for_propogators();
          scft.calculate_densities();
//          scft.save_VTK(scft_iteration++);
          ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                             scft_iteration,
                             scft.get_energy(),
                             scft.get_pressure_force(),
                             scft.get_exchange_force()); CHKERRXX(ierr);
        }
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();
        scft_iteration++;
        bc_iters++;

        if (scft.get_exchange_force() < scft_bc_tol.val && bc_iters >= bc_adjust_min.val)
        {
          scft.initialize_bc_smart();
          if (smooth_pressure.val)
          {
            scft.smooth_singularity_in_pressure_field();
            smooth_pressure.val = false;
          }
          ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
          bc_iters = 0;
        }

        scft.save_VTK(scft_iteration);
        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
      }

      rho_avg_old = scft.get_rho_avg();

      scft.sync_and_extend();
      scft.dsa_initialize();

      Vec mu_t_tmp = scft.dsa_get_mu_t();
      Vec nu_m_tmp = scft.dsa_get_nu_m();
      Vec nu_p_tmp = scft.dsa_get_nu_p();

      VecCopyGhost(mu_t, mu_t_tmp);
      VecCopyGhost(nu_m, nu_m_tmp);
      VecCopyGhost(nu_p, nu_p_tmp);

      scft_iteration = 0;
      scft_error = 2.*scft_tol.val;
      while (scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val)
      {
        scft.dsa_solve_for_propogators();
        scft.dsa_compute_densities();
        scft.dsa_update_potentials();
        ierr = PetscPrintf(mpi.comm(), "Energy: %e; Pressure: %e; Exchange: %e\n", scft.dsa_get_nu_0(), scft.dsa_get_pressure_force(), scft.dsa_get_exchange_force()); CHKERRXX(ierr);
        scft_error = fabs(scft.dsa_get_exchange_force());
        scft_iteration++;
      }

      scft.dsa_sync_and_extend();

      cost_function_nnn = scft.dsa_get_cost_function();
      scft.dsa_compute_shape_gradient(0, velo);

      VecScaleGhost(velo, -1);

      VecCopyGhost(mu_m_tmp, mu_m);
      VecCopyGhost(mu_p_tmp, mu_p);

      VecCopyGhost(nu_m_tmp, nu_m);
      VecCopyGhost(nu_p_tmp, nu_p);
    }
    else if (scheme.val == 1)
    {
      sample_cf_on_nodes(p4est, nodes, target_cf, mu_m);

      VecSetGhost(nu_m, 0);
      VecSetGhost(nu_p, 0);

      my_p4est_scft_t scft(ngbd, ns.val);

      scft.set_scaling(scaling);
      scft.set_polymer(f.val, XN.val);
      scft.add_boundary(phi, MLS_INT, gamma_A_cf, gamma_B_cf);
      scft.set_rho_avg(1);

      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      scft.initialize_solvers();
      scft.initialize_bc_smart(true);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol.val;
      while (scft_iteration < max_scft_iterations.val && scft_error > scft_tol.val)
      {
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials(false, true);
        scft_iteration++;

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = fabs(scft.get_pressure_force());
      }

      scft.sync_and_extend();

      VecCopyGhost(mu_p_tmp, velo);
      VecScaleGhost(velo, 1);
      VecCopyGhost(mu_p_tmp, mu_p);

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, velo, phi);
    }
    else if (scheme.val == 0)
    {
      sample_cf_on_nodes(p4est, nodes, target_cf, mu_m);
      VecSetGhost(mu_p, 0);
      VecSetGhost(velo, 0);

      VecSetGhost(nu_m, 0);
      VecSetGhost(nu_p, 0);
    }

    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e\n", cost_function_nnn, cost_function_nnn-cost_function_nm1);
    cost_function_nm1 = cost_function_nnn;

    /* compute velocity and contact angle */
    Vec surf_tns;
    Vec cos_angle;

    ierr = VecDuplicate(phi, &surf_tns); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &cos_angle); CHKERRXX(ierr);

    double *mu_m_ptr;
    double *mu_t_ptr;
    double *nu_m_ptr;
    double *surf_tns_ptr;

    ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);
//      double gA = gamma_A_cf.value(xyz)*scaling;
//      double gB = gamma_B_cf.value(xyz)*scaling;

      surf_tns_ptr[n] = curvature_penalty.val + (sqrt(XN_wall_avg.val+XN_wall_del.val) - sqrt(XN_wall_avg.val-XN_wall_del.val))*(mu_m_ptr[n] - mu_t_ptr[n] + nu_m_ptr[n])/XN.val;
    }

    ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);

    // average velocity
    Vec velo_full;
    Vec velo_full_tmp;
    Vec surf_tns_d[P4EST_DIM];
    Vec tmp;

    ierr = VecDuplicate(phi, &velo_full); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &velo_full_tmp); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &surf_tns_d[dim]); CHKERRXX(ierr); }

    ls.extend_from_interface_to_whole_domain_TVD(phi, surf_tns, tmp);
    VecPointwiseMultGhost(velo_full_tmp, kappa, tmp);

    ngbd->first_derivatives_central(surf_tns, surf_tns_d);
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(surf_tns_d[dim], surf_tns_d[dim], normal[dim]);
      ls.extend_from_interface_to_whole_domain_TVD(phi, surf_tns_d[dim], tmp);
      VecAXPBYGhost(velo_full_tmp, 1, 1, tmp);
    }

    VecScaleGhost(velo_full_tmp, -1);
    VecAXPBYGhost(velo_full_tmp, 1, 1, velo);

    ls.extend_from_interface_to_whole_domain_TVD(phi, velo_full_tmp, velo_full);

    ierr = VecDestroy(tmp); CHKERRXX(ierr);
    ierr = VecDestroy(velo_full_tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDestroy(surf_tns_d[dim]); CHKERRXX(ierr); }

    double vn_avg = integration.integrate_over_interface(0, velo_full)/integration.measure_of_interface(0);

//    VecShiftGhost(velo, -vn_avg);
//    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi, velo, phi);

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

      dt_local = MIN(dt_local, cfl.val*fabs(s_min/velo_ptr[n]));
    }

    ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);

    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

    ierr = PetscPrintf(mpi.comm(), "Avg velo: %e, Time step: %e\n", vn_avg, dt);

//    Vec surf_tns_tmp;
//    ierr = VecDuplicate(phi, &surf_tns_tmp); CHKERRXX(ierr);
//    ls.extend_from_interface_to_whole_domain_TVD(phi, surf_tns, surf_tns_tmp, 20);
//    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
//    surf_tns = surf_tns_tmp;

    /* save data */
    if (save_vtk.val && iteration%save_every_dn.val == 0)
    {
      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nx.valyztrees[2] <<
       #endif
             "." << (int) round(iteration/save_every_dn.val);

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
      double *kappa_ptr;
      double *surf_tns_ptr;
      double *mu_t_ptr;
      double *mu_m_ptr;
      double *mu_p_ptr;
      double *nu_m_ptr;
      double *nu_p_ptr;
      double *velo_ptr;
      double *velo_full_ptr;

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             10, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_t", mu_t_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "nu_m", nu_m_ptr,
                             VTK_POINT_DATA, "nu_p", nu_p_ptr,
                             VTK_POINT_DATA, "velo", velo_ptr,
                             VTK_POINT_DATA, "velo_full", velo_full_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_t, &mu_t_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(nu_m, &nu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(nu_p, &nu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    /* advect interface and impose contact angle */
    ls.set_use_neumann_for_contact_angle(0);
    ls.set_contact_angle_extension(0);
    ls.advect_in_normal_direction_with_contact_angle(velo, surf_tns, NULL, NULL, phi, dt);

    ls.reinitialize_1st_order_time_2nd_order_space(phi, 50);

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    ierr = VecDestroy(velo_full); CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);

    /* refine and coarsen grid */
    {
      double *phi_ptr;
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin.val, lmax.val, lip.val);
      sp.set_refine_only_inside(1);

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      bool is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_ptr);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);

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

        interpolate_between_grids(interp, p4est_np1, nodes_np1, &phi, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &mu_m, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &mu_p, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &nu_m, phi, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, &nu_p, phi, interpolation_between_grids);

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
    ierr = VecDestroy(mu_t); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}

