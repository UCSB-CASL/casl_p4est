/*
 * Title: level_set_advection
 * Description:
 * Author: Fernando Temprano
 * Date Created: 11-06-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#endif

#include <iostream>
#include <iomanip>
#include <time.h>
#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>
#include <src/petsc_compatibility.h>

using namespace std;

// -----------------------------------------------------------------------------------------------------------------------
// Description of the main file
// -----------------------------------------------------------------------------------------------------------------------

const static std::string main_description = "\
 In this example, we illustrate and test the procedures to advect a moving interface using the level-set method within \n\
 the parCASL library.                                                                                                  \n\
 Example of application of interest: Advection of a moving interface following the level-set method.                   \n\
 Developer: Fernando Temprano-Coleto (ftempranocoleto@ucsb.edu), November 2019.                                        \n ";

// -----------------------------------------------------------------------------------------------------------------------
// Definition of the parameters of the example
// -----------------------------------------------------------------------------------------------------------------------

// Declare the parameter list object
param_list_t pl;

// Grid parameters
param_t<int>          n          (pl, 2,    "n",          "Number of trees in each direction (default: 2)");
param_t<unsigned int> lmin       (pl, 3,    "lmin",       "Min. level of refinement (default: 3)");
param_t<unsigned int> lmax       (pl, 5,    "lmax",       "Max. level of refinement (default: 5)");
param_t<double>       lip        (pl, 1.2,  "lip",        "Lipschitz constant (default: 1.2)");
param_t<unsigned int> num_splits (pl, 4,    "num_splits", "Number of recursive splits for convergence analysis (default: 4)");

// Method setup
param_t<double>       CFL        (pl, 1.0,  "cfl",        "CFL number (default: 1.0)");
param_t<double>       duration   (pl, 1.0,  "duration",   "Duration of the simulation (default: 1.0)");
param_t<bool>         save_vtk   (pl, 1,    "save_vtk",   "Duration of the simulation (default: 1)");
param_t<int>          num_it_vtk (pl, 8,    "num_it_vtk", "Save vtk files every num_it_vtk iterations (default: 8)");
param_t<int>          vel_interp (pl, 2,    "vel_interp", "Interpolation method for the velocity field (default: 1)\n"
                                                          "  0 - linear,\n"
                                                          "  1 - quadratic,\n"
                                                          "  2 - quadratic non oscillatory,\n"
                                                          "  3 - quadratic non oscillatory continuous v1,\n"
                                                          "  4 - quadratic non oscillatory continuous v2.");
param_t<int>          phi_interp (pl, 1,    "phi_interp", "Interpolation method for the level-set function (default: 2)\n"
                                                          "  0 - linear,\n"
                                                          "  1 - quadratic,\n"
                                                          "  2 - quadratic non oscillatory,\n"
                                                          "  3 - quadratic non oscillatory continuous v1,\n"
                                                          "  4 - quadratic non oscillatory continuous v2.");

// -----------------------------------------------------------------------------------------------------------------------
// Define auxiliary classes
// -----------------------------------------------------------------------------------------------------------------------

// Continuous signed-distance level-set function representing a circle (2D) or a sphere (3D)
#ifdef P4_TO_P8
struct sphere_ls : CF_3
{
  sphere_ls(double x0_, double y0_, double z0_, double R_): x0(x0_), y0(y0_), z0(z0_), R(R_) {}
#else
struct sphere_ls : CF_2
{
  sphere_ls(double x0_, double y0_, double R_): x0(x0_), y0(y0_), R(R_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
  {
    return sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0)) - R;
#else
  double operator()(double x, double y) const
  {
    return sqrt(SQR(x-x0) + SQR(y-y0)) - R;
#endif
  }
private:
  double x0, y0;
#ifdef P4_TO_P8
  double z0;
#endif
  double R;
};

// Velocity field
#ifdef P4_TO_P8
struct u_t : CF_3
{
private:
  double sign;
public:
  u_t() : sign(1.0) {;}
  void switch_direction() {sign *= -1.0;}
  double operator()(double x, double y, double z) const
  {
    return 2.0*SQR(sin(PI*x))*sin(2*PI*y)*sin(2*PI*z)*sign;
  }
};

struct v_t : CF_3
{
private:
  double sign;
public:
  v_t() : sign(1.0) {;}
  void switch_direction() {sign *= -1.0;}
  double operator()(double x, double y, double z) const
  {
    return -SQR(sin(PI*y))*sin(2*PI*x)*sin(2*PI*z)*sign;
  }
};

struct w_t : CF_3
{
private:
  double sign;
public:
  w_t() : sign(1.0) {;}
  void switch_direction() {sign *= -1.0;}
  double operator()(double x, double y, double z) const
  {
    return -SQR(sin(PI*z))*sin(2*PI*x)*sin(2*PI*y)*sign;
  }
};
#else
struct u_t : CF_2
{
private:
  double sign;
public:
  u_t() : sign(1.0) {;}
  void switch_direction() {sign *= -1.0;}
  double operator()(double x, double y) const
  {
    return -SQR(sin(PI*x))*sin(2*PI*y)*sign;
  }
};

struct v_t : CF_2
{
private:
  double sign;
public:
  v_t() : sign(1.0) {;}
  void switch_direction() {sign *= -1.0;}
  double operator()(double x, double y) const
  {
    return SQR(sin(PI*y))*sin(2*PI*x)*sign;
  }
};
#endif

// -----------------------------------------------------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

  // Prepare the parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Declaration of the stopwatch object
  parStopWatch sw;
  sw.start("Running example: level_set_advection");

  // Declaration of the PETSc error flag variable
  PetscErrorCode ierr;

  // Display initial message
  char init_msg[1024];
  sprintf(init_msg, "\n"
                    "-------------------------------------------------------------------\n"
                    "--------------== parCASL: level_set_advection test ==--------------\n"
                    "-------------------------------------------------------------------\n"
                    "\n");
  ierr = PetscPrintf(mpi.comm(), init_msg); CHKERRXX(ierr);

  // Get parameter values from the run command
  cmdParser cmd;
  pl.initialize_parser(cmd);
  if (cmd.parse(argc, argv, main_description)) return 0;
  pl.set_from_cmd_all(cmd);

  // Domain size information
  const int n_xyz[]      = { n(), n(), n()};
  const double xyz_min[] = { 0.0, 0.0, 0.0};
  const double xyz_max[] = { 1.0, 1.0, 1.0};
  const int periodic[]   = {   0,   0,   0};

  // Define the initial interface
  double R = 0.15;
#ifdef P4_TO_P8
  double x0 = 0.35;
  double y0 = 0.35;
  double z0 = 0.35;
  sphere_ls geom_0(x0,y0,z0,R);
#else
  double x0 = 0.5;
  double y0 = 0.75;
  sphere_ls geom_0(x0,y0,R);
#endif

  // Declaration of the macromesh via the brick and connectivity objects
  my_p4est_brick_t      brick;
  p4est_connectivity_t* conn;
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // Declaration of the error norm vector
  std::vector<double> err_linf(num_splits(), 0.0);

  // Loop through all grid resolutions
  for(unsigned int ll=0; ll<num_splits(); ++ll)
  {
    // Display initial message
    char split_msg[1024];
    sprintf(split_msg, "\n ----------------------== Resolution: %d/%d ==----------------------\n"
                       "\n Iteration %04d: t = %1.4f \n",
                       lmin(), lmax()+ll, 0, 0.0);
    ierr = PetscPrintf(mpi.comm(), split_msg); CHKERRXX(ierr);

    // Define the velocity field array
    u_t u;
    v_t v;
#ifdef P4_TO_P8
    w_t w;
    const CF_3 *v_cf[P4EST_DIM] = { &u, &v, &w };
#else
    const CF_2 *v_cf[P4EST_DIM] = { &u, &v };
#endif

    // Declaration of pointers to p4est variables
    p4est_t*       p4est_n;
    p4est_ghost_t* ghost_n;
    p4est_nodes_t* nodes_n;

    // Create the forest
    p4est_n = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

    // Refine based on distance to the interface
    splitting_criteria_cf_t sp(lmin(), lmax()+ll, &geom_0);
    p4est_n->user_pointer = &sp;
    for(unsigned int iter=0; iter<lmax()+ll; ++iter)
    {
      my_p4est_refine(p4est_n, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
    }

    // Define initial and final times
    double tn = 0.0;
    double tf = duration();

    // Compute grid size data
    double dxyz[P4EST_DIM];
    double dxyz_min;
    double diag_min;
    get_dxyz_min(p4est_n, dxyz, dxyz_min, diag_min);

    // Create ghost layer and nodes structure
    ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
    nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);

    // Create my_p4est objects
    my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
    my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
    ngbd_n->init_neighbors();

    // Declare data Vecs and pointers for read/write
    Vec phi_n;          double *phi_n_p;
    Vec phi_0_exact;    double *phi_0_exact_p;
    Vec v_n[P4EST_DIM]; double *v_n_p[P4EST_DIM];

    // Allocate memory for the Vecs
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n);       CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_0_exact); CHKERRXX(ierr);
    for(unsigned int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &v_n[dir]);  CHKERRXX(ierr);
    }

    // Sample the level-set function at t=0
    sample_cf_on_nodes(p4est_n, nodes_n, geom_0, phi_n);
    sample_cf_on_nodes(p4est_n, nodes_n, geom_0, phi_0_exact);

    // Sample the velocity field at t=0
    for(unsigned int dir=0; dir<P4EST_DIM; ++dir)
    {
      sample_cf_on_nodes(p4est_n, nodes_n, *v_cf[dir], v_n[dir]);
    }

    // Save the initial grid and fields into vtk
    if(save_vtk())
    {
      char name[1024];
      sprintf(name, "visualization_lmin%d_lmax%d_%d", lmin(), lmax()+ll, 0);
      ierr = VecGetArray(phi_n, &phi_n_p);             CHKERRXX(ierr);
      ierr = VecGetArray(phi_0_exact, &phi_0_exact_p); CHKERRXX(ierr);
      for(unsigned int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArray(v_n[dir], &v_n_p[dir]); CHKERRXX(ierr); }
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             2+P4EST_DIM, 0, name,
                             VTK_POINT_DATA, "phi", phi_n_p,
                             VTK_POINT_DATA, "phi_0_exact", phi_0_exact_p,
                             VTK_POINT_DATA, "v_x", v_n_p[0],
                             VTK_POINT_DATA, "v_y", v_n_p[1]
#ifdef P4_TO_P8
                            ,VTK_POINT_DATA, "v_z", v_n_p[2]
#endif
                                                             );
      ierr = VecRestoreArray(phi_n, &phi_n_p);             CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_0_exact, &phi_0_exact_p); CHKERRXX(ierr);
      for(unsigned int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArray(v_n[dir], &v_n_p[dir]); CHKERRXX(ierr); }
      char msg[1024];
      sprintf(msg, " -> Saving vtu files in %s.vtu\n", name);
      ierr = PetscPrintf(mpi.comm(), msg); CHKERRXX(ierr);
    }

    // Define time stepping variables
    bool has_vel_switched = false;
    int iter = 0;
    int vtk_idx = 1;
    double u_norm_max = 1.0; // Known analytically
    double dt_n = CFL()*dxyz_min/u_norm_max;

    while(tn+0.1*dt_n<tf)
    {
      // Clip time step if it's going to go over the final time
      if(tn+dt_n>tf)
        dt_n = tf-tn;

      // Clip time step if it's going to go over the half time
      if(tn+dt_n>=tf/2.0 && !has_vel_switched)
      {
        if(tn+dt_n>tf/2.0)
          dt_n = (tf/2.0)-tn;
        u.switch_direction();
        v.switch_direction();
#ifdef P4_TO_P8
        w.switch_direction();
#endif
        has_vel_switched = true;
      }

      // Declare p4est objects for tnp1
      p4est_t *p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Create semi-lagrangian object
      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n);
      switch(phi_interp())
      {
        case 0:  sl.set_phi_interpolation(linear); break;
        case 1:  sl.set_phi_interpolation(quadratic); break;
        case 2:  sl.set_phi_interpolation(quadratic_non_oscillatory); break;
        case 3:  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v1); break;
        case 4:  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2); break;
        default: throw std::invalid_argument("Invalid method.");
      }
      switch(vel_interp())
      {
        case 0:  sl.set_velo_interpolation(linear); break;
        case 1:  sl.set_velo_interpolation(quadratic); break;
        case 2:  sl.set_velo_interpolation(quadratic_non_oscillatory); break;
        case 3:  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v1); break;
        case 4:  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2); break;
        default: throw std::invalid_argument("Invalid method.");
      }

      // Advect the level-set function one step, then update the grid
      sl.update_p4est(v_n, dt_n, phi_n);

      // Destroy and create new structures
      p4est_destroy(p4est_n); p4est_n = p4est_np1;
      p4est_ghost_destroy(ghost_n); ghost_n = ghost_np1;
      p4est_nodes_destroy(nodes_n); nodes_n = nodes_np1;
      delete hierarchy_n; hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
      delete ngbd_n; ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
      ngbd_n->init_neighbors();

      // Reinitialize level-set function
      my_p4est_level_set_t ls(ngbd_n);
      ls.reinitialize_2nd_order(phi_n);

      // Update time and iteration counter
      tn += dt_n;
      iter++;
      if (tn==tf/2.0) { dt_n = CFL()*dxyz_min/u_norm_max; }

      // Re-sample the velocity field
      for(unsigned int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecDestroy(v_n[dir]);                             CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &v_n[dir]); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est_n, nodes_n, *v_cf[dir], v_n[dir]);
      }

      // Re-sample the exact initial level-set function
      ierr = VecDestroy(phi_0_exact);                             CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_0_exact); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est_n, nodes_n, geom_0, phi_0_exact);

      // Display iteration message
      char iter_msg[1024];
      sprintf(iter_msg, " Iteration %04d: t = %1.4f \n",
                        iter, tn);
      ierr = PetscPrintf(mpi.comm(), iter_msg); CHKERRXX(ierr);

      // Save to vtk format
      if(save_vtk() && (iter >= vtk_idx*num_it_vtk() || tn==tf))
      {
        char name[1024];
        sprintf(name, "visualization_lmin%d_lmax%d_%d", lmin(), lmax()+ll, vtk_idx);
        ierr = VecGetArray(phi_n, &phi_n_p);             CHKERRXX(ierr);
        ierr = VecGetArray(phi_0_exact, &phi_0_exact_p); CHKERRXX(ierr);
        for(unsigned int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArray(v_n[dir], &v_n_p[dir]); CHKERRXX(ierr); }
        my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                               P4EST_TRUE, P4EST_TRUE,
                               2+P4EST_DIM, 0, name,
                               VTK_POINT_DATA, "phi", phi_n_p,
                               VTK_POINT_DATA, "phi_0_exact", phi_0_exact_p,
                               VTK_POINT_DATA, "v_x", v_n_p[0],
                               VTK_POINT_DATA, "v_y", v_n_p[1]
  #ifdef P4_TO_P8
                              ,VTK_POINT_DATA, "v_z", v_n_p[2]
  #endif
                                                               );
        ierr = VecRestoreArray(phi_n, &phi_n_p);             CHKERRXX(ierr);
        ierr = VecRestoreArray(phi_0_exact, &phi_0_exact_p); CHKERRXX(ierr);
        for(unsigned int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArray(v_n[dir], &v_n_p[dir]); CHKERRXX(ierr); }
        char msg[1024];
        sprintf(msg, " -> Saving vtu files in %s.vtu\n", name);
        ierr = PetscPrintf(mpi.comm(), msg); CHKERRXX(ierr);
        ++vtk_idx;
      }
    }

    // Compute error l-inf norm and store it
    ierr = VecGetArray(phi_n, &phi_n_p);              CHKERRXX(ierr);
    ierr = VecGetArray(phi_0_exact,  &phi_0_exact_p); CHKERRXX(ierr);
    for(p4est_locidx_t n = 0; n < nodes_n->num_owned_indeps; ++n)
    {
      if(fabs(phi_n_p[n])<4.0*diag_min)
        err_linf[ll] = MAX(err_linf[ll], fabs(phi_n_p[n]-phi_0_exact_p[n]));
    }
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_linf[ll], 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    ierr = VecRestoreArray(phi_n, &phi_n_p);              CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_0_exact,  &phi_0_exact_p); CHKERRXX(ierr);

    // Destroy the dynamically allocated Vecs
    ierr = VecDestroy(phi_n);       CHKERRXX(ierr);
    ierr = VecDestroy(phi_0_exact); CHKERRXX(ierr);
    for(unsigned int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(v_n[dir]);  CHKERRXX(ierr);
    }

    // Destroy the dynamically allocated p4est and my_p4est structures
    delete ngbd_n;
    delete hierarchy_n;
    p4est_nodes_destroy   (nodes_n);
    p4est_ghost_destroy   (ghost_n);
    p4est_destroy         (p4est_n);
  }

  // Print out error table
  char table_msg[1024];
  sprintf(table_msg, "\n ------------------== Convergence in l-inf norm ==------------------\n\n");
  ierr = PetscPrintf(mpi.comm(), table_msg); CHKERRXX(ierr);
  for(unsigned int ll=0; ll<num_splits(); ++ll)
  {
    if(ll==0)
    {
      ierr = PetscPrintf(mpi.comm(), " Grid levels: %d/%d  |  error: %1.2e  |  order: N/A  \n", lmin(), lmax()+ll, err_linf[ll]); CHKERRXX(ierr);
    }
    else
    {
      ierr = PetscPrintf(mpi.comm(), " Grid levels: %d/%d  |  error: %1.2e  |  order: %1.3g\n", lmin(), lmax()+ll, err_linf[ll], log(err_linf[ll-1]/err_linf[ll])/log(2)); CHKERRXX(ierr);
    }
  }

  // Destroy the dynamically allocated brick and connectivity structures
  my_p4est_brick_destroy(conn, &brick);

  // Display end message
  char end_msg[1024];
  sprintf(end_msg, "\n -------------------------------------------------------------------\n"
                   "\n ");
  ierr = PetscPrintf(mpi.comm(), end_msg); CHKERRXX(ierr);

  // Stop and print global timer
  sw.stop();
  sw.read_duration();
}

