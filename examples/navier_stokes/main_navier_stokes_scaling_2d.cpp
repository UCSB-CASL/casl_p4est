
/*
 * The navier stokes solver
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

double xmin = 0;
double xmax = 1;
double ymin = 0;
double ymax = 1;
#ifdef P4_TO_P8
double zmin = 0;
double zmax = 1;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

/*
 *  ********* 2D and 3D *********
 * Run karman street 10 times, 1 iteration only
 *
 */

double mu;
double rho;
double tn;
double dt;
double u0;
double r0;

#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y, double z) const
  {
    return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2) + SQR(z-(zmax+zmin)/2));
  }
} level_set;

struct BCWALLTYPE_P : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
  }
} bc_wall_type_v;

struct BCWALLTYPE_W : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
  }
} bc_wall_type_w;

struct BCWALLVALUE_U : CF_3
{
  double operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS) return 0; else return u0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_v;

struct BCWALLVALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_w;

struct BCINTERFACE_VALUE_U : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_v;

struct BCINTERFACE_VALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_w;

struct initial_velocity_unm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return u0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return u0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_vnm1;

struct initial_velocity_v_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_vn;

struct initial_velocity_wnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_wnm1;

struct initial_velocity_w_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_wn;

struct external_force_u_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} external_force_u;

struct external_force_v_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} external_force_v;

struct external_force_w_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} external_force_w;

#else

class LEVEL_SET: public CF_2
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2));
  }
} level_set;

struct BCWALLTYPE_P : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
  }
} bc_wall_type_v;

struct BCWALLVALUE_U : CF_2
{
  double operator()(double x, double y) const
  {
    if(fabs(x-xmax)<EPS) return 0; else return u0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} bc_wall_value_v;

struct BCINTERFACE_VALUE_U : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} bc_interface_value_v;

struct initial_velocity_unm1_t : CF_2
{
  double operator()(double x, double y) const
  {
    return u0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_2
{
  double operator()(double x, double y) const
  {
    return u0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} initial_velocity_vn;

struct external_force_u_t : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} external_force_u;

struct external_force_v_t : CF_2
{
  double operator()(double x, double y) const
  {
    return 0;
  }
} external_force_v;

#endif


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "the number of trees in the x direction");
  cmd.add_option("ny", "the number of trees in the y direction");
#ifdef P4_TO_P8
  cmd.add_option("nz", "the number of trees in the z direction");
#endif
  cmd.add_option("tf", "the final time");
  cmd.add_option("Re", "the reynolds number.");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx");
  cmd.add_option("n_times_dt", "dx = n_times_dt * dx/vmax");
  cmd.add_option("thresh", "the threshold used for the refinement criteria");
  cmd.add_option("nb_repeat", "number of time to repeat the process");
  cmd.add_option("nb_iter", "number of iterations for each repeat");
  cmd.parse(argc, argv);

  cmd.print();

  int lmin = cmd.get("lmin", 2);
  int lmax = cmd.get("lmax", 4);
  double n_times_dt = cmd.get("n_times_dt", 2.);
  double threshold_split_cell = cmd.get("thresh", 0.04);
  int nb_repeat = cmd.get("nb_repeat", 1);
  int nb_iter = cmd.get("nb_iter", 1);

  double Re;
  rho = 1;
  u0 = 0;
  double tf;

#ifdef P4_TO_P8
  nx=8; ny=4; nz=4; xmin=0; xmax=32; ymin=-8; ymax=8; zmin=-8; zmax=8; Re=cmd.get("Re",350); r0=1; u0=1; rho=1; mu=2*r0*rho*u0/Re; tf=cmd.get("tf",200);
#else
  nx=8; ny=4; xmin = 0; xmax = 32; ymin =-8; ymax =  8; Re = cmd.get("Re", 200);  r0 = 0.5 ; u0 = 1; rho = 1; mu = 2*r0*rho*u0/Re; tf = cmd.get("tf", 200);
#endif

  tf = cmd.get("tf", tf);
  nx = cmd.get("nx", nx);
  ny = cmd.get("ny", ny);
#ifdef P4_TO_P8
  nz = cmd.get("nz", nz);
#endif

  double uniform_band = .5*r0;

#ifdef P4_TO_P8
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny, (zmax-zmin)/(double)nz) / (1<<lmax);
#else
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny) / (1<<lmax);
#endif
  uniform_band = cmd.get("uniform_band", uniform_band);
  uniform_band /= dxmin;

#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi->mpicomm, "Parameters : mu = %g, rho = %g, grid is %dx%dx%d\n", mu, rho, nx, ny, nz); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi->mpicomm, "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%d\n", Re, mu, rho, nx, ny); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi->mpicomm, "n_times_dt = %g, uniform_band = %g\n", n_times_dt, uniform_band);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  for(int repeat=0; repeat<nb_repeat; ++repeat)
  {
    ierr = PetscPrintf(mpi->mpicomm, "#################### REPEAT %d ####################\n", repeat);

    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, &brick, 0, 0, 0);
#else
    connectivity = my_p4est_brick_new(nx, ny, xmin, xmax, ymin, ymax, &brick, 0, 0);
#endif

    p4est_t *p4est_nm1 = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);

    p4est_nm1->user_pointer = (void*)&data;
    for(int l=0; l<lmax; ++l)
    {
      my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
    }

    /* create the initial forest at time nm1 */
    p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

    ierr = PetscPrintf(mpi->mpicomm, "partitioning nm1 done...\n");

    p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

    ierr = PetscPrintf(mpi->mpicomm, "ceating ghost nm1 done...\n");

    ierr = PetscPrintf(mpi->mpicomm, "starting nodes...\n"); CHKERRXX(ierr);
    p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
    ierr = PetscPrintf(mpi->mpicomm, "starting hierarchy...\n"); CHKERRXX(ierr);
    my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick);
    ierr = PetscPrintf(mpi->mpicomm, "starting nodes_neighbors...\n"); CHKERRXX(ierr);
    my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

    ierr = PetscPrintf(mpi->mpicomm, "forest nm1 done...\n");

    /* create the initial forest at time n */
    p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
    p4est_n->user_pointer = (void*)&data;
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);

    ierr = PetscPrintf(mpi->mpicomm, "partitioning n done...\n");

    p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_n, ghost_n);

    ierr = PetscPrintf(mpi->mpicomm, "ceating ghost n done...\n");

    p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
    my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
    my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
    my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);

    ierr = PetscPrintf(mpi->mpicomm, "forest nm1 done...\n");

    my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, &brick, ngbd_c);

    ierr = PetscPrintf(mpi->mpicomm, "ceating faces done...\n");

    Vec phi;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, level_set, phi);

    my_p4est_level_set_t lsn(ngbd_n);
    lsn.reinitialize_1st_order_time_2nd_order_space(phi, 100);
    lsn.perturb_level_set_function(phi, EPS);

#ifdef P4_TO_P8
    CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1 };
    CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn   };
#else
    CF_2 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 };
    CF_2 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   };
#endif

#ifdef P4_TO_P8
    BoundaryConditions3D bc_v[P4EST_DIM];
    BoundaryConditions3D bc_p;
#else
    BoundaryConditions2D bc_v[P4EST_DIM];
    BoundaryConditions2D bc_p;
#endif

    bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
    bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
    bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif
    bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

    bc_v[0].setInterfaceType(DIRICHLET); bc_v[0].setInterfaceValue(bc_interface_value_u);
    bc_v[1].setInterfaceType(DIRICHLET); bc_v[1].setInterfaceValue(bc_interface_value_v);
#ifdef P4_TO_P8
    bc_v[2].setInterfaceType(DIRICHLET); bc_v[2].setInterfaceValue(bc_interface_value_w);
#endif
    bc_p.setInterfaceType(NEUMANN); bc_p.setInterfaceValue(bc_interface_value_p);

#ifdef P4_TO_P8
    CF_3 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v, &external_force_w };
#else
    CF_2 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v };
#endif

    my_p4est_navier_stokes_t ns(ngbd_nm1, ngbd_n, faces_n);
    ns.set_phi(phi);
    ns.set_parameters(mu, rho, uniform_band, threshold_split_cell, n_times_dt);
    ns.set_external_forces(external_forces);
    ns.set_velocities(vnm1, vn);
    ns.set_bc(bc_v, &bc_p);
    ns.set_dt(dxmin*n_times_dt/u0);

    tn = 0;
    dt = ns.get_dt();


    for(int iter=0; iter<nb_iter; ++iter)
    {
      if(iter>0)
      {
        ns.compute_dt();
        dt = ns.get_dt();
        ns.update_from_tn_to_tnp1(&level_set);
				ierr = PetscPrintf(mpi->mpicomm, "update done...\n");
      }

      ns.solve_viscosity();
			ierr = PetscPrintf(mpi->mpicomm, "viscosity done...\n");
      ns.solve_projection();
			ierr = PetscPrintf(mpi->mpicomm, "projection done...\n");
      ns.compute_velocity_at_nodes();
			ierr = PetscPrintf(mpi->mpicomm, "switch to nodes done...\n");

      tn += dt;

      ierr = PetscPrintf(mpi->mpicomm, "Iteration #%04d : tn = %.5f, percent done : %.1f%%, \t max_L2_norm_u = %.5f, \t number of leaves = %d\n", iter, tn, 100*tn/tf, ns.get_max_L2_norm_u(), ns.get_p4est()->global_num_quadrants); CHKERRXX(ierr);
    }
  }

  return 0;
}
