/* 
 * Title: test
 * Description:
 * Author: ftc
 * Date Created: 02-10-2019
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

// casl_p4est Library
#include <src/Parser.h>
#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_surfactant.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_surfactant.h>
#endif


struct r_from_xyz : public CF_2
{
  double operator()(double x, double y) const
  {
    return sqrt(SQR(x)+SQR(y));
  }
} rad;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct phi_from_xyz : public CF_2
{
  double operator()(double x, double y) const
  {
    return atan2(y,x);
  }
} phi;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct level_set_t : public CF_2
{
  double R;
public:
  level_set_t(double R_input, double t_input=0.0, double lip_input=1.2) {R = R_input; t = t_input; lip = lip_input;}
  double operator()(double x, double y) const
  {
    return R - rad(x,y);
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct level_set_band_t : public CF_2
{
  level_set_t* ls_ptr;
  double delta;
public:
  level_set_band_t(level_set_t* ptr, double delta_val) {t = ptr->t; ls_ptr = ptr; delta = delta_val;}
  double operator()(double x, double y) const
  {
    return fabs(ls_ptr->operator ()(x,y)) - delta/2.0;
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct f_t : public CF_2
{
  double operator()(double x, double y) const
  {
    return 1.0 + 0.5*sin(phi(x,y));
  }
} f;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_sol_t : public CF_2
{
  const double* D_s_ptr;
public:
  exact_sol_t(const double* ptr) {D_s_ptr = ptr;}
  double operator()(double x, double y) const
  {
    return 1.0 + (0.5/(1.0+(*D_s_ptr)))*sin(phi(x,y));
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char** argv) {
  
  // Prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Stopwatch
  parStopWatch watch;
  watch.start("Running example: test");

  // Error flag
  PetscErrorCode ierr;

  // The p4est variables
  p4est_t*                   p4est;
  p4est_nodes_t*             nodes;
  p4est_ghost_t*             ghost;
  my_p4est_hierarchy_t*      hierarchy;
  my_p4est_node_neighbors_t* ngbd;
  p4est_connectivity_t*      conn;
  my_p4est_brick_t           brick;

  // Domain size information
  const int n_xyz[]      = {      1,       1};
  const double xyz_min[] = {-PI/2.0, -PI/2.0};
  const double xyz_max[] = { PI/2.0,  PI/2.0};
  const int periodic[]   = {      0,       0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // Levels
  const int lmin = 3;
  const int lmax = 7;
  double dmin = PI/pow(2.0,lmax);

  // Diffusivity
  const double D_s = 1.0;

  // Exact solution
  exact_sol_t* u_exact = new exact_sol_t(&D_s);

  // Create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // Refine based on distance to a level-set and band of uniform cells
  level_set_t* ls = new level_set_t(1.0);
  level_set_band_t* ls_band = new level_set_band_t(ls, 15.0*dmin);

  splitting_criteria_cf_and_uniform_band_t sp(lmin, lmax, ls_band, 2.0, ls->lip);
  p4est->user_pointer = &sp;
  for(int l=0; l<lmax; ++l)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  // Create tree structures
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);
  hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
  ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  // Sample level set functions
  Vec phi, phi_band;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_band); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *ls, phi);
  sample_cf_on_nodes(p4est, nodes, *ls_band, phi_band);

  // Sample right hand side and exact solution on grid
  Vec f_vec, u_exact_vec;
  ierr = VecCreateGhostNodes(p4est, nodes, &f_vec); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, f, f_vec);
  sample_cf_on_nodes(p4est, nodes, *u_exact, u_exact_vec);

  // Poisson solver
  my_p4est_poisson_nodes_mls_t* solver_Gamma;
  solver_Gamma = new my_p4est_poisson_nodes_mls_t(ngbd);
  solver_Gamma->set_diag(1.0);
  solver_Gamma->set_rhs(f_vec);
  solver_Gamma->set_mu(D_s);
  solver_Gamma->add_boundary(MLS_INTERSECTION, phi_band, DIM(NULL, NULL, NULL), NEUMANN, zero_cf, zero_cf);
  solver_Gamma->set_wc(DIRICHLET, zero_cf);
  solver_Gamma->set_integration_order(1);
  solver_Gamma->set_use_sc_scheme(0);

  Vec u_vec;
  ierr = VecCreateGhostNodes(p4est, nodes, &u_vec); CHKERRXX(ierr);
  solver_Gamma->preassemble_linear_system();
  solver_Gamma->solve(u_vec, false, true, KSPBCGS, PCHYPRE);

  // Extrapolate solution
  my_p4est_level_set_t extrap(ngbd);
  extrap.extend_Over_Interface_TVD(phi_band, u_vec, 20, 0);

  // Save the grid into vtk
  unsigned short count_node_scalars = 0;
  const double *phi_p;      ++count_node_scalars;
  const double *phi_band_p; ++count_node_scalars;
  const double *f_p;        ++count_node_scalars;
  const double *u_p;        ++count_node_scalars;
  const double *u_exact_p;        ++count_node_scalars;
  ierr = VecGetArrayRead(phi,         &phi_p  );    CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_band,    &phi_band_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(f_vec,       &f_p);        CHKERRXX(ierr);
  ierr = VecGetArrayRead(u_vec,       &u_p);        CHKERRXX(ierr);
  ierr = VecGetArrayRead(u_exact_vec, &u_exact_p);  CHKERRXX(ierr);
  my_p4est_vtk_write_all_general(p4est, nodes, ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                                 count_node_scalars, 0, 0, 0, 0, 0,
                                 "/home/temprano/Projects/p4est_test/debug/2d/output",
                                 VTK_NODE_SCALAR, "phi",      phi_p,
                                 VTK_NODE_SCALAR, "phi_band", phi_band_p,
                                 VTK_NODE_SCALAR, "f", f_p,
                                 VTK_NODE_SCALAR, "u", u_p,
                                 VTK_NODE_SCALAR, "u_exact", u_exact_p);
  ierr = VecRestoreArrayRead(phi,         &phi_p  );    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_band,    &phi_band_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(f_vec,       &f_p);        CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(u_vec,       &u_p);        CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(u_exact_vec, &u_exact_p);  CHKERRXX(ierr);


  // Destroy the structures
  delete solver_Gamma;

  delete ls;
  delete ls_band;

  delete u_exact;

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  // Stop and read timer
  watch.stop();
  ierr = PetscPrintf(mpi.comm(),"\n"); CHKERRXX(ierr);
  watch.read_duration();

  // Finish
  return 0;
}

