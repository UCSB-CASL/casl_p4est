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

// Grid parameters
param_t<int>          nx         (pl, 2,    "nx", "Number of trees in the x direction (default: 2)");
param_t<int>          ny         (pl, 2,    "ny", "Number of trees in the y direction (default: 2)");
#ifdef P4_TO_P8
param_t<int>          nz         (pl, 2,    "nz", "Number of trees in the z direction (default: 2)");
#endif
param_t<unsigned int> lmin       (pl, 3,    "lmin", "Min. level of refinement (default: 3)");
param_t<unsigned int> lmax       (pl, 7,    "lmax", "Max. level of refinement (default: 7)");
param_t<double>       lip        (pl, 1.2,  "lip",  "Lipschitz constant (default: 1.2)");

// Method setup


int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: level_set_advection");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz[]      = { 1,  1,  1};
  const double xyz_min[] = {-1, -1, -1};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 0,  0,  0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } circle;
#endif

  splitting_criteria_cf_t sp(3, 8, &circle);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // save the grid into vtk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, "visualization");

  double tn=0.0;
  double tf=1.1;
  double dt=1.0;//TO BE CHANGED

  while(tn+.1*dt<tf)
  {

    if(tn+dt>tf)
      dt = tf-tn;

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(velo_n[dir]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_n, &velo_n[dir]); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *velo_cf[dir], velo_n[dir]);
    }

    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);
    sl.update_p4est(velo_n, dt, phi_n);
//    sl.update_p4est(velo_cf, dt, phi_n);

    p4est_destroy(p4est); p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
    delete ngbd; ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
    ngbd->init_neighbors();

    my_p4est_level_set_t ls(ngbd);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_n);
//    ls.reinitialize_2nd_order(phi_n);

    if(save_vtk && iter % save_every_n == 0)
      save_VTK(p4est, nodes, &brick, phi_n, iter/save_every_n);

    tn += dt;
    iter++;
  }

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

