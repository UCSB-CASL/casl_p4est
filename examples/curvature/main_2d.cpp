/* 
 *
 * Title: curvature
 * Description: Computes the curvature using two methods. Also illustrates how to use foreach macros
 * Author: Mohammad Mirzadeh
 * Date Created: 12-22-2015
 *
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
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

#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_enviroment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: curvature");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz []      = {1, 1, 1};
  const double xyz_min [] = {-1, -1, -1};
  const double xyz_max [] = { 1,  1,  1};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick);

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

  // create node structure
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);

  Vec phi, kappa[2], normal[P4EST_DIM];
  VecCreateGhostNodes(p4est, nodes, &phi);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  VecDuplicate(phi, &kappa[0]);
  VecDuplicate(phi, &kappa[1]);

  // compute levelset
  sample_cf_on_nodes(p4est, nodes, circle, phi);

  // compute normals (reuturns scaled normal)
  compute_normals(neighbors, phi, normal);

  /* compute curvature with two methods
   * 1) using compact stencil (does not require that normal be scaled)
   * 2) using div(normal) expression (normal MUST be scaled)
   */
  compute_mean_curvature(neighbors, phi, normal, kappa[0]);
  compute_mean_curvature(neighbors, normal, kappa[1]);

  // compute normals
  double *phi_p, *kappa_p[2], *normal_p[P4EST_DIM];
  VecGetArray(phi, &phi_p);
  VecGetArray(kappa[0], &kappa_p[0]);
  VecGetArray(kappa[1], &kappa_p[1]);
  foreach_dimension(dim) VecGetArray(normal[dim], &normal_p[dim]);

  // save vtk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         3+P4EST_DIM, 0, "curvature",
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "curvature_compact", kappa_p[0],
                         VTK_POINT_DATA, "curvature_div_n", kappa_p[1],
                         VTK_POINT_DATA, "normal", normal_p[0],
                         VTK_POINT_DATA, "phi_y", normal_p[1]
#ifdef P4_TO_P8
                       , VTK_POINT_DATA, "phi_z", normal_p[2]
#endif
                         );

  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(kappa[0], &kappa_p[1]);
  VecRestoreArray(kappa[1], &kappa_p[1]);
  foreach_dimension(dim) VecRestoreArray(normal[dim], &normal_p[dim]);

  // destroy vectors
  VecDestroy(phi);
  VecDestroy(kappa[0]);
  VecDestroy(kappa[1]);
  foreach_dimension(dim) VecDestroy(normal[dim]);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

