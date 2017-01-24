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
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

#ifdef P4_TO_P8
typedef CF_3 cf_t;
#else
typedef CF_2 cf_t;
#endif

double compute_curvature_err(my_p4est_node_neighbors_t& neighbors, Vec phi, Vec kappa, const cf_t &curvature, Vec error);

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser parser;
  parser.add_option("lmin", "min level in the tree");
  parser.add_option("lmax", "max level in the tree");
  parser.add_option("nsp", "number of splits");
  parser.parse(argc, argv);

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
  const int periodic []   = {0, 0, 0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } interface;

  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 2.0/sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } curvature;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } interface;

  struct:CF_2{
    double operator()(double x, double y) const {
      return 1.0/sqrt(SQR(x) + SQR(y));
    }
  } curvature;
#endif

  const int lmin = parser.get("lmin", 1);
  const int lmax = parser.get("lmax", 3);
  const int nsp  = parser.get("nsp", 8);

  double err[2][nsp+1];
  char filename[FILENAME_MAX];
  for (int s = 0; s < nsp; s++) {
    // create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

    // refine
    splitting_criteria_cf_t sp(lmin+s, lmax+s, &interface, 2.0);
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
    neighbors.init_neighbors();

    Vec phi, error, kappa, normal[P4EST_DIM];
    VecCreateGhostNodes(p4est, nodes, &phi);
    foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
    VecDuplicate(phi, &kappa);
    VecDuplicate(phi, &error);

    // compute levelset
    double *phi_p, *error_p;
    VecGetArray (phi,  &phi_p);
    VecGetArray (error,  &error_p);

    sample_cf_on_nodes(p4est, nodes, interface, phi);

    sprintf(filename, "before_%d.%d", P4EST_DIM, s);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, filename,
                           VTK_POINT_DATA, "phi", phi_p);

    // reinitialize
    my_p4est_level_set_t ls(&neighbors);
    ls.reinitialize_2nd_order(phi, 10);

    // compute normals (reuturns scaled normal)
    compute_normals(neighbors, phi, normal);

    sprintf(filename, "after_%d.%d", P4EST_DIM, s);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, filename,
                           VTK_POINT_DATA, "phi", phi_p);


    /* compute curvature with two methods
     * 1) using compact stencil (does not require that normal be scaled)
     * 2) using div(normal) expression (normal MUST be scaled)
     */
    compute_mean_curvature(neighbors, phi, normal, kappa);
    err[0][s+1] = compute_curvature_err(neighbors, phi, kappa, curvature, error);

    sprintf(filename, "error1_%d.%d", P4EST_DIM, s);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, filename,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "err", error_p);

    compute_mean_curvature(neighbors, normal, kappa);
    err[1][s+1] = compute_curvature_err(neighbors, phi, kappa, curvature, error);

    sprintf(filename, "error2_%d.%d", P4EST_DIM, s);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, filename,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "err", error_p);

    if (mpi.rank() == 0) {
      PetscPrintf(mpi.comm(), "Resolution: (%d,%d)\n", lmin+s, lmax+s);
      PetscPrintf(mpi.comm(), "Compact: err = %e, order = %f\n", err[0][s+1], log2(err[0][s]/err[0][s+1]));
      PetscPrintf(mpi.comm(), "div(n):  err = %e, order = %f\n", err[1][s+1], log2(err[1][s]/err[1][s+1]));
      PetscPrintf(mpi.comm(), "\n");
    }

    // destroy vectors
    VecDestroy(phi);
    VecDestroy(kappa);
    VecDestroy(error);
    foreach_dimension(dim) VecDestroy(normal[dim]);

    // destroy the structures
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(conn, &brick);
  w.stop(); w.read_duration();
}

double compute_curvature_err(my_p4est_node_neighbors_t& neighbors, Vec phi, Vec kappa, const cf_t &curvature, Vec error) {
  double *kappa_p, *phi_p, *error_p;
  VecGetArray(kappa, &kappa_p);
  VecGetArray(phi, &phi_p);
  VecGetArray(error, &error_p);

  double diag_min = p4est_diag_min(neighbors.get_p4est());
  double err = 0;
  int nmax = 0;
  double x[P4EST_DIM];
  node_xyz_fr_n(0, neighbors.get_p4est(), neighbors.get_nodes(), x);
  foreach_node(n, neighbors.get_nodes()) {
    if (fabs(phi_p[n]) < diag_min){
      node_xyz_fr_n(n, neighbors.get_p4est(), neighbors.get_nodes(), x);
#ifdef P4_TO_P8
      double k = curvature(x[0], x[1], x[2]);
#else
      double k = curvature(x[0], x[1]);
#endif
      error_p[n] = fabs(kappa_p[n] - k);
      if (error_p[n] > err) {
        err = error_p[n];
        nmax = n;
      }
    }
  }

  VecRestoreArray(kappa, &kappa_p);
  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(error, &error_p);

  return err;
}

