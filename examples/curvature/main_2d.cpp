/* 
 *
 * Title: curvature
 * Description: Computes the curvature using different methods. Also illustrates how to use some macros
 * Author: Mohammad Mirzadeh, revised and augmented by Raphael Egan
 * Date Created: 12-22-2015, revised and augmented on 08-03-2020
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

double compute_curvature_err(my_p4est_node_neighbors_t& neighbors, Vec phi, Vec kappa, const CF_DIM &curvature, Vec error)
{
  PetscErrorCode ierr;
  double *kappa_p, *phi_p, *error_p;
  ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p);     CHKERRXX(ierr);
  ierr = VecGetArray(error, &error_p); CHKERRXX(ierr);

  double diag_min = p4est_diag_min(neighbors.get_p4est());
  double err = 0;
  double x[P4EST_DIM];
  foreach_node(n, neighbors.get_nodes()) {
    if (fabs(phi_p[n]) < diag_min){
      node_xyz_fr_n(n, neighbors.get_p4est(), neighbors.get_nodes(), x);
      const double k = curvature(DIM(x[0], x[1], x[2]));
      error_p[n] = fabs(kappa_p[n] - k);
      if (error_p[n] > err)
        err = error_p[n];
    }
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, neighbors.get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(error, &error_p); CHKERRXX(ierr);

  return err;
}

int main(int argc, char** argv) {
  
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser parser;
  parser.add_option("lmin", "min level in the tree");
  parser.add_option("lmax", "max level in the tree");
  parser.add_option("nsp", "number of splits");
  parser.add_option("vtk", "activates exportation of visualization files");
  parser.add_option("block", "flag activating the execution with block vectors (for grad phi and the second derivatives of phi)");
  parser.add_option("xxyyzz", "flag activating the precalculation of second derivatives for compact calculation)");

  if(parser.parse(argc, argv))
    return 1;

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
  const int n_xyz [P4EST_DIM]      = {DIM(1, 1, 1)};
  const double xyz_min [P4EST_DIM] = {DIM(-1, -1, -1)};
  const double xyz_max [P4EST_DIM] = {DIM( 1,  1,  1)};
  const int periodic [P4EST_DIM]   = {DIM(0, 0, 0)};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // refine based on distance to a level-set
  struct : CF_DIM {
    double operator()(DIM(double x, double y, double z)) const {
      return 0.5 - sqrt(SUMD(SQR(x), SQR(y), SQR(z)));
    }
  } interface;

  struct : CF_DIM{
    double operator()(DIM(double x, double y, double z)) const {
      return -((double) (P4EST_DIM - 1))/sqrt(SUMD(SQR(x), SQR(y), SQR(z)));
    }
  } curvature;

  const int lmin = parser.get<int>("lmin", 1);
  const int lmax = parser.get<int>("lmax", 3);
  const int nsp  = parser.get<int>("nsp", 8);
  const bool block        = parser.get<bool>("block", true);
  const bool with_xxyyzz  = parser.get<bool>("xxyyzz", true);
  const bool save_vtk = parser.get<bool>("vtk", false);

  double err[2][nsp + 1];
  char filename[FILENAME_MAX];
  PetscErrorCode ierr;
  for (int s = 0; s < nsp; s++) {
    // create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

    // refine
    splitting_criteria_cf_t sp(lmin + s, lmax + s, &interface, 2.0);
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

    Vec phi, error, kappa;
    Vec normal[P4EST_DIM];
    Vec *grad_phi = NULL, grad_phi_block = NULL;
    Vec *phi_xxyyzz = NULL, phi_xxyyzz_block = NULL;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &error); CHKERRXX(ierr);
    foreach_dimension(dim) {
      ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr);
    }
    if(block){
      ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &grad_phi_block); CHKERRXX(ierr);
      if(with_xxyyzz){
        ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &phi_xxyyzz_block); CHKERRXX(ierr); }
    }
    else
    {
      grad_phi = new Vec[P4EST_DIM];
      if(with_xxyyzz)
        phi_xxyyzz = new Vec[P4EST_DIM];
      foreach_dimension(dim) {
        ierr = VecCreateGhostNodes(p4est, nodes, &grad_phi[dim]); CHKERRXX(ierr);
        if(with_xxyyzz){
          ierr = VecCreateGhostNodes(p4est, nodes, &phi_xxyyzz[dim]); CHKERRXX(ierr); }
      }
    }

    // compute levelset
    double *phi_p, *error_p;
    ierr = VecGetArray (phi,  &phi_p);     CHKERRXX(ierr);
    ierr = VecGetArray (error,  &error_p); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, interface, phi);

    if(save_vtk)
    {
      sprintf(filename, "before_%d.%d", P4EST_DIM, s);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             1, 0, filename,
                             VTK_POINT_DATA, "phi", phi_p);
    }

    // reinitialize
    my_p4est_level_set_t ls(&neighbors);
    ls.reinitialize_2nd_order(phi);

    // compute grad phi
    if(block)
      neighbors.first_derivatives_central(phi, grad_phi_block);
    else
      neighbors.first_derivatives_central(phi, grad_phi);
    if(with_xxyyzz)
    {
      if(block)
        neighbors.second_derivatives_central(phi, phi_xxyyzz_block);
      else
        neighbors.second_derivatives_central(phi, phi_xxyyzz);
    }
    // compute normals (component by componet)
    compute_normals(neighbors, phi, normal);
    if(save_vtk)
    {
      sprintf(filename, "after_%d.%d", P4EST_DIM, s);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             1, 0, filename,
                             VTK_POINT_DATA, "phi", phi_p);
    }

    /* compute curvature with two methods
     * 1) using compact stencil (does not require that normal be scaled)
     * 2) using div(normal) expression (normal MUST be scaled)
     */
    compute_mean_curvature(neighbors, phi, grad_phi, grad_phi_block, phi_xxyyzz, phi_xxyyzz_block, kappa);
    err[0][s+1] = compute_curvature_err(neighbors, phi, kappa, curvature, error);

    if(save_vtk)
    {
      sprintf(filename, "error1_%d.%d", P4EST_DIM, s);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, filename,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "err", error_p);
    }


    compute_mean_curvature(neighbors, normal, kappa);
    err[1][s+1] = compute_curvature_err(neighbors, phi, kappa, curvature, error);

    if(save_vtk)
    {
      sprintf(filename, "error2_%d.%d", P4EST_DIM, s);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, filename,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "err", error_p);
    }

    ierr = PetscPrintf(mpi.comm(), "Resolution: (%d,%d)\n", lmin+s, lmax+s); CHKERRXX(ierr);
    if(s > 0)
    {
      ierr = PetscPrintf(mpi.comm(), "Compact:\t err = %e, order = %f\n", err[0][s + 1], log2(err[0][s]/err[0][s + 1])); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "div(n): \t err = %e, order = %f\n", err[1][s + 1], log2(err[1][s]/err[1][s + 1])); CHKERRXX(ierr);
    }
    else
    {
      ierr = PetscPrintf(mpi.comm(), "Compact:\t err = %e\n", err[0][s + 1]); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "div(n): \t err = %e\n", err[1][s + 1]); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "\n"); CHKERRXX(ierr);

    // destroy vectors
    ierr = VecDestroy(phi);               CHKERRXX(ierr);
    ierr = VecDestroy(kappa);             CHKERRXX(ierr);
    ierr = VecDestroy(error);             CHKERRXX(ierr);
    if(grad_phi_block != NULL){
      ierr = VecDestroy(grad_phi_block);    CHKERRXX(ierr); }
    if(phi_xxyyzz_block != NULL){
      ierr = VecDestroy(phi_xxyyzz_block);  CHKERRXX(ierr); }
    foreach_dimension(dim) {
      ierr = VecDestroy(normal[dim]);     CHKERRXX(ierr);
      if(grad_phi != NULL){
        ierr = VecDestroy(grad_phi[dim]);   CHKERRXX(ierr); }
      if(phi_xxyyzz != NULL){
        ierr = VecDestroy(phi_xxyyzz[dim]); CHKERRXX(ierr); }
    }

    if(grad_phi != NULL)
      delete [] grad_phi;
    if(phi_xxyyzz != NULL)
      delete [] phi_xxyyzz;

    // destroy the structures
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(conn, &brick);
  w.stop(); w.read_duration();
}

