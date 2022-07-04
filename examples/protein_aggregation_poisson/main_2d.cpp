/*
 * Title: protein_aggregation_poisson
 * Description:
 * Author: dbochkov
 * Date Created: 06-27-2022
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
#include <src/casl_math.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/casl_math.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

struct:CF_DIM {
  double operator()(DIM(double x, double y, double z)) const {
    double DIM(x0 = -0.3, y0 = 0, z0 = 0), r0 = 0.25;
    double DIM(x1 =  0.3, y1 = 0, z1 = 0), r1 = 0.25;

    double phi0 = r0 - ABSD(x-x0, y-y0, z-z0);
    double phi1 = r1 - ABSD(x-x1, y-y1, z-z1);

    return MAX(phi0, phi1);
  }
} circles_cf;


struct:CF_DIM {
  double operator()(DIM(double x, double y, double z)) const {
    double DIM(x0 = 0, y0 = 0, z0 = 0), r0 = 0.3;

    double phi0 = r0 - ABSD(x-x0, y-y0, z-z0);

    return phi0 > 0 ? 0 : 100;
  }
} robin_coeff_cf;

int main(int argc, char** argv) {
  
  const char* outdir_vtk = getenv("OUT_DIR_VTK");
  if(!outdir_vtk){
      throw std::invalid_argument("You need to set the environment variable OUT_DIR_VTK to save vtk files\n");
  }
  char output[1000];
  sprintf(output,"%s/protein_aggregation_poisson",outdir_vtk);


  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: protein_aggregation_poisson");

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
  const int periodic[]   = { 1,  1,  1};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set

  splitting_criteria_cf_t sp(6, 8, &circles_cf);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // create hierarchy and neighborhood
  my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
  my_p4est_node_neighbors_t ngbd(&hierarchy,nodes);
  ngbd.init_neighbors();

  // create and sample level set function
  Vec phi;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, circles_cf, phi);

  // create and sample robin coefficient field
  Vec robin_coeff;
  ierr = VecDuplicate(phi, &robin_coeff); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, robin_coeff_cf, robin_coeff);

  // example of how field can be extended from interface
  Vec robin_coeff_flattened;
  ierr = VecDuplicate(phi, &robin_coeff_flattened); CHKERRXX(ierr);

  my_p4est_level_set_t ls(&ngbd);
  ls.extend_from_interface_to_whole_domain_TVD(phi, robin_coeff, robin_coeff_flattened);

  // assuming we store robin coefficient in an array we need to create an interpolating function to pass it to poisson solver
  my_p4est_interpolation_nodes_t robin_coeff_interp(&ngbd);
  robin_coeff_interp.set_input(robin_coeff, linear);

  // create and smaple right hand side of equation
  Vec rhs;
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
  ierr = VecSetGhost(rhs, 1); CHKERRXX(ierr);

  // create array to store solution
  Vec solution;
  ierr = VecDuplicate(phi, &solution); CHKERRXX(ierr);

  // create solver and set parameters
  my_p4est_poisson_nodes_mls_t solver(&ngbd);

  solver.set_use_sc_scheme(0);
  solver.set_integration_order(2);

  solver.add_boundary(MLS_INTERSECTION, phi, NULL, ROBIN, zero_cf, robin_coeff_interp);

  solver.set_mu(1);
  solver.set_rhs(rhs);
  solver.set_diag(0.0);

  solver.set_wc(DIRICHLET, zero_cf); // even though domain is periodic we still need to add wall condition so that solver doesn't argue

  solver.set_use_taylor_correction(1);
  solver.set_kink_treatment(1);
  solver.set_enfornce_diag_scaling(1);

  solver.solve(solution);

  // typically we extend solutions over interface for convenience
  ls.extend_Over_Interface_TVD_Full(phi, solution);

  // save the grid and data into vtk
  double *phi_ptr;
  double *robin_coeff_ptr;
  double *robin_coeff_flattened_ptr;
  double *rhs_ptr;
  double *solution_ptr;

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(robin_coeff, &robin_coeff_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(robin_coeff_flattened, &robin_coeff_flattened_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(rhs, &rhs_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(solution, &solution_ptr); CHKERRXX(ierr);

  
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         5, 0, output,
                         VTK_POINT_DATA, "phi", phi_ptr,
                         VTK_POINT_DATA, "robin_coeff", robin_coeff_ptr,
                         VTK_POINT_DATA, "robin_coeff_flattened", robin_coeff_flattened_ptr,
                         VTK_POINT_DATA, "rhs", rhs_ptr,
                         VTK_POINT_DATA, "solution", solution_ptr);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(robin_coeff, &robin_coeff_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(robin_coeff_flattened, &robin_coeff_flattened_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(solution, &solution_ptr); CHKERRXX(ierr);


  ierr = VecDestroy(solution); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDestroy(robin_coeff_flattened); CHKERRXX(ierr);
  ierr = VecDestroy(robin_coeff); CHKERRXX(ierr);
  ierr = VecDestroy(phi); CHKERRXX(ierr);

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

