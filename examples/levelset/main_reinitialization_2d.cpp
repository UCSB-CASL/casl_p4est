// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

// casl_p4est
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

#ifdef P4_TO_P8
struct circle:CF_3{
  circle(double x0_, double y0_, double z0_, double r_): x0(x0_), y0(y0_), z0(z0_), r(r_) {}
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
private:
  double x0, y0, z0, r;
};
#else
struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};
#endif

int main (int argc, char* argv[]){

  try{
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lmin", "min level for refinement");
    cmd.add_option("lmax", "max level for refinement");
    cmd.parse(argc, argv);

#ifdef P4_TO_P8
    circle circ(0, 0, 0, 0.3);
#else
    circle circ(0, 0, 0.3);
#endif
    splitting_criteria_cf_t cf_circle(cmd.get("lmin", 3), cmd.get("lmax", 6), &circ);

    parStopWatch w;
    w.start("total time");

    /* Create the connectivity object */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
    int n_xyz [] = {1, 1, 1};
    double xyz_min [] = {-1, -1, -1};
    double xyz_max [] = { 1,  1,  1};
    int periodic []   = {0, 0, 0};

    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    /* Now create the forest */
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    /* refine the forest using a refinement criteria */
    p4est->user_pointer = (void*)(&cf_circle);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* Finally re-partition */
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    /* Create the ghost structure */
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate the node data structure */
    nodes = my_p4est_nodes_new(p4est, ghost);

    /* initialize the neighbor nodes structure */
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(i, p4est, nodes, xyz);

#ifdef P4_TO_P8
      phi_p[i] = circ(xyz[0], xyz[1], xyz[2]);
#else
      phi_p[i] = circ(xyz[0], xyz[1]);
#endif
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_level_set_t ls(&node_neighbors);
    ls.reinitialize_2nd_order( phi, 100 );

    std::ostringstream oss; oss << "phi_" << mpi.size() << "_" << P4EST_DIM;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    my_p4est_vtk_write_ghost_layer(p4est, ghost);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    /* finally, delete PETSc Vecs by calling 'VecDestroy' function */
    ierr = VecDestroy(phi);     CHKERRXX(ierr);

    /* destroy the p4est and its connectivity structure */
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    p4est_connectivity_destroy (connectivity);

    w.stop(); w.read_duration();
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

