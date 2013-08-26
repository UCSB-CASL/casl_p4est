// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// casl_p4est
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <src/bilinear_interpolating_function.h>

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    circle circ(1, 1, .3);
    splitting_criteria_cf_t cf_data   = {&circ, 15, 0, 1};

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
    connectivity = my_p4est_brick_new(2, 2, &brick);
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    p4est->user_pointer = (void*)(&cf_data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    w2.start("computing phi");
    Vec phi;
    ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

      phi_p[p4est2petsc_local_numbering(nodes,i)] = circ(x,y);
    }
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w2.stop(); w2.read_duration();

    std::ostringstream oss; oss << "phi_" << mpi->mpisize;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    // move the circle to create another grid
    cf_data.max_lvl -= 3;
    circ.update(.75, 1.15, .2);

    // Create a new grid
    w2.start("creating/refining/partitioning new p4est");
    p4est_t *p4est_np1 = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    p4est_np1->user_pointer = (void*)&cf_data;
    p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);
    p4est_partition(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    /*
     * Here we create a new nodes structure. Note that in general if what you
     * want is the same procedure as before. This means if the previous grid
     * included ghost cells in the ghost node struture, usually the new one
     * should also include a NEW ghost structure.
     * Here, however, we do not care about this and simply pass NULL to for the
     * new node structure
     */
    w2.start("creating new node data structure");
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    // Create an interpolating function
    BilinearInterpolatingFunction bif(p4est, nodes, ghost, &brick);

    for (p4est_locidx_t i=0; i<nodes_np1->num_owned_indeps; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, i+nodes_np1->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

      // buffer the point
      bif.add_point_to_buffer(i, x, y);
    }
    // set the vector we want to interpolate from
    bif.update_vector(phi);

    // interpolate on to the new vector
    Vec phi_np1;
    ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    bif.interpolate(phi_np1);

    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    oss.str("");
    oss << "phi_np1_" << mpi->mpisize;

    double *phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est_np1, nodes_np1, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_np1_p);
    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(phi);     CHKERRXX(ierr);
    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    p4est_nodes_destroy (nodes_np1);
    p4est_destroy (p4est_np1);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

