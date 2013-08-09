// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>

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

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  circle circ(1, 1, .3);
  cf_grid_data_t data = {&circ, 8, 3, 1.0};

  Session::init(argc, argv, mpi->mpicomm);

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
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est);

  /* Parallel vector:
   * To save the levelset function, we need a parallel vector. We do this by
   * using PETSc. Here, we just need PETSc's Vec object which is parallel vector
   * To create it, just call 'VecCreateGhost' and pass in p4est, nodes, and the
   * vec object ar arguments. Here we call our vector 'phi_global' to emphasize
   * that it lives across multiple processors.
   */
  Vec phi_global;
  ierr = VecCreateGhost(p4est, nodes, &phi_global); CHKERRXX(ierr);

  /* Computing parallel levelset
   * As the first example, we need to compute the levelset function on the
   * nodes. Now, PETSc is written in pure C so you cannot just do phi_global[i]
   * because C does not understand [] for non-pointer objects. To fix this, we
   * ask PETSc to return a pointer to the actual data. This is done by calling
   * 'VecGetArray' and passing the Vec object and double* pointer.
   *
   * BE CAREFUL: the pointer is literally pointing to the actual data so if you
   * do something silly with it, like change the values it points to by mistake
   * or call free() on it or else, the compiler is not going stop you!
   *
   * NOTE: PETSc will take care of memory management. DO *NOT* FREE THE POINTER
   */
  double *phi;
  ierr = VecGetArray(phi_global, &phi); CHKERRXX(ierr);

  /* Actuall loop:
   * Now that we have the pointer, we need to loop over nodes and compute the
   * levelset. You have two options:
   * 1) You loop over ALL nodes including both local and ghost. This is done as
   * shown below. This generally won't work since you don't know what to do with
   * ghost points ... here, however, its OK since we are just calling a function
   * circle that can be evaluated ANYWHERE. Also, note that if you go with this
   * method, you need to convert the index from p4est to PETSc. This is done by
   * calling p4est2petsc_local_numbering. I'll change it to a map later down the
   * road.
   * 2) The other option you have is to only compute the level set on local
   * nodes and ask PETSc to do the communication for you to find the values of
   * ghost points. This is done for the second method shown below.
   */
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi[p4est2petsc_local_numbering(nodes,i)] = circ(x,y);
  }

  /* Second method:
   * In the second method, we only compute the levelset on local points and then
   * ask PETSc to communicate among processors to update the ghost values. Note
   * that this is really not required here and is redundant, but is shown just
   * to teach you how to do the update when you will need it later on.
   *
   * To do this we first create a duplicate of the old Vec. Note that this does
   * NOT copy data in the old Vec.
   */

  // first create a copy of vector
  circ.update(1.234, 1.4,.34);
  Vec phi_global_copy;
  ierr = VecDuplicate(phi_global, &phi_global_copy); CHKERRXX(ierr);

  // get access to the local's pointer
  double *phi_copy;
  ierr = VecGetArray(phi_global, &phi_copy); CHKERRXX(ierr);

  // do the loop. Note how we only loop over LOCAL nodes
  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
  {
    /* since we want to access the local nodes, we need to 'jump' over intial
     * nonlocal nodes. Number of initial nonlocal nodes is given by
     * nodes->offset_owned_indeps
     */
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    // since this is local, we do not need the mapping
    phi_copy[i] = circ(x,y);
  }

  /* Update ghost points from local:
   * Now that we have calculated the levelset from local nodes, we can ask PETSc
   * to update ghosts from local. This is done by calling 'VecGhostUpdateBegin'
   * and 'VecGhostUpdateEnd' function pairs. You need to pass the Vec object
   * (here phi_global_copy) and two flags. The first one indicates if you want
   * to either add the new values to the old one, or just replace them (or other
   * stuff I do not talk about here). We just need to replace them so we pass
   * INSERT_VALUES.
   * The second flag, asks if you want to update ghost values from local or the
   * reverse process (i.e. update local values from ghost). We want the fist one
   * i.e. we want each processor send its local valid info to other processors
   * sso that they can update their ghost values. For this you use the flag
   * SCATTER_FORWARD. If you want the reverse, meaning you want each processor
   * to update its local values from other's ghosts (used lesss often) you use
   * SCATTER_REVERSE instead.
   */
  ierr = VecGhostUpdateBegin(phi_global_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_global_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // done. lets write both levelset. they MUST be identical when you open them.
  std::ostringstream oss; oss << "partition_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, 1.0,
                         P4EST_TRUE, P4EST_TRUE,
                         2, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi,
                         VTK_POINT_DATA, "phi_copy", phi_copy);

  /* OK, now that we are done with the levelsets, we need to tell that to PETSc
   * so that it can mark its internal data structre. This is good because if
   * after this point you access the pointers by mistake, PETSc is going to
   * throw errors which will be helpful in debugging
   */
  ierr = VecRestoreArray(phi_global, &phi); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_global, &phi_copy); CHKERRXX(ierr);

  // finally, delete PETSc Vecs by calling 'VecDestroy' function
  ierr = VecDestroy(phi_global); CHKERRXX(ierr);
  ierr = VecDestroy(phi_global_copy); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  return 0;
}

