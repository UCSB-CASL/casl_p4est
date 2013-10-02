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

1PetscErrorCode VecCreateGhostP4estNumbering(p4est_t *p4est, p4est_nodes_t *nodes, Vec *v)
{
  PetscErrorCode ierr = 0;
  p4est_locidx_t num_local = nodes->num_owned_indeps;

  std::vector<PetscInt> indecies(nodes->indep_nodes.elem_count, 0);
  std::vector<PetscInt> global_offset_sum(p4est->mpisize + 1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_offset_sum[r+1] = global_offset_sum[r] + (PetscInt)nodes->global_owned_indeps[r];

  PetscInt num_global = global_offset_sum[p4est->mpisize];

  /* compute the global index of all nodes */
  // First patch of ghost nodes
  for (p4est_locidx_t i = 0; i<nodes->offset_owned_indeps; ++i)
  {
    p4est_indep_t *ni  = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    indecies[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i]];
  }
  // local nodes
  for (p4est_locidx_t i = nodes->offset_owned_indeps; i<nodes->offset_owned_indeps+num_local; ++i)
  {
    p4est_indep_t *ni  = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    indecies[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[p4est->mpirank];
  }
  // second patch of ghost nodes
  for (size_t i = nodes->offset_owned_indeps+num_local; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t* ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    indecies[i] = (PetscInt)ni->p.piggy3.local_num + global_offset_sum[nodes->nonlocal_ranks[i-num_local]];
  }

  // Create the mapping object
  ISLocalToGlobalMapping mapping;
  ISLocalToGlobalMappingCreate(p4est->mpicomm, indecies.size(), (const PetscInt*)&indecies[0], PETSC_COPY_VALUES, &mapping); CHKERRQ(ierr);

  // create the ghosted object
  std::vector<PetscInt> ghost_nodes(nodes->indep_nodes.elem_count - num_local, 0); // we dont care about actual ghost index as we will set that later ourself
  ierr = VecCreateGhost(p4est->mpicomm, num_local, num_global, ghost_nodes.size(), (const PetscInt*)&ghost_nodes[0], v); CHKERRQ(ierr);

  // Set the vector local2global mapping
  ierr = VecSetLocalToGlobalMapping(*v, mapping); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v); CHKERRQ(ierr);

  // delete the mapping
  ierr = ISLocalToGlobalMappingDestroy(mapping); CHKERRQ(ierr);

  return ierr;
}

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  circle circ(1, 1, .3);
  splitting_criteria_cf_t data(0, 10, &circ, 1);

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
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  // generate the ghost data-structure
  w2.start("generating ghost data structure");
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
  w2.stop(); w2.read_duration();

  // generate the node data structure
  w2.start("creating nodes data structure");
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

  /* Parallel vector:
   * To save the levelset function, we need a parallel vector. We do this by
   * using PETSc. Here, we just need PETSc's Vec object which is parallel vector
   * To create it, just call 'VecCreateGhost' and pass in p4est, nodes, and the
   * vec object ar arguments. Here we call our vector 'phi_global' to emphasize
   * that it lives across multiple processors.
   */
  w2.start("creating Ghosted vector");
  Vec phi_global;
  ierr = VecCreateGhost(p4est, nodes, &phi_global); CHKERRXX(ierr);

  Vec phi_p4est;
  ierr = VecCreateGhostP4estNumbering(p4est, nodes, &phi_p4est); CHKERRXX(ierr);
  w2.stop(); w2.read_duration();

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
  double *phi, *phi_p4;
  ierr = VecGetArray(phi_global, &phi); CHKERRXX(ierr);
  ierr = VecGetArray(phi_p4est, &phi_p4); CHKERRXX(ierr);

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
  w2.start("setting phi values");
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
    phi_p4[i] = circ(x,y);
  }
  w2.stop(); w2.read_duration();

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
  Vec phi_global_copy;
  ierr = VecDuplicate(phi_global, &phi_global_copy); CHKERRXX(ierr);

  // get access to the local's pointer
  double *phi_copy;
  ierr = VecGetArray(phi_global_copy, &phi_copy); CHKERRXX(ierr);

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
//  my_p4est_vtk_write_all(p4est, nodes, ghost,
//                         P4EST_TRUE, P4EST_TRUE,
//                         2, 0, oss.str().c_str(),
//                         VTK_POINT_DATA, "phi", phi,
//                         VTK_POINT_DATA, "phi_copy", phi_copy);

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p4);

  /* OK, now that we are done with the levelsets, we need to tell that to PETSc
   * so that it can mark its internal data structre. This is good because if
   * after this point you access the pointers by mistake, PETSc is going to
   * throw errors which will be helpful in debugging
   */
  ierr = VecRestoreArray(phi_global, &phi); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_p4est, &phi_p4); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_global_copy, &phi_copy); CHKERRXX(ierr);

  // finally, delete PETSc Vecs by calling 'VecDestroy' function
  ierr = VecDestroy(phi_global); CHKERRXX(ierr);
  ierr = VecDestroy(phi_p4est); CHKERRXX(ierr);
  ierr = VecDestroy(phi_global_copy); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

