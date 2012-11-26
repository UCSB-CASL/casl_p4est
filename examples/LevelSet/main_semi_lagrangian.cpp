// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/semi_lagrangian.h>
#include <src/interpolating_function.h>
#include <src/refine_coarsen.h>

using namespace std;

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t            *p4est;
  my_p4est_nodes_t   *nodes;
  p4est_locidx_t     *e2n;

  PetscErrorCode ierr;

  struct circle:CF_2{
    circle(double r_): r(r_) {}
    void update (double r_) {r = r_; }
    double operator()(double x, double y) const {
      return r - sqrt(SQR(x-0.5) + SQR(y-0.5));
    }
  private:
    double r;
  };

  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5;
    }
  } vx;

  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5;//y - 0.5;
    }
  } vy;

  circle circ(0.25);
  grid_continous_data_t data = {&circ, 6, 0, 1.3};

  Session session(argc, argv);
  session.init(mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


  // Create the connectivity object
  w2.start("connectivity");
  connectivity = p4est_connectivity_new_brick (2, 2, P4EST_FALSE, P4EST_FALSE);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  // Now refine the tree
  w2.start("refine");
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_continous, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  nodes = my_p4est_nodes_new(p4est);
  e2n = nodes->local_nodes;

  Vec phi;
  ierr = VecGhostCreate_p4est(p4est, nodes, &phi); CHKERRXX(ierr);

  // Initialize level-set function
  double *phi_val;
  ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
    p4est_topidx_t v_pp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 3];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_xmax = connectivity->vertices[3*v_pp + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];
    double tree_ymax = connectivity->vertices[3*v_pp + 1];

    double x = (double)node->x / (double)P4EST_ROOT_LEN; x = x*(tree_xmax-tree_xmin) + tree_xmin;
    double y = (double)node->y / (double)P4EST_ROOT_LEN; y = y*(tree_ymax-tree_ymin) + tree_ymin;

    phi_val[i] = circ(x,y);
  }

  my_p4est_vtk_write_all(p4est, NULL, 1.0,
                         1, 0, "init",
                         VTK_POINT_DATA, "phi", phi_val);

  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Restore temporary objects
  ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);

  SemiLagrangian SL(p4est, nodes);

  double tf  = 4;
  double dt  = 0.05;
  int tc     = 0;
  int save   = 1;
  for (double t = 0; t<=tf; t += dt, ++tc)
  {
    if (tc % 1 == 0){
      // Save stuff
      std::ostringstream oss; oss << "levelset." << tc/save;

      ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, NULL, 1.0,
                             1, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_val);
      ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);
    }

    // Advect the level-set using SL method
    SL.advect(vx, vy, dt, phi);

    // Get a referrence to the level-set function
    ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);

    // Create a new forest based on the level-set
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    grid_discrete_data_t data_np1 = {p4est, nodes, phi_val, data.max_lvl, data.min_lvl, data.lip};
    p4est_np1->user_pointer = (void*)&data_np1;

    // coarsen and refine the grid based on the discrete level-set function
    p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_discrete, NULL);
    p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_discrete, NULL);

    // Partition the new forest
    p4est_partition(p4est_np1, NULL);

    // restore the refference to the level-set value
    ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);

    // interpolate new values of the level-set from the old grid
    BilinearInterpolatingFunction BIF(p4est, nodes, phi);

    my_p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1);
    Vec phi_np1;
    BIF.interpolateValuesToNewForest(p4est_np1, nodes_np1, &phi_np1);

    // Now get rid of previous step objects and reassign referrences for next step
    p4est_destroy(p4est);                    p4est = p4est_np1;
    my_p4est_nodes_destroy(nodes);           nodes = nodes_np1;
    ierr = VecDestroy(&phi); CHKERRXX(ierr); phi   = phi_np1;

    // update the SL internal variables for the next time step
    SL.update(p4est, nodes);
  }

  // Destroy PETSc objects
  ierr = VecDestroy(&phi); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  my_p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  return 0;
}
