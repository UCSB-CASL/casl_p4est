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

#include <src/petsc_compatibility.h>

using namespace std;

class: public CF_2
{
public:
    double operator()(double x, double y) const {
        return -sin(M_PI*x)*sin(M_PI*x)*sin(2*M_PI*y);
    }
} vx_vortex;

class: public CF_2
{
public:
    double operator()(double x, double y) const {
        return  sin(M_PI*y)*sin(M_PI*y)*sin(2*M_PI*x);
    }
} vy_vortex;

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t            *p4est;
  my_p4est_nodes_t   *nodes;
  p4est_locidx_t     *e2n;

  PetscErrorCode ierr;

  struct circle:CF_2{
    circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
    void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
    double operator()(double x, double y) const {
      return r - sqrt(SQR(x-x0) + SQR(y-y0));
    }
  private:
    double r, x0, y0;
  };

  const static double vx_max = 0.15;
  const static double vy_max = 0.15;

  struct:CF_2{
    double operator()(double x, double y) const {
      return vx_max;
    }
  } vx;

  struct:CF_2{
    double operator()(double x, double y) const {
      return vy_max;
    }
  } vy;

  circle circ(0.25, 0.25, .15);
  refine_coarsen_data_t data = {&circ, 6, 0, 1.0};

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
  p4est_refine(p4est, P4EST_TRUE, refine_levelset, NULL);
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

  // calculate d_min in all trees
  double dx_min = 1000, dy_min = 1000;
  for (p4est_topidx_t tr_it = 0; tr_it<connectivity->num_trees; ++tr_it)
  {
    p4est_topidx_t vmm = connectivity->tree_to_vertex[tr_it*P4EST_CHILDREN];
    p4est_topidx_t vpp = connectivity->tree_to_vertex[(tr_it+1)*P4EST_CHILDREN - 1];

    double dx = connectivity->vertices[3*vpp + 0] - connectivity->vertices[3*vmm + 0]; dx /= (double)(1 << data.max_lvl);
    double dy = connectivity->vertices[3*vpp + 1] - connectivity->vertices[3*vmm + 1]; dy /= (double)(1 << data.max_lvl);

    dx_min = MIN(dx_min, dx);
    dy_min = MIN(dy_min, dy);
  }

  double tf  = 10;
  double dt_min = 0.1;
  double dt = MIN(dt_min, MIN(dx_min/vx_max, dy_min/vy_max));
  int tc     = 0;
  int save   = 1;
  for (double t = 0; t<=tf; t += dt, ++tc)
  {
    if (tc % save == 0){
      // Save stuff
      std::ostringstream oss; oss << "translate." << tc/save;

      ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, NULL, 1.0,
                             1, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_val);
      ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);
    }

    // Advect the level-set using SL method
    SL.advect(vx, vy, dt, phi);

    // Define an interpolating function
    BilinearInterpolatingFunction BIF(p4est, nodes, phi);

    // Create a new forest based on the level-set
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    data.phi = &BIF;
    p4est_np1->user_pointer = (void*)&data;

    // coarsen and refine the grid based on the discrete level-set function
    p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset, NULL);
    p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset, NULL);

    // Partition the new forest
    p4est_partition(p4est_np1, NULL);

    // interpolate new values of the level-set from the old grid
    my_p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1);
    Vec phi_np1;
    BIF.interpolateValuesToNewForest(p4est_np1, nodes_np1, &phi_np1);

    // Now get rid of previous step objects and reassign referrences for next step
    p4est_destroy(p4est);                    p4est = p4est_np1;
    my_p4est_nodes_destroy(nodes);           nodes = nodes_np1;
    ierr = VecDestroy(phi); CHKERRXX(ierr);  phi   = phi_np1;

    // update the SL internal variables for the next time step
    SL.update(p4est, nodes);
  }

  // Destroy PETSc objects
  ierr = VecDestroy(phi); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  my_p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  p4est_connectivity_destroy (connectivity);

  w1.stop(); w1.read_duration();

  return 0;
}
