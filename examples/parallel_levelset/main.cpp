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
#include <src/semi_lagrangian.h>
#include <src/my_p4est_levelset.h>

using namespace std;

static class: public CF_2
{
  public:
  double operator()(double x, double y) const {
    return -0.15*sin(M_PI*x/2)*sin(M_PI*x/2)*sin(2*M_PI*y/2);
  }
} vx_vortex;

static class: public CF_2
{
  public:
  double operator()(double x, double y) const {
    return  0.15*sin(M_PI*y/2)*sin(M_PI*y/2)*sin(2*M_PI*x/2);
  }
} vy_vortex;

static struct:CF_2{
  double operator()(double x, double y) const {
    (void) x;
    (void) y;
    return 0.3;
  }
} vx_translate;

static struct:CF_2{
  double operator()(double x, double y) const {
    (void) x;
    (void) y;
    return 0.3;
  }
} vy_translate;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double  x0, y0, r;
};

double fake_advect(p4est_t **p4est, p4est_nodes_t **nodes, Vec& phi, double t);

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  p4est_ghost_t      *ghost;
  PetscErrorCode ierr;

  circle circ(0.5, 0.5, .3);
  splitting_criteria_cf_t data(0, 8, &circ, 1.3);

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

  // create the ghost layer
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // Initialize the level-set function
  Vec phi;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);  
  sample_cf_on_nodes(p4est, nodes, circ, phi);

  double *phi_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

  // write the intial data to disk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "init",
                         VTK_POINT_DATA, "phi", phi_ptr);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);

  // SemiLagrangian object
  SemiLagrangian sl(&p4est, &nodes, &ghost, &brick);

  // loop over time
  double tf = 100;
  int tc = 0;
  int save = 10;
  vector<double> vx, vy;
  double dt = 0.1;
  for (double t=0; t<tf; t+=dt, tc++){
    if (tc % save == 0){
      // Save stuff
      std::ostringstream oss; oss << "p_" << p4est->mpisize << "_"
                                  << brick.nxytrees[0] << "x"
                                  << brick.nxytrees[1] << "." << tc/save;

      vx.resize(nodes->indep_nodes.elem_count);
      vy.resize(nodes->indep_nodes.elem_count);

      for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
      {
        p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
        p4est_topidx_t tree_id = node->p.piggy3.which_tree;

        p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

        double tree_xmin = connectivity->vertices[3*v_mm + 0];
        double tree_ymin = connectivity->vertices[3*v_mm + 1];

        double x = int2double_coordinate_transform(node->x) + tree_xmin;
        double y = int2double_coordinate_transform(node->y) + tree_ymin;

        vx[i] = vx_vortex(x,y);
        vy[i] = vy_vortex(x,y);
      }

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             3, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "vx", &vx[0],
                             VTK_POINT_DATA, "vy", &vy[0]);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
    }

    // advect the function in time and get the computed time-step
    w2.start("advecting");
    sl.update_p4est_second_order(vx_vortex, vy_vortex, dt, phi);

    // reinitialize
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    my_p4est_level_set level_set(&node_neighbors);
    level_set.reinitialize_2nd_order(phi,1);

    w2.stop(); w2.read_duration();
  }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecDestroy(phi); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_ghost_destroy(ghost);
  p4est_nodes_destroy(nodes);
  p4est_destroy(p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}
