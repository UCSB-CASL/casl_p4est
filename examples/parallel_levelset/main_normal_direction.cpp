// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>

// casl_p4est
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <src/semi_lagrangian.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/interpolating_function.h>

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

static struct:CF_2{
  double operator()(double x, double y) const {
    (void) x;
    (void) y;
    return sqrt(x*x+y*y);
  }
} vn;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double  x0, y0, r;
};

struct square:CF_2{
  square(double x0_, double y0_, double h_): x0(x0_), y0(y0_), h(h_) {}
  void update (double x0_, double y0_, double h_) {x0 = x0_; y0 = y0_; h = h_; }
  double operator()(double x, double y) const {
    return MIN(h - ABS(x-x0), h - ABS(y-y0));
  }
private:
  double  x0, y0, h;
};

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  p4est_ghost_t      *ghost;
  PetscErrorCode ierr;

  square sq(1, 1, .8);
  splitting_criteria_cf_t data(0, 8, &sq, 1.3);

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
  p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  // Now refine the tree
  w2.start("refine");
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  my_p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  // create the ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // Initialize the level-set function
  Vec phi;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, sq, phi);

  double *phi_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

  // write the intial data to disk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "init",
                         VTK_POINT_DATA, "phi", phi_ptr);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);

  // loop over time
  double tf = 0.5;
  int tc = 0;
  int save = 1;
  double dt = 0;
  vector<double> vx, vy;
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

    // reinitialize
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    my_p4est_level_set level_set(&node_neighbors);

    dt = level_set.advect_in_normal_direction(vn, phi);
    level_set.reinitialize_2nd_order(phi, 6);

    /* reconstruct the grid */
    p4est_t *p4est_np1 = p4est_copy(p4est, NULL);
    p4est_np1->user_pointer = p4est->user_pointer;

    // define interpolating function on the old stuff
    InterpolatingFunction phi_interp(p4est, nodes, ghost, &brick, &node_neighbors);
    phi_interp.set_input_parameters(phi, linear);
    data.phi = &phi_interp;

    // refine and coarsen new p4est
    my_p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition
    my_p4est_partition(p4est_np1, NULL);

    // recompute ghost and nodes
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_DEFAULT);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // transfer solution to the new grid
    Vec phi_np1;
    ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; n++)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = node->x != P4EST_ROOT_LEN - 1 ? (double)node->x/(double)P4EST_ROOT_LEN : 1.0;
      double y = node->y != P4EST_ROOT_LEN - 1 ? (double)node->y/(double)P4EST_ROOT_LEN : 1.0;

      x += tree_xmin;
      y += tree_ymin;

      phi_interp.add_point_to_buffer(n, x, y);
    }
    phi_interp.interpolate(phi_np1);

    // get rid of old stuff and replace them with new
    ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
    p4est_destroy(p4est); p4est = p4est_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;

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

