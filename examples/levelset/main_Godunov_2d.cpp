// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
  double operator()(double x, double y, double z) const {
    (void) x;
    (void) y;
    (void) z;
    return sqrt(x*x+y*y+z*z);
  }
} vn;

struct circle:CF_3{
  circle(double x0_, double y0_, double z0_, double r_)
    : x0(x0_), y0(y0_), z0(z0_), r(r_)
  {}
  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
private:
  double  x0, y0, z0, r;
};

struct square:CF_3{
  square(double x0_, double y0_, double z0_, double h_)
    : x0(x0_), y0(y0_), z0(z0_), h(h_)
  {}
  void update (double x0_, double y0_, double z0_, double h_) {x0 = x0_; y0 = y0_; z0 = z0_; h = h_; }
  double operator()(double x, double y, double z) const {
    return MIN(h - ABS(x-x0), h - ABS(y-y0), h - ABS(z-z0));
  }
private:
  double  x0, y0, z0, h;
};
#else
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
#endif

int main (int argc, char* argv[]){

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  p4est_ghost_t      *ghost;
  PetscErrorCode ierr;

  cmdParser cmd;
  cmd.add_option("lmin", "min resolution in the tree");
  cmd.add_option("lmax", "max resolution in the tree");
  cmd.parse(argc, argv);

#ifdef P4_TO_P8
  square sq(1, 1, 1, .8);
#else
  square sq(1, 1, .8);
#endif
  splitting_criteria_cf_t data(cmd.get("lmin", 0), cmd.get("lmax", 5), &sq, 1.3);


  parStopWatch w1, w2;
  w1.start("total time");

  // Create the connectivity object
  w2.start("connectivity");
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  int n_xyz [] = {2, 2, 2};
  double xyz_min [] = {0, 0, 0};
  double xyz_max [] = {2, 2, 2};
  int periodic []   = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  // Now refine the tree
  w2.start("refine");
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  w2.stop(); w2.read_duration();

  // Finally re-partition
  w2.start("partition");
  my_p4est_partition(p4est, P4EST_TRUE, NULL);
  w2.stop(); w2.read_duration();

  // create the ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // Initialize the level-set function
  Vec phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
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
  for (double t=0; t<tf; t+=dt, tc++){
    if (tc % save == 0){
      PetscPrintf(p4est->mpicomm, "printing %d\n", tc);
      // Save stuff
      std::ostringstream oss; oss << "normal_direction_" << p4est->mpisize << "_"
                                  << brick.nxyztrees[0] << "x"
                                  << brick.nxyztrees[1]
                               #ifdef P4_TO_P8
                                  << "x" << brick.nxyztrees[2]
                               #endif
                                  << "." << tc/save;

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             1, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr);
      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
    }

    // advect the function in time and get the computed time-step
    w2.start("advecting");

    // reinitialize
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    hierarchy.write_vtk("hierarchy");
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    my_p4est_level_set_t level_set(&node_neighbors);

    dt = level_set.advect_in_normal_direction(vn, phi);
    level_set.reinitialize_1st_order_time_2nd_order_space(phi, 6);

    /* reconstruct the grid */
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_np1->user_pointer = p4est->user_pointer;

    // define interpolating function on the old stuff
    my_p4est_interpolation_nodes_t phi_interp(&node_neighbors);
    phi_interp.set_input(phi, linear);
    data.phi = &phi_interp;

    // refine and coarsen new p4est
    my_p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition
    my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

    // recompute ghost and nodes
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // transfer solution to the new grid
    Vec phi_np1;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; n++)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

      double xyz [] =
      {
        node_x_fr_n(node) + tree_xmin,
        node_y_fr_n(node) + tree_ymin
  #ifdef P4_TO_P8
        ,
        node_z_fr_n(node) + tree_zmin
  #endif
      };

      phi_interp.add_point(n, xyz);
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

