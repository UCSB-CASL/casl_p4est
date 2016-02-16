// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cassert>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

#ifdef P4_TO_P8
static class: public CF_3
{
  public:
  double operator()(double x, double y, double z) const {
    return 2.0*SQR(sin(M_PI*x/2))*sin(2*M_PI*y/2)*sin(2*M_PI*z/2);
  }
} vx_vortex;

static class: public CF_3
{
  public:
  double operator()(double x, double y, double z) const {
    return  -SQR(sin(M_PI*y/2))*sin(2*M_PI*x/2)*sin(2*M_PI*z/2);
  }
} vy_vortex;

static class: public CF_3
{
  public:
  double operator()(double x, double y, double z) const {
    return  -SQR(sin(M_PI*z/2))*sin(2*M_PI*x/2)*sin(2*M_PI*y/2);
  }
} vz_vortex;

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
    return h - MIN(ABS(x-x0) , ABS(y-y0), ABS(z-z0));
  }
private:
  double  x0, y0, z0, h;
};
#else
static class: public CF_2
{
  public:
  double operator()(double x, double y) const {
    return -SQR(sin(M_PI*x/2))*sin(2*M_PI*y/2);
  }
} vx_vortex;

static class: public CF_2
{
  public:
  double operator()(double x, double y) const {
    return  SQR(sin(M_PI*y/2))*sin(2*M_PI*x/2);
  }
} vy_vortex;

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
    return h - MIN(ABS(x-x0) , ABS(y-y0));
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
  cmd.add_option("lmin", "min level in the tree");
  cmd.add_option("lmax", "max level in the tree");
  cmd.parse(argc, argv);

#ifdef P4_TO_P8
  circle phi_cf(0.5, 0.5, 0.5, .3);
#else
  circle phi_cf(0.5, 0.5, .3);
#endif
  splitting_criteria_cf_t data(cmd.get("lmin", 0), cmd.get("lmax", 5), &phi_cf, 1.3);

  parStopWatch w1, w2;
  w1.start("total time");

  // Create the connectivity object
  w2.start("connectivity");
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(2, 2, 2,
                                    0, 2, 0, 2, 0, 2, &brick);
#else
  connectivity = my_p4est_brick_new(2, 2, 0, 2, 0, 2, &brick);
#endif
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
  sample_cf_on_nodes(p4est, nodes, phi_cf, phi);

  double *phi_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

  // write the intial data to disk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, "init",
                         VTK_POINT_DATA, "phi", phi_ptr);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);


  // loop over time
  double tf = 30;
  int tc = 0;
  int save = 10;
  double dt = 0.1;

#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      (void) x;
      (void) y;
      (void) z;
      return 1.0;
    }
  } wall_bc_value;

  struct:WallBC3D{
    BoundaryConditionType operator()(double x, double y, double z) const {
      (void) x;
      (void) y;
      (void) z;
      return DIRICHLET;
    }
  } bc_wall_dirichlet_type;

  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
    }
  } interface_bc_value;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      (void) x;
      (void) y;
      return 1.0;
    }
  } wall_bc_value;

  struct:WallBC2D{
    BoundaryConditionType operator()(double x, double y) const {
      (void)x, (void)y;
      return DIRICHLET;
    }
  } bc_wall_dirichlet_type;

  struct:CF_2{
    double operator()(double x, double y) const {
      return cos(2*M_PI*x)*cos(2*M_PI*y);
    }
  } interface_bc_value;
#endif
  for (double t=0; t<tf; t+=dt, tc++){
    // reinitialize
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    my_p4est_semi_lagrangian_t sl(&p4est, &nodes, &ghost, &node_neighbors);

    // advect the function in time and get the computed time-step
    w2.start("advecting");

#ifdef P4_TO_P8
    const CF_3* vel[] = {&vx_vortex, &vy_vortex, &vz_vortex};
#else
    const CF_2* vel[] = {&vx_vortex, &vy_vortex};
#endif
    sl.update_p4est(vel, dt, phi);

    my_p4est_level_set_t level_set(&node_neighbors);
    level_set.reinitialize_1st_order_time_2nd_order_space(phi, 6);

    w2.stop(); w2.read_duration();

    if (tc % save == 0){
      /* This whole Poisson solve is delibarately done in a stupid way to show that InterpolatingFunction
       * can be used in place of interface value which normally takes a CF_2
       */
      Vec interface_vec;
      ierr = VecDuplicate(phi, &interface_vec); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, interface_bc_value, interface_vec);

      my_p4est_interpolation_nodes_t interface_interp(&node_neighbors);
      interface_interp.set_input(interface_vec, linear);

#ifdef P4_TO_P8
      BoundaryConditions3D bc;
#else
      BoundaryConditions2D bc;
#endif
      bc.setInterfaceType(DIRICHLET);
      bc.setInterfaceValue(interface_interp);
      bc.setWallTypes(bc_wall_dirichlet_type);
      bc.setWallValues(wall_bc_value);

      Vec rhs, sol;
      ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
      ierr = VecSet(rhs, 0); CHKERRXX(ierr);
      ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

      w2.start("Solving a poisson equation");
      my_p4est_poisson_nodes_t solver(&node_neighbors);
      solver.set_phi(phi);
      solver.set_rhs(rhs);
      solver.set_bc(bc);
      solver.solve(sol);
      w2.stop(); w2.read_duration();

      ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      bc.setInterfaceValue(interface_bc_value);
      level_set.extend_Over_Interface(phi, sol, DIRICHLET, sol, 2, 4);

      // Save stuff
      std::ostringstream oss; oss << "p_" << p4est->mpisize << "_"
                                  << brick.nxyztrees[0] << "x"
                                  << brick.nxyztrees[1]
                               #ifdef P4_TO_P8
                                  << "x"
                                  << brick.nxyztrees[2]
                               #endif
                                  << "." << tc/save;

      double *sol_ptr;
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "sol", sol_ptr);

      ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);

      ierr = VecDestroy(interface_vec); CHKERRXX(ierr);
      ierr = VecDestroy(rhs); CHKERRXX(ierr);
      ierr = VecDestroy(sol); CHKERRXX(ierr);
    }
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

