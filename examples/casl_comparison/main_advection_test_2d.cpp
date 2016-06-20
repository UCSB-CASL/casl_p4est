/*
 * Title: casl_test
 * Description:
 * Author: Mohammad Mirzadeh
 * Date Created: 01-11-2016
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

#ifdef P4_TO_P8
typedef CF_3 cf_t;
#else
typedef CF_2 cf_t;
#endif

// p4est variables
p4est_connectivity_t* conn;
my_p4est_brick_t      brick;

void advect_using_semilagrangian(const cf_t** vel, cf_t& interface, double cfl, int iter);
void advect_using_normal(const cf_t** vel, cf_t& interface, double cfl, int iter);
void advect_using_upwind(const cf_t** vel, cf_t& interface, double cfl, int iter);
void save_vtk(p4est_t* p4est, p4est_nodes_t* nodes, p4est_ghost_t* ghost, Vec phi, const char* vtkname);

using namespace std;

int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser parser;
  parser.add_option("cfl", "the cfl number");
  parser.add_option("iter", "number of iterations to perform");
  parser.parse(argc, argv);

  const double cfl = parser.get("cfl", 1.0);
  const int iter = parser.get("iter", 20);

  // stopwatch
  parStopWatch w;
  w.start("Running example: casl_test");

  // domain size information
  const int n_xyz []      = {1, 1, 1};
  const double xyz_min [] = {0, 0, 0};
  const double xyz_max [] = {1, 1, 1};
  const int periodic []   = {0, 0, 0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.15 - sqrt(SQR(x-0.25) + SQR(y-0.25) + SQR(z-0.25));
    }
  } interface;

  struct:CF_3{
    double operator() (double x, double y, double z) const {
      return 2.0*SQR(sin(M_PI*x/2))*sin(2*M_PI*y/2)*sin(2*M_PI*z/2);
    }
  } ux;

  struct:CF_3{
    double operator() (double x, double y, double z) const {
      return -SQR(sin(M_PI*y/2))*sin(2*M_PI*x/2)*sin(2*M_PI*z/2);
    }
  } uy;

  struct:CF_3{
    double operator() (double x, double y, double z) const {
      return -SQR(sin(M_PI*z/2))*sin(2*M_PI*x/2)*sin(2*M_PI*y/2);
    }
  } uz;

  CF_3* vel[] = {&ux, &uy, &uz};
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.15 - sqrt(SQR(x-0.25) + SQR(y-0.25));
    }
  } interface;

  struct:CF_2{
    double operator() (double x, double y) const {
      return -SQR(sin(M_PI*x/2))*sin(2*M_PI*y/2);
    }
  } ux;

  struct:CF_2{
    double operator() (double x, double y) const {
      return SQR(sin(M_PI*y/2))*sin(2*M_PI*x/2);
    }
  } uy;

  const CF_2* vel[] = {&ux, &uy};
#endif

  // run the advection
  w.start("advecting using semi-lagrangian");
  advect_using_semilagrangian(vel, interface, cfl, iter);
  w.stop(); w.read_duration();

  w.start("advection using motion in normal direction");
//  advect_using_normal(vel, interface, iter);
  w.stop(); w.read_duration();

  w.start("advecting using upwind scheme");
  advect_using_upwind(vel, interface, cfl, iter);
  w.stop(); w.read_duration();

  // destroy the structures
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

void advect_using_semilagrangian(const cf_t** vel, cf_t &interface, double cfl, int iter)
{
  p4est_t *p4est = my_p4est_new(MPI_COMM_WORLD, conn, 0, NULL, NULL);

  splitting_criteria_cf_t sp(3, 8, &interface);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  p4est_ghost_t* ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  p4est_nodes_t* nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t h(p4est, ghost, &brick);
  my_p4est_node_neighbors_t ngbh(&h, nodes);
  ngbh.init_neighbors();

  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
#ifdef P4_TO_P8
  double dt = cfl*MIN(dx[0], dx[1], dx[2]);
#else
  double dt = cfl*MIN(dx[0], dx[1]);
#endif

  Vec phi;
  VecCreateGhostNodes(p4est, nodes, &phi);
  sample_cf_on_nodes(p4est, nodes, interface, phi);

  const char* filename = "semilagrangian";
  char vtkname[FILENAME_MAX];
  sprintf(vtkname, "%s.%04d", filename, 0);

  save_vtk(p4est, nodes, ghost, phi, vtkname);
  my_p4est_semi_lagrangian_t sl(&p4est, &nodes, &ghost, &ngbh);
  for (int i=0; i<iter; i++){
    sl.update_p4est(vel, dt, phi);

    // reinitialize
    my_p4est_level_set_t ls(&ngbh);
    ls.reinitialize_2nd_order(phi);

    my_p4est_interpolation_nodes_t phi_i(&ngbh);
    phi_i.set_input(phi, quadratic_non_oscillatory);

    if (p4est->mpirank == 0)  {
      p4est_gloidx_t num_nodes = 0;
      for (int r=0; r<p4est->mpisize; r++)
      num_nodes += nodes->global_owned_indeps[r];
      printf("i = %4d, t = %1.5f n = %5lu\n", i, i*dt, num_nodes);
    }

    sprintf(vtkname, "%s.%04d", filename, i+1);
    save_vtk(p4est, nodes, ghost, phi, vtkname);
  }

  VecDestroy(phi);

  p4est_destroy(p4est);
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
}

double upwind_step(my_p4est_node_neighbors_t& neighbors, p4est_locidx_t n, const double *f, const double* const fxx[], const cf_t **vel, double dt)
{
  quad_neighbor_nodes_of_node_t qnnn;
  neighbors.get_neighbors(n, qnnn);

  double x[P4EST_DIM];
  node_xyz_fr_n(n, neighbors.get_p4est(), neighbors.get_nodes(), x);

#ifdef P4_TO_P8
  double ux = (*vel[0])(x[0], x[1], x[2]);
  double uy = (*vel[1])(x[0], x[1], x[2]);
  double uz = (*vel[2])(x[0], x[1], x[2]);
#else
  double ux = (*vel[0])(x[0], x[1]);
  double uy = (*vel[1])(x[0], x[1]);
#endif

  double fx = ux > 0 ? qnnn.dx_backward_quadratic(f, fxx[0]) : qnnn.dx_forward_quadratic(f, fxx[0]);
  double fy = uy > 0 ? qnnn.dy_backward_quadratic(f, fxx[1]) : qnnn.dy_forward_quadratic(f, fxx[1]);
#ifdef P4_TO_P8
  double fz = uz > 0 ? qnnn.dz_backward_quadratic(f, fxx[2]) : qnnn.dz_forward_quadratic(f, fxx[2]);
#endif

#ifdef P4_TO_P8
  return f[n] - dt*(ux*fx+uy*fy+uz*fz);
#else
  return f[n] - dt*(ux*fx+uy*fy);
#endif
}

void update(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_node_neighbors_t& neighbors, Vec &phi)
{
  p4est_t* p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  splitting_criteria_cf_t *sp = (splitting_criteria_cf_t*)p4est->user_pointer;
  p4est_np1->user_pointer = p4est->user_pointer;
  p4est_np1->connectivity = p4est->connectivity;

  double *phi_p;
  VecGetArray(phi, &phi_p);

  splitting_criteria_tag_t tag(sp->min_lvl, sp->max_lvl, sp->lip);
  tag.refine_and_coarsen(p4est_np1, nodes, phi_p);
  my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_interpolation_nodes_t interp(&neighbors);
  interp.set_input(phi, quadratic_non_oscillatory);

  Vec phi_np1;
  VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1);
  double *phi_np1_p;
  VecGetArray(phi_np1, &phi_np1_p);

  foreach_node(n, nodes_np1) {
    double x[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    interp.add_point(n, x);
  }
  interp.interpolate(phi_np1_p);

  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(phi_np1, &phi_np1_p);

  // swap pointers and get rid of old stuff
  VecDestroy(phi); phi = phi_np1;
  p4est_destroy(p4est); p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;

  neighbors.update(p4est, ghost, nodes);
}

void advect_using_upwind(const cf_t** vel, cf_t &interface, double cfl, int iter)
{
  p4est_t *p4est = my_p4est_new(MPI_COMM_WORLD, conn, 0, NULL, NULL);

  splitting_criteria_cf_t sp(3, 8, &interface);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  p4est_ghost_t* ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  p4est_nodes_t* nodes = my_p4est_nodes_new(p4est, ghost);

  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
#ifdef P4_TO_P8
  double dt = cfl*MIN(dx[0], dx[1], dx[2]);
#else
  double dt = cfl*MIN(dx[0], dx[1]);
#endif

  Vec phi;
  VecCreateGhostNodes(p4est, nodes, &phi);
  sample_cf_on_nodes(p4est, nodes, interface, phi);

  const char* filename = "upwind";
  char vtkname[FILENAME_MAX];
  sprintf(vtkname, "%s.%04d", filename, 0);

  save_vtk(p4est, nodes, ghost, phi, vtkname);

  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  for (int i=0; i<iter; i++){
    // TVD-RK2 Method
    Vec phi_np1, phi_xx[P4EST_DIM];
    VecCreateGhostNodes(p4est, nodes, &phi_np1);
    foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &phi_xx[dim]);
    neighbors.second_derivatives_central(phi, phi_xx);

    // get access to array
    double *phi_p, *phi_np1_p, *phi_xx_p[P4EST_DIM];
    VecGetArray(phi, &phi_p);
    VecGetArray(phi_np1, &phi_np1_p);
    foreach_dimension(dim) VecGetArray(phi_xx[dim], &phi_xx_p[dim]);

    // 1) first half-step
    for (size_t i = 0; i<neighbors.get_layer_size(); i++) {
      p4est_locidx_t n = neighbors.get_layer_node(i);
      phi_np1_p[n] = upwind_step(neighbors, n, phi_p, phi_xx_p, vel, dt);
    }
    VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    for (size_t i = 0; i<neighbors.get_local_size(); i++) {
      p4est_locidx_t n = neighbors.get_local_node(i);
      phi_np1_p[n] = upwind_step(neighbors, n, phi_p, phi_xx_p, vel, dt);
    }
    VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD);

    // 2) update gradients
    neighbors.second_derivatives_central(phi_np1, phi_xx);

    // 3) second half-step
    for (size_t i = 0; i<neighbors.get_layer_size(); i++) {
      p4est_locidx_t n = neighbors.get_layer_node(i);
      double phi_np2 = upwind_step(neighbors, n, phi_np1_p, phi_xx_p, vel, dt);
      phi_p[n] = 0.5*(phi_p[n] + phi_np2);
    }
    VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
    for (size_t i = 0; i<neighbors.get_local_size(); i++) {
      p4est_locidx_t n = neighbors.get_local_node(i);
      double phi_np2 = upwind_step(neighbors, n, phi_np1_p, phi_xx_p, vel, dt);
      phi_p[n] = 0.5*(phi_p[n] + phi_np2);
    }
    VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);

    // clear temporaries
    VecRestoreArray(phi, &phi_p);
    foreach_dimension(dim) {
      VecRestoreArray(phi_xx[dim], &phi_xx_p[dim]);
      VecDestroy(phi_xx[dim]);
    }
    VecDestroy(phi_np1);

    // update the grid and levelset
    update(p4est, ghost, nodes, neighbors, phi);

    // reinitialize
    my_p4est_level_set_t ls(&neighbors);
    ls.reinitialize_2nd_order(phi);

    if (p4est->mpirank == 0)  {
      p4est_gloidx_t num_nodes = 0;
      for (int r=0; r<p4est->mpisize; r++)
      num_nodes += nodes->global_owned_indeps[r];
      printf("i = %4d, t = %1.5f n = %5lu\n", i, i*dt, num_nodes);
    }

    sprintf(vtkname, "%s.%04d", filename, i+1);
    save_vtk(p4est, nodes, ghost, phi, vtkname);
  }

  VecDestroy(phi);

  p4est_destroy(p4est);
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
}

void save_vtk(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, const char *vtkname)
{
  double *phi_p;
  VecGetArray(phi, &phi_p);
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, vtkname,
                         VTK_POINT_DATA, "phi", phi_p);
  VecRestoreArray(phi, &phi_p);
}

