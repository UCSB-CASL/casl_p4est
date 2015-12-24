#ifdef P4_TO_P8
#include "one_fluid_solver_3d.h"
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#else
#include "one_fluid_solver_2d.h"
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#endif

namespace {
#ifdef P4_TO_P8
struct bc_wall_type:WallBC3D {
  bc_wall_type(const my_p4est_brick_t* brick): brick(brick) {}
  BoundaryConditionType operator()(double x, double, double) const {
    if (fabs(x-brick->xyz_min[0]) < EPS || fabs(x-brick->xyz_max[0]) < EPS)
      return DIRICHLET;
    else
      return NEUMANN;
  }
private:
  const my_p4est_brick_t* brick;
};

struct bc_wall_value:CF_3 {
  double operator()(double, double, double) const {
    return 0;
  }
};
#else
struct bc_wall_type:WallBC2D {
  bc_wall_type(const my_p4est_brick_t* brick): brick(brick) {}
  BoundaryConditionType operator()(double x, double) const {
    if (fabs(x-brick->xyz_min[0]) < EPS || fabs(x-brick->xyz_max[0]) < EPS)
      return DIRICHLET;
    else
      return NEUMANN;
  }
private:
  const my_p4est_brick_t* brick;
};

struct bc_wall_value:CF_2 {
  double operator()(double, double) const {
    return 0;
  }
};
#endif
}

one_fluid_solver_t::one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick)
  : p4est(p4est), ghost(ghost), nodes(nodes), brick(&brick)
{
  sp   = (splitting_criteria_t*) p4est->user_pointer;
  conn = p4est->connectivity;
}

one_fluid_solver_t::~one_fluid_solver_t() {}

void one_fluid_solver_t::set_properties(cf_t &K_D, cf_t &gamma, cf_t &p_applied)
{
  this->K_D = &K_D;
  this->gamma = &gamma;
  this->p_applied = &p_applied;
}

double one_fluid_solver_t::advect_interface(Vec &phi, Vec &pressure, double cfl)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // compute interface velocity
  Vec vx_tmp[P4EST_DIM];
  double *vx_p[P4EST_DIM], *pressure_p, *phi_p;
  foreach_dimension(dim) {
    VecCreateGhostNodes(p4est, nodes, &vx_tmp[dim]);
    VecGetArray(vx_tmp[dim], &vx_p[dim]);
  }
  VecGetArray(pressure, &pressure_p);
  VecGetArray(phi, &phi_p);

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];
  double vn_max = 0;
  foreach_layer_node(n, neighbors) {
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double k = (*K_D)(x[0], x[1], x[2]);
#else
    double k = (*K_D)(x[0], x[1]);
#endif
    vx_p[0][n] = -k*qnnn.dx_central(pressure_p);
    vx_p[1][n] = -k*qnnn.dy_central(pressure_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -k*qnnn.dz_central(pressure_p);
    vx_max = MAX(vn_max, sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])+SQR(vx_p[2][n])));
#else
    vn_max = MAX(vn_max, sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])));
#endif
  }
  foreach_dimension(dim)
    VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // compute on the local nodes
  foreach_local_node(n, neighbors) {
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double k = (*K_D)(x[0], x[1], x[2]);
#else
    double k = (*K_D)(x[0], x[1]);
#endif
    vx_p[0][n] = -k*qnnn.dx_central(pressure_p);
    vx_p[1][n] = -k*qnnn.dy_central(pressure_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -k*qnnn.dz_central(pressure_p);
    vx_max = MAX(vn_max, sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])+SQR(vx_p[2][n])));
#else
    vn_max = MAX(vn_max, sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])));
#endif
  }
  foreach_dimension(dim)
    VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(pressure, &pressure_p);
  VecRestoreArray(phi, &phi_p);

  // constant extend the velocities from interface to the entire domain
  Vec vx[P4EST_DIM];
  foreach_dimension(dim) VecDuplicate(vx_tmp[dim], &vx[dim]);

  my_p4est_level_set_t ls(&neighbors);
  foreach_dimension(dim){
    ls.extend_from_interface_to_whole_domain_TVD(phi, vx_tmp[dim], vx[dim]);
    VecDestroy(vx_tmp[dim]);
  }

  // compute dt
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double dt = cfl*dmin/vn_max;
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // advect the level-set and update the grid
  p4est_t* p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_np1->connectivity = conn;
  p4est_np1->user_pointer = sp;
  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t* nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, &neighbors);

  sl.update_p4est(vx, dt, phi);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(pressure);
  VecDuplicate(phi, &pressure);

  return dt;
}

double one_fluid_solver_t::solve_one_step(Vec &phi, Vec &pressure, double cfl)
{
  // advect the interface
  double dt = advect_interface(phi, pressure, cfl);

  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);

  // reinitialize the levelset
  my_p4est_level_set_t ls(&neighbors);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  ls.perturb_level_set_function(phi, EPS);

  // compute the curvature. we store it in the boundary condition vector to save space
  Vec bc_val, phi_x[P4EST_DIM];
  VecDuplicate(phi, &bc_val);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &phi_x[dim]);
  neighbors.first_derivatives_central(phi, phi_x);
  compute_mean_curvature(neighbors, phi, phi_x, bc_val);
  foreach_dimension(dim) VecDestroy(phi_x[dim]);

  // compute the boundary condition for the pressure. we use kappa to store the resutls
  double *bc_val_p;
  VecGetArray(bc_val, &bc_val_p);

  foreach_node(n, nodes) {
    double x[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    bc_val_p[n] = (*p_applied)(x[0], x[1], x[2]) - bc_val_p[n]*(*gamma)(x[0], x[1], x[2]);
#else
    bc_val_p[n] = (*p_applied)(x[0], x[1]) - bc_val_p[n]*(*gamma)(x[0], x[1]);
#endif
  }
  VecRestoreArray(bc_val, &bc_val_p);

  // Set the boundary conditions
  my_p4est_interpolation_nodes_t p_interface_value(&neighbors);
  p_interface_value.set_input(bc_val, linear);

  bc_wall_type p_wall_type(brick);
  bc_wall_value p_wall_value;

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(p_interface_value);
  bc.setWallTypes(p_wall_type);
  bc.setWallValues(p_wall_value);

  // solve for the pressure
  Vec K;
  VecCreateGhostNodes(p4est, nodes, &K);
  sample_cf_on_nodes(p4est, nodes, *K_D, K);

  my_p4est_poisson_nodes_t poisson(&neighbors);
  poisson.set_bc(bc);
  poisson.set_mu(K);
  poisson.solve(pressure);

  // extend solution over interface
  ls.extend_Over_Interface(phi, pressure, DIRICHLET, bc_val);

  // destroy uneeded objects
  VecDestroy(bc_val);
  VecDestroy(K);

  return dt;
}


