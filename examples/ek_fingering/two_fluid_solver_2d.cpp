#ifdef P4_TO_P8
#include "two_fluid_solver_3d.h"
#include <src/my_p8est_poisson_jump_nodes_extended.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#else
#include "two_fluid_solver_2d.h"
#include <src/my_p4est_poisson_jump_nodes_extended.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_poisson_nodes.h>
#endif

#include <src/casl_math.h>
#include <cassert>

two_fluid_solver_t::two_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick)
  : p4est(p4est), ghost(ghost), nodes(nodes), brick(&brick)
{
  sp   = (splitting_criteria_t*) p4est->user_pointer;
  conn = p4est->connectivity;

  VecCreateGhostNodes(p4est, nodes, &kappa);
  foreach_dimension (dim) {
    VecDuplicate(kappa, &nx[dim]);
    VecDuplicate(kappa, &n1[dim]);
  }
  VecDuplicate(kappa, &un);

  p4est_nm1 = my_p4est_copy(p4est, 0);
  p4est_nm1->connectivity = conn;
  ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);

  dt_nm1 = 0;
  VecCreateGhostNodes(p4est_nm1, nodes_nm1, &phi_nm1);
  VecCreateGhostNodes(p4est_nm1, nodes_nm1, &press_m_nm1);
  VecCreateGhostNodes(p4est_nm1, nodes_nm1, &press_p_nm1);

  VecDuplicate(phi_nm1, &un_nm1);
}

two_fluid_solver_t::~two_fluid_solver_t()
{
  VecDestroy(kappa);
  foreach_dimension (dim) {
    VecDestroy(nx[dim]);
    VecDestroy(n1[dim]);
  }
  VecDestroy(un_nm1);
  VecDestroy(un);

  p4est_nodes_destroy(nodes_nm1);
  p4est_ghost_destroy(ghost_nm1);
  p4est_destroy(p4est_nm1);

  VecDestroy(phi_nm1);
  VecDestroy(press_m_nm1);
  VecDestroy(press_p_nm1);
}

void two_fluid_solver_t::set_properties(double viscosity_ratio, double Ca, CF_1& Q)
{    
  this->viscosity_ratio = viscosity_ratio;
  this->Ca = Ca;
  this->Q = &Q;
}

void two_fluid_solver_t::set_bc_wall(bc_wall_t &bc_wall_type, cf_t &bc_wall_value)
{
  this->bc_wall_type  = &bc_wall_type;
  this->bc_wall_value = &bc_wall_value;
}

void two_fluid_solver_t::compute_normal_and_curvature_diagonal(my_p4est_node_neighbors_t& neighbors, Vec &phi)
{
  VecDestroy(kappa); VecDuplicate(phi, &kappa);
  foreach_dimension(dim) {
    VecDestroy(nx[dim]); VecDuplicate(phi, &nx[dim]);
    VecDestroy(n1[dim]); VecDuplicate(phi, &n1[dim]);
  }
  Vec kappa_tmp;
  VecDuplicate(kappa, &kappa_tmp);

  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
  double diag = sqrt(SQR(dx[0]) + SQR(dx[1]));

  double *kappa_p, *phi_p, *nx_p[P4EST_DIM], *n1_p[P4EST_DIM];
  VecGetArray(kappa_tmp, &kappa_p);
  VecGetArray(phi, &phi_p);
  foreach_dimension (dim) {
    VecGetArray(nx[dim], &nx_p[dim]);
    VecGetArray(n1[dim], &n1_p[dim]);
  }

  my_p4est_interpolation_nodes_t interp(&neighbors);
  Vec fxx[P4EST_DIM];
  foreach_dimension (dim) VecCreateGhostNodes(p4est, nodes, &fxx[dim]);
  neighbors.second_derivatives_central(phi, fxx);
  interp.set_input(phi, fxx[0], fxx[1], quadratic);

  double f[3][3];
  double x[P4EST_DIM];
  foreach_local_node (n, nodes) {
    if (fabs(phi_p[n]) < 30*diag) {
      node_xyz_fr_n(n, p4est, nodes, x);
      for (short i = 0; i < 3; i++)
        for (short j = 0; j < 3; j++)
          f[i][j] = interp(x[0]+(i-1)*dx[0],x[1]+(j-1)*dx[1]);

      const int i = 1, j = 1;
      double fx  = (f[i+1][j]-f[i-1][j])/(2*dx[0]);
      double fy  = (f[i][j+1]-f[i][j-1])/(2*dx[1]);
      double fxx = (f[i+1][j]-2*f[i][j]+f[i-1][j])/(dx[0]*dx[0]);
      double fyy = (f[i][j+1]-2*f[i][j]+f[i][j-1])/(dx[1]*dx[1]);
      double fxy = (f[i+1][j+1]+f[i-1][j-1]-f[i+1][j-1]-f[i-1][j+1])/(4*dx[0]*dx[1]);

      double fn  = MAX(sqrt(fx*fx+fy*fy), EPS);
      kappa_p[n] = (SQR(fy)*fxx-2*fx*fy*fxy+SQR(fx)*fyy)/std::pow(fn,3);

      nx_p[0][n] = fx/fn;
      nx_p[1][n] = fy/fn;

      double f1  = (f[i+1][j+1]-f[i-1][j-1])/(2*diag);
      double f2  = (f[i-1][j+1]-f[i+1][j-1])/(2*diag);
      fn         = MAX(sqrt(f1*f1+f2*f2), EPS);
      n1_p[0][n] = f1/fn;
      n1_p[1][n] = f2/fn;
    }
  }

  foreach_dimension (dim) VecDestroy(fxx[dim]);

  VecRestoreArray(kappa_tmp, &kappa_p);
  VecRestoreArray(phi, &phi_p);
  foreach_dimension (dim) {
    VecRestoreArray(nx[dim], &nx_p[dim]);
    VecRestoreArray(n1[dim], &n1_p[dim]);
  }

  VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(kappa, INSERT_VALUES, SCATTER_FORWARD);

  foreach_dimension (dim) {
    VecGhostUpdateBegin(nx[dim], INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(nx[dim], INSERT_VALUES, SCATTER_FORWARD);

    VecGhostUpdateBegin(n1[dim], INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(n1[dim], INSERT_VALUES, SCATTER_FORWARD);
  }

  // extend curvature
  my_p4est_level_set_t ls(&neighbors);
  ls.extend_from_interface_to_whole_domain(phi, kappa_tmp, kappa);

  VecDestroy(kappa_tmp);
}

void two_fluid_solver_t::compute_normal_velocity_diagonal(my_p4est_node_neighbors_t& neighbors, Vec& phi, Vec &pressure) {
  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
  double diag = sqrt(SQR(dx[0]) + SQR(dx[1]));

  Vec un_tmp;
  VecDuplicate(un, &un_tmp);

  double *phi_p, *pressure_p, *nx_p[P4EST_DIM], *n1_p[P4EST_DIM], *un_p;
  VecGetArray(un_tmp, &un_p);
  VecGetArray(phi, &phi_p);
  VecGetArray(pressure, &pressure_p);
  foreach_dimension (dim) {
    VecGetArray(nx[dim], &nx_p[dim]);
    VecGetArray(n1[dim], &n1_p[dim]);
  }

  Vec Fxx[P4EST_DIM];
  foreach_dimension (dim) VecCreateGhostNodes(p4est, nodes, &Fxx[dim]);
  neighbors.second_derivatives_central(pressure, Fxx);

  my_p4est_interpolation_nodes_t interp(&neighbors);
  interp.set_input(pressure, Fxx[0], Fxx[1], quadratic);

  double f[3][3];
  double x[P4EST_DIM];
  auto compute_velocity = [&](int n) -> double {
    if (fabs(phi_p[n]) < 30*diag) {
      node_xyz_fr_n(n, p4est, nodes, x);
      for (short i = 0; i < 3; i++) {
        for (short j = 0; j < 3; j++) {
          f[i][j] = interp(x[0]+(i-1)*dx[0], x[1]+(j-1)*dx[1]);
        }
      }

      const int i = 1, j = 1;

      double fx  = (f[i+1][j]-f[i-1][j])/(2*dx[0]);
      double fy  = (f[i][j+1]-f[i][j-1])/(2*dx[1]);
      double f1  = (f[i+1][j+1]-f[i-1][j-1])/(2*diag);
      double f2  = (f[i-1][j+1]-f[i+1][j-1])/(2*diag);

      return -(nx_p[0][n]*fx + nx_p[1][n]*fy +
               n1_p[0][n]*f1 + n1_p[1][n]*f2)/2.0;
    } else {
      return 0;
    }
  };

  for (size_t i = 0; i < neighbors.get_layer_size(); i++) {
    int n = neighbors.get_layer_node(i);
    un_p[n] = compute_velocity(n);
  }
  VecGhostUpdateBegin(un_tmp, INSERT_VALUES, SCATTER_FORWARD);

  for (size_t i = 0; i < neighbors.get_local_size(); i++) {
    int n = neighbors.get_local_node(i);
    un_p[n] = compute_velocity(n);
  }
  VecGhostUpdateEnd(un_tmp, INSERT_VALUES, SCATTER_FORWARD);

  VecRestoreArray(un_tmp, &un_p);

  foreach_dimension (dim) VecDestroy(Fxx[dim]);

  VecRestoreArray(phi, &phi_p);
  VecRestoreArray(pressure, &pressure_p);
  foreach_dimension (dim) {
    VecRestoreArray(nx[dim], &nx_p[dim]);
    VecRestoreArray(n1[dim], &n1_p[dim]);
  }

  // constant extend the velocities from interface to the entire domain
  my_p4est_level_set_t ls(&neighbors);
  ls.extend_from_interface_to_whole_domain(phi, un_tmp, un);

  VecDestroy(un_tmp);
}


double two_fluid_solver_t::advect_interface_godunov(Vec &phi, Vec &press_m, Vec& press_p, double cfl, double dtmax)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // compute normal and curvature
  compute_normal_and_curvature_diagonal(neighbors, phi);
  compute_normal_velocity_diagonal(neighbors, phi, press_p);

  // compute dt based on cfl number and curavture
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double un_max = 1; // minmum vn_max to be used when computing dt.
  double kon_max = 0;
  double *kappa_p, *un_p, *phi_p;
  VecGetArray(un, &un_p);
  VecGetArray(kappa, &kappa_p);
  VecGetArray(phi, &phi_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < 2*diag) {
      un_max  = MAX(un_max, un_p[n]);
      kon_max = MAX(kon_max, fabs(kappa_p[n]*un_p[n]));
    }
  }
  VecRestoreArray(kappa, &kappa_p);

  double dt = MIN(cfl*dmin/un_max, 1.0/kon_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  my_p4est_level_set_t ls(&neighbors);
  static bool first_iteration = true;
  if (first_iteration) {
    dt = ls.advect_in_normal_direction(un, phi, dt);
  } else {
    // compute an approximation to un_np1 using extrapolation
    my_p4est_hierarchy_t h(p4est_nm1, ghost_nm1, brick);
    my_p4est_node_neighbors_t ngbd_nm1(&h, nodes_nm1);
    ngbd_nm1.init_neighbors();

    Vec fxx[P4EST_DIM];
    foreach_dimension (dim) VecCreateGhostNodes(p4est_nm1, nodes_nm1, &fxx[dim]);
    ngbd_nm1.second_derivatives_central(un_nm1, fxx);

    my_p4est_interpolation_nodes_t interp_nm1(&ngbd_nm1);
    interp_nm1.set_input(un_nm1, fxx[0], fxx[1], quadratic);

    Vec un_np1;
    VecCreateGhostNodes(p4est, nodes, &un_np1);

    double x[P4EST_DIM];
    foreach_node (n, nodes) {
      node_xyz_fr_n(n, p4est, nodes, x);
      interp_nm1.add_point(n, x);
    }
    interp_nm1.interpolate(un_np1);

    foreach_dimension (dim) VecDestroy(fxx[dim]);

    double *un_p, *un_np1_p;
    VecGetArray(un, &un_p);
    VecGetArray(un_np1, &un_np1_p);
    foreach_node (n, nodes) {
      un_np1_p[n] = un_p[n] + (un_p[n] - un_np1_p[n]) * dt/dt_nm1;
    }

    VecRestoreArray(un, &un_p);
    VecRestoreArray(un_np1, &un_np1_p);

    dt = ls.advect_in_normal_direction(un, un_np1, phi, dt);

    VecDestroy(un_np1);
  }

  p4est_t* p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_np1->connectivity = conn;
  p4est_np1->user_pointer = sp;

  splitting_criteria_tag_t sp_tag(sp->min_lvl, sp->max_lvl, sp->lip);
  sp_tag.refine_and_coarsen(p4est_np1, nodes, phi_p);

  // partition and compute new strutures
  my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);
  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t* nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  // transfer data from old grid to new
  Vec phi_np1;
  VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1);

  // create an interpolation function between two grids
  my_p4est_interpolation_nodes_t grid_interp(&neighbors);
  double x[P4EST_DIM];
  foreach_node(n, nodes_np1) {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    grid_interp.add_point(n,x);
  }

  // interpolate variables
  grid_interp.set_input(phi, quadratic_non_oscillatory);
  grid_interp.interpolate(phi_np1);

  // n --> nm1
  p4est_destroy(p4est_nm1); p4est_nm1 = p4est;
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes;
  VecDestroy(phi_nm1); phi_nm1 = phi;
  VecDestroy(un_nm1); un_nm1 = un;
  VecDestroy(press_m_nm1); press_m_nm1 = press_m;
  VecDestroy(press_p_nm1); press_p_nm1 = press_p;
  dt_nm1 = dt;

  // np1 --> n
  p4est = p4est_np1;
  ghost = ghost_np1;
  nodes = nodes_np1;
  phi = phi_np1;
  VecDuplicate(phi, &un);
  VecCreateGhostNodes(p4est, nodes, &press_m);
  VecCreateGhostNodes(p4est, nodes, &press_p);

  return dt;
}

double two_fluid_solver_t::advect_interface(Vec &phi, Vec &press_m, Vec& press_p, double cfl, double dtmax)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // compute interface velocity
  Vec vx_tmp[P4EST_DIM];
  double *vx_p[P4EST_DIM], *press_p_p, *press_m_p, *phi_p;
  foreach_dimension(dim) {
    VecCreateGhostNodes(p4est, nodes, &vx_tmp[dim]);
    VecGetArray(vx_tmp[dim], &vx_p[dim]);
  }
  VecGetArray(press_m, &press_m_p);
  VecGetArray(press_p, &press_p_p);
  VecGetArray(phi, &phi_p);

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];
  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);

    vx_p[0][n] = -qnnn.dx_central(press_p_p);
    vx_p[1][n] = -qnnn.dy_central(press_p_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -qnnn.dz_central(press_p_p);
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);

    vx_p[0][n] = -qnnn.dx_central(press_p_p);
    vx_p[1][n] = -qnnn.dy_central(press_p_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -qnnn.dz_central(press_p_p);
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(press_m, &press_m_p);
  VecRestoreArray(press_p, &press_p_p);

  // constant extend the velocities from interface to the entire domain
  my_p4est_level_set_t ls(&neighbors);
  Vec vx[P4EST_DIM];
  foreach_dimension (dim) {
    VecDuplicate(vx_tmp[dim], &vx[dim]);
    ls.extend_from_interface_to_whole_domain_TVD(phi, vx_tmp[dim], vx[dim]);
//    VecCopy(vx_tmp[dim], vx[dim]);
//    VecGhostUpdateBegin(vx[dim], INSERT_VALUES, SCATTER_FORWARD);
//    VecGhostUpdateEnd(vx[dim], INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(vx_tmp[dim]);
  }

  // compute curvature
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, kappa_tmp);
  foreach_dimension(dim) VecDestroy(normal[dim]);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // compute dt based on cfl number and curavture
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double vn_max = 1; // minmum vn_max to be used when computing dt.
  double kvn_max = 0;
  double *kappa_p;
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);
  VecGetArray(kappa, &kappa_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < 2*diag) {
#ifdef P4_TO_P8
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])+SQR(vx_p[2][n]));
#else
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n]));
#endif
      vn_max  = MAX(vn_max, vn);
      kvn_max = MAX(kvn_max, fabs(kappa_p[n]*vn));
    }
  }
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);
  VecRestoreArray(kappa, &kappa_p);
  VecDestroy(kappa);

  double dt = MIN(cfl*dmin/vn_max, 1.0/kvn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // advect the level-set and update the grid
  p4est_t* p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_np1->connectivity = conn;
  p4est_np1->user_pointer = sp;
  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t* nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, &neighbors);

  sl.update_p4est(vx, dt, phi);

  /*
   * Since the voronoi solver requires two layers ghost cells, we need to expand the ghost layer
   * and copy the date to the new vector that has room for extra ghost points
   */
  // destroy old data structures and create new ones
  p4est_destroy(p4est); p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  my_p4est_ghost_expand(p4est, ghost);
  p4est_nodes_destroy(nodes_np1);
  p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

  // copy data
  Vec phi_np1 = phi;
  VecCreateGhostNodes(p4est, nodes, &phi);
  VecCopy(phi_np1, phi);
  VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);

  // destroy old stuff
  VecDestroy(phi_np1);
  VecDestroy(press_m); VecDuplicate(phi, &press_m);
  VecDestroy(press_p); VecDuplicate(phi, &press_p);

  foreach_dimension(dim) VecDestroy(vx[dim]);

  return dt;
}

void two_fluid_solver_t::solve_fields_extended(double t, Vec phi, Vec press_m, Vec press_p)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // reinitialize the levelset
  my_p4est_level_set_t ls(&neighbors);
//  ls.reinitialize_2nd_order(phi);
//  ls.perturb_level_set_function(phi, EPS);

  // compute the curvature. we store it in the boundary condition vector to save space
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, kappa_tmp);

  // extend curvature from interface to the entire domain
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // compute the boundary condition for the pressure.
  Vec jump_p, jump_dp;
  VecDuplicate(phi, &jump_p);
  VecDuplicate(phi, &jump_dp);

  double *jump_p_p, *jump_dp_p, *kappa_p;
  VecGetArray(kappa,   &kappa_p);
  VecGetArray(jump_p,  &jump_p_p);
  VecGetArray(jump_dp, &jump_dp_p);

  double x[P4EST_DIM];
  double diag_min = p4est_diag_min(p4est);

  // compute the singular part
  std::vector<double> pstar(nodes->indep_nodes.elem_count);
  double *normal_p[P4EST_DIM];

  foreach_node(n, nodes) {
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double r = MAX(diag_min, sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2])));
    pstar[n] = (*Q)(t)/(4*PI*r);
#else
    double r = MAX(diag_min, sqrt(SQR(x[0]) + SQR(x[1])));
    pstar[n] = (*Q)(t)/(2*PI)*log(r);
#endif
  }

  // jump in solution
  foreach_node(n, nodes) {
    jump_p_p[n]  = -1/Ca*kappa_p[n] - viscosity_ratio*pstar[n];
  }
  VecRestoreArray(jump_p, &jump_p_p);
  VecRestoreArray(kappa, &kappa_p);
  VecDestroy(kappa);

  // jump in the flux is a bit more involved
  // FIXME: change the definiton of normal in the jump solver to remain consistent
  quad_neighbor_nodes_of_node_t qnnn;
  foreach_dimension(dim) VecGetArray(normal[dim], &normal_p[dim]);

  for (size_t i = 0; i<neighbors.get_layer_size(); i++) {
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);

    double *pstar_p = pstar.data();
    jump_dp_p[n]  = -qnnn.dx_central(pstar_p)*normal_p[0][n] - qnnn.dy_central(pstar_p)*normal_p[1][n];
#ifdef P4_TO_P8
    jump_dp_p[n] += -qnnn.dz_central(pstar_p)*normal_p[2][n];
#endif
  }
  VecGhostUpdateBegin(jump_dp, INSERT_VALUES, SCATTER_FORWARD);

  for (size_t i = 0; i<neighbors.get_local_size(); i++) {
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);

    double *pstar_p = pstar.data();
    jump_dp_p[n]  = -qnnn.dx_central(pstar_p)*normal_p[0][n] - qnnn.dy_central(pstar_p)*normal_p[1][n];
#ifdef P4_TO_P8
    jump_dp_p[n] += -qnnn.dz_central(pstar_p)*normal_p[2][n];
#endif
  }
  VecGhostUpdateEnd(jump_dp, INSERT_VALUES, SCATTER_FORWARD);
  VecRestoreArray(jump_dp, &jump_dp_p);

  // destroy normals
  foreach_dimension(dim) {
    VecRestoreArray(normal[dim], &normal_p[dim]);
    VecDestroy(normal[dim]);
  }

  my_p4est_interpolation_nodes_t jump_p_interp(&neighbors), jump_dp_interp(&neighbors);
  jump_p_interp.set_input(jump_p, linear);
  jump_dp_interp.set_input(jump_dp, linear);

  // solve the pressure jump problem
  my_p4est_poisson_jump_nodes_extended_t jump_solver(&neighbors);

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc_wall_value->t = t;
  bc.setWallTypes(*bc_wall_type);
  bc.setWallValues(*bc_wall_value);

  jump_solver.set_bc(bc);
  jump_solver.set_jump(jump_p_interp, jump_dp_interp);
  jump_solver.set_phi(phi);
  jump_solver.set_mue(1.0, 1.0/MAX(viscosity_ratio, EPS));

  Vec sol;
  VecCreateGhostNodesBlock(p4est, nodes, 2, &sol);
  jump_solver.solve(sol);

  // extract solutions
  double *press_m_p, *press_p_p, *sol_p;
  VecGetArray(press_m, &press_m_p);
  VecGetArray(press_p, &press_p_p);
  VecGetArray(sol, &sol_p);

  foreach_node(n, nodes) {
    press_m_p[n] = sol_p[2*n+0] - viscosity_ratio*pstar[n];
    press_p_p[n] = sol_p[2*n+1];
  }

  VecRestoreArray(sol, &sol_p);
  VecDestroy(sol);
  VecDestroy(jump_p);
  VecDestroy(jump_dp);

  // extend solutions
  VecRestoreArray(press_m, &press_m_p);
  VecRestoreArray(press_p, &press_p_p);

  // (-) --> (+)
  ls.extend_Over_Interface_TVD(phi, press_m);

  // (+) --> (-)
  Vec phi_l;
  VecGhostGetLocalForm(phi, &phi_l);
  VecScale(phi_l, -1);

  ls.extend_Over_Interface_TVD(phi, press_p);

  VecScale(phi_l, -1);
  VecGhostRestoreLocalForm(phi, &phi_l);
}

void two_fluid_solver_t::solve_fields_voronoi(double t, Vec phi, Vec press_m, Vec press_p)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  node_neighbors.init_neighbors();

  // reinitialize the levelset
  my_p4est_level_set_t ls(&node_neighbors);
//  ls.reinitialize_2nd_order(phi);
//  ls.perturb_level_set_function(phi, EPS);

  // compute the curvature. we store it in the boundary condition vector to save space
//  Vec kappa, kappa_tmp, normal[P4EST_DIM];
//  VecDuplicate(phi, &kappa);
//  VecDuplicate(phi, &kappa_tmp);
//  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
//  compute_normals(node_neighbors, phi, normal);
//  {
//    Vec phi_x[P4EST_DIM];
//    VecCreateGhostNodes(p4est, nodes, &phi_x[0]);
//    VecCreateGhostNodes(p4est, nodes, &phi_x[1]);
//    node_neighbors.first_derivatives_central(phi, phi_x);
//    compute_mean_curvature(node_neighbors, phi, phi_x, kappa_tmp);
//    VecDestroy(phi_x[0]);
//    VecDestroy(phi_x[1]);
//  }
//  compute_mean_curvature(node_neighbors, normal, kappa_tmp);

////   extend curvature from interface to the entire domain
//  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
//  VecDestroy(kappa_tmp);

  compute_normal_and_curvature_diagonal(node_neighbors, phi);

  // compute the boundary condition for the pressure.
  Vec jump_p, jump_dp;
  VecDuplicate(phi, &jump_p);
  VecDuplicate(phi, &jump_dp);

  double *jump_p_p, *jump_dp_p, *kappa_p;
  VecGetArray(kappa,   &kappa_p);
  VecGetArray(jump_p,  &jump_p_p);
  VecGetArray(jump_dp, &jump_dp_p);

  double x[P4EST_DIM];

  // compute the singular part
  std::vector<double> pstar(nodes->indep_nodes.elem_count);
  double *nx_p[P4EST_DIM];
  double diag_min = p4est_diag_min(p4est);

  foreach_node(n, nodes) {
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double r = MAX(diag_min, sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2])));
    pstar[n] = (*Q)(t)/(4*PI*r);
#else
    double r = MAX(diag_min, sqrt(SQR(x[0]) + SQR(x[1])));
    pstar[n] = (*Q)(t)/(2*PI)*log(r);
#endif
  }

  // jump in solution
  foreach_node(n, nodes) {
    jump_p_p[n]  = -1/Ca*kappa_p[n] - viscosity_ratio*pstar[n];
  }
  VecRestoreArray(jump_p, &jump_p_p);
  VecRestoreArray(kappa, &kappa_p);
//  VecDestroy(kappa);

  // jump in the flux is a bit more involved
  // FIXME: change the definiton of normal in the jump solver to remain consistent
  quad_neighbor_nodes_of_node_t qnnn;
  foreach_dimension(dim) VecGetArray(nx[dim], &nx_p[dim]);

  for (size_t i = 0; i<node_neighbors.get_layer_size(); i++) {
    p4est_locidx_t n = node_neighbors.get_layer_node(i);
//    node_neighbors.get_neighbors(n, qnnn);

//    double *pstar_p = pstar.data();
//    jump_dp_p[n]  = -qnnn.dx_central(pstar_p)*nx_p[0][n] - qnnn.dy_central(pstar_p)*nx_p[1][n];
//#ifdef P4_TO_P8
//    jump_dp_p[n] += -qnnn.dz_central(pstar_p)*nx_p[2][n];
//#endif
    node_xyz_fr_n(n, p4est, nodes, x);
    double r = sqrt(SQR(x[0]) + SQR(x[1]));
    jump_dp_p[n] = (*Q)(t)/(2*PI)*(x[0]*nx_p[0][n] + x[1]*nx_p[1][n])/SQR(r);
  }
  VecGhostUpdateBegin(jump_dp, INSERT_VALUES, SCATTER_FORWARD);

  for (size_t i = 0; i<node_neighbors.get_local_size(); i++) {
    p4est_locidx_t n = node_neighbors.get_local_node(i);
//    node_neighbors.get_neighbors(n, qnnn);

//    double *pstar_p = pstar.data();
//    jump_dp_p[n]  = -qnnn.dx_central(pstar_p)*nx_p[0][n] - qnnn.dy_central(pstar_p)*nx_p[1][n];
//#ifdef P4_TO_P8
//    jump_dp_p[n] += -qnnn.dz_central(pstar_p)*nx_p[2][n];
//#endif

    node_xyz_fr_n(n, p4est, nodes, x);
    double r = sqrt(SQR(x[0]) + SQR(x[1]));
    jump_dp_p[n] = (*Q)(t)/(2*PI)*(x[0]*nx_p[0][n] + x[1]*nx_p[1][n])/SQR(r);

  }
  VecGhostUpdateEnd(jump_dp, INSERT_VALUES, SCATTER_FORWARD);
  VecRestoreArray(jump_dp, &jump_dp_p);

  // destroy normals
//  foreach_dimension(dim) {
//    VecRestoreArray(normal[dim], &nx_p[dim]);
//    VecDestroy(normal[dim]);
//  }

  // solve the pressure jump problem
  Vec rhs_p, rhs_m, mue_m, mue_p;
  VecDuplicate(phi, &rhs_p);
  VecDuplicate(phi, &rhs_m);
  VecDuplicate(phi, &mue_p);
  VecDuplicate(phi, &mue_m);
  {
    Vec l;
    VecGhostGetLocalForm(rhs_m, &l);
    VecSet(l, 0);
    VecGhostRestoreLocalForm(rhs_m, &l);

    VecGhostGetLocalForm(rhs_p, &l);
    VecSet(l, 0);
    VecGhostRestoreLocalForm(rhs_p, &l);

    VecGhostGetLocalForm(mue_m, &l);
    VecSet(l, 1.0/MAX(viscosity_ratio, EPS));
    VecGhostRestoreLocalForm(mue_m, &l);

    VecGhostGetLocalForm(mue_p, &l);
    VecSet(l, 1.0);
    VecGhostRestoreLocalForm(mue_p, &l);
  }

  my_p4est_cell_neighbors_t cell_neighbors(&hierarchy);

  my_p4est_poisson_jump_nodes_voronoi_t jump_solver(&node_neighbors, &cell_neighbors);

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc_wall_value->t = t;
  bc.setWallTypes(*bc_wall_type);
  bc.setWallValues(*bc_wall_value);

  jump_solver.set_phi(phi);
  jump_solver.set_bc(bc);
  jump_solver.set_mu(mue_m, mue_p);
  jump_solver.set_u_jump(jump_p);
  jump_solver.set_mu_grad_u_jump(jump_dp);
  jump_solver.set_rhs(rhs_m, rhs_p);

  jump_solver.solve(press_p);

//  jump_solver.print_voronoi_VTK("two_fluid_voro");

  VecDestroy(rhs_m);
  VecDestroy(rhs_p);
  VecDestroy(mue_m);
  VecDestroy(mue_p);
  VecDestroy(jump_p);
  VecDestroy(jump_dp);

  // extract solutions
  double *press_m_p, *press_p_p;
  VecGetArray(press_m, &press_m_p);
  VecGetArray(press_p, &press_p_p);

  // add the singular part to the inside solution
  foreach_node(n, nodes) {
    press_m_p[n] = press_p_p[n] - viscosity_ratio*pstar[n];
    assert(!isnan(press_p_p[n]) && !isinf(press_p_p[n]));
    assert(!isnan(press_m_p[n]) && !isinf(press_m_p[n]));
  }

  // extend solutions
  VecRestoreArray(press_m, &press_m_p);
  VecRestoreArray(press_p, &press_p_p);

  Vec phi_l;
  VecGhostGetLocalForm(phi, &phi_l);

  // (-) --> (+)
  VecShift(phi_l, diag_min);
  ls.extend_Over_Interface_TVD(phi, press_m);
  VecShift(phi_l, -diag_min);

  // (+) --> (-)
  VecScale(phi_l, -1);
  VecShift(phi_l, diag_min);
  ls.extend_Over_Interface_TVD(phi, press_p);
  VecShift(phi_l, -diag_min);
  VecScale(phi_l, -1);

  VecGhostRestoreLocalForm(phi, &phi_l); 
}

double two_fluid_solver_t:: solve_one_step(double t, Vec &phi, Vec &press_m, Vec& press_p, double cfl, double dtmax, std::string method)
{
  // advect the interface
  double dt;
  dt = advect_interface_godunov(phi, press_m, press_p, cfl, dtmax);

  // save the grid

  // solve for the pressure
  if (method == "extended")
    solve_fields_extended(t+dt, phi, press_m, press_p);
  else if (method == "voronoi")
    solve_fields_voronoi(t+dt, phi, press_m, press_p);
  else
    throw std::invalid_argument("invalid method");

  return dt;
}


