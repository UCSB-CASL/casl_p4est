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

#include <src/my_p4est_vtk.h>
#include <src/casl_math.h>

one_fluid_solver_t::one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick)
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
  VecDuplicate(phi_nm1, &pressure_nm1);
  foreach_dimension (dim) {
    VecDuplicate(phi_nm1, &vx_nm1[dim]);
  }
  VecDuplicate(phi_nm1, &un_nm1);
}

one_fluid_solver_t::~one_fluid_solver_t()
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
  VecDestroy(pressure_nm1);
  foreach_dimension (dim) {
    VecDestroy(vx_nm1[dim]);
  }
}

one_fluid_solver_t& one_fluid_solver_t::set_properties(cf_t &K_D, cf_t &K_EO, cf_t &gamma)
{  
  this->K_D   = &K_D;
  this->K_EO  = &K_EO;
  this->gamma = &gamma;

  return *this;
}

one_fluid_solver_t& one_fluid_solver_t::set_bc_wall(bc_wall_t &bc_wall_type, cf_t &bc_wall_value)
{
  this->bc_wall_type  = &bc_wall_type;
  this->bc_wall_value = &bc_wall_value;

  return *this;
}

one_fluid_solver_t& one_fluid_solver_t::set_integration(std::string method, double cfl, double dtmax)
{
  this->method = method;
  this->cfl = cfl;
  this->dtmax = dtmax;

  return *this;
}

void one_fluid_solver_t::compute_normal_and_curvature_diagonal(my_p4est_node_neighbors_t& neighbors, Vec &phi)
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
//  interp.set_input(phi, linear);

  double f[3][3];
  double x[P4EST_DIM];  
  foreach_local_node (n, nodes) {
    if (fabs(phi_p[n]) < 15*diag) {
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

void one_fluid_solver_t::compute_normal_velocity_diagonal(my_p4est_node_neighbors_t& neighbors, Vec& phi, Vec &pressure) {
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

  my_p4est_interpolation_nodes_t interp(&neighbors);
  Vec Fxx[P4EST_DIM];
  foreach_dimension (dim) VecCreateGhostNodes(p4est, nodes, &Fxx[dim]);
  neighbors.second_derivatives_central(pressure, Fxx);
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

      double kd  = (*K_D)(x[0], x[1]);
      return -kd*(nx_p[0][n]*fx + nx_p[1][n]*fy + n1_p[0][n]*f1 + n1_p[1][n]*f2)/2.0;
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

double one_fluid_solver_t::advect_interface_semi_lagrangian(Vec &phi, Vec &pressure)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  my_p4est_hierarchy_t hierarchy_nm1(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t neighbors_nm1(&hierarchy_nm1, nodes_nm1);
  neighbors_nm1.init_neighbors();

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
  auto compute_velocity = [&](int n, double* vx_p[P4EST_DIM]){
    double x[P4EST_DIM];
    quad_neighbor_nodes_of_node_t qnnn;

    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double kd  = (*K_D)(x[0], x[1], x[2]);
#else
    double kd  = (*K_D)(x[0], x[1]);
#endif
    vx_p[0][n] = -kd*qnnn.dx_central(pressure_p);
    vx_p[1][n] = -kd*qnnn.dy_central(pressure_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -kd*qnnn.dz_central(pressure_p);
#endif
  };

  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    compute_velocity(n, vx_p);
  }

  foreach_dimension (dim) {
      VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);
  }

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);
    compute_velocity(n, vx_p);
  }

  foreach_dimension (dim) {
      VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);  
  }

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(pressure, &pressure_p);

  // constant extend the velocities from interface to the entire domain
  my_p4est_level_set_t ls(&neighbors);
  Vec vx[P4EST_DIM];
  foreach_dimension (dim) {
    VecDuplicate(vx_tmp[dim], &vx[dim]);
    ls.extend_from_interface_to_whole_domain_TVD(phi, vx_tmp[dim], vx[dim]);
    VecDestroy(vx_tmp[dim]);
  }

  // smooth out the velocity field
  foreach_dimension (dim) VecGetArray(vx[dim], &vx_p[dim]);
  foreach_dimension (dim) VecRestoreArray(vx[dim], &vx_p[dim]);

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
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < 2*diag) {
#ifdef P4_TO_P8
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])+SQR(vx_p[2][n]));
#else
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n]));
#endif
      vn_max  = MAX(vn_max, vn);
    }
  }
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);

  double dt = MIN(cfl*dmin/vn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // advect the level-set and update the grid
  p4est_t* p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_np1->connectivity = conn;
  p4est_np1->user_pointer = sp;
  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t* nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1,
                                &neighbors, &neighbors_nm1);

  // copy phi to phi_nm1 before updating
  Vec phi_np1;
  VecDuplicate(phi, &phi_np1);
  VecGhostCopy(phi, phi_np1);

  // use 1st-order semi-lagrangian on the first iteration and then switch to 2nd
  static bool first_iteration = true;
  if (first_iteration) {
    sl.update_p4est(vx, dt, phi_np1);
    first_iteration = false;
  } else {
    sl.update_p4est(vx_nm1, vx, dt_nm1, dt, phi_np1);
  }

  // n --> nm1
  p4est_destroy(p4est_nm1); p4est_nm1 = p4est;
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes;
  VecDestroy(phi_nm1); phi_nm1 = phi;
  VecDestroy(pressure_nm1); pressure_nm1 = pressure;
  foreach_dimension (dim) {
    VecDestroy(vx_nm1[dim]); vx_nm1[dim] = vx[dim];
  }
  dt_nm1 = dt;

  // np1 --> n
  p4est = p4est_np1;
  ghost = ghost_np1;
  nodes = nodes_np1;
  phi = phi_np1;
  VecDuplicate(phi, &pressure);

  return dt;
}

double one_fluid_solver_t::advect_interface_semi_lagrangian_1st(Vec &phi, Vec &pressure)
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
  auto compute_velocity = [&](int n, double* vx_p[P4EST_DIM]){
    double x[P4EST_DIM];
    quad_neighbor_nodes_of_node_t qnnn;

    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double kd  = (*K_D)(x[0], x[1], x[2]);
#else
    double kd  = (*K_D)(x[0], x[1]);
#endif
    vx_p[0][n] = -kd*qnnn.dx_central(pressure_p);
    vx_p[1][n] = -kd*qnnn.dy_central(pressure_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -kd*qnnn.dz_central(pressure_p);
#endif
  };

  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    compute_velocity(n, vx_p);
  }

  foreach_dimension (dim) {
      VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);
  }

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);
    compute_velocity(n, vx_p);
  }

  foreach_dimension (dim) {
      VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);
  }

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(pressure, &pressure_p);

  // constant extend the velocities from interface to the entire domain
  my_p4est_level_set_t ls(&neighbors);
  Vec vx[P4EST_DIM];
  foreach_dimension (dim) {
    VecDuplicate(vx_tmp[dim], &vx[dim]);
    ls.extend_from_interface_to_whole_domain_TVD(phi, vx_tmp[dim], vx[dim]);
    VecDestroy(vx_tmp[dim]);
  }

  // smooth out the velocity field
  foreach_dimension (dim) VecGetArray(vx[dim], &vx_p[dim]);
  foreach_dimension (dim) VecRestoreArray(vx[dim], &vx_p[dim]);

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
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < 2*diag) {
#ifdef P4_TO_P8
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n])+SQR(vx_p[2][n]));
#else
      double vn = sqrt(SQR(vx_p[0][n])+SQR(vx_p[1][n]));
#endif
      vn_max  = MAX(vn_max, vn);
    }
  }
  foreach_dimension(dim) VecGetArray(vx[dim], &vx_p[dim]);

  double dt = MIN(cfl*dmin/vn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // advect the level-set and update the grid
  p4est_t* p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_np1->connectivity = conn;
  p4est_np1->user_pointer = sp;
  p4est_ghost_t* ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t* nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1,
                                &neighbors);

  // copy phi to phi_nm1 before updating
  Vec phi_np1;
  VecDuplicate(phi, &phi_np1);
  VecGhostCopy(phi, phi_np1);

  sl.update_p4est(vx, dt, phi_np1);

  // destroy old quantities and swap pointers
  // np1 --> n
  p4est_destroy(p4est); p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  VecDestroy(phi); phi = phi_np1;
  VecDestroy(pressure); VecDuplicate(phi, &pressure);

  return dt;
}

double one_fluid_solver_t::advect_interface_godunov(Vec &phi, Vec &pressure)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set_t ls(&neighbors);

  // compute normal and curvature
  compute_normal_and_curvature_diagonal(neighbors, phi);
  compute_normal_velocity_diagonal(neighbors, phi, pressure);

  // grid information
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double *phi_p, *nx_p[P4EST_DIM];
  VecGetArray(phi, &phi_p);
  foreach_dimension (dim) VecGetArray(nx[dim], &nx_p[dim]);

  double *un_p, *pressure_p;
  VecGetArray(pressure, &pressure_p);  

  // compute dt based on cfl number and curavture
  double un_max = 1; // minmum vn_max to be used when computing dt.
  double kun_max = 0;
  double *kappa_p;
  VecGetArray(kappa, &kappa_p);
  VecGetArray(un, &un_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < cfl*diag) {
      un_max  = MAX(un_max, un_p[n]);
      kun_max = MAX(kun_max, fabs(kappa_p[n]*un_p[n]));
    }
  }
  VecRestoreArray(kappa, &kappa_p);
  VecRestoreArray(un, &un_p);
  VecDestroy(kappa);

  double dt = MIN(cfl*dmin/un_max, 1.0/kun_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

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

    double x[P4EST_DIM];
    Vec un_np1;
    VecCreateGhostNodes(p4est, nodes, &un_np1);

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
  VecDestroy(pressure_nm1); pressure_nm1 = pressure;
  VecDestroy(un_nm1); un_nm1 = un;
  dt_nm1 = dt;

  // np1 --> n
  p4est = p4est_np1;
  ghost = ghost_np1;
  nodes = nodes_np1;
  phi = phi_np1;
  VecDuplicate(phi, &pressure);
  VecDuplicate(phi, &un);

  return dt;
}

double one_fluid_solver_t::advect_interface_diagonal(Vec &phi, Vec &pressure)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set_t ls(&neighbors);

  // compute normal and curvature
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, kappa_tmp);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // grid information
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double *phi_p, *n_p[P4EST_DIM];
  VecGetArray(phi, &phi_p);
  foreach_dimension (dim) VecGetArray(normal[dim], &n_p[dim]);

  // compute interface velocity
  Vec vn_tmp;
  VecCreateGhostNodes(p4est, nodes, &vn_tmp);
  double *vn_p, *pressure_p;
  VecGetArray(vn_tmp, &vn_p);
  VecGetArray(pressure, &pressure_p); 

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];    
  double diag_min = p4est_diag_min(p4est);
  my_p4est_interpolation_nodes_t interp1(&neighbors), interp2(&neighbors);
  interp1.set_input(pressure, linear);
  interp2.set_input(phi, linear);

  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);

    if (fabs(phi_p[n]) < 3*diag_min) {
      neighbors.get_neighbors(n, qnnn);
      node_xyz_fr_n(n, p4est, nodes, x);

      // inteprolate the data on a cube
      double f1[P4EST_CHILDREN], f2[P4EST_CHILDREN];
#ifdef P4_TO_P8
      for (short ck=0; ck<2; ck++)
#endif
        for (short cj=0; cj<2; cj++)
          for (short ci=0; ci<2; ci++) {
#ifdef P4_TO_P8
            f1[4*ck+2*cj+ci] = interp1(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                       x[1] + 0.5*(2*cj-1)*dxyz[1], 
                                       x[2] + 0.5*(2*ck-1)*dxyz[2]);

            f2[4*ck+2*cj+ci] = interp2(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                       x[1] + 0.5*(2*cj-1)*dxyz[1], 
                                       x[2] + 0.5*(2*ck-1)*dxyz[2]);
#else
            f1[2*cj+ci] = interp1(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                  x[1] + 0.5*(2*cj-1)*dxyz[1]);

            f2[2*cj+ci] = interp2(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                  x[1] + 0.5*(2*cj-1)*dxyz[1]);
#endif            
          }

#ifdef P4_TO_P8
      double kd  = (*K_D)(x[0], x[1], x[2]);
      double keo = (*K_EO)(x[0], x[1], x[2]);

      // NOTE: Diagonal method not used in 3d ...
      vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*n_p[0][n];
      vn_p[n] += -kd*qnnn.dy_central(pressure_p)*n_p[1][n];
      vn_p[n] += -kd*qnnn.dz_central(pressure_p)*n_p[2][n];
#else
      double kd  = (*K_D)(x[0], x[1]);      

      double ux  = -kd*((f1[3]-f1[2])/dxyz[0] + (f1[1]-f1[0])/dxyz[0])/2.0;
      double uy  = -kd*((f1[3]-f1[1])/dxyz[1] + (f1[2]-f1[0])/dxyz[1])/2.0;
      double ud1 = -kd*(f1[3]-f1[0])/diag_min;
      double ud2 = -kd*(f1[2]-f1[1])/diag_min;

      double nx  = ((f2[3]-f2[2])/dxyz[0] + (f2[1]-f2[0])/dxyz[0])/2.0;
      double ny  = ((f2[3]-f2[1])/dxyz[1] + (f2[2]-f2[0])/dxyz[1])/2.0;
      double abs = sqrt(nx*nx + ny*ny);
      if (abs > EPS) {
        nx /= abs;
        ny /= abs;
      } else {
        nx = 0;
        ny = 0;
      }

      double nd1 = (f2[3]-f2[0])/diag_min;
      double nd2 = (f2[2]-f2[1])/diag_min;
      abs = sqrt(nd1*nd1 + nd2*nd2);
      if (abs > EPS) {
        nd1 /= abs;
        nd2 /= abs;
      } else {
        nd1 = 0;
        nd2 = 0;
      }

      vn_p[n] = 0.5*(ux*nx+uy*ny + ud1*nd1+ud2*nd2);
#endif
      
    } else {
      vn_p[n] = 0;
    }
  }
  VecGhostUpdateBegin(vn_tmp, INSERT_VALUES, SCATTER_FORWARD);

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);

    if (fabs(phi_p[n]) < 3*diag_min) {
      neighbors.get_neighbors(n, qnnn);
      node_xyz_fr_n(n, p4est, nodes, x);

      // inteprolate the data on a cube
      double f1[P4EST_CHILDREN], f2[P4EST_CHILDREN];
#ifdef P4_TO_P8
      for (short ck=0; ck<2; ck++)
#endif
        for (short cj=0; cj<2; cj++)
          for (short ci=0; ci<2; ci++) {
#ifdef P4_TO_P8
            f1[4*ck+2*cj+ci] = interp1(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                       x[1] + 0.5*(2*cj-1)*dxyz[1], 
                                       x[2] + 0.5*(2*ck-1)*dxyz[2]);

            f2[4*ck+2*cj+ci] = interp2(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                       x[1] + 0.5*(2*cj-1)*dxyz[1], 
                                       x[2] + 0.5*(2*ck-1)*dxyz[2]);
#else
            f1[2*cj+ci] = interp1(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                  x[1] + 0.5*(2*cj-1)*dxyz[1]);

            f2[2*cj+ci] = interp2(x[0] + 0.5*(2*ci-1)*dxyz[0], 
                                  x[1] + 0.5*(2*cj-1)*dxyz[1]);
#endif            
          }

#ifdef P4_TO_P8
      double kd  = (*K_D)(x[0], x[1], x[2]);
      double keo = (*K_EO)(x[0], x[1], x[2]);

      // NOTE: Diagonal method not used in 3d ...
      vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*n_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*n_p[0][n];
      vn_p[n] += -kd*qnnn.dy_central(pressure_p)*n_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*n_p[1][n];
      vn_p[n] += -kd*qnnn.dz_central(pressure_p)*n_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*n_p[2][n];
#else
      double kd  = (*K_D)(x[0], x[1]);

      double ux  = -kd*((f1[3]-f1[2])/dxyz[0] + (f1[1]-f1[0])/dxyz[0])/2.0;
      double uy  = -kd*((f1[3]-f1[1])/dxyz[1] + (f1[2]-f1[0])/dxyz[1])/2.0;
      double ud1 = -kd*(f1[3]-f1[0])/diag_min;
      double ud2 = -kd*(f1[2]-f1[1])/diag_min;

      double nx  = ((f2[3]-f2[2])/dxyz[0] + (f2[1]-f2[0])/dxyz[0])/2.0;
      double ny  = ((f2[3]-f2[1])/dxyz[1] + (f2[2]-f2[0])/dxyz[1])/2.0;
      double abs = sqrt(nx*nx + ny*ny);
      if (abs > EPS) {
        nx /= abs;
        ny /= abs;
      } else {
        nx = 0;
        ny = 0;
      }

      double nd1 = (f2[3]-f2[0])/diag_min;
      double nd2 = (f2[2]-f2[1])/diag_min;
      abs = sqrt(nd1*nd1 + nd2*nd2);
      if (abs > EPS) {
        nd1 /= abs;
        nd2 /= abs;
      } else {
        nd1 = 0;
        nd2 = 0;
      }

      vn_p[n] = 0.5*(ux*nx+uy*ny + ud1*nd1+ud2*nd2);
#endif
      
    } else {
      vn_p[n] = 0;
    }

  }
  VecGhostUpdateEnd(vn_tmp, INSERT_VALUES, SCATTER_FORWARD);
  foreach_dimension(dim) {
    VecRestoreArray(normal[dim], &n_p[dim]);
    VecDestroy(normal[dim]);
  }
  VecRestoreArray(vn_tmp, &vn_p);
  VecRestoreArray(pressure, &pressure_p);

  // constant extend the normal velocity from interface to the entire domain
  Vec vn;
  VecDuplicate(vn_tmp, &vn);
  ls.extend_from_interface_to_whole_domain_TVD(phi, vn_tmp, vn);
  VecDestroy(vn_tmp);

  // compute dt based on cfl number and curavture
  double vn_max = 1; // minmum vn_max to be used when computing dt.
  double kvn_max = 0;
  double *kappa_p;
  VecGetArray(kappa, &kappa_p);
  VecGetArray(vn, &vn_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < cfl*diag) {
      vn_max  = MAX(vn_max, vn_p[n]);
      kvn_max = MAX(kvn_max, fabs(kappa_p[n]*vn_p[n]));
    }
  }
  VecRestoreArray(kappa, &kappa_p);
  VecRestoreArray(vn, &vn_p);
  VecDestroy(kappa);

  double dt_cfl = MIN(cfl*dmin/vn_max, 1.0/kvn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt_cfl, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  double dt = ls.advect_in_normal_direction(vn, phi, dt_cfl);

  VecDestroy(vn);

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
  Vec phi_np1, pressure_np1, potential_np1;
  VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1);
  VecDuplicate(phi_np1, &pressure_np1);
  VecDuplicate(phi_np1, &potential_np1);

  // create an interpolation function between two grids
  my_p4est_interpolation_nodes_t grid_interp(&neighbors);  
  foreach_node(n, nodes_np1) {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    grid_interp.add_point(n,x);
  }

  // interpolate variables
  grid_interp.set_input(phi, quadratic_non_oscillatory);
  grid_interp.interpolate(phi_np1);

  grid_interp.set_input(pressure, quadratic_non_oscillatory);
  grid_interp.interpolate(pressure_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi);        phi       = phi_np1;
  VecDestroy(pressure);   pressure  = pressure_np1;

  return dt;
}

double one_fluid_solver_t::advect_interface_normal(Vec &phi, Vec &pressure)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set_t ls(&neighbors);

  // compute normal and curvature
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, kappa_tmp);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // grid information
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double *phi_p, *n_p[P4EST_DIM];
  VecGetArray(phi, &phi_p);
  foreach_dimension (dim) VecGetArray(normal[dim], &n_p[dim]);

  // compute normal velocity based on projection
  my_p4est_interpolation_nodes_t interp(&neighbors);
  double xn[P4EST_DIM], x[P4EST_DIM];
  int ni = 0;
  foreach_node (n, nodes) {
    if (fabs(phi_p[n]) > 3*diag) continue;

    // project the point
    node_xyz_fr_n(n, p4est, nodes, xn);
    foreach_dimension (dim) xn[dim] -= phi_p[n]*n_p[dim][n];

    // add interpolation points
    foreach_dimension (dim) x[dim] = xn[dim] - 0.5*diag*n_p[dim][n];
    interp.add_point(ni, x);

    foreach_dimension (dim) x[dim] = xn[dim] + 0.5*diag*n_p[dim][n];
    interp.add_point(ni+1, x);

    ni += 2;
  }

  // interpolate values
  std::vector<double> pressure_normal(ni);
  interp.set_input(pressure, quadratic_non_oscillatory);
  interp.interpolate(pressure_normal.data());

  // compute normal velocity
  Vec vn;
  VecCreateGhostNodes(p4est, nodes, &vn);
  double *vn_p;
  VecGetArray(vn, &vn_p);

  ni = 0;
  foreach_node (n, nodes) {
    if (fabs(phi_p[n]) > 10*diag) {
      vn_p[n] = 0;
    } else {
      // project the point
      node_xyz_fr_n(n, p4est, nodes, xn);
      foreach_dimension (dim) xn[dim] -= phi_p[n]*n_p[dim][n];
#ifdef P4_TO_P8
      double kd  = (*K_D)(xn[0], xn[1], xn[2]);
#else
      double kd  = (*K_D)(xn[0], xn[1]);
#endif
      vn_p[n] = -kd*(pressure_normal[ni+1] - pressure_normal[ni])/diag;
      ni += 2;
    }
  }
  foreach_dimension(dim) {
    VecRestoreArray(normal[dim], &n_p[dim]);
    VecDestroy(normal[dim]);
  }

  // compute dt based on cfl number and curavture
  double vn_max = 1; // minmum vn_max to be used when computing dt.
  double kvn_max = 0;
  double *kappa_p;
  VecGetArray(kappa, &kappa_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < cfl*diag) {
      vn_max  = MAX(vn_max, vn_p[n]);
      kvn_max = MAX(kvn_max, fabs(kappa_p[n]*vn_p[n]));
    }
  }
  VecRestoreArray(kappa, &kappa_p);
  VecRestoreArray(vn, &vn_p);
  VecDestroy(kappa);

  double dt_cfl = MIN(cfl*dmin/vn_max, 1.0/kvn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt_cfl, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  double dt = ls.advect_in_normal_direction(vn, phi, dt_cfl);
  VecDestroy(vn);

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
  Vec phi_np1, pressure_np1, potential_np1;
  VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1);
  VecDuplicate(phi_np1, &pressure_np1);
  VecDuplicate(phi_np1, &potential_np1);

  // create an interpolation function between two grids
  my_p4est_interpolation_nodes_t grid_interp(&neighbors);  
  foreach_node(n, nodes_np1) {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    grid_interp.add_point(n,x);
  }

  // interpolate variables
  grid_interp.set_input(phi, quadratic_non_oscillatory);
  grid_interp.interpolate(phi_np1);

  grid_interp.set_input(pressure, quadratic_non_oscillatory);
  grid_interp.interpolate(pressure_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi);        phi       = phi_np1;
  VecDestroy(pressure);   pressure  = pressure_np1;

  return dt;
}

void one_fluid_solver_t::solve_field(double t, Vec phi, Vec pressure)
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
  Vec bc_val, bc_val_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &bc_val);
  VecDuplicate(phi, &bc_val_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, bc_val_tmp);
  foreach_dimension(dim) VecDestroy(normal[dim]);

  // extend curvature from interface to the entire domain
  ls.extend_from_interface_to_whole_domain_TVD(phi, bc_val_tmp, bc_val);
  VecDestroy(bc_val_tmp);

  // compute the boundary condition for the pressure. we use kappa to store the resutls
  double *bc_val_p;
  VecGetArray(bc_val, &bc_val_p);

  double x[P4EST_DIM];
//  double diag_min = p4est_diag_min(p4est);
  double *phi_p;
  VecGetArray(phi, &phi_p);
  foreach_node(n, nodes) {
    node_xyz_fr_n(n, p4est, nodes, x);
//    if (fabs(phi_p[n]) < 10*diag_min) {
    double kappa = bc_val_p[n];
//    kappa = CLAMP(kappa, -1.0/(2.0*diag_min), 1.0/(2.0*diag_min));
#ifdef P4_TO_P8
    bc_val_p[n] = kappa*(*gamma)(x[0], x[1], x[2]);
#else
    bc_val_p[n] = kappa*(*gamma)(x[0], x[1]);
#endif
//    } else {
//      double r = MAX(sqrt(SQR(x[0]) + SQR(x[1])), 0.1);
//      bc_val_p[n] = -(*Q)(t)/(2*PI) * log(r);
//    }
  }
  VecRestoreArray(phi, &phi_p);

  VecRestoreArray(bc_val, &bc_val_p);

  Vec K;
  VecCreateGhostNodes(p4est, nodes, &K);
  // solve for pressure
  {
    // Set the boundary conditions
    my_p4est_interpolation_nodes_t p_interface_value(&neighbors);

    Vec Fxx, Fyy;
    VecCreateGhostNodes(p4est, nodes, &Fxx);
    VecCreateGhostNodes(p4est, nodes, &Fyy);
    neighbors.second_derivatives_central(bc_val, Fxx, Fyy);
    p_interface_value.set_input(bc_val, Fxx, Fyy, quadratic);

#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    bc.setInterfaceType(DIRICHLET);
    bc.setInterfaceValue(p_interface_value);
    bc.setWallTypes(*bc_wall_type);
    bc.setWallValues(*bc_wall_value);
    bc_wall_value->t = t;

    my_p4est_poisson_nodes_t poisson(&neighbors);
    poisson.set_phi(phi);
    poisson.set_bc(bc);
//    sample_cf_on_nodes(p4est, nodes, *K_D, K);
//    poisson.set_mu(K);
    poisson.set_mu(1.0);
    poisson.solve(pressure);

    ls.extend_Over_Interface_TVD(phi, pressure);

    VecDestroy(Fxx);
    VecDestroy(Fyy);
  }

  // destroy uneeded objects
  VecDestroy(bc_val);
  VecDestroy(K);
}

double one_fluid_solver_t::solve_one_step(double t, Vec &phi, Vec &pressure)
{

//  // advect the interface
  double dt = 0;
//  parStopWatch w;
  if (method == "semi_lagrangian") {
//    w.start("Advecting usign semi-Lagrangian method");
    dt = advect_interface_semi_lagrangian(phi, pressure);
//    w.stop(); w.read_duration();
  } else if (method == "godunov") {
//    w.start("Advecting usign Godunov method");
    dt = advect_interface_godunov(phi, pressure);
//    w.stop(); w.read_duration();
  } else if (method == "normal") {
//    w.start("Advecting usign normal-velocity method");
    dt = advect_interface_normal(phi, pressure);
//    w.stop(); w.read_duration();
  } else if (method == "diagonal") {
//    w.start("Advecting usign diagonal-velocity method");
    dt = advect_interface_diagonal(phi, pressure);
//    w.stop(); w.read_duration();
  } else {
    throw std::invalid_argument("invalid advection method. Valid options are:\n"
                                " (a) semi_lagrangian,\n"
                                " (b) godunov,\n"
                                " (c) normal.\n");
  }

  // solve for the pressure
//  w.start("Solving for the field variables");
  solve_field(t+dt,phi, pressure);
//  w.stop(); w.read_duration();

  return dt;

}


