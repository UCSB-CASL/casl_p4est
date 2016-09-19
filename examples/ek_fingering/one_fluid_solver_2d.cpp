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
}

one_fluid_solver_t::~one_fluid_solver_t()
{
  VecDestroy(kappa);
  foreach_dimension (dim) {
    VecDestroy(nx[dim]);
    VecDestroy(n1[dim]);
  }
  VecDestroy(un);
}

void one_fluid_solver_t::set_properties(cf_t &K_D, cf_t &K_EO, cf_t &gamma)
{  
  this->K_D   = &K_D;
  this->K_EO  = &K_EO;
  this->gamma = &gamma;
}

void one_fluid_solver_t::set_injection_rates(CF_1 &Q, CF_1 &I, double alpha)
{
  this->Q     = &Q;
  this->I     = &I;
  this->alpha = alpha;
}

void one_fluid_solver_t::set_bc_wall(bc_wall_t &bc_wall_type, cf_t &bc_wall_value)
{
  this->bc_wall_type  = &bc_wall_type;
  this->bc_wall_value = &bc_wall_value;
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
  interp.set_input(phi, quadratic);

  double f[3][3];
  double x[P4EST_DIM];
  foreach_node (n, nodes) {
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

  VecRestoreArray(kappa_tmp, &kappa_p);
  VecRestoreArray(phi, &phi_p);
  foreach_dimension (dim) {
    VecRestoreArray(nx[dim], &nx_p[dim]);
    VecRestoreArray(n1[dim], &n1_p[dim]);
  }

  // extend curvature
  my_p4est_level_set_t ls(&neighbors);
  ls.extend_from_interface_to_whole_domain(phi, kappa_tmp, kappa);

  VecDestroy(kappa_tmp);
}

void one_fluid_solver_t::compute_normal_velocity_diagonal(my_p4est_node_neighbors_t& neighbors, Vec& phi, Vec &pressure) {
  VecDestroy(un); VecDuplicate(pressure, &un);

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
  interp.set_input(pressure, quadratic);

  double f[3][3];
  double x[P4EST_DIM];
  foreach_node (n, nodes) {
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
      un_p[n] = -kd*(nx_p[0][n]*fx + nx_p[1][n]*fy + n1_p[0][n]*f1 + n1_p[1][n]*f2)/2.0;
    }
  }

  VecRestoreArray(un_tmp, &un_p);
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

double one_fluid_solver_t::advect_interface_semi_lagrangian(Vec &phi, Vec &pressure, Vec& potential, double cfl, double dtmax)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // compute interface velocity
  Vec vx_tmp[P4EST_DIM];
  double *vx_p[P4EST_DIM], *pressure_p, *phi_p, *potential_p;
  foreach_dimension(dim) {
    VecCreateGhostNodes(p4est, nodes, &vx_tmp[dim]);
    VecGetArray(vx_tmp[dim], &vx_p[dim]);
  }
  VecGetArray(pressure, &pressure_p);
  VecGetArray(potential, &potential_p);
  VecGetArray(phi, &phi_p);

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];
  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double kd  = (*K_D)(x[0], x[1], x[2]);
    double keo = (*K_EO)(x[0], x[1], x[2]);
#else
    double kd  = (*K_D)(x[0], x[1]);
    double keo = (*K_EO)(x[0], x[1]);
#endif
    vx_p[0][n] = -kd*qnnn.dx_central(pressure_p) - alpha*keo*qnnn.dx_central(potential_p);
    vx_p[1][n] = -kd*qnnn.dy_central(pressure_p) - alpha*keo*qnnn.dy_central(potential_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -kd*qnnn.dz_central(pressure_p) - alpha*keo*qnnn.dz_central(potential_p);
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double kd  = (*K_D)(x[0], x[1], x[2]);
    double keo = (*K_EO)(x[0], x[1], x[2]);
#else
    double kd  = (*K_D)(x[0], x[1]);
    double keo = (*K_EO)(x[0], x[1]);
#endif
    vx_p[0][n] = -kd*qnnn.dx_central(pressure_p) - alpha*keo*qnnn.dx_central(potential_p);
    vx_p[1][n] = -kd*qnnn.dy_central(pressure_p) - alpha*keo*qnnn.dy_central(potential_p);
#ifdef P4_TO_P8
    vx_p[2][n] = -kd*qnnn.dz_central(pressure_p) - alpha*keo*qnnn.dz_central(potential_p);
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);  

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(pressure, &pressure_p);
  VecRestoreArray(potential, &potential_p);

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
  if (0)
  {
    double dxyz[P4EST_DIM];
    p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
    double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
    double dmin = MIN(dxyz[0], dxyz[1]);
#endif
    double dt = 0.25*dmin*dmin;

    Vec tmp;
    VecDuplicate(phi, &tmp);
    double *tmp_p;

    quad_neighbor_nodes_of_node_t qnnn;
    VecGetArray(tmp, &tmp_p);

    // vx
    int itmax = 0;
    for (int it = 0; it < itmax; it++) {
      foreach_node (n, nodes) {
        neighbors.get_neighbors(n, qnnn);
        double fxx, fyy;
        qnnn.laplace(vx_p[0], fxx, fyy);
        tmp_p[n] = vx_p[0][n] + dt*(fxx+fyy);
      }
      VecCopy(tmp, vx[0]);
    }

    // vy
    for (int it = 0; it < itmax; it++) {
      foreach_node (n, nodes) {
        neighbors.get_neighbors(n, qnnn);
        double fxx, fyy;
        qnnn.laplace(vx_p[1], fxx, fyy);
        tmp_p[n] = vx_p[1][n] + dt*(fxx+fyy);
      }
      VecCopy(tmp, vx[1]);
    }

    VecRestoreArray(tmp, &tmp_p);
    VecDestroy(tmp);
  }
  foreach_dimension (dim) VecRestoreArray(vx[dim], &vx_p[dim]);

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

  double dt_cfl = MIN(cfl*dmin/vn_max, 1.0/kvn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt_cfl, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // FIXME: This is stupid! Change the solver to take a single Ca
//  double Ca = DBL_MAX;
//  foreach_node (n,nodes) {
//    node_xyz_fr_n(n, p4est, nodes, x);
//    #ifdef P4_TO_P8
//      Ca = MIN(Ca, 1.0/(*gamma)(x[0], x[1], x[2]));
//    #else
//      Ca = MIN(Ca, 1.0/(*gamma)(x[0], x[1]));
//    #endif
//  }

//  double dt = MIN(Ca*dmin*dmin*dmin/PI/PI/PI, dt_cfl);
//  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  double dt = dt_cfl;

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
  VecDestroy(potential);
  VecDuplicate(phi, &pressure);
  VecDuplicate(phi, &potential);

  foreach_dimension(dim) VecDestroy(vx[dim]);

  return dt;
}

double one_fluid_solver_t::advect_interface_godunov(Vec &phi, Vec &pressure, Vec& potential, double cfl, double dtmax)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set_t ls(&neighbors);

  // compute normal and curvature
  compute_normal_and_curvature_diagonal(neighbors, phi);
  compute_normal_velocity_diagonal(neighbors, phi, pressure);

//  Vec kappa, kappa_tmp, normal[P4EST_DIM];
//  VecDuplicate(phi, &kappa);
//  VecDuplicate(phi, &kappa_tmp);
//  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
//  compute_normals(neighbors, phi, normal);
//  compute_mean_curvature(neighbors, normal, kappa_tmp);
//  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
//  VecDestroy(kappa_tmp);

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

  // compute interface velocity
//  Vec vn_tmp;
//  VecCreateGhostNodes(p4est, nodes, &vn_tmp);

  double *un_p, *pressure_p, *potential_p;
//  VecGetArray(vn_tmp, &un_p);
  VecGetArray(pressure, &pressure_p);
  VecGetArray(potential, &potential_p);

  // compute on the layer nodes
//  quad_neighbor_nodes_of_node_t qnnn;
//  double x[P4EST_DIM];
//  for (size_t i=0; i<neighbors.get_layer_size(); i++){
//    p4est_locidx_t n = neighbors.get_layer_node(i);
//    neighbors.get_neighbors(n, qnnn);
//    node_xyz_fr_n(n, p4est, nodes, x);
//#ifdef P4_TO_P8
//    double kd  = (*K_D)(x[0], x[1], x[2]);
//    double keo = (*K_EO)(x[0], x[1], x[2]);
//#else
//    double kd  = (*K_D)(x[0], x[1]);
//    double keo = (*K_D)(x[0], x[1]);
//#endif

//    vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*nx_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*nx_p[0][n];
//    vn_p[n] += -kd*qnnn.dy_central(pressure_p)*nx_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*nx_p[1][n];
//#ifdef P4_TO_P8
//    vn_p[n] += -kd*qnnn.dz_central(pressure_p)*nx_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*nx_p[2][n];
//#endif
//  }
//  VecGhostUpdateBegin(vn_tmp, INSERT_VALUES, SCATTER_FORWARD);

//  // compute on the local nodes
//  for (size_t i=0; i<neighbors.get_local_size(); i++){
//    p4est_locidx_t n = neighbors.get_local_node(i);
//    neighbors.get_neighbors(n, qnnn);
//    node_xyz_fr_n(n, p4est, nodes, x);

//#ifdef P4_TO_P8
//    double kd  = (*K_D)(x[0], x[1], x[2]);
//    double keo = (*K_EO)(x[0], x[1], x[2]);
//#else
//    double kd  = (*K_D)(x[0], x[1]);
//    double keo = (*K_D)(x[0], x[1]);
//#endif

//    vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*nx_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*nx_p[0][n];
//    vn_p[n] += -kd*qnnn.dy_central(pressure_p)*nx_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*nx_p[1][n];
//#ifdef P4_TO_P8
//    vn_p[n] += -kd*qnnn.dz_central(pressure_p)*nx_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*nx_p[2][n];
//#endif
//  }

//  VecGhostUpdateEnd(vn_tmp, INSERT_VALUES, SCATTER_FORWARD);
//  foreach_dimension(dim) {
//    VecRestoreArray(normal[dim], &nx_p[dim]);
//    VecDestroy(normal[dim]);
//  }
//  VecRestoreArray(vn_tmp, &vn_p);
//  VecRestoreArray(pressure, &pressure_p);

//  // constant extend the normal velocity from interface to the entire domain
//  Vec vn;
//  VecDuplicate(vn_tmp, &vn);
//  ls.extend_from_interface_to_whole_domain_TVD(phi, vn_tmp, vn);
//  VecDestroy(vn_tmp);

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

  double dt_cfl = MIN(cfl*dmin/un_max, 1.0/kun_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt_cfl, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  double dt = ls.advect_in_normal_direction(un, phi, dt_cfl);

//  VecDestroy(vn);

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
  double x[P4EST_DIM];
  foreach_node(n, nodes_np1) {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    grid_interp.add_point(n,x);
  }

  // interpolate variables
  grid_interp.set_input(phi, quadratic_non_oscillatory);
  grid_interp.interpolate(phi_np1);

  grid_interp.set_input(pressure, quadratic_non_oscillatory);
  grid_interp.interpolate(pressure_np1);

  grid_interp.set_input(potential, quadratic_non_oscillatory);
  grid_interp.interpolate(potential_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi);        phi       = phi_np1;
  VecDestroy(pressure);   pressure  = pressure_np1;
  VecDestroy(potential);  potential = potential_np1;

  return dt;
}

double one_fluid_solver_t::advect_interface_diagonal(Vec &phi, Vec &pressure, Vec& potential, double cfl, double dtmax)
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
  double *vn_p, *pressure_p, *potential_p;
  VecGetArray(vn_tmp, &vn_p);
  VecGetArray(pressure, &pressure_p);
  VecGetArray(potential, &potential_p);

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
      vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*n_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*n_p[0][n];
      vn_p[n] += -kd*qnnn.dy_central(pressure_p)*n_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*n_p[1][n];
      vn_p[n] += -kd*qnnn.dz_central(pressure_p)*n_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*n_p[2][n];
#else
      double kd  = (*K_D)(x[0], x[1]);
      double keo = (*K_D)(x[0], x[1]);

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
      double keo = (*K_D)(x[0], x[1]);

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

  grid_interp.set_input(potential, quadratic_non_oscillatory);
  grid_interp.interpolate(potential_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi);        phi       = phi_np1;
  VecDestroy(pressure);   pressure  = pressure_np1;
  VecDestroy(potential);  potential = potential_np1;

  return dt;
}

double one_fluid_solver_t::advect_interface_normal(Vec &phi, Vec &pressure, Vec& potential, double cfl, double dtmax)
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

  std::vector<double> potential_normal(ni);
  interp.set_input(potential, quadratic_non_oscillatory);
  interp.interpolate(potential_normal.data());

  // compute normal velocity
  Vec vn;
  VecCreateGhostNodes(p4est, nodes, &vn);
  double *vn_p;
  VecGetArray(vn, &vn_p);

  ni = 0;
  foreach_node (n, nodes) {
    if (fabs(phi_p[n]) > 3*diag) {
      vn_p[n] = 0;
    } else {
      // project the point
      node_xyz_fr_n(n, p4est, nodes, xn);
      foreach_dimension (dim) xn[dim] -= phi_p[n]*n_p[dim][n];
#ifdef P4_TO_P8
      double kd  = (*K_D)(xn[0], xn[1], xn[2]);
      double keo = (*K_EO)(xn[0], xn[1], xn[2]);
#else
      double kd  = (*K_D)(xn[0], xn[1]);
      double keo = (*K_EO)(xn[0], xn[0]);
#endif
      vn_p[n] = -kd*(pressure_normal[ni+1] - pressure_normal[ni])/diag
                -alpha*keo*(potential_normal[ni+1] - potential_normal[ni])/diag ;
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

  grid_interp.set_input(potential, quadratic_non_oscillatory);
  grid_interp.interpolate(potential_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi);        phi       = phi_np1;
  VecDestroy(pressure);   pressure  = pressure_np1;
  VecDestroy(potential);  potential = potential_np1;

  return dt;
}

void one_fluid_solver_t::solve_fields(double t, Vec phi, Vec pressure, Vec potential)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // smooth out the levelset function
  if (0)
  {
    double dxyz[P4EST_DIM];
    p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
    double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
    double dmin = MIN(dxyz[0], dxyz[1]);
#endif
    double dt = 0.25*dmin*dmin;

    Vec tmp;
    VecDuplicate(phi, &tmp);
    double *phi_p, *tmp_p;

    quad_neighbor_nodes_of_node_t qnnn;
    int itmax = 0;
    for (int it = 0; it < itmax; it++) {
      VecGetArray(phi, &phi_p);
      VecGetArray(tmp, &tmp_p);

      foreach_node (n, nodes) {
        neighbors.get_neighbors(n, qnnn);
        double fxx, fyy;
        qnnn.laplace(phi_p, fxx, fyy);
        tmp_p[n] = phi_p[n] + dt*(fxx+fyy);
      }
      VecRestoreArray(phi, &phi_p);
      VecRestoreArray(tmp, &tmp_p);

      VecCopy(tmp, phi);
    }
    VecDestroy(tmp);
  }

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

  // smooth out the boundary condition
//  if (0)
//  {
//    double dxyz[P4EST_DIM];
//    p4est_dxyz_min(p4est, dxyz);
//#ifdef P4_TO_P8
//    double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
//#else
//    double dmin = MIN(dxyz[0], dxyz[1]);
//#endif
//    double dt = 0.25*dmin*dmin;

//    Vec tmp;
//    VecDuplicate(phi, &tmp);
//    double *tmp_p;

//    quad_neighbor_nodes_of_node_t qnnn;
//    VecGetArray(tmp, &tmp_p);
//    int itmax = 10;
//    for (int it = 0; it < itmax; it++) {
//      foreach_node (n, nodes) {
//        neighbors.get_neighbors(n, qnnn);
//        double fxx, fyy;
//        qnnn.laplace(bc_val_p, fxx, fyy);
//        tmp_p[n] = bc_val_p[n] + dt*(fxx+fyy);
//      }
//      VecCopy(tmp, bc_val);
//    }
//    VecRestoreArray(tmp, &tmp_p);
//    VecDestroy(tmp);
//  }

  VecRestoreArray(bc_val, &bc_val_p);

  Vec K;
  VecCreateGhostNodes(p4est, nodes, &K);
  // solve for pressure
  {
    // define a new levelset for the outer ring boundary
//    Vec phi_ring;
//    VecDuplicate(phi, &phi_ring);
//    double *phi_ring_p;
//    VecGetArray(phi_ring, &phi_ring_p);
//    VecGetArray(phi, &phi_p);
//    double xyz_max[P4EST_DIM];
//    p4est_xyz_max(p4est, xyz_max);
//    foreach_node (n, nodes) {
//      node_xyz_fr_n(n, p4est, nodes, x);
//      phi_ring_p[n] = MAX(phi_p[n], sqrt(SQR(x[0]) + SQR(x[1])) - 0.99*xyz_max[0]);
//    }
//    VecRestoreArray(phi, &phi_p);
//    VecRestoreArray(phi_ring, &phi_ring_p);

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

//    ls.extend_Over_Interface_TVD(phi_ring, pressure);
    ls.extend_Over_Interface_TVD(phi, pressure);
    // Vec l1,l2;
    // VecGhostGetLocalForm(pressure, &l1);
    // VecGhostGetLocalForm(bc_val, &l2);
    // VecCopy(l2, l1);
    // VecGhostRestoreLocalForm(pressure, &l1);
    // VecGhostRestoreLocalForm(bc_val, &l2);
//    VecDestroy(phi_ring);

    VecDestroy(Fxx);
    VecDestroy(Fyy);
  }

  // solve for the potential
//  if (alpha > 0)
//  {
//    VecGetArray(bc_val, &bc_val_p);
//    // Set the boundary condition
//    double x[P4EST_DIM];
//    foreach_node(n, nodes) {
//      node_xyz_fr_n(n, p4est, nodes, x);
//#ifdef P4_TO_P8
//      double r = MAX(sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2])), EPS);
//      bc_val_p[n] = (*I)(t)/(4*PI*r);
//#else
//      double r = MAX(sqrt(SQR(x[0]) + SQR(x[1])), EPS);
//      bc_val_p[n] = (*I)(t)/(2*PI) * log(r);
//#endif
//    }

//    my_p4est_interpolation_nodes_t p_interface_value(&neighbors);
//    p_interface_value.set_input(bc_val, linear);

//#ifdef P4_TO_P8
//    BoundaryConditions3D bc;
//#else
//    BoundaryConditions2D bc;
//#endif
//    bc.setInterfaceType(DIRICHLET);
//    bc.setInterfaceValue(p_interface_value);
//    bc.setWallTypes(*bc_wall_type);
//    bc.setWallValues(*bc_wall_value);
//    bc_wall_value->t = t;

//    // invert the domain
//    Vec phi_l;
//    VecGhostGetLocalForm(phi, &phi_l);
//    VecScale(phi_l, -1);

//    my_p4est_poisson_nodes_t poisson(&neighbors);
//    poisson.set_phi(phi);
//    poisson.set_bc(bc);
////    sample_cf_on_nodes(p4est, nodes, *K_EO, K);
////    poisson.set_mu(K);
//    poisson.set_mu(1.0);
//    poisson.solve(potential, true);

//    ls.extend_Over_Interface_TVD(phi, potential);

//    VecScale(phi_l, -1);
//    VecGhostRestoreLocalForm(phi, &phi_l);

//    // add the regular and singular parts
//    double *potential_p;
//    VecGetArray(potential, &potential_p);
//    foreach_node(n, nodes) {
//      potential_p[n] -= bc_val_p[n];
//    }

//    VecRestoreArray(bc_val, &bc_val_p);
//    VecRestoreArray(potential, &potential_p);
//  }

  // destroy uneeded objects
  VecDestroy(bc_val);
  VecDestroy(K);
}

double one_fluid_solver_t::solve_one_step(double t, Vec &phi, Vec &pressure, Vec& potential, const std::string& method, double cfl, double dtmax)
{
  // advect the interface
  double dt;
  parStopWatch w;
  if (method == "semi_lagrangian") {
//    w.start("Advecting usign semi-Lagrangian method");
    dt = advect_interface_semi_lagrangian(phi, pressure, potential, cfl, dtmax);
//    w.stop(); w.read_duration();
  } else if (method == "godunov") {
//    w.start("Advecting usign Godunov method");
    dt = advect_interface_godunov(phi, pressure, potential, cfl, dtmax);
//    w.stop(); w.read_duration();
  } else if (method == "normal") {
//    w.start("Advecting usign normal-velocity method");
    dt = advect_interface_normal(phi, pressure, potential, cfl, dtmax);
//    w.stop(); w.read_duration();
  } else if (method == "diagonal") {
//    w.start("Advecting usign diagonal-velocity method");
    dt = advect_interface_diagonal(phi, pressure, potential, cfl, dtmax);
//    w.stop(); w.read_duration();
  } else {
    throw std::invalid_argument("invalid advection method. Valid options are:\n"
                                " (a) semi_lagrangian,\n"
                                " (b) godunov,\n"
                                " (c) normal.\n");
  }

  // solve for the pressure
//  w.start("Solving for the field variables");
  solve_fields(t+dt,phi, pressure, potential);
//  w.stop(); w.read_duration();

  return dt;
}


