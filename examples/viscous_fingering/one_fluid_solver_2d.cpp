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

#include <src/CASL_math.h>

one_fluid_solver_t::one_fluid_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick)
  : p4est(p4est), ghost(ghost), nodes(nodes), brick(&brick)
{
  sp   = (splitting_criteria_t*) p4est->user_pointer;
  conn = p4est->connectivity;
}

one_fluid_solver_t::~one_fluid_solver_t() {}

void one_fluid_solver_t::set_properties(cf_t &K_D, cf_t &K_EO, cf_t &gamma)
{  
  this->K_D   = &K_D;
  this->K_EO  = &K_EO;
  this->gamma = &gamma;
}

void one_fluid_solver_t::set_injection_rates(cf_t &Q, cf_t &I, double alpha)
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
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(neighbors, phi, normal);
  compute_mean_curvature(neighbors, normal, kappa_tmp);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // compute interface velocity
  Vec vn_tmp;
  double *vn_p, *pressure_p, *phi_p, *potential_p;
  VecCreateGhostNodes(p4est, nodes, &vn_tmp);
  VecGetArray(vn_tmp, &vn_p);
  VecGetArray(pressure, &pressure_p);
  VecGetArray(potential, &potential_p);
  VecGetArray(phi, &phi_p);

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];
  double *n_p[P4EST_DIM];
  foreach_dimension(dim) VecGetArray(normal[dim], &n_p[dim]);
  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8    
    double kd  = (*K_D)(x[0], x[1], x[2]);
    double keo = (*K_EO)(x[0], x[1], x[2]);
#else
    double kd  = (*K_D)(x[0], x[1]);
    double keo = (*K_D)(x[0], x[1]);
#endif

    vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*n_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*n_p[0][n];
    vn_p[n] += -kd*qnnn.dy_central(pressure_p)*n_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*n_p[1][n];
#ifdef P4_TO_P8
    vn_p[n] += -kd*qnnn.dz_central(pressure_p)*n_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*n_p[2][n];
#endif
  }
  VecGhostUpdateBegin(vn_tmp, INSERT_VALUES, SCATTER_FORWARD);

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
    double keo = (*K_D)(x[0], x[1]);
#endif

    vn_p[n]  = -kd*qnnn.dx_central(pressure_p)*n_p[0][n] - alpha*keo*qnnn.dx_central(potential_p)*n_p[0][n];
    vn_p[n] += -kd*qnnn.dy_central(pressure_p)*n_p[1][n] - alpha*keo*qnnn.dy_central(potential_p)*n_p[1][n];
#ifdef P4_TO_P8
    vn_p[n] += -kd*qnnn.dz_central(pressure_p)*n_p[2][n] - alpha*keo*qnnn.dz_central(potential_p)*n_p[2][n];
#endif
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
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
  double dmin = MIN(dxyz[0], MIN(dxyz[1], dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
  double dmin = MIN(dxyz[0], dxyz[1]);
#endif

  double vn_max = 1; // minmum vn_max to be used when computing dt.
  double kvn_max = 0;
  double *kappa_p;
  VecGetArray(kappa, &kappa_p);
  VecGetArray(vn, &vn_p);
  foreach_node(n, nodes) {
    if (fabs(phi_p[n]) < 2*diag) {
      vn_max  = MAX(vn_max, vn_p[n]);
      kvn_max = MAX(kvn_max, fabs(kappa_p[n]*vn_p[n]));
    }
  }
  VecRestoreArray(kappa, &kappa_p);
  VecRestoreArray(vn, &vn_p);
  VecDestroy(kappa);

  double dt = MIN(cfl*dmin/vn_max, 1.0/kvn_max, dtmax);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  // advect the level-set and update the grid
  dt = ls.advect_in_normal_direction(vn, phi, dt);

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

  my_p4est_interpolation_nodes_t phi_i(&neighbors);
  phi_i.set_input(phi, quadratic_non_oscillatory);
  foreach_node(n, nodes_np1) {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, x);
    phi_i.add_point(n,x);
  }
  phi_i.interpolate(phi_np1);

  // destroy old quantities and swap pointers
  p4est_destroy(p4est);       p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;

  VecDestroy(phi); phi = phi_np1;
  VecDestroy(pressure); VecDuplicate(phi, &pressure);
  VecDestroy(potential); VecDuplicate(phi, &potential);

  return dt;
}

void one_fluid_solver_t::solve_fields(double t, Vec phi, Vec pressure, Vec potential)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // reinitialize the levelset
  my_p4est_level_set_t ls(&neighbors);
  ls.reinitialize_2nd_order(phi);
  ls.perturb_level_set_function(phi, EPS);

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
  foreach_node(n, nodes) {
    node_xyz_fr_n(n, p4est, nodes, x);
    double kappa = bc_val_p[n];
//    kappa = CLAMP(kappa, -1.0/(2.0*diag_min), 1.0/(2.0*diag_min));
#ifdef P4_TO_P8
    bc_val_p[n] = kappa*(*gamma)(x[0], x[1], x[2]);
#else
    bc_val_p[n] = kappa*(*gamma)(x[0], x[1]);
#endif
  }
  VecRestoreArray(bc_val, &bc_val_p);

  Vec K;
  VecCreateGhostNodes(p4est, nodes, &K);
  // solve for pressure
  {
    // Set the boundary conditions
    my_p4est_interpolation_nodes_t p_interface_value(&neighbors);
    p_interface_value.set_input(bc_val, linear);

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
  }

  // solve for the potential
  {
    // Set the boundary condition
    I->t = t;
    double x[P4EST_DIM];
    foreach_node(n, nodes) {
      node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
      double r = MAX(sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2])), EPS);
      bc_val_p[n] = (*I)(x[0],x[1],x[2])/(4*PI*r);
#else
      double r = MAX(sqrt(SQR(x[0]) + SQR(x[1])), EPS);
      bc_val_p[n] = (*I)(x[0],x[1])/(2*PI) * log(r);
#endif
    }

    my_p4est_interpolation_nodes_t p_interface_value(&neighbors);
    p_interface_value.set_input(bc_val, linear);

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

    // invert the domain
    Vec phi_l;
    VecGhostGetLocalForm(phi, &phi_l);
    VecScale(phi_l, -1);

    my_p4est_poisson_nodes_t poisson(&neighbors);
    poisson.set_phi(phi);
    poisson.set_bc(bc);
//    sample_cf_on_nodes(p4est, nodes, *K_EO, K);
//    poisson.set_mu(K);
    poisson.set_mu(1.0);
    poisson.solve(potential);

    ls.extend_Over_Interface_TVD(phi, potential);

    VecScale(phi_l, -1);
    VecGhostRestoreLocalForm(phi, &phi_l);

    // add the regular and singular parts
    double *potential_p;
    VecGetArray(potential, &potential_p);
    foreach_node(n, nodes) {
      potential_p[n] -= bc_val_p[n];
    }

    VecRestoreArray(bc_val, &bc_val_p);
    VecRestoreArray(potential, &potential_p);
  }

  // destroy uneeded objects
  VecDestroy(bc_val);
  VecDestroy(K);
}


double one_fluid_solver_t::solve_one_step(double t, Vec &phi, Vec &pressure, const std::string& method, double cfl, double dtmax)
{
  // advect the interface
  double dt;
  if (method == "semi_lagrangian")
    dt = advect_interface_semi_lagrangian(phi, pressure, cfl, dtmax);
  else if (method == "godunov")
    dt = advect_interface_godunov(phi, pressure, cfl, dtmax);
  else
    throw std::invalid_argument("invalid advection method. Valid options are:\n"
                                "(a) semi_lagrangian, or\n"
                                "(b) godunov.");

  // solve for the pressure
  solve_pressure(t+dt,phi, pressure);

  return dt;
}


