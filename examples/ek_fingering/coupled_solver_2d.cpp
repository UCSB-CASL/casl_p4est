#ifdef P4_TO_P8
#include "coupled_solver_3d.h"
#include <src/my_p8est_poisson_jump_voronoi_block.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#else
#include "coupled_solver_2d.h"
#include <src/my_p4est_poisson_jump_voronoi_block.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/casl_math.h>

coupled_solver_t::coupled_solver_t(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t& brick)
  : p4est(p4est), ghost(ghost), nodes(nodes), brick(&brick)
{
  sp   = (splitting_criteria_t*) p4est->user_pointer;
  conn = p4est->connectivity;
}

coupled_solver_t::~coupled_solver_t() {}

void coupled_solver_t::set_parameters(const parameters& p)
{    
  params = p;
}

void coupled_solver_t::set_boundary_conditions(wall_bc_t& pressure_bc_type, cf_t& pressure_bc_val,
                                               wall_bc_t& potential_bc_type, cf_t& potential_bc_val)
{
  this->pressure_bc_type  = &pressure_bc_type;
  this->pressure_bc_val   = &pressure_bc_val;
  this->potential_bc_type = &potential_bc_type;
  this->potential_bc_val  = &potential_bc_val;
}

void coupled_solver_t::set_injection_rates(const CF_1 &Q, const CF_1 &I)
{
  this->Q = &Q;
  this->I = &I;
}

double coupled_solver_t::advect_interface(Vec &phi,
                                          Vec &pressure_m,  Vec &pressure_p,
                                          Vec &potential_m, Vec &potential_p,
                                          double cfl, double dtmax)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();

  // compute interface velocity
  Vec vx_tmp[P4EST_DIM];
  double *vx_p[P4EST_DIM];
  double *pressure_p_p, *pressure_m_p, *potential_p_p, *potential_m_p;
  double *phi_p;
  foreach_dimension(dim) {
    VecCreateGhostNodes(p4est, nodes, &vx_tmp[dim]);
    VecGetArray(vx_tmp[dim], &vx_p[dim]);
  }
  VecGetArray(pressure_m,  &pressure_m_p);
  VecGetArray(pressure_p,  &pressure_p_p);
  VecGetArray(potential_m, &potential_m_p);
  VecGetArray(potential_p, &potential_p_p);
  VecGetArray(phi, &phi_p);

  // compute on the layer nodes
  quad_neighbor_nodes_of_node_t qnnn;
  double x[P4EST_DIM];
  for (size_t i=0; i<neighbors.get_layer_size(); i++){
    p4est_locidx_t n = neighbors.get_layer_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);

    vx_p[0][n] = -(qnnn.dx_central(pressure_p_p) + params.alpha*qnnn.dx_central(potential_p_p));
    vx_p[1][n] = -(qnnn.dy_central(pressure_p_p) + params.alpha*qnnn.dy_central(potential_p_p));
#ifdef P4_TO_P8
    vx_p[2][n] = -(qnnn.dz_central(pressure_p_p) + params.alpha*qnnn.dz_central(potential_p_p));
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateBegin(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // compute on the local nodes
  for (size_t i=0; i<neighbors.get_local_size(); i++){
    p4est_locidx_t n = neighbors.get_local_node(i);
    neighbors.get_neighbors(n, qnnn);
    node_xyz_fr_n(n, p4est, nodes, x);

    vx_p[0][n] = -(qnnn.dx_central(pressure_p_p) + params.alpha*qnnn.dx_central(potential_p_p));
    vx_p[1][n] = -(qnnn.dy_central(pressure_p_p) + params.alpha*qnnn.dy_central(potential_p_p));
#ifdef P4_TO_P8
    vx_p[2][n] = -(qnnn.dz_central(pressure_p_p) + params.alpha*qnnn.dz_central(potential_p_p));
#endif
  }
  foreach_dimension(dim)
      VecGhostUpdateEnd(vx_tmp[dim], INSERT_VALUES, SCATTER_FORWARD);

  // restore pointers
  foreach_dimension(dim) VecRestoreArray(vx_tmp[dim], &vx_p[dim]);
  VecRestoreArray(pressure_m, &pressure_m_p);
  VecRestoreArray(pressure_p, &pressure_p_p);
  VecRestoreArray(potential_m, &potential_m_p);
  VecRestoreArray(potential_p, &potential_p_p);

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

  // free memeory
  VecDestroy(phi_np1);
  VecDestroy(pressure_m);
  VecDestroy(pressure_p);
  VecDestroy(potential_m);
  VecDestroy(potential_p);
  foreach_dimension(dim) VecDestroy(vx[dim]);

  VecCreateGhostNodes(p4est, nodes, &pressure_m);
  VecCreateGhostNodes(p4est, nodes, &pressure_p);
  VecCreateGhostNodes(p4est, nodes, &potential_m);
  VecCreateGhostNodes(p4est, nodes, &potential_p);

  return dt;
}

void coupled_solver_t::solve_fields(double t, Vec phi,
                                    Vec pressure_m,  Vec pressure_p,
                                    Vec potential_m, Vec potential_p)
{
  // compute neighborhood information
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  node_neighbors.init_neighbors();

  // reinitialize the levelset
  my_p4est_level_set_t ls(&node_neighbors);
  ls.reinitialize_2nd_order(phi);
  ls.perturb_level_set_function(phi, EPS);

  // compute the curvature. we store it in the boundary condition vector to save space
  Vec kappa, kappa_tmp, normal[P4EST_DIM];
  VecDuplicate(phi, &kappa);
  VecDuplicate(phi, &kappa_tmp);
  foreach_dimension(dim) VecCreateGhostNodes(p4est, nodes, &normal[dim]);
  compute_normals(node_neighbors, phi, normal);
  compute_mean_curvature(node_neighbors, normal, kappa_tmp);

  // extend curvature from interface to the entire domain
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
  VecDestroy(kappa_tmp);

  // compute the boundary condition for the pressure.
  Vec jump[2], jump_grad[2];
  VecDuplicate(phi, &jump[0]);
  VecDuplicate(phi, &jump[1]);
  VecCreateGhostNodes(p4est, nodes, &jump_grad[0]);
  VecCreateGhostNodes(p4est, nodes, &jump_grad[1]);

  // compute the singular part
  vector<double> pressure_star(nodes->indep_nodes.elem_count);
  vector<double> potential_star(nodes->indep_nodes.elem_count);

  double A = params.alpha,
      B = params.beta,
      R = params.R,
      M = params.M,
      S = params.S;
  double prefactor = M/(R*M - A*B*SQR(S));

  double x[P4EST_DIM], diag_min = p4est_diag_min(p4est);
  foreach_node(n, nodes) {
    node_xyz_fr_n(n, p4est, nodes, x);
#ifdef P4_TO_P8
    double r = MAX(sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2])), daig_min);
    pressure_star[n]  = prefactor*(R*M*(*Q)(t)-A*S*(*I)(t))/(4*PI*r);
    potential_star[n] = prefactor*(    (*I)(t)-B*S*(*Q)(t))/(4*PI*r);
#else
    double r = MAX(sqrt(SQR(x[0]) + SQR(x[1])), diag_min);
    pressure_star[n]  = prefactor*(R*M*(*Q)(t)-A*S*(*I)(t))/(2*PI)*log(r);
    potential_star[n] = prefactor*(    (*I)(t)-B*S*(*Q)(t))/(2*PI)*log(r);
#endif
  }

  // compute solution jumps
  double *kappa_p, *jump_p[2];
  VecGetArray(kappa, &kappa_p);
  VecGetArray(jump[0], &jump_p[0]);
  VecGetArray(jump[1], &jump_p[1]);
  foreach_node(n, nodes) {
    jump_p[0][n]  = -1/params.Ca*kappa_p[n] - pressure_star[n];
    jump_p[1][n]  = -potential_star[n];
  }
  VecRestoreArray(jump[0], &jump_p[0]);
  VecRestoreArray(jump[1], &jump_p[1]);
  VecRestoreArray(kappa, &kappa_p);
  VecDestroy(kappa);

  // jump in the flux is a bit more involved
  // FIXME: change the definiton of normal in the jump solver to remain consistent
  quad_neighbor_nodes_of_node_t qnnn;
  double *normal_p[P4EST_DIM];
  foreach_dimension(dim) VecGetArray(normal[dim], &normal_p[dim]);

  double *pressure_star_p  = pressure_star.data();
  double *potential_star_p = potential_star.data();
  double *jump_grad_p[2];
  VecGetArray(jump_grad[0], &jump_grad_p[0]);
  VecGetArray(jump_grad[1], &jump_grad_p[1]);

  for (size_t i = 0; i<node_neighbors.get_layer_size(); i++) {
    p4est_locidx_t n = node_neighbors.get_layer_node(i);
    node_neighbors.get_neighbors(n, qnnn);

    // TODO: This can be simplifies since it involves K*inv(K) multipications
    jump_grad_p[0][n]  = -1/M*normal_p[0][n]*(qnnn.dx_central(pressure_star_p) +
                                              qnnn.dx_central(potential_star_p)*A*S) +
                         -1/M*normal_p[1][n]*(qnnn.dy_central(pressure_star_p) +
                                              qnnn.dy_central(potential_star_p)*A*S);
#ifdef P4_TO_P8
    jump_grad_p[0][n] += -1/M*normal_p[2][n]*(qnnn.dz_central(pressure_star_p) +
                                              qnnn.dz_central(potential_star_p)*A*S);
#endif

    jump_grad_p[1][n]  = -1/M*normal_p[0][n]*(qnnn.dx_central(pressure_star_p)*B*S +
                                              qnnn.dx_central(potential_star_p)*R*M) +
                         -1/M*normal_p[1][n]*(qnnn.dy_central(pressure_star_p)*B*S +
                                              qnnn.dy_central(potential_star_p)*R*M);
#ifdef P4_TO_P8
    jump_grad_p[1][n] += -1/M*normal_p[2][n]*(qnnn.dz_central(pressure_star_p)*B*S +
                                              qnnn.dz_central(potential_star_p)*R*M);
#endif
  }
  VecGhostUpdateBegin(jump_grad[0], INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateBegin(jump_grad[1], INSERT_VALUES, SCATTER_FORWARD);

  for (size_t i = 0; i<node_neighbors.get_local_size(); i++) {
    p4est_locidx_t n = node_neighbors.get_local_node(i);
    node_neighbors.get_neighbors(n, qnnn);

    // TODO: This can be simplifies since it involves K*inv(K) multipications
    jump_grad_p[0][n]  = -1/M*normal_p[0][n]*(qnnn.dx_central(pressure_star_p) +
                                              qnnn.dx_central(potential_star_p)*A*S) +
                         -1/M*normal_p[1][n]*(qnnn.dy_central(pressure_star_p) +
                                              qnnn.dy_central(potential_star_p)*A*S);
#ifdef P4_TO_P8
    jump_grad_p[0][n] += -1/M*normal_p[2][n]*(qnnn.dz_central(pressure_star_p) +
                                              qnnn.dz_central(potential_star_p)*A*S);
#endif

    jump_grad_p[1][n]  = -1/M*normal_p[0][n]*(qnnn.dx_central(pressure_star_p)*B*S +
                                              qnnn.dx_central(potential_star_p)*R*M) +
                         -1/M*normal_p[1][n]*(qnnn.dy_central(pressure_star_p)*B*S +
                                              qnnn.dy_central(potential_star_p)*R*M);
#ifdef P4_TO_P8
    jump_grad_p[1][n] += -1/M*normal_p[2][n]*(qnnn.dz_central(pressure_star_p)*B*S +
                                              qnnn.dz_central(potential_star_p)*R*M);
#endif
  }
  VecGhostUpdateEnd(jump_grad[0], INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(jump_grad[1], INSERT_VALUES, SCATTER_FORWARD);

  VecRestoreArray(jump_grad[0], &jump_grad_p[0]);
  VecRestoreArray(jump_grad[1], &jump_grad_p[1]);

  // destroy normals
  foreach_dimension(dim) {
    VecRestoreArray(normal[dim], &normal_p[dim]);
    VecDestroy(normal[dim]);
  }

  // solve the pressure jump problem
#ifdef P4_TO_P8
  struct constant_cf_t:cf_t{
    constant_cf_t(double c) : c(c) {}
    double operator()(double, double, double) const { return c; }
  private:
    double c;
  };
#else
  struct constant_cf_t:cf_t{
    constant_cf_t(double c) : c(c) {}
    double operator()(double, double) const { return c; }
  private:
    double c;
  };
#endif
  constant_cf_t km[][2] =
  {
    { constant_cf_t(1/M),   constant_cf_t(A*S/M) },
    { constant_cf_t(B*S/M), constant_cf_t(R)     },
  };
  vector<vector<cf_t*>> mue_m(2,vector<cf_t*>(2));
  mue_m[0][0] = &km[0][0];
  mue_m[0][1] = &km[0][1];
  mue_m[1][0] = &km[1][0];
  mue_m[1][1] = &km[1][1];

  constant_cf_t kp[][2] =
  {
    { constant_cf_t(1), constant_cf_t(A) },
    { constant_cf_t(B), constant_cf_t(1) },
  };
  vector<vector<cf_t*>> mue_p(2,vector<cf_t*>(2));
  mue_p[0][0] = &kp[0][0];
  mue_p[0][1] = &kp[0][1];
  mue_p[1][0] = &kp[1][0];
  mue_p[1][1] = &kp[1][1];

  constant_cf_t zero(0);
  vector<vector<cf_t*>> add(2,vector<cf_t*>(2));
  add[0][0] = &zero;
  add[0][1] = &zero;
  add[1][0] = &zero;
  add[1][1] = &zero;

  my_p4est_interpolation_nodes_t interp1(&node_neighbors);
  my_p4est_interpolation_nodes_t interp2(&node_neighbors);
  my_p4est_interpolation_nodes_t interp3(&node_neighbors);
  my_p4est_interpolation_nodes_t interp4(&node_neighbors);
  interp1.set_input(jump[0], linear);
  interp2.set_input(jump[1], linear);
  interp3.set_input(jump_grad[0], linear);
  interp4.set_input(jump_grad[1], linear);

  vector<cf_t*> jump_u(2), jump_du(2);
  jump_u[0]  = &interp1;
  jump_u[1]  = &interp2;
  jump_du[0] = &interp3;
  jump_du[1] = &interp4;

  Vec rhs_p[2], rhs_m[2];
  VecDuplicate(phi, &rhs_p[0]); sample_cf_on_nodes(p4est, nodes, zero, rhs_p[0]);
  VecDuplicate(phi, &rhs_p[1]); sample_cf_on_nodes(p4est, nodes, zero, rhs_p[1]);
  VecDuplicate(phi, &rhs_m[0]); sample_cf_on_nodes(p4est, nodes, zero, rhs_m[0]);
  VecDuplicate(phi, &rhs_m[1]); sample_cf_on_nodes(p4est, nodes, zero, rhs_m[1]);

  vector<bc_t> bc(2);
  pressure_bc_val->t  = t;
  potential_bc_val->t = t;
  bc[0].setWallTypes(*pressure_bc_type);
  bc[0].setWallValues(*pressure_bc_val);

  bc[1].setWallTypes(*potential_bc_type);
  bc[1].setWallValues(*potential_bc_val);

  // set up the solver
  my_p4est_cell_neighbors_t cell_neighbors(&hierarchy);

  my_p4est_poisson_jump_voronoi_block_t solver(2, &node_neighbors, &cell_neighbors);
  solver.set_phi(phi);
  solver.set_bc(bc);
  solver.set_diagonal(add);
  solver.set_mu(mue_m, mue_p);
  solver.set_u_jump(jump_u);
  solver.set_mu_grad_u_jump(jump_du);
  solver.set_rhs(rhs_m, rhs_p);

  Vec solutions [] = {pressure_p, potential_p};
  solver.solve(solutions);

  VecDestroy(rhs_m[0]);
  VecDestroy(rhs_p[0]);
  VecDestroy(rhs_m[1]);
  VecDestroy(rhs_p[1]);
  VecDestroy(jump[0]);
  VecDestroy(jump[1]);
  VecDestroy(jump_grad[0]);
  VecDestroy(jump_grad[1]);

  // extract solutions
  double *pressure_m_p,  *pressure_p_p;
  double *potential_m_p, *potential_p_p;
  VecGetArray(pressure_m,  &pressure_m_p);
  VecGetArray(pressure_p,  &pressure_p_p);
  VecGetArray(potential_m, &potential_m_p);
  VecGetArray(potential_p, &potential_p_p);

  foreach_node(n, nodes) {
    pressure_m_p[n]  = pressure_p_p[n] - pressure_star[n];
    potential_m_p[n] = potential_p_p[n] - pressure_star[n];
  }

  // extend solutions
  VecRestoreArray(pressure_m,  &pressure_m_p);
  VecRestoreArray(pressure_p,  &pressure_p_p);
  VecRestoreArray(potential_m, &potential_m_p);
  VecRestoreArray(potential_p, &potential_p_p);

  // (-) --> (+)
  ls.extend_Over_Interface_TVD(phi, pressure_m);
  ls.extend_Over_Interface_TVD(phi, potential_m);

  // (+) --> (-)
  Vec phi_l;
  VecGhostGetLocalForm(phi, &phi_l);
  VecScale(phi_l, -1);

  ls.extend_Over_Interface_TVD(phi, pressure_p);
  ls.extend_Over_Interface_TVD(phi, potential_p);

  VecScale(phi_l, -1);
  VecGhostRestoreLocalForm(phi, &phi_l);
}

double coupled_solver_t:: solve_one_step(double t, Vec &phi,
                                         Vec &pressure_m, Vec& pressure_p,
                                         Vec &potential_m, Vec& potential_p,
                                         double cfl, double dtmax)
{
  // advect the interface
  double dt;
  dt = advect_interface(phi, pressure_m, pressure_p, potential_m, potential_p, cfl, dtmax);

  // solve for the pressure
  solve_fields(t+dt, phi, pressure_m, pressure_p, potential_m, potential_p);

  return dt;
}


