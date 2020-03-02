#ifdef P4_TO_P8
#include "my_p8est_semi_lagrangian.h"
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include "my_p4est_semi_lagrangian.h"
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <sc_notify.h>
#include <mpi.h>
#include <src/ipm_logging.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_vtk.h>

// system
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2;
extern PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order;
extern PetscLogEvent log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order;
extern PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_CF2;
extern PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_1st_order;
extern PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_2nd_order;
extern PetscLogEvent log_my_p4est_semi_lagrangian_update_p4est_multiple_phi;
extern PetscLogEvent log_my_p4est_semi_lagrangian_grid_gen_iter[P4EST_MAXLEVEL];
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

my_p4est_semi_lagrangian_t::my_p4est_semi_lagrangian_t(p4est_t **p4est_np1, p4est_nodes_t **nodes_np1, p4est_ghost_t **ghost_np1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_node_neighbors_t *ngbd_nm1)
  : p_p4est(p4est_np1), p4est(*p4est_np1),
    p_nodes(nodes_np1), nodes(*nodes_np1),
    p_ghost(ghost_np1), ghost(*ghost_np1),
    ngbd_n(ngbd_n),
    ngbd_nm1(ngbd_nm1)
{
  // compute domain sizes
  double *v2c = this->p4est->connectivity->vertices;
  p4est_topidx_t *t2v = this->p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = this->p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (unsigned char i=0; i < P4EST_DIM; i++)
  {
    xyz_min[i]  = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
    xyz_max[i]  = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
    periodic[i] = is_periodic(p4est, i);
  }
  velo_interpolation = quadratic;
  phi_interpolation  = quadratic_non_oscillatory;

  ngbd_phi = ngbd_n;
}

double my_p4est_semi_lagrangian_t::compute_dt(DIM(const CF_DIM &vx, const CF_DIM &vy, const CF_DIM &vz))
{
  double dt = DBL_MAX;

  // get the min dx
  splitting_criteria_t* data = (splitting_criteria_t*)p4est->user_pointer;
  double dx = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN;

  double tree_xyz_max[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    tree_xyz_max[dir] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + P4EST_CHILDREN - 1] + dir];

  dx *= MIN(DIM(tree_xyz_max[0]-xyz_min[0], tree_xyz_max[1]-xyz_min[1], tree_xyz_max[2]-xyz_min[2]));

  double xyz[P4EST_DIM];
  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; n++)
  {
    node_xyz_fr_n(n, p4est, nodes, xyz);
    double vn = sqrt(SUMD(SQR(vx(xyz)), SQR(vy(xyz)), SQR(vz(xyz))));
    dt = MIN(dt, dx/vn);
  }

  // reduce among processors
  double dt_min;
  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  return dt_min;
}

double my_p4est_semi_lagrangian_t::compute_dt(DIM(Vec vx, Vec vy, Vec vz))
{
  PetscErrorCode ierr;
  double dt = DBL_MAX;
  const double *vx_p, *vy_p;

  ierr = VecGetArrayRead(vx, &vx_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vy, &vy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  const double *vz_p;
  ierr = VecGetArrayRead(vz, &vz_p); CHKERRXX(ierr);
#endif

  // get the min dx
  splitting_criteria_t* data = (splitting_criteria_t*)p4est->user_pointer;
  double dx = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN;


  double tree_xyz_max[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    tree_xyz_max[dir] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + P4EST_CHILDREN - 1] + dir];

  dx *= MIN(DIM(tree_xyz_max[0]-xyz_min[0], tree_xyz_max[1]-xyz_min[1], tree_xyz_max[2]-xyz_min[2]));

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++){
    double vn = sqrt(SUMD(SQR(vx_p[i]), SQR(vy_p[i]), SQR(vz_p[i])));
    dt = MIN(dt, dx/vn);
  }

  ierr = VecRestoreArrayRead(vx, &vx_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vy, &vy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArrayRead(vz, &vz_p); CHKERRXX(ierr);
#endif

  // reduce among processors
  double dt_min;
  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  return dt_min;
}

void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt, const CF_DIM **v,
                                                      Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);

  for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

    /* find the departure node via backtracing */
    double xyz_star[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] = xyz[dir] - 0.5*dt*(*v[dir])(xyz);
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    double xyz_d[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_d[dir] = xyz[dir] - dt*(*v[dir])(xyz_star);
    clip_in_domain(xyz_d, xyz_min, xyz_max, periodic);

    /* Buffer the point for interpolation */
    interp.add_point(n, xyz_d);
  }

  interp.set_input(phi_n, DIM(phi_xx_n[0], phi_xx_n[1], phi_xx_n[2]), quadratic_non_oscillatory);
  interp.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt, Vec *v, Vec **vxx, Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);
  my_p4est_interpolation_nodes_t interp_phi(ngbd_phi);

  double *interp_output[P4EST_DIM];

  Vec xx_v_derivatives[P4EST_DIM] = {DIM(vxx[0][0], vxx[1][0], vxx[2][0])};
  Vec yy_v_derivatives[P4EST_DIM] = {DIM(vxx[0][1], vxx[1][1], vxx[2][1])};
#ifdef P4_TO_P8
  Vec zz_v_derivatives[P4EST_DIM] = {vxx[0][2], vxx[1][2], vxx[2][2]};
#endif

  /* find vnp1 */
  std::vector<double> v_tmp[P4EST_DIM];
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);
    interp.add_point(n, xyz);
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    v_tmp[dir].resize(nodes->indep_nodes.elem_count);
    interp_output[dir] = v_tmp[dir].data();
  }
  interp.set_input(v, DIM(xx_v_derivatives, yy_v_derivatives, zz_v_derivatives), quadratic, P4EST_DIM);
  interp.interpolate(interp_output);
  interp.clear();

  /* now find v_star */
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    /* Find initial xy points */
    double xyz_star[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz_star);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] -= 0.5*dt*v_tmp[dir][n];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp.add_point(n, xyz_star);
  }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    interp_output[dir] = v_tmp[dir].data();
  interp.set_input(v, DIM(xx_v_derivatives, yy_v_derivatives, zz_v_derivatives), quadratic, P4EST_DIM);
  interp.interpolate(interp_output);
  interp.clear();

  /* finally, find the backtracing value */
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double xyz_d[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz_d);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_d[dir] -= dt*v_tmp[dir][n];
    clip_in_domain(xyz_d, xyz_min, xyz_max, periodic);

    interp_phi.add_point(n, xyz_d);
  }

  interp_phi.set_input(phi_n, DIM(phi_xx_n[0], phi_xx_n[1], phi_xx_n[2]), phi_interpolation);
  interp_phi.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt_nm1, double dt_n,
                                                      Vec *vnm1, Vec **vxx_nm1,
                                                      Vec *vn  , Vec **vxx_n,
                                                      Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n  (ngbd_n);
  my_p4est_interpolation_nodes_t interp_phi(ngbd_phi);

  double *interp_output[P4EST_DIM];

  Vec xx_vn_derivatives[P4EST_DIM]    = {DIM(vxx_n[0][0],   vxx_n[1][0],    vxx_n[2][0])};
  Vec yy_vn_derivatives[P4EST_DIM]    = {DIM(vxx_n[0][1],   vxx_n[1][1],    vxx_n[2][1])};
  Vec xx_vnm1_derivatives[P4EST_DIM]  = {DIM(vxx_nm1[0][0], vxx_nm1[1][0],  vxx_nm1[2][0])};
  Vec yy_vnm1_derivatives[P4EST_DIM]  = {DIM(vxx_nm1[0][1], vxx_nm1[1][1],  vxx_nm1[2][1])};
#ifdef P4_TO_P8
  Vec zz_vn_derivatives[P4EST_DIM]    = {vxx_n[0][2],   vxx_n[1][2],    vxx_n[2][2]};
  Vec zz_vnm1_derivatives[P4EST_DIM]  = {vxx_nm1[0][2], vxx_nm1[1][2],  vxx_nm1[2][2]};
#endif

  std::vector<double> v_tmp_nm1[P4EST_DIM];
  std::vector<double> v_tmp_n  [P4EST_DIM];

  /* find the velocity field at time np1 */
  for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);
    interp_n.add_point(n, xyz);
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    v_tmp_n[dir].resize(nodes->indep_nodes.elem_count);
    interp_output[dir] = v_tmp_n[dir].data();
  }

  interp_n.set_input(vn, DIM(xx_vn_derivatives, yy_vn_derivatives, zz_vn_derivatives), quadratic, P4EST_DIM);
  interp_n.interpolate(interp_output);
  interp_n.clear();

  /* now find x_star */
  for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double xyz_star[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz_star);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] -= .5*dt_n*v_tmp_n[dir][n];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp_n  .add_point(n, xyz_star);
    interp_nm1.add_point(n, xyz_star);
  }

  /* interpolate vnm1 */
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    v_tmp_nm1[dir].resize(nodes->indep_nodes.elem_count);
    interp_output[dir] = v_tmp_nm1[dir].data();
  }
  interp_nm1.set_input(vnm1, DIM(xx_vnm1_derivatives, yy_vnm1_derivatives, zz_vnm1_derivatives), quadratic, P4EST_DIM);
  interp_nm1.interpolate(interp_output);
  interp_nm1.clear();
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    interp_output[dir] = v_tmp_n[dir].data();
  interp_n.set_input(vn, DIM(xx_vn_derivatives, yy_vn_derivatives, zz_vn_derivatives), quadratic, P4EST_DIM);
  interp_n.interpolate(interp_output);
  interp_n.clear();

  /* finally, find the backtracing value */
  /* find the departure node via backtracing */
  double v_star[P4EST_DIM];
  for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      v_star[dir] = (1.0+.5*dt_n/dt_nm1)*v_tmp_n[dir][n] - 0.5*dt_n/dt_nm1*v_tmp_nm1[dir][n];

    double xyz_d[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz_d);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_d[dir] -= dt_n*v_star[dir];
    clip_in_domain(xyz_d, xyz_min, xyz_max, periodic);

    interp_phi.add_point(n, xyz_d);
  }

  interp_phi.set_input(phi_n, DIM(phi_xx_n[0], phi_xx_n[1], phi_xx_n[2]), phi_interpolation);
  interp_phi.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_semi_lagrangian_t::update_p4est(const CF_DIM **v, double dt, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) { ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx[dir]); CHKERRXX(ierr); }

    ngbd_n->second_derivatives_central(phi, phi_xx);
    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*)ngbd_n->p4est->user_pointer;

  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, phi, phi_xx, phi_np1_p);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est->user_pointer = p4est->user_pointer;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  if (local_derivatives)
  {
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) { ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr); }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_semi_lagrangian_t::update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx, Vec phi_add_refine)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vx_xx, vx_yy */
  Vec *vxx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir == 0)
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecCreateGhostNodes(ngbd_n->p4est, ngbd_n->nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    else
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
    ngbd_n->second_derivatives_central(v[dir], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
  }

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char  dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi, DIM(phi_xx[0], phi_xx[1], phi_xx[2]));
    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*) p4est->user_pointer;

  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, vxx, phi, phi_xx, phi_np1_p);

    Vec phi_np1_eff;
    double *phi_np1_eff_p = phi_np1_p;

    if (phi_add_refine != NULL)
    {
      ierr = VecDuplicate(phi_np1, &phi_np1_eff);
      my_p4est_interpolation_nodes_t interp(ngbd_phi);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }
      interp.set_input(phi_add_refine, phi_interpolation);
      interp.interpolate(phi_np1_eff);

      ierr = VecGetArray(phi_np1_eff, &phi_np1_eff_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        phi_np1_eff_p[n] = MIN(fabs(phi_np1_eff_p[n]), fabs(phi_np1_p[n]));
    }


    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_np1_eff_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    if (phi_add_refine != NULL)
    {
      ierr = VecRestoreArray(phi_np1_eff, &phi_np1_eff_p); CHKERRXX(ierr);
      ierr = VecDestroy(phi_np1_eff); CHKERRXX(ierr);
    }

    if (is_grid_changing) {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx[dir];
  }

  if (local_derivatives) {
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}

// ELYCE TRYING SOMETHING:---------------------------

void my_p4est_semi_lagrangian_t::update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx, Vec phi_add_refine, const unsigned int num_fields, bool use_block, bool enforce_uniform_band,double refine_band, double coarsen_band, Vec *fields, Vec fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn,bool expand_ghost_layer)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  // compute vx_xx, vx_yy
  Vec *vxx[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir == 0)
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecCreateGhostNodes(ngbd_n->p4est, ngbd_n->nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    else
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    ngbd_n->second_derivatives_central(v[dir], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
  }

  // compute phi_xx and phi_yy
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi, DIM(phi_xx[0], phi_xx[1], phi_xx[2]));
    local_derivatives = true;
  }

  // save the old splitting criteria information
  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;

  // Create np1 vectors to update as the grid changes:
  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  Vec fields_np1[num_fields];
  Vec fields_block_np1;
  if(num_fields!=0){
    if(use_block){
        ierr = VecCreateGhostNodesBlock(p4est,nodes,num_fields,&fields_block_np1);CHKERRXX(ierr);
      }
    else{
        for(unsigned int k=0; k<num_fields; k++){
            ierr = VecCreateGhostNodes(p4est,nodes,&fields_np1[k]); CHKERRXX(ierr);
          }
      }
  }


  bool is_grid_changing = true;
  bool additional_phi_is_used = false;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, vxx, phi, phi_xx, phi_np1_p);

    Vec phi_np1_eff;
    double *phi_np1_eff_p = phi_np1_p;


    if (phi_add_refine != NULL)
    {
      additional_phi_is_used = true;
      ierr = VecDuplicate(phi_np1, &phi_np1_eff);

      // Interpolate phi_add_refine onto the current grid, so we can get the effective LSF to refine/coarsen by:
      my_p4est_interpolation_nodes_t interp(ngbd_phi);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }
      interp.set_input(phi_add_refine, quadratic_non_oscillatory_continuous_v2);
      interp.interpolate(phi_np1_eff);

      ierr = VecGetArray(phi_np1_eff, &phi_np1_eff_p); CHKERRXX(ierr);
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
          phi_np1_eff_p[n] = MIN(fabs(phi_np1_eff_p[n]),fabs(phi_np1_p[n]));
      }

      ierr = VecRestoreArray(phi_np1_eff, &phi_np1_eff_p); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    // Interpolate the fields onto the current grid: (either block vector, or array of vectors)
    if(num_fields!=0){
//      PetscPrintf(p4est->mpicomm,"Doing interp of fields \n");
      if(use_block){
          my_p4est_interpolation_nodes_t interp_block(ngbd_phi);
          interp_block.set_input(fields_block, phi_interpolation,num_fields);

          for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
              double xyz[P4EST_DIM];
              node_xyz_fr_n(n, p4est, nodes, xyz);
              interp_block.add_point(n, xyz);
            }
          interp_block.interpolate(fields_block_np1);
          interp_block.clear();

        }
      else{
          my_p4est_interpolation_nodes_t interp_fields(ngbd_phi);
          interp_fields.set_input(fields, phi_interpolation,num_fields);

          double xyz[P4EST_DIM];
          for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
            {
              node_xyz_fr_n(n, p4est, nodes, xyz);
              interp_fields.add_point(n, xyz);
            }
          interp_fields.interpolate(fields_np1);
          interp_fields.clear();
        }
    }

    if(counter > 10){PetscPrintf(p4est->mpicomm,"Grid did not converge ... \n"); break;}

    // Call the refine and coarsen according to phi_effective and the provided fields:
    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est,nodes,additional_phi_is_used?phi_np1_eff:phi_np1,num_fields,use_block,enforce_uniform_band,refine_band,coarsen_band,fields_np1,fields_block_np1,criteria,compare_opn,diag_opn);

    // Destroy the phi_effective now that no longer in use:
    if(additional_phi_is_used){
        ierr = VecDestroy(phi_np1_eff); CHKERRXX(ierr);
      }

    if (is_grid_changing) {
      PetscPrintf(p4est->mpicomm, "Grid changed\n");
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      if(expand_ghost_layer) my_p4est_ghost_expand(p4est,ghost);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

      // Reset the fields to refine by
      if(num_fields!=0){
        if(use_block){
            ierr = VecDestroy(fields_block_np1); CHKERRXX(ierr);
            ierr = VecCreateGhostNodesBlock(p4est,nodes,num_fields,&fields_block_np1);
          }
        else{
            for(unsigned int k=0; k<num_fields; k++){
                ierr = VecDestroy(fields_np1[k]);CHKERRXX(ierr);
                ierr = VecCreateGhostNodes(p4est,nodes,&fields_np1[k]);
              } // end of (for k = 0, ..., num fields)
          } //end of "if use block, else"
      }
    } // end of "if grid changing"
    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  } // end of "while grid is changing"

// Do one more balancing of everything:
//  PetscPrintf(p4est->mpicomm,"Doing final balance \n");
  p4est_balance(p4est,P4EST_CONNECT_FULL,NULL);
  my_p4est_partition(p4est, P4EST_TRUE, NULL);
  p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  if(expand_ghost_layer) my_p4est_ghost_expand(p4est,ghost);
  p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

  // We will need to advect phi one last time using the balanced grid
  ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est,nodes,&phi_np1);
  double* phi_np1_p;
  ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

  advect_from_n_to_np1(dt, v, vxx, phi, phi_xx, phi_np1_p);
  ierr = VecRestoreArray(phi_np1,&phi_np1_p);


  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    for(int dd=0; dd<P4EST_DIM; ++dd)
    {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx[dir];
  }

  if (local_derivatives)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  if(num_fields!=0){
    if(use_block){
        ierr = VecDestroy(fields_block_np1);CHKERRXX(ierr);
      }
    else{
        for(unsigned int k=0; k<num_fields; k++){
            ierr = VecDestroy(fields_np1[k]); CHKERRXX(ierr);
          }
      }
  }
  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}

// END: ELYCE TRYING SOMETHING ----------------

void my_p4est_semi_lagrangian_t::update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);

  if(ngbd_nm1==NULL) throw std::invalid_argument("[ERROR]: you need to set ngbd_nm1 in order to use second order semi-Lagrangian advection.");

  /* compute vx_xx_nm1, vx_yy_nm1, ... */
  Vec *vxx_nm1[P4EST_DIM];
  Vec *vxx_n  [P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vxx_nm1[dir] = new Vec[P4EST_DIM];
    vxx_n  [dir] = new Vec[P4EST_DIM];
    if(dir == 0) {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecCreateGhostNodes(ngbd_nm1->p4est, ngbd_nm1->nodes, &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(ngbd_n  ->p4est, ngbd_n  ->nodes, &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }
    }
    else {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecDuplicate(vxx_nm1[0][dd], &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecDuplicate(vxx_n  [0][dd], &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }
    }
    ngbd_nm1->second_derivatives_central(vnm1[dir], DIM(vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2]));
    ngbd_n  ->second_derivatives_central(vn  [dir], DIM(vxx_n  [dir][0], vxx_n  [dir][1], vxx_n  [dir][2]));
  }

  /* now for phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi, DIM(phi_xx[0], phi_xx[1], phi_xx[2]));

    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*)ngbd_n->p4est->user_pointer;

  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt_nm1, dt_n,
                         vnm1, vxx_nm1,
                         vn, vxx_n,
                         phi, phi_xx,
                         phi_np1_p);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd)
    {
      ierr = VecDestroy(vxx_nm1[dir][dd]); CHKERRXX(ierr);
      ierr = VecDestroy(vxx_n  [dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx_nm1[dir];
    delete[] vxx_n  [dir];
  }


  if(local_derivatives)
  {
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::update_p4est(std::vector<Vec> *v, double dt, std::vector<Vec> &phi)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);
  for(unsigned char i=0; i < P4EST_DIM; ++i)
    P4EST_ASSERT(v[i].size()==phi.size());

  Vec *vxx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir == 0)
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd){
        ierr = VecCreateGhostNodes(ngbd_n->p4est, ngbd_n->nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    else
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd){
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
  }

  Vec phi_xx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir){
    ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
  }

  Vec velo[P4EST_DIM];

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*)ngbd_n->p4est->user_pointer;

  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  /* update the new forest by advecting each level-set one after the other */
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    /* compute vx_xx, vx_yy */
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
    {
      velo[dir] = v[dir][i];
      ngbd_n->second_derivatives_central(v[dir][i], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
    }

    /* compute phi_xx and phi_yy */
    ngbd_phi->second_derivatives_central(phi[i], DIM(phi_xx[0], phi_xx[1], phi_xx[2]));

    bool is_grid_changing = true;

    int counter = 0;
    while (is_grid_changing) {
      ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

      // advect from np1 to n to enable refinement
      double* phi_np1_p;
      ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      advect_from_n_to_np1(dt, velo, vxx, phi[i], phi_xx, phi_np1_p);

      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
      is_grid_changing = sp.refine(p4est, nodes, phi_np1_p);

      ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

      if (is_grid_changing) {
        my_p4est_partition(p4est, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

        ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
      }

      ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
      counter++;
    }
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  /* now update all the new level-sets with their new values */
  for(unsigned int i=0; i<phi.size(); ++i)
  {
    /* compute vx_xx, vx_yy */
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
    {
      velo[dir] = v[dir][i];
      ngbd_n->second_derivatives_central(v[dir][i], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
    }

    /* compute phi_xx and phi_yy (and phi_zz) */
    ngbd_phi->second_derivatives_central(phi[i], DIM(phi_xx[0], phi_xx[1], phi_xx[2]));

    double* phi_np1_p;
    Vec tmp;
    if(i == 0)
      tmp = phi_np1;
    else {
      ierr = VecDuplicate(phi_np1, &tmp); CHKERRXX(ierr);
    }
    ierr = VecGetArray(tmp, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, velo, vxx, phi[i], phi_xx, phi_np1_p);

    ierr = VecRestoreArray(tmp, &phi_np1_p); CHKERRXX(ierr);

    ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
    phi[i] = tmp;
  }

  p4est->user_pointer = (void*)sp_old;

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx[dir];
  }

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_semi_lagrangian_t::update_p4est(Vec *v, double dt, std::vector<Vec> &phi_parts, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vx_xx, vx_yy */
  Vec *vxx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir == 0) {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecCreateGhostNodes(ngbd_n->p4est, ngbd_n->nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    else {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    ngbd_n->second_derivatives_central(v[dir], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
  }

  /* compute phi_xx and phi_yy (and phi_zz) */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir){
      ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi, DIM(phi_xx[0], phi_xx[1], phi_xx[2]));
    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;

  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, vxx, phi, phi_xx, phi_np1_p);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  /* now update all the new level-sets with their new values */
  Vec phi_part_xx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(vxx[0][dir], &phi_part_xx[dir]); CHKERRXX(ierr);
  }

  for(unsigned int i=0; i<phi_parts.size(); ++i)
  {
    /* compute phi_xx and phi_yy */
    ngbd_phi->second_derivatives_central(phi_parts[i], DIM(phi_part_xx[0], phi_part_xx[1], phi_part_xx[2]));
    double* tmp_ptr;
    Vec tmp;
    ierr = VecDuplicate(phi_np1, &tmp); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_ptr); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, vxx, phi_parts[i], phi_xx, tmp_ptr);

    ierr = VecRestoreArray(tmp, &tmp_ptr); CHKERRXX(ierr);

    ierr = VecDestroy(phi_parts[i]); CHKERRXX(ierr);
    phi_parts[i] = tmp;
  }

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(phi_part_xx[dir]); CHKERRXX(ierr);
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx[dir];
  }

  if (local_derivatives) {
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_semi_lagrangian_t::update_p4est(Vec *v, double dt, std::vector<Vec> &phi, std::vector<mls_opn_t> &action, int phi_idx, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);

  int num_lsf = phi.size();

  /* compute vx_xx, vx_yy, vx_zz */
  Vec *vxx[P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir == 0) {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecCreateGhostNodes(ngbd_n->p4est, ngbd_n->nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    else {
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    ngbd_n->second_derivatives_central(v[dir], DIM(vxx[dir][0], vxx[dir][1], vxx[dir][2]));
  }

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir){
      ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi[phi_idx], DIM(phi_xx[0], phi_xx[1], phi_xx[2]));
    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*) p4est->user_pointer;

  std::vector<Vec> phi_np1(num_lsf, NULL);
  for (int i = 0; i < num_lsf; i++) {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1[i]); CHKERRXX(ierr);
  }

  Vec phi_eff;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect specified LSF from np1 to n to enable refinement
    std::vector<double *> phi_np1_ptr(num_lsf, NULL);

    for (int i = 0; i < num_lsf; i++) {
      ierr = VecGetArray(phi_np1[i], &phi_np1_ptr[i]); CHKERRXX(ierr);
    }

    my_p4est_interpolation_nodes_t interp(ngbd_phi);
    if (num_lsf > 1) // prepare points for interpolation
    {
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }
    }

    for (int i = 0; i < num_lsf; i++)
    {
      if (i == phi_idx)
      {
        advect_from_n_to_np1(dt, v, vxx, phi[i], phi_xx, phi_np1_ptr[i]);
      } else {
        interp.set_input(phi[i], phi_interpolation);
        interp.interpolate(phi_np1_ptr[i]);
      }
    }

    interp.clear();

    // construct the effective LSF
    double* phi_eff_ptr;
    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

    for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double phi_total = -1.0e6; // this is quite ugly
      for (unsigned int i = 0; i < phi.size(); i++)
      {
        double phi_current = phi_np1_ptr[i][n];

        if      (action[i] == MLS_INTERSECTION) phi_total = MAX(phi_total, phi_current);
        else if (action[i] == MLS_ADDITION)     phi_total = MIN(phi_total, phi_current);
      }
      phi_eff_ptr[n] = phi_total;
    }

    for (int i = 0; i < num_lsf; i++) { ierr = VecRestoreArray(phi_np1[i], &phi_np1_ptr[i]); CHKERRXX(ierr); }

    // refine and coarsen grid using the effective LSF
    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
//    sp.set_refine_only_inside(true);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_eff_ptr);

    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);

      for (int i = 0; i < num_lsf; i++)
      {
        ierr = VecDestroy(phi_np1[i]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1[i]); CHKERRXX(ierr);
      }
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  for (int i = 0; i < num_lsf; i++)
  {
    ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
    phi[i] = phi_np1[i];
  }

  ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx[dir];
  }

  if (local_derivatives) {
    for(int dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_semi_lagrangian_t::update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, std::vector<Vec> &phi, std::vector<mls_opn_t> &action, int phi_idx, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);

  if(ngbd_nm1==NULL) throw std::invalid_argument("[ERROR]: you need to set ngbd_nm1 in order to use second order semi-Lagrangian advection.");

  int num_lsf = phi.size();

  /* compute vx_xx_nm1, vx_yy_nm1, ... */
  Vec *vxx_nm1[P4EST_DIM];
  Vec *vxx_n  [P4EST_DIM];
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    vxx_nm1[dir] = new Vec[P4EST_DIM];
    vxx_n  [dir] = new Vec[P4EST_DIM];
    if(dir == 0)
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd)
      {
        ierr = VecCreateGhostNodes(ngbd_nm1->p4est, ngbd_nm1->nodes, &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(ngbd_n  ->p4est, ngbd_n  ->nodes, &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }
    else
      for(unsigned char dd=0; dd < P4EST_DIM; ++dd)
      {
        ierr = VecDuplicate(vxx_nm1[0][dd], &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecDuplicate(vxx_n  [0][dd], &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }

    ngbd_nm1->second_derivatives_central(vnm1[dir], DIM(vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2]));
    ngbd_n  ->second_derivatives_central(vn  [dir], DIM(vxx_n  [dir][0], vxx_n  [dir][1], vxx_n  [dir][2]));
  }

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(ngbd_phi->p4est, ngbd_phi->nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

    ngbd_phi->second_derivatives_central(phi[phi_idx], DIM(phi_xx[0], phi_xx[1], phi_xx[2]));
    local_derivatives = true;
  }

  /* save the old splitting criteria information */
  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;

  std::vector<Vec> phi_np1(num_lsf, NULL);
  for (int i = 0; i < num_lsf; i++)
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1[i]); CHKERRXX(ierr);

  Vec phi_eff;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);

  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect specified LSF from np1 to n to enable refinement
    std::vector<double *> phi_np1_ptr(num_lsf, NULL);

    for (int i = 0; i < num_lsf; i++) { ierr = VecGetArray(phi_np1[i], &phi_np1_ptr[i]); CHKERRXX(ierr); }

    my_p4est_interpolation_nodes_t interp(ngbd_phi);
    if (num_lsf > 1) // prepare points for interpolation
      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }

    for (int i = 0; i < num_lsf; i++)
      if (i == phi_idx)
      {
        advect_from_n_to_np1(dt_nm1, dt_n,
                             vnm1, vxx_nm1,
                             vn, vxx_n,
                             phi[i], phi_xx,
                             phi_np1_ptr[i]);
      }
      else
      {
        interp.set_input(phi[i], quadratic_non_oscillatory);
        interp.interpolate(phi_np1_ptr[i]);
      }

    interp.clear();

    // construct the effective LSF
    double* phi_eff_ptr;
    ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

    for (size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double phi_total = -1.0e6; // this is quite ugly
      for (unsigned int i = 0; i < phi.size(); i++)
      {
        double phi_current = phi_np1_ptr[i][n];

        if      (action[i] == MLS_INTERSECTION) phi_total = MAX(phi_total, phi_current);
        else if (action[i] == MLS_ADDITION)     phi_total = MIN(phi_total, phi_current);
      }
      phi_eff_ptr[n] = phi_total;
    }

    for (int i = 0; i < num_lsf; i++) { ierr = VecRestoreArray(phi_np1[i], &phi_np1_ptr[i]); CHKERRXX(ierr); }

    // refine and coarsen grid using the effective LSF
    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est, nodes, phi_eff_ptr);

    ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

    if (is_grid_changing)
    {
      my_p4est_partition(p4est, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);

      for (int i = 0; i < num_lsf; i++)
      {
        ierr = VecDestroy(phi_np1[i]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1[i]); CHKERRXX(ierr);
      }
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est->user_pointer = (void*) sp_old;
  *p_p4est = p4est;
  *p_nodes = nodes;
  *p_ghost = ghost;

  for (int i = 0; i < num_lsf; i++)
  {
    ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
    phi[i] = phi_np1[i];
  }

  ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

  for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
    for(unsigned char dd=0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(vxx_nm1[dir][dd]); CHKERRXX(ierr);
      ierr = VecDestroy(vxx_n  [dir][dd]); CHKERRXX(ierr);
    }
    delete[] vxx_nm1[dir];
    delete[] vxx_n  [dir];
  }
  if (local_derivatives) {
    for(unsigned char dir=0; dir < P4EST_DIM; ++dir) {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete[] phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_multiple_phi, 0, 0, 0, 0); CHKERRXX(ierr);
}
