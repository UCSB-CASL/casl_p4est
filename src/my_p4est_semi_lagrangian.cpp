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
#include <src/CASL_math.h>
#include <sc_notify.h>
#include <mpi.h>
#include <src/ipm_logging.h>

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
extern PetscLogEvent log_my_p4est_semi_lagrangian_grid_gen_iter[P4EST_MAXLEVEL];
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

my_p4est_semi_lagrangian_t::my_p4est_semi_lagrangian_t(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb, my_p4est_node_neighbors_t *ngbd)
  : p_p4est(p4est), p4est(*p4est),
    p_nodes(nodes), nodes(*nodes),
    p_ghost(ghost), ghost(*ghost),
    myb(myb),
    ngbd_n(ngbd),
    hierarchy(ngbd->hierarchy)
{
  // compute domain sizes
  double *v2c = this->p4est->connectivity->vertices;
  p4est_topidx_t *t2v = this->p4est->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = this->p4est->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<P4EST_DIM; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<P4EST_DIM; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}

#ifdef P4_TO_P8
double my_p4est_semi_lagrangian_t::compute_dt(const CF_3 &vx, const CF_3 &vy, const CF_3 &vz)
#else
double my_p4est_semi_lagrangian_t::compute_dt(const CF_2 &vx, const CF_2 &vy)
#endif
{
  double dt = DBL_MAX;

  // get the min dx
  splitting_criteria_t* data = (splitting_criteria_t*)p4est->user_pointer;
  double dx = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN;

  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; n++)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
#endif
#ifdef P4_TO_P8
    double vn = sqrt(SQR(vx(x,y,z)) + SQR(vy(x,y,z)) + SQR(vz(x,y,z)));
#else
    double vn = sqrt(SQR(vx(x,y)) + SQR(vy(x,y)));
#endif

    dt = MIN(dt, dx/vn);
  }

  // reduce among processors
  double dt_min;
  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  return dt_min;
}

#ifdef P4_TO_P8
double my_p4est_semi_lagrangian_t::compute_dt(Vec vx, Vec vy, Vec vz)
#else
double my_p4est_semi_lagrangian_t::compute_dt(Vec vx, Vec vy)
#endif
{
  PetscErrorCode ierr;
  double dt = DBL_MAX;
  double *vx_p, *vy_p;

  ierr = VecGetArray(vx, &vx_p); CHKERRXX(ierr);
  ierr = VecGetArray(vy, &vy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *vz_p;
  ierr = VecGetArray(vz, &vz_p); CHKERRXX(ierr);
#endif

  // get the min dx
  splitting_criteria_t* data = (splitting_criteria_t*)p4est->user_pointer;
  double dx = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN;

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++){
#ifdef P4_TO_P8
    double vn = sqrt(SQR(vx_p[i]) + SQR(vy_p[i]) + SQR(vz_p[i]));
#else
    double vn = sqrt(SQR(vx_p[i]) + SQR(vy_p[i]));
#endif

    dt = MIN(dt, dx/vn);
  }

  ierr = VecRestoreArray(vx, &vx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(vy, &vy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(vz, &vz_p); CHKERRXX(ierr);
#endif

  // reduce among processors
  double dt_min;
  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  return dt_min;
}

void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt,
                                                      #ifdef P4_TO_P8
                                                      const CF_3 *v,
                                                      #else
                                                      const CF_2* v,
                                                      #endif
                                                      Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);

  for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);

    /* find the departure node via backtracing */
#ifdef P4_TO_P8
    double x_star = xyz[0] - 0.5*dt*v[0](xyz[0], xyz[1], xyz[2]);
    double y_star = xyz[0] - 0.5*dt*v[1](xyz[0], xyz[1], xyz[2]);
    double z_star = xyz[2] - 0.5*dt*v[2](xyz[0], xyz[1], xyz[2]);
#else
    double x_star = xyz[0] - 0.5*dt*v[0](xyz[0], xyz[1]);
    double y_star = xyz[0] - 0.5*dt*v[1](xyz[0], xyz[1]);
#endif

    double xyz_departure[] =
    {
  #ifdef P4_TO_P8
      xyz[0] - dt*v[0](x_star, y_star, z_star),
      xyz[1] - dt*v[1](x_star, y_star, z_star),
      xyz[2] - dt*v[2](x_star, y_star, z_star)
  #else
      xyz[0] - dt*v[0](x_star, y_star),
      xyz[1] - dt*v[1](x_star, y_star)
  #endif
    };

    /* Buffer the point for interpolation */
    interp.add_point(n, xyz_departure);
  }

#ifdef P4_TO_P8
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], phi_xx_n[2], quadratic_non_oscillatory);
#else
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], quadratic_non_oscillatory);
#endif
  interp.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt, Vec *v, Vec **vxx, Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);

  /* find vnp1 */
  std::vector<double> v_tmp[P4EST_DIM];
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_tmp[dir].resize(nodes_np1->indep_nodes.elem_count);
#ifdef P4_TO_P8
    interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic_non_oscillatory);
#else
    interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic_non_oscillatory);
#endif
    interp.interpolate(v_tmp[dir].data());
  }
  interp.clear();

  /* now find v_star */
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    /* Find initial xy points */
    double xyz_star[] =
    {
      node_x_fr_n(n, p4est_np1, nodes_np1) - 0.5*dt*v_tmp[0][n],
      node_y_fr_n(n, p4est_np1, nodes_np1) - 0.5*dt*v_tmp[1][n]
  #ifdef P4_TO_P8
      , node_z_fr_n(n, p4est_np1, nodes_np1) - 0.5*dt*v_tmp[2][n]
  #endif
    };

    interp.add_point(n, xyz_star);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
#ifdef P4_TO_P8
    interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic_non_oscillatory);
#else
    interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic_non_oscillatory);
#endif
    interp.interpolate(v_tmp[dir].data());
  }
  interp.clear();

  /* finally, find the backtracing value */
  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz_d[] =
    {
      node_x_fr_n(n, p4est_np1, nodes_np1) - dt*v_tmp[0][n],
      node_y_fr_n(n, p4est_np1, nodes_np1) - dt*v_tmp[1][n],
  #ifdef P4_TO_P8
      node_z_fr_n(n, p4est_np1, nodes_np1) - dt*v_tmp[2][n],
  #endif
    };

    interp.add_point(n, xyz_d);
  }

#ifdef P4_TO_P8
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], phi_xx_n[2], quadratic_non_oscillatory);
#else
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], quadratic_non_oscillatory);
#endif
  interp.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::advect_from_n_to_np1(double dt_nm1, double dt_n,
                                                      Vec *vnm1, Vec **vxx_nm1,
                                                      Vec *vn  , Vec **vxx_n,
                                                      Vec phi_n, Vec *phi_xx_n,
                                                      double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);

  my_p4est_interpolation_nodes_t interp(ngbd_n);

  std::vector<double> v_tmp_nm1[P4EST_DIM];
  std::vector<double> v_tmp_n  [P4EST_DIM];

  /* find the velocity field at time np1 */
  for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_tmp_n[dir].resize(nodes_np1->indep_nodes.elem_count);

#ifdef P4_TO_P8
    interp.set_input(vn[dir], vxx_n[dir][0], vxx_n[dir][1], vxx_n[dir][2], linear);
#else
    interp.set_input(vn[dir], vxx_n[dir][0], vxx_n[dir][1], linear);
#endif
    interp.interpolate(v_tmp_n[dir].data());
  }
  interp.clear();


  /* now find x_star */
  for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz_star[] =
    {
      node_x_fr_n(n, p4est_np1, nodes_np1) - .5*dt_n*v_tmp_n[0][n],
      node_y_fr_n(n, p4est_np1, nodes_np1) - .5*dt_n*v_tmp_n[1][n]
  #ifdef P4_TO_P8
      ,
      node_z_fr_n(ni, p4est_np1, nodes_np1) - .5*dt_n*v_tmp_n[2][ni]
  #endif
    };

    interp.add_point(n, xyz_star);
  }

  /* interpolate vnm1 */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    v_tmp_nm1[dir].resize(nodes_np1->indep_nodes.elem_count);

#ifdef P4_TO_P8
    interp.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2], linear);
#else
    interp.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], linear);
#endif
    interp.interpolate(v_tmp_nm1[dir].data());


#ifdef P4_TO_P8
    interp.set_input(vn[dir], vxx_n[dir][0], vxx_n[dir][1], vxx_n[dir][2], linear);
#else
    interp.set_input(vn[dir], vxx_n[dir][0], vxx_n[dir][1], linear);
#endif
    interp.interpolate(v_tmp_n[dir].data());
  }
  interp.clear();

  /* finally, find the backtracing value */
  /* find the departure node via backtracing */
  for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double vx_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp_n[0][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[0][n];
    double vy_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp_n[1][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[1][n];
#ifdef P4_TO_P8
    double vz_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp_n[2][ni] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[2][ni];
#endif

    double xyz_departure[] =
    {
      node_x_fr_n(n, p4est_np1, nodes_np1) - dt_n*vx_star,
      node_y_fr_n(n, p4est_np1, nodes_np1) - dt_n*vy_star
  #ifdef P4_TO_P8
      ,
      node_z_fr_n(ni, p4est_np1, nodes_np1) - dt_n*vz_star
  #endif
    };

    interp.add_point(n, xyz_departure);
  }

#ifdef P4_TO_P8
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], phi_xx_n[2], quadratic_non_oscillatory);
#else
  interp.set_input(phi_n, phi_xx_n[0], phi_xx_n[1], quadratic_non_oscillatory);
#endif
  interp.interpolate(phi_np1);

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_advect_from_n_to_np1_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::update_p4est(const CF_2 *v, double dt, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx[dir]); CHKERRXX(ierr);
    }

#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1], phi_xx[2]);
#else
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1]);
#endif
    local_derivatives = true;
  }

  p4est_t       *p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, phi, phi_xx, phi_np1_p, p4est_np1, nodes_np1);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est_np1->user_pointer = p4est->user_pointer;

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est);       p4est = *p_p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = *p_nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = *p_ghost = ghost_np1;
  hierarchy->update(p4est, ghost);
  ngbd_n->update(hierarchy, nodes);

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  if (local_derivatives)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_semi_lagrangian_t::update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vx_xx, vx_yy */
  Vec *vxx[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vxx[dir] = new Vec[P4EST_DIM];
    if(dir==0)
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
    else
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
      }
    }
#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2]);
#else
    ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1]);
#endif
  }

  /* compute phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDuplicate(vxx[0][dir], &phi_xx[dir]); CHKERRXX(ierr);
    }

#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1], phi_xx[2]);
#else
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1]);
#endif
    local_derivatives = true;
  }

  p4est_t       *p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
  bool is_grid_changing = true;

  int counter = 0;
  while (is_grid_changing) {
    ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);

    // advect from np1 to n to enable refinement
    double* phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    advect_from_n_to_np1(dt, v, vxx, phi, phi_xx,
                         phi_np1_p, p4est_np1, nodes_np1);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est_np1->user_pointer = p4est->user_pointer;

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est);       p4est = *p_p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = *p_nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = *p_ghost = ghost_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    for(int dd=0; dd<P4EST_DIM; ++dd)
    {
      ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
    }
    delete vxx[dir];
  }

  if (local_derivatives)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_1st_order, 0, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_semi_lagrangian_t::update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, Vec &phi, Vec *phi_xx)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_semi_lagrangian_update_p4est_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute vx_xx_nm1, vx_yy_nm1, ... */
  Vec *vxx_nm1[P4EST_DIM];
  Vec *vxx_n  [P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vxx_nm1[dir] = new Vec[P4EST_DIM];
    vxx_n  [dir] = new Vec[P4EST_DIM];
    if(dir==0)
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }
    }
    else
    {
      for(int dd=0; dd<P4EST_DIM; ++dd)
      {
        ierr = VecDuplicate(vxx_nm1[0][dd], &vxx_nm1[dir][dd]); CHKERRXX(ierr);
        ierr = VecDuplicate(vxx_n  [0][dd], &vxx_n  [dir][dd]); CHKERRXX(ierr);
      }
    }
#ifdef P4_TO_P8
      ngbd_n->second_derivatives_central(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2]);
      ngbd_n->second_derivatives_central(vn  [dir], vxx_n  [dir][0], vxx_n  [dir][1], vxx_n  [dir][2]);
#else
      ngbd_n->second_derivatives_central(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1]);
      ngbd_n->second_derivatives_central(vn  [dir], vxx_n  [dir][0], vxx_n  [dir][1]);
#endif
  }

  /* now for phi_xx and phi_yy */
  bool local_derivatives = false;
  if (phi_xx == NULL)
  {
    phi_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDuplicate(vxx_n[0][dir], &phi_xx[dir]); CHKERRXX(ierr);
    }

#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1], phi_xx[2]);
#else
    ngbd_n->second_derivatives_central(phi, phi_xx[0], phi_xx[1]);
#endif
    local_derivatives = true;
  }

  p4est_t       *p4est_np1 = my_p4est_copy(p4est, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
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
                         phi_np1_p, p4est_np1, nodes_np1);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1_p);

    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

    if (is_grid_changing) {
      my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_grid_gen_iter[counter], 0, 0, 0, 0); CHKERRXX(ierr);
    counter++;
  }

  p4est_np1->user_pointer = p4est->user_pointer;

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est);       p4est = *p_p4est = p4est_np1;
  p4est_nodes_destroy(nodes); nodes = *p_nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = *p_ghost = ghost_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    for(int dd=0; dd<P4EST_DIM; ++dd)
    {
      ierr = VecDestroy(vxx_nm1[dir][dd]); CHKERRXX(ierr);
      ierr = VecDestroy(vxx_n  [dir][dd]); CHKERRXX(ierr);
    }
    delete vxx_nm1[dir];
    delete vxx_n  [dir];
  }


  if(local_derivatives)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(phi_xx[dir]); CHKERRXX(ierr);
    }
    delete phi_xx;
  }

  ierr = PetscLogEventEnd(log_my_p4est_semi_lagrangian_update_p4est_2nd_order, 0, 0, 0, 0); CHKERRXX(ierr);
}
