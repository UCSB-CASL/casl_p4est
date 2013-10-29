#ifdef P4_TO_P8
#include "my_p8est_semi_lagrangian.h"
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include "my_p4est_semi_lagrangian.h"
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>
#include <sc_notify.h>
#include <mpi.h>

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
extern PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_Vec;
extern PetscLogEvent log_Semilagrangian_advect_from_n_to_np1_CF2;
extern PetscLogEvent log_Semilagrangian_update_p4est_second_order_Vec;
extern PetscLogEvent log_Semilagrangian_update_p4est_second_order_CF2;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

SemiLagrangian::SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb)
  : p_p4est_(p4est), p4est_(*p4est),
    p_nodes_(nodes), nodes_(*nodes),
    p_ghost_(ghost), ghost_(*ghost),
    myb_(myb)
{
  // compute domain sizes
  double *v2c = p4est_->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  for (short i=0; i<P4EST_DIM; i++)
    xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
  for (short i=0; i<P4EST_DIM; i++)
    xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
}

#ifdef P4_TO_P8
double SemiLagrangian::compute_dt(const CF_3 &vx, const CF_3 &vy, const CF_3 &vz)
#else
double SemiLagrangian::compute_dt(const CF_2 &vx, const CF_2 &vy)
#endif
{
  double dt = DBL_MAX;

  // loop over trees
  for (p4est_topidx_t tr_it = p4est_->first_local_tree; tr_it <=p4est_->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est_->trees, tr_it);
    p4est_topidx_t *t2v = p4est_->connectivity->tree_to_vertex;
    double *v2c = p4est_->connectivity->vertices;

    double tr_xmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 0];
    double tr_ymin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 1];
#ifdef P4_TO_P8
    double tr_zmin = v2c[3*t2v[P4EST_CHILDREN*tr_it + 0] + 2];
#endif

    // loop over quadrants
    for (size_t qu_it=0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      double dx = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
      double x  = quad_x_fr_i(quad) + 0.5*dx + tr_xmin;
      double y  = quad_y_fr_j(quad) + 0.5*dx + tr_ymin;
#ifdef P4_TO_P8
      double z  = quad_z_fr_k(quad) + 0.5*dx + tr_zmin;
#endif
#ifdef P4_TO_P8
      double vn = sqrt(SQR(vx(x,y,z)) + SQR(vy(x,y,z)) + SQR(vz(x,y,z)));
#else
      double vn = sqrt(SQR(vx(x,y)) + SQR(vy(x,y)));
#endif
      dt = MIN(dt, dx/vn);
    }
  }

  double dt_min;
  PetscErrorCode ierr;
  ierr = MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm); CHKERRXX(ierr);

  return dt_min;
}

void SemiLagrangian::advect_from_n_to_np1(my_p4est_node_neighbors_t &qnnn, double dt,
                                          #ifdef P4_TO_P8
                                          const CF_3& vx, const CF_3& vy, const CF_3& vz,
                                          Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                                          #else
                                          const CF_2& vx, const CF_2& vy,
                                          Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                                          #endif
                                          double *phi_np1,p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
  ierr = PetscLogEventBegin(log_Semilagrangian_advect_from_n_to_np1_CF2, phi_n, 0, 0, 0); CHKERRXX(ierr);

  InterpolatingFunction interp(p4est_, nodes_, ghost_, myb_, &qnnn);
#ifdef P4_TO_P8
  interp.set_input_parameters(phi_n, quadratic_non_oscillatory, phi_xx_n, phi_yy_n, phi_zz_n);
#else
  interp.set_input_parameters(phi_n, quadratic_non_oscillatory, phi_xx_n, phi_yy_n);
#endif

  p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree

  p4est_locidx_t ni_begin = 0;
  p4est_locidx_t ni_end   = nodes_np1->indep_nodes.elem_count;

  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
    double tr_zmin = t2c[3 * tr_mm + 2];
#endif

    /* Find initial xy points */
    double xyz[] =
    {
      node_x_fr_i(indep_node) + tr_xmin,
      node_y_fr_j(indep_node) + tr_ymin
  #ifdef P4_TO_P8
      ,
      node_z_fr_k(indep_node) + tr_zmin
  #endif
    };

    /* find the departure node via backtracing */

#ifdef P4_TO_P8
    double x_star = xyz[0] - .5*dt*vx(xyz[0], xyz[1], xyz[2]);
    double y_star = xyz[1] - .5*dt*vy(xyz[0], xyz[1], xyz[2]);
    double z_star = xyz[2] - .5*dt*vz(xyz[0], xyz[1], xyz[2]);
#else
    double x_star = xyz[0] - .5*dt*vx(xyz[0], xyz[1]);
    double y_star = xyz[1] - .5*dt*vy(xyz[0], xyz[1]);
#endif

    double xyz_departure[] =
    {
  #ifdef P4_TO_P8
      xyz[0] - dt*vx(x_star, y_star, z_star),
      xyz[1] - dt*vy(x_star, y_star, z_star),
      xyz[2] - dt*vz(x_star, y_star, z_star)
  #else
      xyz[0] - dt*vx(x_star, y_star),
      xyz[1] - dt*vy(x_star, y_star)
  #endif
    };

    /* Buffer the point for interpolation */
    interp.add_point_to_buffer(ni, xyz_departure);
  }

  /* interpolate from old vector into our output vector */
  interp.interpolate(phi_np1);

  ierr = PetscLogFlops(20); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_Semilagrangian_advect_from_n_to_np1_CF2, phi_n, 0, 0, 0); CHKERRXX(ierr);
}

void SemiLagrangian::advect_from_n_to_np1(my_p4est_node_neighbors_t &qnnn, double dt,
                                          #ifdef P4_TO_P8
                                          Vec vx, Vec vx_xx, Vec vx_yy, Vec vx_zz,
                                          Vec vy, Vec vy_xx, Vec vy_yy, Vec vy_zz,
                                          Vec vz, Vec vz_xx, Vec vz_yy, Vec vz_zz,
                                          Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                                          #else
                                          Vec vx, Vec vx_xx, Vec vx_yy,
                                          Vec vy, Vec vy_xx, Vec vy_yy,
                                          Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                                          #endif
                                          double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1)
{
  ierr = PetscLogEventBegin(log_Semilagrangian_advect_from_n_to_np1_Vec, 0, 0, 0, 0); CHKERRXX(ierr);

  p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
  double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree

  p4est_locidx_t ni_begin = 0;
  p4est_locidx_t ni_end   = nodes_np1->indep_nodes.elem_count;

  /* first find the velocities at the nodes */
  InterpolatingFunction interp_vel(p4est_, nodes_, ghost_, myb_, &qnnn);

  std::vector<double> vx_tmp(nodes_np1->indep_nodes.elem_count);
  std::vector<double> vy_tmp(nodes_np1->indep_nodes.elem_count);
#ifdef P4_TO_P8
  std::vector<double> vz_tmp(nodes_np1->indep_nodes.elem_count);
#endif


  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
    double tr_zmin = t2c[3 * tr_mm + 2];
#endif

    /* Find initial xy points */
    double xyz[] =
    {
      node_x_fr_i(indep_node) + tr_xmin,
      node_y_fr_j(indep_node) + tr_ymin
  #ifdef P4_TO_P8
      ,
      node_z_fr_k(indep_node) + tr_zmin
  #endif
    };

    interp_vel.add_point_to_buffer(ni, xyz);
  }
#ifdef P4_TO_P8
  interp_vel.set_input_parameters(vx, quadratic, vx_xx, vx_yy, vx_zz);
#else
  interp_vel.set_input_parameters(vx, quadratic, vx_xx, vx_yy);
#endif
  interp_vel.interpolate(vx_tmp.data());

#ifdef P4_TO_P8
  interp_vel.set_input_parameters(vy, quadratic, vy_xx, vy_yy, vy_zz);
#else
  interp_vel.set_input_parameters(vy, quadratic, vy_xx, vy_yy);
#endif
  interp_vel.interpolate(vy_tmp.data());

#ifdef P4_TO_P8
  interp_vel.set_input_parameters(vz, quadratic, vz_xx, vz_yy, vz_zz);
  interp_vel.interpolate(vz_tmp.data());
#endif

  /* now find v_star */
  InterpolatingFunction interp_vel_star(p4est_, nodes_, ghost_, myb_, &qnnn);

  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
    double tr_zmin = t2c[3 * tr_mm + 2];
#endif

    /* Find initial xy points */
    double xyz_star[] =
    {
      node_x_fr_i(indep_node) + tr_xmin - .5*dt*vx_tmp[ni],
      node_y_fr_j(indep_node) + tr_ymin - .5*dt*vy_tmp[ni]
  #ifdef P4_TO_P8
      ,
      node_z_fr_k(indep_node) + tr_zmin - .5*dt*vz_tmp[ni]
  #endif
    };

    interp_vel_star.add_point_to_buffer(ni, xyz_star);
  }

#ifdef P4_TO_P8
  interp_vel_star.set_input_parameters(vx, quadratic, vx_xx, vx_yy, vx_zz);
#else
  interp_vel_star.set_input_parameters(vx, quadratic, vx_xx, vx_yy);
#endif
  interp_vel_star.interpolate(vx_tmp.data());

#ifdef P4_TO_P8
  interp_vel_star.set_input_parameters(vy, quadratic, vy_xx, vy_yy, vy_zz);
#else
  interp_vel_star.set_input_parameters(vy, quadratic, vy_xx, vy_yy);
#endif
  interp_vel_star.interpolate(vy_tmp.data());

#ifdef P4_TO_P8
  interp_vel_star.set_input_parameters(vz, quadratic, vz_xx, vz_yy, vz_zz);
  interp_vel_star.interpolate(vz_tmp.data());
#endif

  /* finally, find the backtracing value */
  InterpolatingFunction interp(p4est_, nodes_, ghost_, myb_, &qnnn);

  /* find the departure node via backtracing */
  for (p4est_locidx_t ni = ni_begin; ni < ni_end; ++ni){ //Loop through all nodes of a single processor
    p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, ni);
    p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

    p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
    double tr_xmin = t2c[3 * tr_mm + 0];
    double tr_ymin = t2c[3 * tr_mm + 1];
#ifdef P4_TO_P8
    double tr_zmin = t2c[3 * tr_mm + 2];
#endif


    /* Find initial xy points */
    double xyz_departure[] =
    {
      node_x_fr_i(indep_node) + tr_xmin - dt*vx_tmp[ni],
      node_y_fr_j(indep_node) + tr_ymin - dt*vy_tmp[ni]
  #ifdef P4_TO_P8
      ,
      node_z_fr_k(indep_node) + tr_zmin - dt*vz_tmp[ni]
  #endif
    };

    /* Buffer the point for interpolation */
    interp.add_point_to_buffer(ni, xyz_departure);
  }
#ifdef P4_TO_P8
  interp.set_input_parameters(phi_n, quadratic_non_oscillatory, phi_xx_n, phi_yy_n, phi_zz_n);
#else
  interp.set_input_parameters(phi_n, quadratic_non_oscillatory, phi_xx_n, phi_yy_n);
#endif

  /* interpolate from old vector into our output vector */
  interp.interpolate(phi_np1);

  ierr = PetscLogFlops(40); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_Semilagrangian_advect_from_n_to_np1_Vec, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void SemiLagrangian::update_p4est_second_order(Vec vx, Vec vy, Vec vz, double dt, Vec &phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void SemiLagrangian::update_p4est_second_order(Vec vx, Vec vy, double dt, Vec &phi, Vec phi_xx, Vec phi_yy)
#endif
{
  ierr = PetscLogEventBegin(log_Semilagrangian_update_p4est_second_order_Vec, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscErrorCode ierr;
  p4est_t *p4est_np1 = my_p4est_new(p4est_->mpicomm, p4est_->connectivity, 0, NULL, NULL);

  /* create hierarchy structure on p4est_n if you want to do quadratic interpolation */
  // TODO: We should think about a smart way to move this outside class
  my_p4est_hierarchy_t hierarchy(p4est_, ghost_, myb_);
  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);
  std::vector<double> phi_tmp;

  // compute vx_xx, vx_yy
  Vec vx_xx, vx_yy;  
  ierr = VecCreateGhost(p4est_, nodes_, &vx_xx); CHKERRXX(ierr);
  ierr = VecCreateGhost(p4est_, nodes_, &vx_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec vx_zz;
  ierr = VecCreateGhost(p4est_, nodes_, &vx_zz); CHKERRXX(ierr);
#endif
#ifdef P4_TO_P8
  qnnn.second_derivatives_central(vx, vx_xx, vx_yy, vx_zz);
#else
  qnnn.second_derivatives_central(vx, vx_xx, vx_yy);
#endif

  // note that vy_xx and vy_yy must duplicate from separate vectors otherwise PETSc throws
  Vec vy_xx, vy_yy;
  ierr = VecDuplicate(vx_xx, &vy_xx); CHKERRXX(ierr);
  ierr = VecDuplicate(vx_yy, &vy_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec vy_zz;
  ierr = VecDuplicate(vx_zz, &vy_zz); CHKERRXX(ierr);
#endif
#ifdef P4_TO_P8
  qnnn.second_derivatives_central(vy, vy_xx, vy_yy, vy_zz);
#else
  qnnn.second_derivatives_central(vy, vy_xx, vy_yy);
#endif

#ifdef P4_TO_P8
  // finally compute vz_xx, vz_yy, vz_zz if in 3D
  Vec vz_xx, vz_yy, vz_zz;
  ierr = VecDuplicate(vx_xx, &vz_xx); CHKERRXX(ierr);
  ierr = VecDuplicate(vx_yy, &vz_yy); CHKERRXX(ierr);
  ierr = VecDuplicate(vx_zz, &vz_zz); CHKERRXX(ierr);

  qnnn.second_derivatives_central(vz, vz_xx, vz_yy, vz_zz);
#endif


  // now for phi_xx and phi_yy
  Vec phi_xx_ = phi_xx, phi_yy_ = phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz_ = phi_zz;
#endif
  bool local_derivatives = false;
#ifdef P4_TO_P8
  if (phi_xx_ == NULL && phi_yy_ == NULL && phi_zz_ == NULL)
#else
  if (phi_xx_ == NULL && phi_yy_ == NULL)
#endif
  {
    ierr = VecDuplicate(vx_xx, &phi_xx_); CHKERRXX(ierr);
    ierr = VecDuplicate(vx_yy, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDuplicate(vx_zz, &phi_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    qnnn.second_derivatives_central(phi, phi_xx_, phi_yy_, phi_zz_);
#else
    qnnn.second_derivatives_central(phi, phi_xx_, phi_yy_);
#endif
    local_derivatives = true;
  }

  int nb_iter = ((splitting_criteria_t*) (p4est_->user_pointer))->max_lvl;

  for( int iter = 0; iter < nb_iter; ++iter )
  {
    p4est_t       *p4est_tmp = p4est_copy(p4est_np1, P4EST_FALSE);
    p4est_nodes_t *nodes_tmp = my_p4est_nodes_new(p4est_tmp,NULL);

    /* compute phi_np1 on intermediate grid */
    phi_tmp.resize(nodes_tmp->indep_nodes.elem_count);

    advect_from_n_to_np1(qnnn, dt,
                     #ifdef P4_TO_P8
                         vx,  vx_xx,   vx_yy,   vx_zz,
                         vy,  vy_xx,   vy_yy,   vy_zz,
                         vz,  vz_xx,   vz_yy,   vz_zz,
                         phi, phi_xx_, phi_yy_, phi_zz_,
                     #else
                         vx,  vx_xx,   vx_yy,
                         vy,  vy_xx,   vy_yy,
                         phi, phi_xx_, phi_yy_,
                     #endif
                         phi_tmp.data(), p4est_tmp, nodes_tmp);

    /* refine p4est_np1 */
    splitting_criteria_t *data = (splitting_criteria_t*) p4est_->user_pointer;
    splitting_criteria_update_t data_np1(1.2, data->min_lvl, data->max_lvl, &phi_tmp, myb_, p4est_tmp, NULL, nodes_tmp);
    p4est_np1->user_pointer = (void*) &data_np1;
    my_p4est_refine (p4est_np1, P4EST_FALSE, refine_criteria_sl , NULL);
    my_p4est_partition(p4est_np1, NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_destroy(p4est_tmp);
  }

  /* restore the user pointer in the p4est */
  p4est_np1->user_pointer = p4est_->user_pointer;

  /* compute new ghost layer */
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);

  /* compute the nodes structure */
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  /* update the values of phi at the new time-step */
  phi_tmp.clear();

  Vec phi_np1;
  double *phi_np1_p;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  advect_from_n_to_np1(qnnn, dt,
                     #ifdef P4_TO_P8
                         vx,  vx_xx,   vx_yy,   vx_zz,
                         vy,  vy_xx,   vy_yy,   vy_zz,
                         vz,  vz_xx,   vz_yy,   vz_zz,
                         phi, phi_xx_, phi_yy_, phi_zz_,
                     #else
                         vx,  vx_xx,   vx_yy,
                         vy,  vy_xx,   vy_yy,
                         phi, phi_xx_, phi_yy_,
                     #endif
                       phi_np1_p, p4est_np1, nodes_np1);
  ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  ierr = VecDestroy(vx_xx); CHKERRXX(ierr);
  ierr = VecDestroy(vx_yy); CHKERRXX(ierr);  
  ierr = VecDestroy(vy_xx); CHKERRXX(ierr);
  ierr = VecDestroy(vy_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(vx_zz); CHKERRXX(ierr);
  ierr = VecDestroy(vy_zz); CHKERRXX(ierr);
  ierr = VecDestroy(vz_xx); CHKERRXX(ierr);
  ierr = VecDestroy(vz_yy); CHKERRXX(ierr);
  ierr = VecDestroy(vz_zz); CHKERRXX(ierr);
#endif
  if (local_derivatives)
  {
    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
#endif
  }

  ierr = PetscLogEventEnd(log_Semilagrangian_update_p4est_second_order_Vec, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void SemiLagrangian::update_p4est_second_order(const CF_3& vx, const CF_3& vy, const CF_3& vz, double dt, Vec &phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void SemiLagrangian::update_p4est_second_order(const CF_2& vx, const CF_2& vy, double dt, Vec &phi, Vec phi_xx, Vec phi_yy)
#endif
{  
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_Semilagrangian_update_p4est_second_order_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
  p4est_t *p4est_np1 = my_p4est_new(p4est_->mpicomm, p4est_->connectivity, 0, NULL, NULL);

  /* create hierarchy structure on p4est_n if you want to do quadratic interpolation */
  my_p4est_hierarchy_t hierarchy(p4est_, ghost_, myb_);
  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);
  std::vector<double> phi_tmp;

  // compute phi_xx and phi_yy
  Vec phi_xx_ = phi_xx, phi_yy_ = phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz_ = phi_zz;
#endif
  bool local_derivatives = false;

#ifdef P4_TO_P8
  if (phi_xx_ == NULL && phi_yy_ == NULL && phi_zz_ == NULL)
#else
  if (phi_xx_ == NULL && phi_yy_ == NULL)
#endif
  {
    ierr = VecCreateGhost(p4est_, nodes_, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhost(p4est_, nodes_, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhost(p4est_, nodes_, &phi_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    qnnn.second_derivatives_central(phi, phi_xx_, phi_yy_, phi_zz_);
#else
    qnnn.second_derivatives_central(phi, phi_xx_, phi_yy_);
#endif

    local_derivatives = true;
  }

  int nb_iter = ((splitting_criteria_t*) (p4est_->user_pointer))->max_lvl;

  for( int iter = 0; iter < nb_iter; ++iter )
  {
    p4est_t       *p4est_tmp = p4est_copy(p4est_np1, P4EST_FALSE);
    p4est_nodes_t *nodes_tmp = my_p4est_nodes_new(p4est_tmp,NULL);

    /* compute phi_np1 on intermediate grid */
    phi_tmp.resize(nodes_tmp->indep_nodes.elem_count);
    advect_from_n_to_np1(qnnn, dt,
                     #ifdef P4_TO_P8
                         vx, vy, vz,
                         phi, phi_xx_, phi_yy_, phi_zz_,
                     #else
                         vx,  vy,
                         phi, phi_xx_, phi_yy_,
                     #endif
                         phi_tmp.data(), p4est_tmp, nodes_tmp);

    /* refine p4est_np1 */
    splitting_criteria_t *data = (splitting_criteria_t*) p4est_->user_pointer;
    splitting_criteria_update_t data_np1(1.2, data->min_lvl, data->max_lvl, &phi_tmp, myb_, p4est_tmp, NULL, nodes_tmp);
    p4est_np1->user_pointer = (void*) &data_np1;
    my_p4est_refine (p4est_np1, P4EST_FALSE, refine_criteria_sl , NULL);
    my_p4est_partition(p4est_np1, NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_destroy(p4est_tmp);
  }

  /* restore the user pointer in the p4est */
  p4est_np1->user_pointer = p4est_->user_pointer;

  /* compute new ghost layer */
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);

  /* compute the nodes structure */
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  /* update the values of phi at the new time-step */
  phi_tmp.clear();

  Vec phi_np1;
  double *phi_np1_p;
  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  advect_from_n_to_np1(qnnn, dt,
                     #ifdef P4_TO_P8
                         vx, vy, vz,
                         phi, phi_xx_, phi_yy_, phi_zz_,
                     #else
                         vx,  vy,
                         phi, phi_xx_, phi_yy_,
                     #endif
                       phi_np1_p, p4est_np1, nodes_np1);
  ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);

  /* now that everything is updated, get rid of old stuff and swap them with new ones */
  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  if (local_derivatives)
  {
    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_Semilagrangian_update_p4est_second_order_CF2, 0, 0, 0, 0); CHKERRXX(ierr);
}

//double SemiLagrangian::update_p4est_intermediate_trees_with_ghost(const CF_2& vx, const CF_2& vy, Vec &phi)
//{

//  PetscErrorCode ierr;
//  p4est *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
//  throw std::invalid_argument("update_p4est_intermediate_trees_with_ghost : This function cannot work as long as a p4est bug has not been fixed, the coarsen case with quadrants accross processors.");

//  /* create hierarchy structure on p4est_n if you want to do quadratic interpolation */
//  my_p4est_hierarchy_t hierarchy(p4est_, ghost_, myb_);
//  my_p4est_node_neighbors_t qnnn(&hierarchy, nodes_);
//  std::vector<double> phi_tmp;

//  double dt = 1;

//  int nb_iter = ((splitting_criteria_t*) (p4est_->user_pointer))->max_lvl - ((splitting_criteria_t*) (p4est_->user_pointer))->min_lvl;

//  for( int iter = 0; iter < nb_iter; ++iter )
//  {
//    p4est_t       *p4est_tmp = p4est_copy(p4est_np1, P4EST_FALSE);
//    p4est_ghost_t *ghost_tmp = p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
//    p4est_nodes_t *nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);

//    /* compute phi_np1 on intermediate grid */
//    phi_tmp.resize(nodes_tmp->indep_nodes.elem_count);
//    advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp.data(), p4est_tmp, nodes_tmp, qnnn);

//    /* refine / coarsen p4est_np1 */
//    splitting_criteria_t *data = (splitting_criteria_t*) p4est_->user_pointer;
//    splitting_criteria_update_t data_np1(1.2, data->min_lvl, data->max_lvl, &phi_tmp, myb_, p4est_tmp, ghost_tmp, nodes_tmp);
//    p4est_np1->user_pointer = (void*) &data_np1;

//    p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_criteria_with_ghost_sl, NULL);
//    throw std::invalid_argument("");
//    p4est_refine (p4est_np1, P4EST_FALSE, refine_criteria_with_ghost_sl , NULL);

//    p4est_nodes_destroy(nodes_tmp);
//    p4est_destroy(p4est_tmp);
//  }

//  p4est_partition(p4est_np1, NULL);

//  /* restore the user pointer in the p4est */
//  p4est_np1->user_pointer = p4est_->user_pointer;

//  /* compute new ghost layer */
//  p4est_ghost_t *ghost_np1 = p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);

//  /* compute the nodes structure */
//  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

//  /* update the values of phi at the new time-step */
//  phi_tmp.resize(nodes_np1->indep_nodes.elem_count);
//  advect_from_n_to_np1(vx, vy, dt, phi, phi_tmp.data(), p4est_np1, nodes_np1, qnnn);

//  /* now that everything is updated, get rid of old stuff and swap them with new ones */
//  p4est_destroy(p4est_); p4est_ = *p_p4est_ = p4est_np1;
//  p4est_nodes_destroy(nodes_); nodes_ = *p_nodes_ = nodes_np1;
//  p4est_ghost_destroy(ghost_); ghost_ = *p_ghost_ = ghost_np1;

//  Vec phi_np1;
//  ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
//  double *f;
//  ierr = VecGetArray(phi_np1, &f); CHKERRXX(ierr);
//  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
//    f[n] = phi_tmp[n];
//  ierr = VecRestoreArray(phi_np1, &f); CHKERRXX(ierr);

//  ierr = VecDestroy(phi); CHKERRXX(ierr);
//  phi = phi_np1;
//  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  return dt;

//  throw std::invalid_argument("[ERROR]: This method is not implemented");
//  return 0;
//}
