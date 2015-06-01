#ifndef MY_P4EST_SEMI_LAGRANGIAN_H
#define MY_P4EST_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <iostream>

#ifdef P4_TO_P8
#include <p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <p4est.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#endif

class my_p4est_semi_lagrangian_t
{
  p4est_t **p_p4est, *p4est;
  p4est_nodes_t **p_nodes, *nodes;
  p4est_ghost_t **p_ghost, *ghost;
  my_p4est_brick_t *myb;
  my_p4est_node_neighbors_t *ngbd_n;
  my_p4est_hierarchy_t *hierarchy;

  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];

  void advect_from_n_to_np1(double dt,
                          #ifdef P4_TO_P8
                            const CF_3 **v,
                          #else
                            const CF_2 **v,
                          #endif
                            Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);

  void advect_from_n_to_np1(double dt, Vec *v, Vec **vxx, Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);


  void advect_from_n_to_np1(double dt_nm1, double dt_n,
                            Vec *vnm1, Vec **vxx_nm1,
                            Vec *vn  , Vec **vxx_n,
                            Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);

public:
  my_p4est_semi_lagrangian_t(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb, my_p4est_node_neighbors_t *ngbd);

#ifdef P4_TO_P8
  double compute_dt(const CF_3& vx, const CF_3& vy, const CF_3& vz);
  double compute_dt(Vec vx, Vec vy, Vec vz);
#else
  double compute_dt(const CF_2& vx, const CF_2& vy);
  double compute_dt(Vec vx, Vec vy);
#endif


  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field given as a continuous function. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   */
#ifdef P4_TO_P8
  void update_p4est(const CF_3 **v, double dt, Vec &phi, Vec *phi_xx=NULL);
#else
  void update_p4est(const CF_2 **v, double dt, Vec &phi, Vec *phi_xx=NULL);
#endif

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   */
  void update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx=NULL);

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with BDF along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param vnm1    the velocity field at time nm1 defined on p4est_n. This is a pointer to an array of dimension P4EST_DIM
   * \param vn      the velocity field at time n defined on p4est_n. This is a pointer to an array of dimension P4EST_DIM
   * \param dt_nm1  the time step from tnm1 to tn
   * \param dt_n    the time step from tn to tnp1
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   */
  void update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, Vec &phi, Vec *phi_xx=NULL);
};

#endif // MY_P4EST_SEMI_LAGRANGIAN_H
